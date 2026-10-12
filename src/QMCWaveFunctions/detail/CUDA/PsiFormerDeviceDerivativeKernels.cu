//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceDerivativeKernels.cu
 * @brief Deterministic CUDA/HIP leaves for the ordinary real-FP64 score reverse.
 */

#include "PsiFormerDeviceDerivativeKernels.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceDerivativeMath.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer::device
{
namespace
{

constexpr unsigned int derivative_block_size = 128;

std::size_t checkedAdd(std::size_t left, std::size_t right, const char* description)
{
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::length_error(description);
  return left + right;
}

std::size_t checkedProduct(std::size_t left, std::size_t right, const char* description)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(description);
  return left * right;
}

std::size_t checkedSpan(std::size_t count,
                        std::size_t stride,
                        std::size_t width,
                        const char* description)
{
  if (count == 0)
    return 0;
  return checkedAdd(checkedProduct(count - 1, stride, description), width, description);
}

unsigned int blockCount(std::size_t work, const char* description)
{
  const std::size_t blocks = work / derivative_block_size +
      (work % derivative_block_size == 0 ? 0 : 1);
  if (blocks > std::numeric_limits<unsigned int>::max())
    throw std::length_error(description);
  return static_cast<unsigned int>(blocks);
}

Error peekAtLastError()
{
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

void validateStridedInterval(std::size_t count,
                             std::size_t offset,
                             std::size_t stride,
                             const char* description)
{
  if (count == 0)
    return;
  if (stride == 0)
    throw std::invalid_argument(description);
  (void)checkedAdd(offset, checkedProduct(count - 1, stride, description), description);
}

std::size_t checkedPieceCount(std::size_t count,
                              std::size_t elements_per_piece,
                              std::size_t capacity,
                              const char* description)
{
  if (count == 0)
    return 0;
  if (elements_per_piece == 0)
    throw std::invalid_argument(description);
  const std::size_t pieces = count / elements_per_piece +
      (count % elements_per_piece == 0 ? 0 : 1);
  if (pieces > capacity)
    throw std::length_error(description);
  if (pieces > std::numeric_limits<unsigned int>::max())
    throw std::length_error(description);
  return pieces;
}

void validateOrbitalLayout(const OrbitalProductReverseLayout& layout)
{
  if (layout.configuration_count == 0 || layout.determinant_count == 0 ||
      layout.electron_count == 0)
    throw std::invalid_argument("PsiFormer orbital reverse dimensions must be positive");
  if (layout.row_stride < layout.electron_count)
    throw std::invalid_argument("PsiFormer orbital reverse row stride is too small");
  const std::size_t matrix_span = checkedSpan(
      layout.electron_count, layout.row_stride, layout.electron_count,
      "PsiFormer orbital reverse matrix extent overflow");
  if (layout.determinant_stride < matrix_span)
    throw std::invalid_argument("PsiFormer orbital reverse determinant stride is too small");
  const std::size_t determinant_span = checkedSpan(
      layout.determinant_count, layout.determinant_stride, matrix_span,
      "PsiFormer orbital reverse determinant extent overflow");
  if (layout.configuration_stride < determinant_span)
    throw std::invalid_argument("PsiFormer orbital reverse configuration stride is too small");
  (void)checkedSpan(layout.configuration_count, layout.configuration_stride,
                    determinant_span,
                    "PsiFormer orbital reverse batch extent overflow");
}

std::size_t validateEnvelopeLayout(const EnvelopeParameterReverseLayout& layout)
{
  if (layout.configuration_count != 1)
    throw std::invalid_argument(
        "PsiFormer envelope reverse requires one outer-serial configuration");
  if (layout.determinant_count == 0 || layout.electron_count == 0 ||
      layout.nucleus_count == 0 ||
      layout.electron_begin >= layout.electron_end ||
      layout.electron_end > layout.electron_count)
    throw std::invalid_argument("PsiFormer envelope reverse dimensions are invalid");
  if (layout.matrix_row_stride < layout.electron_count ||
      layout.distance_electron_stride < layout.nucleus_count ||
      layout.parameter_orbital_stride < layout.nucleus_count)
    throw std::invalid_argument("PsiFormer envelope reverse row stride is too small");

  const std::size_t matrix_span = checkedSpan(
      layout.electron_count, layout.matrix_row_stride, layout.electron_count,
      "PsiFormer envelope reverse matrix extent overflow");
  if (layout.matrix_determinant_stride < matrix_span)
    throw std::invalid_argument("PsiFormer envelope reverse determinant stride is too small");
  const std::size_t matrices_span = checkedSpan(
      layout.determinant_count, layout.matrix_determinant_stride, matrix_span,
      "PsiFormer envelope reverse determinant extent overflow");
  if (layout.matrix_configuration_stride < matrices_span)
    throw std::invalid_argument("PsiFormer envelope reverse matrix configuration stride is too small");
  (void)checkedSpan(layout.configuration_count, layout.matrix_configuration_stride,
                    matrices_span, "PsiFormer envelope reverse matrix batch extent overflow");

  const std::size_t distance_span = checkedSpan(
      layout.electron_count, layout.distance_electron_stride, layout.nucleus_count,
      "PsiFormer envelope reverse distance extent overflow");
  if (layout.distance_configuration_stride < distance_span)
    throw std::invalid_argument("PsiFormer envelope reverse distance configuration stride is too small");
  (void)checkedSpan(layout.configuration_count, layout.distance_configuration_stride,
                    distance_span, "PsiFormer envelope reverse distance batch extent overflow");

  const std::size_t parameter_span = checkedSpan(
      layout.electron_count, layout.parameter_orbital_stride, layout.nucleus_count,
      "PsiFormer envelope reverse parameter orbital extent overflow");
  if (layout.parameter_determinant_stride < parameter_span)
    throw std::invalid_argument("PsiFormer envelope reverse parameter determinant stride is too small");
  const std::size_t all_parameter_span = checkedSpan(
      layout.determinant_count, layout.parameter_determinant_stride, parameter_span,
      "PsiFormer envelope reverse parameter extent overflow");
  const std::size_t pi_end = checkedAdd(layout.pi_response_offset, all_parameter_span,
                                        "PsiFormer envelope pi response extent overflow");
  const std::size_t zeta_end = checkedAdd(layout.zeta_response_offset, all_parameter_span,
                                          "PsiFormer envelope zeta response extent overflow");
  if (!(pi_end <= layout.zeta_response_offset || zeta_end <= layout.pi_response_offset))
    throw std::invalid_argument("PsiFormer envelope pi and zeta response regions overlap");
  return all_parameter_span;
}

__global__ void derivativeZeroKernel(DerivativeVectorLayout layout, double* destination)
{
  const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < layout.element_count)
    destination[layout.destination_offset + index * layout.destination_stride] = 0.0;
}

__global__ void derivativeCopyKernel(DerivativeVectorLayout layout,
                                     const double* source,
                                     double* destination)
{
  const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < layout.element_count)
    destination[layout.destination_offset + index * layout.destination_stride] =
        source[layout.source_offset + index * layout.source_stride];
}

__global__ void realToComplexAxpyKernel(RealToComplexAxpyLayout layout,
                                        double alpha_real,
                                        double alpha_imag,
                                        const double* source,
                                        double* accumulator)
{
  const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= layout.element_count)
    return;
  const double value = source[layout.source_offset + index * layout.source_stride];
  double* output = accumulator + layout.accumulator_offset + index * layout.accumulator_stride;
  output[0] += alpha_real * value;
  output[1] += alpha_imag * value;
}

__global__ void responseDotDirectionKernel(ResponseDotLayout layout,
                                           const double* response,
                                           const double* direction,
                                           double* partials,
                                           DerivativeFiniteStatus* status)
{
  const std::size_t partial = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t partial_count = layout.element_count / layout.elements_per_partial +
      (layout.element_count % layout.elements_per_partial == 0 ? 0 : 1);
  if (partial >= partial_count)
    return;
  const std::size_t begin = partial * layout.elements_per_partial;
  const std::size_t remaining = layout.element_count - begin;
  const std::size_t local_count = remaining < layout.elements_per_partial
      ? remaining
      : layout.elements_per_partial;
  const std::size_t end = begin + local_count;
  double sum = 0.0;
  std::size_t first = layout.element_count;
  std::size_t nonfinite_count = 0;
  for (std::size_t index = begin; index < end; ++index)
  {
    const double left = response[layout.response_offset + index * layout.response_stride];
    const double right = direction[layout.direction_offset + index * layout.direction_stride];
    const double product = left * right;
    if (!device_math::isFiniteBinary64(left) ||
        !device_math::isFiniteBinary64(right) ||
        !device_math::isFiniteBinary64(product))
    {
      if (nonfinite_count == 0)
        first = index;
      ++nonfinite_count;
    }
    else
      sum += product;
  }
  if (!device_math::isFiniteBinary64(sum))
  {
    if (nonfinite_count == 0)
      first = begin;
    ++nonfinite_count;
  }
  partials[partial] = nonfinite_count == 0 ? sum : 0.0;
  status[partial]   = {first, nonfinite_count};
}

__global__ void finalizeResponseDotDirectionKernel(
    std::size_t partial_count,
    std::size_t logical_element_count,
    const double* partials,
    const DerivativeFiniteStatus* partial_status,
    double* result,
    DerivativeFiniteStatus* aggregate_status)
{
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  double sum = 0.0;
  std::size_t first = logical_element_count;
  std::size_t nonfinite_count = 0;
  for (std::size_t partial = 0; partial < partial_count; ++partial)
  {
    if (partial_status[partial].nonfinite_count != 0)
    {
      if (first == logical_element_count)
        first = partial_status[partial].first_nonfinite;
      nonfinite_count += partial_status[partial].nonfinite_count;
    }
    else
      sum += partials[partial];
  }
  if (!device_math::isFiniteBinary64(sum))
  {
    if (first == logical_element_count)
      first = 0;
    ++nonfinite_count;
  }
  *result = nonfinite_count == 0 ? sum : 0.0;
  *aggregate_status = {first, nonfinite_count};
}

__global__ void tanhOutputResidualReverseKernel(BatchedValueLayout layout,
                                                const double* tanh_output,
                                                const double* output_adjoint,
                                                double* tanh_input_adjoint,
                                                double* residual_adjoint)
{
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = layout.configuration_count * layout.rows * layout.width;
  if (logical >= count)
    return;
  const std::size_t feature = logical % layout.width;
  const std::size_t row_index = logical / layout.width;
  const std::size_t row = row_index % layout.rows;
  const std::size_t configuration = row_index / layout.rows;
  const std::size_t offset = configuration * layout.configuration_stride +
      row * layout.row_stride + feature;
  const double upstream = output_adjoint[offset];
  residual_adjoint[offset] = upstream;
  tanh_input_adjoint[offset] = device_derivative::tanhAdjointFromOutput(
      tanh_output[offset], upstream);
}

__global__ void attentionContextAttentionReverseKernel(
    BatchedAttentionForwardLayout layout,
    const double* value,
    const double* context_adjoint,
    double* attention_adjoint)
{
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t rows = layout.attention.rows;
  const std::size_t per_configuration = layout.attention.heads * rows * rows;
  const std::size_t count = layout.configuration_count * per_configuration;
  if (logical >= count)
    return;
  const std::size_t local = logical % per_configuration;
  const std::size_t configuration = logical / per_configuration;
  const std::size_t key_index = local % rows;
  const std::size_t query_index = (local / rows) % rows;
  const std::size_t head = local / (rows * rows);
  double sum = 0.0;
  for (std::size_t feature = 0; feature < layout.attention.head_width; ++feature)
  {
    const std::size_t context_offset = configuration * layout.feature_configuration_stride +
        query_index * layout.attention.feature_row_stride +
        head * layout.attention.head_width + feature;
    const std::size_t value_offset = configuration * layout.feature_configuration_stride +
        key_index * layout.attention.feature_row_stride +
        head * layout.attention.head_width + feature;
    sum += device_derivative::attentionContextReverseContributions(
        0.0, value[value_offset], context_adjoint[context_offset]).attention_adjoint;
  }
  const std::size_t output_offset = configuration * layout.attention_configuration_stride +
      head * layout.attention.attention_head_stride +
      query_index * layout.attention.attention_row_stride + key_index;
  attention_adjoint[output_offset] = sum;
}

__global__ void attentionContextValueReverseKernel(
    BatchedAttentionForwardLayout layout,
    const double* attention,
    const double* context_adjoint,
    double* value_adjoint)
{
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t width = layout.attention.heads * layout.attention.head_width;
  const std::size_t per_configuration = layout.attention.rows * width;
  const std::size_t count = layout.configuration_count * per_configuration;
  if (logical >= count)
    return;
  const std::size_t local = logical % per_configuration;
  const std::size_t configuration = logical / per_configuration;
  const std::size_t feature_in_width = local % width;
  const std::size_t key_index = local / width;
  const std::size_t head = feature_in_width / layout.attention.head_width;
  double sum = 0.0;
  for (std::size_t query_index = 0; query_index < layout.attention.rows; ++query_index)
  {
    const std::size_t attention_offset = configuration * layout.attention_configuration_stride +
        head * layout.attention.attention_head_stride +
        query_index * layout.attention.attention_row_stride + key_index;
    const std::size_t context_offset = configuration * layout.feature_configuration_stride +
        query_index * layout.attention.feature_row_stride + feature_in_width;
    sum += device_derivative::attentionContextReverseContributions(
        attention[attention_offset], 0.0, context_adjoint[context_offset]).value_adjoint;
  }
  const std::size_t output_offset = configuration * layout.feature_configuration_stride +
      key_index * layout.attention.feature_row_stride + feature_in_width;
  value_adjoint[output_offset] = sum;
}

__device__ double softmaxRowReduction(BatchedAttentionForwardLayout layout,
                                      const double* probabilities,
                                      const double* probability_adjoint,
                                      std::size_t configuration,
                                      std::size_t head,
                                      std::size_t query_index)
{
  double reduction = 0.0;
  const std::size_t row_offset = configuration * layout.attention_configuration_stride +
      head * layout.attention.attention_head_stride +
      query_index * layout.attention.attention_row_stride;
  for (std::size_t key_index = 0; key_index < layout.attention.rows; ++key_index)
    reduction += device_derivative::softmaxRowReductionContribution(
        probabilities[row_offset + key_index], probability_adjoint[row_offset + key_index]);
  return reduction;
}

__global__ void softmaxScaledQueryReverseKernel(
    BatchedAttentionForwardLayout layout,
    const double* key,
    const double* probabilities,
    const double* probability_adjoint,
    double* query_adjoint)
{
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t width = layout.attention.heads * layout.attention.head_width;
  const std::size_t per_configuration = layout.attention.rows * width;
  const std::size_t count = layout.configuration_count * per_configuration;
  if (logical >= count)
    return;
  const std::size_t local = logical % per_configuration;
  const std::size_t configuration = logical / per_configuration;
  const std::size_t feature_in_width = local % width;
  const std::size_t query_index = local / width;
  const std::size_t head = feature_in_width / layout.attention.head_width;
  const double reduction = softmaxRowReduction(
      layout, probabilities, probability_adjoint, configuration, head, query_index);
  const double scale = 1.0 / ::sqrt(static_cast<double>(layout.attention.head_width));
  const std::size_t row_offset = configuration * layout.attention_configuration_stride +
      head * layout.attention.attention_head_stride +
      query_index * layout.attention.attention_row_stride;
  double sum = 0.0;
  for (std::size_t key_index = 0; key_index < layout.attention.rows; ++key_index)
  {
    const double logit_adjoint = device_derivative::softmaxLogitAdjoint(
        probabilities[row_offset + key_index], probability_adjoint[row_offset + key_index],
        reduction);
    const std::size_t key_offset = configuration * layout.feature_configuration_stride +
        key_index * layout.attention.feature_row_stride + feature_in_width;
    sum += device_derivative::scaledQkAdjointContribution(
        logit_adjoint, key[key_offset], scale);
  }
  const std::size_t output_offset = configuration * layout.feature_configuration_stride +
      query_index * layout.attention.feature_row_stride + feature_in_width;
  query_adjoint[output_offset] = sum;
}

__global__ void softmaxScaledKeyReverseKernel(
    BatchedAttentionForwardLayout layout,
    const double* query,
    const double* probabilities,
    const double* probability_adjoint,
    double* key_adjoint)
{
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t width = layout.attention.heads * layout.attention.head_width;
  const std::size_t per_configuration = layout.attention.rows * width;
  const std::size_t count = layout.configuration_count * per_configuration;
  if (logical >= count)
    return;
  const std::size_t local = logical % per_configuration;
  const std::size_t configuration = logical / per_configuration;
  const std::size_t feature_in_width = local % width;
  const std::size_t key_index = local / width;
  const std::size_t head = feature_in_width / layout.attention.head_width;
  const double scale = 1.0 / ::sqrt(static_cast<double>(layout.attention.head_width));
  double sum = 0.0;
  for (std::size_t query_index = 0; query_index < layout.attention.rows; ++query_index)
  {
    const double reduction = softmaxRowReduction(
        layout, probabilities, probability_adjoint, configuration, head, query_index);
    const std::size_t probability_offset = configuration * layout.attention_configuration_stride +
        head * layout.attention.attention_head_stride +
        query_index * layout.attention.attention_row_stride + key_index;
    const double logit_adjoint = device_derivative::softmaxLogitAdjoint(
        probabilities[probability_offset], probability_adjoint[probability_offset], reduction);
    const std::size_t query_offset = configuration * layout.feature_configuration_stride +
        query_index * layout.attention.feature_row_stride + feature_in_width;
    sum += device_derivative::scaledQkAdjointContribution(
        logit_adjoint, query[query_offset], scale);
  }
  const std::size_t output_offset = configuration * layout.feature_configuration_stride +
      key_index * layout.attention.feature_row_stride + feature_in_width;
  key_adjoint[output_offset] = sum;
}

__global__ void orbitalBbarEbarKernel(OrbitalProductReverseLayout layout,
                                      const double* backflow,
                                      const double* envelope,
                                      const double* orbital_adjoint,
                                      double* backflow_adjoint,
                                      double* envelope_adjoint)
{
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t matrix_elements = layout.electron_count * layout.electron_count;
  const std::size_t per_configuration = layout.determinant_count * matrix_elements;
  const std::size_t count = layout.configuration_count * per_configuration;
  if (logical >= count)
    return;
  const std::size_t local = logical % per_configuration;
  const std::size_t configuration = logical / per_configuration;
  const std::size_t matrix_local = local % matrix_elements;
  const std::size_t determinant = local / matrix_elements;
  const std::size_t row = matrix_local / layout.electron_count;
  const std::size_t orbital = matrix_local % layout.electron_count;
  const std::size_t offset = configuration * layout.configuration_stride +
      determinant * layout.determinant_stride + row * layout.row_stride + orbital;
  const double upstream = orbital_adjoint[offset];
  backflow_adjoint[offset] = upstream * envelope[offset];
  envelope_adjoint[offset] = upstream * backflow[offset];
}

__global__ void envelopeParameterReverseKernel(EnvelopeParameterReverseLayout layout,
                                               const double* distances,
                                               const double* pi,
                                               const double* zeta,
                                               const double* envelope_adjoint,
                                               double* response)
{
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count =
      layout.determinant_count * layout.electron_count * layout.nucleus_count;
  if (logical >= count)
    return;
  const std::size_t nucleus = logical % layout.nucleus_count;
  const std::size_t orbital_index =
      (logical / layout.nucleus_count) % layout.electron_count;
  const std::size_t determinant = logical /
      (layout.nucleus_count * layout.electron_count);
  const std::size_t parameter = determinant * layout.parameter_determinant_stride +
      orbital_index * layout.parameter_orbital_stride + nucleus;
  double pi_sum = 0.0;
  double zeta_sum = 0.0;
  for (std::size_t electron = layout.electron_begin; electron < layout.electron_end; ++electron)
  {
    const std::size_t distance_offset =
        electron * layout.distance_electron_stride + nucleus;
    const std::size_t matrix_offset = determinant * layout.matrix_determinant_stride +
        electron * layout.matrix_row_stride + orbital_index;
    const auto partials = device_derivative::envelopeValueAndPartials(
        distances[distance_offset], pi[parameter], zeta[parameter]);
    pi_sum += envelope_adjoint[matrix_offset] * partials.pi_partial;
    zeta_sum += envelope_adjoint[matrix_offset] * partials.zeta_partial;
  }
  response[layout.pi_response_offset + parameter] += pi_sum;
  response[layout.zeta_response_offset + parameter] += zeta_sum;
}

__global__ void cuspAlphaReverseKernel(CuspAlphaReverseLayout layout,
                                       const double* distances,
                                       const std::uint8_t* same_spin,
                                       double same_alpha,
                                       double opposite_alpha,
                                       double* response)
{
  const std::size_t categories = layout.has_same_alpha ? 2 : 1;
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (logical >= categories)
    return;
  const bool select_same = layout.has_same_alpha && logical % categories == 0;
  const double alpha = select_same ? same_alpha : opposite_alpha;
  const double factor = select_same ? 0.25 : 0.5;
  double sum = 0.0;
  for (std::size_t pair = 0; pair < layout.pair_count; ++pair)
  {
    const bool pair_is_same = same_spin[pair] != 0;
    if (pair_is_same == select_same)
      sum += device_derivative::cuspValueAndAlphaPartial(
          distances[pair], alpha, factor).alpha_partial;
  }
  const std::size_t offset = select_same ? layout.same_response_offset
                                         : layout.opposite_response_offset;
  response[offset] += sum;
}

__global__ void derivativeFiniteScanKernel(DerivativeFiniteScanLayout layout,
                                           const double* source,
                                           DerivativeFiniteStatus* status)
{
  const std::size_t piece = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t piece_count = layout.element_count / layout.elements_per_status +
      (layout.element_count % layout.elements_per_status == 0 ? 0 : 1);
  if (piece >= piece_count)
    return;
  const std::size_t begin = piece * layout.elements_per_status;
  const std::size_t remaining = layout.element_count - begin;
  const std::size_t local_count = remaining < layout.elements_per_status
      ? remaining
      : layout.elements_per_status;
  const std::size_t end = begin + local_count;
  std::size_t first = layout.element_count;
  std::size_t count = 0;
  for (std::size_t index = begin; index < end; ++index)
    if (!device_math::isFiniteBinary64(
            source[layout.source_offset + index * layout.source_stride]))
    {
      if (count == 0)
        first = index;
      ++count;
    }
  status[piece] = {first, count};
}

} // namespace

std::size_t responseDotPartialCount(const ResponseDotLayout& layout)
{
  if (layout.element_count == 0)
    return 0;
  validateStridedInterval(layout.element_count, layout.response_offset,
                          layout.response_stride,
                          "PsiFormer response-dot response extent is invalid");
  validateStridedInterval(layout.element_count, layout.direction_offset,
                          layout.direction_stride,
                          "PsiFormer response-dot direction extent is invalid");
  return checkedPieceCount(layout.element_count, layout.elements_per_partial,
                           layout.partial_capacity,
                           "PsiFormer response-dot partial capacity is insufficient");
}

std::size_t derivativeFiniteStatusCount(const DerivativeFiniteScanLayout& layout)
{
  if (layout.element_count == 0)
    return 0;
  validateStridedInterval(layout.element_count, layout.source_offset,
                          layout.source_stride,
                          "PsiFormer derivative finite-scan extent is invalid");
  return checkedPieceCount(layout.element_count, layout.elements_per_status,
                           layout.status_capacity,
                           "PsiFormer derivative finite-scan status capacity is insufficient");
}

Error launchDerivativeZero(Stream stream,
                           const DerivativeVectorLayout& layout,
                           double* destination)
{
  if (layout.element_count == 0)
    return success;
  validateStridedInterval(layout.element_count, layout.destination_offset,
                          layout.destination_stride,
                          "PsiFormer derivative zero destination extent is invalid");
  if (!destination)
    throw std::invalid_argument("PsiFormer derivative zero destination is null");
  derivativeZeroKernel<<<blockCount(layout.element_count,
                                    "PsiFormer derivative zero grid exceeds the CUDA/HIP x dimension"),
                         derivative_block_size, 0, stream>>>(layout, destination);
  return peekAtLastError();
}

Error launchDerivativeCopy(Stream stream,
                           const DerivativeVectorLayout& layout,
                           const double* source,
                           double* destination)
{
  if (layout.element_count == 0)
    return success;
  validateStridedInterval(layout.element_count, layout.source_offset,
                          layout.source_stride,
                          "PsiFormer derivative copy source extent is invalid");
  validateStridedInterval(layout.element_count, layout.destination_offset,
                          layout.destination_stride,
                          "PsiFormer derivative copy destination extent is invalid");
  if (!source || !destination)
    throw std::invalid_argument("PsiFormer derivative copy storage is null");
  derivativeCopyKernel<<<blockCount(layout.element_count,
                                    "PsiFormer derivative copy grid exceeds the CUDA/HIP x dimension"),
                         derivative_block_size, 0, stream>>>(layout, source, destination);
  return peekAtLastError();
}

Error launchRealToComplexAxpy(Stream stream,
                              const RealToComplexAxpyLayout& layout,
                              double alpha_real,
                              double alpha_imag,
                              const double* source,
                              double* accumulator)
{
  if (layout.element_count == 0)
    return success;
  validateStridedInterval(layout.element_count, layout.source_offset,
                          layout.source_stride,
                          "PsiFormer real-to-complex AXPY source extent is invalid");
  if (layout.accumulator_stride < 2)
    throw std::invalid_argument("PsiFormer complex accumulator stride is too small");
  (void)checkedAdd(layout.accumulator_offset,
                   checkedAdd(checkedProduct(layout.element_count - 1,
                                             layout.accumulator_stride,
                                             "PsiFormer complex accumulator extent overflow"),
                              1, "PsiFormer complex accumulator extent overflow"),
                   "PsiFormer complex accumulator extent overflow");
  if (!device_math::isFiniteBinary64(alpha_real) ||
      !device_math::isFiniteBinary64(alpha_imag))
    throw std::invalid_argument("PsiFormer real-to-complex AXPY coefficient is non-finite");
  if (!source || !accumulator)
    throw std::invalid_argument("PsiFormer real-to-complex AXPY storage is null");
  realToComplexAxpyKernel<<<blockCount(
                              layout.element_count,
                              "PsiFormer real-to-complex AXPY grid exceeds the CUDA/HIP x dimension"),
                           derivative_block_size, 0, stream>>>(
      layout, alpha_real, alpha_imag, source, accumulator);
  return peekAtLastError();
}

Error launchResponseDotDirectionPartials(Stream stream,
                                         const ResponseDotLayout& layout,
                                         const double* response,
                                         const double* direction,
                                         double* partials,
                                         DerivativeFiniteStatus* partial_status)
{
  const std::size_t partial_count = responseDotPartialCount(layout);
  if (partial_count == 0)
    return success;
  if (!response || !direction || !partials || !partial_status)
    throw std::invalid_argument("PsiFormer response-dot storage is null");
  responseDotDirectionKernel<<<blockCount(
                                  partial_count,
                                  "PsiFormer response-dot grid exceeds the CUDA/HIP x dimension"),
                               derivative_block_size, 0, stream>>>(
      layout, response, direction, partials, partial_status);
  return peekAtLastError();
}

Error launchFinalizeResponseDotDirection(Stream stream,
                                         std::size_t partial_count,
                                         std::size_t logical_element_count,
                                         const double* partials,
                                         const DerivativeFiniteStatus* partial_status,
                                         double* result,
                                         DerivativeFiniteStatus* aggregate_status)
{
  if (partial_count == 0)
    return success;
  if (logical_element_count == 0)
    throw std::invalid_argument("PsiFormer response-dot logical extent is zero");
  if (!partials || !partial_status || !result || !aggregate_status)
    throw std::invalid_argument("PsiFormer response-dot finalization storage is null");
  finalizeResponseDotDirectionKernel<<<1, 1, 0, stream>>>(
      partial_count, logical_element_count, partials, partial_status, result,
      aggregate_status);
  return peekAtLastError();
}

Error launchResponseDotDirection(Stream stream,
                                 const ResponseDotLayout& layout,
                                 const double* response,
                                 const double* direction,
                                 double* partials,
                                 DerivativeFiniteStatus* partial_status,
                                 double* result,
                                 DerivativeFiniteStatus* aggregate_status)
{
  const std::size_t partial_count = responseDotPartialCount(layout);
  if (partial_count == 0)
    return success;
  Error error = launchResponseDotDirectionPartials(
      stream, layout, response, direction, partials, partial_status);
  if (error != success)
    return error;
  return launchFinalizeResponseDotDirection(
      stream, partial_count, layout.element_count, partials, partial_status, result,
      aggregate_status);
}

Error launchTanhOutputResidualReverse(Stream stream,
                                      const BatchedValueLayout& layout,
                                      const double* tanh_output,
                                      const double* output_adjoint,
                                      double* tanh_input_adjoint,
                                      double* residual_adjoint)
{
  if (layout.configuration_count == 0 && layout.rows == 0 && layout.width == 0 &&
      layout.row_stride == 0 && layout.configuration_stride == 0)
    return success;
  validateBatchedValueLayout(layout);
  if (!tanh_output || !output_adjoint || !tanh_input_adjoint || !residual_adjoint)
    throw std::invalid_argument("PsiFormer tanh/output residual reverse storage is null");
  if (tanh_input_adjoint == residual_adjoint)
    throw std::invalid_argument("PsiFormer tanh and residual adjoints must not alias");
  const std::size_t count = checkedProduct(
      checkedProduct(layout.configuration_count, layout.rows,
                     "PsiFormer tanh reverse row extent overflow"),
      layout.width, "PsiFormer tanh reverse extent overflow");
  tanhOutputResidualReverseKernel<<<blockCount(
                                      count,
                                      "PsiFormer tanh reverse grid exceeds the CUDA/HIP x dimension"),
                                   derivative_block_size, 0, stream>>>(
      layout, tanh_output, output_adjoint, tanh_input_adjoint, residual_adjoint);
  return peekAtLastError();
}

Error launchAttentionContextReverse(Stream stream,
                                    const BatchedAttentionForwardLayout& layout,
                                    const double* attention,
                                    const double* value,
                                    const double* context_adjoint,
                                    double* attention_adjoint,
                                    double* value_adjoint)
{
  if (layout.configuration_count == 0 && layout.attention.rows == 0 &&
      layout.attention.heads == 0 && layout.attention.head_width == 0 &&
      layout.feature_configuration_stride == 0 &&
      layout.attention_configuration_stride == 0)
    return success;
  validateBatchedAttentionForwardLayout(layout);
  if (!attention || !value || !context_adjoint || !attention_adjoint || !value_adjoint)
    throw std::invalid_argument("PsiFormer attention-context reverse storage is null");
  if (attention_adjoint == value_adjoint)
    throw std::invalid_argument("PsiFormer attention and value adjoints must not alias");
  const std::size_t feature_count = checkedProduct(
      layout.configuration_count,
      checkedProduct(layout.attention.rows, layout.attention.featureWidth(),
                     "PsiFormer attention-context feature extent overflow"),
      "PsiFormer attention-context feature batch extent overflow");
  const std::size_t attention_count = checkedProduct(
      layout.configuration_count,
      checkedProduct(layout.attention.heads,
                     checkedProduct(layout.attention.rows, layout.attention.rows,
                                    "PsiFormer attention-context matrix extent overflow"),
                     "PsiFormer attention-context headed extent overflow"),
      "PsiFormer attention-context batch extent overflow");
  attentionContextAttentionReverseKernel<<<blockCount(
      attention_count, "PsiFormer attention-context matrix grid exceeds the CUDA/HIP x dimension"),
      derivative_block_size, 0, stream>>>(layout, value, context_adjoint, attention_adjoint);
  Error error = peekAtLastError();
  if (error != success)
    return error;
  attentionContextValueReverseKernel<<<blockCount(
      feature_count, "PsiFormer attention-context value grid exceeds the CUDA/HIP x dimension"),
      derivative_block_size, 0, stream>>>(layout, attention, context_adjoint, value_adjoint);
  return peekAtLastError();
}

Error launchSoftmaxScaledQkReverse(Stream stream,
                                   const BatchedAttentionForwardLayout& layout,
                                   const double* query,
                                   const double* key,
                                   const double* probabilities,
                                   const double* probability_adjoint,
                                   double* query_adjoint,
                                   double* key_adjoint)
{
  if (layout.configuration_count == 0 && layout.attention.rows == 0 &&
      layout.attention.heads == 0 && layout.attention.head_width == 0 &&
      layout.feature_configuration_stride == 0 &&
      layout.attention_configuration_stride == 0)
    return success;
  validateBatchedAttentionForwardLayout(layout);
  if (!query || !key || !probabilities || !probability_adjoint ||
      !query_adjoint || !key_adjoint)
    throw std::invalid_argument("PsiFormer softmax Q/K reverse storage is null");
  if (query_adjoint == key_adjoint)
    throw std::invalid_argument("PsiFormer query and key adjoints must not alias");
  const std::size_t feature_count = checkedProduct(
      layout.configuration_count,
      checkedProduct(layout.attention.rows, layout.attention.featureWidth(),
                     "PsiFormer softmax Q/K feature extent overflow"),
      "PsiFormer softmax Q/K feature batch extent overflow");
  softmaxScaledQueryReverseKernel<<<blockCount(
      feature_count, "PsiFormer softmax query grid exceeds the CUDA/HIP x dimension"),
      derivative_block_size, 0, stream>>>(
      layout, key, probabilities, probability_adjoint, query_adjoint);
  Error error = peekAtLastError();
  if (error != success)
    return error;
  softmaxScaledKeyReverseKernel<<<blockCount(
      feature_count, "PsiFormer softmax key grid exceeds the CUDA/HIP x dimension"),
      derivative_block_size, 0, stream>>>(
      layout, query, probabilities, probability_adjoint, key_adjoint);
  return peekAtLastError();
}

Error launchOrbitalBbarEbar(Stream stream,
                            const OrbitalProductReverseLayout& layout,
                            const double* backflow,
                            const double* envelope,
                            const double* orbital_adjoint,
                            double* backflow_adjoint,
                            double* envelope_adjoint)
{
  if (layout.configuration_count == 0 && layout.determinant_count == 0 &&
      layout.electron_count == 0 && layout.row_stride == 0 &&
      layout.determinant_stride == 0 && layout.configuration_stride == 0)
    return success;
  validateOrbitalLayout(layout);
  if (!backflow || !envelope || !orbital_adjoint ||
      !backflow_adjoint || !envelope_adjoint)
    throw std::invalid_argument("PsiFormer orbital Bbar/Ebar storage is null");
  if (backflow_adjoint == envelope_adjoint)
    throw std::invalid_argument("PsiFormer orbital Bbar and Ebar must not alias");
  const std::size_t count = checkedProduct(
      layout.configuration_count,
      checkedProduct(layout.determinant_count,
                     checkedProduct(layout.electron_count, layout.electron_count,
                                    "PsiFormer orbital reverse matrix extent overflow"),
                     "PsiFormer orbital reverse determinant extent overflow"),
      "PsiFormer orbital reverse batch extent overflow");
  orbitalBbarEbarKernel<<<blockCount(
                            count,
                            "PsiFormer orbital reverse grid exceeds the CUDA/HIP x dimension"),
                         derivative_block_size, 0, stream>>>(
      layout, backflow, envelope, orbital_adjoint, backflow_adjoint, envelope_adjoint);
  return peekAtLastError();
}

Error launchEnvelopeParameterReverse(Stream stream,
                                     const EnvelopeParameterReverseLayout& layout,
                                     const double* distances,
                                     const double* pi,
                                     const double* zeta,
                                     const double* envelope_adjoint,
                                     double* response)
{
  if (layout.configuration_count == 0 && layout.determinant_count == 0 &&
      layout.electron_count == 0 && layout.nucleus_count == 0)
    return success;
  (void)validateEnvelopeLayout(layout);
  if (!distances || !pi || !zeta || !envelope_adjoint || !response)
    throw std::invalid_argument("PsiFormer envelope parameter reverse storage is null");
  const std::size_t count = checkedProduct(
      layout.determinant_count,
      checkedProduct(layout.electron_count, layout.nucleus_count,
                     "PsiFormer envelope reverse orbital extent overflow"),
      "PsiFormer envelope reverse determinant extent overflow");
  envelopeParameterReverseKernel<<<blockCount(
                                     count,
                                     "PsiFormer envelope reverse grid exceeds the CUDA/HIP x dimension"),
                                  derivative_block_size, 0, stream>>>(
      layout, distances, pi, zeta, envelope_adjoint, response);
  return peekAtLastError();
}

Error launchCuspAlphaReverse(Stream stream,
                             const CuspAlphaReverseLayout& layout,
                             const double* distances,
                             const std::uint8_t* same_spin,
                             double same_alpha,
                             double opposite_alpha,
                             double* response)
{
  if (layout.configuration_count == 0 && layout.pair_count == 0 &&
      layout.distance_configuration_stride == 0 && !layout.has_same_alpha)
    return success;
  if (layout.configuration_count != 1)
    throw std::invalid_argument(
        "PsiFormer cusp reverse requires one outer-serial configuration");
  if (layout.pair_count == 0)
    return success;
  if (layout.distance_configuration_stride < layout.pair_count)
    throw std::invalid_argument("PsiFormer cusp reverse distance stride is too small");
  (void)checkedAdd(
      layout.has_same_alpha
          ? (layout.same_response_offset > layout.opposite_response_offset
                 ? layout.same_response_offset
                 : layout.opposite_response_offset)
          : layout.opposite_response_offset,
      1, "PsiFormer cusp response extent overflow");
  if (layout.has_same_alpha &&
      layout.same_response_offset == layout.opposite_response_offset)
    throw std::invalid_argument("PsiFormer cusp alpha response offsets overlap");
  (void)checkedSpan(layout.configuration_count, layout.distance_configuration_stride,
                    layout.pair_count, "PsiFormer cusp distance batch extent overflow");
  if ((layout.has_same_alpha && !device_math::isFiniteBinary64(same_alpha)) ||
      !device_math::isFiniteBinary64(opposite_alpha))
    throw std::invalid_argument("PsiFormer cusp alpha is non-finite");
  if (!distances || !same_spin || !response)
    throw std::invalid_argument("PsiFormer cusp alpha reverse storage is null");
  const std::size_t count = layout.has_same_alpha ? std::size_t{2} : std::size_t{1};
  cuspAlphaReverseKernel<<<blockCount(
                             count,
                             "PsiFormer cusp reverse grid exceeds the CUDA/HIP x dimension"),
                          derivative_block_size, 0, stream>>>(
      layout, distances, same_spin, same_alpha, opposite_alpha, response);
  return peekAtLastError();
}

Error launchDerivativeFiniteScan(Stream stream,
                                 const DerivativeFiniteScanLayout& layout,
                                 const double* source,
                                 DerivativeFiniteStatus* status)
{
  const std::size_t status_count = derivativeFiniteStatusCount(layout);
  if (status_count == 0)
    return success;
  if (!source || !status)
    throw std::invalid_argument("PsiFormer derivative finite-scan storage is null");
  derivativeFiniteScanKernel<<<blockCount(
                                 status_count,
                                 "PsiFormer derivative finite-scan grid exceeds the CUDA/HIP x dimension"),
                              derivative_block_size, 0, stream>>>(layout, source, status);
  return peekAtLastError();
}

} // namespace qmcplusplus::psiformer::device
