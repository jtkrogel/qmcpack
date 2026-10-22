//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceSpatialReverse.cu
 * @brief Deterministic CUDA/HIP full-VGL spatial-jet reverse leaves.
 */

#include "PsiFormerDeviceSpatialReverse.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPeriodicPairAttention.h"

#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer::device
{
namespace
{

constexpr unsigned int spatial_reverse_block_size = 128;

/// Each parameter sums all ordered pair jets without atomic response updates.
__global__ void periodicPairAttentionJetParameterReverseKernel(SpatialAttentionJetLayout layout,
    periodic_geometry::Cell cell, const double* positions, std::size_t spin_up_count,
    const double* logit_adjoint, double* weight_response, double* bias_response)
{
  const auto parameter = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (parameter >= 9 * layout.heads)
    return;
  const auto feature = parameter / layout.heads, head = parameter % layout.heads;
  double response = 0;
  for (std::size_t query = 0; query < layout.rows; ++query)
    for (std::size_t key = 0; key < layout.rows; ++key)
    {
      if (query == key)
        continue;
      periodic_geometry::LogComplementPairJet pair;
      if (!periodic_geometry::evaluateLogComplementPair(cell, positions + 3 * query, positions + 3 * key,
          (query < spin_up_count) == (key < spin_up_count) ? 1 : -1, pair) || !pair.differentiable)
      {
        response = __longlong_as_double(0x7ff8000000000000LL);
        break;
      }
      response += periodic_geometry::logComplementAttentionParameterAdjoint(layout, pair,
          logit_adjoint, 0, head, query, key, feature);
    }
  if (feature < 8)
    weight_response[parameter] += response;
  else
    bias_response[head] += response;
}

__global__ void elementwiseProductJetReverseKernel(
    SpatialElementwiseJetLayout layout,
    const double* left,
    const double* right,
    const double* output_adjoint,
    double scale,
    double* left_adjoint,
    double* right_adjoint)
{
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = layout.row_count * layout.row_width;
  if (logical >= count)
    return;
  const std::size_t element = (logical / layout.row_width) * layout.row_stride +
      logical % layout.row_width;
  spatial_jet_reverse::productJetReverseElement(
      layout.jets, left, element, layout.jets, right, element, layout.jets,
      output_adjoint, element, scale, left_adjoint, right_adjoint);
}

__global__ void tanhJetReverseKernel(SpatialElementwiseJetLayout layout,
                                     const double* input,
                                     const double* tanh_output,
                                     const double* output_adjoint,
                                     double* input_adjoint)
{
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = layout.row_count * layout.row_width;
  if (logical >= count)
    return;
  const std::size_t element = (logical / layout.row_width) * layout.row_stride +
      logical % layout.row_width;
  spatial_jet_reverse::tanhJetReverseElement(
      layout.jets, input, tanh_output, output_adjoint, element, input_adjoint);
}

/// One column owns all its jet adjoints; retain the original serial inner sums.
__device__ inline void softmaxJetColumnReverseElement(
    const double* logits_row,
    const double* probabilities_row,
    const double* probability_adjoint_row,
    std::size_t width,
    std::size_t plane_stride,
    std::size_t gradient_lanes,
    std::size_t electron_count,
    double* logit_adjoint_row,
    std::size_t differentiated) noexcept
{
  // Value cotangents retain the original row reduction and column accumulation.
  const double value_mean = spatial_jet_reverse::detail::weightedMean(
      probabilities_row, probability_adjoint_row, width);
  const double probability = probabilities_row[differentiated];
  double value_result = probability *
      (probability_adjoint_row[differentiated] - value_mean);

  // Each gradient plane keeps its serial means and Laplacian coupling order.
  for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
  {
    const double* logit_gradient = logits_row + (1 + lane) * plane_stride;
    const double* probability_gradient_adjoint =
        probability_adjoint_row + (1 + lane) * plane_stride;
    const double gradient_mean = spatial_jet_reverse::detail::weightedMean(
        probabilities_row, logit_gradient, width);
    const double adjoint_mean = spatial_jet_reverse::detail::weightedMean(
        probabilities_row, probability_gradient_adjoint, width);
    double centered_product_mean = 0.0;
    for (std::size_t index = 0; index < width; ++index)
      centered_product_mean += probabilities_row[index] *
          (probability_gradient_adjoint[index] - adjoint_mean) *
          (logit_gradient[index] - gradient_mean);
    value_result += probability *
        ((probability_gradient_adjoint[differentiated] - adjoint_mean) *
             (logit_gradient[differentiated] - gradient_mean) -
         centered_product_mean);

    double gradient_result = probability *
        (probability_gradient_adjoint[differentiated] - adjoint_mean);
    const std::size_t electron = lane / 3;
    const double* laplacian_adjoint = probability_adjoint_row +
        (1 + gradient_lanes + electron) * plane_stride;
    const double laplacian_adjoint_mean = spatial_jet_reverse::detail::weightedMean(
        probabilities_row, laplacian_adjoint, width);
    double coupling = 0.0;
    for (std::size_t index = 0; index < width; ++index)
      coupling += probabilities_row[index] *
          (laplacian_adjoint[index] - laplacian_adjoint_mean) *
          (logit_gradient[index] - gradient_mean);
    gradient_result += 2.0 * probability *
        ((laplacian_adjoint[differentiated] - laplacian_adjoint_mean) *
             (logit_gradient[differentiated] - gradient_mean) -
         coupling);
    logit_adjoint_row[(1 + lane) * plane_stride + differentiated] +=
        gradient_result;
  }

  // Laplacian cotangents update this column's value and electron plane only.
  for (std::size_t electron = 0; electron < electron_count; ++electron)
  {
    const double* logit_laplacian =
        logits_row + (1 + gradient_lanes + electron) * plane_stride;
    const double* laplacian_adjoint = probability_adjoint_row +
        (1 + gradient_lanes + electron) * plane_stride;
    const double laplacian_mean = spatial_jet_reverse::detail::weightedMean(
        probabilities_row, logit_laplacian, width);
    const double adjoint_mean = spatial_jet_reverse::detail::weightedMean(
        probabilities_row, laplacian_adjoint, width);

    double gradient_square_mean = 0.0;
    for (std::size_t index = 0; index < width; ++index)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const std::size_t lane = 3 * electron + dimension;
        const double* gradient = logits_row + (1 + lane) * plane_stride;
        const double gradient_mean = spatial_jet_reverse::detail::weightedMean(
            probabilities_row, gradient, width);
        const double centered = gradient[index] - gradient_mean;
        gradient_square_mean += probabilities_row[index] * centered * centered;
      }

    double differentiated_gradient_square = 0.0;
    double coupling_dot = 0.0;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const std::size_t lane = 3 * electron + dimension;
      const double* gradient = logits_row + (1 + lane) * plane_stride;
      const double gradient_mean = spatial_jet_reverse::detail::weightedMean(
          probabilities_row, gradient, width);
      const double differentiated_centered =
          gradient[differentiated] - gradient_mean;
      differentiated_gradient_square +=
          differentiated_centered * differentiated_centered;
      double coupling = 0.0;
      for (std::size_t index = 0; index < width; ++index)
        coupling += probabilities_row[index] *
            (laplacian_adjoint[index] - adjoint_mean) *
            (gradient[index] - gradient_mean);
      coupling_dot += coupling * differentiated_centered;
    }
    const double b = logit_laplacian[differentiated] - laplacian_mean +
        differentiated_gradient_square - gradient_square_mean;
    const double r =
        (laplacian_adjoint[differentiated] - adjoint_mean) * b -
        2.0 * coupling_dot;
    double r_mean = 0.0;
    for (std::size_t index = 0; index < width; ++index)
    {
      double square = 0.0;
      double index_coupling_dot = 0.0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const std::size_t lane = 3 * electron + dimension;
        const double* gradient = logits_row + (1 + lane) * plane_stride;
        const double gradient_mean = spatial_jet_reverse::detail::weightedMean(
            probabilities_row, gradient, width);
        const double centered = gradient[index] - gradient_mean;
        square += centered * centered;
        double coupling = 0.0;
        for (std::size_t reduced = 0; reduced < width; ++reduced)
          coupling += probabilities_row[reduced] *
              (laplacian_adjoint[reduced] - adjoint_mean) *
              (gradient[reduced] - gradient_mean);
        index_coupling_dot += coupling * centered;
      }
      const double index_b = logit_laplacian[index] - laplacian_mean +
          square - gradient_square_mean;
      r_mean += probabilities_row[index] *
          ((laplacian_adjoint[index] - adjoint_mean) * index_b -
           2.0 * index_coupling_dot);
    }
    value_result += probability * (r - r_mean);
    logit_adjoint_row[(1 + gradient_lanes + electron) * plane_stride +
                      differentiated] +=
        probability * (laplacian_adjoint[differentiated] - adjoint_mean);
  }
  logit_adjoint_row[differentiated] += value_result;
}

/// Flatten disjoint row/column outputs without atomics or extra numerical owners.
__global__ void softmaxJetColumnReverseKernel(SoftmaxJetRowLayout layout,
    const double* logits, const double* probabilities,
    const double* probability_adjoint, double* logit_adjoint)
{
  const auto logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto row = logical / layout.row_width;
  if (row >= layout.row_count)
    return;
  const auto differentiated = logical % layout.row_width;
  const auto group = row / layout.rows_per_group;
  const auto row_in_group = row % layout.rows_per_group;
  const auto offset = group * layout.row_group_stride + row_in_group * layout.row_stride;
  softmaxJetColumnReverseElement(logits + offset, probabilities + offset,
      probability_adjoint + offset, layout.row_width, layout.jets.plane_stride,
      layout.jets.gradient_lanes, layout.jets.laplacian_lanes,
      logit_adjoint + offset, differentiated);
}

__global__ void softmaxJetReverseKernel(SoftmaxJetRowLayout layout,
                                        const double* logits,
                                        const double* probabilities,
                                        const double* probability_adjoint,
                                        double* logit_adjoint)
{
  const std::size_t row = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= layout.row_count)
    return;
  const std::size_t group = row / layout.rows_per_group;
  const std::size_t row_in_group = row % layout.rows_per_group;
  const std::size_t offset = group * layout.row_group_stride +
      row_in_group * layout.row_stride;
  spatial_jet_reverse::softmaxJetRowReverse(
      logits + offset, probabilities + offset, probability_adjoint + offset,
      layout.row_width, layout.jets.plane_stride, layout.jets.gradient_lanes,
      layout.jets.laplacian_lanes, logit_adjoint + offset);
}

__global__ void attentionContextAttentionJetReverseKernel(
    SpatialAttentionJetLayout layout,
    const double* attention,
    const double* value,
    const double* context_adjoint,
    double* attention_adjoint)
{
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = layout.heads * layout.rows * layout.rows;
  if (logical >= count)
    return;
  const std::size_t head = logical / (layout.rows * layout.rows);
  const std::size_t pair = logical % (layout.rows * layout.rows);
  const std::size_t query = pair / layout.rows;
  const std::size_t key = pair % layout.rows;
  const std::size_t attention_element =
      spatial_jet::attentionElement(layout, head, query, key);
  for (std::size_t plane = 0; plane < layout.attention.plane_count; ++plane)
  {
    double contribution = 0.0;
    for (std::size_t feature = 0; feature < layout.head_width; ++feature)
      contribution += spatial_jet_reverse::detail::productLeftContribution(
          layout.attention, attention, attention_element, layout.features, value,
          spatial_jet::featureElement(layout, key, head, feature), layout.features,
          context_adjoint,
          spatial_jet::featureElement(layout, query, head, feature), plane, 1.0);
    attention_adjoint[layout.attention.uncheckedPlaneOffset(
        0, plane, attention_element)] += contribution;
  }
}

__global__ void attentionContextValueJetReverseKernel(
    SpatialAttentionJetLayout layout,
    const double* attention,
    const double* value,
    const double* context_adjoint,
    double* value_adjoint)
{
  const std::size_t feature_width = layout.heads * layout.head_width;
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = layout.rows * feature_width;
  if (logical >= count)
    return;
  const std::size_t key = logical / feature_width;
  const std::size_t feature_index = logical % feature_width;
  const std::size_t head = feature_index / layout.head_width;
  const std::size_t feature = feature_index % layout.head_width;
  const std::size_t value_element =
      spatial_jet::featureElement(layout, key, head, feature);
  for (std::size_t plane = 0; plane < layout.features.plane_count; ++plane)
  {
    double contribution = 0.0;
    for (std::size_t query = 0; query < layout.rows; ++query)
      contribution += spatial_jet_reverse::detail::productLeftContribution(
          layout.features, value, value_element, layout.attention, attention,
          spatial_jet::attentionElement(layout, head, query, key), layout.features,
          context_adjoint,
          spatial_jet::featureElement(layout, query, head, feature), plane, 1.0);
    value_adjoint[layout.features.uncheckedPlaneOffset(0, plane, value_element)] +=
        contribution;
  }
}

__global__ void attentionQueryJetReverseKernel(
    SpatialAttentionJetLayout layout,
    const double* query,
    const double* key,
    const double* logit_adjoint,
    double* query_adjoint)
{
  const std::size_t feature_width = layout.heads * layout.head_width;
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = layout.rows * feature_width;
  if (logical >= count)
    return;
  const std::size_t row = logical / feature_width;
  const std::size_t feature_index = logical % feature_width;
  const std::size_t head = feature_index / layout.head_width;
  const std::size_t feature = feature_index % layout.head_width;
  const std::size_t query_element =
      spatial_jet::featureElement(layout, row, head, feature);
  const double scale = 1.0 / sqrt(static_cast<double>(layout.head_width));
  for (std::size_t plane = 0; plane < layout.features.plane_count; ++plane)
  {
    double contribution = 0.0;
    for (std::size_t key_row = 0; key_row < layout.rows; ++key_row)
      contribution += spatial_jet_reverse::detail::productLeftContribution(
          layout.features, query, query_element, layout.features, key,
          spatial_jet::featureElement(layout, key_row, head, feature),
          layout.attention, logit_adjoint,
          spatial_jet::attentionElement(layout, head, row, key_row), plane, scale);
    query_adjoint[layout.features.uncheckedPlaneOffset(0, plane, query_element)] +=
        contribution;
  }
}

__global__ void attentionKeyJetReverseKernel(
    SpatialAttentionJetLayout layout,
    const double* query,
    const double* key,
    const double* logit_adjoint,
    double* key_adjoint)
{
  const std::size_t feature_width = layout.heads * layout.head_width;
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = layout.rows * feature_width;
  if (logical >= count)
    return;
  const std::size_t row = logical / feature_width;
  const std::size_t feature_index = logical % feature_width;
  const std::size_t head = feature_index / layout.head_width;
  const std::size_t feature = feature_index % layout.head_width;
  const std::size_t key_element =
      spatial_jet::featureElement(layout, row, head, feature);
  const double scale = 1.0 / sqrt(static_cast<double>(layout.head_width));
  for (std::size_t plane = 0; plane < layout.features.plane_count; ++plane)
  {
    double contribution = 0.0;
    for (std::size_t query_row = 0; query_row < layout.rows; ++query_row)
      contribution += spatial_jet_reverse::detail::productLeftContribution(
          layout.features, key, key_element, layout.features, query,
          spatial_jet::featureElement(layout, query_row, head, feature),
          layout.attention, logit_adjoint,
          spatial_jet::attentionElement(layout, head, query_row, row), plane, scale);
    key_adjoint[layout.features.uncheckedPlaneOffset(0, plane, key_element)] +=
        contribution;
  }
}

unsigned int reverseBlocks(std::size_t count, const char* description)
{
  if (count > std::numeric_limits<unsigned int>::max())
    throw std::length_error(description);
  return static_cast<unsigned int>(
      (count + spatial_reverse_block_size - 1) / spatial_reverse_block_size);
}

Error peekReverseError()
{
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

void rejectAliasedOutputs(const double* first,
                          const double* second,
                          const char* description)
{
  if (first == second)
    throw std::invalid_argument(description);
}

} // namespace

void denseJetsReverse(AcceleratorBlasHandle& handle,
                      Stream stream,
                      const SpatialDenseJetLayout& layout,
                      const double* source,
                      const double* weight,
                      const double* target_adjoint,
                      double* source_adjoint,
                      double* weight_adjoint,
                      double* bias_adjoint)
{
  if (layout.source.configuration_count == 0 &&
      layout.target.configuration_count == 0 && layout.rows == 0 &&
      layout.input_width == 0 && layout.output_width == 0)
    return;
  validateSpatialDenseJetLayout(layout);
  spatial_jet_reverse::validateMatchingFullVglReverseLayouts(
      layout.source, layout.target, layout.target);
  if (!source || !weight || !target_adjoint || !source_adjoint ||
      !weight_adjoint || !bias_adjoint)
    throw std::invalid_argument("PsiFormer dense-jet reverse device storage is null");
  rejectAliasedOutputs(source_adjoint, weight_adjoint,
                       "PsiFormer dense-jet reverse outputs alias");
  rejectAliasedOutputs(source_adjoint, bias_adjoint,
                       "PsiFormer dense-jet reverse outputs alias");
  rejectAliasedOutputs(weight_adjoint, bias_adjoint,
                       "PsiFormer dense-jet reverse outputs alias");
  if (source_adjoint == source || source_adjoint == weight ||
      source_adjoint == target_adjoint || weight_adjoint == source ||
      weight_adjoint == weight || weight_adjoint == target_adjoint ||
      bias_adjoint == source || bias_adjoint == weight ||
      bias_adjoint == target_adjoint)
    throw std::invalid_argument(
        "PsiFormer dense-jet reverse input/output storage aliases");
  const DenseForwardLayout plane_layout = makeDenseForwardLayout(
      layout.rows, layout.input_width, layout.output_width,
      layout.source_row_stride, layout.weight_row_stride,
      layout.target_row_stride);
  for (std::size_t plane = 0; plane < layout.source.plane_count; ++plane)
  {
    const double* source_plane = source + plane * layout.source.plane_stride;
    const double* target_adjoint_plane =
        target_adjoint + plane * layout.target.plane_stride;
    double* source_adjoint_plane =
        source_adjoint + plane * layout.source.plane_stride;
    accumulateDenseInputAdjoint(handle, plane_layout, target_adjoint_plane,
                                weight, source_adjoint_plane);
    accumulateDenseWeightAdjoint(handle, plane_layout, source_plane,
                                 target_adjoint_plane, weight_adjoint);
  }
  const Error error = launchDenseBiasAdjoint(stream, plane_layout, target_adjoint,
                                              bias_adjoint);
  if (error != success)
    throw std::runtime_error("PsiFormer dense-jet bias reverse launch failed");
}

Error launchElementwiseProductJetReverse(
    Stream stream,
    const SpatialElementwiseJetLayout& layout,
    const double* left,
    const double* right,
    const double* output_adjoint,
    double scale,
    double* left_adjoint,
    double* right_adjoint)
{
  if (layout.jets.configuration_count == 0 && layout.row_count == 0 &&
      layout.row_width == 0)
    return success;
  validateSpatialElementwiseJetLayout(layout);
  spatial_jet_reverse::validateFullVglReverseLayout(layout.jets);
  if (!left || !right || !output_adjoint || !left_adjoint || !right_adjoint)
    throw std::invalid_argument("PsiFormer product-jet reverse device storage is null");
  rejectAliasedOutputs(left_adjoint, right_adjoint,
                       "PsiFormer product-jet reverse outputs alias");
  if (left_adjoint == left || left_adjoint == right ||
      left_adjoint == output_adjoint || right_adjoint == left ||
      right_adjoint == right || right_adjoint == output_adjoint)
    throw std::invalid_argument(
        "PsiFormer product-jet reverse input/output storage aliases");
  const std::size_t count = spatial_detail::checkedProduct(
      layout.row_count, layout.row_width,
      "PsiFormer product-jet reverse extent overflow");
  elementwiseProductJetReverseKernel<<<
      reverseBlocks(count, "PsiFormer product-jet reverse grid exceeds device x dimension"),
      spatial_reverse_block_size, 0, stream>>>(
      layout, left, right, output_adjoint, scale, left_adjoint, right_adjoint);
  return peekReverseError();
}

Error launchTanhJetReverse(Stream stream,
                           const SpatialElementwiseJetLayout& layout,
                           const double* input,
                           const double* tanh_output,
                           const double* output_adjoint,
                           double* input_adjoint)
{
  if (layout.jets.configuration_count == 0 && layout.row_count == 0 &&
      layout.row_width == 0)
    return success;
  validateSpatialElementwiseJetLayout(layout);
  spatial_jet_reverse::validateFullVglReverseLayout(layout.jets);
  if (!input || !tanh_output || !output_adjoint || !input_adjoint)
    throw std::invalid_argument("PsiFormer tanh-jet reverse device storage is null");
  if (input_adjoint == input || input_adjoint == tanh_output ||
      input_adjoint == output_adjoint)
    throw std::invalid_argument(
        "PsiFormer tanh-jet reverse input/output storage aliases");
  const std::size_t count = spatial_detail::checkedProduct(
      layout.row_count, layout.row_width,
      "PsiFormer tanh-jet reverse extent overflow");
  tanhJetReverseKernel<<<
      reverseBlocks(count, "PsiFormer tanh-jet reverse grid exceeds device x dimension"),
      spatial_reverse_block_size, 0, stream>>>(
      layout, input, tanh_output, output_adjoint, input_adjoint);
  return peekReverseError();
}

Error launchSoftmaxJetReverse(Stream stream,
                              const SoftmaxJetRowLayout& layout,
                              const double* logits,
                              const double* probabilities,
                              const double* probability_adjoint,
                              double* logit_adjoint,
                              bool differentiated_columns)
{
  if (layout.jets.configuration_count == 0 && layout.row_count == 0 &&
      layout.row_width == 0)
    return success;
  validateSoftmaxJetRowLayout(layout);
  spatial_jet_reverse::validateFullVglReverseLayout(layout.jets);
  if (!logits || !probabilities || !probability_adjoint || !logit_adjoint)
    throw std::invalid_argument("PsiFormer softmax-jet reverse device storage is null");
  if (logit_adjoint == logits || logit_adjoint == probabilities ||
      logit_adjoint == probability_adjoint)
    throw std::invalid_argument(
        "PsiFormer softmax-jet reverse input/output storage aliases");
  if (differentiated_columns)
  {
    const auto count = spatial_detail::checkedProduct(layout.row_count, layout.row_width,
        "PsiFormer softmax column reverse extent overflow");
    softmaxJetColumnReverseKernel<<<
        reverseBlocks(count, "PsiFormer softmax column reverse grid exceeds device x dimension"),
        spatial_reverse_block_size, 0, stream>>>(
        layout, logits, probabilities, probability_adjoint, logit_adjoint);
    return peekReverseError();
  }
  softmaxJetReverseKernel<<<
      reverseBlocks(layout.row_count,
                    "PsiFormer softmax-jet reverse grid exceeds device x dimension"),
      spatial_reverse_block_size, 0, stream>>>(
      layout, logits, probabilities, probability_adjoint, logit_adjoint);
  return peekReverseError();
}

Error launchAttentionContextJetReverse(
    Stream stream,
    const SpatialAttentionJetLayout& layout,
    const double* attention,
    const double* value,
    const double* context_adjoint,
    double* attention_adjoint,
    double* value_adjoint)
{
  if (layout.features.configuration_count == 0 && layout.rows == 0 &&
      layout.heads == 0 && layout.head_width == 0)
    return success;
  validateSpatialAttentionJetLayout(layout);
  spatial_jet_reverse::validateMatchingFullVglReverseLayouts(
      layout.attention, layout.features, layout.features);
  if (!attention || !value || !context_adjoint || !attention_adjoint ||
      !value_adjoint)
    throw std::invalid_argument(
        "PsiFormer attention-context reverse device storage is null");
  rejectAliasedOutputs(attention_adjoint, value_adjoint,
                       "PsiFormer attention-context reverse outputs alias");
  if (attention_adjoint == attention || attention_adjoint == value ||
      attention_adjoint == context_adjoint || value_adjoint == attention ||
      value_adjoint == value || value_adjoint == context_adjoint)
    throw std::invalid_argument(
        "PsiFormer attention-context reverse input/output storage aliases");
  const std::size_t attention_count = spatial_detail::checkedProduct(
      layout.heads,
      spatial_detail::checkedProduct(
          layout.rows, layout.rows,
          "PsiFormer attention-context reverse row-pair count overflow"),
      "PsiFormer attention-context reverse attention extent overflow");
  const std::size_t value_count = spatial_detail::checkedProduct(
      layout.rows,
      spatial_detail::checkedProduct(
          layout.heads, layout.head_width,
          "PsiFormer attention-context reverse feature width overflow"),
      "PsiFormer attention-context reverse value extent overflow");
  attentionContextAttentionJetReverseKernel<<<
      reverseBlocks(attention_count,
                    "PsiFormer attention-context reverse grid exceeds device x dimension"),
      spatial_reverse_block_size, 0, stream>>>(
      layout, attention, value, context_adjoint, attention_adjoint);
  Error error = peekReverseError();
  if (error != success)
    return error;
  attentionContextValueJetReverseKernel<<<
      reverseBlocks(value_count,
                    "PsiFormer attention-context reverse grid exceeds device x dimension"),
      spatial_reverse_block_size, 0, stream>>>(
      layout, attention, value, context_adjoint, value_adjoint);
  return peekReverseError();
}

Error launchAttentionLogitJetReverse(
    Stream stream,
    const SpatialAttentionJetLayout& layout,
    const double* query,
    const double* key,
    const double* logit_adjoint,
    double* query_adjoint,
    double* key_adjoint)
{
  if (layout.features.configuration_count == 0 && layout.rows == 0 &&
      layout.heads == 0 && layout.head_width == 0)
    return success;
  validateSpatialAttentionJetLayout(layout);
  spatial_jet_reverse::validateMatchingFullVglReverseLayouts(
      layout.features, layout.features, layout.attention);
  if (!query || !key || !logit_adjoint || !query_adjoint || !key_adjoint)
    throw std::invalid_argument(
        "PsiFormer attention-logit reverse device storage is null");
  rejectAliasedOutputs(query_adjoint, key_adjoint,
                       "PsiFormer attention-logit reverse outputs alias");
  if (query_adjoint == query || query_adjoint == key ||
      query_adjoint == logit_adjoint || key_adjoint == query ||
      key_adjoint == key || key_adjoint == logit_adjoint)
    throw std::invalid_argument(
        "PsiFormer attention-logit reverse input/output storage aliases");
  const std::size_t count = spatial_detail::checkedProduct(
      layout.rows,
      spatial_detail::checkedProduct(
          layout.heads, layout.head_width,
          "PsiFormer attention-logit reverse feature width overflow"),
      "PsiFormer attention-logit reverse extent overflow");
  attentionQueryJetReverseKernel<<<
      reverseBlocks(count,
                    "PsiFormer attention-logit reverse grid exceeds device x dimension"),
      spatial_reverse_block_size, 0, stream>>>(
      layout, query, key, logit_adjoint, query_adjoint);
  Error error = peekReverseError();
  if (error != success)
    return error;
  attentionKeyJetReverseKernel<<<
      reverseBlocks(count,
                    "PsiFormer attention-logit reverse grid exceeds device x dimension"),
      spatial_reverse_block_size, 0, stream>>>(
      layout, query, key, logit_adjoint, key_adjoint);
  return peekReverseError();
}

Error launchPeriodicPairAttentionJetParameterReverse(Stream stream,
    const SpatialAttentionJetLayout& layout, const periodic_geometry::Cell& cell,
    const double* positions, std::size_t spin_up_count, const double* logit_adjoint,
    double* weight_response, double* bias_response)
{
  validateSpatialAttentionJetLayout(layout);
  spatial_jet_reverse::validateMatchingFullVglReverseLayouts(layout.features, layout.features, layout.attention);
  if (layout.attention.configuration_count != 1 || layout.attention.mode != SpatialJetMode::FULL_VGL)
    throw std::invalid_argument("Periodic pair-attention jet reverse requires one FULL_VGL configuration");
  periodic_geometry::validateCell(cell);
  if (!positions || !logit_adjoint || !weight_response || !bias_response ||
      !spin_up_count || spin_up_count >= layout.rows || weight_response == bias_response ||
      weight_response == logit_adjoint || bias_response == logit_adjoint)
    throw std::invalid_argument("Periodic pair-attention jet reverse inputs or outputs are invalid");
  const auto parameters = spatial_detail::checkedProduct(9, layout.heads, "Pair-attention response extent overflow");
  spatial_detail::checkedProduct(3, layout.rows, "Pair-attention position extent overflow");
  periodicPairAttentionJetParameterReverseKernel<<<reverseBlocks(parameters,
      "Pair-attention jet response grid exceeds device x dimension"), spatial_reverse_block_size, 0, stream>>>(
      layout, cell, positions, spin_up_count, logit_adjoint, weight_response, bias_response);
  return peekReverseError();
}

} // namespace qmcplusplus::psiformer::device
