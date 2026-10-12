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

#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer::device
{
namespace
{

constexpr unsigned int spatial_reverse_block_size = 128;

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
                              double* logit_adjoint)
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

} // namespace qmcplusplus::psiformer::device
