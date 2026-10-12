//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerSpatialJetReverse.h
 * @brief Allocation-free full-VGL reverse primitives for PsiFormer spatial jets.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_SPATIAL_JET_REVERSE_H
#define QMCPLUSPLUS_PSIFORMER_SPATIAL_JET_REVERSE_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerOpenSpatial.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialJetKernels.h"

#include <cstddef>
#include <stdexcept>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QMC_PF_SPATIAL_REVERSE_HOST_DEVICE __host__ __device__
#else
#define QMC_PF_SPATIAL_REVERSE_HOST_DEVICE
#endif

namespace qmcplusplus::psiformer::spatial_jet_reverse
{

inline void validateFullVglReverseLayout(const SpatialJetLayout& layout)
{
  validateSpatialJetLayout(layout);
  if (layout.mode != SpatialJetMode::FULL_VGL || layout.configuration_count != 1)
    throw std::invalid_argument(
        "PsiFormer spatial-jet reverse requires one FULL_VGL configuration");
}

inline void validateMatchingFullVglReverseLayouts(const SpatialJetLayout& left,
                                                  const SpatialJetLayout& right,
                                                  const SpatialJetLayout& output)
{
  validateFullVglReverseLayout(left);
  validateFullVglReverseLayout(right);
  validateFullVglReverseLayout(output);
  if (!haveMatchingSpatialPlanes(left, right) ||
      !haveMatchingSpatialPlanes(left, output))
    throw std::invalid_argument(
        "PsiFormer spatial-jet reverse plane layouts do not match");
}

namespace detail
{

QMC_PF_SPATIAL_REVERSE_HOST_DEVICE inline double productLeftContribution(
    const SpatialJetLayout& left_layout,
    const double* left,
    std::size_t left_element,
    const SpatialJetLayout& right_layout,
    const double* right,
    std::size_t right_element,
    const SpatialJetLayout& output_layout,
    const double* output_adjoint,
    std::size_t output_element,
    std::size_t input_plane,
    double scale) noexcept
{
  const double right_value = right[right_layout.uncheckedValueOffset(0, right_element)];
  if (input_plane == 0)
  {
    double result = output_adjoint[output_layout.uncheckedValueOffset(0, output_element)] *
        right_value;
    for (std::size_t lane = 0; lane < left_layout.gradient_lanes; ++lane)
      result += output_adjoint[output_layout.uncheckedGradientOffset(0, lane, output_element)] *
          right[right_layout.uncheckedGradientOffset(0, lane, right_element)];
    for (std::size_t electron = 0; electron < left_layout.laplacian_lanes; ++electron)
      result += output_adjoint[output_layout.uncheckedLaplacianOffset(
                    0, electron, output_element)] *
          right[right_layout.uncheckedLaplacianOffset(0, electron, right_element)];
    return scale * result;
  }
  if (input_plane <= left_layout.gradient_lanes)
  {
    const std::size_t lane = input_plane - 1;
    const std::size_t electron = lane / 3;
    return scale *
        (output_adjoint[output_layout.uncheckedGradientOffset(0, lane, output_element)] *
             right_value +
         2.0 * output_adjoint[output_layout.uncheckedLaplacianOffset(
                   0, electron, output_element)] *
             right[right_layout.uncheckedGradientOffset(0, lane, right_element)]);
  }
  const std::size_t electron = input_plane - 1 - left_layout.gradient_lanes;
  return scale *
      output_adjoint[output_layout.uncheckedLaplacianOffset(
          0, electron, output_element)] *
      right_value;
}

QMC_PF_SPATIAL_REVERSE_HOST_DEVICE inline std::size_t softmaxRowElement(
    std::size_t index) noexcept
{
  return index;
}

QMC_PF_SPATIAL_REVERSE_HOST_DEVICE inline double weightedMean(
    const double* probabilities,
    const double* values,
    std::size_t width,
    std::size_t value_stride = 1) noexcept
{
  double result = 0.0;
  for (std::size_t index = 0; index < width; ++index)
    result += probabilities[index] * values[index * value_stride];
  return result;
}

} // namespace detail

/** Reverse one scaled product jet, accumulating into both input adjoints. */
QMC_PF_SPATIAL_REVERSE_HOST_DEVICE inline void productJetReverseElement(
    const SpatialJetLayout& left_layout,
    const double* left,
    std::size_t left_element,
    const SpatialJetLayout& right_layout,
    const double* right,
    std::size_t right_element,
    const SpatialJetLayout& output_layout,
    const double* output_adjoint,
    std::size_t output_element,
    double scale,
    double* left_adjoint,
    double* right_adjoint) noexcept
{
  for (std::size_t plane = 0; plane < left_layout.plane_count; ++plane)
  {
    left_adjoint[left_layout.uncheckedPlaneOffset(0, plane, left_element)] +=
        detail::productLeftContribution(
            left_layout, left, left_element, right_layout, right, right_element,
            output_layout, output_adjoint, output_element, plane, scale);
    right_adjoint[right_layout.uncheckedPlaneOffset(0, plane, right_element)] +=
        detail::productLeftContribution(
            right_layout, right, right_element, left_layout, left, left_element,
            output_layout, output_adjoint, output_element, plane, scale);
  }
}

/** Reverse one tanh jet through value, all Cartesian gradients, and trace lanes. */
QMC_PF_SPATIAL_REVERSE_HOST_DEVICE inline void tanhJetReverseElement(
    const SpatialJetLayout& layout,
    const double* input,
    const double* tanh_output,
    const double* output_adjoint,
    std::size_t element,
    double* input_adjoint) noexcept
{
  const double y = tanh_output[layout.uncheckedValueOffset(0, element)];
  const double first  = 1.0 - y * y;
  const double second = -2.0 * y * first;
  const double third  = -2.0 * first * (1.0 - 3.0 * y * y);

  double value_adjoint =
      output_adjoint[layout.uncheckedValueOffset(0, element)] * first;
  for (std::size_t lane = 0; lane < layout.gradient_lanes; ++lane)
    value_adjoint +=
        output_adjoint[layout.uncheckedGradientOffset(0, lane, element)] *
        second * input[layout.uncheckedGradientOffset(0, lane, element)];
  for (std::size_t electron = 0; electron < layout.laplacian_lanes; ++electron)
  {
    double gradient_norm = 0.0;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const std::size_t lane = 3 * electron + dimension;
      const double gradient = input[layout.uncheckedGradientOffset(0, lane, element)];
      gradient_norm += gradient * gradient;
    }
    value_adjoint +=
        output_adjoint[layout.uncheckedLaplacianOffset(0, electron, element)] *
        (second * input[layout.uncheckedLaplacianOffset(0, electron, element)] +
         third * gradient_norm);
  }
  input_adjoint[layout.uncheckedValueOffset(0, element)] += value_adjoint;

  for (std::size_t lane = 0; lane < layout.gradient_lanes; ++lane)
  {
    const std::size_t electron = lane / 3;
    input_adjoint[layout.uncheckedGradientOffset(0, lane, element)] +=
        output_adjoint[layout.uncheckedGradientOffset(0, lane, element)] * first +
        2.0 * output_adjoint[layout.uncheckedLaplacianOffset(
                  0, electron, element)] *
            second * input[layout.uncheckedGradientOffset(0, lane, element)];
  }
  for (std::size_t electron = 0; electron < layout.laplacian_lanes; ++electron)
    input_adjoint[layout.uncheckedLaplacianOffset(0, electron, element)] +=
        output_adjoint[layout.uncheckedLaplacianOffset(0, electron, element)] * first;
}

/** Reverse one full-VGL softmax row without storing a third-derivative tensor. */
QMC_PF_SPATIAL_REVERSE_HOST_DEVICE inline void softmaxJetRowReverse(
    const double* logits_row,
    const double* probabilities_row,
    const double* probability_adjoint_row,
    std::size_t width,
    std::size_t plane_stride,
    std::size_t gradient_lanes,
    std::size_t electron_count,
    double* logit_adjoint_row) noexcept
{
  const double value_mean = detail::weightedMean(
      probabilities_row, probability_adjoint_row, width);
  for (std::size_t differentiated = 0; differentiated < width; ++differentiated)
  {
    const double probability = probabilities_row[differentiated];
    double value_result = probability *
        (probability_adjoint_row[differentiated] - value_mean);

    for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
    {
      const double* logit_gradient = logits_row + (1 + lane) * plane_stride;
      const double* probability_gradient_adjoint =
          probability_adjoint_row + (1 + lane) * plane_stride;
      const double gradient_mean = detail::weightedMean(
          probabilities_row, logit_gradient, width);
      const double adjoint_mean = detail::weightedMean(
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
      const double laplacian_adjoint_mean = detail::weightedMean(
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

    for (std::size_t electron = 0; electron < electron_count; ++electron)
    {
      const double* logit_laplacian =
          logits_row + (1 + gradient_lanes + electron) * plane_stride;
      const double* laplacian_adjoint = probability_adjoint_row +
          (1 + gradient_lanes + electron) * plane_stride;
      const double laplacian_mean = detail::weightedMean(
          probabilities_row, logit_laplacian, width);
      const double adjoint_mean = detail::weightedMean(
          probabilities_row, laplacian_adjoint, width);

      double gradient_square_mean = 0.0;
      for (std::size_t index = 0; index < width; ++index)
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
        {
          const std::size_t lane = 3 * electron + dimension;
          const double* gradient = logits_row + (1 + lane) * plane_stride;
          const double gradient_mean = detail::weightedMean(
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
        const double gradient_mean = detail::weightedMean(
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
          const double gradient_mean = detail::weightedMean(
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
}

inline void elementwiseProductJetsReverseHost(
    const SpatialElementwiseJetLayout& layout,
    const double* left,
    const double* right,
    const double* output_adjoint,
    double scale,
    double* left_adjoint,
    double* right_adjoint)
{
  validateSpatialElementwiseJetLayout(layout);
  validateFullVglReverseLayout(layout.jets);
  if (!left || !right || !output_adjoint || !left_adjoint || !right_adjoint)
    throw std::invalid_argument("PsiFormer product-jet reverse storage is null");
  if (left_adjoint == right_adjoint || left_adjoint == left ||
      left_adjoint == right || left_adjoint == output_adjoint ||
      right_adjoint == left || right_adjoint == right ||
      right_adjoint == output_adjoint)
    throw std::invalid_argument(
        "PsiFormer product-jet reverse input/output storage aliases");
  for (std::size_t row = 0; row < layout.row_count; ++row)
    for (std::size_t column = 0; column < layout.row_width; ++column)
    {
      const std::size_t element = row * layout.row_stride + column;
      productJetReverseElement(layout.jets, left, element, layout.jets, right,
                               element, layout.jets, output_adjoint, element,
                               scale, left_adjoint, right_adjoint);
    }
}

inline void tanhJetsReverseHost(const SpatialElementwiseJetLayout& layout,
                                const double* input,
                                const double* tanh_output,
                                const double* output_adjoint,
                                double* input_adjoint)
{
  validateSpatialElementwiseJetLayout(layout);
  validateFullVglReverseLayout(layout.jets);
  if (!input || !tanh_output || !output_adjoint || !input_adjoint)
    throw std::invalid_argument("PsiFormer tanh-jet reverse storage is null");
  if (input_adjoint == input || input_adjoint == tanh_output ||
      input_adjoint == output_adjoint)
    throw std::invalid_argument(
        "PsiFormer tanh-jet reverse input/output storage aliases");
  for (std::size_t row = 0; row < layout.row_count; ++row)
    for (std::size_t column = 0; column < layout.row_width; ++column)
    {
      const std::size_t element = row * layout.row_stride + column;
      tanhJetReverseElement(layout.jets, input, tanh_output, output_adjoint,
                            element, input_adjoint);
    }
}

inline void denseJetsReverseHost(const SpatialDenseJetLayout& layout,
                                 const double* source,
                                 const double* weight,
                                 const double* target_adjoint,
                                 double* source_adjoint,
                                 double* weight_adjoint,
                                 double* bias_adjoint)
{
  validateSpatialDenseJetLayout(layout);
  validateMatchingFullVglReverseLayouts(layout.source, layout.target, layout.target);
  if (!source || !weight || !target_adjoint || !source_adjoint ||
      !weight_adjoint || !bias_adjoint)
    throw std::invalid_argument("PsiFormer dense-jet reverse storage is null");
  if (source_adjoint == weight_adjoint || source_adjoint == bias_adjoint ||
      weight_adjoint == bias_adjoint || source_adjoint == source ||
      source_adjoint == weight || source_adjoint == target_adjoint ||
      weight_adjoint == source || weight_adjoint == weight ||
      weight_adjoint == target_adjoint || bias_adjoint == source ||
      bias_adjoint == weight || bias_adjoint == target_adjoint)
    throw std::invalid_argument(
        "PsiFormer dense-jet reverse input/output storage aliases");
  for (std::size_t plane = 0; plane < layout.source.plane_count; ++plane)
    for (std::size_t row = 0; row < layout.rows; ++row)
      for (std::size_t output = 0; output < layout.output_width; ++output)
      {
        const double upstream = target_adjoint[layout.target.uncheckedPlaneOffset(
            0, plane, row * layout.target_row_stride + output)];
        for (std::size_t input = 0; input < layout.input_width; ++input)
        {
          source_adjoint[layout.source.uncheckedPlaneOffset(
              0, plane, row * layout.source_row_stride + input)] +=
              upstream * weight[input * layout.weight_row_stride + output];
          weight_adjoint[input * layout.weight_row_stride + output] +=
              source[layout.source.uncheckedPlaneOffset(
                  0, plane, row * layout.source_row_stride + input)] * upstream;
        }
        if (plane == 0)
          bias_adjoint[output] += upstream;
      }
}

inline void softmaxJetRowsReverseHost(const SoftmaxJetRowLayout& layout,
                                      const double* logits,
                                      const double* probabilities,
                                      const double* probability_adjoint,
                                      double* logit_adjoint)
{
  validateSoftmaxJetRowLayout(layout);
  validateFullVglReverseLayout(layout.jets);
  if (!logits || !probabilities || !probability_adjoint || !logit_adjoint)
    throw std::invalid_argument("PsiFormer softmax-jet reverse storage is null");
  if (logit_adjoint == logits || logit_adjoint == probabilities ||
      logit_adjoint == probability_adjoint)
    throw std::invalid_argument(
        "PsiFormer softmax-jet reverse input/output storage aliases");
  for (std::size_t row = 0; row < layout.row_count; ++row)
  {
    const std::size_t group = row / layout.rows_per_group;
    const std::size_t row_in_group = row % layout.rows_per_group;
    const std::size_t offset = group * layout.row_group_stride +
        row_in_group * layout.row_stride;
    softmaxJetRowReverse(logits + offset, probabilities + offset,
                         probability_adjoint + offset, layout.row_width,
                         layout.jets.plane_stride, layout.jets.gradient_lanes,
                         layout.jets.laplacian_lanes, logit_adjoint + offset);
  }
}

inline void attentionContextJetsReverseHost(
    const SpatialAttentionJetLayout& layout,
    const double* attention,
    const double* value,
    const double* context_adjoint,
    double* attention_adjoint,
    double* value_adjoint)
{
  validateSpatialAttentionJetLayout(layout);
  validateMatchingFullVglReverseLayouts(
      layout.attention, layout.features, layout.features);
  if (!attention || !value || !context_adjoint || !attention_adjoint ||
      !value_adjoint)
    throw std::invalid_argument("PsiFormer attention-context reverse storage is null");
  if (attention_adjoint == value_adjoint || attention_adjoint == attention ||
      attention_adjoint == value || attention_adjoint == context_adjoint ||
      value_adjoint == attention || value_adjoint == value ||
      value_adjoint == context_adjoint)
    throw std::invalid_argument(
        "PsiFormer attention-context reverse input/output storage aliases");
  for (std::size_t head = 0; head < layout.heads; ++head)
    for (std::size_t query = 0; query < layout.rows; ++query)
      for (std::size_t key = 0; key < layout.rows; ++key)
      {
        const std::size_t attention_element =
            spatial_jet::attentionElement(layout, head, query, key);
        for (std::size_t plane = 0; plane < layout.attention.plane_count; ++plane)
        {
          double contribution = 0.0;
          for (std::size_t feature = 0; feature < layout.head_width; ++feature)
            contribution += detail::productLeftContribution(
                layout.attention, attention, attention_element, layout.features,
                value, spatial_jet::featureElement(layout, key, head, feature),
                layout.features, context_adjoint,
                spatial_jet::featureElement(layout, query, head, feature), plane, 1.0);
          attention_adjoint[layout.attention.uncheckedPlaneOffset(
              0, plane, attention_element)] += contribution;
        }
      }
  for (std::size_t key = 0; key < layout.rows; ++key)
    for (std::size_t head = 0; head < layout.heads; ++head)
      for (std::size_t feature = 0; feature < layout.head_width; ++feature)
      {
        const std::size_t value_element =
            spatial_jet::featureElement(layout, key, head, feature);
        for (std::size_t plane = 0; plane < layout.features.plane_count; ++plane)
        {
          double contribution = 0.0;
          for (std::size_t query = 0; query < layout.rows; ++query)
            contribution += detail::productLeftContribution(
                layout.features, value, value_element, layout.attention,
                attention, spatial_jet::attentionElement(layout, head, query, key),
                layout.features, context_adjoint,
                spatial_jet::featureElement(layout, query, head, feature), plane, 1.0);
          value_adjoint[layout.features.uncheckedPlaneOffset(
              0, plane, value_element)] += contribution;
        }
      }
}

inline void attentionLogitJetsReverseHost(
    const SpatialAttentionJetLayout& layout,
    const double* query,
    const double* key,
    const double* logit_adjoint,
    double* query_adjoint,
    double* key_adjoint)
{
  validateSpatialAttentionJetLayout(layout);
  validateMatchingFullVglReverseLayouts(
      layout.features, layout.features, layout.attention);
  if (!query || !key || !logit_adjoint || !query_adjoint || !key_adjoint)
    throw std::invalid_argument("PsiFormer attention-logit reverse storage is null");
  if (query_adjoint == key_adjoint || query_adjoint == query ||
      query_adjoint == key || query_adjoint == logit_adjoint ||
      key_adjoint == query || key_adjoint == key || key_adjoint == logit_adjoint)
    throw std::invalid_argument(
        "PsiFormer attention-logit reverse input/output storage aliases");
  const double scale = 1.0 / ::sqrt(static_cast<double>(layout.head_width));
  for (std::size_t row = 0; row < layout.rows; ++row)
    for (std::size_t head = 0; head < layout.heads; ++head)
      for (std::size_t feature = 0; feature < layout.head_width; ++feature)
      {
        const std::size_t query_element =
            spatial_jet::featureElement(layout, row, head, feature);
        for (std::size_t plane = 0; plane < layout.features.plane_count; ++plane)
        {
          double contribution = 0.0;
          for (std::size_t key_row = 0; key_row < layout.rows; ++key_row)
            contribution += detail::productLeftContribution(
                layout.features, query, query_element, layout.features, key,
                spatial_jet::featureElement(layout, key_row, head, feature),
                layout.attention, logit_adjoint,
                spatial_jet::attentionElement(layout, head, row, key_row), plane,
                scale);
          query_adjoint[layout.features.uncheckedPlaneOffset(
              0, plane, query_element)] += contribution;
        }

        const std::size_t key_element = query_element;
        for (std::size_t plane = 0; plane < layout.features.plane_count; ++plane)
        {
          double contribution = 0.0;
          for (std::size_t query_row = 0; query_row < layout.rows; ++query_row)
            contribution += detail::productLeftContribution(
                layout.features, key, key_element, layout.features, query,
                spatial_jet::featureElement(layout, query_row, head, feature),
                layout.attention, logit_adjoint,
                spatial_jet::attentionElement(layout, head, query_row, row), plane,
                scale);
          key_adjoint[layout.features.uncheckedPlaneOffset(
              0, plane, key_element)] += contribution;
        }
      }
}

} // namespace qmcplusplus::psiformer::spatial_jet_reverse

#undef QMC_PF_SPATIAL_REVERSE_HOST_DEVICE

#endif // QMCPLUSPLUS_PSIFORMER_SPATIAL_JET_REVERSE_H
