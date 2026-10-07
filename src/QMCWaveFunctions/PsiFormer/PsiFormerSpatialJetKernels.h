//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerSpatialJetKernels.h
 * @brief Allocation-free host/device algebra for spatial dense-attention jets.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_SPATIAL_JET_KERNELS_H
#define QMCPLUSPLUS_PSIFORMER_SPATIAL_JET_KERNELS_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialLayout.h"

#include <cmath>
#include <cstddef>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QMC_PF_SPATIAL_HOST_DEVICE __host__ __device__
#else
#define QMC_PF_SPATIAL_HOST_DEVICE
#endif

namespace qmcplusplus::psiformer::spatial_jet
{

QMC_PF_SPATIAL_HOST_DEVICE inline std::size_t featureElement(
    const SpatialAttentionJetLayout& layout,
    std::size_t row,
    std::size_t head,
    std::size_t feature) noexcept
{
  return row * layout.feature_row_stride + head * layout.head_width + feature;
}

QMC_PF_SPATIAL_HOST_DEVICE inline std::size_t attentionElement(
    const SpatialAttentionJetLayout& layout,
    std::size_t head,
    std::size_t query,
    std::size_t key) noexcept
{
  return head * layout.attention_head_stride + query * layout.attention_row_stride + key;
}

/** Form one Q*K^T attention logit jet, including every trace cross term. */
QMC_PF_SPATIAL_HOST_DEVICE inline void attentionLogitJetElement(
    const SpatialAttentionJetLayout& layout,
    const double* query,
    const double* key,
    std::size_t configuration,
    std::size_t head,
    std::size_t query_row,
    std::size_t key_row,
    double* logits) noexcept
{
  const std::size_t output_element =
      attentionElement(layout, head, query_row, key_row);
  const std::size_t query_configuration =
      configuration * layout.features.configuration_stride;
  const std::size_t output_configuration =
      configuration * layout.attention.configuration_stride;
  const double scale = 1.0 / ::sqrt(static_cast<double>(layout.head_width));
  double value = 0.0;
  for (std::size_t feature = 0; feature < layout.head_width; ++feature)
  {
    const std::size_t query_element =
        featureElement(layout, query_row, head, feature);
    const std::size_t key_element = featureElement(layout, key_row, head, feature);
    value += scale * query[query_configuration + query_element] *
        key[query_configuration + key_element];
  }
  logits[output_configuration + output_element] = value;

  for (std::size_t lane = 0; lane < layout.features.gradient_lanes; ++lane)
  {
    double first = 0.0;
    const std::size_t query_plane = query_configuration +
        (1 + lane) * layout.features.plane_stride;
    const std::size_t output_plane = output_configuration +
        (1 + lane) * layout.attention.plane_stride;
    for (std::size_t feature = 0; feature < layout.head_width; ++feature)
    {
      const std::size_t query_element =
          featureElement(layout, query_row, head, feature);
      const std::size_t key_element = featureElement(layout, key_row, head, feature);
      first += scale *
          (query[query_plane + query_element] * key[query_configuration + key_element] +
           query[query_configuration + query_element] * key[query_plane + key_element]);
    }
    logits[output_plane + output_element] = first;
  }

  for (std::size_t electron = 0;
       electron < layout.features.laplacian_lanes; ++electron)
  {
    double trace_second = 0.0;
    const std::size_t query_laplacian_plane = query_configuration +
        (1 + layout.features.gradient_lanes + electron) *
            layout.features.plane_stride;
    const std::size_t output_laplacian_plane = output_configuration +
        (1 + layout.attention.gradient_lanes + electron) *
            layout.attention.plane_stride;
    for (std::size_t feature = 0; feature < layout.head_width; ++feature)
    {
      const std::size_t query_element =
          featureElement(layout, query_row, head, feature);
      const std::size_t key_element = featureElement(layout, key_row, head, feature);
      double gradient_dot = 0.0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const std::size_t gradient_plane = query_configuration +
            (1 + 3 * electron + dimension) * layout.features.plane_stride;
        gradient_dot += query[gradient_plane + query_element] *
            key[gradient_plane + key_element];
      }
      trace_second += scale *
          (query[query_laplacian_plane + query_element] *
               key[query_configuration + key_element] +
           2.0 * gradient_dot + query[query_configuration + query_element] *
               key[query_laplacian_plane + key_element]);
    }
    logits[output_laplacian_plane + output_element] = trace_second;
  }
}

/** Form one attention-context feature jet from probability and value jets. */
QMC_PF_SPATIAL_HOST_DEVICE inline void attentionContextJetElement(
    const SpatialAttentionJetLayout& layout,
    const double* probability,
    const double* value,
    std::size_t configuration,
    std::size_t output_row,
    std::size_t head,
    std::size_t feature,
    double* context) noexcept
{
  const std::size_t output_element =
      featureElement(layout, output_row, head, feature);
  const std::size_t feature_configuration =
      configuration * layout.features.configuration_stride;
  const std::size_t attention_configuration =
      configuration * layout.attention.configuration_stride;
  double output_value = 0.0;
  for (std::size_t input_row = 0; input_row < layout.rows; ++input_row)
  {
    const std::size_t probability_element =
        attentionElement(layout, head, output_row, input_row);
    const std::size_t value_element = featureElement(layout, input_row, head, feature);
    output_value += probability[attention_configuration + probability_element] *
        value[feature_configuration + value_element];
  }
  context[feature_configuration + output_element] = output_value;

  for (std::size_t lane = 0; lane < layout.features.gradient_lanes; ++lane)
  {
    double first = 0.0;
    const std::size_t feature_plane = feature_configuration +
        (1 + lane) * layout.features.plane_stride;
    const std::size_t attention_plane = attention_configuration +
        (1 + lane) * layout.attention.plane_stride;
    for (std::size_t input_row = 0; input_row < layout.rows; ++input_row)
    {
      const std::size_t probability_element =
          attentionElement(layout, head, output_row, input_row);
      const std::size_t value_element = featureElement(layout, input_row, head, feature);
      first += probability[attention_plane + probability_element] *
              value[feature_configuration + value_element] +
          probability[attention_configuration + probability_element] *
              value[feature_plane + value_element];
    }
    context[feature_plane + output_element] = first;
  }

  for (std::size_t electron = 0;
       electron < layout.features.laplacian_lanes; ++electron)
  {
    double trace_second = 0.0;
    const std::size_t feature_laplacian_plane = feature_configuration +
        (1 + layout.features.gradient_lanes + electron) *
            layout.features.plane_stride;
    const std::size_t attention_laplacian_plane = attention_configuration +
        (1 + layout.attention.gradient_lanes + electron) *
            layout.attention.plane_stride;
    for (std::size_t input_row = 0; input_row < layout.rows; ++input_row)
    {
      const std::size_t probability_element =
          attentionElement(layout, head, output_row, input_row);
      const std::size_t value_element = featureElement(layout, input_row, head, feature);
      double gradient_dot = 0.0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const std::size_t feature_gradient_plane = feature_configuration +
            (1 + 3 * electron + dimension) * layout.features.plane_stride;
        const std::size_t attention_gradient_plane = attention_configuration +
            (1 + 3 * electron + dimension) * layout.attention.plane_stride;
        gradient_dot += probability[attention_gradient_plane + probability_element] *
            value[feature_gradient_plane + value_element];
      }
      trace_second +=
          probability[attention_laplacian_plane + probability_element] *
              value[feature_configuration + value_element] +
          2.0 * gradient_dot +
          probability[attention_configuration + probability_element] *
              value[feature_laplacian_plane + value_element];
    }
    context[feature_laplacian_plane + output_element] = trace_second;
  }
}

inline void denseJetsHost(const SpatialDenseJetLayout& layout,
                          const double* source,
                          const double* weight,
                          double* target)
{
  validateSpatialDenseJetLayout(layout);
  for (std::size_t configuration = 0;
       configuration < layout.source.configuration_count; ++configuration)
    for (std::size_t plane = 0; plane < layout.source.plane_count; ++plane)
      for (std::size_t row = 0; row < layout.rows; ++row)
        for (std::size_t output = 0; output < layout.output_width; ++output)
        {
          double sum = 0.0;
          for (std::size_t input = 0; input < layout.input_width; ++input)
            sum += source[layout.source.uncheckedPlaneOffset(
                       configuration, plane, row * layout.source_row_stride + input)] *
                weight[input * layout.weight_row_stride + output];
          target[layout.target.uncheckedPlaneOffset(
              configuration, plane, row * layout.target_row_stride + output)] = sum;
        }
}

inline void projectQkvJetsHost(const SpatialDenseJetLayout& layout,
                               const double* source,
                               const double* query_weight,
                               const double* key_weight,
                               const double* value_weight,
                               double* query,
                               double* key,
                               double* value)
{
  denseJetsHost(layout, source, query_weight, query);
  denseJetsHost(layout, source, key_weight, key);
  denseJetsHost(layout, source, value_weight, value);
}

inline void attentionLogitJetsHost(const SpatialAttentionJetLayout& layout,
                                   const double* query,
                                   const double* key,
                                   double* logits)
{
  validateSpatialAttentionJetLayout(layout);
  for (std::size_t configuration = 0;
       configuration < layout.features.configuration_count; ++configuration)
    for (std::size_t head = 0; head < layout.heads; ++head)
      for (std::size_t query_row = 0; query_row < layout.rows; ++query_row)
        for (std::size_t key_row = 0; key_row < layout.rows; ++key_row)
          attentionLogitJetElement(
              layout, query, key, configuration, head, query_row, key_row, logits);
}

inline void attentionContextJetsHost(const SpatialAttentionJetLayout& layout,
                                     const double* probability,
                                     const double* value,
                                     double* context)
{
  validateSpatialAttentionJetLayout(layout);
  for (std::size_t configuration = 0;
       configuration < layout.features.configuration_count; ++configuration)
    for (std::size_t output_row = 0; output_row < layout.rows; ++output_row)
      for (std::size_t head = 0; head < layout.heads; ++head)
        for (std::size_t feature = 0; feature < layout.head_width; ++feature)
          attentionContextJetElement(
              layout, probability, value, configuration, output_row, head,
              feature, context);
}

} // namespace qmcplusplus::psiformer::spatial_jet

#undef QMC_PF_SPATIAL_HOST_DEVICE

#endif // QMCPLUSPLUS_PSIFORMER_SPATIAL_JET_KERNELS_H
