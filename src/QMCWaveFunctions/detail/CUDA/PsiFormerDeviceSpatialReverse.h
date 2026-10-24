//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceSpatialReverse.h
 * @brief CUDA/HIP wrappers for deterministic full-VGL spatial-jet reverse leaves.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_SPATIAL_REVERSE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_SPATIAL_REVERSE_H

#include "PsiFormerDeviceDense.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOpenSpatial.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialJetReverse.h"
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace qmcplusplus::psiformer::device
{

/// Select wide-row column work automatically; explicit 0/1 always overrides.
inline bool configuredSoftmaxColumnReverse(std::size_t row_width = 0)
{
  const char* value = std::getenv("PSIFORMER_GPU_SOFTMAX_COLUMN_REVERSE");
  if (!value)
    return row_width >= 64;
  if (std::string_view(value) == "0")
    return false;
  if (std::string_view(value) == "1")
    return true;
  throw std::invalid_argument("PSIFORMER_GPU_SOFTMAX_COLUMN_REVERSE requires 0 or 1");
}

/** Reverse a one-configuration full-VGL dense jet.
 *
 * All destinations are accumulated. Bias is a value-plane parameter and hence
 * receives only the value-plane row reduction. Padding is never modified.
 */
void denseJetsReverse(AcceleratorBlasHandle& handle,
                      Stream stream,
                      const SpatialDenseJetLayout& layout,
                      const double* source,
                      const double* weight,
                      const double* target_adjoint,
                      double* source_adjoint,
                      double* weight_adjoint,
                      double* bias_adjoint);

Error launchElementwiseProductJetReverse(
    Stream stream,
    const SpatialElementwiseJetLayout& layout,
    const double* left,
    const double* right,
    const double* output_adjoint,
    double scale,
    double* left_adjoint,
    double* right_adjoint);

Error launchTanhJetReverse(Stream stream,
                           const SpatialElementwiseJetLayout& layout,
                           const double* input,
                           const double* tanh_output,
                           const double* output_adjoint,
                           double* input_adjoint);

Error launchSoftmaxJetReverse(Stream stream,
                              const SoftmaxJetRowLayout& layout,
                              const double* logits,
                              const double* probabilities,
                              const double* probability_adjoint,
                              double* logit_adjoint,
                              bool differentiated_columns);

/// Keep explicit bool oracle calls separate from width-aware production selection.
inline Error launchSoftmaxJetReverse(Stream stream,
                                     const SoftmaxJetRowLayout& layout,
                                     const double* logits,
                                     const double* probabilities,
                                     const double* probability_adjoint,
                                     double* logit_adjoint)
{
  return launchSoftmaxJetReverse(stream, layout, logits, probabilities,
      probability_adjoint, logit_adjoint, configuredSoftmaxColumnReverse(layout.row_width));
}

Error launchAttentionContextJetReverse(
    Stream stream,
    const SpatialAttentionJetLayout& layout,
    const double* attention,
    const double* value,
    const double* context_adjoint,
    double* attention_adjoint,
    double* value_adjoint);

Error launchAttentionLogitJetReverse(
    Stream stream,
    const SpatialAttentionJetLayout& layout,
    const double* query,
    const double* key,
    const double* logit_adjoint,
    double* query_adjoint,
    double* key_adjoint);

/// Accumulate one configuration's FULL_VGL value/gradient/Laplacian roots.
Error launchPeriodicPairAttentionJetParameterReverse(Stream stream,
    const SpatialAttentionJetLayout& layout, const periodic_geometry::Cell& cell,
    const double* positions, std::size_t spin_up_count, const double* logit_adjoint,
    double* weight_response, double* bias_response);

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_SPATIAL_REVERSE_H
