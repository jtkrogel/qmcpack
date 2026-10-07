//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceSpatial.h
 * @brief CUDA/HIP entry points for the PsiFormer spatial-jet foundation.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_SPATIAL_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_SPATIAL_H

#include "PsiFormerDeviceDense.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialLayout.h"

namespace qmcplusplus::psiformer::device
{

/** Transform batched spatial-logit rows to softmax jets in place. */
Error launchSoftmaxJetRows(Stream stream,
                           const SoftmaxJetRowLayout& layout,
                           double* jets,
                           device_math::JetMathStatus* row_status);

/** Apply one dense weight matrix independently to every B*plane slice. */
void denseJetsForward(AcceleratorBlasHandle& handle,
                      const SpatialDenseJetLayout& layout,
                      const double* source,
                      const double* weight,
                      double* target);

/** Apply three dense projections independently to every B*plane slice. */
void projectQkvJetsForward(AcceleratorBlasHandle& handle,
                           const SpatialDenseJetLayout& layout,
                           const double* source,
                           const double* query_weight,
                           const double* key_weight,
                           const double* value_weight,
                           double* query,
                           double* key,
                           double* value);

/** Form Q*K^T jets and then transform every head/query row with jet softmax. */
Error launchAttentionJetWeights(
    Stream stream,
    const SpatialAttentionJetLayout& layout,
    const double* query,
    const double* key,
    double* attention,
    device_math::JetMathStatus* row_status);

/** Form context jets from prepared attention probabilities and value jets. */
Error launchAttentionContextJets(Stream stream,
                                 const SpatialAttentionJetLayout& layout,
                                 const double* attention,
                                 const double* value,
                                 double* context);

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_SPATIAL_H
