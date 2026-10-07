//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceLinkCheck.cu
 * @brief Driverless developer target proving that PsiFormer wrappers device-link.
 */

#include "PsiFormerDeviceKernels.h"
#include "PsiFormerDeviceDense.h"
#include "PsiFormerDeviceDeterminantKernels.h"
#include "PsiFormerDeviceSpatial.h"

int main()
{
  using namespace qmcplusplus::psiformer::device;
  // Zero work performs no runtime call but keeps the wrapper translation unit in the
  // executable, forcing CUDA nvlink or the HIP device-link step to inspect it.
  const bool foundation_linked =
      launchResidualJets(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, 0, nullptr, nullptr, nullptr) == success;
  const bool attention_linked =
      launchAttentionSoftmax(nullptr, qmcplusplus::psiformer::AttentionForwardLayout{}, nullptr) == success;
  const bool determinant_linked =
      launchDeterminantFactorization(nullptr, nullptr, 0, 0, 0, false,
                                     nullptr, nullptr, nullptr, nullptr, nullptr) == success;
  const bool determinant_combination_linked =
      launchDeterminantCombination(nullptr, nullptr, nullptr, 0, 0,
                                   nullptr, nullptr, nullptr, nullptr, nullptr) == success;
  const bool determinant_reverse_linked =
      launchDeterminantMatrixReverseSeeds(nullptr, nullptr, nullptr, nullptr, nullptr,
                                          0, 0, 0, nullptr, nullptr) == success;
  const bool determinant_spatial_linked =
      launchDeterminantSpatialTraces(
          nullptr, qmcplusplus::psiformer::device_determinant::SpatialLayout{},
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr) == success;
  const bool spatial_softmax_linked =
      launchSoftmaxJetRows(nullptr, qmcplusplus::psiformer::SoftmaxJetRowLayout{},
                           nullptr, nullptr) == success;
  const bool spatial_dense_linked = &denseJetsForward != nullptr;
  const bool spatial_qkv_linked   = &projectQkvJetsForward != nullptr;
  const bool spatial_attention_linked =
      launchAttentionJetWeights(
          nullptr, qmcplusplus::psiformer::SpatialAttentionJetLayout{},
          nullptr, nullptr, nullptr, nullptr) == success;
  const bool spatial_context_linked =
      launchAttentionContextJets(
          nullptr, qmcplusplus::psiformer::SpatialAttentionJetLayout{},
          nullptr, nullptr, nullptr) == success;
  return foundation_linked && attention_linked && determinant_linked &&
         determinant_combination_linked && determinant_reverse_linked &&
         determinant_spatial_linked && spatial_softmax_linked &&
         spatial_dense_linked && spatial_qkv_linked &&
         spatial_attention_linked && spatial_context_linked
      ? 0
      : 1;
}
