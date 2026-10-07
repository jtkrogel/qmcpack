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
#include "PsiFormerDevicePrecision.h"
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
  const bool open_feature_linked =
      launchOpenFeatureJets(nullptr, qmcplusplus::psiformer::OpenFeatureJetLayout{},
                            nullptr, nullptr, nullptr, nullptr, nullptr) == success;
  const bool tanh_jet_linked =
      launchBiasTanhJets(nullptr, qmcplusplus::psiformer::SpatialElementwiseJetLayout{},
                         nullptr, nullptr, nullptr, nullptr) == success;
  const bool residual_jet_linked =
      launchResidualSpatialJets(nullptr, qmcplusplus::psiformer::SpatialElementwiseJetLayout{},
                                nullptr, nullptr, nullptr, nullptr) == success;
  const bool cusp_jet_linked =
      launchOpenCuspJets(nullptr, qmcplusplus::psiformer::OpenCuspJetLayout{},
                         nullptr, nullptr, 0.0, 0.0, nullptr, nullptr) == success;
  const bool final_spatial_linked =
      launchFinalSpatialCombination(nullptr, qmcplusplus::psiformer::FinalSpatialJetLayout{},
                                    nullptr, nullptr, nullptr, nullptr, nullptr,
                                    nullptr, nullptr, nullptr, nullptr) == success;
  const bool open_orbital_linked =
      launchOpenOrbitalJets(nullptr, qmcplusplus::psiformer::OpenOrbitalJetLayout{},
                            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == success;
  const bool precision_conversion_linked =
      launchFp64ToFp32ParameterConversion(
          nullptr, qmcplusplus::psiformer::PsiFormerParameterConversionTile{},
          nullptr, 0, nullptr, 0, nullptr) == success;
  const bool fp32_dense_linked   = &denseForwardFp32 != nullptr;
  const bool fp32_qkv_linked     = &projectQkvForwardFp32 != nullptr;
  const bool fp32_logits_linked  = &attentionLogitsForwardFp32 != nullptr;
  const bool fp32_context_linked = &attentionContextForwardFp32 != nullptr;
  const bool fp32_softmax_linked =
      launchAttentionSoftmaxFp32(
          nullptr, qmcplusplus::psiformer::BatchedAttentionForwardLayout{},
          nullptr, nullptr) == success;
  const bool fp32_tanh_linked =
      launchBiasTanhValueFp32(
          nullptr, qmcplusplus::psiformer::BatchedValueLayout{},
          nullptr, nullptr, nullptr, nullptr) == success;
  const bool fp32_residual_linked =
      launchResidualValueFp32(
          nullptr, qmcplusplus::psiformer::BatchedValueLayout{},
          nullptr, nullptr, nullptr) == success;
  const bool fp32_to_fp64_linked =
      launchValueFp32ToFp64(
          nullptr, qmcplusplus::psiformer::BatchedValueLayout{},
          nullptr, nullptr, nullptr) == success;
  return foundation_linked && attention_linked && determinant_linked &&
         determinant_combination_linked && determinant_reverse_linked &&
         determinant_spatial_linked && spatial_softmax_linked &&
         spatial_dense_linked && spatial_qkv_linked &&
         spatial_attention_linked && spatial_context_linked && open_feature_linked &&
         tanh_jet_linked && residual_jet_linked && cusp_jet_linked &&
         final_spatial_linked && open_orbital_linked && precision_conversion_linked &&
         fp32_dense_linked && fp32_qkv_linked && fp32_logits_linked &&
         fp32_context_linked && fp32_softmax_linked && fp32_tanh_linked &&
         fp32_residual_linked && fp32_to_fp64_linked
      ? 0
      : 1;
}
