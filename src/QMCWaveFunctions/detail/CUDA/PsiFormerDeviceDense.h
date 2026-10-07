//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceDense.h
 * @brief Shared CUDA/HIP BLAS and softmax entry points for PsiFormer attention.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_DENSE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_DENSE_H

#include "PsiFormerDeviceRuntime.h"
#include "Platforms/Common/AccelBLASHandle.hpp"
#include "QMCWaveFunctions/PsiFormer/PsiFormerAttention.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPrecisionExecution.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPrecisionPolicy.h"

namespace qmcplusplus::psiformer::device
{

using AcceleratorBlasHandle = compute::BLASHandle<PlatformKind::CUDA>;

void denseForward(AcceleratorBlasHandle& handle,
                  const DenseForwardLayout& layout,
                  const double* source,
                  const double* weight,
                  double* target);

void projectQkvForward(AcceleratorBlasHandle& handle,
                       const DenseForwardLayout& layout,
                       const double* source,
                       const double* query_weight,
                       const double* key_weight,
                       const double* value_weight,
                       double* query,
                       double* key,
                       double* value);

void attentionLogitsForward(AcceleratorBlasHandle& handle,
                            const AttentionForwardLayout& layout,
                            const double* query,
                            const double* key,
                            double* logits);

Error launchAttentionSoftmax(Stream stream,
                             const AttentionForwardLayout& layout,
                             double* logits_and_weights);
// Device-side non-finite diagnostic publication is added with runtime orchestration;
// the host-callable specification rejects non-finite rows today.

void attentionContextForward(AcceleratorBlasHandle& handle,
                             const AttentionForwardLayout& layout,
                             const double* attention,
                             const double* value,
                             double* target);

/** Apply one strict-FP32 row-major dense projection.
 *
 * BLAS math-mode containment is added in Task 27.A4; this explicit ABI keeps the
 * storage type separate without changing the established FP64 entry point.
 */
void denseForwardFp32(AcceleratorBlasHandle& handle,
                      const PsiFormerBlasMathModePlan& math_mode,
                      const DenseForwardLayout& layout,
                      const float* source,
                      const float* weight,
                      float* target);

/// Apply three independent FP32 projections into Q, K, and V buffers.
void projectQkvForwardFp32(AcceleratorBlasHandle& handle,
                           const PsiFormerBlasMathModePlan& math_mode,
                           const DenseForwardLayout& layout,
                           const float* source,
                           const float* query_weight,
                           const float* key_weight,
                           const float* value_weight,
                           float* query,
                           float* key,
                           float* value);

/// Form configuration-local FP32 attention logits with FP32 GEMM accumulation.
void attentionLogitsForwardFp32(AcceleratorBlasHandle& handle,
                                const PsiFormerBlasMathModePlan& math_mode,
                                const BatchedAttentionForwardLayout& layout,
                                const float* query,
                                const float* key,
                                float* logits);

/** Normalize FP32 logits in place with FP64 maximum and sum reductions.
 *
 * Invalid rows are zeroed and reported through the bounded diagnostic record.
 */
Error launchAttentionSoftmaxFp32(Stream stream,
                                 const BatchedAttentionForwardLayout& layout,
                                 float* logits_and_weights,
                                 PsiFormerDeviceNumericalDiagnostics* diagnostics);

/// Contract FP32 attention weights with values independently per configuration.
void attentionContextForwardFp32(AcceleratorBlasHandle& handle,
                                 const PsiFormerBlasMathModePlan& math_mode,
                                 const BatchedAttentionForwardLayout& layout,
                                 const float* attention,
                                 const float* value,
                                 float* target);

/** Apply tanh(input+bias) to every logical FP32 value while preserving padding.
 *
 * Non-finite preactivations are counted before tanh can mask infinities and the
 * corresponding output is deterministically zeroed.
 */
Error launchBiasTanhValueFp32(Stream stream,
                              const BatchedValueLayout& layout,
                              const float* input,
                              const float* bias,
                              float* output,
                              PsiFormerDeviceNumericalDiagnostics* diagnostics);

/// Add two FP32 value tensors elementwise while preserving padding.
Error launchResidualValueFp32(Stream stream,
                              const BatchedValueLayout& layout,
                              const float* left,
                              const float* right,
                              float* output);

/** Cast padded logical FP64 feature/value storage into FP32 dense input.
 *
 * Non-finite values and finite magnitudes beyond ``FLT_MAX`` are counted in
 * execution diagnostics and deterministically written as zero.  Padding remains
 * untouched, forcing the later orchestration decision to retry the whole batch.
 */
Error launchValueFp64ToFp32(Stream stream,
                            const BatchedValueLayout& layout,
                            const double* source,
                            float* target,
                            PsiFormerDeviceNumericalDiagnostics* diagnostics);

/** Cross the audited ABI barrier from FP32 value storage back into FP64.
 *
 * Orbital assembly, determinants, spatial jets, and reductions consume only the
 * resulting double buffer; no FP32 overload is provided for those APIs.  Every
 * logical source value is checked here, and invalid values are counted and
 * deterministically promoted as zero so orchestration can reject/retry the epoch.
 */
Error launchValueFp32ToFp64(Stream stream,
                            const BatchedValueLayout& layout,
                            const float* source,
                            double* target,
                            PsiFormerDeviceNumericalDiagnostics* diagnostics);

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_DENSE_H
