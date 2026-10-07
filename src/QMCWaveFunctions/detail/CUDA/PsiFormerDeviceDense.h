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

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_DENSE_H
