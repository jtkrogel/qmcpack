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

#include "PsiFormerDeviceRuntime.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialLayout.h"

namespace qmcplusplus::psiformer::device
{

/** Transform batched spatial-logit rows to softmax jets in place. */
Error launchSoftmaxJetRows(Stream stream,
                           const SoftmaxJetRowLayout& layout,
                           double* jets,
                           device_math::JetMathStatus* row_status);

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_SPATIAL_H
