//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDevicePrecision.h
 * @brief Portable CUDA/HIP wrappers for bounded PsiFormer precision conversion.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_PRECISION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_PRECISION_H

#include "PsiFormerDeviceRuntime.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPrecisionExecution.h"

#include <cstddef>

namespace qmcplusplus::psiformer::device
{

/** Convert one checked FP64 parameter tile into its matching FP32 compute slot.
 *
 * The diagnostic record is accumulated across calls and must be initialized by the
 * caller.  A zero-count tile performs no runtime call and permits null pointers for
 * offline device-link validation.
 */
Error launchFp64ToFp32ParameterConversion(
    Stream stream,
    const PsiFormerParameterConversionTile& tile,
    const double* source,
    std::size_t source_count,
    float* destination,
    std::size_t destination_count,
    PsiFormerParameterConversionDiagnostics* diagnostics);

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_PRECISION_H
