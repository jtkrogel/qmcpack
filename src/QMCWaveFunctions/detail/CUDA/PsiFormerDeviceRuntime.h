//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceRuntime.h
 * @brief Narrow CUDA/HIP runtime type adapter for PsiFormer kernel wrappers.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_RUNTIME_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_RUNTIME_H

#include "config.h"

#ifdef QMC_CUDA2HIP
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif

namespace qmcplusplus::psiformer::device
{

#ifdef QMC_CUDA2HIP
using Stream = hipStream_t;
using Error  = hipError_t;
inline constexpr Error success = hipSuccess;
#else
using Stream = cudaStream_t;
using Error  = cudaError_t;
inline constexpr Error success = cudaSuccess;
#endif

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_RUNTIME_H
