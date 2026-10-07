//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceDeterminantKernels.h
 * @brief CUDA/HIP launch seam for the real PsiFormer determinant baseline.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_DETERMINANT_KERNELS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_DETERMINANT_KERNELS_H

#include "PsiFormerDeviceRuntime.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceDeterminantMath.h"

#include <cstddef>

namespace qmcplusplus::psiformer::device
{

Error launchDeterminantFactorization(
    Stream stream,
    const double* matrices,
    std::size_t configuration_count,
    std::size_t determinant_count,
    std::size_t matrix_size,
    bool prepare_inverse,
    double* lu,
    double* inverse,
    std::size_t* permutation,
    double* solve,
    device_determinant::FactorizationMetadata* metadata);

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_DETERMINANT_KERNELS_H
