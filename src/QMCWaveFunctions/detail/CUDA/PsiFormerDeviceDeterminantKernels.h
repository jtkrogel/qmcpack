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

/** Serial-per-configuration signed-log determinant-channel reduction. */
Error launchDeterminantCombination(
    Stream stream,
    const device_determinant::FactorizationMetadata* channel_metadata,
    const double* coefficients,
    std::size_t configuration_count,
    std::size_t determinant_count,
    double* term_phase,
    double* term_log_abs,
    double* scaled_terms,
    double* normalized_weights,
    device_determinant::CombinationMetadata* combination_metadata);

/** Fill [B,D,N,N] channel reverse seeds from prepared inverses and weights. */
Error launchDeterminantMatrixReverseSeeds(
    Stream stream,
    const device_determinant::FactorizationMetadata* factorization_metadata,
    const device_determinant::CombinationMetadata* combination_metadata,
    const double* normalized_weights,
    const double* inverses,
    std::size_t configuration_count,
    std::size_t determinant_count,
    std::size_t matrix_size,
    double* reverse_seeds,
    device_determinant::DerivativeStatus* status);

/** Evaluate canonical [B,lane,D,N,N] determinant spatial trace planes. */
Error launchDeterminantSpatialTraces(
    Stream stream,
    device_determinant::SpatialLayout layout,
    const device_determinant::FactorizationMetadata* factorization_metadata,
    const device_determinant::CombinationMetadata* combination_metadata,
    const double* normalized_weights,
    const double* inverses,
    const double* matrix_gradients,
    const double* matrix_laplacians,
    double* matrix_product_scratch,
    double* output_log_gradient,
    double* output_lap_ratio,
    double* output_lap_log,
    device_determinant::DerivativeStatus* status);

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_DETERMINANT_KERNELS_H
