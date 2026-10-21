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

/** VALUE-only cooperative alternative; no inverse/solve owners or alias expansion. */
Error launchDeterminantValueFactorizationCooperative(
    Stream stream, const double* matrices, std::size_t configuration_count,
    std::size_t determinant_count, std::size_t matrix_size, double* lu,
    std::size_t* permutation, device_determinant::FactorizationMetadata* metadata);

/** Checked wrapper-facing layout for lifted determinant matrix jets. */
struct LiftedDeterminantLayout
{
  SpatialJetLayout matrices;
  SpatialJetLayout roots;
  std::size_t determinant_count = 0;
  std::size_t matrix_size       = 0;
};

static_assert(std::is_standard_layout_v<LiftedDeterminantLayout>);
static_assert(std::is_trivially_copyable_v<LiftedDeterminantLayout>);

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

/** Pack strided VALUE matrices and reuse only exact finite, successful inverses.
 * The caller owns all buffers on one queue and permits reuse only within one
 * coordinate request. The first tile must pass false, including after failure.
 */
Error launchDeterminantTargetFactorization(
    Stream stream, const double* current_matrices, std::size_t configuration_count,
    std::size_t determinant_count, std::size_t matrix_size, std::size_t configuration_stride,
    bool reuse_previous, double* packed_matrices, double* lu, double* inverse,
    std::size_t* permutation, double* solve,
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

/** Evaluate packed derivative planes with optional configuration strides.
 * Gradient/Laplacian pointers may borrow orbital jet planes directly; zero
 * layout strides retain the canonical contiguous derivative-buffer ABI.
 */
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

/** Build lifted determinant factor jets, one serial thread per configuration. */
Error launchDeterminantFactorJets(
    Stream stream,
    const LiftedDeterminantLayout& layout,
    const device_determinant::FactorizationMetadata* factorization_metadata,
    const device_determinant::CombinationMetadata* combination_metadata,
    const double* normalized_weights,
    const double* inverses,
    const double* orbital_jets,
    const double* determinant_log_gradient,
    const double* determinant_log_laplacian,
    double* factor_jets,
    double* three_matrix_scratch,
    device_determinant::DerivativeStatus* status);

/** Reverse lifted root jets through determinant factor jets. */
Error launchDeterminantJetReverse(
    Stream stream,
    const LiftedDeterminantLayout& layout,
    const device_determinant::DerivativeStatus* factor_status,
    const double* factor_jets,
    const double* root_jets,
    double* orbital_adjoint_jets,
    device_determinant::DerivativeStatus* status);

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_DETERMINANT_KERNELS_H
