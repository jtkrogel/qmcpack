//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_VALUE_KERNELS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_VALUE_KERNELS_H
#include "PsiFormerDeviceSpatial.h"
#include <complex>
#include "Platforms/CUDA/CUDATypeMapping.hpp"

namespace qmcplusplus::psiformer::device
{
/// One scaled complex determinant's unit phase, magnitude and regular/singular state.
struct ComplexValueChannel
{
  cuDoubleComplex phase;
  double log_abs;
  std::uint64_t status;
};
static_assert(sizeof(cuDoubleComplex) == 16 && sizeof(ComplexValueChannel) == 32);
static_assert(std::is_standard_layout_v<ComplexValueChannel>);

/// Complete core amplitude; status0=regular and1=node, never a derivative promise.
struct ComplexValueResult
{
  cuDoubleComplex phase;
  double log_abs, cancellation_ratio;
  std::uint64_t singular_channels, status;
};
static_assert(sizeof(ComplexValueResult) == 48);
static_assert(std::is_standard_layout_v<ComplexValueResult>);

/// Reuse the real periodic envelope formula without constructing a legacy real head.
Error launchComplexValueEnvelopes(Stream stream, OpenOrbitalJetLayout layout, periodic_geometry::Cell cell,
    const double* positions, const double* nuclei, const double* pi_up, const double* pi_down,
    const double* zeta_up, const double* zeta_down, double* envelope, std::uint64_t* status);

/// Reconstruct the exact accepted base using the first point and three by-value moved-row coordinates.
Error launchComplexValueEnvelopeAnchor(Stream stream, OpenOrbitalJetLayout layout, periodic_geometry::Cell cell,
    const double* positions, const double* nuclei, const double* pi_up, const double* pi_down,
    const double* zeta_up, const double* zeta_down, std::size_t moved_row,
    double base_x, double base_y, double base_z, double* anchor, std::uint64_t* anchor_status);

/// Reuse only unchanged envelope rows, or rebuild every element if private anchor preparation failed.
Error launchComplexValueVirtualEnvelopes(Stream stream, OpenOrbitalJetLayout layout, periodic_geometry::Cell cell,
    const double* positions, const double* nuclei, const double* pi_up, const double* pi_down,
    const double* zeta_up, const double* zeta_down, const double* anchor,
    const std::size_t* moved_rows, const std::uint64_t* anchor_status,
    double* envelope, std::uint64_t* status);

/// Consume both real projections immediately into the one complex matrix allocation.
Error launchComplexValuePack(Stream stream, std::size_t count, const double* envelope,
    const double* projected_real, const double* projected_imag, cuDoubleComplex* matrix, std::uint64_t* status);

/// Factor the private head in place, optionally preparing admitted inverse/solve scratch.
/// Cold opt-in PSIFORMER_GPU_COMPLEX_VALUE_COOPERATIVE_LU=1 affects inverse-null calls only.
Error launchComplexValueFactor(Stream stream, std::size_t channels, std::size_t n,
    cuDoubleComplex* matrix, ComplexValueChannel* metadata, std::uint64_t* status,
    cuDoubleComplex* inverse = nullptr, std::size_t* permutation = nullptr, cuDoubleComplex* solve = nullptr);

/// Mixed-only conditioning evidence in a separate bank; channel ABI and FP64 entry point are unchanged.
Error launchComplexValueFactorWithDiagnostics(Stream stream, std::size_t channels, std::size_t n,
    cuDoubleComplex* matrix, ComplexValueChannel* metadata, std::uint64_t* status,
    double* minimum_scaled_pivots);

/// Record the earliest canonical same-spin zero/invalid pair without changing status/results.
Error launchComplexValuePairCheck(Stream stream, std::size_t batch, std::size_t electrons,
    std::size_t spin_up, periodic_geometry::Cell cell, const double* positions, std::uint64_t* events);

/// Compensate both Cartesian components at one log shift; exact cancellation is a valid node.
Error launchComplexValueReduce(Stream stream, std::size_t batch, std::size_t determinants,
    std::size_t electrons, std::size_t spin_up, periodic_geometry::Cell cell, const double* positions,
    const ComplexValueChannel* metadata, const double* cusp, ComplexValueResult* result, std::uint64_t* status,
    std::size_t cusp_stride = 1, const std::uint64_t* pair_events = nullptr);
} // namespace qmcplusplus::psiformer::device
#endif
