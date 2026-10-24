//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_KERNELS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_KERNELS_H
#include "PsiFormerDeviceComplexValueKernels.h"

namespace qmcplusplus::psiformer::device
{
/// Observe the legacy cooperative oracle without widening its parser or default.
bool complexSpatialCooperativeTracesRequested();

/// Stage bounded B1 concurrent outputs; on failure replay the canonical cooperative prefix.
Error launchComplexSpatialTracesTransactional(Stream stream, OpenOrbitalJetLayout layout,
    const ComplexValueChannel* channels, const ComplexValueResult* amplitude, const cuDoubleComplex* inverse,
    const cuDoubleComplex* head, const double* cusp, cuDoubleComplex* product, std::size_t product_capacity,
    cuDoubleComplex* channel_terms, std::size_t term_capacity, cuDoubleComplex* trial, std::size_t trial_capacity,
    cuDoubleComplex* outputs, std::uint64_t* status, std::uint64_t* private_failure);

/// Compose ordinary complex orbital jets directly into [plane,B,D,N,N].
Error launchComplexSpatialHead(Stream stream, OpenOrbitalJetLayout layout, const double* envelope,
    const double* real, const double* imag, cuDoubleComplex* head, std::uint64_t* status);

/// Preserve canonical raw VALUE in head while current VALUE occupies dead trace scratch.
Error launchComplexSpatialHeadSeparateValue(Stream stream, OpenOrbitalJetLayout layout, const double* envelope,
    const double* real, const double* imag, cuDoubleComplex* head, cuDoubleComplex* current_value,
    std::uint64_t* status);

/// Require every inverse and combine physical CORE G/Llog/Lratio without conjugation.
Error launchComplexSpatialTraces(Stream stream, std::size_t batch, std::size_t determinants, std::size_t n,
    const ComplexValueChannel* channels, const ComplexValueResult* amplitude, const cuDoubleComplex* inverse,
    const cuDoubleComplex* head, const double* cusp, cuDoubleComplex* product,
    cuDoubleComplex* gradient, cuDoubleComplex* lap_log, cuDoubleComplex* lap_ratio, std::uint64_t* status);

/// Compact-target input; output configuration strides default to K, or explicit physical N.
Error launchComplexSpatialTraces(Stream stream, OpenOrbitalJetLayout layout,
    const ComplexValueChannel* channels, const ComplexValueResult* amplitude, const cuDoubleComplex* inverse,
    const cuDoubleComplex* head, const double* cusp, cuDoubleComplex* product,
    cuDoubleComplex* gradient, cuDoubleComplex* lap_log, cuDoubleComplex* lap_ratio, std::uint64_t* status,
    std::size_t output_electrons = 0);
} // namespace qmcplusplus::psiformer::device
#endif
