//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_KERNELS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_KERNELS_H
#include "PsiFormerDeviceComplexValueKernels.h"

namespace qmcplusplus::psiformer::device
{
/// Compose ordinary complex orbital jets directly into [plane,B,D,N,N].
Error launchComplexSpatialHead(Stream stream, OpenOrbitalJetLayout layout, const double* envelope,
    const double* real, const double* imag, cuDoubleComplex* head, std::uint64_t* status);

/// Require every inverse and combine physical CORE G/Llog/Lratio without conjugation.
Error launchComplexSpatialTraces(Stream stream, std::size_t batch, std::size_t determinants, std::size_t n,
    const ComplexValueChannel* channels, const ComplexValueResult* amplitude, const cuDoubleComplex* inverse,
    const cuDoubleComplex* head, const double* cusp, cuDoubleComplex* product,
    cuDoubleComplex* gradient, cuDoubleComplex* lap_log, cuDoubleComplex* lap_ratio, std::uint64_t* status);

/// Immediate B1 compact-target overload; physical N matrices, only 3K/K/K output lanes.
Error launchComplexSpatialTraces(Stream stream, OpenOrbitalJetLayout layout,
    const ComplexValueChannel* channels, const ComplexValueResult* amplitude, const cuDoubleComplex* inverse,
    const cuDoubleComplex* head, const double* cusp, cuDoubleComplex* product,
    cuDoubleComplex* gradient, cuDoubleComplex* lap_log, cuDoubleComplex* lap_ratio, std::uint64_t* status);
} // namespace qmcplusplus::psiformer::device
#endif
