//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceComplexSpatialConcurrent.h
 * @brief Opt-in bounded forward contraction over independent determinant targets.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_CONCURRENT_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_CONCURRENT_H
#include "PsiFormerDeviceComplexSpatialKernels.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexSpatialTracePlan.h"

namespace qmcplusplus::psiformer::device
{
/// Consume private preallocated slab scratch, preserving physical output strides.
/// All inputs/scratch outlive the queue; only the enclosing owner publishes results.
Error launchComplexSpatialTracesConcurrent(Stream stream, std::size_t batch, std::size_t determinants,
    std::size_t n, std::size_t targets, std::size_t output_electrons, std::size_t concurrent_targets,
    const ComplexValueChannel* channels, const ComplexValueResult* amplitude,
    const cuDoubleComplex* inverse, const cuDoubleComplex* head, const double* cusp,
    cuDoubleComplex* product, cuDoubleComplex* channel_terms, cuDoubleComplex* gradient,
    cuDoubleComplex* lap_log, cuDoubleComplex* lap_ratio, std::uint64_t* status);
} // namespace qmcplusplus::psiformer::device
#endif
