//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SCORE_KERNELS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SCORE_KERNELS_H
#include "PsiFormerDeviceComplexValueKernels.h"

namespace qmcplusplus::psiformer::device
{
/// Consume admitted inverse storage IN PLACE into ordinary analytic weighted inverses.
Error launchComplexScoreWeightedInverses(Stream stream, std::size_t determinants, std::size_t n,
    const ComplexValueChannel* channels, cuDoubleComplex* inverse, const double* cusp,
    ComplexValueResult* amplitude, std::uint64_t* status);

/// Select an objective/head component and spin mask, or envelope H*Z without A/E.
Error launchComplexScoreHeadSeed(Stream stream, std::size_t determinants, std::size_t n, std::size_t spin_up,
    const cuDoubleComplex* weighted_inverse, const double* envelope, const double* real, const double* imag,
    std::size_t objective, std::size_t head_component, std::size_t spin, double* seed, std::uint64_t* status);

/// Analytic raw cotangents C obey deltaL=Re sum(C*deltaZ); no inverse or cusp.
Error launchComplexOrbitalHeadSeed(Stream stream, std::size_t determinants, std::size_t n, std::size_t spin_up,
    const cuDoubleComplex* cotangents, const double* envelope, const double* real, const double* imag,
    std::size_t head_component, std::size_t spin, double* seed, std::uint64_t* status);

/// Pack a finite completed real objective without exposing an incomplete complexP.
Error launchComplexScorePack(Stream stream, std::size_t parameters, std::size_t component,
    const double* response, cuDoubleComplex* score, std::uint64_t* status);
} // namespace qmcplusplus::psiformer::device
#endif
