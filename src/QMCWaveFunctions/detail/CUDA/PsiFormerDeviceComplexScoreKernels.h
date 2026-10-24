//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SCORE_KERNELS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SCORE_KERNELS_H
#include "PsiFormerDeviceComplexValueKernels.h"
#include <type_traits>

namespace qmcplusplus::psiformer::device
{
/// One private epoch, fully overwritten before the bounded multi-block sequence reads it.
struct ComplexScoreWeightedGridState
{
  double log_weights[16];
  cuDoubleComplex ratios[16];
  double shift, log_total;
  cuDoubleComplex phase;
  std::uint64_t first_bad, rejected;
};
static_assert(sizeof(ComplexScoreWeightedGridState) == 432);
static_assert(std::is_standard_layout_v<ComplexScoreWeightedGridState>);
static_assert(std::is_trivially_copyable_v<ComplexScoreWeightedGridState>);

/// Consume admitted inverse storage IN PLACE into ordinary analytic weighted inverses.
Error launchComplexScoreWeightedInverses(Stream stream, std::size_t determinants, std::size_t n,
    const ComplexValueChannel* channels, cuDoubleComplex* inverse, const double* cusp,
    ComplexValueResult* amplitude, std::uint64_t* status);

/// Optional bounded single-block route; unsupported dimensions retain the scalar entry.
Error launchComplexScoreWeightedInversesCooperative(Stream stream, std::size_t determinants, std::size_t n,
    const ComplexValueChannel* channels, cuDoubleComplex* inverse, const double* cusp,
    ComplexValueResult* amplitude, std::uint64_t* status);

/// Ordered global phases; unsupported dimensions preserve the existing cooperative/scalar route.
Error launchComplexScoreWeightedInversesGrid(Stream stream, std::size_t determinants, std::size_t n,
    const ComplexValueChannel* channels, cuDoubleComplex* inverse, const double* cusp,
    ComplexValueResult* amplitude, std::uint64_t* status,
    ComplexScoreWeightedGridState* state, std::size_t state_count);

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
