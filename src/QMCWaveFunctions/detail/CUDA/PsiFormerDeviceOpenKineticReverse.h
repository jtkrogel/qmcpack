//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceOpenKineticReverse.h
 * @brief CUDA/HIP launch boundary for the serial open-boundary lifted reverse leaves.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_OPEN_KINETIC_REVERSE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_OPEN_KINETIC_REVERSE_H

#include "PsiFormerDeviceRuntime.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOpenKineticReverse.h"

namespace qmcplusplus::psiformer::device
{

/// Observe one open real FULL root and straight-Coulomb geometry without HOST R/G/L.
Error launchResidentOpenEnergy(Stream stream, const SpatialJetLayout& root_layout,
    const double* root, const double* positions, const double* nuclei, std::size_t ions,
    const double* inverse_masses, const double* charges, double* energy, std::uint64_t* status);

/** Reverse one retained open orbital FULL_VGL tape.
 *
 * Null parallel_failure preserves the single-thread reference. An admitted
 * uint32 word selects unique-writer parallel contractions, with only integer
 * fault atomics and no numerical scratch. The call allocates no storage.
 * Mixed-invalid inputs deterministically prefer nonfinite over coalescence;
 * regular forward admission excludes such ambiguous production inputs.
 */
Error launchOpenOrbitalKineticReverse(
    Stream stream,
    const OpenOrbitalKineticReverseLayout& layout,
    const double* features,
    const double* positions,
    const double* nuclei,
    const double* backflow_up,
    const double* backflow_down,
    const double* pi_up,
    const double* pi_down,
    const double* zeta_up,
    const double* zeta_down,
    const double* backflow,
    const double* envelope,
    const double* orbital_adjoint,
    double* backflow_adjoint,
    double* envelope_adjoint,
    double* feature_adjoint,
    double* response,
    OpenKineticReverseStatus* status,
    const periodic_geometry::Cell* periodic_cell = nullptr,
    unsigned int* parallel_failure = nullptr);

/** Reverse one complete open cusp FULL_VGL adjoint into alpha responses. */
Error launchOpenCuspKineticReverse(
    Stream stream,
    const OpenCuspKineticReverseLayout& layout,
    const double* positions,
    double same_alpha,
    double opposite_alpha,
    const double* cusp_adjoint,
    double* response,
    OpenKineticReverseStatus* status,
    const periodic_geometry::Cell* periodic_cell = nullptr);

/// Reverse periodic orbital jets with exact torus radial pi/zeta mixed partials.
inline Error launchPeriodicOrbitalKineticReverse(
    Stream stream, const OpenOrbitalKineticReverseLayout& layout,
    const periodic_geometry::Cell& cell, const double* features,
    const double* positions, const double* nuclei,
    const double* backflow_up, const double* backflow_down,
    const double* pi_up, const double* pi_down, const double* zeta_up, const double* zeta_down,
    const double* backflow, const double* envelope, const double* orbital_adjoint,
    double* backflow_adjoint, double* envelope_adjoint, double* feature_adjoint,
    double* response, OpenKineticReverseStatus* status)
{
  return launchOpenOrbitalKineticReverse(stream, layout, features, positions, nuclei,
      backflow_up, backflow_down, pi_up, pi_down, zeta_up, zeta_down,
      backflow, envelope, orbital_adjoint, backflow_adjoint, envelope_adjoint,
      feature_adjoint, response, status, &cell);
}

/// Reverse periodic cusp jets with full Cartesian radial alpha mixed partials.
inline Error launchPeriodicCuspKineticReverse(
    Stream stream, const OpenCuspKineticReverseLayout& layout,
    const periodic_geometry::Cell& cell, const double* positions,
    double same_alpha, double opposite_alpha, const double* cusp_adjoint,
    double* response, OpenKineticReverseStatus* status)
{
  return launchOpenCuspKineticReverse(stream, layout, positions, same_alpha,
      opposite_alpha, cusp_adjoint, response, status, &cell);
}

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_OPEN_KINETIC_REVERSE_H
