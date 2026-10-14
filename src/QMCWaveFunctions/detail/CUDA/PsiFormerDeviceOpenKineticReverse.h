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

/** Reverse one retained open orbital FULL_VGL tape.
 *
 * Exactly one device thread owns the complete reduction and its six canonical
 * response tensors.  The call allocates no storage and uses no atomics.
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
    const periodic_geometry::Cell* periodic_cell = nullptr);

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
