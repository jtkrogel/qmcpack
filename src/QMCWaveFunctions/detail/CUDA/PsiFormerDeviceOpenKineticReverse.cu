//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceOpenKineticReverse.cu
 * @brief Deterministic CUDA/HIP wrappers for open-boundary lifted reverse leaves.
 */

#include "PsiFormerDeviceOpenKineticReverse.h"

#include <stdexcept>

namespace qmcplusplus::psiformer::device
{
namespace
{

Error peekAtLastError()
{
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

/// Specialize boundary geometry while retaining the single-owner reverse reduction.
template<bool periodic>
__global__ void openOrbitalKineticReverseKernel(
    OpenOrbitalKineticReverseLayout layout,
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
    OpenKineticReverseStatus* status, periodic_geometry::Cell cell)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
    open_kinetic_reverse::reverseOpenOrbitalKineticSerial(
        layout, features, positions, nuclei, backflow_up, backflow_down, pi_up,
        pi_down, zeta_up, zeta_down, backflow, envelope, orbital_adjoint,
        backflow_adjoint, envelope_adjoint, feature_adjoint, response, status,
        periodic ? &cell : nullptr);
}

/// Reverse cusp alpha jets using the same prepared boundary policy as forward.
template<bool periodic>
__global__ void openCuspKineticReverseKernel(
    OpenCuspKineticReverseLayout layout,
    const double* positions,
    double same_alpha,
    double opposite_alpha,
    const double* cusp_adjoint,
    double* response,
    OpenKineticReverseStatus* status, periodic_geometry::Cell cell)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
    open_kinetic_reverse::reverseOpenCuspKineticSerial(
        layout, positions, same_alpha, opposite_alpha, cusp_adjoint, response,
        status, periodic ? &cell : nullptr);
}

} // namespace

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
    OpenKineticReverseStatus* status, const periodic_geometry::Cell* periodic_cell)
{
  validateOpenOrbitalKineticReverseLayout(layout);
  if (isZeroWork(layout))
    return success;
  if (periodic_cell)
    periodic_geometry::validateCell(*periodic_cell);
  if (!features || !positions || !nuclei || !backflow_up || !backflow_down ||
      !pi_up || !pi_down || !zeta_up || !zeta_down || !backflow || !envelope ||
      !orbital_adjoint || !backflow_adjoint || !envelope_adjoint ||
      !feature_adjoint || !response || !status)
    throw std::invalid_argument("PsiFormer open orbital kinetic reverse storage is null");
  validateOpenOrbitalKineticReverseNoAlias(
      features, positions, nuclei, backflow_up, backflow_down, pi_up, pi_down,
      zeta_up, zeta_down, backflow, envelope, orbital_adjoint, backflow_adjoint,
      envelope_adjoint, feature_adjoint, response, status);
  if (periodic_cell)
    openOrbitalKineticReverseKernel<true><<<1, 1, 0, stream>>>(
        layout, features, positions, nuclei, backflow_up, backflow_down, pi_up,
        pi_down, zeta_up, zeta_down, backflow, envelope, orbital_adjoint,
        backflow_adjoint, envelope_adjoint, feature_adjoint, response, status, *periodic_cell);
  else
    openOrbitalKineticReverseKernel<false><<<1, 1, 0, stream>>>(
        layout, features, positions, nuclei, backflow_up, backflow_down, pi_up,
        pi_down, zeta_up, zeta_down, backflow, envelope, orbital_adjoint,
        backflow_adjoint, envelope_adjoint, feature_adjoint, response, status, {});
  return peekAtLastError();
}

Error launchOpenCuspKineticReverse(
    Stream stream,
    const OpenCuspKineticReverseLayout& layout,
    const double* positions,
    double same_alpha,
    double opposite_alpha,
    const double* cusp_adjoint,
    double* response,
    OpenKineticReverseStatus* status, const periodic_geometry::Cell* periodic_cell)
{
  validateOpenCuspKineticReverseLayout(layout);
  if (isZeroWork(layout))
    return success;
  if (periodic_cell)
    periodic_geometry::validateCell(*periodic_cell);
  if (!positions || !cusp_adjoint || !response || !status)
    throw std::invalid_argument("PsiFormer open cusp kinetic reverse storage is null");
  validateOpenCuspKineticReverseNoAlias(
      positions, cusp_adjoint, response, status);
  if (periodic_cell)
    openCuspKineticReverseKernel<true><<<1, 1, 0, stream>>>(
        layout, positions, same_alpha, opposite_alpha, cusp_adjoint, response, status, *periodic_cell);
  else
    openCuspKineticReverseKernel<false><<<1, 1, 0, stream>>>(
        layout, positions, same_alpha, opposite_alpha, cusp_adjoint, response, status, {});
  return peekAtLastError();
}

} // namespace qmcplusplus::psiformer::device
