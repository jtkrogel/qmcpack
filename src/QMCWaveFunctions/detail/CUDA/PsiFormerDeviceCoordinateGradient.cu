//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceCoordinateGradient.cu
 * @brief Portable independent-row coordinate-gradient launches.
 */

#include "PsiFormerDeviceCoordinateGradient.h"

namespace qmcplusplus::psiformer::device
{
namespace
{

/// Each thread owns one complete coordinate row and its admission status.
template<bool periodic>
__global__ void coordinateGradientKernel(OpenOrbitalJetLayout orbital, OpenFeatureJetLayout raw,
    const double* positions, const double* nuclei, const double* raw_bar, const double* envelope_bar,
    const double* pi_up, const double* pi_down, const double* zeta_up, const double* zeta_down,
    double same_alpha, double opposite_alpha, double* gradient, OpenSpatialStatus* status,
    periodic_geometry::Cell cell)
{
  const auto row = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto ne = raw.output.electron_count;
  if (row < raw.output.configuration_count * ne)
    status[row] = coordinate_gradient::finishElectron(orbital, raw, positions, nuclei, raw_bar,
        envelope_bar, pi_up, pi_down, zeta_up, zeta_down, same_alpha, opposite_alpha,
        row / ne, row % ne, gradient + 3 * row, periodic ? &cell : nullptr);
}

} // namespace

// Validate the admitted layouts and dispatch the shared boundary-specific leaf.
Error launchCoordinateGradientLeaves(Stream stream, const OpenOrbitalJetLayout& orbital,
    const OpenFeatureJetLayout& raw, const double* positions, const double* nuclei,
    const double* raw_bar, const double* envelope_bar, const double* pi_up, const double* pi_down,
    const double* zeta_up, const double* zeta_down, double same_alpha, double opposite_alpha,
    double* gradient, OpenSpatialStatus* status, const periodic_geometry::Cell* cell)
{
  validateOpenOrbitalJetLayout(orbital);
  validateOpenFeatureJetLayout(raw, cell != nullptr);
  const auto count = spatial_detail::checkedProduct(raw.output.configuration_count, raw.output.electron_count,
      "Coordinate gradient launch extent overflow");
  if (!positions || !nuclei || !raw_bar || !envelope_bar || !gradient || !status ||
      orbital.features.configuration_count != raw.output.configuration_count ||
      orbital.features.electron_count != raw.output.electron_count || orbital.spin_up_count != raw.spin_up_count ||
      orbital.nucleus_count != raw.nucleus_count ||
      (raw.spin_up_count && (!pi_up || !zeta_up)) ||
      (raw.spin_up_count != raw.output.electron_count && (!pi_down || !zeta_down)))
    throw std::invalid_argument("Coordinate gradient leaf storage or layouts are incompatible");
  if (cell)
    periodic_geometry::validateCell(*cell);
  constexpr unsigned int threads = 128;
  const auto blocks = count / threads + (count % threads != 0);
  if (blocks > std::numeric_limits<unsigned int>::max())
    throw std::length_error("Coordinate gradient grid exceeds device x dimension");
  if (cell)
    coordinateGradientKernel<true><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(orbital, raw,
        positions, nuclei, raw_bar, envelope_bar, pi_up, pi_down, zeta_up, zeta_down, same_alpha,
        opposite_alpha, gradient, status, *cell);
  else
    coordinateGradientKernel<false><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(orbital, raw,
        positions, nuclei, raw_bar, envelope_bar, pi_up, pi_down, zeta_up, zeta_down, same_alpha,
        opposite_alpha, gradient, status, {});
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

} // namespace qmcplusplus::psiformer::device
