//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerCoordinateGradientLeaf.h
 * @brief Complete one real-core coordinate row from raw and envelope cotangents.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_COORDINATE_GRADIENT_LEAF_H
#define QMCPLUSPLUS_PSIFORMER_COORDINATE_GRADIENT_LEAF_H

#include "PsiFormerFeatureCoordinateReverse.h"
#include "PsiFormerOpenOrbital.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QMC_PF_COORDINATE_GRADIENT_HD __host__ __device__
#else
#define QMC_PF_COORDINATE_GRADIENT_HD
#endif

namespace qmcplusplus::psiformer::coordinate_gradient
{

/// Use the same molecular or smooth-torus radius and Cartesian derivative as VGL.
QMC_PF_COORDINATE_GRADIENT_HD inline OpenSpatialStatus radiusGradient(const double* first,
    const double* second, double& radius, double* gradient, const periodic_geometry::Cell* cell) noexcept
{
  if (cell)
  {
    periodic_geometry::Pair pair;
    if (!periodic_geometry::evaluatePair(*cell, first, second, pair))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
    radius = pair.radius;
    if (radius == 0)
      return OpenSpatialStatus::COALESCENCE;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      gradient[dimension] = pair.radius_gradient[dimension];
    return OpenSpatialStatus::REGULAR;
  }
  double squared = 0;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
  {
    if (!device_math::isFiniteBinary64(first[dimension]) || !device_math::isFiniteBinary64(second[dimension]))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
    gradient[dimension] = first[dimension] - second[dimension];
    squared += gradient[dimension] * gradient[dimension];
  }
  radius = ::sqrt(squared);
  if (!device_math::isFiniteBinary64(radius))
    return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
  if (radius == 0)
    return OpenSpatialStatus::COALESCENCE;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    gradient[dimension] /= radius;
  return OpenSpatialStatus::REGULAR;
}

/** Finish one row without parameter responses or spatial tangent planes.
 * Inputs use admitted packed value-head layouts. Envelope cotangents already
 * contain determinant/product seeds. The common cusp contributes once, with
 * unit configuration weight. Fixed outer twist is applied by the caller.
 */
QMC_PF_COORDINATE_GRADIENT_HD inline OpenSpatialStatus finishElectron(
    const OpenOrbitalJetLayout& orbital, const OpenFeatureJetLayout& raw,
    const double* positions, const double* nuclei, const double* raw_bar, const double* envelope_bar,
    const double* pi_up, const double* pi_down, const double* zeta_up, const double* zeta_down,
    double same_alpha, double opposite_alpha, std::size_t configuration, std::size_t electron,
    double* output, const periodic_geometry::Cell* cell = nullptr) noexcept
{
  double result[3]{};
  auto status = feature_coordinate::coordinateAdjointForElectron(raw, positions, nuclei, raw_bar,
      configuration, electron, result, cell);
  if (status != OpenSpatialStatus::REGULAR)
    return status;
  if (!device_math::isFiniteBinary64(same_alpha) || !device_math::isFiniteBinary64(opposite_alpha))
    return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
  const auto ne = raw.output.electron_count;
  const double* configuration_positions = positions + configuration * raw.position_configuration_stride;
  const double* position = configuration_positions + electron * raw.position_row_stride;
  const bool up = electron < raw.spin_up_count;
  const double* pi = up ? pi_up : pi_down;
  const double* zeta = up ? zeta_up : zeta_down;

  // Nuclear geometry is reused across every determinant/orbital in this row.
  for (std::size_t nucleus = 0; nucleus < raw.nucleus_count; ++nucleus)
  {
    double radius, gradient[3];
    status = radiusGradient(position, nuclei + nucleus * raw.nucleus_row_stride, radius, gradient, cell);
    if (status != OpenSpatialStatus::REGULAR)
      return status;
    double radial_adjoint = 0;
    for (std::size_t determinant = 0; determinant < orbital.determinant_count; ++determinant)
      for (std::size_t column = 0; column < ne; ++column)
      {
        const auto parameter = open_orbital::envelopeOffset(orbital, determinant, column, nucleus);
        const auto element = open_orbital::outputElement(orbital, determinant, electron, column);
        const auto offset = orbital.orbitals.uncheckedValueOffset(configuration, element);
        const double decay = ::fabs(zeta[parameter]);
        radial_adjoint -= envelope_bar[offset] * decay * pi[parameter] * ::exp(-decay * radius);
      }
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      result[dimension] += radial_adjoint * gradient[dimension];
  }

  // Match the forward cusp's ordered endpoints and incidence signs. Each output
  // row owns its reduction, so no atomics or pair-sized scratch are required.
  for (std::size_t other = 0; other < ne; ++other)
  {
    if (other == electron)
      continue;
    const auto first = electron < other ? electron : other;
    const auto second = electron < other ? other : electron;
    double radius, gradient[3];
    status = radiusGradient(configuration_positions + first * raw.position_row_stride,
        configuration_positions + second * raw.position_row_stride, radius, gradient, cell);
    if (status != OpenSpatialStatus::REGULAR)
      return status;
    const bool same = up == (other < raw.spin_up_count);
    const double alpha = same ? same_alpha : opposite_alpha;
    const double denominator = alpha + radius;
    const double first_derivative = (same ? 0.25 : 0.5) * alpha * alpha / (denominator * denominator);
    const double sign = electron == first ? 1.0 : -1.0;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      result[dimension] += sign * first_derivative * gradient[dimension];
  }

  for (const double value : result)
    if (!device_math::isFiniteBinary64(value))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    output[dimension] = result[dimension];
  return OpenSpatialStatus::REGULAR;
}

} // namespace qmcplusplus::psiformer::coordinate_gradient

#undef QMC_PF_COORDINATE_GRADIENT_HD

#endif // QMCPLUSPLUS_PSIFORMER_COORDINATE_GRADIENT_LEAF_H
