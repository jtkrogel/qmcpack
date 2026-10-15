//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerFeatureCoordinateReverse.h
 * @brief Row-local transpose of the real raw-feature coordinate Jacobian.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_FEATURE_COORDINATE_REVERSE_H
#define QMCPLUSPLUS_PSIFORMER_FEATURE_COORDINATE_REVERSE_H

#include "PsiFormerOpenSpatial.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QMC_PF_COORDINATE_HOST_DEVICE __host__ __device__
#else
#define QMC_PF_COORDINATE_HOST_DEVICE
#endif

namespace qmcplusplus::psiformer::feature_coordinate
{

/** Contract one raw-feature row into three Cartesian input cotangents.
 *
 * The caller validates the matching open/periodic feature layout before launch.
 * Raw cotangents use its value-plane offsets; positions retain their independent
 * strides. The spin feature has exactly zero coordinate derivative. Output is
 * overwritten only after the entire electron row is finite and differentiable.
 * This leaf owns no storage and performs no parameter or Laplacian work.
 */
QMC_PF_COORDINATE_HOST_DEVICE inline OpenSpatialStatus coordinateAdjointForElectron(
    const OpenFeatureJetLayout& layout, const double* positions, const double* nuclei,
    const double* raw_feature_adjoint, std::size_t configuration, std::size_t electron,
    double* coordinate_adjoint, const periodic_geometry::Cell* periodic_cell = nullptr) noexcept
{
  if (!positions || !nuclei || !raw_feature_adjoint || !coordinate_adjoint ||
      configuration >= layout.output.configuration_count || electron >= layout.output.electron_count)
    return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
  const auto row = electron * layout.feature_width;
  const double* cotangent = raw_feature_adjoint + layout.output.uncheckedValueOffset(configuration, row);
  for (std::size_t feature = 0; feature < layout.feature_width; ++feature)
    if (!device_math::isFiniteBinary64(cotangent[feature]))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
  const double* position = positions + configuration * layout.position_configuration_stride +
      electron * layout.position_row_stride;
  double result[3]{};

  // Geometry and radial derivatives are the exact primitives used by the forward
  // feature jets, including both periodic q and complementary p vector channels.
  for (std::size_t nucleus = 0; nucleus < layout.nucleus_count; ++nucleus)
  {
    const double* center = nuclei + nucleus * layout.nucleus_row_stride;
    if (periodic_cell)
    {
      periodic_geometry::Pair pair;
      if (!periodic_geometry::evaluatePair(*periodic_cell, position, center, pair))
        return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
      if (pair.radius == 0)
        return OpenSpatialStatus::COALESCENCE;
      const auto radial = device_math::softenedRadial(pair.radius);
      const auto base = 7 * nucleus;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        result[dimension] += cotangent[base] * radial.log1p_first * pair.radius_gradient[dimension];
        for (std::size_t vector = 0; vector < 2; ++vector)
          for (std::size_t component = 0; component < 3; ++component)
          {
            const double value = vector == 0 ? pair.displacement[component] : pair.complementary[component];
            const double* jacobian = vector == 0 ? pair.jacobian[component] : pair.complementary_jacobian[component];
            result[dimension] += cotangent[base + 1 + 3 * vector + component] *
                (jacobian[dimension] * radial.log1p_over_radius +
                 value * radial.log1p_over_radius_first * pair.radius_gradient[dimension]);
          }
      }
      continue;
    }

    // The molecular raw row consists of log(1+r) and three softened Cartesian
    // displacements per nucleus; their derivative is defined away from r=0.
    double displacement[3], radius_squared = 0;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      if (!device_math::isFiniteBinary64(position[dimension]) || !device_math::isFiniteBinary64(center[dimension]))
        return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
      displacement[dimension] = position[dimension] - center[dimension];
      radius_squared += displacement[dimension] * displacement[dimension];
    }
    const double radius = ::sqrt(radius_squared);
    if (!device_math::isFiniteBinary64(radius))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
    if (radius == 0)
      return OpenSpatialStatus::COALESCENCE;
    const auto radial = device_math::softenedRadial(radius);
    const auto base = 4 * nucleus;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double direction = displacement[dimension] / radius;
      result[dimension] += cotangent[base] * radial.log1p_first * direction;
      for (std::size_t component = 0; component < 3; ++component)
        result[dimension] += cotangent[base + 1 + component] *
            ((component == dimension ? radial.log1p_over_radius : 0.0) +
             displacement[component] * radial.log1p_over_radius_first * direction);
    }
  }

  for (const double value : result)
    if (!device_math::isFiniteBinary64(value))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    coordinate_adjoint[dimension] = result[dimension];
  return OpenSpatialStatus::REGULAR;
}

} // namespace qmcplusplus::psiformer::feature_coordinate

#undef QMC_PF_COORDINATE_HOST_DEVICE

#endif // QMCPLUSPLUS_PSIFORMER_FEATURE_COORDINATE_REVERSE_H
