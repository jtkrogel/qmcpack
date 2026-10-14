//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerPeriodicGeometry.h
 * @brief Real smooth-torus geometry shared by host, CUDA and HIP boundary kernels.
 *
 * Lattice vectors and their reciprocal duals are Cartesian rows. This geometry
 * is independent of the wavefunction scalar/phase policy; it supplies no twist
 * factor and makes no minimum-image Euclidean approximation.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_PERIODIC_GEOMETRY_H
#define QMCPLUSPLUS_PSIFORMER_PERIODIC_GEOMETRY_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QMC_PF_PERIODIC_HOST_DEVICE __host__ __device__
#else
#define QMC_PF_PERIODIC_HOST_DEVICE
#endif

namespace qmcplusplus::psiformer::periodic_geometry
{

/// Immutable fully periodic cell coefficients, small enough for kernel arguments.
struct Cell
{
  double lattice[3][3]{};
  double reciprocal[3][3]{};
};

static_assert(std::is_trivially_copyable_v<Cell> && std::is_standard_layout_v<Cell>);

/// Build the same validated reciprocal rows used by the CPU geometry cache.
inline Cell makeCell(const std::array<std::array<double, 3>, 3>& lattice,
                     const std::array<bool, 3>& periodic_axes)
{
  Cell result;
  double scale = 0;
  for (std::size_t axis = 0; axis < 3; ++axis)
  {
    if (!periodic_axes[axis])
      throw std::invalid_argument("Periodic PsiFormer device geometry requires three periodic axes");
    for (std::size_t component = 0; component < 3; ++component)
    {
      const double value = lattice[axis][component];
      if (!device_math::isFiniteBinary64(value))
        throw std::invalid_argument("Periodic PsiFormer cell is nonfinite");
      result.lattice[axis][component] = value;
      if (::fabs(value) > scale)
        scale = ::fabs(value);
    }
  }
  double crosses[3][3];
  for (std::size_t axis = 0; axis < 3; ++axis)
  {
    const auto& left = lattice[(axis + 1) % 3];
    const auto& right = lattice[(axis + 2) % 3];
    crosses[axis][0] = left[1] * right[2] - left[2] * right[1];
    crosses[axis][1] = left[2] * right[0] - left[0] * right[2];
    crosses[axis][2] = left[0] * right[1] - left[1] * right[0];
  }
  const double determinant = lattice[0][0] * crosses[0][0] +
      lattice[0][1] * crosses[0][1] + lattice[0][2] * crosses[0][2];
  if (!device_math::isFiniteBinary64(determinant) || scale == 0 ||
      ::fabs(determinant) <= 128 * std::numeric_limits<double>::epsilon() * scale * scale * scale)
    throw std::invalid_argument("Periodic PsiFormer geometry requires a finite full-rank lattice");
  for (std::size_t axis = 0; axis < 3; ++axis)
    for (std::size_t component = 0; component < 3; ++component)
    {
      result.reciprocal[axis][component] = crosses[axis][component] / determinant;
      if (!device_math::isFiniteBinary64(result.reciprocal[axis][component]))
        throw std::invalid_argument("Periodic PsiFormer reciprocal cell is nonfinite");
    }
  return result;
}

/// Check externally supplied POD cell coefficients before submitting any kernels.
inline void validateCell(const Cell& cell)
{
  std::array<std::array<double, 3>, 3> lattice{};
  for (std::size_t axis = 0; axis < 3; ++axis)
    for (std::size_t component = 0; component < 3; ++component)
      lattice[axis][component] = cell.lattice[axis][component];
  const auto expected = makeCell(lattice, {true, true, true});
  for (std::size_t axis = 0; axis < 3; ++axis)
    for (std::size_t component = 0; component < 3; ++component)
      if (!device_math::isFiniteBinary64(cell.reciprocal[axis][component]) ||
          cell.reciprocal[axis][component] != expected.reciprocal[axis][component])
        throw std::invalid_argument("Periodic PsiFormer cell has inconsistent reciprocal coefficients");
}

/// Complete torus pair values and Cartesian first/trace-second derivatives.
struct Pair
{
  double displacement[3]{}, complementary[3]{};
  double jacobian[3][3]{}, complementary_jacobian[3][3]{};
  double laplacian[3]{}, complementary_laplacian[3]{};
  double radius = 0, radius_gradient[3]{}, radius_gradient_norm_squared = 0, radius_laplacian = 0;
};

/// Form CPU-equivalent smooth image-periodic geometry without allocations.
/// A zero radius is a valid value; derivative consumers reject its cusp separately.
QMC_PF_PERIODIC_HOST_DEVICE inline bool evaluatePair(const Cell& cell,
    const double* target, const double* source, Pair& pair) noexcept
{
  pair = Pair{};
  double difference[3];
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
  {
    difference[dimension] = target[dimension] - source[dimension];
    if (!device_math::isFiniteBinary64(target[dimension]) ||
        !device_math::isFiniteBinary64(source[dimension]) ||
        !device_math::isFiniteBinary64(difference[dimension]))
      return false;
  }
  constexpr double two_pi = 6.283185307179586476925286766559;
  constexpr double epsilon = 2.220446049250313080847263336181640625e-16;
  double sine[3], cosine[3], sine_first[3], cosine_first[3], sine_second[3], cosine_second[3];
  for (std::size_t axis = 0; axis < 3; ++axis)
  {
    double fractional = 0;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      fractional += cell.reciprocal[axis][dimension] * difference[dimension];
    if (!device_math::isFiniteBinary64(fractional))
      return false;
    double reduced = fractional - ::nearbyint(fractional);
    if (::fabs(reduced) <= 64 * epsilon * (1 + ::fabs(fractional)))
      reduced = 0;
    const double angle = two_pi * reduced;
    sine[axis] = ::sin(angle) / two_pi;
    cosine[axis] = (1 - ::cos(angle)) / two_pi;
    sine_first[axis] = ::cos(angle);
    cosine_first[axis] = ::sin(angle);
    sine_second[axis] = -two_pi * ::sin(angle);
    cosine_second[axis] = two_pi * ::cos(angle);
  }

  // Vector embedding jets are traced in Cartesian coordinates, including skew
  // cells; treating fractional axes as independent Cartesian axes is incorrect.
  for (std::size_t component = 0; component < 3; ++component)
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
      const double lattice = cell.lattice[axis][component];
      pair.displacement[component] += lattice * sine[axis];
      pair.complementary[component] += lattice * cosine[axis];
      double reciprocal_norm_squared = 0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double reciprocal = cell.reciprocal[axis][dimension];
        pair.jacobian[component][dimension] += lattice * sine_first[axis] * reciprocal;
        pair.complementary_jacobian[component][dimension] += lattice * cosine_first[axis] * reciprocal;
        reciprocal_norm_squared += reciprocal * reciprocal;
      }
      pair.laplacian[component] += lattice * sine_second[axis] * reciprocal_norm_squared;
      pair.complementary_laplacian[component] += lattice * cosine_second[axis] * reciprocal_norm_squared;
    }
  double squared_radius = 0;
  for (std::size_t component = 0; component < 3; ++component)
    squared_radius += pair.displacement[component] * pair.displacement[component] +
        pair.complementary[component] * pair.complementary[component];
  pair.radius = ::sqrt(squared_radius);
  if (pair.radius != 0)
  {
    for (std::size_t derivative = 0; derivative < 3; ++derivative)
    {
      double half_first = 0, half_second = 0;
      for (std::size_t component = 0; component < 3; ++component)
      {
        const double q_first = pair.jacobian[component][derivative];
        const double p_first = pair.complementary_jacobian[component][derivative];
        half_first += pair.displacement[component] * q_first + pair.complementary[component] * p_first;
        half_second += q_first * q_first + p_first * p_first;
      }
      pair.radius_gradient[derivative] = half_first / pair.radius;
      pair.radius_laplacian += half_second / pair.radius -
          half_first * half_first / (pair.radius * pair.radius * pair.radius);
    }
    for (std::size_t component = 0; component < 3; ++component)
      pair.radius_laplacian += (pair.displacement[component] * pair.laplacian[component] +
          pair.complementary[component] * pair.complementary_laplacian[component]) / pair.radius;
    for (double value : pair.radius_gradient)
      pair.radius_gradient_norm_squared += value * value;
  }

  // Fail before consumers can publish partial jets, including overflow in a
  // finite but numerically extreme lattice/position combination.
  if (!device_math::isFiniteBinary64(pair.radius) ||
      !device_math::isFiniteBinary64(pair.radius_gradient_norm_squared) ||
      !device_math::isFiniteBinary64(pair.radius_laplacian))
    return false;
  for (std::size_t component = 0; component < 3; ++component)
  {
    if (!device_math::isFiniteBinary64(pair.displacement[component]) ||
        !device_math::isFiniteBinary64(pair.complementary[component]) ||
        !device_math::isFiniteBinary64(pair.laplacian[component]) ||
        !device_math::isFiniteBinary64(pair.complementary_laplacian[component]) ||
        !device_math::isFiniteBinary64(pair.radius_gradient[component]))
      return false;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      if (!device_math::isFiniteBinary64(pair.jacobian[component][dimension]) ||
          !device_math::isFiniteBinary64(pair.complementary_jacobian[component][dimension]))
        return false;
  }
  return true;
}

} // namespace qmcplusplus::psiformer::periodic_geometry

#undef QMC_PF_PERIODIC_HOST_DEVICE
#endif
