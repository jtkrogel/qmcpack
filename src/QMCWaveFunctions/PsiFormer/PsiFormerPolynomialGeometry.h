//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_POLYNOMIAL_GEOMETRY_H
#define QMCPLUSPLUS_PSIFORMER_POLYNOMIAL_GEOMETRY_H

#include "PsiFormerGeometry.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace pf
{
/// One stack-local polynomial displacement jet, without another persistent cache.
struct PolynomialPeriodicJet
{
  double radius = 0, radius_laplacian = 0;
  GeometryPosition vector{}, radius_gradient{}, vector_laplacian{};
  std::array<GeometryPosition, 3> vector_jacobian{};
  bool differentiable = false;
};

/** Evaluate the minimal polynomial frame from an existing periodic cache entry.
 * Dual contraction recovers sine/cosine angles from the cached Cartesian pair
 * vectors. Reciprocal cross contractions are retained for skew cells. Vector
 * features are C1, not C2, on zero angular-coordinate planes; the flag rejects
 * spatial use there (within the dual-contraction roundoff bound), even when the
 * scalar radius has a smoother special-cell limit. Wrap seams at +/-pi are C2.
 */
inline PolynomialPeriodicJet polynomialPeriodicJet(const GeometryPairTable& table,
    std::size_t pair, const GeometryBoundary& cell, int orientation = 1)
{
  if (pair >= table.size() || cell.kind != GeometryBoundaryKind::PERIODIC ||
      (orientation != 1 && orientation != -1))
    throw std::invalid_argument("Polynomial periodic geometry requires an oriented periodic pair");
  const auto& a = cell.lattice_vectors;
  const auto cross = [](const auto& x, const auto& y) {
    return GeometryPosition{x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2],
                            x[0] * y[1] - x[1] * y[0]};
  };
  std::array<GeometryPosition, 3> reciprocal{cross(a[1], a[2]), cross(a[2], a[0]), cross(a[0], a[1])};
  double volume = 0;
  for (std::size_t d = 0; d < 3; ++d) volume += a[0][d] * reciprocal[0][d];
  if (!std::isfinite(volume) || volume == 0)
    throw std::invalid_argument("Polynomial periodic geometry requires a finite full-rank cell");
  const double pi = std::acos(-1.0), two_pi = 2 * pi;
  for (auto& row : reciprocal)
    for (double& component : row) component *= two_pi / volume;
  PolynomialPeriodicJet result;
  result.differentiable = true;
  double correction = 0, correction_laplacian = 0;
  GeometryPosition correction_gradient{};
  for (std::size_t axis = 0; axis < 3; ++axis)
  {
    // Reciprocal duality recovers this angular sine and cosine from the shared q/p cache.
    double sine = 0, cosine = 1, contraction_scale = 1, b2 = 0, a2 = 0;
    for (std::size_t d = 0; d < 3; ++d)
    {
      const double term = reciprocal[axis][d] * table.displacements()[pair][d];
      sine += orientation * term;
      cosine -= reciprocal[axis][d] * table.complementaryDisplacements()[pair][d];
      contraction_scale += std::abs(term);
      b2 += reciprocal[axis][d] * reciprocal[axis][d];
      a2 += a[axis][d] * a[axis][d] / (two_pi * two_pi);
    }
    const double w = std::atan2(sine, cosine), magnitude = std::abs(w);
    if (cosine > 0 && std::abs(sine) <= 64 * std::numeric_limits<double>::epsilon() * contraction_scale)
      result.differentiable = false;
    const double sign = w < 0 ? -1 : 1;
    // Differentiate the scalar polynomials on the selected side of each C1 zero plane.
    const double f = magnitude - .25 * std::pow(w, 4) / std::pow(pi, 3);
    const double f1 = sign - std::pow(w / pi, 3), f2 = -3 * w * w / std::pow(pi, 3);
    const double g = w - 1.5 * w * magnitude / pi + .5 * w * w * w / (pi * pi);
    const double g1 = 1 - 3 * magnitude / pi + 1.5 * w * w / (pi * pi);
    const double g2 = w == 0 ? 0 : -3 * sign / pi + 3 * w / (pi * pi);
    const double h = (f - std::abs(g)) * (f + std::abs(g));
    const double h1 = 2 * (f * f1 - g * g1);
    const double h2 = 2 * (f1 * f1 + f * f2 - g1 * g1 - g * g2);
    correction += a2 * h;
    correction_laplacian += a2 * h2 * b2;
    for (std::size_t d = 0; d < 3; ++d)
    {
      correction_gradient[d] += a2 * h1 * reciprocal[axis][d];
      const double dual = a[axis][d] / two_pi;
      result.vector[d] += dual * g;
      result.vector_laplacian[d] += dual * g2 * b2;
      for (std::size_t derivative = 0; derivative < 3; ++derivative)
        result.vector_jacobian[d][derivative] += dual * g1 * reciprocal[axis][derivative];
    }
  }
  // Reconstruct d^2=|v|^2+sum |a|^2(f^2-g^2), then differentiate its square root.
  double squared = correction, squared_laplacian = correction_laplacian;
  GeometryPosition squared_gradient = correction_gradient;
  for (std::size_t d = 0; d < 3; ++d)
  {
    squared += result.vector[d] * result.vector[d];
    squared_laplacian += 2 * result.vector[d] * result.vector_laplacian[d];
    for (std::size_t derivative = 0; derivative < 3; ++derivative)
    {
      squared_gradient[derivative] += 2 * result.vector[d] * result.vector_jacobian[d][derivative];
      squared_laplacian += 2 * result.vector_jacobian[d][derivative] * result.vector_jacobian[d][derivative];
    }
  }
  if (!std::isfinite(squared) || squared < 0)
    throw std::overflow_error("Polynomial periodic radius is invalid");
  result.radius = std::sqrt(squared);
  if (!result.radius)
  {
    result.differentiable = false;
    return result;
  }
  double gradient_norm = 0;
  for (std::size_t d = 0; d < 3; ++d)
  {
    result.radius_gradient[d] = squared_gradient[d] / (2 * result.radius);
    gradient_norm += result.radius_gradient[d] * result.radius_gradient[d];
  }
  result.radius_laplacian = (squared_laplacian / 2 - gradient_norm) / result.radius;
  return result;
}
}
#endif
