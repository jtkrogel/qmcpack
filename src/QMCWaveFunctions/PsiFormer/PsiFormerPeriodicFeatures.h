//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_PERIODIC_FEATURES_H
#define QMCPLUSPLUS_PSIFORMER_PERIODIC_FEATURES_H

#include "PsiFormerExecutionPlan.h"
#include <stdexcept>
#include <algorithm>

namespace qmcplusplus::psiformer::periodic_feature
{
/// One ordered displacement's eight features and jets with respect to that displacement.
struct Jet
{
  std::array<double, 8> value{};
  std::array<pf::GeometryPosition, 8> gradient{};
  std::array<double, 8> laplacian{};
  bool differentiable = false;
};

/// Assemble raw or logarithmically rescaled features from one existing pair-cache entry.
inline Jet evaluate(const pf::GeometryPairTable& table, std::size_t pair,
                    const pf::GeometryBoundary& cell, int orientation,
                    PeriodicFeatureTransform transform, double spin_product)
{
  if (pair >= table.size() || (orientation != 1 && orientation != -1) ||
      cell.kind != pf::GeometryBoundaryKind::PERIODIC || (spin_product != 1 && spin_product != -1))
    throw std::invalid_argument("Periodic features require an oriented periodic pair");
  if (transform != PeriodicFeatureTransform::LOG_COMPLEMENT_V1 &&
      transform != PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1)
    throw std::invalid_argument("Unknown periodic feature transform");
  Jet result;
  result.value[7] = spin_product;
  const double radius = table.distances()[pair];
  result.differentiable = radius != 0;
  const auto& radial = table.softenedRadialFactors()[pair];
  const auto& rg = table.distanceGradients()[pair];
  const double rl = table.distanceLaplacians()[pair];
  const double rg2 = table.distanceGradientNormsSquared()[pair];
  const bool raw = transform == PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1;
  result.value[0] = raw ? radius : radial.log1p_radius;
  result.laplacian[0] = raw ? rl : radial.log1p_first * rl + radial.log1p_second * rg2;
  for (std::size_t d = 0; d < 3; ++d)
    result.gradient[0][d] = orientation * rg[d] * (raw ? 1 : radial.log1p_first);
  constexpr double two_pi = 6.283185307179586476925286766559;
  for (std::size_t vector = 0; vector < 2; ++vector)
    for (std::size_t component = 0; component < 3; ++component)
    {
      const auto feature = 1 + 3 * vector + component;
      // Reversing a pair changes q and its Laplacian, but p and its Laplacian are even.
      const double parity = vector == 0 ? orientation : 1;
      double value = parity * (vector == 0 ? table.displacements()[pair][component] :
                                             table.complementaryDisplacements()[pair][component]);
      double laplacian = parity * (vector == 0 ? table.displacementLaplacians()[pair][component] :
                                                  table.complementaryDisplacementLaplacians()[pair][component]);
      pf::GeometryPosition gradient{};
      for (std::size_t d = 0; d < 3; ++d)
        gradient[d] = (vector == 0 ? 1 : orientation) *
            (vector == 0 ? table.displacementJacobians()[pair][component][d] :
                           table.complementaryDisplacementJacobians()[pair][component][d]);
      if (raw && vector == 1)
      {
        double constant = 0;
        for (std::size_t axis = 0; axis < 3; ++axis)
          constant += cell.lattice_vectors[axis][component] / two_pi;
        value = constant - value;
        laplacian = -laplacian;
        for (double& derivative : gradient)
          derivative = -derivative;
      }
      result.value[feature] = raw ? value : value * radial.log1p_over_radius;
      if (raw)
      {
        result.gradient[feature] = gradient;
        result.laplacian[feature] = laplacian;
      }
      else
      {
        double contraction = 0;
        for (std::size_t d = 0; d < 3; ++d)
        {
          const double radius_gradient = orientation * rg[d];
          result.gradient[feature][d] = gradient[d] * radial.log1p_over_radius +
              value * radial.log1p_over_radius_first * radius_gradient;
          contraction += gradient[d] * radius_gradient;
        }
        result.laplacian[feature] = laplacian * radial.log1p_over_radius +
            2 * radial.log1p_over_radius_first * contraction + value *
            (radial.log1p_over_radius_first * rl + radial.log1p_over_radius_second * rg2);
      }
    }
  return result;
}

/// Build one cache with feature-only centers while keeping legacy allocation ownership.
inline pf::PsiFormerGeometryCache makeGeometry(std::size_t electrons, pf::GeometryPositionView nuclei,
    const pf::GeometryBoundary& boundary, const std::shared_ptr<const PeriodicProfileMetadata>& profile)
{
  if (!profile)
    return {electrons, nuclei, boundary};
  if (nuclei.size() != profile->primitive_motif.size())
    throw std::invalid_argument("Periodic feature centers disagree with profile motif");
  for (std::size_t ion = 0; ion < nuclei.size(); ++ion)
    if (nuclei.position(ion) != profile->primitive_motif[ion])
      throw std::invalid_argument("Periodic feature center coordinates disagree with profile motif");
  return {electrons, nuclei, profile->cells};
}

/// Fill head-major additive logits, with zero self bias and one stack-local pair jet.
inline void fillAttentionLogits(const pf::PsiFormerGeometryCache& geometry, std::size_t spin_up,
    PeriodicFeatureTransform transform, const double* weights, const double* bias,
    std::size_t heads, double* logits)
{
  const auto n = geometry.electronCount();
  std::fill_n(logits, heads * n * n, 0.0);
  for (std::size_t pair = 0; pair < geometry.electronPairs().size(); ++pair)
  {
    const auto& identity = geometry.electronPairs()[pair];
    const double spin = (identity.first < spin_up) == (identity.second < spin_up) ? 1 : -1;
    for (int orientation : {1, -1})
    {
      const auto jet = evaluate(geometry.electronElectronPairs(), pair, geometry.boundary(), orientation, transform, spin);
      const auto row = orientation == 1 ? identity.first : identity.second;
      const auto column = orientation == 1 ? identity.second : identity.first;
      for (std::size_t head = 0; head < heads; ++head)
      {
        double value = bias[head];
        for (std::size_t f = 0; f < 8; ++f)
          value += jet.value[f] * weights[f * heads + head];
        logits[(head * n + row) * n + column] = value;
      }
    }
  }
}
} // namespace qmcplusplus::psiformer::periodic_feature

#endif
