//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_OCCUPIED_HEAD_H
#define QMCPLUSPLUS_PSIFORMER_OCCUPIED_HEAD_H

#include "PsiFormerComplexOrbital.h"
#include "PsiFormerExecutionPlan.h"

namespace qmcplusplus::psiformer
{
/// Count prepared Fourier vectors only for the explicit replacement-envelope option.
inline std::size_t fourierModeCount(const PsiFormerExecutionPlan& plan)
{
  const auto& profile = plan.environment().periodic_profile;
  return profile && profile->envelope_policy == PeriodicEnvelopePolicy::FOURIER ? profile->fourier_modes.size() : 0;
}

/// Count only nonzero-G descriptor storage, preserving Gamma's legacy allocation.
inline std::size_t occupiedMomentumCount(const PsiFormerExecutionPlan& plan)
{
  const auto& metadata = plan.environment().periodic_profile;
  if (!metadata || std::none_of(metadata->occupied_columns.begin(), metadata->occupied_columns.end(),
      [](const auto& column) { return column.reciprocal != std::array<int, 3>{}; }))
    return 0;
  const auto& shape = plan.modelShape();
  const std::size_t sectors = (shape.spin_up_electrons != 0) + (shape.spin_down_electrons != 0);
  return spatial_detail::checkedProduct(spatial_detail::checkedProduct(
      sectors, shape.determinants, "occupied momentum count overflow"), shape.electrons(),
      "occupied momentum count overflow");
}

/** Translate validated occupied integer triplets into the full-head column gauge.
 * Both row-spin sectors use the same ordered N occupied columns in every
 * determinant. Band/spin labels identify columns; they do not reorder or mask
 * the learned full matrices. Only simulation-cell G enters this inner phase.
 */
inline ComplexOrbitalHeadDescriptor occupiedFullHeadDescriptor(const PsiFormerExecutionPlan& plan)
{
  const auto& shape = plan.modelShape();
  ComplexOrbitalHeadDescriptor result{shape.electrons(), shape.spin_up_electrons,
      shape.determinants, shape.feature_dimension, ComplexOrbitalStructure::FULL,
      ComplexOrbitalPhase::NONE, {}};
  const auto& metadata = plan.environment().periodic_profile;
  const auto count = occupiedMomentumCount(plan);
  if (fourierModeCount(plan))
    result.fourier_modes.emplace(metadata->cells.simulation, metadata->fourier_modes, metadata->fourier_origin);
  if (!count)
    return result;
  const auto& a = metadata->cells.simulation.lattice_vectors;
  const auto cross = [](const auto& x, const auto& y) {
    return std::array<double, 3>{x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]};
  };
  std::array<std::array<double, 3>, 3> reciprocal{cross(a[1], a[2]), cross(a[2], a[0]), cross(a[0], a[1])};
  double volume = 0;
  for (std::size_t d = 0; d < 3; ++d)
    volume += a[0][d] * reciprocal[0][d];
  const double scale = 2 * std::acos(-1.0) / volume;
  for (auto& row : reciprocal)
    for (double& component : row)
      component *= scale;
  result.reciprocal_momenta.reserve(count);
  for (std::size_t spin = 0; spin < 2; ++spin)
    if (spin ? shape.spin_down_electrons : shape.spin_up_electrons)
      for (std::size_t channel = 0; channel < shape.determinants; ++channel)
        for (const auto& column : metadata->occupied_columns)
        {
          std::array<double, 3> momentum{};
          for (std::size_t d = 0; d < 3; ++d)
            for (std::size_t axis = 0; axis < 3; ++axis)
              momentum[d] += column.reciprocal[axis] * reciprocal[axis][d];
          for (double component : momentum)
            if (!determinant::isFiniteReal(component))
              throw std::overflow_error("occupied reciprocal momentum is nonfinite");
          result.reciprocal_momenta.push_back(momentum);
        }
  result.phase = ComplexOrbitalPhase::RECIPROCAL_ONLY;
  return result;
}

/** Pull ordinary complex cotangents through exp(iG.r_row), in existing storage.
 * This is a transpose product, not a Hermitian product. Value seeds absorb
 * phase-gradient and phase-Laplacian terms before those seed planes change;
 * gradient seeds then absorb the two Laplacian cross terms. No jets are saved.
 */
inline void pullBackOccupiedPhase(const ComplexOrbitalHeadDescriptor& descriptor,
    const SpatialJetLayout& layout, const double* positions, std::size_t position_count,
    determinant::Complex* adjoints, std::size_t adjoint_count, std::size_t active_electron = 0)
{
  if (descriptor.phase == ComplexOrbitalPhase::NONE)
    return;
  validateSpatialJetLayout(layout);
  const auto n = descriptor.electrons;
  const auto matrix_count = spatial_detail::checkedProduct(descriptor.determinants,
      spatial_detail::checkedProduct(n, n, "occupied phase matrix extent overflow"),
      "occupied phase matrix extent overflow");
  if (descriptor.phase != ComplexOrbitalPhase::RECIPROCAL_ONLY || descriptor.spin_up > n ||
      descriptor.structure != ComplexOrbitalStructure::FULL || layout.element_count != matrix_count ||
      position_count != spatial_detail::checkedProduct(layout.configuration_count,
          spatial_detail::checkedProduct(n, 3, "occupied phase coordinate extent overflow"),
          "occupied phase coordinate extent overflow") ||
      adjoint_count != spatialJetSpanElements(layout) || !positions || !adjoints || active_electron >= n)
    throw std::invalid_argument("occupied phase pullback has incompatible full-head views");
  using C = determinant::Complex;
  const auto sector_count = spatial_detail::checkedProduct(descriptor.determinants, n,
                                                           "occupied phase momentum extent overflow");
  const auto expected = spatial_detail::checkedProduct(sector_count,
      (descriptor.spin_up != 0) + (descriptor.spin_up != n), "occupied phase momentum extent overflow");
  if (descriptor.reciprocal_momenta.size() != expected)
    throw std::invalid_argument("occupied phase pullback has incomplete reciprocal metadata");
  for (std::size_t configuration = 0; configuration < layout.configuration_count; ++configuration)
    for (std::size_t channel = 0; channel < descriptor.determinants; ++channel)
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t column = 0; column < n; ++column)
        {
          const auto element = (channel * n + row) * n + column;
          const auto& g = descriptor.reciprocal_momenta[
              (row >= descriptor.spin_up && descriptor.spin_up != 0 ? sector_count : 0) + channel * n + column];
          double argument = 0;
          for (std::size_t d = 0; d < 3; ++d)
            argument += g[d] * positions[(configuration * n + row) * 3 + d];
          const C phase = std::polar(1.0, argument);
          C value = adjoints[layout.uncheckedValueOffset(configuration, element)];
          C laplacian{};
          for (std::size_t target = 0; target < layout.laplacian_lanes; ++target)
            if (layout.physicalTargetElectron(target) == row)
              laplacian = adjoints[layout.uncheckedLaplacianOffset(configuration, target, element)];
          for (std::size_t d = 0; d < 3; ++d)
            value -= g[d] * g[d] * laplacian;
          for (std::size_t lane = 0; lane < layout.gradient_lanes; ++lane)
          {
            const auto target = layout.mode == SpatialJetMode::ACTIVE ? active_electron :
                layout.physicalTargetElectron(lane / 3);
            const auto offset = layout.uncheckedGradientOffset(configuration, lane, element);
            if (target == row)
            {
              value += C{0, g[lane % 3]} * adjoints[offset];
              adjoints[offset] = phase * (adjoints[offset] + C{0, 2 * g[lane % 3]} * laplacian);
            }
            else
              adjoints[offset] *= phase;
          }
          adjoints[layout.uncheckedValueOffset(configuration, element)] = phase * value;
          for (std::size_t target = 0; target < layout.laplacian_lanes; ++target)
            adjoints[layout.uncheckedLaplacianOffset(configuration, target, element)] *= phase;
        }
}
}
#endif
