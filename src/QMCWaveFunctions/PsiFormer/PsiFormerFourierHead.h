//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_FOURIER_HEAD_H
#define QMCPLUSPLUS_PSIFORMER_FOURIER_HEAD_H

#include "PsiFormerFourierFactor.h"
#include "PsiFormerGeometry.h"
#include "PsiFormerSpatialLayout.h"
#include <algorithm>
#include <limits>
#include <vector>

namespace qmcplusplus::psiformer
{
inline constexpr std::size_t FOURIER_MODE_LIMIT = 64;

/** Compute the fixed source-image gauge from every physical ion, not a motif.
 * Long-double accumulation avoids avoidable intermediate overflow; the result
 * must still be representable by the stored FP64 Cartesian origin.
 */
inline pf::GeometryPosition physicalIonFourierOrigin(pf::GeometryPositionView ions)
{
  if (!ions.size())
    throw std::invalid_argument("Fourier origin requires a nonempty physical ion source");
  pf::GeometryPosition result{};
  for (std::size_t d = 0; d < 3; ++d)
  {
    long double sum = 0;
    for (std::size_t ion = 0; ion < ions.size(); ++ion)
    {
      const double coordinate = ions(ion, d);
      if (!determinant::isFiniteReal(coordinate))
        throw std::invalid_argument("Fourier physical ion source must be finite");
      sum += static_cast<long double>(coordinate);
    }
    result[d] = static_cast<double>(sum / ions.size());
    if (!determinant::isFiniteReal(result[d]))
      throw std::overflow_error("Fourier physical ion mean is unrepresentable");
  }
  return result;
}

/// Check the declared fixed gauge against the immutable complete source image.
inline void validateFourierOrigin(const pf::GeometryPosition& origin, pf::GeometryPositionView ions)
{
  const auto expected = physicalIonFourierOrigin(ions);
  for (std::size_t d = 0; d < 3; ++d)
    if (!determinant::isFiniteReal(origin[d]) || std::abs(origin[d] - expected[d]) >
        64 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(expected[d])))
      throw std::invalid_argument("Fourier origin disagrees with the complete physical ion mean");
}

/** Prepared simulation-cell Fourier modes shared by all learned full columns.
 * Only Cartesian mode vectors are copied. Coefficients remain borrowed from
 * canonical parameter storage; no mode-by-row or mode-by-spatial-plane cache.
 */
class FourierHeadModes
{
public:
  /// Validate and prepare the bounded simulation modes once, before walker evaluation.
  FourierHeadModes(const pf::GeometryBoundary& simulation,
      const std::vector<std::array<int, 3>>& integer_modes, pf::GeometryPosition origin)
      : origin_(origin)
  {
    if (integer_modes.empty() || integer_modes.size() > FOURIER_MODE_LIMIT ||
        integer_modes.front() != std::array<int, 3>{})
      throw std::invalid_argument("Fourier modes require a bounded nonempty zero-first list");
    pf::PeriodicGeometryMapping identity;
    identity.primitive = identity.simulation = simulation;
    pf::validatePeriodicGeometryMapping(identity);
    for (double coordinate : origin_)
      if (!determinant::isFiniteReal(coordinate))
        throw std::invalid_argument("Fourier fixed origin must be finite");
    const auto& a = simulation.lattice_vectors;
    const auto cross = [](const auto& x, const auto& y) {
      return pf::GeometryPosition{x[1] * y[2] - x[2] * y[1],
          x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]};
    };
    std::array<pf::GeometryPosition, 3> reciprocal{cross(a[1], a[2]), cross(a[2], a[0]), cross(a[0], a[1])};
    double volume = 0;
    for (std::size_t d = 0; d < 3; ++d)
      volume += a[0][d] * reciprocal[0][d];
    const double scale = 2 * std::acos(-1.0) / volume;
    momenta_.reserve(integer_modes.size());
    for (std::size_t mode = 0; mode < integer_modes.size(); ++mode)
    {
      for (std::size_t previous = 0; previous < mode; ++previous)
        if (integer_modes[mode] == integer_modes[previous])
          throw std::invalid_argument("Fourier mode list contains a repeated integer triplet");
      pf::GeometryPosition momentum{};
      // Reciprocal duality uses the simulation cell, independently of EI tiling.
      for (std::size_t d = 0; d < 3; ++d)
      {
        for (std::size_t axis = 0; axis < 3; ++axis)
          momentum[d] += integer_modes[mode][axis] * reciprocal[axis][d] * scale;
        if (!determinant::isFiniteReal(momentum[d]))
          throw std::overflow_error("Fourier Cartesian mode is unrepresentable");
      }
      momenta_.push_back(momentum);
    }
  }

  /// Borrow the prepared Cartesian simulation modes without copying their storage.
  const std::vector<pf::GeometryPosition>& momenta() const noexcept { return momenta_; }

  /// Read the immutable full-physical-source image gauge used by every column.
  const pf::GeometryPosition& origin() const noexcept { return origin_; }

  /// Charge actual mode-vector capacity; the fixed origin belongs to the descriptor object.
  std::size_t storageBytes() const noexcept { return momenta_.capacity() * sizeof(pf::GeometryPosition); }

  /// Read one [mode,determinant,column] coefficient strided channel without allocation.
  FourierFactorJet evaluate(const pf::GeometryPosition& position, const double* coefficients,
      std::size_t coefficient_count, std::size_t channels, std::size_t channel) const
  {
    if (!coefficients || !channels || channel >= channels ||
        channels > std::numeric_limits<std::size_t>::max() / momenta_.size() ||
        coefficient_count != channels * momenta_.size())
      throw std::invalid_argument("Fourier head coefficient view has incompatible channel extent");
    std::array<double, FOURIER_MODE_LIMIT> column{};
    for (std::size_t mode = 0; mode < momenta_.size(); ++mode)
      column[mode] = coefficients[mode * channels + channel];
    return evaluateFourierFactor(position, origin_, momenta_.data(), column.data(), momenta_.size());
  }

private:
  pf::GeometryPosition origin_;
  std::vector<pf::GeometryPosition> momenta_;
};

/** Apply a Fourier full-head product or its ordinary transpose in existing jets.
 * Coefficients are mode-major [mode,determinant,column], separately for each
 * row-spin sector. Only the physical row carries factor derivatives; the
 * learned affine head may depend on every electron. No per-element tape is kept.
 */
inline void transformFourierHeadJets(const FourierHeadModes& modes, const SpatialJetLayout& layout,
    std::size_t determinants, std::size_t spin_up, const double* positions, std::size_t position_count,
    const std::array<const double*, 2>& coefficients, const std::array<std::size_t, 2>& coefficient_counts,
    FourierFactorJet::Complex* jets, std::size_t jet_count, bool transpose, std::size_t active_electron = 0)
{
  validateSpatialJetLayout(layout);
  const auto n = layout.electron_count;
  const auto channels = spatial_detail::checkedProduct(determinants, n, "Fourier head channel extent overflow");
  const auto elements = spatial_detail::checkedProduct(channels, n, "Fourier head matrix extent overflow");
  const auto coordinates = spatial_detail::checkedProduct(layout.configuration_count,
      spatial_detail::checkedProduct(n, 3, "Fourier coordinate extent overflow"), "Fourier coordinate extent overflow");
  const auto coefficient_count = spatial_detail::checkedProduct(channels, modes.momenta().size(),
      "Fourier coefficient extent overflow");
  if (!determinants || !spin_up || spin_up >= n || layout.element_count != elements || !positions ||
      position_count != coordinates || !jets || jet_count != spatialJetSpanElements(layout) ||
      active_electron >= n || !coefficients[0] || !coefficients[1] ||
      coefficient_counts[0] != coefficient_count || coefficient_counts[1] != coefficient_count)
    throw std::invalid_argument("Fourier full-head transform has incompatible views");
  using C = FourierFactorJet::Complex;
  for (std::size_t configuration = 0; configuration < layout.configuration_count; ++configuration)
    for (std::size_t determinant = 0; determinant < determinants; ++determinant)
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t column = 0; column < n; ++column)
        {
          const auto element = (determinant * n + row) * n + column;
          const auto spin = row >= spin_up;
          const auto factor = modes.evaluate({positions[(configuration * n + row) * 3],
              positions[(configuration * n + row) * 3 + 1], positions[(configuration * n + row) * 3 + 2]},
              coefficients[spin], coefficient_counts[spin], channels, determinant * n + column);
          const auto value_offset = layout.uncheckedValueOffset(configuration, element);
          const C value = jets[value_offset];
          C result_value = factor.value * value;
          if (transpose)
          {
            // Consume all old derivative seeds before overwriting their planes.
            for (std::size_t lane = 0; lane < layout.gradient_lanes; ++lane)
            {
              const auto target = layout.mode == SpatialJetMode::ACTIVE ? active_electron :
                  layout.physicalTargetElectron(lane / 3);
              if (target == row)
                result_value += factor.gradient[lane % 3] *
                    jets[layout.uncheckedGradientOffset(configuration, lane, element)];
            }
            for (std::size_t target = 0; target < layout.laplacian_lanes; ++target)
              if (layout.physicalTargetElectron(target) == row)
                result_value += factor.laplacian * jets[layout.uncheckedLaplacianOffset(configuration, target, element)];
            for (std::size_t lane = 0; lane < layout.gradient_lanes; ++lane)
            {
              const auto target = layout.mode == SpatialJetMode::ACTIVE ? active_electron :
                  layout.physicalTargetElectron(lane / 3);
              const auto offset = layout.uncheckedGradientOffset(configuration, lane, element);
              C result = factor.value * jets[offset];
              if (target == row && layout.laplacian_lanes)
                result += 2.0 * factor.gradient[lane % 3] * jets[layout.uncheckedLaplacianOffset(
                    configuration, lane / 3, element)];
              jets[offset] = result;
            }
            for (std::size_t target = 0; target < layout.laplacian_lanes; ++target)
              jets[layout.uncheckedLaplacianOffset(configuration, target, element)] *= factor.value;
          }
          else
          {
            // Laplacians read the original gradient planes before the product update.
            for (std::size_t target = 0; target < layout.laplacian_lanes; ++target)
            {
              const auto offset = layout.uncheckedLaplacianOffset(configuration, target, element);
              C result = factor.value * jets[offset];
              if (layout.physicalTargetElectron(target) == row)
              {
                result += factor.laplacian * value;
                for (std::size_t d = 0; d < 3; ++d)
                  result += 2.0 * factor.gradient[d] *
                      jets[layout.uncheckedGradientOffset(configuration, target * 3 + d, element)];
              }
              jets[offset] = result;
            }
            for (std::size_t lane = 0; lane < layout.gradient_lanes; ++lane)
            {
              const auto target = layout.mode == SpatialJetMode::ACTIVE ? active_electron :
                  layout.physicalTargetElectron(lane / 3);
              const auto offset = layout.uncheckedGradientOffset(configuration, lane, element);
              jets[offset] = factor.value * jets[offset] + (target == row ? factor.gradient[lane % 3] * value : C{});
            }
          }
          jets[value_offset] = result_value;
        }
}

/** Accumulate canonical real-coefficient responses from unmultiplied affine jets.
 * Seeds use ordinary complex transpose convention. A separately pulled-back
 * occupied phase can be included in the seeds: scalar-factor jet products
 * commute. This avoids reconstructing a base head by division through F.
 */
inline void accumulateFourierCoefficientResponses(const FourierHeadModes& modes, const SpatialJetLayout& layout,
    std::size_t determinants, std::size_t spin_up, const double* positions, std::size_t position_count,
    const FourierFactorJet::Complex* base, const FourierFactorJet::Complex* seeds, std::size_t jet_count,
    const std::array<FourierFactorJet::Complex*, 2>& responses,
    const std::array<std::size_t, 2>& response_counts, std::size_t active_electron = 0)
{
  validateSpatialJetLayout(layout);
  const auto n = layout.electron_count;
  const auto channels = spatial_detail::checkedProduct(determinants, n, "Fourier response channel extent overflow");
  const auto count = spatial_detail::checkedProduct(channels, modes.momenta().size(), "Fourier response extent overflow");
  if (!determinants || !spin_up || spin_up >= n || layout.element_count !=
      spatial_detail::checkedProduct(channels, n, "Fourier response matrix extent overflow") || !positions ||
      position_count != spatial_detail::checkedProduct(layout.configuration_count,
          spatial_detail::checkedProduct(n, 3, "Fourier response coordinate extent overflow"),
          "Fourier response coordinate extent overflow") || !base || !seeds ||
      jet_count != spatialJetSpanElements(layout) || active_electron >= n ||
      !responses[0] || !responses[1] || response_counts[0] != count || response_counts[1] != count)
    throw std::invalid_argument("Fourier coefficient response has incompatible full-head views");
  using C = FourierFactorJet::Complex;
  for (std::size_t configuration = 0; configuration < layout.configuration_count; ++configuration)
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t mode = 0; mode < modes.momenta().size(); ++mode)
      {
        const auto direction = fourierModeJet({positions[(configuration * n + row) * 3],
            positions[(configuration * n + row) * 3 + 1], positions[(configuration * n + row) * 3 + 2]},
            modes.origin(), modes.momenta()[mode]);
        for (std::size_t determinant = 0; determinant < determinants; ++determinant)
          for (std::size_t column = 0; column < n; ++column)
          {
            const auto element = (determinant * n + row) * n + column;
            const auto value_offset = layout.uncheckedValueOffset(configuration, element);
            const C value = base[value_offset];
            C response = seeds[value_offset] * direction.value * value;
            for (std::size_t lane = 0; lane < layout.gradient_lanes; ++lane)
            {
              const auto target = layout.mode == SpatialJetMode::ACTIVE ? active_electron :
                  layout.physicalTargetElectron(lane / 3);
              const auto offset = layout.uncheckedGradientOffset(configuration, lane, element);
              response += seeds[offset] * (direction.value * base[offset] +
                  (target == row ? direction.gradient[lane % 3] * value : C{}));
            }
            for (std::size_t target = 0; target < layout.laplacian_lanes; ++target)
            {
              const auto offset = layout.uncheckedLaplacianOffset(configuration, target, element);
              C laplacian = direction.value * base[offset];
              if (layout.physicalTargetElectron(target) == row)
              {
                laplacian += direction.laplacian * value;
                for (std::size_t d = 0; d < 3; ++d)
                  laplacian += 2.0 * direction.gradient[d] *
                      base[layout.uncheckedGradientOffset(configuration, target * 3 + d, element)];
              }
              response += seeds[offset] * laplacian;
            }
            responses[row >= spin_up][mode * channels + determinant * n + column] += response;
          }
      }
}
} // namespace qmcplusplus::psiformer
#endif
