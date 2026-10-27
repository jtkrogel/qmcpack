//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_RESIDUAL_PAIR_H
#define QMCPLUSPLUS_PSIFORMER_RESIDUAL_PAIR_H

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace qmcplusplus::psiformer
{
/// Preserve rejection of nonfinite proposals under finite-math compiler policies.
inline bool residualPairFinite(double value) noexcept
{
  std::uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

/// Immutable contract for the exact-zero, cusp-free pair correction.
struct ResidualPairMetadata
{
  unsigned version = 1;
  double same_reference = 1;
  double opposite_reference = 1;
  double cutoff = 0;
};

/// Value, spatial radial jets and log-length score; first/r has a finite coalescence limit.
struct ResidualPairJet
{
  double value, radial_first, radial_second, radial_first_over_r, score;
};

/// Evaluate the residual and its radial derivatives without subtracting rounded cusps.
inline ResidualPairJet residualPairJet(double r, double lambda, double reference,
                                       double cutoff, double factor)
{
  if (!residualPairFinite(r) || r < 0 || !residualPairFinite(lambda) ||
      !residualPairFinite(reference) || reference <= 0 || !residualPairFinite(cutoff) || cutoff <= 0)
    throw std::invalid_argument("Invalid residual pair inputs");
  const double alpha = reference * std::exp(lambda);
  const double delta = reference * std::expm1(lambda);
  if (!residualPairFinite(alpha) || alpha <= 0 || !residualPairFinite(delta))
    throw std::overflow_error("Residual pair length is not finite and positive");
  if (r >= cutoff)
    return {0, 0, 0, 0, 0};

  // Factor every spatial derivative by delta, preserving exact zero at lambda=0.
  const double ar = alpha + r, br = reference + r;
  const double denominator = ar * br;
  const double ab = alpha * reference, sum = alpha + reference;
  const double q = -factor * delta * (ab + r * sum) / denominator;
  const double qr = factor * delta * (2 * ab + sum * r) / (denominator * denominator);
  const double q1 = r * qr;
  const double q2 = -2 * factor * delta *
      (-ab * ab + 3 * ab * r * r + sum * r * r * r) /
      (denominator * denominator * denominator);

  // The polynomial cutoff and its first two derivatives vanish at the boundary.
  const double inv_c2 = 1 / (cutoff * cutoff);
  const double u = 1 - r * r * inv_c2;
  const double chi = u * u * u;
  const double chi1r = -6 * inv_c2 * u * u;
  const double chi1 = r * chi1r;
  const double chi2 = chi1r + 24 * r * r * inv_c2 * inv_c2 * u;
  const double score = -factor * alpha * (alpha / ar) * ((alpha + 2 * r) / ar);
  ResidualPairJet result{chi * q, chi1 * q + chi * q1,
      chi2 * q + 2 * chi1 * q1 + chi * q2, chi1r * q + chi * qr, chi * score};
  for (double value : {result.value, result.radial_first, result.radial_second,
                       result.radial_first_over_r, result.score})
    if (!residualPairFinite(value))
      throw std::overflow_error("Residual pair evaluation is not finite");
  return result;
}

/// Fixed lattice and reciprocal face bounds for a compact minimum-image factor.
class ResidualPairGeometry
{
public:
  using Position = std::array<double, 3>;
  using Lattice = std::array<Position, 3>;
  /// Validate reference scales and the cutoff before model/workspace allocation.
  ResidualPairGeometry(ResidualPairMetadata metadata, Lattice lattice)
      : metadata_(metadata), lattice_(lattice)
  {
    if (metadata.version != 1 || !residualPairFinite(metadata.same_reference) || metadata.same_reference <= 0 ||
        !residualPairFinite(metadata.opposite_reference) || metadata.opposite_reference <= 0)
      throw std::invalid_argument("Invalid residual pair metadata");

    // Reciprocal face distances give a conservative unique-image sphere even for skew cells.
    auto cross = [](const Position& a, const Position& b) {
      return Position{a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
    };
    reciprocal_ = {cross(lattice[1], lattice[2]), cross(lattice[2], lattice[0]), cross(lattice[0], lattice[1])};
    double volume = 0;
    for (int d = 0; d < 3; ++d) volume += lattice[0][d]*reciprocal_[0][d];
    if (!residualPairFinite(volume) || volume == 0)
      throw std::invalid_argument("Residual pair lattice is singular");
    double safe_radius = INFINITY;
    for (auto& row : reciprocal_)
    {
      double norm = 0;
      for (double& x : row) { x /= volume; norm += x*x; }
      const double bound = .5/std::sqrt(norm);
      if (bound < safe_radius) safe_radius = bound;
    }
    if (!residualPairFinite(metadata.cutoff) || metadata.cutoff <= 0 || metadata.cutoff >= safe_radius)
      throw std::invalid_argument("Residual pair cutoff must be strictly inside the reciprocal-face sphere");
  }

  /// Wrap fractional coordinates; the active cutoff lies strictly inside every image boundary.
  Position displacement(const Position& first, const Position& second) const
  {
    Position difference{}, fractions{}, result{};
    for (int d = 0; d < 3; ++d) difference[d] = first[d]-second[d];
    for (int a = 0; a < 3; ++a)
    {
      for (int d = 0; d < 3; ++d) fractions[a] += reciprocal_[a][d]*difference[d];
      fractions[a] -= std::nearbyint(fractions[a]);
      for (int d = 0; d < 3; ++d) result[d] += fractions[a]*lattice_[a][d];
    }
    return result;
  }

  /// Evaluate one physical pair using immutable references and Cartesian minimum-image radius.
  ResidualPairJet evaluate(const Position& displacement, double lambda, bool same_spin) const
  {
    double squared = 0;
    for (double x : displacement) squared += x*x;
    return residualPairJet(std::sqrt(squared), lambda,
        same_spin ? metadata_.same_reference : metadata_.opposite_reference,
        metadata_.cutoff, same_spin ? .25 : .5);
  }

  /// Expose fixed model metadata for persistence and identity checks.
  const ResidualPairMetadata& metadata() const { return metadata_; }

private:
  /// Immutable correlation policy and fixed direct/reciprocal simulation cell.
  ResidualPairMetadata metadata_;
  Lattice lattice_, reciprocal_;
};
} // namespace qmcplusplus::psiformer
#endif
