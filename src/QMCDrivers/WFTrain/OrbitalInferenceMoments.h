//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_ORBITAL_INFERENCE_MOMENTS_H
#define QMCPLUSPLUS_ORBITAL_INFERENCE_MOMENTS_H

#include "QMCDrivers/WFTrain/TrainingNumerics.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace qmcplusplus::wftrain
{
/// Block diagnostics on sampled configurations, not physical orbital overlaps.
struct OrbitalInferenceResult
{
  double samples = 0, mean_mse = 0, mse_variance = 0;
  double occupied_column_cosine_squared = std::numeric_limits<double>::quiet_NaN();
  std::size_t undefined_columns = 0;
  double energy = 0, energy_squared = 0;
};

/** Pool raw dot/norm sums before normalization, including across MPI partitions.
 * MSE includes the full matrix's off-spin zeros; cosine uses occupied spin columns.
 */
class OrbitalInferenceMoments
{
public:
  /// Freeze full-matrix dimensions and whether genuine local energies are present.
  OrbitalInferenceMoments(std::size_t up, std::size_t down, std::size_t determinants, bool energy)
      : up_(up), down_(down), determinants_(determinants), energy_(energy)
  {
    if (!up || !down || !determinants || up > std::numeric_limits<std::size_t>::max() - down)
      throw std::invalid_argument("Invalid orbital inference dimensions");
    const auto n = up + down;
    if (n > std::numeric_limits<std::size_t>::max() / n ||
        determinants > std::numeric_limits<std::size_t>::max() / (n * n) ||
        determinants * n > (std::numeric_limits<std::size_t>::max() - 5) / 3)
      throw std::overflow_error("Orbital inference dimensions overflow");
    sums_.resize(5 + 3 * determinants * n, 0.0);
  }

  /// Add one full D x Ne x Ne student/target pair, without sample weights.
  void addSample(const std::vector<double>& student, const std::vector<double>& target,
                 std::optional<double> energy = std::nullopt)
  {
    const auto n = up_ + down_;
    if (student.size() != determinants_ * n * n || target.size() != student.size() ||
        energy.has_value() != energy_ || (energy && !isFiniteTrainingReal(*energy)))
      throw std::invalid_argument("Incompatible orbital inference sample");
    for (std::size_t i = 0; i < student.size(); ++i)
      if (!isFiniteTrainingReal(student[i]) || !isFiniteTrainingReal(target[i]))
        throw std::invalid_argument("Nonfinite orbital inference value");
    double loss = 0.0;
    for (std::size_t d = 0; d < determinants_; ++d)
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t col = 0; col < n; ++col)
        {
          const auto i = (d * n + row) * n + col;
          const double error = student[i] - target[i];
          loss += error * error / (determinants_ * (row < up_ ? up_ : down_) * n);
          if ((row < up_) == (col < up_))
          {
            const auto j = 5 + 3 * (d * n + col);
            sums_[j] += student[i] * target[i];
            sums_[j + 1] += student[i] * student[i];
            sums_[j + 2] += target[i] * target[i];
          }
        }
    sums_[0] += 1;
    sums_[1] += loss;
    sums_[2] += loss * loss;
    if (energy)
    {
      sums_[3] += *energy;
      sums_[4] += *energy * *energy;
    }
  }

  /// Return reducible raw statistics; no per-rank cosine or loss normalization.
  const std::vector<double>& sums() const noexcept { return sums_; }

  /// Normalize one globally summed block and count zero-norm columns explicitly.
  OrbitalInferenceResult summarize(const std::vector<double>& global) const
  {
    if (global.size() != sums_.size() || global[0] <= 0)
      throw std::invalid_argument("Empty or incompatible orbital inference block");
    for (double value : global)
      if (!isFiniteTrainingReal(value))
        throw std::invalid_argument("Nonfinite orbital inference moments");
    OrbitalInferenceResult result;
    result.samples = global[0];
    result.mean_mse = global[1] / global[0];
    result.mse_variance = std::max(0.0, global[2] / global[0] - result.mean_mse * result.mean_mse);
    result.energy = global[3] / global[0];
    result.energy_squared = global[4] / global[0];
    double cosine = 0;
    const auto columns = determinants_ * (up_ + down_);
    for (std::size_t col = 0; col < columns; ++col)
    {
      const auto j = 5 + 3 * col;
      if (global[j + 1] == 0 || global[j + 2] == 0)
        ++result.undefined_columns;
      else
      {
        // Divide norms separately to avoid overflow in their product.
        const double normalized = global[j] / std::sqrt(global[j + 1]) / std::sqrt(global[j + 2]);
        const double square = normalized * normalized;
        if (!isFiniteTrainingReal(square) || square > 1.0 + 64 * std::numeric_limits<double>::epsilon())
          throw std::invalid_argument("Invalid orbital inference dot/norm ratio");
        cosine += std::min(1.0, square);
      }
    }
    if (columns != result.undefined_columns)
      result.occupied_column_cosine_squared = cosine / (columns - result.undefined_columns);
    return result;
  }

  /// Start another independent output block with the same immutable dimensions.
  void reset() noexcept { std::fill(sums_.begin(), sums_.end(), 0.0); }

private:
  std::size_t up_, down_, determinants_;
  bool energy_;
  std::vector<double> sums_;
};
}
#endif
