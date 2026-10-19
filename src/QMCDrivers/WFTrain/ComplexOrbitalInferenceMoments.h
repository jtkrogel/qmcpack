//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_COMPLEX_ORBITAL_INFERENCE_MOMENTS_H
#define QMCPLUSPLUS_COMPLEX_ORBITAL_INFERENCE_MOMENTS_H

#include "QMCDrivers/WFTrain/OrbitalInferenceMoments.h"
#include <complex>

namespace qmcplusplus::wftrain
{
/** Pool complex full-matrix errors and occupied-column Hermitian dot/norm sums.
 * These are sampled diagnostics, not physical orbital overlaps or fitted gauges.
 */
class ComplexOrbitalInferenceMoments
{
public:
  /// Freeze the matrix convention and allocate only O(D Ne) reducible statistics.
  ComplexOrbitalInferenceMoments(std::size_t up, std::size_t down, std::size_t determinants, bool energy)
      : up_(up), down_(down), determinants_(determinants), energy_(energy)
  {
    if (!up || !down || !determinants || up > std::numeric_limits<std::size_t>::max() - down)
      throw std::invalid_argument("Invalid complex orbital inference dimensions");
    const auto n = up + down;
    if (n > std::numeric_limits<std::size_t>::max() / n ||
        determinants > std::numeric_limits<std::size_t>::max() / (n * n) ||
        determinants * n > (std::numeric_limits<std::size_t>::max() - 5) / 4)
      throw std::overflow_error("Complex orbital inference dimensions overflow");
    sums_.resize(5 + 4 * determinants * n, 0.0);
  }

  /// Accumulate one unweighted D x Ne x Ne pair in its fixed occupied-column gauge.
  void addSample(const std::vector<std::complex<double>>& student,
      const std::vector<std::complex<double>>& target, std::optional<double> energy = std::nullopt)
  {
    const auto n = up_ + down_;
    if (student.size() != determinants_ * n * n || target.size() != student.size() ||
        energy.has_value() != energy_ || (energy && !isFiniteTrainingReal(*energy)))
      throw std::invalid_argument("Incompatible complex orbital inference sample");
    for (std::size_t i = 0; i < student.size(); ++i)
      if (!isFiniteTrainingReal(student[i].real()) || !isFiniteTrainingReal(student[i].imag()) ||
          !isFiniteTrainingReal(target[i].real()) || !isFiniteTrainingReal(target[i].imag()))
        throw std::invalid_argument("Nonfinite complex orbital inference value");
    double loss = 0;
    for (std::size_t d = 0; d < determinants_; ++d)
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t col = 0; col < n; ++col)
        {
          const auto i = (d * n + row) * n + col;
          loss += std::norm(student[i] - target[i]) / (determinants_ * (row < up_ ? up_ : down_) * n);
          if ((row < up_) == (col < up_))
          {
            const auto j = 5 + 4 * (d * n + col);
            const auto product = std::conj(student[i]) * target[i];
            sums_[j] += product.real();
            sums_[j + 1] += product.imag();
            sums_[j + 2] += std::norm(student[i]);
            sums_[j + 3] += std::norm(target[i]);
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

  /// Expose raw real statistics for an ordinary sum reduction across ranks.
  const std::vector<double>& sums() const noexcept { return sums_; }

  /// Normalize only the pooled complex dot and norm sums, never rank-local cosines.
  OrbitalInferenceResult summarize(const std::vector<double>& global) const
  {
    if (global.size() != sums_.size() || global[0] <= 0)
      throw std::invalid_argument("Empty or incompatible complex orbital inference block");
    for (double value : global)
      if (!isFiniteTrainingReal(value))
        throw std::invalid_argument("Nonfinite complex orbital inference moments");
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
      const auto j = 5 + 4 * col;
      if (global[j + 2] < 0 || global[j + 3] < 0)
        throw std::invalid_argument("Negative complex orbital inference norm");
      if (global[j + 2] == 0 || global[j + 3] == 0)
        ++result.undefined_columns;
      else
      {
        const double real = global[j] / std::sqrt(global[j + 2]) / std::sqrt(global[j + 3]);
        const double imag = global[j + 1] / std::sqrt(global[j + 2]) / std::sqrt(global[j + 3]);
        const double square = real * real + imag * imag;
        if (!isFiniteTrainingReal(square) || square > 1 + 64 * std::numeric_limits<double>::epsilon())
          throw std::invalid_argument("Invalid complex orbital inference dot/norm ratio");
        cosine += std::min(1.0, square);
      }
    }
    if (columns != result.undefined_columns)
      result.occupied_column_cosine_squared = cosine / (columns - result.undefined_columns);
    return result;
  }

  /// Reuse the same bounded statistics for the next independent observation block.
  void reset() noexcept { std::fill(sums_.begin(), sums_.end(), 0.0); }

private:
  std::size_t up_, down_, determinants_;
  bool energy_;
  std::vector<double> sums_;
};
}
#endif
