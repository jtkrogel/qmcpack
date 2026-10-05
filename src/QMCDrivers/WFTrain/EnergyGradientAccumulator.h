//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file EnergyGradientAccumulator.h
 * @brief O(parameter-count) streaming statistics for an energy objective.
 */

#ifndef QMCPLUSPLUS_ENERGY_GRADIENT_ACCUMULATOR_H
#define QMCPLUSPLUS_ENERGY_GRADIENT_ACCUMULATOR_H

#include "QMCWaveFunctions/Optimization/StructuredParameterProvider.h"

#include <cstddef>
#include <string>
#include <vector>

namespace qmcplusplus::wftrain
{

/// Select the normalization convention applied to the real energy gradient.
enum class EnergyGradientConvention
{
  REAL_VMC,
  HALF_GRADIENT
};

/// Final scalar moments and one parameter-shaped energy gradient.
struct EnergyGradientResult
{
  std::size_t sample_count = 0;
  double weight_sum        = 0.0;
  double mean_energy       = 0.0;
  double energy_variance   = 0.0;
  std::vector<double> gradient;
};

/** Accumulate already contracted derivative chunks without retaining sample rows.
 *
 * A derivative producer supplies sums of ``w O``, ``w E O``, and ``w dE``.
 * The class intentionally has no API accepting an N-sample by P-parameter
 * matrix, keeping its retained memory independent of sample count.
 */
class EnergyGradientAccumulator
{
public:
  explicit EnergyGradientAccumulator(
      const StructuredParameterSchema& schema,
      EnergyGradientConvention convention = EnergyGradientConvention::REAL_VMC);

  /// Add scalar moments for any nonempty producer batch.
  void addScalarSums(std::size_t sample_count,
                     double weight_sum,
                     double weighted_energy_sum,
                     double weighted_energy_squared_sum);

  /// Add one contiguous contraction chunk within a declared parameter block.
  void addDerivativeSums(std::size_t block_index,
                         std::size_t block_offset,
                         const double* weighted_score_sum,
                         const double* weighted_energy_score_sum,
                         const double* weighted_energy_derivative_sum,
                         std::size_t count);

  /// Merge another independently accumulated stream with the same schema.
  void merge(const EnergyGradientAccumulator& other);

  /// Normalize scalar moments and construct the energy gradient exactly once.
  EnergyGradientResult finalize();

  /// Return retained numeric capacity, excluding a separately returned result.
  std::size_t retainedBytes() const noexcept;

  /// Return the number of scalar parameters represented by each contraction vector.
  std::size_t parameterCount() const noexcept { return weighted_score_sum_.size(); }

private:
  void requireAccumulating() const;

  std::string schema_fingerprint_;
  std::vector<ParameterBlockDescriptor> blocks_;
  EnergyGradientConvention convention_;
  std::size_t sample_count_ = 0;
  double weight_sum_ = 0.0;
  double weighted_energy_sum_ = 0.0;
  double weighted_energy_squared_sum_ = 0.0;
  std::vector<double> weighted_score_sum_;
  std::vector<double> weighted_energy_score_sum_;
  std::vector<double> weighted_energy_derivative_sum_;
  bool finalized_ = false;
};

} // namespace qmcplusplus::wftrain

#endif

