//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file EnergyGradientAccumulator.cpp
 * @brief Streaming energy-gradient reductions and normalization.
 */

#include "QMCDrivers/WFTrain/EnergyGradientAccumulator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::wftrain
{
namespace
{

/// Reject non-finite scalar aggregates before they contaminate retained state.
void requireFinite(double value, const char* description)
{
  if (!std::isfinite(value))
    throw std::invalid_argument(std::string("Non-finite ") + description +
                                " in streaming energy-gradient reduction");
}

} // namespace

EnergyGradientAccumulator::EnergyGradientAccumulator(
    const StructuredParameterSchema& schema,
    EnergyGradientConvention convention)
    : schema_fingerprint_(schema.fingerprint()),
      blocks_(schema.blocks()),
      convention_(convention),
      weighted_score_sum_(schema.parameterCount(), 0.0),
      weighted_energy_score_sum_(schema.parameterCount(), 0.0),
      weighted_energy_derivative_sum_(schema.parameterCount(), 0.0)
{}

void EnergyGradientAccumulator::requireAccumulating() const
{
  if (finalized_)
    throw std::logic_error("Streaming energy-gradient accumulator was already finalized");
}

void EnergyGradientAccumulator::addScalarSums(
    std::size_t sample_count,
    double weight_sum,
    double weighted_energy_sum,
    double weighted_energy_squared_sum)
{
  requireAccumulating();
  if (sample_count == 0)
    throw std::invalid_argument("A streaming scalar batch must contain at least one sample");
  requireFinite(weight_sum, "weight sum");
  requireFinite(weighted_energy_sum, "weighted energy sum");
  requireFinite(weighted_energy_squared_sum, "weighted energy-squared sum");
  if (weight_sum < 0.0)
    throw std::invalid_argument("A streaming scalar batch has a negative weight sum");
  if (sample_count_ > std::numeric_limits<std::size_t>::max() - sample_count)
    throw std::overflow_error("Streaming energy-gradient sample count overflow");

  sample_count_ += sample_count;
  weight_sum_ += weight_sum;
  weighted_energy_sum_ += weighted_energy_sum;
  weighted_energy_squared_sum_ += weighted_energy_squared_sum;
}

void EnergyGradientAccumulator::addDerivativeSums(
    std::size_t block_index,
    std::size_t block_offset,
    const double* weighted_score_sum,
    const double* weighted_energy_score_sum,
    const double* weighted_energy_derivative_sum,
    std::size_t count)
{
  requireAccumulating();
  if (block_index >= blocks_.size())
    throw std::out_of_range("Streaming derivative chunk refers to an unknown parameter block");
  const ParameterBlockDescriptor& block = blocks_[block_index];
  if (block_offset > block.count || count > block.count - block_offset)
    throw std::out_of_range("Streaming derivative chunk exceeds its parameter block");
  if (count != 0 &&
      (!weighted_score_sum || !weighted_energy_score_sum || !weighted_energy_derivative_sum))
    throw std::invalid_argument("Streaming derivative chunk has a null input array");

  const std::size_t begin = block.offset + block_offset;
  for (std::size_t index = 0; index < count; ++index)
  {
    requireFinite(weighted_score_sum[index], "weighted score contraction");
    requireFinite(weighted_energy_score_sum[index], "weighted energy-score contraction");
    requireFinite(weighted_energy_derivative_sum[index],
                  "weighted local-energy derivative contraction");
    weighted_score_sum_[begin + index] += weighted_score_sum[index];
    weighted_energy_score_sum_[begin + index] += weighted_energy_score_sum[index];
    weighted_energy_derivative_sum_[begin + index] +=
        weighted_energy_derivative_sum[index];
  }
}

void EnergyGradientAccumulator::merge(const EnergyGradientAccumulator& other)
{
  requireAccumulating();
  other.requireAccumulating();
  if (schema_fingerprint_ != other.schema_fingerprint_ || convention_ != other.convention_)
    throw std::invalid_argument("Cannot merge incompatible streaming energy-gradient accumulators");
  if (sample_count_ > std::numeric_limits<std::size_t>::max() - other.sample_count_)
    throw std::overflow_error("Streaming energy-gradient sample count overflow");

  sample_count_ += other.sample_count_;
  weight_sum_ += other.weight_sum_;
  weighted_energy_sum_ += other.weighted_energy_sum_;
  weighted_energy_squared_sum_ += other.weighted_energy_squared_sum_;
  for (std::size_t parameter = 0; parameter < weighted_score_sum_.size(); ++parameter)
  {
    weighted_score_sum_[parameter] += other.weighted_score_sum_[parameter];
    weighted_energy_score_sum_[parameter] += other.weighted_energy_score_sum_[parameter];
    weighted_energy_derivative_sum_[parameter] +=
        other.weighted_energy_derivative_sum_[parameter];
  }
}

EnergyGradientResult EnergyGradientAccumulator::finalize()
{
  requireAccumulating();
  if (sample_count_ == 0 || !(weight_sum_ > 0.0))
    throw std::runtime_error("Cannot finalize an empty or zero-weight energy-gradient stream");
  requireFinite(weight_sum_, "total weight");
  requireFinite(weighted_energy_sum_, "total weighted energy");
  requireFinite(weighted_energy_squared_sum_, "total weighted energy squared");

  EnergyGradientResult result;
  result.sample_count   = sample_count_;
  result.weight_sum     = weight_sum_;
  result.mean_energy    = weighted_energy_sum_ / weight_sum_;
  const double mean_e2  = weighted_energy_squared_sum_ / weight_sum_;
  result.energy_variance = std::max(0.0, mean_e2 - result.mean_energy * result.mean_energy);
  result.gradient.resize(weighted_score_sum_.size());

  const double scale = convention_ == EnergyGradientConvention::REAL_VMC ? 2.0 : 1.0;
  for (std::size_t parameter = 0; parameter < result.gradient.size(); ++parameter)
    result.gradient[parameter] = scale *
        (weighted_energy_derivative_sum_[parameter] / weight_sum_ +
         weighted_energy_score_sum_[parameter] / weight_sum_ -
         result.mean_energy * weighted_score_sum_[parameter] / weight_sum_);
  finalized_ = true;
  return result;
}

std::size_t EnergyGradientAccumulator::retainedBytes() const noexcept
{
  return sizeof(double) * (weighted_score_sum_.capacity() +
                           weighted_energy_score_sum_.capacity() +
                           weighted_energy_derivative_sum_.capacity());
}

} // namespace qmcplusplus::wftrain

