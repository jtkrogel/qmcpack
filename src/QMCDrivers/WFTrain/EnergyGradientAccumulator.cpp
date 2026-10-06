//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file EnergyGradientAccumulator.cpp
 * @brief Checked streaming energy-gradient reductions and normalization.
 */

#include "QMCDrivers/WFTrain/EnergyGradientAccumulator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::wftrain
{
namespace
{

/// Report whether both parts of one contraction scalar are finite.
bool isFinite(DerivativeValue value) noexcept
{
  return std::isfinite(value.real()) && std::isfinite(value.imag());
}

/// Reject a non-finite real aggregate before it contaminates retained state.
void requireFinite(DerivativeReal value, const char* description)
{
  if (!std::isfinite(value))
    throw std::invalid_argument(std::string("Non-finite ") + description +
                                " in streaming energy-gradient reduction");
}

/// Reject a non-finite complex aggregate before it contaminates retained state.
void requireFinite(DerivativeValue value, const char* description)
{
  if (!isFinite(value))
    throw std::invalid_argument(std::string("Non-finite ") + description +
                                " in streaming energy-gradient reduction");
}

} // namespace

EnergyGradientAccumulator::EnergyGradientAccumulator(
    const StructuredParameterSchema& schema,
    std::size_t parameter_version,
    EnergyGradientEstimator estimator,
    DerivativeAdjoint adjoint)
    : provider_id_(schema.providerId()),
      schema_fingerprint_(schema.fingerprint()),
      parameter_version_(parameter_version),
      estimator_(estimator),
      adjoint_(adjoint),
      weighted_score_sum_(schema.parameterCount(), DerivativeValue{}),
      weighted_energy_score_sum_(schema.parameterCount(), DerivativeValue{}),
      weighted_energy_derivative_sum_(schema.parameterCount(), DerivativeValue{})
{}

void EnergyGradientAccumulator::onBegin(
    const DerivativeStreamDescriptor& descriptor,
    const ParameterChunkPlan&,
    DerivativeArrayView<const VJPCoefficientChannel> channels)
{
  if (descriptor.provider_id != provider_id_ ||
      descriptor.schema_fingerprint != schema_fingerprint_ ||
      descriptor.parameter_version != parameter_version_)
    throw std::invalid_argument("Energy-gradient stream does not match the parameter snapshot");
  if (descriptor.adjoint != adjoint_)
    throw std::invalid_argument("Energy-gradient stream uses an unexpected adjoint convention");
  if (descriptor.parameter_scalar_domain != ParameterScalarDomain::REAL64 ||
      descriptor.result_scalar_domain != ParameterScalarDomain::COMPLEX128)
    throw std::invalid_argument(
        "Energy-gradient accumulation currently requires real parameters and complex contractions");
  if (channels.size() != 3)
    throw std::invalid_argument("Energy-gradient stream must contain exactly three VJP channels");

  channel_kinds_.clear();
  channel_kinds_.reserve(channels.size());
  bool found_weighted_score        = false;
  bool found_energy_weighted_score = false;
  bool found_weighted_local_energy = false;
  std::uint32_t local_energy_term_mask = 0;
  for (const VJPCoefficientChannel& channel : channels)
  {
    if (channel.id == WEIGHTED_SCORE_CHANNEL &&
        channel.product == DerivativeProduct::SCORE_VJP && !found_weighted_score)
    {
      channel_kinds_.push_back(ChannelKind::WEIGHTED_SCORE);
      found_weighted_score = true;
    }
    else if (channel.id == ENERGY_WEIGHTED_SCORE_CHANNEL &&
             channel.product == DerivativeProduct::SCORE_VJP &&
             !found_energy_weighted_score)
    {
      channel_kinds_.push_back(ChannelKind::ENERGY_WEIGHTED_SCORE);
      found_energy_weighted_score = true;
    }
    else if (channel.id == WEIGHTED_LOCAL_ENERGY_CHANNEL &&
             channel.product == DerivativeProduct::LOCAL_ENERGY_VJP &&
             !found_weighted_local_energy)
    {
      if (channel.local_energy_term_mask == 0)
        throw std::invalid_argument(
            "Energy-gradient local-energy channel must name Hamiltonian-term coverage");
      channel_kinds_.push_back(ChannelKind::WEIGHTED_LOCAL_ENERGY);
      found_weighted_local_energy = true;
      local_energy_term_mask      = channel.local_energy_term_mask;
    }
    else
      throw std::invalid_argument("Energy-gradient stream has a duplicate or unknown channel");
  }

  stream_sample_count_ = descriptor.sample_count;
  reduction_domain_    = descriptor.reduction_domain;
  local_energy_term_mask_ = local_energy_term_mask;
}

void EnergyGradientAccumulator::consume(
    std::size_t channel_ordinal,
    const ParameterChunkConstView& chunk)
{
  const DerivativeArrayView<const DerivativeValue> values = chunk.values();
  for (const DerivativeValue value : values)
    requireFinite(value, "derivative contraction");

  std::vector<DerivativeValue>* destination = nullptr;
  switch (channel_kinds_.at(channel_ordinal))
  {
  case ChannelKind::WEIGHTED_SCORE:
    destination = &weighted_score_sum_;
    break;
  case ChannelKind::ENERGY_WEIGHTED_SCORE:
    destination = &weighted_energy_score_sum_;
    break;
  case ChannelKind::WEIGHTED_LOCAL_ENERGY:
    destination = &weighted_energy_derivative_sum_;
    break;
  }

  const std::size_t begin = chunk.descriptor().parameter_offset;
  for (std::size_t index = 0; index < values.size(); ++index)
    (*destination)[begin + index] += values[index];
}

void EnergyGradientAccumulator::onEnd()
{
  derivative_complete_ = true;
}

void EnergyGradientAccumulator::onAbort() noexcept
{
  clearRawSums();
}

void EnergyGradientAccumulator::onReset() noexcept
{
  clearRawSums();
}

void EnergyGradientAccumulator::clearRawSums() noexcept
{
  sample_count_ = 0;
  weight_sum_ = 0.0;
  weighted_energy_sum_ = 0.0;
  weighted_energy_norm_sum_ = 0.0;
  std::fill(weighted_score_sum_.begin(), weighted_score_sum_.end(), DerivativeValue{});
  std::fill(weighted_energy_score_sum_.begin(), weighted_energy_score_sum_.end(),
            DerivativeValue{});
  std::fill(weighted_energy_derivative_sum_.begin(),
            weighted_energy_derivative_sum_.end(), DerivativeValue{});
  channel_kinds_.clear();
  local_energy_term_mask_ = 0;
  stream_sample_count_ = 0;
  reduction_domain_ = ReductionDomain::CROWD_LOCAL;
  derivative_complete_ = false;
  scalar_sums_complete_ = false;
  finalized_ = false;
}

void EnergyGradientAccumulator::addScalarSums(
    std::size_t sample_count,
    DerivativeReal weight_sum,
    DerivativeValue weighted_energy_sum,
    DerivativeReal weighted_energy_norm_sum)
{
  if (state() != DerivativeSinkState::COMPLETE || !derivative_complete_)
    throw std::logic_error("Scalar sums require a complete energy-gradient VJP stream");
  if (scalar_sums_complete_)
    throw std::logic_error("Scalar sums were already supplied for this derivative stream");
  if (sample_count != stream_sample_count_)
    throw std::invalid_argument("Scalar sums do not describe the derivative stream sample batch");
  requireFinite(weight_sum, "weight sum");
  requireFinite(weighted_energy_sum, "weighted energy sum");
  requireFinite(weighted_energy_norm_sum, "weighted energy-norm sum");
  if (weight_sum < 0.0 || weighted_energy_norm_sum < 0.0)
    throw std::invalid_argument("Energy-gradient scalar sums contain a negative weight or norm");

  sample_count_ = sample_count;
  weight_sum_ = weight_sum;
  weighted_energy_sum_ = weighted_energy_sum;
  weighted_energy_norm_sum_ = weighted_energy_norm_sum;
  scalar_sums_complete_ = true;
}

void EnergyGradientAccumulator::requireUsableContribution() const
{
  if (state() == DerivativeSinkState::POISONED)
    throw std::logic_error("Energy-gradient stream is poisoned and must be reset");
  if (!hasCompleteContribution())
    throw std::logic_error("Energy-gradient contribution is incomplete");
  if (finalized_)
    throw std::logic_error("Energy-gradient accumulator was already finalized");
}

void EnergyGradientAccumulator::merge(const EnergyGradientAccumulator& other)
{
  if (state() == DerivativeSinkState::ACTIVE)
    throw std::logic_error("Cannot merge into an active energy-gradient accumulator");
  if (state() == DerivativeSinkState::POISONED)
    throw std::logic_error("Cannot merge into a poisoned energy-gradient accumulator");
  if (finalized_)
    throw std::logic_error("Cannot merge into a finalized energy-gradient accumulator");
  if (derivative_complete_ != scalar_sums_complete_)
    throw std::logic_error("Cannot merge into a half-complete energy-gradient accumulator");
  other.requireUsableContribution();
  if (schema_fingerprint_ != other.schema_fingerprint_ ||
      parameter_version_ != other.parameter_version_ || estimator_ != other.estimator_ ||
      adjoint_ != other.adjoint_)
    throw std::invalid_argument("Cannot merge incompatible energy-gradient accumulators");
  if (hasCompleteContribution() &&
      local_energy_term_mask_ != other.local_energy_term_mask_)
    throw std::invalid_argument(
        "Cannot merge energy-gradient accumulators with different local-energy term coverage");
  if (hasCompleteContribution() && reduction_domain_ != other.reduction_domain_)
    throw std::invalid_argument("Cannot merge energy-gradient accumulators from different domains");
  if (sample_count_ > std::numeric_limits<std::size_t>::max() - other.sample_count_)
    throw std::overflow_error("Streaming energy-gradient sample count overflow");

  // Validate the whole merge before mutating any raw sum, so a numerical
  // overflow cannot leave an apparently usable partial reduction behind.
  requireFinite(weight_sum_ + other.weight_sum_, "merged weight sum");
  requireFinite(weighted_energy_sum_ + other.weighted_energy_sum_,
                "merged weighted energy sum");
  requireFinite(weighted_energy_norm_sum_ + other.weighted_energy_norm_sum_,
                "merged weighted energy norm sum");
  for (std::size_t parameter = 0; parameter < weighted_score_sum_.size(); ++parameter)
  {
    requireFinite(weighted_score_sum_[parameter] + other.weighted_score_sum_[parameter],
                  "merged weighted score contraction");
    requireFinite(
        weighted_energy_score_sum_[parameter] + other.weighted_energy_score_sum_[parameter],
        "merged weighted energy-score contraction");
    requireFinite(weighted_energy_derivative_sum_[parameter] +
                      other.weighted_energy_derivative_sum_[parameter],
                  "merged weighted local-energy contraction");
  }

  sample_count_ += other.sample_count_;
  weight_sum_ += other.weight_sum_;
  weighted_energy_sum_ += other.weighted_energy_sum_;
  weighted_energy_norm_sum_ += other.weighted_energy_norm_sum_;
  for (std::size_t parameter = 0; parameter < weighted_score_sum_.size(); ++parameter)
  {
    weighted_score_sum_[parameter] += other.weighted_score_sum_[parameter];
    weighted_energy_score_sum_[parameter] += other.weighted_energy_score_sum_[parameter];
    weighted_energy_derivative_sum_[parameter] +=
        other.weighted_energy_derivative_sum_[parameter];
  }
  reduction_domain_ = other.reduction_domain_;
  local_energy_term_mask_ = other.local_energy_term_mask_;
  derivative_complete_ = true;
  scalar_sums_complete_ = true;
}

void EnergyGradientAccumulator::completeSingleParticipantReduction()
{
  requireUsableContribution();
  reduction_domain_ = ReductionDomain::GLOBAL;
}

EnergyGradientResult EnergyGradientAccumulator::finalize()
{
  requireUsableContribution();
  if (reduction_domain_ != ReductionDomain::GLOBAL)
    throw std::logic_error("Energy-gradient objective must be globally reduced before finalization");
  if (sample_count_ == 0 || !(weight_sum_ > 0.0))
    throw std::runtime_error("Cannot finalize an empty or zero-weight energy-gradient stream");
  requireFinite(weight_sum_, "total weight");
  requireFinite(weighted_energy_sum_, "total weighted energy");
  requireFinite(weighted_energy_norm_sum_, "total weighted energy norm");

  EnergyGradientResult result;
  result.schema_fingerprint = schema_fingerprint_;
  result.parameter_version  = parameter_version_;
  result.reduction_domain   = reduction_domain_;
  result.estimator          = estimator_;
  result.local_energy_term_mask = local_energy_term_mask_;
  result.sample_count       = sample_count_;
  result.weight_sum         = weight_sum_;
  result.mean_energy        = weighted_energy_sum_ / weight_sum_;
  result.energy_variance = std::max(
      DerivativeReal{}, weighted_energy_norm_sum_ / weight_sum_ - std::norm(result.mean_energy));
  result.gradient.resize(weighted_score_sum_.size());

  for (std::size_t parameter = 0; parameter < result.gradient.size(); ++parameter)
  {
    if (estimator_ == EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN)
    {
      const DerivativeValue contraction =
          weighted_energy_derivative_sum_[parameter] +
          weighted_energy_score_sum_[parameter] -
          result.mean_energy * weighted_score_sum_[parameter];
      result.gradient[parameter] = 2.0 * std::real(contraction) / weight_sum_;
    }
    else
    {
      result.gradient[parameter] =
          std::real(weighted_energy_derivative_sum_[parameter]) / weight_sum_ +
          2.0 * (std::real(weighted_energy_score_sum_[parameter]) / weight_sum_ -
                 std::real(result.mean_energy) *
                     std::real(weighted_score_sum_[parameter]) / weight_sum_);
    }
  }

  finalized_ = true;
  return result;
}

std::size_t EnergyGradientAccumulator::retainedBytes() const noexcept
{
  return sizeof(DerivativeValue) *
      (weighted_score_sum_.capacity() + weighted_energy_score_sum_.capacity() +
       weighted_energy_derivative_sum_.capacity());
}

void accumulateEnergyGradientBatch(
    const StreamingDerivativeOperator& derivative_operator,
    DerivativeArrayView<const DerivativeReal> weights,
    DerivativeArrayView<const DerivativeValue> local_energies,
    std::uint32_t local_energy_term_mask,
    EnergyGradientAccumulator& accumulator,
    DerivativeAdjoint adjoint)
{
  if ((weights.size() != 0 && weights.data() == nullptr) ||
      (local_energies.size() != 0 && local_energies.data() == nullptr))
    throw std::invalid_argument(
        "Nonempty energy-gradient weights and energies require valid storage");
  const std::size_t sample_count = derivative_operator.sampleCount();
  if (weights.size() != sample_count || local_energies.size() != sample_count)
    throw std::invalid_argument("Energy-gradient weights and energies must match the operator batch");

  DerivativeReal weight_sum = 0.0;
  DerivativeValue weighted_energy_sum = 0.0;
  DerivativeReal weighted_energy_norm_sum = 0.0;
  std::vector<DerivativeValue> weighted_score_coefficients(sample_count);
  std::vector<DerivativeValue> energy_weighted_score_coefficients(sample_count);
  for (std::size_t sample = 0; sample < sample_count; ++sample)
  {
    requireFinite(weights[sample], "sample weight");
    requireFinite(local_energies[sample], "sample local energy");
    if (weights[sample] < 0.0)
      throw std::invalid_argument("Energy-gradient sample weight is negative");

    weight_sum += weights[sample];
    weighted_energy_sum += weights[sample] * local_energies[sample];
    weighted_energy_norm_sum += weights[sample] * std::norm(local_energies[sample]);
    weighted_score_coefficients[sample] = weights[sample];
    energy_weighted_score_coefficients[sample] =
        weights[sample] *
        (accumulator.estimator() == EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN
             ? local_energies[sample]
             : DerivativeValue{std::real(local_energies[sample]), 0.0});
  }
  requireFinite(weight_sum, "batch weight sum");
  requireFinite(weighted_energy_sum, "batch weighted energy sum");
  requireFinite(weighted_energy_norm_sum, "batch weighted energy norm sum");

  const StructuredParameterSchema& schema = derivative_operator.parameterSchema();
  const auto make_coefficients = [&](const std::vector<DerivativeValue>& values) {
    return CoefficientView{schema.providerId(), schema.fingerprint(),
                           derivative_operator.parameterVersion(),
                           derivative_operator.batchOrdinal(),
                           derivative_operator.sampleOffset(),
                           {values.data(), values.size()}};
  };

  const std::array<VJPCoefficientChannel, 3> channels{{
      {EnergyGradientAccumulator::WEIGHTED_SCORE_CHANNEL, DerivativeProduct::SCORE_VJP,
       make_coefficients(weighted_score_coefficients), 0},
      {EnergyGradientAccumulator::ENERGY_WEIGHTED_SCORE_CHANNEL,
       DerivativeProduct::SCORE_VJP,
       make_coefficients(energy_weighted_score_coefficients), 0},
      {EnergyGradientAccumulator::WEIGHTED_LOCAL_ENERGY_CHANNEL,
       DerivativeProduct::LOCAL_ENERGY_VJP,
       make_coefficients(weighted_score_coefficients), local_energy_term_mask}}};

  derivative_operator.applyVJPs({channels.data(), channels.size()}, adjoint, accumulator);
  accumulator.addScalarSums(sample_count, weight_sum, weighted_energy_sum,
                            weighted_energy_norm_sum);
}

} // namespace qmcplusplus::wftrain
