//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file OrbitalPretrainingAccumulator.cpp
 * @brief Bounded raw-sum accumulation for orbital pretraining.
 */

#include "QMCDrivers/WFTrain/OrbitalPretrainingAccumulator.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::wftrain
{
OrbitalPretrainingAccumulator::OrbitalPretrainingAccumulator(
    const StructuredParameterSchema& schema,
    std::size_t parameter_version,
    std::uint64_t target_fingerprint,
    std::uint64_t loss_fingerprint)
    : provider_id_(schema.providerId()),
      schema_fingerprint_(schema.fingerprint()),
      parameter_version_(parameter_version),
      target_fingerprint_(target_fingerprint),
      loss_fingerprint_(loss_fingerprint),
      gradient_sum_(schema.parameterCount(), 0.0)
{}

void OrbitalPretrainingAccumulator::addSample(
    DerivativeReal loss,
    std::size_t parameter_version,
    DerivativeArrayView<const DerivativeReal> gradient)
{
  if (finalized_ || poisoned_ || reduction_domain_ == ReductionDomain::GLOBAL)
    throw std::logic_error("Orbital pretraining accumulator is not writable");
  if (parameter_version != parameter_version_ || gradient.size() != gradient_sum_.size())
    throw std::invalid_argument("Orbital pretraining sample metadata is incompatible");
  if (!isFiniteTrainingReal(loss) ||
      !std::all_of(gradient.begin(), gradient.end(), isFiniteTrainingReal))
    throw std::invalid_argument("Orbital pretraining sample is non-finite");
  if (sample_count_ == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("Orbital pretraining sample count overflow");

  const DerivativeReal next_loss = loss_sum_ + loss;
  if (!isFiniteTrainingReal(next_loss))
    throw std::overflow_error("Orbital pretraining loss sum overflow");
  for (std::size_t parameter = 0; parameter < gradient_sum_.size(); ++parameter)
  {
    const DerivativeReal next = gradient_sum_[parameter] + gradient[parameter];
    if (!isFiniteTrainingReal(next))
      throw std::overflow_error("Orbital pretraining gradient sum overflow");
  }

  // Validate the whole candidate sum before changing any accumulator state.
  loss_sum_ = next_loss;
  for (std::size_t parameter = 0; parameter < gradient_sum_.size(); ++parameter)
    gradient_sum_[parameter] += gradient[parameter];
  ++sample_count_;
}

void OrbitalPretrainingAccumulator::requireCompatible(
    const OrbitalPretrainingAccumulator& other) const
{
  if (provider_id_ != other.provider_id_ ||
      schema_fingerprint_ != other.schema_fingerprint_ ||
      parameter_version_ != other.parameter_version_ ||
      target_fingerprint_ != other.target_fingerprint_ ||
      loss_fingerprint_ != other.loss_fingerprint_ ||
      gradient_sum_.size() != other.gradient_sum_.size() ||
      reduction_domain_ != other.reduction_domain_)
    throw std::invalid_argument("Orbital pretraining accumulator metadata mismatch");
  if (finalized_ || poisoned_ || other.finalized_ || other.poisoned_ ||
      reduction_domain_ == ReductionDomain::GLOBAL)
    throw std::logic_error("Orbital pretraining accumulator cannot merge this contribution");
}

void OrbitalPretrainingAccumulator::merge(const OrbitalPretrainingAccumulator& other)
{
  requireCompatible(other);
  if (other.sample_count_ > std::numeric_limits<std::size_t>::max() - sample_count_)
    throw std::overflow_error("Orbital pretraining merged sample count overflow");
  const DerivativeReal next_loss = loss_sum_ + other.loss_sum_;
  if (!isFiniteTrainingReal(next_loss))
    throw std::overflow_error("Orbital pretraining merged loss overflow");
  for (std::size_t parameter = 0; parameter < gradient_sum_.size(); ++parameter)
    if (!isFiniteTrainingReal(gradient_sum_[parameter] + other.gradient_sum_[parameter]))
      throw std::overflow_error("Orbital pretraining merged gradient overflow");

  loss_sum_ = next_loss;
  for (std::size_t parameter = 0; parameter < gradient_sum_.size(); ++parameter)
    gradient_sum_[parameter] += other.gradient_sum_[parameter];
  sample_count_ += other.sample_count_;
}

void OrbitalPretrainingAccumulator::completeSingleParticipantReduction()
{
  if (finalized_ || poisoned_ || reduction_domain_ == ReductionDomain::GLOBAL)
    throw std::logic_error("Orbital pretraining accumulator cannot complete reduction");
  if (sample_count_ == 0)
    throw std::runtime_error("Orbital pretraining population is empty");
  reduction_domain_ = ReductionDomain::GLOBAL;
}

OrbitalPretrainingResult OrbitalPretrainingAccumulator::finalize()
{
  if (finalized_ || poisoned_)
    throw std::logic_error("Orbital pretraining accumulator cannot be finalized");
  if (reduction_domain_ != ReductionDomain::GLOBAL || sample_count_ == 0)
    throw std::logic_error("Orbital pretraining objective is not globally reduced");

  const DerivativeReal inverse_count = 1.0 / static_cast<DerivativeReal>(sample_count_);
  OrbitalPretrainingResult result;
  result.schema_fingerprint = schema_fingerprint_;
  result.parameter_version = parameter_version_;
  result.reduction_domain = reduction_domain_;
  result.target_fingerprint = target_fingerprint_;
  result.loss_fingerprint = loss_fingerprint_;
  result.sample_count = sample_count_;
  result.mean_loss = loss_sum_ * inverse_count;
  result.gradient.resize(gradient_sum_.size());
  for (std::size_t parameter = 0; parameter < gradient_sum_.size(); ++parameter)
    result.gradient[parameter] = gradient_sum_[parameter] * inverse_count;
  finalized_ = true;
  return result;
}

std::size_t OrbitalPretrainingAccumulator::retainedBytes() const noexcept
{
  return gradient_sum_.capacity() * sizeof(DerivativeReal);
}

std::size_t OrbitalPretrainingAccumulator::storageFingerprint() const noexcept
{
  std::size_t hash = UINT64_C(1469598103934665603);
  hash ^= reinterpret_cast<std::uintptr_t>(gradient_sum_.data());
  hash *= UINT64_C(1099511628211);
  hash ^= gradient_sum_.capacity();
  return hash;
}

void OrbitalPretrainingAccumulator::poison() noexcept
{
  std::fill(gradient_sum_.begin(), gradient_sum_.end(), 0.0);
  sample_count_ = 0;
  loss_sum_ = 0.0;
  poisoned_ = true;
}

} // namespace qmcplusplus::wftrain
