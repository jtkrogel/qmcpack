//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file DistributedParameterReduction.cpp
 * @brief Fixed-schedule replicated reductions for high-parameter training.
 */

#include "QMCDrivers/WFTrain/DistributedParameterReduction.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"

#include "Message/CommOperators.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

constexpr std::uint64_t protocol_version = 1;
constexpr std::uint64_t energy_channel_contract = UINT64_C(0x454752414433); // "EGRAD3"
constexpr std::uint64_t orbital_channel_contract = UINT64_C(0x4f52424752414431); // "ORBGRAD1"

/// Classify failures without communicating variable-length exception strings.
enum class ConsensusReason : std::uint64_t
{
  NONE = 0,
  LOCAL_EXCEPTION,
  INVALID_POLICY,
  INCOMPLETE_CONTRIBUTION,
  NONFINITE_CONTRIBUTION,
  INVALID_CANDIDATE
};

/// Return a stable diagnostic label for one fixed-record failure reason.
const char* consensusReasonName(ConsensusReason reason) noexcept
{
  switch (reason)
  {
  case ConsensusReason::NONE:
    return "none";
  case ConsensusReason::LOCAL_EXCEPTION:
    return "rank-local exception";
  case ConsensusReason::INVALID_POLICY:
    return "invalid reduction policy";
  case ConsensusReason::INCOMPLETE_CONTRIBUTION:
    return "incomplete contribution";
  case ConsensusReason::NONFINITE_CONTRIBUTION:
    return "non-finite contribution";
  case ConsensusReason::INVALID_CANDIDATE:
    return "invalid candidate";
  }
  return "unknown";
}

/// Return whether a full-precision contraction scalar is finite.
bool isFinite(DerivativeValue value) noexcept
{
  return std::isfinite(value.real()) && std::isfinite(value.imag());
}

/// Extend a stable FNV-1a identity hash with one byte interval.
void extendHash(std::uint64_t& hash, const void* bytes, std::size_t byte_count) noexcept
{
  const auto* data = static_cast<const unsigned char*>(bytes);
  for (std::size_t index = 0; index < byte_count; ++index)
  {
    hash ^= data[index];
    hash *= UINT64_C(1099511628211);
  }
}

/// Hash a string into one compact fixed-record consensus field.
std::uint64_t hashString(std::string_view text) noexcept
{
  std::uint64_t hash = UINT64_C(14695981039346656037);
  extendHash(hash, text.data(), text.size());
  return hash;
}

/// Fingerprint a complete real parameter snapshot without allocating a transport copy.
std::uint64_t hashSnapshot(const StructuredParameterSnapshot& snapshot) noexcept
{
  std::uint64_t hash = hashString(snapshot.schema_fingerprint);
  const std::uint64_t version = snapshot.version;
  const std::uint64_t count   = snapshot.values.size();
  extendHash(hash, &version, sizeof(version));
  extendHash(hash, &count, sizeof(count));
  if (!snapshot.values.empty())
    extendHash(hash, snapshot.values.data(), snapshot.values.size() * sizeof(double));
  return hash;
}

/// Gather one fixed-width control record on every participant.
template<std::size_t field_count>
std::vector<std::array<std::uint64_t, field_count>> gatherRecords(
    Communicate* communicator,
    const std::array<std::uint64_t, field_count>& local_record)
{
  const std::size_t participant_count = communicator ? communicator->size() : 1;
  std::vector<std::array<std::uint64_t, field_count>> records(participant_count);
  if (communicator)
  {
    // Receive into explicitly flat storage rather than assuming std::array has no
    // tail padding when several records are adjacent in a vector.
    std::vector<std::uint64_t> flat_records(participant_count * field_count);
    auto send_record = local_record;
    communicator->allgather(send_record.data(), flat_records.data(),
                            static_cast<int>(field_count));
    for (std::size_t rank = 0; rank < participant_count; ++rank)
      std::copy_n(flat_records.data() + rank * field_count, field_count,
                  records[rank].begin());
  }
  else
    records.front() = local_record;
  return records;
}

/// Compare readiness metadata while allowing each rank to contribute a different count.
template<std::size_t field_count>
std::size_t firstContributionMismatch(
    const std::vector<std::array<std::uint64_t, field_count>>& records,
    std::size_t sample_count_field,
    std::size_t clip_count_field,
    std::size_t consumed_count_field) noexcept
{
  for (std::size_t rank = 1; rank < records.size(); ++rank)
    for (std::size_t field = 1; field < field_count; ++field)
      if (field != sample_count_field && field != clip_count_field &&
          field != consumed_count_field &&
          records.front()[field] != records[rank][field])
        return rank;
  return records.size();
}

/// Return the binary64 representation used in fixed clipping metadata records.
std::uint64_t doubleBits(double value) noexcept
{
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

/// Rethrow a useful local exception or issue one rank-uniform distributed diagnostic.
[[noreturn]] void throwConsensusFailure(const char* stage,
                                        std::size_t failing_rank,
                                        ConsensusReason reason,
                                        std::exception_ptr local_failure,
                                        std::size_t participant_count)
{
  if (participant_count == 1 && local_failure)
    std::rethrow_exception(local_failure);

  std::ostringstream message;
  message << "Distributed training " << stage << " failed on rank " << failing_rank
          << " (" << consensusReasonName(reason) << ')';
  throw std::runtime_error(message.str());
}

/// Find the lowest rank reporting an explicit local failure.
template<std::size_t field_count>
std::size_t firstFailedRank(
    const std::vector<std::array<std::uint64_t, field_count>>& records) noexcept
{
  for (std::size_t rank = 0; rank < records.size(); ++rank)
    if (records[rank][0] != static_cast<std::uint64_t>(ConsensusReason::NONE))
      return rank;
  return records.size();
}

/// Find the lowest rank whose successful metadata differs from rank zero.
template<std::size_t field_count>
std::size_t firstMismatchingRank(
    const std::vector<std::array<std::uint64_t, field_count>>& records) noexcept
{
  for (std::size_t rank = 1; rank < records.size(); ++rank)
    if (!std::equal(records.front().begin() + 1, records.front().end(),
                    records[rank].begin() + 1))
      return rank;
  return records.size();
}

/// Reject a fixed-record mismatch uniformly before entering any sum collective.
[[noreturn]] void throwMetadataMismatch(const char* stage, std::size_t rank)
{
  std::ostringstream message;
  message << "Distributed training " << stage << " metadata mismatch at rank " << rank;
  throw std::runtime_error(message.str());
}

} // namespace

DistributedParameterReduction::DistributedParameterReduction(
    DistributedReductionPolicy policy)
    : policy_(policy)
{}

DistributedParameterReduction::DistributedParameterReduction(
    Communicate& communicator,
    DistributedReductionPolicy policy)
    : communicator_(&communicator), policy_(policy)
{}

std::size_t DistributedParameterReduction::participantCount() const noexcept
{
  return communicator_ ? static_cast<std::size_t>(communicator_->size()) : 1;
}

void DistributedParameterReduction::preflight(
    const StructuredParameterSchema& schema,
    const StructuredParameterSnapshot* parameters,
    EnergyGradientEstimator estimator,
    std::exception_ptr local_failure) const
{
  ConsensusReason reason = local_failure ? ConsensusReason::LOCAL_EXCEPTION
                                         : ConsensusReason::NONE;
  constexpr std::size_t complex_dimension = 2;
  const std::size_t maximum_mpi_chunk =
      static_cast<std::size_t>(std::numeric_limits<int>::max()) / complex_dimension;
  if (reason == ConsensusReason::NONE &&
      (policy_.maximum_chunk_size == 0 || policy_.maximum_chunk_size > maximum_mpi_chunk))
    reason = ConsensusReason::INVALID_POLICY;
  if (reason == ConsensusReason::NONE && parameters == nullptr)
    reason = ConsensusReason::LOCAL_EXCEPTION;
  if (reason == ConsensusReason::NONE &&
      (parameters->schema_fingerprint != schema.fingerprint() ||
       parameters->values.size() != schema.parameterCount()))
    reason = ConsensusReason::LOCAL_EXCEPTION;

  const std::array<std::uint64_t, 11> local_record{
      static_cast<std::uint64_t>(reason),
      protocol_version,
      hashString(schema.providerId()),
      hashString(schema.fingerprint()),
      parameters ? parameters->version : 0,
      schema.parameterCount(),
      static_cast<std::uint64_t>(estimator),
      energy_channel_contract,
      policy_.maximum_chunk_size,
      participantCount(),
      parameters ? hashSnapshot(*parameters) : 0};
  const auto records = gatherRecords(communicator_, local_record);

  const std::size_t failed_rank = firstFailedRank(records);
  if (failed_rank != records.size())
    throwConsensusFailure("preflight", failed_rank,
                          static_cast<ConsensusReason>(records[failed_rank][0]),
                          failed_rank == 0 ? local_failure : std::exception_ptr{}, records.size());
  const std::size_t mismatch_rank = firstMismatchingRank(records);
  if (mismatch_rank != records.size())
    throwMetadataMismatch("preflight", mismatch_rank);
}

void DistributedParameterReduction::preflightOrbital(
    const StructuredParameterSchema& schema,
    const StructuredParameterSnapshot* parameters,
    std::uint64_t target_fingerprint,
    std::uint64_t loss_fingerprint,
    std::exception_ptr local_failure) const
{
  ConsensusReason reason = local_failure ? ConsensusReason::LOCAL_EXCEPTION
                                         : ConsensusReason::NONE;
  const std::size_t maximum_mpi_chunk =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (reason == ConsensusReason::NONE &&
      (policy_.maximum_chunk_size == 0 || policy_.maximum_chunk_size > maximum_mpi_chunk))
    reason = ConsensusReason::INVALID_POLICY;
  if (reason == ConsensusReason::NONE && !parameters)
    reason = ConsensusReason::LOCAL_EXCEPTION;
  if (reason == ConsensusReason::NONE &&
      (parameters->schema_fingerprint != schema.fingerprint() ||
       parameters->values.size() != schema.parameterCount()))
    reason = ConsensusReason::LOCAL_EXCEPTION;

  const std::array<std::uint64_t, 12> local_record{
      static_cast<std::uint64_t>(reason),
      protocol_version,
      hashString(schema.providerId()),
      hashString(schema.fingerprint()),
      parameters ? parameters->version : 0,
      schema.parameterCount(),
      target_fingerprint,
      loss_fingerprint,
      orbital_channel_contract,
      policy_.maximum_chunk_size,
      participantCount(),
      parameters ? hashSnapshot(*parameters) : 0};
  const auto records = gatherRecords(communicator_, local_record);
  const std::size_t failed_rank = firstFailedRank(records);
  if (failed_rank != records.size())
    throwConsensusFailure("orbital preflight", failed_rank,
                          static_cast<ConsensusReason>(records[failed_rank][0]),
                          failed_rank == 0 ? local_failure : std::exception_ptr{}, records.size());
  const std::size_t mismatch_rank = firstMismatchingRank(records);
  if (mismatch_rank != records.size())
    throwMetadataMismatch("orbital preflight", mismatch_rank);
}

void DistributedParameterReduction::reduce(
    EnergyGradientAccumulator& accumulator,
    std::exception_ptr local_failure) const
{
  ConsensusReason reason = local_failure ? ConsensusReason::LOCAL_EXCEPTION
                                         : ConsensusReason::NONE;
  if (reason == ConsensusReason::NONE &&
      (!accumulator.hasCompleteContribution() || accumulator.finalized_ ||
       accumulator.reduction_domain_ == ReductionDomain::GLOBAL))
    reason = ConsensusReason::INCOMPLETE_CONTRIBUTION;

  if (reason == ConsensusReason::NONE)
  {
    const bool finite_scalars = std::isfinite(accumulator.weight_sum_) &&
        isFinite(accumulator.weighted_energy_sum_) &&
        std::isfinite(accumulator.weighted_energy_norm_sum_) &&
        (!accumulator.clipping_descriptor_ ||
         isFinite(accumulator.clipped_weighted_energy_sum_));
    const auto all_finite = [](const std::vector<DerivativeValue>& values) {
      return std::all_of(values.begin(), values.end(), isFinite);
    };
    if (!finite_scalars || !all_finite(accumulator.weighted_score_sum_) ||
        !all_finite(accumulator.weighted_energy_score_sum_) ||
        !all_finite(accumulator.weighted_energy_derivative_sum_))
      reason = ConsensusReason::NONFINITE_CONTRIBUTION;
    if (accumulator.clipping_descriptor_ &&
        accumulator.clipping_consumed_sample_count_ != accumulator.sample_count_)
      reason = ConsensusReason::INCOMPLETE_CONTRIBUTION;
  }

  const EnergyClippingDescriptor* clipping = accumulator.clipping_descriptor_
      ? &*accumulator.clipping_descriptor_
      : nullptr;
  const std::array<std::uint64_t, 21> local_record{
      static_cast<std::uint64_t>(reason),
      protocol_version,
      accumulator.local_energy_term_mask_,
      static_cast<std::uint64_t>(accumulator.reduction_domain_),
      accumulator.parameterCount(),
      accumulator.sample_count_,
      static_cast<std::uint64_t>(accumulator.estimator_),
      hashString(accumulator.schema_fingerprint_),
      accumulator.parameter_version_,
      static_cast<std::uint64_t>(accumulator.adjoint_),
      clipping != nullptr,
      clipping ? clipping->identity : 0,
      clipping ? static_cast<std::uint64_t>(clipping->policy.scale_rule) : 0,
      clipping ? clipping->population : 0,
      clipping ? doubleBits(clipping->policy.width_multiplier) : 0,
      clipping ? doubleBits(clipping->policy.residual_quantile) : 0,
      clipping ? doubleBits(clipping->center) : 0,
      clipping ? doubleBits(clipping->scale) : 0,
      clipping ? doubleBits(clipping->width) : 0,
      accumulator.clipped_sample_count_,
      accumulator.clipping_consumed_sample_count_};
  const auto records = gatherRecords(communicator_, local_record);

  const std::size_t failed_rank = firstFailedRank(records);
  if (failed_rank != records.size())
  {
    accumulator.abort();
    throwConsensusFailure("production", failed_rank,
                          static_cast<ConsensusReason>(records[failed_rank][0]),
                          failed_rank == 0 ? local_failure : std::exception_ptr{}, records.size());
  }
  const std::size_t mismatch_rank = firstContributionMismatch(records, 5, 19, 20);
  if (mismatch_rank != records.size())
  {
    accumulator.abort();
    throwMetadataMismatch("contribution", mismatch_rank);
  }

  // Counts are gathered exactly before floating-point collectives so overflow and a
  // globally empty population are diagnosed uniformly without unsigned wraparound.
  std::uint64_t global_sample_count = 0;
  std::uint64_t global_clipped_sample_count = 0;
  std::uint64_t global_clipping_consumed_count = 0;
  for (const auto& record : records)
  {
    if (record[5] > std::numeric_limits<std::uint64_t>::max() - global_sample_count)
    {
      accumulator.abort();
      throw std::overflow_error("Distributed energy-gradient sample count overflow");
    }
    global_sample_count += record[5];
    if (record[19] > std::numeric_limits<std::uint64_t>::max() -
            global_clipped_sample_count ||
        record[20] > std::numeric_limits<std::uint64_t>::max() -
            global_clipping_consumed_count)
    {
      accumulator.abort();
      throw std::overflow_error("Distributed energy-gradient clipping count overflow");
    }
    global_clipped_sample_count += record[19];
    global_clipping_consumed_count += record[20];
  }
  if (global_sample_count == 0 ||
      global_sample_count > std::numeric_limits<std::size_t>::max())
  {
    accumulator.abort();
    throw std::runtime_error("Distributed energy-gradient population is globally empty or too large");
  }
  if (clipping &&
      (global_clipping_consumed_count != global_sample_count ||
       global_sample_count != clipping->population))
  {
    accumulator.abort();
    throw std::runtime_error(
        "Distributed energy clipping transform did not consume its complete population");
  }

  try
  {
    if (communicator_)
    {
      // Preserve a fixed scalar-then-channel schedule on every rank.
      communicator_->allreduce_in_place(&accumulator.weight_sum_, 1);
      communicator_->allreduce_in_place(&accumulator.weighted_energy_sum_, 1);
      communicator_->allreduce_in_place(&accumulator.weighted_energy_norm_sum_, 1);
      if (clipping)
        communicator_->allreduce_in_place(&accumulator.clipped_weighted_energy_sum_, 1);

      for (std::size_t offset = 0; offset < accumulator.parameterCount();
           offset += policy_.maximum_chunk_size)
      {
        const std::size_t count =
            std::min(policy_.maximum_chunk_size, accumulator.parameterCount() - offset);
        communicator_->allreduce_in_place(accumulator.weighted_score_sum_.data() + offset,
                                          count);
        communicator_->allreduce_in_place(
            accumulator.weighted_energy_score_sum_.data() + offset, count);
        communicator_->allreduce_in_place(
            accumulator.weighted_energy_derivative_sum_.data() + offset, count);
      }
    }
  }
  catch (...)
  {
    accumulator.abort();
    throw;
  }

  const bool finite_scalars = std::isfinite(accumulator.weight_sum_) &&
      isFinite(accumulator.weighted_energy_sum_) &&
      std::isfinite(accumulator.weighted_energy_norm_sum_) &&
      (!clipping || isFinite(accumulator.clipped_weighted_energy_sum_));
  const auto all_finite = [](const std::vector<DerivativeValue>& values) {
    return std::all_of(values.begin(), values.end(), isFinite);
  };
  if (!finite_scalars || !all_finite(accumulator.weighted_score_sum_) ||
      !all_finite(accumulator.weighted_energy_score_sum_) ||
      !all_finite(accumulator.weighted_energy_derivative_sum_))
  {
    accumulator.abort();
    throw std::runtime_error("Distributed energy-gradient reduction produced a non-finite sum");
  }

  accumulator.sample_count_     = static_cast<std::size_t>(global_sample_count);
  accumulator.clipped_sample_count_ =
      static_cast<std::size_t>(global_clipped_sample_count);
  accumulator.clipping_consumed_sample_count_ =
      static_cast<std::size_t>(global_clipping_consumed_count);
  accumulator.reduction_domain_ = ReductionDomain::GLOBAL;
}

void DistributedParameterReduction::reduce(
    OrbitalPretrainingAccumulator& accumulator,
    std::exception_ptr local_failure) const
{
  ConsensusReason reason = local_failure ? ConsensusReason::LOCAL_EXCEPTION
                                         : ConsensusReason::NONE;
  if (reason == ConsensusReason::NONE &&
      (accumulator.finalized_ || accumulator.poisoned_ ||
       accumulator.reduction_domain_ == ReductionDomain::GLOBAL))
    reason = ConsensusReason::INCOMPLETE_CONTRIBUTION;
  if (reason == ConsensusReason::NONE &&
      (!isFiniteTrainingReal(accumulator.loss_sum_) ||
       !std::all_of(accumulator.gradient_sum_.begin(), accumulator.gradient_sum_.end(),
                    isFiniteTrainingReal)))
    reason = ConsensusReason::NONFINITE_CONTRIBUTION;

  const std::array<std::uint64_t, 11> local_record{
      static_cast<std::uint64_t>(reason),
      protocol_version,
      orbital_channel_contract,
      static_cast<std::uint64_t>(accumulator.reduction_domain_),
      accumulator.parameterCount(),
      accumulator.sample_count_,
      hashString(accumulator.provider_id_),
      hashString(accumulator.schema_fingerprint_),
      accumulator.parameter_version_,
      accumulator.target_fingerprint_,
      accumulator.loss_fingerprint_};
  const auto records = gatherRecords(communicator_, local_record);
  const std::size_t failed_rank = firstFailedRank(records);
  if (failed_rank != records.size())
  {
    accumulator.poison();
    throwConsensusFailure("orbital production", failed_rank,
                          static_cast<ConsensusReason>(records[failed_rank][0]),
                          failed_rank == 0 ? local_failure : std::exception_ptr{}, records.size());
  }
  const std::size_t mismatch_rank = firstContributionMismatch(records, 5, 5, 5);
  if (mismatch_rank != records.size())
  {
    accumulator.poison();
    throwMetadataMismatch("orbital contribution", mismatch_rank);
  }

  std::uint64_t global_sample_count = 0;
  for (const auto& record : records)
  {
    if (record[5] > std::numeric_limits<std::uint64_t>::max() - global_sample_count)
    {
      accumulator.poison();
      throw std::overflow_error("Distributed orbital-pretraining sample count overflow");
    }
    global_sample_count += record[5];
  }
  if (global_sample_count == 0 ||
      global_sample_count > std::numeric_limits<std::size_t>::max())
  {
    accumulator.poison();
    throw std::runtime_error("Distributed orbital-pretraining population is globally empty or too large");
  }

  try
  {
    if (communicator_)
    {
      communicator_->allreduce_in_place(&accumulator.loss_sum_, 1);
      for (std::size_t offset = 0; offset < accumulator.parameterCount();
           offset += policy_.maximum_chunk_size)
      {
        const std::size_t count =
            std::min(policy_.maximum_chunk_size, accumulator.parameterCount() - offset);
        communicator_->allreduce_in_place(accumulator.gradient_sum_.data() + offset, count);
      }
    }
  }
  catch (...)
  {
    accumulator.poison();
    throw;
  }

  if (!isFiniteTrainingReal(accumulator.loss_sum_) ||
      !std::all_of(accumulator.gradient_sum_.begin(), accumulator.gradient_sum_.end(),
                   isFiniteTrainingReal))
  {
    accumulator.poison();
    throw std::runtime_error("Distributed orbital-pretraining reduction produced a non-finite sum");
  }
  accumulator.sample_count_ = static_cast<std::size_t>(global_sample_count);
  accumulator.reduction_domain_ = ReductionDomain::GLOBAL;
}

void DistributedParameterReduction::validateCandidate(
    const StructuredParameterSchema& schema,
    const StructuredParameterSnapshot& parameters,
    const StructuredParameterSnapshot* candidate,
    std::exception_ptr local_failure) const
{
  ConsensusReason reason = local_failure ? ConsensusReason::LOCAL_EXCEPTION
                                         : ConsensusReason::NONE;
  if (reason == ConsensusReason::NONE && candidate == nullptr)
    reason = ConsensusReason::INVALID_CANDIDATE;
  if (reason == ConsensusReason::NONE &&
      (candidate->schema_fingerprint != schema.fingerprint() ||
       candidate->version != parameters.version ||
       candidate->values.size() != schema.parameterCount() ||
       !std::all_of(candidate->values.begin(), candidate->values.end(),
                    isFiniteTrainingReal)))
    reason = ConsensusReason::INVALID_CANDIDATE;

  const std::array<std::uint64_t, 7> local_record{
      static_cast<std::uint64_t>(reason),
      protocol_version,
      hashString(schema.fingerprint()),
      parameters.version,
      schema.parameterCount(),
      candidate ? hashSnapshot(*candidate) : 0,
      candidate ? candidate->version : 0};
  const auto records = gatherRecords(communicator_, local_record);

  const std::size_t failed_rank = firstFailedRank(records);
  if (failed_rank != records.size())
    throwConsensusFailure("update proposal", failed_rank,
                          static_cast<ConsensusReason>(records[failed_rank][0]),
                          failed_rank == 0 ? local_failure : std::exception_ptr{}, records.size());
  const std::size_t mismatch_rank = firstMismatchingRank(records);
  if (mismatch_rank != records.size())
    throwMetadataMismatch("candidate", mismatch_rank);
}

std::size_t DistributedParameterReduction::completePublication(
    std::size_t local_version,
    std::exception_ptr local_failure) const
{
  const ConsensusReason reason = local_failure ? ConsensusReason::LOCAL_EXCEPTION
                                               : ConsensusReason::NONE;
  const std::array<std::uint64_t, 3> local_record{
      static_cast<std::uint64_t>(reason), protocol_version, local_version};
  const auto records = gatherRecords(communicator_, local_record);

  const std::size_t failed_rank = firstFailedRank(records);
  if (failed_rank != records.size())
  {
    if (records.size() == 1 && local_failure)
      std::rethrow_exception(local_failure);
    std::ostringstream message;
    message << "Fatal distributed parameter-publication divergence: rank " << failed_rank
            << " did not publish";
    throw std::runtime_error(message.str());
  }
  const std::size_t mismatch_rank = firstMismatchingRank(records);
  if (mismatch_rank != records.size())
    throw std::runtime_error("Fatal distributed parameter-publication version divergence");
  return static_cast<std::size_t>(records.front()[2]);
}

} // namespace qmcplusplus::wftrain
