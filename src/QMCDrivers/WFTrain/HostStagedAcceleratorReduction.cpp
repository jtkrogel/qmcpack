//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file HostStagedAcceleratorReduction.cpp
 * @brief Fixed-schedule host-staged MPI reduction implementation.
 */

#include "QMCDrivers/WFTrain/HostStagedAcceleratorReduction.h"

#include "Message/CommOperators.h"
#include "Message/Communicate.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qmcplusplus::wftrain
{
namespace
{

constexpr std::uint64_t host_staged_protocol = UINT64_C(1);
constexpr std::uint64_t host_staged_contract = UINT64_C(0x4853544147454431); // "HSTAGED1"

/// Return whether every complex128 element is finite.
bool allFinite(const DerivativeValue* values, std::size_t count) noexcept
{
  return std::all_of(values, values + count, [](DerivativeValue value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
  });
}

/// Validate the mandatory host route and its caller-owned staging intervals.
void validateInputs(const Communicate& communicator,
                    const AcceleratorDistributedPlan& plan,
                    const std::array<AcceleratorHostStagingSlot, 2>& staging_slots)
{
  if (plan.transport != AcceleratorCollectiveTransport::HOST_STAGED_MPI)
    throw std::invalid_argument("Host-staged reduction requires a host-staged distributed plan");
  if (plan.collective_element_bytes != sizeof(DerivativeValue))
    throw std::invalid_argument("Host-staged reduction requires complex128 collective elements");
  if (plan.participant_count != static_cast<std::size_t>(communicator.size()))
    throw std::invalid_argument("Host-staged reduction communicator size does not match the plan");

  for (std::size_t slot = 0; slot < plan.buffer_slots; ++slot)
  {
    if (staging_slots[slot].values == nullptr ||
        staging_slots[slot].capacity < plan.chunk_elements)
      throw std::invalid_argument("Host-staged reduction requires an adequate pinned slot");
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(staging_slots[slot].values);
    if (staging_slots[slot].capacity >
        (std::numeric_limits<std::uintptr_t>::max() - begin) / sizeof(DerivativeValue))
      throw std::overflow_error("Host-staged reduction slot address range overflows");
    const std::uintptr_t end = begin + staging_slots[slot].capacity * sizeof(DerivativeValue);
    for (std::size_t prior = 0; prior < slot; ++prior)
    {
      const std::uintptr_t prior_begin =
          reinterpret_cast<std::uintptr_t>(staging_slots[prior].values);
      const std::uintptr_t prior_end =
          prior_begin + staging_slots[prior].capacity * sizeof(DerivativeValue);
      if (begin < prior_end && prior_begin < end)
        throw std::invalid_argument("Host-staged reduction slots must not alias");
    }
  }
}

} // namespace

HostStagedAcceleratorReduction::HostStagedAcceleratorReduction(
    Communicate& communicator,
    AcceleratorDistributedPlan plan,
    AcceleratorCollectiveCopyQueue& copy_queue,
    std::array<AcceleratorHostStagingSlot, 2> staging_slots)
    : communicator_(communicator),
      copy_queue_(copy_queue),
      state_(std::move(plan)),
      staging_slots_(std::move(staging_slots)),
      gathered_records_(static_cast<std::size_t>(communicator.size()) * local_record_.size())
{
  validateInputs(communicator_, state_.plan(), staging_slots_);
}

std::size_t HostStagedAcceleratorReduction::consensus(
    const AcceleratorCollectiveToken& token,
    ConsensusPhase phase,
    bool local_success)
{
  local_record_ = {local_success ? UINT64_C(0) : UINT64_C(1),
                   host_staged_protocol,
                   host_staged_contract,
                   state_.plan().fingerprint,
                   state_.plan().parameter_version,
                   state_.generation(),
                   token.chunk.ordinal,
                   static_cast<std::uint64_t>(phase)};
  communicator_.allgather(local_record_.data(), gathered_records_.data(),
                          static_cast<int>(local_record_.size()));

  const std::size_t participants = state_.plan().participant_count;
  for (std::size_t rank = 0; rank < participants; ++rank)
  {
    const std::uint64_t* record = gathered_records_.data() + rank * local_record_.size();
    for (std::size_t field = 1; field < local_record_.size(); ++field)
      if (record[field] != local_record_[field])
        throw std::runtime_error("Host-staged reduction control metadata mismatch");
  }
  for (std::size_t rank = 0; rank < participants; ++rank)
    if (gathered_records_[rank * local_record_.size()] != 0)
      return rank;
  return participants;
}

[[noreturn]] void HostStagedAcceleratorReduction::failAndRecover(
    AcceleratorCollectiveFailure failure,
    std::size_t failing_rank,
    const char* stage)
{
  state_.failIteration(failure);
  copy_queue_.drainAfterFailure();
  state_.recoverAfterExternalDrain();

  std::ostringstream message;
  message << "Host-staged accelerator reduction " << stage << " failed on rank "
          << failing_rank;
  throw std::runtime_error(message.str());
}

void HostStagedAcceleratorReduction::reduceComplexVector(
    DerivativeValue* device_values,
    std::size_t count,
    std::uint64_t plan_fingerprint,
    std::size_t parameter_version)
{
  if (device_values == nullptr || count != state_.plan().parameter_count)
    throw std::invalid_argument("Host-staged reduction device vector does not match the plan");

  state_.beginIteration(plan_fingerprint, parameter_version);
  try
  {
    for (const AcceleratorChunkDescriptor& chunk : state_.plan().chunks)
    {
      const AcceleratorCollectiveToken token = state_.acquireProductionSlot();
      if (!(token.chunk == chunk))
        throw std::logic_error("Host-staged reduction received a noncanonical state token");
      state_.markLocalReady(token);

      AcceleratorHostStagingSlot& staging = staging_slots_[token.slot];
      bool copy_succeeded = true;
      state_.beginDeviceToHost(token);
      try
      {
        copy_queue_.submitDeviceToHost(token.slot, staging.values,
                                       device_values + chunk.offset, chunk.count);
        copy_queue_.waitDeviceToHost(token.slot);
      }
      catch (...)
      {
        copy_succeeded = false;
      }
      if (copy_succeeded)
        state_.completeDeviceToHost(token);

      std::size_t failed_rank = consensus(
          token, ConsensusPhase::DEVICE_TO_HOST,
          copy_succeeded && allFinite(staging.values, chunk.count));
      if (failed_rank != state_.plan().participant_count)
        failAndRecover(AcceleratorCollectiveFailure::DEVICE_COPY, failed_rank,
                       "device-to-host copy");

      state_.beginCollective(token);
      communicator_.allreduce_in_place(staging.values, chunk.count);
      state_.completeCollective(token);

      failed_rank = consensus(token, ConsensusPhase::REDUCED_HOST,
                              allFinite(staging.values, chunk.count));
      if (failed_rank != state_.plan().participant_count)
        failAndRecover(AcceleratorCollectiveFailure::COLLECTIVE, failed_rank,
                       "collective result");

      copy_succeeded = true;
      state_.beginHostToDevice(token);
      try
      {
        copy_queue_.submitHostToDevice(token.slot, device_values + chunk.offset,
                                       staging.values, chunk.count);
        copy_queue_.waitHostToDevice(token.slot);
      }
      catch (...)
      {
        copy_succeeded = false;
      }
      if (copy_succeeded)
        state_.completeHostToDevice(token);

      failed_rank = consensus(token, ConsensusPhase::HOST_TO_DEVICE, copy_succeeded);
      if (failed_rank != state_.plan().participant_count)
        failAndRecover(AcceleratorCollectiveFailure::DEVICE_COPY, failed_rank,
                       "host-to-device copy");
      state_.releaseResult(token);
    }
    state_.finishIteration();
  }
  catch (...)
  {
    if (state_.iterationState() == AcceleratorCollectiveIterationState::ACTIVE)
    {
      state_.failIteration(AcceleratorCollectiveFailure::COLLECTIVE);
      copy_queue_.drainAfterFailure();
      state_.recoverAfterExternalDrain();
    }
    throw;
  }
}

} // namespace qmcplusplus::wftrain
