//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file AcceleratorCollectiveState.cpp
 * @brief State transitions for bounded accelerator collective slots.
 */

#include "QMCDrivers/WFTrain/AcceleratorCollectiveState.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace qmcplusplus::wftrain
{
namespace
{

/// Throw one consistent transition error without mutating coordinator state.
[[noreturn]] void throwTransition(const char* operation, const char* reason)
{
  throw std::logic_error(std::string("Accelerator collective ") + operation + " rejected: " + reason);
}

/// Validate that a prepared plan has one exact, bounded canonical chunk schedule.
void validatePlan(const AcceleratorDistributedPlan& plan)
{
  if (plan.fingerprint == 0)
    throw std::invalid_argument("Accelerator collective plan fingerprint must be nonzero");
  if (plan.parameter_count == 0 || plan.collective_element_bytes == 0)
    throw std::invalid_argument("Accelerator collective plan requires nonempty typed parameters");
  if (plan.buffer_slots == 0 || plan.buffer_slots > 2)
    throw std::invalid_argument("Accelerator collective plan must contain one or two slots");
  if (plan.chunk_elements == 0 || plan.chunks.empty())
    throw std::invalid_argument("Accelerator collective plan has no canonical chunks");

  std::size_t expected_offset = 0;
  for (std::size_t ordinal = 0; ordinal < plan.chunks.size(); ++ordinal)
  {
    const AcceleratorChunkDescriptor& chunk = plan.chunks[ordinal];
    if (expected_offset > plan.parameter_count || chunk.ordinal != ordinal ||
        chunk.offset != expected_offset || chunk.count == 0 ||
        chunk.count > plan.chunk_elements || chunk.count > plan.parameter_count - expected_offset)
      throw std::invalid_argument("Accelerator collective plan has an invalid canonical chunk schedule");
    expected_offset += chunk.count;
  }
  if (expected_offset != plan.parameter_count)
    throw std::invalid_argument("Accelerator collective chunks do not cover the parameter vector");
}

} // namespace

AcceleratorCollectiveState::AcceleratorCollectiveState(AcceleratorDistributedPlan plan)
    : plan_(std::move(plan))
{
  validatePlan(plan_);
}

void AcceleratorCollectiveState::requireActive(const char* operation) const
{
  if (iteration_state_ == AcceleratorCollectiveIterationState::FAILED)
    throwTransition(operation, "the iteration is poisoned");
  if (iteration_state_ != AcceleratorCollectiveIterationState::ACTIVE)
    throwTransition(operation, "no iteration is active");
}

AcceleratorCollectiveState::Slot& AcceleratorCollectiveState::requireToken(
    const AcceleratorCollectiveToken& token,
    AcceleratorCollectiveSlotState expected,
    const char* operation)
{
  requireActive(operation);
  if (token.plan_fingerprint != plan_.fingerprint || token.parameter_version != plan_.parameter_version ||
      token.generation != generation_ || token.slot >= plan_.buffer_slots)
    throwTransition(operation, "the token does not identify this live iteration");

  Slot& slot = slots_[token.slot];
  if (slot.state != expected || slot.token.plan_fingerprint != token.plan_fingerprint ||
      slot.token.parameter_version != token.parameter_version || slot.token.generation != token.generation ||
      slot.token.slot != token.slot || !(slot.token.chunk == token.chunk))
    throwTransition(operation, "the token does not own a slot in the required state");
  return slot;
}

void AcceleratorCollectiveState::beginIteration(std::uint64_t plan_fingerprint,
                                                 std::size_t parameter_version)
{
  if (iteration_state_ != AcceleratorCollectiveIterationState::IDLE)
    throwTransition("begin", "the previous iteration is not idle");
  if (plan_fingerprint != plan_.fingerprint || parameter_version != plan_.parameter_version)
    throwTransition("begin", "plan fingerprint or parameter version mismatch");
  if (generation_ == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("Accelerator collective generation overflow");

  for (std::size_t slot = 0; slot < plan_.buffer_slots; ++slot)
    if (slots_[slot].state != AcceleratorCollectiveSlotState::AVAILABLE)
      throwTransition("begin", "a prepared slot is not available");

  ++generation_;
  next_acquire_    = 0;
  next_collective_ = 0;
  next_release_    = 0;
  failure_         = AcceleratorCollectiveFailure::NONE;
  iteration_state_ = AcceleratorCollectiveIterationState::ACTIVE;
}

AcceleratorCollectiveToken AcceleratorCollectiveState::acquireProductionSlot()
{
  requireActive("production acquire");
  if (next_acquire_ == plan_.chunks.size())
    throwTransition("production acquire", "all canonical chunks have been issued");

  const std::size_t slot_index = next_acquire_ % plan_.buffer_slots;
  Slot& slot = slots_[slot_index];
  if (slot.state != AcceleratorCollectiveSlotState::AVAILABLE)
    throwTransition("production acquire", "the next bounded slot is still owned");

  slot.token = {plan_.fingerprint, plan_.parameter_version, generation_, slot_index,
                plan_.chunks[next_acquire_]};
  slot.state = AcceleratorCollectiveSlotState::PRODUCING;
  ++next_acquire_;
  return slot.token;
}

void AcceleratorCollectiveState::markLocalReady(const AcceleratorCollectiveToken& token)
{
  requireToken(token, AcceleratorCollectiveSlotState::PRODUCING, "local-ready publication").state =
      AcceleratorCollectiveSlotState::LOCAL_READY;
}

void AcceleratorCollectiveState::beginDeviceToHost(const AcceleratorCollectiveToken& token)
{
  if (plan_.transport != AcceleratorCollectiveTransport::HOST_STAGED_MPI)
    throwTransition("device-to-host begin", "the selected transport has no host staging");
  requireToken(token, AcceleratorCollectiveSlotState::LOCAL_READY, "device-to-host begin").state =
      AcceleratorCollectiveSlotState::DEVICE_TO_HOST;
}

void AcceleratorCollectiveState::completeDeviceToHost(const AcceleratorCollectiveToken& token)
{
  requireToken(token, AcceleratorCollectiveSlotState::DEVICE_TO_HOST, "device-to-host completion").state =
      AcceleratorCollectiveSlotState::HOST_READY;
}

void AcceleratorCollectiveState::beginCollective(const AcceleratorCollectiveToken& token)
{
  const AcceleratorCollectiveSlotState expected =
      plan_.transport == AcceleratorCollectiveTransport::HOST_STAGED_MPI
      ? AcceleratorCollectiveSlotState::HOST_READY
      : AcceleratorCollectiveSlotState::LOCAL_READY;
  Slot& slot = requireToken(token, expected, "collective begin");
  if (token.chunk.ordinal != next_collective_)
    throwTransition("collective begin", "chunks must enter collectives in canonical order");
  slot.state = AcceleratorCollectiveSlotState::COLLECTIVE;
}

void AcceleratorCollectiveState::completeCollective(const AcceleratorCollectiveToken& token)
{
  Slot& slot = requireToken(token, AcceleratorCollectiveSlotState::COLLECTIVE, "collective completion");
  if (token.chunk.ordinal != next_collective_)
    throwTransition("collective completion", "the completion is not the canonical in-flight chunk");
  ++next_collective_;
  slot.state = plan_.transport == AcceleratorCollectiveTransport::HOST_STAGED_MPI
      ? AcceleratorCollectiveSlotState::REDUCED_HOST
      : AcceleratorCollectiveSlotState::RESULT_READY;
}

void AcceleratorCollectiveState::beginHostToDevice(const AcceleratorCollectiveToken& token)
{
  if (plan_.transport != AcceleratorCollectiveTransport::HOST_STAGED_MPI)
    throwTransition("host-to-device begin", "the selected transport has no host staging");
  requireToken(token, AcceleratorCollectiveSlotState::REDUCED_HOST, "host-to-device begin").state =
      AcceleratorCollectiveSlotState::HOST_TO_DEVICE;
}

void AcceleratorCollectiveState::completeHostToDevice(const AcceleratorCollectiveToken& token)
{
  requireToken(token, AcceleratorCollectiveSlotState::HOST_TO_DEVICE, "host-to-device completion").state =
      AcceleratorCollectiveSlotState::RESULT_READY;
}

void AcceleratorCollectiveState::releaseResult(const AcceleratorCollectiveToken& token)
{
  Slot& slot = requireToken(token, AcceleratorCollectiveSlotState::RESULT_READY, "result release");
  if (token.chunk.ordinal != next_release_)
    throwTransition("result release", "results must be consumed in canonical order");
  ++next_release_;
  slot.state = AcceleratorCollectiveSlotState::AVAILABLE;
  slot.token = {};
}

void AcceleratorCollectiveState::finishIteration()
{
  requireActive("finish");
  if (next_acquire_ != plan_.chunks.size() || next_collective_ != plan_.chunks.size() ||
      next_release_ != plan_.chunks.size())
    throwTransition("finish", "the canonical chunk schedule is incomplete");
  for (std::size_t slot = 0; slot < plan_.buffer_slots; ++slot)
    if (slots_[slot].state != AcceleratorCollectiveSlotState::AVAILABLE)
      throwTransition("finish", "a prepared slot is still owned");
  iteration_state_ = AcceleratorCollectiveIterationState::IDLE;
}

void AcceleratorCollectiveState::failIteration(AcceleratorCollectiveFailure failure)
{
  requireActive("failure report");
  if (failure == AcceleratorCollectiveFailure::NONE)
    throwTransition("failure report", "a failure category is required");
  failure_         = failure;
  iteration_state_ = AcceleratorCollectiveIterationState::FAILED;
  for (std::size_t slot = 0; slot < plan_.buffer_slots; ++slot)
    if (slots_[slot].state != AcceleratorCollectiveSlotState::AVAILABLE)
      slots_[slot].state = AcceleratorCollectiveSlotState::POISONED;
}

void AcceleratorCollectiveState::cancelIteration()
{
  failIteration(AcceleratorCollectiveFailure::CANCELLED);
}

void AcceleratorCollectiveState::recoverAfterExternalDrain()
{
  if (iteration_state_ != AcceleratorCollectiveIterationState::FAILED)
    throwTransition("recovery", "the iteration is not poisoned");
  for (std::size_t slot = 0; slot < plan_.buffer_slots; ++slot)
  {
    slots_[slot].state = AcceleratorCollectiveSlotState::AVAILABLE;
    slots_[slot].token = {};
  }
  next_acquire_    = 0;
  next_collective_ = 0;
  next_release_    = 0;
  iteration_state_ = AcceleratorCollectiveIterationState::IDLE;
}

AcceleratorCollectiveSlotState AcceleratorCollectiveState::slotState(std::size_t slot) const
{
  if (slot >= plan_.buffer_slots)
    throw std::out_of_range("Accelerator collective slot index is outside the prepared plan");
  return slots_[slot].state;
}

} // namespace qmcplusplus::wftrain
