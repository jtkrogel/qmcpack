//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file AcceleratorCollectiveRecording.cpp
 * @brief Offline buffer and operation recording for accelerator collectives.
 */

#include "QMCDrivers/WFTrain/AcceleratorCollectiveRecording.h"

#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace qmcplusplus::wftrain
{
namespace
{

/// Multiply extents without allowing silent byte-count wraparound.
std::size_t checkedMultiply(std::size_t lhs, std::size_t rhs, const char* description)
{
  if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs)
    throw std::overflow_error(std::string("Accelerator recording overflow for ") + description);
  return lhs * rhs;
}

/// Add extents without allowing silent trace-capacity wraparound.
std::size_t checkedAdd(std::size_t lhs, std::size_t rhs, const char* description)
{
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs)
    throw std::overflow_error(std::string("Accelerator recording overflow for ") + description);
  return lhs + rhs;
}

/// Compute a default trace sufficient for one host-staged pass plus failure records.
std::size_t defaultTraceCapacity(const AcceleratorDistributedPlan& plan)
{
  return checkedAdd(checkedMultiply(plan.chunks.size(), 9, "default trace"), 2,
                    "default failure trace");
}

/// Validate all buffers needed by the selected one- or two-slot route.
void validatePreparedSlots(const AcceleratorDistributedPlan& plan,
                           const std::array<AcceleratorCollectivePreparedSlot, 2>& slots)
{
  const std::size_t required_bytes = checkedMultiply(
      plan.chunk_elements, plan.collective_element_bytes, "prepared slot bytes");
  std::set<std::uint64_t> identities;
  for (std::size_t slot = 0; slot < plan.buffer_slots; ++slot)
  {
    const AcceleratorCollectivePreparedSlot& prepared = slots[slot];
    if (prepared.device.identity == 0 ||
        prepared.device.domain != AcceleratorCollectiveMemoryDomain::DEVICE ||
        prepared.device.capacity_bytes < required_bytes)
      throw std::invalid_argument("Accelerator recording requires an adequate device buffer per slot");
    if (!identities.insert(prepared.device.identity).second)
      throw std::invalid_argument("Accelerator recording buffer identities must not alias");

    if (plan.transport == AcceleratorCollectiveTransport::HOST_STAGED_MPI)
    {
      if (prepared.host.identity == 0 ||
          prepared.host.domain != AcceleratorCollectiveMemoryDomain::PINNED_HOST ||
          prepared.host.capacity_bytes < required_bytes)
        throw std::invalid_argument("Host-staged recording requires an adequate pinned buffer per slot");
      if (!identities.insert(prepared.host.identity).second)
        throw std::invalid_argument("Accelerator recording buffer identities must not alias");
    }
  }
}

} // namespace

AcceleratorCollectiveRecordingTransport::AcceleratorCollectiveRecordingTransport(
    AcceleratorDistributedPlan plan,
    std::array<AcceleratorCollectivePreparedSlot, 2> slots,
    std::size_t trace_capacity)
    : state_(std::move(plan)), slots_(std::move(slots))
{
  validatePreparedSlots(state_.plan(), slots_);
  const std::size_t capacity = trace_capacity == 0 ? defaultTraceCapacity(state_.plan()) : trace_capacity;
  if (capacity == 0)
    throw std::invalid_argument("Accelerator recording trace capacity must be positive");
  trace_.reserve(capacity);
}

std::size_t AcceleratorCollectiveRecordingTransport::tokenBytes(
    const AcceleratorCollectiveToken& token) const
{
  return checkedMultiply(token.chunk.count, state_.plan().collective_element_bytes,
                         "token byte count");
}

void AcceleratorCollectiveRecordingTransport::requireTraceCapacity(const char* operation) const
{
  if (trace_.size() == trace_.capacity())
    throw std::runtime_error(std::string("Accelerator recording trace is full before ") + operation);
}

void AcceleratorCollectiveRecordingTransport::append(
    AcceleratorCollectiveRecordedOperation operation,
    const AcceleratorCollectiveToken& token,
    std::uint64_t source,
    std::uint64_t destination,
    std::size_t bytes,
    AcceleratorCollectiveFailure failure)
{
  trace_.push_back({operation, token, source, destination, bytes, failure});
}

void AcceleratorCollectiveRecordingTransport::beginIteration(std::uint64_t plan_fingerprint,
                                                              std::size_t parameter_version)
{
  state_.beginIteration(plan_fingerprint, parameter_version);
}

AcceleratorCollectiveToken AcceleratorCollectiveRecordingTransport::acquireProductionSlot()
{
  requireTraceCapacity("production acquire");
  const AcceleratorCollectiveToken token = state_.acquireProductionSlot();
  const AcceleratorCollectiveBufferRecord& device = slots_[token.slot].device;
  append(AcceleratorCollectiveRecordedOperation::PRODUCTION_ACQUIRE, token,
         device.identity, device.identity, tokenBytes(token));
  return token;
}

void AcceleratorCollectiveRecordingTransport::completeProduction(
    const AcceleratorCollectiveToken& token)
{
  requireTraceCapacity("production completion");
  state_.markLocalReady(token);
  const AcceleratorCollectiveBufferRecord& device = slots_[token.slot].device;
  append(AcceleratorCollectiveRecordedOperation::PRODUCTION_COMPLETE, token,
         device.identity, device.identity, tokenBytes(token));
}

void AcceleratorCollectiveRecordingTransport::submitDeviceToHost(
    const AcceleratorCollectiveToken& token)
{
  requireTraceCapacity("device-to-host submission");
  state_.beginDeviceToHost(token);
  append(AcceleratorCollectiveRecordedOperation::DEVICE_TO_HOST_SUBMIT, token,
         slots_[token.slot].device.identity, slots_[token.slot].host.identity, tokenBytes(token));
}

void AcceleratorCollectiveRecordingTransport::completeDeviceToHost(
    const AcceleratorCollectiveToken& token)
{
  requireTraceCapacity("device-to-host completion");
  state_.completeDeviceToHost(token);
  const AcceleratorCollectiveBufferRecord& host = slots_[token.slot].host;
  append(AcceleratorCollectiveRecordedOperation::DEVICE_TO_HOST_COMPLETE, token,
         host.identity, host.identity, tokenBytes(token));
}

void AcceleratorCollectiveRecordingTransport::submitCollective(
    const AcceleratorCollectiveToken& token)
{
  requireTraceCapacity("collective submission");
  state_.beginCollective(token);
  const bool host_staged = state_.plan().transport == AcceleratorCollectiveTransport::HOST_STAGED_MPI;
  const AcceleratorCollectiveBufferRecord& buffer =
      host_staged ? slots_[token.slot].host : slots_[token.slot].device;
  append(host_staged ? AcceleratorCollectiveRecordedOperation::HOST_MPI_SUBMIT
                     : AcceleratorCollectiveRecordedOperation::DEVICE_MPI_SUBMIT,
         token, buffer.identity, buffer.identity, tokenBytes(token));
}

void AcceleratorCollectiveRecordingTransport::completeCollective(
    const AcceleratorCollectiveToken& token)
{
  requireTraceCapacity("collective completion");
  state_.completeCollective(token);
  const bool host_staged = state_.plan().transport == AcceleratorCollectiveTransport::HOST_STAGED_MPI;
  const AcceleratorCollectiveBufferRecord& buffer =
      host_staged ? slots_[token.slot].host : slots_[token.slot].device;
  append(AcceleratorCollectiveRecordedOperation::MPI_COMPLETE, token,
         buffer.identity, buffer.identity, tokenBytes(token));
}

void AcceleratorCollectiveRecordingTransport::submitHostToDevice(
    const AcceleratorCollectiveToken& token)
{
  requireTraceCapacity("host-to-device submission");
  state_.beginHostToDevice(token);
  append(AcceleratorCollectiveRecordedOperation::HOST_TO_DEVICE_SUBMIT, token,
         slots_[token.slot].host.identity, slots_[token.slot].device.identity, tokenBytes(token));
}

void AcceleratorCollectiveRecordingTransport::completeHostToDevice(
    const AcceleratorCollectiveToken& token)
{
  requireTraceCapacity("host-to-device completion");
  state_.completeHostToDevice(token);
  const AcceleratorCollectiveBufferRecord& device = slots_[token.slot].device;
  append(AcceleratorCollectiveRecordedOperation::HOST_TO_DEVICE_COMPLETE, token,
         device.identity, device.identity, tokenBytes(token));
}

void AcceleratorCollectiveRecordingTransport::releaseResult(
    const AcceleratorCollectiveToken& token)
{
  requireTraceCapacity("result release");
  const std::size_t bytes = tokenBytes(token);
  const std::uint64_t device_identity = slots_[token.slot].device.identity;
  state_.releaseResult(token);
  append(AcceleratorCollectiveRecordedOperation::RESULT_RELEASE, token,
         device_identity, device_identity, bytes);
}

void AcceleratorCollectiveRecordingTransport::finishIteration()
{
  state_.finishIteration();
}

void AcceleratorCollectiveRecordingTransport::failIteration(
    AcceleratorCollectiveFailure failure)
{
  requireTraceCapacity("failure recording");
  state_.failIteration(failure);
  append(AcceleratorCollectiveRecordedOperation::FAILURE, {}, 0, 0, 0, failure);
}

void AcceleratorCollectiveRecordingTransport::recoverAfterExternalDrain()
{
  requireTraceCapacity("external-drain recording");
  const AcceleratorCollectiveFailure failure = state_.failure();
  state_.recoverAfterExternalDrain();
  append(AcceleratorCollectiveRecordedOperation::EXTERNAL_DRAIN, {}, 0, 0, 0, failure);
}

void AcceleratorCollectiveRecordingTransport::clearTrace()
{
  if (state_.iterationState() != AcceleratorCollectiveIterationState::IDLE)
    throw std::logic_error("Accelerator recording trace can be cleared only while idle");
  trace_.clear();
}

} // namespace qmcplusplus::wftrain
