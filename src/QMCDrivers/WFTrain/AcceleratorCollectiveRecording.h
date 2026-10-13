//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file AcceleratorCollectiveRecording.h
 * @brief Offline recording transport for distributed accelerator collectives.
 *
 * Opaque buffer identities make copy, event, and MPI ordering testable without
 * dereferencing memory or invoking a GPU/MPI runtime.  This class is validation
 * scaffolding and does not establish a production transport capability.
 */

#ifndef QMCPLUSPLUS_ACCELERATOR_COLLECTIVE_RECORDING_H
#define QMCPLUSPLUS_ACCELERATOR_COLLECTIVE_RECORDING_H

#include "QMCDrivers/WFTrain/AcceleratorCollectiveState.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace qmcplusplus::wftrain
{

/// Describe the only memory domains admitted by the first transport contract.
enum class AcceleratorCollectiveMemoryDomain : std::uint8_t
{
  DEVICE,
  PINNED_HOST
};

/// Describe one prepared buffer without exposing or dereferencing its pointer.
struct AcceleratorCollectiveBufferRecord
{
  std::uint64_t identity = 0;
  AcceleratorCollectiveMemoryDomain domain = AcceleratorCollectiveMemoryDomain::DEVICE;
  std::size_t capacity_bytes = 0;
};

/// Bind the prepared device and optional pinned-host buffers for one slot.
struct AcceleratorCollectivePreparedSlot
{
  AcceleratorCollectiveBufferRecord device;
  AcceleratorCollectiveBufferRecord host;
};

/// Classify one observable operation in the offline transport trace.
enum class AcceleratorCollectiveRecordedOperation : std::uint8_t
{
  PRODUCTION_ACQUIRE,
  PRODUCTION_COMPLETE,
  DEVICE_TO_HOST_SUBMIT,
  DEVICE_TO_HOST_COMPLETE,
  HOST_MPI_SUBMIT,
  DEVICE_MPI_SUBMIT,
  MPI_COMPLETE,
  HOST_TO_DEVICE_SUBMIT,
  HOST_TO_DEVICE_COMPLETE,
  RESULT_RELEASE,
  FAILURE,
  EXTERNAL_DRAIN
};

/// Retain the exact ownership token and buffers associated with one operation.
struct AcceleratorCollectiveOperationRecord
{
  AcceleratorCollectiveRecordedOperation operation =
      AcceleratorCollectiveRecordedOperation::PRODUCTION_ACQUIRE;
  AcceleratorCollectiveToken token;
  std::uint64_t source_identity = 0;
  std::uint64_t destination_identity = 0;
  std::size_t byte_count = 0;
  AcceleratorCollectiveFailure failure = AcceleratorCollectiveFailure::NONE;
};

/** Drive the pure ownership state while recording, rather than executing, I/O.
 *
 * Trace storage is reserved at construction.  Each warmed method checks remaining
 * capacity before changing state, so successful recording does not allocate and a
 * full trace cannot leave the ownership machine partially advanced.
 */
class AcceleratorCollectiveRecordingTransport
{
public:
  /// Validate prepared buffer identities/domains/capacities and reserve a trace.
  AcceleratorCollectiveRecordingTransport(
      AcceleratorDistributedPlan plan,
      std::array<AcceleratorCollectivePreparedSlot, 2> slots,
      std::size_t trace_capacity = 0);

  AcceleratorCollectiveRecordingTransport(const AcceleratorCollectiveRecordingTransport&) = delete;
  AcceleratorCollectiveRecordingTransport& operator=(const AcceleratorCollectiveRecordingTransport&) = delete;
  AcceleratorCollectiveRecordingTransport(AcceleratorCollectiveRecordingTransport&&) = delete;
  AcceleratorCollectiveRecordingTransport& operator=(AcceleratorCollectiveRecordingTransport&&) = delete;

  /// Start one exact plan/version iteration.
  void beginIteration(std::uint64_t plan_fingerprint, std::size_t parameter_version);

  /// Acquire the next device slot and record local production ownership.
  AcceleratorCollectiveToken acquireProductionSlot();

  /// Record completion of local device production.
  void completeProduction(const AcceleratorCollectiveToken& token);

  /// Record submission and completion of a host-staging copy.
  void submitDeviceToHost(const AcceleratorCollectiveToken& token);
  void completeDeviceToHost(const AcceleratorCollectiveToken& token);

  /// Record submission and completion of host- or device-buffer MPI.
  void submitCollective(const AcceleratorCollectiveToken& token);
  void completeCollective(const AcceleratorCollectiveToken& token);

  /// Record submission and completion of the reduced host result copy.
  void submitHostToDevice(const AcceleratorCollectiveToken& token);
  void completeHostToDevice(const AcceleratorCollectiveToken& token);

  /// Record canonical result consumption and release.
  void releaseResult(const AcceleratorCollectiveToken& token);

  /// Complete a fully released iteration.
  void finishIteration();

  /// Poison the iteration and record its classified failure.
  void failIteration(AcceleratorCollectiveFailure failure);

  /// Record an external drain and reset poisoned slots for a retry.
  void recoverAfterExternalDrain();

  /// Clear diagnostic history only while the ownership machine is idle.
  void clearTrace();

  /// Return the pure ownership state and immutable trace for tests/diagnostics.
  const AcceleratorCollectiveState& state() const noexcept { return state_; }
  const std::vector<AcceleratorCollectiveOperationRecord>& trace() const noexcept { return trace_; }

private:
  /// Return the planned byte count for one exact token without overflow.
  std::size_t tokenBytes(const AcceleratorCollectiveToken& token) const;

  /// Require one trace slot before any ownership transition is attempted.
  void requireTraceCapacity(const char* operation) const;

  /// Append one prevalidated record without growing the prepared trace.
  void append(AcceleratorCollectiveRecordedOperation operation,
              const AcceleratorCollectiveToken& token,
              std::uint64_t source,
              std::uint64_t destination,
              std::size_t bytes,
              AcceleratorCollectiveFailure failure = AcceleratorCollectiveFailure::NONE);

  AcceleratorCollectiveState state_;
  std::array<AcceleratorCollectivePreparedSlot, 2> slots_;
  std::vector<AcceleratorCollectiveOperationRecord> trace_;
};

} // namespace qmcplusplus::wftrain

#endif // QMCPLUSPLUS_ACCELERATOR_COLLECTIVE_RECORDING_H
