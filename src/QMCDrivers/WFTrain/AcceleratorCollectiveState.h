//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file AcceleratorCollectiveState.h
 * @brief Pure ownership and ordering state for bounded accelerator collectives.
 *
 * This layer contains no MPI or accelerator-runtime objects.  Runtime transports
 * drive it around copies, events, and collectives so invalid ordering fails before
 * an external operation is submitted.  Warm transitions use fixed storage and do
 * not allocate; construction and exception paths are outside that guarantee.
 */

#ifndef QMCPLUSPLUS_ACCELERATOR_COLLECTIVE_STATE_H
#define QMCPLUSPLUS_ACCELERATOR_COLLECTIVE_STATE_H

#include "QMCDrivers/WFTrain/AcceleratorTrainingPlan.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace qmcplusplus::wftrain
{

/// Identify the externally visible phase of one prepared collective iteration.
enum class AcceleratorCollectiveIterationState : std::uint8_t
{
  IDLE,
  ACTIVE,
  FAILED
};

/// Identify ownership of one fixed collective buffer slot.
enum class AcceleratorCollectiveSlotState : std::uint8_t
{
  AVAILABLE,
  PRODUCING,
  LOCAL_READY,
  DEVICE_TO_HOST,
  HOST_READY,
  COLLECTIVE,
  REDUCED_HOST,
  HOST_TO_DEVICE,
  RESULT_READY,
  POISONED
};

/// Classify an explicitly reported external failure without retaining text.
enum class AcceleratorCollectiveFailure : std::uint8_t
{
  NONE,
  LOCAL_PRODUCTION,
  DEVICE_COPY,
  COLLECTIVE,
  LOCAL_CONSUMPTION,
  CANCELLED
};

/// Bind a caller operation to one exact plan, iteration, slot, and chunk.
struct AcceleratorCollectiveToken
{
  std::uint64_t plan_fingerprint = 0;
  std::size_t parameter_version  = 0;
  std::size_t generation         = 0;
  std::size_t slot               = 0;
  AcceleratorChunkDescriptor chunk;
};

/** Coordinate one prepared one- or two-slot collective schedule.
 *
 * The class is deliberately noncopyable: one runtime owner must serialize calls
 * around its queue/event and MPI operations.  ``recoverAfterExternalDrain`` is an
 * explicit assertion by that owner that all work submitted before a failure has
 * completed or been cancelled and that the buffers are safe to reuse.
 */
class AcceleratorCollectiveState
{
public:
  /// Validate and retain one immutable distributed plan.
  explicit AcceleratorCollectiveState(AcceleratorDistributedPlan plan);

  AcceleratorCollectiveState(const AcceleratorCollectiveState&) = delete;
  AcceleratorCollectiveState& operator=(const AcceleratorCollectiveState&) = delete;
  AcceleratorCollectiveState(AcceleratorCollectiveState&&) = delete;
  AcceleratorCollectiveState& operator=(AcceleratorCollectiveState&&) = delete;

  /// Start a fresh iteration for the exact prepared plan and parameter version.
  void beginIteration(std::uint64_t plan_fingerprint, std::size_t parameter_version);

  /// Acquire the next canonical chunk for local production.
  AcceleratorCollectiveToken acquireProductionSlot();

  /// Publish successful local production into the token's device slot.
  void markLocalReady(const AcceleratorCollectiveToken& token);

  /// Begin and complete the host-staged device-to-host copy.
  void beginDeviceToHost(const AcceleratorCollectiveToken& token);
  void completeDeviceToHost(const AcceleratorCollectiveToken& token);

  /// Begin and complete the canonical MPI collective for this chunk.
  void beginCollective(const AcceleratorCollectiveToken& token);
  void completeCollective(const AcceleratorCollectiveToken& token);

  /// Begin and complete the host-staged host-to-device result copy.
  void beginHostToDevice(const AcceleratorCollectiveToken& token);
  void completeHostToDevice(const AcceleratorCollectiveToken& token);

  /// Release one completed result in canonical order for local consumption/reuse.
  void releaseResult(const AcceleratorCollectiveToken& token);

  /// Finish only after every planned chunk has been released successfully.
  void finishIteration();

  /// Poison the complete active iteration after a classified external failure.
  void failIteration(AcceleratorCollectiveFailure failure);

  /// Cancel an active iteration and require the same explicit drain as a failure.
  void cancelIteration();

  /// Reset poisoned slots after the runtime owner has drained external work.
  void recoverAfterExternalDrain();

  /// Return the immutable prepared plan.
  const AcceleratorDistributedPlan& plan() const noexcept { return plan_; }

  /// Return the current iteration and failure states.
  AcceleratorCollectiveIterationState iterationState() const noexcept { return iteration_state_; }
  AcceleratorCollectiveFailure failure() const noexcept { return failure_; }

  /// Return one slot's ownership state for diagnostics and transport assertions.
  AcceleratorCollectiveSlotState slotState(std::size_t slot) const;

  /// Return the current token generation.
  std::size_t generation() const noexcept { return generation_; }

  /// Return canonical acquisition, collective, and release positions.
  std::size_t acquiredChunkCount() const noexcept { return next_acquire_; }
  std::size_t reducedChunkCount() const noexcept { return next_collective_; }
  std::size_t releasedChunkCount() const noexcept { return next_release_; }

private:
  /// Retain the current token and ownership state of one prepared slot.
  struct Slot
  {
    AcceleratorCollectiveSlotState state = AcceleratorCollectiveSlotState::AVAILABLE;
    AcceleratorCollectiveToken token;
  };

  /// Validate a token and return its exact live slot.
  Slot& requireToken(const AcceleratorCollectiveToken& token,
                     AcceleratorCollectiveSlotState expected,
                     const char* operation);

  /// Reject operations unless one non-poisoned iteration is active.
  void requireActive(const char* operation) const;

  AcceleratorDistributedPlan plan_;
  std::array<Slot, 2> slots_;
  AcceleratorCollectiveIterationState iteration_state_ = AcceleratorCollectiveIterationState::IDLE;
  AcceleratorCollectiveFailure failure_ = AcceleratorCollectiveFailure::NONE;
  std::size_t generation_      = 0;
  std::size_t next_acquire_    = 0;
  std::size_t next_collective_ = 0;
  std::size_t next_release_    = 0;
};

} // namespace qmcplusplus::wftrain

#endif // QMCPLUSPLUS_ACCELERATOR_COLLECTIVE_STATE_H
