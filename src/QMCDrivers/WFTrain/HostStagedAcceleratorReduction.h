//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file HostStagedAcceleratorReduction.h
 * @brief Serialized host-staged MPI reduction for accelerator-resident vectors.
 *
 * The coordinator owns no device or pinned allocation.  A caller supplies prepared
 * staging slots and a backend copy queue, allowing the fixed MPI and failure protocol
 * to be tested with host memory before CUDA/HIP runtime attachment.
 */

#ifndef QMCPLUSPLUS_HOST_STAGED_ACCELERATOR_REDUCTION_H
#define QMCPLUSPLUS_HOST_STAGED_ACCELERATOR_REDUCTION_H

#include "QMCDrivers/WFTrain/AcceleratorCollectiveState.h"
#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

class Communicate;

namespace qmcplusplus::wftrain
{

/** Abstract asynchronous-copy queue used by the portable host-staged protocol.
 *
 * Implementations submit and wait on one exact slot.  ``drainAfterFailure`` must
 * not throw and must leave every submitted copy complete or cancelled before the
 * coordinator resets poisoned ownership.
 */
class AcceleratorCollectiveCopyQueue
{
public:
  virtual ~AcceleratorCollectiveCopyQueue() = default;

  /// Submit and wait for a device-to-host copy of logical complex128 elements.
  virtual void submitDeviceToHost(std::size_t slot,
                                  DerivativeValue* host_destination,
                                  const DerivativeValue* device_source,
                                  std::size_t count) = 0;
  virtual void waitDeviceToHost(std::size_t slot) = 0;

  /// Submit and wait for a host-to-device copy of reduced complex128 elements.
  virtual void submitHostToDevice(std::size_t slot,
                                  DerivativeValue* device_destination,
                                  const DerivativeValue* host_source,
                                  std::size_t count) = 0;
  virtual void waitHostToDevice(std::size_t slot) = 0;

  /// Drain/cancel submitted work and report whether buffer reuse is confirmed.
  virtual bool drainAfterFailure() noexcept = 0;
};

/// Describe one caller-owned pinned-host complex128 staging interval.
struct AcceleratorHostStagingSlot
{
  DerivativeValue* values = nullptr;
  std::size_t capacity = 0;
};

/** Reduce one accelerator-resident complex parameter vector through host MPI.
 *
 * The object is noncopyable and externally serialized.  Constructor-time control
 * storage is retained and reused; a successful reduction performs no allocation in
 * this layer.  Raw sums are returned to device storage; objective normalization is
 * deliberately a higher-level responsibility.
 */
class HostStagedAcceleratorReduction
{
public:
  /// Validate the unanimous plan and prepared staging slots.
  HostStagedAcceleratorReduction(Communicate& communicator,
                                 AcceleratorDistributedPlan plan,
                                 AcceleratorCollectiveCopyQueue& copy_queue,
                                 std::array<AcceleratorHostStagingSlot, 2> staging_slots);

  HostStagedAcceleratorReduction(const HostStagedAcceleratorReduction&) = delete;
  HostStagedAcceleratorReduction& operator=(const HostStagedAcceleratorReduction&) = delete;
  HostStagedAcceleratorReduction(HostStagedAcceleratorReduction&&) = delete;
  HostStagedAcceleratorReduction& operator=(HostStagedAcceleratorReduction&&) = delete;

  /** Replace one rank-local device vector with its replicated global raw sum.
   *
   * After any exception the vector is invalid because a prefix may already contain
   * global sums; callers must regenerate the complete rank-local input before retry.
   */
  void reduceComplexVector(DerivativeValue* device_values,
                           std::size_t count,
                           std::uint64_t plan_fingerprint,
                           std::size_t parameter_version);

  /// Return immutable plan/state evidence for diagnostics.
  const AcceleratorCollectiveState& state() const noexcept { return state_; }

private:
  /// Fixed copy/finite-check consensus phases shared by every rank.
  enum class ConsensusPhase : std::uint64_t
  {
    DEVICE_TO_HOST = 1,
    REDUCED_HOST = 2,
    HOST_TO_DEVICE = 3
  };

  /// Return the lowest failed rank, or participant count when all ranks succeeded.
  std::size_t consensus(const AcceleratorCollectiveToken& token,
                        ConsensusPhase phase,
                        bool local_success);

  /// Poison ownership, drain copies, reset storage, and throw a uniform error.
  [[noreturn]] void failAndRecover(AcceleratorCollectiveFailure failure,
                                   std::size_t failing_rank,
                                   const char* stage);

  Communicate& communicator_;
  AcceleratorCollectiveCopyQueue& copy_queue_;
  AcceleratorCollectiveState state_;
  std::array<AcceleratorHostStagingSlot, 2> staging_slots_;
  std::array<std::uint64_t, 8> local_record_{};
  std::vector<std::uint64_t> gathered_records_;
};

} // namespace qmcplusplus::wftrain

#endif // QMCPLUSPLUS_HOST_STAGED_ACCELERATOR_REDUCTION_H
