//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file AcceleratorCollectiveDeviceCopy.h
 * @brief CUDA/HIP copy-queue and pinned-slot attachment for host-staged MPI.
 *
 * The adapter is available only in CUDA-family builds.  HIP uses the same QMCPACK
 * queue interface through the existing CUDA-to-HIP portability layer.  Compilation
 * does not establish runtime device or GPU-aware-MPI capability.
 */

#ifndef QMCPLUSPLUS_ACCELERATOR_COLLECTIVE_DEVICE_COPY_H
#define QMCPLUSPLUS_ACCELERATOR_COLLECTIVE_DEVICE_COPY_H

#include "config.h"
#include "QMCDrivers/WFTrain/HostStagedAcceleratorReduction.h"

#if defined(ENABLE_CUDA)

#include "Platforms/PinnedAllocator.h"
#include "Platforms/QueueAliases.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace qmcplusplus::wftrain
{

/** Own one or two preparation-time page-locked host staging vectors.
 *
 * Storage remains fixed after construction, and the returned descriptors stay valid
 * for the owner's lifetime.  Copy and move are disabled to preserve those addresses.
 */
class AcceleratorPinnedHostSlots
{
public:
  /// Allocate exactly the active slot count and maximum planned chunk extent.
  explicit AcceleratorPinnedHostSlots(const AcceleratorDistributedPlan& plan);

  AcceleratorPinnedHostSlots(const AcceleratorPinnedHostSlots&) = delete;
  AcceleratorPinnedHostSlots& operator=(const AcceleratorPinnedHostSlots&) = delete;
  AcceleratorPinnedHostSlots(AcceleratorPinnedHostSlots&&) = delete;
  AcceleratorPinnedHostSlots& operator=(AcceleratorPinnedHostSlots&&) = delete;

  /// Return nonowning descriptors consumed by the host-staged coordinator.
  std::array<AcceleratorHostStagingSlot, 2> descriptors() noexcept;

private:
  using PinnedVector = std::vector<DerivativeValue, PinnedAllocator<DerivativeValue>>;
  std::array<PinnedVector, 2> storage_;
  std::size_t active_slots_ = 0;
};

/** Adapt the common CUDA/HIP compute queue to the host-staged copy protocol.
 *
 * The first implementation is deliberately serialized: wait operations synchronize
 * the owned queue.  Event-based overlap can replace this implementation without
 * changing the coordinator or MPI schedule after runtime parity is established.
 */
class CudaHipAcceleratorCollectiveCopyQueue final : public AcceleratorCollectiveCopyQueue
{
public:
  /// Borrow one externally owned queue for the complete coordinator lifetime.
  explicit CudaHipAcceleratorCollectiveCopyQueue(compute::Queue<PlatformKind::CUDA>& queue)
      : queue_(queue)
  {}

  void submitDeviceToHost(std::size_t slot,
                          DerivativeValue* host_destination,
                          const DerivativeValue* device_source,
                          std::size_t count) override;
  void waitDeviceToHost(std::size_t slot) override;

  void submitHostToDevice(std::size_t slot,
                          DerivativeValue* device_destination,
                          const DerivativeValue* host_source,
                          std::size_t count) override;
  void waitHostToDevice(std::size_t slot) override;

  bool drainAfterFailure() noexcept override;

private:
  /// Reject corrupt slot identities before submitting a runtime operation.
  static void validateSlot(std::size_t slot);

  compute::Queue<PlatformKind::CUDA>& queue_;
};

} // namespace qmcplusplus::wftrain

#endif // ENABLE_CUDA

#endif // QMCPLUSPLUS_ACCELERATOR_COLLECTIVE_DEVICE_COPY_H
