//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file AcceleratorCollectiveDeviceCopy.cpp
 * @brief CUDA/HIP queue implementation of host-staged collective copies.
 */

#include "QMCDrivers/WFTrain/AcceleratorCollectiveDeviceCopy.h"

#if defined(ENABLE_CUDA)

#include <stdexcept>

namespace qmcplusplus::wftrain
{

AcceleratorPinnedHostSlots::AcceleratorPinnedHostSlots(
    const AcceleratorDistributedPlan& plan)
    : active_slots_(plan.buffer_slots)
{
  if (plan.transport != AcceleratorCollectiveTransport::HOST_STAGED_MPI ||
      active_slots_ == 0 || active_slots_ > storage_.size() || plan.chunk_elements == 0)
    throw std::invalid_argument("Pinned collective slots require a valid host-staged plan");
  for (std::size_t slot = 0; slot < active_slots_; ++slot)
    storage_[slot].resize(plan.chunk_elements);
}

std::array<AcceleratorHostStagingSlot, 2> AcceleratorPinnedHostSlots::descriptors() noexcept
{
  std::array<AcceleratorHostStagingSlot, 2> result{};
  for (std::size_t slot = 0; slot < active_slots_; ++slot)
    result[slot] = {storage_[slot].data(), storage_[slot].size()};
  return result;
}

void CudaHipAcceleratorCollectiveCopyQueue::validateSlot(std::size_t slot)
{
  if (slot >= 2)
    throw std::out_of_range("Accelerator collective copy slot is outside the prepared domain");
}

void CudaHipAcceleratorCollectiveCopyQueue::submitDeviceToHost(
    std::size_t slot,
    DerivativeValue* host_destination,
    const DerivativeValue* device_source,
    std::size_t count)
{
  validateSlot(slot);
  if (host_destination == nullptr || device_source == nullptr || count == 0)
    throw std::invalid_argument("Accelerator D2H copy requires nonempty source and destination");
  queue_.memcpy(host_destination, device_source, count);
}

void CudaHipAcceleratorCollectiveCopyQueue::waitDeviceToHost(std::size_t slot)
{
  validateSlot(slot);
  queue_.sync();
}

void CudaHipAcceleratorCollectiveCopyQueue::submitHostToDevice(
    std::size_t slot,
    DerivativeValue* device_destination,
    const DerivativeValue* host_source,
    std::size_t count)
{
  validateSlot(slot);
  if (device_destination == nullptr || host_source == nullptr || count == 0)
    throw std::invalid_argument("Accelerator H2D copy requires nonempty source and destination");
  queue_.memcpy(device_destination, host_source, count);
}

void CudaHipAcceleratorCollectiveCopyQueue::waitHostToDevice(std::size_t slot)
{
  validateSlot(slot);
  queue_.sync();
}

bool CudaHipAcceleratorCollectiveCopyQueue::drainAfterFailure() noexcept
{
  try
  {
    queue_.sync();
    return true;
  }
  catch (...)
  {
    return false;
  }
}

} // namespace qmcplusplus::wftrain

#endif // ENABLE_CUDA
