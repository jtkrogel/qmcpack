//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file accelerator_collective_copy_link_check.cpp
 * @brief Compile/link-only CUDA/HIP host-staging adapter check.
 *
 * This executable must not be run on the offline development host.  Taking each
 * member address forces definitions out of the static driver archive without
 * constructing a queue or invoking an accelerator runtime.
 */

#include "QMCDrivers/WFTrain/AcceleratorCollectiveDeviceCopy.h"

#if !defined(ENABLE_CUDA)
#error "The accelerator collective copy link check requires a CUDA/HIP build"
#endif

int main()
{
  using qmcplusplus::wftrain::CudaHipAcceleratorCollectiveCopyQueue;
  const auto d2h = &CudaHipAcceleratorCollectiveCopyQueue::submitDeviceToHost;
  const auto d2h_wait = &CudaHipAcceleratorCollectiveCopyQueue::waitDeviceToHost;
  const auto h2d = &CudaHipAcceleratorCollectiveCopyQueue::submitHostToDevice;
  const auto h2d_wait = &CudaHipAcceleratorCollectiveCopyQueue::waitHostToDevice;
  const auto drain = &CudaHipAcceleratorCollectiveCopyQueue::drainAfterFailure;
  return d2h == nullptr || d2h_wait == nullptr || h2d == nullptr ||
      h2d_wait == nullptr || drain == nullptr;
}
