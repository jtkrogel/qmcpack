//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file AcceleratorRuntimePlacement.cpp
 * @brief Physical CUDA/HIP placement admission with fixed rank consensus.
 */

#include "QMCDrivers/WFTrain/AcceleratorRuntimePlacement.h"
#include "Message/Communicate.h"
#include "Message/CommOperators.h"
#include "config.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>

#if defined(ENABLE_CUDA)
#include "Platforms/CUDA/CUDAruntime.hpp"
#endif

namespace qmcplusplus::wftrain
{

/// Compare complete physical identifiers rather than process-local device numbers.
void validateAcceleratorRuntimePlacements(const std::vector<AcceleratorRuntimePlacement>& ranks)
{
  if (ranks.empty())
    throw std::invalid_argument("Accelerator placement requires at least one participant");
  for (std::size_t rank = 0; rank < ranks.size(); ++rank)
  {
    const auto& current = ranks[rank];
    if (!current.probe_succeeded)
      throw std::runtime_error("Accelerator placement probe failed on rank " + std::to_string(rank));
    if (current.accelerator != ranks.front().accelerator)
      throw std::runtime_error("Distributed training requires the same host/device route on every rank");
    if (!current.accelerator)
      continue;
    if (current.hostname.front() == '\0' || current.pci_bus_id.front() == '\0' ||
        current.hostname.back() != '\0' || current.pci_bus_id.back() != '\0')
      throw std::invalid_argument("Accelerator placement contains an invalid physical identifier");
    for (std::size_t earlier = 0; earlier < rank; ++earlier)
      if (std::strcmp(current.hostname.data(), ranks[earlier].hostname.data()) == 0 &&
          std::strcmp(current.pci_bus_id.data(), ranks[earlier].pci_bus_id.data()) == 0)
        throw std::runtime_error("Distributed GPU training requires one MPI rank per physical GPU; ranks " +
                                 std::to_string(earlier) + " and " + std::to_string(rank) + " share a device");
  }
}

/// Gather probe status even after a local runtime failure, avoiding stranded peers.
void agreeAcceleratorRuntimePlacement(Communicate& communicator, bool accelerator)
{
  // Explicit fields avoid transmitting struct padding or assuming a vendor UUID ABI.
  constexpr std::size_t hostname_words = 256 / sizeof(std::uint64_t);
  constexpr std::size_t bus_words = 32 / sizeof(std::uint64_t);
  constexpr std::size_t record_words = 2 + hostname_words + bus_words;
  std::array<std::uint64_t, record_words> local{};
  local[0] = accelerator ? 1 : 0;
  local[1] = 1;
  AcceleratorRuntimePlacement placement;
  placement.accelerator = accelerator;
  if (accelerator)
  {
    bool success = gethostname(placement.hostname.data(), placement.hostname.size()) == 0 &&
        placement.hostname.back() == '\0';
#if defined(ENABLE_CUDA)
    int device = -1;
#if defined(QMC_CUDA2HIP)
    success = success && hipGetDevice(&device) == hipSuccess &&
        hipDeviceGetPCIBusId(placement.pci_bus_id.data(), placement.pci_bus_id.size(), device) == hipSuccess;
#else
    success = success && cudaGetDevice(&device) == cudaSuccess &&
        cudaDeviceGetPCIBusId(placement.pci_bus_id.data(), placement.pci_bus_id.size(), device) == cudaSuccess;
#endif
#else
    success = false;
#endif
    local[1] = success ? 1 : 0;
  }
  std::memcpy(local.data() + 2, placement.hostname.data(), placement.hostname.size());
  std::memcpy(local.data() + 2 + hostname_words, placement.pci_bus_id.data(), placement.pci_bus_id.size());

  // This is startup-only O(ranks) storage, independent of walker and parameter counts.
  std::vector<std::uint64_t> gathered(communicator.size() * record_words);
  communicator.allgather(local.data(), gathered.data(), static_cast<int>(record_words));
  std::vector<AcceleratorRuntimePlacement> ranks(communicator.size());
  for (std::size_t rank = 0; rank < ranks.size(); ++rank)
  {
    const auto* record = gathered.data() + rank * record_words;
    ranks[rank].accelerator = record[0] != 0;
    ranks[rank].probe_succeeded = record[1] != 0;
    std::memcpy(ranks[rank].hostname.data(), record + 2, placement.hostname.size());
    std::memcpy(ranks[rank].pci_bus_id.data(), record + 2 + hostname_words, placement.pci_bus_id.size());
  }
  validateAcceleratorRuntimePlacements(ranks);
}

} // namespace qmcplusplus::wftrain
