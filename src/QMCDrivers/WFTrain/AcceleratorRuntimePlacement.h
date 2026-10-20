//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file AcceleratorRuntimePlacement.h
 * @brief Fixed-record physical-device consensus before distributed training.
 */

#ifndef QMCPLUSPLUS_ACCELERATOR_RUNTIME_PLACEMENT_H
#define QMCPLUSPLUS_ACCELERATOR_RUNTIME_PLACEMENT_H

#include <array>
#include <cstdint>
#include <vector>

class Communicate;

namespace qmcplusplus::wftrain
{

/// Describe actual execution and physical placement, independently of visible ordinals.
struct AcceleratorRuntimePlacement
{
  bool accelerator = false;
  bool probe_succeeded = true;
  std::array<char, 256> hostname{};
  std::array<char, 32> pci_bus_id{};
};

/** Reject failed probes, mixed host/device routes and duplicate physical GPUs.
 * Pure validation permits deterministic tests without MPI or accelerator runtime.
 */
void validateAcceleratorRuntimePlacements(const std::vector<AcceleratorRuntimePlacement>& ranks);

/** Check exclusive native GPU consumers across calculation groups, ignoring inactive ranks.
 * Inactive records do not identify legacy GPU consumers; mixed legacy GPU use
 * is outside this native-only check.
 */
void validateNativeAcceleratorRuntimePlacements(const std::vector<AcceleratorRuntimePlacement>& ranks);

/** Probe the selected runtime device and agree on placement before throwing.
 * Every rank must call once, including host providers; no scheduling is performed.
 */
void agreeAcceleratorRuntimePlacement(Communicate& communicator, bool accelerator);

/// One common startup census, with actual native GPU participation on every WORLD rank.
void agreeNativeAcceleratorRuntimePlacement(Communicate& communicator, bool native_accelerator);

} // namespace qmcplusplus::wftrain

#endif
