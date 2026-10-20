//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_AcceleratorRuntimePlacement.cpp
 * @brief Device-mask-independent placement admission without runtime probes.
 */

#include "QMCDrivers/WFTrain/AcceleratorRuntimePlacement.h"
#include "Message/Communicate.h"
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <string>

namespace qmcplusplus::wftrain
{
namespace
{

/// Build zero-padded physical identifiers just as the startup runtime probe does.
AcceleratorRuntimePlacement devicePlacement(const std::string& host, const std::string& bus)
{
  AcceleratorRuntimePlacement result;
  result.accelerator = true;
  std::copy(host.begin(), host.end(), result.hostname.begin());
  std::copy(bus.begin(), bus.end(), result.pci_bus_id.begin());
  return result;
}

} // namespace

TEST_CASE("Job-wide native placement permits CPU groups but rejects cross-group collisions",
          "[drivers][wftrain][accelerator][placement][native_job]")
{
  AcceleratorRuntimePlacement inactive;
  const auto first = devicePlacement("node-a", "0000:01:00.0");
  const auto second = devicePlacement("node-a", "0000:02:00.0");
  CHECK_NOTHROW(validateNativeAcceleratorRuntimePlacements({inactive, first, inactive, second}));
  CHECK_NOTHROW(validateNativeAcceleratorRuntimePlacements({inactive, inactive}));
  CHECK_THROWS(validateNativeAcceleratorRuntimePlacements({}));
  CHECK_THROWS(validateNativeAcceleratorRuntimePlacements({first, inactive, first}));
  auto failed = first;
  failed.probe_succeeded = false;
  CHECK_THROWS(validateNativeAcceleratorRuntimePlacements({inactive, failed}));
  failed.accelerator = false; // Inactive ranks do not probe or reserve a GPU.
  CHECK_NOTHROW(validateNativeAcceleratorRuntimePlacements({first, failed}));
  auto malformed = first;
  malformed.pci_bus_id.back() = 'x';
  CHECK_THROWS(validateNativeAcceleratorRuntimePlacements({inactive, malformed}));
  CHECK_NOTHROW(validateNativeAcceleratorRuntimePlacements({first, devicePlacement("node-b", "0000:01:00.0")}));
}

TEST_CASE("Native job census gathers inactive CPU ranks without a device probe",
          "[drivers][wftrain][accelerator][placement][native_job]")
{
  CHECK_NOTHROW(agreeNativeAcceleratorRuntimePlacement(*OHMMS::Controller, false));
}

TEST_CASE("Accelerator physical placement is independent of visible ordinals",
          "[drivers][wftrain][accelerator][placement]")
{
  const auto first = devicePlacement("node-a", "0000:01:00.0");
  const auto second = devicePlacement("node-a", "0000:02:00.0");
  const auto other_node = devicePlacement("node-b", "0000:01:00.0");
  CHECK_NOTHROW(validateAcceleratorRuntimePlacements({first}));
  CHECK_NOTHROW(validateAcceleratorRuntimePlacements({first, second}));
  CHECK_NOTHROW(validateAcceleratorRuntimePlacements({first, other_node}));
  CHECK_THROWS(validateAcceleratorRuntimePlacements({first, first}));
  CHECK_THROWS(validateAcceleratorRuntimePlacements({first, second, first}));
}

TEST_CASE("Accelerator placement rejects mixed failed or malformed rank records",
          "[drivers][wftrain][accelerator][placement]")
{
  AcceleratorRuntimePlacement host;
  auto device = devicePlacement("node-a", "0000:01:00.0");
  CHECK_NOTHROW(validateAcceleratorRuntimePlacements({host, host}));
  CHECK_THROWS(validateAcceleratorRuntimePlacements({}));
  CHECK_THROWS(validateAcceleratorRuntimePlacements({host, device}));
  CHECK_THROWS(validateAcceleratorRuntimePlacements({device, host}));
  device.probe_succeeded = false;
  CHECK_THROWS(validateAcceleratorRuntimePlacements({device}));
  host.probe_succeeded = false;
  CHECK_THROWS(validateAcceleratorRuntimePlacements({host}));
  device.probe_succeeded = true;
  device.hostname.back() = 'x';
  CHECK_THROWS(validateAcceleratorRuntimePlacements({device}));
  device.hostname = {};
  CHECK_THROWS(validateAcceleratorRuntimePlacements({device}));
}

} // namespace qmcplusplus::wftrain
