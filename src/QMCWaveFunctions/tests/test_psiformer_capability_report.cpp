//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
// CPU-only executable: exercises the actual receipt writer without device/runtime mocks.
#include "QMCWaveFunctions/PsiFormer/PsiFormerCapabilityReport.h"
#include <iostream>
#include <thread>
#include <vector>

using namespace qmcplusplus::psiformer;

void require(bool condition)
{
  if (!condition) throw std::runtime_error("PsiFormer capability fixture assertion failed");
}

int main()
{
  require(capabilityFailure(std::invalid_argument("malformed XML")) == CapabilityFailure::INVALID_INPUT);
  require(capabilityFailure(std::length_error("numeric cap")) == CapabilityFailure::INSUFFICIENT_RESOURCES);
  require(capabilityFailure(std::bad_alloc()) == CapabilityFailure::INSUFFICIENT_RESOURCES);
  require(capabilityFailure(std::runtime_error("unrelated runtime error")) == CapabilityFailure::UNCLASSIFIED);
  for (auto failure : {CapabilityFailure::UNSUPPORTED_OPERATION, CapabilityFailure::UNQUALIFIED_HARDWARE})
  {
    const CapabilityError error(failure, "preserved strict diagnostic");
    require(std::string(error.what()) == "preserved strict diagnostic");
    require(capabilityFailure(error) == failure);
  }
  writeCapabilityReceipt(std::cout, "selection", {
      {"requested_backend", "auto"}, {"selected_backend", "cpu"},
      {"fallback_reason", "backend not compiled"}, {"fallback_count", "1"},
      {"active_crowds", "not_started"}, {"nominal_crowds", "driver_deferred"},
      {"provenance", "cpu_fixture_actual_receipt_writer_no_gpu_qualification"},
      {"escaped", "quote\" backslash\\ newline\n tab\t"}});
  CapabilityDispatchCounters counters;
  {
    CompletedCapabilityDispatch scalar(counters, 1, false);
  }
  try
  {
    CompletedCapabilityDispatch rejected(counters, 8, true);
    throw std::invalid_argument("rejected full VGL");
  }
  catch (const std::invalid_argument&) {}
  std::vector<std::thread> workers;
  for (std::size_t i = 1; i <= 4; ++i)
    workers.emplace_back([&, i]() {
      for (int j = 0; j < 100; ++j) { CompletedCapabilityDispatch completed(counters, i, true); }
    });
  for (auto& worker : workers) worker.join();
  require(counters.scalar_full == 1 && counters.crowd_full == 400 && counters.maximum_crowd_walkers == 4);
  counters.report(std::cout, 42);
}
