//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_AcceleratorCollectiveRecording.cpp
 * @brief Offline tests for accelerator collective buffers and event ordering.
 */

#include "QMCDrivers/WFTrain/AcceleratorCollectiveRecording.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Build a two-rank plan whose final chunk is shorter than its prepared buffers.
AcceleratorDistributedPlan makeRecordingPlan(AcceleratorCollectiveTransport transport,
                                             std::size_t buffer_slots = 2)
{
  std::vector<AcceleratorRankPlanInput> ranks{
      {0, 2, 10, 0, 0, std::size_t{4096}, 1024, 2, 7, 101, 202, true},
      {1, 2, 10, 1, 5, std::size_t{4096}, 1024, 3, 7, 101, 202, true}};
  return makeAcceleratorDistributedPlan(
      ranks, 5, 16,
      transport == AcceleratorCollectiveTransport::HOST_STAGED_MPI
          ? AcceleratorTransportRequest::HOST_STAGED_MPI
          : AcceleratorTransportRequest::DEVICE_MPI,
      buffer_slots);
}

/// Build distinct prepared device and pinned-host identities for two slots.
std::array<AcceleratorCollectivePreparedSlot, 2> makeRecordingSlots()
{
  return {{{{11, AcceleratorCollectiveMemoryDomain::DEVICE, 32},
            {21, AcceleratorCollectiveMemoryDomain::PINNED_HOST, 32}},
           {{12, AcceleratorCollectiveMemoryDomain::DEVICE, 32},
            {22, AcceleratorCollectiveMemoryDomain::PINNED_HOST, 32}}}};
}

/// Record one complete host-staged transfer and result release.
void recordHostChunk(AcceleratorCollectiveRecordingTransport& transport,
                     const AcceleratorCollectiveToken& token)
{
  transport.completeProduction(token);
  transport.submitDeviceToHost(token);
  transport.completeDeviceToHost(token);
  transport.submitCollective(token);
  transport.completeCollective(token);
  transport.submitHostToDevice(token);
  transport.completeHostToDevice(token);
  transport.releaseResult(token);
}

} // namespace

TEST_CASE("Recording transport captures bounded host-staged ordering",
          "[drivers][wftrain][accelerator][recording_transport]")
{
  AcceleratorCollectiveRecordingTransport transport(
      makeRecordingPlan(AcceleratorCollectiveTransport::HOST_STAGED_MPI), makeRecordingSlots());
  const auto* prepared_trace_storage = transport.trace().data();
  transport.beginIteration(transport.state().plan().fingerprint, 7);

  const AcceleratorCollectiveToken first = transport.acquireProductionSlot();
  recordHostChunk(transport, first);
  const AcceleratorCollectiveToken second = transport.acquireProductionSlot();
  recordHostChunk(transport, second);
  const AcceleratorCollectiveToken tail = transport.acquireProductionSlot();
  recordHostChunk(transport, tail);
  transport.finishIteration();

  REQUIRE(transport.trace().size() == 27);
  CHECK(transport.trace().data() == prepared_trace_storage);
  CHECK(transport.trace()[0].operation == AcceleratorCollectiveRecordedOperation::PRODUCTION_ACQUIRE);
  CHECK(transport.trace()[2].operation == AcceleratorCollectiveRecordedOperation::DEVICE_TO_HOST_SUBMIT);
  CHECK(transport.trace()[4].operation == AcceleratorCollectiveRecordedOperation::HOST_MPI_SUBMIT);
  CHECK(transport.trace()[6].operation == AcceleratorCollectiveRecordedOperation::HOST_TO_DEVICE_SUBMIT);
  CHECK(transport.trace()[8].operation == AcceleratorCollectiveRecordedOperation::RESULT_RELEASE);
  CHECK(transport.trace().back().byte_count == 16);
  CHECK(transport.trace().back().source_identity == 11);
  CHECK(transport.state().iterationState() == AcceleratorCollectiveIterationState::IDLE);
}

TEST_CASE("Recording transport distinguishes direct device MPI and overlap",
          "[drivers][wftrain][accelerator][recording_transport]")
{
  AcceleratorCollectiveRecordingTransport transport(
      makeRecordingPlan(AcceleratorCollectiveTransport::DEVICE_MPI), makeRecordingSlots());
  transport.beginIteration(transport.state().plan().fingerprint, 7);
  const AcceleratorCollectiveToken first = transport.acquireProductionSlot();
  const AcceleratorCollectiveToken second = transport.acquireProductionSlot();
  transport.completeProduction(second);
  CHECK_THROWS_WITH(transport.submitCollective(second),
                    Catch::Matchers::ContainsSubstring("canonical order"));

  transport.completeProduction(first);
  transport.submitCollective(first);
  transport.completeCollective(first);
  transport.releaseResult(first);
  transport.submitCollective(second);
  transport.completeCollective(second);
  transport.releaseResult(second);

  const AcceleratorCollectiveToken tail = transport.acquireProductionSlot();
  transport.completeProduction(tail);
  transport.submitCollective(tail);
  transport.completeCollective(tail);
  transport.releaseResult(tail);
  transport.finishIteration();

  REQUIRE(transport.trace().size() == 15);
  CHECK(transport.trace()[4].operation == AcceleratorCollectiveRecordedOperation::DEVICE_MPI_SUBMIT);
  CHECK(transport.trace()[4].source_identity == 11);
  CHECK(transport.trace()[4].destination_identity == 11);
}

TEST_CASE("Recording transport rejects invalid prepared memory",
          "[drivers][wftrain][accelerator][recording_transport]")
{
  auto slots = makeRecordingSlots();
  slots[0].host.domain = AcceleratorCollectiveMemoryDomain::DEVICE;
  CHECK_THROWS_WITH(AcceleratorCollectiveRecordingTransport(
                        makeRecordingPlan(AcceleratorCollectiveTransport::HOST_STAGED_MPI), slots),
                    Catch::Matchers::ContainsSubstring("pinned"));

  slots = makeRecordingSlots();
  slots[1].device.identity = slots[0].host.identity;
  CHECK_THROWS_WITH(AcceleratorCollectiveRecordingTransport(
                        makeRecordingPlan(AcceleratorCollectiveTransport::HOST_STAGED_MPI), slots),
                    Catch::Matchers::ContainsSubstring("must not alias"));

  slots = makeRecordingSlots();
  slots[1].device.capacity_bytes = 31;
  CHECK_THROWS_WITH(AcceleratorCollectiveRecordingTransport(
                        makeRecordingPlan(AcceleratorCollectiveTransport::DEVICE_MPI), slots),
                    Catch::Matchers::ContainsSubstring("adequate device"));
}

TEST_CASE("Recording transport fails before mutation and records drained recovery",
          "[drivers][wftrain][accelerator][recording_transport]")
{
  AcceleratorCollectiveRecordingTransport transport(
      makeRecordingPlan(AcceleratorCollectiveTransport::HOST_STAGED_MPI, 1), makeRecordingSlots(), 3);
  transport.beginIteration(transport.state().plan().fingerprint, 7);
  const AcceleratorCollectiveToken token = transport.acquireProductionSlot();
  transport.completeProduction(token);
  transport.submitDeviceToHost(token);

  CHECK_THROWS_WITH(transport.completeDeviceToHost(token),
                    Catch::Matchers::ContainsSubstring("trace is full"));
  CHECK(transport.state().slotState(0) == AcceleratorCollectiveSlotState::DEVICE_TO_HOST);

  // Use a fresh, adequately sized recorder to exercise poison/drain/retry.
  AcceleratorCollectiveRecordingTransport retry(
      makeRecordingPlan(AcceleratorCollectiveTransport::HOST_STAGED_MPI, 1), makeRecordingSlots());
  retry.beginIteration(retry.state().plan().fingerprint, 7);
  const AcceleratorCollectiveToken stale = retry.acquireProductionSlot();
  retry.completeProduction(stale);
  retry.failIteration(AcceleratorCollectiveFailure::DEVICE_COPY);
  CHECK(retry.trace().back().operation == AcceleratorCollectiveRecordedOperation::FAILURE);
  CHECK_THROWS(retry.submitDeviceToHost(stale));
  retry.recoverAfterExternalDrain();
  CHECK(retry.trace().back().operation == AcceleratorCollectiveRecordedOperation::EXTERNAL_DRAIN);
  CHECK(retry.trace().back().failure == AcceleratorCollectiveFailure::DEVICE_COPY);

  retry.clearTrace();
  CHECK(retry.trace().empty());
  retry.beginIteration(retry.state().plan().fingerprint, 7);
  CHECK_THROWS_WITH(retry.completeProduction(stale),
                    Catch::Matchers::ContainsSubstring("live iteration"));
  retry.failIteration(AcceleratorCollectiveFailure::CANCELLED);
  retry.recoverAfterExternalDrain();
}

} // namespace qmcplusplus::wftrain
