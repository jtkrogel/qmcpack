//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_AcceleratorCollectiveState.cpp
 * @brief Deterministic tests for bounded accelerator collective ownership.
 */

#include "QMCDrivers/WFTrain/AcceleratorCollectiveState.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Build a small valid plan with a partial final chunk and optional staging.
AcceleratorDistributedPlan makePlan(AcceleratorCollectiveTransport transport,
                                    std::size_t slots,
                                    std::size_t local_samples = 3)
{
  std::vector<AcceleratorRankPlanInput> ranks{
      {/*rank=*/0,
       /*participant_count=*/2,
       /*node_id=*/10,
       /*device_id=*/0,
       /*local_sample_count=*/local_samples,
       /*device_budget_bytes=*/std::size_t{4096},
       /*base_required_device_bytes=*/1024,
       /*maximum_chunk_elements=*/2,
       /*parameter_version=*/7,
       /*model_fingerprint=*/101,
       /*precision_fingerprint=*/202,
       /*device_mpi_available=*/true},
      {/*rank=*/1,
       /*participant_count=*/2,
       /*node_id=*/10,
       /*device_id=*/1,
       /*local_sample_count=*/5,
       /*device_budget_bytes=*/std::size_t{4096},
       /*base_required_device_bytes=*/1024,
       /*maximum_chunk_elements=*/3,
       /*parameter_version=*/7,
       /*model_fingerprint=*/101,
       /*precision_fingerprint=*/202,
       /*device_mpi_available=*/true}};
  return makeAcceleratorDistributedPlan(
      ranks, /*parameter_count=*/5, /*collective_element_bytes=*/16,
      transport == AcceleratorCollectiveTransport::HOST_STAGED_MPI
          ? AcceleratorTransportRequest::HOST_STAGED_MPI
          : AcceleratorTransportRequest::DEVICE_MPI,
      slots);
}

/// Drive one host-staged token through copies, collective, and release.
void completeHostStaged(AcceleratorCollectiveState& state,
                        const AcceleratorCollectiveToken& token)
{
  state.markLocalReady(token);
  state.beginDeviceToHost(token);
  state.completeDeviceToHost(token);
  state.beginCollective(token);
  state.completeCollective(token);
  state.beginHostToDevice(token);
  state.completeHostToDevice(token);
  state.releaseResult(token);
}

/// Drive one direct-device token through its shorter transition graph.
void completeDeviceMPI(AcceleratorCollectiveState& state,
                       const AcceleratorCollectiveToken& token)
{
  state.markLocalReady(token);
  state.beginCollective(token);
  state.completeCollective(token);
  state.releaseResult(token);
}

} // namespace

TEST_CASE("Accelerator collective serializes one host-staged slot",
          "[drivers][wftrain][accelerator][collective_state]")
{
  AcceleratorCollectiveState state(makePlan(AcceleratorCollectiveTransport::HOST_STAGED_MPI, 1));
  state.beginIteration(state.plan().fingerprint, 7);

  for (std::size_t ordinal = 0; ordinal < 3; ++ordinal)
  {
    const AcceleratorCollectiveToken token = state.acquireProductionSlot();
    CHECK(token.chunk.ordinal == ordinal);
    CHECK(token.chunk.count == (ordinal == 2 ? 1 : 2));
    CHECK_THROWS_WITH(state.acquireProductionSlot(),
                      Catch::Matchers::ContainsSubstring(ordinal == 2 ? "all canonical chunks"
                                                                      : "still owned"));
    completeHostStaged(state, token);
  }

  CHECK(state.releasedChunkCount() == 3);
  CHECK(state.slotState(0) == AcceleratorCollectiveSlotState::AVAILABLE);
  state.finishIteration();
  CHECK(state.iterationState() == AcceleratorCollectiveIterationState::IDLE);
}

TEST_CASE("Accelerator collective overlaps two slots but preserves canonical order",
          "[drivers][wftrain][accelerator][collective_state]")
{
  AcceleratorCollectiveState state(makePlan(AcceleratorCollectiveTransport::DEVICE_MPI, 2, 0));
  CHECK(state.plan().total_sample_count == 5); // A zero-work rank retains the common schedule.
  state.beginIteration(state.plan().fingerprint, state.plan().parameter_version);

  const AcceleratorCollectiveToken first  = state.acquireProductionSlot();
  const AcceleratorCollectiveToken second = state.acquireProductionSlot();
  state.markLocalReady(second);
  CHECK_THROWS_WITH(state.beginCollective(second),
                    Catch::Matchers::ContainsSubstring("canonical order"));

  state.markLocalReady(first);
  state.beginCollective(first);
  state.completeCollective(first);
  CHECK_THROWS_WITH(state.releaseResult(second),
                    Catch::Matchers::ContainsSubstring("required state"));
  state.releaseResult(first);

  state.beginCollective(second);
  state.completeCollective(second);
  state.releaseResult(second);

  const AcceleratorCollectiveToken tail = state.acquireProductionSlot();
  CHECK(tail.slot == 0);
  CHECK(tail.chunk == AcceleratorChunkDescriptor{2, 4, 1});
  completeDeviceMPI(state, tail);
  state.finishIteration();
}

TEST_CASE("Accelerator collective rejects foreign stale and malformed transitions",
          "[drivers][wftrain][accelerator][collective_state]")
{
  AcceleratorCollectiveState state(makePlan(AcceleratorCollectiveTransport::HOST_STAGED_MPI, 2));
  CHECK_THROWS_WITH(state.beginIteration(state.plan().fingerprint + 1, 7),
                    Catch::Matchers::ContainsSubstring("fingerprint"));
  CHECK_THROWS_WITH(state.beginIteration(state.plan().fingerprint, 8),
                    Catch::Matchers::ContainsSubstring("version"));

  state.beginIteration(state.plan().fingerprint, 7);
  const AcceleratorCollectiveToken stale = state.acquireProductionSlot();
  CHECK_THROWS_WITH(state.beginCollective(stale),
                    Catch::Matchers::ContainsSubstring("required state"));
  CHECK(state.slotState(stale.slot) == AcceleratorCollectiveSlotState::PRODUCING);

  state.cancelIteration();
  CHECK(state.failure() == AcceleratorCollectiveFailure::CANCELLED);
  CHECK(state.slotState(stale.slot) == AcceleratorCollectiveSlotState::POISONED);
  CHECK_THROWS_WITH(state.markLocalReady(stale),
                    Catch::Matchers::ContainsSubstring("poisoned"));
  state.recoverAfterExternalDrain();

  state.beginIteration(state.plan().fingerprint, 7);
  CHECK(state.generation() != stale.generation);
  CHECK_THROWS_WITH(state.markLocalReady(stale),
                    Catch::Matchers::ContainsSubstring("live iteration"));
  state.cancelIteration();
  state.recoverAfterExternalDrain();
}

TEST_CASE("Accelerator collective failures poison all live slots and permit drained retry",
          "[drivers][wftrain][accelerator][collective_state]")
{
  constexpr std::array<AcceleratorCollectiveFailure, 4> failures{
      AcceleratorCollectiveFailure::LOCAL_PRODUCTION,
      AcceleratorCollectiveFailure::DEVICE_COPY,
      AcceleratorCollectiveFailure::COLLECTIVE,
      AcceleratorCollectiveFailure::LOCAL_CONSUMPTION};

  AcceleratorCollectiveState state(makePlan(AcceleratorCollectiveTransport::HOST_STAGED_MPI, 2));
  for (const AcceleratorCollectiveFailure failure : failures)
  {
    state.beginIteration(state.plan().fingerprint, 7);
    const AcceleratorCollectiveToken first = state.acquireProductionSlot();
    const AcceleratorCollectiveToken second = state.acquireProductionSlot();
    state.markLocalReady(first);
    state.failIteration(failure);

    CHECK(state.iterationState() == AcceleratorCollectiveIterationState::FAILED);
    CHECK(state.failure() == failure);
    CHECK(state.slotState(first.slot) == AcceleratorCollectiveSlotState::POISONED);
    CHECK(state.slotState(second.slot) == AcceleratorCollectiveSlotState::POISONED);
    CHECK_THROWS(state.finishIteration());
    state.recoverAfterExternalDrain();
    CHECK(state.slotState(0) == AcceleratorCollectiveSlotState::AVAILABLE);
    CHECK(state.slotState(1) == AcceleratorCollectiveSlotState::AVAILABLE);
  }

  state.beginIteration(state.plan().fingerprint, 7);
  while (state.acquiredChunkCount() < state.plan().chunks.size())
  {
    const AcceleratorCollectiveToken token = state.acquireProductionSlot();
    completeHostStaged(state, token);
  }
  state.finishIteration();
  CHECK(state.iterationState() == AcceleratorCollectiveIterationState::IDLE);
}

TEST_CASE("Accelerator collective validates prepared schedule and transport path",
          "[drivers][wftrain][accelerator][collective_state]")
{
  AcceleratorDistributedPlan malformed = makePlan(AcceleratorCollectiveTransport::HOST_STAGED_MPI, 1);
  malformed.chunks.back().offset = 3;
  CHECK_THROWS_AS(AcceleratorCollectiveState(std::move(malformed)), std::invalid_argument);

  AcceleratorCollectiveState host_state(makePlan(AcceleratorCollectiveTransport::HOST_STAGED_MPI, 1));
  host_state.beginIteration(host_state.plan().fingerprint, 7);
  const AcceleratorCollectiveToken host_token = host_state.acquireProductionSlot();
  host_state.markLocalReady(host_token);
  CHECK_THROWS_WITH(host_state.beginCollective(host_token),
                    Catch::Matchers::ContainsSubstring("required state"));
  host_state.cancelIteration();
  host_state.recoverAfterExternalDrain();

  AcceleratorCollectiveState device_state(makePlan(AcceleratorCollectiveTransport::DEVICE_MPI, 1));
  device_state.beginIteration(device_state.plan().fingerprint, 7);
  const AcceleratorCollectiveToken device_token = device_state.acquireProductionSlot();
  device_state.markLocalReady(device_token);
  CHECK_THROWS_WITH(device_state.beginDeviceToHost(device_token),
                    Catch::Matchers::ContainsSubstring("no host staging"));
  CHECK(device_state.slotState(0) == AcceleratorCollectiveSlotState::LOCAL_READY);
  device_state.cancelIteration();
  device_state.recoverAfterExternalDrain();
}

} // namespace qmcplusplus::wftrain
