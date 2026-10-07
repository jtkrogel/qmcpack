//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_AcceleratorTrainingPlan.cpp
 * @brief Pure tests for distributed accelerator planning and publication consensus.
 */

#include "QMCDrivers/WFTrain/AcceleratorTrainingPlan.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <limits>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Construct one valid rank record with independently variable work and capacity.
AcceleratorRankPlanInput makeRank(std::size_t rank,
                                  std::size_t local_samples,
                                  std::size_t chunk,
                                  bool device_mpi)
{
  return {/*rank=*/rank,
          /*participant_count=*/3,
          /*node_id=*/rank < 2 ? 10U : 11U,
          /*device_id=*/rank < 2 ? rank : 0U,
          /*local_sample_count=*/local_samples,
          /*device_budget_bytes=*/std::size_t{4096},
          /*base_required_device_bytes=*/2048,
          /*maximum_chunk_elements=*/chunk,
          /*parameter_version=*/7,
          /*model_fingerprint=*/101,
          /*precision_fingerprint=*/202,
          /*device_mpi_available=*/device_mpi};
}

} // namespace

TEST_CASE("Accelerator training plan selects one bounded canonical schedule",
          "[drivers][wftrain][accelerator]")
{
  const std::vector<AcceleratorRankPlanInput> ranks{
      makeRank(2, 0, 5, true), makeRank(0, 7, 4, true), makeRank(1, 3, 2, false)};
  const AcceleratorDistributedPlan plan = makeAcceleratorDistributedPlan(
      ranks, /*parameter_count=*/5, /*collective_element_bytes=*/16,
      AcceleratorTransportRequest::AUTO, /*buffer_slots=*/2);

  CHECK(plan.transport == AcceleratorCollectiveTransport::HOST_STAGED_MPI);
  CHECK(plan.participant_count == 3);
  CHECK(plan.total_sample_count == 10);
  CHECK(plan.parameter_version == 7);
  CHECK(plan.chunk_elements == 2);
  CHECK(plan.host_staging_bytes_per_rank == 64);
  CHECK(plan.device_collective_bytes_per_rank == 64);
  CHECK(plan.required_device_bytes_per_rank == std::vector<std::size_t>{2112, 2112, 2112});
  CHECK(plan.fingerprint != 0);
  REQUIRE(plan.chunks.size() == 3);
  CHECK(plan.chunks[0] == AcceleratorChunkDescriptor{0, 0, 2});
  CHECK(plan.chunks[1] == AcceleratorChunkDescriptor{1, 2, 2});
  CHECK(plan.chunks[2] == AcceleratorChunkDescriptor{2, 4, 1});

  const AcceleratorDistributedPlan repeated = makeAcceleratorDistributedPlan(
      ranks, 5, 16, AcceleratorTransportRequest::AUTO, 2);
  CHECK(repeated.fingerprint == plan.fingerprint);
  CHECK(repeated.chunks == plan.chunks);
}

TEST_CASE("Accelerator training transport and placement requests fail closed",
          "[drivers][wftrain][accelerator]")
{
  std::vector<AcceleratorRankPlanInput> ranks{
      makeRank(0, 1, 4, true), makeRank(1, 1, 4, true), makeRank(2, 1, 4, true)};
  AcceleratorDistributedPlan plan = makeAcceleratorDistributedPlan(
      ranks, 8, 8, AcceleratorTransportRequest::DEVICE_MPI);
  CHECK(plan.transport == AcceleratorCollectiveTransport::DEVICE_MPI);
  CHECK(plan.host_staging_bytes_per_rank == 0);
  CHECK(plan.device_collective_bytes_per_rank == 32);

  ranks[1].device_mpi_available = false;
  CHECK_THROWS_WITH(makeAcceleratorDistributedPlan(
                        ranks, 8, 8, AcceleratorTransportRequest::DEVICE_MPI),
                    Catch::Matchers::ContainsSubstring("not available on every rank"));

  ranks[1].device_mpi_available = true;
  ranks[1].device_id            = ranks[0].device_id;
  CHECK_THROWS_WITH(makeAcceleratorDistributedPlan(
                        ranks, 8, 8, AcceleratorTransportRequest::HOST_STAGED_MPI),
                    Catch::Matchers::ContainsSubstring("same device"));
}

TEST_CASE("Accelerator training plan validates all rank metadata and budgets",
          "[drivers][wftrain][accelerator]")
{
  std::vector<AcceleratorRankPlanInput> ranks{
      makeRank(0, 1, 4, false), makeRank(1, 1, 4, false), makeRank(2, 1, 4, false)};

  ranks[2].precision_fingerprint = 999;
  CHECK_THROWS_AS(makeAcceleratorDistributedPlan(
                      ranks, 8, 16, AcceleratorTransportRequest::AUTO),
                  std::invalid_argument);
  ranks[2].precision_fingerprint = 202;

  ranks[1].base_required_device_bytes = 4097;
  CHECK_THROWS_WITH(makeAcceleratorDistributedPlan(
                        ranks, 8, 16, AcceleratorTransportRequest::AUTO),
                    Catch::Matchers::ContainsSubstring("hard budget"));
  ranks[1].base_required_device_bytes = 2048;

  // The collective allocation is part of feasibility, not an unaccounted
  // post-plan addition to an otherwise legal base workspace.
  ranks[0].device_budget_bytes = 2050;
  CHECK_THROWS_WITH(makeAcceleratorDistributedPlan(
                        ranks, 8, 16, AcceleratorTransportRequest::AUTO),
                    Catch::Matchers::ContainsSubstring("hard budget"));
  ranks[0].device_budget_bytes = 4096;

  ranks[0].maximum_chunk_elements = 0;
  CHECK_THROWS_AS(makeAcceleratorDistributedPlan(
                      ranks, 8, 16, AcceleratorTransportRequest::AUTO),
                  std::invalid_argument);
  ranks[0].maximum_chunk_elements = 4;

  CHECK_THROWS_AS(makeAcceleratorDistributedPlan(
                      ranks, 0, 16, AcceleratorTransportRequest::AUTO),
                  std::invalid_argument);
  CHECK_THROWS_AS(makeAcceleratorDistributedPlan(
                      ranks, 8, 16, AcceleratorTransportRequest::AUTO, 3),
                  std::invalid_argument);
  for (AcceleratorRankPlanInput& rank : ranks)
    rank.maximum_chunk_elements = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(makeAcceleratorDistributedPlan(
                      ranks, std::numeric_limits<std::size_t>::max(), 16,
                      AcceleratorTransportRequest::AUTO, 2),
                  std::overflow_error);
}

TEST_CASE("Accelerator publication consensus is failure atomic",
          "[drivers][wftrain][accelerator]")
{
  std::vector<AcceleratorPublicationReport> reports{
      {2, 3, 7, 8, AcceleratorPublicationStatus::READY},
      {0, 3, 7, 8, AcceleratorPublicationStatus::READY},
      {1, 3, 7, 8, AcceleratorPublicationStatus::READY}};
  AcceleratorPublicationDecision decision = decideAcceleratorPublication(reports);
  CHECK(decision.accepted);
  CHECK(decision.active_version == 8);
  CHECK_FALSE(decision.failing_rank);

  reports[1].status = AcceleratorPublicationStatus::NONFINITE_CANDIDATE;
  decision = decideAcceleratorPublication(reports);
  CHECK_FALSE(decision.accepted);
  CHECK(decision.active_version == 7);
  REQUIRE(decision.failing_rank);
  CHECK(*decision.failing_rank == 0);
  CHECK(decision.failure_status == AcceleratorPublicationStatus::NONFINITE_CANDIDATE);

  reports[1].status = AcceleratorPublicationStatus::READY;
  reports[2].candidate_version = 9;
  CHECK_THROWS_AS(decideAcceleratorPublication(reports), std::invalid_argument);
}

TEST_CASE("Accelerator checkpoint restart distinguishes exact and repartitioned runs",
          "[drivers][wftrain][accelerator]")
{
  AcceleratorCheckpointManifest manifest{/*complete=*/true,
                                         /*participant_count=*/4,
                                         /*parameter_version=*/12,
                                         /*model_fingerprint=*/101,
                                         /*precision_fingerprint=*/202,
                                         /*plan_fingerprint=*/303};
  CHECK(validateAcceleratorRestart(manifest, 4, 101, 202, 303, false) ==
        AcceleratorRestartMode::EXACT_DECOMPOSITION);
  CHECK(validateAcceleratorRestart(manifest, 2, 101, 202, 999, true) ==
        AcceleratorRestartMode::STATISTICAL_REPARTITION);
  CHECK_THROWS_WITH(validateAcceleratorRestart(manifest, 2, 101, 202, 999, false),
                    Catch::Matchers::ContainsSubstring("rank-count change"));

  CHECK_THROWS_WITH(validateAcceleratorRestart(manifest, 4, 101, 202, 999, false),
                    Catch::Matchers::ContainsSubstring("execution-plan fingerprint"));

  manifest.complete = false;
  CHECK_THROWS_WITH(validateAcceleratorRestart(manifest, 4, 101, 202, 303, false),
                    Catch::Matchers::ContainsSubstring("incomplete"));
  manifest.complete = true;
  CHECK_THROWS_AS(validateAcceleratorRestart(manifest, 4, 999, 202, 303, false),
                  std::invalid_argument);
}

} // namespace qmcplusplus::wftrain
