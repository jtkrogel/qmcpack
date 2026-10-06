//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: QMCPACK developers
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "Utilities/BatchExecutionMemory.h"

#include <limits>
#include <stdexcept>

namespace qmcplusplus
{
namespace
{

/** Build a small exact estimator whose equal tile slopes expose tie ordering. */
BatchMemoryEstimate equalSlopeEstimate(const BatchTileCapacities& capacities)
{
  BatchMemoryEstimate estimate;
  estimate.add(BatchMemoryCategory::FIXED_CLONE_STATE, {100, 20});
  const std::size_t total_tiles = checkedBatchMemoryAdd(
      checkedBatchMemoryAdd(capacities.value, capacities.full_vgl, "test tiles"),
      checkedBatchMemoryAdd(capacities.active_gradient, capacities.ecp_outer, "test tiles"), "test tiles");
  estimate.add(BatchMemoryCategory::INNER_TILE_SCRATCH,
               {checkedBatchMemoryMultiply(total_tiles, 10, "test host tile bytes"),
                checkedBatchMemoryMultiply(total_tiles, 2, "test device tile bytes")});
  return estimate;
}

/** Provide a complete four-mode selection input with small deterministic limits. */
BatchExecutionSelectionInput makeSelectionInput()
{
  BatchExecutionSelectionInput input;
  input.requirements.require(BatchExecutionMode::VALUE);
  input.requirements.require(BatchExecutionMode::FULL_VGL);
  input.requirements.require(BatchExecutionMode::ACTIVE_GRADIENT);
  input.requirements.require(BatchExecutionMode::ECP_OUTER);
  input.topology.initial_walkers_per_crowd = {3, 2, 0};
  input.topology.reserve_walkers_per_crowd = {4, 3, 0};
  input.topology.run_kind                  = "vmc-pbyp";
  input.logical_maximum                    = {8, 8, 8, 8};
  input.preference.id                      = "test-cpu-v1";
  input.preference.preferred               = {4, 4, 4, 4};
  input.participant_ids                    = {"wavefunction/psiformer[0]", "hamiltonian/nonlocal_ecp[0]"};
  return input;
}

} // namespace

TEST_CASE("Batch execution memory checked arithmetic", "[utilities][batch_memory]")
{
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  CHECK(checkedBatchMemoryAdd(2, 3, "test") == 5);
  CHECK(checkedBatchMemoryMultiply(6, 7, "test") == 42);
  CHECK_THROWS_AS(checkedBatchMemoryAdd(maximum, 1, "test"), std::overflow_error);
  CHECK_THROWS_AS(checkedBatchMemoryMultiply(maximum, 2, "test"), std::overflow_error);

  BatchMemoryEstimate per_owner;
  per_owner.add(BatchMemoryCategory::LOGICAL_INPUT_OUTPUT, {11, 7});
  const BatchMemoryEstimate aggregate =
      aggregateBatchMemoryEstimates({{"component/a", 3, per_owner}, {"component/b", 0, per_owner}});
  CHECK(aggregate.at(BatchMemoryCategory::LOGICAL_INPUT_OUTPUT) == BatchMemoryBytes{33, 21});
  CHECK_THROWS(aggregateBatchMemoryEstimates({{"duplicate", 1, per_owner}, {"duplicate", 1, per_owner}}));
}

TEST_CASE("Batch execution memory automatic selection boundaries", "[utilities][batch_memory]")
{
  BatchExecutionSelectionInput input = makeSelectionInput();

  SECTION("generous or absent budgets retain the versioned preference")
  {
    const BatchExecutionPlan plan = selectBatchExecutionPlan(input, equalSlopeEstimate);
    CHECK(plan.selectedCapacities() == BatchTileCapacities{4, 4, 4, 4});
    CHECK(plan.selectedEstimate().total() == BatchMemoryBytes{260, 52});
    CHECK(plan.fixedMinimumEstimate().total() == BatchMemoryBytes{140, 28});
  }

  SECTION("one decrement uses VALUE as the stable equal-saving tie break")
  {
    input.policy.host_budget = 259;
    const BatchExecutionPlan plan = selectBatchExecutionPlan(input, equalSlopeEstimate);
    CHECK(plan.selectedCapacities() == BatchTileCapacities{3, 4, 4, 4});
    CHECK(plan.selectedEstimate().total().host == 250);
  }

  SECTION("a zero automatic preference is promoted to the required minimum")
  {
    input.preference.preferred = {0, 0, 0, 0};
    const BatchExecutionPlan plan = selectBatchExecutionPlan(input, equalSlopeEstimate);
    CHECK(plan.selectedCapacities() == BatchTileCapacities{1, 1, 1, 1});
  }

  SECTION("the exact minimum fits")
  {
    input.policy.host_budget   = 140;
    input.policy.device_budget = 28;
    const BatchExecutionPlan plan = selectBatchExecutionPlan(input, equalSlopeEstimate);
    CHECK(plan.selectedCapacities() == BatchTileCapacities{1, 1, 1, 1});
  }

  SECTION("one byte below the minimum fails before a plan is returned")
  {
    input.policy.host_budget = 139;
    CHECK_THROWS_WITH(selectBatchExecutionPlan(input, equalSlopeEstimate),
                      "Batch execution memory minimum does not fit the configured budget (host=140B, device=28B)");
  }
}

TEST_CASE("Batch execution memory selection relieves the capped space", "[utilities][batch_memory]")
{
  BatchExecutionSelectionInput input = makeSelectionInput();
  input.policy.host_budget            = 114;

  auto asymmetric_estimate = [](const BatchTileCapacities& capacities) {
    BatchMemoryEstimate estimate;
    estimate.add(BatchMemoryCategory::FIXED_CLONE_STATE, {100, 100});
    estimate.add(BatchMemoryCategory::INNER_TILE_SCRATCH,
                 {capacities.full_vgl * 10 + capacities.value,
                  capacities.value * 100 + capacities.full_vgl});
    return estimate;
  };

  const BatchExecutionPlan plan = selectBatchExecutionPlan(input, asymmetric_estimate);
  // Only host is capped, so shrinking FULL_VGL provides useful relief even
  // though shrinking VALUE would release much more uncapped device storage.
  CHECK(plan.selectedCapacities().value == 4);
  CHECK(plan.selectedCapacities().full_vgl == 1);
}

TEST_CASE("Batch execution memory hard requests and logical modes", "[utilities][batch_memory]")
{
  BatchExecutionSelectionInput input = makeSelectionInput();
  input.policy.tiles.value            = BatchTileRequest::fixed(7);
  input.logical_maximum.value         = 5;
  input.policy.host_budget            = 180;

  const BatchExecutionPlan plan = selectBatchExecutionPlan(input, equalSlopeEstimate);
  CHECK(plan.selectedCapacities().value == 5);
  CHECK(plan.selectedCapacities() == BatchTileCapacities{5, 1, 1, 1});

  SECTION("a hard request is not reduced to rescue an impossible budget")
  {
    input.policy.host_budget = 179;
    CHECK_THROWS(selectBatchExecutionPlan(input, equalSlopeEstimate));
  }

  SECTION("unused modes have zero capacity even when a preference is nonzero")
  {
    BatchExecutionSelectionInput value_only;
    value_only.requirements.require(BatchExecutionMode::VALUE);
    value_only.logical_maximum      = {9, 9, 9, 9};
    value_only.preference.preferred = {4, 4, 4, 4};
    const BatchExecutionPlan value_plan = selectBatchExecutionPlan(value_only, equalSlopeEstimate);
    CHECK(value_plan.selectedCapacities() == BatchTileCapacities{4, 0, 0, 0});
  }

  SECTION("weighted ECP score requires the shared ECP outer capacity")
  {
    BatchExecutionSelectionInput weighted_ecp;
    weighted_ecp.requirements.require(BatchExecutionMode::ECP_WEIGHTED_SCORE);
    weighted_ecp.logical_maximum      = {0, 0, 0, 7};
    weighted_ecp.preference.preferred = {4, 4, 4, 4};
    const BatchExecutionPlan ecp_plan = selectBatchExecutionPlan(weighted_ecp, equalSlopeEstimate);
    CHECK(ecp_plan.selectedCapacities() == BatchTileCapacities{0, 0, 0, 4});
  }
}

TEST_CASE("Batch execution memory plans have stable exact-content fingerprints", "[utilities][batch_memory]")
{
  BatchExecutionSelectionInput input = makeSelectionInput();
  input.policy.host_budget            = 240;
  const BatchExecutionPlan first      = selectBatchExecutionPlan(input, equalSlopeEstimate);
  const BatchExecutionPlan repeated   = selectBatchExecutionPlan(input, equalSlopeEstimate);
  CHECK(first.fingerprint() == repeated.fingerprint());

  input.topology.initial_walkers_per_crowd = {2, 3, 0};
  const BatchExecutionPlan changed_topology = selectBatchExecutionPlan(input, equalSlopeEstimate);
  CHECK(first.fingerprint() != changed_topology.fingerprint());

  input.topology.initial_walkers_per_crowd = {3, 2, 0};
  input.participant_ids.push_back("wavefunction/psiformer[1]");
  const BatchExecutionPlan changed_participants = selectBatchExecutionPlan(input, equalSlopeEstimate);
  CHECK(first.fingerprint() != changed_participants.fingerprint());

  SECTION("the fixed minimum estimate is part of the immutable content")
  {
    BatchExecutionSelectionInput minimum_input;
    minimum_input.requirements.require(BatchExecutionMode::VALUE);
    minimum_input.logical_maximum      = {2, 0, 0, 0};
    minimum_input.preference.preferred = {2, 0, 0, 0};

    auto estimate_with_minimum = [](std::size_t minimum_bytes) {
      return [minimum_bytes](const BatchTileCapacities& capacities) {
        BatchMemoryEstimate estimate;
        estimate.add(BatchMemoryCategory::FIXED_CLONE_STATE,
                     {capacities.value == 1 ? minimum_bytes : 17, 0});
        return estimate;
      };
    };

    const BatchExecutionPlan minimum_one =
        selectBatchExecutionPlan(minimum_input, estimate_with_minimum(1));
    const BatchExecutionPlan minimum_two =
        selectBatchExecutionPlan(minimum_input, estimate_with_minimum(2));
    CHECK(minimum_one.selectedEstimate() == minimum_two.selectedEstimate());
    CHECK_FALSE(minimum_one.fixedMinimumEstimate() == minimum_two.fixedMinimumEstimate());
    CHECK(minimum_one.fingerprint() != minimum_two.fingerprint());
  }
}

TEST_CASE("Batch execution memory compares dual-space relief without overflow", "[utilities][batch_memory]")
{
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  BatchExecutionSelectionInput input;
  input.requirements.require(BatchExecutionMode::VALUE);
  input.requirements.require(BatchExecutionMode::FULL_VGL);
  input.logical_maximum      = {2, 2, 0, 0};
  input.preference.preferred = {2, 2, 0, 0};
  input.policy.host_budget   = maximum - 1;
  input.policy.device_budget = maximum - 1;

  auto large_dual_space_estimate = [maximum](const BatchTileCapacities& capacities) {
    BatchMemoryEstimate estimate;
    BatchMemoryBytes bytes;
    if (capacities.value == 2 && capacities.full_vgl == 2)
      bytes = {maximum, maximum};
    else if (capacities.value == 1 && capacities.full_vgl == 2)
      bytes = {0, maximum / 2};
    else if (capacities.value == 2 && capacities.full_vgl == 1)
      bytes = {maximum / 2, maximum / 2};
    estimate.add(BatchMemoryCategory::INNER_TILE_SCRATCH, bytes);
    return estimate;
  };

  const BatchExecutionPlan plan = selectBatchExecutionPlan(input, large_dual_space_estimate);
  CHECK(plan.selectedCapacities() == BatchTileCapacities{1, 2, 0, 0});
}

TEST_CASE("Batch execution memory rejects invalid selection contracts", "[utilities][batch_memory]")
{
  BatchExecutionSelectionInput input = makeSelectionInput();

  SECTION("required logical maximum is zero")
  {
    input.logical_maximum.value = 0;
    CHECK_THROWS(selectBatchExecutionPlan(input, equalSlopeEstimate));
  }

  SECTION("reserve topology must preserve the crowd count")
  {
    input.topology.reserve_walkers_per_crowd = {7};
    CHECK_THROWS(selectBatchExecutionPlan(input, equalSlopeEstimate));
  }

  SECTION("stable participant IDs must be unique")
  {
    input.participant_ids = {"same", "same"};
    CHECK_THROWS(selectBatchExecutionPlan(input, equalSlopeEstimate));
  }

  SECTION("an estimator must be monotone under tile reduction")
  {
    input.policy.host_budget = 15;
    auto nonmonotone = [](const BatchTileCapacities& capacities) {
      BatchMemoryEstimate estimate;
      std::size_t tile_bytes =
          capacities.value + capacities.full_vgl + capacities.active_gradient + capacities.ecp_outer;
      if (capacities.value == 3)
        tile_bytes = 1000;
      estimate.add(BatchMemoryCategory::INNER_TILE_SCRATCH, {tile_bytes, 0});
      return estimate;
    };
    CHECK_THROWS(selectBatchExecutionPlan(input, nonmonotone));
  }
}

} // namespace qmcplusplus
