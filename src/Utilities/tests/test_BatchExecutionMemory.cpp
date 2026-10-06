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
#include <memory>
#include <stdexcept>
#include <utility>

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

/** Adapt a single-owner estimate to the participant-aware provider contract. */
BatchMemoryContributionProvider makeProvider(BatchMemoryEstimator estimator,
                                             std::string participant_id = "twf/component/0/Test/component",
                                             std::size_t owner_multiplicity = 1)
{
  return [estimator = std::move(estimator), participant_id = std::move(participant_id),
          owner_multiplicity](const BatchExecutionPlanningContext& context) {
    BatchMemoryContribution contribution;
    contribution.logical_maximum    = context.logical_maximum;
    contribution.owner_multiplicity = owner_multiplicity;
    contribution.fully_accounted    = true;
    contribution.per_owner          = estimator(context.candidate_capacities);
    return std::vector<BatchMemoryParticipantContribution>{{participant_id, std::move(contribution)}};
  };
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
  input.active_parameter_count             = 17;
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

  BatchMemoryContribution contribution;
  contribution.owner_multiplicity = 3;
  contribution.fully_accounted    = true;
  contribution.per_owner.add(BatchMemoryCategory::LOGICAL_INPUT_OUTPUT, {11, 7});
  BatchMemoryContribution zero_owner = contribution;
  zero_owner.owner_multiplicity       = 0;
  const BatchMemoryEstimate aggregate = aggregateBatchMemoryContributions(
      {{"component/a", contribution}, {"component/b", zero_owner}});
  CHECK(aggregate.at(BatchMemoryCategory::LOGICAL_INPUT_OUTPUT) == BatchMemoryBytes{33, 21});
  CHECK_THROWS(aggregateBatchMemoryContributions(
      {{"duplicate", contribution}, {"duplicate", contribution}}));

  contribution.per_owner = {};
  contribution.per_owner.add(BatchMemoryCategory::LOGICAL_INPUT_OUTPUT, {maximum, 0});
  contribution.owner_multiplicity = 2;
  CHECK_THROWS_AS(aggregateBatchMemoryContributions({{"overflow", contribution}}), std::overflow_error);
}

TEST_CASE("Batch execution logical maxima combine elementwise", "[utilities][batch_memory]")
{
  BatchTileCapacities aggregate{4, 1, 8, 2};
  includeBatchExecutionLogicalMaximum(aggregate, {3, 7, 5, 9});
  CHECK(aggregate == BatchTileCapacities{4, 7, 8, 9});

  BatchExecutionRequirements scalar;
  scalar.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  CHECK(batchExecutionModeIsRequired(scalar, BatchExecutionMode::VALUE));
  CHECK_FALSE(batchExecutionModeIsRequired(scalar, BatchExecutionMode::FULL_VGL));

  BatchExecutionRequirements weighted_ecp;
  weighted_ecp.require(BatchExecutionMode::ECP_WEIGHTED_SCORE);
  CHECK(batchExecutionModeIsRequired(weighted_ecp, BatchExecutionMode::ECP_OUTER));
}

TEST_CASE("Batch execution memory automatic selection boundaries", "[utilities][batch_memory]")
{
  BatchExecutionSelectionInput input = makeSelectionInput();

  SECTION("generous or absent budgets retain the versioned preference")
  {
    const BatchExecutionPlan plan = selectBatchExecutionPlan(input, makeProvider(equalSlopeEstimate));
    CHECK(plan.minimumCapacities() == BatchTileCapacities{1, 1, 1, 1});
    CHECK(plan.selectedCapacities() == BatchTileCapacities{4, 4, 4, 4});
    CHECK(plan.selectedEstimate().total() == BatchMemoryBytes{260, 52});
    CHECK(plan.fixedMinimumEstimate().total() == BatchMemoryBytes{140, 28});
    CHECK(plan.activeParameterCount() == 17);
    REQUIRE(plan.participantEvidence().size() == 1);
    const BatchMemoryParticipantEvidence& evidence = plan.participantEvidence().front();
    CHECK(evidence.participant_id == "twf/component/0/Test/component");
    CHECK(evidence.logical_maximum == input.logical_maximum);
    CHECK(evidence.owner_multiplicity == 1);
    CHECK(evidence.fully_accounted);
    CHECK(evidence.fixed_minimum_per_owner.total() == BatchMemoryBytes{140, 28});
    CHECK(evidence.selected_per_owner.total() == BatchMemoryBytes{260, 52});
  }

  SECTION("one decrement uses VALUE as the stable equal-saving tie break")
  {
    input.policy.host_budget = 259;
    const BatchExecutionPlan plan = selectBatchExecutionPlan(input, makeProvider(equalSlopeEstimate));
    CHECK(plan.selectedCapacities() == BatchTileCapacities{3, 4, 4, 4});
    CHECK(plan.selectedEstimate().total().host == 250);
  }

  SECTION("a zero automatic preference is promoted to the required minimum")
  {
    input.preference.preferred = {0, 0, 0, 0};
    const BatchExecutionPlan plan = selectBatchExecutionPlan(input, makeProvider(equalSlopeEstimate));
    CHECK(plan.selectedCapacities() == BatchTileCapacities{1, 1, 1, 1});
  }

  SECTION("the exact minimum fits")
  {
    input.policy.host_budget   = 140;
    input.policy.device_budget = 28;
    const BatchExecutionPlan plan = selectBatchExecutionPlan(input, makeProvider(equalSlopeEstimate));
    CHECK(plan.selectedCapacities() == BatchTileCapacities{1, 1, 1, 1});
  }

  SECTION("one byte below the minimum fails before a plan is returned")
  {
    input.policy.host_budget = 139;
    CHECK_THROWS_WITH(selectBatchExecutionPlan(input, makeProvider(equalSlopeEstimate)),
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

  const BatchExecutionPlan plan = selectBatchExecutionPlan(input, makeProvider(asymmetric_estimate));
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

  const BatchExecutionPlan plan = selectBatchExecutionPlan(input, makeProvider(equalSlopeEstimate));
  CHECK(plan.minimumCapacities() == BatchTileCapacities{5, 1, 1, 1});
  CHECK(plan.selectedCapacities().value == 5);
  CHECK(plan.selectedCapacities() == BatchTileCapacities{5, 1, 1, 1});

  SECTION("a hard request is not reduced to rescue an impossible budget")
  {
    input.policy.host_budget = 179;
    CHECK_THROWS(selectBatchExecutionPlan(input, makeProvider(equalSlopeEstimate)));
  }

  SECTION("unused modes have zero capacity even when a preference is nonzero")
  {
    BatchExecutionSelectionInput value_only;
    value_only.requirements.require(BatchExecutionMode::VALUE);
    value_only.logical_maximum      = {9, 9, 9, 9};
    value_only.preference.preferred = {4, 4, 4, 4};
    const BatchExecutionPlan value_plan =
        selectBatchExecutionPlan(value_only, makeProvider(equalSlopeEstimate));
    CHECK(value_plan.selectedCapacities() == BatchTileCapacities{4, 0, 0, 0});
  }

  SECTION("scalar value compatibility alone activates the value capacity")
  {
    BatchExecutionSelectionInput scalar_compatibility;
    scalar_compatibility.requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
    scalar_compatibility.logical_maximum      = {9, 9, 9, 9};
    scalar_compatibility.preference.preferred = {4, 4, 4, 4};
    const BatchExecutionPlan scalar_plan =
        selectBatchExecutionPlan(scalar_compatibility, makeProvider(equalSlopeEstimate));
    CHECK(scalar_plan.minimumCapacities() == BatchTileCapacities{1, 0, 0, 0});
    CHECK(scalar_plan.selectedCapacities() == BatchTileCapacities{4, 0, 0, 0});
  }

  SECTION("weighted ECP score requires the shared ECP outer capacity")
  {
    BatchExecutionSelectionInput weighted_ecp;
    weighted_ecp.requirements.require(BatchExecutionMode::ECP_WEIGHTED_SCORE);
    weighted_ecp.logical_maximum      = {0, 0, 0, 7};
    weighted_ecp.preference.preferred = {4, 4, 4, 4};
    const BatchExecutionPlan ecp_plan =
        selectBatchExecutionPlan(weighted_ecp, makeProvider(equalSlopeEstimate));
    CHECK(ecp_plan.minimumCapacities() == BatchTileCapacities{0, 0, 0, 1});
    CHECK(ecp_plan.selectedCapacities() == BatchTileCapacities{0, 0, 0, 4});
  }

  SECTION("T-move candidates require the shared ECP outer capacity")
  {
    BatchExecutionSelectionInput tmove_ecp;
    tmove_ecp.requirements.require(BatchExecutionMode::ECP_TMOVE_CANDIDATES);
    tmove_ecp.logical_maximum      = {0, 0, 0, 7};
    tmove_ecp.preference.preferred = {4, 4, 4, 4};
    const BatchExecutionPlan ecp_plan =
        selectBatchExecutionPlan(tmove_ecp, makeProvider(equalSlopeEstimate));
    CHECK(ecp_plan.selectedCapacities() == BatchTileCapacities{0, 0, 0, 4});
  }

  SECTION("listener output requires the shared ECP outer capacity")
  {
    BatchExecutionSelectionInput listener_ecp;
    listener_ecp.requirements.require(BatchExecutionMode::ECP_LISTENER_OUTPUT);
    listener_ecp.logical_maximum      = {0, 0, 0, 7};
    listener_ecp.preference.preferred = {4, 4, 4, 4};
    const BatchExecutionPlan ecp_plan =
        selectBatchExecutionPlan(listener_ecp, makeProvider(equalSlopeEstimate));
    CHECK(ecp_plan.selectedCapacities() == BatchTileCapacities{0, 0, 0, 4});
  }
}

TEST_CASE("Batch execution memory plans have stable exact-content fingerprints", "[utilities][batch_memory]")
{
  BatchExecutionSelectionInput input = makeSelectionInput();
  input.policy.host_budget            = 240;
  const BatchExecutionPlan first    = selectBatchExecutionPlan(input, makeProvider(equalSlopeEstimate));
  const BatchExecutionPlan repeated = selectBatchExecutionPlan(input, makeProvider(equalSlopeEstimate));
  CHECK(first.fingerprint() == repeated.fingerprint());

  input.topology.initial_walkers_per_crowd = {2, 3, 0};
  const BatchExecutionPlan changed_topology =
      selectBatchExecutionPlan(input, makeProvider(equalSlopeEstimate));
  CHECK(first.fingerprint() != changed_topology.fingerprint());

  input.topology.initial_walkers_per_crowd = {3, 2, 0};
  const BatchExecutionPlan changed_participants = selectBatchExecutionPlan(
      input, makeProvider(equalSlopeEstimate, "twf/component/1/Test/component"));
  CHECK(first.fingerprint() != changed_participants.fingerprint());

  input.active_parameter_count = 18;
  const BatchExecutionPlan changed_parameter_count =
      selectBatchExecutionPlan(input, makeProvider(equalSlopeEstimate));
  CHECK(first.fingerprint() != changed_parameter_count.fingerprint());

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
        selectBatchExecutionPlan(minimum_input, makeProvider(estimate_with_minimum(1)));
    const BatchExecutionPlan minimum_two =
        selectBatchExecutionPlan(minimum_input, makeProvider(estimate_with_minimum(2)));
    CHECK(minimum_one.selectedEstimate() == minimum_two.selectedEstimate());
    CHECK_FALSE(minimum_one.fixedMinimumEstimate() == minimum_two.fixedMinimumEstimate());
    CHECK(minimum_one.fingerprint() != minimum_two.fingerprint());
  }

  SECTION("ordered participant evidence is fingerprinted even when aggregate totals match")
  {
    BatchExecutionSelectionInput evidence_input;
    evidence_input.requirements.require(BatchExecutionMode::VALUE);
    evidence_input.logical_maximum      = {1, 0, 0, 0};
    evidence_input.preference.preferred = {1, 0, 0, 0};

    auto split_provider = [](bool exchange_bytes, bool reverse_order) {
      return [exchange_bytes, reverse_order](const BatchExecutionPlanningContext& context) {
        BatchMemoryContribution first;
        first.logical_maximum    = context.logical_maximum;
        first.owner_multiplicity = 1;
        first.fully_accounted    = true;
        first.per_owner.add(BatchMemoryCategory::FIXED_CLONE_STATE,
                            {exchange_bytes ? 20U : 10U, 0});

        BatchMemoryContribution second = first;
        second.per_owner                = {};
        second.per_owner.add(BatchMemoryCategory::FIXED_CLONE_STATE,
                             {exchange_bytes ? 10U : 20U, 0});

        if (reverse_order)
          return std::vector<BatchMemoryParticipantContribution>{{"participant/b", std::move(second)},
                                                                  {"participant/a", std::move(first)}};
        return std::vector<BatchMemoryParticipantContribution>{{"participant/a", std::move(first)},
                                                                {"participant/b", std::move(second)}};
      };
    };

    const BatchExecutionPlan original =
        selectBatchExecutionPlan(evidence_input, split_provider(false, false));
    const BatchExecutionPlan changed_detail =
        selectBatchExecutionPlan(evidence_input, split_provider(true, false));
    const BatchExecutionPlan changed_order =
        selectBatchExecutionPlan(evidence_input, split_provider(false, true));
    CHECK(original.selectedEstimate() == changed_detail.selectedEstimate());
    CHECK(original.selectedEstimate() == changed_order.selectedEstimate());
    CHECK(original.fingerprint() != changed_detail.fingerprint());
    CHECK(original.fingerprint() != changed_order.fingerprint());
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

  const BatchExecutionPlan plan =
      selectBatchExecutionPlan(input, makeProvider(large_dual_space_estimate));
  CHECK(plan.selectedCapacities() == BatchTileCapacities{1, 2, 0, 0});
}

TEST_CASE("Batch execution planning context reaches participant providers", "[utilities][batch_memory]")
{
  BatchExecutionSelectionInput input;
  input.requirements.require(BatchExecutionMode::VALUE);
  input.topology.initial_walkers_per_crowd = {2, 0};
  input.topology.reserve_walkers_per_crowd = {3, 1};
  input.topology.run_kind                  = "context-test";
  input.logical_maximum                    = {1, 0, 0, 0};
  input.preference.preferred               = {1, 0, 0, 0};
  input.active_parameter_count             = 1234;

  bool provider_called = false;
  auto provider = [&](const BatchExecutionPlanningContext& context) {
    provider_called = true;
    CHECK(context.requirements == input.requirements);
    CHECK(context.topology.initial_walkers_per_crowd == input.topology.initial_walkers_per_crowd);
    CHECK(context.topology.reserve_walkers_per_crowd == input.topology.reserve_walkers_per_crowd);
    CHECK(context.topology.run_kind == "context-test");
    CHECK(context.logical_maximum == input.logical_maximum);
    CHECK(context.candidate_capacities == BatchTileCapacities{1, 0, 0, 0});
    CHECK(context.active_parameter_count == 1234);

    BatchMemoryContribution contribution;
    contribution.logical_maximum    = context.logical_maximum;
    contribution.owner_multiplicity = 1;
    contribution.fully_accounted    = true;
    return std::vector<BatchMemoryParticipantContribution>{{"participant/context", contribution}};
  };

  const BatchExecutionPlan plan = selectBatchExecutionPlan(input, provider);
  CHECK(provider_called);
  CHECK(plan.activeParameterCount() == 1234);
}

TEST_CASE("Batch execution participant views have explicit shared and null semantics",
          "[utilities][batch_memory]")
{
  BatchExecutionSelectionInput input;
  input.requirements.require(BatchExecutionMode::VALUE);
  input.logical_maximum      = {1, 0, 0, 0};
  input.preference.preferred = {1, 0, 0, 0};

  auto provider = [](const BatchExecutionPlanningContext& context) {
    BatchMemoryContribution contribution;
    contribution.logical_maximum    = context.logical_maximum;
    contribution.owner_multiplicity = 1;
    contribution.fully_accounted    = true;
    return std::vector<BatchMemoryParticipantContribution>{{"participant/a", contribution},
                                                            {"participant/b", contribution}};
  };

  auto shared_plan =
      std::make_shared<const BatchExecutionPlan>(selectBatchExecutionPlan(input, provider));
  const BatchExecutionParticipantPlan view_a =
      makeBatchExecutionParticipantPlan(shared_plan, "participant/a");
  const BatchExecutionParticipantPlan repeated_a =
      makeBatchExecutionParticipantPlan(shared_plan, "participant/a");
  const BatchExecutionParticipantPlan view_b =
      makeBatchExecutionParticipantPlan(shared_plan, "participant/b");

  CHECK(view_a.hasPlan());
  CHECK(static_cast<bool>(view_a));
  CHECK(&view_a.plan() == shared_plan.get());
  CHECK(view_a.evidence().participant_id == "participant/a");
  CHECK(view_a.sameBinding(repeated_a));
  CHECK_FALSE(view_a.sameBinding(view_b));

  auto equivalent_plan = std::make_shared<const BatchExecutionPlan>(*shared_plan);
  const BatchExecutionParticipantPlan equivalent_a =
      makeBatchExecutionParticipantPlan(equivalent_plan, "participant/a");
  CHECK(equivalent_a.plan().fingerprint() == view_a.plan().fingerprint());
  CHECK_FALSE(view_a.sameBinding(equivalent_a));

  const BatchExecutionParticipantPlan empty =
      makeBatchExecutionParticipantPlan(std::shared_ptr<const BatchExecutionPlan>{}, "ignored/id");
  CHECK_FALSE(empty.hasPlan());
  CHECK_FALSE(static_cast<bool>(empty));
  CHECK(empty.sameBinding(BatchExecutionParticipantPlan{}));
  CHECK_FALSE(empty.sameBinding(view_a));
  CHECK_THROWS_AS(empty.plan(), std::logic_error);
  CHECK_THROWS_AS(empty.evidence(), std::logic_error);

  CHECK_THROWS_AS(makeBatchExecutionParticipantPlan(shared_plan, ""), std::invalid_argument);
  CHECK_THROWS_AS(makeBatchExecutionParticipantPlan(shared_plan, "participant/missing"), std::invalid_argument);
}

TEST_CASE("Batch participant identity segments are unambiguous", "[utilities][batch_memory]")
{
  CHECK(escapeBatchParticipantIdSegment("Az09-._~") == "Az09-._~");
  CHECK(escapeBatchParticipantIdSegment("A/B% C") == "A%2FB%25%20C");
  const std::string arbitrary_bytes{"\x01\xff", 2};
  CHECK(escapeBatchParticipantIdSegment(arbitrary_bytes) == "%01%FF");
  CHECK(escapeBatchParticipantIdSegment("a/b") + "/c" !=
        escapeBatchParticipantIdSegment("a") + "/" + escapeBatchParticipantIdSegment("b/c"));
}

TEST_CASE("Batch execution memory rejects invalid selection contracts", "[utilities][batch_memory]")
{
  BatchExecutionSelectionInput input = makeSelectionInput();

  SECTION("a contribution provider is required")
  {
    CHECK_THROWS_AS(selectBatchExecutionPlan(input, BatchMemoryContributionProvider{}), std::invalid_argument);
  }

  SECTION("required logical maximum is zero")
  {
    input.logical_maximum.value = 0;
    CHECK_THROWS(selectBatchExecutionPlan(input, makeProvider(equalSlopeEstimate)));
  }

  SECTION("reserve topology must preserve the crowd count")
  {
    input.topology.reserve_walkers_per_crowd = {7};
    CHECK_THROWS(selectBatchExecutionPlan(input, makeProvider(equalSlopeEstimate)));
  }

  SECTION("rank reserve topology must cover the initial population")
  {
    input.topology.reserve_walkers_per_crowd = {2, 2, 0};
    CHECK_THROWS_AS(selectBatchExecutionPlan(input, makeProvider(equalSlopeEstimate)), std::invalid_argument);
  }

  SECTION("participant IDs must be nonempty and unique")
  {
    auto empty_id = makeProvider(equalSlopeEstimate, "");
    CHECK_THROWS_AS(selectBatchExecutionPlan(input, empty_id), std::invalid_argument);

    auto duplicate_ids = [base = makeProvider(equalSlopeEstimate)](
                             const BatchExecutionPlanningContext& context) mutable {
      auto contributions = base(context);
      contributions.push_back(contributions.front());
      return contributions;
    };
    CHECK_THROWS_AS(selectBatchExecutionPlan(input, duplicate_ids), std::invalid_argument);
  }

  SECTION("unaccounted participants fail instead of silently contributing zero")
  {
    auto incomplete = [](const BatchExecutionPlanningContext& context) {
      BatchMemoryContribution contribution;
      contribution.logical_maximum = context.logical_maximum;
      return std::vector<BatchMemoryParticipantContribution>{{"participant/incomplete", contribution}};
    };
    CHECK_THROWS_WITH(selectBatchExecutionPlan(input, incomplete),
                      "Batch memory participant is not fully accounted: participant/incomplete");
  }

  SECTION("participant logical maxima cannot exceed the driver envelope")
  {
    auto excessive = [base = makeProvider(equalSlopeEstimate)](
                         const BatchExecutionPlanningContext& context) mutable {
      auto contributions = base(context);
      contributions.front().contribution.logical_maximum.value = context.logical_maximum.value + 1;
      return contributions;
    };
    CHECK_THROWS_AS(selectBatchExecutionPlan(input, excessive), std::invalid_argument);
  }

  SECTION("participant identity and order are invariant across candidates")
  {
    auto changing_identity = [base = makeProvider(equalSlopeEstimate)](
                                 const BatchExecutionPlanningContext& context) mutable {
      auto contributions = base(context);
      if (context.candidate_capacities.value > 1)
        contributions.front().participant_id = "participant/changed";
      return contributions;
    };
    CHECK_THROWS_AS(selectBatchExecutionPlan(input, changing_identity), std::logic_error);

    auto changing_order = [](const BatchExecutionPlanningContext& context) {
      BatchMemoryContribution contribution;
      contribution.logical_maximum    = context.logical_maximum;
      contribution.owner_multiplicity = 1;
      contribution.fully_accounted    = true;
      contribution.per_owner          = equalSlopeEstimate(context.candidate_capacities);
      if (context.candidate_capacities.value > 1)
        return std::vector<BatchMemoryParticipantContribution>{{"participant/b", contribution},
                                                                {"participant/a", contribution}};
      return std::vector<BatchMemoryParticipantContribution>{{"participant/a", contribution},
                                                              {"participant/b", contribution}};
    };
    CHECK_THROWS_AS(selectBatchExecutionPlan(input, changing_order), std::logic_error);
  }

  SECTION("participant maxima and owner multiplicity are invariant across candidates")
  {
    auto changing_maximum = [base = makeProvider(equalSlopeEstimate)](
                                const BatchExecutionPlanningContext& context) mutable {
      auto contributions = base(context);
      if (context.candidate_capacities.value > 1)
        --contributions.front().contribution.logical_maximum.value;
      return contributions;
    };
    CHECK_THROWS_AS(selectBatchExecutionPlan(input, changing_maximum), std::logic_error);

    auto changing_multiplicity = [base = makeProvider(equalSlopeEstimate)](
                                     const BatchExecutionPlanningContext& context) mutable {
      auto contributions = base(context);
      if (context.candidate_capacities.value > 1)
        contributions.front().contribution.owner_multiplicity = 2;
      return contributions;
    };
    CHECK_THROWS_AS(selectBatchExecutionPlan(input, changing_multiplicity), std::logic_error);
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
    CHECK_THROWS(selectBatchExecutionPlan(input, makeProvider(nonmonotone)));
  }
}

} // namespace qmcplusplus
