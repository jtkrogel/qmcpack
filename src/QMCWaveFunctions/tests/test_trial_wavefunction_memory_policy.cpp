//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_trial_wavefunction_memory_policy.cpp
 * @brief Unit tests for exact TrialWaveFunction aggregate memory accounting.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/TrialWaveFunctionMemoryPolicy.h"

#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus
{
namespace
{

/** Build exact widths from the real types used by the current executable. */
TrialWaveFunctionMemoryTypeSizes makeTypeSizes()
{
  return makeTrialWaveFunctionMemoryTypeSizes<
      TrialWaveFunction::ValueType,
      ParticleSet::ParticleGradient::value_type,
      ParticleSet::ParticleLaplacian::value_type,
      std::reference_wrapper<WaveFunctionComponent>,
      std::reference_wrapper<ParticleSet::ParticleGradient>,
      std::reference_wrapper<ParticleSet::ParticleLaplacian>,
      TrialWaveFunction::ParameterDerivativeView,
      TrialWaveFunction::EvaluationStamp,
      unsigned char>();
}

/** Return a fully covered pure-policy input without enabling production paths. */
TrialWaveFunctionMemoryPolicyInput makePolicyInput()
{
  TrialWaveFunctionMemoryPolicyInput input;
  input.type_sizes        = makeTypeSizes();
  input.accounting_claims = TrialWaveFunctionMemoryAccountingClaims::complete();
  input.sole_child        = {1, true, true};
  input.component_count   = 1;
  return input;
}

/** Assemble a planning context and derive its aggregate logical envelope. */
BatchExecutionPlanningContext makeContext(BatchExecutionRequirements requirements,
                                          std::vector<std::size_t> initial,
                                          std::vector<std::size_t> reserve,
                                          BatchTileCapacities candidate,
                                          std::size_t particle_count,
                                          std::size_t active_parameter_count = 0,
                                          std::size_t parameter_derivative_width = 0,
                                          BatchExecutionTargetCoordinate target_coordinate =
                                              BatchExecutionTargetCoordinate::POS_ONLY)
{
  BatchExecutionTopology topology;
  topology.initial_walkers_per_crowd = std::move(initial);
  topology.reserve_walkers_per_crowd = std::move(reserve);
  const BatchExecutionWorkloadContext workload{requirements, topology, particle_count, active_parameter_count,
                                                parameter_derivative_width, target_coordinate};
  return {requirements, topology, trialWaveFunctionBatchLogicalMaximum(workload), candidate, particle_count,
          active_parameter_count, parameter_derivative_width, target_coordinate};
}

/** Return one host category and assert that its device counterpart is empty. */
std::size_t hostBytes(const BatchMemoryContribution& contribution, BatchMemoryCategory category)
{
  CHECK(contribution.per_owner.at(category).device == 0);
  return contribution.per_owner.at(category).host;
}

/** Check that every field in a canonical empty resource descriptor is zero. */
void checkEmptyResource(const TrialWaveFunctionResourceStorageRequirement& storage)
{
  CHECK(storage.component_reference_slots == 0);
  CHECK(storage.aggregate_gradient_reference_slots == 0);
  CHECK(storage.aggregate_laplacian_reference_slots == 0);
  CHECK(storage.weighted_ecp_private_ratios == 0);
  CHECK(storage.weighted_ecp_total_weights == 0);
  CHECK(storage.weighted_parameter_derivative_delta == 0);
  CHECK(storage.weighted_parameter_derivative_views == 0);
  CHECK(storage.weighted_value_stamps == 0);
  CHECK(storage.transaction_flags == 0);
  CHECK(storage.totalBytes() == 0);
}

} // namespace

TEST_CASE("TrialWaveFunction aggregate policy accounts exact uneven crowd storage",
          "[wavefunction][trialwf][batch_memory]")
{
  TrialWaveFunctionMemoryPolicyInput input = makePolicyInput();
  const TrialWaveFunctionMemoryTypeSizes& sizes = input.type_sizes;
  CHECK(TRIAL_WAVEFUNCTION_MEMORY_PARTICIPANT_ID == "twf/aggregate");
  CHECK(sizes.value_type == sizeof(TrialWaveFunction::ValueType));
  CHECK(sizes.particle_gradient_element == sizeof(ParticleSet::ParticleGradient::value_type));
  CHECK(sizes.particle_laplacian_element == sizeof(ParticleSet::ParticleLaplacian::value_type));

  BatchExecutionRequirements requirements;
  requirements.require(BatchExecutionMode::VALUE);
  requirements.require(BatchExecutionMode::FULL_VGL);
  requirements.require(BatchExecutionMode::ACTIVE_GRADIENT);
  requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  requirements.require(BatchExecutionMode::SCORE);
  requirements.require(BatchExecutionMode::KINETIC);
  requirements.require(BatchExecutionMode::ECP_WEIGHTED_SCORE);
  const BatchExecutionPlanningContext context =
      makeContext(requirements, {2, 1, 1}, {3, 0, 1}, {3, 3, 3, 7}, 4, 2, 5);

  const std::vector<TrialWaveFunctionCrowdMemoryPlan> plans =
      makeTrialWaveFunctionCrowdMemoryPlans(input, context);
  REQUIRE(plans.size() == 3);
  CHECK(plans[0].initial_walkers == 2);
  CHECK(plans[0].reserve_walkers == 3);
  CHECK(plans[0].particle_count == 4);
  CHECK(plans[0].component_count == 1);
  CHECK(plans[0].ecp_outer_capacity == 7);
  CHECK(plans[0].parameter_derivative_width == 5);
  CHECK(plans[0].weighted_ecp_required);

  const auto expectedCloneBytes = [&](std::size_t reserve) {
    const std::size_t per_clone = 2 * 4 * sizes.particle_gradient_element +
        2 * 4 * sizes.particle_laplacian_element;
    return reserve * per_clone;
  };
  const auto expectedReferenceBytes = [&](std::size_t reserve) {
    return reserve * sizes.wavefunction_component_reference + reserve * sizes.particle_gradient_reference +
        reserve * sizes.particle_laplacian_reference;
  };
  const auto expectedPublicationBytes = [&](std::size_t reserve) {
    return reserve * 5 * sizes.value_type + reserve * sizes.parameter_derivative_view +
        reserve * sizes.transaction_flag;
  };
  const std::size_t expected_outer    = 2 * 7 * sizes.value_type;
  const std::size_t expected_metadata = sizes.evaluation_stamp;

  CHECK(plans[0].clone_storage.totalBytes() == expectedCloneBytes(3));
  CHECK(plans[0].resource_storage.logicalInputOutputBytes() == expectedReferenceBytes(3));
  CHECK(plans[0].resource_storage.outerTileScratchBytes() == expected_outer);
  CHECK(plans[0].resource_storage.publicationStagingBytes() == expectedPublicationBytes(3));
  CHECK(plans[0].resource_storage.ecpMetadataBytes() == expected_metadata);

  CHECK(plans[1].initial_walkers == 1);
  CHECK(plans[1].reserve_walkers == 0);
  CHECK(plans[1].clone_storage.totalBytes() == 0);
  checkEmptyResource(plans[1].resource_storage);
  CHECK(plans[1].expected_storage.total().host == 0);
  CHECK(plans[1].expected_storage.total().device == 0);

  CHECK(plans[2].clone_storage.totalBytes() == expectedCloneBytes(1));
  CHECK(plans[2].resource_storage.logicalInputOutputBytes() == expectedReferenceBytes(1));
  CHECK(plans[2].resource_storage.outerTileScratchBytes() == expected_outer);
  CHECK(plans[2].resource_storage.publicationStagingBytes() == expectedPublicationBytes(1));
  CHECK(plans[2].resource_storage.ecpMetadataBytes() == expected_metadata);

  const BatchMemoryContribution contribution = estimateTrialWaveFunctionBatchMemory(input, context);
  CHECK(contribution.owner_multiplicity == 1);
  CHECK(contribution.fully_accounted);
  CHECK(contribution.logical_maximum == BatchTileCapacities{5, 3, 3, 0});
  CHECK(hostBytes(contribution, BatchMemoryCategory::FIXED_CLONE_STATE) ==
        expectedCloneBytes(3) + expectedCloneBytes(1));
  CHECK(hostBytes(contribution, BatchMemoryCategory::LOGICAL_INPUT_OUTPUT) ==
        expectedReferenceBytes(3) + expectedReferenceBytes(1));
  CHECK(hostBytes(contribution, BatchMemoryCategory::OUTER_TILE_SCRATCH) == 2 * expected_outer);
  CHECK(hostBytes(contribution, BatchMemoryCategory::PUBLICATION_STAGING) ==
        expectedPublicationBytes(3) + expectedPublicationBytes(1));
  CHECK(hostBytes(contribution, BatchMemoryCategory::ECP_METADATA) == 2 * expected_metadata);

  // P is the global destination width.  The aggregate must not substitute the
  // smaller active-parameter count when it sizes the private derivative rows.
  CHECK(context.active_parameter_count == 2);
  CHECK(context.parameter_derivative_width == 5);
  CHECK(plans[0].resource_storage.weighted_parameter_derivative_delta == 3 * 5 * sizes.value_type);
}

TEST_CASE("TrialWaveFunction aggregate policy keeps non-weighted transactions bounded",
          "[wavefunction][trialwf][batch_memory]")
{
  const TrialWaveFunctionMemoryPolicyInput input = makePolicyInput();
  BatchExecutionRequirements requirements;
  requirements.require(BatchExecutionMode::VALUE);
  requirements.require(BatchExecutionMode::FULL_VGL);
  requirements.require(BatchExecutionMode::ECP_OUTER);
  const BatchExecutionPlanningContext context = makeContext(requirements, {2, 1}, {}, {2, 2, 0, 11}, 3);

  const std::vector<TrialWaveFunctionCrowdMemoryPlan> plans =
      makeTrialWaveFunctionCrowdMemoryPlans(input, context);
  REQUIRE(plans.size() == 2);
  CHECK(plans[0].reserve_walkers == 2);
  CHECK(plans[0].ecp_outer_capacity == 11);
  CHECK(plans[0].parameter_derivative_width == 0);
  CHECK_FALSE(plans[0].weighted_ecp_required);
  CHECK(plans[0].resource_storage.transaction_flags == 2 * input.type_sizes.transaction_flag);
  CHECK(plans[0].resource_storage.weighted_ecp_private_ratios == 0);
  CHECK(plans[0].resource_storage.weighted_ecp_total_weights == 0);
  CHECK(plans[0].resource_storage.weighted_parameter_derivative_delta == 0);
  CHECK(plans[0].resource_storage.weighted_parameter_derivative_views == 0);
  CHECK(plans[0].resource_storage.weighted_value_stamps == 0);

  const BatchMemoryContribution contribution = estimateTrialWaveFunctionBatchMemory(input, context);
  CHECK(contribution.fully_accounted);
  CHECK(contribution.logical_maximum == BatchTileCapacities{2, 2, 0, 0});
  CHECK(hostBytes(contribution, BatchMemoryCategory::OUTER_TILE_SCRATCH) == 0);
  CHECK(hostBytes(contribution, BatchMemoryCategory::ECP_METADATA) == 0);
  CHECK(hostBytes(contribution, BatchMemoryCategory::PUBLICATION_STAGING) ==
        3 * input.type_sizes.transaction_flag);
}

TEST_CASE("TrialWaveFunction aggregate completeness is mode-conditional and fail-closed",
          "[wavefunction][trialwf][batch_memory]")
{
  BatchExecutionRequirements base_requirements;
  base_requirements.require(BatchExecutionMode::VALUE);
  const BatchExecutionPlanningContext base_context =
      makeContext(base_requirements, {2}, {3}, {2, 0, 0, 0}, 4);
  TrialWaveFunctionMemoryPolicyInput input = makePolicyInput();
  CHECK(estimateTrialWaveFunctionBatchMemory(input, base_context).fully_accounted);

  BatchExecutionPlanningContext unknown_target = base_context;
  unknown_target.target_coordinate = BatchExecutionTargetCoordinate::UNKNOWN;
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, unknown_target).fully_accounted);
  BatchExecutionPlanningContext spin_target = base_context;
  spin_target.target_coordinate = BatchExecutionTargetCoordinate::POS_SPIN;
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, spin_target).fully_accounted);

  for (const BatchExecutionTargetCoordinate unsupported_target :
       {BatchExecutionTargetCoordinate::UNKNOWN, BatchExecutionTargetCoordinate::POS_SPIN})
  {
    BatchExecutionSelectionInput selection;
    selection.requirements                       = base_requirements;
    selection.topology                           = base_context.topology;
    selection.logical_maximum                    = base_context.logical_maximum;
    selection.preference.preferred               = {2, 0, 0, 0};
    selection.particle_count                     = base_context.particle_count;
    selection.target_coordinate                  = unsupported_target;
    CHECK_THROWS_WITH(
        selectBatchExecutionPlan(
            selection, [&input](const BatchExecutionPlanningContext& candidate) {
              return std::vector<BatchMemoryParticipantContribution>{
                  {std::string(TRIAL_WAVEFUNCTION_MEMORY_PARTICIPANT_ID),
                   estimateTrialWaveFunctionBatchMemory(input, candidate)}};
            }),
        "Batch memory participant is not fully accounted: twf/aggregate");
  }

  BatchExecutionPlanningContext value_only_global_width = base_context;
  value_only_global_width.active_parameter_count        = 2;
  value_only_global_width.parameter_derivative_width    = 7;
  CHECK(estimateTrialWaveFunctionBatchMemory(input, value_only_global_width).fully_accounted);

  TrialWaveFunctionMemoryPolicyInput unclaimed;
  unclaimed.type_sizes      = makeTypeSizes();
  unclaimed.sole_child      = {1, true, true};
  unclaimed.component_count = 1;
  const BatchMemoryContribution unclaimed_contribution =
      estimateTrialWaveFunctionBatchMemory(unclaimed, base_context);
  CHECK_FALSE(unclaimed_contribution.fully_accounted);
  CHECK_THROWS(aggregateBatchMemoryContributions(
      {{std::string(TRIAL_WAVEFUNCTION_MEMORY_PARTICIPANT_ID), unclaimed_contribution}}));

  input.accounting_claims.runtime_preflight_and_unsupported_fail_closed = false;
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, base_context).fully_accounted);
  input = makePolicyInput();
  input.accounting_claims.clone_state = false;
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, base_context).fully_accounted);
  input = makePolicyInput();
  input.accounting_claims.reference_views = false;
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, base_context).fully_accounted);
  input = makePolicyInput();
  input.accounting_claims.sole_component_dispatch = false;
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, base_context).fully_accounted);

  using UnsupportedConfiguration = void (*)(TrialWaveFunctionMemoryPolicyInput&);
  for (const UnsupportedConfiguration configure_unsupported : {
           +[](TrialWaveFunctionMemoryPolicyInput& policy) { policy.component_count = 2; },
           +[](TrialWaveFunctionMemoryPolicyInput& policy) { policy.use_tasking = true; },
           +[](TrialWaveFunctionMemoryPolicyInput& policy) { policy.fallback_path_reachable = true; },
           +[](TrialWaveFunctionMemoryPolicyInput& policy) { policy.sole_child.owner_multiplicity = 2; },
           +[](TrialWaveFunctionMemoryPolicyInput& policy) { policy.sole_child.fully_accounted = false; },
           +[](TrialWaveFunctionMemoryPolicyInput& policy) { policy.sole_child.atomic_publication = false; }})
  {
    input = makePolicyInput();
    configure_unsupported(input);
    CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, base_context).fully_accounted);
  }

  input = makePolicyInput();
  BatchExecutionPlanningContext serialized = base_context;
  serialized.topology.serialized_walkers   = true;
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, serialized).fully_accounted);
  BatchExecutionPlanningContext missing_shape = base_context;
  missing_shape.particle_count                 = 0;
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, missing_shape).fully_accounted);

  BatchExecutionRequirements scalar_requirements = base_requirements;
  scalar_requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  const BatchExecutionPlanningContext scalar_context =
      makeContext(scalar_requirements, {1}, {1}, {1, 0, 0, 0}, 4);
  input.accounting_claims.scalar_value_forwarding = false;
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, scalar_context).fully_accounted);

  input = makePolicyInput();
  BatchExecutionRequirements full_requirements;
  full_requirements.require(BatchExecutionMode::FULL_VGL);
  const BatchExecutionPlanningContext full_context =
      makeContext(full_requirements, {1}, {1}, {0, 1, 0, 0}, 4);
  input.accounting_claims.full_vgl_and_selected_move_transaction = false;
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, full_context).fully_accounted);

  input = makePolicyInput();
  BatchExecutionRequirements active_requirements;
  active_requirements.require(BatchExecutionMode::ACTIVE_GRADIENT);
  const BatchExecutionPlanningContext active_context =
      makeContext(active_requirements, {1}, {1}, {0, 0, 1, 0}, 4);
  input.accounting_claims.active_gradient_transaction = false;
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, active_context).fully_accounted);
}

TEST_CASE("TrialWaveFunction aggregate policy separates active count from derivative width",
          "[wavefunction][trialwf][batch_memory]")
{
  TrialWaveFunctionMemoryPolicyInput input = makePolicyInput();
  for (const BatchExecutionMode mode :
       {BatchExecutionMode::SCORE, BatchExecutionMode::KINETIC, BatchExecutionMode::ECP_WEIGHTED_SCORE})
  {
    BatchExecutionRequirements requirements;
    requirements.require(mode);
    if (mode == BatchExecutionMode::ECP_WEIGHTED_SCORE)
      requirements.require(BatchExecutionMode::VALUE);
    const BatchTileCapacities candidate = mode == BatchExecutionMode::ECP_WEIGHTED_SCORE
        ? BatchTileCapacities{2, 0, 0, 3}
        : BatchTileCapacities{};
    const BatchExecutionPlanningContext context = makeContext(requirements, {2}, {2}, candidate, 4, 2, 9);
    CHECK(estimateTrialWaveFunctionBatchMemory(input, context).fully_accounted);

    BatchExecutionPlanningContext missing_width = context;
    missing_width.parameter_derivative_width    = 0;
    CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, missing_width).fully_accounted);

    BatchExecutionPlanningContext too_narrow = context;
    too_narrow.parameter_derivative_width    = 1;
    CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, too_narrow).fully_accounted);

    BatchExecutionPlanningContext inactive = context;
    inactive.active_parameter_count        = 0;
    CHECK(estimateTrialWaveFunctionBatchMemory(input, inactive).fully_accounted);
    inactive.parameter_derivative_width = 0;
    CHECK(estimateTrialWaveFunctionBatchMemory(input, inactive).fully_accounted);

    input.accounting_claims.parameter_derivative_forwarding = false;
    CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, context).fully_accounted);
    input = makePolicyInput();
  }

  BatchExecutionRequirements weighted_requirements;
  weighted_requirements.require(BatchExecutionMode::VALUE);
  weighted_requirements.require(BatchExecutionMode::ECP_WEIGHTED_SCORE);
  const BatchExecutionPlanningContext weighted_context =
      makeContext(weighted_requirements, {2}, {2}, {2, 0, 0, 3}, 4, 2, 9);
  input.accounting_claims.weighted_ecp_outer_scratch = false;
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, weighted_context).fully_accounted);
  input = makePolicyInput();
  input.accounting_claims.weighted_ecp_derivative_staging = false;
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, weighted_context).fully_accounted);
  input = makePolicyInput();
  input.accounting_claims.weighted_ecp_metadata = false;
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, weighted_context).fully_accounted);

  input = makePolicyInput();
  BatchExecutionPlanningContext missing_outer = weighted_context;
  missing_outer.candidate_capacities.ecp_outer = 0;
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, missing_outer).fully_accounted);

  BatchExecutionRequirements missing_value_requirements;
  missing_value_requirements.require(BatchExecutionMode::ECP_WEIGHTED_SCORE);
  const BatchExecutionPlanningContext missing_value =
      makeContext(missing_value_requirements, {2}, {2}, {0, 0, 0, 3}, 4, 2, 9);
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, missing_value).fully_accounted);

  BatchExecutionRequirements unweighted_missing_value_requirements;
  unweighted_missing_value_requirements.require(BatchExecutionMode::ECP_OUTER);
  const BatchExecutionPlanningContext unweighted_missing_value =
      makeContext(unweighted_missing_value_requirements, {2}, {2}, {0, 0, 0, 3}, 4);
  CHECK_FALSE(estimateTrialWaveFunctionBatchMemory(input, unweighted_missing_value).fully_accounted);
}

TEST_CASE("TrialWaveFunction aggregate policy checks every exact extent",
          "[wavefunction][trialwf][batch_memory]")
{
  const TrialWaveFunctionMemoryTypeSizes sizes = makeTypeSizes();
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(trialWaveFunctionCloneStorageRequirement(maximum, 2, sizes), std::overflow_error);
  CHECK_THROWS_AS(trialWaveFunctionResourceStorageRequirement(maximum, 2, 1, 1, false, sizes),
                  std::overflow_error);
  CHECK_THROWS_AS(trialWaveFunctionResourceStorageRequirement(2, 1, 1, maximum, true, sizes),
                  std::overflow_error);
  CHECK_THROWS_AS(trialWaveFunctionResourceStorageRequirement(1, 1, maximum, 1, true, sizes),
                  std::overflow_error);

  TrialWaveFunctionMemoryTypeSizes missing_width = sizes;
  missing_width.particle_laplacian_element       = 0;
  CHECK_THROWS_AS(trialWaveFunctionCloneStorageRequirement(1, 1, missing_width), std::invalid_argument);
  CHECK_THROWS_AS(trialWaveFunctionResourceStorageRequirement(1, 1, 1, 1, true, missing_width),
                  std::invalid_argument);

  BatchExecutionRequirements scalar_requirements;
  scalar_requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  BatchExecutionTopology topology;
  topology.initial_walkers_per_crowd = {0};
  const BatchExecutionWorkloadContext overflow_workload{scalar_requirements, topology, maximum, 0, 0};
  CHECK_THROWS_AS(trialWaveFunctionBatchLogicalMaximum(overflow_workload), std::overflow_error);

  BatchExecutionTopology mismatched;
  mismatched.initial_walkers_per_crowd = {1};
  mismatched.reserve_walkers_per_crowd = {1, 2};
  CHECK_THROWS_AS(trialWaveFunctionReserveWalkersPerCrowd(mismatched), std::invalid_argument);
}

TEST_CASE("TrialWaveFunction aggregate zero reserves retain provenance without storage",
          "[wavefunction][trialwf][batch_memory]")
{
  const TrialWaveFunctionMemoryPolicyInput input = makePolicyInput();
  BatchExecutionRequirements requirements;
  requirements.require(BatchExecutionMode::VALUE);
  requirements.require(BatchExecutionMode::ECP_WEIGHTED_SCORE);
  const BatchExecutionPlanningContext context = makeContext(requirements, {0, 0}, {0, 0}, {1, 0, 0, 8}, 3, 4, 7);

  const std::vector<TrialWaveFunctionCrowdMemoryPlan> plans =
      makeTrialWaveFunctionCrowdMemoryPlans(input, context);
  REQUIRE(plans.size() == 2);
  for (const TrialWaveFunctionCrowdMemoryPlan& plan : plans)
  {
    CHECK(plan.reserve_walkers == 0);
    CHECK(plan.clone_storage.totalBytes() == 0);
    checkEmptyResource(plan.resource_storage);
    CHECK(plan.expected_storage.total().host == 0);
    CHECK(plan.expected_storage.total().device == 0);
  }

  const BatchMemoryContribution contribution = estimateTrialWaveFunctionBatchMemory(input, context);
  CHECK(contribution.owner_multiplicity == 1);
  CHECK(contribution.fully_accounted);
  CHECK(contribution.per_owner.total().host == 0);
  CHECK(contribution.per_owner.total().device == 0);
}

} // namespace qmcplusplus
