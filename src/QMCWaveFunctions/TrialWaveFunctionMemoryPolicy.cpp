//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file TrialWaveFunctionMemoryPolicy.cpp
 * @brief Pure rank-local memory accounting for TrialWaveFunction aggregate state.
 */

#include "QMCWaveFunctions/TrialWaveFunctionMemoryPolicy.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace qmcplusplus
{
namespace
{

/** Reject incomplete type descriptions before doing any byte arithmetic. */
void validateTypeSizes(const TrialWaveFunctionMemoryTypeSizes& sizes)
{
  if (sizes.value_type == 0 || sizes.particle_gradient_element == 0 ||
      sizes.particle_laplacian_element == 0 || sizes.wavefunction_component_reference == 0 ||
      sizes.particle_gradient_reference == 0 || sizes.particle_laplacian_reference == 0 ||
      sizes.parameter_derivative_view == 0 || sizes.evaluation_stamp == 0 || sizes.transaction_flag == 0)
    throw std::invalid_argument("TrialWaveFunction memory accounting requires every type width to be positive");
}

/** Add one host-only byte category to an exact estimate. */
void addHostBytes(BatchMemoryEstimate& estimate,
                  BatchMemoryCategory category,
                  std::size_t bytes,
                  const std::string& context)
{
  estimate.add(category, {bytes, 0}, context);
}

/** Report whether weighted flattened-ECP derivatives require aggregate staging. */
bool weightedEcpRequired(const BatchExecutionRequirements& requirements) noexcept
{
  return requirements.requires(BatchExecutionMode::ECP_WEIGHTED_SCORE);
}

/** Decide whether every reachable aggregate path is explicitly covered. */
bool accountingIsComplete(const TrialWaveFunctionMemoryPolicyInput& input,
                          const BatchExecutionPlanningContext& context,
                          const std::vector<std::size_t>& reserves)
{
  const bool weighted_ecp = weightedEcpRequired(context.requirements);
  const bool scalar_value = context.requirements.requires(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  const bool full_vgl = context.requirements.requires(BatchExecutionMode::FULL_VGL);
  const bool active_gradient = context.requirements.requires(BatchExecutionMode::ACTIVE_GRADIENT);
  const bool parameter_derivative = context.requirements.requires(BatchExecutionMode::SCORE) ||
      context.requirements.requires(BatchExecutionMode::KINETIC) || weighted_ecp;
  const bool flattened_ecp =
      batchExecutionModeIsRequired(context.requirements, BatchExecutionMode::ECP_OUTER);
  const bool derivative_storage_required = parameter_derivative && context.active_parameter_count != 0;

  bool complete = !context.topology.serialized_walkers && input.component_count == 1 && !input.use_tasking &&
      !input.spinor_path_reachable && !input.fallback_path_reachable &&
      input.accounting_claims.runtime_preflight_and_unsupported_fail_closed && input.accounting_claims.clone_state &&
      input.accounting_claims.reference_views && input.accounting_claims.sole_component_dispatch &&
      input.sole_child.owner_multiplicity == 1 && input.sole_child.fully_accounted &&
      input.sole_child.atomic_publication;

  complete = complete && context.particle_count != 0;
  complete = complete && (!flattened_ecp || context.requirements.requires(BatchExecutionMode::VALUE));

  // A known global width may remain in a value-only plan.  With one admitted
  // component, every reachable derivative path must at least hold all active
  // entries; sparse global mappings may make the destination wider, never
  // narrower.
  complete = complete &&
      (!derivative_storage_required || context.parameter_derivative_width >= context.active_parameter_count);

  if (scalar_value)
    complete = complete && input.accounting_claims.scalar_value_forwarding;
  if (full_vgl)
    complete = complete && input.accounting_claims.full_vgl_and_selected_move_transaction;
  if (active_gradient)
    complete = complete && input.accounting_claims.active_gradient_transaction;
  if (derivative_storage_required)
    complete = complete && input.accounting_claims.parameter_derivative_forwarding;

  if (weighted_ecp)
    complete = complete && context.candidate_capacities.ecp_outer != 0 &&
        input.accounting_claims.weighted_ecp_outer_scratch &&
        input.accounting_claims.weighted_ecp_derivative_staging && input.accounting_claims.weighted_ecp_metadata;

  // Keep topology validation tied to the same reserve view used by planning.
  complete = complete && reserves.size() == context.topology.initial_walkers_per_crowd.size();
  return complete;
}

} // namespace

std::size_t TrialWaveFunctionCloneStorageRequirement::totalBytes() const
{
  const std::size_t gradients = checkedBatchMemoryAdd(accepted_gradients, proposed_gradients,
                                                       "TWF aggregate clone gradients");
  const std::size_t laplacians = checkedBatchMemoryAdd(accepted_laplacians, proposed_laplacians,
                                                        "TWF aggregate clone laplacians");
  return checkedBatchMemoryAdd(gradients, laplacians, "TWF aggregate clone storage");
}

std::size_t TrialWaveFunctionResourceStorageRequirement::logicalInputOutputBytes() const
{
  const std::size_t component_and_gradient = checkedBatchMemoryAdd(
      component_reference_slots, aggregate_gradient_reference_slots, "TWF aggregate component/gradient references");
  return checkedBatchMemoryAdd(component_and_gradient, aggregate_laplacian_reference_slots,
                               "TWF aggregate reference views");
}

std::size_t TrialWaveFunctionResourceStorageRequirement::outerTileScratchBytes() const
{
  return checkedBatchMemoryAdd(weighted_ecp_private_ratios, weighted_ecp_total_weights,
                               "TWF weighted-ECP outer scratch");
}

std::size_t TrialWaveFunctionResourceStorageRequirement::publicationStagingBytes() const
{
  const std::size_t derivative_staging = checkedBatchMemoryAdd(
      weighted_parameter_derivative_delta, weighted_parameter_derivative_views,
      "TWF weighted-ECP derivative publication");
  return checkedBatchMemoryAdd(derivative_staging, transaction_flags, "TWF aggregate transaction publication");
}

std::size_t TrialWaveFunctionResourceStorageRequirement::ecpMetadataBytes() const
{
  return weighted_value_stamps;
}

std::size_t TrialWaveFunctionResourceStorageRequirement::totalBytes() const
{
  std::size_t bytes = logicalInputOutputBytes();
  bytes             = checkedBatchMemoryAdd(bytes, outerTileScratchBytes(), "TWF aggregate resource storage");
  bytes             = checkedBatchMemoryAdd(bytes, publicationStagingBytes(), "TWF aggregate resource storage");
  return checkedBatchMemoryAdd(bytes, ecpMetadataBytes(), "TWF aggregate resource storage");
}

const std::vector<std::size_t>& trialWaveFunctionReserveWalkersPerCrowd(const BatchExecutionTopology& topology)
{
  validateBatchExecutionTopology(topology);
  return topology.reserve_walkers_per_crowd.empty() ? topology.initial_walkers_per_crowd
                                                     : topology.reserve_walkers_per_crowd;
}

BatchTileCapacities trialWaveFunctionBatchLogicalMaximum(const BatchExecutionWorkloadContext& context)
{
  const std::vector<std::size_t>& reserves = trialWaveFunctionReserveWalkersPerCrowd(context.topology);
  std::size_t maximum_reserve              = 0;
  for (const std::size_t reserve : reserves)
    maximum_reserve = std::max(maximum_reserve, reserve);

  BatchTileCapacities maximum;
  if (context.requirements.requires(BatchExecutionMode::VALUE))
    maximum.value = maximum_reserve;
  if (context.requirements.requires(BatchExecutionMode::FULL_VGL))
    maximum.full_vgl = maximum_reserve;
  if (context.requirements.requires(BatchExecutionMode::ACTIVE_GRADIENT))
    maximum.active_gradient = maximum_reserve;
  if (context.requirements.requires(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY))
  {
    const std::size_t scalar_extent =
        checkedBatchMemoryAdd(context.particle_count, 1, "TWF scalar VALUE logical maximum");
    maximum.value = std::max(maximum.value, scalar_extent);
  }

  // The nonlocal-pseudopotential participant supplies the flattened outer envelope.
  maximum.ecp_outer = 0;
  return maximum;
}

TrialWaveFunctionCloneStorageRequirement trialWaveFunctionCloneStorageRequirement(
    std::size_t reserve_walkers,
    std::size_t particle_count,
    const TrialWaveFunctionMemoryTypeSizes& type_sizes)
{
  validateTypeSizes(type_sizes);
  TrialWaveFunctionCloneStorageRequirement storage;
  if (reserve_walkers == 0)
    return storage;

  const std::size_t resident_particles = checkedBatchMemoryMultiply(
      reserve_walkers, particle_count, "TWF aggregate resident-particle count");
  const std::size_t gradient_bytes = checkedBatchMemoryMultiply(
      resident_particles, type_sizes.particle_gradient_element, "TWF aggregate clone gradient storage");
  const std::size_t laplacian_bytes = checkedBatchMemoryMultiply(
      resident_particles, type_sizes.particle_laplacian_element, "TWF aggregate clone laplacian storage");

  storage.accepted_gradients  = gradient_bytes;
  storage.proposed_gradients  = gradient_bytes;
  storage.accepted_laplacians = laplacian_bytes;
  storage.proposed_laplacians = laplacian_bytes;
  (void)storage.totalBytes();
  return storage;
}

TrialWaveFunctionResourceStorageRequirement trialWaveFunctionResourceStorageRequirement(
    std::size_t reserve_walkers,
    std::size_t component_count,
    std::size_t ecp_outer_capacity,
    std::size_t parameter_derivative_width,
    bool weighted_ecp_required,
    const TrialWaveFunctionMemoryTypeSizes& type_sizes)
{
  validateTypeSizes(type_sizes);
  TrialWaveFunctionResourceStorageRequirement storage;
  if (reserve_walkers == 0)
    return storage;

  const std::size_t component_slots = checkedBatchMemoryMultiply(
      reserve_walkers, component_count, "TWF aggregate component-reference count");
  storage.component_reference_slots = checkedBatchMemoryMultiply(
      component_slots, type_sizes.wavefunction_component_reference, "TWF aggregate component references");
  storage.aggregate_gradient_reference_slots = checkedBatchMemoryMultiply(
      reserve_walkers, type_sizes.particle_gradient_reference, "TWF aggregate gradient references");
  storage.aggregate_laplacian_reference_slots = checkedBatchMemoryMultiply(
      reserve_walkers, type_sizes.particle_laplacian_reference, "TWF aggregate laplacian references");
  storage.transaction_flags = checkedBatchMemoryMultiply(
      reserve_walkers, type_sizes.transaction_flag, "TWF aggregate transaction flags");

  if (weighted_ecp_required)
  {
    storage.weighted_ecp_private_ratios = checkedBatchMemoryMultiply(
        ecp_outer_capacity, type_sizes.value_type, "TWF weighted-ECP private ratios");
    storage.weighted_ecp_total_weights = checkedBatchMemoryMultiply(
        ecp_outer_capacity, type_sizes.value_type, "TWF weighted-ECP total weights");

    const std::size_t derivative_elements = checkedBatchMemoryMultiply(
        reserve_walkers, parameter_derivative_width, "TWF weighted-ECP derivative elements");
    storage.weighted_parameter_derivative_delta = checkedBatchMemoryMultiply(
        derivative_elements, type_sizes.value_type, "TWF weighted-ECP derivative delta");
    storage.weighted_parameter_derivative_views = checkedBatchMemoryMultiply(
        reserve_walkers, type_sizes.parameter_derivative_view, "TWF weighted-ECP derivative views");
    storage.weighted_value_stamps = checkedBatchMemoryMultiply(
        component_count, type_sizes.evaluation_stamp, "TWF weighted-ECP value stamps");
  }

  (void)storage.totalBytes();
  return storage;
}

std::vector<TrialWaveFunctionCrowdMemoryPlan> makeTrialWaveFunctionCrowdMemoryPlans(
    const TrialWaveFunctionMemoryPolicyInput& input,
    const BatchExecutionPlanningContext& context)
{
  validateTypeSizes(input.type_sizes);
  const std::vector<std::size_t>& reserves = trialWaveFunctionReserveWalkersPerCrowd(context.topology);
  const std::vector<std::size_t>& initial  = context.topology.initial_walkers_per_crowd;
  const bool weighted_ecp                 = weightedEcpRequired(context.requirements);

  std::vector<TrialWaveFunctionCrowdMemoryPlan> plans;
  plans.reserve(reserves.size());
  for (std::size_t crowd_index = 0; crowd_index < reserves.size(); ++crowd_index)
  {
    TrialWaveFunctionCrowdMemoryPlan plan;
    plan.initial_walkers            = initial[crowd_index];
    plan.reserve_walkers            = reserves[crowd_index];
    plan.particle_count             = context.particle_count;
    plan.component_count            = input.component_count;
    plan.ecp_outer_capacity         = context.candidate_capacities.ecp_outer;
    plan.parameter_derivative_width = context.parameter_derivative_width;
    plan.weighted_ecp_required      = weighted_ecp;
    plan.clone_storage = trialWaveFunctionCloneStorageRequirement(plan.reserve_walkers, plan.particle_count,
                                                                   input.type_sizes);
    plan.resource_storage = trialWaveFunctionResourceStorageRequirement(
        plan.reserve_walkers, plan.component_count, plan.ecp_outer_capacity, plan.parameter_derivative_width,
        plan.weighted_ecp_required, input.type_sizes);

    addHostBytes(plan.expected_clone_storage, BatchMemoryCategory::FIXED_CLONE_STATE,
                 plan.clone_storage.totalBytes(), "TWF aggregate crowd clone storage");
    addHostBytes(plan.expected_resource_storage, BatchMemoryCategory::LOGICAL_INPUT_OUTPUT,
                 plan.resource_storage.logicalInputOutputBytes(), "TWF aggregate crowd reference storage");
    addHostBytes(plan.expected_resource_storage, BatchMemoryCategory::OUTER_TILE_SCRATCH,
                 plan.resource_storage.outerTileScratchBytes(), "TWF weighted-ECP outer scratch");
    addHostBytes(plan.expected_resource_storage, BatchMemoryCategory::PUBLICATION_STAGING,
                 plan.resource_storage.publicationStagingBytes(), "TWF weighted-ECP derivative publication");
    addHostBytes(plan.expected_resource_storage, BatchMemoryCategory::ECP_METADATA,
                 plan.resource_storage.ecpMetadataBytes(), "TWF weighted-ECP transaction metadata");
    plan.expected_storage.add(plan.expected_clone_storage, "TWF aggregate crowd clone total");
    plan.expected_storage.add(plan.expected_resource_storage, "TWF aggregate crowd resource total");
    plans.push_back(std::move(plan));
  }
  return plans;
}

BatchMemoryContribution estimateTrialWaveFunctionBatchMemory(const TrialWaveFunctionMemoryPolicyInput& input,
                                                             const BatchExecutionPlanningContext& context)
{
  const std::vector<std::size_t>& reserves = trialWaveFunctionReserveWalkersPerCrowd(context.topology);
  const std::vector<TrialWaveFunctionCrowdMemoryPlan> plans = makeTrialWaveFunctionCrowdMemoryPlans(input, context);

  BatchMemoryContribution contribution;
  contribution.logical_maximum = trialWaveFunctionBatchLogicalMaximum(
      {context.requirements, context.topology, context.particle_count, context.active_parameter_count,
       context.parameter_derivative_width});
  contribution.owner_multiplicity = 1;
  contribution.fully_accounted    = accountingIsComplete(input, context, reserves);
  for (const TrialWaveFunctionCrowdMemoryPlan& plan : plans)
    contribution.per_owner.add(plan.expected_storage, "TWF aggregate rank storage");
  return contribution;
}

} // namespace qmcplusplus
