//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerMemoryPolicy.cpp
 * @brief Pure rank-local batch-memory accounting for one PsiFormer component family.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerMemoryPolicy.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace qmcplusplus::psiformer
{
namespace
{

bool directBackend(PsiFormerMemoryBackend backend) noexcept
{
  return backend == PsiFormerMemoryBackend::DIRECT;
}

bool flattenedEcpRequired(const BatchExecutionRequirements& requirements) noexcept
{
  return batchExecutionModeIsRequired(requirements, BatchExecutionMode::ECP_OUTER);
}

bool scoreTapeRequired(const BatchExecutionRequirements& requirements) noexcept
{
  return requirements.requires(BatchExecutionMode::SCORE) ||
      requirements.requires(BatchExecutionMode::ECP_WEIGHTED_SCORE);
}

void validateModeStructure(const BatchExecutionRequirements& requirements)
{
  if (flattenedEcpRequired(requirements) &&
      !requirements.requires(BatchExecutionMode::VALUE))
    throw std::invalid_argument(
        "PsiFormer flattened ECP execution requires the VALUE mode");
}

void addHostBytes(BatchMemoryEstimate& estimate,
                  BatchMemoryCategory category,
                  std::size_t bytes,
                  const std::string& context)
{
  estimate.add(category, {bytes, 0}, context);
}

void addDirectStorage(BatchMemoryEstimate& estimate,
                      const pf::DirectBatchStorageRequirement& storage,
                      std::size_t multiplicity,
                      const std::string& context)
{
  const std::size_t logical = checkedBatchMemoryAdd(
      checkedBatchMemoryAdd(storage.dense_logical, storage.sparse_logical,
                            context + " logical input"),
      storage.logical_outputs, context + " logical output");
  std::size_t inner_tile = 0;
  for (const std::size_t bytes : {
           storage.sparse_tile_positions, storage.value_tile,
           storage.full_vgl_tile, storage.active_gradient_tile,
           storage.shared_spatial_arena})
    inner_tile = checkedBatchMemoryAdd(inner_tile, bytes,
                                       context + " inner tile");

  addHostBytes(estimate, BatchMemoryCategory::LOGICAL_INPUT_OUTPUT,
               checkedBatchMemoryMultiply(logical, multiplicity,
                                          context + " logical multiplicity"),
               context + " logical storage");
  addHostBytes(estimate, BatchMemoryCategory::INNER_TILE_SCRATCH,
               checkedBatchMemoryMultiply(inner_tile, multiplicity,
                                          context + " tile multiplicity"),
               context + " tile storage");
  addHostBytes(
      estimate, BatchMemoryCategory::REALLOCATION_TRANSIENT,
      checkedBatchMemoryMultiply(storage.replacementTransientBytes(), multiplicity,
                                 context + " transient multiplicity"),
      context + " replacement transient");
}

bool accountingIsComplete(const PsiFormerMemoryPolicyInput& input,
                          const BatchExecutionPlanningContext& context)
{
  const BatchExecutionRequirements& requirements = context.requirements;
  const bool value = requirements.requires(BatchExecutionMode::VALUE);
  const bool full = requirements.requires(BatchExecutionMode::FULL_VGL);
  const bool active = requirements.requires(BatchExecutionMode::ACTIVE_GRADIENT);
  const bool scalar =
      requirements.requires(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  const bool ecp = flattenedEcpRequired(requirements);
  const bool score = scoreTapeRequired(requirements) && input.active_parameter_count != 0;
  const bool kinetic = requirements.requires(BatchExecutionMode::KINETIC) &&
      input.active_parameter_count != 0;
  const bool resource_direct = value || full || active || ecp;
  const bool publication = resource_direct || score || kinetic;

  bool complete = context.target_coordinate == BatchExecutionTargetCoordinate::POS_ONLY &&
      input.accounting_claims.clone_state &&
      !context.topology.serialized_walkers;
  if (resource_direct)
    complete = complete && input.accounting_claims.direct_batch;
  if (publication)
    complete = complete && input.accounting_claims.publication_staging;
  if (score)
    complete = complete && input.accounting_claims.score_tape;
  if (kinetic)
    complete = complete && input.accounting_claims.kinetic_tape;
  if (scalar)
    complete = complete && input.accounting_claims.scalar_value_compatibility;
  if (ecp)
    complete = complete && input.flattened_ecp &&
        input.accounting_claims.flattened_ecp;

  if ((value || scalar || ecp) && !directBackend(input.backends.value))
    complete = false;
  if ((full || active) && !directBackend(input.backends.spatial))
    complete = false;
  if (score && !directBackend(input.backends.score))
    complete = false;
  if (kinetic && !directBackend(input.backends.kinetic))
    complete = false;
  return complete;
}

} // namespace

const std::vector<std::size_t>& psiFormerReserveWalkersPerCrowd(
    const BatchExecutionTopology& topology)
{
  validateBatchExecutionTopology(topology);
  return topology.reserve_walkers_per_crowd.empty()
      ? topology.initial_walkers_per_crowd
      : topology.reserve_walkers_per_crowd;
}

PsiFormerMemoryTopologySummary summarizePsiFormerMemoryTopology(
    const BatchExecutionTopology& topology)
{
  PsiFormerMemoryTopologySummary result;
  for (const std::size_t reserve : psiFormerReserveWalkersPerCrowd(topology))
  {
    result.resident_walkers = checkedBatchMemoryAdd(
        result.resident_walkers, reserve,
        "PsiFormer resident walker count");
    if (reserve != 0)
      result.prepared_crowds = checkedBatchMemoryAdd(
          result.prepared_crowds, 1,
          "PsiFormer prepared crowd count");
  }
  return result;
}

BatchTileCapacities psiFormerBatchLogicalMaximum(
    const PsiFormerMemoryPolicyInput& input,
    const BatchExecutionWorkloadContext& context)
{
  validateModeStructure(context.requirements);
  const std::vector<std::size_t>& reserves =
      psiFormerReserveWalkersPerCrowd(context.topology);
  const std::size_t maximum_reserve = reserves.empty()
      ? 0
      : *std::max_element(reserves.begin(), reserves.end());

  BatchTileCapacities maximum;
  if (context.requirements.requires(BatchExecutionMode::VALUE))
    maximum.value = maximum_reserve;
  if (context.requirements.requires(BatchExecutionMode::FULL_VGL))
    maximum.full_vgl = maximum_reserve;
  if (context.requirements.requires(BatchExecutionMode::ACTIVE_GRADIENT))
    maximum.active_gradient = maximum_reserve;
  if (context.requirements.requires(
          BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY))
  {
    if (input.scalar_value_logical_maximum == 0)
      throw std::invalid_argument(
          "PsiFormer scalar VALUE compatibility requires a finite logical envelope");
    maximum.value = std::max(maximum.value,
                             input.scalar_value_logical_maximum);
  }

  // ECP_OUTER belongs to the nonlocal-pseudopotential participant.  PsiFormer
  // consumes the selected capacity but must not guess or inflate its maximum.
  maximum.ecp_outer = 0;
  return maximum;
}

pf::DirectBatchCapacityPlan makePsiFormerScalarValueCapacityPlan(
    const PsiFormerMemoryPolicyInput& input,
    const BatchExecutionRequirements& requirements,
    const BatchTileCapacities& selected_capacities)
{
  pf::DirectBatchCapacityPlan plan;
  plan.tile = {0, 0, 0};
  if (!requirements.requires(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY))
    return plan;

  if (input.scalar_value_logical_maximum == 0)
    throw std::invalid_argument(
        "PsiFormer scalar VALUE compatibility requires a finite logical envelope");
  if (selected_capacities.value == 0)
    throw std::invalid_argument(
        "PsiFormer scalar VALUE compatibility requires a positive VALUE tile");

  plan.logical.value_dense = input.scalar_value_logical_maximum;
  plan.tile.value           = selected_capacities.value;
  return plan;
}

pf::DirectBatchCapacityPlan makePsiFormerDirectBatchCapacityPlan(
    const BatchExecutionRequirements& requirements,
    const BatchTileCapacities& selected_capacities,
    std::size_t reserve_walkers)
{
  validateModeStructure(requirements);
  pf::DirectBatchCapacityPlan plan;
  plan.tile = {0, 0, 0};
  if (reserve_walkers == 0)
    return plan;

  const bool value = requirements.requires(BatchExecutionMode::VALUE);
  const bool full = requirements.requires(BatchExecutionMode::FULL_VGL);
  const bool active = requirements.requires(BatchExecutionMode::ACTIVE_GRADIENT);
  const bool ecp = flattenedEcpRequired(requirements);
  plan.logical.value_dense = value ? reserve_walkers : 0;
  plan.logical.full_vgl = full ? reserve_walkers : 0;
  plan.logical.active_gradient = active ? reserve_walkers : 0;
  plan.logical.sparse_references = ecp
      ? std::min(reserve_walkers, selected_capacities.ecp_outer)
      : 0;
  plan.logical.sparse_replacements = ecp ? selected_capacities.ecp_outer : 0;
  plan.tile.value = (value || ecp) ? selected_capacities.value : 0;
  plan.tile.full_vgl = full ? selected_capacities.full_vgl : 0;
  plan.tile.active_gradient = active
      ? selected_capacities.active_gradient
      : 0;
  return plan;
}

std::vector<pf::DirectBatchCapacityPlan> makePsiFormerDirectBatchCapacityPlans(
    const BatchExecutionPlanningContext& context)
{
  const std::vector<std::size_t>& reserves =
      psiFormerReserveWalkersPerCrowd(context.topology);
  std::vector<pf::DirectBatchCapacityPlan> plans;
  plans.reserve(reserves.size());
  for (const std::size_t reserve : reserves)
    plans.push_back(makePsiFormerDirectBatchCapacityPlan(
        context.requirements, context.candidate_capacities, reserve));
  return plans;
}

std::vector<PsiFormerCrowdMemoryPlan> makePsiFormerCrowdMemoryPlans(
    const PsiFormerMemoryPolicyInput& input,
    const BatchExecutionPlanningContext& context)
{
  if (input.active_parameter_count > input.storage_shape.parameter_count)
    throw std::invalid_argument(
        "PsiFormer component active-parameter count exceeds the model parameter count");
  if (input.active_parameter_count > context.active_parameter_count)
    throw std::invalid_argument(
        "PsiFormer component active-parameter count exceeds the plan-wide active count");

  validateModeStructure(context.requirements);
  const std::vector<std::size_t>& reserves =
      psiFormerReserveWalkersPerCrowd(context.topology);
  const std::vector<std::size_t>& initial =
      context.topology.initial_walkers_per_crowd;

  const bool value = context.requirements.requires(BatchExecutionMode::VALUE);
  const bool full = context.requirements.requires(BatchExecutionMode::FULL_VGL);
  const bool active = context.requirements.requires(
      BatchExecutionMode::ACTIVE_GRADIENT);
  const bool ecp = flattenedEcpRequired(context.requirements);
  const bool weighted_ecp = context.requirements.requires(
      BatchExecutionMode::ECP_WEIGHTED_SCORE);
  const bool score = scoreTapeRequired(context.requirements) &&
      input.active_parameter_count != 0;
  const bool kinetic = context.requirements.requires(BatchExecutionMode::KINETIC) &&
      input.active_parameter_count != 0;
  if ((score || kinetic) && input.storage_shape.parameter_count == 0)
    throw std::invalid_argument(
        "PsiFormer active derivatives require a nonzero model parameter count");

  // Validate the clone representation even when no crowd currently reserves a
  // walker.  This keeps malformed type-width inputs independent of topology.
  const pf::CloneStateStorageRequirement clone_state =
      pf::cloneStateStorageRequirement(
          input.storage_shape.electrons, input.type_sizes.value_type,
          input.type_sizes.gradient_type);

  const bool scalar = context.requirements.requires(
      BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  pf::DirectBatchStorageRequirement scalar_storage;
  std::size_t scalar_publication_bytes = 0;
  if (scalar)
  {
    const pf::DirectBatchCapacityPlan scalar_plan =
        makePsiFormerScalarValueCapacityPlan(
            input, context.requirements, context.candidate_capacities);
    scalar_storage =
        pf::directBatchStorageRequirement(input.storage_shape, scalar_plan);
    scalar_publication_bytes =
        pf::scalarValuePublicationStorageRequirement(
            input.scalar_value_logical_maximum, input.type_sizes.value_type);
  }

  std::vector<PsiFormerCrowdMemoryPlan> plans;
  plans.reserve(reserves.size());
  for (std::size_t crowd = 0; crowd < reserves.size(); ++crowd)
  {
    PsiFormerCrowdMemoryPlan plan;
    plan.initial_walkers = initial[crowd];
    plan.reserve_walkers = reserves[crowd];
    plan.score_required = score;
    plan.kinetic_required = kinetic;

    addHostBytes(
        plan.expected_clone_storage, BatchMemoryCategory::FIXED_CLONE_STATE,
        checkedBatchMemoryMultiply(
            clone_state.totalBytes(), plan.reserve_walkers,
            "PsiFormer crowd clone-state storage"),
        "PsiFormer crowd clone-state storage");

    if (scalar)
    {
      addDirectStorage(plan.expected_clone_storage, scalar_storage,
                       plan.reserve_walkers,
                       "PsiFormer scalar VALUE compatibility");
      addHostBytes(
          plan.expected_clone_storage,
          BatchMemoryCategory::PUBLICATION_STAGING,
          checkedBatchMemoryMultiply(
              scalar_publication_bytes, plan.reserve_walkers,
              "PsiFormer scalar VALUE publication multiplicity"),
          "PsiFormer scalar VALUE publication");
    }

    // Construct the canonical direct descriptor even for an empty reserve;
    // its zero tile distinguishes a prepared empty crowd from default legacy
    // workspace capacities.
    plan.direct_batch = makePsiFormerDirectBatchCapacityPlan(
        context.requirements, context.candidate_capacities,
        plan.reserve_walkers);
    plan.direct_storage = pf::directBatchStorageRequirement(
        input.storage_shape, plan.direct_batch);

    // A resource still has a stable crowd record when its reserve is zero, but
    // no resource-owned arrays or derivative tapes may be allocated for it.
    if (plan.reserve_walkers == 0)
    {
      plan.expected_storage.add(plan.expected_clone_storage,
                                "PsiFormer crowd clone storage");
      plans.push_back(std::move(plan));
      continue;
    }

    addDirectStorage(plan.expected_resource_storage, plan.direct_storage, 1,
                     "PsiFormer crowd " + std::to_string(crowd));

    plan.publication_staging = {
        plan.reserve_walkers,
        plan.direct_batch.logical.sparse_references,
        plan.direct_batch.logical.sparse_replacements,
        input.active_parameter_count,
        input.type_sizes.value_type,
        input.type_sizes.log_value_type,
        input.type_sizes.gradient_type,
        input.type_sizes.selected_delta_element,
        value,
        full,
        active,
        ecp,
        weighted_ecp,
        score,
        kinetic};
    plan.publication_storage = pf::resourceStagingStorageRequirement(
        plan.publication_staging);
    addHostBytes(plan.expected_resource_storage,
                 BatchMemoryCategory::PUBLICATION_STAGING,
                 plan.publication_storage.totalBytes(),
                 "PsiFormer crowd publication staging");

    if (score)
    {
      plan.score_workspace_bytes =
          pf::scoreWorkspaceStorageRequirement(input.storage_shape);
      addHostBytes(plan.expected_resource_storage,
                   BatchMemoryCategory::SCORE_TAPE,
                   plan.score_workspace_bytes,
                   "PsiFormer crowd score tape");
    }
    if (kinetic)
    {
      plan.kinetic_workspace_bytes =
          pf::kineticWorkspaceStorageRequirement(input.storage_shape);
      plan.total_log_gradient_bytes = checkedBatchMemoryMultiply(
          checkedBatchMemoryMultiply(
              input.storage_shape.electrons, 3,
              "PsiFormer kinetic total-drift extent"),
          sizeof(double), "PsiFormer kinetic total-drift bytes");
      addHostBytes(
          plan.expected_resource_storage, BatchMemoryCategory::KINETIC_TAPE,
          checkedBatchMemoryAdd(
              plan.kinetic_workspace_bytes,
              plan.total_log_gradient_bytes,
              "PsiFormer kinetic tape and total drift"),
          "PsiFormer crowd kinetic tape");
    }

    plan.expected_storage.add(plan.expected_clone_storage,
                              "PsiFormer crowd clone storage");
    plan.expected_storage.add(plan.expected_resource_storage,
                              "PsiFormer crowd resource storage");
    plans.push_back(std::move(plan));
  }
  return plans;
}

BatchMemoryContribution estimatePsiFormerBatchMemory(
    const PsiFormerMemoryPolicyInput& input,
    const BatchExecutionPlanningContext& context)
{
  const BatchExecutionWorkloadContext workload{
      context.requirements, context.topology, context.particle_count, context.active_parameter_count,
      context.parameter_derivative_width, context.target_coordinate};
  BatchMemoryContribution contribution;
  contribution.logical_maximum =
      psiFormerBatchLogicalMaximum(input, workload);
  contribution.owner_multiplicity = 1;
  contribution.fully_accounted = accountingIsComplete(input, context);

  for (const PsiFormerCrowdMemoryPlan& plan :
       makePsiFormerCrowdMemoryPlans(input, context))
    contribution.per_owner.add(plan.expected_storage,
                               "PsiFormer rank crowd storage");

  return contribution;
}

} // namespace qmcplusplus::psiformer
