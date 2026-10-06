//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_memory_policy.cpp
 * @brief Unit tests for exact rank-local PsiFormer batch-memory accounting.
 */

#include <catch2/catch_test_macros.hpp>

#include "QMCWaveFunctions/PsiFormer/PsiFormerMemoryPolicy.h"

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmcplusplus
{
namespace
{

using psiformer::PsiFormerMemoryAccountingClaims;
using psiformer::PsiFormerMemoryBackend;
using psiformer::PsiFormerMemoryPolicyInput;

PsiFormerMemoryPolicyInput makePolicyInput()
{
  PsiFormerMemoryPolicyInput input;
  input.storage_shape = {/* electrons */ 2,
                         /* nuclei */ 1,
                         /* determinants */ 1,
                         /* feature_width */ 4,
                         /* attention_heads */ 1,
                         /* input_width */ 8,
                         /* attention_blocks */ 1,
                         /* parameter_count */ 7};
  input.type_sizes =
      psiformer::makePsiFormerMemoryTypeSizes<
          double, std::complex<double>, std::array<double, 3>,
          std::pair<std::size_t, double>>();
  input.accounting_claims = PsiFormerMemoryAccountingClaims::complete();
  return input;
}

BatchExecutionPlanningContext makeContext(
    const PsiFormerMemoryPolicyInput& input,
    BatchExecutionRequirements requirements,
    std::vector<std::size_t> initial,
    std::vector<std::size_t> reserve,
    BatchTileCapacities candidate,
    std::size_t active_parameters = 0)
{
  BatchExecutionTopology topology;
  topology.initial_walkers_per_crowd = std::move(initial);
  topology.reserve_walkers_per_crowd = std::move(reserve);
  const BatchExecutionWorkloadContext workload{requirements, topology, 0,
                                                active_parameters, 0};
  return {requirements, topology,
          psiformer::psiFormerBatchLogicalMaximum(input, workload), candidate,
          0, active_parameters, 0};
}

std::size_t hostBytes(const BatchMemoryContribution& contribution,
                      BatchMemoryCategory category)
{
  return contribution.per_owner.at(category).host;
}

void checkHostOnly(const BatchMemoryContribution& contribution)
{
  for (std::size_t category = 0;
       category < static_cast<std::size_t>(BatchMemoryCategory::COUNT);
       ++category)
    CHECK(contribution.per_owner.at(
              static_cast<BatchMemoryCategory>(category)).device == 0);
}

std::size_t logicalBytes(const pf::DirectBatchStorageRequirement& storage)
{
  return checkedBatchMemoryAdd(
      checkedBatchMemoryAdd(storage.dense_logical, storage.sparse_logical,
                            "test logical input"),
      storage.logical_outputs, "test logical output");
}

std::size_t innerTileBytes(const pf::DirectBatchStorageRequirement& storage)
{
  std::size_t total = 0;
  for (const std::size_t bytes : {
           storage.sparse_tile_positions, storage.value_tile,
           storage.full_vgl_tile, storage.active_gradient_tile,
           storage.shared_spatial_arena})
    total = checkedBatchMemoryAdd(total, bytes, "test inner tile");
  return total;
}

/** Compute dense publication storage from its public ownership contract.
 *
 * This intentionally does not call ``resourceStagingStorageRequirement`` so
 * the policy tests do not use the implementation under test as their oracle.
 */
std::size_t densePublicationBytes(std::size_t walkers,
                                  std::size_t value_type_bytes,
                                  std::size_t log_value_type_bytes,
                                  std::size_t gradient_type_bytes,
                                  bool value,
                                  bool full_vgl,
                                  bool active_gradient)
{
  std::size_t total = 0;
  const auto add_array = [&total, walkers](std::size_t element_bytes,
                                           const char* context) {
    total = checkedBatchMemoryAdd(
        total, checkedBatchMemoryMultiply(walkers, element_bytes, context),
        context);
  };

  if (value || full_vgl || active_gradient)
  {
    add_array(sizeof(std::size_t), "test walker-index publication");
    add_array(sizeof(std::uint64_t), "test configuration publication");
    add_array(sizeof(double), "test sign publication");
    add_array(sizeof(double), "test log-magnitude publication");
    add_array(full_vgl ? std::max(value_type_bytes, log_value_type_bytes)
                       : value_type_bytes,
              "test ratio publication");
  }
  if (value)
    add_array(sizeof(unsigned char), "test preservation publication");
  if (full_vgl)
    add_array(sizeof(std::size_t), "test batch-slot publication");
  if (full_vgl || active_gradient)
    add_array(gradient_type_bytes, "test gradient publication");
  if (active_gradient)
    add_array(sizeof(std::size_t), "test active-electron publication");
  return total;
}

} // namespace

TEST_CASE("PsiFormer memory policy discovers stable logical envelopes",
          "[wavefunction][psiformer][batch_memory]")
{
  PsiFormerMemoryPolicyInput input = makePolicyInput();
  input.scalar_value_logical_maximum = 5;

  BatchExecutionRequirements requirements;
  requirements.require(BatchExecutionMode::VALUE);
  requirements.require(BatchExecutionMode::FULL_VGL);
  requirements.require(BatchExecutionMode::ACTIVE_GRADIENT);
  requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);

  BatchExecutionTopology topology;
  topology.initial_walkers_per_crowd = {1, 4, 0};
  topology.reserve_walkers_per_crowd = {3, 0, 2};
  const BatchExecutionWorkloadContext workload{requirements, topology, 0, 0, 0};
  CHECK(psiformer::psiFormerBatchLogicalMaximum(input, workload) ==
        BatchTileCapacities{5, 3, 3, 0});

  const psiformer::PsiFormerMemoryTopologySummary summary =
      psiformer::summarizePsiFormerMemoryTopology(topology);
  CHECK(summary.resident_walkers == 5);
  CHECK(summary.prepared_crowds == 2);

  // Omitting the reserve vector uses the uneven initial topology exactly.
  topology.reserve_walkers_per_crowd.clear();
  const BatchExecutionWorkloadContext fallback{requirements, topology, 0, 0, 0};
  CHECK(psiformer::psiFormerBatchLogicalMaximum(input, fallback) ==
        BatchTileCapacities{5, 4, 4, 0});
  const psiformer::PsiFormerMemoryTopologySummary fallback_summary =
      psiformer::summarizePsiFormerMemoryTopology(topology);
  CHECK(fallback_summary.resident_walkers == 5);
  CHECK(fallback_summary.prepared_crowds == 2);

  BatchExecutionTopology empty_reserve;
  empty_reserve.initial_walkers_per_crowd = {0, 0};
  empty_reserve.reserve_walkers_per_crowd = {0, 0};
  const psiformer::PsiFormerMemoryTopologySummary empty_summary =
      psiformer::summarizePsiFormerMemoryTopology(empty_reserve);
  CHECK(empty_summary.resident_walkers == 0);
  CHECK(empty_summary.prepared_crowds == 0);

  BatchExecutionTopology insufficient_reserve;
  insufficient_reserve.initial_walkers_per_crowd = {2, 1};
  insufficient_reserve.reserve_walkers_per_crowd = {0, 2};
  CHECK_THROWS_AS(
      psiformer::summarizePsiFormerMemoryTopology(insufficient_reserve),
      std::invalid_argument);

  // Scalar compatibility implies the VALUE tile but does not create a crowd VALUE
  // envelope or claim the operator-owned ECP_OUTER maximum.
  BatchExecutionRequirements scalar_only;
  scalar_only.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  const BatchExecutionWorkloadContext scalar_workload{scalar_only, topology, 0, 0, 0};
  CHECK(batchExecutionModeIsRequired(scalar_only, BatchExecutionMode::VALUE));
  CHECK(psiformer::psiFormerBatchLogicalMaximum(input, scalar_workload) ==
        BatchTileCapacities{5, 0, 0, 0});

  input.scalar_value_logical_maximum = 0;
  CHECK_THROWS_AS(
      psiformer::psiFormerBatchLogicalMaximum(input, scalar_workload),
      std::invalid_argument);
}

TEST_CASE("PsiFormer memory policy sums exact uneven crowd owners",
          "[wavefunction][psiformer][batch_memory]")
{
  const PsiFormerMemoryPolicyInput input = makePolicyInput();
  BatchExecutionRequirements requirements;
  requirements.require(BatchExecutionMode::VALUE);
  requirements.require(BatchExecutionMode::FULL_VGL);
  requirements.require(BatchExecutionMode::ACTIVE_GRADIENT);
  const BatchExecutionPlanningContext context = makeContext(
      input, requirements, {2, 2, 0}, {3, 0, 1}, {2, 1, 1, 0});

  const BatchMemoryContribution contribution =
      psiformer::estimatePsiFormerBatchMemory(input, context);
  CHECK(contribution.owner_multiplicity == 1);
  CHECK(contribution.fully_accounted);
  CHECK(contribution.logical_maximum == BatchTileCapacities{3, 3, 3, 0});
  checkHostOnly(contribution);

  // Each resident clone owns accepted/proposed gradients and Laplacians.  Keep
  // this expectation independent of the production storage descriptor.
  const std::size_t clone_bytes = checkedBatchMemoryMultiply(
      2 * input.storage_shape.electrons,
      input.type_sizes.gradient_type + input.type_sizes.value_type,
      "test clone state");
  CHECK(hostBytes(contribution, BatchMemoryCategory::FIXED_CLONE_STATE) ==
        checkedBatchMemoryMultiply(clone_bytes, 4,
                                   "test clone multiplicity"));

  std::size_t expected_logical = 0;
  std::size_t expected_inner = 0;
  std::size_t expected_publication = 0;
  for (const std::size_t reserve : {std::size_t{3}, std::size_t{1}})
  {
    const pf::DirectBatchCapacityPlan plan =
        psiformer::makePsiFormerDirectBatchCapacityPlan(
            requirements, context.candidate_capacities, reserve);
    const pf::DirectBatchStorageRequirement storage =
        pf::directBatchStorageRequirement(input.storage_shape, plan);
    expected_logical = checkedBatchMemoryAdd(
        expected_logical, logicalBytes(storage), "test logical sum");
    expected_inner = checkedBatchMemoryAdd(
        expected_inner, innerTileBytes(storage), "test inner sum");
    expected_publication = checkedBatchMemoryAdd(
        expected_publication,
        densePublicationBytes(reserve, input.type_sizes.value_type,
                              input.type_sizes.log_value_type,
                              input.type_sizes.gradient_type, true, true, true),
        "test publication sum");
  }
  CHECK(hostBytes(contribution, BatchMemoryCategory::LOGICAL_INPUT_OUTPUT) ==
        expected_logical);
  CHECK(hostBytes(contribution, BatchMemoryCategory::INNER_TILE_SCRATCH) ==
        expected_inner);
  CHECK(hostBytes(contribution, BatchMemoryCategory::PUBLICATION_STAGING) ==
        expected_publication);
  CHECK(hostBytes(contribution, BatchMemoryCategory::REALLOCATION_TRANSIENT) == 0);

  // A zero reserve envelope stays represented in the plan vector but owns no
  // logical or numeric workspace.
  const std::vector<pf::DirectBatchCapacityPlan> plans =
      psiformer::makePsiFormerDirectBatchCapacityPlans(context);
  REQUIRE(plans.size() == 3);
  CHECK(plans[1].logical.value_dense == 0);
  CHECK(plans[1].tile.value == 0);
}

TEST_CASE("PsiFormer crowd plans are exact allocation and category targets",
          "[wavefunction][psiformer][batch_memory]")
{
  PsiFormerMemoryPolicyInput input = makePolicyInput();
  input.active_parameter_count = 3;
  input.scalar_value_logical_maximum = input.storage_shape.electrons + 1;

  BatchExecutionRequirements requirements;
  requirements.require(BatchExecutionMode::VALUE);
  requirements.require(BatchExecutionMode::FULL_VGL);
  requirements.require(BatchExecutionMode::ACTIVE_GRADIENT);
  requirements.require(BatchExecutionMode::KINETIC);
  requirements.require(BatchExecutionMode::ECP_WEIGHTED_SCORE);
  requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  BatchExecutionPlanningContext context = makeContext(
      input, requirements, {2, 2, 0}, {3, 0, 1}, {2, 1, 1, 2}, 5);
  context.logical_maximum.ecp_outer = 2;

  const std::vector<psiformer::PsiFormerCrowdMemoryPlan> plans =
      psiformer::makePsiFormerCrowdMemoryPlans(input, context);
  REQUIRE(plans.size() == 3);

  const psiformer::PsiFormerCrowdMemoryPlan& first = plans[0];
  CHECK(first.initial_walkers == 2);
  CHECK(first.reserve_walkers == 3);
  CHECK(first.direct_batch.logical.value_dense == 3);
  CHECK(first.direct_batch.logical.full_vgl == 3);
  CHECK(first.direct_batch.logical.active_gradient == 3);
  CHECK(first.direct_batch.logical.sparse_references == 2);
  CHECK(first.direct_batch.logical.sparse_replacements == 2);
  CHECK(first.direct_batch.tile.value == 2);
  CHECK(first.direct_batch.tile.full_vgl == 1);
  CHECK(first.direct_batch.tile.active_gradient == 1);

  CHECK(first.publication_staging.reserve_walkers == 3);
  CHECK(first.publication_staging.sparse_references == 2);
  CHECK(first.publication_staging.sparse_replacements == 2);
  CHECK(first.publication_staging.active_parameters == 3);
  CHECK(first.publication_staging.value_type_bytes == sizeof(double));
  CHECK(first.publication_staging.log_value_type_bytes ==
        sizeof(std::complex<double>));
  CHECK(first.publication_staging.gradient_type_bytes ==
        sizeof(std::array<double, 3>));
  CHECK(first.publication_staging.value);
  CHECK(first.publication_staging.full_vgl);
  CHECK(first.publication_staging.active_gradient);
  CHECK(first.publication_staging.flattened_ecp);
  CHECK(first.publication_staging.weighted_ecp_score);
  CHECK(first.publication_staging.score);
  CHECK(first.publication_staging.kinetic);

  CHECK(first.score_required);
  CHECK(first.kinetic_required);
  CHECK(first.score_workspace_bytes ==
        pf::scoreWorkspaceStorageRequirement(input.storage_shape));
  CHECK(first.kinetic_workspace_bytes ==
        pf::kineticWorkspaceStorageRequirement(input.storage_shape));
  CHECK(first.total_log_gradient_bytes ==
        3 * input.storage_shape.electrons * sizeof(double));

  const pf::CloneStateStorageRequirement clone_storage =
      pf::cloneStateStorageRequirement(
          input.storage_shape.electrons, input.type_sizes.value_type,
          input.type_sizes.gradient_type);
  CHECK(first.expected_clone_storage.at(
            BatchMemoryCategory::FIXED_CLONE_STATE).host ==
        3 * clone_storage.totalBytes());
  CHECK(first.expected_storage.at(BatchMemoryCategory::FIXED_CLONE_STATE).host ==
        3 * clone_storage.totalBytes());

  const pf::DirectBatchCapacityPlan scalar_plan =
      psiformer::makePsiFormerScalarValueCapacityPlan(
          input, requirements, context.candidate_capacities);
  const pf::DirectBatchStorageRequirement scalar_storage =
      pf::directBatchStorageRequirement(input.storage_shape, scalar_plan);
  CHECK(first.expected_clone_storage.at(
            BatchMemoryCategory::LOGICAL_INPUT_OUTPUT).host ==
        3 * logicalBytes(scalar_storage));
  CHECK(first.expected_resource_storage.at(
            BatchMemoryCategory::LOGICAL_INPUT_OUTPUT).host ==
        logicalBytes(first.direct_storage));
  CHECK(first.expected_storage.at(BatchMemoryCategory::LOGICAL_INPUT_OUTPUT).host ==
        logicalBytes(first.direct_storage) +
            3 * logicalBytes(scalar_storage));
  CHECK(first.expected_storage.at(BatchMemoryCategory::INNER_TILE_SCRATCH).host ==
        innerTileBytes(first.direct_storage) +
            3 * innerTileBytes(scalar_storage));
  CHECK(first.expected_storage.at(BatchMemoryCategory::REALLOCATION_TRANSIENT).host ==
        first.direct_storage.replacementTransientBytes() +
            3 * scalar_storage.replacementTransientBytes());
  CHECK(first.expected_storage.at(BatchMemoryCategory::PUBLICATION_STAGING).host ==
        first.publication_storage.totalBytes() +
            3 * input.scalar_value_logical_maximum * sizeof(double));
  CHECK(first.expected_resource_storage.at(
            BatchMemoryCategory::PUBLICATION_STAGING).host ==
        first.publication_storage.totalBytes());
  CHECK(first.expected_storage.at(BatchMemoryCategory::SCORE_TAPE).host ==
        first.score_workspace_bytes);
  CHECK(first.expected_storage.at(BatchMemoryCategory::KINETIC_TAPE).host ==
        first.kinetic_workspace_bytes + first.total_log_gradient_bytes);

  // A zero-reserve record preserves its original topology position without
  // inventing any resource allocation, even when every mode is required.
  const psiformer::PsiFormerCrowdMemoryPlan& empty = plans[1];
  CHECK(empty.initial_walkers == 2);
  CHECK(empty.reserve_walkers == 0);
  CHECK(empty.score_required);
  CHECK(empty.kinetic_required);
  CHECK(empty.direct_batch.logical.value_dense == 0);
  CHECK(empty.direct_batch.logical.full_vgl == 0);
  CHECK(empty.direct_batch.logical.active_gradient == 0);
  CHECK(empty.direct_batch.logical.sparse_references == 0);
  CHECK(empty.direct_batch.logical.sparse_replacements == 0);
  CHECK(empty.direct_batch.tile.value == 0);
  CHECK(empty.direct_batch.tile.full_vgl == 0);
  CHECK(empty.direct_batch.tile.active_gradient == 0);
  CHECK(empty.publication_staging.reserve_walkers == 0);
  CHECK(empty.score_workspace_bytes == 0);
  CHECK(empty.kinetic_workspace_bytes == 0);
  CHECK(empty.total_log_gradient_bytes == 0);
  CHECK(empty.expected_clone_storage.total().host == 0);
  CHECK(empty.expected_resource_storage.total().host == 0);
  CHECK(empty.expected_storage.total().host == 0);
  CHECK(empty.expected_storage.total().device == 0);

  CHECK(plans[2].initial_walkers == 0);
  CHECK(plans[2].reserve_walkers == 1);
  CHECK(plans[2].direct_batch.logical.sparse_references == 1);
  CHECK(plans[2].direct_batch.logical.sparse_replacements == 2);

  BatchExecutionPlanningContext fallback_context = context;
  fallback_context.topology.initial_walkers_per_crowd = {1, 3};
  fallback_context.topology.reserve_walkers_per_crowd.clear();
  const std::vector<psiformer::PsiFormerCrowdMemoryPlan> fallback_plans =
      psiformer::makePsiFormerCrowdMemoryPlans(input, fallback_context);
  REQUIRE(fallback_plans.size() == 2);
  CHECK(fallback_plans[0].initial_walkers == 1);
  CHECK(fallback_plans[0].reserve_walkers == 1);
  CHECK(fallback_plans[1].initial_walkers == 3);
  CHECK(fallback_plans[1].reserve_walkers == 3);

  // The public estimator is deliberately just the checked category sum of the
  // allocation plans consumed later by live resource preparation.
  BatchMemoryEstimate expected_rank;
  for (const psiformer::PsiFormerCrowdMemoryPlan& plan : plans)
    expected_rank.add(plan.expected_storage, "test crowd-plan sum");
  const BatchMemoryContribution contribution =
      psiformer::estimatePsiFormerBatchMemory(input, context);
  CHECK(contribution.per_owner == expected_rank);
  checkHostOnly(contribution);
}

TEST_CASE("PsiFormer crowd plans preserve build-dependent element widths",
          "[wavefunction][psiformer][batch_memory]")
{
  PsiFormerMemoryPolicyInput input = makePolicyInput();
  input.type_sizes = psiformer::makePsiFormerMemoryTypeSizes<
      std::complex<double>, std::complex<double>,
      std::array<std::complex<double>, 3>,
      std::pair<std::size_t, std::complex<double>>>();

  BatchExecutionRequirements requirements;
  requirements.require(BatchExecutionMode::FULL_VGL);
  const BatchExecutionPlanningContext context = makeContext(
      input, requirements, {1}, {2}, {0, 1, 0, 0});
  const std::vector<psiformer::PsiFormerCrowdMemoryPlan> plans =
      psiformer::makePsiFormerCrowdMemoryPlans(input, context);
  REQUIRE(plans.size() == 1);

  const psiformer::PsiFormerCrowdMemoryPlan& plan = plans.front();
  CHECK(plan.publication_staging.value_type_bytes ==
        sizeof(std::complex<double>));
  CHECK(plan.publication_staging.gradient_type_bytes ==
        sizeof(std::array<std::complex<double>, 3>));
  CHECK(plan.publication_storage.ratios ==
        2 * sizeof(std::complex<double>));
  CHECK(plan.publication_storage.gradients ==
        2 * sizeof(std::array<std::complex<double>, 3>));

  const pf::CloneStateStorageRequirement clone_storage =
      pf::cloneStateStorageRequirement(
          input.storage_shape.electrons, input.type_sizes.value_type,
          input.type_sizes.gradient_type);
  CHECK(plan.expected_storage.at(BatchMemoryCategory::FIXED_CLONE_STATE).host ==
        2 * clone_storage.totalBytes());
  checkHostOnly(psiformer::estimatePsiFormerBatchMemory(input, context));
}

TEST_CASE("PsiFormer memory policy keeps scalar VALUE ownership clone local",
          "[wavefunction][psiformer][batch_memory]")
{
  PsiFormerMemoryPolicyInput input = makePolicyInput();
  input.scalar_value_logical_maximum = input.storage_shape.electrons + 1;
  BatchExecutionRequirements requirements;
  requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  const BatchExecutionPlanningContext context = makeContext(
      input, requirements, {2, 1}, {}, {2, 0, 0, 0});

  const BatchMemoryContribution contribution =
      psiformer::estimatePsiFormerBatchMemory(input, context);
  CHECK(contribution.owner_multiplicity == 1);
  CHECK(contribution.fully_accounted);
  CHECK(contribution.logical_maximum == BatchTileCapacities{3, 0, 0, 0});

  const pf::DirectBatchCapacityPlan scalar_plan =
      psiformer::makePsiFormerScalarValueCapacityPlan(
          input, requirements, context.candidate_capacities);
  CHECK(scalar_plan.logical.value_dense == 3);
  CHECK(scalar_plan.logical.full_vgl == 0);
  CHECK(scalar_plan.logical.active_gradient == 0);
  CHECK(scalar_plan.tile.value == 2);
  CHECK(scalar_plan.tile.full_vgl == 0);
  CHECK(scalar_plan.tile.active_gradient == 0);
  const pf::DirectBatchStorageRequirement scalar_storage =
      pf::directBatchStorageRequirement(input.storage_shape, scalar_plan);
  CHECK(hostBytes(contribution, BatchMemoryCategory::LOGICAL_INPUT_OUTPUT) ==
        checkedBatchMemoryMultiply(logicalBytes(scalar_storage), 3,
                                   "test scalar clones"));
  CHECK(hostBytes(contribution, BatchMemoryCategory::INNER_TILE_SCRATCH) ==
        checkedBatchMemoryMultiply(innerTileBytes(scalar_storage), 3,
                                   "test scalar clones"));
  CHECK(hostBytes(contribution, BatchMemoryCategory::PUBLICATION_STAGING) ==
        3 * 3 * sizeof(double));

  BatchExecutionRequirements no_scalar;
  const pf::DirectBatchCapacityPlan empty_plan =
      psiformer::makePsiFormerScalarValueCapacityPlan(
          input, no_scalar, context.candidate_capacities);
  CHECK(empty_plan.logical.value_dense == 0);
  CHECK(empty_plan.tile.value == 0);

  input.scalar_value_logical_maximum = 0;
  CHECK_THROWS_AS(
      psiformer::makePsiFormerScalarValueCapacityPlan(
          input, requirements, context.candidate_capacities),
      std::invalid_argument);
  input.scalar_value_logical_maximum = 3;
  CHECK_THROWS_AS(
      psiformer::makePsiFormerScalarValueCapacityPlan(
          input, requirements, BatchTileCapacities{}),
      std::invalid_argument);
}

TEST_CASE("PsiFormer memory policy maps flattened sparse and derivative storage",
          "[wavefunction][psiformer][batch_memory]")
{
  PsiFormerMemoryPolicyInput input = makePolicyInput();
  input.active_parameter_count = 3;
  BatchExecutionRequirements requirements;
  requirements.require(BatchExecutionMode::VALUE);
  requirements.require(BatchExecutionMode::ECP_WEIGHTED_SCORE);
  BatchExecutionPlanningContext context = makeContext(
      input, requirements, {4, 3}, {5, 2}, {2, 0, 0, 3}, 3);
  context.logical_maximum.ecp_outer = 3; // supplied by the ECP participant

  const std::vector<pf::DirectBatchCapacityPlan> plans =
      psiformer::makePsiFormerDirectBatchCapacityPlans(context);
  REQUIRE(plans.size() == 2);
  CHECK(plans[0].logical.sparse_references == 3);
  CHECK(plans[0].logical.sparse_replacements == 3);
  CHECK(plans[1].logical.sparse_references == 2);
  CHECK(plans[1].logical.sparse_replacements == 3);

  const BatchMemoryContribution contribution =
      psiformer::estimatePsiFormerBatchMemory(input, context);
  CHECK(contribution.fully_accounted);
  CHECK(hostBytes(contribution, BatchMemoryCategory::SCORE_TAPE) ==
        2 * pf::scoreWorkspaceStorageRequirement(input.storage_shape));
  CHECK(hostBytes(contribution, BatchMemoryCategory::KINETIC_TAPE) == 0);
  checkHostOnly(contribution);

  std::size_t expected_publication = 0;
  for (const std::size_t reserve : {std::size_t{5}, std::size_t{2}})
  {
    const std::size_t references = std::min(reserve, std::size_t{3});
    std::size_t crowd_bytes = 0;
    const auto add = [&crowd_bytes](std::size_t bytes) {
      crowd_bytes = checkedBatchMemoryAdd(crowd_bytes, bytes,
                                          "test weighted publication");
    };
    add(reserve * sizeof(std::size_t));             // walker indices
    add(reserve * sizeof(std::uint64_t));           // configuration identities
    add(reserve * sizeof(double));                  // signs
    add(reserve * sizeof(double));                  // log magnitudes
    add(reserve * sizeof(double));                  // value ratios
    add(reserve * sizeof(unsigned char));           // preservation flags
    add(references * sizeof(std::size_t));          // active virtual walkers
    add(reserve * sizeof(std::size_t));             // reference-index map
    add(3 * sizeof(double));                        // flattened ratios
    add(references * sizeof(double));               // reference weights
    add(3 * sizeof(std::size_t));                   // active parameter indices
    add(3 * sizeof(std::pair<std::size_t, double>)); // selected score delta
    add(references * 3 * sizeof(double));           // weighted derivative rows
    expected_publication = checkedBatchMemoryAdd(
        expected_publication, crowd_bytes, "test weighted crowd sum");
  }
  CHECK(hostBytes(contribution, BatchMemoryCategory::PUBLICATION_STAGING) ==
        expected_publication);

  // Determinant factorization storage is already transitive through each VALUE
  // slot; it is not registered as another participant or category.
  const pf::DirectBatchStorageRequirement first_storage =
      pf::directBatchStorageRequirement(input.storage_shape, plans[0]);
  CHECK(first_storage.value_tile >=
        2 * pf::determinantStorageRequirement(
                input.storage_shape.determinants,
                input.storage_shape.electrons));

  const BatchMemoryContribution before = contribution;
  context.candidate_capacities.value = 1;
  const BatchMemoryContribution smaller_value =
      psiformer::estimatePsiFormerBatchMemory(input, context);
  CHECK(smaller_value.logical_maximum == before.logical_maximum);
  CHECK(hostBytes(smaller_value, BatchMemoryCategory::FIXED_CLONE_STATE) ==
        hostBytes(before, BatchMemoryCategory::FIXED_CLONE_STATE));
  CHECK(hostBytes(smaller_value, BatchMemoryCategory::LOGICAL_INPUT_OUTPUT) ==
        hostBytes(before, BatchMemoryCategory::LOGICAL_INPUT_OUTPUT));
  CHECK(hostBytes(smaller_value, BatchMemoryCategory::SCORE_TAPE) ==
        hostBytes(before, BatchMemoryCategory::SCORE_TAPE));
  CHECK(hostBytes(smaller_value, BatchMemoryCategory::PUBLICATION_STAGING) ==
        hostBytes(before, BatchMemoryCategory::PUBLICATION_STAGING));
  CHECK(hostBytes(smaller_value, BatchMemoryCategory::INNER_TILE_SCRATCH) <
        hostBytes(before, BatchMemoryCategory::INNER_TILE_SCRATCH));
}

TEST_CASE("PsiFormer score and kinetic minima require active parameters",
          "[wavefunction][psiformer][batch_memory]")
{
  PsiFormerMemoryPolicyInput inactive_input = makePolicyInput();
  BatchExecutionRequirements requirements;
  requirements.require(BatchExecutionMode::SCORE);
  requirements.require(BatchExecutionMode::KINETIC);

  // A nonzero plan-wide count may belong entirely to other components and must
  // not cause PsiFormer to allocate derivative tapes or staging.
  const BatchExecutionPlanningContext inactive_context = makeContext(
      inactive_input, requirements, {2, 0, 1}, {3, 0, 2}, {}, 5);
  const BatchMemoryContribution inactive =
      psiformer::estimatePsiFormerBatchMemory(inactive_input, inactive_context);
  CHECK(hostBytes(inactive, BatchMemoryCategory::SCORE_TAPE) == 0);
  CHECK(hostBytes(inactive, BatchMemoryCategory::KINETIC_TAPE) == 0);
  CHECK(hostBytes(inactive, BatchMemoryCategory::PUBLICATION_STAGING) == 0);

  PsiFormerMemoryPolicyInput active_input = makePolicyInput();
  active_input.active_parameter_count = 2;
  const BatchExecutionPlanningContext active_context = makeContext(
      active_input, requirements, {2, 0, 1}, {3, 0, 2}, {}, 5);
  const BatchMemoryContribution active =
      psiformer::estimatePsiFormerBatchMemory(active_input, active_context);
  CHECK(hostBytes(active, BatchMemoryCategory::SCORE_TAPE) ==
        2 * pf::scoreWorkspaceStorageRequirement(active_input.storage_shape));
  const std::size_t kinetic_per_crowd = checkedBatchMemoryAdd(
      pf::kineticWorkspaceStorageRequirement(active_input.storage_shape),
      3 * active_input.storage_shape.electrons * sizeof(double),
      "test kinetic tape");
  CHECK(hostBytes(active, BatchMemoryCategory::KINETIC_TAPE) ==
        2 * kinetic_per_crowd);
  const std::size_t publication_per_crowd = checkedBatchMemoryAdd(
      2 * sizeof(std::size_t),
      4 * sizeof(std::pair<std::size_t, double>),
      "test local derivative publication");
  CHECK(hostBytes(active, BatchMemoryCategory::PUBLICATION_STAGING) ==
        2 * publication_per_crowd);
  CHECK(hostBytes(active, BatchMemoryCategory::FIXED_CLONE_STATE) ==
        hostBytes(inactive, BatchMemoryCategory::FIXED_CLONE_STATE));

  // The five plan-wide parameters include other components; PsiFormer staging
  // is bounded by its two local active parameters.
  const pf::ResourceStagingStorageRequirement local_staging =
      pf::resourceStagingStorageRequirement(
          {3, 0, 0, 2, active_input.type_sizes.value_type,
           active_input.type_sizes.log_value_type,
           active_input.type_sizes.gradient_type,
           active_input.type_sizes.selected_delta_element,
           false, false, false, false, false, true, true});
  CHECK(local_staging.active_parameter_indices == 2 * sizeof(std::size_t));
  CHECK(local_staging.selected_derivative_deltas ==
        4 * sizeof(std::pair<std::size_t, double>));

  active_input.active_parameter_count = active_input.storage_shape.parameter_count + 1;
  CHECK_THROWS_AS(
      psiformer::estimatePsiFormerBatchMemory(active_input, active_context),
      std::invalid_argument);
  active_input.active_parameter_count = 3;
  BatchExecutionPlanningContext too_small_global = active_context;
  too_small_global.active_parameter_count = 2;
  CHECK_THROWS_AS(
      psiformer::estimatePsiFormerBatchMemory(active_input, too_small_global),
      std::invalid_argument);
}

TEST_CASE("PsiFormer publication staging uses operation-specific scalar widths",
          "[wavefunction][psiformer][batch_memory]")
{
  pf::ResourceStagingCapacityPlan real_full;
  real_full.reserve_walkers      = 3;
  real_full.value_type_bytes     = sizeof(double);
  real_full.log_value_type_bytes = sizeof(std::complex<double>);
  real_full.gradient_type_bytes  = sizeof(std::array<double, 3>);
  real_full.full_vgl             = true;
  const pf::ResourceStagingStorageRequirement real_storage =
      pf::resourceStagingStorageRequirement(real_full);
  CHECK(real_storage.ratios == 3 * sizeof(std::complex<double>));

  pf::ResourceStagingCapacityPlan real_value = real_full;
  real_value.full_vgl = false;
  real_value.value    = true;
  const pf::ResourceStagingStorageRequirement value_storage =
      pf::resourceStagingStorageRequirement(real_value);
  CHECK(value_storage.ratios == 3 * sizeof(double));

  pf::ResourceStagingCapacityPlan complex_full = real_full;
  complex_full.value_type_bytes = sizeof(std::complex<double>);
  const pf::ResourceStagingStorageRequirement complex_storage =
      pf::resourceStagingStorageRequirement(complex_full);
  CHECK(complex_storage.ratios == 3 * sizeof(std::complex<double>));
}

TEST_CASE("PsiFormer memory policy fails closed for unsupported execution",
          "[wavefunction][psiformer][batch_memory]")
{
  BatchExecutionRequirements requirements;
  requirements.require(BatchExecutionMode::VALUE);
  PsiFormerMemoryPolicyInput input = makePolicyInput();
  BatchExecutionPlanningContext context = makeContext(
      input, requirements, {2}, {3}, {2, 0, 0, 0});

  input.accounting_claims = {};
  BatchMemoryContribution contribution =
      psiformer::estimatePsiFormerBatchMemory(input, context);
  CHECK_FALSE(contribution.fully_accounted);
  CHECK_THROWS(aggregateBatchMemoryContributions(
      {{"twf/component/0/PsiFormer/test", contribution}}));

  input.accounting_claims = PsiFormerMemoryAccountingClaims::complete();
  CHECK(psiformer::estimatePsiFormerBatchMemory(input, context).fully_accounted);

  input.backends.value = PsiFormerMemoryBackend::ORACLE;
  CHECK_FALSE(
      psiformer::estimatePsiFormerBatchMemory(input, context).fully_accounted);
  input.backends.value = PsiFormerMemoryBackend::COMPARE;
  CHECK_FALSE(
      psiformer::estimatePsiFormerBatchMemory(input, context).fully_accounted);
  input.backends.value = PsiFormerMemoryBackend::DIRECT;

  context.topology.serialized_walkers = true;
  CHECK_FALSE(
      psiformer::estimatePsiFormerBatchMemory(input, context).fully_accounted);
  context.topology.serialized_walkers = false;

  requirements.require(BatchExecutionMode::ECP_OUTER);
  context.requirements = requirements;
  context.candidate_capacities.ecp_outer = 2;
  input.flattened_ecp = false;
  CHECK_FALSE(
      psiformer::estimatePsiFormerBatchMemory(input, context).fully_accounted);

  BatchExecutionRequirements invalid_ecp;
  invalid_ecp.require(BatchExecutionMode::ECP_OUTER);
  BatchExecutionTopology zero_topology;
  zero_topology.initial_walkers_per_crowd = {0};
  zero_topology.reserve_walkers_per_crowd = {0};
  const BatchExecutionWorkloadContext invalid_workload{invalid_ecp, zero_topology, 0, 0, 0};
  CHECK_THROWS_AS(
      psiformer::psiFormerBatchLogicalMaximum(input, invalid_workload),
      std::invalid_argument);
  CHECK_THROWS_AS(
      psiformer::makePsiFormerDirectBatchCapacityPlan(invalid_ecp, {}, 0),
      std::invalid_argument);
}

TEST_CASE("PsiFormer memory policy checks every ownership extent",
          "[wavefunction][psiformer][batch_memory]")
{
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(
      pf::cloneStateStorageRequirement(maximum, 2, 1), std::length_error);

  pf::ResourceStagingCapacityPlan weighted;
  weighted.sparse_references = maximum;
  weighted.sparse_replacements = 1;
  weighted.active_parameters = 2;
  weighted.value_type_bytes = sizeof(double);
  weighted.log_value_type_bytes = sizeof(std::complex<double>);
  weighted.gradient_type_bytes = sizeof(std::array<double, 3>);
  weighted.selected_delta_bytes = sizeof(std::pair<std::size_t, double>);
  weighted.flattened_ecp = true;
  weighted.weighted_ecp_score = true;
  CHECK_THROWS_AS(pf::resourceStagingStorageRequirement(weighted),
                  std::length_error);

  pf::ResourceStagingCapacityPlan weighted_reference;
  weighted_reference.sparse_references = 1;
  weighted_reference.flattened_ecp = true;
  weighted_reference.weighted_ecp_score = true;
  CHECK_THROWS_AS(
      pf::resourceStagingStorageRequirement(weighted_reference),
      std::invalid_argument);

  pf::ResourceStagingCapacityPlan weighted_product;
  weighted_product.sparse_references = 2;
  weighted_product.sparse_replacements = 1;
  weighted_product.active_parameters = maximum / 2 + 1;
  weighted_product.value_type_bytes = 1;
  weighted_product.log_value_type_bytes = 1;
  weighted_product.gradient_type_bytes = 1;
  weighted_product.selected_delta_bytes = 1;
  weighted_product.flattened_ecp = true;
  weighted_product.weighted_ecp_score = true;
  CHECK_THROWS_AS(
      pf::resourceStagingStorageRequirement(weighted_product),
      std::length_error);

  pf::ResourceStagingCapacityPlan parameter_extent;
  parameter_extent.active_parameters = maximum;
  parameter_extent.value_type_bytes = 1;
  parameter_extent.log_value_type_bytes = 1;
  parameter_extent.selected_delta_bytes = 1;
  parameter_extent.score = true;
  CHECK_THROWS_AS(
      pf::resourceStagingStorageRequirement(parameter_extent),
      std::length_error);

  BatchExecutionTopology topology;
  topology.initial_walkers_per_crowd = {maximum, 1};
  CHECK_THROWS_AS(psiformer::summarizePsiFormerMemoryTopology(topology),
                  std::overflow_error);

  PsiFormerMemoryPolicyInput input = makePolicyInput();
  BatchExecutionRequirements requirements;
  requirements.require(BatchExecutionMode::VALUE);
  requirements.require(BatchExecutionMode::ECP_OUTER);
  const BatchExecutionPlanningContext sparse_context = makeContext(
      input, requirements, {1}, {1}, {1, 0, 0, maximum});
  CHECK_THROWS(
      psiformer::makePsiFormerCrowdMemoryPlans(input, sparse_context));
  CHECK_THROWS(psiformer::estimatePsiFormerBatchMemory(input, sparse_context));

  const BatchExecutionPlanningContext clone_multiplicity_context = makeContext(
      input, BatchExecutionRequirements{BatchExecutionMode::VALUE},
      {1}, {maximum}, {1, 0, 0, 0});
  CHECK_THROWS_AS(
      psiformer::makePsiFormerCrowdMemoryPlans(
          input, clone_multiplicity_context),
      std::overflow_error);

  BatchExecutionTopology mismatched;
  mismatched.initial_walkers_per_crowd = {1, 2};
  mismatched.reserve_walkers_per_crowd = {3};
  CHECK_THROWS_AS(
      psiformer::summarizePsiFormerMemoryTopology(mismatched),
      std::invalid_argument);
}

} // namespace qmcplusplus
