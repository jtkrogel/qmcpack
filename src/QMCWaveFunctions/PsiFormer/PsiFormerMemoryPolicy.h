//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerMemoryPolicy.h
 * @brief Pure rank-local batch-memory accounting for one PsiFormer component family.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_MEMORY_POLICY_H
#define QMCPLUSPLUS_PSIFORMER_MEMORY_POLICY_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerStorageRequirements.h"
#include "Utilities/BatchExecutionMemory.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace qmcplusplus::psiformer
{

/** Native execution backends with materially different storage ownership.
 *
 * Explicit memory plans currently admit only DIRECT.  Keeping the backend facts
 * in the pure input lets the estimator fail closed without depending on PsiFormerWF.
 */
enum class PsiFormerMemoryBackend : std::uint8_t
{
  DIRECT,
  ORACLE,
  COMPARE
};

/** Actual build-dependent element widths at the PsiFormerWF ownership boundary. */
struct PsiFormerMemoryTypeSizes
{
  std::size_t value_type               = 0;
  std::size_t psi_value_type           = 0;
  std::size_t log_value_type           = 0;
  std::size_t gradient_type            = 0;
  std::size_t selected_delta_element   = 0;
  std::size_t full_precision_real_type = 0;
};

/** Construct build-exact widths without exposing the corresponding types here. */
template<class ValueType,
         class PsiValueType,
         class LogValueType,
         class GradientType,
         class SelectedDeltaElement,
         class FullPrecisionRealType = double>
constexpr PsiFormerMemoryTypeSizes makePsiFormerMemoryTypeSizes() noexcept
{
  return {sizeof(ValueType), sizeof(PsiValueType), sizeof(LogValueType),
          sizeof(GradientType), sizeof(SelectedDeltaElement),
          sizeof(FullPrecisionRealType)};
}

/** Backends selected for the four independently dispatched native families. */
struct PsiFormerMemoryBackends
{
  PsiFormerMemoryBackend value   = PsiFormerMemoryBackend::DIRECT;
  PsiFormerMemoryBackend spatial = PsiFormerMemoryBackend::DIRECT;
  PsiFormerMemoryBackend score   = PsiFormerMemoryBackend::DIRECT;
  PsiFormerMemoryBackend kinetic = PsiFormerMemoryBackend::DIRECT;
};

/** Explicit implementation coverage claims; all default false by design.
 *
 * A contribution is fully accounted only when every claim needed by its reachable
 * modes is true.  This prevents a partially migrated runtime path from advertising
 * a hard memory cap merely because its already-described buffers were estimated.
 */
struct PsiFormerMemoryAccountingClaims
{
  bool clone_state                = false;
  bool direct_batch               = false;
  bool publication_staging        = false;
  bool score_tape                 = false;
  bool kinetic_tape               = false;
  bool flattened_ecp              = false;
  bool scalar_value_compatibility = false;
  bool walker_record              = false;

  static constexpr PsiFormerMemoryAccountingClaims complete() noexcept
  {
    return {true, true, true, true, true, true, true, true};
  }
};

/** Candidate-independent PsiFormer facts used by the pure policy functions. */
struct PsiFormerMemoryPolicyInput
{
  pf::PsiFormerStorageShape storage_shape;
  PsiFormerMemoryTypeSizes type_sizes;
  PsiFormerMemoryBackends backends;
  PsiFormerMemoryAccountingClaims accounting_claims;

  /** Active parameters owned by this component, excluding other wavefunctions. */
  std::size_t active_parameter_count = 0;

  /** Finite clone-local scalar VALUE envelope, e.g. Ne+1 for all-to-one.
   * Zero means that no scalar compatibility path has been described.
   */
  std::size_t scalar_value_logical_maximum = 0;

  /** Byte alignment used by PooledMemory for independent bulk cursor advances. */
  std::size_t walker_buffer_alignment = 0;

  /** The admitted nonlocal path uses sparse flattened references/replacements. */
  bool flattened_ecp = true;
};

/** Exact rank-local multiplicities retained for diagnostics and preparation. */
struct PsiFormerMemoryTopologySummary
{
  std::size_t resident_walkers = 0;
  std::size_t prepared_crowds  = 0;
};

/** Exact allocation and accounting target for one rank-local crowd.
 *
 * Zero-reserve crowds retain their topology identity and canonical metadata but
 * have empty owned capacities and storage.  ``expected_storage`` includes fixed
 * clone state, shared crowd-resource storage, and caller-owned external records,
 * so summing these records reproduces the complete rank-local policy estimate.
 */
struct PsiFormerCrowdMemoryPlan
{
  std::size_t initial_walkers = 0;
  std::size_t reserve_walkers = 0;

  pf::DirectBatchCapacityPlan direct_batch;
  pf::ResourceStagingCapacityPlan publication_staging;
  pf::DirectBatchStorageRequirement direct_storage;
  pf::ResourceStagingStorageRequirement publication_storage;
  /** Metadata-only descriptor for externally owned bulk and scalar record regions. */
  pf::WalkerBufferLayout walker_buffer_layout;

  bool score_required   = false;
  bool kinetic_required = false;
  std::size_t score_workspace_bytes       = 0;
  std::size_t kinetic_workspace_bytes     = 0;
  std::size_t total_log_gradient_bytes    = 0;

  /** Clone-owned fixed and scalar-compatibility storage for this reserve. */
  BatchMemoryEstimate expected_clone_storage;
  /** Shared crowd-resource storage, suitable for exact post-prepare checks. */
  BatchMemoryEstimate expected_resource_storage;
  /** Caller-owned persistent records, never clone or crowd-resource actual bytes. */
  BatchMemoryEstimate expected_external_walker_record_storage;
  /** Checked sum of clone, resource, and external storage used by rank policy. */
  BatchMemoryEstimate expected_storage;
};

/** Select reserve crowds, falling back to initial crowds, and validate the topology. */
const std::vector<std::size_t>& psiFormerReserveWalkersPerCrowd(
    const BatchExecutionTopology& topology);

/** Return checked rank-local walker and nonempty-resource multiplicities. */
PsiFormerMemoryTopologySummary summarizePsiFormerMemoryTopology(
    const BatchExecutionTopology& topology);

/** Return the checked, build-exact external walker-record layout. */
pf::WalkerBufferLayout makePsiFormerWalkerBufferLayout(
    const PsiFormerMemoryPolicyInput& input);

/** Return candidate-independent component maxima; PsiFormer never supplies ECP_OUTER. */
BatchTileCapacities psiFormerBatchLogicalMaximum(
    const PsiFormerMemoryPolicyInput& input,
    const BatchExecutionWorkloadContext& context);

/** Construct the exact clone-local scalar VALUE plan, or an empty plan when
 * scalar compatibility is not requested.
 *
 * Keeping this translation shared by accounting and live preparation prevents
 * the retained workspace from drifting away from the bytes selected by policy.
 */
pf::DirectBatchCapacityPlan makePsiFormerScalarValueCapacityPlan(
    const PsiFormerMemoryPolicyInput& input,
    const BatchExecutionRequirements& requirements,
    const BatchTileCapacities& selected_capacities);

/** Construct the exact direct-workspace plan for one reserve crowd. */
pf::DirectBatchCapacityPlan makePsiFormerDirectBatchCapacityPlan(
    const BatchExecutionRequirements& requirements,
    const BatchTileCapacities& selected_capacities,
    std::size_t reserve_walkers);

/** Construct exact direct-workspace plans in stable crowd-index order. */
std::vector<pf::DirectBatchCapacityPlan> makePsiFormerDirectBatchCapacityPlans(
    const BatchExecutionPlanningContext& context);

/** Construct exact allocation and category targets in stable crowd-index order. */
std::vector<PsiFormerCrowdMemoryPlan> makePsiFormerCrowdMemoryPlans(
    const PsiFormerMemoryPolicyInput& input,
    const BatchExecutionPlanningContext& context);

/** Estimate one exact rank-local PsiFormer owner contribution.
 *
 * ``owner_multiplicity`` is always one.  ``per_owner`` is the checked sum of
 * every reserve clone and every nonempty prepared crowd on this MPI rank.
 */
BatchMemoryContribution estimatePsiFormerBatchMemory(
    const PsiFormerMemoryPolicyInput& input,
    const BatchExecutionPlanningContext& context);

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_MEMORY_POLICY_H
