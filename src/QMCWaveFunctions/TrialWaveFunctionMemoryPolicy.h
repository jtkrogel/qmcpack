//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file TrialWaveFunctionMemoryPolicy.h
 * @brief Pure rank-local memory accounting for TrialWaveFunction aggregate state.
 */

#ifndef QMCPLUSPLUS_TRIAL_WAVEFUNCTION_MEMORY_POLICY_H
#define QMCPLUSPLUS_TRIAL_WAVEFUNCTION_MEMORY_POLICY_H

#include "Utilities/BatchExecutionMemory.h"

#include <cstddef>
#include <string_view>
#include <vector>

namespace qmcplusplus
{

/** Stable participant identity reserved for the TrialWaveFunction aggregate owner. */
inline constexpr std::string_view TRIAL_WAVEFUNCTION_MEMORY_PARTICIPANT_ID = "twf/aggregate";

/** Actual build-dependent widths at the TrialWaveFunction ownership boundary. */
struct TrialWaveFunctionMemoryTypeSizes
{
  std::size_t value_type                       = 0;
  std::size_t particle_gradient_element        = 0;
  std::size_t particle_laplacian_element       = 0;
  std::size_t wavefunction_component_reference = 0;
  std::size_t particle_gradient_reference      = 0;
  std::size_t particle_laplacian_reference     = 0;
  std::size_t parameter_derivative_view        = 0;
  std::size_t evaluation_stamp                 = 0;
  std::size_t transaction_flag                 = 0;
};

/** Construct exact type widths without coupling the pure policy to TWF headers. */
template<class ValueType,
         class ParticleGradientElement,
         class ParticleLaplacianElement,
         class WaveFunctionComponentReference,
         class ParticleGradientReference,
         class ParticleLaplacianReference,
         class ParameterDerivativeView,
         class EvaluationStamp,
         class TransactionFlag>
constexpr TrialWaveFunctionMemoryTypeSizes makeTrialWaveFunctionMemoryTypeSizes() noexcept
{
  return {sizeof(ValueType),
          sizeof(ParticleGradientElement),
          sizeof(ParticleLaplacianElement),
          sizeof(WaveFunctionComponentReference),
          sizeof(ParticleGradientReference),
          sizeof(ParticleLaplacianReference),
          sizeof(ParameterDerivativeView),
          sizeof(EvaluationStamp),
          sizeof(TransactionFlag)};
}

/** Explicit aggregate-path coverage claims; every claim is fail-closed by default. */
struct TrialWaveFunctionMemoryAccountingClaims
{
  bool runtime_preflight_and_unsupported_fail_closed = false;
  bool clone_state                                  = false;
  bool reference_views                              = false;
  bool sole_component_dispatch                      = false;
  bool scalar_value_forwarding                      = false;
  bool full_vgl_and_selected_move_transaction       = false;
  bool active_gradient_transaction                  = false;
  bool parameter_derivative_forwarding              = false;
  bool weighted_ecp_outer_scratch                   = false;
  bool weighted_ecp_derivative_staging              = false;
  bool weighted_ecp_metadata                        = false;

  /** Return complete test evidence; production must enable claims only after migration. */
  static constexpr TrialWaveFunctionMemoryAccountingClaims complete() noexcept
  {
    return {true, true, true, true, true, true, true, true, true, true, true};
  }
};

/** Evidence required to rely on the only child component for atomic publication. */
struct TrialWaveFunctionSoleChildEvidence
{
  std::size_t owner_multiplicity = 0;
  bool fully_accounted           = false;
  bool atomic_publication        = false;
};

/** Candidate-independent facts used to estimate one aggregate TWF owner. */
struct TrialWaveFunctionMemoryPolicyInput
{
  TrialWaveFunctionMemoryTypeSizes type_sizes;
  TrialWaveFunctionMemoryAccountingClaims accounting_claims;
  TrialWaveFunctionSoleChildEvidence sole_child;

  // Target-coordinate support belongs to the planning context so missing
  // workload evidence cannot be mistaken for a non-spinor default here.
  std::size_t component_count  = 0;
  bool use_tasking             = false;
  bool fallback_path_reachable = false;
};

/** Exact clone-owned accepted and proposed aggregate gradient/laplacian bytes. */
struct TrialWaveFunctionCloneStorageRequirement
{
  std::size_t accepted_gradients  = 0;
  std::size_t proposed_gradients  = 0;
  std::size_t accepted_laplacians = 0;
  std::size_t proposed_laplacians = 0;

  std::size_t totalBytes() const;
};

/** Exact crowd-resource bytes grouped by their runtime purpose. */
struct TrialWaveFunctionResourceStorageRequirement
{
  std::size_t component_reference_slots           = 0;
  std::size_t aggregate_gradient_reference_slots  = 0;
  std::size_t aggregate_laplacian_reference_slots = 0;

  std::size_t weighted_ecp_private_ratios = 0;
  std::size_t weighted_ecp_total_weights  = 0;

  std::size_t weighted_parameter_derivative_delta = 0;
  std::size_t weighted_parameter_derivative_views = 0;
  std::size_t weighted_value_stamps                = 0;
  /// Shared abort/accept transaction flags needed by selected moves and weighted ECP.
  std::size_t transaction_flags                    = 0;

  std::size_t logicalInputOutputBytes() const;
  std::size_t outerTileScratchBytes() const;
  std::size_t publicationStagingBytes() const;
  std::size_t ecpMetadataBytes() const;
  std::size_t totalBytes() const;
};

/** Exact allocation and accounting target for one rank-local TWF crowd. */
struct TrialWaveFunctionCrowdMemoryPlan
{
  std::size_t initial_walkers            = 0;
  std::size_t reserve_walkers            = 0;
  std::size_t particle_count             = 0;
  std::size_t component_count            = 0;
  std::size_t ecp_outer_capacity         = 0;
  std::size_t parameter_derivative_width = 0;
  bool weighted_ecp_required             = false;

  TrialWaveFunctionCloneStorageRequirement clone_storage;
  TrialWaveFunctionResourceStorageRequirement resource_storage;
  BatchMemoryEstimate expected_clone_storage;
  BatchMemoryEstimate expected_resource_storage;
  BatchMemoryEstimate expected_storage;
};

/** Select reserve crowd capacities, falling back to the initial topology. */
const std::vector<std::size_t>& trialWaveFunctionReserveWalkersPerCrowd(
    const BatchExecutionTopology& topology);

/** Return candidate-independent aggregate maxima; ECP_OUTER remains NLPP-owned. */
BatchTileCapacities trialWaveFunctionBatchLogicalMaximum(
    const BatchExecutionWorkloadContext& context);

/** Describe aggregate clone storage for a complete crowd reserve. */
TrialWaveFunctionCloneStorageRequirement trialWaveFunctionCloneStorageRequirement(
    std::size_t reserve_walkers,
    std::size_t particle_count,
    const TrialWaveFunctionMemoryTypeSizes& type_sizes);

/** Describe aggregate crowd-resource storage for one selected candidate. */
TrialWaveFunctionResourceStorageRequirement trialWaveFunctionResourceStorageRequirement(
    std::size_t reserve_walkers,
    std::size_t component_count,
    std::size_t ecp_outer_capacity,
    std::size_t parameter_derivative_width,
    bool weighted_ecp_required,
    const TrialWaveFunctionMemoryTypeSizes& type_sizes);

/** Construct exact crowd plans in stable topology order. */
std::vector<TrialWaveFunctionCrowdMemoryPlan> makeTrialWaveFunctionCrowdMemoryPlans(
    const TrialWaveFunctionMemoryPolicyInput& input,
    const BatchExecutionPlanningContext& context);

/** Estimate one exact rank-local TrialWaveFunction aggregate contribution. */
BatchMemoryContribution estimateTrialWaveFunctionBatchMemory(
    const TrialWaveFunctionMemoryPolicyInput& input,
    const BatchExecutionPlanningContext& context);

} // namespace qmcplusplus

#endif // QMCPLUSPLUS_TRIAL_WAVEFUNCTION_MEMORY_POLICY_H
