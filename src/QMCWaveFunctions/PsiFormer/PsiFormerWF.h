//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerWF.h
 * @brief QMCPACK WaveFunctionComponent interface to the native PsiFormer
 * evaluator.
 */
#ifndef QMCPLUSPLUS_PSIFORMERWF_H
#define QMCPLUSPLUS_PSIFORMERWF_H

#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "QMCWaveFunctions/Optimization/StructuredParameterProvider.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerStorageRequirements.h"
#include "ResourceHandle.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pf
{
/// Native PsiFormer evaluator shared by QMCPACK component clones.
struct PsiFormer;

/// Clone-local fixed-storage workspace for direct value evaluation.
class DirectValueWorkspace;

/// Clone-local forward tape and adjoints for direct parameter-score evaluation.
class DirectScoreWorkspace;

/// Clone-local trace-jet tape and adjoints for direct kinetic response.
class DirectKineticWorkspace;

/// Clone-local trace-jet storage for direct spatial evaluation.
class DirectSpatialWorkspace;

/// Crowd- or clone-owned batch-capacity storage for direct evaluations.
class DirectBatchWorkspace;

/// Native high-level observables returned by one model evaluation.
struct Result;

/// Scalar result returned by the fixed-storage direct value executor.
struct DirectValueResult;

/// Non-owning spatial result backed by one clone-local direct workspace.
struct DirectSpatialResultView;

/// Non-owning result of one direct parameter-score evaluation.
struct DirectScoreResult;
} // namespace pf

namespace qmcplusplus
{
namespace psiformer
{
/// Dependency-light result of deterministic native PsiFormer initialization.
struct InitializedPsiFormerParameters;

/// Pure input used to estimate this component family's planned batch storage.
struct PsiFormerMemoryPolicyInput;

/// Exact allocation record retained by one prepared PsiFormer crowd resource.
struct PsiFormerCrowdMemoryPlan;
}

/// Shared, versioned native model state used by all clones of one component.
class PsiFormerSharedState;

/// Implementation-only guard binding one complete operation to one model version.
class PsiFormerReadTransaction;

/// Implementation-only model/optimizer snapshot using the fixed lock order.
class PsiFormerDerivativeReadTransaction;

/// Shared optimizer registration and flat-index mapping for one clone family.
class PsiFormerOptimizationMetadata;

namespace testing
{
/** Describe clone-local native evaluator scratch without exposing implementation
 * workspace types through the public wavefunction interface. */
struct PsiFormerWorkspaceDiagnostics
{
  bool owns_value_workspace          = false;
  bool owns_full_spatial_workspace   = false;
  bool owns_active_spatial_workspace = false;
  bool owns_batch_workspace          = false;
  bool owns_score_workspace          = false;
  bool owns_kinetic_workspace        = false;
  bool has_prepared_clone_plan       = false;

  std::size_t value_bytes                       = 0;
  std::size_t full_spatial_bytes                = 0;
  std::size_t active_spatial_bytes              = 0;
  std::size_t batch_bytes                       = 0;
  std::size_t score_bytes                       = 0;
  std::size_t kinetic_bytes                     = 0;
  std::size_t total_log_gradient_bytes          = 0;
  std::size_t scalar_value_publication_bytes    = 0;
  std::size_t accepted_spatial_bytes            = 0;
  std::size_t proposed_spatial_bytes            = 0;
  std::size_t batch_storage_fingerprint         = 0;
  const void* batch_workspace_identity          = nullptr;
  const void* scalar_value_publication_identity = nullptr;
  std::size_t scalar_value_publication_size = 0;
  std::size_t scalar_value_publication_capacity = 0;

  bool prepared_scalar_value_compatibility = false;
  const void* prepared_batch_workspace_identity = nullptr;
  std::size_t prepared_batch_storage_fingerprint = 0;
  std::size_t prepared_batch_bytes = 0;
  const void* prepared_scalar_value_publication_identity = nullptr;
  std::size_t prepared_scalar_value_publication_size = 0;
  std::size_t prepared_scalar_value_publication_capacity = 0;
  /// Walker-record layout evidence for caller-owned storage; never component heap bytes.
  pf::WalkerBufferLayout prepared_walker_buffer_layout;

  /** Return all explicitly accounted clone-local evaluator scratch bytes.
   * The prepared walker-record layout is evidence for caller-owned storage and
   * deliberately contributes no component-owned bytes here.
   */
  std::size_t accountedBytes() const noexcept
  {
    return value_bytes + full_spatial_bytes + active_spatial_bytes + batch_bytes +
        score_bytes + kinetic_bytes + total_log_gradient_bytes +
        scalar_value_publication_bytes;
  }

  /** Return the number of independently owned native evaluator workspaces.
   * The prepared walker layout describes no component-owned workspace.
   */
  std::size_t ownedWorkspaceCount() const noexcept
  {
    return static_cast<std::size_t>(owns_value_workspace) +
        static_cast<std::size_t>(owns_full_spatial_workspace) +
        static_cast<std::size_t>(owns_active_spatial_workspace) +
        static_cast<std::size_t>(owns_batch_workspace) +
        static_cast<std::size_t>(owns_score_workspace) +
        static_cast<std::size_t>(owns_kinetic_workspace);
  }
};

/** Describe optimizer metadata ownership without exposing the shared metadata
 * implementation through the public wavefunction interface. */
struct PsiFormerOptimizationMetadataDiagnostics
{
  const void* identity                    = nullptr;
  std::size_t shared_owner_count          = 0;
  std::size_t selected_index_count        = 0;
  std::size_t shared_variable_count       = 0;
  std::size_t mapped_variable_count       = 0;
  std::size_t inherited_variable_count    = 0;
};

/// Public test spelling of the scalar representation selected for ratio scratch.
enum class PsiFormerRatioArenaKind : std::uint8_t
{
  NONE,
  PSI_VALUE,
  LOG_VALUE
};

/** Describe both typed ratio-arena candidates and their immutable preparation
 * records without granting tests mutable access to either allocation. */
struct PsiFormerRatioArenaDiagnostics
{
  PsiFormerRatioArenaKind kind          = PsiFormerRatioArenaKind::NONE;
  PsiFormerRatioArenaKind prepared_kind = PsiFormerRatioArenaKind::NONE;
  const void* psi_value_data          = nullptr;
  const void* log_value_data          = nullptr;
  const void* prepared_psi_value_data = nullptr;
  const void* prepared_log_value_data = nullptr;
  std::size_t psi_value_size          = 0;
  std::size_t log_value_size          = 0;
  std::size_t psi_value_capacity      = 0;
  std::size_t log_value_capacity      = 0;
  std::size_t prepared_psi_value_size     = 0;
  std::size_t prepared_log_value_size     = 0;
  std::size_t prepared_psi_value_capacity = 0;
  std::size_t prepared_log_value_capacity = 0;

  /// Compare the complete read-only identity and extent snapshot.
  bool operator==(const PsiFormerRatioArenaDiagnostics& other) const noexcept
  {
    return kind == other.kind && prepared_kind == other.prepared_kind &&
        psi_value_data == other.psi_value_data &&
        log_value_data == other.log_value_data &&
        prepared_psi_value_data == other.prepared_psi_value_data &&
        prepared_log_value_data == other.prepared_log_value_data &&
        psi_value_size == other.psi_value_size &&
        log_value_size == other.log_value_size &&
        psi_value_capacity == other.psi_value_capacity &&
        log_value_capacity == other.log_value_capacity &&
        prepared_psi_value_size == other.prepared_psi_value_size &&
        prepared_log_value_size == other.prepared_log_value_size &&
        prepared_psi_value_capacity == other.prepared_psi_value_capacity &&
        prepared_log_value_capacity == other.prepared_log_value_capacity;
  }
};

/** Describe one acquired crowd resource without exposing mutable workspace
 * storage or implementation types. */
struct PsiFormerCrowdWorkspaceDiagnostics
{
  const void* shared_model_identity      = nullptr;
  const void* resource_identity          = nullptr;
  const void* batch_workspace_identity   = nullptr;
  const void* score_workspace_identity   = nullptr;
  const void* kinetic_workspace_identity = nullptr;
  std::uint64_t persistent_model_identity = 0;
  std::size_t parameter_version           = 0;
  std::size_t batch_bytes                 = 0;
  std::size_t score_bytes                 = 0;
  std::size_t kinetic_bytes               = 0;
  std::size_t transient_bytes             = 0;
  /// Numerical-result generation; lifecycle-only calls must not advance it.
  std::size_t successful_batch_generation = 0;
  /// Sparse counters from the most recent completed direct crowd batch.
  std::size_t reference_configurations       = 0;
  std::size_t replacement_configurations     = 0;
  std::size_t reference_evaluations          = 0;
  std::size_t dense_coordinate_bytes_avoided = 0;
  /// Logical score configurations from the most recent completed reduction.
  std::size_t weighted_reference_configurations   = 0;
  std::size_t weighted_replacement_configurations = 0;
  std::size_t weighted_active_parameters           = 0;
  /// Capacity of the compact active-walker by active-parameter staging buffer.
  std::size_t weighted_derivative_staging_bytes = 0;
  /// Hard-plan provenance published only after exact crowd preparation.
  bool has_expected_plan                    = false;
  bool has_prepared_plan                    = false;
  const void* prepared_plan_identity        = nullptr;
  std::uint64_t prepared_plan_fingerprint   = 0;
  std::string participant_id;
  std::size_t prepared_crowd_index          = 0;
  std::size_t initial_walker_capacity       = 0;
  std::size_t reserve_walker_capacity       = 0;
  std::size_t prepared_storage_fingerprint  = 0;
  std::size_t current_storage_fingerprint   = 0;
  PsiFormerRatioArenaDiagnostics ratio_arena;
  std::array<std::size_t, 22> logical_sizes = {};
  BatchMemoryEstimate expected_resource_storage;
  BatchMemoryEstimate actual_resource_storage;
  std::array<std::string, 4> backend_modes;

  /// Return all explicitly accounted numeric storage owned by the resource.
  std::size_t accountedBytes() const noexcept
  { return batch_bytes + score_bytes + kinetic_bytes + transient_bytes; }
};

/// Test-only copy of the live selected-proposal compaction prefixes.
struct PsiFormerSelectedProposalMapDiagnostics
{
  std::vector<std::size_t> batch_slots;
  std::vector<std::size_t> walker_indices;
};

/// Test-only accessor for bounded crowd-workspace ownership diagnostics.
class TestPsiFormerWF;

/// Test-only accessor for flattened virtual-batch state-isolation diagnostics.
class TestPsiFormerVirtualBatch;
}

/**
 * Wavefunction component for an imported or internally initialized PsiFormer model.
 *
 * Parameters remain fixed unless optimization is explicitly enabled.
 * Optimization may register selected canonical flat indices or the complete
 * network in that same ordering. Full-network resets use the native complete
 * vector path to avoid sorting and revalidating millions of canonical indices.
 *
 * Clones share one versioned native model behind a reader/writer lock, while
 * accepted and proposed move state remains clone-local. Object-specific VP
 * records persist the complete model independently of the selected scalar list.
 */
class PsiFormerWF : public WaveFunctionComponent,
                    public OptimizableObject,
                    public wftrain::StructuredParameterProvider
{
public:
  /// Load a native model and optionally expose selected flat parameters to QMCPACK.
  PsiFormerWF(std::string name,
              std::string parameters,
              std::string configuration,
              bool optimize = false,
              std::vector<std::size_t> selected_flat_indices = {},
              bool optimize_all = false,
              std::string optimized_parameter_export = {});

  /** Construct a self-contained model from initialized parameters and QMCPACK
   * electron/ion particle sets, without reading a model or configuration file. */
  PsiFormerWF(std::string name,
              psiformer::InitializedPsiFormerParameters initialized_parameters,
              const ParticleSet& electrons,
              const ParticleSet& ions,
              bool optimize = false,
              std::vector<std::size_t> selected_flat_indices = {},
              bool optimize_all = false,
              std::string optimized_parameter_export = {});

  /// Copy accepted state and optimizer mappings while dropping any in-flight proposal.
  PsiFormerWF(const PsiFormerWF& other);

  /// Destroy the clone-local direct workspace after its complete type is visible.
  ~PsiFormerWF() override;

  /// Return the component name used by QMCPACK diagnostics.
  std::string getClassName() const override { return "PsiFormerWF"; }

  /// Mark the sign-changing PsiFormer ansatz as fermionic.
  bool isFermionic() const override { return true; }

  /// Report whether this component explicitly exposes selected parameters.
  bool isOptimizable() const override;

  /// Expose this component to the structured route without scalar registration.
  wftrain::StructuredParameterProvider* structuredParameterProvider() noexcept override { return this; }

  /// Return native tensor metadata in canonical DeepQMC flat order.
  const wftrain::StructuredParameterSchema& parameterSchema() const noexcept override;

  /// Copy the complete native vector under the shared model lock.
  wftrain::StructuredParameterSnapshot snapshotParameters() const override;

  /// Construct an O(P)+O(B) score-product provider bound to this model version.
  std::unique_ptr<wftrain::StreamingDerivativeOperator> makeStreamingDerivativeOperator(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      std::size_t batch_ordinal,
      std::size_t sample_offset,
      std::size_t maximum_parameter_chunk_size) const override;

  /// Atomically publish one complete, version-matched native vector.
  std::size_t publishParameters(const wftrain::StructuredParameterSnapshot& candidate,
                                std::size_t expected_version) override;

  /// Add this component's optimization object when selected-parameter optimization is enabled.
  void extractOptimizableObjectRefs(UniqueOptObjRefs& opt_obj_refs) override;

  /// Insert selected PsiFormer parameters into QMCPACK's global active-variable set.
  void checkInVariablesExclusive(OptVariables& active) override;

  /// Map component-local selected parameters to global active indices.
  void checkOutVariables(const OptVariables& active) override;

  /// Apply active values to the synchronized native flat parameter storage.
  void resetParametersExclusive(const OptVariables& active) override;

  /// Store the complete model state and selection in the object-specific VP group.
  void writeVariationalParameters(hdf_archive& output) override;

  /// Restore and validate the authoritative complete model state from a VP group.
  void readVariationalParameters(hdf_archive& input) override;

  /// Return the shared model version for diagnostics and synchronization tests.
  std::size_t parameterVersion() const;

  /// Export current parameters in the flat DeepQMC-compatible HDF5 format.
  void exportParameters(const std::string& path) const;

  /// Validate electron spins and ionic geometry/effective charges against the export.
  void validateSystem(const ParticleSet& electrons, const ParticleSet& ions, const std::string& system_kind);

  /// Evaluate log(psi), gradients, and logarithmic Laplacians for a full configuration.
  LogValue evaluateLog(const ParticleSet& particles,
                       ParticleSet::ParticleGradient& gradient,
                       ParticleSet::ParticleLaplacian& laplacian) override;

  /// Evaluate complete VGL data through one direct crowd batch boundary.
  void mw_evaluateLog(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                      const RefVectorWithLeader<ParticleSet>& p_list,
                      const RefVector<ParticleSet::ParticleGradient>& gradient_list,
                      const RefVector<ParticleSet::ParticleLaplacian>& laplacian_list) const override;

  /// Reuse the same direct full-spatial batch for evaluateGL callers.
  void mw_evaluateGL(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                     const RefVectorWithLeader<ParticleSet>& p_list,
                     const RefVector<ParticleSet::ParticleGradient>& gradient_list,
                     const RefVector<ParticleSet::ParticleLaplacian>& laplacian_list,
                     bool from_scratch) const override;

  /// PsiFormer can evaluate one atomic selected-electron proposal per walker.
  bool supportsMultiParticleMoves() const noexcept override { return true; }

  /// Evaluate complete proposed VGL state from descriptor-owned absolute positions.
  void mw_evaluateMultiParticleMove(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const MCMultiParticleMoves<CoordsType::POS>& moves,
      std::vector<LogValue>& log_ratios,
      const RefVector<ParticleSet::ParticleGradient>& proposed_gradient_list,
      const RefVector<ParticleSet::ParticleLaplacian>& proposed_laplacian_list) const override;

  /// Atomically promote or discard each clone's complete selected-electron proposal.
  void mw_accept_rejectMultiParticleMove(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const MCMultiParticleMoves<CoordsType::POS>& moves,
      const std::vector<bool>& accepted) const override;

  /// Retain the inherited scalar recompute route only without a hard batch plan.
  void recompute(const ParticleSet& particles) override;

  /// Refresh selected accepted values without serial component dispatch.
  void mw_recompute(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                    const RefVectorWithLeader<ParticleSet>& p_list,
                    const std::vector<bool>& recompute) const override;

  /// Validate planned group preparation as an allocation-free component no-op.
  void prepareGroup(ParticleSet& particles, int group_index) override;

  /// Validate planned crowd group preparation without inherited lane dispatch.
  void mw_prepareGroup(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                       const RefVectorWithLeader<ParticleSet>& p_list,
                       int group_index) const override;

  /// Commit the wavefunction state cached by the most recent proposed move.
  void acceptMove(ParticleSet& particles, int particle_index, bool safe_to_delay = false) override;

  /// Discard the wavefunction state cached for a rejected move.
  void restore(int particle_index) override;

  /// Evaluate psi(new)/psi(old) for one active-particle proposal.
  PsiValue ratio(ParticleSet& particles, int particle_index) override;

  /// Evaluate one common electron proposal for every walker in a crowd.
  void mw_calcRatio(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                    const RefVectorWithLeader<ParticleSet>& p_list,
                    int particle_index,
                    std::vector<PsiValue>& ratios) const override;

  /// Evaluate the logarithmic gradient of one electron at the accepted configuration.
  GradType evalGrad(ParticleSet& particles, int particle_index) override;

  /// Preserve inherited spin-independent gradient semantics only without a hard plan.
  GradType evalGradWithSpin(ParticleSet& particles,
                            int particle_index,
                            ComplexType& spin_gradient) override;

  /// Reject ionic gradients because imported PsiFormer models have fixed nuclei.
  GradType evalGradSource(ParticleSet& particles, ParticleSet& source, int particle_index) override;

  /// Reject force estimators requiring ionic derivatives of electron VGL data.
  GradType evalGradSource(ParticleSet& particles,
                          ParticleSet& source,
                          int particle_index,
                          TinyVector<ParticleSet::ParticleGradient, OHMMS_DIM>& grad_grad,
                          TinyVector<ParticleSet::ParticleLaplacian, OHMMS_DIM>& lapl_grad) override;

  /// Evaluate accepted active-electron gradients for a crowd.
  void mw_evalGrad(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                   const RefVectorWithLeader<ParticleSet>& p_list,
                   int particle_index,
                   std::vector<GradType>& gradients) const override;

  /// Preserve inherited POS_SPIN gradient dispatch only without a hard plan.
  void mw_evalGradWithSpin(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      int particle_index,
      std::vector<GradType>& gradients,
      std::vector<ComplexType>& spin_gradients) const override;

  /// Evaluate a proposed ratio and active-electron logarithmic gradient together.
  PsiValue ratioGrad(ParticleSet& particles, int particle_index, GradType& gradient) override;

  /// Preserve inherited spin-independent ratio-gradient semantics only without a hard plan.
  PsiValue ratioGradWithSpin(ParticleSet& particles,
                             int particle_index,
                             GradType& gradient,
                             ComplexType& spin_gradient) override;

  /// Evaluate proposal ratios and active-electron gradients in one crowd traversal.
  void mw_ratioGrad(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                    const RefVectorWithLeader<ParticleSet>& p_list,
                    int particle_index,
                    std::vector<PsiValue>& ratios,
                    std::vector<GradType>& gradients) const override;

  /// Preserve inherited POS_SPIN ratio-gradient dispatch only without a hard plan.
  void mw_ratioGradWithSpin(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      int particle_index,
      std::vector<PsiValue>& ratios,
      std::vector<GradType>& gradients,
      std::vector<ComplexType>& spin_gradients) const override;

  /// Commit or discard each walker's independently cached proposal state.
  void mw_accept_rejectMove(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                            const RefVectorWithLeader<ParticleSet>& p_list,
                            int particle_index,
                            const std::vector<bool>& is_accepted,
                            bool safe_to_delay = false) const override;

  /// Validate planned scalar completion as an allocation-free component no-op.
  void completeUpdates() override;

  /// Validate planned crowd completion from retained acquisition provenance.
  void mw_completeUpdates(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const override;

  /// Evaluate replacement of each electron by ParticleSet::getActivePos() without state changes.
  void evaluateRatiosAlltoOne(ParticleSet& particles, std::vector<ValueType>& ratios) override;

  /// Evaluate every nonlocal-pseudopotential virtual-move ratio by full model reevaluation.
  void evaluateRatios(const VirtualParticleSet& virtual_particles, std::vector<ValueType>& ratios) override;

  /// Flatten ragged virtual-position lists into one direct crowd batch.
  void mw_evaluateRatios(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                         const RefVectorWithLeader<const VirtualParticleSet>& virtual_particle_list,
                         std::vector<std::vector<ValueType>>& ratios) const override;

  /** Evaluate a flattened ragged virtual batch from sparse references and
   * replacements while returning the shared parameter-version stamp. */
  EvaluationStamp mw_evaluateVirtualRatios(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
      const VirtualParticleBatch& batch,
      std::vector<ValueType>& ratios) const override;

  /// Reject crowd spin-orbit ratios before ignoring any spin quadrature multipliers.
  void mw_evaluateSpinorRatios(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<const VirtualParticleSet>& virtual_particle_list,
      const RefVector<std::pair<ValueVector, ValueVector>>& spinor_multiplier_list,
      std::vector<std::vector<ValueType>>& ratios) const override;

  /// Add virtual-move changes in logarithmic parameter derivatives for the nonlocal ECP operator.
  void evaluateDerivRatios(const VirtualParticleSet& virtual_particles,
                           const OptVariables& optvars,
                           std::vector<ValueType>& ratios,
                           Matrix<ValueType>& derivative_ratios) override;

  /// Reduce weighted virtual score differences without a knot-by-parameter matrix.
  void evaluateDerivRatiosWeighted(const VirtualParticleSet& virtual_particles,
                                   const OptVariables& optvars,
                                   const std::vector<ValueType>& total_weights,
                                   ParameterDerivativeView weighted_derivatives) override;

  /** Reduce one flattened virtual batch with one reference score per active
   * walker and compact active-parameter staging. */
  EvaluationStamp mw_evaluateVirtualDerivRatiosWeighted(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
      const VirtualParticleBatch& batch,
      const OptVariables& optvars,
      const std::vector<ValueType>& total_weights,
      const std::vector<ParameterDerivativeView>& weighted_derivatives) const override;

  /// Reduce ragged walker rows serially through one resource-owned direct score tape.
  void mw_evaluateDerivRatiosWeighted(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
      const OptVariables& optvars,
      const RefVector<const std::vector<ValueType>>& total_weights,
      const std::vector<ParameterDerivativeView>& weighted_derivatives) const override;

  /// Reject spin-orbit virtual moves because the imported ansatz has fixed discrete spin labels.
  void evaluateSpinorRatios(const VirtualParticleSet& virtual_particles,
                            const std::pair<ValueVector, ValueVector>& spinor_multiplier,
                            std::vector<ValueType>& ratios) override;

  /// Reject spin-orbit derivative ratios at the component interface.
  void evaluateSpinorDerivRatios(const VirtualParticleSet& virtual_particles,
                                 const std::pair<ValueVector, ValueVector>& spinor_multiplier,
                                 const OptVariables& optvars,
                                 std::vector<ValueType>& ratios,
                                 Matrix<ValueType>& derivative_ratios) override;

  /// Reserve the versioned accepted-state record in a walker's persistent buffer.
  void registerData(ParticleSet& particles, WFBufferType& buffer) override;

  /// Reuse or refresh accepted VGL state and write a complete persistent record.
  LogValue updateBuffer(ParticleSet& particles, WFBufferType& buffer, bool from_scratch = false) override;

  /// Restore a matching accepted-state record or invalidate a stale one.
  void copyFromBuffer(ParticleSet& particles, WFBufferType& buffer) override;

  /// Declare the full-spatial initialization path required by this component.
  void contributeBatchExecutionRequirements(BatchExecutionRequirements& requirements) const override;

  /// Return shape- and topology-derived logical maxima without allocating scratch.
  BatchTileCapacities batchExecutionLogicalMaximum(
      const BatchExecutionWorkloadContext& context) const override;

  /// Estimate exact rank-local ownership while execution storage remains fail-closed.
  BatchMemoryContribution estimateBatchExecutionMemory(
      const BatchExecutionPlanningContext& context) const override;

  /// Check the retained participant identity without rebuilding plan evidence.
  bool hasBatchExecutionPlanBinding(
      const BatchExecutionParticipantPlan& plan) const noexcept override;

  /// Check that exact clone-local preparation was published for one plan view.
  bool hasPreparedBatchExecutionClone(
      const BatchExecutionParticipantPlan& plan) const noexcept override;

  /// Validate an immutable participant view and its exact selected evidence.
  void validateBatchExecutionPlanBinding(
      const BatchExecutionParticipantPlan& plan) const override;

  /// Publish a validated participant view, including an explicit null clearing view.
  void bindBatchExecutionPlan(BatchExecutionParticipantPlan plan) noexcept override;

  /// Prepare exact clone-owned state and any admitted scalar VALUE workspace.
  void prepareBatchExecutionClone(const BatchExecutionParticipantPlan& plan) override;

  /// Add one cloneable crowd workspace resource to the collection.
  void createResource(ResourceCollection& collection) const override;

  /// Acquire exclusive crowd ownership of the batch-capacity workspace.
  void acquireResource(ResourceCollection& collection,
                       const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const override;

  /// Return the crowd workspace and invalidate the leader's handle.
  void releaseResource(ResourceCollection& collection,
                       const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const override;

  /// Add logarithmic and component kinetic-energy derivatives for active parameters.
  void evaluateDerivatives(ParticleSet& particles,
                           const OptVariables& optvars,
                           Vector<ValueType>& dlogpsi,
                           Vector<ValueType>& dhpsioverpsi) override;

  /// Preserve one output row per walker under component-major optimizer dispatch.
  void mw_evaluateParameterDerivatives(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const OptVariables& optvars,
      RecordArray<ValueType>& dlogpsi,
      RecordArray<ValueType>& dhpsioverpsi) const override;

  /// Add only logarithmic wavefunction derivatives for active parameters.
  void evaluateDerivativesWF(ParticleSet& particles,
                             const OptVariables& optvars,
                             Vector<ValueType>& dlogpsi) override;

  /// Evaluate score-only rows serially through one resource-owned direct score tape.
  void mw_evaluateParameterDerivativesWF(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const OptVariables& optvars,
      RecordArray<ValueType>& dlogpsi) const override;

  /// Clone per-component move state while sharing the synchronized native model.
  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet& particles) const override;

private:
  struct PsiFormerMultiWalkerResource;

  /// Identify the exact planned entry point whose common runtime contract is checked.
  enum class PlannedOperation
  {
    FULL_VGL,
    RECOMPUTE_VALUE,
    CALC_RATIO,
    ACTIVE_GRADIENT,
    RATIO_GRADIENT,
    ACCEPT_REJECT_VALUE,
    SINGLE_CANCEL,
    SELECTED_PROPOSE,
    SELECTED_RESOLVE,
    SELECTED_CANCEL,
    ECP_VALUE,
    ECP_WEIGHTED_SCORE,
    SCORE_DERIVATIVES,
    KINETIC_DERIVATIVES,
    SCALAR_VALUE_COMPATIBILITY,
    BUFFER_READ,
    BUFFER_WRITE,
    PREPARE_GROUP,
    COMPLETE_UPDATES
  };

  /// State whether a planned operation permits or requires clone-local proposal state.
  enum class ProposalRequirement
  {
    NONE,
    ABSENT,
    SINGLE_PENDING,
    SELECTED_PENDING
  };

  /// Identify the exact evaluator that published clone-local proposal state.
  enum class ProposalOrigin : std::uint8_t
  {
    NONE                         = 0,
    SCALAR_RATIO_VALUE           = 1,
    SCALAR_RATIO_GRADIENT_ACTIVE = 2,
    MW_CALC_RATIO_VALUE          = 3,
    MW_RATIO_GRADIENT_ACTIVE     = 4,
    MW_SELECTED_FULL_VGL         = 5
  };
  static_assert(sizeof(ProposalOrigin) == sizeof(std::uint8_t));

  /** Carry allocation-free runtime extents and proposal identity into common
   * planned-operation validation. */
  struct PlannedRuntimeRequest
  {
    PlannedOperation operation;
    std::size_t live_walkers         = 0;
    std::size_t dense_configurations = 0;
    std::size_t sparse_references    = 0;
    std::size_t sparse_replacements  = 0;
    std::size_t selected_parameters  = 0;
    std::size_t derivative_width     = 0;
    std::optional<std::size_t> active_electron;
    std::optional<std::uint64_t> descriptor_fingerprint;
    std::optional<std::size_t> expected_proposal_version;
    std::optional<ProposalOrigin> expected_proposal_origin;
    std::optional<std::uint64_t> expected_single_transaction_fingerprint;
  };

  /** Return read-only bindings proved by preflight without changing logical
   * extents, parameter versions, proposals, or caller state. */
  struct PlannedRuntimeAccess
  {
    PsiFormerMultiWalkerResource& resource;
    const BatchExecutionParticipantPlan& participant;
    const psiformer::PsiFormerCrowdMemoryPlan& crowd;
    std::size_t storage_fingerprint;
    std::optional<std::uint64_t> single_transaction_fingerprint;
    std::optional<std::uint64_t> selected_transaction_fingerprint;
  };

  /// Identify one clone-local scalar VALUE compatibility call family.
  enum class PlannedScalarValueOperation
  {
    ALL_TO_ONE,
    VIRTUAL_PARTICLE_VALUE
  };

  /// Carry immutable scalar inputs and caller-owned output evidence into preflight.
  struct PlannedScalarValueRequest
  {
    PlannedScalarValueOperation operation;
    const ParticleSet* reference                = nullptr;
    const VirtualParticleSet* virtual_particles = nullptr;
    std::size_t configuration_count             = 0;
    ValueType* output_data                      = nullptr;
    std::size_t output_size                     = 0;
    std::size_t output_capacity                 = 0;
  };

  /// Return exact prepared scalar bindings proved without changing logical state.
  struct PlannedScalarValueAccess
  {
    pf::DirectBatchWorkspace& workspace;
    ValueType* publication;
    const BatchExecutionParticipantPlan& participant;
    std::size_t workspace_fingerprint;
    std::size_t workspace_bytes;
    std::uint64_t input_fingerprint;
    ValueType* output_data;
    std::size_t output_size;
    std::size_t output_capacity;
  };

  /// Select one transient post-evaluation corruption for Phase-B regressions.
  enum class PlannedScalarValueFaultForTesting
  {
    NONE,
    RESULT_OWNER,
    RESULT_GENERATION,
    RESULT_SIZE,
    RESULT_VERSION,
    RESULT_SIGN,
    RESULT_LOG_MAGNITUDE,
    RESULT_RATIO,
    INPUT_FINGERPRINT,
    OUTPUT_IDENTITY,
    WORKSPACE_EVIDENCE,
    PUBLICATION_EVIDENCE
  };

  /// Select one reversible corruption of the prepared scalar capacity record.
  enum class PreparedScalarWorkspaceFaultForTesting
  {
    NONE,
    LOGICAL_CAPACITY,
    TILE_CAPACITY
  };

  /// Identify one planned walker-buffer contract without borrowing crowd state.
  enum class PlannedWalkerBufferOperation
  {
    REGISTER,
    READ,
    WRITE
  };

  /// Select one reversible inspection-only mutation between Phase A and B.
  enum class PlannedWalkerBufferBetweenPhaseFaultForTesting
  {
    NONE,
    STORAGE_POINTER,
    BULK_CURSOR,
    RECORD_CONTENT,
    PARTICLE_INPUT,
    PLAN_BINDING,
    LAYOUT_EVIDENCE,
    PREPARED_STORAGE_EVIDENCE
  };

  /// Select one final buffer-transaction failure after every Phase-B check.
  enum class PlannedWalkerBufferLateFaultForTesting
  {
    NONE,
    REGISTER,
    RESTORE,
    REFRESH
  };

  /// Classify a completely validated persistent component record.
  enum class WalkerBufferRecordClassification : std::uint8_t
  {
    RESTORABLE,
    VALID_ZERO,
    VALID_STALE,
    MALFORMED
  };

  /** Capture both independent PooledMemory cursors and immutable storage evidence.
   * Pointer-derived offsets are populated only after numeric provenance checks.
   * The caller must retain exclusive ownership of this buffer lane while the
   * snapshot is inspected; fingerprints cannot make concurrent mutation safe.
   */
  struct WalkerBufferCursorSnapshot
  {
    PlannedWalkerBufferOperation operation =
        PlannedWalkerBufferOperation::REGISTER;
    const WFBufferType* buffer = nullptr;
    const BatchExecutionPlan* plan_identity = nullptr;
    const char* data = nullptr;
    const FullPrecRealType* scalar_data = nullptr;
    std::size_t size = 0;
    std::size_t capacity = 0;
    std::size_t bulk_cursor = 0;
    std::size_t scalar_cursor = 0;
    std::size_t scalar_offset = 0;
    std::size_t scalar_capacity = 0;
    bool attached_storage = false;
    std::uint64_t plan_fingerprint = 0;
    std::uint64_t layout_fingerprint = 0;
    std::uint64_t configuration_identity = 0;
    std::uint64_t storage_fingerprint = 0;
    std::uint64_t input_fingerprint = 0;
  };

  /** Hold checked nonowning byte ranges and optionally decoded metadata.
   * WRITE uses only the ranges and raw content identity; no pointer in this
   * view is dereferenced as a typed object.
   */
  struct WalkerBufferRecordView
  {
    WalkerBufferCursorSnapshot cursor;
    const char* gradient_data = nullptr;
    const char* laplacian_data = nullptr;
    const char* scalar_data = nullptr;
    std::size_t gradient_payload_bytes = 0;
    std::size_t laplacian_payload_bytes = 0;
    std::size_t scalar_payload_bytes = 0;
    std::size_t next_bulk_cursor = 0;
    std::size_t next_scalar_cursor = 0;
    std::uint64_t magic = 0;
    std::uint64_t schema = 0;
    std::uint64_t requirement = 0;
    std::uint64_t model_identity = 0;
    std::uint64_t parameter_version = 0;
    std::uint64_t configuration_identity = 0;
    std::uint64_t electron_count = 0;
    double sign = 0.0;
    double log_magnitude = 0.0;
    double phase = 0.0;
    std::uint64_t content_fingerprint = 0;
    WalkerBufferRecordClassification classification =
        WalkerBufferRecordClassification::MALFORMED;
  };

  /// Retain the complete no-allocation evidence for one cache-reuse refresh.
  struct WalkerBufferRefreshEvidence
  {
    std::uint64_t input_fingerprint = 0;
    std::size_t parameter_version   = 0;
    LogValue log_value              = LogValue(0);
  };

  /// Select one transient malformed snapshot or nonowning range for unit tests.
  enum class PlannedWalkerBufferFaultForTesting
  {
    NULL_BACKING,
    NULL_SCALAR,
    SIZE_EXCEEDS_CAPACITY,
    ATTACHED_STORAGE,
    MISALIGNED_BACKING,
    SCALAR_BEFORE_BACKING,
    SCALAR_AFTER_BACKING,
    MISALIGNED_SCALAR,
    MISALIGNED_BULK_CURSOR,
    BULK_CURSOR_BEYOND_DOMAIN,
    SCALAR_CURSOR_BEYOND_DOMAIN,
    BULK_CURSOR_OVERFLOW,
    SCALAR_CURSOR_OVERFLOW,
    TRUNCATED_BULK,
    TRUNCATED_SCALAR,
    ACCEPTED_STORAGE_ALIAS,
    PROPOSED_STORAGE_ALIAS,
    PARTICLE_STORAGE_ALIAS
  };

  /// Fixed metadata published by the lifecycle-only selected proposal seam.
  struct PlannedSelectedProposalEvidence
  {
    std::uint64_t transaction_fingerprint;
    std::size_t proposal_version;
  };

  /// Return the exact explicit mode mask assigned to one typed operation.
  static BatchExecutionRequirements plannedOperationRequiredModes(
      PlannedOperation operation) noexcept;

  /// Return the exact proposal-state contract assigned to one typed operation.
  static ProposalRequirement plannedOperationProposalRequirement(
      PlannedOperation operation) noexcept;

  /// Complete common construction once either HDF5 import or internal initialization creates shared state.
  PsiFormerWF(std::string name,
              std::shared_ptr<PsiFormerSharedState> model_state,
              bool optimize,
              std::vector<std::size_t> selected_flat_indices,
              bool optimize_all,
              std::string optimized_parameter_export);

  /// Identify the minimum native products required by one QMCPACK entry point.
  enum class EvaluationPurpose
  {
    VALUE_ONLY,
    FULL_SPATIAL,
    ACTIVE_ELECTRON_GRADIENT,
    SCORE_ONLY,
    SCORE_AND_KINETIC
  };

  /// Rank the products currently valid for the clone-local accepted configuration.
  enum class AcceptedStateRequirement : std::uint64_t
  {
    INVALID      = 0,
    VALUE_ONLY   = 1,
    FULL_SPATIAL = 2
  };

  /// Evaluate through an already-held model transaction without reacquiring its mutex.
  pf::Result evaluatePositionsUnderRead(const PsiFormerReadTransaction& transaction,
                                        const ParticleSet& particles,
                                        int replaced_particle,
                                        const PosType* replacement_position,
                                        EvaluationPurpose purpose,
                                        int active_gradient_particle = -1);

  /// Evaluate and publish one accepted full-VGL state under one model transaction.
  LogValue evaluateLogUnderRead(const PsiFormerReadTransaction& transaction,
                                const ParticleSet& particles,
                                ParticleSet::ParticleGradient& gradients,
                                ParticleSet::ParticleLaplacian& laplacians);

  /// Evaluate a scalar value while the caller retains the model read transaction.
  pf::DirectValueResult evaluateDirectValuePositionsUnderRead(
      const PsiFormerReadTransaction& transaction,
      const ParticleSet& particles,
      int replaced_particle,
      const PosType* replacement_position);

  /// Evaluate a spatial request while the caller retains the model read transaction.
  pf::DirectSpatialResultView evaluateDirectSpatialPositionsUnderRead(
      const PsiFormerReadTransaction& transaction,
      const ParticleSet& particles,
      int replaced_particle,
      const PosType* replacement_position,
      EvaluationPurpose purpose,
      int active_gradient_particle);

  /// Validate one homogeneous clone crowd and return its exclusively acquired resource.
  PsiFormerMultiWalkerResource& requireMultiWalkerResource(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const;

  /** Validate the immutable component, ParticleSet, plan, resource, operation,
   * capacity, and proposal contract before a planned numerical transaction. */
  PlannedRuntimeAccess requirePlannedMultiWalkerOperation(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const PlannedRuntimeRequest& request) const;

  /** Validate one scalar lifecycle no-op without acquiring numerical scratch,
   * synchronizing parameters, or changing proposal state. */
  void requirePlannedScalarLifecycleOperation(
      PlannedOperation operation,
      const ParticleSet& particles,
      std::optional<int> group_index) const;

  /** Validate one acquired lifecycle team using caller lanes for preparation
   * or retained bound ParticleSets for completion. */
  void requirePlannedMultiWalkerLifecycleOperation(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>* p_list,
      PlannedOperation operation,
      std::optional<int> group_index) const;

  /** Check only the immutable plan and ParticleSet preparation evidence needed
   * by lifecycle no-ops, without consulting numerical workspace storage. */
  bool hasPreparedLifecycleClone(
      const BatchExecutionParticipantPlan& plan) const noexcept;

  /// Validate the immutable POS-only particle facts shared by lifecycle hooks.
  void requirePlannedLifecycleParticleSet(
      const ParticleSet& particles,
      const BatchExecutionPlan& plan,
      std::optional<int> group_index) const;

  /// Hash one validated acquired team without allocating or dereferencing scratch.
  std::uint64_t selectedTeamFingerprint(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list) const noexcept;

  /// Domain-separate a selected descriptor from the exact team that owns it.
  std::uint64_t selectedTransactionFingerprint(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      std::uint64_t descriptor_fingerprint) const noexcept;

  /** Bind one planned one-electron proposal to its exact team, producer,
   * parameter version, electron, and ordered accepted/proposed identities. */
  std::uint64_t singleTransactionFingerprint(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      ProposalOrigin origin,
      std::size_t active_electron,
      std::size_t proposal_version) const noexcept;

  /// Reserve one shared one-electron transaction slot without throwing.
  bool tryRegisterPlannedSingleTransaction() const noexcept;

  /// Withdraw one previously registered one-electron transaction without underflow.
  void unregisterPlannedSingleTransaction() const noexcept;

  /** Validate and abandon one planned one-electron proposal, including stale
   * proposals that ordinary version synchronization must not silently clear. */
  void cancelPlannedSingleProposal(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      std::size_t active_electron,
      ProposalOrigin expected_origin,
      std::size_t expected_proposal_version,
      std::uint64_t expected_transaction_fingerprint) const;

  /// Publish lifecycle metadata only; planned selected numerical evaluation is not implemented here.
  PlannedSelectedProposalEvidence publishPlannedSelectedProposalMetadata(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      std::uint64_t descriptor_fingerprint) const;

  /// Reserve one shared selected-transaction slot without throwing.
  bool tryRegisterPlannedSelectedTransaction() const noexcept;

  /// Withdraw one previously registered selected transaction without underflow.
  void unregisterPlannedSelectedTransaction() const noexcept;

  /// Validate and abandon one planned selected proposal without a public API.
  void cancelPlannedSelectedProposal(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const MCMultiParticleMoves<CoordsType::POS>& moves,
      std::size_t expected_proposal_version) const;

  /// Lazily create fixed storage for scalar value evaluation.
  pf::DirectValueWorkspace& requireDirectValueWorkspace();

  /// Lazily create the requested scalar spatial-derivative workspace.
  pf::DirectSpatialWorkspace& requireDirectSpatialWorkspace(EvaluationPurpose purpose);

  /// Lazily create legacy batch scratch only when no explicit plan is bound.
  pf::DirectBatchWorkspace& requireDirectBatchWorkspace();

  /// Fingerprint one exact ordered scalar VALUE input without allocating.
  std::uint64_t scalarValueInputFingerprint(
      const PlannedScalarValueRequest& request) const noexcept;

  /// Prove exact clone-local scalar input, output, plan, and storage evidence.
  PlannedScalarValueAccess requirePlannedScalarValueOperation(
      const PlannedScalarValueRequest& request);

  /// Execute and atomically publish one prepared scalar VALUE transaction.
  void evaluatePlannedScalarValue(PlannedScalarValueOperation operation,
                                  const ParticleSet& reference,
                                  const VirtualParticleSet* virtual_particles,
                                  std::vector<ValueType>& ratios);

  /// Reject unaccounted scalar value/spatial evaluators under an explicit plan.
  void requireUnplannedScalarEvaluation(const char* operation) const;

  /// Reject legacy scalar virtual-particle dispatch when flattened ECP is planned.
  void requireNoPlannedEcpScalarDispatch(const char* operation) const;

  /// Reject clone-local score/kinetic tapes while an explicit plan is bound.
  void requireUnplannedScalarDerivative(const char* operation) const;

  /// Reject deferred crowd owners at their first explicit-plan dispatch point.
  void requireUnplannedMultiWalkerOperation(const char* operation) const;

  /// Lazily create the clone-local score tape used by scalar evaluation paths.
  pf::DirectScoreWorkspace& requireDirectScoreWorkspace();

  /// Lazily create the clone-local kinetic tape used by scalar evaluation paths.
  pf::DirectKineticWorkspace& requireDirectKineticWorkspace();

  /// Lazily size the clone-local complete-drift buffer used by scalar kinetic calls.
  std::vector<double>& requireDirectTotalLogGradient();

  /// Evaluate a score in caller-selected scratch under one model read transaction.
  pf::DirectScoreResult evaluateDirectScorePositionsUnderRead(
      const PsiFormerReadTransaction& transaction,
      const ParticleSet& particles,
      int replaced_particle,
      const PosType* replacement_position,
      pf::DirectScoreWorkspace& workspace);

  /// Evaluate virtual ratios while retaining a caller-owned model transaction.
  void evaluateRatiosUnderRead(const PsiFormerReadTransaction& transaction,
                               const VirtualParticleSet& virtual_particles,
                               std::vector<ValueType>& ratios);

  using SelectedDerivativeDelta = std::vector<std::pair<std::size_t, ValueType>>;

  /// Copy and validate active score entries while state and metadata remain locked.
  void gatherSelectedGradientUnderRead(
      const PsiFormerDerivativeReadTransaction& transaction,
      const double* flat_gradient,
      std::size_t gradient_size,
      ValueType scale,
      std::size_t destination_size,
      SelectedDerivativeDelta& output) const;

  /// Validate and reduce weighted virtual score differences using optional crowd scratch.
  void evaluateDerivRatiosWeightedImpl(const PsiFormerDerivativeReadTransaction& transaction,
                                       const VirtualParticleSet& virtual_particles,
                                       const OptVariables& optvars,
                                       const std::vector<ValueType>& total_weights,
                                       std::size_t destination_size,
                                       pf::DirectScoreWorkspace* crowd_workspace,
                                       SelectedDerivativeDelta& output);

  /// Evaluate score and kinetic response with optional resource-owned scratch.
  void evaluateDerivativesImpl(const PsiFormerDerivativeReadTransaction& transaction,
                               ParticleSet& particles,
                               const OptVariables& optvars,
                               std::size_t score_destination_size,
                               std::size_t kinetic_destination_size,
                               pf::DirectKineticWorkspace* crowd_workspace,
                               std::vector<double>* crowd_total_log_gradient,
                               SelectedDerivativeDelta& score_output,
                               SelectedDerivativeDelta& kinetic_output);

  /// Count clone- and crowd-owned kinetic tapes for the bounded-memory regression.
  std::array<std::size_t, 2> directKineticWorkspaceOwnershipForTesting(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const;

  /// Report clone-local evaluator ownership and explicitly reserved numeric bytes.
  testing::PsiFormerWorkspaceDiagnostics directWorkspaceDiagnosticsForTesting() const;

  /// Corrupt or canonically restore only the prepared scalar capacity record.
  void setPreparedScalarWorkspaceFaultForTesting(
      PreparedScalarWorkspaceFaultForTesting fault);

  /// Inspect one typed preflight without parsing or changing either cursor.
  WalkerBufferCursorSnapshot inspectPlannedWalkerBufferPreflightForTesting(
      PlannedWalkerBufferOperation operation,
      const ParticleSet& particles,
      const WFBufferType& buffer) const;

  /// Parse one planned read record without advancing cursors or publishing state.
  WalkerBufferRecordView inspectPlannedWalkerBufferForTesting(
      const ParticleSet& particles,
      const WFBufferType& buffer) const;

  /// Inject one reversible mutation inside the actual two-phase inspection flow.
  void probePlannedWalkerBufferBetweenPhaseFaultForTesting(
      ParticleSet& particles,
      WFBufferType& buffer,
      PlannedWalkerBufferBetweenPhaseFaultForTesting fault);

  /// Select or clear one true late failure in a public buffer transaction.
  void setPlannedWalkerBufferLateFaultForTesting(
      PlannedWalkerBufferLateFaultForTesting fault) noexcept;

  /// Exercise one reversible malformed snapshot or range through production checks.
  void probePlannedWalkerBufferFaultForTesting(
      const ParticleSet& particles,
      const WFBufferType& buffer,
      PlannedWalkerBufferFaultForTesting fault) const;

  /// Report opaque identity and numeric capacity for one acquired crowd resource.
  testing::PsiFormerCrowdWorkspaceDiagnostics crowdWorkspaceDiagnosticsForTesting(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const;

  /// Exercise the allocation-free typed ratio conversion independently of a call path.
  static PsiValue ratioArenaRoundTripForTesting(PsiValue value,
                                                bool use_log_value_arena);

  /// Identify one deliberately malformed ratio-arena condition for friend-only tests.
  enum class RatioArenaFaultForTesting
  {
    WRONG_KIND,
    WRONG_PREFIX,
    CHANGED_POINTER,
    CHANGED_SIZE,
    CHANGED_CAPACITY,
    DUAL_ARENAS,
    NO_ARENA,
    NONZERO_LOG_IMAGINARY
  };

  /** Exercise production ratio-arena validation through a transient malformed
   * state.  Every resource mutation is restored before this test-only seam
   * returns or propagates an exception. */
  PsiValue ratioArenaFaultForTesting(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      RatioArenaFaultForTesting fault) const;

  /// Copy bounded selected-compaction prefixes without exposing mutable scratch.
  testing::PsiFormerSelectedProposalMapDiagnostics
  selectedProposalMapDiagnosticsForTesting(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      std::size_t live_walkers,
      std::size_t evaluated_rows) const;

  /// Report the model-wide selected-transaction count without changing it.
  std::size_t plannedSelectedTransactionCountForTesting() const noexcept;

  /// Report the model-wide planned one-electron count without changing it.
  std::size_t plannedSingleTransactionCountForTesting() const noexcept;

  /// Advance the shared model version behind a pending proposal for a stale-state test.
  std::size_t advanceParameterVersionForTesting();

  /// Report shared optimizer metadata ownership and the empty inherited variable set.
  testing::PsiFormerOptimizationMetadataDiagnostics optimizationMetadataDiagnosticsForTesting() const;

  /// Invalidate accepted/proposed caches and record the newly observed shared version.
  void invalidateParameterCaches(std::size_t parameter_version);

  /// Lazily invalidate this clone when another clone changed the shared parameters.
  void synchronizeParameterVersion(std::size_t parameter_version);

  /// Reset proposal metadata while deliberately retaining the publication marker.
  void resetProposalMetadata() noexcept;

  /// Reset all proposal metadata, publishing has_proposal_ false last.
  void clearProposalState() noexcept;

  /// Publish one legacy one-electron proposal with an explicit parameter-version key.
  void cacheSingleParticleProposal(double sign,
                                   double logabs,
                                   std::uint64_t configuration_identity,
                                   int particle,
                                   std::size_t parameter_version,
                                   ProposalOrigin origin);

  /// Reject lifecycle operations that could silently overwrite a selected transaction.
  void requireNoSelectedParticleProposal(const char* operation) const;

  /// Reject shared-parameter mutation that would strand any planned crowd proposal.
  void requireNoPlannedProposalMutation(const char* operation) const;

  /// Resize clone-local complete proposed G/L storage without publishing a proposal.
  void resizeProposedSpatialStorage(std::size_t electron_count);

  /// Resize clone-local accepted G/L storage to one runtime electron configuration.
  void resizeAcceptedSpatialStorage(std::size_t electron_count);

  /// Return whether accepted state satisfies an identity, version, and product requirement.
  bool acceptedStateMatches(const ParticleSet& particles,
                            std::size_t parameter_version,
                            AcceptedStateRequirement requirement) const;

  /// Translate immutable model, backend, and optimizer facts into the pure policy input.
  psiformer::PsiFormerMemoryPolicyInput makeBatchMemoryPolicyInput() const;

  /// Add cached component G/L contributions to caller-owned wavefunction accumulators.
  void accumulateAcceptedSpatial(ParticleSet::ParticleGradient& gradient,
                                 ParticleSet::ParticleLaplacian& laplacian) const;

  /// Prove clone, plan, ParticleSet, and cursor evidence for one buffer contract.
  WalkerBufferCursorSnapshot requirePlannedWalkerBufferOperation(
      PlannedWalkerBufferOperation operation,
      const ParticleSet& particles,
      const WFBufferType& buffer) const;

  /// Validate allocated PooledMemory domains and derive their checked scalar extent.
  WalkerBufferCursorSnapshot validateWalkerBufferCursorSnapshot(
      WalkerBufferCursorSnapshot snapshot) const;

  /// Hash exact operation, PooledMemory identity, cursors, and prepared layout.
  static WalkerBufferCursorSnapshot fingerprintWalkerBufferCursorSnapshot(
      WalkerBufferCursorSnapshot snapshot) noexcept;

  /// Form checked destination ranges without interpreting their old contents.
  WalkerBufferRecordView makePlannedWalkerBufferRecordView(
      const ParticleSet& particles,
      WalkerBufferCursorSnapshot snapshot) const;

  /// Decode and classify one complete record without mutating either cursor.
  WalkerBufferRecordView parsePlannedWalkerBufferRecord(
      const ParticleSet& particles,
      WalkerBufferCursorSnapshot snapshot) const;

  /// Apply authoritative version/configuration evidence to a valid record.
  static WalkerBufferRecordView classifyPlannedWalkerBufferRecord(
      WalkerBufferRecordView record,
      std::size_t authoritative_parameter_version);

  /// Reject overlap between record ranges and component or ParticleSet storage.
  void requireDisjointWalkerBufferRecord(
      const ParticleSet& particles,
      const WalkerBufferRecordView& record) const;

  /// Validate and fingerprint one complete cache-reuse refresh input set.
  WalkerBufferRefreshEvidence requirePlannedWalkerBufferRefreshInputs(
      const ParticleSet& particles,
      std::size_t parameter_version) const;

  /// Reject aliasing among ParticleSet outputs and component spatial caches.
  void requireDisjointPlannedWalkerBufferRefresh(
      const ParticleSet& particles) const;

  /// Require a repeat observation to preserve every Phase-A cursor identity.
  static void requireSameWalkerBufferCursorObservation(
      const WalkerBufferCursorSnapshot& expected,
      const WalkerBufferCursorSnapshot& observed);

  /// Require an exact Phase-B match to one earlier cursor and record observation.
  static void requireSameWalkerBufferObservation(
      const WalkerBufferRecordView& expected,
      const WalkerBufferRecordView& observed);

  /// Reserve a planned persistent record without touching allocated storage.
  void registerDataPlanned(ParticleSet& particles, WFBufferType& buffer);

  /// Restore one checked planned record under authoritative model ownership.
  void copyFromBufferPlanned(ParticleSet& particles, WFBufferType& buffer);

  /// Write one current accepted cache through the planned no-evaluation route.
  LogValue updateBufferPlanned(ParticleSet& particles,
                               WFBufferType& buffer,
                               bool from_scratch);

  /// Serialize the complete accepted-state record at the current buffer cursor.
  void putAcceptedState(WFBufferType& buffer) const;

  /// Consume and validate one accepted-state record from the current buffer cursor.
  void getAcceptedState(const ParticleSet& particles,
                        WFBufferType& buffer,
                        std::size_t parameter_version);

  /// Versioned native model protected against evaluation/reset overlap.
  std::shared_ptr<PsiFormerSharedState> model_state_;
  /// Clone-family optimizer names, global indices, and canonical flat-index mapping.
  std::shared_ptr<PsiFormerOptimizationMetadata> optimization_metadata_;
  /// Clone-shared tensor metadata for the scalar-registration-free training route.
  std::shared_ptr<const wftrain::StructuredParameterSchema> structured_parameter_schema_;
  /// Lazily present fixed-size value buffers owned independently by this clone.
  std::unique_ptr<pf::DirectValueWorkspace> direct_value_workspace_;
  /// Lazily present fixed-size score tape owned independently by an optimizable clone.
  std::unique_ptr<pf::DirectScoreWorkspace> direct_score_workspace_;
  /// Lazily present combined score/kinetic tape used only by scalar calls.
  std::unique_ptr<pf::DirectKineticWorkspace> direct_kinetic_workspace_;
  /// Lazily sized complete TrialWaveFunction drift used only by scalar calls.
  std::vector<double> direct_total_log_gradient_;
  /// Lazily present full-gradient and trace-Laplacian storage owned by this clone.
  std::unique_ptr<pf::DirectSpatialWorkspace> direct_full_spatial_workspace_;
  /// Lazily present first-order-only storage reused for active-electron gradients.
  std::unique_ptr<pf::DirectSpatialWorkspace> direct_active_spatial_workspace_;
  /// Lazily present batch scratch used by scalar all-to-one and virtual-ratio calls.
  std::unique_ptr<pf::DirectBatchWorkspace> direct_batch_workspace_;
  /// Exact clone-local output staging for admitted scalar VALUE compatibility.
  std::vector<ValueType> scalar_value_publication_;
  /// ResourceCollection-owned workspace handle populated only on the crowd leader.
  ResourceHandle<PsiFormerMultiWalkerResource> mw_resource_handle_;
  /// Immutable selected-plan slice copied to component clones without copying scratch.
  BatchExecutionParticipantPlan batch_execution_plan_;
  /// Binding whose clone-local storage has completed exact preparation.
  BatchExecutionParticipantPlan prepared_clone_batch_execution_plan_;
  /// Exact non-owning ParticleSet binding retained by clone preparation.
  const ParticleSet* prepared_bound_particle_set_ = nullptr;
  /// Layout evidence for externally owned persistent walker-record regions.
  pf::WalkerBufferLayout prepared_walker_buffer_layout_;
  /// Fixed allocation identities published immediately before the preparation marker.
  const GradType* prepared_accepted_gradient_data_ = nullptr;
  const ValueType* prepared_accepted_laplacian_data_ = nullptr;
  const GradType* prepared_proposed_gradient_data_ = nullptr;
  const ValueType* prepared_proposed_laplacian_data_ = nullptr;
  /// Exact admitted capacities paired with the retained allocation identities.
  std::size_t prepared_accepted_gradient_capacity_ = 0;
  std::size_t prepared_accepted_laplacian_capacity_ = 0;
  std::size_t prepared_proposed_gradient_capacity_ = 0;
  std::size_t prepared_proposed_laplacian_capacity_ = 0;
  /// Exact scalar owner evidence published immediately before the plan marker.
  bool prepared_scalar_value_compatibility_ = false;
  const pf::DirectBatchWorkspace* prepared_batch_workspace_identity_ = nullptr;
  std::size_t prepared_batch_storage_fingerprint_ = 0;
  std::size_t prepared_batch_bytes_ = 0;
  const ValueType* prepared_scalar_value_publication_data_ = nullptr;
  std::size_t prepared_scalar_value_publication_size_ = 0;
  std::size_t prepared_scalar_value_publication_capacity_ = 0;
  /// Inject a late clone-preparation failure for the strong-guarantee regression.
  bool fail_clone_preparation_before_publish_for_testing_ = false;
  /// Inject a post-evaluation FULL_VGL failure before any public-state publication.
  bool fail_planned_full_vgl_before_publish_for_testing_ = false;
  /// Inject a post-evaluation recompute failure before accepted-value publication.
  bool fail_planned_recompute_before_publish_for_testing_ = false;
  /// Inject a post-evaluation active-gradient failure before caller publication.
  bool fail_planned_active_gradient_before_publish_for_testing_ = false;
  /// Inject a post-evaluation selected-proposal failure before publication.
  bool fail_planned_selected_proposal_before_publish_for_testing_ = false;
  /// Inject a selected-resolution failure after its final read-only recheck.
  bool fail_planned_selected_resolution_before_publish_for_testing_ = false;
  /// Inject a one-electron producer failure after its final read-only recheck.
  bool fail_planned_single_proposal_before_publish_for_testing_ = false;
  /// Inject a one-electron resolution failure after its final read-only recheck.
  bool fail_planned_single_resolution_before_publish_for_testing_ = false;
  /// Inject a one-electron cancellation failure after its final preflight.
  bool fail_planned_single_cancellation_before_publish_for_testing_ = false;
  /// Inject a scalar VALUE failure after its final recheck but before publication.
  bool fail_planned_scalar_value_before_publish_for_testing_ = false;
  /// Apply one reversible corruption between evaluation and scalar Phase B.
  PlannedScalarValueFaultForTesting
      planned_scalar_value_fault_for_testing_ =
          PlannedScalarValueFaultForTesting::NONE;
  /// Inject a final failure after one walker-buffer transaction's Phase B.
  PlannedWalkerBufferLateFaultForTesting
      planned_walker_buffer_late_fault_for_testing_ =
          PlannedWalkerBufferLateFaultForTesting::NONE;
  /// Substitute a finite maximum contribution to exercise additive overflow.
  bool force_planned_ratio_gradient_overflow_for_testing_ = false;
  /// Friend-only seam enabling complete Stage-5 ownership evidence in tests.
  bool complete_batch_memory_accounting_for_testing_ = false;
  /// Runtime system declaration validated against the export and QMCPACK particle sets.
  std::string system_kind_ = "unvalidated";
  /// Non-owning lane identity established by system validation or clone construction.
  const ParticleSet* bound_particle_set_ = nullptr;
  /// Borrowed collection identity retained while the leader owns its crowd resource.
  const ResourceCollection* acquired_resource_collection_ = nullptr;
  /// Exact cursor and loan counts immediately after this component acquired its resource.
  std::size_t acquired_resource_cursor_            = 0;
  std::size_t acquired_resource_outstanding_loans_ = 0;
  /// Exact ephemeral lane order published after the crowd resource loan succeeds.
  const PsiFormerWF* acquired_crowd_leader_ = nullptr;
  std::size_t acquired_lane_index_           = 0;
  std::size_t acquired_crowd_size_           = 0;
  /// Optional DeepQMC-format destination written with the final QMCPACK VP report.
  std::string optimized_parameter_export_;
  /// Last shared parameter version observed by this component clone.
  std::size_t observed_parameter_version_ = 0;
  /// Require the generic selected values to agree after an authoritative VP restore.
  bool restore_validation_pending_ = false;
  /// Track whether log_value_ belongs to the current shared parameter version.
  bool accepted_value_valid_ = false;
  /// Component-only accepted gradient contribution persisted with each walker.
  ParticleSet::ParticleGradient accepted_gradient_;
  /// Component-only accepted logarithmic-Laplacian contribution persisted with each walker.
  ParticleSet::ParticleLaplacian accepted_laplacian_;
  /// Exact-coordinate fingerprint of the accepted state represented by the cache.
  std::uint64_t accepted_configuration_identity_ = 0;
  /// Parameter version under which the accepted products were evaluated.
  std::size_t accepted_parameter_version_ = 0;
  /// Strongest accepted product family currently available in clone-local storage.
  AcceptedStateRequirement accepted_state_requirement_ = AcceptedStateRequirement::INVALID;
  /// Accepted and proposed sign/log-value state used by particle-by-particle moves.
  double current_sign_ = 1.0, proposed_sign_ = 1.0;
  LogValue proposed_log_value_ = LogValue(0);
  /// Complete component-only spatial products retained during a selected transaction.
  ParticleSet::ParticleGradient proposed_gradient_;
  ParticleSet::ParticleLaplacian proposed_laplacian_;
  /// Fingerprint and electron index associated with the pending proposal.
  std::uint64_t proposed_configuration_identity_ = 0;
  /// Exact transaction identity required by planned single- or selected-particle resolution.
  std::uint64_t proposed_descriptor_fingerprint_ = 0;
  /// Parameter version used to evaluate the pending proposal.
  std::size_t proposed_parameter_version_ = 0;
  int proposed_particle_ = -1;
  ProposalOrigin proposal_origin_ = ProposalOrigin::NONE;
  bool has_proposal_              = false;

  friend class testing::TestPsiFormerWF;
  friend class testing::TestPsiFormerVirtualBatch;
};

} // namespace qmcplusplus

#endif
