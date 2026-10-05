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
/// Shared, versioned native model state used by all clones of one component.
class PsiFormerSharedState;

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

  std::size_t value_bytes          = 0;
  std::size_t full_spatial_bytes   = 0;
  std::size_t active_spatial_bytes = 0;
  std::size_t batch_bytes          = 0;
  std::size_t score_bytes          = 0;
  std::size_t kinetic_bytes        = 0;
  std::size_t total_log_gradient_bytes = 0;

  /// Return all explicitly accounted clone-local evaluator scratch bytes.
  std::size_t accountedBytes() const noexcept
  {
    return value_bytes + full_spatial_bytes + active_spatial_bytes + batch_bytes +
        score_bytes + kinetic_bytes + total_log_gradient_bytes;
  }

  /// Return the number of independently owned native evaluator workspaces.
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

/// Test-only accessor for bounded crowd-workspace ownership diagnostics.
class TestPsiFormerWF;
}

/**
 * Wavefunction component for a PsiFormer model exported from DeepQMC.
 *
 * Imported parameters remain fixed unless optimization is explicitly enabled.
 * Optimization may register selected canonical flat indices or the complete
 * network in that same ordering. Full-network resets use the native complete
 * vector path to avoid sorting and revalidating millions of canonical indices.
 *
 * Clones share one versioned native model behind a reader/writer lock, while
 * accepted and proposed move state remains clone-local. Object-specific VP
 * records persist the complete model independently of the selected scalar list.
 */
class PsiFormerWF : public WaveFunctionComponent, public OptimizableObject
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

  /// Refresh selected accepted values without serial component dispatch.
  void mw_recompute(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                    const RefVectorWithLeader<ParticleSet>& p_list,
                    const std::vector<bool>& recompute) const override;

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

  /// Evaluate a proposed ratio and active-electron logarithmic gradient together.
  PsiValue ratioGrad(ParticleSet& particles, int particle_index, GradType& gradient) override;

  /// Evaluate proposal ratios and active-electron gradients in one crowd traversal.
  void mw_ratioGrad(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                    const RefVectorWithLeader<ParticleSet>& p_list,
                    int particle_index,
                    std::vector<PsiValue>& ratios,
                    std::vector<GradType>& gradients) const override;

  /// Commit or discard each walker's independently cached proposal state.
  void mw_accept_rejectMove(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                            const RefVectorWithLeader<ParticleSet>& p_list,
                            int particle_index,
                            const std::vector<bool>& is_accepted,
                            bool safe_to_delay = false) const override;

  /// Evaluate replacement of each electron by ParticleSet::getActivePos() without state changes.
  void evaluateRatiosAlltoOne(ParticleSet& particles, std::vector<ValueType>& ratios) override;

  /// Evaluate every nonlocal-pseudopotential virtual-move ratio by full model reevaluation.
  void evaluateRatios(const VirtualParticleSet& virtual_particles, std::vector<ValueType>& ratios) override;

  /// Flatten ragged virtual-position lists into one direct crowd batch.
  void mw_evaluateRatios(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                         const RefVectorWithLeader<const VirtualParticleSet>& virtual_particle_list,
                         std::vector<std::vector<ValueType>>& ratios) const override;

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

  /// Evaluate an accepted or proposed configuration for one public-call purpose.
  pf::Result evaluate(const ParticleSet& particles,
                      int replaced_particle,
                      EvaluationPurpose purpose,
                      int active_gradient_particle = -1);

  /// Evaluate a configuration with an optional explicit replacement for one electron position.
  pf::Result evaluatePositions(const ParticleSet& particles,
                               int replaced_particle,
                               const PosType* replacement_position,
                               EvaluationPurpose purpose,
                               int active_gradient_particle = -1);

  /// Evaluate a scalar value request directly into clone-local fixed storage.
  pf::DirectValueResult evaluateDirectValuePositions(const ParticleSet& particles,
                                                     int replaced_particle,
                                                     const PosType* replacement_position);

  /// Evaluate a scalar spatial request and return a view into clone-local fixed storage.
  pf::DirectSpatialResultView evaluateDirectSpatialPositions(
      const ParticleSet& particles,
      int replaced_particle,
      const PosType* replacement_position,
      EvaluationPurpose purpose,
      int active_gradient_particle);

  /// Validate one homogeneous clone crowd and return its exclusively acquired resource.
  PsiFormerMultiWalkerResource& requireMultiWalkerResource(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const;

  /// Lazily create fixed storage for scalar value evaluation.
  pf::DirectValueWorkspace& requireDirectValueWorkspace();

  /// Lazily create the requested scalar spatial-derivative workspace.
  pf::DirectSpatialWorkspace& requireDirectSpatialWorkspace(EvaluationPurpose purpose);

  /// Lazily create batch scratch for scalar all-to-one and virtual-ratio calls.
  pf::DirectBatchWorkspace& requireDirectBatchWorkspace();

  /// Lazily create the clone-local score tape used by scalar evaluation paths.
  pf::DirectScoreWorkspace& requireDirectScoreWorkspace();

  /// Lazily create the clone-local kinetic tape used by scalar evaluation paths.
  pf::DirectKineticWorkspace& requireDirectKineticWorkspace();

  /// Lazily size the clone-local complete-drift buffer used by scalar kinetic calls.
  std::vector<double>& requireDirectTotalLogGradient();

  /// Evaluate a direct score and return a view into the clone-local score workspace.
  pf::DirectScoreResult evaluateDirectScorePositions(const ParticleSet& particles,
                                                     int replaced_particle,
                                                     const PosType* replacement_position);

  /// Evaluate a direct score into caller-owned scratch, enabling crowd-level reuse.
  pf::DirectScoreResult evaluateDirectScorePositions(const ParticleSet& particles,
                                                     int replaced_particle,
                                                     const PosType* replacement_position,
                                                     pf::DirectScoreWorkspace& workspace);

  /// Validate and reduce weighted virtual score differences using optional crowd scratch.
  void evaluateDerivRatiosWeightedImpl(const VirtualParticleSet& virtual_particles,
                                       const OptVariables& optvars,
                                       const std::vector<ValueType>& total_weights,
                                       ParameterDerivativeView weighted_derivatives,
                                       pf::DirectScoreWorkspace* crowd_workspace);

  /// Evaluate score and kinetic response with optional resource-owned scratch.
  void evaluateDerivativesImpl(ParticleSet& particles,
                               const OptVariables& optvars,
                               Vector<ValueType>& dlogpsi,
                               Vector<ValueType>& dhpsioverpsi,
                               pf::DirectKineticWorkspace* crowd_workspace,
                               std::vector<double>* crowd_total_log_gradient);

  /// Count clone- and crowd-owned kinetic tapes for the bounded-memory regression.
  std::array<std::size_t, 2> directKineticWorkspaceOwnershipForTesting(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const;

  /// Report clone-local evaluator ownership and explicitly reserved numeric bytes.
  testing::PsiFormerWorkspaceDiagnostics directWorkspaceDiagnosticsForTesting() const;

  /// Report shared optimizer metadata ownership and the empty inherited variable set.
  testing::PsiFormerOptimizationMetadataDiagnostics optimizationMetadataDiagnosticsForTesting() const;

  /// Return true when at least one selected local parameter maps to a global active variable.
  bool hasActiveParameters() const;

  /// Add selected entries from a native flat gradient to a QMCPACK derivative vector.
  void addSelectedGradient(const std::vector<double>& flat_gradient, Vector<ValueType>& output) const;

  /// Accumulate selected entries directly from a non-owning canonical score buffer.
  void addSelectedGradient(const double* flat_gradient,
                           std::size_t gradient_size,
                           Vector<ValueType>& output) const;

  /// Add a scaled native score directly to one active-parameter destination view.
  void addSelectedGradientScaled(const double* flat_gradient,
                                 std::size_t gradient_size,
                                 ValueType scale,
                                 ParameterDerivativeView output) const;

  /// Add a scaled native score directly to one row of a compatibility matrix.
  void addSelectedGradientScaled(const double* flat_gradient,
                                 std::size_t gradient_size,
                                 ValueType scale,
                                 Matrix<ValueType>& output,
                                 std::size_t row) const;

  /// Add selected differences between virtual and reference score vectors to one matrix row.
  void addSelectedGradientDifference(const std::vector<double>& reference_gradient,
                                     const std::vector<double>& virtual_gradient,
                                     Matrix<ValueType>& output,
                                     std::size_t row) const;

  /// Invalidate accepted/proposed caches and record the newly observed shared version.
  void invalidateParameterCaches(std::size_t parameter_version);

  /// Lazily invalidate this clone when another clone changed the shared parameters.
  void synchronizeParameterVersion(std::size_t parameter_version);

  /// Resize clone-local accepted G/L storage to one runtime electron configuration.
  void resizeAcceptedSpatialStorage(std::size_t electron_count);

  /// Return whether accepted state satisfies an identity, version, and product requirement.
  bool acceptedStateMatches(const ParticleSet& particles,
                            std::size_t parameter_version,
                            AcceptedStateRequirement requirement) const;

  /// Add cached component G/L contributions to caller-owned wavefunction accumulators.
  void accumulateAcceptedSpatial(ParticleSet::ParticleGradient& gradient,
                                 ParticleSet::ParticleLaplacian& laplacian) const;

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
  /// ResourceCollection-owned workspace handle populated only on the crowd leader.
  ResourceHandle<PsiFormerMultiWalkerResource> mw_resource_handle_;
  /// Runtime system declaration validated against the export and QMCPACK particle sets.
  std::string system_kind_ = "unvalidated";
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
  /// Fingerprint and electron index associated with the pending proposal.
  std::uint64_t proposed_configuration_identity_ = 0;
  int proposed_particle_ = -1;
  bool has_proposal_           = false;

  friend class testing::TestPsiFormerWF;
};

} // namespace qmcplusplus

#endif
