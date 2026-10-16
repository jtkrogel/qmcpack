//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerComplexWF.h
 * @brief Native-free paired-head inference sibling with explicit CPU or serialized GPU numerics.
 * CPU structured training and a separately capped hybrid GPU driver are explicit.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_COMPLEX_WF_H
#define QMCPLUSPLUS_PSIFORMER_COMPLEX_WF_H
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "QMCWaveFunctions/Optimization/StructuredParameterProvider.h"
#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"
#include <array>
#include <memory>
#include <string>

namespace qmcplusplus
{
class TrialWaveFunction;
namespace psiformer
{
struct InitializedPsiFormerParameters;
}

#ifdef QMC_COMPLEX
/// Full/G-free inference with prepared structured state; GPU training discovery remains closed.
class PsiFormerComplexWF final : public WaveFunctionComponent, public wftrain::StructuredParameterProvider
{
public:
  /// Bind current geometry; optional exact native root supplies parameters only, not restart state.
  PsiFormerComplexWF(std::string name, psiformer::InitializedPsiFormerParameters initialized,
                     const ParticleSet& electrons, const ParticleSet& ions,
                     std::size_t maximum_clone_numeric_bytes,
                     std::array<double, 3> reduced_twist = {}, std::size_t maximum_gpu_numeric_bytes = 0,
                     const std::string& parameter_checkpoint = {});

  /// Destroy the Native-free PIMPL in its sole native implementation translation unit.
  ~PsiFormerComplexWF() override;

  /// Numerical workspaces and move state cannot be copied implicitly.
  PsiFormerComplexWF(const PsiFormerComplexWF&) = delete;

  /// Preserve independently owned scalar scratch and identity.
  PsiFormerComplexWF& operator=(const PsiFormerComplexWF&) = delete;

  /// Report the explicitly qualified component rather than impersonating the real adapter.
  std::string getClassName() const override { return "PsiFormerComplexWF"; }

  /// Only the CPU sibling participates in generic structured-provider discovery.
  wftrain::StructuredParameterProvider* structuredParameterProvider() noexcept override
  { return gpuInferenceEnabled() ? nullptr : this; }

  /// Stable physical restart identity, independent of parameter values/version and runtime addresses.
  std::string trainingConfigurationFingerprint() const;
  /// Export paired weights and immutable physical metadata without sampler state.
  void exportModel(const std::string& path) const;

  /// Return immutable canonical REAL64 tensors, including distinct w and w_imag blocks.
  const wftrain::StructuredParameterSchema& parameterSchema() const noexcept override;

  /// Copy one version-coherent complete canonical vector under a shared read transaction.
  wftrain::StructuredParameterSnapshot snapshotParameters() const override;

  /// Atomically replace same-sized real parameters after schema/version/finite/idle admission.
  std::size_t publishParameters(const wftrain::StructuredParameterSnapshot& candidate,
                                std::size_t expected_version) override;

  /// Named NEW inference generation only; excludes native/candidate/image HOST storage and retained old generations.
  struct GpuParameterPublicationRequirements
  {
    std::size_t device_bytes, host_bytes;
  };

  /// Exact image plus frozen VALUE/SPATIAL owners and HOST staging, not an aggregate transition-peak cap.
  GpuParameterPublicationRequirements gpuParameterPublicationRequirements() const;

  /// Explicit expert/opt-in hybrid-driver publication; generic GPU discovery/publication remains closed.
  std::size_t publishBudgetedGpuParameters(const wftrain::StructuredParameterSnapshot& candidate,
      std::size_t expected_version, std::size_t additional_device_cap, std::size_t additional_host_cap);

  /// Exact two-tape lane plus immutable samples/JVP output, not the inference clone budget.
  std::size_t streamingDerivativeRequiredBytes(std::size_t sample_count) const;

  /// Explicit capped numerical binding; the unbudgeted factory remains closed.
  std::unique_ptr<wftrain::StreamingDerivativeOperator> makeBudgetedStreamingDerivativeOperator(
      const RefVectorWithLeader<WaveFunctionComponent>& components,
      const RefVectorWithLeader<ParticleSet>& particles, std::size_t batch_ordinal,
      std::size_t sample_offset, std::size_t maximum_parameter_chunk_size,
      std::size_t maximum_numeric_bytes) const;

  /// Additional derivative ownership only; excludes admitted inference/image/vendor and caller coefficients.
  struct GpuDerivativeRequirements
  {
    std::size_t device_bytes, host_bytes, matrix_work;
  };

  /// Aggregate R(S), including S>maxB: one shared lane plus HOST snapshots; individual operators require B<=maxB.
  GpuDerivativeRequirements gpuDerivativeRequirements(std::size_t sample_count,
      std::size_t maximum_batch_samples, std::size_t targets, std::size_t chunk) const;

  /// Immutable sample products for explicit analysis/hybrid driver: sole TrialWF, TRANSPOSE SCORE/physical dK.
  std::unique_ptr<wftrain::StreamingDerivativeOperator> makeBudgetedGpuDerivativeOperator(
      const RefVectorWithLeader<TrialWaveFunction>& wavefunctions,
      const RefVectorWithLeader<ParticleSet>& particles, std::size_t batch_ordinal, std::size_t sample_offset,
      std::size_t maximum_batch_samples, std::size_t targets, std::size_t chunk,
      std::size_t maximum_device_bytes, std::size_t maximum_host_bytes, std::size_t maximum_matrix_work) const;

  /// This component supplies the complete antisymmetric fermionic ansatz.
  bool isFermionic() const override { return true; }

  /// Report the prepared determinant expansion's actual extent.
  bool isMultiDet() const override;

  /// Prepared GPU inference profile; opt-in dense teams remain serialized and inference-only.
  bool gpuInferenceEnabled() const noexcept;

  /// Add full physical logarithmic G/L and publish the accepted scalar amplitude.
  LogValue evaluateLog(const ParticleSet&, ParticleSet::ParticleGradient&, ParticleSet::ParticleLaplacian&) override;

  /// Opt-in dense GPU VALUE/FULL tiles; scalar/B1 and CPU routes retain base behavior.
  void mw_evaluateLog(const RefVectorWithLeader<WaveFunctionComponent>&,
      const RefVectorWithLeader<ParticleSet>&, const RefVector<ParticleSet::ParticleGradient>&,
      const RefVector<ParticleSet::ParticleLaplacian>&) const override;

  void mw_evaluateGL(const RefVectorWithLeader<WaveFunctionComponent>&,
      const RefVectorWithLeader<ParticleSet>&, const RefVector<ParticleSet::ParticleGradient>&,
      const RefVector<ParticleSet::ParticleLaplacian>&, bool fromscratch) const override;

  void mw_calcRatio(const RefVectorWithLeader<WaveFunctionComponent>&,
      const RefVectorWithLeader<ParticleSet>&, int electron, std::vector<PsiValue>&) const override;

  /// Dense accepted FULL gradients write outputs without mutating ParticleSet or caches.
  void mw_evalGrad(const RefVectorWithLeader<WaveFunctionComponent>&,
      const RefVectorWithLeader<ParticleSet>&, int electron, std::vector<GradType>&) const override;

  /// Dense proposed FULL gradients add outputs and commit proposals only after every tail validates.
  void mw_ratioGrad(const RefVectorWithLeader<WaveFunctionComponent>&,
      const RefVectorWithLeader<ParticleSet>&, int electron, std::vector<PsiValue>&,
      std::vector<GradType>&) const override;

  void createResource(ResourceCollection&) const override;

  void acquireResource(ResourceCollection&, const RefVectorWithLeader<WaveFunctionComponent>&) const override;

  void releaseResource(ResourceCollection&, const RefVectorWithLeader<WaveFunctionComponent>&) const override;

  /// Evaluate the value-only active replacement without committing it.
  PsiValue ratio(ParticleSet&, int) override;

  /// Return one accepted physical logarithmic gradient.
  GradType evalGrad(ParticleSet&, int) override;

  /// Add a proposed physical gradient and cache its finite ratio atomically.
  PsiValue ratioGrad(ParticleSet&, int, GradType&) override;

  /// Commit only a matching cached single-electron proposal.
  void acceptMove(ParticleSet&, int, bool safe_to_delay = false) override;

  /// Discard only the matching pending proposal.
  void restore(int) override;

  /// Share immutable parameters, but prepare independent scratch and no accepted/proposal state.
  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet&) const override;

  /// Per-clone numeric buffers only: excludes shared model, allocator overhead, ParticleSet and WFBuffer.
  struct NumericStorage
  {
    std::size_t required_bytes, allocated_bytes, limit_bytes, fingerprint;
    std::size_t shared_gpu_bytes = 0, shared_gpu_host_bytes = 0, shared_image_bytes = 0;
    const void* shared_image_identity = nullptr;
  };

  /// Report exact preparation; this is not an aggregate budget or adaptive large-system qualification.
  NumericStorage numericStorage() const;

  /// Reserve an explicit complex scalar record; caller allocates the shared walker buffer.
  void registerData(ParticleSet&, WFBufferType&) override;

  /// Add own physical G/L to caller totals, refreshing FULL when necessary, and persist them.
  LogValue updateBuffer(ParticleSet&, WFBufferType&, bool from_scratch = false) override;

  /// Restore only complete matching complex records; failed reads revoke accepted caches.
  void copyFromBuffer(ParticleSet&, WFBufferType&) override;

  /// Complete complex Coulomb observations retain imaginary kinetic/local-energy parts.
  struct CoulombObservation
  {
    LogValue kinetic, local_energy;
    FullPrecRealType potential;
  };

  /** Require current FULL own physical G/L and actual target masses.
   * Caller must supply the real Coulomb potential for this exact configuration and
   * Hamiltonian sample; finiteness alone cannot prove that identity. Not a training API.
   */
  CoulombObservation observeCoulombEnergy(const ParticleSet&, FullPrecRealType same_sample_potential) const;

  /// Identify the fixed ion source for final-system Coulomb admission, not a movable source API.
  const ParticleSet& sourceParticles() const;

  /// Identify this clone's actual bound target, not a shared model's original target.
  const ParticleSet& targetParticles() const;

  /// Reject source-force fallbacks that would otherwise silently return zero.
  GradType evalGradSource(ParticleSet&, ParticleSet&, int) override;

  /// Reject mixed source/electron derivatives without fabricating products.
  GradType evalGradSource(ParticleSet&, ParticleSet&, int,
      TinyVector<ParticleSet::ParticleGradient, OHMMS_DIM>&,
      TinyVector<ParticleSet::ParticleLaplacian, OHMMS_DIM>&) override;

  /// Reject Hessians rather than use the base abort fallback.
  void evaluateHessian(ParticleSet&, HessVector&) override;

  /// Spinor requests cannot silently become positional gradients.
  GradType evalGradWithSpin(ParticleSet&, int, ComplexType&) override;

  /// Spinor proposal requests cannot silently become positional proposals.
  PsiValue ratioGradWithSpin(ParticleSet&, int, GradType&, ComplexType&) override;

  /// The scalar proposal API does not implement all-to-one virtual geometry.
  void evaluateRatiosAlltoOne(ParticleSet&, std::vector<ValueType>&) override;

  /// ECP/virtual ratios require a separately qualified complex path.
  void evaluateRatios(const VirtualParticleSet&, std::vector<ValueType>&) override;

  /// Derivative ratios are not implied by internal complex score availability.
  void evaluateDerivRatios(const VirtualParticleSet&, const OptVariables&,
                          std::vector<ValueType>&, Matrix<ValueType>&) override;

  /// Reject weighted virtual derivatives before the base compatibility matrix allocates.
  void evaluateDerivRatiosWeighted(const VirtualParticleSet&, const OptVariables&,
                                 const std::vector<ValueType>&, ParameterDerivativeView) override;

  /// Explicit aggregate planning is unqualified, not zero-cost accounting.
  BatchMemoryContribution estimateBatchExecutionMemory(const BatchExecutionPlanningContext&) const override;

  /// Reject prospective nonempty plans before any publication.
  void validateBatchExecutionPlanBinding(const BatchExecutionParticipantPlan&) const override;

  /// Ordinary unplanned CPU preparation remains a no-op.
  void prepareBatchExecutionClone(const BatchExecutionParticipantPlan&) override;

  /// Complex score availability alone does not qualify public kinetic optimization.
  void evaluateDerivatives(ParticleSet&, const OptVariables&, Vector<ValueType>&, Vector<ValueType>&) override;

private:
  struct Impl;
  /// Reuse FULL scatter while the caller already holds exactly one model read transaction.
  LogValue evaluateLogLocked(const ParticleSet&, ParticleSet::ParticleGradient&, ParticleSet::ParticleLaplacian&);
  /// Keep explicit unsupported-operation diagnostics independent of legacy fallbacks.
  [[noreturn]] static void rejectUnsupported(const char* operation);
  /// Internal clone entry point retains immutable model ownership and the explicit numeric cap.
  PsiFormerComplexWF(std::string name, std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};
#endif
} // namespace qmcplusplus
#endif
