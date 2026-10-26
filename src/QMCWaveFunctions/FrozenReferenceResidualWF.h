//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file FrozenReferenceResidualWF.h
 * Fixed-coefficient real CPU sum of a complete frozen SJ reference and PsiFormer.
 */
#ifndef QMCPLUSPLUS_FROZEN_REFERENCE_RESIDUAL_WF_H
#define QMCPLUSPLUS_FROZEN_REFERENCE_RESIDUAL_WF_H

#include "WaveFunctionComponent.h"
#include "FrozenReferenceSumMath.h"
#include "Optimization/StructuredParameterProvider.h"
#include "Particle/MCMultiParticleMoves.h"
#include <memory>
#include <vector>

namespace qmcplusplus
{
class TrialWaveFunction;
class PsiFormerWF;
class SlaterDet;
struct FrozenResidualDerivativeWorkspace;

/** Own independent child evaluation state, versioned neural weights and frozen reference storage.
 * Neural training uses the total parent score; reference parameters remain private and frozen.
 */
class FrozenReferenceResidualWF final : public WaveFunctionComponent, public wftrain::StructuredParameterProvider
{
public:
  /// Clone and freeze two previously constructed complete wavefunction definitions.
  FrozenReferenceResidualWF(ParticleSet& target, const TrialWaveFunction& reference,
                           const TrialWaveFunction& neural, double coefficient);

  /// Destroy child wavefunctions before their private particle sets.
  ~FrozenReferenceResidualWF() override;

  /// Identify the component in native timers, diagnostics and clone topology.
  std::string getClassName() const override { return "FrozenReferenceResidualWF"; }

  /// Preserve fermionic virtual-ratio dispatch for the complete summed amplitude.
  bool isFermionic() const override { return true; }

  /// Evaluate the complete signed sum and add its logarithmic spatial derivatives.
  LogValue evaluateLog(const ParticleSet& particles, ParticleSet::ParticleGradient& gradient,
                       ParticleSet::ParticleLaplacian& laplacian) override;

  /// Opt into neural-only structured training; fixed-zero c cannot train the neural branch.
  void enableNeuralTraining();

  /// Expose the parent provider only after explicit training selection.
  wftrain::StructuredParameterProvider* structuredParameterProvider() noexcept override
  { return training_enabled_ ? this : nullptr; }

  /// Return the parent schema, containing only neural tensors and no reference parameters.
  const wftrain::StructuredParameterSchema& parameterSchema() const noexcept override;

  /// Inspect the shared model version without copying its parameter vector.
  std::size_t currentParameterVersion() const;

  /// Copy the private neural state under the parent schema identity.
  wftrain::StructuredParameterSnapshot snapshotParameters() const override;

  /// Validate the parent identity before atomically publishing to the shared neural family.
  std::size_t publishParameters(const wftrain::StructuredParameterSnapshot& candidate,
                               std::size_t expected_version) override;

  /// Detach speculative optimizer candidates without changing the frozen reference.
  void isolateParameterState() override;

  /// Preserve neural weight classification for explicitly selected decoupled decay.
  std::vector<std::string> neuralWeightBlocks() const override;

  /// Preserve affine block order while weighting observations by the complete parent amplitude.
  std::vector<wftrain::AffineParameterBlock> affineParameterBlocks() const override;

  /// Preserve the child generic-diagonal fallback block selection.
  std::vector<std::size_t> genericKFACParameterBlocks() const override;

  /// Evaluate one canonical total-parent score and optionally its affine observations.
  void evaluateParameterScore(const ParticleSet& particles, std::vector<double>& score,
                              wftrain::AffineScoreObserver* observer = nullptr) const;

  /// Feed parent-weighted affine sensitivities into the existing KFAC factor collector.
  void observeAffineScores(const ParticleSet& particles, wftrain::AffineScoreObserver& observer,
                           std::vector<double>& score) const override;

  /// Bind bounded score products to immutable coordinates and a shared clone-family workspace.
  std::unique_ptr<wftrain::StreamingDerivativeOperator> makeStreamingDerivativeOperator(
      const RefVectorWithLeader<WaveFunctionComponent>& components,
      const RefVectorWithLeader<ParticleSet>& particles, std::size_t batch_ordinal,
      std::size_t sample_offset, std::size_t maximum_parameter_chunk_size) const override;

  /// Identify immutable reference content and the fixed relative coefficient for restart checks.
  const std::string& frozenModelIdentity() const noexcept { return frozen_model_identity_; }

  /// Export neural topology/weights together with the frozen-reference reconstruction contract.
  void exportModel(const std::string& path) const;

  /// Read and validate a composite model before publishing any neural values.
  wftrain::StructuredParameterSnapshot readModel(const std::string& path) const;

  /// Import a composite parameter artifact for VMC or a fresh training stage.
  void importModel(const std::string& path);

  /// Return the accepted total log gradient for one electron.
  GradType evalGrad(ParticleSet& particles, int electron) override;

  /// Stage a single absolute-position proposal using full isolated child recomputation.
  PsiValue ratio(ParticleSet& particles, int electron) override;

  /// Stage the same proposal and add its total proposed gradient.
  PsiValue ratioGrad(ParticleSet& particles, int electron, GradType& gradient) override;

  /// Publish the complete previously evaluated single-electron proposal.
  void acceptMove(ParticleSet& particles, int electron, bool safe_to_delay = false) override;

  /// Discard speculative state without advancing either accepted branch.
  void restore(int electron) override;

  /// Advertise the explicitly implemented atomic absolute-coordinate proposal route.
  bool supportsMultiParticleMoves() const noexcept override { return true; }

  /// Stage one complete total-amplitude proposal per walker without accepted-state mutation.
  void mw_evaluateMultiParticleMove(const RefVectorWithLeader<WaveFunctionComponent>& components,
      const RefVectorWithLeader<ParticleSet>& particles, const MCMultiParticleMoves<CoordsType::POS>& moves,
      std::vector<LogValue>& log_ratios, const RefVector<ParticleSet::ParticleGradient>& gradients,
      const RefVector<ParticleSet::ParticleLaplacian>& laplacians) const override;

  /// Resolve all staged walker proposals after validating the complete transaction.
  void mw_accept_rejectMultiParticleMove(const RefVectorWithLeader<WaveFunctionComponent>& components,
      const RefVectorWithLeader<ParticleSet>& particles, const MCMultiParticleMoves<CoordsType::POS>& moves,
      const std::vector<bool>& accepted) const override;

  /// Evaluate the sum at each nonlocal quadrature position, not child ratio averages.
  void evaluateRatios(const VirtualParticleSet& particles, std::vector<ValueType>& ratios) override;

  /// Reserve a fixed-size record for the accepted total state in the native walker buffer.
  void registerData(ParticleSet& particles, WFBufferType& buffer) override;

  /// Refresh accepted state and overwrite its existing fixed-size walker record.
  LogValue updateBuffer(ParticleSet& particles, WFBufferType& buffer, bool fromscratch = false) override;

  /// Restore the total cache; child scratch is always recomputed before it is consumed.
  void copyFromBuffer(ParticleSet& particles, WFBufferType& buffer) override;

  /// Refuse the legacy full-response interface; neural training uses bounded structured scores.
  void evaluateDerivatives(ParticleSet& particles, const OptVariables& variables,
                           Vector<ValueType>& score, Vector<ValueType>& energy_response) override;

  /// Clone private mutable state while preserving the already isolated frozen neural model.
  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet& particles) const override;

private:
  /** Complete accepted or speculative spatial state, independent of child scratch caches. */
  struct State
  {
    residual::LogSpatialJet spatial;
    std::vector<PosType> positions;
    std::size_t parameter_version = 0;
  };

  /// Distinguish the initial model detachment from subsequent read-only walker clones.
  FrozenReferenceResidualWF(ParticleSet& target, const TrialWaveFunction& reference,
                           const TrialWaveFunction& neural, double coefficient, bool detach_neural);

  /// Compute the frozen reference through determinant numerators and positive Jastrow jets.
  residual::ScaledSpatialJet referenceAmplitude();

  /// Recompute both child amplitudes at one complete physical configuration.
  State evaluateAt(const std::vector<PosType>& positions);

  /// Refresh accepted state only when the caller has supplied a different configuration.
  void ensureAccepted(const ParticleSet& particles);

  /// Convert one complete total cache into the native complex-log representation.
  static LogValue nativeLog(const State& state);

  /// Convert long-double intermediates to finite native scalar results.
  static RealType checkedReal(long double value);

  /// Add only this component's total G/L contributions to caller-owned arrays.
  static void addSpatial(const State& state, ParticleSet::ParticleGradient& gradient,
                         ParticleSet::ParticleLaplacian& laplacian);

  /// Form a signed ratio with the accepted nonzero total amplitude as denominator.
  PsiValue proposedRatio(const State& proposed) const;

  /// Publish a prepared cache without inspecting stale child scratch.
  void commitProposal();

  /// Allocate the shared workspace owner without exposing its implementation in this header.
  static std::shared_ptr<FrozenResidualDerivativeWorkspace> createDerivativeWorkspace();

  mutable std::shared_ptr<FrozenResidualDerivativeWorkspace> derivative_workspace_;
  const double coefficient_;
  std::string frozen_model_identity_;
  bool training_enabled_ = false;
  std::unique_ptr<wftrain::StructuredParameterSchema> parameter_schema_;
  std::unique_ptr<ParticleSet> reference_particles_;
  std::unique_ptr<ParticleSet> neural_particles_;
  std::unique_ptr<TrialWaveFunction> reference_;
  std::unique_ptr<TrialWaveFunction> neural_;
  SlaterDet* slater_ = nullptr;
  PsiFormerWF* neural_component_ = nullptr;
  State accepted_;
  State proposed_;
  bool accepted_valid_ = false;
  bool proposal_pending_ = false;
  int proposed_electron_ = -1;
  std::uint64_t proposal_fingerprint_ = 0;
};

} // namespace qmcplusplus
#endif
