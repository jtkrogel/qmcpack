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
#include "Particle/MCMultiParticleMoves.h"
#include <memory>
#include <vector>

namespace qmcplusplus
{
class TrialWaveFunction;
class PsiFormerWF;
class SlaterDet;

/** Own independent child evaluation state while sharing frozen read-only model storage.
 * The first profile is small real Gamma CPU VMC; no training provider is exposed.
 */
class FrozenReferenceResidualWF final : public WaveFunctionComponent
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

  /// Refuse training derivatives: this bounded component exposes fixed-parameter VMC only.
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

  const double coefficient_;
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
