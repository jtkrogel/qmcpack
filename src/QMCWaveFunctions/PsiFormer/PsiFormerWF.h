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
#include <memory>
#include <string>

namespace pf
{
/// Native PsiFormer evaluator shared by QMCPACK component clones.
struct PsiFormer;

/// Native high-level observables returned by one model evaluation.
struct Result;
} // namespace pf

namespace qmcplusplus
{
/**
 * Wavefunction component for a fixed PsiFormer model exported from DeepQMC.
 *
 * Clones share the immutable model parameters while retaining independent
 * accepted and proposed wavefunction state. Optimization derivatives are
 * intentionally not exposed: this component evaluates imported parameters
 * rather than registering them with QMCPACK.
 */
class PsiFormerWF : public WaveFunctionComponent
{
public:
  /// Load a fixed native model for use as one QMCPACK wavefunction component.
  PsiFormerWF(std::string name, std::string parameters, std::string configuration);

  /// Copy proposal state while sharing the immutable native model.
  PsiFormerWF(const PsiFormerWF&) = default;

  /// Return the component name used by QMCPACK diagnostics.
  std::string getClassName() const override { return "PsiFormerWF"; }

  /// Mark the sign-changing PsiFormer ansatz as fermionic.
  bool isFermionic() const override { return true; }

  /// Evaluate log(psi), gradients, and logarithmic Laplacians for a full configuration.
  LogValue evaluateLog(const ParticleSet& particles,
                       ParticleSet::ParticleGradient& gradient,
                       ParticleSet::ParticleLaplacian& laplacian) override;

  /// Commit the wavefunction state cached by the most recent proposed move.
  void acceptMove(ParticleSet& particles, int particle_index, bool safe_to_delay = false) override;

  /// Discard the wavefunction state cached for a rejected move.
  void restore(int particle_index) override;

  /// Evaluate psi(new)/psi(old) for one active-particle proposal.
  PsiValue ratio(ParticleSet& particles, int particle_index) override;

  /// Evaluate the logarithmic gradient of one electron at the accepted configuration.
  GradType evalGrad(ParticleSet& particles, int particle_index) override;

  /// Evaluate a proposed ratio and active-electron logarithmic gradient together.
  PsiValue ratioGrad(ParticleSet& particles, int particle_index, GradType& gradient) override;

  /// Register no walker-buffer data because the native component stores no such cache.
  void registerData(ParticleSet&, WFBufferType&) override {}

  /// Recompute the component and refresh the ParticleSet gradient/Laplacian accumulators.
  LogValue updateBuffer(ParticleSet& particles, WFBufferType& buffer, bool from_scratch = false) override;

  /// Restore no walker-buffer data because this component registers none.
  void copyFromBuffer(ParticleSet&, WFBufferType&) override {}

  /// Provide no QMCPACK optimization derivatives because imported parameters are fixed.
  void evaluateDerivatives(ParticleSet&, const OptVariables&, Vector<ValueType>&, Vector<ValueType>&) override {}

  /// Clone per-component move state while sharing the loaded native model.
  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet& particles) const override;

private:
  /// Evaluate either the accepted configuration or one active-particle
  /// proposal.
  pf::Result evaluate(const ParticleSet& particles, int active_particle = -1) const;

  /// Imported parameters and system metadata; immutable after construction and
  /// clone-safe.
  std::shared_ptr<pf::PsiFormer> model_;
  /// Accepted and proposed sign/log-value state used by particle-by-particle
  /// moves.
  double current_sign_ = 1.0, proposed_sign_ = 1.0;
  LogValue proposed_log_value_ = LogValue(0);
  bool has_proposal_           = false;
};

} // namespace qmcplusplus

#endif
