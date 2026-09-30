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
struct PsiFormer;
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
  PsiFormerWF(std::string name, std::string parameters, std::string configuration);
  PsiFormerWF(const PsiFormerWF&) = default;
  std::string getClassName() const override { return "PsiFormerWF"; }
  bool isFermionic() const override { return true; }
  LogValue evaluateLog(const ParticleSet& particles,
                       ParticleSet::ParticleGradient& gradient,
                       ParticleSet::ParticleLaplacian& laplacian) override;
  void acceptMove(ParticleSet& particles, int particle_index, bool safe_to_delay = false) override;
  void restore(int particle_index) override;
  PsiValue ratio(ParticleSet& particles, int particle_index) override;
  GradType evalGrad(ParticleSet& particles, int particle_index) override;
  PsiValue ratioGrad(ParticleSet& particles, int particle_index, GradType& gradient) override;
  void registerData(ParticleSet&, WFBufferType&) override {}
  LogValue updateBuffer(ParticleSet& particles, WFBufferType& buffer, bool from_scratch = false) override;
  void copyFromBuffer(ParticleSet&, WFBufferType&) override {}
  void evaluateDerivatives(ParticleSet&, const OptVariables&, Vector<ValueType>&, Vector<ValueType>&) override {}
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
