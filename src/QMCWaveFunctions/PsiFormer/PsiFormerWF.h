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
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

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
 * Wavefunction component for a PsiFormer model exported from DeepQMC.
 *
 * Imported parameters remain fixed unless optimization is explicitly enabled.
 * The initial optimization path registers a selected set of canonical flat
 * indices so that native derivatives can be validated end to end before the
 * million-parameter registration and optimizer-storage work is introduced.
 */
class PsiFormerWF : public WaveFunctionComponent, public OptimizableObject
{
public:
  /// Load a native model and optionally expose selected flat parameters to QMCPACK.
  PsiFormerWF(std::string name,
              std::string parameters,
              std::string configuration,
              bool optimize = false,
              std::vector<std::size_t> selected_flat_indices = {});

  /// Copy component-local state while sharing the model used at optimizer synchronization points.
  PsiFormerWF(const PsiFormerWF&) = default;

  /// Return the component name used by QMCPACK diagnostics.
  std::string getClassName() const override { return "PsiFormerWF"; }

  /// Mark the sign-changing PsiFormer ansatz as fermionic.
  bool isFermionic() const override { return true; }

  /// Report whether this component explicitly exposes selected parameters.
  bool isOptimizable() const override { return optimization_enabled_; }

  /// Add this component's optimization object when selected-parameter optimization is enabled.
  void extractOptimizableObjectRefs(UniqueOptObjRefs& opt_obj_refs) override;

  /// Insert selected PsiFormer parameters into QMCPACK's global active-variable set.
  void checkInVariablesExclusive(OptVariables& active) override;

  /// Map component-local selected parameters to global active indices.
  void checkOutVariables(const OptVariables& active) override;

  /// Apply active values to the synchronized native flat parameter storage.
  void resetParametersExclusive(const OptVariables& active) override;

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

  /// Add logarithmic and component kinetic-energy derivatives for active parameters.
  void evaluateDerivatives(ParticleSet& particles,
                           const OptVariables& optvars,
                           Vector<ValueType>& dlogpsi,
                           Vector<ValueType>& dhpsioverpsi) override;

  /// Add only logarithmic wavefunction derivatives for active parameters.
  void evaluateDerivativesWF(ParticleSet& particles,
                             const OptVariables& optvars,
                             Vector<ValueType>& dlogpsi) override;

  /// Clone per-component move state while sharing the synchronized native model.
  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet& particles) const override;

private:
  /// Evaluate an accepted/proposed configuration and requested native parameter derivatives.
  pf::Result evaluate(const ParticleSet& particles,
                      int active_particle = -1,
                      bool with_parameter_gradient = false,
                      bool with_kinetic_parameter_gradient = false) const;

  /// Return true when at least one selected local parameter maps to a global active variable.
  bool hasActiveParameters() const;

  /// Add selected entries from a native flat gradient to a QMCPACK derivative vector.
  void addSelectedGradient(const std::vector<double>& flat_gradient, Vector<ValueType>& output) const;

  /// Native model and synchronized parameter leaves shared by component clones.
  std::shared_ptr<pf::PsiFormer> model_;
  /// Canonically sorted native flat indices represented by this optimization object.
  std::vector<std::size_t> selected_flat_indices_;
  /// Enable registration and derivative work only when requested by input.
  bool optimization_enabled_ = false;
  /// Accepted and proposed sign/log-value state used by particle-by-particle moves.
  double current_sign_ = 1.0, proposed_sign_ = 1.0;
  LogValue proposed_log_value_ = LogValue(0);
  bool has_proposal_           = false;
};

} // namespace qmcplusplus

#endif
