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
/// Shared, versioned native model state used by all clones of one component.
class PsiFormerSharedState;

/**
 * Wavefunction component for a PsiFormer model exported from DeepQMC.
 *
 * Imported parameters remain fixed unless optimization is explicitly enabled.
 * The initial optimization path registers a selected set of canonical flat
 * indices so that native derivatives can be validated end to end before the
 * million-parameter registration and optimizer-storage work is introduced.
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
              std::vector<std::size_t> selected_flat_indices = {});

  /// Copy accepted state and optimizer mappings while dropping any in-flight proposal.
  PsiFormerWF(const PsiFormerWF& other);

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

  /// Store the complete model state and selection in the object-specific VP group.
  void writeVariationalParameters(hdf_archive& output) override;

  /// Restore and validate the authoritative complete model state from a VP group.
  void readVariationalParameters(hdf_archive& input) override;

  /// Return the shared model version for diagnostics and synchronization tests.
  std::size_t parameterVersion() const;

  /// Export current parameters in the flat DeepQMC-compatible HDF5 format.
  void exportParameters(const std::string& path) const;

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
                      bool with_kinetic_parameter_gradient = false);

  /// Return true when at least one selected local parameter maps to a global active variable.
  bool hasActiveParameters() const;

  /// Add selected entries from a native flat gradient to a QMCPACK derivative vector.
  void addSelectedGradient(const std::vector<double>& flat_gradient, Vector<ValueType>& output) const;

  /// Invalidate accepted/proposed caches and record the newly observed shared version.
  void invalidateParameterCaches(std::size_t parameter_version);

  /// Lazily invalidate this clone when another clone changed the shared parameters.
  void synchronizeParameterVersion(std::size_t parameter_version);

  /// Versioned native model protected against evaluation/reset overlap.
  std::shared_ptr<PsiFormerSharedState> model_state_;
  /// Canonically sorted native flat indices represented by this optimization object.
  std::vector<std::size_t> selected_flat_indices_;
  /// Enable registration and derivative work only when requested by input.
  bool optimization_enabled_ = false;
  /// Last shared parameter version observed by this component clone.
  std::size_t observed_parameter_version_ = 0;
  /// Require the generic selected values to agree after an authoritative VP restore.
  bool restore_validation_pending_ = false;
  /// Track whether log_value_ belongs to the current shared parameter version.
  bool accepted_value_valid_ = false;
  /// Accepted and proposed sign/log-value state used by particle-by-particle moves.
  double current_sign_ = 1.0, proposed_sign_ = 1.0;
  LogValue proposed_log_value_ = LogValue(0);
  bool has_proposal_           = false;
};

} // namespace qmcplusplus

#endif
