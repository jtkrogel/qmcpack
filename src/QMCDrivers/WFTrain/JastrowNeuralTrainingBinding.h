//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_JASTROW_NEURAL_TRAINING_BINDING_H
#define QMCPLUSPLUS_JASTROW_NEURAL_TRAINING_BINDING_H

#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"
#include "QMCWaveFunctions/VariableSet.h"
#include "type_traits/template_types.hpp"
#include <filesystem>
#include <memory>
#include <mutex>

class Communicate;
namespace qmcplusplus
{
class TrialWaveFunction;
class PsiFormerWF;
class QMCHamiltonian;
namespace wftrain
{
struct JastrowNeuralScoreWorkspace;
struct JastrowNeuralReplayLane;

/// Bounded CPU binding for one neural determinant factor and ordinary B-spline J1/J2.
/// The binding lock precedes native model locks; publication runs with MC crowds idle.
class JastrowNeuralTrainingBinding final : public StructuredParameterProvider
{
public:
  static constexpr std::size_t MAXIMUM_JASTROW_PARAMETERS = 256;

  JastrowNeuralTrainingBinding(TrialWaveFunction&, ParticleSet&, Communicate&, bool orbital_pretraining);
  ~JastrowNeuralTrainingBinding() override;

  const StructuredParameterSchema& parameterSchema() const noexcept override { return *schema_; }
  StructuredParameterSnapshot snapshotParameters() const override;
  std::size_t publishParameters(const StructuredParameterSnapshot&, std::size_t expected_version) override;
  std::vector<std::string> neuralWeightBlocks() const override;
  std::vector<AffineParameterBlock> affineParameterBlocks() const override;
  std::vector<std::size_t> genericKFACParameterBlocks() const override;
  void observeAffineScores(const ParticleSet&, AffineScoreObserver&, std::vector<double>&) const override;

  /// Stable numerical factor and original provider identity for orbital-only target APIs.
  PsiFormerWF& neural() const;
  void attachLive(TrialWaveFunction&, ParticleSet&, QMCHamiltonian* = nullptr);
  std::string modelIdentity() const;
  /// Export native neural weights plus a matching current-J wavefunction association.
  void exportModel(const std::filesystem::path&, const std::string& associated_model_filename = {}) const;
  /// Validate an export against this binding; this does not restore optimizer/sampler state.
  StructuredParameterSnapshot readModel(const std::filesystem::path&) const;
  std::size_t parameterVersion() const;
  std::size_t replayNumericBytes() const;
  std::size_t jastrowParameterCount() const noexcept { return jastrow_count_; }
  std::size_t jastrowTotalParameterCount() const noexcept { return jastrow_total_count_; }

  /// Pad the orbital gradient with explicitly frozen product coordinates.
  double evaluateOrbitalMSE(const ParticleSet&, const std::vector<double>& target,
                            std::vector<double>& gradient, std::size_t expected_version) const;

  /// Read-only replay retains configurations and shares a single bounded score workspace.
  std::unique_ptr<StreamingDerivativeOperator> makeScoreOperator(
      const RefVector<ParticleSet>&, std::size_t batch, std::size_t offset, std::size_t chunk) const;

  /// Private full-product trial state; no coefficient bank aliases the live model.
  std::unique_ptr<TrialWaveFunction> makeCandidate(ParticleSet&, const StructuredParameterSnapshot&) const;

private:
  /// Existing model/particle identities that receive the same joint publication.
  struct LiveTarget
  {
    TrialWaveFunction* wavefunction;
    ParticleSet* particles;
    QMCHamiltonian* hamiltonian;
  };
  /// Driver-owned prototype and physical particle binding outlive this adapter.
  TrialWaveFunction& prototype_;
  ParticleSet& particles_;
  Communicate& communicator_;
  bool pretraining_;
  std::size_t neural_count_ = 0, jastrow_count_ = 0, jastrow_total_count_ = 0, version_ = 0;
  std::shared_ptr<const StructuredParameterSchema> schema_;
  /// Only the modest conventional coefficient subset uses legacy registration.
  optimize::VariableSet registration_;
  std::vector<std::size_t> active_locations_;
  std::vector<std::size_t> jastrow_indices_;
  std::vector<LiveTarget> live_;
  /// Original export association and canonical scientific identity have distinct roles.
  std::string definition_, coefficient_mapping_, immutable_jastrow_definition_;
  /// Serialize retained replay and the publication preparation/commit boundary.
  mutable std::mutex mutex_;
  mutable std::vector<double> neural_score_;
  std::shared_ptr<JastrowNeuralScoreWorkspace> score_workspace_;
  mutable std::unique_ptr<JastrowNeuralReplayLane> replay_lane_;

  /// Reset only the small conventional subset in detached candidate components.
  void resetJastrows(TrialWaveFunction&, const StructuredParameterSnapshot&) const;
  void validateCandidate(const StructuredParameterSnapshot&, std::size_t expected_version) const;
};
} // namespace wftrain
} // namespace qmcplusplus
#endif
