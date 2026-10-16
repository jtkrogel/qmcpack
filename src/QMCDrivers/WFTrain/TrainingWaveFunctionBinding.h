//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_TRAINING_WAVEFUNCTION_BINDING_H
#define QMCPLUSPLUS_TRAINING_WAVEFUNCTION_BINDING_H

#include "QMCWaveFunctions/Optimization/LegacyWaveFunctionAdapter.h"
#include <filesystem>
#include <utility>

class Communicate;
namespace qmcplusplus
{
class QMCHamiltonian;
namespace wftrain
{
/** Driver-owned CPU binding for the first fixed-Slater J1/J2 training profile.
 * All fallible model resets occur in private objects, followed by collective
 * readiness and no-throw component swaps. Post-commit live cache refresh remains
 * a fatal boundary: it cannot roll back arbitrary legacy evaluations.
 */
class TrainingWaveFunctionBinding final : public StructuredParameterProvider
{
public:
  TrainingWaveFunctionBinding(TrialWaveFunction& prototype, ParticleSet& particles,
                             LegacyWaveFunctionAdapterOptions options, Communicate& communicator);
  ~TrainingWaveFunctionBinding() override;
  const StructuredParameterSchema& parameterSchema() const noexcept override;
  StructuredParameterSnapshot snapshotParameters() const override;
  std::size_t publishParameters(const StructuredParameterSnapshot&, std::size_t expected_version) override;
  void attachLive(TrialWaveFunction&, ParticleSet&);
  /// Prepare candidate validation against every current population configuration.
  void setHamiltonian(QMCHamiltonian& hamiltonian) noexcept { hamiltonian_ = &hamiltonian; }
  std::unique_ptr<TrialWaveFunction> makeCandidate(ParticleSet&, const StructuredParameterSnapshot&) const;
  LegacyWaveFunctionAdapter& adapter() noexcept { return adapter_; }
  void exportModel(const std::filesystem::path&) const;
  StructuredParameterSnapshot readModel(const std::filesystem::path&) const;

private:
  TrialWaveFunction& prototype_;
  ParticleSet& prototype_particles_;
  Communicate& communicator_;
  LegacyWaveFunctionAdapter adapter_;
  QMCHamiltonian* hamiltonian_ = nullptr;
  std::vector<std::pair<TrialWaveFunction*, ParticleSet*>> live_;
};
}
}
#endif
