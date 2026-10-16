//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_TRAINING_WAVEFUNCTION_BINDING_H
#define QMCPLUSPLUS_TRAINING_WAVEFUNCTION_BINDING_H

#include "QMCWaveFunctions/Optimization/LegacyWaveFunctionAdapter.h"
#include "TrainingCheckpoint.h"
#include <filesystem>
#include <utility>

class Communicate;
namespace qmcplusplus
{
class QMCHamiltonian;
class hdf_archive;
namespace wftrain
{
/// Complete accepted orbital frame; active parameters alone are path dependent.
struct ConventionalOrbitalFrame
{
  struct Orbitals
  {
    std::string name;
    std::size_t rows = 0, columns = 0;
    std::vector<double> full_rotation, accepted_coefficients, reference_coefficients;
  };
  std::size_t source_version = 0;
  std::vector<int> determinant_aliases;
  std::vector<Orbitals> orbitals;
};
/** Driver-owned CPU binding for SJ/MDJ with J1/J2/J3 and CI coordinates.
 * Single-Slater CPU LCAO rotations use complete accepted coefficient/rotation
 * frames; their active coordinates alone are not a physical model checkpoint.
 * All fallible model resets occur in private objects, followed by collective
 * readiness and no-throw component swaps. Post-commit live cache refresh remains
 * a fatal boundary: it cannot roll back arbitrary legacy evaluations.
 */
class TrainingWaveFunctionBinding final : public StructuredParameterProvider, public ModelCheckpointContributor
{
public:
  TrainingWaveFunctionBinding(TrialWaveFunction& prototype, ParticleSet& particles,
                             LegacyWaveFunctionAdapterOptions options, Communicate& communicator);
  ~TrainingWaveFunctionBinding() override;
  const StructuredParameterSchema& parameterSchema() const noexcept override;
  StructuredParameterSnapshot snapshotParameters() const override;
  std::size_t publishParameters(const StructuredParameterSnapshot&, std::size_t expected_version) override;
  /// Cold restore publishes the complete saved frame, never an active-only reset.
  std::size_t publishOrbitalFrame(const StructuredParameterSnapshot&, std::size_t expected_version,
                                 const ConventionalOrbitalFrame&);
  void attachLive(TrialWaveFunction&, ParticleSet&, QMCHamiltonian* = nullptr);
  /// Prepare candidate validation against every current population configuration.
  void setHamiltonian(QMCHamiltonian& hamiltonian);
  std::unique_ptr<TrialWaveFunction> makeCandidate(ParticleSet&, const StructuredParameterSnapshot&) const;
  /// Clone and detach mutable CI banks before any variable registration/reset.
  std::unique_ptr<TrialWaveFunction> makePrivateClone(ParticleSet&) const;
  LegacyWaveFunctionAdapter& adapter() noexcept { return adapter_; }
  void exportModel(const std::filesystem::path&) const;
  /// Validate a complete export and return canonical coordinates for comparison.
  /// For orbital models this is NOT an import: publishing the returned active
  /// vector loses accumulated rotation history. Restore the complete frame.
  StructuredParameterSnapshot readModel(const std::filesystem::path&) const;
  /// Fresh model import validates/owns all physical state before collective
  /// publication. This restores no optimizer, sampler or training counters.
  void importModel(const std::filesystem::path&);
  /// Immutable inline model definition and registration identity for exact continuation.
  std::string modelIdentity() const;
  bool hasOrbitalRotations() const noexcept { return orbital_rotations_; }
  void writeOrbitalFrame(hdf_archive&) const;
  ConventionalOrbitalFrame readOrbitalFrame(hdf_archive&, std::size_t source_version,
                                            const StructuredParameterSnapshot&) const;
  void writeModelCheckpointPayload(hdf_archive&) const override;
  std::unique_ptr<ModelCheckpointContributor::PreparedRestore> prepareModelCheckpointRestore(
      hdf_archive&, std::size_t source_version, const StructuredParameterSnapshot&) override;

private:
  TrialWaveFunction& prototype_;
  ParticleSet& prototype_particles_;
  Communicate& communicator_;
  LegacyWaveFunctionAdapter adapter_;
  std::shared_ptr<const optimize::VariableSet> registration_identity_;
  QMCHamiltonian* hamiltonian_ = nullptr;
  struct LiveTarget
  {
    TrialWaveFunction* wavefunction;
    ParticleSet* particles;
    QMCHamiltonian* hamiltonian;
  };
  std::vector<LiveTarget> live_;
  std::string definition_;
  bool external_assets_ = false;
  bool inline_spos_ = true;
  bool multideterminant_ = false;
  bool orbital_rotations_ = false;
  bool nonlocal_ecp_ = false;
  std::string physical_definition_;
  std::size_t publishPrepared(const StructuredParameterSnapshot&, std::size_t,
                              const ConventionalOrbitalFrame*);
  void restorePrivateOrbitalFrame(TrialWaveFunction&, const ConventionalOrbitalFrame&,
                                  const StructuredParameterSnapshot&) const;
  StructuredParameterSnapshot readModelState(const std::filesystem::path&, ConventionalOrbitalFrame*) const;
};
}
}
#endif
