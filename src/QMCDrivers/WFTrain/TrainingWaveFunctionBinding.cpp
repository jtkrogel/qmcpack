//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#include "TrainingWaveFunctionBinding.h"
#include "ParameterUpdateTransaction.h"
#include "TrainingNumerics.h"
#include "Message/Communicate.h"
#include "Message/CommOperators.h"
#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "io/hdf/hdf_archive.h"
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>

namespace qmcplusplus::wftrain
{
TrainingWaveFunctionBinding::TrainingWaveFunctionBinding(TrialWaveFunction& prototype, ParticleSet& particles,
    LegacyWaveFunctionAdapterOptions options, Communicate& communicator)
    : prototype_(prototype), prototype_particles_(particles), communicator_(communicator),
      adapter_("conventional", prototype, options)
{
  prototype.validatePreparedJastrowActivation(prototype);
  attachLive(prototype, particles);
}
TrainingWaveFunctionBinding::~TrainingWaveFunctionBinding() = default;
const StructuredParameterSchema& TrainingWaveFunctionBinding::parameterSchema() const noexcept
{ return adapter_.parameterSchema(); }
StructuredParameterSnapshot TrainingWaveFunctionBinding::snapshotParameters() const
{ return adapter_.snapshotParameters(); }
void TrainingWaveFunctionBinding::attachLive(TrialWaveFunction& wf, ParticleSet& particles)
{
  for (const auto& target : live_)
    if (target.first == &wf)
    {
      if (target.second != &particles)
        throw std::logic_error("Conventional wavefunction has multiple particle targets");
      return;
    }
  wf.validatePreparedJastrowActivation(wf);
  live_.emplace_back(&wf, &particles);
}
std::unique_ptr<TrialWaveFunction> TrainingWaveFunctionBinding::makeCandidate(
    ParticleSet& particles, const StructuredParameterSnapshot& snapshot) const
{
  const int tables = particles.getNumDistTables();
  auto candidate = prototype_.makeClone(particles);
  if (particles.getNumDistTables() != tables)
    throw std::logic_error("Conventional candidate changed distance-table topology");
  adapter_.installSnapshot(*candidate, particles, snapshot);
  return candidate;
}
std::size_t TrainingWaveFunctionBinding::publishParameters(
    const StructuredParameterSnapshot& candidate, std::size_t expected_version)
{
  std::vector<std::unique_ptr<TrialWaveFunction>> prepared;
  std::optional<StructuredParameterSnapshot> detached;
  std::exception_ptr failure;
  try
  {
    const auto old = snapshotParameters();
    validateTrainingCandidate(parameterSchema(), old, candidate);
    if (expected_version != old.version)
      throw std::invalid_argument("Stale conventional training publication");
    if (old.version == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Conventional parameter version overflow");
    detached.emplace(candidate);
    prepared.reserve(live_.size());
    for (const auto& [wf, particles] : live_)
    {
      // Validate numerical state on private particles; no production RNG draws.
      ParticleSet validation_particles(*particles);
      validation_particles.update();
      auto validation = makeCandidate(validation_particles, candidate);
      if (!isFiniteTrainingReal(validation->getLogPsi()))
        throw std::runtime_error("Nonfinite conventional candidate wavefunction");
      if (hamiltonian_)
      {
        auto ham = hamiltonian_->makeClone(validation_particles, *validation);
        if (!isFiniteTrainingReal(ham->evaluateDeterministic(*validation, validation_particles)))
          throw std::runtime_error("Nonfinite conventional candidate energy");
      }
      // Components must bind to the live target's existing distance tables.
      const int tables = particles->getNumDistTables();
      auto replacement = wf->makeClone(*particles);
      if (particles->getNumDistTables() != tables)
        throw std::logic_error("Conventional replacement changed live distance-table topology");
      adapter_.resetPrivateParameters(*replacement, candidate);
      wf->validatePreparedJastrowActivation(*replacement);
      prepared.push_back(std::move(replacement));
    }
  }
  catch (...) { failure = std::current_exception(); }
  int failed = failure ? 1 : 0;
  communicator_.allreduce(failed);
  if (failed)
  {
    if (failure) std::rethrow_exception(failure);
    throw std::runtime_error("Another rank failed to prepare conventional model publication");
  }
  // Every allocation/reset precedes readiness. The exclusive driver publisher
  // commits the prepared detached bank and Jastrow pointers without allocation.
  // The driver must immediately refresh live value/VGL caches.
  const auto version = adapter_.commitPreparedParameters(std::move(*detached));
  for (std::size_t i = 0; i < live_.size(); ++i)
    live_[i].first->activatePreparedJastrows(*prepared[i]);
  return version;
}

void TrainingWaveFunctionBinding::exportModel(const std::filesystem::path& path) const
{
  const auto snapshot = snapshotParameters();
  hdf_archive output;
  if (!output.create(path, H5F_ACC_EXCL))
    throw std::runtime_error("Cannot create conventional model export");
  output.push("conventional_wavefunction");
  output.write(std::vector<int>{1, 0, 0}, "format_version");
  output.write(parameterSchema().providerId(), "provider_id");
  output.write(snapshot.schema_fingerprint, "schema_fingerprint");
  output.write(snapshot.values, "parameters");
  std::vector<std::string> names;
  std::vector<int> types, indices;
  std::vector<double> fixed_values;
  const auto& variables = adapter_.registrationVariables();
  for (std::size_t i = 0; i < variables.size(); ++i)
  {
    names.push_back(variables.name(i));
    types.push_back(variables.getType(i));
    indices.push_back(variables.where(i));
    fixed_values.push_back(std::real(variables[i]));
  }
  // Ordered registration and original XML identify the fixed conventional model;
  // active parameters are authoritative in the separate canonical vector.
  output.write(names, "variable_names");
  output.write(types, "variable_types");
  output.write(indices, "active_indices");
  output.write(fixed_values, "registration_values");
  xmlBufferPtr xml = xmlBufferCreate();
  if (!xml) throw std::bad_alloc();
  std::string definition;
  if (prototype_.getNode())
  {
    xmlNodeDump(xml, prototype_.getNode()->doc, prototype_.getNode(), 0, 1);
    definition.assign(reinterpret_cast<const char*>(xmlBufferContent(xml)));
  }
  xmlBufferFree(xml);
  output.write(definition, "wavefunction_xml");
  output.write(std::string("conventional-fixed-slater-j1-j2-v1"), "completion_cookie");
  output.pop();
  output.flush();
}
StructuredParameterSnapshot TrainingWaveFunctionBinding::readModel(const std::filesystem::path& path) const
{
  hdf_archive input;
  if (!input.open(path, H5F_ACC_RDONLY)) throw std::runtime_error("Cannot read conventional model export");
  input.push("conventional_wavefunction", false);
  std::string cookie, fingerprint, provider;
  std::vector<int> version;
  auto result = snapshotParameters();
  input.read(version, "format_version");
  input.read(cookie, "completion_cookie");
  input.read(provider, "provider_id");
  input.read(fingerprint, "schema_fingerprint");
  input.read(result.values, "parameters");
  if (version != std::vector<int>{1, 0, 0} || cookie != "conventional-fixed-slater-j1-j2-v1" ||
      fingerprint != parameterSchema().fingerprint() || provider != parameterSchema().providerId())
    throw std::runtime_error("Conventional model export identity mismatch");
  validateTrainingCandidate(parameterSchema(), snapshotParameters(), result);
  return result;
}
}
