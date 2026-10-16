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
#include "Particle/DistanceTable.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/Fermion/MultiSlaterDetTableMethod.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "io/hdf/hdf_archive.h"
#include <exception>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <optional>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace qmcplusplus::wftrain
{
namespace
{
/// Reject external fixed-model inputs until their content identity is qualified.
bool hasExternalAssets(xmlNodePtr first)
{
  for (auto* node = first; node; node = node->next)
  {
    if (node->type != XML_ELEMENT_NODE)
      continue;
    for (auto* attribute = node->properties; attribute; attribute = attribute->next)
    {
      const std::string name(reinterpret_cast<const char*>(attribute->name));
      if (name == "href" || name == "src" || name == "file" || name == "filename")
        return true;
    }
    if (hasExternalAssets(node->children))
      return true;
  }
  return false;
}

/// Copy one XML attribute without retaining storage owned by libxml.
std::string attributeText(xmlNodePtr node, const char* name)
{
  xmlChar* raw = xmlGetProp(node, reinterpret_cast<const xmlChar*>(name));
  if (!raw)
    return {};
  const std::string value(reinterpret_cast<const char*>(raw));
  xmlFree(raw);
  return value;
}

/// Resolve supported single-/multideterminant SPO names inside the captured model.
void collectSPOBindings(xmlNodePtr first, std::set<std::string>& definitions, std::set<std::string>& references)
{
  for (auto* node = first; node; node = node->next)
  {
    if (node->type != XML_ELEMENT_NODE)
      continue;
    const std::string kind(reinterpret_cast<const char*>(node->name));
    if (kind == "sposet")
    {
      const auto name = attributeText(node, "name");
      if (!name.empty())
        definitions.insert(name);
    }
    bool has_reference = false;
    for (const auto* attribute : {"sposet", "spo_0", "spo_1", "spo_up", "spo_dn"})
    {
      const auto reference = attributeText(node, attribute);
      if (!reference.empty())
      {
        references.insert(reference);
        has_reference = true;
      }
    }
    if (!has_reference && (kind == "determinant" || kind == "multideterminant"))
      references.insert(""); // Implicit/global SPO lookup is not a qualified inline definition.
    collectSPOBindings(node->children, definitions, references);
  }
}

/// Capture fixed schema/lattice and, only for stationary sources, coordinates.
std::string particleIdentity(const ParticleSet& particles, bool positions)
{
  std::ostringstream out;
  out << std::hexfloat << particles.getName() << ':' << particles.getTotalNum() << ':' << particles.groups();
  const auto& species = particles.getSpeciesSet();
  for (int i = 0; i < species.getTotalNum(); ++i)
  {
    out << '|' << species.speciesName[i];
    for (int attribute = 0; attribute < species.numAttributes(); ++attribute)
      out << ':' << species.attribName[attribute] << '=' << species(attribute, i);
  }
  for (int i = 0; i < particles.getTotalNum(); ++i)
  {
    out << '|' << particles.GroupID[i];
    if (positions)
      for (int d = 0; d < OHMMS_DIM; ++d)
        out << ':' << particles.R[i][d];
  }
  const auto& lattice = particles.getLattice();
  for (int i = 0; i < OHMMS_DIM; ++i)
  {
    out << "|bc=" << lattice.BoxBConds[i];
    for (int j = 0; j < OHMMS_DIM; ++j)
      out << ':' << lattice.R(i, j);
  }
  return out.str();
}
}

TrainingWaveFunctionBinding::TrainingWaveFunctionBinding(TrialWaveFunction& prototype, ParticleSet& particles,
    LegacyWaveFunctionAdapterOptions options, Communicate& communicator)
    : prototype_(prototype), prototype_particles_(particles), communicator_(communicator),
      adapter_("conventional", prototype, options)
{
  prototype.validatePreparedJastrowActivation(prototype);
  multideterminant_ = !prototype.findMSD().empty();
  for (auto& determinant : prototype.findMSD())
    determinant.get().validatePreparedCoefficientState(determinant.get());
  const auto& variables = adapter_.registrationVariables();
  for (std::size_t i = 0; i < variables.size(); ++i)
    if (variables.where(i) >= 0 &&
        (variables.getType(i) == optimize::SPO_P || variables.getType(i) == optimize::BACKFLOW_P))
      throw std::invalid_argument("Conventional coefficient training requires frozen orbitals and no backflow");
  if (auto* node = prototype.getNode())
  {
    external_assets_ = hasExternalAssets(node);
    std::set<std::string> definitions, references;
    collectSPOBindings(node, definitions, references);
    inline_spos_ = (prototype.findSD().empty() && !multideterminant_) || !definitions.empty();
    for (const auto& reference : references)
      inline_spos_ = inline_spos_ && definitions.count(reference) != 0;
    xmlBufferPtr xml = xmlBufferCreate();
    if (!xml) throw std::bad_alloc();
    const int result = xmlNodeDump(xml, node->doc, node, 0, 1);
    if (result >= 0)
      definition_.assign(reinterpret_cast<const char*>(xmlBufferContent(xml)));
    xmlBufferFree(xml);
    if (result < 0) throw std::runtime_error("Cannot capture conventional model XML");
  }
  attachLive(prototype, particles);
}

TrainingWaveFunctionBinding::~TrainingWaveFunctionBinding() = default;

void TrainingWaveFunctionBinding::setHamiltonian(QMCHamiltonian& hamiltonian)
{
  std::map<std::string, std::string> particles;
  particles.emplace(prototype_particles_.getName(), particleIdentity(prototype_particles_, false));
  for (int table = 0; table < prototype_particles_.getNumDistTables(); ++table)
  {
    const auto& origin = prototype_particles_.getDistTable(table).get_origin();
    const auto identity = particleIdentity(origin, &origin != &prototype_particles_);
    auto [entry, inserted] = particles.emplace(origin.getName(), identity);
    if (!inserted && entry->second != identity)
      throw std::runtime_error("Conventional physical identity has ambiguous particle-set names");
  }
  for (int d = 0; d < OHMMS_DIM; ++d)
    if (prototype_particles_.getLattice().BoxBConds[d])
      throw std::runtime_error("Conventional first training profile currently requires open boundaries");
  std::ostringstream identity;
  for (const auto& [name, definition] : particles)
    identity << name.size() << ':' << name << ':' << definition.size() << ':' << definition << ';';
  for (int i = 0; i < hamiltonian.size(); ++i)
  {
    const auto* component = hamiltonian.getComponent(i);
    const auto kind = component->getClassName();
    if (kind != "BareKineticEnergy" && kind != "CoulombPotential")
      throw std::runtime_error("Conventional first training profile requires kinetic plus open Coulomb operators");
    std::ostringstream description;
    component->get(description);
    if (kind == "CoulombPotential")
    {
      const auto text = description.str();
      const std::string prefix = text.find("CoulombAB source=") == 0
          ? "CoulombAB source=" : "CoulombAA source/target ";
      std::istringstream source(text.substr(prefix.size()));
      std::string name;
      source >> name;
      if (!particles.count(name))
        throw std::runtime_error("Conventional Coulomb source is absent from captured particle identity");
    }
    identity << kind << ':' << component->getName() << ':' << description.str() << ';';
  }
  physical_definition_ = identity.str();
  hamiltonian_ = &hamiltonian;
}
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
  auto candidate = makePrivateClone(particles);
  adapter_.installSnapshot(*candidate, particles, snapshot);
  return candidate;
}
std::unique_ptr<TrialWaveFunction> TrainingWaveFunctionBinding::makePrivateClone(ParticleSet& particles) const
{
  const int tables = particles.getNumDistTables();
  auto candidate = prototype_.makeClone(particles);
  if (particles.getNumDistTables() != tables)
    throw std::logic_error("Conventional candidate changed distance-table topology");
  for (auto& determinant : candidate->findMSD())
    determinant.get().isolateOptimizationState();
  return candidate;
}
std::size_t TrainingWaveFunctionBinding::publishParameters(
    const StructuredParameterSnapshot& candidate, std::size_t expected_version)
{
  std::vector<std::unique_ptr<TrialWaveFunction>> prepared;
  std::vector<std::vector<std::size_t>> component_indices;
  using CoefficientBank = std::pair<MultiSlaterDetTableMethod*, const MultiSlaterDetTableMethod*>;
  std::vector<std::vector<CoefficientBank>> coefficient_banks;
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
    component_indices.reserve(live_.size());
    coefficient_banks.reserve(live_.size());
    for (const auto& [wf, particles] : live_)
    {
      // Only sampled walker configurations are numerical admission points. The
      // unsampled golden prototype may contain coincident/default coordinates.
      if (particles != &prototype_particles_)
      {
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
      }
      // Components must bind to the live target's existing distance tables.
      const int tables = particles->getNumDistTables();
      auto replacement = wf->makeClone(*particles);
      if (particles->getNumDistTables() != tables)
        throw std::logic_error("Conventional replacement changed live distance-table topology");
      for (auto& determinant : replacement->findMSD())
        determinant.get().isolateOptimizationState();
      adapter_.resetPrivateParameters(*replacement, candidate);
      component_indices.push_back(wf->validatePreparedJastrowActivation(*replacement));
      // Keep MSD/Phi/C2node/cache identities live. Every accepted clone adopts
      // the same immutable canonical coefficient bank after collective readiness.
      const auto canonical = (prepared.empty() ? *replacement : *prepared.front()).findMSD();
      const auto targets = wf->findMSD();
      if (canonical.size() != targets.size())
        throw std::logic_error("Conventional coefficient-bank topology mismatch");
      coefficient_banks.emplace_back();
      coefficient_banks.back().reserve(targets.size());
      for (std::size_t i = 0; i < targets.size(); ++i)
      {
        targets[i].get().validatePreparedCoefficientState(canonical[i].get());
        coefficient_banks.back().emplace_back(&targets[i].get(), &canonical[i].get());
      }
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
  {
    live_[i].first->activatePreparedJastrows(*prepared[i], component_indices[i]);
    for (const auto& [target, bank] : coefficient_banks[i])
      target->adoptPreparedCoefficientState(*bank);
  }
  return version;
}

void TrainingWaveFunctionBinding::exportModel(const std::filesystem::path& path) const
{
  const auto identity = modelIdentity();
  const auto snapshot = snapshotParameters();
  hdf_archive output;
  if (!output.create(path, H5F_ACC_EXCL))
    throw std::runtime_error("Cannot create conventional model export");
  output.push("conventional_wavefunction");
  output.write(std::vector<int>{1, 0, 0}, "format_version");
  output.write(parameterSchema().providerId(), "provider_id");
  output.write(snapshot.schema_fingerprint, "schema_fingerprint");
  output.write(snapshot.values, "parameters");
  output.write(identity, "model_identity");
  output.write(physical_definition_, "physical_definition");
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
  output.write(definition_, "wavefunction_xml");
  output.write(std::string(multideterminant_ ? "conventional-fixed-orbital-mdj-v1"
                                           : "conventional-fixed-slater-j1-j2-v1"), "completion_cookie");
  output.pop();
  output.flush();
}
StructuredParameterSnapshot TrainingWaveFunctionBinding::readModel(const std::filesystem::path& path) const
{
  hdf_archive input;
  if (!input.open(path, H5F_ACC_RDONLY)) throw std::runtime_error("Cannot read conventional model export");
  input.push("conventional_wavefunction", false);
  std::string cookie, fingerprint, provider, definition, identity, physical;
  std::vector<std::string> names;
  std::vector<int> types, indices;
  std::vector<double> values;
  std::vector<int> version;
  auto result = snapshotParameters();
  input.read(version, "format_version");
  input.read(cookie, "completion_cookie");
  input.read(provider, "provider_id");
  input.read(fingerprint, "schema_fingerprint");
  input.read(result.values, "parameters");
  input.read(definition, "wavefunction_xml");
  input.read(identity, "model_identity");
  input.read(physical, "physical_definition");
  input.read(names, "variable_names");
  input.read(types, "variable_types");
  input.read(indices, "active_indices");
  input.read(values, "registration_values");
  const std::string expected_cookie = multideterminant_ ? "conventional-fixed-orbital-mdj-v1"
                                                       : "conventional-fixed-slater-j1-j2-v1";
  if (version != std::vector<int>{1, 0, 0} || cookie != expected_cookie ||
      fingerprint != parameterSchema().fingerprint() || provider != parameterSchema().providerId() ||
      definition != definition_ || identity != modelIdentity() || physical != physical_definition_)
    throw std::runtime_error("Conventional model export identity mismatch");
  const auto& variables = adapter_.registrationVariables();
  if (names.size() != variables.size() || types.size() != variables.size() ||
      indices.size() != variables.size() || values.size() != variables.size())
    throw std::runtime_error("Conventional model export registration extent mismatch");
  for (std::size_t i = 0; i < variables.size(); ++i)
    if (names[i] != variables.name(i) || types[i] != variables.getType(i) ||
        indices[i] != variables.where(i) || values[i] != std::real(variables[i]))
      throw std::runtime_error("Conventional model export registration mismatch");
  validateTrainingCandidate(parameterSchema(), snapshotParameters(), result);
  return result;
}

std::string TrainingWaveFunctionBinding::modelIdentity() const
{
  if (physical_definition_.empty())
    throw std::runtime_error("Conventional checkpoint/export requires an attached physical Hamiltonian and particle identity");
  if (definition_.empty())
    throw std::runtime_error("Conventional checkpoint/export requires the original complete wavefunction XML");
  if (external_assets_)
    throw std::runtime_error("Conventional checkpoint/export currently requires inline model assets; external asset content identity is not yet qualified");
  if (!inline_spos_)
    throw std::runtime_error("Conventional checkpoint/export requires named SPO definitions inside the wavefunction XML");
  std::uint64_t hash = UINT64_C(14695981039346656037);
  auto mix = [&](const std::string& text) {
    for (const unsigned char byte : text)
    {
      hash ^= byte;
      hash *= UINT64_C(1099511628211);
    }
    hash ^= 0;
    hash *= UINT64_C(1099511628211);
  };
  mix(definition_);
  mix(physical_definition_);
  const auto& variables = adapter_.registrationVariables();
  for (std::size_t i = 0; i < variables.size(); ++i)
  {
    mix(variables.name(i));
    mix(std::to_string(variables.getType(i)));
    mix(std::to_string(variables.where(i)));
    std::ostringstream value;
    value << std::hexfloat << std::real(variables[i]);
    mix(value.str());
  }
  std::ostringstream identity;
  identity << (multideterminant_ ? "conventional-inline-mdj-v1:" : "conventional-inline-sj-v1:") << std::hex << hash;
  return identity.str();
}
}
