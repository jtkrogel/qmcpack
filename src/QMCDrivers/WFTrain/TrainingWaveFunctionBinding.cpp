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
#include "QMCWaveFunctions/Fermion/SlaterDet.h"
#include "QMCWaveFunctions/RotatedSPOs.h"
#include "QMCWaveFunctions/LCAO/LCAOrbitalSet.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCHamiltonians/NonLocalECPotential.h"
#include "io/hdf/hdf_archive.h"
#include <exception>
#include <algorithm>
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
struct RotationLayout
{
  std::vector<RotatedSPOs*> unique;
  std::vector<int> aliases;
};

/// Follow determinant references, not the prototype's factory SPO map: clones
/// own remapped SPOs and may share one wrapper between both spin determinants.
RotationLayout rotationLayout(const TrialWaveFunction& wavefunction)
{
  RotationLayout result;
  std::set<std::string> names;
  for (auto& slater : wavefunction.findSD())
    for (int determinant = 0; determinant < slater.get().getNumDets(); ++determinant)
    {
      auto* rotated = dynamic_cast<RotatedSPOs*>(&slater.get().getPhi(determinant));
      if (!rotated)
      {
        result.aliases.push_back(-1);
        continue;
      }
      auto found = std::find(result.unique.begin(), result.unique.end(), rotated);
      if (found == result.unique.end())
      {
        if (rotated->SPOSet::getName().empty() || !names.insert(rotated->SPOSet::getName()).second)
          throw std::invalid_argument("Conventional orbital frames require unique nonempty SPO names");
        found = result.unique.insert(result.unique.end(), rotated);
      }
      result.aliases.push_back(static_cast<int>(found - result.unique.begin()));
    }
  return result;
}

std::vector<double> realValues(const QMCTraits::ValueType* values, std::size_t count)
{
  std::vector<double> result(count);
  for (std::size_t i = 0; i < count; ++i)
    result[i] = std::real(values[i]);
  return result;
}

std::vector<QMCTraits::ValueType> orbitalValues(const std::vector<double>& values)
{
  return {values.begin(), values.end()};
}

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
    if (kind == "sposet" || kind == "rotated_sposet")
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
  const auto rotations = rotationLayout(prototype);
  orbital_rotations_ = !rotations.unique.empty();
  if (orbital_rotations_ && (multideterminant_ || prototype.findSD().size() != 1))
    throw std::invalid_argument("Conventional orbital training requires one single-Slater component");
  for (const auto* rotated : rotations.unique)
    rotated->validatePreparedRotationState(*rotated);
  for (auto& determinant : prototype.findMSD())
    determinant.get().validatePreparedCoefficientState(determinant.get());
  const auto& variables = adapter_.registrationVariables();
  for (std::size_t i = 0; i < variables.size(); ++i)
    if (variables.where(i) >= 0 &&
        (variables.getType(i) == optimize::BACKFLOW_P ||
         (variables.getType(i) == optimize::SPO_P && !orbital_rotations_)))
      throw std::invalid_argument("Conventional training requires qualified CPU LCAO rotations and no backflow");
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
  const bool nonlocal_ecp = hamiltonian.hasPhysicalNLPP();
  if (nonlocal_ecp)
  {
    const auto& provenance = hamiltonian.ecpModelProvenance();
    if (!provenance || !provenance->ordinary_locality_vp || provenance->scientific_definition.empty())
      throw std::runtime_error("Conventional ECP training requires complete parsed-PP content and ordinary-locality VP identity");
    if (multideterminant_ || orbital_rotations_ || prototype_.findSD().size() != 1)
      throw std::runtime_error("Conventional ECP training currently requires one fixed single-Slater component");
    for (const auto& component : prototype_.getOrbitals())
    {
      const auto kind = component->getClassName();
      if (kind != "SlaterDet" && kind != "J1OrbitalSoA" && kind != "TwoBodyJastrow")
        throw std::runtime_error("Conventional ECP training currently admits only fixed Slater and J1/J2");
    }
  }
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
    if (kind != "BareKineticEnergy" && kind != "CoulombPotential" &&
        !(nonlocal_ecp && (kind == "LocalECPotential" || kind == "NonLocalECPotential")))
      throw std::runtime_error("Conventional first training profile requires kinetic plus open Coulomb operators");
    if (const auto* nonlocal = dynamic_cast<const NonLocalECPotential*>(component))
      if (nonlocal->usesDLA() || !nonlocal->usesVirtualParticles())
        throw std::runtime_error("Conventional ECP training requires ordinary locality with virtual particles");
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
  if (nonlocal_ecp)
  {
    const auto& scientific = hamiltonian.ecpModelProvenance()->scientific_definition;
    identity << "ecp_scientific_definition:" << scientific.size() << ':' << scientific << ';';
  }
  physical_definition_ = identity.str();
  hamiltonian_ = &hamiltonian;
  nonlocal_ecp_ = nonlocal_ecp;
  for (auto& target : live_)
    if (target.particles == &prototype_particles_ && !target.hamiltonian)
      target.hamiltonian = &hamiltonian;
}
const StructuredParameterSchema& TrainingWaveFunctionBinding::parameterSchema() const noexcept
{ return adapter_.parameterSchema(); }
StructuredParameterSnapshot TrainingWaveFunctionBinding::snapshotParameters() const
{ return adapter_.snapshotParameters(); }
void TrainingWaveFunctionBinding::attachLive(TrialWaveFunction& wf, ParticleSet& particles, QMCHamiltonian* hamiltonian)
{
  for (auto& target : live_)
    if (target.wavefunction == &wf)
    {
      if (target.particles != &particles ||
          (hamiltonian && target.hamiltonian && hamiltonian != target.hamiltonian))
        throw std::logic_error("Conventional wavefunction has multiple particle targets");
      if (hamiltonian)
        target.hamiltonian = hamiltonian;
      return;
    }
  wf.validatePreparedJastrowActivation(wf);
  live_.push_back({&wf, &particles, hamiltonian});
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
  for (auto* rotated : rotationLayout(*candidate).unique)
    rotated->isolateOptimizationState();
  return candidate;
}
std::size_t TrainingWaveFunctionBinding::publishParameters(
    const StructuredParameterSnapshot& candidate, std::size_t expected_version)
{
  return publishPrepared(candidate, expected_version, nullptr);
}

std::size_t TrainingWaveFunctionBinding::publishOrbitalFrame(
    const StructuredParameterSnapshot& candidate, std::size_t expected_version,
    const ConventionalOrbitalFrame& frame)
{
  return publishPrepared(candidate, expected_version, &frame);
}

std::size_t TrainingWaveFunctionBinding::publishPrepared(
    const StructuredParameterSnapshot& candidate, std::size_t expected_version,
    const ConventionalOrbitalFrame* frame)
{
  std::vector<std::unique_ptr<TrialWaveFunction>> prepared;
  std::vector<std::vector<std::size_t>> component_indices;
  using CoefficientBank = std::pair<MultiSlaterDetTableMethod*, const MultiSlaterDetTableMethod*>;
  std::vector<std::vector<CoefficientBank>> coefficient_banks;
  struct RotationBank { RotatedSPOs* target; RotatedSPOs* prepared; const RotatedSPOs* canonical; };
  std::vector<std::vector<RotationBank>> rotation_banks;
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
    rotation_banks.reserve(live_.size());
    for (const auto& [wf, particles, live_hamiltonian] : live_)
    {
      // Only sampled walker configurations are numerical admission points. The
      // unsampled golden prototype may contain coincident/default coordinates.
      if (particles != &prototype_particles_)
      {
        ParticleSet validation_particles(*particles);
        validation_particles.update();
        auto validation = makePrivateClone(validation_particles);
        if (frame)
          restorePrivateOrbitalFrame(*validation, *frame, candidate);
        adapter_.installSnapshot(*validation, validation_particles, candidate);
        if (!isFiniteTrainingReal(validation->getLogPsi()))
          throw std::runtime_error("Nonfinite conventional candidate wavefunction");
        if (nonlocal_ecp_ && !live_hamiltonian)
          throw std::runtime_error("Conventional ECP publication requires each live Hamiltonian's current grid");
        if (auto* source_hamiltonian = live_hamiltonian ? live_hamiltonian : hamiltonian_)
        {
          auto ham = source_hamiltonian->makeClone(validation_particles, *validation);
          // This keeps the clone's copied current quadrature and cannot draw
          // from any shared live production RNG. No stochastic refresh here.
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
      for (auto* rotated : rotationLayout(*replacement).unique)
        rotated->isolateOptimizationState();
      if (frame)
        restorePrivateOrbitalFrame(*replacement, *frame, candidate);
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
      const auto target_rotations = rotationLayout(*wf);
      const auto prepared_rotations = rotationLayout(*replacement);
      const auto canonical_rotations = rotationLayout(prepared.empty() ? *replacement : *prepared.front());
      if (target_rotations.aliases != prepared_rotations.aliases ||
          target_rotations.aliases != canonical_rotations.aliases)
        throw std::logic_error("Conventional orbital alias topology changed");
      rotation_banks.emplace_back();
      rotation_banks.back().reserve(target_rotations.unique.size());
      for (std::size_t i = 0; i < target_rotations.unique.size(); ++i)
      {
        target_rotations.unique[i]->validatePreparedRotationState(*prepared_rotations.unique[i]);
        target_rotations.unique[i]->validatePreparedRotationState(*canonical_rotations.unique[i]);
        rotation_banks.back().push_back(
            {target_rotations.unique[i], prepared_rotations.unique[i], canonical_rotations.unique[i]});
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
    live_[i].wavefunction->activatePreparedJastrows(*prepared[i], component_indices[i]);
    for (const auto& [target, bank] : coefficient_banks[i])
      target->adoptPreparedCoefficientState(*bank);
    for (const auto& bank : rotation_banks[i])
      bank.target->adoptPreparedRotationState(*bank.prepared, *bank.canonical);
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
  if (nonlocal_ecp_)
    output.write(hamiltonian_->ecpModelProvenance()->source_report, "ecp_source_report");
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
  writeOrbitalFrame(output);
  output.write(std::string(orbital_rotations_ ? "conventional-rotated-lcao-sj-v1"
                          : multideterminant_ ? "conventional-fixed-orbital-mdj-v1"
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
  const std::string expected_cookie = orbital_rotations_ ? "conventional-rotated-lcao-sj-v1"
                                     : multideterminant_ ? "conventional-fixed-orbital-mdj-v1"
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
  // Model exports also carry physical orbital state; active-only readback must
  // still validate the complete frame rather than silently ignoring it.
  if (orbital_rotations_)
  {
    std::uint64_t source_version = 0;
    input.push("orbital_frame", false);
    if (!input.readEntry(source_version, "source_version") ||
        source_version > std::numeric_limits<std::size_t>::max())
      throw std::runtime_error("Conventional orbital export is missing a valid source version");
    input.pop();
    readOrbitalFrame(input, source_version, result);
  }
  return result;
}

void TrainingWaveFunctionBinding::writeOrbitalFrame(hdf_archive& archive) const
{
  if (!orbital_rotations_)
    return;
  const auto layout = rotationLayout(prototype_);
  archive.push("orbital_frame");
  archive.write(std::vector<int>{1, 0, 0}, "format_version");
  archive.write(static_cast<std::uint64_t>(snapshotParameters().version), "source_version");
  archive.write(layout.aliases, "determinant_aliases");
  archive.write(static_cast<std::uint64_t>(layout.unique.size()), "orbital_count");
  for (std::size_t i = 0; i < layout.unique.size(); ++i)
  {
    const auto& rotated = *layout.unique[i];
    const auto& lcao = dynamic_cast<const LCAOrbitalSet&>(*rotated.Phi_);
    const auto& reference = lcao.originalRotationCoefficients();
    archive.push("spo_" + std::to_string(i));
    archive.write(rotated.SPOSet::getName(), "name");
    archive.write(std::vector<std::uint64_t>{lcao.C->rows(), lcao.C->cols()}, "shape");
    const auto& full = rotated.fullRotationCoordinates();
    archive.write(realValues(full.data(), full.size()), "full_rotation");
    archive.write(realValues(lcao.C->data(), lcao.C->size()), "accepted_coefficients");
    archive.write(realValues(reference.data(), reference.size()), "reference_coefficients");
    archive.pop();
  }
  archive.pop();
}

void TrainingWaveFunctionBinding::writeModelCheckpointPayload(hdf_archive& archive) const
{
  writeOrbitalFrame(archive);
}

std::unique_ptr<ModelCheckpointContributor::PreparedRestore>
TrainingWaveFunctionBinding::prepareModelCheckpointRestore(
    hdf_archive& archive, std::size_t source_version, const StructuredParameterSnapshot& candidate)
{
  auto frame = readOrbitalFrame(archive, source_version, candidate);
  if (!orbital_rotations_)
    return {};
  class PreparedFrame final : public ModelCheckpointContributor::PreparedRestore
  {
  public:
    PreparedFrame(TrainingWaveFunctionBinding& binding, ConventionalOrbitalFrame frame)
        : binding_(binding), frame_(std::move(frame))
    {}
    std::size_t publishModel(const StructuredParameterSnapshot& candidate, std::size_t expected_version) override
    {
      return binding_.publishOrbitalFrame(candidate, expected_version, frame_);
    }
  private:
    TrainingWaveFunctionBinding& binding_;
    ConventionalOrbitalFrame frame_;
  };
  return std::make_unique<PreparedFrame>(*this, std::move(frame));
}

ConventionalOrbitalFrame TrainingWaveFunctionBinding::readOrbitalFrame(
    hdf_archive& archive, std::size_t source_version,
    const StructuredParameterSnapshot& candidate) const
{
  ConventionalOrbitalFrame frame;
  frame.source_version = source_version;
  if (!orbital_rotations_)
  {
    if (archive.is_group("orbital_frame"))
      throw std::runtime_error("Unexpected conventional orbital frame");
    return frame;
  }
  const auto layout = rotationLayout(prototype_);
  archive.push("orbital_frame", false);
  std::vector<int> version;
  std::uint64_t saved_version = 0, count = 0;
  if (!archive.readEntry(version, "format_version") ||
      !archive.readEntry(saved_version, "source_version") ||
      !archive.readEntry(count, "orbital_count") ||
      !archive.readEntry(frame.determinant_aliases, "determinant_aliases"))
    throw std::runtime_error("Conventional orbital frame is missing required metadata");
  if (version != std::vector<int>{1, 0, 0} || saved_version != source_version ||
      count != layout.unique.size() || frame.determinant_aliases != layout.aliases)
    throw std::runtime_error("Conventional orbital frame version or alias identity mismatch");
  frame.orbitals.resize(layout.unique.size());
  for (std::size_t i = 0; i < frame.orbitals.size(); ++i)
  {
    auto& orbital = frame.orbitals[i];
    std::vector<std::uint64_t> shape;
    const auto& lcao = dynamic_cast<const LCAOrbitalSet&>(*layout.unique[i]->Phi_);
    const auto& reference = lcao.originalRotationCoefficients();
    archive.push("spo_" + std::to_string(i), false);
    if (!archive.readEntry(orbital.name, "name") || !archive.readEntry(shape, "shape") ||
        !archive.readEntry(orbital.full_rotation, "full_rotation") ||
        !archive.readEntry(orbital.accepted_coefficients, "accepted_coefficients") ||
        !archive.readEntry(orbital.reference_coefficients, "reference_coefficients"))
      throw std::runtime_error("Conventional orbital frame is missing required state");
    archive.pop();
    if (orbital.name != layout.unique[i]->SPOSet::getName() ||
        shape != std::vector<std::uint64_t>{lcao.C->rows(), lcao.C->cols()} ||
        orbital.full_rotation.size() != layout.unique[i]->fullRotationCoordinates().size() ||
        orbital.accepted_coefficients.size() != lcao.C->size() ||
        orbital.reference_coefficients != realValues(reference.data(), reference.size()))
      throw std::runtime_error("Conventional orbital frame shape or reference identity mismatch");
    orbital.rows = shape[0];
    orbital.columns = shape[1];
    for (const auto* values : {&orbital.full_rotation, &orbital.accepted_coefficients,
                               &orbital.reference_coefficients})
      for (double value : *values)
        if (!isFiniteTrainingReal(value))
          throw std::runtime_error("Nonfinite conventional orbital frame");
  }
  archive.pop();
  // Validate the saved C against the complete rotation in an isolated evaluator.
  // No accepted object, population coordinates or RNG state changes here.
  ParticleSet particles(prototype_particles_);
  auto validation = makePrivateClone(particles);
  restorePrivateOrbitalFrame(*validation, frame, candidate);
  return frame;
}

void TrainingWaveFunctionBinding::restorePrivateOrbitalFrame(
    TrialWaveFunction& wavefunction, const ConventionalOrbitalFrame& frame,
    const StructuredParameterSnapshot& candidate) const
{
  const auto layout = rotationLayout(wavefunction);
  if (frame.determinant_aliases != layout.aliases || frame.orbitals.size() != layout.unique.size())
    throw std::runtime_error("Conventional orbital restore alias layout mismatch");
  auto active = adapter_.registrationVariables();
  for (std::size_t i = 0; i < active.size(); ++i)
    if (active.where(i) >= 0)
      active[i] = candidate.values.at(active.where(i));
  for (std::size_t i = 0; i < layout.unique.size(); ++i)
  {
    auto& rotated = *layout.unique[i];
    const auto& saved = frame.orbitals[i];
    auto& lcao = dynamic_cast<LCAOrbitalSet&>(*rotated.Phi_);
    if (saved.name != rotated.SPOSet::getName() || saved.rows != lcao.C->rows() || saved.columns != lcao.C->cols())
      throw std::runtime_error("Conventional orbital restore SPO identity mismatch");
    // Establish registration/checkOut before restoring the wrapper's active
    // values; all rotation and coefficient operations remain private.
    rotated.checkOutVariables(active);
    rotated.restoreFullRotationState(orbitalValues(saved.full_rotation), active);
    LCAOrbitalSet::OffloadValueMatrix accepted(saved.rows, saved.columns);
    LCAOrbitalSet::OffloadValueMatrix reference(saved.rows, saved.columns);
    if (saved.accepted_coefficients.size() != accepted.size() ||
        saved.reference_coefficients.size() != reference.size())
      throw std::runtime_error("Conventional orbital restore coefficient extent mismatch");
    std::copy(saved.accepted_coefficients.begin(), saved.accepted_coefficients.end(), accepted.data());
    std::copy(saved.reference_coefficients.begin(), saved.reference_coefficients.end(), reference.data());
    lcao.restorePreparedRotationCoefficients(accepted, reference);
  }
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
