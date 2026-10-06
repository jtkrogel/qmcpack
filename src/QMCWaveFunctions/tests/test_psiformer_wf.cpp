//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_wf.cpp
 * @brief Selected-parameter QMCPACK integration tests for PsiFormerWF.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Message/Communicate.h"
#include "OhmmsData/Libxml2Doc.h"
#include "Particle/ParticleSet.h"
#include "Particle/VirtualParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeterminant.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWaveFunctionBuilder.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "ResourceCollection.h"
#include "Utilities/RuntimeOptions.h"
#include "io/hdf/hdf_archive.h"
#include "psiformer_test_utils.h"

#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <filesystem>
#include <functional>
#include <future>
#include <sstream>
#include <vector>

namespace qmcplusplus
{
namespace testing
{
/** Access only the crowd-workspace ownership diagnostic used by this test. */
class TestPsiFormerWF
{
public:
  /// Report the scalar evaluator workspaces currently owned by one component clone.
  static PsiFormerWorkspaceDiagnostics directWorkspaceDiagnostics(const PsiFormerWF& component)
  {
    return component.directWorkspaceDiagnosticsForTesting();
  }

  static std::array<std::size_t, 2> directKineticWorkspaceOwnership(
      const PsiFormerWF& leader,
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list)
  {
    return leader.directKineticWorkspaceOwnershipForTesting(wfc_list);
  }

  /// Report optimizer metadata sharing without exposing the implementation type.
  static PsiFormerOptimizationMetadataDiagnostics optimizationMetadataDiagnostics(
      const PsiFormerWF& component)
  {
    return component.optimizationMetadataDiagnosticsForTesting();
  }

  /// Report whether the component currently retains a nonempty participant view.
  static bool hasBatchExecutionPlan(const PsiFormerWF& component)
  {
    return static_cast<bool>(component.batch_execution_plan_);
  }
};
} // namespace testing

namespace
{
using namespace testing::psiformer;
using ValueType = QMCTraits::ValueType;

/// Own a unique scratch directory used only for object-specific VP round trips.
struct ScopedTestDirectory
{
  std::filesystem::path path;

  explicit ScopedTestDirectory(const std::string& label)
  {
    static std::atomic<std::uint64_t> sequence{0};
    path = std::filesystem::temp_directory_path() /
        ("qmcpack_psiformer_" + label + "_" +
         std::to_string(static_cast<long long>(getpid())) + "_" +
         std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
    std::filesystem::create_directories(path);
  }

  ~ScopedTestDirectory()
  {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

/// Construct the electron ParticleSet matching one generated LiH fixture.
ParticleSet makeLiHElectrons(const SimulationCell& simulation_cell,
                             const std::string& system = "lih")
{
  const Geometry geometry = makeGeometry(system);
  ParticleSet electrons(simulation_cell);
  electrons.setName("e");
  electrons.create({static_cast<int>(geometry.nup),
                    static_cast<int>(geometry.electrons.size() / 3 - geometry.nup)});
  SpeciesSet& species = electrons.getSpeciesSet();
  species.addSpecies("u");
  species.addSpecies("d");
  const int mass = species.addAttribute("mass");
  species(mass, 0) = 1.0;
  species(mass, 1) = 1.0;
  electrons.resetGroups();
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      electrons.R[electron][dimension] = geometry.electrons[3 * electron + dimension];
  electrons.update();
  return electrons;
}

/// Construct source ions whose positions and charges exactly match the generated export.
std::unique_ptr<ParticleSet> makeLiHIons(const SimulationCell& simulation_cell, const std::string& system = "lih")
{
  const Geometry geometry = makeGeometry(system);
  auto ions              = std::make_unique<ParticleSet>(simulation_cell);
  ions->setName("ion0");
  ions->create(std::vector<int>(geometry.charges.size(), 1));
  SpeciesSet& species = ions->getSpeciesSet();
  const int charge     = species.addAttribute("charge");
  for (std::size_t nucleus = 0; nucleus < geometry.charges.size(); ++nucleus)
  {
    species.addSpecies("ion_" + std::to_string(nucleus));
    species(charge, nucleus) = geometry.charges[nucleus];
    for (int dimension = 0; dimension < 3; ++dimension)
      ions->R[nucleus][dimension] = geometry.nuclei[3 * nucleus + dimension];
  }
  ions->resetGroups();
  ions->update();
  return ions;
}

/// Add the gradient of a fixed linear log factor to emulate composition with another component.
std::vector<double> addLinearLogGradient(ParticleSet& electrons)
{
  std::vector<double> extra_gradient(3 * electrons.getTotalNum());
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      const std::size_t coordinate   = 3 * electron + dimension;
      extra_gradient[coordinate]     = 0.01 * (dimension + 1);
      electrons.G[electron][dimension] += ValueType(extra_gradient[coordinate]);
    }
  return extra_gradient;
}

/// Evaluate the kinetic local energy from the logarithmic gradients and Laplacians in ParticleSet.
double kineticEnergy(const ParticleSet& electrons)
{
  double kinetic = 0.0;
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
  {
    double squared_gradient = 0.0;
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      const double real_part = std::real(electrons.G[electron][dimension]);
      const double imag_part = std::imag(electrons.G[electron][dimension]);
      squared_gradient += real_part * real_part + imag_part * imag_part;
    }
    kinetic -= 0.5 * (std::real(electrons.L[electron]) + squared_gradient);
  }
  return kinetic;
}

/// Evaluate the three straight-Coulomb potential terms for the LiH fixture.
double coulombPotential(const ParticleSet& electrons)
{
  const Geometry geometry = makeGeometry("lih");
  auto electronNucleusDistance = [&electrons, &geometry](int electron, std::size_t nucleus) {
    double squared_distance = 0.0;
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      const double displacement =
          electrons.R[electron][dimension] - geometry.nuclei[3 * nucleus + dimension];
      squared_distance += displacement * displacement;
    }
    return std::sqrt(squared_distance);
  };

  double potential = 0.0;
  for (int first = 0; first < electrons.getTotalNum(); ++first)
    for (int second = first + 1; second < electrons.getTotalNum(); ++second)
    {
      double squared_distance = 0.0;
      for (int dimension = 0; dimension < 3; ++dimension)
      {
        const double displacement =
            electrons.R[first][dimension] - electrons.R[second][dimension];
        squared_distance += displacement * displacement;
      }
      potential += 1.0 / std::sqrt(squared_distance);
    }

  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (std::size_t nucleus = 0; nucleus < geometry.charges.size(); ++nucleus)
      potential -= geometry.charges[nucleus] / electronNucleusDistance(electron, nucleus);

  for (std::size_t first = 0; first < geometry.charges.size(); ++first)
    for (std::size_t second = first + 1; second < geometry.charges.size(); ++second)
    {
      double squared_distance = 0.0;
      for (int dimension = 0; dimension < 3; ++dimension)
      {
        const double displacement =
            geometry.nuclei[3 * first + dimension] - geometry.nuclei[3 * second + dimension];
        squared_distance += displacement * displacement;
      }
      potential += geometry.charges[first] * geometry.charges[second] / std::sqrt(squared_distance);
    }
  return potential;
}

/// Capture the high-level observables and selected derivatives of one component.
struct ComponentSnapshot
{
  double log_value;
  double phase;
  double wavefunction_value;
  double local_energy;
  std::vector<double> gradient;
  std::vector<double> laplacian;
  std::vector<double> log_parameter_derivative;
  std::vector<double> kinetic_parameter_derivative;
};

/// Register a component's selected parameters through the normal QMCPACK mapping path.
OptVariables registerSelectedParameters(PsiFormerWF& component)
{
  OptVariables active;
  component.checkInVariablesExclusive(active);
  active.resetIndex();
  component.checkOutVariables(active);
  return active;
}

/// Build a production-shape PsiFormer directly from XML and QMCPACK ParticleSets.
std::unique_ptr<PsiFormerWF> buildInternalPsiFormer(PsiFormerWaveFunctionBuilder& builder,
                                                   const std::string& name,
                                                   std::uint64_t seed,
                                                   const std::string& system = "all_electron",
                                                   const std::string& selected_indices = "0 514")
{
  std::ostringstream xml;
  xml << "<psiformer name=\"" << name
      << "\" initialization=\"" << psiformer::DEEPQMC_PSIFORMER_V1
      << "\" initialization_seed=\"" << seed
      << "\" source=\"ion0\" system=\"" << system << "\" optimize=\"yes\" "
         "optimize_scope=\"indices\" optimize_indices=\"" << selected_indices << "\"/>";

  Libxml2Document document;
  if (!document.parseFromString(xml.str()))
    throw std::runtime_error("Unable to parse internally initialized PsiFormer test XML");
  std::unique_ptr<WaveFunctionComponent> component = builder.buildComponent(document.getRoot());
  auto* psiformer_component = dynamic_cast<PsiFormerWF*>(component.get());
  if (psiformer_component == nullptr)
    throw std::runtime_error("PsiFormer builder returned the wrong component type");
  component.release();
  return std::unique_ptr<PsiFormerWF>(psiformer_component);
}

/// Evaluate the component from scratch and flatten its public QMCPACK outputs.
ComponentSnapshot evaluateComponent(PsiFormerWF& component, ParticleSet& electrons, const OptVariables& active)
{
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  const PsiFormerWF::LogValue log_value = component.evaluateLog(electrons, electrons.G, electrons.L);

  Vector<ValueType> dlogpsi(active.size());
  Vector<ValueType> dhpsioverpsi(active.size());
  dlogpsi      = ValueType(0);
  dhpsioverpsi = ValueType(0);
  component.evaluateDerivatives(electrons, active, dlogpsi, dhpsioverpsi);

  ComponentSnapshot snapshot;
  snapshot.log_value          = std::real(log_value);
  snapshot.phase              = std::imag(log_value);
  snapshot.wavefunction_value = std::real(std::exp(log_value));
  snapshot.local_energy       = kineticEnergy(electrons) + coulombPotential(electrons);
  snapshot.gradient.reserve(3 * electrons.getTotalNum());
  snapshot.laplacian.reserve(electrons.getTotalNum());
  snapshot.log_parameter_derivative.reserve(active.size());
  snapshot.kinetic_parameter_derivative.reserve(active.size());
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      snapshot.gradient.push_back(std::real(electrons.G[electron][dimension]));
    snapshot.laplacian.push_back(std::real(electrons.L[electron]));
  }
  for (int parameter = 0; parameter < active.size(); ++parameter)
  {
    snapshot.log_parameter_derivative.push_back(std::real(dlogpsi[parameter]));
    snapshot.kinetic_parameter_derivative.push_back(std::real(dhpsioverpsi[parameter]));
  }
  return snapshot;
}

/// Compare complete component snapshots at deterministic native-evaluator tolerance.
void checkComponentSnapshot(const ComponentSnapshot& actual, const ComponentSnapshot& expected)
{
  CHECK(actual.log_value == Catch::Approx(expected.log_value).epsilon(2e-10).margin(2e-10));
  CHECK(actual.phase == Catch::Approx(expected.phase).epsilon(2e-10).margin(2e-10));
  CHECK(actual.wavefunction_value ==
        Catch::Approx(expected.wavefunction_value).epsilon(2e-9).margin(1e-24));
  CHECK(actual.local_energy == Catch::Approx(expected.local_energy).epsilon(2e-9).margin(2e-9));
  REQUIRE(actual.gradient.size() == expected.gradient.size());
  REQUIRE(actual.laplacian.size() == expected.laplacian.size());
  REQUIRE(actual.log_parameter_derivative.size() == expected.log_parameter_derivative.size());
  REQUIRE(actual.kinetic_parameter_derivative.size() == expected.kinetic_parameter_derivative.size());

  for (std::size_t index = 0; index < actual.gradient.size(); ++index)
    CHECK(actual.gradient[index] == Catch::Approx(expected.gradient[index]).epsilon(2e-9).margin(2e-9));
  for (std::size_t index = 0; index < actual.laplacian.size(); ++index)
    CHECK(actual.laplacian[index] == Catch::Approx(expected.laplacian[index]).epsilon(2e-8).margin(2e-8));
  for (std::size_t index = 0; index < actual.log_parameter_derivative.size(); ++index)
    CHECK(actual.log_parameter_derivative[index] ==
          Catch::Approx(expected.log_parameter_derivative[index]).epsilon(2e-8).margin(2e-8));
  for (std::size_t index = 0; index < actual.kinetic_parameter_derivative.size(); ++index)
    CHECK(actual.kinetic_parameter_derivative[index] ==
          Catch::Approx(expected.kinetic_parameter_derivative[index]).epsilon(2e-8).margin(2e-8));
}

} // namespace

TEST_CASE("PsiFormer builder is fixed by default and parses selected indices", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  WaveFunctionComponentBuilder::PSetMap particle_sets;
  auto ions = makeLiHIons(simulation_cell);
  particle_sets.emplace(ions->getName(), std::move(ions));
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, electrons, particle_sets);

  std::ostringstream fixed_xml;
  fixed_xml << "<psiformer name=\"pf_fixed\" parameters=\"" << files.parameters.string()
            << "\" configuration=\"" << files.configuration.string() << "\"/>";
  Libxml2Document fixed_document;
  REQUIRE(fixed_document.parseFromString(fixed_xml.str()));
  std::unique_ptr<WaveFunctionComponent> fixed = builder.buildComponent(fixed_document.getRoot());
  REQUIRE(fixed != nullptr);
  CHECK_FALSE(fixed->isOptimizable());
  UniqueOptObjRefs fixed_refs;
  fixed->extractOptimizableObjectRefs(fixed_refs);
  CHECK(fixed_refs.empty());

  std::ostringstream optimized_xml;
  optimized_xml << "<psiformer name=\"pf_selected\" parameters=\"" << files.parameters.string()
                << "\" configuration=\"" << files.configuration.string()
                << "\" system=\"all_electron\" optimize=\"yes\" optimize_scope=\"indices\" "
                   "optimize_indices=\"127, 0 1\"/>";
  Libxml2Document optimized_document;
  REQUIRE(optimized_document.parseFromString(optimized_xml.str()));
  std::unique_ptr<WaveFunctionComponent> optimized = builder.buildComponent(optimized_document.getRoot());
  REQUIRE(optimized != nullptr);
  CHECK(optimized->isOptimizable());
  UniqueOptObjRefs optimized_refs;
  optimized->extractOptimizableObjectRefs(optimized_refs);
  REQUIRE(optimized_refs.size() == 1);

  std::ostringstream all_xml;
  all_xml << "<psiformer name=\"pf_all\" parameters=\"" << files.parameters.string()
          << "\" configuration=\"" << files.configuration.string()
          << "\" system=\"all_electron\" optimize=\"yes\" optimize_scope=\"all\"/>";
  Libxml2Document all_document;
  REQUIRE(all_document.parseFromString(all_xml.str()));
  std::unique_ptr<WaveFunctionComponent> all = builder.buildComponent(all_document.getRoot());
  auto* all_psiformer = dynamic_cast<PsiFormerWF*>(all.get());
  REQUIRE(all_psiformer != nullptr);
  OptVariables all_active = registerSelectedParameters(*all_psiformer);
  const std::vector<Leaf> layout = makeLayout(4, 2);
  const std::size_t expected_parameter_count = std::accumulate(
      layout.begin(), layout.end(), std::size_t{0}, [](std::size_t count, const Leaf& leaf) {
        return count + product(leaf.shape);
      });
  CHECK(all_active.size() == expected_parameter_count);

  // Full-network clones share the O(P) selection, names, values, and global
  // indices. The inherited per-object VariableSet remains empty, preventing a
  // second O(P) copy from being created by OptimizableObject's copy constructor.
  std::unique_ptr<WaveFunctionComponent> all_clone_storage = all_psiformer->makeClone(electrons);
  auto* all_clone = dynamic_cast<PsiFormerWF*>(all_clone_storage.get());
  REQUIRE(all_clone != nullptr);
  const auto all_diagnostics = testing::TestPsiFormerWF::optimizationMetadataDiagnostics(*all_psiformer);
  const auto clone_diagnostics = testing::TestPsiFormerWF::optimizationMetadataDiagnostics(*all_clone);
  CHECK(all_diagnostics.identity == clone_diagnostics.identity);
  CHECK(all_diagnostics.shared_owner_count == 2);
  CHECK(clone_diagnostics.shared_owner_count == 2);
  CHECK(all_diagnostics.selected_index_count == expected_parameter_count);
  CHECK(all_diagnostics.shared_variable_count == expected_parameter_count);
  CHECK(all_diagnostics.mapped_variable_count == expected_parameter_count);
  CHECK(all_diagnostics.inherited_variable_count == 0);
  CHECK(clone_diagnostics.inherited_variable_count == 0);

  std::ostringstream unsupported_xml;
  unsupported_xml << "<psiformer parameters=\"" << files.parameters.string() << "\" configuration=\""
                  << files.configuration.string()
                  << "\" system=\"all_electron\" optimize=\"yes\" optimize_scope=\"all\" "
                     "optimize_indices=\"0\"/>";
  Libxml2Document unsupported_document;
  REQUIRE(unsupported_document.parseFromString(unsupported_xml.str()));
  CHECK_THROWS_AS(builder.buildComponent(unsupported_document.getRoot()), std::invalid_argument);

  std::ostringstream undeclared_system_xml;
  undeclared_system_xml << "<psiformer parameters=\"" << files.parameters.string() << "\" configuration=\""
                        << files.configuration.string()
                        << "\" optimize=\"yes\" optimize_indices=\"0\"/>";
  Libxml2Document undeclared_system_document;
  REQUIRE(undeclared_system_document.parseFromString(undeclared_system_xml.str()));
  CHECK_THROWS_WITH(builder.buildComponent(undeclared_system_document.getRoot()),
                    Catch::Matchers::ContainsSubstring("requires system="));

  ParticleSet spinor_electrons = makeLiHElectrons(simulation_cell);
  spinor_electrons.setSpinor(true);
  PsiFormerWaveFunctionBuilder spinor_builder(OHMMS::Controller, spinor_electrons, particle_sets);
  CHECK_THROWS_WITH(spinor_builder.buildComponent(fixed_document.getRoot()),
                    Catch::Matchers::ContainsSubstring("spinor"));
}

TEST_CASE("PsiFormer builder validates internal initialization XML", "[wavefunction][psiformer][initialization]")
{
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  WaveFunctionComponentBuilder::PSetMap particle_sets;
  auto ions = makeLiHIons(simulation_cell);
  particle_sets.emplace(ions->getName(), std::move(ions));
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, electrons, particle_sets);

  auto check_invalid = [&builder](const std::string& xml, const std::string& message) {
    Libxml2Document document;
    REQUIRE(document.parseFromString(xml));
    CHECK_THROWS_WITH(builder.buildComponent(document.getRoot()),
                      Catch::Matchers::ContainsSubstring(message));
  };

  check_invalid(
      "<psiformer initialization=\"deepqmc_psiformer_v1\" initialization_seed=\"7\" "
      "parameters=\"parameters.h5\" source=\"ion0\" system=\"all_electron\"/>",
      "cannot be combined");
  check_invalid(
      "<psiformer parameters=\"parameters.h5\" configuration=\"configuration.h5\" "
      "initialization_seed=\"7\"/>",
      "requires internal initialization");
  check_invalid(
      "<psiformer initialization=\"deepqmc_psiformer_v1\" initialization_seed=\"7\" "
      "source=\"ion0\"/>",
      "requires explicit system");
  check_invalid(
      "<psiformer initialization=\"deepqmc_psiformer_v1\" initialization_seed=\"7\" "
      "system=\"all_electron\"/>",
      "requires an explicit source");
  check_invalid(
      "<psiformer initialization=\"unversioned\" initialization_seed=\"7\" "
      "source=\"ion0\" system=\"all_electron\"/>",
      "Unsupported PsiFormer initialization profile");
  check_invalid(
      "<psiformer initialization=\"deepqmc_psiformer_v1\" initialization_seed=\"-1\" "
      "source=\"ion0\" system=\"all_electron\"/>",
      "must be an unsigned integer");
}

TEST_CASE("PsiFormer internal initialization evaluates and restores without model files",
          "[wavefunction][psiformer][initialization]")
{
  constexpr std::uint64_t initialization_seed = 17;
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  WaveFunctionComponentBuilder::PSetMap particle_sets;
  auto ions = makeLiHIons(simulation_cell);
  particle_sets.emplace(ions->getName(), std::move(ions));
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, electrons, particle_sets);

  std::unique_ptr<PsiFormerWF> original =
      buildInternalPsiFormer(builder, "pf_internal", initialization_seed);
  OptVariables active = registerSelectedParameters(*original);
  REQUIRE(active.size() == 2);

  // Identical seeds reproduce all public values, while changing the seed
  // changes a selected parameter in the first random tensor. Index zero is an
  // analytic cusp constant and remains seed independent.
  ParticleSet repeated_electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<PsiFormerWF> repeated =
      buildInternalPsiFormer(builder, "pf_internal_repeated", initialization_seed);
  OptVariables repeated_active = registerSelectedParameters(*repeated);
  REQUIRE(repeated_active.size() == active.size());
  for (int parameter = 0; parameter < active.size(); ++parameter)
    CHECK(std::real(repeated_active[parameter]) == std::real(active[parameter]));

  std::unique_ptr<PsiFormerWF> changed_seed =
      buildInternalPsiFormer(builder, "pf_internal_changed", initialization_seed + 1);
  OptVariables changed_active = registerSelectedParameters(*changed_seed);
  REQUIRE(changed_active.size() == active.size());
  CHECK(std::real(changed_active[0]) == std::real(active[0]));
  CHECK(std::real(changed_active[1]) != std::real(active[1]));
  changed_seed.reset();

  const ComponentSnapshot baseline = evaluateComponent(*original, electrons, active);
  const ComponentSnapshot repeated_snapshot =
      evaluateComponent(*repeated, repeated_electrons, repeated_active);
  checkComponentSnapshot(repeated_snapshot, baseline);
  repeated.reset();

  CHECK(std::isfinite(baseline.log_value));
  CHECK(std::isfinite(baseline.phase));
  CHECK(std::isfinite(baseline.wavefunction_value));
  CHECK(std::isfinite(baseline.local_energy));
  for (double value : baseline.gradient)
    CHECK(std::isfinite(value));
  for (double value : baseline.laplacian)
    CHECK(std::isfinite(value));
  for (double value : baseline.log_parameter_derivative)
    CHECK(std::isfinite(value));
  for (double value : baseline.kinetic_parameter_derivative)
    CHECK(std::isfinite(value));

  // Update through normal optimizer registration, then persist both the
  // generic selected list and the complete object-specific model payload.
  active[0] += 1.5e-4;
  active[1] -= 2.5e-4;
  original->resetParametersExclusive(active);
  const ComponentSnapshot expected = evaluateComponent(*original, electrons, active);

  ScopedTestDirectory files("internal_restart");
  const std::filesystem::path state_path = files.path / "psiformer_internal.vp.h5";
  hdf_archive output;
  active.writeToHDF(state_path.string(), output);
  original->writeVariationalParameters(output);
  output.close();

  ParticleSet restored_electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<PsiFormerWF> restored =
      buildInternalPsiFormer(builder, "pf_internal", initialization_seed);
  OptVariables restored_active = registerSelectedParameters(*restored);
  hdf_archive input;
  restored_active.readFromHDF(state_path.string(), input);
  restored->readVariationalParameters(input);
  input.close();
  restored->resetParametersExclusive(restored_active);
  const ComponentSnapshot restarted =
      evaluateComponent(*restored, restored_electrons, restored_active);
  checkComponentSnapshot(restarted, expected);

  // The seed is part of restart identity, so an otherwise compatible model
  // cannot silently accept a payload produced from another initial state.
  std::unique_ptr<PsiFormerWF> mismatched_seed =
      buildInternalPsiFormer(builder, "pf_internal", initialization_seed + 1);
  hdf_archive mismatch_input;
  REQUIRE(mismatch_input.open(state_path, H5F_ACC_RDONLY));
  CHECK_THROWS_WITH(mismatched_seed->readVariationalParameters(mismatch_input),
                    Catch::Matchers::ContainsSubstring("initialization seed"));
  mismatch_input.close();
}

TEST_CASE("PsiFormer internal initialization uses the canonical pseudo-LiH layout",
          "[wavefunction][psiformer][initialization][ecp]")
{
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell, "lih_pp");
  WaveFunctionComponentBuilder::PSetMap particle_sets;
  auto ions = makeLiHIons(simulation_cell, "lih_pp");
  particle_sets.emplace(ions->getName(), std::move(ions));
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, electrons, particle_sets);

  std::unique_ptr<PsiFormerWF> component = buildInternalPsiFormer(
      builder, "pf_internal_pseudo", 23, "pseudopotential", "0 257");
  OptVariables active = registerSelectedParameters(*component);
  REQUIRE(active.size() == 2);

  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  const PsiFormerWF::LogValue log_value =
      component->evaluateLog(electrons, electrons.G, electrons.L);
  CHECK(std::isfinite(std::real(log_value)));
  CHECK(std::isfinite(std::imag(log_value)));
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      CHECK(std::isfinite(std::real(electrons.G[electron][dimension])));
    CHECK(std::isfinite(std::real(electrons.L[electron])));
  }

  Vector<ValueType> dlogpsi(active.size(), ValueType(0));
  component->evaluateDerivativesWF(electrons, active, dlogpsi);
  for (const ValueType derivative : dlogpsi)
    CHECK(std::isfinite(std::real(derivative)));
}

TEST_CASE("PsiFormer specialized public evaluation paths preserve high-level results", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF component(
      "pf_requests", files.parameters.string(), files.configuration.string(), true, {0, 127});
  OptVariables active = registerSelectedParameters(component);

  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  const PsiFormerWF::LogValue reference_log = component.evaluateLog(electrons, electrons.G, electrons.L);
  const ParticleSet::ParticleGradient reference_gradient = electrons.G;

  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
  {
    const PsiFormerWF::GradType active_gradient = component.evalGrad(electrons, electron);
    for (int dimension = 0; dimension < 3; ++dimension)
      CHECK(std::real(active_gradient[dimension]) ==
            Catch::Approx(std::real(reference_gradient[electron][dimension])).epsilon(2e-9).margin(2e-9));
  }

  constexpr int moved_electron = 1;
  const ParticleSet::SingleParticlePos displacement{0.08, -0.03, 0.02};
  ParticleSet moved = makeLiHElectrons(simulation_cell);
  moved.R[moved_electron] += displacement;
  moved.update();
  PsiFormerWF moved_reference("pf_requests_moved", files.parameters.string(), files.configuration.string());
  moved.G = ValueType(0);
  moved.L = ValueType(0);
  const PsiFormerWF::LogValue moved_log = moved_reference.evaluateLog(moved, moved.G, moved.L);
  const auto expected_ratio             = std::exp(moved_log - reference_log);

  electrons.makeMove(moved_electron, displacement);
  const ValueType ratio = component.ratio(electrons, moved_electron);
  CHECK(std::real(ratio) == Catch::Approx(std::real(expected_ratio)).epsilon(2e-9).margin(2e-12));
  CHECK(std::imag(ratio) == Catch::Approx(std::imag(expected_ratio)).epsilon(2e-9).margin(2e-12));
  component.restore(moved_electron);
  electrons.rejectMove(moved_electron);

  electrons.makeMove(moved_electron, displacement);
  PsiFormerWF::GradType proposed_gradient;
  const ValueType ratio_with_gradient = component.ratioGrad(electrons, moved_electron, proposed_gradient);
  CHECK(std::real(ratio_with_gradient) ==
        Catch::Approx(std::real(expected_ratio)).epsilon(2e-9).margin(2e-12));
  CHECK(std::imag(ratio_with_gradient) ==
        Catch::Approx(std::imag(expected_ratio)).epsilon(2e-9).margin(2e-12));
  for (int dimension = 0; dimension < 3; ++dimension)
    CHECK(std::real(proposed_gradient[dimension]) ==
          Catch::Approx(std::real(moved.G[moved_electron][dimension])).epsilon(2e-9).margin(2e-9));
  component.restore(moved_electron);
  electrons.rejectMove(moved_electron);

  Vector<ValueType> score_only(active.size());
  Vector<ValueType> score_with_kinetic(active.size());
  Vector<ValueType> kinetic(active.size());
  score_only         = ValueType(0.375);
  score_with_kinetic = ValueType(-0.125);
  kinetic            = ValueType(0.625);
  component.evaluateDerivativesWF(electrons, active, score_only);
  component.evaluateDerivatives(electrons, active, score_with_kinetic, kinetic);
  for (int parameter = 0; parameter < active.size(); ++parameter)
    CHECK(std::real(score_only[parameter]) - 0.375 ==
          Catch::Approx(std::real(score_with_kinetic[parameter]) + 0.125).epsilon(2e-10).margin(2e-10));
}

TEST_CASE("PsiFormer clone-local evaluator workspaces are allocated on demand",
          "[wavefunction][psiformer][memory]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet source_electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF source("pf_lazy_source", files.parameters.string(), files.configuration.string());

  // Establish accepted state once. Cloning preserves that state but deliberately
  // does not copy the source's now-populated full-VGL evaluator scratch.
  source_electrons.G = ValueType(0);
  source_electrons.L = ValueType(0);
  source.evaluateLog(source_electrons, source_electrons.G, source_electrons.L);

  auto make_lazy_clone = [&](ParticleSet& electrons) {
    std::unique_ptr<WaveFunctionComponent> storage = source.makeClone(electrons);
    auto* component = dynamic_cast<PsiFormerWF*>(storage.get());
    REQUIRE(component != nullptr);
    return storage;
  };
  auto as_psiformer = [](std::unique_ptr<WaveFunctionComponent>& storage) -> PsiFormerWF& {
    auto* component = dynamic_cast<PsiFormerWF*>(storage.get());
    REQUIRE(component != nullptr);
    return *component;
  };
  auto check_empty = [](const PsiFormerWF& component) {
    const auto diagnostics = testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
    CHECK(diagnostics.ownedWorkspaceCount() == 0);
    CHECK(diagnostics.accountedBytes() == 0);
  };

  ParticleSet value_electrons  = makeLiHElectrons(simulation_cell);
  ParticleSet full_electrons   = makeLiHElectrons(simulation_cell);
  ParticleSet active_electrons = makeLiHElectrons(simulation_cell);
  ParticleSet batch_electrons  = makeLiHElectrons(simulation_cell);
  ParticleSet untouched_electrons = makeLiHElectrons(simulation_cell);
  auto value_storage     = make_lazy_clone(value_electrons);
  auto full_storage      = make_lazy_clone(full_electrons);
  auto active_storage    = make_lazy_clone(active_electrons);
  auto batch_storage     = make_lazy_clone(batch_electrons);
  auto untouched_storage = make_lazy_clone(untouched_electrons);
  PsiFormerWF& value_component     = as_psiformer(value_storage);
  PsiFormerWF& full_component      = as_psiformer(full_storage);
  PsiFormerWF& active_component    = as_psiformer(active_storage);
  PsiFormerWF& batch_component     = as_psiformer(batch_storage);
  PsiFormerWF& untouched_component = as_psiformer(untouched_storage);

  check_empty(value_component);
  check_empty(full_component);
  check_empty(active_component);
  check_empty(batch_component);
  check_empty(untouched_component);

  value_electrons.makeMove(0, ParticleSet::SingleParticlePos{0.01, -0.02, 0.015});
  value_component.ratio(value_electrons, 0);
  value_component.restore(0);
  value_electrons.rejectMove(0);
  const auto value_diagnostics =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(value_component);
  CHECK(value_diagnostics.owns_value_workspace);
  CHECK(value_diagnostics.ownedWorkspaceCount() == 1);
  CHECK(value_diagnostics.value_bytes > 0);
  CHECK(value_diagnostics.accountedBytes() == value_diagnostics.value_bytes);

  full_electrons.G = ValueType(0);
  full_electrons.L = ValueType(0);
  full_component.evaluateLog(full_electrons, full_electrons.G, full_electrons.L);
  const auto full_diagnostics =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(full_component);
  CHECK(full_diagnostics.owns_full_spatial_workspace);
  CHECK(full_diagnostics.ownedWorkspaceCount() == 1);
  CHECK(full_diagnostics.full_spatial_bytes > 0);
  CHECK(full_diagnostics.accountedBytes() == full_diagnostics.full_spatial_bytes);

  active_component.evalGrad(active_electrons, 0);
  const auto active_diagnostics =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(active_component);
  CHECK(active_diagnostics.owns_active_spatial_workspace);
  CHECK(active_diagnostics.ownedWorkspaceCount() == 1);
  CHECK(active_diagnostics.active_spatial_bytes > 0);
  CHECK(active_diagnostics.accountedBytes() == active_diagnostics.active_spatial_bytes);

  batch_electrons.makeVirtualMoves(ParticleSet::SingleParticlePos{0.37, -0.22, 0.41});
  std::vector<ValueType> ratios(batch_electrons.getTotalNum());
  batch_component.evaluateRatiosAlltoOne(batch_electrons, ratios);
  const auto batch_diagnostics =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(batch_component);
  CHECK(batch_diagnostics.owns_batch_workspace);
  CHECK(batch_diagnostics.ownedWorkspaceCount() == 1);
  CHECK(batch_diagnostics.batch_bytes > 0);
  CHECK(batch_diagnostics.accountedBytes() == batch_diagnostics.batch_bytes);

  // An entirely untouched clone remains free of evaluator scratch after other
  // clones sharing the same immutable model exercise every scalar inference mode.
  check_empty(untouched_component);

  ParticleSet crowd_electrons0 = makeLiHElectrons(simulation_cell);
  ParticleSet crowd_electrons1 = makeLiHElectrons(simulation_cell);
  auto crowd_storage0 = make_lazy_clone(crowd_electrons0);
  auto crowd_storage1 = make_lazy_clone(crowd_electrons1);
  PsiFormerWF& crowd_component0 = as_psiformer(crowd_storage0);
  PsiFormerWF& crowd_component1 = as_psiformer(crowd_storage1);
  RefVectorWithLeader<WaveFunctionComponent> components(
      crowd_component0, {crowd_component0, crowd_component1});
  RefVectorWithLeader<ParticleSet> particles(
      crowd_electrons0, {crowd_electrons0, crowd_electrons1});
  std::array<ParticleSet::ParticleGradient, 2> gradients{
      ParticleSet::ParticleGradient(crowd_electrons0.getTotalNum()),
      ParticleSet::ParticleGradient(crowd_electrons1.getTotalNum())};
  std::array<ParticleSet::ParticleLaplacian, 2> laplacians{
      ParticleSet::ParticleLaplacian(crowd_electrons0.getTotalNum()),
      ParticleSet::ParticleLaplacian(crowd_electrons1.getTotalNum())};
  RefVector<ParticleSet::ParticleGradient> gradient_list{gradients[0], gradients[1]};
  RefVector<ParticleSet::ParticleLaplacian> laplacian_list{laplacians[0], laplacians[1]};

  ResourceCollection resource_template("psiformer_lazy_workspace_template");
  crowd_component0.createResource(resource_template);
  ResourceCollection crowd_resource(resource_template);
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource, components);
    crowd_component0.mw_evaluateLog(
        components, particles, gradient_list, laplacian_list);
    check_empty(crowd_component0);
    check_empty(crowd_component1);
  }
}

TEST_CASE("PsiFormer exposes fail-closed batch planning hooks",
          "[wavefunction][psiformer][batch_memory]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF component("pf_batch_policy", files.parameters.string(),
                        files.configuration.string(), true, {0, 1});

  BatchExecutionRequirements requirements;
  component.contributeBatchExecutionRequirements(requirements);
  CHECK(requirements.requires(BatchExecutionMode::FULL_VGL));
  CHECK_FALSE(requirements.requires(BatchExecutionMode::VALUE));
  CHECK_FALSE(requirements.requires(BatchExecutionMode::ACTIVE_GRADIENT));
  CHECK_FALSE(requirements.requires(BatchExecutionMode::SCORE));
  CHECK_FALSE(requirements.requires(BatchExecutionMode::KINETIC));

  BatchExecutionRequirements broad_requirements = requirements;
  broad_requirements.require(BatchExecutionMode::VALUE);
  broad_requirements.require(BatchExecutionMode::ACTIVE_GRADIENT);
  broad_requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  BatchExecutionTopology topology;
  topology.initial_walkers_per_crowd = {1, 2};
  topology.reserve_walkers_per_crowd = {3, 2};
  topology.run_kind                  = "psiformer-hook-test";

  const BatchExecutionWorkloadContext workload{
      broad_requirements, topology, 2};
  const BatchTileCapacities logical_maximum =
      component.batchExecutionLogicalMaximum(workload);
  CHECK(logical_maximum == BatchTileCapacities{5, 3, 3, 0});

  const BatchExecutionPlanningContext context{
      broad_requirements, topology, logical_maximum,
      BatchTileCapacities{2, 2, 1, 0}, 2};
  const BatchMemoryContribution contribution =
      component.estimateBatchExecutionMemory(context);
  CHECK(contribution.logical_maximum == logical_maximum);
  CHECK(contribution.owner_multiplicity == 1);
  CHECK_FALSE(contribution.fully_accounted);
  CHECK(contribution.per_owner.total().host > 0);
  CHECK(contribution.per_owner.total().device == 0);

  // A real plan cannot be selected from partial evidence.  Later preparation
  // stages will enable accounting claims only as the corresponding ownership
  // and runtime guards become complete.
  BatchExecutionSelectionInput selection;
  selection.requirements                       = requirements;
  selection.topology                           = topology;
  selection.active_parameter_count             = 2;
  const BatchExecutionWorkloadContext selection_workload{
      requirements, topology, 2};
  selection.logical_maximum =
      component.batchExecutionLogicalMaximum(selection_workload);
  CHECK_THROWS_WITH(
      selectBatchExecutionPlan(
          selection,
          [&component](const BatchExecutionPlanningContext& candidate) {
            return std::vector<BatchMemoryParticipantContribution>{
                {"twf/component/0/PsiFormerWF/pf_batch_policy",
                 component.estimateBatchExecutionMemory(candidate)}};
          }),
      Catch::Matchers::ContainsSubstring("not fully accounted"));

  // Even internally consistent evidence cannot authorize a plan that omitted
  // the component-owned FULL_VGL initialization requirement.
  BatchExecutionSelectionInput missing_requirement_selection = selection;
  missing_requirement_selection.requirements = {};
  const BatchExecutionWorkloadContext missing_workload{
      {}, topology, 2};
  missing_requirement_selection.logical_maximum =
      component.batchExecutionLogicalMaximum(missing_workload);
  const std::string participant_id =
      "twf/component/0/PsiFormerWF/pf_batch_policy";
  auto missing_plan = std::make_shared<const BatchExecutionPlan>(
      selectBatchExecutionPlan(
          missing_requirement_selection,
          [&component, &participant_id](
              const BatchExecutionPlanningContext& candidate) {
            BatchMemoryContribution fabricated =
                component.estimateBatchExecutionMemory(candidate);
            fabricated.fully_accounted = true;
            return std::vector<BatchMemoryParticipantContribution>{
                {participant_id, std::move(fabricated)}};
          }));
  const BatchExecutionParticipantPlan missing_view =
      makeBatchExecutionParticipantPlan(missing_plan, participant_id);
  CHECK_THROWS_WITH(
      component.validateBatchExecutionPlanBinding(missing_view),
      Catch::Matchers::ContainsSubstring("component-owned requirement"));

  // The explicit empty view is always safe while idle and is copied as empty
  // state rather than causing any evaluator scratch to be materialized.
  BatchExecutionParticipantPlan empty_plan;
  component.validateBatchExecutionPlanBinding(empty_plan);
  component.bindBatchExecutionPlan(empty_plan);
  CHECK_FALSE(testing::TestPsiFormerWF::hasBatchExecutionPlan(component));
  std::unique_ptr<WaveFunctionComponent> clone_storage =
      component.makeClone(electrons);
  auto* clone = dynamic_cast<PsiFormerWF*>(clone_storage.get());
  REQUIRE(clone != nullptr);
  CHECK_FALSE(testing::TestPsiFormerWF::hasBatchExecutionPlan(*clone));
}

TEST_CASE("PsiFormer kinetic parameter derivatives require unit electron masses",
          "[wavefunction][psiformer][optimizer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF component(
      "pf_unit_mass", files.parameters.string(), files.configuration.string(), true, {0});
  OptVariables active = registerSelectedParameters(component);

  SpeciesSet& species = electrons.getSpeciesSet();
  const int mass       = species.getAttribute("mass");
  REQUIRE(mass < species.numAttributes());
  SECTION("equal nonunit masses")
  {
    species(mass, 0) = 2.0;
    species(mass, 1) = 2.0;
  }
  SECTION("unequal masses")
  {
    species(mass, 0) = 1.0;
    species(mass, 1) = 2.0;
  }
  electrons.resetGroups();

  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  component.evaluateLog(electrons, electrons.G, electrons.L);
  Vector<ValueType> score(active.size());
  Vector<ValueType> kinetic_response(active.size());
  score            = ValueType(0);
  kinetic_response = ValueType(0);

  // A score-only reverse does not use the electron masses and remains valid.
  CHECK_NOTHROW(component.evaluateDerivativesWF(electrons, active, score));
  CHECK_THROWS_WITH(component.evaluateDerivatives(electrons, active, score, kinetic_response),
                    Catch::Matchers::ContainsSubstring("require unit electron masses"));
}

TEST_CASE("PsiFormer nonlocal virtual ratios and parameter derivatives", "[wavefunction][psiformer][ecp]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF component("pf_virtual", files.parameters.string(), files.configuration.string(), true, {0, 127});
  OptVariables active = registerSelectedParameters(component);

  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  const PsiFormerWF::LogValue reference_log = component.evaluateLog(electrons, electrons.G, electrons.L);
  const std::vector<ParticleSet::SingleParticlePos> displacements{{0.08, -0.03, 0.02},
                                                                   {-0.04, 0.06, -0.05},
                                                                   {0.03, 0.01, 0.07}};
  VirtualParticleSet virtual_particles(electrons);
  virtual_particles.makeMoves(electrons, 1, displacements);

  std::vector<ValueType> ratios(displacements.size());
  Matrix<ValueType> derivative_ratios(displacements.size(), active.size());
  derivative_ratios = ValueType(0);
  component.evaluateDerivRatios(virtual_particles, active, ratios, derivative_ratios);

  // Full reevaluation supplies an independent high-level ratio check at each
  // quadrature point; no particle-by-particle proposal cache is involved.
  for (std::size_t move = 0; move < displacements.size(); ++move)
  {
    ParticleSet moved = makeLiHElectrons(simulation_cell);
    moved.R[1] += displacements[move];
    moved.update();
    PsiFormerWF fixed("pf_virtual_fixed", files.parameters.string(), files.configuration.string());
    moved.G = ValueType(0);
    moved.L = ValueType(0);
    const PsiFormerWF::LogValue moved_log = fixed.evaluateLog(moved, moved.G, moved.L);
    const auto expected_ratio             = std::exp(moved_log - reference_log);
    CHECK(std::real(ratios[move]) ==
          Catch::Approx(std::real(expected_ratio)).epsilon(2e-9).margin(2e-12));
    CHECK(std::imag(ratios[move]) ==
          Catch::Approx(std::imag(expected_ratio)).epsilon(2e-9).margin(2e-12));
  }

  // The ECP contract is d log(Psi_virtual/Psi_reference)/d theta. Verify
  // both selected columns by centered finite differences of full log values.
  const double parameter_step = 2e-5;
  for (int parameter = 0; parameter < active.size(); ++parameter)
  {
    const double original = active[parameter];
    std::vector<double> log_ratio_plus(displacements.size());
    std::vector<double> log_ratio_minus(displacements.size());
    for (int direction : {-1, 1})
    {
      active[parameter] = original + direction * parameter_step;
      component.resetParametersExclusive(active);
      electrons.G = ValueType(0);
      electrons.L = ValueType(0);
      const double base_log = std::real(component.evaluateLog(electrons, electrons.G, electrons.L));
      for (std::size_t move = 0; move < displacements.size(); ++move)
      {
        ParticleSet moved = makeLiHElectrons(simulation_cell);
        moved.R[1] += displacements[move];
        moved.update();
        moved.G = ValueType(0);
        moved.L = ValueType(0);
        const double moved_log = std::real(component.evaluateLog(moved, moved.G, moved.L));
        (direction > 0 ? log_ratio_plus : log_ratio_minus)[move] = moved_log - base_log;
      }
    }
    active[parameter] = original;
    component.resetParametersExclusive(active);
    for (std::size_t move = 0; move < displacements.size(); ++move)
    {
      const double finite_difference = (log_ratio_plus[move] - log_ratio_minus[move]) / (2 * parameter_step);
      CHECK(std::real(derivative_ratios(move, parameter)) ==
            Catch::Approx(finite_difference).epsilon(8e-5).margin(8e-5));
    }
  }
}

TEST_CASE("PsiFormer full-network derivatives update and restart", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet full_electrons     = makeLiHElectrons(simulation_cell);
  ParticleSet selected_electrons = makeLiHElectrons(simulation_cell);

  PsiFormerWF full("pf_full", files.parameters.string(), files.configuration.string(), true, {}, true);
  OptVariables full_active = registerSelectedParameters(full);
  REQUIRE(full_active.size() > 2048);
  const std::vector<std::size_t> probes{0, 127, 2047, full_active.size() - 1};

  full_electrons.G = ValueType(0);
  full_electrons.L = ValueType(0);
  const double baseline_log = std::real(full.evaluateLog(full_electrons, full_electrons.G, full_electrons.L));
  Vector<ValueType> full_dlog(full_active.size());
  Vector<ValueType> full_denergy(full_active.size());
  full_dlog    = ValueType(0);
  full_denergy = ValueType(0);
  full.evaluateDerivatives(full_electrons, full_active, full_dlog, full_denergy);

  PsiFormerWF selected(
      "pf_selected_full_check", files.parameters.string(), files.configuration.string(), true, probes);
  OptVariables selected_active = registerSelectedParameters(selected);
  const ComponentSnapshot selected_snapshot = evaluateComponent(selected, selected_electrons, selected_active);
  for (std::size_t probe = 0; probe < probes.size(); ++probe)
  {
    CHECK(std::real(full_dlog[probes[probe]]) ==
          Catch::Approx(selected_snapshot.log_parameter_derivative[probe]).epsilon(2e-10).margin(2e-10));
    CHECK(std::real(full_denergy[probes[probe]]) ==
          Catch::Approx(selected_snapshot.kinetic_parameter_derivative[probe]).epsilon(2e-9).margin(2e-9));
  }

  // Exercise the full-vector reset while perturbing only two entries. The
  // optimizer still supplies the complete active vector on every update.
  full_active[0] -= 1e-5 * std::real(full_dlog[0]);
  full_active[127] -= 1e-5 * std::real(full_dlog[127]);
  full.resetParametersExclusive(full_active);
  full_electrons.G = ValueType(0);
  full_electrons.L = ValueType(0);
  const double updated_log = std::real(full.evaluateLog(full_electrons, full_electrons.G, full_electrons.L));
  CHECK(std::abs(updated_log - baseline_log) > 1e-10);

  const std::filesystem::path state_path = files.directory / "psiformer_full_restart.vp.h5";
  hdf_archive output;
  REQUIRE(output.create(state_path));
  full.writeVariationalParameters(output);
  output.close();

  ParticleSet restored_electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF restored("pf_full", files.parameters.string(), files.configuration.string(), true, {}, true);
  hdf_archive input;
  REQUIRE(input.open(state_path, H5F_ACC_RDONLY));
  restored.readVariationalParameters(input);
  input.close();
  OptVariables restored_active = registerSelectedParameters(restored);
  restored.resetParametersExclusive(restored_active);
  restored_electrons.G = ValueType(0);
  restored_electrons.L = ValueType(0);
  const double restored_log =
      std::real(restored.evaluateLog(restored_electrons, restored_electrons.G, restored_electrons.L));
  CHECK(restored_log == Catch::Approx(updated_log).epsilon(2e-10).margin(2e-10));
}

TEST_CASE("PsiFormer LiH pair full-network update", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih_pair");
  const Geometry geometry = makeGeometry("lih_pair");
  const SimulationCell simulation_cell;
  ParticleSet electrons(simulation_cell);
  electrons.setName("e");
  electrons.create({4, 4});
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      electrons.R[electron][dimension] = geometry.electrons[3 * electron + dimension];
  electrons.update();

  PsiFormerWF full("pf_pair_full", files.parameters.string(), files.configuration.string(), true, {}, true);
  OptVariables active = registerSelectedParameters(full);
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  const double initial_log = std::real(full.evaluateLog(electrons, electrons.G, electrons.L));
  active[0] += 1e-4;
  full.resetParametersExclusive(active);
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  const double updated_log = std::real(full.evaluateLog(electrons, electrons.G, electrons.L));
  CHECK(std::abs(updated_log - initial_log) > 1e-10);
}

TEST_CASE("PsiFormer selected parameters follow QMCPACK registration reset and derivative paths",
          "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  RuntimeOptions runtime_options;
  TrialWaveFunction trial_wavefunction(runtime_options, "psiformer_selected_test");
  trial_wavefunction.addComponent(std::make_unique<PsiFormerWF>(
      "pf", files.parameters.string(), files.configuration.string(), true, std::vector<std::size_t>{127, 0, 1}));

  OptVariables active;
  trial_wavefunction.checkInVariables(active);
  active.resetIndex();
  trial_wavefunction.checkOutVariables(active);
  REQUIRE(active.size() == 3);
  CHECK(active.name(0) == "pf_pf_0000000");
  CHECK(active.name(1) == "pf_pf_0000001");
  CHECK(active.name(2) == "pf_pf_0000127");
  REQUIRE(trial_wavefunction.extractOptimizableObjectRefs().size() == 1);

  const double baseline_log = trial_wavefunction.evaluateLog(electrons);
  addLinearLogGradient(electrons);

  Vector<ValueType> dlogpsi(active.size());
  Vector<ValueType> dhpsioverpsi(active.size());
  Vector<ValueType> dlogpsi_wf(active.size());
  dlogpsi       = ValueType(-0.125);
  dhpsioverpsi  = ValueType(0.625);
  dlogpsi_wf    = ValueType(0.375);
  trial_wavefunction.evaluateDerivatives(electrons, active, dlogpsi, dhpsioverpsi);
  trial_wavefunction.evaluateDerivativesWF(electrons, active, dlogpsi_wf);

  const double log_derivative = std::real(dlogpsi[0]) + 0.125;
  const double kinetic_derivative = std::real(dhpsioverpsi[0]) - 0.625;
  CHECK(std::real(dlogpsi_wf[0]) - 0.375 == Catch::Approx(log_derivative).epsilon(2e-10).margin(2e-10));

  const double original_value = active[0];
  const double parameter_step = 2e-5;
  active[0] = original_value + parameter_step;
  trial_wavefunction.resetParameters(active);
  const double plus_log = trial_wavefunction.evaluateLog(electrons);
  addLinearLogGradient(electrons);
  const double plus_kinetic = kineticEnergy(electrons);

  active[0] = original_value - parameter_step;
  trial_wavefunction.resetParameters(active);
  const double minus_log = trial_wavefunction.evaluateLog(electrons);
  addLinearLogGradient(electrons);
  const double minus_kinetic = kineticEnergy(electrons);

  active[0] = original_value;
  trial_wavefunction.resetParameters(active);
  const double restored_log = trial_wavefunction.evaluateLog(electrons);
  CHECK(restored_log == Catch::Approx(baseline_log).epsilon(2e-10).margin(2e-10));

  const double log_finite_difference = (plus_log - minus_log) / (2 * parameter_step);
  const double kinetic_finite_difference = (plus_kinetic - minus_kinetic) / (2 * parameter_step);
  CHECK(log_finite_difference == Catch::Approx(log_derivative).epsilon(5e-5).margin(5e-5));
  CHECK(kinetic_finite_difference == Catch::Approx(kinetic_derivative).epsilon(2e-4).margin(2e-4));

  // Apply one deterministic first-order update through the public reset path.
  // This is the small vertical slice used before streaming full-network descent.
  active[0] = original_value - 1e-4 * log_derivative;
  trial_wavefunction.resetParameters(active);
  const double updated_log = trial_wavefunction.evaluateLog(electrons);
  CHECK(std::abs(updated_log - baseline_log) > 1e-8);
}

TEST_CASE("PsiFormer component-major kinetic derivatives reuse one crowd tape",
          "[wavefunction][psiformer][multiwalker]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;

  ParticleSet batch_electrons0 = makeLiHElectrons(simulation_cell);
  ParticleSet batch_electrons1 = makeLiHElectrons(simulation_cell);
  batch_electrons1.R[0][0] += 0.11;
  batch_electrons1.update();

  PsiFormerWF leader(
      "pf_kinetic_pool", files.parameters.string(), files.configuration.string(), true, {0, 127});
  OptVariables active = registerSelectedParameters(leader);
  std::unique_ptr<WaveFunctionComponent> clone_storage = leader.makeClone(batch_electrons1);
  auto* clone = dynamic_cast<PsiFormerWF*>(clone_storage.get());
  REQUIRE(clone != nullptr);
  clone->checkOutVariables(active);

  // Build the complete TrialWaveFunction drift independently for each walker.
  // The added factors emulate distinct surrounding wavefunction components.
  auto initialize_total_drift = [](PsiFormerWF& component, ParticleSet& electrons, double scale) {
    electrons.G = ValueType(0);
    electrons.L = ValueType(0);
    component.evaluateLog(electrons, electrons.G, electrons.L);
    for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
      for (int dimension = 0; dimension < 3; ++dimension)
        electrons.G[electron][dimension] +=
            ValueType(scale * (1 + 3 * electron + dimension));
  };
  initialize_total_drift(leader, batch_electrons0, 0.007);
  initialize_total_drift(*clone, batch_electrons1, -0.011);

  RefVectorWithLeader<WaveFunctionComponent> components(leader, {leader, *clone});
  RefVectorWithLeader<ParticleSet> particles(
      batch_electrons0, {batch_electrons0, batch_electrons1});
  RecordArray<ValueType> batch_scores(2, active.size());
  RecordArray<ValueType> batch_kinetic(2, active.size());
  std::fill(batch_scores.begin(), batch_scores.end(), ValueType(0.25));
  std::fill(batch_kinetic.begin(), batch_kinetic.end(), ValueType(-0.5));

  ResourceCollection resource_template("psiformer_kinetic_pool_template");
  leader.createResource(resource_template);
  ResourceCollection crowd_resource(resource_template);
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource, components);
    const std::array<std::size_t, 2> no_kinetic_tapes{0, 0};
    CHECK(testing::TestPsiFormerWF::directKineticWorkspaceOwnership(leader, components) == no_kinetic_tapes);

    // Reject a heterogeneous-mass crowd before allocating the shared tape.
    SpeciesSet& second_species = batch_electrons1.getSpeciesSet();
    const int second_mass      = second_species.getAttribute("mass");
    REQUIRE(second_mass < second_species.numAttributes());
    second_species(second_mass, 1) = 2.0;
    batch_electrons1.resetGroups();
    CHECK_THROWS_WITH(
        leader.mw_evaluateParameterDerivatives(
            components, particles, active, batch_scores, batch_kinetic),
        Catch::Matchers::ContainsSubstring("require unit electron masses"));
    CHECK(testing::TestPsiFormerWF::directKineticWorkspaceOwnership(leader, components) == no_kinetic_tapes);
    second_species(second_mass, 1) = 1.0;
    batch_electrons1.resetGroups();

    leader.mw_evaluateParameterDerivatives(
        components, particles, active, batch_scores, batch_kinetic);

    // Neither component clone owns a kinetic tape; exactly one tape belongs to
    // the acquired crowd resource after the first component-major call.
    const std::array<std::size_t, 2> one_crowd_kinetic_tape{0, 1};
    CHECK(testing::TestPsiFormerWF::directKineticWorkspaceOwnership(leader, components) ==
          one_crowd_kinetic_tape);
  }

  // Independent scalar calls provide the numerical oracle and, because their
  // external drifts differ, catch failure to repack ParticleSet::G per walker.
  ParticleSet scalar_electrons0 = makeLiHElectrons(simulation_cell);
  ParticleSet scalar_electrons1 = makeLiHElectrons(simulation_cell);
  scalar_electrons1.R[0][0] += 0.11;
  scalar_electrons1.update();
  PsiFormerWF scalar0(
      "pf_kinetic_scalar0", files.parameters.string(), files.configuration.string(), true, {0, 127});
  PsiFormerWF scalar1(
      "pf_kinetic_scalar1", files.parameters.string(), files.configuration.string(), true, {0, 127});
  OptVariables scalar_active0 = registerSelectedParameters(scalar0);
  OptVariables scalar_active1 = registerSelectedParameters(scalar1);
  initialize_total_drift(scalar0, scalar_electrons0, 0.007);
  initialize_total_drift(scalar1, scalar_electrons1, -0.011);

  std::array<Vector<ValueType>, 2> scalar_scores{
      Vector<ValueType>(active.size()), Vector<ValueType>(active.size())};
  std::array<Vector<ValueType>, 2> scalar_kinetic{
      Vector<ValueType>(active.size()), Vector<ValueType>(active.size())};
  for (int walker = 0; walker < 2; ++walker)
  {
    scalar_scores[walker]  = ValueType(0.25);
    scalar_kinetic[walker] = ValueType(-0.5);
  }
  scalar0.evaluateDerivatives(
      scalar_electrons0, scalar_active0, scalar_scores[0], scalar_kinetic[0]);
  scalar1.evaluateDerivatives(
      scalar_electrons1, scalar_active1, scalar_scores[1], scalar_kinetic[1]);

  for (int walker = 0; walker < 2; ++walker)
    for (std::size_t parameter = 0; parameter < active.size(); ++parameter)
    {
      CHECK(std::abs(batch_scores[walker][parameter] - scalar_scores[walker][parameter]) <
            2e-10 * (1 + std::abs(scalar_scores[walker][parameter])));
      CHECK(std::abs(batch_kinetic[walker][parameter] - scalar_kinetic[walker][parameter]) <
            2e-9 * (1 + std::abs(scalar_kinetic[walker][parameter])));
    }
  CHECK(std::real(batch_kinetic[0][0]) != Approx(std::real(batch_kinetic[1][0])));
}

TEST_CASE("PsiFormer registration maps through surrounding ordinary parameters",
          "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF component(
      "pf_block", files.parameters.string(), files.configuration.string(), true, {0, 1, 127});

  OptVariables active;
  active.insert("ordinary_before", -1.0, true, optimize::LINEAR_P);
  component.checkInVariablesExclusive(active);
  active.insert("ordinary_after", 1.0, true, optimize::LOGLINEAR_P);
  active.resetIndex();
  component.checkOutVariables(active);

  REQUIRE(active.size() == 5);
  CHECK(active.name(0) == "ordinary_before");
  CHECK(active.name(1) == "pf_block_pf_0000000");
  CHECK(active.name(2) == "pf_block_pf_0000001");
  CHECK(active.name(3) == "pf_block_pf_0000127");
  CHECK(active.name(4) == "ordinary_after");

  // PsiFormer must scatter its results only into mapped global entries,
  // leaving derivative contributions owned by neighboring objects untouched.
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  component.evaluateLog(electrons, electrons.G, electrons.L);
  Vector<ValueType> dlogpsi(active.size());
  Vector<ValueType> dhpsioverpsi(active.size());
  dlogpsi      = ValueType(-91.0);
  dhpsioverpsi = ValueType(37.0);
  component.evaluateDerivatives(electrons, active, dlogpsi, dhpsioverpsi);

  CHECK(std::real(dlogpsi[0]) == Approx(-91.0));
  CHECK(std::real(dhpsioverpsi[0]) == Approx(37.0));
  CHECK(std::real(dlogpsi[4]) == Approx(-91.0));
  CHECK(std::real(dhpsioverpsi[4]) == Approx(37.0));
  for (int global_index = 1; global_index <= 3; ++global_index)
  {
    CHECK(psiformer::determinant::isFiniteReal(std::real(dlogpsi[global_index])));
    CHECK(psiformer::determinant::isFiniteReal(std::real(dhpsioverpsi[global_index])));
    CHECK(std::real(dlogpsi[global_index]) != Approx(-91.0));
    CHECK(std::real(dhpsioverpsi[global_index]) != Approx(37.0));
  }
}

TEST_CASE("PsiFormer clones share versioned parameters and retain local move state", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet leader_electrons = makeLiHElectrons(simulation_cell);
  ParticleSet clone_electrons  = makeLiHElectrons(simulation_cell);

  PsiFormerWF leader(
      "pf_clone", files.parameters.string(), files.configuration.string(), true, std::vector<std::size_t>{0, 127});
  OptVariables active = registerSelectedParameters(leader);
  std::unique_ptr<WaveFunctionComponent> clone_base = leader.makeClone(clone_electrons);
  auto* clone = dynamic_cast<PsiFormerWF*>(clone_base.get());
  REQUIRE(clone != nullptr);

  const ComponentSnapshot initial_leader = evaluateComponent(leader, leader_electrons, active);
  const ComponentSnapshot initial_clone  = evaluateComponent(*clone, clone_electrons, active);
  checkComponentSnapshot(initial_clone, initial_leader);

  const std::size_t initial_version = leader.parameterVersion();
  active[0] += 2e-4;
  leader.resetParametersExclusive(active);
  CHECK(leader.parameterVersion() == initial_version + 1);
  CHECK(clone->parameterVersion() == initial_version + 1);
  OptVariables clone_active = registerSelectedParameters(*clone);
  CHECK(std::real(clone_active[0]) == Catch::Approx(std::real(active[0])));

  const ComponentSnapshot updated_leader = evaluateComponent(leader, leader_electrons, active);
  const ComponentSnapshot updated_clone  = evaluateComponent(*clone, clone_electrons, active);
  checkComponentSnapshot(updated_clone, updated_leader);
  CHECK(std::abs(updated_leader.log_value - initial_leader.log_value) > 1e-8);

  // Replaying the same global reset through a clone must not create another
  // model version or rewrite the shared parameter leaves.
  clone->resetParametersExclusive(active);
  CHECK(leader.parameterVersion() == initial_version + 1);

  // Cache a proposal in the clone, update through the leader, and verify that
  // accepting the now-stale proposal cannot promote its old log value.
  clone_electrons.makeMove(0, ParticleSet::SingleParticlePos{0.01, -0.02, 0.015});
  clone->ratio(clone_electrons, 0);
  active[0] += 1e-4;
  leader.resetParametersExclusive(active);
  CHECK(leader.parameterVersion() == initial_version + 2);
  clone->acceptMove(clone_electrons, 0);
  CHECK(std::real(clone->get_log_value()) == 0.0);
  clone_electrons.rejectMove(0);

  // Two clone-local evaluations may read the same immutable parameter version
  // concurrently. The shared lock excludes optimizer resets during each
  // native graph traversal.
  std::promise<void> start_promise;
  const std::shared_future<void> start = start_promise.get_future().share();
  auto evaluate_log = [](PsiFormerWF& component, ParticleSet& electrons, std::shared_future<void> gate) {
    gate.wait();
    electrons.G = ValueType(0);
    electrons.L = ValueType(0);
    return std::real(component.evaluateLog(electrons, electrons.G, electrons.L));
  };
  auto leader_future =
      std::async(std::launch::async, evaluate_log, std::ref(leader), std::ref(leader_electrons), start);
  auto clone_future =
      std::async(std::launch::async, evaluate_log, std::ref(*clone), std::ref(clone_electrons), start);
  start_promise.set_value();

  const double concurrent_leader_log = leader_future.get();
  const double concurrent_clone_log  = clone_future.get();
  CHECK(concurrent_clone_log ==
        Catch::Approx(concurrent_leader_log).epsilon(2e-10).margin(2e-10));
}

TEST_CASE("PsiFormer complete model persistence and DeepQMC export round trip", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet original_electrons = makeLiHElectrons(simulation_cell);

  const std::vector<std::size_t> selected_indices{0, 127, 2047};
  const std::filesystem::path export_path = files.directory / "parameters_optimized.h5";
  PsiFormerWF original(
      "pf_restart", files.parameters.string(), files.configuration.string(), true, selected_indices, false,
      export_path.string());
  OptVariables original_active = registerSelectedParameters(original);
  original_active[0] += 1.5e-4;
  original_active[1] -= 2.0e-4;
  original_active[2] += 2.5e-4;
  original.resetParametersExclusive(original_active);
  const ComponentSnapshot expected = evaluateComponent(original, original_electrons, original_active);

  const std::filesystem::path vp_path = files.directory / "psiformer_restart.vp.h5";
  hdf_archive output;
  original_active.writeToHDF(vp_path.string(), output);
  original.writeVariationalParameters(output);
  output.close();
  CHECK(std::filesystem::exists(export_path));

  ParticleSet restored_electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF restored(
      "pf_restart", files.parameters.string(), files.configuration.string(), true, selected_indices);
  OptVariables restored_active = registerSelectedParameters(restored);
  hdf_archive input;
  restored_active.readFromHDF(vp_path.string(), input);
  restored.readVariationalParameters(input);
  input.close();
  restored.resetParametersExclusive(restored_active);

  const ComponentSnapshot restarted = evaluateComponent(restored, restored_electrons, restored_active);
  checkComponentSnapshot(restarted, expected);

  // The explicit export is intentionally separate from optimizer restart: it
  // contains only the DeepQMC flat values and immutable tensor layout.
  ParticleSet exported_electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF exported(
      "pf_export", export_path.string(), files.configuration.string(), true, selected_indices);
  OptVariables exported_active = registerSelectedParameters(exported);
  const ComponentSnapshot exported_snapshot = evaluateComponent(exported, exported_electrons, exported_active);
  checkComponentSnapshot(exported_snapshot, expected);

  // A restart payload cannot silently bind to a different internal selection.
  PsiFormerWF mismatched_selection(
      "pf_restart", files.parameters.string(), files.configuration.string(), true, {0, 128, 2047});
  hdf_archive mismatch_input;
  REQUIRE(mismatch_input.open(vp_path.string(), H5F_ACC_RDONLY));
  CHECK_THROWS_AS(mismatched_selection.readVariationalParameters(mismatch_input), std::runtime_error);
  mismatch_input.close();

  // The compact generic selected list is duplicated for compatibility. It
  // must agree with the authoritative complete model payload on restart.
  PsiFormerWF inconsistent_generic(
      "pf_restart", files.parameters.string(), files.configuration.string(), true, selected_indices);
  OptVariables inconsistent_active = registerSelectedParameters(inconsistent_generic);
  hdf_archive inconsistent_input;
  inconsistent_active.readFromHDF(vp_path.string(), inconsistent_input);
  inconsistent_generic.readVariationalParameters(inconsistent_input);
  inconsistent_input.close();
  inconsistent_active[0] += 1e-3;
  CHECK_THROWS_AS(inconsistent_generic.resetParametersExclusive(inconsistent_active), std::runtime_error);
}

} // namespace qmcplusplus
