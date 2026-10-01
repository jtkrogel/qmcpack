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
#include "Utilities/for_testing/Catch2Approx.h"

#include "Message/Communicate.h"
#include "OhmmsData/Libxml2Doc.h"
#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWaveFunctionBuilder.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "Utilities/RuntimeOptions.h"
#include "io/hdf/hdf_archive.h"
#include "psiformer_test_utils.h"

#include <cmath>
#include <complex>
#include <filesystem>
#include <functional>
#include <future>
#include <sstream>
#include <vector>

namespace qmcplusplus
{
namespace
{
using namespace testing::psiformer;
using ValueType = QMCTraits::ValueType;

/// Construct the four-electron ParticleSet matching the generated LiH fixture.
ParticleSet makeLiHElectrons(const SimulationCell& simulation_cell)
{
  const Geometry geometry = makeGeometry("lih");
  ParticleSet electrons(simulation_cell);
  electrons.setName("e");
  electrons.create({2, 2});
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      electrons.R[electron][dimension] = geometry.electrons[3 * electron + dimension];
  electrons.update();
  return electrons;
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
                << "\" optimize=\"yes\" optimize_scope=\"indices\" optimize_indices=\"127, 0 1\"/>";
  Libxml2Document optimized_document;
  REQUIRE(optimized_document.parseFromString(optimized_xml.str()));
  std::unique_ptr<WaveFunctionComponent> optimized = builder.buildComponent(optimized_document.getRoot());
  REQUIRE(optimized != nullptr);
  CHECK(optimized->isOptimizable());
  UniqueOptObjRefs optimized_refs;
  optimized->extractOptimizableObjectRefs(optimized_refs);
  REQUIRE(optimized_refs.size() == 1);

  std::ostringstream unsupported_xml;
  unsupported_xml << "<psiformer parameters=\"" << files.parameters.string() << "\" configuration=\""
                  << files.configuration.string()
                  << "\" optimize=\"yes\" optimize_scope=\"all\" optimize_indices=\"0\"/>";
  Libxml2Document unsupported_document;
  REQUIRE(unsupported_document.parseFromString(unsupported_xml.str()));
  CHECK_THROWS_AS(builder.buildComponent(unsupported_document.getRoot()), std::invalid_argument);
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
    CHECK(std::isfinite(std::real(dlogpsi[global_index])));
    CHECK(std::isfinite(std::real(dhpsioverpsi[global_index])));
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
  PsiFormerWF original(
      "pf_restart", files.parameters.string(), files.configuration.string(), true, selected_indices);
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
  const std::filesystem::path export_path = files.directory / "parameters_optimized.h5";
  restored.exportParameters(export_path.string());
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
