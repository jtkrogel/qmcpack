//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "QMCDrivers/tests/QMCDriverNewTestWrapper.h"
#include "QMCDrivers/Crowd.h"
#include "QMCDrivers/VMC/VMCBatchedSampling.h"
#include "QMCDrivers/VMC/VMCDriverInput.h"
#include "Concurrency/Info.hpp"
#include "Concurrency/UtilityFunctions.hpp"
#include "EstimatorInputDelegates.h"
#include "SetupPools.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"
#include "QMCDrivers/WFTrain/OrbitalInferenceMoments.h"
#include "QMCDrivers/WFTrain/WFTrainInference.h"
#include "QMCDrivers/WFTrain/FirstOrderOptimizer.h"
#include "QMCDrivers/WFTrain/WFTrainDriverInput.h"
#include "io/hdf/hdf_archive.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOrbitalTargets.h"
#include "QMCWaveFunctions/Fermion/SlaterDet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCHamiltonians/OperatorBase.h"
#include <MinimalParticlePool.h>
#include <MinimalWaveFunctionPool.h>
#include <MinimalHamiltonianPool.h>
#include <cmath>
#include <memory>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <set>

namespace qmcplusplus
{
namespace
{
/// Count physical and auxiliary evaluations through all population clones.
class CountingObservation final : public OperatorDependsOnlyOnParticleSet
{
public:
  explicit CountingObservation(std::shared_ptr<std::size_t> count) : count_(std::move(count))
  { setEnergyDomain(POTENTIAL); }

  std::string getClassName() const override { return "CountingObservation"; }

  Return_t evaluate(ParticleSet&) override { ++*count_; return 1.0; }

  bool put(xmlNodePtr) override { return true; }

  bool get(std::ostream&) const override { return true; }

  std::unique_ptr<OperatorBase> makeClone(ParticleSet&) const override
  { return std::make_unique<CountingObservation>(count_); }

private:
  std::shared_ptr<std::size_t> count_;
};

/// Exercise the real initialization and warmup/production sampling policy seams.
class OrbitalObservationTestDriver final : public testing::QMCDriverNewTestWrapper
{
public:
  using QMCDriverNewTestWrapper::QMCDriverNewTestWrapper;

  void exercise(const VMCDriverInput& vmc)
  {
    process(nullptr);
    ContextForSteps context(rngs_[0]);
    RefVector<ContextForSteps> contexts{context};
    initialLogEvaluationWavefunctionOnly(0, crowds_, contexts, false);
    VMCBatchedSampling::State state(qmcdriver_input_, vmc, *drift_modifier_, population_, 1, false);
    state.evaluate_energy = false;
    VMCBatchedSampling::Timers timers{timers_.resource_timer, timers_.movepbyp_timer, timers_.buffer_timer,
        timers_.hamiltonian_timer, timers_.collectables_timer, timers_.estimators_timer,
        [](Crowd&, std::string_view) {}};
    VMCBatchedSampling::runWarmupStep(0, state, timers, rngs_, crowds_);
    state.step = state.global_step = 0;
    VMCBatchedSampling::runVMCStep(0, state, timers, rngs_, crowds_);
    CHECK(crowds_[0]->get_estimator_manager_crowd().get_block_weight() == 0.0);
    CHECK_FALSE(wftrain::isFiniteTrainingReal(
        crowds_[0]->get_walkers()[0].get().Properties(WalkerProperties::LOCALENERGY)));
  }

  void exerciseEnergy(const VMCDriverInput& vmc)
  {
    ContextForSteps context(rngs_[0]);
    RefVector<ContextForSteps> contexts{context};
    initialLogEvaluation(0, crowds_, contexts, false);
    VMCBatchedSampling::State state(qmcdriver_input_, vmc, *drift_modifier_, population_, 1, false);
    VMCBatchedSampling::Timers timers{timers_.resource_timer, timers_.movepbyp_timer, timers_.buffer_timer,
        timers_.hamiltonian_timer, timers_.collectables_timer, timers_.estimators_timer,
        [](Crowd&, std::string_view) {}};
    VMCBatchedSampling::runWarmupStep(0, state, timers, rngs_, crowds_);
    CHECK(std::isfinite(crowds_[0]->get_walkers()[0].get().Properties(WalkerProperties::LOCALENERGY)));
    state.accumulate_estimators = false;
    VMCBatchedSampling::runVMCStep(0, state, timers, rngs_, crowds_);
    CHECK(crowds_[0]->get_estimator_manager_crowd().get_block_weight() == 0.0);
  }
};
}

TEST_CASE("Objective-only sampling never evaluates physical or auxiliary Hamiltonians",
          "[wftrain][orbital_inference]")
{
  using namespace testing;
  Concurrency::OverrideMaxCapacity<> capacity(1);
  ProjectData project("orbital_observation_policy", ProjectData::DriverVersion::BATCH);
  auto* comm = OHMMS::Controller;
  Libxml2Document document;
  REQUIRE(document.parseFromString("<qmc><parameter name='walkers_per_rank'>1</parameter>"
      "<parameter name='crowds'>1</parameter><parameter name='blocks'>1</parameter>"
      "<parameter name='steps'>1</parameter><parameter name='substeps'>1</parameter>"
      "<parameter name='timestep'>0.01</parameter><parameter name='usedrift'>no</parameter></qmc>"));
  QMCDriverInput input;
  input.readXML(document.getRoot());
  VMCDriverInput vmc;
  vmc.readXML(document.getRoot());
  auto particles = MinimalParticlePool::make_diamondC_1x1x1(comm);
  auto wavefunctions = MinimalWaveFunctionPool::make_diamondC_1x1x1(project.getRuntimeOptions(), comm, particles);
  auto hamiltonians = MinimalHamiltonianPool::make_hamWithEE(comm, particles, wavefunctions);
  auto physical = std::make_shared<std::size_t>(0), auxiliary = std::make_shared<std::size_t>(0);
  auto& hamiltonian = hamiltonians.getHamiltonian().value().get();
  hamiltonian.addOperator(std::make_unique<CountingObservation>(physical), "physical_counter", true);
  hamiltonian.addOperator(std::make_unique<CountingObservation>(auxiliary), "auxiliary_counter", false);
  WalkerConfigurations walkers;
  RandomNumberGeneratorPool generators(1);
  OrbitalObservationTestDriver driver(project, std::move(input), walkers,
      MCPopulation(comm->size(), comm->rank(), *particles.getParticleSet("e"),
                    wavefunctions.getWaveFunction().value().get(), hamiltonian), generators.getRngRefs(), comm);
  driver.exercise(vmc);
  CHECK(*physical == 0);
  CHECK(*auxiliary == 0);
  driver.exerciseEnergy(vmc);
  CHECK(*physical > 0);
  CHECK(*auxiliary > 0);
}

TEST_CASE("Value-only orbital forward reproduces the native spin-normalized MSE",
          "[wftrain][orbital_inference]")
{
  using namespace psiformer;
  const SimulationCell cell;
  ParticleSet electrons(cell), ions(cell);
  electrons.setName("e");
  electrons.create({1, 2});
  auto& species = electrons.getSpeciesSet();
  species.addSpecies("u");
  species.addSpecies("d");
  const int mass = species.addAttribute("mass");
  species(mass, 0) = species(mass, 1) = 1.0;
  electrons.resetGroups();
  ions.setName("ion0");
  ions.create({1, 1});
  auto& nuclear_species = ions.getSpeciesSet();
  nuclear_species.addSpecies("Li");
  nuclear_species.addSpecies("H");
  const int charge = nuclear_species.addAttribute("charge");
  nuclear_species(charge, 0) = 3.0;
  nuclear_species(charge, 1) = 1.0;
  ions.resetGroups();
  ions.R[0] = {0.0, 0.0, 0.0};
  ions.R[1] = {3.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i) electrons.R[i] = {0.2 * (i + 1), 0.1, 0.3};
  const ModelShape shape{1, 2, 2, 16, 256, 4, 4};
  auto initialized = initializePsiFormerParameters(shape, 31);
  PsiFormerWF component("orbital_forward", std::move(initialized), electrons, ions);
  const auto parameters = component.snapshotParameters();
  std::vector<double> values;
  component.evaluateOrbitalValues(electrons, values, parameters.version);
  REQUIRE(values.size() == 16 * 3 * 3);
  auto target = values;
  for (double& value : target) value += 0.5;
  std::vector<double> gradient(parameters.values.size());
  const double loss = component.evaluateOrbitalMSE(electrons, target, gradient, parameters.version);
  CHECK(loss == Catch::Approx(0.5).epsilon(1e-12));
  std::vector<double> repeated;
  component.evaluateOrbitalValues(electrons, repeated, parameters.version);
  CHECK(repeated == values);
  CHECK(component.snapshotParameters().values == parameters.values);
  CHECK_THROWS(component.evaluateOrbitalValues(electrons, repeated, parameters.version + 1));
}

TEST_CASE("Orbital inference merges raw moments before empirical column normalization",
          "[wftrain][orbital_inference]")
{
  wftrain::OrbitalInferenceMoments first(1, 1, 1, true), second(1, 1, 1, true), together(1, 1, 1, true);
  const std::vector<double> target{2, 0, 0, 3}, a{-2, 4, 5, -3}, b{-4, 4, 5, -6};
  first.addSample(a, target, 0.3);
  second.addSample(b, target, 0.5);
  together.addSample(a, target, 0.3);
  together.addSample(b, target, 0.5);
  auto global = first.sums();
  for (std::size_t i = 0; i < global.size(); ++i) global[i] += second.sums()[i];
  CHECK(global == together.sums());
  const auto result = first.summarize(global);
  CHECK(result.samples == 2);
  CHECK(result.mean_mse == Catch::Approx(62.75));
  CHECK(result.mse_variance == Catch::Approx(264.0625));
  CHECK(result.occupied_column_cosine_squared == Catch::Approx(0.9));
  CHECK(result.undefined_columns == 0);
  CHECK(result.energy == Catch::Approx(0.4));
  CHECK(result.energy_squared == Catch::Approx(0.17));
  CHECK(first.summarize(first.sums()).occupied_column_cosine_squared == Catch::Approx(1.0));
  // Off-spin errors contribute to MSE but never to occupied-column cosine.
  CHECK(first.summarize(first.sums()).mean_mse == Catch::Approx(46.5));
  first.reset();
  CHECK_THROWS(first.summarize(first.sums()));
  CHECK_THROWS(first.addSample(a, target));
  CHECK_THROWS(first.addSample({1}, target, 0.3));
  wftrain::OrbitalInferenceMoments zero(1, 1, 1, false);
  zero.addSample(std::vector<double>(4, 0), std::vector<double>(4, 0));
  const auto undefined = zero.summarize(zero.sums());
  CHECK(undefined.undefined_columns == 2);
  CHECK_FALSE(wftrain::isFiniteTrainingReal(undefined.occupied_column_cosine_squared));
  CHECK(undefined.mean_mse == 0);
  CHECK_THROWS(wftrain::OrbitalInferenceMoments(0, 1, 1, false));
}

TEST_CASE("Orbital inference scalar observer controls every Hamiltonian evaluation",
          "[wftrain][orbital_inference]")
{
  using namespace testing;
  Concurrency::OverrideMaxCapacity<> capacity(1);
  auto* comm = OHMMS::Controller;
  ProjectData project("orbital_scalar_policy", ProjectData::DriverVersion::BATCH);
  auto particles = MinimalParticlePool::make_diamondC_1x1x1(comm);
  auto wavefunctions = MinimalWaveFunctionPool::make_diamondC_1x1x1(project.getRuntimeOptions(), comm, particles);
  auto hamiltonians = MinimalHamiltonianPool::make_hamWithEE(comm, particles, wavefunctions);
  auto physical = std::make_shared<std::size_t>(0), auxiliary = std::make_shared<std::size_t>(0);
  auto& hamiltonian = hamiltonians.getHamiltonian().value().get();
  hamiltonian.addOperator(std::make_unique<CountingObservation>(physical), "orbital_physical", true);
  hamiltonian.addOperator(std::make_unique<CountingObservation>(auxiliary), "orbital_auxiliary", false);
  for (bool energy : {false, true})
  {
    *physical = *auxiliary = 0;
    Libxml2Document document;
    REQUIRE(document.parseFromString("<qmc><inference><parameter name='walkers_per_rank'>1</parameter>"
      "<parameter name='crowds'>1</parameter><parameter name='blocks'>1</parameter>"
      "<parameter name='steps'>1</parameter><parameter name='warmupsteps'>1</parameter>"
      "<parameter name='substeps'>1</parameter><parameter name='timestep'>0.01</parameter>"
      "<parameter name='usedrift'>no</parameter></inference></qmc>"));
    WFTrainInferenceInput input;
    input.readXML(document.getRoot(), energy);
    WalkerConfigurations walkers;
    RandomNumberGeneratorPool rngs(1);
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() /
        ("orbital-scalar-unit-" + std::to_string(stamp) + "-" + std::to_string(comm->rank()));
    std::filesystem::create_directory(directory);
    const auto prefix = (directory / "observation").string();
    std::size_t samples = 0;
    WFTrainInference inference(project, input, walkers,
        MCPopulation(comm->size(), comm->rank(), *particles.getParticleSet("e"),
            wavefunctions.getWaveFunction().value(), hamiltonian), rngs.getRngRefs(), comm, prefix, 0, 0,
        [&](Crowd& crowd) { samples += crowd.size(); }, [&]() { return std::to_string(samples); }, "SampleCount");
    inference.process(nullptr);
    inference.run();
    CHECK(inference.complete());
    CHECK(samples == 1);
    CHECK((*physical > 0) == energy);
    CHECK((*auxiliary > 0) == energy);
    if (comm->rank() == 0)
    {
      std::ifstream scalar(prefix + ".scalar.dat");
      std::string header, row;
      REQUIRE(std::getline(scalar, header));
      REQUIRE(std::getline(scalar, row));
      CHECK(header == "# index SampleCount");
      CHECK(row == "0 1");
      CHECK(std::distance(std::filesystem::directory_iterator(directory), std::filesystem::directory_iterator{}) == 1);
      scalar.close();
      std::filesystem::remove(prefix + ".scalar.dat");
    }
    std::filesystem::remove(directory);
  }
}

TEST_CASE("Frozen Cartesian target reproduces occupied orbital values",
          "[.orbital_target_fixture]")
{
  const char* path = std::getenv("QMC_ORBITAL_TARGET_FIXTURE");
  REQUIRE(path != nullptr);
  const std::filesystem::path directory(path);
  auto* comm = OHMMS::Controller;
  ParticleSetPool particles(comm);
  Libxml2Document structure;
  REQUIRE(structure.parse((directory / "frozen_exact.structure.xml").string()));
  for (auto child = xmlFirstElementChild(structure.getRoot()); child; child = xmlNextElementSibling(child))
    REQUIRE(particles.put(child));
  ProjectData project("frozen_orbital_target", ProjectData::DriverVersion::BATCH);
  WaveFunctionPool wavefunctions(project.getRuntimeOptions(), particles, comm);
  Libxml2Document source;
  REQUIRE(source.parse((directory / "frozen_orbitals.wfnoj.xml").string()));
  auto wavefunction = xmlFirstElementChild(source.getRoot());
  auto determinantset = xmlFirstElementChild(wavefunction);
  // Analytic Cartesian Gaussians avoid introducing a radial-spline tolerance.
  xmlSetProp(determinantset, BAD_CAST "transform", BAD_CAST "no");
  xmlSetProp(determinantset, BAD_CAST "keyword", BAD_CAST "GTO");
  xmlSetProp(determinantset, BAD_CAST "href", BAD_CAST (directory / "frozen_orbitals.h5").c_str());
  REQUIRE(wavefunctions.put(wavefunction));
  auto& teacher = wavefunctions.getWaveFunction().value().get();
  auto determinants = teacher.findSD();
  REQUIRE(determinants.size() == 1);
  auto target = psiformer::OrbitalTargetEvaluator::fromSingleReference(determinants[0], "frozen-cartesian-target", 16);
  auto& electrons = *particles.getParticleSet("e");
  const std::array<ParticleSet::PosType, 4> points{{{.2, .3, -.4}, {1, -.2, .5}, {2.8, .4, .1}, {-.7, .1, .2}}};
  const std::array<double, 4> expected{{.027654020378702484, .08717414156316633,
      .2820921903779103, -.01056934773879696}};
  for (std::size_t point = 0; point < points.size(); ++point)
  {
    electrons.R[0] = electrons.R[1] = points[point];
    electrons.update();
    SPOSet::ValueVector values(49);
    determinants[0].get().getPhi().evaluateValue(electrons, 0, values);
    CAPTURE(point, values[0], expected[point]);
    CHECK(std::abs(values[0] - expected[point]) < 1e-10);
    const auto& matrix = target.evaluate(electrons);
    REQUIRE(matrix.size() == 64);
    for (std::size_t d = 0; d < 16; ++d)
    {
      CHECK(std::abs(matrix[4 * d] - expected[point]) < 1e-10);
      CHECK(std::abs(matrix[4 * d + 3] - expected[point]) < 1e-10);
      CHECK(matrix[4 * d + 1] == 0);
      CHECK(matrix[4 * d + 2] == 0);
    }
  }
}

TEST_CASE("Fixed orbital MSE VJP and published fresh Adam update match a frozen oracle",
          "[.orbital_update_fixture]")
{
  using namespace psiformer;
  using namespace wftrain;
  const char* path = std::getenv("QMC_ORBITAL_UPDATE_ORACLE");
  REQUIRE(path != nullptr);
  hdf_archive oracle;
  REQUIRE(oracle.open(path, H5F_ACC_RDONLY));
  std::vector<double> before_values, expected_gradient, expected_delta, expected_after;
  double expected_loss;
  REQUIRE(oracle.readEntry(before_values, "parameters_before"));
  REQUIRE(oracle.readEntry(expected_gradient, "gradient"));
  REQUIRE(oracle.readEntry(expected_delta, "adam_delta"));
  REQUIRE(oracle.readEntry(expected_after, "parameters_after"));
  REQUIRE(oracle.readEntry(expected_loss, "loss"));
  const SimulationCell cell;
  ParticleSet electrons(cell), ions(cell);
  electrons.setName("e");
  electrons.create({1, 1});
  auto& species = electrons.getSpeciesSet();
  species.addSpecies("u");
  species.addSpecies("d");
  const int mass = species.addAttribute("mass");
  species(mass, 0) = species(mass, 1) = 1.0;
  electrons.resetGroups();
  ions.setName("ion0");
  ions.create({1, 1});
  auto& nuclear_species = ions.getSpeciesSet();
  nuclear_species.addSpecies("Li");
  nuclear_species.addSpecies("H");
  const int charge = nuclear_species.addAttribute("charge");
  nuclear_species(charge, 0) = nuclear_species(charge, 1) = 1.0;
  ions.resetGroups();
  ions.R[0] = {0, 0, 0};
  ions.R[1] = {3.0154361724853516, 0, 0};
  const ModelShape shape{1, 1, 2, 16, 256, 4, 4};
  auto initialized = initializePsiFormerParameters(shape, 31);
  REQUIRE(initialized.values.size() == before_values.size());
  initialized.values = before_values;
  PsiFormerWF component("orbital_update", std::move(initialized), electrons, ions);
  const auto before = component.snapshotParameters();
  REQUIRE(before.values == before_values);
  const std::array<ParticleSet::PosType, 4> points{{{.2, .3, -.4}, {1, -.2, .5}, {2.8, .4, .1}, {-.7, .1, .2}}};
  const std::array<double, 4> occupied{{.027654020378702484, .08717414156316633,
      .2820921903779103, -.01056934773879696}};
  std::vector<double> gradient(before.values.size(), 0.0), sample_gradient(before.values.size()), target(64, 0.0);
  double loss = 0;
  for (std::size_t sample = 0; sample < 2; ++sample)
  {
    electrons.R[0] = points[2 * sample];
    electrons.R[1] = points[2 * sample + 1];
    for (std::size_t d = 0; d < 16; ++d)
    {
      target[4 * d] = occupied[2 * sample];
      target[4 * d + 3] = occupied[2 * sample + 1];
    }
    loss += 0.5 * component.evaluateOrbitalMSE(electrons, target, sample_gradient, before.version);
    for (std::size_t i = 0; i < gradient.size(); ++i) gradient[i] += 0.5 * sample_gradient[i];
  }
  REQUIRE(gradient.size() == expected_gradient.size());
  double maximum_scaled_gradient_error = 0;
  for (std::size_t i = 0; i < gradient.size(); ++i)
    maximum_scaled_gradient_error = std::max(maximum_scaled_gradient_error,
        std::abs(gradient[i] - expected_gradient[i]) / (1e-10 + 1e-8 * std::abs(expected_gradient[i])));
  CAPTURE(loss, expected_loss, maximum_scaled_gradient_error);
  CHECK(std::abs(loss - expected_loss) < 1e-12);
  REQUIRE(maximum_scaled_gradient_error <= 1);
  FirstOrderOptimizerOptions options;
  options.method = FirstOrderMethod::ADAM;
  options.adam_beta1 = .9;
  options.adam_beta2 = .999;
  options.epsilon = 1e-8;
  std::set<std::string> groups;
  for (const auto& block : component.parameterSchema().blocks())
    if (block.trainable) groups.insert(block.update_group);
  for (const auto& group : groups) options.learning_rates.push_back({group, 3e-4});
  FirstOrderOptimizer optimizer(component.parameterSchema(), options);
  ParameterGradientView objective{before.schema_fingerprint, before.version, ReductionDomain::GLOBAL,
      {gradient.data(), gradient.size()}};
  REQUIRE(optimizer.acceptedUpdateCount() == 0);
  const auto candidate = optimizer.propose(component.parameterSchema(), before, objective);
  REQUIRE(candidate.values.size() == expected_delta.size());
  REQUIRE(candidate.values.size() == expected_after.size());
  double maximum_delta_error = 0, maximum_after_error = 0;
  for (std::size_t i = 0; i < gradient.size(); ++i)
  {
    maximum_delta_error = std::max(maximum_delta_error,
        std::abs(candidate.values[i] - before.values[i] - expected_delta[i]));
    maximum_after_error = std::max(maximum_after_error, std::abs(candidate.values[i] - expected_after[i]));
  }
  CAPTURE(maximum_delta_error, maximum_after_error);
  REQUIRE(maximum_delta_error < 1e-12);
  REQUIRE(maximum_after_error < 1e-12);
  const auto published = component.publishParameters(candidate, before.version);
  optimizer.proposalAccepted(component.parameterSchema(), before, objective);
  CHECK(published == before.version + 1);
  CHECK(component.snapshotParameters().values == candidate.values);
  CHECK(optimizer.acceptedUpdateCount() == 1);
}

TEST_CASE("Prepared native training fixture passes complete input admission",
          "[.wftrain_input_fixture]")
{
  const char* path = std::getenv("QMC_TRAINING_INPUT_FIXTURE");
  REQUIRE(path != nullptr);
  Libxml2Document document;
  REQUIRE(document.parse(path));
  std::size_t phases = 0;
  for (auto child = xmlFirstElementChild(document.getRoot()); child; child = xmlNextElementSibling(child))
    if (std::string(castXMLCharToChar(child->name)) == "qmc")
    {
      QMCDriverInput sampling;
      VMCDriverInput vmc;
      WFTrainDriverInput training;
      WFTrainInferenceInput inference;
      REQUIRE_NOTHROW(sampling.readXML(child));
      REQUIRE_NOTHROW(vmc.readXML(child));
      REQUIRE_NOTHROW(training.readXML(child));
      REQUIRE_NOTHROW(inference.readXML(child, training.getObjective() != WFTrainDriverInput::Objective::ORBITAL_MSE));
      CHECK(sampling.get_walkers_per_rank() == 512);
      CHECK(sampling.get_sub_steps() == 20);
      CHECK_FALSE(vmc.get_use_drift());
      ++phases;
    }
  REQUIRE(phases > 0);
}
}
