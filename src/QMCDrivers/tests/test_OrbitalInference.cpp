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
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCHamiltonians/OperatorBase.h"
#include <MinimalParticlePool.h>
#include <MinimalWaveFunctionPool.h>
#include <MinimalHamiltonianPool.h>
#include <cmath>
#include <memory>

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
}
