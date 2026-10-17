//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2020 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Message/Communicate.h"
#include "QMCDrivers/VMC/VMCDriverInput.h"
#include "QMCDrivers/VMC/VMCBatched.h"
#include "QMCDrivers/tests/ValidQMCInputSections.h"
#include "EstimatorInputDelegates.h"
#include <MinimalParticlePool.h>
#include <MinimalWaveFunctionPool.h>
#include <MinimalHamiltonianPool.h>
#include "Concurrency/Info.hpp"
#include "Concurrency/UtilityFunctions.hpp"
#include "Particle/SampleStack.h"
#include "QMCDrivers/VMC/VMCBatchedSampling.h"
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/tests/psiformer_test_utils.h"
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceRuntime.h"
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceMetrics.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "QMCHamiltonians/CoulombPotential.h"
#include "ResourceCollection.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <optional>
#endif

namespace qmcplusplus
{
namespace testing
{
class VMCBatchedTest
{
public:
  VMCBatchedTest(const ProjectData& project_data) : project_data_(project_data)
  {
    Concurrency::OverrideMaxCapacity<> override(8);
    comm_ = OHMMS::Controller;
  }

  void testCalcDefaultLocalWalkers()
  {
    using namespace testing;
    Concurrency::OverrideMaxCapacity<> override(8);

    Libxml2Document doc;
    doc.parseFromString(valid_vmc_input_sections[valid_vmc_input_vmc_batch_index]);
    xmlNodePtr node = doc.getRoot();
    QMCDriverInput qmcdriver_input;
    qmcdriver_input.readXML(node);

    auto particle_pool = MinimalParticlePool::make_diamondC_1x1x1(comm_);
    auto wavefunction_pool =
        MinimalWaveFunctionPool::make_diamondC_1x1x1(project_data_.getRuntimeOptions(), comm_, particle_pool);
    auto hamiltonian_pool = MinimalHamiltonianPool::make_hamWithEE(comm_, particle_pool, wavefunction_pool);

    VMCDriverInput vmc_input;
    vmc_input.readXML(node);
    CHECK_NOTHROW(VMCBatchedSampling::validateMoveConfiguration(
        vmc_input, *particle_pool.getParticleSet("e"), wavefunction_pool.getWaveFunction().value(), false));
    CHECK_FALSE(VMCBatchedSampling::canReuseRejectedDeviceFullState(
        wavefunction_pool.getWaveFunction().value(), *particle_pool.getParticleSet("e"), 0, false));
  }

private:
  Communicate* comm_;
  const ProjectData& project_data_;
};
} // namespace testing

TEST_CASE("VMCBatched::calc_default_local_walkers", "[drivers]")
{
  using namespace testing;
  ProjectData test_project("test", ProjectData::DriverVersion::BATCH);
  VMCBatchedTest vbt(test_project);
  vbt.testCalcDefaultLocalWalkers();
}

#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
TEST_CASE("Device sampling reuses FULL state only after proved rejection",
          "[drivers][psiformer][rejected_full_reuse]")
{
  namespace ps = psiformer;
  namespace fixture = testing::psiformer;
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
  { WARN("NOT_RUN: rejected FULL reuse requires a HIP device"); return; }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
  { WARN("NOT_RUN: rejected FULL reuse requires a CUDA device"); return; }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  struct Controls
  {
    std::vector<std::pair<std::string, std::optional<std::string>>> saved;
    Controls()
    {
      for (const auto& item : {std::pair{"PSIFORMER_GPU_METROPOLIS_BYTES", "16777216"},
                              std::pair{"PSIFORMER_GPU_METROPOLIS_DRIFT_BYTES", ""},
                              std::pair{"PSIFORMER_GPU_AGGREGATE_DEVICE_BYTES", "3221225472"}})
      {
        const char* previous = std::getenv(item.first);
        saved.push_back({item.first, previous ? std::optional<std::string>(previous) : std::nullopt});
        if (*item.second) REQUIRE(setenv(item.first, item.second, 1) == 0);
        else REQUIRE(unsetenv(item.first) == 0);
      }
    }
    ~Controls()
    {
      for (const auto& item : saved)
        if (item.second) setenv(item.first.c_str(), item.second->c_str(), 1);
        else unsetenv(item.first.c_str());
    }
  } controls;
  auto leaves = fixture::makeLayout(2, 2, false, 1, 4);
  for (auto& leaf : leaves)
    for (auto& extent : leaf.shape)
      if (extent == 256) extent = 8;
      else if (extent == 32) extent = 4;
  ps::InitializedPsiFormerParameters initialized;
  initialized.model_shape = {1, 1, 2, 2, 8, 2, 1};
  initialized.profile = ps::DEEPQMC_PSIFORMER_V1;
  initialized.seed = 37;
  initialized.values = fixture::makeParameters("h2", leaves);
  std::size_t cursor = 0;
  for (const auto& leaf : leaves)
  {
    const auto begin = cursor; cursor += fixture::product(leaf.shape);
    initialized.layouts.push_back({leaf.module, leaf.name, leaf.shape, begin, cursor});
  }
  const SimulationCell cell;
  ParticleSet electrons(cell), ions(cell);
  electrons.setName("e"); electrons.create({1, 1});
  auto& es = electrons.getSpeciesSet(); es.addSpecies("u"); es.addSpecies("d");
  const int mass = es.addAttribute("mass"), charge = es.addAttribute("charge");
  for (int spin = 0; spin < 2; ++spin) { es(mass, spin) = 1; es(charge, spin) = -1; }
  electrons.R[0] = {.7, .2, .4}; electrons.R[1] = {-.5, .3, 1.0}; electrons.resetGroups();
  ions.setName("ion0"); ions.create({2}); ions.getSpeciesSet().addSpecies("H");
  const int ion_charge = ions.getSpeciesSet().addAttribute("charge");
  ions.getSpeciesSet()(ion_charge, 0) = 1;
  ions.R[0] = {0, 0, 0}; ions.R[1] = {0, 0, 2.5}; ions.resetGroups(); ions.update();
  auto resolution = ps::PsiFormerAcceleratorResolution{};
#ifdef QMC_CUDA2HIP
  resolution.backend = ps::PsiFormerAcceleratorBackend::HIP;
#else
  resolution.backend = ps::PsiFormerAcceleratorBackend::CUDA;
#endif
  resolution.reason = ps::PsiFormerAcceleratorResolutionReason::ACCELERATOR_READY;
  resolution.readiness_records_consulted = 1;
  resolution.validated_operations = ps::psiFormerAcceleratorInferenceOperations();
  RuntimeOptions options;
  TrialWaveFunction trial(options);
  auto owner = std::make_unique<PsiFormerWF>("reject_full", initialized, electrons, ions,
      false, std::vector<std::size_t>{}, false, "", std::array<double, 3>{}, resolution);
  auto& component = *owner;
  trial.addComponent(std::move(owner));
  BareKineticEnergy kinetic(electrons);
  CoulombPotential ee(electrons, true, false), ei(ions, electrons, true), ii(ions, false, false);
  electrons.update();
  const auto energy = [&]() { return kinetic.evaluate(trial, electrons) + ee.evaluate(electrons) +
                                    ei.evaluate(electrons) + ii.evaluate(ions); };
  RefVectorWithLeader<TrialWaveFunction> trials(trial, {trial});
  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons});
  RefVectorWithLeader<WaveFunctionComponent> components(component, {component});
  ResourceCollection prototype("rejected_full"); trial.createResource(prototype);
  ResourceCollection resources(prototype);
  ResourceCollectionTeamLock<TrialWaveFunction> lock(resources, trials);
  TrialWaveFunction::mw_evaluateLog(trials, particles);
  const auto old_g = electrons.G;
  const auto old_l = electrons.L;
  const auto old_log = trial.getLogPsi();
  const auto old_phase = trial.getPhase();
  const auto old_energy = energy();
  auto& metrics = ps::device::deviceMemoryMetrics(ordinal);
  std::size_t gaussian_draws = 0, uniform_draws = 0;
  const auto rejected = component.mw_advanceDeviceSampling(components, particles, 1.0, 2,
      [&](std::vector<double>& values) { ++gaussian_draws; std::fill(values.begin(), values.end(), 20.0); },
      [&]() { ++uniform_draws; return .999999999; }, false, {});
  REQUIRE(rejected.first == 0); REQUIRE(rejected.second == 2);
  CHECK(gaussian_draws == 2); CHECK(uniform_draws == 2);
  electrons.update(); // The native table refresh remains mandatory.
  const auto before = metrics.snapshot();
  REQUIRE(VMCBatchedSampling::canReuseRejectedDeviceFullState(trial, electrons, rejected.first, false));
  const auto after = metrics.snapshot();
  CHECK(after.download_calls == before.download_calls);
  CHECK(after.upload_calls == before.upload_calls);
  CHECK(after.allocation_calls == before.allocation_calls);
  CHECK(trial.getLogPsi() == old_log); CHECK(trial.getPhase() == old_phase);
  CHECK(energy() == old_energy);
  for (int i = 0; i < electrons.getTotalNum(); ++i)
  { CHECK(electrons.G[i] == old_g[i]); CHECK(electrons.L[i] == old_l[i]); }
  TrialWaveFunction::mw_evaluateLog(trials, particles); // Independent forced replay oracle.
  CHECK(energy() == Catch::Approx(old_energy).epsilon(1e-12).margin(1e-12));
  CHECK_FALSE(VMCBatchedSampling::canReuseRejectedDeviceFullState(trial, electrons, 0, true));
  const auto accepted = component.mw_advanceDeviceSampling(components, particles, .01, 1,
      [&](std::vector<double>& values) { ++gaussian_draws; std::fill(values.begin(), values.end(), .01); },
      [&]() { ++uniform_draws; return 0.0; }, false, {});
  REQUIRE(accepted.first == 1); REQUIRE(accepted.second == 0);
  CHECK(gaussian_draws == 3); CHECK(uniform_draws == 3);
  electrons.update();
  CHECK_FALSE(VMCBatchedSampling::canReuseRejectedDeviceFullState(trial, electrons, accepted.first, false));
  const auto before_refresh = metrics.snapshot();
  TrialWaveFunction::mw_evaluateLog(trials, particles);
  CHECK(metrics.snapshot().download_calls > before_refresh.download_calls);
  auto candidate = component.snapshotParameters(); candidate.values[0] += 1e-4;
  component.publishParameters(candidate, candidate.version);
  CHECK_FALSE(VMCBatchedSampling::canReuseRejectedDeviceFullState(trial, electrons, 0, false));
  const auto before_version_refresh = metrics.snapshot();
  TrialWaveFunction::mw_evaluateLog(trials, particles);
  CHECK(metrics.snapshot().download_calls > before_version_refresh.download_calls);
  CHECK(VMCBatchedSampling::canReuseRejectedDeviceFullState(trial, electrons, 0, false));
  TrialWaveFunction missing(options);
  missing.addComponent(std::make_unique<PsiFormerWF>("missing_full", initialized, electrons, ions,
      false, std::vector<std::size_t>{}, false, "", std::array<double, 3>{}, resolution));
  CHECK_FALSE(VMCBatchedSampling::canReuseRejectedDeviceFullState(missing, electrons, 0, false));
}
#endif

} // namespace qmcplusplus
