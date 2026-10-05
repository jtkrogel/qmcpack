//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_threading.cpp
 * @brief Threading regressions for shared-model PsiFormer clones and crowd resources.
 */

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Concurrency/ParallelExecutor.hpp"
#include "Particle/MCMultiParticleMoves.h"
#include "Particle/ParticleSet.h"
#include "Particle/VirtualParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "ResourceCollection.h"
#include "psiformer_test_utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace qmcplusplus
{
namespace testing
{
/** Narrow access to the workspace ownership diagnostics used by this isolated target. */
class TestPsiFormerWF
{
public:
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
};
} // namespace testing

namespace
{
using namespace testing::psiformer;
using Value = QMCTraits::ValueType;

constexpr int thread_count = 2;

#ifdef PSIFORMER_THREAD_EXECUTOR_STD
using PsiFormerThreadExecutor = ParallelExecutor<Executor::STD_THREADS>;
#else
using PsiFormerThreadExecutor = ParallelExecutor<Executor::OPENMP>;
#endif

/// Construct one deterministically displaced LiH walker.
std::unique_ptr<ParticleSet> makeThreadWalker(const SimulationCell& simulation_cell,
                                              std::size_t walker)
{
  const Geometry geometry = makeGeometry("lih");
  auto particles          = std::make_unique<ParticleSet>(simulation_cell);
  particles->setName("thread_walker_" + std::to_string(walker));
  particles->create({static_cast<int>(geometry.nup),
                     static_cast<int>(geometry.electrons.size() / 3 - geometry.nup)});
  SpeciesSet& species = particles->getSpeciesSet();
  species.addSpecies("u");
  species.addSpecies("d");
  const int mass = species.addAttribute("mass");
  species(mass, 0) = 1.0;
  species(mass, 1) = 1.0;
  particles->resetGroups();
  for (int electron = 0; electron < particles->getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      particles->R[electron][dimension] = geometry.electrons[3 * electron + dimension] +
          0.006 * static_cast<double>(walker) *
              static_cast<double>((electron + 1) * (dimension + 1));
  particles->update();
  return particles;
}

/// Compare real or complex QMCPACK values using one common tolerance.
template<class Actual, class Expected>
void checkThreadValue(const Actual& actual, const Expected& expected, double tolerance)
{
  CHECK(std::real(actual) == Catch::Approx(std::real(expected)).epsilon(tolerance).margin(tolerance));
  CHECK(std::imag(actual) == Catch::Approx(std::imag(expected)).epsilon(tolerance).margin(tolerance));
}

template<class Actual, class Expected>
bool threadValuesClose(const Actual& actual, const Expected& expected, double tolerance)
{
  const double real_expected = std::real(expected);
  const double imag_expected = std::imag(expected);
  return std::abs(std::real(actual) - real_expected) <= tolerance * (1.0 + std::abs(real_expected)) &&
      std::abs(std::imag(actual) - imag_expected) <= tolerance * (1.0 + std::abs(imag_expected));
}

void requireThreadCapacity()
{
#ifdef PSIFORMER_THREAD_EXECUTOR_STD
  REQUIRE(Concurrency::maxCapacity<Executor::STD_THREADS>() >= thread_count);
#else
  REQUIRE(omp_get_max_threads() >= thread_count);
#endif
}

void rethrowThreadFailures(const std::array<std::exception_ptr, thread_count>& failures)
{
  for (const std::exception_ptr& failure : failures)
    if (failure)
      std::rethrow_exception(failure);
}

/** Synchronize two tasks only after the selected executor has two workers. */
void enterThreadedSection(std::atomic<int>& ready)
{
  ready.fetch_add(1, std::memory_order_acq_rel);
  while (ready.load(std::memory_order_acquire) != thread_count)
    std::this_thread::yield();
}

struct CrowdFingerprint
{
  std::vector<PsiFormerWF::LogValue> logs;
  std::vector<Value> gradients;
  std::vector<Value> laplacians;
};

CrowdFingerprint evaluateAcquiredCrowd(
    PsiFormerWF& leader,
    const RefVectorWithLeader<WaveFunctionComponent>& components,
    const RefVectorWithLeader<ParticleSet>& particles,
    std::vector<ParticleSet::ParticleGradient>& gradient_storage,
    std::vector<ParticleSet::ParticleLaplacian>& laplacian_storage)
{
  RefVector<ParticleSet::ParticleGradient> gradients;
  RefVector<ParticleSet::ParticleLaplacian> laplacians;
  for (std::size_t walker = 0; walker < components.size(); ++walker)
  {
    gradient_storage[walker]  = Value(0);
    laplacian_storage[walker] = Value(0);
    gradients.push_back(gradient_storage[walker]);
    laplacians.push_back(laplacian_storage[walker]);
  }

  leader.mw_evaluateLog(components, particles, gradients, laplacians);

  CrowdFingerprint fingerprint;
  fingerprint.logs.reserve(components.size());
  for (std::size_t walker = 0; walker < components.size(); ++walker)
  {
    fingerprint.logs.push_back(components.getCastedElement<PsiFormerWF>(walker).get_log_value());
    const int electron_count = particles[walker].getTotalNum();
    for (int electron = 0; electron < electron_count; ++electron)
    {
      for (int dimension = 0; dimension < 3; ++dimension)
        fingerprint.gradients.push_back(gradient_storage[walker][electron][dimension]);
      fingerprint.laplacians.push_back(laplacian_storage[walker][electron]);
    }
  }
  return fingerprint;
}

bool fingerprintMatches(const CrowdFingerprint& actual, const CrowdFingerprint& expected)
{
  if (actual.logs.size() != expected.logs.size() ||
      actual.gradients.size() != expected.gradients.size() ||
      actual.laplacians.size() != expected.laplacians.size())
    return false;
  for (std::size_t index = 0; index < actual.logs.size(); ++index)
    if (!threadValuesClose(actual.logs[index], expected.logs[index], 3.0e-9))
      return false;
  for (std::size_t index = 0; index < actual.gradients.size(); ++index)
    if (!threadValuesClose(actual.gradients[index], expected.gradients[index], 3.0e-8))
      return false;
  for (std::size_t index = 0; index < actual.laplacians.size(); ++index)
    if (!threadValuesClose(actual.laplacians[index], expected.laplacians[index], 3.0e-7))
      return false;
  return true;
}

OptVariables registerSelectedParameters(PsiFormerWF& component)
{
  OptVariables active;
  component.checkInVariablesExclusive(active);
  active.resetIndex();
  component.checkOutVariables(active);
  return active;
}

void initializeTotalDrift(PsiFormerWF& component, ParticleSet& particles, double scale)
{
  particles.G = Value(0);
  particles.L = Value(0);
  component.evaluateLog(particles, particles.G, particles.L);
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      particles.G[electron][dimension] += Value(scale * (1 + 3 * electron + dimension));
}

} // namespace

TEST_CASE("PsiFormer independent crowd resources run concurrently",
          "[wavefunction][psiformer][multiwalker][threading]")
{
  requireThreadCapacity();
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  constexpr std::size_t walker_count = 4;
  constexpr std::size_t crowd_size   = 2;

  std::vector<std::unique_ptr<ParticleSet>> walkers;
  for (std::size_t walker = 0; walker < walker_count; ++walker)
    walkers.push_back(makeThreadWalker(simulation_cell, walker));

  PsiFormerWF first_leader("pf_threaded", files.parameters.string(),
                           files.configuration.string());
  std::vector<std::unique_ptr<WaveFunctionComponent>> clone_storage;
  for (std::size_t walker = 1; walker < walker_count; ++walker)
    clone_storage.push_back(first_leader.makeClone(*walkers[walker]));
  auto& first_follower  = dynamic_cast<PsiFormerWF&>(*clone_storage[0]);
  auto& second_leader   = dynamic_cast<PsiFormerWF&>(*clone_storage[1]);
  auto& second_follower = dynamic_cast<PsiFormerWF&>(*clone_storage[2]);

  RefVectorWithLeader<WaveFunctionComponent> first_components(first_leader);
  first_components.push_back(first_leader);
  first_components.push_back(first_follower);
  RefVectorWithLeader<WaveFunctionComponent> second_components(second_leader);
  second_components.push_back(second_leader);
  second_components.push_back(second_follower);

  RefVectorWithLeader<ParticleSet> first_particles(*walkers[0]);
  first_particles.push_back(*walkers[0]);
  first_particles.push_back(*walkers[1]);
  RefVectorWithLeader<ParticleSet> second_particles(*walkers[2]);
  second_particles.push_back(*walkers[2]);
  second_particles.push_back(*walkers[3]);

  std::vector<ParticleSet::ParticleGradient> expected_gradient(walker_count);
  std::vector<ParticleSet::ParticleLaplacian> expected_laplacian(walker_count);
  std::vector<PsiFormerWF::LogValue> expected_log(walker_count);
  std::vector<ParticleSet::ParticleGradient> actual_gradient(walker_count);
  std::vector<ParticleSet::ParticleLaplacian> actual_laplacian(walker_count);
  std::vector<PsiFormerWF*> components{
      &first_leader, &first_follower, &second_leader, &second_follower};
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    const std::size_t electrons = walkers[walker]->getTotalNum();
    expected_gradient[walker].resize(electrons);
    expected_laplacian[walker].resize(electrons);
    actual_gradient[walker].resize(electrons);
    actual_laplacian[walker].resize(electrons);
    expected_gradient[walker]  = Value(0);
    expected_laplacian[walker] = Value(0);
    actual_gradient[walker]    = Value(0);
    actual_laplacian[walker]   = Value(0);
    expected_log[walker]       = components[walker]->evaluateLog(
        *walkers[walker], expected_gradient[walker], expected_laplacian[walker]);
  }

  RefVector<ParticleSet::ParticleGradient> first_gradients;
  RefVector<ParticleSet::ParticleLaplacian> first_laplacians;
  RefVector<ParticleSet::ParticleGradient> second_gradients;
  RefVector<ParticleSet::ParticleLaplacian> second_laplacians;
  for (std::size_t walker = 0; walker < crowd_size; ++walker)
  {
    first_gradients.push_back(actual_gradient[walker]);
    first_laplacians.push_back(actual_laplacian[walker]);
    second_gradients.push_back(actual_gradient[crowd_size + walker]);
    second_laplacians.push_back(actual_laplacian[crowd_size + walker]);
  }

  ResourceCollection resource_template("psiformer_thread_template");
  first_leader.createResource(resource_template);
  ResourceCollection first_resource(resource_template);
  ResourceCollection second_resource(resource_template);

  std::array<std::exception_ptr, thread_count> failures{};
  std::atomic<int> ready{0};
  PsiFormerThreadExecutor executor;
  executor(thread_count, [&](int task) {
    try
    {
      enterThreadedSection(ready);
      if (task == 0)
      {
        ResourceCollectionTeamLock<WaveFunctionComponent> lock(first_resource, first_components);
        first_leader.mw_evaluateLog(
            first_components, first_particles, first_gradients, first_laplacians);
      }
      else
      {
        ResourceCollectionTeamLock<WaveFunctionComponent> lock(second_resource, second_components);
        second_leader.mw_evaluateLog(
            second_components, second_particles, second_gradients, second_laplacians);
      }
    }
    catch (...)
    {
      failures[task] = std::current_exception();
    }
  });
  REQUIRE_NOTHROW(rethrowThreadFailures(failures));

  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    checkThreadValue(components[walker]->get_log_value(), expected_log[walker], 3.0e-9);
    for (int electron = 0; electron < walkers[walker]->getTotalNum(); ++electron)
    {
      for (int dimension = 0; dimension < 3; ++dimension)
        checkThreadValue(actual_gradient[walker][electron][dimension],
                         expected_gradient[walker][electron][dimension], 3.0e-8);
      checkThreadValue(actual_laplacian[walker][electron],
                       expected_laplacian[walker][electron], 3.0e-7);
    }
  }
}

TEST_CASE("PsiFormer readers observe one complete published parameter version",
          "[wavefunction][psiformer][threading][publication]")
{
  requireThreadCapacity();
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  constexpr std::size_t crowd_size        = 2;
  constexpr std::size_t publication_count = 12;
  constexpr std::size_t observation_count = 16;

  std::vector<std::unique_ptr<ParticleSet>> walkers;
  for (std::size_t walker = 0; walker < crowd_size; ++walker)
    walkers.push_back(makeThreadWalker(simulation_cell, walker + 1));

  PsiFormerWF publisher("pf_publisher", files.parameters.string(), files.configuration.string());
  std::unique_ptr<WaveFunctionComponent> reader_leader_storage = publisher.makeClone(*walkers[0]);
  std::unique_ptr<WaveFunctionComponent> reader_follower_storage = publisher.makeClone(*walkers[1]);
  auto& reader_leader   = dynamic_cast<PsiFormerWF&>(*reader_leader_storage);
  auto& reader_follower = dynamic_cast<PsiFormerWF&>(*reader_follower_storage);

  RefVectorWithLeader<WaveFunctionComponent> reader_components(reader_leader);
  reader_components.push_back(reader_leader);
  reader_components.push_back(reader_follower);
  RefVectorWithLeader<ParticleSet> reader_particles(*walkers[0]);
  reader_particles.push_back(*walkers[0]);
  reader_particles.push_back(*walkers[1]);

  std::vector<ParticleSet::ParticleGradient> gradient_storage(crowd_size);
  std::vector<ParticleSet::ParticleLaplacian> laplacian_storage(crowd_size);
  for (std::size_t walker = 0; walker < crowd_size; ++walker)
  {
    gradient_storage[walker].resize(walkers[walker]->getTotalNum());
    laplacian_storage[walker].resize(walkers[walker]->getTotalNum());
  }

  ResourceCollection resource_template("psiformer_publication_template");
  reader_leader.createResource(resource_template);
  ResourceCollection reader_resource(resource_template);
  auto evaluate_reader = [&]() {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(reader_resource, reader_components);
    return evaluateAcquiredCrowd(reader_leader, reader_components, reader_particles,
                                 gradient_storage, laplacian_storage);
  };

  const wftrain::StructuredParameterSnapshot state_a = publisher.snapshotParameters();
  REQUIRE_FALSE(state_a.values.empty());
  const CrowdFingerprint expected_a = evaluate_reader();

  wftrain::StructuredParameterSnapshot state_b = state_a;
  state_b.values[0] += 1.0e-3;
  const std::size_t version_b = publisher.publishParameters(state_b, state_a.version);
  const CrowdFingerprint expected_b = evaluate_reader();
  REQUIRE_FALSE(fingerprintMatches(expected_a, expected_b));

  wftrain::StructuredParameterSnapshot restore_a = state_a;
  restore_a.version = version_b;
  const std::size_t restored_version = publisher.publishParameters(restore_a, version_b);
  const CrowdFingerprint restored_a = evaluate_reader();
  REQUIRE(fingerprintMatches(restored_a, expected_a));

  std::array<std::exception_ptr, thread_count> failures{};
  std::array<std::size_t, publication_count> committed_versions{};
  std::vector<CrowdFingerprint> observations(observation_count);
  std::atomic<int> ready{0};
  PsiFormerThreadExecutor executor;
  executor(thread_count, [&](int task) {
    try
    {
      enterThreadedSection(ready);
      if (task == 0)
      {
        for (std::size_t update = 0; update < publication_count; ++update)
        {
          wftrain::StructuredParameterSnapshot candidate = publisher.snapshotParameters();
          candidate.values = update % 2 == 0 ? state_b.values : state_a.values;
          committed_versions[update] = publisher.publishParameters(candidate, candidate.version);
        }
      }
      else
      {
        ResourceCollectionTeamLock<WaveFunctionComponent> lock(reader_resource, reader_components);
        for (std::size_t observation = 0; observation < observation_count; ++observation)
          observations[observation] = evaluateAcquiredCrowd(
              reader_leader, reader_components, reader_particles,
              gradient_storage, laplacian_storage);
      }
    }
    catch (...)
    {
      failures[task] = std::current_exception();
    }
  });
  REQUIRE_NOTHROW(rethrowThreadFailures(failures));

  std::size_t previous_version = restored_version;
  for (const std::size_t version : committed_versions)
  {
    CHECK(version == previous_version + 1);
    previous_version = version;
  }
  const wftrain::StructuredParameterSnapshot final_snapshot = publisher.snapshotParameters();
  CHECK(final_snapshot.version == previous_version);
  CHECK(final_snapshot.values == state_a.values);

  for (const CrowdFingerprint& observation : observations)
  {
    const bool matches_a = fingerprintMatches(observation, expected_a);
    const bool matches_b = fingerprintMatches(observation, expected_b);
    CHECK(matches_a != matches_b);
  }
}

TEST_CASE("PsiFormer crowds initialize score and kinetic workspaces concurrently",
          "[wavefunction][psiformer][threading][derivatives]")
{
  requireThreadCapacity();
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  constexpr std::size_t walker_count = 4;
  constexpr std::size_t crowd_size   = 2;
  const std::array<double, walker_count> drift_scales{0.007, -0.011, 0.013, -0.017};

  std::vector<std::unique_ptr<ParticleSet>> walkers;
  for (std::size_t walker = 0; walker < walker_count; ++walker)
    walkers.push_back(makeThreadWalker(simulation_cell, walker));

  PsiFormerWF first_leader(
      "pf_lazy_threaded", files.parameters.string(), files.configuration.string(), true, {0, 127});
  OptVariables active = registerSelectedParameters(first_leader);
  REQUIRE(active.size() == 2);

  std::vector<std::unique_ptr<WaveFunctionComponent>> clone_storage;
  for (std::size_t walker = 1; walker < walker_count; ++walker)
  {
    clone_storage.push_back(first_leader.makeClone(*walkers[walker]));
    clone_storage.back()->checkOutVariables(active);
  }
  auto& first_follower  = dynamic_cast<PsiFormerWF&>(*clone_storage[0]);
  auto& second_leader   = dynamic_cast<PsiFormerWF&>(*clone_storage[1]);
  auto& second_follower = dynamic_cast<PsiFormerWF&>(*clone_storage[2]);
  std::array<PsiFormerWF*, walker_count> components{
      &first_leader, &first_follower, &second_leader, &second_follower};

  for (std::size_t walker = 0; walker < walker_count; ++walker)
    initializeTotalDrift(*components[walker], *walkers[walker], drift_scales[walker]);

  RefVectorWithLeader<WaveFunctionComponent> first_components(first_leader);
  first_components.push_back(first_leader);
  first_components.push_back(first_follower);
  RefVectorWithLeader<WaveFunctionComponent> second_components(second_leader);
  second_components.push_back(second_leader);
  second_components.push_back(second_follower);
  RefVectorWithLeader<ParticleSet> first_particles(*walkers[0]);
  first_particles.push_back(*walkers[0]);
  first_particles.push_back(*walkers[1]);
  RefVectorWithLeader<ParticleSet> second_particles(*walkers[2]);
  second_particles.push_back(*walkers[2]);
  second_particles.push_back(*walkers[3]);

  ResourceCollection resource_template("psiformer_lazy_derivative_template");
  first_leader.createResource(resource_template);
  ResourceCollection first_resource(resource_template);
  ResourceCollection second_resource(resource_template);

  std::array<RecordArray<Value>, thread_count> score_only_first;
  std::array<RecordArray<Value>, thread_count> score_first;
  std::array<RecordArray<Value>, thread_count> kinetic_first;
  std::array<RecordArray<Value>, thread_count> score_only_warm;
  std::array<RecordArray<Value>, thread_count> score_warm;
  std::array<RecordArray<Value>, thread_count> kinetic_warm;
  for (int crowd = 0; crowd < thread_count; ++crowd)
  {
    score_only_first[crowd].resize(crowd_size, active.size());
    score_first[crowd].resize(crowd_size, active.size());
    kinetic_first[crowd].resize(crowd_size, active.size());
    score_only_warm[crowd].resize(crowd_size, active.size());
    score_warm[crowd].resize(crowd_size, active.size());
    kinetic_warm[crowd].resize(crowd_size, active.size());
    std::fill(score_only_first[crowd].begin(), score_only_first[crowd].end(), Value(0.25));
    std::fill(score_first[crowd].begin(), score_first[crowd].end(), Value(0.25));
    std::fill(kinetic_first[crowd].begin(), kinetic_first[crowd].end(), Value(-0.5));
    std::fill(score_only_warm[crowd].begin(), score_only_warm[crowd].end(), Value(0.25));
    std::fill(score_warm[crowd].begin(), score_warm[crowd].end(), Value(0.25));
    std::fill(kinetic_warm[crowd].begin(), kinetic_warm[crowd].end(), Value(-0.5));
  }

  // A separate scalar component provides numerical oracles without warming any
  // clone- or crowd-local derivative workspace exercised below.
  PsiFormerWF oracle(
      "pf_lazy_oracle", files.parameters.string(), files.configuration.string(), true, {0, 127});
  OptVariables oracle_active = registerSelectedParameters(oracle);
  std::array<Vector<Value>, walker_count> expected_score_only;
  std::array<Vector<Value>, walker_count> expected_score;
  std::array<Vector<Value>, walker_count> expected_kinetic;
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    std::unique_ptr<ParticleSet> oracle_walker = makeThreadWalker(simulation_cell, walker);
    initializeTotalDrift(oracle, *oracle_walker, drift_scales[walker]);
    expected_score_only[walker].resize(active.size());
    expected_score[walker].resize(active.size());
    expected_kinetic[walker].resize(active.size());
    expected_score_only[walker] = Value(0.25);
    expected_score[walker]      = Value(0.25);
    expected_kinetic[walker]    = Value(-0.5);
    oracle.evaluateDerivativesWF(
        *oracle_walker, oracle_active, expected_score_only[walker]);
    oracle.evaluateDerivatives(
        *oracle_walker, oracle_active, expected_score[walker], expected_kinetic[walker]);
  }

  using WorkspaceOwnership = std::array<std::size_t, 2>;
  std::array<WorkspaceOwnership, thread_count> first_ownership{};
  std::array<WorkspaceOwnership, thread_count> warm_ownership{};
  auto run_derivatives = [&](int task, bool warm) {
    PsiFormerWF& leader = task == 0 ? first_leader : second_leader;
    RefVectorWithLeader<WaveFunctionComponent>& crowd_components =
        task == 0 ? first_components : second_components;
    RefVectorWithLeader<ParticleSet>& crowd_particles =
        task == 0 ? first_particles : second_particles;
    ResourceCollection& crowd_resource = task == 0 ? first_resource : second_resource;
    RecordArray<Value>& score_only = warm ? score_only_warm[task] : score_only_first[task];
    RecordArray<Value>& score      = warm ? score_warm[task] : score_first[task];
    RecordArray<Value>& kinetic    = warm ? kinetic_warm[task] : kinetic_first[task];

    ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource, crowd_components);
    leader.mw_evaluateParameterDerivativesWF(
        crowd_components, crowd_particles, active, score_only);
    leader.mw_evaluateParameterDerivatives(
        crowd_components, crowd_particles, active, score, kinetic);
    (warm ? warm_ownership[task] : first_ownership[task]) =
        testing::TestPsiFormerWF::directKineticWorkspaceOwnership(leader, crowd_components);
  };

  std::array<std::exception_ptr, thread_count> failures{};
  std::atomic<int> ready{0};
  PsiFormerThreadExecutor executor;
  executor(thread_count, [&](int task) {
    try
    {
      enterThreadedSection(ready);
      run_derivatives(task, false);
      // Reacquire the same resource to cover the initialized/reuse path as well
      // as the simultaneous lazy-first-use path above.
      run_derivatives(task, true);
    }
    catch (...)
    {
      failures[task] = std::current_exception();
    }
  });
  REQUIRE_NOTHROW(rethrowThreadFailures(failures));

  const WorkspaceOwnership expected_ownership{0, 1};
  for (int crowd = 0; crowd < thread_count; ++crowd)
  {
    CHECK(first_ownership[crowd] == expected_ownership);
    CHECK(warm_ownership[crowd] == expected_ownership);
    for (std::size_t row = 0; row < crowd_size; ++row)
    {
      const std::size_t walker = crowd * crowd_size + row;
      for (std::size_t parameter = 0; parameter < active.size(); ++parameter)
      {
        checkThreadValue(score_only_first[crowd][row][parameter],
                         expected_score_only[walker][parameter], 3.0e-8);
        checkThreadValue(score_first[crowd][row][parameter],
                         expected_score[walker][parameter], 3.0e-8);
        checkThreadValue(kinetic_first[crowd][row][parameter],
                         expected_kinetic[walker][parameter], 3.0e-7);
        checkThreadValue(score_only_warm[crowd][row][parameter],
                         expected_score_only[walker][parameter], 3.0e-8);
        checkThreadValue(score_warm[crowd][row][parameter],
                         expected_score[walker][parameter], 3.0e-8);
        checkThreadValue(kinetic_warm[crowd][row][parameter],
                         expected_kinetic[walker][parameter], 3.0e-7);
      }
    }
  }

  for (const PsiFormerWF* component : components)
  {
    const testing::PsiFormerWorkspaceDiagnostics diagnostics =
        testing::TestPsiFormerWF::directWorkspaceDiagnostics(*component);
    CHECK_FALSE(diagnostics.owns_score_workspace);
    CHECK_FALSE(diagnostics.owns_kinetic_workspace);
  }
}

TEST_CASE("PsiFormer move and virtual derivative crowds remain isolated",
          "[wavefunction][psiformer][multiwalker][threading][ecp]")
{
  requireThreadCapacity();
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  constexpr std::size_t walker_count = 4;
  constexpr std::size_t crowd_size   = 2;

  std::vector<std::unique_ptr<ParticleSet>> walkers;
  for (std::size_t walker = 0; walker < walker_count; ++walker)
    walkers.push_back(makeThreadWalker(simulation_cell, walker));

  PsiFormerWF move_leader(
      "pf_mixed_threaded", files.parameters.string(), files.configuration.string(), true, {0, 127});
  OptVariables active = registerSelectedParameters(move_leader);
  std::vector<std::unique_ptr<WaveFunctionComponent>> clone_storage;
  for (std::size_t walker = 1; walker < walker_count; ++walker)
  {
    clone_storage.push_back(move_leader.makeClone(*walkers[walker]));
    clone_storage.back()->checkOutVariables(active);
  }
  auto& move_follower    = dynamic_cast<PsiFormerWF&>(*clone_storage[0]);
  auto& virtual_leader   = dynamic_cast<PsiFormerWF&>(*clone_storage[1]);
  auto& virtual_follower = dynamic_cast<PsiFormerWF&>(*clone_storage[2]);
  std::array<PsiFormerWF*, walker_count> components{
      &move_leader, &move_follower, &virtual_leader, &virtual_follower};

  std::array<PsiFormerWF::LogValue, walker_count> accepted_logs{};
  std::array<std::vector<ParticleSet::PosType>, crowd_size> accepted_positions;
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    walkers[walker]->G = Value(0);
    walkers[walker]->L = Value(0);
    accepted_logs[walker] = components[walker]->evaluateLog(
        *walkers[walker], walkers[walker]->G, walkers[walker]->L);
    if (walker < crowd_size)
      accepted_positions[walker].assign(walkers[walker]->R.begin(), walkers[walker]->R.end());
  }

  RefVectorWithLeader<WaveFunctionComponent> move_components(move_leader);
  move_components.push_back(move_leader);
  move_components.push_back(move_follower);
  RefVectorWithLeader<ParticleSet> move_particles(*walkers[0]);
  move_particles.push_back(*walkers[0]);
  move_particles.push_back(*walkers[1]);
  RefVectorWithLeader<WaveFunctionComponent> virtual_components(virtual_leader);
  virtual_components.push_back(virtual_leader);
  virtual_components.push_back(virtual_follower);

  ResourceCollection resource_template("psiformer_mixed_operation_template");
  move_leader.createResource(resource_template);
  ResourceCollection move_resource(resource_template);
  ResourceCollection virtual_resource(resource_template);

  using Moves = MCMultiParticleMoves<CoordsType::POS>;
  const std::vector<std::size_t> move_offsets{0, 1, 2};
  const std::vector<Moves::IndexType> move_indices{0, 2};
  const std::vector<Moves::PosType> move_positions{
      walkers[0]->R[0] + Moves::PosType{0.017, -0.009, 0.006},
      walkers[1]->R[2] + Moves::PosType{-0.012, 0.015, 0.008}};
  const Moves moves(move_offsets, move_indices, move_positions);

  PsiFormerWF scalar_oracle(
      "pf_mixed_oracle", files.parameters.string(), files.configuration.string(), true, {0, 127});
  OptVariables oracle_active = registerSelectedParameters(scalar_oracle);
  std::array<PsiFormerWF::LogValue, crowd_size> expected_proposed_logs{};
  std::array<ParticleSet::ParticleGradient, crowd_size> expected_proposed_gradients;
  std::array<ParticleSet::ParticleLaplacian, crowd_size> expected_proposed_laplacians;
  for (std::size_t walker = 0; walker < crowd_size; ++walker)
  {
    std::unique_ptr<ParticleSet> proposed = makeThreadWalker(simulation_cell, walker);
    const auto move = moves.slice(walker);
    proposed->R[move.particleIndex(0)] = move.proposedPosition(0);
    proposed->update();
    expected_proposed_gradients[walker].resize(proposed->getTotalNum());
    expected_proposed_laplacians[walker].resize(proposed->getTotalNum());
    expected_proposed_gradients[walker]  = Value(0);
    expected_proposed_laplacians[walker] = Value(0);
    expected_proposed_logs[walker] = scalar_oracle.evaluateLog(
        *proposed, expected_proposed_gradients[walker], expected_proposed_laplacians[walker]);
  }

  std::array<std::vector<ParticleSet::SingleParticlePos>, crowd_size> displacements{
      std::vector<ParticleSet::SingleParticlePos>{{0.05, -0.02, 0.04}, {-0.03, 0.07, -0.01}},
      std::vector<ParticleSet::SingleParticlePos>{{-0.06, 0.01, 0.02}, {0.08, -0.05, 0.03},
                                                   {0.02, 0.04, -0.07}}};
  std::vector<std::unique_ptr<VirtualParticleSet>> virtual_storage;
  for (std::size_t walker = 0; walker < crowd_size; ++walker)
  {
    const std::size_t source_walker = crowd_size + walker;
    virtual_storage.push_back(std::make_unique<VirtualParticleSet>(*walkers[source_walker]));
    virtual_storage.back()->makeMoves(
        *walkers[source_walker], static_cast<int>(walker + 1), displacements[walker]);
  }
  RefVectorWithLeader<const VirtualParticleSet> virtual_particles(*virtual_storage[0]);
  virtual_particles.push_back(*virtual_storage[0]);
  virtual_particles.push_back(*virtual_storage[1]);
  const std::array<std::vector<Value>, crowd_size> bare_weights{
      std::vector<Value>{Value(0.17), Value(-0.09)},
      std::vector<Value>{Value(-0.13), Value(0.21), Value(0.08)}};

  std::array<std::vector<Value>, crowd_size> expected_ratios;
  std::array<std::vector<Value>, crowd_size> total_weights;
  std::array<Vector<Value>, crowd_size> expected_weighted;
  for (std::size_t walker = 0; walker < crowd_size; ++walker)
  {
    const std::size_t source_walker = crowd_size + walker;
    ParticleSet::ParticleGradient oracle_gradient(walkers[source_walker]->getTotalNum());
    ParticleSet::ParticleLaplacian oracle_laplacian(walkers[source_walker]->getTotalNum());
    oracle_gradient  = Value(0);
    oracle_laplacian = Value(0);
    scalar_oracle.evaluateLog(
        *walkers[source_walker], oracle_gradient, oracle_laplacian);
    expected_ratios[walker].resize(displacements[walker].size());
    Matrix<Value> materialized_derivatives(displacements[walker].size(), active.size());
    materialized_derivatives = Value(0);
    scalar_oracle.evaluateDerivRatios(
        *virtual_storage[walker], oracle_active, expected_ratios[walker], materialized_derivatives);
    total_weights[walker].resize(displacements[walker].size());
    for (std::size_t move = 0; move < displacements[walker].size(); ++move)
      total_weights[walker][move] = bare_weights[walker][move] * expected_ratios[walker][move];
    expected_weighted[walker].resize(active.size());
    expected_weighted[walker] = Value(0.25 * (walker + 1));
    for (std::size_t move = 0; move < displacements[walker].size(); ++move)
      for (std::size_t parameter = 0; parameter < active.size(); ++parameter)
        expected_weighted[walker][parameter] +=
            total_weights[walker][move] * materialized_derivatives(move, parameter);
  }
  RefVector<const std::vector<Value>> weight_views;
  weight_views.push_back(std::cref(total_weights[0]));
  weight_views.push_back(std::cref(total_weights[1]));

  constexpr std::size_t pass_count = 2;
  std::array<std::vector<PsiFormerWF::LogValue>, pass_count> move_log_ratios;
  std::array<std::vector<ParticleSet::ParticleGradient>, pass_count> move_gradients;
  std::array<std::vector<ParticleSet::ParticleLaplacian>, pass_count> move_laplacians;
  std::array<std::vector<std::vector<Value>>, pass_count> virtual_ratios;
  std::array<std::array<Vector<Value>, crowd_size>, pass_count> virtual_weighted;
  for (std::size_t pass = 0; pass < pass_count; ++pass)
  {
    move_log_ratios[pass].resize(crowd_size);
    move_gradients[pass].resize(crowd_size);
    move_laplacians[pass].resize(crowd_size);
    virtual_ratios[pass].resize(crowd_size);
    for (std::size_t walker = 0; walker < crowd_size; ++walker)
    {
      move_gradients[pass][walker].resize(walkers[walker]->getTotalNum());
      move_laplacians[pass][walker].resize(walkers[walker]->getTotalNum());
      move_gradients[pass][walker]  = Value(0);
      move_laplacians[pass][walker] = Value(0);
      virtual_ratios[pass][walker].resize(displacements[walker].size());
      virtual_weighted[pass][walker].resize(active.size());
      virtual_weighted[pass][walker] = Value(0.25 * (walker + 1));
    }
  }

  auto run_move_pass = [&](std::size_t pass) {
    RefVector<ParticleSet::ParticleGradient> gradient_views;
    RefVector<ParticleSet::ParticleLaplacian> laplacian_views;
    for (std::size_t walker = 0; walker < crowd_size; ++walker)
    {
      gradient_views.push_back(move_gradients[pass][walker]);
      laplacian_views.push_back(move_laplacians[pass][walker]);
    }
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(move_resource, move_components);
    move_leader.mw_evaluateMultiParticleMove(
        move_components, move_particles, moves, move_log_ratios[pass],
        gradient_views, laplacian_views);
    move_leader.mw_accept_rejectMultiParticleMove(
        move_components, move_particles, moves, std::vector<bool>(crowd_size, false));
  };

  auto run_virtual_pass = [&](std::size_t pass) {
    std::vector<WaveFunctionComponent::ParameterDerivativeView> derivative_views;
    for (std::size_t walker = 0; walker < crowd_size; ++walker)
      derivative_views.push_back(
          {virtual_weighted[pass][walker].data(),
           static_cast<std::size_t>(virtual_weighted[pass][walker].size())});
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(virtual_resource, virtual_components);
    virtual_leader.mw_evaluateRatios(
        virtual_components, virtual_particles, virtual_ratios[pass]);
    virtual_leader.mw_evaluateDerivRatiosWeighted(
        virtual_components, virtual_particles, active, weight_views, derivative_views);
  };

  std::array<std::exception_ptr, thread_count> failures{};
  std::atomic<int> ready{0};
  PsiFormerThreadExecutor executor;
  executor(thread_count, [&](int task) {
    try
    {
      enterThreadedSection(ready);
      for (std::size_t pass = 0; pass < pass_count; ++pass)
      {
        if (task == 0)
          run_move_pass(pass);
        else
          run_virtual_pass(pass);
      }
    }
    catch (...)
    {
      failures[task] = std::current_exception();
    }
  });
  REQUIRE_NOTHROW(rethrowThreadFailures(failures));

  for (std::size_t pass = 0; pass < pass_count; ++pass)
    for (std::size_t walker = 0; walker < crowd_size; ++walker)
    {
      checkThreadValue(move_log_ratios[pass][walker],
                       expected_proposed_logs[walker] - accepted_logs[walker], 3.0e-9);
      for (int electron = 0; electron < walkers[walker]->getTotalNum(); ++electron)
      {
        for (int dimension = 0; dimension < 3; ++dimension)
          checkThreadValue(move_gradients[pass][walker][electron][dimension],
                           expected_proposed_gradients[walker][electron][dimension], 3.0e-8);
        checkThreadValue(move_laplacians[pass][walker][electron],
                         expected_proposed_laplacians[walker][electron], 3.0e-7);
      }
      for (std::size_t move = 0; move < expected_ratios[walker].size(); ++move)
        checkThreadValue(virtual_ratios[pass][walker][move], expected_ratios[walker][move], 3.0e-9);
      for (std::size_t parameter = 0; parameter < active.size(); ++parameter)
        checkThreadValue(virtual_weighted[pass][walker][parameter],
                         expected_weighted[walker][parameter], 3.0e-8);
    }

  for (std::size_t walker = 0; walker < walker_count; ++walker)
    checkThreadValue(components[walker]->get_log_value(), accepted_logs[walker], 3.0e-9);
  for (std::size_t walker = 0; walker < crowd_size; ++walker)
    for (int electron = 0; electron < walkers[walker]->getTotalNum(); ++electron)
      CHECK(walkers[walker]->R[electron] == accepted_positions[walker][electron]);
}

} // namespace qmcplusplus
