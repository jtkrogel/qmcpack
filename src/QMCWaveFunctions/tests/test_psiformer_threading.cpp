//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_threading.cpp
 * @brief Concurrent-resource regression test for two PsiFormer crowds.
 */

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "ResourceCollection.h"
#include "psiformer_test_utils.h"

#include <complex>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace qmcplusplus
{
namespace
{
using namespace testing::psiformer;
using Value = QMCTraits::ValueType;

/// Construct one deterministically displaced LiH walker.
std::unique_ptr<ParticleSet> makeThreadWalker(const SimulationCell& simulation_cell,
                                              std::size_t walker)
{
  const Geometry geometry = makeGeometry("lih");
  auto particles          = std::make_unique<ParticleSet>(simulation_cell);
  particles->setName("thread_walker_" + std::to_string(walker));
  particles->create({2, 2});
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

} // namespace

TEST_CASE("PsiFormer independent crowd resources run concurrently",
          "[wavefunction][psiformer][multiwalker][threading]")
{
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
    expected_gradient[walker] = Value(0);
    expected_laplacian[walker] = Value(0);
    actual_gradient[walker] = Value(0);
    actual_laplacian[walker] = Value(0);
    expected_log[walker] = components[walker]->evaluateLog(
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

  ResourceCollection first_template("psiformer_first_thread_template");
  ResourceCollection second_template("psiformer_second_thread_template");
  first_leader.createResource(first_template);
  second_leader.createResource(second_template);
  ResourceCollection first_resource(first_template);
  ResourceCollection second_resource(second_template);

  std::promise<void> start_promise;
  const std::shared_future<void> start = start_promise.get_future().share();
  auto evaluate_first = std::async(std::launch::async, [&]() {
    start.wait();
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(first_resource, first_components);
    first_leader.mw_evaluateLog(
        first_components, first_particles, first_gradients, first_laplacians);
  });
  auto evaluate_second = std::async(std::launch::async, [&]() {
    start.wait();
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(second_resource, second_components);
    second_leader.mw_evaluateLog(
        second_components, second_particles, second_gradients, second_laplacians);
  });
  start_promise.set_value();
  REQUIRE_NOTHROW(evaluate_first.get());
  REQUIRE_NOTHROW(evaluate_second.get());

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

} // namespace qmcplusplus
