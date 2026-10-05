//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_multiwalker.cpp
 * @brief Deterministic public-API tests for PsiFormer crowd and virtual batches.
 */
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Particle/ParticleSet.h"
#include "Particle/VirtualParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "ResourceCollection.h"
#include "psiformer_test_utils.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace qmcplusplus
{
namespace
{
using namespace testing::psiformer;
using Value = QMCTraits::ValueType;

std::unique_ptr<ParticleSet> makeWalker(const SimulationCell& simulation_cell, std::size_t walker)
{
  const Geometry geometry = makeGeometry("lih");
  auto particles = std::make_unique<ParticleSet>(simulation_cell);
  particles->setName("e" + std::to_string(walker));
  particles->create({2, 2});
  for (int electron = 0; electron < particles->getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      particles->R[electron][dimension] = geometry.electrons[3 * electron + dimension] +
          0.007 * static_cast<double>(walker) * static_cast<double>((electron + 1) * (dimension + 1));
  particles->update();
  return particles;
}

void checkValue(Value actual, Value expected, double tolerance = 3.0e-9)
{
  CHECK(std::real(actual) == Catch::Approx(std::real(expected)).epsilon(tolerance).margin(tolerance));
  CHECK(std::imag(actual) == Catch::Approx(std::imag(expected)).epsilon(tolerance).margin(tolerance));
}

void checkLog(PsiFormerWF::LogValue actual,
              PsiFormerWF::LogValue expected,
              double tolerance = 3.0e-9)
{
  CHECK(std::real(actual) == Catch::Approx(std::real(expected)).epsilon(tolerance).margin(tolerance));
  CHECK(std::imag(actual) == Catch::Approx(std::imag(expected)).epsilon(tolerance).margin(tolerance));
}

void checkGrad(const PsiFormerWF::GradType& actual,
               const PsiFormerWF::GradType& expected,
               double tolerance = 3.0e-8)
{
  for (int dimension = 0; dimension < 3; ++dimension)
    checkValue(actual[dimension], expected[dimension], tolerance);
}

struct Crowd
{
  Crowd(const GeneratedFiles& files, const SimulationCell& simulation_cell, std::size_t size)
      : leader("pf_mw", files.parameters.string(), files.configuration.string()),
        wfc_list(leader)
  {
    walkers.reserve(size);
    components.reserve(size);
    walkers.push_back(makeWalker(simulation_cell, 0));
    components.push_back(&leader);
    for (std::size_t walker = 1; walker < size; ++walker)
    {
      walkers.push_back(makeWalker(simulation_cell, walker));
      clone_storage.push_back(leader.makeClone(*walkers.back()));
      components.push_back(static_cast<PsiFormerWF*>(clone_storage.back().get()));
    }
    p_list = std::make_unique<RefVectorWithLeader<ParticleSet>>(*walkers.front());
    for (std::size_t walker = 0; walker < size; ++walker)
    {
      p_list->push_back(*walkers[walker]);
      wfc_list.push_back(*components[walker]);
    }
  }

  PsiFormerWF leader;
  std::vector<std::unique_ptr<ParticleSet>> walkers;
  std::vector<std::unique_ptr<WaveFunctionComponent>> clone_storage;
  std::vector<PsiFormerWF*> components;
  RefVectorWithLeader<WaveFunctionComponent> wfc_list;
  std::unique_ptr<RefVectorWithLeader<ParticleSet>> p_list;
};

} // namespace

TEST_CASE("PsiFormer crowd APIs match scalar paths for batches 1 2 and 4",
          "[wavefunction][psiformer][multiwalker]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;

  for (const std::size_t batch_size : {std::size_t{1}, std::size_t{2}, std::size_t{4}})
  {
    DYNAMIC_SECTION("batch size " << batch_size)
    {
      Crowd crowd(files, simulation_cell, batch_size);
      constexpr int moved_electron = 1;
      const std::size_t electrons = crowd.walkers.front()->getTotalNum();

      std::vector<ParticleSet::ParticleGradient> scalar_g(batch_size);
      std::vector<ParticleSet::ParticleLaplacian> scalar_l(batch_size);
      std::vector<PsiFormerWF::LogValue> scalar_log(batch_size);
      std::vector<ParticleSet::ParticleGradient> batch_g(batch_size);
      std::vector<ParticleSet::ParticleLaplacian> batch_l(batch_size);
      RefVector<ParticleSet::ParticleGradient> batch_g_list;
      RefVector<ParticleSet::ParticleLaplacian> batch_l_list;
      for (std::size_t walker = 0; walker < batch_size; ++walker)
      {
        scalar_g[walker].resize(electrons);
        scalar_l[walker].resize(electrons);
        batch_g[walker].resize(electrons);
        batch_l[walker].resize(electrons);
        scalar_g[walker] = Value(0.125 * (walker + 1));
        scalar_l[walker] = Value(-0.25 * (walker + 1));
        batch_g[walker] = scalar_g[walker];
        batch_l[walker] = scalar_l[walker];
        scalar_log[walker] = crowd.components[walker]->evaluateLog(
            *crowd.walkers[walker], scalar_g[walker], scalar_l[walker]);
        batch_g_list.push_back(batch_g[walker]);
        batch_l_list.push_back(batch_l[walker]);
      }

      ResourceCollection resource_template("psiformer_resource_template");
      crowd.leader.createResource(resource_template);
      ResourceCollection crowd_resource(resource_template);
      {
        ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource, crowd.wfc_list);
        crowd.leader.mw_evaluateLog(crowd.wfc_list, *crowd.p_list, batch_g_list, batch_l_list);

        for (std::size_t walker = 0; walker < batch_size; ++walker)
        {
          checkLog(crowd.components[walker]->get_log_value(), scalar_log[walker]);
          for (std::size_t electron = 0; electron < electrons; ++electron)
          {
            checkGrad(batch_g[walker][electron], scalar_g[walker][electron]);
            checkValue(batch_l[walker][electron], scalar_l[walker][electron], 3.0e-7);
          }
        }

        std::vector<PsiFormerWF::GradType> scalar_active(batch_size);
        std::vector<PsiFormerWF::GradType> batch_active(batch_size);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
          scalar_active[walker] = crowd.components[walker]->evalGrad(
              *crowd.walkers[walker], moved_electron);
        crowd.leader.mw_evalGrad(
            crowd.wfc_list, *crowd.p_list, moved_electron, batch_active);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
          checkGrad(batch_active[walker], scalar_active[walker]);

        for (std::size_t walker = 0; walker < batch_size; ++walker)
        {
          const ParticleSet::SingleParticlePos displacement{
              0.012 * (walker + 1), -0.009 * (walker + 1), 0.006 * (walker + 1)};
          crowd.walkers[walker]->makeMove(moved_electron, displacement);
        }

        std::vector<Value> scalar_ratios(batch_size);
        std::vector<Value> batch_ratios(batch_size);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
        {
          scalar_ratios[walker] = crowd.components[walker]->ratio(
              *crowd.walkers[walker], moved_electron);
          crowd.components[walker]->restore(moved_electron);
        }
        crowd.leader.mw_calcRatio(
            crowd.wfc_list, *crowd.p_list, moved_electron, batch_ratios);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
          checkValue(batch_ratios[walker], scalar_ratios[walker]);

        std::vector<PsiFormerWF::GradType> scalar_ratio_grads(batch_size);
        std::vector<PsiFormerWF::GradType> batch_ratio_grads(batch_size);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
        {
          const PsiFormerWF::GradType seed(Value(0.31 + walker), Value(-0.17), Value(0.23));
          scalar_ratio_grads[walker] = seed;
          batch_ratio_grads[walker] = seed;
          scalar_ratios[walker] = crowd.components[walker]->ratioGrad(
              *crowd.walkers[walker], moved_electron, scalar_ratio_grads[walker]);
          crowd.components[walker]->restore(moved_electron);
        }
        crowd.leader.mw_ratioGrad(crowd.wfc_list, *crowd.p_list, moved_electron,
                                  batch_ratios, batch_ratio_grads);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
        {
          checkValue(batch_ratios[walker], scalar_ratios[walker]);
          checkGrad(batch_ratio_grads[walker], scalar_ratio_grads[walker]);
        }

        std::vector<PsiFormerWF::LogValue> old_logs(batch_size);
        std::vector<PsiFormerWF::LogValue> proposed_logs(batch_size);
        std::vector<bool> accepted(batch_size);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
        {
          old_logs[walker] = scalar_log[walker];
          const double current_sign = std::abs(std::imag(old_logs[walker])) > 1.0 ? -1.0 : 1.0;
          const double ratio_sign = std::real(batch_ratios[walker]) < 0.0 ? -1.0 : 1.0;
          proposed_logs[walker] = PsiFormerWF::LogValue(
              std::real(old_logs[walker]) + std::log(std::abs(std::real(batch_ratios[walker]))),
              current_sign * ratio_sign < 0.0 ? M_PI : 0.0);
          accepted[walker] = walker % 2 == 0;
        }
        crowd.leader.mw_accept_rejectMove(
            crowd.wfc_list, *crowd.p_list, moved_electron, accepted, true);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
          checkLog(crowd.components[walker]->get_log_value(),
                   accepted[walker] ? proposed_logs[walker] : old_logs[walker]);
      }

      // The copied ResourceCollection owns independent scratch and release clears
      // the leader handle, so a direct crowd call outside a team lock fails early.
      std::vector<PsiFormerWF::GradType> gradients(batch_size);
      CHECK_THROWS_AS(crowd.leader.mw_evalGrad(
                          crowd.wfc_list, *crowd.p_list, moved_electron, gradients),
                      std::logic_error);
    }
  }
}

TEST_CASE("PsiFormer all-to-one and ragged virtual batches are state isolated",
          "[wavefunction][psiformer][multiwalker][ecp]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;

  // Scalar all-to-one has no framework multiwalker entry point, so it uses the
  // clone-local native batch workspace and must not populate proposal state.
  auto particles = makeWalker(simulation_cell, 0);
  PsiFormerWF all_to_one("pf_all_to_one", files.parameters.string(), files.configuration.string());
  ParticleSet::ParticleGradient gradient(particles->getTotalNum());
  ParticleSet::ParticleLaplacian laplacian(particles->getTotalNum());
  gradient = Value(0);
  laplacian = Value(0);
  const PsiFormerWF::LogValue reference_log =
      all_to_one.evaluateLog(*particles, gradient, laplacian);
  const ParticleSet::SingleParticlePos common_position{0.37, -0.22, 0.41};
  particles->makeVirtualMoves(common_position);
  std::vector<Value> all_ratios(particles->getTotalNum());
  all_to_one.evaluateRatiosAlltoOne(*particles, all_ratios);
  checkLog(all_to_one.get_log_value(), reference_log);
  all_to_one.acceptMove(*particles, 0);
  checkLog(all_to_one.get_log_value(), reference_log);

  PsiFormerWF all_to_one_oracle("pf_all_to_one_oracle", files.parameters.string(),
                                files.configuration.string());
  for (int electron = 0; electron < particles->getTotalNum(); ++electron)
  {
    auto moved = makeWalker(simulation_cell, 0);
    moved->R[electron] = common_position;
    moved->update();
    ParticleSet::ParticleGradient moved_g(moved->getTotalNum());
    ParticleSet::ParticleLaplacian moved_l(moved->getTotalNum());
    moved_g = Value(0);
    moved_l = Value(0);
    const auto moved_log = all_to_one_oracle.evaluateLog(*moved, moved_g, moved_l);
    checkValue(all_ratios[electron], Value(std::real(std::exp(moved_log - reference_log))));
  }

  Crowd crowd(files, simulation_cell, 4);
  std::vector<std::unique_ptr<VirtualParticleSet>> virtual_storage;
  std::vector<std::vector<ParticleSet::SingleParticlePos>> displacements(4);
  for (std::size_t walker = 0; walker < 4; ++walker)
  {
    for (std::size_t move = 0; move < walker + 1; ++move)
      displacements[walker].push_back(ParticleSet::SingleParticlePos{
          0.01 * (move + 1), -0.013 * (walker + 1), 0.008 * (move + walker + 1)});
    virtual_storage.push_back(std::make_unique<VirtualParticleSet>(*crowd.walkers[walker]));
    virtual_storage.back()->makeMoves(*crowd.walkers[walker], static_cast<int>(walker % 4),
                                      displacements[walker]);
  }

  RefVectorWithLeader<const VirtualParticleSet> virtual_list(*virtual_storage.front());
  std::vector<std::vector<Value>> expected(4);
  std::vector<std::vector<Value>> actual(4);
  std::vector<PsiFormerWF::LogValue> state_before(4);
  for (std::size_t walker = 0; walker < 4; ++walker)
  {
    virtual_list.push_back(*virtual_storage[walker]);
    expected[walker].resize(displacements[walker].size());
    actual[walker].resize(displacements[walker].size());
    ParticleSet::ParticleGradient g(crowd.walkers[walker]->getTotalNum());
    ParticleSet::ParticleLaplacian l(crowd.walkers[walker]->getTotalNum());
    g = Value(0);
    l = Value(0);
    state_before[walker] = crowd.components[walker]->evaluateLog(*crowd.walkers[walker], g, l);
    crowd.components[walker]->evaluateRatios(*virtual_storage[walker], expected[walker]);
  }

  ResourceCollection resource_template("psiformer_virtual_template");
  crowd.leader.createResource(resource_template);
  for (int resource_clone = 0; resource_clone < 2; ++resource_clone)
  {
    ResourceCollection crowd_resource(resource_template);
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource, crowd.wfc_list);
    crowd.leader.mw_evaluateRatios(crowd.wfc_list, virtual_list, actual);
    for (std::size_t walker = 0; walker < 4; ++walker)
    {
      REQUIRE(actual[walker].size() == expected[walker].size());
      for (std::size_t move = 0; move < actual[walker].size(); ++move)
        checkValue(actual[walker][move], expected[walker][move]);
      checkLog(crowd.components[walker]->get_log_value(), state_before[walker]);
    }
    RefVector<std::pair<WaveFunctionComponent::ValueVector, WaveFunctionComponent::ValueVector>>
        unused_spin_multipliers;
    CHECK_THROWS_AS(crowd.leader.mw_evaluateSpinorRatios(
                        crowd.wfc_list, virtual_list, unused_spin_multipliers, actual),
                    std::invalid_argument);
  }
}

} // namespace qmcplusplus
