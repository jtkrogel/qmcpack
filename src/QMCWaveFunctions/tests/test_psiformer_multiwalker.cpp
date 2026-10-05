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

#include "Particle/MCMultiParticleMoves.h"
#include "Particle/ParticleSet.h"
#include "Particle/VirtualParticleBatch.h"
#include "Particle/VirtualParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "ResourceCollection.h"
#include "psiformer_test_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus
{
namespace testing
{
/** Exact clone-local state snapshot used to prove virtual evaluations are read-only. */
struct PsiFormerCloneStateSnapshot
{
  PsiFormerWF::LogValue log_value;
  std::size_t observed_parameter_version;
  bool restore_validation_pending;
  bool accepted_value_valid;
  ParticleSet::ParticleGradient accepted_gradient;
  ParticleSet::ParticleLaplacian accepted_laplacian;
  std::uint64_t accepted_configuration_identity;
  std::size_t accepted_parameter_version;
  std::uint64_t accepted_state_requirement;
  double current_sign;
  double proposed_sign;
  PsiFormerWF::LogValue proposed_log_value;
  ParticleSet::ParticleGradient proposed_gradient;
  ParticleSet::ParticleLaplacian proposed_laplacian;
  std::uint64_t proposed_configuration_identity;
  std::uint64_t proposed_descriptor_fingerprint;
  std::size_t proposed_parameter_version;
  int proposed_particle;
  std::uint64_t proposal_kind;
  bool has_proposal;
};

/** Narrow friend accessor for state-isolation and crowd-workspace diagnostics. */
class TestPsiFormerVirtualBatch
{
public:
  static PsiFormerCloneStateSnapshot cloneState(const PsiFormerWF& component)
  {
    return {component.log_value_,
            component.observed_parameter_version_,
            component.restore_validation_pending_,
            component.accepted_value_valid_,
            component.accepted_gradient_,
            component.accepted_laplacian_,
            component.accepted_configuration_identity_,
            component.accepted_parameter_version_,
            static_cast<std::uint64_t>(component.accepted_state_requirement_),
            component.current_sign_,
            component.proposed_sign_,
            component.proposed_log_value_,
            component.proposed_gradient_,
            component.proposed_laplacian_,
            component.proposed_configuration_identity_,
            component.proposed_descriptor_fingerprint_,
            component.proposed_parameter_version_,
            component.proposed_particle_,
            static_cast<std::uint64_t>(component.proposal_kind_),
            component.has_proposal_};
  }

  static bool cloneStateMatches(const PsiFormerWF& component,
                                const PsiFormerCloneStateSnapshot& snapshot)
  {
    if (component.log_value_ != snapshot.log_value ||
        component.observed_parameter_version_ != snapshot.observed_parameter_version ||
        component.restore_validation_pending_ != snapshot.restore_validation_pending ||
        component.accepted_value_valid_ != snapshot.accepted_value_valid ||
        component.accepted_configuration_identity_ != snapshot.accepted_configuration_identity ||
        component.accepted_parameter_version_ != snapshot.accepted_parameter_version ||
        static_cast<std::uint64_t>(component.accepted_state_requirement_) !=
            snapshot.accepted_state_requirement ||
        component.current_sign_ != snapshot.current_sign ||
        component.proposed_sign_ != snapshot.proposed_sign ||
        component.proposed_log_value_ != snapshot.proposed_log_value ||
        component.proposed_configuration_identity_ != snapshot.proposed_configuration_identity ||
        component.proposed_descriptor_fingerprint_ != snapshot.proposed_descriptor_fingerprint ||
        component.proposed_parameter_version_ != snapshot.proposed_parameter_version ||
        component.proposed_particle_ != snapshot.proposed_particle ||
        static_cast<std::uint64_t>(component.proposal_kind_) != snapshot.proposal_kind ||
        component.has_proposal_ != snapshot.has_proposal)
      return false;

    return sameGradient(component.accepted_gradient_, snapshot.accepted_gradient) &&
        sameLaplacian(component.accepted_laplacian_, snapshot.accepted_laplacian) &&
        sameGradient(component.proposed_gradient_, snapshot.proposed_gradient) &&
        sameLaplacian(component.proposed_laplacian_, snapshot.proposed_laplacian);
  }

  static PsiFormerCrowdWorkspaceDiagnostics crowdWorkspaceDiagnostics(
      const PsiFormerWF& component,
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list)
  {
    return component.crowdWorkspaceDiagnosticsForTesting(wfc_list);
  }

private:
  static bool sameGradient(const ParticleSet::ParticleGradient& actual,
                           const ParticleSet::ParticleGradient& expected)
  {
    if (actual.size() != expected.size())
      return false;
    for (std::size_t electron = 0; electron < actual.size(); ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        if (actual[electron][dimension] != expected[electron][dimension])
          return false;
    return true;
  }

  static bool sameLaplacian(const ParticleSet::ParticleLaplacian& actual,
                            const ParticleSet::ParticleLaplacian& expected)
  {
    if (actual.size() != expected.size())
      return false;
    for (std::size_t electron = 0; electron < actual.size(); ++electron)
      if (actual[electron] != expected[electron])
        return false;
    return true;
  }
};
} // namespace testing

namespace
{
using namespace testing::psiformer;
using Value = QMCTraits::ValueType;

class ScopedEnvironmentVariable
{
public:
  ScopedEnvironmentVariable(std::string name, const char* value) : name_(std::move(name))
  {
    if (const char* previous = std::getenv(name_.c_str()))
      previous_ = previous;
    if (setenv(name_.c_str(), value, 1) != 0)
      throw std::runtime_error("Unable to set PsiFormer test environment variable");
  }

  ~ScopedEnvironmentVariable()
  {
    if (previous_)
      setenv(name_.c_str(), previous_->c_str(), 1);
    else
      unsetenv(name_.c_str());
  }

private:
  std::string name_;
  std::optional<std::string> previous_;
};

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

/// Own one compatibility VirtualParticleSet scratch object per reference walker.
struct VirtualScratchCrowd
{
  explicit VirtualScratchCrowd(const Crowd& crowd)
  {
    storage.reserve(crowd.walkers.size());
    for (const auto& walker : crowd.walkers)
      storage.push_back(std::make_unique<VirtualParticleSet>(*walker));
    list = std::make_unique<RefVectorWithLeader<VirtualParticleSet>>(*storage.front());
    for (const auto& scratch : storage)
      list->push_back(*scratch);
  }

  std::vector<std::unique_ptr<VirtualParticleSet>> storage;
  std::unique_ptr<RefVectorWithLeader<VirtualParticleSet>> list;
};

/// Append one off-sphere segment using deterministic displacements from its reference electron.
void appendVirtualSegment(
    const Crowd& crowd,
    std::size_t walker,
    int electron,
    std::initializer_list<ParticleSet::PosType> displacements,
    std::vector<std::size_t>& offsets,
    std::vector<VirtualParticleBatch::Segment>& segments,
    std::vector<ParticleSet::PosType>& positions)
{
  segments.emplace_back(static_cast<int>(walker), electron);
  for (const ParticleSet::PosType& displacement : displacements)
    positions.push_back(crowd.walkers[walker]->R[electron] + displacement);
  offsets.push_back(positions.size());
}

/// Evaluate the unchanged scalar virtual interface segment by segment as an independent oracle.
std::vector<Value> evaluateScalarVirtualBatch(Crowd& crowd,
                                              const VirtualParticleBatch& batch)
{
  std::vector<Value> ratios(batch.size());
  for (std::size_t segment_index = 0; segment_index < batch.segmentCount();
       ++segment_index)
  {
    const VirtualParticleBatch::Slice slice = batch.slice(segment_index);
    const std::size_t walker = static_cast<std::size_t>(slice.walkerId());
    VirtualParticleSet scratch(*crowd.walkers[walker]);
    scratch.makeMovesAbsolute(*crowd.walkers[walker], slice.electronId(),
                              slice.positions(), slice.isOnSphere(),
                              slice.sourceCenterId());
    std::vector<Value> segment_ratios(slice.size());
    crowd.components[walker]->evaluateRatios(scratch, segment_ratios);
    std::copy(segment_ratios.begin(), segment_ratios.end(),
              ratios.begin() + slice.flatOffset());
  }
  return ratios;
}

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

TEST_CASE("PsiFormer flattened virtual batches share sparse references and preserve state",
          "[wavefunction][psiformer][multiwalker][ecp][sparse]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  Crowd crowd(files, simulation_cell, 4);

  // Deliberately order segments independently of walker order, name walker 2
  // twice with different electrons, and leave walker 3 empty.
  std::vector<std::size_t> offsets{0};
  std::vector<VirtualParticleBatch::Segment> segments;
  std::vector<ParticleSet::PosType> positions;
  appendVirtualSegment(crowd, 2, 3,
                       {{0.012, -0.007, 0.005}, {-0.009, 0.011, 0.004}},
                       offsets, segments, positions);
  appendVirtualSegment(crowd, 0, 1, {{0.006, 0.003, -0.008}},
                       offsets, segments, positions);
  appendVirtualSegment(crowd, 2, 0,
                       {{-0.004, 0.008, 0.013}, {0.015, -0.006, -0.002}},
                       offsets, segments, positions);
  appendVirtualSegment(crowd, 1, 2,
                       {{0.007, -0.014, 0.009}, {-0.011, 0.005, 0.012}},
                       offsets, segments, positions);
  const VirtualParticleBatch batch(crowd.walkers.size(), offsets, segments,
                                   positions);
  const std::vector<Value> expected = evaluateScalarVirtualBatch(crowd, batch);
  VirtualScratchCrowd scratch(crowd);

  // Populate complete accepted caches, then leave one ordinary proposal live.
  for (std::size_t walker = 0; walker < crowd.walkers.size(); ++walker)
  {
    crowd.walkers[walker]->G = Value(0);
    crowd.walkers[walker]->L = Value(0);
    crowd.components[walker]->evaluateLog(
        *crowd.walkers[walker], crowd.walkers[walker]->G,
        crowd.walkers[walker]->L);
  }
  constexpr int proposed_electron = 1;
  crowd.walkers[2]->makeMove(
      proposed_electron, ParticleSet::PosType{0.003, -0.005, 0.007});
  const Value pending_ratio = crowd.components[2]->ratio(
      *crowd.walkers[2], proposed_electron);
  CHECK(std::isfinite(std::real(pending_ratio)));

  std::vector<testing::PsiFormerCloneStateSnapshot> states_before;
  for (const PsiFormerWF* component : crowd.components)
    states_before.push_back(testing::TestPsiFormerVirtualBatch::cloneState(*component));

  ResourceCollection resource_template("psiformer_flattened_virtual_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection crowd_resource(resource_template);
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource,
                                                            crowd.wfc_list);

    // Shape failure is detected before resource or component state publication.
    std::vector<Value> wrong_extent(batch.size() - 1, Value(-17));
    CHECK_THROWS_AS(crowd.leader.mw_evaluateVirtualRatios(
                        crowd.wfc_list, *crowd.p_list, *scratch.list, batch,
                        wrong_extent),
                    std::invalid_argument);
    CHECK(std::all_of(wrong_extent.begin(), wrong_extent.end(),
                      [](Value value) { return value == Value(-17); }));

    std::vector<Value> actual(batch.size(), Value(-23));
    const WaveFunctionComponent::EvaluationStamp first_stamp =
        crowd.leader.mw_evaluateVirtualRatios(
            crowd.wfc_list, *crowd.p_list, *scratch.list, batch, actual);
    REQUIRE(first_stamp.isVersioned());
    for (std::size_t virtual_index = 0; virtual_index < batch.size();
         ++virtual_index)
      checkValue(actual[virtual_index], expected[virtual_index]);

    const testing::PsiFormerCrowdWorkspaceDiagnostics first_diagnostics =
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            crowd.leader, crowd.wfc_list);
    CHECK(first_diagnostics.reference_configurations == 3);
    CHECK(first_diagnostics.replacement_configurations == batch.size());
    CHECK(first_diagnostics.reference_evaluations == 3);
    CHECK(first_diagnostics.dense_coordinate_bytes_avoided ==
          batch.size() * (crowd.walkers.front()->getTotalNum() - 1) * 3 *
              sizeof(double));

    std::fill(actual.begin(), actual.end(), Value(-31));
    const WaveFunctionComponent::EvaluationStamp repeated_stamp =
        crowd.leader.mw_evaluateVirtualRatios(
            crowd.wfc_list, *crowd.p_list, *scratch.list, batch, actual);
    CHECK(repeated_stamp == first_stamp);
    for (std::size_t virtual_index = 0; virtual_index < batch.size();
         ++virtual_index)
      checkValue(actual[virtual_index], expected[virtual_index]);

    const testing::PsiFormerCrowdWorkspaceDiagnostics repeated_diagnostics =
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            crowd.leader, crowd.wfc_list);
    CHECK(repeated_diagnostics.batch_workspace_identity ==
          first_diagnostics.batch_workspace_identity);
    CHECK(repeated_diagnostics.batch_bytes == first_diagnostics.batch_bytes);
    CHECK(repeated_diagnostics.transient_bytes ==
          first_diagnostics.transient_bytes);

    // Empty work still reports the model version needed by an outer tiled caller.
    const std::vector<std::size_t> empty_offsets{0};
    const std::vector<VirtualParticleBatch::Segment> empty_segments;
    const std::vector<ParticleSet::PosType> empty_positions;
    const VirtualParticleBatch empty_batch(
        crowd.walkers.size(), empty_offsets, empty_segments, empty_positions);
    std::vector<Value> empty_ratios;
    const WaveFunctionComponent::EvaluationStamp empty_stamp =
        crowd.leader.mw_evaluateVirtualRatios(
            crowd.wfc_list, *crowd.p_list, *scratch.list, empty_batch,
            empty_ratios);
    CHECK(empty_stamp == first_stamp);
    CHECK(empty_ratios.empty());

    // Same-version flattened evaluation must not consume or overwrite accepted
    // values, VGL products, or the deliberately pending proposal.
    for (std::size_t walker = 0; walker < crowd.components.size(); ++walker)
      CHECK(testing::TestPsiFormerVirtualBatch::cloneStateMatches(
          *crowd.components[walker], states_before[walker]));

    // A genuine publication changes the opaque stamp.  The following sparse
    // call performs the established lazy invalidation of stale clone caches.
    wftrain::StructuredParameterSnapshot candidate =
        crowd.leader.snapshotParameters();
    candidate.values.at(127) += 1.0e-4;
    crowd.leader.publishParameters(candidate, candidate.version);
    const WaveFunctionComponent::EvaluationStamp changed_stamp =
        crowd.leader.mw_evaluateVirtualRatios(
            crowd.wfc_list, *crowd.p_list, *scratch.list, batch, actual);
    CHECK(changed_stamp != first_stamp);
    for (Value value : actual)
      CHECK(std::isfinite(std::real(value)));
    const WaveFunctionComponent::EvaluationStamp stable_changed_stamp =
        crowd.leader.mw_evaluateVirtualRatios(
            crowd.wfc_list, *crowd.p_list, *scratch.list, batch, actual);
    CHECK(stable_changed_stamp == changed_stamp);
  }

  crowd.walkers[2]->rejectMove(proposed_electron);
}

TEST_CASE("PsiFormer flattened virtual batches publish atomically after ratio failure",
          "[wavefunction][psiformer][multiwalker][ecp][sparse]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  Crowd crowd(files, simulation_cell, 2);
  for (std::size_t walker = 0; walker < crowd.walkers.size(); ++walker)
  {
    crowd.walkers[walker]->G = Value(0);
    crowd.walkers[walker]->L = Value(0);
    crowd.components[walker]->evaluateLog(
        *crowd.walkers[walker], crowd.walkers[walker]->G,
        crowd.walkers[walker]->L);
  }

  const ParticleSet::PosType saved_position = crowd.walkers[1]->R[1];
  crowd.walkers[1]->R[1] = crowd.walkers[1]->R[0];
  crowd.walkers[1]->update();

  std::vector<std::size_t> offsets{0};
  std::vector<VirtualParticleBatch::Segment> segments;
  std::vector<ParticleSet::PosType> positions;
  appendVirtualSegment(crowd, 1, 0, {{0.09, -0.04, 0.03}}, offsets,
                       segments, positions);
  const VirtualParticleBatch batch(crowd.walkers.size(), offsets, segments,
                                   positions);
  VirtualScratchCrowd scratch(crowd);

  std::vector<testing::PsiFormerCloneStateSnapshot> states_before;
  for (const PsiFormerWF* component : crowd.components)
    states_before.push_back(testing::TestPsiFormerVirtualBatch::cloneState(*component));

  ResourceCollection resource_template("psiformer_flattened_failure_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection crowd_resource(resource_template);
  ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource,
                                                          crowd.wfc_list);

  // The exact same-spin reference node is accepted by the value evaluator;
  // forming a finite moved/reference ratio then fails after all sparse outputs exist.
  std::vector<Value> ratios(batch.size(), Value(-41));
  CHECK_THROWS_AS(crowd.leader.mw_evaluateVirtualRatios(
                      crowd.wfc_list, *crowd.p_list, *scratch.list, batch,
                      ratios),
                  std::runtime_error);
  CHECK(ratios.front() == Value(-41));
  for (std::size_t walker = 0; walker < crowd.components.size(); ++walker)
    CHECK(testing::TestPsiFormerVirtualBatch::cloneStateMatches(
        *crowd.components[walker], states_before[walker]));

  // Correcting the reference makes the same resource immediately reusable.
  crowd.walkers[1]->R[1] = saved_position;
  crowd.walkers[1]->update();
  const std::vector<Value> expected = evaluateScalarVirtualBatch(crowd, batch);
  const WaveFunctionComponent::EvaluationStamp retry_stamp =
      crowd.leader.mw_evaluateVirtualRatios(
          crowd.wfc_list, *crowd.p_list, *scratch.list, batch, ratios);
  CHECK(retry_stamp.isVersioned());
  checkValue(ratios.front(), expected.front());
  for (std::size_t walker = 0; walker < crowd.components.size(); ++walker)
    CHECK(testing::TestPsiFormerVirtualBatch::cloneStateMatches(
        *crowd.components[walker], states_before[walker]));
}

TEST_CASE("PsiFormer flattened virtual batches honor oracle and compare backends",
          "[wavefunction][psiformer][multiwalker][ecp][sparse]")
{
  const SimulationCell simulation_cell;
  for (const char* backend : {"oracle", "compare"})
  {
    DYNAMIC_SECTION("backend " << backend)
    {
      ScopedEnvironmentVariable backend_mode("PSIFORMER_VALUE_BACKEND", backend);
      GeneratedFiles files = generateFiles("lih");
      Crowd crowd(files, simulation_cell, 2);
      std::vector<std::size_t> offsets{0};
      std::vector<VirtualParticleBatch::Segment> segments;
      std::vector<ParticleSet::PosType> positions;
      appendVirtualSegment(crowd, 1, 2,
                           {{0.007, -0.006, 0.005}, {-0.004, 0.009, 0.003}},
                           offsets, segments, positions);
      appendVirtualSegment(crowd, 0, 0, {{0.011, 0.002, -0.008}},
                           offsets, segments, positions);
      const VirtualParticleBatch batch(crowd.walkers.size(), offsets, segments,
                                       positions);
      const std::vector<Value> expected = evaluateScalarVirtualBatch(crowd, batch);
      VirtualScratchCrowd scratch(crowd);

      std::vector<testing::PsiFormerCloneStateSnapshot> states_before;
      for (const PsiFormerWF* component : crowd.components)
        states_before.push_back(testing::TestPsiFormerVirtualBatch::cloneState(*component));

      ResourceCollection resource_template("psiformer_flattened_backend_template");
      crowd.leader.createResource(resource_template);
      ResourceCollection crowd_resource(resource_template);
      ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource,
                                                              crowd.wfc_list);
      std::vector<Value> actual(batch.size(), Value(-53));
      const WaveFunctionComponent::EvaluationStamp stamp =
          crowd.leader.mw_evaluateVirtualRatios(
              crowd.wfc_list, *crowd.p_list, *scratch.list, batch, actual);
      CHECK(stamp.isVersioned());
      for (std::size_t virtual_index = 0; virtual_index < batch.size();
           ++virtual_index)
        checkValue(actual[virtual_index], expected[virtual_index]);
      for (std::size_t walker = 0; walker < crowd.components.size(); ++walker)
        CHECK(testing::TestPsiFormerVirtualBatch::cloneStateMatches(
            *crowd.components[walker], states_before[walker]));

      const testing::PsiFormerCrowdWorkspaceDiagnostics diagnostics =
          testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
              crowd.leader, crowd.wfc_list);
      if (std::string(backend) == "compare")
      {
        CHECK(diagnostics.reference_configurations == 2);
        CHECK(diagnostics.replacement_configurations == batch.size());
        CHECK(diagnostics.reference_evaluations == 2);
      }
      else
      {
        CHECK(diagnostics.reference_configurations == 0);
        CHECK(diagnostics.replacement_configurations == 0);
        CHECK(diagnostics.reference_evaluations == 0);
      }
    }
  }
}

TEST_CASE("PsiFormer selected-electron proposals are atomic full-VGL transactions",
          "[wavefunction][psiformer][multiwalker][multiparticle]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  constexpr std::size_t walker_count = 3;
  Crowd crowd(files, simulation_cell, walker_count);
  const std::size_t electron_count = crowd.walkers.front()->getTotalNum();

  std::vector<ParticleSet::ParticleGradient> initial_gradient(walker_count);
  std::vector<ParticleSet::ParticleLaplacian> initial_laplacian(walker_count);
  std::vector<PsiFormerWF::LogValue> initial_log(walker_count);
  std::vector<std::vector<ParticleSet::PosType>> initial_positions(walker_count);
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    initial_gradient[walker].resize(electron_count);
    initial_laplacian[walker].resize(electron_count);
    initial_gradient[walker]  = Value(0);
    initial_laplacian[walker] = Value(0);
    initial_log[walker] = crowd.components[walker]->evaluateLog(
        *crowd.walkers[walker], initial_gradient[walker], initial_laplacian[walker]);
    initial_positions[walker].assign(crowd.walkers[walker]->R.begin(),
                                     crowd.walkers[walker]->R.end());
  }

  using Moves = MCMultiParticleMoves<CoordsType::POS>;
  const std::vector<std::size_t> offsets{0, 2, 3, 5};
  const std::vector<Moves::IndexType> indices{0, 2, 1, 0, 3};
  std::vector<Moves::PosType> positions{
      crowd.walkers[0]->R[0] + Moves::PosType{0.021, -0.014, 0.009},
      crowd.walkers[0]->R[2] + Moves::PosType{-0.017, 0.011, 0.006},
      // An exact no-op replacement exercises safe reuse of accepted full VGL state.
      crowd.walkers[1]->R[1],
      crowd.walkers[2]->R[0] + Moves::PosType{0.013, 0.019, -0.008},
      crowd.walkers[2]->R[3] + Moves::PosType{-0.015, 0.007, 0.012}};
  const Moves moves(offsets, indices, positions);

  std::vector<ParticleSet::ParticleGradient> expected_gradient(walker_count);
  std::vector<ParticleSet::ParticleLaplacian> expected_laplacian(walker_count);
  std::vector<PsiFormerWF::LogValue> expected_log(walker_count);
  PsiFormerWF oracle("pf_selected_oracle", files.parameters.string(), files.configuration.string());
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    auto proposed = makeWalker(simulation_cell, walker);
    const auto slice = moves.slice(walker);
    for (std::size_t selected = 0; selected < slice.size(); ++selected)
      proposed->R[slice.particleIndex(selected)] = slice.proposedPosition(selected);
    proposed->update();
    expected_gradient[walker].resize(electron_count);
    expected_laplacian[walker].resize(electron_count);
    expected_gradient[walker]  = Value(0);
    expected_laplacian[walker] = Value(0);
    expected_log[walker] = oracle.evaluateLog(
        *proposed, expected_gradient[walker], expected_laplacian[walker]);
  }

  const Value gradient_seed(0.125);
  const Value laplacian_seed(-0.375);
  std::vector<ParticleSet::ParticleGradient> proposed_gradient(walker_count);
  std::vector<ParticleSet::ParticleLaplacian> proposed_laplacian(walker_count);
  RefVector<ParticleSet::ParticleGradient> proposed_gradient_list;
  RefVector<ParticleSet::ParticleLaplacian> proposed_laplacian_list;
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    proposed_gradient[walker].resize(electron_count);
    proposed_laplacian[walker].resize(electron_count);
    proposed_gradient[walker]  = gradient_seed;
    proposed_laplacian[walker] = laplacian_seed;
    proposed_gradient_list.push_back(proposed_gradient[walker]);
    proposed_laplacian_list.push_back(proposed_laplacian[walker]);
  }
  std::vector<PsiFormerWF::LogValue> log_ratios(walker_count, PsiFormerWF::LogValue(19.0));

  ResourceCollection wf_template("psiformer_selected_template");
  crowd.leader.createResource(wf_template);
  ResourceCollection wf_resources(wf_template);
  ResourceCollection particle_resources("psiformer_selected_particles");
  crowd.walkers.front()->createResource(particle_resources);
  CHECK_THROWS_AS(crowd.leader.mw_evaluateMultiParticleMove(
                      crowd.wfc_list, *crowd.p_list, moves, log_ratios,
                      proposed_gradient_list, proposed_laplacian_list),
                  std::logic_error);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, *crowd.p_list);
  ResourceCollectionTeamLock<WaveFunctionComponent> wf_lock(wf_resources, crowd.wfc_list);

  REQUIRE(crowd.leader.supportsMultiParticleMoves());
  proposed_laplacian.back().resize(electron_count - 1);
  CHECK_THROWS_AS(crowd.leader.mw_evaluateMultiParticleMove(
                      crowd.wfc_list, *crowd.p_list, moves, log_ratios,
                      proposed_gradient_list, proposed_laplacian_list),
                  std::invalid_argument);
  proposed_laplacian.back().resize(electron_count);
  proposed_laplacian.back() = laplacian_seed;
  crowd.leader.mw_evaluateMultiParticleMove(
      crowd.wfc_list, *crowd.p_list, moves, log_ratios,
      proposed_gradient_list, proposed_laplacian_list);

  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    // Evaluation consumes descriptor-owned absolute coordinates and leaves P accepted.
    for (std::size_t electron = 0; electron < electron_count; ++electron)
      for (int dimension = 0; dimension < 3; ++dimension)
        CHECK(crowd.walkers[walker]->R[electron][dimension] ==
              initial_positions[walker][electron][dimension]);
    checkLog(crowd.components[walker]->get_log_value(), initial_log[walker]);
    checkLog(log_ratios[walker], expected_log[walker] - initial_log[walker]);
    for (std::size_t electron = 0; electron < electron_count; ++electron)
    {
      for (int dimension = 0; dimension < 3; ++dimension)
        checkValue(proposed_gradient[walker][electron][dimension],
                   gradient_seed + expected_gradient[walker][electron][dimension], 3.0e-8);
      checkValue(proposed_laplacian[walker][electron],
                 laplacian_seed + expected_laplacian[walker][electron], 3.0e-7);
    }
  }

  // A different descriptor cannot consume the pending proposal, and failure is
  // crowd-atomic so the original transaction remains resolvable.
  std::vector<Moves::PosType> mismatched_positions = positions;
  mismatched_positions.back()[0] += 1.0e-4;
  const Moves mismatched_moves(offsets, indices, std::move(mismatched_positions));
  CHECK_THROWS_AS(crowd.leader.mw_accept_rejectMultiParticleMove(
                      crowd.wfc_list, *crowd.p_list, mismatched_moves,
                      std::vector<bool>(walker_count, false)),
                  std::logic_error);
  PsiFormerWF::WFBufferType pending_buffer;
  CHECK_THROWS_AS(crowd.components.front()->registerData(
                      *crowd.walkers.front(), pending_buffer),
                  std::logic_error);

  std::vector<bool> valid;
  ParticleSet::mw_makeMoveSelectedParticles(*crowd.p_list, moves, valid);
  CHECK(std::all_of(valid.begin(), valid.end(), [](bool value) { return value; }));
  const std::vector<bool> accepted{true, false, true};
  crowd.leader.mw_accept_rejectMultiParticleMove(
      crowd.wfc_list, *crowd.p_list, moves, accepted);
  ParticleSet::mw_accept_rejectMoveSelectedParticles(*crowd.p_list, accepted);

  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    const auto& final_gradient = accepted[walker] ? expected_gradient[walker] : initial_gradient[walker];
    const auto& final_laplacian = accepted[walker] ? expected_laplacian[walker] : initial_laplacian[walker];
    checkLog(crowd.components[walker]->get_log_value(),
             accepted[walker] ? expected_log[walker] : initial_log[walker]);

    // updateBuffer(false) must be able to reuse the promoted complete spatial cache.
    PsiFormerWF::WFBufferType buffer;
    crowd.components[walker]->registerData(*crowd.walkers[walker], buffer);
    buffer.allocate();
    buffer.rewind();
    crowd.walkers[walker]->G = Value(0);
    crowd.walkers[walker]->L = Value(0);
    checkLog(crowd.components[walker]->updateBuffer(
                 *crowd.walkers[walker], buffer, false),
             accepted[walker] ? expected_log[walker] : initial_log[walker]);
    for (std::size_t electron = 0; electron < electron_count; ++electron)
    {
      checkGrad(crowd.walkers[walker]->G[electron], final_gradient[electron]);
      checkValue(crowd.walkers[walker]->L[electron], final_laplacian[electron], 3.0e-7);
    }
  }
}

TEST_CASE("PsiFormer selected-electron proposals honor oracle and compare backends",
          "[wavefunction][psiformer][multiwalker][multiparticle][threading]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  constexpr std::size_t walker_count = 3;

  for (const char* backend : {"oracle", "compare"})
  {
    DYNAMIC_SECTION("spatial backend " << backend)
    {
      ScopedEnvironmentVariable backend_mode("PSIFORMER_SPATIAL_BACKEND", backend);
      Crowd crowd(files, simulation_cell, walker_count);
      const std::size_t electron_count = crowd.walkers.front()->getTotalNum();

      std::vector<ParticleSet::ParticleGradient> accepted_gradient(walker_count);
      std::vector<ParticleSet::ParticleLaplacian> accepted_laplacian(walker_count);
      std::vector<PsiFormerWF::LogValue> accepted_log(walker_count);
      for (std::size_t walker = 0; walker < walker_count; ++walker)
      {
        accepted_gradient[walker].resize(electron_count);
        accepted_laplacian[walker].resize(electron_count);
        accepted_gradient[walker]  = Value(0);
        accepted_laplacian[walker] = Value(0);
        accepted_log[walker] = crowd.components[walker]->evaluateLog(
            *crowd.walkers[walker], accepted_gradient[walker],
            accepted_laplacian[walker]);
      }

      using Moves = MCMultiParticleMoves<CoordsType::POS>;
      const std::vector<std::size_t> offsets{0, 1, 2, 4};
      const std::vector<Moves::IndexType> indices{0, 1, 0, 3};
      const std::vector<Moves::PosType> positions{
          crowd.walkers[0]->R[0] + Moves::PosType{0.013, -0.008, 0.005},
          // Exact replacement keeps one row on the accepted full-VGL reuse path.
          crowd.walkers[1]->R[1],
          crowd.walkers[2]->R[0] + Moves::PosType{-0.009, 0.012, 0.004},
          crowd.walkers[2]->R[3] + Moves::PosType{0.007, -0.006, 0.011}};
      const Moves moves(offsets, indices, positions);

      std::vector<ParticleSet::ParticleGradient> expected_gradient(walker_count);
      std::vector<ParticleSet::ParticleLaplacian> expected_laplacian(walker_count);
      std::vector<PsiFormerWF::LogValue> expected_log(walker_count);
      PsiFormerWF scalar("pf_selected_backend_scalar", files.parameters.string(),
                         files.configuration.string());
      for (std::size_t walker = 0; walker < walker_count; ++walker)
      {
        auto proposed = makeWalker(simulation_cell, walker);
        const auto selected = moves.slice(walker);
        for (std::size_t move = 0; move < selected.size(); ++move)
          proposed->R[selected.particleIndex(move)] = selected.proposedPosition(move);
        proposed->update();
        expected_gradient[walker].resize(electron_count);
        expected_laplacian[walker].resize(electron_count);
        expected_gradient[walker]  = Value(0);
        expected_laplacian[walker] = Value(0);
        expected_log[walker] = scalar.evaluateLog(
            *proposed, expected_gradient[walker], expected_laplacian[walker]);
      }

      std::vector<ParticleSet::ParticleGradient> proposed_gradient(walker_count);
      std::vector<ParticleSet::ParticleLaplacian> proposed_laplacian(walker_count);
      RefVector<ParticleSet::ParticleGradient> proposed_gradient_list;
      RefVector<ParticleSet::ParticleLaplacian> proposed_laplacian_list;
      for (std::size_t walker = 0; walker < walker_count; ++walker)
      {
        proposed_gradient[walker].resize(electron_count);
        proposed_laplacian[walker].resize(electron_count);
        proposed_gradient[walker]  = Value(0);
        proposed_laplacian[walker] = Value(0);
        proposed_gradient_list.push_back(proposed_gradient[walker]);
        proposed_laplacian_list.push_back(proposed_laplacian[walker]);
      }
      std::vector<PsiFormerWF::LogValue> log_ratios(walker_count);

      ResourceCollection resource_template("psiformer_selected_backend_template");
      crowd.leader.createResource(resource_template);
      ResourceCollection crowd_resource(resource_template);
      ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource,
                                                              crowd.wfc_list);
      crowd.leader.mw_evaluateMultiParticleMove(
          crowd.wfc_list, *crowd.p_list, moves, log_ratios,
          proposed_gradient_list, proposed_laplacian_list);

      for (std::size_t walker = 0; walker < walker_count; ++walker)
      {
        checkLog(log_ratios[walker], expected_log[walker] - accepted_log[walker]);
        for (std::size_t electron = 0; electron < electron_count; ++electron)
        {
          checkGrad(proposed_gradient[walker][electron],
                    expected_gradient[walker][electron]);
          checkValue(proposed_laplacian[walker][electron],
                     expected_laplacian[walker][electron], 3.0e-7);
        }
      }

      crowd.leader.mw_accept_rejectMultiParticleMove(
          crowd.wfc_list, *crowd.p_list, moves,
          std::vector<bool>(walker_count, false));
    }
  }
}

TEST_CASE("PsiFormer resource mismatch leaves both crowds immediately reusable",
          "[wavefunction][psiformer][multiwalker][resource][threading]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  Crowd crowd_a(files, simulation_cell, 1);
  Crowd crowd_b(files, simulation_cell, 1);

  ResourceCollection template_a("psiformer_model_a_template");
  ResourceCollection template_b("psiformer_model_b_template");
  crowd_a.leader.createResource(template_a);
  crowd_b.leader.createResource(template_b);
  ResourceCollection resource_a(template_a);
  ResourceCollection resource_b(template_b);

  // A same-typed resource from a distinct shared model must be rejected without
  // publishing a leader handle or consuming the collection cursor.
  CHECK_THROWS_AS(ResourceCollectionTeamLock<WaveFunctionComponent>(resource_a, crowd_b.wfc_list),
                  std::logic_error);

  auto evaluate_one = [](Crowd& crowd, ResourceCollection& resource) {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(resource, crowd.wfc_list);
    std::vector<PsiFormerWF::GradType> gradients(crowd.wfc_list.size());
    crowd.leader.mw_evalGrad(crowd.wfc_list, *crowd.p_list, 0, gradients);
    for (const auto& gradient : gradients)
      for (int dimension = 0; dimension < 3; ++dimension)
        CHECK(std::isfinite(std::real(gradient[dimension])));
  };

  // Both the rejected leader and the mismatched collection remain usable.
  evaluate_one(crowd_b, resource_b);
  evaluate_one(crowd_a, resource_a);

  // Heterogeneous component lists fail before lending any resource.
  RefVectorWithLeader<WaveFunctionComponent> mixed_components(crowd_a.leader);
  mixed_components.push_back(crowd_a.leader);
  mixed_components.push_back(crowd_b.leader);
  CHECK_THROWS_AS(ResourceCollectionTeamLock<WaveFunctionComponent>(resource_a, mixed_components),
                  std::invalid_argument);
  evaluate_one(crowd_a, resource_a);
}

} // namespace qmcplusplus
