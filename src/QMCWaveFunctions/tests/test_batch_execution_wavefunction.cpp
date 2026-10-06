//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_batch_execution_wavefunction.cpp
 * @brief Tests generic batch-memory aggregation and binding through TrialWaveFunction.
 */

#include <catch2/catch_test_macros.hpp>

#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/ConstantOrbital.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "SimulationCell.h"
#include "Utilities/ResourceCollection.h"
#include "Utilities/RuntimeOptions.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus
{
namespace
{

/** Minimal concrete component exposing every generic planning lifecycle hook. */
class PlanningComponent : public WaveFunctionComponent
{
public:
  PlanningComponent(std::string class_name,
                    std::string name,
                    BatchExecutionMode required_mode,
                    BatchTileCapacities logical_maximum,
                    std::size_t bytes)
      : WaveFunctionComponent(name),
        class_name_(std::move(class_name)),
        required_mode_(required_mode),
        logical_maximum_(logical_maximum),
        bytes_(bytes)
  {}

  std::string getClassName() const override { return class_name_; }

  LogValue evaluateLog(const ParticleSet&,
                       ParticleSet::ParticleGradient& gradients,
                       ParticleSet::ParticleLaplacian& laplacians) override
  {
    gradients = 0.0;
    laplacians = 0.0;
    return 0.0;
  }

  void acceptMove(ParticleSet&, int, bool = false) override {}
  void restore(int) override {}
  PsiValue ratio(ParticleSet&, int) override { return 1.0; }
  GradType evalGrad(ParticleSet&, int) override { return GradType(0.0); }
  PsiValue ratioGrad(ParticleSet&, int, GradType&) override { return 1.0; }
  void registerData(ParticleSet&, WFBufferType&) override {}
  LogValue updateBuffer(ParticleSet&, WFBufferType&, bool = false) override
  { return 0.0; }
  void copyFromBuffer(ParticleSet&, WFBufferType&) override {}
  void evaluateDerivatives(ParticleSet&,
                           const OptVariables&,
                           Vector<ValueType>&,
                           Vector<ValueType>&) override
  {}

  void contributeBatchExecutionRequirements(
      BatchExecutionRequirements& requirements) const override
  {
    requirements.require(required_mode_);
    ++requirement_calls_;
  }

  BatchMemoryContribution estimateBatchExecutionMemory(
      const BatchExecutionPlanningContext&) const override
  {
    ++estimate_calls_;
    BatchMemoryContribution contribution;
    contribution.logical_maximum = logical_maximum_;
    contribution.owner_multiplicity = 1;
    contribution.fully_accounted = true;
    contribution.per_owner.add(BatchMemoryCategory::FIXED_CLONE_STATE,
                               {bytes_, 0});
    return contribution;
  }

  void validateBatchExecutionPlanBinding(
      const BatchExecutionParticipantPlan& plan) const override
  {
    ++validation_calls_;
    if ((plan && reject_nonempty_binding_) ||
        (!plan && reject_empty_binding_))
      throw std::runtime_error("deliberate component plan validation failure");
  }

  void bindBatchExecutionPlan(BatchExecutionParticipantPlan plan) noexcept override
  {
    bound_plan_ = std::move(plan);
    ++bind_calls_;
  }

  void prepareBatchExecutionClone(
      const BatchExecutionParticipantPlan& plan) override
  {
    if (!bound_plan_.sameBinding(plan))
      throw std::logic_error("component preparation received the wrong plan view");
    ++prepare_calls_;
  }

  void acquireResource(
      ResourceCollection&,
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const override
  {
    auto& leader = wfc_list.getCastedLeader<PlanningComponent>();
    ++leader.acquire_calls_;
    if (leader.throw_on_acquire_)
      throw std::runtime_error("deliberate component acquisition failure");
  }

  void releaseResource(
      ResourceCollection&,
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const override
  {
    auto& leader = wfc_list.getCastedLeader<PlanningComponent>();
    ++leader.release_calls_;
  }

  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet&) const override
  {
    auto clone = std::make_unique<PlanningComponent>(
        class_name_, getName(), required_mode_, logical_maximum_, bytes_);
    clone->copy_binding_in_clone_ = copy_binding_in_clone_;
    if (copy_binding_in_clone_)
      clone->bound_plan_ = bound_plan_;
    return clone;
  }

  void setClassName(std::string class_name) { class_name_ = std::move(class_name); }
  void rejectNonemptyBinding(bool reject) noexcept
  { reject_nonempty_binding_ = reject; }
  void rejectEmptyBinding(bool reject) noexcept
  { reject_empty_binding_ = reject; }
  void throwOnAcquire(bool should_throw) noexcept
  { throw_on_acquire_ = should_throw; }
  void copyBindingInClone(bool copy) noexcept
  { copy_binding_in_clone_ = copy; }

  const BatchExecutionParticipantPlan& boundPlan() const noexcept
  { return bound_plan_; }
  std::size_t requirementCalls() const noexcept { return requirement_calls_; }
  std::size_t estimateCalls() const noexcept { return estimate_calls_; }
  std::size_t validationCalls() const noexcept { return validation_calls_; }
  std::size_t bindCalls() const noexcept { return bind_calls_; }
  std::size_t prepareCalls() const noexcept { return prepare_calls_; }
  std::size_t acquireCalls() const noexcept { return acquire_calls_; }
  std::size_t releaseCalls() const noexcept { return release_calls_; }

private:
  std::string class_name_;
  BatchExecutionMode required_mode_;
  BatchTileCapacities logical_maximum_;
  std::size_t bytes_;
  bool reject_nonempty_binding_ = false;
  bool reject_empty_binding_ = false;
  bool throw_on_acquire_ = false;
  bool copy_binding_in_clone_ = false;
  mutable std::size_t requirement_calls_ = 0;
  mutable std::size_t estimate_calls_ = 0;
  mutable std::size_t validation_calls_ = 0;
  std::size_t bind_calls_ = 0;
  std::size_t prepare_calls_ = 0;
  mutable std::size_t acquire_calls_ = 0;
  mutable std::size_t release_calls_ = 0;
  BatchExecutionParticipantPlan bound_plan_;
};

/// Select a small immutable plan using the TrialWaveFunction as provider.
std::shared_ptr<const BatchExecutionPlan> makePlan(
    const TrialWaveFunction& wavefunction,
    std::string profile_id = "twf-test-v1",
    std::size_t preferred_value_tile = 3)
{
  BatchExecutionSelectionInput input;
  wavefunction.contributeBatchExecutionRequirements(input.requirements);
  input.topology.initial_walkers_per_crowd = {2};
  input.topology.reserve_walkers_per_crowd = {3};
  input.topology.run_kind = "wavefunction-unit-test";
  input.logical_maximum = {8, 6, 4, 0};
  input.preference.id = std::move(profile_id);
  input.preference.preferred = {preferred_value_tile, 2, 2, 0};
  return std::make_shared<const BatchExecutionPlan>(
      selectBatchExecutionPlan(
          input, [&wavefunction](const BatchExecutionPlanningContext& context) {
            return wavefunction.estimateBatchExecutionMemory(context);
          }));
}

/// Return one component with a checked concrete type from a test wavefunction.
PlanningComponent& planningComponent(TrialWaveFunction& wavefunction,
                                     std::size_t index)
{
  return dynamic_cast<PlanningComponent&>(
      *wavefunction.getOrbitals().at(index));
}

} // namespace

TEST_CASE("TrialWaveFunction aggregates ordered batch planning participants",
          "[wavefunction][batch_memory]")
{
  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "planning");
  auto first = std::make_unique<PlanningComponent>(
      "A/B% C", "n/%", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 11);
  PlanningComponent* first_ptr = first.get();
  wavefunction.addComponent(std::move(first));
  auto second = std::make_unique<PlanningComponent>(
      "Second", "", BatchExecutionMode::FULL_VGL,
      BatchTileCapacities{0, 6, 0, 0}, 13);
  PlanningComponent* second_ptr = second.get();
  wavefunction.addComponent(std::move(second));

  BatchExecutionRequirements requirements;
  wavefunction.contributeBatchExecutionRequirements(requirements);
  CHECK(requirements.requires(BatchExecutionMode::VALUE));
  CHECK(requirements.requires(BatchExecutionMode::FULL_VGL));
  CHECK_FALSE(requirements.requires(BatchExecutionMode::ACTIVE_GRADIENT));
  CHECK(first_ptr->requirementCalls() == 1);
  CHECK(second_ptr->requirementCalls() == 1);

  BatchExecutionPlanningContext context;
  context.requirements = requirements;
  context.logical_maximum = {8, 6, 4, 0};
  context.candidate_capacities = {3, 2, 0, 0};
  const auto contributions = wavefunction.estimateBatchExecutionMemory(context);
  REQUIRE(contributions.size() == 2);
  CHECK(contributions[0].participant_id ==
        "twf/component/0/A%2FB%25%20C/n%2F%25");
  CHECK(contributions[1].participant_id ==
        "twf/component/1/Second/");
  CHECK(contributions[0].contribution.per_owner.total().host == 11);
  CHECK(contributions[1].contribution.per_owner.total().host == 13);
  CHECK(contributions[0].contribution.fully_accounted);
  CHECK(contributions[1].contribution.fully_accounted);

  ConstantOrbital legacy_component;
  BatchExecutionRequirements legacy_requirements;
  legacy_component.contributeBatchExecutionRequirements(legacy_requirements);
  CHECK(legacy_requirements.empty());
  const BatchMemoryContribution legacy_contribution =
      legacy_component.estimateBatchExecutionMemory(context);
  CHECK_FALSE(legacy_contribution.fully_accounted);
  CHECK(legacy_contribution.owner_multiplicity == 0);
  CHECK(legacy_contribution.per_owner.total() == BatchMemoryBytes{});
}

TEST_CASE("TrialWaveFunction batch plan binding is aggregate-atomic",
          "[wavefunction][batch_memory]")
{
  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "binding");
  auto first = std::make_unique<PlanningComponent>(
      "First", "one", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 17);
  PlanningComponent* first_ptr = first.get();
  wavefunction.addComponent(std::move(first));
  auto second = std::make_unique<PlanningComponent>(
      "Second", "two", BatchExecutionMode::FULL_VGL,
      BatchTileCapacities{0, 6, 0, 0}, 19);
  PlanningComponent* second_ptr = second.get();
  wavefunction.addComponent(std::move(second));

  const auto first_plan = makePlan(wavefunction, "binding-v1", 3);
  const auto second_plan = makePlan(wavefunction, "binding-v2", 2);
  wavefunction.bindBatchExecutionPlan(first_plan);
  REQUIRE(wavefunction.batchExecutionPlan().get() == first_plan.get());
  REQUIRE(first_ptr->boundPlan());
  REQUIRE(second_ptr->boundPlan());
  CHECK(&first_ptr->boundPlan().plan() == first_plan.get());
  CHECK(&second_ptr->boundPlan().plan() == first_plan.get());
  CHECK(first_ptr->boundPlan().evidence().participant_id ==
        "twf/component/0/First/one");
  CHECK(second_ptr->boundPlan().evidence().participant_id ==
        "twf/component/1/Second/two");

  CHECK_THROWS_AS(
      wavefunction.addComponent(std::make_unique<ConstantOrbital>()),
      std::logic_error);

  const std::size_t first_bind_count = first_ptr->bindCalls();
  second_ptr->rejectNonemptyBinding(true);
  CHECK_THROWS_AS(wavefunction.bindBatchExecutionPlan(second_plan),
                  std::runtime_error);
  CHECK(wavefunction.batchExecutionPlan().get() == first_plan.get());
  CHECK(&first_ptr->boundPlan().plan() == first_plan.get());
  CHECK(first_ptr->bindCalls() == first_bind_count);

  second_ptr->rejectNonemptyBinding(false);
  wavefunction.bindBatchExecutionPlan(second_plan);
  CHECK(wavefunction.batchExecutionPlan().get() == second_plan.get());
  CHECK(&first_ptr->boundPlan().plan() == second_plan.get());
  CHECK(&second_ptr->boundPlan().plan() == second_plan.get());

  second_ptr->rejectEmptyBinding(true);
  CHECK_THROWS_AS(wavefunction.bindBatchExecutionPlan(nullptr),
                  std::runtime_error);
  CHECK(wavefunction.batchExecutionPlan().get() == second_plan.get());
  CHECK(first_ptr->boundPlan());

  second_ptr->rejectEmptyBinding(false);
  wavefunction.bindBatchExecutionPlan(nullptr);
  CHECK_FALSE(wavefunction.batchExecutionPlan());
  CHECK_FALSE(first_ptr->boundPlan());
  CHECK_FALSE(second_ptr->boundPlan());

  // Null-to-null binding still visits every child to clear stale copied state.
  const std::size_t null_bind_count = first_ptr->bindCalls();
  wavefunction.bindBatchExecutionPlan(nullptr);
  CHECK(first_ptr->bindCalls() == null_bind_count + 1);
  wavefunction.addComponent(std::make_unique<ConstantOrbital>());
}

TEST_CASE("TrialWaveFunction propagates plan identity and defers clone preparation",
          "[wavefunction][batch_memory][resources]")
{
  RuntimeOptions runtime_options;
  SimulationCell simulation_cell;
  ParticleSet particles(simulation_cell);
  particles.create({1});

  TrialWaveFunction leader(runtime_options, "clone");
  leader.addComponent(std::make_unique<PlanningComponent>(
      "Cloneable", "component", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 23));
  const auto plan = makePlan(leader);
  leader.bindBatchExecutionPlan(plan);

  std::unique_ptr<TrialWaveFunction> clone = leader.makeClone(particles);
  CHECK(clone->batchExecutionPlan().get() == plan.get());
  PlanningComponent& clone_component = planningComponent(*clone, 0);
  REQUIRE(clone_component.boundPlan());
  CHECK(&clone_component.boundPlan().plan() == plan.get());
  CHECK(clone_component.prepareCalls() == 0);
  clone->prepareBatchExecutionClones();
  CHECK(clone_component.prepareCalls() == 1);

  PlanningComponent& leader_component = planningComponent(leader, 0);
  leader_component.setClassName("mutated/class");
  CHECK_THROWS_AS(leader.prepareBatchExecutionClones(), std::logic_error);
  leader_component.setClassName("Cloneable");

  RefVectorWithLeader<TrialWaveFunction> wavefunctions(
      leader, {leader, *clone});
  ResourceCollection resources("batch-planning-wavefunction");

  clone_component.setClassName("mutated/clone");
  CHECK_THROWS_AS(
      TrialWaveFunction::acquireResource(resources, wavefunctions),
      std::invalid_argument);
  CHECK_FALSE(leader.hasAcquiredResource());
  CHECK_FALSE(clone->hasAcquiredResource());
  CHECK(leader_component.acquireCalls() == 0);
  clone_component.setClassName("Cloneable");

  TrialWaveFunction::acquireResource(resources, wavefunctions);
  CHECK(leader.hasAcquiredResource());
  CHECK(clone->hasAcquiredResource());

  leader_component.setClassName("mutated/while-acquired");
  CHECK_THROWS_AS(leader.bindBatchExecutionPlan(plan), std::logic_error);
  leader_component.setClassName("Cloneable");

  const std::size_t leader_bind_count = leader_component.bindCalls();
  leader.bindBatchExecutionPlan(plan);
  CHECK(leader_component.bindCalls() == leader_bind_count);
  CHECK_THROWS_AS(leader.bindBatchExecutionPlan(nullptr), std::logic_error);
  CHECK_THROWS_AS(leader.prepareBatchExecutionClones(), std::logic_error);
  CHECK_THROWS_AS(leader.makeClone(particles), std::logic_error);
  CHECK_THROWS_AS(
      leader.addComponent(std::make_unique<ConstantOrbital>()),
      std::logic_error);

  TrialWaveFunction::releaseResource(resources, wavefunctions);
  CHECK_FALSE(leader.hasAcquiredResource());
  CHECK_FALSE(clone->hasAcquiredResource());
  CHECK_THROWS_AS(
      TrialWaveFunction::releaseResource(resources, wavefunctions),
      std::logic_error);

  const auto other_plan = makePlan(leader, "other-plan", 1);
  clone->bindBatchExecutionPlan(other_plan);
  CHECK_THROWS_AS(
      TrialWaveFunction::acquireResource(resources, wavefunctions),
      std::invalid_argument);
  CHECK_FALSE(leader.hasAcquiredResource());
  CHECK_FALSE(clone->hasAcquiredResource());
}

TEST_CASE("TrialWaveFunction acquisition failure preserves unacquired state",
          "[wavefunction][batch_memory][resources]")
{
  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "rollback");
  auto first = std::make_unique<PlanningComponent>(
      "First", "", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 5);
  PlanningComponent* first_ptr = first.get();
  wavefunction.addComponent(std::move(first));
  auto second = std::make_unique<PlanningComponent>(
      "Second", "", BatchExecutionMode::FULL_VGL,
      BatchTileCapacities{0, 6, 0, 0}, 7);
  PlanningComponent* second_ptr = second.get();
  second_ptr->throwOnAcquire(true);
  wavefunction.addComponent(std::move(second));
  const auto plan = makePlan(wavefunction);
  wavefunction.bindBatchExecutionPlan(plan);

  RefVectorWithLeader<TrialWaveFunction> wavefunctions(
      wavefunction, {wavefunction});
  ResourceCollection resources("batch-planning-rollback");
  CHECK_THROWS_AS(
      TrialWaveFunction::acquireResource(resources, wavefunctions),
      std::runtime_error);
  CHECK_FALSE(wavefunction.hasAcquiredResource());
  CHECK(first_ptr->acquireCalls() == 1);
  CHECK(first_ptr->releaseCalls() == 1);
  CHECK(second_ptr->acquireCalls() == 1);
  CHECK(second_ptr->releaseCalls() == 0);

  second_ptr->throwOnAcquire(false);
  TrialWaveFunction::acquireResource(resources, wavefunctions);
  CHECK(wavefunction.hasAcquiredResource());
  TrialWaveFunction::releaseResource(resources, wavefunctions);
  CHECK_FALSE(wavefunction.hasAcquiredResource());
}

TEST_CASE("TrialWaveFunction resource lifecycle accepts legacy batch lane shapes",
          "[wavefunction][batch_memory][resources]")
{
  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "lane-shapes");
  auto component = std::make_unique<PlanningComponent>(
      "LaneCompatible", "", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 5);
  PlanningComponent* component_ptr = component.get();
  wavefunction.addComponent(std::move(component));
  wavefunction.bindBatchExecutionPlan(makePlan(wavefunction));
  ResourceCollection resources("batch-planning-lane-shapes");

  SECTION("one object may occupy multiple batch lanes")
  {
    RefVectorWithLeader<TrialWaveFunction> wavefunctions(
        wavefunction, {wavefunction, wavefunction});
    TrialWaveFunction::acquireResource(resources, wavefunctions);
    CHECK(wavefunction.hasAcquiredResource());
    CHECK(component_ptr->acquireCalls() == 1);

    TrialWaveFunction::releaseResource(resources, wavefunctions);
    CHECK_FALSE(wavefunction.hasAcquiredResource());
    CHECK(component_ptr->releaseCalls() == 1);
  }

  SECTION("an empty lane vector still manages its designated leader")
  {
    RefVectorWithLeader<TrialWaveFunction> wavefunctions(wavefunction);
    TrialWaveFunction::acquireResource(resources, wavefunctions);
    CHECK(wavefunction.hasAcquiredResource());
    CHECK(component_ptr->acquireCalls() == 1);

    TrialWaveFunction::releaseResource(resources, wavefunctions);
    CHECK_FALSE(wavefunction.hasAcquiredResource());
    CHECK(component_ptr->releaseCalls() == 1);
  }
}

TEST_CASE("TrialWaveFunction null clone binding clears stale child state",
          "[wavefunction][batch_memory]")
{
  RuntimeOptions runtime_options;
  SimulationCell simulation_cell;
  ParticleSet particles(simulation_cell);
  particles.create({1});

  TrialWaveFunction plan_source(runtime_options, "plan-source");
  plan_source.addComponent(std::make_unique<PlanningComponent>(
      "LegacyClone", "", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 3));
  const auto stale_plan = makePlan(plan_source);

  TrialWaveFunction unbound(runtime_options, "unbound");
  auto component = std::make_unique<PlanningComponent>(
      "LegacyClone", "", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 3);
  PlanningComponent* component_ptr = component.get();
  component_ptr->copyBindingInClone(true);
  unbound.addComponent(std::move(component));
  component_ptr->bindBatchExecutionPlan(makeBatchExecutionParticipantPlan(
      stale_plan, "twf/component/0/LegacyClone/"));
  REQUIRE(component_ptr->boundPlan());
  REQUIRE_FALSE(unbound.batchExecutionPlan());

  std::unique_ptr<TrialWaveFunction> clone = unbound.makeClone(particles);
  CHECK_FALSE(clone->batchExecutionPlan());
  CHECK_FALSE(planningComponent(*clone, 0).boundPlan());
}

} // namespace qmcplusplus
