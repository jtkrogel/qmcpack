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
#include "QMCWaveFunctions/TrialWaveFunctionMemoryPolicy.h"
#include "SimulationCell.h"
#include "Utilities/ResourceCollection.h"
#include "Utilities/RuntimeOptions.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace qmcplusplus
{
namespace testing
{
/** Friend-only access to aggregate accounting state and binding diagnostics. */
class TestTrialWaveFunction
{
public:
  /// Enable complete aggregate claims only for explicitly bounded test paths.
  static void useCompleteBatchMemoryAccounting(TrialWaveFunction& wavefunction,
                                                bool enabled = true)
  {
    wavefunction.complete_batch_memory_accounting_for_testing_ = enabled;
  }

  /// Return the aggregate participant view retained by the wavefunction.
  static const BatchExecutionParticipantPlan& aggregatePlan(
      const TrialWaveFunction& wavefunction)
  {
    return wavefunction.bound_batch_topology_.aggregate_plan;
  }

  /// Return the exact sole-component participant view retained inline.
  static const BatchExecutionParticipantPlan& soleComponentPlan(
      const TrialWaveFunction& wavefunction)
  {
    return wavefunction.bound_batch_topology_.sole_component_plan;
  }

  /// Report whether the fixed-size bound topology snapshot is active.
  static bool hasBoundTopology(const TrialWaveFunction& wavefunction)
  {
    return wavefunction.bound_batch_topology_.engaged;
  }

  /// Report whether a fixed-size resource-loan topology snapshot is active.
  static bool hasAcquiredTopology(const TrialWaveFunction& wavefunction)
  {
    return wavefunction.acquired_batch_topology_.engaged;
  }

  /// Return the retained component count without exposing private state types.
  static std::size_t boundComponentCount(const TrialWaveFunction& wavefunction)
  {
    return wavefunction.bound_batch_topology_.component_count;
  }

  /// Return the resource-loan component count from fixed-size retained state.
  static std::size_t acquiredComponentCount(const TrialWaveFunction& wavefunction)
  {
    return wavefunction.acquired_batch_topology_.component_count;
  }

  /// Return the fixed-width structural identity retained for the bound topology.
  static std::uint64_t boundTopologyFingerprint(const TrialWaveFunction& wavefunction)
  {
    return wavefunction.bound_batch_topology_.participant_fingerprint;
  }

  /// Confirm publication of the inline state cannot throw after validation.
  static constexpr bool inlineTopologyPublicationIsNothrow()
  {
    return std::is_nothrow_copy_assignable_v<TrialWaveFunction::InlineBatchTopologyState>;
  }

  /// Install the currently unaccounted fast-derivative fallback for a gate test.
  static void installFastDerivativeFallback(TrialWaveFunction& wavefunction)
  {
    wavefunction.twf_fastderiv_ = std::make_unique<TWFFastDerivWrapper>();
  }
};
} // namespace testing

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

  BatchTileCapacities batchExecutionLogicalMaximum(
      const BatchExecutionWorkloadContext& context) const override
  {
    last_workload_requirements_    = context.requirements;
    last_workload_topology_        = context.topology;
    last_workload_parameter_count_ = context.active_parameter_count;
    ++logical_maximum_calls_;
    return logical_maximum_;
  }

  BatchMemoryContribution estimateBatchExecutionMemory(
      const BatchExecutionPlanningContext&) const override
  {
    ++estimate_calls_;
    BatchMemoryContribution contribution;
    contribution.logical_maximum    = logical_maximum_;
    contribution.owner_multiplicity = 1;
    contribution.fully_accounted    = true;
    contribution.per_owner.add(BatchMemoryCategory::FIXED_CLONE_STATE,
                               {bytes_, 0});
    return contribution;
  }

  bool supportsAtomicBatchPublication() const noexcept override
  {
    return atomic_publication_;
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
    clone->atomic_publication_    = atomic_publication_;
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
  void setAtomicPublication(bool atomic) noexcept
  { atomic_publication_ = atomic; }

  const BatchExecutionParticipantPlan& boundPlan() const noexcept
  { return bound_plan_; }
  std::size_t requirementCalls() const noexcept { return requirement_calls_; }
  std::size_t logicalMaximumCalls() const noexcept { return logical_maximum_calls_; }
  const BatchExecutionRequirements& lastWorkloadRequirements() const noexcept
  {
    return last_workload_requirements_;
  }
  const BatchExecutionTopology& lastWorkloadTopology() const noexcept { return last_workload_topology_; }
  std::size_t lastWorkloadParameterCount() const noexcept { return last_workload_parameter_count_; }
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
  bool reject_nonempty_binding_                       = false;
  bool reject_empty_binding_                          = false;
  bool throw_on_acquire_                              = false;
  bool copy_binding_in_clone_                         = false;
  bool atomic_publication_                            = true;
  mutable std::size_t requirement_calls_              = 0;
  mutable std::size_t logical_maximum_calls_           = 0;
  mutable BatchExecutionRequirements last_workload_requirements_;
  mutable BatchExecutionTopology last_workload_topology_;
  mutable std::size_t last_workload_parameter_count_   = 0;
  mutable std::size_t estimate_calls_                  = 0;
  mutable std::size_t validation_calls_                = 0;
  std::size_t bind_calls_                               = 0;
  std::size_t prepare_calls_                            = 0;
  mutable std::size_t acquire_calls_                   = 0;
  mutable std::size_t release_calls_                   = 0;
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
  input.topology.run_kind              = "wavefunction-unit-test";
  input.particle_count                 = 4;
  input.active_parameter_count         = 17;
  input.parameter_derivative_width     = 17;
  input.logical_maximum = wavefunction.batchExecutionLogicalMaximum(
      {input.requirements, input.topology, input.particle_count,
       input.active_parameter_count, input.parameter_derivative_width});
  input.preference.id        = std::move(profile_id);
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

  BatchExecutionWorkloadContext workload_context;
  workload_context.requirements                       = requirements;
  workload_context.topology.initial_walkers_per_crowd = {2, 3};
  workload_context.topology.reserve_walkers_per_crowd = {4, 5};
  workload_context.topology.run_kind                  = "logical-envelope-test";
  workload_context.particle_count                     = 4;
  workload_context.active_parameter_count             = 19;
  workload_context.parameter_derivative_width         = 23;
  CHECK(wavefunction.batchExecutionLogicalMaximum(workload_context) ==
        BatchTileCapacities{8, 6, 0, 0});
  CHECK(first_ptr->logicalMaximumCalls() == 1);
  CHECK(first_ptr->lastWorkloadRequirements() == requirements);
  CHECK(first_ptr->lastWorkloadTopology().initial_walkers_per_crowd == std::vector<std::size_t>{2, 3});
  CHECK(first_ptr->lastWorkloadTopology().reserve_walkers_per_crowd == std::vector<std::size_t>{4, 5});
  CHECK(first_ptr->lastWorkloadTopology().run_kind == "logical-envelope-test");
  CHECK(first_ptr->lastWorkloadParameterCount() == 19);

  BatchExecutionPlanningContext context;
  context.requirements               = requirements;
  context.topology                   = workload_context.topology;
  context.logical_maximum            = {8, 6, 4, 0};
  context.candidate_capacities       = {3, 2, 0, 0};
  context.particle_count             = 4;
  context.active_parameter_count     = 19;
  context.parameter_derivative_width = 23;
  const auto contributions = wavefunction.estimateBatchExecutionMemory(context);
  REQUIRE(contributions.size() == 3);
  CHECK(contributions[0].participant_id == TRIAL_WAVEFUNCTION_MEMORY_PARTICIPANT_ID);
  CHECK(contributions[1].participant_id ==
        "twf/component/0/A%2FB%25%20C/n%2F%25");
  CHECK(contributions[2].participant_id ==
        "twf/component/1/Second/");
  CHECK_FALSE(contributions[0].contribution.fully_accounted);
  CHECK(contributions[0].contribution.owner_multiplicity == 1);
  CHECK(contributions[1].contribution.per_owner.total().host == 11);
  CHECK(contributions[2].contribution.per_owner.total().host == 13);
  CHECK(contributions[1].contribution.fully_accounted);
  CHECK(contributions[2].contribution.fully_accounted);
  CHECK(first_ptr->estimateCalls() == 1);
  CHECK(second_ptr->estimateCalls() == 1);

  ConstantOrbital legacy_component;
  BatchExecutionRequirements legacy_requirements;
  legacy_component.contributeBatchExecutionRequirements(legacy_requirements);
  CHECK(legacy_requirements.empty());
  const BatchMemoryContribution legacy_contribution =
      legacy_component.estimateBatchExecutionMemory(context);
  CHECK_FALSE(legacy_contribution.fully_accounted);
  CHECK(legacy_contribution.owner_multiplicity == 0);
  CHECK(legacy_contribution.per_owner.total() == BatchMemoryBytes{});
  CHECK_FALSE(legacy_component.supportsAtomicBatchPublication());
}

TEST_CASE("TrialWaveFunction batch plan binding is aggregate-atomic",
          "[wavefunction][batch_memory]")
{
  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "binding");
  auto component = std::make_unique<PlanningComponent>(
      "First", "one", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 17);
  PlanningComponent* component_ptr = component.get();
  wavefunction.addComponent(std::move(component));
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(wavefunction);

  const auto first_plan = makePlan(wavefunction, "binding-v1", 3);
  const auto second_plan = makePlan(wavefunction, "binding-v2", 2);
  wavefunction.bindBatchExecutionPlan(first_plan);
  REQUIRE(wavefunction.batchExecutionPlan().get() == first_plan.get());
  STATIC_CHECK(testing::TestTrialWaveFunction::inlineTopologyPublicationIsNothrow());
  CHECK(testing::TestTrialWaveFunction::hasBoundTopology(wavefunction));
  CHECK(testing::TestTrialWaveFunction::boundComponentCount(wavefunction) == 1);
  CHECK(testing::TestTrialWaveFunction::boundTopologyFingerprint(wavefunction) != 0);
  REQUIRE(testing::TestTrialWaveFunction::aggregatePlan(wavefunction));
  REQUIRE(testing::TestTrialWaveFunction::soleComponentPlan(wavefunction));
  REQUIRE(component_ptr->boundPlan());
  CHECK(&testing::TestTrialWaveFunction::aggregatePlan(wavefunction).plan() == first_plan.get());
  CHECK(&component_ptr->boundPlan().plan() == first_plan.get());
  CHECK(testing::TestTrialWaveFunction::soleComponentPlan(wavefunction).sameBinding(
      component_ptr->boundPlan()));
  CHECK(testing::TestTrialWaveFunction::aggregatePlan(wavefunction).evidence().participant_id ==
        TRIAL_WAVEFUNCTION_MEMORY_PARTICIPANT_ID);
  CHECK(component_ptr->boundPlan().evidence().participant_id ==
        "twf/component/0/First/one");

  CHECK_THROWS_AS(
      wavefunction.addComponent(std::make_unique<ConstantOrbital>()),
      std::logic_error);

  component_ptr->setAtomicPublication(false);
  CHECK_THROWS_AS(wavefunction.bindBatchExecutionPlan(second_plan),
                  std::invalid_argument);
  CHECK(wavefunction.batchExecutionPlan().get() == first_plan.get());
  CHECK(&testing::TestTrialWaveFunction::aggregatePlan(wavefunction).plan() == first_plan.get());
  CHECK(&component_ptr->boundPlan().plan() == first_plan.get());
  component_ptr->setAtomicPublication(true);

  const std::size_t first_bind_count = component_ptr->bindCalls();
  component_ptr->rejectNonemptyBinding(true);
  CHECK_THROWS_AS(wavefunction.bindBatchExecutionPlan(second_plan),
                  std::runtime_error);
  CHECK(wavefunction.batchExecutionPlan().get() == first_plan.get());
  CHECK(&testing::TestTrialWaveFunction::aggregatePlan(wavefunction).plan() == first_plan.get());
  CHECK(&component_ptr->boundPlan().plan() == first_plan.get());
  CHECK(component_ptr->bindCalls() == first_bind_count);

  component_ptr->rejectNonemptyBinding(false);
  wavefunction.bindBatchExecutionPlan(second_plan);
  CHECK(wavefunction.batchExecutionPlan().get() == second_plan.get());
  CHECK(&testing::TestTrialWaveFunction::aggregatePlan(wavefunction).plan() == second_plan.get());
  CHECK(&component_ptr->boundPlan().plan() == second_plan.get());

  component_ptr->rejectEmptyBinding(true);
  CHECK_THROWS_AS(wavefunction.bindBatchExecutionPlan(nullptr),
                  std::runtime_error);
  CHECK(wavefunction.batchExecutionPlan().get() == second_plan.get());
  CHECK(testing::TestTrialWaveFunction::aggregatePlan(wavefunction));
  CHECK(component_ptr->boundPlan());

  component_ptr->rejectEmptyBinding(false);
  wavefunction.bindBatchExecutionPlan(nullptr);
  CHECK_FALSE(wavefunction.batchExecutionPlan());
  CHECK_FALSE(testing::TestTrialWaveFunction::hasBoundTopology(wavefunction));
  CHECK(testing::TestTrialWaveFunction::boundComponentCount(wavefunction) == 0);
  CHECK(testing::TestTrialWaveFunction::boundTopologyFingerprint(wavefunction) == 0);
  CHECK_FALSE(testing::TestTrialWaveFunction::aggregatePlan(wavefunction));
  CHECK_FALSE(testing::TestTrialWaveFunction::soleComponentPlan(wavefunction));
  CHECK_FALSE(component_ptr->boundPlan());

  // Null-to-null binding still visits every child to clear stale copied state.
  const std::size_t null_bind_count = component_ptr->bindCalls();
  wavefunction.bindBatchExecutionPlan(nullptr);
  CHECK(component_ptr->bindCalls() == null_bind_count + 1);
  wavefunction.addComponent(std::make_unique<ConstantOrbital>());
}

TEST_CASE("TrialWaveFunction hard planning rejects multiple components",
          "[wavefunction][batch_memory]")
{
  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "multiple-components");
  wavefunction.addComponent(std::make_unique<PlanningComponent>(
      "First", "", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 5));
  wavefunction.addComponent(std::make_unique<PlanningComponent>(
      "Second", "", BatchExecutionMode::FULL_VGL,
      BatchTileCapacities{0, 8, 0, 0}, 7));
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(wavefunction);

  CHECK_THROWS_AS(makePlan(wavefunction), std::invalid_argument);
  CHECK_FALSE(wavefunction.batchExecutionPlan());
  CHECK_FALSE(testing::TestTrialWaveFunction::aggregatePlan(wavefunction));
}

TEST_CASE("TrialWaveFunction hard planning rejects fast-derivative fallback",
          "[wavefunction][batch_memory]")
{
  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "fast-derivative-fallback");
  wavefunction.addComponent(std::make_unique<PlanningComponent>(
      "Sole", "", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 5));
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(wavefunction);
  testing::TestTrialWaveFunction::installFastDerivativeFallback(wavefunction);

  CHECK_THROWS_AS(makePlan(wavefunction), std::invalid_argument);
  CHECK_FALSE(wavefunction.batchExecutionPlan());
  CHECK_FALSE(testing::TestTrialWaveFunction::hasBoundTopology(wavefunction));
}

TEST_CASE("TrialWaveFunction aggregate binding rejects fabricated accounting evidence",
          "[wavefunction][batch_memory]")
{
  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "fabricated-accounting");
  wavefunction.addComponent(std::make_unique<PlanningComponent>(
      "Sole", "", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 5));

  BatchExecutionSelectionInput input;
  wavefunction.contributeBatchExecutionRequirements(input.requirements);
  input.topology.initial_walkers_per_crowd = {2};
  input.topology.reserve_walkers_per_crowd = {3};
  input.topology.run_kind                  = "fabricated-accounting-test";
  input.particle_count                    = 4;
  input.active_parameter_count            = 2;
  input.parameter_derivative_width        = 2;
  input.logical_maximum = wavefunction.batchExecutionLogicalMaximum(
      {input.requirements, input.topology, input.particle_count,
       input.active_parameter_count, input.parameter_derivative_width});
  input.preference.id        = "fabricated-accounting-v1";
  input.preference.preferred = {3, 1, 1, 0};

  const auto fabricated_plan = std::make_shared<const BatchExecutionPlan>(
      selectBatchExecutionPlan(
          input, [&wavefunction](const BatchExecutionPlanningContext& context) {
            auto contributions = wavefunction.estimateBatchExecutionMemory(context);
            if (contributions.empty() || contributions.front().participant_id !=
                    TRIAL_WAVEFUNCTION_MEMORY_PARTICIPANT_ID)
              throw std::logic_error("aggregate participant is not first");
            contributions.front().contribution.fully_accounted = true;
            return contributions;
          }));

  CHECK_THROWS_AS(wavefunction.bindBatchExecutionPlan(fabricated_plan),
                  std::invalid_argument);
  CHECK_FALSE(wavefunction.batchExecutionPlan());
  CHECK_FALSE(testing::TestTrialWaveFunction::aggregatePlan(wavefunction));
  CHECK_FALSE(planningComponent(wavefunction, 0).boundPlan());
}

TEST_CASE("TrialWaveFunction aggregate binding recomputes selected evidence",
          "[wavefunction][batch_memory]")
{
  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "stale-selected");
  wavefunction.addComponent(std::make_unique<PlanningComponent>(
      "Sole", "", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 5));
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(wavefunction);

  BatchExecutionSelectionInput input;
  wavefunction.contributeBatchExecutionRequirements(input.requirements);
  input.topology.initial_walkers_per_crowd = {2};
  input.topology.reserve_walkers_per_crowd = {3};
  input.topology.run_kind                  = "stale-selected-test";
  input.particle_count                    = 4;
  input.logical_maximum = wavefunction.batchExecutionLogicalMaximum(
      {input.requirements, input.topology, input.particle_count, 0, 0});
  input.preference.id        = "stale-selected-v1";
  input.preference.preferred = {3, 1, 1, 0};

  const auto stale_plan = std::make_shared<const BatchExecutionPlan>(
      selectBatchExecutionPlan(
          input, [&wavefunction](const BatchExecutionPlanningContext& context) {
            auto contributions = wavefunction.estimateBatchExecutionMemory(context);
            contributions.front().contribution.per_owner.add(
                BatchMemoryCategory::RETAINED_HIGH_WATER, {1, 0},
                "deliberately stale selected evidence");
            return contributions;
          }));

  CHECK_THROWS_AS(wavefunction.bindBatchExecutionPlan(stale_plan),
                  std::invalid_argument);
  CHECK_FALSE(wavefunction.batchExecutionPlan());
  CHECK_FALSE(testing::TestTrialWaveFunction::aggregatePlan(wavefunction));
  CHECK_FALSE(planningComponent(wavefunction, 0).boundPlan());
}

TEST_CASE("TrialWaveFunction aggregate binding recomputes minimum evidence",
          "[wavefunction][batch_memory]")
{
  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "stale-minimum");
  wavefunction.addComponent(std::make_unique<PlanningComponent>(
      "Sole", "", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 5));
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(wavefunction);

  BatchExecutionSelectionInput input;
  wavefunction.contributeBatchExecutionRequirements(input.requirements);
  input.topology.initial_walkers_per_crowd = {2};
  input.topology.reserve_walkers_per_crowd = {3};
  input.topology.run_kind                  = "stale-minimum-test";
  input.particle_count                    = 4;
  input.logical_maximum = wavefunction.batchExecutionLogicalMaximum(
      {input.requirements, input.topology, input.particle_count, 0, 0});
  input.preference.id        = "stale-minimum-v1";
  input.preference.preferred = {3, 1, 1, 0};

  const auto stale_plan = std::make_shared<const BatchExecutionPlan>(
      selectBatchExecutionPlan(
          input, [&wavefunction](const BatchExecutionPlanningContext& context) {
            auto contributions = wavefunction.estimateBatchExecutionMemory(context);
            contributions.back().contribution.per_owner.add(
                BatchMemoryCategory::INNER_TILE_SCRATCH,
                {2 * context.candidate_capacities.value, 0},
                "monotone synthetic child slope");
            if (context.candidate_capacities.value == 1)
              contributions.front().contribution.per_owner.add(
                  BatchMemoryCategory::RETAINED_HIGH_WATER, {1, 0},
                  "deliberately stale minimum evidence");
            return contributions;
          }));

  CHECK_THROWS_AS(wavefunction.bindBatchExecutionPlan(stale_plan),
                  std::invalid_argument);
  CHECK_FALSE(wavefunction.batchExecutionPlan());
  CHECK_FALSE(testing::TestTrialWaveFunction::aggregatePlan(wavefunction));
  CHECK_FALSE(planningComponent(wavefunction, 0).boundPlan());
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
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(leader);
  const auto plan = makePlan(leader);
  leader.bindBatchExecutionPlan(plan);

  std::unique_ptr<TrialWaveFunction> clone = leader.makeClone(particles);
  CHECK(clone->batchExecutionPlan().get() == plan.get());
  CHECK(testing::TestTrialWaveFunction::hasBoundTopology(*clone));
  CHECK(testing::TestTrialWaveFunction::boundComponentCount(*clone) == 1);
  CHECK(testing::TestTrialWaveFunction::boundTopologyFingerprint(*clone) ==
        testing::TestTrialWaveFunction::boundTopologyFingerprint(leader));
  REQUIRE(testing::TestTrialWaveFunction::aggregatePlan(*clone));
  CHECK(testing::TestTrialWaveFunction::aggregatePlan(*clone).sameBinding(
      testing::TestTrialWaveFunction::aggregatePlan(leader)));
  CHECK(testing::TestTrialWaveFunction::soleComponentPlan(*clone).sameBinding(
      testing::TestTrialWaveFunction::soleComponentPlan(leader)));
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
  CHECK(testing::TestTrialWaveFunction::hasAcquiredTopology(leader));
  CHECK(testing::TestTrialWaveFunction::hasAcquiredTopology(*clone));
  CHECK(testing::TestTrialWaveFunction::acquiredComponentCount(leader) == 1);
  CHECK(testing::TestTrialWaveFunction::acquiredComponentCount(*clone) == 1);

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
  CHECK_FALSE(testing::TestTrialWaveFunction::hasAcquiredTopology(leader));
  CHECK_FALSE(testing::TestTrialWaveFunction::hasAcquiredTopology(*clone));
  CHECK(testing::TestTrialWaveFunction::acquiredComponentCount(leader) == 0);
  CHECK(testing::TestTrialWaveFunction::acquiredComponentCount(*clone) == 0);
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

TEST_CASE("TrialWaveFunction legacy multi-component acquisition preserves rollback",
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

  // Multiple components remain supported without a hard plan while the
  // aggregate planner deliberately rejects that topology.
  REQUIRE_FALSE(wavefunction.batchExecutionPlan());

  RefVectorWithLeader<TrialWaveFunction> wavefunctions(
      wavefunction, {wavefunction});
  ResourceCollection resources("batch-planning-rollback");
  CHECK_THROWS_AS(
      TrialWaveFunction::acquireResource(resources, wavefunctions),
      std::runtime_error);
  CHECK_FALSE(wavefunction.hasAcquiredResource());
  CHECK_FALSE(testing::TestTrialWaveFunction::hasAcquiredTopology(wavefunction));
  CHECK(first_ptr->acquireCalls() == 1);
  CHECK(first_ptr->releaseCalls() == 1);
  CHECK(second_ptr->acquireCalls() == 1);
  CHECK(second_ptr->releaseCalls() == 0);

  second_ptr->throwOnAcquire(false);
  TrialWaveFunction::acquireResource(resources, wavefunctions);
  CHECK(wavefunction.hasAcquiredResource());
  CHECK(testing::TestTrialWaveFunction::hasAcquiredTopology(wavefunction));
  CHECK(testing::TestTrialWaveFunction::acquiredComponentCount(wavefunction) == 2);
  TrialWaveFunction::releaseResource(resources, wavefunctions);
  CHECK_FALSE(wavefunction.hasAcquiredResource());
  CHECK_FALSE(testing::TestTrialWaveFunction::hasAcquiredTopology(wavefunction));
  CHECK(testing::TestTrialWaveFunction::acquiredComponentCount(wavefunction) == 0);
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
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(wavefunction);
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
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(plan_source);
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
