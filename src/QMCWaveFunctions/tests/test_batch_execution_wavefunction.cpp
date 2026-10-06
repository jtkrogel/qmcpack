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
#include <catch2/matchers/catch_matchers_string.hpp>

#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/ConstantOrbital.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/TrialWaveFunctionMemoryPolicy.h"
#include "SimulationCell.h"
#include "Utilities/ResourceCollection.h"
#include "Utilities/BatchResourcePreparation.h"
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

  /// Inspect exact typed storage and live bindings of the aggregate resource.
  static auto aggregateResourceDiagnostics(
      const TrialWaveFunction& wavefunction)
  {
    return wavefunction.aggregateResourceDiagnosticsForTesting();
  }

  /// Corrupt only crowd provenance to exercise post-loan validation rollback.
  static void setAggregateResourceCrowd(ResourceCollection& collection,
                                        std::size_t crowd_index)
  {
    TrialWaveFunction::setAggregateResourceCrowdForTesting(collection,
                                                           crowd_index);
  }

  /// Verify prepared idle views refer only to destination-owned fillers.
  static bool aggregatePlaceholdersMatchFillers(ResourceCollection& collection)
  {
    return TrialWaveFunction::aggregateResourcePlaceholdersMatchFillersForTesting(
        collection);
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

  /// Aggregate clone storage and preparation provenance exposed only to tests.
  struct AggregateCloneDiagnostics
  {
    bool prepared                          = false;
    bool storage_shape_matches_plan        = false;
    bool allocation_identity_matches       = false;
    const void* plan_identity              = nullptr;
    const void* accepted_gradient_data     = nullptr;
    const void* accepted_laplacian_data    = nullptr;
    const void* proposed_gradient_data     = nullptr;
    const void* proposed_laplacian_data    = nullptr;
    std::size_t accepted_gradient_size     = 0;
    std::size_t accepted_gradient_capacity = 0;
    std::size_t accepted_laplacian_size     = 0;
    std::size_t accepted_laplacian_capacity = 0;
    std::size_t proposed_gradient_size      = 0;
    std::size_t proposed_gradient_capacity  = 0;
    std::size_t proposed_laplacian_size     = 0;
    std::size_t proposed_laplacian_capacity = 0;
    std::size_t accepted_gradient_bytes     = 0;
    std::size_t accepted_laplacian_bytes    = 0;
    std::size_t proposed_gradient_bytes     = 0;
    std::size_t proposed_laplacian_bytes    = 0;
  };

  /// Snapshot exact aggregate clone allocation diagnostics without mutation.
  static AggregateCloneDiagnostics aggregateCloneDiagnostics(
      const TrialWaveFunction& wavefunction)
  {
    AggregateCloneDiagnostics diagnostics;
    diagnostics.prepared =
        static_cast<bool>(wavefunction.prepared_aggregate_batch_execution_plan_);
    if (diagnostics.prepared)
      diagnostics.plan_identity =
          &wavefunction.prepared_aggregate_batch_execution_plan_.plan();
    diagnostics.accepted_gradient_data  = wavefunction.G.data();
    diagnostics.accepted_laplacian_data = wavefunction.L.data();
    diagnostics.proposed_gradient_data =
        wavefunction.multi_particle_proposed_gradient_.data();
    diagnostics.proposed_laplacian_data =
        wavefunction.multi_particle_proposed_laplacian_.data();
    diagnostics.accepted_gradient_size     = wavefunction.G.size();
    diagnostics.accepted_gradient_capacity = wavefunction.G.capacity();
    diagnostics.accepted_laplacian_size     = wavefunction.L.size();
    diagnostics.accepted_laplacian_capacity = wavefunction.L.capacity();
    diagnostics.proposed_gradient_size =
        wavefunction.multi_particle_proposed_gradient_.size();
    diagnostics.proposed_gradient_capacity =
        wavefunction.multi_particle_proposed_gradient_.capacity();
    diagnostics.proposed_laplacian_size =
        wavefunction.multi_particle_proposed_laplacian_.size();
    diagnostics.proposed_laplacian_capacity =
        wavefunction.multi_particle_proposed_laplacian_.capacity();
    diagnostics.accepted_gradient_bytes =
        diagnostics.accepted_gradient_capacity *
        sizeof(ParticleSet::ParticleGradient::value_type);
    diagnostics.accepted_laplacian_bytes =
        diagnostics.accepted_laplacian_capacity *
        sizeof(ParticleSet::ParticleLaplacian::value_type);
    diagnostics.proposed_gradient_bytes =
        diagnostics.proposed_gradient_capacity *
        sizeof(ParticleSet::ParticleGradient::value_type);
    diagnostics.proposed_laplacian_bytes =
        diagnostics.proposed_laplacian_capacity *
        sizeof(ParticleSet::ParticleLaplacian::value_type);
    if (diagnostics.prepared)
    {
      const std::size_t particle_count =
          wavefunction.prepared_aggregate_batch_execution_plan_.plan().particleCount();
      diagnostics.storage_shape_matches_plan =
          !wavefunction.G.isAttached() && !wavefunction.L.isAttached() &&
          !wavefunction.multi_particle_proposed_gradient_.isAttached() &&
          !wavefunction.multi_particle_proposed_laplacian_.isAttached() &&
          diagnostics.accepted_gradient_size == particle_count &&
          diagnostics.accepted_gradient_capacity == particle_count &&
          diagnostics.accepted_laplacian_size == particle_count &&
          diagnostics.accepted_laplacian_capacity == particle_count &&
          diagnostics.proposed_gradient_size == particle_count &&
          diagnostics.proposed_gradient_capacity == particle_count &&
          diagnostics.proposed_laplacian_size == particle_count &&
          diagnostics.proposed_laplacian_capacity == particle_count;
    }
    diagnostics.allocation_identity_matches = !diagnostics.prepared ||
        (diagnostics.accepted_gradient_data ==
             wavefunction.prepared_aggregate_accepted_gradient_data_ &&
         diagnostics.accepted_laplacian_data ==
             wavefunction.prepared_aggregate_accepted_laplacian_data_ &&
         diagnostics.proposed_gradient_data ==
             wavefunction.prepared_aggregate_proposed_gradient_data_ &&
         diagnostics.proposed_laplacian_data ==
             wavefunction.prepared_aggregate_proposed_laplacian_data_);
    return diagnostics;
  }

  /// Exercise aggregate participant identity validation independently of the wrapper.
  static void prepareAggregateClone(
      TrialWaveFunction& wavefunction,
      const BatchExecutionParticipantPlan& aggregate_plan)
  {
    wavefunction.prepareBatchExecutionClone(aggregate_plan);
  }

  /// Model an unresolved selected-electron proposal for lifecycle guard tests.
  static void setMultiParticleProposalPending(TrialWaveFunction& wavefunction,
                                              bool pending)
  {
    wavefunction.multi_particle_proposal_pending_ = pending;
  }

  /// Observe whether aggregate selected-move publication is still pending.
  static bool multiParticleProposalPending(
      const TrialWaveFunction& wavefunction) noexcept
  { return wavefunction.multi_particle_proposal_pending_; }
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
  {
    topology_token_ = makeTopologyToken(class_name_, getName());
  }

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
  void prepareGroup(ParticleSet&, int) override
  { ++prepare_group_calls_; }
  void mw_prepareGroup(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>&,
      int) const override
  {
    ++wfc_list.getCastedLeader<PlanningComponent>().mw_prepare_group_calls_;
  }
  void completeUpdates() override
  { ++complete_updates_calls_; }
  void mw_completeUpdates(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const override
  {
    ++wfc_list.getCastedLeader<PlanningComponent>().mw_complete_updates_calls_;
  }
  void registerData(ParticleSet&, WFBufferType&) override
  { ++register_data_calls_; }
  LogValue updateBuffer(ParticleSet&, WFBufferType&, bool = false) override
  {
    ++update_buffer_calls_;
    return 0.0;
  }
  void copyFromBuffer(ParticleSet&, WFBufferType&) override
  { ++copy_from_buffer_calls_; }
  void evaluateDerivatives(ParticleSet&,
                           const OptVariables&,
                           Vector<ValueType>&,
                           Vector<ValueType>&) override
  {}

  void contributeBatchExecutionRequirements(
      BatchExecutionRequirements& requirements) const override
  {
    requirements.require(required_mode_);
    if (require_value_mode_)
      requirements.require(BatchExecutionMode::VALUE);
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

  bool hasBatchExecutionPlanBinding(
      const BatchExecutionParticipantPlan& plan) const noexcept override
  {
    return bound_plan_.sameBinding(plan) &&
        bound_topology_token_ == topology_token_;
  }

  bool hasPreparedBatchExecutionClone(
      const BatchExecutionParticipantPlan& plan) const noexcept override
  {
    return prepared_plan_.sameBinding(plan) &&
        prepared_topology_token_ == topology_token_;
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
    if (!bound_plan_.sameBinding(plan))
      prepared_plan_ = {};
    bound_plan_ = std::move(plan);
    bound_topology_token_ = topology_token_;
    ++bind_calls_;
  }

  void prepareBatchExecutionClone(
      const BatchExecutionParticipantPlan& plan) override
  {
    if (!bound_plan_.sameBinding(plan))
      throw std::logic_error("component preparation received the wrong plan view");
    ++prepare_calls_;
    if (throw_on_prepare_)
      throw std::runtime_error("deliberate component clone-preparation failure");
    prepared_plan_ = plan;
    prepared_topology_token_ = topology_token_;
  }

  void createResource(ResourceCollection& collection) const override
  {
    if (resource_backed_)
      collection.addResource(
          std::make_unique<DummyResource>("PlanningComponentResource"));
  }

  void acquireResource(
      ResourceCollection& collection,
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const override
  {
    auto& leader = wfc_list.getCastedLeader<PlanningComponent>();
    ++leader.acquire_calls_;
    if (leader.resource_backed_)
    {
      const std::size_t entry_cursor = collection.getCursor();
      auto candidate = collection.lendResource<DummyResource>();
      if (leader.throw_on_acquire_)
      {
        collection.rewind(entry_cursor);
        collection.takebackResource(candidate);
        collection.rewind(entry_cursor);
        throw std::runtime_error(
            "deliberate component acquisition failure after child loan");
      }
      leader.resource_handle_ = std::move(candidate);
      return;
    }
    if (leader.throw_on_acquire_)
      throw std::runtime_error("deliberate component acquisition failure");
  }

  void releaseResource(
      ResourceCollection& collection,
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const override
  {
    auto& leader = wfc_list.getCastedLeader<PlanningComponent>();
    if (leader.resource_backed_ && !leader.skip_resource_release_)
      collection.takebackResource(leader.resource_handle_);
    ++leader.release_calls_;
  }

  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet&) const override
  {
    auto clone = std::make_unique<PlanningComponent>(
        class_name_, getName(), required_mode_, logical_maximum_, bytes_);
    clone->copy_binding_in_clone_ = copy_binding_in_clone_;
    clone->atomic_publication_    = atomic_publication_;
    clone->resource_backed_       = resource_backed_;
    clone->require_value_mode_     = require_value_mode_;
    if (copy_binding_in_clone_)
      clone->bound_plan_ = bound_plan_;
    return clone;
  }

  void setClassName(std::string class_name)
  {
    class_name_ = std::move(class_name);
    topology_token_ = makeTopologyToken(class_name_, getName());
  }
  void rejectNonemptyBinding(bool reject) noexcept
  { reject_nonempty_binding_ = reject; }
  void rejectEmptyBinding(bool reject) noexcept
  { reject_empty_binding_ = reject; }
  void throwOnAcquire(bool should_throw) noexcept
  { throw_on_acquire_ = should_throw; }
  void throwOnPrepare(bool should_throw) noexcept
  { throw_on_prepare_ = should_throw; }
  void copyBindingInClone(bool copy) noexcept
  { copy_binding_in_clone_ = copy; }
  void setAtomicPublication(bool atomic) noexcept
  { atomic_publication_ = atomic; }
  void useResource(bool enabled = true) noexcept { resource_backed_ = enabled; }
  void skipResourceRelease(bool skip = true) noexcept
  { skip_resource_release_ = skip; }
  void requireValueMode(bool required = true) noexcept
  { require_value_mode_ = required; }

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
  std::size_t prepareGroupCalls() const noexcept
  { return prepare_group_calls_; }
  std::size_t mwPrepareGroupCalls() const noexcept
  { return mw_prepare_group_calls_; }
  std::size_t completeUpdatesCalls() const noexcept
  { return complete_updates_calls_; }
  std::size_t mwCompleteUpdatesCalls() const noexcept
  { return mw_complete_updates_calls_; }
  std::size_t registerDataCalls() const noexcept
  { return register_data_calls_; }
  std::size_t updateBufferCalls() const noexcept
  { return update_buffer_calls_; }
  std::size_t copyFromBufferCalls() const noexcept
  { return copy_from_buffer_calls_; }

private:
  static std::uint64_t makeTopologyToken(std::string_view class_name,
                                         std::string_view name) noexcept
  {
    std::uint64_t hash = 14695981039346656037ULL;
    for (unsigned char byte : class_name)
    {
      hash ^= byte;
      hash *= 1099511628211ULL;
    }
    hash ^= 0xffU;
    hash *= 1099511628211ULL;
    for (unsigned char byte : name)
    {
      hash ^= byte;
      hash *= 1099511628211ULL;
    }
    return hash;
  }

  std::string class_name_;
  BatchExecutionMode required_mode_;
  BatchTileCapacities logical_maximum_;
  std::size_t bytes_;
  bool reject_nonempty_binding_                       = false;
  bool reject_empty_binding_                          = false;
  bool throw_on_acquire_                              = false;
  bool throw_on_prepare_                              = false;
  bool copy_binding_in_clone_                         = false;
  bool atomic_publication_                            = true;
  bool resource_backed_                               = false;
  bool skip_resource_release_                         = false;
  bool require_value_mode_                            = false;
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
  std::size_t prepare_group_calls_                     = 0;
  std::size_t mw_prepare_group_calls_                  = 0;
  std::size_t complete_updates_calls_                  = 0;
  std::size_t mw_complete_updates_calls_               = 0;
  std::size_t register_data_calls_                     = 0;
  std::size_t update_buffer_calls_                     = 0;
  std::size_t copy_from_buffer_calls_                  = 0;
  BatchExecutionParticipantPlan bound_plan_;
  BatchExecutionParticipantPlan prepared_plan_;
  std::uint64_t topology_token_          = 0;
  std::uint64_t bound_topology_token_    = 0;
  std::uint64_t prepared_topology_token_ = 0;
  mutable ResourceHandle<DummyResource> resource_handle_;
};

/// Select a small immutable plan using the TrialWaveFunction as provider.
std::shared_ptr<const BatchExecutionPlan> makePlan(
    const TrialWaveFunction& wavefunction,
    std::string profile_id = "twf-test-v1",
    std::size_t preferred_value_tile = 3,
    std::size_t particle_count = 4,
    std::vector<std::size_t> initial_walkers = {2},
    std::vector<std::size_t> reserve_walkers = {3},
    std::size_t preferred_ecp_outer_tile = 0)
{
  BatchExecutionSelectionInput input;
  wavefunction.contributeBatchExecutionRequirements(input.requirements);
  input.topology.initial_walkers_per_crowd = std::move(initial_walkers);
  input.topology.reserve_walkers_per_crowd = std::move(reserve_walkers);
  input.topology.run_kind              = "wavefunction-unit-test";
  input.particle_count                 = particle_count;
  input.active_parameter_count         = 17;
  input.parameter_derivative_width     = 17;
  input.target_coordinate              = BatchExecutionTargetCoordinate::POS_ONLY;
  input.logical_maximum = wavefunction.batchExecutionLogicalMaximum(
      {input.requirements, input.topology, input.particle_count,
       input.active_parameter_count, input.parameter_derivative_width, input.target_coordinate});
  input.preference.id        = std::move(profile_id);
  input.preference.preferred = {preferred_value_tile, 2, 2,
                                preferred_ecp_outer_tile};
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

  // A distinct nonempty plan requires an explicit null boundary, so prepared
  // aggregate/child provenance cannot silently cross plan identities.
  CHECK_THROWS_AS(wavefunction.bindBatchExecutionPlan(second_plan),
                  std::logic_error);
  CHECK(wavefunction.batchExecutionPlan().get() == first_plan.get());
  CHECK(&testing::TestTrialWaveFunction::aggregatePlan(wavefunction).plan() == first_plan.get());
  CHECK(&component_ptr->boundPlan().plan() == first_plan.get());

  wavefunction.bindBatchExecutionPlan(nullptr);
  CHECK_FALSE(wavefunction.batchExecutionPlan());

  component_ptr->setAtomicPublication(false);
  CHECK_THROWS_AS(wavefunction.bindBatchExecutionPlan(second_plan),
                  std::invalid_argument);
  CHECK_FALSE(wavefunction.batchExecutionPlan());
  CHECK_FALSE(testing::TestTrialWaveFunction::aggregatePlan(wavefunction));
  CHECK_FALSE(component_ptr->boundPlan());
  component_ptr->setAtomicPublication(true);

  const std::size_t first_bind_count = component_ptr->bindCalls();
  component_ptr->rejectNonemptyBinding(true);
  CHECK_THROWS_AS(wavefunction.bindBatchExecutionPlan(second_plan),
                  std::runtime_error);
  CHECK_FALSE(wavefunction.batchExecutionPlan());
  CHECK_FALSE(testing::TestTrialWaveFunction::aggregatePlan(wavefunction));
  CHECK_FALSE(component_ptr->boundPlan());
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

TEST_CASE("TrialWaveFunction planned walker-buffer aggregate entries fail first",
          "[wavefunction][batch_memory][walker_transaction]")
{
  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "buffer-entry-guard");
  auto component = std::make_unique<PlanningComponent>(
      "BufferGuard", "sole", BatchExecutionMode::VALUE,
      BatchTileCapacities{4, 0, 0, 0}, 17);
  PlanningComponent* component_ptr = component.get();
  wavefunction.addComponent(std::move(component));
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(
      wavefunction);
  wavefunction.bindBatchExecutionPlan(
      makePlan(wavefunction, "buffer-entry-guard-v1", 2));

  const SimulationCell simulation_cell;
  ParticleSet particles(simulation_cell);
  particles.setName("buffer_guard_particles");
  particles.create({4});
  for (std::size_t particle = 0; particle < particles.G.size(); ++particle)
  {
    for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
      particles.G[particle][dimension] =
          QMCTraits::ValueType(0.125 * (1 + 3 * particle + dimension));
    particles.L[particle] = QMCTraits::ValueType(-0.25 * (1 + particle));
  }
  const ParticleSet::ParticleGradient gradients_before = particles.G;
  const ParticleSet::ParticleLaplacian laplacians_before = particles.L;

  TrialWaveFunction::WFBufferType buffer;
  TrialWaveFunction::GradType prefix_gradient;
  prefix_gradient = QMCTraits::ValueType(0.75);
  QMCTraits::FullPrecRealType prefix_scalar = 2.5;
  buffer.add(&prefix_gradient, &prefix_gradient + 1);
  buffer.add(prefix_scalar);
  const std::size_t bulk_cursor_before = buffer.current();
  const std::size_t scalar_cursor_before = buffer.current_scalar();
  const std::size_t size_before = buffer.myData.size();
  const std::size_t capacity_before = buffer.myData.capacity();

  const auto check_unchanged = [&]() {
    CHECK(buffer.current() == bulk_cursor_before);
    CHECK(buffer.current_scalar() == scalar_cursor_before);
    CHECK(buffer.myData.size() == size_before);
    CHECK(buffer.myData.capacity() == capacity_before);
    REQUIRE(particles.G.size() == gradients_before.size());
    REQUIRE(particles.L.size() == laplacians_before.size());
    for (std::size_t particle = 0; particle < particles.G.size(); ++particle)
    {
      for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
        CHECK(particles.G[particle][dimension] ==
              gradients_before[particle][dimension]);
      CHECK(particles.L[particle] == laplacians_before[particle]);
    }
    CHECK(component_ptr->registerDataCalls() == 0);
    CHECK(component_ptr->updateBufferCalls() == 0);
    CHECK(component_ptr->copyFromBufferCalls() == 0);
  };

  CHECK_THROWS_WITH(
      wavefunction.registerData(particles, buffer),
      Catch::Matchers::ContainsSubstring(
          "planned aggregate walker-buffer ownership is deferred"));
  check_unchanged();
  CHECK_THROWS_WITH(
      wavefunction.copyFromBuffer(particles, buffer),
      Catch::Matchers::ContainsSubstring(
          "planned aggregate walker-buffer ownership is deferred"));
  check_unchanged();
  CHECK_THROWS_WITH(
      wavefunction.updateBuffer(particles, buffer, false),
      Catch::Matchers::ContainsSubstring(
          "planned aggregate walker-buffer ownership is deferred"));
  check_unchanged();
}

TEST_CASE("TrialWaveFunction planned aggregate lifecycle entries fail first",
          "[wavefunction][batch_memory][lifecycle_guard][resources]")
{
  constexpr const char* deferred_diagnostic =
      "TrialWaveFunction planned aggregate lifecycle ownership is deferred";

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "lifecycle-entry-guard");
  auto component = std::make_unique<PlanningComponent>(
      "LifecycleGuard", "sole", BatchExecutionMode::VALUE,
      BatchTileCapacities{4, 0, 0, 0}, 17);
  PlanningComponent* component_ptr = component.get();
  component_ptr->useResource();
  wavefunction.addComponent(std::move(component));
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(
      wavefunction);
  const auto plan =
      makePlan(wavefunction, "lifecycle-entry-guard-v1", 2);
  wavefunction.bindBatchExecutionPlan(plan);
  wavefunction.prepareBatchExecutionClones();

  const SimulationCell simulation_cell;
  ParticleSet particles(simulation_cell);
  particles.setName("lifecycle_guard_particles");
  particles.create({4});
  for (std::size_t particle = 0; particle < particles.G.size(); ++particle)
  {
    for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
    {
      particles.R[particle][dimension] =
          QMCTraits::RealType(0.0625 * (1 + 3 * particle + dimension));
      particles.G[particle][dimension] =
          QMCTraits::ValueType(0.125 * (1 + 3 * particle + dimension));
      wavefunction.G[particle][dimension] =
          QMCTraits::ValueType(-0.375 * (1 + 3 * particle + dimension));
    }
    particles.L[particle] = QMCTraits::ValueType(-0.25 * (1 + particle));
    wavefunction.L[particle] =
        QMCTraits::ValueType(0.5 * (1 + particle));
  }
  const ParticleSet::ParticlePos positions_before = particles.R;
  const ParticleSet::ParticleGradient particle_gradients_before = particles.G;
  const ParticleSet::ParticleLaplacian particle_laplacians_before = particles.L;
  const ParticleSet::ParticleGradient aggregate_gradients_before =
      wavefunction.G;
  const ParticleSet::ParticleLaplacian aggregate_laplacians_before =
      wavefunction.L;
  wavefunction.setPhase(QMCTraits::RealType(0.875));
  wavefunction.setLogPsi(QMCTraits::RealType(-3.25));
  const TrialWaveFunction::RealType phase_before = wavefunction.getPhase();
  const TrialWaveFunction::RealType log_before = wavefunction.getLogPsi();

  TrialWaveFunction::WFBufferType buffer;
  TrialWaveFunction::GradType buffer_gradient;
  buffer_gradient = QMCTraits::ValueType(0.75);
  QMCTraits::FullPrecRealType buffer_scalar = 2.5;
  buffer.add(&buffer_gradient, &buffer_gradient + 1);
  buffer.add(buffer_scalar);
  const auto buffer_data_before = buffer.myData;
  const std::size_t buffer_cursor_before = buffer.current();
  const std::size_t buffer_scalar_cursor_before = buffer.current_scalar();
  const std::size_t buffer_capacity_before = buffer.myData.capacity();

  RefVectorWithLeader<TrialWaveFunction> wavefunctions(
      wavefunction, {wavefunction});
  RefVectorWithLeader<ParticleSet> particle_sets(particles, {particles});
  ResourceCollection resources("planned-lifecycle-entry-guard");
  wavefunction.createResource(resources);
  resources.prepareBatchResources({plan, 0});
  TrialWaveFunction::acquireResource(resources, wavefunctions);
  testing::TestTrialWaveFunction::setMultiParticleProposalPending(
      wavefunction, true);
  const bool proposal_pending_before =
      testing::TestTrialWaveFunction::multiParticleProposalPending(
          wavefunction);

  const auto clone_before =
      testing::TestTrialWaveFunction::aggregateCloneDiagnostics(wavefunction);
  const auto resource_before =
      testing::TestTrialWaveFunction::aggregateResourceDiagnostics(
          wavefunction);
  const std::size_t resource_cursor_before = resources.getCursor();
  const std::size_t loan_count_before = resources.getOutstandingLoanCount();
  const std::size_t acquire_calls_before = component_ptr->acquireCalls();
  const std::size_t release_calls_before = component_ptr->releaseCalls();

  const auto check_first_entry_atomicity = [&]() {
    CHECK(wavefunction.batchExecutionPlan().get() == plan.get());
    CHECK(wavefunction.hasAcquiredResource());
    CHECK(testing::TestTrialWaveFunction::hasAcquiredTopology(wavefunction));
    CHECK(testing::TestTrialWaveFunction::acquiredComponentCount(wavefunction) ==
          1);
    CHECK(resources.getCursor() == resource_cursor_before);
    CHECK(resources.getOutstandingLoanCount() == loan_count_before);
    CHECK(component_ptr->acquireCalls() == acquire_calls_before);
    CHECK(component_ptr->releaseCalls() == release_calls_before);
    CHECK(wavefunction.getPhase() == phase_before);
    CHECK(wavefunction.getLogPsi() == log_before);
    CHECK(testing::TestTrialWaveFunction::multiParticleProposalPending(
              wavefunction) == proposal_pending_before);

    const auto clone_after =
        testing::TestTrialWaveFunction::aggregateCloneDiagnostics(
            wavefunction);
    CHECK(clone_after.prepared == clone_before.prepared);
    CHECK(clone_after.storage_shape_matches_plan ==
          clone_before.storage_shape_matches_plan);
    CHECK(clone_after.allocation_identity_matches ==
          clone_before.allocation_identity_matches);
    CHECK(clone_after.plan_identity == clone_before.plan_identity);
    CHECK(clone_after.accepted_gradient_data ==
          clone_before.accepted_gradient_data);
    CHECK(clone_after.accepted_laplacian_data ==
          clone_before.accepted_laplacian_data);
    CHECK(clone_after.proposed_gradient_data ==
          clone_before.proposed_gradient_data);
    CHECK(clone_after.proposed_laplacian_data ==
          clone_before.proposed_laplacian_data);
    CHECK(clone_after.accepted_gradient_size ==
          clone_before.accepted_gradient_size);
    CHECK(clone_after.accepted_gradient_capacity ==
          clone_before.accepted_gradient_capacity);
    CHECK(clone_after.accepted_laplacian_size ==
          clone_before.accepted_laplacian_size);
    CHECK(clone_after.accepted_laplacian_capacity ==
          clone_before.accepted_laplacian_capacity);
    CHECK(clone_after.proposed_gradient_size ==
          clone_before.proposed_gradient_size);
    CHECK(clone_after.proposed_gradient_capacity ==
          clone_before.proposed_gradient_capacity);
    CHECK(clone_after.proposed_laplacian_size ==
          clone_before.proposed_laplacian_size);
    CHECK(clone_after.proposed_laplacian_capacity ==
          clone_before.proposed_laplacian_capacity);
    CHECK(clone_after.accepted_gradient_bytes ==
          clone_before.accepted_gradient_bytes);
    CHECK(clone_after.accepted_laplacian_bytes ==
          clone_before.accepted_laplacian_bytes);
    CHECK(clone_after.proposed_gradient_bytes ==
          clone_before.proposed_gradient_bytes);
    CHECK(clone_after.proposed_laplacian_bytes ==
          clone_before.proposed_laplacian_bytes);

    const auto resource_after =
        testing::TestTrialWaveFunction::aggregateResourceDiagnostics(
            wavefunction);
    CHECK(resource_after.prepared == resource_before.prepared);
    CHECK(resource_after.plan_identity == resource_before.plan_identity);
    CHECK(resource_after.crowd_index == resource_before.crowd_index);
    CHECK(resource_after.reserve_walkers == resource_before.reserve_walkers);
    CHECK(resource_after.storage_fingerprint ==
          resource_before.storage_fingerprint);
    CHECK(resource_after.expected_bytes == resource_before.expected_bytes);
    CHECK(resource_after.actual_bytes == resource_before.actual_bytes);
    CHECK(resource_after.component_reference_bytes ==
          resource_before.component_reference_bytes);
    CHECK(resource_after.gradient_reference_bytes ==
          resource_before.gradient_reference_bytes);
    CHECK(resource_after.laplacian_reference_bytes ==
          resource_before.laplacian_reference_bytes);
    CHECK(resource_after.private_ratio_bytes ==
          resource_before.private_ratio_bytes);
    CHECK(resource_after.total_weight_bytes ==
          resource_before.total_weight_bytes);
    CHECK(resource_after.derivative_delta_bytes ==
          resource_before.derivative_delta_bytes);
    CHECK(resource_after.derivative_view_bytes ==
          resource_before.derivative_view_bytes);
    CHECK(resource_after.value_stamp_bytes ==
          resource_before.value_stamp_bytes);
    CHECK(resource_after.transaction_flag_bytes ==
          resource_before.transaction_flag_bytes);
    CHECK(resource_after.component_reference_data ==
          resource_before.component_reference_data);
    CHECK(resource_after.gradient_reference_data ==
          resource_before.gradient_reference_data);
    CHECK(resource_after.laplacian_reference_data ==
          resource_before.laplacian_reference_data);
    CHECK(resource_after.component_leader == resource_before.component_leader);
    CHECK(resource_after.first_component == resource_before.first_component);
    CHECK(resource_after.gradient_leader == resource_before.gradient_leader);
    CHECK(resource_after.first_gradient == resource_before.first_gradient);
    CHECK(resource_after.laplacian_leader == resource_before.laplacian_leader);
    CHECK(resource_after.first_laplacian == resource_before.first_laplacian);

    REQUIRE(particles.R.size() == positions_before.size());
    REQUIRE(particles.G.size() == particle_gradients_before.size());
    REQUIRE(particles.L.size() == particle_laplacians_before.size());
    REQUIRE(wavefunction.G.size() == aggregate_gradients_before.size());
    REQUIRE(wavefunction.L.size() == aggregate_laplacians_before.size());
    for (std::size_t particle = 0; particle < particles.G.size(); ++particle)
    {
      for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
      {
        CHECK(particles.R[particle][dimension] ==
              positions_before[particle][dimension]);
        CHECK(particles.G[particle][dimension] ==
              particle_gradients_before[particle][dimension]);
        CHECK(wavefunction.G[particle][dimension] ==
              aggregate_gradients_before[particle][dimension]);
      }
      CHECK(particles.L[particle] == particle_laplacians_before[particle]);
      CHECK(wavefunction.L[particle] == aggregate_laplacians_before[particle]);
    }

    CHECK(buffer.current() == buffer_cursor_before);
    CHECK(buffer.current_scalar() == buffer_scalar_cursor_before);
    CHECK(buffer.myData.capacity() == buffer_capacity_before);
    CHECK(buffer.myData == buffer_data_before);
    CHECK(component_ptr->prepareGroupCalls() == 0);
    CHECK(component_ptr->mwPrepareGroupCalls() == 0);
    CHECK(component_ptr->completeUpdatesCalls() == 0);
    CHECK(component_ptr->mwCompleteUpdatesCalls() == 0);
  };

  CHECK_THROWS_WITH(wavefunction.prepareGroup(particles, 0),
                    deferred_diagnostic);
  check_first_entry_atomicity();
  CHECK_THROWS_WITH(wavefunction.completeUpdates(), deferred_diagnostic);
  check_first_entry_atomicity();
  CHECK_THROWS_WITH(
      TrialWaveFunction::mw_prepareGroup(wavefunctions, particle_sets, 0),
      deferred_diagnostic);
  check_first_entry_atomicity();
  CHECK_THROWS_WITH(TrialWaveFunction::mw_completeUpdates(wavefunctions),
                    deferred_diagnostic);
  check_first_entry_atomicity();

  // Once the loan is safely returned and the plan is cleared, the unchanged
  // legacy dispatch path remains available for all four lifecycle entries.
  testing::TestTrialWaveFunction::setMultiParticleProposalPending(
      wavefunction, false);
  TrialWaveFunction::releaseResource(resources, wavefunctions);
  CHECK_FALSE(wavefunction.hasAcquiredResource());
  CHECK_FALSE(testing::TestTrialWaveFunction::hasAcquiredTopology(
      wavefunction));
  CHECK(resources.getOutstandingLoanCount() == 0);
  CHECK(resources.getCursor() == resource_cursor_before);
  wavefunction.bindBatchExecutionPlan(nullptr);
  CHECK_FALSE(wavefunction.batchExecutionPlan());
  CHECK_FALSE(component_ptr->boundPlan());
  wavefunction.prepareGroup(particles, 0);
  wavefunction.completeUpdates();
  TrialWaveFunction::mw_prepareGroup(wavefunctions, particle_sets, 0);
  TrialWaveFunction::mw_completeUpdates(wavefunctions);
  CHECK(component_ptr->prepareGroupCalls() == 1);
  CHECK(component_ptr->mwPrepareGroupCalls() == 1);
  CHECK(component_ptr->completeUpdatesCalls() == 1);
  CHECK(component_ptr->mwCompleteUpdatesCalls() == 1);
}

TEST_CASE("TrialWaveFunction planned lifecycle scans every batch lane before dispatch",
          "[wavefunction][batch_memory][lifecycle_guard]")
{
  constexpr const char* deferred_diagnostic =
      "TrialWaveFunction planned aggregate lifecycle ownership is deferred";

  RuntimeOptions runtime_options;
  TrialWaveFunction leader(runtime_options, "legacy-lifecycle-leader");
  auto leader_component = std::make_unique<PlanningComponent>(
      "LifecycleLane", "sole", BatchExecutionMode::VALUE,
      BatchTileCapacities{4, 0, 0, 0}, 17);
  PlanningComponent* leader_component_ptr = leader_component.get();
  leader.addComponent(std::move(leader_component));

  TrialWaveFunction planned_lane(runtime_options, "planned-lifecycle-lane");
  auto lane_component = std::make_unique<PlanningComponent>(
      "LifecycleLane", "sole", BatchExecutionMode::VALUE,
      BatchTileCapacities{4, 0, 0, 0}, 17);
  PlanningComponent* lane_component_ptr = lane_component.get();
  planned_lane.addComponent(std::move(lane_component));
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(
      planned_lane);
  const auto lane_plan =
      makePlan(planned_lane, "nonleader-lifecycle-guard-v1", 2);
  planned_lane.bindBatchExecutionPlan(lane_plan);

  const SimulationCell simulation_cell;
  ParticleSet leader_particles(simulation_cell);
  leader_particles.create({4});
  ParticleSet lane_particles(simulation_cell);
  lane_particles.create({4});
  leader_particles.G = QMCTraits::ValueType(0.25);
  leader_particles.L = QMCTraits::ValueType(-0.5);
  lane_particles.G = QMCTraits::ValueType(0.75);
  lane_particles.L = QMCTraits::ValueType(-1.0);
  const ParticleSet::ParticleGradient leader_gradients_before =
      leader_particles.G;
  const ParticleSet::ParticleLaplacian leader_laplacians_before =
      leader_particles.L;
  const ParticleSet::ParticleGradient lane_gradients_before =
      lane_particles.G;
  const ParticleSet::ParticleLaplacian lane_laplacians_before =
      lane_particles.L;

  RefVectorWithLeader<TrialWaveFunction> wavefunctions(
      leader, {leader, planned_lane});
  RefVectorWithLeader<ParticleSet> particle_sets(
      leader_particles, {leader_particles, lane_particles});

  const auto check_no_dispatch = [&]() {
    CHECK_FALSE(leader.batchExecutionPlan());
    CHECK(planned_lane.batchExecutionPlan().get() == lane_plan.get());
    CHECK(leader_component_ptr->prepareGroupCalls() == 0);
    CHECK(leader_component_ptr->mwPrepareGroupCalls() == 0);
    CHECK(leader_component_ptr->completeUpdatesCalls() == 0);
    CHECK(leader_component_ptr->mwCompleteUpdatesCalls() == 0);
    CHECK(lane_component_ptr->prepareGroupCalls() == 0);
    CHECK(lane_component_ptr->mwPrepareGroupCalls() == 0);
    CHECK(lane_component_ptr->completeUpdatesCalls() == 0);
    CHECK(lane_component_ptr->mwCompleteUpdatesCalls() == 0);
    for (std::size_t particle = 0; particle < leader_particles.G.size();
         ++particle)
    {
      for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
      {
        CHECK(leader_particles.G[particle][dimension] ==
              leader_gradients_before[particle][dimension]);
        CHECK(lane_particles.G[particle][dimension] ==
              lane_gradients_before[particle][dimension]);
      }
      CHECK(leader_particles.L[particle] ==
            leader_laplacians_before[particle]);
      CHECK(lane_particles.L[particle] == lane_laplacians_before[particle]);
    }
  };

  CHECK_THROWS_WITH(
      TrialWaveFunction::mw_prepareGroup(wavefunctions, particle_sets, 1),
      deferred_diagnostic);
  check_no_dispatch();
  CHECK_THROWS_WITH(TrialWaveFunction::mw_completeUpdates(wavefunctions),
                    deferred_diagnostic);
  check_no_dispatch();

  planned_lane.bindBatchExecutionPlan(nullptr);
  CHECK_FALSE(planned_lane.batchExecutionPlan());
  CHECK_FALSE(lane_component_ptr->boundPlan());
  TrialWaveFunction::mw_prepareGroup(wavefunctions, particle_sets, 1);
  TrialWaveFunction::mw_completeUpdates(wavefunctions);
  CHECK(leader_component_ptr->mwPrepareGroupCalls() == 1);
  CHECK(leader_component_ptr->mwCompleteUpdatesCalls() == 1);
  CHECK(lane_component_ptr->mwPrepareGroupCalls() == 0);
  CHECK(lane_component_ptr->mwCompleteUpdatesCalls() == 0);
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
  input.target_coordinate                 = BatchExecutionTargetCoordinate::POS_ONLY;
  input.logical_maximum = wavefunction.batchExecutionLogicalMaximum(
      {input.requirements, input.topology, input.particle_count,
       input.active_parameter_count, input.parameter_derivative_width, input.target_coordinate});
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
  input.target_coordinate                 = BatchExecutionTargetCoordinate::POS_ONLY;
  input.logical_maximum = wavefunction.batchExecutionLogicalMaximum(
      {input.requirements, input.topology, input.particle_count, 0, 0, input.target_coordinate});
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
  input.target_coordinate                 = BatchExecutionTargetCoordinate::POS_ONLY;
  input.logical_maximum = wavefunction.batchExecutionLogicalMaximum(
      {input.requirements, input.topology, input.particle_count, 0, 0, input.target_coordinate});
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

TEST_CASE("TrialWaveFunction prepares exact aggregate clone storage",
          "[wavefunction][batch_memory]")
{
  constexpr std::size_t particle_count = 4;
  RuntimeOptions runtime_options;
  TrialWaveFunction resident(runtime_options, "aggregate-clone-storage");
  auto component = std::make_unique<PlanningComponent>(
      "Prepared", "component", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 23);
  PlanningComponent* component_ptr = component.get();
  resident.addComponent(std::move(component));
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(resident);

  const auto plan = makePlan(resident, "aggregate-clone-v1", 3, particle_count);
  resident.bindBatchExecutionPlan(plan);
  const auto empty =
      testing::TestTrialWaveFunction::aggregateCloneDiagnostics(resident);
  CHECK_FALSE(empty.prepared);
  CHECK(empty.accepted_gradient_capacity == 0);
  CHECK(empty.accepted_laplacian_capacity == 0);
  CHECK(empty.proposed_gradient_capacity == 0);
  CHECK(empty.proposed_laplacian_capacity == 0);

  // A child failure may retain only bounded aggregate high water.  It publishes
  // no marker, and retry reuses all four allocations before publishing last.
  component_ptr->throwOnPrepare(true);
  CHECK_THROWS_AS(resident.prepareBatchExecutionClones(), std::runtime_error);
  const auto partial =
      testing::TestTrialWaveFunction::aggregateCloneDiagnostics(resident);
  CHECK_FALSE(partial.prepared);
  CHECK(partial.accepted_gradient_size == particle_count);
  CHECK(partial.accepted_gradient_capacity == particle_count);
  CHECK(partial.accepted_laplacian_size == particle_count);
  CHECK(partial.accepted_laplacian_capacity == particle_count);
  CHECK(partial.proposed_gradient_size == particle_count);
  CHECK(partial.proposed_gradient_capacity == particle_count);
  CHECK(partial.proposed_laplacian_size == particle_count);
  CHECK(partial.proposed_laplacian_capacity == particle_count);

  component_ptr->throwOnPrepare(false);
  resident.prepareBatchExecutionClones();
  const auto prepared =
      testing::TestTrialWaveFunction::aggregateCloneDiagnostics(resident);
  REQUIRE(prepared.prepared);
  CHECK(prepared.plan_identity == plan.get());
  CHECK(prepared.storage_shape_matches_plan);
  CHECK(prepared.allocation_identity_matches);
  CHECK(prepared.accepted_gradient_data == partial.accepted_gradient_data);
  CHECK(prepared.accepted_laplacian_data == partial.accepted_laplacian_data);
  CHECK(prepared.proposed_gradient_data == partial.proposed_gradient_data);
  CHECK(prepared.proposed_laplacian_data == partial.proposed_laplacian_data);
  CHECK(component_ptr->prepareCalls() == 2);

  const TrialWaveFunctionMemoryTypeSizes type_sizes =
      makeTrialWaveFunctionMemoryTypeSizes<
          TrialWaveFunction::ValueType,
          ParticleSet::ParticleGradient::value_type,
          ParticleSet::ParticleLaplacian::value_type,
          std::reference_wrapper<WaveFunctionComponent>,
          std::reference_wrapper<ParticleSet::ParticleGradient>,
          std::reference_wrapper<ParticleSet::ParticleLaplacian>,
          TrialWaveFunction::ParameterDerivativeView,
          TrialWaveFunction::EvaluationStamp,
          unsigned char>();
  const TrialWaveFunctionCloneStorageRequirement expected =
      trialWaveFunctionCloneStorageRequirement(1, particle_count, type_sizes);
  CHECK(prepared.accepted_gradient_bytes == expected.accepted_gradients);
  CHECK(prepared.proposed_gradient_bytes == expected.proposed_gradients);
  CHECK(prepared.accepted_laplacian_bytes == expected.accepted_laplacians);
  CHECK(prepared.proposed_laplacian_bytes == expected.proposed_laplacians);

  // Repeating the same preparation is a strict no-op after exact capacity and
  // allocation-identity validation.
  resident.prepareBatchExecutionClones();
  const auto repeated =
      testing::TestTrialWaveFunction::aggregateCloneDiagnostics(resident);
  CHECK(component_ptr->prepareCalls() == 2);
  CHECK(repeated.accepted_gradient_data == prepared.accepted_gradient_data);
  CHECK(repeated.accepted_laplacian_data == prepared.accepted_laplacian_data);
  CHECK(repeated.proposed_gradient_data == prepared.proposed_gradient_data);
  CHECK(repeated.proposed_laplacian_data == prepared.proposed_laplacian_data);

  resident.G[0] = TrialWaveFunction::GradType(1.25);
  resident.L[0] = TrialWaveFunction::ValueType(2.5);
  const TrialWaveFunction::GradType accepted_gradient_sentinel = resident.G[0];
  const TrialWaveFunction::ValueType accepted_laplacian_sentinel = resident.L[0];

  SimulationCell simulation_cell;
  ParticleSet particles(simulation_cell);
  particles.create({static_cast<int>(particle_count)});
  std::unique_ptr<TrialWaveFunction> clone = resident.makeClone(particles);
  const auto clone_empty =
      testing::TestTrialWaveFunction::aggregateCloneDiagnostics(*clone);
  CHECK(clone->batchExecutionPlan().get() == plan.get());
  CHECK_FALSE(clone_empty.prepared);
  CHECK(clone_empty.accepted_gradient_capacity == 0);
  CHECK(clone_empty.accepted_laplacian_capacity == 0);
  CHECK(clone_empty.proposed_gradient_capacity == 0);
  CHECK(clone_empty.proposed_laplacian_capacity == 0);

  clone->prepareBatchExecutionClones();
  const auto clone_prepared =
      testing::TestTrialWaveFunction::aggregateCloneDiagnostics(*clone);
  CHECK(clone_prepared.prepared);
  CHECK(clone_prepared.storage_shape_matches_plan);
  CHECK(clone_prepared.allocation_identity_matches);
  CHECK(clone_prepared.accepted_gradient_data != prepared.accepted_gradient_data);
  CHECK(clone_prepared.accepted_laplacian_data != prepared.accepted_laplacian_data);
  CHECK(clone_prepared.proposed_gradient_data != prepared.proposed_gradient_data);
  CHECK(clone_prepared.proposed_laplacian_data != prepared.proposed_laplacian_data);

  // Null binding clears provenance but retains fixed same-shape high water.
  // Accepted state in particular is physical and must remain bit-for-bit intact.
  resident.bindBatchExecutionPlan(nullptr);
  const auto unbound =
      testing::TestTrialWaveFunction::aggregateCloneDiagnostics(resident);
  CHECK_FALSE(unbound.prepared);
  CHECK(unbound.accepted_gradient_data == prepared.accepted_gradient_data);
  CHECK(unbound.accepted_laplacian_data == prepared.accepted_laplacian_data);
  CHECK(unbound.proposed_gradient_data == prepared.proposed_gradient_data);
  CHECK(unbound.proposed_laplacian_data == prepared.proposed_laplacian_data);
  CHECK(resident.G[0] == accepted_gradient_sentinel);
  CHECK(resident.L[0] == accepted_laplacian_sentinel);

  const auto replanned =
      makePlan(resident, "aggregate-clone-v2", 2, particle_count);
  resident.bindBatchExecutionPlan(replanned);
  resident.prepareBatchExecutionClones();
  const auto after_replan =
      testing::TestTrialWaveFunction::aggregateCloneDiagnostics(resident);
  CHECK(after_replan.prepared);
  CHECK(after_replan.plan_identity == replanned.get());
  CHECK(after_replan.storage_shape_matches_plan);
  CHECK(after_replan.allocation_identity_matches);
  CHECK(after_replan.accepted_gradient_data == prepared.accepted_gradient_data);
  CHECK(after_replan.accepted_laplacian_data == prepared.accepted_laplacian_data);
  CHECK(after_replan.proposed_gradient_data == prepared.proposed_gradient_data);
  CHECK(after_replan.proposed_laplacian_data == prepared.proposed_laplacian_data);
  CHECK(resident.G[0] == accepted_gradient_sentinel);
  CHECK(resident.L[0] == accepted_laplacian_sentinel);
  CHECK(component_ptr->prepareCalls() == 3);
}

TEST_CASE("TrialWaveFunction aggregate clone preparation guards lifecycle and identity",
          "[wavefunction][batch_memory][resources]")
{
  constexpr std::size_t particle_count = 4;
  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "aggregate-clone-guards");
  wavefunction.addComponent(std::make_unique<PlanningComponent>(
      "Guarded", "component", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 19));
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(wavefunction);
  const auto plan = makePlan(wavefunction, "aggregate-guards-v1", 3, particle_count);
  const auto other_plan =
      makePlan(wavefunction, "aggregate-guards-v2", 2, particle_count);
  wavefunction.bindBatchExecutionPlan(plan);

  CHECK_THROWS_AS(
      testing::TestTrialWaveFunction::prepareAggregateClone(
          wavefunction,
          testing::TestTrialWaveFunction::soleComponentPlan(wavefunction)),
      std::logic_error);
  CHECK_THROWS_AS(
      testing::TestTrialWaveFunction::prepareAggregateClone(
          wavefunction,
          makeBatchExecutionParticipantPlan(
              other_plan, TRIAL_WAVEFUNCTION_MEMORY_PARTICIPANT_ID)),
      std::logic_error);
  CHECK_FALSE(testing::TestTrialWaveFunction::aggregateCloneDiagnostics(
                  wavefunction).prepared);

  testing::TestTrialWaveFunction::setMultiParticleProposalPending(
      wavefunction, true);
  CHECK_THROWS_AS(wavefunction.prepareBatchExecutionClones(), std::logic_error);
  CHECK_THROWS_AS(wavefunction.bindBatchExecutionPlan(nullptr), std::logic_error);
  CHECK(wavefunction.batchExecutionPlan().get() == plan.get());
  testing::TestTrialWaveFunction::setMultiParticleProposalPending(
      wavefunction, false);

  RefVectorWithLeader<TrialWaveFunction> wavefunctions(wavefunction);
  ResourceCollection resources("aggregate-clone-preparation-guard");
  wavefunction.prepareBatchExecutionClones();
  wavefunction.createResource(resources);
  resources.prepareBatchResources({plan, 0});
  TrialWaveFunction::acquireResource(resources, wavefunctions);
  CHECK_THROWS_AS(wavefunction.prepareBatchExecutionClones(), std::logic_error);
  TrialWaveFunction::releaseResource(resources, wavefunctions);

  // Logical size alone is insufficient: a shrink retains five allocated
  // entries and must be rejected against the exact four-particle descriptor.
  wavefunction.G.resize(particle_count + 1);
  wavefunction.G.resize(particle_count);
  wavefunction.L.resize(particle_count);
  CHECK_THROWS_AS(wavefunction.prepareBatchExecutionClones(), std::length_error);
  const auto over_capacity =
      testing::TestTrialWaveFunction::aggregateCloneDiagnostics(wavefunction);
  CHECK(over_capacity.prepared);
  CHECK(over_capacity.accepted_gradient_size == particle_count);
  CHECK(over_capacity.accepted_gradient_capacity == particle_count + 1);
  CHECK(over_capacity.proposed_gradient_capacity == particle_count);
  CHECK(planningComponent(wavefunction, 0).prepareCalls() == 1);

  TrialWaveFunction prepared_capacity_guard(
      runtime_options, "prepared-capacity-guard");
  prepared_capacity_guard.addComponent(std::make_unique<PlanningComponent>(
      "PreparedGuard", "", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 11));
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(
      prepared_capacity_guard);
  prepared_capacity_guard.bindBatchExecutionPlan(
      makePlan(prepared_capacity_guard, "prepared-capacity", 2,
               particle_count));
  prepared_capacity_guard.prepareBatchExecutionClones();
  prepared_capacity_guard.G.resize(particle_count + 1);
  prepared_capacity_guard.G.resize(particle_count);
  CHECK_THROWS_AS(prepared_capacity_guard.prepareBatchExecutionClones(),
                  std::length_error);
  const auto corrupted_prepared =
      testing::TestTrialWaveFunction::aggregateCloneDiagnostics(
          prepared_capacity_guard);
  CHECK(corrupted_prepared.prepared);
  CHECK_FALSE(corrupted_prepared.storage_shape_matches_plan);
  CHECK(planningComponent(prepared_capacity_guard, 0).prepareCalls() == 1);

  TrialWaveFunction attached_guard(runtime_options, "attached-storage-guard");
  attached_guard.addComponent(std::make_unique<PlanningComponent>(
      "AttachedGuard", "", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 13));
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(
      attached_guard);
  attached_guard.bindBatchExecutionPlan(
      makePlan(attached_guard, "attached-storage", 2, particle_count));
  ParticleSet::ParticleGradient::value_type external_gradient[particle_count] = {};
  attached_guard.G.attachReference(external_gradient, particle_count);
  CHECK_THROWS_AS(attached_guard.prepareBatchExecutionClones(),
                  std::logic_error);
  CHECK(planningComponent(attached_guard, 0).prepareCalls() == 0);

  TrialWaveFunction zero_shape(runtime_options, "zero-particle-plan");
  zero_shape.addComponent(std::make_unique<PlanningComponent>(
      "Zero", "", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 7));
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(zero_shape);
  CHECK_THROWS_AS(makePlan(zero_shape, "zero-particle", 2, 0),
                  std::invalid_argument);
}

TEST_CASE("TrialWaveFunction propagates plan identity and defers clone preparation",
          "[wavefunction][batch_memory][resources]")
{
  RuntimeOptions runtime_options;
  SimulationCell simulation_cell;
  ParticleSet particles(simulation_cell);
  particles.create({4});

  TrialWaveFunction leader(runtime_options, "clone");
  leader.addComponent(std::make_unique<PlanningComponent>(
      "Cloneable", "component", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 23));
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(leader);
  const auto plan = makePlan(leader);
  leader.bindBatchExecutionPlan(plan);
  CHECK_THROWS_AS(leader.getOrCreateTWFFastDerivWrapper(particles),
                  std::logic_error);

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
  leader.prepareBatchExecutionClones();
  leader.createResource(resources);
  resources.prepareBatchResources({plan, 0});

  clone_component.setClassName("mutated/clone");
  CHECK_THROWS_AS(
      TrialWaveFunction::acquireResource(resources, wavefunctions),
      std::logic_error);
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
  clone->bindBatchExecutionPlan(nullptr);
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
  first_ptr->useResource();
  wavefunction.addComponent(std::move(first));
  auto second = std::make_unique<PlanningComponent>(
      "Second", "", BatchExecutionMode::FULL_VGL,
      BatchTileCapacities{0, 6, 0, 0}, 7);
  PlanningComponent* second_ptr = second.get();
  second_ptr->useResource();
  second_ptr->throwOnAcquire(true);
  wavefunction.addComponent(std::move(second));

  // Multiple components remain supported without a hard plan while the
  // aggregate planner deliberately rejects that topology.
  REQUIRE_FALSE(wavefunction.batchExecutionPlan());

  RefVectorWithLeader<TrialWaveFunction> wavefunctions(
      wavefunction, {wavefunction});
  ResourceCollection resources("batch-planning-rollback");
  wavefunction.createResource(resources);
  // Omitting a hard plan preserves the historical child-only template:
  // there is no aggregate prefix and both component resources retain order.
  REQUIRE(resources.size() == 2);
  auto first_resource = resources.lendResource<DummyResource>();
  auto second_resource = resources.lendResource<DummyResource>();
  CHECK(first_resource.getResource().getName() == "PlanningComponentResource");
  CHECK(second_resource.getResource().getName() == "PlanningComponentResource");
  resources.rewind();
  resources.takebackResource(first_resource);
  resources.takebackResource(second_resource);
  resources.rewind();
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
  resources.rewind();
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
  wavefunction.prepareBatchExecutionClones();
  wavefunction.createResource(resources);
  resources.prepareBatchResources({wavefunction.batchExecutionPlan(), 0});

  SECTION("one object may occupy multiple batch lanes")
  {
    RefVectorWithLeader<TrialWaveFunction> wavefunctions(
        wavefunction, {wavefunction, wavefunction});
    TrialWaveFunction::acquireResource(resources, wavefunctions);
    CHECK(wavefunction.hasAcquiredResource());
    CHECK(component_ptr->acquireCalls() == 1);
    CHECK(resources.getOutstandingLoanCount() == 1);
    CHECK(resources.getCursor() == resources.size());
    const auto first_resource =
        testing::TestTrialWaveFunction::aggregateResourceDiagnostics(
            wavefunction);
    REQUIRE(first_resource.prepared);
    CHECK(first_resource.plan_identity ==
          wavefunction.batchExecutionPlan().get());
    CHECK(first_resource.reserve_walkers == 3);
    CHECK(first_resource.expected_bytes == first_resource.actual_bytes);
    CHECK(first_resource.component_reference_bytes ==
          3 * sizeof(std::reference_wrapper<WaveFunctionComponent>));
    CHECK(first_resource.gradient_reference_bytes ==
          3 * sizeof(std::reference_wrapper<ParticleSet::ParticleGradient>));
    CHECK(first_resource.laplacian_reference_bytes ==
          3 * sizeof(std::reference_wrapper<ParticleSet::ParticleLaplacian>));
    CHECK(first_resource.transaction_flag_bytes == 3);
    CHECK(first_resource.private_ratio_bytes == 0);
    CHECK(first_resource.derivative_delta_bytes == 0);
    CHECK(first_resource.component_leader == component_ptr);
    CHECK(first_resource.first_component == component_ptr);
    CHECK(first_resource.gradient_leader == &wavefunction.G);
    CHECK(first_resource.first_gradient == &wavefunction.G);
    CHECK(first_resource.laplacian_leader == &wavefunction.L);
    CHECK(first_resource.first_laplacian == &wavefunction.L);

    TrialWaveFunction::releaseResource(resources, wavefunctions);
    CHECK_FALSE(wavefunction.hasAcquiredResource());
    CHECK(component_ptr->releaseCalls() == 1);
    CHECK(resources.getOutstandingLoanCount() == 0);
    CHECK(resources.getCursor() == resources.size());

    resources.rewind();
    TrialWaveFunction::acquireResource(resources, wavefunctions);
    const auto second_resource =
        testing::TestTrialWaveFunction::aggregateResourceDiagnostics(
            wavefunction);
    CHECK(second_resource.storage_fingerprint ==
          first_resource.storage_fingerprint);
    CHECK(second_resource.component_reference_data ==
          first_resource.component_reference_data);
    CHECK(second_resource.gradient_reference_data ==
          first_resource.gradient_reference_data);
    CHECK(second_resource.laplacian_reference_data ==
          first_resource.laplacian_reference_data);
    TrialWaveFunction::releaseResource(resources, wavefunctions);

    resources.rewind();
    RefVectorWithLeader<TrialWaveFunction> full_reserve(
        wavefunction, {wavefunction, wavefunction, wavefunction});
    TrialWaveFunction::acquireResource(resources, full_reserve);
    const auto full_resource =
        testing::TestTrialWaveFunction::aggregateResourceDiagnostics(
            wavefunction);
    CHECK(full_resource.storage_fingerprint ==
          first_resource.storage_fingerprint);
    CHECK(full_resource.component_reference_data ==
          first_resource.component_reference_data);
    TrialWaveFunction::releaseResource(resources, full_reserve);
  }

  SECTION("an empty lane vector still manages its designated leader")
  {
    RefVectorWithLeader<TrialWaveFunction> wavefunctions(wavefunction);
    TrialWaveFunction::acquireResource(resources, wavefunctions);
    CHECK(wavefunction.hasAcquiredResource());
    CHECK(component_ptr->acquireCalls() == 1);
    const auto diagnostics =
        testing::TestTrialWaveFunction::aggregateResourceDiagnostics(
            wavefunction);
    CHECK(diagnostics.component_leader == component_ptr);
    CHECK(diagnostics.first_component == nullptr);

    TrialWaveFunction::releaseResource(resources, wavefunctions);
    CHECK_FALSE(wavefunction.hasAcquiredResource());
    CHECK(component_ptr->releaseCalls() == 1);
  }
}

TEST_CASE("TrialWaveFunction prepares exact aggregate resources for uneven and zero crowds",
          "[wavefunction][batch_memory][resources]")
{
  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "aggregate-crowd-shapes");
  wavefunction.addComponent(std::make_unique<PlanningComponent>(
      "CrowdShape", "", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 7));
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(wavefunction);

  const auto uneven_plan = makePlan(wavefunction, "uneven-crowds", 3, 4,
                                    {1, 2}, {2, 4});
  wavefunction.bindBatchExecutionPlan(uneven_plan);
  wavefunction.prepareBatchExecutionClones();
  ResourceCollection resources("uneven-aggregate-resource");
  wavefunction.createResource(resources);
  CHECK(resources.size() == 1);
  resources.prepareBatchResources({uneven_plan, 1});
  CHECK(testing::TestTrialWaveFunction::aggregatePlaceholdersMatchFillers(
      resources));

  RefVectorWithLeader<TrialWaveFunction> two_lanes(
      wavefunction, {wavefunction, wavefunction});
  TrialWaveFunction::acquireResource(resources, two_lanes);
  const auto uneven =
      testing::TestTrialWaveFunction::aggregateResourceDiagnostics(wavefunction);
  REQUIRE(uneven.prepared);
  CHECK(uneven.crowd_index == 1);
  CHECK(uneven.reserve_walkers == 4);
  CHECK(uneven.expected_bytes == uneven.actual_bytes);
  CHECK(uneven.component_reference_bytes ==
        4 * sizeof(std::reference_wrapper<WaveFunctionComponent>));
  CHECK(uneven.gradient_reference_bytes ==
        4 * sizeof(std::reference_wrapper<ParticleSet::ParticleGradient>));
  CHECK(uneven.laplacian_reference_bytes ==
        4 * sizeof(std::reference_wrapper<ParticleSet::ParticleLaplacian>));
  CHECK(uneven.transaction_flag_bytes == 4);
  TrialWaveFunction::releaseResource(resources, two_lanes);

  // Copies of prepared collections preserve provenance but intentionally copy
  // no prepared storage.  An explicit null clear makes the copy reusable.
  ResourceCollection derived(resources);
  CHECK(derived.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::DERIVED_REQUIRES_CLEAR);
  CHECK_FALSE(testing::TestTrialWaveFunction::aggregatePlaceholdersMatchFillers(
      derived));
  CHECK_THROWS_AS(TrialWaveFunction::acquireResource(derived, two_lanes),
                  std::logic_error);
  CHECK(derived.getOutstandingLoanCount() == 0);
  derived.prepareBatchResources({nullptr, 0});
  derived.prepareBatchResources({uneven_plan, 0});
  CHECK(testing::TestTrialWaveFunction::aggregatePlaceholdersMatchFillers(
      derived));
  TrialWaveFunction::acquireResource(derived, two_lanes);
  const auto copied =
      testing::TestTrialWaveFunction::aggregateResourceDiagnostics(wavefunction);
  CHECK(copied.crowd_index == 0);
  CHECK(copied.reserve_walkers == 2);
  TrialWaveFunction::releaseResource(derived, two_lanes);

  // A zero-reserve crowd still carries exact plan/crowd provenance while all
  // category backing remains empty and its fingerprint stays valid.
  wavefunction.bindBatchExecutionPlan(nullptr);
  const auto zero_plan = makePlan(wavefunction, "zero-crowd", 1, 4,
                                  {0}, {0});
  wavefunction.bindBatchExecutionPlan(zero_plan);
  wavefunction.prepareBatchExecutionClones();
  resources.prepareBatchResources({nullptr, 0});
  resources.prepareBatchResources({zero_plan, 0});
  CHECK(testing::TestTrialWaveFunction::aggregatePlaceholdersMatchFillers(
      resources));
  RefVectorWithLeader<TrialWaveFunction> empty_lanes(wavefunction);
  TrialWaveFunction::acquireResource(resources, empty_lanes);
  const auto zero =
      testing::TestTrialWaveFunction::aggregateResourceDiagnostics(wavefunction);
  REQUIRE(zero.prepared);
  CHECK(zero.reserve_walkers == 0);
  CHECK(zero.expected_bytes == 0);
  CHECK(zero.actual_bytes == 0);
  CHECK(zero.component_reference_bytes == 0);
  CHECK(zero.gradient_reference_bytes == 0);
  CHECK(zero.laplacian_reference_bytes == 0);
  CHECK(zero.transaction_flag_bytes == 0);
  CHECK(zero.storage_fingerprint != 0);
  TrialWaveFunction::releaseResource(resources, empty_lanes);
}

TEST_CASE("TrialWaveFunction prepares exact weighted aggregate staging",
          "[wavefunction][batch_memory][resources]")
{
  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "weighted-aggregate-resource");
  wavefunction.addComponent(std::make_unique<PlanningComponent>(
      "Weighted", "", BatchExecutionMode::ECP_WEIGHTED_SCORE,
      BatchTileCapacities{0, 0, 0, 8}, 13));
  planningComponent(wavefunction, 0).requireValueMode();
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(wavefunction);
  const auto plan = makePlan(wavefunction, "weighted-aggregate", 1, 4,
                             {2}, {3}, 2);
  wavefunction.bindBatchExecutionPlan(plan);
  wavefunction.prepareBatchExecutionClones();

  ResourceCollection resources("weighted-aggregate-resource");
  wavefunction.createResource(resources);
  resources.prepareBatchResources({plan, 0});
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(
      wavefunction, {wavefunction});
  TrialWaveFunction::acquireResource(resources, wavefunctions);
  const auto diagnostics =
      testing::TestTrialWaveFunction::aggregateResourceDiagnostics(wavefunction);
  const std::size_t reserve = 3;
  const std::size_t outer = plan->selectedCapacities().ecp_outer;
  const std::size_t derivative_width = plan->parameterDerivativeWidth();
  REQUIRE(outer > 0);
  CHECK(diagnostics.expected_bytes == diagnostics.actual_bytes);
  CHECK(diagnostics.private_ratio_bytes == outer * sizeof(TrialWaveFunction::ValueType));
  CHECK(diagnostics.total_weight_bytes == outer * sizeof(TrialWaveFunction::ValueType));
  CHECK(diagnostics.derivative_delta_bytes ==
        reserve * derivative_width * sizeof(TrialWaveFunction::ValueType));
  CHECK(diagnostics.derivative_view_bytes ==
        reserve * sizeof(TrialWaveFunction::ParameterDerivativeView));
  CHECK(diagnostics.value_stamp_bytes ==
        sizeof(TrialWaveFunction::EvaluationStamp));
  CHECK(diagnostics.transaction_flag_bytes == reserve);
  TrialWaveFunction::releaseResource(resources, wavefunctions);
}

TEST_CASE("TrialWaveFunction planned aggregate acquisition is transactional",
          "[wavefunction][batch_memory][resources]")
{
  RuntimeOptions runtime_options;
  SimulationCell simulation_cell;
  ParticleSet particles(simulation_cell);
  particles.create({4});

  TrialWaveFunction leader(runtime_options, "aggregate-transaction");
  leader.addComponent(std::make_unique<PlanningComponent>(
      "Transactional", "", BatchExecutionMode::VALUE,
      BatchTileCapacities{8, 0, 0, 0}, 11));
  planningComponent(leader, 0).useResource();
  testing::TestTrialWaveFunction::useCompleteBatchMemoryAccounting(leader);
  const auto plan = makePlan(leader, "aggregate-transaction-v1");
  leader.bindBatchExecutionPlan(plan);
  leader.prepareBatchExecutionClones();
  auto clone = leader.makeClone(particles);
  clone->prepareBatchExecutionClones();
  auto second_clone = leader.makeClone(particles);
  second_clone->prepareBatchExecutionClones();
  auto unprepared = leader.makeClone(particles);

  ResourceCollection resources("aggregate-transaction-resource");
  leader.createResource(resources);
  resources.addResource(std::make_unique<DummyResource>("trailing-resource"));
  CHECK(resources.size() == 3);
  resources.prepareBatchResources({plan, 0});
  const std::size_t entry_cursor = resources.getCursor();

  RefVectorWithLeader<TrialWaveFunction> too_many(
      leader, {leader, leader, leader, leader});
  CHECK_THROWS_AS(TrialWaveFunction::acquireResource(resources, too_many),
                  std::length_error);
  CHECK(resources.getOutstandingLoanCount() == 0);
  CHECK(resources.getCursor() == entry_cursor);

  RefVectorWithLeader<TrialWaveFunction> unprepared_team(leader, {*unprepared});
  CHECK_THROWS_AS(
      TrialWaveFunction::acquireResource(resources, unprepared_team),
      std::logic_error);
  CHECK(resources.getOutstandingLoanCount() == 0);
  CHECK_FALSE(leader.hasAcquiredResource());
  CHECK_FALSE(unprepared->hasAcquiredResource());

  RefVectorWithLeader<TrialWaveFunction> team(leader, {*clone});
  testing::TestTrialWaveFunction::setMultiParticleProposalPending(*clone, true);
  CHECK_THROWS_AS(TrialWaveFunction::acquireResource(resources, team),
                  std::logic_error);
  testing::TestTrialWaveFunction::setMultiParticleProposalPending(*clone, false);
  CHECK(resources.getOutstandingLoanCount() == 0);

  testing::TestTrialWaveFunction::setAggregateResourceCrowd(resources, 1);
  CHECK_THROWS_AS(TrialWaveFunction::acquireResource(resources, team),
                  std::logic_error);
  CHECK(resources.getOutstandingLoanCount() == 0);
  CHECK(resources.getCursor() == entry_cursor);
  testing::TestTrialWaveFunction::setAggregateResourceCrowd(resources, 0);

  PlanningComponent& component = planningComponent(leader, 0);
  component.throwOnAcquire(true);
  CHECK_THROWS_AS(TrialWaveFunction::acquireResource(resources, team),
                  std::runtime_error);
  CHECK(resources.getOutstandingLoanCount() == 0);
  CHECK(resources.getCursor() == entry_cursor);
  CHECK_FALSE(leader.hasAcquiredResource());
  CHECK_FALSE(clone->hasAcquiredResource());

  // The same collection remains usable after rollback of an aggregate-first
  // loan whose child acquisition failed.
  component.throwOnAcquire(false);
  RefVectorWithLeader<TrialWaveFunction> exact_team(
      leader, {*clone, *second_clone});
  TrialWaveFunction::acquireResource(resources, exact_team);
  CHECK(resources.getOutstandingLoanCount() == 2);
  CHECK(resources.getCursor() == 2);

  ResourceCollection wrong_collection("wrong-release-collection");
  CHECK_THROWS_AS(
      TrialWaveFunction::releaseResource(wrong_collection, exact_team),
      std::logic_error);

  auto trailing = resources.lendResource<DummyResource>();
  CHECK_THROWS_AS(TrialWaveFunction::releaseResource(resources, exact_team),
                  std::logic_error);
  resources.rewind(2);
  resources.takebackResource(trailing);
  CHECK(resources.getOutstandingLoanCount() == 2);

  RefVectorWithLeader<TrialWaveFunction> subset(leader, {*clone});
  CHECK_THROWS_AS(TrialWaveFunction::releaseResource(resources, subset),
                  std::invalid_argument);
  RefVectorWithLeader<TrialWaveFunction> reordered(
      leader, {*second_clone, *clone});
  CHECK_THROWS_AS(TrialWaveFunction::releaseResource(resources, reordered),
                  std::invalid_argument);
  RefVectorWithLeader<TrialWaveFunction> wrong_lane(
      leader, {*clone, *unprepared});
  CHECK_THROWS_AS(TrialWaveFunction::releaseResource(resources, wrong_lane),
                  std::logic_error);
  CHECK(resources.getOutstandingLoanCount() == 2);
  CHECK(leader.hasAcquiredResource());
  CHECK(clone->hasAcquiredResource());
  CHECK(second_clone->hasAcquiredResource());

  component.skipResourceRelease(true);
  CHECK_THROWS_AS(TrialWaveFunction::releaseResource(resources, exact_team),
                  std::logic_error);
  CHECK(resources.getOutstandingLoanCount() == 2);
  CHECK(leader.hasAcquiredResource());
  component.skipResourceRelease(false);
  TrialWaveFunction::releaseResource(resources, exact_team);
  CHECK(resources.getOutstandingLoanCount() == 0);
  CHECK(resources.getCursor() == 2);
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
