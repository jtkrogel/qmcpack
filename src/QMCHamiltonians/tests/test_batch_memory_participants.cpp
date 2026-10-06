//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: QMCPACK developers
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "Particle/SimulationCell.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
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

/** Small typed resource used to verify Hamiltonian acquisition rollback. */
class PlanningResource : public Resource
{
public:
  explicit PlanningResource(int id) : Resource("PlanningResource"), id_(id) {}

  std::unique_ptr<Resource> makeClone() const override
  {
    return std::make_unique<PlanningResource>(*this);
  }

  int id() const noexcept { return id_; }

private:
  int id_;
};

/** Configurable operator exposing the generic batch-planning and resource hooks. */
class PlanningOperator : public OperatorDependsOnlyOnParticleSet
{
public:
  PlanningOperator(int id, std::string class_name, std::vector<int>& release_order)
      : id_(id), class_name_(std::move(class_name)), release_order_(&release_order)
  {}

  Return_t evaluate(ParticleSet&) override { return 0.0; }

  bool put(xmlNodePtr) override { return true; }

  bool get(std::ostream& output) const override
  {
    output << class_name_;
    return true;
  }

  std::string getClassName() const override { return class_name_; }

  std::unique_ptr<OperatorBase> makeClone(ParticleSet&) const override
  {
    auto clone = std::make_unique<PlanningOperator>(id_, class_name_, *release_order_);
    clone->host_bytes_per_value_ = host_bytes_per_value_;
    clone->logical_maximum_      = logical_maximum_;
    clone->requirements_         = requirements_;
    clone->throw_on_validation_  = throw_on_validation_;
    clone->throw_on_prepare_     = throw_on_prepare_;
    clone->throw_on_acquire_     = throw_on_acquire_;
    return clone;
  }

  void contributeBatchExecutionRequirements(BatchExecutionRequirements& requirements) const override
  {
    if (requirements_.requires(BatchExecutionMode::VALUE))
      requirements.require(BatchExecutionMode::VALUE);
    if (requirements_.requires(BatchExecutionMode::SCORE))
      requirements.require(BatchExecutionMode::SCORE);
  }

  BatchTileCapacities batchExecutionLogicalMaximum(
      const BatchExecutionWorkloadContext&) const override
  {
    return logical_maximum_;
  }

  BatchMemoryContribution estimateBatchExecutionMemory(
      const BatchExecutionPlanningContext& context) const override
  {
    BatchMemoryContribution contribution;
    contribution.logical_maximum    = logical_maximum_;
    contribution.owner_multiplicity = 1;
    contribution.fully_accounted    = true;
    contribution.per_owner.add(
        BatchMemoryCategory::INNER_TILE_SCRATCH,
        {checkedBatchMemoryMultiply(host_bytes_per_value_,
                                    context.candidate_capacities.value,
                                    "planning test value bytes"),
         0});
    return contribution;
  }

  void validateBatchExecutionPlanBinding(
      const BatchExecutionParticipantPlan&) const override
  {
    ++validation_count_;
    if (throw_on_validation_)
      throw std::runtime_error("deliberate Hamiltonian plan-validation failure");
  }

  void bindBatchExecutionPlan(
      BatchExecutionParticipantPlan participant_plan) noexcept override
  {
    ++binding_count_;
    participant_plan_ = std::move(participant_plan);
  }

  void prepareBatchExecutionClone(
      const BatchExecutionParticipantPlan& participant_plan) override
  {
    ++prepare_count_;
    if (!participant_plan_.sameBinding(participant_plan))
      throw std::logic_error("operator prepared with a different participant plan");
    if (throw_on_prepare_)
      throw std::runtime_error("deliberate Hamiltonian preparation failure");
  }

  void createResource(ResourceCollection& collection) const override
  {
    collection.addResource(std::make_unique<PlanningResource>(id_));
  }

  void acquireResource(ResourceCollection& collection,
                       const RefVectorWithLeader<OperatorBase>& operators) const override
  {
    auto& leader = operators.getCastedLeader<PlanningOperator>();
    if (leader.throw_on_acquire_)
      throw std::runtime_error("deliberate Hamiltonian resource-acquisition failure");
    leader.resource_handle_ = collection.lendResource<PlanningResource>();
  }

  void releaseResource(ResourceCollection& collection,
                       const RefVectorWithLeader<OperatorBase>& operators) const override
  {
    auto& leader          = operators.getCastedLeader<PlanningOperator>();
    const int resource_id = leader.resource_handle_.getResource().id();
    collection.takebackResource(leader.resource_handle_);
    leader.release_order_->push_back(resource_id);
  }

  void require(BatchExecutionMode mode) { requirements_.require(mode); }
  void setHostBytesPerValue(std::size_t bytes) noexcept { host_bytes_per_value_ = bytes; }
  void setLogicalMaximum(BatchTileCapacities maximum) noexcept { logical_maximum_ = maximum; }
  void setThrowOnValidation(bool should_throw) noexcept { throw_on_validation_ = should_throw; }
  void setThrowOnPrepare(bool should_throw) noexcept { throw_on_prepare_ = should_throw; }
  void setThrowOnAcquire(bool should_throw) noexcept { throw_on_acquire_ = should_throw; }

  bool hasPlan() const noexcept { return participant_plan_.hasPlan(); }
  bool hasResource() const noexcept { return resource_handle_.hasResource(); }
  const std::string& participantId() const { return participant_plan_.evidence().participant_id; }
  int validationCount() const noexcept { return validation_count_; }
  int bindingCount() const noexcept { return binding_count_; }
  int prepareCount() const noexcept { return prepare_count_; }

private:
  int id_;
  std::string class_name_;
  std::vector<int>* release_order_;
  std::size_t host_bytes_per_value_ = 1;
  BatchTileCapacities logical_maximum_{4, 0, 0, 0};
  BatchExecutionRequirements requirements_;
  bool throw_on_validation_ = false;
  bool throw_on_prepare_    = false;
  bool throw_on_acquire_    = false;
  mutable int validation_count_ = 0;
  int binding_count_             = 0;
  int prepare_count_             = 0;
  BatchExecutionParticipantPlan participant_plan_;
  ResourceHandle<PlanningResource> resource_handle_;
};

/** Build an immutable plan directly from one Hamiltonian's participants. */
std::shared_ptr<const BatchExecutionPlan> makePlan(const QMCHamiltonian& hamiltonian,
                                                   std::size_t active_parameter_count = 0)
{
  BatchExecutionSelectionInput input;
  hamiltonian.contributeBatchExecutionRequirements(input.requirements);
  input.preference.preferred   = {3, 0, 0, 0};
  input.active_parameter_count = active_parameter_count;
  input.target_coordinate      = BatchExecutionTargetCoordinate::POS_ONLY;
  input.logical_maximum        = hamiltonian.batchExecutionLogicalMaximum(
      {input.requirements, input.topology, input.particle_count,
       input.active_parameter_count, input.parameter_derivative_width, input.target_coordinate});
  auto provider = [&hamiltonian](const BatchExecutionPlanningContext& context) {
    return hamiltonian.estimateBatchExecutionMemory(context);
  };
  return std::make_shared<const BatchExecutionPlan>(
      selectBatchExecutionPlan(input, provider));
}

/** Add one configurable mock while retaining a non-owning test pointer. */
PlanningOperator* addPlanningOperator(QMCHamiltonian& hamiltonian,
                                      int id,
                                      const std::string& class_name,
                                      const std::string& name,
                                      bool physical,
                                      std::vector<int>& release_order)
{
  auto component = std::make_unique<PlanningOperator>(id, class_name, release_order);
  auto* result   = component.get();
  result->require(BatchExecutionMode::VALUE);
  hamiltonian.addOperator(std::move(component), name, physical);
  return result;
}

} // namespace

TEST_CASE("QMCHamiltonian batch participants bind atomically and clone plans",
          "[hamiltonian][batch_memory]")
{
  std::vector<int> release_order;
  QMCHamiltonian hamiltonian("planning-test");
  PlanningOperator* physical = addPlanningOperator(
      hamiltonian, 1, "Plan/Op", "phys/name%", true, release_order);
  PlanningOperator* auxiliary = addPlanningOperator(
      hamiltonian, 2, "Aux Op", "aux/name", false, release_order);
  physical->setHostBytesPerValue(3);
  auxiliary->setHostBytesPerValue(5);
  physical->setLogicalMaximum({5, 2, 0, 7});
  auxiliary->setLogicalMaximum({3, 6, 4, 1});
  auxiliary->require(BatchExecutionMode::SCORE);

  BatchExecutionRequirements requirements;
  hamiltonian.contributeBatchExecutionRequirements(requirements);
  CHECK(requirements.requires(BatchExecutionMode::VALUE));
  CHECK(requirements.requires(BatchExecutionMode::SCORE));

  BatchExecutionWorkloadContext workload_context{requirements, {}, 0, 17, 0};
  CHECK(hamiltonian.batchExecutionLogicalMaximum(workload_context) ==
        BatchTileCapacities{5, 6, 4, 7});

  BatchExecutionPlanningContext context{
      requirements, {}, {5, 6, 4, 7}, {2, 0, 0, 0}, 0, 17, 0};
  const auto contributions = hamiltonian.estimateBatchExecutionMemory(context);
  REQUIRE(contributions.size() == 2);
  CHECK(contributions[0].participant_id ==
        "ham/physical/operator/0/Plan%2FOp/phys%2Fname%25");
  CHECK(contributions[1].participant_id ==
        "ham/auxiliary/operator/0/Aux%20Op/aux%2Fname");
  CHECK(contributions[0].contribution.per_owner
            .at(BatchMemoryCategory::INNER_TILE_SCRATCH)
            .host == 6);
  CHECK(contributions[1].contribution.per_owner
            .at(BatchMemoryCategory::INNER_TILE_SCRATCH)
            .host == 10);

  const auto plan = makePlan(hamiltonian, 17);
  REQUIRE(plan->participantEvidence().size() == 2);
  CHECK(plan->activeParameterCount() == 17);

  auxiliary->setThrowOnValidation(true);
  CHECK_THROWS_WITH(hamiltonian.bindBatchExecutionPlan(plan),
                    "deliberate Hamiltonian plan-validation failure");
  CHECK_FALSE(physical->hasPlan());
  CHECK_FALSE(auxiliary->hasPlan());
  CHECK(physical->bindingCount() == 0);

  auxiliary->setThrowOnValidation(false);
  hamiltonian.bindBatchExecutionPlan(plan);
  REQUIRE(hamiltonian.batchExecutionPlan() == plan);
  CHECK(physical->participantId() == contributions[0].participant_id);
  CHECK(auxiliary->participantId() == contributions[1].participant_id);

  const int physical_bindings = physical->bindingCount();
  hamiltonian.bindBatchExecutionPlan(plan);
  CHECK(physical->bindingCount() == physical_bindings);

  hamiltonian.prepareBatchExecutionClones();
  CHECK(physical->prepareCount() == 1);
  CHECK(auxiliary->prepareCount() == 1);

  auto rejected = std::make_unique<PlanningOperator>(3, "late", release_order);
  CHECK_THROWS_AS(
      hamiltonian.addOperator(std::move(rejected), "late", true),
      std::logic_error);

  const SimulationCell simulation_cell;
  ParticleSet particles(simulation_cell);
  TrialWaveFunction wavefunction(RuntimeOptions{});
  auto clone = hamiltonian.makeClone(particles, wavefunction);
  REQUIRE(clone->batchExecutionPlan() == plan);
  auto& clone_physical = dynamic_cast<PlanningOperator&>(*clone->getComponent(0));
  CHECK(clone_physical.hasPlan());
  CHECK(clone_physical.participantId() == contributions[0].participant_id);

  hamiltonian.bindBatchExecutionPlan(nullptr);
  CHECK_FALSE(hamiltonian.batchExecutionPlan());
  CHECK_FALSE(physical->hasPlan());
  CHECK_FALSE(auxiliary->hasPlan());
  const int bindings_after_clear = physical->bindingCount();
  hamiltonian.bindBatchExecutionPlan(nullptr);
  CHECK(physical->bindingCount() == bindings_after_clear + 1);
}

TEST_CASE("QMCHamiltonian batch resources reject drift and unwind acquisition",
          "[hamiltonian][batch_memory][resources]")
{
  std::vector<int> release_order;
  QMCHamiltonian hamiltonian("resource-test");
  PlanningOperator* first = addPlanningOperator(
      hamiltonian, 1, "PlanningOperator", "first", true, release_order);
  PlanningOperator* second = addPlanningOperator(
      hamiltonian, 2, "PlanningOperator", "second", true, release_order);
  const auto plan = makePlan(hamiltonian);
  hamiltonian.bindBatchExecutionPlan(plan);

  RefVectorWithLeader<QMCHamiltonian> family(hamiltonian, {hamiltonian});
  ResourceCollection resources("Hamiltonian batch rollback");
  hamiltonian.createResource(resources);
  REQUIRE(resources.size() == 3);

  second->setThrowOnAcquire(true);
  CHECK_THROWS_WITH(QMCHamiltonian::acquireResource(resources, family),
                    "deliberate Hamiltonian resource-acquisition failure");
  CHECK(resources.getCursor() == 0);
  CHECK_FALSE(hamiltonian.hasAcquiredResource());
  CHECK_FALSE(first->hasResource());
  CHECK_FALSE(second->hasResource());
  CHECK(release_order == (std::vector<int>{1}));

  second->setThrowOnAcquire(false);
  release_order.clear();
  {
    ResourceCollectionTeamLock lock(resources, family);
    CHECK(hamiltonian.hasAcquiredResource());
    CHECK(first->hasResource());
    CHECK(second->hasResource());
    CHECK(resources.getCursor() == 3);
    CHECK_NOTHROW(hamiltonian.bindBatchExecutionPlan(plan));
    CHECK_THROWS_AS(hamiltonian.bindBatchExecutionPlan(nullptr),
                    std::logic_error);
    const auto equivalent_plan = makePlan(hamiltonian);
    CHECK(equivalent_plan->fingerprint() == plan->fingerprint());
    CHECK_THROWS_AS(hamiltonian.bindBatchExecutionPlan(equivalent_plan),
                    std::logic_error);
  }
  CHECK_FALSE(hamiltonian.hasAcquiredResource());
  CHECK_FALSE(first->hasResource());
  CHECK_FALSE(second->hasResource());
  CHECK(release_order == (std::vector<int>{1, 2}));

  first->setName("renamed-after-binding");
  CHECK_THROWS_AS(hamiltonian.prepareBatchExecutionClones(), std::logic_error);
  resources.rewind();
  CHECK_THROWS_AS(QMCHamiltonian::acquireResource(resources, family),
                  std::logic_error);
  CHECK(resources.getCursor() == 0);
  first->setName("first");

  first->getUpdateMode().set(OperatorBase::PHYSICAL, false);
  CHECK_THROWS_AS(hamiltonian.prepareBatchExecutionClones(), std::logic_error);
  first->getUpdateMode().set(OperatorBase::PHYSICAL, true);

  RefVectorWithLeader<QMCHamiltonian> duplicate_family(
      hamiltonian, {hamiltonian, hamiltonian});
  resources.rewind();
  {
    ResourceCollectionTeamLock lock(resources, duplicate_family);
    CHECK(hamiltonian.hasAcquiredResource());
  }
  CHECK_FALSE(hamiltonian.hasAcquiredResource());
}

TEST_CASE("QMCHamiltonian tracks an absent leader with a null batch plan",
          "[hamiltonian][batch_memory][resources]")
{
  std::vector<int> release_order;
  QMCHamiltonian leader("leader");
  addPlanningOperator(leader, 7, "PlanningOperator", "shared", true,
                      release_order);
  QMCHamiltonian follower("follower");
  addPlanningOperator(follower, 7, "PlanningOperator", "shared", true,
                      release_order);

  // RefVectorWithLeader permits the explicit leader to be absent from the
  // element vector.  Both aggregate lifecycles must still be published.
  RefVectorWithLeader<QMCHamiltonian> follower_only(leader, {follower});
  ResourceCollection resources("Hamiltonian absent leader");
  leader.createResource(resources);
  REQUIRE(resources.size() == 2);

  {
    ResourceCollectionTeamLock lock(resources, follower_only);
    CHECK(leader.hasAcquiredResource());
    CHECK(follower.hasAcquiredResource());
    CHECK_FALSE(leader.batchExecutionPlan());
    CHECK_NOTHROW(leader.bindBatchExecutionPlan(nullptr));

    auto late_component =
        std::make_unique<PlanningOperator>(8, "late", release_order);
    CHECK_THROWS_AS(
        leader.addOperator(std::move(late_component), "late", true),
        std::logic_error);

    auto& leader_component =
        dynamic_cast<PlanningOperator&>(*leader.getComponent(0));
    leader_component.setName("changed");
    CHECK_THROWS_AS(leader.bindBatchExecutionPlan(nullptr), std::logic_error);
    leader_component.setName("shared");
  }

  CHECK_FALSE(leader.hasAcquiredResource());
  CHECK_FALSE(follower.hasAcquiredResource());
  CHECK(release_order == (std::vector<int>{7}));

  // Preserve legacy leader-only and repeated-reference resource contracts when
  // no bounded plan requires distinct clone-local storage.
  RefVectorWithLeader<QMCHamiltonian> leader_only(leader);
  {
    ResourceCollectionTeamLock lock(resources, leader_only);
    CHECK(leader.hasAcquiredResource());
  }
  CHECK_FALSE(leader.hasAcquiredResource());

  RefVectorWithLeader<QMCHamiltonian> repeated_legacy(
      leader, {leader, leader});
  {
    ResourceCollectionTeamLock lock(resources, repeated_legacy);
    CHECK(leader.hasAcquiredResource());
  }
  CHECK_FALSE(leader.hasAcquiredResource());
  CHECK(release_order == (std::vector<int>{7, 7, 7}));
}

} // namespace qmcplusplus
