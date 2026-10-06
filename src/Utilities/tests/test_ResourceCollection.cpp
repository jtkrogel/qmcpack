//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <sstream>
#include "BatchResourcePreparation.h"
#include "ResourceCollection.h"

namespace qmcplusplus
{
class MemoryResource : public Resource
{
public:
  MemoryResource(const std::string& name) : Resource(name) {}

  std::unique_ptr<Resource> makeClone() const override { return std::make_unique<MemoryResource>(*this); }

  std::vector<int> data;
};

/** Externally records preflight, cloning, and preparation hook calls. */
struct PreparationCallCounts
{
  int validate = 0;
  int clone    = 0;
  int prepare  = 0;
};

/** Records the context delivered by transactional batch preparation. */
class PreparingResource : public Resource
{
public:
  PreparingResource(const std::string& name,
                    bool fail_on_prepare = false,
                    bool fail_on_validate = false,
                    std::shared_ptr<PreparationCallCounts> call_counts = nullptr)
      : Resource(name),
        fail_on_prepare_(fail_on_prepare),
        fail_on_validate_(fail_on_validate),
        call_counts_(std::move(call_counts))
  {}

  std::unique_ptr<Resource> makeClone() const override
  {
    if (call_counts_)
      ++call_counts_->clone;
    return std::make_unique<PreparingResource>(*this);
  }

  void validateBatchResourcePreparation(const BatchResourcePreparationContext&) const override
  {
    if (call_counts_)
      ++call_counts_->validate;
    if (fail_on_validate_)
      throw std::runtime_error("deliberate batch resource preflight failure");
  }

  void prepareBatchResource(const BatchResourcePreparationContext& context) override
  {
    if (call_counts_)
      ++call_counts_->prepare;
    ++prepare_count;
    prepared_plan    = context.plan;
    prepared_crowd   = context.crowd_index;
    initial_capacity = context.plan ? context.initialWalkerCapacity() : 0;
    reserve_capacity = context.plan ? context.reserveWalkerCapacity() : 0;
    if (fail_on_prepare_)
      throw std::runtime_error("deliberate batch resource preparation failure");
  }

  std::shared_ptr<const BatchExecutionPlan> prepared_plan;
  std::size_t prepared_crowd   = 0;
  std::size_t initial_capacity = 0;
  std::size_t reserve_capacity = 0;
  int prepare_count            = 0;

private:
  bool fail_on_prepare_;
  bool fail_on_validate_;
  std::shared_ptr<PreparationCallCounts> call_counts_;
};

/** Build a storage-free plan with the requested crowd topology. */
std::shared_ptr<const BatchExecutionPlan> makeResourcePreparationPlan(
    std::vector<std::size_t> initial_walkers,
    std::vector<std::size_t> reserve_walkers = {})
{
  BatchExecutionSelectionInput input;
  input.topology.initial_walkers_per_crowd = std::move(initial_walkers);
  input.topology.reserve_walkers_per_crowd = std::move(reserve_walkers);
  return std::make_shared<const BatchExecutionPlan>(
      selectBatchExecutionPlan(input, [](const BatchExecutionPlanningContext&) {
        return std::vector<BatchMemoryParticipantContribution>{};
      }));
}

TEST_CASE("Resource", "[utilities]")
{
  auto mem_res = std::make_unique<MemoryResource>("test_res");
  mem_res->data.resize(5);

  auto res_copy      = mem_res->makeClone();
  auto& res_copy_ref = dynamic_cast<MemoryResource&>(*res_copy);
  REQUIRE(res_copy_ref.data.size() == 5);
}

TEST_CASE("DummyResource", "[utilities]")
{
  DummyResource dummy;
  auto dummy2 = dummy.makeClone();
  REQUIRE(dummy2->getName() == "Dummy");
  DummyResource dummy_alt("dummy_alt_name");
  REQUIRE(dummy_alt.getName() == "dummy_alt_name");
}

class WFCResourceConsumer
{
public:
  void createResource(ResourceCollection& collection)
  {
    auto memory_handle = std::make_unique<MemoryResource>("test_res");
    memory_handle->data.resize(5);
    collection.addResource(std::move(memory_handle));
  }

  void acquireResource(ResourceCollection& collection, const RefVectorWithLeader<WFCResourceConsumer>& wfcrc_list)
  { external_memory_handle = collection.lendResource<MemoryResource>(); }

  void releaseResource(ResourceCollection& collection, const RefVectorWithLeader<WFCResourceConsumer>& wfcrc_list)
  { collection.takebackResource(external_memory_handle); }

  auto& getResourceHandle() { return external_memory_handle; }

private:
  ResourceHandle<MemoryResource> external_memory_handle;
};

class ConstructorThrowingResourceConsumer
{
public:
  void acquireResource(ResourceCollection& collection,
                       const RefVectorWithLeader<ConstructorThrowingResourceConsumer>& consumer_list)
  {
    auto transient_handle = collection.lendResource<MemoryResource>();
    throw std::runtime_error("deliberate resource acquisition failure");
  }

  void releaseResource(ResourceCollection& collection,
                       const RefVectorWithLeader<ConstructorThrowingResourceConsumer>& consumer_list)
  {
    release_called = true;
  }

  bool release_called = false;
};

TEST_CASE("ResourceCollection", "[utilities]")
{
  ResourceCollection res_collection("abc");
  WFCResourceConsumer wfc, wfc1, wfc2;
  REQUIRE(wfc.getResourceHandle().hasResource() == false);

  wfc.createResource(res_collection);
  REQUIRE(wfc.getResourceHandle().hasResource() == false);

  RefVectorWithLeader wfc_list(wfc, {wfc, wfc1, wfc2});

  {
    ResourceCollectionTeamLock lock(res_collection, wfc_list);
    auto& res_handle = wfc.getResourceHandle();
    REQUIRE(res_handle);
    CHECK(res_collection.getOutstandingLoanCount() == 1);

    MemoryResource& mem_res             = res_handle;
    const MemoryResource& const_mem_res = res_handle;
    CHECK(mem_res.data.size() == 5);
    CHECK(const_mem_res.data.size() == 5);
  }

  REQUIRE(wfc.getResourceHandle().hasResource() == false);
  CHECK(res_collection.getOutstandingLoanCount() == 0);

  // Normal release leaves cursor traversal at the end, but no live handle
  // remains and preparation is therefore legal without another rewind.
  res_collection.prepareBatchResources({nullptr, 0});
}


TEST_CASE("ResourceCollection::printResources", "[utilities]")
{
  ResourceCollection res_collection("test_collection");
  res_collection.addResource(std::make_unique<DummyResource>("dummy1"));
  res_collection.addResource(std::make_unique<DummyResource>("dummy2"));

  std::stringstream buffer;
  res_collection.printResources(buffer);

  std::string output = buffer.str();
  REQUIRE(output.find("list resources in test_collection") != std::string::npos);
  REQUIRE(output.find("resource 0    name: dummy1") != std::string::npos);
  REQUIRE(output.find("resource 1    name: dummy2") != std::string::npos);
}

TEST_CASE("ResourceCollection typed lend failure preserves cursor", "[utilities]")
{
  ResourceCollection collection("typed_lend_failure");
  collection.addResource(std::make_unique<MemoryResource>("memory"));
  collection.addResource(std::make_unique<DummyResource>("dummy"));

  auto memory_handle = collection.lendResource<MemoryResource>();
  REQUIRE(collection.getCursor() == 1);
  REQUIRE(collection.getOutstandingLoanCount() == 1);
  CHECK_THROWS_AS(collection.lendResource<MemoryResource>(), std::bad_cast);
  CHECK(collection.getCursor() == 1);
  CHECK(collection.getOutstandingLoanCount() == 1);

  auto dummy_handle = collection.lendResource<DummyResource>();
  CHECK(collection.getCursor() == 2);
  CHECK(collection.getOutstandingLoanCount() == 2);

  collection.rewind();
  CHECK_THROWS_AS(collection.takebackResource(dummy_handle), std::runtime_error);
  CHECK(collection.getCursor() == 0);
  CHECK(collection.getOutstandingLoanCount() == 2);
  CHECK(dummy_handle.hasResource());

  collection.takebackResource(memory_handle);
  collection.takebackResource(dummy_handle);
  CHECK_FALSE(memory_handle.hasResource());
  CHECK_FALSE(dummy_handle.hasResource());
  CHECK(collection.getOutstandingLoanCount() == 0);
}

TEST_CASE("ResourceCollectionTeamLock construction failure preserves cursor", "[utilities]")
{
  ResourceCollection collection("team_lock_construction_failure");
  collection.addResource(std::make_unique<MemoryResource>("memory"));
  ConstructorThrowingResourceConsumer consumer;
  RefVectorWithLeader consumer_list(consumer, {consumer});

  CHECK_THROWS_AS(ResourceCollectionTeamLock(collection, consumer_list), std::runtime_error);
  CHECK(collection.getCursor() == 0);
  CHECK(collection.getOutstandingLoanCount() == 0);
  CHECK_FALSE(consumer.release_called);

  auto recovered_handle = collection.lendResource<MemoryResource>();
  REQUIRE(recovered_handle.hasResource());
  CHECK(collection.getOutstandingLoanCount() == 1);
  collection.rewind();
  collection.takebackResource(recovered_handle);
  CHECK(collection.getOutstandingLoanCount() == 0);
}

TEST_CASE("ResourceCollection prepares zero-lane and no-policy resources", "[utilities][batch_resource]")
{
  auto first_counts  = std::make_shared<PreparationCallCounts>();
  auto second_counts = std::make_shared<PreparationCallCounts>();
  ResourceCollection collection("batch_preparation");
  collection.addResource(std::make_unique<PreparingResource>("first", false, false, first_counts));
  collection.addResource(std::make_unique<PreparingResource>("second", false, false, second_counts));

  const std::shared_ptr<const BatchExecutionPlan> plan = makeResourcePreparationPlan({0}, {0});
  collection.prepareBatchResources({plan, 0});

  const BatchResourcePreparationProvenance& prepared_provenance =
      collection.getBatchResourcePreparationProvenance();
  CHECK(prepared_provenance.state == BatchResourcePreparationState::PREPARED);
  CHECK(prepared_provenance.plan.get() == plan.get());
  CHECK(prepared_provenance.crowd_index == 0);

  auto first  = collection.lendResource<PreparingResource>();
  auto second = collection.lendResource<PreparingResource>();
  CHECK(first.getResource().prepare_count == 1);
  CHECK(second.getResource().prepare_count == 1);
  CHECK(first.getResource().prepared_plan.get() == plan.get());
  CHECK(second.getResource().prepared_plan.get() == plan.get());
  CHECK(first.getResource().prepared_crowd == 0);
  CHECK(first.getResource().initial_capacity == 0);
  CHECK(first.getResource().reserve_capacity == 0);
  CHECK(first_counts->validate == 1);
  CHECK(second_counts->validate == 1);
  CHECK(first_counts->clone == 1);
  CHECK(second_counts->clone == 1);
  CHECK(first_counts->prepare == 1);
  CHECK(second_counts->prepare == 1);
  collection.rewind();
  collection.takebackResource(first);
  collection.takebackResource(second);

  // The explicit no-policy state is forwarded instead of being mistaken for
  // "nothing to do".  Its crowd index is unrestricted because it has no topology.
  collection.prepareBatchResources({nullptr, 91});
  const BatchResourcePreparationProvenance& cleared_provenance =
      collection.getBatchResourcePreparationProvenance();
  CHECK(cleared_provenance.state == BatchResourcePreparationState::UNPREPARED);
  CHECK_FALSE(cleared_provenance.plan);
  CHECK(cleared_provenance.crowd_index == 0);
  first  = collection.lendResource<PreparingResource>();
  second = collection.lendResource<PreparingResource>();
  CHECK(first.getResource().prepare_count == 2);
  CHECK(second.getResource().prepare_count == 2);
  CHECK_FALSE(first.getResource().prepared_plan);
  CHECK_FALSE(second.getResource().prepared_plan);
  CHECK(first.getResource().prepared_crowd == 91);
  CHECK(first_counts->validate == 2);
  CHECK(second_counts->validate == 2);
  CHECK(first_counts->clone == 2);
  CHECK(second_counts->clone == 2);
  CHECK(first_counts->prepare == 2);
  CHECK(second_counts->prepare == 2);
  collection.rewind();
  collection.takebackResource(first);
  collection.takebackResource(second);
}

TEST_CASE("ResourceCollection preparation provenance requires explicit clearing",
          "[utilities][batch_resource]")
{
  auto call_counts = std::make_shared<PreparationCallCounts>();
  ResourceCollection collection("preparation_provenance");
  collection.addResource(std::make_unique<PreparingResource>("tracked", false, false, call_counts));

  const std::shared_ptr<const BatchExecutionPlan> plan = makeResourcePreparationPlan({1}, {2});
  collection.prepareBatchResources({plan, 0});
  REQUIRE(collection.getBatchResourcePreparationProvenance().state ==
          BatchResourcePreparationState::PREPARED);
  REQUIRE(collection.getBatchResourcePreparationProvenance().plan.get() == plan.get());

  const PreparationCallCounts counts_after_preparation = *call_counts;
  CHECK_THROWS_AS(collection.prepareBatchResources({plan, 0}), std::logic_error);
  CHECK(call_counts->validate == counts_after_preparation.validate);
  CHECK(call_counts->clone == counts_after_preparation.clone);
  CHECK(call_counts->prepare == counts_after_preparation.prepare);
  CHECK(collection.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::PREPARED);
  CHECK(collection.getBatchResourcePreparationProvenance().plan.get() == plan.get());
  CHECK_THROWS_AS(collection.addResource(std::make_unique<DummyResource>()), std::logic_error);

  ResourceCollection derived(collection);
  CHECK(derived.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::DERIVED_REQUIRES_CLEAR);
  CHECK(derived.getBatchResourcePreparationProvenance().plan.get() == plan.get());
  CHECK(derived.getBatchResourcePreparationProvenance().crowd_index == 0);

  const PreparationCallCounts counts_after_copy = *call_counts;
  CHECK_THROWS_AS(derived.prepareBatchResources({plan, 0}), std::logic_error);
  CHECK(call_counts->validate == counts_after_copy.validate);
  CHECK(call_counts->clone == counts_after_copy.clone);
  CHECK(call_counts->prepare == counts_after_copy.prepare);
  CHECK(derived.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::DERIVED_REQUIRES_CLEAR);
  CHECK(derived.getBatchResourcePreparationProvenance().plan.get() == plan.get());
  CHECK_THROWS_AS(derived.addResource(std::make_unique<DummyResource>()), std::logic_error);

  // A null preparation rebuilds no-policy resources and explicitly clears the
  // provenance barrier before a later nonnull plan may be applied.
  derived.prepareBatchResources({nullptr, 71});
  CHECK(derived.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::UNPREPARED);
  CHECK_FALSE(derived.getBatchResourcePreparationProvenance().plan);
  derived.prepareBatchResources({plan, 0});
  CHECK(derived.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::PREPARED);

  ResourceCollection moved(std::move(derived));
  CHECK(moved.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::PREPARED);
  CHECK(moved.getBatchResourcePreparationProvenance().plan.get() == plan.get());
  CHECK(derived.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::UNPREPARED);
  CHECK_FALSE(derived.getBatchResourcePreparationProvenance().plan);
}

TEST_CASE("ResourceCollection successful preflight is mutation free",
          "[utilities][batch_resource]")
{
  auto call_counts = std::make_shared<PreparationCallCounts>();
  ResourceCollection collection("successful_preflight");
  collection.addResource(std::make_unique<PreparingResource>("tracked", false, false, call_counts));

  const std::shared_ptr<const BatchExecutionPlan> plan = makeResourcePreparationPlan({1}, {3});
  collection.validateBatchResourcePreparation({plan, 0});
  collection.validateBatchResourcePreparation({nullptr, 97});
  CHECK(call_counts->validate == 2);
  CHECK(call_counts->clone == 0);
  CHECK(call_counts->prepare == 0);
  CHECK(collection.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::UNPREPARED);

  auto tracked = collection.lendResource<PreparingResource>();
  CHECK(tracked.getResource().prepare_count == 0);
  CHECK_FALSE(tracked.getResource().prepared_plan);
  collection.rewind();
  collection.takebackResource(tracked);
}

TEST_CASE("ResourceCollection preparation rejects invalid topology before mutation",
          "[utilities][batch_resource]")
{
  ResourceCollection collection("invalid_batch_preparation");
  collection.addResource(std::make_unique<PreparingResource>("tracked"));

  const std::shared_ptr<const BatchExecutionPlan> plan = makeResourcePreparationPlan({0});
  CHECK_THROWS_AS(collection.prepareBatchResources({plan, 1}), std::out_of_range);

  auto tracked = collection.lendResource<PreparingResource>();
  CHECK(tracked.getResource().prepare_count == 0);
  CHECK_FALSE(tracked.getResource().prepared_plan);
  collection.rewind();
  collection.takebackResource(tracked);

  CHECK_THROWS_AS(makeResourcePreparationPlan({0}, {0, 0}), std::invalid_argument);
  const std::shared_ptr<const BatchExecutionPlan> empty_topology_plan = makeResourcePreparationPlan({});
  CHECK_THROWS_AS(collection.prepareBatchResources({empty_topology_plan, 0}), std::out_of_range);
}

TEST_CASE("ResourceCollection preflight rejects before cloning or preparation",
          "[utilities][batch_resource]")
{
  auto first_counts     = std::make_shared<PreparationCallCounts>();
  auto rejecting_counts = std::make_shared<PreparationCallCounts>();

  ResourceCollection collection("failed_preflight");
  collection.addResource(std::make_unique<PreparingResource>("first", false, false, first_counts));
  collection.addResource(std::make_unique<PreparingResource>("rejecting", false, true, rejecting_counts));

  const std::shared_ptr<const BatchExecutionPlan> plan = makeResourcePreparationPlan({1}, {2});
  CHECK_THROWS_AS(collection.prepareBatchResources({plan, 0}), std::runtime_error);
  CHECK(collection.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::UNPREPARED);
  CHECK_FALSE(collection.getBatchResourcePreparationProvenance().plan);
  CHECK(first_counts->validate == 1);
  CHECK(rejecting_counts->validate == 1);
  CHECK(first_counts->clone == 0);
  CHECK(rejecting_counts->clone == 0);
  CHECK(first_counts->prepare == 0);
  CHECK(rejecting_counts->prepare == 0);

  // Failure leaves the source resources in their original, unprepared state.
  auto first     = collection.lendResource<PreparingResource>();
  auto rejecting = collection.lendResource<PreparingResource>();
  CHECK(first.getResource().prepare_count == 0);
  CHECK(rejecting.getResource().prepare_count == 0);
  CHECK_FALSE(first.getResource().prepared_plan);
  CHECK_FALSE(rejecting.getResource().prepared_plan);
  collection.rewind();
  collection.takebackResource(first);
  collection.takebackResource(rejecting);
}

TEST_CASE("ResourceCollection live-loan preflight precedes resource hooks",
          "[utilities][batch_resource]")
{
  auto call_counts = std::make_shared<PreparationCallCounts>();
  ResourceCollection collection("live_loan_preflight");
  collection.addResource(std::make_unique<PreparingResource>("tracked", false, false, call_counts));

  auto tracked = collection.lendResource<PreparingResource>();
  const std::shared_ptr<const BatchExecutionPlan> plan = makeResourcePreparationPlan({1}, {1});
  CHECK_THROWS_AS(collection.prepareBatchResources({plan, 0}), std::logic_error);
  CHECK(call_counts->validate == 0);
  CHECK(call_counts->clone == 0);
  CHECK(call_counts->prepare == 0);
  CHECK(tracked.getResource().prepare_count == 0);

  collection.rewind();
  collection.takebackResource(tracked);
}

TEST_CASE("ResourceCollection preparation is transactional", "[utilities][batch_resource]")
{
  ResourceCollection collection("transactional_batch_preparation");
  collection.addResource(std::make_unique<PreparingResource>("first"));
  collection.addResource(std::make_unique<PreparingResource>("throwing", true));

  const std::shared_ptr<const BatchExecutionPlan> plan = makeResourcePreparationPlan({2}, {4});
  CHECK_THROWS_AS(collection.prepareBatchResources({plan, 0}), std::runtime_error);
  CHECK(collection.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::UNPREPARED);
  CHECK_FALSE(collection.getBatchResourcePreparationProvenance().plan);

  // Although the first candidate clone ran its hook, neither candidate was published.
  auto first    = collection.lendResource<PreparingResource>();
  auto throwing = collection.lendResource<PreparingResource>();
  CHECK(first.getResource().prepare_count == 0);
  CHECK(throwing.getResource().prepare_count == 0);
  CHECK_FALSE(first.getResource().prepared_plan);
  CHECK_FALSE(throwing.getResource().prepared_plan);
  collection.rewind();
  collection.takebackResource(first);
  collection.takebackResource(throwing);

  // Preparation is an idle-boundary operation and refuses to replace lent storage.
  collection.rewind();
  first = collection.lendResource<PreparingResource>();
  CHECK(collection.getOutstandingLoanCount() == 1);
  CHECK_THROWS_AS(collection.prepareBatchResources({plan, 0}), std::logic_error);
  CHECK(first.getResource().prepare_count == 0);

  // Rewinding traversal must not disguise a still-live handle as an idle collection.
  collection.rewind();
  CHECK_THROWS_AS(collection.prepareBatchResources({plan, 0}), std::logic_error);
  CHECK(collection.getOutstandingLoanCount() == 1);
  collection.takebackResource(first);
  CHECK(collection.getOutstandingLoanCount() == 0);
}

TEST_CASE("ResourceCollection rejects moving live loans", "[utilities][batch_resource]")
{
  ResourceCollection collection("move_with_live_loan");
  collection.addResource(std::make_unique<MemoryResource>("memory"));

  auto memory = collection.lendResource<MemoryResource>();
  CHECK_THROWS_AS([&collection] { ResourceCollection moved(std::move(collection)); }(), std::logic_error);
  CHECK(collection.size() == 1);
  CHECK(collection.getOutstandingLoanCount() == 1);
  CHECK(memory.hasResource());

  collection.rewind();
  collection.takebackResource(memory);
  ResourceCollection moved(std::move(collection));
  CHECK(moved.size() == 1);
  CHECK(moved.getOutstandingLoanCount() == 0);
  CHECK(collection.size() == 0);
  CHECK(collection.getOutstandingLoanCount() == 0);
}

} // namespace qmcplusplus
