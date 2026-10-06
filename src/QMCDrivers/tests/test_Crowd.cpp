//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>

#include "Message/Communicate.h"
#include "QMCDrivers/Crowd.h"
#include "type_traits/template_types.hpp"
#include "Estimators/EstimatorManagerNew.h"
#include <MinimalWaveFunctionPool.h>
#include <MinimalParticlePool.h>
#include <MinimalHamiltonianPool.h>

#include "QMCDrivers/tests/SetupPools.h"

namespace qmcplusplus
{
namespace testing
{
/** Externally records resource lifecycle hooks across transactional candidates. */
struct CrowdPreparationCallCounts
{
  int validate = 0;
  int clone    = 0;
  int prepare  = 0;
};

/** Test resource that exposes crowd-plan preparation without requiring a walker leader. */
class CrowdPreparationResource : public Resource
{
public:
  CrowdPreparationResource(const std::string& name,
                           bool fail_on_prepare = false,
                           bool fail_on_validate = false,
                           std::shared_ptr<CrowdPreparationCallCounts> call_counts = nullptr)
      : Resource(name),
        fail_on_prepare_(fail_on_prepare),
        fail_on_validate_(fail_on_validate),
        call_counts_(std::move(call_counts))
  {}

  std::unique_ptr<Resource> makeClone() const override
  {
    if (call_counts_)
      ++call_counts_->clone;
    return std::make_unique<CrowdPreparationResource>(*this);
  }

  void validateBatchResourcePreparation(const BatchResourcePreparationContext&) const override
  {
    if (call_counts_)
      ++call_counts_->validate;
    if (fail_on_validate_)
      throw std::runtime_error("deliberate DriverWalker resource preflight failure");
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
      throw std::runtime_error("deliberate DriverWalker resource preparation failure");
  }

  std::shared_ptr<const BatchExecutionPlan> prepared_plan;
  std::size_t prepared_crowd   = 0;
  std::size_t initial_capacity = 0;
  std::size_t reserve_capacity = 0;
  int prepare_count            = 0;

private:
  bool fail_on_prepare_;
  bool fail_on_validate_;
  std::shared_ptr<CrowdPreparationCallCounts> call_counts_;
};

/** Build a storage-free plan suitable for driver resource lifecycle tests. */
std::shared_ptr<const BatchExecutionPlan> makeCrowdPreparationPlan(std::vector<std::size_t> initial_walkers,
                                                                  std::vector<std::size_t> reserve_walkers)
{
  BatchExecutionSelectionInput input;
  input.topology.initial_walkers_per_crowd = std::move(initial_walkers);
  input.topology.reserve_walkers_per_crowd = std::move(reserve_walkers);
  return std::make_shared<const BatchExecutionPlan>(
      selectBatchExecutionPlan(input, [](const BatchExecutionPlanningContext&) {
        return std::vector<BatchMemoryParticipantContribution>{};
      }));
}

class CrowdWithWalkers
{
public:
  using MCPWalker = Walker<QMCTraits, PtclOnLatticeTraits>;

  EstimatorManagerNew em;
  UPtr<Crowd> crowd_ptr;
  Crowd& get_crowd() { return *crowd_ptr; }
  UPtrVector<MCPWalker> walkers;
  UPtrVector<ParticleSet> psets;
  UPtrVector<TrialWaveFunction> twfs;
  UPtrVector<QMCHamiltonian> hams;
  std::vector<TinyVector<double, 3>> tpos;
  DriverWalkerResourceCollection driverwalker_resource_collection_;

public:
  CrowdWithWalkers(SetupPools& pools) : em(pools.hamiltonian_pool->getHamiltonian().value(), pools.comm)
  {
    crowd_ptr =
        std::make_unique<Crowd>(em, driverwalker_resource_collection_, *pools.particle_pool->getParticleSet("e"),
                                pools.wavefunction_pool->getWaveFunction().value(),
                                pools.hamiltonian_pool->getHamiltonian().value());
    Crowd& crowd = *crowd_ptr;
    // To match the minimal particle set
    int num_particles = 2;
    // for testing we update the first position in the walker
    auto makePointWalker = [this, &pools, &crowd, num_particles](TinyVector<double, 3> pos) {
      walkers.emplace_back(std::make_unique<MCPWalker>(num_particles));
      walkers.back()->R[0] = pos;
      psets.emplace_back(std::make_unique<ParticleSet>(*(pools.particle_pool->getParticleSet("e"))));
      twfs.emplace_back(pools.wavefunction_pool->getWaveFunction().value().get().makeClone(*psets.back()));
      hams.emplace_back(pools.hamiltonian_pool->getHamiltonian().value().get().makeClone(*psets.back(), *twfs.back()));
      crowd.addWalker(*walkers.back(), *psets.back(), *twfs.back(), *hams.back());
    };

    tpos.push_back(TinyVector<double, 3>(1.0, 0.0, 0.0));
    makePointWalker(tpos.back());
    tpos.push_back(TinyVector<double, 3>(1.0, 2.0, 0.0));
    makePointWalker(tpos.back());
  }

  void makeAnotherPointWalker()
  {
    walkers.emplace_back(std::make_unique<MCPWalker>(*walkers.back()));
    psets.emplace_back(std::make_unique<ParticleSet>(*psets.back()));
    twfs.emplace_back(twfs.back()->makeClone(*psets.back()));
    hams.emplace_back(hams.back()->makeClone(*psets.back(), *twfs.back()));
  }
};
} // namespace testing

TEST_CASE("Crowd integration", "[drivers]")
{
  Communicate* comm = OHMMS::Controller;
  using namespace testing;
  SetupPools pools;

  EstimatorManagerNew em(pools.hamiltonian_pool->getHamiltonian().value(), comm);

  DriverWalkerResourceCollection driverwalker_resource_collection_;

  Crowd crowd(em, driverwalker_resource_collection_, *pools.particle_pool->getParticleSet("e"),
              pools.wavefunction_pool->getWaveFunction().value(), pools.hamiltonian_pool->getHamiltonian().value());
}

TEST_CASE("Crowd redistribute walkers")
{
  using namespace testing;
  SetupPools pools;

  CrowdWithWalkers crowd_with_walkers(pools);
  Crowd& crowd = crowd_with_walkers.get_crowd();

  crowd_with_walkers.makeAnotherPointWalker();
  crowd.clearWalkers();
  for (int iw = 0; iw < crowd_with_walkers.walkers.size(); ++iw)
    crowd.addWalker(*crowd_with_walkers.walkers[iw], *crowd_with_walkers.psets[iw], *crowd_with_walkers.twfs[iw],
                    *crowd_with_walkers.hams[iw]);
  REQUIRE(crowd.size() == 3);
}

TEST_CASE("Crowd prepares reserve resources without a living walker", "[drivers][batch_resource]")
{
  using namespace testing;
  SetupPools pools;
  EstimatorManagerNew estimator_manager(pools.hamiltonian_pool->getHamiltonian().value(), pools.comm);

  DriverWalkerResourceCollection golden_resources;
  golden_resources.pset_res.addResource(std::make_unique<CrowdPreparationResource>("zero_walker_resource"));

  Crowd crowd(estimator_manager, golden_resources, *pools.particle_pool->getParticleSet("e"),
              pools.wavefunction_pool->getWaveFunction().value(),
              pools.hamiltonian_pool->getHamiltonian().value());
  REQUIRE(crowd.size() == 0);
  crowd.reserve(5);

  const std::shared_ptr<const BatchExecutionPlan> plan = makeCrowdPreparationPlan({0}, {5});
  crowd.getSharedResource().prepareBatchResources({plan, 0});

  CHECK(crowd.getSharedResource().pset_res.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::PREPARED);
  CHECK(crowd.getSharedResource().twf_res.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::PREPARED);
  CHECK(crowd.getSharedResource().ham_res.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::PREPARED);
  CHECK(crowd.getSharedResource().pset_res.getBatchResourcePreparationProvenance().plan.get() == plan.get());

  ResourceCollection& particle_resources = crowd.getSharedResource().pset_res;
  auto prepared = particle_resources.lendResource<CrowdPreparationResource>();
  CHECK(prepared.getResource().prepare_count == 1);
  CHECK(prepared.getResource().prepared_plan.get() == plan.get());
  CHECK(prepared.getResource().prepared_crowd == 0);
  CHECK(prepared.getResource().initial_capacity == 0);
  CHECK(prepared.getResource().reserve_capacity == 5);
  particle_resources.rewind();
  particle_resources.takebackResource(prepared);

  // A later no-policy section still visits the clone and clears its prior plan.
  crowd.getSharedResource().prepareBatchResources({nullptr, 17});
  CHECK(crowd.getSharedResource().pset_res.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::UNPREPARED);
  CHECK(crowd.getSharedResource().twf_res.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::UNPREPARED);
  CHECK(crowd.getSharedResource().ham_res.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::UNPREPARED);
  CHECK_FALSE(crowd.getSharedResource().pset_res.getBatchResourcePreparationProvenance().plan);
  prepared = particle_resources.lendResource<CrowdPreparationResource>();
  CHECK(prepared.getResource().prepare_count == 2);
  CHECK_FALSE(prepared.getResource().prepared_plan);
  CHECK(prepared.getResource().prepared_crowd == 17);
  particle_resources.rewind();
  particle_resources.takebackResource(prepared);
}

TEST_CASE("DriverWalker resource preparation is atomic across families", "[drivers][batch_resource]")
{
  using namespace testing;
  DriverWalkerResourceCollection resources;
  resources.pset_res.addResource(std::make_unique<CrowdPreparationResource>("particle"));
  resources.twf_res.addResource(std::make_unique<CrowdPreparationResource>("throwing_wavefunction", true));
  resources.ham_res.addResource(std::make_unique<CrowdPreparationResource>("hamiltonian"));

  const std::shared_ptr<const BatchExecutionPlan> plan = makeCrowdPreparationPlan({0}, {3});
  CHECK_THROWS_AS(resources.prepareBatchResources({plan, 0}), std::runtime_error);

  // The particle candidate completed, but failure in the next family prevented
  // publication in all three original collections.
  auto particle = resources.pset_res.lendResource<CrowdPreparationResource>();
  auto wavefunction = resources.twf_res.lendResource<CrowdPreparationResource>();
  auto hamiltonian = resources.ham_res.lendResource<CrowdPreparationResource>();
  CHECK(particle.getResource().prepare_count == 0);
  CHECK(wavefunction.getResource().prepare_count == 0);
  CHECK(hamiltonian.getResource().prepare_count == 0);
  CHECK_FALSE(particle.getResource().prepared_plan);
  CHECK_FALSE(wavefunction.getResource().prepared_plan);
  CHECK_FALSE(hamiltonian.getResource().prepared_plan);

  resources.pset_res.rewind();
  resources.pset_res.takebackResource(particle);
  resources.twf_res.rewind();
  resources.twf_res.takebackResource(wavefunction);
  resources.ham_res.rewind();
  resources.ham_res.takebackResource(hamiltonian);
}

TEST_CASE("DriverWalker resource preflight covers all families before cloning",
          "[drivers][batch_resource]")
{
  using namespace testing;
  auto particle_counts     = std::make_shared<CrowdPreparationCallCounts>();
  auto wavefunction_counts = std::make_shared<CrowdPreparationCallCounts>();
  auto hamiltonian_counts  = std::make_shared<CrowdPreparationCallCounts>();

  DriverWalkerResourceCollection resources;
  resources.pset_res.addResource(
      std::make_unique<CrowdPreparationResource>("particle", false, false, particle_counts));
  resources.twf_res.addResource(
      std::make_unique<CrowdPreparationResource>("wavefunction", false, false, wavefunction_counts));
  resources.ham_res.addResource(
      std::make_unique<CrowdPreparationResource>("rejecting_hamiltonian", false, true, hamiltonian_counts));

  const std::shared_ptr<const BatchExecutionPlan> plan = makeCrowdPreparationPlan({1}, {3});
  CHECK_THROWS_AS(resources.prepareBatchResources({plan, 0}), std::runtime_error);

  CHECK(particle_counts->validate == 1);
  CHECK(wavefunction_counts->validate == 1);
  CHECK(hamiltonian_counts->validate == 1);
  CHECK(particle_counts->clone == 0);
  CHECK(wavefunction_counts->clone == 0);
  CHECK(hamiltonian_counts->clone == 0);
  CHECK(particle_counts->prepare == 0);
  CHECK(wavefunction_counts->prepare == 0);
  CHECK(hamiltonian_counts->prepare == 0);

  // No family was replaced or modified by the failed preflight.
  auto particle     = resources.pset_res.lendResource<CrowdPreparationResource>();
  auto wavefunction = resources.twf_res.lendResource<CrowdPreparationResource>();
  auto hamiltonian  = resources.ham_res.lendResource<CrowdPreparationResource>();
  CHECK(particle.getResource().prepare_count == 0);
  CHECK(wavefunction.getResource().prepare_count == 0);
  CHECK(hamiltonian.getResource().prepare_count == 0);
  CHECK_FALSE(particle.getResource().prepared_plan);
  CHECK_FALSE(wavefunction.getResource().prepared_plan);
  CHECK_FALSE(hamiltonian.getResource().prepared_plan);

  resources.pset_res.rewind();
  resources.pset_res.takebackResource(particle);
  resources.twf_res.rewind();
  resources.twf_res.takebackResource(wavefunction);
  resources.ham_res.rewind();
  resources.ham_res.takebackResource(hamiltonian);
}

TEST_CASE("DriverWalker resource preflight rejects a late-family live loan",
          "[drivers][batch_resource]")
{
  using namespace testing;
  auto particle_counts     = std::make_shared<CrowdPreparationCallCounts>();
  auto wavefunction_counts = std::make_shared<CrowdPreparationCallCounts>();
  auto hamiltonian_counts  = std::make_shared<CrowdPreparationCallCounts>();

  DriverWalkerResourceCollection resources;
  resources.pset_res.addResource(
      std::make_unique<CrowdPreparationResource>("particle", false, false, particle_counts));
  resources.twf_res.addResource(
      std::make_unique<CrowdPreparationResource>("wavefunction", false, false, wavefunction_counts));
  resources.ham_res.addResource(
      std::make_unique<CrowdPreparationResource>("hamiltonian", false, false, hamiltonian_counts));

  auto hamiltonian = resources.ham_res.lendResource<CrowdPreparationResource>();
  const std::shared_ptr<const BatchExecutionPlan> plan = makeCrowdPreparationPlan({1}, {2});
  CHECK_THROWS_AS(resources.prepareBatchResources({plan, 0}), std::logic_error);

  CHECK(particle_counts->validate == 1);
  CHECK(wavefunction_counts->validate == 1);
  CHECK(hamiltonian_counts->validate == 0);
  CHECK(particle_counts->clone == 0);
  CHECK(wavefunction_counts->clone == 0);
  CHECK(hamiltonian_counts->clone == 0);
  CHECK(particle_counts->prepare == 0);
  CHECK(wavefunction_counts->prepare == 0);
  CHECK(hamiltonian_counts->prepare == 0);
  CHECK(hamiltonian.getResource().prepare_count == 0);

  resources.ham_res.rewind();
  resources.ham_res.takebackResource(hamiltonian);
}

} // namespace qmcplusplus
