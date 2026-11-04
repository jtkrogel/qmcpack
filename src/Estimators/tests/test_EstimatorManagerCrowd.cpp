//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File refactored from: Refactored from test_manager.cpp
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Message/Communicate.h"
#include "OhmmsData/Libxml2Doc.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "Estimators/EstimatorManagerNew.h"
#include "Estimators/EstimatorManagerCrowd.h"
#include "Estimators/ScalarEstimatorBase.h"
#include "Estimators/tests/EstimatorManagerNewTest.h"
#include "Estimators/EstimatorInputDelegates.h"
#include "EstimatorManagerInputTest.h"
#include <MinimalParticlePool.h>
#include <MinimalHamiltonianPool.h>
#include <MinimalWaveFunctionPool.h>
#include "Utilities/ResourceCollection.h"
#include "Utilities/ProjectData.h"

#include <array>
#include <cmath>

namespace qmcplusplus
{
TEST_CASE("EstimatorManagerCrowd::EstimatorManagerCrowd", "[estimators]")
{
  ProjectData test_project("test", ProjectData::DriverVersion::BATCH);
  Communicate* comm = OHMMS::Controller;

  using namespace testing;
  Libxml2Document estimators_doc = createEstimatorManagerNewInputXML();
  EstimatorManagerInput emi(estimators_doc.getRoot());

  auto particle_pool = MinimalParticlePool::make_diamondC_1x1x1(comm);
  auto wavefunction_pool =
      MinimalWaveFunctionPool::make_diamondC_1x1x1(test_project.getRuntimeOptions(), comm, particle_pool);
  auto& pset            = *(particle_pool.getParticleSet("e"));
  auto hamiltonian_pool = MinimalHamiltonianPool::make_hamWithEE(comm, particle_pool, wavefunction_pool);
  TrialWaveFunction& twf(wavefunction_pool.getWaveFunction().value());
  QMCHamiltonian& ham(hamiltonian_pool.getHamiltonian().value());

  EstimatorManagerNew emn(ham, comm);
  emn.constructEstimators(std::move(emi), pset, twf, ham, particle_pool.getPool());

  CHECK(emn.getNumEstimators() == n_opest_new_input_xml);
  CHECK(emn.getNumScalarEstimators() == 0);

  EstimatorManagerCrowd emc(emn);
}

TEST_CASE("EstimatorManagerCrowd PerParticleHamiltonianLogger integration", "[estimators]")
{
  ProjectData test_project("test", ProjectData::DriverVersion::BATCH);
  Communicate* comm = OHMMS::Controller;

  using namespace testing;
  Libxml2Document estimators_doc = createEstimatorManagerNewVMCInputXML();
  EstimatorManagerInput emi(estimators_doc.getRoot());


  auto particle_pool = MinimalParticlePool::make_diamondC_1x1x1(comm);
  auto wavefunction_pool =
      MinimalWaveFunctionPool::make_diamondC_1x1x1(test_project.getRuntimeOptions(), comm, particle_pool);
  auto& pset = *(particle_pool.getParticleSet("e"));
  // This is where the pset properties "properies" gain the different hamiltonian operator values.
  auto hamiltonian_pool = MinimalHamiltonianPool::makeHamWithEI(comm, particle_pool, wavefunction_pool);

  TrialWaveFunction& twf(wavefunction_pool.getWaveFunction().value());
  QMCHamiltonian& ham(hamiltonian_pool.getHamiltonian().value());

  ham.informOperatorsOfListener();

  PerParticleHamiltonianLoggerInput pphli;
  emi.append(std::move(pphli));

  EstimatorManagerNew emn(ham, comm);
  emn.constructEstimators(std::move(emi), pset, twf, ham, particle_pool.getPool());

  CHECK(emn.getNumEstimators() == 4);
  CHECK(emn.getNumScalarEstimators() == 0);

  // We repeat a bunch of test_QMCHamiltonian here to get things set up.

  UPtrVector<QMCHamiltonian> hams;
  UPtrVector<TrialWaveFunction> twfs;
  std::vector<ParticleSet> psets;

  int num_walkers = 4;

  for (int iw = 0; iw < num_walkers; ++iw)
  {
    psets.emplace_back(pset);
    psets.back().randomizeFromSource(*particle_pool.getParticleSet("ion"));
    twfs.emplace_back(twf.makeClone(psets.back()));
    hams.emplace_back(ham.makeClone(psets.back(), *twfs.back()));
  }

  EstimatorManagerCrowd emc(emn);

  using MCPWalker = Walker<QMCTraits, PtclOnLatticeTraits>;

  std::vector<MCPWalker> walkers(num_walkers, MCPWalker(pset.getTotalNum()));

  for (auto& walker : walkers)
  {
    walker.R          = pset.R;
    walker.spins      = pset.spins;
    walker.Properties = pset.Properties;
    walker.registerData();
    walker.DataSet.allocate();
  }

  auto walker_refs = makeRefVector<MCPWalker>(walkers);
  RefVectorWithLeader<MCPWalker> walker_list{walker_refs[0], walker_refs};

  auto p_refs = makeRefVector<ParticleSet>(psets);
  RefVectorWithLeader<ParticleSet> p_list{p_refs[0], p_refs};

  ResourceCollection pset_res("test_pset_res");
  p_list.getLeader().createResource(pset_res);
  ResourceCollectionTeamLock<ParticleSet> pset_lock(pset_res, p_list);

  auto twf_refs = convertUPtrToRefVector(twfs);
  RefVectorWithLeader<TrialWaveFunction> twf_list{twf_refs[0], twf_refs};
  ResourceCollection wfc_res("test_wfc_res");
  twf_list.getLeader().createResource(wfc_res);
  ResourceCollectionTeamLock<TrialWaveFunction> mw_wfc_lock(wfc_res, twf_list);

  auto ham_refs = convertUPtrToRefVector(hams);
  RefVectorWithLeader<QMCHamiltonian> ham_list(ham_refs[0], ham_refs);

  ResourceCollection ham_res("test_ham_res");
  ham_list.getLeader().createResource(ham_res);
  ResourceCollectionTeamLock<QMCHamiltonian> mw_ham_lock(ham_res, ham_list);

  emc.registerListeners(ham_list);

  //   Setup RNG
  FakeRandom<OHMMS_PRECISION_FULL> rng;

  // Without this QMCHamiltonian::mw_evaluate segfaults
  // Because the CoulombPBCAA hamiltonian component has PtclRhoK (StructFact) that is invalid.
  ParticleSet::mw_update(p_list);
  TrialWaveFunction::mw_evaluateLog(twf_list, p_list);
  QMCHamiltonian::mw_evaluate(ham_list, twf_list, p_list);

  auto savePropertiesIntoWalker = [](QMCHamiltonian& ham, MCPWalker& walker) {
    ham.saveProperty(walker.getPropertyBase());
  };
  for (int iw = 0; iw < num_walkers; ++iw)
    savePropertiesIntoWalker(*(hams[iw]), walkers[iw]);

  emc.accumulate(walker_refs, p_refs, twf_refs, ham_refs, rng);
}



TEST_CASE("EstimatorManagerCrowd preserves fractional-weight scalar normalization", "[estimators][scalar_weight_precision]")
{
  bool exact_weight_argument = false;
  SECTION("crowd weight totals") {}
  SECTION("exact full-precision weight argument") { exact_weight_argument = true; }

  Communicate* comm = OHMMS::Controller;
  QMCHamiltonian ham;
  EstimatorManagerNew manager(ham, comm);
  manager.startDriverRun();
  EstimatorManagerCrowd first(manager), second(manager);
  using MCPWalker = EstimatorManagerCrowd::MCPWalker;
  using WP = WalkerProperties::Indexes;
  std::vector<MCPWalker> walkers_first(2, MCPWalker(1)), walkers_second(3, MCPWalker(1));
  auto refs_first = makeRefVector<MCPWalker>(walkers_first);
  auto refs_second = makeRefVector<MCPWalker>(walkers_second);
  RefVector<ParticleSet> psets;
  RefVector<TrialWaveFunction> wfns;
  RefVector<QMCHamiltonian> hams;
  FakeRandom<OHMMS_PRECISION_FULL> rng;
  constexpr double energy = -5.0931317943;
  constexpr double constant_potential = -2.91138651660044;
  const std::array<double,5> base = {1+std::ldexp(1.,-26), .5+std::ldexp(1.,-27),
                                    2+std::ldexp(1.,-25), .125+std::ldexp(1.,-29),
                                    .75+std::ldexp(1.,-28)};
  auto weight = [&](int rank,int step,int index) {
    return (rank+1)*(step+1)*base[index]+std::ldexp(double(index+1),-31);
  };
  auto potential = [&](int block,int rank,int step,int index) {
    return block==0 ? constant_potential : .25*(index+1)+.5*rank+.125*step;
  };
  for(int block=0;block<2;++block)
  {
    manager.startBlock(3);
    first.startBlock(3);second.startBlock(3);
    CHECK(first.get_block_weight()==0);
    CHECK(second.get_block_weight()==0);
    std::array<long double,2> local_weight = {0,0};
    for(int step=0;step<3;++step)
    {
      for(int index=0;index<5;++index)
      {
        auto& walker = index<2 ? walkers_first[index] : walkers_second[index-2];
        walker.Weight=weight(comm->rank(),step,index);
        walker.Properties(0,WP::LOCALENERGY)=energy;
        walker.Properties(0,WP::LOCALPOTENTIAL)=potential(block,comm->rank(),step,index);
        local_weight[index<2?0:1]+=walker.Weight;
      }
      first.accumulate(refs_first,psets,wfns,hams,rng);
      second.accumulate(refs_second,psets,wfns,hams,rng);
    }
    CHECK(first.get_block_weight()==Approx(double(local_weight[0])).epsilon(1e-13).margin(1e-13));
    CHECK(second.get_block_weight()==Approx(double(local_weight[1])).epsilon(1e-13).margin(1e-13));
    RefVector<ScalarEstimatorBase> estimators;
    estimators.push_back(first.get_main_estimator());
    estimators.push_back(second.get_main_estimator());
    manager.collectMainEstimators(estimators);
    const double total = exact_weight_argument ? double(local_weight[0]+local_weight[1]) :
        double(first.get_block_weight())+second.get_block_weight();
    manager.stopBlock(11,7,total);

    // Independent long-double weighted reference, including unequal crowds,
    // nonuniform rank weights, a changing observable, and a second block.
    long double global_weight=0, weighted_potential=0;
    for(int rank=0;rank<comm->size();++rank)
      for(int step=0;step<3;++step)
        for(int index=0;index<5;++index)
        {
          const long double w=weight(rank,step,index);
          global_weight+=w;
          weighted_potential+=w*potential(block,rank,step,index);
        }
    if(comm->rank()==0)
    {
      const auto& average=manager.get_AverageCache();
      REQUIRE(average.size()==3);
      CHECK(average[0]==Approx(energy).epsilon(1e-13).margin(1e-13));
      CHECK(average[1]==Approx(energy*energy).epsilon(1e-13).margin(1e-13));
      CHECK(average[2]==Approx(double(weighted_potential/global_weight)).epsilon(1e-13).margin(1e-13));
    }
  }
  manager.stopDriverRun();
}

} // namespace qmcplusplus
