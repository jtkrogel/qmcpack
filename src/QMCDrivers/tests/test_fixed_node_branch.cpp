//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2018 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
//
// File created by: Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"


#include "Utilities/RandomGenerator.h"
#include "OhmmsData/Libxml2Doc.h"
#include "OhmmsPETE/OhmmsMatrix.h"
#include "Particle/ParticleSet.h"
#include "Particle/MCWalkerConfiguration.h"
#include "Estimators/EstimatorManagerBase.h"
#include "QMCDrivers/SimpleFixedNodeBranch.h"


#include <stdio.h>
#include <string>
#include <random>
#include <vector>


using std::string;

namespace qmcplusplus
{
TEST_CASE("Fixed node branch", "[drivers][walker_control]")
{
  Communicate* c = OHMMS::Controller;

  auto emb_uptr = std::make_unique<EstimatorManagerBase>(c);
  auto emb      = emb_uptr.get();

  double tau = 0.5;
  int nideal = 1;
  SimpleFixedNodeBranch fnb(tau, nideal);

  fnb.setEstimatorManager(std::move(emb_uptr));
  REQUIRE(fnb.getEstimatorManager() == emb);

  CHECK(fnb.getTau() == Approx(tau));

  fnb.advanceQMCCounter();
  REQUIRE(fnb.iParam[SimpleFixedNodeBranch::B_COUNTER] == 0);

  // default is classic cutoff scheme
  //fnb.setBranchCutoff();
}

TEST_CASE("Fixed node branch reset before a restarted controller is initialized", "[drivers][legacy][branch_restart]")
{
  Communicate* comm = OHMMS::Controller;
  const int target = 2 * comm->size();
  SimpleFixedNodeBranch branch(.02, target);
  branch.setEstimatorManager(std::make_unique<EstimatorManagerBase>(comm));
  // DMC mode/statistics are serialized; the local controller is not.
  branch.BranchMode.set(SimpleFixedNodeBranch::B_DMC);
  branch.BranchMode.set(SimpleFixedNodeBranch::B_RESTART);
  branch.EnergyHist(-2.75);
  branch.VarianceHist(.4);
  branch.vParam[SimpleFixedNodeBranch::SBVP::EREF] = -2.75;
  branch.vParam[SimpleFixedNodeBranch::SBVP::ETRIAL] = -2.70;
  REQUIRE(branch.WalkerController == nullptr);

  Libxml2Document input;
  REQUIRE(input.parseFromString("<qmc method='dmc'><parameter name='timestep'>.01</parameter>"
                                "<parameter name='warmupSteps'>2</parameter>"
                                "<parameter name='target_walkers'>" + std::to_string(target) + "</parameter></qmc>"));
  CHECK(branch.resetRun(input.getRoot()) == 1);
  CHECK(branch.WalkerController == nullptr);
  CHECK(branch.BranchMode[SimpleFixedNodeBranch::B_RESTART]);
  CHECK(branch.BranchMode[SimpleFixedNodeBranch::B_DMC]);
  CHECK(branch.getTau() == Approx(.01));
  CHECK(branch.iParam[SimpleFixedNodeBranch::B_WARMUPSTEPS] == 2);
  CHECK(branch.iParam[SimpleFixedNodeBranch::B_TARGETWALKERS] == target);
  CHECK(branch.EnergyHist.mean() == Approx(-2.75));
  CHECK(branch.VarianceHist.mean() == Approx(.4));
  CHECK(branch.vParam[SimpleFixedNodeBranch::SBVP::EREF] == Approx(-2.75));
  CHECK(branch.vParam[SimpleFixedNodeBranch::SBVP::ETRIAL] == Approx(-2.70));

  const SimulationCell cell;
  MCWalkerConfiguration walkers(cell);
  walkers.create({1});
  walkers.createWalkers(2);
  std::vector<int> offsets(comm->size() + 1);
  for (int rank = 0; rank <= comm->size(); ++rank)
    offsets[rank] = 2 * rank;
  walkers.setWalkerOffsets(offsets);
  CHECK(branch.initWalkerController(walkers, false, false) == 1);
  CHECK(branch.WalkerController != nullptr);
  CHECK(branch.BranchMode[SimpleFixedNodeBranch::B_RESTART]);
  CHECK(branch.BranchMode[SimpleFixedNodeBranch::B_DMC]);
  CHECK(branch.getTau() == Approx(.01));
  CHECK(branch.EnergyHist.mean() == Approx(-2.75));
  CHECK(branch.VarianceHist.mean() == Approx(.4));
  CHECK(branch.vParam[SimpleFixedNodeBranch::SBVP::ETRIAL] == Approx(-2.70));

  // Once initialized, a later section still reuses/resets the same controller.
  auto* controller = branch.WalkerController.get();
  Libxml2Document continuation;
  REQUIRE(continuation.parseFromString("<qmc method='dmc'><parameter name='timestep'>.005</parameter>"
                                       "<parameter name='warmupSteps'>3</parameter></qmc>"));
  CHECK(branch.resetRun(continuation.getRoot()) == 1);
  CHECK(branch.WalkerController.get() == controller);
  CHECK(branch.getTau() == Approx(.005));
  CHECK(branch.iParam[SimpleFixedNodeBranch::B_WARMUPSTEPS] == 3);
  CHECK(branch.ToDoSteps == 3);
  CHECK_FALSE(branch.BranchMode[SimpleFixedNodeBranch::B_DMCSTAGE]);
}

} // namespace qmcplusplus
