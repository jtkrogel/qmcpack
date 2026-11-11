//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:  Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include <MinimalParticlePool.h>
#include "Particle/VirtualParticleSet.h"
#include "ResourceCollection.h"
#include "DistanceTable.h"
#include "QMCHamiltonians/NLPPJob.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace qmcplusplus
{
TEST_CASE("VirtualParticleSet", "[particle]")
{
  auto pset_pool = MinimalParticlePool::make_NiO_a4(OHMMS::Controller);

  auto& ions  = *pset_pool.getParticleSet("i");
  auto& elecs = *pset_pool.getParticleSet("e");

  elecs.R[0] = {1, 2, 3};
  elecs.R[1] = {2, 1, 3};
  elecs.R[2] = {3, 1, 2};
  elecs.R[3] = {3, 2, 1};

  ions.addTable(ions);
  ions.update();
  elecs.addTable(ions);
  elecs.addTable(elecs);
  elecs.update();


  ParticleSet elecs_clone(elecs);
  elecs_clone.update();

  VirtualParticleSet vp_Ni(elecs);
  VirtualParticleSet vp_O(elecs);

  VirtualParticleSet vp_Ni_clone(elecs_clone);
  VirtualParticleSet vp_O_clone(elecs_clone);

  vp_Ni_clone.makeMoves(elecs_clone, 3, {{0.1, 0.2, 0.3}, {0.2, 0.1, 0.3}, {0.3, 0.1, 0.2}});
  const DistanceTableAB& dt_vp_ion = vp_Ni_clone.getDistTableAB(0);
  CHECK(Approx(dt_vp_ion.getDistances()[2][1]) == 2.5020600118);

  // two walkers form a workgroup.
  // One electron of the first walker gets inside O sphere and one electron of the other gets inside Ni sphere.
  RefVectorWithLeader<VirtualParticleSet> vp_list(vp_Ni, {vp_O, vp_Ni_clone});
  ResourceCollection collection{"NLPPcollection"};
  vp_Ni.createResource(collection);

  {
    ResourceCollectionTeamLock<VirtualParticleSet> vp_res_lock(collection, vp_list);
  }

  vp_Ni_clone.makeMoves(elecs_clone, 3, {{0.1, 0.2, 0.3}, {0.3, 0.1, 0.2}, {0.2, 0.1, 0.3}});
  CHECK(Approx(vp_Ni_clone.R[2][0]) == 3.2);
  CHECK(Approx(vp_Ni_clone.R[2][1]) == 2.1);
  CHECK(Approx(vp_Ni_clone.R[2][2]) == 1.3);

  REQUIRE(dt_vp_ion.getDistances().size() == 3);
  CHECK(Approx(dt_vp_ion.getDistances()[2][1]) == 2.5784519198);
}

TEST_CASE("VirtualParticleSet growing borrowed distance rows", "[particle][virtual_growth]")
{
  auto pool   = MinimalParticlePool::make_NiO_a4(OHMMS::Controller);
  auto& ions  = *pool.getParticleSet("i");
  auto& elecs = *pool.getParticleSet("e");
  elecs.R[0] = {1, 2, 3};
  elecs.R[1] = {2, 1, 3};
  elecs.R[2] = {3, 1, 2};
  elecs.R[3] = {3, 2, 1};
  ions.update();
  const auto ion_table = elecs.addTable(ions);
  elecs.addTable(elecs);
  elecs.getDistTable(ion_table).setModes(DTModes::NEED_VP_FULL_TABLE_ON_HOST);
  elecs.update();
  ParticleSet clone(elecs);
  clone.update();
  VirtualParticleSet vp0(elecs), vp1(clone);

  const auto displacements = [](int count) {
    std::vector<ParticleSet::PosType> values(count);
    for (int i = 0; i < count; ++i)
      values[i] = {0.01 * (i + 1), 0.02, -0.03};
    return values;
  };
  const auto check_distances = [&](const VirtualParticleSet& vp) {
    const auto& rows = vp.getDistTableAB(ion_table).getDistances();
    REQUIRE(rows.size() == vp.getTotalNum());
    for (int i = 0; i < vp.getTotalNum(); ++i)
      for (int j = 0; j < ions.getTotalNum(); ++j)
      {
        double best = std::numeric_limits<double>::max();
        for (int x = -2; x <= 2; ++x)
          for (int y = -2; y <= 2; ++y)
            for (int z = -2; z <= 2; ++z)
            {
              const auto image = ions.getLattice().toCart(ParticleSet::PosType{double(x), double(y), double(z)});
              const ParticleSet::PosType delta = vp.R[i] - ions.R[j] - image;
              best             = std::min(best, std::sqrt(dot(delta, delta)));
            }
        CHECK(rows[i][j] == Approx(best));
      }
  };

  // Growth past std::vector capacity used to copy borrowed distance rows into
  // owning Vectors, making attachReference throw on the OpenMP offload path.
  for (const int count : {1, 3, 17, 2, 35})
  {
    vp0.makeMoves(elecs, 3, displacements(count));
    check_distances(vp0);
  }
  ResourceCollection collection{"growing virtual rows"};
  vp0.createResource(collection);
  RefVectorWithLeader<VirtualParticleSet> vp_list(vp0, {vp0, vp1});
  RefVectorWithLeader<ParticleSet> refs(elecs, {elecs, clone});
  const NLPPJob<ParticleSet::RealType> job0(0, 3, 1.0, {0, 0, 0});
  const NLPPJob<ParticleSet::RealType> job1(0, 2, 1.0, {0, 0, 0});
  const RefVector<const NLPPJob<ParticleSet::RealType>> jobs{job0, job1};
  {
    ResourceCollectionTeamLock<VirtualParticleSet> lock(collection, vp_list);
    for (const int count : {1, 19, 2, 47})
    {
      const auto delta0 = displacements(count), delta1 = displacements(count + 3);
      const RefVector<const std::vector<ParticleSet::PosType>> deltas{delta0, delta1};
      VirtualParticleSet::mw_makeMoves(vp_list, refs, deltas, jobs, false);
      check_distances(vp0);
      check_distances(vp1);
    }
  }
  vp0.makeMoves(elecs, 3, displacements(5));
  check_distances(vp0);
}
} // namespace qmcplusplus
