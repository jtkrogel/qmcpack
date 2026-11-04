//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 and QMCPACK developers.
//
// File developed by: Yubo "Paul" Yang, yubo.paul.yang@gmail.com, University of Illinois Urbana-Champaign
//
// File created by: Yubo "Paul" Yang, yubo.paul.yang@gmail.com, University of Illinois Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include <vector>
#include <complex>
#include "ParticleSet.h"
#include "LongRange/StructFact.h"
#include "ResourceCollection.h"

namespace qmcplusplus
{
/** evalaute bare Coulomb in 3D using LRHandlerTemp
 */
TEST_CASE("StructFact", "[lrhandler]")
{
  Lattice lattice;
  lattice.BoxBConds     = true;
  lattice.LR_dim_cutoff = 30.;
  lattice.R.diagonal(5.0);
  lattice.reset();
  CHECK(Approx(lattice.Volume) == 125);
  lattice.SetLRCutoffs(lattice.Rv);
  lattice.printCutoffs(app_log());
  CHECK(Approx(lattice.LR_rc) == 2.5);
  CHECK(Approx(lattice.LR_kc) == 12);

  lattice.LR_dim_cutoff = 125;
  const SimulationCell simulation_cell(lattice);
  ParticleSet ref(simulation_cell); // handler needs ref.getSimulationCell().getKLists()

  SpeciesSet& tspecies = ref.getSpeciesSet();
  tspecies.addSpecies("u");
  tspecies.addSpecies("d");

  ref.create({3, 1});
  ref.R[0] = {0.0, 1.0, 2.0};
  ref.R[1] = {1.0, 0.2, 3.0};
  ref.R[2] = {0.3, 4.0, 1.4};
  ref.R[3] = {3.2, 4.7, 0.7};

  REQUIRE(simulation_cell.getKLists().getNumK() == 263786);
  StructFact sk(ref.getLRBox(), simulation_cell.getKLists());
  sk.updateAllPart(ref);

  CHECK(sk.rhok_r.rows() == ref.groups());
  CHECK(sk.rhok_i.rows() == ref.groups());
  CHECK(sk.rhok_r.cols() == simulation_cell.getKLists().getNumK());
  CHECK(sk.rhok_i.cols() == simulation_cell.getKLists().getNumK());

  std::vector<std::complex<double>> rhok_sum_ref{-125.80618630936, 68.199075127271};

  for (int i = 0; i < ref.groups(); i++)
  {
    std::complex<QMCTraits::FullPrecRealType> rhok_sum;
    for (int ik = 0; ik < simulation_cell.getKLists().getNumK(); ik++)
      rhok_sum += std::complex<QMCTraits::FullPrecRealType>(sk.rhok_r[i][ik], sk.rhok_i[i][ik]);

    //app_log() << std::setprecision(14) << rhok_sum << std::endl;
    CHECK(ComplexApprox(rhok_sum).epsilon(5e-5) == rhok_sum_ref[i]);
  }
}

TEST_CASE("StructFact batched phases and per-particle storage", "[lrhandler][structfact_batch]")
{
  const bool skew = GENERATE(false, true);
  const auto kind = GENERATE(DynamicCoordinateKind::DC_POS, DynamicCoordinateKind::DC_POS_OFFLOAD);
  Lattice lattice;
  lattice.BoxBConds     = true;
  lattice.LR_dim_cutoff = 8.0;
  lattice.R = skew ? ParticleSet::Tensor_t(4.2, .3, -.2, .4, 3.8, .1, -.1, .2, 5.3)
                   : ParticleSet::Tensor_t(5., 0., 0., 0., 5., 0., 0., 0., 5.);
  lattice.reset();
  const SimulationCell cell(lattice);
  ParticleSet first(cell, kind);
  first.getSpeciesSet().addSpecies("u");
  first.getSpeciesSet().addSpecies("d");
  first.create({3, 1});
  first.R[0] = {-.000000001, .2, 5.300000001};
  first.R[1] = {.33, 2.14, -.4};
  first.R[2] = {-.72, 1.23, 2.4};
  first.R[3] = {1.7, .6, 2.77};
  first.update();
  ParticleSet second(first);
  second.R[1][0] += .17;
  second.update();
  RefVectorWithLeader<ParticleSet> particles(first, {first, second});
  ResourceCollection resources("structure_factor_batch");
  first.createResource(resources);
  ResourceCollectionTeamLock<ParticleSet> lock(resources, particles);
  ParticleSet::mw_update(particles);
  StructFact first_sk(first.getLRBox(), cell.getKLists());
  StructFact second_sk(second.getLRBox(), cell.getKLists());
  // Production initializes each structure factor before batched updates.
  first_sk.updateAllPart(first);
  second_sk.updateAllPart(second);
  RefVectorWithLeader<StructFact> factors(first_sk, {first_sk, second_sk});
  SKMultiWalkerMem staging;
  const double tolerance = sizeof(QMCTraits::RealType) == sizeof(float) ? 3e-5 : 1e-12;
  auto check_phases       = [&]()
  {
    for (int walker = 0; walker < 2; ++walker)
      for (int group = 0; group < particles[walker].groups(); ++group)
        for (int ik = 0; ik < cell.getKLists().getNumK(); ++ik)
        {
          std::complex<double> expected{};
          for (int ip = particles[walker].first(group); ip < particles[walker].last(group); ++ip)
          {
            double phase = 0.0;
            for (int axis = 0; axis < 3; ++axis)
              phase += static_cast<double>(cell.getKLists().getKptsCart()[ik][axis]) *
                  static_cast<double>(particles[walker].R[ip][axis]);
            const std::complex<double> one(std::cos(phase), std::sin(phase));
            expected += one;
            if (factors[walker].isStorePerParticle())
            {
              CHECK(std::abs(factors[walker].eikr_r(ip, ik) - one.real()) < tolerance);
              CHECK(std::abs(factors[walker].eikr_i(ip, ik) - one.imag()) < tolerance);
            }
          }
          CHECK(std::abs(factors[walker].rhok_r(group, ik) - expected.real()) < tolerance);
          CHECK(std::abs(factors[walker].rhok_i(group, ik) - expected.imag()) < tolerance);
        }
  };
  StructFact::mw_updateAllPart(factors, particles, staging);
  check_phases();
  second.R[2][1] -= .23;
  ParticleSet::mw_update(particles);
  StructFact::mw_updateAllPart(factors, particles, staging);
  check_phases();
  first_sk.turnOnStorePerParticle(first);
  second_sk.turnOnStorePerParticle(second);
  StructFact::mw_updateAllPart(factors, particles, staging);
  check_phases();
}

} // namespace qmcplusplus
