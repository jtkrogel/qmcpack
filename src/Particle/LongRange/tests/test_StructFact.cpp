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
#include "Utilities/for_testing/Catch2Approx.h"

#include <vector>
#include <complex>
#include "ParticleSet.h"
#include "LongRange/StructFact.h"

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

TEST_CASE("StructFact coordinate reuse invalidates on positions species and reciprocal list", "[lrhandler][ewald]")
{
  Lattice lattice;
  lattice.BoxBConds = true;
  lattice.R.diagonal(5.0);
  lattice.LR_dim_cutoff = 6.0;
  lattice.reset();
  const SimulationCell cell(lattice);
  ParticleSet particles(cell);
  particles.getSpeciesSet().addSpecies("u");
  particles.getSpeciesSet().addSpecies("d");
  particles.create({2, 1});
  particles.R[0] = {0.1, 0.2, 0.3};
  particles.R[1] = {0.7, 0.5, 0.4};
  particles.R[2] = {1.2, 0.8, 1.7};
  KContainer k_list;
  k_list.updateKLists(lattice, 4.0, 3);
  StructFact sk(particles.getLRBox(), k_list);
  const auto check = [&]() {
    sk.updateAllPart(particles);
    for (int group = 0; group < particles.groups(); ++group)
      for (int k = 0; k < k_list.getNumK(); ++k)
      {
        double real = 0, imag = 0;
        for (int i = 0; i < particles.getTotalNum(); ++i)
          if (particles.getGroupID(i) == group)
          {
            const double phase = dot(k_list.getKptsCart()[k], particles.R[i]);
            real += std::cos(phase);
            imag += std::sin(phase);
          }
        CHECK(sk.rhok_r[group][k] == Approx(real).margin(1e-11));
        CHECK(sk.rhok_i[group][k] == Approx(imag).margin(1e-11));
      }
  };
  check();
  check(); // identical-coordinate reuse
  particles.R[1][0] += 0.25;
  check();
  std::swap(particles.GroupID[0], particles.GroupID[2]);
  check();
  const int original_nk = k_list.getNumK();
  Lattice expanded(lattice);
  expanded.R *= 1.2;
  expanded.reset();
  k_list.updateKLists(expanded, 4.0 / 1.2, 3);
  REQUIRE(k_list.getNumK() == original_nk);
  check(); // same-size list with different Cartesian vectors
  KContainer replacement;
  replacement.updateKLists(lattice, 4.0, 3);
  k_list = replacement;
  check(); // assignment from another list must not alias its generation
  sk.turnOnStorePerParticle(particles);
  particles.R[0][1] -= 0.1;
  check();
  for (int i = 0; i < particles.getTotalNum(); ++i)
    for (int k = 0; k < k_list.getNumK(); ++k)
    {
      const double phase = dot(k_list.getKptsCart()[k], particles.R[i]);
      CHECK(sk.eikr_r[i][k] == Approx(std::cos(phase)).margin(1e-11));
      CHECK(sk.eikr_i[i][k] == Approx(std::sin(phase)).margin(1e-11));
    }
}

} // namespace qmcplusplus
