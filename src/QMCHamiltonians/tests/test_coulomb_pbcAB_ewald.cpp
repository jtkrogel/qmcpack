//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "OhmmsData/Libxml2Doc.h"
#include "OhmmsPETE/OhmmsMatrix.h"
#include "Particle/ParticleSet.h"
#include "QMCHamiltonians/CoulombPBCAB.h"
#include "QMCHamiltonians/CoulombPBCAA.h"
#include "LongRange/EwaldHandler3D.h"

#include <stdio.h>
#include <string>
#include <cmath>
#include <memory>
#include <vector>

using std::string;

namespace qmcplusplus
{
TEST_CASE("Coulomb PBC A-B Ewald3D", "[hamiltonian]")
{
  Lattice lattice;
  lattice.BoxBConds = true; // periodic
  lattice.R.diagonal(1.0);
  lattice.reset();

  const SimulationCell simulation_cell(lattice);
  ParticleSet ions(simulation_cell);
  ParticleSet elec(simulation_cell);

  ions.setName("ion");
  ions.create({1});
  ions.R[0]                     = {0.0, 0.0, 0.0};
  SpeciesSet& ion_species       = ions.getSpeciesSet();
  int pIdx                      = ion_species.addSpecies("H");
  int pChargeIdx                = ion_species.addAttribute("charge");
  ion_species(pChargeIdx, pIdx) = 1;
  ions.createSK();
  ions.update();


  elec.setName("elec");
  elec.create({1});
  elec.R[0]                  = {0.5, 0.0, 0.0};
  SpeciesSet& tspecies       = elec.getSpeciesSet();
  int upIdx                  = tspecies.addSpecies("u");
  int chargeIdx              = tspecies.addAttribute("charge");
  int massIdx                = tspecies.addAttribute("mass");
  tspecies(chargeIdx, upIdx) = -1;
  tspecies(massIdx, upIdx)   = 1.0;

  elec.resetGroups();
  elec.createSK();
  elec.addTable(ions);
  elec.update();


  LRCoulombSingleton::CoulombHandler = std::make_unique<EwaldHandler3D>(ions);
  LRCoulombSingleton::CoulombHandler->initBreakup(ions);


  CoulombPBCAB cab(ions, elec);

  // Self energy plus Background charge term
  double consts = cab.evalConsts(elec);
  CHECK(consts == Approx(0.0523598776 * 2)); //not validated

  double val_ei = cab.evaluate(elec);
  CHECK(val_ei == Approx(-0.008302 + 0.0523598776 * 2)); //Not validated

  CoulombPBCAA caa_elec(elec, true, false, false);
  CoulombPBCAA caa_ion(ions, false, false, false);
  double val_ee = caa_elec.evaluate(elec);
  double val_ii = caa_ion.evaluate(ions);
  double sum    = val_ee + val_ii + val_ei;

  CHECK(val_ee == Approx(-1.418927));
  CHECK(val_ii == Approx(-1.418927));
  CHECK(sum == Approx(-2.741436)); // Can be validated via Ewald summation elsewhere
                                   // -2.74136517454081

  LRCoulombSingleton::CoulombHandler.reset(nullptr);
}

TEST_CASE("Coulomb PBC A-B BCC H Ewald3D", "[hamiltonian]")
{
  Lattice lattice;
  lattice.BoxBConds = true; // periodic
  lattice.R.diagonal(3.77945227);
  lattice.reset();

  const SimulationCell simulation_cell(lattice);
  ParticleSet ions(simulation_cell);
  ParticleSet elec(simulation_cell);

  ions.setName("ion");
  ions.create({2});
  ions.R[0]                     = {0.0, 0.0, 0.0};
  ions.R[1]                     = {1.88972614, 1.88972614, 1.88972614};
  SpeciesSet& ion_species       = ions.getSpeciesSet();
  int pIdx                      = ion_species.addSpecies("H");
  int pChargeIdx                = ion_species.addAttribute("charge");
  ion_species(pChargeIdx, pIdx) = 1;
  ions.createSK();
  ions.update();


  elec.setName("elec");
  elec.create({2});
  elec.R[0]                  = {0.5, 0.0, 0.0};
  elec.R[1]                  = {0.0, 0.5, 0.0};
  SpeciesSet& tspecies       = elec.getSpeciesSet();
  int upIdx                  = tspecies.addSpecies("u");
  int chargeIdx              = tspecies.addAttribute("charge");
  int massIdx                = tspecies.addAttribute("mass");
  tspecies(chargeIdx, upIdx) = -1;
  tspecies(massIdx, upIdx)   = 1.0;

  elec.resetGroups();
  elec.createSK();
  elec.addTable(ions);
  elec.update();


  LRCoulombSingleton::CoulombHandler = std::make_unique<EwaldHandler3D>(ions);
  LRCoulombSingleton::CoulombHandler->initBreakup(ions);

  CoulombPBCAB cab(ions, elec);

  // Background charge term
  double consts = cab.evalConsts(elec);
  CHECK(consts == Approx(0.0277076538 * 4)); //not validated


  double val_ei = cab.evaluate(elec);
  CHECK(val_ei == Approx(-2.223413 + 0.0277076538 * 4)); //Not validated


  CoulombPBCAA caa_elec(elec, false, false, false);
  CoulombPBCAA caa_ion(ions, false, false, false);
  double val_ee = caa_elec.evaluate(elec);
  double val_ii = caa_ion.evaluate(ions);
  double sum    = val_ee + val_ii + val_ei;

  CHECK(val_ee == Approx(-0.012808 - 0.0277076538 * 2));
  CHECK(val_ii == Approx(-0.907659 - 0.0277076538 * 2));
  CHECK(sum == Approx(-3.143880)); // Can be validated via Ewald summation elsewhere
                                   // -3.14349127313640

  LRCoulombSingleton::CoulombHandler.reset(nullptr);
}

TEST_CASE("Mixed periodic local potentials retain bare species across cloning", "[hamiltonian][mixed_local_pp]")
{
  // Restore the shared breakup owner even when an assertion exits the test early.
  struct RestoreHandler
  {
    decltype(LRCoulombSingleton::CoulombHandler) saved = std::move(LRCoulombSingleton::CoulombHandler);
    ~RestoreHandler() { LRCoulombSingleton::CoulombHandler = std::move(saved); }
  } restore;
  Lattice lattice;
  lattice.R = {6.6, 0, 0, 0, 6.6, 0, 0, 0, 6.6};
  lattice.BoxBConds = true;
  lattice.reset();
  const SimulationCell cell(lattice);
  ParticleSet ions(cell), electrons(cell);
  ions.setName("ion0");
  ions.create({1, 1});
  auto& species = ions.getSpeciesSet();
  species.addSpecies("Li");
  species.addSpecies("H");
  const int charge = species.addAttribute("charge");
  species(charge, 0) = species(charge, 1) = 1;
  ions.R[0] = {0, 0, 0};
  ions.R[1] = {3.3, 3.3, 3.3};
  ions.resetGroups();
  ions.createSK();
  ions.update();
  electrons.setName("e");
  electrons.create({2});
  auto& electronic = electrons.getSpeciesSet();
  electronic.addSpecies("u");
  electronic(electronic.addAttribute("charge"), 0) = -1;
  electronic(electronic.addAttribute("mass"), 0) = 1;
  electrons.R[0] = {.7, .4, .2};
  electrons.R[1] = {2.8, 3.1, 3.5};
  electrons.resetGroups();
  electrons.createSK();
  electrons.addTable(ions);
  electrons.update();
  LRCoulombSingleton::CoulombHandler = std::make_unique<EwaldHandler3D>(ions);
  LRCoulombSingleton::CoulombHandler->initBreakup(ions);
  auto potential = std::make_unique<CoulombPBCAB>(ions, electrons);
  const double baseline = potential->evaluate(electrons);
  std::weak_ptr<const CoulombPBCAB::RadFunctorType> bare_owner = potential->V0;
  const auto* bare = potential->V0.get();

  // Replace Li alone with an analytic compact softening; H must remain exactly bare.
  LinearGrid<double> grid;
  constexpr int count = 5001;
  grid.set(0, potential->myRcut, count);
  std::vector<double> radial(count);
  for (int point = 0; point < count; ++point)
  {
    const double r = grid[point];
    radial[point] = 1 - (r < 2 ? std::pow(1 - r * r / 4, 4) : 0);
  }
  auto pp = std::make_unique<CoulombPBCAB::RadFunctorType>(grid.makeClone(), radial);
  pp->spline(0, 0, count - 1, 0);
  potential->add(0, std::move(pp));
  CHECK(potential->Vat[1] == bare);
  REQUIRE_FALSE(bare_owner.expired());
  REQUIRE(potential->V0);
  double expected = baseline;
  for (const auto& position : electrons.R)
  {
    const double r = std::sqrt(dot(position, position));
    if (r < 2)
      expected += std::pow(1 - r * r / 4, 4) / r;
  }
  CHECK(potential->evaluate(electrons) == Approx(expected).margin(2e-8));
  ParticleSet copy(electrons);
  copy.update();
  auto clone = potential->makeClone(copy);
  potential.reset();
  CHECK_FALSE(bare_owner.expired());
  CHECK(dynamic_cast<CoulombPBCAB&>(*clone).evaluate(copy) == Approx(expected).margin(2e-8));
}

TEST_CASE("Periodic local radial origin preserves the supplied endpoint", "[hamiltonian][local_pp_origin]")
{
  struct RestoreHandler
  {
    decltype(LRCoulombSingleton::CoulombHandler) saved = std::move(LRCoulombSingleton::CoulombHandler);
    ~RestoreHandler() { LRCoulombSingleton::CoulombHandler = std::move(saved); }
  } restore;
  Lattice lattice;
  lattice.R = {6.6, 0, 0, 0, 6.6, 0, 0, 0, 6.6};
  lattice.BoxBConds = true;
  lattice.reset();
  const SimulationCell cell(lattice);
  ParticleSet ions(cell), electrons(cell);
  ions.setName("ion");
  ions.create({1});
  auto& ionic = ions.getSpeciesSet();
  ionic.addSpecies("X");
  ionic(ionic.addAttribute("charge"), 0) = 1;
  ions.R[0] = {0, 0, 0};
  ions.resetGroups();
  ions.createSK();
  ions.update();
  electrons.setName("e");
  electrons.create({1});
  auto& electronic = electrons.getSpeciesSet();
  electronic.addSpecies("u");
  electronic(electronic.addAttribute("charge"), 0) = -1;
  electronic(electronic.addAttribute("mass"), 0) = 1;
  electrons.R[0] = {.2, .3, .4};
  electrons.resetGroups();
  electrons.createSK();
  electrons.update();
  LRCoulombSingleton::CoulombHandler = std::make_unique<EwaldHandler3D>(ions);
  LRCoulombSingleton::CoulombHandler->initBreakup(ions);

  for (double endpoint : {0.0, 0.37})
  {
    CoulombPBCAB potential(ions, electrons);
    LinearGrid<double> grid;
    constexpr int count = 10001;
    grid.set(0, 10, count);
    std::vector<double> values(count);
    const double slope = (1 - endpoint) * 2 / std::sqrt(std::acos(-1.0));
    for (int point = 0; point < count; ++point)
      values[point] = endpoint + (1 - endpoint) * std::erf(grid[point]);
    auto pp = std::make_unique<CoulombPBCAB::RadFunctorType>(grid.makeClone(), values);
    pp->spline(0, slope, count - 1, 0);
    potential.add(0, std::move(pp));
    const auto& radial = *potential.Vspec[0];
    double first, second;
    CHECK(radial.splint(0, first, second) == Approx(endpoint).margin(1e-15));
    CHECK(first == Approx(slope - potential.AB->evaluateLR_r0()).margin(1e-12));
    // A zero supplied endpoint must remain regular even far below the first knot.
    // Nonzero endpoints retain their declared Coulomb singularity instead.
    for (double r : {1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2})
    {
      const double reconstructed = radial.splint(r) / r + potential.AB->evaluateLR(r);
      const double expected = endpoint / r + (1 - endpoint) * std::erf(r) / r;
      CHECK(reconstructed == Approx(expected).margin(1e-7));
      electrons.R[0] = {r, 0, 0};
      electrons.update();
      const double bare_difference = r < potential.V0->grid()[1]
          ? -r * potential.AB->evaluateLR(r)
          : potential.V0->splint(r) - 1;
      CHECK(potential.evalSR(electrons) == Approx(-expected - bare_difference / r).margin(1e-7));
    }
  }
}

TEST_CASE("Periodic local energy retains the nearest-image radial correction", "[hamiltonian][local_pp_tail]")
{
  // Isolate both the cached breakup and its policy from preceding suite cases.
  struct RestoreHandler
  {
    decltype(LRCoulombSingleton::CoulombHandler) saved = std::move(LRCoulombSingleton::CoulombHandler);
    const LRCoulombSingleton::lr_type saved_type = LRCoulombSingleton::this_lr_type;
    ~RestoreHandler()
    {
      LRCoulombSingleton::CoulombHandler = std::move(saved);
      LRCoulombSingleton::this_lr_type = saved_type;
    }
  } restore;
  const bool ordinary_ewald = GENERATE(false, true);
  LRCoulombSingleton::CoulombHandler.reset();
  LRCoulombSingleton::this_lr_type = LRCoulombSingleton::ESLER;
  Lattice lattice;
  constexpr double a = 3.837088895929358;
  lattice.R = {0, a, a, a, 0, a, a, a, 0};
  lattice.BoxBConds = true;
  lattice.LR_dim_cutoff = 15;
  lattice.reset();
  const SimulationCell cell(lattice);
  ParticleSet ions(cell), electrons(cell);
  ions.setName("ion");
  ions.create({1, 1});
  auto& ionic = ions.getSpeciesSet();
  ionic.addSpecies("Li");
  ionic.addSpecies("H");
  const int charge = ionic.addAttribute("charge");
  ionic(charge, 0) = ionic(charge, 1) = 1;
  ions.R[0] = {0, 0, 0};
  ions.R[1] = {a, a, a};
  ions.resetGroups();
  ions.createSK();
  ions.update();
  electrons.setName("e");
  electrons.create({1, 1});
  auto& electronic = electrons.getSpeciesSet();
  electronic.addSpecies("u");
  electronic.addSpecies("d");
  const int electron_charge = electronic.addAttribute("charge");
  const int mass = electronic.addAttribute("mass");
  for (int spin = 0; spin < 2; ++spin)
  {
    electronic(electron_charge, spin) = -1;
    electronic(mass, spin) = 1;
  }
  electrons.R[0] = {.2, .3, .4};
  electrons.R[1] = {.7, .8, .9};
  electrons.resetGroups();
  electrons.createSK();
  electrons.update();
  if (ordinary_ewald)
  {
    LRCoulombSingleton::CoulombHandler = std::make_unique<EwaldHandler3D>(ions);
    LRCoulombSingleton::CoulombHandler->initBreakup(ions);
  }
  CoulombPBCAB bare(ions, electrons), local(ions, electrons);
  auto correction = [](double r) {
    return std::exp(-15 * r * r) / r - 1.24272969818004 * std::exp(-1.80605426846072 * r * r)
        + 15 * r * std::exp(-15.0479971422127 * r * r);
  };
  LinearGrid<double> grid;
  constexpr int count = 100000;
  grid.set(0, 10, count);
  std::vector<double> radial(count);
  radial[0] = 0;
  for (int point = 1; point < count; ++point)
    radial[point] = 1 - grid[point] * correction(grid[point]);
  auto pp = std::make_unique<CoulombPBCAB::RadFunctorType>(grid.makeClone(), radial);
  pp->spline(0, 1.24272969818004, count - 1, 0);
  local.add(0, std::move(pp));
  CHECK_FALSE(pp);
  REQUIRE(local.Vat[1] == local.V0.get());
  // Sweep across and beyond the Coulomb cutoff, then sample fractional-cell points.
  for (int point = 0; point < 99; ++point)
  {
    if (point >= 96)
      electrons.R[0] = {local.V0->grid()[1] * (1 + (point - 97) * 1e-8), 0, 0};
    else if (point < 32)
      electrons.R[0] = {local.myRcut - .01 + point * .001, 0, 0};
    else
    {
      const double u = std::fmod(.173 + point * .6180339887498949, 1.);
      const double v = std::fmod(.291 + point * .4142135623730950, 1.);
      const double w = std::fmod(.437 + point * .7320508075688772, 1.);
      electrons.R[0] = {a * (v + w), a * (u + w), a * (u + v)};
    }
    electrons.update();
    double expected = bare.evaluate(electrons);
    const auto& distances = electrons.getDistTableAB(local.myTableIndex);
    for (int electron = 0; electron < 2; ++electron)
      expected += correction(distances.getDistRow(electron)[0]);
    CHECK(local.evaluate(electrons) == Approx(expected).margin(1e-6));
    if (point == 98)
    {
      ParticleSet copy(electrons);
      copy.update();
      auto clone = local.makeClone(copy);
      CHECK(dynamic_cast<CoulombPBCAB&>(*clone).evaluate(copy) == Approx(expected).margin(1e-6));
    }
  }
}

} // namespace qmcplusplus
