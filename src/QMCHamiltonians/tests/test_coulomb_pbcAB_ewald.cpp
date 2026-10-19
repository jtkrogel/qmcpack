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

} // namespace qmcplusplus
