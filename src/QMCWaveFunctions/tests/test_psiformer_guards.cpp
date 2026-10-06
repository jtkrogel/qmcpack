//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_guards.cpp
 * @brief External-data-free tests for unsupported PsiFormer execution semantics.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "OhmmsData/Libxml2Doc.h"
#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWaveFunctionBuilder.h"
#include "psiformer_test_utils.h"

#include <complex>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus
{
namespace
{
using namespace testing::psiformer;

/// Construct a LiH electron set with explicit masses for builder validation.
ParticleSet makeMassTaggedElectrons(const SimulationCell& simulation_cell,
                                    double up_mass,
                                    double down_mass)
{
  const Geometry geometry = makeGeometry("lih");
  ParticleSet electrons(simulation_cell);
  electrons.setName("e");
  electrons.create({2, 2});

  SpeciesSet& species = electrons.getSpeciesSet();
  const int up        = species.addSpecies("u");
  const int down      = species.addSpecies("d");
  const int mass      = species.addAttribute("mass");
  species(mass, up)   = up_mass;
  species(mass, down) = down_mass;
  electrons.resetGroups();

  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      electrons.R[electron][dimension] = geometry.electrons[3 * electron + dimension];
  electrons.update();
  return electrons;
}

/// Construct source ions matching the generated all-electron LiH export.
std::unique_ptr<ParticleSet> makeGuardTestIons(const SimulationCell& simulation_cell)
{
  const Geometry geometry = makeGeometry("lih");
  auto ions               = std::make_unique<ParticleSet>(simulation_cell);
  ions->setName("ion0");
  ions->create(std::vector<int>(geometry.charges.size(), 1));

  SpeciesSet& species = ions->getSpeciesSet();
  const int charge     = species.addAttribute("charge");
  for (std::size_t nucleus = 0; nucleus < geometry.charges.size(); ++nucleus)
  {
    species.addSpecies("ion_" + std::to_string(nucleus));
    species(charge, nucleus) = geometry.charges[nucleus];
    for (int dimension = 0; dimension < 3; ++dimension)
      ions->R[nucleus][dimension] = geometry.nuclei[3 * nucleus + dimension];
  }
  ions->resetGroups();
  ions->update();
  return ions;
}

/// Format the common fixed or optimizable PsiFormer XML used by guard tests.
std::string makePsiFormerXml(const GeneratedFiles& files, bool optimize)
{
  std::ostringstream xml;
  xml << "<psiformer name=\"pf_guard\" parameters=\"" << files.parameters.string()
      << "\" configuration=\"" << files.configuration.string() << "\"";
  if (optimize)
    xml << " system=\"all_electron\" optimize=\"yes\" optimize_indices=\"0\"";
  xml << "/>";
  return xml.str();
}

} // namespace

TEST_CASE("PsiFormer builder requires an explicit periodic feature policy",
          "[wavefunction][psiformer][hardening]")
{
  GeneratedFiles files = generateFiles("lih");

  Lattice lattice;
  lattice.R         = {30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0};
  lattice.BoxBConds = {true, true, true};
  lattice.reset();
  const SimulationCell periodic_cell(lattice);

  ParticleSet electrons = makeMassTaggedElectrons(periodic_cell, 1.0, 1.0);
  WaveFunctionComponentBuilder::PSetMap particle_sets;
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, electrons, particle_sets);
  Libxml2Document document;
  REQUIRE(document.parseFromString(makePsiFormerXml(files, false)));
  CHECK_THROWS_WITH(builder.buildComponent(document.getRoot()),
                    Catch::Matchers::ContainsSubstring("feature_policy=periodic_torus_v1"));
}

TEST_CASE("PsiFormer periodic import binds ordered runtime ions",
          "[wavefunction][psiformer][hardening][periodic]")
{
  GeneratedFiles files = generateFiles("lih", 4, 7);

  Lattice lattice;
  lattice.R         = {8.0, 0.0, 0.0, 0.6, 7.4, 0.0, -0.3, 0.5, 8.5};
  lattice.BoxBConds = {true, true, true};
  lattice.reset();
  const SimulationCell periodic_cell(lattice);

  const auto make_document = [&files]() {
    std::ostringstream xml;
    xml << "<psiformer name=\"pf_periodic_import\" parameters=\""
        << files.parameters.string() << "\" configuration=\""
        << files.configuration.string()
        << "\" source=\"ion0\" system=\"all_electron\" "
           "feature_policy=\"periodic_torus_v1\"/>";
    Libxml2Document document;
    if (!document.parseFromString(xml.str()))
      throw std::runtime_error("Unable to parse periodic PsiFormer guard XML");
    return document;
  };

  SECTION("common translation and individual images")
  {
    ParticleSet electrons = makeMassTaggedElectrons(periodic_cell, 1.0, 1.0);
    WaveFunctionComponentBuilder::PSetMap particle_sets;
    auto ions = makeGuardTestIons(periodic_cell);
    const ParticleSet::PosType translation{0.31, -0.27, 0.18};
    for (int nucleus = 0; nucleus < ions->getTotalNum(); ++nucleus)
      ions->R[nucleus] += translation;
    for (int dimension = 0; dimension < 3; ++dimension)
      ions->R[1][dimension] += lattice.R(1, dimension);
    ions->update();
    particle_sets.emplace(ions->getName(), std::move(ions));

    PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, electrons, particle_sets);
    Libxml2Document document = make_document();
    CHECK_NOTHROW(builder.buildComponent(document.getRoot()));
  }

  SECTION("non-image geometry change")
  {
    ParticleSet electrons = makeMassTaggedElectrons(periodic_cell, 1.0, 1.0);
    WaveFunctionComponentBuilder::PSetMap particle_sets;
    auto ions = makeGuardTestIons(periodic_cell);
    ions->R[1][0] += 0.125;
    ions->update();
    particle_sets.emplace(ions->getName(), std::move(ions));

    PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, electrons, particle_sets);
    Libxml2Document document = make_document();
    CHECK_THROWS_WITH(builder.buildComponent(document.getRoot()),
                      Catch::Matchers::ContainsSubstring("ordering/geometry"));
  }
}

TEST_CASE("PsiFormer system validation rejects mismatched electron and source-ion lattices",
          "[wavefunction][psiformer][hardening][capability]")
{
  GeneratedFiles files = generateFiles("lih");

  Lattice lattice;
  lattice.R         = {30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0};
  lattice.BoxBConds = {true, true, true};
  lattice.reset();
  const SimulationCell periodic_cell(lattice);
  const SimulationCell open_cell;

  PsiFormerWF component("pf_guard", files.parameters.string(), files.configuration.string());

  SECTION("electron lattice")
  {
    ParticleSet electrons = makeMassTaggedElectrons(periodic_cell, 1.0, 1.0);
    auto ions             = makeGuardTestIons(open_cell);
    CHECK_THROWS_WITH(component.validateSystem(electrons, *ions, "all_electron"),
                      Catch::Matchers::ContainsSubstring("boundary conditions differ"));
  }

  SECTION("source-ion lattice")
  {
    ParticleSet electrons = makeMassTaggedElectrons(open_cell, 1.0, 1.0);
    auto ions             = makeGuardTestIons(periodic_cell);
    CHECK_THROWS_WITH(component.validateSystem(electrons, *ions, "all_electron"),
                      Catch::Matchers::ContainsSubstring("boundary conditions differ"));
  }

  SECTION("different open bounding cells")
  {
    Lattice alternate_lattice;
    alternate_lattice.R = {17.0, 0.0, 0.0, 0.0, 19.0, 0.0, 0.0, 0.0, 23.0};
    alternate_lattice.BoxBConds = {false, false, false};
    alternate_lattice.reset();
    const SimulationCell alternate_open_cell(alternate_lattice);

    ParticleSet electrons = makeMassTaggedElectrons(open_cell, 1.0, 1.0);
    auto ions             = makeGuardTestIons(alternate_open_cell);
    CHECK_NOTHROW(component.validateSystem(electrons, *ions, "all_electron"));
  }
}

TEST_CASE("PsiFormer system rebinding preserves a pending proposal",
          "[wavefunction][psiformer][hardening][lifecycle]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell open_cell;
  ParticleSet electrons = makeMassTaggedElectrons(open_cell, 1.0, 1.0);
  ParticleSet rebound_electrons(electrons);
  auto ions = makeGuardTestIons(open_cell);
  PsiFormerWF component("pf_guard", files.parameters.string(),
                        files.configuration.string());
  component.validateSystem(electrons, *ions, "all_electron");

  ParticleSet::ParticleGradient gradient(electrons.getTotalNum());
  ParticleSet::ParticleLaplacian laplacian(electrons.getTotalNum());
  gradient = QMCTraits::ValueType(0);
  laplacian = QMCTraits::ValueType(0);
  component.evaluateLog(electrons, gradient, laplacian);
  electrons.makeMove(0, ParticleSet::PosType{0.01, -0.005, 0.002});
  component.ratio(electrons, 0);

  CHECK_THROWS_WITH(
      component.validateSystem(rebound_electrons, *ions, "all_electron"),
      Catch::Matchers::ContainsSubstring("proposal is pending"));

  // The failed rebind did not consume the proposal; its original scalar
  // resolver remains valid, after which the same rebind succeeds.
  CHECK_NOTHROW(component.restore(0));
  electrons.rejectMove(0);
  CHECK_NOTHROW(component.validateSystem(
      rebound_electrons, *ions, "all_electron"));
}

TEST_CASE("PsiFormer rejects both source-gradient force interfaces",
          "[wavefunction][psiformer][hardening][capability]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell open_cell;
  ParticleSet electrons = makeMassTaggedElectrons(open_cell, 1.0, 1.0);
  auto ions             = makeGuardTestIons(open_cell);
  PsiFormerWF component("pf_guard", files.parameters.string(), files.configuration.string());
  WaveFunctionComponent& base_component = component;

  CHECK_THROWS_WITH(base_component.evalGradSource(electrons, *ions, 0),
                    Catch::Matchers::ContainsSubstring("fixed nuclei"));

  TinyVector<ParticleSet::ParticleGradient, OHMMS_DIM> grad_grad;
  TinyVector<ParticleSet::ParticleLaplacian, OHMMS_DIM> lapl_grad;
  CHECK_THROWS_WITH(base_component.evalGradSource(electrons, *ions, 0, grad_grad, lapl_grad),
                    Catch::Matchers::ContainsSubstring("fixed nuclei"));
}

TEST_CASE("PsiFormer optimization rejects nonunit or unequal electron masses",
          "[wavefunction][psiformer][hardening]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell open_cell;

  for (const auto masses : {std::pair{1.0, 2.0}, std::pair{2.0, 2.0}})
  {
    DYNAMIC_SECTION("masses " << masses.first << " and " << masses.second)
    {
      ParticleSet electrons = makeMassTaggedElectrons(open_cell, masses.first, masses.second);
      WaveFunctionComponentBuilder::PSetMap particle_sets;
      auto ions = makeGuardTestIons(open_cell);
      particle_sets.emplace(ions->getName(), std::move(ions));
      PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, electrons, particle_sets);

      Libxml2Document optimized_document;
      REQUIRE(optimized_document.parseFromString(makePsiFormerXml(files, true)));
      CHECK_THROWS_WITH(builder.buildComponent(optimized_document.getRoot()),
                        Catch::Matchers::ContainsSubstring("unit electron masses"));

      // Fixed inference exposes only wavefunction observables, which do not
      // depend on the Hamiltonian mass convention and remains valid.
      Libxml2Document fixed_document;
      REQUIRE(fixed_document.parseFromString(makePsiFormerXml(files, false)));
      CHECK_NOTHROW(builder.buildComponent(fixed_document.getRoot()));
    }
  }
}

TEST_CASE("PsiFormer exact same-spin node produces a zero public ratio",
          "[wavefunction][psiformer][hardening][ratio]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell open_cell;
  ParticleSet electrons = makeMassTaggedElectrons(open_cell, 1.0, 1.0);

  // Use exactly representable coordinates so moving electron 0 onto the other
  // spin-up electron produces bit-identical orbital rows in every determinant.
  electrons.R[0] = ParticleSet::PosType{0.0, 0.0, 0.0};
  electrons.R[1] = ParticleSet::PosType{1.0, 1.0, 1.0};
  electrons.update();

  PsiFormerWF component("pf_node", files.parameters.string(), files.configuration.string());
  electrons.G = QMCTraits::ValueType(0);
  electrons.L = QMCTraits::ValueType(0);
  component.evaluateLog(electrons, electrons.G, electrons.L);

  const ParticleSet::SingleParticlePos displacement = electrons.R[1] - electrons.R[0];
  electrons.makeMove(0, displacement);
  const PsiFormerWF::PsiValue ratio = component.ratio(electrons, 0);
  CHECK(std::real(ratio) == 0.0);
  CHECK(std::imag(ratio) == 0.0);
  component.restore(0);
  electrons.rejectMove(0);
}

#ifdef QMC_COMPLEX
TEST_CASE("PsiFormer kinetic response rejects a genuinely complex total drift",
          "[wavefunction][psiformer][hardening][complex]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell open_cell;
  ParticleSet electrons = makeMassTaggedElectrons(open_cell, 1.0, 1.0);
  PsiFormerWF component(
      "pf_complex_drift", files.parameters.string(), files.configuration.string(), true, {0});

  OptVariables active;
  component.checkInVariablesExclusive(active);
  active.resetIndex();
  component.checkOutVariables(active);

  electrons.G = QMCTraits::ValueType(0);
  electrons.L = QMCTraits::ValueType(0);
  component.evaluateLog(electrons, electrons.G, electrons.L);
  electrons.G[0][0] += QMCTraits::ValueType(0.0, 0.125);

  Vector<QMCTraits::ValueType> score(active.size());
  Vector<QMCTraits::ValueType> kinetic_response(active.size());
  score            = QMCTraits::ValueType(0);
  kinetic_response = QMCTraits::ValueType(0);

  // Scores do not contract against the total spatial drift and remain valid
  // for the real ansatz embedded in a complex QMCPACK build.
  CHECK_NOTHROW(component.evaluateDerivativesWF(electrons, active, score));
  CHECK_THROWS_WITH(
      component.evaluateDerivatives(electrons, active, score, kinetic_response),
      Catch::Matchers::ContainsSubstring("real total wavefunction drift"));
}
#endif

} // namespace qmcplusplus
