//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_hamiltonian.cpp
 * @brief End-to-end all-electron Hamiltonian tests for the public PsiFormer interface.
 */

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Particle/ParticleSet.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "QMCHamiltonians/CoulombPotential.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "ResourceCollection.h"
#include "Utilities/RuntimeOptions.h"
#include "QMCWaveFunctions/tests/psiformer_test_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace qmcplusplus
{
namespace
{
using testing::psiformer::GeneratedFiles;
using testing::psiformer::Geometry;
using testing::psiformer::generateFiles;
using testing::psiformer::makeGeometry;
using ValueType = QMCTraits::ValueType;

/// Construct the two physical LiH nuclei represented by the generated fixture.
ParticleSet makeLiHIons(const SimulationCell& simulation_cell)
{
  const Geometry geometry = makeGeometry("lih");
  ParticleSet ions(simulation_cell);
  ions.setName("ion0");
  ions.create(std::vector<int>(geometry.charges.size(), 1));

  SpeciesSet& species = ions.getSpeciesSet();
  const int charge    = species.addAttribute("charge");
  const int atomic_no = species.addAttribute("atomic_number");
  for (std::size_t nucleus = 0; nucleus < geometry.charges.size(); ++nucleus)
  {
    const int species_index = species.addSpecies("ion_" + std::to_string(nucleus));
    species(charge, species_index)    = geometry.charges[nucleus];
    species(atomic_no, species_index) = geometry.charges[nucleus];
    for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
      ions.R[nucleus][dimension] = geometry.nuclei[OHMMS_DIM * nucleus + dimension];
  }
  ions.resetGroups();
  ions.update();
  return ions;
}

/// Construct four unit-mass electrons at the generated all-electron LiH sample.
ParticleSet makeLiHElectrons(const SimulationCell& simulation_cell)
{
  const Geometry geometry = makeGeometry("lih");
  ParticleSet electrons(simulation_cell);
  electrons.setName("e");
  electrons.create({2, 2});

  SpeciesSet& species   = electrons.getSpeciesSet();
  const int up          = species.addSpecies("u");
  const int down        = species.addSpecies("d");
  const int charge      = species.addAttribute("charge");
  const int mass        = species.addAttribute("mass");
  species(charge, up)   = -1.0;
  species(charge, down) = -1.0;
  species(mass, up)     = 1.0;
  species(mass, down)   = 1.0;
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
      electrons.R[electron][dimension] = geometry.electrons[OHMMS_DIM * electron + dimension];
  electrons.resetGroups();
  electrons.update();
  return electrons;
}

/// Build the physical open-boundary LiH Hamiltonian in production operator order.
std::unique_ptr<QMCHamiltonian> makeLiHHamiltonian(ParticleSet& ions, ParticleSet& electrons)
{
  auto hamiltonian = std::make_unique<QMCHamiltonian>("lih_psiformer");
  hamiltonian->addOperator(std::make_unique<BareKineticEnergy>(electrons), "Kinetic");
  hamiltonian->addOperator(std::make_unique<CoulombPotential>(electrons, true, false), "ElecElec");
  hamiltonian->addOperator(std::make_unique<CoulombPotential>(ions, electrons, true), "ElecIon");
  hamiltonian->addOperator(std::make_unique<CoulombPotential>(ions, false, false), "IonIon");
  hamiltonian->addObservables(electrons);

  // The Coulomb constructors add distance tables; update after the complete set
  // is present so every table is current for the first energy evaluation.
  electrons.update();
  return hamiltonian;
}

/// Register the selected PsiFormer variables through TrialWaveFunction's public path.
OptVariables registerParameters(TrialWaveFunction& wavefunction)
{
  OptVariables active;
  wavefunction.checkInVariables(active);
  active.resetIndex();
  wavefunction.checkOutVariables(active);
  return active;
}

/// Return the Euclidean separation between two Cartesian points.
template<class PositionA, class PositionB>
double distance(const PositionA& first, const PositionB& second)
{
  double squared_distance = 0.0;
  for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
  {
    const double displacement = first[dimension] - second[dimension];
    squared_distance += displacement * displacement;
  }
  return std::sqrt(squared_distance);
}

/// Independently sum kinetic plus all three straight-Coulomb contributions.
double referenceLocalEnergy(const ParticleSet& ions, const ParticleSet& electrons)
{
  double kinetic = 0.0;
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
  {
    double gradient_squared = 0.0;
    for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
    {
      const ValueType gradient = electrons.G[electron][dimension];
      gradient_squared += std::real(gradient) * std::real(gradient) +
          std::imag(gradient) * std::imag(gradient);
    }
    kinetic -= 0.5 * (std::real(electrons.L[electron]) + gradient_squared);
  }

  const Geometry geometry = makeGeometry("lih");
  double potential        = 0.0;
  for (int first = 0; first < electrons.getTotalNum(); ++first)
    for (int second = first + 1; second < electrons.getTotalNum(); ++second)
      potential += 1.0 / distance(electrons.R[first], electrons.R[second]);

  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int nucleus = 0; nucleus < ions.getTotalNum(); ++nucleus)
      potential -= geometry.charges[nucleus] / distance(electrons.R[electron], ions.R[nucleus]);

  for (int first = 0; first < ions.getTotalNum(); ++first)
    for (int second = first + 1; second < ions.getTotalNum(); ++second)
      potential += geometry.charges[first] * geometry.charges[second] /
          distance(ions.R[first], ions.R[second]);
  return kinetic + potential;
}

/// Evaluate the wavefunction VGL followed by the full physical Hamiltonian.
double evaluateEnergy(TrialWaveFunction& wavefunction,
                      QMCHamiltonian& hamiltonian,
                      ParticleSet& electrons)
{
  wavefunction.evaluateLog(electrons);
  return hamiltonian.evaluate(wavefunction, electrons);
}

} // namespace

TEST_CASE("PsiFormer all-electron LiH through QMCHamiltonian",
          "[hamiltonian][psiformer][all-electron]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet ions      = makeLiHIons(simulation_cell);
  ParticleSet electrons = makeLiHElectrons(simulation_cell);

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "psiformer_hamiltonian");
  auto psiformer = std::make_unique<PsiFormerWF>(
      "pf_hamiltonian", files.parameters.string(), files.configuration.string(), true,
      std::vector<std::size_t>{0, 127});
  psiformer->validateSystem(electrons, ions, "all_electron");
  wavefunction.addComponent(std::move(psiformer));
  OptVariables active = registerParameters(wavefunction);
  REQUIRE(active.size_of_active() == 2);

  std::unique_ptr<QMCHamiltonian> hamiltonian = makeLiHHamiltonian(ions, electrons);
  const double scalar_energy = evaluateEnergy(wavefunction, *hamiltonian, electrons);
  CHECK(scalar_energy == Catch::Approx(referenceLocalEnergy(ions, electrons)).epsilon(2e-10).margin(2e-10));
  CHECK(hamiltonian->getKineticEnergy() + hamiltonian->getLocalPotential() ==
        Catch::Approx(scalar_energy).epsilon(2e-12).margin(2e-12));

  // Exercise the production Hamiltonian derivative entry point.  The kinetic
  // operator obtains both quantities from PsiFormer; all local Coulomb terms
  // are parameter independent and leave the energy derivative unchanged.
  Vector<ValueType> score(active.size_of_active());
  Vector<ValueType> energy_derivative(active.size_of_active());
  score             = ValueType(0);
  energy_derivative = ValueType(0);
  wavefunction.evaluateLog(electrons);
  const double derivative_energy = hamiltonian->evaluateValueAndDerivatives(
      wavefunction, electrons, active, score, energy_derivative);
  CHECK(derivative_energy == Catch::Approx(scalar_energy).epsilon(2e-10).margin(2e-10));

  // Centered differences of the complete public wavefunction and Hamiltonian
  // validate one selected score and local-energy derivative end to end.
  const double original_parameter = std::real(active[0]);
  const double parameter_step     = 2e-5;
  active[0] = original_parameter + parameter_step;
  wavefunction.resetParameters(active);
  const double plus_log    = wavefunction.evaluateLog(electrons);
  const double plus_energy = hamiltonian->evaluate(wavefunction, electrons);

  active[0] = original_parameter - parameter_step;
  wavefunction.resetParameters(active);
  const double minus_log    = wavefunction.evaluateLog(electrons);
  const double minus_energy = hamiltonian->evaluate(wavefunction, electrons);

  active[0] = original_parameter;
  wavefunction.resetParameters(active);
  const double score_finite_difference = (plus_log - minus_log) / (2.0 * parameter_step);
  const double energy_finite_difference = (plus_energy - minus_energy) / (2.0 * parameter_step);
  CHECK(std::real(score[0]) ==
        Catch::Approx(score_finite_difference).epsilon(6e-5).margin(6e-5));
  CHECK(std::real(energy_derivative[0]) ==
        Catch::Approx(energy_finite_difference).epsilon(4e-4).margin(4e-4));

  // Create a genuinely different second walker and test the complete crowd
  // wavefunction/Hamiltonian resource path against independent scalar calls.
  ParticleSet electrons2(electrons);
  electrons2.R[0] += QMCTraits::PosType{0.11, -0.04, 0.03};
  electrons2.update();
  std::unique_ptr<TrialWaveFunction> wavefunction2 = wavefunction.makeClone(electrons2);
  std::unique_ptr<QMCHamiltonian> hamiltonian2 = hamiltonian->makeClone(electrons2, *wavefunction2);

  const double scalar_energy0 = evaluateEnergy(wavefunction, *hamiltonian, electrons);
  const double scalar_energy1 = evaluateEnergy(*wavefunction2, *hamiltonian2, electrons2);
  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons, electrons2});
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(
      wavefunction, {wavefunction, *wavefunction2});
  RefVectorWithLeader<QMCHamiltonian> hamiltonians(
      *hamiltonian, {*hamiltonian, *hamiltonian2});

  ResourceCollection particle_resources("psiformer_hamiltonian_particles");
  ResourceCollection wavefunction_resources("psiformer_hamiltonian_wavefunctions");
  ResourceCollection hamiltonian_resources("psiformer_hamiltonian_operators");
  electrons.createResource(particle_resources);
  wavefunction.createResource(wavefunction_resources);
  hamiltonian->createResource(hamiltonian_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
  ResourceCollectionTeamLock<TrialWaveFunction> wavefunction_lock(
      wavefunction_resources, wavefunctions);
  ResourceCollectionTeamLock<QMCHamiltonian> hamiltonian_lock(
      hamiltonian_resources, hamiltonians);

  ParticleSet::mw_update(particles);
  TrialWaveFunction::mw_evaluateLog(wavefunctions, particles);
  const std::vector<QMCHamiltonian::FullPrecRealType> batch_energies =
      QMCHamiltonian::mw_evaluate(hamiltonians, wavefunctions, particles);
  REQUIRE(batch_energies.size() == 2);
  CHECK(batch_energies[0] ==
        Catch::Approx(scalar_energy0).epsilon(2e-10).margin(2e-10));
  CHECK(batch_energies[1] ==
        Catch::Approx(scalar_energy1).epsilon(2e-10).margin(2e-10));

  RecordArray<ValueType> batch_scores(2, active.size_of_active());
  RecordArray<ValueType> batch_energy_derivatives(2, active.size_of_active());
  std::fill(batch_scores.begin(), batch_scores.end(), ValueType(0));
  std::fill(batch_energy_derivatives.begin(), batch_energy_derivatives.end(), ValueType(0));
  TrialWaveFunction::mw_evaluateLog(wavefunctions, particles);
  const std::vector<QMCHamiltonian::FullPrecRealType> derivative_batch_energies =
      QMCHamiltonian::mw_evaluateValueAndDerivatives(
          hamiltonians, wavefunctions, particles, active, batch_scores,
          batch_energy_derivatives);

  CHECK(derivative_batch_energies[0] ==
        Catch::Approx(scalar_energy0).epsilon(2e-10).margin(2e-10));
  CHECK(derivative_batch_energies[1] ==
        Catch::Approx(scalar_energy1).epsilon(2e-10).margin(2e-10));
  for (int parameter = 0; parameter < active.size_of_active(); ++parameter)
  {
    CHECK(std::abs(batch_scores[0][parameter] - score[parameter]) <=
          2e-9 * (1.0 + std::abs(score[parameter])));
    CHECK(std::abs(batch_energy_derivatives[0][parameter] - energy_derivative[parameter]) <=
          2e-8 * (1.0 + std::abs(energy_derivative[parameter])));
    CHECK(std::isfinite(std::real(batch_scores[1][parameter])));
    CHECK(std::isfinite(std::real(batch_energy_derivatives[1][parameter])));
  }
}

} // namespace qmcplusplus
