//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_hamiltonian.cpp
 * @brief End-to-end Hamiltonian tests for the public PsiFormer interface.
 */

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Particle/ParticleSet.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "QMCHamiltonians/CoulombPotential.h"
#include "QMCHamiltonians/ECPComponentBuilder.h"
#include "QMCHamiltonians/LocalECPotential.h"
#include "QMCHamiltonians/NonLocalECPotential.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "ResourceCollection.h"
#include "Utilities/FakeRandom.h"
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

/// Count deterministic RNG draws while retaining FakeRandom's fixed value.
class CountingFakeRandom : public FakeRandom<QMCTraits::FullPrecRealType>
{
public:
  QMCTraits::FullPrecRealType operator()() override
  {
    ++calls_;
    return FakeRandom<QMCTraits::FullPrecRealType>::operator()();
  }

  std::size_t calls() const noexcept { return calls_; }

private:
  std::size_t calls_ = 0;
};

/** Retain only selected canonical entries from one streamed parameter VJP. */
class SelectedParameterSink final : public wftrain::ParameterReductionSink
{
public:
  explicit SelectedParameterSink(std::array<std::size_t, 2> selected)
      : selected_(selected)
  {}

  const std::array<wftrain::DerivativeValue, 2>& result() const
  {
    if (state() != wftrain::DerivativeSinkState::COMPLETE)
      throw std::logic_error("selected parameter sink has no completed result");
    return result_;
  }

protected:
  void onBegin(const wftrain::DerivativeStreamDescriptor&,
               const wftrain::ParameterChunkPlan&,
               wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel>
                   channels) override
  {
    if (channels.size() != 1)
      throw std::invalid_argument("selected parameter sink requires one channel");
    result_.fill({});
  }

  void consume(std::size_t channel,
               const wftrain::ParameterChunkConstView& chunk) override
  {
    if (channel != 0)
      throw std::invalid_argument("selected parameter sink received a foreign channel");
    for (std::size_t probe = 0; probe < selected_.size(); ++probe)
      if (selected_[probe] >= chunk.descriptor().parameter_offset &&
          selected_[probe] < chunk.descriptor().parameter_offset +
                  chunk.descriptor().count)
        result_[probe] =
            chunk.values()[selected_[probe] - chunk.descriptor().parameter_offset];
  }

  void onAbort() noexcept override { result_.fill({}); }
  void onReset() noexcept override { result_.fill({}); }

private:
  std::array<std::size_t, 2> selected_;
  std::array<wftrain::DerivativeValue, 2> result_{};
};

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

/// Construct the two effective-charge-one nuclei in the pseudo-LiH fixture.
ParticleSet makePseudoLiHIons(const SimulationCell& simulation_cell)
{
  const Geometry geometry = makeGeometry("lih_pp");
  ParticleSet ions(simulation_cell);
  ions.setName("ion0");
  ions.create({2});

  SpeciesSet& species = ions.getSpeciesSet();
  const int sodium     = species.addSpecies("Na");
  const int charge     = species.addAttribute("charge");
  const int atomic_no  = species.addAttribute("atomic_number");
  species(charge, sodium)    = 1.0;
  species(atomic_no, sodium) = 11.0;
  for (int nucleus = 0; nucleus < ions.getTotalNum(); ++nucleus)
    for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
      ions.R[nucleus][dimension] = geometry.nuclei[OHMMS_DIM * nucleus + dimension];
  ions.resetGroups();
  ions.update();
  return ions;
}

/// Construct the two explicitly represented electrons in the pseudo-LiH fixture.
ParticleSet makePseudoLiHElectrons(const SimulationCell& simulation_cell)
{
  const Geometry geometry = makeGeometry("lih_pp");
  ParticleSet electrons(simulation_cell);
  electrons.setName("e");
  electrons.create({1, 1});

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

/// Build the open-boundary pseudo-LiH Hamiltonian in production operator order.
std::unique_ptr<QMCHamiltonian> makePseudoLiHHamiltonian(ParticleSet& ions, ParticleSet& electrons)
{
  ECPComponentBuilder ecp("psiformer_generated_ecp", OHMMS::Controller);
  REQUIRE(ecp.read_pp_file("Na.BFD.xml"));
  REQUIRE(ecp.pp_loc != nullptr);
  REQUIRE(ecp.pp_nonloc != nullptr);

  auto hamiltonian = std::make_unique<QMCHamiltonian>("lih_pp_psiformer");
  hamiltonian->addOperator(std::make_unique<BareKineticEnergy>(electrons), "Kinetic");
  hamiltonian->addOperator(std::make_unique<CoulombPotential>(electrons, true, false), "ElecElec");

  auto local_ecp = std::make_unique<LocalECPotential>(ions, electrons);
  local_ecp->add(0, std::move(ecp.pp_loc), ecp.Zeff);
  hamiltonian->addOperator(std::move(local_ecp), "LocalECP");

  auto nonlocal_ecp = std::make_unique<NonLocalECPotential>(
      ions, electrons, false /* enable_DLA */, true /* use_VP */);
  nonlocal_ecp->addComponent(0, std::move(ecp.pp_nonloc));
  hamiltonian->addOperator(std::move(nonlocal_ecp), "NonLocalECP");
  hamiltonian->addOperator(std::make_unique<CoulombPotential>(ions, false, false), "IonIon");
  hamiltonian->addObservables(electrons);

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

TEST_CASE("PsiFormer pseudopotential LiH through QMCHamiltonian",
          "[hamiltonian][psiformer][ecp][multiwalker]")
{
  using FullPrecReal = QMCTraits::FullPrecRealType;
  constexpr int walker_count = 2;

  GeneratedFiles files = generateFiles("lih_pp");
  const SimulationCell simulation_cell;
  ParticleSet ions      = makePseudoLiHIons(simulation_cell);
  ParticleSet electrons = makePseudoLiHElectrons(simulation_cell);

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "psiformer_pseudo_hamiltonian");
  auto psiformer = std::make_unique<PsiFormerWF>(
      "pf_pseudo_hamiltonian", files.parameters.string(), files.configuration.string(), true,
      std::vector<std::size_t>{0, 127});
  psiformer->validateSystem(electrons, ions, "pseudopotential");
  wavefunction.addComponent(std::move(psiformer));
  OptVariables active = registerParameters(wavefunction);
  REQUIRE(active.size_of_active() == 2);
  CHECK(active.name(0) == "pf_pseudo_hamiltonian_pf_0000000");
  CHECK(active.name(1) == "pf_pseudo_hamiltonian_pf_0000127");

  std::unique_ptr<QMCHamiltonian> hamiltonian = makePseudoLiHHamiltonian(ions, electrons);
  CountingFakeRandom grid_rng1;
  grid_rng1.set_value(0.371);
  hamiltonian->setRandomGenerator(&grid_rng1);

  ParticleSet electrons2(electrons);
  electrons2.R[0] += QMCTraits::PosType{0.11, -0.04, 0.03};
  electrons2.update();
  std::unique_ptr<TrialWaveFunction> wavefunction2 = wavefunction.makeClone(electrons2);
  std::unique_ptr<QMCHamiltonian> hamiltonian2 = hamiltonian->makeClone(electrons2, *wavefunction2);
  CountingFakeRandom grid_rng2;
  grid_rng2.set_value(0.619);
  hamiltonian2->setRandomGenerator(&grid_rng2);

  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons, electrons2});
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(
      wavefunction, {wavefunction, *wavefunction2});
  RefVectorWithLeader<QMCHamiltonian> hamiltonians(
      *hamiltonian, {*hamiltonian, *hamiltonian2});

  const int parameter_count = active.size_of_active();
  std::array<Vector<ValueType>, walker_count> scalar_scores{
      Vector<ValueType>(parameter_count), Vector<ValueType>(parameter_count)};
  std::array<Vector<ValueType>, walker_count> scalar_energy_derivatives{
      Vector<ValueType>(parameter_count), Vector<ValueType>(parameter_count)};
  std::array<FullPrecReal, walker_count> scalar_energies;
  for (int walker = 0; walker < walker_count; ++walker)
  {
    scalar_scores[walker]             = ValueType(0);
    scalar_energy_derivatives[walker] = ValueType(0);
    wavefunctions[walker].evaluateLog(particles[walker]);
    scalar_energies[walker] = hamiltonians[walker].evaluateValueAndDerivatives(
        wavefunctions[walker], particles[walker], active, scalar_scores[walker],
        scalar_energy_derivatives[walker]);
    CHECK(std::isfinite(scalar_energies[walker]));
  }

  ResourceCollection particle_resources("psiformer_pseudo_hamiltonian_particles");
  ResourceCollection wavefunction_resources("psiformer_pseudo_hamiltonian_wavefunctions");
  ResourceCollection hamiltonian_resources("psiformer_pseudo_hamiltonian_operators");
  electrons.createResource(particle_resources);
  wavefunction.createResource(wavefunction_resources);
  hamiltonian->createResource(hamiltonian_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
  ResourceCollectionTeamLock<TrialWaveFunction> wavefunction_lock(
      wavefunction_resources, wavefunctions);
  ResourceCollectionTeamLock<QMCHamiltonian> hamiltonian_lock(
      hamiltonian_resources, hamiltonians);

  // Produce full local energies and the isolated weighted ECP response from one
  // accepted grid.  The kinetic provider is composed later, without asking it
  // to reproduce the nonlocal term.
  auto& psiformer_leader =
      dynamic_cast<PsiFormerWF&>(*wavefunction.getOrbitals()[0]);
  auto& psiformer_clone =
      dynamic_cast<PsiFormerWF&>(*wavefunction2->getOrbitals()[0]);
  RefVectorWithLeader<WaveFunctionComponent> psiformer_components(
      psiformer_leader, {psiformer_leader, psiformer_clone});
  auto ecp_consumer = psiformer_leader.makeNonLocalECPDerivativeConsumer(
      psiformer_components, particles, 257);
  const auto parameter_snapshot = psiformer_leader.snapshotParameters();
  const std::vector<wftrain::DerivativeValue> objective_weights(walker_count, 1.0);
  const wftrain::CoefficientView objective_coefficients{
      psiformer_leader.parameterSchema().providerId(),
      psiformer_leader.parameterSchema().fingerprint(), parameter_snapshot.version,
      0, 0, {objective_weights.data(), objective_weights.size()}};
  const std::array<std::size_t, 2> canonical_parameters{0, 127};
  const wftrain::VJPCoefficientChannel ecp_channel{
      "weighted_local_energy",
      wftrain::DerivativeProduct::LOCAL_ENERGY_VJP, objective_coefficients,
      wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP)};
  SelectedParameterSink ecp_capture(canonical_parameters);

  ParticleSet::mw_update(particles);
  TrialWaveFunction::mw_evaluateLog(wavefunctions, particles);
  const std::vector<wftrain::DerivativeValue> short_objective_weights{1.0};
  const wftrain::CoefficientView short_objective_coefficients{
      psiformer_leader.parameterSchema().providerId(),
      psiformer_leader.parameterSchema().fingerprint(), parameter_snapshot.version,
      0, 0, {short_objective_weights.data(), short_objective_weights.size()}};
  const wftrain::VJPCoefficientChannel short_ecp_channel{
      "weighted_local_energy",
      wftrain::DerivativeProduct::LOCAL_ENERGY_VJP, short_objective_coefficients,
      wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP)};
  SelectedParameterSink rejected_capture(canonical_parameters);
  const std::size_t rng_calls_before_rejection = grid_rng1.calls() + grid_rng2.calls();
  CHECK_THROWS_AS(QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
                      hamiltonians, wavefunctions, particles, *ecp_consumer,
                      {&short_ecp_channel, 1}, rejected_capture),
                  std::invalid_argument);
  CHECK(grid_rng1.calls() + grid_rng2.calls() == rng_calls_before_rejection);
  CHECK(rejected_capture.state() == wftrain::DerivativeSinkState::IDLE);

  const std::size_t rng_calls_before_streaming = grid_rng1.calls() + grid_rng2.calls();
  const auto streaming_result =
      QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
          hamiltonians, wavefunctions, particles, *ecp_consumer,
          {&ecp_channel, 1}, ecp_capture);
  const std::size_t streaming_rng_calls =
      grid_rng1.calls() + grid_rng2.calls() - rng_calls_before_streaming;
  REQUIRE(streaming_result.local_energies.size() == walker_count);
  CHECK(streaming_rng_calls == 3 * walker_count);
  CHECK(streaming_result.grid_fingerprint != 0);
  CHECK(ecp_capture.state() == wftrain::DerivativeSinkState::COMPLETE);

  auto kinetic_derivative = psiformer_leader.makeStreamingDerivativeOperator(
      psiformer_components, particles, 0, 0, 257);
  const wftrain::VJPCoefficientChannel kinetic_channel{
      "weighted_local_energy", wftrain::DerivativeProduct::LOCAL_ENERGY_VJP,
      objective_coefficients,
      wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC)};
  SelectedParameterSink kinetic_capture(canonical_parameters);
  kinetic_derivative->applyVJPs({&kinetic_channel, 1},
                                wftrain::DerivativeAdjoint::TRANSPOSE,
                                kinetic_capture);

  ParticleSet::mw_update(particles);
  TrialWaveFunction::mw_evaluateLog(wavefunctions, particles);
  RecordArray<ValueType> batch_scores(walker_count, parameter_count);
  RecordArray<ValueType> batch_energy_derivatives(walker_count, parameter_count);
  std::fill(batch_scores.begin(), batch_scores.end(), ValueType(0));
  std::fill(batch_energy_derivatives.begin(), batch_energy_derivatives.end(), ValueType(0));
  const std::size_t rng_calls_before_legacy = grid_rng1.calls() + grid_rng2.calls();
  const std::vector<FullPrecReal> batch_energies =
      QMCHamiltonian::mw_evaluateValueAndDerivatives(
          hamiltonians, wavefunctions, particles, active, batch_scores,
          batch_energy_derivatives);
  const std::size_t legacy_rng_calls =
      grid_rng1.calls() + grid_rng2.calls() - rng_calls_before_legacy;
  CHECK(streaming_rng_calls == legacy_rng_calls);

  REQUIRE(batch_energies.size() == walker_count);
  for (int walker = 0; walker < walker_count; ++walker)
  {
    CHECK(streaming_result.local_energies[walker] ==
          Catch::Approx(batch_energies[walker]).epsilon(2e-10).margin(2e-10));
    CHECK(batch_energies[walker] ==
          Catch::Approx(scalar_energies[walker]).epsilon(2e-10).margin(2e-10));
    for (int parameter = 0; parameter < parameter_count; ++parameter)
    {
      CHECK(std::abs(batch_scores[walker][parameter] - scalar_scores[walker][parameter]) <=
            2e-9 * (1.0 + std::abs(scalar_scores[walker][parameter])));
      CHECK(std::abs(batch_energy_derivatives[walker][parameter] -
                     scalar_energy_derivatives[walker][parameter]) <=
            2e-8 * (1.0 + std::abs(scalar_energy_derivatives[walker][parameter])));
    }
  }

  for (int parameter = 0; parameter < parameter_count; ++parameter)
  {
    ValueType expected_energy_response = 0.0;
    for (int walker = 0; walker < walker_count; ++walker)
      expected_energy_response += batch_energy_derivatives[walker][parameter];
    const auto composed_response = kinetic_capture.result()[parameter] +
        ecp_capture.result()[parameter];
    CHECK(std::abs(composed_response - expected_energy_response) <=
          3e-8 * (1.0 + std::abs(expected_energy_response)));
  }

  // Differentiate the same complete public multiwalker Hamiltonian result for
  // both selected parameters. Constant per-walker generators make every
  // analytic and +/- evaluation use the same rotated nonlocal grids.
  const double parameter_step = 2e-5;
  auto evaluate_batch = [&](std::array<double, walker_count>& logs) {
    ParticleSet::mw_update(particles);
    TrialWaveFunction::mw_evaluateLog(wavefunctions, particles);
    for (int walker = 0; walker < walker_count; ++walker)
      logs[walker] = wavefunctions[walker].getLogPsi();
    return QMCHamiltonian::mw_evaluate(hamiltonians, wavefunctions, particles);
  };

  for (int parameter = 0; parameter < parameter_count; ++parameter)
  {
    const double original_parameter = std::real(active[parameter]);
    std::array<double, walker_count> plus_logs;
    std::array<double, walker_count> minus_logs;

    active[parameter] = original_parameter + parameter_step;
    wavefunction.resetParameters(active);
    const std::vector<FullPrecReal> plus_energies = evaluate_batch(plus_logs);

    active[parameter] = original_parameter - parameter_step;
    wavefunction.resetParameters(active);
    const std::vector<FullPrecReal> minus_energies = evaluate_batch(minus_logs);

    active[parameter] = original_parameter;
    wavefunction.resetParameters(active);
    for (int walker = 0; walker < walker_count; ++walker)
    {
      const double score_finite_difference =
          (plus_logs[walker] - minus_logs[walker]) / (2.0 * parameter_step);
      const double energy_finite_difference =
          (plus_energies[walker] - minus_energies[walker]) / (2.0 * parameter_step);
      CHECK(std::isfinite(score_finite_difference));
      CHECK(std::isfinite(energy_finite_difference));
      CHECK(std::real(batch_scores[walker][parameter]) ==
            Catch::Approx(score_finite_difference).epsilon(8e-5).margin(8e-5));
      CHECK(std::real(batch_energy_derivatives[walker][parameter]) ==
            Catch::Approx(energy_finite_difference).epsilon(8e-4).margin(8e-4));
#if defined(QMC_COMPLEX)
      // This open-boundary fixture has an entirely real model and Hamiltonian;
      // the complex build must not hide a shared spurious imaginary response.
      CHECK(std::imag(batch_scores[walker][parameter]) == Catch::Approx(0.0).margin(2e-10));
      CHECK(std::imag(batch_energy_derivatives[walker][parameter]) ==
            Catch::Approx(0.0).margin(2e-9));
#endif
    }
  }
}

} // namespace qmcplusplus
