//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "Message/Communicate.h"
#include "OhmmsData/Libxml2Doc.h"
#include "Particle/ParticleSet.h"
#include "QMCDrivers/WFTrain/TrainingWaveFunctionBinding.h"
#include "QMCWaveFunctions/Jastrow/BsplineFunctor.h"
#include "QMCWaveFunctions/Jastrow/TwoBodyJastrow.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "QMCHamiltonians/CoulombPotential.h"
#include "SimulationCell.h"
#include "io/hdf/hdf_archive.h"
#include <chrono>
#include <filesystem>
#include <limits>

namespace qmcplusplus::wftrain
{
TEST_CASE("Conventional binding activates actual Jastrows and rejects failed preparation collectively",
          "[drivers][training][conventional_binding][mpi]")
{
  Communicate& comm = *OHMMS::Controller;
  const SimulationCell cell;
  ParticleSet particles(cell);
  particles.setName("e");
  particles.create({1, 1});
  particles.R[0] = {0.2, -0.3, 0.4};
  particles.R[1] = {-0.5, 0.1, 0.7};
  auto& species = particles.getSpeciesSet();
  const int up = species.addSpecies("u"), down = species.addSpecies("d");
  const int mass = species.addAttribute("mass");
  species(mass, up) = species(mass, down) = 1.0;
  const int charge = species.addAttribute("charge");
  species(charge, up) = species(charge, down) = -1.0;
  particles.resetGroups();
  using Functor = BsplineFunctor<QMCTraits::RealType>;
  auto functor = std::make_unique<Functor>("binding_functor");
  functor->cutoff_radius = 4.0;
  functor->resize(4);
  for (int i = 0; i < 4; ++i)
  {
    functor->Parameters[i] = 0.04 * (i + 1);
    functor->myVars.insert("binding_" + std::to_string(i), functor->Parameters[i], true, optimize::LOGLINEAR_P);
  }
  functor->reset();
  auto j2 = std::make_unique<TwoBodyJastrow<Functor>>("binding_j2", particles, false);
  j2->addFunc(0, 1, std::move(functor));
  const RuntimeOptions runtime;
  TrialWaveFunction prototype(runtime, "binding_test");
  prototype.addComponent(std::move(j2));
  Libxml2Document definition;
  REQUIRE(definition.parseFromString("<wavefunction name='binding_test'><fixed coefficient='0.4'/></wavefunction>"));
  prototype.storeXMLNode(definition.getRoot());
  particles.update();
  ParticleSet clone_particles(particles);
  auto clone = prototype.makeClone(clone_particles);
  TrainingWaveFunctionBinding binding(prototype, particles, {256, 65536, 16}, comm);
  CHECK_THROWS(binding.modelIdentity());
  QMCHamiltonian sampling_hamiltonian;
  sampling_hamiltonian.addOperator(std::make_unique<BareKineticEnergy>(particles), "Kinetic");
  sampling_hamiltonian.addOperator(std::make_unique<CoulombPotential>(particles, true, false), "ElecElec");
  binding.setHamiltonian(sampling_hamiltonian);
  binding.attachLive(*clone, clone_particles);
  const auto old = binding.snapshotParameters();
  auto candidate = old;
  candidate.values[0] += 0.07;
  const auto original_positions = particles.R;
  const auto original_clone_positions = clone_particles.R;
  const auto* original = prototype.getOrbitals()[0].get();
  const auto* original_clone = clone->getOrbitals()[0].get();
  auto invalid = candidate;
  if (comm.rank() == 0) invalid.values[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(binding.publishParameters(invalid, old.version));
  CHECK(binding.snapshotParameters().values == old.values);
  CHECK(prototype.getOrbitals()[0].get() == original);
  CHECK(clone->getOrbitals()[0].get() == original_clone);
  CHECK(binding.publishParameters(candidate, old.version) == old.version + 1);
  CHECK(prototype.getOrbitals()[0].get() != original);
  CHECK(clone->getOrbitals()[0].get() != original_clone);
  optimize::VariableSet actual, actual_clone;
  prototype.checkInVariables(actual);
  clone->checkInVariables(actual_clone);
  CHECK(std::real(actual[0]) == Catch::Approx(candidate.values[0]));
  CHECK(std::real(actual_clone[0]) == Catch::Approx(candidate.values[0]));
  for (int electron = 0; electron < 2; ++electron)
    for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
    {
      CHECK(particles.R[electron][dimension] == original_positions[electron][dimension]);
      CHECK(clone_particles.R[electron][dimension] == original_clone_positions[electron][dimension]);
    }
  auto next = binding.snapshotParameters();
  next.values[1] -= 0.03;
  CHECK(binding.publishParameters(next, next.version) == next.version + 1);
  CHECK_THROWS(binding.publishParameters(candidate, old.version));
  const auto file = std::filesystem::temp_directory_path() /
      ("conventional_binding_" + std::to_string(comm.rank()) + "_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".h5");
  struct Cleanup
  {
    std::filesystem::path path;
    ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); }
  } cleanup{file};
  binding.exportModel(file);
  CHECK(binding.readModel(file).values == binding.snapshotParameters().values);
  {
    hdf_archive output;
    REQUIRE(output.open(file, H5F_ACC_RDWR));
    output.push("conventional_wavefunction", false);
    std::string xml;
    output.read(xml, "wavefunction_xml");
    xml[0] = '!';
    output.write(xml, "wavefunction_xml");
  }
  CHECK_THROWS(binding.readModel(file));
  Libxml2Document external_definition;
  REQUIRE(external_definition.parseFromString("<wavefunction><sposet_builder href='unqualified.h5'/></wavefunction>"));
  auto external = prototype.makeClone(particles);
  external->storeXMLNode(external_definition.getRoot());
  TrainingWaveFunctionBinding external_binding(*external, particles, {256, 65536, 16}, comm);
  external_binding.setHamiltonian(sampling_hamiltonian);
  CHECK_THROWS(external_binding.modelIdentity());
  Libxml2Document global_spo_definition;
  REQUIRE(global_spo_definition.parseFromString("<wavefunction><determinant sposet='global_spo'/></wavefunction>"));
  auto global_spo = prototype.makeClone(particles);
  global_spo->storeXMLNode(global_spo_definition.getRoot());
  TrainingWaveFunctionBinding global_spo_binding(*global_spo, particles, {256, 65536, 16}, comm);
  global_spo_binding.setHamiltonian(sampling_hamiltonian);
  CHECK_THROWS(global_spo_binding.modelIdentity());

  ParticleSet ions(cell);
  ions.setName("ion0");
  ions.create({1});
  ions.R[0] = {0.0, 0.0, 1.5};
  auto& ion_species = ions.getSpeciesSet();
  ion_species.addSpecies("H");
  ion_species(ion_species.addAttribute("charge"), 0) = 1.0;
  ions.resetGroups();
  QMCHamiltonian hamiltonian;
  hamiltonian.addOperator(std::make_unique<BareKineticEnergy>(particles), "Kinetic");
  hamiltonian.addOperator(std::make_unique<CoulombPotential>(ions, particles, true), "IonElec");
  binding.setHamiltonian(hamiltonian);
  const auto physics = binding.modelIdentity();
  particles.R[0][0] += 0.2;
  binding.setHamiltonian(hamiltonian);
  CHECK(binding.modelIdentity() == physics); // Electron coordinates are sampler state.
  ions.R[0][0] += 0.2;
  binding.setHamiltonian(hamiltonian);
  CHECK(binding.modelIdentity() != physics);
}
}
