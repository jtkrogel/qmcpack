//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include <hdf5.h>

#include "OhmmsData/Libxml2Doc.h"
#include "Particle/ParticleSet.h"
#include "QMCDrivers/MFRQMCMode.h"
#include "QMCHamiltonians/KineticJastrowResidual.h"
#include "QMCHamiltonians/MFRPotential.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCWaveFunctions/ConstantOrbital.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "Utilities/RuntimeOptions.h"
#include "Utilities/ResourceCollection.h"
#include "type_traits/template_types.hpp"

namespace qmcplusplus
{
namespace
{
using Catch::Approx;

void writeStringAttribute(hid_t object, const char* name, const std::string& value)
{
  const hid_t type = H5Tcopy(H5T_C_S1);
  REQUIRE(type >= 0);
  REQUIRE(H5Tset_size(type, value.size() + 1) >= 0);
  REQUIRE(H5Tset_strpad(type, H5T_STR_NULLTERM) >= 0);
  const hid_t space = H5Screate(H5S_SCALAR);
  REQUIRE(space >= 0);
  const hid_t attribute = H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(attribute >= 0);
  REQUIRE(H5Awrite(attribute, type, value.c_str()) >= 0);
  H5Aclose(attribute);
  H5Sclose(space);
  H5Tclose(type);
}

template<typename T>
void writeScalarAttribute(hid_t object, const char* name, hid_t type, const T& value)
{
  const hid_t space = H5Screate(H5S_SCALAR);
  REQUIRE(space >= 0);
  const hid_t attribute = H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(attribute >= 0);
  REQUIRE(H5Awrite(attribute, type, &value) >= 0);
  H5Aclose(attribute);
  H5Sclose(space);
}

void writeGridDimensions(hid_t object, const std::array<int, 3>& dimensions)
{
  const hsize_t array_size = 3;
  const hid_t type         = H5Tarray_create2(H5T_NATIVE_INT, 1, &array_size);
  REQUIRE(type >= 0);
  const hid_t space = H5Screate(H5S_SCALAR);
  REQUIRE(space >= 0);
  const hid_t attribute = H5Acreate2(object, "grid_dimensions", type, space, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(attribute >= 0);
  REQUIRE(H5Awrite(attribute, type, dimensions.data()) >= 0);
  H5Aclose(attribute);
  H5Sclose(space);
  H5Tclose(type);
}

void writeDoubleDataset(hid_t file, const char* name, const std::vector<hsize_t>& dimensions, const double* values)
{
  const hid_t space = H5Screate_simple(dimensions.size(), dimensions.data(), nullptr);
  REQUIRE(space >= 0);
  const hid_t dataset = H5Dcreate2(file, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  REQUIRE(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values) >= 0);
  H5Dclose(dataset);
  H5Sclose(space);
}

class TemporaryMFRFile
{
public:
  TemporaryMFRFile(std::vector<double> spin_values,
                   double total_energy_mf = 1.23456789012345,
                   std::string quantity  = "Vee_MF = v_H + v_xc",
                   bool varying_field    = false)
      : file_name_("mfr_unit_" + std::to_string(next_id_++) + ".h5")
  {
    const hid_t file = H5Fcreate(file_name_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(file >= 0);

    const int schema_version = 2;
    const int spin_channels  = spin_values.size();
    const std::array<int, 3> grid_dimensions{4, 4, 4};
    writeScalarAttribute(file, "schema_version", H5T_NATIVE_INT, schema_version);
    writeScalarAttribute(file, "spin_channels", H5T_NATIVE_INT, spin_channels);
    writeGridDimensions(file, grid_dimensions);
    writeStringAttribute(file, "quantity", quantity);
    writeStringAttribute(file, "units", "rydberg");
    writeStringAttribute(file, "functional", "PBE");
    writeStringAttribute(file, "grid_order", "Fortran: i fastest, then j, then k");
    writeStringAttribute(file, "periodic_grid_endpoint", "excluded");
    writeStringAttribute(file, "potential_includes_ions", "false");
    writeStringAttribute(file, "total_energy_mf_units", "rydberg");
    writeStringAttribute(file, "total_energy_mf_definition", "sum occupied KS eigenvalues + ion-ion energy");
    writeScalarAttribute(file, "total_energy_mf", H5T_NATIVE_DOUBLE, total_energy_mf);

    const hid_t mfr_group = H5Gcreate2(file, "/mfr", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(mfr_group >= 0);
    H5Gclose(mfr_group);
    const hid_t v_hxc_group = H5Gcreate2(file, "/mfr/v_hxc", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(v_hxc_group >= 0);
    H5Gclose(v_hxc_group);

    const std::array<double, 9> lattice{2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0};
    writeDoubleDataset(file, "/mfr/lattice_vectors", {3, 3}, lattice.data());

    for (int spin = 0; spin < spin_channels; ++spin)
    {
      std::vector<double> field(64, spin_values[spin]);
      if (varying_field)
        for (size_t index = 0; index < field.size(); ++index)
          field[index] += 0.125 * index;
      const std::string name = "/mfr/v_hxc/spin_" + std::to_string(spin + 1);
      writeDoubleDataset(file, name.c_str(), {4, 4, 4}, field.data());
    }
    H5Fclose(file);
  }

  ~TemporaryMFRFile() { std::remove(file_name_.c_str()); }

  const std::string& name() const { return file_name_; }

private:
  std::string file_name_;
  inline static int next_id_ = 0;
};

Lattice makePeriodicLattice()
{
  Lattice lattice;
  lattice.R         = {2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0};
  lattice.BoxBConds = true;
  lattice.reset();
  return lattice;
}

ParticleSet makeElectrons(const std::vector<int>& groups)
{
  static const SimulationCell simulation_cell(makePeriodicLattice());
  ParticleSet electrons(simulation_cell);
  electrons.setName("e");
  electrons.create(groups);
  SpeciesSet& species = electrons.getSpeciesSet();
  const int mass       = species.addAttribute("mass");
  for (int group = 0; group < groups.size(); ++group)
  {
    const int species_id       = species.addSpecies("s" + std::to_string(group));
    species(mass, species_id) = 1.0;
  }
  electrons.resetGroups();
  return electrons;
}

void loadMFRPotential(MFRPotential& potential, const std::string& file_name)
{
  Libxml2Document document;
  REQUIRE(document.parseFromString("<estimator type=\"mfr\" href=\"" + file_name + "\"/>"));
  REQUIRE(potential.put(document.getRoot()));
}

class PrescribedOrbital : public ConstantOrbital
{
public:
  PrescribedOrbital(bool fermionic, GradType gradient, ValueType laplacian)
      : fermionic_(fermionic), gradient_(gradient), laplacian_(laplacian)
  {}

  bool isFermionic() const override { return fermionic_; }
  std::string getClassName() const override { return "PrescribedOrbital"; }

  int log_calls = 0;
  int gl_calls = 0;

  LogValue evaluateLog(const ParticleSet& particles,
                       ParticleSet::ParticleGradient& gradients,
                       ParticleSet::ParticleLaplacian& laplacians) override
  {
    ++log_calls;
    accumulate(particles, gradients, laplacians);
    return 0.0;
  }

  LogValue evaluateGL(const ParticleSet& particles,
                      ParticleSet::ParticleGradient& gradients,
                      ParticleSet::ParticleLaplacian& laplacians,
                      bool fromscratch) override
  {
    ++gl_calls;
    accumulate(particles, gradients, laplacians);
    return 0.0;
  }

  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet&) const override
  {
    return std::make_unique<PrescribedOrbital>(fermionic_, gradient_, laplacian_);
  }

private:
  void accumulate(const ParticleSet& particles,
                  ParticleSet::ParticleGradient& gradients,
                  ParticleSet::ParticleLaplacian& laplacians)
  {
    for (int particle = 0; particle < particles.getTotalNum(); ++particle)
    {
      gradients[particle] += gradient_;
      laplacians[particle] += laplacian_;
    }
  }
  bool fermionic_;
  GradType gradient_;
  ValueType laplacian_;
};

class ConstantOperator : public OperatorBase
{
public:
  ConstantOperator(std::string class_name, Return_t value)
      : class_name_(std::move(class_name)), fixed_value_(value)
  {}

  std::string getClassName() const override { return class_name_; }
  Return_t evaluate(TrialWaveFunction&, ParticleSet&) override
  {
    value_ = fixed_value_;
    return value_;
  }
  bool put(xmlNodePtr) override { return true; }
  bool get(std::ostream&) const override { return true; }
  std::unique_ptr<OperatorBase> makeClone(ParticleSet&, TrialWaveFunction&) const override
  {
    return std::make_unique<ConstantOperator>(class_name_, fixed_value_);
  }

private:
  std::string class_name_;
  Return_t fixed_value_;
};

std::unique_ptr<ConstantOperator> constantOperator(const std::string& class_name, double value = 0.0)
{
  return std::make_unique<ConstantOperator>(class_name, value);
}

void addStandaloneSkeleton(QMCHamiltonian& hamiltonian)
{
  hamiltonian.addOperator(constantOperator("BareKineticEnergy"), "Kinetic");
  hamiltonian.addOperator(constantOperator("CoulombPotential"), "ElecElec");
  hamiltonian.addOperator(constantOperator("MFRPotential"), "MFRPotential", false);
  hamiltonian.addOperator(constantOperator("KineticJastrowResidual"), "KineticJastrowResidual", false);
}

void addStandaloneCoreActionSkeleton(QMCHamiltonian& hamiltonian)
{
  hamiltonian.addOperator(constantOperator("BareKineticEnergy"), "Kinetic");
  hamiltonian.addOperator(constantOperator("CoulombPotential"), "ElecElec");
  hamiltonian.addOperator(constantOperator("MFRCoreAction"), "MFRCoreAction", false);
  hamiltonian.addOperator(constantOperator("KineticJastrowResidual"), "KineticJastrowResidual", false);
}
} // namespace

TEST_CASE("MFRPotential deterministic interpolation", "[hamiltonian][mfr]")
{
  TemporaryMFRFile file({2.0});
  ParticleSet electrons = makeElectrons({2});
  electrons.R[0]         = {0.25, 0.50, 0.75};
  electrons.R[1]         = {2.25, -1.50, 4.75};

  MFRPotential potential(electrons);
  loadMFRPotential(potential, file.name());

  CHECK(potential.evaluate(electrons) == Approx(2.0));
  CHECK(potential.getTotalEnergyMF() == Approx(0.5 * 1.23456789012345).epsilon(1.e-13));

  ParticleSet clone_particles(electrons);
  auto clone       = potential.makeClone(clone_particles);
  auto* cloned_mfr = dynamic_cast<MFRPotential*>(clone.get());
  REQUIRE(cloned_mfr != nullptr);
  CHECK(cloned_mfr->evaluate(clone_particles) == Approx(2.0));
}

TEST_CASE("MFRPotential spin channels", "[hamiltonian][mfr]")
{
  TemporaryMFRFile file({2.0, 6.0});
  ParticleSet electrons = makeElectrons({1, 1});
  electrons.R[0]         = {0.1, 0.2, 0.3};
  electrons.R[1]         = {1.1, 1.2, 1.3};

  MFRPotential potential(electrons);
  loadMFRPotential(potential, file.name());
  CHECK(potential.evaluate(electrons) == Approx(4.0));
}

TEST_CASE("MFRPotential periodic wrapping", "[hamiltonian][mfr]")
{
  TemporaryMFRFile file({2.0}, 1.0, "Vee_MF = v_H + v_xc", true);
  ParticleSet electrons = makeElectrons({1});
  MFRPotential potential(electrons);
  loadMFRPotential(potential, file.name());

  electrons.R[0] = {0.31, 0.72, 1.43};
  const double reference = potential.evaluate(electrons);
  electrons.R[0] += QMCTraits::PosType{4.0, -2.0, 6.0};
  CHECK(potential.evaluate(electrons) == Approx(reference).margin(1.e-6));
}

TEST_CASE("MFR scalar retains precision just below a periodic endpoint", "[hamiltonian][mfr][mfr_boundary]")
{
  TemporaryMFRFile file({0.0});
  std::array<double, 64> values{};
  const std::array<double, 4> odd{0, 1, 0, -1};
  for (int z = 0; z < 4; ++z)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        values[(z * 4 + y) * 4 + x] = odd[x];
  const hid_t f = H5Fopen(file.name().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  REQUIRE(f >= 0);
  const hid_t d = H5Dopen2(f, "/mfr/v_hxc/spin_1", H5P_DEFAULT);
  REQUIRE(d >= 0);
  REQUIRE(H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
  H5Dclose(d);
  H5Fclose(f);
  auto electrons = makeElectrons({1});
  MFRPotential potential(electrons);
  loadMFRPotential(potential, file.name());
  electrons.R[0]        = {2e-9, .3, .7};
  const double positive = potential.evaluate(electrons);
  electrons.R[0]        = {-2e-9, .3, .7};
  const double negative = potential.evaluate(electrons);
  CHECK(positive > 2.5e-9);
  CHECK(positive < 3.5e-9); // Ry -> Ha is unchanged.
  CHECK(std::abs(positive + negative) < 1e-12);
}

TEST_CASE("MFRPotential rejects invalid metadata", "[hamiltonian][mfr]")
{
  TemporaryMFRFile file({2.0}, 1.0, "not Vee_MF");
  ParticleSet electrons = makeElectrons({1});
  MFRPotential potential(electrons);

  Libxml2Document document;
  REQUIRE(document.parseFromString("<estimator type=\"mfr\" href=\"" + file.name() + "\"/>"));
  CHECK_THROWS(potential.put(document.getRoot()));
}

TEST_CASE("MFRPotential rejects incompatible spin groups", "[hamiltonian][mfr]")
{
  TemporaryMFRFile file({2.0, 6.0});
  ParticleSet electrons = makeElectrons({2});
  MFRPotential potential(electrons);
  Libxml2Document document;
  REQUIRE(document.parseFromString("<estimator type=\"mfr\" href=\"" + file.name() + "\"/>"));
  CHECK_THROWS(potential.put(document.getRoot()));
}

TEST_CASE("MFRPotential requires an HDF5 path", "[hamiltonian][mfr]")
{
  ParticleSet electrons = makeElectrons({1});
  MFRPotential potential(electrons);
  Libxml2Document document;
  REQUIRE(document.parseFromString("<estimator type=\"mfr\"/>"));
  CHECK_THROWS(potential.put(document.getRoot()));
}

TEST_CASE("KineticJastrowResidual analytic values", "[hamiltonian][mfr]")
{
  ParticleSet electrons = makeElectrons({1});
  RuntimeOptions runtime_options;

  SECTION("determinant only")
  {
    TrialWaveFunction wavefunction(runtime_options);
    wavefunction.addComponent(
        std::make_unique<PrescribedOrbital>(true, QMCTraits::GradType{1.0, 0.0, 0.0}, 2.0));
    KineticJastrowResidual residual(electrons);
    CHECK(residual.evaluate(wavefunction, electrons) == Approx(0.0));
  }

  SECTION("Slater Jastrow cross term")
  {
    TrialWaveFunction wavefunction(runtime_options);
    wavefunction.addComponent(
        std::make_unique<PrescribedOrbital>(true, QMCTraits::GradType{1.0, 0.0, 0.0}, 2.0));
    wavefunction.addComponent(
        std::make_unique<PrescribedOrbital>(false, QMCTraits::GradType{3.0, 0.0, 0.0}, 4.0));
    KineticJastrowResidual residual(electrons);
    wavefunction.evaluateLog(electrons);
    CHECK(residual.evaluate(wavefunction, electrons) == Approx(-9.5));
  }

  SECTION("fermionic component required")
  {
    TrialWaveFunction wavefunction(runtime_options);
    wavefunction.addComponent(
        std::make_unique<PrescribedOrbital>(false, QMCTraits::GradType{0.0, 0.0, 0.0}, 0.0));
    KineticJastrowResidual residual(electrons);
    CHECK_THROWS(residual.evaluate(wavefunction, electrons));
  }
}

TEST_CASE("KineticJastrowResidual reuses current wavefunction derivatives", "[hamiltonian][mfr]")
{
  ParticleSet electrons = makeElectrons({2,1});
  auto& species = electrons.getSpeciesSet();
  species(species.addAttribute("mass"),1)=2.0;
  RuntimeOptions options;
  TrialWaveFunction psi(options);
  using Value = QMCTraits::ValueType;
  using Grad = QMCTraits::GradType;
  Grad gd{.7,-.4,.2}, gj1{.3,.1,-.2}, gj2{-.1,.2,.5};
  Value lj1=.6, lj2=-.3;
#if defined(QMC_COMPLEX)
  gd[0]+=Value(0,.23);
  gj1[1]+=Value(0,-.17);
  gj2[2]+=Value(0,.11);
  lj2+=Value(0,.13);
#endif
  auto fermion = std::make_unique<PrescribedOrbital>(true,gd,Value(1.2));
  auto first = std::make_unique<PrescribedOrbital>(false,gj1,lj1);
  auto second = std::make_unique<PrescribedOrbital>(false,gj2,lj2);
  auto* determinant=fermion.get();
  auto* j1=first.get();
  auto* j2=second.get();
  psi.addComponent(std::move(fermion));
  psi.addComponent(std::move(first));
  psi.addComponent(std::move(second));
  psi.evaluateLog(electrons);
  const auto saved_g=electrons.G;
  const auto saved_l=electrons.L;
  const auto gj=gj1+gj2;
  // Independent expansion of T[D exp(J)] - T[D], with unequal masses.
  Value correction=lj1+lj2;
  for(int axis=0;axis<3;++axis) correction+=gj[axis]*gj[axis]+Value(2)*gd[axis]*gj[axis];
  const double expected=-(2*.5+.25)*std::real(correction);
  const double tolerance=sizeof(QMCTraits::RealType)==sizeof(float)?2e-6:1e-12;
  KineticJastrowResidual residual(electrons);
  for(int repeat=0;repeat<3;++repeat)
    CHECK(residual.evaluate(psi,electrons)==Approx(expected).margin(tolerance));
  CHECK(determinant->log_calls==1);
  CHECK(determinant->gl_calls==0);
  CHECK(j1->log_calls==1);
  CHECK(j2->log_calls==1);
  CHECK(j1->gl_calls==3);
  CHECK(j2->gl_calls==3);
  for(int particle=0;particle<3;++particle)
  {
    CHECK(electrons.L[particle]==saved_l[particle]);
    for(int axis=0;axis<3;++axis)CHECK(electrons.G[particle][axis]==saved_g[particle][axis]);
  }
}

TEST_CASE("QMCHamiltonian MFR mode resolution", "[hamiltonian][mfr]")
{
  SECTION("no MFR")
  {
    QMCHamiltonian hamiltonian;
    hamiltonian.configureMFRQMCMode(MFRQMCModeRequest::AUTO);
    CHECK(hamiltonian.getMFRQMCMode() == QMCHamiltonian::MFRQMCMode::STANDARD);
    CHECK_THROWS(hamiltonian.configureMFRQMCMode(MFRQMCModeRequest::YES));
  }

  SECTION("automatic standalone")
  {
    QMCHamiltonian hamiltonian;
    addStandaloneSkeleton(hamiltonian);
    hamiltonian.configureMFRQMCMode(MFRQMCModeRequest::AUTO);
    CHECK(hamiltonian.getMFRQMCMode() == QMCHamiltonian::MFRQMCMode::MFR);
  }

  SECTION("direct action standalone and comparison")
  {
    QMCHamiltonian hamiltonian;
    addStandaloneCoreActionSkeleton(hamiltonian);
    hamiltonian.configureMFRQMCMode(MFRQMCModeRequest::AUTO);
    CHECK(hamiltonian.getMFRQMCMode() == QMCHamiltonian::MFRQMCMode::MFR);
    hamiltonian.configureMFRQMCMode(MFRQMCModeRequest::NO);
    CHECK(hamiltonian.getMFRQMCMode() == QMCHamiltonian::MFRQMCMode::STANDARD_WITH_MFR);
  }

  SECTION("scalar and action routes cannot be combined")
  {
    QMCHamiltonian hamiltonian;
    addStandaloneCoreActionSkeleton(hamiltonian);
    hamiltonian.addOperator(constantOperator("MFRPotential"), "MFRPotential", false);
    CHECK_THROWS(hamiltonian.configureMFRQMCMode(MFRQMCModeRequest::AUTO));
  }

  SECTION("explicit comparison")
  {
    QMCHamiltonian hamiltonian;
    addStandaloneSkeleton(hamiltonian);
    hamiltonian.configureMFRQMCMode(MFRQMCModeRequest::NO);
    CHECK(hamiltonian.getMFRQMCMode() == QMCHamiltonian::MFRQMCMode::STANDARD_WITH_MFR);
  }

  SECTION("ionic operator rejected")
  {
    QMCHamiltonian hamiltonian;
    addStandaloneSkeleton(hamiltonian);
    hamiltonian.addOperator(constantOperator("CoulombPBCAA"), "IonIon");
    CHECK_THROWS(hamiltonian.configureMFRQMCMode(MFRQMCModeRequest::YES));
  }

  SECTION("electron ion operator rejected")
  {
    QMCHamiltonian hamiltonian;
    addStandaloneSkeleton(hamiltonian);
    hamiltonian.addOperator(constantOperator("CoulombPBCAB"), "ElecIon");
    CHECK_THROWS(hamiltonian.configureMFRQMCMode(MFRQMCModeRequest::YES));
  }

  SECTION("pseudopotential rejected")
  {
    QMCHamiltonian hamiltonian;
    addStandaloneSkeleton(hamiltonian);
    hamiltonian.addOperator(constantOperator("NonLocalECPotential"), "PseudoPot");
    CHECK_THROWS(hamiltonian.configureMFRQMCMode(MFRQMCModeRequest::YES));
  }

  SECTION("MPC rejected")
  {
    QMCHamiltonian hamiltonian;
    addStandaloneSkeleton(hamiltonian);
    hamiltonian.addOperator(constantOperator("MPC"), "MPC");
    CHECK_THROWS(hamiltonian.configureMFRQMCMode(MFRQMCModeRequest::YES));
  }

  SECTION("missing electron electron operator rejected")
  {
    QMCHamiltonian hamiltonian;
    hamiltonian.addOperator(constantOperator("BareKineticEnergy"), "Kinetic");
    hamiltonian.addOperator(constantOperator("MFRPotential"), "MFRPotential", false);
    hamiltonian.addOperator(constantOperator("KineticJastrowResidual"), "KineticJastrowResidual", false);
    CHECK_THROWS(hamiltonian.configureMFRQMCMode(MFRQMCModeRequest::YES));
  }

  SECTION("missing residual rejected")
  {
    QMCHamiltonian hamiltonian;
    hamiltonian.addOperator(constantOperator("BareKineticEnergy"), "Kinetic");
    hamiltonian.addOperator(constantOperator("CoulombPotential"), "ElecElec");
    hamiltonian.addOperator(constantOperator("MFRPotential"), "MFRPotential", false);
    CHECK_THROWS(hamiltonian.configureMFRQMCMode(MFRQMCModeRequest::YES));
  }
}

TEST_CASE("QMCHamiltonian comparison mode preserves standard energy", "[hamiltonian][mfr]")
{
  ParticleSet electrons = makeElectrons({1});
  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options);

  QMCHamiltonian hamiltonian;
  hamiltonian.addOperator(constantOperator("BareKineticEnergy", 5.0), "Kinetic");
  hamiltonian.addOperator(constantOperator("CoulombPotential", 3.0), "ElecElec");
  hamiltonian.addOperator(constantOperator("MFRPotential", 100.0), "MFRPotential", false);
  hamiltonian.configureMFRQMCMode(MFRQMCModeRequest::NO);
  hamiltonian.addObservables(electrons);

  CHECK(hamiltonian.evaluate(wavefunction, electrons) == Approx(8.0));
  auto clone = hamiltonian.makeClone(electrons, wavefunction);
  CHECK(clone->getMFRQMCMode() == QMCHamiltonian::MFRQMCMode::STANDARD_WITH_MFR);
}

TEST_CASE("QMCHamiltonian standalone MFR arithmetic", "[hamiltonian][mfr]")
{
  TemporaryMFRFile file({8.0}, 20.0);
  ParticleSet electrons = makeElectrons({1});
  electrons.R[0]         = {0.2, 0.4, 0.6};

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options);
  wavefunction.addComponent(
      std::make_unique<PrescribedOrbital>(true, QMCTraits::GradType{0.0, 0.0, 0.0}, 0.0));
  wavefunction.addComponent(
      std::make_unique<PrescribedOrbital>(false, QMCTraits::GradType{0.0, 0.0, 0.0}, -4.0));

  auto mfr = std::make_unique<MFRPotential>(electrons);
  loadMFRPotential(*mfr, file.name());

  QMCHamiltonian hamiltonian;
  hamiltonian.addOperator(constantOperator("BareKineticEnergy", 99.0), "Kinetic");
  hamiltonian.addOperator(constantOperator("CoulombPotential", 3.0), "ElecElec");
  hamiltonian.addOperator(std::move(mfr), "MFRPotential", false);
  hamiltonian.addOperator(std::make_unique<KineticJastrowResidual>(electrons), "KineticJastrowResidual", false);
  hamiltonian.configureMFRQMCMode(MFRQMCModeRequest::YES);
  hamiltonian.addObservables(electrons);

  // 10 (TotalEnergyMF) + 2 (Jastrow residual) + 3 (ElecElec) - 4 (MFRPotential)
  CHECK(hamiltonian.evaluate(wavefunction, electrons) == Approx(11.0));
  CHECK(hamiltonian.getLocalEnergy() == Approx(11.0));

  std::vector<ParticleSet> particle_sets{electrons, electrons};
  UPtrVector<TrialWaveFunction> wavefunctions;
  UPtrVector<QMCHamiltonian> hamiltonians;
  for (ParticleSet& particles : particle_sets)
  {
    wavefunctions.emplace_back(wavefunction.makeClone(particles));
    hamiltonians.emplace_back(hamiltonian.makeClone(particles, *wavefunctions.back()));
    CHECK(hamiltonians.back()->getMFRQMCMode() == QMCHamiltonian::MFRQMCMode::MFR);
  }

  auto particle_refs = makeRefVector<ParticleSet>(particle_sets);
  RefVectorWithLeader<ParticleSet> particle_list{particle_refs[0], particle_refs};
  auto wavefunction_refs = convertUPtrToRefVector(wavefunctions);
  RefVectorWithLeader<TrialWaveFunction> wavefunction_list{wavefunction_refs[0], wavefunction_refs};
  auto hamiltonian_refs = convertUPtrToRefVector(hamiltonians);
  RefVectorWithLeader<QMCHamiltonian> hamiltonian_list{hamiltonian_refs[0], hamiltonian_refs};

  ResourceCollection resources("mfr_hamiltonian_test");
  hamiltonian_list.getLeader().createResource(resources);
  ResourceCollectionTeamLock<QMCHamiltonian> resource_lock(resources, hamiltonian_list);
  const auto batched_energies =
      QMCHamiltonian::mw_evaluate(hamiltonian_list, wavefunction_list, particle_list);
  REQUIRE(batched_energies.size() == 2);
  CHECK(batched_energies[0] == Approx(11.0));
  CHECK(batched_energies[1] == Approx(11.0));
}

} // namespace qmcplusplus

