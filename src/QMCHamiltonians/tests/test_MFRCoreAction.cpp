//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "hdf/hdf_archive.h"
#include "ResourceCollection.h"
#include "QMCWaveFunctions/BsplineFactory/BsplineSet.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "Particle/ParticleSetPool.h"
#include "QMCWaveFunctions/BsplineFactory/EinsplineSetBuilder.h"
#include "QMCWaveFunctions/Fermion/DiracDeterminant.h"
#include "QMCWaveFunctions/Fermion/DiracDeterminantBatched.h"
#include "QMCWaveFunctions/Fermion/SlaterDet.h"
#include "QMCWaveFunctions/ConstantOrbital.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCHamiltonians/KineticJastrowResidual.h"

#include "Utilities/for_testing/Catch2Approx.h"

#include <array>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <hdf5.h>

#include "OhmmsData/Libxml2Doc.h"
#include "Particle/ParticleSet.h"
#include "QMCHamiltonians/MFRCoreAction.h"
#if !defined(QMC_COMPLEX)
#include "QMCWaveFunctions/BsplineFactory/SplineR2R.h"
#include "spline2/MultiBsplineOffloadMapperBase.hpp"
#endif
#include "Utilities/RuntimeOptions.h"
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

template<typename T>
void writeArrayAttribute(hid_t object, const char* name, hid_t base_type, const std::array<T, 3>& values)
{
  const hsize_t length = 3;
  const hid_t type     = H5Tarray_create2(base_type, 1, &length);
  REQUIRE(type >= 0);
  const hid_t space = H5Screate(H5S_SCALAR);
  REQUIRE(space >= 0);
  const hid_t attribute = H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(attribute >= 0);
  REQUIRE(H5Awrite(attribute, type, values.data()) >= 0);
  H5Aclose(attribute);
  H5Sclose(space);
  H5Tclose(type);
}

void writeDoubleDataset(hid_t file,
                        const char* name,
                        const std::vector<hsize_t>& dimensions,
                        const std::vector<double>& values)
{
  const hid_t space = H5Screate_simple(dimensions.size(), dimensions.data(), nullptr);
  REQUIRE(space >= 0);
  const hid_t dataset = H5Dcreate2(file, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  REQUIRE(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
  H5Dclose(dataset);
  H5Sclose(space);
}

class TemporaryCoreActionFile
{
public:
  explicit TemporaryCoreActionFile(int schema_version              = 1,
                                   double real_value               = 2.5,
                                   double imag_value               = -0.75,
                                   int complete_flag               = 1,
                                   std::array<double, 3> reduced_k = {0.0, 0.0, 0.0},
                                   double eigenvalue               = -1.25)
      : file_name_("mfr_core_action_unit_" + std::to_string(next_id_++) + ".h5")
  {
    const hid_t file = H5Fcreate(file_name_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(file >= 0);
    writeStringAttribute(file, "schema_family", "mfr_core_action");
    writeScalarAttribute(file, "schema_version", H5T_NATIVE_INT, schema_version);
    const int complete          = complete_flag;
    const int spin_channels     = 1;
    const int number_of_kpoints = 1;
    const int number_of_states  = 1;
    const std::array<int, 3> grid_dimensions{4, 4, 4};
    writeScalarAttribute(file, "complete", H5T_NATIVE_INT, complete);
    writeStringAttribute(file, "representation", "periodic_bloch_core_action_grid");
    writeStringAttribute(file, "energy_units", "hartree");
    writeStringAttribute(file, "length_units", "bohr");
    writeStringAttribute(file, "orbital_normalization", "sum_G c_G exp(+i G.r); no volume factor");
    writeStringAttribute(file, "bloch_phase", "exp(+2 pi i reduced_k.reduced_r)");
    writeStringAttribute(file, "grid_order", "HDF5: z,y,x; x fastest");
    writeStringAttribute(file, "periodic_grid_endpoint", "excluded");
    writeArrayAttribute(file, "grid_dimensions", H5T_NATIVE_INT, grid_dimensions);
    writeScalarAttribute(file, "spin_channels", H5T_NATIVE_INT, spin_channels);
    writeScalarAttribute(file, "number_of_kpoints", H5T_NATIVE_INT, number_of_kpoints);
    writeScalarAttribute(file, "number_of_states", H5T_NATIVE_INT, number_of_states);
    const double ion_ion_energy = -0.125;
    writeScalarAttribute(file, "ion_ion_energy", H5T_NATIVE_DOUBLE, ion_ion_energy);
    writeDoubleDataset(file, "/lattice_vectors", {3, 3}, {2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0});

    const hid_t kgroup = H5Gcreate2(file, "/kpoint_0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(kgroup >= 0);
    writeArrayAttribute(kgroup, "reduced_k", H5T_NATIVE_DOUBLE, reduced_k);
    H5Gclose(kgroup);
    const hid_t spin_group = H5Gcreate2(file, "/kpoint_0/spin_0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(spin_group >= 0);
    H5Gclose(spin_group);
    const hid_t state_group = H5Gcreate2(file, "/kpoint_0/spin_0/state_0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(state_group >= 0);
    writeScalarAttribute(state_group, "eigenvalue", H5T_NATIVE_DOUBLE, eigenvalue);
    H5Gclose(state_group);
    writeDoubleDataset(file, "/kpoint_0/spin_0/state_0/action_real", {4, 4, 4}, std::vector<double>(64, real_value));
    writeDoubleDataset(file, "/kpoint_0/spin_0/state_0/action_imag", {4, 4, 4}, std::vector<double>(64, imag_value));
    H5Fclose(file);
  }

  ~TemporaryCoreActionFile() { std::remove(file_name_.c_str()); }
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

ParticleSet makeElectrons()
{
  static const SimulationCell simulation_cell(makePeriodicLattice());
  ParticleSet electrons(simulation_cell);
  electrons.setName("e");
  electrons.create({1});
  SpeciesSet& species       = electrons.getSpeciesSet();
  const int mass            = species.addAttribute("mass");
  const int species_id      = species.addSpecies("u");
  species(mass, species_id) = 1.0;
  electrons.resetGroups();
  return electrons;
}

void loadAction(MFRCoreAction& action, const std::string& file_name)
{
  Libxml2Document document;
  REQUIRE(document.parseFromString("<extpot type=\"mfr_core_action\" href=\"" + file_name + "\"/>"));
  REQUIRE(action.put(document.getRoot()));
}
} // namespace

TEST_CASE("MFRCoreAction reads periodic complex action grids and clones shared data", "[hamiltonian][mfr]")
{
  TemporaryCoreActionFile file;
  ParticleSet electrons = makeElectrons();
  MFRCoreAction action(electrons);
  loadAction(action, file.name());
  action.loadActionState(0, 0, 0);

  const auto value = action.evaluatePeriodicAction(0, 0, 0, MFRCoreAction::PosType(0.25, 0.5, 0.75));
  CHECK(value.real() == Approx(2.5).margin(1e-12));
  CHECK(value.imag() == Approx(-0.75).margin(1e-12));
  const auto wrapped = action.evaluatePeriodicAction(0, 0, 0, MFRCoreAction::PosType(2.25, -1.5, 4.75));
  CHECK(wrapped.real() == Approx(2.5).margin(1e-12));
  CHECK(wrapped.imag() == Approx(-0.75).margin(1e-12));
  CHECK(action.getIonIonEnergyMF() == Approx(-0.125));

  auto clone          = action.makeClone(electrons);
  auto* cloned_action = dynamic_cast<MFRCoreAction*>(clone.get());
  REQUIRE(cloned_action != nullptr);
  const auto cloned_value = cloned_action->evaluatePeriodicAction(0, 0, 0, MFRCoreAction::PosType(0.1, 0.2, 0.3));
  CHECK(cloned_value.real() == Approx(2.5).margin(1e-12));
  CHECK(cloned_value.imag() == Approx(-0.75).margin(1e-12));

  // MFR action state is external and immutable.  Reconstructing the operator
  // from the same file models the Hamiltonian side of a QMCPACK checkpoint
  // restart, where the action cache must be populated again from its source.
  MFRCoreAction restarted_action(electrons);
  loadAction(restarted_action, file.name());
  restarted_action.loadActionState(0, 0, 0);
  const auto restarted_value =
      restarted_action.evaluatePeriodicAction(0, 0, 0, MFRCoreAction::PosType(0.25, 0.5, 0.75));
  CHECK(restarted_value.real() == Approx(value.real()).margin(1e-12));
  CHECK(restarted_value.imag() == Approx(value.imag()).margin(1e-12));
  TemporaryCoreActionFile replacement(1, -1.2, .35);
  loadAction(action, replacement.name());
  action.loadActionState(0, 0, 0);
  const auto replaced = action.evaluatePeriodicAction(0, 0, 0, MFRCoreAction::PosType(.1, .2, .3));
  CHECK(replaced.real() == Approx(-1.2).margin(1e-12));
  CHECK(replaced.imag() == Approx(.35).margin(1e-12));
  const auto preserved = cloned_action->evaluatePeriodicAction(0, 0, 0, MFRCoreAction::PosType(.1, .2, .3));
  CHECK(preserved.real() == Approx(2.5).margin(1e-12));
  CHECK(preserved.imag() == Approx(-.75).margin(1e-12));
  Libxml2Document missing_href;
  REQUIRE(missing_href.parseFromString("<extpot type='mfr_core_action'/>"));
  CHECK_THROWS_WITH(action.put(missing_href.getRoot()), Catch::Matchers::ContainsSubstring("requires an href"));
}

#if !defined(QMC_COMPLEX)
TEST_CASE("MFRCoreAction preserves real antiperiodic Bloch signs across cell crossings", "[hamiltonian][mfr]")
{
  TemporaryCoreActionFile file(1, 2.5, 0.0, 1, {0.5, 0.0, 0.0});
  ParticleSet electrons = makeElectrons();
  MFRCoreAction action(electrons);
  loadAction(action, file.name());
  action.loadActionState(0, 0, 0);

  // Only the source map is needed: action splines are owned by MFRCoreAction.
  SplineR2R<double> orbitals("antiperiodic", 1, electrons.getLattice(), nullptr);
  BsplineSet::MFRCoreActionOrbitalSource source;
  source.band = 0;
  source.part = BsplineSet::MFRCoreActionPart::REAL;
  orbitals.getMFRCoreActionSources().push_back(source);
  const auto base          = action.evaluateOrbitalAction(orbitals, 0, {0.0, 0.0, 0.0});
  const auto next_cell     = action.evaluateOrbitalAction(orbitals, 0, {2.0, 0.0, 0.0});
  const auto previous_cell = action.evaluateOrbitalAction(orbitals, 0, {-2.0, 0.0, 0.0});
  CHECK(base.real() == Approx(2.5));
  CHECK(next_cell.real() == Approx(-2.5));
  CHECK(previous_cell.real() == Approx(-2.5));
}
#endif

TEST_CASE("MFR action retains precision just below a periodic endpoint", "[hamiltonian][mfr][mfr_boundary]")
{
  TemporaryCoreActionFile file(1, 0.0, 0.0);
  std::array<double, 64> values{};
  const std::array<double, 4> odd{0, 1, 0, -1};
  for (int z = 0; z < 4; ++z)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        values[(z * 4 + y) * 4 + x] = odd[x];
  const hid_t f = H5Fopen(file.name().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  REQUIRE(f >= 0);
  const hid_t d = H5Dopen2(f, "/kpoint_0/spin_0/state_0/action_real", H5P_DEFAULT);
  REQUIRE(d >= 0);
  REQUIRE(H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
  H5Dclose(d);
  H5Fclose(f);
  auto electrons = makeElectrons();
  MFRCoreAction action(electrons);
  loadAction(action, file.name());
  action.loadActionState(0, 0, 0);
  const double positive = action.evaluatePeriodicAction(0, 0, 0, MFRCoreAction::PosType(2e-9, .3, .7)).real();
  const double negative = action.evaluatePeriodicAction(0, 0, 0, MFRCoreAction::PosType(-2e-9, .3, .7)).real();
  // Periodic cubic interpolation preserves odd symmetry. A float fractional
  // coordinate rounds the negative point to 1, an invalid einspline endpoint.
  CHECK(positive > 5e-9);
  CHECK(positive < 7e-9);
  CHECK(std::abs(positive + negative) < 1e-12);
}

TEST_CASE("MFRCoreAction rejects unsupported schema versions", "[hamiltonian][mfr]")
{
  TemporaryCoreActionFile file(99);
  ParticleSet electrons = makeElectrons();
  MFRCoreAction action(electrons);
  Libxml2Document document;
  REQUIRE(document.parseFromString("<extpot type=\"mfr_core_action\" href=\"" + file.name() + "\"/>"));
  CHECK_THROWS(action.put(document.getRoot()));
}

TEST_CASE("MFRCoreAction rejects non-finite state values in optimized builds", "[hamiltonian][mfr]")
{
  ParticleSet electrons = makeElectrons();
  MFRCoreAction action(electrons);
  SECTION("NaN action grid")
  {
    TemporaryCoreActionFile file(1, std::numeric_limits<double>::quiet_NaN());
    loadAction(action, file.name());
    CHECK_THROWS(action.loadActionState(0, 0, 0));
  }
  SECTION("Infinite action grid")
  {
    TemporaryCoreActionFile file(1, std::numeric_limits<double>::infinity());
    loadAction(action, file.name());
    CHECK_THROWS(action.loadActionState(0, 0, 0));
  }
  SECTION("NaN eigenvalue")
  {
    TemporaryCoreActionFile file(1, 2.5, 0.0, 1, {0.0, 0.0, 0.0}, std::numeric_limits<double>::quiet_NaN());
    loadAction(action, file.name());
    CHECK_THROWS(action.loadActionState(0, 0, 0));
  }
}

TEST_CASE("MFRCoreAction rejects malformed attribute storage", "[hamiltonian][mfr]")
{
  TemporaryCoreActionFile file;
  const hid_t h5file = H5Fopen(file.name().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  REQUIRE(h5file >= 0);
  SECTION("scalar version stored as a vector")
  {
    REQUIRE(H5Adelete(h5file, "schema_version") >= 0);
    const hsize_t length  = 2;
    const hid_t space     = H5Screate_simple(1, &length, nullptr);
    const hid_t attribute = H5Acreate2(h5file, "schema_version", H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
    const int versions[2] = {1, 1};
    REQUIRE(H5Awrite(attribute, H5T_NATIVE_INT, versions) >= 0);
    H5Aclose(attribute);
    H5Sclose(space);
  }
  SECTION("schema family stored as a variable-length string")
  {
    REQUIRE(H5Adelete(h5file, "schema_family") >= 0);
    const hid_t type = H5Tcopy(H5T_C_S1);
    REQUIRE(H5Tset_size(type, H5T_VARIABLE) >= 0);
    const hid_t space     = H5Screate(H5S_SCALAR);
    const hid_t attribute = H5Acreate2(h5file, "schema_family", type, space, H5P_DEFAULT, H5P_DEFAULT);
    const char* family    = "mfr_core_action";
    REQUIRE(H5Awrite(attribute, type, &family) >= 0);
    H5Aclose(attribute);
    H5Sclose(space);
    H5Tclose(type);
  }
  H5Fclose(h5file);
  ParticleSet electrons = makeElectrons();
  MFRCoreAction action(electrons);
  Libxml2Document document;
  REQUIRE(document.parseFromString("<extpot type=\"mfr_core_action\" href=\"" + file.name() + "\"/>"));
  CHECK_THROWS(action.put(document.getRoot()));
}

TEST_CASE("MFRCoreAction rejects incomplete exports", "[hamiltonian][mfr]")
{
  TemporaryCoreActionFile file(1, 2.5, -0.75, 0);
  ParticleSet electrons = makeElectrons();
  MFRCoreAction action(electrons);
  Libxml2Document document;
  REQUIRE(document.parseFromString("<extpot type=\"mfr_core_action\" href=\"" + file.name() + "\"/>"));
  CHECK_THROWS(action.put(document.getRoot()));
}
} // namespace qmcplusplus

// Independent continuum oracle plus the actual ES-HDF -> spline -> determinant
// -> MFR reader/contraction path.  No physical pseudopotential fixture is used.
namespace qmcplusplus
{
namespace
{
using Z  = std::complex<double>;
using R3 = QMCTraits::QTFull::PosType;

struct AnalyticCoreModel
{
  std::array<std::array<int, 3>, 7> g{{{0, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
  std::array<std::array<Z, 7>, 2> c{{{Z(1), Z(.075), Z(.075), Z(0, -.06), Z(0, .06), Z(0), Z(0)},
                                     {Z(.4), Z(0, -.25), Z(0, .25), Z(.15), Z(.15), Z(0), Z(0)}}};
  std::array<Z, 7> beta{{Z(.3), Z(.1), Z(.1), Z(0, -.05), Z(0, .05), Z(0), Z(0)}};
  R3 k{0.0};
  Lattice lattice;
  explicit AnalyticCoreModel(bool boundary, bool skew)
  {
    lattice.R         = {2 * M_PI, skew ? .4 * M_PI : 0, 0, 0, 2 * M_PI, skew ? .2 * M_PI : 0, 0, 0, 2 * M_PI};
    lattice.BoxBConds = true;
    lattice.reset();
#if defined(QMC_COMPLEX)
    k = boundary ? R3(0.0) : R3(.19, -.11, .07);
    c[0][1] += Z(.03, .07);
    c[1][3] += Z(-.04, .08);
#else
    if (boundary)
    {
      k = {.5, 0, 0};
      for (auto& row : c)
        for (auto& value : row)
          value = 0;
      for (auto& value : beta)
        value = 0;
      c[0][0] = c[0][2] = 1 / std::sqrt(2.0);
      c[1][0]           = Z(0, -1 / std::sqrt(2.0));
      c[1][2]           = Z(0, 1 / std::sqrt(2.0));
      beta[0] = beta[2] = 1 / std::sqrt(2.0);
    }
#endif
    double norm0 = 0;
    for (const auto v : c[0])
      norm0 += std::norm(v);
    for (auto& v : c[0])
      v /= std::sqrt(norm0);
    Z overlap{};
    for (int i = 0; i < 7; ++i)
      overlap += std::conj(c[0][i]) * c[1][i];
    double norm1 = 0;
    for (int i = 0; i < 7; ++i)
    {
      c[1][i] -= overlap * c[0][i];
      norm1 += std::norm(c[1][i]);
    }
    for (auto& v : c[1])
      v /= std::sqrt(norm1);
  }
  struct Field
  {
    Z value{}, kinetic{}, nonlocal{};
    std::array<Z, 3> grad{};
  };
  Field field(int band, const R3& r) const
  {
    Field out;
    Z projection{};
    for (int ig = 0; ig < 7; ++ig)
      projection += std::conj(beta[ig]) * c[band][ig];
    for (int ig = 0; ig < 7; ++ig)
    {
      const R3 wave      = lattice.k_cart(R3(g[ig][0] + k[0], g[ig][1] + k[1], g[ig][2] + k[2]));
      const double angle = dot(wave, r), norm = dot(wave, wave);
      const Z phase = std::exp(Z(0, angle));
      out.value += c[band][ig] * phase;
      out.kinetic += .5 * norm * c[band][ig] * phase;
      out.nonlocal += .7 * beta[ig] * projection * phase;
      for (int d = 0; d < 3; ++d)
        out.grad[d] += Z(0, wave[d]) * c[band][ig] * phase;
    }
    return out;
  }
  double local(const R3& r) const { return .21 + .13 * std::cos(dot(lattice.k_cart(R3(1, -1, 0)), r)); }
  Z action(int band, const R3& r) const
  {
    const auto f = field(band, r);
    return f.kinetic + local(r) * f.value + f.nonlocal;
  }
};

class AnalyticCoreFiles
{
public:
  TemporaryCoreActionFile core;
  std::string orbital_file;
  AnalyticCoreModel model;
  explicit AnalyticCoreFiles(int mesh, bool boundary, bool skew) : model(boundary, skew)
  {
    orbital_file  = core.name() + ".orbitals.h5";
    const hid_t f = H5Fopen(core.name().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    REQUIRE(f >= 0);
    auto replace_attribute = [&](const char* name, hid_t type, const void* value) {
      const hid_t a = H5Aopen(f, name, H5P_DEFAULT);
      REQUIRE(a >= 0);
      REQUIRE(H5Awrite(a, type, value) >= 0);
      H5Aclose(a);
    };
    const int bands = 2;
    replace_attribute("number_of_states", H5T_NATIVE_INT, &bands);
    const std::array<int, 3> dims{mesh, mesh, mesh};
    const hsize_t three = 3;
    const hid_t at      = H5Tarray_create2(H5T_NATIVE_INT, 1, &three);
    replace_attribute("grid_dimensions", at, dims.data());
    H5Tclose(at);
    H5Ldelete(f, "/lattice_vectors", H5P_DEFAULT);
    writeDoubleDataset(f, "/lattice_vectors", {3, 3},
                       std::vector<double>(model.lattice.R.data(), model.lattice.R.data() + 9));
    const hid_t kg = H5Gopen2(f, "/kpoint_0", H5P_DEFAULT);
    const hid_t ka = H5Aopen(kg, "reduced_k", H5P_DEFAULT);
    const hid_t kt = H5Aget_type(ka);
    REQUIRE(H5Awrite(ka, kt, model.k.data()) >= 0);
    H5Tclose(kt);
    H5Aclose(ka);
    H5Gclose(kg);
    // The source-orbital identity datasets deliberately use explicit complex
    // Fourier coefficients, independently of the sampled core-action grids.
    const hid_t space = H5Screate_simple(2, std::array<hsize_t, 2>{7, 3}.data(), nullptr);
    const hid_t gd =
        H5Dcreate2(f, "/kpoint_0/spin_0/gvectors", H5T_STD_I32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(H5Dwrite(gd, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, model.g.data()) >= 0);
    H5Dclose(gd);
    H5Sclose(space);
    for (int band = 0; band < 2; ++band)
    {
      const std::string path = "/kpoint_0/spin_0/state_" + std::to_string(band);
      if (band == 0)
      {
        H5Ldelete(f, (path + "/action_real").c_str(), H5P_DEFAULT);
        H5Ldelete(f, (path + "/action_imag").c_str(), H5P_DEFAULT);
      }
      else
      {
        const hid_t sg = H5Gcreate2(f, path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        writeScalarAttribute(sg, "eigenvalue", H5T_NATIVE_DOUBLE, -.75);
        H5Gclose(sg);
      }
      std::vector<double> re(mesh * mesh * mesh), im(re.size()), cr(7), ci(7);
      for (int z = 0; z < mesh; ++z)
        for (int y = 0; y < mesh; ++y)
          for (int x = 0; x < mesh; ++x)
          {
            const R3 r = model.lattice.toCart(R3(double(x) / mesh, double(y) / mesh, double(z) / mesh));
            // Store the periodic part; the production reader restores Bloch phase.
            const Z a   = model.action(band, r) * std::exp(Z(0, -dot(model.lattice.k_cart(model.k), r)));
            const int i = (z * mesh + y) * mesh + x;
            re[i]       = a.real();
            im[i]       = a.imag();
          }
      for (int ig = 0; ig < 7; ++ig)
      {
        cr[ig] = model.c[band][ig].real();
        ci[ig] = model.c[band][ig].imag();
      }
      writeDoubleDataset(f, (path + "/action_real").c_str(), {hsize_t(mesh), hsize_t(mesh), hsize_t(mesh)}, re);
      writeDoubleDataset(f, (path + "/action_imag").c_str(), {hsize_t(mesh), hsize_t(mesh), hsize_t(mesh)}, im);
      writeDoubleDataset(f, (path + "/orbital_g_real").c_str(), {7}, cr);
      writeDoubleDataset(f, (path + "/orbital_g_imag").c_str(), {7}, ci);
    }
    writeDoubleDataset(f, "/ion_positions", {1, 3}, {0, 0, 0});
    const int ion_species   = 1;
    const hsize_t ion_count = 1;
    const hid_t ion_type    = H5Tarray_create2(H5T_NATIVE_INT, 1, &ion_count);
    const hid_t ion_space   = H5Screate(H5S_SCALAR);
    const hid_t ion_attr    = H5Acreate2(f, "ion_species", ion_type, ion_space, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(H5Awrite(ion_attr, ion_type, &ion_species) >= 0);
    H5Aclose(ion_attr);
    H5Sclose(ion_space);
    H5Tclose(ion_type);
    const hid_t sg = H5Gcreate2(f, "/species_0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    writeStringAttribute(sg, "element", "He");
    writeScalarAttribute(sg, "valence_charge", H5T_NATIVE_DOUBLE, 2.0);
    H5Gclose(sg);
    H5Fclose(f);

    hdf_archive es;
    REQUIRE(es.create(orbital_file));
    TinyVector<int, 3> version(2, 0, 0);
    es.write(version, "version");
    es.write(std::string("ES-HDF"), "format");
    es.push("supercell", true);
    Tensor<double, 3> lattice = model.lattice.R;
    es.write(lattice, "primitive_vectors");
    es.pop();
    es.push("atoms", true);
    int one = 1;
    es.write(one, "number_of_atoms");
    es.write(one, "number_of_species");
    Vector<int> ids(1);
    ids[0] = 0;
    es.write(ids, "species_ids");
    Vector<R3> positions(1);
    positions[0] = R3(0.0);
    es.write(positions, "positions");
    es.push("species_0", true);
    int atomic_number = 2;
    es.write(atomic_number, "atomic_number");
    es.write(std::string("He\0", 3), "name");
    es.write(2.0, "valence_charge");
    es.pop();
    es.pop();
    es.push("electrons", true);
    es.write(one, "number_of_spins");
    es.write(one, "number_of_kpoints");
    TinyVector<int, 2> ne(2, 1);
    es.write(ne, "number_of_electrons");
    es.push("kpoint_0", true);
    es.write(model.k, "reduced_k");
    es.write(1.0, "weight");
    Vector<TinyVector<int, 3>> gs(7);
    for (int i = 0; i < 7; ++i)
      for (int d = 0; d < 3; ++d)
        gs[i][d] = model.g[i][d];
    es.write(gs, "gvectors");
    int ng = 7;
    es.write(ng, "number_of_gvectors");
    es.push("spin_0", true);
    int ns = 2;
    es.write(ns, "number_of_states");
    Vector<double> eigenvalues(2);
    eigenvalues[0] = -1.25;
    eigenvalues[1] = -.75;
    es.write(eigenvalues, "eigenvalues");
    for (int b = 0; b < 2; ++b)
    {
      es.push("state_" + std::to_string(b), true);
      Vector<Z> coefficients(7);
      for (int i = 0; i < 7; ++i)
        coefficients[i] = model.c[b][i];
      es.write(coefficients, "psi_g");
      es.pop();
    }
    es.close();
  }
  ~AnalyticCoreFiles() { std::remove(orbital_file.c_str()); }
};

// Smooth, genuinely two-particle periodic log(J), with closed-form derivatives.
class PeriodicPairJastrow : public ConstantOrbital
{
public:
  explicit PeriodicPairJastrow(double a, const Lattice& lattice) : a_(a), lattice_(lattice) {}
  LogValue evaluateLog(const ParticleSet& p,
                       ParticleSet::ParticleGradient& gradient,
                       ParticleSet::ParticleLaplacian& laplacian) override
  {
    double j = 0;
    for (int i = 0; i < p.getTotalNum(); ++i)
      for (int l = 0; l < i; ++l)
        for (int d = 0; d < 2; ++d)
        {
          R3 unit(0.0);
          unit[d]        = 1;
          const R3 wave  = lattice_.k_cart(unit);
          const double a = a_ * (d == 0 ? 1 : .5), delta = dot(wave, R3(p.R[i]) - R3(p.R[l]));
          j += a * std::cos(delta);
          for (int axis = 0; axis < 3; ++axis)
          {
            gradient[i][axis] -= a * std::sin(delta) * wave[axis];
            gradient[l][axis] += a * std::sin(delta) * wave[axis];
          }
          laplacian[i] -= a * std::cos(delta) * dot(wave, wave);
          laplacian[l] -= a * std::cos(delta) * dot(wave, wave);
        }
    return j;
  }
  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet&) const override
  {
    return std::make_unique<PeriodicPairJastrow>(a_, lattice_);
  }

private:
  double a_;
  Lattice lattice_;
};
class IdentityElectronEnergy : public OperatorDependsOnlyOnParticleSet
{
public:
  std::string getClassName() const override { return "CoulombPotential"; }
  Return_t evaluate(ParticleSet&) override { return value_ = .37; }
  bool put(xmlNodePtr) override { return true; }
  bool get(std::ostream&) const override { return true; }
  std::unique_ptr<OperatorBase> makeClone(ParticleSet&) const override
  {
    return std::make_unique<IdentityElectronEnergy>();
  }
};
} // namespace

TEST_CASE("MFR core action integrated periodic determinant and DLA identity", "[hamiltonian][mfr][mfr_identity]")
{
  const bool batched     = GENERATE(false, true);
  const bool boundary    = GENERATE(false, true);
  const bool skew        = GENERATE(false, true);
  const double jastrow_a = GENERATE(0.0, .08);
  // Cubic spline values converge as h^4. Derivatives used in the Jastrow
  // residual converge separately; the strict contraction test uses the
  // production SPO values and an independent 2x2 closed-form inverse.
  double previous_error = 1;
  for (int mesh : {16, 32})
  {
    AnalyticCoreFiles files(mesh, boundary, skew);
    const Lattice& lattice = files.model.lattice;
    ParticleSetPool pool(OHMMS::Controller);
    pool.createSimulationCellByLattice(lattice);
    auto ions = std::make_unique<ParticleSet>(pool.getSimulationCell());
    ions->setName("ion");
    ions->create({1});
    ions->R[0]        = R3(0.0);
    auto& ion_species = ions->getSpeciesSet();
    ion_species.addSpecies("He");
    const int ion_charge          = ion_species.addAttribute("charge");
    ion_species(ion_charge, 0)    = 2;
    const int atomic_number       = ion_species.addAttribute("atomicnumber");
    ion_species(atomic_number, 0) = 2;
    ions->resetGroups();
    ions->update();
    pool.addParticleSet(std::move(ions));
    ParticleSet electrons(pool.getSimulationCell());
    electrons.setName("e");
    electrons.create({2, 1});
    electrons.R[0] = {.21, .37, .18};
    electrons.R[1] = {1.3, .7, .5};
    electrons.R[2] = {2.1, 1.2, .8};
    auto& species  = electrons.getSpeciesSet();
    const int mass = species.addAttribute("mass");
    species.addSpecies("u");
    species.addSpecies("d");
    species(mass, 0) = species(mass, 1) = 1.0;
    electrons.resetGroups();
    electrons.update();
    const auto original_positions = electrons.R;
    Libxml2Document doc;
    REQUIRE(doc.parseFromString("<sposet_collection type='einspline' href='" + files.orbital_file +
                                "' tilematrix='1 0 0 0 1 0 0 0 1' twistnum='0' source='ion' meshfactor='" +
                                std::to_string(mesh / 4) +
                                "' precision='double' save_coefs='no'><sposet name='mfr_analytic' size='2'/><sposet "
                                "name='mfr_down' size='1'/></sposet_collection>"));
    EinsplineSetBuilder builder(electrons, pool.getPool(), OHMMS::Controller, doc.getRoot());
    auto spo = builder.createSPOSetFromXML(xmlFirstElementChild(doc.getRoot()));
    REQUIRE(spo);
    auto down_spo = builder.createSPOSetFromXML(xmlNextElementSibling(xmlFirstElementChild(doc.getRoot())));
    REQUIRE(down_spo);
    builder.H5File.close(); // SPO construction is complete; negative tests reopen this file for writing.
    SPOSet& orbitals = *spo;
    std::vector<std::unique_ptr<DiracDeterminantBase>> determinants;
    for (const auto bounds : {std::pair<int, int>{0, 2}, {2, 3}})
      if (batched)
        determinants.push_back(
            std::make_unique<DiracDeterminantBatched<PlatformKind::OMPTARGET, QMCTraits::ValueType,
                                                     QMCTraits::QTFull::ValueType>>(bounds.first == 0 ? *spo
                                                                                                      : *down_spo,
                                                                                    bounds.first, bounds.second,
                                                                                    bounds.first == 0 ? 2 : 1));
      else
        determinants.push_back(std::make_unique<DiracDeterminant<>>(bounds.first == 0 ? *spo : *down_spo, bounds.first,
                                                                    bounds.second, bounds.first == 0 ? 2 : 1));
    std::vector<std::unique_ptr<SPOSet>> sets;
    sets.push_back(std::move(spo));
    sets.push_back(std::move(down_spo));
    RuntimeOptions options;
    TrialWaveFunction psi(options);
    psi.addComponent(std::make_unique<SlaterDet>(electrons, std::move(sets), std::move(determinants)));
    if (jastrow_a != 0)
      psi.addComponent(std::make_unique<PeriodicPairJastrow>(jastrow_a, lattice));
    psi.evaluateLog(electrons);
    MFRCoreAction core(electrons, &psi);
    loadAction(core, files.core.name());
    const auto actual = core.evaluateTrace(psi, electrons);
    if (mesh == 16 && !batched && !boundary && !skew && jastrow_a == 0)
    {
      auto must_reject = [&](const std::string& diagnostic) {
        MFRCoreAction invalid(electrons, &psi);
        Libxml2Document input;
        REQUIRE(input.parseFromString("<extpot type='mfr_core_action' href='" + files.core.name() + "'/>"));
        CHECK_THROWS_WITH(invalid.put(input.getRoot()), Catch::Matchers::ContainsSubstring(diagnostic));
      };
      auto* source = pool.getParticleSet("ion");
      REQUIRE(source);
      source->R[0][0] = .02;
      must_reject("QMC source particle set");
      source->R[0][0]   = 0;
      auto& sp          = source->getSpeciesSet();
      sp(ion_charge, 0) = 3;
      must_reject("source species/charge mismatch");
      sp(ion_charge, 0)    = 2;
      sp(atomic_number, 0) = 10;
      must_reject("source species/charge mismatch");
      sp(atomic_number, 0) = 2;
      const hid_t cf       = H5Fopen(files.core.name().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
      REQUIRE(cf >= 0);
      auto set_int = [&](const char* name, int value) {
        const hid_t a = H5Aopen(cf, name, H5P_DEFAULT);
        REQUIRE(a >= 0);
        REQUIRE(H5Awrite(a, H5T_NATIVE_INT, &value) >= 0);
        H5Aclose(a);
      };
      const hid_t positions = H5Dopen2(cf, "/ion_positions", H5P_DEFAULT);
      double shifted[3]     = {.02, 0, 0};
      REQUIRE(H5Dwrite(positions, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, shifted) >= 0);
      H5Fflush(cf, H5F_SCOPE_GLOBAL);
      must_reject("paired ES-HDF geometry");
      shifted[0] = 0;
      REQUIRE(H5Dwrite(positions, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, shifted) >= 0);
      H5Dclose(positions);
      set_int("number_of_states", 1);
      H5Fflush(cf, H5F_SCOPE_GLOBAL);
      must_reject("outside the exported state range");
      set_int("number_of_states", 2);
      set_int("spin_channels", 2);
      H5Fflush(cf, H5F_SCOPE_GLOBAL);
      must_reject("selected spin does not match");
      set_int("spin_channels", 1);
      H5Fclose(cf);
      const hid_t ef = H5Fopen(files.orbital_file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
      REQUIRE(ef >= 0);
      auto replace_name = [&](hsize_t count, bool variable) {
        REQUIRE(H5Ldelete(ef, "/atoms/species_0/name", H5P_DEFAULT) >= 0);
        const hid_t type = H5Tcopy(H5T_C_S1);
        REQUIRE(H5Tset_size(type, variable ? H5T_VARIABLE : 3) >= 0);
        const hid_t space = H5Screate_simple(1, &count, nullptr);
        const hid_t dataset =
            H5Dcreate2(ef, "/atoms/species_0/name", type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        const char value[6] = {'H', 'e', 0, 'H', 'e', 0};
        const char* pointer = value;
        REQUIRE(H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                         variable ? static_cast<const void*>(&pointer) : static_cast<const void*>(value)) >= 0);
        H5Dclose(dataset);
        H5Sclose(space);
        H5Tclose(type);
        H5Fflush(ef, H5F_SCOPE_GLOBAL);
      };
      replace_name(2, false);
      must_reject("bounded fixed-string");
      replace_name(1, true);
      must_reject("bounded fixed-string");
      replace_name(1, false);
      REQUIRE(H5Ldelete(ef, "/atoms/species_0/valence_charge", H5P_DEFAULT) >= 0);
      writeDoubleDataset(ef, "/atoms/species_0/valence_charge", {2}, {2, 2});
      H5Fflush(ef, H5F_SCOPE_GLOBAL);
      must_reject("finite numeric scalar");
      REQUIRE(H5Ldelete(ef, "/atoms/species_0/valence_charge", H5P_DEFAULT) >= 0);
      writeDoubleDataset(ef, "/atoms/species_0/valence_charge", {1}, {2});
      H5Fclose(ef);
      MFRCoreAction restored(electrons, &psi);
      REQUIRE_NOTHROW(loadAction(restored, files.core.name()));
      CHECK(std::abs(Z(restored.evaluateTrace(psi, electrons)) - Z(actual)) < 1e-6);
    }
    CHECK(core.getTotalEnergyMF() == Approx(-3.375)); // three occupied levels + one ionic constant
    SPOSet::ValueMatrix m(3, 2), lap(3, 2);
    SPOSet::GradMatrix grad(3, 2);
    orbitals.evaluate_notranspose(electrons, 0, 3, m, grad, lap);
    const auto det = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
    REQUIRE(std::abs(det) > .1);
    // Explicit inverse-transpose: no production inversion/contraction helper.
    Z weights[3][2] = {{Z(m(1, 1) / det), Z(-m(1, 0) / det)},
                       {Z(-m(0, 1) / det), Z(m(0, 0) / det)},
                       {Z(1.0) / Z(m(2, 0)), Z(0)}};
    double norm_m = 0, norm_inverse = 0;
    for (int i = 0; i < 2; ++i)
      for (int b = 0; b < 2; ++b)
      {
        norm_m += std::norm(Z(m(i, b)));
        norm_inverse += std::norm(weights[i][b]);
      }
    REQUIRE(std::sqrt(norm_m * norm_inverse) < 100);
    Z matched{}, continuum{};
    std::array<std::array<Z, 3>, 3> dgrad{};
    auto* bsp = dynamic_cast<BsplineSet*>(&orbitals);
    REQUIRE(bsp);
    for (int i = 0; i < 3; ++i)
      for (int b = 0; b < (i == 2 ? 1 : 2); ++b)
      {
        matched += weights[i][b] * core.evaluateOrbitalAction(*bsp, b, electrons.R[i]);
        for (int d = 0; d < 3; ++d)
          dgrad[i][d] += weights[i][b] * Z(grad(i, b)[d]);
      }
    const double algebra_tol = sizeof(QMCTraits::RealType) == sizeof(float) ? 2e-5 : 1e-10;
    CHECK(std::abs(Z(actual) - matched) < algebra_tol);
    // A matching fixed occupied-space rotation of M and G must leave the
    // two-electron block trace unchanged. The down block occupies one state.
    const double cs = std::cos(.37), sn = std::sin(.37);
#if defined(QMC_COMPLEX)
    const Z rotation[2][2] = {{cs, Z(0, sn)}, {Z(0, sn), cs}};
#else
    const Z rotation[2][2] = {{cs, -sn}, {sn, cs}};
#endif
    Z rotated_m[2][2]{}, rotated_g[2][2]{};
    for (int row = 0; row < 2; ++row)
      for (int col = 0; col < 2; ++col)
        for (int old = 0; old < 2; ++old)
        {
          rotated_m[row][col] += Z(m(row, old)) * rotation[old][col];
          rotated_g[row][col] += core.evaluateOrbitalAction(*bsp, old, electrons.R[row]) * rotation[old][col];
        }
    const Z rotated_det   = rotated_m[0][0] * rotated_m[1][1] - rotated_m[0][1] * rotated_m[1][0];
    const Z rotated_trace = (rotated_m[1][1] * rotated_g[0][0] - rotated_m[1][0] * rotated_g[0][1] -
                             rotated_m[0][1] * rotated_g[1][0] + rotated_m[0][0] * rotated_g[1][1]) /
        rotated_det;
    const Z down_trace = core.evaluateOrbitalAction(*bsp, 0, electrons.R[2]) / Z(m(2, 0));
    CHECK(std::abs(rotated_trace + down_trace - Z(actual)) < algebra_tol);

    // Independent continuum determinant weights and bare kinetic/local/nonlocal
    // action. The phase/gauge cancels columnwise; no QMC source map is used.
    const auto a = files.model.field(0, electrons.R[0]).value, b = files.model.field(1, electrons.R[0]).value;
    const auto c = files.model.field(0, electrons.R[1]).value, d = files.model.field(1, electrons.R[1]).value;
    const Z exactdet     = a * d - b * c;
    const Z exactw[3][2] = {{d / exactdet, -c / exactdet},
                            {-b / exactdet, a / exactdet},
                            {1.0 / files.model.field(0, electrons.R[2]).value, Z(0)}};
    for (int i = 0; i < 3; ++i)
      for (int band = 0; band < (i == 2 ? 1 : 2); ++band)
        continuum += exactw[i][band] * files.model.action(band, electrons.R[i]);
    const double error = std::abs(Z(actual) - continuum);
    INFO("mesh=" << mesh << " batched=" << batched << " J=" << jastrow_a << " boundary=" << boundary << " skew=" << skew
                 << " continuum error=" << error);
    CHECK(error < .003 * std::pow(16.0 / mesh, 4) + algebra_tol);
    if (mesh == 32)
      CHECK(error < previous_error / 5 + algebra_tol);
    previous_error = error;
    // Analytic pair-J derivatives, evaluated independently from the component.
    double gj[3][3]{}, lj[3]{};
    for (int i = 0; i < 3; ++i)
      for (int n = 0; n < 3; ++n)
        if (i != n)
          for (int dim = 0; dim < 2; ++dim)
          {
            R3 unit(0.0);
            unit[dim]          = 1;
            const R3 wave      = lattice.k_cart(unit);
            const double aa    = jastrow_a * (dim == 0 ? 1 : .5),
                         delta = dot(wave, R3(electrons.R[i]) - R3(electrons.R[n]));
            for (int axis = 0; axis < 3; ++axis)
              gj[i][axis] -= aa * std::sin(delta) * wave[axis];
            lj[i] -= aa * std::cos(delta) * dot(wave, wave);
          }
    Z delta_t{};
    for (int i = 0; i < 3; ++i)
    {
      delta_t -= .5 * lj[i];
      for (int dim = 0; dim < 3; ++dim)
        delta_t -= .5 * gj[i][dim] * gj[i][dim] + dgrad[i][dim] * gj[i][dim];
    }
    KineticJastrowResidual residual(electrons);
    CHECK(residual.evaluate(psi, electrons) == Approx(delta_t.real()).margin(algebra_tol));
    // Standard DLA: full kinetic plus local ionic plus determinant-only NL.
    // Express the kinetic from the actual spline derivatives, independently.
    Z standard{};
    for (int i = 0; i < 3; ++i)
      for (int band = 0; band < (i == 2 ? 1 : 2); ++band)
        standard += weights[i][band] *
            (-.5 * Z(lap(i, band)) + files.model.local(electrons.R[i]) * Z(m(i, band)) +
             files.model.field(band, electrons.R[i]).nonlocal * bsp->getMFRCoreActionSources()[band].rotation_phase);
    standard += delta_t - .125;
    const double mfr = std::real(actual) + residual.getValue() - .125;
    // A second derivative of a cubic spline has O(h^2) continuum error.
    CHECK(std::abs(mfr - standard.real()) < .12 * std::pow(16.0 / mesh, 2) + algebra_tol);
    app_log() << "MFR identity mesh=" << mesh << " batched=" << batched << " J=" << jastrow_a
              << " boundary=" << boundary << " skew=" << skew << " contraction_error=" << std::abs(Z(actual) - matched)
              << " continuum_core_error=" << error
              << " residual_error=" << std::abs(residual.getValue() - delta_t.real())
              << " DLA_spline_error=" << std::abs(mfr - standard.real()) << std::endl;
    // Exercise the actual Hamiltonian replacement/branching-energy arithmetic,
    // including a nonzero common electron-electron contribution and one ion constant.
    QMCHamiltonian ham;
    ham.addOperator(std::make_unique<BareKineticEnergy>(electrons), "Kinetic");
    ham.addOperator(std::make_unique<IdentityElectronEnergy>(), "ElecElec");
    ham.addOperator(core.makeClone(electrons), "MFRCoreAction", false);
    ham.addOperator(std::make_unique<KineticJastrowResidual>(electrons), "KineticJastrowResidual", false);
    ham.configureMFRQMCMode(MFRQMCModeRequest::YES);
    ham.addObservables(electrons);
    CHECK(ham.evaluate(psi, electrons) == Approx(mfr + .37).margin(algebra_tol));

    REQUIRE(core.getActionRowEvaluationCount() == 3);
    core.evaluateTrace(psi, electrons);
    CHECK(core.getActionRowEvaluationCount() == 3);
    // A wavefunction/operator clone starts with its own empty row cache.
    ParticleSet clone_particles(electrons);
    auto clone_psi = psi.makeClone(clone_particles);
    clone_psi->evaluateLog(clone_particles);
    auto clone_op    = core.makeClone(clone_particles);
    auto& clone_core = dynamic_cast<MFRCoreAction&>(*clone_op);
    CHECK(clone_core.getActionRowEvaluationCount() == 0);
    CHECK(std::abs(Z(clone_core.evaluateTrace(*clone_psi, clone_particles)) - Z(actual)) < algebra_tol);
    CHECK(clone_core.getActionRowEvaluationCount() == 3);
    if (mesh == 16 && !boundary && !skew && jastrow_a != 0)
    {
      // Distinct walker configurations, each with a wavefunction/operator clone.
      // Compare the production multiwalker boundary to individual evaluation.
      std::vector<ParticleSet> particles{electrons, electrons};
      particles[1].R[0][0] += .043;
      UPtrVector<TrialWaveFunction> wavefunctions;
      UPtrVector<QMCHamiltonian> hamiltonians;
      std::vector<double> individual;
      for (auto& particle : particles)
      {
        particle.update();
        wavefunctions.push_back(psi.makeClone(particle));
        wavefunctions.back()->evaluateLog(particle);
        hamiltonians.push_back(ham.makeClone(particle, *wavefunctions.back()));
        individual.push_back(hamiltonians.back()->evaluate(*wavefunctions.back(), particle));
      }
      auto p_refs = makeRefVector<ParticleSet>(particles);
      RefVectorWithLeader<ParticleSet> p_list(p_refs[0], p_refs);
      auto w_refs = convertUPtrToRefVector(wavefunctions);
      RefVectorWithLeader<TrialWaveFunction> w_list(w_refs[0], w_refs);
      auto h_refs = convertUPtrToRefVector(hamiltonians);
      RefVectorWithLeader<QMCHamiltonian> h_list(h_refs[0], h_refs);
      ResourceCollection resources("mfr_identity_hamiltonians");
      h_list.getLeader().createResource(resources);
      ResourceCollectionTeamLock<QMCHamiltonian> lock(resources, h_list);
      const auto energies = QMCHamiltonian::mw_evaluate(h_list, w_list, p_list);
      for (int walker = 0; walker < 2; ++walker)
        CHECK(energies[walker] == Approx(individual[walker]).margin(algebra_tol));
    }


    if (jastrow_a == 0)
    {
      // Pending trial positions must not invalidate accepted-position rows.
      electrons.makeMove(0, R3(.03, -.02, .01));
      psi.calcRatio(electrons, 0);
      psi.rejectMove(0);
      electrons.rejectMove(0);
      CHECK(std::abs(Z(core.evaluateTrace(psi, electrons)) - Z(actual)) < algebra_tol);
      CHECK(core.getActionRowEvaluationCount() == 3);
      size_t expected_rows = 3;
      for (const int electron : {0, 1, 0, 2, 0})
      {
        const R3 displacement = expected_rows == 7 ? R3(lattice.a(0)) : R3(.031, -.017, .023);
        electrons.makeMove(electron, displacement);
        psi.calcRatio(electrons, electron);
        psi.acceptMove(electrons, electron, true);
        electrons.acceptMove(electron);
        // evaluateTrace must flush the accepted delayed determinant update itself.
        const Z cached = core.evaluateTrace(psi, electrons);
        CHECK(core.getActionRowEvaluationCount() == ++expected_rows);
        orbitals.evaluate_notranspose(electrons, 0, 3, m, grad, lap);
        const auto determinant = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
        const Z w[3][2]        = {{Z(m(1, 1) / determinant), Z(-m(1, 0) / determinant)},
                                  {Z(-m(0, 1) / determinant), Z(m(0, 0) / determinant)},
                                  {Z(1.0) / Z(m(2, 0)), Z(0)}};
        Z fresh{};
        for (int row = 0; row < 3; ++row)
          for (int band = 0; band < (row == 2 ? 1 : 2); ++band)
            fresh += w[row][band] * core.evaluateOrbitalAction(*bsp, band, electrons.R[row]);
        CHECK(std::abs(cached - fresh) < algebra_tol);
        psi.evaluateLog(electrons);
        CHECK(std::abs(Z(core.evaluateTrace(psi, electrons)) - fresh) < algebra_tol);
        CHECK(core.getActionRowEvaluationCount() == expected_rows);
      }
      // A buffer/configuration restore can change every row without invoking
      // per-particle move callbacks. Accepted-coordinate stamps cover it too.
      electrons.R = original_positions;
      electrons.update();
      psi.evaluateLog(electrons);
      CHECK(std::abs(Z(core.evaluateTrace(psi, electrons)) - Z(actual)) < algebra_tol);
      CHECK(core.getActionRowEvaluationCount() == expected_rows + 3);
      loadAction(core, files.core.name());
      CHECK(core.getActionRowEvaluationCount() == 0);
      CHECK(std::abs(Z(core.evaluateTrace(psi, electrons)) - Z(actual)) < algebra_tol);
      CHECK(core.getActionRowEvaluationCount() == 3);
    }
  }
}
} // namespace qmcplusplus
