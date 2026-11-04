//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#include "MFRCoreAction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include "OhmmsData/AttributeSet.h"
#include "QMCWaveFunctions/BsplineFactory/BsplineSet.h"
#include "QMCWaveFunctions/BsplineFactory/OneSplineOrbData.hpp"
#include "QMCWaveFunctions/Fermion/DiracDeterminantBase.h"
#include "QMCWaveFunctions/Fermion/OneBodyOperatorTrace.hpp"
#include "QMCWaveFunctions/Fermion/SlaterDet.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "hdf/hdf_archive.h"

namespace qmcplusplus
{
namespace
{
// HDF5 is an external data boundary.  std::isfinite can be folded to true
// by the release build's -ffast-math, so classify IEEE-754 exponent bits.
bool isFiniteHDFValue(double value)
{
  static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == sizeof(std::uint64_t));
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr std::uint64_t exponent = UINT64_C(0x7ff0000000000000);
  return (bits & exponent) != exponent;
}

std::string trimAttribute(std::string value)
{
  const auto nul = value.find('\0');
  if (nul != std::string::npos)
    value.resize(nul);
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
    value.pop_back();
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
    value.erase(value.begin());
  return value;
}

std::string readFixedStringDataset(hid_t file, const std::string& name)
{
  const hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0)
    throw std::runtime_error("missing fixed-string HDF5 dataset '" + name + "'");
  const hid_t type  = H5Dget_type(dataset);
  const hid_t space = H5Dget_space(dataset);
  const size_t size = type < 0 ? 0 : H5Tget_size(type);
  const bool valid  = type >= 0 && space >= 0 && H5Sget_simple_extent_npoints(space) == 1 &&
      H5Tget_class(type) == H5T_STRING && H5Tis_variable_str(type) == 0 && size > 0 && size <= 256;
  std::vector<char> buffer(valid ? size + 1 : 1, '\0');
  const bool read = valid && H5Dread(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()) >= 0;
  if (space >= 0)
    H5Sclose(space);
  if (type >= 0)
    H5Tclose(type);
  H5Dclose(dataset);
  if (!read)
    throw std::runtime_error("expected one bounded fixed-string HDF5 dataset '" + name + "'");
  return trimAttribute(std::string(buffer.data(), size));
}

double readNumericScalarDataset(hid_t file, const std::string& name)
{
  const hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0)
    throw std::runtime_error("missing numeric scalar HDF5 dataset '" + name + "'");
  const hid_t type = H5Dget_type(dataset), space = H5Dget_space(dataset);
  const H5T_class_t category = type < 0 ? H5T_NO_CLASS : H5Tget_class(type);
  const bool valid = space >= 0 && H5Sget_simple_extent_npoints(space) == 1 && H5Sget_simple_extent_ndims(space) <= 1 &&
      (category == H5T_INTEGER || category == H5T_FLOAT);
  double value    = 0;
  const bool read = valid && H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value) >= 0;
  if (space >= 0)
    H5Sclose(space);
  if (type >= 0)
    H5Tclose(type);
  H5Dclose(dataset);
  if (!read || !isFiniteHDFValue(value))
    throw std::runtime_error("expected one finite numeric scalar HDF5 dataset '" + name + "'");
  return value;
}

bool isScalarAttribute(hid_t attribute)
{
  const hid_t space = H5Aget_space(attribute);
  if (space < 0)
    return false;
  const bool scalar = H5Sget_simple_extent_type(space) == H5S_SCALAR;
  H5Sclose(space);
  return scalar;
}

std::string readStringAttribute(hid_t object, const char* name)
{
  const hid_t attribute = H5Aopen(object, name, H5P_DEFAULT);
  if (attribute < 0)
    throw std::runtime_error(std::string("missing HDF5 attribute '") + name + "'");
  if (!isScalarAttribute(attribute))
  {
    H5Aclose(attribute);
    throw std::runtime_error(std::string("expected scalar HDF5 attribute '") + name + "'");
  }
  const hid_t datatype = H5Aget_type(attribute);
  if (datatype < 0)
  {
    H5Aclose(attribute);
    throw std::runtime_error(std::string("cannot inspect HDF5 attribute '") + name + "'");
  }
  if (H5Tget_class(datatype) != H5T_STRING || H5Tis_variable_str(datatype) != 0)
  {
    H5Tclose(datatype);
    H5Aclose(attribute);
    throw std::runtime_error(std::string("expected fixed-length HDF5 string attribute '") + name + "'");
  }
  const size_t size = H5Tget_size(datatype);
  std::vector<char> buffer(size + 1, '\0');
  const herr_t status = H5Aread(attribute, datatype, buffer.data());
  H5Tclose(datatype);
  H5Aclose(attribute);
  if (status < 0)
    throw std::runtime_error(std::string("cannot read HDF5 attribute '") + name + "'");
  return trimAttribute(std::string(buffer.data(), size));
}

int readIntAttribute(hid_t object, const char* name)
{
  const hid_t attribute = H5Aopen(object, name, H5P_DEFAULT);
  if (attribute < 0)
    throw std::runtime_error(std::string("missing HDF5 attribute '") + name + "'");
  if (!isScalarAttribute(attribute))
  {
    H5Aclose(attribute);
    throw std::runtime_error(std::string("expected scalar HDF5 attribute '") + name + "'");
  }
  int value           = 0;
  const herr_t status = H5Aread(attribute, H5T_NATIVE_INT, &value);
  H5Aclose(attribute);
  if (status < 0)
    throw std::runtime_error(std::string("cannot read HDF5 attribute '") + name + "'");
  return value;
}

double readDoubleAttribute(hid_t object, const char* name)
{
  const hid_t attribute = H5Aopen(object, name, H5P_DEFAULT);
  if (attribute < 0)
    throw std::runtime_error(std::string("missing HDF5 attribute '") + name + "'");
  if (!isScalarAttribute(attribute))
  {
    H5Aclose(attribute);
    throw std::runtime_error(std::string("expected scalar HDF5 attribute '") + name + "'");
  }
  double value        = 0.0;
  const herr_t status = H5Aread(attribute, H5T_NATIVE_DOUBLE, &value);
  H5Aclose(attribute);
  if (status < 0)
    throw std::runtime_error(std::string("cannot read HDF5 attribute '") + name + "'");
  return value;
}

template<typename T, size_t N>
std::array<T, N> readArrayAttribute(hid_t object, const char* name, hid_t memory_type)
{
  const hid_t attribute = H5Aopen(object, name, H5P_DEFAULT);
  if (attribute < 0)
    throw std::runtime_error(std::string("missing HDF5 attribute '") + name + "'");
  if (!isScalarAttribute(attribute))
  {
    H5Aclose(attribute);
    throw std::runtime_error(std::string("expected scalar HDF5 attribute '") + name + "'");
  }
  const hid_t datatype = H5Aget_type(attribute);
  if (datatype < 0)
  {
    H5Aclose(attribute);
    throw std::runtime_error(std::string("cannot inspect HDF5 attribute '") + name + "'");
  }
  if (H5Tget_class(datatype) != H5T_ARRAY || H5Tget_array_ndims(datatype) != 1)
  {
    H5Tclose(datatype);
    H5Aclose(attribute);
    throw std::runtime_error(std::string("unexpected HDF5 attribute shape for '") + name + "'");
  }
  hsize_t length = 0;
  H5Tget_array_dims2(datatype, &length);
  if (length != N)
  {
    H5Tclose(datatype);
    H5Aclose(attribute);
    throw std::runtime_error(std::string("unexpected HDF5 attribute length for '") + name + "'");
  }
  std::array<T, N> values{};
  const hsize_t memory_length   = N;
  const hid_t array_memory_type = H5Tarray_create2(memory_type, 1, &memory_length);
  if (array_memory_type < 0)
  {
    H5Tclose(datatype);
    H5Aclose(attribute);
    throw std::runtime_error(std::string("cannot construct array datatype for '") + name + "'");
  }
  const herr_t status = H5Aread(attribute, array_memory_type, values.data());
  H5Tclose(array_memory_type);
  H5Tclose(datatype);
  H5Aclose(attribute);
  if (status < 0)
    throw std::runtime_error(std::string("cannot read HDF5 attribute '") + name + "'");
  return values;
}

std::vector<double> readDoubleDataset(hid_t file,
                                      const std::string& name,
                                      const std::vector<hsize_t>& expected_dimensions)
{
  const hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0)
    throw std::runtime_error("missing HDF5 dataset '" + name + "'");
  const hid_t dataspace = H5Dget_space(dataset);
  const int rank        = dataspace < 0 ? -1 : H5Sget_simple_extent_ndims(dataspace);
  if (rank < 0)
  {
    if (dataspace >= 0)
      H5Sclose(dataspace);
    H5Dclose(dataset);
    throw std::runtime_error("cannot inspect HDF5 dataset '" + name + "'");
  }
  std::vector<hsize_t> dimensions(rank);
  H5Sget_simple_extent_dims(dataspace, dimensions.data(), nullptr);
  if (dimensions != expected_dimensions)
  {
    H5Sclose(dataspace);
    H5Dclose(dataset);
    throw std::runtime_error("unexpected dimensions for HDF5 dataset '" + name + "'");
  }
  size_t size = 1;
  for (const hsize_t dimension : dimensions)
    size *= dimension;
  std::vector<double> values(size);
  const herr_t status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Sclose(dataspace);
  H5Dclose(dataset);
  if (status < 0)
    throw std::runtime_error("cannot read HDF5 dataset '" + name + "'");
  if (!std::all_of(values.begin(), values.end(), [](double value) { return isFiniteHDFValue(value); }))
    throw std::runtime_error("non-finite values in HDF5 dataset '" + name + "'");
  return values;
}

std::vector<double> readDoubleVector(hid_t file, const std::string& name)
{
  const hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0)
    throw std::runtime_error("missing HDF5 dataset '" + name + "'");
  const hid_t dataspace = H5Dget_space(dataset);
  if (H5Sget_simple_extent_ndims(dataspace) != 1)
  {
    H5Sclose(dataspace);
    H5Dclose(dataset);
    throw std::runtime_error("expected rank-one HDF5 dataset '" + name + "'");
  }
  hsize_t length = 0;
  H5Sget_simple_extent_dims(dataspace, &length, nullptr);
  std::vector<double> values(length);
  const herr_t status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Sclose(dataspace);
  H5Dclose(dataset);
  if (status < 0)
    throw std::runtime_error("cannot read HDF5 dataset '" + name + "'");
  if (!std::all_of(values.begin(), values.end(), [](double value) { return isFiniteHDFValue(value); }))
    throw std::runtime_error("non-finite values in HDF5 dataset '" + name + "'");
  return values;
}

std::vector<int> readIntegerGvectors(hid_t file, const std::string& name, size_t& count)
{
  const hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0)
    throw std::runtime_error("missing HDF5 dataset '" + name + "'");
  const hid_t dataspace = H5Dget_space(dataset);
  hsize_t dimensions[2]{};
  if (H5Sget_simple_extent_ndims(dataspace) != 2)
  {
    H5Sclose(dataspace);
    H5Dclose(dataset);
    throw std::runtime_error("expected rank-two G-vector dataset '" + name + "'");
  }
  H5Sget_simple_extent_dims(dataspace, dimensions, nullptr);
  if (dimensions[1] != 3 || dimensions[0] == 0)
  {
    H5Sclose(dataspace);
    H5Dclose(dataset);
    throw std::runtime_error("unexpected G-vector dimensions in '" + name + "'");
  }
  count = dimensions[0];
  std::vector<int> values(count * 3);
  const herr_t status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Sclose(dataspace);
  H5Dclose(dataset);
  if (status < 0)
    throw std::runtime_error("cannot read HDF5 dataset '" + name + "'");
  return values;
}

std::string indexedPath(const char* format, int a, int b = -1, int c = -1)
{
  std::array<char, 256> buffer{};
  int length = 0;
  if (c >= 0)
    length = std::snprintf(buffer.data(), buffer.size(), format, a, b, c);
  else if (b >= 0)
    length = std::snprintf(buffer.data(), buffer.size(), format, a, b);
  else
    length = std::snprintf(buffer.data(), buffer.size(), format, a);
  if (length < 0 || static_cast<size_t>(length) >= buffer.size())
    throw std::runtime_error("failed to generate MFR action HDF5 path");
  return std::string(buffer.data(), length);
}

std::vector<int> readIntegerVectorAttribute(hid_t object, const char* name, size_t count)
{
  const hid_t attribute = H5Aopen(object, name, H5P_DEFAULT);
  if (attribute < 0)
    throw std::runtime_error(std::string("missing integer vector attribute '") + name + "'");
  const hid_t type  = H5Aget_type(attribute);
  const hid_t space = H5Aget_space(attribute);
  hid_t memory_type = -1;
  hsize_t dimension = 0;
  if (H5Sget_simple_extent_type(space) == H5S_SCALAR && H5Tget_class(type) == H5T_ARRAY &&
      H5Tget_array_ndims(type) == 1)
  {
    H5Tget_array_dims2(type, &dimension);
    const hid_t base = H5Tget_super(type);
    if (dimension == count && H5Tget_class(base) == H5T_INTEGER)
      memory_type = H5Tarray_create2(H5T_NATIVE_INT, 1, &dimension);
    H5Tclose(base);
  }
  else if (H5Sget_simple_extent_ndims(space) == 1 && H5Tget_class(type) == H5T_INTEGER)
  {
    H5Sget_simple_extent_dims(space, &dimension, nullptr);
    if (dimension == count)
      memory_type = H5Tcopy(H5T_NATIVE_INT);
  }
  std::vector<int> values(count);
  const bool valid = memory_type >= 0 && H5Aread(attribute, memory_type, values.data()) >= 0;
  if (memory_type >= 0)
    H5Tclose(memory_type);
  H5Sclose(space);
  H5Tclose(type);
  H5Aclose(attribute);
  if (!valid)
    throw std::runtime_error(std::string("invalid integer vector attribute '") + name + "'");
  return values;
}

using GKey = std::array<int, 3>;

std::map<GKey, std::complex<double>> readESHDFOrbitals(const std::string& file_name,
                                                       int spin,
                                                       int twist,
                                                       int band,
                                                       const std::array<double, 3>& expected_reduced_k)
{
  hdf_archive file;
  if (!file.open(file_name, H5F_ACC_RDONLY))
    throw std::runtime_error("cannot open paired orbital HDF5 file '" + file_name + "'");
  Vector<TinyVector<int, 3>> gvectors;
  Vector<std::complex<double>> coefficients;
  Vector<double> reduced_k;
  const std::string k_path = indexedPath("/electrons/kpoint_%d/reduced_k", twist);
  const std::string g_path = indexedPath("/electrons/kpoint_%d/gvectors", twist);
  const std::string c_path = indexedPath("/electrons/kpoint_%d/spin_%d/state_%d/psi_g", twist, spin, band);
  if (!file.readEntry(gvectors, g_path) || !file.readEntry(coefficients, c_path) || !file.readEntry(reduced_k, k_path))
  {
    file.close();
    throw std::runtime_error("paired ES-HDF orbital datasets are missing for selected MFR action state");
  }
  file.close();
  if (reduced_k.size() != 3)
    throw std::runtime_error("paired ES-HDF reduced k point must have three components");
  for (int direction = 0; direction < 3; ++direction)
    if (!isFiniteHDFValue(reduced_k[direction]) ||
        std::abs(reduced_k[direction] - expected_reduced_k[direction]) > 1.0e-10)
      throw std::runtime_error("MFRCoreAction reduced k point does not match the selected ES-HDF state");
  if (gvectors.size() != coefficients.size())
    throw std::runtime_error("paired ES-HDF G-vector and coefficient counts differ");
  std::map<GKey, std::complex<double>> result;
  for (size_t i = 0; i < gvectors.size(); ++i)
  {
    GKey key{gvectors[i][0], gvectors[i][1], gvectors[i][2]};
    if (!result.emplace(key, coefficients[i]).second)
      throw std::runtime_error("duplicate G vector in paired ES-HDF orbital");
  }
  return result;
}
} // namespace

MFRCoreAction::MFRCoreAction(ParticleSet& pset, TrialWaveFunction* psi, ReferencePayload payload)
    : payload_(payload), pset_(pset), psi_(psi)
{
  setEnergyDomain(POTENTIAL);
  oneBodyQuantumDomain(pset);
}

bool MFRCoreAction::put(xmlNodePtr cur)
{
  const std::string expected_name = payload_ == ReferencePayload::CORE_ACTION ? "MFRCoreAction" : "MFRMetaFields";
  std::string name(expected_name), file_name;
  OhmmsAttributeSet attributes;
  attributes.add(file_name, "href");
  attributes.add(name, "name");
  attributes.put(cur);
  if (file_name.empty())
    throw std::runtime_error("MFRCoreAction requires an href attribute");
  if (name != expected_name)
    throw std::runtime_error("MFR reference observable name must be '" + expected_name + "'");

  // Reconfiguration owns a fresh snapshot. Existing clones retain their
  // immutable shared tables; no state/row from the old payload is reused.
  file_name_ = std::move(file_name);
  data_      = std::make_shared<SharedData>();
  row_cache_.clear();
  action_row_evaluations_ = 0;
  value_ = imaginary_trace_ = total_energy_mf_ = 0.0;
  readHeader();
  validateCell(pset_);
  if (psi_ != nullptr)
    prepareWaveFunction(*psi_);
  app_log() << "  MFRCoreAction file: " << file_name_ << '\n'
            << "  MFRCoreAction grid: " << grid_dimensions_[0] << ' ' << grid_dimensions_[1] << ' '
            << grid_dimensions_[2] << '\n'
            << "  MFRCoreAction spin channels: " << spin_channels_ << '\n'
            << "  IonIonMF: " << ion_ion_energy_mf_ << " Hartree\n";
  return true;
}

void MFRCoreAction::readHeader()
{
  const hid_t file = H5Fopen(file_name_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
    throw std::runtime_error("MFRCoreAction failed to open HDF5 file '" + file_name_ + "'");
  try
  {
    const bool core_payload = payload_ == ReferencePayload::CORE_ACTION;
    if (readStringAttribute(file, "schema_family") != (core_payload ? "mfr_core_action" : "mfr_meta_fields"))
      throw std::runtime_error("unexpected MFR core-action schema family");
    if (readIntAttribute(file, "schema_version") != 1)
      throw std::runtime_error("unsupported MFR core-action schema version");
    if (readIntAttribute(file, "complete") != 1)
      throw std::runtime_error("MFR core-action file is incomplete");
    if (readStringAttribute(file, "representation") !=
        (core_payload ? "periodic_bloch_core_action_grid" : "periodic_bloch_meta_fields"))
      throw std::runtime_error("unsupported MFR core-action representation");
    if (readStringAttribute(file, "energy_units") != "hartree")
      throw std::runtime_error("MFR core-action energy units must be Hartree");
    if (readStringAttribute(file, "length_units") != "bohr")
      throw std::runtime_error("MFR core-action length units must be bohr");
    if (readStringAttribute(file, "orbital_normalization") != "sum_G c_G exp(+i G.r); no volume factor")
      throw std::runtime_error("unsupported MFR core-action orbital normalization");
    if (readStringAttribute(file, "bloch_phase") != "exp(+2 pi i reduced_k.reduced_r)")
      throw std::runtime_error("unsupported MFR core-action Bloch-phase convention");
    if (core_payload && readStringAttribute(file, "grid_order") != "HDF5: z,y,x; x fastest")
      throw std::runtime_error("unsupported MFR core-action grid ordering");
    if (readStringAttribute(file, "periodic_grid_endpoint") != "excluded")
      throw std::runtime_error("MFR core-action grid must exclude the periodic endpoint");

    if (core_payload)
      grid_dimensions_ = readArrayAttribute<int, 3>(file, "grid_dimensions", H5T_NATIVE_INT);
    else
    {
      if (readStringAttribute(file, "tau_definition") != "sum_j occupation_j*abs(grad(phi_j))^2/2" ||
          readStringAttribute(file, "differential_operator") != "-0.5*div(q*grad)" ||
          readStringAttribute(file, "approximation") != "two_field_no_reference_residual" ||
          readStringAttribute(file, "potential_includes_ions") != "false" ||
          readDoubleAttribute(file, "hartree_g0") != 0.0)
        throw std::runtime_error("unsupported MFR meta-field differential/potential convention");
    }
    spin_channels_     = readIntAttribute(file, "spin_channels");
    number_of_kpoints_ = readIntAttribute(file, "number_of_kpoints");
    number_of_states_  = readIntAttribute(file, "number_of_states");
    ion_ion_energy_mf_ = readDoubleAttribute(file, "ion_ion_energy");
    if (spin_channels_ != 1 && spin_channels_ != 2)
      throw std::runtime_error("MFRCoreAction requires one or two collinear spin channels");
    if (number_of_kpoints_ < 1 || number_of_states_ < 1)
      throw std::runtime_error("MFRCoreAction file has no k points or states");
    if (core_payload && std::any_of(grid_dimensions_.begin(), grid_dimensions_.end(), [](int n) { return n < 4; }))
      throw std::runtime_error("MFRCoreAction grid dimensions must be at least four for cubic splines");
    if (!isFiniteHDFValue(ion_ion_energy_mf_))
      throw std::runtime_error("MFRCoreAction ion-ion energy is non-finite");

    const auto lattice_values = readDoubleDataset(file, "/lattice_vectors", {3, 3});
    Tensor<FullPrecRealType, OHMMS_DIM> lattice;
    for (int i = 0; i < OHMMS_DIM; ++i)
      for (int j = 0; j < OHMMS_DIM; ++j)
        lattice(i, j) = lattice_values[i * OHMMS_DIM + j];
    const double volume = det(lattice);
    double scale        = 0.0;
    for (int i = 0; i < 9; ++i)
      scale = std::max(scale, std::abs(lattice[i]));
    if (!isFiniteHDFValue(volume) ||
        std::abs(volume) <= 8.0 * std::numeric_limits<double>::epsilon() * scale * scale * scale)
      throw std::runtime_error("MFRCoreAction lattice is singular or numerically degenerate");
    data_->field_lattice.set(lattice);
  }
  catch (...)
  {
    H5Fclose(file);
    throw;
  }
  H5Fclose(file);
  if (spin_channels_ == 2 && pset_.groups() != 2)
    throw std::runtime_error("two-channel MFRCoreAction requires two electron groups");
}

void MFRCoreAction::validateCell(const ParticleSet& pset) const
{
  const Lattice& simulation_lattice = pset.getLattice();
  for (int direction = 0; direction < OHMMS_DIM; ++direction)
  {
    if (!simulation_lattice.getBoxBConds()[direction])
      throw std::runtime_error("MFRCoreAction requires a fully periodic simulation cell");
    const PosType reduced_vector = data_->field_lattice.toUnit(simulation_lattice.a(direction));
    for (int component = 0; component < OHMMS_DIM; ++component)
    {
      const double expected = component == direction ? 1.0 : 0.0;
      if (std::abs(reduced_vector[component] - expected) > 1.0e-8)
        throw std::runtime_error("QMCPACK cell must match the primitive MFRCoreAction cell without tiling");
    }
  }
}

void MFRCoreAction::validateIons(hid_t file, const BsplineSet& orbital_set) const
{
  const ParticleSet* ions = orbital_set.getMFRCoreActionIons();
  if (!ions || ions->getTotalNum() == 0)
    throw std::runtime_error("MFRCoreAction requires the B-spline ionic source particle set");
  const size_t nat       = ions->getTotalNum();
  const auto positions   = readDoubleDataset(file, "/ion_positions", {nat, 3});
  const auto species_ids = readIntegerVectorAttribute(file, "ion_species", nat);

  // Pair to both the selected orbital file and the actual QMC ionic source.
  // ES-HDF names can be labels (C1, C2); atomic numbers are the chemical IDs.
  hdf_archive es;
  if (!es.open(orbital_set.getMFRCoreActionOrbitalFile(), H5F_ACC_RDONLY))
    throw std::runtime_error("cannot open paired ES-HDF for MFR ionic validation");
  using PrecisePos              = QMCTraits::QTFull::PosType;
  const auto es_positions       = readDoubleDataset(es.getFileID(), "/atoms/positions", {nat, 3});
  const auto es_ids             = readDoubleDataset(es.getFileID(), "/atoms/species_ids", {nat});
  const auto& actual_species    = ions->getSpeciesSet();
  const int charge_index        = actual_species.findAttribute("charge");
  const int atomic_number_index = actual_species.findAttribute("atomicnumber");
  if (charge_index < 0)
    throw std::runtime_error("MFRCoreAction ionic source species require charge attributes");
  std::vector<bool> matched(nat, false);
  for (size_t i = 0; i < nat; ++i)
  {
    if (species_ids[i] < 1 || species_ids[i] > nat || es_ids[i] < 0 || es_ids[i] >= nat ||
        es_ids[i] != std::floor(es_ids[i]))
      throw std::runtime_error("invalid MFRCoreAction/ES-HDF ionic species index");
    PrecisePos position;
    for (int d = 0; d < 3; ++d)
      position[d] = positions[3 * i + d];
    auto minimum_displacement = [&](const PrecisePos& a, const PrecisePos& b) {
      PrecisePos reduced = data_->field_lattice.toUnit(a - b);
      for (int d = 0; d < 3; ++d)
        reduced[d] -= std::round(reduced[d]);
      return data_->field_lattice.toCart(reduced);
    };
    const PrecisePos paired_position(es_positions[3 * i], es_positions[3 * i + 1], es_positions[3 * i + 2]);
    const auto paired_displacement = minimum_displacement(position, paired_position);
    if (dot(paired_displacement, paired_displacement) > 1e-16)
      throw std::runtime_error("MFRCoreAction ion positions do not match the paired ES-HDF geometry");
    const std::string group_name = indexedPath("/species_%d", species_ids[i] - 1);
    const hid_t group            = H5Gopen2(file, group_name.c_str(), H5P_DEFAULT);
    if (group < 0)
      throw std::runtime_error("MFRCoreAction is missing ionic species metadata");
    std::string label;
    double charge;
    try
    {
      label  = readStringAttribute(group, "element");
      charge = readDoubleAttribute(group, "valence_charge");
    }
    catch (...)
    {
      H5Gclose(group);
      throw;
    }
    H5Gclose(group);
    if (!isFiniteHDFValue(charge))
      throw std::runtime_error("MFRCoreAction ionic charge is non-finite");
    const std::string es_species = indexedPath("/atoms/species_%d", static_cast<int>(es_ids[i]));
    const double atomic_number   = readNumericScalarDataset(es.getFileID(), es_species + "/atomic_number");
    const double es_charge       = readNumericScalarDataset(es.getFileID(), es_species + "/valence_charge");
    const std::string es_label   = readFixedStringDataset(es.getFileID(), es_species + "/name");
    if (atomic_number <= 0 || atomic_number != std::floor(atomic_number) || label != es_label ||
        std::abs(charge - es_charge) > 1e-8)
      throw std::runtime_error("MFRCoreAction ionic species do not match the paired ES-HDF species");
    double coordinate_scale = 1.0;
    for (int axis = 0; axis < 3; ++axis)
      coordinate_scale =
          std::max(coordinate_scale, std::sqrt(dot(data_->field_lattice.a(axis), data_->field_lattice.a(axis))));
    const double coordinate_tolerance =
        std::max(1e-8, 4.0 * std::numeric_limits<RealType>::epsilon() * coordinate_scale);
    bool found = false;
    for (size_t j = 0; j < nat; ++j)
    {
      if (matched[j])
        continue;
      const auto dr = minimum_displacement(position, ions->R[j]);
      if (dot(dr, dr) > coordinate_tolerance * coordinate_tolerance)
        continue;
      const int actual_id = ions->GroupID[j];
      if (!isFiniteHDFValue(actual_species(charge_index, actual_id)) ||
          (atomic_number_index >= 0 && !isFiniteHDFValue(actual_species(atomic_number_index, actual_id))) ||
          std::abs(actual_species(charge_index, actual_id) - charge) > 1e-8 ||
          (atomic_number_index >= 0 ? std::abs(actual_species(atomic_number_index, actual_id) - atomic_number) > 1e-8
                                    : actual_species.getSpeciesName(actual_id) != label))
        throw std::runtime_error(
            "MFRCoreAction ionic source species/charge mismatch (provide atomicnumber for differing labels)");
      matched[j] = true;
      found      = true;
      break;
    }
    if (!found)
      throw std::runtime_error("MFRCoreAction ionic geometry differs from the QMC source particle set");
  }
}

MFRCoreAction::ActionState MFRCoreAction::readState(const StateKey& key, BsplineSet* orbital_set, int orbital_index)
{
  if (key.spin < 0 || key.spin >= spin_channels_ || key.twist < 0 || key.twist >= number_of_kpoints_ || key.band < 0 ||
      key.band >= number_of_states_)
    throw std::runtime_error("selected MFRCoreAction spin/k-point/band is outside the exported state range");

  const hid_t file = H5Fopen(file_name_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
    throw std::runtime_error("MFRCoreAction failed to reopen HDF5 file '" + file_name_ + "'");
  ActionState state;
  try
  {
    const std::string k_path     = indexedPath("/kpoint_%d", key.twist);
    const std::string spin_path  = indexedPath("/kpoint_%d/spin_%d", key.twist, key.spin);
    const std::string state_path = indexedPath("/kpoint_%d/spin_%d/state_%d", key.twist, key.spin, key.band);
    const hid_t k_group          = H5Gopen2(file, k_path.c_str(), H5P_DEFAULT);
    const hid_t state_group      = H5Gopen2(file, state_path.c_str(), H5P_DEFAULT);
    if (k_group < 0 || state_group < 0)
    {
      if (k_group >= 0)
        H5Gclose(k_group);
      if (state_group >= 0)
        H5Gclose(state_group);
      throw std::runtime_error("MFRCoreAction is missing a selected k-point or state group");
    }
    state.reduced_k  = readArrayAttribute<double, 3>(k_group, "reduced_k", H5T_NATIVE_DOUBLE);
    state.eigenvalue = readDoubleAttribute(state_group, "eigenvalue");
    H5Gclose(k_group);
    H5Gclose(state_group);
    if (!isFiniteHDFValue(state.eigenvalue) ||
        !std::all_of(state.reduced_k.begin(), state.reduced_k.end(),
                     [](double value) { return isFiniteHDFValue(value); }))
      throw std::runtime_error("MFRCoreAction state has a non-finite eigenvalue or reduced k point");

    if (payload_ == ReferencePayload::CORE_ACTION)
    {
      const auto real_values =
          readDoubleDataset(file, state_path + "/action_real",
                            {static_cast<hsize_t>(grid_dimensions_[2]), static_cast<hsize_t>(grid_dimensions_[1]),
                             static_cast<hsize_t>(grid_dimensions_[0])});
      const auto imag_values =
          readDoubleDataset(file, state_path + "/action_imag",
                            {static_cast<hsize_t>(grid_dimensions_[2]), static_cast<hsize_t>(grid_dimensions_[1]),
                             static_cast<hsize_t>(grid_dimensions_[0])});
      Array<double, 3> real_grid(grid_dimensions_[0], grid_dimensions_[1], grid_dimensions_[2]);
      Array<double, 3> imag_grid(grid_dimensions_[0], grid_dimensions_[1], grid_dimensions_[2]);
      for (int i = 0; i < grid_dimensions_[0]; ++i)
        for (int j = 0; j < grid_dimensions_[1]; ++j)
          for (int k = 0; k < grid_dimensions_[2]; ++k)
          {
            const size_t index = (static_cast<size_t>(k) * grid_dimensions_[1] + j) * grid_dimensions_[0] + i;
            real_grid(i, j, k) = real_values[index];
            imag_grid(i, j, k) = imag_values[index];
          }

      Ugrid grids[3];
      BCtype_d boundaries[3];
      for (int d = 0; d < 3; ++d)
      {
        grids[d].start      = 0.0;
        grids[d].end        = 1.0;
        grids[d].num        = grid_dimensions_[d];
        boundaries[d].lCode = boundaries[d].rCode = PERIODIC;
        boundaries[d].lVal = boundaries[d].rVal = 0.0;
      }
      state.real = std::shared_ptr<UBspline_3d_d>(create_UBspline_3d_d(grids[0], grids[1], grids[2], boundaries[0],
                                                                       boundaries[1], boundaries[2], real_grid.data()),
                                                  destroy_Bspline);
      state.imag = std::shared_ptr<UBspline_3d_d>(create_UBspline_3d_d(grids[0], grids[1], grids[2], boundaries[0],
                                                                       boundaries[1], boundaries[2], imag_grid.data()),
                                                  destroy_Bspline);
      if (!state.real || !state.imag)
        throw std::runtime_error("failed to create periodic MFRCoreAction cubic splines");
    }

    if (orbital_set != nullptr)
    {
      validateIons(file, *orbital_set);
      size_t num_gvectors       = 0;
      const std::string g_path  = spin_path + "/gvectors";
      const auto gvectors_flat  = readIntegerGvectors(file, g_path, num_gvectors);
      const auto orbital_g_real = readDoubleVector(file, state_path + "/orbital_g_real");
      const auto orbital_g_imag = readDoubleVector(file, state_path + "/orbital_g_imag");
      if (orbital_g_real.size() != num_gvectors || orbital_g_imag.size() != num_gvectors)
        throw std::runtime_error("MFRCoreAction G-vector and orbital coefficient counts differ");

      std::map<GKey, std::complex<double>> action_orbital;
      const bool needs_action_phase = payload_ == ReferencePayload::CORE_ACTION;
      std::vector<TinyVector<int, 3>> gvector_values(needs_action_phase ? num_gvectors : 0);
      Vector<std::complex<double>> coefficients(needs_action_phase ? num_gvectors : 0);
      for (size_t ig = 0; ig < num_gvectors; ++ig)
      {
        GKey g{gvectors_flat[3 * ig], gvectors_flat[3 * ig + 1], gvectors_flat[3 * ig + 2]};
        const std::complex<double> coefficient(orbital_g_real[ig], orbital_g_imag[ig]);
        if (!action_orbital.emplace(g, coefficient).second)
          throw std::runtime_error("duplicate G vector in MFRCoreAction orbital identity data");
        if (needs_action_phase)
        {
          gvector_values[ig] = TinyVector<int, 3>(g[0], g[1], g[2]);
          coefficients[ig]   = coefficient;
        }
      }

      const auto paired_orbital =
          readESHDFOrbitals(orbital_set->getMFRCoreActionOrbitalFile(), key.spin, key.twist, key.band, state.reduced_k);
      if (paired_orbital.size() != action_orbital.size())
        throw std::runtime_error("MFRCoreAction and ES-HDF orbital G-vector counts do not match");
      for (const auto& [g, coefficient] : action_orbital)
      {
        const auto iter = paired_orbital.find(g);
        if (iter == paired_orbital.end() || std::abs(iter->second - coefficient) > 1.0e-12)
          throw std::runtime_error("MFRCoreAction orbital coefficients do not match the selected ES-HDF state");
      }

      // The core-action grids must follow the SPO's real/complex phase.
      // Meta fields act on the actual SPO VGL and need identity validation
      // only; reconstructing a phase here wastes an orbital-sized FFT and
      // can also overwrite phase state owned by a comparison core provider.
      if (needs_action_phase)
      {
        OneSplineOrbData phase_orbital(orbital_set->getMFRCoreActionMesh(), orbital_set->getMFRCoreActionHalfG(),
                                       orbital_set->usesDuplexSplinesForMFRCoreAction());
        phase_orbital.fft_spline(coefficients, gvector_values,
                                 TinyVector<double, 3>(state.reduced_k[0], state.reduced_k[1], state.reduced_k[2]),
                                 orbital_set->rotatesOrbitalsForMFRCoreAction());
        orbital_set->setMFRCoreActionRotationPhase(orbital_index, phase_orbital.getRotatePhase());
      }
    }
  }
  catch (...)
  {
    H5Fclose(file);
    throw;
  }
  H5Fclose(file);
  return state;
}

void MFRCoreAction::loadActionState(int spin, int twist, int band)
{
  const StateKey key{spin, twist, band};
  if (data_->states.find(key) == data_->states.end())
    data_->states.emplace(key, readState(key, nullptr, -1));
}

void MFRCoreAction::prepareWaveFunction(TrialWaveFunction& psi)
{
  if (!psi.findMSD().empty())
    throw std::runtime_error("MFRCoreAction currently requires a single Slater determinant product");
  auto slater_dets = psi.findSD();
  if (slater_dets.size() != 1)
    throw std::runtime_error("MFRCoreAction requires exactly one SlaterDet wavefunction component");

  total_energy_mf_      = ion_ion_energy_mf_;
  SlaterDet& slater_det = slater_dets[0].get();
  for (int idet = 0; idet < slater_det.getNumDets(); ++idet)
  {
    DiracDeterminantBase& determinant = slater_det.getDet(idet);
    auto* orbital_set                 = dynamic_cast<BsplineSet*>(&determinant.getPhi());
    if (orbital_set == nullptr)
      throw std::runtime_error("MFRCoreAction requires fixed B-spline SPOs");
    if (orbital_set->getMFRCoreActionSources().size() != orbital_set->size())
      throw std::runtime_error("B-spline SPO has no complete MFRCoreAction source map");

    auto& sources               = orbital_set->getMFRCoreActionSources();
    const int occupied_orbitals = determinant.getLastIndex() - determinant.getFirstIndex();
    if (occupied_orbitals > sources.size())
      throw std::runtime_error("MFRCoreAction has fewer source orbitals than the selected determinant");
    for (int orbital = 0; orbital < occupied_orbitals; ++orbital)
    {
      const auto& source = sources[orbital];
      if (spin_channels_ == 2 && source.spin != pset_.GroupID[determinant.getFirstIndex()])
        throw std::runtime_error("MFRCoreAction selected spin does not match the determinant electron group");
      const StateKey key{source.spin, source.twist, source.band};
      ActionState state     = readState(key, orbital_set, orbital);
      auto [iter, inserted] = data_->states.emplace(key, std::move(state));
      total_energy_mf_ += iter->second.eigenvalue;
    }
  }
}

std::complex<double> MFRCoreAction::evaluatePeriodicAction(int spin, int twist, int band, const PosType& position) const
{
  if (payload_ != ReferencePayload::CORE_ACTION)
    throw std::logic_error("meta reference contains no orbital-action splines");
  const auto iter = data_->states.find(StateKey{spin, twist, band});
  if (iter == data_->states.end())
    throw std::runtime_error("requested MFRCoreAction state was not loaded");
  // Keep both wrapping tolerance and spline coordinates in full precision.
  // A float epsilon clips small negative positions, and narrowing a wrapped
  // point can round it to the excluded upper endpoint.
  const auto reduced = data_->field_lattice.toUnit_floor(QMCTraits::QTFull::PosType(position));
  double real_value  = 0.0;
  double imag_value  = 0.0;
  eval_UBspline_3d_d(iter->second.real.get(), reduced[0], reduced[1], reduced[2], &real_value);
  eval_UBspline_3d_d(iter->second.imag.get(), reduced[0], reduced[1], reduced[2], &imag_value);
  return {real_value, imag_value};
}

std::complex<double> MFRCoreAction::evaluateOrbitalAction(const BsplineSet& orbital_set,
                                                          int orbital_index,
                                                          const PosType& position) const
{
  const auto& source = orbital_set.getMFRCoreActionSources().at(orbital_index);
  const StateKey key{source.spin, source.twist, source.band};
  auto value = evaluatePeriodicAction(key.spin, key.twist, key.band, position) * source.rotation_phase;
  if (orbital_set.usesDuplexSplinesForMFRCoreAction())
  {
    const double angle = -dot(position, source.qmc_k_cart);
    value *= std::complex<double>(std::cos(angle), std::sin(angle));
  }
  else
  {
    const auto& state = data_->states.at(key);
    // The periodic table is wrapped, but the Bloch phase uses the original
    // coordinate.  Dropping the cell winding loses antiperiodic signs at TRIM.
    const auto reduced = data_->field_lattice.toUnit(QMCTraits::QTFull::PosType(position));
    const double angle = 2.0 * M_PI *
        (state.reduced_k[0] * reduced[0] + state.reduced_k[1] * reduced[1] + state.reduced_k[2] * reduced[2]);
    value *= std::complex<double>(std::cos(angle), std::sin(angle));
  }

  switch (source.part)
  {
  case BsplineSet::MFRCoreActionPart::COMPLEX:
    return value;
  case BsplineSet::MFRCoreActionPart::REAL:
    return {value.real(), 0.0};
  case BsplineSet::MFRCoreActionPart::IMAGINARY:
    return {value.imag(), 0.0};
  }
  throw std::logic_error("unknown MFRCoreAction real/complex orbital component");
}

void MFRCoreAction::ensureReference(TrialWaveFunction& psi)
{
  if (data_->states.empty())
    prepareWaveFunction(psi);
}

bool MFRCoreAction::matchesReference(const MFRCoreAction& other) const
{
  if (spin_channels_ != other.spin_channels_ || data_->states.size() != other.data_->states.size() ||
      std::abs(ion_ion_energy_mf_ - other.ion_ion_energy_mf_) > 1e-12)
    return false;
  for (const auto& [key, state] : data_->states)
  {
    const auto found = other.data_->states.find(key);
    if (found == other.data_->states.end() || std::abs(state.eigenvalue - found->second.eigenvalue) > 1e-12)
      return false;
    for (int d = 0; d < 3; ++d)
      if (std::abs(state.reduced_k[d] - found->second.reduced_k[d]) > 1e-12)
        return false;
  }
  return true;
}

MFRCoreAction::FullPrecValueType MFRCoreAction::evaluateTrace(TrialWaveFunction& psi, ParticleSet& pset)
{
  if (payload_ != ReferencePayload::CORE_ACTION)
    throw std::logic_error("meta reference cannot evaluate a core-action trace");
  if (data_->states.empty())
    prepareWaveFunction(psi);
  psi.completeUpdates();
  const auto slater_dets = psi.findSD();
  if (slater_dets.size() != 1)
    throw std::runtime_error("MFRCoreAction requires exactly one SlaterDet wavefunction component");

  std::complex<double> total_trace(0.0, 0.0);
  row_cache_.resize(slater_dets[0].get().getNumDets());
  for (int idet = 0; idet < slater_dets[0].get().getNumDets(); ++idet)
  {
    DiracDeterminantBase& determinant = slater_dets[0].get().getDet(idet);
    auto* orbital_set                 = dynamic_cast<BsplineSet*>(&determinant.getPhi());
    if (orbital_set == nullptr)
      throw std::runtime_error("MFRCoreAction requires fixed B-spline SPOs");

    SPOSet::ValueMatrix inverse_transpose;
    determinant.copyInverseTranspose(inverse_transpose);
    auto& cache = row_cache_[idet];
    if (cache.orbital_set != orbital_set || cache.action.rows() != inverse_transpose.rows() ||
        cache.action.cols() != inverse_transpose.cols())
    {
      cache.orbital_set = orbital_set;
      cache.action.resize(inverse_transpose.rows(), inverse_transpose.cols());
      cache.positions.resize(inverse_transpose.rows());
      cache.valid.assign(inverse_transpose.rows(), false);
    }
    auto& action = cache.action;
    for (int row = 0; row < inverse_transpose.rows(); ++row)
    {
      const int electron = determinant.getFirstIndex() + row;
      bool changed       = !cache.valid[row];
      for (int d = 0; d < OHMMS_DIM; ++d)
        changed |= cache.positions[row][d] != pset.R[electron][d];
      if (!changed)
        continue;
      for (int orbital = 0; orbital < inverse_transpose.cols(); ++orbital)
      {
        const auto value = evaluateOrbitalAction(*orbital_set, orbital, pset.R[electron]);
#if defined(QMC_COMPLEX)
        action(row, orbital) = FullPrecValueType(value.real(), value.imag());
#else
        if (std::abs(value.imag()) > 1.0e-8)
          throw std::runtime_error("real B-spline MFRCoreAction produced a complex orbital action");
        action(row, orbital) = value.real();
#endif
      }
      cache.positions[row] = pset.R[electron];
      cache.valid[row]     = true;
      ++action_row_evaluations_;
    }
    const FullPrecValueType block_trace = oneBodyOperatorTrace(inverse_transpose, action);
    total_trace += std::complex<double>(std::real(block_trace), std::imag(block_trace));
  }

  imaginary_trace_ = total_trace.imag();
  value_           = total_trace.real();
#if defined(QMC_COMPLEX)
  return FullPrecValueType(total_trace.real(), total_trace.imag());
#else
  return total_trace.real();
#endif
}

MFRCoreAction::Return_t MFRCoreAction::evaluate(ParticleSet& pset) { return value_; }

bool MFRCoreAction::get(std::ostream& os) const
{
  os << "Direct bare-core orbital action from " << file_name_ << std::endl;
  return true;
}

void MFRCoreAction::addObservables(PropertySetType& plist, BufferType& collectables)
{
  observable_indices_[0] = plist.add("MFRCoreAction");
  observable_indices_[1] = plist.add("MFRCoreActionImag");
  observable_indices_[2] = plist.add("TotalEnergyMF");
  observable_indices_[3] = plist.add("IonIonMF");
}

void MFRCoreAction::registerObservables(std::vector<ObservableHelper>& h5desc, hdf_archive& file) const
{
  const char* names[] = {"MFRCoreAction", "MFRCoreActionImag", "TotalEnergyMF", "IonIonMF"};
  for (size_t i = 0; i < observable_indices_.size(); ++i)
  {
    if (std::any_of(h5desc.begin(), h5desc.end(), [&](const auto& h) {
          return h.lower_bound == observable_indices_[i];
        }))
      continue;
    h5desc.emplace_back(hdf_path{names[i]});
    h5desc.back().set_dimensions(std::vector<int>(1, 1), observable_indices_[i]);
  }
}

void MFRCoreAction::setObservables(PropertySetType& plist)
{
  plist[observable_indices_[0]] = value_;
  plist[observable_indices_[1]] = imaginary_trace_;
  plist[observable_indices_[2]] = total_energy_mf_;
  plist[observable_indices_[3]] = ion_ion_energy_mf_;
}

void MFRCoreAction::setParticlePropertyList(PropertySetType& plist, int offset)
{
  plist[observable_indices_[0] + offset] = value_;
  plist[observable_indices_[1] + offset] = imaginary_trace_;
  plist[observable_indices_[2] + offset] = total_energy_mf_;
  plist[observable_indices_[3] + offset] = ion_ion_energy_mf_;
}

std::unique_ptr<OperatorBase> MFRCoreAction::makeClone(ParticleSet& pset) const
{
  auto clone                = std::make_unique<MFRCoreAction>(pset, nullptr, payload_);
  clone->file_name_         = file_name_;
  clone->grid_dimensions_   = grid_dimensions_;
  clone->spin_channels_     = spin_channels_;
  clone->number_of_kpoints_ = number_of_kpoints_;
  clone->number_of_states_  = number_of_states_;
  clone->ion_ion_energy_mf_ = ion_ion_energy_mf_;
  clone->total_energy_mf_   = total_energy_mf_;
  clone->imaginary_trace_   = imaginary_trace_;
  clone->data_              = data_;
  clone->observable_indices_ = observable_indices_;
  clone->setName(getName());
  return clone;
}
} // namespace qmcplusplus
