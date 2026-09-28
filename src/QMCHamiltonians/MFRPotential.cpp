//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#include "MFRPotential.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "OhmmsData/AttributeSet.h"
#include "hdf/hdf_archive.h"

namespace qmcplusplus
{
namespace
{
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

std::string readStringAttribute(hid_t object, const char* name)
{
  const hid_t attribute = H5Aopen(object, name, H5P_DEFAULT);
  if (attribute < 0)
    throw std::runtime_error(std::string("missing HDF5 attribute '") + name + "'");
  const hid_t datatype = H5Aget_type(attribute);
  if (datatype < 0)
  {
    H5Aclose(attribute);
    throw std::runtime_error(std::string("cannot inspect HDF5 attribute '") + name + "'");
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
  int value;
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
  double value;
  const herr_t status = H5Aread(attribute, H5T_NATIVE_DOUBLE, &value);
  H5Aclose(attribute);
  if (status < 0)
    throw std::runtime_error(std::string("cannot read HDF5 attribute '") + name + "'");
  return value;
}

std::array<int, 3> readGridDimensions(hid_t object)
{
  const hid_t attribute = H5Aopen(object, "grid_dimensions", H5P_DEFAULT);
  if (attribute < 0)
    throw std::runtime_error("missing HDF5 attribute 'grid_dimensions'");
  const hid_t datatype = H5Aget_type(attribute);
  if (datatype < 0)
  {
    H5Aclose(attribute);
    throw std::runtime_error("cannot inspect HDF5 attribute \x27grid_dimensions\x27");
  }
  std::array<int, 3> dimensions;
  const herr_t status = H5Aread(attribute, datatype, dimensions.data());
  H5Tclose(datatype);
  H5Aclose(attribute);
  if (status < 0)
    throw std::runtime_error("cannot read HDF5 attribute 'grid_dimensions'");
  return dimensions;
}

std::vector<double> readDoubleDataset(hid_t file, const std::string& name, const std::vector<hsize_t>& expected_dims)
{
  const hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0)
    throw std::runtime_error("missing HDF5 dataset '" + name + "'");
  const hid_t dataspace = H5Dget_space(dataset);
  const int rank = H5Sget_simple_extent_ndims(dataspace);
  std::vector<hsize_t> dimensions(rank);
  H5Sget_simple_extent_dims(dataspace, dimensions.data(), nullptr);
  if (dimensions != expected_dims)
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
  return values;
}
} // namespace

MFRPotential::MFRPotential(ParticleSet& pset) : pset_(pset)
{
  setEnergyDomain(POTENTIAL);
  oneBodyQuantumDomain(pset);
}

MFRPotential::MFRPotential(ParticleSet& pset,
                           const Lattice& field_lattice,
                           std::vector<std::shared_ptr<UBspline_3d_d>> splines,
                           int spin_channels,
                           std::string file_name,
                           RealType total_energy_mf)
    : pset_(pset),
      field_lattice_(field_lattice),
      splines_(std::move(splines)),
      spin_channels_(spin_channels),
      file_name_(std::move(file_name)),
      total_energy_mf_(total_energy_mf)
{
  setEnergyDomain(POTENTIAL);
  oneBodyQuantumDomain(pset);
}

bool MFRPotential::put(xmlNodePtr cur)
{
  double scale = 1.0;
  std::string name("MFRPotential");
  OhmmsAttributeSet attributes;
  attributes.add(file_name_, "href");
  attributes.add(file_name_, "file_name");
  attributes.add(name, "name");
  attributes.add(scale, "scale");
  attributes.put(cur);

  if (file_name_.empty())
    throw std::runtime_error("MFRPotential requires an href attribute");
  if (name != "MFRPotential")
    throw std::runtime_error("MFRPotential observable name must be 'MFRPotential'");
  if (scale <= 0.0)
    throw std::runtime_error("MFRPotential scale must be positive");

  const hid_t file = H5Fopen(file_name_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
    throw std::runtime_error("MFRPotential failed to open HDF5 file '" + file_name_ + "'");

  try
  {
    const int schema_version = readIntAttribute(file, "schema_version");
    spin_channels_ = readIntAttribute(file, "spin_channels");
    const auto grid_dims = readGridDimensions(file);
    const std::string quantity = readStringAttribute(file, "quantity");
    const std::string units = readStringAttribute(file, "units");
    const std::string functional = readStringAttribute(file, "functional");
    const std::string grid_order = readStringAttribute(file, "grid_order");
    const std::string endpoint = readStringAttribute(file, "periodic_grid_endpoint");
    const std::string includes_ions = readStringAttribute(file, "potential_includes_ions");
    const std::string total_energy_units = readStringAttribute(file, "total_energy_mf_units");
    const std::string total_energy_definition = readStringAttribute(file, "total_energy_mf_definition");
    const double total_energy_rydberg = readDoubleAttribute(file, "total_energy_mf");

    if (schema_version != 2)
      throw std::runtime_error("unsupported MFR HDF5 schema version");
    if (quantity != "Vee_MF = v_H + v_xc")
      throw std::runtime_error("unexpected MFR HDF5 quantity '" + quantity + "'");
    if (units != "rydberg")
      throw std::runtime_error("unsupported MFR potential units '" + units + "'");
    if (functional.empty())
      throw std::runtime_error("MFR potential functional metadata is empty");
    if (grid_order != "Fortran: i fastest, then j, then k")
      throw std::runtime_error("unsupported MFR grid ordering '" + grid_order + "'");
    if (endpoint != "excluded")
      throw std::runtime_error("MFR grid must exclude the periodic endpoint");
    if (includes_ions != "false")
      throw std::runtime_error("MFR potential must not include ionic contributions");
    if (total_energy_units != "rydberg")
      throw std::runtime_error("unsupported mean-field total-energy units '" + total_energy_units + "'");
    if (total_energy_definition != "sum occupied KS eigenvalues + ion-ion energy")
      throw std::runtime_error("unexpected mean-field total-energy definition '" + total_energy_definition + "'");
    total_energy_mf_ = 0.5 * total_energy_rydberg;
    if (spin_channels_ != 1 && spin_channels_ != 2)
      throw std::runtime_error("MFR potential requires one or two spin channels");
    if (spin_channels_ == 2 && pset_.groups() != 2)
      throw std::runtime_error("two-channel MFR potential requires two electron groups");
    if (std::any_of(grid_dims.begin(), grid_dims.end(), [](int n) { return n < 4; }))
      throw std::runtime_error("MFR grid dimensions must be at least four for cubic splines");

    const std::vector<double> lattice_values = readDoubleDataset(file, "/mfr/lattice_vectors", {3, 3});
    Tensor<RealType, OHMMS_DIM> lattice;
    for (int i = 0; i < OHMMS_DIM; ++i)
      for (int j = 0; j < OHMMS_DIM; ++j)
        lattice(i, j) = lattice_values[i * OHMMS_DIM + j];
    field_lattice_.set(lattice);

    const Lattice& simulation_lattice = pset_.getLattice();
    for (int i = 0; i < OHMMS_DIM; ++i)
    {
      if (!simulation_lattice.getBoxBConds()[i])
        throw std::runtime_error("MFR potential requires a fully periodic simulation cell");
      const PosType reduced_vector = field_lattice_.toUnit(simulation_lattice.a(i));
      for (int j = 0; j < OHMMS_DIM; ++j)
        if (std::abs(reduced_vector[j] - std::round(reduced_vector[j])) > 1.0e-8)
          throw std::runtime_error("QMCPACK simulation cell is not commensurate with the MFR field lattice");
    }

    Ugrid grids[3];
    BCtype_d boundaries[3];
    for (int d = 0; d < 3; ++d)
    {
      grids[d].start = 0.0;
      grids[d].end = 1.0;
      grids[d].num = grid_dims[d];
      boundaries[d].lCode = boundaries[d].rCode = PERIODIC;
    }

    splines_.clear();
    splines_.reserve(spin_channels_);
    for (int spin = 0; spin < spin_channels_; ++spin)
    {
      const std::string dataset_name = "/mfr/v_hxc/spin_" + std::to_string(spin + 1);
      const std::vector<double> qe_values = readDoubleDataset(
          file, dataset_name,
          {static_cast<hsize_t>(grid_dims[2]), static_cast<hsize_t>(grid_dims[1]),
           static_cast<hsize_t>(grid_dims[0])});
      Array<double, 3> spline_values(grid_dims[0], grid_dims[1], grid_dims[2]);
      for (int i = 0; i < grid_dims[0]; ++i)
        for (int j = 0; j < grid_dims[1]; ++j)
          for (int k = 0; k < grid_dims[2]; ++k)
          {
            const size_t qe_index = (static_cast<size_t>(k) * grid_dims[1] + j) * grid_dims[0] + i;
            spline_values(i, j, k) = 0.5 * scale * qe_values[qe_index];
          }
      splines_.emplace_back(create_UBspline_3d_d(grids[0], grids[1], grids[2], boundaries[0], boundaries[1],
                                                 boundaries[2], spline_values.data()),
                            destroy_Bspline);
      if (!splines_.back())
        throw std::runtime_error("failed to construct MFR periodic B-spline");
    }

    app_log() << "  MFRPotential file: " << file_name_ << '\n'
              << "  MFRPotential grid: " << grid_dims[0] << ' ' << grid_dims[1] << ' ' << grid_dims[2] << '\n'
              << "  MFRPotential functional: " << functional << '\n'
              << "  MFRPotential spin channels: " << spin_channels_ << '\n'
              << "  MFRPotential conversion: " << 0.5 * scale << " Hartree/Rydberg\n"
              << "  TotalEnergyMF: " << total_energy_mf_ << " Hartree\n";
  }
  catch (...)
  {
    H5Fclose(file);
    throw;
  }
  H5Fclose(file);
  return true;
}

bool MFRPotential::get(std::ostream& os) const
{
  os << "Positive mean-field residual potential from " << file_name_ << std::endl;
  return true;
}

void MFRPotential::addObservables(PropertySetType& plist, BufferType& collectables)
{
  my_index_ = plist.size();
  plist.add(name_);
  plist.add("TotalEnergyMF");
}

void MFRPotential::registerObservables(std::vector<ObservableHelper>& h5desc, hdf_archive& file) const
{
  const std::vector<int> dimensions(1, 1);
  h5desc.emplace_back(hdf_path{name_});
  h5desc.back().set_dimensions(dimensions, my_index_);
  h5desc.emplace_back(hdf_path{"TotalEnergyMF"});
  h5desc.back().set_dimensions(dimensions, my_index_ + 1);
}

void MFRPotential::setObservables(PropertySetType& plist)
{
  plist[my_index_] = value_;
  plist[my_index_ + 1] = total_energy_mf_;
}

void MFRPotential::setParticlePropertyList(PropertySetType& plist, int offset)
{
  plist[my_index_ + offset] = value_;
  plist[my_index_ + offset + 1] = total_energy_mf_;
}

MFRPotential::RealType MFRPotential::evaluateOne(const ParticleSet& pset, int particle_index) const
{
  const int channel = spin_channels_ == 1 ? 0 : pset.getGroupID(particle_index);
  if (channel < 0 || channel >= spin_channels_)
    throw std::runtime_error("electron group is incompatible with MFR spin channels");
  const PosType reduced = field_lattice_.toUnit_floor(pset.R[particle_index]);
  double value = 0.0;
  eval_UBspline_3d_d(splines_[channel].get(), reduced[0], reduced[1], reduced[2], &value);
  return value;
}

MFRPotential::Return_t MFRPotential::evaluate(ParticleSet& pset)
{
  value_ = 0.0;
  for (int i = 0; i < pset.getTotalNum(); ++i)
  {
    const RealType particle_value = evaluateOne(pset, i);
#if !defined(REMOVE_TRACEMANAGER)
    if (streaming_particles_)
      (*v_sample_)(i) = particle_value;
#endif
    value_ += particle_value;
  }
  return value_;
}

std::unique_ptr<OperatorBase> MFRPotential::makeClone(ParticleSet& pset) const
{
  return std::unique_ptr<OperatorBase>(
      new MFRPotential(pset, field_lattice_, splines_, spin_channels_, file_name_, total_energy_mf_));
}

#if !defined(REMOVE_TRACEMANAGER)
void MFRPotential::contributeParticleQuantities() { request_.contribute_array(name_); }

void MFRPotential::checkoutParticleQuantities(TraceManager& tm)
{
  streaming_particles_ = request_.streaming_array(name_);
  if (streaming_particles_)
    v_sample_ = tm.checkout_real<1>(name_, pset_);
}

void MFRPotential::deleteParticleQuantities()
{
  if (streaming_particles_)
    delete v_sample_;
  v_sample_ = nullptr;
}
#endif

} // namespace qmcplusplus
