//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file psiformer_test_utils.h
 * @brief Deterministic, external-data-free PsiFormer HDF5 fixtures for native
 * and WaveFunctionComponent tests.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_TEST_UTILS_H
#define QMCPLUSPLUS_PSIFORMER_TEST_UTILS_H

#include <catch2/catch_test_macros.hpp>
#include <hdf5.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace qmcplusplus::testing::psiformer
{
using Shape = std::vector<std::size_t>;

inline constexpr std::uint64_t MIX_INCREMENT = 0x9E3779B97F4A7C15ULL;
inline constexpr std::uint64_t PARAMETER_RECIPE_SEED = 0xC0FFEE1234000000ULL;
inline constexpr std::uint64_t GEOMETRY_RECIPE_SEED  = 0x1234ABCDEF000000ULL;
inline constexpr std::uint32_t FIXTURE_RECIPE_VERSION = 1;

/// Provide deterministic uniform values without relying on library PRNG details.
class SplitMix64
{
public:
  explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

  std::uint64_t next()
  {
    std::uint64_t value = (state_ += MIX_INCREMENT);
    value               = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value               = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
  }

  double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }

  double symmetric() { return 2.0 * uniform() - 1.0; }

private:
  std::uint64_t state_;
};

/// Mix a counter for compact full-gradient projection checks.
inline std::uint64_t mix64(std::uint64_t value)
{
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

/// Return the scalar size of one row-major fixture tensor.
inline std::size_t product(const Shape& shape)
{
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
}

/// Describe one generated DeepQMC parameter leaf.
struct Leaf
{
  std::string module;
  std::string name;
  Shape shape;
};

/// Construct the complete four-block PsiFormer layout used by integration tests.
inline std::vector<Leaf> makeLayout(std::size_t electron_count,
                                    std::size_t nucleus_count,
                                    bool has_same_spin_pair = true)
{
  const std::string prefix = "neural_network_wave_function/~/";
  std::vector<Leaf> leaves{
      {prefix + "electronic_cusp_asymptotic", "anti_alpha", {}},
      {prefix + "exponential_envelopes", "pi_down", {16 * electron_count, nucleus_count}},
      {prefix + "exponential_envelopes", "pi_up", {16 * electron_count, nucleus_count}},
      {prefix + "exponential_envelopes", "zetas_down", {16 * electron_count, nucleus_count}},
      {prefix + "exponential_envelopes", "zetas_up", {16 * electron_count, nucleus_count}},
      {prefix + "omni_net/~/Backflow/~/mlp/linear_0", "w", {256, 16 * electron_count}},
      {prefix + "omni_net/~/Backflow_1/~/mlp/linear_0", "w", {256, 16 * electron_count}},
      {prefix + "omni_net/~/electron_gnn/~/electron_embedding/linear", "w", {4 * nucleus_count + 1, 256}},
  };
  if (has_same_spin_pair)
    leaves.insert(leaves.begin() + 1,
                  {prefix + "electronic_cusp_asymptotic", "same_alpha", {}});
  for (int layer = 0; layer < 4; ++layer)
  {
    const std::string layer_name = layer == 0 ? "electron_gnn_layer" : "electron_gnn_layer_" + std::to_string(layer);
    const std::string base = prefix + "omni_net/~/electron_gnn/~/" + layer_name +
        "/~/node_attention_electron_update_feature/";
    leaves.insert(leaves.end(), {{base + "mlp/linear_0", "b", {256}},
                                 {base + "mlp/linear_0", "w", {256, 256}},
                                 {base + "mlp/linear_1", "b", {256}},
                                 {base + "mlp/linear_1", "w", {256, 256}},
                                 {base + "multi_head_attention/key", "w", {256, 256}},
                                 {base + "multi_head_attention/linear", "w", {256, 256}},
                                 {base + "multi_head_attention/query", "w", {256, 256}},
                                 {base + "multi_head_attention/value", "w", {256, 256}}});
  }
  return leaves;
}

/// Fill every parameter region with deterministic values appropriate to its role.
inline std::vector<double> makeParameters(const std::string& system, const std::vector<Leaf>& leaves)
{
  const std::size_t electron_count = system == "lih" ? 4 : (system == "lih_pair" ? 8 : 2);
  SplitMix64 random(PARAMETER_RECIPE_SEED + electron_count);
  std::vector<double> values;
  for (const Leaf& leaf : leaves)
    for (std::size_t element = 0; element < product(leaf.shape); ++element)
    {
      double value;
      if (leaf.name.size() >= 5 && leaf.name.substr(leaf.name.size() - 5) == "alpha")
        value = 0.8 + 0.4 * random.uniform();
      else if (leaf.name.rfind("zetas", 0) == 0)
        value = 0.6 + 0.8 * random.uniform();
      else if (leaf.name.rfind("pi_", 0) == 0)
        value = 0.15 + 0.2 * random.symmetric();
      else if (leaf.name == "b")
        value = 0.02 * random.symmetric();
      else if (leaf.module.find("electron_embedding") != std::string::npos)
        value = 0.08 * random.symmetric();
      else if (leaf.module.find("Backflow") != std::string::npos)
        value = 0.06 * random.symmetric();
      else
        value = 0.04 * random.symmetric();
      values.push_back(value);
    }
  return values;
}

/// Hold one molecule's deterministic nuclei, charges, electrons, and spin split.
struct Geometry
{
  std::vector<double> nuclei;
  std::vector<double> charges;
  std::vector<double> electrons;
  std::size_t nup;
};

/// Construct all-electron LiH, a separated LiH pair, or two-electron pseudo-LiH geometry.
inline Geometry makeGeometry(const std::string& system)
{
  Geometry geometry;
  std::vector<std::size_t> centers;
  if (system == "lih")
  {
    geometry.nuclei = {0, 0, 0, 3.05, 0.08, -0.03};
    geometry.charges = {3, 1};
    centers          = {0, 1, 0, 1};
    geometry.nup     = 2;
  }
  else if (system == "lih_pair")
  {
    geometry.nuclei = {0, 0, 0, 3.05, 0.08, -0.03, 0.12, 14.7, 0.06, 3.17, 14.78, 0.03};
    geometry.charges = {3, 1, 3, 1};
    centers          = {0, 1, 2, 3, 0, 1, 2, 3};
    geometry.nup     = 4;
  }
  else if (system == "lih_pp")
  {
    geometry.nuclei = {0, 0, 0, 3.05, 0.08, -0.03};
    geometry.charges = {1, 1};
    centers          = {0, 1};
    geometry.nup     = 1;
  }
  else
    throw std::invalid_argument("Unknown generated PsiFormer test system: " + system);

  SplitMix64 random(GEOMETRY_RECIPE_SEED + centers.size());
  for (double& coordinate : geometry.nuclei)
    coordinate += 0.025 * random.symmetric();
  geometry.electrons.resize(3 * centers.size());
  for (std::size_t electron = 0; electron < centers.size(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      const double magnitude = 0.35 + 0.9 * random.uniform();
      const double sign      = (random.next() & 1) ? 1.0 : -1.0;
      geometry.electrons[3 * electron + dimension] =
          geometry.nuclei[3 * centers[electron] + dimension] + sign * magnitude;
    }
  return geometry;
}

/// Write one variable-length string dataset to a generated HDF5 fixture.
inline void writeStrings(hid_t file, const char* name, const std::vector<std::string>& strings)
{
  const hsize_t size = strings.size();
  const hid_t space  = H5Screate_simple(1, &size, nullptr);
  const hid_t type   = H5Tcopy(H5T_C_S1);
  H5Tset_size(type, H5T_VARIABLE);
  const hid_t dataset = H5Dcreate2(file, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  std::vector<const char*> pointers;
  for (const std::string& string : strings)
    pointers.push_back(string.c_str());
  REQUIRE(H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, pointers.data()) >= 0);
  H5Dclose(dataset);
  H5Tclose(type);
  H5Sclose(space);
}

/// Write one numeric dataset to a generated HDF5 fixture.
template<class T>
inline void writeNumeric(hid_t file,
                         const char* name,
                         hid_t type,
                         const std::vector<hsize_t>& shape,
                         const std::vector<T>& values)
{
  const hid_t space   = H5Screate_simple(shape.size(), shape.data(), nullptr);
  const hid_t dataset = H5Dcreate2(file, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
  H5Dclose(dataset);
  H5Sclose(space);
}

/// Write one signed integer attribute to a generated HDF5 fixture.
inline void writeIntAttribute(hid_t file, const char* name, std::int64_t value)
{
  const hid_t space = H5Screate(H5S_SCALAR);
  const hid_t attr  = H5Acreate2(file, name, H5T_NATIVE_LLONG, space, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(H5Awrite(attr, H5T_NATIVE_LLONG, &value) >= 0);
  H5Aclose(attr);
  H5Sclose(space);
}

/// Own and remove one pair of generated parameter/configuration files.
struct GeneratedFiles
{
  std::filesystem::path directory;
  std::filesystem::path parameters;
  std::filesystem::path configuration;

  GeneratedFiles() = default;
  GeneratedFiles(const GeneratedFiles&) = delete;
  GeneratedFiles& operator=(const GeneratedFiles&) = delete;
  GeneratedFiles(GeneratedFiles&& other) noexcept
      : directory(std::move(other.directory)),
        parameters(std::move(other.parameters)),
        configuration(std::move(other.configuration))
  {}

  ~GeneratedFiles()
  {
    std::error_code error;
    if (!directory.empty())
      std::filesystem::remove_all(directory, error);
  }
};

/// Generate a full-shape parameter file and matching physical-system configuration.
inline GeneratedFiles generateFiles(const std::string& system)
{
  static std::atomic<std::uint64_t> fixture_sequence{0};

  GeneratedFiles files;
  files.directory = std::filesystem::temp_directory_path() /
      ("qmcpack_psiformer_random_v" + std::to_string(FIXTURE_RECIPE_VERSION) + "_" + system + "_" +
       std::to_string(static_cast<long long>(getpid())) + "_" +
       std::to_string(fixture_sequence.fetch_add(1, std::memory_order_relaxed)));
  std::filesystem::create_directories(files.directory);
  files.parameters    = files.directory / "parameters.h5";
  files.configuration = files.directory / "configuration.h5";

  const Geometry geometry       = makeGeometry(system);
  const std::size_t electron_count = geometry.electrons.size() / 3;
  const std::size_t nucleus_count  = geometry.nuclei.size() / 3;
  const bool has_same_spin_pair = geometry.nup >= 2 || electron_count - geometry.nup >= 2;
  const auto leaves = makeLayout(electron_count, nucleus_count, has_same_spin_pair);
  const auto values             = makeParameters(system, leaves);
  std::vector<std::string> modules, names;
  std::vector<std::int64_t> ranks, shapes(2 * leaves.size(), 1), offsets{0};
  for (std::size_t parameter = 0; parameter < leaves.size(); ++parameter)
  {
    modules.push_back(leaves[parameter].module);
    names.push_back(leaves[parameter].name);
    ranks.push_back(leaves[parameter].shape.size());
    for (std::size_t axis = 0; axis < leaves[parameter].shape.size(); ++axis)
      shapes[2 * parameter + axis] = leaves[parameter].shape[axis];
    offsets.push_back(offsets.back() + product(leaves[parameter].shape));
  }

  hid_t file = H5Fcreate(files.parameters.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  writeNumeric(file, "/values", H5T_NATIVE_DOUBLE, {values.size()}, values);
  H5Gclose(H5Gcreate2(file, "/layout", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  writeStrings(file, "/layout/modules", modules);
  writeStrings(file, "/layout/names", names);
  writeNumeric(file, "/layout/ranks", H5T_NATIVE_LLONG, {ranks.size()}, ranks);
  writeNumeric(file, "/layout/shapes", H5T_NATIVE_LLONG, {leaves.size(), 2}, shapes);
  writeNumeric(file, "/layout/offsets", H5T_NATIVE_LLONG, {offsets.size()}, offsets);
  H5Fclose(file);

  file = H5Fcreate(files.configuration.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  writeNumeric(file, "/nuclear_positions", H5T_NATIVE_DOUBLE, {nucleus_count, 3}, geometry.nuclei);
  writeNumeric(file, "/nuclear_charges", H5T_NATIVE_DOUBLE, {nucleus_count}, geometry.charges);
  writeNumeric(file, "/electron_positions", H5T_NATIVE_DOUBLE, {1, electron_count, 3}, geometry.electrons);
  writeIntAttribute(file, "n_up", geometry.nup);
  writeIntAttribute(file, "n_down", electron_count - geometry.nup);
  H5Fclose(file);
  return files;
}

} // namespace qmcplusplus::testing::psiformer

#endif
