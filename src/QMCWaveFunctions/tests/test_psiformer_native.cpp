//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <string>
#include <unistd.h>

namespace
{
constexpr std::uint64_t MIX_INCREMENT = 0x9E3779B97F4A7C15ULL;

class SplitMix64
{
public:
  explicit SplitMix64(std::uint64_t seed) : state_(seed) {}
  std::uint64_t next()
  {
    std::uint64_t z = (state_ += MIX_INCREMENT);
    z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
  double symmetric() { return 2.0 * uniform() - 1.0; }

private:
  std::uint64_t state_;
};

std::uint64_t mix64(std::uint64_t z)
{
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

struct Leaf
{
  std::string module;
  std::string name;
  pf::Shape shape;
};

std::vector<Leaf> makeLayout(std::size_t nelec, std::size_t nnuc)
{
  const std::string p = "neural_network_wave_function/~/";
  std::vector<Leaf> leaves{
      {p + "electronic_cusp_asymptotic", "anti_alpha", {}},
      {p + "electronic_cusp_asymptotic", "same_alpha", {}},
      {p + "exponential_envelopes", "pi_down", {16 * nelec, nnuc}},
      {p + "exponential_envelopes", "pi_up", {16 * nelec, nnuc}},
      {p + "exponential_envelopes", "zetas_down", {16 * nelec, nnuc}},
      {p + "exponential_envelopes", "zetas_up", {16 * nelec, nnuc}},
      {p + "omni_net/~/Backflow/~/mlp/linear_0", "w", {256, 16 * nelec}},
      {p + "omni_net/~/Backflow_1/~/mlp/linear_0", "w", {256, 16 * nelec}},
      {p + "omni_net/~/electron_gnn/~/electron_embedding/linear", "w", {4 * nnuc + 1, 256}},
  };
  for (int layer = 0; layer < 4; ++layer)
  {
    const std::string layer_name = layer == 0 ? "electron_gnn_layer" : "electron_gnn_layer_" + std::to_string(layer);
    const std::string base = p + "omni_net/~/electron_gnn/~/" + layer_name +
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

std::vector<double> makeParameters(const std::string& system, const std::vector<Leaf>& leaves)
{
  const std::size_t nelec = system == "lih" ? 4 : 8;
  SplitMix64 rng(0xC0FFEE1234000000ULL + nelec);
  std::vector<double> values;
  for (const Leaf& leaf : leaves)
    for (std::size_t i = 0; i < pf::product(leaf.shape); ++i)
    {
      double value;
      if (leaf.name.size() >= 5 && leaf.name.substr(leaf.name.size() - 5) == "alpha")
        value = 0.8 + 0.4 * rng.uniform();
      else if (leaf.name.rfind("zetas", 0) == 0)
        value = 0.6 + 0.8 * rng.uniform();
      else if (leaf.name.rfind("pi_", 0) == 0)
        value = 0.15 + 0.2 * rng.symmetric();
      else if (leaf.name == "b")
        value = 0.02 * rng.symmetric();
      else if (leaf.module.find("electron_embedding") != std::string::npos)
        value = 0.08 * rng.symmetric();
      else if (leaf.module.find("Backflow") != std::string::npos)
        value = 0.06 * rng.symmetric();
      else
        value = 0.04 * rng.symmetric();
      values.push_back(value);
    }
  return values;
}

struct Geometry
{
  std::vector<double> nuclei;
  std::vector<double> charges;
  std::vector<double> electrons;
  std::size_t nup;
};

Geometry makeGeometry(const std::string& system)
{
  Geometry g;
  std::vector<std::size_t> centers;
  if (system == "lih")
  {
    g.nuclei = {0, 0, 0, 3.05, 0.08, -0.03};
    g.charges = {3, 1};
    centers   = {0, 1, 0, 1};
    g.nup     = 2;
  }
  else
  {
    g.nuclei = {0, 0, 0, 3.05, 0.08, -0.03, 0.12, 14.7, 0.06, 3.17, 14.78, 0.03};
    g.charges = {3, 1, 3, 1};
    centers   = {0, 1, 2, 3, 0, 1, 2, 3};
    g.nup     = 4;
  }
  SplitMix64 rng(0x1234ABCDEF000000ULL + centers.size());
  for (double& coordinate : g.nuclei)
    coordinate += 0.025 * rng.symmetric();
  g.electrons.resize(3 * centers.size());
  for (std::size_t i = 0; i < centers.size(); ++i)
    for (int d = 0; d < 3; ++d)
    {
      const double magnitude = 0.35 + 0.9 * rng.uniform();
      const double sign      = (rng.next() & 1) ? 1.0 : -1.0;
      g.electrons[3 * i + d] = g.nuclei[3 * centers[i] + d] + sign * magnitude;
    }
  return g;
}

void writeStrings(hid_t file, const char* name, const std::vector<std::string>& strings)
{
  const hsize_t size = strings.size();
  hid_t space        = H5Screate_simple(1, &size, nullptr);
  hid_t type         = H5Tcopy(H5T_C_S1);
  H5Tset_size(type, H5T_VARIABLE);
  hid_t dataset = H5Dcreate2(file, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  std::vector<const char*> pointers;
  for (const std::string& string : strings)
    pointers.push_back(string.c_str());
  REQUIRE(H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, pointers.data()) >= 0);
  H5Dclose(dataset);
  H5Tclose(type);
  H5Sclose(space);
}

template<class T>
void writeNumeric(hid_t file, const char* name, hid_t type, const std::vector<hsize_t>& shape, const std::vector<T>& values)
{
  hid_t space   = H5Screate_simple(shape.size(), shape.data(), nullptr);
  hid_t dataset = H5Dcreate2(file, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
  H5Dclose(dataset);
  H5Sclose(space);
}

void writeIntAttribute(hid_t file, const char* name, std::int64_t value)
{
  hid_t space = H5Screate(H5S_SCALAR);
  hid_t attr  = H5Acreate2(file, name, H5T_NATIVE_LLONG, space, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(H5Awrite(attr, H5T_NATIVE_LLONG, &value) >= 0);
  H5Aclose(attr);
  H5Sclose(space);
}

struct GeneratedFiles
{
  std::filesystem::path directory;
  std::filesystem::path parameters;
  std::filesystem::path configuration;
  GeneratedFiles() = default;
  GeneratedFiles(const GeneratedFiles&) = delete;
  GeneratedFiles& operator=(const GeneratedFiles&) = delete;
  GeneratedFiles(GeneratedFiles&& other) noexcept
      : directory(std::move(other.directory)), parameters(std::move(other.parameters)),
        configuration(std::move(other.configuration))
  {}
  ~GeneratedFiles()
  {
    std::error_code error;
    if (!directory.empty())
      std::filesystem::remove_all(directory, error);
  }
};

GeneratedFiles generateFiles(const std::string& system)
{
  GeneratedFiles files;
  files.directory = std::filesystem::temp_directory_path() /
      ("qmcpack_psiformer_random_" + system + "_" + std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(files.directory);
  std::filesystem::create_directories(files.directory);
  files.parameters  = files.directory / "parameters.h5";
  files.configuration = files.directory / "configuration.h5";

  const Geometry geometry = makeGeometry(system);
  const std::size_t nelec  = geometry.electrons.size() / 3;
  const std::size_t nnuc   = geometry.nuclei.size() / 3;
  const auto leaves        = makeLayout(nelec, nnuc);
  const auto values        = makeParameters(system, leaves);
  std::vector<std::string> modules, names;
  std::vector<std::int64_t> ranks, shapes(2 * leaves.size(), 1), offsets{0};
  for (std::size_t i = 0; i < leaves.size(); ++i)
  {
    modules.push_back(leaves[i].module);
    names.push_back(leaves[i].name);
    ranks.push_back(leaves[i].shape.size());
    for (std::size_t d = 0; d < leaves[i].shape.size(); ++d)
      shapes[2 * i + d] = leaves[i].shape[d];
    offsets.push_back(offsets.back() + pf::product(leaves[i].shape));
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
  writeNumeric(file, "/nuclear_positions", H5T_NATIVE_DOUBLE, {nnuc, 3}, geometry.nuclei);
  writeNumeric(file, "/nuclear_charges", H5T_NATIVE_DOUBLE, {nnuc}, geometry.charges);
  writeNumeric(file, "/electron_positions", H5T_NATIVE_DOUBLE, {1, nelec, 3}, geometry.electrons);
  writeIntAttribute(file, "n_up", geometry.nup);
  writeIntAttribute(file, "n_down", nelec - geometry.nup);
  H5Fclose(file);
  return files;
}

struct Golden
{
  double sign;
  double logabs;
  double value;
  std::vector<double> gradient;
  std::vector<double> lap_log;
  std::vector<double> lap_ratio;
  std::array<double, 3> potential;
  double local_energy;
  double parameter_sum;
  double parameter_abs_sum;
  double parameter_square_sum;
  std::array<double, 3> parameter_projections;
  std::vector<std::size_t> selected_indices;
  std::vector<double> selected_parameters;
  double local_energy_parameter_sum;
  double local_energy_parameter_abs_sum;
  double local_energy_parameter_square_sum;
  std::array<double, 3> local_energy_parameter_projections;
  std::vector<double> selected_local_energy_parameters;
};

// These compact references were generated independently with the DeepQMC/JAX
// PsiFormer in float64 from the exact PRNG recipe above. They must not be
// regenerated from the native implementation under test.

Golden lihGolden()
{
  return {-1, -23.077667467509848, -9.495030490017006e-11,
          {0.13821734732102417, 0.79435815447019209, -2.7709716589173525, 0.31363994259021311,
           0.79768832588162031, 1.6304777638742229, -3.4590352892338929, -1.3614672567480892,
           1.4157613451655826, -0.62679873678084308, 0.64331102535753593, -1.8005394170930145},
          {-12.948135315101172, -3.6943412044856432, -10.285466006833225, -5.1600794176296043},
          {-4.6197424679042385, -0.3012067871615427, 5.5374324029944031, -1.1114114933473269},
          {2.814284555459444, -8.2405022065141544, 0.9830451688891676}, -4.19570830945619,
          214.4094896606337, 177548.6601236603, 191890.4196573512,
          {135.75244123129642, 78.48531530433469, 258.6322930689181},
          {0, 1, 127, 128, 2047, 536832, 805249, 1610497},
          {-0.8695799883021037, -0.31347937666221071, -0.13890660257357479, -3.7570630618216381,
           -1.0659645930567652, -0.40454552610181804, -0.0016287744064348037, 0.066008319151184311},
          -227.83666730389504, 165018.97787525321, 164383.10827915635,
          {-275.71812381090683, 32.562659317325455, 253.87739681751484},
          {-0.027378914765929355, -0.15919463981076282, -3.6318764922253006, -2.3475447308091617,
           -0.4589721606959287, -0.19439318137311723, -0.02000558539414888, 0.19675543087012382}};
}

Golden pairGolden()
{
  return {-1, -35.19505925284793, -5.187761194606605e-16,
          {-0.51039359144009766, -0.64181835029801526, -0.56554578886861506, 0.11953688533544767,
           -0.40072217696523665, 0.62142613906272137, 0.56114724072042799, -0.38668360913347499,
           0.54183333491777408, 0.46267479367736408, -0.56435288767847713, -0.49317880081085291,
           -0.20015882506933289, -0.80800066604879506, 0.38701482713969687, -0.74100861969047793,
           0.27356357958271049, 0.0017890314830455352, 0.36347922677028238, 0.42084405027643595,
           0.50993799525183048, -0.70770775912668826, 1.0112844306971727, 0.6251745846552339},
          {-0.69870618499601422, -1.2805395515924511, -2.4548764662685221, -1.0182609103275793,
           -1.4358864737190578, -1.4079945406549548, -1.1268147848603813, -1.1579542485264198},
          {0.29356826727339702, -0.71950177521459202, -1.6968826640997654, -0.242473434222921,
           -0.59317736570465607, -0.78406053349161409, -0.55755116291232776, 0.75643548487104773},
          {5.0349316197469465, -18.591804342058118, 3.037106526369481}, -8.747944604190977,
          -116.89716684324083, 42031.44399687277, 9749.460123913907,
          {7.152529376205798, 74.79386681066573, -34.54328237149691},
          {0, 1, 127, 128, 2047, 548950, 823425, 1646849},
          {-2.3705368380082334, -0.56221196746345248, 0.014193225492927742, 0.02179686963981798,
           0.00012059685956394148, 0.0020275413443317375, 0.067400036242015071,
           0.027234831249253858},
          63.84149320878478, 17198.9773899136, 1648.8246231637158,
          {45.21822546194204, -0.5776053862966525, 8.188951223721432},
          {0.17763836352321852, 0.029026557833640224, -0.0015775310875903378, -0.005649916789854315,
           1.9426570517442113e-05, 0.03492990236430089, -0.0005105783615902042,
           -0.011817645765095516}};
}

double parameterProjection(const std::vector<double>& gradient, int stream)
{
  double result = 0;
  for (std::size_t i = 0; i < gradient.size(); ++i)
    result += gradient[i] * ((mix64(i + MIX_INCREMENT * (stream + 1)) & 1) ? 1.0 : -1.0);
  return result;
}

void checkClose(double actual, double expected, double relative = 2e-10, double absolute = 2e-10)
{
  CHECK(actual == Catch::Approx(expected).epsilon(relative).margin(absolute));
}

void validateCase(const std::string& system, const Golden& golden, bool finite_differences)
{
  GeneratedFiles files = generateFiles(system);
  pf::PsiFormer model(files.parameters, files.configuration);
  pf::Tensor electrons = model.cfg.configuration(0);
  const pf::Result result = model.evaluate(electrons, true);

  CHECK(result.sign == golden.sign);
  checkClose(result.logabs, golden.logabs);
  checkClose(result.value, golden.value, 2e-9, 1e-24);
  REQUIRE(result.gradient.size() == golden.gradient.size());
  REQUIRE(result.lap_log.size() == golden.lap_log.size());
  REQUIRE(result.lap_ratio.size() == golden.lap_ratio.size());
  for (std::size_t i = 0; i < result.gradient.size(); ++i)
    checkClose(result.gradient[i], golden.gradient[i], 2e-9, 2e-9);
  for (std::size_t i = 0; i < result.lap_log.size(); ++i)
  {
    checkClose(result.lap_log[i], golden.lap_log[i], 2e-8, 2e-8);
    checkClose(result.lap_ratio[i], golden.lap_ratio[i], 2e-8, 2e-8);
  }
  for (int i = 0; i < 3; ++i)
    checkClose(result.potential[i], golden.potential[i]);
  checkClose(result.local_energy, golden.local_energy, 2e-9, 2e-9);

  REQUIRE(result.param_gradient.size() == model.p.values.size());
  REQUIRE(result.local_energy_param_gradient.size() == model.p.values.size());
  const double parameter_sum = std::accumulate(result.param_gradient.begin(), result.param_gradient.end(), 0.0);
  double parameter_abs_sum = 0, parameter_square_sum = 0;
  for (double value : result.param_gradient)
  {
    parameter_abs_sum += std::abs(value);
    parameter_square_sum += value * value;
  }
  checkClose(parameter_sum, golden.parameter_sum, 2e-9, 2e-8);
  checkClose(parameter_abs_sum, golden.parameter_abs_sum, 2e-9, 2e-7);
  checkClose(parameter_square_sum, golden.parameter_square_sum, 2e-9, 2e-7);
  for (int stream = 0; stream < 3; ++stream)
    checkClose(parameterProjection(result.param_gradient, stream), golden.parameter_projections[stream], 2e-8, 2e-8);
  for (std::size_t i = 0; i < golden.selected_indices.size(); ++i)
    checkClose(result.param_gradient[golden.selected_indices[i]], golden.selected_parameters[i], 2e-8, 2e-9);

  const double local_parameter_sum =
      std::accumulate(result.local_energy_param_gradient.begin(), result.local_energy_param_gradient.end(), 0.0);
  double local_parameter_abs_sum = 0, local_parameter_square_sum = 0;
  for (double value : result.local_energy_param_gradient)
  {
    local_parameter_abs_sum += std::abs(value);
    local_parameter_square_sum += value * value;
  }
  checkClose(local_parameter_sum, golden.local_energy_parameter_sum, 2e-8, 2e-7);
  checkClose(local_parameter_abs_sum, golden.local_energy_parameter_abs_sum, 2e-8, 2e-6);
  checkClose(local_parameter_square_sum, golden.local_energy_parameter_square_sum, 2e-8, 2e-6);
  for (int stream = 0; stream < 3; ++stream)
    checkClose(parameterProjection(result.local_energy_param_gradient, stream),
               golden.local_energy_parameter_projections[stream], 2e-8, 2e-7);
  for (std::size_t i = 0; i < golden.selected_indices.size(); ++i)
    checkClose(result.local_energy_param_gradient[golden.selected_indices[i]],
               golden.selected_local_energy_parameters[i], 2e-8, 2e-8);

  if (!finite_differences)
    return;

  const double coordinate_step = 2e-5;
  for (std::size_t coordinate : {std::size_t{0}, std::size_t{5}, electrons.size() - 1})
  {
    pf::Tensor plus = electrons, minus = electrons;
    plus.x[coordinate] += coordinate_step;
    minus.x[coordinate] -= coordinate_step;
    const double finite_difference =
        (model.evaluate(plus, false).logabs - model.evaluate(minus, false).logabs) / (2 * coordinate_step);
    CHECK(finite_difference == Catch::Approx(result.gradient[coordinate]).epsilon(3e-5).margin(3e-5));
  }

  const double laplacian_step = 2e-4;
  double finite_laplacian = 0;
  for (int xyz = 0; xyz < 3; ++xyz)
  {
    pf::Tensor plus = electrons, minus = electrons;
    plus.x[xyz] += laplacian_step;
    minus.x[xyz] -= laplacian_step;
    finite_laplacian += (model.evaluate(plus, false).logabs - 2 * result.logabs +
                         model.evaluate(minus, false).logabs) /
        (laplacian_step * laplacian_step);
  }
  CHECK(finite_laplacian == Catch::Approx(result.lap_log[0]).epsilon(3e-4).margin(3e-4));

  SplitMix64 selection(0x5E1EC7EDULL);
  const double parameter_step = 2e-5;
  for (int sample = 0; sample < 3; ++sample)
  {
    const std::size_t flat_index = selection.next() % model.p.values.size();
    for (const pf::Layout& layout : model.p.layouts)
      if (layout.begin <= flat_index && flat_index < layout.end)
      {
        auto parameter = model.p.nodes[{layout.module, layout.name}];
        const std::size_t local_index = flat_index - layout.begin;
        const double original = parameter->value.x[local_index];
        parameter->value.x[local_index] = original + parameter_step;
        const pf::Result plus = model.evaluate(electrons, false);
        parameter->value.x[local_index] = original - parameter_step;
        const pf::Result minus = model.evaluate(electrons, false);
        parameter->value.x[local_index] = original;
        const double log_finite_difference = (plus.logabs - minus.logabs) / (2 * parameter_step);
        CHECK(log_finite_difference == Catch::Approx(result.param_gradient[flat_index]).epsilon(5e-5).margin(5e-5));
        const double energy_finite_difference = (plus.local_energy - minus.local_energy) / (2 * parameter_step);
        CHECK(energy_finite_difference ==
              Catch::Approx(result.local_energy_param_gradient[flat_index]).epsilon(2e-4).margin(2e-4));
        break;
      }
  }
}
} // namespace

TEST_CASE("PsiFormer randomized full-shape LiH high-level observables", "[wavefunction][psiformer]")
{
  validateCase("lih", lihGolden(), true);
}

TEST_CASE("PsiFormer randomized full-shape separated LiH pair high-level observables", "[wavefunction][psiformer]")
{
  validateCase("lih_pair", pairGolden(), false);
}
