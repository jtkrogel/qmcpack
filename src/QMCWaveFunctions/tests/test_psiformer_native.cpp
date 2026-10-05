//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "psiformer_test_utils.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <numeric>
#include <string>
#include <unistd.h>

namespace
{
using namespace qmcplusplus::testing::psiformer;

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

void checkVectorClose(const std::vector<double>& actual,
                      const std::vector<double>& expected,
                      double relative,
                      double absolute)
{
  REQUIRE(actual.size() == expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index)
    checkClose(actual[index], expected[index], relative, absolute);
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

using namespace qmcplusplus::testing::psiformer;

TEST_CASE("PsiFormer randomized full-shape LiH high-level observables", "[wavefunction][psiformer]")
{
  validateCase("lih", lihGolden(), true);
}

TEST_CASE("PsiFormer randomized full-shape separated LiH pair high-level observables", "[wavefunction][psiformer]")
{
  validateCase("lih_pair", pairGolden(), false);
}

TEST_CASE("PsiFormer synchronized flat parameter mutation and export", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  pf::PsiFormer model(files.parameters, files.configuration);
  const pf::Tensor electrons = model.cfg.configuration(0);
  const std::vector<double> original_values = model.p.flat_values();
  const std::string original_fingerprint     = model.p.layout_fingerprint();
  CHECK(original_fingerprint.size() == 16);

  REQUIRE(model.p.size() == 1610498);
  CHECK(model.p.version() == 0);
  const pf::Layout& first_layout = model.p.layout_for_flat_index(0);
  CHECK(first_layout.begin == 0);
  CHECK(first_layout.end == 1);
  CHECK_THROWS_AS(model.p.layout_for_flat_index(model.p.size()), std::out_of_range);

  const std::size_t scalar_index = 127;
  const pf::Layout& scalar_layout = model.p.layout_for_flat_index(scalar_index);
  const double changed_value = original_values[scalar_index] + 0.03125;
  model.p.set_flat_value(scalar_index, changed_value);
  CHECK(model.p.version() == 1);
  CHECK(model.p.flat_values()[scalar_index] == changed_value);
  CHECK(model.p.nodes.at({scalar_layout.module, scalar_layout.name})
            ->value.x[scalar_index - scalar_layout.begin] == changed_value);

  std::vector<double> replacement = original_values;
  replacement[0] += 0.015625;
  replacement[scalar_index] -= 0.0078125;
  model.p.set_flat_values(replacement);
  CHECK(model.p.version() == 2);
  CHECK(model.p.flat_values() == replacement);
  CHECK(model.p.layout_fingerprint() == original_fingerprint);

  CHECK_THROWS_AS(model.p.set_flat_values(std::vector<double>{1.0}), std::invalid_argument);
  CHECK_THROWS_AS(model.p.set_flat_values(std::vector<std::size_t>{0, 0}, std::vector<double>{1.0, 2.0}),
                  std::invalid_argument);
  CHECK_THROWS_AS(model.p.set_flat_value(0, std::numeric_limits<double>::infinity()), std::invalid_argument);
  CHECK(model.p.version() == 2);

  const pf::Result changed_result = model.evaluate(electrons, false);
  const std::filesystem::path export_path = files.directory / "parameters_exported.h5";
  model.p.write(export_path.string());
  pf::PsiFormer reloaded(export_path.string(), files.configuration.string());
  CHECK(reloaded.p.flat_values() == model.p.flat_values());
  CHECK(reloaded.p.layout_fingerprint() == original_fingerprint);
  const pf::Result reloaded_result = reloaded.evaluate(electrons, false);
  checkClose(reloaded_result.logabs, changed_result.logabs);
  checkClose(reloaded_result.local_energy, changed_result.local_energy, 2e-9, 2e-9);
}

TEST_CASE("PsiFormer explicit parameter derivative requests and total-gradient seed", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  pf::PsiFormer model(files.parameters, files.configuration);
  const pf::Tensor electrons = model.cfg.configuration(0);

  const pf::Result no_derivatives =
      model.evaluate(electrons, pf::EvaluationRequest{pf::ParameterDerivativeRequest::NONE, nullptr});
  CHECK(no_derivatives.param_gradient.empty());
  CHECK(no_derivatives.local_energy_param_gradient.empty());

  const pf::Result log_only =
      model.evaluate(electrons, pf::EvaluationRequest{pf::ParameterDerivativeRequest::LOG_ONLY, nullptr});
  REQUIRE(log_only.param_gradient.size() == model.p.size());
  CHECK(log_only.local_energy_param_gradient.empty());

  const pf::Result standalone = model.evaluate(
      electrons, pf::EvaluationRequest{pf::ParameterDerivativeRequest::LOG_AND_KINETIC, nullptr});
  checkVectorClose(log_only.param_gradient, standalone.param_gradient, 2e-10, 2e-10);
  REQUIRE(standalone.local_energy_param_gradient.size() == model.p.size());

  std::vector<double> extra_gradient(standalone.gradient.size());
  std::vector<double> total_gradient(standalone.gradient.size());
  for (std::size_t coordinate = 0; coordinate < extra_gradient.size(); ++coordinate)
  {
    extra_gradient[coordinate] = 0.01 * (1 + coordinate % 3);
    total_gradient[coordinate] = standalone.gradient[coordinate] + extra_gradient[coordinate];
  }
  const pf::Result composed = model.evaluate(
      electrons, pf::EvaluationRequest{pf::ParameterDerivativeRequest::LOG_AND_KINETIC, &total_gradient});

  auto composed_local_energy = [&]() {
    const pf::Result result = model.evaluate(electrons, false);
    double cross_term = 0.0;
    double extra_norm = 0.0;
    for (std::size_t coordinate = 0; coordinate < extra_gradient.size(); ++coordinate)
    {
      cross_term += result.gradient[coordinate] * extra_gradient[coordinate];
      extra_norm += extra_gradient[coordinate] * extra_gradient[coordinate];
    }
    return result.local_energy - cross_term - 0.5 * extra_norm;
  };

  const std::size_t flat_index = 0;
  const double original_value = model.p.flat_values()[flat_index];
  const double parameter_step = 2e-5;
  model.p.set_flat_value(flat_index, original_value + parameter_step);
  const double plus_energy = composed_local_energy();
  model.p.set_flat_value(flat_index, original_value - parameter_step);
  const double minus_energy = composed_local_energy();
  model.p.set_flat_value(flat_index, original_value);
  const double finite_difference = (plus_energy - minus_energy) / (2 * parameter_step);
  CHECK(finite_difference ==
        Catch::Approx(composed.local_energy_param_gradient[flat_index]).epsilon(2e-4).margin(2e-4));

  const std::vector<double> wrong_total_gradient(1, 0.0);
  CHECK_THROWS_AS(model.evaluate(
                      electrons,
                      pf::EvaluationRequest{pf::ParameterDerivativeRequest::LOG_AND_KINETIC, &wrong_total_gradient}),
                  std::invalid_argument);
}

TEST_CASE("PsiFormer observable requests return only requested products", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  pf::PsiFormer model(files.parameters, files.configuration);
  const pf::Tensor electrons = model.cfg.configuration(0);
  const pf::Result oracle    = model.evaluate(electrons, true);

  pf::EvaluationRequest value_request;
  value_request.spatial_derivatives    = pf::SpatialDerivativeRequest::NONE;
  value_request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  const pf::Result value_only          = model.evaluate(electrons, value_request);
  checkClose(value_only.sign, oracle.sign);
  checkClose(value_only.logabs, oracle.logabs);
  checkClose(value_only.value, oracle.value, 2e-9, 1e-24);
  CHECK(value_only.gradient.empty());
  CHECK(value_only.active_gradient.empty());
  CHECK(value_only.lap_log.empty());
  CHECK(value_only.lap_ratio.empty());
  CHECK(value_only.potential.empty());
  CHECK_FALSE(value_only.has_local_energy);
  CHECK(value_only.param_gradient.empty());
  CHECK(value_only.local_energy_param_gradient.empty());

  pf::EvaluationRequest spatial_request;
  spatial_request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  const pf::Result spatial_only          = model.evaluate(electrons, spatial_request);
  checkVectorClose(spatial_only.gradient, oracle.gradient, 2e-9, 2e-9);
  checkVectorClose(spatial_only.lap_log, oracle.lap_log, 2e-8, 2e-8);
  checkVectorClose(spatial_only.lap_ratio, oracle.lap_ratio, 2e-8, 2e-8);
  CHECK(spatial_only.active_gradient.empty());
  CHECK(spatial_only.potential.empty());
  CHECK_FALSE(spatial_only.has_local_energy);

  for (std::size_t electron = 0; electron < model.ne; ++electron)
  {
    pf::EvaluationRequest active_request;
    active_request.spatial_derivatives    = pf::SpatialDerivativeRequest::ACTIVE_ELECTRON_GRADIENT;
    active_request.active_electron        = electron;
    active_request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
    const pf::Result active               = model.evaluate(electrons, active_request);
    REQUIRE(active.active_gradient.size() == 3);
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      checkClose(active.active_gradient[dimension], oracle.gradient[3 * electron + dimension], 2e-9, 2e-9);
    CHECK(active.gradient.empty());
    CHECK(active.lap_log.empty());
    CHECK(active.lap_ratio.empty());
  }

  pf::EvaluationRequest score_request;
  score_request.parameter_derivatives    = pf::ParameterDerivativeRequest::LOG_ONLY;
  score_request.spatial_derivatives      = pf::SpatialDerivativeRequest::NONE;
  score_request.validation_hamiltonian   = pf::ValidationHamiltonianRequest::NONE;
  const pf::Result score_only            = model.evaluate(electrons, score_request);
  checkVectorClose(score_only.param_gradient, oracle.param_gradient, 2e-8, 2e-9);
  CHECK(score_only.gradient.empty());
  CHECK(score_only.lap_log.empty());
  CHECK(score_only.potential.empty());
  CHECK_FALSE(score_only.has_local_energy);
  CHECK(score_only.local_energy_param_gradient.empty());

  pf::EvaluationRequest kinetic_request;
  kinetic_request.parameter_derivatives  = pf::ParameterDerivativeRequest::LOG_AND_KINETIC;
  kinetic_request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  const pf::Result kinetic                = model.evaluate(electrons, kinetic_request);
  checkVectorClose(kinetic.gradient, oracle.gradient, 2e-9, 2e-9);
  checkVectorClose(kinetic.param_gradient, oracle.param_gradient, 2e-8, 2e-9);
  checkVectorClose(kinetic.local_energy_param_gradient, oracle.local_energy_param_gradient, 2e-8, 2e-8);
  CHECK(kinetic.potential.empty());
  CHECK_FALSE(kinetic.has_local_energy);
}

TEST_CASE("PsiFormer rejects incompatible observable requests", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  pf::PsiFormer model(files.parameters, files.configuration);
  const pf::Tensor electrons = model.cfg.configuration(0);

  pf::EvaluationRequest invalid_active;
  invalid_active.spatial_derivatives    = pf::SpatialDerivativeRequest::ACTIVE_ELECTRON_GRADIENT;
  invalid_active.active_electron        = model.ne;
  invalid_active.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  CHECK_THROWS_AS(model.evaluate(electrons, invalid_active), std::out_of_range);

  pf::EvaluationRequest invalid_kinetic;
  invalid_kinetic.parameter_derivatives  = pf::ParameterDerivativeRequest::LOG_AND_KINETIC;
  invalid_kinetic.spatial_derivatives    = pf::SpatialDerivativeRequest::NONE;
  invalid_kinetic.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  CHECK_THROWS_AS(model.evaluate(electrons, invalid_kinetic), std::invalid_argument);

  std::vector<double> total_gradient(3 * model.ne, 0.0);
  pf::EvaluationRequest invalid_seed;
  invalid_seed.total_log_gradient      = &total_gradient;
  invalid_seed.validation_hamiltonian  = pf::ValidationHamiltonianRequest::NONE;
  CHECK_THROWS_AS(model.evaluate(electrons, invalid_seed), std::invalid_argument);
}

TEST_CASE("PsiFormer rejects malformed or non-finite HDF5 exports", "[wavefunction][psiformer]")
{
  auto overwrite_scalar = [](const std::filesystem::path& path,
                             const char* dataset_name,
                             hsize_t index,
                             hid_t type,
                             const void* value) {
    const hid_t file    = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    const hid_t dataset = H5Dopen2(file, dataset_name, H5P_DEFAULT);
    const hid_t space   = H5Dget_space(dataset);
    const hsize_t count = 1;
    REQUIRE(H5Sselect_hyperslab(space, H5S_SELECT_SET, &index, nullptr, &count, nullptr) >= 0);
    const hid_t memory_space = H5Screate_simple(1, &count, nullptr);
    REQUIRE(H5Dwrite(dataset, type, memory_space, space, H5P_DEFAULT, value) >= 0);
    H5Sclose(memory_space);
    H5Sclose(space);
    H5Dclose(dataset);
    H5Fclose(file);
  };

  SECTION("non-finite parameter")
  {
    GeneratedFiles files = generateFiles("lih");
    const double nonfinite = std::numeric_limits<double>::quiet_NaN();
    overwrite_scalar(files.parameters, "/values", 0, H5T_NATIVE_DOUBLE, &nonfinite);
    CHECK_THROWS_WITH(pf::PsiFormer(files.parameters, files.configuration),
                      Catch::Matchers::ContainsSubstring("non-finite"));
  }

  SECTION("inconsistent final layout offset")
  {
    GeneratedFiles files = generateFiles("lih");
    const hsize_t final_offset_index = makeLayout(4, 2).size();
    const std::int64_t invalid_offset = 7;
    overwrite_scalar(files.parameters, "/layout/offsets", final_offset_index, H5T_NATIVE_LLONG, &invalid_offset);
    CHECK_THROWS_WITH(pf::PsiFormer(files.parameters, files.configuration),
                      Catch::Matchers::ContainsSubstring("layout metadata"));
  }
}
