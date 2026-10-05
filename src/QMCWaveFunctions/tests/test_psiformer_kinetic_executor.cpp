//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////
// Exact graph-free score/kinetic-response regression for the staged executor.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#if __has_include("QMCWaveFunctions/PsiFormer/PsiFormerKineticExecutor.h")
#include "QMCWaveFunctions/PsiFormer/PsiFormerKineticExecutor.h"
#else
#include "PsiFormerKineticExecutor.h"
#endif
#include "psiformer_test_utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace qmcplusplus::testing::psiformer;

std::atomic<bool> count_allocations{false};
std::atomic<std::size_t> allocation_count{0};

qmcplusplus::psiformer::PsiFormerExecutionPlan makePlan(const pf::PsiFormer& model)
{
  return qmcplusplus::psiformer::PsiFormerExecutionPlan::fromParameters(
      model.p, {model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet,
                model.dim, model.heads, 4});
}

void checkClose(double actual, double expected, double relative, double absolute)
{
  CHECK(actual == Catch::Approx(expected).epsilon(relative).margin(absolute));
}

double mixedKinetic(const pf::DirectKineticResultView& result,
                    const std::vector<double>& extra_gradient)
{
  double kinetic = -0.5 * std::accumulate(result.lap_log.begin(), result.lap_log.end(), 0.0);
  for (std::size_t coordinate = 0; coordinate < result.gradient.size(); ++coordinate)
  {
    const double total = result.gradient[coordinate] + extra_gradient[coordinate];
    kinetic -= 0.5 * total * total;
  }
  return kinetic;
}

/// Gradient of a fixed two-body Pade Jastrow log factor a*r/(1+b*r).
std::vector<double> jastrowGradient(const pf::Tensor& positions, std::size_t electrons)
{
  constexpr double a = 0.23;
  constexpr double b = 0.41;
  std::vector<double> gradient(3 * electrons, 0.0);
  for (std::size_t first = 0; first < electrons; ++first)
    for (std::size_t second = first + 1; second < electrons; ++second)
    {
      double displacement[3]{};
      double squared_radius = 0.0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        displacement[dimension] =
            positions.x[3 * first + dimension] - positions.x[3 * second + dimension];
        squared_radius += displacement[dimension] * displacement[dimension];
      }
      const double radius = std::sqrt(squared_radius);
      const double radial_first = a / ((1.0 + b * radius) * (1.0 + b * radius));
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double component = radial_first * displacement[dimension] / radius;
        gradient[3 * first + dimension] += component;
        gradient[3 * second + dimension] -= component;
      }
    }
  return gradient;
}

struct JaxKineticGolden
{
  double score_sum;
  double kinetic_sum;
  std::array<std::size_t, 8> indices;
  std::array<double, 8> score;
  std::array<double, 8> kinetic;
};

JaxKineticGolden lihGolden()
{
  return {214.4094896606337,
          -227.83666730389504,
          {0, 1, 127, 128, 2047, 536832, 805249, 1610497},
          {-0.8695799883021037, -0.31347937666221071, -0.13890660257357479,
           -3.7570630618216381, -1.0659645930567652, -0.40454552610181804,
           -0.0016287744064348037, 0.066008319151184311},
          {-0.027378914765929355, -0.15919463981076282, -3.6318764922253006,
           -2.3475447308091617, -0.4589721606959287, -0.19439318137311723,
           -0.02000558539414888, 0.19675543087012382}};
}

JaxKineticGolden pairGolden()
{
  return {-116.89716684324083,
          63.84149320878478,
          {0, 1, 127, 128, 2047, 548950, 823425, 1646849},
          {-2.3705368380082334, -0.56221196746345248, 0.014193225492927742,
           0.02179686963981798, 0.00012059685956394148, 0.0020275413443317375,
           0.067400036242015071, 0.027234831249253858},
          {0.17763836352321852, 0.029026557833640224, -0.0015775310875903378,
           -0.005649916789854315, 1.9426570517442113e-05, 0.03492990236430089,
           -0.0005105783615902042, -0.011817645765095516}};
}

void validateSystem(const std::string& system,
                    const JaxKineticGolden& golden,
                    bool finite_difference)
{
  GeneratedFiles files = generateFiles(system);
  pf::PsiFormer model(files.parameters, files.configuration);
  const pf::Tensor positions = model.cfg.configuration(0);
  const auto plan = makePlan(model);
  pf::DirectKineticExecutor executor(model, plan);
  auto workspace = executor.makeWorkspace();
  workspace->setPositions(
      pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  const std::size_t fingerprint = workspace->storageFingerprint();

  pf::EvaluationRequest request;
  request.spatial_derivatives    = pf::SpatialDerivativeRequest::FULL_VGL;
  request.parameter_derivatives  = pf::ParameterDerivativeRequest::LOG_AND_KINETIC;
  request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  const pf::Result native = model.evaluate(positions, request);
  const pf::DirectKineticResultView direct = executor.evaluate(*workspace);
  const double baseline_logabs = direct.logabs;
  const std::vector<double> baseline_gradient(direct.gradient.begin(), direct.gradient.end());
  const std::vector<double> baseline_lap_log(direct.lap_log.begin(), direct.lap_log.end());

  CHECK(direct.sign == native.sign);
  checkClose(direct.logabs, native.logabs, 3e-10, 3e-10);
  REQUIRE(direct.gradient.size() == native.gradient.size());
  REQUIRE(direct.lap_log.size() == native.lap_log.size());
  REQUIRE(direct.parameter_score.size() == native.param_gradient.size());
  REQUIRE(direct.kinetic_parameter_response.size() ==
          native.local_energy_param_gradient.size());
  for (std::size_t coordinate = 0; coordinate < direct.gradient.size(); ++coordinate)
    checkClose(direct.gradient[coordinate], native.gradient[coordinate], 2e-8, 2e-8);
  for (std::size_t electron = 0; electron < direct.lap_log.size(); ++electron)
    checkClose(direct.lap_log[electron], native.lap_log[electron], 3e-7, 3e-7);

  double maximum_score_error = 0.0;
  double maximum_kinetic_error = 0.0;
  for (std::size_t parameter = 0; parameter < direct.parameter_score.size(); ++parameter)
  {
    maximum_score_error = std::max(
        maximum_score_error,
        std::abs(direct.parameter_score[parameter] - native.param_gradient[parameter]));
    maximum_kinetic_error = std::max(
        maximum_kinetic_error,
        std::abs(direct.kinetic_parameter_response[parameter] -
                 native.local_energy_param_gradient[parameter]));
  }
  CHECK(maximum_score_error < 3e-8);
  CHECK(maximum_kinetic_error < 3e-6);
  if (std::getenv("PSIFORMER_KINETIC_DIAGNOSTICS"))
    std::cout << "stage6b_" << system
              << " max_score_error=" << maximum_score_error
              << " max_kinetic_error=" << maximum_kinetic_error
              << " workspace_bytes=" << workspace->vectorStorageBytes() << '\n';

  const double score_sum =
      std::accumulate(direct.parameter_score.begin(), direct.parameter_score.end(), 0.0);
  const double kinetic_sum = std::accumulate(direct.kinetic_parameter_response.begin(),
                                             direct.kinetic_parameter_response.end(), 0.0);
  checkClose(score_sum, golden.score_sum, 2e-8, 2e-7);
  checkClose(kinetic_sum, golden.kinetic_sum, 3e-7, 3e-6);
  for (std::size_t probe = 0; probe < golden.indices.size(); ++probe)
  {
    checkClose(direct.parameter_score[golden.indices[probe]], golden.score[probe], 2e-8, 2e-8);
    checkClose(direct.kinetic_parameter_response[golden.indices[probe]], golden.kinetic[probe],
               3e-7, 3e-7);
  }

  // Mixed PsiFormer+Jastrow drift.  The Jastrow is fixed with respect to PsiFormer
  // parameters, but its nonuniform pair gradient must enter the kinetic cross term.
  std::vector<double> extra_gradient = jastrowGradient(positions, model.ne);
  std::vector<double> total_gradient(direct.gradient.size());
  for (std::size_t coordinate = 0; coordinate < direct.gradient.size(); ++coordinate)
    total_gradient[coordinate] = direct.gradient[coordinate] + extra_gradient[coordinate];
  CHECK_THROWS_AS(executor.evaluate(*workspace, total_gradient.data(), total_gradient.size() - 1),
                  std::invalid_argument);
  request.total_log_gradient = &total_gradient;
  const pf::Result native_mixed = model.evaluate(positions, request);
  const pf::DirectKineticResultView direct_mixed =
      executor.evaluate(*workspace, total_gradient.data(), total_gradient.size());
  double maximum_mixed_error = 0.0;
  for (std::size_t parameter = 0; parameter < direct_mixed.parameter_score.size(); ++parameter)
    maximum_mixed_error = std::max(
        maximum_mixed_error,
        std::abs(direct_mixed.kinetic_parameter_response[parameter] -
                 native_mixed.local_energy_param_gradient[parameter]));
  CHECK(maximum_mixed_error < 3e-6);

  if (finite_difference)
  {
    const std::array<std::size_t, 3> probes{0, 127, golden.indices.back()};
    std::array<double, 3> baseline_score{};
    std::array<double, 3> baseline_kinetic{};
    for (std::size_t probe = 0; probe < probes.size(); ++probe)
    {
      baseline_score[probe] = direct_mixed.parameter_score[probes[probe]];
      baseline_kinetic[probe] = direct_mixed.kinetic_parameter_response[probes[probe]];
    }
    const double step = 2e-5;
    for (std::size_t probe = 0; probe < probes.size(); ++probe)
    {
      const std::size_t parameter = probes[probe];
      const double original = model.p.flat_values()[parameter];
      model.p.set_flat_value(parameter, original + step);
      const pf::DirectKineticResultView plus = executor.evaluate(*workspace);
      const double plus_log = plus.logabs;
      const double plus_kinetic = mixedKinetic(plus, extra_gradient);
      model.p.set_flat_value(parameter, original - step);
      const pf::DirectKineticResultView minus = executor.evaluate(*workspace);
      const double minus_log = minus.logabs;
      const double minus_kinetic = mixedKinetic(minus, extra_gradient);
      model.p.set_flat_value(parameter, original);
      const double score_fd = (plus_log - minus_log) / (2.0 * step);
      const double kinetic_fd = (plus_kinetic - minus_kinetic) / (2.0 * step);
      checkClose(baseline_score[probe], score_fd, 6e-5, 6e-5);
      checkClose(baseline_kinetic[probe], kinetic_fd, 3e-4, 3e-4);
    }

    // Independent coordinate finite differences of the same trace-jet forward tape.
    const double gradient_step = 2e-5;
    const std::size_t coordinate = 5;
    workspace->setPosition(coordinate / 3, coordinate % 3,
                           positions.x[coordinate] + gradient_step);
    const double coordinate_plus = executor.evaluate(*workspace).logabs;
    workspace->setPosition(coordinate / 3, coordinate % 3,
                           positions.x[coordinate] - gradient_step);
    const double coordinate_minus = executor.evaluate(*workspace).logabs;
    workspace->setPosition(coordinate / 3, coordinate % 3, positions.x[coordinate]);
    checkClose(baseline_gradient[coordinate],
               (coordinate_plus - coordinate_minus) / (2.0 * gradient_step), 5e-5, 5e-5);

    const double laplacian_step = 2e-4;
    double finite_laplacian = 0.0;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      workspace->setPosition(0, dimension, positions.x[dimension] + laplacian_step);
      const double plus = executor.evaluate(*workspace).logabs;
      workspace->setPosition(0, dimension, positions.x[dimension] - laplacian_step);
      const double minus = executor.evaluate(*workspace).logabs;
      workspace->setPosition(0, dimension, positions.x[dimension]);
      finite_laplacian +=
          (plus - 2.0 * baseline_logabs + minus) / (laplacian_step * laplacian_step);
    }
    checkClose(baseline_lap_log[0], finite_laplacian, 7e-4, 7e-4);
  }

  if (system == "lih" && std::getenv("PSIFORMER_KINETIC_BENCHMARK"))
  {
    constexpr int direct_repeats = 5;
    const auto direct_begin = std::chrono::steady_clock::now();
    volatile double timing_sink = 0.0;
    for (int repetition = 0; repetition < direct_repeats; ++repetition)
      timing_sink += executor.evaluate(*workspace).kinetic_parameter_response[127];
    const auto direct_end = std::chrono::steady_clock::now();
    request.total_log_gradient = nullptr;
    const auto native_begin = std::chrono::steady_clock::now();
    timing_sink += model.evaluate(positions, request).local_energy_param_gradient[127];
    const auto native_end = std::chrono::steady_clock::now();
    const double direct_ms = std::chrono::duration<double, std::milli>(
        direct_end - direct_begin).count() / direct_repeats;
    const double native_ms = std::chrono::duration<double, std::milli>(
        native_end - native_begin).count();
    std::cout << "stage6b_lih workspace_bytes=" << workspace->vectorStorageBytes()
              << " direct_ms=" << direct_ms << " native_ms=" << native_ms
              << " speedup=" << native_ms / direct_ms << " sink=" << timing_sink << '\n';
  }

  // All explicit storage is fixed after construction.  The fingerprint also covers
  // determinant LU/inverse factors and both parameter outputs.
  CHECK(workspace->storageFingerprint() == fingerprint);
  const double* score_address = workspace->scoreData();
  const double* kinetic_address = workspace->kineticData();
  count_allocations.store(true, std::memory_order_relaxed);
  allocation_count.store(0, std::memory_order_relaxed);
  volatile double sink = executor.evaluate(*workspace).kinetic_parameter_response[0];
  count_allocations.store(false, std::memory_order_relaxed);
  CHECK(allocation_count.load(std::memory_order_relaxed) == 0);
  CHECK(workspace->scoreData() == score_address);
  CHECK(workspace->kineticData() == kinetic_address);
  CHECK(std::isfinite(sink));
}

/// Compare the full score and kinetic response for the canonical 1+1 layout.
void validatePairFreePseudoLiH()
{
  GeneratedFiles files = generateFiles("lih_pp");
  pf::PsiFormer model(files.parameters, files.configuration);
  const pf::Tensor positions = model.cfg.configuration(0);
  const auto plan = makePlan(model);
  CHECK_FALSE(plan.hasParameter(qmcplusplus::psiformer::ParameterRole::CUSP_SAME_ALPHA));

  pf::DirectKineticExecutor executor(model, plan);
  auto workspace = executor.makeWorkspace();
  workspace->setPositions(
      pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));

  pf::EvaluationRequest request;
  request.spatial_derivatives    = pf::SpatialDerivativeRequest::FULL_VGL;
  request.parameter_derivatives  = pf::ParameterDerivativeRequest::LOG_AND_KINETIC;
  request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  const pf::Result native = model.evaluate(positions, request);
  const pf::DirectKineticResultView direct = executor.evaluate(*workspace);
  REQUIRE(direct.parameter_score.size() == native.param_gradient.size());
  REQUIRE(direct.kinetic_parameter_response.size() ==
          native.local_energy_param_gradient.size());

  double maximum_score_error   = 0.0;
  double maximum_kinetic_error = 0.0;
  for (std::size_t parameter = 0; parameter < direct.parameter_score.size(); ++parameter)
  {
    maximum_score_error = std::max(
        maximum_score_error,
        std::abs(direct.parameter_score[parameter] - native.param_gradient[parameter]));
    maximum_kinetic_error = std::max(
        maximum_kinetic_error,
        std::abs(direct.kinetic_parameter_response[parameter] -
                 native.local_energy_param_gradient[parameter]));
  }
  CHECK(maximum_score_error < 3e-8);
  CHECK(maximum_kinetic_error < 3e-6);
}

} // namespace

void* operator new(std::size_t size)
{
  if (count_allocations.load(std::memory_order_relaxed))
    allocation_count.fetch_add(1, std::memory_order_relaxed);
  if (void* pointer = std::malloc(size))
    return pointer;
  throw std::bad_alloc();
}

void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

TEST_CASE("PsiFormer exact graph-free score and kinetic response LiH",
          "[wavefunction][psiformer][kinetic]")
{
  validateSystem("lih", lihGolden(), true);
}

TEST_CASE("PsiFormer exact graph-free score and kinetic response separated pair",
          "[wavefunction][psiformer][kinetic]")
{
  validateSystem("lih_pair", pairGolden(), false);
}

TEST_CASE("PsiFormer exact score and kinetic response omit unused pseudo-LiH same cusp",
          "[wavefunction][psiformer][kinetic][ecp]")
{
  validatePairFreePseudoLiH();
}
