//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_score_executor.cpp
 * @brief External-data-free tests for the direct PsiFormer parameter-score pass.
 */

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"
#include "psiformer_test_utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>

namespace
{
std::atomic<bool> count_allocations{false};
std::atomic<std::size_t> allocation_count{0};
}

/// Count ordinary heap allocations only inside a warmed direct-score audit window.
void* operator new(std::size_t bytes)
{
  if (count_allocations.load(std::memory_order_relaxed))
    allocation_count.fetch_add(1, std::memory_order_relaxed);
  if (void* storage = std::malloc(bytes))
    return storage;
  throw std::bad_alloc();
}

/// Apply the same scoped accounting to array allocations.
void* operator new[](std::size_t bytes)
{
  return ::operator new(bytes);
}

/// Release storage allocated by the test-local ordinary new override.
void operator delete(void* storage) noexcept { std::free(storage); }

/// Release storage allocated by the test-local array new override.
void operator delete[](void* storage) noexcept { std::free(storage); }

/// Support sized deallocation selected by optimized builds.
void operator delete(void* storage, std::size_t) noexcept { std::free(storage); }

/// Support sized array deallocation selected by optimized builds.
void operator delete[](void* storage, std::size_t) noexcept { std::free(storage); }

namespace
{
using namespace qmcplusplus::testing::psiformer;

/// Hold compact independent JAX checks for one deterministic generated fixture.
struct ScoreGolden
{
  double parameter_sum;
  double parameter_abs_sum;
  double parameter_square_sum;
  std::array<double, 3> projections;
  std::vector<std::size_t> selected_indices;
  std::vector<double> selected_values;
};

/// Return independent DeepQMC/JAX score aggregates for the generated LiH fixture.
ScoreGolden lihScoreGolden()
{
  return {214.4094896606337,
          177548.6601236603,
          191890.4196573512,
          {135.75244123129642, 78.48531530433469, 258.6322930689181},
          {0, 1, 127, 128, 2047, 536832, 805249, 1610497},
          {-0.8695799883021037, -0.31347937666221071, -0.13890660257357479,
           -3.7570630618216381, -1.0659645930567652, -0.40454552610181804,
           -0.0016287744064348037, 0.066008319151184311}};
}

/// Return independent DeepQMC/JAX score aggregates for the separated LiH-pair fixture.
ScoreGolden pairScoreGolden()
{
  return {-116.89716684324083,
          42031.44399687277,
          9749.460123913907,
          {7.152529376205798, 74.79386681066573, -34.54328237149691},
          {0, 1, 127, 128, 2047, 548950, 823425, 1646849},
          {-2.3705368380082334, -0.56221196746345248, 0.014193225492927742,
           0.02179686963981798, 0.00012059685956394148, 0.0020275413443317375,
           0.067400036242015071, 0.027234831249253858}};
}

/// Form a deterministic dense projection that detects broad score-ordering mistakes.
double parameterProjection(pf::DirectParameterScoreView gradient, int stream)
{
  double result = 0.0;
  for (std::size_t index = 0; index < gradient.size; ++index)
    result += gradient[index] *
        ((mix64(index + MIX_INCREMENT * static_cast<std::size_t>(stream + 1)) & 1) ? 1.0 : -1.0);
  return result;
}

/// Check one scalar with a mixed tolerance appropriate to imported float64 fixtures.
void checkClose(double actual, double expected, double relative = 2.0e-10, double absolute = 2.0e-10)
{
  CHECK(actual == Catch::Approx(expected).epsilon(relative).margin(absolute));
}

/// Validate full canonical output, independent aggregates, mutation, and allocations.
void validateScore(const std::string& system, const ScoreGolden& golden)
{
  GeneratedFiles files = generateFiles(system);
  pf::PsiFormer model(files.parameters, files.configuration);
  const qmcplusplus::psiformer::ModelShape shape{
      model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet, model.dim, model.heads, 4};
  const auto plan = qmcplusplus::psiformer::PsiFormerExecutionPlan::fromParameters(model.p, shape);
  pf::DirectScoreExecutor executor(model, plan);
  std::unique_ptr<pf::DirectScoreWorkspace> workspace = executor.makeWorkspace();
  CHECK(workspace->geometryStorageBytes() > 0);
  CHECK(workspace->vectorStorageBytes() == workspace->requiredStorageBytes());
  const pf::Tensor electrons = model.cfg.configuration(0);
  workspace->setPositions(pf::GeometryPositionView::interleaved(electrons.x.data(), model.ne));

  pf::EvaluationRequest request;
  request.spatial_derivatives    = pf::SpatialDerivativeRequest::NONE;
  request.parameter_derivatives  = pf::ParameterDerivativeRequest::LOG_ONLY;
  request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  const pf::Result native              = model.evaluate(electrons, request);
  const pf::DirectScoreResult direct   = executor.evaluate(*workspace);
  REQUIRE(direct.parameter_score.size == model.p.size());
  REQUIRE(native.param_gradient.size() == model.p.size());
  CHECK(direct.sign == native.sign);
  checkClose(direct.logabs, native.logabs, 2.0e-11, 2.0e-11);
  double maximum_native_absolute_error = 0.0;
  double maximum_native_scaled_error   = 0.0;
  for (std::size_t parameter = 0; parameter < model.p.size(); ++parameter)
  {
    const double absolute_error =
        std::abs(direct.parameter_score[parameter] - native.param_gradient[parameter]);
    const double scale = std::max({1.0, std::abs(direct.parameter_score[parameter]),
                                   std::abs(native.param_gradient[parameter])});
    maximum_native_absolute_error = std::max(maximum_native_absolute_error, absolute_error);
    maximum_native_scaled_error   = std::max(maximum_native_scaled_error, absolute_error / scale);
  }
  CHECK(maximum_native_absolute_error < 2.0e-9);
  CHECK(maximum_native_scaled_error < 2.0e-10);

  const double parameter_sum = std::accumulate(direct.parameter_score.begin(),
                                               direct.parameter_score.end(), 0.0);
  double parameter_abs_sum    = 0.0;
  double parameter_square_sum = 0.0;
  for (double value : direct.parameter_score)
  {
    parameter_abs_sum += std::abs(value);
    parameter_square_sum += value * value;
  }
  checkClose(parameter_sum, golden.parameter_sum, 2.0e-9, 2.0e-8);
  checkClose(parameter_abs_sum, golden.parameter_abs_sum, 2.0e-9, 2.0e-7);
  checkClose(parameter_square_sum, golden.parameter_square_sum, 2.0e-9, 2.0e-7);
  for (int stream = 0; stream < 3; ++stream)
    checkClose(parameterProjection(direct.parameter_score, stream), golden.projections[stream],
               2.0e-8, 2.0e-8);
  for (std::size_t selected = 0; selected < golden.selected_indices.size(); ++selected)
    checkClose(direct.parameter_score[golden.selected_indices[selected]], golden.selected_values[selected],
               2.0e-8, 2.0e-9);

  // Exercise every non-owning output contract.  Selected and weighted sinks
  // preserve caller ordering, while accumulation must retain existing sentinels.
  std::vector<double> full_output(direct.parameter_score.size, -9.0);
  pf::DirectScoreOutput::writeFull(direct.parameter_score, full_output.data(), full_output.size());
  CHECK(full_output.front() == direct.parameter_score[0]);
  CHECK(full_output.back() == direct.parameter_score[direct.parameter_score.size - 1]);

  std::vector<double> selected_output(golden.selected_indices.size(), -7.0);
  pf::DirectScoreOutput::writeSelected(direct.parameter_score, golden.selected_indices.data(),
                                       golden.selected_indices.size(), selected_output.data());
  for (std::size_t selected = 0; selected < selected_output.size(); ++selected)
    CHECK(selected_output[selected] == direct.parameter_score[golden.selected_indices[selected]]);

  std::vector<std::size_t> destination_indices(golden.selected_indices.size());
  std::iota(destination_indices.begin(), destination_indices.end(), std::size_t{2});
  std::vector<double> accumulated(golden.selected_indices.size() + 4, 3.0);
  pf::DirectScoreOutput::accumulateSelected(
      direct.parameter_score, golden.selected_indices.data(), destination_indices.data(),
      golden.selected_indices.size(), 0.25, accumulated.data(), accumulated.size());
  CHECK(accumulated.front() == 3.0);
  CHECK(accumulated[1] == 3.0);
  for (std::size_t selected = 0; selected < golden.selected_indices.size(); ++selected)
    checkClose(accumulated[destination_indices[selected]],
               3.0 + 0.25 * direct.parameter_score[golden.selected_indices[selected]]);

  std::vector<double> selected_reference = selected_output;
  for (double& value : selected_reference)
    value -= 0.5;
  pf::DirectScoreOutput::accumulateSelectedDifference(
      direct.parameter_score, selected_reference.data(), golden.selected_indices.data(),
      destination_indices.data(), golden.selected_indices.size(), 0.2,
      accumulated.data(), accumulated.size());
  for (std::size_t selected = 0; selected < golden.selected_indices.size(); ++selected)
    checkClose(accumulated[destination_indices[selected]],
               3.1 + 0.25 * direct.parameter_score[golden.selected_indices[selected]]);

  std::vector<double> product_vector(golden.selected_indices.size());
  for (std::size_t selected = 0; selected < product_vector.size(); ++selected)
    product_vector[selected] = 0.1 * static_cast<double>(selected + 1);
  double expected_product = 0.0;
  for (std::size_t selected = 0; selected < product_vector.size(); ++selected)
    expected_product += (direct.parameter_score[golden.selected_indices[selected]] - 0.125) *
        product_vector[selected];
  checkClose(pf::DirectScoreOutput::selectedProduct(
                 direct.parameter_score, golden.selected_indices.data(), product_vector.data(),
                 product_vector.size(), 0.125),
             expected_product);

  // A standard optimizer update must be observed without rebuilding tensor descriptors.
  model.p.set_flat_value(127, model.p.flat_values()[127] + 1.0e-3);
  const pf::DirectScoreResult changed = executor.evaluate(*workspace);
  const pf::Result changed_native     = model.evaluate(electrons, request);
  checkClose(changed.logabs, changed_native.logabs, 2.0e-11, 2.0e-11);
  checkClose(changed.parameter_score[127], changed_native.param_gradient[127], 2.0e-8, 2.0e-9);
  CHECK(changed.parameter_version == model.p.version());

  // Once workspace construction is complete, neither tape nor score output may allocate.
  const double* stable_score_address = workspace->scoreData();
  volatile double allocation_sink    = changed.parameter_score[127];
  allocation_count.store(0, std::memory_order_relaxed);
  count_allocations.store(true, std::memory_order_relaxed);
  for (int repetition = 0; repetition < 3; ++repetition)
    allocation_sink += executor.evaluate(*workspace).parameter_score[127];
  count_allocations.store(false, std::memory_order_relaxed);
  CHECK(allocation_count.load(std::memory_order_relaxed) == 0);
  CHECK(workspace->scoreData() == stable_score_address);
  CHECK(std::isfinite(allocation_sink));
}

} // namespace

TEST_CASE("PsiFormer direct parameter score matches native and JAX LiH",
          "[wavefunction][psiformer]")
{
  validateScore("lih", lihScoreGolden());
}

TEST_CASE("PsiFormer direct parameter score matches native and JAX separated LiH pair",
          "[wavefunction][psiformer]")
{
  validateScore("lih_pair", pairScoreGolden());
}

TEST_CASE("PsiFormer direct score supports canonical pseudo-LiH without same-spin alpha",
          "[wavefunction][psiformer][ecp]")
{
  GeneratedFiles files = generateFiles("lih_pp");
  pf::PsiFormer model(files.parameters, files.configuration);
  const qmcplusplus::psiformer::ModelShape shape{
      model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet, model.dim,
      model.heads, model.blocks};
  const auto plan = qmcplusplus::psiformer::PsiFormerExecutionPlan::fromParameters(model.p, shape);
  CHECK_FALSE(plan.hasParameter(qmcplusplus::psiformer::ParameterRole::CUSP_SAME_ALPHA));

  pf::DirectScoreExecutor executor(model, plan);
  std::unique_ptr<pf::DirectScoreWorkspace> workspace = executor.makeWorkspace();
  const pf::Tensor electrons = model.cfg.configuration(0);
  workspace->setPositions(
      pf::GeometryPositionView::interleaved(electrons.x.data(), model.ne));

  pf::EvaluationRequest request;
  request.spatial_derivatives    = pf::SpatialDerivativeRequest::NONE;
  request.parameter_derivatives  = pf::ParameterDerivativeRequest::LOG_ONLY;
  request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  const pf::Result native = model.evaluate(electrons, request);
  const pf::DirectScoreResult direct = executor.evaluate(*workspace);
  REQUIRE(direct.parameter_score.size == model.p.size());
  REQUIRE(native.param_gradient.size() == model.p.size());
  CHECK(direct.sign == native.sign);
  checkClose(direct.logabs, native.logabs, 2e-11, 2e-11);

  double maximum_scaled_error = 0.0;
  for (std::size_t parameter = 0; parameter < model.p.size(); ++parameter)
  {
    const double error = std::abs(direct.parameter_score[parameter] -
                                  native.param_gradient[parameter]);
    const double scale = std::max({1.0, std::abs(direct.parameter_score[parameter]),
                                   std::abs(native.param_gradient[parameter])});
    maximum_scaled_error = std::max(maximum_scaled_error, error / scale);
  }
  CHECK(maximum_scaled_error < 2e-10);
}

TEST_CASE("PsiFormer orbital MSE uses the direct tape and spin-sector normalization",
          "[wavefunction][psiformer][pretraining]")
{
  GeneratedFiles files = generateFiles("unequal");
  pf::PsiFormer model(files.parameters, files.configuration);
  const qmcplusplus::psiformer::ModelShape shape{
      model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet, model.dim,
      model.heads, model.blocks};
  const auto plan = qmcplusplus::psiformer::PsiFormerExecutionPlan::fromParameters(model.p, shape);
  pf::DirectScoreExecutor executor(model, plan);
  std::unique_ptr<pf::DirectScoreWorkspace> workspace = executor.makeWorkspace();
  const pf::Tensor electrons = model.cfg.configuration(0);
  workspace->setPositions(pf::GeometryPositionView::interleaved(electrons.x.data(), model.ne));

  const std::size_t orbital_count = model.ndet * model.ne * model.ne;
  std::vector<double> target(orbital_count, 0.0);
  const pf::DirectOrbitalMSEResult result =
      executor.evaluateOrbitalMSE(*workspace, {target.data(), target.size()});
  REQUIRE(result.orbital_count == orbital_count);
  REQUIRE(result.parameter_gradient.size == model.p.size());
  CHECK(result.parameter_version == model.p.version());

  double expected_up = 0.0;
  double expected_down = 0.0;
  for (std::size_t determinant = 0; determinant < model.ndet; ++determinant)
    for (std::size_t electron = 0; electron < model.ne; ++electron)
      for (std::size_t orbital = 0; orbital < model.ne; ++orbital)
      {
        const std::size_t index = (determinant * model.ne + electron) * model.ne + orbital;
        const double square = result.predicted_orbitals[index] * result.predicted_orbitals[index];
        if (electron < model.cfg.nup)
          expected_up += square / static_cast<double>(model.ndet * model.cfg.nup * model.ne);
        else
          expected_down += square /
              static_cast<double>(model.ndet * model.cfg.ndown * model.ne);
      }
  checkClose(result.loss, expected_up + expected_down, 2e-12, 2e-12);

  using qmcplusplus::psiformer::ParameterRole;
  const std::array<std::size_t, 5> checked_parameters{
      plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).begin,
      plan.parameter(ParameterRole::ATTENTION_QUERY_WEIGHT, 0).begin,
      plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT).begin,
      plan.parameter(ParameterRole::ENVELOPE_PI_DOWN).begin,
      plan.parameter(ParameterRole::ENVELOPE_ZETA_UP).begin};
  std::array<double, checked_parameters.size()> analytic;
  for (std::size_t index = 0; index < checked_parameters.size(); ++index)
    analytic[index] = result.parameter_gradient[checked_parameters[index]];

  // A different target must not leave residual matrix or parameter adjoints in
  // the reusable workspace when the original target is evaluated again.
  const std::size_t storage_fingerprint = workspace->storageFingerprint();
  target[0] = 1.25;
  executor.evaluateOrbitalMSE(*workspace, {target.data(), target.size()});
  target[0] = 0.0;
  const pf::DirectOrbitalMSEResult repeated =
      executor.evaluateOrbitalMSE(*workspace, {target.data(), target.size()});
  for (std::size_t index = 0; index < checked_parameters.size(); ++index)
    checkClose(repeated.parameter_gradient[checked_parameters[index]], analytic[index], 2e-12, 2e-12);
  CHECK(workspace->storageFingerprint() == storage_fingerprint);

  const double epsilon = 2.0e-6;
  for (std::size_t index = 0; index < checked_parameters.size(); ++index)
  {
    const std::size_t parameter = checked_parameters[index];
    const double original = model.p.flat_values()[parameter];
    model.p.set_flat_value(parameter, original + epsilon);
    const double plus = executor.evaluateOrbitalMSE(*workspace, {target.data(), target.size()}).loss;
    model.p.set_flat_value(parameter, original - epsilon);
    const double minus = executor.evaluateOrbitalMSE(*workspace, {target.data(), target.size()}).loss;
    model.p.set_flat_value(parameter, original);
    CHECK(analytic[index] == Catch::Approx((plus - minus) / (2.0 * epsilon))
                                 .epsilon(3.0e-5).margin(3.0e-7));
  }

  CHECK(result.parameter_gradient.size == model.p.size());
  CHECK(executor.evaluateOrbitalMSE(*workspace, {target.data(), target.size()})
            .parameter_gradient[plan.parameter(ParameterRole::CUSP_SAME_ALPHA).begin] == 0.0);
  CHECK(executor.evaluateOrbitalMSE(*workspace, {target.data(), target.size()})
            .parameter_gradient[plan.parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin] == 0.0);
  CHECK_THROWS_AS(executor.evaluateOrbitalMSE(*workspace, {target.data(), target.size() - 1}),
                  std::invalid_argument);
}
