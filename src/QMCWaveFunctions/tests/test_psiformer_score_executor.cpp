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
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeterminantOverlap.h"
#include "QMCDrivers/WFTrain/OrbitalPretrainingAccumulator.h"
#include "psiformer_test_utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
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

/// Own one copied affine boundary after the executor has reused its reverse scratch.
struct CapturedAffineObservation
{
  qmcplusplus::psiformer::ParameterRole role;
  std::size_t attention_block;
  std::size_t rows;
  std::size_t input_width;
  std::size_t output_width;
  std::vector<double> activations;
  std::vector<double> sensitivities;
};

/// Capture transient affine rows solely for an independent test reconstruction.
class CapturingAffineSink final : public pf::DirectAffineObservationSink
{
public:
  void observe(const pf::DirectAffineObservation& observation) override
  {
    observations.push_back(
        {observation.role, observation.attention_block, observation.row_count,
         observation.input_width, observation.output_width,
         {observation.activations,
          observation.activations + observation.row_count * observation.input_width},
         {observation.sensitivities,
          observation.sensitivities + observation.row_count * observation.output_width}});
  }

  std::vector<CapturedAffineObservation> observations;
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

  // Stream the actual direct reverse result through the bounded Stage B sink.
  // This integration check prevents the generic producer seam from drifting from
  // the Stage A result's pointer, size, version, or normalization conventions.
  const qmcplusplus::wftrain::StructuredParameterSchema schema(
      "psiformer/direct-orbital-test",
      {{"network", {model.p.size()}, 0, model.p.size(),
        qmcplusplus::wftrain::ParameterScalarDomain::REAL64, true, "network"}});
  qmcplusplus::wftrain::OrbitalPretrainingAccumulator accumulator(
      schema, result.parameter_version, 0x1234, 0x5678);
  accumulator.addSample(result.loss, result.parameter_version,
                        {result.parameter_gradient.data,
                         result.parameter_gradient.size});
  accumulator.completeSingleParticipantReduction();
  const qmcplusplus::wftrain::OrbitalPretrainingResult streamed = accumulator.finalize();
  CHECK(streamed.sample_count == 1);
  CHECK(streamed.mean_loss == result.loss);
  REQUIRE(streamed.gradient.size() == result.parameter_gradient.size);
  for (std::size_t parameter : {std::size_t{0}, model.p.size() / 2,
                                model.p.size() - 1})
    CHECK(streamed.gradient[parameter] == result.parameter_gradient[parameter]);

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

TEST_CASE("PsiFormer independent channel amplitudes and weighted score share one reverse",
          "[wavefunction][psiformer][score][overlap]")
{
  GeneratedFiles files = generateFiles("unequal");
  pf::PsiFormer model(files.parameters, files.configuration);
  const qmcplusplus::psiformer::ModelShape shape{
      model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet,
      model.dim, model.heads, model.blocks};
  const auto plan =
      qmcplusplus::psiformer::PsiFormerExecutionPlan::fromParameters(model.p, shape);
  pf::DirectScoreExecutor executor(model, plan);
  auto workspace = executor.makeWorkspace();
  const pf::Tensor electrons = model.cfg.configuration(0);
  workspace->setPositions(
      pf::GeometryPositionView::interleaved(electrons.x.data(), model.ne));

  std::vector<double> phases(model.ndet);
  std::vector<double> log_abs(model.ndet);
  const pf::DirectChannelAmplitudeResult amplitudes =
      executor.evaluateChannelAmplitudes(*workspace, phases.data(), log_abs.data(),
                                         phases.size());
  REQUIRE(amplitudes.channel_count == model.ndet);
  CHECK(amplitudes.phases == phases.data());
  CHECK(amplitudes.log_abs == log_abs.data());
  CHECK(amplitudes.parameter_version == model.p.version());
  for (std::size_t channel = 0; channel < model.ndet; ++channel)
  {
    CHECK((phases[channel] == -1.0 || phases[channel] == 1.0));
    CHECK(std::isfinite(log_abs[channel]));
  }

  // The analytic cusp is a common multiplicative factor of every determinant
  // channel, so changing one cusp parameter must shift every channel log equally.
  using qmcplusplus::psiformer::ParameterRole;
  const std::vector<double> baseline_log_abs = log_abs;
  const std::size_t anti_cusp =
      plan.parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin;
  const double original_anti_cusp = model.p.flat_values()[anti_cusp];
  model.p.set_flat_value(anti_cusp, original_anti_cusp + 1.0e-5);
  executor.evaluateChannelAmplitudes(*workspace, phases.data(), log_abs.data(),
                                     phases.size());
  const double common_cusp_shift = log_abs[0] - baseline_log_abs[0];
  for (std::size_t channel = 1; channel < model.ndet; ++channel)
    checkClose(log_abs[channel] - baseline_log_abs[channel], common_cusp_shift,
               2.0e-10, 2.0e-12);
  model.p.set_flat_value(anti_cusp, original_anti_cusp);

  std::vector<double> weights(model.ndet, 0.0);
  weights[0] = 0.7;
  weights[1] = -0.2;
  weights[2] = 0.4;
  const pf::DirectWeightedChannelScoreResult weighted =
      executor.evaluateWeightedChannelScore(*workspace, weights.data(), weights.size());
  REQUIRE(weighted.parameter_score.size == model.p.size());
  CHECK(weighted.parameter_version == model.p.version());

  const std::array<std::size_t, 7> checked_parameters{
      plan.parameter(ParameterRole::CUSP_SAME_ALPHA).begin,
      plan.parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin,
      plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).begin,
      plan.parameter(ParameterRole::ATTENTION_QUERY_WEIGHT, 0).begin,
      plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT).begin,
      plan.parameter(ParameterRole::ENVELOPE_PI_DOWN).begin,
      plan.parameter(ParameterRole::ENVELOPE_ZETA_UP).begin};
  std::array<double, checked_parameters.size()> analytic{};
  for (std::size_t selected = 0; selected < checked_parameters.size(); ++selected)
    analytic[selected] = weighted.parameter_score[checked_parameters[selected]];

  auto weightedLogAmplitude = [&]() {
    executor.evaluateChannelAmplitudes(*workspace, phases.data(), log_abs.data(),
                                       phases.size());
    double value = 0.0;
    for (std::size_t channel = 0; channel < model.ndet; ++channel)
      value += weights[channel] * log_abs[channel];
    return value;
  };

  const double epsilon = 2.0e-6;
  for (std::size_t selected = 0; selected < checked_parameters.size(); ++selected)
  {
    const std::size_t parameter = checked_parameters[selected];
    const double original = model.p.flat_values()[parameter];
    model.p.set_flat_value(parameter, original + epsilon);
    const double plus = weightedLogAmplitude();
    model.p.set_flat_value(parameter, original - epsilon);
    const double minus = weightedLogAmplitude();
    model.p.set_flat_value(parameter, original);
    CHECK(analytic[selected] == Catch::Approx((plus - minus) / (2.0 * epsilon))
                                    .epsilon(5.0e-5).margin(5.0e-7));
  }

  // Independent channel seeds must add linearly even though their feature trunk and
  // cusp parameters are shared.  Check representative parameters across all roles.
  std::vector<double> first_weight(model.ndet, 0.0);
  std::vector<double> second_weight(model.ndet, 0.0);
  first_weight[0]  = weights[0];
  second_weight[1] = weights[1];
  std::array<double, checked_parameters.size()> first_score{};
  std::array<double, checked_parameters.size()> second_score{};
  const auto first = executor.evaluateWeightedChannelScore(
      *workspace, first_weight.data(), first_weight.size());
  for (std::size_t selected = 0; selected < checked_parameters.size(); ++selected)
    first_score[selected] = first.parameter_score[checked_parameters[selected]];
  const auto second = executor.evaluateWeightedChannelScore(
      *workspace, second_weight.data(), second_weight.size());
  for (std::size_t selected = 0; selected < checked_parameters.size(); ++selected)
    second_score[selected] = second.parameter_score[checked_parameters[selected]];

  std::vector<double> first_two_weight(model.ndet, 0.0);
  first_two_weight[0] = weights[0];
  first_two_weight[1] = weights[1];
  const auto first_two = executor.evaluateWeightedChannelScore(
      *workspace, first_two_weight.data(), first_two_weight.size());
  for (std::size_t selected = 0; selected < checked_parameters.size(); ++selected)
    checkClose(first_two.parameter_score[checked_parameters[selected]],
               first_score[selected] + second_score[selected], 2.0e-10, 2.0e-10);

  // The channel-amplitude and weighted reverse routes must keep every warmed buffer
  // stable and allocate no storage internally.
  const std::size_t fingerprint = workspace->storageFingerprint();
  const double* score_address = workspace->scoreData();
  volatile double allocation_sink = first_two.parameter_score[checked_parameters[0]];
  allocation_count.store(0, std::memory_order_relaxed);
  count_allocations.store(true, std::memory_order_relaxed);
  for (int repetition = 0; repetition < 3; ++repetition)
  {
    allocation_sink += executor.evaluateChannelAmplitudes(
        *workspace, phases.data(), log_abs.data(), phases.size()).log_abs[0];
    allocation_sink += executor.evaluateWeightedChannelScore(
        *workspace, weights.data(), weights.size()).parameter_score[checked_parameters[1]];
  }
  count_allocations.store(false, std::memory_order_relaxed);
  CHECK(allocation_count.load(std::memory_order_relaxed) == 0);
  CHECK(workspace->storageFingerprint() == fingerprint);
  CHECK(workspace->scoreData() == score_address);
  CHECK(std::isfinite(allocation_sink));

  CHECK_THROWS_AS(executor.evaluateChannelAmplitudes(
                      *workspace, phases.data(), log_abs.data(), phases.size() - 1),
                  std::invalid_argument);
  CHECK_THROWS_AS(executor.evaluateWeightedChannelScore(
                      *workspace, weights.data(), weights.size() - 1),
                  std::invalid_argument);
  weights[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_AS(executor.evaluateWeightedChannelScore(
                      *workspace, weights.data(), weights.size()),
                  std::invalid_argument);
}

TEST_CASE("PsiFormer affine observations reconstruct every supported score block",
          "[wavefunction][psiformer][score][kfac]")
{
  GeneratedFiles files = generateFiles("lih");
  pf::PsiFormer model(files.parameters, files.configuration);
  const qmcplusplus::psiformer::ModelShape shape{
      model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet,
      model.dim, model.heads, model.blocks};
  const auto plan =
      qmcplusplus::psiformer::PsiFormerExecutionPlan::fromParameters(model.p, shape);
  pf::DirectScoreExecutor executor(model, plan);
  auto workspace = executor.makeWorkspace();
  const pf::Tensor electrons = model.cfg.configuration(0);
  workspace->setPositions(
      pf::GeometryPositionView::interleaved(electrons.x.data(), model.ne));

  CapturingAffineSink sink;
  const pf::DirectScoreResult score =
      executor.evaluateWithAffineObservations(*workspace, sink);
  REQUIRE(sink.observations.size() == 3 + 6 * model.blocks);

  using qmcplusplus::psiformer::ParameterRole;
  for (const CapturedAffineObservation& observation : sink.observations)
  {
    const auto& weight = plan.parameter(observation.role,
                                        observation.attention_block);
    REQUIRE(weight.size() == observation.input_width * observation.output_width);
    double maximum_scaled_error = 0.0;
    for (std::size_t input = 0; input < observation.input_width; ++input)
      for (std::size_t output = 0; output < observation.output_width; ++output)
      {
        double reconstructed = 0.0;
        for (std::size_t row = 0; row < observation.rows; ++row)
          reconstructed +=
              observation.activations[row * observation.input_width + input] *
              observation.sensitivities[row * observation.output_width + output];
        const double actual = score.parameter_score[
            weight.begin + input * observation.output_width + output];
        maximum_scaled_error = std::max(
            maximum_scaled_error,
            std::abs(actual - reconstructed) /
                std::max({1.0, std::abs(actual), std::abs(reconstructed)}));
      }
    CHECK(maximum_scaled_error < 2e-11);

    ParameterRole bias_role = ParameterRole::COUNT;
    if (observation.role == ParameterRole::UPDATE_HIDDEN_WEIGHT)
      bias_role = ParameterRole::UPDATE_HIDDEN_BIAS;
    else if (observation.role == ParameterRole::UPDATE_OUTPUT_WEIGHT)
      bias_role = ParameterRole::UPDATE_OUTPUT_BIAS;
    if (bias_role != ParameterRole::COUNT)
    {
      const auto& bias = plan.parameter(bias_role, observation.attention_block);
      double maximum_bias_scaled_error = 0.0;
      for (std::size_t output = 0; output < observation.output_width; ++output)
      {
        double reconstructed = 0.0;
        for (std::size_t row = 0; row < observation.rows; ++row)
          reconstructed += observation.sensitivities[
              row * observation.output_width + output];
        const double actual = score.parameter_score[bias.begin + output];
        maximum_bias_scaled_error = std::max(
            maximum_bias_scaled_error,
            std::abs(actual - reconstructed) /
                std::max({1.0, std::abs(actual), std::abs(reconstructed)}));
      }
      CHECK(maximum_bias_scaled_error < 2e-11);
    }
  }
}

TEST_CASE("PsiFormer determinant overlap is two-pass stable and differentiable",
          "[wavefunction][psiformer][overlap]")
{
  using qmcplusplus::psiformer::DeterminantOverlapAccumulator;
  using qmcplusplus::psiformer::RealSignedLogAmplitude;
  const std::array<double, 2> objective_weights{0.7, 1.3};
  const std::array<std::array<double, 2>, 5> features{{
      {{-1.2, 0.4}}, {{-0.3, -0.8}}, {{0.2, 1.1}}, {{0.9, -0.1}}, {{1.4, 0.7}}}};
  const std::array<RealSignedLogAmplitude, 5> teachers{{
      {1.0, -0.7}, {-1.0, 0.2}, {1.0, 1.1}, {-1.0, -1.4}, {1.0, 0.5}}};

  auto makeAccumulator = [&](double parameter, std::size_t begin, std::size_t end) {
    DeterminantOverlapAccumulator accumulator(
        {objective_weights.begin(), objective_weights.end()});
    for (std::size_t sample = begin; sample < end; ++sample)
    {
      std::array<RealSignedLogAmplitude, 2> targets{teachers[sample], teachers[sample]};
      std::array<RealSignedLogAmplitude, 2> students;
      for (std::size_t channel = 0; channel < students.size(); ++channel)
        students[channel] = {teachers[sample].phase,
                             teachers[sample].log_abs + parameter * features[sample][channel]};
      accumulator.addSample(teachers[sample], targets.data(), students.data(), students.size());
    }
    return accumulator;
  };

  constexpr double parameter = 0.37;
  auto accumulator = makeAccumulator(parameter, 0, teachers.size());
  const std::size_t fingerprint = accumulator.storageFingerprint();
  const std::size_t retained_bytes = accumulator.retainedScalarBytes();
  const auto moment_snapshot = accumulator.snapshotMoments();
  DeterminantOverlapAccumulator restored(
      {objective_weights.begin(), objective_weights.end()});
  restored.replaceMoments(moment_snapshot);
  const double loss = accumulator.finalize();
  CHECK(restored.finalize() == Catch::Approx(loss).epsilon(2.0e-14));
  CHECK(loss > 0.0);
  CHECK(accumulator.sampleCount() == teachers.size());
  CHECK(accumulator.diagnostics()[0].fidelity <= 1.0);

  // Compare the signed-log streaming reduction with an explicit dense reference.
  double dense_loss = 0.0;
  for (std::size_t channel = 0; channel < objective_weights.size(); ++channel)
  {
    double cross = 0.0;
    double student_square = 0.0;
    for (const auto& sample_features : features)
    {
      const double ratio = std::exp(parameter * sample_features[channel]);
      cross += ratio;
      student_square += ratio * ratio;
    }
    const double fidelity = cross * cross /
        (static_cast<double>(features.size()) * student_square);
    dense_loss -= objective_weights[channel] * std::log(fidelity);
  }
  CHECK(loss == Catch::Approx(dense_loss).epsilon(2.0e-14));

  double analytic = 0.0;
  for (std::size_t sample = 0; sample < teachers.size(); ++sample)
  {
    std::array<RealSignedLogAmplitude, 2> targets{teachers[sample], teachers[sample]};
    std::array<RealSignedLogAmplitude, 2> students;
    for (std::size_t channel = 0; channel < students.size(); ++channel)
      students[channel] = {teachers[sample].phase,
                           teachers[sample].log_abs + parameter * features[sample][channel]};
    std::array<double, 2> score_weights;
    accumulator.channelScoreWeights(teachers[sample], targets.data(), students.data(),
                                    students.size(), score_weights.data());
    for (std::size_t channel = 0; channel < score_weights.size(); ++channel)
      analytic += score_weights[channel] * features[sample][channel];
  }

  constexpr double epsilon = 1.0e-6;
  auto plus = makeAccumulator(parameter + epsilon, 0, teachers.size());
  auto minus = makeAccumulator(parameter - epsilon, 0, teachers.size());
  const double finite_difference = (plus.finalize() - minus.finalize()) / (2.0 * epsilon);
  CHECK(analytic == Catch::Approx(finite_difference).epsilon(2.0e-8).margin(2.0e-9));

  // Independently accumulated rank-local moments merge to the dense first pass.
  auto left = makeAccumulator(parameter, 0, 2);
  auto right = makeAccumulator(parameter, 2, teachers.size());
  left.merge(right);
  CHECK(left.finalize() == Catch::Approx(loss).epsilon(2.0e-14));

  // Reset and warmed two-pass reuse preserve every retained allocation.
  std::array<double, 2> score_weights{};
  allocation_count.store(0, std::memory_order_relaxed);
  count_allocations.store(true, std::memory_order_relaxed);
  accumulator.reset();
  for (std::size_t sample = 0; sample < teachers.size(); ++sample)
  {
    std::array<RealSignedLogAmplitude, 2> targets{teachers[sample], teachers[sample]};
    std::array<RealSignedLogAmplitude, 2> students;
    for (std::size_t channel = 0; channel < students.size(); ++channel)
      students[channel] = {teachers[sample].phase,
                           teachers[sample].log_abs + parameter * features[sample][channel]};
    accumulator.addSample(teachers[sample], targets.data(), students.data(), students.size());
  }
  accumulator.finalize();
  std::array<RealSignedLogAmplitude, 2> final_targets{teachers.back(), teachers.back()};
  std::array<RealSignedLogAmplitude, 2> final_students;
  for (std::size_t channel = 0; channel < final_students.size(); ++channel)
    final_students[channel] = {teachers.back().phase,
                               teachers.back().log_abs + parameter * features.back()[channel]};
  accumulator.channelScoreWeights(teachers.back(), final_targets.data(), final_students.data(),
                                  final_students.size(), score_weights.data());
  count_allocations.store(false, std::memory_order_relaxed);
  CHECK(allocation_count.load(std::memory_order_relaxed) == 0);
  CHECK(accumulator.storageFingerprint() == fingerprint);
  CHECK(accumulator.retainedScalarBytes() == retained_bytes);
}

TEST_CASE("PsiFormer determinant overlap rejects nodes cancellation and bad lifecycle",
          "[wavefunction][psiformer][overlap]")
{
  using qmcplusplus::psiformer::DeterminantOverlapAccumulator;
  using qmcplusplus::psiformer::RealSignedLogAmplitude;
  DeterminantOverlapAccumulator accumulator({1.0});
  const RealSignedLogAmplitude one{1.0, 0.0};
  const RealSignedLogAmplitude zero{};
  CHECK_THROWS_AS(accumulator.addSample(zero, &one, &one, 1), std::domain_error);
  for (double malformed_log : {std::numeric_limits<double>::infinity(), 0.0,
                               std::numeric_limits<double>::quiet_NaN()})
  {
    const RealSignedLogAmplitude malformed_zero{0.0, malformed_log};
    CHECK_THROWS_AS(accumulator.addSample(malformed_zero, &one, &one, 1), std::invalid_argument);
  }
  CHECK_THROWS_AS(accumulator.finalize(), std::domain_error);

  accumulator.addSample(one, &one, &one, 1);
  RealSignedLogAmplitude negative{-1.0, 0.0};
  accumulator.addSample(one, &negative, &one, 1);
  CHECK_THROWS_AS(accumulator.finalize(), std::domain_error);

  DeterminantOverlapAccumulator valid({1.0});
  valid.addSample(one, &one, &one, 1);
  valid.finalize();
  CHECK_THROWS_AS(valid.addSample(one, &one, &one, 1), std::logic_error);
  double score_weight = 0.0;
  CHECK_THROWS_AS(valid.channelScoreWeights(one, &one, &zero, 1, &score_weight),
                  std::domain_error);
}
