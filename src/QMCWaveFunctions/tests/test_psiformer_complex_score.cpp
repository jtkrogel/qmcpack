//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file test_psiformer_complex_score.cpp
 * @brief Complex real-parameter scores from complete VALUE differences and real parity.
 */
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexValueExecutor.h"
#include "psiformer_value_test_utils.h"

namespace
{
namespace ps = qmcplusplus::psiformer;
using namespace qmcplusplus::testing::psiformer::value;
using C = std::complex<double>;

/// Check both components without suppressing a real parameter's learned phase response.
void close(C actual, C expected, double tolerance = 2e-8)
{
  CHECK(actual.real() == Catch::Approx(expected.real()).margin(tolerance));
  CHECK(actual.imag() == Catch::Approx(expected.imag()).margin(tolerance));
}

/// Form stable relative tiny-fixture amplitudes without materializing their small magnitude.
C relative(const pf::DirectComplexValueResult& value, const ps::determinant::ComplexLogAmplitude& center)
{
  return value.amplitude.phase / center.phase * std::exp(value.amplitude.log_abs - center.log_abs);
}

/// Independent complete-VALUE symmetric difference for one real canonical coordinate.
C difference(pf::PsiFormer& model, const pf::DirectComplexValueExecutor& executor,
             pf::DirectComplexValueWorkspace& workspace, std::size_t parameter)
{
  const auto center = executor.evaluate(workspace).amplitude;
  const auto original = model.p.flat_values();
  auto values = original;
  const double step = 1e-5;
  values[parameter] += step;
  model.p.set_flat_values(values);
  const C plus = relative(executor.evaluate(workspace), center);
  values[parameter] -= 2 * step;
  model.p.set_flat_values(values);
  const C minus = relative(executor.evaluate(workspace), center);
  model.p.set_flat_values(original);
  return (plus - minus) / (2 * step);
}
} // namespace

TEST_CASE("PsiFormer complete complex score covers every real canonical tensor and tail",
          "[wavefunction][psiformer][complex_score]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  pf::DirectComplexScoreExecutor score(model, execution_plan);
  pf::DirectComplexValueExecutor value(model, execution_plan);
  auto workspace = score.makeWorkspace();
  auto reference = value.makeWorkspace();
  const auto positions = model.cfg.configuration(0);
  const auto view = pf::GeometryPositionView::interleaved(positions.x.data(), model.ne);
  workspace->setPositions(view);
  reference->setPositions(view);
  const auto result = score.evaluate(*workspace);
  const auto expected = value.evaluate(*reference);
  close(result.amplitude.phase, expected.amplitude.phase, 2e-11);
  CHECK(result.amplitude.log_abs == Catch::Approx(expected.amplitude.log_abs).margin(2e-11));
  REQUIRE(result.parameter_count == model.p.size());
  const std::vector<C> analytic(result.parameter_score, result.parameter_score + result.parameter_count);
  for (const auto& tensor : execution_plan.parameterTensors())
    for (std::size_t parameter : {tensor.begin, tensor.end - 1})
    {
      INFO(tensor.module << "/" << tensor.name << " index=" << parameter);
      close(analytic[parameter], difference(model, value, *reference, parameter));
    }
  // A vanishing envelope must not erase its parameter response by dividing A/E.
  auto parameters = model.p.flat_values();
  const auto& pi = execution_plan.parameter(ps::ParameterRole::ENVELOPE_PI_UP);
  parameters[pi.begin] = parameters[pi.begin + 1] = 0;
  model.p.set_flat_values(parameters);
  const auto zero_envelope = score.evaluate(*workspace);
  const C zero_response = zero_envelope.parameter_score[pi.begin];
  close(zero_response, difference(model, value, *reference, pi.begin));
  CHECK(std::abs(zero_response) > 1e-8);
  const auto fingerprint = workspace->storageFingerprint();
  const auto bytes = workspace->vectorStorageBytes();
  score.evaluate(*workspace);
  CHECK(workspace->storageFingerprint() == fingerprint);
  CHECK(workspace->vectorStorageBytes() == bytes);
  pf::DirectScoreExecutor trunk(model, execution_plan, pf::DirectScorePreparation::REAL_TRUNK_ONLY);
  auto trunk_workspace = trunk.makeTrunkWorkspace();
  const ps::ComplexOrbitalHeadDescriptor descriptor{model.ne, model.cfg.nup, model.ndet, model.dim,
      ps::ComplexOrbitalStructure::FULL, ps::ComplexOrbitalPhase::NONE, {}};
  const auto features = ps::makeSpatialJetLayout(1, model.ne * model.dim, model.ne, ps::SpatialJetMode::VALUE_ONLY);
  const auto orbitals = ps::makeSpatialJetLayout(1, model.ndet * model.ne * model.ne,
                                               model.ne, ps::SpatialJetMode::VALUE_ONLY);
  ps::ComplexOrbitalHeadWorkspace head(descriptor, features, orbitals);
  ps::determinant::ComplexDeterminantWorkspace determinant(model.ndet, model.ne);
  CHECK(bytes == trunk_workspace->requiredStorageBytes() + head.storageBytes() + determinant.storageBytes() +
      sizeof(double) * (orbitals.element_count + features.element_count) +
      sizeof(C) * (orbitals.element_count + model.p.size()));
}

TEST_CASE("PsiFormer zero-imaginary score preserves real scores and nonzero imaginary-head responses",
          "[wavefunction][psiformer][complex_score]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  auto parameters = model.p.flat_values();
  for (auto role : {ps::ParameterRole::BACKFLOW_UP_WEIGHT, ps::ParameterRole::BACKFLOW_DOWN_WEIGHT})
  {
    const auto& imaginary = execution_plan.parameter(role, ps::NO_ATTENTION_BLOCK, ps::HeadTensorComponent::IMAG);
    std::fill(parameters.begin() + imaginary.begin, parameters.begin() + imaginary.end, 0);
  }
  model.p.set_flat_values(parameters);
  pf::DirectComplexScoreExecutor complex(model, execution_plan);
  auto workspace = complex.makeWorkspace();
  auto real_model = tinyModel(true);
  const auto real_plan = plan(real_model, true);
  pf::DirectScoreExecutor real(real_model, real_plan);
  auto real_workspace = real.makeWorkspace();
  const auto positions = model.cfg.configuration(0);
  const auto view = pf::GeometryPositionView::interleaved(positions.x.data(), model.ne);
  workspace->setPositions(view);
  real_workspace->setPositions(view);
  const auto result = complex.evaluate(*workspace);
  const auto expected = real.evaluate(*real_workspace);
  for (const auto& tensor : real_plan.parameterTensors())
    for (std::size_t i = 0; i < tensor.size(); ++i)
      close(result.parameter_score[tensor.begin + i], C{expected.parameter_score[tensor.begin + i]}, 2e-10);
  double imaginary_norm = 0;
  for (auto role : {ps::ParameterRole::BACKFLOW_UP_WEIGHT, ps::ParameterRole::BACKFLOW_DOWN_WEIGHT})
  {
    const auto& real_head = execution_plan.parameter(role);
    const auto& imaginary = execution_plan.parameter(role, ps::NO_ATTENTION_BLOCK, ps::HeadTensorComponent::IMAG);
    for (std::size_t i = 0; i < imaginary.size(); ++i)
    {
      const C derivative = result.parameter_score[imaginary.begin + i];
      close(derivative, C{0, expected.parameter_score[real_head.begin + i]}, 2e-10);
      imaginary_norm += std::norm(derivative);
    }
  }
  CHECK(imaginary_norm > 1e-8);
  CHECK_THROWS_AS(pf::DirectComplexScoreExecutor(real_model, real_plan), std::invalid_argument);
}

TEST_CASE("PsiFormer complex score publication rejects nodes foreign owners and failed generations",
          "[wavefunction][psiformer][complex_score]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  pf::DirectComplexScoreExecutor executor(model, execution_plan), foreign(model, execution_plan);
  auto workspace = executor.makeWorkspace();
  const auto positions = model.cfg.configuration(0);
  const auto view = pf::GeometryPositionView::interleaved(positions.x.data(), model.ne);
  workspace->setPositions(view);
  const auto first = executor.evaluate(*workspace);
  auto parameters = model.p.flat_values();
  parameters[1] += .01;
  model.p.set_flat_values(parameters);
  const auto updated = executor.evaluate(*workspace);
  CHECK(updated.parameter_version == model.p.version());
  CHECK(updated.parameter_version != first.parameter_version);
  CHECK(updated.evaluation_generation != first.evaluation_generation);
  CHECK_THROWS_AS(foreign.evaluate(*workspace), std::invalid_argument);
  CHECK_THROWS(workspace->result());
  workspace->setPosition(0, 0, std::numeric_limits<double>::quiet_NaN());
  CHECK_THROWS(executor.evaluate(*workspace));
  CHECK_THROWS(workspace->result());
  CHECK(workspace->observedParameterVersion() == std::numeric_limits<std::size_t>::max());
  workspace->setPositions(view);
  executor.evaluate(*workspace);
  for (std::size_t dim = 0; dim < 3; ++dim)
    workspace->setPosition(1, dim, positions.x[dim]);
  CHECK_THROWS_AS(executor.evaluate(*workspace), std::domain_error);
  CHECK_THROWS(workspace->result());
  workspace->setPositions(view);
  const auto recovered = executor.evaluate(*workspace);
  CHECK(recovered.parameter_count == model.p.size());
}

TEST_CASE("Complex raw orbital MSE has real parameter gradients and exact spin normalization",
          "[wavefunction][psiformer][complex_orbital_mse]")
{
  auto model = tinyModel(true, true);
  model.cfg.nup = 1;
  model.cfg.ndown = 3;
  const auto execution_plan = plan(model, true, true);
  pf::DirectComplexScoreExecutor executor(model, execution_plan);
  auto workspace = executor.makeWorkspace();
  const auto positions = model.cfg.configuration(0);
  workspace->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  std::vector<C> target(model.ndet * model.ne * model.ne);
  std::vector<double> gradient(model.p.size());
  auto evaluate = [&]() {
    return executor.evaluateOrbitalMSE(*workspace, {target.data(), target.size(), gradient.data(), gradient.size()});
  };
  const auto zero_target = evaluate();
  double expected_loss = 0;
  for (std::size_t d = 0; d < model.ndet; ++d)
    for (std::size_t row = 0; row < model.ne; ++row)
      for (std::size_t column = 0; column < model.ne; ++column)
      {
        const auto index = (d * model.ne + row) * model.ne + column;
        const bool same_spin = (row < model.cfg.nup) == (column < model.cfg.nup);
        // Off-spin targets are zero, but FULL student heads still train those entries.
        target[index] = same_spin ? C{0.03 * (index + 1), -0.02 * (row + 1)} : C{};
        const double scale = 1.0 / (model.ndet * model.ne * (row < model.cfg.nup ? model.cfg.nup : model.cfg.ndown));
        expected_loss += scale * std::norm(zero_target.orbital_matrices[index] - target[index]);
      }
  const auto result = evaluate();
  CHECK(result.loss == Catch::Approx(expected_loss).margin(1e-12));
  CHECK(result.parameter_version == model.p.version());
  CHECK(result.orbital_count == target.size());
  CHECK(workspace->orbitalMSEResult().evaluation_generation == result.evaluation_generation);
  CHECK_THROWS(workspace->result());
  const auto analytic = gradient;
  const auto original = model.p.flat_values();
  constexpr double step = 1e-5;
  double imaginary_head_norm = 0, trunk_norm = 0;
  for (const auto& tensor : execution_plan.parameterTensors())
    for (std::size_t parameter : {tensor.begin, tensor.end - 1})
    {
      INFO(tensor.module << "/" << tensor.name << " index=" << parameter);
      auto values = original;
      values[parameter] += step;
      model.p.set_flat_values(values);
      const double plus = evaluate().loss;
      values[parameter] -= 2 * step;
      model.p.set_flat_values(values);
      const double minus = evaluate().loss;
      model.p.set_flat_values(original);
      CHECK(analytic[parameter] == Catch::Approx((plus - minus) / (2 * step)).margin(3e-7));
      if (tensor.name == "w_imag")
        imaginary_head_norm += std::abs(analytic[parameter]);
      if (tensor.module.find("attention") != std::string::npos)
        trunk_norm += std::abs(analytic[parameter]);
    }
  CHECK(imaginary_head_norm > 1e-8);
  CHECK(trunk_norm > 1e-8);
  for (auto role : {ps::ParameterRole::CUSP_SAME_ALPHA, ps::ParameterRole::CUSP_OPPOSITE_ALPHA})
    CHECK(analytic[execution_plan.parameter(role).begin] == 0);
  const auto bytes = workspace->vectorStorageBytes(), fingerprint = workspace->storageFingerprint();
  evaluate();
  evaluate();
  CHECK(workspace->vectorStorageBytes() == bytes);
  CHECK(workspace->storageFingerprint() == fingerprint);
  executor.evaluate(*workspace);
  CHECK_THROWS(workspace->orbitalMSEResult());
  CHECK(workspace->result().parameter_version == model.p.version());
  // Empty sectors remain outside the existing model/plan admission, not silently widened here.
  model.cfg.nup = 0;
  model.cfg.ndown = 4;
  CHECK_THROWS_AS(plan(model, true, true), std::invalid_argument);
}

TEST_CASE("Complex orbital MSE remains finite at an exact determinant node",
          "[wavefunction][psiformer][complex_orbital_mse]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  pf::DirectComplexScoreExecutor executor(model, execution_plan);
  auto workspace = executor.makeWorkspace();
  const auto positions = model.cfg.configuration(0);
  workspace->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    workspace->setPosition(1, dimension, positions.x[dimension]);
  CHECK_THROWS_AS(executor.evaluate(*workspace), std::domain_error);
  std::vector<C> target(model.ndet * model.ne * model.ne, C{0.2, -0.4});
  std::vector<double> gradient(model.p.size());
  const auto result = executor.evaluateOrbitalMSE(*workspace,
      {target.data(), target.size(), gradient.data(), gradient.size()});
  ps::determinant::ComplexDeterminantWorkspace determinant(model.ndet, model.ne);
  CHECK(determinant.evaluate(result.orbital_matrices).amplitude.isZero());
  CHECK(std::isfinite(result.loss));
  CHECK(std::all_of(gradient.begin(), gradient.end(), [](double value) { return std::isfinite(value); }));
  CHECK(std::any_of(gradient.begin(), gradient.end(), [](double value) { return std::abs(value) > 1e-8; }));
}

TEST_CASE("Complex orbital MSE rejects invalid requests before caller publication and recovers",
          "[wavefunction][psiformer][complex_orbital_mse]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  pf::DirectComplexScoreExecutor executor(model, execution_plan), foreign(model, execution_plan);
  auto workspace = executor.makeWorkspace();
  const auto positions = model.cfg.configuration(0);
  workspace->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  std::vector<C> target(model.ndet * model.ne * model.ne, C{0.1, -0.2});
  std::vector<double> gradient(model.p.size(), 123.0);
  const auto original = model.p.flat_values();
  const auto valid = pf::DirectComplexOrbitalMSEArguments{target.data(), target.size(), gradient.data(), gradient.size()};
  auto failure = [&](auto arguments, const pf::DirectComplexScoreExecutor& selected) {
    std::fill(gradient.begin(), gradient.end(), 123.0);
    const auto generation = workspace->evaluationGeneration();
    CHECK_THROWS(selected.evaluateOrbitalMSE(*workspace, arguments));
    CHECK(workspace->evaluationGeneration() > generation);
    CHECK_THROWS(workspace->orbitalMSEResult());
    CHECK_THROWS(workspace->result());
    CHECK(workspace->observedParameterVersion() == std::numeric_limits<std::size_t>::max());
    CHECK(std::all_of(gradient.begin(), gradient.end(), [](double value) { return value == 123.0; }));
    CHECK(model.p.flat_values() == original);
  };
  auto bad = valid;
  --bad.target_size;
  failure(bad, executor);
  bad = valid;
  --bad.gradient_size;
  failure(bad, executor);
  failure(valid, foreign);
  target[0] = C{std::numeric_limits<double>::quiet_NaN(), 0};
  failure(valid, executor);
  target[0] = C{std::numeric_limits<double>::max(), 0};
  failure(valid, executor);
  target[0] = C{0.1, -0.2};
  bad = valid;
  bad.parameter_gradient = const_cast<double*>(model.p.flat_values().data());
  failure(bad, executor);
  bad = valid;
  bad.parameter_gradient = reinterpret_cast<double*>(target.data());
  failure(bad, executor);
  const auto first = executor.evaluateOrbitalMSE(*workspace, valid);
  bad = valid;
  bad.parameter_gradient = reinterpret_cast<double*>(const_cast<C*>(first.orbital_matrices));
  failure(bad, executor);
  bad = valid;
  bad.target = first.orbital_matrices;
  failure(bad, executor);
  workspace->setPosition(0, 0, std::numeric_limits<double>::quiet_NaN());
  failure(valid, executor);
  workspace->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  const auto recovered = executor.evaluateOrbitalMSE(*workspace, valid);
  CHECK(recovered.parameter_version == model.p.version());
  CHECK(recovered.evaluation_generation > first.evaluation_generation);
  CHECK(workspace->observedParameterVersion() == model.p.version());
}
