//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file test_psiformer_complex_score.cpp
 * @brief Complex real-parameter scores from complete VALUE differences and real parity.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
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

/// Reconstruct one real response channel from explicit affine rows in test code.
class AffineResponseOracle final : public pf::DirectAffineObservationSink
{
public:
  explicit AffineResponseOracle(const ps::PsiFormerExecutionPlan& plan)
      : gradient(plan.parameterCount()), covered(plan.parameterCount()), plan_(plan) {}

  void observe(const pf::DirectAffineObservation& observation) override
  {
    const auto& tensor = plan_.parameter(observation.role, observation.attention_block);
    REQUIRE(tensor.size() == observation.input_width * observation.output_width);
    for (std::size_t input = 0; input < observation.input_width; ++input)
      for (std::size_t output = 0; output < observation.output_width; ++output)
      {
        const auto parameter = tensor.begin + input * observation.output_width + output;
        REQUIRE_FALSE(covered[parameter]);
        covered[parameter] = true;
        for (std::size_t row = 0; row < observation.row_count; ++row)
          gradient[parameter] += observation.activations[row * observation.input_width + input] *
              observation.sensitivities[row * observation.output_width + output];
      }
    using Role = ps::ParameterRole;
    if (observation.role == Role::UPDATE_HIDDEN_WEIGHT || observation.role == Role::UPDATE_OUTPUT_WEIGHT)
    {
      const auto bias_role = observation.role == Role::UPDATE_HIDDEN_WEIGHT
          ? Role::UPDATE_HIDDEN_BIAS : Role::UPDATE_OUTPUT_BIAS;
      const auto& bias = plan_.parameter(bias_role, observation.attention_block);
      for (std::size_t output = 0; output < observation.output_width; ++output)
      {
        covered[bias.begin + output] = true;
        for (std::size_t row = 0; row < observation.row_count; ++row)
          gradient[bias.begin + output] += observation.sensitivities[row * observation.output_width + output];
      }
    }
    ++observations;
  }

  std::vector<double> gradient;
  std::vector<bool> covered;
  std::size_t observations = 0;
private:
  const ps::PsiFormerExecutionPlan& plan_;
};
} // namespace

TEST_CASE("Complex affine observations reconstruct both trunk and paired head scores",
          "[wavefunction][psiformer][complex_score][complex_affine]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  pf::DirectComplexScoreExecutor score(model, execution_plan);
  auto workspace = score.makeWorkspace();
  const auto positions = model.cfg.configuration(0);
  workspace->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  const auto bytes = workspace->vectorStorageBytes(), fingerprint = workspace->storageFingerprint();
  AffineResponseOracle real(execution_plan), imaginary(execution_plan);
  const auto result = score.evaluate(*workspace, &real, &imaginary);
  REQUIRE(result.head_activations != nullptr);
  REQUIRE(result.head_sensitivities != nullptr);
  CHECK(real.observations == 1 + 6 * model.blocks);
  CHECK(imaginary.observations == real.observations);
  for (std::size_t parameter = 0; parameter < result.parameter_count; ++parameter)
    if (real.covered[parameter])
      close({real.gradient[parameter], imaginary.gradient[parameter]}, result.parameter_score[parameter], 2e-11);
  const auto channels = model.ndet * model.ne;
  for (std::size_t spin = 0; spin < 2; ++spin)
  {
    const auto role = spin ? ps::ParameterRole::BACKFLOW_DOWN_WEIGHT : ps::ParameterRole::BACKFLOW_UP_WEIGHT;
    const auto& real_head = execution_plan.parameter(role);
    const auto& imaginary_head = execution_plan.parameter(role, ps::NO_ATTENTION_BLOCK, ps::HeadTensorComponent::IMAG);
    const auto begin = spin ? model.cfg.nup : 0, end = spin ? model.ne : model.cfg.nup;
    for (std::size_t feature = 0; feature < model.dim; ++feature)
      for (std::size_t channel = 0; channel < channels; ++channel)
      {
        C response{};
        for (std::size_t row = begin; row < end; ++row)
          response += result.head_activations[row * model.dim + feature] *
              result.head_sensitivities[(channel / model.ne * model.ne + row) * model.ne + channel % model.ne];
        const auto index = feature * channels + channel;
        close(response, result.parameter_score[real_head.begin + index], 2e-11);
        close(C{0, 1} * response, result.parameter_score[imaginary_head.begin + index], 2e-11);
      }
  }
  CHECK(workspace->vectorStorageBytes() == bytes);
  CHECK(workspace->storageFingerprint() == fingerprint);
  CHECK_THROWS(score.evaluate(*workspace, &real, nullptr));
  CHECK(score.evaluate(*workspace).head_sensitivities == nullptr);
}

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
  using Normalization = qmcplusplus::wftrain::OrbitalLossNormalization;
  const auto normalization = GENERATE(Normalization::SPIN_SECTOR_SUM, Normalization::FULL_MATRIX_MEAN);
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
    return executor.evaluateOrbitalMSE(*workspace,
        {target.data(), target.size(), gradient.data(), gradient.size(), normalization});
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
        const auto rows = normalization == Normalization::FULL_MATRIX_MEAN
            ? model.ne : (row < model.cfg.nup ? model.cfg.nup : model.cfg.ndown);
        const double scale = 1.0 / (model.ndet * model.ne * rows);
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
  // G-free raw heads and this fixed periodic-part target share one-cell periodicity.
  const auto periodic = evaluate();
  const auto periodic_gradient = gradient;
  workspace->setPosition(0, 0, positions.x[0] + 8.0);
  CHECK(evaluate().loss == Catch::Approx(periodic.loss).margin(2e-8));
  for (std::size_t parameter = 0; parameter < gradient.size(); ++parameter)
    CHECK(gradient[parameter] == Catch::Approx(periodic_gradient[parameter]).margin(2e-8));
  workspace->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
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
  // Exactly zero paired heads produce an exact determinant node at distinct positions.
  auto parameters = model.p.flat_values();
  for (auto role : {ps::ParameterRole::BACKFLOW_UP_WEIGHT, ps::ParameterRole::BACKFLOW_DOWN_WEIGHT})
    for (auto component : {ps::HeadTensorComponent::REAL, ps::HeadTensorComponent::IMAG})
    {
      const auto& tensor = execution_plan.parameter(role, ps::NO_ATTENTION_BLOCK, component);
      std::fill(parameters.begin() + tensor.begin, parameters.begin() + tensor.end, 0.0);
    }
  model.p.set_flat_values(parameters);
  pf::DirectComplexScoreExecutor executor(model, execution_plan);
  auto workspace = executor.makeWorkspace();
  const auto positions = model.cfg.configuration(0);
  workspace->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  CHECK_THROWS_AS(executor.evaluate(*workspace), std::domain_error);
  std::vector<C> target(model.ndet * model.ne * model.ne, C{0.2, -0.4});
  std::vector<double> gradient(model.p.size());
  const auto result = executor.evaluateOrbitalMSE(*workspace,
      {target.data(), target.size(), gradient.data(), gradient.size()});
  for (std::size_t index = 0; index < result.orbital_count; ++index)
    CHECK(result.orbital_matrices[index] == C{});
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
  bad.normalization = static_cast<qmcplusplus::wftrain::OrbitalLossNormalization>(99);
  failure(bad, executor);
  bad = valid;
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
