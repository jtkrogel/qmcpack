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
