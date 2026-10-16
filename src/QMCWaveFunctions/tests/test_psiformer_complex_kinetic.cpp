//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
//////////////////////////////////////////////////////////////////////////////////////
/** @file test_psiformer_complex_kinetic.cpp
 * @brief Complete complex kinetic parameter differences, real parity and strict admission.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexKineticExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexValueExecutor.h"
#include "psiformer_value_test_utils.h"

namespace
{
namespace ps = qmcplusplus::psiformer;
using namespace qmcplusplus::testing::psiformer::value;
using C = std::complex<double>;

/// Compare both Cartesian components, never a magnitude-only or real-energy surrogate.
void close(C actual, C expected, double tolerance = 2e-7)
{
  CHECK(actual.real() == Catch::Approx(expected.real()).margin(tolerance));
  CHECK(actual.imag() == Catch::Approx(expected.imag()).margin(tolerance));
}

/// Independently form complete physical K from the qualified spatial owner at one sample.
C kinetic(const pf::DirectComplexSpatialResult& value, const std::vector<double>& masses,
          const std::array<double, 3>& outer)
{
  C result{};
  for (std::size_t electron = 0; electron < masses.size(); ++electron)
  {
    C ratio = value.laplacian_log[electron];
    for (std::size_t dim = 0; dim < 3; ++dim)
    {
      const C gradient = value.gradient[3 * electron + dim] + C{0, outer[dim]};
      ratio += gradient * gradient;
    }
    result -= 0.5 * masses[electron] * ratio;
  }
  return result;
}

/// Test-only symmetric parameter difference of complete spatial K, restoring real parameters.
C difference(pf::PsiFormer& model, const pf::DirectComplexSpatialExecutor& spatial,
             pf::DirectComplexSpatialWorkspace& workspace, const std::vector<double>& masses,
             const std::array<double, 3>& outer, std::size_t parameter)
{
  const auto original = model.p.flat_values();
  auto values = original;
  values[parameter] += 1e-5;
  model.p.set_flat_values(values);
  const C plus = kinetic(spatial.evaluateFull(workspace), masses, outer);
  values[parameter] -= 2e-5;
  model.p.set_flat_values(values);
  const C minus = kinetic(spatial.evaluateFull(workspace), masses, outer);
  model.p.set_flat_values(original);
  return (plus - minus) / 2e-5;
}
} // namespace

TEST_CASE("PsiFormer complete complex kinetic response covers canonical roles and tails",
          "[psiformer][complex_kinetic]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  const std::array<double, 3> outer{0.12, -0.07, 0.04};
  const std::vector<double> masses{0.45, 0.62, 0.79, 0.96};
  pf::DirectComplexKineticExecutor executor(model, execution_plan, outer);
  const auto required = executor.requiredStorageBytes();
  CHECK_THROWS(executor.makeWorkspace(0));
  CHECK_THROWS(executor.makeWorkspace(std::numeric_limits<std::size_t>::max()));
  CHECK_THROWS(executor.makeWorkspace(required - 1));
  auto workspace = executor.makeWorkspace(required);
  CHECK(workspace->vectorStorageBytes() == required);
  pf::DirectComplexSpatialExecutor spatial(model, execution_plan);
  auto reference = spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  const auto positions = model.cfg.configuration(0);
  const auto view = pf::GeometryPositionView::interleaved(positions.x.data(), model.ne);
  workspace->setPositions(view);
  reference->setPositions(view);
  const auto result = executor.evaluate(*workspace, masses.data(), masses.size());
  const auto expected = spatial.evaluateFull(*reference);
  close(result.kinetic, kinetic(expected, masses, outer), 2e-10);
  close(result.amplitude.phase, expected.amplitude.phase, 2e-10);
  CHECK(result.amplitude.log_abs == Catch::Approx(expected.amplitude.log_abs).margin(2e-10));
  const std::vector<C> analytic(result.parameter_response, result.parameter_response + result.parameter_count);
  for (const auto& tensor : execution_plan.parameterTensors())
    for (std::size_t parameter : {tensor.begin, tensor.end - 1})
    {
      INFO(tensor.module << "/" << tensor.name << " index=" << parameter);
      close(analytic[parameter], difference(model, spatial, *reference, masses, outer, parameter));
    }

  // A regular FULL determinant can have zero up-row envelopes in one channel/column.
  auto parameters = model.p.flat_values();
  const auto& pi = execution_plan.parameter(ps::ParameterRole::ENVELOPE_PI_UP);
  parameters[pi.begin] = parameters[pi.begin + 1] = 0;
  model.p.set_flat_values(parameters);
  const auto zero = executor.evaluate(*workspace, masses.data(), masses.size());
  const C zero_response = zero.parameter_response[pi.begin];
  close(zero_response, difference(model, spatial, *reference, masses, outer, pi.begin));
  CHECK(std::abs(zero_response) > 1e-8);
  const auto fingerprint = workspace->storageFingerprint();
  executor.evaluate(*workspace, masses.data(), masses.size());
  CHECK(workspace->storageFingerprint() == fingerprint);
  CHECK(workspace->vectorStorageBytes() == required);
  const auto generation = workspace->evaluationGeneration();
  CHECK_THROWS(executor.evaluate(*workspace, masses.data(), masses.size() - 1));
  CHECK(workspace->evaluationGeneration() != generation);
  CHECK_THROWS(workspace->result());
  auto invalid = masses;
  invalid[0] = 0;
  CHECK_THROWS(executor.evaluate(*workspace, invalid.data(), invalid.size()));
  invalid[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(executor.evaluate(*workspace, invalid.data(), invalid.size()));
  CHECK_NOTHROW(executor.evaluate(*workspace, masses.data(), masses.size()));
  pf::DirectComplexKineticExecutor foreign(model, execution_plan, outer);
  CHECK_THROWS(foreign.evaluate(*workspace, masses.data(), masses.size()));
  CHECK_THROWS(workspace->result());
  workspace->setPosition(0, 0, std::numeric_limits<double>::quiet_NaN());
  CHECK_THROWS(executor.evaluate(*workspace, masses.data(), masses.size()));
  CHECK_THROWS(workspace->result());
  workspace->setPositions(view);
  const auto recovered = executor.evaluate(*workspace, masses.data(), masses.size());
  CHECK(recovered.parameter_version == model.p.version());
  CHECK(recovered.evaluation_generation == workspace->evaluationGeneration());

  // Forward tickets also bind borrowed geometry and packed-coordinate identities.
  pf::DirectKineticExecutor trunk(model, execution_plan, pf::DirectKineticPreparation::REAL_TRUNK_ONLY);
  auto trunk_workspace = trunk.makeTrunkWorkspace(required);
  trunk_workspace->setPositions(view);
  const auto ticket = trunk.evaluateTrunk(*trunk_workspace);
  pf::DirectTraceJetBuffer feature_seed(model.ne * model.dim, 3 * model.ne, model.ne);
  pf::DirectTraceJetBuffer cusp_seed(1, 3 * model.ne, model.ne);
  auto forged = ticket;
  forged.geometry = nullptr;
  CHECK_THROWS(trunk.reverseTrunk(*trunk_workspace, forged, feature_seed, cusp_seed));
  forged = ticket;
  forged.packed_positions = nullptr;
  CHECK_THROWS(trunk.reverseTrunk(*trunk_workspace, forged, feature_seed, cusp_seed));
}

TEST_CASE("PsiFormer zero-imaginary kinetic response specializes to the complete legacy real owner",
          "[psiformer][complex_kinetic]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  auto parameters = model.p.flat_values();
  for (const auto& tensor : execution_plan.parameterTensors())
    if (tensor.head_component == ps::HeadTensorComponent::IMAG)
      std::fill(parameters.begin() + tensor.begin, parameters.begin() + tensor.end, 0);
  model.p.set_flat_values(parameters);
  auto real_model = tinyModel(true);
  const auto real_plan = plan(real_model, true);
  auto real_parameters = real_model.p.flat_values();
  for (const auto& tensor : real_plan.parameterTensors())
  {
    const auto& source = execution_plan.parameter(tensor.role, tensor.attention_block);
    std::copy_n(parameters.begin() + source.begin, tensor.size(), real_parameters.begin() + tensor.begin);
  }
  real_model.p.set_flat_values(real_parameters);
  pf::DirectComplexKineticExecutor complex(model, execution_plan);
  pf::DirectKineticExecutor real(real_model, real_plan);
  auto workspace = complex.makeWorkspace(complex.requiredStorageBytes());
  auto real_workspace = real.makeWorkspace();
  const auto positions = model.cfg.configuration(0);
  const auto view = pf::GeometryPositionView::interleaved(positions.x.data(), model.ne);
  workspace->setPositions(view);
  real_workspace->setPositions(view);
  const std::vector<double> masses{0.45, 0.62, 0.79, 0.96};
  const auto result = complex.evaluate(*workspace, masses.data(), masses.size());
  const auto expected = real.evaluate(*real_workspace, nullptr, 0, masses.data(), masses.size());
  C energy{};
  for (std::size_t e = 0; e < model.ne; ++e)
    energy -= 0.5 * masses[e] * expected.lap_ratio[e];
  close(result.kinetic, energy, 2e-9);
  for (const auto& tensor : real_plan.parameterTensors())
  {
    const auto& target = execution_plan.parameter(tensor.role, tensor.attention_block);
    for (std::size_t i = 0; i < tensor.size(); ++i)
      close(result.parameter_response[target.begin + i], expected.kinetic_parameter_response[tensor.begin + i], 2e-9);
  }
  for (auto role : {ps::ParameterRole::BACKFLOW_UP_WEIGHT, ps::ParameterRole::BACKFLOW_DOWN_WEIGHT})
  {
    const auto& real_head = execution_plan.parameter(role);
    const auto& imaginary = execution_plan.parameter(role, ps::NO_ATTENTION_BLOCK, ps::HeadTensorComponent::IMAG);
    for (std::size_t i = 0; i < imaginary.size(); ++i)
      close(result.parameter_response[imaginary.begin + i],
            C{0, result.parameter_response[real_head.begin + i].real()}, 2e-9);
  }
}

TEST_CASE("PsiFormer complex kinetic rejects missing inverses separately from a total node",
          "[psiformer][complex_kinetic]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  pf::DirectComplexKineticExecutor executor(model, execution_plan);
  pf::DirectComplexValueExecutor value(model, execution_plan);
  auto workspace = executor.makeWorkspace(executor.requiredStorageBytes());
  auto value_workspace = value.makeWorkspace();
  const auto positions = model.cfg.configuration(0);
  const auto view = pf::GeometryPositionView::interleaved(positions.x.data(), model.ne);
  workspace->setPositions(view);
  value_workspace->setPositions(view);
  const std::vector<double> masses(model.ne, 1);
  auto parameters = model.p.flat_values();
  for (const auto& tensor : execution_plan.parameterTensors())
    if (tensor.role == ps::ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ps::ParameterRole::BACKFLOW_DOWN_WEIGHT)
      for (std::size_t feature = 0; feature < model.dim; ++feature)
        parameters[tensor.begin + feature * model.ndet * model.ne] = 0;
  model.p.set_flat_values(parameters);
  CHECK_FALSE(value.evaluate(*value_workspace).amplitude.isZero());
  CHECK_THROWS_WITH(executor.evaluate(*workspace, masses.data(), masses.size()),
      "complex determinant derivatives unavailable: required channel inverse missing");
  CHECK_THROWS(workspace->result());
  for (const auto& tensor : execution_plan.parameterTensors())
    if (tensor.role == ps::ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ps::ParameterRole::BACKFLOW_DOWN_WEIGHT)
      std::fill(parameters.begin() + tensor.begin, parameters.begin() + tensor.end, 0);
  model.p.set_flat_values(parameters);
  CHECK(value.evaluate(*value_workspace).amplitude.isZero());
  CHECK_THROWS_WITH(executor.evaluate(*workspace, masses.data(), masses.size()),
      "complex determinant derivatives unavailable at total wavefunction node");

  // Normal-valued entries produce an overflowing inverse in a channel whose weight narrows to zero.
  ps::determinant::ComplexDeterminantWorkspace determinant(2, 2, 0, 6, 2);
  const C matrices[]{1e-300, 1e-300, 1e-300, 1e-300 * (1 + 1e-12), 1, 0, 0, 1};
  const auto diagnostic = determinant.evaluate(matrices);
  CHECK_FALSE(diagnostic.amplitude.isZero());
  CHECK(diagnostic.singular_channels == 0);
  CHECK_FALSE(diagnostic.all_required_inverses_available);
  std::vector<C> gradient(48), laplacian(16), g(6), l(2), ratio(2);
  CHECK_THROWS_WITH(determinant.evaluateSpatial(matrices, gradient.data(), 6, laplacian.data(), 2,
      g.data(), l.data(), ratio.data()),
      "complex determinant derivatives unavailable: required channel inverse missing");

  // A narrowed-zero weight can still yield representable w*inverse orbital cotangents.
  const C regular[]{1e-200, 0, 0, 1e-200, 1, 0, 0, 1};
  CHECK(determinant.evaluateSpatial(regular, gradient.data(), 6, laplacian.data(), 2,
      g.data(), l.data(), ratio.data()).all_required_inverses_available);
  std::vector<C> root(72), gradient_root(6), laplacian_root{1, 0};
  std::vector<std::complex<long double>> scratch(12);
  determinant.fillSpatialRootMatrixAdjoints(gradient.data(), laplacian.data(),
      gradient_root.data(), laplacian_root.data(), root.data(), root.size(), scratch.data(), scratch.size());
  close(root[56], C{1e-200}, 1e-212);
  // A partial call in a FULL-capacity workspace must not leave a usable FULL root cache.
  determinant.evaluateSpatial(regular, gradient.data(), 3, laplacian.data(), 1,
      g.data(), l.data(), ratio.data());
  CHECK_THROWS_AS(determinant.fillSpatialRootMatrixAdjoints(gradient.data(), laplacian.data(),
      gradient_root.data(), laplacian_root.data(), root.data(), root.size(), scratch.data(), scratch.size()),
      std::invalid_argument);
}
