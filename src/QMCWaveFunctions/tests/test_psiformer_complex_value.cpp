//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file test_psiformer_complex_value.cpp
 * @brief Complete G-free complex CPU VALUE tests using the shared tiny real trunk.
 */
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexValueExecutor.h"
#include "psiformer_value_test_utils.h"

namespace
{
namespace ps = qmcplusplus::psiformer;
using namespace qmcplusplus::testing::psiformer::value;
using C = std::complex<double>;

/// Materialize tiny generated amplitudes solely for independent finite differences.
C value(const pf::DirectComplexValueResult& result)
{
  return result.amplitude.isZero() ? C{} : result.amplitude.phase * std::exp(result.amplitude.log_abs);
}

/// Expand a small complex determinant independently of the production LU kernel.
C permutationDeterminant(const C* matrix, std::size_t n)
{
  std::vector<std::size_t> permutation(n);
  std::iota(permutation.begin(), permutation.end(), 0);
  C sum{};
  do
  {
    C term{1};
    std::size_t inversions = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
      term *= matrix[i * n + permutation[i]];
      for (std::size_t j = 0; j < i; ++j)
        inversions += permutation[j] > permutation[i];
    }
    sum += inversions % 2 ? -term : term;
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  return sum;
}

/// Independent scalar head/envelope matrices, paired channel sum and cusp factor.
struct Reference
{
  std::vector<C> matrices;
  std::vector<double> envelopes;
  C sum{};
  double cusp = 0;
};

/// Consume already-qualified real trunk features, independently projecting both heads.
Reference reference(const pf::PsiFormer& model, const ps::PsiFormerExecutionPlan& execution_plan,
                    const pf::DirectValueTrunkResultView& trunk)
{
  const std::size_t n = model.ne, nuclei = model.cfg.nuclei.shape[0], channels = model.ndet * n;
  Reference result;
  result.matrices.resize(model.ndet * n * n);
  result.envelopes.resize(result.matrices.size());
  result.cusp = trunk.cusp;
  const auto& values = model.p.flat_values();
  const auto& distances = trunk.geometry->electronNucleusPairs().distances();
  for (std::size_t row = 0; row < n; ++row)
  {
    const bool up = row < model.cfg.nup;
    const auto weight_role = up ? ps::ParameterRole::BACKFLOW_UP_WEIGHT : ps::ParameterRole::BACKFLOW_DOWN_WEIGHT;
    const auto& real = execution_plan.parameter(weight_role);
    const auto& imaginary = execution_plan.parameter(weight_role, ps::NO_ATTENTION_BLOCK, ps::HeadTensorComponent::IMAG);
    const auto& pi = execution_plan.parameter(up ? ps::ParameterRole::ENVELOPE_PI_UP : ps::ParameterRole::ENVELOPE_PI_DOWN);
    const auto& zeta = execution_plan.parameter(up ? ps::ParameterRole::ENVELOPE_ZETA_UP : ps::ParameterRole::ENVELOPE_ZETA_DOWN);
    for (std::size_t d = 0; d < model.ndet; ++d)
      for (std::size_t orbital = 0; orbital < n; ++orbital)
      {
        const std::size_t channel = d * n + orbital, element = (d * n + row) * n + orbital;
        C projection{};
        for (std::size_t feature = 0; feature < model.dim; ++feature)
        {
          const std::size_t offset = feature * channels + channel;
          projection += C{values[real.begin + offset], values[imaginary.begin + offset]} *
              trunk.features[row * model.dim + feature];
        }
        double envelope = 0;
        for (std::size_t nucleus = 0; nucleus < nuclei; ++nucleus)
        {
          const std::size_t offset = channel * nuclei + nucleus;
          envelope += values[pi.begin + offset] *
              std::exp(-std::abs(values[zeta.begin + offset] * distances[row * nuclei + nucleus]));
        }
        result.envelopes[element] = envelope;
        result.matrices[element] = envelope * projection;
      }
  }
  for (std::size_t d = 0; d < model.ndet; ++d)
    result.sum += permutationDeterminant(result.matrices.data() + d * n * n, n);
  return result;
}
} // namespace

TEST_CASE("PsiFormer complex CPU value specializes to the existing real periodic amplitude",
          "[wavefunction][psiformer][complex_value]")
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
  pf::DirectComplexValueExecutor complex(model, execution_plan);
  auto workspace = complex.makeWorkspace();
  const auto positions = model.cfg.configuration(0);
  workspace->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  const auto result = complex.evaluate(*workspace);

  auto real_model = tinyModel(true, false);
  pf::DirectValueExecutor real(real_model, plan(real_model, true));
  auto real_workspace = real.makeWorkspace();
  real_workspace->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  const auto expected = real.evaluate(*real_workspace);
  REQUIRE_FALSE(result.amplitude.isZero());
  CHECK(result.amplitude.phase.real() == expected.sign);
  CHECK(result.amplitude.phase.imag() == 0);
  CHECK(result.amplitude.log_abs == Catch::Approx(expected.logabs).margin(2e-11));
  CHECK(result.parameter_version == model.p.version());
  CHECK_THROWS_AS(pf::DirectComplexValueExecutor(real_model, plan(real_model, true)), std::invalid_argument);
}

TEST_CASE("PsiFormer complex CPU value matches scalar head and permutation references",
          "[wavefunction][psiformer][complex_value]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  const auto positions = model.cfg.configuration(0);
  pf::DirectValueExecutor trunk(model, execution_plan, pf::DirectValuePreparation::REAL_TRUNK_ONLY);
  auto trunk_workspace = trunk.makeTrunkWorkspace();
  trunk_workspace->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  const auto features = trunk.evaluateTrunk(*trunk_workspace);
  const auto expected = reference(model, execution_plan, features);
  pf::DirectComplexValueExecutor complex(model, execution_plan);
  auto workspace = complex.makeWorkspace();
  workspace->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  const auto result = complex.evaluate(*workspace);
  REQUIRE(std::abs(expected.sum) > 0);
  REQUIRE_FALSE(result.amplitude.isZero());
  CHECK(result.amplitude.log_abs == Catch::Approx(std::log(std::abs(expected.sum)) + expected.cusp).margin(2e-11));
  CHECK(std::abs(result.amplitude.phase - expected.sum / std::abs(expected.sum)) < 2e-11);
  CHECK(std::abs(result.amplitude.phase.imag()) > 1e-4);
  CHECK(result.singular_channels == 0);
  CHECK(result.cancellation_ratio > 0);
  CHECK(result.evaluation_generation == workspace->evaluationGeneration());

  // A real coordinate in the imaginary head changes one determinant column.
  // Multilinearity gives its exact derivative by column replacement, without LU.
  const auto& imaginary = execution_plan.parameter(ps::ParameterRole::BACKFLOW_UP_WEIGHT,
      ps::NO_ATTENTION_BLOCK, ps::HeadTensorComponent::IMAG);
  std::vector<C> derivative(expected.matrices.begin(), expected.matrices.begin() + model.ne * model.ne);
  for (std::size_t row = 0; row < model.ne; ++row)
    derivative[row * model.ne] = row < model.cfg.nup ?
        C{0, features.features[row * model.dim] * expected.envelopes[row * model.ne]} : C{};
  const C exact_log_direction = permutationDeterminant(derivative.data(), model.ne) / expected.sum;
  CHECK(std::abs(exact_log_direction.imag()) > 1e-4);
  const double original = model.p.flat_values()[imaginary.begin], step = 1e-5;
  model.p.set_flat_value(imaginary.begin, original + step);
  const auto plus = complex.evaluate(*workspace);
  model.p.set_flat_value(imaginary.begin, original - step);
  const auto minus = complex.evaluate(*workspace);
  const C finite_difference = (value(plus) - value(minus)) / (2 * step * value(result));
  CHECK(std::abs(finite_difference - exact_log_direction) < 2e-7 * (1 + std::abs(exact_log_direction)));
  CHECK(minus.parameter_version == model.p.version());
  CHECK(minus.parameter_version != result.parameter_version);
  CHECK(minus.evaluation_generation != result.evaluation_generation);

  // Explicit owned-byte accounting includes current complex primitive scratch,
  // without a dummy full real-head preparation or a second neural trunk.
  ps::ComplexOrbitalHeadDescriptor descriptor{model.ne, model.cfg.nup, model.ndet, model.dim,
      ps::ComplexOrbitalStructure::FULL, ps::ComplexOrbitalPhase::NONE, {}};
  const auto feature_layout = ps::makeSpatialJetLayout(1, model.ne * model.dim, model.ne, ps::SpatialJetMode::VALUE_ONLY);
  const auto orbital_layout = ps::makeSpatialJetLayout(1, model.ndet * model.ne * model.ne,
      model.ne, ps::SpatialJetMode::VALUE_ONLY);
  ps::ComplexOrbitalHeadWorkspace head(descriptor, feature_layout, orbital_layout);
  ps::determinant::ComplexDeterminantWorkspace determinant(model.ndet, model.ne);
  const std::size_t expected_bytes = trunk_workspace->vectorStorageBytes() +
      model.ndet * model.ne * model.ne * sizeof(double) + head.storageBytes() + determinant.storageBytes();
  CHECK(workspace->vectorStorageBytes() == expected_bytes);
  const auto fingerprint = workspace->storageFingerprint();
  for (int repeat = 0; repeat < 3; ++repeat)
    complex.evaluate(*workspace);
  CHECK(workspace->storageFingerprint() == fingerprint);
  CHECK(workspace->vectorStorageBytes() == expected_bytes);
  std::cout << "complex_value Ne=" << model.ne << " D=" << model.ndet << " F=" << model.dim
            << " owned_numeric_bytes=" << expected_bytes << '\n';
}

TEST_CASE("PsiFormer complex CPU value preserves nodes and invalidates failed generations",
          "[wavefunction][psiformer][complex_value]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  pf::DirectComplexValueExecutor executor(model, execution_plan);
  auto workspace = executor.makeWorkspace();
  auto positions = model.cfg.configuration(0);
  workspace->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  const auto first = executor.evaluate(*workspace);
  workspace->setPosition(0, 0, std::numeric_limits<double>::quiet_NaN());
  CHECK_THROWS(executor.evaluate(*workspace));
  CHECK_THROWS_AS(workspace->result(), std::logic_error);
  CHECK(workspace->observedParameterVersion() == std::numeric_limits<std::size_t>::max());
  CHECK(workspace->evaluationGeneration() != first.evaluation_generation);
  workspace->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  executor.evaluate(*workspace);
  pf::DirectComplexValueExecutor other(model, execution_plan);
  CHECK_THROWS_AS(other.evaluate(*workspace), std::invalid_argument);
  CHECK_THROWS_AS(workspace->result(), std::logic_error);
  for (std::size_t dim = 0; dim < 3; ++dim)
    positions.x[3 + dim] = positions.x[dim];
  workspace->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  const auto node = executor.evaluate(*workspace);
  CHECK(node.amplitude.isZero());
  CHECK(node.amplitude.log_abs == -std::numeric_limits<double>::infinity());
  CHECK(node.cancellation_ratio == 0);
  CHECK(node.parameter_version == model.p.version());
}
