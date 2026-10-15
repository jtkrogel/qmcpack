//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file test_psiformer_score_trunk.cpp
 * @brief Immutable real score-trunk tape, arbitrary cotangents and lifecycle tests.
 */
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"
#include "psiformer_value_test_utils.h"

namespace
{
namespace ps = qmcplusplus::psiformer;
using namespace qmcplusplus::testing::psiformer::value;

/// Test-only access snapshots every forward tape buffer, never numerical reverse scratch.
struct InspectableTape : pf::DirectScoreTrunkWorkspace
{
  /// Reuse exactly the numerical base preparation with the supplied identity.
  InspectableTape(const ps::PsiFormerExecutionPlan& plan, const pf::PsiFormer& model,
                  const pf::DirectScoreExecutor* owner)
      : DirectScoreTrunkWorkspace(plan,
          pf::GeometryPositionView::interleaved(model.cfg.nuclei.x.data(), model.cfg.nuclei.shape[0]), {}, owner)
  {}

  /// Materialize a test-owned baseline to detect any hidden QKV/probability mutation.
  std::vector<double> snapshot() const
  {
    std::vector<double> result;
    for (const auto* buffer : {&raw_features_, &features_, &queries_, &keys_, &values_,
             &attention_weights_, &contexts_, &residuals_, &hidden_, &updates_})
      result.insert(result.end(), buffer->begin(), buffer->end());
    return result;
  }
};

/// Compare tiny fixture derivatives independently propagated through Native's test graph.
void close(double actual, double expected, double tolerance = 2e-11)
{
  CHECK(actual == Catch::Approx(expected).margin(tolerance));
}
} // namespace

TEST_CASE("PsiFormer score trunk reverses successive real cotangents without tape mutation",
          "[wavefunction][psiformer][score_trunk]")
{
  auto model = tinyModel();
  const auto execution_plan = plan(model);
  pf::DirectScoreExecutor executor(model, execution_plan);
  InspectableTape workspace(execution_plan, model, &executor);
  const auto positions = model.cfg.configuration(0);
  workspace.setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  const auto tape = executor.evaluateTrunk(workspace);
  const auto baseline = workspace.snapshot();
  const auto fingerprint = workspace.storageFingerprint();
  CHECK(workspace.vectorStorageBytes() == workspace.requiredStorageBytes());
  auto complete = executor.makeWorkspace();
  CHECK(complete->vectorStorageBytes() == complete->requiredStorageBytes());
  CHECK(complete->vectorStorageBytes() - workspace.vectorStorageBytes() ==
        sizeof(double) * 3 * model.ndet * model.ne * model.ne + pf::determinantStorageRequirement(model.ndet, model.ne));
  std::vector<double> first(tape.feature_count), second(first.size()), combined(first.size());
  for (std::size_t i = 0; i < first.size(); ++i)
  {
    first[i] = .1 * std::sin(static_cast<double>(i + 1));
    second[i] = .2 * std::cos(static_cast<double>(i + 2));
    combined[i] = first[i] + second[i];
  }
  const auto coordinates = pf::coordinates(positions, {});
  auto features = model.embedding(coordinates);
  for (std::size_t block = 0; block < model.blocks; ++block)
    features = model.attention_block(features, static_cast<int>(block));
  const auto objective = pf::sum_all(pf::mul(features, pf::constant(pf::Tensor(features->value.shape, first))));
  const auto expected = model.p.flat_gradient(pf::backward(objective));
  const auto initial = executor.reverseTrunk(workspace, tape, first.data(), first.size());
  const std::vector<double> first_score(initial.parameter_score.begin(), initial.parameter_score.end());
  REQUIRE(first_score.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
    close(first_score[i], expected[i]);
  const auto next = executor.reverseTrunk(workspace, tape, second.data(), second.size());
  const std::vector<double> second_score(next.parameter_score.begin(), next.parameter_score.end());
  CHECK(next.result_generation != initial.result_generation);
  CHECK(next.forward_generation == initial.forward_generation);
  const auto sum = executor.reverseTrunk(workspace, tape, combined.data(), combined.size());
  for (std::size_t i = 0; i < first_score.size(); ++i)
    close(sum.parameter_score[i], first_score[i] + second_score[i]);
  CHECK(workspace.snapshot() == baseline);
  CHECK(workspace.storageFingerprint() == fingerprint);
  const auto original = model.p.flat_values();
  for (const auto role : {ps::ParameterRole::ELECTRON_EMBEDDING_WEIGHT, ps::ParameterRole::UPDATE_HIDDEN_WEIGHT})
  {
    const auto& descriptor = execution_plan.parameter(role,
        role == ps::ParameterRole::ELECTRON_EMBEDDING_WEIGHT ? ps::NO_ATTENTION_BLOCK : 0);
    const std::size_t parameter = descriptor.begin + 1;
    auto values = original;
    const double step = 1e-5;
    values[parameter] += step;
    model.p.set_flat_values(values);
    const auto plus = executor.evaluateTrunk(workspace);
    double plus_value = std::inner_product(first.begin(), first.end(), plus.features, 0.0);
    values[parameter] -= 2 * step;
    model.p.set_flat_values(values);
    const auto minus = executor.evaluateTrunk(workspace);
    double minus_value = std::inner_product(first.begin(), first.end(), minus.features, 0.0);
    close(first_score[parameter], (plus_value - minus_value) / (2 * step), 2e-8);
    model.p.set_flat_values(original);
  }
}

TEST_CASE("PsiFormer score trunk rejects stale identities and failures then recovers",
          "[wavefunction][psiformer][score_trunk]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  CHECK_THROWS_AS(pf::DirectScoreExecutor(model, execution_plan), std::invalid_argument);
  pf::DirectScoreExecutor executor(model, execution_plan, pf::DirectScorePreparation::REAL_TRUNK_ONLY);
  CHECK_THROWS(executor.makeWorkspace());
  auto workspace = executor.makeTrunkWorkspace();
  const auto positions = model.cfg.configuration(0);
  const auto view = pf::GeometryPositionView::interleaved(positions.x.data(), model.ne);
  workspace->setPositions(view);
  const auto tape = executor.evaluateTrunk(*workspace);
  std::vector<double> seed(tape.feature_count, .1);
  executor.reverseTrunk(*workspace, tape, seed.data(), seed.size());
  const auto generation = workspace->resultGeneration();
  CHECK_THROWS(executor.reverseTrunk(*workspace, tape, seed.data(), seed.size() - 1));
  CHECK(workspace->resultGeneration() != generation);
  CHECK_THROWS(workspace->trunkParameterScore());
  executor.reverseTrunk(*workspace, tape, seed.data(), seed.size());
  CHECK(workspace->trunkParameterScore().size == model.p.size());
  auto values = model.p.flat_values();
  values[1] += .01;
  model.p.set_flat_values(values);
  CHECK_THROWS(executor.reverseTrunk(*workspace, tape, seed.data(), seed.size()));
  const auto updated = executor.evaluateTrunk(*workspace);
  CHECK(updated.forward_generation != tape.forward_generation);
  CHECK(updated.parameter_version == model.p.version());
  CHECK_THROWS(executor.reverseTrunk(*workspace, tape, seed.data(), seed.size()));
  executor.reverseTrunk(*workspace, updated, seed.data(), seed.size());
  pf::DirectScoreExecutor foreign(model, execution_plan, pf::DirectScorePreparation::REAL_TRUNK_ONLY);
  CHECK_THROWS(foreign.reverseTrunk(*workspace, updated, seed.data(), seed.size()));
  workspace->setPosition(0, 0, std::numeric_limits<double>::quiet_NaN());
  CHECK_THROWS(executor.evaluateTrunk(*workspace));
  CHECK_THROWS(executor.reverseTrunk(*workspace, updated, seed.data(), seed.size()));
  workspace->setPositions(view);
  const auto recovered = executor.evaluateTrunk(*workspace);
  executor.reverseTrunk(*workspace, recovered, seed.data(), seed.size());
  CHECK(workspace->trunkParameterScore().size == model.p.size());
}
