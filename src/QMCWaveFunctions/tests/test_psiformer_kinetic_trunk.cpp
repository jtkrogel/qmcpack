//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
//////////////////////////////////////////////////////////////////////////////////////
/** @file test_psiformer_kinetic_trunk.cpp
 * @brief Retained real kinetic feature-jet reverse, storage and ticket contracts.
 */
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerKineticExecutor.h"
#include "psiformer_value_test_utils.h"

namespace
{
namespace ps = qmcplusplus::psiformer;
using namespace qmcplusplus::testing::psiformer::value;

/// Keep one attention block to exercise restoration of odd reverse ping-pong identities.
pf::PsiFormer oddBlockModel()
{
  auto model = tinyModel();
  const auto execution_plan = plan(model);
  std::vector<double> values;
  std::vector<pf::Layout> layouts;
  for (const auto& tensor : execution_plan.parameterTensors())
    if (tensor.attention_block != 1)
    {
      const std::size_t begin = values.size();
      const auto& original = model.p.flat_values();
      values.insert(values.end(), original.begin() + tensor.begin, original.begin() + tensor.end);
      layouts.push_back({tensor.module, tensor.name, tensor.shape, begin, values.size()});
    }
  return pf::PsiFormer(pf::Parameters(std::move(values), std::move(layouts)),
                      model.cfg, model.ndet, model.dim, model.heads, 1);
}

/// Inspect retained forward planes without exposing reverse scratch in production.
struct InspectableTape : pf::DirectKineticTrunkWorkspace
{
  /// Bind the same model/owner and geometry as the prepared factory workspace.
  InspectableTape(const ps::PsiFormerExecutionPlan& execution_plan, const pf::PsiFormer& model,
                  const pf::DirectKineticExecutor* owner)
      : DirectKineticTrunkWorkspace(execution_plan,
          pf::GeometryPositionView::interleaved(model.cfg.nuclei.x.data(), model.cfg.nuclei.shape[0]))
  { owner_ = owner; }

  /// Snapshot all forward tensors, including QKV and probabilities, between reverse passes.
  std::vector<double> snapshot() const
  {
    std::vector<double> result;
    const auto append = [&result](const pf::DirectTraceJetBuffer& jet) {
      for (const auto* plane : {&jet.value, &jet.gradient, &jet.laplacian})
        result.insert(result.end(), plane->begin(), plane->end());
    };
    append(raw_features_);
    for (const auto* tape : {&features_, &queries_, &keys_, &values_, &logits_, &attention_,
                            &contexts_, &residuals_, &hidden_pre_, &hidden_, &update_pre_, &updates_})
      for (const auto& jet : *tape)
        append(jet);
    result.insert(result.end(), cusp_gradient_.begin(), cusp_gradient_.end());
    result.insert(result.end(), cusp_laplacian_.begin(), cusp_laplacian_.end());
    return result;
  }
};

/// Contract a FULL feature/cusp jet ticket with independent test-owned root seeds.
double objective(const pf::DirectKineticTrunkResultView& tape,
                 const pf::DirectTraceJetBuffer& seeds, const pf::DirectTraceJetBuffer& cusp)
{
  double value = cusp.value[0] * tape.cusp_value;
  const auto dot = [&value](const auto& left, const auto& right) {
    for (std::size_t i = 0; i < left.size(); ++i)
      value += left[i] * right[i];
  };
  dot(seeds.value, tape.features->value);
  dot(seeds.gradient, tape.features->gradient);
  dot(seeds.laplacian, tape.features->laplacian);
  dot(cusp.gradient, tape.cusp_gradient);
  dot(cusp.laplacian, tape.cusp_laplacian);
  return value;
}

/// Set all three seed planes, including nonzero mixed gradient and Laplacian weights.
void fillSeeds(pf::DirectTraceJetBuffer& seeds, double scale)
{
  for (auto* plane : {&seeds.value, &seeds.gradient, &seeds.laplacian})
    for (std::size_t i = 0; i < plane->size(); ++i)
      (*plane)[i] = scale * std::sin(0.17 * (i + 1));
}
} // namespace

TEST_CASE("PsiFormer kinetic trunk retains immutable jet tape for seeded reverse", "[psiformer][kinetic_trunk]")
{
  auto model = oddBlockModel();
  const auto execution_plan = plan(model);
  pf::DirectKineticExecutor executor(model, execution_plan, pf::DirectKineticPreparation::REAL_TRUNK_ONLY);
  InspectableTape workspace(execution_plan, model, &executor);
  const auto configuration = model.cfg.configuration(0);
  const auto positions = pf::GeometryPositionView::interleaved(configuration.x.data(), model.ne);
  workspace.setPositions(positions);
  auto tape = executor.evaluateTrunk(workspace);
  const auto snapshot = workspace.snapshot();
  const auto fingerprint = workspace.storageFingerprint();
  const auto bytes = workspace.vectorStorageBytes();
  CHECK(bytes == workspace.requiredStorageBytes());
  CHECK(workspace.parameterStorageBytes() == model.p.size() * sizeof(double));
  CHECK_THROWS(executor.makeTrunkWorkspace(bytes - 1));
  CHECK_THROWS(executor.makeTrunkWorkspace(0));
  CHECK_THROWS(executor.makeTrunkWorkspace(std::numeric_limits<std::size_t>::max()));
  CHECK(executor.makeTrunkWorkspace(bytes)->vectorStorageBytes() == bytes);
  CHECK_THROWS(executor.makeWorkspace());
  pf::DirectKineticExecutor complete(model, execution_plan);
  const auto full = complete.makeWorkspace();
  CHECK(full->vectorStorageBytes() == full->requiredStorageBytes());
  CHECK(full->vectorStorageBytes() > bytes);

  pf::DirectTraceJetBuffer seeds(model.ne * model.dim, 3 * model.ne, model.ne);
  pf::DirectTraceJetBuffer cusp(1, 3 * model.ne, model.ne);
  fillSeeds(seeds, 0.03);
  fillSeeds(cusp, 0.07);
  const auto first = executor.reverseTrunk(workspace, tape, seeds, cusp);
  const std::vector<double> response(first.begin(), first.end());
  fillSeeds(seeds, -0.06);
  fillSeeds(cusp, -0.14);
  const auto second = executor.reverseTrunk(workspace, tape, seeds, cusp);
  for (std::size_t i = 0; i < model.p.size(); ++i)
    CHECK(second[i] == Catch::Approx(-2 * response[i]).margin(1e-12));
  CHECK(workspace.snapshot() == snapshot);
  CHECK(workspace.storageFingerprint() == fingerprint);
  CHECK(workspace.vectorStorageBytes() == bytes);

  // Independent parameter finite differences of the complete seeded feature/cusp objective.
  const auto original = model.p.flat_values();
  for (const auto& descriptor : execution_plan.parameterTensors())
    for (const std::size_t index : {descriptor.begin, descriptor.end - 1})
    {
      auto values = original;
      values[index] += 1e-5;
      model.p.set_flat_values(values);
      const double plus = objective(executor.evaluateTrunk(workspace), seeds, cusp);
      values[index] -= 2e-5;
      model.p.set_flat_values(values);
      const double minus = objective(executor.evaluateTrunk(workspace), seeds, cusp);
      CHECK((plus - minus) / 2e-5 == Catch::Approx(-2 * response[index]).margin(2e-6));
    }
  model.p.set_flat_values(original);
  CHECK_THROWS(executor.reverseTrunk(workspace, tape, seeds, cusp));
  tape = executor.evaluateTrunk(workspace);
  pf::DirectKineticExecutor foreign(model, execution_plan, pf::DirectKineticPreparation::REAL_TRUNK_ONLY);
  CHECK_THROWS(foreign.reverseTrunk(workspace, tape, seeds, cusp));
  pf::DirectTraceJetBuffer malformed(1, 1, 1);
  const auto generation = workspace.resultGeneration();
  CHECK_THROWS(executor.reverseTrunk(workspace, tape, malformed, cusp));
  CHECK(workspace.resultGeneration() != generation);
  CHECK_THROWS(workspace.trunkParameterResponse());
  CHECK_NOTHROW(executor.reverseTrunk(workspace, tape, seeds, cusp));
  workspace.setPosition(0, 0, std::numeric_limits<double>::quiet_NaN());
  CHECK_THROWS(executor.evaluateTrunk(workspace));
  CHECK_THROWS(executor.reverseTrunk(workspace, tape, seeds, cusp));
  workspace.setPositions(positions);
  CHECK_NOTHROW(executor.evaluateTrunk(workspace));
}

TEST_CASE("PsiFormer paired metadata admits only the numerical kinetic trunk", "[psiformer][kinetic_trunk]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  CHECK_THROWS(pf::DirectKineticExecutor(model, execution_plan));
  pf::DirectKineticExecutor executor(model, execution_plan, pf::DirectKineticPreparation::REAL_TRUNK_ONLY);
  auto workspace = executor.makeTrunkWorkspace(16 * 1024 * 1024);
  const auto configuration = model.cfg.configuration(0);
  workspace->setPositions(pf::GeometryPositionView::interleaved(configuration.x.data(), model.ne));
  CHECK_NOTHROW(executor.evaluateTrunk(*workspace));
  CHECK_THROWS(executor.makeWorkspace());
}
