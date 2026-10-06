//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_spatial_executor.cpp
 * @brief Deterministic high-level tests for direct PsiFormer spatial evaluation.
 */

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "psiformer_test_utils.h"

#include <algorithm>
#include <string>
#include <vector>

namespace
{
using namespace qmcplusplus::testing::psiformer;

/// Store independently generated JAX float64 spatial observables.
struct SpatialGolden
{
  double sign;
  double logabs;
  std::vector<double> gradient;
  std::vector<double> lap_log;
  std::vector<double> lap_ratio;
};

/// Return the embedded LiH JAX reference for the deterministic fixture recipe.
SpatialGolden lihGolden()
{
  return {-1, -23.077667467509848,
          {0.13821734732102417, 0.79435815447019209, -2.7709716589173525,
           0.31363994259021311, 0.79768832588162031, 1.6304777638742229,
           -3.4590352892338929, -1.3614672567480892, 1.4157613451655826,
           -0.62679873678084308, 0.64331102535753593, -1.8005394170930145},
          {-12.948135315101172, -3.6943412044856432, -10.285466006833225,
           -5.1600794176296043},
          {-4.6197424679042385, -0.3012067871615427, 5.5374324029944031,
           -1.1114114933473269}};
}

/// Return the embedded separated-LiH-pair JAX reference for the fixture recipe.
SpatialGolden pairGolden()
{
  return {-1, -35.19505925284793,
          {-0.51039359144009766, -0.64181835029801526, -0.56554578886861506,
           0.11953688533544767, -0.40072217696523665, 0.62142613906272137,
           0.56114724072042799, -0.38668360913347499, 0.54183333491777408,
           0.46267479367736408, -0.56435288767847713, -0.49317880081085291,
           -0.20015882506933289, -0.80800066604879506, 0.38701482713969687,
           -0.74100861969047793, 0.27356357958271049, 0.0017890314830455352,
           0.36347922677028238, 0.42084405027643595, 0.50993799525183048,
           -0.70770775912668826, 1.0112844306971727, 0.6251745846552339},
          {-0.69870618499601422, -1.2805395515924511, -2.4548764662685221,
           -1.0182609103275793, -1.4358864737190578, -1.4079945406549548,
           -1.1268147848603813, -1.1579542485264198},
          {0.29356826727339702, -0.71950177521459202, -1.6968826640997654,
           -0.242473434222921, -0.59317736570465607, -0.78406053349161409,
           -0.55755116291232776, 0.75643548487104773}};
}

/// Compare one scalar at the deterministic spatial-derivative tolerance.
void checkClose(double actual, double expected, double relative = 2e-8, double absolute = 2e-8)
{
  CHECK(actual == Catch::Approx(expected).epsilon(relative).margin(absolute));
}

/// Evaluate direct log|psi| after copying one complete configuration.
double valueLogAbs(pf::DirectValueExecutor& executor,
                   pf::DirectValueWorkspace& workspace,
                   const std::vector<double>& positions,
                   std::size_t electron_count)
{
  workspace.setPositions(pf::GeometryPositionView::interleaved(positions.data(), electron_count));
  return executor.evaluate(workspace).logabs;
}

/// Build the same immutable typed plan used by the production shared state.
qmcplusplus::psiformer::PsiFormerExecutionPlan makePlan(const pf::PsiFormer& model)
{
  return qmcplusplus::psiformer::PsiFormerExecutionPlan::fromParameters(
      model.p, {model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet,
                model.dim, model.heads, 4});
}

/// Validate full/active modes, storage stability, JAX goldens, and finite differences.
void validateSystem(const std::string& system, const SpatialGolden& golden, bool finite_difference)
{
  GeneratedFiles files = generateFiles(system);
  pf::PsiFormer model(files.parameters, files.configuration);
  const pf::Tensor positions = model.cfg.configuration(0);
  const pf::GeometryPositionView position_view =
      pf::GeometryPositionView::interleaved(positions.x.data(), model.ne);

  const qmcplusplus::psiformer::PsiFormerExecutionPlan plan = makePlan(model);
  pf::DirectValueExecutor value_executor(model, plan);
  pf::DirectSpatialExecutor executor(model, value_executor, plan);
  std::unique_ptr<pf::DirectSpatialWorkspace> full_workspace =
      executor.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  std::unique_ptr<pf::DirectSpatialWorkspace> active_workspace =
      executor.makeWorkspace(pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT);
  CHECK(full_workspace->geometryStorageBytes() > 0);
  CHECK(active_workspace->geometryStorageBytes() > 0);
  CHECK(full_workspace->vectorStorageBytes() ==
        full_workspace->requiredStorageBytes());
  CHECK(active_workspace->vectorStorageBytes() ==
        active_workspace->requiredStorageBytes());
  full_workspace->setPositions(position_view);
  active_workspace->setPositions(position_view);
  const std::size_t full_storage   = full_workspace->storageFingerprint();
  const std::size_t active_storage = active_workspace->storageFingerprint();

  pf::EvaluationRequest oracle_request;
  oracle_request.spatial_derivatives    = pf::SpatialDerivativeRequest::FULL_VGL;
  oracle_request.parameter_derivatives  = pf::ParameterDerivativeRequest::NONE;
  oracle_request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  const pf::Result oracle = model.evaluate(positions, oracle_request);
  const pf::DirectSpatialResultView direct_view = executor.evaluateFull(*full_workspace);
  const std::vector<double> direct_gradient(direct_view.gradient.begin(), direct_view.gradient.end());
  const std::vector<double> direct_lap_log(direct_view.lap_log.begin(), direct_view.lap_log.end());
  const std::vector<double> direct_lap_ratio(direct_view.lap_ratio.begin(), direct_view.lap_ratio.end());

  CHECK(direct_view.sign == oracle.sign);
  CHECK(direct_view.sign == golden.sign);
  checkClose(direct_view.logabs, oracle.logabs, 3e-10, 3e-10);
  checkClose(direct_view.logabs, golden.logabs, 3e-9, 3e-9);
  REQUIRE(direct_gradient.size() == golden.gradient.size());
  REQUIRE(direct_lap_log.size() == golden.lap_log.size());
  REQUIRE(direct_lap_ratio.size() == golden.lap_ratio.size());
  for (std::size_t coordinate = 0; coordinate < direct_gradient.size(); ++coordinate)
  {
    checkClose(direct_gradient[coordinate], oracle.gradient[coordinate]);
    checkClose(direct_gradient[coordinate], golden.gradient[coordinate], 2e-7, 2e-7);
  }
  for (std::size_t electron = 0; electron < direct_lap_log.size(); ++electron)
  {
    checkClose(direct_lap_log[electron], oracle.lap_log[electron], 3e-7, 3e-7);
    checkClose(direct_lap_ratio[electron], oracle.lap_ratio[electron], 3e-7, 3e-7);
    checkClose(direct_lap_log[electron], golden.lap_log[electron], 2e-6, 2e-6);
    checkClose(direct_lap_ratio[electron], golden.lap_ratio[electron], 2e-6, 2e-6);
  }

  for (std::size_t electron = 0; electron < model.ne; ++electron)
  {
    const pf::DirectSpatialResultView active = executor.evaluateActive(*active_workspace, electron);
    REQUIRE(active.gradient.size() == 3);
    CHECK(active.lap_log.empty());
    CHECK(active.lap_ratio.empty());
    checkClose(active.logabs, direct_view.logabs, 3e-10, 3e-10);
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      checkClose(active.gradient[dimension], direct_gradient[3 * electron + dimension]);
  }
  CHECK(full_workspace->storageFingerprint() == full_storage);
  CHECK(active_workspace->storageFingerprint() == active_storage);
  CHECK(full_workspace->observedParameterVersion() == model.p.version());
  CHECK(active_workspace->observedParameterVersion() == model.p.version());

  if (!finite_difference)
    return;

  std::unique_ptr<pf::DirectValueWorkspace> value_workspace = value_executor.makeWorkspace();
  const double gradient_step = 2e-5;
  for (std::size_t coordinate : {std::size_t{0}, std::size_t{5}, positions.x.size() - 1})
  {
    std::vector<double> plus = positions.x;
    std::vector<double> minus = positions.x;
    plus[coordinate] += gradient_step;
    minus[coordinate] -= gradient_step;
    const double difference =
        (valueLogAbs(value_executor, *value_workspace, plus, model.ne) -
         valueLogAbs(value_executor, *value_workspace, minus, model.ne)) /
        (2 * gradient_step);
    checkClose(direct_gradient[coordinate], difference, 4e-5, 4e-5);
  }

  const double laplacian_step = 2e-4;
  double finite_laplacian = 0;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
  {
    std::vector<double> plus = positions.x;
    std::vector<double> minus = positions.x;
    plus[dimension] += laplacian_step;
    minus[dimension] -= laplacian_step;
    finite_laplacian +=
        (valueLogAbs(value_executor, *value_workspace, plus, model.ne) - 2 * direct_view.logabs +
         valueLogAbs(value_executor, *value_workspace, minus, model.ne)) /
        (laplacian_step * laplacian_step);
  }
  checkClose(direct_lap_log[0], finite_laplacian, 5e-4, 5e-4);
}

} // namespace

TEST_CASE("PsiFormer direct spatial LiH observables", "[wavefunction][psiformer]")
{
  validateSystem("lih", lihGolden(), true);
}

TEST_CASE("PsiFormer direct spatial separated LiH-pair observables", "[wavefunction][psiformer]")
{
  validateSystem("lih_pair", pairGolden(), false);
}

TEST_CASE("PsiFormer direct spatial supports canonical pseudo-LiH without same-spin alpha",
          "[wavefunction][psiformer][ecp]")
{
  GeneratedFiles files = generateFiles("lih_pp");
  pf::PsiFormer model(files.parameters, files.configuration);
  const pf::Tensor positions = model.cfg.configuration(0);
  const qmcplusplus::psiformer::PsiFormerExecutionPlan plan = makePlan(model);
  CHECK_FALSE(plan.hasParameter(qmcplusplus::psiformer::ParameterRole::CUSP_SAME_ALPHA));

  pf::DirectValueExecutor value_executor(model, plan);
  pf::DirectSpatialExecutor executor(model, value_executor, plan);
  std::unique_ptr<pf::DirectSpatialWorkspace> workspace =
      executor.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  workspace->setPositions(
      pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));

  pf::EvaluationRequest request;
  request.spatial_derivatives    = pf::SpatialDerivativeRequest::FULL_VGL;
  request.parameter_derivatives  = pf::ParameterDerivativeRequest::NONE;
  request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  const pf::Result oracle = model.evaluate(positions, request);
  const pf::DirectSpatialResultView direct = executor.evaluateFull(*workspace);
  CHECK(direct.sign == oracle.sign);
  checkClose(direct.logabs, oracle.logabs, 3e-10, 3e-10);
  REQUIRE(direct.gradient.size() == oracle.gradient.size());
  REQUIRE(direct.lap_log.size() == oracle.lap_log.size());
  REQUIRE(direct.lap_ratio.size() == oracle.lap_ratio.size());
  for (std::size_t coordinate = 0; coordinate < direct.gradient.size(); ++coordinate)
    checkClose(direct.gradient[coordinate], oracle.gradient[coordinate]);
  for (std::size_t electron = 0; electron < direct.lap_log.size(); ++electron)
  {
    checkClose(direct.lap_log[electron], oracle.lap_log[electron], 3e-7, 3e-7);
    checkClose(direct.lap_ratio[electron], oracle.lap_ratio[electron], 3e-7, 3e-7);
  }
}

TEST_CASE("PsiFormer direct spatial request validation", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  pf::PsiFormer model(files.parameters, files.configuration);
  const qmcplusplus::psiformer::PsiFormerExecutionPlan plan = makePlan(model);
  pf::DirectValueExecutor value_executor(model, plan);
  pf::DirectSpatialExecutor executor(model, value_executor, plan);
  std::unique_ptr<pf::DirectSpatialWorkspace> full =
      executor.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  std::unique_ptr<pf::DirectSpatialWorkspace> active =
      executor.makeWorkspace(pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT);
  const pf::Tensor positions = model.cfg.configuration(0);
  const pf::GeometryPositionView view =
      pf::GeometryPositionView::interleaved(positions.x.data(), model.ne);
  full->setPositions(view);
  active->setPositions(view);

  CHECK_THROWS_AS(executor.evaluateFull(*active), std::invalid_argument);
  CHECK_THROWS_AS(executor.evaluateActive(*full, 0), std::invalid_argument);
  CHECK_THROWS_AS(executor.evaluateActive(*active, model.ne), std::out_of_range);

  // Flat optimizer mutations retain the typed layout and are observed on the next call.
  const double original_parameter = model.p.flat_values()[127];
  model.p.set_flat_value(127, original_parameter + 1e-3);
  const pf::DirectSpatialResultView changed = executor.evaluateFull(*full);
  pf::EvaluationRequest request;
  request.spatial_derivatives    = pf::SpatialDerivativeRequest::FULL_VGL;
  request.parameter_derivatives  = pf::ParameterDerivativeRequest::NONE;
  request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  const pf::Result changed_oracle = model.evaluate(positions, request);
  CHECK(changed.parameter_version == 1);
  CHECK(full->observedParameterVersion() == 1);
  checkClose(changed.logabs, changed_oracle.logabs, 3e-10, 3e-10);
  for (std::size_t coordinate = 0; coordinate < changed.gradient.size(); ++coordinate)
    checkClose(changed.gradient[coordinate], changed_oracle.gradient[coordinate]);
  for (std::size_t electron = 0; electron < changed.lap_log.size(); ++electron)
    checkClose(changed.lap_log[electron], changed_oracle.lap_log[electron], 3e-7, 3e-7);
}
