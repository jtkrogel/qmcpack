//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerBatchExecutor.h"
#include "psiformer_value_test_utils.h"

namespace ps = qmcplusplus::psiformer;
namespace fixture = qmcplusplus::testing::psiformer::value;

TEST_CASE("Residual pair CPU value VGL and score agree", "[psiformer][residual_pair]")
{
  auto original = fixture::tinyModel(true);
  auto values = original.p.flat_values();
  auto layouts = original.p.layouts;
  std::vector<std::size_t> lengths;
  for (auto& layout : layouts)
    if (layout.module.find("electronic_cusp_asymptotic") != std::string::npos)
    {
      layout.module = "neural_network_wave_function/~/residual_pair_v1";
      layout.name = layout.name == "same_alpha" ? "same_log_length" : "opposite_log_length";
      values[layout.begin] = 0;
      lengths.push_back(layout.begin);
    }
  pf::PsiFormer model(pf::Parameters(values, layouts), original.cfg, original.ndet, original.dim,
                       original.heads, original.blocks);
  auto environment = fixture::environment(true, false);
  environment.residual_pair = std::make_shared<const ps::ResidualPairGeometry>(
      ps::ResidualPairMetadata{1, 1.2, .9, 3}, environment.lattice_vectors);
  const auto plan = ps::PsiFormerExecutionPlan::fromParameters(model.p,
      {2,2,2,model.ndet,model.dim,model.heads,model.blocks}, environment);
  pf::DirectValueExecutor value(model, plan);
  pf::DirectSpatialExecutor spatial(model, value, plan);
  pf::DirectScoreExecutor score(model, plan);
  auto vw = value.makeWorkspace();
  auto sw = spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  auto ow = score.makeWorkspace();
  const auto positions = model.cfg.configuration(0).x;
  auto coordinates = positions;
  auto set = [&] {
    const auto view = pf::GeometryPositionView::interleaved(coordinates.data(), model.ne);
    vw->setPositions(view); sw->setPositions(view); ow->setPositions(view);
  };
  set();
  const auto zero_value = value.evaluate(*vw).logabs;
  const auto zero_spatial = spatial.evaluateFull(*sw);
  const std::vector<double> zero_gradient(zero_spatial.gradient.begin(), zero_spatial.gradient.end());
  const std::vector<double> zero_lap(zero_spatial.lap_log.begin(), zero_spatial.lap_log.end());
  const auto zero_score = score.evaluate(*ow);
  for (auto index : lengths) CHECK(std::abs(zero_score.parameter_score[index]) > 0);
  auto baseline_values = original.p.flat_values();
  for (auto index : lengths) baseline_values[index] = 0;
  original.p.set_flat_values(baseline_values);
  const auto baseline_plan = fixture::plan(original, true);
  pf::DirectValueExecutor base_value(original, baseline_plan);
  pf::DirectSpatialExecutor base_spatial(original, base_value, baseline_plan);
  auto base_work = base_spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  base_work->setPositions(pf::GeometryPositionView::interleaved(coordinates.data(), model.ne));
  const auto base = base_spatial.evaluateFull(*base_work);
  CHECK(base.logabs == zero_value);
  for (std::size_t lane = 0; lane < zero_gradient.size(); ++lane) CHECK(base.gradient[lane] == zero_gradient[lane]);
  for (std::size_t e = 0; e < zero_lap.size(); ++e) CHECK(base.lap_log[e] == zero_lap[e]);
  for (auto index : lengths) values[index] = index == lengths.front() ? .3 : -.2;
  model.p.set_flat_values(values);
  set();
  const auto active_value = value.evaluate(*vw).logabs;
  CHECK(active_value != zero_value);
  const auto active = spatial.evaluateFull(*sw);
  const std::vector<double> gradient(active.gradient.begin(), active.gradient.end());
  const std::vector<double> lap(active.lap_log.begin(), active.lap_log.end());
  const auto active_score = score.evaluate(*ow);
  const std::vector<double> scores(active_score.parameter_score.begin(), active_score.parameter_score.end());
  CHECK(active_score.logabs == Catch::Approx(active_value).margin(1e-12));
  const double h = 2e-5;
  std::vector<double> fd_lap(model.ne, 0);
  for (std::size_t lane = 0; lane < coordinates.size(); ++lane)
  {
    coordinates[lane] = positions[lane]+h; set(); const double plus = value.evaluate(*vw).logabs;
    coordinates[lane] = positions[lane]-h; set(); const double minus = value.evaluate(*vw).logabs;
    coordinates[lane] = positions[lane];
    CHECK(gradient[lane] == Catch::Approx((plus-minus)/(2*h)).epsilon(2e-6).margin(2e-6));
    fd_lap[lane/3] += (plus+minus-2*active_value)/(h*h);
  }
  for (std::size_t e = 0; e < model.ne; ++e)
    CHECK(lap[e] == Catch::Approx(fd_lap[e]).epsilon(2e-3).margin(2e-3));
  set();
  for (auto index : lengths)
  {
    auto shifted = values; shifted[index] += h; model.p.set_flat_values(shifted);
    const double plus = value.evaluate(*vw).logabs;
    shifted[index] -= 2*h; model.p.set_flat_values(shifted);
    const double minus = value.evaluate(*vw).logabs;
    CHECK(scores[index] == Catch::Approx((plus-minus)/(2*h)).epsilon(2e-6).margin(2e-6));
    model.p.set_flat_values(values);
  }
  auto translated = positions;
  for (std::size_t d = 0; d < 3; ++d) translated[d] += environment.lattice_vectors[0][d];
  coordinates = translated; set();
  CHECK(value.evaluate(*vw).logabs == Catch::Approx(active_value).margin(1e-11));
}
