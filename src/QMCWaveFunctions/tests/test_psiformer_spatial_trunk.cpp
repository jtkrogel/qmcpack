//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file test_psiformer_spatial_trunk.cpp
 * @brief Shared real spatial trunk parity, storage and publication tests.
 */
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "psiformer_value_test_utils.h"

namespace
{
namespace ps = qmcplusplus::psiformer;
using namespace qmcplusplus::testing::psiformer::value;

/// Compare independently propagated real spatial jets at the tiny reference tolerance.
void close(double actual, double expected)
{
  CHECK(actual == Catch::Approx(expected).epsilon(2e-11).margin(2e-11));
}

/// Check the exact legacy head/output storage removed by head-free preparation.
void checkStorage(pf::DirectSpatialExecutor& executor, pf::DirectSpatialMode mode, const pf::PsiFormer& model)
{
  auto complete = executor.makeWorkspace(mode);
  auto trunk = executor.makeTrunkWorkspace(mode);
  CHECK(complete->vectorStorageBytes() == complete->requiredStorageBytes());
  CHECK(trunk->vectorStorageBytes() == trunk->requiredStorageBytes());
  const std::size_t gradients = mode == pf::DirectSpatialMode::FULL_VGL ? 3 * model.ne : 3;
  const std::size_t laplacians = mode == pf::DirectSpatialMode::FULL_VGL ? model.ne : 0;
  const std::size_t removed_bytes = sizeof(double) *
      (model.ndet * model.ne * model.ne * (1 + gradients + laplacians) + gradients + 2 * laplacians) +
      pf::determinantStorageRequirement(model.ndet, model.ne, gradients, laplacians);
  CHECK(complete->vectorStorageBytes() - trunk->vectorStorageBytes() == removed_bytes);
}
} // namespace

TEST_CASE("PsiFormer head-free spatial trunk matches native feature and cusp jets",
          "[wavefunction][psiformer][spatial_trunk]")
{
  auto model = tinyModel();
  const auto execution_plan = plan(model);
  pf::DirectValueExecutor value(model, execution_plan);
  pf::DirectSpatialExecutor spatial(model, value, execution_plan);
  auto workspace = spatial.makeTrunkWorkspace(pf::DirectSpatialMode::FULL_VGL);
  const auto electrons = model.cfg.configuration(0);
  workspace->setPositions(pf::GeometryPositionView::interleaved(electrons.x.data(), model.ne));
  const auto result = spatial.evaluateTrunkFull(*workspace);
  std::vector<std::size_t> seeds(3 * model.ne);
  std::iota(seeds.begin(), seeds.end(), 0);
  const auto coordinates = pf::coordinates(electrons, seeds);
  auto features = model.embedding(coordinates);
  for (std::size_t layer = 0; layer < model.blocks; ++layer)
    features = model.attention_block(features, static_cast<int>(layer));
  const auto cusp = model.cusp(coordinates);
  const std::size_t elements = model.ne * model.dim;
  REQUIRE(result.features->value.size() == elements);
  REQUIRE(result.features->gradient.size() == elements * 3 * model.ne);
  REQUIRE(result.features->laplacian.size() == elements * model.ne);
  for (std::size_t element = 0; element < elements; ++element)
  {
    close(result.features->value[element], features->value.x[element]);
    for (std::size_t lane = 0; lane < 3 * model.ne; ++lane)
      close(result.features->gradient[lane * elements + element], features->d1.x[lane * elements + element]);
    for (std::size_t electron = 0; electron < model.ne; ++electron)
    {
      double laplacian = 0;
      for (std::size_t dim = 0; dim < 3; ++dim)
        laplacian += features->d2.x[(3 * electron + dim) * elements + element];
      close(result.features->laplacian[electron * elements + element], laplacian);
    }
  }
  close(result.cusp, cusp->value.x[0]);
  for (std::size_t lane = 0; lane < 3 * model.ne; ++lane)
    close(result.cusp_gradient[lane], cusp->d1.x[lane]);
  for (std::size_t electron = 0; electron < model.ne; ++electron)
    close(result.cusp_laplacian[electron], cusp->d2.x[3 * electron] + cusp->d2.x[3 * electron + 1] +
                                           cusp->d2.x[3 * electron + 2]);
  CHECK(result.geometry->electronCount() == model.ne);
  CHECK(result.packed_positions[5] == electrons.x[5]);
  CHECK(result.parameter_version == model.p.version());
  CHECK(result.evaluation_generation == workspace->evaluationGeneration());
  CHECK(workspace->overlapsStorage(result.features->gradient.data(), sizeof(double)));
  CHECK_FALSE(workspace->overlapsStorage(electrons.x.data(), sizeof(double)));
  const auto fingerprint = workspace->storageFingerprint();
  spatial.evaluateTrunkFull(*workspace);
  CHECK(workspace->storageFingerprint() == fingerprint);
  checkStorage(spatial, pf::DirectSpatialMode::FULL_VGL, model);
  checkStorage(spatial, pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT, model);
}

TEST_CASE("PsiFormer spatial trunk preserves physical ACTIVE lanes with paired periodic metadata",
          "[wavefunction][psiformer][spatial_trunk]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  pf::DirectValueExecutor value(model, execution_plan, pf::DirectValuePreparation::REAL_TRUNK_ONLY);
  CHECK_THROWS_AS(pf::DirectSpatialExecutor(model, value, execution_plan), std::invalid_argument);
  pf::DirectSpatialExecutor spatial(model, value, execution_plan, pf::DirectOrbitalProjection::SCALAR,
                                     pf::DirectValuePreparation::REAL_TRUNK_ONLY);
  CHECK_THROWS_AS(spatial.layout(), std::invalid_argument);
  CHECK_THROWS_AS(spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL), std::invalid_argument);
  auto full_workspace = spatial.makeTrunkWorkspace(pf::DirectSpatialMode::FULL_VGL);
  auto active_workspace = spatial.makeTrunkWorkspace(pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT);
  const auto electrons = model.cfg.configuration(0);
  const auto positions = pf::GeometryPositionView::interleaved(electrons.x.data(), model.ne);
  full_workspace->setPositions(positions);
  active_workspace->setPositions(positions);
  const auto full = spatial.evaluateTrunkFull(*full_workspace);
  const auto active = spatial.evaluateTrunkActive(*active_workspace, 2);
  CHECK(active.active_electron == 2);
  CHECK(active.features->laplacian.empty());
  CHECK(active.cusp_laplacian.empty());
  CHECK(full_workspace->vectorStorageBytes() == full_workspace->requiredStorageBytes());
  CHECK(active_workspace->vectorStorageBytes() == active_workspace->requiredStorageBytes());
  const std::size_t elements = model.ne * model.dim;
  for (std::size_t element = 0; element < elements; ++element)
  {
    close(active.features->value[element], full.features->value[element]);
    for (std::size_t dim = 0; dim < 3; ++dim)
      close(active.features->gradient[dim * elements + element], full.features->gradient[(6 + dim) * elements + element]);
  }
  close(active.cusp, full.cusp);
  for (std::size_t dim = 0; dim < 3; ++dim)
    close(active.cusp_gradient[dim], full.cusp_gradient[6 + dim]);
  CHECK_THROWS_AS(spatial.evaluateTrunkActive(*full_workspace, 2), std::invalid_argument);
  CHECK_THROWS_AS(spatial.evaluateTrunkActive(*active_workspace, model.ne), std::invalid_argument);
}

TEST_CASE("PsiFormer spatial trunk binds identity and invalidates failed generations",
          "[wavefunction][psiformer][spatial_trunk]")
{
  auto model = tinyModel();
  const auto execution_plan = plan(model);
  pf::DirectValueExecutor value(model, execution_plan);
  pf::DirectSpatialExecutor spatial(model, value, execution_plan);
  auto workspace = spatial.makeTrunkWorkspace(pf::DirectSpatialMode::FULL_VGL);
  const auto electrons = model.cfg.configuration(0);
  workspace->setPositions(pf::GeometryPositionView::interleaved(electrons.x.data(), model.ne));
  const auto first = spatial.evaluateTrunkFull(*workspace);
  const std::vector<double> old_features = first.features->value;
  const auto& embedding = execution_plan.parameter(ps::ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
  model.p.set_flat_value(embedding.begin, model.p.flat_values()[embedding.begin] + .01);
  const auto updated = spatial.evaluateTrunkFull(*workspace);
  CHECK(updated.parameter_version == model.p.version());
  CHECK(updated.evaluation_generation != first.evaluation_generation);
  CHECK_FALSE(old_features == updated.features->value);
  workspace->setPosition(0, 0, std::numeric_limits<double>::quiet_NaN());
  CHECK_THROWS(spatial.evaluateTrunkFull(*workspace));
  CHECK(workspace->observedParameterVersion() == std::numeric_limits<std::size_t>::max());
  CHECK(workspace->evaluationGeneration() != updated.evaluation_generation);

  auto foreign_model = tinyModel();
  const auto foreign_plan = plan(foreign_model);
  pf::DirectValueExecutor foreign_value(foreign_model, foreign_plan);
  pf::DirectSpatialExecutor foreign(foreign_model, foreign_value, foreign_plan);
  auto wrong_identity = foreign.makeTrunkWorkspace(pf::DirectSpatialMode::FULL_VGL);
  CHECK_THROWS_AS(spatial.evaluateTrunkFull(*wrong_identity), std::invalid_argument);
  auto nuclei = model.cfg.nuclei.x;
  nuclei[0] += .25;
  pf::DirectSpatialTrunkWorkspace wrong_nuclei(*spatial.trunkLayout(),
      pf::GeometryPositionView::interleaved(nuclei.data(), model.cfg.nuclei.shape[0]), pf::DirectSpatialMode::FULL_VGL);
  CHECK_THROWS_AS(spatial.evaluateTrunkFull(wrong_nuclei), std::invalid_argument);
}
