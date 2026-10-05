//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_batch_executor.cpp
 * @brief Direct tests for bounded true multi-configuration PsiFormer execution.
 */

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerBatchExecutor.h"
#include "psiformer_test_utils.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace
{
std::atomic<bool> count_allocations{false};
std::atomic<std::size_t> allocation_count{0};
}

void* operator new(std::size_t bytes)
{
  if (count_allocations.load(std::memory_order_relaxed))
    allocation_count.fetch_add(1, std::memory_order_relaxed);
  if (void* storage = std::malloc(bytes))
    return storage;
  throw std::bad_alloc();
}

void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* storage) noexcept { std::free(storage); }
void operator delete[](void* storage) noexcept { std::free(storage); }
void operator delete(void* storage, std::size_t) noexcept { std::free(storage); }
void operator delete[](void* storage, std::size_t) noexcept { std::free(storage); }

namespace
{
using namespace qmcplusplus::testing::psiformer;

qmcplusplus::psiformer::PsiFormerExecutionPlan makePlan(const pf::PsiFormer& model)
{
  return qmcplusplus::psiformer::PsiFormerExecutionPlan::fromParameters(
      model.p, {model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet,
                model.dim, model.heads, model.blocks});
}

std::vector<double> displacedConfiguration(const pf::Tensor& base,
                                           std::size_t electrons,
                                           std::size_t configuration)
{
  std::vector<double> positions = base.x;
  for (std::size_t electron = 0; electron < electrons; ++electron)
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      positions[3 * electron + dimension] +=
          0.003 * static_cast<double>(configuration) *
          static_cast<double>((electron + 1) * (dimension + 1));
  return positions;
}

void loadBatch(pf::DirectBatchWorkspace& workspace,
               const pf::Tensor& base,
               std::size_t configurations)
{
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
  {
    const std::vector<double> positions =
        displacedConfiguration(base, workspace.electronCount(), configuration);
    workspace.setPositions(configuration, pf::GeometryPositionView::interleaved(
        positions.data(), workspace.electronCount()));
  }
}

void checkValue(const pf::DirectBatchValueResultView& batch,
                std::size_t configuration,
                const pf::DirectValueResult& scalar)
{
  CHECK(batch.sign[configuration] == scalar.sign);
  CHECK(batch.logabs[configuration] ==
        Catch::Approx(scalar.logabs).epsilon(3e-10).margin(3e-10));
  CHECK(batch.value[configuration] ==
        Catch::Approx(scalar.value).epsilon(3e-9).margin(1e-24));
  CHECK(batch.parameter_version[configuration] == scalar.parameter_version);
}

void validateBatches(const std::string& system,
                     const std::vector<std::size_t>& batch_sizes,
                     const std::vector<std::size_t>& tile_sizes)
{
  GeneratedFiles files = generateFiles(system);
  pf::PsiFormer model(files.parameters, files.configuration);
  const auto plan = makePlan(model);
  pf::DirectValueExecutor value_executor(model, plan);
  pf::DirectSpatialExecutor spatial_executor(model, value_executor, plan);
  pf::DirectBatchExecutor batch_executor(value_executor, spatial_executor);
  auto workspace = batch_executor.makeWorkspace();
  auto scalar_workspace = value_executor.makeWorkspace();
  const pf::Tensor base = model.cfg.configuration(0);

  for (const std::size_t tile_size : tile_sizes)
  {
    workspace->prepareTileCapacity(tile_size);
    for (const std::size_t batch_size : batch_sizes)
    {
      DYNAMIC_SECTION(system << " B=" << batch_size << " T=" << tile_size)
      {
        workspace->resize(pf::DirectBatchMode::VALUE_ONLY, batch_size);
        loadBatch(*workspace, base, batch_size);
        const pf::DirectBatchValueResultView batch =
            batch_executor.evaluateValues(*workspace);
        REQUIRE(batch.size == batch_size);
        for (std::size_t configuration = 0; configuration < batch_size; ++configuration)
        {
          const std::vector<double> positions =
              displacedConfiguration(base, model.ne, configuration);
          scalar_workspace->setPositions(
              pf::GeometryPositionView::interleaved(positions.data(), model.ne));
          checkValue(batch, configuration, value_executor.evaluate(*scalar_workspace));
        }

        const auto& statistics = workspace->executionStatistics();
        const std::size_t expected_tiles =
            batch_size == 0 ? 0 : (batch_size + tile_size - 1) / tile_size;
        CHECK(statistics.tiles_executed == expected_tiles);
        CHECK(statistics.max_tile_occupancy == std::min(batch_size, tile_size));
        CHECK(statistics.scalar_executor_calls == 0);
        if (batch_size == 0)
        {
          CHECK(statistics.grouped_dense_calls == 0);
          CHECK(statistics.max_grouped_rows == 0);
        }
        else
        {
          const std::size_t dense_calls_per_tile =
              1 + 6 * model.blocks + (model.cfg.nup != 0 ? 1 : 0) +
              (model.cfg.ndown != 0 ? 1 : 0);
          CHECK(statistics.grouped_dense_calls == expected_tiles * dense_calls_per_tile);
          CHECK(statistics.max_grouped_rows ==
                std::min(batch_size, tile_size) * model.ne);
        }
      }
    }
  }
}

} // namespace

TEST_CASE("PsiFormer value batches use true bounded tile kernels",
          "[wavefunction][psiformer][batch]")
{
  validateBatches("lih", {0, 1, 2, 3, 4, 7}, {1, 2, 4});
}

TEST_CASE("PsiFormer value batches cover pair and pseudopotential shapes",
          "[wavefunction][psiformer][batch][ecp]")
{
  validateBatches("lih_pair", {1, 3}, {2});
  validateBatches("lih_pp", {1, 3}, {2});
}

TEST_CASE("PsiFormer value batch storage is bounded stable and allocation free",
          "[wavefunction][psiformer][batch]")
{
  GeneratedFiles files = generateFiles("lih");
  pf::PsiFormer model(files.parameters, files.configuration);
  const auto plan = makePlan(model);
  pf::DirectValueExecutor value_executor(model, plan);
  pf::DirectSpatialExecutor spatial_executor(model, value_executor, plan);
  pf::DirectBatchExecutor batch_executor(value_executor, spatial_executor);
  auto workspace = batch_executor.makeWorkspace();
  const pf::Tensor base = model.cfg.configuration(0);

  workspace->prepareTileCapacity(2);
  workspace->resize(pf::DirectBatchMode::VALUE_ONLY, 4);
  const std::size_t tile_bytes = workspace->tileScratchBytes();
  const std::size_t logical_bytes = workspace->logicalStorageBytes();
  REQUIRE(tile_bytes > 0);
  workspace->resize(pf::DirectBatchMode::VALUE_ONLY, 64);
  CHECK(workspace->tileScratchBytes() == tile_bytes);
  CHECK(workspace->logicalStorageBytes() > logical_bytes);
  CHECK(workspace->allocatedTileCapacity() == 2);
  CHECK(workspace->capacity(pf::DirectBatchMode::VALUE_ONLY) == 64);

  workspace->prepareTileCapacity(4);
  workspace->resize(pf::DirectBatchMode::VALUE_ONLY, 7);
  loadBatch(*workspace, base, 7);
  batch_executor.evaluateValues(*workspace);
  const std::size_t fingerprint =
      workspace->storageFingerprint(pf::DirectBatchMode::VALUE_ONLY);

  volatile double sink = 0;
  allocation_count.store(0, std::memory_order_relaxed);
  count_allocations.store(true, std::memory_order_relaxed);
  for (int repetition = 0; repetition < 5; ++repetition)
    sink += batch_executor.evaluateValues(*workspace).logabs[repetition % 7];
  count_allocations.store(false, std::memory_order_relaxed);
  CHECK(allocation_count.load(std::memory_order_relaxed) == 0);
  CHECK(std::isfinite(sink));
  CHECK(workspace->storageFingerprint(pf::DirectBatchMode::VALUE_ONLY) == fingerprint);

  workspace->resize(pf::DirectBatchMode::VALUE_ONLY, 3);
  loadBatch(*workspace, base, 3);
  batch_executor.evaluateValues(*workspace);
  CHECK(workspace->storageFingerprint(pf::DirectBatchMode::VALUE_ONLY) == fingerprint);
  workspace->resize(pf::DirectBatchMode::VALUE_ONLY, 7);
  loadBatch(*workspace, base, 7);
  batch_executor.evaluateValues(*workspace);
  CHECK(workspace->storageFingerprint(pf::DirectBatchMode::VALUE_ONLY) == fingerprint);
}

TEST_CASE("PsiFormer value batch validates complete finite transactions",
          "[wavefunction][psiformer][batch]")
{
  GeneratedFiles files = generateFiles("lih");
  pf::PsiFormer model(files.parameters, files.configuration);
  const auto plan = makePlan(model);
  pf::DirectValueExecutor value_executor(model, plan);
  pf::DirectSpatialExecutor spatial_executor(model, value_executor, plan);
  pf::DirectBatchExecutor batch_executor(value_executor, spatial_executor);
  auto workspace = batch_executor.makeWorkspace();
  const pf::Tensor base = model.cfg.configuration(0);
  const auto base_view = pf::GeometryPositionView::interleaved(base.x.data(), model.ne);

  CHECK_THROWS_AS(workspace->prepareTileCapacity(0), std::invalid_argument);
  workspace->resize(pf::DirectBatchMode::VALUE_ONLY, 2);
  workspace->setPositions(0, base_view);
  CHECK_THROWS_AS(batch_executor.evaluateValues(*workspace), std::logic_error);
  CHECK_THROWS_AS(workspace->setPositions(
                      1, pf::GeometryPositionView::interleaved(base.x.data(), model.ne - 1)),
                  std::invalid_argument);
  CHECK_THROWS_AS(workspace->setPosition(2, 0, 0, 0.0), std::out_of_range);
  CHECK_THROWS_AS(workspace->setPosition(1, model.ne, 0, 0.0), std::out_of_range);
  CHECK_THROWS_AS(workspace->setPosition(1, 0, 3, 0.0), std::out_of_range);
  CHECK_THROWS_AS(workspace->setPosition(
                      1, 0, 0, std::numeric_limits<double>::infinity()),
                  std::invalid_argument);
  CHECK_THROWS_AS(workspace->resize(
                      pf::DirectBatchMode::VALUE_ONLY,
                      std::numeric_limits<std::size_t>::max()),
                  std::length_error);

  workspace->resize(pf::DirectBatchMode::FULL_VGL, 1);
  workspace->setPositions(0, base_view);
  CHECK_THROWS_AS(batch_executor.evaluateValues(*workspace), std::logic_error);
}

TEST_CASE("PsiFormer value batches preserve exact nodes and parameter versions",
          "[wavefunction][psiformer][batch]")
{
  GeneratedFiles files = generateFiles("lih");
  pf::PsiFormer model(files.parameters, files.configuration);
  const auto plan = makePlan(model);
  pf::DirectValueExecutor value_executor(model, plan);
  pf::DirectSpatialExecutor spatial_executor(model, value_executor, plan);
  pf::DirectBatchExecutor batch_executor(value_executor, spatial_executor);
  auto workspace = batch_executor.makeWorkspace();
  auto scalar_workspace = value_executor.makeWorkspace();
  const pf::Tensor base = model.cfg.configuration(0);

  workspace->prepareTileCapacity(2);
  workspace->resize(pf::DirectBatchMode::VALUE_ONLY, 2);
  loadBatch(*workspace, base, 2);
  const std::vector<double> displaced_node =
      displacedConfiguration(base, model.ne, 1);
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    workspace->setPosition(1, 1, dimension, displaced_node[dimension]);
  const auto node_batch = batch_executor.evaluateValues(*workspace);
  CHECK(node_batch.sign[1] == 0.0);
  CHECK(node_batch.logabs[1] == -std::numeric_limits<double>::infinity());
  CHECK(node_batch.value[1] == 0.0);

  workspace->resize(pf::DirectBatchMode::VALUE_ONLY, 2);
  loadBatch(*workspace, base, 2);
  model.p.set_flat_value(127, model.p.flat_values()[127] + 1.0e-3);
  const auto changed_batch = batch_executor.evaluateValues(*workspace);
  for (std::size_t configuration = 0; configuration < 2; ++configuration)
  {
    const std::vector<double> positions =
        displacedConfiguration(base, model.ne, configuration);
    scalar_workspace->setPositions(
        pf::GeometryPositionView::interleaved(positions.data(), model.ne));
    checkValue(changed_batch, configuration, value_executor.evaluate(*scalar_workspace));
    CHECK(changed_batch.parameter_version[configuration] == model.p.version());
  }
}
