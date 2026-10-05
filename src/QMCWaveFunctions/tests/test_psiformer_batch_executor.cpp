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

/** Make orbital column one nearly duplicate column zero without moving particles. */
void makeNearSingularOrbitalColumns(
    pf::PsiFormer& model,
    const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan,
    double perturbation)
{
  using qmcplusplus::psiformer::ParameterRole;
  const std::size_t channels = model.ndet * model.ne;
  const std::size_t nuclei = model.cfg.nuclei.shape[0];
  REQUIRE(model.ne >= 2);
  std::vector<std::size_t> indices;
  std::vector<double> values;
  const auto& flat = model.p.flat_values();

  auto copy_backflow_columns = [&](ParameterRole role) {
    const auto& tensor = plan.parameter(role);
    REQUIRE(tensor.size() == model.dim * channels);
    for (std::size_t determinant = 0; determinant < model.ndet;
         ++determinant)
      for (std::size_t feature = 0; feature < model.dim; ++feature)
      {
        const std::size_t source = tensor.begin + feature * channels +
            determinant * model.ne;
        const std::size_t destination = source + 1;
        double value = flat[source];
        if (feature == determinant % model.dim)
          value += perturbation * static_cast<double>(determinant + 1);
        indices.push_back(destination);
        values.push_back(value);
      }
  };
  auto copy_envelope_columns = [&](ParameterRole role) {
    const auto& tensor = plan.parameter(role);
    REQUIRE(tensor.size() == channels * nuclei);
    for (std::size_t determinant = 0; determinant < model.ndet;
         ++determinant)
      for (std::size_t nucleus = 0; nucleus < nuclei; ++nucleus)
      {
        const std::size_t source = tensor.begin +
            (determinant * model.ne) * nuclei + nucleus;
        indices.push_back(source + nuclei);
        values.push_back(flat[source]);
      }
  };

  copy_backflow_columns(ParameterRole::BACKFLOW_UP_WEIGHT);
  copy_backflow_columns(ParameterRole::BACKFLOW_DOWN_WEIGHT);
  copy_envelope_columns(ParameterRole::ENVELOPE_PI_UP);
  copy_envelope_columns(ParameterRole::ENVELOPE_PI_DOWN);
  copy_envelope_columns(ParameterRole::ENVELOPE_ZETA_UP);
  copy_envelope_columns(ParameterRole::ENVELOPE_ZETA_DOWN);
  model.p.set_flat_values(indices, values);
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

struct SparseReplacement
{
  std::size_t reference;
  std::size_t electron;
  pf::GeometryPosition position;
};

std::vector<std::vector<double>> makeSparseReferences(const pf::Tensor& base,
                                                      std::size_t electrons)
{
  return {displacedConfiguration(base, electrons, 1),
          displacedConfiguration(base, electrons, 3)};
}

std::vector<SparseReplacement> makeSparseReplacements(
    const std::vector<std::vector<double>>& references,
    std::size_t electrons,
    std::size_t replacement_count = 5)
{
  std::vector<SparseReplacement> replacements;
  replacements.reserve(replacement_count);
  for (std::size_t replacement = 0; replacement < replacement_count;
       ++replacement)
  {
    const std::size_t reference = replacement % references.size();
    const std::size_t electron = (2 * replacement + 1) % electrons;
    pf::GeometryPosition position;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      position[dimension] = references[reference][electron * 3 + dimension] +
          0.0007 * static_cast<double>((replacement + 1) * (dimension + 1));
    replacements.push_back({reference, electron, position});
  }
  return replacements;
}

std::vector<double> materializeReplacement(
    const std::vector<std::vector<double>>& references,
    const SparseReplacement& replacement)
{
  std::vector<double> positions = references[replacement.reference];
  std::copy(replacement.position.begin(), replacement.position.end(),
            positions.begin() + replacement.electron * 3);
  return positions;
}

void loadSparseBatch(pf::DirectBatchWorkspace& workspace,
                     const std::vector<std::vector<double>>& references,
                     const std::vector<SparseReplacement>& replacements)
{
  workspace.resizeSparseValues(references.size(), replacements.size());
  for (std::size_t reference = 0; reference < references.size(); ++reference)
    workspace.setReferenceConfiguration(
        reference, pf::GeometryPositionView::interleaved(
                       references[reference].data(), workspace.electronCount()));
  for (std::size_t replacement = 0; replacement < replacements.size();
       ++replacement)
    workspace.setVirtualReplacement(
        replacement, replacements[replacement].reference,
        replacements[replacement].electron,
        replacements[replacement].position);
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

void checkSpatial(const pf::DirectBatchSpatialResultView& batch,
                  std::size_t configuration,
                  const pf::DirectSpatialResultView& scalar)
{
  CHECK(batch.sign[configuration] == scalar.sign);
  CHECK(batch.logabs[configuration] ==
        Catch::Approx(scalar.logabs).epsilon(3e-10).margin(3e-10));
  CHECK(batch.value[configuration] ==
        Catch::Approx(scalar.value).epsilon(3e-9).margin(1e-24));
  CHECK(batch.parameter_version[configuration] == scalar.parameter_version);
  REQUIRE(batch.gradient_stride == scalar.gradient.size());
  for (std::size_t lane = 0; lane < scalar.gradient.size(); ++lane)
    CHECK(batch.gradient[configuration * batch.gradient_stride + lane] ==
          Catch::Approx(scalar.gradient[lane]).epsilon(3e-8).margin(3e-8));

  REQUIRE(batch.laplacian_stride == scalar.lap_log.size());
  REQUIRE(scalar.lap_log.size() == scalar.lap_ratio.size());
  for (std::size_t electron = 0; electron < scalar.lap_log.size(); ++electron)
  {
    const std::size_t output = configuration * batch.laplacian_stride + electron;
    CHECK(batch.lap_log[output] ==
          Catch::Approx(scalar.lap_log[electron]).epsilon(3e-7).margin(3e-7));
    CHECK(batch.lap_ratio[output] ==
          Catch::Approx(scalar.lap_ratio[electron]).epsilon(3e-7).margin(3e-7));
  }
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
        CHECK(statistics.reference_configurations == 0);
        CHECK(statistics.replacement_configurations == 0);
        CHECK(statistics.reference_evaluations == 0);
        CHECK(statistics.dense_coordinate_bytes_avoided == 0);
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

void validateSparseBatches(const std::string& system,
                           const std::vector<std::size_t>& tile_sizes)
{
  GeneratedFiles files = generateFiles(system);
  pf::PsiFormer model(files.parameters, files.configuration);
  const auto plan = makePlan(model);
  pf::DirectValueExecutor value_executor(model, plan);
  pf::DirectSpatialExecutor spatial_executor(model, value_executor, plan);
  pf::DirectBatchExecutor batch_executor(value_executor, spatial_executor);
  auto sparse_workspace = batch_executor.makeWorkspace();
  auto dense_workspace = batch_executor.makeWorkspace();
  auto scalar_workspace = value_executor.makeWorkspace();
  const pf::Tensor base = model.cfg.configuration(0);
  const auto references = makeSparseReferences(base, model.ne);
  const auto replacements = makeSparseReplacements(references, model.ne);
  const std::size_t total = references.size() + replacements.size();

  for (const std::size_t tile_size : tile_sizes)
  {
    DYNAMIC_SECTION(system << " sparse R=" << references.size()
                           << " Q=" << replacements.size()
                           << " T=" << tile_size)
    {
      sparse_workspace->prepareTileCapacity(tile_size);
      loadSparseBatch(*sparse_workspace, references, replacements);
      const auto sparse = batch_executor.evaluateValues(*sparse_workspace);
      REQUIRE(sparse.size == total);
      CHECK(sparse_workspace->valueInput() ==
            pf::DirectBatchValueInput::SPARSE_REPLACEMENTS);
      CHECK(sparse_workspace->referenceCount() == references.size());
      CHECK(sparse_workspace->replacementCount() == replacements.size());

      dense_workspace->prepareTileCapacity(tile_size);
      dense_workspace->resize(pf::DirectBatchMode::VALUE_ONLY, total);
      for (std::size_t reference = 0; reference < references.size(); ++reference)
        dense_workspace->setPositions(
            reference, pf::GeometryPositionView::interleaved(
                           references[reference].data(), model.ne));
      for (std::size_t replacement = 0; replacement < replacements.size();
           ++replacement)
      {
        const std::vector<double> positions =
            materializeReplacement(references, replacements[replacement]);
        dense_workspace->setPositions(
            references.size() + replacement,
            pf::GeometryPositionView::interleaved(positions.data(), model.ne));
      }
      const auto dense = batch_executor.evaluateValues(*dense_workspace);

      for (std::size_t configuration = 0; configuration < total;
           ++configuration)
      {
        const std::vector<double> positions = configuration < references.size()
            ? references[configuration]
            : materializeReplacement(
                  references, replacements[configuration - references.size()]);
        scalar_workspace->setPositions(
            pf::GeometryPositionView::interleaved(positions.data(), model.ne));
        checkValue(sparse, configuration,
                   value_executor.evaluate(*scalar_workspace));
        CHECK(sparse.sign[configuration] == dense.sign[configuration]);
        CHECK(sparse.logabs[configuration] ==
              Catch::Approx(dense.logabs[configuration])
                  .epsilon(3e-10).margin(3e-10));
        CHECK(sparse.value[configuration] ==
              Catch::Approx(dense.value[configuration])
                  .epsilon(3e-9).margin(1e-24));
        CHECK(sparse.parameter_version[configuration] ==
              dense.parameter_version[configuration]);
      }

      const auto& statistics = sparse_workspace->executionStatistics();
      const std::size_t expected_tiles = (total + tile_size - 1) / tile_size;
      const std::size_t dense_calls_per_tile =
          1 + 6 * model.blocks + (model.cfg.nup != 0 ? 1 : 0) +
          (model.cfg.ndown != 0 ? 1 : 0);
      CHECK(statistics.tiles_executed == expected_tiles);
      CHECK(statistics.max_tile_occupancy == std::min(total, tile_size));
      CHECK(statistics.grouped_dense_calls ==
            expected_tiles * dense_calls_per_tile);
      CHECK(statistics.max_grouped_rows ==
            std::min(total, tile_size) * model.ne);
      CHECK(statistics.scalar_executor_calls == 0);
      CHECK(statistics.reference_configurations == references.size());
      CHECK(statistics.replacement_configurations == replacements.size());
      CHECK(statistics.reference_evaluations == references.size());
      CHECK(statistics.dense_coordinate_bytes_avoided ==
            replacements.size() * (model.ne - 1) * 3 * sizeof(double));

      const auto& dense_statistics = dense_workspace->executionStatistics();
      CHECK(dense_statistics.reference_configurations == 0);
      CHECK(dense_statistics.replacement_configurations == 0);
      CHECK(dense_statistics.reference_evaluations == 0);
      CHECK(dense_statistics.dense_coordinate_bytes_avoided == 0);
    }
  }
}

void validateSpatialBatches(const std::string& system,
                            pf::DirectSpatialMode mode,
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
  auto scalar_workspace = spatial_executor.makeWorkspace(mode);
  const pf::Tensor base = model.cfg.configuration(0);
  const pf::DirectBatchMode batch_mode = mode == pf::DirectSpatialMode::FULL_VGL
      ? pf::DirectBatchMode::FULL_VGL
      : pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT;

  for (const std::size_t tile_size : tile_sizes)
  {
    workspace->prepareTileCapacity(tile_size);
    for (const std::size_t batch_size : batch_sizes)
    {
      DYNAMIC_SECTION(system << " spatial mode=" << static_cast<int>(mode)
                             << " B=" << batch_size << " T=" << tile_size)
      {
        workspace->resize(batch_mode, batch_size);
        loadBatch(*workspace, base, batch_size);
        std::vector<std::size_t> active_electrons(batch_size);
        for (std::size_t configuration = 0; configuration < batch_size;
             ++configuration)
          active_electrons[configuration] = configuration % model.ne;
        const pf::DirectBatchSpatialResultView batch =
            mode == pf::DirectSpatialMode::FULL_VGL
            ? batch_executor.evaluateFull(*workspace)
            : batch_executor.evaluateActive(
                  *workspace,
                  batch_size == 0 ? nullptr : active_electrons.data());

        REQUIRE(batch.size == batch_size);
        CHECK(batch.mode == mode);
        CHECK(batch.gradient_stride ==
              (mode == pf::DirectSpatialMode::FULL_VGL ? 3 * model.ne : 3));
        CHECK(batch.laplacian_stride ==
              (mode == pf::DirectSpatialMode::FULL_VGL ? model.ne : 0));
        if (mode == pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT)
        {
          CHECK(batch.lap_log == nullptr);
          CHECK(batch.lap_ratio == nullptr);
        }

        for (std::size_t configuration = 0; configuration < batch_size;
             ++configuration)
        {
          const std::vector<double> positions =
              displacedConfiguration(base, model.ne, configuration);
          scalar_workspace->setPositions(
              pf::GeometryPositionView::interleaved(positions.data(), model.ne));
          const pf::DirectSpatialResultView scalar =
              mode == pf::DirectSpatialMode::FULL_VGL
              ? spatial_executor.evaluateFull(*scalar_workspace)
              : spatial_executor.evaluateActive(
                    *scalar_workspace, active_electrons[configuration]);
          checkSpatial(batch, configuration, scalar);
        }

        const auto& statistics = workspace->executionStatistics();
        const std::size_t expected_tiles =
            batch_size == 0 ? 0 : (batch_size + tile_size - 1) / tile_size;
        CHECK(statistics.tiles_executed == expected_tiles);
        CHECK(statistics.max_tile_occupancy == std::min(batch_size, tile_size));
        CHECK(statistics.scalar_executor_calls == 0);
        CHECK(statistics.reference_configurations == 0);
        CHECK(statistics.replacement_configurations == 0);
        CHECK(statistics.reference_evaluations == 0);
        CHECK(statistics.dense_coordinate_bytes_avoided == 0);
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
          const std::size_t gradient_lanes =
              mode == pf::DirectSpatialMode::FULL_VGL ? 3 * model.ne : 3;
          const std::size_t laplacian_lanes =
              mode == pf::DirectSpatialMode::FULL_VGL ? model.ne : 0;
          CHECK(statistics.grouped_dense_calls ==
                expected_tiles * dense_calls_per_tile);
          CHECK(statistics.max_grouped_rows ==
                std::min(batch_size, tile_size) *
                    (1 + gradient_lanes + laplacian_lanes) * model.ne);
        }
      }
    }
  }
}

void validateSpatialStorage(pf::DirectSpatialMode mode)
{
  GeneratedFiles files = generateFiles("lih");
  pf::PsiFormer model(files.parameters, files.configuration);
  const auto plan = makePlan(model);
  pf::DirectValueExecutor value_executor(model, plan);
  pf::DirectSpatialExecutor spatial_executor(model, value_executor, plan);
  pf::DirectBatchExecutor batch_executor(value_executor, spatial_executor);
  auto workspace = batch_executor.makeWorkspace();
  auto scalar_workspace = spatial_executor.makeWorkspace(mode);
  const pf::Tensor base = model.cfg.configuration(0);
  const pf::DirectBatchMode batch_mode = mode == pf::DirectSpatialMode::FULL_VGL
      ? pf::DirectBatchMode::FULL_VGL
      : pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT;

  workspace->prepareTileCapacity(2);
  workspace->resize(batch_mode, 4);
  REQUIRE(scalar_workspace->geometryStorageBytes() > 0);
  const std::size_t geometry_fingerprint =
      scalar_workspace->geometryStorageFingerprint();
  CHECK(geometry_fingerprint != 0);
  CHECK(scalar_workspace->vectorStorageBytes() >
        scalar_workspace->geometryStorageBytes());
  CHECK(workspace->tileScratchBytes() ==
        2 * scalar_workspace->vectorStorageBytes() +
            workspace->spatialTileKernelBytes());
  scalar_workspace->setPositions(
      pf::GeometryPositionView::interleaved(base.x.data(), model.ne));
  if (mode == pf::DirectSpatialMode::FULL_VGL)
    spatial_executor.evaluateFull(*scalar_workspace);
  else
    spatial_executor.evaluateActive(*scalar_workspace, 0);
  CHECK(scalar_workspace->geometryStorageFingerprint() == geometry_fingerprint);
  const std::size_t tile_bytes = workspace->tileScratchBytes();
  const std::size_t logical_bytes = workspace->logicalStorageBytes();
  REQUIRE(tile_bytes > 0);
  workspace->resize(batch_mode, 64);
  CHECK(workspace->tileScratchBytes() == tile_bytes);
  CHECK(workspace->logicalStorageBytes() > logical_bytes);
  CHECK(workspace->allocatedTileCapacity() == 2);
  CHECK(workspace->capacity(batch_mode) == 64);

  workspace->prepareTileCapacity(4);
  workspace->resize(batch_mode, 7);
  loadBatch(*workspace, base, 7);
  std::vector<std::size_t> active_electrons(7);
  for (std::size_t configuration = 0; configuration < 7; ++configuration)
    active_electrons[configuration] = configuration % model.ne;
  auto evaluate = [&]() {
    return mode == pf::DirectSpatialMode::FULL_VGL
        ? batch_executor.evaluateFull(*workspace)
        : batch_executor.evaluateActive(*workspace, active_electrons.data());
  };
  evaluate();
  const std::size_t fingerprint = workspace->storageFingerprint(batch_mode);

  volatile double sink = 0;
  allocation_count.store(0, std::memory_order_relaxed);
  count_allocations.store(true, std::memory_order_relaxed);
  for (int repetition = 0; repetition < 3; ++repetition)
  {
    const pf::DirectBatchSpatialResultView result = evaluate();
    sink += result.logabs[repetition] + result.gradient[repetition];
  }
  count_allocations.store(false, std::memory_order_relaxed);
  CHECK(allocation_count.load(std::memory_order_relaxed) == 0);
  CHECK(std::isfinite(sink));
  CHECK(workspace->storageFingerprint(batch_mode) == fingerprint);

  workspace->resize(batch_mode, 3);
  loadBatch(*workspace, base, 3);
  active_electrons.resize(3);
  evaluate();
  CHECK(workspace->storageFingerprint(batch_mode) == fingerprint);
  workspace->resize(batch_mode, 7);
  loadBatch(*workspace, base, 7);
  active_electrons.resize(7);
  for (std::size_t configuration = 0; configuration < 7; ++configuration)
    active_electrons[configuration] = configuration % model.ne;
  evaluate();
  CHECK(workspace->storageFingerprint(batch_mode) == fingerprint);
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

TEST_CASE("PsiFormer sparse value batches share references without dense packing",
          "[wavefunction][psiformer][batch][sparse]")
{
  validateSparseBatches("lih", {1, 2, 3, 4});
  validateSparseBatches("lih_pair", {3});
  validateSparseBatches("lih_pp", {3});
}

TEST_CASE("PsiFormer sparse value input validates complete atomic transactions",
          "[wavefunction][psiformer][batch][sparse]")
{
  GeneratedFiles files = generateFiles("lih");
  pf::PsiFormer model(files.parameters, files.configuration);
  const auto plan = makePlan(model);
  pf::DirectValueExecutor value_executor(model, plan);
  pf::DirectSpatialExecutor spatial_executor(model, value_executor, plan);
  pf::DirectBatchExecutor batch_executor(value_executor, spatial_executor);
  auto workspace = batch_executor.makeWorkspace();
  const pf::Tensor base = model.cfg.configuration(0);
  const auto references = makeSparseReferences(base, model.ne);
  const auto replacements = makeSparseReplacements(references, model.ne, 2);

  workspace->resizeSparseValues(0, 0);
  const auto empty = batch_executor.evaluateValues(*workspace);
  CHECK(empty.size == 0);
  CHECK(workspace->executionStatistics().tiles_executed == 0);
  CHECK(workspace->valueInput() ==
        pf::DirectBatchValueInput::SPARSE_REPLACEMENTS);
  CHECK_THROWS_AS(workspace->resizeSparseValues(0, 1),
                  std::invalid_argument);

  workspace->resizeSparseValues(1, 0);
  workspace->setReferenceConfiguration(
      0, pf::GeometryPositionView::interleaved(references[0].data(), model.ne));
  CHECK(batch_executor.evaluateValues(*workspace).size == 1);

  workspace->resize(pf::DirectBatchMode::VALUE_ONLY, 1);
  CHECK_THROWS_AS(workspace->setReferenceConfiguration(
                      0, pf::GeometryPositionView::interleaved(
                             references[0].data(), model.ne)),
                  std::logic_error);
  CHECK_THROWS_AS(workspace->setVirtualReplacement(
                      0, 0, 0, replacements[0].position),
                  std::logic_error);

  workspace->resizeSparseValues(2, 2);
  CHECK_THROWS_AS(workspace->setPositions(
                      0, pf::GeometryPositionView::interleaved(
                             references[0].data(), model.ne)),
                  std::logic_error);
  CHECK_THROWS_AS(workspace->setPosition(0, 0, 0, 0.0),
                  std::logic_error);
  CHECK_THROWS_AS(workspace->setReferenceConfiguration(
                      2, pf::GeometryPositionView::interleaved(
                             references[0].data(), model.ne)),
                  std::out_of_range);
  CHECK_THROWS_AS(workspace->setReferenceConfiguration(
                      0, pf::GeometryPositionView::interleaved(
                             references[0].data(), model.ne - 1)),
                  std::invalid_argument);
  CHECK_THROWS_AS(workspace->setReferencePosition(2, 0, 0, 0.0),
                  std::out_of_range);
  CHECK_THROWS_AS(workspace->setReferencePosition(0, model.ne, 0, 0.0),
                  std::out_of_range);
  CHECK_THROWS_AS(workspace->setReferencePosition(0, 0, 3, 0.0),
                  std::out_of_range);
  CHECK_THROWS_AS(workspace->setReferencePosition(
                      0, 0, 0, std::numeric_limits<double>::infinity()),
                  std::invalid_argument);
  CHECK_THROWS_AS(workspace->setVirtualReplacement(
                      2, 0, 0, replacements[0].position),
                  std::out_of_range);
  CHECK_THROWS_AS(workspace->setVirtualReplacement(
                      0, 2, 0, replacements[0].position),
                  std::out_of_range);
  CHECK_THROWS_AS(workspace->setVirtualReplacement(
                      0, 0, model.ne, replacements[0].position),
                  std::out_of_range);

  for (std::size_t electron = 0; electron < model.ne; ++electron)
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      workspace->setReferencePosition(
          0, electron, dimension, references[0][electron * 3 + dimension]);
  workspace->setVirtualReplacement(
      0, replacements[0].reference, replacements[0].electron,
      replacements[0].position);
  workspace->setVirtualReplacement(
      1, replacements[1].reference, replacements[1].electron,
      replacements[1].position);
  CHECK_THROWS_AS(batch_executor.evaluateValues(*workspace), std::logic_error);
  workspace->setReferenceConfiguration(
      1, pf::GeometryPositionView::interleaved(references[1].data(), model.ne));
  const auto valid = batch_executor.evaluateValues(*workspace);
  const std::vector<double> valid_logabs(valid.logabs, valid.logabs + valid.size);

  std::vector<double> nonfinite_reference = references[0];
  nonfinite_reference[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_AS(workspace->setReferenceConfiguration(
                      0, pf::GeometryPositionView::interleaved(
                             nonfinite_reference.data(), model.ne)),
                  std::invalid_argument);
  pf::GeometryPosition nonfinite_replacement = replacements[0].position;
  nonfinite_replacement[1] = std::numeric_limits<double>::infinity();
  CHECK_THROWS_AS(workspace->setVirtualReplacement(
                      0, replacements[0].reference,
                      replacements[0].electron, nonfinite_replacement),
                  std::invalid_argument);
  const auto unchanged = batch_executor.evaluateValues(*workspace);
  CHECK(std::equal(valid_logabs.begin(), valid_logabs.end(), unchanged.logabs));

  const std::size_t size_before = workspace->size();
  const std::size_t logical_bytes_before = workspace->logicalStorageBytes();
  const std::size_t tile_bytes_before = workspace->tileScratchBytes();
  const std::size_t fingerprint_before = workspace->storageFingerprint(
      pf::DirectBatchMode::VALUE_ONLY);
  CHECK_THROWS_AS(workspace->resizeSparseValues(
                      std::numeric_limits<std::size_t>::max(), 1),
                  std::length_error);
  CHECK_THROWS_AS(workspace->resizeSparseValues(
                      1, std::numeric_limits<std::size_t>::max() / 3 + 1),
                  std::length_error);
  CHECK(workspace->size() == size_before);
  CHECK(workspace->logicalStorageBytes() == logical_bytes_before);
  CHECK(workspace->tileScratchBytes() == tile_bytes_before);
  CHECK(workspace->storageFingerprint(pf::DirectBatchMode::VALUE_ONLY) ==
        fingerprint_before);

  const std::size_t oversized_batch =
      static_cast<std::size_t>(std::numeric_limits<int>::max()) / model.ne + 1;
  workspace->prepareTileCapacity(oversized_batch);
  CHECK_THROWS_AS(workspace->resizeSparseValues(1, oversized_batch - 1),
                  std::length_error);
  CHECK(workspace->size() == size_before);
  CHECK(workspace->logicalStorageBytes() == logical_bytes_before);
  CHECK(workspace->tileScratchBytes() == tile_bytes_before);
  CHECK(workspace->storageFingerprint(pf::DirectBatchMode::VALUE_ONLY) ==
        fingerprint_before);
}

TEST_CASE("PsiFormer sparse value storage is tile bounded and allocation free",
          "[wavefunction][psiformer][batch][sparse]")
{
  GeneratedFiles files = generateFiles("lih");
  pf::PsiFormer model(files.parameters, files.configuration);
  const auto plan = makePlan(model);
  pf::DirectValueExecutor value_executor(model, plan);
  pf::DirectSpatialExecutor spatial_executor(model, value_executor, plan);
  pf::DirectBatchExecutor batch_executor(value_executor, spatial_executor);
  auto workspace = batch_executor.makeWorkspace();
  const pf::Tensor base = model.cfg.configuration(0);
  const auto references = makeSparseReferences(base, model.ne);
  const auto replacements_4 = makeSparseReplacements(references, model.ne, 4);
  const auto replacements_7 = makeSparseReplacements(references, model.ne, 7);
  const auto replacements_64 = makeSparseReplacements(references, model.ne, 64);

  workspace->prepareTileCapacity(2);
  loadSparseBatch(*workspace, references, replacements_4);
  batch_executor.evaluateValues(*workspace);
  const std::size_t tile_bytes = workspace->tileScratchBytes();
  const std::size_t sparse_tile_bytes = workspace->sparseTilePositionBytes();
  const std::size_t sparse_input_4 = workspace->sparseInputStorageBytes();
  const std::size_t logical_bytes_4 = workspace->logicalStorageBytes();
  CHECK(sparse_tile_bytes >= 2 * model.ne * 3 * sizeof(double));

  loadSparseBatch(*workspace, references, replacements_64);
  const auto large = batch_executor.evaluateValues(*workspace);
  CHECK(large.size == references.size() + replacements_64.size());
  CHECK(workspace->tileScratchBytes() == tile_bytes);
  CHECK(workspace->sparseTilePositionBytes() == sparse_tile_bytes);
  CHECK(workspace->sparseInputStorageBytes() > sparse_input_4);
  CHECK(workspace->logicalStorageBytes() > logical_bytes_4);
  CHECK(workspace->allocatedTileCapacity() == 2);
  CHECK(workspace->capacity(pf::DirectBatchMode::VALUE_ONLY) == large.size);
  CHECK(workspace->sparseInputStorageBytes() <
        large.size * model.ne * 3 * sizeof(double));
  CHECK(workspace->executionStatistics().dense_coordinate_bytes_avoided ==
        replacements_64.size() * (model.ne - 1) * 3 * sizeof(double));

  const std::size_t fingerprint = workspace->storageFingerprint(
      pf::DirectBatchMode::VALUE_ONLY);
  volatile double sink = 0;
  allocation_count.store(0, std::memory_order_relaxed);
  count_allocations.store(true, std::memory_order_relaxed);
  for (int repetition = 0; repetition < 3; ++repetition)
    sink += batch_executor.evaluateValues(*workspace).logabs[repetition];
  count_allocations.store(false, std::memory_order_relaxed);
  CHECK(allocation_count.load(std::memory_order_relaxed) == 0);
  CHECK(std::isfinite(sink));
  CHECK(workspace->storageFingerprint(pf::DirectBatchMode::VALUE_ONLY) ==
        fingerprint);

  loadSparseBatch(*workspace, references, replacements_7);
  batch_executor.evaluateValues(*workspace);
  CHECK(workspace->storageFingerprint(pf::DirectBatchMode::VALUE_ONLY) ==
        fingerprint);
  loadSparseBatch(*workspace, references, replacements_64);
  batch_executor.evaluateValues(*workspace);
  CHECK(workspace->storageFingerprint(pf::DirectBatchMode::VALUE_ONLY) ==
        fingerprint);

  workspace->prepareTileCapacity(4);
  CHECK(workspace->sparseTilePositionBytes() >=
        4 * model.ne * 3 * sizeof(double));
  CHECK(workspace->sparseTilePositionBytes() > sparse_tile_bytes);
  const std::size_t larger_tile_bytes = workspace->tileScratchBytes();
  loadSparseBatch(*workspace, references, replacements_7);
  batch_executor.evaluateValues(*workspace);
  CHECK(workspace->tileScratchBytes() == larger_tile_bytes);

  const std::size_t retained_sparse_input = workspace->sparseInputStorageBytes();
  const std::size_t retained_sparse_tile = workspace->sparseTilePositionBytes();
  workspace->resize(pf::DirectBatchMode::VALUE_ONLY, 3);
  loadBatch(*workspace, base, 3);
  batch_executor.evaluateValues(*workspace);
  CHECK(workspace->valueInput() ==
        pf::DirectBatchValueInput::DENSE_CONFIGURATIONS);
  CHECK(workspace->sparseInputStorageBytes() == retained_sparse_input);
  CHECK(workspace->sparseTilePositionBytes() == retained_sparse_tile);
  CHECK(workspace->executionStatistics().reference_configurations == 0);
  CHECK(workspace->executionStatistics().replacement_configurations == 0);
  CHECK(workspace->executionStatistics().reference_evaluations == 0);
  CHECK(workspace->executionStatistics().dense_coordinate_bytes_avoided == 0);
}

TEST_CASE("PsiFormer sparse value batches preserve nodes versions and atomic output",
          "[wavefunction][psiformer][batch][sparse]")
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
  REQUIRE(model.cfg.nup >= 2);

  const std::vector<double> normal_reference =
      displacedConfiguration(base, model.ne, 1);
  std::vector<std::vector<double>> references{normal_reference};
  pf::GeometryPosition shifted_position;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    shifted_position[dimension] = normal_reference[3 + dimension] +
        0.001 * static_cast<double>(dimension + 1);
  std::vector<SparseReplacement> replacements{{0, 1, shifted_position}};

  workspace->prepareTileCapacity(1);
  loadSparseBatch(*workspace, references, replacements);
  const auto valid = batch_executor.evaluateValues(*workspace);
  const std::vector<double> valid_sign(valid.sign, valid.sign + valid.size);
  const std::vector<double> valid_logabs(valid.logabs, valid.logabs + valid.size);
  const std::vector<double> valid_value(valid.value, valid.value + valid.size);
  const std::vector<std::size_t> valid_version(
      valid.parameter_version, valid.parameter_version + valid.size);
  const pf::DirectBatchExecutionStatistics valid_statistics =
      workspace->executionStatistics();

  std::vector<double> node_reference = normal_reference;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    node_reference[3 + dimension] = node_reference[dimension];
  pf::GeometryPosition recover_position;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    recover_position[dimension] = normal_reference[3 + dimension];
  references[0] = node_reference;
  replacements[0] = {0, 1, recover_position};
  loadSparseBatch(*workspace, references, replacements);

  const auto& same_alpha = plan.parameter(
      qmcplusplus::psiformer::ParameterRole::CUSP_SAME_ALPHA);
  REQUIRE(same_alpha.size() == 1);
  const double original_same_alpha = model.p.flat_values()[same_alpha.begin];
  model.p.set_flat_value(same_alpha.begin, 1.0e308);
  CHECK_THROWS_AS(batch_executor.evaluateValues(*workspace), std::runtime_error);
  CHECK(std::equal(valid_sign.begin(), valid_sign.end(), valid.sign));
  CHECK(std::equal(valid_logabs.begin(), valid_logabs.end(), valid.logabs));
  CHECK(std::equal(valid_value.begin(), valid_value.end(), valid.value));
  CHECK(std::equal(valid_version.begin(), valid_version.end(),
                   valid.parameter_version));
  CHECK(workspace->executionStatistics().tiles_executed ==
        valid_statistics.tiles_executed);
  CHECK(workspace->executionStatistics().grouped_dense_calls ==
        valid_statistics.grouped_dense_calls);
  CHECK(workspace->executionStatistics().max_tile_occupancy ==
        valid_statistics.max_tile_occupancy);
  CHECK(workspace->executionStatistics().max_grouped_rows ==
        valid_statistics.max_grouped_rows);
  CHECK(workspace->executionStatistics().scalar_executor_calls ==
        valid_statistics.scalar_executor_calls);
  CHECK(workspace->executionStatistics().reference_configurations ==
        valid_statistics.reference_configurations);
  CHECK(workspace->executionStatistics().replacement_configurations ==
        valid_statistics.replacement_configurations);
  CHECK(workspace->executionStatistics().reference_evaluations ==
        valid_statistics.reference_evaluations);
  CHECK(workspace->executionStatistics().dense_coordinate_bytes_avoided ==
        valid_statistics.dense_coordinate_bytes_avoided);

  model.p.set_flat_value(same_alpha.begin, original_same_alpha);
  const auto retried = batch_executor.evaluateValues(*workspace);
  CHECK(retried.sign[0] == 0.0);
  CHECK(retried.logabs[0] == -std::numeric_limits<double>::infinity());
  CHECK(retried.value[0] == 0.0);
  scalar_workspace->setPositions(pf::GeometryPositionView::interleaved(
      normal_reference.data(), model.ne));
  checkValue(retried, 1, value_executor.evaluate(*scalar_workspace));
  for (std::size_t configuration = 0; configuration < retried.size;
       ++configuration)
    CHECK(retried.parameter_version[configuration] == model.p.version());

  references[0] = normal_reference;
  pf::GeometryPosition node_position;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    node_position[dimension] = normal_reference[dimension];
  replacements[0] = {0, 1, node_position};
  loadSparseBatch(*workspace, references, replacements);
  const auto replacement_node = batch_executor.evaluateValues(*workspace);
  CHECK(replacement_node.sign[1] == 0.0);
  CHECK(replacement_node.logabs[1] ==
        -std::numeric_limits<double>::infinity());
  CHECK(replacement_node.value[1] == 0.0);
}

TEST_CASE("PsiFormer spatial batches use true bounded tile kernels",
          "[wavefunction][psiformer][batch]")
{
  validateSpatialBatches("lih", pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT,
                         {0, 1, 2, 3, 5}, {1, 2, 4});
  validateSpatialBatches("lih", pf::DirectSpatialMode::FULL_VGL,
                         {0, 1, 2, 3, 5}, {1, 2, 4});
}

TEST_CASE("PsiFormer spatial batches cover pair and pseudopotential shapes",
          "[wavefunction][psiformer][batch][ecp]")
{
  for (const std::string system : {"lih_pair", "lih_pp"})
  {
    validateSpatialBatches(system,
                           pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT,
                           {1, 3}, {2});
    validateSpatialBatches(system, pf::DirectSpatialMode::FULL_VGL,
                           {1, 3}, {2});
  }
}

TEST_CASE("PsiFormer odd-block spatial batches preserve allocation identity",
          "[wavefunction][psiformer][batch]")
{
  GeneratedFiles files = generateFiles("lih", 3);
  pf::PsiFormer model(files.parameters, files.configuration);
  model.blocks = 3;
  const auto plan = makePlan(model);
  pf::DirectValueExecutor value_executor(model, plan);
  pf::DirectSpatialExecutor spatial_executor(model, value_executor, plan);
  pf::DirectBatchExecutor batch_executor(value_executor, spatial_executor);
  auto batch_workspace = batch_executor.makeWorkspace();
  auto scalar_workspace = spatial_executor.makeWorkspace(
      pf::DirectSpatialMode::FULL_VGL);
  const pf::Tensor base = model.cfg.configuration(0);

  batch_workspace->prepareTileCapacity(2);
  batch_workspace->resize(pf::DirectBatchMode::FULL_VGL, 3);
  loadBatch(*batch_workspace, base, 3);
  const std::size_t batch_fingerprint = batch_workspace->storageFingerprint(
      pf::DirectBatchMode::FULL_VGL);
  const pf::DirectBatchSpatialResultView batch =
      batch_executor.evaluateFull(*batch_workspace);
  CHECK(batch_workspace->storageFingerprint(pf::DirectBatchMode::FULL_VGL) ==
        batch_fingerprint);

  for (std::size_t configuration = 0; configuration < 3; ++configuration)
  {
    const std::vector<double> positions =
        displacedConfiguration(base, model.ne, configuration);
    scalar_workspace->setPositions(
        pf::GeometryPositionView::interleaved(positions.data(), model.ne));
    const std::size_t scalar_fingerprint =
        scalar_workspace->storageFingerprint();
    checkSpatial(batch, configuration,
                 spatial_executor.evaluateFull(*scalar_workspace));
    CHECK(scalar_workspace->storageFingerprint() == scalar_fingerprint);
  }

  batch_executor.evaluateFull(*batch_workspace);
  CHECK(batch_workspace->storageFingerprint(pf::DirectBatchMode::FULL_VGL) ==
        batch_fingerprint);
}

TEST_CASE("PsiFormer spatial batches preserve near-singular projected determinants",
          "[wavefunction][psiformer][batch]")
{
  GeneratedFiles files = generateFiles("lih");
  pf::PsiFormer model(files.parameters, files.configuration);
  const auto plan = makePlan(model);
  const pf::Tensor base = model.cfg.configuration(0);

  pf::DirectValueExecutor baseline_value_executor(model, plan);
  pf::DirectSpatialExecutor baseline_spatial_executor(
      model, baseline_value_executor, plan);
  auto baseline_workspace = baseline_spatial_executor.makeWorkspace(
      pf::DirectSpatialMode::FULL_VGL);
  baseline_workspace->setPositions(
      pf::GeometryPositionView::interleaved(base.x.data(), model.ne));
  const double baseline_logabs =
      baseline_spatial_executor.evaluateFull(*baseline_workspace).logabs;

  // Duplicate learned orbital columns rather than particle positions.  The small
  // independent backflow perturbation keeps every determinant nonsingular while
  // driving the stable reduction close to its singular boundary.
  makeNearSingularOrbitalColumns(model, plan, 1.0e-5);
  pf::DirectValueExecutor value_executor(model, plan);
  pf::DirectSpatialExecutor spatial_executor(model, value_executor, plan);
  pf::DirectBatchExecutor batch_executor(value_executor, spatial_executor);
  auto batch_workspace = batch_executor.makeWorkspace();
  auto scalar_workspace = spatial_executor.makeWorkspace(
      pf::DirectSpatialMode::FULL_VGL);
  batch_workspace->prepareTileCapacity(2);
  batch_workspace->resize(pf::DirectBatchMode::FULL_VGL, 3);
  loadBatch(*batch_workspace, base, 3);
  const auto batch = batch_executor.evaluateFull(*batch_workspace);
  REQUIRE(batch.sign[0] != 0.0);
  CHECK(std::abs(batch.logabs[0] - baseline_logabs) > 2.0);
  for (std::size_t configuration = 0; configuration < 3; ++configuration)
  {
    const std::vector<double> positions =
        displacedConfiguration(base, model.ne, configuration);
    scalar_workspace->setPositions(
        pf::GeometryPositionView::interleaved(positions.data(), model.ne));
    checkSpatial(batch, configuration,
                 spatial_executor.evaluateFull(*scalar_workspace));
  }
  CHECK(batch_workspace->executionStatistics().tiles_executed == 2);
  CHECK(batch_workspace->executionStatistics().scalar_executor_calls == 0);
}

TEST_CASE("PsiFormer spatial batch storage is bounded stable and allocation free",
          "[wavefunction][psiformer][batch]")
{
  validateSpatialStorage(pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT);
  validateSpatialStorage(pf::DirectSpatialMode::FULL_VGL);
}

TEST_CASE("PsiFormer spatial batches validate and commit atomically",
          "[wavefunction][psiformer][batch]")
{
  GeneratedFiles files = generateFiles("lih");
  pf::PsiFormer model(files.parameters, files.configuration);
  const auto plan = makePlan(model);
  pf::DirectValueExecutor value_executor(model, plan);
  pf::DirectSpatialExecutor spatial_executor(model, value_executor, plan);
  pf::DirectBatchExecutor batch_executor(value_executor, spatial_executor);
  auto workspace = batch_executor.makeWorkspace();
  auto scalar_workspace =
      spatial_executor.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  const pf::Tensor base = model.cfg.configuration(0);

  workspace->prepareTileCapacity(1);
  workspace->resize(pf::DirectBatchMode::FULL_VGL, 2);
  loadBatch(*workspace, base, 2);
  const pf::DirectBatchSpatialResultView valid =
      batch_executor.evaluateFull(*workspace);
  const std::vector<double> valid_sign(valid.sign, valid.sign + valid.size);
  const std::vector<double> valid_logabs(valid.logabs, valid.logabs + valid.size);
  const std::vector<double> valid_value(valid.value, valid.value + valid.size);
  const std::vector<double> valid_gradient(
      valid.gradient, valid.gradient + valid.size * valid.gradient_stride);
  const std::vector<double> valid_lap_log(
      valid.lap_log, valid.lap_log + valid.size * valid.laplacian_stride);
  const std::vector<double> valid_lap_ratio(
      valid.lap_ratio, valid.lap_ratio + valid.size * valid.laplacian_stride);
  const pf::DirectBatchExecutionStatistics valid_statistics =
      workspace->executionStatistics();

  // A close same-spin pair exercises stable near-node reduction before the exact
  // node below.  Compare against the unchanged scalar path at the same geometry.
  std::vector<double> near_node = displacedConfiguration(base, model.ne, 0);
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    near_node[3 + dimension] = near_node[dimension];
  near_node[3] += 1.0e-4;
  workspace->resize(pf::DirectBatchMode::FULL_VGL, 1);
  workspace->setPositions(
      0, pf::GeometryPositionView::interleaved(near_node.data(), model.ne));
  const pf::DirectBatchSpatialResultView near_batch =
      batch_executor.evaluateFull(*workspace);
  scalar_workspace->setPositions(
      pf::GeometryPositionView::interleaved(near_node.data(), model.ne));
  checkSpatial(near_batch, 0,
               spatial_executor.evaluateFull(*scalar_workspace));

  // Re-establish the public output snapshot, then fail in tile one after tile zero
  // has completed.  No partial result may become observable.
  workspace->resize(pf::DirectBatchMode::FULL_VGL, 2);
  loadBatch(*workspace, base, 2);
  batch_executor.evaluateFull(*workspace);
  workspace->resize(pf::DirectBatchMode::FULL_VGL, 2);
  loadBatch(*workspace, base, 2);
  const std::vector<double> node = displacedConfiguration(base, model.ne, 1);
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    workspace->setPosition(1, 1, dimension, node[dimension]);
  CHECK_THROWS_AS(batch_executor.evaluateFull(*workspace), std::runtime_error);
  CHECK(std::equal(valid_sign.begin(), valid_sign.end(), valid.sign));
  CHECK(std::equal(valid_logabs.begin(), valid_logabs.end(), valid.logabs));
  CHECK(std::equal(valid_value.begin(), valid_value.end(), valid.value));
  CHECK(std::equal(valid_gradient.begin(), valid_gradient.end(), valid.gradient));
  CHECK(std::equal(valid_lap_log.begin(), valid_lap_log.end(), valid.lap_log));
  CHECK(std::equal(valid_lap_ratio.begin(), valid_lap_ratio.end(), valid.lap_ratio));
  CHECK(workspace->executionStatistics().tiles_executed ==
        valid_statistics.tiles_executed);
  CHECK(workspace->executionStatistics().grouped_dense_calls ==
        valid_statistics.grouped_dense_calls);

  // A corrected transaction remains usable after the failed later tile.
  workspace->resize(pf::DirectBatchMode::FULL_VGL, 2);
  loadBatch(*workspace, base, 2);
  const auto retried = batch_executor.evaluateFull(*workspace);
  CHECK(retried.logabs[0] ==
        Catch::Approx(valid_logabs[0]).epsilon(3e-10).margin(3e-10));
  CHECK(retried.logabs[1] ==
        Catch::Approx(valid_logabs[1]).epsilon(3e-10).margin(3e-10));

  workspace->resize(pf::DirectBatchMode::FULL_VGL, 2);
  loadBatch(*workspace, base, 2);
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    workspace->setPosition(1, 0, dimension,
                           model.cfg.nuclei.x[dimension]);
  CHECK_THROWS_AS(batch_executor.evaluateFull(*workspace), std::runtime_error);
  CHECK(std::equal(valid_logabs.begin(), valid_logabs.end(), valid.logabs));

  workspace->resize(pf::DirectBatchMode::FULL_VGL, 2);
  loadBatch(*workspace, base, 2);
  model.p.set_flat_value(127, model.p.flat_values()[127] + 1.0e-3);
  const auto changed = batch_executor.evaluateFull(*workspace);
  for (std::size_t configuration = 0; configuration < changed.size;
       ++configuration)
    CHECK(changed.parameter_version[configuration] == model.p.version());

  workspace->resize(pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT, 0);
  const auto empty = batch_executor.evaluateActive(*workspace, nullptr);
  CHECK(empty.size == 0);
  CHECK(workspace->executionStatistics().tiles_executed == 0);
  CHECK_THROWS_AS(batch_executor.evaluateFull(*workspace), std::logic_error);

  workspace->resize(pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT, 2);
  loadBatch(*workspace, base, 2);
  CHECK_THROWS_AS(batch_executor.evaluateActive(*workspace, nullptr),
                  std::invalid_argument);
  const std::size_t invalid_active[2]{0, model.ne};
  CHECK_THROWS_AS(batch_executor.evaluateActive(*workspace, invalid_active),
                  std::out_of_range);
  workspace->resize(pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT, 2);
  workspace->setPositions(
      0, pf::GeometryPositionView::interleaved(base.x.data(), model.ne));
  const std::size_t valid_active[2]{0, 1};
  CHECK_THROWS_AS(batch_executor.evaluateActive(*workspace, valid_active),
                  std::logic_error);

  // A spatial executor is permanently bound to the immutable layout supplied by
  // its value executor.  Equal dimensions from another layout instance are not
  // sufficient because packed scratch is sized from descriptor identity.
  pf::DirectValueExecutor foreign_value_executor(model, plan);
  CHECK_THROWS_AS(
      pf::DirectBatchWorkspace(foreign_value_executor, spatial_executor),
      std::invalid_argument);
  CHECK_THROWS_AS(
      pf::DirectBatchExecutor(foreign_value_executor, spatial_executor),
      std::invalid_argument);

  pf::PsiFormer foreign_model(files.parameters, files.configuration);
  CHECK_THROWS_AS(
      pf::DirectSpatialExecutor(foreign_model, value_executor, plan),
      std::invalid_argument);
  const double original_nucleus = model.cfg.nuclei.x[0];
  model.cfg.nuclei.x[0] += 0.125;
  CHECK_THROWS_AS(
      pf::DirectSpatialExecutor(model, value_executor, plan),
      std::invalid_argument);
  model.cfg.nuclei.x[0] = original_nucleus;

  const std::size_t beyond_blas_int =
      static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
  CHECK_THROWS_AS(qmcplusplus::psiformer::dense::checkedBlasDimension(
                      beyond_blas_int, "test BLAS extent"),
                  std::length_error);
  CHECK_THROWS_AS(qmcplusplus::psiformer::dense::productBlasReal(
                      nullptr, nullptr, nullptr, beyond_blas_int, 1, 1,
                      nullptr),
                  std::length_error);

  // A large tile policy is harmless while B is small.  Once B makes that
  // occupancy effective, resize must reject the BLAS row extent before growing
  // either logical storage or any spatial arena.
  const std::size_t full_rows_per_slot =
      (1 + 4 * model.ne) * model.ne;
  const std::size_t oversized_batch =
      static_cast<std::size_t>(std::numeric_limits<int>::max()) /
          full_rows_per_slot +
      1;
  CHECK_NOTHROW(workspace->prepareTileCapacity(oversized_batch));
  CHECK(workspace->tileCapacity() == oversized_batch);
  const std::size_t logical_bytes_before_preflight =
      workspace->logicalStorageBytes();
  const std::size_t tile_bytes_before_preflight = workspace->tileScratchBytes();
  const std::size_t fingerprint_before_preflight = workspace->storageFingerprint(
      pf::DirectBatchMode::FULL_VGL);
  CHECK_THROWS_AS(workspace->resize(pf::DirectBatchMode::FULL_VGL,
                                    oversized_batch),
                  std::length_error);
  CHECK(workspace->size() == 2);
  CHECK(workspace->capacity(pf::DirectBatchMode::FULL_VGL) == 2);
  CHECK(workspace->logicalStorageBytes() == logical_bytes_before_preflight);
  CHECK(workspace->tileScratchBytes() == tile_bytes_before_preflight);
  CHECK(workspace->storageFingerprint(
            pf::DirectBatchMode::FULL_VGL) ==
        fingerprint_before_preflight);
}

TEST_CASE("PsiFormer batch mode switching has a bounded additive high water",
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
  const std::size_t active_electrons[4]{0, 1, 2, 3};

  workspace->prepareTileCapacity(2);
  workspace->resize(pf::DirectBatchMode::VALUE_ONLY, 4);
  loadBatch(*workspace, base, 4);
  batch_executor.evaluateValues(*workspace);
  const std::size_t value_high_water = workspace->tileScratchBytes();

  workspace->resize(pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT, 4);
  loadBatch(*workspace, base, 4);
  batch_executor.evaluateActive(*workspace, active_electrons);
  const std::size_t active_high_water = workspace->tileScratchBytes();
  CHECK(active_high_water > value_high_water);

  workspace->resize(pf::DirectBatchMode::FULL_VGL, 4);
  loadBatch(*workspace, base, 4);
  batch_executor.evaluateFull(*workspace);
  const std::size_t all_mode_high_water = workspace->tileScratchBytes();
  CHECK(all_mode_high_water > active_high_water);
  CHECK(workspace->allocatedTileCapacity() == 2);

  // Logical growth and subsequent mode switching retain, but do not multiply,
  // the already allocated T=2 scratch families.
  for (const pf::DirectBatchMode mode : {
           pf::DirectBatchMode::VALUE_ONLY,
           pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT,
           pf::DirectBatchMode::FULL_VGL})
    workspace->resize(mode, 64);
  CHECK(workspace->tileScratchBytes() == all_mode_high_water);
  CHECK(workspace->logicalStorageBytes() > 0);

  workspace->resize(pf::DirectBatchMode::VALUE_ONLY, 3);
  loadBatch(*workspace, base, 3);
  batch_executor.evaluateValues(*workspace);
  CHECK(workspace->tileScratchBytes() == all_mode_high_water);
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
