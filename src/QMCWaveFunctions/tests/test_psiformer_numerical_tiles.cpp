//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_numerical_tiles.cpp
 * @brief External-data-free tests of numerical tile publication and failures.
 */
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerBatchExecutor.h"
#include "psiformer_test_utils.h"

namespace
{

/** Return deliberately synthetic strided data, isolating publication from physics. */
class SyntheticTileBackend final : public pf::DirectBatchTileBackend
{
public:
  /// Retain only the electron count needed to generate independent lane labels.
  explicit SyntheticTileBackend(std::size_t electrons) : electrons_(electrons) {}

  /// Preallocate padded result storage; evaluation must never grow these vectors.
  std::unique_ptr<pf::DirectBatchTileBackendWorkspace> prepare(std::size_t capacity) const override
  {
    ++preparations;
    return std::make_unique<Context>(capacity, electrons_);
  }

  /// Label every lane from its input coordinate, not from the local tile index.
  pf::DirectBatchTileResultView evaluate(pf::DirectBatchTileBackendWorkspace& workspace,
      pf::DirectBatchMode, const double* positions, std::size_t count, std::size_t version) const override
  {
    auto& context = dynamic_cast<Context&>(workspace);
    ++calls;
    for (std::size_t row = 0; row < count; ++row)
    {
      const double tag = positions[row * electrons_ * 3];
      context.sign[row] = 1;
      context.values[row * 5] = -tag;
      for (std::size_t lane = 0; lane < electrons_ * 3; ++lane)
        context.gradient[row * (3 * electrons_ + 2) + lane] = tag + 0.01 * lane;
      for (std::size_t electron = 0; electron < electrons_; ++electron)
      {
        context.lap_log[row * (electrons_ + 2) + electron] = tag - 1 - electron;
        context.lap_ratio[row * (electrons_ + 1) + electron] = tag + 1 + electron;
      }
    }
    if (calls == nonfinite_call)
      context.values[0] = std::numeric_limits<double>::quiet_NaN();
    if (calls == nodal_call)
    {
      context.sign[0] = 0;
      context.values[0] = -std::numeric_limits<double>::infinity();
    }
    return {count, wrong_version ? version + 1 : version, context.sign.data(), context.values.data(), 5,
            context.gradient.data(), 3 * electrons_ + 2, 1,
            context.lap_log.data(), electrons_ + 2, 1,
            context.lap_ratio.data(), electrons_ + 1};
  }

  mutable std::size_t calls = 0, preparations = 0;
  std::size_t nonfinite_call = std::numeric_limits<std::size_t>::max();
  std::size_t nodal_call = std::numeric_limits<std::size_t>::max();
  bool wrong_version = false;

private:
  /** Own padded numerical result arrays without model data or runtime handles. */
  struct Context final : pf::DirectBatchTileBackendWorkspace
  {
    /// Allocate every lane at preparation time, including intentionally unused padding.
    Context(std::size_t capacity, std::size_t electrons)
        : sign(capacity), values(capacity * 5), gradient(capacity * (3 * electrons + 2)),
          lap_log(capacity * (electrons + 2)), lap_ratio(capacity * (electrons + 1))
    {}

    std::vector<double> sign, values, gradient, lap_log, lap_ratio;
  };

  const std::size_t electrons_;
};

} // namespace

/// Test tails, arbitrary strides and active lanes through the ordinary native ABI.
TEST_CASE("PsiFormer numerical tiles preserve atomic native publication",
          "[wavefunction][psiformer][batch][numerical-tiles]")
{
  using namespace qmcplusplus::testing::psiformer;
  auto files = generateFiles("lih");
  pf::PsiFormer model(files.parameters.string(), files.configuration.string());
  const auto plan = qmcplusplus::psiformer::PsiFormerExecutionPlan::fromParameters(
      model.p, {model.cfg.nup, model.cfg.ndown, 2, model.ndet, model.dim, model.heads, model.blocks});
  pf::DirectValueExecutor value(model, plan);
  pf::DirectSpatialExecutor spatial(model, value, plan);
  auto tiles = std::make_shared<SyntheticTileBackend>(model.ne);
  pf::DirectBatchExecutor batch(value, spatial, tiles);
  auto workspace = batch.makeWorkspace();
  workspace->prepareTileCapacity(2);
  const std::size_t count = 5;
  std::vector<std::size_t> active{0, 1, 2, 3, 0};
  auto pack = [&](pf::DirectBatchMode mode) {
    workspace->resize(mode, count);
    for (std::size_t row = 0; row < count; ++row)
    {
      auto positions = model.cfg.configuration(0);
      positions.x[0] = 0.1 + 0.15 * row;
      workspace->setPositions(row, pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
    }
  };

  pack(pf::DirectBatchMode::VALUE_ONLY);
  auto published = batch.evaluateValues(*workspace);
  for (std::size_t row = 0; row < count; ++row)
    CHECK(published.logabs[row] == Catch::Approx(-0.1 - 0.15 * row));
  CHECK(tiles->preparations == 1);
  CHECK(tiles->calls == 3);

  pack(pf::DirectBatchMode::FULL_VGL);
  const auto full = batch.evaluateFull(*workspace);
  for (std::size_t row = 0; row < count; ++row)
  {
    const double tag = 0.1 + 0.15 * row;
    for (std::size_t lane = 0; lane < 3 * model.ne; ++lane)
      CHECK(full.gradient[row * full.gradient_stride + lane] == Catch::Approx(tag + 0.01 * lane));
    for (std::size_t electron = 0; electron < model.ne; ++electron)
    {
      CHECK(full.lap_log[row * model.ne + electron] == Catch::Approx(tag - 1 - electron));
      CHECK(full.lap_ratio[row * model.ne + electron] == Catch::Approx(tag + 1 + electron));
    }
  }
  pack(pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT);
  const auto selected = batch.evaluateActive(*workspace, active.data());
  for (std::size_t row = 0; row < count; ++row)
    for (std::size_t lane = 0; lane < 3; ++lane)
      CHECK(selected.gradient[row * 3 + lane] == Catch::Approx(0.1 + 0.15 * row + 0.01 * (3 * active[row] + lane)));
  CHECK(tiles->preparations == 1);

  // Failed later tiles or mismatched versions must not publish partial rows.
  pack(pf::DirectBatchMode::VALUE_ONLY);
  published = batch.evaluateValues(*workspace);
  const std::vector<double> snapshot(published.logabs, published.logabs + count);
  tiles->nonfinite_call = tiles->calls + 2;
  REQUIRE_THROWS_AS(batch.evaluateValues(*workspace), std::runtime_error);
  for (std::size_t row = 0; row < count; ++row) CHECK(published.logabs[row] == snapshot[row]);
  tiles->nonfinite_call = std::numeric_limits<std::size_t>::max();
  tiles->wrong_version = true;
  REQUIRE_THROWS_AS(batch.evaluateValues(*workspace), std::logic_error);
  for (std::size_t row = 0; row < count; ++row) CHECK(published.logabs[row] == snapshot[row]);
  tiles->wrong_version = false;

  // Sparse inputs reuse the established reference-plus-replacement ordering.
  workspace->resizeSparseValues(1, 1);
  auto positions = model.cfg.configuration(0);
  positions.x[0] = 0.2;
  workspace->setReferenceConfiguration(0, pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  workspace->setVirtualReplacement(0, 0, 0, {0.7, positions.x[1], positions.x[2]});
  const auto sparse = batch.evaluateValues(*workspace);
  CHECK(sparse.logabs[0] == Catch::Approx(-0.2));
  CHECK(sparse.logabs[1] == Catch::Approx(-0.7));

  // Test the node's bits rather than a fast-math-sensitive Catch infinity comparison.
  workspace->resize(pf::DirectBatchMode::VALUE_ONLY, 1);
  workspace->setPositions(0, pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  tiles->nodal_call = tiles->calls + 1;
  const auto node = batch.evaluateValues(*workspace);
  CHECK(node.sign[0] == 0);
  CHECK(node.value[0] == 0);
  std::uint64_t bits;
  std::memcpy(&bits, node.logabs, sizeof(bits));
  CHECK(bits == UINT64_C(0xfff0000000000000));
}
