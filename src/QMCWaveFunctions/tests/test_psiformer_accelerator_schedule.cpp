//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_accelerator_schedule.cpp
 * @brief CPU tests for pointer-free PsiFormer device arena and launch schedules.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorSchedule.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus::psiformer
{
namespace
{

/// Record prepared allocations and launches without pretending to execute device math.
class RecordingAccelerator
{
public:
  /// Prepare exactly one arena before execution begins.
  void prepare(PsiFormerDeviceArenaLayout layout)
  {
    if (sealed_)
      throw std::logic_error("cannot prepare a sealed accelerator recorder");
    layout_ = std::move(layout);
    ++allocation_count_;
  }

  /// Close the allocation boundary used by steady-state launch tests.
  void seal() noexcept { sealed_ = true; }

  /// Record one launch only after its required arena region has been prepared.
  void launch(PsiFormerDeviceArenaRegion region, const PsiFormerLaunchTile& tile)
  {
    if (!sealed_ || layout_.find(region) == nullptr)
      throw std::logic_error("accelerator launch lacks a sealed prepared region");
    if (tile.count != tile.launch.item_count)
      throw std::invalid_argument("accelerator launch tile count is inconsistent");
    launches_.push_back(tile);
  }

  /// Return the number of simulated allocation calls made before sealing.
  std::size_t allocationCount() const noexcept { return allocation_count_; }

  /// Return the launches observed in submission order.
  const std::vector<PsiFormerLaunchTile>& launches() const noexcept { return launches_; }

private:
  PsiFormerDeviceArenaLayout layout_;
  std::vector<PsiFormerLaunchTile> launches_;
  std::size_t allocation_count_ = 0;
  bool sealed_ = false;
};

} // namespace

TEST_CASE("PsiFormer device arena layout is canonical and aligned",
          "[wavefunction][psiformer][accelerator]")
{
  const std::vector<PsiFormerDeviceArenaRequest> unordered{
      {PsiFormerDeviceArenaRegion::ECP_TILE, 17, 16},
      {PsiFormerDeviceArenaRegion::MODEL_PARAMETERS, 33, 32},
      {PsiFormerDeviceArenaRegion::VALUE_WORKSPACE, 65, 64},
      {PsiFormerDeviceArenaRegion::VENDOR_WORKSPACE, 0, 256}};

  const PsiFormerDeviceArenaLayout first = makePsiFormerDeviceArenaLayout(unordered);
  std::vector<PsiFormerDeviceArenaRequest> reversed = unordered;
  std::reverse(reversed.begin(), reversed.end());
  const PsiFormerDeviceArenaLayout second = makePsiFormerDeviceArenaLayout(reversed);

  REQUIRE(first.slices == second.slices);
  CHECK(first.fingerprint == second.fingerprint);
  CHECK(first.fingerprint != 0);
  CHECK(first.slices.size() == unordered.size());

  std::size_t previous_end = 0;
  for (const PsiFormerDeviceArenaSlice& slice : first.slices)
  {
    CHECK(slice.offset % slice.alignment == 0);
    CHECK(slice.offset >= previous_end);
    previous_end = slice.end();
  }
  CHECK(first.total_bytes == previous_end);
  REQUIRE(first.find(PsiFormerDeviceArenaRegion::VALUE_WORKSPACE));
  CHECK(first.find(PsiFormerDeviceArenaRegion::VALUE_WORKSPACE)->bytes == 65);
  CHECK(first.find(PsiFormerDeviceArenaRegion::SPATIAL_WORKSPACE) == nullptr);
  CHECK(std::string(psiFormerDeviceArenaRegionName(PsiFormerDeviceArenaRegion::ECP_TILE)) == "ecp_tile");
}

TEST_CASE("PsiFormer device arena rejects invalid requests",
          "[wavefunction][psiformer][accelerator]")
{
  CHECK_THROWS_AS(makePsiFormerDeviceArenaLayout({
                      {PsiFormerDeviceArenaRegion::MODEL_PARAMETERS, 1, 8},
                      {PsiFormerDeviceArenaRegion::MODEL_PARAMETERS, 2, 8}}),
                  std::invalid_argument);
  CHECK_THROWS_AS(makePsiFormerDeviceArenaLayout({
                      {PsiFormerDeviceArenaRegion::VALUE_WORKSPACE, 1, 3}}),
                  std::invalid_argument);
  CHECK_THROWS_AS(makePsiFormerDeviceArenaLayout({
                      {PsiFormerDeviceArenaRegion::COUNT, 1, 1}}),
                  std::invalid_argument);
  CHECK_THROWS_AS(makePsiFormerDeviceArenaLayout({
                      {PsiFormerDeviceArenaRegion::MODEL_PARAMETERS,
                       std::numeric_limits<std::size_t>::max(), 1},
                      {PsiFormerDeviceArenaRegion::MODEL_CONSTANTS, 1, 2}}),
                  std::length_error);
}

TEST_CASE("PsiFormer tiled launches cover domains exactly",
          "[wavefunction][psiformer][accelerator]")
{
  const PsiFormerTiledLaunchSchedule schedule =
      makePsiFormerTiledLaunchSchedule(/*total_items=*/103, /*tile_capacity=*/32,
                                       /*block_size=*/16);
  REQUIRE(schedule.tiles.size() == 4);
  CHECK(schedule.fingerprint != 0);

  std::size_t expected_begin = 0;
  std::size_t covered = 0;
  for (const PsiFormerLaunchTile& tile : schedule.tiles)
  {
    CHECK(tile.begin == expected_begin);
    CHECK(tile.count <= schedule.tile_capacity);
    CHECK(tile.launch.item_count == tile.count);
    CHECK(tile.launch.block_count == tile.count / schedule.block_size +
              (tile.count % schedule.block_size == 0 ? 0 : 1));
    expected_begin = tile.end();
    covered += tile.count;
  }
  CHECK(expected_begin == schedule.total_items);
  CHECK(covered == schedule.total_items);
  CHECK(schedule.tiles.back().count == 7);
  CHECK(schedule.tiles.back().launch.tail_count == 7);

  const PsiFormerTiledLaunchSchedule empty =
      makePsiFormerTiledLaunchSchedule(0, 8, 4);
  CHECK(empty.tiles.empty());
  CHECK(empty.fingerprint != 0);
  CHECK_THROWS_AS(makePsiFormerTiledLaunchSchedule(1, 0, 1), std::invalid_argument);
  CHECK_THROWS_AS(makePsiFormerTiledLaunchSchedule(1, 1, 0), std::invalid_argument);
}

TEST_CASE("PsiFormer recording backend enforces the preparation boundary",
          "[wavefunction][psiformer][accelerator]")
{
  RecordingAccelerator recorder;
  recorder.prepare(makePsiFormerDeviceArenaLayout({
      {PsiFormerDeviceArenaRegion::VALUE_WORKSPACE, 4096, 256}}));
  recorder.seal();

  const PsiFormerTiledLaunchSchedule schedule =
      makePsiFormerTiledLaunchSchedule(70, 24, 16);
  for (const PsiFormerLaunchTile& tile : schedule.tiles)
    recorder.launch(PsiFormerDeviceArenaRegion::VALUE_WORKSPACE, tile);

  CHECK(recorder.allocationCount() == 1);
  CHECK(recorder.launches() == schedule.tiles);
  CHECK_THROWS_AS(recorder.prepare(PsiFormerDeviceArenaLayout{}), std::logic_error);
  CHECK_THROWS_AS(recorder.launch(PsiFormerDeviceArenaRegion::SPATIAL_WORKSPACE,
                                  schedule.tiles.front()),
                  std::logic_error);
}

} // namespace qmcplusplus::psiformer
