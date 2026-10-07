//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerAcceleratorSchedule.h
 * @brief Pointer-free device-arena and launch schedules for PsiFormer accelerators.
 *
 * The records in this file contain values only.  They can be validated in CPU-only
 * tests and then consumed by CUDA or HIP launch adapters without duplicating extent,
 * alignment, or tail-tile calculations.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_ACCELERATOR_SCHEDULE_H
#define QMCPLUSPLUS_PSIFORMER_ACCELERATOR_SCHEDULE_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace qmcplusplus::psiformer
{

/// Identify persistent and temporary regions within one prepared device arena.
enum class PsiFormerDeviceArenaRegion : std::uint8_t
{
  MODEL_PARAMETERS,
  MODEL_CONSTANTS,
  VALUE_WORKSPACE,
  SPATIAL_WORKSPACE,
  REVERSE_WORKSPACE,
  PARAMETER_ACCUMULATORS,
  ECP_TILE,
  HOST_STAGING,
  VENDOR_WORKSPACE,
  COUNT
};

/// Request one uniquely named, aligned region in a contiguous device arena.
struct PsiFormerDeviceArenaRequest
{
  PsiFormerDeviceArenaRegion region = PsiFormerDeviceArenaRegion::COUNT;
  std::size_t bytes                  = 0;
  std::size_t alignment              = 1;
};

/// Describe one canonical non-overlapping slice of a contiguous device arena.
struct PsiFormerDeviceArenaSlice
{
  PsiFormerDeviceArenaRegion region = PsiFormerDeviceArenaRegion::COUNT;
  std::size_t offset                 = 0;
  std::size_t bytes                  = 0;
  std::size_t alignment              = 1;

  /// Return the first byte after this slice.
  std::size_t end() const noexcept { return offset + bytes; }

  friend bool operator==(const PsiFormerDeviceArenaSlice& lhs,
                         const PsiFormerDeviceArenaSlice& rhs) noexcept
  {
    return lhs.region == rhs.region && lhs.offset == rhs.offset &&
        lhs.bytes == rhs.bytes && lhs.alignment == rhs.alignment;
  }
};

/// Hold the canonical layout and stable identity of one prepared device arena.
struct PsiFormerDeviceArenaLayout
{
  std::vector<PsiFormerDeviceArenaSlice> slices;
  std::size_t total_bytes = 0;
  std::uint64_t fingerprint = 0;

  /// Find a named slice, returning nullptr when the region was not requested.
  const PsiFormerDeviceArenaSlice* find(PsiFormerDeviceArenaRegion region) const noexcept;
};

/// Describe one one-dimensional launch without exposing backend runtime types.
struct PsiFormerLinearLaunch
{
  std::size_t item_count  = 0;
  std::size_t block_size  = 0;
  std::size_t block_count = 0;
  std::size_t tail_count  = 0;

  /// Report whether the launch contains no logical work.
  bool empty() const noexcept { return item_count == 0; }

  friend bool operator==(const PsiFormerLinearLaunch& lhs,
                         const PsiFormerLinearLaunch& rhs) noexcept
  {
    return lhs.item_count == rhs.item_count && lhs.block_size == rhs.block_size &&
        lhs.block_count == rhs.block_count && lhs.tail_count == rhs.tail_count;
  }
};

/// Describe one bounded contiguous item tile and its launch geometry.
struct PsiFormerLaunchTile
{
  std::size_t begin = 0;
  std::size_t count = 0;
  PsiFormerLinearLaunch launch;

  /// Return the first logical item after this tile.
  std::size_t end() const noexcept { return begin + count; }

  friend bool operator==(const PsiFormerLaunchTile& lhs,
                         const PsiFormerLaunchTile& rhs) noexcept
  {
    return lhs.begin == rhs.begin && lhs.count == rhs.count &&
        lhs.launch == rhs.launch;
  }
};

/// Hold an exact, canonical tiling of one logical launch domain.
struct PsiFormerTiledLaunchSchedule
{
  std::size_t total_items  = 0;
  std::size_t tile_capacity = 0;
  std::size_t block_size    = 0;
  std::vector<PsiFormerLaunchTile> tiles;
  std::uint64_t fingerprint = 0;
};

/** Build a canonical aligned device-arena layout.
 *
 * Requests may arrive in any order.  The returned slices are sorted by region ID so
 * equivalent inputs produce the same offsets and fingerprint.  Duplicate or invalid
 * region IDs, non-power-of-two alignments, and arithmetic overflow are rejected.
 */
PsiFormerDeviceArenaLayout makePsiFormerDeviceArenaLayout(
    const std::vector<PsiFormerDeviceArenaRequest>& requests);

/// Build one checked one-dimensional launch description.
PsiFormerLinearLaunch makePsiFormerLinearLaunch(std::size_t item_count,
                                                std::size_t block_size);

/// Divide a logical domain into exact bounded tiles with no gaps or overlap.
PsiFormerTiledLaunchSchedule makePsiFormerTiledLaunchSchedule(
    std::size_t total_items,
    std::size_t tile_capacity,
    std::size_t block_size);

/// Return a stable diagnostic name for one device-arena region.
const char* psiFormerDeviceArenaRegionName(PsiFormerDeviceArenaRegion region) noexcept;

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_ACCELERATOR_SCHEDULE_H
