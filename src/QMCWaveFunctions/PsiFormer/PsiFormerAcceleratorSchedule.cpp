//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerAcceleratorSchedule.cpp
 * @brief Checked construction of pointer-free PsiFormer accelerator schedules.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorSchedule.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer
{
namespace
{

constexpr std::uint64_t FNV_OFFSET = 14695981039346656037ULL;
constexpr std::uint64_t FNV_PRIME  = 1099511628211ULL;

/// Mix one fixed-width integer into a stable little-endian FNV-1a hash.
void mixInteger(std::uint64_t& hash, std::uint64_t value) noexcept
{
  for (unsigned shift = 0; shift < 64; shift += 8)
  {
    hash ^= static_cast<std::uint8_t>((value >> shift) & 0xffU);
    hash *= FNV_PRIME;
  }
}

/// Return whether a requested alignment is valid for integer address arithmetic.
bool isPowerOfTwo(std::size_t value) noexcept
{
  return value != 0 && (value & (value - 1)) == 0;
}

/// Add two extents while rejecting wraparound.
std::size_t checkedAdd(std::size_t lhs, std::size_t rhs, const char* message)
{
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    throw std::length_error(message);
  return lhs + rhs;
}

/// Round an offset upward to a power-of-two alignment with checked arithmetic.
std::size_t checkedAlign(std::size_t offset, std::size_t alignment)
{
  if (!isPowerOfTwo(alignment))
    throw std::invalid_argument("PsiFormer device-arena alignment must be a nonzero power of two");
  const std::size_t mask = alignment - 1;
  return checkedAdd(offset, mask, "PsiFormer device-arena alignment overflowed") & ~mask;
}

} // namespace

const PsiFormerDeviceArenaSlice* PsiFormerDeviceArenaLayout::find(
    PsiFormerDeviceArenaRegion region) const noexcept
{
  const auto found = std::find_if(slices.begin(), slices.end(),
                                  [region](const PsiFormerDeviceArenaSlice& slice) {
                                    return slice.region == region;
                                  });
  return found == slices.end() ? nullptr : &*found;
}

PsiFormerDeviceArenaLayout makePsiFormerDeviceArenaLayout(
    const std::vector<PsiFormerDeviceArenaRequest>& requests)
{
  constexpr std::size_t region_count = static_cast<std::size_t>(PsiFormerDeviceArenaRegion::COUNT);
  std::array<bool, region_count> seen{};
  std::vector<PsiFormerDeviceArenaRequest> ordered = requests;

  for (const PsiFormerDeviceArenaRequest& request : ordered)
  {
    const std::size_t region = static_cast<std::size_t>(request.region);
    if (region >= region_count)
      throw std::invalid_argument("PsiFormer device-arena request has an invalid region");
    if (seen[region])
      throw std::invalid_argument("PsiFormer device-arena request repeats a region");
    if (!isPowerOfTwo(request.alignment))
      throw std::invalid_argument("PsiFormer device-arena alignment must be a nonzero power of two");
    seen[region] = true;
  }

  std::sort(ordered.begin(), ordered.end(),
            [](const PsiFormerDeviceArenaRequest& lhs, const PsiFormerDeviceArenaRequest& rhs) {
              return static_cast<std::uint8_t>(lhs.region) < static_cast<std::uint8_t>(rhs.region);
            });

  PsiFormerDeviceArenaLayout layout;
  layout.slices.reserve(ordered.size());
  for (const PsiFormerDeviceArenaRequest& request : ordered)
  {
    const std::size_t offset = checkedAlign(layout.total_bytes, request.alignment);
    layout.slices.push_back({request.region, offset, request.bytes, request.alignment});
    layout.total_bytes = checkedAdd(offset, request.bytes,
                                    "PsiFormer device-arena extent overflowed");
  }

  std::uint64_t hash = FNV_OFFSET;
  mixInteger(hash, layout.slices.size());
  for (const PsiFormerDeviceArenaSlice& slice : layout.slices)
  {
    mixInteger(hash, static_cast<std::uint8_t>(slice.region));
    mixInteger(hash, slice.offset);
    mixInteger(hash, slice.bytes);
    mixInteger(hash, slice.alignment);
  }
  mixInteger(hash, layout.total_bytes);
  layout.fingerprint = hash == 0 ? 1 : hash;
  return layout;
}

PsiFormerLinearLaunch makePsiFormerLinearLaunch(std::size_t item_count,
                                                std::size_t block_size)
{
  if (block_size == 0)
    throw std::invalid_argument("PsiFormer accelerator block size must be positive");

  const std::size_t tail_count = item_count % block_size;
  const std::size_t block_count = item_count / block_size + (tail_count == 0 ? 0 : 1);
  return {item_count, block_size, block_count, tail_count};
}

PsiFormerTiledLaunchSchedule makePsiFormerTiledLaunchSchedule(
    std::size_t total_items,
    std::size_t tile_capacity,
    std::size_t block_size)
{
  if (tile_capacity == 0)
    throw std::invalid_argument("PsiFormer accelerator tile capacity must be positive");

  PsiFormerTiledLaunchSchedule schedule;
  schedule.total_items   = total_items;
  schedule.tile_capacity = tile_capacity;
  schedule.block_size    = block_size;

  std::size_t begin = 0;
  while (begin < total_items)
  {
    const std::size_t count = std::min(tile_capacity, total_items - begin);
    schedule.tiles.push_back({begin, count, makePsiFormerLinearLaunch(count, block_size)});
    begin = checkedAdd(begin, count, "PsiFormer accelerator tile schedule overflowed");
  }

  std::uint64_t hash = FNV_OFFSET;
  mixInteger(hash, total_items);
  mixInteger(hash, tile_capacity);
  mixInteger(hash, block_size);
  mixInteger(hash, schedule.tiles.size());
  for (const PsiFormerLaunchTile& tile : schedule.tiles)
  {
    mixInteger(hash, tile.begin);
    mixInteger(hash, tile.count);
    mixInteger(hash, tile.launch.block_count);
    mixInteger(hash, tile.launch.tail_count);
  }
  schedule.fingerprint = hash == 0 ? 1 : hash;
  return schedule;
}

const char* psiFormerDeviceArenaRegionName(PsiFormerDeviceArenaRegion region) noexcept
{
  switch (region)
  {
  case PsiFormerDeviceArenaRegion::MODEL_PARAMETERS:
    return "model_parameters";
  case PsiFormerDeviceArenaRegion::MODEL_CONSTANTS:
    return "model_constants";
  case PsiFormerDeviceArenaRegion::VALUE_WORKSPACE:
    return "value_workspace";
  case PsiFormerDeviceArenaRegion::SPATIAL_WORKSPACE:
    return "spatial_workspace";
  case PsiFormerDeviceArenaRegion::REVERSE_WORKSPACE:
    return "reverse_workspace";
  case PsiFormerDeviceArenaRegion::PARAMETER_ACCUMULATORS:
    return "parameter_accumulators";
  case PsiFormerDeviceArenaRegion::ECP_TILE:
    return "ecp_tile";
  case PsiFormerDeviceArenaRegion::HOST_STAGING:
    return "host_staging";
  case PsiFormerDeviceArenaRegion::VENDOR_WORKSPACE:
    return "vendor_workspace";
  case PsiFormerDeviceArenaRegion::MODEL_PARAMETERS_STAGING:
    return "model_parameters_staging";
  case PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_0:
    return "model_compute_parameters_0";
  case PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_1:
    return "model_compute_parameters_1";
  case PsiFormerDeviceArenaRegion::PRECISION_CONVERSION_WORKSPACE:
    return "precision_conversion_workspace";
  case PsiFormerDeviceArenaRegion::NUMERICAL_DIAGNOSTICS:
    return "numerical_diagnostics";
  case PsiFormerDeviceArenaRegion::COUNT:
    return "invalid";
  }
  return "invalid";
}

} // namespace qmcplusplus::psiformer
