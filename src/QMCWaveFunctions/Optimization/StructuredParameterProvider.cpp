//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file StructuredParameterProvider.cpp
 * @brief Validation and stable identities for structured parameter schemas.
 */

#include "QMCWaveFunctions/Optimization/StructuredParameterProvider.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace qmcplusplus::wftrain
{
namespace
{

/// Initial state and multiplier for deterministic FNV-1a schema identities.
constexpr std::uint64_t FNV_OFFSET = UINT64_C(14695981039346656037);
constexpr std::uint64_t FNV_PRIME  = UINT64_C(1099511628211);

/// Add one fixed-width integer to a deterministic schema hash.
void mixInteger(std::uint64_t& hash, std::uint64_t value) noexcept
{
  for (int byte = 0; byte < 8; ++byte)
  {
    hash ^= static_cast<std::uint8_t>(value >> (8 * byte));
    hash *= FNV_PRIME;
  }
}

/// Add one length-delimited string to a deterministic schema hash.
void mixString(std::uint64_t& hash, const std::string& value) noexcept
{
  mixInteger(hash, value.size());
  for (const unsigned char character : value)
  {
    hash ^= character;
    hash *= FNV_PRIME;
  }
}

/// Compute one tensor's scalar extent while detecting invalid or overflowing shapes.
std::size_t shapeProduct(const std::vector<std::size_t>& shape)
{
  std::size_t count = 1;
  for (const std::size_t extent : shape)
  {
    if (extent == 0)
      throw std::invalid_argument("Structured parameter block has a zero tensor extent");
    if (count > std::numeric_limits<std::size_t>::max() / extent)
      throw std::overflow_error("Structured parameter tensor extent overflows size_t");
    count *= extent;
  }
  return count;
}

} // namespace

StructuredParameterSchema::StructuredParameterSchema(
    std::string provider_id,
    std::vector<ParameterBlockDescriptor> blocks)
    : provider_id_(std::move(provider_id)), blocks_(std::move(blocks))
{
  if (provider_id_.empty())
    throw std::invalid_argument("Structured parameter provider identity must not be empty");
  if (blocks_.empty())
    throw std::invalid_argument("Structured parameter schema must contain at least one block");

  std::set<std::string> identifiers;
  std::size_t next_offset = 0;
  for (const ParameterBlockDescriptor& block : blocks_)
  {
    if (block.id.empty())
      throw std::invalid_argument("Structured parameter block identity must not be empty");
    if (!identifiers.insert(block.id).second)
      throw std::invalid_argument("Duplicate structured parameter block identity: " + block.id);
    if (block.offset != next_offset)
      throw std::invalid_argument("Structured parameter blocks must be contiguous and ordered");

    const std::size_t expected_count = shapeProduct(block.shape);
    if (block.count != expected_count)
      throw std::invalid_argument("Structured parameter block count disagrees with its tensor shape");
    if (next_offset > std::numeric_limits<std::size_t>::max() - block.count)
      throw std::overflow_error("Structured parameter schema size overflows size_t");
    next_offset += block.count;
  }
  parameter_count_ = next_offset;

  std::uint64_t hash = FNV_OFFSET;
  mixString(hash, provider_id_);
  mixInteger(hash, blocks_.size());
  for (const ParameterBlockDescriptor& block : blocks_)
  {
    mixString(hash, block.id);
    mixInteger(hash, block.shape.size());
    for (const std::size_t extent : block.shape)
      mixInteger(hash, extent);
    mixInteger(hash, block.offset);
    mixInteger(hash, block.count);
    mixInteger(hash, static_cast<std::uint64_t>(block.scalar_domain));
    mixInteger(hash, block.trainable ? 1 : 0);
    mixString(hash, block.update_group);
  }

  std::ostringstream formatted;
  formatted << std::hex << std::setfill('0') << std::setw(16) << hash;
  fingerprint_ = formatted.str();
}

} // namespace qmcplusplus::wftrain

