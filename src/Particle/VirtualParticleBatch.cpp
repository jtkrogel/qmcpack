//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

/** @file VirtualParticleBatch.cpp
 * @brief Validation and accessors for ragged virtual-particle descriptors.
 */

#include "VirtualParticleBatch.h"

#include "ParticleSet.h"
#include "type_traits/RefVectorWithLeader.h"

#include <cstring>
#include <limits>
#include <memory>
#include <string>

namespace qmcplusplus
{
namespace
{
constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime        = 1099511628211ULL;

std::size_t checkedAdd(std::size_t left, std::size_t right, const char* description)
{
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(std::string("Virtual-particle batch size overflow in ") + description + ".");
  return left + right;
}

void mixByte(std::uint64_t& hash, unsigned char value) noexcept
{
  hash ^= value;
  hash *= fnv_prime;
}

void mixUint64(std::uint64_t& hash, std::uint64_t value) noexcept
{
  for (int byte = 0; byte < 8; ++byte)
  {
    mixByte(hash, static_cast<unsigned char>(value & UINT64_C(0xff)));
    value >>= 8;
  }
}

void mixReal(std::uint64_t& hash, QMCTraits::RealType value) noexcept
{
  static_assert(sizeof(QMCTraits::RealType) == sizeof(std::uint32_t) ||
                sizeof(QMCTraits::RealType) == sizeof(std::uint64_t));
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(value));
  mixByte(hash, static_cast<unsigned char>(sizeof(value)));
  mixUint64(hash, bits);
}

template<typename ParticleSetType>
void validateForImpl(const VirtualParticleBatch& batch,
                     const RefVectorWithLeader<ParticleSetType>& p_list,
                     std::optional<std::size_t> source_center_count)
{
  if (batch.walkerCount() != p_list.size())
    throw std::invalid_argument("Virtual-particle batch walker count does not match the reference-particle list.");

  for (std::size_t iw = 0; iw < p_list.size(); ++iw)
    for (std::size_t jw = 0; jw < iw; ++jw)
      if (std::addressof(p_list[iw]) == std::addressof(p_list[jw]))
        throw std::invalid_argument("Virtual-particle batch reference-particle list contains duplicate walkers.");

  for (std::size_t isegment = 0; isegment < batch.segmentCount(); ++isegment)
  {
    const VirtualParticleBatch::Segment& segment = batch.segment(isegment);
    const ParticleSet& reference = p_list[static_cast<std::size_t>(segment.walkerId())];
    if (static_cast<std::size_t>(segment.electronId()) >= reference.getTotalNum())
      throw std::invalid_argument("Virtual-particle batch electron index is out of range for its walker.");
    if (source_center_count && segment.isOnSphere() &&
        static_cast<std::size_t>(segment.sourceCenterId()) >= *source_center_count)
      throw std::invalid_argument("Virtual-particle batch source-center index is out of range.");
  }
}
} // namespace

namespace virtual_particle_batch_detail
{
bool isFiniteReal(QMCTraits::RealType value) noexcept
{
  if constexpr (sizeof(QMCTraits::RealType) == sizeof(std::uint32_t))
  {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
  }
  else
  {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
  }
}
} // namespace virtual_particle_batch_detail

VirtualParticleBatch::VirtualParticleBatch(std::size_t walker_count,
                                           OffsetView segment_offsets,
                                           SegmentView segments,
                                           PositionView absolute_positions)
    : walker_count_(walker_count),
      segment_offsets_(segment_offsets),
      segments_(segments),
      absolute_positions_(absolute_positions)
{
  validateIntrinsic();
}

void VirtualParticleBatch::validateIntrinsic() const
{
  const std::size_t expected_offset_count = checkedAdd(segments_.size(), 1, "segment-offset extent");
  if (segment_offsets_.size() != expected_offset_count)
    throw std::invalid_argument("Virtual-particle batch needs exactly one more offset than segments.");
  if (segment_offsets_[0] != 0)
    throw std::invalid_argument("Virtual-particle batch segment offsets must start at zero.");
  if (segment_offsets_[segments_.size()] != absolute_positions_.size())
    throw std::invalid_argument("Virtual-particle batch final segment offset does not match the position count.");

  for (std::size_t isegment = 0; isegment < segments_.size(); ++isegment)
  {
    const std::size_t begin = segment_offsets_[isegment];
    const std::size_t end   = segment_offsets_[isegment + 1];
    if (end < begin || end > absolute_positions_.size())
      throw std::invalid_argument("Virtual-particle batch segment offsets are not monotonic and in range.");
    if (begin == end)
      throw std::invalid_argument("Virtual-particle batch segments must not be empty.");

    const Segment& segment = segments_[isegment];
    if (segment.walkerId() < 0 || static_cast<std::size_t>(segment.walkerId()) >= walker_count_)
      throw std::invalid_argument("Virtual-particle batch walker index is out of range.");
    if (segment.electronId() < 0)
      throw std::invalid_argument("Virtual-particle batch electron indices must be nonnegative.");
    if (segment.isOnSphere() && segment.sourceCenterId() < 0)
      throw std::invalid_argument("On-sphere virtual-particle segments require a source center.");
    if (!segment.isOnSphere() && segment.sourceCenterId() != NO_SOURCE)
      throw std::invalid_argument("Off-sphere virtual-particle segments must not specify a source center.");
  }

  for (const PosType& position : absolute_positions_)
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
      if (!virtual_particle_batch_detail::isFiniteReal(position[idim]))
        throw std::invalid_argument("Virtual-particle batch absolute positions must be finite.");
}

const VirtualParticleBatch::Segment& VirtualParticleBatch::segment(std::size_t segment_index) const
{
  return segments_.at(segment_index);
}

VirtualParticleBatch::Slice VirtualParticleBatch::slice(std::size_t segment_index) const
{
  const Segment& selected = segment(segment_index);
  return Slice(selected, absolute_positions_.data(), segment_offsets_[segment_index],
               segment_offsets_[segment_index + 1]);
}

const VirtualParticleBatch::PosType& VirtualParticleBatch::Slice::absolutePosition(std::size_t local_index) const
{
  if (local_index >= size())
    throw std::out_of_range("Virtual-particle batch slice position is out of range.");
  return positions_[begin_ + local_index];
}

void VirtualParticleBatch::validateFor(const RefVectorWithLeader<ParticleSet>& p_list,
                                       std::optional<std::size_t> source_center_count) const
{
  validateForImpl(*this, p_list, source_center_count);
}

void VirtualParticleBatch::validateFor(const RefVectorWithLeader<const ParticleSet>& p_list,
                                       std::optional<std::size_t> source_center_count) const
{
  validateForImpl(*this, p_list, source_center_count);
}

void VirtualParticleBatch::validateOutputExtent(std::size_t output_extent) const
{
  if (output_extent != size())
    throw std::invalid_argument("Virtual-particle batch output extent does not match the position count.");
}

std::uint64_t VirtualParticleBatch::fingerprint() const noexcept
{
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
  static_assert(sizeof(IndexType) <= sizeof(std::uint64_t));
  static_assert(std::is_signed_v<IndexType>);
  constexpr std::uint64_t descriptor_tag = UINT64_C(0x5650424154434801);
  std::uint64_t hash                      = fnv_offset_basis;
  mixUint64(hash, descriptor_tag);
  mixUint64(hash, static_cast<std::uint64_t>(walker_count_));
  const std::size_t segment_count = segments_.size();
  const std::size_t position_count = absolute_positions_.size();
  mixUint64(hash, static_cast<std::uint64_t>(segment_count));
  mixUint64(hash, static_cast<std::uint64_t>(position_count));
  for (const std::size_t offset : segment_offsets_)
    mixUint64(hash, static_cast<std::uint64_t>(offset));
  for (const Segment& segment : segments_)
  {
    const IndexType walker_id        = segment.walkerId();
    const IndexType electron_id      = segment.electronId();
    const bool on_sphere             = segment.isOnSphere();
    const IndexType source_center_id = segment.sourceCenterId();
    mixUint64(hash, static_cast<std::uint64_t>(walker_id));
    mixUint64(hash, static_cast<std::uint64_t>(electron_id));
    mixUint64(hash, on_sphere ? UINT64_C(1) : UINT64_C(0));
    mixUint64(hash, static_cast<std::uint64_t>(source_center_id));
  }
  for (const PosType& position : absolute_positions_)
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
      mixReal(hash, position[idim]);
  return hash;
}

} // namespace qmcplusplus
