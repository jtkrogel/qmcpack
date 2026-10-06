//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#include "MCMultiParticleMoves.h"

#include "ParticleSet.h"
#include "type_traits/RefVectorWithLeader.h"

#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace qmcplusplus
{
namespace
{
constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime        = 1099511628211ULL;

void mixByte(std::uint64_t& hash, std::uint8_t value) noexcept
{
  hash ^= value;
  hash *= fnv_prime;
}

void mixUint64(std::uint64_t& hash, std::uint64_t value) noexcept
{
  for (int byte = 0; byte < 8; ++byte)
  {
    mixByte(hash, static_cast<std::uint8_t>(value & 0xffU));
    value >>= 8;
  }
}

void mixReal(std::uint64_t& hash, QMCTraits::RealType value) noexcept
{
  static_assert(std::is_floating_point_v<QMCTraits::RealType>);
  static_assert(sizeof(QMCTraits::RealType) == sizeof(std::uint32_t) ||
                sizeof(QMCTraits::RealType) == sizeof(std::uint64_t));
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(value));
  mixByte(hash, static_cast<std::uint8_t>(sizeof(value)));
  mixUint64(hash, bits);
}

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
} // namespace

MCMultiParticleMoves<CoordsType::POS>::MCMultiParticleMoves(std::vector<std::size_t> walker_offsets,
                                                            std::vector<IndexType> particle_indices,
                                                            std::vector<PosType> proposed_positions)
    : walker_offsets_(std::move(walker_offsets)),
      particle_indices_(std::move(particle_indices)),
      proposed_positions_(std::move(proposed_positions))
{
  validateIntrinsic();
}

void MCMultiParticleMoves<CoordsType::POS>::validateIntrinsic() const
{
  if (walker_offsets_.empty())
    throw std::invalid_argument("Multi-particle move walker offsets must contain the initial zero offset.");
  if (walker_offsets_.front() != 0)
    throw std::invalid_argument("Multi-particle move walker offsets must start at zero.");
  if (particle_indices_.size() != proposed_positions_.size())
    throw std::invalid_argument("Multi-particle move index and position counts do not match.");
  if (walker_offsets_.back() != particle_indices_.size())
    throw std::invalid_argument("Multi-particle move final walker offset does not match the entry count.");

  for (std::size_t iw = 0; iw + 1 < walker_offsets_.size(); ++iw)
  {
    const std::size_t begin = walker_offsets_[iw];
    const std::size_t end   = walker_offsets_[iw + 1];
    if (end < begin || end > particle_indices_.size())
      throw std::invalid_argument("Multi-particle move walker offsets are not monotonic and in range.");
    for (std::size_t entry = begin; entry < end; ++entry)
    {
      if (particle_indices_[entry] < 0)
        throw std::invalid_argument("Multi-particle move particle indices must be nonnegative.");
      if (entry != begin && particle_indices_[entry - 1] >= particle_indices_[entry])
        throw std::invalid_argument("Multi-particle move indices must be strictly increasing within each walker.");
    }
  }

  for (const PosType& position : proposed_positions_)
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
      if (!isFiniteReal(position[idim]))
        throw std::invalid_argument("Multi-particle move proposed positions must be finite.");
}

MCMultiParticleMoves<CoordsType::POS>::IndexType
MCMultiParticleMoves<CoordsType::POS>::Slice::particleIndex(std::size_t local_index) const
{
  if (local_index >= size())
    throw std::out_of_range("Multi-particle move slice index is out of range.");
  return moves_.particle_indices_[begin_ + local_index];
}

const MCMultiParticleMoves<CoordsType::POS>::PosType&
MCMultiParticleMoves<CoordsType::POS>::Slice::proposedPosition(std::size_t local_index) const
{
  if (local_index >= size())
    throw std::out_of_range("Multi-particle move slice position is out of range.");
  return moves_.proposed_positions_[begin_ + local_index];
}

MCMultiParticleMoves<CoordsType::POS>::Slice MCMultiParticleMoves<CoordsType::POS>::slice(
    std::size_t walker_index) const
{
  if (walker_index >= walkerCount())
    throw std::out_of_range("Multi-particle move walker index is out of range.");
  return Slice(*this, walker_offsets_[walker_index], walker_offsets_[walker_index + 1]);
}

void MCMultiParticleMoves<CoordsType::POS>::validateFor(const RefVectorWithLeader<ParticleSet>& p_list) const
{
  if (p_list.empty())
    throw std::invalid_argument("Multi-particle move crowd must not be empty.");
  if (walkerCount() != p_list.size())
    throw std::invalid_argument("Multi-particle move walker count does not match the crowd.");

  const std::size_t num_particles = p_list.getLeader().getTotalNum();
  for (std::size_t iw = 0; iw < p_list.size(); ++iw)
  {
    if (p_list[iw].getTotalNum() != num_particles)
      throw std::invalid_argument("Multi-particle move crowd particle counts do not match.");
    const Slice walker_slice = slice(iw);
    for (std::size_t entry = 0; entry < walker_slice.size(); ++entry)
      if (static_cast<std::size_t>(walker_slice.particleIndex(entry)) >= num_particles)
        throw std::invalid_argument("Multi-particle move particle index is out of range.");
  }
}

std::uint64_t MCMultiParticleMoves<CoordsType::POS>::fingerprint() const noexcept
{
  std::uint64_t hash = fnv_offset_basis;
  mixUint64(hash, static_cast<std::uint64_t>(CoordsType::POS));
  mixUint64(hash, static_cast<std::uint64_t>(walker_offsets_.size()));
  for (const std::size_t offset : walker_offsets_)
    mixUint64(hash, static_cast<std::uint64_t>(offset));
  mixUint64(hash, static_cast<std::uint64_t>(particle_indices_.size()));
  for (const IndexType index : particle_indices_)
    mixUint64(hash, static_cast<std::uint64_t>(index));
  for (const PosType& position : proposed_positions_)
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
      mixReal(hash, position[idim]);
  return hash;
}

} // namespace qmcplusplus
