//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_MCMULTIPARTICLEMOVES_H
#define QMCPLUSPLUS_MCMULTIPARTICLEMOVES_H

#include "Configuration.h"
#include "MCCoords.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace qmcplusplus
{
class ParticleSet;
template<typename T>
class RefVectorWithLeader;

/** Owning walker-major CSR description of one atomic multi-particle proposal per walker.
 *
 * walker_offsets partitions particle_indices and proposed_positions by walker.  Positions
 * are absolute proposed coordinates, not displacements.  Indices in every walker slice are
 * canonical (strictly increasing), which makes the descriptor unambiguous and fingerprintable.
 */
template<CoordsType CT>
class MCMultiParticleMoves;

template<>
class MCMultiParticleMoves<CoordsType::POS>
{
public:
  using IndexType = QMCTraits::IndexType;
  using PosType   = QMCTraits::PosType;

  /** Non-owning view of one walker's selected indices and absolute proposed positions. */
  class Slice
  {
  public:
    std::size_t size() const noexcept { return end_ - begin_; }
    bool empty() const noexcept { return begin_ == end_; }
    std::size_t flatOffset() const noexcept { return begin_; }

    IndexType particleIndex(std::size_t local_index) const;
    const PosType& proposedPosition(std::size_t local_index) const;

  private:
    friend class MCMultiParticleMoves<CoordsType::POS>;
    Slice(const MCMultiParticleMoves& moves, std::size_t begin, std::size_t end)
        : moves_(moves), begin_(begin), end_(end)
    {}

    const MCMultiParticleMoves& moves_;
    std::size_t begin_;
    std::size_t end_;
  };

  MCMultiParticleMoves(std::vector<std::size_t> walker_offsets,
                       std::vector<IndexType> particle_indices,
                       std::vector<PosType> proposed_positions);

  std::size_t walkerCount() const noexcept { return walker_offsets_.empty() ? 0 : walker_offsets_.size() - 1; }
  std::size_t size() const noexcept { return particle_indices_.size(); }
  bool empty() const noexcept { return particle_indices_.empty(); }

  Slice slice(std::size_t walker_index) const;

  const std::vector<std::size_t>& walkerOffsets() const noexcept { return walker_offsets_; }
  const std::vector<IndexType>& particleIndices() const noexcept { return particle_indices_; }
  const std::vector<PosType>& proposedPositions() const noexcept { return proposed_positions_; }

  /** Validate crowd-dependent invariants, including walker count and index bounds. */
  void validateFor(const RefVectorWithLeader<ParticleSet>& p_list) const;

  /** Deterministic exact-content fingerprint of the CSR partition, indices, and positions. */
  std::uint64_t fingerprint() const noexcept;

private:
  void validateIntrinsic() const;

  std::vector<std::size_t> walker_offsets_;
  std::vector<IndexType> particle_indices_;
  std::vector<PosType> proposed_positions_;
};

} // namespace qmcplusplus

#endif
