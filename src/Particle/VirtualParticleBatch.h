//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

/** @file VirtualParticleBatch.h
 * @brief Immutable non-owning description of ragged virtual-particle positions.
 */
#ifndef QMCPLUSPLUS_VIRTUALPARTICLEBATCH_H
#define QMCPLUSPLUS_VIRTUALPARTICLEBATCH_H

#include "Configuration.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace qmcplusplus
{
class ParticleSet;
template<typename T>
class RefVectorWithLeader;

namespace virtual_particle_batch_detail
{
/** Bit-safe even under release builds compiled with -ffast-math. */
bool isFiniteReal(QMCTraits::RealType value) noexcept;
} // namespace virtual_particle_batch_detail

/** Read-only CSR description of virtual-particle replacement positions.
 *
 * Every segment identifies one reference walker and one moved electron.  Its
 * slice of absolute_positions contains exact replacement coordinates in caller
 * order.  A segment may also carry the source-center context required by
 * on-sphere virtual-particle evaluation.  Hamiltonian-specific job, knot, and
 * output-scatter identities intentionally do not belong in this descriptor.
 *
 * The descriptor does not own its arrays.  The caller must keep their storage
 * alive and unchanged for the descriptor's lifetime.
 */
class VirtualParticleBatch
{
public:
  using IndexType = QMCTraits::IndexType;
  using RealType  = QMCTraits::RealType;
  using PosType   = QMCTraits::PosType;

  static constexpr IndexType NO_SOURCE = -1;

  /** Minimal C++17 replacement for a read-only std::span. */
  template<typename T>
  class ReadOnlyView
  {
  public:
    ReadOnlyView() noexcept = default;

    ReadOnlyView(const T* data, std::size_t size) : data_(data), size_(size)
    {
      if (size_ != 0 && data_ == nullptr)
        throw std::invalid_argument("Virtual-particle batch view has a null data pointer for nonzero size.");
    }

    template<typename Allocator>
    ReadOnlyView(const std::vector<T, Allocator>& storage) noexcept : data_(storage.data()), size_(storage.size())
    {}

    template<typename Allocator>
    ReadOnlyView(std::vector<T, Allocator>&&) = delete;

    template<typename Allocator>
    ReadOnlyView(const std::vector<T, Allocator>&&) = delete;

    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    const T& operator[](std::size_t index) const noexcept { return data_[index]; }

    const T& at(std::size_t index) const
    {
      if (index >= size_)
        throw std::out_of_range("Virtual-particle batch view index is out of range.");
      return data_[index];
    }

    const T* begin() const noexcept { return data_; }
    const T* end() const noexcept { return size_ == 0 ? data_ : data_ + size_; }

  private:
    const T* data_    = nullptr;
    std::size_t size_ = 0;
  };

  /** Context needed to reproduce one VirtualParticleSet call. */
  class Segment
  {
  public:
    Segment(IndexType walker_id,
            IndexType electron_id,
            bool on_sphere = false,
            IndexType source_center_id = NO_SOURCE) noexcept
        : walker_id_(walker_id),
          electron_id_(electron_id),
          on_sphere_(on_sphere),
          source_center_id_(source_center_id)
    {}

    IndexType walkerId() const noexcept { return walker_id_; }
    IndexType electronId() const noexcept { return electron_id_; }
    bool isOnSphere() const noexcept { return on_sphere_; }
    IndexType sourceCenterId() const noexcept { return source_center_id_; }

  private:
    IndexType walker_id_;
    IndexType electron_id_;
    bool on_sphere_;
    IndexType source_center_id_;
  };

  using OffsetView   = ReadOnlyView<std::size_t>;
  using SegmentView  = ReadOnlyView<Segment>;
  using PositionView = ReadOnlyView<PosType>;

  /** Checked view of one segment and its exact absolute positions. */
  class Slice
  {
  public:
    std::size_t size() const noexcept { return end_ - begin_; }
    bool empty() const noexcept { return begin_ == end_; }
    std::size_t flatOffset() const noexcept { return begin_; }

    IndexType walkerId() const noexcept { return segment_.walkerId(); }
    IndexType electronId() const noexcept { return segment_.electronId(); }
    bool isOnSphere() const noexcept { return segment_.isOnSphere(); }
    IndexType sourceCenterId() const noexcept { return segment_.sourceCenterId(); }

    const PosType& absolutePosition(std::size_t local_index) const;
    PositionView positions() const noexcept { return PositionView(positions_ + begin_, size()); }

  private:
    friend class VirtualParticleBatch;
    Slice(const Segment& segment, const PosType* positions, std::size_t begin, std::size_t end) noexcept
        : segment_(segment), positions_(positions), begin_(begin), end_(end)
    {}

    const Segment& segment_;
    const PosType* positions_;
    std::size_t begin_;
    std::size_t end_;
  };

  VirtualParticleBatch(std::size_t walker_count,
                       OffsetView segment_offsets,
                       SegmentView segments,
                       PositionView absolute_positions);

  VirtualParticleBatch(const VirtualParticleBatch&) = default;
  VirtualParticleBatch& operator=(const VirtualParticleBatch&) = delete;

  std::size_t walkerCount() const noexcept { return walker_count_; }
  std::size_t segmentCount() const noexcept { return segments_.size(); }
  std::size_t size() const noexcept { return absolute_positions_.size(); }
  bool empty() const noexcept { return absolute_positions_.empty(); }

  OffsetView segmentOffsets() const noexcept { return segment_offsets_; }
  SegmentView segments() const noexcept { return segments_; }
  PositionView absolutePositions() const noexcept { return absolute_positions_; }

  const Segment& segment(std::size_t segment_index) const;
  Slice slice(std::size_t segment_index) const;

  /** Validate unique ParticleSet references and walker/electron bounds.
   *
   * Several segments may intentionally identify the same walker.
   * When source_center_count is supplied, every on-sphere source center is
   * also required to lie in [0, source_center_count).
   */
  void validateFor(const RefVectorWithLeader<ParticleSet>& p_list,
                   std::optional<std::size_t> source_center_count = std::nullopt) const;
  void validateFor(const RefVectorWithLeader<const ParticleSet>& p_list,
                   std::optional<std::size_t> source_center_count = std::nullopt) const;

  /** Require one output entry per flat virtual position. */
  void validateOutputExtent(std::size_t output_extent) const;

  /** Deterministic exact-content fingerprint of shape, context, and positions. */
  std::uint64_t fingerprint() const noexcept;

private:
  void validateIntrinsic() const;

  const std::size_t walker_count_;
  const OffsetView segment_offsets_;
  const SegmentView segments_;
  const PositionView absolute_positions_;
};

} // namespace qmcplusplus

#endif
