//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

/** @file NLPPVirtualBatch.h
 * @brief Bounded tile storage for flattened nonlocal-pseudopotential work.
 */

#ifndef QMCPLUSPLUS_NLPPVIRTUALBATCH_H
#define QMCPLUSPLUS_NLPPVIRTUALBATCH_H

#include "Configuration.h"
#include "NonLocalData.h"
#include "Particle/VirtualParticleBatch.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace qmcplusplus
{

/** Own ECP job metadata, logical outputs, and one bounded virtual-position tile.
 *
 * Jobs are appended in canonical group/walker/job order and remain O(job count).
 * Optional T-move candidates are final logical output and remain O(total knot
 * count).  Every other flat knot array is bounded by tileCapacity().  The class
 * deliberately does not traverse particles or evaluate a wavefunction; callers
 * fill each packed tile and then expose it through VirtualParticleBatch.
 */
class NLPPVirtualBatchStorage : public QMCTraits
{
public:
  /** Immutable identity and quadrature extent of one ion-electron job. */
  struct Job
  {
    IndexType group_id;
    IndexType walker_id;
    IndexType ion_id;
    IndexType electron_id;
    std::size_t walker_job_ordinal;
    std::size_t knot_count;
  };

  /** Identity of one whole or partial job resident in the current tile. */
  class TileSegment
  {
  public:
    std::size_t globalJobId() const noexcept { return global_job_id_; }
    IndexType groupId() const noexcept { return group_id_; }
    IndexType walkerId() const noexcept { return walker_id_; }
    IndexType ionId() const noexcept { return ion_id_; }
    IndexType electronId() const noexcept { return electron_id_; }
    std::size_t walkerJobOrdinal() const noexcept { return walker_job_ordinal_; }
    std::size_t firstKnot() const noexcept { return first_knot_; }
    std::size_t knotCount() const noexcept { return knot_count_; }
    std::size_t jobKnotCount() const noexcept { return job_knot_count_; }
    std::size_t tileOffset() const noexcept { return tile_offset_; }
    std::size_t globalKnotOffset() const noexcept { return global_knot_offset_; }
    std::size_t walkerKnotOffset() const noexcept { return walker_knot_offset_; }
    bool beginsJob() const noexcept { return first_knot_ == 0; }
    bool endsJob() const noexcept { return first_knot_ + knot_count_ == job_knot_count_; }

  private:
    friend class NLPPVirtualBatchStorage;

    TileSegment(std::size_t global_job_id,
                const Job& job,
                std::size_t first_knot,
                std::size_t knot_count,
                std::size_t tile_offset,
                std::size_t global_knot_offset,
                std::size_t walker_knot_offset) noexcept
        : global_job_id_(global_job_id),
          group_id_(job.group_id),
          walker_id_(job.walker_id),
          ion_id_(job.ion_id),
          electron_id_(job.electron_id),
          walker_job_ordinal_(job.walker_job_ordinal),
          first_knot_(first_knot),
          knot_count_(knot_count),
          tile_offset_(tile_offset),
          global_knot_offset_(global_knot_offset),
          walker_knot_offset_(walker_knot_offset),
          job_knot_count_(job.knot_count)
    {}

    std::size_t global_job_id_;
    IndexType group_id_;
    IndexType walker_id_;
    IndexType ion_id_;
    IndexType electron_id_;
    std::size_t walker_job_ordinal_;
    std::size_t first_knot_;
    std::size_t knot_count_;
    std::size_t tile_offset_;
    std::size_t global_knot_offset_;
    std::size_t walker_knot_offset_;
    std::size_t job_knot_count_;
  };

  /** Workload, packing, high-water, and separately classified byte counts. */
  struct Statistics
  {
    std::size_t logical_jobs            = 0;
    std::size_t logical_knots           = 0;
    std::size_t candidate_entries       = 0;
    std::size_t tiles_packed            = 0;
    std::size_t tile_segments_packed    = 0;
    std::size_t split_job_continuations = 0;
    std::size_t tail_tiles              = 0;
    std::size_t max_tile_occupancy      = 0;
    std::size_t max_tile_segments       = 0;
    std::size_t logical_job_bytes       = 0;
    std::size_t candidate_output_bytes  = 0;
    std::size_t bounded_tile_bytes      = 0;
  };

  /** Preallocate all tile-local arrays for a fixed nonzero knot capacity. */
  explicit NLPPVirtualBatchStorage(std::size_t tile_capacity);

  NLPPVirtualBatchStorage(const NLPPVirtualBatchStorage&) = delete;
  NLPPVirtualBatchStorage& operator=(const NLPPVirtualBatchStorage&) = delete;
  NLPPVirtualBatchStorage(NLPPVirtualBatchStorage&&) = default;
  NLPPVirtualBatchStorage& operator=(NLPPVirtualBatchStorage&&) = default;

  /** Create independent empty storage with the same capacity and output policy. */
  std::unique_ptr<NLPPVirtualBatchStorage> makeEmptyClone() const;

  /** Begin a new workload while retaining existing allocations. */
  void reset(std::size_t walker_count, bool stage_candidates);

  /** Append one validated job and return its stable global job identifier. */
  std::size_t appendJob(const Job& job);

  /** Finish logical construction and allocate only the requested final outputs. */
  void seal();

  /** Validate private logical output extents before an atomic public-state swap. */
  void validateLogicalOutputExtents() const;

  /** Rewind deterministic tile traversal and clear its counters. */
  void rewindTiles() noexcept;

  /** Pack the next group-bounded tile, returning false after the final tile. */
  bool packNextTile();

  /** Validate caller-filled positions, deltas, and bare weights for this tile. */
  void finalizeTileInput();

  /** Construct a read-only generic descriptor after finalizeTileInput().
   *
   * The returned non-owning view is valid only until the next non-const call on
   * this storage.  In particular, packNextTile() reuses its backing arrays.
   */
  VirtualParticleBatch makeVirtualParticleBatch() const;

  std::size_t walkerCount() const noexcept { return walker_count_; }
  std::size_t tileCapacity() const noexcept { return tile_capacity_; }
  std::size_t jobCount() const noexcept { return jobs_.size(); }
  std::size_t totalKnotCount() const noexcept { return total_knot_count_; }
  bool stagesCandidates() const noexcept { return stage_candidates_; }
  bool isSealed() const noexcept { return sealed_; }
  bool hasActiveTile() const noexcept { return tile_active_; }
  bool tileInputReady() const noexcept { return tile_input_ready_; }
  bool hasNextTile() const noexcept;

  const std::vector<Job>& jobs() const noexcept { return jobs_; }
  const std::vector<std::size_t>& jobOffsets() const noexcept { return job_offsets_; }
  const std::vector<std::size_t>& jobWalkerKnotOffsets() const noexcept { return job_walker_knot_offsets_; }
  const std::vector<std::size_t>& walkerCandidateCounts() const noexcept { return walker_candidate_counts_; }
  const std::vector<TileSegment>& tileSegments() const noexcept { return tile_segments_; }
  const std::vector<std::size_t>& tileSegmentOffsets() const noexcept { return tile_segment_offsets_; }

  /** Return persistent pair-energy accumulation for a possibly split job. */
  RealType& jobPairPotential(std::size_t global_job_id);
  const RealType& jobPairPotential(std::size_t global_job_id) const;

  /** Return optional final candidate staging for one walker, never tile scratch.
   *
   * Callers may update pre-sized elements but must not resize the vector.
   * validateLogicalOutputExtents() detects an accidental extent change before
   * public state is committed.
   */
  std::vector<NonLocalData>& walkerCandidates(std::size_t walker_id);
  const std::vector<NonLocalData>& walkerCandidates(std::size_t walker_id) const;

  /** Return writable geometric input and invalidate prior input finalization. */
  std::vector<PosType>& mutableTileAbsolutePositions();
  std::vector<PosType>& mutableTileDeltas();
  std::vector<RealType>& mutableTileBareWeights();

  /** Return tile-local ratio and reduction buffers after a tile is packed. */
  std::vector<ValueType>& mutableTileRatios();
  std::vector<ValueType>& mutableTileFermionicRatios();
  std::vector<ValueType>& mutableTileNonfermionicRatios();
  std::vector<RealType>& mutableTileTransformedWeights();

  const std::vector<PosType>& tileAbsolutePositions() const noexcept { return tile_absolute_positions_; }
  const std::vector<PosType>& tileDeltas() const noexcept { return tile_deltas_; }
  const std::vector<RealType>& tileBareWeights() const noexcept { return tile_bare_weights_; }
  const std::vector<ValueType>& tileRatios() const noexcept { return tile_ratios_; }
  const std::vector<ValueType>& tileFermionicRatios() const noexcept { return tile_fermionic_ratios_; }
  const std::vector<ValueType>& tileNonfermionicRatios() const noexcept { return tile_nonfermionic_ratios_; }
  const std::vector<RealType>& tileTransformedWeights() const noexcept { return tile_transformed_weights_; }

  /** Return deterministic identity hashes for the workload and current tile. */
  std::uint64_t logicalFingerprint() const noexcept;
  std::uint64_t tileFingerprint() const noexcept;

  /** Return a process-local allocation identity used by warmed-storage tests. */
  std::size_t storageFingerprint() const noexcept;

  /** Return current high-water counters and checked retained-byte accounting. */
  Statistics statistics() const;

private:
  void requireBuilding(const char* operation) const;
  void requireSealed(const char* operation) const;
  void requireActiveTile(const char* operation) const;
  void validateAppend(const Job& job) const;
  void reserveTileStorage();
  void resizeTileStorage(std::size_t occupancy);
  void invalidateTileInput() noexcept { tile_input_ready_ = false; }

  const std::size_t tile_capacity_;
  std::size_t walker_count_     = 0;
  std::size_t total_knot_count_ = 0;
  bool stage_candidates_        = false;
  bool sealed_                  = false;

  std::vector<Job> jobs_;
  std::vector<std::size_t> job_offsets_{0};
  std::vector<std::size_t> job_walker_knot_offsets_;
  std::vector<std::size_t> walker_candidate_counts_;
  std::vector<RealType> job_pair_potentials_;
  std::vector<std::vector<NonLocalData>> walker_candidates_;

  std::size_t next_job_index_ = 0;
  std::size_t next_job_knot_  = 0;
  bool tile_active_           = false;
  bool tile_input_ready_      = false;

  std::vector<TileSegment> tile_segments_;
  std::vector<std::size_t> tile_segment_offsets_;
  std::vector<VirtualParticleBatch::Segment> virtual_segments_;
  std::vector<PosType> tile_absolute_positions_;
  std::vector<PosType> tile_deltas_;
  std::vector<RealType> tile_bare_weights_;
  std::vector<ValueType> tile_ratios_;
  std::vector<ValueType> tile_fermionic_ratios_;
  std::vector<ValueType> tile_nonfermionic_ratios_;
  std::vector<RealType> tile_transformed_weights_;

  std::size_t tiles_packed_            = 0;
  std::size_t tile_segments_packed_    = 0;
  std::size_t split_job_continuations_ = 0;
  std::size_t tail_tiles_              = 0;
  std::size_t max_tile_occupancy_      = 0;
  std::size_t max_tile_segments_       = 0;
};

} // namespace qmcplusplus

#endif
