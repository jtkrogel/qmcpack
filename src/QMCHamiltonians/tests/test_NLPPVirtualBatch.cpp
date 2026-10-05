//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_NLPPVirtualBatch.cpp
 * @brief Focused tests for bounded nonlocal-ECP virtual-position tiling.
 */

#include <catch2/catch_test_macros.hpp>

#include "QMCHamiltonians/NLPPVirtualBatch.h"

#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace qmcplusplus
{
namespace
{
using Storage = NLPPVirtualBatchStorage;

/** Fill every required geometric input with finite values derived from global order. */
void fillTileInput(Storage& storage)
{
  auto& positions = storage.mutableTileAbsolutePositions();
  auto& deltas    = storage.mutableTileDeltas();
  auto& weights   = storage.mutableTileBareWeights();
  for (const Storage::TileSegment& segment : storage.tileSegments())
    for (std::size_t local = 0; local < segment.knotCount(); ++local)
    {
      const std::size_t tile_index   = segment.tileOffset() + local;
      const std::size_t global_index = segment.globalKnotOffset() + local;
      const Storage::RealType value  = static_cast<Storage::RealType>(global_index + 1);
      positions[tile_index]          = {value, value + 0.25, value + 0.5};
      deltas[tile_index]             = {-value, value * 0.5, value * 0.25};
      weights[tile_index]            = value * 0.125;
    }
}

/** Capture the identity-bearing fields of one tile segment for exact comparison. */
auto segmentTuple(const Storage::TileSegment& segment)
{
  return std::make_tuple(segment.globalJobId(), segment.groupId(), segment.walkerId(), segment.ionId(),
                         segment.electronId(), segment.walkerJobOrdinal(), segment.firstKnot(),
                         segment.knotCount(), segment.tileOffset(), segment.globalKnotOffset(),
                         segment.walkerKnotOffset(), segment.beginsJob(), segment.endsJob());
}
} // namespace

TEST_CASE("NLPPVirtualBatch preserves canonical identity across split tiles", "[hamiltonian][nlpp_batch]")
{
  Storage storage(4);
  storage.reset(3, true);
  CHECK(storage.appendJob({0, 0, 2, 1, 0, 3}) == 0);
  CHECK(storage.appendJob({0, 0, 1, 2, 1, 5}) == 1);
  CHECK(storage.appendJob({0, 2, 0, 4, 0, 2}) == 2);
  CHECK(storage.appendJob({1, 0, 3, 5, 0, 3}) == 3);

  const std::vector<std::size_t> expected_job_offsets{0, 3, 8, 10, 13};
  const std::vector<std::size_t> expected_walker_job_offsets{0, 3, 0, 8};
  const std::vector<std::size_t> expected_walker_candidate_counts{11, 0, 2};
  CHECK(storage.jobOffsets() == expected_job_offsets);
  CHECK(storage.jobWalkerKnotOffsets() == expected_walker_job_offsets);
  CHECK(storage.walkerCandidateCounts() == expected_walker_candidate_counts);
  const std::uint64_t logical_fingerprint = storage.logicalFingerprint();
  storage.seal();

  CHECK(storage.isSealed());
  CHECK(storage.jobCount() == 4);
  CHECK(storage.totalKnotCount() == 13);
  CHECK(storage.walkerCandidates(0).size() == 11);
  CHECK(storage.walkerCandidates(1).empty());
  CHECK(storage.walkerCandidates(2).size() == 2);
  CHECK_NOTHROW(storage.validateLogicalOutputExtents());
  CHECK(storage.logicalFingerprint() == logical_fingerprint);
  REQUIRE_THROWS_AS(storage.makeVirtualParticleBatch(), std::logic_error);

  using SegmentTuple = decltype(segmentTuple(std::declval<const Storage::TileSegment&>()));
  const std::vector<std::vector<SegmentTuple>> expected_tiles{
      {std::make_tuple(std::size_t(0), 0, 0, 2, 1, std::size_t(0), std::size_t(0), std::size_t(3),
                       std::size_t(0), std::size_t(0), std::size_t(0), true, true),
       std::make_tuple(std::size_t(1), 0, 0, 1, 2, std::size_t(1), std::size_t(0), std::size_t(1),
                       std::size_t(3), std::size_t(3), std::size_t(3), true, false)},
      {std::make_tuple(std::size_t(1), 0, 0, 1, 2, std::size_t(1), std::size_t(1), std::size_t(4),
                       std::size_t(0), std::size_t(4), std::size_t(4), false, true)},
      {std::make_tuple(std::size_t(2), 0, 2, 0, 4, std::size_t(0), std::size_t(0), std::size_t(2),
                       std::size_t(0), std::size_t(8), std::size_t(0), true, true)},
      {std::make_tuple(std::size_t(3), 1, 0, 3, 5, std::size_t(0), std::size_t(0), std::size_t(3),
                       std::size_t(0), std::size_t(10), std::size_t(8), true, true)}};

  std::size_t tile_index = 0;
  while (storage.packNextTile())
  {
    REQUIRE(tile_index < expected_tiles.size());
    const auto& expected = expected_tiles[tile_index];
    REQUIRE(storage.tileSegments().size() == expected.size());
    for (std::size_t segment_index = 0; segment_index < expected.size(); ++segment_index)
      CHECK(segmentTuple(storage.tileSegments()[segment_index]) == expected[segment_index]);

    // Packing poisons geometric input, preventing an accidentally stale view.
    REQUIRE_FALSE(storage.tileInputReady());
    REQUIRE_THROWS_AS(storage.finalizeTileInput(), std::invalid_argument);
    REQUIRE_THROWS_AS(storage.makeVirtualParticleBatch(), std::logic_error);

    const std::uint64_t tile_fingerprint = storage.tileFingerprint();
    fillTileInput(storage);
    storage.finalizeTileInput();
    CHECK(storage.tileInputReady());
    CHECK(storage.tileFingerprint() == tile_fingerprint);

    const VirtualParticleBatch batch = storage.makeVirtualParticleBatch();
    CHECK(batch.walkerCount() == 3);
    CHECK(batch.segmentCount() == expected.size());
    CHECK(batch.size() == storage.tileAbsolutePositions().size());
    for (std::size_t segment_index = 0; segment_index < batch.segmentCount(); ++segment_index)
    {
      const auto& ecp_segment             = storage.tileSegments()[segment_index];
      const VirtualParticleBatch::Slice slice = batch.slice(segment_index);
      CHECK(slice.walkerId() == ecp_segment.walkerId());
      CHECK(slice.electronId() == ecp_segment.electronId());
      CHECK(slice.isOnSphere());
      CHECK(slice.sourceCenterId() == ecp_segment.ionId());
      CHECK(slice.flatOffset() == ecp_segment.tileOffset());
      CHECK(slice.size() == ecp_segment.knotCount());

      for (std::size_t local = 0; local < slice.size(); ++local)
      {
        const std::size_t global_index = ecp_segment.globalKnotOffset() + local;
        const std::size_t output_index = ecp_segment.tileOffset() + local;
        CHECK(slice.absolutePosition(local) == storage.tileAbsolutePositions()[output_index]);
        storage.walkerCandidates(static_cast<std::size_t>(ecp_segment.walkerId()))
                                [ecp_segment.walkerKnotOffset() + local] =
            NonLocalData(ecp_segment.electronId(), static_cast<Storage::RealType>(global_index),
                         storage.tileDeltas()[output_index]);
        storage.jobPairPotential(ecp_segment.globalJobId()) += Storage::RealType(1);
      }
    }

    // A writable geometric accessor revokes publication until revalidated.
    (void)storage.mutableTileBareWeights();
    CHECK_FALSE(storage.tileInputReady());
    REQUIRE_THROWS_AS(storage.makeVirtualParticleBatch(), std::logic_error);
    storage.finalizeTileInput();
    ++tile_index;
  }

  CHECK(tile_index == expected_tiles.size());
  CHECK_FALSE(storage.hasNextTile());
  REQUIRE_THROWS_AS(storage.makeVirtualParticleBatch(), std::logic_error);
  CHECK(storage.jobPairPotential(0) == Storage::RealType(3));
  CHECK(storage.jobPairPotential(1) == Storage::RealType(5));
  CHECK(storage.jobPairPotential(2) == Storage::RealType(2));
  CHECK(storage.jobPairPotential(3) == Storage::RealType(3));
  const std::vector<std::size_t> walker_zero_global_order{0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12};
  for (std::size_t knot = 0; knot < walker_zero_global_order.size(); ++knot)
    CHECK(storage.walkerCandidates(0)[knot].Weight ==
          static_cast<Storage::RealType>(walker_zero_global_order[knot]));
  CHECK(storage.walkerCandidates(2)[0].Weight == Storage::RealType(8));
  CHECK(storage.walkerCandidates(2)[1].Weight == Storage::RealType(9));
  CHECK_NOTHROW(storage.validateLogicalOutputExtents());

  storage.walkerCandidates(0).push_back(NonLocalData{});
  REQUIRE_THROWS_AS(storage.validateLogicalOutputExtents(), std::logic_error);
  storage.walkerCandidates(0).pop_back();
  CHECK_NOTHROW(storage.validateLogicalOutputExtents());

  const Storage::Statistics stats = storage.statistics();
  CHECK(stats.logical_jobs == 4);
  CHECK(stats.logical_knots == 13);
  CHECK(stats.candidate_entries == 13);
  CHECK(stats.tiles_packed == 4);
  CHECK(stats.tile_segments_packed == 5);
  CHECK(stats.split_job_continuations == 1);
  CHECK(stats.tail_tiles == 2);
  CHECK(stats.max_tile_occupancy == 4);
  CHECK(stats.max_tile_segments == 2);
}

TEST_CASE("NLPPVirtualBatch rejects invalid order and checked extent overflow", "[hamiltonian][nlpp_batch]")
{
  REQUIRE_THROWS_AS(Storage(0), std::invalid_argument);
  REQUIRE_THROWS_AS(Storage(std::numeric_limits<std::size_t>::max()), std::overflow_error);

  Storage storage(2);
  storage.reset(2, false);
  const std::uint64_t empty_fingerprint = storage.logicalFingerprint();
  const std::vector<std::size_t> empty_offsets{0};

  REQUIRE_THROWS_AS(storage.appendJob({-1, 0, 0, 0, 0, 1}), std::invalid_argument);
  REQUIRE_THROWS_AS(storage.appendJob({0, -1, 0, 0, 0, 1}), std::invalid_argument);
  REQUIRE_THROWS_AS(storage.appendJob({0, 2, 0, 0, 0, 1}), std::invalid_argument);
  REQUIRE_THROWS_AS(storage.appendJob({0, 0, -1, 0, 0, 1}), std::invalid_argument);
  REQUIRE_THROWS_AS(storage.appendJob({0, 0, 0, -1, 0, 1}), std::invalid_argument);
  REQUIRE_THROWS_AS(storage.appendJob({0, 0, 0, 0, 0, 0}), std::invalid_argument);
  REQUIRE_THROWS_AS(storage.appendJob({0, 0, 0, 0, 1, 1}), std::invalid_argument);
  CHECK(storage.jobCount() == 0);
  CHECK(storage.jobOffsets() == empty_offsets);
  CHECK(storage.logicalFingerprint() == empty_fingerprint);

  storage.appendJob({0, 0, 0, 0, 0, 2});
  const std::uint64_t one_job_fingerprint = storage.logicalFingerprint();
  REQUIRE_THROWS_AS(storage.appendJob({0, 0, 1, 1, 0, 1}), std::invalid_argument);
  REQUIRE_THROWS_AS(storage.appendJob({0, 1, 1, 1, 1, 1}), std::invalid_argument);
  REQUIRE_THROWS_AS(storage.appendJob({-1, 1, 1, 1, 0, 1}), std::invalid_argument);
  CHECK(storage.jobCount() == 1);
  CHECK(storage.logicalFingerprint() == one_job_fingerprint);

  Storage overflow(2);
  overflow.reset(1, false);
  overflow.appendJob({0, 0, 0, 0, 0, std::numeric_limits<std::size_t>::max()});
  const std::uint64_t before_overflow = overflow.logicalFingerprint();
  REQUIRE_THROWS_AS(overflow.appendJob({1, 0, 0, 0, 0, 1}), std::overflow_error);
  CHECK(overflow.jobCount() == 1);
  CHECK(overflow.logicalFingerprint() == before_overflow);

  Storage candidate_overflow(2);
  candidate_overflow.reset(1, true);
  candidate_overflow.appendJob({0, 0, 0, 0, 0, std::numeric_limits<std::size_t>::max()});
  REQUIRE_THROWS_AS(candidate_overflow.seal(), std::length_error);
  CHECK_FALSE(candidate_overflow.isSealed());

  storage.seal();
  REQUIRE_THROWS_AS(storage.appendJob({1, 0, 0, 0, 0, 1}), std::logic_error);
  REQUIRE_THROWS_AS(storage.seal(), std::logic_error);
  REQUIRE_THROWS_AS(storage.walkerCandidates(0), std::logic_error);
  REQUIRE_THROWS_AS(storage.jobPairPotential(1), std::out_of_range);
}

TEST_CASE("NLPPVirtualBatch bounds warm tile storage and clones empty", "[hamiltonian][nlpp_batch]")
{
  Storage storage(3);
  storage.reset(2, true);
  storage.appendJob({0, 0, 1, 0, 0, 10});
  storage.seal();

  const Storage::Statistics before = storage.statistics();
  const std::size_t storage_fingerprint = storage.storageFingerprint();
  CHECK(before.logical_jobs == 1);
  CHECK(before.logical_knots == 10);
  CHECK(before.candidate_entries == 10);
  CHECK(before.candidate_output_bytes >= 10 * sizeof(NonLocalData));
  CHECK(before.bounded_tile_bytes > 0);

  while (storage.packNextTile())
  {
    fillTileInput(storage);
    storage.finalizeTileInput();
    (void)storage.makeVirtualParticleBatch();
  }
  CHECK(storage.storageFingerprint() == storage_fingerprint);

  const Storage::Statistics after = storage.statistics();
  CHECK(after.tiles_packed == 4);
  CHECK(after.tile_segments_packed == 4);
  CHECK(after.split_job_continuations == 3);
  CHECK(after.tail_tiles == 1);
  CHECK(after.max_tile_occupancy == 3);
  CHECK(after.max_tile_segments == 1);
  CHECK(after.bounded_tile_bytes == before.bounded_tile_bytes);
  CHECK(after.candidate_output_bytes == before.candidate_output_bytes);

  storage.rewindTiles();
  const Storage::Statistics rewound = storage.statistics();
  CHECK(rewound.tiles_packed == 0);
  CHECK(rewound.max_tile_occupancy == 0);
  CHECK(rewound.bounded_tile_bytes == before.bounded_tile_bytes);
  CHECK(storage.storageFingerprint() == storage_fingerprint);

  std::unique_ptr<Storage> clone = storage.makeEmptyClone();
  REQUIRE(clone);
  CHECK(clone->tileCapacity() == storage.tileCapacity());
  CHECK(clone->stagesCandidates());
  CHECK_FALSE(clone->isSealed());
  CHECK(clone->walkerCount() == 0);
  CHECK(clone->jobCount() == 0);
  CHECK(clone->totalKnotCount() == 0);
  CHECK(clone->statistics().candidate_entries == 0);
  CHECK(clone->statistics().bounded_tile_bytes == storage.statistics().bounded_tile_bytes);
  CHECK(clone->storageFingerprint() != storage.storageFingerprint());

  clone->reset(1, true);
  clone->appendJob({0, 0, 0, 0, 0, 2});
  clone->seal();
  CHECK(clone->packNextTile());
  fillTileInput(*clone);
  clone->finalizeTileInput();
  CHECK(clone->makeVirtualParticleBatch().size() == 2);
  CHECK(clone->walkerCandidates(0).size() == 2);
  CHECK(clone->walkerCandidates(0).data() != storage.walkerCandidates(0).data());
  CHECK(storage.jobCount() == 1);
  CHECK(storage.totalKnotCount() == 10);

  const NonLocalData* retained_candidate_storage = storage.walkerCandidates(0).data();
  const std::size_t retained_candidate_bytes      = storage.statistics().candidate_output_bytes;
  storage.reset(2, true);
  storage.appendJob({0, 0, 1, 0, 0, 5});
  storage.seal();
  CHECK(storage.walkerCandidates(0).data() == retained_candidate_storage);
  CHECK(storage.walkerCandidates(0).size() == 5);
  CHECK(storage.statistics().candidate_output_bytes == retained_candidate_bytes);
  CHECK(storage.statistics().bounded_tile_bytes == before.bounded_tile_bytes);

  Storage no_candidates(3);
  no_candidates.reset(2, false);
  no_candidates.appendJob({0, 0, 1, 0, 0, 10});
  no_candidates.seal();
  const Storage::Statistics no_candidate_stats = no_candidates.statistics();
  CHECK(no_candidate_stats.candidate_entries == 0);
  CHECK(no_candidate_stats.candidate_output_bytes == 0);
  CHECK(no_candidate_stats.bounded_tile_bytes == before.bounded_tile_bytes);
  REQUIRE_THROWS_AS(no_candidates.walkerCandidates(0), std::logic_error);

  Storage empty(2);
  empty.reset(5, false);
  empty.seal();
  CHECK_FALSE(empty.hasNextTile());
  CHECK_FALSE(empty.packNextTile());
  CHECK(empty.statistics().logical_jobs == 0);
  CHECK(empty.statistics().logical_knots == 0);
  CHECK(empty.statistics().tiles_packed == 0);
}

TEST_CASE("NLPPVirtualBatch fingerprints exact logical and tile metadata", "[hamiltonian][nlpp_batch]")
{
  Storage left(3);
  Storage right(3);
  for (Storage* storage : {&left, &right})
  {
    storage->reset(2, true);
    storage->appendJob({0, 0, 1, 2, 0, 4});
    storage->appendJob({0, 1, 2, 3, 0, 1});
    storage->seal();
    REQUIRE(storage->packNextTile());
  }
  CHECK(left.logicalFingerprint() == right.logicalFingerprint());
  CHECK(left.tileFingerprint() == right.tileFingerprint());

  fillTileInput(left);
  left.finalizeTileInput();
  CHECK(left.tileFingerprint() == right.tileFingerprint());

  Storage changed(3);
  changed.reset(2, true);
  changed.appendJob({0, 0, 1, 2, 0, 3});
  changed.appendJob({0, 1, 2, 3, 0, 2});
  changed.seal();
  REQUIRE(changed.packNextTile());
  CHECK(changed.logicalFingerprint() != left.logicalFingerprint());
  CHECK(changed.tileFingerprint() != left.tileFingerprint());

  Storage changed_policy(3);
  changed_policy.reset(2, false);
  changed_policy.appendJob({0, 0, 1, 2, 0, 4});
  changed_policy.appendJob({0, 1, 2, 3, 0, 1});
  CHECK(changed_policy.logicalFingerprint() != left.logicalFingerprint());
}

} // namespace qmcplusplus
