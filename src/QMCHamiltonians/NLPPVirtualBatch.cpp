//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

/** @file NLPPVirtualBatch.cpp
 * @brief Checked storage and deterministic tiling for nonlocal ECP virtual work.
 */

#include "NLPPVirtualBatch.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace qmcplusplus
{
namespace
{
constexpr std::uint64_t fnv_offset_basis = UINT64_C(14695981039346656037);
constexpr std::uint64_t fnv_prime        = UINT64_C(1099511628211);

/** Add two extents without allowing size_t wraparound. */
std::size_t checkedAdd(std::size_t left, std::size_t right, const char* description)
{
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(std::string("NLPP virtual-batch size overflow in ") + description + ".");
  return left + right;
}

/** Multiply two extents without allowing size_t wraparound. */
std::size_t checkedMultiply(std::size_t left, std::size_t right, const char* description)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(std::string("NLPP virtual-batch size overflow in ") + description + ".");
  return left * right;
}

/** Add one byte to an FNV-1a exact-content fingerprint. */
void mixByte(std::uint64_t& hash, unsigned char value) noexcept
{
  hash ^= value;
  hash *= fnv_prime;
}

/** Add a fixed-width integer in canonical little-endian order. */
void mixUint64(std::uint64_t& hash, std::uint64_t value) noexcept
{
  for (int byte = 0; byte < 8; ++byte)
  {
    mixByte(hash, static_cast<unsigned char>(value & UINT64_C(0xff)));
    value >>= 8;
  }
}

/** Add one signed QMCPACK index without depending on its host width. */
void mixIndex(std::uint64_t& hash, QMCTraits::IndexType value) noexcept
{
  static_assert(std::is_signed_v<QMCTraits::IndexType>);
  static_assert(sizeof(QMCTraits::IndexType) <= sizeof(std::uint64_t));
  mixUint64(hash, static_cast<std::uint64_t>(value));
}

/** Add one vector's retained payload bytes to a checked total. */
template<typename T>
void addCapacityBytes(std::size_t& total, const std::vector<T>& storage, const char* description)
{
  const std::size_t bytes = checkedMultiply(storage.capacity(), sizeof(T), description);
  total                   = checkedAdd(total, bytes, description);
}

/** Mix allocation identity and capacity into a process-local diagnostic. */
template<typename T>
void mixStorage(std::size_t& hash, const std::vector<T>& storage) noexcept
{
  const auto address = reinterpret_cast<std::uintptr_t>(storage.data());
  hash ^= static_cast<std::size_t>(address) + std::size_t(0x9e3779b9) + (hash << 6) + (hash >> 2);
  hash ^= storage.capacity() + std::size_t(0x9e3779b9) + (hash << 6) + (hash >> 2);
}
} // namespace

NLPPVirtualBatchStorage::NLPPVirtualBatchStorage(std::size_t tile_capacity) : tile_capacity_(tile_capacity)
{
  if (tile_capacity_ == 0)
    throw std::invalid_argument("NLPP virtual-batch tile capacity must be nonzero.");
  reserveTileStorage();
  reset(0, false);
}

std::unique_ptr<NLPPVirtualBatchStorage> NLPPVirtualBatchStorage::makeEmptyClone() const
{
  auto clone = std::make_unique<NLPPVirtualBatchStorage>(tile_capacity_);
  clone->reset(0, stage_candidates_);
  return clone;
}

void NLPPVirtualBatchStorage::reset(std::size_t walker_count, bool stage_candidates)
{
  if (walker_count > walker_candidate_counts_.max_size() || walker_count > walker_candidates_.max_size())
    throw std::length_error("NLPP virtual-batch walker metadata exceeds vector limits.");

  walker_count_     = walker_count;
  total_knot_count_ = 0;
  stage_candidates_ = stage_candidates;
  sealed_           = false;

  jobs_.clear();
  job_offsets_.clear();
  job_offsets_.push_back(0);
  job_walker_knot_offsets_.clear();
  walker_candidate_counts_.assign(walker_count_, 0);
  job_pair_potentials_.clear();
  walker_candidates_.resize(walker_count_);
  for (std::vector<NonLocalData>& candidates : walker_candidates_)
    candidates.clear();

  rewindTiles();
}

std::size_t NLPPVirtualBatchStorage::appendJob(const Job& job)
{
  requireBuilding("appendJob");
  validateAppend(job);

  const std::size_t next_total = checkedAdd(total_knot_count_, job.knot_count, "total knot count");
  const std::size_t walker_id   = static_cast<std::size_t>(job.walker_id);
  const std::size_t walker_offset = walker_candidate_counts_[walker_id];
  const std::size_t next_walker_total =
      checkedAdd(walker_offset, job.knot_count, "per-walker candidate count");
  if (jobs_.size() == jobs_.max_size() || job_offsets_.size() == job_offsets_.max_size() ||
      job_walker_knot_offsets_.size() == job_walker_knot_offsets_.max_size())
    throw std::length_error("NLPP virtual-batch job metadata exceeds vector limits.");

  const std::size_t global_job_id = jobs_.size();
  jobs_.push_back(job);
  try
  {
    job_offsets_.push_back(next_total);
    try
    {
      job_walker_knot_offsets_.push_back(walker_offset);
    }
    catch (...)
    {
      job_offsets_.pop_back();
      throw;
    }
  }
  catch (...)
  {
    jobs_.pop_back();
    throw;
  }
  total_knot_count_                    = next_total;
  walker_candidate_counts_[walker_id] = next_walker_total;
  return global_job_id;
}

void NLPPVirtualBatchStorage::seal()
{
  requireBuilding("seal");
  if (jobs_.size() > job_pair_potentials_.max_size())
    throw std::length_error("NLPP virtual-batch pair-potential storage exceeds vector limits.");
  if (walker_candidates_.size() != walker_count_ || walker_candidate_counts_.size() != walker_count_)
    throw std::logic_error("NLPP virtual-batch walker candidate metadata has an inconsistent extent.");
  if (stage_candidates_)
    for (std::size_t walker = 0; walker < walker_count_; ++walker)
      if (walker_candidate_counts_[walker] > walker_candidates_[walker].max_size())
        throw std::length_error("NLPP virtual-batch candidate output exceeds vector limits.");

  job_pair_potentials_.assign(jobs_.size(), RealType{});
  for (std::size_t walker = 0; walker < walker_count_; ++walker)
    if (stage_candidates_)
      walker_candidates_[walker].resize(walker_candidate_counts_[walker]);
    else
      walker_candidates_[walker].clear();

  sealed_ = true;
  rewindTiles();
}

void NLPPVirtualBatchStorage::validateLogicalOutputExtents() const
{
  requireSealed("validateLogicalOutputExtents");
  if (job_pair_potentials_.size() != jobs_.size() || walker_candidates_.size() != walker_count_ ||
      walker_candidate_counts_.size() != walker_count_)
    throw std::logic_error("NLPP virtual-batch logical output metadata has an inconsistent extent.");
  for (std::size_t walker = 0; walker < walker_count_; ++walker)
  {
    const std::size_t expected = stage_candidates_ ? walker_candidate_counts_[walker] : 0;
    if (walker_candidates_[walker].size() != expected)
      throw std::logic_error("NLPP virtual-batch per-walker candidate output extent changed unexpectedly.");
  }
}

void NLPPVirtualBatchStorage::rewindTiles() noexcept
{
  next_job_index_ = 0;
  next_job_knot_  = 0;
  tile_active_      = false;
  tile_input_ready_ = false;

  tile_segments_.clear();
  tile_segment_offsets_.clear();
  virtual_segments_.clear();
  tile_absolute_positions_.clear();
  tile_deltas_.clear();
  tile_bare_weights_.clear();
  tile_ratios_.clear();
  tile_fermionic_ratios_.clear();
  tile_nonfermionic_ratios_.clear();
  tile_transformed_weights_.clear();

  tiles_packed_            = 0;
  tile_segments_packed_    = 0;
  split_job_continuations_ = 0;
  tail_tiles_              = 0;
  max_tile_occupancy_      = 0;
  max_tile_segments_       = 0;
}

bool NLPPVirtualBatchStorage::packNextTile()
{
  requireSealed("packNextTile");
  tile_active_      = false;
  tile_input_ready_ = false;
  tile_segments_.clear();
  tile_segment_offsets_.clear();
  virtual_segments_.clear();

  if (!hasNextTile())
  {
    resizeTileStorage(0);
    return false;
  }

  tile_segment_offsets_.push_back(0);
  const IndexType tile_group = jobs_[next_job_index_].group_id;
  std::size_t occupancy       = 0;
  std::size_t continuations   = 0;

  // Group boundaries remain visible because TrialWaveFunction::prepareGroup
  // is an orchestration boundary.  A large job may still continue next tile.
  while (occupancy < tile_capacity_ && next_job_index_ < jobs_.size() &&
         jobs_[next_job_index_].group_id == tile_group)
  {
    const Job& job             = jobs_[next_job_index_];
    const std::size_t remaining = job.knot_count - next_job_knot_;
    const std::size_t count     = std::min(remaining, tile_capacity_ - occupancy);
    const std::size_t global_knot_offset =
        checkedAdd(job_offsets_[next_job_index_], next_job_knot_, "global knot offset");
    const std::size_t walker_knot_offset =
        checkedAdd(job_walker_knot_offsets_[next_job_index_], next_job_knot_, "walker knot offset");

    tile_segments_.push_back(TileSegment(next_job_index_, job, next_job_knot_, count, occupancy,
                                         global_knot_offset, walker_knot_offset));
    virtual_segments_.emplace_back(job.walker_id, job.electron_id, true, job.ion_id);
    occupancy = checkedAdd(occupancy, count, "tile occupancy");
    tile_segment_offsets_.push_back(occupancy);
    if (next_job_knot_ != 0)
      continuations = checkedAdd(continuations, 1, "split-job continuation count");

    next_job_knot_ = checkedAdd(next_job_knot_, count, "job knot cursor");
    if (next_job_knot_ == job.knot_count)
    {
      next_job_knot_ = 0;
      ++next_job_index_;
    }
  }

  resizeTileStorage(occupancy);
  tile_active_ = true;

  tiles_packed_ = checkedAdd(tiles_packed_, 1, "packed tile count");
  tile_segments_packed_ =
      checkedAdd(tile_segments_packed_, tile_segments_.size(), "packed tile-segment count");
  split_job_continuations_ =
      checkedAdd(split_job_continuations_, continuations, "split-job continuation count");
  if (occupancy < tile_capacity_)
    tail_tiles_ = checkedAdd(tail_tiles_, 1, "tail tile count");
  max_tile_occupancy_ = std::max(max_tile_occupancy_, occupancy);
  max_tile_segments_  = std::max(max_tile_segments_, tile_segments_.size());
  return true;
}

void NLPPVirtualBatchStorage::finalizeTileInput()
{
  requireActiveTile("finalizeTileInput");
  for (std::size_t knot = 0; knot < tile_absolute_positions_.size(); ++knot)
  {
    for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
    {
      if (!virtual_particle_batch_detail::isFiniteReal(tile_absolute_positions_[knot][dimension]))
        throw std::invalid_argument("NLPP virtual-batch absolute positions must be finite before publication.");
      if (!virtual_particle_batch_detail::isFiniteReal(tile_deltas_[knot][dimension]))
        throw std::invalid_argument("NLPP virtual-batch displacement vectors must be finite before publication.");
    }
    if (!virtual_particle_batch_detail::isFiniteReal(tile_bare_weights_[knot]))
      throw std::invalid_argument("NLPP virtual-batch bare weights must be finite before publication.");
  }
  tile_input_ready_ = true;
}

VirtualParticleBatch NLPPVirtualBatchStorage::makeVirtualParticleBatch() const
{
  requireActiveTile("makeVirtualParticleBatch");
  if (!tile_input_ready_)
    throw std::logic_error("NLPP virtual-batch input must be finalized before descriptor publication.");
  return VirtualParticleBatch(walker_count_, tile_segment_offsets_, virtual_segments_, tile_absolute_positions_);
}

bool NLPPVirtualBatchStorage::hasNextTile() const noexcept
{
  return sealed_ && next_job_index_ < jobs_.size();
}

NLPPVirtualBatchStorage::RealType& NLPPVirtualBatchStorage::jobPairPotential(std::size_t global_job_id)
{
  requireSealed("jobPairPotential");
  return job_pair_potentials_.at(global_job_id);
}

const NLPPVirtualBatchStorage::RealType& NLPPVirtualBatchStorage::jobPairPotential(
    std::size_t global_job_id) const
{
  requireSealed("jobPairPotential");
  return job_pair_potentials_.at(global_job_id);
}

std::vector<NonLocalData>& NLPPVirtualBatchStorage::walkerCandidates(std::size_t walker_id)
{
  requireSealed("walkerCandidates");
  if (!stage_candidates_)
    throw std::logic_error("NLPP virtual-batch candidate output was not requested.");
  return walker_candidates_.at(walker_id);
}

const std::vector<NonLocalData>& NLPPVirtualBatchStorage::walkerCandidates(std::size_t walker_id) const
{
  requireSealed("walkerCandidates");
  if (!stage_candidates_)
    throw std::logic_error("NLPP virtual-batch candidate output was not requested.");
  return walker_candidates_.at(walker_id);
}

std::vector<NLPPVirtualBatchStorage::PosType>& NLPPVirtualBatchStorage::mutableTileAbsolutePositions()
{
  requireActiveTile("mutableTileAbsolutePositions");
  invalidateTileInput();
  return tile_absolute_positions_;
}

std::vector<NLPPVirtualBatchStorage::PosType>& NLPPVirtualBatchStorage::mutableTileDeltas()
{
  requireActiveTile("mutableTileDeltas");
  invalidateTileInput();
  return tile_deltas_;
}

std::vector<NLPPVirtualBatchStorage::RealType>& NLPPVirtualBatchStorage::mutableTileBareWeights()
{
  requireActiveTile("mutableTileBareWeights");
  invalidateTileInput();
  return tile_bare_weights_;
}

std::vector<NLPPVirtualBatchStorage::ValueType>& NLPPVirtualBatchStorage::mutableTileRatios()
{
  requireActiveTile("mutableTileRatios");
  return tile_ratios_;
}

std::vector<NLPPVirtualBatchStorage::ValueType>& NLPPVirtualBatchStorage::mutableTileFermionicRatios()
{
  requireActiveTile("mutableTileFermionicRatios");
  return tile_fermionic_ratios_;
}

std::vector<NLPPVirtualBatchStorage::ValueType>& NLPPVirtualBatchStorage::mutableTileNonfermionicRatios()
{
  requireActiveTile("mutableTileNonfermionicRatios");
  return tile_nonfermionic_ratios_;
}

std::vector<NLPPVirtualBatchStorage::RealType>& NLPPVirtualBatchStorage::mutableTileTransformedWeights()
{
  requireActiveTile("mutableTileTransformedWeights");
  return tile_transformed_weights_;
}

std::uint64_t NLPPVirtualBatchStorage::logicalFingerprint() const noexcept
{
  constexpr std::uint64_t logical_tag = UINT64_C(0x4e4c50504c4f4701);
  std::uint64_t hash                  = fnv_offset_basis;
  mixUint64(hash, logical_tag);
  mixUint64(hash, static_cast<std::uint64_t>(walker_count_));
  mixUint64(hash, stage_candidates_ ? UINT64_C(1) : UINT64_C(0));
  mixUint64(hash, static_cast<std::uint64_t>(jobs_.size()));
  mixUint64(hash, static_cast<std::uint64_t>(total_knot_count_));
  for (const Job& job : jobs_)
  {
    mixIndex(hash, job.group_id);
    mixIndex(hash, job.walker_id);
    mixIndex(hash, job.ion_id);
    mixIndex(hash, job.electron_id);
    mixUint64(hash, static_cast<std::uint64_t>(job.walker_job_ordinal));
    mixUint64(hash, static_cast<std::uint64_t>(job.knot_count));
  }
  for (const std::size_t offset : job_offsets_)
    mixUint64(hash, static_cast<std::uint64_t>(offset));
  for (const std::size_t offset : job_walker_knot_offsets_)
    mixUint64(hash, static_cast<std::uint64_t>(offset));
  for (const std::size_t count : walker_candidate_counts_)
    mixUint64(hash, static_cast<std::uint64_t>(count));
  return hash;
}

std::uint64_t NLPPVirtualBatchStorage::tileFingerprint() const noexcept
{
  constexpr std::uint64_t tile_tag = UINT64_C(0x4e4c505054494c01);
  std::uint64_t hash               = fnv_offset_basis;
  mixUint64(hash, tile_tag);
  mixUint64(hash, static_cast<std::uint64_t>(tile_capacity_));
  mixUint64(hash, tile_active_ ? UINT64_C(1) : UINT64_C(0));
  mixUint64(hash, static_cast<std::uint64_t>(tile_segments_.size()));
  mixUint64(hash, static_cast<std::uint64_t>(tile_absolute_positions_.size()));
  for (const TileSegment& segment : tile_segments_)
  {
    mixUint64(hash, static_cast<std::uint64_t>(segment.globalJobId()));
    mixIndex(hash, segment.groupId());
    mixIndex(hash, segment.walkerId());
    mixIndex(hash, segment.ionId());
    mixIndex(hash, segment.electronId());
    mixUint64(hash, static_cast<std::uint64_t>(segment.walkerJobOrdinal()));
    mixUint64(hash, static_cast<std::uint64_t>(segment.firstKnot()));
    mixUint64(hash, static_cast<std::uint64_t>(segment.knotCount()));
    mixUint64(hash, static_cast<std::uint64_t>(segment.jobKnotCount()));
    mixUint64(hash, static_cast<std::uint64_t>(segment.tileOffset()));
    mixUint64(hash, static_cast<std::uint64_t>(segment.globalKnotOffset()));
    mixUint64(hash, static_cast<std::uint64_t>(segment.walkerKnotOffset()));
  }
  for (const std::size_t offset : tile_segment_offsets_)
    mixUint64(hash, static_cast<std::uint64_t>(offset));
  return hash;
}

std::size_t NLPPVirtualBatchStorage::storageFingerprint() const noexcept
{
  std::size_t hash = tile_capacity_;
  mixStorage(hash, jobs_);
  mixStorage(hash, job_offsets_);
  mixStorage(hash, job_walker_knot_offsets_);
  mixStorage(hash, walker_candidate_counts_);
  mixStorage(hash, job_pair_potentials_);
  mixStorage(hash, walker_candidates_);
  for (const std::vector<NonLocalData>& candidates : walker_candidates_)
    mixStorage(hash, candidates);
  mixStorage(hash, tile_segments_);
  mixStorage(hash, tile_segment_offsets_);
  mixStorage(hash, virtual_segments_);
  mixStorage(hash, tile_absolute_positions_);
  mixStorage(hash, tile_deltas_);
  mixStorage(hash, tile_bare_weights_);
  mixStorage(hash, tile_ratios_);
  mixStorage(hash, tile_fermionic_ratios_);
  mixStorage(hash, tile_nonfermionic_ratios_);
  mixStorage(hash, tile_transformed_weights_);
  return hash;
}

NLPPVirtualBatchStorage::Statistics NLPPVirtualBatchStorage::statistics() const
{
  Statistics result;
  result.logical_jobs            = jobs_.size();
  result.logical_knots           = total_knot_count_;
  for (const std::vector<NonLocalData>& candidates : walker_candidates_)
    result.candidate_entries = checkedAdd(result.candidate_entries, candidates.size(), "candidate entry count");
  result.tiles_packed            = tiles_packed_;
  result.tile_segments_packed    = tile_segments_packed_;
  result.split_job_continuations = split_job_continuations_;
  result.tail_tiles              = tail_tiles_;
  result.max_tile_occupancy      = max_tile_occupancy_;
  result.max_tile_segments       = max_tile_segments_;

  addCapacityBytes(result.logical_job_bytes, jobs_, "logical job bytes");
  addCapacityBytes(result.logical_job_bytes, job_offsets_, "logical job-offset bytes");
  addCapacityBytes(result.logical_job_bytes, job_walker_knot_offsets_, "walker-local job-offset bytes");
  addCapacityBytes(result.logical_job_bytes, walker_candidate_counts_, "walker candidate-count bytes");
  addCapacityBytes(result.logical_job_bytes, job_pair_potentials_, "logical pair-potential bytes");
  addCapacityBytes(result.logical_job_bytes, walker_candidates_, "walker candidate-vector bytes");
  for (const std::vector<NonLocalData>& candidates : walker_candidates_)
    addCapacityBytes(result.candidate_output_bytes, candidates, "candidate output bytes");

  addCapacityBytes(result.bounded_tile_bytes, tile_segments_, "tile-segment bytes");
  addCapacityBytes(result.bounded_tile_bytes, tile_segment_offsets_, "tile segment-offset bytes");
  addCapacityBytes(result.bounded_tile_bytes, virtual_segments_, "virtual-segment bytes");
  addCapacityBytes(result.bounded_tile_bytes, tile_absolute_positions_, "tile absolute-position bytes");
  addCapacityBytes(result.bounded_tile_bytes, tile_deltas_, "tile displacement bytes");
  addCapacityBytes(result.bounded_tile_bytes, tile_bare_weights_, "tile bare-weight bytes");
  addCapacityBytes(result.bounded_tile_bytes, tile_ratios_, "tile ratio bytes");
  addCapacityBytes(result.bounded_tile_bytes, tile_fermionic_ratios_, "tile fermionic-ratio bytes");
  addCapacityBytes(result.bounded_tile_bytes, tile_nonfermionic_ratios_, "tile nonfermionic-ratio bytes");
  addCapacityBytes(result.bounded_tile_bytes, tile_transformed_weights_, "tile transformed-weight bytes");
  return result;
}

void NLPPVirtualBatchStorage::requireBuilding(const char* operation) const
{
  if (sealed_)
    throw std::logic_error(std::string("NLPP virtual-batch ") + operation + " requires an unsealed workload.");
}

void NLPPVirtualBatchStorage::requireSealed(const char* operation) const
{
  if (!sealed_)
    throw std::logic_error(std::string("NLPP virtual-batch ") + operation + " requires a sealed workload.");
}

void NLPPVirtualBatchStorage::requireActiveTile(const char* operation) const
{
  requireSealed(operation);
  if (!tile_active_)
    throw std::logic_error(std::string("NLPP virtual-batch ") + operation + " requires an active tile.");
}

void NLPPVirtualBatchStorage::validateAppend(const Job& job) const
{
  if (job.group_id < 0 || job.walker_id < 0 || job.ion_id < 0 || job.electron_id < 0)
    throw std::invalid_argument("NLPP virtual-batch job indices must be nonnegative.");
  if (static_cast<std::size_t>(job.walker_id) >= walker_count_)
    throw std::invalid_argument("NLPP virtual-batch job walker index is out of range.");
  if (job.knot_count == 0)
    throw std::invalid_argument("NLPP virtual-batch jobs must contain at least one quadrature knot.");

  std::size_t expected_ordinal = 0;
  if (!jobs_.empty())
  {
    const Job& previous = jobs_.back();
    if (job.group_id < previous.group_id ||
        (job.group_id == previous.group_id && job.walker_id < previous.walker_id))
      throw std::invalid_argument("NLPP virtual-batch jobs are not in canonical group/walker order.");
    if (job.group_id == previous.group_id && job.walker_id == previous.walker_id)
      expected_ordinal = checkedAdd(previous.walker_job_ordinal, 1, "walker job ordinal");
  }
  if (job.walker_job_ordinal != expected_ordinal)
    throw std::invalid_argument("NLPP virtual-batch walker job ordinals must be contiguous and start at zero.");
}

void NLPPVirtualBatchStorage::reserveTileStorage()
{
  const std::size_t offset_capacity = checkedAdd(tile_capacity_, 1, "tile segment-offset capacity");

  // Check the complete bounded payload before the first allocation so a
  // nonsensical capacity fails deterministically rather than partway through.
  std::size_t checked_bytes = 0;
  checked_bytes = checkedAdd(checked_bytes,
                             checkedMultiply(tile_capacity_, sizeof(TileSegment), "tile-segment capacity"),
                             "bounded tile capacity");
  checked_bytes = checkedAdd(checked_bytes,
                             checkedMultiply(offset_capacity, sizeof(std::size_t), "tile segment-offset capacity"),
                             "bounded tile capacity");
  checked_bytes = checkedAdd(
      checked_bytes,
      checkedMultiply(tile_capacity_, sizeof(VirtualParticleBatch::Segment), "virtual-segment capacity"),
      "bounded tile capacity");
  checked_bytes = checkedAdd(
      checked_bytes, checkedMultiply(tile_capacity_, 2 * sizeof(PosType), "tile position capacity"),
      "bounded tile capacity");
  checked_bytes = checkedAdd(
      checked_bytes, checkedMultiply(tile_capacity_, 2 * sizeof(RealType), "tile real capacity"),
      "bounded tile capacity");
  checked_bytes = checkedAdd(
      checked_bytes, checkedMultiply(tile_capacity_, 3 * sizeof(ValueType), "tile ratio capacity"),
      "bounded tile capacity");
  (void)checked_bytes;

  tile_segments_.reserve(tile_capacity_);
  tile_segment_offsets_.reserve(offset_capacity);
  virtual_segments_.reserve(tile_capacity_);
  tile_absolute_positions_.reserve(tile_capacity_);
  tile_deltas_.reserve(tile_capacity_);
  tile_bare_weights_.reserve(tile_capacity_);
  tile_ratios_.reserve(tile_capacity_);
  tile_fermionic_ratios_.reserve(tile_capacity_);
  tile_nonfermionic_ratios_.reserve(tile_capacity_);
  tile_transformed_weights_.reserve(tile_capacity_);
}

void NLPPVirtualBatchStorage::resizeTileStorage(std::size_t occupancy)
{
  if (occupancy > tile_capacity_)
    throw std::logic_error("NLPP virtual-batch tile occupancy exceeds its fixed capacity.");

  tile_absolute_positions_.resize(occupancy);
  tile_deltas_.resize(occupancy);
  tile_bare_weights_.resize(occupancy);
  tile_ratios_.assign(occupancy, ValueType{});
  tile_fermionic_ratios_.assign(occupancy, ValueType{});
  tile_nonfermionic_ratios_.assign(occupancy, ValueType{});
  tile_transformed_weights_.assign(occupancy, RealType{});

  // Non-finite sentinels prevent a shorter or forgotten caller fill from
  // publishing coordinates left over from the preceding tile.
  const RealType sentinel = std::numeric_limits<RealType>::quiet_NaN();
  for (std::size_t knot = 0; knot < occupancy; ++knot)
  {
    for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
    {
      tile_absolute_positions_[knot][dimension] = sentinel;
      tile_deltas_[knot][dimension]             = sentinel;
    }
    tile_bare_weights_[knot] = sentinel;
  }
}

} // namespace qmcplusplus
