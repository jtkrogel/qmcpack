//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file WFTrainSamplerCheckpoint.cpp
 * @brief Exact rank-local walker and RNG checkpoint contributor for WFTrain.
 */

#include "QMCDrivers/WFTrain/WFTrainSamplerCheckpoint.h"

#include "QMCDrivers/WFTrain/TrainingNumerics.h"
#include "io/hdf/hdf_archive.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qmcplusplus::wftrain
{
namespace
{

constexpr char CONTRIBUTOR_ID[] = "qmcpack.wftrain.rank_local_sampler";

/// Extend one stable FNV-1a fingerprint with a raw byte range.
void extendFingerprint(std::uint64_t& hash, const void* data, std::size_t bytes) noexcept
{
  const auto* input = static_cast<const unsigned char*>(data);
  for (std::size_t index = 0; index < bytes; ++index)
  {
    hash ^= input[index];
    hash *= UINT64_C(1099511628211);
  }
}

/// Extend a fingerprint with a vector's element count and exact bit pattern.
template<class T>
void extendVectorFingerprint(std::uint64_t& hash, const std::vector<T>& values) noexcept
{
  const std::uint64_t size = values.size();
  extendFingerprint(hash, &size, sizeof(size));
  if (!values.empty())
    extendFingerprint(hash, values.data(), values.size() * sizeof(T));
}

/// Render one stable fingerprint for persisted sampler fields.
std::string samplerFingerprint(const std::vector<double>& positions,
                               const std::vector<double>& spins,
                               const std::vector<double>& walker_reals,
                               const std::vector<std::int64_t>& walker_ids,
                               const std::vector<int>& walker_integers,
                               const std::vector<std::uint64_t>& rng_sizes,
                               const std::vector<std::uint64_t>& rng_state,
                               std::int64_t global_step)
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  extendVectorFingerprint(hash, positions);
  extendVectorFingerprint(hash, spins);
  extendVectorFingerprint(hash, walker_reals);
  extendVectorFingerprint(hash, walker_ids);
  extendVectorFingerprint(hash, walker_integers);
  extendVectorFingerprint(hash, rng_sizes);
  extendVectorFingerprint(hash, rng_state);
  extendFingerprint(hash, &global_step, sizeof(global_step));
  std::ostringstream text;
  text << std::hex << std::setw(16) << std::setfill('0') << hash;
  return text.str();
}

/// Convert a runtime size to portable storage without accepting truncation.
std::uint64_t persistedSize(std::size_t value)
{
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t))
    if (value > std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("WFTrain sampler topology exceeds uint64 storage");
  return static_cast<std::uint64_t>(value);
}

/// Convert a persisted size without accepting truncation.
std::size_t runtimeSize(std::uint64_t value, const char* field)
{
  if (value > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error(std::string("WFTrain sampler ") + field +
                             " exceeds this build's size range");
  return static_cast<std::size_t>(value);
}

/// Multiply trusted runtime extents while retaining a bounded shape check.
std::size_t checkedProduct(std::size_t left, std::size_t right, const char* field)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::runtime_error(std::string("WFTrain sampler ") + field + " extent overflow");
  return left * right;
}

} // namespace

/// Owning serialized sampler state used by save and prepared restore.
struct WFTrainSamplerCheckpoint::Snapshot
{
  std::vector<double> positions;
  std::vector<double> spins;
  std::vector<double> walker_reals;
  std::vector<std::int64_t> walker_ids;
  std::vector<int> walker_integers;
  std::vector<std::uint64_t> rng_sizes;
  std::vector<std::uint64_t> rng_state;
  std::int64_t global_step = 0;

  std::string fingerprint() const
  {
    return samplerFingerprint(positions, spins, walker_reals, walker_ids,
                              walker_integers, rng_sizes, rng_state, global_step);
  }
};

namespace
{

/** Commit a fully validated sampler snapshot using only fixed-size assignments. */
class PreparedSamplerRestore final : public SamplerCheckpointContributor::PreparedRestore
{
public:
  PreparedSamplerRestore(MCPopulation& population,
                         const RefVector<WFTrainSamplerCheckpoint::RNG>& random_generators,
                         int& global_step,
                         std::vector<double> positions,
                         std::vector<double> spins,
                         std::vector<double> walker_reals,
                         std::vector<std::int64_t> walker_ids,
                         std::vector<int> walker_integers,
                         std::vector<std::unique_ptr<WFTrainSamplerCheckpoint::RNG>> rng_state,
                         int restored_global_step)
      : population_(population),
        random_generators_(random_generators),
        global_step_(global_step),
        positions_(std::move(positions)),
        spins_(std::move(spins)),
        walker_reals_(std::move(walker_reals)),
        walker_ids_(std::move(walker_ids)),
        walker_integers_(std::move(walker_integers)),
        rng_state_(std::move(rng_state)),
        restored_global_step_(restored_global_step)
  {}

  void commit() noexcept override
  {
    std::size_t coordinate = 0;
    std::size_t walker_index = 0;
    for (auto& walker_owner : population_.get_walkers())
    {
      auto& walker = *walker_owner;
      for (std::size_t particle = 0; particle < walker.R.size(); ++particle)
      {
        for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
          walker.R[particle][dimension] = positions_[coordinate++];
        walker.spins[particle] = spins_[walker_index * walker.R.size() + particle];
      }
      walker.Weight       = walker_reals_[2 * walker_index];
      walker.Multiplicity = walker_reals_[2 * walker_index + 1];
      walker.setWalkerID(static_cast<long>(walker_ids_[2 * walker_index]));
      walker.setParentID(static_cast<long>(walker_ids_[2 * walker_index + 1]));
      walker.Generation = walker_integers_[2 * walker_index];
      walker.Age        = walker_integers_[2 * walker_index + 1];
      walker.wasTouched = true;
      ++walker_index;
    }
    for (std::size_t rng = 0; rng < rng_state_.size(); ++rng)
      random_generators_[rng].get().swapState(*rng_state_[rng]);
    global_step_ = restored_global_step_;
  }

private:
  MCPopulation& population_;
  const RefVector<WFTrainSamplerCheckpoint::RNG>& random_generators_;
  int& global_step_;
  std::vector<double> positions_;
  std::vector<double> spins_;
  std::vector<double> walker_reals_;
  std::vector<std::int64_t> walker_ids_;
  std::vector<int> walker_integers_;
  std::vector<std::unique_ptr<WFTrainSamplerCheckpoint::RNG>> rng_state_;
  int restored_global_step_;
};

} // namespace

WFTrainSamplerCheckpoint::WFTrainSamplerCheckpoint(
    MCPopulation& population,
    const RefVector<RNG>& random_generators,
    int& global_step,
    std::size_t participant_count,
    std::size_t rank,
    std::size_t crowd_count)
    : population_(population),
      random_generators_(random_generators),
      global_step_(global_step)
{
  topology_.participant_count = participant_count;
  topology_.rank              = rank;
  topology_.crowd_count       = crowd_count;
  topology_.walker_count      = population_.get_walkers().size();
  topology_.particle_count    = population_.get_golden_electrons().getTotalNum();
  topology_.rng_state_sizes.reserve(random_generators_.size());
  for (const RNG& rng : random_generators_)
    topology_.rng_state_sizes.push_back(rng.state_size());
  if (topology_.participant_count == 0 || topology_.rank >= topology_.participant_count)
    throw std::invalid_argument("WFTrain sampler checkpoint has invalid rank topology");
  if (topology_.crowd_count > topology_.rng_state_sizes.size())
    throw std::invalid_argument(
        "WFTrain sampler checkpoint requires at least one RNG for each live crowd");
}

WFTrainSamplerCheckpoint::Snapshot WFTrainSamplerCheckpoint::capture() const
{
  Snapshot snapshot;
  const std::size_t walker_particles = checkedProduct(
      topology_.walker_count, topology_.particle_count, "walker-particle");
  snapshot.positions.reserve(checkedProduct(walker_particles, OHMMS_DIM, "coordinate"));
  snapshot.spins.reserve(walker_particles);
  const std::size_t pair_fields = checkedProduct(topology_.walker_count, 2, "walker-pair");
  snapshot.walker_reals.reserve(pair_fields);
  snapshot.walker_ids.reserve(pair_fields);
  snapshot.walker_integers.reserve(pair_fields);

  for (const auto& walker_owner : population_.get_walkers())
  {
    const auto& walker = *walker_owner;
    if (static_cast<std::size_t>(walker.size()) != topology_.particle_count)
      throw std::logic_error("WFTrain sampler walker particle count changed after construction");
    for (std::size_t particle = 0; particle < walker.R.size(); ++particle)
    {
      for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
        snapshot.positions.push_back(walker.R[particle][dimension]);
      snapshot.spins.push_back(walker.spins[particle]);
    }
    snapshot.walker_reals.push_back(walker.Weight);
    snapshot.walker_reals.push_back(walker.Multiplicity);
    snapshot.walker_ids.push_back(walker.getWalkerID());
    snapshot.walker_ids.push_back(walker.getParentID());
    snapshot.walker_integers.push_back(walker.Generation);
    snapshot.walker_integers.push_back(walker.Age);
  }
  if (population_.get_walkers().size() != topology_.walker_count)
    throw std::logic_error("WFTrain sampler walker count changed after construction");
  if (!std::all_of(snapshot.positions.begin(), snapshot.positions.end(),
                   [](double value) { return isFiniteTrainingReal(value); }) ||
      !std::all_of(snapshot.spins.begin(), snapshot.spins.end(),
                   [](double value) { return isFiniteTrainingReal(value); }) ||
      !std::all_of(snapshot.walker_reals.begin(), snapshot.walker_reals.end(),
                   [](double value) { return isFiniteTrainingReal(value); }))
    throw std::runtime_error("WFTrain sampler checkpoint contains non-finite walker state");

  for (std::size_t index = 0; index < random_generators_.size(); ++index)
  {
    std::vector<RNG::uint_type> state;
    random_generators_[index].get().save(state);
    if (state.size() != topology_.rng_state_sizes[index])
      throw std::logic_error("WFTrain sampler RNG state size changed after construction");
    snapshot.rng_sizes.push_back(persistedSize(state.size()));
    for (RNG::uint_type value : state)
      snapshot.rng_state.push_back(static_cast<std::uint64_t>(value));
  }
  if (global_step_ < 0)
    throw std::logic_error("WFTrain sampler global step must not be negative");
  snapshot.global_step = global_step_;
  return snapshot;
}

std::string WFTrainSamplerCheckpoint::configurationFingerprint() const
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const std::uint64_t fields[]{persistedSize(topology_.participant_count),
                               persistedSize(topology_.rank),
                               persistedSize(topology_.crowd_count),
                               persistedSize(topology_.walker_count),
                               persistedSize(topology_.particle_count)};
  extendFingerprint(hash, fields, sizeof(fields));
  for (std::size_t size : topology_.rng_state_sizes)
  {
    const std::uint64_t persisted = persistedSize(size);
    extendFingerprint(hash, &persisted, sizeof(persisted));
  }
  std::ostringstream text;
  text << std::hex << std::setw(16) << std::setfill('0') << hash;
  return text.str();
}

CheckpointContributorMetadata WFTrainSamplerCheckpoint::checkpointMetadata() const
{
  const Snapshot snapshot = capture();
  return {CONTRIBUTOR_ID, {1, 0, 0}, configurationFingerprint(), snapshot.fingerprint()};
}

void WFTrainSamplerCheckpoint::writeCheckpointPayload(hdf_archive& archive) const
{
  const Snapshot snapshot = capture();
  const std::vector<std::uint64_t> topology{
      persistedSize(topology_.participant_count), persistedSize(topology_.rank),
      persistedSize(topology_.crowd_count), persistedSize(topology_.walker_count),
      persistedSize(topology_.particle_count), persistedSize(topology_.rng_state_sizes.size())};
  archive.write(topology, "topology");
  archive.write(snapshot.positions, "positions");
  archive.write(snapshot.spins, "spins");
  archive.write(snapshot.walker_reals, "walker_reals");
  archive.write(snapshot.walker_ids, "walker_ids");
  archive.write(snapshot.walker_integers, "walker_integers");
  archive.write(snapshot.rng_sizes, "rng_sizes");
  archive.write(snapshot.rng_state, "rng_state");
  archive.write(snapshot.global_step, "global_step");
}

std::unique_ptr<SamplerCheckpointContributor::PreparedRestore>
WFTrainSamplerCheckpoint::prepareCheckpointRestore(
    hdf_archive& archive,
    const CheckpointContributorMetadata& saved_metadata)
{
  std::vector<std::uint64_t> topology;
  Snapshot restored;
  archive.read(topology, "topology");
  archive.read(restored.positions, "positions");
  archive.read(restored.spins, "spins");
  archive.read(restored.walker_reals, "walker_reals");
  archive.read(restored.walker_ids, "walker_ids");
  archive.read(restored.walker_integers, "walker_integers");
  archive.read(restored.rng_sizes, "rng_sizes");
  archive.read(restored.rng_state, "rng_state");
  archive.read(restored.global_step, "global_step");

  if (topology.size() != 6 ||
      runtimeSize(topology[0], "participant count") != topology_.participant_count ||
      runtimeSize(topology[1], "rank") != topology_.rank ||
      runtimeSize(topology[2], "crowd count") != topology_.crowd_count ||
      runtimeSize(topology[3], "walker count") != topology_.walker_count ||
      runtimeSize(topology[4], "particle count") != topology_.particle_count ||
      runtimeSize(topology[5], "RNG count") != topology_.rng_state_sizes.size())
    throw std::runtime_error("WFTrain sampler checkpoint topology mismatch");
  const std::size_t walker_particles = checkedProduct(
      topology_.walker_count, topology_.particle_count, "walker-particle");
  const std::size_t coordinates = checkedProduct(walker_particles, OHMMS_DIM, "coordinate");
  const std::size_t pair_fields = checkedProduct(topology_.walker_count, 2, "walker-pair");
  if (restored.positions.size() != coordinates ||
      restored.spins.size() != walker_particles ||
      restored.walker_reals.size() != pair_fields ||
      restored.walker_ids.size() != pair_fields ||
      restored.walker_integers.size() != pair_fields ||
      restored.rng_sizes.size() != topology_.rng_state_sizes.size())
    throw std::runtime_error("WFTrain sampler checkpoint payload shape mismatch");
  if (!std::all_of(restored.positions.begin(), restored.positions.end(),
                   [](double value) { return isFiniteTrainingReal(value); }) ||
      !std::all_of(restored.spins.begin(), restored.spins.end(),
                   [](double value) { return isFiniteTrainingReal(value); }) ||
      !std::all_of(restored.walker_reals.begin(), restored.walker_reals.end(),
                   [](double value) { return isFiniteTrainingReal(value); }))
    throw std::runtime_error("WFTrain sampler checkpoint contains non-finite walker state");
  if (restored.global_step < 0 || restored.global_step > std::numeric_limits<int>::max())
    throw std::runtime_error("WFTrain sampler checkpoint global step is out of range");
  if constexpr (sizeof(long) < sizeof(std::int64_t))
    for (const std::int64_t id : restored.walker_ids)
      if (id < std::numeric_limits<long>::min() || id > std::numeric_limits<long>::max())
        throw std::runtime_error("WFTrain sampler walker ID exceeds this build's long range");

  std::size_t total_rng_values = 0;
  std::vector<std::unique_ptr<RNG>> prepared_rngs;
  prepared_rngs.reserve(restored.rng_sizes.size());
  for (std::size_t rng = 0; rng < restored.rng_sizes.size(); ++rng)
  {
    const std::size_t state_size = runtimeSize(restored.rng_sizes[rng], "RNG state size");
    if (total_rng_values > restored.rng_state.size() ||
        state_size != topology_.rng_state_sizes[rng] ||
        state_size > restored.rng_state.size() - total_rng_values)
      throw std::runtime_error("WFTrain sampler checkpoint RNG state shape mismatch");
    std::vector<RNG::uint_type> native;
    native.reserve(state_size);
    for (std::size_t value = 0; value < state_size; ++value)
    {
      const std::uint64_t persisted = restored.rng_state[total_rng_values++];
      if (persisted > std::numeric_limits<RNG::uint_type>::max())
        throw std::runtime_error("WFTrain sampler RNG value exceeds this build's range");
      native.push_back(static_cast<RNG::uint_type>(persisted));
    }
    auto prepared_rng = random_generators_[rng].get().makeClone();
    if (!prepared_rng)
      throw std::runtime_error("WFTrain sampler RNG failed to create a prepared clone");
    prepared_rng->load(native);
    if (prepared_rng->state_size() != random_generators_[rng].get().state_size())
      throw std::runtime_error("WFTrain sampler prepared RNG has incompatible concrete state");
    std::vector<RNG::uint_type> roundtrip_state;
    prepared_rng->save(roundtrip_state);
    if (roundtrip_state != native)
      throw std::runtime_error("WFTrain sampler prepared RNG rejected its saved state");
    prepared_rngs.push_back(std::move(prepared_rng));
  }
  if (total_rng_values != restored.rng_state.size())
    throw std::runtime_error("WFTrain sampler checkpoint has trailing RNG state");
  if (restored.fingerprint() != saved_metadata.state_fingerprint)
    throw std::runtime_error("WFTrain sampler checkpoint state fingerprint mismatch");

  return std::make_unique<PreparedSamplerRestore>(
      population_, random_generators_, global_step_, std::move(restored.positions),
      std::move(restored.spins), std::move(restored.walker_reals),
      std::move(restored.walker_ids), std::move(restored.walker_integers),
      std::move(prepared_rngs), static_cast<int>(restored.global_step));
}

} // namespace qmcplusplus::wftrain
