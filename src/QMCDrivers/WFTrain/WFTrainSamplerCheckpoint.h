//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file WFTrainSamplerCheckpoint.h
 * @brief Exact rank-local walker and RNG checkpoint contributor for WFTrain.
 */

#ifndef QMCPLUSPLUS_WFTRAIN_SAMPLER_CHECKPOINT_H
#define QMCPLUSPLUS_WFTRAIN_SAMPLER_CHECKPOINT_H

#include "QMCDrivers/WFTrain/TrainingCheckpoint.h"
#include "QMCDrivers/MCPopulation.h"
#include "Utilities/RandomBase.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qmcplusplus::wftrain
{

/// Exact process topology required to resume one sampler shard.
struct SamplerShardTopology
{
  std::size_t participant_count = 0;
  std::size_t rank              = 0;
  std::size_t crowd_count       = 0;
  std::size_t walker_count      = 0;
  std::size_t particle_count    = 0;
  std::vector<std::size_t> rng_state_sizes;
};

/** Persist the rank-local WFTrain Markov-chain state without model duplication.
 *
 * Accepted ECP grids are persisted. Derived wavefunction/Hamiltonian caches are
 * rebuilt deterministically after restoring walkers, quadrature grids and RNGs.
 */
class WFTrainSamplerCheckpoint final : public SamplerCheckpointContributor
{
public:
  using FullPrecRealType = QMCTraits::FullPrecRealType;
  using RNG              = RandomBase<FullPrecRealType>;

  WFTrainSamplerCheckpoint(MCPopulation& population,
                           const RefVector<RNG>& random_generators,
                           int& global_step,
                           std::size_t participant_count,
                           std::size_t rank,
                           std::size_t crowd_count,
                           std::vector<QMCTraits::IndexType>* subset_cursors = nullptr);

  const SamplerShardTopology& topology() const noexcept { return topology_; }

  CheckpointContributorMetadata checkpointMetadata() const override;
  void writeCheckpointPayload(hdf_archive& archive) const override;
  std::unique_ptr<PreparedRestore> prepareCheckpointRestore(
      hdf_archive& archive,
      const CheckpointContributorMetadata& saved_metadata) override;

private:
  struct Snapshot;

  Snapshot capture() const;
  std::string configurationFingerprint() const;

  MCPopulation& population_;
  const RefVector<RNG>& random_generators_;
  int& global_step_;
  std::vector<QMCTraits::IndexType>* subset_cursors_;
  SamplerShardTopology topology_;
};

} // namespace qmcplusplus::wftrain

#endif
