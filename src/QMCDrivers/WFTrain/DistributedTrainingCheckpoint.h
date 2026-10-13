//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file DistributedTrainingCheckpoint.h
 * @brief Manifest-coordinated distributed checkpoints for WFTrain.
 */

#ifndef QMCPLUSPLUS_DISTRIBUTED_TRAINING_CHECKPOINT_H
#define QMCPLUSPLUS_DISTRIBUTED_TRAINING_CHECKPOINT_H

#include "QMCDrivers/WFTrain/TrainingCheckpoint.h"
#include "QMCDrivers/WFTrain/WFTrainSamplerCheckpoint.h"

#include <filesystem>

class Communicate;

namespace qmcplusplus::wftrain
{

/** Coordinate one replicated root envelope and rank-local sampler shards.
 *
 * Generation-qualified payload files are never overwritten. The root manifest
 * is atomically published last, so an interrupted save leaves the preceding
 * generation selected and any incomplete new generation unreachable.
 */
class DistributedTrainingCheckpoint
{
public:
  static constexpr std::array<int, 3> FORMAT_VERSION{1, 0, 0};

  static void saveAtomic(const std::filesystem::path& manifest,
                         Communicate& communicator,
                         const StructuredParameterProvider& provider,
                         const TrainingIterationState& coordinator_state,
                         const TrainingStageState& stage_state,
                         const OptimizerCheckpointContributor* optimizer,
                         const SamplerCheckpointContributor& sampler,
                         const SamplerShardTopology& topology);

  static void restore(const std::filesystem::path& manifest,
                      Communicate& communicator,
                      StructuredParameterProvider& provider,
                      TrainingIterationState& coordinator_state,
                      TrainingStageState& stage_state,
                      OptimizerCheckpointContributor* optimizer,
                      SamplerCheckpointContributor& sampler,
                      const SamplerShardTopology& topology);
};

} // namespace qmcplusplus::wftrain

#endif
