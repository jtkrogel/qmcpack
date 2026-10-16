//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file TrainingCheckpoint.h
 * @brief Atomic, versioned checkpoints for the high-parameter training route.
 */

#ifndef QMCPLUSPLUS_TRAINING_CHECKPOINT_H
#define QMCPLUSPLUS_TRAINING_CHECKPOINT_H

#include "QMCDrivers/WFTrain/HighParameterTraining.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

namespace qmcplusplus
{
class hdf_archive;

namespace wftrain
{

/// Persistent progress and configuration identity for one training stage.
struct TrainingStageState
{
  std::string stage_id;
  std::size_t stage_ordinal = 0;
  std::size_t completed_stage_iterations = 0;
  std::string configuration_fingerprint;
};

/// Self-identifying metadata common to the two optional checkpoint roles.
struct CheckpointContributorMetadata
{
  std::string contributor_id;
  std::array<int, 3> format_version{1, 0, 0};
  std::string configuration_fingerprint;
  std::string state_fingerprint;
};

/** Optional owner of optimizer recurrence state.
 *
 * Restore first builds an owning prepared action. Its final commit must only
 * swap or publish values and therefore cannot fail after model publication.
 */
class OptimizerCheckpointContributor
{
public:
  class PreparedRestore
  {
  public:
    virtual ~PreparedRestore() = default;
    virtual void commit() noexcept = 0;
  };

  virtual ~OptimizerCheckpointContributor() = default;
  virtual CheckpointContributorMetadata checkpointMetadata() const = 0;
  virtual void writeCheckpointPayload(hdf_archive& archive) const = 0;
  virtual std::unique_ptr<PreparedRestore> prepareCheckpointRestore(
      hdf_archive& archive,
      const CheckpointContributorMetadata& saved_metadata) = 0;
};

/** Optional owner of sampler, walker, and random-number-generator state.
 *
 * The production sampler adapter will use this role once that driver exists;
 * the fixed contract is independently testable with deterministic state.
 */
class SamplerCheckpointContributor
{
public:
  class PreparedRestore
  {
  public:
    virtual ~PreparedRestore() = default;
    virtual void commit() noexcept = 0;
  };

  virtual ~SamplerCheckpointContributor() = default;
  virtual CheckpointContributorMetadata checkpointMetadata() const = 0;
  virtual void writeCheckpointPayload(hdf_archive& archive) const = 0;
  virtual std::unique_ptr<PreparedRestore> prepareCheckpointRestore(
      hdf_archive& archive,
      const CheckpointContributorMetadata& saved_metadata) = 0;
};

/** Read and write the fixed v1 high-parameter training checkpoint envelope.
 *
 * A save publishes by same-directory rename only after the completion cookie
 * is flushed. Restore prepares every optional role before changing live state.
 */
class TrainingCheckpoint
{
public:
  static constexpr std::array<int, 3> FORMAT_VERSION{1, 0, 0};

  /** Fully validated owning restore state whose live targets are still unchanged. */
  class PreparedRestore
  {
  public:
    ~PreparedRestore();
    PreparedRestore(PreparedRestore&&) noexcept;
    PreparedRestore& operator=(PreparedRestore&&) noexcept;

    PreparedRestore(const PreparedRestore&)            = delete;
    PreparedRestore& operator=(const PreparedRestore&) = delete;

    /// Publish the prepared model (including a conventional orbital frame when
    /// present), retaining all other prepared state until publication consensus.
    std::size_t publishModel();

    /// Install driver and contributor state after distributed publication consensus.
    void commitAfterPublication(std::size_t committed_version) noexcept;

  private:
    friend class TrainingCheckpoint;
    struct Impl;
    explicit PreparedRestore(std::unique_ptr<Impl> implementation);
    std::unique_ptr<Impl> implementation_;
  };

  /// Atomically replace destination with one complete post-iteration checkpoint.
  static void saveAtomic(const std::filesystem::path& destination,
                         const StructuredParameterProvider& provider,
                         const TrainingIterationState& coordinator_state,
                         const TrainingStageState& stage_state,
                         const OptimizerCheckpointContributor* optimizer = nullptr,
                         const SamplerCheckpointContributor* sampler = nullptr);

  /** Read canonical parameters from one exact root envelope without publication.
   *
   * Validates mandatory model metadata against expected_schema. The returned
   * version is source provenance only; no optimizer, stage, sampler or RNG state
   * is restored. Current input physics defines the new wavefunction.
   */
  static StructuredParameterSnapshot readModelParameters(
      const std::filesystem::path& source,
      const StructuredParameterSchema& expected_schema);

  /// Exact canonical double bit-pattern fingerprint shared by model-only exports.
  static std::string valuesFingerprint(const std::vector<double>& values);

  /** Restore a complete checkpoint while preserving failure atomicity.
   *
   * stage_state supplies the expected stage/configuration identity. Saved
   * counters replace its counters only after model publication succeeds.
   */
  static void restore(const std::filesystem::path& source,
                      StructuredParameterProvider& provider,
                      TrainingIterationState& coordinator_state,
                      TrainingStageState& stage_state,
                      OptimizerCheckpointContributor* optimizer = nullptr,
                      SamplerCheckpointContributor* sampler = nullptr);

  /** Validate and own a restore without changing any live target. */
  static PreparedRestore prepareRestore(
      const std::filesystem::path& source,
      StructuredParameterProvider& provider,
      TrainingIterationState& coordinator_state,
      TrainingStageState& stage_state,
      OptimizerCheckpointContributor* optimizer = nullptr,
      SamplerCheckpointContributor* sampler = nullptr);

  /// Atomically write one sampler-only rank shard.
  static void saveSamplerShardAtomic(
      const std::filesystem::path& destination,
      const SamplerCheckpointContributor& sampler);

  /// Validate one sampler-only shard and return its non-mutating prepared action.
  static std::unique_ptr<SamplerCheckpointContributor::PreparedRestore>
  prepareSamplerShardRestore(const std::filesystem::path& source,
                             SamplerCheckpointContributor& sampler);
};

} // namespace wftrain
} // namespace qmcplusplus

#endif
