//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_DistributedTrainingCheckpoint.cpp
 * @brief Serial and MPI tests for manifest-coordinated WFTrain checkpoints.
 */

#include <catch2/catch_test_macros.hpp>

#include "Message/CommOperators.h"
#include "QMCDrivers/WFTrain/DistributedTrainingCheckpoint.h"
#include "io/hdf/hdf_archive.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Minimal replicated provider used to test distributed checkpoint coordination.
class DistributedToyProvider final : public StructuredParameterProvider
{
public:
  DistributedToyProvider()
      : schema_("distributed/checkpoint/toy",
                {{"weight", {1}, 0, 1, ParameterScalarDomain::REAL64, true, "weight"}})
  {}

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }

  StructuredParameterSnapshot snapshotParameters() const override
  {
    return {schema_.fingerprint(), version_, {value_}};
  }

  std::size_t publishParameters(const StructuredParameterSnapshot& candidate,
                                std::size_t expected_version) override
  {
    if (expected_version != version_ || candidate.version != version_ ||
        candidate.schema_fingerprint != schema_.fingerprint() || candidate.values.size() != 1)
      throw std::runtime_error("Distributed toy provider rejected a candidate");
    value_ = candidate.values[0];
    return ++version_;
  }

private:
  StructuredParameterSchema schema_;
  std::size_t version_ = 0;
  double value_ = 1.0;
};

/// Minimal rank-local sampler contributor with injectable shard-write failure.
class DistributedToySampler final : public SamplerCheckpointContributor
{
public:
  DistributedToySampler(int rank, int participants)
      : rank_(rank), participants_(participants), value_(100 + rank)
  {}

  CheckpointContributorMetadata checkpointMetadata() const override
  {
    return {"distributed/toy/sampler", {1, 0, 0},
            "participants=" + std::to_string(participants_) + ";rank=" + std::to_string(rank_),
            std::to_string(value_)};
  }

  void writeCheckpointPayload(hdf_archive& archive) const override
  {
    if (fail_write_)
      throw std::runtime_error("synthetic distributed sampler write failure");
    archive.write(value_, "value");
  }

  std::unique_ptr<PreparedRestore> prepareCheckpointRestore(
      hdf_archive& archive,
      const CheckpointContributorMetadata& saved_metadata) override
  {
    std::uint64_t restored = 0;
    archive.read(restored, "value");
    if (saved_metadata.state_fingerprint != std::to_string(restored))
      throw std::runtime_error("Distributed toy sampler fingerprint mismatch");

    class Prepared final : public PreparedRestore
    {
    public:
      Prepared(DistributedToySampler& owner, std::uint64_t value)
          : owner_(owner), value_(value)
      {}

      void commit() noexcept override { owner_.value_ = value_; }

    private:
      DistributedToySampler& owner_;
      std::uint64_t value_;
    };
    return std::make_unique<Prepared>(*this, restored);
  }

  void advance() noexcept { value_ += 17; }
  void failWrites(bool fail) noexcept { fail_write_ = fail; }
  std::uint64_t value() const noexcept { return value_; }

private:
  int rank_;
  int participants_;
  std::uint64_t value_;
  bool fail_write_ = false;
};

/// Remove only files owned by one test manifest family.
void removeCheckpointFamily(const std::filesystem::path& manifest, Communicate& communicator)
{
  communicator.barrier();
  if (communicator.rank() == 0)
  {
    const std::string prefix = manifest.filename().string();
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(manifest.parent_path().empty()
                                                                     ? std::filesystem::path(".")
                                                                     : manifest.parent_path()))
      if (entry.path().filename().string().rfind(prefix, 0) == 0)
        std::filesystem::remove(entry.path(), error);
  }
  communicator.barrier();
}

/// Return the exact local topology used by the toy contributor.
SamplerShardTopology toyTopology(Communicate& communicator)
{
  return {static_cast<std::size_t>(communicator.size()),
          static_cast<std::size_t>(communicator.rank()), 1, 1, 1, {2}};
}

} // namespace

TEST_CASE("Distributed training checkpoint continues and preserves published generation",
          "[drivers][training][checkpoint][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  const std::filesystem::path manifest = "wftrain_distributed_checkpoint_test.h5";
  removeCheckpointFamily(manifest, communicator);

  DistributedToyProvider source_provider;
  StructuredParameterSnapshot first_model = source_provider.snapshotParameters();
  first_model.values[0] = 2.5;
  TrainingIterationState source_state{
      1, source_provider.publishParameters(first_model, first_model.version),
      source_provider.parameterSchema().fingerprint()};
  TrainingStageState source_stage{"energy", 0, 1, "distributed-test-v1"};
  DistributedToySampler source_sampler(communicator.rank(), communicator.size());
  source_sampler.advance();
  const std::uint64_t durable_sampler_value = source_sampler.value();
  const double durable_model_value = source_provider.snapshotParameters().values[0];
  const auto topology = toyTopology(communicator);

  DistributedTrainingCheckpoint::saveAtomic(
      manifest, communicator, source_provider, source_state, source_stage, nullptr,
      source_sampler, topology);

  StructuredParameterSnapshot second_model = source_provider.snapshotParameters();
  second_model.values[0] = 9.0;
  source_state.parameter_version =
      source_provider.publishParameters(second_model, second_model.version);
  ++source_state.completed_iterations;
  ++source_stage.completed_stage_iterations;
  source_sampler.advance();
  source_sampler.failWrites(communicator.rank() == communicator.size() - 1);
  CHECK_THROWS(DistributedTrainingCheckpoint::saveAtomic(
      manifest, communicator, source_provider, source_state, source_stage, nullptr,
      source_sampler, topology));
  source_sampler.failWrites(false);

  DistributedToyProvider restored_provider;
  DistributedToySampler restored_sampler(communicator.rank(), communicator.size());
  restored_sampler.advance();
  restored_sampler.advance();
  TrainingIterationState restored_state{
      99, restored_provider.snapshotParameters().version,
      restored_provider.parameterSchema().fingerprint()};
  TrainingStageState restored_stage{"energy", 7, 88, "distributed-test-v1"};
  DistributedTrainingCheckpoint::restore(
      manifest, communicator, restored_provider, restored_state, restored_stage, nullptr,
      restored_sampler, topology);

  CHECK(restored_provider.snapshotParameters().values[0] == durable_model_value);
  CHECK(restored_sampler.value() == durable_sampler_value);
  CHECK(restored_state.completed_iterations == 1);
  CHECK(restored_stage.completed_stage_iterations == 1);

  removeCheckpointFamily(manifest, communicator);
}

} // namespace qmcplusplus::wftrain
