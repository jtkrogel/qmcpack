//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_TrainingCheckpoint.cpp
 * @brief Deterministic continuation and failure-atomicity tests for WFTrain checkpoints.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "QMCDrivers/WFTrain/TrainingCheckpoint.h"
#include "io/hdf/hdf_archive.h"

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Remove test-owned checkpoint files when a section exits through an exception.
class CheckpointFileCleanup
{
public:
  explicit CheckpointFileCleanup(std::initializer_list<std::filesystem::path> paths)
      : paths_(paths)
  {}

  ~CheckpointFileCleanup()
  {
    std::error_code error;
    for (const std::filesystem::path& path : paths_)
      std::filesystem::remove(path, error);
  }

private:
  std::vector<std::filesystem::path> paths_;
};

/// Small versioned provider whose alternative layout exercises schema rejection.
class CheckpointToyProvider final : public StructuredParameterProvider
{
public:
  explicit CheckpointToyProvider(std::string provider_id = "checkpoint/toy",
                                 bool alternative_schema = false)
      : schema_(std::move(provider_id),
                {{"weights", {3}, 0, 3, ParameterScalarDomain::REAL64, true,
                  alternative_schema ? "alternate" : "weights"}})
  {}

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }

  StructuredParameterSnapshot snapshotParameters() const override
  {
    return {schema_.fingerprint(), version_, values_};
  }

  std::size_t publishParameters(const StructuredParameterSnapshot& candidate,
                                std::size_t expected_version) override
  {
    if (reject_publish_)
    {
      reject_publish_ = false;
      throw std::runtime_error("synthetic checkpoint publication failure");
    }
    if (expected_version != version_ || candidate.version != version_)
      throw std::runtime_error("stale checkpoint toy version");
    if (candidate.schema_fingerprint != schema_.fingerprint() ||
        candidate.values.size() != values_.size())
      throw std::invalid_argument("invalid checkpoint toy candidate");
    if (!std::all_of(candidate.values.begin(), candidate.values.end(),
                     [](double value) { return std::isfinite(value); }))
      throw std::invalid_argument("non-finite checkpoint toy candidate");
    values_ = candidate.values;
    return ++version_;
  }

  void rejectNextPublish() noexcept { reject_publish_ = true; }

private:
  StructuredParameterSchema schema_;
  std::size_t version_ = 0;
  std::vector<double> values_{1.0, -2.0, 0.5};
  bool reject_publish_ = false;
};

/// Extend the compact deterministic fingerprint used by toy contributor payloads.
void extendToyFingerprint(std::uint64_t& hash, const void* data, std::size_t bytes) noexcept
{
  const auto* input = static_cast<const unsigned char*>(data);
  for (std::size_t index = 0; index < bytes; ++index)
  {
    hash ^= input[index];
    hash *= UINT64_C(1099511628211);
  }
}

/// Fingerprint one vector/counter payload without textual floating-point conversion.
std::string toyStateFingerprint(const std::vector<double>& values, std::uint64_t counter)
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  extendToyFingerprint(hash, &counter, sizeof(counter));
  if (!values.empty())
    extendToyFingerprint(hash, values.data(), values.size() * sizeof(double));
  std::ostringstream text;
  text << std::hex << std::setw(16) << std::setfill('0') << hash;
  return text.str();
}

/// Toy recurrence state implementing the optimizer checkpoint role.
class ToyOptimizerCheckpoint final : public OptimizerCheckpointContributor
{
public:
  ToyOptimizerCheckpoint(std::vector<double> moments = {0.0, 0.0, 0.0},
                         std::uint64_t updates = 0,
                         std::string identity = "toy/optimizer",
                         std::string configuration = "toy-adam-v1",
                         int major_version = 1)
      : moments_(std::move(moments)),
        updates_(updates),
        identity_(std::move(identity)),
        configuration_(std::move(configuration)),
        major_version_(major_version)
  {}

  CheckpointContributorMetadata checkpointMetadata() const override
  {
    return {identity_, {major_version_, 0, 0}, configuration_,
            toyStateFingerprint(moments_, updates_)};
  }

  void writeCheckpointPayload(hdf_archive& archive) const override
  {
    if (fail_write_)
      throw std::runtime_error("synthetic optimizer checkpoint write failure");
    archive.write(moments_, "moments");
    archive.write(updates_, "update_count");
  }

  std::unique_ptr<PreparedRestore> prepareCheckpointRestore(
      hdf_archive& archive,
      const CheckpointContributorMetadata& saved_metadata) override
  {
    std::vector<double> moments;
    std::uint64_t updates = 0;
    archive.read(moments, "moments");
    archive.read(updates, "update_count");
    if (moments.size() != 3 ||
        !std::all_of(moments.begin(), moments.end(),
                     [](double value) { return std::isfinite(value); }))
      throw std::runtime_error("Toy optimizer checkpoint payload has invalid size or values");
    if (saved_metadata.state_fingerprint != toyStateFingerprint(moments, updates))
      throw std::runtime_error("Toy optimizer checkpoint state fingerprint mismatch");

    class Prepared final : public PreparedRestore
    {
    public:
      Prepared(ToyOptimizerCheckpoint& owner,
               std::vector<double> moments,
               std::uint64_t updates)
          : owner_(owner), moments_(std::move(moments)), updates_(updates)
      {}

      void commit() noexcept override
      {
        owner_.moments_.swap(moments_);
        owner_.updates_ = updates_;
      }

    private:
      ToyOptimizerCheckpoint& owner_;
      std::vector<double> moments_;
      std::uint64_t updates_;
    };

    return std::make_unique<Prepared>(*this, std::move(moments), updates);
  }

  /// Advance the deterministic recurrence and return the next update vector.
  std::vector<double> nextUpdate()
  {
    ++updates_;
    for (std::size_t index = 0; index < moments_.size(); ++index)
      moments_[index] = 0.5 * moments_[index] +
          0.125 * static_cast<double>(updates_ + index + 1);
    return moments_;
  }

  void failWrites(bool fail = true) noexcept { fail_write_ = fail; }
  const std::vector<double>& moments() const noexcept { return moments_; }
  std::uint64_t updates() const noexcept { return updates_; }

private:
  std::vector<double> moments_;
  std::uint64_t updates_;
  std::string identity_;
  std::string configuration_;
  int major_version_;
  bool fail_write_ = false;
};

/// Toy walker/RNG state implementing the sampler checkpoint role.
class ToySamplerCheckpoint final : public SamplerCheckpointContributor
{
public:
  ToySamplerCheckpoint(std::vector<double> positions = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5},
                       std::uint64_t rng_state = 17,
                       std::uint64_t sweeps = 0,
                       std::string identity = "toy/sampler",
                       std::string configuration = "ranks=1;threads=1;walkers=2",
                       int major_version = 1)
      : positions_(std::move(positions)),
        rng_state_(rng_state),
        sweeps_(sweeps),
        identity_(std::move(identity)),
        configuration_(std::move(configuration)),
        major_version_(major_version)
  {}

  CheckpointContributorMetadata checkpointMetadata() const override
  {
    std::vector<double> payload = positions_;
    payload.push_back(static_cast<double>(sweeps_));
    return {identity_, {major_version_, 0, 0}, configuration_,
            toyStateFingerprint(payload, rng_state_)};
  }

  void writeCheckpointPayload(hdf_archive& archive) const override
  {
    archive.write(positions_, "walker_positions");
    archive.write(rng_state_, "rng_state");
    archive.write(sweeps_, "sweep_count");
  }

  std::unique_ptr<PreparedRestore> prepareCheckpointRestore(
      hdf_archive& archive,
      const CheckpointContributorMetadata& saved_metadata) override
  {
    std::vector<double> positions;
    std::uint64_t rng_state = 0;
    std::uint64_t sweeps    = 0;
    archive.read(positions, "walker_positions");
    archive.read(rng_state, "rng_state");
    archive.read(sweeps, "sweep_count");
    if (positions.size() != 6 ||
        !std::all_of(positions.begin(), positions.end(),
                     [](double value) { return std::isfinite(value); }))
      throw std::runtime_error("Toy sampler checkpoint payload has invalid size or values");
    std::vector<double> payload = positions;
    payload.push_back(static_cast<double>(sweeps));
    if (saved_metadata.state_fingerprint != toyStateFingerprint(payload, rng_state))
      throw std::runtime_error("Toy sampler checkpoint state fingerprint mismatch");

    class Prepared final : public PreparedRestore
    {
    public:
      Prepared(ToySamplerCheckpoint& owner,
               std::vector<double> positions,
               std::uint64_t rng_state,
               std::uint64_t sweeps)
          : owner_(owner),
            positions_(std::move(positions)),
            rng_state_(rng_state),
            sweeps_(sweeps)
      {}

      void commit() noexcept override
      {
        owner_.positions_.swap(positions_);
        owner_.rng_state_ = rng_state_;
        owner_.sweeps_    = sweeps_;
      }

    private:
      ToySamplerCheckpoint& owner_;
      std::vector<double> positions_;
      std::uint64_t rng_state_;
      std::uint64_t sweeps_;
    };

    return std::make_unique<Prepared>(*this, std::move(positions), rng_state, sweeps);
  }

  /// Advance walker coordinates and a tiny deterministic linear-congruential RNG.
  double advance()
  {
    rng_state_ = rng_state_ * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    const double displacement =
        static_cast<double>((rng_state_ >> 58) & UINT64_C(0x3f)) * 0.000125;
    for (std::size_t index = 0; index < positions_.size(); ++index)
      positions_[index] += displacement * static_cast<double>(index + 1);
    ++sweeps_;
    return displacement;
  }

  const std::vector<double>& positions() const noexcept { return positions_; }
  std::uint64_t rngState() const noexcept { return rng_state_; }
  std::uint64_t sweeps() const noexcept { return sweeps_; }

private:
  std::vector<double> positions_;
  std::uint64_t rng_state_;
  std::uint64_t sweeps_;
  std::string identity_;
  std::string configuration_;
  int major_version_;
};

/// Advance all state that must agree for an exact post-update continuation.
void advanceToyTrajectory(CheckpointToyProvider& provider,
                          ToyOptimizerCheckpoint& optimizer,
                          ToySamplerCheckpoint& sampler,
                          TrainingIterationState& coordinator,
                          TrainingStageState& stage,
                          std::size_t iterations)
{
  for (std::size_t iteration = 0; iteration < iterations; ++iteration)
  {
    const std::vector<double> update = optimizer.nextUpdate();
    const double displacement       = sampler.advance();
    StructuredParameterSnapshot candidate = provider.snapshotParameters();
    for (std::size_t parameter = 0; parameter < candidate.values.size(); ++parameter)
      candidate.values[parameter] -= 0.01 * update[parameter] + displacement;
    coordinator.parameter_version =
        provider.publishParameters(candidate, candidate.version);
    ++coordinator.completed_iterations;
    coordinator.schema_fingerprint = provider.parameterSchema().fingerprint();
    ++stage.completed_stage_iterations;
  }
}

/// Increment only the fresh provider generation to exercise restore rebasing.
void advanceProviderVersion(CheckpointToyProvider& provider, std::size_t count)
{
  for (std::size_t iteration = 0; iteration < count; ++iteration)
  {
    StructuredParameterSnapshot candidate = provider.snapshotParameters();
    provider.publishParameters(candidate, candidate.version);
  }
}

/// Copy one valid checkpoint before applying an adversarial raw-HDF mutation.
void copyCheckpoint(const std::filesystem::path& source,
                    const std::filesystem::path& destination)
{
  std::error_code error;
  std::filesystem::copy_file(source, destination,
                             std::filesystem::copy_options::overwrite_existing, error);
  if (error)
    throw std::runtime_error("Unable to copy checkpoint test file: " + error.message());
}

/// Delete one HDF link to model an interrupted file with missing mandatory data.
void deleteHDFLink(const std::filesystem::path& file, const char* path)
{
  const hid_t handle = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  if (handle < 0)
    throw std::runtime_error("Unable to open checkpoint mutation file");
  const herr_t deleted = H5Ldelete(handle, path, H5P_DEFAULT);
  H5Fclose(handle);
  if (deleted < 0)
    throw std::runtime_error("Unable to delete checkpoint mutation link");
}

/// Overwrite one scalar uint64 dataset in an otherwise valid checkpoint.
void overwriteUint64(const std::filesystem::path& file,
                     const char* path,
                     std::uint64_t value)
{
  const hid_t handle  = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  const hid_t dataset = H5Dopen2(handle, path, H5P_DEFAULT);
  if (handle < 0 || dataset < 0 ||
      H5Dwrite(dataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value) < 0)
  {
    if (dataset >= 0)
      H5Dclose(dataset);
    if (handle >= 0)
      H5Fclose(handle);
    throw std::runtime_error("Unable to mutate uint64 checkpoint dataset");
  }
  H5Dclose(dataset);
  H5Fclose(handle);
}

/// Overwrite the first version component to exercise format rejection.
void overwriteFormatMajor(const std::filesystem::path& file, int major)
{
  const hid_t handle  = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  const hid_t dataset = H5Dopen2(handle, "/wftrain_checkpoint/format_version", H5P_DEFAULT);
  std::array<int, 3> version{major, 0, 0};
  if (handle < 0 || dataset < 0 ||
      H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, version.data()) < 0)
  {
    if (dataset >= 0)
      H5Dclose(dataset);
    if (handle >= 0)
      H5Fclose(handle);
    throw std::runtime_error("Unable to mutate checkpoint format version");
  }
  H5Dclose(dataset);
  H5Fclose(handle);
}

/// Overwrite one fixed-size vector dataset without changing its HDF shape.
void overwriteDoubleVector(const std::filesystem::path& file,
                           const char* path,
                           const std::vector<double>& values)
{
  const hid_t handle  = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  const hid_t dataset = H5Dopen2(handle, path, H5P_DEFAULT);
  if (handle < 0 || dataset < 0 ||
      H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               values.data()) < 0)
  {
    if (dataset >= 0)
      H5Dclose(dataset);
    if (handle >= 0)
      H5Fclose(handle);
    throw std::runtime_error("Unable to mutate checkpoint vector dataset");
  }
  H5Dclose(dataset);
  H5Fclose(handle);
}

/// Replace one vector dataset, allowing an adversarial payload-size change.
void replaceDoubleVector(const std::filesystem::path& file,
                         const char* group,
                         const char* name,
                         const std::vector<double>& values)
{
  const std::string absolute_path =
      std::string("/wftrain_checkpoint/") + group + "/" + name;
  deleteHDFLink(file, absolute_path.c_str());
  hdf_archive archive;
  if (!archive.open(file))
    throw std::runtime_error("Unable to open checkpoint vector replacement file");
  archive.push("wftrain_checkpoint", false);
  archive.push(group, false);
  archive.write(values, name);
  archive.pop();
  archive.pop();
  archive.close();
}

/// Verify all mutable restore targets retain their captured pre-call state.
void checkRestoreTargetsUnchanged(const CheckpointToyProvider& provider,
                                  const StructuredParameterSnapshot& model_before,
                                  const TrainingIterationState& coordinator,
                                  const TrainingIterationState& coordinator_before,
                                  const TrainingStageState& stage,
                                  const TrainingStageState& stage_before,
                                  const ToyOptimizerCheckpoint& optimizer,
                                  const std::vector<double>& moments_before,
                                  std::uint64_t updates_before,
                                  const ToySamplerCheckpoint& sampler,
                                  const std::vector<double>& positions_before,
                                  std::uint64_t rng_before,
                                  std::uint64_t sweeps_before)
{
  const StructuredParameterSnapshot model_after = provider.snapshotParameters();
  CHECK(model_after.version == model_before.version);
  CHECK(model_after.values == model_before.values);
  CHECK(coordinator.completed_iterations == coordinator_before.completed_iterations);
  CHECK(coordinator.parameter_version == coordinator_before.parameter_version);
  CHECK(coordinator.schema_fingerprint == coordinator_before.schema_fingerprint);
  CHECK(stage.stage_id == stage_before.stage_id);
  CHECK(stage.stage_ordinal == stage_before.stage_ordinal);
  CHECK(stage.completed_stage_iterations == stage_before.completed_stage_iterations);
  CHECK(stage.configuration_fingerprint == stage_before.configuration_fingerprint);
  CHECK(optimizer.moments() == moments_before);
  CHECK(optimizer.updates() == updates_before);
  CHECK(sampler.positions() == positions_before);
  CHECK(sampler.rngState() == rng_before);
  CHECK(sampler.sweeps() == sweeps_before);
}

} // namespace

TEST_CASE("Training checkpoint exactly continues deterministic toy state",
          "[drivers][training][checkpoint]")
{
  const std::filesystem::path file = "wftrain_checkpoint_continuation.h5";
  CheckpointFileCleanup cleanup{file};
  TrainingStageState initial_stage{"optimization", 1, 0, "lih-stage-v1"};

  CheckpointToyProvider uninterrupted_provider;
  ToyOptimizerCheckpoint uninterrupted_optimizer;
  ToySamplerCheckpoint uninterrupted_sampler;
  TrainingIterationState uninterrupted_coordinator;
  TrainingStageState uninterrupted_stage = initial_stage;
  advanceToyTrajectory(uninterrupted_provider, uninterrupted_optimizer,
                       uninterrupted_sampler, uninterrupted_coordinator,
                       uninterrupted_stage, 2);
  TrainingCheckpoint::saveAtomic(file, uninterrupted_provider,
                                 uninterrupted_coordinator, uninterrupted_stage,
                                 &uninterrupted_optimizer, &uninterrupted_sampler);
  advanceToyTrajectory(uninterrupted_provider, uninterrupted_optimizer,
                       uninterrupted_sampler, uninterrupted_coordinator,
                       uninterrupted_stage, 3);

  CheckpointToyProvider resumed_provider;
  advanceProviderVersion(resumed_provider, 5);
  ToyOptimizerCheckpoint resumed_optimizer({9.0, 8.0, 7.0}, 41);
  ToySamplerCheckpoint resumed_sampler({9.0, 8.0, 7.0, 6.0, 5.0, 4.0}, 999, 31);
  TrainingIterationState resumed_coordinator{99, 5, "discarded-runtime-schema"};
  TrainingStageState resumed_stage{"optimization", 77, 88, "lih-stage-v1"};

  TrainingCheckpoint::restore(file, resumed_provider, resumed_coordinator,
                              resumed_stage, &resumed_optimizer, &resumed_sampler);
  CHECK(resumed_provider.snapshotParameters().version == 6);
  CHECK(resumed_coordinator.parameter_version == 6);
  CHECK(resumed_coordinator.completed_iterations == 2);
  CHECK(resumed_stage.stage_ordinal == 1);
  CHECK(resumed_stage.completed_stage_iterations == 2);
  advanceToyTrajectory(resumed_provider, resumed_optimizer, resumed_sampler,
                       resumed_coordinator, resumed_stage, 3);

  CHECK(resumed_provider.snapshotParameters().values ==
        uninterrupted_provider.snapshotParameters().values);
  CHECK(resumed_optimizer.moments() == uninterrupted_optimizer.moments());
  CHECK(resumed_optimizer.updates() == uninterrupted_optimizer.updates());
  CHECK(resumed_sampler.positions() == uninterrupted_sampler.positions());
  CHECK(resumed_sampler.rngState() == uninterrupted_sampler.rngState());
  CHECK(resumed_sampler.sweeps() == uninterrupted_sampler.sweeps());
  CHECK(resumed_coordinator.completed_iterations ==
        uninterrupted_coordinator.completed_iterations);
  CHECK(resumed_stage.completed_stage_iterations ==
        uninterrupted_stage.completed_stage_iterations);
}

TEST_CASE("Training checkpoint interrupted save preserves prior generation",
          "[drivers][training][checkpoint]")
{
  const std::filesystem::path file = "wftrain_checkpoint_atomic_save.h5";
  CheckpointFileCleanup cleanup{file};
  CheckpointToyProvider provider;
  ToyOptimizerCheckpoint optimizer;
  ToySamplerCheckpoint sampler;
  TrainingIterationState coordinator;
  TrainingStageState stage{"optimization", 2, 0, "atomic-stage-v1"};
  advanceToyTrajectory(provider, optimizer, sampler, coordinator, stage, 1);
  TrainingCheckpoint::saveAtomic(file, provider, coordinator, stage, &optimizer, &sampler);
  const StructuredParameterSnapshot durable_model = provider.snapshotParameters();

  advanceToyTrajectory(provider, optimizer, sampler, coordinator, stage, 1);
  optimizer.failWrites();
  CHECK_THROWS_WITH(
      TrainingCheckpoint::saveAtomic(file, provider, coordinator, stage, &optimizer, &sampler),
      "synthetic optimizer checkpoint write failure");

  CheckpointToyProvider restored_provider;
  ToyOptimizerCheckpoint restored_optimizer;
  ToySamplerCheckpoint restored_sampler;
  TrainingIterationState restored_coordinator;
  TrainingStageState restored_stage{"optimization", 0, 0, "atomic-stage-v1"};
  TrainingCheckpoint::restore(file, restored_provider, restored_coordinator,
                              restored_stage, &restored_optimizer, &restored_sampler);
  CHECK(restored_provider.snapshotParameters().values == durable_model.values);
  CHECK(restored_coordinator.completed_iterations == 1);
  CHECK(restored_stage.completed_stage_iterations == 1);
}

TEST_CASE("Training checkpoint validation failures leave every target unchanged",
          "[drivers][training][checkpoint]")
{
  const std::filesystem::path valid_file = "wftrain_checkpoint_valid.h5";
  const std::filesystem::path damaged_file = "wftrain_checkpoint_damaged.h5";
  CheckpointFileCleanup cleanup{valid_file, damaged_file};
  CheckpointToyProvider source_provider;
  ToyOptimizerCheckpoint source_optimizer;
  ToySamplerCheckpoint source_sampler;
  TrainingIterationState source_coordinator;
  TrainingStageState source_stage{"optimization", 3, 0, "validation-stage-v1"};
  advanceToyTrajectory(source_provider, source_optimizer, source_sampler,
                       source_coordinator, source_stage, 2);
  TrainingCheckpoint::saveAtomic(valid_file, source_provider, source_coordinator,
                                 source_stage, &source_optimizer, &source_sampler);

  auto run_failure = [&](const std::filesystem::path& input,
                         CheckpointToyProvider& target_provider,
                         ToyOptimizerCheckpoint& target_optimizer,
                         ToySamplerCheckpoint& target_sampler,
                         TrainingStageState target_stage) {
    TrainingIterationState target_coordinator{71, target_provider.snapshotParameters().version,
                                               "live-coordinator-sentinel"};
    const StructuredParameterSnapshot model_before = target_provider.snapshotParameters();
    const TrainingIterationState coordinator_before = target_coordinator;
    const TrainingStageState stage_before = target_stage;
    const std::vector<double> moments_before = target_optimizer.moments();
    const std::uint64_t updates_before = target_optimizer.updates();
    const std::vector<double> positions_before = target_sampler.positions();
    const std::uint64_t rng_before = target_sampler.rngState();
    const std::uint64_t sweeps_before = target_sampler.sweeps();
    CHECK_THROWS(TrainingCheckpoint::restore(input, target_provider, target_coordinator,
                                             target_stage, &target_optimizer,
                                             &target_sampler));
    checkRestoreTargetsUnchanged(
        target_provider, model_before, target_coordinator, coordinator_before,
        target_stage, stage_before, target_optimizer, moments_before, updates_before,
        target_sampler, positions_before, rng_before, sweeps_before);
  };

  SECTION("missing completion cookie")
  {
    copyCheckpoint(valid_file, damaged_file);
    deleteHDFLink(damaged_file, "/wftrain_checkpoint/completion_cookie");
    CheckpointToyProvider target_provider;
    ToyOptimizerCheckpoint target_optimizer;
    ToySamplerCheckpoint target_sampler;
    run_failure(damaged_file, target_provider, target_optimizer, target_sampler,
                {"optimization", 0, 0, "validation-stage-v1"});
  }

  SECTION("incompatible envelope version")
  {
    copyCheckpoint(valid_file, damaged_file);
    overwriteFormatMajor(damaged_file, 2);
    CheckpointToyProvider target_provider;
    ToyOptimizerCheckpoint target_optimizer;
    ToySamplerCheckpoint target_sampler;
    run_failure(damaged_file, target_provider, target_optimizer, target_sampler,
                {"optimization", 0, 0, "validation-stage-v1"});
  }

  SECTION("provider and schema identities")
  {
    CheckpointToyProvider wrong_provider("checkpoint/other");
    ToyOptimizerCheckpoint target_optimizer;
    ToySamplerCheckpoint target_sampler;
    run_failure(valid_file, wrong_provider, target_optimizer, target_sampler,
                {"optimization", 0, 0, "validation-stage-v1"});

    CheckpointToyProvider wrong_schema("checkpoint/toy", true);
    run_failure(valid_file, wrong_schema, target_optimizer, target_sampler,
                {"optimization", 0, 0, "validation-stage-v1"});
  }

  SECTION("stage and contributor identities")
  {
    CheckpointToyProvider target_provider;
    ToyOptimizerCheckpoint target_optimizer;
    ToySamplerCheckpoint target_sampler;
    run_failure(valid_file, target_provider, target_optimizer, target_sampler,
                {"different-stage", 0, 0, "validation-stage-v1"});
    run_failure(valid_file, target_provider, target_optimizer, target_sampler,
                {"optimization", 0, 0, "different-stage-config"});

    ToyOptimizerCheckpoint wrong_identity({0.0, 0.0, 0.0}, 0, "other/optimizer");
    run_failure(valid_file, target_provider, wrong_identity, target_sampler,
                {"optimization", 0, 0, "validation-stage-v1"});
    ToyOptimizerCheckpoint wrong_version({0.0, 0.0, 0.0}, 0, "toy/optimizer",
                                         "toy-adam-v1", 2);
    run_failure(valid_file, target_provider, wrong_version, target_sampler,
                {"optimization", 0, 0, "validation-stage-v1"});
    ToySamplerCheckpoint wrong_topology({0.0, 0.5, 1.0, 1.5, 2.0, 2.5}, 17, 0,
                                        "toy/sampler", "ranks=2;threads=1;walkers=2");
    run_failure(valid_file, target_provider, target_optimizer, wrong_topology,
                {"optimization", 0, 0, "validation-stage-v1"});
  }

  SECTION("non-finite model values")
  {
    copyCheckpoint(valid_file, damaged_file);
    const std::vector<double> bad_values{
        1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    overwriteDoubleVector(damaged_file, "/wftrain_checkpoint/model/values", bad_values);
    CheckpointToyProvider target_provider;
    ToyOptimizerCheckpoint target_optimizer;
    ToySamplerCheckpoint target_sampler;
    run_failure(damaged_file, target_provider, target_optimizer, target_sampler,
                {"optimization", 0, 0, "validation-stage-v1"});
  }

  SECTION("finite model values with a stale fingerprint")
  {
    copyCheckpoint(valid_file, damaged_file);
    const std::vector<double> changed_values{9.0, 8.0, 7.0};
    overwriteDoubleVector(damaged_file, "/wftrain_checkpoint/model/values", changed_values);
    CheckpointToyProvider target_provider;
    ToyOptimizerCheckpoint target_optimizer;
    ToySamplerCheckpoint target_sampler;
    run_failure(damaged_file, target_provider, target_optimizer, target_sampler,
                {"optimization", 0, 0, "validation-stage-v1"});
  }

  SECTION("model and coordinator source-version mismatch")
  {
    copyCheckpoint(valid_file, damaged_file);
    overwriteUint64(damaged_file,
                    "/wftrain_checkpoint/coordinator/source_parameter_version", 999);
    CheckpointToyProvider target_provider;
    ToyOptimizerCheckpoint target_optimizer;
    ToySamplerCheckpoint target_sampler;
    run_failure(damaged_file, target_provider, target_optimizer, target_sampler,
                {"optimization", 0, 0, "validation-stage-v1"});
  }

  SECTION("contributor payload size")
  {
    copyCheckpoint(valid_file, damaged_file);
    replaceDoubleVector(damaged_file, "optimizer", "moments", {1.0, 2.0});
    CheckpointToyProvider target_provider;
    ToyOptimizerCheckpoint target_optimizer;
    ToySamplerCheckpoint target_sampler;
    run_failure(damaged_file, target_provider, target_optimizer, target_sampler,
                {"optimization", 0, 0, "validation-stage-v1"});
  }

  SECTION("provider publication failure after complete preparation")
  {
    CheckpointToyProvider target_provider;
    target_provider.rejectNextPublish();
    ToyOptimizerCheckpoint target_optimizer;
    ToySamplerCheckpoint target_sampler;
    run_failure(valid_file, target_provider, target_optimizer, target_sampler,
                {"optimization", 0, 0, "validation-stage-v1"});
  }
}

TEST_CASE("Training checkpoint supports stateless and generation-qualified files",
          "[drivers][training][checkpoint]")
{
  const std::filesystem::path first = "wftrain_checkpoint_generation_0001.h5";
  const std::filesystem::path second = "wftrain_checkpoint_generation_0002.h5";
  CheckpointFileCleanup cleanup{first, second};
  CheckpointToyProvider provider;
  TrainingIterationState coordinator{0, 0, provider.parameterSchema().fingerprint()};
  TrainingStageState stage{"inference", 0, 0, "stateless-stage-v1"};
  TrainingCheckpoint::saveAtomic(first, provider, coordinator, stage);

  StructuredParameterSnapshot candidate = provider.snapshotParameters();
  candidate.values[0] = 4.25;
  coordinator.parameter_version =
      provider.publishParameters(candidate, candidate.version);
  ++coordinator.completed_iterations;
  ++stage.completed_stage_iterations;
  TrainingCheckpoint::saveAtomic(second, provider, coordinator, stage);
  CHECK(std::filesystem::exists(first));
  CHECK(std::filesystem::exists(second));

  CheckpointToyProvider first_provider;
  TrainingIterationState first_coordinator;
  TrainingStageState first_stage{"inference", 19, 19, "stateless-stage-v1"};
  TrainingCheckpoint::restore(first, first_provider, first_coordinator, first_stage);
  CHECK(first_provider.snapshotParameters().values[0] == 1.0);
  CHECK(first_coordinator.completed_iterations == 0);

  CheckpointToyProvider second_provider;
  TrainingIterationState second_coordinator;
  TrainingStageState second_stage{"inference", 19, 19, "stateless-stage-v1"};
  TrainingCheckpoint::restore(second, second_provider, second_coordinator, second_stage);
  CHECK(second_provider.snapshotParameters().values[0] == 4.25);
  CHECK(second_coordinator.completed_iterations == 1);
}

} // namespace qmcplusplus::wftrain
