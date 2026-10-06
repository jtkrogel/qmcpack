//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file TrainingCheckpoint.cpp
 * @brief Failure-atomic HDF persistence for high-parameter training state.
 */

#include "QMCDrivers/WFTrain/TrainingCheckpoint.h"

#include "io/hdf/hdf_archive.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

constexpr char ROOT_GROUP[]        = "wftrain_checkpoint";
constexpr char COMPLETION_COOKIE[] = "wftrain-checkpoint-complete-v1";

/// Convert a platform-sized counter to the portable checkpoint representation.
std::uint64_t checkedPersistedSize(std::size_t value)
{
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t))
    if (value > std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("Training checkpoint counter exceeds uint64 storage");
  return static_cast<std::uint64_t>(value);
}

/// Convert a persisted counter without accepting truncation on narrow platforms.
std::size_t checkedRuntimeSize(std::uint64_t value, const char* field)
{
  if (value > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error(std::string("Training checkpoint ") + field +
                             " exceeds this build's size_t range");
  return static_cast<std::size_t>(value);
}

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

/// Fingerprint the exact stored parameter bit pattern and count.
std::string parameterFingerprint(const std::vector<double>& values)
{
  std::uint64_t hash        = UINT64_C(1469598103934665603);
  const std::uint64_t count = values.size();
  extendFingerprint(hash, &count, sizeof(count));
  if (!values.empty())
    extendFingerprint(hash, values.data(), values.size() * sizeof(double));
  std::ostringstream text;
  text << std::hex << std::setw(16) << std::setfill('0') << hash;
  return text.str();
}

/// Reject incomplete or internally inconsistent mandatory checkpoint state.
void validateSaveState(const StructuredParameterSchema& schema,
                       const StructuredParameterSnapshot& model,
                       const TrainingIterationState& coordinator,
                       const TrainingStageState& stage)
{
  if (schema.providerId().empty() || schema.fingerprint().empty())
    throw std::invalid_argument("Training checkpoint requires model identity metadata");
  if (model.schema_fingerprint != schema.fingerprint() ||
      model.values.size() != schema.parameterCount())
    throw std::invalid_argument("Training checkpoint model snapshot does not match its schema");
  if (!std::all_of(model.values.begin(), model.values.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("Training checkpoint model contains non-finite parameters");
  if (coordinator.parameter_version != model.version ||
      coordinator.schema_fingerprint != schema.fingerprint())
    throw std::invalid_argument(
        "Training checkpoint coordinator does not match the model source version and schema");
  if (stage.stage_id.empty() || stage.configuration_fingerprint.empty())
    throw std::invalid_argument("Training checkpoint requires stage identity metadata");
}

/// Reject incomplete contributor metadata before opening the output file.
void validateContributorMetadata(const CheckpointContributorMetadata& metadata,
                                 const char* role)
{
  if (metadata.contributor_id.empty() || metadata.configuration_fingerprint.empty() ||
      metadata.state_fingerprint.empty())
    throw std::invalid_argument(std::string("Training checkpoint ") + role +
                                " contributor metadata is incomplete");
  if (metadata.format_version[0] < 1)
    throw std::invalid_argument(std::string("Training checkpoint ") + role +
                                " contributor format version is invalid");
}

/// Persist one contributor's fixed envelope metadata in its already-open group.
void writeContributorMetadata(hdf_archive& archive,
                              const CheckpointContributorMetadata& metadata)
{
  const std::vector<int> version(metadata.format_version.begin(), metadata.format_version.end());
  archive.write(metadata.contributor_id, "contributor_id");
  archive.write(version, "format_version");
  archive.write(metadata.configuration_fingerprint, "configuration_fingerprint");
  archive.write(metadata.state_fingerprint, "state_fingerprint");
}

/// Read and shape-check one contributor's fixed envelope metadata.
CheckpointContributorMetadata readContributorMetadata(hdf_archive& archive,
                                                       const char* role)
{
  CheckpointContributorMetadata metadata;
  std::vector<int> version;
  archive.read(metadata.contributor_id, "contributor_id");
  archive.read(version, "format_version");
  archive.read(metadata.configuration_fingerprint, "configuration_fingerprint");
  archive.read(metadata.state_fingerprint, "state_fingerprint");
  if (version.size() != metadata.format_version.size())
    throw std::runtime_error(std::string("Training checkpoint ") + role +
                             " contributor version has invalid shape");
  std::copy(version.begin(), version.end(), metadata.format_version.begin());
  validateContributorMetadata(metadata, role);
  return metadata;
}

/// Require the saved role identity and configuration expected by the live owner.
void validateContributorIdentity(const CheckpointContributorMetadata& saved,
                                 const CheckpointContributorMetadata& expected,
                                 const char* role)
{
  if (saved.contributor_id != expected.contributor_id)
    throw std::runtime_error(std::string("Training checkpoint ") + role +
                             " contributor identity mismatch");
  if (saved.format_version[0] != expected.format_version[0])
    throw std::runtime_error(std::string("Training checkpoint ") + role +
                             " contributor major version mismatch");
  if (saved.configuration_fingerprint != expected.configuration_fingerprint)
    throw std::runtime_error(std::string("Training checkpoint ") + role +
                             " contributor configuration mismatch");
}

/// Construct a unique same-directory sibling owned by this save attempt.
std::filesystem::path temporaryPathFor(const std::filesystem::path& destination)
{
  static std::atomic<std::uint64_t> sequence{0};
  const std::uint64_t ordinal  = sequence.fetch_add(1, std::memory_order_relaxed);
  const auto clock_tick = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto process_salt = reinterpret_cast<std::uintptr_t>(&sequence);
  return destination.parent_path() /
      (destination.filename().string() + ".tmp." + std::to_string(process_salt) + "." +
       std::to_string(clock_tick) + "." + std::to_string(ordinal));
}

/// Install prepared coordinator and stage state using nonthrowing swaps.
void installDriverState(TrainingIterationState& coordinator,
                        TrainingIterationState& prepared_coordinator,
                        TrainingStageState& stage,
                        TrainingStageState& prepared_stage) noexcept
{
  coordinator.completed_iterations = prepared_coordinator.completed_iterations;
  coordinator.parameter_version    = prepared_coordinator.parameter_version;
  coordinator.schema_fingerprint.swap(prepared_coordinator.schema_fingerprint);
  stage.stage_id.swap(prepared_stage.stage_id);
  stage.stage_ordinal              = prepared_stage.stage_ordinal;
  stage.completed_stage_iterations = prepared_stage.completed_stage_iterations;
  stage.configuration_fingerprint.swap(prepared_stage.configuration_fingerprint);
}

} // namespace

void TrainingCheckpoint::saveAtomic(
    const std::filesystem::path& destination,
    const StructuredParameterProvider& provider,
    const TrainingIterationState& coordinator_state,
    const TrainingStageState& stage_state,
    const OptimizerCheckpointContributor* optimizer,
    const SamplerCheckpointContributor* sampler)
{
  if (destination.empty())
    throw std::invalid_argument("Training checkpoint destination must not be empty");

  const StructuredParameterSchema& schema = provider.parameterSchema();
  const StructuredParameterSnapshot model = provider.snapshotParameters();
  validateSaveState(schema, model, coordinator_state, stage_state);

  CheckpointContributorMetadata optimizer_metadata;
  CheckpointContributorMetadata sampler_metadata;
  if (optimizer)
  {
    optimizer_metadata = optimizer->checkpointMetadata();
    validateContributorMetadata(optimizer_metadata, "optimizer");
  }
  if (sampler)
  {
    sampler_metadata = sampler->checkpointMetadata();
    validateContributorMetadata(sampler_metadata, "sampler");
  }

  const std::filesystem::path temporary = temporaryPathFor(destination);
  std::error_code cleanup_error;
  bool owns_temporary = false;
  try
  {
    hdf_archive archive;
    if (!archive.create(temporary, H5F_ACC_EXCL))
      throw std::runtime_error("Unable to create training checkpoint temporary " +
                               temporary.string());
    owns_temporary = true;

    archive.push(ROOT_GROUP);
    const std::vector<int> format_version(FORMAT_VERSION.begin(), FORMAT_VERSION.end());
    const std::vector<int> contributor_presence{optimizer ? 1 : 0, sampler ? 1 : 0};
    archive.write(format_version, "format_version");
    archive.write(contributor_presence, "contributor_presence");

    archive.push("model");
    archive.write(schema.providerId(), "provider_id");
    archive.write(schema.fingerprint(), "schema_fingerprint");
    archive.write(checkedPersistedSize(model.version), "source_parameter_version");
    archive.write(checkedPersistedSize(model.values.size()), "parameter_count");
    archive.write(model.values, "values");
    archive.write(parameterFingerprint(model.values), "values_fingerprint");
    archive.pop();

    archive.push("coordinator");
    archive.write(checkedPersistedSize(coordinator_state.completed_iterations),
                  "completed_iterations");
    archive.write(checkedPersistedSize(coordinator_state.parameter_version),
                  "source_parameter_version");
    archive.pop();

    archive.push("stage");
    archive.write(stage_state.stage_id, "stage_id");
    archive.write(checkedPersistedSize(stage_state.stage_ordinal), "stage_ordinal");
    archive.write(checkedPersistedSize(stage_state.completed_stage_iterations),
                  "completed_stage_iterations");
    archive.write(stage_state.configuration_fingerprint, "configuration_fingerprint");
    archive.pop();

    if (optimizer)
    {
      archive.push("optimizer");
      writeContributorMetadata(archive, optimizer_metadata);
      optimizer->writeCheckpointPayload(archive);
      archive.pop();
    }
    if (sampler)
    {
      archive.push("sampler");
      writeContributorMetadata(archive, sampler_metadata);
      sampler->writeCheckpointPayload(archive);
      archive.pop();
    }

    // The cookie is deliberately the final dataset written. A reader rejects any
    // temporary or damaged file that did not reach this post-payload boundary.
    archive.write(std::string(COMPLETION_COOKIE), "completion_cookie");
    archive.pop();
    archive.flush();
    archive.close();

    std::error_code rename_error;
    std::filesystem::rename(temporary, destination, rename_error);
    if (rename_error)
      throw std::runtime_error("Unable to publish training checkpoint " +
                               destination.string() + ": " + rename_error.message());
    owns_temporary = false;
  }
  catch (...)
  {
    if (owns_temporary)
      std::filesystem::remove(temporary, cleanup_error);
    throw;
  }
}

void TrainingCheckpoint::restore(const std::filesystem::path& source,
                                 StructuredParameterProvider& provider,
                                 TrainingIterationState& coordinator_state,
                                 TrainingStageState& stage_state,
                                 OptimizerCheckpointContributor* optimizer,
                                 SamplerCheckpointContributor* sampler)
{
  const StructuredParameterSchema& schema = provider.parameterSchema();
  const StructuredParameterSnapshot runtime_model = provider.snapshotParameters();
  if (runtime_model.schema_fingerprint != schema.fingerprint() ||
      runtime_model.values.size() != schema.parameterCount())
    throw std::logic_error("Training checkpoint target provider returned an invalid snapshot");

  hdf_archive archive;
  if (!archive.open(source, H5F_ACC_RDONLY))
    throw std::runtime_error("Unable to open training checkpoint " + source.string());
  if (!archive.is_group(ROOT_GROUP))
    throw std::runtime_error("Training checkpoint is missing its root group");
  archive.push(ROOT_GROUP, false);

  std::vector<int> format_version;
  std::vector<int> contributor_presence;
  std::string completion_cookie;
  archive.read(format_version, "format_version");
  archive.read(contributor_presence, "contributor_presence");
  archive.read(completion_cookie, "completion_cookie");
  if (format_version.size() != FORMAT_VERSION.size() ||
      format_version[0] != FORMAT_VERSION[0])
    throw std::runtime_error("Training checkpoint has an incompatible format version");
  if (contributor_presence.size() != 2 ||
      (contributor_presence[0] != 0 && contributor_presence[0] != 1) ||
      (contributor_presence[1] != 0 && contributor_presence[1] != 1))
    throw std::runtime_error("Training checkpoint contributor presence has invalid shape or values");
  if (completion_cookie != COMPLETION_COOKIE)
    throw std::runtime_error("Training checkpoint is incomplete");
  if ((contributor_presence[0] != 0) != (optimizer != nullptr) ||
      (contributor_presence[1] != 0) != (sampler != nullptr))
    throw std::runtime_error("Training checkpoint contributor presence does not match restore request");

  archive.push("model", false);
  std::string saved_provider_id;
  std::string saved_schema_fingerprint;
  std::string saved_values_fingerprint;
  std::uint64_t saved_model_version = 0;
  std::uint64_t parameter_count     = 0;
  std::vector<double> saved_values;
  archive.read(saved_provider_id, "provider_id");
  archive.read(saved_schema_fingerprint, "schema_fingerprint");
  archive.read(saved_model_version, "source_parameter_version");
  archive.read(parameter_count, "parameter_count");
  archive.read(saved_values, "values");
  archive.read(saved_values_fingerprint, "values_fingerprint");
  archive.pop();
  if (saved_provider_id != schema.providerId())
    throw std::runtime_error("Training checkpoint model provider identity mismatch");
  if (saved_schema_fingerprint != schema.fingerprint())
    throw std::runtime_error("Training checkpoint model schema mismatch");
  if (checkedRuntimeSize(parameter_count, "parameter count") != schema.parameterCount() ||
      saved_values.size() != schema.parameterCount())
    throw std::runtime_error("Training checkpoint model parameter count mismatch");
  if (!std::all_of(saved_values.begin(), saved_values.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::runtime_error("Training checkpoint model contains non-finite parameters");
  if (saved_values_fingerprint != parameterFingerprint(saved_values))
    throw std::runtime_error("Training checkpoint model values fingerprint mismatch");

  archive.push("coordinator", false);
  std::uint64_t completed_iterations       = 0;
  std::uint64_t coordinator_source_version = 0;
  archive.read(completed_iterations, "completed_iterations");
  archive.read(coordinator_source_version, "source_parameter_version");
  archive.pop();
  if (coordinator_source_version != saved_model_version)
    throw std::runtime_error(
        "Training checkpoint model and coordinator source versions disagree");

  archive.push("stage", false);
  TrainingStageState prepared_stage;
  std::uint64_t stage_ordinal = 0;
  std::uint64_t completed_stage_iterations = 0;
  archive.read(prepared_stage.stage_id, "stage_id");
  archive.read(stage_ordinal, "stage_ordinal");
  archive.read(completed_stage_iterations, "completed_stage_iterations");
  archive.read(prepared_stage.configuration_fingerprint, "configuration_fingerprint");
  archive.pop();
  prepared_stage.stage_ordinal = checkedRuntimeSize(stage_ordinal, "stage ordinal");
  prepared_stage.completed_stage_iterations =
      checkedRuntimeSize(completed_stage_iterations, "completed stage iterations");
  if (prepared_stage.stage_id != stage_state.stage_id ||
      prepared_stage.configuration_fingerprint != stage_state.configuration_fingerprint)
    throw std::runtime_error("Training checkpoint stage identity or configuration mismatch");

  // Both contributors prepare owning state while the archive is open, still
  // before any live model, driver, optimizer, or sampler state can change.
  std::unique_ptr<OptimizerCheckpointContributor::PreparedRestore> prepared_optimizer;
  std::unique_ptr<SamplerCheckpointContributor::PreparedRestore> prepared_sampler;
  if (optimizer)
  {
    archive.push("optimizer", false);
    const CheckpointContributorMetadata saved =
        readContributorMetadata(archive, "optimizer");
    validateContributorIdentity(saved, optimizer->checkpointMetadata(), "optimizer");
    prepared_optimizer = optimizer->prepareCheckpointRestore(archive, saved);
    archive.pop();
    if (!prepared_optimizer)
      throw std::runtime_error("Training checkpoint optimizer returned no prepared restore");
  }
  if (sampler)
  {
    archive.push("sampler", false);
    const CheckpointContributorMetadata saved = readContributorMetadata(archive, "sampler");
    validateContributorIdentity(saved, sampler->checkpointMetadata(), "sampler");
    prepared_sampler = sampler->prepareCheckpointRestore(archive, saved);
    archive.pop();
    if (!prepared_sampler)
      throw std::runtime_error("Training checkpoint sampler returned no prepared restore");
  }
  archive.pop();
  archive.close();

  StructuredParameterSnapshot restored_model{schema.fingerprint(), runtime_model.version,
                                               std::move(saved_values)};
  TrainingIterationState prepared_coordinator{
      checkedRuntimeSize(completed_iterations, "completed iterations"),
      runtime_model.version, schema.fingerprint()};
  const std::size_t committed_version =
      provider.publishParameters(restored_model, runtime_model.version);
  prepared_coordinator.parameter_version = committed_version;
  installDriverState(coordinator_state, prepared_coordinator, stage_state, prepared_stage);
  if (prepared_optimizer)
    prepared_optimizer->commit();
  if (prepared_sampler)
    prepared_sampler->commit();
}

} // namespace qmcplusplus::wftrain
