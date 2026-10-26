//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file DistributedTrainingCheckpoint.cpp
 * @brief Manifest-coordinated distributed checkpoints for WFTrain.
 */

#include "QMCDrivers/WFTrain/DistributedTrainingCheckpoint.h"
#include "QMCDrivers/WFTrain/TrainingWaveFunctionBinding.h"

#include "Message/CommOperators.h"
#include "io/hdf/hdf_archive.h"
#include "QMCWaveFunctions/FrozenReferenceResidualWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerModelFile.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

constexpr char MANIFEST_ROOT[]       = "wftrain_distributed_manifest";
constexpr char MANIFEST_COOKIE[]     = "wftrain-distributed-manifest-complete-v1";
constexpr std::size_t TOPOLOGY_FIELDS = 5;
constexpr char INDEX_ROOT[] = "wftrain_latest_checkpoint";

std::string bundleRankSuffix(std::size_t rank)
{
  std::ostringstream text;
  text << ".rank_" << std::setw(6) << std::setfill('0') << rank << ".h5";
  return text.str();
}

std::filesystem::path resolveManifest(const std::filesystem::path& requested)
{
  // Resolve aliases before interpreting relative payloads; this also accepts directory trailing slashes.
  const auto source = std::filesystem::canonical(requested);
  if (std::filesystem::is_directory(source))
  {
    const auto name = source.filename().string();
    const std::string suffix = ".checkpoint";
    if (name.size() <= suffix.size() || name.substr(name.size() - suffix.size()) != suffix)
      throw std::runtime_error("Training bundle directory must end in .checkpoint");
    return source / (name.substr(0, name.size() - suffix.size()) + ".manifest.h5");
  }
  hdf_archive input;
  if (!input.open(source, H5F_ACC_RDONLY))
    throw std::runtime_error("Unable to open training checkpoint " + source.string());
  if (!input.is_group(INDEX_ROOT))
    return source;
  input.push(INDEX_ROOT, false);
  std::vector<int> version;
  std::string relative, cookie;
  input.read(version, "format_version");
  input.read(relative, "manifest_relative_path");
  input.read(cookie, "completion_cookie");
  const std::filesystem::path path(relative);
  if (version != std::vector<int>{1, 0, 0} || cookie != "wftrain-latest-complete-v1" ||
      path.empty() || path.is_absolute() || path.lexically_normal() != path)
    throw std::runtime_error("Invalid training latest checkpoint index");
  for (const auto& component : path)
    if (component == ".." || component == ".")
      throw std::runtime_error("Unsafe training latest checkpoint relative path");
  return source.parent_path() / path;
}

/// Return a generation token created once on rank zero and shared verbatim.
std::string makeGenerationToken(Communicate& communicator)
{
  std::string token;
  if (communicator.rank() == 0)
  {
    static std::atomic<std::uint64_t> sequence{0};
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    std::ostringstream text;
    text << std::hex << tick << '-' << sequence.fetch_add(1, std::memory_order_relaxed);
    token = text.str();
  }
  communicator.bcast(token);
  return token;
}

/// Construct a fixed-width rank suffix so each process owns one deterministic shard.
std::string rankSuffix(std::size_t rank)
{
  std::ostringstream text;
  text << ".rank-" << std::setw(8) << std::setfill('0') << rank << ".h5";
  return text.str();
}

/// Construct a unique same-directory temporary used for manifest publication.
std::filesystem::path manifestTemporary(const std::filesystem::path& destination)
{
  static std::atomic<std::uint64_t> sequence{0};
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  return destination.parent_path() /
      (destination.filename().string() + ".tmp." + std::to_string(tick) + "." +
       std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
}

/// Convert one size without accepting truncation in persistent storage.
std::uint64_t persistedSize(std::size_t value)
{
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t))
    if (value > std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("Distributed training checkpoint size exceeds uint64 storage");
  return static_cast<std::uint64_t>(value);
}

/// Convert one saved size without accepting truncation in the current executable.
std::size_t runtimeSize(std::uint64_t value, const char* field)
{
  if (value > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error(std::string("Distributed training checkpoint ") + field +
                             " exceeds this build's size range");
  return static_cast<std::size_t>(value);
}

/// Flatten the fixed local topology recorded in the published manifest.
std::array<std::uint64_t, TOPOLOGY_FIELDS> localTopologyRecord(
    const SamplerShardTopology& topology)
{
  std::size_t rng_state_total = 0;
  for (const std::size_t size : topology.rng_state_sizes)
  {
    if (size > std::numeric_limits<std::size_t>::max() - rng_state_total)
      throw std::overflow_error("Distributed training checkpoint RNG state total overflow");
    rng_state_total += size;
  }
  return {persistedSize(topology.walker_count), persistedSize(topology.particle_count),
          persistedSize(topology.crowd_count), persistedSize(topology.rng_state_sizes.size()),
          persistedSize(rng_state_total)};
}

/// Validate communicator identity and construct the noncollective local record.
std::array<std::uint64_t, TOPOLOGY_FIELDS> prepareLocalTopology(
    Communicate& communicator,
    const SamplerShardTopology& topology)
{
  if (topology.participant_count != static_cast<std::size_t>(communicator.size()) ||
      topology.rank != static_cast<std::size_t>(communicator.rank()))
    throw std::invalid_argument(
        "Distributed training checkpoint communicator and sampler topology disagree");
  return localTopologyRecord(topology);
}

/// Gather an already allocated fixed topology record in rank order.
void gatherTopology(
    Communicate& communicator,
    std::array<std::uint64_t, TOPOLOGY_FIELDS>& local,
    std::vector<std::uint64_t>& gathered)
{
  communicator.allgather(local.data(), gathered.data(), static_cast<int>(local.size()));
}

/// Turn local exceptions into a collective phase failure without skipping collectives.
void requireCollectiveSuccess(Communicate& communicator,
                              std::exception_ptr local_failure,
                              const char* phase)
{
  int failures = local_failure ? 1 : 0;
  communicator.allreduce(failures);
  if (failures == 0)
    return;
  if (local_failure)
    std::rethrow_exception(local_failure);
  throw std::runtime_error(std::string("Distributed training checkpoint ") + phase +
                           " failed on another rank");
}

/// Atomically publish the small manifest after every generation payload is durable.
void writeManifest(const std::filesystem::path& destination,
                   std::size_t participant_count,
                   const std::string& generation,
                   const std::string& root_filename,
                   const std::string& shard_prefix,
                   const std::vector<std::uint64_t>& topology)
{
  const std::filesystem::path temporary = manifestTemporary(destination);
  std::error_code cleanup_error;
  bool owns_temporary = false;
  try
  {
    hdf_archive archive;
    if (!archive.create(temporary, H5F_ACC_EXCL))
      throw std::runtime_error("Unable to create distributed checkpoint manifest temporary " +
                               temporary.string());
    owns_temporary = true;
    archive.push(MANIFEST_ROOT);
    const std::vector<int> version(DistributedTrainingCheckpoint::FORMAT_VERSION.begin(),
                                   DistributedTrainingCheckpoint::FORMAT_VERSION.end());
    archive.write(version, "format_version");
    archive.write(persistedSize(participant_count), "participant_count");
    archive.write(generation, "generation");
    archive.write(root_filename, "root_envelope_filename");
    archive.write(shard_prefix, "sampler_shard_prefix");
    archive.write(topology, "rank_topology");
    archive.write(std::string(MANIFEST_COOKIE), "completion_cookie");
    archive.pop();
    archive.flush();
    archive.close();

    std::error_code rename_error;
    std::filesystem::rename(temporary, destination, rename_error);
    if (rename_error)
      throw std::runtime_error("Unable to publish distributed checkpoint manifest " +
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

/// Validated immutable contents of one published distributed manifest.
struct Manifest
{
  std::size_t participant_count;
  std::string generation;
  std::string root_filename;
  std::string shard_prefix;
  std::vector<std::uint64_t> topology;
};

/// Fingerprint the manifest view so stale filesystem observations cannot diverge ranks.
std::uint64_t manifestFingerprint(const Manifest& manifest) noexcept
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  auto extend = [&hash](const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index < size; ++index)
    {
      hash ^= bytes[index];
      hash *= UINT64_C(1099511628211);
    }
  };
  const std::uint64_t participants = manifest.participant_count;
  extend(&participants, sizeof(participants));
  for (const std::string* field :
       {&manifest.generation, &manifest.root_filename, &manifest.shard_prefix})
  {
    const std::uint64_t size = field->size();
    extend(&size, sizeof(size));
    extend(field->data(), field->size());
  }
  if (!manifest.topology.empty())
    extend(manifest.topology.data(), manifest.topology.size() * sizeof(std::uint64_t));
  return hash;
}

/// Read and completely validate one manifest without changing live state.
Manifest readManifest(const std::filesystem::path& source)
{
  hdf_archive archive;
  if (!archive.open(source, H5F_ACC_RDONLY))
    throw std::runtime_error("Unable to open distributed checkpoint manifest " + source.string());
  if (!archive.is_group(MANIFEST_ROOT))
    throw std::runtime_error("Distributed checkpoint manifest is missing its root group");
  archive.push(MANIFEST_ROOT, false);

  std::vector<int> version;
  std::uint64_t participant_count = 0;
  Manifest manifest;
  std::string completion_cookie;
  archive.read(version, "format_version");
  archive.read(participant_count, "participant_count");
  archive.read(manifest.generation, "generation");
  archive.read(manifest.root_filename, "root_envelope_filename");
  archive.read(manifest.shard_prefix, "sampler_shard_prefix");
  archive.read(manifest.topology, "rank_topology");
  archive.read(completion_cookie, "completion_cookie");
  archive.pop();
  archive.close();

  if (version.size() != DistributedTrainingCheckpoint::FORMAT_VERSION.size() ||
      version[0] != DistributedTrainingCheckpoint::FORMAT_VERSION[0])
    throw std::runtime_error("Distributed checkpoint manifest format is incompatible");
  manifest.participant_count = runtimeSize(participant_count, "participant count");
  if (manifest.participant_count >
      std::numeric_limits<std::size_t>::max() / TOPOLOGY_FIELDS)
    throw std::runtime_error("Distributed checkpoint manifest topology extent overflows");
  const auto root_path = std::filesystem::path(manifest.root_filename);
  const auto shard_path = std::filesystem::path(manifest.shard_prefix);
  if (manifest.participant_count == 0 || manifest.generation.empty() ||
      manifest.root_filename.empty() || manifest.shard_prefix.empty() ||
      manifest.topology.size() != manifest.participant_count * TOPOLOGY_FIELDS ||
      root_path.is_absolute() || root_path.has_parent_path() ||
      shard_path.is_absolute() || shard_path.has_parent_path())
    throw std::runtime_error("Distributed checkpoint manifest is incomplete or malformed");
  if (completion_cookie != MANIFEST_COOKIE)
    throw std::runtime_error("Distributed checkpoint manifest publication is incomplete");
  return manifest;
}

} // namespace

void DistributedTrainingCheckpoint::saveIterationBundle(
    const std::filesystem::path& project_root, std::size_t iteration_index,
    Communicate& communicator, const StructuredParameterProvider& provider,
    const TrainingIterationState& coordinator_state, const TrainingStageState& stage_state,
    const OptimizerCheckpointContributor* optimizer, const SamplerCheckpointContributor& sampler,
    const SamplerShardTopology& topology)
{
  std::ostringstream identifier;
  identifier << project_root.filename().string() << ".i" << std::setw(7) << std::setfill('0') << iteration_index;
  const std::string root = identifier.str();
  const auto parent = project_root.parent_path() / (project_root.filename().string() + ".training");
  const auto destination = parent / (root + ".checkpoint");
  const auto parameters = parent / (root + ".parameters.h5");
  const auto generation = makeGenerationToken(communicator);
  const auto staging = parent / (root + ".checkpoint.tmp." + generation);
  const auto temporary_parameters = parent / (root + ".parameters.tmp." + generation + ".h5");
  std::array<std::uint64_t, TOPOLOGY_FIELDS> local{};
  std::vector<std::uint64_t> gathered;
  std::exception_ptr failure;
  try
  {
    if (project_root.empty())
      throw std::invalid_argument("Training project root is empty");
    local = prepareLocalTopology(communicator, topology);
    gathered.resize(TOPOLOGY_FIELDS * static_cast<std::size_t>(communicator.size()));
    if (std::filesystem::exists(destination) || std::filesystem::exists(parameters))
      throw std::runtime_error("Refusing to overwrite archived training iteration " + root);
    if (communicator.rank() == 0)
    {
      std::filesystem::create_directories(parent);
      if (!std::filesystem::create_directory(staging))
        throw std::runtime_error("Unable to create training bundle staging directory");
    }
  }
  catch (...) { failure = std::current_exception(); }
  requireCollectiveSuccess(communicator, failure, "bundle preflight");
  gatherTopology(communicator, local, gathered);
  failure = nullptr;
  try
  {
    TrainingCheckpoint::saveSamplerShardAtomic(staging / (root + bundleRankSuffix(communicator.rank())), sampler);
    if (communicator.rank() == 0)
    {
      TrainingCheckpoint::saveAtomic(staging / (root + ".state.h5"), provider, coordinator_state,
                                     stage_state, optimizer, nullptr);
      if (const auto* real = dynamic_cast<const PsiFormerWF*>(&provider))
        real->exportModel(temporary_parameters.string());
#ifdef QMC_COMPLEX
      else if (const auto* paired = dynamic_cast<const PsiFormerComplexWF*>(&provider))
        paired->exportModel(temporary_parameters.string());
#endif
      else if (const auto* parent = dynamic_cast<const FrozenReferenceResidualWF*>(&provider))
        parent->exportModel(temporary_parameters.string());
      else if (const auto* conventional = dynamic_cast<const TrainingWaveFunctionBinding*>(&provider))
        conventional->exportModel(temporary_parameters);
      else
        throw std::runtime_error("Training bundle model export requires a supported training binding");
      const auto saved = TrainingCheckpoint::readModelParameters(staging / (root + ".state.h5"),
                                                                provider.parameterSchema());
      const auto exported = [&]() {
        if (const auto* parent = dynamic_cast<const FrozenReferenceResidualWF*>(&provider))
          return parent->readModel(temporary_parameters.string()).values;
        if (const auto* conventional = dynamic_cast<const TrainingWaveFunctionBinding*>(&provider))
          return conventional->readModel(temporary_parameters).values;
        return psiformer::readPsiFormerModelFile(temporary_parameters.string()).parameters.values;
      }();
      if (saved.values != exported || provider.snapshotParameters().version != saved.version)
        throw std::runtime_error("Training model export and checkpoint generations disagree");
    }
  }
  catch (...) { failure = std::current_exception(); }
  requireCollectiveSuccess(communicator, failure, "bundle payload save");
  failure = nullptr;
  if (communicator.rank() == 0)
    try
    {
      writeManifest(staging / (root + ".manifest.h5"), communicator.size(), "iteration-bundle-v1:" + root,
                    root + ".state.h5", root, gathered);
      std::filesystem::rename(temporary_parameters, parameters);
      std::filesystem::rename(staging, destination);
      const auto index = project_root.string() + ".last_checkpoint.h5";
      const auto temporary = manifestTemporary(index);
      hdf_archive output;
      if (!output.create(temporary, H5F_ACC_EXCL))
        throw std::runtime_error("Unable to create latest checkpoint index");
      output.push(INDEX_ROOT);
      output.write(std::vector<int>{1, 0, 0}, "format_version");
      output.write((std::filesystem::path(project_root.filename().string() + ".training") /
          destination.filename() / (root + ".manifest.h5")).generic_string(), "manifest_relative_path");
      output.write(persistedSize(iteration_index), "iteration_index");
      output.write(std::string("wftrain-latest-complete-v1"), "completion_cookie");
      output.pop();
      output.flush();
      output.close();
      std::filesystem::rename(temporary, index);
    }
    catch (...) { failure = std::current_exception(); }
  requireCollectiveSuccess(communicator, failure, "bundle publication");
}

void DistributedTrainingCheckpoint::saveAtomic(
    const std::filesystem::path& manifest,
    Communicate& communicator,
    const StructuredParameterProvider& provider,
    const TrainingIterationState& coordinator_state,
    const TrainingStageState& stage_state,
    const OptimizerCheckpointContributor* optimizer,
    const SamplerCheckpointContributor& sampler,
    const SamplerShardTopology& topology)
{
  if (manifest.empty())
    throw std::invalid_argument("Distributed training checkpoint manifest must not be empty");

  std::array<std::uint64_t, TOPOLOGY_FIELDS> local_topology{};
  std::vector<std::uint64_t> rank_topology;
  std::exception_ptr topology_failure;
  try
  {
    local_topology = prepareLocalTopology(communicator, topology);
    rank_topology.resize(TOPOLOGY_FIELDS * static_cast<std::size_t>(communicator.size()));
  }
  catch (...)
  {
    topology_failure = std::current_exception();
  }
  requireCollectiveSuccess(communicator, topology_failure, "topology preflight");
  gatherTopology(communicator, local_topology, rank_topology);

  const std::string generation = makeGenerationToken(communicator);
  const std::string base = manifest.filename().string() + ".generation-" + generation;
  const std::string root_filename = base + ".root.h5";
  const std::string shard_prefix  = base + ".sampler";
  const auto root_path = manifest.parent_path() / root_filename;
  const auto shard_path = manifest.parent_path() /
      (shard_prefix + rankSuffix(static_cast<std::size_t>(communicator.rank())));

  std::exception_ptr shard_failure;
  try
  {
    TrainingCheckpoint::saveSamplerShardAtomic(shard_path, sampler);
  }
  catch (...)
  {
    shard_failure = std::current_exception();
  }
  requireCollectiveSuccess(communicator, shard_failure, "sampler-shard save");

  std::exception_ptr root_failure;
  if (communicator.rank() == 0)
    try
    {
      // The replicated O(P) model and optimizer state are deliberately stored once.
      TrainingCheckpoint::saveAtomic(root_path, provider, coordinator_state,
                                     stage_state, optimizer, nullptr);
    }
    catch (...)
    {
      root_failure = std::current_exception();
    }
  requireCollectiveSuccess(communicator, root_failure, "root-envelope save");

  std::exception_ptr manifest_failure;
  if (communicator.rank() == 0)
    try
    {
      writeManifest(manifest, static_cast<std::size_t>(communicator.size()), generation,
                    root_filename, shard_prefix, rank_topology);
    }
    catch (...)
    {
      manifest_failure = std::current_exception();
    }
  requireCollectiveSuccess(communicator, manifest_failure, "manifest publication");
}

void DistributedTrainingCheckpoint::restore(
    const std::filesystem::path& source,
    Communicate& communicator,
    StructuredParameterProvider& provider,
    TrainingIterationState& coordinator_state,
    TrainingStageState& stage_state,
    OptimizerCheckpointContributor* optimizer,
    SamplerCheckpointContributor& sampler,
    const SamplerShardTopology& topology)
{
  std::filesystem::path manifest_path;
  std::optional<Manifest> manifest;
  std::array<std::uint64_t, TOPOLOGY_FIELDS> local_topology{};
  std::vector<std::uint64_t> live_topology;
  std::vector<std::uint64_t> manifest_fingerprints;
  std::vector<std::uint64_t> published_versions;
  std::optional<TrainingCheckpoint::PreparedRestore> prepared_root;
  std::unique_ptr<SamplerCheckpointContributor::PreparedRestore> prepared_sampler;
  std::exception_ptr prepare_failure;
  try
  {
    manifest_path = resolveManifest(source);
    manifest.emplace(readManifest(manifest_path));
    local_topology = prepareLocalTopology(communicator, topology);
    live_topology.resize(TOPOLOGY_FIELDS * static_cast<std::size_t>(communicator.size()));
    manifest_fingerprints.resize(static_cast<std::size_t>(communicator.size()));
    published_versions.resize(static_cast<std::size_t>(communicator.size()));
  }
  catch (...)
  {
    prepare_failure = std::current_exception();
  }
  requireCollectiveSuccess(communicator, prepare_failure, "manifest/topology preflight");

  std::uint64_t local_manifest_fingerprint = manifestFingerprint(*manifest);
  communicator.allgather(&local_manifest_fingerprint, manifest_fingerprints.data(), 1);
  if (!std::all_of(manifest_fingerprints.begin(), manifest_fingerprints.end(),
                   [local_manifest_fingerprint](std::uint64_t value) {
                     return value == local_manifest_fingerprint;
                   }))
    throw std::runtime_error(
        "Distributed checkpoint ranks observed inconsistent manifest generations");

  gatherTopology(communicator, local_topology, live_topology);
  prepare_failure = nullptr;
  try
  {
    if (manifest->participant_count != static_cast<std::size_t>(communicator.size()) ||
        manifest->topology != live_topology)
      throw std::runtime_error(
          "Distributed checkpoint requires the identical MPI/crowd/walker/RNG topology");
    const auto root_path = manifest_path.parent_path() / manifest->root_filename;
    const auto shard_path = manifest_path.parent_path() /
        (manifest->shard_prefix + (manifest->generation.rfind("iteration-bundle-v1:", 0) == 0
          ? bundleRankSuffix(static_cast<std::size_t>(communicator.rank()))
          : rankSuffix(static_cast<std::size_t>(communicator.rank()))));
    prepared_root.emplace(TrainingCheckpoint::prepareRestore(
        root_path, provider, coordinator_state, stage_state, optimizer, nullptr));
    prepared_sampler = TrainingCheckpoint::prepareSamplerShardRestore(shard_path, sampler);
  }
  catch (...)
  {
    prepare_failure = std::current_exception();
  }
  requireCollectiveSuccess(communicator, prepare_failure, "restore preparation");

  // Complete-model bindings may use their own fallible collectives, generic
  // prepared actions have no-fail commit, and legacy providers remain one-phase.
  // Agree that schedule BEFORE any rank publishes; reuse cold allocated storage.
  std::uint64_t publication_role = static_cast<std::uint64_t>(prepared_root->publicationRole());
  communicator.allgather(&publication_role, published_versions.data(), 1);
  if (!std::all_of(published_versions.begin(), published_versions.end(),
                   [publication_role](std::uint64_t role) { return role == publication_role; }))
    throw std::runtime_error("Distributed checkpoint model publication role mismatch");

  std::size_t committed_version = 0;
  std::exception_ptr publication_failure;
  try
  {
    committed_version = prepared_root->publishModel();
  }
  catch (...)
  {
    publication_failure = std::current_exception();
  }
  int publication_failures = publication_failure ? 1 : 0;
  communicator.allreduce(publication_failures);
  if (publication_failures != 0)
  {
    // Legacy/complete-model publication can still fail after a peer publishes.
    // Prepared commit is no-fail; no path promises rollback after publication.
    if (publication_failure)
      std::rethrow_exception(publication_failure);
    throw std::runtime_error(
        "Distributed checkpoint model publication failed on another rank after this rank "
        "published; the process set must terminate rather than continue with mixed state");
  }

  std::uint64_t persisted_version = persistedSize(committed_version);
  communicator.allgather(&persisted_version, published_versions.data(), 1);
  for (const std::uint64_t version : published_versions)
    if (version != persisted_version)
      throw std::runtime_error(
          "Distributed checkpoint providers returned inconsistent committed versions");

  // Every operation below is contractually noexcept and occurs only after every
  // rank has prepared and published successfully.
  prepared_root->commitAfterPublication(committed_version);
  prepared_sampler->commit();
}

} // namespace qmcplusplus::wftrain
