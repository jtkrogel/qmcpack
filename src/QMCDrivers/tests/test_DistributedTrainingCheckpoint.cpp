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
#include <catch2/matchers/catch_matchers_string.hpp>

#include "Message/CommOperators.h"
#include "QMCDrivers/WFTrain/DistributedTrainingCheckpoint.h"
#include "QMCDrivers/WFTrain/FirstOrderOptimizer.h"
#include "io/hdf/hdf_archive.h"
#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerModelFile.h"
#include "../../QMCWaveFunctions/tests/psiformer_test_utils.h"

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
class DistributedToyProvider : public StructuredParameterProvider
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

protected:
  /// Same scalar replacement, with every expected validation already prepared.
  std::size_t installPrepared(double value, std::size_t expected_version) noexcept
  {
    if (version_ != expected_version)
      std::terminate();
    value_ = value;
    return ++version_;
  }

private:
  StructuredParameterSchema schema_;
  std::size_t version_ = 0;
  double value_ = 1.0;
};

/// Prepared checkpoint provider with no hidden model reentry while authority is held.
class PreparedCheckpointProvider final : public DistributedToyProvider,
                                         public PreparedParameterPublicationContributor
{
public:
  std::unique_ptr<PreparedParameterPublication> prepareParameterPublication(
      const StructuredParameterSnapshot& candidate, std::size_t expected_version) override
  {
    ++prepare_calls;
    if (fail_preparation)
      throw std::runtime_error("injected checkpoint model preparation failure");
    const auto current = snapshotParameters();
    validateTrainingCandidate(parameterSchema(), current, candidate);
    if (exclusive_ || current.version != expected_version)
      throw std::logic_error("Checkpoint model preparation lost exclusive version");
    auto prepared = std::make_unique<Action>(*this, candidate.values[0], expected_version);
    exclusive_ = true;
    return prepared;
  }

  std::size_t publishParameters(const StructuredParameterSnapshot& candidate, std::size_t version) override
  {
    ++legacy_calls;
    return DistributedToyProvider::publishParameters(candidate, version);
  }

  bool fail_preparation = false;
  std::size_t prepare_calls = 0;
  std::size_t commit_calls = 0;
  std::size_t discard_calls = 0;
  std::size_t legacy_calls = 0;

private:
  class Action final : public PreparedParameterPublication
  {
  public:
    Action(PreparedCheckpointProvider& owner, double value, std::size_t expected_version)
        : owner_(owner), value_(value), expected_version_(expected_version)
    {}
    ~Action() override
    {
      if (!committed_)
        ++owner_.discard_calls;
      owner_.exclusive_ = false;
    }
    std::size_t commit() noexcept override
    {
      if (committed_ || !owner_.exclusive_)
        std::terminate();
      const auto version = owner_.installPrepared(value_, expected_version_);
      ++owner_.commit_calls;
      committed_ = true;
      return version;
    }

  private:
    PreparedCheckpointProvider& owner_;
    const double value_;
    const std::size_t expected_version_;
    bool committed_ = false;
  };
  bool exclusive_ = false;
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

/// Exercise the existing file family with real Adam state and group-local control.
void checkPreparedCheckpointRestore(Communicate& communicator, int group_id, bool reject_first)
{
  const std::filesystem::path manifest = "wftrain_prepared_restore.g" + std::to_string(group_id) + ".h5";
  removeCheckpointFamily(manifest, communicator);
  const auto topology = toyTopology(communicator);
  FirstOrderOptimizerOptions options;
  options.learning_rates = {{"weight", 0.2 * (group_id + 1)}};
  DistributedToyProvider source_provider;
  FirstOrderOptimizer source_optimizer(source_provider.parameterSchema(), options);
  const auto original_source = source_provider.snapshotParameters();
  const std::vector<double> gradient{0.3 * (group_id + 1)};
  const ParameterGradientView objective{source_provider.parameterSchema().fingerprint(), original_source.version,
                                        ReductionDomain::GLOBAL, {gradient.data(), gradient.size()}};
  const auto source_version = completeParameterUpdate(
      source_provider, original_source, objective, source_optimizer, {});
  TrainingIterationState source_state{1, source_version, source_provider.parameterSchema().fingerprint()};
  TrainingStageState source_stage{"energy", 0, 1, "prepared-restore-v1"};
  DistributedToySampler source_sampler(communicator.rank(), communicator.size());
  source_sampler.advance();
  DistributedTrainingCheckpoint::saveAtomic(manifest, communicator, source_provider, source_state,
      source_stage, &source_optimizer, source_sampler, topology);

  PreparedCheckpointProvider target;
  FirstOrderOptimizer optimizer(target.parameterSchema(), options);
  DistributedToySampler sampler(communicator.rank(), communicator.size());
  sampler.advance();
  sampler.advance();
  TrainingIterationState state{99, 0, target.parameterSchema().fingerprint()};
  TrainingStageState stage{"energy", 7, 88, "prepared-restore-v1"};
  const auto old_model = target.snapshotParameters();
  const auto old_sampler = sampler.value();
  const auto old_first = optimizer.firstMoment();
  const auto old_second = optimizer.secondMoment();
  target.fail_preparation = reject_first && communicator.rank() == communicator.size() - 1;
  if (reject_first)
  {
    CHECK_THROWS(DistributedTrainingCheckpoint::restore(manifest, communicator, target, state, stage,
                                                       &optimizer, sampler, topology));
    CHECK(target.snapshotParameters().values == old_model.values);
    CHECK(target.snapshotParameters().version == old_model.version);
    CHECK(target.commit_calls == 0);
    CHECK(target.discard_calls == (target.fail_preparation ? 0 : 1));
    CHECK(target.legacy_calls == 0);
    CHECK(optimizer.firstMoment() == old_first);
    CHECK(optimizer.secondMoment() == old_second);
    CHECK(optimizer.acceptedUpdateCount() == 0);
    CHECK(sampler.value() == old_sampler);
    CHECK(state.completed_iterations == 99);
    CHECK(state.parameter_version == 0);
    CHECK(stage.stage_ordinal == 7);
    CHECK(stage.completed_stage_iterations == 88);
  }
  else
  {
    CHECK_NOTHROW(DistributedTrainingCheckpoint::restore(manifest, communicator, target, state, stage,
                                                         &optimizer, sampler, topology));
    CHECK(target.commit_calls == 1);
  }

  target.fail_preparation = false;
  CHECK_NOTHROW(DistributedTrainingCheckpoint::restore(manifest, communicator, target, state, stage,
                                                       &optimizer, sampler, topology));
  CHECK(target.snapshotParameters().values == source_provider.snapshotParameters().values);
  CHECK(target.snapshotParameters().version == (reject_first ? 1 : 2));
  CHECK(target.commit_calls == (reject_first ? 1 : 2));
  CHECK(target.legacy_calls == 0);
  CHECK(optimizer.firstMoment() == source_optimizer.firstMoment());
  CHECK(optimizer.secondMoment() == source_optimizer.secondMoment());
  CHECK(optimizer.acceptedUpdateCount() == source_optimizer.acceptedUpdateCount());
  CHECK(sampler.value() == source_sampler.value());
  CHECK(state.completed_iterations == source_state.completed_iterations);
  CHECK(state.parameter_version == target.snapshotParameters().version);
  CHECK(stage.stage_ordinal == source_stage.stage_ordinal);
  CHECK(stage.completed_stage_iterations == source_stage.completed_stage_iterations);

  if (communicator.size() > 1)
  {
    DistributedToyProvider legacy;
    StructuredParameterProvider& mixed = communicator.rank() == communicator.size() - 1
        ? static_cast<StructuredParameterProvider&>(legacy) : static_cast<StructuredParameterProvider&>(target);
    const auto before_model = mixed.snapshotParameters();
    const auto before_commits = target.commit_calls;
    const auto before_sampler = sampler.value();
    const auto before_first = optimizer.firstMoment();
    const auto before_second = optimizer.secondMoment();
    const auto before_state = state;
    CHECK_THROWS_WITH(DistributedTrainingCheckpoint::restore(manifest, communicator, mixed, state, stage,
                                                            &optimizer, sampler, topology),
                      Catch::Matchers::ContainsSubstring("publication role mismatch"));
    CHECK(mixed.snapshotParameters().values == before_model.values);
    CHECK(mixed.snapshotParameters().version == before_model.version);
    CHECK(target.commit_calls == before_commits);
    CHECK(optimizer.firstMoment() == before_first);
    CHECK(optimizer.secondMoment() == before_second);
    CHECK(sampler.value() == before_sampler);
    CHECK(state.completed_iterations == before_state.completed_iterations);
    CHECK(state.parameter_version == before_state.parameter_version);
  }
  removeCheckpointFamily(manifest, communicator);
}

} // namespace

TEST_CASE("Distributed checkpoint prepares model publication before restoring live state",
          "[drivers][training][checkpoint][prepared_restore][mpi]")
{
  checkPreparedCheckpointRestore(*OHMMS::Controller, 0, true);
}

TEST_CASE("Prepared checkpoint rejection and continuation remain subgroup local",
          "[drivers][training][checkpoint][prepared_restore][subgroup][mpi]")
{
  Communicate& world = *OHMMS::Controller;
  if (world.size() < 4)
    return;
  Communicate group(world, 2);
  checkPreparedCheckpointRestore(group, group.getGroupID(), group.getGroupID() == 0);
}

TEST_CASE("Distributed iteration bundles preserve immutable archives and latest resolution",
          "[drivers][training][checkpoint][mpi][iteration_bundle]")
{
  Communicate& communicator = *OHMMS::Controller;
  namespace fixture = qmcplusplus::testing::psiformer;
  const auto geometry = fixture::makeGeometry("lih");
  const SimulationCell cell;
  ParticleSet electrons(cell), ions(cell);
  electrons.create({2, 2});
  ions.create({1, 1});
  electrons.getSpeciesSet().addSpecies("u");
  electrons.getSpeciesSet().addSpecies("d");
  auto& species = ions.getSpeciesSet();
  species.addSpecies("Li");
  species.addSpecies("H");
  const int charge = species.addAttribute("charge");
  species(charge, 0) = 3;
  species(charge, 1) = 1;
  electrons.resetGroups();
  ions.resetGroups();
  for (int atom = 0; atom < 2; ++atom)
    for (int d = 0; d < 3; ++d)
      ions.R[atom][d] = geometry.nuclei[3 * atom + d];
  for (int e = 0; e < 4; ++e)
    for (int d = 0; d < 3; ++d)
      electrons.R[e][d] = geometry.electrons[3 * e + d];
  auto leaves = fixture::makeLayout(4, 2, true, 2, 4);
  for (auto& leaf : leaves)
    for (auto& extent : leaf.shape)
      if (extent == 256 || extent == 64)
        extent = 8;
  psiformer::InitializedPsiFormerParameters initialized;
  initialized.model_shape = {2, 2, 2, 2, 8, 2, 2};
  initialized.profile = psiformer::DEEPQMC_PSIFORMER_V1;
  initialized.values = fixture::makeParameters("lih", leaves);
  std::size_t offset = 0;
  for (const auto& leaf : leaves)
  {
    const auto end = offset + fixture::product(leaf.shape);
    initialized.layouts.push_back({leaf.module, leaf.name, leaf.shape, offset, end});
    offset = end;
  }
  PsiFormerWF provider("bundle", initialized, electrons, ions);
  provider.validateSystem(electrons, ions, "all_electron");
  std::string token;
  if (communicator.rank() == 0)
    token = std::to_string(::getpid());
  communicator.bcast(token);
  const std::filesystem::path project = "wftrain_bundle_test_" + token + ".s000";
  const auto training = std::filesystem::path(project.string() + ".training");
  const auto latest = std::filesystem::path(project.string() + ".last_checkpoint.h5");
  const auto root = project.string() + ".i0000000";
  const auto bundle = training / (root + ".checkpoint");
  const auto manifest = bundle / (root + ".manifest.h5");
  TrainingIterationState state{1, provider.snapshotParameters().version, provider.parameterSchema().fingerprint()};
  TrainingStageState stage{"energy", 0, 1, "bundle-test-v1"};
  DistributedToySampler sampler(communicator.rank(), communicator.size());
  const auto topology = toyTopology(communicator);
  DistributedTrainingCheckpoint::saveIterationBundle(project, 0, communicator, provider, state, stage,
                                                     nullptr, sampler, topology);
  CHECK(std::filesystem::exists(manifest));
  CHECK(std::filesystem::exists(bundle / (root + ".state.h5")));
  const auto exported = psiformer::readPsiFormerModelFile((training / (root + ".parameters.h5")).string());
  CHECK(exported.parameters.values == provider.snapshotParameters().values);
  CHECK_THROWS(DistributedTrainingCheckpoint::saveIterationBundle(project, 0, communicator, provider,
      state, stage, nullptr, sampler, topology));
  // Parameter-only exports are deliberately unnecessary for an exact restart.
  if (communicator.rank() == 0)
  {
    std::filesystem::remove(training / (root + ".parameters.h5"));
    std::filesystem::create_symlink(std::filesystem::absolute(latest), training / "latest_alias.h5");
  }
  communicator.barrier();
  const auto saved = provider.snapshotParameters().values;
  for (const auto& source : {latest, bundle, manifest, bundle / "", training / "latest_alias.h5"})
  {
    auto changed = provider.snapshotParameters();
    changed.values[0] += .125;
    provider.publishParameters(changed, changed.version);
    sampler.advance();
    state.completed_iterations = 5;
    DistributedTrainingCheckpoint::restore(source, communicator, provider, state, stage, nullptr, sampler, topology);
    CHECK(provider.snapshotParameters().values == saved);
    CHECK(state.completed_iterations == 1);
    CHECK(sampler.value() == static_cast<std::uint64_t>(100 + communicator.rank()));
  }
  // One failed local shard prevents latest advancement on every rank.
  state.completed_iterations = 2;
  state.parameter_version = provider.snapshotParameters().version;
  sampler.failWrites(communicator.rank() == communicator.size() - 1);
  CHECK_THROWS(DistributedTrainingCheckpoint::saveIterationBundle(project, 1, communicator, provider,
      state, stage, nullptr, sampler, topology));
  sampler.failWrites(false);
  DistributedTrainingCheckpoint::restore(latest, communicator, provider, state, stage, nullptr, sampler, topology);
  CHECK(state.completed_iterations == 1);
  auto changed_topology = topology;
  ++changed_topology.walker_count;
  CHECK_THROWS(DistributedTrainingCheckpoint::restore(latest, communicator, provider, state, stage,
      nullptr, sampler, changed_topology));
  const auto unsafe = std::filesystem::path(project.string() + ".unsafe_index.h5");
  if (communicator.rank() == 0)
  {
    hdf_archive index;
    REQUIRE(index.create(unsafe));
    index.push("wftrain_latest_checkpoint");
    index.write(std::vector<int>{1, 0, 0}, "format_version");
    index.write(std::string("../outside.manifest.h5"), "manifest_relative_path");
    index.write(std::string("wftrain-latest-complete-v1"), "completion_cookie");
  }
  communicator.barrier();
  CHECK_THROWS(DistributedTrainingCheckpoint::restore(unsafe, communicator, provider, state, stage,
      nullptr, sampler, topology));
  const auto shard = bundle / (root + ".rank_000000.h5");
  if (communicator.rank() == 0)
    std::filesystem::rename(shard, shard.string() + ".missing");
  communicator.barrier();
  CHECK_THROWS(DistributedTrainingCheckpoint::restore(latest, communicator, provider, state, stage,
      nullptr, sampler, topology));
  communicator.barrier();
  const auto envelope = bundle / (root + ".state.h5");
  if (communicator.rank() == 0)
  {
    std::filesystem::rename(shard.string() + ".missing", shard);
    std::filesystem::rename(envelope, envelope.string() + ".missing");
  }
  communicator.barrier();
  CHECK_THROWS(DistributedTrainingCheckpoint::restore(latest, communicator, provider, state, stage,
      nullptr, sampler, topology));
  communicator.barrier();
  if (communicator.rank() == 0)
  {
    std::filesystem::remove_all(training);
    std::filesystem::remove(latest);
    std::filesystem::remove(unsafe);
  }
  communicator.barrier();
}

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
