//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_OrbitalPretraining.cpp
 * @brief Deterministic tests for bounded orbital-pretraining orchestration.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "QMCDrivers/WFTrain/FirstOrderOptimizer.h"
#include "QMCDrivers/WFTrain/OrbitalPretraining.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Small versioned model used to test objective and publication transactions.
class OrbitalToyProvider final : public StructuredParameterProvider
{
public:
  OrbitalToyProvider()
      : schema_("orbital-pretraining/toy",
                {{"weights", {2}, 0, 2, ParameterScalarDomain::REAL64, true, "network"},
                 {"frozen", {1}, 2, 1, ParameterScalarDomain::REAL64, false, "fixed"}})
  {}

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }

  StructuredParameterSnapshot snapshotParameters() const override
  {
    return {schema_.fingerprint(), version_, values_};
  }

  std::size_t publishParameters(const StructuredParameterSnapshot& candidate,
                                std::size_t expected_version) override
  {
    if (reject_next_)
    {
      reject_next_ = false;
      throw std::runtime_error("synthetic orbital publication failure");
    }
    if (expected_version != version_ || candidate.version != version_ ||
        candidate.schema_fingerprint != schema_.fingerprint() ||
        candidate.values.size() != values_.size())
      throw std::invalid_argument("invalid orbital toy publication");
    values_ = candidate.values;
    return ++version_;
  }

  void rejectNextPublish() noexcept { reject_next_ = true; }

private:
  StructuredParameterSchema schema_;
  std::vector<double> values_{2.0, -1.0, 7.0};
  std::size_t version_ = 0;
  bool reject_next_ = false;
};

/// Stream a configurable number of identical convex sample objectives.
class ConvexOrbitalProducer final : public OrbitalPretrainingProducer
{
public:
  explicit ConvexOrbitalProducer(std::size_t samples) : samples_(samples) {}

  TrainingCapabilities capabilities() const noexcept override
  {
    return {TrainingCapability::REAL_PARAMETERS, TrainingCapability::ORBITAL_MSE_VJP};
  }

  void accumulate(const StructuredParameterSnapshot& parameters,
                  OrbitalPretrainingAccumulator& accumulator) override
  {
    const double loss = parameters.values[0] * parameters.values[0] +
        parameters.values[1] * parameters.values[1];
    const std::vector<double> gradient{
        2.0 * parameters.values[0], 2.0 * parameters.values[1], 123.0};
    for (std::size_t sample = 0; sample < samples_; ++sample)
      accumulator.addSample(loss, parameters.version,
                            {gradient.data(), gradient.size()});
  }

private:
  std::size_t samples_;
};

/// Minimal bounded derivative operator used to exercise the real energy coordinator.
class HandoffEnergyOperator final : public StreamingDerivativeOperator
{
public:
  HandoffEnergyOperator(const StructuredParameterSchema& schema, std::size_t version)
      : schema_(schema), version_(version), plan_(schema, version, 2)
  {}

  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities result;
    result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
        derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_VJP);
    result.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::TRANSPOSE);
    result.parameter_scalar_domain = ParameterScalarDomain::REAL64;
    result.result_scalar_domain = ParameterScalarDomain::COMPLEX128;
    result.reduction_domain = ReductionDomain::CROWD_LOCAL;
    result.execution_domain = DerivativeExecutionDomain::HOST;
    result.local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
    result.maximum_vjp_channels = 3;
    result.maximum_parameter_chunk_size = 2;
    result.maximum_sample_tile_size = 1;
    result.block_streaming = true;
    return result;
  }

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }
  std::size_t parameterVersion() const noexcept override { return version_; }
  std::size_t batchOrdinal() const noexcept override { return 0; }
  std::size_t sampleOffset() const noexcept override { return 0; }
  std::size_t sampleCount() const noexcept override { return 1; }
  const ParameterChunkPlan& parameterChunkPlan() const noexcept override { return plan_; }

protected:
  void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                    DerivativeAdjoint,
                    ParameterReductionSink& sink) const override
  {
    for (const ParameterChunkDescriptor& chunk : plan_.chunks())
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
      {
        std::vector<DerivativeValue> values(chunk.count);
        for (std::size_t local = 0; local < chunk.count; ++local)
        {
          const std::size_t parameter = chunk.parameter_offset + local;
          const double derivative =
              channels[channel].product == DerivativeProduct::SCORE_VJP
              ? 0.1 * static_cast<double>(parameter + 1)
              : 0.01 * static_cast<double>(parameter + 1);
          values[local] = derivative * channels[channel].coefficients.values[0];
        }
        sink.add(channel, {chunk, {values.data(), values.size()}});
      }
  }

  void evaluateScoreJVP(const StructuredParameterVectorConstView&,
                        SampleProductSink&) const override
  {
    throw std::logic_error("HandoffEnergyOperator does not implement score JVP");
  }

private:
  const StructuredParameterSchema& schema_;
  std::size_t version_;
  ParameterChunkPlan plan_;
};

/// Produce one complete energy-gradient sample after orbital pretraining handoff.
class HandoffEnergyProducer final : public GradientProducer
{
public:
  explicit HandoffEnergyProducer(const StructuredParameterSchema& schema) : schema_(schema) {}

  TrainingCapabilities capabilities() const noexcept override
  {
    return {TrainingCapability::REAL_PARAMETERS, TrainingCapability::SCORE_VJP,
            TrainingCapability::LOCAL_ENERGY_VJP};
  }

  void accumulate(const StructuredParameterSnapshot& parameters,
                  EnergyGradientAccumulator& accumulator) override
  {
    HandoffEnergyOperator derivative(schema_, parameters.version);
    const std::vector<DerivativeReal> weights{1.0};
    const std::vector<DerivativeValue> energies{-1.0};
    accumulateEnergyGradientBatch(
        derivative, {weights.data(), weights.size()},
        {energies.data(), energies.size()},
        localEnergyTermBit(LocalEnergyTerm::KINETIC), accumulator);
  }

private:
  const StructuredParameterSchema& schema_;
};

/// Count post-publication cache notifications without throwing.
class CountingObserver final : public ParameterUpdateObserver
{
public:
  void parametersPublished(std::size_t version) noexcept override
  {
    ++calls;
    last_version = version;
  }

  std::size_t calls = 0;
  std::size_t last_version = 0;
};

/// Record whether central gradient validation admitted a proposal call.
class BlindUpdater final : public TrainingUpdateRule
{
public:
  StructuredParameterSnapshot propose(const StructuredParameterSchema&,
                                      const StructuredParameterSnapshot& parameters,
                                      ParameterGradientView) override
  {
    ++calls;
    return parameters;
  }

  std::size_t calls = 0;
};

/// Remove one test-owned checkpoint at scope exit.
class CheckpointCleanup
{
public:
  explicit CheckpointCleanup(std::filesystem::path path) : path_(std::move(path)) {}

  ~CheckpointCleanup()
  {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

private:
  std::filesystem::path path_;
};

/// Return a one-group optimizer configuration suitable for the toy schema.
FirstOrderOptimizerOptions optimizerOptions(FirstOrderMethod method,
                                            double learning_rate)
{
  FirstOrderOptimizerOptions options;
  options.method = method;
  options.learning_rates = {{"network", learning_rate}};
  options.adam_beta1 = 0.5;
  options.adam_beta2 = 0.75;
  options.epsilon = 1.0e-8;
  return options;
}

/// Construct common coordinator state for a fresh provider.
TrainingIterationState initialIterationState(const OrbitalToyProvider& provider)
{
  return {0, provider.snapshotParameters().version,
          provider.parameterSchema().fingerprint()};
}

} // namespace

TEST_CASE("Orbital pretraining accumulator merges uneven bounded contributions",
          "[drivers][training][orbital-pretraining]")
{
  OrbitalToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  OrbitalPretrainingAccumulator empty(provider.parameterSchema(), parameters.version,
                                      17, 19);
  OrbitalPretrainingAccumulator two(provider.parameterSchema(), parameters.version,
                                    17, 19);
  const std::vector<double> first{1.0, 2.0, 3.0};
  const std::vector<double> second{3.0, 4.0, 5.0};
  two.addSample(2.0, parameters.version, {first.data(), first.size()});
  two.addSample(6.0, parameters.version, {second.data(), second.size()});

  const std::size_t bytes = empty.retainedBytes();
  const std::size_t storage = empty.storageFingerprint();
  empty.merge(two);
  CHECK(empty.retainedBytes() == bytes);
  CHECK(empty.storageFingerprint() == storage);
  CHECK(empty.sampleCount() == 2);
  empty.completeSingleParticipantReduction();
  const OrbitalPretrainingResult result = empty.finalize();
  CHECK(result.sample_count == 2);
  CHECK(result.mean_loss == Catch::Approx(4.0));
  CHECK(result.gradient == std::vector<double>{2.0, 3.0, 4.0});
  CHECK(result.parameterGradient().schema_fingerprint ==
        provider.parameterSchema().fingerprint());
  CHECK(result.parameterGradient().reduction_domain == ReductionDomain::GLOBAL);

  OrbitalPretrainingAccumulator wrong_target(provider.parameterSchema(), parameters.version,
                                             23, 19);
  CHECK_THROWS_WITH(two.merge(wrong_target),
                    Catch::Matchers::ContainsSubstring("metadata mismatch"));
  OrbitalPretrainingAccumulator globally_empty(provider.parameterSchema(), parameters.version,
                                               17, 19);
  CHECK_THROWS_WITH(globally_empty.completeSingleParticipantReduction(),
                    Catch::Matchers::ContainsSubstring("empty"));
}

TEST_CASE("Orbital coordinator reuses first-order updates and lowers a convex loss",
          "[drivers][training][orbital-pretraining][optimizer]")
{
  OrbitalToyProvider provider;
  OrbitalPretrainingCoordinator coordinator(0x1234);
  TrainingIterationState state = initialIterationState(provider);
  TrainingStageState stage = coordinator.makeStageState(2);
  FirstOrderOptimizer optimizer(provider.parameterSchema(),
                                optimizerOptions(FirstOrderMethod::SGD, 0.1));
  ConvexOrbitalProducer producer(3);
  CountingObserver observer;

  std::vector<double> losses;
  for (std::size_t step = 0; step < 3; ++step)
  {
    const OrbitalPretrainingIterationResult result = coordinator.runIteration(
        provider, producer, optimizer, state, stage, &observer);
    losses.push_back(result.objective.mean_loss);
    CHECK(result.objective.sample_count == 3);
    CHECK(result.parameter_version == step + 1);
    CHECK(result.completed_stage_iteration == step + 1);
  }
  CHECK(losses[1] < losses[0]);
  CHECK(losses[2] < losses[1]);
  CHECK(provider.snapshotParameters().values[2] == 7.0);
  CHECK(optimizer.acceptedUpdateCount() == 3);
  CHECK(observer.calls == 3);
  CHECK(observer.last_version == 3);
}

TEST_CASE("Orbital publication failure preserves optimizer and coordinator state",
          "[drivers][training][orbital-pretraining][atomic]")
{
  OrbitalToyProvider provider;
  OrbitalPretrainingCoordinator coordinator(0x2345);
  TrainingIterationState state = initialIterationState(provider);
  TrainingStageState stage = coordinator.makeStageState();
  FirstOrderOptimizer optimizer(provider.parameterSchema(),
                                optimizerOptions(FirstOrderMethod::ADAM, 0.05));
  ConvexOrbitalProducer producer(1);
  CountingObserver observer;
  const StructuredParameterSnapshot before = provider.snapshotParameters();
  provider.rejectNextPublish();

  CHECK_THROWS_WITH(coordinator.runIteration(provider, producer, optimizer, state,
                                             stage, &observer),
                    Catch::Matchers::ContainsSubstring("synthetic orbital"));
  CHECK(provider.snapshotParameters().values == before.values);
  CHECK(provider.snapshotParameters().version == before.version);
  CHECK(optimizer.acceptedUpdateCount() == 0);
  CHECK_FALSE(optimizer.hasLiveProposal());
  CHECK(std::all_of(optimizer.firstMoment().begin(), optimizer.firstMoment().end(),
                    [](double value) { return value == 0.0; }));
  CHECK(std::all_of(optimizer.secondMoment().begin(), optimizer.secondMoment().end(),
                    [](double value) { return value == 0.0; }));
  CHECK(state.completed_iterations == 0);
  CHECK(stage.completed_stage_iterations == 0);
  CHECK(observer.calls == 0);
}

TEST_CASE("Parameter update transaction rejects malformed objective-neutral views",
          "[drivers][training][orbital-pretraining][validation]")
{
  OrbitalToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  DistributedParameterReduction reduction;
  BlindUpdater updater;
  std::vector<double> gradient{1.0, 2.0, 3.0};
  ParameterGradientView view{parameters.schema_fingerprint, parameters.version,
                             ReductionDomain::GLOBAL,
                             {gradient.data(), gradient.size()}};

  ParameterGradientView malformed = view;
  malformed.reduction_domain = ReductionDomain::RANK_LOCAL;
  CHECK_THROWS_WITH(completeParameterUpdate(provider, parameters, malformed, updater,
                                            reduction),
                    Catch::Matchers::ContainsSubstring("globally reduced"));
  malformed = view;
  malformed.parameter_version += 1;
  CHECK_THROWS_WITH(completeParameterUpdate(provider, parameters, malformed, updater,
                                            reduction),
                    Catch::Matchers::ContainsSubstring("incompatible"));
  malformed = view;
  malformed.gradient = {gradient.data(), gradient.size() - 1};
  CHECK_THROWS_WITH(completeParameterUpdate(provider, parameters, malformed, updater,
                                            reduction),
                    Catch::Matchers::ContainsSubstring("incompatible"));
  StructuredParameterSnapshot malformed_parameters = parameters;
  malformed_parameters.values.pop_back();
  CHECK_THROWS_WITH(completeParameterUpdate(provider, malformed_parameters, view, updater,
                                            reduction),
                    Catch::Matchers::ContainsSubstring("snapshot is incompatible"));
  malformed_parameters = parameters;
  malformed_parameters.values[0] = std::numeric_limits<double>::infinity();
  CHECK_THROWS_WITH(completeParameterUpdate(provider, malformed_parameters, view, updater,
                                            reduction),
                    Catch::Matchers::ContainsSubstring("snapshot is non-finite"));
  gradient[1] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_WITH(completeParameterUpdate(provider, parameters, view, updater,
                                            reduction),
                    Catch::Matchers::ContainsSubstring("non-finite"));
  CHECK(updater.calls == 0);
  CHECK(provider.snapshotParameters().version == parameters.version);
}

TEST_CASE("Orbital pretraining checkpoint resumes and binds target identity",
          "[drivers][training][orbital-pretraining][checkpoint]")
{
  const std::filesystem::path file = "wftrain_orbital_pretraining_restart.h5";
  CheckpointCleanup cleanup(file);
  ConvexOrbitalProducer producer(2);

  OrbitalToyProvider uninterrupted_provider;
  OrbitalPretrainingCoordinator coordinator(0x3456);
  TrainingIterationState uninterrupted_state = initialIterationState(uninterrupted_provider);
  TrainingStageState uninterrupted_stage = coordinator.makeStageState(4);
  FirstOrderOptimizer uninterrupted_optimizer(
      uninterrupted_provider.parameterSchema(),
      optimizerOptions(FirstOrderMethod::ADAM, 0.05));
  for (std::size_t step = 0; step < 2; ++step)
    coordinator.runIteration(uninterrupted_provider, producer,
                             uninterrupted_optimizer, uninterrupted_state,
                             uninterrupted_stage);
  TrainingCheckpoint::saveAtomic(file, uninterrupted_provider, uninterrupted_state,
                                 uninterrupted_stage, &uninterrupted_optimizer);
  coordinator.runIteration(uninterrupted_provider, producer, uninterrupted_optimizer,
                           uninterrupted_state, uninterrupted_stage);

  OrbitalToyProvider resumed_provider;
  TrainingIterationState resumed_state = initialIterationState(resumed_provider);
  TrainingStageState resumed_stage = coordinator.makeStageState(99);
  FirstOrderOptimizer resumed_optimizer(
      resumed_provider.parameterSchema(),
      optimizerOptions(FirstOrderMethod::ADAM, 0.05));
  TrainingCheckpoint::restore(file, resumed_provider, resumed_state, resumed_stage,
                              &resumed_optimizer);
  coordinator.runIteration(resumed_provider, producer, resumed_optimizer,
                           resumed_state, resumed_stage);
  CHECK(resumed_provider.snapshotParameters().values ==
        uninterrupted_provider.snapshotParameters().values);
  CHECK(resumed_state.completed_iterations == uninterrupted_state.completed_iterations);
  CHECK(resumed_stage.completed_stage_iterations ==
        uninterrupted_stage.completed_stage_iterations);

  OrbitalToyProvider wrong_provider;
  OrbitalPretrainingCoordinator wrong_coordinator(0x9876);
  TrainingIterationState wrong_state = initialIterationState(wrong_provider);
  TrainingStageState wrong_stage = wrong_coordinator.makeStageState();
  FirstOrderOptimizer wrong_optimizer(
      wrong_provider.parameterSchema(),
      optimizerOptions(FirstOrderMethod::ADAM, 0.05));
  CHECK_THROWS_WITH(TrainingCheckpoint::restore(file, wrong_provider, wrong_state,
                                                wrong_stage, &wrong_optimizer),
                    Catch::Matchers::ContainsSubstring("configuration"));
  CHECK(wrong_provider.snapshotParameters().version == 0);
}

TEST_CASE("Committed orbital parameters hand off to an energy objective update",
          "[drivers][training][orbital-pretraining][handoff]")
{
  OrbitalToyProvider provider;
  OrbitalPretrainingCoordinator coordinator(0x4567);
  TrainingIterationState state = initialIterationState(provider);
  TrainingStageState stage = coordinator.makeStageState();
  FirstOrderOptimizer pretrainer(provider.parameterSchema(),
                                 optimizerOptions(FirstOrderMethod::SGD, 0.1));
  ConvexOrbitalProducer producer(1);
  coordinator.runIteration(provider, producer, pretrainer, state, stage);

  const StructuredParameterSnapshot handoff = provider.snapshotParameters();
  HandoffEnergyProducer energy_producer(provider.parameterSchema());
  FirstOrderOptimizer energy_optimizer(provider.parameterSchema(),
      optimizerOptions(FirstOrderMethod::SGD, 0.01));
  HighParameterTraining energy_training(TrainingCapabilities{});
  const TrainingIterationResult energy_result = energy_training.runIteration(
      provider, energy_producer, energy_optimizer, state);
  CHECK(energy_result.parameter_version == handoff.version + 1);
  CHECK(energy_result.completed_iteration == 2);
  CHECK(provider.snapshotParameters().values[2] == 7.0);
  CHECK(energy_optimizer.acceptedUpdateCount() == 1);
}

} // namespace qmcplusplus::wftrain
