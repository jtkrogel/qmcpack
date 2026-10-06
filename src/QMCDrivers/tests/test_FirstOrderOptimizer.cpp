//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_FirstOrderOptimizer.cpp
 * @brief Deterministic numerical and restart tests for bounded first-order updates.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "QMCDrivers/WFTrain/FirstOrderOptimizer.h"

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Versioned provider with two trainable groups and one frozen scalar.
class OptimizerToyProvider final : public StructuredParameterProvider
{
public:
  OptimizerToyProvider()
      : schema_("optimizer/toy",
                {{"dense", {2}, 0, 2, ParameterScalarDomain::REAL64, true, "fast"},
                 {"bias", {1}, 2, 1, ParameterScalarDomain::REAL64, true, "slow"},
                 {"frozen", {1}, 3, 1, ParameterScalarDomain::REAL64, false, "fixed"}})
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
      throw std::runtime_error("synthetic optimizer provider failure");
    }
    if (expected_version != version_ || candidate.version != version_)
      throw std::runtime_error("stale optimizer toy version");
    if (candidate.schema_fingerprint != schema_.fingerprint() ||
        candidate.values.size() != values_.size())
      throw std::invalid_argument("invalid optimizer toy candidate");
    values_ = candidate.values;
    return ++version_;
  }

  void rejectNextPublish() noexcept { reject_next_ = true; }

private:
  StructuredParameterSchema schema_;
  std::vector<double> values_{1.0, -2.0, 0.5, 7.0};
  std::size_t version_ = 0;
  bool reject_next_    = false;
};

/// Remove test-owned checkpoint artifacts at scope exit.
class OptimizerCheckpointCleanup
{
public:
  explicit OptimizerCheckpointCleanup(std::filesystem::path path) : path_(std::move(path)) {}

  ~OptimizerCheckpointCleanup()
  {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

private:
  std::filesystem::path path_;
};

/// Return shared options with deliberately distinct block learning rates.
FirstOrderOptimizerOptions makeOptions(FirstOrderMethod method)
{
  FirstOrderOptimizerOptions options;
  options.method         = method;
  options.learning_rates = {{"fast", 0.1}, {"slow", 0.01}};
  options.momentum_decay = 0.5;
  options.rms_decay      = 0.75;
  options.adam_beta1     = 0.5;
  options.adam_beta2     = 0.75;
  options.epsilon        = 1.0e-6;
  return options;
}

/// Construct a complete global objective for one provider version.
EnergyGradientResult makeObjective(const OptimizerToyProvider& provider,
                                   std::vector<double> gradient)
{
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  EnergyGradientResult objective;
  objective.schema_fingerprint = parameters.schema_fingerprint;
  objective.parameter_version  = parameters.version;
  objective.reduction_domain   = ReductionDomain::GLOBAL;
  objective.gradient           = std::move(gradient);
  return objective;
}

/// Publish one direct optimizer proposal and finish both transaction participants.
void applyOptimizerStep(OptimizerToyProvider& provider,
                        FirstOrderOptimizer& optimizer,
                        const std::vector<double>& gradient,
                        TrainingIterationState& coordinator,
                        TrainingStageState& stage)
{
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  EnergyGradientResult objective = makeObjective(provider, gradient);
  StructuredParameterSnapshot candidate =
      optimizer.propose(provider.parameterSchema(), parameters, objective);
  const std::size_t committed_version =
      provider.publishParameters(candidate, parameters.version);
  optimizer.proposalAccepted(provider.parameterSchema(), parameters, objective);
  ++coordinator.completed_iterations;
  coordinator.parameter_version  = committed_version;
  coordinator.schema_fingerprint = provider.parameterSchema().fingerprint();
  ++stage.completed_stage_iterations;
}

/// Independently advance the reference recurrence and parameter vector.
void advanceReference(FirstOrderMethod method,
                      const FirstOrderOptimizerOptions& options,
                      std::uint64_t iteration,
                      const std::vector<double>& gradient,
                      std::vector<double>& parameters,
                      std::vector<double>& first,
                      std::vector<double>& second)
{
  const std::array<double, 4> learning_rates{0.1, 0.1, 0.01, 0.0};
  for (std::size_t index = 0; index < 3; ++index)
  {
    double direction = gradient[index];
    switch (method)
    {
    case FirstOrderMethod::SGD:
      break;
    case FirstOrderMethod::MOMENTUM_SGD:
      first[index] = options.momentum_decay * first[index] + gradient[index];
      direction    = first[index];
      break;
    case FirstOrderMethod::RMSPROP:
      second[index] = options.rms_decay * second[index] +
          (1.0 - options.rms_decay) * gradient[index] * gradient[index];
      direction = gradient[index] / (std::sqrt(second[index]) + options.epsilon);
      break;
    case FirstOrderMethod::ADAM:
      first[index] = options.adam_beta1 * first[index] +
          (1.0 - options.adam_beta1) * gradient[index];
      second[index] = options.adam_beta2 * second[index] +
          (1.0 - options.adam_beta2) * gradient[index] * gradient[index];
      direction = (first[index] / (1.0 - std::pow(options.adam_beta1, iteration))) /
          (std::sqrt(second[index] /
                     (1.0 - std::pow(options.adam_beta2, iteration))) +
           options.epsilon);
      break;
    }
    parameters[index] -= learning_rates[index] * direction;
  }
}

/// Overwrite one checkpointed moment vector without updating its fingerprint.
void overwriteFirstMoment(const std::filesystem::path& file,
                          const std::array<double, 4>& values)
{
  const hid_t handle = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  const hid_t dataset =
      handle >= 0
      ? H5Dopen2(handle, "/wftrain_checkpoint/optimizer/first_moment", H5P_DEFAULT)
      : -1;
  if (handle < 0 || dataset < 0 ||
      H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               values.data()) < 0)
  {
    if (dataset >= 0)
      H5Dclose(dataset);
    if (handle >= 0)
      H5Fclose(handle);
    throw std::runtime_error("Unable to mutate optimizer checkpoint moment");
  }
  H5Dclose(dataset);
  H5Fclose(handle);
}

} // namespace

TEST_CASE("First-order optimizers match independent multistep references",
          "[drivers][training][optimizer]")
{
  const std::array<FirstOrderMethod, 4> methods{
      FirstOrderMethod::SGD, FirstOrderMethod::MOMENTUM_SGD,
      FirstOrderMethod::RMSPROP, FirstOrderMethod::ADAM};
  const std::array<std::vector<double>, 3> gradients{
      std::vector<double>{1.0, -2.0, 0.5, 101.0},
      std::vector<double>{-0.25, 0.75, -1.0, 102.0},
      std::vector<double>{0.6, 0.2, 0.1, 103.0}};

  for (const FirstOrderMethod method : methods)
  {
    OptimizerToyProvider provider;
    const FirstOrderOptimizerOptions options = makeOptions(method);
    FirstOrderOptimizer optimizer(provider.parameterSchema(), options);
    StructuredParameterSnapshot parameters = provider.snapshotParameters();
    std::vector<double> expected_parameters = parameters.values;
    std::vector<double> expected_first(4, 0.0);
    std::vector<double> expected_second(4, 0.0);

    for (std::size_t step = 0; step < gradients.size(); ++step)
    {
      EnergyGradientResult objective = makeObjective(provider, gradients[step]);
      StructuredParameterSnapshot candidate =
          optimizer.propose(provider.parameterSchema(), parameters, objective);
      CHECK(optimizer.hasLiveProposal());
      advanceReference(method, options, step + 1, gradients[step], expected_parameters,
                       expected_first, expected_second);
      REQUIRE(candidate.values.size() == expected_parameters.size());
      for (std::size_t index = 0; index < candidate.values.size(); ++index)
        CHECK(candidate.values[index] == Catch::Approx(expected_parameters[index]).epsilon(1e-12));
      CHECK(candidate.values[3] == parameters.values[3]);

      const std::size_t version = provider.publishParameters(candidate, parameters.version);
      optimizer.proposalAccepted(provider.parameterSchema(), parameters, objective);
      CHECK_FALSE(optimizer.hasLiveProposal());
      parameters               = provider.snapshotParameters();
      CHECK(parameters.version == version);
    }

    CHECK(optimizer.acceptedUpdateCount() == gradients.size());
    const std::size_t expected_vectors =
        method == FirstOrderMethod::SGD ? 0
        : method == FirstOrderMethod::ADAM ? 2
                                           : 1;
    CHECK(optimizer.retainedBytes() == expected_vectors * 4 * sizeof(double));
    if (!optimizer.firstMoment().empty())
    {
      CHECK(optimizer.firstMoment()[3] == 0.0);
      if (method == FirstOrderMethod::MOMENTUM_SGD || method == FirstOrderMethod::ADAM)
        for (std::size_t index = 0; index < 3; ++index)
          CHECK(optimizer.firstMoment()[index] ==
                Catch::Approx(expected_first[index]).epsilon(1e-12));
    }
    if (!optimizer.secondMoment().empty())
    {
      CHECK(optimizer.secondMoment()[3] == 0.0);
      for (std::size_t index = 0; index < 3; ++index)
        CHECK(optimizer.secondMoment()[index] ==
              Catch::Approx(expected_second[index]).epsilon(1e-12));
    }
  }
}

TEST_CASE("First-order optimizer validates configuration and objective identity",
          "[drivers][training][optimizer]")
{
  OptimizerToyProvider provider;

  SECTION("learning-rate groups")
  {
    FirstOrderOptimizerOptions missing = makeOptions(FirstOrderMethod::SGD);
    missing.learning_rates.pop_back();
    CHECK_THROWS_WITH(FirstOrderOptimizer(provider.parameterSchema(), missing),
                      Catch::Matchers::ContainsSubstring("Missing"));

    FirstOrderOptimizerOptions duplicate = makeOptions(FirstOrderMethod::SGD);
    duplicate.learning_rates.push_back({"fast", 0.2});
    CHECK_THROWS_WITH(FirstOrderOptimizer(provider.parameterSchema(), duplicate),
                      Catch::Matchers::ContainsSubstring("Duplicate"));

    FirstOrderOptimizerOptions unknown = makeOptions(FirstOrderMethod::SGD);
    unknown.learning_rates.push_back({"unused", 0.2});
    CHECK_THROWS_WITH(FirstOrderOptimizer(provider.parameterSchema(), unknown),
                      Catch::Matchers::ContainsSubstring("Unknown"));

    FirstOrderOptimizerOptions nonpositive = makeOptions(FirstOrderMethod::SGD);
    nonpositive.learning_rates[0].learning_rate = 0.0;
    CHECK_THROWS_WITH(FirstOrderOptimizer(provider.parameterSchema(), nonpositive),
                      Catch::Matchers::ContainsSubstring("positive"));
  }

  SECTION("hyperparameters")
  {
    FirstOrderOptimizerOptions invalid = makeOptions(FirstOrderMethod::ADAM);
    invalid.adam_beta2 = 1.0;
    CHECK_THROWS_WITH(FirstOrderOptimizer(provider.parameterSchema(), invalid),
                      Catch::Matchers::ContainsSubstring("beta2"));
    invalid = makeOptions(FirstOrderMethod::RMSPROP);
    invalid.epsilon = std::numeric_limits<double>::infinity();
    CHECK_THROWS_WITH(FirstOrderOptimizer(provider.parameterSchema(), invalid),
                      Catch::Matchers::ContainsSubstring("epsilon"));
  }

  SECTION("objective metadata and finite values")
  {
    FirstOrderOptimizer optimizer(provider.parameterSchema(),
                                  makeOptions(FirstOrderMethod::ADAM));
    const StructuredParameterSnapshot parameters = provider.snapshotParameters();
    EnergyGradientResult objective = makeObjective(provider, {1.0, 2.0, 3.0, 4.0});
    objective.reduction_domain = ReductionDomain::RANK_LOCAL;
    CHECK_THROWS_WITH(optimizer.propose(provider.parameterSchema(), parameters, objective),
                      Catch::Matchers::ContainsSubstring("globally reduced"));
    objective.reduction_domain  = ReductionDomain::GLOBAL;
    objective.parameter_version = parameters.version + 1;
    CHECK_THROWS_WITH(optimizer.propose(provider.parameterSchema(), parameters, objective),
                      Catch::Matchers::ContainsSubstring("incompatible"));
    objective.parameter_version = parameters.version;
    objective.gradient[1]       = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_WITH(optimizer.propose(provider.parameterSchema(), parameters, objective),
                      Catch::Matchers::ContainsSubstring("finite"));
    objective.gradient[1] = std::numeric_limits<double>::max();
    CHECK_THROWS_WITH(optimizer.propose(provider.parameterSchema(), parameters, objective),
                      Catch::Matchers::ContainsSubstring("overflow"));
  }

  SECTION("complex parameter blocks")
  {
    const StructuredParameterSchema complex_schema(
        "optimizer/complex",
        {{"z", {1}, 0, 1, ParameterScalarDomain::COMPLEX128, true, "fast"}});
    FirstOrderOptimizerOptions options = makeOptions(FirstOrderMethod::SGD);
    options.learning_rates             = {{"fast", 0.1}};
    CHECK_THROWS_WITH(FirstOrderOptimizer(complex_schema, options),
                      Catch::Matchers::ContainsSubstring("real parameter"));
  }
}

TEST_CASE("First-order optimizer proposal rejection preserves committed recurrence",
          "[drivers][training][optimizer]")
{
  OptimizerToyProvider provider;
  FirstOrderOptimizer optimizer(provider.parameterSchema(),
                                makeOptions(FirstOrderMethod::MOMENTUM_SGD));
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  EnergyGradientResult objective = makeObjective(provider, {1.0, -2.0, 0.5, 9.0});
  const StructuredParameterSnapshot first =
      optimizer.propose(provider.parameterSchema(), parameters, objective);
  CHECK_THROWS_WITH(optimizer.checkpointMetadata(),
                    Catch::Matchers::ContainsSubstring("live proposal"));
  CHECK_THROWS_WITH(optimizer.propose(provider.parameterSchema(), parameters, objective),
                    Catch::Matchers::ContainsSubstring("live proposal"));
  optimizer.proposalRejected();
  CHECK(optimizer.acceptedUpdateCount() == 0);
  CHECK(std::all_of(optimizer.firstMoment().begin(), optimizer.firstMoment().end(),
                    [](double value) { return value == 0.0; }));

  const StructuredParameterSnapshot retry =
      optimizer.propose(provider.parameterSchema(), parameters, objective);
  CHECK(retry.values == first.values);
  optimizer.proposalRejected();
}

TEST_CASE("First-order optimizer checkpoint exactly resumes every recurrence",
          "[drivers][training][optimizer][checkpoint]")
{
  const std::array<FirstOrderMethod, 4> methods{
      FirstOrderMethod::SGD, FirstOrderMethod::MOMENTUM_SGD,
      FirstOrderMethod::RMSPROP, FirstOrderMethod::ADAM};
  const std::array<std::vector<double>, 4> gradients{
      std::vector<double>{1.0, -2.0, 0.5, 0.0},
      std::vector<double>{-0.25, 0.75, -1.0, 0.0},
      std::vector<double>{0.6, 0.2, 0.1, 0.0},
      std::vector<double>{-0.4, 0.3, 0.8, 0.0}};

  std::size_t ordinal = 0;
  for (const FirstOrderMethod method : methods)
  {
    const std::filesystem::path file =
        "wftrain_first_order_restart_" + std::to_string(ordinal++) + ".h5";
    OptimizerCheckpointCleanup cleanup(file);
    const FirstOrderOptimizerOptions options = makeOptions(method);

    OptimizerToyProvider uninterrupted_provider;
    FirstOrderOptimizer uninterrupted_optimizer(uninterrupted_provider.parameterSchema(),
                                                options);
    TrainingIterationState uninterrupted_coordinator{
        0, 0, uninterrupted_provider.parameterSchema().fingerprint()};
    TrainingStageState uninterrupted_stage{"optimization", 1, 0,
                                            "first-order-stage-v1"};
    for (std::size_t step = 0; step < 2; ++step)
      applyOptimizerStep(uninterrupted_provider, uninterrupted_optimizer, gradients[step],
                         uninterrupted_coordinator, uninterrupted_stage);
    TrainingCheckpoint::saveAtomic(file, uninterrupted_provider,
                                   uninterrupted_coordinator, uninterrupted_stage,
                                   &uninterrupted_optimizer);
    for (std::size_t step = 2; step < gradients.size(); ++step)
      applyOptimizerStep(uninterrupted_provider, uninterrupted_optimizer, gradients[step],
                         uninterrupted_coordinator, uninterrupted_stage);

    OptimizerToyProvider resumed_provider;
    FirstOrderOptimizerOptions reordered_options = options;
    std::reverse(reordered_options.learning_rates.begin(),
                 reordered_options.learning_rates.end());
    FirstOrderOptimizer resumed_optimizer(resumed_provider.parameterSchema(),
                                          reordered_options);
    TrainingIterationState resumed_coordinator{
        99, 0, resumed_provider.parameterSchema().fingerprint()};
    TrainingStageState resumed_stage{"optimization", 9, 9, "first-order-stage-v1"};
    TrainingCheckpoint::restore(file, resumed_provider, resumed_coordinator, resumed_stage,
                                &resumed_optimizer);
    for (std::size_t step = 2; step < gradients.size(); ++step)
      applyOptimizerStep(resumed_provider, resumed_optimizer, gradients[step],
                         resumed_coordinator, resumed_stage);

    const std::vector<double>& expected =
        uninterrupted_provider.snapshotParameters().values;
    const std::vector<double>& actual = resumed_provider.snapshotParameters().values;
    REQUIRE(actual.size() == expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index)
      CHECK(actual[index] == Catch::Approx(expected[index]).epsilon(1e-13));
    CHECK(resumed_optimizer.firstMoment() == uninterrupted_optimizer.firstMoment());
    CHECK(resumed_optimizer.secondMoment() == uninterrupted_optimizer.secondMoment());
    CHECK(resumed_optimizer.acceptedUpdateCount() ==
          uninterrupted_optimizer.acceptedUpdateCount());
    CHECK(resumed_coordinator.completed_iterations ==
          uninterrupted_coordinator.completed_iterations);
    CHECK(resumed_stage.completed_stage_iterations ==
          uninterrupted_stage.completed_stage_iterations);
  }
}

TEST_CASE("First-order optimizer rejects corrupt checkpoint state atomically",
          "[drivers][training][optimizer][checkpoint]")
{
  const std::filesystem::path file = "wftrain_first_order_corrupt.h5";
  OptimizerCheckpointCleanup cleanup(file);
  OptimizerToyProvider source_provider;
  FirstOrderOptimizer source_optimizer(
      source_provider.parameterSchema(), makeOptions(FirstOrderMethod::MOMENTUM_SGD));
  TrainingIterationState source_coordinator{
      0, 0, source_provider.parameterSchema().fingerprint()};
  TrainingStageState source_stage{"optimization", 1, 0, "first-order-corrupt-v1"};
  applyOptimizerStep(source_provider, source_optimizer, {1.0, -2.0, 0.5, 0.0},
                     source_coordinator, source_stage);
  TrainingCheckpoint::saveAtomic(file, source_provider, source_coordinator, source_stage,
                                 &source_optimizer);

  // A valid payload is still unusable by a differently configured recurrence.
  OptimizerToyProvider incompatible_provider;
  FirstOrderOptimizerOptions incompatible_options =
      makeOptions(FirstOrderMethod::MOMENTUM_SGD);
  incompatible_options.learning_rates[0].learning_rate = 0.2;
  FirstOrderOptimizer incompatible_optimizer(incompatible_provider.parameterSchema(),
                                             incompatible_options);
  TrainingIterationState incompatible_coordinator{
      0, 0, incompatible_provider.parameterSchema().fingerprint()};
  TrainingStageState incompatible_stage{"optimization", 0, 0,
                                         "first-order-corrupt-v1"};
  CHECK_THROWS_WITH(
      TrainingCheckpoint::restore(file, incompatible_provider, incompatible_coordinator,
                                  incompatible_stage, &incompatible_optimizer),
      Catch::Matchers::ContainsSubstring("configuration mismatch"));
  CHECK(incompatible_provider.snapshotParameters().version == 0);
  CHECK(incompatible_optimizer.acceptedUpdateCount() == 0);

  overwriteFirstMoment(file, {1.0, -2.0, 0.5, 4.0});

  OptimizerToyProvider target_provider;
  FirstOrderOptimizer target_optimizer(
      target_provider.parameterSchema(), makeOptions(FirstOrderMethod::MOMENTUM_SGD));
  TrainingIterationState target_coordinator{
      17, 0, target_provider.parameterSchema().fingerprint()};
  TrainingStageState target_stage{"optimization", 8, 9, "first-order-corrupt-v1"};
  const StructuredParameterSnapshot model_before = target_provider.snapshotParameters();
  CHECK_THROWS_WITH(
      TrainingCheckpoint::restore(file, target_provider, target_coordinator, target_stage,
                                  &target_optimizer),
      Catch::Matchers::ContainsSubstring("nonzero frozen first moment"));
  CHECK(target_provider.snapshotParameters().values == model_before.values);
  CHECK(target_provider.snapshotParameters().version == model_before.version);
  CHECK(target_optimizer.acceptedUpdateCount() == 0);
  CHECK(target_coordinator.completed_iterations == 17);
  CHECK(target_stage.stage_ordinal == 8);
  CHECK(target_stage.completed_stage_iterations == 9);
}

TEST_CASE("First-order optimizer retained state scales only with parameter count",
          "[drivers][training][optimizer]")
{
  constexpr std::size_t parameter_count = 200000;
  const StructuredParameterSchema schema(
      "optimizer/large",
      {{"weights", {parameter_count}, 0, parameter_count,
        ParameterScalarDomain::REAL64, true, "all"}});

  for (const auto [method, vectors] : {
           std::pair{FirstOrderMethod::SGD, std::size_t{0}},
           std::pair{FirstOrderMethod::MOMENTUM_SGD, std::size_t{1}},
           std::pair{FirstOrderMethod::RMSPROP, std::size_t{1}},
           std::pair{FirstOrderMethod::ADAM, std::size_t{2}}})
  {
    FirstOrderOptimizerOptions options = makeOptions(method);
    options.learning_rates             = {{"all", 0.01}};
    FirstOrderOptimizer optimizer(schema, options);
    CHECK(optimizer.retainedBytes() == vectors * parameter_count * sizeof(double));
  }
}

} // namespace qmcplusplus::wftrain
