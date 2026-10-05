//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_HighParameterTraining.cpp
 * @brief Unit tests for low-memory training contracts and atomic iteration ordering.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "QMCDrivers/WFTrain/HighParameterTraining.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Small versioned provider used to verify coordinator transaction semantics.
class ToyProvider final : public StructuredParameterProvider
{
public:
  ToyProvider()
      : schema_("toy/model",
                {{"weight", {2}, 0, 2, ParameterScalarDomain::REAL64, true, "weights"},
                 {"frozen", {1}, 2, 1, ParameterScalarDomain::REAL64, false, "fixed"}}),
        values_{1.0, -2.0, 7.0}
  {}

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }

  StructuredParameterSnapshot snapshotParameters() const override
  { return {schema_.fingerprint(), version_, values_}; }

  std::size_t publishParameters(const StructuredParameterSnapshot& candidate,
                                std::size_t expected_version) override
  {
    if (expected_version != version_)
      throw std::runtime_error("stale toy version");
    if (candidate.schema_fingerprint != schema_.fingerprint() ||
        candidate.values.size() != values_.size())
      throw std::invalid_argument("invalid toy candidate");
    for (const double value : candidate.values)
      if (!std::isfinite(value))
        throw std::invalid_argument("non-finite toy candidate");
    values_ = candidate.values;
    return ++version_;
  }

private:
  StructuredParameterSchema schema_;
  std::size_t version_ = 0;
  std::vector<double> values_;
};

/// Aggregate a fixed set of sample rows before crossing the streaming sink boundary.
class ReferenceProducer final : public GradientProducer
{
public:
  TrainingCapabilities capabilities() const noexcept override
  { return {TrainingCapability::REAL_PARAMETERS, TrainingCapability::SCORE_VJP}; }

  void accumulate(const StructuredParameterSnapshot&,
                  EnergyGradientAccumulator& accumulator) override
  {
    const std::vector<double> weights{1.0, 2.0, 1.0};
    const std::vector<double> energies{-1.0, 0.5, 2.0};
    const std::vector<std::vector<double>> scores{{1.0, 0.0, 3.0},
                                                  {2.0, -1.0, 3.0},
                                                  {-1.0, 4.0, 3.0}};
    const std::vector<std::vector<double>> energy_derivatives{{0.1, 0.2, 0.0},
                                                              {0.3, -0.1, 0.0},
                                                              {-0.2, 0.4, 0.0}};

    double weight_sum = 0.0;
    double energy_sum = 0.0;
    double energy_squared_sum = 0.0;
    std::vector<double> score_sum(3, 0.0);
    std::vector<double> energy_score_sum(3, 0.0);
    std::vector<double> energy_derivative_sum(3, 0.0);
    for (std::size_t sample = 0; sample < weights.size(); ++sample)
    {
      weight_sum += weights[sample];
      energy_sum += weights[sample] * energies[sample];
      energy_squared_sum += weights[sample] * energies[sample] * energies[sample];
      for (std::size_t parameter = 0; parameter < score_sum.size(); ++parameter)
      {
        score_sum[parameter] += weights[sample] * scores[sample][parameter];
        energy_score_sum[parameter] +=
            weights[sample] * energies[sample] * scores[sample][parameter];
        energy_derivative_sum[parameter] +=
            weights[sample] * energy_derivatives[sample][parameter];
      }
    }

    accumulator.addScalarSums(weights.size(), weight_sum, energy_sum, energy_squared_sum);
    accumulator.addDerivativeSums(0, 0, score_sum.data(), energy_score_sum.data(),
                                  energy_derivative_sum.data(), 2);
    accumulator.addDerivativeSums(1, 0, score_sum.data() + 2, energy_score_sum.data() + 2,
                                  energy_derivative_sum.data() + 2, 1);
  }
};

/// Deterministic testing update used before production optimizers are introduced.
class ScaleAndSubtract final : public TrainingUpdateRule
{
public:
  explicit ScaleAndSubtract(double scale) : scale_(scale) {}

  StructuredParameterSnapshot propose(const StructuredParameterSchema& schema,
                                      const StructuredParameterSnapshot& parameters,
                                      const EnergyGradientResult& objective) override
  {
    StructuredParameterSnapshot result = parameters;
    for (const ParameterBlockDescriptor& block : schema.blocks())
      if (block.trainable)
        for (std::size_t index = block.offset; index < block.offset + block.count; ++index)
          result.values[index] -= scale_ * objective.gradient[index];
    return result;
  }

private:
  double scale_;
};

/// Record the sampler-cache refresh barrier without introducing sampler dependencies.
class CountingObserver final : public ParameterUpdateObserver
{
public:
  void parametersPublished(std::size_t new_version) noexcept override
  {
    ++calls;
    version = new_version;
  }

  std::size_t calls   = 0;
  std::size_t version = 0;
};

/// Producer that proves a failed reduction leaves provider and iteration state untouched.
class FailingProducer final : public GradientProducer
{
public:
  TrainingCapabilities capabilities() const noexcept override
  { return {TrainingCapability::REAL_PARAMETERS, TrainingCapability::SCORE_VJP}; }

  void accumulate(const StructuredParameterSnapshot&,
                  EnergyGradientAccumulator& accumulator) override
  {
    accumulator.addScalarSums(1, 1.0, 0.0, 0.0);
    throw std::runtime_error("synthetic producer failure");
  }
};

} // namespace

TEST_CASE("Energy gradient reducer matches an explicit weighted reference",
          "[drivers][training]")
{
  ToyProvider provider;
  ReferenceProducer producer;
  EnergyGradientAccumulator accumulator(provider.parameterSchema());
  producer.accumulate(provider.snapshotParameters(), accumulator);
  CHECK(accumulator.retainedBytes() == 3 * 3 * sizeof(double));

  const EnergyGradientResult result = accumulator.finalize();
  CHECK(result.sample_count == 3);
  CHECK(result.weight_sum == Catch::Approx(4.0));
  CHECK(result.mean_energy == Catch::Approx(0.5));
  CHECK(result.energy_variance == Catch::Approx(1.125));
  REQUIRE(result.gradient.size() == 3);
  CHECK(result.gradient[0] == Catch::Approx(-1.25));
  CHECK(result.gradient[1] == Catch::Approx(3.2));
  CHECK(result.gradient[2] == Catch::Approx(0.0));
}

TEST_CASE("Training capability preflight lists every missing operation",
          "[drivers][training]")
{
  const TrainingCapabilities offered{TrainingCapability::REAL_PARAMETERS};
  const TrainingCapabilities required{TrainingCapability::SCORE_VJP,
                                      TrainingCapability::NONLOCAL_ECP};
  CHECK_THROWS_WITH(requireTrainingCapabilities(offered, required, "toy producer"),
                    Catch::Matchers::ContainsSubstring("score_vjp") &&
                        Catch::Matchers::ContainsSubstring("nonlocal_ecp"));
}

TEST_CASE("High-parameter iteration publishes then refreshes sampler state",
          "[drivers][training]")
{
  ToyProvider provider;
  ReferenceProducer producer;
  ScaleAndSubtract updater(0.1);
  CountingObserver observer;
  TrainingIterationState state;
  HighParameterTraining training(
      {TrainingCapability::REAL_PARAMETERS, TrainingCapability::SCORE_VJP});

  const TrainingIterationResult first =
      training.runIteration(provider, producer, updater, state, &observer);
  CHECK(first.completed_iteration == 1);
  CHECK(first.parameter_version == 1);
  CHECK(observer.calls == 1);
  CHECK(observer.version == 1);
  const StructuredParameterSnapshot after_first = provider.snapshotParameters();
  CHECK(after_first.values[0] == Catch::Approx(1.125));
  CHECK(after_first.values[1] == Catch::Approx(-2.32));
  CHECK(after_first.values[2] == Catch::Approx(7.0));

  // A copied state represents an in-memory deterministic restart boundary.
  const TrainingIterationState restarted_state = state;
  const TrainingIterationResult second =
      training.runIteration(provider, producer, updater, state, &observer);
  CHECK(second.completed_iteration == restarted_state.completed_iterations + 1);
  CHECK(state.parameter_version == 2);
  CHECK(observer.calls == 2);
}

TEST_CASE("High-parameter iteration failure is atomic", "[drivers][training]")
{
  ToyProvider provider;
  FailingProducer producer;
  ScaleAndSubtract updater(0.1);
  TrainingIterationState state;
  HighParameterTraining training(
      {TrainingCapability::REAL_PARAMETERS, TrainingCapability::SCORE_VJP});
  const StructuredParameterSnapshot before = provider.snapshotParameters();

  CHECK_THROWS_WITH(training.runIteration(provider, producer, updater, state),
                    "synthetic producer failure");
  const StructuredParameterSnapshot after = provider.snapshotParameters();
  CHECK(after.version == before.version);
  CHECK(after.values == before.values);
  CHECK(state.completed_iterations == 0);
  CHECK(state.schema_fingerprint.empty());
}

} // namespace qmcplusplus::wftrain
