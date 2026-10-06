//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_HighParameterTraining.cpp
 * @brief Unit tests for bounded derivative products and atomic training iterations.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "QMCDrivers/WFTrain/HighParameterTraining.h"
#include "QMCDrivers/WFTrain/FirstOrderOptimizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

using ComplexRows = std::vector<std::vector<DerivativeValue>>;

/// Small versioned provider used to verify coordinator transaction semantics.
class ToyProvider final : public StructuredParameterProvider
{
public:
  explicit ToyProvider(std::vector<std::string>* events = nullptr)
      : schema_("toy/model",
                {{"weight", {2}, 0, 2, ParameterScalarDomain::REAL64, true, "weights"},
                 {"frozen", {1}, 2, 1, ParameterScalarDomain::REAL64, false, "fixed"}}),
        values_{1.0, -2.0, 7.0},
        events_(events)
  {}

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }

  StructuredParameterSnapshot snapshotParameters() const override
  {
    return {schema_.fingerprint(), version_, values_};
  }

  std::size_t publishParameters(const StructuredParameterSnapshot& candidate,
                                std::size_t expected_version) override
  {
    if (reject_next_publish_)
    {
      reject_next_publish_ = false;
      throw std::runtime_error("stale toy version");
    }
    if (expected_version != version_)
      throw std::runtime_error("stale toy version");
    if (candidate.schema_fingerprint != schema_.fingerprint() ||
        candidate.values.size() != values_.size())
      throw std::invalid_argument("invalid toy candidate");
    for (const double value : candidate.values)
      if (!std::isfinite(value))
        throw std::invalid_argument("non-finite toy candidate");
    if (events_)
      events_->push_back("publish");
    values_ = candidate.values;
    return ++version_;
  }

  void rejectNextPublish() noexcept { reject_next_publish_ = true; }

private:
  StructuredParameterSchema schema_;
  std::size_t version_ = 0;
  std::vector<double> values_;
  std::vector<std::string>* events_ = nullptr;
  bool reject_next_publish_ = false;
};

/** Tiny dense derivative oracle confined to test code.
 *
 * Production APIs see only bounded parameter chunks. The explicit sample-by-parameter
 * rows here make the expected VJP algebra independent and easy to inspect.
 */
class ToyStreamingOperator final : public StreamingDerivativeOperator
{
public:
  ToyStreamingOperator(const StructuredParameterSchema& schema,
                       std::size_t parameter_version,
                       ComplexRows scores,
                       ComplexRows local_energy_derivatives,
                       bool fail_after_first_chunk = false,
                       const EnergyGradientAccumulator* merge_source = nullptr,
                       std::uint32_t local_energy_term_mask =
                           localEnergyTermBit(LocalEnergyTerm::KINETIC))
      : schema_(schema),
        parameter_version_(parameter_version),
        scores_(std::move(scores)),
        local_energy_derivatives_(std::move(local_energy_derivatives)),
        plan_(schema, parameter_version, 1),
        fail_after_first_chunk_(fail_after_first_chunk),
        merge_source_(merge_source),
        local_energy_term_mask_(local_energy_term_mask)
  {}

  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities result;
    result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
        derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_VJP);
    result.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::TRANSPOSE) |
        derivativeAdjointBit(DerivativeAdjoint::HERMITIAN);
    result.parameter_scalar_domain = ParameterScalarDomain::REAL64;
    result.result_scalar_domain    = ParameterScalarDomain::COMPLEX128;
    result.reduction_domain        = ReductionDomain::CROWD_LOCAL;
    result.execution_domain        = DerivativeExecutionDomain::HOST;
    result.local_energy_term_mask     = local_energy_term_mask_;
    result.maximum_vjp_channels          = 3;
    result.maximum_parameter_chunk_size = 1;
    result.maximum_sample_tile_size      = scores_.size();
    result.block_streaming                = true;
    return result;
  }

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }

  std::size_t parameterVersion() const noexcept override { return parameter_version_; }

  std::size_t batchOrdinal() const noexcept override { return 4; }

  std::size_t sampleOffset() const noexcept override { return 9; }

  std::size_t sampleCount() const noexcept override { return scores_.size(); }

  const ParameterChunkPlan& parameterChunkPlan() const noexcept override { return plan_; }

protected:
  void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                    DerivativeAdjoint adjoint,
                    ParameterReductionSink& sink) const override
  {
    std::size_t emitted_chunks = 0;
    for (const ParameterChunkDescriptor& descriptor : plan_.chunks())
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
      {
        const ComplexRows& jacobian =
            channels[channel].product == DerivativeProduct::SCORE_VJP
            ? scores_
            : local_energy_derivatives_;
        std::vector<DerivativeValue> contraction(descriptor.count);
        for (std::size_t parameter = 0; parameter < descriptor.count; ++parameter)
          for (std::size_t sample = 0; sample < scores_.size(); ++sample)
          {
            DerivativeValue derivative =
                jacobian[sample][descriptor.parameter_offset + parameter];
            if (adjoint == DerivativeAdjoint::HERMITIAN)
              derivative = std::conj(derivative);
            contraction[parameter] +=
                derivative * channels[channel].coefficients.values[sample];
          }

        sink.add(channel, {descriptor, {contraction.data(), contraction.size()}});
        if (merge_source_ && emitted_chunks == 0)
          dynamic_cast<EnergyGradientAccumulator&>(sink).merge(*merge_source_);
        if (fail_after_first_chunk_ && emitted_chunks == 0)
          throw std::runtime_error("synthetic streaming failure");
        ++emitted_chunks;
      }
  }

  void evaluateScoreJVP(const StructuredParameterVectorConstView&,
                        SampleProductSink&) const override
  {
    throw std::logic_error("Toy energy-gradient operator does not implement score JVP");
  }

private:
  const StructuredParameterSchema& schema_;
  std::size_t parameter_version_;
  ComplexRows scores_;
  ComplexRows local_energy_derivatives_;
  ParameterChunkPlan plan_;
  bool fail_after_first_chunk_;
  const EnergyGradientAccumulator* merge_source_;
  std::uint32_t local_energy_term_mask_;
};

/// Return the shared deterministic score rows for energy-objective tests.
ComplexRows makeScores()
{
  return {{1.0, 0.0, 3.0}, {2.0, -1.0, 3.0}, {-1.0, 4.0, 3.0}};
}

/// Return the shared deterministic local-energy response rows.
ComplexRows makeLocalEnergyDerivatives()
{
  return {{0.1, 0.2, 0.0}, {0.3, -0.1, 0.0}, {-0.2, 0.4, 0.0}};
}

/// Apply only the derivative half of an objective transaction for lifecycle tests.
void applyEnergyVJPsOnly(const StreamingDerivativeOperator& derivative_operator,
                         DerivativeArrayView<const DerivativeReal> weights,
                         DerivativeArrayView<const DerivativeValue> local_energies,
                         EnergyGradientAccumulator& accumulator)
{
  std::vector<DerivativeValue> weighted_score(weights.size());
  std::vector<DerivativeValue> energy_weighted_score(weights.size());
  for (std::size_t sample = 0; sample < weights.size(); ++sample)
  {
    weighted_score[sample] = weights[sample];
    energy_weighted_score[sample] = weights[sample] * local_energies[sample];
  }

  const StructuredParameterSchema& schema = derivative_operator.parameterSchema();
  const auto coefficients = [&](const std::vector<DerivativeValue>& values) {
    return CoefficientView{schema.providerId(), schema.fingerprint(),
                           derivative_operator.parameterVersion(),
                           derivative_operator.batchOrdinal(),
                           derivative_operator.sampleOffset(),
                           {values.data(), values.size()}};
  };
  const std::array<VJPCoefficientChannel, 3> channels{{
      {EnergyGradientAccumulator::WEIGHTED_SCORE_CHANNEL, DerivativeProduct::SCORE_VJP,
       coefficients(weighted_score), 0},
      {EnergyGradientAccumulator::ENERGY_WEIGHTED_SCORE_CHANNEL,
       DerivativeProduct::SCORE_VJP, coefficients(energy_weighted_score), 0},
      {EnergyGradientAccumulator::WEIGHTED_LOCAL_ENERGY_CHANNEL,
       DerivativeProduct::LOCAL_ENERGY_VJP, coefficients(weighted_score),
       localEnergyTermBit(LocalEnergyTerm::KINETIC)}}};
  derivative_operator.applyVJPs({channels.data(), channels.size()},
                                DerivativeAdjoint::TRANSPOSE, accumulator);
}

/// Produce one objective through the checked three-channel streaming boundary.
class ReferenceProducer final : public GradientProducer
{
public:
  explicit ReferenceProducer(std::vector<std::string>* events = nullptr,
                             bool offer_local_energy_vjp = true)
      : events_(events), offer_local_energy_vjp_(offer_local_energy_vjp)
  {}

  TrainingCapabilities capabilities() const noexcept override
  {
    TrainingCapabilities result{TrainingCapability::REAL_PARAMETERS,
                                TrainingCapability::SCORE_VJP};
    if (offer_local_energy_vjp_)
      result.add(TrainingCapability::LOCAL_ENERGY_VJP);
    return result;
  }

  void accumulate(const StructuredParameterSnapshot& parameters,
                  EnergyGradientAccumulator& accumulator) override
  {
    ++calls;
    if (events_)
      events_->push_back("produce");
    ToyStreamingOperator derivative_operator(*schema_, parameters.version, makeScores(),
                                               makeLocalEnergyDerivatives(), fail_next_);
    fail_next_ = false;
    const std::vector<DerivativeReal> weights{1.0, 2.0, 1.0};
    const std::vector<DerivativeValue> energies{-1.0, 0.5, 2.0};
    accumulateEnergyGradientBatch(
        derivative_operator, {weights.data(), weights.size()},
        {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
        accumulator);
  }

  void bindSchema(const StructuredParameterSchema& schema) noexcept { schema_ = &schema; }

  void failNext() noexcept { fail_next_ = true; }

  std::size_t calls = 0;

private:
  const StructuredParameterSchema* schema_ = nullptr;
  std::vector<std::string>* events_ = nullptr;
  bool offer_local_energy_vjp_ = true;
  bool fail_next_ = false;
};

/// Deterministic testing update used before production optimizers are introduced.
class ScaleAndSubtract final : public TrainingUpdateRule
{
public:
  ScaleAndSubtract(double scale, std::vector<std::string>* events = nullptr)
      : scale_(scale), events_(events)
  {}

  StructuredParameterSnapshot propose(const StructuredParameterSchema& schema,
                                      const StructuredParameterSnapshot& parameters,
                                      const EnergyGradientResult& objective) override
  {
    if (events_)
      events_->push_back("propose");
    StructuredParameterSnapshot result = parameters;
    for (const ParameterBlockDescriptor& block : schema.blocks())
      if (block.trainable)
        for (std::size_t index = block.offset; index < block.offset + block.count; ++index)
          result.values[index] -= scale_ * objective.gradient[index];
    return result;
  }

private:
  double scale_;
  std::vector<std::string>* events_;
};

/// Adversarial updater used to verify schema-owned frozen parameters are immutable.
class ModifyFrozenParameter final : public TrainingUpdateRule
{
public:
  explicit ModifyFrozenParameter(std::vector<std::string>* events = nullptr) : events_(events) {}

  StructuredParameterSnapshot propose(const StructuredParameterSchema& schema,
                                      const StructuredParameterSnapshot& parameters,
                                      const EnergyGradientResult&) override
  {
    if (events_)
      events_->push_back("propose");
    StructuredParameterSnapshot result = parameters;
    for (const ParameterBlockDescriptor& block : schema.blocks())
      if (!block.trainable && block.count != 0)
      {
        result.values[block.offset] += 1.0;
        return result;
      }
    throw std::logic_error("Toy schema has no frozen parameter");
  }

private:
  std::vector<std::string>* events_;
};

/// Record the sampler-cache refresh barrier without introducing sampler dependencies.
class CountingObserver final : public ParameterUpdateObserver
{
public:
  explicit CountingObserver(std::vector<std::string>* events = nullptr,
                            const TrainingIterationState* state = nullptr)
      : events_(events), state_(state)
  {}

  void parametersPublished(std::size_t new_version) noexcept override
  {
    if (events_)
      events_->push_back("observe");
    observed_committed_state =
        !state_ || (state_->completed_iterations != 0 &&
                    state_->parameter_version == new_version &&
                    !state_->schema_fingerprint.empty());
    ++calls;
    version = new_version;
  }

  std::size_t calls   = 0;
  std::size_t version = 0;
  bool observed_committed_state = false;

private:
  std::vector<std::string>* events_;
  const TrainingIterationState* state_;
};

} // namespace

TEST_CASE("Energy gradient streaming matches both finite-sample estimators",
          "[drivers][training]")
{
  ToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  const std::vector<DerivativeReal> weights{1.0, 2.0, 1.0};
  const std::vector<DerivativeValue> energies{-1.0, 0.5, 2.0};

  for (const auto [estimator, expected_0, expected_1] : {
           std::tuple{EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN, -1.25, 3.2},
           std::tuple{EnergyGradientEstimator::PATHWISE_LOCAL_ENERGY, -1.375, 3.1}})
  {
    ToyStreamingOperator derivative_operator(provider.parameterSchema(), parameters.version,
                                               makeScores(), makeLocalEnergyDerivatives());
    EnergyGradientAccumulator accumulator(provider.parameterSchema(), parameters.version,
                                           estimator);
    accumulateEnergyGradientBatch(
        derivative_operator, {weights.data(), weights.size()},
        {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
        accumulator);
    CHECK(accumulator.retainedBytes() == 3 * 3 * sizeof(DerivativeValue));
    accumulator.completeSingleParticipantReduction();

    const EnergyGradientResult result = accumulator.finalize();
    CHECK(result.sample_count == 3);
    CHECK(result.weight_sum == Catch::Approx(4.0));
    CHECK(result.mean_energy.real() == Catch::Approx(0.5));
    CHECK(result.energy_variance == Catch::Approx(1.125));
    CHECK(result.local_energy_term_mask == localEnergyTermBit(LocalEnergyTerm::KINETIC));
    REQUIRE(result.gradient.size() == 3);
    CHECK(result.gradient[0] == Catch::Approx(expected_0));
    CHECK(result.gradient[1] == Catch::Approx(expected_1));
    CHECK(result.gradient[2] == Catch::Approx(0.0));
  }
}

TEST_CASE("Energy-gradient batches reject nonempty null input views before sink work",
          "[drivers][training]")
{
  ToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  ToyStreamingOperator derivative_operator(provider.parameterSchema(), parameters.version,
                                             makeScores(), makeLocalEnergyDerivatives());
  const std::vector<DerivativeReal> weights{1.0, 2.0, 1.0};
  const std::vector<DerivativeValue> energies{-1.0, 0.5, 2.0};

  EnergyGradientAccumulator invalid_weights(provider.parameterSchema(), parameters.version);
  CHECK_THROWS_WITH(
      accumulateEnergyGradientBatch(
          derivative_operator, {nullptr, weights.size()},
          {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
          invalid_weights),
      Catch::Matchers::ContainsSubstring("require valid storage"));
  CHECK(invalid_weights.state() == DerivativeSinkState::IDLE);

  EnergyGradientAccumulator invalid_energies(provider.parameterSchema(), parameters.version);
  CHECK_THROWS_WITH(
      accumulateEnergyGradientBatch(
          derivative_operator, {weights.data(), weights.size()}, {nullptr, energies.size()},
          localEnergyTermBit(LocalEnergyTerm::KINETIC), invalid_energies),
      Catch::Matchers::ContainsSubstring("require valid storage"));
  CHECK(invalid_energies.state() == DerivativeSinkState::IDLE);
}

TEST_CASE("Poisoned energy-gradient stream requires reset before retry",
          "[drivers][training]")
{
  ToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  const std::vector<DerivativeReal> weights{1.0, 2.0, 1.0};
  const std::vector<DerivativeValue> energies{-1.0, 0.5, 2.0};
  EnergyGradientAccumulator accumulator(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator failing_operator(provider.parameterSchema(), parameters.version,
                                         makeScores(), makeLocalEnergyDerivatives(), true);

  CHECK_THROWS_WITH(
      accumulateEnergyGradientBatch(
          failing_operator, {weights.data(), weights.size()},
          {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
          accumulator),
      "synthetic streaming failure");
  CHECK(accumulator.state() == DerivativeSinkState::POISONED);
  CHECK_FALSE(accumulator.hasCompleteContribution());
  CHECK(accumulator.localEnergyTermMask() == 0);
  CHECK_THROWS(accumulator.finalize());

  accumulator.reset();
  CHECK(accumulator.localEnergyTermMask() == 0);
  ToyStreamingOperator retry_operator(provider.parameterSchema(), parameters.version,
                                       makeScores(), makeLocalEnergyDerivatives());
  accumulateEnergyGradientBatch(
      retry_operator, {weights.data(), weights.size()}, {energies.data(), energies.size()},
      localEnergyTermBit(LocalEnergyTerm::KINETIC), accumulator);
  CHECK(accumulator.localEnergyTermMask() == localEnergyTermBit(LocalEnergyTerm::KINETIC));
  accumulator.completeSingleParticipantReduction();
  CHECK(accumulator.finalize().gradient[0] == Catch::Approx(-1.25));
}

TEST_CASE("Complete crowd-local energy-gradient contributions merge deterministically",
          "[drivers][training]")
{
  ToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  const ComplexRows scores = makeScores();
  const ComplexRows local_energy_derivatives = makeLocalEnergyDerivatives();

  EnergyGradientAccumulator first(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator first_operator(provider.parameterSchema(), parameters.version,
                                       {scores[0]}, {local_energy_derivatives[0]});
  const std::vector<DerivativeReal> first_weights{1.0};
  const std::vector<DerivativeValue> first_energies{-1.0};
  accumulateEnergyGradientBatch(
      first_operator, {first_weights.data(), first_weights.size()},
      {first_energies.data(), first_energies.size()},
      localEnergyTermBit(LocalEnergyTerm::KINETIC), first);

  EnergyGradientAccumulator second(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator second_operator(
      provider.parameterSchema(), parameters.version, {scores[1], scores[2]},
      {local_energy_derivatives[1], local_energy_derivatives[2]});
  const std::vector<DerivativeReal> second_weights{2.0, 1.0};
  const std::vector<DerivativeValue> second_energies{0.5, 2.0};
  accumulateEnergyGradientBatch(
      second_operator, {second_weights.data(), second_weights.size()},
      {second_energies.data(), second_energies.size()},
      localEnergyTermBit(LocalEnergyTerm::KINETIC), second);

  // The first merge exercises an empty destination; the second exercises a
  // destination that already owns one complete compatible contribution.
  EnergyGradientAccumulator merged(provider.parameterSchema(), parameters.version);
  merged.merge(first);
  merged.merge(second);
  merged.completeSingleParticipantReduction();
  const EnergyGradientResult merged_result = merged.finalize();

  EnergyGradientAccumulator reference(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator reference_operator(provider.parameterSchema(), parameters.version,
                                           scores, local_energy_derivatives);
  const std::vector<DerivativeReal> reference_weights{1.0, 2.0, 1.0};
  const std::vector<DerivativeValue> reference_energies{-1.0, 0.5, 2.0};
  accumulateEnergyGradientBatch(
      reference_operator, {reference_weights.data(), reference_weights.size()},
      {reference_energies.data(), reference_energies.size()},
      localEnergyTermBit(LocalEnergyTerm::KINETIC), reference);
  reference.completeSingleParticipantReduction();
  const EnergyGradientResult reference_result = reference.finalize();

  CHECK(merged_result.sample_count == 3);
  CHECK(merged_result.sample_count == reference_result.sample_count);
  CHECK(merged_result.weight_sum == Catch::Approx(4.0));
  CHECK(merged_result.weight_sum == Catch::Approx(reference_result.weight_sum));
  CHECK(merged_result.mean_energy.real() == Catch::Approx(reference_result.mean_energy.real()));
  CHECK(merged_result.energy_variance == Catch::Approx(reference_result.energy_variance));
  CHECK(merged_result.local_energy_term_mask ==
        localEnergyTermBit(LocalEnergyTerm::KINETIC));
  CHECK(merged_result.local_energy_term_mask == reference_result.local_energy_term_mask);
  REQUIRE(merged_result.gradient.size() == reference_result.gradient.size());
  for (std::size_t parameter = 0; parameter < merged_result.gradient.size(); ++parameter)
    CHECK(merged_result.gradient[parameter] ==
          Catch::Approx(reference_result.gradient[parameter]));
}

TEST_CASE("Energy-gradient merge rejects invalid receiver lifecycle states",
          "[drivers][training]")
{
  ToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  const std::vector<DerivativeReal> weights{1.0, 2.0, 1.0};
  const std::vector<DerivativeValue> energies{-1.0, 0.5, 2.0};

  EnergyGradientAccumulator source(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator source_operator(provider.parameterSchema(), parameters.version,
                                        makeScores(), makeLocalEnergyDerivatives());
  accumulateEnergyGradientBatch(
      source_operator, {weights.data(), weights.size()}, {energies.data(), energies.size()},
      localEnergyTermBit(LocalEnergyTerm::KINETIC), source);

  // A checked derivative stream without its scalar moments is half complete.
  // After rejection, supplying those moments must recover the original result,
  // which proves merge did not mutate the receiver.
  EnergyGradientAccumulator half_complete(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator half_operator(provider.parameterSchema(), parameters.version,
                                      makeScores(), makeLocalEnergyDerivatives());
  applyEnergyVJPsOnly(half_operator, {weights.data(), weights.size()},
                      {energies.data(), energies.size()}, half_complete);
  CHECK_THROWS_WITH(half_complete.merge(source),
                    "Cannot merge into a half-complete energy-gradient accumulator");
  half_complete.addScalarSums(3, 4.0, DerivativeValue{2.0, 0.0}, 5.5);
  half_complete.completeSingleParticipantReduction();
  CHECK(half_complete.finalize().gradient[0] == Catch::Approx(-1.25));

  EnergyGradientAccumulator finalized(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator finalized_operator(provider.parameterSchema(), parameters.version,
                                           makeScores(), makeLocalEnergyDerivatives());
  accumulateEnergyGradientBatch(
      finalized_operator, {weights.data(), weights.size()}, {energies.data(), energies.size()},
      localEnergyTermBit(LocalEnergyTerm::KINETIC), finalized);
  finalized.completeSingleParticipantReduction();
  finalized.finalize();
  CHECK_THROWS_WITH(finalized.merge(source),
                    "Cannot merge into a finalized energy-gradient accumulator");

  // Force merge to execute while the checked sink transaction is active. The
  // operator wrapper poisons the receiver after propagating the lifecycle error.
  EnergyGradientAccumulator active(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator active_operator(provider.parameterSchema(), parameters.version,
                                        makeScores(), makeLocalEnergyDerivatives(), false,
                                        &source);
  CHECK_THROWS_WITH(
      accumulateEnergyGradientBatch(
          active_operator, {weights.data(), weights.size()},
          {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
          active),
      "Cannot merge into an active energy-gradient accumulator");
  CHECK(active.state() == DerivativeSinkState::POISONED);
  CHECK_THROWS_WITH(active.merge(source),
                    "Cannot merge into a poisoned energy-gradient accumulator");
}

TEST_CASE("Energy-gradient merge requires identical Hamiltonian-term coverage",
          "[drivers][training]")
{
  ToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  const std::vector<DerivativeReal> weights{1.0, 2.0, 1.0};
  const std::vector<DerivativeValue> energies{-1.0, 0.5, 2.0};
  const std::uint32_t kinetic_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
  const std::uint32_t kinetic_ecp_mask =
      kinetic_mask | localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);

  EnergyGradientAccumulator kinetic(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator kinetic_operator(provider.parameterSchema(), parameters.version,
                                         makeScores(), makeLocalEnergyDerivatives(), false,
                                         nullptr, kinetic_mask);
  accumulateEnergyGradientBatch(
      kinetic_operator, {weights.data(), weights.size()}, {energies.data(), energies.size()},
      kinetic_mask, kinetic);

  EnergyGradientAccumulator kinetic_ecp(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator kinetic_ecp_operator(
      provider.parameterSchema(), parameters.version, makeScores(),
      makeLocalEnergyDerivatives(), false, nullptr, kinetic_ecp_mask);
  accumulateEnergyGradientBatch(
      kinetic_ecp_operator, {weights.data(), weights.size()},
      {energies.data(), energies.size()}, kinetic_ecp_mask, kinetic_ecp);

  EnergyGradientAccumulator merged(provider.parameterSchema(), parameters.version);
  merged.merge(kinetic);
  CHECK_THROWS_WITH(
      merged.merge(kinetic_ecp),
      Catch::Matchers::ContainsSubstring("different local-energy term coverage"));

  // The failed second merge leaves the original kinetic-only contribution usable.
  merged.completeSingleParticipantReduction();
  const EnergyGradientResult result = merged.finalize();
  CHECK(result.sample_count == 3);
  CHECK(result.weight_sum == Catch::Approx(4.0));
  CHECK(result.gradient[0] == Catch::Approx(-1.25));
  CHECK(result.local_energy_term_mask == kinetic_mask);
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
  std::vector<std::string> events;
  ToyProvider provider(&events);
  ReferenceProducer producer(&events);
  producer.bindSchema(provider.parameterSchema());
  ScaleAndSubtract updater(0.1, &events);
  TrainingIterationState state;
  CountingObserver observer(&events, &state);
  HighParameterTraining training({});

  const TrainingIterationResult result =
      training.runIteration(provider, producer, updater, state, &observer);
  CHECK(result.completed_iteration == 1);
  CHECK(result.parameter_version == 1);
  CHECK(result.objective.parameter_version == 0);
  CHECK(observer.calls == 1);
  CHECK(observer.version == 1);
  CHECK(observer.observed_committed_state);
  CHECK(events == std::vector<std::string>{"produce", "propose", "publish", "observe"});

  const StructuredParameterSnapshot after = provider.snapshotParameters();
  CHECK(after.values[0] == Catch::Approx(1.125));
  CHECK(after.values[1] == Catch::Approx(-2.32));
  CHECK(after.values[2] == Catch::Approx(7.0));
}

TEST_CASE("Streaming and stale publication failures are atomic and retryable",
          "[drivers][training]")
{
  ToyProvider provider;
  ReferenceProducer producer;
  producer.bindSchema(provider.parameterSchema());
  ScaleAndSubtract updater(0.1);
  CountingObserver observer;
  TrainingIterationState state;
  HighParameterTraining training({});
  const StructuredParameterSnapshot initial = provider.snapshotParameters();

  producer.failNext();
  CHECK_THROWS_WITH(training.runIteration(provider, producer, updater, state, &observer),
                    "synthetic streaming failure");
  CHECK(provider.snapshotParameters().values == initial.values);
  CHECK(state.completed_iterations == 0);
  CHECK(observer.calls == 0);

  provider.rejectNextPublish();
  CHECK_THROWS_WITH(training.runIteration(provider, producer, updater, state, &observer),
                    "stale toy version");
  CHECK(provider.snapshotParameters().values == initial.values);
  CHECK(provider.snapshotParameters().version == initial.version);
  CHECK(state.completed_iterations == 0);
  CHECK(observer.calls == 0);

  const TrainingIterationResult retry =
      training.runIteration(provider, producer, updater, state, &observer);
  CHECK(retry.completed_iteration == 1);
  CHECK(observer.calls == 1);
}

TEST_CASE("Stateful optimizer publication failure preserves exact retry state",
          "[drivers][training][optimizer]")
{
  FirstOrderOptimizerOptions options;
  options.method         = FirstOrderMethod::ADAM;
  options.learning_rates = {{"weights", 0.05}};
  options.adam_beta1     = 0.5;
  options.adam_beta2     = 0.75;
  options.epsilon        = 1.0e-7;

  ToyProvider retry_provider;
  ReferenceProducer retry_producer;
  retry_producer.bindSchema(retry_provider.parameterSchema());
  FirstOrderOptimizer retry_optimizer(retry_provider.parameterSchema(), options);
  TrainingIterationState retry_state;
  HighParameterTraining training({});
  const StructuredParameterSnapshot initial = retry_provider.snapshotParameters();

  retry_provider.rejectNextPublish();
  CHECK_THROWS_WITH(
      training.runIteration(retry_provider, retry_producer, retry_optimizer, retry_state),
      "stale toy version");
  CHECK(retry_provider.snapshotParameters().values == initial.values);
  CHECK(retry_optimizer.acceptedUpdateCount() == 0);
  CHECK_FALSE(retry_optimizer.hasLiveProposal());
  CHECK(std::all_of(retry_optimizer.firstMoment().begin(),
                    retry_optimizer.firstMoment().end(),
                    [](double value) { return value == 0.0; }));
  CHECK(std::all_of(retry_optimizer.secondMoment().begin(),
                    retry_optimizer.secondMoment().end(),
                    [](double value) { return value == 0.0; }));

  training.runIteration(retry_provider, retry_producer, retry_optimizer, retry_state);

  ToyProvider reference_provider;
  ReferenceProducer reference_producer;
  reference_producer.bindSchema(reference_provider.parameterSchema());
  FirstOrderOptimizer reference_optimizer(reference_provider.parameterSchema(), options);
  TrainingIterationState reference_state;
  training.runIteration(reference_provider, reference_producer, reference_optimizer,
                        reference_state);

  CHECK(retry_provider.snapshotParameters().values ==
        reference_provider.snapshotParameters().values);
  CHECK(retry_optimizer.firstMoment() == reference_optimizer.firstMoment());
  CHECK(retry_optimizer.secondMoment() == reference_optimizer.secondMoment());
  CHECK(retry_optimizer.acceptedUpdateCount() == 1);
  CHECK(retry_state.completed_iterations == 1);
}

TEST_CASE("Training iteration-count overflow fails before producer and publication",
          "[drivers][training]")
{
  std::vector<std::string> events;
  ToyProvider provider(&events);
  ReferenceProducer producer(&events);
  producer.bindSchema(provider.parameterSchema());
  ScaleAndSubtract updater(0.1, &events);
  CountingObserver observer(&events);
  const StructuredParameterSnapshot before = provider.snapshotParameters();
  TrainingIterationState state{std::numeric_limits<std::size_t>::max(), before.version,
                               provider.parameterSchema().fingerprint()};
  HighParameterTraining training({});

  CHECK_THROWS_WITH(training.runIteration(provider, producer, updater, state, &observer),
                    "High-parameter training iteration count overflow");
  CHECK(producer.calls == 0);
  CHECK(events.empty());
  CHECK(provider.snapshotParameters().version == before.version);
  CHECK(provider.snapshotParameters().values == before.values);
  CHECK(state.completed_iterations == std::numeric_limits<std::size_t>::max());
  CHECK(observer.calls == 0);
}

TEST_CASE("Updater cannot modify a frozen parameter block",
          "[drivers][training]")
{
  std::vector<std::string> events;
  ToyProvider provider(&events);
  ReferenceProducer producer(&events);
  producer.bindSchema(provider.parameterSchema());
  ModifyFrozenParameter updater(&events);
  TrainingIterationState state;
  CountingObserver observer(&events, &state);
  HighParameterTraining training({});
  const StructuredParameterSnapshot before = provider.snapshotParameters();

  CHECK_THROWS_WITH(training.runIteration(provider, producer, updater, state, &observer),
                    Catch::Matchers::ContainsSubstring("modified frozen parameter block"));
  const StructuredParameterSnapshot after = provider.snapshotParameters();
  CHECK(after.version == before.version);
  CHECK(after.values == before.values);
  CHECK(state.completed_iterations == 0);
  CHECK(state.schema_fingerprint.empty());
  CHECK(observer.calls == 0);
  CHECK(events == std::vector<std::string>{"produce", "propose"});
}

TEST_CASE("Invalid distributed reduction policy fails before producer work",
          "[drivers][training]")
{
  ToyProvider provider;
  ReferenceProducer producer;
  producer.bindSchema(provider.parameterSchema());
  ScaleAndSubtract updater(0.1);
  TrainingIterationState state;
  HighParameterTraining training(
      {}, EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN,
      DistributedParameterReduction{DistributedReductionPolicy{0}});

  CHECK_THROWS_WITH(training.runIteration(provider, producer, updater, state),
                    Catch::Matchers::ContainsSubstring("invalid reduction policy"));
  CHECK(producer.calls == 0);
}

TEST_CASE("Missing streaming capability fails before producer work",
          "[drivers][training]")
{
  ToyProvider provider;
  ReferenceProducer producer(nullptr, false);
  producer.bindSchema(provider.parameterSchema());
  ScaleAndSubtract updater(0.1);
  TrainingIterationState state;
  HighParameterTraining training({});

  CHECK_THROWS_WITH(training.runIteration(provider, producer, updater, state),
                    Catch::Matchers::ContainsSubstring("local_energy_vjp"));
  CHECK(producer.calls == 0);
}

} // namespace qmcplusplus::wftrain
