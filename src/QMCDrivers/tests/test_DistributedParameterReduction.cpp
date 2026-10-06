//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_DistributedParameterReduction.cpp
 * @brief MPI tests for replicated, chunked high-parameter reductions.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "Message/Communicate.h"
#include "QMCDrivers/WFTrain/HighParameterTraining.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Return an uneven partition containing an empty interior rank when possible.
std::size_t localSampleCount(int rank, int size) noexcept
{
  return size > 1 && rank == 1 ? 0 : static_cast<std::size_t>(rank + 1);
}

/// Five-parameter schema gives chunk size two a non-divisor tail.
StructuredParameterSchema makeSchema(std::string provider_id = "distributed/toy")
{
  return {std::move(provider_id),
          {{"weight", {5}, 0, 5, ParameterScalarDomain::REAL64, true, "weights"}}};
}

/// Generate deterministic rank/sample data independently of the derivative producer.
void makeSamples(int rank,
                 std::size_t count,
                 std::vector<DerivativeReal>& weights,
                 std::vector<DerivativeValue>& energies)
{
  weights.resize(count);
  energies.resize(count);
  for (std::size_t sample = 0; sample < count; ++sample)
  {
    const double identity = static_cast<double>(10 * rank + sample + 1);
    weights[sample]       = 1.0 + 0.25 * sample;
    energies[sample]      = -0.5 + 0.1 * identity;
  }
}

/// Deterministic bounded VJP producer used only by the distributed reduction test.
class RankStreamingOperator final : public StreamingDerivativeOperator
{
public:
  RankStreamingOperator(const StructuredParameterSchema& schema,
                        std::size_t version,
                        int rank,
                        std::size_t sample_count,
                        DerivativeAdjoint adjoint = DerivativeAdjoint::TRANSPOSE)
      : schema_(schema), version_(version), rank_(rank), sample_count_(sample_count),
        adjoint_(adjoint), plan_(schema, version, 2)
  {}

  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities result;
    result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
        derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_VJP);
    result.adjoint_mask = derivativeAdjointBit(adjoint_);
    result.parameter_scalar_domain = ParameterScalarDomain::REAL64;
    result.result_scalar_domain    = ParameterScalarDomain::COMPLEX128;
    result.reduction_domain        = ReductionDomain::RANK_LOCAL;
    result.execution_domain        = DerivativeExecutionDomain::HOST;
    result.local_energy_term_mask  = localEnergyTermBit(LocalEnergyTerm::KINETIC);
    result.maximum_vjp_channels = 3;
    result.maximum_parameter_chunk_size = 2;
    result.maximum_sample_tile_size = std::max<std::size_t>(sample_count_, 1);
    result.block_streaming = true;
    return result;
  }

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }
  std::size_t parameterVersion() const noexcept override { return version_; }
  std::size_t batchOrdinal() const noexcept override { return 0; }
  std::size_t sampleOffset() const noexcept override { return static_cast<std::size_t>(10 * rank_); }
  std::size_t sampleCount() const noexcept override { return sample_count_; }
  const ParameterChunkPlan& parameterChunkPlan() const noexcept override { return plan_; }

protected:
  void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                    DerivativeAdjoint,
                    ParameterReductionSink& sink) const override
  {
    for (const ParameterChunkDescriptor& descriptor : plan_.chunks())
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
      {
        std::vector<DerivativeValue> contraction(descriptor.count);
        for (std::size_t local_parameter = 0; local_parameter < descriptor.count;
             ++local_parameter)
        {
          const std::size_t parameter = descriptor.parameter_offset + local_parameter;
          for (std::size_t sample = 0; sample < sample_count_; ++sample)
          {
            const double identity = static_cast<double>(10 * rank_ + sample + 1);
            const DerivativeValue derivative =
                channels[channel].product == DerivativeProduct::SCORE_VJP
                ? 0.01 * identity * static_cast<double>(parameter + 1)
                : 0.001 * (identity + 2.0) * static_cast<double>(parameter + 1);
            contraction[local_parameter] +=
                derivative * channels[channel].coefficients.values[sample];
          }
        }
        sink.add(channel, {descriptor, {contraction.data(), contraction.size()}});
      }
  }

  void evaluateScoreJVP(const StructuredParameterVectorConstView&,
                        SampleProductSink&) const override
  {
    throw std::logic_error("RankStreamingOperator does not implement score JVP");
  }

private:
  const StructuredParameterSchema& schema_;
  std::size_t version_;
  int rank_;
  std::size_t sample_count_;
  DerivativeAdjoint adjoint_;
  ParameterChunkPlan plan_;
};

/// Construct one complete rank-local contribution, including the zero-sample case.
std::unique_ptr<EnergyGradientAccumulator> makeContribution(
    const StructuredParameterSchema& schema,
    std::size_t version,
    int rank,
    std::size_t sample_count,
    DerivativeAdjoint adjoint = DerivativeAdjoint::TRANSPOSE)
{
  auto accumulator = std::make_unique<EnergyGradientAccumulator>(
      schema, version, EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN, adjoint);
  RankStreamingOperator derivative_operator(schema, version, rank, sample_count, adjoint);
  std::vector<DerivativeReal> weights;
  std::vector<DerivativeValue> energies;
  makeSamples(rank, sample_count, weights, energies);
  accumulateEnergyGradientBatch(
      derivative_operator, {weights.data(), weights.size()},
      {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
      *accumulator, adjoint);
  return accumulator;
}

/// Form the independent global objective expected from all synthetic ranks.
EnergyGradientResult makeReferenceResult(int size)
{
  DerivativeReal weight_sum = 0.0;
  DerivativeValue energy_sum = 0.0;
  DerivativeReal energy_norm_sum = 0.0;
  std::vector<DerivativeValue> score(5);
  std::vector<DerivativeValue> energy_score(5);
  std::vector<DerivativeValue> response(5);
  std::size_t sample_count = 0;

  for (int rank = 0; rank < size; ++rank)
    for (std::size_t sample = 0; sample < localSampleCount(rank, size); ++sample)
    {
      const double identity = static_cast<double>(10 * rank + sample + 1);
      const double weight   = 1.0 + 0.25 * sample;
      const double energy   = -0.5 + 0.1 * identity;
      ++sample_count;
      weight_sum += weight;
      energy_sum += weight * energy;
      energy_norm_sum += weight * energy * energy;
      for (std::size_t parameter = 0; parameter < 5; ++parameter)
      {
        const double score_derivative = 0.01 * identity * (parameter + 1);
        const double energy_derivative = 0.001 * (identity + 2.0) * (parameter + 1);
        score[parameter] += weight * score_derivative;
        energy_score[parameter] += weight * energy * score_derivative;
        response[parameter] += weight * energy_derivative;
      }
    }

  EnergyGradientResult result;
  result.sample_count = sample_count;
  result.weight_sum = weight_sum;
  result.mean_energy = energy_sum / weight_sum;
  result.energy_variance =
      energy_norm_sum / weight_sum - std::norm(result.mean_energy);
  result.gradient.resize(5);
  for (std::size_t parameter = 0; parameter < 5; ++parameter)
    result.gradient[parameter] =
        2.0 * std::real(response[parameter] + energy_score[parameter] -
                        result.mean_energy * score[parameter]) /
        weight_sum;
  return result;
}

/// Construct one raw orbital objective with deterministic rank/sample values.
std::unique_ptr<OrbitalPretrainingAccumulator> makeOrbitalContribution(
    const StructuredParameterSchema& schema,
    std::size_t version,
    int rank,
    std::size_t sample_count,
    std::uint64_t target_fingerprint = 0x1234,
    std::uint64_t loss_fingerprint = 0x5678)
{
  auto accumulator = std::make_unique<OrbitalPretrainingAccumulator>(
      schema, version, target_fingerprint, loss_fingerprint);
  std::vector<double> gradient(schema.parameterCount());
  for (std::size_t sample = 0; sample < sample_count; ++sample)
  {
    const double identity = static_cast<double>(10 * rank + sample + 1);
    for (std::size_t parameter = 0; parameter < gradient.size(); ++parameter)
      gradient[parameter] = 0.1 * identity * static_cast<double>(parameter + 1);
    accumulator->addSample(identity * identity, version,
                           {gradient.data(), gradient.size()});
  }
  return accumulator;
}

/// Form the independently summed mean loss and gradient for all MPI ranks.
OrbitalPretrainingResult makeOrbitalReference(int size)
{
  OrbitalPretrainingResult result;
  result.gradient.assign(5, 0.0);
  for (int rank = 0; rank < size; ++rank)
    for (std::size_t sample = 0; sample < localSampleCount(rank, size); ++sample)
    {
      const double identity = static_cast<double>(10 * rank + sample + 1);
      ++result.sample_count;
      result.mean_loss += identity * identity;
      for (std::size_t parameter = 0; parameter < result.gradient.size(); ++parameter)
        result.gradient[parameter] += 0.1 * identity * static_cast<double>(parameter + 1);
    }
  result.mean_loss /= result.sample_count;
  for (double& value : result.gradient)
    value /= result.sample_count;
  return result;
}

/// Versioned provider used to test the complete distributed publication ordering.
class RankProvider final : public StructuredParameterProvider
{
public:
  explicit RankProvider(std::vector<std::string>* events = nullptr)
      : schema_(makeSchema()), values_(5, 0.5), events_(events)
  {}

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }
  StructuredParameterSnapshot snapshotParameters() const override
  {
    return {schema_.fingerprint(), version_, values_};
  }
  std::size_t publishParameters(const StructuredParameterSnapshot& candidate,
                                std::size_t expected_version) override
  {
    if (expected_version != version_)
      throw std::runtime_error("stale rank provider");
    if (events_)
      events_->push_back("publish");
    values_ = candidate.values;
    return ++version_;
  }

private:
  StructuredParameterSchema schema_;
  std::vector<double> values_;
  std::vector<std::string>* events_;
  std::size_t version_ = 0;
};

/// Produce the rank-local synthetic batch through the normal training interface.
class RankProducer final : public GradientProducer
{
public:
  RankProducer(const StructuredParameterSchema& schema,
               Communicate& communicator,
               std::vector<std::string>* events = nullptr)
      : schema_(schema), communicator_(communicator), events_(events)
  {}

  TrainingCapabilities capabilities() const noexcept override
  {
    return {TrainingCapability::REAL_PARAMETERS, TrainingCapability::SCORE_VJP,
            TrainingCapability::LOCAL_ENERGY_VJP};
  }

  void accumulate(const StructuredParameterSnapshot& parameters,
                  EnergyGradientAccumulator& accumulator) override
  {
    if (events_)
      events_->push_back("produce");
    const std::size_t count = localSampleCount(communicator_.rank(), communicator_.size());
    RankStreamingOperator derivative_operator(schema_, parameters.version,
                                               communicator_.rank(), count);
    std::vector<DerivativeReal> weights;
    std::vector<DerivativeValue> energies;
    makeSamples(communicator_.rank(), count, weights, energies);
    accumulateEnergyGradientBatch(
        derivative_operator, {weights.data(), weights.size()},
        {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
        accumulator);
  }

private:
  const StructuredParameterSchema& schema_;
  Communicate& communicator_;
  std::vector<std::string>* events_;
};

/// Deterministic updater with optional rank-local failure or candidate divergence.
class RankUpdater final : public TrainingUpdateRule
{
public:
  RankUpdater(Communicate& communicator,
              bool fail_last = false,
              bool diverge_last = false,
              std::vector<std::string>* events = nullptr)
      : communicator_(communicator), fail_last_(fail_last),
        diverge_last_(diverge_last), events_(events)
  {}

  StructuredParameterSnapshot propose(const StructuredParameterSchema&,
                                      const StructuredParameterSnapshot& parameters,
                                      ParameterGradientView objective) override
  {
    if (events_)
      events_->push_back("propose");
    if (fail_last_ && communicator_.rank() == communicator_.size() - 1)
      throw std::runtime_error("rank-local update failure");
    StructuredParameterSnapshot candidate = parameters;
    for (std::size_t parameter = 0; parameter < candidate.values.size(); ++parameter)
      candidate.values[parameter] -= 0.01 * objective.gradient[parameter];
    if (diverge_last_ && communicator_.size() > 1 &&
        communicator_.rank() == communicator_.size() - 1)
      candidate.values.front() += 0.25;
    return candidate;
  }

private:
  Communicate& communicator_;
  bool fail_last_;
  bool diverge_last_;
  std::vector<std::string>* events_;
};

/// Observe the final cache-refresh barrier in the successful path.
class RankObserver final : public ParameterUpdateObserver
{
public:
  explicit RankObserver(std::vector<std::string>& events) : events_(events) {}
  void parametersPublished(std::size_t) noexcept override
  {
    events_.push_back("observe");
    ++calls;
  }
  std::size_t calls = 0;

private:
  std::vector<std::string>& events_;
};

} // namespace

TEST_CASE("Distributed parameter reduction matches an independent global reference",
          "[drivers][training][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  const StructuredParameterSchema schema = makeSchema();
  const StructuredParameterSnapshot parameters{schema.fingerprint(), 0,
                                                std::vector<double>(5, 0.5)};
  const EnergyGradientResult reference = makeReferenceResult(communicator.size());

  for (const std::size_t chunk_size : {std::size_t{1}, std::size_t{2}})
  {
    DistributedParameterReduction reduction(communicator, {chunk_size});
    reduction.preflight(schema, &parameters,
                        EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN);
    auto accumulator = makeContribution(
        schema, parameters.version, communicator.rank(),
        localSampleCount(communicator.rank(), communicator.size()));
    const std::size_t retained_bytes = accumulator->retainedBytes();
    reduction.reduce(*accumulator);
    CHECK(accumulator->retainedBytes() == retained_bytes);
    const EnergyGradientResult result = accumulator->finalize();
    CHECK(result.reduction_domain == ReductionDomain::GLOBAL);
    CHECK(result.sample_count == reference.sample_count);
    CHECK(result.weight_sum == Catch::Approx(reference.weight_sum));
    CHECK(result.mean_energy.real() == Catch::Approx(reference.mean_energy.real()));
    CHECK(result.energy_variance == Catch::Approx(reference.energy_variance).margin(1e-14));
    for (std::size_t parameter = 0; parameter < result.gradient.size(); ++parameter)
      CHECK(result.gradient[parameter] == Catch::Approx(reference.gradient[parameter]));
  }
}

TEST_CASE("Distributed reduction failures are uniform and retryable",
          "[drivers][training][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  const StructuredParameterSchema schema = makeSchema();
  const StructuredParameterSnapshot parameters{schema.fingerprint(), 0,
                                                std::vector<double>(5, 0.5)};
  DistributedParameterReduction reduction(communicator, {2});
  reduction.preflight(schema, &parameters,
                      EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN);

  auto empty = makeContribution(schema, 0, communicator.rank(), 0);
  CHECK_THROWS_WITH(reduction.reduce(*empty),
                    Catch::Matchers::ContainsSubstring("globally empty"));

  std::unique_ptr<EnergyGradientAccumulator> failed;
  std::exception_ptr local_failure;
  if (communicator.rank() == 0)
  {
    failed = std::make_unique<EnergyGradientAccumulator>(schema, 0);
    const std::size_t count = localSampleCount(communicator.rank(), communicator.size());
    RankStreamingOperator derivative_operator(schema, 0, communicator.rank(), count);
    std::vector<DerivativeReal> weights;
    std::vector<DerivativeValue> energies;
    makeSamples(communicator.rank(), count, weights, energies);
    energies.front() = {std::numeric_limits<double>::quiet_NaN(), 0.0};
    try
    {
      accumulateEnergyGradientBatch(
          derivative_operator, {weights.data(), weights.size()},
          {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
          *failed);
    }
    catch (...)
    {
      local_failure = std::current_exception();
    }
  }
  else
    failed = makeContribution(schema, 0, communicator.rank(),
                              localSampleCount(communicator.rank(), communicator.size()));
  CHECK_THROWS(reduction.reduce(*failed, local_failure));

  auto retry = makeContribution(schema, 0, communicator.rank(),
                                localSampleCount(communicator.rank(), communicator.size()));
  CHECK_NOTHROW(reduction.reduce(*retry));
  CHECK_NOTHROW(retry->finalize());
}

TEST_CASE("Distributed orbital reduction handles uneven and zero-sample ranks",
          "[drivers][training][orbital-pretraining][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  const StructuredParameterSchema schema = makeSchema();
  const StructuredParameterSnapshot parameters{schema.fingerprint(), 3,
                                                std::vector<double>(5, 0.5)};
  const OrbitalPretrainingResult reference = makeOrbitalReference(communicator.size());
  DistributedParameterReduction reduction(communicator, {2});
  reduction.preflightOrbital(schema, &parameters, 0x1234, 0x5678);
  auto accumulator = makeOrbitalContribution(
      schema, parameters.version, communicator.rank(),
      localSampleCount(communicator.rank(), communicator.size()));
  const std::size_t retained = accumulator->retainedBytes();
  reduction.reduce(*accumulator);
  CHECK(accumulator->retainedBytes() == retained);
  const OrbitalPretrainingResult result = accumulator->finalize();
  CHECK(result.reduction_domain == ReductionDomain::GLOBAL);
  CHECK(result.sample_count == reference.sample_count);
  CHECK(result.mean_loss == Catch::Approx(reference.mean_loss));
  REQUIRE(result.gradient.size() == reference.gradient.size());
  for (std::size_t parameter = 0; parameter < result.gradient.size(); ++parameter)
    CHECK(result.gradient[parameter] == Catch::Approx(reference.gradient[parameter]));
}

TEST_CASE("Distributed orbital failures are uniform and retryable",
          "[drivers][training][orbital-pretraining][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  const StructuredParameterSchema schema = makeSchema();
  const StructuredParameterSnapshot parameters{schema.fingerprint(), 0,
                                                std::vector<double>(5, 0.5)};
  DistributedParameterReduction reduction(communicator, {2});
  reduction.preflightOrbital(schema, &parameters, 0x1234, 0x5678);

  auto empty = makeOrbitalContribution(schema, 0, communicator.rank(), 0);
  CHECK_THROWS_WITH(reduction.reduce(*empty),
                    Catch::Matchers::ContainsSubstring("globally empty"));

  auto failed = makeOrbitalContribution(
      schema, 0, communicator.rank(),
      localSampleCount(communicator.rank(), communicator.size()));
  std::exception_ptr local_failure;
  if (communicator.rank() == 0)
    local_failure = std::make_exception_ptr(std::runtime_error("orbital producer failed"));
  CHECK_THROWS(reduction.reduce(*failed, local_failure));

  auto retry = makeOrbitalContribution(
      schema, 0, communicator.rank(),
      localSampleCount(communicator.rank(), communicator.size()));
  CHECK_NOTHROW(reduction.reduce(*retry));
  CHECK_NOTHROW(retry->finalize());
}

TEST_CASE("Distributed orbital target and loss metadata mismatch independently",
          "[drivers][training][orbital-pretraining][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  if (communicator.size() == 1)
    return;
  const bool last_rank = communicator.rank() == communicator.size() - 1;
  const StructuredParameterSchema schema = makeSchema();
  const StructuredParameterSnapshot parameters{schema.fingerprint(), 0,
                                                std::vector<double>(5, 0.5)};
  DistributedParameterReduction reduction(communicator, {2});

  CHECK_THROWS_WITH(reduction.preflightOrbital(
                        schema, &parameters, last_rank ? 0x9999 : 0x1234, 0x5678),
                    Catch::Matchers::ContainsSubstring("metadata mismatch"));
  CHECK_THROWS_WITH(reduction.preflightOrbital(
                        schema, &parameters, 0x1234, last_rank ? 0x9999 : 0x5678),
                    Catch::Matchers::ContainsSubstring("metadata mismatch"));

  reduction.preflightOrbital(schema, &parameters, 0x1234, 0x5678);
  auto target_mismatch = makeOrbitalContribution(
      schema, 0, communicator.rank(),
      localSampleCount(communicator.rank(), communicator.size()),
      last_rank ? 0x9999 : 0x1234, 0x5678);
  CHECK_THROWS_WITH(reduction.reduce(*target_mismatch),
                    Catch::Matchers::ContainsSubstring("metadata mismatch"));
  auto loss_mismatch = makeOrbitalContribution(
      schema, 0, communicator.rank(),
      localSampleCount(communicator.rank(), communicator.size()),
      0x1234, last_rank ? 0x9999 : 0x5678);
  CHECK_THROWS_WITH(reduction.reduce(*loss_mismatch),
                    Catch::Matchers::ContainsSubstring("metadata mismatch"));
}

TEST_CASE("Distributed metadata and candidate mismatches precede publication",
          "[drivers][training][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  if (communicator.size() == 1)
    return;

  const bool last_rank = communicator.rank() == communicator.size() - 1;
  const StructuredParameterSchema mismatched_schema =
      makeSchema(last_rank ? "distributed/different" : "distributed/toy");
  const StructuredParameterSnapshot mismatched_parameters{
      mismatched_schema.fingerprint(), 0, std::vector<double>(5, 0.5)};
  DistributedParameterReduction reduction(communicator, {2});
  CHECK_THROWS_WITH(
      reduction.preflight(mismatched_schema, &mismatched_parameters,
                          EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN),
      Catch::Matchers::ContainsSubstring("metadata mismatch"));

  const StructuredParameterSchema schema = makeSchema();
  const StructuredParameterSnapshot parameters{schema.fingerprint(), 0,
                                                std::vector<double>(5, 0.5)};
  reduction.preflight(schema, &parameters,
                      EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN);

  auto mismatched_version = makeContribution(
      schema, last_rank ? 1 : 0, communicator.rank(),
      localSampleCount(communicator.rank(), communicator.size()));
  CHECK_THROWS_WITH(reduction.reduce(*mismatched_version),
                    Catch::Matchers::ContainsSubstring("contribution metadata mismatch"));

  auto mismatched_adjoint = makeContribution(
      schema, 0, communicator.rank(),
      localSampleCount(communicator.rank(), communicator.size()),
      last_rank ? DerivativeAdjoint::HERMITIAN : DerivativeAdjoint::TRANSPOSE);
  CHECK_THROWS_WITH(reduction.reduce(*mismatched_adjoint),
                    Catch::Matchers::ContainsSubstring("contribution metadata mismatch"));

  StructuredParameterSnapshot candidate = parameters;
  if (last_rank)
    candidate.values.front() += 1.0;
  CHECK_THROWS_WITH(reduction.validateCandidate(schema, parameters, &candidate),
                    Catch::Matchers::ContainsSubstring("candidate metadata mismatch"));
}

TEST_CASE("Distributed candidate validation rejects non-finite values under fast math",
          "[drivers][training][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  const StructuredParameterSchema schema = makeSchema();
  const StructuredParameterSnapshot parameters{schema.fingerprint(), 0,
                                                std::vector<double>(5, 0.5)};
  StructuredParameterSnapshot candidate = parameters;
  candidate.values.front() = std::numeric_limits<double>::quiet_NaN();
  DistributedParameterReduction reduction(communicator, {2});

  CHECK_THROWS_WITH(reduction.validateCandidate(schema, parameters, &candidate),
                    Catch::Matchers::ContainsSubstring("invalid candidate"));
}

TEST_CASE("Distributed publication failure is reported as fatal divergence",
          "[drivers][training][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  if (communicator.size() == 1)
    return;

  DistributedParameterReduction reduction(communicator, {2});
  std::exception_ptr local_failure;
  if (communicator.rank() == communicator.size() - 1)
    local_failure = std::make_exception_ptr(std::runtime_error("publication failed"));

  CHECK_THROWS_WITH(reduction.completePublication(1, local_failure),
                    Catch::Matchers::ContainsSubstring(
                        "Fatal distributed parameter-publication divergence"));
}

TEST_CASE("Distributed high-parameter iteration preserves publication ordering",
          "[drivers][training][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  std::vector<std::string> events;
  RankProvider provider(&events);
  RankProducer producer(provider.parameterSchema(), communicator, &events);
  TrainingIterationState state;
  RankObserver observer(events);
  HighParameterTraining training(
      {}, EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN,
      DistributedParameterReduction{communicator, {2}});

  RankUpdater failing_updater(communicator, true, false, &events);
  CHECK_THROWS(training.runIteration(provider, producer, failing_updater, state, &observer));
  CHECK(provider.snapshotParameters().version == 0);
  CHECK(state.completed_iterations == 0);
  CHECK(observer.calls == 0);
  CHECK(events == std::vector<std::string>{"produce", "propose"});

  events.clear();
  if (communicator.size() > 1)
  {
    RankUpdater divergent_updater(communicator, false, true, &events);
    CHECK_THROWS_WITH(
        training.runIteration(provider, producer, divergent_updater, state, &observer),
        Catch::Matchers::ContainsSubstring("candidate metadata mismatch"));
    CHECK(provider.snapshotParameters().version == 0);
    CHECK(state.completed_iterations == 0);
    CHECK(observer.calls == 0);
    CHECK(events == std::vector<std::string>{"produce", "propose"});
    events.clear();
  }

  RankUpdater successful_updater(communicator, false, false, &events);
  const TrainingIterationResult result =
      training.runIteration(provider, producer, successful_updater, state, &observer);
  CHECK(result.completed_iteration == 1);
  CHECK(result.parameter_version == 1);
  CHECK(state.completed_iterations == 1);
  CHECK(observer.calls == 1);
  CHECK(events ==
        std::vector<std::string>{"produce", "propose", "publish", "observe"});
}

} // namespace qmcplusplus::wftrain
