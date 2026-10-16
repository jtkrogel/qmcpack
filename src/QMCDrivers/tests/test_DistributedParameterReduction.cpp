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
#include "QMCDrivers/WFTrain/HostStagedAcceleratorReduction.h"
#include "QMCDrivers/WFTrain/HighParameterTraining.h"
#include "QMCDrivers/WFTrain/MatrixFreeStochasticReconfiguration.h"
#include "QMCDrivers/WFTrain/MinimumStepStochasticReconfiguration.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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

/// Execute the abstract accelerator-copy protocol against ordinary host vectors.
class HostMemoryCopyQueue final : public AcceleratorCollectiveCopyQueue
{
public:
  void submitDeviceToHost(std::size_t,
                          DerivativeValue* destination,
                          const DerivativeValue* source,
                          std::size_t count) override
  {
    if (fail_device_to_host_)
      throw std::runtime_error("injected device-to-host failure");
    std::memcpy(destination, source, count * sizeof(DerivativeValue));
    ++device_to_host_submissions_;
  }

  void waitDeviceToHost(std::size_t) override { ++device_to_host_waits_; }

  void submitHostToDevice(std::size_t,
                          DerivativeValue* destination,
                          const DerivativeValue* source,
                          std::size_t count) override
  {
    if (fail_host_to_device_)
      throw std::runtime_error("injected host-to-device failure");
    std::memcpy(destination, source, count * sizeof(DerivativeValue));
    ++host_to_device_submissions_;
  }

  void waitHostToDevice(std::size_t) override { ++host_to_device_waits_; }

  bool drainAfterFailure() noexcept override
  {
    ++drain_count_;
    return drain_succeeds_;
  }

  bool fail_device_to_host_ = false;
  bool fail_host_to_device_ = false;
  bool drain_succeeds_ = true;
  std::size_t device_to_host_submissions_ = 0;
  std::size_t device_to_host_waits_ = 0;
  std::size_t host_to_device_submissions_ = 0;
  std::size_t host_to_device_waits_ = 0;
  std::size_t drain_count_ = 0;
};

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

/// Two-parameter schema used by the distributed SR dense oracle.
StructuredParameterSchema makeSRSchema()
{
  return {"distributed/sr",
          {{"weights", {2}, 0, 2, ParameterScalarDomain::REAL64, true,
            "weights"}}};
}

/// Stream deterministic rank-local score rows and support injected pass failures.
class DistributedSRStreamingOperator final : public StreamingDerivativeOperator
{
public:
  DistributedSRStreamingOperator(const StructuredParameterSchema& schema,
                                 std::size_t version,
                                 int rank,
                                 std::size_t sample_count,
                                 bool* fail_jvp,
                                 bool* fail_vjp)
      : schema_(schema), version_(version), rank_(rank),
        sample_count_(sample_count), plan_(schema_, version_, 1),
        fail_jvp_(fail_jvp), fail_vjp_(fail_vjp)
  {}

  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities result;
    result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
        derivativeProductBit(DerivativeProduct::SCORE_JVP);
    result.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::HERMITIAN);
    result.parameter_scalar_domain = ParameterScalarDomain::REAL64;
    result.result_scalar_domain = ParameterScalarDomain::COMPLEX128;
    result.reduction_domain = ReductionDomain::RANK_LOCAL;
    result.execution_domain = DerivativeExecutionDomain::HOST;
    result.maximum_vjp_channels = 1;
    result.maximum_parameter_chunk_size = 1;
    result.maximum_sample_tile_size = std::max<std::size_t>(sample_count_, 1);
    result.block_streaming = true;
    return result;
  }

  const StructuredParameterSchema& parameterSchema() const noexcept override
  {
    return schema_;
  }
  std::size_t parameterVersion() const noexcept override { return version_; }
  std::size_t batchOrdinal() const noexcept override
  {
    return static_cast<std::size_t>(rank_);
  }
  std::size_t sampleOffset() const noexcept override
  {
    return static_cast<std::size_t>(100 * rank_);
  }
  std::size_t sampleCount() const noexcept override { return sample_count_; }
  const ParameterChunkPlan& parameterChunkPlan() const noexcept override
  {
    return plan_;
  }

  /// Return one deterministic real score component without stored rows.
  DerivativeValue score(std::size_t sample, std::size_t parameter) const noexcept
  {
    const double identity = static_cast<double>(10 * rank_ + sample + 1);
    return parameter == 0 ? DerivativeValue{identity, 0.0}
                          : DerivativeValue{1.0 - 0.1 * identity, 0.0};
  }

protected:
  void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                    DerivativeAdjoint,
                    ParameterReductionSink& sink) const override
  {
    if (*fail_vjp_)
      throw std::runtime_error("injected distributed SR VJP failure");
    for (const ParameterChunkDescriptor& chunk : plan_.chunks())
    {
      DerivativeValue value{};
      for (std::size_t sample = 0; sample < sample_count_; ++sample)
        value += channels[0].coefficients.values[sample] *
            std::conj(score(sample, chunk.parameter_offset));
      sink.add(0, {chunk, {&value, 1}});
    }
  }

  void evaluateScoreJVP(const StructuredParameterVectorConstView& direction,
                        SampleProductSink& sink) const override
  {
    if (*fail_jvp_)
      throw std::runtime_error("injected distributed SR JVP failure");
    std::vector<DerivativeValue> products(sample_count_);
    for (std::size_t sample = 0; sample < sample_count_; ++sample)
      for (std::size_t parameter = 0; parameter < 2; ++parameter)
        products[sample] += score(sample, parameter) *
            direction.values()[parameter];
    if (!products.empty())
    {
      const SampleProductTileDescriptor descriptor{
          schema_.providerId(), schema_.fingerprint(), version_, batchOrdinal(),
          sampleOffset(), products.size(), 0};
      sink.add({descriptor, {products.data(), products.size()}});
    }
  }

private:
  StructuredParameterSchema schema_;
  std::size_t version_;
  int rank_;
  std::size_t sample_count_;
  ParameterChunkPlan plan_;
  bool* fail_jvp_;
  bool* fail_vjp_;
};

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
EnergyGradientResult makeReferenceResult(int size, int sample_rank_offset = 0)
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
      const double identity = static_cast<double>(10 * (rank + sample_rank_offset) + sample + 1);
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

/// Recording model with real preallocated ownership swaps, not a GPU positive.
class PreparedRankProvider final : public StructuredParameterProvider,
                                   public PreparedParameterPublicationContributor
{
public:
  explicit PreparedRankProvider(int group = 0)
      : schema_(makeSchema("prepared/group/" + std::to_string(group))), values_(5, 0.5 + group)
  {}

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }
  StructuredParameterSnapshot snapshotParameters() const override
  {
    return {schema_.fingerprint(), version_, values_};
  }
  std::size_t publishParameters(const StructuredParameterSnapshot&, std::size_t) override
  {
    ++legacy_calls;
    throw std::logic_error("Prepared provider must not use legacy publication");
  }

  std::unique_ptr<PreparedParameterPublication> prepareParameterPublication(
      const StructuredParameterSnapshot& candidate, std::size_t expected_version) override
  {
    ++prepare_calls;
    if (fail_preparation)
      throw std::runtime_error("injected successor preparation failure");
    if (exclusive_ || version_ != expected_version)
      throw std::logic_error("Prepared provider lost exclusive expected version");
    validateTrainingCandidate(schema_, snapshotParameters(), candidate);
    auto action = std::make_unique<Action>(*this, candidate.values, expected_version + 1);
    exclusive_ = true;
    return action;
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
    Action(PreparedRankProvider& owner, const std::vector<double>& values, std::size_t next_version)
        : owner_(owner), values_(values), next_version_(next_version)
    {}
    ~Action() override
    {
      if (!committed_)
        ++owner_.discard_calls;
      owner_.exclusive_ = false;
    }
    std::size_t commit() noexcept override
    {
      owner_.values_.swap(values_);
      owner_.version_ = next_version_;
      ++owner_.commit_calls;
      committed_ = true;
      return next_version_;
    }

  private:
    PreparedRankProvider& owner_;
    std::vector<double> values_;
    const std::size_t next_version_;
    bool committed_ = false;
  };

  StructuredParameterSchema schema_;
  std::vector<double> values_;
  std::size_t version_ = 0;
  bool exclusive_ = false;
};

/// Produce the rank-local synthetic batch through the normal training interface.
class RankProducer final : public GradientProducer
{
public:
  RankProducer(const StructuredParameterSchema& schema,
               Communicate& communicator,
               std::vector<std::string>* events = nullptr, int sample_rank_offset = 0)
      : schema_(schema), communicator_(communicator), events_(events), sample_rank_offset_(sample_rank_offset)
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
                                               communicator_.rank() + sample_rank_offset_, count);
    std::vector<DerivativeReal> weights;
    std::vector<DerivativeValue> energies;
    makeSamples(communicator_.rank() + sample_rank_offset_, count, weights, energies);
    accumulateEnergyGradientBatch(
        derivative_operator, {weights.data(), weights.size()},
        {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
        accumulator);
  }

private:
  const StructuredParameterSchema& schema_;
  Communicate& communicator_;
  std::vector<std::string>* events_;
  int sample_rank_offset_;
};

/// Deterministic updater with optional rank-local failure or candidate divergence.
class RankUpdater final : public TrainingUpdateRule
{
public:
  RankUpdater(Communicate& communicator,
              bool fail_last = false,
              bool diverge_last = false,
              std::vector<std::string>* events = nullptr, double learning_rate = 0.01)
      : communicator_(communicator), fail_last_(fail_last),
        diverge_last_(diverge_last), events_(events), learning_rate_(learning_rate)
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
      candidate.values[parameter] -= learning_rate_ * objective.gradient[parameter];
    if (diverge_last_ && communicator_.size() > 1 &&
        communicator_.rank() == communicator_.size() - 1)
      candidate.values.front() += 0.25;
    return candidate;
  }

  void proposalAccepted(const StructuredParameterSchema&, const StructuredParameterSnapshot&,
                        ParameterGradientView) noexcept override
  {
    ++accepted_recurrences;
  }
  void proposalRejected() noexcept override { ++rejected_proposals; }

  std::size_t accepted_recurrences = 0;
  std::size_t rejected_proposals = 0;

private:
  Communicate& communicator_;
  bool fail_last_;
  bool diverge_last_;
  std::vector<std::string>* events_;
  double learning_rate_;
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

TEST_CASE("Distributed accelerator-plan preflight is fixed-record and retryable",
          "[drivers][training][accelerator][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  const std::size_t rank = static_cast<std::size_t>(communicator.rank());
  const std::size_t participants = static_cast<std::size_t>(communicator.size());
  AcceleratorRankPlanInput local{
      /*rank=*/rank,
      /*participant_count=*/participants,
      /*node_id=*/rank / 2,
      /*device_id=*/rank % 2,
      /*local_sample_count=*/rank == 1 ? 0 : rank + 1,
      /*device_budget_bytes=*/std::size_t{4096},
      /*base_required_device_bytes=*/128,
      /*maximum_chunk_elements=*/rank + 2,
      /*parameter_version=*/7,
      /*model_fingerprint=*/101,
      /*precision_fingerprint=*/202,
      /*device_mpi_available=*/false};
  DistributedParameterReduction reduction(communicator, {2});
  const AcceleratorDistributedPlan plan = reduction.preflightAcceleratorPlan(
      local, /*parameter_count=*/5, /*collective_element_bytes=*/16,
      AcceleratorTransportRequest::AUTO, /*buffer_slots=*/2);

  std::size_t expected_samples = 0;
  for (std::size_t remote_rank = 0; remote_rank < participants; ++remote_rank)
    expected_samples += remote_rank == 1 ? 0 : remote_rank + 1;
  CHECK(plan.participant_count == participants);
  CHECK(plan.total_sample_count == expected_samples);
  CHECK(plan.transport == AcceleratorCollectiveTransport::HOST_STAGED_MPI);
  CHECK(plan.chunk_elements == 2);
  CHECK(plan.parameter_version == 7);
  CHECK(plan.host_staging_bytes_per_rank == 64);
  REQUIRE(plan.required_device_bytes_per_rank.size() == participants);
  for (const std::size_t required : plan.required_device_bytes_per_rank)
    CHECK(required == 192);

  if (participants > 1)
  {
    AcceleratorRankPlanInput mismatched = local;
    if (rank == participants - 1)
      mismatched.precision_fingerprint = 999;
    CHECK_THROWS_WITH(
        reduction.preflightAcceleratorPlan(
            mismatched, 5, 16, AcceleratorTransportRequest::AUTO, 2),
        Catch::Matchers::ContainsSubstring("semantic metadata mismatch"));

    // A failed preflight performs no bulk collective and leaves the same
    // coordinator immediately reusable with corrected metadata.
    CHECK_NOTHROW(reduction.preflightAcceleratorPlan(
        local, 5, 16, AcceleratorTransportRequest::AUTO, 2));
  }

  CHECK_THROWS_WITH(
      reduction.preflightAcceleratorPlan(
          local, 5, 16, AcceleratorTransportRequest::DEVICE_MPI, 2),
      Catch::Matchers::ContainsSubstring("not available on every rank"));
}

TEST_CASE("Host-staged accelerator reduction matches complex global sums",
          "[drivers][training][accelerator][host_staged][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  const std::size_t rank = static_cast<std::size_t>(communicator.rank());
  const std::size_t participants = static_cast<std::size_t>(communicator.size());
  AcceleratorRankPlanInput local{
      /*rank=*/rank,
      /*participant_count=*/participants,
      /*node_id=*/rank / 2,
      /*device_id=*/rank % 2,
      /*local_sample_count=*/rank == 1 ? 0 : rank + 1,
      /*device_budget_bytes=*/std::size_t{4096},
      /*base_required_device_bytes=*/128,
      /*maximum_chunk_elements=*/2,
      /*parameter_version=*/7,
      /*model_fingerprint=*/101,
      /*precision_fingerprint=*/202,
      /*device_mpi_available=*/false};
  DistributedParameterReduction preflight(communicator, {2});
  const AcceleratorDistributedPlan plan = preflight.preflightAcceleratorPlan(
      local, /*parameter_count=*/5, /*collective_element_bytes=*/sizeof(DerivativeValue),
      AcceleratorTransportRequest::HOST_STAGED_MPI, /*buffer_slots=*/2);

  std::array<std::vector<DerivativeValue>, 2> host_storage{
      std::vector<DerivativeValue>(plan.chunk_elements),
      std::vector<DerivativeValue>(plan.chunk_elements)};
  const std::array<AcceleratorHostStagingSlot, 2> staging{{
      {host_storage[0].data(), host_storage[0].size()},
      {host_storage[1].data(), host_storage[1].size()}}};
  HostMemoryCopyQueue queue;
  HostStagedAcceleratorReduction reduction(communicator, plan, queue, staging);

  const auto make_local_values = [&]() {
    const double factor = participants > 1 && rank == 1 ? 0.0
                                                         : static_cast<double>(rank + 1);
    std::vector<DerivativeValue> values(5);
    for (std::size_t parameter = 0; parameter < values.size(); ++parameter)
      values[parameter] = factor * DerivativeValue{
          static_cast<double>(parameter + 1), -0.125 * static_cast<double>(parameter + 1)};
    return values;
  };

  double global_factor = 0.0;
  for (std::size_t remote_rank = 0; remote_rank < participants; ++remote_rank)
    if (!(participants > 1 && remote_rank == 1))
      global_factor += static_cast<double>(remote_rank + 1);

  std::vector<DerivativeValue> values = make_local_values();
  reduction.reduceComplexVector(values.data(), values.size(), plan.fingerprint,
                                plan.parameter_version);
  for (std::size_t parameter = 0; parameter < values.size(); ++parameter)
  {
    CHECK(values[parameter].real() ==
          Catch::Approx(global_factor * static_cast<double>(parameter + 1)));
    CHECK(values[parameter].imag() ==
          Catch::Approx(-0.125 * global_factor * static_cast<double>(parameter + 1)));
  }
  CHECK(queue.device_to_host_submissions_ == 3);
  CHECK(queue.device_to_host_waits_ == 3);
  CHECK(queue.host_to_device_submissions_ == 3);
  CHECK(queue.host_to_device_waits_ == 3);
  CHECK(reduction.state().iterationState() == AcceleratorCollectiveIterationState::IDLE);

  // One rank-local copy error is converted to a uniform control-collective error
  // before any participant enters the corresponding bulk parameter collective.
  values = make_local_values();
  queue.fail_device_to_host_ = rank == participants - 1;
  CHECK_THROWS_WITH(reduction.reduceComplexVector(
                        values.data(), values.size(), plan.fingerprint,
                        plan.parameter_version),
                    Catch::Matchers::ContainsSubstring(
                        "device-to-host copy failed on rank " +
                        std::to_string(participants - 1)));
  CHECK(reduction.state().iterationState() == AcceleratorCollectiveIterationState::IDLE);
  CHECK(queue.drain_count_ == 1);

  // Reset rank-local data and the injected queue condition before a clean retry.
  values = make_local_values();
  queue.fail_device_to_host_ = false;
  reduction.reduceComplexVector(values.data(), values.size(), plan.fingerprint,
                                plan.parameter_version);
  for (std::size_t parameter = 0; parameter < values.size(); ++parameter)
    CHECK(values[parameter].real() ==
          Catch::Approx(global_factor * static_cast<double>(parameter + 1)));
}

TEST_CASE("Host-staged accelerator reduction rejects nonfinite device contributions",
          "[drivers][training][accelerator][host_staged][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  const auto rank = static_cast<std::size_t>(communicator.rank());
  const auto participants = static_cast<std::size_t>(communicator.size());
  AcceleratorRankPlanInput local{rank, participants, rank / 2, rank % 2, 1,
      std::size_t{4096}, 128, 2, 7, 101, 202, false};
  DistributedParameterReduction preflight(communicator, {2});
  const auto plan = preflight.preflightAcceleratorPlan(local, 5, sizeof(DerivativeValue),
      AcceleratorTransportRequest::HOST_STAGED_MPI, 2);
  std::array<std::vector<DerivativeValue>, 2> storage{
      std::vector<DerivativeValue>(plan.chunk_elements), std::vector<DerivativeValue>(plan.chunk_elements)};
  const std::array<AcceleratorHostStagingSlot, 2> slots{{
      {storage[0].data(), storage[0].size()}, {storage[1].data(), storage[1].size()}}};
  HostMemoryCopyQueue queue;
  HostStagedAcceleratorReduction reduction(communicator, plan, queue, slots);
  std::vector<DerivativeValue> values(5, DerivativeValue(1, -0.5));

  // Only the last rank injects a NaN, in the one-element final chunk. This
  // exercises admission after earlier chunks have already completed reductions.
  if (rank == participants - 1)
    values.back() = DerivativeValue(std::numeric_limits<double>::quiet_NaN(), 0);
  CHECK_THROWS(reduction.reduceComplexVector(values.data(), values.size(), plan.fingerprint,
                                              plan.parameter_version));
  CHECK(reduction.state().iterationState() == AcceleratorCollectiveIterationState::IDLE);
  std::fill(values.begin(), values.end(), DerivativeValue(1, -0.5));
  CHECK_NOTHROW(reduction.reduceComplexVector(values.data(), values.size(), plan.fingerprint,
                                               plan.parameter_version));
  CHECK(values.back().real() == Catch::Approx(static_cast<double>(participants)));
}

TEST_CASE("Host-staged accelerator reduction reports post-collective copy failure",
          "[drivers][training][accelerator][host_staged][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  const std::size_t rank = static_cast<std::size_t>(communicator.rank());
  const std::size_t participants = static_cast<std::size_t>(communicator.size());
  AcceleratorRankPlanInput local{
      rank, participants, rank / 2, rank % 2, rank + 1, std::size_t{4096},
      128, 2, 7, 101, 202, false};
  DistributedParameterReduction preflight(communicator, {2});
  const AcceleratorDistributedPlan plan = preflight.preflightAcceleratorPlan(
      local, 5, sizeof(DerivativeValue), AcceleratorTransportRequest::HOST_STAGED_MPI, 1);

  std::vector<DerivativeValue> staging_storage(plan.chunk_elements);
  const std::array<AcceleratorHostStagingSlot, 2> staging{{
      {staging_storage.data(), staging_storage.size()}, {nullptr, 0}}};
  HostMemoryCopyQueue queue;
  HostStagedAcceleratorReduction reduction(communicator, plan, queue, staging);
  std::vector<DerivativeValue> values(5, DerivativeValue{static_cast<double>(rank + 1), 0.0});

  queue.fail_host_to_device_ = rank == 0;
  queue.drain_succeeds_ = false;
  CHECK_THROWS_WITH(reduction.reduceComplexVector(
                        values.data(), values.size(), plan.fingerprint,
                        plan.parameter_version),
                    Catch::Matchers::ContainsSubstring(
                        "coordinator remains poisoned"));
  CHECK(reduction.state().iterationState() == AcceleratorCollectiveIterationState::FAILED);
  CHECK(queue.drain_count_ == 1);
}

TEST_CASE("Distributed matrix-free vectors are reduced once into replicated storage",
          "[drivers][training][matrix-free][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  const StructuredParameterSchema complex_schema{
      "distributed/matrix_free_complex",
      {{"weights", {5}, 0, 5, ParameterScalarDomain::COMPLEX128, true,
        "weights"}}};

  for (const std::size_t chunk_size : {std::size_t{1}, std::size_t{2}})
  {
    // Rank one deliberately contributes zero when it exists. This checks both the
    // fixed collective schedule and a non-divisible final transport chunk.
    const double rank_factor = communicator.size() > 1 && communicator.rank() == 1
        ? 0.0
        : static_cast<double>(communicator.rank() + 1);
    std::vector<DerivativeValue> values(5);
    for (std::size_t parameter = 0; parameter < values.size(); ++parameter)
      values[parameter] = rank_factor *
          DerivativeValue{static_cast<double>(parameter + 1),
                          -0.25 * static_cast<double>(parameter + 1)};

    double global_factor = 0.0;
    for (int rank = 0; rank < communicator.size(); ++rank)
      if (!(communicator.size() > 1 && rank == 1))
        global_factor += rank + 1;

    DistributedParameterReduction reduction(communicator, {chunk_size});
    CHECK(reduction.reduceParameterVector(
              complex_schema, 9, {values.data(), values.size()}) ==
          ReductionDomain::GLOBAL);
    for (std::size_t parameter = 0; parameter < values.size(); ++parameter)
    {
      CHECK(values[parameter].real() ==
            Catch::Approx(global_factor * static_cast<double>(parameter + 1)));
      CHECK(values[parameter].imag() ==
            Catch::Approx(-0.25 * global_factor *
                          static_cast<double>(parameter + 1)));
    }
  }
}

TEST_CASE("Distributed streaming SR matches one global centered dense population",
          "[drivers][training][matrix-free][sr][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  const StructuredParameterSchema schema = makeSRSchema();
  const std::size_t local_count =
      localSampleCount(communicator.rank(), communicator.size());
  bool fail_jvp = false;
  bool fail_vjp = false;
  DistributedSRStreamingOperator stream(schema, 5, communicator.rank(),
                                         local_count, &fail_jvp, &fail_vjp);
  std::vector<DerivativeReal> weights(local_count);
  for (std::size_t sample = 0; sample < local_count; ++sample)
    weights[sample] = 1.0 + 0.25 * sample;
  const StochasticReconfigurationBatch batch{
      &stream, {weights.data(), weights.size()}};
  StochasticReconfigurationOperator covariance(
      schema, 5, {&batch, 1},
      DistributedParameterReduction(communicator, {1}));

  const std::vector<DerivativeValue> direction{{0.3, 0.0}, {-0.7, 0.0}};
  const StructuredParameterVectorConstView direction_view(
      schema, 5, {direction.data(), direction.size()});
  std::vector<DerivativeValue> result(2);
  covariance.apply(direction_view, {result.data(), result.size()});

  DerivativeReal global_weight = 0.0;
  DerivativeValue global_product{};
  for (int rank = 0; rank < communicator.size(); ++rank)
  {
    const std::size_t count = localSampleCount(rank, communicator.size());
    for (std::size_t sample = 0; sample < count; ++sample)
    {
      const DerivativeReal weight = 1.0 + 0.25 * sample;
      const double identity = static_cast<double>(10 * rank + sample + 1);
      const DerivativeValue product = identity * direction[0] +
          (1.0 - 0.1 * identity) * direction[1];
      global_weight += weight;
      global_product += weight * product;
    }
  }
  const DerivativeValue mean_product = global_product / global_weight;
  std::vector<DerivativeValue> expected(2);
  for (int rank = 0; rank < communicator.size(); ++rank)
  {
    const std::size_t count = localSampleCount(rank, communicator.size());
    for (std::size_t sample = 0; sample < count; ++sample)
    {
      const DerivativeReal weight = 1.0 + 0.25 * sample;
      const double identity = static_cast<double>(10 * rank + sample + 1);
      const DerivativeValue score0{identity, 0.0};
      const DerivativeValue score1{1.0 - 0.1 * identity, 0.0};
      const DerivativeValue product = score0 * direction[0] +
          score1 * direction[1];
      expected[0] += std::conj(score0) * (weight / global_weight) *
          (product - mean_product);
      expected[1] += std::conj(score1) * (weight / global_weight) *
          (product - mean_product);
    }
  }
  CHECK(result[0].real() ==
        Catch::Approx(expected[0].real()).epsilon(1.0e-12).margin(1.0e-12));
  CHECK(result[1].real() ==
        Catch::Approx(expected[1].real()).epsilon(1.0e-12).margin(1.0e-12));
  CHECK(result[0].imag() == 0.0);
  CHECK(result[1].imag() == 0.0);

  // Each injected pass failure is reported uniformly before the corresponding
  // data collective. Reusing the same prepared action afterwards must remain safe.
  fail_jvp = communicator.rank() == 0;
  CHECK_THROWS(covariance.apply(direction_view, {result.data(), result.size()}));
  fail_jvp = false;
  CHECK_NOTHROW(covariance.apply(direction_view, {result.data(), result.size()}));
  int vjp_failure_rank = communicator.size() - 1;
  while (vjp_failure_rank > 0 &&
         localSampleCount(vjp_failure_rank, communicator.size()) == 0)
    --vjp_failure_rank;
  fail_vjp = communicator.rank() == vjp_failure_rank;
  CHECK_THROWS(covariance.apply(direction_view, {result.data(), result.size()}));
  fail_vjp = false;
  CHECK_NOTHROW(covariance.apply(direction_view, {result.data(), result.size()}));
}

TEST_CASE("Distributed minSR handles uneven and zero-sample ranks",
          "[drivers][training][minsr][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  const StructuredParameterSchema schema = makeSRSchema();
  constexpr std::size_t version = 7;
  const std::size_t local_count =
      localSampleCount(communicator.rank(), communicator.size());
  bool fail_jvp = false;
  bool fail_vjp = false;
  DistributedSRStreamingOperator stream(schema, version, communicator.rank(),
                                         local_count, &fail_jvp, &fail_vjp);
  std::vector<DerivativeReal> weights(local_count);
  std::vector<DerivativeValue> energies(local_count);
  for (std::size_t sample = 0; sample < local_count; ++sample)
  {
    const double identity = static_cast<double>(10 * communicator.rank() + sample + 1);
    weights[sample] = 1.0 + 0.25 * sample;
    energies[sample] = {-1.0 + 0.05 * identity, 0.0};
  }
  const ScoreCovarianceBatch batch{
      &stream, {weights.data(), weights.size()},
      {energies.data(), energies.size()}};
  const DerivativeArrayView<const ScoreCovarianceBatch> local_batches =
      local_count == 0
      ? DerivativeArrayView<const ScoreCovarianceBatch>{}
      : DerivativeArrayView<const ScoreCovarianceBatch>{&batch, 1};
  ScoreCovarianceFactor factor(
      schema, version, local_batches, ScoreSampleDomain::REAL,
      DistributedParameterReduction(communicator, {1}));

  DerivativeReal global_weight = 0.0;
  DerivativeReal global_energy_sum = 0.0;
  for (int rank = 0; rank < communicator.size(); ++rank)
    for (std::size_t sample = 0; sample < localSampleCount(rank, communicator.size()); ++sample)
    {
      const double identity = static_cast<double>(10 * rank + sample + 1);
      const double weight = 1.0 + 0.25 * sample;
      global_weight += weight;
      global_energy_sum += weight * (-1.0 + 0.05 * identity);
    }
  const double mean_energy = global_energy_sum / global_weight;

  std::array<double, 4> covariance{};
  std::array<double, 2> gradient{};
  std::array<double, 2> mean_score{};
  for (int rank = 0; rank < communicator.size(); ++rank)
    for (std::size_t sample = 0; sample < localSampleCount(rank, communicator.size()); ++sample)
    {
      const double identity = static_cast<double>(10 * rank + sample + 1);
      const double weight = 1.0 + 0.25 * sample;
      mean_score[0] += weight * identity / global_weight;
      mean_score[1] += weight * (1.0 - 0.1 * identity) / global_weight;
    }
  for (int rank = 0; rank < communicator.size(); ++rank)
    for (std::size_t sample = 0; sample < localSampleCount(rank, communicator.size()); ++sample)
    {
      const double identity = static_cast<double>(10 * rank + sample + 1);
      const double weight = 1.0 + 0.25 * sample;
      const double q = weight / global_weight;
      const std::array<double, 2> centered{
          identity - mean_score[0], 1.0 - 0.1 * identity - mean_score[1]};
      const double energy = -1.0 + 0.05 * identity;
      for (std::size_t row = 0; row < 2; ++row)
      {
        gradient[row] += 2.0 * q * centered[row] * (energy - mean_energy);
        for (std::size_t column = 0; column < 2; ++column)
          covariance[2 * row + column] += q * centered[row] * centered[column];
      }
    }

  const std::vector<DerivativeReal> streamed_gradient = factor.scoreCovarianceGradient();
  CHECK(streamed_gradient[0] == Catch::Approx(gradient[0]).epsilon(1.0e-11).margin(1.0e-11));
  CHECK(streamed_gradient[1] == Catch::Approx(gradient[1]).epsilon(1.0e-11).margin(1.0e-11));
  const double damping = 0.2;
  const double a00 = covariance[0] + damping;
  const double a01 = covariance[1];
  const double a10 = covariance[2];
  const double a11 = covariance[3] + damping;
  const double determinant = a00 * a11 - a01 * a10;
  const std::array<double, 2> expected{
      (a11 * gradient[0] - a01 * gradient[1]) / determinant,
      (-a10 * gradient[0] + a00 * gradient[1]) / determinant};

  MinimumStepSampleOperator sample_operator(factor, damping, 1.0);
  KrylovSolverControl control;
  control.relative_tolerance = 1.0e-11;
  control.maximum_iterations = 30;
  control.maximum_operator_applications = 31;
  const auto solve = solveDistributedSamplePCG(
      sample_operator, factor.residual(), control);
  REQUIRE(solve.converged);
  std::array<DerivativeValue, 2> direction{};
  factor.applyAdjoint({solve.solution.data(), solve.solution.size()},
                      {direction.data(), direction.size()});
  CHECK(direction[0].real() == Catch::Approx(expected[0]).epsilon(1.0e-9).margin(1.0e-10));
  CHECK(direction[1].real() == Catch::Approx(expected[1]).epsilon(1.0e-9).margin(1.0e-10));
  CHECK(factor.localSampleVectorSize() == local_count);
  std::size_t expected_global_count = 0;
  for (int rank = 0; rank < communicator.size(); ++rank)
    expected_global_count += localSampleCount(rank, communicator.size());
  CHECK(factor.globalSampleVectorSize() == expected_global_count);
}

TEST_CASE("Distributed matrix-free vector failures precede data collectives",
          "[drivers][training][matrix-free][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  const bool last_rank = communicator.rank() == communicator.size() - 1;
  DistributedParameterReduction reduction(communicator, {2});

  const StructuredParameterSchema schema = makeSchema();
  std::vector<DerivativeValue> values(5, DerivativeValue{1.0, 0.0});
  std::exception_ptr local_failure;
  if (communicator.rank() == 0)
    local_failure =
        std::make_exception_ptr(std::runtime_error("matrix-free action failed"));
  CHECK_THROWS(reduction.reduceParameterVector(
      schema, 0, {values.data(), values.size()}, local_failure));

  // An imaginary REAL64 contribution must fail before reduction even if two ranks
  // would otherwise cancel it to an apparently valid real result.
  if (communicator.size() > 1)
  {
    values.assign(5, DerivativeValue{1.0, 0.0});
    if (communicator.rank() == 0)
      values.front() = {1.0, 1.0};
    else if (communicator.rank() == 1)
      values.front() = {1.0, -1.0};
    CHECK_THROWS_WITH(
        reduction.reduceParameterVector(schema, 0,
                                        {values.data(), values.size()}),
        Catch::Matchers::ContainsSubstring("scalar-domain violation"));
  }

  if (communicator.size() > 1)
  {
    const StructuredParameterSchema mismatched =
        makeSchema(last_rank ? "distributed/matrix_free_mismatch" :
                               "distributed/toy");
    values.assign(5, DerivativeValue{1.0, 0.0});
    CHECK_THROWS_WITH(
        reduction.reduceParameterVector(mismatched, 0,
                                        {values.data(), values.size()}),
        Catch::Matchers::ContainsSubstring("metadata mismatch"));
  }

  // The same reduction owner remains usable after every pre-collective failure.
  values.assign(5, DerivativeValue{static_cast<double>(communicator.rank() + 1),
                                   0.0});
  CHECK_NOTHROW(reduction.reduceParameterVector(
      schema, 0, {values.data(), values.size()}));
}

TEST_CASE("Distributed matrix-free post-collective failures poison the result",
          "[drivers][training][matrix-free][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  if (communicator.size() < 2)
    return;

  const StructuredParameterSchema schema = makeSchema();
  std::vector<DerivativeValue> values(
      5, DerivativeValue{std::numeric_limits<double>::max(), 0.0});
  DistributedParameterReduction reduction(communicator, {2});
  CHECK_THROWS_WITH(
      reduction.reduceParameterVector(schema, 0, {values.data(), values.size()}),
      Catch::Matchers::ContainsSubstring("non-finite sum"));
  for (const DerivativeValue value : values)
  {
    CHECK_FALSE(isFiniteTrainingReal(value.real()));
    CHECK_FALSE(isFiniteTrainingReal(value.imag()));
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

TEST_CASE("Distributed prepared publication rejects before commit and retries",
          "[drivers][training][accelerator][prepared_publication][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  PreparedRankProvider provider;
  RankProducer producer(provider.parameterSchema(), communicator);
  RankUpdater updater(communicator);
  TrainingIterationState state;
  std::vector<std::string> events;
  RankObserver observer(events);
  HighParameterTraining training({}, EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN,
                                DistributedParameterReduction{communicator, {2}});
  const auto original = provider.snapshotParameters();

  if (communicator.size() > 1)
  {
    DistributedParameterReduction reduction(communicator, {2});
    CHECK_THROWS_WITH(
        reduction.validatePublicationReadiness(0, communicator.rank() == communicator.size() - 1),
        Catch::Matchers::ContainsSubstring("readiness metadata or decision preparation failed"));
    CHECK_NOTHROW(reduction.validatePublicationReadiness(0, true));
  }

  provider.fail_preparation = communicator.rank() == communicator.size() - 1;
  CHECK_THROWS(training.runIteration(provider, producer, updater, state, &observer));
  CHECK(provider.snapshotParameters().values == original.values);
  CHECK(provider.snapshotParameters().version == original.version);
  CHECK(provider.commit_calls == 0);
  CHECK(provider.legacy_calls == 0);
  CHECK(provider.discard_calls == (provider.fail_preparation ? 0 : 1));
  CHECK(state.completed_iterations == 0);
  CHECK(state.schema_fingerprint.empty());
  CHECK(updater.accepted_recurrences == 0);
  CHECK(updater.rejected_proposals == 1);
  CHECK(observer.calls == 0);

  provider.fail_preparation = false;
  const auto result = training.runIteration(provider, producer, updater, state, &observer);
  CHECK(result.parameter_version == 1);
  CHECK(provider.snapshotParameters().version == 1);
  CHECK(provider.prepare_calls == 2);
  CHECK(provider.commit_calls == 1);
  CHECK(provider.legacy_calls == 0);
  CHECK(state.completed_iterations == 1);
  CHECK(updater.accepted_recurrences == 1);
  CHECK(updater.rejected_proposals == 1);
  CHECK(observer.calls == 1);
  const auto reference = makeReferenceResult(communicator.size());
  const auto accepted = provider.snapshotParameters();
  for (std::size_t parameter = 0; parameter < accepted.values.size(); ++parameter)
    CHECK(accepted.values[parameter] ==
          Catch::Approx(original.values[parameter] - 0.01 * reference.gradient[parameter]));
}

TEST_CASE("Prepared publication and objective collectives stay inside ensemble groups",
          "[drivers][training][accelerator][prepared_publication][subgroup][mpi]")
{
  Communicate& world = *OHMMS::Controller;
  if (world.size() < 4)
    return;

  Communicate group(world, 2);
  const int group_id = group.getGroupID();
  const double rate = 0.01 * (group_id + 1);
  PreparedRankProvider provider(group_id);
  RankProducer producer(provider.parameterSchema(), group, nullptr, 10 * group_id);
  RankUpdater updater(group, false, false, nullptr, rate);
  TrainingIterationState state;
  std::vector<std::string> events;
  RankObserver observer(events);
  HighParameterTraining training({}, EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN,
                                DistributedParameterReduction{group, {2}});
  const auto original = provider.snapshotParameters();
  provider.fail_preparation = group_id == 0 && group.rank() == group.size() - 1;
  if (group_id == 0)
  {
    CHECK_THROWS(training.runIteration(provider, producer, updater, state, &observer));
    CHECK(provider.snapshotParameters().values == original.values);
    CHECK(provider.snapshotParameters().version == 0);
    CHECK(provider.commit_calls == 0);
    CHECK(state.completed_iterations == 0);
    CHECK(updater.accepted_recurrences == 0);
    CHECK(observer.calls == 0);
  }
  else
  {
    CHECK_NOTHROW(training.runIteration(provider, producer, updater, state, &observer));
    CHECK(provider.snapshotParameters().version == 1);
    CHECK(provider.commit_calls == 1);
    CHECK(updater.accepted_recurrences == 1);
    CHECK(observer.calls == 1);
  }

  // Both groups now retry independently; group 1 advances for the second time.
  provider.fail_preparation = false;
  const auto result = training.runIteration(provider, producer, updater, state, &observer);
  const std::size_t expected_updates = group_id == 0 ? 1 : 2;
  CHECK(result.parameter_version == expected_updates);
  CHECK(provider.commit_calls == expected_updates);
  CHECK(state.completed_iterations == expected_updates);
  CHECK(updater.accepted_recurrences == expected_updates);
  CHECK(provider.legacy_calls == 0);
  const auto reference = makeReferenceResult(group.size(), 10 * group_id);
  const auto accepted = provider.snapshotParameters();
  for (std::size_t parameter = 0; parameter < accepted.values.size(); ++parameter)
  {
    CHECK(result.objective.gradient[parameter] == Catch::Approx(reference.gradient[parameter]));
    CHECK(accepted.values[parameter] ==
          Catch::Approx(original.values[parameter] - expected_updates * rate * reference.gradient[parameter]));
  }
}

} // namespace qmcplusplus::wftrain
