//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_MatrixFreeStochasticReconfiguration.cpp
 * @brief Dense-oracle tests for streaming matrix-free stochastic reconfiguration.
 */

#include "QMCDrivers/WFTrain/MatrixFreeStochasticReconfiguration.h"
#include "Utilities/for_testing/Catch2Approx.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <complex>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Construct a homogeneous synthetic schema, optionally freezing its final entry.
StructuredParameterSchema makeSRSchema(std::string provider,
                                       std::size_t count,
                                       ParameterScalarDomain domain,
                                       bool freeze_last = false)
{
  std::vector<ParameterBlockDescriptor> blocks;
  const std::size_t trainable = freeze_last ? count - 1 : count;
  blocks.push_back({"trainable", {trainable}, 0, trainable, domain, true,
                    "weights"});
  if (freeze_last)
    blocks.push_back({"frozen", {1}, trainable, 1, domain, false, "frozen"});
  return {std::move(provider), std::move(blocks)};
}

/** Deterministic PsiFormer-shaped score producer with bounded JVP/VJP streams.
 *
 * Rows are sample-major score derivatives. The class exposes precisely the public
 * product contract consumed by the production SR operator.
 */
class ScoreRowStreamingOperator final : public StreamingDerivativeOperator
{
public:
  ScoreRowStreamingOperator(const StructuredParameterSchema& schema,
                            std::size_t version,
                            std::size_t batch_ordinal,
                            std::size_t sample_offset,
                            std::vector<std::vector<DerivativeValue>> rows,
                            bool* fail_jvp = nullptr,
                            bool* fail_vjp = nullptr,
                            bool include_frozen_chunks = false)
      : schema_(schema), version_(version), batch_ordinal_(batch_ordinal),
        sample_offset_(sample_offset), rows_(std::move(rows)),
        plan_(schema_, version_, 2,
              include_frozen_chunks ? FrozenBlockPolicy::INCLUDE_FOR_DIAGNOSTICS
                                    : FrozenBlockPolicy::EXCLUDE),
        fail_jvp_(fail_jvp), fail_vjp_(fail_vjp)
  {
    for (const auto& row : rows_)
      REQUIRE(row.size() == schema_.parameterCount());
  }

  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities result;
    result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
        derivativeProductBit(DerivativeProduct::SCORE_JVP);
    result.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::TRANSPOSE) |
        derivativeAdjointBit(DerivativeAdjoint::HERMITIAN);
    result.parameter_scalar_domain = schema_.blocks().front().scalar_domain;
    result.result_scalar_domain = ParameterScalarDomain::COMPLEX128;
    result.reduction_domain = ReductionDomain::CROWD_LOCAL;
    result.execution_domain = DerivativeExecutionDomain::HOST;
    result.maximum_vjp_channels = 1;
    result.maximum_parameter_chunk_size = 2;
    result.maximum_sample_tile_size = std::max<std::size_t>(rows_.size(), 1);
    result.block_streaming = true;
    return result;
  }

  const StructuredParameterSchema& parameterSchema() const noexcept override
  {
    return schema_;
  }

  std::size_t parameterVersion() const noexcept override { return version_; }
  std::size_t batchOrdinal() const noexcept override { return batch_ordinal_; }
  std::size_t sampleOffset() const noexcept override { return sample_offset_; }
  std::size_t sampleCount() const noexcept override { return rows_.size(); }
  const ParameterChunkPlan& parameterChunkPlan() const noexcept override
  {
    return plan_;
  }

protected:
  void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                    DerivativeAdjoint adjoint,
                    ParameterReductionSink& sink) const override
  {
    if (fail_vjp_ && *fail_vjp_)
      throw std::runtime_error("injected SR VJP failure");
    for (const ParameterChunkDescriptor& chunk : plan_.chunks())
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
      {
        std::vector<DerivativeValue> values(chunk.count);
        for (std::size_t local = 0; local < chunk.count; ++local)
        {
          const std::size_t parameter = chunk.parameter_offset + local;
          for (std::size_t sample = 0; sample < rows_.size(); ++sample)
          {
            const DerivativeValue derivative = adjoint == DerivativeAdjoint::HERMITIAN
                ? std::conj(rows_[sample][parameter])
                : rows_[sample][parameter];
            values[local] += channels[channel].coefficients.values[sample] *
                derivative;
          }
        }
        sink.add(channel, {chunk, {values.data(), values.size()}});
      }
  }

  void evaluateScoreJVP(const StructuredParameterVectorConstView& direction,
                        SampleProductSink& sink) const override
  {
    if (fail_jvp_ && *fail_jvp_)
      throw std::runtime_error("injected SR JVP failure");
    std::vector<DerivativeValue> values(rows_.size());
    for (std::size_t sample = 0; sample < rows_.size(); ++sample)
      for (std::size_t parameter = 0; parameter < rows_[sample].size(); ++parameter)
        values[sample] += rows_[sample][parameter] * direction.values()[parameter];
    if (!values.empty())
    {
      const SampleProductTileDescriptor descriptor{
          schema_.providerId(), schema_.fingerprint(), version_, batch_ordinal_,
          sample_offset_, values.size(), 0};
      sink.add({descriptor, {values.data(), values.size()}});
    }
  }

private:
  StructuredParameterSchema schema_;
  std::size_t version_;
  std::size_t batch_ordinal_;
  std::size_t sample_offset_;
  std::vector<std::vector<DerivativeValue>> rows_;
  ParameterChunkPlan plan_;
  bool* fail_jvp_;
  bool* fail_vjp_;
};

/// Assemble the tiny weighted centered covariance used only as an independent oracle.
std::vector<DerivativeValue> denseCovarianceAction(
    const std::vector<std::vector<DerivativeValue>>& rows,
    const std::vector<DerivativeReal>& weights,
    const std::vector<DerivativeValue>& direction,
    ParameterScalarDomain domain,
    std::size_t trainable_count)
{
  const std::size_t parameter_count = direction.size();
  DerivativeReal total_weight = 0.0;
  DerivativeValue weighted_product{};
  std::vector<DerivativeValue> products(rows.size());
  for (std::size_t sample = 0; sample < rows.size(); ++sample)
  {
    for (std::size_t parameter = 0; parameter < trainable_count; ++parameter)
      products[sample] += rows[sample][parameter] * direction[parameter];
    total_weight += weights[sample];
    weighted_product += weights[sample] * products[sample];
  }
  const DerivativeValue mean = weighted_product / total_weight;
  std::vector<DerivativeValue> result(parameter_count);
  for (std::size_t parameter = 0; parameter < trainable_count; ++parameter)
    for (std::size_t sample = 0; sample < rows.size(); ++sample)
      result[parameter] += std::conj(rows[sample][parameter]) *
          (weights[sample] / total_weight) * (products[sample] - mean);
  if (domain == ParameterScalarDomain::REAL64)
    for (DerivativeValue& value : result)
      value = {value.real(), 0.0};
  return result;
}

/// Compare one derivative scalar with tight absolute and relative tolerances.
void checkClose(DerivativeValue actual, DerivativeValue expected)
{
  CHECK(actual.real() == Catch::Approx(expected.real()).epsilon(2e-11).margin(2e-12));
  CHECK(actual.imag() == Catch::Approx(expected.imag()).epsilon(2e-11).margin(2e-12));
}

/** Mimic a large positive-semidefinite contraction with a tiny negative residue.
 *
 * The rank-one outer product uses an alternating vector, so its action on an all-one
 * direction cancels over P terms.  The tiny negative tail models accumulated
 * floating-point error rather than a physically relevant negative metric mode.
 */
class CancellationSensitiveCovariance final : public MatrixFreeLinearOperator
{
public:
  CancellationSensitiveCovariance(const StructuredParameterSchema& schema,
                                  std::size_t version)
      : MatrixFreeLinearOperator(schema, version, ReductionDomain::GLOBAL, true)
  {}

protected:
  void evaluate(const StructuredParameterVectorConstView& direction,
                DerivativeArrayView<DerivativeValue> result) const override
  {
    const std::size_t split = result.size() / 2;
    DerivativeValue projection{};
    for (std::size_t parameter = 0; parameter < result.size(); ++parameter)
      projection += (parameter < split ? 1.0 : -1.0) *
          direction.values()[parameter];
    for (std::size_t parameter = 0; parameter < result.size(); ++parameter)
      result[parameter] = (parameter < split ? 1.0 : -1.0) * projection;
    result[result.size() - 1] -=
        1.0e-11 * direction.values()[direction.values().size() - 1];
  }
};

} // namespace

TEST_CASE("Streaming SR covariance matches a weighted dense oracle",
          "[drivers][wftrain][sr]")
{
  const StructuredParameterSchema schema =
      makeSRSchema("sr/real", 3, ParameterScalarDomain::REAL64);
  const std::vector<std::vector<DerivativeValue>> rows0{
      {{1.0, 0.0}, {2.0, 0.0}, {-0.5, 0.0}},
      {{2.0, 0.0}, {-1.0, 0.0}, {1.0, 0.0}}};
  const std::vector<std::vector<DerivativeValue>> rows1{
      {{-0.25, 0.0}, {1.5, 0.0}, {3.0, 0.0}}};
  ScoreRowStreamingOperator batch0(schema, 7, 2, 10, rows0);
  ScoreRowStreamingOperator batch1(schema, 7, 3, 20, rows1);
  const std::vector<DerivativeReal> weights0{1.0, 2.0};
  const std::vector<DerivativeReal> weights1{3.0};
  const std::array<StochasticReconfigurationBatch, 2> batches{{
      {&batch0, {weights0.data(), weights0.size()}},
      {&batch1, {weights1.data(), weights1.size()}}}};
  StochasticReconfigurationOperator covariance(
      schema, 7, {batches.data(), batches.size()});

  const std::vector<DerivativeValue> direction{{0.7, 0.0}, {-0.3, 0.0},
                                                {0.2, 0.0}};
  std::vector<DerivativeValue> result(3);
  const StructuredParameterVectorConstView direction_view(
      schema, 7, {direction.data(), direction.size()});
  covariance.apply(direction_view, {result.data(), result.size()});

  std::vector<std::vector<DerivativeValue>> rows = rows0;
  rows.insert(rows.end(), rows1.begin(), rows1.end());
  const std::vector<DerivativeReal> weights{1.0, 2.0, 3.0};
  const std::vector<DerivativeValue> expected = denseCovarianceAction(
      rows, weights, direction, ParameterScalarDomain::REAL64, 3);
  for (std::size_t parameter = 0; parameter < result.size(); ++parameter)
    checkClose(result[parameter], expected[parameter]);

  const auto storage = covariance.storageDiagnostics();
  CHECK(storage.parameter_count == 3);
  CHECK(storage.local_sample_count == 3);
  CHECK(storage.real_sample_values == 3);
  CHECK(storage.complex_sample_values == 3);
  CHECK(storage.complex_parameter_vectors == 2);
  CHECK(storage.retained_numeric_bytes ==
        3 * sizeof(DerivativeReal) + 3 * sizeof(DerivativeValue) +
            6 * sizeof(DerivativeValue));

  // A common rescaling of all weights must leave the normalized action invariant.
  const std::vector<DerivativeReal> scaled0{7.0, 14.0};
  const std::vector<DerivativeReal> scaled1{21.0};
  const std::array<StochasticReconfigurationBatch, 2> scaled_batches{{
      {&batch0, {scaled0.data(), scaled0.size()}},
      {&batch1, {scaled1.data(), scaled1.size()}}}};
  StochasticReconfigurationOperator scaled_covariance(
      schema, 7, {scaled_batches.data(), scaled_batches.size()});
  std::vector<DerivativeValue> scaled_result(3);
  scaled_covariance.apply(direction_view,
                          {scaled_result.data(), scaled_result.size()});
  for (std::size_t parameter = 0; parameter < result.size(); ++parameter)
    checkClose(scaled_result[parameter], result[parameter]);
}

TEST_CASE("Streaming SR applies a two-sided frozen-parameter projection",
          "[drivers][wftrain][sr]")
{
  const StructuredParameterSchema schema =
      makeSRSchema("sr/frozen", 3, ParameterScalarDomain::REAL64, true);
  const std::vector<std::vector<DerivativeValue>> rows{
      {{1.0, 0.0}, {0.0, 0.0}, {100.0, 0.0}},
      {{0.0, 0.0}, {2.0, 0.0}, {-80.0, 0.0}},
      {{2.0, 0.0}, {-1.0, 0.0}, {25.0, 0.0}}};
  ScoreRowStreamingOperator stream(schema, 4, 0, 0, rows, nullptr, nullptr,
                                   true);
  const std::vector<DerivativeReal> weights{1.0, 1.0, 1.0};
  const StochasticReconfigurationBatch batch{
      &stream, {weights.data(), weights.size()}};
  StochasticReconfigurationOperator covariance(schema, 4, {&batch, 1});

  const std::vector<DerivativeValue> direction{{0.5, 0.0}, {-0.25, 0.0},
                                                {999.0, 0.0}};
  std::vector<DerivativeValue> result(3);
  const StructuredParameterVectorConstView direction_view(
      schema, 4, {direction.data(), direction.size()});
  covariance.apply(direction_view, {result.data(), result.size()});
  const auto expected = denseCovarianceAction(
      rows, weights, direction, ParameterScalarDomain::REAL64, 2);
  checkClose(result[0], expected[0]);
  checkClose(result[1], expected[1]);
  CHECK(result[2] == DerivativeValue{});
}

TEST_CASE("Streaming SR preserves complex Hermitian covariance algebra",
          "[drivers][wftrain][sr]")
{
  const StructuredParameterSchema schema =
      makeSRSchema("sr/complex", 2, ParameterScalarDomain::COMPLEX128);
  const std::vector<std::vector<DerivativeValue>> rows{
      {{1.0, 0.5}, {-0.2, 0.7}},
      {{-0.3, 0.4}, {1.1, -0.6}},
      {{0.8, -0.2}, {0.5, 0.3}}};
  ScoreRowStreamingOperator stream(schema, 9, 0, 0, rows);
  const std::vector<DerivativeReal> weights{0.5, 1.5, 2.0};
  const StochasticReconfigurationBatch batch{
      &stream, {weights.data(), weights.size()}};
  StochasticReconfigurationOperator covariance(schema, 9, {&batch, 1});
  const std::vector<DerivativeValue> direction{{0.4, -0.3}, {-0.1, 0.6}};
  std::vector<DerivativeValue> result(2);
  const StructuredParameterVectorConstView direction_view(
      schema, 9, {direction.data(), direction.size()});
  covariance.apply(direction_view, {result.data(), result.size()});
  const auto expected = denseCovarianceAction(
      rows, weights, direction, ParameterScalarDomain::COMPLEX128, 2);
  checkClose(result[0], expected[0]);
  checkClose(result[1], expected[1]);
  const DerivativeValue quadratic = parameterVectorHermitianDot(
      schema, {direction.data(), direction.size()}, {result.data(), result.size()});
  CHECK(quadratic.real() >= 0.0);
  CHECK(std::abs(quadratic.imag()) < 1.0e-12);
}

TEST_CASE("Damped SR update uses task-19 PCG and transactional trust bounds",
          "[drivers][wftrain][sr]")
{
  const StructuredParameterSchema schema =
      makeSRSchema("sr/update", 2, ParameterScalarDomain::REAL64);
  const std::vector<std::vector<DerivativeValue>> rows{
      {{1.0, 0.0}, {0.0, 0.0}},
      {{0.0, 0.0}, {2.0, 0.0}},
      {{2.0, 0.0}, {-1.0, 0.0}}};
  ScoreRowStreamingOperator stream(schema, 3, 0, 0, rows);
  const std::vector<DerivativeReal> weights{1.0, 2.0, 1.0};
  const StochasticReconfigurationBatch batch{
      &stream, {weights.data(), weights.size()}};
  StochasticReconfigurationOperator covariance(schema, 3, {&batch, 1});
  const std::vector<DerivativeReal> preconditioner_diagonal{1.0, 1.0};
  DiagonalPreconditioner preconditioner(
      schema, 3, {preconditioner_diagonal.data(), preconditioner_diagonal.size()},
      1.0e-8);

  StochasticReconfigurationUpdateControl control;
  control.learning_rate = 0.4;
  control.initial_damping = 0.2;
  control.maximum_damping_attempts = 2;
  control.maximum_update_norm = 0.05;
  control.krylov.relative_tolerance = 1.0e-12;
  control.krylov.maximum_iterations = 8;
  control.krylov.maximum_operator_applications = 9;
  StochasticReconfigurationUpdateRule update(covariance, preconditioner, control);

  const StructuredParameterSnapshot parameters{schema.fingerprint(), 3,
                                                {0.5, -0.25}};
  const std::vector<DerivativeReal> gradient{1.0, -0.4};
  const ParameterGradientView objective{
      schema.fingerprint(), 3, ReductionDomain::GLOBAL,
      {gradient.data(), gradient.size()}};
  const StructuredParameterSnapshot candidate =
      update.propose(schema, parameters, objective);
  REQUIRE(update.lastDiagnostics());
  CHECK(update.lastDiagnostics()->solve.converged);
  CHECK(update.lastDiagnostics()->damping_attempts == 1);
  CHECK(update.lastDiagnostics()->applied_scale < 1.0);
  const double displacement = std::hypot(candidate.values[0] - parameters.values[0],
                                         candidate.values[1] - parameters.values[1]);
  CHECK(displacement == Catch::Approx(0.05).epsilon(1.0e-10));
  CHECK(update.hasLiveProposal());
  CHECK_THROWS(update.propose(schema, parameters, objective));
  update.proposalRejected();
  CHECK_FALSE(update.hasLiveProposal());
  CHECK_NOTHROW(update.propose(schema, parameters, objective));
  update.proposalAccepted(schema, parameters, objective);
  CHECK_FALSE(update.hasLiveProposal());

  StochasticReconfigurationUpdateControl fail_control = control;
  fail_control.initial_damping = 0.0;
  fail_control.maximum_damping_attempts = 3;
  fail_control.krylov.maximum_iterations = 0;
  StochasticReconfigurationUpdateRule failing(covariance, preconditioner,
                                               fail_control);
  CHECK_THROWS(failing.propose(schema, parameters, objective));
  REQUIRE(failing.lastDiagnostics());
  CHECK(failing.lastDiagnostics()->damping_attempts == 3);
  CHECK(failing.lastDiagnostics()->damping == Catch::Approx(1.0e-7));
  CHECK_FALSE(failing.hasLiveProposal());
}

TEST_CASE("SR metric validation scales its cancellation bound with parameter count",
          "[drivers][wftrain][sr]")
{
  constexpr std::size_t parameter_count = 4096;
  const StructuredParameterSchema schema = makeSRSchema(
      "sr/metric_roundoff", parameter_count, ParameterScalarDomain::REAL64);
  CancellationSensitiveCovariance covariance(schema, 12);
  IdentityPreconditioner preconditioner(schema, 12);

  StochasticReconfigurationUpdateControl control;
  control.learning_rate = 1.0e-3;
  control.initial_damping = 2.0;
  control.maximum_damping_attempts = 1;
  control.krylov.relative_tolerance = 1.0e-12;
  control.krylov.maximum_iterations = 8;
  control.krylov.maximum_operator_applications = 9;
  StochasticReconfigurationUpdateRule update(covariance, preconditioner, control);

  const StructuredParameterSnapshot parameters{
      schema.fingerprint(), 12, std::vector<DerivativeReal>(parameter_count)};
  std::vector<DerivativeReal> gradient(parameter_count);
  std::fill(gradient.begin(), gradient.end(), control.initial_damping);
  gradient.back() -= 1.0e-11;
  const ParameterGradientView objective{
      schema.fingerprint(), 12, ReductionDomain::GLOBAL,
      {gradient.data(), gradient.size()}};

  const StructuredParameterSnapshot candidate =
      update.propose(schema, parameters, objective);
  REQUIRE(update.lastDiagnostics());
  CHECK(update.lastDiagnostics()->solve.converged);
  CHECK(update.lastDiagnostics()->metric_norm == 0.0);
  CHECK(candidate.values.front() == Catch::Approx(-control.learning_rate));
  CHECK(candidate.values.back() == Catch::Approx(-control.learning_rate));
  update.proposalRejected();
}

TEST_CASE("Streaming SR rejects invalid batch and objective contracts",
          "[drivers][wftrain][sr]")
{
  const StructuredParameterSchema schema =
      makeSRSchema("sr/contracts", 2, ParameterScalarDomain::REAL64);
  const std::vector<std::vector<DerivativeValue>> rows{
      {{1.0, 0.0}, {2.0, 0.0}}};
  bool fail_jvp = false;
  bool fail_vjp = false;
  ScoreRowStreamingOperator stream(schema, 2, 0, 0, rows, &fail_jvp, &fail_vjp);

  const std::vector<DerivativeReal> negative{-1.0};
  StochasticReconfigurationBatch negative_batch{
      &stream, {negative.data(), negative.size()}};
  CHECK_THROWS(StochasticReconfigurationOperator(schema, 2,
                                                  {&negative_batch, 1}));
  const std::vector<DerivativeReal> zero{0.0};
  StochasticReconfigurationBatch zero_batch{&stream, {zero.data(), zero.size()}};
  StochasticReconfigurationOperator zero_covariance(schema, 2,
                                                     {&zero_batch, 1});
  const std::vector<DerivativeValue> direction{{1.0, 0.0}, {0.0, 0.0}};
  std::vector<DerivativeValue> result(2);
  const StructuredParameterVectorConstView direction_view(
      schema, 2, {direction.data(), direction.size()});
  CHECK_THROWS(zero_covariance.apply(direction_view,
                                     {result.data(), result.size()}));

  const std::vector<DerivativeReal> weights{1.0};
  StochasticReconfigurationBatch batch{&stream, {weights.data(), weights.size()}};
  StochasticReconfigurationOperator covariance(schema, 2, {&batch, 1});
  fail_jvp = true;
  CHECK_THROWS(covariance.apply(direction_view, {result.data(), result.size()}));
  fail_jvp = false;
  CHECK_NOTHROW(covariance.apply(direction_view, {result.data(), result.size()}));
  fail_vjp = true;
  CHECK_THROWS(covariance.apply(direction_view, {result.data(), result.size()}));
  fail_vjp = false;

  IdentityPreconditioner preconditioner(schema, 2);
  CHECK_THROWS(StochasticReconfigurationUpdateRule(
      covariance, preconditioner,
      StochasticReconfigurationUpdateControl{1.0, 0.0, 10.0, 0.0, 2}));

  const StructuredParameterSchema complex_schema =
      makeSRSchema("sr/update_complex", 1, ParameterScalarDomain::COMPLEX128);
  class ComplexIdentity final : public MatrixFreeLinearOperator
  {
  public:
    explicit ComplexIdentity(const StructuredParameterSchema& schema)
        : MatrixFreeLinearOperator(schema, 0, ReductionDomain::GLOBAL, true)
    {}
    void evaluate(const StructuredParameterVectorConstView& direction,
                  DerivativeArrayView<DerivativeValue> output) const override
    {
      std::copy(direction.values().begin(), direction.values().end(), output.begin());
    }
  } complex_identity(complex_schema);
  IdentityPreconditioner complex_preconditioner(complex_schema, 0);
  CHECK_THROWS(StochasticReconfigurationUpdateRule(complex_identity,
                                                    complex_preconditioner));
}

TEST_CASE("Weighted sample consensus rejects inconsistent zero-weight moments",
          "[drivers][wftrain][sr]")
{
  const StructuredParameterSchema schema =
      makeSRSchema("sr/moments", 1, ParameterScalarDomain::REAL64);
  DistributedParameterReduction reduction;
  DistributedWeightedSampleMoments inconsistent;
  inconsistent.sample_count = 1;
  inconsistent.weighted_value_sum = {1.0, 0.0};
  CHECK_THROWS(reduction.reduceWeightedSampleMoments(schema, 0, inconsistent));

  DistributedWeightedSampleMoments empty;
  CHECK_THROWS(reduction.reduceWeightedSampleMoments(schema, 0, empty));
}

} // namespace qmcplusplus::wftrain
