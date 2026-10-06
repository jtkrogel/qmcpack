//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_QuantileClipping.cpp
 * @brief Local and distributed tests for robust energy-gradient clipping.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "Message/Communicate.h"
#include "QMCDrivers/WFTrain/DistributedParameterReduction.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

using Catch::Approx;

/// Construct the small canonical schema used by the clipping objective tests.
StructuredParameterSchema makeClippingSchema()
{
  return {"clipping/toy",
          {{"weights", {2}, 0, 2, ParameterScalarDomain::REAL64, true, "weights"}}};
}

/** Emit exact score and local-energy contractions for a supplied synthetic batch.
 *
 * Rows are retained only in this tiny test double; the production API still observes
 * exclusively bounded VJP chunks.
 */
class ClippingStreamingOperator final : public StreamingDerivativeOperator
{
public:
  ClippingStreamingOperator(const StructuredParameterSchema& schema,
                            std::vector<std::vector<DerivativeValue>> score_rows,
                            std::vector<std::vector<DerivativeValue>> response_rows,
                            ReductionDomain domain = ReductionDomain::CROWD_LOCAL)
      : schema_(schema), score_rows_(std::move(score_rows)),
        response_rows_(std::move(response_rows)), domain_(domain), plan_(schema_, 0, 1)
  {
    if (score_rows_.size() != response_rows_.size())
      throw std::invalid_argument("Synthetic clipping rows have different populations");
  }

  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities result;
    result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
        derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_VJP);
    result.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::TRANSPOSE);
    result.parameter_scalar_domain = ParameterScalarDomain::REAL64;
    result.result_scalar_domain = ParameterScalarDomain::COMPLEX128;
    result.reduction_domain = domain_;
    result.execution_domain = DerivativeExecutionDomain::HOST;
    result.local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
    result.maximum_vjp_channels = 3;
    result.maximum_parameter_chunk_size = 1;
    result.maximum_sample_tile_size = std::max<std::size_t>(score_rows_.size(), 1);
    result.block_streaming = true;
    return result;
  }

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }
  std::size_t parameterVersion() const noexcept override { return 0; }
  std::size_t batchOrdinal() const noexcept override { return 0; }
  std::size_t sampleOffset() const noexcept override { return 0; }
  std::size_t sampleCount() const noexcept override { return score_rows_.size(); }
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
          for (std::size_t sample = 0; sample < score_rows_.size(); ++sample)
          {
            const auto& rows = channels[channel].product == DerivativeProduct::SCORE_VJP
                ? score_rows_
                : response_rows_;
            contraction[local_parameter] +=
                channels[channel].coefficients.values[sample] * rows[sample][parameter];
          }
        }
        sink.add(channel, {descriptor, {contraction.data(), contraction.size()}});
      }
  }

  void evaluateScoreJVP(const StructuredParameterVectorConstView&,
                        SampleProductSink&) const override
  {
    throw std::logic_error("ClippingStreamingOperator does not implement score JVP");
  }

private:
  const StructuredParameterSchema& schema_;
  std::vector<std::vector<DerivativeValue>> score_rows_;
  std::vector<std::vector<DerivativeValue>> response_rows_;
  ReductionDomain domain_;
  ParameterChunkPlan plan_;
};

/// Build deterministic rank-local samples, including a zero-local rank when possible.
void makeDistributedSamples(int rank,
                            int size,
                            std::vector<DerivativeReal>& weights,
                            std::vector<DerivativeValue>& energies,
                            std::vector<std::vector<DerivativeValue>>& scores,
                            std::vector<std::vector<DerivativeValue>>& responses)
{
  const std::size_t count = size > 1 && rank == 1 ? 0 : static_cast<std::size_t>(rank + 1);
  weights.resize(count);
  energies.resize(count);
  scores.resize(count, std::vector<DerivativeValue>(2));
  responses.resize(count, std::vector<DerivativeValue>(2));
  for (std::size_t sample = 0; sample < count; ++sample)
  {
    const double identity = 10.0 * rank + sample + 1.0;
    weights[sample] = 1.0 + 0.125 * sample;
    energies[sample] = {-4.0 + 1.25 * identity, 0.05 * identity};
    scores[sample] = {0.01 * identity, -0.03 * (identity + 1.0)};
    responses[sample] = {0.002 * (identity + 2.0), 0.004 * (identity - 1.0)};
  }
}

/// Append the deterministic data for every synthetic rank in rank order.
void makeSerialDistributedReference(
    int size,
    std::vector<DerivativeReal>& weights,
    std::vector<DerivativeValue>& energies,
    std::vector<std::vector<DerivativeValue>>& scores,
    std::vector<std::vector<DerivativeValue>>& responses)
{
  for (int rank = 0; rank < size; ++rank)
  {
    std::vector<DerivativeReal> rank_weights;
    std::vector<DerivativeValue> rank_energies;
    std::vector<std::vector<DerivativeValue>> rank_scores;
    std::vector<std::vector<DerivativeValue>> rank_responses;
    makeDistributedSamples(rank, size, rank_weights, rank_energies, rank_scores,
                           rank_responses);
    weights.insert(weights.end(), rank_weights.begin(), rank_weights.end());
    energies.insert(energies.end(), rank_energies.begin(), rank_energies.end());
    scores.insert(scores.end(), rank_scores.begin(), rank_scores.end());
    responses.insert(responses.end(), rank_responses.begin(), rank_responses.end());
  }
}

/// Evaluate and finalize one synthetic objective with an optional clipping transform.
EnergyGradientResult evaluateObjective(
    const StructuredParameterSchema& schema,
    const std::vector<DerivativeReal>& weights,
    const std::vector<DerivativeValue>& energies,
    std::vector<std::vector<DerivativeValue>> scores,
    std::vector<std::vector<DerivativeValue>> responses,
    const EnergyClippingTransform* clipping,
    ReductionDomain domain = ReductionDomain::CROWD_LOCAL)
{
  ClippingStreamingOperator derivative_operator(schema, std::move(scores),
                                                std::move(responses), domain);
  EnergyGradientAccumulator accumulator(schema, 0);
  accumulateEnergyGradientBatch(
      derivative_operator, {weights.data(), weights.size()},
      {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
      accumulator, DerivativeAdjoint::TRANSPOSE, clipping);
  accumulator.completeSingleParticipantReduction();
  return accumulator.finalize();
}

} // namespace

TEST_CASE("Energy clipping exact local statistics", "[drivers][training][clipping]")
{
  const std::vector<DerivativeValue> energies{{-4.0, 0.1}, {-1.0, 0.2},
                                               {1.0, 0.3}, {9.0, 0.4}};
  EnergyClippingPolicy mad_policy;
  mad_policy.width_multiplier = 1.0;
  const auto mad = prepareEnergyClippingTransform(
      {energies.data(), energies.size()}, mad_policy);
  CHECK(mad.descriptor().population == 4);
  CHECK(mad.descriptor().center == Approx(0.0));
  CHECK(mad.descriptor().scale == Approx(3.75));
  CHECK(mad.descriptor().width == Approx(3.75));
  CHECK(mad.apply({20.0, 7.0}).real() == Approx(3.75));
  CHECK(mad.apply({20.0, 7.0}).imag() == Approx(7.0));

  const auto deepqmc_default = prepareEnergyClippingTransform(
      {energies.data(), energies.size()});
  CHECK(deepqmc_default.descriptor().width == Approx(5.0 * 3.75));

  EnergyClippingPolicy quantile_policy;
  quantile_policy.scale_rule = EnergyClippingScaleRule::EMPIRICAL_QUANTILE;
  quantile_policy.width_multiplier = 1.0;
  quantile_policy.residual_quantile = 0.5;
  const auto median_residual = prepareEnergyClippingTransform(
      {energies.data(), energies.size()}, quantile_policy);
  CHECK(median_residual.descriptor().scale == Approx(1.0));

  quantile_policy.residual_quantile = 0.0;
  CHECK(prepareEnergyClippingTransform({energies.data(), energies.size()}, quantile_policy)
            .descriptor().scale == Approx(1.0));
  quantile_policy.residual_quantile = 1.0;
  CHECK(prepareEnergyClippingTransform({energies.data(), energies.size()}, quantile_policy)
            .descriptor().scale == Approx(9.0));

  // q*N is exactly above one but rounds to one in binary64 multiplication.  The
  // exact ceil(q*N)-1 definition must therefore choose the second residual.
  const std::vector<DerivativeValue> three_energies{{0.0, 0.0}, {1.0, 0.0},
                                                      {2.0, 0.0}};
  quantile_policy.residual_quantile = std::nextafter(1.0 / 3.0, 1.0);
  CHECK(prepareEnergyClippingTransform(
            {three_energies.data(), three_energies.size()}, quantile_policy)
            .descriptor().scale == Approx(1.0));

  const std::vector<DerivativeValue> singleton{{-2.5, 4.0}};
  const auto zero_scale = prepareEnergyClippingTransform(
      {singleton.data(), singleton.size()}, mad_policy);
  CHECK(zero_scale.descriptor().center == Approx(-2.5));
  CHECK(zero_scale.descriptor().scale == Approx(0.0));
  CHECK(zero_scale.apply({8.0, -3.0}) == DerivativeValue{-2.5, -3.0});

  const std::vector<DerivativeValue> signed_zero{{-0.0, 0.0}, {0.0, 0.0}};
  CHECK(prepareEnergyClippingTransform({signed_zero.data(), signed_zero.size()}, mad_policy)
            .descriptor().center == 0.0);
}

TEST_CASE("Energy clipping validation rejects invalid inputs",
          "[drivers][training][clipping]")
{
  const std::vector<DerivativeValue> empty;
  CHECK_THROWS(prepareEnergyClippingTransform({empty.data(), empty.size()}));

  const std::vector<DerivativeValue> nan_energy{
      {std::numeric_limits<double>::quiet_NaN(), 0.0}};
  CHECK_THROWS(prepareEnergyClippingTransform({nan_energy.data(), nan_energy.size()}));

  const double maximum = std::numeric_limits<double>::max();
  const std::vector<DerivativeValue> overflowing_residuals{{-maximum, 0.0},
                                                            {maximum, 0.0}};
  CHECK_THROWS(prepareEnergyClippingTransform(
      {overflowing_residuals.data(), overflowing_residuals.size()}));

  // A low quantile must not hide an overflowing residual above its ordinal.
  const std::vector<DerivativeValue> individually_overflowing_residuals{
      {-maximum, 0.0}, {-maximum, 0.0}, {maximum, 0.0}};
  EnergyClippingPolicy low_quantile;
  low_quantile.scale_rule = EnergyClippingScaleRule::EMPIRICAL_QUANTILE;
  low_quantile.residual_quantile = 0.0;
  CHECK_THROWS_WITH(
      prepareEnergyClippingTransform({individually_overflowing_residuals.data(),
                                      individually_overflowing_residuals.size()},
                                     low_quantile),
      Catch::Matchers::ContainsSubstring("residual construction"));

  const std::vector<DerivativeValue> finite{{1.0, 0.0}};
  EnergyClippingPolicy invalid;
  invalid.width_multiplier = -1.0;
  CHECK_THROWS(prepareEnergyClippingTransform({finite.data(), finite.size()}, invalid));
  invalid.width_multiplier = 1.0;
  invalid.residual_quantile = 1.1;
  CHECK_THROWS(prepareEnergyClippingTransform({finite.data(), finite.size()}, invalid));
}

TEST_CASE("Clipping changes only the energy-score covariance channel",
          "[drivers][training][clipping]")
{
  const StructuredParameterSchema schema = makeClippingSchema();
  const std::vector<DerivativeReal> weights{1.0, 1.0, 1.0};
  const std::vector<DerivativeValue> energies{{0.0, 0.0}, {2.0, 0.0}, {100.0, 0.0}};
  const std::vector<std::vector<DerivativeValue>> scores{{1.0, 0.0}, {2.0, 0.0},
                                                         {3.0, 0.0}};
  const std::vector<std::vector<DerivativeValue>> responses{{10.0, 1.0}, {20.0, 1.0},
                                                            {30.0, 1.0}};
  EnergyClippingPolicy policy;
  policy.scale_rule = EnergyClippingScaleRule::EMPIRICAL_QUANTILE;
  policy.width_multiplier = 1.0;
  policy.residual_quantile = 0.5;
  const auto clipping = prepareEnergyClippingTransform(
      {energies.data(), energies.size()}, policy);
  const auto raw = evaluateObjective(schema, weights, energies, scores, responses, nullptr);
  const auto clipped = evaluateObjective(schema, weights, energies, scores, responses, &clipping);

  CHECK(clipping.descriptor().center == Approx(2.0));
  CHECK(clipping.descriptor().width == Approx(2.0));
  CHECK(raw.mean_energy == clipped.mean_energy);
  CHECK(raw.energy_variance == Approx(clipped.energy_variance));
  CHECK(clipped.clipping.has_value());
  REQUIRE(clipped.clipped_mean_energy.has_value());
  CHECK(clipped.clipped_mean_energy->real() == Approx(2.0));
  CHECK(clipped.clipped_sample_count == 1);
  CHECK(clipped.gradient[0] == Approx(2.0 * (60.0 + 16.0 - 2.0 * 6.0) / 3.0));
  CHECK(raw.gradient[0] == Approx(2.0 * (60.0 + 304.0 - 34.0 * 6.0) / 3.0));

  EnergyClippingPolicy inactive_policy = policy;
  inactive_policy.width_multiplier = 1.0e6;
  const auto inactive = prepareEnergyClippingTransform(
      {energies.data(), energies.size()}, inactive_policy);
  const auto unchanged = evaluateObjective(schema, weights, energies, scores, responses,
                                           &inactive);
  CHECK(unchanged.gradient == raw.gradient);
  CHECK(unchanged.mean_energy == raw.mean_energy);
  CHECK(unchanged.energy_variance == raw.energy_variance);
  CHECK(unchanged.clipped_sample_count == 0);
}

TEST_CASE("Clipping merge enforces transform and complete-population identity",
          "[drivers][training][clipping]")
{
  const StructuredParameterSchema schema = makeClippingSchema();
  const std::vector<DerivativeReal> weights{1.0, 1.0};
  const std::vector<DerivativeValue> all_energies{{0.0, 0.0}, {2.0, 0.0}, {20.0, 0.0}};
  const std::vector<DerivativeValue> energies{all_energies.begin(), all_energies.begin() + 2};
  const std::vector<std::vector<DerivativeValue>> scores{{1.0, 2.0}, {2.0, 3.0}};
  const std::vector<std::vector<DerivativeValue>> responses{{0.1, 0.2}, {0.2, 0.3}};
  EnergyClippingPolicy first_policy;
  first_policy.width_multiplier = 1.0;
  EnergyClippingPolicy second_policy = first_policy;
  second_policy.width_multiplier = 2.0;
  const auto first_transform = prepareEnergyClippingTransform(
      {all_energies.data(), all_energies.size()}, first_policy);
  const auto second_transform = prepareEnergyClippingTransform(
      {all_energies.data(), all_energies.size()}, second_policy);

  ClippingStreamingOperator first_operator(schema, scores, responses);
  EnergyGradientAccumulator first(schema, 0);
  accumulateEnergyGradientBatch(
      first_operator, {weights.data(), weights.size()}, {energies.data(), energies.size()},
      localEnergyTermBit(LocalEnergyTerm::KINETIC), first,
      DerivativeAdjoint::TRANSPOSE, &first_transform);
  EnergyGradientAccumulator second(schema, 0);
  accumulateEnergyGradientBatch(
      first_operator, {weights.data(), weights.size()}, {energies.data(), energies.size()},
      localEnergyTermBit(LocalEnergyTerm::KINETIC), second,
      DerivativeAdjoint::TRANSPOSE, &second_transform);

  EnergyGradientAccumulator destination(schema, 0);
  destination.merge(first);
  CHECK_THROWS_WITH(destination.merge(second),
                    Catch::Matchers::ContainsSubstring("different clipping transforms"));
  destination.completeSingleParticipantReduction();
  CHECK_THROWS_WITH(destination.finalize(),
                    Catch::Matchers::ContainsSubstring("complete population"));
}

TEST_CASE("Distributed clipping matches serial thresholds and gradients",
          "[drivers][training][clipping][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  std::vector<DerivativeReal> weights;
  std::vector<DerivativeValue> energies;
  std::vector<std::vector<DerivativeValue>> scores;
  std::vector<std::vector<DerivativeValue>> responses;
  makeDistributedSamples(communicator.rank(), communicator.size(), weights, energies, scores,
                         responses);

  std::vector<DerivativeReal> reference_weights;
  std::vector<DerivativeValue> reference_energies;
  std::vector<std::vector<DerivativeValue>> reference_scores;
  std::vector<std::vector<DerivativeValue>> reference_responses;
  makeSerialDistributedReference(communicator.size(), reference_weights, reference_energies,
                                 reference_scores, reference_responses);

  for (const EnergyClippingScaleRule rule :
       {EnergyClippingScaleRule::MEAN_ABSOLUTE_DEVIATION,
        EnergyClippingScaleRule::EMPIRICAL_QUANTILE})
  {
    EnergyClippingPolicy policy;
    policy.scale_rule = rule;
    policy.width_multiplier = 1.75;
    policy.residual_quantile = 0.7;
    const auto distributed_transform = prepareEnergyClippingTransform(
        {energies.data(), energies.size()}, policy, &communicator);
    const auto reference_transform = prepareEnergyClippingTransform(
        {reference_energies.data(), reference_energies.size()}, policy);
    CHECK(distributed_transform.descriptor().equivalent(reference_transform.descriptor()));

    const StructuredParameterSchema schema = makeClippingSchema();
    ClippingStreamingOperator derivative_operator(schema, scores, responses,
                                                  ReductionDomain::RANK_LOCAL);
    EnergyGradientAccumulator accumulator(schema, 0);
    accumulateEnergyGradientBatch(
        derivative_operator, {weights.data(), weights.size()},
        {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
        accumulator, DerivativeAdjoint::TRANSPOSE, &distributed_transform);
    DistributedParameterReduction(communicator, {1}).reduce(accumulator);
    const EnergyGradientResult distributed = accumulator.finalize();
    const EnergyGradientResult reference = evaluateObjective(
        schema, reference_weights, reference_energies, reference_scores,
        reference_responses, &reference_transform);

    CHECK(distributed.sample_count == reference.sample_count);
    CHECK(distributed.clipped_sample_count == reference.clipped_sample_count);
    CHECK(distributed.mean_energy.real() == Approx(reference.mean_energy.real()));
    CHECK(distributed.mean_energy.imag() == Approx(reference.mean_energy.imag()));
    CHECK(distributed.energy_variance == Approx(reference.energy_variance));
    REQUIRE(distributed.gradient.size() == reference.gradient.size());
    for (std::size_t parameter = 0; parameter < distributed.gradient.size(); ++parameter)
      CHECK(distributed.gradient[parameter] == Approx(reference.gradient[parameter]));
  }
}

TEST_CASE("Distributed clipping failures are rank uniform before selection",
          "[drivers][training][clipping][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  std::vector<DerivativeValue> energies{{static_cast<double>(communicator.rank()), 0.0}};
  EnergyClippingPolicy policy;
  if (communicator.size() > 1 && communicator.rank() == communicator.size() - 1)
    policy.width_multiplier = 4.0;
  if (communicator.size() > 1)
    CHECK_THROWS_WITH(
        prepareEnergyClippingTransform({energies.data(), energies.size()}, policy,
                                       &communicator),
        Catch::Matchers::ContainsSubstring("policy metadata mismatch"));

  policy.width_multiplier = 5.0;
  if (communicator.rank() == communicator.size() - 1)
    energies.front() = {std::numeric_limits<double>::quiet_NaN(), 0.0};
  CHECK_THROWS_WITH(
      prepareEnergyClippingTransform({energies.data(), energies.size()}, policy,
                                     &communicator),
      Catch::Matchers::ContainsSubstring("preflight failed"));
}

} // namespace qmcplusplus::wftrain
