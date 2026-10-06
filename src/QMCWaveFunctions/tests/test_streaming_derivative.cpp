//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_streaming_derivative.cpp
 * @brief Tests for bounded, schema/version-aware derivative streaming.
 */

#include <catch2/catch_test_macros.hpp>

#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"
#include "Utilities/for_testing/Catch2Approx.h"

#include <algorithm>
#include <array>
#include <complex>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

using ComplexVector = std::vector<DerivativeValue>;
using DenseOracle   = std::vector<ComplexVector>;

/// Construct a complex schema with trainable blocks separated by a frozen block.
StructuredParameterSchema makeSchema(ParameterScalarDomain scalar_domain)
{
  return StructuredParameterSchema(
      "toy/complex",
      {{"input/weight", {2, 2}, 0, 4, scalar_domain, true, "weights"},
       {"frozen/shift", {2}, 4, 2, scalar_domain, false, "frozen"},
       {"output/bias", {3}, 6, 3, scalar_domain, true, "biases"}});
}

/// Construct the complex-domain schema used by the numerical oracle tests.
StructuredParameterSchema makeComplexSchema()
{
  return makeSchema(ParameterScalarDomain::COMPLEX128);
}

/// Return a deterministic genuinely complex score Jacobian used only as a test oracle.
DenseOracle makeScoreJacobian()
{
  return {{{1.0, 0.5}, {2.0, -1.0}, {-0.5, 0.25}, {0.75, 1.5}, {0.2, 0.1}, {-0.3, 0.4},
           {1.2, -0.8}, {-0.7, 0.6}, {0.9, 0.3}},
          {{-1.0, 0.75}, {0.5, 0.2}, {1.5, -0.4}, {-0.25, 0.9}, {0.6, -0.2}, {0.1, 0.5},
           {-0.8, -0.3}, {1.1, 0.4}, {0.35, -1.2}},
          {{0.4, -1.1}, {-0.9, 0.8}, {0.3, 0.7}, {1.4, -0.6}, {-0.2, 0.9}, {0.5, 0.5},
           {0.65, 1.0}, {-1.3, -0.2}, {0.15, 0.85}}};
}

/// Return an independent local-energy Jacobian for mixed-product VJP testing.
DenseOracle makeLocalEnergyJacobian()
{
  DenseOracle result = makeScoreJacobian();
  for (std::size_t sample = 0; sample < result.size(); ++sample)
    for (std::size_t parameter = 0; parameter < result[sample].size(); ++parameter)
      result[sample][parameter] =
          DerivativeValue{0.3 * result[sample][parameter].real() + 0.1 * parameter,
                          -0.4 * result[sample][parameter].imag() + 0.2 * sample};
  return result;
}

/// Evaluate a dense test-only VJP reference without sharing streamed-kernel code.
ComplexVector denseVJP(const DenseOracle& jacobian,
                       const ComplexVector& coefficients,
                       DerivativeAdjoint adjoint)
{
  ComplexVector result(jacobian.front().size());
  for (std::size_t parameter = 0; parameter < result.size(); ++parameter)
    for (std::size_t sample = 0; sample < jacobian.size(); ++sample)
    {
      const DerivativeValue derivative = adjoint == DerivativeAdjoint::HERMITIAN
          ? std::conj(jacobian[sample][parameter])
          : jacobian[sample][parameter];
      result[parameter] += derivative * coefficients[sample];
    }
  return result;
}

/// Evaluate a dense test-only score JVP reference.
ComplexVector denseJVP(const DenseOracle& jacobian, const ComplexVector& direction)
{
  ComplexVector result(jacobian.size());
  for (std::size_t sample = 0; sample < jacobian.size(); ++sample)
    for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
      result[sample] += jacobian[sample][parameter] * direction[parameter];
  return result;
}

/// Form the normalized dense J^T (I - 11^T/S) J covariance action independently.
ComplexVector denseCenteredCovarianceAction(const DenseOracle& jacobian, const ComplexVector& direction)
{
  ComplexVector sample_response(jacobian.size());
  DerivativeValue response_sum;
  for (std::size_t sample = 0; sample < jacobian.size(); ++sample)
  {
    for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
      sample_response[sample] += jacobian[sample][parameter] * direction[parameter];
    response_sum += sample_response[sample];
  }

  const DerivativeValue inverse_sample_count{1.0 / jacobian.size(), 0.0};
  const DerivativeValue response_mean = response_sum * inverse_sample_count;
  ComplexVector result(direction.size());
  for (std::size_t parameter = 0; parameter < result.size(); ++parameter)
    for (std::size_t sample = 0; sample < jacobian.size(); ++sample)
      result[parameter] +=
          jacobian[sample][parameter] * (sample_response[sample] - response_mean) * inverse_sample_count;
  return result;
}

/// Compare complex vectors component by component with a tight full-precision tolerance.
void checkComplexVector(const ComplexVector& actual, const ComplexVector& expected)
{
  REQUIRE(actual.size() == expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index)
  {
    CHECK(actual[index].real() == Catch::Approx(expected[index].real()).epsilon(1e-12));
    CHECK(actual[index].imag() == Catch::Approx(expected[index].imag()).epsilon(1e-12));
  }
}

/// Collect canonical streamed parameter chunks into one vector per VJP channel.
class RecordingParameterSink final : public ParameterReductionSink
{
public:
  /// Return completed channel results; partial or poisoned results are inaccessible.
  const std::vector<ComplexVector>& results() const
  {
    if (state() != DerivativeSinkState::COMPLETE)
      throw std::logic_error("Recording parameter sink has no completed result");
    return results_;
  }

  /// Report how many validated chunk records reached the derived sink.
  std::size_t consumedRecords() const noexcept { return consumed_records_; }

  /// Report how many transactions reached the derived begin hook.
  std::size_t beginCalls() const noexcept { return begin_calls_; }

protected:
  /// Allocate only O(KP) objective result storage at transaction start.
  void onBegin(const DerivativeStreamDescriptor&,
               const ParameterChunkPlan& plan,
               DerivativeArrayView<const VJPCoefficientChannel> channels) override
  {
    std::size_t parameter_extent = 0;
    for (const ParameterChunkDescriptor& chunk : plan.chunks())
      parameter_extent = std::max(parameter_extent, chunk.parameter_offset + chunk.count);
    results_.assign(channels.size(), ComplexVector(parameter_extent));
    consumed_records_ = 0;
    ++begin_calls_;
  }

  /// Copy one bounded interval into its channel's canonical output position.
  void consume(std::size_t channel_ordinal, const ParameterChunkConstView& chunk) override
  {
    std::copy(chunk.values().begin(), chunk.values().end(),
              results_[channel_ordinal].begin() + chunk.descriptor().parameter_offset);
    ++consumed_records_;
  }

  /// Erase any result that did not complete atomically.
  void onAbort() noexcept override { results_.clear(); }

  /// Reinitialize result and counters deterministically for reuse.
  void onReset() noexcept override
  {
    results_.clear();
    consumed_records_ = 0;
    begin_calls_       = 0;
  }

private:
  std::vector<ComplexVector> results_;
  std::size_t consumed_records_ = 0;
  std::size_t begin_calls_       = 0;
};

/// Collect monotonically ordered JVP sample tiles into one scalar vector.
class RecordingSampleSink final : public SampleProductSink
{
public:
  /// Return the completed sample product.
  const ComplexVector& result() const
  {
    if (state() != DerivativeSinkState::COMPLETE)
      throw std::logic_error("Recording sample sink has no completed result");
    return result_;
  }

  /// Report how many transactions reached the derived begin hook.
  std::size_t beginCalls() const noexcept { return begin_calls_; }

protected:
  /// Allocate only one scalar per sample, never a sample-by-parameter object.
  void onBegin(const DerivativeStreamDescriptor& descriptor) override
  {
    sample_offset_ = descriptor.sample_offset;
    result_.assign(descriptor.sample_count, DerivativeValue{});
    ++begin_calls_;
  }

  /// Copy one validated bounded tile into the result vector.
  void consume(const SampleProductTileConstView& tile) override
  {
    const std::size_t local_offset = tile.descriptor().sample_offset - sample_offset_;
    std::copy(tile.values().begin(), tile.values().end(), result_.begin() + local_offset);
  }

  /// Erase any incomplete result.
  void onAbort() noexcept override { result_.clear(); }

  /// Reinitialize the recording sink for deterministic reuse.
  void onReset() noexcept override
  {
    result_.clear();
    sample_offset_ = 0;
    begin_calls_   = 0;
  }

private:
  std::size_t sample_offset_ = 0;
  ComplexVector result_;
  std::size_t begin_calls_ = 0;
};

/** Test-only operator backed by explicit tiny Jacobians.
 *
 * The dense representation is intentionally confined to this unit test.  Its producer
 * surface emits only bounded chunks and tiles through the production contract.
 */
class DenseOracleOperator final : public StreamingDerivativeOperator
{
public:
  /// Bind deterministic Jacobians to one schema, parameter version, and sample batch.
  DenseOracleOperator(std::size_t sample_count = 3,
                      std::uint32_t local_energy_terms = allLocalEnergyTerms(),
                      ParameterScalarDomain parameter_domain = ParameterScalarDomain::COMPLEX128,
                      DerivativeExecutionDomain execution_domain = DerivativeExecutionDomain::HOST)
      : schema_(makeSchema(parameter_domain)),
        plan_(schema_, PARAMETER_VERSION, 2),
        score_(makeScoreJacobian()),
        local_energy_(makeLocalEnergyJacobian()),
        sample_count_(sample_count),
        local_energy_terms_(local_energy_terms),
        parameter_domain_(parameter_domain),
        execution_domain_(execution_domain)
  {
    score_.resize(sample_count_);
    local_energy_.resize(sample_count_);
  }

  /// Return all local-energy terms represented by the test oracle.
  static constexpr std::uint32_t allLocalEnergyTerms() noexcept
  {
    return localEnergyTermBit(LocalEnergyTerm::KINETIC) |
        localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
  }

  /// Advertise fixed channel, parameter-chunk, and sample-tile capacities.
  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities result;
    result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
        derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_VJP) |
        derivativeProductBit(DerivativeProduct::SCORE_JVP);
    result.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::TRANSPOSE) |
        derivativeAdjointBit(DerivativeAdjoint::HERMITIAN);
    result.parameter_scalar_domain      = parameter_domain_;
    result.result_scalar_domain         = ParameterScalarDomain::COMPLEX128;
    result.execution_domain             = execution_domain_;
    result.local_energy_term_mask       = local_energy_terms_;
    result.maximum_vjp_channels         = 3;
    result.maximum_parameter_chunk_size = 2;
    result.maximum_sample_tile_size     = 2;
    result.block_streaming              = true;
    return result;
  }

  /// Return the immutable complex test schema.
  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }

  /// Return the version shared by schema-bound test views.
  std::size_t parameterVersion() const noexcept override { return PARAMETER_VERSION; }

  /// Return the ordered sample-batch identity.
  std::size_t batchOrdinal() const noexcept override { return BATCH_ORDINAL; }

  /// Return a nonzero sample offset to exercise local/global interval metadata.
  std::size_t sampleOffset() const noexcept override { return SAMPLE_OFFSET; }

  /// Return the selected test sample count.
  std::size_t sampleCount() const noexcept override { return sample_count_; }

  /// Return the trainable-only immutable canonical chunk plan.
  const ParameterChunkPlan& parameterChunkPlan() const noexcept override { return plan_; }

  /// Report how often the VJP evaluator was entered after preflight.
  std::size_t vjpEvaluationCount() const noexcept { return vjp_evaluation_count_; }

  /// Report how often the JVP evaluator was entered after preflight.
  std::size_t jvpEvaluationCount() const noexcept { return jvp_evaluation_count_; }

protected:
  /// Contract the dense oracle internally and emit only bounded parameter chunks.
  void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                    DerivativeAdjoint adjoint,
                    ParameterReductionSink& sink) const override
  {
    ++vjp_evaluation_count_;
    for (const ParameterChunkDescriptor& chunk : plan_.chunks())
      for (std::size_t channel_index = 0; channel_index < channels.size(); ++channel_index)
      {
        const DenseOracle& jacobian = channels[channel_index].product == DerivativeProduct::SCORE_VJP
            ? score_
            : local_energy_;
        ComplexVector values(chunk.count);
        for (std::size_t local_parameter = 0; local_parameter < chunk.count; ++local_parameter)
          for (std::size_t sample = 0; sample < sample_count_; ++sample)
          {
            DerivativeValue derivative = jacobian[sample][chunk.parameter_offset + local_parameter];
            if (adjoint == DerivativeAdjoint::HERMITIAN)
              derivative = std::conj(derivative);
            values[local_parameter] += derivative * channels[channel_index].coefficients.values[sample];
          }
        sink.add(channel_index, {chunk, {values.data(), values.size()}});
      }
  }

  /// Multiply the dense score oracle by a direction and emit bounded sample tiles.
  void evaluateScoreJVP(const StructuredParameterVectorConstView& direction,
                        SampleProductSink& sink) const override
  {
    ++jvp_evaluation_count_;
    for (std::size_t sample_begin = 0, ordinal = 0; sample_begin < sample_count_; sample_begin += 2, ++ordinal)
    {
      const std::size_t count = std::min<std::size_t>(2, sample_count_ - sample_begin);
      ComplexVector values(count);
      for (std::size_t local_sample = 0; local_sample < count; ++local_sample)
        for (std::size_t parameter = 0; parameter < direction.values().size(); ++parameter)
          values[local_sample] += score_[sample_begin + local_sample][parameter] * direction.values()[parameter];

      const SampleProductTileDescriptor descriptor{schema_.providerId(), schema_.fingerprint(), PARAMETER_VERSION,
                                                   BATCH_ORDINAL, SAMPLE_OFFSET + sample_begin, count, ordinal};
      sink.add({descriptor, {values.data(), values.size()}});
    }
  }

private:
  static constexpr std::size_t PARAMETER_VERSION = 7;
  static constexpr std::size_t BATCH_ORDINAL     = 11;
  static constexpr std::size_t SAMPLE_OFFSET     = 19;

  StructuredParameterSchema schema_;
  ParameterChunkPlan plan_;
  DenseOracle score_;
  DenseOracle local_energy_;
  std::size_t sample_count_;
  std::uint32_t local_energy_terms_;
  ParameterScalarDomain parameter_domain_;
  DerivativeExecutionDomain execution_domain_;
  mutable std::size_t vjp_evaluation_count_ = 0;
  mutable std::size_t jvp_evaluation_count_ = 0;
};

/// Construct exact coefficient metadata for one operator and backing vector.
CoefficientView coefficientView(const DenseOracleOperator& op, const ComplexVector& values)
{
  return {op.parameterSchema().providerId(), op.parameterSchema().fingerprint(), op.parameterVersion(),
          op.batchOrdinal(), op.sampleOffset(), {values.data(), values.size()}};
}

} // namespace

TEST_CASE("Streaming derivative chunk plans preserve structured boundaries", "[wavefunction][training]")
{
  const StructuredParameterSchema schema = makeComplexSchema();
  const ParameterChunkPlan plan(schema, 7, 2);
  REQUIRE(plan.chunks().size() == 4);
  CHECK(plan.selectedParameterCount() == 7);
  CHECK(plan.maximumChunkSize() == 2);

  CHECK(plan.chunks()[0].block_index == 0);
  CHECK(plan.chunks()[0].parameter_offset == 0);
  CHECK(plan.chunks()[1].parameter_offset == 2);
  CHECK(plan.chunks()[2].block_index == 2);
  CHECK(plan.chunks()[2].parameter_offset == 6);
  CHECK(plan.chunks()[3].block_offset == 2);
  CHECK(plan.chunks()[3].count == 1); // Exact tail; no chunk crosses into another block.
  for (std::size_t ordinal = 0; ordinal < plan.chunks().size(); ++ordinal)
  {
    CHECK(plan.chunks()[ordinal].ordinal == ordinal);
    CHECK(plan.chunks()[ordinal].count <= plan.maximumChunkSize());
    CHECK(plan.chunks()[ordinal].trainable);
  }

  const ParameterChunkPlan diagnostic(schema, 7, 2, FrozenBlockPolicy::INCLUDE_FOR_DIAGNOSTICS);
  CHECK(diagnostic.selectedParameterCount() == schema.parameterCount());
  REQUIRE(diagnostic.chunks().size() == 5);
  CHECK(diagnostic.chunks()[2].block_id == "frozen/shift");
  CHECK_FALSE(diagnostic.chunks()[2].trainable);

  CHECK_THROWS_AS(ParameterChunkPlan(schema, 7, 0), std::invalid_argument);
}

TEST_CASE("Streaming derivative VJP and JVP match independent complex oracles", "[wavefunction][training]")
{
  const DenseOracleOperator op;
  const ComplexVector score_coefficients{{1.0, 0.2}, {-0.5, 0.75}, {0.3, -0.4}};
  const ComplexVector energy_coefficients{{0.6, -0.1}, {1.1, 0.25}, {-0.8, 0.5}};
  const std::uint32_t energy_terms = DenseOracleOperator::allLocalEnergyTerms();
  const std::array<VJPCoefficientChannel, 2> channels{{
      {"weighted_score", DerivativeProduct::SCORE_VJP, coefficientView(op, score_coefficients), 0},
      {"weighted_local_energy", DerivativeProduct::LOCAL_ENERGY_VJP,
       coefficientView(op, energy_coefficients), energy_terms}}};

  RecordingParameterSink transpose_sink;
  op.applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::TRANSPOSE, transpose_sink);
  REQUIRE(transpose_sink.results().size() == channels.size());
  CHECK(transpose_sink.consumedRecords() == op.parameterChunkPlan().chunks().size() * channels.size());

  ComplexVector expected_score = denseVJP(makeScoreJacobian(), score_coefficients, DerivativeAdjoint::TRANSPOSE);
  ComplexVector expected_energy =
      denseVJP(makeLocalEnergyJacobian(), energy_coefficients, DerivativeAdjoint::TRANSPOSE);
  // Frozen parameters are deliberately absent from the trainable chunk plan.
  expected_score[4] = expected_score[5] = DerivativeValue{};
  expected_energy[4] = expected_energy[5] = DerivativeValue{};
  checkComplexVector(transpose_sink.results()[0], expected_score);
  checkComplexVector(transpose_sink.results()[1], expected_energy);

  RecordingParameterSink hermitian_sink;
  op.applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::HERMITIAN, hermitian_sink);
  ComplexVector expected_hermitian =
      denseVJP(makeScoreJacobian(), score_coefficients, DerivativeAdjoint::HERMITIAN);
  expected_hermitian[4] = expected_hermitian[5] = DerivativeValue{};
  checkComplexVector(hermitian_sink.results()[0], expected_hermitian);
  CHECK(hermitian_sink.results()[0][0].imag() != Catch::Approx(transpose_sink.results()[0][0].imag()));
  CHECK(hermitian_sink.results()[0][0].imag() != Catch::Approx(0.0));

  ComplexVector direction{{0.1, 0.2}, {-0.3, 0.4}, {0.5, -0.1}, {0.2, 0.3},
                          {0.0, 0.0}, {0.0, 0.0}, {-0.4, -0.2}, {0.6, 0.1}, {0.25, -0.5}};
  const StructuredParameterVectorConstView direction_view(op.parameterSchema(), op.parameterVersion(),
                                                           {direction.data(), direction.size()});
  RecordingSampleSink jvp_sink;
  op.applyScoreJVP(direction_view, jvp_sink);
  checkComplexVector(jvp_sink.result(), denseJVP(makeScoreJacobian(), direction));
}

TEST_CASE("Streaming derivative sinks poison incomplete or invalid transactions", "[wavefunction][training]")
{
  const DenseOracleOperator op;
  const ComplexVector coefficients{{1.0, 0.0}, {0.5, 0.25}, {-0.2, 0.3}};
  const std::array<VJPCoefficientChannel, 1> channels{{
      {"score", DerivativeProduct::SCORE_VJP, coefficientView(op, coefficients), 0}}};
  const StreamingDerivativeCapabilities support = op.capabilities();

  DerivativeStreamDescriptor descriptor;
  descriptor.provider_id                  = op.parameterSchema().providerId();
  descriptor.schema_fingerprint           = op.parameterSchema().fingerprint();
  descriptor.parameter_version            = op.parameterVersion();
  descriptor.product_mask                 = derivativeProductBit(DerivativeProduct::SCORE_VJP);
  descriptor.parameter_scalar_domain      = ParameterScalarDomain::COMPLEX128;
  descriptor.result_scalar_domain         = ParameterScalarDomain::COMPLEX128;
  descriptor.batch_ordinal                = op.batchOrdinal();
  descriptor.sample_offset                = op.sampleOffset();
  descriptor.sample_count                 = op.sampleCount();
  descriptor.maximum_vjp_channels         = support.maximum_vjp_channels;
  descriptor.maximum_parameter_chunk_size = support.maximum_parameter_chunk_size;
  descriptor.maximum_sample_tile_size     = support.maximum_sample_tile_size;

  RecordingParameterSink sink;
  sink.begin(descriptor, op.parameterChunkPlan(), {channels.data(), channels.size()});
  CHECK_THROWS_AS(sink.end(), std::logic_error);
  CHECK(sink.state() == DerivativeSinkState::POISONED);
  CHECK_THROWS_AS(sink.results(), std::logic_error);

  sink.reset();
  CHECK(sink.state() == DerivativeSinkState::IDLE);
  op.applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::TRANSPOSE, sink);
  CHECK(sink.state() == DerivativeSinkState::COMPLETE);

  sink.reset();
  sink.begin(descriptor, op.parameterChunkPlan(), {channels.data(), channels.size()});
  ComplexVector wrong_values(op.parameterChunkPlan().chunks()[1].count);
  CHECK_THROWS_AS(sink.add(0, {op.parameterChunkPlan().chunks()[1],
                               {wrong_values.data(), wrong_values.size()}}),
                  std::logic_error);
  CHECK(sink.state() == DerivativeSinkState::POISONED);

  sink.reset();
  VJPCoefficientChannel mismatched = channels[0];
  mismatched.coefficients.parameter_version = op.parameterVersion() + 1;
  CHECK_THROWS_AS(sink.begin(descriptor, op.parameterChunkPlan(), {&mismatched, 1}), std::logic_error);
  CHECK(sink.state() == DerivativeSinkState::POISONED);
}

TEST_CASE("Streaming derivative composition matches a centered covariance action", "[wavefunction][training]")
{
  const DenseOracleOperator op;
  ComplexVector direction{{0.1, 0.2}, {-0.3, 0.4}, {0.5, -0.1}, {0.2, 0.3},
                          {0.0, 0.0}, {0.0, 0.0}, {-0.4, -0.2}, {0.6, 0.1}, {0.25, -0.5}};
  const StructuredParameterVectorConstView direction_view(op.parameterSchema(), op.parameterVersion(),
                                                           {direction.data(), direction.size()});

  // First stream Jv as bounded sample tiles, then center and normalize the scalar
  // response before returning it through the independently checked score-VJP path.
  RecordingSampleSink jvp_sink;
  op.applyScoreJVP(direction_view, jvp_sink);
  ComplexVector centered_coefficients = jvp_sink.result();
  DerivativeValue response_mean;
  for (const DerivativeValue value : centered_coefficients)
    response_mean += value;
  response_mean /= static_cast<DerivativeReal>(centered_coefficients.size());
  for (DerivativeValue& value : centered_coefficients)
    value = (value - response_mean) / static_cast<DerivativeReal>(centered_coefficients.size());

  const VJPCoefficientChannel covariance_channel{
      "centered_score", DerivativeProduct::SCORE_VJP, coefficientView(op, centered_coefficients), 0};
  RecordingParameterSink covariance_sink;
  op.applyVJPs({&covariance_channel, 1}, DerivativeAdjoint::TRANSPOSE, covariance_sink);

  ComplexVector expected = denseCenteredCovarianceAction(makeScoreJacobian(), direction);
  expected[4] = expected[5] = DerivativeValue{}; // Frozen blocks are absent from the trainable output plan.
  checkComplexVector(covariance_sink.results().front(), expected);
}

TEST_CASE("Streaming derivative preflight rejects unsafe direction and execution metadata",
          "[wavefunction][training]")
{
  using DirectionView = DerivativeArrayView<const DerivativeValue>;
  static_assert(std::is_constructible_v<StructuredParameterVectorConstView,
                                        const StructuredParameterSchema&,
                                        std::size_t,
                                        DirectionView>);
  static_assert(!std::is_constructible_v<StructuredParameterVectorConstView,
                                         StructuredParameterSchema&&,
                                         std::size_t,
                                         DirectionView>);
  static_assert(!std::is_constructible_v<StructuredParameterVectorConstView,
                                         const StructuredParameterSchema&&,
                                         std::size_t,
                                         DirectionView>);

  const StructuredParameterSchema real_schema = makeSchema(ParameterScalarDomain::REAL64);
  ComplexVector real_values(real_schema.parameterCount(), DerivativeValue{0.25, 0.0});
  real_values[0] = {0.25, 0.1};
  CHECK_THROWS_AS(StructuredParameterVectorConstView(
                      real_schema, 7, {real_values.data(), real_values.size()}),
                  std::invalid_argument);
  real_values[0] = {std::numeric_limits<DerivativeReal>::quiet_NaN(), 0.0};
  CHECK_THROWS_AS(StructuredParameterVectorConstView(
                      real_schema, 7, {real_values.data(), real_values.size()}),
                  std::invalid_argument);

  // Complex schemas retain genuinely complex directions; no projection occurs.
  const StructuredParameterSchema complex_schema = makeComplexSchema();
  ComplexVector complex_values(complex_schema.parameterCount(), DerivativeValue{0.25, 0.1});
  const StructuredParameterVectorConstView complex_view(
      complex_schema, 7, {complex_values.data(), complex_values.size()});
  CHECK(complex_view.values()[0] == DerivativeValue{0.25, 0.1});

  // Revalidate the nonowning backing storage at application time because callers may
  // mutate it after constructing a valid view.
  const DenseOracleOperator real_op(3, DenseOracleOperator::allLocalEnergyTerms(),
                                    ParameterScalarDomain::REAL64);
  real_values.assign(real_op.parameterSchema().parameterCount(), DerivativeValue{0.25, 0.0});
  const StructuredParameterVectorConstView real_direction(
      real_op.parameterSchema(), real_op.parameterVersion(), {real_values.data(), real_values.size()});
  real_values[0] = {0.25, 0.1};
  RecordingSampleSink real_sink;
  CHECK_THROWS_AS(real_op.applyScoreJVP(real_direction, real_sink), std::invalid_argument);
  CHECK(real_sink.state() == DerivativeSinkState::IDLE);
  CHECK(real_sink.beginCalls() == 0);
  CHECK(real_op.jvpEvaluationCount() == 0);

  real_values[0] = {std::numeric_limits<DerivativeReal>::infinity(), 0.0};
  CHECK_THROWS_AS(real_op.applyScoreJVP(real_direction, real_sink), std::invalid_argument);
  CHECK(real_sink.state() == DerivativeSinkState::IDLE);
  CHECK(real_op.jvpEvaluationCount() == 0);

  const DenseOracleOperator host_op;
  const ComplexVector coefficients(host_op.sampleCount(), DerivativeValue{1.0, 0.0});
  const VJPCoefficientChannel host_channel{
      "score", DerivativeProduct::SCORE_VJP, coefficientView(host_op, coefficients), 0};
  const auto invalid_adjoint = static_cast<DerivativeAdjoint>(
      derivativeAdjointBit(DerivativeAdjoint::TRANSPOSE) |
      derivativeAdjointBit(DerivativeAdjoint::HERMITIAN));
  CHECK_FALSE(host_op.capabilities().supports(invalid_adjoint));
  RecordingParameterSink invalid_adjoint_sink;
  CHECK_THROWS_AS(host_op.applyVJPs({&host_channel, 1}, invalid_adjoint, invalid_adjoint_sink),
                  std::runtime_error);
  CHECK(invalid_adjoint_sink.state() == DerivativeSinkState::IDLE);
  CHECK(invalid_adjoint_sink.beginCalls() == 0);
  CHECK(host_op.vjpEvaluationCount() == 0);

  const DenseOracleOperator device_op(3, DenseOracleOperator::allLocalEnergyTerms(),
                                      ParameterScalarDomain::COMPLEX128,
                                      DerivativeExecutionDomain::DEVICE);
  const VJPCoefficientChannel device_channel{
      "score", DerivativeProduct::SCORE_VJP, coefficientView(device_op, coefficients), 0};
  RecordingParameterSink device_vjp_sink;
  CHECK_THROWS_AS(device_op.applyVJPs({&device_channel, 1}, DerivativeAdjoint::TRANSPOSE,
                                     device_vjp_sink),
                  std::runtime_error);
  CHECK(device_vjp_sink.state() == DerivativeSinkState::IDLE);
  CHECK(device_vjp_sink.beginCalls() == 0);
  CHECK(device_op.vjpEvaluationCount() == 0);

  ComplexVector device_direction_values(device_op.parameterSchema().parameterCount(),
                                         DerivativeValue{0.1, 0.2});
  const StructuredParameterVectorConstView device_direction(
      device_op.parameterSchema(), device_op.parameterVersion(),
      {device_direction_values.data(), device_direction_values.size()});
  RecordingSampleSink device_jvp_sink;
  CHECK_THROWS_AS(device_op.applyScoreJVP(device_direction, device_jvp_sink), std::runtime_error);
  CHECK(device_jvp_sink.state() == DerivativeSinkState::IDLE);
  CHECK(device_jvp_sink.beginCalls() == 0);
  CHECK(device_op.jvpEvaluationCount() == 0);

  // Public sink entry points independently reject device descriptors and malformed
  // adjoints without entering a derived hook.
  DerivativeStreamDescriptor direct_descriptor;
  direct_descriptor.provider_id                  = device_op.parameterSchema().providerId();
  direct_descriptor.schema_fingerprint           = device_op.parameterSchema().fingerprint();
  direct_descriptor.parameter_version            = device_op.parameterVersion();
  direct_descriptor.product_mask                 = derivativeProductBit(DerivativeProduct::SCORE_VJP);
  direct_descriptor.adjoint                      = DerivativeAdjoint::TRANSPOSE;
  direct_descriptor.parameter_scalar_domain      = ParameterScalarDomain::COMPLEX128;
  direct_descriptor.result_scalar_domain         = ParameterScalarDomain::COMPLEX128;
  direct_descriptor.batch_ordinal                = device_op.batchOrdinal();
  direct_descriptor.sample_offset                = device_op.sampleOffset();
  direct_descriptor.sample_count                 = device_op.sampleCount();
  direct_descriptor.execution_domain             = DerivativeExecutionDomain::DEVICE;
  direct_descriptor.maximum_vjp_channels         = 3;
  direct_descriptor.maximum_parameter_chunk_size = 2;
  direct_descriptor.maximum_sample_tile_size     = 2;

  RecordingParameterSink direct_device_sink;
  CHECK_THROWS_AS(direct_device_sink.begin(direct_descriptor, device_op.parameterChunkPlan(),
                                            {&device_channel, 1}),
                  std::logic_error);
  CHECK(direct_device_sink.state() == DerivativeSinkState::POISONED);
  CHECK(direct_device_sink.beginCalls() == 0);

  direct_descriptor.execution_domain = DerivativeExecutionDomain::HOST;
  direct_descriptor.adjoint          = invalid_adjoint;
  RecordingParameterSink direct_adjoint_sink;
  CHECK_THROWS_AS(direct_adjoint_sink.begin(direct_descriptor, device_op.parameterChunkPlan(),
                                             {&device_channel, 1}),
                  std::logic_error);
  CHECK(direct_adjoint_sink.state() == DerivativeSinkState::POISONED);
  CHECK(direct_adjoint_sink.beginCalls() == 0);

  DerivativeStreamDescriptor direct_sample_descriptor = direct_descriptor;
  direct_sample_descriptor.product_mask = derivativeProductBit(DerivativeProduct::SCORE_JVP);
  direct_sample_descriptor.adjoint = DerivativeAdjoint::TRANSPOSE;
  direct_sample_descriptor.execution_domain = DerivativeExecutionDomain::DEVICE;
  RecordingSampleSink direct_sample_sink;
  CHECK_THROWS_AS(direct_sample_sink.begin(direct_sample_descriptor), std::logic_error);
  CHECK(direct_sample_sink.state() == DerivativeSinkState::POISONED);
  CHECK(direct_sample_sink.beginCalls() == 0);
}

TEST_CASE("Streaming derivative capacities bound storage and zero-sample transitions", "[wavefunction][training]")
{
  static_assert(sizeof(DerivativeArrayView<const DerivativeValue>) <= 2 * sizeof(void*));
  static_assert(std::is_trivially_copyable_v<DerivativeArrayView<const DerivativeValue>>);
  static_assert(std::is_trivially_copyable_v<VJPCoefficientChannel>);
  static_assert(std::is_trivially_copyable_v<DerivativeStreamDescriptor>);
  static_assert(!std::is_constructible_v<DerivativeArrayView<const DerivativeValue>,
                                         std::size_t,
                                         std::size_t>);
  static_assert(!std::is_copy_constructible_v<ParameterReductionSink>);

  const DenseOracleOperator op;
  CHECK(op.capabilities().maximum_vjp_channels == 3);
  CHECK(op.capabilities().maximum_parameter_chunk_size == 2);
  CHECK(op.capabilities().maximum_sample_tile_size == 2);
  CHECK(op.parameterChunkPlan().maximumChunkSize() == 2);

  const ComplexVector coefficients(op.sampleCount(), {1.0, 0.0});
  const std::array<VJPCoefficientChannel, 4> too_many{{
      {"score/0", DerivativeProduct::SCORE_VJP, coefficientView(op, coefficients), 0},
      {"score/1", DerivativeProduct::SCORE_VJP, coefficientView(op, coefficients), 0},
      {"score/2", DerivativeProduct::SCORE_VJP, coefficientView(op, coefficients), 0},
      {"score/3", DerivativeProduct::SCORE_VJP, coefficientView(op, coefficients), 0}}};
  RecordingParameterSink untouched_sink;
  CHECK_THROWS_AS(op.applyVJPs({too_many.data(), too_many.size()}, DerivativeAdjoint::TRANSPOSE,
                               untouched_sink),
                  std::invalid_argument);
  CHECK(untouched_sink.state() == DerivativeSinkState::IDLE);

  const DenseOracleOperator kinetic_only_op(3, localEnergyTermBit(LocalEnergyTerm::KINETIC));
  const VJPCoefficientChannel missing_ecp_channel{
      "local_energy",
      DerivativeProduct::LOCAL_ENERGY_VJP,
      coefficientView(kinetic_only_op, coefficients),
      localEnergyTermBit(LocalEnergyTerm::KINETIC) | localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)};
  CHECK_THROWS_AS(kinetic_only_op.applyVJPs({&missing_ecp_channel, 1}, DerivativeAdjoint::TRANSPOSE,
                                            untouched_sink),
                  std::runtime_error);
  CHECK(untouched_sink.state() == DerivativeSinkState::IDLE);

  const DenseOracleOperator empty_op(0);
  const ComplexVector empty_coefficients;
  const std::array<VJPCoefficientChannel, 1> empty_channel{{
      {"score", DerivativeProduct::SCORE_VJP, coefficientView(empty_op, empty_coefficients), 0}}};
  RecordingParameterSink empty_parameter_sink;
  empty_op.applyVJPs({empty_channel.data(), empty_channel.size()}, DerivativeAdjoint::TRANSPOSE,
                     empty_parameter_sink);
  CHECK(empty_parameter_sink.state() == DerivativeSinkState::COMPLETE);
  CHECK(empty_parameter_sink.consumedRecords() == 0);

  ComplexVector direction(empty_op.parameterSchema().parameterCount());
  const StructuredParameterVectorConstView direction_view(empty_op.parameterSchema(), empty_op.parameterVersion(),
                                                           {direction.data(), direction.size()});
  RecordingSampleSink empty_sample_sink;
  empty_op.applyScoreJVP(direction_view, empty_sample_sink);
  CHECK(empty_sample_sink.state() == DerivativeSinkState::COMPLETE);
  CHECK(empty_sample_sink.result().empty());

  // Ordinary end is forbidden for zero samples; the explicit transition is required.
  DerivativeStreamDescriptor zero_descriptor;
  zero_descriptor.provider_id              = empty_op.parameterSchema().providerId();
  zero_descriptor.schema_fingerprint       = empty_op.parameterSchema().fingerprint();
  zero_descriptor.parameter_version        = empty_op.parameterVersion();
  zero_descriptor.product_mask             = derivativeProductBit(DerivativeProduct::SCORE_JVP);
  zero_descriptor.parameter_scalar_domain  = ParameterScalarDomain::COMPLEX128;
  zero_descriptor.result_scalar_domain     = ParameterScalarDomain::COMPLEX128;
  zero_descriptor.batch_ordinal            = empty_op.batchOrdinal();
  zero_descriptor.sample_offset            = empty_op.sampleOffset();
  zero_descriptor.sample_count             = 0;
  zero_descriptor.maximum_sample_tile_size = 2;
  empty_sample_sink.reset();
  empty_sample_sink.begin(zero_descriptor);
  CHECK_THROWS_AS(empty_sample_sink.end(), std::logic_error);
  CHECK(empty_sample_sink.state() == DerivativeSinkState::POISONED);
}

} // namespace qmcplusplus::wftrain
