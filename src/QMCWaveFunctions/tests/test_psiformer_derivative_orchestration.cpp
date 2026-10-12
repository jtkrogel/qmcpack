//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_derivative_orchestration.cpp
 * @brief CPU oracle tests for streamed derivative folding and host publication.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerDerivativeOrchestration.h"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer
{
namespace
{

constexpr std::size_t PARAMETER_COUNT = 22;
constexpr std::size_t MAXIMUM_EVENTS = 128;

PsiFormerDeviceLayout makeDeviceLayout()
{
  PsiFormerDeviceLayout layout;
  layout.model_shape = {/*spin_up_electrons=*/1,
                        /*spin_down_electrons=*/1,
                        /*nuclei=*/2,
                        /*determinants=*/2,
                        /*feature_dimension=*/2,
                        /*attention_heads=*/1,
                        /*attention_blocks=*/1};
  layout.boundary = BoundaryCondition::OPEN;
  layout.geometry_feature_policy = GeometryFeaturePolicy::OPEN_EUCLIDEAN_V1;
  layout.parameter_count = PARAMETER_COUNT;
  layout.tensors = {
      {ParameterRole::ELECTRON_EMBEDDING_WEIGHT, NO_ATTENTION_BLOCK, 2,
       {9, 2}, 0, 18},
      {ParameterRole::UPDATE_HIDDEN_BIAS, 0, 1, {2, 0}, 18, 20},
      {ParameterRole::UPDATE_OUTPUT_BIAS, 0, 1, {2, 0}, 20, 22}};
  layout.fingerprint = 1701;
  return layout;
}

PsiFormerDerivativePlan makePlan()
{
  const PsiFormerDeviceLayout device = makeDeviceLayout();
  PsiFormerDerivativePlanningRequirements requirements;
  requirements.retained_tape_fp64_elements =
      minimumPsiFormerScoreTapeFP64Elements(device);
  requirements.adjoint_tape_fp64_elements = 127;
  requirements.complex_accumulator_count = 3;
  requirements.maximum_parameter_chunk_elements = 5;
  requirements.maximum_jvp_output_complex_elements = 1;
  requirements.block_size = 8;
  return makePsiFormerDerivativePlan(device, 19, requirements);
}

wftrain::StructuredParameterSchema makeSchema()
{
  return wftrain::StructuredParameterSchema(
      "psiformer/orchestration",
      {{"embedding", {9, 2}, 0, 18, wftrain::ParameterScalarDomain::REAL64,
        true, "network"},
       {"frozen", {2}, 18, 2, wftrain::ParameterScalarDomain::REAL64,
        false, "frozen"},
       {"output", {2}, 20, 2, wftrain::ParameterScalarDomain::REAL64,
        true, "network"}});
}

enum class EventKind : std::uint8_t
{
  COPY_COMPLEX,
  COPY_JVP,
  SYNCHRONIZE,
  ADD_PARAMETER,
  ADD_SAMPLE
};

struct EventLog
{
  std::array<EventKind, MAXIMUM_EVENTS> events{};
  std::size_t size = 0;

  void append(EventKind kind)
  {
    if (size >= events.size())
      throw std::logic_error("test event log capacity exceeded");
    events[size++] = kind;
  }
};

class RecordingStager final : public PsiFormerDerivativeHostStager
{
public:
  explicit RecordingStager(EventLog& log) : log_(log) {}

  void copyComplexChunk(const double* source,
                        std::size_t count,
                        wftrain::DerivativeValue* destination) override
  {
    log_.append(EventKind::COPY_COMPLEX);
    for (std::size_t index = 0; index < count; ++index)
      destination[index] = {source[2 * index], source[2 * index + 1]};
  }

  void copyRealJVPScalar(const double* source,
                         wftrain::DerivativeValue* destination) override
  {
    log_.append(EventKind::COPY_JVP);
    destination[0] = {*source, 0.0};
  }

  void synchronize() override { log_.append(EventKind::SYNCHRONIZE); }

private:
  EventLog& log_;
};

class FixedParameterSink final : public wftrain::ParameterReductionSink
{
public:
  FixedParameterSink(EventLog& log, const wftrain::DerivativeValue* staging)
      : log_(log), staging_(staging)
  {}

  const auto& values() const noexcept { return values_; }
  const auto& touched() const noexcept { return touched_; }
  bool alwaysUsedStaging() const noexcept { return always_used_staging_; }

protected:
  void consume(std::size_t channel,
               const wftrain::ParameterChunkConstView& chunk) override
  {
    log_.append(EventKind::ADD_PARAMETER);
    always_used_staging_ &= chunk.values().data() == staging_;
    for (std::size_t index = 0; index < chunk.values().size(); ++index)
    {
      const std::size_t parameter = chunk.descriptor().parameter_offset + index;
      values_[channel][parameter] = chunk.values()[index];
      touched_[parameter] = true;
    }
  }

private:
  EventLog& log_;
  const wftrain::DerivativeValue* staging_;
  std::array<std::array<wftrain::DerivativeValue, PARAMETER_COUNT>, 3> values_{};
  std::array<bool, PARAMETER_COUNT> touched_{};
  bool always_used_staging_ = true;
};

class FixedSampleSink final : public wftrain::SampleProductSink
{
public:
  FixedSampleSink(EventLog& log, const wftrain::DerivativeValue* staging)
      : log_(log), staging_(staging)
  {}

  wftrain::DerivativeValue value() const noexcept { return value_; }
  bool usedStaging() const noexcept { return used_staging_; }

protected:
  void consume(const wftrain::SampleProductTileConstView& tile) override
  {
    log_.append(EventKind::ADD_SAMPLE);
    used_staging_ = tile.values().data() == staging_;
    value_ = tile.values()[0];
  }

private:
  EventLog& log_;
  const wftrain::DerivativeValue* staging_;
  wftrain::DerivativeValue value_{};
  bool used_staging_ = false;
};

std::array<wftrain::VJPCoefficientChannel, 3> makeChannels(
    const wftrain::StructuredParameterSchema& schema,
    const std::array<wftrain::DerivativeValue, 3>& coefficients)
{
  std::array<wftrain::VJPCoefficientChannel, 3> channels;
  constexpr std::array<std::string_view, 3> ids{"first", "second", "third"};
  for (std::size_t channel = 0; channel < channels.size(); ++channel)
    channels[channel] = {
        ids[channel], wftrain::DerivativeProduct::SCORE_VJP,
        {schema.providerId(), schema.fingerprint(), 19, 4, 7,
         {coefficients.data() + channel, 1}}, 0};
  return channels;
}

wftrain::DerivativeStreamDescriptor makeVJPDescriptor(
    const wftrain::StructuredParameterSchema& schema,
    wftrain::DerivativeAdjoint adjoint)
{
  return {schema.providerId(), schema.fingerprint(), 19,
          wftrain::derivativeProductBit(wftrain::DerivativeProduct::SCORE_VJP),
          adjoint, wftrain::ParameterScalarDomain::REAL64,
          wftrain::ParameterScalarDomain::COMPLEX128, 4, 7, 1,
          wftrain::ReductionDomain::CROWD_LOCAL,
          wftrain::DerivativeExecutionDomain::HOST, 3, 5, 1};
}

wftrain::DerivativeStreamDescriptor makeJVPDescriptor(
    const wftrain::StructuredParameterSchema& schema)
{
  return {schema.providerId(), schema.fingerprint(), 19,
          wftrain::derivativeProductBit(wftrain::DerivativeProduct::SCORE_JVP),
          wftrain::DerivativeAdjoint::TRANSPOSE,
          wftrain::ParameterScalarDomain::REAL64,
          wftrain::ParameterScalarDomain::COMPLEX128, 4, 7, 1,
          wftrain::ReductionDomain::CROWD_LOCAL,
          wftrain::DerivativeExecutionDomain::HOST, 3, 5, 1};
}

void checkClose(wftrain::DerivativeValue actual,
                wftrain::DerivativeValue expected)
{
  CHECK(actual.real() == Catch::Approx(expected.real()).epsilon(1e-13));
  CHECK(actual.imag() == Catch::Approx(expected.imag()).epsilon(1e-13));
}

} // namespace

TEST_CASE("PsiFormer real response fold preserves transpose and Hermitian semantics",
          "[wavefunction][psiformer][derivative_orchestration]")
{
  const PsiFormerDerivativePlan plan = makePlan();
  std::array<double, PARAMETER_COUNT> real_response{};
  std::array<double, PARAMETER_COUNT> imaginary_response{};
  for (std::size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter)
  {
    real_response[parameter] = 0.25 + parameter;
    imaginary_response[parameter] = -0.5 + 0.125 * parameter;
  }
  const std::array<wftrain::DerivativeValue, 3> coefficients{
      wftrain::DerivativeValue{1.0, 2.0}, {-0.5, 0.25}, {0.0, -1.0}};

  for (const wftrain::DerivativeAdjoint adjoint :
       {wftrain::DerivativeAdjoint::TRANSPOSE,
        wftrain::DerivativeAdjoint::HERMITIAN})
  {
    std::array<double, 3 * 64> accumulator_storage{};
    REQUIRE(plan.accumulators.total_fp64_elements <= accumulator_storage.size());
    foldPsiFormerRealResponse(
        plan, real_response.data(), real_response.size(),
        {coefficients.data(), coefficients.size()},
        PsiFormerDerivativeResponsePart::REAL, adjoint,
        accumulator_storage.data(), accumulator_storage.size());
    foldPsiFormerRealResponse(
        plan, imaginary_response.data(), imaginary_response.size(),
        {coefficients.data(), coefficients.size()},
        PsiFormerDerivativeResponsePart::IMAGINARY, adjoint,
        accumulator_storage.data(), accumulator_storage.size());

    const wftrain::DerivativeValue imaginary_unit =
        adjoint == wftrain::DerivativeAdjoint::TRANSPOSE
        ? wftrain::DerivativeValue{0.0, 1.0}
        : wftrain::DerivativeValue{0.0, -1.0};
    for (std::size_t channel = 0; channel < coefficients.size(); ++channel)
      for (std::size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter)
      {
        const std::size_t offset = channel *
            plan.accumulators.complex_accumulator_stride_fp64_elements +
            2 * parameter;
        checkClose({accumulator_storage[offset], accumulator_storage[offset + 1]},
                   coefficients[channel] *
                       (real_response[parameter] +
                        imaginary_unit * imaginary_response[parameter]));
      }
  }
}

TEST_CASE("PsiFormer response fold preflights every channel before mutation",
          "[wavefunction][psiformer][derivative_orchestration]")
{
  const PsiFormerDerivativePlan plan = makePlan();
  std::array<double, PARAMETER_COUNT> response{};
  response.fill(1.0);
  const std::array<wftrain::DerivativeValue, 2> coefficients{
      wftrain::DerivativeValue{1.0, 0.0},
      wftrain::DerivativeValue{std::numeric_limits<double>::quiet_NaN(), 0.0}};
  std::array<double, 3 * 64> accumulator_storage{};
  accumulator_storage.fill(7.0);

  CHECK_THROWS_AS(foldPsiFormerRealResponse(
                      plan, response.data(), response.size(),
                      {coefficients.data(), coefficients.size()},
                      PsiFormerDerivativeResponsePart::REAL,
                      wftrain::DerivativeAdjoint::TRANSPOSE,
                      accumulator_storage.data(), accumulator_storage.size()),
                  std::invalid_argument);
  for (double value : accumulator_storage)
    CHECK(value == 7.0);
}

TEST_CASE("PsiFormer canonical chunks use one synchronized host staging interval",
          "[wavefunction][psiformer][derivative_orchestration]")
{
  const PsiFormerDerivativePlan plan = makePlan();
  const wftrain::StructuredParameterSchema schema = makeSchema();
  const wftrain::ParameterChunkPlan chunks(schema, 19, 5);
  REQUIRE(chunks.chunks().size() == 5);
  const PsiFormerDerivativeHostPublisher publisher(
      plan, wftrain::ParameterChunkPlan(schema, 19, 5));
  const PsiFormerDerivativeCompletionToken completion =
      makePsiFormerVJPCompletionToken(
          plan, wftrain::DerivativeAdjoint::TRANSPOSE, 3);

  const std::array<wftrain::DerivativeValue, 3> coefficients{
      wftrain::DerivativeValue{1.0, 2.0}, {-0.5, 0.25}, {0.0, -1.0}};
  const auto channels = makeChannels(schema, coefficients);
  std::array<double, PARAMETER_COUNT> response{};
  for (std::size_t parameter = 0; parameter < response.size(); ++parameter)
    response[parameter] = 1.0 + 0.5 * parameter;
  std::array<double, 3 * 64> accumulator_storage{};
  foldPsiFormerRealResponse(
      plan, response.data(), response.size(),
      {coefficients.data(), coefficients.size()},
      PsiFormerDerivativeResponsePart::REAL,
      wftrain::DerivativeAdjoint::TRANSPOSE,
      accumulator_storage.data(), accumulator_storage.size());

  std::array<wftrain::DerivativeValue, 5> staging{};
  EventLog log;
  RecordingStager stager(log);
  FixedParameterSink sink(log, staging.data());
  const auto descriptor =
      makeVJPDescriptor(schema, wftrain::DerivativeAdjoint::TRANSPOSE);
  sink.begin(descriptor, chunks, {channels.data(), channels.size()});
  publisher.publishParameterChunks(
      completion, accumulator_storage.data(), accumulator_storage.size(),
      staging.data(), staging.size(), stager, sink);
  sink.end();

  CHECK(sink.alwaysUsedStaging());
  CHECK(log.size == chunks.chunks().size() * channels.size() * 3);
  for (std::size_t event = 0; event < log.size; event += 3)
  {
    CHECK(log.events[event] == EventKind::COPY_COMPLEX);
    CHECK(log.events[event + 1] == EventKind::SYNCHRONIZE);
    CHECK(log.events[event + 2] == EventKind::ADD_PARAMETER);
  }
  for (std::size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter)
  {
    const bool selected = parameter < 18 || parameter >= 20;
    CHECK(sink.touched()[parameter] == selected);
    if (selected)
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
        checkClose(sink.values()[channel][parameter],
                   coefficients[channel] * response[parameter]);
  }
}

TEST_CASE("PsiFormer publication rejects blocks that cross device tensors",
          "[wavefunction][psiformer][derivative_orchestration]")
{
  const PsiFormerDerivativePlan plan = makePlan();
  const wftrain::StructuredParameterSchema crossed(
      "psiformer/orchestration",
      {{"embedding", {9, 2}, 0, 18, wftrain::ParameterScalarDomain::REAL64,
        true, "network"},
       {"crossed", {4}, 18, 4, wftrain::ParameterScalarDomain::REAL64,
        true, "network"}});
  CHECK_THROWS_AS(PsiFormerDerivativeHostPublisher(
                      plan, wftrain::ParameterChunkPlan(crossed, 19, 5)),
                  std::invalid_argument);
}

TEST_CASE("PsiFormer publication preflights sink plan adjoint and channel count",
          "[wavefunction][psiformer][derivative_orchestration]")
{
  const PsiFormerDerivativePlan plan = makePlan();
  const wftrain::StructuredParameterSchema schema = makeSchema();
  const wftrain::ParameterChunkPlan chunks(schema, 19, 5);
  const PsiFormerDerivativeHostPublisher publisher(plan, chunks);
  const std::array<wftrain::DerivativeValue, 3> coefficients{
      wftrain::DerivativeValue{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}};
  const auto channels = makeChannels(schema, coefficients);
  std::array<double, 3 * 64> accumulator_storage{};
  std::array<wftrain::DerivativeValue, 5> staging{};

  {
    EventLog log;
    RecordingStager stager(log);
    FixedParameterSink sink(log, staging.data());
    const auto descriptor =
        makeVJPDescriptor(schema, wftrain::DerivativeAdjoint::HERMITIAN);
    sink.begin(descriptor, chunks, {channels.data(), channels.size()});
    const auto transpose = makePsiFormerVJPCompletionToken(
        plan, wftrain::DerivativeAdjoint::TRANSPOSE, channels.size());
    CHECK_THROWS_AS(publisher.publishParameterChunks(
                        transpose, accumulator_storage.data(),
                        accumulator_storage.size(), staging.data(), staging.size(),
                        stager, sink),
                    std::invalid_argument);
    CHECK(log.size == 0);
    CHECK(sink.state() == wftrain::DerivativeSinkState::ACTIVE);
    sink.abort();
  }

  {
    const wftrain::ParameterChunkPlan diagnostic(
        schema, 19, 5, wftrain::FrozenBlockPolicy::INCLUDE_FOR_DIAGNOSTICS);
    EventLog log;
    RecordingStager stager(log);
    FixedParameterSink sink(log, staging.data());
    const auto descriptor =
        makeVJPDescriptor(schema, wftrain::DerivativeAdjoint::TRANSPOSE);
    sink.begin(descriptor, diagnostic, {channels.data(), channels.size()});
    const auto completion = makePsiFormerVJPCompletionToken(
        plan, wftrain::DerivativeAdjoint::TRANSPOSE, channels.size());
    CHECK_THROWS_AS(publisher.publishParameterChunks(
                        completion, accumulator_storage.data(),
                        accumulator_storage.size(), staging.data(), staging.size(),
                        stager, sink),
                    std::invalid_argument);
    CHECK(log.size == 0);
    CHECK(sink.state() == wftrain::DerivativeSinkState::ACTIVE);
    sink.abort();
  }

  {
    EventLog log;
    RecordingStager stager(log);
    FixedParameterSink sink(log, staging.data());
    const auto descriptor =
        makeVJPDescriptor(schema, wftrain::DerivativeAdjoint::TRANSPOSE);
    sink.begin(descriptor, chunks, {channels.data(), channels.size()});
    const auto two_channels = makePsiFormerVJPCompletionToken(
        plan, wftrain::DerivativeAdjoint::TRANSPOSE, 2);
    CHECK_THROWS_AS(publisher.publishParameterChunks(
                        two_channels, accumulator_storage.data(),
                        accumulator_storage.size(), staging.data(), staging.size(),
                        stager, sink),
                    std::invalid_argument);
    CHECK(log.size == 0);
    CHECK(sink.state() == wftrain::DerivativeSinkState::ACTIVE);
    sink.abort();
  }
}

TEST_CASE("PsiFormer completion gates publication and JVP emits one scalar tile",
          "[wavefunction][psiformer][derivative_orchestration]")
{
  const PsiFormerDerivativePlan plan = makePlan();
  const wftrain::StructuredParameterSchema schema = makeSchema();
  const wftrain::ParameterChunkPlan chunks(schema, 19, 5);
  const PsiFormerDerivativeHostPublisher publisher(plan, chunks);
  const auto completion = makePsiFormerJVPCompletionToken(plan, 4, 7, 0);

  std::array<wftrain::DerivativeValue, 5> staging{};
  EventLog log;
  RecordingStager stager(log);
  FixedSampleSink sink(log, staging.data());
  const auto descriptor = makeJVPDescriptor(schema);
  sink.begin(descriptor);
  const double result = -3.75;
  publisher.publishJVPScalar(completion, &result, staging.data(), staging.size(),
                             stager, sink);
  sink.end();
  checkClose(sink.value(), {-3.75, 0.0});
  CHECK(sink.usedStaging());
  REQUIRE(log.size == 3);
  CHECK(log.events[0] == EventKind::COPY_JVP);
  CHECK(log.events[1] == EventKind::SYNCHRONIZE);
  CHECK(log.events[2] == EventKind::ADD_SAMPLE);

  log.size = 0;
  FixedSampleSink rejected_sink(log, staging.data());
  const auto rejected_descriptor = makeJVPDescriptor(schema);
  rejected_sink.begin(rejected_descriptor);
  PsiFormerDerivativeCompletionToken failed = completion;
  failed.status = 1;
  CHECK_THROWS_AS(publisher.publishJVPScalar(
                      failed, &result, staging.data(), staging.size(), stager,
                      rejected_sink),
                  std::runtime_error);
  CHECK(log.size == 0);
  rejected_sink.abort();

  log.size = 0;
  FixedSampleSink mismatched_sink(log, staging.data());
  const auto mismatched_descriptor = makeJVPDescriptor(schema);
  mismatched_sink.begin(mismatched_descriptor);
  const auto wrong_sample = makePsiFormerJVPCompletionToken(plan, 4, 8, 1);
  CHECK_THROWS_AS(publisher.publishJVPScalar(
                      wrong_sample, &result, staging.data(), staging.size(), stager,
                      mismatched_sink),
                  std::invalid_argument);
  CHECK(log.size == 0);
  CHECK(mismatched_sink.state() == wftrain::DerivativeSinkState::ACTIVE);
  mismatched_sink.abort();
}

} // namespace qmcplusplus::psiformer

int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
