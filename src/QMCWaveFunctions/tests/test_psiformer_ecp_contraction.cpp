//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_ecp_contraction.cpp
 * @brief CPU oracle tests for bounded flattened ECP contraction.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerECPContraction.h"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

namespace qmcplusplus::psiformer
{
namespace
{

static_assert(!std::is_copy_constructible_v<PsiFormerECPContraction>);
static_assert(!std::is_copy_assignable_v<PsiFormerECPContraction>);
static_assert(!std::is_move_constructible_v<PsiFormerECPContraction>);
static_assert(!std::is_move_assignable_v<PsiFormerECPContraction>);

class RawAlignedStorage
{
public:
  explicit RawAlignedStorage(std::size_t bytes)
      : bytes_(bytes),
        data_(::operator new(bytes_,
                             std::align_val_t{alignof(std::max_align_t)}))
  {}
  RawAlignedStorage(const RawAlignedStorage&) = delete;
  RawAlignedStorage& operator=(const RawAlignedStorage&) = delete;
  ~RawAlignedStorage()
  {
    ::operator delete(data_, std::align_val_t{alignof(std::max_align_t)});
  }

  void* data() noexcept { return data_; }
  std::size_t size() const noexcept { return bytes_; }

private:
  std::size_t bytes_;
  void* data_;
};

constexpr std::size_t PARAMETER_COUNT = 22;

QMCTraits::ValueType makeQMCValue(double real, double imaginary)
{
#ifdef QMC_COMPLEX
  return {real, imaginary};
#else
  (void)imaginary;
  return real;
#endif
}

wftrain::DerivativeValue derivativeValue(QMCTraits::ValueType value)
{
  return {static_cast<double>(std::real(value)),
          static_cast<double>(std::imag(value))};
}

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
  layout.fingerprint = 26074;
  return layout;
}

PsiFormerDerivativePlan makePlan()
{
  const PsiFormerDeviceLayout device = makeDeviceLayout();
  PsiFormerDerivativePlanningRequirements requirements;
  requirements.retained_tape_fp64_elements =
      minimumPsiFormerScoreTapeFP64Elements(device);
  requirements.adjoint_tape_fp64_elements = 31;
  requirements.complex_accumulator_count = 3;
  requirements.maximum_parameter_chunk_elements = 5;
  requirements.maximum_ecp_sample_count = 2;
  requirements.maximum_ecp_segment_count = 3;
  requirements.maximum_ecp_point_count = 3;
  requirements.maximum_ecp_evaluation_stamp_count = 2;
  requirements.block_size = 8;
  return makePsiFormerDerivativePlan(device, 41, requirements);
}

class AnalyticScoreEvaluator final : public PsiFormerECPScoreEvaluator
{
public:
  struct Call
  {
    std::size_t replaced_electron;
    bool reference;
  };

  void mutateLiveInputsOnFirstCall(VirtualParticleBatch::PosType* positions,
                                   QMCTraits::ValueType* bare_weights,
                                   QMCTraits::ValueType* ratios) noexcept
  {
    hostile_positions_ = positions;
    hostile_bare_weights_ = bare_weights;
    hostile_ratios_ = ratios;
  }

  void evaluate(const double* reference_positions,
                std::size_t electron_count,
                std::size_t replaced_electron,
                const double* replacement_position,
                double* response,
                std::size_t parameter_count) override
  {
    if (response_address_ == nullptr)
      response_address_ = response;
    else
      CHECK(response == response_address_);
    REQUIRE(call_count_ < calls_.size());
    calls_[call_count_++] = {replaced_electron, replacement_position == nullptr};

    double coordinate_sum = 0.0;
    for (std::size_t electron = 0; electron < electron_count; ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        coordinate_sum += reference_positions[3 * electron + dimension];
    if (replacement_position != nullptr)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        coordinate_sum += replacement_position[dimension] -
            reference_positions[3 * replaced_electron + dimension];
    for (std::size_t parameter = 0; parameter < parameter_count; ++parameter)
      response[parameter] =
          (1.0 + static_cast<double>(parameter)) * coordinate_sum +
          0.125 * static_cast<double>(parameter);
    if (!mutated_ && hostile_positions_ && hostile_bare_weights_ &&
        hostile_ratios_)
    {
      hostile_positions_[1] = {91.0, 92.0, 93.0};
      hostile_positions_[2] = {94.0, 95.0, 96.0};
      hostile_bare_weights_[1] = makeQMCValue(97.0, 0.0);
      hostile_ratios_[1] = makeQMCValue(98.0, 99.0);
      mutated_ = true;
    }
  }

  std::size_t callCount() const noexcept { return call_count_; }
  const Call& call(std::size_t index) const noexcept { return calls_[index]; }

private:
  double* response_address_ = nullptr;
  std::array<Call, 16> calls_{};
  std::size_t call_count_ = 0;
  VirtualParticleBatch::PosType* hostile_positions_ = nullptr;
  QMCTraits::ValueType* hostile_bare_weights_ = nullptr;
  QMCTraits::ValueType* hostile_ratios_ = nullptr;
  bool mutated_ = false;
};

class MaximumScoreEvaluator final : public PsiFormerECPScoreEvaluator
{
public:
  void evaluate(const double*, std::size_t, std::size_t, const double*,
                double* response, std::size_t parameter_count) override
  {
    std::fill(response, response + parameter_count,
              std::numeric_limits<double>::max());
  }
};

wftrain::DerivativeValue expectedDelta(
    const std::array<double, 12>& references,
    std::size_t sample,
    std::size_t electron,
    const VirtualParticleBatch::PosType& replacement,
    std::size_t parameter)
{
  double delta_sum = 0.0;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    delta_sum += static_cast<double>(replacement[dimension]) -
        references[(sample * 2 + electron) * 3 + dimension];
  return {(1.0 + static_cast<double>(parameter)) * delta_sum, 0.0};
}

class DirectStager final : public PsiFormerDerivativeHostStager
{
public:
  void copyComplexChunk(const double* source,
                        std::size_t count,
                        wftrain::DerivativeValue* destination) override
  {
    for (std::size_t index = 0; index < count; ++index)
      destination[index] = {source[2 * index], source[2 * index + 1]};
  }
  void copyRealJVPScalar(const double*, wftrain::DerivativeValue*) override {}
  void synchronize() override {}
};

class CaptureSink final : public wftrain::ParameterReductionSink
{
public:
  const auto& values() const noexcept { return values_; }

protected:
  void consume(std::size_t channel,
               const wftrain::ParameterChunkConstView& chunk) override
  {
    for (std::size_t index = 0; index < chunk.values().size(); ++index)
      values_[channel][chunk.descriptor().parameter_offset + index] =
          chunk.values()[index];
  }

private:
  std::array<std::array<wftrain::DerivativeValue, PARAMETER_COUNT>, 3> values_{};
};

wftrain::StructuredParameterSchema makeSchema()
{
  return wftrain::StructuredParameterSchema(
      "psiformer/ecp-offline",
      {{"embedding", {9, 2}, 0, 18, wftrain::ParameterScalarDomain::REAL64,
        true, "network"},
       {"hidden", {2}, 18, 2, wftrain::ParameterScalarDomain::REAL64,
        true, "network"},
       {"output", {2}, 20, 2, wftrain::ParameterScalarDomain::REAL64,
        true, "network"}});
}

} // namespace

TEST_CASE("PsiFormer ECP tile copy and contraction preserve sparse order",
          "[wavefunction][psiformer][ecp_contraction]")
{
  const PsiFormerDerivativePlan plan = makePlan();
  RawAlignedStorage tile_storage(plan.ecp_tile.total_bytes);
  std::array<double, PARAMETER_COUNT> response{};
  std::vector<double> accumulators(plan.accumulators.total_fp64_elements, -9.0);
  AnalyticScoreEvaluator evaluator;
  PsiFormerECPContraction contraction(
      plan, tile_storage.data(), tile_storage.size(),
      response.data(), response.size(), accumulators.data(), accumulators.size(),
      evaluator);

  const std::array<double, 12> references{
      0.0, 0.1, 0.2, 1.0, 1.1, 1.2,
      2.0, 2.1, 2.2, 3.0, 3.1, 3.2};
  const std::array<wftrain::DerivativeValue, 4> coefficients{
      wftrain::DerivativeValue{1.0, 0.5}, {-0.25, 1.0},
      wftrain::DerivativeValue{2.0, -0.5}, {0.75, 0.25}};
  contraction.begin({901, 902, 4, 2, 2,
                     wftrain::DerivativeAdjoint::TRANSPOSE},
                    {references.data(), references.size()},
                    {coefficients.data(), coefficients.size()});

  const std::array<std::size_t, 3> offsets0{0, 2, 3};
  const std::array<VirtualParticleBatch::Segment, 2> segments0{
      VirtualParticleBatch::Segment(0, 1, true, 1),
      VirtualParticleBatch::Segment(1, 0)};
  std::array<VirtualParticleBatch::PosType, 3> positions0{
      VirtualParticleBatch::PosType{1.4, 1.5, 1.6},
      VirtualParticleBatch::PosType{0.7, 0.8, 0.9},
      VirtualParticleBatch::PosType{2.5, 2.6, 2.7}};
  const auto original_positions0 = positions0;
  const VirtualParticleBatch batch0(
      2, {offsets0.data(), offsets0.size()},
      {segments0.data(), segments0.size()},
      {positions0.data(), positions0.size()});
  const std::uint64_t original_batch0_fingerprint = batch0.fingerprint();
  std::array<QMCTraits::ValueType, 3> bare0{2.0, -0.5, 1.25};
  std::array<QMCTraits::ValueType, 3> ratios0{
      makeQMCValue(0.5, 0.25), makeQMCValue(1.5, -0.5),
      makeQMCValue(-0.8, 0.4)};
  const auto original_bare0 = bare0;
  const auto original_ratios0 = ratios0;
  evaluator.mutateLiveInputsOnFirstCall(positions0.data(), bare0.data(),
                                        ratios0.data());
  int opaque_source = 0;
  const std::array<WaveFunctionComponent::EvaluationStamp, 1> live_stamps{
      WaveFunctionComponent::EvaluationStamp::versioned(&opaque_source, 41)};
  const std::array<PsiFormerECPEvaluationStampRecord, 1> stamp_records{
      PsiFormerECPEvaluationStampRecord{902, 41}};
  const PsiFormerECPTileLaunchRecord first = contraction.consume(
      {batch0, {bare0.data(), bare0.size()}, {ratios0.data(), ratios0.size()},
       {live_stamps.data(), live_stamps.size()}, 0, 901},
      {stamp_records.data(), stamp_records.size()});
  CHECK(first.metadata.flat_point_begin == 0);
  CHECK(first.metadata.tile_fingerprint == original_batch0_fingerprint);
  CHECK(first.metadata.point_count == 3);
  CHECK(first.launch.tail_count == 3);
  const auto checked_first =
      makePsiFormerECPTileLaunchRecord(plan, first.metadata, 2);
  CHECK(first.fingerprint == checked_first.fingerprint);
  CHECK(first.device_arena_fingerprint ==
        checked_first.device_arena_fingerprint);

  const PsiFormerCopiedECPTileView copied = contraction.copiedTileView();
  CHECK(copied.metadata().tile_fingerprint == original_batch0_fingerprint);
  CHECK(copied.segmentOffset(0) == 0);
  CHECK(copied.segmentOffset(1) == 2);
  CHECK(copied.segmentOffset(2) == 3);
  CHECK(copied.segmentWalkerId(0) == 0);
  CHECK(copied.segmentElectronId(0) == 1);
  CHECK(copied.segmentSourceCenterId(0) == 1);
  CHECK(copied.segmentFlags(0) == ECP_SEGMENT_ON_SPHERE);
  CHECK(copied.segmentSourceCenterId(1) == PSIFORMER_ECP_NO_SOURCE);
  CHECK(copied.segmentFlags(1) == 0);
  CHECK(copied.replacementPosition(1)[0] == Catch::Approx(0.7));
  CHECK(copied.bareWeight(2) == Catch::Approx(1.25));
  CHECK(copied.completeRatio(2).real() == Catch::Approx(-0.8));
  CHECK(copied.completeRatio(2).imag() ==
        Catch::Approx(static_cast<double>(std::imag(original_ratios0[2]))));
  CHECK(copied.evaluationStamp(0).producer_fingerprint == 902);
  CHECK(copied.evaluationStamp(0).parameter_version == 41);
  CHECK(copied.referencePosition(11) == Catch::Approx(3.2));
  CHECK(copied.coefficient(0).imag() == Catch::Approx(0.5));
  CHECK(copied.pointStatus(2) == 0);

  const std::array<std::size_t, 2> offsets1{0, 1};
  const std::array<VirtualParticleBatch::Segment, 1> segments1{
      VirtualParticleBatch::Segment(0, 1, true, 1)};
  const std::array<VirtualParticleBatch::PosType, 1> positions1{
      VirtualParticleBatch::PosType{1.8, 1.9, 2.0}};
  const VirtualParticleBatch batch1(
      2, {offsets1.data(), offsets1.size()},
      {segments1.data(), segments1.size()},
      {positions1.data(), positions1.size()});
  const std::array<QMCTraits::ValueType, 1> bare1{0.75};
  const std::array<QMCTraits::ValueType, 1> ratios1{
      makeQMCValue(2.0, -0.75)};
  const PsiFormerECPTileLaunchRecord second = contraction.consume(
      {batch1, {bare1.data(), bare1.size()}, {ratios1.data(), ratios1.size()},
       {live_stamps.data(), live_stamps.size()}, 1, 901},
      {stamp_records.data(), stamp_records.size()});
  CHECK(second.metadata.flat_point_begin == 3);
  CHECK(contraction.consumedPointCount() == 4);

  const PsiFormerDerivativeCompletionToken completion = contraction.finish();
  CHECK(completion.channel_count == 2);
  CHECK(completion.parameter_version == 41);
  CHECK(completion.status == 0);
  const auto checked_completion = makePsiFormerVJPCompletionToken(
      plan, wftrain::DerivativeAdjoint::TRANSPOSE, 2, 0);
  CHECK(completion.plan_fingerprint == checked_completion.plan_fingerprint);
  CHECK(completion.device_layout_fingerprint ==
        checked_completion.device_layout_fingerprint);
  CHECK(completion.device_arena_fingerprint ==
        checked_completion.device_arena_fingerprint);
  CHECK(completion.parameter_version == checked_completion.parameter_version);
  CHECK(completion.kind == checked_completion.kind);
  CHECK(completion.adjoint == checked_completion.adjoint);
  CHECK(completion.channel_count == checked_completion.channel_count);
  CHECK(completion.batch_ordinal == checked_completion.batch_ordinal);
  CHECK(completion.sample_offset == checked_completion.sample_offset);
  CHECK(completion.tile_ordinal == checked_completion.tile_ordinal);
  CHECK(completion.status == checked_completion.status);
  CHECK(evaluator.callCount() == 6);
  CHECK_FALSE(evaluator.call(0).reference);
  CHECK_FALSE(evaluator.call(3).reference);
  CHECK(evaluator.call(4).reference);
  CHECK(evaluator.call(5).reference);

  const std::array<wftrain::DerivativeValue, 4> weights{
      derivativeValue(original_bare0[0] * original_ratios0[0]),
      derivativeValue(original_bare0[1] * original_ratios0[1]),
      derivativeValue(original_bare0[2] * original_ratios0[2]),
      derivativeValue(bare1[0] * ratios1[0])};
  for (std::size_t channel = 0; channel < 2; ++channel)
    for (std::size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter)
    {
      wftrain::DerivativeValue expected{};
      expected += coefficients[channel * 2] * weights[0] *
          expectedDelta(references, 0, 1, original_positions0[0], parameter);
      expected += coefficients[channel * 2] * weights[1] *
          expectedDelta(references, 0, 1, original_positions0[1], parameter);
      expected += coefficients[channel * 2 + 1] * weights[2] *
          expectedDelta(references, 1, 0, original_positions0[2], parameter);
      expected += coefficients[channel * 2] * weights[3] *
          expectedDelta(references, 0, 1, positions1[0], parameter);
      const std::size_t offset =
          channel * plan.accumulators.complex_accumulator_stride_fp64_elements +
          2 * parameter;
      CHECK(accumulators[offset] == Catch::Approx(expected.real()).epsilon(1e-13));
      CHECK(accumulators[offset + 1] ==
            Catch::Approx(expected.imag()).epsilon(1e-13));
    }

  // A7.4 hands the completed accumulators to the existing synchronized A7.3
  // publication boundary; it does not invent a device-visible public sink.
  const wftrain::StructuredParameterSchema schema = makeSchema();
  const wftrain::ParameterChunkPlan chunks(schema, 41, 5);
  PsiFormerDerivativeHostPublisher publisher(
      plan, wftrain::ParameterChunkPlan(schema, 41, 5));
  std::array<std::array<wftrain::DerivativeValue, 2>, 2> coefficient_rows{{
      {coefficients[0], coefficients[1]},
      {coefficients[2], coefficients[3]}}};
  std::array<wftrain::VJPCoefficientChannel, 2> channels;
  for (std::size_t channel = 0; channel < channels.size(); ++channel)
    channels[channel] = {
        channel == 0 ? "first" : "second",
        wftrain::DerivativeProduct::LOCAL_ENERGY_VJP,
        {schema.providerId(), schema.fingerprint(), 41, 3, 9,
         {coefficient_rows[channel].data(), 2}},
        wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP)};
  const wftrain::DerivativeStreamDescriptor descriptor{
      schema.providerId(), schema.fingerprint(), 41,
      wftrain::derivativeProductBit(wftrain::DerivativeProduct::LOCAL_ENERGY_VJP),
      wftrain::DerivativeAdjoint::TRANSPOSE,
      wftrain::ParameterScalarDomain::REAL64,
      wftrain::ParameterScalarDomain::COMPLEX128, 3, 9, 2,
      wftrain::ReductionDomain::CROWD_LOCAL,
      wftrain::DerivativeExecutionDomain::HOST, 3, 5, 1};
  std::array<wftrain::DerivativeValue, 5> staging{};
  DirectStager stager;
  CaptureSink sink;
  sink.begin(descriptor, chunks, {channels.data(), channels.size()});
  publisher.publishParameterChunks(
      completion, accumulators.data(), accumulators.size(), staging.data(),
      staging.size(), stager, sink);
  sink.end();
  for (std::size_t channel = 0; channel < 2; ++channel)
    for (std::size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter)
    {
      const std::size_t offset =
          channel * plan.accumulators.complex_accumulator_stride_fp64_elements +
          2 * parameter;
      CHECK(sink.values()[channel][parameter].real() ==
            Catch::Approx(accumulators[offset]).epsilon(1e-13));
      CHECK(sink.values()[channel][parameter].imag() ==
            Catch::Approx(accumulators[offset + 1]).epsilon(1e-13));
    }
}

TEST_CASE("PsiFormer ECP contraction rejects unbound stamp translations",
          "[wavefunction][psiformer][ecp_contraction]")
{
  const PsiFormerDerivativePlan plan = makePlan();
  RawAlignedStorage tile_storage(plan.ecp_tile.total_bytes);
  std::array<double, PARAMETER_COUNT> response{};
  std::vector<double> accumulators(plan.accumulators.total_fp64_elements);
  AnalyticScoreEvaluator evaluator;
  PsiFormerECPContraction contraction(
      plan, tile_storage.data(), tile_storage.size(),
      response.data(), response.size(), accumulators.data(), accumulators.size(),
      evaluator);
  std::array<double, 12> references{};
  std::array<wftrain::DerivativeValue, 2> coefficients{
      wftrain::DerivativeValue{1.0, 0.0}, {1.0, 0.0}};
  contraction.begin({11, 12, 1, 2, 1,
                     wftrain::DerivativeAdjoint::TRANSPOSE},
                    {references.data(), references.size()},
                    {coefficients.data(), coefficients.size()});

  const std::array<std::size_t, 2> offsets{0, 1};
  const std::array<VirtualParticleBatch::Segment, 1> segments{
      VirtualParticleBatch::Segment(0, 0)};
  const std::array<VirtualParticleBatch::PosType, 1> positions{
      VirtualParticleBatch::PosType{0.1, 0.2, 0.3}};
  const VirtualParticleBatch batch(
      2, {offsets.data(), offsets.size()}, {segments.data(), segments.size()},
      {positions.data(), positions.size()});
  const std::array<QMCTraits::ValueType, 1> bare{1.0};
  const std::array<QMCTraits::ValueType, 1> ratios{1.0};
  int opaque_source = 0;
  const std::array<WaveFunctionComponent::EvaluationStamp, 1> live_stamps{
      WaveFunctionComponent::EvaluationStamp::versioned(&opaque_source, 41)};
  const std::array<PsiFormerECPEvaluationStampRecord, 1> wrong_record{
      PsiFormerECPEvaluationStampRecord{99, 41}};
  CHECK_THROWS_AS(contraction.consume(
                      {batch, {bare.data(), bare.size()},
                       {ratios.data(), ratios.size()},
                       {live_stamps.data(), live_stamps.size()}, 0, 11},
                      {wrong_record.data(), wrong_record.size()}),
                  std::invalid_argument);
  CHECK_THROWS_AS(contraction.finish(), std::logic_error);
  CHECK(evaluator.callCount() == 0);
}

TEST_CASE("PsiFormer ECP contraction rejects aliases and never completes nonfinite output",
          "[wavefunction][psiformer][ecp_contraction]")
{
  const PsiFormerDerivativePlan plan = makePlan();
  RawAlignedStorage tile_storage(plan.ecp_tile.total_bytes);
  std::array<double, PARAMETER_COUNT> response{};
  std::vector<double> accumulators(plan.accumulators.total_fp64_elements);
  MaximumScoreEvaluator evaluator;
  CHECK_THROWS_AS(PsiFormerECPContraction(
                      plan, tile_storage.data(), tile_storage.size(),
                      reinterpret_cast<double*>(tile_storage.data()),
                      PARAMETER_COUNT, accumulators.data(), accumulators.size(),
                      evaluator),
                  std::invalid_argument);

  PsiFormerECPContraction contraction(
      plan, tile_storage.data(), tile_storage.size(),
      response.data(), response.size(), accumulators.data(), accumulators.size(),
      evaluator);
  std::array<double, 12> references{};
  std::array<wftrain::DerivativeValue, 2> coefficients{
      wftrain::DerivativeValue{1.0, 0.0}, {1.0, 0.0}};
  contraction.begin({71, 72, 1, 2, 1,
                     wftrain::DerivativeAdjoint::TRANSPOSE},
                    {references.data(), references.size()},
                    {coefficients.data(), coefficients.size()});
  const std::array<std::size_t, 2> offsets{0, 1};
  const std::array<VirtualParticleBatch::Segment, 1> segments{
      VirtualParticleBatch::Segment(0, 0)};
  const std::array<VirtualParticleBatch::PosType, 1> positions{
      VirtualParticleBatch::PosType{0.1, 0.2, 0.3}};
  const VirtualParticleBatch batch(
      2, {offsets.data(), offsets.size()}, {segments.data(), segments.size()},
      {positions.data(), positions.size()});
  const std::array<QMCTraits::ValueType, 1> bare{2.0};
  const std::array<QMCTraits::ValueType, 1> ratios{makeQMCValue(1.0, 0.0)};
  int opaque_source = 0;
  const std::array<WaveFunctionComponent::EvaluationStamp, 1> live_stamps{
      WaveFunctionComponent::EvaluationStamp::versioned(&opaque_source, 41)};
  const std::array<PsiFormerECPEvaluationStampRecord, 1> stamp_records{
      PsiFormerECPEvaluationStampRecord{72, 41}};
  CHECK_NOTHROW(contraction.consume(
      {batch, {bare.data(), bare.size()}, {ratios.data(), ratios.size()},
       {live_stamps.data(), live_stamps.size()}, 0, 71},
      {stamp_records.data(), stamp_records.size()}));
  CHECK_THROWS_AS(contraction.finish(), std::invalid_argument);
  CHECK_THROWS_AS(contraction.finish(), std::logic_error);
}

} // namespace qmcplusplus::psiformer

int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
