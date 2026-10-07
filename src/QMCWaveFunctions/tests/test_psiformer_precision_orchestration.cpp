//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_precision_orchestration.cpp
 * @brief Focused CPU tests for the pointer-free mixed-precision event recorder.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerPrecisionOrchestration.h"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <new>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace qmcplusplus::psiformer;

namespace allocation_probe
{

thread_local bool enabled = false;
thread_local std::size_t count = 0;

/// Record one allocation made while a focused recorder call is under observation.
void noteAllocation() noexcept
{
  if (enabled)
    ++count;
}

/// Measure actual global allocation calls in the current thread over one scope.
class Scope
{
public:
  Scope()
  {
    count = 0;
    enabled = true;
  }

  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

  ~Scope() { enabled = false; }

  std::size_t finish() noexcept
  {
    enabled = false;
    return count;
  }
};

} // namespace allocation_probe

/** Route this focused executable's ordinary allocations through the thread-local probe. */
void* operator new(std::size_t bytes)
{
  allocation_probe::noteAllocation();
  if (void* storage = std::malloc(bytes == 0 ? 1 : bytes))
    return storage;
  throw std::bad_alloc();
}

/// Route array allocations through the same focused allocation probe.
void* operator new[](std::size_t bytes)
{
  return ::operator new(bytes);
}

/// Release storage allocated by the focused executable's global operator new.
void operator delete(void* storage) noexcept
{
  std::free(storage);
}

/// Release array storage allocated by the focused executable.
void operator delete[](void* storage) noexcept
{
  std::free(storage);
}

/// Provide the sized-delete partner selected by some C++17 toolchains.
void operator delete(void* storage, std::size_t) noexcept
{
  std::free(storage);
}

/// Provide the sized array-delete partner selected by some C++17 toolchains.
void operator delete[](void* storage, std::size_t) noexcept
{
  std::free(storage);
}

namespace
{

/// Construct the small B=2 padded descriptor set used by all recording tests.
PsiFormerMixedValueDescriptors makeDescriptors()
{
  const DenseForwardLayout dense = makeDenseForwardLayout(
      /*rows=*/6, /*input_width=*/5, /*output_width=*/4,
      /*source_row_stride=*/7, /*weight_row_stride=*/6,
      /*target_row_stride=*/7);
  const AttentionForwardLayout attention = makeAttentionForwardLayout(
      /*rows=*/3, /*heads=*/2, /*head_width=*/2,
      /*feature_row_stride=*/7, /*attention_row_stride=*/5,
      /*attention_head_stride=*/17);
  const BatchedAttentionForwardLayout batched =
      makeBatchedAttentionForwardLayout(
          /*configuration_count=*/2, attention,
          /*feature_configuration_stride=*/21,
          /*attention_configuration_stride=*/36);
  const BatchedValueLayout dense_source = makeBatchedValueLayout(
      /*configuration_count=*/2, /*rows=*/3, /*width=*/5,
      /*row_stride=*/7, /*configuration_stride=*/21);
  const BatchedValueLayout value = makeBatchedValueLayout(
      /*configuration_count=*/2, /*rows=*/3, /*width=*/4,
      /*row_stride=*/7, /*configuration_stride=*/21);
  return makePsiFormerMixedValueDescriptors(
      dense, dense_source, batched, value);
}

/** Supply aggregate maxima representative of several blocks and sensitive sites. */
PsiFormerExecutionDiagnosticBounds makeDiagnosticBounds()
{
  return makePsiFormerExecutionDiagnosticBounds(
      /*nonfinite_count=*/64, /*invalid_softmax_count=*/12,
      /*small_determinant_pivot_count=*/8,
      /*severe_cancellation_count=*/8, /*extreme_ecp_ratio_count=*/16);
}

/// Construct one exact mixed plan for the requested canonical version.
PsiFormerPrecisionExecutionPlan makePlan(std::size_t version = 5,
                                         bool retry_available = true,
                                         PsiFormerAcceleratorBackend backend =
                                             PsiFormerAcceleratorBackend::CUDA)
{
  return makePsiFormerPrecisionExecutionPlan(
      PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
      PsiFormerBackendMathMode::FP32_STRICT, backend,
      /*tf32_hardware_supported=*/false, /*parameter_count=*/10,
      /*cast_tile_parameters=*/4, /*block_size=*/3,
      /*device_layout_fingerprint=*/77, version, retry_available);
}

/// Count one event kind without coupling tests to vector implementation details.
std::size_t countEvents(const std::vector<PsiFormerPrecisionRecordEvent>& events,
                        PsiFormerPrecisionRecordEventKind kind)
{
  return static_cast<std::size_t>(
      std::count_if(events.begin(), events.end(),
                    [kind](const PsiFormerPrecisionRecordEvent& event) {
                      return event.kind == kind;
                    }));
}

} // namespace

TEST_CASE("PsiFormer mixed descriptors require one exact padded interpretation",
          "[psiformer][precision_orchestration]")
{
  const PsiFormerMixedValueDescriptors descriptors = makeDescriptors();
  CHECK(descriptors.fingerprint != 0);
  CHECK(descriptors.dense.targetElements() == 39);
  CHECK(descriptors.dense_source.storageElements() == 40);
  CHECK(descriptors.attention.featureElements() == 39);
  CHECK(descriptors.value.storageElements() == 39);
  CHECK(makeDescriptors().fingerprint == descriptors.fingerprint);

  const PsiFormerExecutionDiagnosticBounds bounds = makeDiagnosticBounds();
  CHECK(bounds.aggregate_count == 108);
  CHECK(bounds.fingerprint != 0);
  CHECK(makeDiagnosticBounds().fingerprint == bounds.fingerprint);

  DenseForwardLayout bad_dense = descriptors.dense;
  bad_dense.rows = 5;
  CHECK_THROWS_AS(makePsiFormerMixedValueDescriptors(
                      bad_dense, descriptors.dense_source,
                      descriptors.attention, descriptors.value),
                  std::invalid_argument);

  bad_dense = descriptors.dense;
  bad_dense.output_width = 3;
  CHECK_THROWS_AS(makePsiFormerMixedValueDescriptors(
                      bad_dense, descriptors.dense_source,
                      descriptors.attention, descriptors.value),
                  std::invalid_argument);

  BatchedValueLayout bad_source = descriptors.dense_source;
  bad_source.configuration_stride += 1;
  CHECK_THROWS_WITH(makePsiFormerMixedValueDescriptors(
                        descriptors.dense, bad_source,
                        descriptors.attention, descriptors.value),
                    Catch::Matchers::ContainsSubstring("source layout"));

  BatchedAttentionForwardLayout bad_attention = descriptors.attention;
  bad_attention.feature_configuration_stride += 1;
  CHECK_THROWS_AS(makePsiFormerMixedValueDescriptors(
                      descriptors.dense, descriptors.dense_source,
                      bad_attention, descriptors.value),
                  std::invalid_argument);

  BatchedValueLayout bad_value = descriptors.value;
  bad_value.row_stride += 1;
  CHECK_THROWS_AS(makePsiFormerMixedValueDescriptors(
                      descriptors.dense, descriptors.dense_source,
                      descriptors.attention, bad_value),
                  std::invalid_argument);
}

TEST_CASE("PsiFormer recording preparation seals allocation and slot identities",
          "[psiformer][precision_orchestration]")
{
  PsiFormerPrecisionRecordingOrchestrator recorder(
      makePlan(), makeDescriptors(), makeDiagnosticBounds(), /*active_version=*/4,
      /*active_fingerprint=*/41, /*active_slot=*/0);
  CHECK_FALSE(recorder.prepared());
  CHECK_THROWS_AS(recorder.recordPublicationStart(), std::logic_error);

  recorder.prepare();
  CHECK(recorder.prepared());
  CHECK(recorder.allocationCount() == 1);
  CHECK(recorder.preparedEventCapacity() >= 14);
  CHECK_THROWS_AS(recorder.prepare(), std::logic_error);
  const std::size_t prepared_capacity = recorder.preparedEventCapacity();

  CHECK_THROWS_AS(recorder.recordValueEvaluationStart(
                      /*version=*/5, recorder.metadata().plan_fingerprint),
                  std::logic_error);
  CHECK(recorder.events().empty());

  allocation_probe::Scope publication_probe;
  recorder.recordPublicationStart();
  const std::size_t publication_allocations = publication_probe.finish();
  CHECK(publication_allocations == 0);
  CHECK(recorder.publicationState().activeVersion() == 4);
  CHECK(recorder.publicationState().publicationPending());
  const std::vector<PsiFormerPrecisionRecordEvent> start_graph = recorder.events();
  CHECK_THROWS_AS(recorder.recordValueEvaluationStart(
                      /*version=*/5, recorder.metadata().plan_fingerprint),
                  std::logic_error);
  CHECK(recorder.events() == start_graph);

  PsiFormerParameterConversionDiagnostics excessive_conversion;
  excessive_conversion.overflow_count = 1;
  excessive_conversion.subnormal_output_count = 10;
  CHECK_THROWS_AS(recorder.observePublicationCompletion(excessive_conversion),
                  std::invalid_argument);
  CHECK(recorder.events() == start_graph);
  CHECK(recorder.publicationState().publicationPending());

  allocation_probe::Scope completion_probe;
  const PsiFormerPrecisionPublicationResult result =
      recorder.observePublicationCompletion({});
  const std::size_t completion_allocations = completion_probe.finish();
  CHECK(result == PsiFormerPrecisionPublicationResult::PUBLISHED);
  CHECK(completion_allocations == 0);
  CHECK(recorder.allocationCount() == 1);
  CHECK(recorder.preparedEventCapacity() == prepared_capacity);
  CHECK(recorder.publicationState().activeVersion() == 5);
  CHECK(recorder.publicationState().activeFingerprint() ==
        recorder.metadata().plan_fingerprint);
  CHECK(recorder.publicationState().activeSlot() == 1);
  CHECK_FALSE(recorder.publicationState().publicationPending());

  const auto& events = recorder.events();
  REQUIRE(events.size() == 10);
  CHECK(events[0].kind ==
        PsiFormerPrecisionRecordEventKind::INACTIVE_MASTER_COPY);
  CHECK(events[0].slot == 1);
  CHECK(events[0].destination_region ==
        PsiFormerDeviceArenaRegion::MODEL_PARAMETERS_STAGING);
  CHECK(events[1].kind ==
        PsiFormerPrecisionRecordEventKind::COMBINED_DIAGNOSTICS_CLEAR);
  CHECK(events[1].diagnostic_offset == 0);
  CHECK(events[1].diagnostic_bytes ==
        sizeof(PsiFormerDeviceNumericalDiagnostics));
  CHECK(events[1].diagnostic_epoch == 1);

  const std::vector<std::size_t> expected_begins{0, 4, 8};
  const std::vector<std::size_t> expected_counts{4, 4, 2};
  for (std::size_t tile = 0; tile < expected_begins.size(); ++tile)
  {
    const PsiFormerPrecisionRecordEvent& event = events[tile + 2];
    CHECK(event.kind ==
          PsiFormerPrecisionRecordEventKind::PARAMETER_CONVERSION_TILE);
    CHECK(event.source_begin == expected_begins[tile]);
    CHECK(event.destination_begin == expected_begins[tile]);
    CHECK(event.count == expected_counts[tile]);
    CHECK(event.slot == 1);
    CHECK(event.source_region ==
          PsiFormerDeviceArenaRegion::MODEL_PARAMETERS_STAGING);
    CHECK(event.destination_region ==
          PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_1);
    CHECK(event.diagnostic_offset ==
          offsetof(PsiFormerDeviceNumericalDiagnostics, conversion));
    CHECK(event.diagnostic_bytes ==
          sizeof(PsiFormerParameterConversionDiagnostics));
    CHECK(event.diagnostic_increment_bound == 10);
    CHECK(event.diagnostic_epoch == 1);
  }
  CHECK(events[5].kind == PsiFormerPrecisionRecordEventKind::
        PARAMETER_DEVICE_COMPLETION_BARRIER);
  CHECK(events[6].kind == PsiFormerPrecisionRecordEventKind::
        PARAMETER_DIAGNOSTIC_READBACK_BARRIER);
  CHECK(events[7].kind ==
        PsiFormerPrecisionRecordEventKind::PARAMETER_DIAGNOSTIC_DECISION);
  CHECK_FALSE(events[7].hard_hazard);
  CHECK(events[8].kind ==
        PsiFormerPrecisionRecordEventKind::PARAMETER_COMPLETION);
  CHECK(events[8].diagnostic_epoch == 1);
  CHECK(events[9].kind ==
        PsiFormerPrecisionRecordEventKind::PARAMETER_PUBLICATION);
  CHECK(events[9].diagnostic_epoch == 1);

  const std::vector<PsiFormerPrecisionRecordEvent> complete_graph = events;
  CHECK_THROWS_AS(recorder.recordPublicationStart(), std::invalid_argument);
  CHECK(recorder.events() == complete_graph);
}

TEST_CASE("PsiFormer conversion hazards preserve the old complete slot pair",
          "[psiformer][precision_orchestration]")
{
  PsiFormerPrecisionRecordingOrchestrator recorder(
      makePlan(), makeDescriptors(), makeDiagnosticBounds(), /*active_version=*/4,
      /*active_fingerprint=*/43, /*active_slot=*/1);
  recorder.prepare();

  recorder.recordPublicationStart();
  CHECK(recorder.publicationState().publicationPending());
  recorder.cancelPublicationStart();
  CHECK_FALSE(recorder.publicationState().publicationPending());
  CHECK(recorder.publicationState().activeVersion() == 4);
  CHECK(recorder.events().back().kind ==
        PsiFormerPrecisionRecordEventKind::PARAMETER_CANCELLATION);

  PsiFormerParameterConversionDiagnostics invalid;
  invalid.overflow_count = 1;
  recorder.recordPublicationStart();
  CHECK(recorder.observePublicationCompletion(invalid) ==
        PsiFormerPrecisionPublicationResult::REJECTED_DIAGNOSTICS);
  CHECK(recorder.publicationState().activeVersion() == 4);
  CHECK(recorder.publicationState().activeFingerprint() == 43);
  CHECK(recorder.publicationState().activeSlot() == 1);
  CHECK_FALSE(recorder.publicationState().publicationPending());
  CHECK(countEvents(recorder.events(),
                    PsiFormerPrecisionRecordEventKind::PARAMETER_COMPLETION) == 0);
  CHECK(countEvents(recorder.events(),
                    PsiFormerPrecisionRecordEventKind::PARAMETER_PUBLICATION) == 0);
  REQUIRE_FALSE(recorder.events().empty());
  CHECK(recorder.events().back().kind ==
        PsiFormerPrecisionRecordEventKind::PARAMETER_DIAGNOSTIC_DECISION);
  CHECK(recorder.events().back().hard_hazard);

  const std::vector<PsiFormerPrecisionRecordEvent> rejected_graph =
      recorder.events();
  CHECK_THROWS_AS(recorder.recordValueEvaluationStart(
                      5, recorder.metadata().plan_fingerprint),
                  std::logic_error);
  CHECK(recorder.events() == rejected_graph);

  recorder.recordPublicationStart();
  CHECK(recorder.observePublicationCompletion({}) ==
        PsiFormerPrecisionPublicationResult::PUBLISHED);
  CHECK(recorder.publicationState().activeSlot() == 0);
  CHECK(recorder.publicationState().activeVersion() == 5);
}

TEST_CASE("PsiFormer mixed value graph records boundaries and one bounded retry",
          "[psiformer][precision_orchestration]")
{
  PsiFormerPrecisionRecordingOrchestrator recorder(
      makePlan(), makeDescriptors(), makeDiagnosticBounds(), /*active_version=*/4,
      /*active_fingerprint=*/47, /*active_slot=*/0);
  recorder.prepare();
  recorder.recordPublicationStart();
  REQUIRE(recorder.observePublicationCompletion({}) ==
          PsiFormerPrecisionPublicationResult::PUBLISHED);
  const std::uint64_t fingerprint = recorder.metadata().plan_fingerprint;
  const std::size_t capacity = recorder.preparedEventCapacity();

  allocation_probe::Scope mixed_probe;
  recorder.recordValueEvaluationStart(5, fingerprint);
  const PsiFormerPrecisionEvaluationResult mixed_result =
      recorder.observeMixedValueCompletion({});
  const std::size_t mixed_allocations = mixed_probe.finish();
  CHECK(mixed_result == PsiFormerPrecisionEvaluationResult::MIXED_SUCCESS);
  CHECK(mixed_allocations == 0);
  const auto& ordinary = recorder.events();
  REQUIRE(ordinary.size() == 9);
  CHECK(ordinary[0].kind ==
        PsiFormerPrecisionRecordEventKind::EXECUTION_DIAGNOSTICS_CLEAR);
  CHECK(ordinary[0].diagnostic_offset ==
        offsetof(PsiFormerDeviceNumericalDiagnostics, execution));
  CHECK(ordinary[0].diagnostic_bytes == sizeof(PsiFormerNumericalDiagnostics));
  CHECK(ordinary[1].kind ==
        PsiFormerPrecisionRecordEventKind::VALUE_FP64_TO_FP32);
  CHECK(ordinary[1].count == 30);
  CHECK(ordinary[2].kind ==
        PsiFormerPrecisionRecordEventKind::VALUE_FP32_NETWORK);
  CHECK(ordinary[2].source_region ==
        PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_1);
  CHECK(ordinary[3].kind ==
        PsiFormerPrecisionRecordEventKind::VALUE_FP32_TO_FP64);
  CHECK(ordinary[4].kind ==
        PsiFormerPrecisionRecordEventKind::VALUE_FP64_SENSITIVE);
  CHECK(ordinary[4].count == 2);
  CHECK(ordinary[5].kind == PsiFormerPrecisionRecordEventKind::
        EXECUTION_DEVICE_COMPLETION_BARRIER);
  CHECK(ordinary[6].kind == PsiFormerPrecisionRecordEventKind::
        EXECUTION_DIAGNOSTIC_READBACK_BARRIER);
  CHECK(ordinary[7].kind ==
        PsiFormerPrecisionRecordEventKind::EXECUTION_DIAGNOSTIC_DECISION);
  CHECK_FALSE(ordinary[7].hard_hazard);
  CHECK(ordinary[8].kind ==
        PsiFormerPrecisionRecordEventKind::EVALUATION_COMPLETE);
  CHECK(ordinary[8].diagnostic_epoch == ordinary[0].diagnostic_epoch);
  CHECK(recorder.executionDiagnosticIncrementBound() == 108);
  CHECK(ordinary[0].diagnostic_increment_bound == 108);
  CHECK(ordinary[6].diagnostic_increment_bound == 108);

  PsiFormerNumericalDiagnostics mixed_invalid;
  mixed_invalid.nonfinite_count = 48;
  mixed_invalid.invalid_softmax_count = 8;
  mixed_invalid.small_determinant_pivot_count = 4;
  mixed_invalid.severe_cancellation_count = 4;
  mixed_invalid.extreme_ecp_ratio_count = 12;
  allocation_probe::Scope retry_probe;
  recorder.recordValueEvaluationStart(5, fingerprint);
  const PsiFormerPrecisionEvaluationResult pending_result =
      recorder.observeMixedValueCompletion(mixed_invalid);
  const PsiFormerPrecisionEvaluationResult retry_result =
      recorder.observeFullPrecisionRetryCompletion({});
  const std::size_t retry_allocations = retry_probe.finish();
  CHECK(pending_result ==
        PsiFormerPrecisionEvaluationResult::FP64_RETRY_PENDING);
  CHECK(retry_result == PsiFormerPrecisionEvaluationResult::FP64_RETRY_SUCCESS);
  CHECK(retry_allocations == 0);
  const auto& recovered = recorder.events();
  REQUIRE(recovered.size() == 14);
  CHECK(countEvents(recovered,
                    PsiFormerPrecisionRecordEventKind::WHOLE_BATCH_FP64_RETRY) == 1);
  CHECK(countEvents(recovered,
                    PsiFormerPrecisionRecordEventKind::EXECUTION_DIAGNOSTICS_CLEAR) == 2);
  CHECK(recovered[7].hard_hazard);
  CHECK(recovered[8].diagnostic_epoch == recovered[7].diagnostic_epoch + 1);
  CHECK(recovered[9].kind ==
        PsiFormerPrecisionRecordEventKind::WHOLE_BATCH_FP64_RETRY);
  CHECK(recovered[9].count == 2);
  CHECK(recovered[10].kind == PsiFormerPrecisionRecordEventKind::
        EXECUTION_DEVICE_COMPLETION_BARRIER);
  CHECK(recovered[11].kind == PsiFormerPrecisionRecordEventKind::
        EXECUTION_DIAGNOSTIC_READBACK_BARRIER);
  CHECK(recovered[12].kind ==
        PsiFormerPrecisionRecordEventKind::EXECUTION_DIAGNOSTIC_DECISION);
  CHECK_FALSE(recovered[12].hard_hazard);
  CHECK(recovered[13].kind ==
        PsiFormerPrecisionRecordEventKind::EVALUATION_COMPLETE);
  for (const PsiFormerPrecisionRecordEvent& event : recovered)
    if (event.diagnostic_bytes != 0)
    {
      CHECK(event.diagnostic_offset <=
            sizeof(PsiFormerDeviceNumericalDiagnostics));
      CHECK(event.diagnostic_bytes <=
            sizeof(PsiFormerDeviceNumericalDiagnostics) -
                event.diagnostic_offset);
    }

  PsiFormerNumericalDiagnostics retry_invalid;
  retry_invalid.invalid_softmax_count = 1;
  recorder.recordValueEvaluationStart(5, fingerprint);
  CHECK(recorder.observeMixedValueCompletion(mixed_invalid) ==
        PsiFormerPrecisionEvaluationResult::FP64_RETRY_PENDING);
  CHECK(recorder.observeFullPrecisionRetryCompletion(retry_invalid) ==
        PsiFormerPrecisionEvaluationResult::FINAL_FAILURE);
  CHECK(countEvents(recorder.events(),
                    PsiFormerPrecisionRecordEventKind::WHOLE_BATCH_FP64_RETRY) == 1);
  CHECK(recorder.events().back().kind ==
        PsiFormerPrecisionRecordEventKind::FINAL_FAILURE);
  CHECK(recorder.events().back().hard_hazard);
  CHECK(recorder.allocationCount() == 1);
  CHECK(recorder.preparedEventCapacity() == capacity);

  const std::vector<PsiFormerPrecisionRecordEvent> failed_graph =
      recorder.events();
  CHECK_THROWS_AS(recorder.recordValueEvaluationStart(4, fingerprint),
                  std::logic_error);
  CHECK(recorder.events() == failed_graph);
  CHECK_THROWS_AS(recorder.recordValueEvaluationStart(5, fingerprint + 1),
                  std::logic_error);
  CHECK(recorder.events() == failed_graph);

  recorder.recordValueEvaluationStart(5, fingerprint);
  const std::vector<PsiFormerPrecisionRecordEvent> pending_graph =
      recorder.events();
  PsiFormerNumericalDiagnostics excessive;
  excessive.invalid_softmax_count =
      recorder.executionDiagnosticBounds().invalid_softmax_count + 1;
  CHECK_THROWS_AS(recorder.observeMixedValueCompletion(excessive),
                  std::invalid_argument);
  CHECK(recorder.events() == pending_graph);
  CHECK(recorder.observeMixedValueCompletion({}) ==
        PsiFormerPrecisionEvaluationResult::MIXED_SUCCESS);
}

TEST_CASE("PsiFormer unavailable retry terminates without a second mixed attempt",
          "[psiformer][precision_orchestration]")
{
  PsiFormerPrecisionRecordingOrchestrator recorder(
      makePlan(/*version=*/6, /*retry_available=*/false), makeDescriptors(),
      makeDiagnosticBounds(),
      /*active_version=*/5, /*active_fingerprint=*/53, /*active_slot=*/1);
  recorder.prepare();
  recorder.recordPublicationStart();
  REQUIRE(recorder.observePublicationCompletion({}) ==
          PsiFormerPrecisionPublicationResult::PUBLISHED);

  PsiFormerNumericalDiagnostics mixed_invalid;
  mixed_invalid.severe_cancellation_count = 1;
  recorder.recordValueEvaluationStart(
      6, recorder.metadata().plan_fingerprint);
  CHECK(recorder.observeMixedValueCompletion(mixed_invalid) ==
        PsiFormerPrecisionEvaluationResult::FINAL_FAILURE);
  CHECK(countEvents(recorder.events(),
                    PsiFormerPrecisionRecordEventKind::WHOLE_BATCH_FP64_RETRY) == 0);
  CHECK(recorder.events().back().kind ==
        PsiFormerPrecisionRecordEventKind::FINAL_FAILURE);
  const std::vector<PsiFormerPrecisionRecordEvent> failed_graph =
      recorder.events();
  CHECK_THROWS_AS(recorder.observeFullPrecisionRetryCompletion({}),
                  std::logic_error);
  CHECK(recorder.events() == failed_graph);
}

TEST_CASE("PsiFormer participant metadata agrees across simulated crowds and ranks",
          "[psiformer][precision_orchestration]")
{
  const PsiFormerMixedValueDescriptors descriptors = makeDescriptors();
  const PsiFormerExecutionDiagnosticBounds bounds = makeDiagnosticBounds();
  PsiFormerExecutionDiagnosticBounds mutated_bounds = bounds;
  mutated_bounds.fingerprint ^= UINT64_C(1);
  CHECK_THROWS_WITH(makePsiFormerPrecisionParticipantMetadata(
                        makePlan(), descriptors, mutated_bounds),
                    Catch::Matchers::ContainsSubstring("fingerprint"));
  const PsiFormerPrecisionParticipantMetadata metadata =
      makePsiFormerPrecisionParticipantMetadata(makePlan(), descriptors, bounds);
  std::vector<PsiFormerPrecisionParticipantMetadata> participants(6, metadata);
  CHECK_NOTHROW(validatePsiFormerPrecisionParticipantMetadata(participants));
  CHECK_THROWS_AS(validatePsiFormerPrecisionParticipantMetadata({}),
                  std::invalid_argument);

  participants[4].value_descriptor_fingerprint ^= UINT64_C(1);
  CHECK_THROWS_WITH(validatePsiFormerPrecisionParticipantMetadata(participants),
                    Catch::Matchers::ContainsSubstring("differs"));
  participants.assign(6, metadata);
  participants[2].execution_diagnostic_bounds_fingerprint ^= UINT64_C(1);
  CHECK_THROWS_WITH(validatePsiFormerPrecisionParticipantMetadata(participants),
                    Catch::Matchers::ContainsSubstring("differs"));
  participants.assign(6, metadata);
  participants[0].native_math_mode = PsiFormerNativeBlasMathMode::CUDA_TF32;
  CHECK_THROWS_WITH(validatePsiFormerPrecisionParticipantMetadata(participants),
                    Catch::Matchers::ContainsSubstring("inconsistent"));

  participants.assign(6, metadata);
  participants[0].math_mode = PsiFormerBackendMathMode::FP64_STRICT;
  participants[0].native_math_mode = PsiFormerNativeBlasMathMode::CUDA_PEDANTIC;
  CHECK_THROWS_WITH(validatePsiFormerPrecisionParticipantMetadata(participants),
                    Catch::Matchers::ContainsSubstring("policy/math mode"));

  participants.assign(6, metadata);
  participants[5] = makePsiFormerPrecisionParticipantMetadata(
      makePlan(/*version=*/5, /*retry_available=*/true,
               PsiFormerAcceleratorBackend::HIP),
      descriptors, bounds);
  CHECK_THROWS_WITH(validatePsiFormerPrecisionParticipantMetadata(participants),
                    Catch::Matchers::ContainsSubstring("differs"));
}

TEST_CASE("PsiFormer recorder rejects mutated prepared identities",
          "[psiformer][precision_orchestration]")
{
  PsiFormerPrecisionExecutionPlan overlapping = makePlan();
  PsiFormerDeviceArenaSlice* compute_1 = nullptr;
  for (PsiFormerDeviceArenaSlice& slice : overlapping.arena.slices)
    if (slice.region == PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_1)
      compute_1 = &slice;
  REQUIRE(compute_1);
  compute_1->offset =
      overlapping.arena.find(PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_0)
          ->offset;
  PsiFormerPrecisionRecordingOrchestrator overlap_recorder(
      std::move(overlapping), makeDescriptors(), makeDiagnosticBounds(), 4, 59, 0);
  CHECK_THROWS_WITH(overlap_recorder.prepare(),
                    Catch::Matchers::ContainsSubstring("arena identity"));

  PsiFormerPrecisionExecutionPlan diagnostic_overlap = makePlan();
  PsiFormerDeviceArenaSlice* diagnostics = nullptr;
  for (PsiFormerDeviceArenaSlice& slice : diagnostic_overlap.arena.slices)
    if (slice.region == PsiFormerDeviceArenaRegion::NUMERICAL_DIAGNOSTICS)
      diagnostics = &slice;
  REQUIRE(diagnostics);
  diagnostics->offset = diagnostic_overlap.arena
                            .find(PsiFormerDeviceArenaRegion::
                                      PRECISION_CONVERSION_WORKSPACE)
                            ->offset;
  PsiFormerPrecisionRecordingOrchestrator diagnostic_overlap_recorder(
      std::move(diagnostic_overlap), makeDescriptors(), makeDiagnosticBounds(),
      4, 60, 0);
  CHECK_THROWS_WITH(diagnostic_overlap_recorder.prepare(),
                    Catch::Matchers::ContainsSubstring("arena identity"));

  PsiFormerMixedValueDescriptors mutated = makeDescriptors();
  mutated.fingerprint ^= UINT64_C(1);
  PsiFormerPrecisionRecordingOrchestrator descriptor_recorder(
      makePlan(), mutated, makeDiagnosticBounds(), 4, 61, 0);
  CHECK_THROWS_WITH(descriptor_recorder.prepare(),
                    Catch::Matchers::ContainsSubstring("fingerprint"));
}

int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
