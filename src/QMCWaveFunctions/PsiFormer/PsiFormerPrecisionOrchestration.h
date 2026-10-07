//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerPrecisionOrchestration.h
 * @brief Pointer-free recording contracts for mixed-precision publication/evaluation.
 *
 * This component records an attachable event graph but does not execute a device
 * runtime.  Device pointers, streams, events, collectives, and builder dispatch
 * remain owned by the later production executor.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_PRECISION_ORCHESTRATION_H
#define QMCPLUSPLUS_PSIFORMER_PRECISION_ORCHESTRATION_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerAttention.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPrecisionExecution.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace qmcplusplus::psiformer
{

/** Tie one flat dense target to the exact batched attention/value interpretation. */
struct PsiFormerMixedValueDescriptors
{
  DenseForwardLayout dense;
  BatchedAttentionForwardLayout attention;
  BatchedValueLayout value;
  std::uint64_t fingerprint = 0;
};

static_assert(std::is_standard_layout_v<PsiFormerMixedValueDescriptors>);
static_assert(std::is_trivially_copyable_v<PsiFormerMixedValueDescriptors>);

/** Construct and validate one deterministic mixed-value descriptor identity. */
PsiFormerMixedValueDescriptors makePsiFormerMixedValueDescriptors(
    const DenseForwardLayout& dense,
    const BatchedAttentionForwardLayout& attention,
    const BatchedValueLayout& value);

/** Metadata which must agree across every crowd/rank participant in one operation. */
struct PsiFormerPrecisionParticipantMetadata
{
  PsiFormerPrecisionPolicy policy = PsiFormerPrecisionPolicy::FP64_REFERENCE;
  PsiFormerBackendMathMode math_mode = PsiFormerBackendMathMode::FP64_STRICT;
  PsiFormerAcceleratorBackend backend = PsiFormerAcceleratorBackend::CPU;
  PsiFormerNativeBlasMathMode native_math_mode =
      PsiFormerNativeBlasMathMode::HOST_DEFAULT;
  std::size_t source_version = 0;
  std::uint64_t plan_fingerprint = 0;
  std::uint64_t device_layout_fingerprint = 0;
  std::uint64_t arena_fingerprint = 0;
  std::uint64_t conversion_fingerprint = 0;
  std::uint64_t value_descriptor_fingerprint = 0;
  bool reduced_multiply = false;
  bool full_precision_retry_available = false;

  friend bool operator==(const PsiFormerPrecisionParticipantMetadata& lhs,
                         const PsiFormerPrecisionParticipantMetadata& rhs) noexcept
  {
    return lhs.policy == rhs.policy && lhs.math_mode == rhs.math_mode &&
        lhs.backend == rhs.backend &&
        lhs.native_math_mode == rhs.native_math_mode &&
        lhs.source_version == rhs.source_version &&
        lhs.plan_fingerprint == rhs.plan_fingerprint &&
        lhs.device_layout_fingerprint == rhs.device_layout_fingerprint &&
        lhs.arena_fingerprint == rhs.arena_fingerprint &&
        lhs.conversion_fingerprint == rhs.conversion_fingerprint &&
        lhs.value_descriptor_fingerprint == rhs.value_descriptor_fingerprint &&
        lhs.reduced_multiply == rhs.reduced_multiply &&
        lhs.full_precision_retry_available == rhs.full_precision_retry_available;
  }
};

static_assert(std::is_standard_layout_v<PsiFormerPrecisionParticipantMetadata>);
static_assert(std::is_trivially_copyable_v<PsiFormerPrecisionParticipantMetadata>);

/** Derive the immutable metadata advertised by one prepared participant. */
PsiFormerPrecisionParticipantMetadata makePsiFormerPrecisionParticipantMetadata(
    const PsiFormerPrecisionExecutionPlan& plan,
    const PsiFormerMixedValueDescriptors& descriptors);

/** Reject an empty or nonidentical simulated crowd/rank metadata set. */
void validatePsiFormerPrecisionParticipantMetadata(
    const std::vector<PsiFormerPrecisionParticipantMetadata>& participants);

/// Identify one logical operation in the attachable recording graph.
enum class PsiFormerPrecisionRecordEventKind : std::uint8_t
{
  INACTIVE_MASTER_COPY,
  COMBINED_DIAGNOSTICS_CLEAR,
  PARAMETER_CONVERSION_TILE,
  PARAMETER_DEVICE_COMPLETION_BARRIER,
  PARAMETER_DIAGNOSTIC_READBACK_BARRIER,
  PARAMETER_DIAGNOSTIC_DECISION,
  PARAMETER_COMPLETION,
  PARAMETER_PUBLICATION,
  PARAMETER_CANCELLATION,
  EXECUTION_DIAGNOSTICS_CLEAR,
  // Contract-only until A7 supplies the initial FP64 feature/value cast leaf.
  VALUE_FP64_TO_FP32,
  VALUE_FP32_NETWORK,
  VALUE_FP32_TO_FP64,
  VALUE_FP64_SENSITIVE,
  EXECUTION_DEVICE_COMPLETION_BARRIER,
  EXECUTION_DIAGNOSTIC_READBACK_BARRIER,
  EXECUTION_DIAGNOSTIC_DECISION,
  WHOLE_BATCH_FP64_RETRY,
  EVALUATION_COMPLETE,
  FINAL_FAILURE
};

/** Pointer-free description of one ordered publication/evaluation event. */
struct PsiFormerPrecisionRecordEvent
{
  PsiFormerPrecisionRecordEventKind kind =
      PsiFormerPrecisionRecordEventKind::FINAL_FAILURE;
  std::size_t version = 0;
  std::uint8_t slot = 0;
  std::size_t diagnostic_epoch = 0;
  PsiFormerDeviceArenaRegion source_region = PsiFormerDeviceArenaRegion::COUNT;
  PsiFormerDeviceArenaRegion destination_region = PsiFormerDeviceArenaRegion::COUNT;
  std::size_t source_begin = 0;
  std::size_t destination_begin = 0;
  std::size_t count = 0;
  std::size_t diagnostic_offset = 0;
  std::size_t diagnostic_bytes = 0;
  std::size_t diagnostic_increment_bound = 0;
  bool hard_hazard = false;

  friend bool operator==(const PsiFormerPrecisionRecordEvent& lhs,
                         const PsiFormerPrecisionRecordEvent& rhs) noexcept
  {
    return lhs.kind == rhs.kind && lhs.version == rhs.version &&
        lhs.slot == rhs.slot && lhs.diagnostic_epoch == rhs.diagnostic_epoch &&
        lhs.source_region == rhs.source_region &&
        lhs.destination_region == rhs.destination_region &&
        lhs.source_begin == rhs.source_begin &&
        lhs.destination_begin == rhs.destination_begin && lhs.count == rhs.count &&
        lhs.diagnostic_offset == rhs.diagnostic_offset &&
        lhs.diagnostic_bytes == rhs.diagnostic_bytes &&
        lhs.diagnostic_increment_bound == rhs.diagnostic_increment_bound &&
        lhs.hard_hazard == rhs.hard_hazard;
  }
};

static_assert(std::is_standard_layout_v<PsiFormerPrecisionRecordEvent>);
static_assert(std::is_trivially_copyable_v<PsiFormerPrecisionRecordEvent>);

/// Report the failure-atomic outcome of one parameter publication recording.
enum class PsiFormerPrecisionPublicationResult : std::uint8_t
{
  PUBLISHED,
  REJECTED_DIAGNOSTICS
};

/// Report whether mixed evaluation, its sole FP64 retry, or neither succeeded.
enum class PsiFormerPrecisionEvaluationResult : std::uint8_t
{
  MIXED_SUCCESS,
  FP64_RETRY_PENDING,
  FP64_RETRY_SUCCESS,
  FINAL_FAILURE
};

/** Record exact mixed-precision graphs after one allocation-bearing prepare step.
 *
 * The class owns no runtime resources.  Its event vector is reserved to the maximum
 * graph size in ``prepare()`` and may not change capacity while recording.  Events
 * form a strict single-queue order: a translator submits them synchronously in vector
 * order and may call an ``observe*Completion`` method only after the preceding device
 * completion and diagnostic-readback barriers have completed.
 *
 * Value/retry records remain contract-only until Task 26.A8 provides production value
 * and retry workspaces/dispatch.  The initial FP64-to-FP32 value cast also awaits its
 * Task 27.A7 device leaf.  Host-recorder allocation tests make no production-runtime
 * allocation claim.
 */
class PsiFormerPrecisionRecordingOrchestrator
{
public:
  explicit PsiFormerPrecisionRecordingOrchestrator(
      PsiFormerPrecisionExecutionPlan plan,
      PsiFormerMixedValueDescriptors descriptors,
      std::size_t active_version = 0,
      std::uint64_t active_fingerprint = 1,
      std::uint8_t active_slot = 0);

  /// Validate immutable inputs and reserve the complete recording buffer once.
  void prepare();

  /** Record copy, clear, conversion, and barriers without publishing future work. */
  void recordPublicationStart();

  /** Cancel a recorded but not yet observed parameter publication. */
  void cancelPublicationStart();

  /** Observe completed/read-back conversion work, then decide and publish atomically. */
  PsiFormerPrecisionPublicationResult observePublicationCompletion(
      const PsiFormerParameterConversionDiagnostics& diagnostics);

  /** Record one mixed value attempt and its completion/readback barriers. */
  void recordValueEvaluationStart(
      std::size_t version,
      std::uint64_t execution_fingerprint);

  /** Observe mixed diagnostics and complete, fail, or record exactly one FP64 retry. */
  PsiFormerPrecisionEvaluationResult observeMixedValueCompletion(
      const PsiFormerNumericalDiagnostics& mixed_diagnostics);

  /** Observe the sole FP64 retry diagnostics and complete or fail finally. */
  PsiFormerPrecisionEvaluationResult observeFullPrecisionRetryCompletion(
      const PsiFormerNumericalDiagnostics& retry_diagnostics);

  bool prepared() const noexcept { return prepared_; }
  std::size_t allocationCount() const noexcept { return allocation_count_; }
  std::size_t preparedEventCapacity() const noexcept { return prepared_event_capacity_; }
  std::size_t diagnosticEpoch() const noexcept { return diagnostic_epoch_; }
  std::size_t executionDiagnosticIncrementBound() const noexcept
  {
    return execution_diagnostic_increment_bound_;
  }
  const std::vector<PsiFormerPrecisionRecordEvent>& events() const noexcept
  {
    return events_;
  }
  const PsiFormerMixedPublicationState& publicationState() const noexcept
  {
    return publication_;
  }
  const PsiFormerPrecisionParticipantMetadata& metadata() const noexcept
  {
    return metadata_;
  }

private:
  enum class RecordingPhase : std::uint8_t
  {
    IDLE,
    PUBLICATION_PENDING,
    MIXED_VALUE_PENDING,
    FP64_RETRY_PENDING
  };

  void requirePrepared() const;
  void beginRecording();
  void append(PsiFormerPrecisionRecordEvent event);
  std::size_t beginDiagnosticEpoch(PsiFormerPrecisionRecordEventKind clear_kind,
                                   std::size_t diagnostic_offset,
                                   std::size_t diagnostic_bytes,
                                   std::size_t version,
                                   std::uint8_t slot);
  void validateConversionDiagnostics(
      const PsiFormerParameterConversionDiagnostics& diagnostics) const;
  void validateExecutionDiagnostics(
      const PsiFormerNumericalDiagnostics& diagnostics) const;

  PsiFormerPrecisionExecutionPlan plan_;
  PsiFormerMixedValueDescriptors descriptors_;
  PsiFormerPrecisionParticipantMetadata metadata_;
  PsiFormerMixedPublicationState publication_;
  std::vector<PsiFormerPrecisionRecordEvent> events_;
  std::size_t prepared_event_capacity_ = 0;
  std::size_t allocation_count_ = 0;
  std::size_t diagnostic_epoch_ = 0;
  std::size_t execution_diagnostic_increment_bound_ = 0;
  std::size_t pending_diagnostic_epoch_ = 0;
  RecordingPhase phase_ = RecordingPhase::IDLE;
  bool prepared_ = false;
};

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_PRECISION_ORCHESTRATION_H
