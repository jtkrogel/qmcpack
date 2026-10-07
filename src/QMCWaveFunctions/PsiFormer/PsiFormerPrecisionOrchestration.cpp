//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerPrecisionOrchestration.cpp
 * @brief Validation and recording for mixed-precision publication/evaluation graphs.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerPrecisionOrchestration.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace qmcplusplus::psiformer
{
namespace
{

constexpr std::uint64_t FNV_OFFSET = UINT64_C(14695981039346656037);
constexpr std::uint64_t FNV_PRIME  = UINT64_C(1099511628211);

/// Mix one fixed-width integer into a stable little-endian FNV-1a hash.
void mixInteger(std::uint64_t& hash, std::uint64_t value) noexcept
{
  for (unsigned int shift = 0; shift < 64; shift += 8)
  {
    hash ^= static_cast<std::uint8_t>((value >> shift) & UINT64_C(0xff));
    hash *= FNV_PRIME;
  }
}

/// Multiply extents without permitting a wrapped descriptor or event count.
std::size_t checkedProduct(std::size_t left,
                           std::size_t right,
                           const char* description)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(description);
  return left * right;
}

/// Add extents without permitting a wrapped recording capacity.
std::size_t checkedSum(std::size_t left,
                       std::size_t right,
                       const char* description)
{
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::length_error(description);
  return left + right;
}

/// Return whether two nonempty arena slices overlap.
bool slicesOverlap(const PsiFormerDeviceArenaSlice& lhs,
                   const PsiFormerDeviceArenaSlice& rhs) noexcept
{
  return lhs.bytes != 0 && rhs.bytes != 0 && lhs.offset < rhs.end() &&
      rhs.offset < lhs.end();
}

/// Require one exact arena region and byte count.
const PsiFormerDeviceArenaSlice& requireArenaSlice(
    const PsiFormerPrecisionExecutionPlan& plan,
    PsiFormerDeviceArenaRegion region,
    std::size_t expected_bytes)
{
  const PsiFormerDeviceArenaSlice* slice = plan.arena.find(region);
  if (!slice)
    throw std::invalid_argument(
        std::string("PsiFormer precision orchestration lacks arena region ") +
        psiFormerDeviceArenaRegionName(region));
  if (slice->bytes != expected_bytes)
    throw std::invalid_argument(
        std::string("PsiFormer precision orchestration arena size mismatch for ") +
        psiFormerDeviceArenaRegionName(region));
  return *slice;
}

/// Map a logical FP64 master slot to its stable arena-region identity.
PsiFormerDeviceArenaRegion masterRegion(std::uint8_t slot)
{
  if (slot == 0)
    return PsiFormerDeviceArenaRegion::MODEL_PARAMETERS;
  if (slot == 1)
    return PsiFormerDeviceArenaRegion::MODEL_PARAMETERS_STAGING;
  throw std::invalid_argument("PsiFormer precision master slot must be zero or one");
}

/// Map a logical FP32 compute slot to its stable arena-region identity.
PsiFormerDeviceArenaRegion computeRegion(std::uint8_t slot)
{
  if (slot == 0)
    return PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_0;
  if (slot == 1)
    return PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_1;
  throw std::invalid_argument("PsiFormer precision compute slot must be zero or one");
}

/** Validate the mixed arena, schedule, and math metadata consumed by the recorder. */
void validateRecordingPlan(const PsiFormerPrecisionExecutionPlan& plan)
{
  if (plan.policy == PsiFormerPrecisionPolicy::FP64_REFERENCE)
    throw std::invalid_argument(
        "PsiFormer mixed recording orchestrator requires a mixed precision policy");
  if (plan.fingerprint == 0 || plan.device_layout_fingerprint == 0 ||
      plan.arena.fingerprint == 0 || plan.conversion.fingerprint == 0)
    throw std::invalid_argument(
        "PsiFormer mixed recording orchestrator requires complete fingerprints");
  if (plan.compute_copy_version != plan.canonical_source_version)
    throw std::invalid_argument(
        "PsiFormer mixed recording plan has mismatched master/compute versions");
  if ((plan.policy == PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE &&
       plan.math_mode != PsiFormerBackendMathMode::FP32_STRICT) ||
      (plan.policy == PsiFormerPrecisionPolicy::TF32_DENSE_FP64_SENSITIVE &&
       plan.math_mode != PsiFormerBackendMathMode::CUDA_TF32))
    throw std::invalid_argument(
        "PsiFormer mixed recording plan has inconsistent policy/math mode");
  if (!(plan.blas_math ==
        makePsiFormerBlasMathModePlan(plan.math_mode, plan.backend)))
    throw std::invalid_argument(
        "PsiFormer mixed recording plan has inconsistent backend math metadata");

  const std::size_t master_bytes =
      checkedProduct(plan.parameter_count, sizeof(double),
                     "PsiFormer FP64 master recording extent overflow");
  const std::size_t compute_bytes =
      checkedProduct(plan.parameter_count, sizeof(float),
                     "PsiFormer FP32 compute recording extent overflow");
  const std::size_t conversion_workspace_bytes = checkedProduct(
      plan.conversion.tile_capacity, sizeof(float),
      "PsiFormer conversion workspace recording extent overflow");
  const std::vector<PsiFormerDeviceArenaRequest> arena_requests{
      {PsiFormerDeviceArenaRegion::MODEL_PARAMETERS, master_bytes, 256},
      {PsiFormerDeviceArenaRegion::MODEL_PARAMETERS_STAGING, master_bytes, 256},
      {PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_0, compute_bytes, 256},
      {PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_1, compute_bytes, 256},
      {PsiFormerDeviceArenaRegion::PRECISION_CONVERSION_WORKSPACE,
       conversion_workspace_bytes, 256},
      {PsiFormerDeviceArenaRegion::NUMERICAL_DIAGNOSTICS,
       sizeof(PsiFormerDeviceNumericalDiagnostics),
       alignof(PsiFormerDeviceNumericalDiagnostics)}};
  const PsiFormerDeviceArenaLayout canonical_arena =
      makePsiFormerDeviceArenaLayout(arena_requests);
  if (canonical_arena.slices != plan.arena.slices ||
      canonical_arena.total_bytes != plan.arena.total_bytes ||
      canonical_arena.fingerprint != plan.arena.fingerprint)
    throw std::invalid_argument(
        "PsiFormer mixed recording arena identity is not canonical");

  if (plan.master_slot_bytes != master_bytes ||
      plan.compute_slot_bytes != compute_bytes ||
      plan.conversion_workspace_bytes != conversion_workspace_bytes ||
      plan.diagnostic_bytes != sizeof(PsiFormerDeviceNumericalDiagnostics))
    throw std::invalid_argument(
        "PsiFormer mixed recording plan has inconsistent storage bytes");

  const PsiFormerDeviceArenaSlice& master_0 = requireArenaSlice(
      plan, PsiFormerDeviceArenaRegion::MODEL_PARAMETERS, master_bytes);
  const PsiFormerDeviceArenaSlice& master_1 = requireArenaSlice(
      plan, PsiFormerDeviceArenaRegion::MODEL_PARAMETERS_STAGING, master_bytes);
  const PsiFormerDeviceArenaSlice& compute_0 = requireArenaSlice(
      plan, PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_0, compute_bytes);
  const PsiFormerDeviceArenaSlice& compute_1 = requireArenaSlice(
      plan, PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_1, compute_bytes);
  (void)requireArenaSlice(
      plan, PsiFormerDeviceArenaRegion::PRECISION_CONVERSION_WORKSPACE,
      plan.conversion_workspace_bytes);
  (void)requireArenaSlice(
      plan, PsiFormerDeviceArenaRegion::NUMERICAL_DIAGNOSTICS,
      sizeof(PsiFormerDeviceNumericalDiagnostics));

  if (slicesOverlap(master_0, master_1) || slicesOverlap(compute_0, compute_1) ||
      slicesOverlap(master_0, compute_0) || slicesOverlap(master_0, compute_1) ||
      slicesOverlap(master_1, compute_0) || slicesOverlap(master_1, compute_1))
    throw std::invalid_argument(
        "PsiFormer active and staged precision slots must not overlap");

  if (plan.conversion.parameter_count != plan.parameter_count)
    throw std::invalid_argument(
        "PsiFormer conversion schedule does not cover the parameter image");
  if (plan.conversion.block_size == 0 ||
      (plan.parameter_count == 0 && plan.conversion.tile_capacity != 0) ||
      (plan.parameter_count != 0 && plan.conversion.tile_capacity == 0))
    throw std::invalid_argument(
        "PsiFormer conversion schedule has inconsistent tile geometry");
  const PsiFormerTiledLaunchSchedule canonical_conversion =
      makePsiFormerTiledLaunchSchedule(
          plan.parameter_count,
          plan.parameter_count == 0 ? 1 : plan.conversion.tile_capacity,
          plan.conversion.block_size);
  if (canonical_conversion.fingerprint != plan.conversion.fingerprint ||
      canonical_conversion.tiles.size() != plan.conversion.tiles.size())
    throw std::invalid_argument(
        "PsiFormer conversion schedule identity is not canonical");
  std::size_t expected_begin = 0;
  for (std::size_t index = 0; index < plan.conversion.tiles.size(); ++index)
  {
    const PsiFormerParameterConversionTile& tile = plan.conversion.tiles[index];
    const PsiFormerLaunchTile& canonical_tile = canonical_conversion.tiles[index];
    if (tile.source_begin != expected_begin ||
        tile.destination_begin != expected_begin || tile.count == 0 ||
        tile.count > plan.conversion.tile_capacity ||
        tile.launch.item_count != tile.count ||
        tile.launch.block_size != plan.conversion.block_size ||
        canonical_tile.begin != tile.source_begin ||
        canonical_tile.count != tile.count ||
        !(canonical_tile.launch == tile.launch))
      throw std::invalid_argument(
          "PsiFormer conversion schedule is not deterministic and contiguous");
    expected_begin = checkedSum(expected_begin, tile.count,
                                "PsiFormer conversion schedule extent overflow");
  }
  if (expected_begin != plan.parameter_count)
    throw std::invalid_argument(
        "PsiFormer conversion schedule is incomplete");
}

/// Validate that one diagnostic subspan is bounded by the combined record.
void validateDiagnosticSpan(std::size_t offset,
                            std::size_t bytes,
                            std::size_t combined_bytes)
{
  if (offset > combined_bytes || bytes > combined_bytes - offset)
    throw std::invalid_argument(
        "PsiFormer recording diagnostic span exceeds the combined record");
}

} // namespace

PsiFormerMixedValueDescriptors makePsiFormerMixedValueDescriptors(
    const DenseForwardLayout& dense,
    const BatchedAttentionForwardLayout& attention,
    const BatchedValueLayout& value)
{
  validateDenseForwardLayout(dense);
  validateBatchedAttentionForwardLayout(attention);
  validateBatchedValueLayout(value);

  const std::size_t dense_rows = checkedProduct(
      attention.configuration_count, attention.attention.rows,
      "PsiFormer mixed dense row count overflow");
  const std::size_t feature_width = attention.attention.featureWidth();
  const std::size_t expected_configuration_stride = checkedProduct(
      attention.attention.rows, dense.target_row_stride,
      "PsiFormer mixed feature configuration stride overflow");

  if (dense.rows != dense_rows || dense.output_width != feature_width ||
      dense.target_row_stride != attention.attention.feature_row_stride ||
      attention.feature_configuration_stride != expected_configuration_stride)
    throw std::invalid_argument(
        "PsiFormer dense target is incompatible with batched attention features");
  if (value.configuration_count != attention.configuration_count ||
      value.rows != attention.attention.rows || value.width != feature_width ||
      value.row_stride != dense.target_row_stride ||
      value.configuration_stride != attention.feature_configuration_stride)
    throw std::invalid_argument(
        "PsiFormer value layout is incompatible with batched attention features");
  if (dense.targetElements() != attention.featureElements() ||
      dense.targetElements() != value.storageElements())
    throw std::invalid_argument(
        "PsiFormer mixed value descriptors do not name the same target span");

  std::uint64_t hash = FNV_OFFSET;
  mixInteger(hash, UINT64_C(1));
  mixInteger(hash, dense.rows);
  mixInteger(hash, dense.input_width);
  mixInteger(hash, dense.output_width);
  mixInteger(hash, dense.source_row_stride);
  mixInteger(hash, dense.weight_row_stride);
  mixInteger(hash, dense.target_row_stride);
  mixInteger(hash, attention.configuration_count);
  mixInteger(hash, attention.attention.rows);
  mixInteger(hash, attention.attention.heads);
  mixInteger(hash, attention.attention.head_width);
  mixInteger(hash, attention.attention.feature_row_stride);
  mixInteger(hash, attention.attention.attention_row_stride);
  mixInteger(hash, attention.attention.attention_head_stride);
  mixInteger(hash, attention.feature_configuration_stride);
  mixInteger(hash, attention.attention_configuration_stride);
  mixInteger(hash, value.configuration_count);
  mixInteger(hash, value.rows);
  mixInteger(hash, value.width);
  mixInteger(hash, value.row_stride);
  mixInteger(hash, value.configuration_stride);

  return {dense, attention, value, hash == 0 ? 1 : hash};
}

PsiFormerPrecisionParticipantMetadata makePsiFormerPrecisionParticipantMetadata(
    const PsiFormerPrecisionExecutionPlan& plan,
    const PsiFormerMixedValueDescriptors& descriptors)
{
  if (descriptors.fingerprint == 0)
    throw std::invalid_argument(
        "PsiFormer precision participant requires validated value descriptors");
  return {plan.policy,
          plan.math_mode,
          plan.backend,
          plan.blas_math.native,
          plan.canonical_source_version,
          plan.fingerprint,
          plan.device_layout_fingerprint,
          plan.arena.fingerprint,
          plan.conversion.fingerprint,
          descriptors.fingerprint,
          plan.blas_math.reduced_multiply,
          plan.full_precision_retry_available};
}

void validatePsiFormerPrecisionParticipantMetadata(
    const std::vector<PsiFormerPrecisionParticipantMetadata>& participants)
{
  if (participants.empty())
    throw std::invalid_argument(
        "PsiFormer precision operation requires participant metadata");
  const PsiFormerPrecisionParticipantMetadata& canonical = participants.front();
  if (canonical.plan_fingerprint == 0 ||
      canonical.device_layout_fingerprint == 0 ||
      canonical.arena_fingerprint == 0 ||
      canonical.conversion_fingerprint == 0 ||
      canonical.value_descriptor_fingerprint == 0)
    throw std::invalid_argument(
        "PsiFormer precision participant metadata is incomplete");
  if ((canonical.policy == PsiFormerPrecisionPolicy::FP64_REFERENCE &&
       canonical.math_mode != PsiFormerBackendMathMode::FP64_STRICT) ||
      (canonical.policy == PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE &&
       canonical.math_mode != PsiFormerBackendMathMode::FP32_STRICT) ||
      (canonical.policy == PsiFormerPrecisionPolicy::TF32_DENSE_FP64_SENSITIVE &&
       canonical.math_mode != PsiFormerBackendMathMode::CUDA_TF32))
    throw std::invalid_argument(
        "PsiFormer precision participant metadata has inconsistent policy/math mode");
  const PsiFormerBlasMathModePlan mapped =
      makePsiFormerBlasMathModePlan(canonical.math_mode, canonical.backend);
  if (mapped.native != canonical.native_math_mode ||
      mapped.reduced_multiply != canonical.reduced_multiply)
    throw std::invalid_argument(
        "PsiFormer precision participant metadata has inconsistent backend math mode");
  for (std::size_t index = 1; index < participants.size(); ++index)
    if (!(participants[index] == canonical))
      throw std::invalid_argument(
          "PsiFormer precision participant metadata differs across crowd/rank set");
}

PsiFormerPrecisionRecordingOrchestrator::PsiFormerPrecisionRecordingOrchestrator(
    PsiFormerPrecisionExecutionPlan plan,
    PsiFormerMixedValueDescriptors descriptors,
    std::size_t active_version,
    std::uint64_t active_fingerprint,
    std::uint8_t active_slot)
    : plan_(std::move(plan)),
      descriptors_(std::move(descriptors)),
      metadata_(makePsiFormerPrecisionParticipantMetadata(plan_, descriptors_)),
      publication_(plan_.policy, active_version, active_fingerprint, active_slot)
{}

void PsiFormerPrecisionRecordingOrchestrator::prepare()
{
  if (prepared_)
    throw std::logic_error("PsiFormer precision recorder is already prepared");
  validateRecordingPlan(plan_);
  const PsiFormerMixedValueDescriptors validated_descriptors =
      makePsiFormerMixedValueDescriptors(
      descriptors_.dense, descriptors_.attention, descriptors_.value);
  if (validated_descriptors.fingerprint != descriptors_.fingerprint)
    throw std::invalid_argument(
        "PsiFormer mixed value descriptor fingerprint does not match its contents");

  const std::size_t attention_rows = checkedProduct(
      descriptors_.attention.configuration_count,
      descriptors_.attention.attention.heads,
      "PsiFormer diagnostic attention-row bound overflow");
  const std::size_t softmax_rows = checkedProduct(
      attention_rows, descriptors_.attention.attention.rows,
      "PsiFormer diagnostic softmax-row bound overflow");
  const std::size_t attention_values = checkedProduct(
      softmax_rows, descriptors_.attention.attention.rows,
      "PsiFormer diagnostic attention-value bound overflow");
  const std::size_t nonlinear_and_promotion = checkedProduct(
      descriptors_.value.logicalElements(), 2,
      "PsiFormer diagnostic value-path bound overflow");
  const std::size_t sensitive_statuses = checkedProduct(
      descriptors_.attention.configuration_count, 3,
      "PsiFormer diagnostic sensitive-status bound overflow");
  execution_diagnostic_increment_bound_ = checkedSum(
      checkedSum(attention_values, softmax_rows,
                 "PsiFormer diagnostic counter bound overflow"),
      checkedSum(nonlinear_and_promotion, sensitive_statuses,
                 "PsiFormer diagnostic counter bound overflow"),
      "PsiFormer diagnostic counter bound overflow");

  constexpr std::size_t maximum_value_events = 14;
  const std::size_t maximum_publication_events = checkedSum(
      plan_.conversion.tiles.size(), 7,
      "PsiFormer precision publication event count overflow");
  const std::size_t requested_event_capacity =
      std::max(maximum_publication_events, maximum_value_events);
  events_.reserve(requested_event_capacity);
  if (events_.capacity() < requested_event_capacity)
    throw std::logic_error(
        "PsiFormer precision recorder failed to reserve its event graph");
  prepared_event_capacity_ = events_.capacity();
  ++allocation_count_;
  prepared_ = true;
}

void PsiFormerPrecisionRecordingOrchestrator::requirePrepared() const
{
  if (!prepared_)
    throw std::logic_error("PsiFormer precision recorder must be prepared first");
}

void PsiFormerPrecisionRecordingOrchestrator::beginRecording()
{
  requirePrepared();
  if (events_.capacity() != prepared_event_capacity_)
    throw std::logic_error(
        "PsiFormer precision recorder capacity changed after preparation");
  events_.clear();
}

void PsiFormerPrecisionRecordingOrchestrator::append(
    PsiFormerPrecisionRecordEvent event)
{
  if (events_.size() >= prepared_event_capacity_)
    throw std::logic_error(
        "PsiFormer precision recording exceeds its prepared event capacity");
  const std::size_t capacity = events_.capacity();
  events_.push_back(event);
  if (events_.capacity() != capacity)
    throw std::logic_error(
        "PsiFormer precision recording allocated after preparation");
}

std::size_t PsiFormerPrecisionRecordingOrchestrator::beginDiagnosticEpoch(
    PsiFormerPrecisionRecordEventKind clear_kind,
    std::size_t diagnostic_offset,
    std::size_t diagnostic_bytes,
    std::size_t version,
    std::uint8_t slot)
{
  validateDiagnosticSpan(
      diagnostic_offset, diagnostic_bytes, plan_.diagnostic_bytes);
  if (diagnostic_epoch_ == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("PsiFormer diagnostic epoch would wrap");
  ++diagnostic_epoch_;
  PsiFormerPrecisionRecordEvent event;
  event.kind = clear_kind;
  event.version = version;
  event.slot = slot;
  event.diagnostic_epoch = diagnostic_epoch_;
  event.destination_region = PsiFormerDeviceArenaRegion::NUMERICAL_DIAGNOSTICS;
  event.diagnostic_offset = diagnostic_offset;
  event.diagnostic_bytes = diagnostic_bytes;
  event.diagnostic_increment_bound = clear_kind ==
          PsiFormerPrecisionRecordEventKind::COMBINED_DIAGNOSTICS_CLEAR
      ? plan_.parameter_count
      : execution_diagnostic_increment_bound_;
  append(event);
  return diagnostic_epoch_;
}

void PsiFormerPrecisionRecordingOrchestrator::validateConversionDiagnostics(
    const PsiFormerParameterConversionDiagnostics& diagnostics) const
{
  const std::uint64_t bound = plan_.parameter_count;
  if (diagnostics.nonfinite_input_count > bound ||
      diagnostics.overflow_count > bound ||
      diagnostics.subnormal_output_count > bound ||
      diagnostics.underflow_to_zero_count > bound)
    throw std::invalid_argument(
        "PsiFormer conversion diagnostics exceed the prepared parameter bound");
  if (diagnostics.nonfinite_input_count >
          bound - diagnostics.overflow_count ||
      diagnostics.subnormal_output_count >
          bound - diagnostics.underflow_to_zero_count)
    throw std::invalid_argument(
        "PsiFormer conversion diagnostic categories exceed one conversion epoch");
  const std::uint64_t hard_count =
      diagnostics.nonfinite_input_count + diagnostics.overflow_count;
  const std::uint64_t benign_count =
      diagnostics.subnormal_output_count + diagnostics.underflow_to_zero_count;
  if (benign_count > bound - hard_count)
    throw std::invalid_argument(
        "PsiFormer conversion diagnostic categories exceed one conversion epoch");
}

void PsiFormerPrecisionRecordingOrchestrator::validateExecutionDiagnostics(
    const PsiFormerNumericalDiagnostics& diagnostics) const
{
  const std::uint64_t bound = execution_diagnostic_increment_bound_;
  std::uint64_t total = 0;
  const std::uint64_t counters[]{
      diagnostics.nonfinite_count,
      diagnostics.invalid_softmax_count,
      diagnostics.small_determinant_pivot_count,
      diagnostics.severe_cancellation_count,
      diagnostics.extreme_ecp_ratio_count};
  for (const std::uint64_t count : counters)
  {
    if (count > bound || total > bound - count)
      throw std::invalid_argument(
          "PsiFormer execution diagnostics exceed the prepared epoch bound");
    total += count;
  }
}

void PsiFormerPrecisionRecordingOrchestrator::recordPublicationStart()
{
  requirePrepared();
  if (phase_ != RecordingPhase::IDLE)
    throw std::logic_error("PsiFormer precision recorder already has pending work");

  const std::size_t version = plan_.canonical_source_version;
  publication_.begin(version, plan_.fingerprint);
  try
  {
    beginRecording();
    const std::uint8_t slot = *publication_.pendingSlot();
    if (slot == publication_.activeSlot())
      throw std::logic_error(
          "PsiFormer publication attempted to overwrite the active slot");

    PsiFormerPrecisionRecordEvent copy;
    copy.kind = PsiFormerPrecisionRecordEventKind::INACTIVE_MASTER_COPY;
    copy.version = version;
    copy.slot = slot;
    copy.destination_region = masterRegion(slot);
    copy.count = plan_.parameter_count;
    append(copy);

    const std::size_t epoch = beginDiagnosticEpoch(
        PsiFormerPrecisionRecordEventKind::COMBINED_DIAGNOSTICS_CLEAR,
        /*diagnostic_offset=*/0, sizeof(PsiFormerDeviceNumericalDiagnostics),
        version, slot);
    for (const PsiFormerParameterConversionTile& tile : plan_.conversion.tiles)
    {
      PsiFormerPrecisionRecordEvent conversion;
      conversion.kind =
          PsiFormerPrecisionRecordEventKind::PARAMETER_CONVERSION_TILE;
      conversion.version = version;
      conversion.slot = slot;
      conversion.diagnostic_epoch = epoch;
      conversion.source_region = masterRegion(slot);
      conversion.destination_region = computeRegion(slot);
      conversion.source_begin = tile.source_begin;
      conversion.destination_begin = tile.destination_begin;
      conversion.count = tile.count;
      conversion.diagnostic_offset =
          offsetof(PsiFormerDeviceNumericalDiagnostics, conversion);
      conversion.diagnostic_bytes =
          sizeof(PsiFormerParameterConversionDiagnostics);
      conversion.diagnostic_increment_bound = plan_.parameter_count;
      append(conversion);
    }

    PsiFormerPrecisionRecordEvent completion_barrier;
    completion_barrier.kind =
        PsiFormerPrecisionRecordEventKind::PARAMETER_DEVICE_COMPLETION_BARRIER;
    completion_barrier.version = version;
    completion_barrier.slot = slot;
    completion_barrier.diagnostic_epoch = epoch;
    completion_barrier.source_region = computeRegion(slot);
    completion_barrier.count = plan_.parameter_count;
    append(completion_barrier);

    PsiFormerPrecisionRecordEvent readback_barrier;
    readback_barrier.kind = PsiFormerPrecisionRecordEventKind::
        PARAMETER_DIAGNOSTIC_READBACK_BARRIER;
    readback_barrier.version = version;
    readback_barrier.slot = slot;
    readback_barrier.diagnostic_epoch = epoch;
    readback_barrier.source_region =
        PsiFormerDeviceArenaRegion::NUMERICAL_DIAGNOSTICS;
    readback_barrier.diagnostic_offset =
        offsetof(PsiFormerDeviceNumericalDiagnostics, conversion);
    readback_barrier.diagnostic_bytes =
        sizeof(PsiFormerParameterConversionDiagnostics);
    readback_barrier.diagnostic_increment_bound = plan_.parameter_count;
    append(readback_barrier);

    pending_diagnostic_epoch_ = epoch;
    phase_ = RecordingPhase::PUBLICATION_PENDING;
  }
  catch (...)
  {
    if (publication_.pendingVersion() == version)
      publication_.cancel(version);
    throw;
  }
}

void PsiFormerPrecisionRecordingOrchestrator::cancelPublicationStart()
{
  requirePrepared();
  if (phase_ != RecordingPhase::PUBLICATION_PENDING ||
      publication_.pendingVersion() != plan_.canonical_source_version)
    throw std::logic_error("PsiFormer has no recorded publication to cancel");

  const std::size_t version = plan_.canonical_source_version;
  const std::uint8_t slot = *publication_.pendingSlot();
  publication_.cancel(version);
  PsiFormerPrecisionRecordEvent cancellation;
  cancellation.kind = PsiFormerPrecisionRecordEventKind::PARAMETER_CANCELLATION;
  cancellation.version = version;
  cancellation.slot = slot;
  cancellation.diagnostic_epoch = pending_diagnostic_epoch_;
  append(cancellation);
  phase_ = RecordingPhase::IDLE;
}

PsiFormerPrecisionPublicationResult
PsiFormerPrecisionRecordingOrchestrator::observePublicationCompletion(
    const PsiFormerParameterConversionDiagnostics& diagnostics)
{
  requirePrepared();
  if (phase_ != RecordingPhase::PUBLICATION_PENDING ||
      publication_.pendingVersion() != plan_.canonical_source_version)
    throw std::logic_error(
        "PsiFormer publication completion lacks matching recorded work");
  validateConversionDiagnostics(diagnostics);

  const std::size_t version = plan_.canonical_source_version;
  const std::uint8_t slot = *publication_.pendingSlot();
  try
  {
    publication_.markMasterCopied(version);
    publication_.markComputeConversionComplete(version);

    PsiFormerPrecisionRecordEvent decision;
    decision.kind =
        PsiFormerPrecisionRecordEventKind::PARAMETER_DIAGNOSTIC_DECISION;
    decision.version = version;
    decision.slot = slot;
    decision.diagnostic_epoch = pending_diagnostic_epoch_;
    decision.source_region = PsiFormerDeviceArenaRegion::NUMERICAL_DIAGNOSTICS;
    decision.diagnostic_offset =
        offsetof(PsiFormerDeviceNumericalDiagnostics, conversion);
    decision.diagnostic_bytes = sizeof(PsiFormerParameterConversionDiagnostics);
    decision.diagnostic_increment_bound = plan_.parameter_count;
    decision.hard_hazard = diagnostics.hasPublicationHazard();
    append(decision);

    if (!publication_.acceptDiagnostics(version, diagnostics))
    {
      phase_ = RecordingPhase::IDLE;
      return PsiFormerPrecisionPublicationResult::REJECTED_DIAGNOSTICS;
    }

    PsiFormerPrecisionRecordEvent completion;
    completion.kind = PsiFormerPrecisionRecordEventKind::PARAMETER_COMPLETION;
    completion.version = version;
    completion.slot = slot;
    completion.diagnostic_epoch = pending_diagnostic_epoch_;
    append(completion);

    publication_.publish(version);
    PsiFormerPrecisionRecordEvent publication;
    publication.kind = PsiFormerPrecisionRecordEventKind::PARAMETER_PUBLICATION;
    publication.version = version;
    publication.slot = slot;
    publication.diagnostic_epoch = pending_diagnostic_epoch_;
    append(publication);
    phase_ = RecordingPhase::IDLE;
    return PsiFormerPrecisionPublicationResult::PUBLISHED;
  }
  catch (...)
  {
    if (publication_.pendingVersion() == version)
      publication_.cancel(version);
    phase_ = RecordingPhase::IDLE;
    throw;
  }
}

void PsiFormerPrecisionRecordingOrchestrator::recordValueEvaluationStart(
    std::size_t version,
    std::uint64_t execution_fingerprint)
{
  requirePrepared();
  if (phase_ != RecordingPhase::IDLE)
    throw std::logic_error("PsiFormer precision recorder already has pending work");
  if (version != plan_.canonical_source_version ||
      execution_fingerprint != plan_.fingerprint ||
      !publication_.isReady(version, execution_fingerprint))
    throw std::logic_error(
        "PsiFormer mixed evaluation requested an unpublished or stale model");

  beginRecording();
  const std::uint8_t slot = publication_.activeSlot();
  const std::size_t execution_offset =
      offsetof(PsiFormerDeviceNumericalDiagnostics, execution);
  const std::size_t epoch = beginDiagnosticEpoch(
      PsiFormerPrecisionRecordEventKind::EXECUTION_DIAGNOSTICS_CLEAR,
      execution_offset, sizeof(PsiFormerNumericalDiagnostics), version, slot);

  const std::size_t dense_source_values = checkedProduct(
      descriptors_.dense.rows, descriptors_.dense.input_width,
      "PsiFormer mixed dense source event extent overflow");
  const std::size_t value_count = descriptors_.value.logicalElements();

  PsiFormerPrecisionRecordEvent fp32_input;
  fp32_input.kind = PsiFormerPrecisionRecordEventKind::VALUE_FP64_TO_FP32;
  fp32_input.version = version;
  fp32_input.slot = slot;
  fp32_input.diagnostic_epoch = epoch;
  fp32_input.count = dense_source_values;
  append(fp32_input);

  PsiFormerPrecisionRecordEvent network;
  network.kind = PsiFormerPrecisionRecordEventKind::VALUE_FP32_NETWORK;
  network.version = version;
  network.slot = slot;
  network.diagnostic_epoch = epoch;
  network.source_region = computeRegion(slot);
  network.count = value_count;
  append(network);

  PsiFormerPrecisionRecordEvent fp64_output;
  fp64_output.kind = PsiFormerPrecisionRecordEventKind::VALUE_FP32_TO_FP64;
  fp64_output.version = version;
  fp64_output.slot = slot;
  fp64_output.diagnostic_epoch = epoch;
  fp64_output.count = value_count;
  append(fp64_output);

  PsiFormerPrecisionRecordEvent sensitive;
  sensitive.kind = PsiFormerPrecisionRecordEventKind::VALUE_FP64_SENSITIVE;
  sensitive.version = version;
  sensitive.slot = slot;
  sensitive.diagnostic_epoch = epoch;
  sensitive.count = descriptors_.attention.configuration_count;
  append(sensitive);

  PsiFormerPrecisionRecordEvent completion_barrier;
  completion_barrier.kind =
      PsiFormerPrecisionRecordEventKind::EXECUTION_DEVICE_COMPLETION_BARRIER;
  completion_barrier.version = version;
  completion_barrier.slot = slot;
  completion_barrier.diagnostic_epoch = epoch;
  completion_barrier.count = descriptors_.attention.configuration_count;
  append(completion_barrier);

  PsiFormerPrecisionRecordEvent readback_barrier;
  readback_barrier.kind = PsiFormerPrecisionRecordEventKind::
      EXECUTION_DIAGNOSTIC_READBACK_BARRIER;
  readback_barrier.version = version;
  readback_barrier.slot = slot;
  readback_barrier.diagnostic_epoch = epoch;
  readback_barrier.source_region =
      PsiFormerDeviceArenaRegion::NUMERICAL_DIAGNOSTICS;
  readback_barrier.diagnostic_offset = execution_offset;
  readback_barrier.diagnostic_bytes = sizeof(PsiFormerNumericalDiagnostics);
  readback_barrier.diagnostic_increment_bound =
      execution_diagnostic_increment_bound_;
  append(readback_barrier);

  pending_diagnostic_epoch_ = epoch;
  phase_ = RecordingPhase::MIXED_VALUE_PENDING;
}

PsiFormerPrecisionEvaluationResult
PsiFormerPrecisionRecordingOrchestrator::observeMixedValueCompletion(
    const PsiFormerNumericalDiagnostics& mixed_diagnostics)
{
  requirePrepared();
  if (phase_ != RecordingPhase::MIXED_VALUE_PENDING)
    throw std::logic_error(
        "PsiFormer mixed completion lacks matching recorded work");
  validateExecutionDiagnostics(mixed_diagnostics);

  const std::size_t version = plan_.canonical_source_version;
  const std::uint8_t slot = publication_.activeSlot();
  const std::size_t execution_offset =
      offsetof(PsiFormerDeviceNumericalDiagnostics, execution);
  const bool mixed_hazard = mixed_diagnostics.requiresFullPrecisionRetry();

  PsiFormerPrecisionRecordEvent decision;
  decision.kind =
      PsiFormerPrecisionRecordEventKind::EXECUTION_DIAGNOSTIC_DECISION;
  decision.version = version;
  decision.slot = slot;
  decision.diagnostic_epoch = pending_diagnostic_epoch_;
  decision.source_region = PsiFormerDeviceArenaRegion::NUMERICAL_DIAGNOSTICS;
  decision.diagnostic_offset = execution_offset;
  decision.diagnostic_bytes = sizeof(PsiFormerNumericalDiagnostics);
  decision.diagnostic_increment_bound = execution_diagnostic_increment_bound_;
  decision.hard_hazard = mixed_hazard;
  append(decision);

  if (!mixed_hazard)
  {
    PsiFormerPrecisionRecordEvent complete;
    complete.kind = PsiFormerPrecisionRecordEventKind::EVALUATION_COMPLETE;
    complete.version = version;
    complete.slot = slot;
    complete.diagnostic_epoch = pending_diagnostic_epoch_;
    append(complete);
    phase_ = RecordingPhase::IDLE;
    return PsiFormerPrecisionEvaluationResult::MIXED_SUCCESS;
  }

  if (!plan_.full_precision_retry_available)
  {
    PsiFormerPrecisionRecordEvent failure;
    failure.kind = PsiFormerPrecisionRecordEventKind::FINAL_FAILURE;
    failure.version = version;
    failure.slot = slot;
    failure.diagnostic_epoch = pending_diagnostic_epoch_;
    failure.hard_hazard = true;
    append(failure);
    phase_ = RecordingPhase::IDLE;
    return PsiFormerPrecisionEvaluationResult::FINAL_FAILURE;
  }

  const std::size_t retry_epoch = beginDiagnosticEpoch(
      PsiFormerPrecisionRecordEventKind::EXECUTION_DIAGNOSTICS_CLEAR,
      execution_offset, sizeof(PsiFormerNumericalDiagnostics), version, slot);
  PsiFormerPrecisionRecordEvent retry;
  retry.kind = PsiFormerPrecisionRecordEventKind::WHOLE_BATCH_FP64_RETRY;
  retry.version = version;
  retry.slot = slot;
  retry.diagnostic_epoch = retry_epoch;
  retry.count = descriptors_.attention.configuration_count;
  append(retry);

  PsiFormerPrecisionRecordEvent completion_barrier;
  completion_barrier.kind =
      PsiFormerPrecisionRecordEventKind::EXECUTION_DEVICE_COMPLETION_BARRIER;
  completion_barrier.version = version;
  completion_barrier.slot = slot;
  completion_barrier.diagnostic_epoch = retry_epoch;
  completion_barrier.count = descriptors_.attention.configuration_count;
  append(completion_barrier);

  PsiFormerPrecisionRecordEvent readback_barrier;
  readback_barrier.kind = PsiFormerPrecisionRecordEventKind::
      EXECUTION_DIAGNOSTIC_READBACK_BARRIER;
  readback_barrier.version = version;
  readback_barrier.slot = slot;
  readback_barrier.diagnostic_epoch = retry_epoch;
  readback_barrier.source_region =
      PsiFormerDeviceArenaRegion::NUMERICAL_DIAGNOSTICS;
  readback_barrier.diagnostic_offset = execution_offset;
  readback_barrier.diagnostic_bytes = sizeof(PsiFormerNumericalDiagnostics);
  readback_barrier.diagnostic_increment_bound =
      execution_diagnostic_increment_bound_;
  append(readback_barrier);

  pending_diagnostic_epoch_ = retry_epoch;
  phase_ = RecordingPhase::FP64_RETRY_PENDING;
  return PsiFormerPrecisionEvaluationResult::FP64_RETRY_PENDING;
}

PsiFormerPrecisionEvaluationResult
PsiFormerPrecisionRecordingOrchestrator::observeFullPrecisionRetryCompletion(
    const PsiFormerNumericalDiagnostics& retry_diagnostics)
{
  requirePrepared();
  if (phase_ != RecordingPhase::FP64_RETRY_PENDING)
    throw std::logic_error(
        "PsiFormer FP64 retry completion lacks matching recorded work");
  validateExecutionDiagnostics(retry_diagnostics);

  const std::size_t version = plan_.canonical_source_version;
  const std::uint8_t slot = publication_.activeSlot();
  const std::size_t execution_offset =
      offsetof(PsiFormerDeviceNumericalDiagnostics, execution);
  const bool retry_hazard = retry_diagnostics.requiresFullPrecisionRetry();

  PsiFormerPrecisionRecordEvent decision;
  decision.kind =
      PsiFormerPrecisionRecordEventKind::EXECUTION_DIAGNOSTIC_DECISION;
  decision.version = version;
  decision.slot = slot;
  decision.diagnostic_epoch = pending_diagnostic_epoch_;
  decision.source_region = PsiFormerDeviceArenaRegion::NUMERICAL_DIAGNOSTICS;
  decision.diagnostic_offset = execution_offset;
  decision.diagnostic_bytes = sizeof(PsiFormerNumericalDiagnostics);
  decision.diagnostic_increment_bound = execution_diagnostic_increment_bound_;
  decision.hard_hazard = retry_hazard;
  append(decision);

  PsiFormerPrecisionRecordEvent final_event;
  final_event.kind = retry_hazard
      ? PsiFormerPrecisionRecordEventKind::FINAL_FAILURE
      : PsiFormerPrecisionRecordEventKind::EVALUATION_COMPLETE;
  final_event.version = version;
  final_event.slot = slot;
  final_event.diagnostic_epoch = pending_diagnostic_epoch_;
  final_event.hard_hazard = retry_hazard;
  append(final_event);
  phase_ = RecordingPhase::IDLE;
  return retry_hazard
      ? PsiFormerPrecisionEvaluationResult::FINAL_FAILURE
      : PsiFormerPrecisionEvaluationResult::FP64_RETRY_SUCCESS;
}

} // namespace qmcplusplus::psiformer
