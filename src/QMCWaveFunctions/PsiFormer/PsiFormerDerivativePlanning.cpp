//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDerivativePlanning.cpp
 * @brief Checked FP64 layouts and launch records for PsiFormer derivatives.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerDerivativePlanning.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer
{
namespace
{

constexpr std::uint64_t FNV_OFFSET = UINT64_C(14695981039346656037);
constexpr std::uint64_t FNV_PRIME  = UINT64_C(1099511628211);
constexpr std::size_t FP64_BYTES   = 8;
constexpr std::size_t FP64_ALIGNMENT_ELEMENTS = 16;
constexpr std::size_t DEVICE_REGION_ALIGNMENT_BYTES = 256;

static_assert(sizeof(double) == FP64_BYTES, "PsiFormer derivative planning requires binary64 storage");

void mixInteger(std::uint64_t& hash, std::uint64_t value) noexcept
{
  for (unsigned shift = 0; shift < 64; shift += 8)
  {
    hash ^= static_cast<std::uint8_t>((value >> shift) & UINT64_C(0xff));
    hash *= FNV_PRIME;
  }
}

std::uint64_t finishFingerprint(std::uint64_t hash) noexcept { return hash == 0 ? 1 : hash; }

std::size_t checkedAdd(std::size_t left, std::size_t right, const char* message)
{
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::length_error(message);
  return left + right;
}

std::size_t checkedProduct(std::size_t left, std::size_t right, const char* message)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(message);
  return left * right;
}

std::size_t checkedAlign(std::size_t value, std::size_t alignment, const char* message)
{
  const std::size_t mask = alignment - 1;
  return checkedAdd(value, mask, message) & ~mask;
}

std::size_t checkedBytes(std::size_t fp64_elements, const char* message)
{
  return checkedProduct(fp64_elements, FP64_BYTES, message);
}

std::size_t appendAlignedFp64(std::size_t& high_water,
                              std::size_t count,
                              const char* message)
{
  if (count == 0)
    return high_water;
  const std::size_t offset = checkedAlign(high_water, FP64_ALIGNMENT_ELEMENTS, message);
  high_water = checkedAdd(offset, count, message);
  return offset;
}

std::size_t appendAlignedBytes(std::size_t& high_water,
                               std::size_t bytes,
                               std::size_t alignment,
                               const char* message)
{
  if (bytes == 0)
    return high_water;
  const std::size_t offset = checkedAlign(high_water, alignment, message);
  high_water = checkedAdd(offset, bytes, message);
  return offset;
}

std::size_t rootSeedCapacity(std::size_t electron_count, bool kinetic_enabled)
{
  if (!kinetic_enabled)
    return 1;
  return checkedAdd(1, checkedProduct(4, electron_count,
                                      "PsiFormer derivative root-seed extent overflowed"),
                    "PsiFormer derivative root-seed extent overflowed");
}

std::size_t minimumScoreTape(std::size_t electron_count,
                             std::size_t determinant_count,
                             std::size_t embedding_input_width,
                             std::size_t feature_width,
                             std::size_t attention_head_count,
                             std::size_t attention_block_count)
{
  const std::size_t raw_features = checkedProduct(
      electron_count, embedding_input_width,
      "PsiFormer score-tape raw-feature extent overflowed");
  const std::size_t feature_plane = checkedProduct(
      electron_count, feature_width,
      "PsiFormer score-tape feature extent overflowed");
  const std::size_t feature_boundaries = checkedProduct(
      checkedAdd(attention_block_count, 1,
                 "PsiFormer score-tape block count overflowed"),
      feature_plane, "PsiFormer score-tape feature-boundary extent overflowed");
  const std::size_t block_vectors = checkedProduct(
      checkedProduct(7, attention_block_count,
                     "PsiFormer score-tape block-vector count overflowed"),
      feature_plane, "PsiFormer score-tape block-vector extent overflowed");
  const std::size_t electron_square = checkedProduct(
      electron_count, electron_count,
      "PsiFormer score-tape electron-square extent overflowed");
  const std::size_t attention = checkedProduct(
      checkedProduct(attention_block_count, attention_head_count,
                     "PsiFormer score-tape attention count overflowed"),
      electron_square, "PsiFormer score-tape attention extent overflowed");
  const std::size_t orbitals = checkedProduct(
      determinant_count, electron_square,
      "PsiFormer score-tape orbital extent overflowed");
  return checkedAdd(
      checkedAdd(raw_features,
                 checkedAdd(feature_boundaries, block_vectors,
                            "PsiFormer score-tape extent overflowed"),
                 "PsiFormer score-tape extent overflowed"),
      checkedAdd(attention, orbitals,
                 "PsiFormer score-tape extent overflowed"),
      "PsiFormer score-tape extent overflowed");
}

PsiFormerDerivativeReverseLayout makeReverseLayout(
    std::size_t parameter_count,
    std::size_t electron_count,
    const PsiFormerDerivativePlanningRequirements& requirements)
{
  PsiFormerDerivativeReverseLayout layout;
  layout.retained_tape_fp64_elements = requirements.retained_tape_fp64_elements;
  layout.adjoint_tape_fp64_elements = requirements.adjoint_tape_fp64_elements;
  layout.determinant_factor_fp64_elements = requirements.determinant_factor_fp64_elements;
  layout.determinant_scratch_fp64_elements = requirements.determinant_scratch_fp64_elements;
  layout.root_seed_capacity_fp64_elements =
      parameter_count == 0 ? 0 : rootSeedCapacity(electron_count,
                                                  requirements.enable_kinetic_response);
  layout.response_fp64_elements = parameter_count;

  std::size_t high_water = 0;
  layout.retained_tape_offset_fp64_elements = appendAlignedFp64(
      high_water, layout.retained_tape_fp64_elements,
      "PsiFormer retained derivative tape extent overflowed");
  layout.adjoint_tape_offset_fp64_elements = appendAlignedFp64(
      high_water, layout.adjoint_tape_fp64_elements,
      "PsiFormer derivative adjoint tape extent overflowed");
  layout.determinant_factor_offset_fp64_elements = appendAlignedFp64(
      high_water, layout.determinant_factor_fp64_elements,
      "PsiFormer determinant-factor derivative extent overflowed");
  layout.determinant_scratch_offset_fp64_elements = appendAlignedFp64(
      high_water, layout.determinant_scratch_fp64_elements,
      "PsiFormer determinant derivative scratch extent overflowed");
  layout.root_seed_offset_fp64_elements = appendAlignedFp64(
      high_water, layout.root_seed_capacity_fp64_elements,
      "PsiFormer derivative root-seed extent overflowed");
  layout.response_offset_fp64_elements = appendAlignedFp64(
      high_water, layout.response_fp64_elements,
      "PsiFormer derivative response extent overflowed");
  layout.total_fp64_elements = high_water;
  layout.total_bytes = checkedBytes(high_water,
                                    "PsiFormer reverse-workspace byte extent overflowed");

  std::uint64_t hash = FNV_OFFSET;
  mixInteger(hash, UINT64_C(1));
  for (const std::size_t value : {
           layout.retained_tape_offset_fp64_elements,
           layout.retained_tape_fp64_elements,
           layout.adjoint_tape_offset_fp64_elements,
           layout.adjoint_tape_fp64_elements,
           layout.determinant_factor_offset_fp64_elements,
           layout.determinant_factor_fp64_elements,
           layout.determinant_scratch_offset_fp64_elements,
           layout.determinant_scratch_fp64_elements,
           layout.root_seed_offset_fp64_elements,
           layout.root_seed_capacity_fp64_elements,
           layout.response_offset_fp64_elements,
           layout.response_fp64_elements,
           layout.total_fp64_elements,
           layout.total_bytes})
    mixInteger(hash, value);
  layout.fingerprint = finishFingerprint(hash);
  return layout;
}

PsiFormerParameterAccumulatorLayout makeAccumulatorLayout(std::size_t parameter_count,
                                                          std::size_t accumulator_count)
{
  PsiFormerParameterAccumulatorLayout layout;
  layout.parameter_count = parameter_count;
  layout.complex_accumulator_count = accumulator_count;
  if (parameter_count != 0 && accumulator_count != 0)
  {
    const std::size_t complex_elements = checkedProduct(
        2, parameter_count, "PsiFormer complex parameter-accumulator extent overflowed");
    layout.complex_accumulator_stride_fp64_elements = checkedAlign(
        complex_elements, FP64_ALIGNMENT_ELEMENTS,
        "PsiFormer complex parameter-accumulator alignment overflowed");
    layout.jvp_direction_fp64_elements = parameter_count;
    layout.total_fp64_elements = checkedProduct(
        accumulator_count, layout.complex_accumulator_stride_fp64_elements,
        "PsiFormer parameter-accumulator extent overflowed");
  }
  layout.total_bytes = checkedBytes(layout.total_fp64_elements,
                                    "PsiFormer parameter-accumulator byte extent overflowed");

  std::uint64_t hash = FNV_OFFSET;
  mixInteger(hash, UINT64_C(1));
  for (const std::size_t value : {
           layout.parameter_count, layout.complex_accumulator_count,
           layout.complex_accumulator_stride_fp64_elements,
           layout.jvp_direction_offset_fp64_elements,
           layout.jvp_direction_fp64_elements, layout.total_fp64_elements,
           layout.total_bytes})
    mixInteger(hash, value);
  layout.fingerprint = finishFingerprint(hash);
  return layout;
}

PsiFormerECPTileLayout makeECPTileLayout(std::size_t sample_capacity,
                                         std::size_t segment_capacity,
                                         std::size_t point_capacity,
                                         std::size_t channel_capacity,
                                         std::size_t electron_count,
                                         std::size_t stamp_capacity)
{
  PsiFormerECPTileLayout layout;
  layout.maximum_sample_count = sample_capacity;
  layout.maximum_segment_count = segment_capacity;
  layout.maximum_point_count = point_capacity;
  layout.maximum_channel_count = channel_capacity;
  layout.electron_count = electron_count;
  layout.maximum_evaluation_stamp_count = stamp_capacity;
  if (sample_capacity == 0 && segment_capacity == 0 && point_capacity == 0)
  {
    std::uint64_t hash = FNV_OFFSET;
    mixInteger(hash, UINT64_C(1));
    for (const std::size_t value : {sample_capacity, segment_capacity,
                                    point_capacity, channel_capacity,
                                    electron_count, stamp_capacity})
      mixInteger(hash, value);
    layout.fingerprint = finishFingerprint(hash);
    return layout;
  }

  std::size_t high_water = 0;
  layout.metadata_bytes = sizeof(PsiFormerECPTileMetadata);
  layout.metadata_offset_bytes = appendAlignedBytes(
      high_water, layout.metadata_bytes, alignof(PsiFormerECPTileMetadata),
      "PsiFormer ECP metadata extent overflowed");
  layout.segment_offsets_offset_bytes = appendAlignedBytes(
      high_water,
      checkedProduct(checkedAdd(segment_capacity, 1,
                                "PsiFormer ECP segment-count overflowed"),
                     sizeof(std::size_t), "PsiFormer ECP segment-offset extent overflowed"),
      alignof(std::size_t), "PsiFormer ECP segment-offset extent overflowed");
  layout.segment_walker_ids_offset_bytes = appendAlignedBytes(
      high_water,
      checkedProduct(segment_capacity, sizeof(std::size_t),
                     "PsiFormer ECP segment-walker extent overflowed"),
      alignof(std::size_t), "PsiFormer ECP segment-walker extent overflowed");
  layout.segment_electron_ids_offset_bytes = appendAlignedBytes(
      high_water,
      checkedProduct(segment_capacity, sizeof(std::size_t),
                     "PsiFormer ECP segment-electron extent overflowed"),
      alignof(std::size_t), "PsiFormer ECP segment-electron extent overflowed");
  layout.segment_source_center_ids_offset_bytes = appendAlignedBytes(
      high_water,
      checkedProduct(segment_capacity, sizeof(std::size_t),
                     "PsiFormer ECP source-center extent overflowed"),
      alignof(std::size_t), "PsiFormer ECP source-center extent overflowed");
  layout.segment_flags_offset_bytes = appendAlignedBytes(
      high_water,
      checkedProduct(segment_capacity, sizeof(std::uint8_t),
                     "PsiFormer ECP segment-flag extent overflowed"),
      alignof(std::uint8_t), "PsiFormer ECP segment-flag extent overflowed");
  const std::size_t reference_position_count = checkedProduct(
      checkedProduct(sample_capacity, electron_count,
                     "PsiFormer ECP reference-position extent overflowed"),
      3, "PsiFormer ECP reference-position extent overflowed");
  layout.reference_positions_offset_bytes = appendAlignedBytes(
      high_water,
      checkedBytes(reference_position_count,
                   "PsiFormer ECP reference-position byte extent overflowed"),
      alignof(double), "PsiFormer ECP reference-position extent overflowed");
  layout.replacement_positions_offset_bytes = appendAlignedBytes(
      high_water,
      checkedBytes(checkedProduct(3, point_capacity,
                                  "PsiFormer ECP replacement-position extent overflowed"),
                   "PsiFormer ECP replacement-position byte extent overflowed"),
      alignof(double), "PsiFormer ECP replacement-position byte extent overflowed");
  layout.bare_weights_offset_bytes = appendAlignedBytes(
      high_water,
      checkedBytes(point_capacity,
                   "PsiFormer ECP real bare-weight byte extent overflowed"),
      alignof(double), "PsiFormer ECP bare-weight extent overflowed");
  layout.complete_ratios_offset_bytes = appendAlignedBytes(
      high_water,
      checkedBytes(checkedProduct(2, point_capacity,
                                  "PsiFormer ECP complex-ratio extent overflowed"),
                   "PsiFormer ECP complex-ratio byte extent overflowed"),
      alignof(double), "PsiFormer ECP complex-ratio extent overflowed");
  layout.evaluation_stamps_offset_bytes = appendAlignedBytes(
      high_water,
      checkedProduct(stamp_capacity, sizeof(PsiFormerECPEvaluationStampRecord),
                     "PsiFormer ECP evaluation-stamp extent overflowed"),
      alignof(PsiFormerECPEvaluationStampRecord),
      "PsiFormer ECP evaluation-stamp extent overflowed");
  layout.coefficients_offset_bytes = appendAlignedBytes(
      high_water,
      checkedBytes(checkedProduct(
          2, checkedProduct(channel_capacity, sample_capacity,
                            "PsiFormer ECP coefficient extent overflowed"),
          "PsiFormer ECP complex coefficient extent overflowed"),
          "PsiFormer ECP coefficient byte extent overflowed"),
      alignof(double), "PsiFormer ECP coefficient extent overflowed");
  layout.sample_weight_sums_offset_bytes = appendAlignedBytes(
      high_water,
      checkedBytes(checkedProduct(2, sample_capacity,
                                  "PsiFormer ECP complex-weight-sum extent overflowed"),
                   "PsiFormer ECP complex-weight-sum byte extent overflowed"),
      alignof(double), "PsiFormer ECP complex-weight-sum extent overflowed");
  layout.point_status_offset_bytes = appendAlignedBytes(
      high_water,
      checkedProduct(point_capacity, sizeof(std::uint32_t),
                     "PsiFormer ECP point-status extent overflowed"),
      alignof(std::uint32_t), "PsiFormer ECP point-status extent overflowed");
  layout.total_bytes = high_water;

  std::uint64_t hash = FNV_OFFSET;
  mixInteger(hash, UINT64_C(1));
  for (const std::size_t value : {
           layout.maximum_sample_count, layout.maximum_segment_count,
           layout.maximum_point_count, layout.maximum_channel_count,
           layout.electron_count, layout.maximum_evaluation_stamp_count,
           layout.metadata_offset_bytes, layout.metadata_bytes,
           layout.segment_offsets_offset_bytes,
           layout.segment_walker_ids_offset_bytes,
           layout.segment_electron_ids_offset_bytes,
           layout.segment_source_center_ids_offset_bytes,
           layout.segment_flags_offset_bytes,
           layout.reference_positions_offset_bytes,
           layout.replacement_positions_offset_bytes,
           layout.bare_weights_offset_bytes, layout.complete_ratios_offset_bytes,
           layout.evaluation_stamps_offset_bytes, layout.coefficients_offset_bytes,
           layout.sample_weight_sums_offset_bytes,
           layout.point_status_offset_bytes, layout.total_bytes})
    mixInteger(hash, value);
  layout.fingerprint = finishFingerprint(hash);
  return layout;
}

PsiFormerPinnedHostStagingLayout makeHostStagingLayout(std::size_t chunk_capacity,
                                                       std::size_t jvp_capacity)
{
  PsiFormerPinnedHostStagingLayout layout;
  layout.maximum_parameter_chunk_elements = chunk_capacity;
  layout.maximum_jvp_output_complex_elements = jvp_capacity;
  layout.complex_chunk_bytes = checkedBytes(
      checkedProduct(2, chunk_capacity,
                     "PsiFormer pinned-host complex chunk extent overflowed"),
      "PsiFormer pinned-host complex chunk byte extent overflowed");
  layout.jvp_output_bytes = checkedBytes(
      checkedProduct(2, jvp_capacity,
                     "PsiFormer pinned-host complex JVP extent overflowed"),
      "PsiFormer pinned-host complex JVP byte extent overflowed");
  // These two host views are never live together and intentionally alias byte zero.
  layout.total_bytes = std::max(layout.complex_chunk_bytes, layout.jvp_output_bytes);

  std::uint64_t hash = FNV_OFFSET;
  mixInteger(hash, UINT64_C(1));
  for (const std::size_t value : {
           layout.maximum_parameter_chunk_elements,
           layout.maximum_jvp_output_complex_elements,
           layout.complex_chunk_offset_bytes, layout.complex_chunk_bytes,
           layout.jvp_output_offset_bytes, layout.jvp_output_bytes,
           layout.total_bytes})
    mixInteger(hash, value);
  layout.fingerprint = finishFingerprint(hash);
  return layout;
}

bool sameReverseLayout(const PsiFormerDerivativeReverseLayout& left,
                       const PsiFormerDerivativeReverseLayout& right) noexcept
{
  return left.retained_tape_offset_fp64_elements == right.retained_tape_offset_fp64_elements &&
      left.retained_tape_fp64_elements == right.retained_tape_fp64_elements &&
      left.adjoint_tape_offset_fp64_elements == right.adjoint_tape_offset_fp64_elements &&
      left.adjoint_tape_fp64_elements == right.adjoint_tape_fp64_elements &&
      left.determinant_factor_offset_fp64_elements == right.determinant_factor_offset_fp64_elements &&
      left.determinant_factor_fp64_elements == right.determinant_factor_fp64_elements &&
      left.determinant_scratch_offset_fp64_elements == right.determinant_scratch_offset_fp64_elements &&
      left.determinant_scratch_fp64_elements == right.determinant_scratch_fp64_elements &&
      left.root_seed_offset_fp64_elements == right.root_seed_offset_fp64_elements &&
      left.root_seed_capacity_fp64_elements == right.root_seed_capacity_fp64_elements &&
      left.response_offset_fp64_elements == right.response_offset_fp64_elements &&
      left.response_fp64_elements == right.response_fp64_elements &&
      left.total_fp64_elements == right.total_fp64_elements &&
      left.total_bytes == right.total_bytes && left.fingerprint == right.fingerprint;
}

bool sameAccumulatorLayout(const PsiFormerParameterAccumulatorLayout& left,
                           const PsiFormerParameterAccumulatorLayout& right) noexcept
{
  return left.parameter_count == right.parameter_count &&
      left.complex_accumulator_count == right.complex_accumulator_count &&
      left.complex_accumulator_stride_fp64_elements ==
          right.complex_accumulator_stride_fp64_elements &&
      left.jvp_direction_offset_fp64_elements == right.jvp_direction_offset_fp64_elements &&
      left.jvp_direction_fp64_elements == right.jvp_direction_fp64_elements &&
      left.total_fp64_elements == right.total_fp64_elements &&
      left.total_bytes == right.total_bytes && left.fingerprint == right.fingerprint;
}

bool sameECPTileLayout(const PsiFormerECPTileLayout& left,
                       const PsiFormerECPTileLayout& right) noexcept
{
  return left.maximum_sample_count == right.maximum_sample_count &&
      left.maximum_segment_count == right.maximum_segment_count &&
      left.maximum_point_count == right.maximum_point_count &&
      left.maximum_channel_count == right.maximum_channel_count &&
      left.electron_count == right.electron_count &&
      left.maximum_evaluation_stamp_count == right.maximum_evaluation_stamp_count &&
      left.metadata_offset_bytes == right.metadata_offset_bytes &&
      left.metadata_bytes == right.metadata_bytes &&
      left.segment_offsets_offset_bytes == right.segment_offsets_offset_bytes &&
      left.segment_walker_ids_offset_bytes == right.segment_walker_ids_offset_bytes &&
      left.segment_electron_ids_offset_bytes == right.segment_electron_ids_offset_bytes &&
      left.segment_source_center_ids_offset_bytes ==
          right.segment_source_center_ids_offset_bytes &&
      left.segment_flags_offset_bytes == right.segment_flags_offset_bytes &&
      left.reference_positions_offset_bytes == right.reference_positions_offset_bytes &&
      left.replacement_positions_offset_bytes == right.replacement_positions_offset_bytes &&
      left.bare_weights_offset_bytes == right.bare_weights_offset_bytes &&
      left.complete_ratios_offset_bytes == right.complete_ratios_offset_bytes &&
      left.evaluation_stamps_offset_bytes == right.evaluation_stamps_offset_bytes &&
      left.coefficients_offset_bytes == right.coefficients_offset_bytes &&
      left.sample_weight_sums_offset_bytes == right.sample_weight_sums_offset_bytes &&
      left.point_status_offset_bytes == right.point_status_offset_bytes &&
      left.total_bytes == right.total_bytes && left.fingerprint == right.fingerprint;
}

bool sameHostStagingLayout(const PsiFormerPinnedHostStagingLayout& left,
                           const PsiFormerPinnedHostStagingLayout& right) noexcept
{
  return left.maximum_parameter_chunk_elements == right.maximum_parameter_chunk_elements &&
      left.maximum_jvp_output_complex_elements ==
          right.maximum_jvp_output_complex_elements &&
      left.complex_chunk_offset_bytes == right.complex_chunk_offset_bytes &&
      left.complex_chunk_bytes == right.complex_chunk_bytes &&
      left.jvp_output_offset_bytes == right.jvp_output_offset_bytes &&
      left.jvp_output_bytes == right.jvp_output_bytes &&
      left.total_bytes == right.total_bytes && left.fingerprint == right.fingerprint;
}

std::uint64_t parameterChunkFingerprint(const PsiFormerParameterChunkSchedule& schedule)
{
  std::uint64_t hash = FNV_OFFSET;
  mixInteger(hash, UINT64_C(1));
  mixInteger(hash, schedule.parameter_count);
  mixInteger(hash, schedule.maximum_chunk_elements);
  mixInteger(hash, schedule.chunks.size());
  for (const PsiFormerParameterChunkRecord& chunk : schedule.chunks)
  {
    mixInteger(hash, chunk.chunk_ordinal);
    mixInteger(hash, chunk.tensor_ordinal);
    mixInteger(hash, chunk.canonical_begin_fp64_elements);
    mixInteger(hash, chunk.count_fp64_elements);
  }
  return finishFingerprint(hash);
}

std::uint64_t derivativePlanFingerprint(const PsiFormerDerivativePlan& plan)
{
  std::uint64_t hash = FNV_OFFSET;
  mixInteger(hash, UINT64_C(1));
  mixInteger(hash, static_cast<std::uint8_t>(plan.boundary));
  for (const std::size_t value : {
           plan.electron_count, plan.nucleus_count, plan.determinant_count,
           plan.feature_width, plan.attention_head_count,
           plan.attention_block_count, plan.parameter_count,
           plan.parameter_tensor_count,
           plan.minimum_retained_tape_fp64_elements,
           plan.parameter_version, plan.block_size})
    mixInteger(hash, value);
  mixInteger(hash, plan.device_layout_fingerprint);
  mixInteger(hash, plan.kinetic_response_enabled ? 1 : 0);
  mixInteger(hash, plan.imaginary_gradient_response_enabled ? 1 : 0);
  mixInteger(hash, plan.reverse.fingerprint);
  mixInteger(hash, plan.accumulators.fingerprint);
  mixInteger(hash, plan.ecp_tile.fingerprint);
  mixInteger(hash, plan.device_arena.fingerprint);
  mixInteger(hash, plan.host_staging.fingerprint);
  mixInteger(hash, plan.parameter_chunks.fingerprint);
  return finishFingerprint(hash);
}

std::size_t expectedEmbeddingInputWidth(BoundaryCondition boundary,
                                        std::size_t nucleus_count)
{
  std::size_t pair_width = 0;
  switch (boundary)
  {
  case BoundaryCondition::OPEN:
    pair_width = 4;
    break;
  case BoundaryCondition::PERIODIC:
    pair_width = 7;
    break;
  default:
    throw std::invalid_argument("PsiFormer derivative planning received an invalid boundary condition");
  }
  return checkedAdd(
      checkedProduct(pair_width, nucleus_count,
                     "PsiFormer embedding input width overflowed"),
      1, "PsiFormer embedding input width overflowed");
}

std::size_t deviceEmbeddingInputWidth(const PsiFormerDeviceLayout& layout)
{
  const PsiFormerDeviceTensorDescriptor* embedding = nullptr;
  for (const PsiFormerDeviceTensorDescriptor& tensor : layout.tensors)
    if (tensor.role == ParameterRole::ELECTRON_EMBEDDING_WEIGHT)
    {
      if (embedding != nullptr)
        throw std::invalid_argument(
            "PsiFormer derivative planning requires exactly one electron-embedding tensor");
      embedding = &tensor;
    }
  if (embedding == nullptr)
    throw std::invalid_argument(
        "PsiFormer derivative planning requires exactly one electron-embedding tensor");
  if (embedding->rank != 2 || embedding->extents[0] == 0 ||
      embedding->extents[1] != layout.model_shape.feature_dimension)
    throw std::invalid_argument(
        "PsiFormer derivative planning requires a rank-2 electron-embedding "
        "tensor with the model feature width");

  const std::size_t tensor_elements = checkedProduct(
      embedding->extents[0], embedding->extents[1],
      "PsiFormer electron-embedding tensor extent overflowed");
  if (tensor_elements != embedding->size() ||
      embedding->extents[0] !=
          expectedEmbeddingInputWidth(layout.boundary, layout.model_shape.nuclei))
    throw std::invalid_argument(
        "PsiFormer electron-embedding tensor shape is incompatible with the model geometry");
  return embedding->extents[0];
}

void validateDeviceLayout(const PsiFormerDeviceLayout& layout)
{
  if (layout.fingerprint == 0)
    throw std::invalid_argument("PsiFormer derivative planning requires a nonzero device-layout fingerprint");
  if (layout.parameter_scalar_domain != ScalarDomain::REAL ||
      layout.compute_scalar_domain != ScalarDomain::REAL)
    throw std::invalid_argument("PsiFormer derivative planning supports only real FP64 parameters and reverse compute");
  if (layout.model_shape.electrons() == 0 || layout.model_shape.nuclei == 0 ||
      layout.model_shape.determinants == 0 || layout.model_shape.feature_dimension == 0 ||
      layout.model_shape.attention_heads == 0)
    throw std::invalid_argument("PsiFormer derivative planning requires positive model dimensions");

  std::size_t expected_begin = 0;
  for (const PsiFormerDeviceTensorDescriptor& tensor : layout.tensors)
  {
    if (tensor.begin != expected_begin || tensor.end <= tensor.begin ||
        tensor.end > layout.parameter_count)
      throw std::invalid_argument("PsiFormer derivative planning requires canonical nonempty tensor intervals");
    expected_begin = tensor.end;
  }
  if (expected_begin != layout.parameter_count)
    throw std::invalid_argument("PsiFormer derivative planning requires complete canonical parameter coverage");
  (void)deviceEmbeddingInputWidth(layout);
}

} // namespace

std::size_t minimumPsiFormerScoreTapeFP64Elements(
    const PsiFormerDeviceLayout& device_layout)
{
  validateDeviceLayout(device_layout);
  return minimumScoreTape(device_layout.model_shape.electrons(),
                          device_layout.model_shape.determinants,
                          deviceEmbeddingInputWidth(device_layout),
                          device_layout.model_shape.feature_dimension,
                          device_layout.model_shape.attention_heads,
                          device_layout.model_shape.attention_blocks);
}

PsiFormerDerivativePlan makePsiFormerDerivativePlan(
    const PsiFormerDeviceLayout& device_layout,
    std::size_t parameter_version,
    const PsiFormerDerivativePlanningRequirements& requirements)
{
  validateDeviceLayout(device_layout);
  if (requirements.block_size == 0)
    throw std::invalid_argument("PsiFormer derivative launch block size must be positive");
  if (requirements.complex_accumulator_count == 0 ||
      requirements.complex_accumulator_count > 3)
    throw std::invalid_argument("PsiFormer derivative planning requires one to three complex accumulators");
  if (requirements.maximum_parameter_chunk_elements == 0 ||
      requirements.maximum_parameter_chunk_elements > device_layout.parameter_count)
    throw std::invalid_argument("PsiFormer derivative chunk capacity must be within the parameter vector");
  if (requirements.retained_tape_fp64_elements == 0 ||
      requirements.adjoint_tape_fp64_elements == 0)
    throw std::invalid_argument("PsiFormer derivative planning requires retained and adjoint FP64 tapes");
  const bool ecp_disabled = requirements.maximum_ecp_sample_count == 0 &&
      requirements.maximum_ecp_segment_count == 0 &&
      requirements.maximum_ecp_point_count == 0 &&
      requirements.maximum_ecp_evaluation_stamp_count == 0;
  const bool ecp_complete = requirements.maximum_ecp_sample_count != 0 &&
      requirements.maximum_ecp_segment_count != 0 &&
      requirements.maximum_ecp_point_count != 0 &&
      requirements.maximum_ecp_evaluation_stamp_count != 0;
  if (!ecp_disabled && !ecp_complete)
    throw std::invalid_argument("PsiFormer ECP planning requires B, M, Q, and stamp capacities together");
  if (requirements.maximum_jvp_output_complex_elements > 1)
    throw std::invalid_argument("PsiFormer baseline derivative planning supports one JVP output at a time");
  if (requirements.enable_imaginary_gradient_response &&
      !requirements.enable_kinetic_response)
    throw std::invalid_argument("PsiFormer imaginary-gradient response requires the full kinetic trace tape");
  if (device_layout.boundary == BoundaryCondition::PERIODIC &&
      requirements.enable_kinetic_response)
    throw std::invalid_argument("PsiFormer periodic accelerator kinetic reverse is not implemented");

  const std::size_t minimum_score_tape =
      minimumPsiFormerScoreTapeFP64Elements(device_layout);
  const std::size_t minimum_retained_tape = requirements.enable_kinetic_response
      ? checkedProduct(minimum_score_tape,
                       rootSeedCapacity(device_layout.model_shape.electrons(), true),
                       "PsiFormer full-trace retained-tape extent overflowed")
      : minimum_score_tape;
  if (requirements.retained_tape_fp64_elements < minimum_retained_tape)
    throw std::invalid_argument("PsiFormer retained FP64 tape is smaller than the derived mode floor");

  PsiFormerDerivativePlan plan;
  plan.boundary = device_layout.boundary;
  plan.electron_count = device_layout.model_shape.electrons();
  plan.nucleus_count = device_layout.model_shape.nuclei;
  plan.determinant_count = device_layout.model_shape.determinants;
  plan.feature_width = device_layout.model_shape.feature_dimension;
  plan.attention_head_count = device_layout.model_shape.attention_heads;
  plan.attention_block_count = device_layout.model_shape.attention_blocks;
  plan.parameter_count = device_layout.parameter_count;
  plan.parameter_tensor_count = device_layout.tensors.size();
  plan.minimum_retained_tape_fp64_elements = minimum_retained_tape;
  plan.parameter_version = parameter_version;
  plan.block_size = requirements.block_size;
  plan.device_layout_fingerprint = device_layout.fingerprint;
  plan.kinetic_response_enabled = requirements.enable_kinetic_response;
  plan.imaginary_gradient_response_enabled =
      requirements.enable_imaginary_gradient_response;
  plan.reverse = makeReverseLayout(plan.parameter_count, plan.electron_count, requirements);
  plan.accumulators = makeAccumulatorLayout(plan.parameter_count,
                                            requirements.complex_accumulator_count);
  plan.ecp_tile = makeECPTileLayout(
      requirements.maximum_ecp_sample_count,
      requirements.maximum_ecp_segment_count,
      requirements.maximum_ecp_point_count,
      requirements.complex_accumulator_count, plan.electron_count,
      requirements.maximum_ecp_evaluation_stamp_count);
  plan.host_staging = makeHostStagingLayout(
      requirements.maximum_parameter_chunk_elements,
      requirements.maximum_jvp_output_complex_elements);

  plan.parameter_chunks.parameter_count = plan.parameter_count;
  plan.parameter_chunks.maximum_chunk_elements =
      requirements.maximum_parameter_chunk_elements;
  for (std::size_t tensor_ordinal = 0;
       tensor_ordinal < device_layout.tensors.size(); ++tensor_ordinal)
  {
    const PsiFormerDeviceTensorDescriptor& tensor = device_layout.tensors[tensor_ordinal];
    std::size_t begin = tensor.begin;
    while (begin < tensor.end)
    {
      const std::size_t count = std::min(requirements.maximum_parameter_chunk_elements,
                                         tensor.end - begin);
      plan.parameter_chunks.chunks.push_back({plan.parameter_chunks.chunks.size(),
                                               tensor_ordinal, begin, count});
      begin = checkedAdd(begin, count, "PsiFormer derivative chunk extent overflowed");
    }
  }
  plan.parameter_chunks.fingerprint = parameterChunkFingerprint(plan.parameter_chunks);

  plan.device_arena = makePsiFormerDeviceArenaLayout({
      {PsiFormerDeviceArenaRegion::REVERSE_WORKSPACE, plan.reverse.total_bytes,
       DEVICE_REGION_ALIGNMENT_BYTES},
      {PsiFormerDeviceArenaRegion::PARAMETER_ACCUMULATORS, plan.accumulators.total_bytes,
       DEVICE_REGION_ALIGNMENT_BYTES},
      {PsiFormerDeviceArenaRegion::ECP_TILE, plan.ecp_tile.total_bytes,
       DEVICE_REGION_ALIGNMENT_BYTES}});
  plan.fingerprint = derivativePlanFingerprint(plan);
  validatePsiFormerDerivativePlan(plan);
  return plan;
}

void validatePsiFormerDerivativePlan(const PsiFormerDerivativePlan& plan)
{
  if (plan.device_layout_fingerprint == 0 || plan.fingerprint == 0)
    throw std::invalid_argument("PsiFormer derivative plan fingerprints must be nonzero");
  if (plan.block_size == 0 || plan.parameter_count == 0 ||
      plan.parameter_tensor_count == 0)
    throw std::invalid_argument("PsiFormer derivative plan dimensions are incomplete");
  if (plan.electron_count == 0 || plan.nucleus_count == 0 ||
      plan.determinant_count == 0 || plan.feature_width == 0 ||
      plan.attention_head_count == 0)
    throw std::invalid_argument("PsiFormer derivative plan model dimensions must be positive");
  if (plan.imaginary_gradient_response_enabled && !plan.kinetic_response_enabled)
    throw std::invalid_argument("PsiFormer imaginary-gradient response lacks a kinetic trace tape");
  if (plan.boundary == BoundaryCondition::PERIODIC && plan.kinetic_response_enabled)
    throw std::invalid_argument("PsiFormer periodic accelerator kinetic reverse is not implemented");

  const std::size_t minimum_score_tape = minimumScoreTape(
      plan.electron_count, plan.determinant_count,
      expectedEmbeddingInputWidth(plan.boundary, plan.nucleus_count),
      plan.feature_width,
      plan.attention_head_count, plan.attention_block_count);
  const std::size_t expected_minimum_tape = plan.kinetic_response_enabled
      ? checkedProduct(minimum_score_tape,
                       rootSeedCapacity(plan.electron_count, true),
                       "PsiFormer full-trace retained-tape extent overflowed")
      : minimum_score_tape;
  if (plan.minimum_retained_tape_fp64_elements != expected_minimum_tape ||
      plan.reverse.retained_tape_fp64_elements < expected_minimum_tape)
    throw std::invalid_argument("PsiFormer retained FP64 tape is smaller than the derived mode floor");

  PsiFormerDerivativePlanningRequirements requirements;
  requirements.retained_tape_fp64_elements = plan.reverse.retained_tape_fp64_elements;
  requirements.adjoint_tape_fp64_elements = plan.reverse.adjoint_tape_fp64_elements;
  requirements.determinant_factor_fp64_elements = plan.reverse.determinant_factor_fp64_elements;
  requirements.determinant_scratch_fp64_elements = plan.reverse.determinant_scratch_fp64_elements;
  requirements.enable_kinetic_response = plan.kinetic_response_enabled;
  const PsiFormerDerivativeReverseLayout expected_reverse =
      makeReverseLayout(plan.parameter_count, plan.electron_count, requirements);
  if (!sameReverseLayout(plan.reverse, expected_reverse))
    throw std::invalid_argument("PsiFormer derivative reverse layout is noncanonical or aliases storage");

  const PsiFormerParameterAccumulatorLayout expected_accumulators =
      makeAccumulatorLayout(plan.parameter_count,
                            plan.accumulators.complex_accumulator_count);
  if (plan.accumulators.complex_accumulator_count == 0 ||
      plan.accumulators.complex_accumulator_count > 3 ||
      !sameAccumulatorLayout(plan.accumulators, expected_accumulators))
    throw std::invalid_argument("PsiFormer derivative accumulator layout is noncanonical or aliases storage");

  const PsiFormerECPTileLayout expected_ecp = makeECPTileLayout(
      plan.ecp_tile.maximum_sample_count, plan.ecp_tile.maximum_segment_count,
      plan.ecp_tile.maximum_point_count, plan.ecp_tile.maximum_channel_count,
      plan.ecp_tile.electron_count,
      plan.ecp_tile.maximum_evaluation_stamp_count);
  if (plan.ecp_tile.maximum_channel_count !=
          plan.accumulators.complex_accumulator_count ||
      plan.ecp_tile.electron_count != plan.electron_count ||
      !sameECPTileLayout(plan.ecp_tile, expected_ecp))
    throw std::invalid_argument("PsiFormer ECP tile layout is noncanonical or aliases storage");

  const PsiFormerPinnedHostStagingLayout expected_host = makeHostStagingLayout(
      plan.host_staging.maximum_parameter_chunk_elements,
      plan.host_staging.maximum_jvp_output_complex_elements);
  if (!sameHostStagingLayout(plan.host_staging, expected_host))
    throw std::invalid_argument("PsiFormer pinned-host staging layout is noncanonical or aliases storage");
  if (plan.host_staging.maximum_parameter_chunk_elements !=
      plan.parameter_chunks.maximum_chunk_elements)
    throw std::invalid_argument("PsiFormer host staging and chunk capacities disagree");

  if (plan.parameter_chunks.parameter_count != plan.parameter_count ||
      plan.parameter_chunks.maximum_chunk_elements == 0 ||
      plan.parameter_chunks.maximum_chunk_elements > plan.parameter_count)
    throw std::invalid_argument("PsiFormer derivative chunk schedule dimensions are invalid");
  std::size_t expected_begin = 0;
  std::size_t prior_tensor = 0;
  bool first_chunk = true;
  for (std::size_t ordinal = 0; ordinal < plan.parameter_chunks.chunks.size(); ++ordinal)
  {
    const PsiFormerParameterChunkRecord& chunk = plan.parameter_chunks.chunks[ordinal];
    if (chunk.chunk_ordinal != ordinal || chunk.tensor_ordinal >= plan.parameter_tensor_count ||
        chunk.canonical_begin_fp64_elements != expected_begin ||
        chunk.count_fp64_elements == 0 ||
        chunk.count_fp64_elements > plan.parameter_chunks.maximum_chunk_elements ||
        (!first_chunk && chunk.tensor_ordinal < prior_tensor))
      throw std::invalid_argument("PsiFormer derivative chunk schedule is noncanonical");
    expected_begin = checkedAdd(expected_begin, chunk.count_fp64_elements,
                                "PsiFormer derivative chunk coverage overflowed");
    prior_tensor = chunk.tensor_ordinal;
    first_chunk = false;
  }
  if (expected_begin != plan.parameter_count ||
      plan.parameter_chunks.fingerprint != parameterChunkFingerprint(plan.parameter_chunks))
    throw std::invalid_argument("PsiFormer derivative chunks do not cover the canonical parameters");

  const PsiFormerDeviceArenaLayout expected_arena = makePsiFormerDeviceArenaLayout({
      {PsiFormerDeviceArenaRegion::REVERSE_WORKSPACE, plan.reverse.total_bytes,
       DEVICE_REGION_ALIGNMENT_BYTES},
      {PsiFormerDeviceArenaRegion::PARAMETER_ACCUMULATORS, plan.accumulators.total_bytes,
       DEVICE_REGION_ALIGNMENT_BYTES},
      {PsiFormerDeviceArenaRegion::ECP_TILE, plan.ecp_tile.total_bytes,
       DEVICE_REGION_ALIGNMENT_BYTES}});
  if (plan.device_arena.slices != expected_arena.slices ||
      plan.device_arena.total_bytes != expected_arena.total_bytes ||
      plan.device_arena.fingerprint != expected_arena.fingerprint ||
      plan.device_arena.find(PsiFormerDeviceArenaRegion::HOST_STAGING) != nullptr)
    throw std::invalid_argument("PsiFormer derivative device arena is noncanonical or contains host staging");
  if (plan.fingerprint != derivativePlanFingerprint(plan))
    throw std::invalid_argument("PsiFormer derivative plan fingerprint does not match its contents");
}

PsiFormerDerivativeLaunchRecord makePsiFormerDerivativeLaunchRecord(
    const PsiFormerDerivativePlan& plan,
    PsiFormerDerivativeMode mode)
{
  validatePsiFormerDerivativePlan(plan);
  std::size_t root_offset = plan.reverse.root_seed_offset_fp64_elements;
  std::size_t root_count = 0;
  switch (mode)
  {
  case PsiFormerDerivativeMode::ORDINARY_SCORE:
    root_count = 1;
    break;
  case PsiFormerDerivativeMode::KINETIC_RESPONSE:
    if (!plan.kinetic_response_enabled || plan.boundary == BoundaryCondition::PERIODIC)
      throw std::invalid_argument("PsiFormer kinetic reverse mode is not enabled for this plan");
    root_count = rootSeedCapacity(plan.electron_count, true);
    break;
  case PsiFormerDerivativeMode::IMAGINARY_GRADIENT_RESPONSE:
    if (!plan.imaginary_gradient_response_enabled ||
        plan.boundary == BoundaryCondition::PERIODIC)
      throw std::invalid_argument("PsiFormer imaginary-gradient reverse mode is not enabled for this plan");
    root_offset = checkedAdd(root_offset, 1,
                             "PsiFormer imaginary-gradient root offset overflowed");
    root_count = checkedProduct(3, plan.electron_count,
                                "PsiFormer imaginary-gradient root extent overflowed");
    break;
  default:
    throw std::invalid_argument("PsiFormer derivative launch mode is invalid");
  }
  if (checkedAdd(root_offset, root_count,
                 "PsiFormer derivative root interval overflowed") >
      checkedAdd(plan.reverse.root_seed_offset_fp64_elements,
                 plan.reverse.root_seed_capacity_fp64_elements,
                 "PsiFormer derivative root capacity overflowed"))
    throw std::logic_error("PsiFormer derivative root interval exceeds prepared storage");

  PsiFormerDerivativeLaunchRecord record;
  record.plan_fingerprint = plan.fingerprint;
  record.device_layout_fingerprint = plan.device_layout_fingerprint;
  record.device_arena_fingerprint = plan.device_arena.fingerprint;
  record.parameter_version = plan.parameter_version;
  record.mode = mode;
  record.electron_count = plan.electron_count;
  record.nucleus_count = plan.nucleus_count;
  record.determinant_count = plan.determinant_count;
  record.feature_width = plan.feature_width;
  record.attention_head_count = plan.attention_head_count;
  record.attention_block_count = plan.attention_block_count;
  record.parameter_count = plan.parameter_count;
  record.canonical_tensor_count = plan.parameter_tensor_count;
  record.root_seed_offset_fp64_elements = root_offset;
  record.root_seed_count_fp64_elements = root_count;
  record.response_offset_fp64_elements = plan.reverse.response_offset_fp64_elements;
  record.response_count_fp64_elements = plan.parameter_count;
  record.accumulator_stride_fp64_elements =
      plan.accumulators.complex_accumulator_stride_fp64_elements;
  record.complex_accumulator_count = plan.accumulators.complex_accumulator_count;
  record.launch = makePsiFormerLinearLaunch(plan.parameter_count, plan.block_size);

  std::uint64_t hash = FNV_OFFSET;
  mixInteger(hash, UINT64_C(1));
  mixInteger(hash, record.plan_fingerprint);
  mixInteger(hash, record.device_layout_fingerprint);
  mixInteger(hash, record.device_arena_fingerprint);
  mixInteger(hash, record.parameter_version);
  mixInteger(hash, static_cast<std::uint8_t>(record.mode));
  for (const std::size_t value : {
           record.electron_count, record.nucleus_count, record.determinant_count,
           record.feature_width, record.attention_head_count,
           record.attention_block_count, record.parameter_count,
           record.canonical_tensor_begin, record.canonical_tensor_count,
           record.root_seed_offset_fp64_elements,
           record.root_seed_count_fp64_elements,
           record.response_offset_fp64_elements,
           record.response_count_fp64_elements,
           record.accumulator_offset_fp64_elements,
           record.accumulator_stride_fp64_elements,
           record.complex_accumulator_count, record.launch.item_count,
           record.launch.block_size, record.launch.block_count,
           record.launch.tail_count})
    mixInteger(hash, value);
  record.fingerprint = finishFingerprint(hash);
  return record;
}

PsiFormerECPTileLaunchRecord makePsiFormerECPTileLaunchRecord(
    const PsiFormerDerivativePlan& plan,
    const PsiFormerECPTileMetadata& metadata,
    std::size_t channel_count)
{
  validatePsiFormerDerivativePlan(plan);
  return makePsiFormerECPTileLaunchRecordFromPrevalidatedPlan(
      plan, metadata, channel_count);
}

PsiFormerECPTileLaunchRecord
makePsiFormerECPTileLaunchRecordFromPrevalidatedPlan(
    const PsiFormerDerivativePlan& plan,
    const PsiFormerECPTileMetadata& metadata,
    std::size_t channel_count)
{
  if (metadata.grid_fingerprint == 0 || metadata.tile_fingerprint == 0)
    throw std::invalid_argument("PsiFormer ECP launch requires nonzero grid and tile fingerprints");
  if (metadata.parameter_version != plan.parameter_version)
    throw std::invalid_argument("PsiFormer ECP launch parameter version is stale");
  if (metadata.sample_count > plan.ecp_tile.maximum_sample_count ||
      metadata.segment_count > plan.ecp_tile.maximum_segment_count ||
      metadata.point_count > plan.ecp_tile.maximum_point_count ||
      metadata.evaluation_stamp_count >
          plan.ecp_tile.maximum_evaluation_stamp_count ||
      channel_count == 0 || channel_count > plan.ecp_tile.maximum_channel_count ||
      (metadata.point_count != 0 &&
       (metadata.sample_count == 0 || metadata.segment_count == 0 ||
        metadata.evaluation_stamp_count == 0)) ||
      metadata.segment_count > metadata.point_count)
    throw std::invalid_argument("PsiFormer ECP launch exceeds the prepared tile capacities");
  constexpr std::uint32_t required_flags = ECP_ORDINARY_LOCALITY |
      ECP_USES_VIRTUAL_PARTICLES | ECP_SCALAR_RELATIVISTIC;
  constexpr std::uint32_t known_flags = required_flags |
      ECP_DENSE_DERIVATIVE_FALLBACK;
  if ((metadata.flags & required_flags) != required_flags ||
      (metadata.flags & ECP_DENSE_DERIVATIVE_FALLBACK) != 0 ||
      (metadata.flags & ~known_flags) != 0)
    throw std::invalid_argument("PsiFormer ECP launch flags do not describe ordinary scalar-relativistic locality");
  if (metadata.status != 0)
    throw std::invalid_argument("PsiFormer ECP launch status must be clear before submission");
  (void)checkedAdd(metadata.flat_point_begin, metadata.point_count,
                   "PsiFormer ECP flat point interval overflowed");

  PsiFormerECPTileLaunchRecord record;
  record.device_layout_fingerprint = plan.device_layout_fingerprint;
  record.device_arena_fingerprint = plan.device_arena.fingerprint;
  record.metadata = metadata;
  record.channel_count = channel_count;
  record.metadata_offset_bytes = plan.ecp_tile.metadata_offset_bytes;
  record.segment_offsets_offset_bytes = plan.ecp_tile.segment_offsets_offset_bytes;
  record.segment_walker_ids_offset_bytes = plan.ecp_tile.segment_walker_ids_offset_bytes;
  record.segment_electron_ids_offset_bytes = plan.ecp_tile.segment_electron_ids_offset_bytes;
  record.segment_source_center_ids_offset_bytes =
      plan.ecp_tile.segment_source_center_ids_offset_bytes;
  record.segment_flags_offset_bytes = plan.ecp_tile.segment_flags_offset_bytes;
  record.reference_positions_offset_bytes = plan.ecp_tile.reference_positions_offset_bytes;
  record.replacement_positions_offset_bytes = plan.ecp_tile.replacement_positions_offset_bytes;
  record.bare_weights_offset_bytes = plan.ecp_tile.bare_weights_offset_bytes;
  record.complete_ratios_offset_bytes = plan.ecp_tile.complete_ratios_offset_bytes;
  record.evaluation_stamps_offset_bytes = plan.ecp_tile.evaluation_stamps_offset_bytes;
  record.coefficients_offset_bytes = plan.ecp_tile.coefficients_offset_bytes;
  record.sample_weight_sums_offset_bytes = plan.ecp_tile.sample_weight_sums_offset_bytes;
  record.point_status_offset_bytes = plan.ecp_tile.point_status_offset_bytes;
  record.launch = makePsiFormerLinearLaunch(metadata.point_count, plan.block_size);

  std::uint64_t hash = FNV_OFFSET;
  mixInteger(hash, UINT64_C(1));
  mixInteger(hash, record.device_layout_fingerprint);
  mixInteger(hash, record.device_arena_fingerprint);
  mixInteger(hash, record.metadata.grid_fingerprint);
  mixInteger(hash, record.metadata.tile_fingerprint);
  for (const std::size_t value : {
           record.metadata.parameter_version, record.metadata.tile_ordinal,
           record.metadata.flat_point_begin, record.metadata.sample_count,
           record.metadata.segment_count, record.metadata.point_count,
           record.metadata.evaluation_stamp_count, record.channel_count,
           record.segment_offset_stride_elements,
           record.position_component_count, record.complex_component_count,
           record.metadata_offset_bytes,
           record.segment_offsets_offset_bytes,
           record.segment_walker_ids_offset_bytes,
           record.segment_electron_ids_offset_bytes,
           record.segment_source_center_ids_offset_bytes,
           record.segment_flags_offset_bytes,
           record.reference_positions_offset_bytes,
           record.replacement_positions_offset_bytes,
           record.bare_weights_offset_bytes,
           record.complete_ratios_offset_bytes,
           record.evaluation_stamps_offset_bytes,
           record.coefficients_offset_bytes,
           record.sample_weight_sums_offset_bytes,
           record.point_status_offset_bytes, record.launch.item_count,
           record.launch.block_size, record.launch.block_count,
           record.launch.tail_count})
    mixInteger(hash, value);
  mixInteger(hash, record.metadata.flags);
  mixInteger(hash, record.metadata.status);
  record.fingerprint = finishFingerprint(hash);
  return record;
}

} // namespace qmcplusplus::psiformer
