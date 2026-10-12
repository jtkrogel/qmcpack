//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDerivativePlanning.h
 * @brief Pointer-free FP64 planning records for PsiFormer parameter derivatives.
 *
 * This is a planning-only surface.  It allocates no memory and launches no reverse
 * kernel.  Every offset names either FP64 elements relative to one device region or
 * bytes relative to one heterogeneous ECP/host-staging allocation, as stated by the
 * field name.  Pinned host staging is deliberately not a device-arena region.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DERIVATIVE_PLANNING_H
#define QMCPLUSPLUS_PSIFORMER_DERIVATIVE_PLANNING_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorPlanning.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorSchedule.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace qmcplusplus::psiformer
{

/// Identify one sequential reverse pass over the retained FP64 tape.
enum class PsiFormerDerivativeMode : std::uint8_t
{
  ORDINARY_SCORE,
  KINETIC_RESPONSE,
  IMAGINARY_GRADIENT_RESPONSE
};

/** Caller-supplied bounded capacities for one derivative plan.
 *
 * Tape and determinant counts are supplied by the eventual A6/A7 executor because
 * this layer does not yet define the exact numerical reverse tape.  The planner
 * checks a derived ordinary-score or conservative full-trace baseline floor, but
 * the supplied extent may be larger.  Counts describe one configuration only and
 * must never be multiplied by a walker batch size.
 */
struct PsiFormerDerivativePlanningRequirements
{
  std::size_t retained_tape_fp64_elements       = 0;
  std::size_t adjoint_tape_fp64_elements        = 0;
  std::size_t determinant_factor_fp64_elements  = 0;
  std::size_t determinant_scratch_fp64_elements = 0;
  std::size_t complex_accumulator_count         = 0;
  std::size_t maximum_parameter_chunk_elements  = 0;
  std::size_t maximum_ecp_sample_count          = 0;
  std::size_t maximum_ecp_segment_count         = 0;
  std::size_t maximum_ecp_point_count           = 0;
  std::size_t maximum_ecp_evaluation_stamp_count = 0;
  std::size_t maximum_jvp_output_complex_elements = 0;
  std::size_t block_size = 0;
  bool enable_kinetic_response           = false;
  bool enable_imaginary_gradient_response = false;
};

/// Required semantic flags copied from the current ordinary-locality transaction.
enum PsiFormerECPTileFlag : std::uint32_t
{
  ECP_ORDINARY_LOCALITY = UINT32_C(1) << 0,
  ECP_USES_VIRTUAL_PARTICLES = UINT32_C(1) << 1,
  ECP_SCALAR_RELATIVISTIC = UINT32_C(1) << 2,
  ECP_DENSE_DERIVATIVE_FALLBACK = UINT32_C(1) << 3
};

/** Fixed-layout header copied with one live ECP tile.
 *
 * The device producer must leave ``status`` zero on success.  The bounded status
 * array in the tile holds one additional uint32 value per point.
 */
struct PsiFormerECPTileMetadata
{
  std::uint64_t grid_fingerprint = 0;
  std::uint64_t tile_fingerprint = 0;
  std::size_t parameter_version = 0;
  std::size_t tile_ordinal = 0;
  std::size_t flat_point_begin = 0;
  std::size_t sample_count = 0;
  std::size_t segment_count = 0;
  std::size_t point_count = 0;
  std::size_t evaluation_stamp_count = 0;
  std::uint32_t flags = 0;
  std::uint32_t status = 0;
};

/// Pointer-free translation of one opaque host evaluation stamp.
struct PsiFormerECPEvaluationStampRecord
{
  std::uint64_t producer_fingerprint = 0;
  std::uint64_t parameter_version = 0;
};

/** Canonical FP64 sublayout of REVERSE_WORKSPACE.
 *
 * All offsets and extents are measured in FP64 elements, not bytes.  The response is
 * exactly one real canonical parameter vector and is reused between reverse modes.
 */
struct PsiFormerDerivativeReverseLayout
{
  std::size_t retained_tape_offset_fp64_elements = 0;
  std::size_t retained_tape_fp64_elements        = 0;
  std::size_t adjoint_tape_offset_fp64_elements  = 0;
  std::size_t adjoint_tape_fp64_elements         = 0;
  std::size_t determinant_factor_offset_fp64_elements = 0;
  std::size_t determinant_factor_fp64_elements        = 0;
  std::size_t determinant_scratch_offset_fp64_elements = 0;
  std::size_t determinant_scratch_fp64_elements        = 0;
  std::size_t root_seed_offset_fp64_elements     = 0;
  std::size_t root_seed_capacity_fp64_elements   = 0;
  std::size_t response_offset_fp64_elements      = 0;
  std::size_t response_fp64_elements             = 0;
  std::size_t total_fp64_elements                = 0;
  std::size_t total_bytes                        = 0;
  std::uint64_t fingerprint                      = 0;
};

/** Canonical FP64 sublayout of PARAMETER_ACCUMULATORS.
 *
 * Each complex value occupies two adjacent FP64 elements.  The first P real elements
 * may be reused as one JVP direction when no VJP accumulator is live.
 */
struct PsiFormerParameterAccumulatorLayout
{
  std::size_t parameter_count                    = 0;
  std::size_t complex_accumulator_count          = 0;
  std::size_t complex_accumulator_stride_fp64_elements = 0;
  std::size_t jvp_direction_offset_fp64_elements = 0;
  std::size_t jvp_direction_fp64_elements        = 0;
  std::size_t total_fp64_elements                = 0;
  std::size_t total_bytes                        = 0;
  std::uint64_t fingerprint                      = 0;
};

/** Heterogeneous byte layout of one bounded flattened ECP tile.
 *
 * This record intentionally has no parameter-count field.  Segment metadata and
 * sparse replacement tuples are O(reference_count + point_count), never Q*P.
 */
struct PsiFormerECPTileLayout
{
  std::size_t maximum_sample_count = 0;
  std::size_t maximum_segment_count = 0;
  std::size_t maximum_point_count = 0;
  std::size_t maximum_channel_count = 0;
  std::size_t electron_count = 0;
  std::size_t maximum_evaluation_stamp_count = 0;
  std::size_t metadata_offset_bytes   = 0;
  std::size_t metadata_bytes          = 0;
  std::size_t segment_offsets_offset_bytes             = 0;
  std::size_t segment_walker_ids_offset_bytes           = 0;
  std::size_t segment_electron_ids_offset_bytes         = 0;
  std::size_t segment_source_center_ids_offset_bytes    = 0;
  std::size_t segment_flags_offset_bytes                = 0;
  std::size_t reference_positions_offset_bytes          = 0;
  std::size_t replacement_positions_offset_bytes        = 0;
  std::size_t bare_weights_offset_bytes                 = 0;
  std::size_t complete_ratios_offset_bytes              = 0;
  std::size_t evaluation_stamps_offset_bytes            = 0;
  std::size_t coefficients_offset_bytes                 = 0;
  std::size_t sample_weight_sums_offset_bytes           = 0;
  std::size_t point_status_offset_bytes                 = 0;
  std::size_t total_bytes = 0;
  std::uint64_t fingerprint = 0;
};

/** Separate allocation contract for pinned, host-accessible publication staging.
 *
 * One complex chunk is reused in chunk-major/channel-minor order.  Complex values
 * are represented by two FP64 elements.  Chunk and JVP views deliberately alias at
 * byte zero because only one is live; total bytes are the larger view, not their
 * sum.  This allocation is not included in the derivative device-arena fingerprint
 * or byte total.
 */
struct PsiFormerPinnedHostStagingLayout
{
  std::size_t maximum_parameter_chunk_elements = 0;
  std::size_t maximum_jvp_output_complex_elements = 0;
  std::size_t complex_chunk_offset_bytes = 0;
  std::size_t complex_chunk_bytes        = 0;
  std::size_t jvp_output_offset_bytes    = 0;
  std::size_t jvp_output_bytes           = 0;
  std::size_t total_bytes                = 0;
  std::uint64_t fingerprint              = 0;
};

/// One tensor-bounded canonical publication chunk.
struct PsiFormerParameterChunkRecord
{
  std::size_t chunk_ordinal = 0;
  std::size_t tensor_ordinal = 0;
  std::size_t canonical_begin_fp64_elements = 0;
  std::size_t count_fp64_elements = 0;

  std::size_t canonicalEndFp64Elements() const noexcept
  {
    return canonical_begin_fp64_elements + count_fp64_elements;
  }
};

/// Deterministic tensor-bounded chunk schedule used by synchronized host publication.
struct PsiFormerParameterChunkSchedule
{
  std::size_t parameter_count = 0;
  std::size_t maximum_chunk_elements = 0;
  std::vector<PsiFormerParameterChunkRecord> chunks;
  std::uint64_t fingerprint = 0;
};

/** Complete, independently valid FP64 derivative planning contract.
 *
 * ``device_arena`` contains exactly REVERSE_WORKSPACE, PARAMETER_ACCUMULATORS,
 * and ECP_TILE.  ``host_staging`` is a distinct pinned-host allocation contract.
 */
struct PsiFormerDerivativePlan
{
  BoundaryCondition boundary = BoundaryCondition::OPEN;
  std::size_t electron_count = 0;
  std::size_t nucleus_count = 0;
  std::size_t determinant_count = 0;
  std::size_t feature_width = 0;
  std::size_t attention_head_count = 0;
  std::size_t attention_block_count = 0;
  std::size_t parameter_count = 0;
  std::size_t parameter_tensor_count = 0;
  /// Checked baseline floor; the executor-supplied exact tape may be larger.
  std::size_t minimum_retained_tape_fp64_elements = 0;
  std::size_t parameter_version = 0;
  std::size_t block_size = 0;
  std::uint64_t device_layout_fingerprint = 0;
  bool kinetic_response_enabled = false;
  bool imaginary_gradient_response_enabled = false;
  PsiFormerDerivativeReverseLayout reverse;
  PsiFormerParameterAccumulatorLayout accumulators;
  PsiFormerECPTileLayout ecp_tile;
  PsiFormerDeviceArenaLayout device_arena;
  PsiFormerPinnedHostStagingLayout host_staging;
  PsiFormerParameterChunkSchedule parameter_chunks;
  std::uint64_t fingerprint = 0;
};

/// Pointer-free full-response launch record.  All region offsets are FP64 elements.
struct PsiFormerDerivativeLaunchRecord
{
  std::uint64_t plan_fingerprint = 0;
  std::uint64_t device_layout_fingerprint = 0;
  std::uint64_t device_arena_fingerprint = 0;
  std::size_t parameter_version = 0;
  PsiFormerDerivativeMode mode = PsiFormerDerivativeMode::ORDINARY_SCORE;
  std::size_t electron_count = 0;
  std::size_t nucleus_count = 0;
  std::size_t determinant_count = 0;
  std::size_t feature_width = 0;
  std::size_t attention_head_count = 0;
  std::size_t attention_block_count = 0;
  std::size_t parameter_count = 0;
  std::size_t canonical_tensor_begin = 0;
  std::size_t canonical_tensor_count = 0;
  std::size_t root_seed_offset_fp64_elements = 0;
  std::size_t root_seed_count_fp64_elements = 0;
  std::size_t response_offset_fp64_elements = 0;
  std::size_t response_count_fp64_elements = 0;
  std::size_t accumulator_offset_fp64_elements = 0;
  std::size_t accumulator_stride_fp64_elements = 0;
  std::size_t complex_accumulator_count = 0;
  PsiFormerLinearLaunch launch;
  std::uint64_t fingerprint = 0;
};

/** Pointer-free launch record for one live flattened ECP tile.
 *
 * ECP offsets are bytes relative to ECP_TILE.  The record has no P extent and cannot
 * describe Q*P staging.
 */
struct PsiFormerECPTileLaunchRecord
{
  std::uint64_t device_layout_fingerprint = 0;
  std::uint64_t device_arena_fingerprint = 0;
  PsiFormerECPTileMetadata metadata;
  std::size_t channel_count = 0;
  std::size_t segment_offset_stride_elements = 1;
  std::size_t position_component_count = 3;
  std::size_t complex_component_count = 2;
  std::size_t metadata_offset_bytes = 0;
  std::size_t segment_offsets_offset_bytes = 0;
  std::size_t segment_walker_ids_offset_bytes = 0;
  std::size_t segment_electron_ids_offset_bytes = 0;
  std::size_t segment_source_center_ids_offset_bytes = 0;
  std::size_t segment_flags_offset_bytes = 0;
  std::size_t reference_positions_offset_bytes = 0;
  std::size_t replacement_positions_offset_bytes = 0;
  std::size_t bare_weights_offset_bytes = 0;
  std::size_t complete_ratios_offset_bytes = 0;
  std::size_t evaluation_stamps_offset_bytes = 0;
  std::size_t coefficients_offset_bytes = 0;
  std::size_t sample_weight_sums_offset_bytes = 0;
  std::size_t point_status_offset_bytes = 0;
  PsiFormerLinearLaunch launch;
  std::uint64_t fingerprint = 0;
};

static_assert(std::is_standard_layout_v<PsiFormerDerivativePlanningRequirements>);
static_assert(std::is_trivially_copyable_v<PsiFormerDerivativePlanningRequirements>);
static_assert(std::is_standard_layout_v<PsiFormerECPTileMetadata>);
static_assert(std::is_trivially_copyable_v<PsiFormerECPTileMetadata>);
static_assert(std::is_standard_layout_v<PsiFormerECPEvaluationStampRecord>);
static_assert(std::is_trivially_copyable_v<PsiFormerECPEvaluationStampRecord>);
static_assert(std::is_standard_layout_v<PsiFormerDerivativeReverseLayout>);
static_assert(std::is_trivially_copyable_v<PsiFormerDerivativeReverseLayout>);
static_assert(std::is_standard_layout_v<PsiFormerParameterAccumulatorLayout>);
static_assert(std::is_trivially_copyable_v<PsiFormerParameterAccumulatorLayout>);
static_assert(std::is_standard_layout_v<PsiFormerECPTileLayout>);
static_assert(std::is_trivially_copyable_v<PsiFormerECPTileLayout>);
static_assert(std::is_standard_layout_v<PsiFormerPinnedHostStagingLayout>);
static_assert(std::is_trivially_copyable_v<PsiFormerPinnedHostStagingLayout>);
static_assert(std::is_standard_layout_v<PsiFormerParameterChunkRecord>);
static_assert(std::is_trivially_copyable_v<PsiFormerParameterChunkRecord>);
static_assert(std::is_standard_layout_v<PsiFormerDerivativeLaunchRecord>);
static_assert(std::is_trivially_copyable_v<PsiFormerDerivativeLaunchRecord>);
static_assert(std::is_standard_layout_v<PsiFormerECPTileLaunchRecord>);
static_assert(std::is_trivially_copyable_v<PsiFormerECPTileLaunchRecord>);

/// Construct one checked FP64 derivative plan for a canonical device layout.
PsiFormerDerivativePlan makePsiFormerDerivativePlan(
    const PsiFormerDeviceLayout& device_layout,
    std::size_t parameter_version,
    const PsiFormerDerivativePlanningRequirements& requirements);

/// Return the checked retained-value-tape floor for the ordinary-score baseline.
std::size_t minimumPsiFormerScoreTapeFP64Elements(
    const PsiFormerDeviceLayout& device_layout);

/// Revalidate canonical offsets, exact high-water marks, and region separation.
void validatePsiFormerDerivativePlan(const PsiFormerDerivativePlan& plan);

/// Construct one checked full-response reverse launch record.
PsiFormerDerivativeLaunchRecord makePsiFormerDerivativeLaunchRecord(
    const PsiFormerDerivativePlan& plan,
    PsiFormerDerivativeMode mode);

/// Construct one checked flattened ECP tile launch record, including zero work.
PsiFormerECPTileLaunchRecord makePsiFormerECPTileLaunchRecord(
    const PsiFormerDerivativePlan& plan,
    const PsiFormerECPTileMetadata& metadata,
    std::size_t channel_count);

/** Construct an ECP launch record after the owning preparation boundary validated plan.
 *
 * This path performs only bounded POD checks and arithmetic and allocates no memory.
 * Passing a plan not previously accepted by validatePsiFormerDerivativePlan is a
 * caller contract violation.
 */
PsiFormerECPTileLaunchRecord
makePsiFormerECPTileLaunchRecordFromPrevalidatedPlan(
    const PsiFormerDerivativePlan& plan,
    const PsiFormerECPTileMetadata& metadata,
    std::size_t channel_count);

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_DERIVATIVE_PLANNING_H
