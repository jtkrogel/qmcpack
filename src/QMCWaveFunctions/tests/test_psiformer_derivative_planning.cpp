//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_derivative_planning.cpp
 * @brief Deterministic CPU tests for FP64 PsiFormer derivative planning.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerDerivativePlanning.h"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer
{
namespace
{

PsiFormerDeviceLayout makeDeviceLayout(std::size_t feature_width = 8,
                                       BoundaryCondition boundary = BoundaryCondition::OPEN,
                                       std::uint64_t fingerprint = 701)
{
  PsiFormerDeviceLayout layout;
  layout.model_shape = {/*spin_up_electrons=*/2,
                        /*spin_down_electrons=*/1,
                        /*nuclei=*/2,
                        /*determinants=*/3,
                        /*feature_dimension=*/feature_width,
                        /*attention_heads=*/2,
                        /*attention_blocks=*/2};
  layout.boundary = boundary;
  layout.geometry_feature_policy = boundary == BoundaryCondition::OPEN
      ? GeometryFeaturePolicy::OPEN_EUCLIDEAN_V1
      : GeometryFeaturePolicy::PERIODIC_TORUS_V1;
  const std::size_t embedding_input_width =
      (boundary == BoundaryCondition::PERIODIC ? 7 : 4) * layout.model_shape.nuclei + 1;
  const std::size_t embedding_end = embedding_input_width * feature_width;
  const std::size_t hidden_bias_end = embedding_end + feature_width;
  layout.parameter_count = hidden_bias_end + feature_width;
  layout.tensors = {
      {ParameterRole::ELECTRON_EMBEDDING_WEIGHT, NO_ATTENTION_BLOCK, 2,
       {embedding_input_width, feature_width}, 0, embedding_end},
      {ParameterRole::UPDATE_HIDDEN_BIAS, 0, 1,
       {feature_width, 0}, embedding_end, hidden_bias_end},
      {ParameterRole::UPDATE_OUTPUT_BIAS, 0, 1,
       {feature_width, 0}, hidden_bias_end, layout.parameter_count}};
  layout.fingerprint = fingerprint;
  return layout;
}

PsiFormerDerivativePlanningRequirements makeRequirements(
    const PsiFormerDeviceLayout& layout)
{
  PsiFormerDerivativePlanningRequirements requirements;
  requirements.retained_tape_fp64_elements =
      minimumPsiFormerScoreTapeFP64Elements(layout);
  requirements.adjoint_tape_fp64_elements = 233;
  requirements.determinant_factor_fp64_elements = 51;
  requirements.determinant_scratch_fp64_elements = 29;
  requirements.complex_accumulator_count = 3;
  requirements.maximum_parameter_chunk_elements = 8;
  requirements.maximum_ecp_sample_count = 4;
  requirements.maximum_ecp_segment_count = 6;
  requirements.maximum_ecp_point_count = 13;
  requirements.maximum_ecp_evaluation_stamp_count = 5;
  requirements.maximum_jvp_output_complex_elements = 1;
  requirements.block_size = 8;
  return requirements;
}

constexpr std::uint32_t REQUIRED_ECP_FLAGS = ECP_ORDINARY_LOCALITY |
    ECP_USES_VIRTUAL_PARTICLES | ECP_SCALAR_RELATIVISTIC;

PsiFormerECPTileMetadata makeTileMetadata()
{
  PsiFormerECPTileMetadata metadata;
  metadata.grid_fingerprint = 901;
  metadata.tile_fingerprint = 902;
  metadata.parameter_version = 11;
  metadata.tile_ordinal = 2;
  metadata.flat_point_begin = 16;
  metadata.sample_count = 4;
  metadata.segment_count = 6;
  metadata.point_count = 13;
  metadata.evaluation_stamp_count = 3;
  metadata.flags = REQUIRED_ECP_FLAGS;
  return metadata;
}

} // namespace

TEST_CASE("PsiFormer derivative plan has exact separated FP64 storage",
          "[wavefunction][psiformer][derivative_planning]")
{
  const PsiFormerDeviceLayout device = makeDeviceLayout();
  const PsiFormerDerivativePlanningRequirements requirements =
      makeRequirements(device);
  const PsiFormerDerivativePlan plan =
      makePsiFormerDerivativePlan(device, /*parameter_version=*/11, requirements);

  CHECK(plan.minimum_retained_tape_fp64_elements == 498);
  CHECK(plan.reverse.retained_tape_fp64_elements == 498);
  CHECK(plan.reverse.response_fp64_elements == device.parameter_count);
  CHECK(plan.accumulators.parameter_count == device.parameter_count);
  CHECK(plan.accumulators.complex_accumulator_count == 3);
  CHECK(plan.accumulators.jvp_direction_fp64_elements == device.parameter_count);
  CHECK(plan.accumulators.complex_accumulator_stride_fp64_elements >=
        2 * device.parameter_count);
  CHECK(plan.reverse.total_bytes == plan.reverse.total_fp64_elements * sizeof(double));
  CHECK(plan.accumulators.total_bytes ==
        plan.accumulators.total_fp64_elements * sizeof(double));

  REQUIRE(plan.device_arena.slices.size() == 3);
  REQUIRE(plan.device_arena.find(PsiFormerDeviceArenaRegion::REVERSE_WORKSPACE));
  REQUIRE(plan.device_arena.find(PsiFormerDeviceArenaRegion::PARAMETER_ACCUMULATORS));
  REQUIRE(plan.device_arena.find(PsiFormerDeviceArenaRegion::ECP_TILE));
  CHECK(plan.device_arena.find(PsiFormerDeviceArenaRegion::HOST_STAGING) == nullptr);
  CHECK(plan.device_arena.find(PsiFormerDeviceArenaRegion::REVERSE_WORKSPACE)->bytes ==
        plan.reverse.total_bytes);
  CHECK(plan.device_arena.find(PsiFormerDeviceArenaRegion::PARAMETER_ACCUMULATORS)->bytes ==
        plan.accumulators.total_bytes);
  CHECK(plan.device_arena.find(PsiFormerDeviceArenaRegion::ECP_TILE)->bytes ==
        plan.ecp_tile.total_bytes);
  CHECK(plan.host_staging.total_bytes == plan.host_staging.complex_chunk_bytes);
  CHECK(plan.host_staging.complex_chunk_offset_bytes == 0);
  CHECK(plan.host_staging.jvp_output_offset_bytes == 0);
  CHECK(plan.host_staging.complex_chunk_bytes == 2 * 8 * sizeof(double));
  CHECK_NOTHROW(validatePsiFormerDerivativePlan(plan));

  PsiFormerDerivativePlan aliased_reverse = plan;
  aliased_reverse.reverse.response_offset_fp64_elements =
      aliased_reverse.reverse.root_seed_offset_fp64_elements;
  CHECK_THROWS_WITH(validatePsiFormerDerivativePlan(aliased_reverse),
                    Catch::Matchers::ContainsSubstring("aliases"));

  PsiFormerDerivativePlan aliased_host = plan;
  aliased_host.host_staging.jvp_output_offset_bytes = sizeof(double);
  CHECK_THROWS_WITH(validatePsiFormerDerivativePlan(aliased_host),
                    Catch::Matchers::ContainsSubstring("host staging"));
}

TEST_CASE("PsiFormer derivative chunks respect tensors and score launch is full P",
          "[wavefunction][psiformer][derivative_planning]")
{
  const PsiFormerDeviceLayout device = makeDeviceLayout();
  const PsiFormerDerivativePlan plan =
      makePsiFormerDerivativePlan(device, 11, makeRequirements(device));

  REQUIRE(plan.parameter_chunks.chunks.size() == 11);
  CHECK(plan.parameter_chunks.chunks[0].canonical_begin_fp64_elements == 0);
  CHECK(plan.parameter_chunks.chunks[0].count_fp64_elements == 8);
  CHECK(plan.parameter_chunks.chunks[8].tensor_ordinal == 0);
  CHECK(plan.parameter_chunks.chunks[8].canonical_begin_fp64_elements == 64);
  CHECK(plan.parameter_chunks.chunks[9].tensor_ordinal == 1);
  CHECK(plan.parameter_chunks.chunks[9].canonical_begin_fp64_elements == 72);
  CHECK(plan.parameter_chunks.chunks[10].tensor_ordinal == 2);
  CHECK(plan.parameter_chunks.chunks[10].canonical_begin_fp64_elements == 80);
  CHECK(plan.parameter_chunks.chunks[10].canonicalEndFp64Elements() == 88);

  const PsiFormerDerivativeLaunchRecord score =
      makePsiFormerDerivativeLaunchRecord(plan,
                                          PsiFormerDerivativeMode::ORDINARY_SCORE);
  CHECK(score.parameter_count == 88);
  CHECK(score.canonical_tensor_count == 3);
  CHECK(score.root_seed_count_fp64_elements == 1);
  CHECK(score.response_count_fp64_elements == 88);
  CHECK(score.complex_accumulator_count == 3);
  CHECK(score.launch.item_count == 88);
  CHECK(score.launch.block_count == 11);
  CHECK(score.launch.tail_count == 0);
  CHECK(score.fingerprint != 0);
  CHECK_THROWS_WITH(makePsiFormerDerivativeLaunchRecord(
                        plan, PsiFormerDerivativeMode::KINETIC_RESPONSE),
                    Catch::Matchers::ContainsSubstring("not enabled"));
}

TEST_CASE("PsiFormer future kinetic records are open-only and reuse one response",
          "[wavefunction][psiformer][derivative_planning]")
{
  const PsiFormerDeviceLayout open = makeDeviceLayout();
  PsiFormerDerivativePlanningRequirements requirements = makeRequirements(open);
  const PsiFormerDerivativePlan ordinary_open_plan =
      makePsiFormerDerivativePlan(open, 11, requirements);
  requirements.enable_kinetic_response = true;
  requirements.enable_imaginary_gradient_response = true;
  requirements.retained_tape_fp64_elements =
      minimumPsiFormerScoreTapeFP64Elements(open) * (1 + 4 * open.model_shape.electrons());
  const PsiFormerDerivativePlan plan =
      makePsiFormerDerivativePlan(open, 11, requirements);

  const PsiFormerDerivativeLaunchRecord score =
      makePsiFormerDerivativeLaunchRecord(plan,
                                          PsiFormerDerivativeMode::ORDINARY_SCORE);
  const PsiFormerDerivativeLaunchRecord kinetic =
      makePsiFormerDerivativeLaunchRecord(plan,
                                          PsiFormerDerivativeMode::KINETIC_RESPONSE);
  const PsiFormerDerivativeLaunchRecord imaginary =
      makePsiFormerDerivativeLaunchRecord(
          plan, PsiFormerDerivativeMode::IMAGINARY_GRADIENT_RESPONSE);
  CHECK(kinetic.root_seed_count_fp64_elements == 13);
  CHECK(imaginary.root_seed_offset_fp64_elements ==
        plan.reverse.root_seed_offset_fp64_elements + 1);
  CHECK(imaginary.root_seed_count_fp64_elements == 9);
  CHECK(score.response_offset_fp64_elements == kinetic.response_offset_fp64_elements);
  CHECK(kinetic.response_offset_fp64_elements == imaginary.response_offset_fp64_elements);

  const PsiFormerDeviceLayout periodic =
      makeDeviceLayout(8, BoundaryCondition::PERIODIC, 702);
  CHECK(minimumPsiFormerScoreTapeFP64Elements(periodic) == 516);
  PsiFormerDerivativePlanningRequirements periodic_score = makeRequirements(periodic);
  const PsiFormerDerivativePlan periodic_plan =
      makePsiFormerDerivativePlan(periodic, 11, periodic_score);
  CHECK(periodic_plan.minimum_retained_tape_fp64_elements == 516);
  CHECK(periodic_plan.reverse.fingerprint != ordinary_open_plan.reverse.fingerprint);
  CHECK(periodic_plan.fingerprint != ordinary_open_plan.fingerprint);
  periodic_score.enable_kinetic_response = true;
  periodic_score.retained_tape_fp64_elements =
      minimumPsiFormerScoreTapeFP64Elements(periodic) * 13;
  CHECK_THROWS_WITH(makePsiFormerDerivativePlan(periodic, 11, periodic_score),
                    Catch::Matchers::ContainsSubstring("periodic accelerator kinetic"));
}

TEST_CASE("PsiFormer ECP tile is typed bounded and independent of P",
          "[wavefunction][psiformer][derivative_planning]")
{
  const PsiFormerDeviceLayout first_device = makeDeviceLayout(8, BoundaryCondition::OPEN, 701);
  const PsiFormerDeviceLayout second_device = makeDeviceLayout(10, BoundaryCondition::OPEN, 703);
  PsiFormerDerivativePlanningRequirements first_requirements = makeRequirements(first_device);
  PsiFormerDerivativePlanningRequirements second_requirements = makeRequirements(second_device);
  const PsiFormerDerivativePlan first =
      makePsiFormerDerivativePlan(first_device, 11, first_requirements);
  const PsiFormerDerivativePlan second =
      makePsiFormerDerivativePlan(second_device, 11, second_requirements);

  CHECK(first.ecp_tile.maximum_sample_count == 4);
  CHECK(first.ecp_tile.maximum_segment_count == 6);
  CHECK(first.ecp_tile.maximum_point_count == 13);
  CHECK(first.ecp_tile.maximum_channel_count == 3);
  CHECK(first.ecp_tile.electron_count == 3);
  CHECK(first.ecp_tile.metadata_bytes == sizeof(PsiFormerECPTileMetadata));
  CHECK(first.ecp_tile.complete_ratios_offset_bytes -
            first.ecp_tile.bare_weights_offset_bytes ==
        13 * sizeof(double));
  CHECK(first.ecp_tile.total_bytes == second.ecp_tile.total_bytes);
  CHECK(first.ecp_tile.fingerprint == second.ecp_tile.fingerprint);
  CHECK(first.reverse.total_bytes != second.reverse.total_bytes);
  CHECK(first.accumulators.total_bytes != second.accumulators.total_bytes);
  CHECK(first.fingerprint != second.fingerprint);

  const PsiFormerDerivativePlan changed_version =
      makePsiFormerDerivativePlan(first_device, 12, first_requirements);
  CHECK(changed_version.fingerprint != first.fingerprint);

  const PsiFormerECPTileMetadata metadata = makeTileMetadata();
  const PsiFormerECPTileLaunchRecord tile =
      makePsiFormerECPTileLaunchRecord(first, metadata, /*channel_count=*/3);
  CHECK(tile.metadata.point_count == 13);
  CHECK(tile.channel_count == 3);
  CHECK(tile.launch.block_count == 2);
  CHECK(tile.launch.tail_count == 5);
  CHECK(tile.reference_positions_offset_bytes ==
        first.ecp_tile.reference_positions_offset_bytes);
  CHECK(tile.coefficients_offset_bytes == first.ecp_tile.coefficients_offset_bytes);
  CHECK(tile.point_status_offset_bytes == first.ecp_tile.point_status_offset_bytes);

  PsiFormerECPTileMetadata zero = metadata;
  zero.tile_ordinal = 3;
  zero.tile_fingerprint = 903;
  zero.flat_point_begin = 29;
  zero.sample_count = 0;
  zero.segment_count = 0;
  zero.point_count = 0;
  zero.evaluation_stamp_count = 0;
  const PsiFormerECPTileLaunchRecord empty =
      makePsiFormerECPTileLaunchRecord(first, zero, 1);
  CHECK(empty.launch.empty());
  CHECK(empty.launch.block_count == 0);

  PsiFormerECPTileMetadata fallback = metadata;
  fallback.flags |= ECP_DENSE_DERIVATIVE_FALLBACK;
  CHECK_THROWS_WITH(makePsiFormerECPTileLaunchRecord(first, fallback, 1),
                    Catch::Matchers::ContainsSubstring("flags"));
  PsiFormerECPTileMetadata stale = metadata;
  ++stale.parameter_version;
  CHECK_THROWS_WITH(makePsiFormerECPTileLaunchRecord(first, stale, 1),
                    Catch::Matchers::ContainsSubstring("stale"));
  PsiFormerECPTileMetadata too_many = metadata;
  ++too_many.point_count;
  CHECK_THROWS_WITH(makePsiFormerECPTileLaunchRecord(first, too_many, 1),
                    Catch::Matchers::ContainsSubstring("capacities"));

  PsiFormerDerivativePlan aliased_ecp = first;
  aliased_ecp.ecp_tile.coefficients_offset_bytes =
      aliased_ecp.ecp_tile.complete_ratios_offset_bytes;
  CHECK_THROWS_WITH(validatePsiFormerDerivativePlan(aliased_ecp),
                    Catch::Matchers::ContainsSubstring("ECP tile"));
}

TEST_CASE("PsiFormer derivative planner rejects underallocation and overflow",
          "[wavefunction][psiformer][derivative_planning]")
{
  const PsiFormerDeviceLayout device = makeDeviceLayout();
  PsiFormerDerivativePlanningRequirements requirements = makeRequirements(device);
  --requirements.retained_tape_fp64_elements;
  CHECK_THROWS_WITH(makePsiFormerDerivativePlan(device, 11, requirements),
                    Catch::Matchers::ContainsSubstring("derived mode floor"));

  requirements = makeRequirements(device);
  requirements.complex_accumulator_count = 4;
  CHECK_THROWS_WITH(makePsiFormerDerivativePlan(device, 11, requirements),
                    Catch::Matchers::ContainsSubstring("one to three"));

  requirements = makeRequirements(device);
  requirements.maximum_ecp_segment_count = 0;
  CHECK_THROWS_WITH(makePsiFormerDerivativePlan(device, 11, requirements),
                    Catch::Matchers::ContainsSubstring("B, M, Q"));

  requirements = makeRequirements(device);
  requirements.maximum_ecp_sample_count =
      std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(makePsiFormerDerivativePlan(device, 11, requirements),
                  std::length_error);

  PsiFormerDeviceLayout complex_compute = device;
  complex_compute.compute_scalar_domain = ScalarDomain::COMPLEX;
  complex_compute.fingerprint = 704;
  CHECK_THROWS_WITH(makePsiFormerDerivativePlan(
                        complex_compute, 11, makeRequirements(complex_compute)),
                    Catch::Matchers::ContainsSubstring("real FP64"));

  PsiFormerDeviceLayout invalid_embedding = device;
  invalid_embedding.tensors.front().rank = 1;
  invalid_embedding.fingerprint = 705;
  CHECK_THROWS_WITH(minimumPsiFormerScoreTapeFP64Elements(invalid_embedding),
                    Catch::Matchers::ContainsSubstring("rank-2 electron-embedding"));

  PsiFormerDeviceLayout duplicate_embedding = device;
  duplicate_embedding.tensors[1].role = ParameterRole::ELECTRON_EMBEDDING_WEIGHT;
  duplicate_embedding.fingerprint = 706;
  CHECK_THROWS_WITH(minimumPsiFormerScoreTapeFP64Elements(duplicate_embedding),
                    Catch::Matchers::ContainsSubstring("exactly one electron-embedding"));
}

} // namespace qmcplusplus::psiformer

int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
