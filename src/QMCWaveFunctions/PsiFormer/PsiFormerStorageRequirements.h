//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerStorageRequirements.h
 * @brief Checked numeric-storage descriptors shared by direct PsiFormer workspaces.
 *
 * These descriptors count owned numeric backing storage, not C++ object overhead,
 * allocator metadata, stacks, immutable model parameters, or caller-owned outputs.
 * Keeping the extent arithmetic beside the corresponding workspace allocation code
 * gives the batch-memory policy a deterministic estimate without constructing scratch.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_STORAGE_REQUIREMENTS_H
#define QMCPLUSPLUS_PSIFORMER_STORAGE_REQUIREMENTS_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerDeterminant.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerGeometry.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pf
{

/// Add byte or element extents while rejecting wraparound before allocation.
inline std::size_t checkedStorageSum(std::size_t left,
                                     std::size_t right,
                                     const char* quantity)
{
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::length_error(quantity);
  return left + right;
}

/// Multiply byte or element extents while rejecting wraparound before allocation.
inline std::size_t checkedStorageProduct(std::size_t left,
                                         std::size_t right,
                                         const char* quantity)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(quantity);
  return left * right;
}

/// Convert a typed element count to bytes with checked arithmetic.
template<class T>
inline std::size_t checkedStorageBytes(std::size_t elements, const char* quantity)
{
  return checkedStorageProduct(elements, sizeof(T), quantity);
}

/// Add one checked byte category to an accumulated requirement.
inline void addStorageBytes(std::size_t& total,
                            std::size_t bytes,
                            const char* quantity)
{
  total = checkedStorageSum(total, bytes, quantity);
}

/// Fixed model dimensions needed by allocation-free storage estimators.
struct PsiFormerStorageShape
{
  std::size_t electrons     = 0;
  std::size_t nuclei        = 0;
  std::size_t determinants  = 0;
  std::size_t feature_width = 0;
  std::size_t attention_heads = 0;
  std::size_t input_width     = 0;
  std::size_t attention_blocks = 0;
  std::size_t parameter_count   = 0;
};

/// Independently selected tile capacities for each direct batch call family.
struct DirectBatchTileCapacities
{
  std::size_t value           = 4;
  std::size_t full_vgl        = 4;
  std::size_t active_gradient = 4;
};

/// Logical high-water bounds for dense and sparse batch requests.
struct DirectBatchLogicalCapacities
{
  std::size_t value_dense         = 0;
  std::size_t full_vgl            = 0;
  std::size_t active_gradient     = 0;
  std::size_t sparse_references   = 0;
  std::size_t sparse_replacements = 0;
};

/** Describe one complete direct-batch allocation target.
 *
 * A zero tile is valid only when the associated logical family is unused.  Sparse
 * references and replacements use the VALUE tile and coexist with dense input.
 */
struct DirectBatchCapacityPlan
{
  DirectBatchLogicalCapacities logical;
  DirectBatchTileCapacities tile;
};

/** Categorize retained direct-batch numeric storage without double counting.
 *
 * VALUE, FULL_VGL, and ACTIVE_GRADIENT pools coexist and are additive.  The two
 * spatial modes share one packed source/target arena, which is counted once at the
 * larger retained extent.  Dense and sparse logical inputs also coexist.
 */
struct DirectBatchStorageRequirement
{
  std::size_t dense_logical          = 0;
  std::size_t sparse_logical         = 0;
  std::size_t logical_outputs        = 0;
  std::size_t sparse_tile_positions  = 0;
  std::size_t value_tile             = 0;
  std::size_t full_vgl_tile          = 0;
  std::size_t active_gradient_tile   = 0;
  std::size_t shared_spatial_arena   = 0;
  /// Additional setup-only storage conservatively allowed during a warmed replan.
  std::size_t replacement_transient  = 0;

  /// Return retained execution storage, excluding setup-only replacement overlap.
  std::size_t executionBytes() const
  {
    std::size_t total = 0;
    for (const std::size_t bytes : {dense_logical, sparse_logical, logical_outputs,
                                    sparse_tile_positions, value_tile, full_vgl_tile,
                                    active_gradient_tile, shared_spatial_arena})
      addStorageBytes(total, bytes, "PsiFormer batch storage total overflowed");
    return total;
  }

  /// Preserve the original diagnostic name with execution-only semantics.
  std::size_t totalBytes() const { return executionBytes(); }

  /// Return the additional conservative setup-only replacement allowance.
  std::size_t replacementTransientBytes() const noexcept
  { return replacement_transient; }

  /// Return the conservative peak while publishing a warmed replacement plan.
  std::size_t setupPeakBytes() const
  {
    return checkedStorageSum(executionBytes(), replacement_transient,
                             "PsiFormer batch setup peak overflowed");
  }
};

/** Categorize the accepted/proposed spatial arrays owned by one component clone.
 *
 * Element widths are supplied by PsiFormerWF so real and complex-adapter builds
 * charge their actual ``ValueType`` and ``GradType`` representations.
 */
struct CloneStateStorageRequirement
{
  std::size_t accepted_gradients   = 0;
  std::size_t proposed_gradients   = 0;
  std::size_t accepted_laplacians  = 0;
  std::size_t proposed_laplacians  = 0;

  std::size_t totalBytes() const
  {
    std::size_t total = 0;
    for (const std::size_t bytes : {accepted_gradients, proposed_gradients,
                                    accepted_laplacians, proposed_laplacians})
      addStorageBytes(total, bytes,
                      "PsiFormer clone-state storage total overflowed");
    return total;
  }
};

/** Return exact numeric backing for one resident component clone's fixed state. */
inline CloneStateStorageRequirement cloneStateStorageRequirement(
    std::size_t electrons,
    std::size_t value_type_bytes,
    std::size_t gradient_type_bytes)
{
  if (electrons != 0 && (value_type_bytes == 0 || gradient_type_bytes == 0))
    throw std::invalid_argument(
        "PsiFormer clone-state element widths must be positive");

  CloneStateStorageRequirement result;
  result.accepted_gradients = checkedStorageProduct(
      electrons, gradient_type_bytes,
      "PsiFormer accepted-gradient storage overflowed");
  result.proposed_gradients = result.accepted_gradients;
  result.accepted_laplacians = checkedStorageProduct(
      electrons, value_type_bytes,
      "PsiFormer accepted-Laplacian storage overflowed");
  result.proposed_laplacians = result.accepted_laplacians;
  return result;
}

/** Logical capacities for shared, resource-owned publication staging.
 *
 * Dense operation families share typed staging arrays at the reserve-walker high
 * water.  Sparse flattened ECP arrays coexist because an ECP call consumes both
 * reference and replacement metadata.  Legacy unflattened storage is deliberately
 * absent: an explicit memory policy must reject that path.
 */
struct ResourceStagingCapacityPlan
{
  std::size_t reserve_walkers      = 0;
  std::size_t sparse_references    = 0;
  std::size_t sparse_replacements  = 0;
  std::size_t active_parameters    = 0;
  std::size_t value_type_bytes     = 0;
  std::size_t log_value_type_bytes = 0;
  std::size_t gradient_type_bytes  = 0;
  std::size_t selected_delta_bytes = 0;
  bool value                       = false;
  bool full_vgl                    = false;
  bool active_gradient             = false;
  bool flattened_ecp               = false;
  bool weighted_ecp_score          = false;
  bool score                       = false;
  bool kinetic                     = false;
};

/** Categorized numeric buffers used to validate then publish direct results. */
struct ResourceStagingStorageRequirement
{
  std::size_t walker_indices              = 0;
  std::size_t active_electrons            = 0;
  std::size_t configuration_identities    = 0;
  std::size_t batch_slots                 = 0;
  std::size_t signs                       = 0;
  std::size_t log_magnitudes              = 0;
  std::size_t ratios                      = 0;
  std::size_t gradients                   = 0;
  std::size_t preservation_flags          = 0;
  std::size_t active_virtual_walkers      = 0;
  std::size_t virtual_reference_indices   = 0;
  std::size_t flattened_virtual_ratios    = 0;
  std::size_t virtual_reference_weights   = 0;
  std::size_t active_parameter_indices    = 0;
  std::size_t selected_derivative_deltas  = 0;
  std::size_t weighted_derivatives        = 0;

  std::size_t totalBytes() const
  {
    std::size_t total = 0;
    for (const std::size_t bytes : {
             walker_indices, active_electrons, configuration_identities,
             batch_slots, signs, log_magnitudes, ratios, gradients,
             preservation_flags, active_virtual_walkers,
             virtual_reference_indices, flattened_virtual_ratios,
             virtual_reference_weights, active_parameter_indices,
             selected_derivative_deltas, weighted_derivatives})
      addStorageBytes(total, bytes,
                      "PsiFormer publication staging total overflowed");
    return total;
  }
};

/** Estimate exact shared publication staging for one prepared crowd. */
inline ResourceStagingStorageRequirement resourceStagingStorageRequirement(
    const ResourceStagingCapacityPlan& plan)
{
  if (plan.weighted_ecp_score && !plan.flattened_ecp)
    throw std::invalid_argument(
        "PsiFormer weighted ECP staging requires flattened ECP storage");

  const bool dense = plan.value || plan.full_vgl || plan.active_gradient;
  if ((plan.value || plan.active_gradient) && plan.reserve_walkers != 0 && plan.value_type_bytes == 0)
    throw std::invalid_argument(
        "PsiFormer publication value element width must be positive");
  if (plan.full_vgl && plan.reserve_walkers != 0 && plan.log_value_type_bytes == 0)
    throw std::invalid_argument(
        "PsiFormer publication log-value element width must be positive");
  if ((plan.full_vgl || plan.active_gradient) &&
      plan.reserve_walkers != 0 && plan.gradient_type_bytes == 0)
    throw std::invalid_argument(
        "PsiFormer publication gradient element width must be positive");
  const bool flattened_values =
      plan.flattened_ecp && plan.sparse_replacements != 0;
  const bool weighted_reference_values =
      plan.weighted_ecp_score && plan.sparse_references != 0;
  if ((flattened_values || weighted_reference_values) &&
      plan.value_type_bytes == 0)
    throw std::invalid_argument(
        "PsiFormer flattened ECP value element width must be positive");
  if ((plan.weighted_ecp_score || plan.score || plan.kinetic) &&
      plan.active_parameters != 0 &&
      plan.selected_delta_bytes == 0)
    throw std::invalid_argument(
        "PsiFormer selected-derivative element width must be positive");

  const auto typed_bytes = [](std::size_t elements, std::size_t element_bytes,
                              const char* quantity) {
    return checkedStorageProduct(elements, element_bytes, quantity);
  };

  ResourceStagingStorageRequirement result;
  if (dense)
  {
    result.walker_indices = checkedStorageBytes<std::size_t>(
        plan.reserve_walkers,
        "PsiFormer walker-index staging bytes overflowed");
    result.configuration_identities = checkedStorageBytes<std::uint64_t>(
        plan.reserve_walkers,
        "PsiFormer configuration-identity staging bytes overflowed");
    result.signs = checkedStorageBytes<double>(
        plan.reserve_walkers,
        "PsiFormer sign staging bytes overflowed");
    result.log_magnitudes = checkedStorageBytes<double>(
        plan.reserve_walkers,
        "PsiFormer log-magnitude staging bytes overflowed");
    const std::size_t ratio_element_bytes = plan.full_vgl
        ? std::max(plan.value_type_bytes, plan.log_value_type_bytes)
        : plan.value_type_bytes;
    result.ratios = typed_bytes(
        plan.reserve_walkers, ratio_element_bytes,
        "PsiFormer ratio staging bytes overflowed");
  }
  if (plan.value)
    result.preservation_flags = checkedStorageBytes<unsigned char>(
        plan.reserve_walkers,
        "PsiFormer preservation-flag staging bytes overflowed");
  if (plan.full_vgl)
    result.batch_slots = checkedStorageBytes<std::size_t>(
        plan.reserve_walkers,
        "PsiFormer batch-slot staging bytes overflowed");
  if (plan.full_vgl || plan.active_gradient)
    result.gradients = typed_bytes(
        plan.reserve_walkers, plan.gradient_type_bytes,
        "PsiFormer gradient staging bytes overflowed");
  if (plan.active_gradient)
    result.active_electrons = checkedStorageBytes<std::size_t>(
        plan.reserve_walkers,
        "PsiFormer active-electron staging bytes overflowed");

  if (plan.flattened_ecp)
  {
    result.active_virtual_walkers = checkedStorageBytes<std::size_t>(
        plan.sparse_references,
        "PsiFormer active-virtual-walker staging bytes overflowed");
    result.virtual_reference_indices = checkedStorageBytes<std::size_t>(
        plan.reserve_walkers,
        "PsiFormer virtual-reference-index staging bytes overflowed");
    result.flattened_virtual_ratios = typed_bytes(
        plan.sparse_replacements, plan.value_type_bytes,
        "PsiFormer flattened-ratio staging bytes overflowed");
  }

  if (plan.weighted_ecp_score)
  {
    result.virtual_reference_weights = typed_bytes(
        plan.sparse_references, plan.value_type_bytes,
        "PsiFormer reference-weight staging bytes overflowed");
    result.weighted_derivatives = typed_bytes(
        checkedStorageProduct(
            plan.sparse_references, plan.active_parameters,
            "PsiFormer weighted-derivative staging extent overflowed"),
        plan.value_type_bytes,
        "PsiFormer weighted-derivative staging bytes overflowed");
  }
  if (plan.weighted_ecp_score || plan.score || plan.kinetic)
  {
    result.active_parameter_indices = checkedStorageBytes<std::size_t>(
        plan.active_parameters,
        "PsiFormer active-parameter-index staging bytes overflowed");
    const std::size_t delta_copies = plan.kinetic ? 2 : 1;
    result.selected_derivative_deltas = typed_bytes(
        checkedStorageProduct(
            delta_copies, plan.active_parameters,
            "PsiFormer selected-derivative staging extent overflowed"),
        plan.selected_delta_bytes,
        "PsiFormer selected-derivative staging bytes overflowed");
  }
  return result;
}

/** Return clone-local scalar VALUE publication scratch at its logical envelope. */
inline std::size_t scalarValuePublicationStorageRequirement(
    std::size_t logical_maximum,
    std::size_t value_type_bytes)
{
  if (logical_maximum != 0 && value_type_bytes == 0)
    throw std::invalid_argument(
        "PsiFormer scalar VALUE element width must be positive");
  return checkedStorageProduct(
      logical_maximum, value_type_bytes,
      "PsiFormer scalar VALUE publication bytes overflowed");
}

/// Return N*(N-1)/2 with checked arithmetic and no overflowing intermediate.
inline std::size_t checkedUniqueElectronPairs(std::size_t electrons)
{
  if (electrons < 2)
    return 0;
  std::size_t left  = electrons;
  std::size_t right = electrons - 1;
  if ((left & 1U) == 0)
    left /= 2;
  else
    right /= 2;
  return checkedStorageProduct(left, right,
                               "PsiFormer geometry pair extent overflowed");
}

/// Estimate every vector owned by one fixed-size geometry cache.
inline std::size_t geometryStorageRequirement(std::size_t electrons,
                                              std::size_t nuclei)
{
  const std::size_t electron_pairs = checkedUniqueElectronPairs(electrons);
  const std::size_t electron_nucleus_pairs = checkedStorageProduct(
      electrons, nuclei, "PsiFormer electron-nucleus extent overflowed");
  const auto pair_table_bytes = [](std::size_t pairs) {
    std::size_t per_pair = 0;
    addStorageBytes(per_pair, sizeof(GeometryPosition),
                    "PsiFormer pair-table element bytes overflowed");
    addStorageBytes(per_pair, 2 * sizeof(GeometryReal),
                    "PsiFormer pair-table element bytes overflowed");
    addStorageBytes(per_pair, sizeof(SoftenedRadialFactors),
                    "PsiFormer pair-table element bytes overflowed");
    return checkedStorageProduct(pairs, per_pair,
                                 "PsiFormer pair-table bytes overflowed");
  };

  std::size_t bytes = 0;
  addStorageBytes(bytes, checkedStorageBytes<GeometryPosition>(
                             nuclei, "PsiFormer nuclear geometry bytes overflowed"),
                  "PsiFormer geometry bytes overflowed");
  addStorageBytes(bytes, checkedStorageBytes<GeometryPosition>(
                             electrons, "PsiFormer electron geometry bytes overflowed"),
                  "PsiFormer geometry bytes overflowed");
  addStorageBytes(bytes, checkedStorageBytes<ElectronPair>(
                             electron_pairs, "PsiFormer pair identity bytes overflowed"),
                  "PsiFormer geometry bytes overflowed");
  addStorageBytes(bytes, checkedStorageBytes<std::size_t>(
                             checkedStorageSum(electrons, 1,
                                               "PsiFormer incidence offset extent overflowed"),
                             "PsiFormer incidence offset bytes overflowed"),
                  "PsiFormer geometry bytes overflowed");
  addStorageBytes(bytes, checkedStorageBytes<ElectronPairIncidence>(
                             checkedStorageProduct(2, electron_pairs,
                                                   "PsiFormer incidence extent overflowed"),
                             "PsiFormer incidence bytes overflowed"),
                  "PsiFormer geometry bytes overflowed");
  addStorageBytes(bytes, pair_table_bytes(electron_nucleus_pairs),
                  "PsiFormer geometry bytes overflowed");
  addStorageBytes(bytes, pair_table_bytes(electron_pairs),
                  "PsiFormer geometry bytes overflowed");
  return bytes;
}

/// Estimate every vector owned by one determinant factorization workspace.
inline std::size_t determinantStorageRequirement(std::size_t determinants,
                                                 std::size_t electrons,
                                                 std::size_t gradient_lanes = 0,
                                                 std::size_t laplacian_lanes = 0)
{
  using qmcplusplus::psiformer::determinant::ChannelFactorization;
  using qmcplusplus::psiformer::determinant::detail::CompensatedSum;
  const std::size_t matrix = checkedStorageProduct(
      electrons, electrons, "PsiFormer determinant matrix extent overflowed");
  const std::size_t determinant_matrix = checkedStorageProduct(
      determinants, matrix, "PsiFormer determinant batch extent overflowed");
  const std::size_t determinant_rows = checkedStorageProduct(
      determinants, electrons, "PsiFormer determinant row extent overflowed");

  std::size_t bytes = 0;
  addStorageBytes(bytes, checkedStorageBytes<double>(
                             checkedStorageProduct(2, determinant_matrix,
                                                   "PsiFormer determinant factor extent overflowed"),
                             "PsiFormer determinant factor bytes overflowed"),
                  "PsiFormer determinant bytes overflowed");
  addStorageBytes(bytes, checkedStorageBytes<std::size_t>(
                             determinant_rows,
                             "PsiFormer determinant permutation bytes overflowed"),
                  "PsiFormer determinant bytes overflowed");
  addStorageBytes(bytes, checkedStorageBytes<ChannelFactorization>(
                             determinants,
                             "PsiFormer determinant record bytes overflowed"),
                  "PsiFormer determinant bytes overflowed");
  addStorageBytes(bytes, checkedStorageBytes<double>(
                             checkedStorageProduct(3, determinants,
                                                   "PsiFormer determinant reduction extent overflowed"),
                             "PsiFormer determinant reduction bytes overflowed"),
                  "PsiFormer determinant bytes overflowed");
  addStorageBytes(bytes, checkedStorageBytes<long double>(
                             determinants,
                             "PsiFormer determinant scaled-term bytes overflowed"),
                  "PsiFormer determinant bytes overflowed");
  addStorageBytes(bytes, checkedStorageBytes<double>(
                             electrons, "PsiFormer determinant solve bytes overflowed"),
                  "PsiFormer determinant bytes overflowed");
  addStorageBytes(bytes, checkedStorageBytes<long double>(
                             matrix, "PsiFormer determinant product bytes overflowed"),
                  "PsiFormer determinant bytes overflowed");
  addStorageBytes(bytes, checkedStorageBytes<CompensatedSum>(
                             gradient_lanes,
                             "PsiFormer determinant gradient bytes overflowed"),
                  "PsiFormer determinant bytes overflowed");
  addStorageBytes(bytes, checkedStorageBytes<CompensatedSum>(
                             laplacian_lanes,
                             "PsiFormer determinant Laplacian bytes overflowed"),
                  "PsiFormer determinant bytes overflowed");
  return bytes;
}

/// Estimate the inclusive geometry, algebra, and determinant value workspace.
inline std::size_t valueWorkspaceStorageRequirement(const PsiFormerStorageShape& shape)
{
  const std::size_t electron_features = checkedStorageProduct(
      shape.electrons, shape.feature_width,
      "PsiFormer value feature extent overflowed");
  const std::size_t electron_input = checkedStorageProduct(
      shape.electrons, shape.input_width,
      "PsiFormer value input extent overflowed");
  const std::size_t electron_square = checkedStorageProduct(
      shape.electrons, shape.electrons,
      "PsiFormer value electron-square extent overflowed");
  const std::size_t attention = checkedStorageProduct(
      shape.attention_heads, electron_square,
      "PsiFormer value attention extent overflowed");
  const std::size_t orbitals = checkedStorageProduct(
      shape.determinants, electron_square,
      "PsiFormer value orbital extent overflowed");

  std::size_t elements = checkedStorageProduct(3, shape.electrons,
                                               "PsiFormer position extent overflowed");
  addStorageBytes(elements, electron_input, "PsiFormer value element extent overflowed");
  addStorageBytes(elements, checkedStorageProduct(7, electron_features,
                                                  "PsiFormer value feature extent overflowed"),
                  "PsiFormer value element extent overflowed");
  addStorageBytes(elements, attention, "PsiFormer value element extent overflowed");
  addStorageBytes(elements, orbitals, "PsiFormer value element extent overflowed");

  std::size_t bytes = checkedStorageBytes<double>(
      elements, "PsiFormer value vector bytes overflowed");
  addStorageBytes(bytes, geometryStorageRequirement(shape.electrons, shape.nuclei),
                  "PsiFormer value workspace bytes overflowed");
  addStorageBytes(bytes, determinantStorageRequirement(shape.determinants,
                                                       shape.electrons),
                  "PsiFormer value workspace bytes overflowed");
  return bytes;
}

/// Estimate one spatial workspace, including geometry and determinant storage.
inline std::size_t spatialWorkspaceStorageRequirement(const PsiFormerStorageShape& shape,
                                                      bool full_vgl)
{
  const std::size_t gradient_lanes = full_vgl
      ? checkedStorageProduct(3, shape.electrons,
                              "PsiFormer spatial gradient extent overflowed")
      : 3;
  const std::size_t laplacian_lanes = full_vgl ? shape.electrons : 0;
  const std::size_t planes = checkedStorageSum(
      1, checkedStorageSum(gradient_lanes, laplacian_lanes,
                           "PsiFormer spatial plane extent overflowed"),
      "PsiFormer spatial plane extent overflowed");
  const std::size_t feature = checkedStorageProduct(
      shape.electrons, shape.feature_width,
      "PsiFormer spatial feature extent overflowed");
  const std::size_t input = checkedStorageProduct(
      shape.electrons, shape.input_width,
      "PsiFormer spatial input extent overflowed");
  const std::size_t square = checkedStorageProduct(
      shape.electrons, shape.electrons,
      "PsiFormer spatial electron-square extent overflowed");
  const std::size_t attention = checkedStorageProduct(
      shape.attention_heads, square,
      "PsiFormer spatial attention extent overflowed");
  const std::size_t orbitals = checkedStorageProduct(
      shape.determinants, square,
      "PsiFormer spatial orbital extent overflowed");

  std::size_t jet_values = input;
  addStorageBytes(jet_values, checkedStorageProduct(7, feature,
                                                    "PsiFormer spatial feature extent overflowed"),
                  "PsiFormer spatial jet extent overflowed");
  addStorageBytes(jet_values, attention, "PsiFormer spatial jet extent overflowed");
  addStorageBytes(jet_values, orbitals, "PsiFormer spatial jet extent overflowed");
  std::size_t elements = checkedStorageProduct(
      jet_values, planes, "PsiFormer spatial jet storage overflowed");
  addStorageBytes(elements, checkedStorageProduct(3, shape.electrons,
                                                  "PsiFormer spatial position extent overflowed"),
                  "PsiFormer spatial element extent overflowed");
  addStorageBytes(elements, checkedStorageProduct(2, gradient_lanes,
                                                  "PsiFormer spatial scratch extent overflowed"),
                  "PsiFormer spatial element extent overflowed");
  addStorageBytes(elements, checkedStorageProduct(3, laplacian_lanes,
                                                  "PsiFormer spatial scratch extent overflowed"),
                  "PsiFormer spatial element extent overflowed");

  std::size_t bytes = checkedStorageBytes<double>(
      elements, "PsiFormer spatial vector bytes overflowed");
  addStorageBytes(bytes, geometryStorageRequirement(shape.electrons, shape.nuclei),
                  "PsiFormer spatial workspace bytes overflowed");
  addStorageBytes(bytes, determinantStorageRequirement(
                             shape.determinants, shape.electrons,
                             gradient_lanes, laplacian_lanes),
                  "PsiFormer spatial workspace bytes overflowed");
  return bytes;
}

/// Estimate the compact forward/reverse parameter-score tape inclusively.
inline std::size_t scoreWorkspaceStorageRequirement(const PsiFormerStorageShape& shape)
{
  const std::size_t feature = checkedStorageProduct(
      shape.electrons, shape.feature_width,
      "PsiFormer score feature extent overflowed");
  const std::size_t input = checkedStorageProduct(
      shape.electrons, shape.input_width,
      "PsiFormer score input extent overflowed");
  const std::size_t square = checkedStorageProduct(
      shape.electrons, shape.electrons,
      "PsiFormer score electron-square extent overflowed");
  const std::size_t attention = checkedStorageProduct(
      shape.attention_heads, square,
      "PsiFormer score attention extent overflowed");
  const std::size_t orbitals = checkedStorageProduct(
      shape.determinants, square,
      "PsiFormer score orbital extent overflowed");
  const std::size_t block_features = checkedStorageProduct(
      shape.attention_blocks, feature,
      "PsiFormer score block-feature extent overflowed");
  const std::size_t block_attention = checkedStorageProduct(
      shape.attention_blocks, attention,
      "PsiFormer score block-attention extent overflowed");

  std::size_t elements = checkedStorageProduct(
      3, shape.electrons, "PsiFormer score position extent overflowed");
  addStorageBytes(elements, input, "PsiFormer score element extent overflowed");
  addStorageBytes(elements, checkedStorageProduct(
                                checkedStorageSum(shape.attention_blocks, 1,
                                                  "PsiFormer score feature tape overflowed"),
                                feature, "PsiFormer score feature tape overflowed"),
                  "PsiFormer score element extent overflowed");
  addStorageBytes(elements, checkedStorageProduct(
                                7, block_features,
                                "PsiFormer score block tape overflowed"),
                  "PsiFormer score element extent overflowed");
  addStorageBytes(elements, block_attention,
                  "PsiFormer score element extent overflowed");
  addStorageBytes(elements, checkedStorageProduct(
                                2, orbitals,
                                "PsiFormer score orbital extent overflowed"),
                  "PsiFormer score element extent overflowed");
  addStorageBytes(elements, shape.parameter_count,
                  "PsiFormer score element extent overflowed");
  addStorageBytes(elements, checkedStorageProduct(
                                9, feature,
                                "PsiFormer score adjoint extent overflowed"),
                  "PsiFormer score element extent overflowed");
  addStorageBytes(elements, attention,
                  "PsiFormer score element extent overflowed");

  std::size_t bytes = checkedStorageBytes<double>(
      elements, "PsiFormer score vector bytes overflowed");
  addStorageBytes(bytes, geometryStorageRequirement(shape.electrons, shape.nuclei),
                  "PsiFormer score workspace bytes overflowed");
  addStorageBytes(bytes, determinantStorageRequirement(shape.determinants,
                                                       shape.electrons),
                  "PsiFormer score workspace bytes overflowed");
  return bytes;
}

/// Return bytes in one value/gradient/Laplacian trace-jet allocation.
inline std::size_t traceJetStorageRequirement(std::size_t values,
                                              std::size_t gradient_lanes,
                                              std::size_t laplacian_lanes)
{
  const std::size_t planes = checkedStorageSum(
      1, checkedStorageSum(gradient_lanes, laplacian_lanes,
                           "PsiFormer trace-jet plane extent overflowed"),
      "PsiFormer trace-jet plane extent overflowed");
  return checkedStorageBytes<double>(
      checkedStorageProduct(values, planes,
                            "PsiFormer trace-jet extent overflowed"),
      "PsiFormer trace-jet bytes overflowed");
}

/// Estimate the fixed exact kinetic-response tape, including geometry and factors.
inline std::size_t kineticWorkspaceStorageRequirement(const PsiFormerStorageShape& shape)
{
  const std::size_t gradient_lanes = checkedStorageProduct(
      3, shape.electrons, "PsiFormer kinetic gradient extent overflowed");
  const std::size_t laplacian_lanes = shape.electrons;
  const std::size_t feature = checkedStorageProduct(
      shape.electrons, shape.feature_width,
      "PsiFormer kinetic feature extent overflowed");
  const std::size_t input = checkedStorageProduct(
      shape.electrons, shape.input_width,
      "PsiFormer kinetic input extent overflowed");
  const std::size_t square = checkedStorageProduct(
      shape.electrons, shape.electrons,
      "PsiFormer kinetic electron-square extent overflowed");
  const std::size_t attention = checkedStorageProduct(
      shape.attention_heads, square,
      "PsiFormer kinetic attention extent overflowed");
  const std::size_t orbitals = checkedStorageProduct(
      shape.determinants, square,
      "PsiFormer kinetic orbital extent overflowed");

  std::size_t bytes = checkedStorageBytes<double>(
      checkedStorageProduct(3, shape.electrons,
                            "PsiFormer kinetic position extent overflowed"),
      "PsiFormer kinetic position bytes overflowed");
  addStorageBytes(bytes, traceJetStorageRequirement(
                             input, gradient_lanes, laplacian_lanes),
                  "PsiFormer kinetic workspace bytes overflowed");
  addStorageBytes(bytes, checkedStorageProduct(
                             checkedStorageSum(shape.attention_blocks, 1,
                                               "PsiFormer kinetic feature tape overflowed"),
                             traceJetStorageRequirement(
                                 feature, gradient_lanes, laplacian_lanes),
                             "PsiFormer kinetic feature tape overflowed"),
                  "PsiFormer kinetic workspace bytes overflowed");

  std::size_t per_block = checkedStorageProduct(
      9, traceJetStorageRequirement(feature, gradient_lanes, laplacian_lanes),
      "PsiFormer kinetic block-feature bytes overflowed");
  addStorageBytes(per_block, checkedStorageProduct(
                                2, traceJetStorageRequirement(
                                       attention, gradient_lanes,
                                       laplacian_lanes),
                                "PsiFormer kinetic block-attention bytes overflowed"),
                  "PsiFormer kinetic block bytes overflowed");
  addStorageBytes(bytes, checkedStorageProduct(
                             shape.attention_blocks, per_block,
                             "PsiFormer kinetic block tape bytes overflowed"),
                  "PsiFormer kinetic workspace bytes overflowed");

  addStorageBytes(bytes, checkedStorageProduct(
                             7, traceJetStorageRequirement(
                                    orbitals, gradient_lanes,
                                    laplacian_lanes),
                             "PsiFormer kinetic orbital tape bytes overflowed"),
                  "PsiFormer kinetic workspace bytes overflowed");
  addStorageBytes(bytes, checkedStorageProduct(
                             11, traceJetStorageRequirement(
                                     feature, gradient_lanes,
                                     laplacian_lanes),
                             "PsiFormer kinetic feature-adjoint bytes overflowed"),
                  "PsiFormer kinetic workspace bytes overflowed");
  addStorageBytes(bytes, checkedStorageProduct(
                             2, traceJetStorageRequirement(
                                    attention, gradient_lanes,
                                    laplacian_lanes),
                             "PsiFormer kinetic attention-adjoint bytes overflowed"),
                  "PsiFormer kinetic workspace bytes overflowed");
  addStorageBytes(bytes, checkedStorageProduct(
                             2, traceJetStorageRequirement(
                                    shape.electrons, gradient_lanes,
                                    laplacian_lanes),
                             "PsiFormer kinetic row tape bytes overflowed"),
                  "PsiFormer kinetic workspace bytes overflowed");
  addStorageBytes(bytes, checkedStorageProduct(
                             5, traceJetStorageRequirement(
                                    1, gradient_lanes, laplacian_lanes),
                             "PsiFormer kinetic scalar tape bytes overflowed"),
                  "PsiFormer kinetic workspace bytes overflowed");

  std::size_t plain_elements = checkedStorageProduct(
      3, gradient_lanes, "PsiFormer kinetic gradient output overflowed");
  addStorageBytes(plain_elements, checkedStorageProduct(
                                      5, laplacian_lanes,
                                      "PsiFormer kinetic Laplacian output overflowed"),
                  "PsiFormer kinetic output extent overflowed");
  addStorageBytes(plain_elements, checkedStorageProduct(
                                      2, shape.parameter_count,
                                      "PsiFormer kinetic parameter output overflowed"),
                  "PsiFormer kinetic output extent overflowed");
  addStorageBytes(plain_elements, checkedStorageProduct(
                                      3, square,
                                      "PsiFormer kinetic matrix scratch overflowed"),
                  "PsiFormer kinetic output extent overflowed");
  addStorageBytes(bytes, checkedStorageBytes<double>(
                             plain_elements,
                             "PsiFormer kinetic output bytes overflowed"),
                  "PsiFormer kinetic workspace bytes overflowed");
  addStorageBytes(bytes, geometryStorageRequirement(shape.electrons, shape.nuclei),
                  "PsiFormer kinetic workspace bytes overflowed");
  addStorageBytes(bytes, determinantStorageRequirement(
                             shape.determinants, shape.electrons,
                             gradient_lanes, laplacian_lanes),
                  "PsiFormer kinetic workspace bytes overflowed");
  return bytes;
}

/// Return the packed dense-kernel arena elements required by one spatial tile.
inline std::size_t spatialPackedElements(const PsiFormerStorageShape& shape,
                                         std::size_t tile,
                                         bool full_vgl)
{
  const std::size_t gradient_lanes = full_vgl
      ? checkedStorageProduct(3, shape.electrons,
                              "PsiFormer packed gradient extent overflowed")
      : 3;
  const std::size_t laplacian_lanes = full_vgl ? shape.electrons : 0;
  const std::size_t planes = checkedStorageSum(
      1, checkedStorageSum(gradient_lanes, laplacian_lanes,
                           "PsiFormer packed plane extent overflowed"),
      "PsiFormer packed plane extent overflowed");
  const std::size_t orbital_channels = checkedStorageProduct(
      shape.determinants, shape.electrons,
      "PsiFormer packed orbital width overflowed");
  const std::size_t width = std::max(
      {shape.input_width, shape.feature_width, orbital_channels});
  return checkedStorageProduct(
      checkedStorageProduct(
          checkedStorageProduct(tile, planes,
                                "PsiFormer packed tile extent overflowed"),
          shape.electrons, "PsiFormer packed row extent overflowed"),
      width, "PsiFormer packed element extent overflowed");
}

/// Estimate complete retained direct-batch storage for a capacity plan.
inline DirectBatchStorageRequirement directBatchStorageRequirement(
    const PsiFormerStorageShape& shape,
    const DirectBatchCapacityPlan& plan)
{
  const std::size_t sparse_size = checkedStorageSum(
      plan.logical.sparse_references, plan.logical.sparse_replacements,
      "PsiFormer sparse logical extent overflowed");
  const std::size_t value_logical = std::max(plan.logical.value_dense, sparse_size);
  const std::size_t maximum_logical = std::max(
      {value_logical, plan.logical.full_vgl, plan.logical.active_gradient});
  const std::size_t maximum_dense = std::max(
      {plan.logical.value_dense, plan.logical.full_vgl,
       plan.logical.active_gradient});
  const std::size_t value_tile = std::min(plan.tile.value, value_logical);
  const std::size_t full_tile = std::min(plan.tile.full_vgl,
                                         plan.logical.full_vgl);
  const std::size_t active_tile = std::min(plan.tile.active_gradient,
                                           plan.logical.active_gradient);
  if ((value_logical != 0 && plan.tile.value == 0) ||
      (plan.logical.full_vgl != 0 && plan.tile.full_vgl == 0) ||
      (plan.logical.active_gradient != 0 && plan.tile.active_gradient == 0))
    throw std::invalid_argument(
        "PsiFormer used batch modes require positive tile capacities");

  DirectBatchStorageRequirement result;
  const std::size_t position_scalars = checkedStorageProduct(
      checkedStorageProduct(maximum_dense, shape.electrons,
                            "PsiFormer dense logical extent overflowed"),
      3, "PsiFormer dense logical extent overflowed");
  result.dense_logical = checkedStorageSum(
      checkedStorageBytes<double>(position_scalars,
                                  "PsiFormer dense position bytes overflowed"),
      checkedStorageBytes<unsigned char>(position_scalars,
                                         "PsiFormer dense readiness bytes overflowed"),
      "PsiFormer dense logical bytes overflowed");

  const std::size_t reference_scalars = checkedStorageProduct(
      checkedStorageProduct(plan.logical.sparse_references, shape.electrons,
                            "PsiFormer sparse reference extent overflowed"),
      3, "PsiFormer sparse reference extent overflowed");
  std::size_t sparse_bytes = checkedStorageBytes<double>(
      reference_scalars, "PsiFormer sparse reference bytes overflowed");
  addStorageBytes(sparse_bytes, checkedStorageBytes<unsigned char>(
                                      reference_scalars,
                                      "PsiFormer sparse readiness bytes overflowed"),
                  "PsiFormer sparse logical bytes overflowed");
  addStorageBytes(sparse_bytes, checkedStorageBytes<std::size_t>(
                                      checkedStorageProduct(
                                          2, plan.logical.sparse_replacements,
                                          "PsiFormer replacement metadata overflowed"),
                                      "PsiFormer replacement metadata bytes overflowed"),
                  "PsiFormer sparse logical bytes overflowed");
  addStorageBytes(sparse_bytes, checkedStorageBytes<double>(
                                      checkedStorageProduct(
                                          3, plan.logical.sparse_replacements,
                                          "PsiFormer replacement position extent overflowed"),
                                      "PsiFormer replacement position bytes overflowed"),
                  "PsiFormer sparse logical bytes overflowed");
  addStorageBytes(sparse_bytes, checkedStorageBytes<unsigned char>(
                                      plan.logical.sparse_replacements,
                                      "PsiFormer replacement readiness bytes overflowed"),
                  "PsiFormer sparse logical bytes overflowed");
  result.sparse_logical = sparse_bytes;

  std::size_t output_bytes = checkedStorageBytes<double>(
      checkedStorageProduct(6, maximum_logical,
                            "PsiFormer value output extent overflowed"),
      "PsiFormer value output bytes overflowed");
  addStorageBytes(output_bytes, checkedStorageBytes<std::size_t>(
                                      checkedStorageProduct(
                                          2, maximum_logical,
                                          "PsiFormer version output extent overflowed"),
                                      "PsiFormer version output bytes overflowed"),
                  "PsiFormer output bytes overflowed");
  const std::size_t gradient_high_water = std::max(
      checkedStorageProduct(
          checkedStorageProduct(plan.logical.full_vgl, shape.electrons,
                                "PsiFormer full gradient extent overflowed"),
          3, "PsiFormer full gradient extent overflowed"),
      checkedStorageProduct(plan.logical.active_gradient, 3,
                            "PsiFormer active gradient extent overflowed"));
  addStorageBytes(output_bytes, checkedStorageBytes<double>(
                                      checkedStorageProduct(
                                          2, gradient_high_water,
                                          "PsiFormer gradient output extent overflowed"),
                                      "PsiFormer gradient output bytes overflowed"),
                  "PsiFormer output bytes overflowed");
  addStorageBytes(output_bytes, checkedStorageBytes<double>(
                                      checkedStorageProduct(
                                          checkedStorageProduct(
                                              4, plan.logical.full_vgl,
                                              "PsiFormer Laplacian output extent overflowed"),
                                          shape.electrons,
                                          "PsiFormer Laplacian output extent overflowed"),
                                      "PsiFormer Laplacian output bytes overflowed"),
                  "PsiFormer output bytes overflowed");
  result.logical_outputs = output_bytes;

  if (sparse_size != 0)
  {
    const std::size_t sparse_tile_scalars = checkedStorageProduct(
        checkedStorageProduct(value_tile, shape.electrons,
                              "PsiFormer sparse tile extent overflowed"),
        3, "PsiFormer sparse tile extent overflowed");
    result.sparse_tile_positions = checkedStorageBytes<double>(
        sparse_tile_scalars, "PsiFormer sparse tile bytes overflowed");
  }

  if (value_tile != 0)
  {
    const std::size_t feature = checkedStorageProduct(
        checkedStorageProduct(value_tile, shape.electrons,
                              "PsiFormer value tile row extent overflowed"),
        shape.feature_width, "PsiFormer value tile feature extent overflowed");
    const std::size_t raw = checkedStorageProduct(
        checkedStorageProduct(value_tile, shape.electrons,
                              "PsiFormer value tile row extent overflowed"),
        shape.input_width, "PsiFormer value tile input extent overflowed");
    const std::size_t square = checkedStorageProduct(
        shape.electrons, shape.electrons,
        "PsiFormer value tile electron-square extent overflowed");
    const std::size_t attention = checkedStorageProduct(
        checkedStorageProduct(value_tile, shape.attention_heads,
                              "PsiFormer value tile attention extent overflowed"),
        square, "PsiFormer value tile attention extent overflowed");
    const std::size_t orbitals = checkedStorageProduct(
        checkedStorageProduct(value_tile, shape.determinants,
                              "PsiFormer value tile orbital extent overflowed"),
        square, "PsiFormer value tile orbital extent overflowed");
    std::size_t elements = raw;
    addStorageBytes(elements, checkedStorageProduct(
                                  8, feature,
                                  "PsiFormer value tile feature extent overflowed"),
                    "PsiFormer value tile element extent overflowed");
    addStorageBytes(elements, attention,
                    "PsiFormer value tile element extent overflowed");
    addStorageBytes(elements, checkedStorageProduct(
                                  2, orbitals,
                                  "PsiFormer value tile orbital extent overflowed"),
                    "PsiFormer value tile element extent overflowed");
    result.value_tile = checkedStorageBytes<double>(
        elements, "PsiFormer value tile vector bytes overflowed");
    addStorageBytes(result.value_tile,
                    checkedStorageProduct(
                        value_tile,
                        geometryStorageRequirement(shape.electrons, shape.nuclei),
                        "PsiFormer value tile geometry bytes overflowed"),
                    "PsiFormer value tile bytes overflowed");
    addStorageBytes(result.value_tile,
                    checkedStorageProduct(
                        value_tile,
                        determinantStorageRequirement(shape.determinants,
                                                      shape.electrons),
                        "PsiFormer value tile determinant bytes overflowed"),
                    "PsiFormer value tile bytes overflowed");
  }

  result.full_vgl_tile = checkedStorageProduct(
      full_tile, spatialWorkspaceStorageRequirement(shape, true),
      "PsiFormer full-VGL tile bytes overflowed");
  result.active_gradient_tile = checkedStorageProduct(
      active_tile, spatialWorkspaceStorageRequirement(shape, false),
      "PsiFormer active-gradient tile bytes overflowed");
  const std::size_t shared_elements = std::max(
      spatialPackedElements(shape, full_tile, true),
      spatialPackedElements(shape, active_tile, false));
  result.shared_spatial_arena = checkedStorageBytes<double>(
      checkedStorageProduct(2, shared_elements,
                            "PsiFormer shared spatial arena extent overflowed"),
      "PsiFormer shared spatial arena bytes overflowed");
  return result;
}

} // namespace pf

#endif // QMCPLUSPLUS_PSIFORMER_STORAGE_REQUIREMENTS_H
