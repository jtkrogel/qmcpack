//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerWorkspace.cpp
 * @brief Capacity planning and aligned-storage management for PsiFormer evaluation.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerWorkspace.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace qmcplusplus::psiformer
{
namespace
{

/// Multiply nonnegative capacities while detecting size_t overflow.
std::size_t checkedMultiply(std::size_t left, std::size_t right, const char* description)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(std::string("PsiFormer workspace size overflow in ") + description);
  return left * right;
}

/// Add nonnegative capacities while detecting size_t overflow.
std::size_t checkedAdd(std::size_t left, std::size_t right, const char* description)
{
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(std::string("PsiFormer workspace size overflow in ") + description);
  return left + right;
}

/// Return a product of three capacities with overflow checks.
std::size_t checkedProduct(std::size_t first, std::size_t second, std::size_t third, const char* description)
{
  return checkedMultiply(checkedMultiply(first, second, description), third, description);
}

/// Round an element count so every region begins at the configured SIMD alignment.
template<class Scalar>
std::size_t alignedElementCount(std::size_t elements)
{
  constexpr std::size_t alignment_elements = QMC_SIMD_ALIGNMENT / sizeof(Scalar);
  static_assert(QMC_SIMD_ALIGNMENT % sizeof(Scalar) == 0,
                "PsiFormer workspace alignment must be a multiple of the scalar size");
  const std::size_t remainder = elements % alignment_elements;
  return remainder == 0 ? elements : checkedAdd(elements, alignment_elements - remainder, "alignment padding");
}

/// Report whether one evaluation mode propagates first spatial derivatives.
bool needsFirstDerivatives(EvaluationMode mode)
{
  return mode == EvaluationMode::ACTIVE_ELECTRON_GRADIENT || mode == EvaluationMode::FULL_VGL ||
      mode == EvaluationMode::PARAMETER_SCORE_AND_KINETIC;
}

/// Report whether one evaluation mode propagates diagonal second derivatives.
bool needsSecondDerivatives(EvaluationMode mode)
{
  return mode == EvaluationMode::FULL_VGL || mode == EvaluationMode::PARAMETER_SCORE_AND_KINETIC;
}

/// Report whether one evaluation mode performs a parameter reverse product.
bool needsParameterReverse(EvaluationMode mode)
{
  return mode == EvaluationMode::PARAMETER_SCORE || mode == EvaluationMode::PARAMETER_SCORE_AND_KINETIC ||
      mode == EvaluationMode::VIRTUAL_WEIGHTED_PARAMETER_VJP;
}

} // namespace

// Compute conservative direct-kernel high-water marks without allocating storage.
WorkspaceRequirements makeWorkspaceRequirements(const PsiFormerExecutionPlan& plan, WorkspaceWorkload workload)
{
  if (workload.walkers == 0)
    throw std::invalid_argument("PsiFormer workspace requires at least one walker");
  if ((workload.mode == EvaluationMode::VIRTUAL_RATIOS ||
       workload.mode == EvaluationMode::VIRTUAL_WEIGHTED_PARAMETER_VJP) &&
      workload.virtual_positions == 0)
    throw std::invalid_argument("PsiFormer virtual-move workspace requires at least one virtual position");

  const ModelShape& model      = plan.modelShape();
  const std::size_t electrons  = model.electrons();
  const std::size_t dimensions = model.feature_dimension;
  const std::size_t virtual_multiplier =
      workload.mode == EvaluationMode::VIRTUAL_RATIOS ||
          workload.mode == EvaluationMode::VIRTUAL_WEIGHTED_PARAMETER_VJP
      ? workload.virtual_positions
      : 1;
  const std::size_t samples = checkedMultiply(workload.walkers, virtual_multiplier, "sample batch");

  // Features cover two residual operands plus fused Q/K/V projections.  The
  // direct executor is expected to reuse these slots block by block.
  const std::size_t feature_matrix = checkedMultiply(electrons, dimensions, "feature matrix");
  const std::size_t features_per_sample = checkedMultiply(feature_matrix, 5, "feature buffers");

  // Attention stores one logits/softmax matrix per head and one context matrix.
  const std::size_t attention_matrix =
      checkedProduct(model.attention_heads, electrons, electrons, "attention matrix");
  const std::size_t attention_per_sample = checkedAdd(
      checkedMultiply(attention_matrix, 2, "attention logits and weights"), feature_matrix, "attention context");

  const std::size_t orbital_per_sample =
      checkedProduct(model.determinants, electrons, electrons, "orbital matrices");
  const std::size_t determinant_per_sample = checkedMultiply(
      model.determinants,
      checkedAdd(checkedMultiply(model.spin_up_electrons, model.spin_up_electrons, "up determinant"),
                 checkedMultiply(model.spin_down_electrons, model.spin_down_electrons, "down determinant"),
                 "spin determinant matrices"),
      "determinant batch");

  WorkspaceRequirements requirements;
  requirements.workload = workload;
  auto& elements        = requirements.elements;
  // Geometry is owned by the fixed-size PsiFormerGeometryCache rather than
  // duplicated in the generic scalar arena.  A future periodic policy may
  // supply a different cache while retaining this zero-copy boundary.
  elements[static_cast<std::size_t>(WorkspaceRegion::GEOMETRY)] = 0;
  elements[static_cast<std::size_t>(WorkspaceRegion::FEATURES)] =
      checkedMultiply(samples, features_per_sample, "batched features");
  elements[static_cast<std::size_t>(WorkspaceRegion::ATTENTION)] =
      checkedMultiply(samples, attention_per_sample, "batched attention");
  elements[static_cast<std::size_t>(WorkspaceRegion::ORBITALS)] =
      checkedMultiply(samples, orbital_per_sample, "batched orbitals");
  elements[static_cast<std::size_t>(WorkspaceRegion::DETERMINANTS)] =
      checkedMultiply(samples, determinant_per_sample, "batched determinants");

  // Coordinate lanes are explicit so full VGL can be tiled without changing
  // the workspace API.  A zero lane request selects the exact current-mode
  // default: three for an active electron or all 3N Cartesian lanes otherwise.
  std::size_t derivative_lanes = workload.derivative_lanes;
  if (needsFirstDerivatives(workload.mode) && derivative_lanes == 0)
    derivative_lanes = workload.mode == EvaluationMode::ACTIVE_ELECTRON_GRADIENT ? 3 : 3 * electrons;
  if (!needsFirstDerivatives(workload.mode) && derivative_lanes != 0)
    throw std::invalid_argument("PsiFormer derivative lanes were requested for a value-only workspace mode");

  const std::size_t differentiable_primal = checkedAdd(
      checkedAdd(features_per_sample, attention_per_sample, "differentiable feature state"),
      checkedAdd(orbital_per_sample, determinant_per_sample, "differentiable determinant state"),
      "differentiable state");
  if (needsFirstDerivatives(workload.mode))
    elements[static_cast<std::size_t>(WorkspaceRegion::FIRST_DERIVATIVES)] =
        checkedProduct(samples, derivative_lanes, differentiable_primal, "first derivatives");
  if (needsSecondDerivatives(workload.mode))
    elements[static_cast<std::size_t>(WorkspaceRegion::SECOND_DERIVATIVES)] =
        checkedProduct(samples, derivative_lanes, differentiable_primal, "second derivatives");

  if (needsParameterReverse(workload.mode))
  {
    elements[static_cast<std::size_t>(WorkspaceRegion::REVERSE_ADJOINTS)] =
        checkedMultiply(samples, differentiable_primal, "reverse adjoints");
    const std::size_t output_batches = workload.mode == EvaluationMode::VIRTUAL_WEIGHTED_PARAMETER_VJP ? 1 : samples;
    elements[static_cast<std::size_t>(WorkspaceRegion::PARAMETER_OUTPUT)] =
        checkedMultiply(output_batches, plan.parameterCount(), "parameter output");
  }

  // Reductions hold determinant signs/logarithms and per-sample scratch.  The
  // deliberately small fixed multiplier counts typed Scalar objects and is
  // therefore independent of their real or future complex representation.
  elements[static_cast<std::size_t>(WorkspaceRegion::REDUCTIONS)] =
      checkedProduct(samples, model.determinants, 4, "determinant reductions");
  return requirements;
}

// Grow high-water capacities once and rebuild the aligned region layout.
template<class Scalar>
bool BasicPsiFormerWorkspace<Scalar>::prepare(const WorkspaceRequirements& requirements)
{
  bool growth_required = false;
  std::array<std::size_t, WORKSPACE_REGION_COUNT> new_capacities = capacities_;
  for (std::size_t region = 0; region < WORKSPACE_REGION_COUNT; ++region)
    if (requirements.elements[region] > new_capacities[region])
    {
      new_capacities[region] = requirements.elements[region];
      growth_required        = true;
    }
  if (!growth_required)
    return false;

  std::array<std::size_t, WORKSPACE_REGION_COUNT> new_offsets{};
  std::size_t total_elements = 0;
  for (std::size_t region = 0; region < WORKSPACE_REGION_COUNT; ++region)
  {
    total_elements      = alignedElementCount<Scalar>(total_elements);
    new_offsets[region] = total_elements;
    total_elements      = checkedAdd(total_elements, new_capacities[region], "workspace storage");
  }

  // Constructing a replacement vector makes allocation-generation semantics
  // deterministic: every successful growth invalidates all previous views.
  aligned_vector<Scalar> replacement(total_elements);
  storage_.swap(replacement);
  capacities_ = new_capacities;
  offsets_    = new_offsets;
  ++allocation_generation_;
  ++layout_generation_;
  return true;
}

// Assemble mutable views over the current region high-water marks.
template<class Scalar>
typename BasicPsiFormerWorkspace<Scalar>::WorkspaceView BasicPsiFormerWorkspace<Scalar>::view()
{
  WorkspaceView result;
  result.allocation_generation = allocation_generation_;
  result.layout_generation     = layout_generation_;
  for (std::size_t region = 0; region < WORKSPACE_REGION_COUNT; ++region)
  {
    Scalar* region_data = storage_.empty() ? nullptr : storage_.data() + offsets_[region];
    result.regions[region] = ArrayView<Scalar>(region_data, capacities_[region]);
  }
  return result;
}

// Assemble const views over the current region high-water marks.
template<class Scalar>
typename BasicPsiFormerWorkspace<Scalar>::ConstWorkspaceView BasicPsiFormerWorkspace<Scalar>::view() const
{
  ConstWorkspaceView result;
  result.allocation_generation = allocation_generation_;
  result.layout_generation     = layout_generation_;
  for (std::size_t region = 0; region < WORKSPACE_REGION_COUNT; ++region)
  {
    const Scalar* region_data = storage_.empty() ? nullptr : storage_.data() + offsets_[region];
    result.regions[region]    = ArrayView<const Scalar>(region_data, capacities_[region]);
  }
  return result;
}

// Instantiate only the real scalar workspace supported by this implementation.
template class BasicPsiFormerWorkspace<double>;

// Provide readable region names for allocation reports and profiler output.
const char* workspaceRegionName(WorkspaceRegion region)
{
  switch (region)
  {
  case WorkspaceRegion::GEOMETRY:
    return "geometry";
  case WorkspaceRegion::FEATURES:
    return "features";
  case WorkspaceRegion::ATTENTION:
    return "attention";
  case WorkspaceRegion::ORBITALS:
    return "orbitals";
  case WorkspaceRegion::DETERMINANTS:
    return "determinants";
  case WorkspaceRegion::FIRST_DERIVATIVES:
    return "first_derivatives";
  case WorkspaceRegion::SECOND_DERIVATIVES:
    return "second_derivatives";
  case WorkspaceRegion::REVERSE_ADJOINTS:
    return "reverse_adjoints";
  case WorkspaceRegion::PARAMETER_OUTPUT:
    return "parameter_output";
  case WorkspaceRegion::REDUCTIONS:
    return "reductions";
  case WorkspaceRegion::COUNT:
    break;
  }
  return "unknown";
}

} // namespace qmcplusplus::psiformer
