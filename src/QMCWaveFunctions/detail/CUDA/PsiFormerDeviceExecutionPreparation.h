//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceExecutionPreparation.h
 * @brief Explicit per-owner target profiles for existing numerical consumers.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_EXECUTION_PREPARATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_EXECUTION_PREPARATION_H

#include "PsiFormerDeviceTargetKinetic.h"

#include <array>
#include <limits>
#include <ostream>
#include <string_view>
#include <tuple>
#include <utility>

namespace qmcplusplus::psiformer::device
{

/** Immutable independent VGL and kinetic target preparations; zero keeps legacy.
 * Limits are per numeric owner/request, not a process/device aggregate budget.
 * Shared images, other crowds, solver channels and vendor reserves remain separate.
 */
struct DeviceExecutionPreparation
{
  TargetVglPreparation vgl;
  TargetKineticPreparation kinetic;
  /// Optional per-provider immutable sample-coordinate owner, not an aggregate cap.
  std::size_t retained_coordinate_workspace_budget_bytes = 0;
  /// Optional standalone per-crowd injected-native-draw sampler owner, not an aggregate cap.
  std::size_t metropolis_workspace_budget_bytes = 0;
  /// Explicit drift additions only: retained VALUE delta, reverse and Green buffer.
  std::size_t metropolis_drift_budget_bytes = 0;
  /// Optional per-rank optimizer recurrence scratch; not provider/crowd storage.
  std::size_t adam_workspace_budget_bytes = 0;
  /// Opt-in work/memory selection; resolved VGL is reselected for each actual B.
  bool select_vgl_targets = false, select_kinetic_targets = false;
  /// Explicit FP64 inference overlap only; prepared inactive owners stay charged.
  std::size_t active_inference_crowds = 1;
};

/// Parse a selection request without reading environment or changing explicit defaults.
inline std::pair<bool, bool> parseDeviceTargetSelection(const char* value)
{
  if (!value || std::string_view(value) == "off")
    return {false, false};
  const std::string_view policy(value);
  if (policy == "vgl" || policy == "kinetic" || policy == "both")
    return {policy != "kinetic", policy != "vgl"};
  throw std::invalid_argument("PSIFORMER_GPU_TARGET_SELECTION must be off, vgl, kinetic or both");
}

/// Choose only among ten bounded K candidates using exact owner bytes and a work proxy.
template<class Estimate>
inline std::size_t selectDeviceTargetCapacity(std::size_t electrons, std::size_t bytes,
                                             std::size_t work, Estimate&& estimate)
{
  if (!electrons || !bytes || !work || bytes == std::numeric_limits<std::size_t>::max() ||
      work == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("GPU target selection needs positive finite owner/request limits");
  const std::array<std::size_t, 10> candidates{1, 2, 4, 8, 16, 32, 64, 128, 256, electrons};
  std::size_t selected = 0, previous = 0;
  std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> best;
  for (const std::size_t k : candidates)
  {
    if (k > electrons || k == previous)
      continue;
    previous = k;
    try
    {
      const auto cost = estimate(k);
      if (cost.device_workspace_bytes > bytes || cost.charged_scalar_work_units > work)
        continue;
      const auto rank = std::make_tuple(cost.charged_scalar_work_units,
          cost.device_workspace_bytes, cost.host_workspace_bytes, k);
      if (!selected || rank < best)
      {
        selected = k;
        best = rank;
      }
    }
    catch (const std::length_error&)
    {
      // Checked extent/work overflow makes this candidate infeasible, never wrapped.
    }
  }
  if (!selected)
    throw std::length_error("No GPU target candidate fits the per-owner/request limits");
  return selected;
}

/// Resolve selected families using actual VGL capacity; kinetic owners are always B=1.
inline DeviceExecutionPreparation resolveDeviceTargetSelection(const PsiFormerExecutionPlan& plan,
    DeviceExecutionPreparation preparation, std::size_t vgl_capacity = 1)
{
  if (!preparation.select_vgl_targets && !preparation.select_kinetic_targets)
    return preparation;
  const auto& environment = plan.environment();
  if (environment.parameter_scalar_domain != ScalarDomain::REAL ||
      environment.compute_scalar_domain != ScalarDomain::REAL ||
      environment.amplitude_scalar_domain != ScalarDomain::REAL || !environment.fixed_nuclei ||
      (environment.boundary != BoundaryCondition::OPEN && environment.boundary != BoundaryCondition::PERIODIC) ||
      (environment.boundary == BoundaryCondition::PERIODIC &&
       environment.geometry_feature_policy != GeometryFeaturePolicy::PERIODIC_TORUS_V1))
    throw std::invalid_argument("GPU target selection requires qualified fixed-ion real target geometry");
  const auto ne = plan.modelShape().electrons();
  if (preparation.select_vgl_targets)
    preparation.vgl.max_targets = selectDeviceTargetCapacity(ne, preparation.vgl.device_workspace_budget_bytes,
        preparation.vgl.scalar_work_budget, [&](std::size_t k) { return estimateTargetVglCost(plan, vgl_capacity, k); });
  if (preparation.select_kinetic_targets)
    preparation.kinetic.max_targets = selectDeviceTargetCapacity(ne, preparation.kinetic.device_workspace_budget_bytes,
        preparation.kinetic.scalar_work_budget, [&](std::size_t k) {
          return estimateTargetKineticCost(plan, k, preparation.kinetic.lifted_channel_slots);
        });
  return preparation;
}

/// Report one metadata-ranked owner; neither the work proxy nor byte cap predicts total runtime/fit.
template<class Cost>
inline void reportDeviceTargetSelection(std::ostream& out, const char* operation,
    const PsiFormerExecutionPlan& plan, std::size_t batch, std::size_t targets, const Cost& cost,
    std::size_t bytes, std::size_t work)
{
  const auto n = plan.modelShape().electrons();
  std::size_t candidates = 0;
  for (std::size_t k = 1; k <= 256; k *= 2)
    candidates += k <= n;
  candidates += n > 256 || (n & (n - 1)) != 0;
  out << "  PsiFormer GPU target selection: policy=deterministic_work_memory_v1, operation=" << operation
      << ", boundary=" << (plan.environment().boundary == BoundaryCondition::OPEN ? "open" : "periodic")
      << ", Ne=" << n << ", B=" << batch << ", candidates=" << candidates << ", selected_K=" << targets
      << ", tiles=" << cost.tile_count << ", owner_device_bytes=" << cost.device_workspace_bytes
      << ", separate_host_bytes=" << cost.host_workspace_bytes << ", scalar_contraction_sites=" << cost.charged_scalar_work_units
      << ", owner_device_byte_limit=" << bytes << ", request_work_limit=" << work
      << "; excludes images, VALUE graph, other crowds, score/JVP/solver/ECP/teacher products,"
         " retained coordinates, sampler/Adam, vendor and allocator reserves"
      << "; calibrated_fastest=no; aggregate_process_fit=no\n";
}

/// Reject partial profiles and impossible per-owner limits before graph reserves.
inline void validateDeviceExecutionPreparation(const PsiFormerExecutionPlan& plan,
    const DeviceExecutionPreparation& preparation, std::size_t vgl_capacity = 1)
{
  if (!preparation.active_inference_crowds ||
      preparation.active_inference_crowds == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("Active GPU inference crowds must be positive and finite");
  const auto validate = [](std::size_t targets, std::size_t bytes, std::size_t work) {
    if (targets == 0 ? (bytes != 0 || work != 0) : (bytes == 0 || work == 0))
      throw std::invalid_argument("Device target preparation requires positive targets and both per-owner budgets");
  };
  validate(preparation.vgl.max_targets, preparation.vgl.device_workspace_budget_bytes,
      preparation.vgl.scalar_work_budget);
  validate(preparation.kinetic.max_targets, preparation.kinetic.device_workspace_budget_bytes,
      preparation.kinetic.scalar_work_budget);
  if (preparation.kinetic.lifted_channel_slots && !preparation.kinetic.max_targets)
    throw std::invalid_argument("Lifted GEMM channels require explicit kinetic targets");
  if (preparation.vgl.max_targets)
  {
    const auto cost = estimateTargetVglCost(plan, vgl_capacity, preparation.vgl.max_targets);
    if (cost.device_workspace_bytes > preparation.vgl.device_workspace_budget_bytes ||
        cost.charged_scalar_work_units > preparation.vgl.scalar_work_budget)
      throw std::length_error("Device target VGL exceeds its per-owner workspace/request work limit");
  }
  if (preparation.kinetic.max_targets)
  {
    const auto cost = estimateTargetKineticCost(plan, preparation.kinetic.max_targets,
        preparation.kinetic.lifted_channel_slots);
    if (cost.device_workspace_bytes > preparation.kinetic.device_workspace_budget_bytes ||
        cost.charged_scalar_work_units > preparation.kinetic.scalar_work_budget)
      throw std::length_error("Device target kinetic exceeds its per-owner workspace/request work limit");
  }
}

} // namespace qmcplusplus::psiformer::device
#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_EXECUTION_PREPARATION_H
