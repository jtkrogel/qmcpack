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
};

/// Reject partial profiles and impossible per-owner limits before graph reserves.
inline void validateDeviceExecutionPreparation(const PsiFormerExecutionPlan& plan,
    const DeviceExecutionPreparation& preparation, std::size_t vgl_capacity = 1)
{
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
