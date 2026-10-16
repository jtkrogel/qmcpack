//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceTargetSelectionValidation.h
 * @brief Compact explicit/selected parity, independent of broad model validation.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_TARGET_SELECTION_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_TARGET_SELECTION_VALIDATION_H

#include "PsiFormerDeviceExecutionPreparation.h"

namespace qmcplusplus::psiformer::device
{

/// Compare the same qualified numerical owners under explicit and selected preparation.
inline void validateTargetSelectionNumerics(const std::shared_ptr<const PreparedDeviceModel>& image,
    const std::vector<double>& positions, compute::Queue<PlatformKind::CUDA>& queue,
    AcceleratorBlasHandle& blas, std::ostream& out)
{
  const auto& plan = image->plan();
  const auto k = std::min(std::size_t(4), plan.modelShape().electrons());
  const auto vgl = estimateTargetVglCost(plan, 1, k);
  const auto kinetic = estimateTargetKineticCost(plan, k);
  DeviceExecutionPreparation explicit_preparation;
  explicit_preparation.vgl = {k, vgl.device_workspace_bytes, vgl.charged_scalar_work_units};
  explicit_preparation.kinetic = {k, kinetic.device_workspace_bytes, kinetic.charged_scalar_work_units, 0};
  auto request = explicit_preparation;
  request.select_vgl_targets = request.select_kinetic_targets = true;
  request.vgl.max_targets = request.kinetic.max_targets = 0;
  const auto selected = resolveDeviceTargetSelection(plan, request);
  validateDeviceExecutionPreparation(plan, selected);
  if (selected.vgl.max_targets != k || selected.kinetic.max_targets != k)
    throw std::runtime_error("Compact selection did not match explicit exact-cost preparation");
  const auto compare = [](const std::vector<double>& actual, const std::vector<double>& expected) {
    if (actual.size() != expected.size())
      throw std::runtime_error("Selected target output shape differs from explicit preparation");
    for (std::size_t i = 0; i < actual.size(); ++i)
      if (!device_math::isFiniteBinary64(actual[i]) || !device_math::isFiniteBinary64(expected[i]) ||
          std::abs(actual[i] - expected[i]) > 2e-11 * (1 + std::abs(expected[i])))
        throw std::runtime_error("Selected target output differs from explicit preparation");
  };
  {
    PreparedDeviceInference manual(image, 1, queue, blas, SpatialJetMode::FULL_VGL,
        false, false, configuredInferenceStatusPolicy(), explicit_preparation.vgl);
    PreparedDeviceInference automatic(image, 1, queue, blas, SpatialJetMode::FULL_VGL,
        false, false, configuredInferenceStatusPolicy(), selected.vgl);
    manual.evaluate(positions);
    const auto n = plan.modelShape().electrons();
    std::vector<double> expected_phase(1), expected_jets(spatialJetSpanElements(manual.resultLayout())), expected_ratios(n);
    std::vector<double> phase(1), jets(expected_jets.size()), ratios(n);
    manual.downloadResults(expected_phase, expected_jets, expected_ratios);
    const auto before = deviceMemoryMetrics(0).snapshot();
    for (int repeat = 0; repeat < 2; ++repeat)
    {
      automatic.evaluate(positions);
      automatic.downloadResults(phase, jets, ratios);
      compare(phase, expected_phase);
      compare(jets, expected_jets);
      compare(ratios, expected_ratios);
    }
    if (deviceMemoryMetrics(0).snapshot().allocation_calls != before.allocation_calls)
      throw std::runtime_error("Selected VGL owner allocated during warmed requests");
  }
  {
    PreparedDeviceTargetKinetic manual(image, explicit_preparation.kinetic, queue, blas);
    PreparedDeviceTargetKinetic automatic(image, selected.kinetic, queue, blas);
    std::vector<double> expected(plan.parameterCount()), actual(plan.parameterCount());
    manual.evaluate(positions, {1.0});
    manual.downloadResponse(expected);
    const auto before = deviceMemoryMetrics(0).snapshot();
    for (int repeat = 0; repeat < 2; ++repeat)
    {
      automatic.evaluate(positions, {1.0});
      automatic.downloadResponse(actual);
      compare(actual, expected);
    }
    if (deviceMemoryMetrics(0).snapshot().allocation_calls != before.allocation_calls)
      throw std::runtime_error("Selected kinetic owner allocated during warmed requests");
  }
  out << "PASS deterministic_work_memory_v1 compact explicit/selected parity; Ne="
      << plan.modelShape().electrons() << ", K=" << k << "; warm_device_allocations=0\n";
}

} // namespace qmcplusplus::psiformer::device
#endif
