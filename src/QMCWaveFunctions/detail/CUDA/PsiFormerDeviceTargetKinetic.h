//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceTargetKinetic.h
 * @brief Separately admitted max-K retained kinetic response for one configuration.
 *
 * The immutable image is shared; this owner retains only one tile's forward and
 * reverse graph and one canonical P sum. Ordinary score remains an outer operation.
 * Scalar determinant response is a correctness reference, not large-Ne throughput.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_TARGET_KINETIC_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_TARGET_KINETIC_H

#include "PsiFormerDeviceKineticReverse.h"

namespace qmcplusplus::psiformer::device
{

/// Immutable one-configuration target capacity and combined numeric admission limits.
struct TargetKineticPreparation
{
  std::size_t max_targets = 0;
  std::size_t device_workspace_budget_bytes = 0;
  std::size_t scalar_work_budget = 0;
  // Zero preserves reference factors; 1..16 explicitly admits GEMM channel scratch.
  std::size_t lifted_channel_slots = 0;
};

/// Predicted persistent host/device storage and complete-request scalar work.
struct TargetKineticCost
{
  std::size_t device_workspace_bytes = 0;
  std::size_t host_workspace_bytes = 0;
  std::size_t charged_scalar_work_units = 0;
  std::size_t tile_count = 0;
};

/** Exact persistent forward/reverse buffers, excluding shared images and vendors.
 * Work proxy counts 26*D*Ne^4 + 4*D*ceil(Ne/K)*Ne^3 scalar contraction sites;
 * it remains conservative for optional GEMM products: a reproducible admission
 * limit, not a backend FLOP or elapsed-time prediction.
 */
inline TargetKineticCost estimateTargetKineticCost(const PsiFormerExecutionPlan& plan,
                                                   std::size_t targets,
                                                   std::size_t lifted_channel_slots = 0)
{
  const auto forward = estimateRetainedTargetForwardCost(plan, targets);
  const auto reverse = estimateTargetKineticReverseCost(plan, targets, lifted_channel_slots);
  const auto add = [](std::size_t a, std::size_t b) {
    return spatial_detail::checkedAdd(a, b, "Target kinetic storage/work overflow");
  };
  const auto mul = [](std::size_t a, std::size_t b) {
    return spatial_detail::checkedProduct(a, b, "Target kinetic work overflow");
  };
  const auto& shape = plan.modelShape();
  const auto ne = shape.electrons(), cube = mul(ne, mul(ne, ne));
  TargetKineticCost cost;
  cost.device_workspace_bytes = add(forward.device_workspace_bytes, reverse.device_workspace_bytes);
  cost.host_workspace_bytes = add(forward.host_workspace_bytes, reverse.host_workspace_bytes);
  cost.tile_count = forward.tile_count;
  cost.charged_scalar_work_units = add(mul(26, mul(shape.determinants, mul(ne, cube))),
      mul(4, mul(shape.determinants, mul(cost.tile_count, cube))));
  return cost;
}

/** One logical request streams ascending targets and publishes only its final P sum.
 * No full reference owner or score tape is constructed as a fallback. Callers keep
 * the usual image/publication lease for an evaluation and consume its response
 * before rebind or another request. Concurrent use is unsupported.
 */
class PreparedDeviceTargetKinetic
{
public:
  /// Admit the combined graph before constructing either numeric workspace.
  PreparedDeviceTargetKinetic(std::shared_ptr<const PreparedDeviceModel> model,
                              TargetKineticPreparation preparation,
                              compute::Queue<PlatformKind::CUDA>& queue,
                              AcceleratorBlasHandle& blas,
                              InferenceStatusPolicy policy = configuredInferenceStatusPolicy())
      : PreparedDeviceTargetKinetic(model, preparation, queue, blas, policy, admit(model, preparation))
  {}

  /// Coefficients contain one weight, all physical masses, and optional 3Ne drift.
  const double* evaluate(const std::vector<double>& positions, const std::vector<double>& weights,
                         const std::vector<double>& inverse_masses = {},
                         const std::vector<double>& total_gradient = {}, bool gradient_only = false)
  {
    ready_ = false;
    reverse_.ready_ = false;
    const auto* image = forward_.modelImage().get();
    try
    {
      reverse_.beginRequest(weights, inverse_masses, total_gradient, gradient_only, 1);
      forward_.evaluateRetainedTargets(positions, [&](const auto& view) { reverse_.accumulateTarget(view); });
      if (forward_.admittedTargetTiles() != cost_.tile_count || forward_.modelImage().get() != image)
        throw std::logic_error("Target kinetic request did not complete its admitted image/coverage");
      const auto* response = reverse_.finishRequest();
      ready_ = true;
      return response;
    }
    catch (...)
    {
      reverse_.ready_ = false;
      throw;
    }
  }

  /// Fixed-outer-phase imaginary cross term uses complete drift and no trace seed.
  const double* evaluateGradientResponse(const std::vector<double>& positions, const std::vector<double>& weights,
      const std::vector<double>& inverse_masses, const std::vector<double>& total_gradient)
  {
    return evaluate(positions, weights, inverse_masses, total_gradient, true);
  }

  /// A failed tile/request never exposes a partial canonical response.
  void downloadResponse(std::vector<double>& destination) const
  {
    if (!ready_)
      throw std::logic_error("Target kinetic response is unavailable");
    reverse_.downloadResponse(destination);
  }

  /// Stream completed intervals without replay or a second P allocation.
  void downloadResponseChunk(std::size_t begin, double* destination, std::size_t count) const
  {
    if (!ready_)
      throw std::logic_error("Target kinetic response is unavailable");
    reverse_.downloadResponseChunk(begin, destination, count);
  }

  /// Fixed-shape successor binding invalidates response and all retained views.
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> successor)
  {
    ready_ = false;
    reverse_.ready_ = false;
    forward_.modelImage()->validateSuccessor(successor);
    forward_.rebindModel(successor);
    reverse_.rebindModel(std::move(successor));
  }

  /// Return the immutable pre-allocation combined cost descriptor.
  const TargetKineticCost& cost() const noexcept { return cost_; }

  /// Return the explicitly admitted max-K profile and its original limits.
  const TargetKineticPreparation& preparation() const noexcept { return preparation_; }

  /// Identify the one-upload coordinate request independently of tile count.
  std::size_t evaluationGeneration() const noexcept { return forward_.evaluationGeneration(); }

  /// Report completed immediate reverses without exposing partial numeric results.
  std::size_t admittedTargetTiles() const noexcept { return forward_.admittedTargetTiles(); }

  /// Count both retained owners, excluding the separately shared model image.
  std::size_t deviceWorkspaceBytes() const
  {
    return spatial_detail::checkedAdd(forward_.deviceWorkspaceBytes(), reverse_.deviceWorkspaceBytes(),
        "Target kinetic owned device storage overflow");
  }
  /// Count explicit host staging vectors independently of device storage.
  std::size_t hostWorkspaceBytes() const
  {
    return spatial_detail::checkedAdd(forward_.hostWorkspaceBytes(), reverse_.hostWorkspaceBytes(),
        "Target kinetic owned host storage overflow");
  }

private:
  /// Validate immutable admission before member construction can reserve a graph.
  static TargetKineticCost admit(const std::shared_ptr<const PreparedDeviceModel>& model,
                                  const TargetKineticPreparation& preparation)
  {
    if (!model || preparation.max_targets == 0 || preparation.device_workspace_budget_bytes == 0 ||
        preparation.scalar_work_budget == 0)
      throw std::invalid_argument("Target kinetic requires a model and positive explicit capacities/budgets");
    model->validateCurrentDevice();
    const auto cost = estimateTargetKineticCost(model->plan(), preparation.max_targets,
        preparation.lifted_channel_slots);
    if (cost.device_workspace_bytes > preparation.device_workspace_budget_bytes ||
        cost.charged_scalar_work_units > preparation.scalar_work_budget)
      throw std::length_error("Target kinetic exceeds its combined owned-device or whole-request work budget");
    return cost;
  }

  /// Only this admitted construction can reach Inference's retained-target seam.
  PreparedDeviceTargetKinetic(std::shared_ptr<const PreparedDeviceModel> model,
                              TargetKineticPreparation preparation,
                              compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas,
                              InferenceStatusPolicy policy, TargetKineticCost cost)
      : preparation_(preparation), cost_(cost),
        forward_(model, 1, queue, blas, SpatialJetMode::FULL_VGL, false, true, policy,
            TargetVglPreparation{preparation.max_targets, cost.device_workspace_bytes, cost.charged_scalar_work_units}, true),
        reverse_(std::move(model), 1, queue, blas, preparation.max_targets, preparation.lifted_channel_slots)
  {
    if (deviceWorkspaceBytes() != cost_.device_workspace_bytes || hostWorkspaceBytes() != cost_.host_workspace_bytes)
      throw std::logic_error("Target kinetic combined storage differs from its pre-allocation estimate");
  }

  const TargetKineticPreparation preparation_;
  const TargetKineticCost cost_;
  PreparedDeviceInference forward_;
  PreparedDeviceKineticReverse reverse_;
  bool ready_ = false;
};

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_TARGET_KINETIC_H
