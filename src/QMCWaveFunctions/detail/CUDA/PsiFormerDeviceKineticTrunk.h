//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_KINETIC_TRUNK_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_KINETIC_TRUNK_H
#include "PsiFormerDeviceKineticReverse.h"
#include <utility>

namespace qmcplusplus::psiformer::device
{
/// Explicit private prefix/tape/reverse buffers, excluding image/vendor/object metadata.
struct KineticTrunkCost
{
  std::size_t forward_device_bytes, reverse_device_bytes, device_bytes, host_bytes;
};

/// Count only max-K real jets, retained blocks and one canonical P before allocation.
inline KineticTrunkCost estimateKineticTrunkCost(const PsiFormerExecutionPlan& plan, std::size_t targets)
{
  const auto& shape = plan.modelShape();
  const auto n = shape.electrons(), f = shape.feature_dimension, p = plan.parameterCount();
  if (!targets || targets > n || !p)
    throw std::invalid_argument("Kinetic trunk requires a positive physical max-K and canonical extent");
  const auto mul = [](auto a, auto b) { return spatial_detail::checkedProduct(a, b, "Kinetic trunk extent overflow"); };
  const auto add = [](auto a, auto b) { return spatial_detail::checkedAdd(a, b, "Kinetic trunk bytes overflow"); };
  const auto s = add(1, mul(4, targets)), u = mul(n, f), a = mul(shape.attention_heads, mul(n, n));
  const auto raw = mul(n, plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0]);
  const auto statuses = add(mul(u, sizeof(OpenSpatialStatus)),
      mul(mul(shape.attention_heads, n), sizeof(device_math::JetMathStatus)));
  const auto prefix = add(mul(8, add(mul(3, n), mul(s, add(add(raw, mul(8, u)), add(a, 1))))),
      add(statuses, add(sizeof(std::size_t), sizeof(InferenceStatusWord))));
  const auto tape = mul(8, mul(shape.attention_blocks, mul(s, add(mul(10, u), mul(2, a)))));
  const auto finite = mul(p / 4096 + (p % 4096 != 0), sizeof(DerivativeFiniteStatus));
  const auto reverse = add(mul(8, add(mul(s, add(add(mul(9, u), mul(2, a)), 1)), add(p, f))),
      add(sizeof(OpenKineticReverseStatus), finite));
  return {add(prefix, tape), reverse, add(add(prefix, tape), reverse), add(statuses, finite)};
}

/** Private reusable real kinetic trunk, not a public amplitude/derivative capability.
 * One immutable image, queue/BLAS and exclusive publication/execution lease must
 * outlive this owner. Only the numerical composer and its focused test adapter
 * may produce seeds; no mutable buffer or ticket escapes a public API.
 */
class PreparedDeviceKineticTrunk
{
public:
  /// Release both private numeric owners before caller execution resources expire.
  ~PreparedDeviceKineticTrunk() = default;

private:
  friend class PreparedDeviceComplexKinetic;
  friend struct KineticTrunkValidationAccess;
  friend struct ComplexKineticOwnerValidationAccess;

  /// Combined admission precedes preparation of either head-free numeric owner.
  static std::unique_ptr<PreparedDeviceKineticTrunk> prepare(std::shared_ptr<const PreparedDeviceModel> image,
      std::size_t targets, std::size_t budget, compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
  {
    if (!image)
      throw std::invalid_argument("Kinetic trunk requires an immutable image");
    image->validateCurrentDevice();
    const auto cost = estimateKineticTrunkCost(image->plan(), targets);
    if (cost.device_bytes > budget)
      throw std::length_error("Kinetic trunk exceeds its combined owner budget");
    return std::unique_ptr<PreparedDeviceKineticTrunk>(new PreparedDeviceKineticTrunk(image, targets, cost, queue, blas));
  }

  /// Reserve no legacy head, determinant/factors, normalized coefficients or full fallback.
  PreparedDeviceKineticTrunk(std::shared_ptr<const PreparedDeviceModel> image, std::size_t targets,
      KineticTrunkCost cost, compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
      : cost_(cost), forward_(image, 1, queue, blas, SpatialJetMode::FULL_VGL, false, true,
            configuredInferenceStatusPolicy(), {targets, cost.device_bytes, std::numeric_limits<std::size_t>::max()},
            false, true, true),
        reverse_(std::move(image), 1, queue, blas, targets, 0, true)
  {
    if (forward_.deviceWorkspaceBytes() != cost.forward_device_bytes ||
        reverse_.deviceWorkspaceBytes() != cost.reverse_device_bytes ||
        forward_.hostWorkspaceBytes() + reverse_.hostWorkspaceBytes() != cost.host_bytes)
      throw std::logic_error("Kinetic trunk exact preallocation descriptor mismatch");
  }

  /// Borrow each current K tape for immediate friend-only single or multiple objectives.
  template<class Consumer>
  void withTargets(const std::vector<double>& positions, Consumer&& consume)
  {
    reverse_.ready_ = false;
    forward_.model_->validateCurrentDevice();
    try
    {
      forward_.evaluateKineticTrunk(positions, std::forward<Consumer>(consume));
    }
    catch (...)
    {
      reverse_.ready_ = false;
      throw;
    }
  }

  /// Seed one real objective on the SAME immediate ticket; optionally clear its oneP.
  template<class Producer>
  void reverseSeeded(const PreparedDeviceInference::KineticTrunkTicket& ticket, Producer&& produce, bool clear_response)
  {
    reverse_.ready_ = false;
    forward_.validateKineticTrunk(ticket);
    if (ticket.image != reverse_.model_.get())
      throw std::invalid_argument("Kinetic trunk seeded pass has a different reverse image");
    reverse_.setTargetRange(ticket.features.target_electron_begin, ticket.features.laplacian_lanes);
    if (clear_response)
      reverse_.zero(reverse_.response_);
    reverse_.zero(reverse_.feature_);
    reverse_.zero(reverse_.root_adjoint_);
    // Only a friend producer owns these live destinations; no public mutable borrow.
    produce(ticket, reverse_.feature_.data(), reverse_.root_adjoint_.data(), reverse_.response_.data());
    reverse_.accumulateTrunkSeeds(ticket);
  }

  /// Admit all canonical elements before a friend may consume the completed real response.
  void finishResponse()
  {
    forward_.model_->validateCurrentDevice();
    reverse_.finishRequest();
  }

  /// Borrow completed oneP read-only until another seeded pass, request or rebind.
  const double* response() const
  {
    forward_.model_->validateCurrentDevice();
    if (!reverse_.ready_)
      throw std::logic_error("Kinetic trunk response unavailable");
    return reverse_.response_.data();
  }

  /// Existing one-objective unit wrapper sums all targets without replaying forward.
  template<class Producer>
  void evaluate(const std::vector<double>& positions, Producer&& produce)
  {
    withTargets(positions, [&](const auto& ticket) { reverseSeeded(ticket, produce, ticket.ordinal == 0); });
    finishResponse();
  }

  /// A same-family successor revokes completed response and all old tile tickets.
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> image)
  {
    reverse_.ready_ = false;
    forward_.rebindModel(image);
    reverse_.rebindModel(std::move(image));
  }

  KineticTrunkCost cost_;
  PreparedDeviceInference forward_;
  PreparedDeviceKineticReverse reverse_;
};
} // namespace qmcplusplus::psiformer::device
#endif
