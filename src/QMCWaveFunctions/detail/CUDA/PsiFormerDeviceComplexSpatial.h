//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_H
#include "PsiFormerDeviceComplexValue.h"
#include "PsiFormerDeviceComplexSpatialKernels.h"

namespace qmcplusplus::psiformer::device
{
/// Explicit FULL mixed-domain buffers, excluding shared image/vendor storage.
struct ComplexSpatialCost
{
  std::size_t trunk_bytes, device_bytes, host_bytes, matrices, head_elements;
};

/// Admit the complete real trunk/head/inverse/traces inventory before allocation.
inline ComplexSpatialCost estimateComplexSpatialCost(const PsiFormerExecutionPlan& plan, std::size_t batch)
{
  const auto base = complex_detail::baseCost(plan, batch, SpatialJetMode::FULL_VGL);
  const auto mul = [](auto a, auto b) { return spatial_detail::checkedProduct(a, b, "Complex FULL extent overflow"); };
  const auto add = [](auto a, auto b) { return spatial_detail::checkedAdd(a, b, "Complex FULL bytes overflow"); };
  const auto n = plan.modelShape().electrons(), bn = mul(batch, n), vectors = mul(bn, plan.modelShape().determinants);
  const auto planes = add(1, mul(4, n)), head = mul(planes, base.matrices);
  if (head - 1 > (std::numeric_limits<std::uint64_t>::max() >> 8) ||
      (head - 1) / 128 >= std::size_t(std::numeric_limits<int>::max()))
    throw std::overflow_error("Complex FULL head index exceeds launch/status admission");
  const auto extra = add(add(mul(16, base.matrices), mul(16 + sizeof(std::size_t), vectors)),
      add(mul(16, mul(bn, n)), mul(80, bn)));
  return {base.trunk_bytes, add(base.device_bytes, extra), base.host_bytes, base.matrices, head};
}

/// Count the forward-only B1 paired target profile, never a kinetic/reverse owner.
inline ComplexSpatialCost estimateComplexSpatialTargetCost(const PsiFormerExecutionPlan& plan, std::size_t targets)
{
  const auto base = complex_detail::baseCost(plan, 1, SpatialJetMode::VALUE_ONLY);
  const auto prefix = estimateHeadFreeTargetPrefixCost(plan, targets);
  const auto mul = [](auto a, auto b) { return spatial_detail::checkedProduct(a, b, "Complex target extent overflow"); };
  const auto add = [](auto a, auto b) { return spatial_detail::checkedAdd(a, b, "Complex target bytes overflow"); };
  const auto& s = plan.modelShape();
  const auto n = s.electrons(), h = mul(add(1, mul(4, targets)), base.matrices);
  if (h - 1 > (std::numeric_limits<std::uint64_t>::max() >> 8) ||
      (h - 1) / 128 >= std::size_t(std::numeric_limits<int>::max()))
    throw std::overflow_error("Complex target head exceeds launch/status admission");
  const auto bytes = add(prefix.device_workspace_bytes, add(mul(40, h), add(mul(16, base.matrices),
      add(mul(16 + sizeof(std::size_t), mul(s.determinants, n)), add(mul(16, mul(n, n)),
          add(mul(80, n), add(mul(sizeof(ComplexValueChannel), s.determinants), sizeof(ComplexValueResult) + 8)))))));
  return {prefix.device_workspace_bytes, bytes, add(prefix.host_workspace_bytes, sizeof(ComplexValueResult)), base.matrices, h};
}

/** Numerical CORE FULL_VGL over one real trunk, paired head and canonical image.
 * Caller queue/BLAS/exclusive execution lease outlive the owner. No public GPU
 * XML/score/optimizer/inner-G/outer-twist/performance capability is implied.
 */
class PreparedDeviceComplexSpatial
{
public:
  /// Borrow complete amplitudes and physical complex log G/L and Laplacian ratios.
  struct ResultView
  {
    const PreparedDeviceComplexSpatial* owner;
    const PreparedDeviceModel* image;
    std::size_t generation, count;
    const ComplexValueResult* amplitude;
    const cuDoubleComplex *gradient, *lap_log, *lap_ratio;
  };

  /// Reject malformed profiles/extents and byte limits BEFORE preparing numeric buffers.
  static std::unique_ptr<PreparedDeviceComplexSpatial> prepare(std::shared_ptr<const PreparedDeviceModel> image,
      std::size_t capacity, compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, std::size_t budget)
  {
    if (!image)
      throw std::invalid_argument("Complex FULL requires a paired numerical image");
    image->validateCurrentDevice();
    const auto cost = estimateComplexSpatialCost(image->plan(), capacity);
    if (cost.device_bytes > budget)
      throw std::length_error("Complex FULL exceeds its standalone owner workspace budget");
    return std::unique_ptr<PreparedDeviceComplexSpatial>(new PreparedDeviceComplexSpatial(std::move(image), capacity, queue, blas, cost));
  }

  /// Explicit B1 target admission precedes every prefix/head/output allocation.
  static std::unique_ptr<PreparedDeviceComplexSpatial> prepare(std::shared_ptr<const PreparedDeviceModel> image,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, TargetVglPreparation preparation)
  {
    if (!image || !preparation.max_targets || !preparation.device_workspace_budget_bytes || !preparation.scalar_work_budget)
      throw std::invalid_argument("Complex target VGL requires a paired image, positive K and explicit caps");
    image->validateCurrentDevice();
    const auto cost = estimateComplexSpatialTargetCost(image->plan(), preparation.max_targets);
    const auto prefix = estimateHeadFreeTargetPrefixCost(image->plan(), preparation.max_targets);
    if (cost.device_bytes > preparation.device_workspace_budget_bytes || prefix.charged_scalar_work_units > preparation.scalar_work_budget)
      throw std::length_error("Complex target VGL exceeds its per-owner byte/work caps");
    return std::unique_ptr<PreparedDeviceComplexSpatial>(new PreparedDeviceComplexSpatial(std::move(image), 1, queue, blas, cost, preparation));
  }

  /// Revoke all products before a same-family successor can replace the current image.
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> image)
  {
    invalidate();
    trunk_->rebindModel(std::move(image));
  }

  /// Evaluate every live configuration before publishing any complex spatial output.
  void evaluate(const std::vector<double>& positions)
  {
    invalidate();
    if (canonical_)
    {
      evaluateTargets(positions);
      return;
    }
    auto& t = *trunk_;
    t.beginCoordinateEvaluation();
    if (positions.empty() || positions.size() % (3 * t.ne_) || positions.size() > t.positions_.capacity())
      throw std::invalid_argument("Complex FULL packed coordinates exceed capacity");
    t.setBatchCount(positions.size() / (3 * t.ne_));
    t.positions_.upload(positions.data(), positions.size());
    t.resetStatusAdmission();
    t.evaluatePreparedTrunk();
    t.evaluateCusp();
    t.checkOpen(t.batch_);

    finishHead();
    checkRuntime(launchComplexSpatialTraces(t.queue_.getNative(), t.batch_, t.shape_.determinants, t.ne_, channels_.data(),
        amplitude_.data(), inverse_.data(), head_.data(), t.cusp_jets_.data(), product_.data(), outputs_.data(),
        outputs_.data() + 3 * capacity_electrons_, outputs_.data() + 4 * capacity_electrons_, status_.data()), "complex FULL traces");
    admitStatus();
    valid_ = true;
  }

  /// Expose only the complete current physical-core result, never head/LU pointers.
  ResultView resultView() const
  {
    if (!valid_)
      throw std::logic_error("Complex FULL result unavailable");
    return {this, trunk_->model_.get(), generation_, trunk_->batch_, amplitude_.data(), outputs_.data(),
        outputs_.data() + 3 * capacity_electrons_, outputs_.data() + 4 * capacity_electrons_};
  }

  /// Reject stale/foreign/tail-mismatched views before downloading their live spans.
  void download(const ResultView& view, std::vector<ComplexValueResult>& amplitude,
      std::vector<cuDoubleComplex>& gradient, std::vector<cuDoubleComplex>& lap_log, std::vector<cuDoubleComplex>& lap_ratio) const
  {
    trunk_->model_->validateCurrentDevice();
    const auto n = trunk_->ne_, count = trunk_->batch_;
    if (!valid_ || view.owner != this || view.image != trunk_->model_.get() || view.generation != generation_ ||
        view.count != count || view.amplitude != amplitude_.data() || view.gradient != outputs_.data() ||
        view.lap_log != outputs_.data() + 3 * capacity_electrons_ || view.lap_ratio != outputs_.data() + 4 * capacity_electrons_ ||
        amplitude.size() != count || gradient.size() != 3 * count * n || lap_log.size() != count * n || lap_ratio.size() != count * n)
      throw std::invalid_argument("Complex FULL result is stale, foreign or wrong extent");
    amplitude_.download(amplitude.data(), count);
    outputs_.download(gradient.data(), gradient.size());
    outputs_.download(lap_log.data(), lap_log.size(), 3 * capacity_electrons_);
    outputs_.download(lap_ratio.data(), lap_ratio.size(), 4 * capacity_electrons_);
  }

  /// Count standalone owned storage, excluding shared model/vendor reserves.
  const ComplexSpatialCost& cost() const noexcept { return cost_; }

private:
  friend struct ComplexSpatialTargetValidationAccess;

  /// One mapped prefix/cusp plus paired head; no reverse ticket or retained tape.
  void prepareTarget(std::size_t begin, std::size_t count)
  {
    auto& t = *trunk_;
    t.setTargetRange(begin, count);
    t.resetStatusAdmission();
    t.evaluatePreparedTrunk();
    t.evaluateCusp();
    t.checkOpen(1);
    finishHead();
  }

  /// Write live K lanes directly into final physical output ranges, without scatter.
  void finishTarget(std::size_t begin)
  {
    auto& t = *trunk_;
    checkRuntime(launchComplexSpatialTraces(t.queue_.getNative(), t.orbitals_, channels_.data(), amplitude_.data(),
        inverse_.data(), head_.data(), t.cusp_jets_.data(), product_.data(), outputs_.data() + 3 * begin,
        outputs_.data() + 3 * t.ne_ + begin, outputs_.data() + 4 * t.ne_ + begin, status_.data()), "complex target traces");
    admitStatus();
  }

  /// Upload once; only complete consistent targets can publish the full physical root.
  void evaluateTargets(const std::vector<double>& positions)
  {
    auto& t = *trunk_;
    t.beginCoordinateEvaluation();
    if (positions.size() != 3 * t.ne_)
      throw std::invalid_argument("Complex target VGL requires one complete physical configuration");
    t.positions_.upload(positions.data(), positions.size());
    for (std::size_t begin = 0; begin < t.ne_;)
    {
      const auto count = std::min(t.target_preparation_.max_targets, t.ne_ - begin);
      prepareTarget(begin, count);
      ComplexValueResult current;
      amplitude_.download(&current, 1);
      if (!begin)
        *canonical_ = current;
      else if (std::abs(current.log_abs - canonical_->log_abs) > 2e-11 ||
          std::hypot(cuCreal(current.phase) - cuCreal(canonical_->phase), cuCimag(current.phase) - cuCimag(canonical_->phase)) > 2e-11)
        throw std::runtime_error("Complex target VGL base changed within one request");
      finishTarget(begin);
      begin += count;
    }
    amplitude_.upload(canonical_.get(), 1);
    t.queue_.sync();
    valid_ = true;
  }

  /// Consume the shared admitted prefix in the original paired-head/factor order.
  void finishHead()
  {
    auto& t = *trunk_;
    // Two projections consume the SAME real FULL trunk; E supplies only radial jets.
    projectOpenOrbitalJets(t.blas_, t.orbitals_, t.a_.data(), parameter(ParameterRole::BACKFLOW_UP_WEIGHT),
        parameter(ParameterRole::BACKFLOW_DOWN_WEIGHT), projections_.data());
    projectOpenOrbitalJets(t.blas_, t.orbitals_, t.a_.data(), parameter(ParameterRole::BACKFLOW_UP_WEIGHT, HeadTensorComponent::IMAG),
        parameter(ParameterRole::BACKFLOW_DOWN_WEIGHT, HeadTensorComponent::IMAG), projections_.data() + cost_.head_elements);
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemsetAsync(status_.data(), 0xff, 8, t.queue_.getNative()), "complex FULL status reset");
#else
    checkRuntime(cudaMemsetAsync(status_.data(), 0xff, 8, t.queue_.getNative()), "complex FULL status reset");
#endif
    checkRuntime(launchComplexValueEnvelopes(t.queue_.getNative(), t.orbitals_, *t.model_->periodicCell(),
        t.positions_.data(), t.model_->nuclei(), parameter(ParameterRole::ENVELOPE_PI_UP), parameter(ParameterRole::ENVELOPE_PI_DOWN),
        parameter(ParameterRole::ENVELOPE_ZETA_UP), parameter(ParameterRole::ENVELOPE_ZETA_DOWN), envelope_.data(), status_.data()),
        "complex FULL envelopes");
    checkRuntime(launchComplexSpatialHead(t.queue_.getNative(), t.orbitals_, envelope_.data(), projections_.data(),
        projections_.data() + cost_.head_elements, head_.data(), status_.data()), "complex FULL product-rule head");
    admitStatus();

    // Only the contiguous VALUE plane becomes LU; all derivative planes survive.
    checkRuntime(launchComplexValueFactor(t.queue_.getNative(), t.batch_ * t.shape_.determinants, t.ne_, head_.data(),
        channels_.data(), status_.data(), inverse_.data(), permutation_.data(), solve_.data()), "complex FULL inverse");
    admitStatus();
    checkRuntime(launchComplexValueReduce(t.queue_.getNative(), t.batch_, t.shape_.determinants, t.ne_,
        t.shape_.spin_up_electrons, *t.model_->periodicCell(), t.positions_.data(), channels_.data(), t.cusp_jets_.data(),
        amplitude_.data(), status_.data(), t.cusp_.output.configuration_stride), "complex FULL amplitude");
    admitStatus();
  }

  /// Allocate only the checked FULL trunk/head/inverse/traces inventory.
  PreparedDeviceComplexSpatial(std::shared_ptr<const PreparedDeviceModel> image, std::size_t capacity,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, ComplexSpatialCost cost,
      TargetVglPreparation preparation = {})
      : cost_(cost), trunk_(new PreparedDeviceInference(std::move(image), capacity, queue, blas,
          SpatialJetMode::FULL_VGL, false, false, configuredInferenceStatusPolicy(), preparation, false, true)),
        capacity_electrons_(capacity * trunk_->ne_), envelope_(cost.head_elements), projections_(2 * cost.head_elements),
        head_(cost.head_elements), inverse_(cost.matrices), solve_(capacity_electrons_ * trunk_->shape_.determinants),
        product_(capacity_electrons_ * trunk_->ne_), outputs_(5 * capacity_electrons_), permutation_(solve_.capacity()),
        channels_(capacity * trunk_->shape_.determinants), amplitude_(capacity), status_(1),
        canonical_(preparation.max_targets ? new ComplexValueResult : nullptr)
  {
    auto actual = trunk_->deviceWorkspaceBytes() + (envelope_.capacity() + projections_.capacity()) * sizeof(double);
    for (const auto* buffer : {&head_, &inverse_, &solve_, &product_, &outputs_})
      actual += buffer->capacity() * sizeof(cuDoubleComplex);
    actual += permutation_.capacity() * sizeof(std::size_t) + channels_.capacity() * sizeof(ComplexValueChannel) +
        amplitude_.capacity() * sizeof(ComplexValueResult) + 8;
    if (actual != cost.device_bytes || trunk_->deviceWorkspaceBytes() != cost.trunk_bytes ||
        trunk_->hostWorkspaceBytes() + (canonical_ ? sizeof(ComplexValueResult) : 0) != cost.host_bytes)
      throw std::logic_error("Complex FULL owned bytes differ from exact preflight");
  }

  /// Resolve current canonical intervals without retaining old parameter pointers.
  const double* parameter(ParameterRole role, HeadTensorComponent component = HeadTensorComponent::REAL) const
  { return trunk_->model_->parameters() + trunk_->plan_.parameter(role, NO_ATTENTION_BLOCK, component).begin; }

  /// Revoke previous products even if input or successor admission fails next.
  void invalidate()
  {
    valid_ = false;
    if (generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Complex FULL generation exhausted");
    ++generation_;
  }

  /// Admit every numeric phase before the sole final publication flag changes.
  void admitStatus()
  {
    trunk_->queue_.sync();
    std::uint64_t status;
    status_.download(&status, 1);
    if (status != std::numeric_limits<std::uint64_t>::max())
      throw std::runtime_error("Complex FULL node/inverse/nonfinite admission failed, status=" + std::to_string(status));
  }

  ComplexSpatialCost cost_;
  std::unique_ptr<PreparedDeviceInference> trunk_;
  std::size_t capacity_electrons_;
  DeviceBuffer<double> envelope_, projections_;
  DeviceBuffer<cuDoubleComplex> head_, inverse_, solve_, product_, outputs_;
  DeviceBuffer<std::size_t> permutation_;
  DeviceBuffer<ComplexValueChannel> channels_;
  DeviceBuffer<ComplexValueResult> amplitude_;
  DeviceBuffer<std::uint64_t> status_;
  std::unique_ptr<ComplexValueResult> canonical_;
  std::size_t generation_ = 0;
  bool valid_ = false;
};
} // namespace qmcplusplus::psiformer::device
#endif
