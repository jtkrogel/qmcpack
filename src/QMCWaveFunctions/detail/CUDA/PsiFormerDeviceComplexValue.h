//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_VALUE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_VALUE_H
#include "PsiFormerDeviceInference.h"
#include "PsiFormerDeviceComplexValueKernels.h"

namespace qmcplusplus::psiformer::device
{
/// Explicit mixed-domain owner bytes; shared image/vendor storage is excluded.
struct ComplexValueCost
{
  std::size_t trunk_bytes, device_bytes, host_bytes, matrices;
};

namespace complex_detail
{
/// Count the shared head-free real trunk and mixed head for one admitted jet mode.
inline ComplexValueCost baseCost(const PsiFormerExecutionPlan& plan, std::size_t batch, SpatialJetMode mode)
{
  const auto& shape = plan.modelShape();
  const auto& env = plan.environment();
  if (!batch || (mode != SpatialJetMode::VALUE_ONLY && mode != SpatialJetMode::FULL_VGL) ||
      env.boundary != BoundaryCondition::PERIODIC || !env.fixed_nuclei ||
      env.head_representation != HeadRepresentation::PAIRED_REAL_COMPLEX ||
      env.parameter_scalar_domain != ScalarDomain::REAL || env.compute_scalar_domain != ScalarDomain::REAL ||
      env.amplitude_scalar_domain != ScalarDomain::COMPLEX)
    throw std::invalid_argument("Numerical paired jets require fixed-ion real-trunk periodic FULL/G-free metadata");
  const auto mul = [](auto a, auto b) { return spatial_detail::checkedProduct(a, b, "Complex VALUE extent overflow"); };
  const auto add = [](auto a, auto b) { return spatial_detail::checkedAdd(a, b, "Complex VALUE bytes overflow"); };
  const auto n = shape.electrons(), f = shape.feature_dimension, channels = mul(batch, shape.determinants);
  const auto m = mul(channels, mul(n, n)), rows = mul(batch, mul(n, f));
  if (batch > std::size_t(std::numeric_limits<int>::max()) || f > std::size_t(std::numeric_limits<int>::max()) ||
      mul(shape.determinants, n) > std::size_t(std::numeric_limits<int>::max()) ||
      m - 1 > (std::numeric_limits<std::uint64_t>::max() >> 8) ||
      (m - 1) / 128 >= std::size_t(std::numeric_limits<int>::max()))
    throw std::overflow_error("Complex VALUE index/BLAS dimensions exceed admission");
  const auto features = makePeriodicFeatureJetLayout(batch, n, shape.nuclei, shape.spin_up_electrons, mode);
  const auto attention = makeSpatialAttentionJetLayout(batch, n, shape.attention_heads,
      f / shape.attention_heads, n, mode);
  const auto planes = features.output.plane_count;
  const auto jets = mul(batch, mul(shape.attention_heads, n));
  const auto doubles = add(add(mul(3, mul(batch, n)), spatialJetSpanElements(features.output)),
      add(mul(8, mul(rows, planes)), add(spatialJetSpanElements(attention.attention), mul(batch, planes))));
  const auto host = add(checkedDeviceBytes<OpenSpatialStatus>(rows), checkedDeviceBytes<device_math::JetMathStatus>(jets));
  const auto trunk = add(add(checkedDeviceBytes<double>(doubles), checkedDeviceBytes<std::size_t>(batch)),
      add(host, sizeof(InferenceStatusWord)));
  return {trunk, add(add(add(trunk, mul(40, mul(m, planes))), checkedDeviceBytes<ComplexValueChannel>(channels)),
      add(checkedDeviceBytes<ComplexValueResult>(batch), sizeof(std::uint64_t))), host, m};
}
} // namespace complex_detail

/// Preserve the qualified VALUE estimate and allocation inventory exactly.
inline ComplexValueCost estimateComplexValueCost(const PsiFormerExecutionPlan& plan, std::size_t batch)
{
  return complex_detail::baseCost(plan, batch, SpatialJetMode::VALUE_ONLY);
}

/** Complete numerical-only complex VALUE over ONE real trunk and canonical image.
 * Caller queue/BLAS and its exclusive execution lease outlive this noncopyable
 * owner. No public GPU XML/score/VGL/inner-G/outer-twist capability is implied.
 */
class PreparedDeviceComplexValue
{
public:
  /// Borrow only published core amplitudes, not consumed/in-place factored heads.
  struct ResultView
  {
    const PreparedDeviceComplexValue* owner;
    const PreparedDeviceModel* image;
    std::size_t generation, count;
    const ComplexValueResult* data;
  };

  /// Exact combined preflight precedes constructor/trunk/device allocation.
  static std::unique_ptr<PreparedDeviceComplexValue> prepare(std::shared_ptr<const PreparedDeviceModel> image,
      std::size_t capacity, compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas,
      std::size_t budget)
  {
    if (!image)
      throw std::invalid_argument("Complex VALUE requires a named paired image");
    image->validateCurrentDevice();
    const auto cost = estimateComplexValueCost(image->plan(), capacity);
    if (cost.device_bytes > budget)
      throw std::length_error("Numerical complex VALUE exceeds its standalone owner budget");
    return std::unique_ptr<PreparedDeviceComplexValue>(new PreparedDeviceComplexValue(std::move(image), capacity, queue, blas, cost));
  }

  /// Rebind only the existing immutable family, revoking all published epochs.
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> successor)
  {
    invalidate();
    trunk_->rebindModel(std::move(successor));
  }

  /// Check a publication successor without invalidating the accepted result epoch.
  void validateModelRebind(const std::shared_ptr<const PreparedDeviceModel>& successor) const
  {
    if (generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Complex VALUE generation exhausted");
    trunk_->validateModelRebind(successor);
  }

  /// Caller retains exclusive authority and an old-image owner after validated preparation.
  void adoptPreparedModel(const std::shared_ptr<const PreparedDeviceModel>& successor) noexcept
  {
    valid_ = false;
    ++generation_;
    trunk_->adoptPreparedModel(successor);
  }

  /// Complete every live configuration before publishing the sole result prefix.
  void evaluate(const std::vector<double>& positions)
  {
    invalidate();
    auto& t = *trunk_;
    t.beginCoordinateEvaluation();
    const auto stride = t.product(3, t.ne_);
    if (positions.empty() || positions.size() % stride || positions.size() > t.positions_.capacity())
      throw std::invalid_argument("Complex VALUE packed physical input exceeds capacity");
    t.setBatchCount(positions.size() / stride);
    t.positions_.upload(positions.data(), positions.size());
    t.resetStatusAdmission();
    t.evaluatePreparedTrunk();
    t.evaluateCusp();
    t.checkOpen(t.batch_);

    // Project both head components from this SAME admitted real trunk generation.
    const auto m = t.product(t.batch_, t.matrices_per_configuration_);
    projectOpenOrbitalJets(t.blas_, t.orbitals_, t.a_.data(), parameter(ParameterRole::BACKFLOW_UP_WEIGHT),
        parameter(ParameterRole::BACKFLOW_DOWN_WEIGHT), projections_.data());
    projectOpenOrbitalJets(t.blas_, t.orbitals_, t.a_.data(), parameter(ParameterRole::BACKFLOW_UP_WEIGHT, HeadTensorComponent::IMAG),
        parameter(ParameterRole::BACKFLOW_DOWN_WEIGHT, HeadTensorComponent::IMAG), projections_.data() + cost_.matrices);
    resetStatus();
    checkRuntime(launchComplexValueEnvelopes(t.queue_.getNative(), t.orbitals_, *t.model_->periodicCell(),
        t.positions_.data(), t.model_->nuclei(), parameter(ParameterRole::ENVELOPE_PI_UP), parameter(ParameterRole::ENVELOPE_PI_DOWN),
        parameter(ParameterRole::ENVELOPE_ZETA_UP), parameter(ParameterRole::ENVELOPE_ZETA_DOWN), envelope_.data(), status_.data()),
        "paired VALUE envelopes");
    checkRuntime(launchComplexValuePack(t.queue_.getNative(), m, envelope_.data(), projections_.data(),
        projections_.data() + cost_.matrices, matrix_.data(), status_.data()), "paired VALUE head");
    admitStatus();

    // The private head epoch ends here: matrix storage now becomes LU IN PLACE.
    checkRuntime(launchComplexValueFactor(t.queue_.getNative(), t.batch_ * t.shape_.determinants, t.ne_,
        matrix_.data(), channels_.data(), status_.data(), inverse_.data(), permutation_.data(), solve_.data()), "complex VALUE factorization");
    admitStatus();
    checkRuntime(launchComplexValueReduce(t.queue_.getNative(), t.batch_, t.shape_.determinants,
        t.ne_, t.shape_.spin_up_electrons, *t.model_->periodicCell(), t.positions_.data(),
        channels_.data(), t.cusp_jets_.data(), results_.data(), status_.data()), "complex VALUE mixture");
    admitStatus();
    t.value_tape_valid_ = !t.value_tape_.empty();
    valid_ = true;
  }

  /// Expose a generation-stamped complete amplitude; never expose the overwritten head.
  ResultView resultView() const
  {
    if (!valid_)
      throw std::logic_error("Complex VALUE result is unavailable");
    return {this, trunk_->model_.get(), generation_, trunk_->batch_, results_.data()};
  }

  /// Validate owner/image/generation/count before any device result is consumed.
  void download(const ResultView& view, std::vector<ComplexValueResult>& destination) const
  {
    trunk_->model_->validateCurrentDevice();
    if (!valid_ || view.owner != this || view.image != trunk_->model_.get() || view.generation != generation_ ||
        view.count != trunk_->batch_ || view.data != results_.data() || destination.size() != view.count)
      throw std::invalid_argument("Complex VALUE result view is stale, foreign or wrong extent");
    results_.download(destination.data(), view.count);
  }

  /// Report checked standalone storage, excluding shared image/queue/vendor reserves.
  const ComplexValueCost& cost() const noexcept { return cost_; }

private:
  friend class PreparedDeviceComplexScore;

  /// Prepare no real head/determinant/reverse buffers; assert the exact allocation inventory.
  PreparedDeviceComplexValue(std::shared_ptr<const PreparedDeviceModel> image, std::size_t capacity,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, ComplexValueCost cost, bool retain_score = false)
      : cost_(cost), trunk_(new PreparedDeviceInference(std::move(image), capacity, queue, blas,
          SpatialJetMode::VALUE_ONLY, retain_score, false, configuredInferenceStatusPolicy(), {}, false, true)),
        envelope_(cost.matrices), projections_(2 * cost.matrices), matrix_(cost.matrices),
        channels_(capacity * trunk_->shape_.determinants), results_(capacity), status_(1),
        inverse_(retain_score ? cost.matrices : 0), solve_(retain_score ? capacity * trunk_->shape_.determinants * trunk_->ne_ : 0),
        permutation_(solve_.capacity())
  {
    const auto actual = trunk_->deviceWorkspaceBytes() + envelope_.capacity() * sizeof(double) +
        projections_.capacity() * sizeof(double) + matrix_.capacity() * sizeof(cuDoubleComplex) +
        channels_.capacity() * sizeof(ComplexValueChannel) + results_.capacity() * sizeof(ComplexValueResult) + 8 +
        (inverse_.capacity() + solve_.capacity()) * sizeof(cuDoubleComplex) + permutation_.capacity() * sizeof(std::size_t);
    if (actual != cost.device_bytes || trunk_->deviceWorkspaceBytes() != cost.trunk_bytes ||
        trunk_->hostWorkspaceBytes() != cost.host_bytes)
      throw std::logic_error("Complex VALUE owned storage differs from exact preflight");
  }

  /// Resolve canonical real/imag intervals from the current image, never a saved P pointer.
  const double* parameter(ParameterRole role, HeadTensorComponent component = HeadTensorComponent::REAL) const
  { return trunk_->model_->parameters() + trunk_->plan_.parameter(role, NO_ATTENTION_BLOCK, component).begin; }

  /// Revoke old products even when replacement input or successor admission fails.
  void invalidate()
  {
    valid_ = false;
    if (generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Complex VALUE generation exhausted");
    ++generation_;
  }

  /// Reset only the owner scalar diagnostic, with no host-to-device status copy.
  void resetStatus()
  {
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemsetAsync(status_.data(), 0xff, 8, trunk_->queue_.getNative()), "complex VALUE status reset");
#else
    checkRuntime(cudaMemsetAsync(status_.data(), 0xff, 8, trunk_->queue_.getNative()), "complex VALUE status reset");
#endif
  }

  /// Synchronize/status-admit before transitioning from head to LU or final publication.
  void admitStatus()
  {
    trunk_->queue_.sync();
    std::uint64_t status;
    status_.download(&status, 1);
    if (status != std::numeric_limits<std::uint64_t>::max())
      throw std::runtime_error("Complex VALUE numerical failure, status=" + std::to_string(status));
  }

  const ComplexValueCost cost_;
  std::unique_ptr<PreparedDeviceInference> trunk_;
  DeviceBuffer<double> envelope_, projections_;
  DeviceBuffer<cuDoubleComplex> matrix_;
  DeviceBuffer<ComplexValueChannel> channels_;
  DeviceBuffer<ComplexValueResult> results_;
  DeviceBuffer<std::uint64_t> status_;
  DeviceBuffer<cuDoubleComplex> inverse_, solve_;
  DeviceBuffer<std::size_t> permutation_;
  std::size_t generation_ = 0;
  bool valid_ = false;
};
} // namespace qmcplusplus::psiformer::device
#endif
