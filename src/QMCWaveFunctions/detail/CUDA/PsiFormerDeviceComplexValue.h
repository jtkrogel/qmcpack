//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_VALUE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_VALUE_H
#include "PsiFormerDeviceInference.h"
#include "PsiFormerDeviceComplexValueKernels.h"
#include <array>

namespace qmcplusplus::psiformer::device
{
/// Explicit mixed-domain owner bytes; shared image/vendor storage is excluded.
struct ComplexValueCost
{
  std::size_t trunk_bytes, device_bytes, host_bytes, matrices;
  std::size_t pair_event_bytes = 0;
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

/// Capture the ordinary VALUE option once; retained/derivative roles never parse it.
inline bool configuredComplexValuePairCheck()
{
  static const bool enabled = [] {
    const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_VALUE_COOPERATIVE_PAIR_CHECK");
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("PSIFORMER_GPU_COMPLEX_VALUE_COOPERATIVE_PAIR_CHECK must be 0 or 1");
  }();
  return enabled;
}

/// Count an ordinary VALUE owner, including its admitted optional pair records.
inline ComplexValueCost estimateComplexValueCost(const PsiFormerExecutionPlan& plan, std::size_t batch,
                                               bool allow_pair_check = true)
{
  auto cost = complex_detail::baseCost(plan, batch, SpatialJetMode::VALUE_ONLY);
  const auto& e = plan.environment();
  const auto n = plan.modelShape().electrons();
  if (allow_pair_check && configuredComplexValuePairCheck() && n >= 64 && !e.periodic_profile &&
      e.geometry_feature_policy == GeometryFeaturePolicy::PERIODIC_TORUS_V1 &&
      std::all_of(e.periodic_axes.begin(), e.periodic_axes.end(), [](bool axis) { return axis; }))
  {
    const auto pairs = spatial_detail::checkedProduct(n, n, "Complex pair event extent overflow");
    if (pairs > std::numeric_limits<std::uint64_t>::max() / 2)
      throw std::overflow_error("Complex pair event encoding exceeds uint64 admission");
    cost.pair_event_bytes = checkedDeviceBytes<std::uint64_t>(batch);
    cost.device_bytes = spatial_detail::checkedAdd(cost.device_bytes, cost.pair_event_bytes,
                                                  "Complex pair event bytes overflow");
  }
  return cost;
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

  /// One absolute row replacement; every streamed point starts from the same base.
  struct Replacement
  {
    std::size_t electron;
    std::array<double, 3> position;
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
    if (cost.pair_event_bytes)
      std::clog << "  Complex VALUE pair-check preparation: selected=cooperative Ne="
                << image->plan().modelShape().electrons() << " capacity=" << capacity
                << " event_device_owner_bytes=" << cost.pair_event_bytes << "; before_allocation=yes\n";
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
    t.positions_.uploadOnStream(positions.data(), positions.size(), t.queue_.getNative());
    evaluateOwnedCoordinates();
  }

  /** Stream independent single-row replacements through one coordinate upload.
   * Accessors/consumers are immediate and caller-owned; no per-point tape or
   * output storage is retained here. The scoped invocation runs the unchanged
   * VALUE body, allowing the caller to preserve evaluation timer boundaries.
   */
  template<class ReplacementAt, class Consume, class ScopedEvaluation>
  void evaluateReplacementStream(const std::vector<double>& reference, std::size_t count,
      ReplacementAt&& replacement_at, Consume&& consume, ScopedEvaluation&& evaluate)
  {
    invalidate();
    auto& t = *trunk_;
    t.beginCoordinateEvaluation();
    if (!count || reference.size() != t.product(3, t.ne_) ||
        count - 1 > std::numeric_limits<std::size_t>::max() - generation_)
      throw std::invalid_argument("Complex VALUE replacement stream has invalid base/count");
    for (double coordinate : reference)
      if (!device_math::isFiniteBinary64(coordinate))
        throw std::invalid_argument("Complex VALUE replacement base is nonfinite");
    for (std::size_t point = 0; point < count; ++point)
      validateReplacement(replacement_at(point));
    t.setBatchCount(1);
    std::size_t previous = t.ne_;
    try
    {
      for (std::size_t point = 0; point < count; ++point)
      {
        if (point)
        {
          invalidate();
          t.beginCoordinateEvaluation();
        }
        const Replacement replacement = replacement_at(point);
        validateReplacement(replacement);
        evaluate([&] {
          if (!point)
            t.positions_.uploadOnStream(reference.data(), reference.size(), t.queue_.getNative());
          // Restore the preceding row before changing indices; replacements never accumulate.
          if (previous < t.ne_ && previous != replacement.electron)
            uploadReplacementRow(previous, reference.data() + 3 * previous);
          uploadReplacementRow(replacement.electron, replacement.position.data());
          evaluateOwnedCoordinates();
        });
        consume(point, resultView());
        previous = replacement.electron;
      }
    }
    catch (...)
    {
      valid_ = false;
      t.value_tape_valid_ = false;
      throw;
    }
  }

private:
  /// Prepare the common paired orbital matrix before any determinant overwrite.
  void evaluateHeadOwnedCoordinates(bool include_cusp)
  {
    auto& t = *trunk_;
    t.resetStatusAdmission();
    t.evaluatePreparedTrunk();
    if (include_cusp)
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
  }

  /// All coordinate loading paths share exactly the same trunk, head and determinant algebra.
  void evaluateOwnedCoordinates()
  {
    auto& t = *trunk_;
    evaluateHeadOwnedCoordinates(true);

    // The private head epoch ends here: matrix storage now becomes LU IN PLACE.
    checkRuntime(launchComplexValueFactor(t.queue_.getNative(), t.batch_ * t.shape_.determinants, t.ne_,
        matrix_.data(), channels_.data(), status_.data(), inverse_.data(), permutation_.data(), solve_.data()), "complex VALUE factorization");
    admitStatus();
    if (pair_events_.capacity())
      checkRuntime(launchComplexValuePairCheck(t.queue_.getNative(), t.batch_, t.ne_, t.shape_.spin_up_electrons,
          *t.model_->periodicCell(), t.positions_.data(), pair_events_.data()), "complex VALUE pair check");
    checkRuntime(launchComplexValueReduce(t.queue_.getNative(), t.batch_, t.shape_.determinants,
        t.ne_, t.shape_.spin_up_electrons, *t.model_->periodicCell(), t.positions_.data(),
        channels_.data(), t.cusp_jets_.data(), results_.data(), status_.data(), 1, pair_events_.data()), "complex VALUE mixture");
    admitStatus();
    t.value_tape_valid_ = !t.value_tape_.empty();
    valid_ = true;
  }

public:
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
    results_.downloadOnStream(destination.data(), view.count, trunk_->queue_.getNative());
  }

  /// Report checked standalone storage, excluding shared image/queue/vendor reserves.
  const ComplexValueCost& cost() const noexcept { return cost_; }

  /// A consumer failed after numerical publication; no borrowed result may survive.
  void discardResults() noexcept { valid_ = false; }

private:
  friend class PreparedDeviceComplexScore;

  /// Friend-only raw orbital forward: no cusp, physical phase, solve or determinant inverse.
  void evaluateOrbitalHead(const std::vector<double>& positions)
  {
    invalidate();
    auto& t = *trunk_;
    t.beginCoordinateEvaluation();
    const auto stride = t.product(3, t.ne_);
    if (positions.empty() || positions.size() % stride || positions.size() > t.positions_.capacity())
      throw std::invalid_argument("Complex orbital head packed physical input exceeds capacity");
    t.setBatchCount(positions.size() / stride);
    t.positions_.uploadOnStream(positions.data(), positions.size(), t.queue_.getNative());
    evaluateHeadOwnedCoordinates(false);
    t.value_tape_valid_ = !t.value_tape_.empty();
  }

  /// Row validation precedes every upload, including reentrant accessors.
  void validateReplacement(const Replacement& replacement) const
  {
    if (replacement.electron >= trunk_->ne_)
      throw std::invalid_argument("Complex VALUE replacement electron is out of range");
    for (double coordinate : replacement.position)
      if (!device_math::isFiniteBinary64(coordinate))
        throw std::invalid_argument("Complex VALUE replacement position is nonfinite");
  }

  /// Preserve the ordinary synchronous host-input lifetime and transfer accounting contract.
  void uploadReplacementRow(std::size_t electron, const double* position)
  {
    auto& t = *trunk_;
    constexpr auto bytes = 3 * sizeof(double);
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemcpyAsync(t.positions_.data() + 3 * electron, position, bytes,
        hipMemcpyHostToDevice, t.queue_.getNative()), "paired replacement row H2D");
#else
    checkRuntime(cudaMemcpyAsync(t.positions_.data() + 3 * electron, position, bytes,
        cudaMemcpyHostToDevice, t.queue_.getNative()), "paired replacement row H2D");
#endif
    t.queue_.sync();
    deviceMemoryMetrics(t.positions_.deviceOrdinal()).uploaded(bytes);
  }

  /// Prepare no real head/determinant/reverse buffers; assert the exact allocation inventory.
  PreparedDeviceComplexValue(std::shared_ptr<const PreparedDeviceModel> image, std::size_t capacity,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, ComplexValueCost cost,
      bool retain_score = false, bool retain_inverse = true)
      : cost_(cost), trunk_(new PreparedDeviceInference(std::move(image), capacity, queue, blas,
          SpatialJetMode::VALUE_ONLY, retain_score, false, configuredInferenceStatusPolicy(), {}, false, true)),
        envelope_(cost.matrices, "ComplexValue::envelope_", __FILE__, __LINE__),
        projections_(2 * cost.matrices, "ComplexValue::projections_", __FILE__, __LINE__),
        matrix_(cost.matrices, "ComplexValue::matrix_", __FILE__, __LINE__),
        channels_(capacity * trunk_->shape_.determinants, "ComplexValue::channels_", __FILE__, __LINE__),
        results_(capacity, "ComplexValue::results_", __FILE__, __LINE__),
        status_(1, "ComplexValue::status_", __FILE__, __LINE__),
        inverse_(retain_score && retain_inverse ? cost.matrices : 0, "ComplexValue::inverse_", __FILE__, __LINE__),
        solve_(retain_score && retain_inverse ? capacity * trunk_->shape_.determinants * trunk_->ne_ : 0, "ComplexValue::solve_", __FILE__, __LINE__),
        permutation_(solve_.capacity(), "ComplexValue::permutation_", __FILE__, __LINE__),
        pair_events_(cost.pair_event_bytes / sizeof(std::uint64_t), "ComplexValue::pair_events_", __FILE__, __LINE__)
  {
    const auto actual = trunk_->deviceWorkspaceBytes() + envelope_.capacity() * sizeof(double) +
        projections_.capacity() * sizeof(double) + matrix_.capacity() * sizeof(cuDoubleComplex) +
        channels_.capacity() * sizeof(ComplexValueChannel) + results_.capacity() * sizeof(ComplexValueResult) + 8 +
        (inverse_.capacity() + solve_.capacity()) * sizeof(cuDoubleComplex) + permutation_.capacity() * sizeof(std::size_t) +
        pair_events_.capacity() * sizeof(std::uint64_t);
    if ((cost.pair_event_bytes && (retain_score || pair_events_.capacity() != capacity)) ||
        pair_events_.capacity() * sizeof(std::uint64_t) != cost.pair_event_bytes)
      throw std::logic_error("Complex VALUE pair records differ from admitted ordinary owner");
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
    status_.downloadOnStream(&status, 1, trunk_->queue_.getNative());
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
  DeviceBuffer<std::uint64_t> pair_events_;
  std::size_t generation_ = 0;
  bool valid_ = false;
};
} // namespace qmcplusplus::psiformer::device
#endif
