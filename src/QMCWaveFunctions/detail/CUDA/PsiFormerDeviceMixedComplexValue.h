//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_COMPLEX_VALUE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_COMPLEX_VALUE_H
#include "PsiFormerDeviceComplexValue.h"

namespace qmcplusplus::psiformer::device
{
/// Complete VALUE-only mixed trunk with a FP64 paired head and whole-request FP64 replay.
/// This numerical owner grants no XML, spatial derivative, score or training capability.
class PreparedDeviceMixedComplexValue
{
public:
  struct ResultView
  {
    const PreparedDeviceMixedComplexValue* owner;
    const PreparedDeviceModel* image;
    std::size_t generation, count;
    const ComplexValueResult* data;
  };

  /// Count both complete graphs and all private casts/pivots before allocating any of them.
  static std::size_t estimateDeviceBytes(const PsiFormerExecutionPlan& plan, std::size_t capacity)
  {
    if (!supportsMixedPairedTrunkEnvironment(plan.environment()))
      throw std::invalid_argument("Mixed paired VALUE requires a fixed legacy periodic real trunk");
    const auto mul = [](auto a, auto b) {
      return spatial_detail::checkedProduct(a, b, "Mixed paired VALUE extent overflow");
    };
    const auto add = [](auto a, auto b) {
      return spatial_detail::checkedAdd(a, b, "Mixed paired VALUE owner bytes overflow");
    };
    const auto& shape = plan.modelShape();
    const auto rows = mul(capacity, shape.electrons());
    const auto width = shape.feature_dimension;
    const auto input = plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0];
    auto bytes = mul(2, estimateComplexValueCost(plan, capacity, true, false).device_bytes);
    bytes = add(bytes, checkedDeviceBytes<float>(plan.parameterCount()));
    bytes = add(bytes, checkedDeviceBytes<float>(mul(rows, add(std::max(input, width), width))));
    bytes = add(bytes, checkedDeviceBytes<double>(mul(capacity, shape.determinants)));
    bytes = add(bytes, sizeof(PsiFormerDeviceNumericalDiagnostics));
    // Conversion runs after both FP64 graphs but before casts/pivots exist. Its
    // temporary diagnostics must also fit this same admitted numeric peak.
    const auto cold = add(add(mul(2, estimateComplexValueCost(plan, capacity, true, false).device_bytes),
                              checkedDeviceBytes<float>(plan.parameterCount())),
                          sizeof(PsiFormerParameterConversionDiagnostics));
    if (cold > bytes)
      throw std::logic_error("Mixed paired final inventory does not dominate cold conversion storage");
    return bytes;
  }

  /// A composing owner charges its one shared immutable FP32 cache separately.
  static std::size_t estimatePrivateDeviceBytes(const PsiFormerExecutionPlan& plan, std::size_t capacity)
  {
    return estimateDeviceBytes(plan, capacity) -
        checkedDeviceBytes<float>(plan.parameterCount());
  }

  /// Bind every private resource to the same canonical image, queue and exclusive execution lease.
  PreparedDeviceMixedComplexValue(std::shared_ptr<const PreparedDeviceModel> model, std::size_t capacity,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, std::size_t budget,
      bool allow_fp64_retry = true,
      std::shared_ptr<const PreparedDeviceMixedDenseParameters> shared_parameters = {})
      : model_(std::move(model)), queue_(queue), owns_parameters_(!shared_parameters), allow_retry_(allow_fp64_retry)
  {
    if (!model_)
      throw std::invalid_argument("Mixed paired VALUE requires a canonical model");
    validateSharedParameters(model_, shared_parameters);
    const auto bytes = owns_parameters_
        ? estimateDeviceBytes(model_->plan(), capacity)
        : estimatePrivateDeviceBytes(model_->plan(), capacity);
    if (bytes > budget)
      throw std::length_error("Mixed paired VALUE exceeds its combined owner budget");
    model_->validateCurrentDevice();
    reference_ = PreparedDeviceComplexValue::prepare(model_, capacity, queue, blas, budget, false);
    approximate_ = PreparedDeviceComplexValue::prepare(model_, capacity, queue, blas, budget, false);
    parameters_ = std::move(shared_parameters);
    if (owns_parameters_)
      parameters_ = std::make_shared<PreparedDeviceMixedDenseParameters>(model_, queue_,
          MixedTrunkHeadPolicy::PAIRED_COMPLEX_FP64_HEAD, allow_retry_);
    if (!allow_retry_ && !parameters_->usable())
      throw MixedPrecisionRetryRequired("Shared paired affine cache requires FP64 replay");
    adapter_ = std::make_shared<PreparedDeviceMixedValueDense>(parameters_,
        spatial_detail::checkedProduct(capacity, model_->plan().modelShape().electrons(), "Mixed paired rows overflow"),
        queue, blas);
    pivots_.reserveNamed(spatial_detail::checkedProduct(capacity, model_->plan().modelShape().determinants,
        "Mixed paired pivot extent overflow"), "MixedComplexValue::minimum_scaled_pivots", __FILE__, __LINE__);
    approximate_->trunk_->setMixedDenseForPairedTrunk(adapter_);
    if (deviceWorkspaceBytes() != bytes)
      throw std::logic_error("Mixed paired VALUE owner accounting disagrees with prepared storage");
  }

  /// Retire every borrowed asynchronous pivot/cast view before any private allocation is freed.
  ~PreparedDeviceMixedComplexValue()
  {
    try { queue_.sync(); } catch (...) {}
  }

  /// Revoke old authority before attempting either complete graph; never splice partial fields.
  void evaluate(const std::vector<double>& positions, bool strict_mixed = false)
  {
    advanceGeneration();
    selected_ = nullptr;
    retry_reason_.clear();
    used_retry_ = false;
    reference_->discardResults();
    approximate_->discardResults();
    try
    {
      approximate_->evaluateCoordinates(positions, pivots_.data());
      selected_ = approximate_.get();
      return;
    }
    catch (const MixedPrecisionRetryRequired& error)
    {
      if (!allow_retry_ || strict_mixed) throw;
      retry_reason_ = error.what();
    }
    catch (const MixedComplexValueFailure& error)
    {
      if (!allow_retry_ || strict_mixed || error.stage() == ComplexValueFailureStage::PHYSICAL_ENVELOPE) throw;
      retry_reason_ = error.what();
    }
    approximate_->discardResults();
    used_retry_ = true;
    reference_->evaluate(positions);
    selected_ = reference_.get();
  }

  /// Select the prepared FP64 primal for an entire rejected native epoch, without mixed work.
  void evaluateFp64(const std::vector<double>& positions)
  {
    if (!allow_retry_)
      throw std::logic_error("Explicit paired FP64 replay requires the admitted retry policy");
    advanceGeneration();
    selected_ = nullptr;
    used_retry_ = true;
    retry_reason_ = "native_epoch";
    reference_->discardResults();
    approximate_->discardResults();
    reference_->evaluate(positions);
    selected_ = reference_.get();
  }

  /// Expose wrapper-owned authority, not an interchangeable private-owner ticket.
  ResultView resultView() const
  {
    if (!selected_)
      throw std::logic_error("Mixed paired VALUE has no admitted result");
    const auto view = selected_->resultView();
    return {this, model_.get(), generation_, view.count, view.data};
  }

  void download(const ResultView& view, std::vector<ComplexValueResult>& destination) const
  {
    const auto current = resultView();
    if (view.owner != this || view.image != current.image || view.generation != current.generation ||
        view.count != current.count || view.data != current.data)
      throw std::invalid_argument("Mixed paired VALUE result is stale or foreign");
    selected_->download(selected_->resultView(), destination);
  }

  /// Prepare conversion evidence before publishing either successor; failed preparation keeps old results valid.
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> successor, std::size_t pending_budget,
      std::shared_ptr<const PreparedDeviceMixedDenseParameters> shared_successor = {})
  {
    model_->validateSuccessor(successor);
    if (successorWorkspaceBytes() > pending_budget)
      throw std::length_error("Mixed paired successor cache exceeds pending owner budget");
    if (owns_parameters_ == bool(shared_successor))
      throw std::invalid_argument("Mixed paired cache ownership mode must remain fixed across rebind");
    std::shared_ptr<const PreparedDeviceMixedDenseParameters> prepared = std::move(shared_successor);
    if (owns_parameters_)
      prepared = std::make_shared<PreparedDeviceMixedDenseParameters>(successor, queue_,
          MixedTrunkHeadPolicy::PAIRED_COMPLEX_FP64_HEAD, allow_retry_);
    validatePreparedRebind(successor, prepared);
    adoptPreparedRebind(successor, prepared);
  }

  /// Validate all failure points without revoking this leaf's accepted results.
  void validatePreparedRebind(const std::shared_ptr<const PreparedDeviceModel>& successor,
      const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& prepared) const
  {
    if (generation_ == std::numeric_limits<std::size_t>::max() ||
        reference_->generation_ == std::numeric_limits<std::size_t>::max() ||
        approximate_->generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Mixed paired leaf generation exhausted");
    model_->validateSuccessor(successor);
    if (!prepared)
      throw std::invalid_argument("Mixed paired successor cache is null");
    validateSharedParameters(successor, prepared);
    if (!allow_retry_ && !prepared->usable())
      throw MixedPrecisionRetryRequired("Shared paired successor cache requires FP64 replay");
    adapter_->validateParameterRebind(prepared);
    reference_->trunk_->validateModelRebind(successor);
    approximate_->trunk_->validateModelRebindWithMixedParameters(successor, prepared);
  }

  /// Adopt only after every composing leaf preflights under the same exclusive owner lease.
  void adoptPreparedRebind(const std::shared_ptr<const PreparedDeviceModel>& successor,
      const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& prepared) noexcept
  {
    adapter_->adoptPreparedParameters(prepared);
    reference_->adoptPreparedModel(successor);
    approximate_->adoptPreparedModel(successor);
    parameters_ = prepared;
    model_ = successor;
    ++generation_;
    selected_ = nullptr;
    used_retry_ = false;
    retry_reason_.clear();
  }

  bool usedFp64Retry() const noexcept { return used_retry_; }
  const std::string& retryReason() const noexcept { return retry_reason_; }
  const MixedValueDenseStatistics& statistics() const noexcept { return adapter_->statistics(); }

  /// Additional peak while admitting a successor cache; the old FP32 cache remains accepted until publication.
  std::size_t successorWorkspaceBytes() const
  {
    if (!owns_parameters_)
      return 0; // The composing owner admits its one prepared successor cache.
    return spatial_detail::checkedAdd(checkedDeviceBytes<float>(model_->plan().parameterCount()),
        sizeof(PsiFormerParameterConversionDiagnostics), "Mixed paired successor bytes overflow");
  }

  /// Explicit numeric HOST staging; shared image, fixed C++ descriptors and vendor storage are excluded.
  std::size_t hostNumericBytes() const
  {
    return spatial_detail::checkedAdd(
        spatial_detail::checkedAdd(reference_->cost().host_bytes, approximate_->cost().host_bytes,
                                  "Mixed paired HOST bytes overflow"),
        sizeof(PsiFormerDeviceNumericalDiagnostics) +
            (owns_parameters_ ? sizeof(PsiFormerParameterConversionDiagnostics) : 0),
        "Mixed paired HOST diagnostics overflow");
  }

  /// Excludes the shared immutable FP64 image and caller-owned vendor/queue reserves.
  std::size_t deviceWorkspaceBytes() const
  {
    const auto add = [](auto a, auto b) {
      return spatial_detail::checkedAdd(a, b, "Mixed paired VALUE live bytes overflow");
    };
    auto bytes = add(reference_->cost().device_bytes, approximate_->cost().device_bytes);
    if (owns_parameters_) bytes = add(bytes, parameters_->deviceStorageBytes());
    bytes = add(bytes, adapter_->deviceWorkspaceBytes());
    return add(bytes, checkedDeviceBytes<double>(pivots_.capacity()));
  }

  /// Identity proves that all borrowed affine weights represent this exact canonical image.
  const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& parameterCache() const noexcept
  {
    return parameters_;
  }

private:
  static void validateSharedParameters(const std::shared_ptr<const PreparedDeviceModel>& image,
      const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& parameters)
  {
    if (parameters && (parameters->modelImage().get() != image.get() ||
        parameters->trunkHeadPolicy() != MixedTrunkHeadPolicy::PAIRED_COMPLEX_FP64_HEAD))
      throw std::invalid_argument("Mixed paired affine cache has a foreign image or trunk policy");
  }

  void advanceGeneration()
  {
    if (generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Mixed paired VALUE generation exhausted");
    ++generation_;
  }
  std::shared_ptr<const PreparedDeviceModel> model_;
  compute::Queue<PlatformKind::CUDA>& queue_;
  const bool owns_parameters_;
  std::unique_ptr<PreparedDeviceComplexValue> reference_, approximate_;
  std::shared_ptr<const PreparedDeviceMixedDenseParameters> parameters_;
  std::shared_ptr<PreparedDeviceMixedValueDense> adapter_;
  DeviceBuffer<double> pivots_;
  PreparedDeviceComplexValue* selected_ = nullptr;
  std::size_t generation_ = 0;
  bool allow_retry_, used_retry_ = false;
  std::string retry_reason_;
};
} // namespace qmcplusplus::psiformer::device
#endif
