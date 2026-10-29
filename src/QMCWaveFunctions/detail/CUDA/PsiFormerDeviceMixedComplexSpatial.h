//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_COMPLEX_SPATIAL_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_COMPLEX_SPATIAL_H
#include "PsiFormerDeviceComplexSpatial.h"

namespace qmcplusplus::psiformer::device
{
/// Numerical ordinary FULL composition with ONE graph and complete same-image FP64 replay.
/// No target, retained, XML, score, optimizer or physical-accuracy admission is granted here.
class PreparedDeviceMixedComplexSpatial
{
public:
  /// Borrow one complete graph result only while this wrapper generation is current.
  struct ResultView
  {
    const PreparedDeviceMixedComplexSpatial* owner;
    std::size_t generation;
    PreparedDeviceComplexSpatial::ResultView graph;
  };

  /// Fold all requested derivative planes into the affine cast rows before allocation.
  static std::size_t maximumRows(const PsiFormerExecutionPlan& plan, std::size_t capacity)
  {
    const auto n = plan.modelShape().electrons();
    return spatial_detail::checkedProduct(spatial_detail::checkedProduct(capacity, n,
        "Mixed paired FULL row extent overflow"), spatial_detail::checkedAdd(1,
        spatial_detail::checkedProduct(4, n, "Mixed paired FULL plane extent overflow"),
        "Mixed paired FULL plane extent overflow"), "Mixed paired FULL row extent overflow");
  }

  /// Admit one complete graph plus the cache/casts/pivots/diagnostics, including cold peak.
  static std::size_t estimateDeviceBytes(const PsiFormerExecutionPlan& plan, std::size_t capacity,
      std::size_t concurrent_targets = 0)
  {
    if (!supportsMixedPairedTrunkEnvironment(plan.environment()))
      throw std::invalid_argument("Mixed paired FULL requires a fixed legacy periodic real trunk");
    const auto add = [](auto a, auto b) {
      return spatial_detail::checkedAdd(a, b, "Mixed paired FULL owner bytes overflow");
    };
    const auto mul = [](auto a, auto b) {
      return spatial_detail::checkedProduct(a, b, "Mixed paired FULL owner extent overflow");
    };
    const auto width = plan.modelShape().feature_dimension;
    const auto input = plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0];
    const auto graph = estimateComplexSpatialCost(plan, capacity, concurrent_targets).device_bytes;
    const auto cache = checkedDeviceBytes<float>(plan.parameterCount());
    auto bytes = add(graph, cache);
    bytes = add(bytes, checkedDeviceBytes<float>(mul(maximumRows(plan, capacity), add(std::max(input, width), width))));
    bytes = add(bytes, checkedDeviceBytes<double>(mul(capacity, plan.modelShape().determinants)));
    bytes = add(bytes, sizeof(PsiFormerDeviceNumericalDiagnostics));
    if (add(add(graph, cache), sizeof(PsiFormerParameterConversionDiagnostics)) > bytes)
      throw std::logic_error("Mixed paired FULL inventory does not dominate cold conversion peak");
    return bytes;
  }

  /// A composing owner charges its one shared immutable FP32 cache separately.
  static std::size_t estimatePrivateDeviceBytes(const PsiFormerExecutionPlan& plan, std::size_t capacity,
      std::size_t concurrent_targets = 0)
  {
    return estimateDeviceBytes(plan, capacity, concurrent_targets) -
        checkedDeviceBytes<float>(plan.parameterCount());
  }

  /// The caller's image/queue/vendor reserves are shared once and admitted separately.
  PreparedDeviceMixedComplexSpatial(std::shared_ptr<const PreparedDeviceModel> image, std::size_t capacity,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, std::size_t budget,
      bool allow_fp64_retry = true, std::size_t concurrent_targets = 0,
      std::shared_ptr<const PreparedDeviceMixedDenseParameters> shared_parameters = {})
      : model_(std::move(image)), queue_(queue), owns_parameters_(!shared_parameters), allow_retry_(allow_fp64_retry)
  {
    if (!model_) throw std::invalid_argument("Mixed paired FULL requires a canonical image");
    validateSharedParameters(model_, shared_parameters);
    const auto bytes = owns_parameters_
        ? estimateDeviceBytes(model_->plan(), capacity, concurrent_targets)
        : estimatePrivateDeviceBytes(model_->plan(), capacity, concurrent_targets);
    if (bytes > budget) throw std::length_error("Mixed paired FULL exceeds its combined owner budget");
    model_->validateCurrentDevice();
    // One immutable trace slab belongs to this same graph on mixed and FP64 attempts.
    graph_ = PreparedDeviceComplexSpatial::prepare(model_, capacity, queue, blas, budget, concurrent_targets);
    parameters_ = std::move(shared_parameters);
    if (owns_parameters_)
      parameters_ = std::make_shared<PreparedDeviceMixedDenseParameters>(model_, queue_,
          MixedTrunkHeadPolicy::PAIRED_COMPLEX_FP64_HEAD, allow_retry_);
    if (!allow_retry_ && !parameters_->usable())
      throw MixedPrecisionRetryRequired("Shared paired affine cache requires FP64 replay");
    adapter_ = std::make_shared<PreparedDeviceMixedValueDense>(parameters_, maximumRows(model_->plan(), capacity), queue, blas);
    pivots_.reserveNamed(spatial_detail::checkedProduct(capacity, model_->plan().modelShape().determinants,
        "Mixed paired FULL pivots overflow"), "MixedComplexSpatial::minimum_scaled_pivots", __FILE__, __LINE__);
    graph_->trunk_->setMixedDenseForPairedTrunk(adapter_, SpatialJetMode::FULL_VGL);
    if (deviceWorkspaceBytes() != bytes)
      throw std::logic_error("Mixed paired FULL owner accounting differs from checked admission");
  }

  /// Retire all borrowed asynchronous views before destroying any private workspace.
  ~PreparedDeviceMixedComplexSpatial()
  { try { queue_.sync(); } catch (...) {} }

  /// Never retain any approximate field when a complete FP64 replay is required.
  void evaluate(const std::vector<double>& positions, bool strict_mixed = false)
  {
    advanceGeneration();
    valid_ = used_retry_ = false;
    retry_reason_.clear();
    try
    {
      graph_->evaluateCoordinates(positions, pivots_.data());
      valid_ = true;
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
    evaluateFp64Graph(positions);
  }

  /// Select FP64 for the complete native epoch without allocating a second FULL graph.
  void evaluateFp64(const std::vector<double>& positions)
  {
    if (!allow_retry_)
      throw std::logic_error("Explicit paired FP64 replay requires the admitted retry policy");
    advanceGeneration();
    valid_ = false;
    retry_reason_ = "native_epoch";
    evaluateFp64Graph(positions);
  }

  /// Publish a ticket only after the whole mixed or same-image FP64 attempt succeeds.
  ResultView resultView() const
  {
    if (!valid_) throw std::logic_error("Mixed paired FULL has no admitted complete result");
    return {this, generation_, graph_->resultView()};
  }

  /// Validate wrapper and graph authority before copying every requested FULL field.
  void download(const ResultView& view, std::vector<ComplexValueResult>& amplitude,
      std::vector<cuDoubleComplex>& gradient, std::vector<cuDoubleComplex>& lap_log,
      std::vector<cuDoubleComplex>& lap_ratio) const
  {
    const auto current = resultView();
    if (view.owner != this || view.generation != current.generation)
      throw std::invalid_argument("Mixed paired FULL result is stale or foreign");
    graph_->download(view.graph, amplitude, gradient, lap_log, lap_ratio);
  }

  /// Admit all potentially throwing successor work before the no-throw publication boundary.
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
        graph_->generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Mixed paired leaf generation exhausted");
    model_->validateSuccessor(successor);
    if (!prepared)
      throw std::invalid_argument("Mixed paired successor cache is null");
    validateSharedParameters(successor, prepared);
    if (!allow_retry_ && !prepared->usable())
      throw MixedPrecisionRetryRequired("Shared paired successor cache requires FP64 replay");
    adapter_->validateParameterRebind(prepared);
    graph_->trunk_->validateModelRebindWithMixedParameters(successor, prepared);
  }

  /// Adopt only after every composing leaf preflights under the same exclusive owner lease.
  void adoptPreparedRebind(const std::shared_ptr<const PreparedDeviceModel>& successor,
      const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& prepared) noexcept
  {
    adapter_->adoptPreparedParameters(prepared);
    graph_->adoptPreparedModel(successor);
    parameters_ = prepared;
    model_ = successor;
    ++generation_;
    valid_ = used_retry_ = false;
    retry_reason_.clear();
  }

  /// Expose actual precision execution without treating an enabled policy as mixed work.
  bool usedFp64Retry() const noexcept { return used_retry_; }

  const std::string& retryReason() const noexcept { return retry_reason_; }

  const MixedValueDenseStatistics& statistics() const noexcept { return adapter_->statistics(); }

  /// Report additional pending cache ownership before successor publication.
  std::size_t successorWorkspaceBytes() const
  {
    if (!owns_parameters_)
      return 0; // The composing owner admits its one prepared successor cache.
    return spatial_detail::checkedAdd(checkedDeviceBytes<float>(model_->plan().parameterCount()),
        sizeof(PsiFormerParameterConversionDiagnostics), "Mixed paired FULL successor bytes overflow");
  }

  /// Keep HOST numeric staging separate from shared model/vendor and object storage.
  std::size_t hostNumericBytes() const
  {
    return spatial_detail::checkedAdd(graph_->cost().host_bytes,
        sizeof(PsiFormerDeviceNumericalDiagnostics) +
            (owns_parameters_ ? sizeof(PsiFormerParameterConversionDiagnostics) : 0),
        "Mixed paired FULL HOST bytes overflow");
  }

  /// Count exactly one complete graph plus its private cache, casts and diagnostics.
  std::size_t deviceWorkspaceBytes() const
  {
    auto bytes = spatial_detail::checkedAdd(graph_->cost().device_bytes,
        owns_parameters_ ? parameters_->deviceStorageBytes() : 0,
        "Mixed paired FULL live bytes overflow");
    bytes = spatial_detail::checkedAdd(bytes, adapter_->deviceWorkspaceBytes(), "Mixed paired FULL live bytes overflow");
    return spatial_detail::checkedAdd(bytes, checkedDeviceBytes<double>(pivots_.capacity()), "Mixed paired FULL live bytes overflow");
  }

  /// Identity proves that all borrowed affine weights represent this exact canonical image.
  const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& parameterCache() const noexcept
  {
    return parameters_;
  }

private:
  /// Detach only while evaluating one complete FP64 graph; restore on success and failure.
  void evaluateFp64Graph(const std::vector<double>& positions)
  {
    // The failed attempt's RAII transaction has cancelled before detachment.
    // Restore privately on every exit; the public setter would revoke replay results.
    struct RestoreAdapter
    {
      std::shared_ptr<PreparedDeviceMixedValueDense>& slot;
      std::shared_ptr<PreparedDeviceMixedValueDense> saved;
      explicit RestoreAdapter(std::shared_ptr<PreparedDeviceMixedValueDense>& selected)
          : slot(selected), saved(std::move(selected)) {}
      ~RestoreAdapter() { slot = std::move(saved); }
    } restore(graph_->trunk_->mixed_dense_);
    used_retry_ = true;
    graph_->evaluate(positions); // This invalidates/reset/overwrites the SAME full graph.
    valid_ = true;
  }

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
      throw std::overflow_error("Mixed paired FULL generation exhausted");
    ++generation_;
  }
  std::shared_ptr<const PreparedDeviceModel> model_;
  compute::Queue<PlatformKind::CUDA>& queue_;
  const bool owns_parameters_;
  std::unique_ptr<PreparedDeviceComplexSpatial> graph_;
  std::shared_ptr<const PreparedDeviceMixedDenseParameters> parameters_;
  std::shared_ptr<PreparedDeviceMixedValueDense> adapter_;
  DeviceBuffer<double> pivots_;
  std::size_t generation_ = 0;
  bool allow_retry_, valid_ = false, used_retry_ = false;
  std::string retry_reason_;
};
} // namespace qmcplusplus::psiformer::device
#endif
