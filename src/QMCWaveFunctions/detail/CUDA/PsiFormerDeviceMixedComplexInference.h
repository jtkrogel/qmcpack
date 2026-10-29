//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_COMPLEX_INFERENCE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_COMPLEX_INFERENCE_H
#include "PsiFormerDeviceMixedComplexValue.h"
#include "PsiFormerDeviceMixedComplexSpatial.h"
#include <map>

namespace qmcplusplus::psiformer::device
{
/// Explicit inference-only composition: two VALUE graphs and ONE complete FULL graph.
/// The immutable FP64 image, queue and vendor resources are shared and charged outside.
class PreparedDeviceMixedComplexInference
{
public:
  /// Separate live DEVICE/HOST owners from temporary pending conversion admission.
  struct Cost
  {
    std::size_t value_device_bytes, full_device_bytes, device_bytes, host_bytes, pending_device_bytes;
  };

  /// Count only completed native graph requests; hard failures never publish success.
  struct Counters
  {
    std::size_t value_attempts = 0, value_requests = 0, value_mixed = 0, value_replays = 0;
    std::size_t full_attempts = 0, full_requests = 0, full_mixed = 0, full_replays = 0;
    std::size_t value_epoch_replays = 0, full_epoch_replays = 0;
  };

  /// Sum private graphs and ONE shared FP32 cache, plus one pending successor conversion.
  static Cost required(const PsiFormerExecutionPlan& plan, std::size_t capacity,
      std::size_t concurrent_targets = 0, bool owns_parameters = true)
  {
    const auto add = [](auto a, auto b) {
      return spatial_detail::checkedAdd(a, b, "Paired mixed inference ownership overflow");
    };
    const auto value = PreparedDeviceMixedComplexValue::estimatePrivateDeviceBytes(plan, capacity);
    const auto full = PreparedDeviceMixedComplexSpatial::estimatePrivateDeviceBytes(plan, capacity, concurrent_targets);
    const auto host = add(add(estimateComplexValueCost(plan, capacity, true, false).host_bytes,
                              estimateComplexValueCost(plan, capacity, true, false).host_bytes),
                          estimateComplexSpatialCost(plan, capacity, concurrent_targets).host_bytes);
    const auto diagnostics = add(2 * sizeof(PsiFormerDeviceNumericalDiagnostics),
        owns_parameters ? sizeof(PsiFormerParameterConversionDiagnostics) : 0);
    const auto shared_cache = owns_parameters ? checkedDeviceBytes<float>(plan.parameterCount()) : 0;
    const auto pending = owns_parameters ? add(checkedDeviceBytes<float>(plan.parameterCount()),
        sizeof(PsiFormerParameterConversionDiagnostics)) : 0;
    return {value, full, add(add(value, full), shared_cache), add(host, diagnostics), pending};
  }

  /// Reject training, targets and nonlegacy profiles before any numeric owner exists.
  static void validateConfiguration(const ExecutionEnvironment& environment, bool optimizing, std::size_t targets)
  {
    if (optimizing || targets || !supportsMixedPairedTrunkEnvironment(environment))
      throw std::invalid_argument("Paired mixed capability is legacy ordinary FULL inference only");
  }

  /// Pure cold admission is testable at exact-minus-one without allocating a graph.
  static Cost admittedCost(const PsiFormerExecutionPlan& plan, std::size_t capacity, std::size_t owner_budget,
      std::size_t concurrent_targets = 0, bool owns_parameters = true)
  {
    validateConfiguration(plan.environment(), false, 0);
    const auto cost = required(plan, capacity, concurrent_targets, owns_parameters);
    if (spatial_detail::checkedAdd(cost.device_bytes, cost.pending_device_bytes,
          "Paired mixed inference pending ownership overflow") > owner_budget)
      throw std::length_error("Paired mixed inference owners and pending caches exceed admission");
    return cost;
  }

  /// Construction never allocates an uncharged shadow FULL graph or changes FP64 islands.
  PreparedDeviceMixedComplexInference(std::shared_ptr<const PreparedDeviceModel> image,
      std::size_t capacity, compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas,
      std::size_t owner_budget, std::size_t concurrent_targets = 0,
      std::shared_ptr<const PreparedDeviceMixedDenseParameters> shared_parameters = {})
      : image_(std::move(image)),
        cost_(admittedCost(checkedPlan(image_), capacity, owner_budget, concurrent_targets, !shared_parameters)),
        owns_parameters_(!shared_parameters)
  {
    if (shared_parameters && (shared_parameters->modelImage().get() != image_.get() ||
        shared_parameters->trunkHeadPolicy() != MixedTrunkHeadPolicy::PAIRED_COMPLEX_FP64_HEAD))
      throw std::invalid_argument("Paired mixed inference cache has a foreign canonical image or head policy");
    parameters_ = std::move(shared_parameters);
    if (owns_parameters_)
      parameters_ = std::make_shared<PreparedDeviceMixedDenseParameters>(image_, queue,
          MixedTrunkHeadPolicy::PAIRED_COMPLEX_FP64_HEAD, true);
    value_ = std::make_unique<PreparedDeviceMixedComplexValue>(image_, capacity, queue, blas,
        cost_.value_device_bytes, true, parameters_);
    full_ = std::make_unique<PreparedDeviceMixedComplexSpatial>(image_, capacity, queue, blas,
        cost_.full_device_bytes, true, concurrent_targets, parameters_);
    if (value_->parameterCache().get() != parameters_.get() || full_->parameterCache().get() != parameters_.get() ||
        value_->deviceWorkspaceBytes() + full_->deviceWorkspaceBytes() +
            (owns_parameters_ ? parameters_->deviceStorageBytes() : 0) != cost_.device_bytes ||
        value_->hostNumericBytes() + full_->hostNumericBytes() +
            (owns_parameters_ ? sizeof(PsiFormerParameterConversionDiagnostics) : 0) != cost_.host_bytes)
      throw std::logic_error("Paired mixed inference actual owner inventory differs from cold admission");
  }

  /// Inference-only lanes cannot publish parameter updates or partially catch up owners.
  void validateImage(const std::shared_ptr<const PreparedDeviceModel>& current) const
  {
    if (!current)
      throw std::invalid_argument("Paired mixed inference current image is null");
    current->validateCurrentDevice();
    if (current != image_)
      throw std::logic_error("Paired mixed inference does not admit parameter publication or training");
  }

  /// Preflight BOTH numerical leaves before any active cache/image can change.
  void validatePreparedRebind(const std::shared_ptr<const PreparedDeviceModel>& successor,
      const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& prepared) const
  {
    image_->validateSuccessor(successor);
    if (!prepared || prepared->modelImage().get() != successor.get() ||
        prepared->trunkHeadPolicy() != MixedTrunkHeadPolicy::PAIRED_COMPLEX_FP64_HEAD)
      throw std::invalid_argument("Paired mixed successor cache has a foreign image or head policy");
    value_->validatePreparedRebind(successor, prepared);
    full_->validatePreparedRebind(successor, prepared);
  }

  /// Existing native publication retains the old image/cache until this boundary completes.
  void adoptPreparedRebind(const std::shared_ptr<const PreparedDeviceModel>& successor,
      const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& prepared) noexcept
  {
    value_->adoptPreparedRebind(successor, prepared);
    full_->adoptPreparedRebind(successor, prepared);
    parameters_ = prepared;
    image_ = successor;
  }

  /// Consume the complete selected VALUE result before returning a native amplitude.
  void evaluateValue(const std::vector<double>& positions,
      PsiFormerGpuTrainingEpochMode mode = PsiFormerGpuTrainingEpochMode::INFERENCE)
  {
    validateEpochMode(mode);
    ++counters_.value_attempts;
    if (mode == PsiFormerGpuTrainingEpochMode::FP64_REPLAY)
      value_->evaluateFp64(positions);
    else
      value_->evaluate(positions, mode == PsiFormerGpuTrainingEpochMode::STRICT_MIXED);
    ++counters_.value_requests;
    if (mode == PsiFormerGpuTrainingEpochMode::FP64_REPLAY) ++counters_.value_epoch_replays;
    if (value_->usedFp64Retry()) { ++counters_.value_replays; recordReason(value_reasons_, value_->retryReason()); }
    else ++counters_.value_mixed;
  }

  /// Copy only the complete admitted VALUE output into native staging.
  void downloadValue(std::vector<ComplexValueResult>& amplitude) const
  { value_->download(value_->resultView(), amplitude); }

  /// Consume all fields from one admitted mixed or complete same-image FP64 FULL attempt.
  void evaluateFull(const std::vector<double>& positions,
      PsiFormerGpuTrainingEpochMode mode = PsiFormerGpuTrainingEpochMode::INFERENCE)
  {
    validateEpochMode(mode);
    ++counters_.full_attempts;
    if (mode == PsiFormerGpuTrainingEpochMode::FP64_REPLAY)
      full_->evaluateFp64(positions);
    else
      full_->evaluate(positions, mode == PsiFormerGpuTrainingEpochMode::STRICT_MIXED);
    ++counters_.full_requests;
    if (mode == PsiFormerGpuTrainingEpochMode::FP64_REPLAY) ++counters_.full_epoch_replays;
    if (full_->usedFp64Retry()) { ++counters_.full_replays; recordReason(full_reasons_, full_->retryReason()); }
    else ++counters_.full_mixed;
  }

  /// Copy ALL admitted FULL fields after the native evaluation timer closes.
  void downloadFull(std::vector<ComplexValueResult>& amplitude, std::vector<cuDoubleComplex>& gradient,
      std::vector<cuDoubleComplex>& lap_log, std::vector<cuDoubleComplex>& lap_ratio) const
  { full_->download(full_->resultView(), amplitude, gradient, lap_log, lap_ratio); }

  /// Numeric inventory is distinct from successful request counters and retry reasons.
  const Cost& cost() const noexcept { return cost_; }

  /// Report attempted, completed mixed and complete replay requests separately.
  const Counters& counters() const noexcept { return counters_; }

  /// Explain the most recent VALUE precision selection.
  const std::string& valueRetryReason() const noexcept { return value_->retryReason(); }

  /// Explain the most recent FULL precision selection.
  const std::string& fullRetryReason() const noexcept { return full_->retryReason(); }

  /// Summarize bounded VALUE replay causes without unbounded diagnostic growth.
  const std::map<std::string, std::size_t>& valueReasons() const noexcept { return value_reasons_; }

  /// Summarize bounded FULL replay causes independently of VALUE requests.
  const std::map<std::string, std::size_t>& fullReasons() const noexcept { return full_reasons_; }

  /// Share immutable affine authority without exposing any mutable graph scratch.
  const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& parameterCache() const noexcept
  {
    return parameters_;
  }

private:
  /// Reject unknown epoch policy before any request counter or numeric state changes.
  static void validateEpochMode(PsiFormerGpuTrainingEpochMode mode)
  {
    switch (mode)
    {
    case PsiFormerGpuTrainingEpochMode::INFERENCE:
    case PsiFormerGpuTrainingEpochMode::STRICT_MIXED:
    case PsiFormerGpuTrainingEpochMode::FP64_REPLAY: return;
    }
    throw std::invalid_argument("Paired native numerical epoch mode is invalid");
  }

  /// Bound diagnostic metadata even when failure indices differ over long continuations.
  static void recordReason(std::map<std::string, std::size_t>& reasons, const std::string& reason)
  {
    const auto found = reasons.find(reason);
    if (found != reasons.end())
      ++found->second;
    else if (reasons.size() < 8)
      reasons.emplace(reason, 1);
    else
      ++reasons["other_replay_reasons"];
  }

  /// Validate the shared canonical image before reading plan metadata in initialization.
  static const PsiFormerExecutionPlan& checkedPlan(const std::shared_ptr<const PreparedDeviceModel>& image)
  {
    if (!image) throw std::invalid_argument("Paired mixed inference requires an immutable FP64 image");
    return image->plan();
  }
  std::shared_ptr<const PreparedDeviceModel> image_;
  const Cost cost_;
  const bool owns_parameters_;
  std::shared_ptr<const PreparedDeviceMixedDenseParameters> parameters_;
  std::unique_ptr<PreparedDeviceMixedComplexValue> value_;
  std::unique_ptr<PreparedDeviceMixedComplexSpatial> full_;
  Counters counters_;
  std::map<std::string, std::size_t> value_reasons_, full_reasons_;
};
} // namespace qmcplusplus::psiformer::device
#endif
