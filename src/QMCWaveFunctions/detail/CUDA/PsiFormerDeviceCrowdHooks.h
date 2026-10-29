//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceCrowdHooks.h
 * @brief Numerical CUDA/HIP value/VGL binding for the accelerator control plane.
 *
 * One exclusive crowd owns its queue, BLAS handle, prepared scratch and typed
 * results. Immutable parameters are shared through the production tile backend.
 * A released crowd can bind a published successor without rebuilding its scratch.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_CROWD_HOOKS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_CROWD_HOOKS_H

#include "PsiFormerDeviceInference.h"
#include "PsiFormerDeviceExecutionPreparation.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorExecutor.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

namespace qmcplusplus::psiformer::device
{

/** Bind an immutable model to real numerical operations for one crowd identity.
 * All requests use the existing executor's lease/provenance protocol. A later
 * production publisher must explicitly bind newer images; none is inferred here.
 */
class PreparedDeviceCrowdHooks final : public PsiFormerAcceleratorHooks
{
public:
  /// Charge the selected spatial graph and shared VALUE/spatial casts/diagnostics before allocation.
  /// Excludes the separate VALUE graph, canonical image and immutable FP32 cache admitted by the backend.
  static std::size_t mixedTargetDeviceBytes(const PsiFormerExecutionPlan& plan, std::size_t capacity,
      TargetVglPreparation preparation)
  {
    if (!preparation.max_targets || preparation.max_targets > plan.modelShape().electrons() || !capacity ||
        !preparation.device_workspace_budget_bytes ||
        preparation.device_workspace_budget_bytes == std::numeric_limits<std::size_t>::max() ||
        !preparation.scalar_work_budget || preparation.scalar_work_budget == std::numeric_limits<std::size_t>::max() ||
        preparation.parallel_trace_workspace_budget_bytes || preparation.reuse_cusp_state)
      throw std::invalid_argument("Mixed real targets require finite caps without optional trace or cusp owners");
    const auto add = [](std::size_t a, std::size_t b) {
      return spatial_detail::checkedAdd(a, b, "Mixed real target owner bytes overflow");
    };
    const auto mul = [](std::size_t a, std::size_t b) {
      return spatial_detail::checkedProduct(a, b, "Mixed real target owner extent overflow");
    };
    const auto cost = estimatePreparedTargetVglCost(plan, capacity, preparation);
    const auto rows = mul(mul(capacity, plan.modelShape().electrons()), add(1, mul(4, preparation.max_targets)));
    const auto width = plan.modelShape().feature_dimension;
    const auto input = plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0];
    const auto casts = add(checkedDeviceBytes<float>(mul(rows, add(std::max(input, width), width))),
        sizeof(PsiFormerDeviceNumericalDiagnostics));
    const auto bytes = add(add(cost.device_workspace_bytes, casts),
        PreparedDeviceCanonicalRealHead::required(plan, capacity).device_bytes);
    if (bytes > preparation.device_workspace_budget_bytes ||
        add(mixedTargetRequestScalarWork(cost.charged_scalar_work_units), canonicalHeadScalarWork(plan, capacity)) >
            preparation.scalar_work_budget)
      throw std::length_error("Mixed real target graph and private casts exceed byte/work caps");
    return bytes;
  }

  /// Construct vendor handles without allocating per-walker model images.
  PreparedDeviceCrowdHooks(std::shared_ptr<const PreparedDeviceModel> image,
                          const PsiFormerAcceleratorSharedModel& owner,
                          std::uint64_t crowd_identity,
                          std::shared_ptr<const PreparedDeviceMixedDenseParameters> mixed_parameters = {},
                          TargetVglPreparation target_preparation = {}, bool lazy_spatial = false)
      : image_(std::move(image)), mixed_parameters_(std::move(mixed_parameters)),
        target_preparation_(target_preparation), lazy_spatial_(lazy_spatial), owner_(owner),
        crowd_identity_(crowd_identity), blas_(queue_)
  {
    if (!image_ || crowd_identity_ == 0)
      throw std::invalid_argument("Numerical crowd hooks require an image and nonzero identity");
    image_->validateCurrentDevice();
    validateDeviceExecutionPreparation(image_->plan(), {target_preparation_, {}});
    if (mixed_parameters_ && target_preparation_.max_targets)
      mixedTargetDeviceBytes(image_->plan(), 1, target_preparation_);
    if (lazy_spatial_ && mixed_parameters_)
      throw std::invalid_argument("Lazy VGL preparation requires FP64 crowd resources");
    const auto provenance = owner_.activeProvenance();
    if (provenance.backend != backend() || provenance.model_version != image_->version())
      throw std::invalid_argument("Numerical crowd model/backend does not match active provenance");
    model_provenance_ = provenance;
  }

  // Prepared graphs retain references to this object's queue/BLAS handle.
  // Relocating an adapter would invalidate those references and hook identity.
  PreparedDeviceCrowdHooks(const PreparedDeviceCrowdHooks&) = delete;
  PreparedDeviceCrowdHooks& operator=(const PreparedDeviceCrowdHooks&) = delete;
  PreparedDeviceCrowdHooks(PreparedDeviceCrowdHooks&&) = delete;
  PreparedDeviceCrowdHooks& operator=(PreparedDeviceCrowdHooks&&) = delete;

  /// Retarget released scratch after the shared owner commits a newer image.
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> image,
                   std::shared_ptr<const PreparedDeviceMixedDenseParameters> mixed_parameters = {})
  {
    if (acquired_ || recorded_ || source_ || failure_)
      throw std::logic_error("Numerical crowd rebind requires released, completed resources");
    image_->validateSuccessor(image);
    const auto provenance = owner_.activeProvenance();
    if (provenance.model_version != image->version() ||
        provenance.layout_fingerprint != model_provenance_.layout_fingerprint)
      throw std::logic_error("Numerical crowd successor does not match its published owner");

    if (mixed_dense_ && (!mixed_parameters || mixed_parameters->modelImage().get() != image.get()))
      throw std::logic_error("Numerical crowd successor has no matching mixed parameter image");
    // Preflight both shared-cache leaves before either graph revokes its current products.
    if (canonical_head_) canonical_head_->validateModelRebind(image);
    if (values_)
      values_->validateModelRebindWithMixedParameters(image, {});
    if (spatial_)
      spatial_->validateModelRebindWithMixedParameters(image, mixed_parameters);
    // Old image/cache members retain lifetime through both no-throw adoptions.
    if (canonical_head_) canonical_head_->adoptPreparedModel(image);
    if (values_)
      values_->adoptPreparedModelWithMixedParameters(image, {});
    if (spatial_)
      spatial_->adoptPreparedModelWithMixedParameters(image, mixed_parameters);
    image_ = std::move(image);
    mixed_parameters_ = std::move(mixed_parameters);
    model_provenance_ = provenance;
    last_graph_ = nullptr;
    ready_ = false;
  }

  /// Report the resident image version so a released context can catch up lazily.
  std::size_t modelVersion() const noexcept { return image_->version(); }

  /// Return precisely the numerical operations implemented by this adapter.
  static constexpr PsiFormerAcceleratorOperationMask coverage() noexcept
  {
    return acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES) |
        acceleratorOperationBit(PsiFormerAcceleratorOperation::VGL);
  }

  /// Report the compiled vendor rather than accepting a simulated backend.
  static constexpr PsiFormerAcceleratorBackend backend() noexcept
  {
#ifdef QMC_CUDA2HIP
    return PsiFormerAcceleratorBackend::HIP;
#else
    return PsiFormerAcceleratorBackend::CUDA;
#endif
  }

  /// Borrow a coordinate prefix until upload, with explicit operation/generation.
  void bindRequest(const std::vector<double>& positions,
                   PsiFormerAcceleratorOperationMask operations,
                   std::size_t coordinate_generation)
  {
    if (!acquired_ || source_ || recorded_ || failure_)
      throw std::logic_error("Numerical request requires an acquired idle crowd");
    const auto coordinates_per_configuration = coordinatesPerConfiguration();
    if (operations == 0 || (operations & ~coverage()) != 0 || coordinate_generation == 0 ||
        positions.empty() || positions.size() % coordinates_per_configuration != 0 ||
        positions.size() / coordinates_per_configuration > capacity_)
      throw std::invalid_argument("Numerical crowd request has invalid operations, coordinates or generation");
    if (!values_ || ((operations & acceleratorOperationBit(PsiFormerAcceleratorOperation::VGL)) && !spatial_))
      throw std::logic_error("Numerical request has no prepared graph for its operation");
    source_ = &positions;
    request_operations_ = operations;
    executed_operations_ = 0;
    coordinate_generation_ = coordinate_generation;
    ready_ = false;
    last_graph_ = nullptr;
  }

  /// Execute checked control-plane events without changing their ownership rules.
  void submit(const PsiFormerAcceleratorEvent& event) override
  {
    if (event.backend != backend() || event.crowd_identity != crowd_identity_ ||
        event.layout_fingerprint != model_provenance_.layout_fingerprint)
      throw std::logic_error("Numerical crowd event has mismatched routing");
    if (event.kind != PsiFormerAcceleratorEventKind::CROWD_PREPARE &&
        event.resource_generation != resource_generation_)
      throw std::logic_error("Numerical crowd event names stale prepared storage");
    switch (event.kind)
    {
    case PsiFormerAcceleratorEventKind::CROWD_PREPARE:
      prepare(event.item_count);
      resource_generation_ = event.resource_generation;
      return;
    case PsiFormerAcceleratorEventKind::CROWD_ACQUIRE:
      validateModel(event);
      if (!values_ || (!spatial_ && !lazy_spatial_) || acquired_)
        throw std::logic_error("Numerical crowd acquire has unprepared or borrowed storage");
      acquired_ = true;
      acquisition_generation_ = event.acquisition_generation;
      ready_ = false;
      return;
    case PsiFormerAcceleratorEventKind::MODEL_WAIT:
      validateModel(event);
      return;
    case PsiFormerAcceleratorEventKind::INPUT_UPLOAD:
      validateModel(event);
      if (!source_ || event.coordinate_generation != coordinate_generation_ ||
          event.acquisition_generation != acquisition_generation_ ||
          event.item_count != source_->size() / coordinatesPerConfiguration())
        throw std::logic_error("Numerical crowd upload has stale or unbound coordinates");
      positions_.assign(source_->begin(), source_->end());
      source_ = nullptr;
      result_generation_ = event.result_generation;
      item_count_ = event.item_count;
      return;
    case PsiFormerAcceleratorEventKind::OPERATION:
      validateResultEvent(event);
      if ((request_operations_ & acceleratorOperationBit(event.operation)) == 0)
        throw std::logic_error("Numerical event operation is absent from the bound request");
      executed_operations_ |= acceleratorOperationBit(event.operation);
      // Never throw after a launch from an OPERATION hook: the control plane
      // must know that earlier work exists before it enters poison/recovery.
      try
      {
        if (!values_ || (event.operation == PsiFormerAcceleratorOperation::VGL && !spatial_))
          throw std::logic_error("Numerical dispatch has incomplete graphs after failed preparation; rebuild before retry");
        if (event.operation == PsiFormerAcceleratorOperation::VGL)
        {
          evaluateGraph(*spatial_);
          last_graph_ = spatial_.get();
          ++numerical_batches_;
        }
        else if (event.operation == PsiFormerAcceleratorOperation::VALUES)
        {
          // A VGL request already supplies values; do not run two graphs.
          if ((request_operations_ & acceleratorOperationBit(PsiFormerAcceleratorOperation::VGL)) == 0)
          {
            evaluateGraph(*values_);
            last_graph_ = values_.get();
            ++numerical_batches_;
          }
        }
        else
          throw std::logic_error("Numerical crowd operation is not implemented");
      }
      catch (...)
      {
        failure_ = std::current_exception();
      }
      return;
    case PsiFormerAcceleratorEventKind::RESULT_RECORD:
      validateResultEvent(event);
      if (failure_)
        std::rethrow_exception(failure_);
      if (executed_operations_ != request_operations_)
        throw std::logic_error("Numerical request and control-plane operation masks differ");
      if (!last_graph_)
        throw std::logic_error("Numerical crowd has no computed result to record");
      phases_.resize(event.item_count);
      jets_.resize(spatialJetSpanElements(last_graph_->resultLayout()));
      ratios_.resize(event.item_count * last_graph_->resultLayout().laplacian_lanes);
      // Synchronous transfers are the initial baseline. Results remain hidden
      // until RESULT_WAIT, even though the copy is complete at this boundary.
      last_graph_->downloadResults(phases_, jets_, ratios_);
      recorded_ = true;
      return;
    case PsiFormerAcceleratorEventKind::RESULT_WAIT:
      validateResultEvent(event);
      if (!recorded_)
        throw std::logic_error("Numerical crowd wait has no recorded result");
      recorded_ = false;
      ready_ = true;
      return;
    case PsiFormerAcceleratorEventKind::CROWD_RELEASE:
      validateModel(event);
      if (recorded_ || source_ || failure_)
        throw std::logic_error("Numerical crowd cannot release incomplete work");
      acquired_ = false;
      return;
    case PsiFormerAcceleratorEventKind::CROWD_RECOVER:
      // Synchronization must succeed before leases/resources can be released.
      queue_.sync();
      source_ = nullptr;
      failure_ = nullptr;
      last_graph_ = nullptr;
      acquired_ = recorded_ = ready_ = false;
      return;
    default:
      throw std::logic_error("Numerical inference hooks do not implement model publication");
    }
  }

  /// Expose phase results only after the matching executor wait has succeeded.
  const std::vector<double>& phases() const { requireReady(); return phases_; }

  /// Expose compact signed-log/spatial output through its typed layout.
  const std::vector<double>& jets() const { requireReady(); return jets_; }

  /// Expose Laplacian ratios only for a completed VGL request.
  const std::vector<double>& laplacianRatios() const { requireReady(); return ratios_; }

  /// Preserve canonical lane strides for the completed logical batch.
  const SpatialJetLayout& resultLayout() const { requireReady(); return last_graph_->resultLayout(); }

  /// Count real numerical batches, not recording-only events or CPU fallbacks.
  std::size_t numericalBatches() const noexcept { return numerical_batches_; }

  /// Identify the spatial graph prepared from the explicit target profile.
  std::size_t vglTargetCapacity() const noexcept { return target_preparation_.max_targets; }

  /// Count actual approximate successes separately from complete FP64 numerical retries.
  std::size_t mixedBatches() const noexcept { return mixed_batches_; }
  std::size_t mixedRetries() const noexcept { return mixed_retries_; }
  std::size_t canonicalValueBatches() const noexcept { return canonical_value_batches_; }
  const std::string& mixedRetryReason() const noexcept { return mixed_retry_reason_; }

  /// Freeze one native epoch policy only while this physical crowd is released.
  void setTrainingEpochMode(PsiFormerGpuTrainingEpochMode mode)
  {
    if (acquired_ || recorded_ || source_ || failure_)
      throw std::logic_error("Physical inference epoch selection requires an idle crowd");
    training_epoch_mode_ = mode;
  }

private:
  // The named sampler borrows this crowd's existing VALUE graph under its lease.
  friend class PreparedDeviceBatchBackend;
  friend struct RealEnvelopeValidationAccess;

  /// Cold-upgrade released VALUE storage without replacing its graph or borrowers.
  bool prepareSpatialGraph()
  {
    if (acquired_ || recorded_ || source_ || failure_ || !values_)
      throw std::logic_error("Spatial preparation requires released, completed VALUE resources");
    if (spatial_)
      return false;
    last_graph_ = nullptr;
    ready_ = false;
    auto spatial = std::make_unique<PreparedDeviceInference>(image_, capacity_, queue_, blas_,
        SpatialJetMode::FULL_VGL, false, false, values_->statusPolicy(), target_preparation_);
    std::vector<double> jets, ratios;
    jets.reserve(spatialJetSpanElements(spatial->resultLayout()));
    ratios.reserve(spatial_detail::checkedProduct(capacity_, image_->plan().modelShape().electrons(),
                                                 "Numerical ratio staging overflow"));
    spatial_ = std::move(spatial);
    jets_.swap(jets);
    ratios_.swap(ratios);
    return true;
  }

  /// Preserve atomic result publication while applying the explicitly selected numerical policy.
  void evaluateGraph(PreparedDeviceInference& graph)
  {
    bool retried = false;
    mixed_retry_reason_.clear();
    if (training_epoch_mode_ == PsiFormerGpuTrainingEpochMode::FP64_REPLAY)
    {
      graph.evaluateFp64(positions_);
      return; // Explicit native replay is separate from inference's automatic retry counter.
    }
    if (mixed_dense_ && &graph == values_.get())
    {
      // Scalar MC/ECP calls define the same full-FP64 primal as the sensitive head.
      graph.evaluate(positions_);
      ++canonical_value_batches_;
      return;
    }
    if (canonical_head_)
    {
      try
      {
        const auto head = canonical_head_->evaluate(positions_);
        graph.evaluateCanonicalHead(head);
      }
      catch (const MixedPrecisionRetryRequired& error)
      {
        if (training_epoch_mode_ == PsiFormerGpuTrainingEpochMode::STRICT_MIXED) throw;
        mixed_retry_reason_ = error.what(); retried = true;
      }
      catch (const InferenceStatusError& error)
      {
        if (training_epoch_mode_ == PsiFormerGpuTrainingEpochMode::STRICT_MIXED || error.stage() == 0) throw;
        mixed_retry_reason_ = error.what(); retried = true;
      }
      if (retried) graph.evaluateFp64(positions_);
    }
    else if (training_epoch_mode_ == PsiFormerGpuTrainingEpochMode::STRICT_MIXED)
      graph.evaluate(positions_);
    else
      retried = graph.evaluateWithFp64Retry(positions_, mixed_retry_reason_);
    if (mixed_dense_)
    {
      if (retried)
        ++mixed_retries_;
      else
      {
        if (mixed_dense_->statistics().projections == 0)
          throw std::logic_error("Mixed inference completed without an affine projection");
        ++mixed_batches_;
      }
    }
  }

  /// Upgrade only the existing VALUE graph after the backend admits all drift additions.
  void retainValueTape()
  {
    if (retain_values_)
      return;
    if (acquired_ || mixed_parameters_)
      throw std::logic_error("Drift VALUE upgrade needs released FP64 resources");
    last_graph_ = nullptr;
    ready_ = false;
    const auto status_policy = values_ ? values_->statusPolicy() : configuredInferenceStatusPolicy();
    values_.reset(); // Never overlap ordinary and retained VALUE graphs.
    values_ = std::make_unique<PreparedDeviceInference>(image_, capacity_, queue_, blas_,
        SpatialJetMode::VALUE_ONLY, true, false, status_policy);
    retain_values_ = true;
  }

  /// Allocate admitted graphs/staging before acquisition; preserve a completed FULL upgrade on recovery.
  void prepare(std::size_t capacity)
  {
    if (acquired_ || capacity == 0)
      throw std::logic_error("Numerical crowd preparation requires released positive capacity");
    validateDeviceExecutionPreparation(image_->plan(), {target_preparation_, {}}, capacity);
    if (mixed_parameters_ && target_preparation_.max_targets)
      mixedTargetDeviceBytes(image_->plan(), capacity, target_preparation_);
    const auto status_policy = configuredInferenceStatusPolicy();
    auto values = std::make_unique<PreparedDeviceInference>(image_, capacity, queue_, blas_, SpatialJetMode::VALUE_ONLY,
        retain_values_, false, status_policy);
    std::unique_ptr<PreparedDeviceInference> spatial;
    if (!lazy_spatial_ || spatial_)
      spatial = std::make_unique<PreparedDeviceInference>(image_, capacity, queue_, blas_,
          SpatialJetMode::FULL_VGL, false, false, status_policy, target_preparation_);
    std::shared_ptr<PreparedDeviceMixedValueDense> mixed_dense;
    if (mixed_parameters_)
    {
      mixed_dense = std::make_shared<PreparedDeviceMixedValueDense>(mixed_parameters_,
          spatial->mixedDenseRowCapacity(), queue_, blas_);
      values->configureCanonicalScalarValue();
      spatial->setMixedDense(mixed_dense);
    }
    std::vector<double> positions, phases, jets, ratios;
    positions.reserve(spatial_detail::checkedProduct(capacity, coordinatesPerConfiguration(),
                                                   "Numerical coordinate staging overflow"));
    phases.reserve(capacity);
    jets.reserve(spatialJetSpanElements(spatial ? spatial->resultLayout() : values->resultLayout()));
    if (spatial)
      ratios.reserve(spatial_detail::checkedProduct(capacity, image_->plan().modelShape().electrons(),
                                                   "Numerical ratio staging overflow"));
    values_ = std::move(values);
    spatial_ = std::move(spatial);
    mixed_dense_ = std::move(mixed_dense);
    if (mixed_parameters_)
      canonical_head_ = PreparedDeviceCanonicalRealHead::prepare(image_, capacity, queue_, blas_,
          PreparedDeviceCanonicalRealHead::required(image_->plan(), capacity).device_bytes);
    positions_.swap(positions);
    phases_.swap(phases);
    jets_.swap(jets);
    ratios_.swap(ratios);
    capacity_ = capacity;
    last_graph_ = nullptr;
    ready_ = false;
  }

  /// Reject a control event naming another immutable image/owner/version.
  void validateModel(const PsiFormerAcceleratorEvent& event) const
  {
    if (event.model_identity != reinterpret_cast<std::uintptr_t>(&owner_) ||
        event.model_version != image_->version() || event.model_slot != model_provenance_.active_slot ||
        event.publication_generation != model_provenance_.publication_generation)
      throw std::logic_error("Numerical crowd event names a stale or foreign model image");
  }

  /// Check coordinate/result generations before exposing or executing results.
  void validateResultEvent(const PsiFormerAcceleratorEvent& event) const
  {
    validateModel(event);
    if (!acquired_ || event.coordinate_generation != coordinate_generation_ ||
        event.result_generation != result_generation_ || event.item_count != item_count_ ||
        event.acquisition_generation != acquisition_generation_)
      throw std::logic_error("Numerical crowd event names stale result provenance");
  }

  /// Guard typed host views against observing an unfinished result.
  void requireReady() const
  {
    if (!ready_ || !last_graph_)
      throw std::logic_error("Numerical crowd result requested before completion");
  }

  /// Derive the coordinate extent with the same checked arithmetic as preparation.
  std::size_t coordinatesPerConfiguration() const
  {
    return spatial_detail::checkedProduct(3, image_->plan().modelShape().electrons(),
                                         "Numerical configuration coordinate overflow");
  }

  std::shared_ptr<const PreparedDeviceModel> image_;
  std::shared_ptr<const PreparedDeviceMixedDenseParameters> mixed_parameters_;
  const TargetVglPreparation target_preparation_;
  const bool lazy_spatial_;
  std::unique_ptr<PreparedDeviceCanonicalRealHead> canonical_head_;
  std::shared_ptr<PreparedDeviceMixedValueDense> mixed_dense_;
  std::size_t mixed_batches_ = 0, mixed_retries_ = 0, canonical_value_batches_ = 0;
  std::string mixed_retry_reason_;
  PsiFormerGpuTrainingEpochMode training_epoch_mode_ = PsiFormerGpuTrainingEpochMode::INFERENCE;
  const PsiFormerAcceleratorSharedModel& owner_;
  const std::uint64_t crowd_identity_;
  PsiFormerAcceleratorModelProvenance model_provenance_;
  compute::Queue<PlatformKind::CUDA> queue_;
  AcceleratorBlasHandle blas_;
  std::unique_ptr<PreparedDeviceInference> values_, spatial_;
  bool retain_values_ = false;
  PreparedDeviceInference* last_graph_ = nullptr;
  std::vector<double> positions_, phases_, jets_, ratios_;
  const std::vector<double>* source_ = nullptr;
  std::exception_ptr failure_;
  PsiFormerAcceleratorOperationMask request_operations_ = 0;
  PsiFormerAcceleratorOperationMask executed_operations_ = 0;
  std::size_t capacity_ = 0, coordinate_generation_ = 0, result_generation_ = 0, numerical_batches_ = 0;
  std::size_t resource_generation_ = 0, acquisition_generation_ = 0, item_count_ = 0;
  bool acquired_ = false, recorded_ = false, ready_ = false;
};

} // namespace qmcplusplus::psiformer::device

#endif
