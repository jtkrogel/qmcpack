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
#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorExecutor.h"

#include <exception>
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
  /// Construct vendor handles without allocating per-walker model images.
  PreparedDeviceCrowdHooks(std::shared_ptr<const PreparedDeviceModel> image,
                          const PsiFormerAcceleratorSharedModel& owner,
                          std::uint64_t crowd_identity,
                          std::shared_ptr<const PreparedDeviceMixedDenseParameters> mixed_parameters = {})
      : image_(std::move(image)), mixed_parameters_(std::move(mixed_parameters)),
        owner_(owner), crowd_identity_(crowd_identity), blas_(queue_)
  {
    if (!image_ || crowd_identity_ == 0)
      throw std::invalid_argument("Numerical crowd hooks require an image and nonzero identity");
    image_->validateCurrentDevice();
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

    if (mixed_dense_)
    {
      if (!mixed_parameters || mixed_parameters->modelImage().get() != image.get())
        throw std::logic_error("Numerical crowd successor has no matching mixed parameter image");
      mixed_dense_->rebindParameters(mixed_parameters);
    }
    // Both graphs belong to the same metadata family. These host-only rebinds
    // invalidate old products without reallocating any prepared device storage.
    if (values_)
      values_->rebindModel(image);
    if (spatial_)
      spatial_->rebindModel(image);
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
      if (!values_ || !spatial_ || acquired_)
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

  /// Count actual approximate successes separately from complete FP64 numerical retries.
  std::size_t mixedBatches() const noexcept { return mixed_batches_; }
  std::size_t mixedRetries() const noexcept { return mixed_retries_; }
  const std::string& mixedRetryReason() const noexcept { return mixed_retry_reason_; }

private:
  /// Preserve atomic result publication while applying the explicitly selected numerical policy.
  void evaluateGraph(PreparedDeviceInference& graph)
  {
    const bool retried = graph.evaluateWithFp64Retry(positions_, mixed_retry_reason_);
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

  /// Allocate all crowd scratch/staging before any request can borrow it.
  void prepare(std::size_t capacity)
  {
    if (acquired_ || capacity == 0)
      throw std::logic_error("Numerical crowd preparation requires released positive capacity");
    auto values = std::make_unique<PreparedDeviceInference>(image_, capacity, queue_, blas_, SpatialJetMode::VALUE_ONLY);
    auto spatial = std::make_unique<PreparedDeviceInference>(image_, capacity, queue_, blas_);
    if (mixed_parameters_)
    {
      const auto rows = spatial_detail::checkedProduct(
          spatial_detail::checkedProduct(capacity, image_->plan().modelShape().electrons(),
                                         "Mixed crowd electron rows overflow"),
          spatial->resultLayout().plane_count, "Mixed crowd spatial rows overflow");
      mixed_dense_ = std::make_shared<PreparedDeviceMixedValueDense>(mixed_parameters_, rows, queue_, blas_);
      values->setMixedDense(mixed_dense_);
      spatial->setMixedDense(mixed_dense_);
    }
    std::vector<double> positions, phases, jets, ratios;
    positions.reserve(spatial_detail::checkedProduct(capacity, coordinatesPerConfiguration(),
                                                   "Numerical coordinate staging overflow"));
    phases.reserve(capacity);
    jets.reserve(spatialJetSpanElements(spatial->resultLayout()));
    ratios.reserve(spatial_detail::checkedProduct(capacity, image_->plan().modelShape().electrons(),
                                                "Numerical ratio staging overflow"));
    values_ = std::move(values);
    spatial_ = std::move(spatial);
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
  std::shared_ptr<PreparedDeviceMixedValueDense> mixed_dense_;
  std::size_t mixed_batches_ = 0, mixed_retries_ = 0;
  std::string mixed_retry_reason_;
  const PsiFormerAcceleratorSharedModel& owner_;
  const std::uint64_t crowd_identity_;
  PsiFormerAcceleratorModelProvenance model_provenance_;
  compute::Queue<PlatformKind::CUDA> queue_;
  AcceleratorBlasHandle blas_;
  std::unique_ptr<PreparedDeviceInference> values_, spatial_;
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
