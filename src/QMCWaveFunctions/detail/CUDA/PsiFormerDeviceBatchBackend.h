//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceBatchBackend.h
 * @brief Bind native batch tile publication to prepared numerical device crowds.
 *
 * Include after the native value/spatial/batch executor definitions. Host batch
 * ownership, sparse coordinate materialization and atomic commit remain intact.
 * Publication preserves the native/device version barrier. Training product
 * dispatch and device-aware hard plans have separate capability gates.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_BATCH_BACKEND_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_BATCH_BACKEND_H

#include "PsiFormerDeviceCrowdHooks.h"
#include "PsiFormerDeviceProductSession.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerBatchExecutor.h"

#include <atomic>
#include <functional>
#include <iostream>
#include <mutex>

namespace qmcplusplus::psiformer::device
{

/** Supply exclusive GPU scratch per batch workspace and one shared model image.
 * The control mutex protects non-atomic model leases. The first synchronous
 * implementation serializes submissions; asynchronous overlap is not claimed.
 */
class PreparedDeviceBatchBackend final : public pf::DirectBatchTileBackend
{
public:
  /// Bind one immutable parameter version and its canonical layout fingerprint.
  explicit PreparedDeviceBatchBackend(std::shared_ptr<const PreparedDeviceModel> image,
                                      std::function<void()> first_dispatch_reporter = {},
                                      PsiFormerPrecisionPolicy precision = PsiFormerPrecisionPolicy::FP64_REFERENCE)
      : image_(checkedImage(std::move(image))),
        owner_(PreparedDeviceCrowdHooks::backend(),
               makePsiFormerDeviceLayout(image_->plan()).fingerprint, image_->version()),
        first_dispatch_reporter_(std::move(first_dispatch_reporter))
  {
    if (precision == PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS)
    {
      preparation_queue_ = std::make_unique<compute::Queue<PlatformKind::CUDA>>();
      mixed_parameters_ = std::make_shared<PreparedDeviceMixedDenseParameters>(image_, *preparation_queue_, true);
    }
    else if (precision != PsiFormerPrecisionPolicy::FP64_REFERENCE)
      throw std::invalid_argument("Production GPU inference supports only FP64 or explicit FP32 dense/FP64 islands");
  }

  // Contexts retain references to the owner/mutex, so this backend is address-stable.
  PreparedDeviceBatchBackend(const PreparedDeviceBatchBackend&) = delete;
  PreparedDeviceBatchBackend& operator=(const PreparedDeviceBatchBackend&) = delete;
  PreparedDeviceBatchBackend(PreparedDeviceBatchBackend&&) = delete;
  PreparedDeviceBatchBackend& operator=(PreparedDeviceBatchBackend&&) = delete;

  /// Prepare all vendor and coordinate staging resources before tile evaluation.
  std::unique_ptr<pf::DirectBatchTileBackendWorkspace> prepare(std::size_t capacity) const override
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    return std::make_unique<Context>(this, image_, mixed_parameters_, owner_, capacity, nextIdentity());
  }

  /// Prepare one exclusive, fixed-profile lane sharing this backend's publication owner.
  std::unique_ptr<PreparedDeviceProductSession> prepareProductSession(
      ProductSessionProfile profile = ProductSessionProfile::SCORE_AND_KINETIC) const
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (mixed_parameters_)
      throw std::invalid_argument("Mixed PsiFormer precision does not support optimization or reverse products");
    return std::make_unique<PreparedDeviceProductSession>(image_, owner_, control_mutex_, nextIdentity(), profile);
  }

  /// Return typed device products while leaving host publication to the caller.
  pf::DirectBatchTileResultView evaluate(pf::DirectBatchTileBackendWorkspace& workspace,
      pf::DirectBatchMode mode, const double* coordinates, std::size_t size,
      std::size_t parameter_version) const override
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    auto* context = dynamic_cast<Context*>(&workspace);
    if (!context || context->owner != this || !coordinates || size == 0 ||
        size > context->capacity || parameter_version != image_->version())
      throw std::invalid_argument("Device batch tile has foreign resources, wrong capacity or stale parameters");
    const bool spatial = mode != pf::DirectBatchMode::VALUE_ONLY;
    const auto operations = acceleratorOperationBit(spatial ? PsiFormerAcceleratorOperation::VGL
                                                           : PsiFormerAcceleratorOperation::VALUES);
    const auto count = spatial_detail::checkedProduct(size,
        spatial_detail::checkedProduct(3, image_->plan().modelShape().electrons(), "Device tile coordinates overflow"),
        "Device tile coordinate extent overflow");
    context->positions.resize(count);
    std::copy_n(coordinates, count, context->positions.begin());
    if (context->coordinate_generation == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Device tile coordinate generation exhausted");
    const auto generation = ++context->coordinate_generation;

    // Lease transitions and submission use the existing non-atomic control plane.
    // This explicit correctness lock can be narrowed after asynchronous integration.
    if (context->hooks.modelVersion() != image_->version())
      context->hooks.rebindModel(image_, mixed_parameters_);
    const auto previous_mixed_retries = context->hooks.mixedRetries();
    context->executor.acquire(owner_, context->crowd, context->hooks);
    try
    {
      context->hooks.bindRequest(context->positions, operations, generation);
      const auto pending = context->executor.evaluate(owner_, context->crowd, operations, size, generation, context->hooks);
      context->executor.wait(context->crowd, pending.result_generation, context->hooks);
      (void)context->crowd.completedResult(pending.result_generation, generation);
      context->executor.release(owner_, context->crowd, context->hooks);
    }
    catch (...)
    {
      if (context->crowd.poisoned())
      {
        context->executor.recover(owner_, context->crowd, context->hooks);
        context->executor.prepare(context->crowd, owner_.activeProvenance().layout_fingerprint,
                                  context->capacity, context->hooks);
      }
      else if (context->crowd.acquired())
        context->executor.release(owner_, context->crowd, context->hooks);
      throw;
    }

    const auto& layout = context->hooks.resultLayout();
    pf::DirectBatchTileResultView result;
    result.size = size;
    result.parameter_version = parameter_version;
    result.sign = context->hooks.phases().data();
    result.logabs = context->hooks.jets().data();
    result.value_stride = layout.configuration_stride;
    if (spatial)
    {
      result.gradient = result.logabs + layout.uncheckedGradientOffset(0, 0, 0);
      result.gradient_configuration_stride = layout.configuration_stride;
      result.gradient_lane_stride = layout.plane_stride;
      result.lap_log = result.logabs + layout.uncheckedLaplacianOffset(0, 0, 0);
      result.lap_log_configuration_stride = layout.configuration_stride;
      result.lap_log_lane_stride = layout.plane_stride;
      result.lap_ratio = context->hooks.laplacianRatios().data();
      result.lap_ratio_configuration_stride = layout.electron_count;
    }
    if (numerical_tiles_.fetch_add(1, std::memory_order_relaxed) == 0 && first_dispatch_reporter_)
      first_dispatch_reporter_();
    if (mixed_parameters_ && (!mixed_reported_.exchange(true, std::memory_order_relaxed) ||
                             context->hooks.mixedRetries() != previous_mixed_retries))
      std::clog << "PsiFormer explicit mixed inference completed: mixed_batches=" << context->hooks.mixedBatches()
                << ", whole_fp64_retries=" << context->hooks.mixedRetries()
                << ", retry_reason=" << context->hooks.mixedRetryReason() << std::endl;
    return result;
  }

  /// Count completed GPU tiles across all crowds sharing this numerical backend.
  std::size_t numericalTileCount() const noexcept override
  {
    return numerical_tiles_.load(std::memory_order_relaxed);
  }

  /// Upload privately, then commit native values and active control provenance.
  /// The caller holds the model's exclusive lock throughout this transaction.
  void publishParameters(const std::vector<double>& values, std::size_t version,
                         const std::function<void()>& commit_native) const override
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!commit_native)
      throw std::invalid_argument("Device publication requires an authoritative native commit");
    auto candidate = image_->makeUpdatedImage(values, version);
    auto mixed_candidate = mixed_parameters_ ? std::make_shared<PreparedDeviceMixedDenseParameters>(
        candidate, *preparation_queue_, true) : std::shared_ptr<PreparedDeviceMixedDenseParameters>{};
    publication_hooks_.candidate = &candidate;
    publication_hooks_.mixed_candidate = &mixed_candidate;
    publication_hooks_.commit_native = &commit_native;
    try
    {
      owner_.stage(owner_.activeProvenance().layout_fingerprint, version, publication_hooks_);
      owner_.complete(version, publication_hooks_);
    }
    catch (...)
    {
      if (owner_.publicationPending())
        owner_.cancel(version, publication_hooks_);
      publication_hooks_.candidate = nullptr;
      publication_hooks_.mixed_candidate = nullptr;
      publication_hooks_.commit_native = nullptr;
      throw;
    }
    publication_hooks_.candidate = nullptr;
    publication_hooks_.mixed_candidate = nullptr;
    publication_hooks_.commit_native = nullptr;
  }

private:
  /** Keep the publication hook identity stable across stage/complete/cancel.
   * No native change occurs during stage. Completion invokes the prevalidated
   * native update before the noexcept shared-image swap and owner transition.
   */
  struct PublicationHooks final : PsiFormerAcceleratorHooks
  {
    /// Borrow the backend's active image for its entire address-stable lifetime.
    explicit PublicationHooks(std::shared_ptr<const PreparedDeviceModel>& active,
                              std::shared_ptr<const PreparedDeviceMixedDenseParameters>& mixed)
        : active(active), mixed(mixed) {}

    /// Accept only this backend's prepared candidate and publication event kinds.
    void submit(const PsiFormerAcceleratorEvent& event) override
    {
      if (!candidate || !*candidate || !mixed_candidate || !commit_native || event.model_version != (*candidate)->version())
        throw std::logic_error("Device publication event has no matching prepared candidate");
      switch (event.kind)
      {
      case PsiFormerAcceleratorEventKind::MODEL_STAGE:
        active->validateSuccessor(*candidate);
        return;
      case PsiFormerAcceleratorEventKind::MODEL_COMPLETE:
        (*commit_native)();
        active = *candidate;
        mixed = *mixed_candidate;
        return;
      case PsiFormerAcceleratorEventKind::MODEL_CANCEL:
        return;
      default:
        throw std::logic_error("Device publication hook received a crowd event");
      }
    }

    std::shared_ptr<const PreparedDeviceModel>& active;
    std::shared_ptr<const PreparedDeviceMixedDenseParameters>& mixed;
    const std::shared_ptr<PreparedDeviceMixedDenseParameters>* mixed_candidate = nullptr;
    const std::shared_ptr<const PreparedDeviceModel>* candidate = nullptr;
    const std::function<void()>* commit_native = nullptr;
  };

  /** Own one address-stable numerical crowd, with preallocated host coordinates. */
  struct Context final : pf::DirectBatchTileBackendWorkspace
  {
    /// Construct a released, prepared crowd using its backend's shared owner.
    Context(const PreparedDeviceBatchBackend* backend, std::shared_ptr<const PreparedDeviceModel> image,
            std::shared_ptr<const PreparedDeviceMixedDenseParameters> mixed_parameters,
            const PsiFormerAcceleratorSharedModel& model,
            std::size_t capacity, std::uint64_t identity)
        : owner(backend), capacity(capacity), executor(PreparedDeviceCrowdHooks::backend(),
            PreparedDeviceCrowdHooks::coverage()), crowd(identity), hooks(image, model, identity, std::move(mixed_parameters))
    {
      positions.reserve(spatial_detail::checkedProduct(capacity,
          spatial_detail::checkedProduct(3, image->plan().modelShape().electrons(), "Device tile coordinates overflow"),
          "Device tile staging overflow"));
      executor.prepare(crowd, model.activeProvenance().layout_fingerprint, capacity, hooks);
    }

    const PreparedDeviceBatchBackend* const owner;
    const std::size_t capacity;
    PsiFormerAcceleratorExecutor executor;
    PsiFormerAcceleratorCrowdState crowd;
    PreparedDeviceCrowdHooks hooks;
    std::vector<double> positions;
    std::size_t coordinate_generation = 0;
  };

  /// Reject absent images before model provenance construction.
  static std::shared_ptr<const PreparedDeviceModel> checkedImage(std::shared_ptr<const PreparedDeviceModel> image)
  {
    if (!image)
      throw std::invalid_argument("Device batch backend requires an immutable model image");
    return image;
  }

  /// Allocate distinct crowd routing identities only during preparation.
  std::uint64_t nextIdentity() const
  {
    if (next_identity_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("Device tile crowd identity exhausted");
    return ++next_identity_;
  }

  mutable std::shared_ptr<const PreparedDeviceModel> image_;
  mutable std::shared_ptr<const PreparedDeviceMixedDenseParameters> mixed_parameters_;
  std::unique_ptr<compute::Queue<PlatformKind::CUDA>> preparation_queue_;
  mutable PublicationHooks publication_hooks_{image_, mixed_parameters_};
  mutable std::mutex control_mutex_;
  mutable PsiFormerAcceleratorSharedModel owner_;
  mutable std::uint64_t next_identity_ = 0;
  const std::function<void()> first_dispatch_reporter_;
  mutable std::atomic<std::size_t> numerical_tiles_{0};
  mutable std::atomic<bool> mixed_reported_{false};
};

} // namespace qmcplusplus::psiformer::device

#endif
