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
#include "PsiFormerDeviceMetropolis.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerBatchExecutor.h"

#include <atomic>
#include <functional>
#include <iostream>
#include <mutex>
#include "QMCWaveFunctions/PsiFormer/PsiFormerInferenceConcurrency.h"

namespace qmcplusplus::psiformer::device
{

/** Supply exclusive GPU scratch per batch workspace and one shared model image.
 * The control mutex protects non-atomic model leases. The first synchronous
 * default serializes submissions. Explicit bounded FP64 inference leases allow
 * independent queues while publication excludes every in-flight evaluation.
 */
class PreparedDeviceBatchBackend final : public pf::DirectBatchTileBackend
{
public:
  /// Bind one immutable parameter version and its canonical layout fingerprint.
  explicit PreparedDeviceBatchBackend(std::shared_ptr<const PreparedDeviceModel> image,
                                      std::function<void()> first_dispatch_reporter = {},
                                      PsiFormerPrecisionPolicy precision = PsiFormerPrecisionPolicy::FP64_REFERENCE,
                                      DeviceExecutionPreparation preparation = {})
      : image_(checkedImage(std::move(image))),
        preparation_(validatedPreparation(image_->plan(), precision, preparation)),
        inference_concurrency_(preparation_.active_inference_crowds),
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
    // B=1 startup admission cannot authorize a later larger numerical owner.
    const auto preparation = resolveDeviceTargetSelection(image_->plan(), preparation_, capacity);
    if (preparation_.select_vgl_targets)
      reportDeviceTargetSelection(std::cout, "VGL_actual_owner", image_->plan(), capacity,
          preparation.vgl.max_targets, estimateTargetVglCost(image_->plan(), capacity, preparation.vgl.max_targets),
          preparation.vgl.device_workspace_budget_bytes, preparation.vgl.scalar_work_budget);
    return std::make_unique<Context>(this, image_, mixed_parameters_, owner_, capacity, nextIdentity(), preparation.vgl);
  }

  /// Prepare one exclusive, fixed-profile lane sharing this backend's publication owner.
  std::unique_ptr<PreparedDeviceProductSession> prepareProductSession(
      ProductSessionProfile profile = ProductSessionProfile::SCORE_AND_KINETIC,
      std::size_t score_capacity = 1) const
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (preparation_.active_inference_crowds != 1)
      throw std::invalid_argument("Concurrent GPU inference does not admit parameter-product or training sessions");
    if (mixed_parameters_)
      throw std::invalid_argument("Mixed PsiFormer precision does not support optimization or reverse products");
    return std::make_unique<PreparedDeviceProductSession>(
        image_, owner_, control_mutex_, nextIdentity(), profile, score_capacity, preparation_);
  }

  /// Return typed device products while leaving host publication to the caller.
  pf::DirectBatchTileResultView evaluate(pf::DirectBatchTileBackendWorkspace& workspace,
      pf::DirectBatchMode mode, const double* coordinates, std::size_t size,
      std::size_t parameter_version) const override
  {
    auto inference_lease = inference_concurrency_.acquire();
    std::unique_lock<std::mutex> lock(control_mutex_);
    auto* context = dynamic_cast<Context*>(&workspace);
    if (!context || context->owner != this || !coordinates || size == 0 ||
        size > context->capacity || parameter_version != image_->version())
      throw std::invalid_argument("Device batch tile has foreign resources, wrong capacity or stale parameters");
    std::unique_lock context_lock(context->evaluation_mutex, std::try_to_lock);
    if (!context_lock.owns_lock())
      throw std::logic_error("Device inference cannot re-enter the same native crowd resource");
    // Native DeviceManager binds OpenMP workers. Validate foreign callers too,
    // before any stream use; never silently select a different process device.
    image_->validateCurrentDevice();
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

    // Publication stays excluded by the shared inference permit. Only lease
    // mutations need the control mutex in explicit concurrent inference mode.
    if (context->hooks.modelVersion() != image_->version())
      context->hooks.rebindModel(image_, mixed_parameters_);
    const auto previous_mixed_retries = context->hooks.mixedRetries();
    context->executor.acquire(owner_, context->crowd, context->hooks);
    if (preparation_.active_inference_crowds > 1)
      lock.unlock();
    try
    {
      context->hooks.bindRequest(context->positions, operations, generation);
      const auto pending = context->executor.evaluate(owner_, context->crowd, operations, size, generation, context->hooks);
      context->executor.wait(context->crowd, pending.result_generation, context->hooks);
      (void)context->crowd.completedResult(pending.result_generation, generation);
      if (!lock.owns_lock())
        lock.lock();
      context->executor.release(owner_, context->crowd, context->hooks);
    }
    catch (...)
    {
      if (!lock.owns_lock())
        lock.lock();
      if (context->crowd.poisoned())
      {
        context->sampler.reset();
        context->executor.recover(owner_, context->crowd, context->hooks);
        context->executor.prepare(context->crowd, owner_.activeProvenance().layout_fingerprint,
                                  context->capacity, context->hooks);
      }
      else if (context->crowd.acquired())
        context->executor.release(owner_, context->crowd, context->hooks);
      throw;
    }
    if (preparation_.active_inference_crowds > 1)
      lock.unlock();

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

  /// Host permit evidence only, not a claim of simultaneous kernel execution.
  std::size_t peakAdmittedInferenceCalls() const { return inference_concurrency_.peakAdmittedCalls(); }

  /// Report the explicit per-crowd sampler admission limit; zero preserves host moves.
  std::size_t metropolisBudget() const noexcept { return preparation_.metropolis_workspace_budget_bytes; }

  /// An explicit additions cap is also the opt-in drift profile discriminator.
  std::size_t metropolisDriftBudget() const noexcept { return preparation_.metropolis_drift_budget_bytes; }

  /// Report the independent explicit per-optimizer/rank real Adam limit.
  std::size_t adamBudget() const noexcept { return preparation_.adam_workspace_budget_bytes; }

  /** Borrow the existing VALUE graph for native-engine whole-system substeps.
   * The caller holds its native model read transaction and acquired host crowd.
   * Draw callbacks must not reenter the backend. Returned coordinates are only
   * materialized after every substep resolves; failures abort, never fall back.
   */
  std::pair<std::size_t, std::size_t> advanceMetropolis(pf::DirectBatchTileBackendWorkspace& workspace,
      const std::vector<double>& positions, const std::vector<double>& inverse_mass, double tau,
      std::size_t substeps, std::size_t version,
      const std::function<void(std::vector<double>&)>& gaussian_draw,
      const std::function<double()>& uniform_draw, std::vector<double>& accepted_positions,
      const std::function<void(const std::vector<std::uint32_t>&)>& validation_mask_observer = {},
      bool use_drift = false, NativeDriftCallback drift = {}) const
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    auto* context = dynamic_cast<Context*>(&workspace);
    const auto ne = image_->plan().modelShape().electrons();
    if (!context || context->owner != this || !metropolisBudget() || mixed_parameters_ ||
        positions.empty() || positions.size() % (3 * ne) || inverse_mass.size() != ne ||
        !substeps || !gaussian_draw || !uniform_draw || version != image_->version() ||
        !device_math::isFiniteBinary64(tau) || tau <= 0)
      throw std::invalid_argument("Device sampling requires its explicit real/open FP64 crowd profile and current image");
    if (use_drift != bool(metropolisDriftBudget()) || (use_drift && !drift))
      throw std::invalid_argument("Device sampling usedrift must match its explicit additions cap and native policy");
    const auto cost = estimateMetropolisCost(positions.size() / (3 * ne), ne);
    if (cost.device_bytes > metropolisBudget())
      throw std::length_error("Device sampling exceeds its explicit standalone per-crowd budget before draws");
    const auto drift_cost = use_drift ? estimateMetropolisDriftCost(image_->plan(), context->capacity, cost.population)
                                    : CoordinateGradientCost{};
    if (drift_cost.device_workspace_bytes > metropolisDriftBudget())
      throw std::length_error("Device drift additions exceed per-crowd budget before VALUE upgrade and draws");
    (void)spatial_detail::checkedProduct(cost.population, substeps, "Device sampling counter overflow");
    for (const auto value : positions)
      if (!device_math::isFiniteBinary64(value))
        throw std::invalid_argument("Device sampling host coordinates are nonfinite");
    for (const auto value : inverse_mass)
      if (!device_math::isFiniteBinary64(value) || value <= 0)
        throw std::invalid_argument("Device sampling inverse mass must be finite and positive");
    if (accepted_positions.size() != positions.size())
      throw std::invalid_argument("Device sampling accepted export must be pre-sized");

    // Context replacement destroys its sampler first; rebind retains graph storage.
    if (context->hooks.modelVersion() != image_->version())
      context->hooks.rebindModel(image_);
    if (use_drift && !context->hooks.retain_values_)
    {
      context->sampler.reset(); // Revoke every borrower before deleting the ordinary VALUE graph.
      context->hooks.retainValueTape();
    }
    const bool reseed = !context->sampler || context->sampler_positions != positions ||
        context->sampler_masses != inverse_mass;
    context->sampler_gaussian.resize(positions.size());
    context->sampler_uniforms.resize(cost.population);
    std::size_t accepts = 0, uniform_draws = 0;
    try
    {
      context->executor.acquire(owner_, context->crowd, context->hooks);
      if (reseed)
      {
        // Release a stale owner before replacement: no transient double-owner peak.
        context->sampler.reset();
        context->sampler_positions = positions;
        context->sampler_masses = inverse_mass;
        context->sampler = use_drift
            ? PreparedDeviceMetropolis::prepareDrift(*context->hooks.values_, context->hooks.queue_,
                positions, inverse_mass, metropolisBudget(), metropolisDriftBudget(), drift)
            : PreparedDeviceMetropolis::prepare(*context->hooks.values_, context->hooks.queue_,
                positions, inverse_mass, metropolisBudget());
        context->sampler_version = version;
      }
      else if (context->sampler_version != version)
      {
        context->sampler->refreshAccepted(version);
        context->sampler_version = version;
      }

      // Preserve native Gaussian flattening and conditional walker-order uniforms.
      if (use_drift)
        context->sampler->bindNativeDriftPolicy(drift);
      for (std::size_t step = 0; step < substeps; ++step)
      {
        gaussian_draw(context->sampler_gaussian);
        const auto token = context->sampler->propose(context->sampler_gaussian, tau, version);
        const auto& eligible = context->sampler->eligibility(token);
        for (std::size_t walker = 0; walker < cost.population; ++walker)
          if (eligible[walker] & METROPOLIS_ELIGIBLE)
          {
            context->sampler_uniforms[walker] = uniform_draw();
            ++uniform_draws;
          }
        const auto& mask = context->sampler->resolve(token, context->sampler_uniforms);
        for (const auto value : mask)
          accepts += (value & METROPOLIS_ACCEPTED) != 0;
        if (validation_mask_observer)
          validation_mask_observer(mask);
      }
      context->sampler->exportAccepted(accepted_positions);
      std::copy(accepted_positions.begin(), accepted_positions.end(), context->sampler_positions.begin());
      context->executor.release(owner_, context->crowd, context->hooks);
    }
    catch (...)
    {
      // Earlier substeps/RNG draws are not rolled back; the caller must abort.
      context->sampler.reset();
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
    std::clog << "PsiFormer device sampling completed: B=" << cost.population << ", Ne=" << ne
              << ", usedrift=" << (use_drift ? "yes" : "no")
              << ", drift_additions_device_bytes=" << drift_cost.device_workspace_bytes
              << ", drift_additions_host_bytes=" << drift_cost.host_workspace_bytes
              << ", drift_accepted_host_G_bytes=" << (use_drift ? checkedDeviceBytes<double>(cost.coordinates) : 0)
              << ", drift_staged_host_G_bytes=" << (use_drift ? checkedDeviceBytes<double>(cost.coordinates) : 0)
              << ", drift_host_microtile_G_bytes=" << (use_drift ? checkedDeviceBytes<double>(context->capacity * 3 * ne) : 0)
              << ", drift_host_displacement_bytes=" << (use_drift ? checkedDeviceBytes<double>(cost.coordinates) : 0)
              << ", drift_host_Green_bytes=" << (use_drift ? checkedDeviceBytes<double>(3 * cost.population) : 0)
              << ", drift_additions_budget_bytes=" << metropolisDriftBudget()
              << ", drift_policy=" << (use_drift ? "native_host_callback_failclosed" : "none")
              << ", VALUE_capacity=" << context->capacity << ", substeps=" << substeps
              << ", parameter_version=" << version << ", per_crowd_budget_bytes=" << metropolisBudget()
              << ", sampler_device_bytes=" << cost.device_bytes << ", owner_host_mask_bytes=" << cost.host_bytes
              << ", host_scratch_bytes=" << sizeof(double) * (context->sampler_positions.capacity() +
                    context->sampler_gaussian.capacity() + context->sampler_uniforms.capacity() +
                    context->sampler_masses.capacity())
              << ", bridge_host_staging_bytes=" << sizeof(double) * (positions.capacity() +
                    inverse_mass.capacity() + accepted_positions.capacity())
              << ", borrowed_VALUE_graph_bytes=" << context->hooks.values_->deviceWorkspaceBytes()
              << ", uniform_draws=" << uniform_draws << ", accepts=" << accepts
              << ", host_VGL_boundary=outer_step, RNG=host_native, Hamiltonian=host, CPU_fallback=no\n";
    return {accepts, cost.population * substeps - accepts};
  }

  /// Count completed GPU tiles across all crowds sharing this numerical backend.
  std::size_t numericalTileCount() const noexcept override
  {
    return numerical_tiles_.load(std::memory_order_relaxed);
  }

  /// Retain the exact immutable active image for a cold download or successor preparation.
  std::shared_ptr<const PreparedDeviceModel> currentImage() const
  {
    std::lock_guard lock(control_mutex_);
    return image_;
  }

  /** Own final backend publication authority and both images. The component
   * publisher additionally owns its model lock and exact moment receipt.
   */
  class PreparedDevicePublication
  {
  public:
    /// Stage only prevalidated FP64 same-family images before readiness.
    PreparedDevicePublication(std::shared_ptr<const PreparedDeviceBatchBackend> owner,
        std::shared_ptr<const PreparedDeviceModel> source, std::shared_ptr<const PreparedDeviceModel> next)
        : owner_(std::move(owner)), source_(std::move(source)), next_(std::move(next)),
          publication_(owner_->inference_concurrency_.publicationBoundary()), control_(owner_->control_mutex_)
    {
      if (owner_->image_ != source_ || owner_->mixed_parameters_)
        throw std::invalid_argument("DEVICE authority needs the exact current FP64 image");
      source_->validateCurrentDevice();
      source_->validateSuccessor(next_);
      auto& hooks = owner_->publication_hooks_;
      hooks.candidate = &next_;
      hooks.mixed_candidate = &mixed_;
      hooks.commit_native = &unused_;
      try
      {
        owner_->owner_.stage(owner_->owner_.activeProvenance().layout_fingerprint, next_->version(), hooks);
        staged_ = true;
        completion_ = owner_->owner_.prepareCompletion(next_->version(), hooks);
      }
      catch (...)
      {
        if (staged_) owner_->owner_.cancel(next_->version(), hooks);
        clearHooks();
        throw;
      }
    }

    /// Cancel uncommitted control while locked; images retire only after both locks.
    ~PreparedDevicePublication()
    {
      if (staged_)
      {
        owner_->owner_.cancel(next_->version(), owner_->publication_hooks_);
        clearHooks();
      }
    }

    /// Adopt without hooks, runtime calls or last-owner retirement; authority remains held.
    void commit() noexcept
    {
      if (!staged_) std::terminate();
      owner_->image_ = next_;
      owner_->owner_.completePrepared(completion_);
      staged_ = false;
      clearHooks();
    }

    /// Release only after the component's version/moments also become current.
    void unlock() noexcept { control_.unlock(); publication_.unlock(); }

  private:
    /// Hook fields borrow this action only during staged ownership.
    void clearHooks() noexcept
    {
      owner_->publication_hooks_.candidate = nullptr;
      owner_->publication_hooks_.mixed_candidate = nullptr;
      owner_->publication_hooks_.commit_native = nullptr;
    }

    std::shared_ptr<const PreparedDeviceBatchBackend> owner_;
    std::shared_ptr<const PreparedDeviceModel> source_, next_;
    std::shared_ptr<PreparedDeviceMixedDenseParameters> mixed_;
    std::function<void()> unused_;
    PsiFormerAcceleratorSharedModel::PreparedCompletion completion_{};
    bool staged_ = false;
    std::unique_lock<std::shared_mutex> publication_;
    std::unique_lock<std::mutex> control_;
  };

  /// Upload privately, then commit native values and active control provenance.
  /// The caller holds the model's exclusive lock throughout this transaction.
  void publishParameters(const std::vector<double>& values, std::size_t version,
                         const std::function<void()>& commit_native) const override
  {
    auto publication_boundary = inference_concurrency_.publicationBoundary();
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
            std::size_t capacity, std::uint64_t identity, TargetVglPreparation target_preparation)
        : owner(backend), capacity(capacity), executor(PreparedDeviceCrowdHooks::backend(),
            PreparedDeviceCrowdHooks::coverage()), crowd(identity),
          hooks(image, model, identity, std::move(mixed_parameters), target_preparation)
    {
      positions.reserve(spatial_detail::checkedProduct(capacity,
          spatial_detail::checkedProduct(3, image->plan().modelShape().electrons(), "Device tile coordinates overflow"),
          "Device tile staging overflow"));
      executor.prepare(crowd, model.activeProvenance().layout_fingerprint, capacity, hooks);
    }

    const PreparedDeviceBatchBackend* const owner;
    const std::size_t capacity;
    std::mutex evaluation_mutex;
    PsiFormerAcceleratorExecutor executor;
    PsiFormerAcceleratorCrowdState crowd;
    PreparedDeviceCrowdHooks hooks;
    // Declared after hooks: sampler must die before its borrowed graph/queue.
    std::unique_ptr<PreparedDeviceMetropolis> sampler;
    std::vector<double> sampler_positions, sampler_gaussian, sampler_uniforms, sampler_masses;
    std::size_t sampler_version = 0;
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

  /// Reject mixed targets and incomplete/over-budget profiles before graph reserve.
  static DeviceExecutionPreparation validatedPreparation(const PsiFormerExecutionPlan& plan,
      PsiFormerPrecisionPolicy precision, DeviceExecutionPreparation preparation)
  {
    if (!preparation.active_inference_crowds ||
        preparation.active_inference_crowds == std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("Active GPU inference crowds must be positive and finite");
    if (preparation.active_inference_crowds > 1 &&
        (precision != PsiFormerPrecisionPolicy::FP64_REFERENCE ||
         preparation.metropolis_workspace_budget_bytes || preparation.adam_workspace_budget_bytes ||
         preparation.retained_coordinate_workspace_budget_bytes || preparation.kinetic.max_targets ||
         preparation.select_kinetic_targets))
      throw std::invalid_argument("Concurrent GPU inference requires FP64 and excludes sampling/training preparations");
    if (preparation.active_inference_crowds > 1 && !configuredDeviceMemoryAdmission().limit_bytes)
      throw std::invalid_argument("Concurrent GPU inference requires explicit native aggregate device admission");
    if (preparation.metropolis_drift_budget_bytes && (!preparation.metropolis_workspace_budget_bytes ||
        preparation.metropolis_drift_budget_bytes == std::numeric_limits<std::size_t>::max()))
      throw std::invalid_argument("Explicit device drift requires sampler preparation and a finite additions cap");
    if (preparation.adam_workspace_budget_bytes && precision != PsiFormerPrecisionPolicy::FP64_REFERENCE)
      throw std::invalid_argument("Explicit device Adam requires FP64 reference precision");
    if (preparation.metropolis_workspace_budget_bytes &&
        (precision != PsiFormerPrecisionPolicy::FP64_REFERENCE ||
         plan.environment().boundary != BoundaryCondition::OPEN))
      throw std::invalid_argument("Explicit device sampling requires real/open FP64 preparation");
    if (precision != PsiFormerPrecisionPolicy::FP64_REFERENCE &&
        (preparation.vgl.max_targets || preparation.kinetic.max_targets ||
         preparation.select_vgl_targets || preparation.select_kinetic_targets))
      throw std::invalid_argument("Mixed precision does not support explicit GPU target preparation");
    preparation = resolveDeviceTargetSelection(plan, preparation);
    validateDeviceExecutionPreparation(plan, preparation);
    return preparation;
  }

  /// Allocate distinct crowd routing identities only during preparation.
  std::uint64_t nextIdentity() const
  {
    if (next_identity_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("Device tile crowd identity exhausted");
    return ++next_identity_;
  }

  mutable std::shared_ptr<const PreparedDeviceModel> image_;
  const DeviceExecutionPreparation preparation_;
  mutable PsiFormerInferenceConcurrency inference_concurrency_;
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
