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
#include "PsiFormerDeviceECPObservation.h"
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

/// Freeze the bounded released-context FULL preparation opt-in before allocation.
inline bool configuredLazyVglPreparation()
{
  const char* value = std::getenv("PSIFORMER_GPU_LAZY_VGL_PREPARATION");
  if (!value || std::string_view(value) == "0")
    return false;
  if (std::string_view(value) == "1")
    return true;
  throw std::invalid_argument("PSIFORMER_GPU_LAZY_VGL_PREPARATION requires 0 or 1");
}

/// Capture the experimental large-score policy once at backend construction.
inline bool configuredLargeScoreTile4()
{
  const char* value = std::getenv("PSIFORMER_GPU_LARGE_SCORE_TILE4");
  if (!value || std::string_view(value) == "0")
    return false;
  if (std::string_view(value) == "1")
    return true;
  throw std::invalid_argument("PSIFORMER_GPU_LARGE_SCORE_TILE4 requires 0 or 1");
}

/// Keep lazy VALUE ownership on the original OPEN or fixed legacy real torus lane.
inline bool supportsLazyRealInferenceEnvironment(const ExecutionEnvironment& environment)
{
  if (environment.amplitude_scalar_domain != ScalarDomain::REAL ||
      environment.parameter_scalar_domain != ScalarDomain::REAL ||
      environment.compute_scalar_domain != ScalarDomain::REAL)
    return false;
  if (environment.boundary == BoundaryCondition::OPEN)
    return true;
  return environment.boundary == BoundaryCondition::PERIODIC && environment.fixed_nuclei &&
      environment.geometry_feature_policy == GeometryFeaturePolicy::PERIODIC_TORUS_V1 &&
      environment.head_representation == HeadRepresentation::REAL && !environment.periodic_profile &&
      environment.periodic_axes[0] && environment.periodic_axes[1] && environment.periodic_axes[2];
}

/// Recognize only DeviceBuffer's labelled malloc OOM for optional preparation.
/// Other device, stream and context errors must keep their original failure.
inline bool recoverOptionalEnvelopeAllocation(const std::runtime_error& error)
{
  const std::string_view message(error.what());
  if ((message.substr(0, 11) != "cudaMalloc:" && message.substr(0, 10) != "hipMalloc:") ||
      message.find("out of memory") == std::string_view::npos)
    return false;
#ifdef QMC_CUDA2HIP
  const auto last = hipGetLastError();
  if (last != hipSuccess && last != hipErrorOutOfMemory)
    checkRuntime(last, "Sparse VALUE allocation recovery");
#else
  const auto last = cudaGetLastError();
  if (last != cudaSuccess && last != cudaErrorMemoryAllocation)
    checkRuntime(last, "Sparse VALUE allocation recovery");
#endif
  return true;
}

/** Supply exclusive GPU scratch per batch workspace and one shared model image.
 * The control mutex protects non-atomic model leases. The first synchronous
 * default serializes submissions. Explicit bounded FP64 inference leases allow
 * independent queues while publication excludes every in-flight evaluation.
 */
class PreparedDeviceBatchBackend final : public pf::DirectBatchTileBackend
{
  friend struct RealEnvelopeValidationAccess;
  /// Shared cold receipt survives either backend/context destruction order.
  struct MixedFinalStatistics
  {
    ~MixedFinalStatistics() noexcept
    {
      try
      {
        std::clog << "PsiFormer explicit mixed inference summary: scope=all_destroyed_contexts"
                  << ", contexts=" << contexts.load(std::memory_order_relaxed)
                  << ", mixed_batches=" << successes.load(std::memory_order_relaxed)
                  << ", canonical_fp64_value_batches=" << canonical_values.load(std::memory_order_relaxed)
                  << ", whole_fp64_retries=" << retries.load(std::memory_order_relaxed) << std::endl;
      }
      catch (...) {} // Diagnostics must not throw during resource destruction.
    }
    std::atomic<std::size_t> contexts{0}, successes{0}, retries{0}, canonical_values{0};
  };

public:
  /// Bind one immutable parameter version and its canonical layout fingerprint.
  explicit PreparedDeviceBatchBackend(std::shared_ptr<const PreparedDeviceModel> image,
                                      std::function<void()> first_dispatch_reporter = {},
                                      PsiFormerPrecisionPolicy precision = PsiFormerPrecisionPolicy::FP64_REFERENCE,
                                      DeviceExecutionPreparation preparation = {},
                                      bool lazy_spatial = configuredLazyVglPreparation(),
                                      bool outer_twist = false)
      : image_(checkedImage(std::move(image))),
        preparation_(validatedPreparation(image_->plan(), precision, preparation)), lazy_spatial_(lazy_spatial),
        large_score_requested_(configuredLargeScoreTile4()),
        large_score_eligible_(precision == PsiFormerPrecisionPolicy::FP64_REFERENCE && !outer_twist &&
            !lazy_spatial && !preparation_.select_vgl_targets && !preparation_.select_kinetic_targets &&
            !preparation_.metropolis_workspace_budget_bytes && !preparation_.metropolis_drift_budget_bytes &&
            !preparation_.retained_coordinate_workspace_budget_bytes && !preparation_.adam_workspace_budget_bytes &&
            preparation_.active_inference_crowds == 1 &&
            image_->plan().environment().boundary == BoundaryCondition::PERIODIC &&
            supportsLazyRealInferenceEnvironment(image_->plan().environment()) &&
            image_->plan().modelShape().electrons() >= 64 && image_->plan().modelShape().electrons() <= 300 &&
            image_->plan().modelShape().nuclei <= 300 && image_->plan().modelShape().determinants <= 16 &&
            image_->plan().modelShape().feature_dimension <= 256 &&
            image_->plan().modelShape().attention_heads <= 4 && image_->plan().modelShape().attention_blocks <= 4),
        pair_geometry_requested_(configuredPeriodicValuePairGeometry()),
        pair_geometry_eligible_(precision == PsiFormerPrecisionPolicy::FP64_REFERENCE && !outer_twist &&
            image_->plan().modelShape().nuclei != 0 &&
            !preparation_.select_vgl_targets && !preparation_.select_kinetic_targets &&
            !preparation_.metropolis_workspace_budget_bytes && !preparation_.metropolis_drift_budget_bytes &&
            preparation_.active_inference_crowds == 1 && image_->plan().modelShape().electrons() >= 64 &&
            image_->plan().environment().boundary == BoundaryCondition::PERIODIC &&
            supportsLazyRealInferenceEnvironment(image_->plan().environment())),
        sparse_envelope_requested_(configuredRealVirtualEnvelopeReuse()),
        sparse_envelope_eligible_(precision == PsiFormerPrecisionPolicy::FP64_REFERENCE && !outer_twist &&
            !preparation_.select_vgl_targets &&
            !preparation_.metropolis_workspace_budget_bytes && !preparation_.metropolis_drift_budget_bytes &&
            preparation_.active_inference_crowds == 1 && image_->plan().modelShape().electrons() >= 64 &&
            image_->plan().environment().boundary == BoundaryCondition::PERIODIC &&
            supportsLazyRealInferenceEnvironment(image_->plan().environment())),
        inference_concurrency_(preparation_.active_inference_crowds),
        owner_(PreparedDeviceCrowdHooks::backend(),
               makePsiFormerDeviceLayout(image_->plan()).fingerprint, image_->version()),
        first_dispatch_reporter_(std::move(first_dispatch_reporter))
  {
    const auto& environment = image_->plan().environment();
    if (lazy_spatial_ && (precision != PsiFormerPrecisionPolicy::FP64_REFERENCE ||
        preparation_.active_inference_crowds != 1 || preparation_.metropolis_workspace_budget_bytes ||
        preparation_.metropolis_drift_budget_bytes || outer_twist || !supportsLazyRealInferenceEnvironment(environment)))
      throw std::invalid_argument("Lazy VGL preparation requires single-crowd real OPEN or fixed legacy Gamma FP64 without device sampling");
    if (precision == PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS)
    {
      preparation_queue_ = std::make_unique<compute::Queue<PlatformKind::CUDA>>();
      mixed_parameters_ = std::make_shared<PreparedDeviceMixedDenseParameters>(image_, *preparation_queue_, true);
      mixed_statistics_ = std::make_shared<MixedFinalStatistics>();
    }
    else if (precision != PsiFormerPrecisionPolicy::FP64_REFERENCE)
      throw std::invalid_argument("Production GPU inference supports only FP64 or explicit FP32 dense/FP64 islands");
    if (sparse_envelope_requested_ && !sparse_envelope_eligible_)
      std::clog << "PsiFormer REAL virtual envelope preparation: effective=0 reason=profile_or_automatic_target_selection\n";
    if (pair_geometry_requested_ && !pair_geometry_eligible_)
      std::clog << "PsiFormer REAL VALUE pair geometry: effective=0 reason=ineligible_backend_profile\n";
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
    validateDeviceExecutionPreparation(image_->plan(), preparation, capacity);
    if (!lazy_spatial_)
      reportSpatialPreparation(preparation.vgl, capacity);
    return std::make_unique<Context>(this, image_, mixed_parameters_, owner_, capacity, nextIdentity(),
                                     preparation.vgl, lazy_spatial_);
  }

  /// Report the immutable prepared limit, independent of later environment changes.
  std::size_t activeInferenceCrowdCapacity() const noexcept
  { return preparation_.active_inference_crowds; }

  /// Report the immutable preparation policy, never a later environment value.
  bool lazySpatialPreparation() const noexcept
  { return lazy_spatial_; }

  /// Cold driver topology proof, not a maximum-concurrent-lease assumption.
  void noteLargeScorePreparationContext(bool unplanned_single_crowd) const
  {
    if (!large_score_requested_)
      return;
    std::lock_guard<std::mutex> lock(control_mutex_);
    large_score_single_crowd_ = unplanned_single_crowd;
  }

  /// Reuse existing crowd scratch only under the same cold score/C1 proof.
  /// This does not admit score capacity4 or relax its exact live-owner guard.
  bool permitsScalarEcpCrowdScratchBorrow() const
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    return large_score_requested_ && large_score_eligible_ && large_score_single_crowd_ &&
        preparation_.active_inference_crowds == 1 && !sparse_envelope_requested_;
  }

  /// Observe actual eager context ownership without authorizing or allocating a lane.
  std::size_t liveEagerContextsForTesting() const noexcept
  { return live_eager_contexts_.load(std::memory_order_acquire); }

  /// Prepare one exclusive, fixed-profile lane sharing this backend's publication owner.
  std::unique_ptr<PreparedDeviceProductSession> prepareProductSession(
      ProductSessionProfile profile = ProductSessionProfile::SCORE_AND_KINETIC,
      std::size_t score_capacity = 1, bool force_fp64 = false) const
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    const bool independent_score_lane = preparation_.active_inference_crowds == 2 &&
        profile == ProductSessionProfile::SCORE_ONLY;
    if (preparation_.active_inference_crowds != 1 && !independent_score_lane)
      throw std::invalid_argument("Concurrent GPU parameter products require two score-only lanes");
    if (mixed_parameters_ && (profile != ProductSessionProfile::SCORE_ONLY ||
        preparation_.active_inference_crowds != 1))
      throw std::invalid_argument("Mixed PsiFormer products require one score-only crowd");
    if (force_fp64 && !mixed_parameters_)
      throw std::invalid_argument("Explicit FP64 product replay requires a mixed policy owner");
    // The eager ordinary context is already live. Reserve the unchanged B1
    // kinetic root, future C3 sums and successor image before optional B growth.
    // The target-cost full-Ne bound may conservatively exceed the old FULL
    // owner's exact storage; it is never multiplied by score capacity.
    if (large_score_requested_ && large_score_eligible_ && large_score_single_crowd_ &&
        live_eager_contexts_.load(std::memory_order_acquire) == 1 &&
        score_capacity > 1 && configuredDeviceMemoryAdmission().limit_bytes)
    {
      if (score_capacity > 4)
        throw std::invalid_argument("Large REAL score capacity must remain within1..4");
      const auto& plan = image_->plan();
      const auto selected = estimateScoreTileStorage(plan, score_capacity);
      const auto baseline = estimateScoreTileStorage(plan, 1);
      const auto kinetic = profile == ProductSessionProfile::SCORE_AND_KINETIC
          ? estimateTargetKineticCost(plan,
              preparation_.kinetic.max_targets ? preparation_.kinetic.max_targets : plan.modelShape().electrons(),
              preparation_.kinetic.lifted_channel_slots,
              preparation_.kinetic.parallel_trace_workspace_budget_bytes).device_workspace_bytes : 0;
      auto reserve = spatial_detail::checkedAdd(kinetic,
          PreparedDeviceScoreContraction::requiredBytes(plan.parameterCount(), 3), "Score baseline reserve overflow");
      reserve = spatial_detail::checkedAdd(reserve, image_->deviceStorageBytes(), "Score successor reserve overflow");
      reserve = spatial_detail::checkedAdd(reserve, selected.device_bytes, "Score candidate reserve overflow");
      int ordinal = -1;
#ifdef QMC_CUDA2HIP
      checkRuntime(hipGetDevice(&ordinal), "Large score current device");
#else
      checkRuntime(cudaGetDevice(&ordinal), "Large score current device");
#endif
      try
      {
        { auto preflight = deviceMemoryMetrics(ordinal).reserveAllocation(reserve); }
        std::clog << "PsiFormer large REAL score before allocation: requested=" << score_capacity
                  << " DEVICE_score_graph_delta=" << selected.device_bytes - baseline.device_bytes
                  << " HOST_score_graph_numeric_delta=" << selected.host_numeric_bytes - baseline.host_numeric_bytes
                  << " HOST_scope=requested_numeric_not_aggregate_RSS\n";
        auto candidate = std::make_unique<PreparedDeviceProductSession>(
            image_, owner_, control_mutex_, nextIdentity(), profile, score_capacity, preparation_, true);
        std::clog << "PsiFormer large REAL score preparation: selected=" << score_capacity
                  << " DEVICE_score_graph_bytes=" << selected.device_bytes
                  << " DEVICE_score_graph_delta=" << selected.device_bytes - baseline.device_bytes
                  << " unchanged_baseline_overlap_bound=" << reserve - selected.device_bytes
                  << " HOST_score_graph_numeric_delta=" << selected.host_numeric_bytes - baseline.host_numeric_bytes
                  << " HOST_actual_session_numeric_bytes=" << candidate->hostWorkspaceBytes()
                  << " HOST_session_stage_numeric_delta=" << checkedDeviceBytes<double>(
                      spatial_detail::checkedProduct(score_capacity - 1,
                          3 * plan.modelShape().electrons() + 2, "Score HOST staging overflow"))
                  << " HOST_ECP_stage_if_used_bytes=" << checkedDeviceBytes<double>(
                      spatial_detail::checkedProduct(score_capacity,
                          3 * plan.modelShape().electrons() + 1, "Score HOST ECP staging overflow"))
                  << " HOST_scope=numeric_not_aggregate_fit kinetic_owners="
                  << (profile == ProductSessionProfile::SCORE_AND_KINETIC ? 1 : 0) << '\n';
        return candidate;
      }
      catch (const std::length_error&)
      {
        std::clog << "PsiFormer large REAL score preparation: selected=1 reason=aggregate_fit\n";
      }
      catch (const std::bad_alloc&)
      {
        std::clog << "PsiFormer large REAL score preparation: selected=1 reason=optional_HOST_allocation\n";
      }
      catch (const std::runtime_error& error)
      {
        if (!recoverOptionalEnvelopeAllocation(error))
          throw;
        std::clog << "PsiFormer large REAL score preparation: selected=1 reason=optional_DEVICE_allocation\n";
      }
    }
    else if (large_score_requested_ && score_capacity > 1)
      std::clog << "PsiFormer large REAL score preparation: selected=1 reason=profile_topology_or_unbounded_admission\n";
    return std::make_unique<PreparedDeviceProductSession>(
        image_, owner_, control_mutex_, nextIdentity(), profile, score_capacity, preparation_, false,
        independent_score_lane, mixed_parameters_ && !force_fp64 ? &mixed_parameters_ : nullptr);
  }

  /// Immutable mixed parameter authority; the public inference label stays unchanged.
  bool usesMixedAffineProducts() const noexcept { return bool(mixed_statistics_); }

  /// Prepare both fixed-policy training owners under one cache/image admission boundary.
  std::pair<std::unique_ptr<PreparedDeviceProductSession>, std::unique_ptr<PreparedDeviceProductSession>>
  prepareMixedTrainingProductSessions(std::size_t requested_capacity = 1) const
  {
    std::lock_guard lock(control_mutex_);
    if (!mixed_parameters_ || preparation_.active_inference_crowds != 1)
      throw std::invalid_argument("Mixed real training requires one score-only crowd");
    const auto capacity = PreparedDeviceProductSession::admittedScoreCapacity(image_->plan().modelShape(), requested_capacity);
    const auto mixed = estimateMixedScoreTileStorage(image_->plan(), capacity);
    const auto reference = estimateScoreTileStorage(image_->plan(), capacity);
    const auto bytes = spatial_detail::checkedAdd(mixed.device_bytes, reference.device_bytes,
        "Mixed and replay product inventory overflow");
    int ordinal = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&ordinal), "Mixed product admission device");
#else
    checkRuntime(cudaGetDevice(&ordinal), "Mixed product admission device");
#endif
    { auto admission = deviceMemoryMetrics(ordinal).reserveAllocation(bytes); }
    std::clog << "PsiFormer mixed score before allocation: capacity=" << capacity
              << " mixed_DEVICE_graph_bytes=" << mixed.device_bytes
              << " FP64_replay_DEVICE_graph_bytes=" << reference.device_bytes
              << "; shared_master_and_compute_cache_excluded=yes\n";
    auto approximate = std::make_unique<PreparedDeviceProductSession>(image_, owner_, control_mutex_, nextIdentity(),
        ProductSessionProfile::SCORE_ONLY, capacity, preparation_, false, false, &mixed_parameters_);
    auto fallback = std::make_unique<PreparedDeviceProductSession>(image_, owner_, control_mutex_, nextIdentity(),
        ProductSessionProfile::SCORE_ONLY, capacity, preparation_);
    return {std::move(approximate), std::move(fallback)};
  }

  /// Read the current native epoch policy without selecting a different parameter image.
  PsiFormerGpuTrainingEpochMode trainingEpochMode() const noexcept
  { return training_epoch_mode_.load(std::memory_order_acquire); }

  /// Fence escaped derivative operators when either epoch boundary changes selection.
  std::size_t trainingEpochGeneration() const noexcept
  { return training_epoch_generation_.load(std::memory_order_acquire); }

  /// Caller owns native ModelWrite; transition only released C1 numerical resources.
  std::size_t enterTrainingEpoch(std::size_t version, PsiFormerGpuTrainingEpochMode mode) const
  {
    using Mode = PsiFormerGpuTrainingEpochMode;
    if (mode != Mode::STRICT_MIXED && mode != Mode::FP64_REPLAY)
      throw std::invalid_argument("Mixed real training requires strict mixed or FP64 replay mode");
    auto boundary = inference_concurrency_.publicationBoundary();
    std::lock_guard lock(control_mutex_);
    if (!mixed_parameters_ || preparation_.active_inference_crowds != 1 || version != image_->version() ||
        owner_.publicationPending() || trainingEpochMode() != Mode::INFERENCE)
      throw std::logic_error("Mixed real epoch requires an idle exact-version C1 backend");
    if (trainingEpochGeneration() > std::numeric_limits<std::size_t>::max() - 2)
      throw std::overflow_error("Mixed real arithmetic epoch generation exhausted");
    training_epoch_mode_.store(mode, std::memory_order_release);
    return training_epoch_generation_.fetch_add(1, std::memory_order_acq_rel) + 1;
  }

  /// Caller retains native ModelWrite and ends the epoch before optimizer publication.
  void leaveTrainingEpoch(std::size_t generation) const
  {
    auto boundary = inference_concurrency_.publicationBoundary();
    std::lock_guard lock(control_mutex_);
    if (generation != trainingEpochGeneration() || trainingEpochMode() == PsiFormerGpuTrainingEpochMode::INFERENCE)
      throw std::logic_error("Mixed real arithmetic epoch owner is stale");
    training_epoch_mode_.store(PsiFormerGpuTrainingEpochMode::INFERENCE, std::memory_order_release);
    training_epoch_generation_.fetch_add(1, std::memory_order_acq_rel);
  }

  /// Return typed device products while leaving host publication to the caller.
  pf::DirectBatchTileResultView evaluate(pf::DirectBatchTileBackendWorkspace& workspace,
      pf::DirectBatchMode mode, const double* coordinates, std::size_t size,
      std::size_t parameter_version) const override
  {
    return evaluateTile(workspace, mode, coordinates, size, parameter_version, nullptr);
  }

  /// Freeze hint storage only for the bounded ordinary legacy Gamma FP64 route.
  bool sparseValueHintsRequested() const noexcept override
  {
    return sparse_envelope_requested_ && sparse_envelope_eligible_;
  }

  /// Request only the narrow cold pair policy, independently of anchor reuse.
  bool periodicValuePairsRequested() const noexcept
  {
    return pair_geometry_requested_ && pair_geometry_eligible_;
  }

  /// Preserve the original numerical tile method when no scoped hints are present.
  pf::DirectBatchTileResultView evaluateSparseValueTile(pf::DirectBatchTileBackendWorkspace& workspace,
      const double* coordinates, std::size_t size, std::size_t parameter_version,
      const pf::DirectSparseValueTileHint& hint) const override
  {
    return evaluateTile(workspace, pf::DirectBatchMode::VALUE_ONLY, coordinates, size, parameter_version, &hint);
  }

private:
  pf::DirectBatchTileResultView evaluateTile(pf::DirectBatchTileBackendWorkspace& workspace,
      pf::DirectBatchMode mode, const double* coordinates, std::size_t size,
      std::size_t parameter_version, const pf::DirectSparseValueTileHint* hint) const
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
    context->hooks.setTrainingEpochMode(trainingEpochMode());
    // Recovery/capacity replacement must not silently drop a frozen admitted bank.
    if (context->pair_preparation_effective &&
        (!context->hooks.values_ || !context->hooks.values_->hasPeriodicValuePairs()))
      throw std::logic_error("VALUE pair context replacement requires fresh cold admission");
    // Native DeviceManager binds OpenMP workers. Validate foreign callers too,
    // before any stream use; never silently select a different process device.
    image_->validateCurrentDevice();
    const bool spatial = mode != pf::DirectBatchMode::VALUE_ONLY;
    const auto operations = acceleratorOperationBit(spatial ? PsiFormerAcceleratorOperation::VGL
                                                           : PsiFormerAcceleratorOperation::VALUES);
    // Publication and same-context exclusion remain held throughout cold preparation.
    if (context->hooks.modelVersion() != image_->version())
      context->hooks.rebindModel(image_, mixed_parameters_);
    if (spatial && context->hooks.prepareSpatialGraph())
      reportSpatialPreparation(context->hooks.target_preparation_, context->capacity);
    // The existing context lock excludes reuse/reentry. Never retain borrowed
    // HOST spans or cookie pointers beyond this synchronous tile submission.
    auto* value_graph = context->hooks.values_.get();
    struct SparseTileScope
    {
      PreparedDeviceInference* graph;
      bool completed = false;
      bool last = true;
      void invalidate() noexcept
      {
        if (graph)
          graph->endSparseEnvelopeTile(false, true);
        graph = nullptr;
      }
      ~SparseTileScope() { if (graph) graph->endSparseEnvelopeTile(completed, last); }
    } sparse_scope{value_graph};
    const bool valid_hint = value_graph && !spatial && hint && hint->request_owner && hint->live_cookie &&
        hint->request_generation && hint->live_cookie->load(std::memory_order_acquire) == hint->request_generation;
    if (valid_hint)
    {
      value_graph->beginSparseEnvelopeTile(hint->request_owner, hint->request_generation,
          hint->reference_count, hint->logical_begin, hint->references, hint->changed_electrons, size);
      sparse_scope.last = hint->last_tile;
    }
    else if (value_graph)
      value_graph->endSparseEnvelopeTile(false, true);
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
      // Recovery may replace the graph. Revoke its anchor before retiring it.
      sparse_scope.invalidate();
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
    if (spatial && context->observation_capture)
      context->observation_capture->append(context->hooks.spatial_->completedSpatialFinals(),
          context->hooks.queue_, context->positions, size, generation);
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
    // Distinguish a completed batched request from capacity-only preparation, once per exclusive Context.
    if (preparation_.vgl.parallel_trace_workspace_budget_bytes && spatial && size > 1 && !context->parallel_batch_reported)
    {
      std::clog << "PsiFormer parallel VGL tile completed: configurations=" << size
                << ", capacity=" << context->capacity << std::endl;
      context->parallel_batch_reported = true;
    }
    if (mixed_parameters_ && (!mixed_reported_.exchange(true, std::memory_order_relaxed) ||
                             context->hooks.mixedRetries() != previous_mixed_retries))
      std::clog << "PsiFormer explicit mixed inference completed: mixed_batches=" << context->hooks.mixedBatches()
                << ", canonical_fp64_value_batches=" << context->hooks.canonicalValueBatches()
                << ", whole_fp64_retries=" << context->hooks.mixedRetries()
                << ", retry_reason=" << context->hooks.mixedRetryReason() << std::endl;
    if (valid_hint && hint->live_cookie->load(std::memory_order_acquire) != hint->request_generation)
      throw std::logic_error("Sparse VALUE request cookie changed before tile publication");
    sparse_scope.completed = true;
    return result;
  }

public:

  /** Cold pre-RNG admission for the ordinary and optional dedicated VALUE16 owner.
   * A lazy ordinary FULL uses the existing conservative target-graph bound.
   * All actual allocations still use the existing aggregate DEVICE pool.
   */
  bool prepareSparseValueReuse(pf::DirectBatchTileBackendWorkspace& ordinary_workspace,
      pf::DirectBatchTileBackendWorkspace* value16_workspace = nullptr) const
  {
    if (!sparseValueHintsRequested())
      return false;
    std::lock_guard lock(control_mutex_);
    auto* ordinary = dynamic_cast<Context*>(&ordinary_workspace);
    auto* value16 = dynamic_cast<Context*>(value16_workspace);
    if (!ordinary || ordinary->owner != this ||
        (value16_workspace && (!value16 || value16->owner != this || value16 == ordinary)))
      throw std::invalid_argument("Sparse VALUE preparation requires two distinct bound contexts");
    std::unique_lock first_lock(ordinary->evaluation_mutex, std::try_to_lock);
    std::unique_lock<std::mutex> second_lock;
    if (value16)
      second_lock = std::unique_lock(value16->evaluation_mutex, std::try_to_lock);
    if (!first_lock.owns_lock() || (value16 && !second_lock.owns_lock()))
      throw std::logic_error("Sparse VALUE preparation requires released contexts");
    if (ordinary->sparse_preparation_checked)
      return ordinary->sparse_preparation_effective;
    ordinary->sparse_preparation_checked = true;
    auto& first = ordinary->hooks.values_;
    auto* second = value16 ? value16->hooks.values_.get() : nullptr;
    if (!first || !first->admitsSparseEnvelopeAnchor() ||
        (value16 && (!second || !second->admitsSparseEnvelopeAnchor())) ||
        !configuredDeviceMemoryAdmission().limit_bytes)
    {
      std::clog << "PsiFormer REAL virtual envelope preparation: effective=0 reason=unbounded_or_ineligible_baseline\n";
      return false;
    }
    const auto& plan = image_->plan();
    const auto future_full = ordinary->hooks.spatial_ ? 0 :
        (ordinary->hooks.target_preparation_.max_targets
          ? estimatePreparedTargetVglCost(plan, ordinary->capacity, ordinary->hooks.target_preparation_).device_workspace_bytes
          : estimateTargetVglCost(plan, ordinary->capacity, plan.modelShape().electrons()).device_workspace_bytes);
    const auto extra = spatial_detail::checkedAdd(first->sparseEnvelopeDeviceBytes(),
        second ? second->sparseEnvelopeDeviceBytes() : 0, "Sparse VALUE two-owner bytes overflow");
    int ordinal = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&ordinal), "Sparse VALUE preparation current device");
#else
    checkRuntime(cudaGetDevice(&ordinal), "Sparse VALUE preparation current device");
#endif
    auto rollback = [&] {
      first->discardSparseEnvelopeAnchor();
      if (second)
        second->discardSparseEnvelopeAnchor();
    };
    try
    {
      auto future_reservation = deviceMemoryMetrics(ordinal).reserveAllocation(future_full);
      { auto optional_preflight = deviceMemoryMetrics(ordinal).reserveAllocation(extra); }
      first->prepareSparseEnvelopeAnchor();
      if (second)
        second->prepareSparseEnvelopeAnchor();
    }
    catch (const std::length_error&)
    {
      rollback();
      std::clog << "PsiFormer REAL virtual envelope preparation: effective=0 reason=aggregate_fit requested_device_bytes="
                << extra << " future_FULL_numeric_bound=" << future_full << '\n';
      return false;
    }
    catch (const std::bad_alloc&)
    {
      rollback();
      std::clog << "PsiFormer REAL virtual envelope preparation: effective=0 reason=optional_HOST_allocation\n";
      return false;
    }
    catch (const std::runtime_error& error)
    {
      rollback();
      if (!recoverOptionalEnvelopeAllocation(error))
        throw;
      std::clog << "PsiFormer REAL virtual envelope preparation: effective=0 reason=optional_DEVICE_allocation\n";
      return false;
    }
    catch (...)
    {
      rollback();
      throw;
    }
    ordinary->sparse_preparation_effective = true;
    if (value16)
      value16->sparse_preparation_checked = value16->sparse_preparation_effective = true;
    std::clog << "PsiFormer REAL virtual envelope preparation: effective=1 participants=" << (value16 ? 2 : 1)
              << " requested_device_bytes=" << extra << " future_FULL_numeric_bound=" << future_full
              << " HOST_scope=numeric_not_aggregate_fit\n";
    return true;
  }

  /// Admit all simultaneous VALUE banks while preserving existing anchor state.
  bool preparePeriodicValuePairs(pf::DirectBatchTileBackendWorkspace& ordinary_workspace,
      pf::DirectBatchTileBackendWorkspace* value16_workspace = nullptr) const
  {
    if (!periodicValuePairsRequested())
      return false;
    std::lock_guard lock(control_mutex_);
    auto* ordinary = dynamic_cast<Context*>(&ordinary_workspace);
    auto* value16 = dynamic_cast<Context*>(value16_workspace);
    if (!ordinary || ordinary->owner != this ||
        (value16_workspace && (!value16 || value16->owner != this || value16 == ordinary)))
      throw std::invalid_argument("VALUE pair preparation requires distinct bound contexts");
    std::unique_lock first_lock(ordinary->evaluation_mutex, std::try_to_lock);
    std::unique_lock<std::mutex> second_lock;
    if (value16)
      second_lock = std::unique_lock(value16->evaluation_mutex, std::try_to_lock);
    if (!first_lock.owns_lock() || (value16 && !second_lock.owns_lock()))
      throw std::logic_error("VALUE pair preparation requires released contexts");
    auto* first = ordinary->hooks.values_.get();
    auto* second = value16 ? value16->hooks.values_.get() : nullptr;
    if (ordinary->pair_preparation_checked)
    {
      if (ordinary->pair_preparation_effective &&
          (!first || !first->hasPeriodicValuePairs() ||
           (value16 && (!second || !second->hasPeriodicValuePairs()))))
        throw std::logic_error("VALUE pair context replacement requires fresh cold admission");
      return ordinary->pair_preparation_effective;
    }
    ordinary->pair_preparation_checked = true;
    if (!first || !first->admitsSparseEnvelopeAnchor() ||
        (value16 && (!second || !second->admitsSparseEnvelopeAnchor())) ||
        !configuredDeviceMemoryAdmission().limit_bytes)
    {
      std::clog << "PsiFormer REAL VALUE pair geometry: effective=0 reason=unbounded_or_ineligible_baseline\n";
      return false;
    }
    const auto& plan = image_->plan();
    const auto future_full = ordinary->hooks.spatial_ ? 0 :
        (ordinary->hooks.target_preparation_.max_targets
          ? estimatePreparedTargetVglCost(plan, ordinary->capacity, ordinary->hooks.target_preparation_).device_workspace_bytes
          : estimateTargetVglCost(plan, ordinary->capacity, plan.modelShape().electrons()).device_workspace_bytes);
    const auto extra = spatial_detail::checkedAdd(first->periodicValuePairDeviceBytes(),
        second ? second->periodicValuePairDeviceBytes() : 0, "VALUE pair combined bytes overflow");
    int ordinal = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&ordinal), "VALUE pair preparation current device");
#else
    checkRuntime(cudaGetDevice(&ordinal), "VALUE pair preparation current device");
#endif
    const auto rollback = [&] {
      first->discardPeriodicValuePairs();
      if (second)
        second->discardPeriodicValuePairs();
    };
    try
    {
      auto future_reservation = deviceMemoryMetrics(ordinal).reserveAllocation(future_full);
      { auto optional_preflight = deviceMemoryMetrics(ordinal).reserveAllocation(extra); }
      first->preparePeriodicValuePairs();
      if (second)
        second->preparePeriodicValuePairs();
    }
    catch (const std::length_error&)
    {
      rollback();
      std::clog << "PsiFormer REAL VALUE pair geometry: effective=0 reason=aggregate_fit requested_device_bytes="
                << extra << " future_FULL_numeric_bound=" << future_full << '\n';
      return false;
    }
    catch (const std::bad_alloc&)
    {
      rollback();
      std::clog << "PsiFormer REAL VALUE pair geometry: effective=0 reason=optional_HOST_allocation\n";
      return false;
    }
    catch (const std::runtime_error& error)
    {
      rollback();
      if (!recoverOptionalEnvelopeAllocation(error))
        throw;
      std::clog << "PsiFormer REAL VALUE pair geometry: effective=0 reason=optional_DEVICE_allocation\n";
      return false;
    }
    catch (...)
    {
      rollback();
      throw;
    }
    ordinary->pair_preparation_effective = true;
    if (value16)
      value16->pair_preparation_checked = value16->pair_preparation_effective = true;
    std::clog << "PsiFormer REAL VALUE pair geometry: effective=1 participants=" << (value16 ? 2 : 1)
              << " requested_device_bytes=" << extra << " future_FULL_numeric_bound=" << future_full
              << " HOST_scope=numeric_not_aggregate_fit\n";
    return true;
  }

  /// Capture all FULL microtiles of one explicitly admitted native real OPEN call.
  void beginECPObservation(pf::DirectBatchTileBackendWorkspace& workspace, std::size_t samples,
      const std::vector<double>& inverse_masses, const std::string& physical_source, std::size_t numeric_cap) const
  {
    std::lock_guard lock(control_mutex_);
    auto* context = dynamic_cast<Context*>(&workspace);
    if (!context || context->owner != this || context->observation_capture ||
        preparation_.active_inference_crowds != 1 || context->hooks.target_preparation_.max_targets)
      throw std::invalid_argument("ECP observation capture requires one exclusive untargeted FULL crowd");
    // Two immutable generations can overlap at the parameter-publication fence.
    context->observation_capture = std::make_shared<PreparedDeviceECPObservation>(image_, samples,
        inverse_masses, context->observation_source, physical_source, numeric_cap / 2, numeric_cap / 2);
  }

  std::shared_ptr<PreparedDeviceECPObservation> finishECPObservation(
      pf::DirectBatchTileBackendWorkspace& workspace) const
  {
    std::lock_guard lock(control_mutex_);
    auto* context = dynamic_cast<Context*>(&workspace);
    if (!context || context->owner != this || !context->observation_capture)
      throw std::invalid_argument("ECP observation capture is absent or foreign");
    context->observation_capture->finishCapture(image_->version());
    return std::move(context->observation_capture);
  }

  void cancelECPObservation(pf::DirectBatchTileBackendWorkspace& workspace) const noexcept
  {
    std::lock_guard lock(control_mutex_);
    auto* context = dynamic_cast<Context*>(&workspace);
    if (context && context->owner == this) context->observation_capture.reset();
  }

  void sealECPObservation(pf::DirectBatchTileBackendWorkspace& workspace,
      PreparedDeviceECPObservation& observation, std::uint64_t grid, const std::string& physical_source,
      const std::vector<double>& positions, const std::vector<double>& energy,
      const std::vector<double>& native_kinetic) const
  {
    std::lock_guard lock(control_mutex_);
    auto* context = dynamic_cast<Context*>(&workspace);
    if (!context || context->owner != this || observation.source_.get() != context->observation_source.get())
      throw std::invalid_argument("ECP observation belongs to another native crowd source");
    observation.seal(context->hooks.queue_, image_->version(), grid, physical_source,
        positions, energy, native_kinetic);
  }

  /// Host permit evidence only, not a claim of simultaneous kernel execution.
  std::size_t peakAdmittedInferenceCalls() const { return inference_concurrency_.peakAdmittedCalls(); }

  /// Report the explicit per-crowd sampler admission limit; zero preserves host moves.
  std::size_t metropolisBudget() const noexcept { return preparation_.metropolis_workspace_budget_bytes; }

  /// An explicit additions cap is also the opt-in drift profile discriminator.
  std::size_t metropolisDriftBudget() const noexcept { return preparation_.metropolis_drift_budget_bytes; }

  /// Report the independent explicit per-optimizer/rank real Adam limit.
  std::size_t adamBudget() const noexcept { return preparation_.adam_workspace_budget_bytes; }

  /// Borrow only under the caller's retained crowd resource and model read lease.
  PackedDeviceCoordinatesView residentAcceptedCoordinates(pf::DirectBatchTileBackendWorkspace& workspace,
      std::size_t sample, std::size_t version) const
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    auto* context = dynamic_cast<Context*>(&workspace);
    if (!context || context->owner != this || !context->sampler || !context->resident_coordinates ||
        context->sampler_version != version || version != image_->version())
      throw std::logic_error("Resident accepted coordinates are absent, foreign or stale");
    return context->sampler->acceptedView(sample, 1);
  }

  /// The sole explicit DEVICE-to-HOST coordinate boundary; FULL/H remain dirty.
  void materializeResidentAccepted(pf::DirectBatchTileBackendWorkspace& workspace,
      std::vector<double>& positions, std::size_t version) const
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    auto* context = dynamic_cast<Context*>(&workspace);
    if (!context || context->owner != this || !context->sampler || !context->resident_coordinates ||
        context->sampler_version > version || version != image_->version())
      throw std::logic_error("Resident coordinate fence requires its current sampler");
    context->sampler->exportAccepted(positions);
    std::copy(positions.begin(), positions.end(), context->sampler_positions.begin());
    context->resident_coordinates = false;
  }

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
      bool use_drift = false, NativeDriftCallback drift = {}, bool retain_device_coordinates = false,
      std::size_t particles_per_move = 0,
      const std::function<void(std::vector<std::size_t>&)>& select_indices = {}) const
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    auto* context = dynamic_cast<Context*>(&workspace);
    const auto ne = image_->plan().modelShape().electrons();
    const bool selected = particles_per_move != 0;
    if (!context || context->owner != this || !metropolisBudget() || mixed_parameters_ ||
        positions.empty() || positions.size() % (3 * ne) || inverse_mass.size() != ne ||
        !substeps || !gaussian_draw || !uniform_draw || version != image_->version() ||
        !device_math::isFiniteBinary64(tau) || tau <= 0)
      throw std::invalid_argument("Device sampling requires its explicit real FP64 crowd profile and current image");
    if (selected != bool(select_indices) || particles_per_move >= ne)
      throw std::invalid_argument("Device sampling selected moves require a proper subset and index callback");
    if (use_drift != bool(metropolisDriftBudget()) || (use_drift && !drift))
      throw std::invalid_argument("Device sampling usedrift must match its explicit additions cap and native policy");
    if ((retain_device_coordinates && use_drift) ||
        (context->resident_coordinates && !retain_device_coordinates))
      throw std::invalid_argument("Resident sampling requires no drift and an explicit cold coordinate fence before HOST reuse");
    if (context->resident_coordinates && (context->sampler_masses != inverse_mass ||
        (context->sampler && context->sampler->cost().selected != selected)))
      throw std::invalid_argument("Resident sampling needs a cold coordinate fence before changing masses or selection workspace");
    const auto cost = estimateMetropolisCost(positions.size() / (3 * ne), ne, selected);
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
    const bool reseed = !context->sampler || (!context->resident_coordinates && context->sampler_positions != positions) ||
        context->sampler_masses != inverse_mass || context->sampler->cost().selected != selected;
    const auto selected_count = spatial_detail::checkedProduct(cost.population, particles_per_move,
                                                               "Device sampler selected extent overflow");
    context->sampler_indices.resize(selected_count);
    context->sampler_gaussian.resize(selected ? spatial_detail::checkedProduct(selected_count, 3,
        "Device sampler Gaussian extent overflow") : positions.size());
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
                positions, inverse_mass, metropolisBudget(), metropolisDriftBudget(), drift, selected)
            : PreparedDeviceMetropolis::prepare(*context->hooks.values_, context->hooks.queue_,
                positions, inverse_mass, metropolisBudget(), selected);
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
        if (selected)
        {
          select_indices(context->sampler_indices);
          if (context->sampler_indices.size() != selected_count)
            throw std::invalid_argument("Device sampler index callback changed the selected extent");
        }
        const auto token = context->sampler->propose(context->sampler_gaussian, tau, version,
                                                    context->sampler_indices);
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
      if (!retain_device_coordinates)
      {
        context->sampler->exportAccepted(accepted_positions);
        std::copy(accepted_positions.begin(), accepted_positions.end(), context->sampler_positions.begin());
      }
      context->resident_coordinates = retain_device_coordinates;
      context->executor.release(owner_, context->crowd, context->hooks);
    }
    catch (...)
    {
      // Earlier substeps/RNG draws are not rolled back; the caller must abort.
      context->sampler.reset();
      context->resident_coordinates = false;
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
              << ", accepted_coordinate_authority=" << (retain_device_coordinates ? "DEVICE" : "HOST")
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
              << ", sampler_device_bytes=" << cost.device_bytes
              << ", owner_host_mask_bytes=" << checkedDeviceBytes<std::uint32_t>(cost.population)
              << ", owner_host_bytes=" << cost.host_bytes
              << ", selected_electrons_per_joint_proposal=" << (selected ? particles_per_move : ne)
              << ", host_scratch_bytes=" << sizeof(double) * (context->sampler_positions.capacity() +
                    context->sampler_gaussian.capacity() + context->sampler_uniforms.capacity() +
                    context->sampler_masses.capacity()) + sizeof(std::size_t) * context->sampler_indices.capacity()
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
    if (trainingEpochMode() != PsiFormerGpuTrainingEpochMode::INFERENCE)
      throw std::logic_error("End the native arithmetic epoch before parameter publication");
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
            std::size_t capacity, std::uint64_t identity, TargetVglPreparation target_preparation, bool lazy_spatial)
        : owner(backend), capacity(capacity), mixed_statistics(backend->mixed_statistics_), executor(PreparedDeviceCrowdHooks::backend(),
            PreparedDeviceCrowdHooks::coverage()), crowd(identity),
          hooks(image, model, identity, std::move(mixed_parameters), target_preparation, lazy_spatial)
    {
      positions.reserve(spatial_detail::checkedProduct(capacity,
          spatial_detail::checkedProduct(3, image->plan().modelShape().electrons(), "Device tile coordinates overflow"),
          "Device tile staging overflow"));
      executor.prepare(crowd, model.activeProvenance().layout_fingerprint, capacity, hooks);
      if (owner->large_score_requested_ && !lazy_spatial)
        owner->live_eager_contexts_.fetch_add(1, std::memory_order_release);
    }

    /// Collect existing counters only; do not create graphs or synchronize a queue.
    ~Context() override
    {
      if (owner->large_score_requested_ && !owner->lazy_spatial_)
        owner->live_eager_contexts_.fetch_sub(1, std::memory_order_release);
      // The existing destruction boundary, not the virtual-request hot path.
      try { if (hooks.values_) hooks.values_->reportSparseEnvelopeAnchor(); }
      catch (...) {} // Optional diagnostics cannot interfere with retirement.
      if (mixed_statistics)
      {
        mixed_statistics->successes.fetch_add(hooks.mixedBatches(), std::memory_order_relaxed);
        mixed_statistics->canonical_values.fetch_add(hooks.canonicalValueBatches(), std::memory_order_relaxed);
        mixed_statistics->retries.fetch_add(hooks.mixedRetries(), std::memory_order_relaxed);
        mixed_statistics->contexts.fetch_add(1, std::memory_order_relaxed);
      }
    }

    const PreparedDeviceBatchBackend* const owner;
    const std::size_t capacity;
    const std::shared_ptr<MixedFinalStatistics> mixed_statistics;
    std::mutex evaluation_mutex;
    PsiFormerAcceleratorExecutor executor;
    PsiFormerAcceleratorCrowdState crowd;
    PreparedDeviceCrowdHooks hooks;
    // Declared after hooks: sampler must die before its borrowed graph/queue.
    std::shared_ptr<const std::size_t> observation_source = std::make_shared<const std::size_t>(0);
    std::shared_ptr<PreparedDeviceECPObservation> observation_capture;
    std::unique_ptr<PreparedDeviceMetropolis> sampler;
    std::vector<double> sampler_positions, sampler_gaussian, sampler_uniforms, sampler_masses;
    std::vector<std::size_t> sampler_indices;
    std::size_t sampler_version = 0;
    bool resident_coordinates = false;
    bool sparse_preparation_checked = false, sparse_preparation_effective = false;
    bool pair_preparation_checked = false, pair_preparation_effective = false;
    std::vector<double> positions;
    std::size_t coordinate_generation = 0;
    bool parallel_batch_reported = false;
  };

  /// Report the exact capacity authority only when a FULL graph is actually prepared.
  void reportSpatialPreparation(const TargetVglPreparation& preparation, std::size_t capacity) const
  {
    if (mixed_parameters_ && preparation.max_targets)
    {
      auto cost = estimatePreparedTargetVglCost(image_->plan(), capacity, preparation);
      cost.device_workspace_bytes =
          PreparedDeviceCrowdHooks::mixedTargetDeviceBytes(image_->plan(), capacity, preparation);
      cost.host_workspace_bytes = spatial_detail::checkedAdd(cost.host_workspace_bytes,
          spatial_detail::checkedAdd(sizeof(PsiFormerDeviceNumericalDiagnostics),
              PreparedDeviceCanonicalRealHead::required(image_->plan(), capacity).host_bytes,
              "Canonical target HOST bytes overflow"),
          "Mixed target HOST diagnostic bytes overflow");
      cost.charged_scalar_work_units = spatial_detail::checkedAdd(
          mixedTargetRequestScalarWork(cost.charged_scalar_work_units), canonicalHeadScalarWork(image_->plan(), capacity),
          "Canonical mixed target request work overflow");
      reportDeviceTargetSelection(std::cout, "MIXED_VGL_profile", image_->plan(), capacity,
          preparation.max_targets, cost, preparation.device_workspace_budget_bytes, preparation.scalar_work_budget);
      std::cout << "  Mixed target request work includes one canonical FP64 VALUE head, one approximate sweep and one complete same-K FP64 retry;"
                   " shared canonical/FP32 parameter images and separate scalar VALUE graph excluded; canonical head graph included; descriptor_only=yes; not iteration work or FLOPs\n";
    }
    if (preparation.parallel_trace_workspace_budget_bytes)
      reportDeviceTargetSelection(std::cout, "VGL_parallel_actual_owner", image_->plan(), capacity,
          preparation.max_targets, estimatePreparedTargetVglCost(image_->plan(), capacity, preparation),
          preparation.device_workspace_budget_bytes, preparation.scalar_work_budget);
    if (preparation_.select_vgl_targets)
      reportDeviceTargetSelection(std::cout, "VGL_actual_owner", image_->plan(), capacity,
          preparation.max_targets, estimatePreparedTargetVglCost(image_->plan(), capacity, preparation),
          preparation.device_workspace_budget_bytes, preparation.scalar_work_budget);
  }

  /// Reject absent images before model provenance construction.
  static std::shared_ptr<const PreparedDeviceModel> checkedImage(std::shared_ptr<const PreparedDeviceModel> image)
  {
    if (!image)
      throw std::invalid_argument("Device batch backend requires an immutable model image");
    return image;
  }

  /// Admit only the selected mixed VGL seam and reject incomplete or over-budget profiles.
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
    if (preparation.metropolis_workspace_budget_bytes)
      validateMetropolisEnvironment(plan.environment());
    if (preparation.metropolis_workspace_budget_bytes &&
        (precision != PsiFormerPrecisionPolicy::FP64_REFERENCE ||
         (preparation.metropolis_drift_budget_bytes && plan.environment().boundary != BoundaryCondition::OPEN)))
      throw std::invalid_argument("Explicit device sampling requires real FP64; periodic sampling excludes drift");
    if (precision != PsiFormerPrecisionPolicy::FP64_REFERENCE &&
        (preparation.kinetic.max_targets || preparation.select_vgl_targets || preparation.select_kinetic_targets ||
         preparation.retained_coordinate_workspace_budget_bytes))
      throw std::invalid_argument("Mixed precision admits explicit VGL targets without kinetic/selection/retained preparation");
    preparation = resolveDeviceTargetSelection(plan, preparation);
    validateDeviceExecutionPreparation(plan, preparation);
    if (precision == PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS && preparation.vgl.max_targets)
      PreparedDeviceCrowdHooks::mixedTargetDeviceBytes(plan, 1, preparation.vgl);
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
  const bool lazy_spatial_;
  const bool large_score_requested_;
  const bool large_score_eligible_;
  mutable bool large_score_single_crowd_ = false;
  mutable std::atomic<std::size_t> live_eager_contexts_{0};
  const bool pair_geometry_requested_;
  const bool pair_geometry_eligible_;
  const bool sparse_envelope_requested_;
  const bool sparse_envelope_eligible_;
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
  mutable std::atomic<PsiFormerGpuTrainingEpochMode> training_epoch_mode_{PsiFormerGpuTrainingEpochMode::INFERENCE};
  mutable std::atomic<std::size_t> training_epoch_generation_{0};
  std::shared_ptr<MixedFinalStatistics> mixed_statistics_;
};

} // namespace qmcplusplus::psiformer::device

#endif
