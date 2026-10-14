//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceProductSession.h
 * @brief Exclusive, versioned device parameter products using real control leases.
 *
 * Include after the native executor definitions, like DeviceBatchBackend.h.
 * One configuration is replayed at a time into fixed scratch. This bounded
 * compatibility lane avoids B-by-P storage; it is not a batched training claim.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_PRODUCT_SESSION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_PRODUCT_SESSION_H

#include "PsiFormerDeviceScoreReverse.h"
#include "PsiFormerDeviceScoreJvp.h"
#include "PsiFormerDeviceKineticReverse.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorExecutor.h"

#include <exception>
#include <functional>
#include <mutex>

namespace qmcplusplus::psiformer::device
{

/** Lease one prepared numerical derivative lane, sharing its backend's image.
 * The backend must outlive the session; the model owner and mutex are borrowed.
 * An outer provider scratch lease rejects callback re-entry and serializes use.
 */
class PreparedDeviceProductSession final : private PsiFormerAcceleratorHooks
{
public:
  /// Allocate scratch once while the owning backend's control mutex is held.
  PreparedDeviceProductSession(const std::shared_ptr<const PreparedDeviceModel>& active_image,
                               PsiFormerAcceleratorSharedModel& owner,
                               std::mutex& control_mutex, std::uint64_t identity)
      : active_image_(active_image), image_(active_image), owner_(owner), mutex_(control_mutex),
        blas_(queue_), values_(image_, 1, queue_, blas_, SpatialJetMode::VALUE_ONLY, true),
        spatial_(image_, 1, queue_, blas_, SpatialJetMode::FULL_VGL, false, true),
        score_(image_, 1, queue_, blas_), jvp_(image_, 1, queue_, blas_),
        kinetic_(image_, 1, queue_, blas_),
        executor_(compiledBackend(), coverage()), crowd_(identity), identity_(identity),
        positions_(3 * image_->plan().modelShape().electrons()),
        drift_(positions_.size()), masses_(image_->plan().modelShape().electrons()),
        score_response_(image_->plan().parameterCount()), kinetic_response_(score_response_.size()),
        products_(1)
  {
    executor_.prepare(crowd_, owner_.activeProvenance().layout_fingerprint, 1, *this);
  }

  // Prepared graphs borrow this address-stable session's queue and BLAS handle.
  PreparedDeviceProductSession(const PreparedDeviceProductSession&) = delete;
  PreparedDeviceProductSession& operator=(const PreparedDeviceProductSession&) = delete;

  /// Return a canonical real score, valid until this session's next score call.
  const std::vector<double>& score(const double* positions, std::size_t version)
  {
    execute(PsiFormerAcceleratorOperation::PARAMETER_SCORE, positions, version, [&]() {
      values_.evaluate(positions_);
      score_.reverse(values_, weights_);
      score_.downloadResponse(score_response_);
    });
    return score_response_;
  }

  /// Return the complete real kinetic parameter response with caller's total drift.
  const std::vector<double>& kinetic(const double* positions, const double* total_drift,
                                     const double* inverse_masses, std::size_t version)
  {
    if (!total_drift || !inverse_masses)
      throw std::invalid_argument("Device kinetic session requires total drift and inverse masses");
    execute(PsiFormerAcceleratorOperation::KINETIC_RESPONSE, positions, version, [&]() {
      std::copy_n(total_drift, drift_.size(), drift_.begin());
      std::copy_n(inverse_masses, masses_.size(), masses_.begin());
      spatial_.evaluate(positions_);
      kinetic_.evaluate(spatial_, weights_, masses_, drift_);
      kinetic_.downloadResponse(kinetic_response_);
    });
    return kinetic_response_;
  }

  /// Apply a real parameter direction without materializing a parameter score.
  double scoreJvp(const double* positions, const std::vector<double>& direction, std::size_t version)
  {
    execute(PsiFormerAcceleratorOperation::PARAMETER_SCORE, positions, version, [&]() {
      values_.evaluate(positions_);
      jvp_.evaluate(values_, direction);
      jvp_.downloadProducts(products_);
    });
    return products_[0];
  }

  /// Account allocated device scratch, excluding the shared immutable image.
  std::size_t deviceWorkspaceBytes() const
  {
    std::size_t bytes = 0;
    for (auto amount : {values_.deviceWorkspaceBytes(), spatial_.deviceWorkspaceBytes(),
                        score_.deviceWorkspaceBytes(), jvp_.deviceWorkspaceBytes(),
                        kinetic_.deviceWorkspaceBytes()})
      bytes = spatial_detail::checkedAdd(bytes, amount, "Device product session byte overflow");
    return bytes;
  }

  /// Count completed numerical products, excluding failed/stale actions.
  std::size_t numericalProducts() const noexcept { return numerical_products_; }

  /// Count host vectors independently; model images and vendor-private bytes are excluded.
  std::size_t hostWorkspaceBytes() const
  {
    std::size_t bytes = 0;
    for (const auto* values : {&positions_, &drift_, &masses_, &score_response_,
                               &kinetic_response_, &products_, &weights_})
      bytes = spatial_detail::checkedAdd(bytes, checkedDeviceBytes<double>(values->capacity()),
                                         "Device product host staging overflow");
    for (auto amount : {values_.hostWorkspaceBytes(), spatial_.hostWorkspaceBytes(),
                        score_.hostWorkspaceBytes(), jvp_.hostWorkspaceBytes(), kinetic_.hostWorkspaceBytes()})
      bytes = spatial_detail::checkedAdd(bytes, amount, "Device product host status overflow");
    return bytes;
  }

private:
  /// Match the vendor compiled into the shared kernel translation units.
  static constexpr PsiFormerAcceleratorBackend compiledBackend() noexcept
  {
#ifdef QMC_CUDA2HIP
    return PsiFormerAcceleratorBackend::HIP;
#else
    return PsiFormerAcceleratorBackend::CUDA;
#endif
  }

  /// Advertise only the product roots actually submitted by this session.
  static constexpr PsiFormerAcceleratorOperationMask coverage() noexcept
  {
    return acceleratorOperationBit(PsiFormerAcceleratorOperation::PARAMETER_SCORE) |
        acceleratorOperationBit(PsiFormerAcceleratorOperation::KINETIC_RESPONSE);
  }

  /// Catch up released fixed-shape resources after an accepted publication.
  void rebind()
  {
    if (image_.get() == active_image_.get())
      return;
    image_->validateSuccessor(active_image_);
    values_.rebindModel(active_image_);
    spatial_.rebindModel(active_image_);
    score_.rebindModel(active_image_);
    jvp_.rebindModel(active_image_);
    kinetic_.rebindModel(active_image_);
    image_ = active_image_;
    ready_ = false;
  }

  /// Preserve provenance and failure recovery around one synchronous product.
  void execute(PsiFormerAcceleratorOperation operation, const double* positions,
                std::size_t version, const std::function<void()>& action)
  {
    std::lock_guard lock(mutex_);
    ready_ = false;
    if (!positions || version != active_image_->version())
      throw std::invalid_argument("Device parameter product has missing coordinates or stale version");
    if (coordinate_generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Device product coordinate generation exhausted");
    rebind();
    std::copy_n(positions, positions_.size(), positions_.begin());
    ++coordinate_generation_;
    action_ = &action;
    operation_ = operation;
    try
    {
      executor_.acquire(owner_, crowd_, *this);
      const auto pending = executor_.evaluate(owner_, crowd_, acceleratorOperationBit(operation),
                                               1, coordinate_generation_, *this);
      executor_.wait(crowd_, pending.result_generation, *this);
      (void)crowd_.completedResult(pending.result_generation, coordinate_generation_);
      executor_.release(owner_, crowd_, *this);
      ++numerical_products_;
      action_ = nullptr;
    }
    catch (...)
    {
      if (crowd_.poisoned())
      {
        executor_.recover(owner_, crowd_, *this);
        executor_.prepare(crowd_, owner_.activeProvenance().layout_fingerprint, 1, *this);
      }
      else if (crowd_.acquired())
        executor_.release(owner_, crowd_, *this);
      action_ = nullptr;
      throw;
    }
  }

  /// Execute product events without bypassing model, crowd or result generations.
  void submit(const PsiFormerAcceleratorEvent& event) override
  {
    const auto model = owner_.activeProvenance();
    if (event.backend != compiledBackend() || event.crowd_identity != identity_ ||
        event.layout_fingerprint != model.layout_fingerprint)
      throw std::logic_error("Device product event has foreign routing");
    if (event.kind == PsiFormerAcceleratorEventKind::CROWD_PREPARE)
    {
      if (event.item_count != 1)
        throw std::logic_error("Device product lane requires one prepared configuration");
      resource_generation_ = event.resource_generation;
      return;
    }
    if (event.resource_generation != resource_generation_)
      throw std::logic_error("Device product event has stale scratch");
    if (event.kind == PsiFormerAcceleratorEventKind::CROWD_RECOVER)
    {
      queue_.sync();
      failure_ = nullptr;
      recorded_ = ready_ = false;
      return;
    }
    if (event.model_identity != model.model_identity || event.model_version != image_->version() ||
        event.model_slot != model.active_slot || event.publication_generation != model.publication_generation)
      throw std::logic_error("Device product event has stale model provenance");
    switch (event.kind)
    {
    case PsiFormerAcceleratorEventKind::CROWD_ACQUIRE:
      acquisition_generation_ = event.acquisition_generation;
      return;
    case PsiFormerAcceleratorEventKind::MODEL_WAIT:
      return;
    case PsiFormerAcceleratorEventKind::INPUT_UPLOAD:
      if (event.coordinate_generation != coordinate_generation_ || event.item_count != 1 ||
          event.acquisition_generation != acquisition_generation_)
        throw std::logic_error("Device product upload has stale coordinates/acquisition");
      result_generation_ = event.result_generation;
      return;
    case PsiFormerAcceleratorEventKind::OPERATION:
      validateResult(event);
      if (!action_ || event.operation != operation_)
        throw std::logic_error("Device product event differs from bound numerical action");
      try { (*action_)(); }
      catch (...) { failure_ = std::current_exception(); }
      return;
    case PsiFormerAcceleratorEventKind::RESULT_RECORD:
      validateResult(event);
      if (failure_)
        std::rethrow_exception(failure_);
      recorded_ = true;
      return;
    case PsiFormerAcceleratorEventKind::RESULT_WAIT:
      validateResult(event);
      if (!recorded_)
        throw std::logic_error("Device product wait has no recorded result");
      recorded_ = false;
      ready_ = true;
      return;
    case PsiFormerAcceleratorEventKind::CROWD_RELEASE:
      if (recorded_ || failure_)
        throw std::logic_error("Device product release has unfinished work");
      return;
    default:
      throw std::logic_error("Device product session received a publication event");
    }
  }

  /// Check all request identities before executing or exposing a numerical product.
  void validateResult(const PsiFormerAcceleratorEvent& event) const
  {
    if (event.coordinate_generation != coordinate_generation_ || event.result_generation != result_generation_ ||
        event.acquisition_generation != acquisition_generation_ || event.item_count != 1)
      throw std::logic_error("Device product event has stale result provenance");
  }

  const std::shared_ptr<const PreparedDeviceModel>& active_image_;
  std::shared_ptr<const PreparedDeviceModel> image_;
  PsiFormerAcceleratorSharedModel& owner_;
  std::mutex& mutex_;
  compute::Queue<PlatformKind::CUDA> queue_;
  AcceleratorBlasHandle blas_;
  PreparedDeviceInference values_, spatial_;
  PreparedDeviceScoreReverse score_;
  PreparedDeviceScoreJvp jvp_;
  PreparedDeviceKineticReverse kinetic_;
  PsiFormerAcceleratorExecutor executor_;
  PsiFormerAcceleratorCrowdState crowd_;
  const std::uint64_t identity_;
  std::vector<double> positions_, drift_, masses_, score_response_, kinetic_response_, products_;
  const std::vector<double> weights_{1.0};
  const std::function<void()>* action_ = nullptr;
  std::exception_ptr failure_;
  PsiFormerAcceleratorOperation operation_ = PsiFormerAcceleratorOperation::PARAMETER_SCORE;
  std::size_t resource_generation_ = 0, acquisition_generation_ = 0;
  std::size_t coordinate_generation_ = 0, result_generation_ = 0, numerical_products_ = 0;
  bool recorded_ = false, ready_ = false;
};

} // namespace qmcplusplus::psiformer::device

#endif
