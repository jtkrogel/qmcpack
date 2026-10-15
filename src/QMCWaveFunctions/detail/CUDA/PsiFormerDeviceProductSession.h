//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceProductSession.h
 * @brief Exclusive, versioned device parameter products using real control leases.
 *
 * Include after the native executor definitions, like DeviceBatchBackend.h.
 * Score tapes may prepare up to four configurations; other product roots remain
 * single-configuration. Weighted score tiles avoid B-by-P storage.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_PRODUCT_SESSION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_PRODUCT_SESSION_H

#include "PsiFormerDeviceScoreReverse.h"
#include "PsiFormerDeviceScoreJvp.h"
#include "PsiFormerDeviceScoreContraction.h"
#include "PsiFormerDeviceKineticReverse.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorExecutor.h"

#include <exception>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>

namespace qmcplusplus::psiformer::device
{

/** Fix the numerical roots and their scratch at an explicit preparation boundary. */
enum class ProductSessionProfile
{
  SCORE_ONLY,
  SCORE_AND_KINETIC
};

/** Lease one prepared numerical derivative lane, sharing its backend's image.
 * The backend must outlive the session; the model owner and mutex are borrowed.
 * An outer provider scratch lease rejects callback re-entry and serializes use.
 */
class PreparedDeviceProductSession final : private PsiFormerAcceleratorHooks
{
  /** Keep the lifted forward/reverse pair absent from a score-only lane. */
  struct KineticWorkspace
  {
    PreparedDeviceInference spatial;
    PreparedDeviceKineticReverse reverse;

    KineticWorkspace(const std::shared_ptr<const PreparedDeviceModel>& image,
        compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
        : spatial(image, 1, queue, blas, SpatialJetMode::FULL_VGL, false, true),
          reverse(image, 1, queue, blas)
    {}
  };

public:
  /// Allocate scratch once while the owning backend's control mutex is held.
  PreparedDeviceProductSession(const std::shared_ptr<const PreparedDeviceModel>& active_image,
                               PsiFormerAcceleratorSharedModel& owner,
                               std::mutex& control_mutex, std::uint64_t identity,
                               ProductSessionProfile profile = ProductSessionProfile::SCORE_AND_KINETIC,
                               std::size_t score_capacity = 1)
      : active_image_(active_image), image_(active_image), owner_(owner), mutex_(control_mutex),
        profile_(validatedProfile(profile)),
        score_capacity_(admittedScoreCapacity(image_->plan().modelShape(), score_capacity)),
        blas_(queue_), values_(image_, score_capacity_, queue_, blas_, SpatialJetMode::VALUE_ONLY, true),
        score_(image_, score_capacity_, queue_, blas_), jvp_(image_, 1, queue_, blas_),
        kinetic_(profile_ == ProductSessionProfile::SCORE_AND_KINETIC
            ? std::make_unique<KineticWorkspace>(image_, queue_, blas_) : nullptr),
        executor_(compiledBackend(), coverage(profile_)), crowd_(identity), identity_(identity),
        positions_(3 * image_->plan().modelShape().electrons()),
        drift_(positions_.size()), masses_(image_->plan().modelShape().electrons()),
        score_response_(image_->plan().parameterCount()),
        kinetic_response_(kinetic_ ? score_response_.size() : 0),
        products_(1), channel_coefficients_(image_->plan().modelShape().determinants),
        channel_phases_(channel_coefficients_.size()), channel_logs_(channel_coefficients_.size()),
        factors_(channel_coefficients_.size()), cusp_(1),
        orbital_backflow_(channel_coefficients_.size() * masses_.size() * masses_.size()),
        orbital_envelope_(orbital_backflow_.size()), orbital_seeds_(orbital_backflow_.size())
  {
    // Logical single-root sizes stay 3Ne even when position capacity admits tiles.
    positions_.reserve(spatial_detail::checkedProduct(positions_.size(), score_capacity_,
        "Device score tile position extent overflow"));
    tile_weights_.reserve(score_capacity_);
    executor_.prepare(crowd_, owner_.activeProvenance().layout_fingerprint, score_capacity_, *this);
  }

  // Prepared graphs borrow this address-stable session's queue and BLAS handle.
  PreparedDeviceProductSession(const PreparedDeviceProductSession&) = delete;
  PreparedDeviceProductSession& operator=(const PreparedDeviceProductSession&) = delete;

  /// Report the immutable capabilities selected before any numerical request.
  ProductSessionProfile profile() const noexcept { return profile_; }

  /// Immutable prepared capacity; capability reads need no lazy-owner access.
  std::size_t scoreCapacity() const noexcept { return score_capacity_; }

  /// Capacity4 is qualified only for this small-model envelope. Larger models
  /// retain capacity1; this policy is not a claim that their full kinetic tape
  /// fits. In particular, never multiply 100--300-electron head/attention storage
  /// by four merely because a caller requested the small-system optimization.
  static std::size_t admittedScoreCapacity(const ModelShape& shape, std::size_t requested)
  {
    const auto capacity = validatedScoreCapacity(requested);
    return shape.spin_up_electrons <= 4 && shape.spin_down_electrons <= 4 && shape.electrons() <= 4 &&
        shape.determinants <= 16 && shape.feature_dimension <= 256 &&
        shape.attention_heads <= 4 && shape.attention_blocks <= 4 && shape.nuclei <= 2 ? capacity : 1;
  }

  /// Completed weighted tile counters are diagnostic, not admission state.
  std::size_t weightedScoreTiles() const noexcept { return weighted_score_tiles_; }

  std::size_t weightedScoreConfigurations() const noexcept { return weighted_score_configurations_; }

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

  /// Explicit cold preparation for the actual admitted score-channel count.
  /// Host/kinetic compatibility callers never need this optional O(CP) storage.
  void prepareScoreContractions(std::size_t channels, std::size_t version)
  {
    std::lock_guard lock(mutex_);
    invalidateScoreContraction();
    validateContractionControl(version);
    if (channels == 0 || channels > 3)
      throw std::invalid_argument("Device score contraction supports one to three channels");
    if (!score_contraction_ || score_contraction_->channelCapacity() < channels)
    {
      // A previous incomplete transaction may still have ordered AXPYs queued.
      // Growth is an explicit phase boundary, so drain before replacing storage.
      queue_.sync();
      auto prepared = std::make_unique<PreparedDeviceScoreContraction>(
          image_->plan().parameterCount(), channels, queue_);
      score_contraction_ = std::move(prepared);
      contraction_device_bytes_.store(score_contraction_->deviceWorkspaceBytes(), std::memory_order_release);
      contraction_host_bytes_.store(score_contraction_->hostWorkspaceBytes(), std::memory_order_release);
      contraction_capacity_.store(channels, std::memory_order_release);
      contraction_generation_.fetch_add(1, std::memory_order_release);
    }
  }

  /// Start a versioned contraction under the caller's outer exclusive scratch lease.
  void beginScoreContraction(std::size_t channels, std::size_t version)
  {
    std::lock_guard lock(mutex_);
    invalidateScoreContraction();
    validateContractionControl(version);
    if (!score_contraction_)
      throw std::logic_error("Device score contraction was not prepared");
    score_contraction_->begin(channels);
  }

  /// Keep one sample's canonical score resident and fold all algebraic channels.
  void accumulateScoreContraction(const double* positions, const std::complex<double>* coefficients,
                                  std::size_t channels, std::size_t version)
  {
    execute(PsiFormerAcceleratorOperation::PARAMETER_SCORE, positions, version, [&]() {
      if (!score_contraction_)
        throw std::logic_error("Device score contraction was not prepared");
      values_.evaluate(positions_);
      const double* response = score_.reverse(values_, weights_);
      score_contraction_->accumulate(response, coefficients, channels);
    });
  }

  /// One batched forward/reverse contracts real sample weights directly to P.
  /// The outer C1 transaction spans all tiles and admits the final sum together.
  void accumulateWeightedScoreTile(const double* positions, const double* weights,
                                    std::size_t count, std::size_t version)
  {
    execute(PsiFormerAcceleratorOperation::PARAMETER_SCORE, positions, version, [&]() {
      if (!weights || !score_contraction_)
        throw std::invalid_argument("Weighted score tile requires weights and a prepared contraction");
      tile_weights_.resize(count);
      std::copy_n(weights, count, tile_weights_.begin());
      // reverse() validates every finite signed weight and every determinant,
      // including configurations with zero weight. It returns a weighted sum.
      values_.evaluate(positions_);
      const double* response = score_.reverse(values_, tile_weights_);
      const std::complex<double> unit(1, 0);
      score_contraction_->accumulate(response, &unit, 1);
    }, count);
    ++weighted_score_tiles_;
    weighted_score_configurations_ += count;
  }

  /// Admit the complete finite sum before exposing any channel to the host.
  void finishScoreContraction(std::size_t version)
  {
    std::lock_guard lock(mutex_);
    try
    {
      validateContractionControl(version);
      if (!score_contraction_)
        throw std::logic_error("Device score contraction was not prepared");
      score_contraction_->finish();
    }
    catch (...)
    {
      invalidateScoreContraction();
      throw;
    }
  }

  /// Transfer one canonical complex P-vector after all channels have been admitted.
  void downloadScoreContraction(std::size_t channel, std::complex<double>* destination,
                                 std::size_t count, std::size_t version)
  {
    std::lock_guard lock(mutex_);
    try
    {
      validateContractionControl(version);
      if (!score_contraction_)
        throw std::logic_error("Device score contraction was not prepared");
      score_contraction_->download(channel, destination, count);
    }
    catch (...)
    {
      invalidateScoreContraction();
      throw;
    }
  }

  std::size_t scoreContractionCapacity() const noexcept
  {
    // Capability admission may run before the outer scratch lease is acquired.
    return contraction_capacity_.load(std::memory_order_acquire);
  }

  std::size_t scoreContractionGeneration() const noexcept
  {
    return contraction_generation_.load(std::memory_order_acquire);
  }

  /// Return the complete real kinetic parameter response with caller's total drift.
  const std::vector<double>& kinetic(const double* positions, const double* total_drift,
                                     const double* inverse_masses, std::size_t version)
  {
    requireKinetic();
    if (!total_drift || !inverse_masses)
      throw std::invalid_argument("Device kinetic session requires total drift and inverse masses");
    execute(PsiFormerAcceleratorOperation::KINETIC_RESPONSE, positions, version, [&]() {
      std::copy_n(total_drift, drift_.size(), drift_.begin());
      std::copy_n(inverse_masses, masses_.size(), masses_.begin());
      kinetic_->spatial.evaluate(positions_);
      kinetic_->reverse.evaluate(kinetic_->spatial, weights_, masses_, drift_);
      kinetic_->reverse.downloadResponse(kinetic_response_);
    });
    return kinetic_response_;
  }

  /// Return the gradient-only kinetic response for an explicit total drift.
  /// A fixed complex outer phase supplies its imaginary drift through this lane.
  const std::vector<double>& gradientResponse(const double* positions, const double* total_drift,
      const double* inverse_masses, std::size_t version)
  {
    requireKinetic();
    if (!total_drift || !inverse_masses)
      throw std::invalid_argument("Device gradient response requires total drift and inverse masses");
    execute(PsiFormerAcceleratorOperation::KINETIC_RESPONSE, positions, version, [&]() {
      std::copy_n(total_drift, drift_.size(), drift_.begin());
      std::copy_n(inverse_masses, masses_.size(), masses_.begin());
      kinetic_->spatial.evaluate(positions_);
      kinetic_->reverse.evaluateGradientResponse(kinetic_->spatial, weights_, masses_, drift_);
      kinetic_->reverse.downloadResponse(kinetic_response_);
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

  /// Upload one direction while the crowd is released, without creating a fake
  /// coordinate/result lease. The outer scratch lease spans all subsequent uses.
  void prepareScoreDirection(const std::vector<double>& direction, std::size_t version)
  {
    std::lock_guard lock(mutex_);
    ready_ = false;
    jvp_.clearPreparedDirection();
    if (crowd_.acquired() || crowd_.poisoned())
      throw std::logic_error("Device score direction preparation requires a released, healthy crowd");
    if (version != active_image_->version())
      throw std::invalid_argument("Device score direction preparation has stale version");
    rebind();
    jvp_.prepareDirection(direction);
  }

  /// Apply the resident direction under the ordinary versioned product lease.
  /// Publication invalidates the direction; callers must prepare it again.
  double scoreJvpPrepared(const double* positions, std::size_t version)
  {
    execute(PsiFormerAcceleratorOperation::PARAMETER_SCORE, positions, version, [&]() {
      values_.evaluate(positions_);
      jvp_.evaluatePrepared(values_);
      jvp_.downloadProducts(products_);
    });
    return products_[0];
  }

  /// Return independent determinant amplitudes, including the common cusp.
  void channelAmplitudes(const double* positions, double* phases, double* logs,
                          std::size_t count, std::size_t version)
  {
    if (!phases || !logs || count != channel_phases_.size())
      throw std::invalid_argument("Device channel amplitude output extent mismatch");
    execute(PsiFormerAcceleratorOperation::PARAMETER_SCORE, positions, version, [&]() {
      values_.evaluate(positions_);
      values_.downloadValueFactorizations(factors_);
      values_.downloadValueCusp(cusp_);
      for (std::size_t channel = 0; channel < count; ++channel)
      {
        channel_phases_[channel] = factors_[channel].phase;
        channel_logs_[channel] = factors_[channel].phase == 0.0 ? factors_[channel].log_abs
                                                               : factors_[channel].log_abs + cusp_[0];
        if (factors_[channel].phase != 0.0 && !device_math::isFiniteBinary64(channel_logs_[channel]))
          throw std::runtime_error("Device channel amplitude is nonfinite");
      }
    });
    std::copy(channel_phases_.begin(), channel_phases_.end(), phases);
    std::copy(channel_logs_.begin(), channel_logs_.end(), logs);
  }

  /// Contract arbitrary finite independent-channel coefficients in one reverse.
  const std::vector<double>& channelScore(const double* positions, const double* coefficients,
                                          std::size_t count, std::size_t version)
  {
    if (!coefficients || count != channel_coefficients_.size())
      throw std::invalid_argument("Device channel score coefficient extent mismatch");
    execute(PsiFormerAcceleratorOperation::PARAMETER_SCORE, positions, version, [&]() {
      std::copy_n(coefficients, count, channel_coefficients_.begin());
      values_.evaluate(positions_);
      score_.reverseChannels(values_, channel_coefficients_);
      score_.downloadResponse(score_response_);
    });
    return score_response_;
  }

  /// Reverse the per-spin normalized raw-orbital MSE against a supplied teacher target.
  /// Teacher construction is external; neither a determinant-sum nor cusp loss is added.
  double orbitalMse(const double* positions, const double* target, std::size_t count,
                    double* gradient, std::size_t parameter_count, std::size_t version)
  {
    if (!target || count != orbital_seeds_.size() || !gradient || parameter_count != score_response_.size())
      throw std::invalid_argument("Device orbital-MSE target/gradient extent mismatch");
    double loss = 0;
    execute(PsiFormerAcceleratorOperation::PARAMETER_SCORE, positions, version, [&]() {
      for (std::size_t index = 0; index < count; ++index)
        if (!device_math::isFiniteBinary64(target[index]))
          throw std::invalid_argument("Device orbital target is nonfinite");
      values_.evaluate(positions_);
      values_.downloadValueOrbitalFactors(orbital_backflow_, orbital_envelope_);
      const auto& shape = image_->plan().modelShape();
      const auto ne = shape.electrons();
      for (std::size_t index = 0; index < count; ++index)
      {
        const auto electron = (index / ne) % ne;
        const auto spin_size = electron < shape.spin_up_electrons ? shape.spin_up_electrons : shape.spin_down_electrons;
        const double scale = 1.0 / static_cast<double>(shape.determinants * spin_size * ne);
        const double residual = orbital_backflow_[index] * orbital_envelope_[index] - target[index];
        orbital_seeds_[index] = 2.0 * scale * residual;
        loss += scale * residual * residual;
      }
      if (!device_math::isFiniteBinary64(loss))
        throw std::runtime_error("Device orbital-MSE loss is nonfinite");
      score_.reverseOrbitals(values_, orbital_seeds_);
      score_.downloadResponse(score_response_);
    });
    std::copy(score_response_.begin(), score_response_.end(), gradient);
    return loss;
  }

  /// Account allocated device scratch, excluding the shared immutable image.
  std::size_t deviceWorkspaceBytes() const
  {
    std::size_t bytes = 0;
    for (auto amount : {values_.deviceWorkspaceBytes(), score_.deviceWorkspaceBytes(), jvp_.deviceWorkspaceBytes()})
      bytes = spatial_detail::checkedAdd(bytes, amount, "Device product session byte overflow");
    if (kinetic_)
    {
      bytes = spatial_detail::checkedAdd(bytes, kinetic_->spatial.deviceWorkspaceBytes(),
          "Device product spatial workspace overflow");
      bytes = spatial_detail::checkedAdd(bytes, kinetic_->reverse.deviceWorkspaceBytes(),
          "Device product kinetic workspace overflow");
    }
    bytes = spatial_detail::checkedAdd(bytes, contraction_device_bytes_.load(std::memory_order_acquire),
                                       "Device product contraction byte overflow");
    return bytes;
  }

  /// Count completed numerical products, excluding failed/stale actions.
  std::size_t numericalProducts() const noexcept { return numerical_products_; }

  /// Count host vectors independently; model images and vendor-private bytes are excluded.
  std::size_t hostWorkspaceBytes() const
  {
    std::size_t bytes = 0;
    for (const auto* values : {&positions_, &drift_, &masses_, &score_response_,
                               &kinetic_response_, &products_, &weights_, &channel_coefficients_,
                               &channel_phases_, &channel_logs_, &cusp_, &orbital_backflow_,
                               &orbital_envelope_, &orbital_seeds_, &tile_weights_})
      bytes = spatial_detail::checkedAdd(bytes, checkedDeviceBytes<double>(values->capacity()),
                                         "Device product host staging overflow");
    for (auto amount : {values_.hostWorkspaceBytes(), score_.hostWorkspaceBytes(), jvp_.hostWorkspaceBytes()})
      bytes = spatial_detail::checkedAdd(bytes, amount, "Device product host status overflow");
    if (kinetic_)
    {
      bytes = spatial_detail::checkedAdd(bytes, kinetic_->spatial.hostWorkspaceBytes(),
          "Device product spatial host status overflow");
      bytes = spatial_detail::checkedAdd(bytes, kinetic_->reverse.hostWorkspaceBytes(),
          "Device product kinetic host status overflow");
    }
    bytes = spatial_detail::checkedAdd(bytes,
        checkedDeviceBytes<device_determinant::FactorizationMetadata>(factors_.capacity()),
        "Device product channel factorization overflow");
    bytes = spatial_detail::checkedAdd(bytes, contraction_host_bytes_.load(std::memory_order_acquire),
                                       "Device product contraction host byte overflow");
    return bytes;
  }

private:
  /// Reject an invalid profile before allocating its numerical workspaces.
  static ProductSessionProfile validatedProfile(ProductSessionProfile profile)
  {
    if (profile != ProductSessionProfile::SCORE_ONLY && profile != ProductSessionProfile::SCORE_AND_KINETIC)
      throw std::invalid_argument("Unknown device product preparation profile");
    return profile;
  }

  /// Keep tiling bounded and reject capacity before numerical allocation.
  static std::size_t validatedScoreCapacity(std::size_t capacity)
  {
    if (capacity == 0 || capacity > 4)
      throw std::invalid_argument("Device score tile capacity must be between one and four");
    return capacity;
  }

  /// Missing roots are a preparation contract error, before any product event.
  /// A rejected request cannot allocate scratch or invalidate a prepared direction.
  void requireKinetic() const
  {
    if (!kinetic_)
      throw std::logic_error("Device product session was prepared for score products only");
  }

  /// A failed/rebound lane can never publish an earlier partial or completed sum.
  void invalidateScoreContraction() noexcept
  {
    if (score_contraction_)
      score_contraction_->invalidate();
  }

  /// Preparation/finalization do not fabricate coordinate/result executor events.
  void validateContractionControl(std::size_t version)
  {
    if (crowd_.acquired() || crowd_.poisoned() || version != active_image_->version())
    {
      invalidateScoreContraction();
      throw std::invalid_argument("Device score contraction requires a released healthy crowd and current version");
    }
    active_image_->validateCurrentDevice();
    rebind();
  }

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
  static constexpr PsiFormerAcceleratorOperationMask coverage(ProductSessionProfile profile) noexcept
  {
    if (profile == ProductSessionProfile::SCORE_ONLY)
      return acceleratorOperationBit(PsiFormerAcceleratorOperation::PARAMETER_SCORE);
    return acceleratorOperationBit(PsiFormerAcceleratorOperation::PARAMETER_SCORE) |
        acceleratorOperationBit(PsiFormerAcceleratorOperation::KINETIC_RESPONSE);
  }

  /// Catch up released fixed-shape resources after an accepted publication.
  void rebind()
  {
    if (image_.get() == active_image_.get())
      return;
    invalidateScoreContraction();
    image_->validateSuccessor(active_image_);
    values_.rebindModel(active_image_);
    score_.rebindModel(active_image_);
    jvp_.rebindModel(active_image_);
    if (kinetic_)
    {
      kinetic_->spatial.rebindModel(active_image_);
      kinetic_->reverse.rebindModel(active_image_);
    }
    image_ = active_image_;
    ready_ = false;
  }

  /// Preserve provenance and failure recovery around one synchronous product.
  void execute(PsiFormerAcceleratorOperation operation, const double* positions,
                std::size_t version, const std::function<void()>& action, std::size_t count = 1)
  {
    std::lock_guard lock(mutex_);
    ready_ = false;
    if (!positions || version != active_image_->version() || count == 0 || count > score_capacity_)
    {
      invalidateScoreContraction();
      throw std::invalid_argument("Device parameter product has missing coordinates, stale version or invalid tile count");
    }
    if (coordinate_generation_ == std::numeric_limits<std::size_t>::max())
    {
      invalidateScoreContraction();
      throw std::overflow_error("Device product coordinate generation exhausted");
    }
    rebind();
    request_count_ = count;
    positions_.resize(count * 3 * masses_.size());
    std::copy_n(positions, positions_.size(), positions_.begin());
    ++coordinate_generation_;
    action_ = &action;
    operation_ = operation;
    try
    {
      executor_.acquire(owner_, crowd_, *this);
      const auto pending = executor_.evaluate(owner_, crowd_, acceleratorOperationBit(operation),
                                               request_count_, coordinate_generation_, *this);
      executor_.wait(crowd_, pending.result_generation, *this);
      (void)crowd_.completedResult(pending.result_generation, coordinate_generation_);
      executor_.release(owner_, crowd_, *this);
      ++numerical_products_;
      action_ = nullptr;
    }
    catch (...)
    {
      invalidateScoreContraction();
      if (crowd_.poisoned())
      {
        executor_.recover(owner_, crowd_, *this);
        executor_.prepare(crowd_, owner_.activeProvenance().layout_fingerprint, score_capacity_, *this);
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
      if (event.item_count != score_capacity_)
        throw std::logic_error("Device product lane has the wrong prepared score capacity");
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
      if (event.coordinate_generation != coordinate_generation_ || event.item_count != request_count_ ||
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
        event.acquisition_generation != acquisition_generation_ || event.item_count != request_count_)
      throw std::logic_error("Device product event has stale result provenance");
  }

  const std::shared_ptr<const PreparedDeviceModel>& active_image_;
  std::shared_ptr<const PreparedDeviceModel> image_;
  PsiFormerAcceleratorSharedModel& owner_;
  std::mutex& mutex_;
  const ProductSessionProfile profile_;
  const std::size_t score_capacity_;
  compute::Queue<PlatformKind::CUDA> queue_;
  AcceleratorBlasHandle blas_;
  PreparedDeviceInference values_;
  PreparedDeviceScoreReverse score_;
  PreparedDeviceScoreJvp jvp_;
  std::unique_ptr<KineticWorkspace> kinetic_;
  std::unique_ptr<PreparedDeviceScoreContraction> score_contraction_;
  // Publish metadata without exposing a concurrently replaced lazy owner.
  // Provider diagnostics hold the outer scratch mutex for a coherent snapshot.
  std::atomic<std::size_t> contraction_capacity_{0}, contraction_device_bytes_{0}, contraction_host_bytes_{0};
  std::atomic<std::size_t> contraction_generation_{0};
  PsiFormerAcceleratorExecutor executor_;
  PsiFormerAcceleratorCrowdState crowd_;
  const std::uint64_t identity_;
  std::vector<double> positions_, drift_, masses_, score_response_, kinetic_response_, products_;
  std::vector<double> channel_coefficients_, channel_phases_, channel_logs_;
  std::vector<device_determinant::FactorizationMetadata> factors_;
  std::vector<double> cusp_, orbital_backflow_, orbital_envelope_, orbital_seeds_;
  std::vector<double> tile_weights_;
  const std::vector<double> weights_{1.0};
  const std::function<void()>* action_ = nullptr;
  std::exception_ptr failure_;
  PsiFormerAcceleratorOperation operation_ = PsiFormerAcceleratorOperation::PARAMETER_SCORE;
  std::size_t resource_generation_ = 0, acquisition_generation_ = 0;
  std::size_t coordinate_generation_ = 0, result_generation_ = 0, numerical_products_ = 0;
  std::size_t request_count_ = 1;
  std::atomic<std::size_t> weighted_score_tiles_{0}, weighted_score_configurations_{0};
  bool recorded_ = false, ready_ = false;
};

} // namespace qmcplusplus::psiformer::device

#endif
