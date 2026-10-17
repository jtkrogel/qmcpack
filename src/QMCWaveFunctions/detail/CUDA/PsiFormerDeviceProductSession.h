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
#include "PsiFormerDeviceAdam.h"
#include "PsiFormerDeviceKineticReverse.h"
#include "PsiFormerDeviceExecutionPreparation.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorExecutor.h"

#include <exception>
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>

namespace qmcplusplus::psiformer::device
{
/// Ordered ENERGY channels: local-energy response, energy-times-score, then score.
enum class ScoreContractionRole { GENERIC, ENERGY_T_EO_O };


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
  friend class PreparedDeviceSrMetric;
  friend class PreparedDevicePcg;
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

  /// Optional physical constants/scalars only; all FULL and P arenas are reused.
  struct ResidentOpenWorkspace
  {
    std::vector<double> masses, charges;
    DeviceBuffer<double> device_masses, device_charges, energy;
    DeviceBuffer<std::uint64_t> status;
    std::array<double, 5> host_energy{};
    std::uint64_t host_status = 0;
    std::shared_ptr<const std::size_t> token = std::make_shared<const std::size_t>(0);
    std::size_t epoch = 0;
    ResidentOpenWorkspace(const std::vector<double>& m, const std::vector<double>& z)
        : masses(m), charges(z), device_masses(m.size()), device_charges(z.size()), energy(5), status(1)
    { device_masses.upload(masses.data(), masses.size()); device_charges.upload(charges.data(), charges.size()); }
  };

public:
  struct ResidentOpenEnergyCost { std::size_t device_bytes, host_bytes; };

  /// Scalar observation pins provenance, not the borrowed session/backend lifetime.
  struct ResidentOpenEnergyObservation
  {
    std::array<double, 5> energy{}; // K, ee, ei, ii, total
  private:
    friend class PreparedDeviceProductSession;
    PackedDeviceCoordinatesView coordinates;
    std::shared_ptr<const PreparedDeviceModel> image;
    std::shared_ptr<const std::size_t> token;
    std::size_t epoch = 0;
  };

  /// Numeric arrays plus one revocation token; metadata/vendor/caller receipts excluded.
  static ResidentOpenEnergyCost estimateResidentOpenEnergyCost(std::size_t electrons, std::size_t ions)
  {
    if (!electrons || !ions) throw std::invalid_argument("Resident open energy requires electrons and ions");
    const auto elements = spatial_detail::checkedAdd(spatial_detail::checkedAdd(electrons, ions,
        "Resident open energy extent overflow"), 6, "Resident open energy extent overflow");
    const auto bytes = checkedDeviceBytes<double>(elements);
    return {bytes, spatial_detail::checkedAdd(bytes, sizeof(std::size_t), "Resident open token overflow")};
  }

  /// Cold numerical preparation, deliberately not an arbitrary native Hamiltonian claim.
  void prepareResidentOpenEnergy(const std::vector<double>& inverse_masses,
      const std::vector<double>& charges, std::size_t device_cap, std::size_t host_cap)
  {
    std::lock_guard lock(mutex_);
    validateContractionControl(active_image_->version());
    if (resident_open_ || !kinetic_ || target_kinetic_ || image_->periodicCell() ||
        image_->plan().environment().amplitude_scalar_domain != ScalarDomain::REAL ||
        inverse_masses.size() != image_->plan().modelShape().electrons() ||
        charges.size() != image_->plan().modelShape().nuclei)
      throw std::invalid_argument("Resident open energy requires an unprepared real/open full-kinetic lane");
    for (auto value : inverse_masses)
      if (!device_math::isFiniteBinary64(value) || value <= 0)
        throw std::invalid_argument("Resident inverse masses must be finite and positive");
    for (auto value : charges)
      if (!device_math::isFiniteBinary64(value)) throw std::invalid_argument("Resident ion charge is nonfinite");
    const auto cost = estimateResidentOpenEnergyCost(inverse_masses.size(), charges.size());
    if (cost.device_bytes > device_cap || cost.host_bytes > host_cap)
      throw std::length_error("Resident open energy cold numeric capacity exceeded");
    auto prepared = std::make_unique<ResidentOpenWorkspace>(inverse_masses, charges);
    const auto actual_host = estimateResidentOpenEnergyCost(
        prepared->masses.capacity(), prepared->charges.capacity()).host_bytes;
    if (actual_host > host_cap) throw std::length_error("Resident open energy actual HOST capacity exceeded");
    resident_open_host_bytes_ = actual_host;
    resident_open_ = std::move(prepared);
  }

  /// Observe one checked DEVICE coordinate generation using the existing FULL arena.
  ResidentOpenEnergyObservation observeResidentOpenEnergy(const PackedDeviceCoordinatesView& coordinates,
      std::size_t version)
  {
    ResidentOpenEnergyObservation result;
    execute(PsiFormerAcceleratorOperation::KINETIC_RESPONSE, nullptr, version, [&] {
      if (!resident_open_ || !resident_open_->token)
        throw std::logic_error("Resident open energy was not prepared or its generation is exhausted");
      kinetic_->spatial.evaluateDeviceCoordinates(coordinates);
      const auto head = kinetic_->spatial.spatialHeadTape();
      checkRuntime(launchResidentOpenEnergy(queue_.getNative(), head.root_layout, head.root, head.positions,
          image_->nuclei(), resident_open_->charges.size(), resident_open_->device_masses.data(),
          resident_open_->device_charges.data(), resident_open_->energy.data(), resident_open_->status.data()),
          "resident open energy");
      queue_.sync();
      resident_open_->status.download(&resident_open_->host_status, 1);
      if (resident_open_->host_status != static_cast<std::uint64_t>(OpenKineticReverseStatus::REGULAR))
        throw std::runtime_error("Resident open energy is nonfinite or coalescent");
      resident_open_->energy.download(resident_open_->host_energy.data(), resident_open_->host_energy.size());
      result.energy = resident_open_->host_energy;
      result.coordinates = coordinates;
      result.image = image_;
      result.token = resident_open_->token;
      result.epoch = resident_open_->epoch;
    }, 1, &coordinates);
    return result;
  }

  /// Recompute FULL before dK: an earlier sample's arena is never assumed retained.
  void accumulateResidentKineticContraction(const ResidentOpenEnergyObservation& observation,
      const std::complex<double>* score_coefficients, const std::complex<double>* kinetic_coefficients,
      std::size_t channels, std::size_t version)
  {
    execute(PsiFormerAcceleratorOperation::KINETIC_RESPONSE, nullptr, version, [&] {
      if (!resident_open_ || observation.token != resident_open_->token || observation.image != image_ ||
          observation.epoch != resident_open_->epoch || !score_contraction_ ||
          !score_coefficients || !kinetic_coefficients || !channels || channels > 3)
        throw std::invalid_argument("Resident kinetic observation or channels are stale/foreign");
      for (const auto* coefficients : {score_coefficients, kinetic_coefficients})
        for (std::size_t c = 0; c < channels; ++c)
          if (!device_math::isFiniteBinary64(coefficients[c].real()) ||
              !device_math::isFiniteBinary64(coefficients[c].imag()))
            throw std::invalid_argument("Resident kinetic coefficient is nonfinite");
      // copyTo checks source lifetime/epoch before either score or dK can be folded.
      kinetic_->spatial.evaluateDeviceCoordinates(observation.coordinates);
      values_.evaluateDeviceCoordinates(observation.coordinates);
      score_contraction_->accumulate(score_.reverse(values_, weights_), score_coefficients, channels);
      score_contraction_->accumulate(kinetic_->reverse.evaluate(kinetic_->spatial, weights_,
          resident_open_->masses, {}), kinetic_coefficients, channels);
    }, 1, &observation.coordinates);
  }
  /** Pin completed numeric provenance, not session lifetime. The outer owner and
   * exclusive provider lease must retain this session/backend through all reads.
   */
  struct CompletedScoreContraction
  {
  private:
    friend class PreparedDeviceProductSession;
    const PreparedDeviceProductSession* owner = nullptr;
    std::shared_ptr<const PreparedDeviceModel> image;
    std::size_t generation = 0, channels = 0;
  };

  /// Allocate scratch once while the owning backend's control mutex is held.
  /// Product graphs are FP64: no mixed adapter is attached, and the backend rejects mixed preparation.
  PreparedDeviceProductSession(const std::shared_ptr<const PreparedDeviceModel>& active_image,
                               PsiFormerAcceleratorSharedModel& owner,
                               std::mutex& control_mutex, std::uint64_t identity,
                               ProductSessionProfile profile = ProductSessionProfile::SCORE_AND_KINETIC,
                               std::size_t score_capacity = 1,
                               DeviceExecutionPreparation preparation = {})
      : active_image_(active_image), image_(active_image), owner_(owner), mutex_(control_mutex),
        profile_(validatedProfile(profile)),
        score_capacity_(admittedScoreCapacity(image_->plan().modelShape(), score_capacity)),
        preparation_(validatedPreparation(image_->plan(), preparation)),
        blas_(queue_), values_(image_, score_capacity_, queue_, blas_, SpatialJetMode::VALUE_ONLY, true),
        score_(image_, score_capacity_, queue_, blas_), jvp_(image_, score_capacity_, queue_, blas_),
        kinetic_(profile_ == ProductSessionProfile::SCORE_AND_KINETIC && preparation_.kinetic.max_targets == 0
            ? std::make_unique<KineticWorkspace>(image_, queue_, blas_) : nullptr),
        target_kinetic_(profile_ == ProductSessionProfile::SCORE_AND_KINETIC && preparation_.kinetic.max_targets != 0
            ? std::make_unique<PreparedDeviceTargetKinetic>(image_, preparation_.kinetic, queue_, blas_) : nullptr),
        executor_(compiledBackend(), coverage(profile_)), crowd_(identity), identity_(identity),
        positions_(3 * image_->plan().modelShape().electrons()),
        drift_(positions_.size()), masses_(image_->plan().modelShape().electrons()),
        score_response_(image_->plan().parameterCount()),
        kinetic_response_(profile_ == ProductSessionProfile::SCORE_AND_KINETIC ? score_response_.size() : 0),
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
    products_.reserve(score_capacity_);
    executor_.prepare(crowd_, owner_.activeProvenance().layout_fingerprint, score_capacity_, *this);
  }

  // Prepared graphs borrow this address-stable session's queue and BLAS handle.
  PreparedDeviceProductSession(const PreparedDeviceProductSession&) = delete;
  PreparedDeviceProductSession& operator=(const PreparedDeviceProductSession&) = delete;

  /// Report the immutable capabilities selected before any numerical request.
  ProductSessionProfile profile() const noexcept { return profile_; }

  /// Immutable prepared capacity; capability reads need no lazy-owner access.
  std::size_t scoreCapacity() const noexcept { return score_capacity_; }

  /// Return the explicit per-provider coordinate limit; zero retains host input.
  std::size_t retainedCoordinateBudget() const noexcept
  { return preparation_.retained_coordinate_workspace_budget_bytes; }

  /// Prepare/upload the complete sample population once, even above tile capacity.
  /// The returned owner retains its metadata family, never an old P image.
  std::unique_ptr<PreparedDeviceCoordinates> prepareSampleCoordinates(const std::vector<double>& positions,
                                                                    std::size_t version)
  {
    std::lock_guard lock(mutex_);
    validateContractionControl(version);
    if (!retainedCoordinateBudget())
      throw std::logic_error("Retained device sample coordinates were not explicitly admitted");
    const auto stride = 3 * masses_.size();
    if (positions.empty() || positions.size() % stride != 0)
      throw std::invalid_argument("Retained sample coordinates have an incomplete population");
    auto source = image_->prepareCoordinates(positions.size() / stride, retainedCoordinateBudget());
    source->upload(positions);
    return source;
  }

  /// Identify the actually owned kinetic route without allocating or upgrading it.
  std::size_t kineticTargetCapacity() const noexcept
  { return target_kinetic_ ? preparation_.kinetic.max_targets : 0; }

  /// Snapshot device-wide counters, including other crowds and scalar status traffic.
  DeviceMemorySnapshot transferSnapshot() const
  {
    image_->validateCurrentDevice();
    int device = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&device), "product transfer metrics device");
#else
    checkRuntime(cudaGetDevice(&device), "product transfer metrics device");
#endif
    return deviceMemoryMetrics(device).snapshot();
  }

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

  /// Evaluate one retained sample without another coordinate host upload.
  const std::vector<double>& scoreDeviceCoordinates(const PackedDeviceCoordinatesView& positions,
                                                   std::size_t version)
  {
    execute(PsiFormerAcceleratorOperation::PARAMETER_SCORE, nullptr, version, [&]() {
      values_.evaluateDeviceCoordinates(positions);
      score_.reverse(values_, weights_);
      score_.downloadResponse(score_response_);
    }, 1, &positions);
    return score_response_;
  }

  /// Explicit cold preparation for the actual admitted score-channel count.
  /// Host/kinetic compatibility callers never need this optional O(CP) storage.
  void prepareScoreContractions(std::size_t channels, std::size_t version,
      std::size_t budget = std::numeric_limits<std::size_t>::max())
  {
    std::lock_guard lock(mutex_);
    invalidateScoreContraction();
    validateContractionControl(version);
    if (channels == 0 || channels > 3)
      throw std::invalid_argument("Device score contraction supports one to three channels");
    const auto capacity = std::max(channels, score_contraction_ ? score_contraction_->channelCapacity() : 0);
    if (PreparedDeviceScoreContraction::requiredBytes(image_->plan().parameterCount(), capacity) > budget)
      throw std::length_error("Device score contraction exceeds its explicit byte budget");
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
  std::size_t beginScoreContraction(std::size_t channels, std::size_t version,
      ScoreContractionRole role = ScoreContractionRole::GENERIC)
  {
    std::lock_guard lock(mutex_);
    invalidateScoreContraction(false);
    validateContractionControl(version);
    if (!score_contraction_)
      throw std::logic_error("Device score contraction was not prepared");
    if (contraction_transaction_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Device score contraction transaction generation exhausted");
    if (role != ScoreContractionRole::GENERIC &&
        (role != ScoreContractionRole::ENERGY_T_EO_O || channels != 3 || image_->periodicCell() ||
         image_->plan().environment().amplitude_scalar_domain != ScalarDomain::REAL))
      throw std::invalid_argument("Device ENERGY contraction requires ordered real/open C3 channels");
    score_contraction_->begin(channels);
    energy_image_ = role == ScoreContractionRole::ENERGY_T_EO_O ? image_.get() : nullptr;
    return ++contraction_transaction_;
  }

  /// Consume only this completed transaction; no device view escapes the owner lock.
  double consumeEnergyAdam(PreparedDeviceAdam& adam, std::size_t version, std::size_t generation,
      const std::vector<double>& parameters, double weight, double energy, double beta1, double beta2,
      double correction1, double correction2, double epsilon, std::vector<double>& candidate)
  {
    std::lock_guard lock(mutex_);
    try
    {
      validateContractionControl(version);
      if (energy_image_ != image_.get() || generation != contraction_transaction_ || !score_contraction_)
        throw std::invalid_argument("Device ENERGY contraction owner/image/generation is stale or untagged");
      const auto norm = adam.proposeEnergy(*score_contraction_, image_->parameters(), parameters,
          weight, energy, beta1, beta2, correction1, correction2, epsilon, candidate);
      invalidateScoreContraction();
      return norm;
    }
    catch (...)
    {
      adam.reject();
      invalidateScoreContraction();
      throw;
    }
  }

  /// Finish ENERGY into owning DEVICE bytes; this does not publish a model or native mirror.
  PreparedDeviceAdam::DeviceEnergyCandidate consumeEnergyAdamDeviceCandidate(PreparedDeviceAdam& adam,
      std::size_t version, std::size_t generation, double weight, double energy, double beta1, double beta2,
      double correction1, double correction2, double epsilon, std::size_t candidate_device_budget,
      bool with_displacement_norm = false)
  {
    std::lock_guard lock(mutex_);
    try
    {
      validateContractionControl(version);
      if (energy_image_ != image_.get() || generation != contraction_transaction_ || !score_contraction_)
        throw std::invalid_argument("Device ENERGY candidate source/image/generation is stale or untagged");
      auto result = adam.proposeEnergyDeviceCandidate(*score_contraction_, image_, weight, energy,
          beta1, beta2, correction1, correction2, epsilon, candidate_device_budget, with_displacement_norm);
      invalidateScoreContraction();
      return result;
    }
    catch (...)
    {
      adam.reject();
      invalidateScoreContraction();
      throw;
    }
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

  /// Fold all algebraic score channels for one retained sample into existing sums.
  void accumulateScoreContractionDeviceCoordinates(const PackedDeviceCoordinatesView& positions,
      const std::complex<double>* coefficients, std::size_t channels, std::size_t version)
  {
    execute(PsiFormerAcceleratorOperation::PARAMETER_SCORE, nullptr, version, [&]() {
      if (!score_contraction_)
        throw std::logic_error("Device score contraction was not prepared");
      values_.evaluateDeviceCoordinates(positions);
      score_contraction_->accumulate(score_.reverse(values_, weights_), coefficients, channels);
    }, 1, &positions);
  }

  /// One batched forward/reverse contracts real sample weights directly to P.
  /// The outer transaction admits all tiles together; optional coefficients route
  /// the weighted response into an already active multi-channel ENERGY sum.
  void accumulateWeightedScoreTile(const double* positions, const double* weights,
                                    std::size_t count, std::size_t version,
                                    const std::complex<double>* channels = nullptr,
                                    std::size_t channel_count = 1)
  {
    execute(PsiFormerAcceleratorOperation::PARAMETER_SCORE, positions, version, [&]() {
      if (!weights || !score_contraction_ || (!channels && channel_count != 1))
        throw std::invalid_argument("Weighted score tile requires weights and a prepared contraction");
      tile_weights_.resize(count);
      std::copy_n(weights, count, tile_weights_.begin());
      // reverse() validates every finite signed weight and every determinant,
      // including configurations with zero weight. It returns a weighted sum.
      values_.evaluate(positions_);
      const double* response = score_.reverse(values_, tile_weights_);
      const std::complex<double> unit(1, 0);
      score_contraction_->accumulate(response, channels ? channels : &unit, channel_count);
    }, count);
    ++weighted_score_tiles_;
    weighted_score_configurations_ += count;
  }

  /// Contract signed real sample weights from one retained full/tail tile.
  void accumulateWeightedScoreTileDeviceCoordinates(const PackedDeviceCoordinatesView& positions,
                                                     const double* weights, std::size_t version)
  {
    const auto count = positions.batchCount();
    execute(PsiFormerAcceleratorOperation::PARAMETER_SCORE, nullptr, version, [&]() {
      if (!weights || !score_contraction_)
        throw std::invalid_argument("Weighted score tile requires weights and a prepared contraction");
      tile_weights_.resize(count);
      std::copy_n(weights, count, tile_weights_.begin());
      values_.evaluateDeviceCoordinates(positions);
      const double* response = score_.reverse(values_, tile_weights_);
      const std::complex<double> unit(1, 0);
      score_contraction_->accumulate(response, &unit, 1);
    }, count, &positions);
    ++weighted_score_tiles_;
    weighted_score_configurations_ += count;
  }

  /** Fold one sample's score and kinetic roots into the existing complex sums.
   * Coefficients already encode the caller's transpose/Hermitian convention.
   * Every device P response is consumed on this queue before its owner is reused;
   * no response-sized host transfer or additional P temporary is needed.
   * Any failed root invalidates the entire existing contraction transaction.
   */
  void accumulateKineticContraction(const double* positions, const double* total_drift,
      const double* imaginary_drift, const double* inverse_masses,
      const std::complex<double>* score_coefficients, const std::complex<double>* kinetic_coefficients,
      const std::complex<double>* imaginary_coefficients, std::size_t channels, std::size_t version)
  {
    execute(PsiFormerAcceleratorOperation::KINETIC_RESPONSE, positions, version, [&]() {
      requireKinetic();
      if (!total_drift || !imaginary_drift || !inverse_masses ||
          !score_coefficients || !kinetic_coefficients || !imaginary_coefficients)
        throw std::invalid_argument("Resident kinetic contraction requires complete physical inputs and coefficients");
      if (!score_contraction_ || channels == 0 || channels > 3)
        throw std::invalid_argument("Resident kinetic contraction requires prepared bounded channels");
      for (const auto* coefficients : {score_coefficients, kinetic_coefficients, imaginary_coefficients})
        for (std::size_t channel = 0; channel < channels; ++channel)
          if (!device_math::isFiniteBinary64(coefficients[channel].real()) ||
              !device_math::isFiniteBinary64(coefficients[channel].imag()))
            throw std::invalid_argument("Resident kinetic contraction coefficient is nonfinite");
      std::copy_n(total_drift, drift_.size(), drift_.begin());
      std::copy_n(inverse_masses, masses_.size(), masses_.begin());

      // Ordinary score is evaluated once, independently of kinetic target tiles.
      values_.evaluate(positions_);
      const double* response = score_.reverse(values_, weights_);
      score_contraction_->accumulate(response, score_coefficients, channels);
      if (target_kinetic_)
        response = target_kinetic_->evaluate(positions_, weights_, masses_, drift_);
      else
      {
        kinetic_->spatial.evaluate(positions_);
        response = kinetic_->reverse.evaluate(kinetic_->spatial, weights_, masses_, drift_);
      }
      score_contraction_->accumulate(response, kinetic_coefficients, channels);

      // The same physical drift convention supplies the optional imaginary cross
      // term. The fixed outer phase has no parameter-dependent Laplacian seed.
      const bool imaginary = imaginary_drift && std::any_of(imaginary_drift,
          imaginary_drift + drift_.size(), [](double value) { return value != 0.0; });
      if (imaginary)
      {
        std::copy_n(imaginary_drift, drift_.size(), drift_.begin());
        if (target_kinetic_)
          response = target_kinetic_->evaluateGradientResponse(positions_, weights_, masses_, drift_);
        else
          response = kinetic_->reverse.evaluateGradientResponse(kinetic_->spatial, weights_, masses_, drift_);
        score_contraction_->accumulate(response, imaginary_coefficients, channels);
      }
    });
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

  /// Expose only a completed, versioned canonical range to bounded HOST scratch.
  void downloadScoreContractionChunk(std::size_t channel, std::size_t offset,
      std::complex<double>* destination, std::size_t count, std::size_t version)
  {
    std::lock_guard lock(mutex_);
    try
    {
      validateContractionControl(version);
      if (!score_contraction_)
        throw std::logic_error("Device score contraction was not prepared");
      score_contraction_->downloadChunk(channel, offset, destination, count);
    }
    catch (...)
    {
      invalidateScoreContraction();
      throw;
    }
  }

  /// Freeze admitted output before releasing model authority for caller callbacks.
  CompletedScoreContraction completedScoreContraction(std::size_t version)
  {
    std::lock_guard lock(mutex_);
    validateContractionControl(version);
    if (!score_contraction_ || !score_contraction_->completedChannels())
      throw std::logic_error("Device score contraction has no completed generation");
    CompletedScoreContraction result;
    result.owner = this;
    result.image = image_;
    result.generation = contraction_transaction_;
    result.channels = score_contraction_->completedChannels();
    return result;
  }

  /// Read the frozen old generation without adopting a newly published image.
  void downloadScoreContractionChunk(const CompletedScoreContraction& receipt,
      std::size_t channel, std::size_t offset, std::complex<double>* destination, std::size_t count)
  {
    std::lock_guard lock(mutex_);
    if (receipt.owner != this || !receipt.image || receipt.image.get() != image_.get() ||
        receipt.generation != contraction_transaction_ || !score_contraction_ ||
        receipt.channels != score_contraction_->completedChannels() || !receipt.channels ||
        crowd_.acquired() || crowd_.poisoned())
      throw std::invalid_argument("Device score contraction receipt is foreign, stale or revoked");
    receipt.image->validateCurrentDevice();
    try
    {
      score_contraction_->downloadChunk(channel, offset, destination, count);
    }
    catch (...)
    {
      invalidateScoreContraction();
      throw;
    }
  }

  /// Revoke queued/partial sums; the next begin clears them in the same stream.
  void abortScoreContraction() noexcept
  {
    std::lock_guard lock(mutex_);
    invalidateScoreContraction();
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
      if (target_kinetic_)
      {
        target_kinetic_->evaluate(positions_, weights_, masses_, drift_);
        target_kinetic_->downloadResponse(kinetic_response_);
      }
      else
      {
        kinetic_->spatial.evaluate(positions_);
        kinetic_->reverse.evaluate(kinetic_->spatial, weights_, masses_, drift_);
        kinetic_->reverse.downloadResponse(kinetic_response_);
      }
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
      if (target_kinetic_)
      {
        target_kinetic_->evaluateGradientResponse(positions_, weights_, masses_, drift_);
        target_kinetic_->downloadResponse(kinetic_response_);
      }
      else
      {
        kinetic_->spatial.evaluate(positions_);
        kinetic_->reverse.evaluateGradientResponse(kinetic_->spatial, weights_, masses_, drift_);
        kinetic_->reverse.downloadResponse(kinetic_response_);
      }
    });
    return kinetic_response_;
  }

  /// Apply a real parameter direction without materializing a parameter score.
  double scoreJvp(const double* positions, const std::vector<double>& direction, std::size_t version)
  {
    execute(PsiFormerAcceleratorOperation::PARAMETER_SCORE, positions, version, [&]() {
      values_.evaluate(positions_);
      jvp_.evaluate(values_, direction);
      products_.resize(1);
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
    return scoreJvpPreparedTile(positions, 1, version)[0];
  }

  /// Apply one resident direction to a live tile, including a short tail.
  /// Consume the returned scalars immediately; the next request invalidates them.
  /// A failed tile exposes no partial products and keeps the prepared capacity.
  const std::vector<double>& scoreJvpPreparedTile(const double* positions, std::size_t count,
                                                 std::size_t version)
  {
    try
    {
      execute(PsiFormerAcceleratorOperation::PARAMETER_SCORE, positions, version, [&]() {
        values_.evaluate(positions_);
        jvp_.evaluatePrepared(values_);
        products_.resize(count);
        jvp_.downloadProducts(products_);
      }, count);
    }
    catch (...)
    {
      products_.clear();
      throw;
    }
    return products_;
  }

  /// Apply the resident direction to a retained tile, publishing no failed tail.
  const std::vector<double>& scoreJvpPreparedTileDeviceCoordinates(const PackedDeviceCoordinatesView& positions,
                                                                  std::size_t version)
  {
    try
    {
      execute(PsiFormerAcceleratorOperation::PARAMETER_SCORE, nullptr, version, [&]() {
        values_.evaluateDeviceCoordinates(positions);
        jvp_.evaluatePrepared(values_);
        products_.resize(positions.batchCount());
        jvp_.downloadProducts(products_);
      }, positions.batchCount(), &positions);
    }
    catch (...)
    {
      products_.clear();
      throw;
    }
    return products_;
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
    if (target_kinetic_)
      bytes = spatial_detail::checkedAdd(bytes, target_kinetic_->deviceWorkspaceBytes(),
          "Device product target kinetic workspace overflow");
    bytes = spatial_detail::checkedAdd(bytes, contraction_device_bytes_.load(std::memory_order_acquire),
                                       "Device product contraction byte overflow");
    if (resident_open_)
      bytes = spatial_detail::checkedAdd(bytes, estimateResidentOpenEnergyCost(
          resident_open_->masses.size(), resident_open_->charges.size()).device_bytes, "Resident device overflow");
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
    if (target_kinetic_)
      bytes = spatial_detail::checkedAdd(bytes, target_kinetic_->hostWorkspaceBytes(),
          "Device product target kinetic host status overflow");
    bytes = spatial_detail::checkedAdd(bytes,
        checkedDeviceBytes<device_determinant::FactorizationMetadata>(factors_.capacity()),
        "Device product channel factorization overflow");
    bytes = spatial_detail::checkedAdd(bytes, contraction_host_bytes_.load(std::memory_order_acquire),
                                       "Device product contraction host byte overflow");
    bytes = spatial_detail::checkedAdd(bytes, resident_open_host_bytes_, "Resident HOST overflow");
    return bytes;
  }

private:
  /// Validate explicit per-owner settings before any numerical graph allocation.
  static DeviceExecutionPreparation validatedPreparation(const PsiFormerExecutionPlan& plan,
                                                           DeviceExecutionPreparation preparation)
  {
    validateDeviceExecutionPreparation(plan, preparation);
    return preparation;
  }

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
    if (!kinetic_ && !target_kinetic_)
      throw std::logic_error("Device product session was prepared for score products only");
  }

  /// A failed/rebound lane can never publish an earlier partial or completed sum.
  void invalidateScoreContraction(bool revoke_observations = true) noexcept
  {
    if (revoke_observations && resident_open_)
    {
      if (resident_open_->epoch == std::numeric_limits<std::size_t>::max()) resident_open_->token.reset();
      else ++resident_open_->epoch;
    }
    energy_image_ = nullptr;
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
    if (target_kinetic_)
      target_kinetic_->rebindModel(active_image_);
    image_ = active_image_;
    ready_ = false;
  }

  /// Preserve provenance and failure recovery around one synchronous product.
  void execute(PsiFormerAcceleratorOperation operation, const double* positions,
                std::size_t version, const std::function<void()>& action, std::size_t count = 1,
                const PackedDeviceCoordinatesView* device_input = nullptr)
  {
    std::lock_guard lock(mutex_);
    executeUnderControl(operation, positions, version, action, count, device_input);
  }

  /// Preserve the same product protocol while a scoped numerical owner holds mutex_.
  void executeUnderControl(PsiFormerAcceleratorOperation operation, const double* positions,
      std::size_t version, const std::function<void()>& action, std::size_t count = 1,
      const PackedDeviceCoordinatesView* device_input = nullptr)
  {
    ready_ = false;
    if ((!positions && !device_input) || version != active_image_->version() || count == 0 || count > score_capacity_ ||
        (device_input && (device_input->batchCount() != count || device_input->physicalElectrons() != masses_.size())))
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
    // Typed actions borrow their view synchronously; no source pointer survives
    // this call. Host callers retain their original bounded staging path.
    if (!device_input)
    {
      positions_.resize(count * 3 * masses_.size());
      std::copy_n(positions, positions_.size(), positions_.begin());
    }
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
  const DeviceExecutionPreparation preparation_;
  compute::Queue<PlatformKind::CUDA> queue_;
  AcceleratorBlasHandle blas_;
  PreparedDeviceInference values_;
  PreparedDeviceScoreReverse score_;
  PreparedDeviceScoreJvp jvp_;
  std::unique_ptr<KineticWorkspace> kinetic_;
  std::unique_ptr<ResidentOpenWorkspace> resident_open_;
  std::size_t resident_open_host_bytes_ = 0;
  std::unique_ptr<PreparedDeviceTargetKinetic> target_kinetic_;
  std::unique_ptr<PreparedDeviceScoreContraction> score_contraction_;
  const PreparedDeviceModel* energy_image_ = nullptr;
  std::size_t contraction_transaction_ = 0;
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
