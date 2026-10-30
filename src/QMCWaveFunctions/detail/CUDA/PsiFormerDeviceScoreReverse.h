//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceScoreReverse.h
 * @brief Bounded-tile canonical weighted log-wavefunction parameter reverse.
 *
 * Contracts sum_s weight_s d log|Psi_s|/dp without retaining a B-by-P Jacobian.
 * Supports immutable open-real models with regular determinant channels. Exact
 * total nodes and singular channels fail explicitly: a cofactor fallback remains
 * necessary before claiming general singular-channel score coverage. This
 * numerical evaluator is not yet a production training/provider capability.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_SCORE_REVERSE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_SCORE_REVERSE_H

#include "PsiFormerDeviceNetworkReverse.h"

namespace qmcplusplus::psiformer::device
{

/** Own reusable head adjoints, geometry and canonical response destinations. */
class PreparedDeviceScoreReverse
{
public:
  /// Preallocate one tile and O(P) accumulators with the same immutable model image.
  PreparedDeviceScoreReverse(std::shared_ptr<const PreparedDeviceModel> model,
                             std::size_t capacity,
                             compute::Queue<PlatformKind::CUDA>& queue,
                             AcceleratorBlasHandle& blas,
                             std::shared_ptr<PreparedDeviceMixedAffineDerivatives> mixed = {})
      : model_(std::move(model)), capacity_(capacity), queue_(queue), blas_(blas),
        network_(model_, capacity, queue, blas, NetworkReverseOutput::PARAMETER_SCORE, std::move(mixed))
  {
    const auto& shape = model_->plan().modelShape();
    const auto ne = shape.electrons();
    const auto matrices = multiply(multiply(multiply(capacity_, shape.determinants), ne), ne);
    seeds_.reserveNamed(matrices, "ScoreReverse::seeds_", __FILE__, __LINE__);
    up_.reserveNamed(matrices, "ScoreReverse::up_", __FILE__, __LINE__);
    down_.reserveNamed(matrices, "ScoreReverse::down_", __FILE__, __LINE__);
    envelope_.reserveNamed(matrices, "ScoreReverse::envelope_", __FILE__, __LINE__);
    feature_.reserveNamed(multiply(multiply(capacity_, ne), shape.feature_dimension),
        "ScoreReverse::feature_", __FILE__, __LINE__);
    ion_distances_.reserveNamed(multiply(multiply(capacity_, ne), shape.nuclei),
        "ScoreReverse::ion_distances_", __FILE__, __LINE__);
    pair_count_ = multiply(ne, ne - 1) / 2;
    pair_distances_.reserveNamed(multiply(capacity_, pair_count_), "ScoreReverse::pair_distances_", __FILE__, __LINE__);
    same_spin_.reserveNamed(pair_count_, "ScoreReverse::same_spin_", __FILE__, __LINE__);
    weights_.reserveNamed(capacity_, "ScoreReverse::weights_", __FILE__, __LINE__);
    channel_weights_.reserveNamed(multiply(capacity_, shape.determinants),
        "ScoreReverse::channel_weights_", __FILE__, __LINE__);
    unit_weights_.resize(capacity_, 1.0);
    cusp_weights_.resize(capacity_);
    head_response_.reserveNamed(model_->plan().parameterCount(), "ScoreReverse::head_response_", __FILE__, __LINE__);
    response_.reserveNamed(model_->plan().parameterCount(), "ScoreReverse::response_", __FILE__, __LINE__);
    status_.reserveNamed(capacity_, "ScoreReverse::status_", __FILE__, __LINE__);
    host_status_.resize(capacity_);
    host_factors_.resize(multiply(capacity_, shape.determinants));
    finite_layout_ = {response_.capacity(), 4096,
        response_.capacity() / 4096 + (response_.capacity() % 4096 != 0), 0, 1};
    finite_status_.reserveNamed(finite_layout_.status_capacity, "ScoreReverse::finite_status_", __FILE__, __LINE__);
    host_finite_status_.resize(finite_layout_.status_capacity);
  }

  std::size_t sensitiveFp64AffineProjections() const noexcept { return network_.sensitiveFp64AffineProjections(); }

  /** Reverse one completed value tape into a canonical weighted score vector.
   * Weights are arbitrary finite signed contractions, not implicitly normalized
   * probabilities. Sample/model version and coordinates remain owned by forward.
   */
  const double* reverse(const PreparedDeviceInference& forward, const std::vector<double>& weights)
  {
    return reverseSeeded(forward, weights, nullptr, nullptr, nullptr);
  }

  /// Reverse a retained tape using the same-request FP64 determinant linearization and ordinary physical cusp.
  const double* reverseCanonicalHead(const PreparedDeviceInference& forward, const RealCanonicalHeadView& head,
      const std::vector<double>& weights)
  {
    return reverseSeeded(forward, weights, nullptr, nullptr, nullptr, nullptr, &head);
  }

  /// Curvature observes one unweighted real/open sample before scratch reuse.
  const double* observeAffine(const PreparedDeviceInference& forward, DeviceAffineObservationSink& affine)
  {
    ready_ = false;
    if (forward.valueTapeBatchCount() != 1 || model_->plan().environment().periodic_profile ||
        model_->plan().environment().amplitude_scalar_domain != ScalarDomain::REAL)
      throw std::invalid_argument("Device affine score observations require one real open sample");
    unit_weights_.resize(1);
    unit_weights_[0] = 1.0;
    return reverseSeeded(forward, unit_weights_, nullptr, nullptr, nullptr, &affine);
  }

  /// Feed KFAC affine observations from the same canonical-head score seeds as the gradient.
  const double* observeAffineCanonicalHead(const PreparedDeviceInference& forward, const RealCanonicalHeadView& head,
      DeviceAffineObservationSink& affine)
  {
    ready_ = false;
    if (forward.valueTapeBatchCount() != 1 || model_->plan().environment().periodic_profile ||
        model_->plan().environment().amplitude_scalar_domain != ScalarDomain::REAL)
      throw std::invalid_argument("Canonical-head affine observations require one ordinary REAL sample");
    unit_weights_.resize(1);
    unit_weights_[0] = 1.0;
    return reverseSeeded(forward, unit_weights_, nullptr, nullptr, nullptr, &affine, &head);
  }

  /// Reverse independent determinant log amplitudes, rather than log of their sum.
  const double* reverseChannels(const PreparedDeviceInference& forward, const std::vector<double>& coefficients)
  {
    ready_ = false;
    const auto count = forward.valueTapeBatchCount();
    const auto determinants = model_->plan().modelShape().determinants;
    if (count > capacity_ || coefficients.size() != multiply(count, determinants))
      throw std::invalid_argument("Independent channel coefficient extent mismatch");
    unit_weights_.resize(count, 1.0);
    cusp_weights_.resize(count);
    for (std::size_t configuration = 0; configuration < count; ++configuration)
    {
      double sum = 0;
      for (std::size_t determinant = 0; determinant < determinants; ++determinant)
      {
        const auto coefficient = coefficients[configuration * determinants + determinant];
        if (!device_math::isFiniteBinary64(coefficient))
          throw std::invalid_argument("Independent channel coefficient is nonfinite");
        sum += coefficient;
      }
      if (!device_math::isFiniteBinary64(sum))
        throw std::overflow_error("Independent channel cusp coefficient overflow");
      cusp_weights_[configuration] = sum;
    }
    return reverseSeeded(forward, unit_weights_, &coefficients, nullptr, &cusp_weights_);
  }

  /// Reverse raw orbital cotangents without determinant inverses or cusp derivatives.
  const double* reverseOrbitals(const PreparedDeviceInference& forward, const std::vector<double>& adjoints)
  {
    ready_ = false;
    const auto count = forward.valueTapeBatchCount();
    const auto& shape = model_->plan().modelShape();
    const auto extent = multiply(multiply(count, shape.determinants), multiply(shape.electrons(), shape.electrons()));
    if (count > capacity_ || adjoints.size() != extent)
      throw std::invalid_argument("Raw orbital cotangent extent mismatch");
    for (double value : adjoints)
      if (!device_math::isFiniteBinary64(value))
        throw std::invalid_argument("Raw orbital cotangent is nonfinite");
    unit_weights_.resize(count, 1.0);
    cusp_weights_.assign(count, 0.0);
    return reverseSeeded(forward, unit_weights_, nullptr, &adjoints, &cusp_weights_);
  }

private:
  /// Share the head/trunk reverse while preserving each objective's seed rules.
  const double* reverseSeeded(const PreparedDeviceInference& forward, const std::vector<double>& weights,
      const std::vector<double>* channels, const std::vector<double>* raw_seeds,
      const std::vector<double>* cusp_weights, DeviceAffineObservationSink* affine = nullptr,
      const RealCanonicalHeadView* canonical_head = nullptr)
  {
    ready_ = false;
    model_->validateCurrentDevice();
    const auto count = forward.valueTapeBatchCount();
    const auto& shape = model_->plan().modelShape();
    const auto ne = shape.electrons(), determinants = shape.determinants, width = shape.feature_dimension;
    if (forward.modelImage().get() != model_.get() || count > capacity_ || weights.size() != count)
      throw std::invalid_argument("Score reverse model, capacity or weight extent mismatch");
    for (double weight : weights)
      if (!device_math::isFiniteBinary64(weight))
        throw std::invalid_argument("Score reverse weight is nonfinite");

    if (canonical_head)
    {
      canonical_head->validate(model_, model_->version(), &queue_, count, ne, determinants);
      if (!forward.evaluationCoordinates().sameSourceRange(canonical_head->coordinates()))
        throw std::invalid_argument("Canonical score head and retained tape consumed different coordinate intervals");
    }
    // Inverse seeds cannot represent singular-channel cofactor derivatives even
    // when the determinant sum is nonzero. Reject rather than silently drop them.
    if (!raw_seeds && !canonical_head)
    {
      forward.downloadValueFactorizations(host_factors_);
      for (std::size_t channel = 0; channel < count * determinants; ++channel)
        if ((!channels || (*channels)[channel] != 0.0) && !host_factors_[channel].inverse_available)
          throw std::runtime_error("GPU score requires regular selected determinant channels; cofactor fallback unavailable");
    }
    const auto head = canonical_head ? forward.valueHeadTape(*canonical_head) : forward.valueHeadTape();
    const auto stream = queue_.getNative();
    const OrbitalProductReverseLayout orbital{count, determinants, ne, ne, ne * ne, determinants * ne * ne};
    if (raw_seeds)
      seeds_.upload(raw_seeds->data(), raw_seeds->size());
    else if (channels)
    {
      channel_weights_.upload(channels->data(), channels->size());
      checkRuntime(launchIndependentChannelSeeds(stream, orbital, channel_weights_.data(),
          canonical_head ? canonical_head->inverses() : head.inverses, seeds_.data()), "independent determinant seeds");
    }
    else
    {
      checkRuntime(launchDeterminantMatrixReverseSeeds(stream,
          canonical_head ? canonical_head->factorizations() : head.factorizations,
          canonical_head ? canonical_head->combinations() : head.combinations,
          canonical_head ? canonical_head->normalizedWeights() : head.normalized_weights,
          canonical_head ? canonical_head->inverses() : head.inverses, count, determinants, ne, seeds_.data(), status_.data()),
          "score determinant seeds");
      queue_.sync();
      status_.download(host_status_.data(), count);
      for (std::size_t configuration = 0; configuration < count; ++configuration)
        if (host_status_[configuration] != device_determinant::DerivativeStatus::AVAILABLE)
          throw std::runtime_error("GPU log score is undefined or nonfinite for this configuration");
    }

    // Output head: weighted matrix cotangents, spin-row packing, geometry and
    // orbital/envelope parameter contractions. No division by B or E is used.
    weights_.upload(weights.data(), count);
    checkRuntime(launchWeightedOrbitalProductReverse(stream, orbital, shape.spin_up_electrons,
        weights_.data(), head.backflow, head.envelope, seeds_.data(), up_.data(), down_.data(), envelope_.data()),
        "score orbital product reverse");
    checkRuntime(launchScoreGeometry(stream, count, ne, shape.nuclei, shape.spin_up_electrons,
        head.positions, model_->nuclei(), ion_distances_.data(), pair_distances_.data(), same_spin_.data(),
        model_->periodicCell()),
        "score geometry");
    zero(head_response_, head_response_.capacity());
    zero(feature_, multiply(multiply(count, ne), width));
    const DenseForwardLayout backflow{count * ne, width, determinants * ne, width,
                                      determinants * ne, determinants * ne};
    for (const auto& spin : {std::pair{ParameterRole::BACKFLOW_UP_WEIGHT, &up_},
                            std::pair{ParameterRole::BACKFLOW_DOWN_WEIGHT, &down_}})
    {
      const auto offset = model_->plan().parameter(spin.first).begin;
      accumulateDenseWeightAdjoint(blas_, backflow, forward.valueFinalFeatures(), spin.second->data(),
                                   head_response_.data() + offset);
      if (affine)
      {
        const auto begin = spin.first == ParameterRole::BACKFLOW_UP_WEIGHT ? 0 : shape.spin_up_electrons;
        const auto rows = spin.first == ParameterRole::BACKFLOW_UP_WEIGHT ? shape.spin_up_electrons :
            shape.spin_down_electrons;
        if (rows)
          affine->observe(spin.first, NO_ATTENTION_BLOCK, forward.valueFinalFeatures() + begin * width,
              spin.second->data() + begin * determinants * ne, rows, width, determinants * ne);
      }
      accumulateDenseInputAdjoint(blas_, backflow, spin.second->data(), model_->parameters() + offset,
                                  feature_.data());
    }
    envelopeReverse(count, 0, shape.spin_up_electrons,
                    ParameterRole::ENVELOPE_PI_UP, ParameterRole::ENVELOPE_ZETA_UP);
    envelopeReverse(count, shape.spin_up_electrons, ne,
                    ParameterRole::ENVELOPE_PI_DOWN, ParameterRole::ENVELOPE_ZETA_DOWN);
    if (pair_count_ != 0)
    {
      if (cusp_weights)
        weights_.upload(cusp_weights->data(), count);
      const auto* same = model_->plan().optionalParameter(ParameterRole::CUSP_SAME_ALPHA);
      const CuspAlphaReverseLayout cusp{count, pair_count_, pair_count_, same ? same->begin : 0,
          model_->plan().parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin, same != nullptr};
      checkRuntime(launchCuspAlphaReverse(stream, cusp, pair_distances_.data(), same_spin_.data(),
          model_->sameAlpha(), model_->oppositeAlpha(), head_response_.data(), weights_.data()), "score cusp reverse");
    }

    // Remaining trunk and feature layer: one device cotangent drives the explicit
    // shared-weight reverse. Add independent head/trunk intervals canonically.
    // Same-stream composition needs no intermediate HOST status round trip.
    // Any nonfinite trunk element remains nonfinite after adding the head,
    // including opposite infinities, so the final scan admits both together.
    const double* trunk = network_.reverseUnadmitted(forward, feature_.data(), affine);
    checkRuntime(launchAddParameterResponses(stream, response_.capacity(), trunk,
        head_response_.data(), response_.data()), "score canonical response sum");
    checkRuntime(launchDerivativeFiniteScan(stream, finite_layout_, response_.data(), finite_status_.data()),
                 "score finite scan");
    queue_.sync();
    finite_status_.download(host_finite_status_.data(), host_finite_status_.size());
    for (const auto& status : host_finite_status_)
      if (status.nonfinite_count != 0)
        throw std::runtime_error("GPU canonical score response is nonfinite");
    ready_ = true;
    return response_.data();
  }

public:
  /// Download a completed canonical response, rejecting stale results after failure.
  void downloadResponse(std::vector<double>& destination) const
  {
    if (!ready_)
      throw std::logic_error("GPU score has no completed response");
    if (destination.size() != response_.capacity())
      throw std::invalid_argument("GPU score destination extent mismatch");
    response_.download(destination.data(), destination.size());
  }

  /** Stream a completed canonical interval without replaying the reverse graph.
   * Caller supplies bounded host storage; active-variable mapping remains the
   * provider adapter's responsibility. Empty intervals permit a null destination.
   */
  void downloadResponseChunk(std::size_t begin, double* destination, std::size_t count) const
  {
    if (!ready_)
      throw std::logic_error("GPU score has no completed response");
    if (begin > response_.capacity() || count > response_.capacity() - begin)
      throw std::out_of_range("GPU score chunk exceeds the canonical response");
    response_.download(destination, count, begin);
  }

  /// Tag response chunks with the immutable model version used for this evaluator.
  std::size_t parameterVersion() const noexcept { return model_->version(); }

  /// Rebind head/trunk to one compatible successor and invalidate prior responses.
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> successor)
  {
    model_->validateSuccessor(successor);
    network_.rebindModel(successor);
    model_ = std::move(successor);
    ready_ = false;
  }

  /// Preflight the head and trunk while retaining their accepted predecessor image.
  void validateModelRebindWithMixedParameters(const std::shared_ptr<const PreparedDeviceModel>& successor,
      const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& incoming) const
  {
    model_->validateSuccessor(successor);
    network_.validateModelRebindWithMixedParameters(successor, incoming);
  }

  /// All validation and predecessor lifetime authority remain with the native transaction.
  void adoptPreparedModelWithMixedParameters(const std::shared_ptr<const PreparedDeviceModel>& successor,
      const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& incoming) noexcept
  {
    network_.adoptPreparedModelWithMixedParameters(successor, incoming);
    model_ = successor;
    ready_ = false;
  }

  /// Count host admission/status storage, including the shared trunk reverse.
  std::size_t hostWorkspaceBytes() const
  {
    return network_.hostWorkspaceBytes() + checkedDeviceBytes<double>(unit_weights_.capacity()) +
        checkedDeviceBytes<double>(cusp_weights_.capacity()) +
        checkedDeviceBytes<device_determinant::DerivativeStatus>(host_status_.capacity()) +
        checkedDeviceBytes<device_determinant::FactorizationMetadata>(host_factors_.capacity()) +
        checkedDeviceBytes<DerivativeFiniteStatus>(host_finite_status_.capacity());
  }

  /// Account explicitly owned device storage, excluding the shared model image.
  std::size_t deviceWorkspaceBytes() const
  {
    std::size_t bytes = network_.deviceWorkspaceBytes();
    const auto add = [&](std::size_t amount) {
      bytes = spatial_detail::checkedAdd(bytes, amount, "Score workspace storage overflow");
    };
    for (const auto* buffer : {&seeds_, &up_, &down_, &envelope_, &feature_, &ion_distances_,
                              &pair_distances_, &weights_, &channel_weights_, &head_response_, &response_})
      add(checkedDeviceBytes<double>(buffer->capacity()));
    add(checkedDeviceBytes<std::uint8_t>(same_spin_.capacity()));
    add(checkedDeviceBytes<device_determinant::DerivativeStatus>(status_.capacity()));
    add(checkedDeviceBytes<DerivativeFiniteStatus>(finite_status_.capacity()));
    return bytes;
  }

private:
  /// Check storage extents once at preparation boundaries.
  static std::size_t multiply(std::size_t left, std::size_t right)
  { return spatial_detail::checkedProduct(left, right, "Score reverse extent overflow"); }

  /// Zero a prepared contiguous destination without allocating hot-path buffers.
  void zero(DeviceBuffer<double>& buffer, std::size_t count)
  { checkRuntime(launchDerivativeZero(queue_.getNative(), {count, 0, 1, 0, 1}, buffer.data()), "score zero"); }

  /// Reduce one spin-row interval into canonical envelope pi/zeta parameters.
  void envelopeReverse(std::size_t count, std::size_t begin, std::size_t end,
                       ParameterRole pi_role, ParameterRole zeta_role)
  {
    if (begin == end)
      return;
    const auto& shape = model_->plan().modelShape();
    const auto ne = shape.electrons(), nuclei = shape.nuclei;
    const auto pi = model_->plan().parameter(pi_role).begin;
    const auto zeta = model_->plan().parameter(zeta_role).begin;
    const EnvelopeParameterReverseLayout layout{count, shape.determinants, ne, nuclei, begin, end,
        ne, ne * ne, shape.determinants * ne * ne, nuclei, ne * nuclei, nuclei, ne * nuclei, pi, zeta};
    checkRuntime(launchEnvelopeParameterReverse(queue_.getNative(), layout, ion_distances_.data(),
        model_->parameters() + pi, model_->parameters() + zeta, envelope_.data(), head_response_.data()),
        "score envelope reverse");
  }

  std::shared_ptr<const PreparedDeviceModel> model_;
  std::size_t capacity_, pair_count_;
  compute::Queue<PlatformKind::CUDA>& queue_;
  AcceleratorBlasHandle& blas_;
  PreparedDeviceNetworkReverse network_;
  DeviceBuffer<double> seeds_, up_, down_, envelope_, feature_, ion_distances_, pair_distances_, weights_;
  DeviceBuffer<double> head_response_, response_;
  DeviceBuffer<double> channel_weights_;
  std::vector<double> unit_weights_, cusp_weights_;
  DeviceBuffer<std::uint8_t> same_spin_;
  DeviceBuffer<device_determinant::DerivativeStatus> status_;
  std::vector<device_determinant::DerivativeStatus> host_status_;
  std::vector<device_determinant::FactorizationMetadata> host_factors_;
  DerivativeFiniteScanLayout finite_layout_;
  DeviceBuffer<DerivativeFiniteStatus> finite_status_;
  std::vector<DerivativeFiniteStatus> host_finite_status_;
  bool ready_ = false;
};

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_SCORE_REVERSE_H
