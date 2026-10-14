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
                             AcceleratorBlasHandle& blas)
      : model_(std::move(model)), capacity_(capacity), queue_(queue), blas_(blas),
        network_(model_, capacity, queue, blas)
  {
    const auto& shape = model_->plan().modelShape();
    const auto ne = shape.electrons();
    const auto matrices = multiply(multiply(multiply(capacity_, shape.determinants), ne), ne);
    for (auto* buffer : {&seeds_, &up_, &down_, &envelope_})
      buffer->reserve(matrices);
    feature_.reserve(multiply(multiply(capacity_, ne), shape.feature_dimension));
    ion_distances_.reserve(multiply(multiply(capacity_, ne), shape.nuclei));
    pair_count_ = multiply(ne, ne - 1) / 2;
    pair_distances_.reserve(multiply(capacity_, pair_count_));
    same_spin_.reserve(pair_count_);
    weights_.reserve(capacity_);
    head_response_.reserve(model_->plan().parameterCount());
    response_.reserve(model_->plan().parameterCount());
    status_.reserve(capacity_);
    host_status_.resize(capacity_);
    host_factors_.resize(multiply(capacity_, shape.determinants));
    finite_layout_ = {response_.capacity(), 4096,
        response_.capacity() / 4096 + (response_.capacity() % 4096 != 0), 0, 1};
    finite_status_.reserve(finite_layout_.status_capacity);
    host_finite_status_.resize(finite_layout_.status_capacity);
  }

  /** Reverse one completed value tape into a canonical weighted score vector.
   * Weights are arbitrary finite signed contractions, not implicitly normalized
   * probabilities. Sample/model version and coordinates remain owned by forward.
   */
  const double* reverse(const PreparedDeviceInference& forward, const std::vector<double>& weights)
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

    // Inverse seeds cannot represent singular-channel cofactor derivatives even
    // when the determinant sum is nonzero. Reject rather than silently drop them.
    forward.downloadValueFactorizations(host_factors_);
    for (std::size_t channel = 0; channel < count * determinants; ++channel)
      if (!host_factors_[channel].inverse_available)
        throw std::runtime_error("GPU score requires regular determinant channels; cofactor fallback unavailable");
    const auto head = forward.valueHeadTape();
    const auto stream = queue_.getNative();
    checkRuntime(launchDeterminantMatrixReverseSeeds(stream, head.factorizations, head.combinations,
        head.normalized_weights, head.inverses, count, determinants, ne, seeds_.data(), status_.data()),
        "score determinant seeds");
    queue_.sync();
    status_.download(host_status_.data(), count);
    for (std::size_t configuration = 0; configuration < count; ++configuration)
      if (host_status_[configuration] != device_determinant::DerivativeStatus::AVAILABLE)
        throw std::runtime_error("GPU log score is undefined or nonfinite for this configuration");

    // Output head: weighted matrix cotangents, spin-row packing, geometry and
    // orbital/envelope parameter contractions. No division by B or E is used.
    weights_.upload(weights.data(), count);
    const OrbitalProductReverseLayout orbital{count, determinants, ne, ne, ne * ne, determinants * ne * ne};
    checkRuntime(launchWeightedOrbitalProductReverse(stream, orbital, shape.spin_up_electrons,
        weights_.data(), head.backflow, head.envelope, seeds_.data(), up_.data(), down_.data(), envelope_.data()),
        "score orbital product reverse");
    checkRuntime(launchScoreGeometry(stream, count, ne, shape.nuclei, shape.spin_up_electrons,
        head.positions, model_->nuclei(), ion_distances_.data(), pair_distances_.data(), same_spin_.data()),
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
      accumulateDenseInputAdjoint(blas_, backflow, spin.second->data(), model_->parameters() + offset,
                                  feature_.data());
    }
    envelopeReverse(count, 0, shape.spin_up_electrons,
                    ParameterRole::ENVELOPE_PI_UP, ParameterRole::ENVELOPE_ZETA_UP);
    envelopeReverse(count, shape.spin_up_electrons, ne,
                    ParameterRole::ENVELOPE_PI_DOWN, ParameterRole::ENVELOPE_ZETA_DOWN);
    if (pair_count_ != 0)
    {
      const auto* same = model_->plan().optionalParameter(ParameterRole::CUSP_SAME_ALPHA);
      const CuspAlphaReverseLayout cusp{count, pair_count_, pair_count_, same ? same->begin : 0,
          model_->plan().parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin, same != nullptr};
      checkRuntime(launchCuspAlphaReverse(stream, cusp, pair_distances_.data(), same_spin_.data(),
          model_->sameAlpha(), model_->oppositeAlpha(), head_response_.data(), weights_.data()), "score cusp reverse");
    }

    // Remaining trunk and feature layer: one device cotangent drives the explicit
    // shared-weight reverse. Add independent head/trunk intervals canonically.
    const double* trunk = network_.reverse(forward, {}, feature_.data());
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

  /// Download a completed canonical response, rejecting stale results after failure.
  void downloadResponse(std::vector<double>& destination) const
  {
    if (!ready_)
      throw std::logic_error("GPU score has no completed response");
    if (destination.size() != response_.capacity())
      throw std::invalid_argument("GPU score destination extent mismatch");
    response_.download(destination.data(), destination.size());
  }

  /// Account all explicitly owned numerical storage, not the shared model image.
  std::size_t deviceWorkspaceBytes() const
  {
    std::size_t bytes = network_.deviceWorkspaceBytes();
    const auto add = [&](std::size_t amount) {
      bytes = spatial_detail::checkedAdd(bytes, amount, "Score workspace storage overflow");
    };
    for (const auto* buffer : {&seeds_, &up_, &down_, &envelope_, &feature_, &ion_distances_,
                              &pair_distances_, &weights_, &head_response_, &response_})
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
