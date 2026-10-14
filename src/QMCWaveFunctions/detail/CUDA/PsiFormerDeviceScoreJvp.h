//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceScoreJvp.h
 * @brief Explicit ordinary forward parameter-direction log-score product.
 *
 * Uses one canonical O(P) direction, bounded feature/attention/orbital tangents
 * and one scalar result per configuration. No coordinate derivative lane is
 * repurposed and no B-by-P Jacobian is formed. Like the weighted reverse, this
 * first numerical route requires regular determinant channels; production
 * training/provider capability and singular-channel cofactors remain separate.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_SCORE_JVP_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_SCORE_JVP_H

#include "PsiFormerDeviceInference.h"

namespace qmcplusplus::psiformer::device
{

/** Own fixed-capacity tangent scratch for one parameter direction and model image. */
class PreparedDeviceScoreJvp
{
public:
  /// Prepare O(P) direction storage and bounded tile scratch before evaluation.
  PreparedDeviceScoreJvp(std::shared_ptr<const PreparedDeviceModel> model,
                         std::size_t capacity,
                         compute::Queue<PlatformKind::CUDA>& queue,
                         AcceleratorBlasHandle& blas)
      : model_(std::move(model)), capacity_(capacity), queue_(queue), blas_(blas)
  {
    if (!model_ || capacity_ == 0)
      throw std::invalid_argument("Score JVP requires a model and nonzero capacity");
    model_->validateCurrentDevice();
    const auto& shape = model_->plan().modelShape();
    const auto features = multiply(multiply(capacity_, shape.electrons()), shape.feature_dimension);
    for (auto* buffer : {&feature_, &query_, &key_, &value_, &context_, &residual_,
                         &hidden_, &projected_, &work_left_, &work_right_})
      buffer->reserve(features);
    probability_.reserve(multiply(multiply(multiply(capacity_, shape.attention_heads),
                                            shape.electrons()), shape.electrons()));
    const auto matrices = multiply(multiply(multiply(capacity_, shape.determinants),
                                            shape.electrons()), shape.electrons());
    seeds_.reserve(matrices);
    orbital_.reserve(matrices);
    direction_.reserve(model_->plan().parameterCount());
    products_.reserve(capacity_);
    status_.reserve(capacity_);
    host_status_.resize(capacity_);
    host_factors_.resize(multiply(capacity_, shape.determinants));
    finite_status_.reserve(1);
    host_finite_status_.resize(1);
  }

  /** Produce d log|Psi_s|/dp dot direction for every live configuration.
   * The ordinary forward tape and model are immutable for this complete call.
   */
  const double* evaluate(const PreparedDeviceInference& forward, const std::vector<double>& direction)
  {
    ready_ = false;
    model_->validateCurrentDevice();
    const auto count = forward.valueTapeBatchCount();
    const auto& shape = model_->plan().modelShape();
    const auto ne = shape.electrons(), width = shape.feature_dimension;
    if (forward.modelImage().get() != model_.get() || count > capacity_ || direction.size() != direction_.capacity())
      throw std::invalid_argument("Score JVP model, capacity or direction extent mismatch");
    for (double value : direction)
      if (!device_math::isFiniteBinary64(value))
        throw std::invalid_argument("Score JVP direction is nonfinite");
    forward.downloadValueFactorizations(host_factors_);
    for (std::size_t channel = 0; channel < count * shape.determinants; ++channel)
      if (!host_factors_[channel].inverse_available)
        throw std::runtime_error("GPU score JVP requires regular determinant channels; cofactor fallback unavailable");
    direction_.upload(direction.data(), direction.size());
    const auto stream = queue_.getNative();
    const auto head = forward.valueHeadTape();
    checkRuntime(launchDeterminantMatrixReverseSeeds(stream, head.factorizations, head.combinations,
        head.normalized_weights, head.inverses, count, shape.determinants, ne, seeds_.data(), status_.data()),
        "score JVP determinant seeds");
    queue_.sync();
    status_.download(host_status_.data(), count);
    for (std::size_t configuration = 0; configuration < count; ++configuration)
      if (host_status_[configuration] != device_determinant::DerivativeStatus::AVAILABLE)
        throw std::runtime_error("GPU log-score JVP is undefined or nonfinite");

    // Feature layer: fixed physical features have zero parameter tangent; only
    // the embedding weights contribute before entering the shared neural trunk.
    const auto& embedding = model_->plan().parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
    const DenseForwardLayout feature_dense{count * ne, embedding.shape[0], width,
                                           embedding.shape[0], width, width};
    denseForward(blas_, feature_dense, forward.valueRawFeatures(),
                 direction_.data() + embedding.begin, feature_.data());
    const DenseForwardLayout dense{count * ne, width, width, width, width, width};
    const BatchedValueLayout values{count, ne, width, width, ne * width};
    const BatchedAttentionForwardLayout attention{count,
        {ne, shape.attention_heads, width / shape.attention_heads, width, ne, ne * ne},
        ne * width, shape.attention_heads * ne * ne};
    const auto feature_count = multiply(multiply(count, ne), width);

    // Remaining layers: each dense tangent is dX W + X dW, with separate
    // ordinary attention and nonlinear rules rather than coordinate-jet reuse.
    for (std::size_t layer = 0; layer < shape.attention_blocks; ++layer)
    {
      const auto& tape = forward.valueBlockTape(layer);
      project(dense, tape.input.data(), feature_.data(), ParameterRole::ATTENTION_QUERY_WEIGHT, layer, query_);
      project(dense, tape.input.data(), feature_.data(), ParameterRole::ATTENTION_KEY_WEIGHT, layer, key_);
      project(dense, tape.input.data(), feature_.data(), ParameterRole::ATTENTION_VALUE_WEIGHT, layer, value_);
      checkRuntime(launchAttentionParameterDirection(stream, attention, tape.query.data(), tape.key.data(),
          tape.value.data(), tape.probabilities.data(), query_.data(), key_.data(), value_.data(),
          probability_.data(), context_.data()), "score JVP attention");
      project(dense, tape.context.data(), context_.data(), ParameterRole::ATTENTION_OUTPUT_WEIGHT, layer, projected_);
      add(feature_, projected_, residual_, feature_count);
      project(dense, tape.residual.data(), residual_.data(), ParameterRole::UPDATE_HIDDEN_WEIGHT, layer, hidden_);
      tanh(values, tape.hidden.data(), ParameterRole::UPDATE_HIDDEN_BIAS, layer, hidden_);
      project(dense, tape.hidden.data(), hidden_.data(), ParameterRole::UPDATE_OUTPUT_WEIGHT, layer, projected_);
      tanh(values, tape.projected.data(), ParameterRole::UPDATE_OUTPUT_BIAS, layer, projected_);
      add(residual_, projected_, feature_, feature_count);
    }

    // Head: full spin-row orbital tangents, determinant seed contraction, cusp.
    const auto orbital_layout = makeOpenOrbitalJetLayout(count, shape.determinants, ne, shape.nuclei,
        width, shape.spin_up_electrons, SpatialJetMode::VALUE_ONLY);
    checkRuntime(launchOpenOrbitalParameterDirection(stream, orbital_layout, forward.valueFinalFeatures(),
        feature_.data(), head.positions, model_->nuclei(), orbitalParameters(model_->parameters()),
        orbitalParameters(direction_.data()), head.backflow, head.envelope, orbital_.data()), "score JVP orbitals");
    const auto* same = model_->plan().optionalParameter(ParameterRole::CUSP_SAME_ALPHA);
    const auto opposite = model_->plan().parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin;
    checkRuntime(launchLogScoreParameterDirection(stream, count, ne, shape.determinants, shape.spin_up_electrons,
        head.positions, seeds_.data(), orbital_.data(), model_->sameAlpha(), model_->oppositeAlpha(),
        same ? direction[same->begin] : 0.0, direction[opposite], products_.data()), "score JVP reduction");
    const DerivativeFiniteScanLayout finite{count, count, 1, 0, 1};
    checkRuntime(launchDerivativeFiniteScan(stream, finite, products_.data(), finite_status_.data()), "score JVP finite scan");
    queue_.sync();
    finite_status_.download(host_finite_status_.data(), 1);
    if (host_finite_status_[0].nonfinite_count != 0)
      throw std::runtime_error("GPU score JVP produced a nonfinite product");
    live_count_ = count;
    ready_ = true;
    return products_.data();
  }

  /// Download completed scalar products; reject stale results after any failed call.
  void downloadProducts(std::vector<double>& destination) const
  {
    if (!ready_)
      throw std::logic_error("Score JVP has no completed products");
    if (destination.size() != live_count_)
      throw std::invalid_argument("Score JVP destination extent mismatch");
    products_.download(destination.data(), destination.size());
  }

  /// Rebind immutable weights while preserving prepared tangent/direction storage.
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> successor)
  {
    model_->validateSuccessor(successor);
    model_ = std::move(successor);
    ready_ = false;
    live_count_ = 0;
  }

  /// Account prepared device scratch separately from the immutable model image.
  /// Count host-side regularity and finite-result admission arrays.
  std::size_t hostWorkspaceBytes() const
  {
    return checkedDeviceBytes<device_determinant::DerivativeStatus>(host_status_.capacity()) +
        checkedDeviceBytes<device_determinant::FactorizationMetadata>(host_factors_.capacity()) +
        checkedDeviceBytes<DerivativeFiniteStatus>(host_finite_status_.capacity());
  }

  /// Count prepared device tangent, direction and admission buffers.
  std::size_t deviceWorkspaceBytes() const
  {
    std::size_t bytes = 0;
    const auto add = [&](std::size_t amount) {
      bytes = spatial_detail::checkedAdd(bytes, amount, "Score JVP storage overflow");
    };
    for (const auto* buffer : {&feature_, &query_, &key_, &value_, &context_, &residual_,
        &hidden_, &projected_, &work_left_, &work_right_, &probability_, &seeds_, &orbital_, &direction_, &products_})
      add(checkedDeviceBytes<double>(buffer->capacity()));
    add(checkedDeviceBytes<device_determinant::DerivativeStatus>(status_.capacity()));
    add(checkedDeviceBytes<DerivativeFiniteStatus>(finite_status_.capacity()));
    return bytes;
  }

private:
  /// Check all prepared extents before allocation or layout construction.
  static std::size_t multiply(std::size_t left, std::size_t right)
  { return spatial_detail::checkedProduct(left, right, "Score JVP extent overflow"); }

  /// Bind spin-channel intervals from the plan, not inferred adjacent offsets.
  OrbitalDirectionParameters orbitalParameters(const double* source) const
  {
    const auto bind = [&](ParameterRole role) { return source + model_->plan().parameter(role).begin; };
    return {bind(ParameterRole::BACKFLOW_UP_WEIGHT), bind(ParameterRole::BACKFLOW_DOWN_WEIGHT),
            bind(ParameterRole::ENVELOPE_PI_UP), bind(ParameterRole::ENVELOPE_PI_DOWN),
            bind(ParameterRole::ENVELOPE_ZETA_UP), bind(ParameterRole::ENVELOPE_ZETA_DOWN)};
  }

  /// Form a dense parameter tangent using two GEMMs and prepared sum scratch.
  void project(const DenseForwardLayout& layout, const double* source, const double* source_direction,
               ParameterRole role, std::size_t layer, DeviceBuffer<double>& target)
  {
    const auto begin = model_->plan().parameter(role, layer).begin;
    denseForward(blas_, layout, source_direction, model_->parameters() + begin, work_left_.data());
    denseForward(blas_, layout, source, direction_.data() + begin, work_right_.data());
    add(work_left_, work_right_, target, layout.rows * layout.output_width);
  }

  /// Add two ordinary tangent arrays into a distinct prepared destination.
  void add(const DeviceBuffer<double>& left, const DeviceBuffer<double>& right,
           DeviceBuffer<double>& target, std::size_t count)
  { checkRuntime(launchAddParameterResponses(queue_.getNative(), count, left.data(), right.data(), target.data()), "score JVP sum"); }

  /// Apply the tanh chain rule plus the canonical bias direction in place.
  void tanh(const BatchedValueLayout& layout, const double* value,
            ParameterRole bias, std::size_t layer, DeviceBuffer<double>& tangent)
  {
    checkRuntime(launchBiasTanhParameterDirection(queue_.getNative(), layout, value, tangent.data(),
        direction_.data() + model_->plan().parameter(bias, layer).begin, tangent.data()), "score JVP tanh");
  }

  std::shared_ptr<const PreparedDeviceModel> model_;
  std::size_t capacity_, live_count_ = 0;
  compute::Queue<PlatformKind::CUDA>& queue_;
  AcceleratorBlasHandle& blas_;
  DeviceBuffer<double> feature_, query_, key_, value_, context_, residual_, hidden_, projected_, work_left_, work_right_;
  DeviceBuffer<double> probability_, seeds_, orbital_, direction_, products_;
  DeviceBuffer<device_determinant::DerivativeStatus> status_;
  std::vector<device_determinant::DerivativeStatus> host_status_;
  std::vector<device_determinant::FactorizationMetadata> host_factors_;
  DeviceBuffer<DerivativeFiniteStatus> finite_status_;
  std::vector<DerivativeFiniteStatus> host_finite_status_;
  bool ready_ = false;
};

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_SCORE_JVP_H
