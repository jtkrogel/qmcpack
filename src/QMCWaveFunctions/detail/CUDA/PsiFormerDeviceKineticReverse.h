//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceKineticReverse.h
 * @brief Bounded, explicitly contracted lifted reverse of real kinetic energy.
 *
 * Uses retained full-VGL jets, one O(P) accumulator and one-configuration
 * scratch. Configurations accumulate in deterministic order through existing
 * serial lifted leaves. This is a correctness-first numerical route, not a
 * batched performance or production-training claim. Coulomb potentials have
 * zero parameter response at fixed geometry; nonlocal ECP response is separate.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_KINETIC_REVERSE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_KINETIC_REVERSE_H

#include "PsiFormerDeviceInference.h"
#include "PsiFormerDeviceSpatialReverse.h"
#include "PsiFormerDeviceOpenKineticReverse.h"

namespace qmcplusplus::psiformer::device
{

/** Own one lifted reverse workspace, independent of walker-by-parameter matrices. */
class PreparedDeviceKineticReverse
{
public:
  /// Prepare one-config jet scratch and a bounded sample-input capacity.
  PreparedDeviceKineticReverse(std::shared_ptr<const PreparedDeviceModel> model,
                               std::size_t capacity,
                               compute::Queue<PlatformKind::CUDA>& queue,
                               AcceleratorBlasHandle& blas)
      : model_(std::move(model)), capacity_(capacity), queue_(queue), blas_(blas)
  {
    if (!model_ || capacity_ == 0)
      throw std::invalid_argument("Kinetic reverse requires a model and nonzero capacity");
    model_->validateCurrentDevice();
    const auto& shape = model_->plan().modelShape();
    const auto ne = shape.electrons(), width = shape.feature_dimension;
    dense_ = makeSpatialDenseJetLayout(1, ne, width, width, ne, SpatialJetMode::FULL_VGL);
    elementwise_ = makeSpatialElementwiseJetLayout(1, ne, width, ne, SpatialJetMode::FULL_VGL);
    attention_ = makeSpatialAttentionJetLayout(1, ne, shape.attention_heads, width / shape.attention_heads, ne,
                                              SpatialJetMode::FULL_VGL);
    orbitals_ = makeOpenOrbitalJetLayout(1, shape.determinants, ne, shape.nuclei, width,
                                        shape.spin_up_electrons, SpatialJetMode::FULL_VGL);
    cusp_ = makeOpenCuspJetLayout(1, ne, shape.spin_up_electrons, SpatialJetMode::FULL_VGL);
    root_ = makeFinalSpatialJetLayout(1, ne, SpatialJetMode::FULL_VGL).output;
    const auto features = spatialJetSpanElements(dense_.source);
    for (auto* buffer : {&feature_, &residual_, &pre_, &hidden_, &input_, &context_, &query_, &key_, &value_})
      buffer->reserve(features);
    probability_.reserve(spatialJetSpanElements(attention_.attention));
    logit_.reserve(probability_.capacity());
    const auto matrices = spatialJetSpanElements(orbitals_.orbitals);
    for (auto* buffer : {&factor_, &orbital_, &backflow_, &envelope_})
      buffer->reserve(matrices);
    matrix_scratch_.reserve(multiply(3, multiply(ne, ne)));
    root_adjoint_.reserve(spatialJetSpanElements(root_));
    response_.reserve(model_->plan().parameterCount());
    dummy_bias_.reserve(width);
    weights_.reserve(capacity_);
    masses_.reserve(ne);
    total_gradient_.reserve(multiply(capacity_, multiply(3, ne)));
    determinant_status_.reserve(1);
    reverse_status_.reserve(1);
    open_status_.reserve(1);
    kinetic_status_.reserve(1);
    host_determinant_status_.resize(1);
    host_open_status_.resize(1);
    host_factors_.resize(multiply(capacity_, shape.determinants));
    finite_layout_ = {response_.capacity(), 4096,
        response_.capacity() / 4096 + (response_.capacity() % 4096 != 0), 0, 1};
    finite_status_.reserve(finite_layout_.status_capacity);
    host_finite_status_.resize(finite_layout_.status_capacity);
  }

  /** Contract fixed-coordinate kinetic responses with finite signed sample weights.
   * Empty masses mean unit inverse masses; empty total drift uses this component.
   * Nonempty total drift must contain all configurations' complete 3N lanes.
   * Gradient-only mode requires explicit drift and omits the Laplacian seed;
   * use the named wrapper for a complex outer phase's imaginary cross term.
   */
  const double* evaluate(const PreparedDeviceInference& forward, const std::vector<double>& weights,
                         const std::vector<double>& inverse_masses = {},
                         const std::vector<double>& total_gradient = {}, bool gradient_only = false)
  {
    ready_ = false;
    model_->validateCurrentDevice();
    const auto head = forward.spatialHeadTape();
    const auto count = head.orbital_layout.features.configuration_count;
    const auto& shape = model_->plan().modelShape();
    const auto ne = shape.electrons(), width = shape.feature_dimension;
    if (forward.modelImage().get() != model_.get() || count > capacity_ || weights.size() != count ||
        (!inverse_masses.empty() && inverse_masses.size() != ne) ||
        (!total_gradient.empty() && total_gradient.size() != count * 3 * ne) ||
        (gradient_only && total_gradient.empty()))
      throw std::invalid_argument("Kinetic reverse model, capacity or input extent mismatch");
    for (const auto* values : {&weights, &inverse_masses, &total_gradient})
      for (double value : *values)
        if (!device_math::isFiniteBinary64(value))
          throw std::invalid_argument("Kinetic reverse input is nonfinite");
    for (double mass : inverse_masses)
      if (mass <= 0)
        throw std::invalid_argument("Kinetic reverse inverse masses must be positive");
    forward.downloadSpatialFactorizations(host_factors_);
    for (std::size_t channel = 0; channel < count * shape.determinants; ++channel)
      if (!host_factors_[channel].inverse_available)
        throw std::runtime_error("Kinetic reverse requires regular determinant channels; cofactor fallback unavailable");
    weights_.upload(weights.data(), count);
    if (!inverse_masses.empty())
      masses_.upload(inverse_masses.data(), ne);
    if (!total_gradient.empty())
      total_gradient_.upload(total_gradient.data(), total_gradient.size());
    zero(response_);
    const auto stream = queue_.getNative();
    const LiftedDeterminantLayout determinant{orbitals_.orbitals, root_, shape.determinants, ne};

    // Existing lifted leaves admit one configuration. Reuse their scratch and
    // accumulate one canonical response rather than retaining B separate vectors.
    for (std::size_t configuration = 0; configuration < count; ++configuration)
    {
      const auto orbital_offset = configuration * head.orbital_layout.orbitals.configuration_stride;
      const auto feature_offset = configuration * head.orbital_layout.features.configuration_stride;
      const auto root_offset = configuration * head.root_layout.configuration_stride;
      checkRuntime(launchKineticRootAdjoints(stream, root_, head.root + root_offset, weights_.data() + configuration,
          inverse_masses.empty() ? nullptr : masses_.data(),
          total_gradient.empty() ? nullptr : total_gradient_.data() + configuration * 3 * ne,
          total_gradient.empty() ? 0 : 3 * ne, root_adjoint_.data(), open_status_.data(), gradient_only), "kinetic root adjoint");
      checkOpen();
      checkRuntime(launchDeterminantFactorJets(stream, determinant,
          head.factorizations + configuration * shape.determinants, head.combinations + configuration,
          head.normalized_weights + configuration * shape.determinants,
          head.inverses + configuration * shape.determinants * ne * ne, head.orbitals + orbital_offset,
          head.determinant_log_gradient + configuration * 3 * ne,
          head.determinant_log_laplacian + configuration * ne, factor_.data(), matrix_scratch_.data(),
          determinant_status_.data()), "kinetic determinant factor jets");
      checkDeterminant(determinant_status_);
      checkRuntime(launchDeterminantJetReverse(stream, determinant, determinant_status_.data(), factor_.data(),
          root_adjoint_.data(), orbital_.data(), reverse_status_.data()), "kinetic determinant reverse");
      checkDeterminant(reverse_status_);

      // Output head: full orbital-product, envelope/backflow and cusp reverse.
      zero(feature_);
      zero(backflow_);
      zero(envelope_);
      const OpenOrbitalKineticReverseLayout orbital_layout{orbitals_, response_.capacity(),
          offset(ParameterRole::BACKFLOW_UP_WEIGHT), offset(ParameterRole::BACKFLOW_DOWN_WEIGHT),
          offset(ParameterRole::ENVELOPE_PI_UP), offset(ParameterRole::ENVELOPE_PI_DOWN),
          offset(ParameterRole::ENVELOPE_ZETA_UP), offset(ParameterRole::ENVELOPE_ZETA_DOWN)};
      checkRuntime(launchOpenOrbitalKineticReverse(stream, orbital_layout,
          head.features + feature_offset, head.positions + configuration * 3 * ne, model_->nuclei(),
          parameter(ParameterRole::BACKFLOW_UP_WEIGHT), parameter(ParameterRole::BACKFLOW_DOWN_WEIGHT),
          parameter(ParameterRole::ENVELOPE_PI_UP), parameter(ParameterRole::ENVELOPE_PI_DOWN),
          parameter(ParameterRole::ENVELOPE_ZETA_UP), parameter(ParameterRole::ENVELOPE_ZETA_DOWN),
          head.backflow + orbital_offset, head.envelope + orbital_offset, orbital_.data(),
          backflow_.data(), envelope_.data(), feature_.data(), response_.data(), kinetic_status(),
          model_->periodicCell()), "kinetic orbital reverse");
      checkKinetic();
      const auto* same = model_->plan().optionalParameter(ParameterRole::CUSP_SAME_ALPHA);
      const OpenCuspKineticReverseLayout cusp_layout{cusp_, response_.capacity(), same ? same->begin : 0,
          offset(ParameterRole::CUSP_OPPOSITE_ALPHA), same != nullptr};
      checkRuntime(launchOpenCuspKineticReverse(stream, cusp_layout, head.positions + configuration * 3 * ne,
          model_->sameAlpha(), model_->oppositeAlpha(), root_adjoint_.data(), response_.data(), kinetic_status(),
          model_->periodicCell()),
          "kinetic cusp reverse");
      checkKinetic();

      // Remaining neural layers reverse all value/gradient/Laplacian cotangents.
      for (std::size_t layer = shape.attention_blocks; layer-- > 0;)
      {
        const auto& tape = forward.spatialBlockTape(layer);
        const auto attention_offset = configuration * attention_.attention.configuration_stride;
        const auto at = [&](const DeviceBuffer<double>& buffer) { return buffer.data() + feature_offset; };
        copy(feature_, residual_);
        zero(pre_);
        checkRuntime(launchTanhJetReverse(stream, elementwise_, at(tape.projected_pre), at(tape.projected),
            feature_.data(), pre_.data()), "kinetic output tanh reverse");
        zero(hidden_);
        denseReverse(at(tape.hidden), pre_, hidden_, ParameterRole::UPDATE_OUTPUT_WEIGHT, layer,
                     response_.data() + offset(ParameterRole::UPDATE_OUTPUT_BIAS, layer));
        zero(pre_);
        checkRuntime(launchTanhJetReverse(stream, elementwise_, at(tape.hidden_pre), at(tape.hidden),
            hidden_.data(), pre_.data()), "kinetic hidden tanh reverse");
        denseReverse(at(tape.residual), pre_, residual_, ParameterRole::UPDATE_HIDDEN_WEIGHT, layer,
                     response_.data() + offset(ParameterRole::UPDATE_HIDDEN_BIAS, layer));
        copy(residual_, input_);
        zero(context_);
        denseReverse(at(tape.context), residual_, context_, ParameterRole::ATTENTION_OUTPUT_WEIGHT, layer);
        zero(probability_);
        zero(value_);
        checkRuntime(launchAttentionContextJetReverse(stream, attention_, tape.probabilities.data() + attention_offset,
            at(tape.value), context_.data(), probability_.data(), value_.data()), "kinetic context reverse");
        zero(logit_);
        checkRuntime(launchSoftmaxJetReverse(stream, makeAttentionSoftmaxJetRowLayout(attention_),
            tape.logits.data() + attention_offset, tape.probabilities.data() + attention_offset,
            probability_.data(), logit_.data()), "kinetic softmax reverse");
        zero(query_);
        zero(key_);
        checkRuntime(launchAttentionLogitJetReverse(stream, attention_, at(tape.query), at(tape.key),
            logit_.data(), query_.data(), key_.data()), "kinetic Q/K reverse");
        denseReverse(at(tape.input), query_, input_, ParameterRole::ATTENTION_QUERY_WEIGHT, layer);
        denseReverse(at(tape.input), key_, input_, ParameterRole::ATTENTION_KEY_WEIGHT, layer);
        denseReverse(at(tape.input), value_, input_, ParameterRole::ATTENTION_VALUE_WEIGHT, layer);
        copy(input_, feature_);
      }

      // Feature-layer reverse stops at fixed physical inputs. Fold all spatial
      // planes for the embedding weight response without allocating raw adjoints.
      const auto& embedding = model_->plan().parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
      auto embedding_layout = makeSpatialDenseJetLayout(1, ne, embedding.shape[0], width, ne, SpatialJetMode::FULL_VGL);
      embedding_layout.source = withSpatialTargetRange(embedding_layout.source,
          head.root_layout.target_electron_begin, head.root_layout.laplacian_lanes);
      embedding_layout.target = withSpatialTargetRange(embedding_layout.target,
          head.root_layout.target_electron_begin, head.root_layout.laplacian_lanes);
      accumulateDenseWeightAdjoint(blas_, makeFoldedSpatialDenseLayout(embedding_layout),
          head.raw_features + configuration * embedding_layout.source.configuration_stride, feature_.data(),
          response_.data() + embedding.begin);
    }
    checkRuntime(launchDerivativeFiniteScan(stream, finite_layout_, response_.data(), finite_status_.data()), "kinetic finite scan");
    queue_.sync();
    finite_status_.download(host_finite_status_.data(), host_finite_status_.size());
    for (const auto& status : host_finite_status_)
      if (status.nonfinite_count != 0)
        throw std::runtime_error("Kinetic parameter response is nonfinite");
    ready_ = true;
    return response_.data();
  }

  /** Contract -sum_e m_e^-1 drift_e dot d_theta grad_e log|Psi|.
   * The explicit drift can be the imaginary total TrialWaveFunction drift.
   * This reuses the same lifted reverse/storage, with no Laplacian seed.
   */
  const double* evaluateGradientResponse(const PreparedDeviceInference& forward,
      const std::vector<double>& weights, const std::vector<double>& inverse_masses,
      const std::vector<double>& total_gradient)
  {
    return evaluate(forward, weights, inverse_masses, total_gradient, true);
  }

  /// Download a completed canonical contraction without exposing stale responses.
  void downloadResponse(std::vector<double>& destination) const
  {
    if (!ready_)
      throw std::logic_error("Kinetic response is unavailable");
    if (destination.size() != response_.capacity())
      throw std::invalid_argument("Kinetic response destination extent differs");
    response_.download(destination.data(), destination.size());
  }

  /// Rebind a fixed-layout successor without reallocating lifted scratch.
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> successor)
  {
    model_->validateSuccessor(successor);
    model_ = std::move(successor);
    ready_ = false;
  }

  /// Stream a completed canonical interval without replaying the lifted reverse.
  void downloadResponseChunk(std::size_t begin, double* destination, std::size_t count) const
  {
    if (!ready_)
      throw std::logic_error("Kinetic response is unavailable");
    if (begin > response_.capacity() || count > response_.capacity() - begin)
      throw std::out_of_range("Kinetic response chunk exceeds canonical extent");
    response_.download(destination, count, begin);
  }

  /// Account owned device scratch; forward tape/model/vendor BLAS are separate.
  /// Count retained host regularity/status storage for the lifted reverse.
  std::size_t hostWorkspaceBytes() const
  {
    return checkedDeviceBytes<device_determinant::DerivativeStatus>(host_determinant_status_.capacity()) +
        checkedDeviceBytes<OpenSpatialStatus>(host_open_status_.capacity()) +
        checkedDeviceBytes<device_determinant::FactorizationMetadata>(host_factors_.capacity()) +
        checkedDeviceBytes<DerivativeFiniteStatus>(host_finite_status_.capacity());
  }

  /// Count prepared lifted device reverse and canonical response storage.
  std::size_t deviceWorkspaceBytes() const
  {
    std::size_t bytes = 0;
    const auto add = [&](std::size_t amount) {
      bytes = spatial_detail::checkedAdd(bytes, amount, "Kinetic reverse storage overflow");
    };
    for (const auto* buffer : {&feature_, &residual_, &pre_, &hidden_, &input_, &context_, &query_, &key_,
        &value_, &probability_, &logit_, &factor_, &orbital_, &backflow_, &envelope_, &matrix_scratch_,
        &root_adjoint_, &response_, &dummy_bias_, &weights_, &masses_, &total_gradient_})
      add(checkedDeviceBytes<double>(buffer->capacity()));
    add(checkedDeviceBytes<device_determinant::DerivativeStatus>(determinant_status_.capacity()));
    add(checkedDeviceBytes<device_determinant::DerivativeStatus>(reverse_status_.capacity()));
    add(checkedDeviceBytes<OpenSpatialStatus>(open_status_.capacity()));
    add(checkedDeviceBytes<OpenKineticReverseStatus>(kinetic_status_.capacity()));
    add(checkedDeviceBytes<DerivativeFiniteStatus>(finite_status_.capacity()));
    return bytes;
  }

private:
  /// Check storage multiplication before allocating prepared scratch.
  static std::size_t multiply(std::size_t left, std::size_t right)
  { return spatial_detail::checkedProduct(left, right, "Kinetic reverse extent overflow"); }

  /// Bind a canonical parameter role without inferred tensor adjacency.
  std::size_t offset(ParameterRole role, std::size_t layer = NO_ATTENTION_BLOCK) const
  { return model_->plan().parameter(role, layer).begin; }

  /// Return a read-only parameter interval from this exact immutable model image.
  const double* parameter(ParameterRole role, std::size_t layer = NO_ATTENTION_BLOCK) const
  { return model_->parameters() + offset(role, layer); }

  /// Clear a prepared logical adjoint before an accumulating lifted leaf.
  void zero(DeviceBuffer<double>& buffer)
  { checkRuntime(launchDerivativeZero(queue_.getNative(), {buffer.capacity(), 0, 1, 0, 1}, buffer.data()), "kinetic zero"); }

  /// Copy complete single-configuration feature jets between prepared buffers.
  void copy(const DeviceBuffer<double>& source, DeviceBuffer<double>& destination)
  { checkRuntime(launchDerivativeCopy(queue_.getNative(), {source.capacity(), 0, 1, 0, 1}, source.data(), destination.data()), "kinetic jet copy"); }

  /// Reverse dense jet planes; projections without bias use bounded dummy storage.
  void denseReverse(const double* source, const DeviceBuffer<double>& target, DeviceBuffer<double>& input,
                    ParameterRole weight, std::size_t layer, double* bias_destination = nullptr)
  {
    if (!bias_destination)
    {
      zero(dummy_bias_);
      bias_destination = dummy_bias_.data();
    }
    denseJetsReverse(blas_, queue_.getNative(), dense_, source, parameter(weight, layer), target.data(),
        input.data(), response_.data() + offset(weight, layer), bias_destination);
  }

  /// Synchronize and reject nonregular root seed diagnostics.
  void checkOpen()
  {
    queue_.sync();
    open_status_.download(host_open_status_.data(), 1);
    if (host_open_status_[0] != OpenSpatialStatus::REGULAR)
      throw std::runtime_error("Kinetic root seed is nonregular");
  }

  /// Check a completed determinant factor or lifted reverse status.
  void checkDeterminant(const DeviceBuffer<device_determinant::DerivativeStatus>& status)
  {
    queue_.sync();
    status.download(host_determinant_status_.data(), 1);
    if (host_determinant_status_[0] != device_determinant::DerivativeStatus::AVAILABLE)
      throw std::runtime_error("Kinetic determinant response is unavailable");
  }

  /// Expose a separate correctly typed kinetic-leaf status destination.
  OpenKineticReverseStatus* kinetic_status()
  {
    if (kinetic_status_.capacity() == 0)
      throw std::logic_error("Kinetic status storage was not prepared");
    return kinetic_status_.data();
  }

  /// Check the serial open orbital/cusp reverse status before continuing.
  void checkKinetic()
  {
    queue_.sync();
    kinetic_status_.download(&host_kinetic_status_, 1);
    if (host_kinetic_status_ != OpenKineticReverseStatus::REGULAR)
      throw std::runtime_error("Kinetic orbital/cusp response is nonregular");
  }

  std::shared_ptr<const PreparedDeviceModel> model_;
  std::size_t capacity_;
  compute::Queue<PlatformKind::CUDA>& queue_;
  AcceleratorBlasHandle& blas_;
  SpatialDenseJetLayout dense_;
  SpatialElementwiseJetLayout elementwise_;
  SpatialAttentionJetLayout attention_;
  OpenOrbitalJetLayout orbitals_;
  OpenCuspJetLayout cusp_;
  SpatialJetLayout root_;
  DeviceBuffer<double> feature_, residual_, pre_, hidden_, input_, context_, query_, key_, value_, probability_, logit_;
  DeviceBuffer<double> factor_, orbital_, backflow_, envelope_, matrix_scratch_, root_adjoint_, response_, dummy_bias_;
  DeviceBuffer<double> weights_, masses_, total_gradient_;
  DeviceBuffer<device_determinant::DerivativeStatus> determinant_status_, reverse_status_;
  DeviceBuffer<OpenSpatialStatus> open_status_;
  DeviceBuffer<OpenKineticReverseStatus> kinetic_status_;
  std::vector<device_determinant::DerivativeStatus> host_determinant_status_;
  std::vector<OpenSpatialStatus> host_open_status_;
  std::vector<device_determinant::FactorizationMetadata> host_factors_;
  OpenKineticReverseStatus host_kinetic_status_;
  DerivativeFiniteScanLayout finite_layout_;
  DeviceBuffer<DerivativeFiniteStatus> finite_status_;
  std::vector<DerivativeFiniteStatus> host_finite_status_;
  bool ready_ = false;
};

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_KINETIC_REVERSE_H
