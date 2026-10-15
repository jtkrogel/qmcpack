//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceNetworkReverse.h
 * @brief Explicit bounded-tile reverse of the embedding and attention/MLP trunk.
 *
 * This is a numerical building block, not a complete log-wavefunction score
 * provider. Orbital, envelope, determinant and cusp seeds belong to the caller.
 * Parameter mode contracts into one O(P) destination without a B-by-P Jacobian.
 * Raw-input mode omits that destination and all parameter-adjoint contractions.
 * Model/tape lifetimes are retained; stream/BLAS ownership is external.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_NETWORK_REVERSE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_NETWORK_REVERSE_H

#include "PsiFormerDeviceInference.h"
#include <utility>

namespace qmcplusplus::psiformer::device
{

/// Select the destination before allocating the reverse owner.
enum class NetworkReverseOutput
{
  PARAMETER_SCORE,
  RAW_FEATURE_ADJOINT_ONLY
};

/** Completed [configuration,electron,raw feature] device cotangents.
 * The view is borrowed until the next reverse request or model rebind. Its
 * layout describes the live batch, independently of the prepared capacity.
 * Consumers match model version and evaluation generation to their coordinates.
 */
struct RawFeatureAdjointView
{
  const double* data;
  BatchedValueLayout layout;
  std::size_t model_version;
  std::size_t evaluation_generation;
};

/** Own fixed-capacity adjoints for one explicit neural-trunk reverse operation. */
class PreparedDeviceNetworkReverse
{
public:
  /// Prepare reusable adjoints and one canonical accumulator outside the hot path.
  PreparedDeviceNetworkReverse(std::shared_ptr<const PreparedDeviceModel> model,
                               std::size_t capacity,
                               compute::Queue<PlatformKind::CUDA>& queue,
                               AcceleratorBlasHandle& blas,
                               NetworkReverseOutput output = NetworkReverseOutput::PARAMETER_SCORE)
      : model_(std::move(model)), capacity_(capacity), queue_(queue), blas_(blas), output_(output)
  {
    if (!model_ || capacity_ == 0)
      throw std::invalid_argument("Network reverse requires a model and nonzero capacity");
    model_->validateCurrentDevice();
    const auto& shape = model_->plan().modelShape();
    const auto features = multiply(multiply(capacity_, shape.electrons()), shape.feature_dimension);
    for (auto* buffer : {&upstream_, &residual_, &tanh_, &hidden_, &input_, &context_,
                         &query_, &key_, &value_, &discard_})
      buffer->reserve(features);
    probability_.reserve(multiply(multiply(multiply(capacity_, shape.attention_heads),
                                            shape.electrons()), shape.electrons()));
    if (output_ == NetworkReverseOutput::PARAMETER_SCORE)
      response_.reserve(model_->plan().parameterCount());
    else if (output_ == NetworkReverseOutput::RAW_FEATURE_ADJOINT_ONLY)
    {
      const auto& embedding = model_->plan().parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
      raw_feature_.reserve(multiply(multiply(capacity_, shape.electrons()), embedding.shape[0]));
    }
    else
      throw std::invalid_argument("Network reverse output selection is invalid");
    const auto destination_capacity = response_.capacity() + raw_feature_.capacity();
    finite_layout_ = {destination_capacity, 4096,
        destination_capacity / 4096 + (destination_capacity % 4096 != 0), 0, 1};
    finite_status_.reserve(finite_layout_.status_capacity);
    host_finite_status_.resize(finite_layout_.status_capacity);
  }

  /** Reverse a completed tape against a caller-supplied final-feature cotangent.
   * The seed is [configuration,electron,feature]; sample weights are already
   * included. Results are complete after return and borrowed until the next call.
   */
  const double* reverse(const PreparedDeviceInference& forward, const std::vector<double>& seed,
                        const double* device_seed = nullptr)
  {
    result_ready_ = false;
    if (output_ != NetworkReverseOutput::PARAMETER_SCORE)
      throw std::logic_error("Parameter response requested from raw-feature-only network reverse");
    reverseImpl(forward, seed, device_seed);
    return response_.data();
  }

  /// Reverse the seeded trunk into raw inputs without computing parameter adjoints.
  RawFeatureAdjointView reverseRawFeatures(const PreparedDeviceInference& forward,
      const std::vector<double>& seed, const double* device_seed = nullptr)
  {
    result_ready_ = false;
    if (output_ != NetworkReverseOutput::RAW_FEATURE_ADJOINT_ONLY)
      throw std::logic_error("Raw feature adjoints requested from parameter-score network reverse");
    reverseImpl(forward, seed, device_seed);
    return rawFeatureAdjoints();
  }

private:
  /// Execute the shared input-adjoint algebra and admit only the selected output.
  void reverseImpl(const PreparedDeviceInference& forward, const std::vector<double>& seed,
                   const double* device_seed)
  {
    result_ready_ = false;
    model_->validateCurrentDevice();
    const auto count = forward.valueTapeBatchCount();
    const auto& shape = model_->plan().modelShape();
    const auto ne = shape.electrons(), width = shape.feature_dimension;
    const auto elements = multiply(multiply(count, ne), width);
    if (forward.modelImage().get() != model_.get() || count > capacity_ ||
        (device_seed ? !seed.empty() : seed.size() != elements))
      throw std::invalid_argument("Network reverse model, capacity or seed extent mismatch");
    for (double value : seed)
      if (!device_math::isFiniteBinary64(value))
        throw std::invalid_argument("Network reverse seed is nonfinite");
    if (device_seed)
      checkRuntime(launchDerivativeCopy(queue_.getNative(), {elements, 0, 1, 0, 1}, device_seed,
          upstream_.data()), "network device seed copy");
    else
      upstream_.upload(seed.data(), seed.size());
    if (output_ == NetworkReverseOutput::PARAMETER_SCORE)
      zero(response_, response_.capacity());
    const DenseForwardLayout dense{count * ne, width, width, width, width, width};
    const BatchedValueLayout values{count, ne, width, width, ne * width};
    const BatchedAttentionForwardLayout attention{count,
        {ne, shape.attention_heads, width / shape.attention_heads, width, ne, ne * ne},
        ne * width, shape.attention_heads * ne * ne};
    const auto stream = queue_.getNative();

    // Remaining neural layers: reverse each residual/MLP, then attention block.
    for (std::size_t layer = shape.attention_blocks; layer-- > 0;)
    {
      const auto& tape = forward.valueBlockTape(layer);
      checkRuntime(launchTanhOutputResidualReverse(stream, values, tape.projected.data(),
          upstream_.data(), tanh_.data(), residual_.data()), "network output tanh reverse");
      weightReverse(dense, tape.hidden.data(), tanh_.data(), ParameterRole::UPDATE_OUTPUT_WEIGHT, layer);
      biasReverse(dense, tanh_.data(), ParameterRole::UPDATE_OUTPUT_BIAS, layer);
      zero(hidden_, elements);
      accumulateDenseInputAdjoint(blas_, dense, tanh_.data(),
          parameter(ParameterRole::UPDATE_OUTPUT_WEIGHT, layer), hidden_.data());
      checkRuntime(launchTanhOutputResidualReverse(stream, values, tape.hidden.data(),
          hidden_.data(), tanh_.data(), discard_.data()), "network hidden tanh reverse");
      weightReverse(dense, tape.residual.data(), tanh_.data(), ParameterRole::UPDATE_HIDDEN_WEIGHT, layer);
      biasReverse(dense, tanh_.data(), ParameterRole::UPDATE_HIDDEN_BIAS, layer);
      accumulateDenseInputAdjoint(blas_, dense, tanh_.data(),
          parameter(ParameterRole::UPDATE_HIDDEN_WEIGHT, layer), residual_.data());

      // The attention residual contributes directly to the preceding block input.
      copy(residual_, input_, elements);
      weightReverse(dense, tape.context.data(), residual_.data(), ParameterRole::ATTENTION_OUTPUT_WEIGHT, layer);
      zero(context_, elements);
      accumulateDenseInputAdjoint(blas_, dense, residual_.data(),
          parameter(ParameterRole::ATTENTION_OUTPUT_WEIGHT, layer), context_.data());
      checkRuntime(launchAttentionContextReverse(stream, attention, tape.probabilities.data(),
          tape.value.data(), context_.data(), probability_.data(), value_.data()), "network context reverse");
      checkRuntime(launchSoftmaxScaledQkReverse(stream, attention, tape.query.data(), tape.key.data(),
          tape.probabilities.data(), probability_.data(), query_.data(), key_.data()), "network attention reverse");
      for (const auto& projection : {std::pair{ParameterRole::ATTENTION_QUERY_WEIGHT, &query_},
                                    std::pair{ParameterRole::ATTENTION_KEY_WEIGHT, &key_},
                                    std::pair{ParameterRole::ATTENTION_VALUE_WEIGHT, &value_}})
      {
        weightReverse(dense, tape.input.data(), projection.second->data(), projection.first, layer);
        accumulateDenseInputAdjoint(blas_, dense, projection.second->data(),
            parameter(projection.first, layer), input_.data());
      }
      copy(input_, upstream_, elements);
    }

    // Parameter score stops at geometry; raw-only mode instead applies the final
    // embedding transpose into its separately bounded input-adjoint destination.
    const auto& embedding = model_->plan().parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
    const DenseForwardLayout feature_dense{count * ne, embedding.shape[0], width,
                                          embedding.shape[0], width, width};
    weightReverse(feature_dense, forward.valueRawFeatures(), upstream_.data(),
                  ParameterRole::ELECTRON_EMBEDDING_WEIGHT, NO_ATTENTION_BLOCK);
    const double* result = response_.data();
    auto scan = finite_layout_;
    if (output_ == NetworkReverseOutput::RAW_FEATURE_ADJOINT_ONLY)
    {
      const auto raw_elements = multiply(multiply(count, ne), embedding.shape[0]);
      zero(raw_feature_, raw_elements);
      accumulateDenseInputAdjoint(blas_, feature_dense, upstream_.data(),
          parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT, NO_ATTENTION_BLOCK), raw_feature_.data());
      result = raw_feature_.data();
      scan.element_count = raw_elements;
      scan.status_capacity = raw_elements / scan.elements_per_status +
          (raw_elements % scan.elements_per_status != 0);
    }

    // Reject invalid contractions before publishing the borrowed accumulator.
    checkRuntime(launchDerivativeFiniteScan(stream, scan, result,
        finite_status_.data()), "network response finite scan");
    queue_.sync();
    finite_status_.download(host_finite_status_.data(), scan.status_capacity);
    for (std::size_t index = 0; index < scan.status_capacity; ++index)
      if (host_finite_status_[index].nonfinite_count != 0)
        throw std::runtime_error("Network reverse produced a nonfinite selected output");
    result_count_ = count;
    result_model_version_ = model_->version();
    result_evaluation_generation_ = forward.evaluationGeneration();
    result_ready_ = true;
  }

public:
  /// Return canonical destination extent, zero in raw-feature-only preparation.
  std::size_t parameterCount() const noexcept { return response_.capacity(); }

  /// Return the preparation's output contract without exposing incomplete data.
  NetworkReverseOutput output() const noexcept { return output_; }

  /// Return the live raw-feature view only after successful finite admission.
  RawFeatureAdjointView rawFeatureAdjoints() const
  {
    if (output_ != NetworkReverseOutput::RAW_FEATURE_ADJOINT_ONLY || !result_ready_)
      throw std::logic_error("Network reverse has no completed raw feature adjoints");
    const auto ne = model_->plan().modelShape().electrons();
    const auto width = model_->plan().parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0];
    return {raw_feature_.data(), {result_count_, ne, width, width, ne * width},
        result_model_version_, result_evaluation_generation_};
  }

  /// Copy only the completed live raw-input cotangents for numerical validation.
  void downloadRawFeatureAdjoints(std::vector<double>& destination) const
  {
    const auto view = rawFeatureAdjoints();
    const auto elements = multiply(view.layout.configuration_count, view.layout.configuration_stride);
    if (destination.size() != elements)
      throw std::invalid_argument("Network raw feature destination extent mismatch");
    raw_feature_.download(destination.data(), destination.size());
  }

  /// Rebind a compatible successor without reallocating adjoints or retaining old weights.
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> successor)
  {
    model_->validateSuccessor(successor);
    model_ = std::move(successor);
    result_ready_ = false;
  }

  /// Account the prepared numerical storage, excluding the shared model and BLAS.
  /// Count host-side finite-scan storage retained by this reverse lane.
  std::size_t hostWorkspaceBytes() const
  {
    return checkedDeviceBytes<DerivativeFiniteStatus>(host_finite_status_.capacity());
  }

  /// Count prepared device adjoints, selected output and finite-scan buffers.
  std::size_t deviceWorkspaceBytes() const
  {
    std::size_t bytes = 0;
    for (const auto* buffer : {&upstream_, &residual_, &tanh_, &hidden_, &input_, &context_,
                              &query_, &key_, &value_, &probability_, &discard_, &response_, &raw_feature_})
      bytes = spatial_detail::checkedAdd(bytes, checkedDeviceBytes<double>(buffer->capacity()),
                                         "Network reverse storage overflow");
    return spatial_detail::checkedAdd(bytes,
        checkedDeviceBytes<DerivativeFiniteStatus>(finite_status_.capacity()),
        "Network reverse status storage overflow");
  }

  /// Copy the completed canonical accumulator for numerical validation/host adapters.
  void downloadResponse(std::vector<double>& destination) const
  {
    if (output_ != NetworkReverseOutput::PARAMETER_SCORE)
      throw std::logic_error("Raw-feature-only network reverse has no parameter response");
    if (!result_ready_)
      throw std::logic_error("Network reverse has no completed response");
    if (destination.size() != response_.capacity())
      throw std::invalid_argument("Network reverse destination extent mismatch");
    response_.download(destination.data(), destination.size());
  }

private:
  /// Check extent multiplication before allocation or BLAS layout construction.
  static std::size_t multiply(std::size_t left, std::size_t right)
  { return spatial_detail::checkedProduct(left, right, "Network reverse extent overflow"); }

  /// Bind a canonical immutable weight without inferring tensor adjacency.
  const double* parameter(ParameterRole role, std::size_t layer) const
  { return model_->parameters() + model_->plan().parameter(role, layer).begin; }

  /// Reset a contiguous logical destination on the caller's ordered stream.
  void zero(DeviceBuffer<double>& buffer, std::size_t count)
  { checkRuntime(launchDerivativeZero(queue_.getNative(), {count, 0, 1, 0, 1}, buffer.data()), "network adjoint zero"); }

  /// Copy a contiguous cotangent between preallocated destinations.
  void copy(const DeviceBuffer<double>& source, DeviceBuffer<double>& target, std::size_t count)
  { checkRuntime(launchDerivativeCopy(queue_.getNative(), {count, 0, 1, 0, 1}, source.data(), target.data()), "network adjoint copy"); }

  /// Fold configurations/electrons into one canonical shared-weight contraction.
  void weightReverse(const DenseForwardLayout& layout, const double* source, const double* adjoint,
                     ParameterRole role, std::size_t layer)
  {
    if (output_ == NetworkReverseOutput::RAW_FEATURE_ADJOINT_ONLY)
      return;
    accumulateDenseWeightAdjoint(blas_, layout, source, adjoint,
        response_.data() + model_->plan().parameter(role, layer).begin);
  }

  /// Sum bias cotangents into the exact canonical interval.
  void biasReverse(const DenseForwardLayout& layout, const double* adjoint,
                   ParameterRole role, std::size_t layer)
  {
    if (output_ == NetworkReverseOutput::RAW_FEATURE_ADJOINT_ONLY)
      return;
    checkRuntime(launchDenseBiasAdjoint(queue_.getNative(), layout, adjoint,
        response_.data() + model_->plan().parameter(role, layer).begin), "network bias reverse");
  }

  std::shared_ptr<const PreparedDeviceModel> model_;
  std::size_t capacity_;
  compute::Queue<PlatformKind::CUDA>& queue_;
  AcceleratorBlasHandle& blas_;
  const NetworkReverseOutput output_;
  DeviceBuffer<double> upstream_, residual_, tanh_, hidden_, input_, context_;
  DeviceBuffer<double> query_, key_, value_, probability_, discard_, response_, raw_feature_;
  DerivativeFiniteScanLayout finite_layout_;
  DeviceBuffer<DerivativeFiniteStatus> finite_status_;
  std::vector<DerivativeFiniteStatus> host_finite_status_;
  bool result_ready_ = false;
  std::size_t result_count_ = 0;
  std::size_t result_model_version_ = 0;
  std::size_t result_evaluation_generation_ = 0;
};

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_NETWORK_REVERSE_H
