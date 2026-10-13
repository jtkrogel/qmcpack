//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceInference.h
 * @brief Prepared FP64 open-real forward/VGL graph for opt-in hardware validation.
 *
 * This chains actual CUDA/HIP leaves for a complete imported model. It is not
 * attached to PsiFormerWF's production publication/crowd-resource protocols and
 * must not advertise production DEVICE capability. Parameters are immutable for
 * this instance. Per-stage synchronization deliberately favors diagnostic clarity;
 * performance tuning and production dispatch remain separate acceptance work.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_INFERENCE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_INFERENCE_H

#include "PsiFormerDeviceResources.h"
#include "PsiFormerDeviceSpatial.h"
#include "PsiFormerDeviceDeterminantKernels.h"
#include "PsiFormerDeviceDerivativeKernels.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"
#include "Platforms/CUDA/AccelBLAS_CUDA.hpp"

#include <algorithm>
#include <vector>

namespace qmcplusplus::psiformer::device
{

/** Preallocate one model image and a fixed-capacity configuration-major VGL graph.
 *
 * Feature/attention work stays on the GPU. Explicit packing bridges jet tensor
 * strides to the existing determinant ABI. Evaluation allocates no caller-owned
 * device buffers or per-stage host vectors; vendor BLAS internal workspace still
 * requires runtime profiling. Stream/BLAS ownership remains with the caller.
 */
class PreparedDeviceInference
{
public:
  /// Validate the open-real model and prepare all storage before numerical launches.
  PreparedDeviceInference(const PsiFormerExecutionPlan& plan,
                          const std::vector<double>& parameters,
                          const std::vector<double>& nuclei,
                          std::size_t batch_size,
                          compute::Queue<PlatformKind::CUDA>& queue,
                          AcceleratorBlasHandle& blas)
      : plan_(plan), shape_(plan.modelShape()), batch_(batch_size), queue_(queue), blas_(blas)
  {
    const auto& environment = plan_.environment();
    if (batch_ == 0 || environment.boundary != BoundaryCondition::OPEN ||
        environment.parameter_scalar_domain != ScalarDomain::REAL ||
        environment.compute_scalar_domain != ScalarDomain::REAL ||
        environment.amplitude_scalar_domain != ScalarDomain::REAL || !environment.fixed_nuclei)
      throw std::invalid_argument("Device inference validation supports fixed-ion open-real FP64 models only");
    if (parameters.size() != plan_.parameterCount() || nuclei.size() != 3 * shape_.nuclei)
      throw std::invalid_argument("Device inference parameter/nuclear extent mismatch");
    ne_ = shape_.electrons();
    const std::size_t width = shape_.feature_dimension;
    features_ = makeOpenFeatureJetLayout(batch_, ne_, shape_.nuclei, shape_.spin_up_electrons, mode_);
    const auto& embedding = plan_.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
    if (embedding.shape[0] != features_.feature_width)
      throw std::invalid_argument("Device inference feature contract mismatch");
    embedding_ = makeSpatialDenseJetLayout(batch_, ne_, features_.feature_width, width, ne_, mode_);
    dense_ = makeSpatialDenseJetLayout(batch_, ne_, width, width, ne_, mode_);
    elementwise_ = makeSpatialElementwiseJetLayout(batch_, ne_, width, ne_, mode_);
    attention_ = makeSpatialAttentionJetLayout(batch_, ne_, shape_.attention_heads,
                                               width / shape_.attention_heads, ne_, mode_);
    orbitals_ = makeOpenOrbitalJetLayout(batch_, shape_.determinants, ne_, shape_.nuclei,
                                         width, shape_.spin_up_electrons, mode_);
    cusp_ = makeOpenCuspJetLayout(batch_, ne_, shape_.spin_up_electrons, mode_);
    final_ = makeFinalSpatialJetLayout(batch_, ne_, mode_);
    matrices_per_configuration_ = orbitals_.orbitals.element_count;
    gradient_lanes_ = dense_.source.gradient_lanes;

    params_.reserve(parameters.size()); params_.upload(parameters.data(), parameters.size());
    nuclei_.reserve(nuclei.size()); nuclei_.upload(nuclei.data(), nuclei.size());
    positions_.reserve(product(batch_, product(ne_, 3)));
    active_.reserve(batch_);
    const std::vector<std::size_t> zero_active(batch_, 0);
    active_.upload(zero_active.data(), batch_);
    raw_.reserve(spatialJetSpanElements(features_.output));
    const auto feature_elements = spatialJetSpanElements(dense_.source);
    for (DeviceBuffer<double>* buffer : {&a_, &b_, &q_, &k_, &v_, &context_, &hidden_, &projected_})
      buffer->reserve(feature_elements);
    weights_.reserve(spatialJetSpanElements(attention_.attention));
    orbital_jets_.reserve(spatialJetSpanElements(orbitals_.orbitals));
    matrices_.reserve(product(batch_, matrices_per_configuration_));
    gradients_.reserve(product(matrices_.capacity(), gradient_lanes_));
    laplacians_.reserve(product(matrices_.capacity(), ne_));
    lu_.reserve(matrices_.capacity()); inverse_.reserve(matrices_.capacity());
    permutations_.reserve(product(product(batch_, shape_.determinants), ne_));
    solve_.reserve(permutations_.capacity());
    factors_.reserve(product(batch_, shape_.determinants));
    combinations_.reserve(batch_);
    for (DeviceBuffer<double>* buffer : {&term_phase_, &term_log_, &scaled_, &normalized_})
      buffer->reserve(factors_.capacity());
    matrix_product_.reserve(product(batch_, product(ne_, ne_)));
    det_gradient_.reserve(product(batch_, gradient_lanes_));
    det_lap_ratio_.reserve(product(batch_, ne_)); det_lap_log_.reserve(product(batch_, ne_));
    det_status_.reserve(batch_);
    cusp_jets_.reserve(spatialJetSpanElements(cusp_.output));
    phase_.reserve(batch_); output_.reserve(spatialJetSpanElements(final_.output));
    lap_ratio_.reserve(product(batch_, ne_));
    const auto status_count = std::max(product(batch_, product(ne_, width)), matrices_.capacity());
    open_status_.reserve(status_count); host_status_.resize(status_count);
    jet_status_.reserve(product(batch_, product(shape_.attention_heads, ne_)));
    host_jet_status_.resize(jet_status_.capacity());

    same_alpha_ = 1;
    if (const auto* same = plan_.optionalParameter(ParameterRole::CUSP_SAME_ALPHA))
      same_alpha_ = parameters[same->begin];
    opposite_alpha_ = parameters[plan_.parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin];
  }

  /// Submit feature layers, remaining network layers, orbital/determinant and cusp VGL.
  void evaluate(const std::vector<double>& positions)
  {
    if (positions.size() != positions_.capacity())
      throw std::invalid_argument("Device inference positions have wrong fixed batch extent");
    positions_.upload(positions.data(), positions.size());
    const auto stream = queue_.getNative();

    // Feature layers: physical radial/Cartesian jets followed by electron embedding.
    checkRuntime(launchOpenFeatureJets(stream, features_, positions_.data(), nuclei_.data(),
                                       active_.data(), raw_.data(), open_status_.data()), "model feature VGL");
    checkOpen(batch_);
    denseJetsForward(blas_, embedding_, raw_.data(), parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT), a_.data());

    // Remaining neural layers: four validated attention/residual/MLP blocks.
    for (std::size_t layer = 0; layer < shape_.attention_blocks; ++layer)
    {
      projectQkvJetsForward(blas_, dense_, a_.data(),
          parameter(ParameterRole::ATTENTION_QUERY_WEIGHT, layer),
          parameter(ParameterRole::ATTENTION_KEY_WEIGHT, layer),
          parameter(ParameterRole::ATTENTION_VALUE_WEIGHT, layer), q_.data(), k_.data(), v_.data());
      checkRuntime(launchAttentionJetWeights(stream, attention_, q_.data(), k_.data(),
                                             weights_.data(), jet_status_.data()), "model attention VGL");
      checkJets();
      checkRuntime(launchAttentionContextJets(stream, attention_, weights_.data(),
                                              v_.data(), context_.data()), "model attention context");
      denseJetsForward(blas_, dense_, context_.data(), parameter(ParameterRole::ATTENTION_OUTPUT_WEIGHT, layer), b_.data());
      residual(a_.data(), b_.data(), b_.data());
      denseJetsForward(blas_, dense_, b_.data(), parameter(ParameterRole::UPDATE_HIDDEN_WEIGHT, layer), hidden_.data());
      activate(hidden_.data(), parameter(ParameterRole::UPDATE_HIDDEN_BIAS, layer));
      denseJetsForward(blas_, dense_, hidden_.data(), parameter(ParameterRole::UPDATE_OUTPUT_WEIGHT, layer), projected_.data());
      activate(projected_.data(), parameter(ParameterRole::UPDATE_OUTPUT_BIAS, layer));
      residual(b_.data(), projected_.data(), a_.data());
    }

    // Spin-conditioned orbitals and learned exponential envelopes remain device-side.
    checkRuntime(launchOpenOrbitalJets(stream, orbitals_, a_.data(), positions_.data(), nuclei_.data(),
        parameter(ParameterRole::BACKFLOW_UP_WEIGHT), parameter(ParameterRole::BACKFLOW_DOWN_WEIGHT),
        parameter(ParameterRole::ENVELOPE_PI_UP), parameter(ParameterRole::ENVELOPE_PI_DOWN),
        parameter(ParameterRole::ENVELOPE_ZETA_UP), parameter(ParameterRole::ENVELOPE_ZETA_DOWN),
        active_.data(), orbital_jets_.data(), open_status_.data()), "model orbital VGL");
    checkOpen(matrices_.capacity());

    // Pack existing jet planes into the contiguous determinant ABI, without host math.
    for (std::size_t configuration = 0; configuration < batch_; ++configuration)
    {
      const auto origin = configuration * orbitals_.orbitals.configuration_stride;
      copy(orbital_jets_, origin, matrices_, configuration * matrices_per_configuration_, matrices_per_configuration_);
      copy(orbital_jets_, origin + orbitals_.orbitals.plane_stride, gradients_,
           configuration * gradient_lanes_ * matrices_per_configuration_, gradient_lanes_ * matrices_per_configuration_);
      copy(orbital_jets_, origin + (1 + gradient_lanes_) * orbitals_.orbitals.plane_stride, laplacians_,
           configuration * ne_ * matrices_per_configuration_, ne_ * matrices_per_configuration_);
    }
    checkRuntime(launchDeterminantFactorization(stream, matrices_.data(), batch_, shape_.determinants, ne_,
        true, lu_.data(), inverse_.data(), permutations_.data(), solve_.data(), factors_.data()), "model determinants");
    checkRuntime(launchDeterminantCombination(stream, factors_.data(), nullptr, batch_, shape_.determinants,
        term_phase_.data(), term_log_.data(), scaled_.data(), normalized_.data(), combinations_.data()), "model determinant sum");
    device_determinant::SpatialLayout determinant_layout{batch_, shape_.determinants, ne_, gradient_lanes_, ne_};
    checkRuntime(launchDeterminantSpatialTraces(stream, determinant_layout, factors_.data(), combinations_.data(),
        normalized_.data(), inverse_.data(), gradients_.data(), laplacians_.data(), matrix_product_.data(),
        det_gradient_.data(), det_lap_ratio_.data(), det_lap_log_.data(), det_status_.data()), "model determinant spatial traces");
    checkRuntime(launchOpenCuspJets(stream, cusp_, positions_.data(), active_.data(), same_alpha_, opposite_alpha_,
                                    cusp_jets_.data(), open_status_.data()), "model cusp VGL");
    checkOpen(batch_);
    checkRuntime(launchFinalSpatialCombination(stream, final_, combinations_.data(), det_status_.data(),
        det_gradient_.data(), det_lap_log_.data(), cusp_jets_.data(), phase_.data(), output_.data(),
        lap_ratio_.data(), open_status_.data()), "model final VGL");
    checkOpen(batch_);
  }

  /// Download signed-log amplitude, complete log-gradient/log-Laplacian jets and lap ratios.
  void downloadResults(std::vector<double>& phases, std::vector<double>& jets, std::vector<double>& ratios) const
  {
    if (phases.size() != phase_.capacity() || jets.size() != output_.capacity() || ratios.size() != lap_ratio_.capacity())
      throw std::invalid_argument("Device inference result buffers have wrong extent");
    phase_.download(phases.data(), phases.size());
    output_.download(jets.data(), jets.size());
    lap_ratio_.download(ratios.data(), ratios.size());
  }

  /// Expose the fixed output layout for result comparison, not internal scratch.
  const SpatialJetLayout& resultLayout() const noexcept { return final_.output; }

private:
  /// Keep multiplication overflow checks at preparation boundaries.
  static std::size_t product(std::size_t left, std::size_t right)
  { return spatial_detail::checkedProduct(left, right, "Device inference extent overflow"); }

  /// Bind a canonical device parameter interval without hot-path string lookup.
  const double* parameter(ParameterRole role, std::size_t layer = NO_ATTENTION_BLOCK) const
  { return params_.data() + plan_.parameter(role, layer).begin; }

  /// Check each stage before a later launch can overwrite its error diagnostics.
  void checkOpen(std::size_t count)
  {
    queue_.sync();
    open_status_.download(host_status_.data(), count);
    for (std::size_t i = 0; i < count; ++i)
      if (host_status_[i] != OpenSpatialStatus::REGULAR)
        throw std::runtime_error("Device inference stage has nonregular spatial status");
  }

  /// Reject invalid softmax jets before using them as context weights.
  void checkJets()
  {
    queue_.sync();
    jet_status_.download(host_jet_status_.data(), host_jet_status_.size());
    for (auto status : host_jet_status_)
      if (status != device_math::JetMathStatus::REGULAR)
        throw std::runtime_error("Device inference attention has nonregular jet status");
  }

  /// Apply a bias/tanh transformation in place with preallocated diagnostics.
  void activate(double* values, const double* bias)
  {
    checkRuntime(launchBiasTanhJets(queue_.getNative(), elementwise_, values, bias, values,
                                    open_status_.data()), "model bias/tanh");
    checkOpen(batch_ * ne_ * shape_.feature_dimension);
  }

  /// Apply a residual sum, permitting exact pointwise input/output aliasing.
  void residual(const double* left, const double* right, double* destination)
  {
    checkRuntime(launchResidualSpatialJets(queue_.getNative(), elementwise_, left, right, destination,
                                           open_status_.data()), "model residual");
    checkOpen(batch_ * ne_ * shape_.feature_dimension);
  }

  /// Pack one contiguous jet interval into already prepared determinant storage.
  void copy(const DeviceBuffer<double>& source, std::size_t source_offset,
            DeviceBuffer<double>& target, std::size_t target_offset, std::size_t count)
  {
    DerivativeVectorLayout layout{count, source_offset, 1, target_offset, 1};
    checkRuntime(launchDerivativeCopy(queue_.getNative(), layout, source.data(), target.data()), "model jet packing");
  }

  PsiFormerExecutionPlan plan_;
  ModelShape shape_;
  std::size_t batch_, ne_ = 0, gradient_lanes_ = 0, matrices_per_configuration_ = 0;
  compute::Queue<PlatformKind::CUDA>& queue_;
  AcceleratorBlasHandle& blas_;
  const SpatialJetMode mode_ = SpatialJetMode::FULL_VGL;
  OpenFeatureJetLayout features_;
  SpatialDenseJetLayout embedding_, dense_;
  SpatialElementwiseJetLayout elementwise_;
  SpatialAttentionJetLayout attention_;
  OpenOrbitalJetLayout orbitals_;
  OpenCuspJetLayout cusp_;
  FinalSpatialJetLayout final_;
  double same_alpha_ = 1, opposite_alpha_ = 1;
  DeviceBuffer<double> params_, nuclei_, positions_, raw_, a_, b_, q_, k_, v_, weights_, context_, hidden_, projected_;
  DeviceBuffer<double> orbital_jets_, matrices_, gradients_, laplacians_, lu_, inverse_, solve_;
  DeviceBuffer<double> term_phase_, term_log_, scaled_, normalized_, matrix_product_;
  DeviceBuffer<double> det_gradient_, det_lap_ratio_, det_lap_log_, cusp_jets_, phase_, output_, lap_ratio_;
  DeviceBuffer<std::size_t> active_, permutations_;
  DeviceBuffer<device_determinant::FactorizationMetadata> factors_;
  DeviceBuffer<device_determinant::CombinationMetadata> combinations_;
  DeviceBuffer<device_determinant::DerivativeStatus> det_status_;
  DeviceBuffer<OpenSpatialStatus> open_status_;
  DeviceBuffer<device_math::JetMathStatus> jet_status_;
  std::vector<OpenSpatialStatus> host_status_;
  std::vector<device_math::JetMathStatus> host_jet_status_;
};

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_INFERENCE_H
