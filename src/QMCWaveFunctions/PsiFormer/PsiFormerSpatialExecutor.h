//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerSpatialExecutor.h
 * @brief Allocation-free direct spatial derivatives for molecular PsiFormer models.
 *
 * The executor propagates Cartesian gradients and per-electron Laplacian traces
 * directly through the embedding, attention, orbital, determinant, and cusp layers.
 * It never forms an electron Hessian: a full evaluation stores 3*Ne first-derivative
 * lanes and Ne Laplacian lanes, while an active-electron evaluation stores only three
 * first-derivative lanes.  All storage is allocated with the clone-local workspace.
 *
 * Geometry remains real and open-boundary.  Results use sign/log-magnitude form so a
 * future complex phase policy can be added without changing the spatial-output API;
 * periodic and complex numerical kernels are intentionally not implemented here.
 */

//////////////////////////////////////////////////////////////////////////////////////
// INCLUSION RESTRICTION
// PsiFormerNative.h defines non-inline functions and is included by exactly one QMCPACK
// translation unit.  Include this helper after PsiFormerNative.h and
// PsiFormerValueExecutor.h in that same translation unit.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_PSIFORMER_VALUE_EXECUTOR_H
#error "Include PsiFormerValueExecutor.h before PsiFormerSpatialExecutor.h"
#endif

#ifndef QMCPLUSPLUS_PSIFORMER_SPATIAL_EXECUTOR_H
#define QMCPLUSPLUS_PSIFORMER_SPATIAL_EXECUTOR_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace pf
{

/// Select either the three-lane active-electron path or complete molecular VGL.
enum class DirectSpatialMode
{
  ACTIVE_ELECTRON_GRADIENT,
  FULL_VGL
};

/// Provide a non-owning read-only view whose lifetime is tied to its workspace.
class DirectConstDoubleView
{
public:
  /// Construct an empty result view.
  DirectConstDoubleView() = default;

  /// Construct a view over a contiguous immutable interval.
  DirectConstDoubleView(const double* data, std::size_t size) : data_(data), size_(size) {}

  /// Return the number of scalar values in the interval.
  std::size_t size() const noexcept { return size_; }

  /// Report whether the interval is empty.
  bool empty() const noexcept { return size_ == 0; }

  /// Access one result scalar without bounds checking.
  const double& operator[](std::size_t index) const noexcept { return data_[index]; }

  /// Return the first scalar address.
  const double* data() const noexcept { return data_; }

  /// Return an iterator to the first scalar.
  const double* begin() const noexcept { return data_; }

  /// Return an iterator one past the final scalar.
  const double* end() const noexcept { return size_ == 0 ? data_ : data_ + size_; }

private:
  const double* data_ = nullptr;
  std::size_t size_   = 0;
};

/** Hold a tensor value, Cartesian gradients, and per-electron Laplacian traces.
 *
 * Gradient and Laplacian arrays are lane-major.  Their vector sizes are fixed at
 * workspace construction; clearing or swapping a buffer cannot allocate.
 */
struct DirectSpatialJetBuffer
{
  /// Allocate exact fixed capacities for one tensor shape and derivative mode.
  DirectSpatialJetBuffer(std::size_t value_size,
                         std::size_t gradient_lanes,
                         std::size_t laplacian_lanes)
      : value(value_size),
        gradient(value_size * gradient_lanes),
        laplacian(value_size * laplacian_lanes)
  {}

  /// Reset all values and derivative lanes before overwriting a tensor.
  void clear()
  {
    std::fill(value.begin(), value.end(), 0.0);
    std::fill(gradient.begin(), gradient.end(), 0.0);
    std::fill(laplacian.begin(), laplacian.end(), 0.0);
  }

  std::vector<double> value;
  std::vector<double> gradient;
  std::vector<double> laplacian;
};

/** Own all mutable storage for one direct spatial evaluator clone.
 *
 * FULL_VGL workspaces carry a 3-vector gradient and scalar Laplacian for every
 * differentiating electron.  ACTIVE_ELECTRON_GRADIENT workspaces omit every
 * second-derivative allocation and can be reused for any active electron.
 */
class DirectSpatialWorkspace
{
public:
  /// Allocate fixed geometry, forward-jet, determinant, and output buffers.
  DirectSpatialWorkspace(const DirectValueParameterLayout& layout,
                         GeometryPositionView nuclei,
                         DirectSpatialMode mode,
                         GeometryBoundary boundary = {})
      : mode_(mode),
        gradient_lanes_(mode == DirectSpatialMode::FULL_VGL ? 3 * layout.electronCount() : 3),
        laplacian_lanes_(mode == DirectSpatialMode::FULL_VGL ? layout.electronCount() : 0),
        electron_positions_(3 * layout.electronCount()),
        geometry_(layout.electronCount(), nuclei, boundary),
        raw_features_(layout.electronCount() * layout.inputWidth(), gradient_lanes_, laplacian_lanes_),
        features_a_(layout.electronCount() * layout.featureWidth(), gradient_lanes_, laplacian_lanes_),
        features_b_(layout.electronCount() * layout.featureWidth(), gradient_lanes_, laplacian_lanes_),
        query_(layout.electronCount() * layout.featureWidth(), gradient_lanes_, laplacian_lanes_),
        key_(layout.electronCount() * layout.featureWidth(), gradient_lanes_, laplacian_lanes_),
        projected_value_(layout.electronCount() * layout.featureWidth(), gradient_lanes_, laplacian_lanes_),
        attention_(layout.headCount() * layout.electronCount() * layout.electronCount(),
                   gradient_lanes_, laplacian_lanes_),
        attended_(layout.electronCount() * layout.featureWidth(), gradient_lanes_, laplacian_lanes_),
        hidden_(layout.electronCount() * layout.featureWidth(), gradient_lanes_, laplacian_lanes_),
        orbital_matrices_(layout.determinantCount() * layout.electronCount() * layout.electronCount(),
                          gradient_lanes_, laplacian_lanes_),
        determinant_workspace_(layout.determinantCount(), layout.electronCount(),
                               gradient_lanes_, laplacian_lanes_),
        scalar_gradient_scratch_(gradient_lanes_),
        scalar_laplacian_scratch_(laplacian_lanes_),
        output_gradient_(gradient_lanes_),
        output_lap_log_(laplacian_lanes_),
        output_lap_ratio_(laplacian_lanes_)
  {}

  DirectSpatialWorkspace(const DirectSpatialWorkspace&) = delete;
  DirectSpatialWorkspace& operator=(const DirectSpatialWorkspace&) = delete;
  DirectSpatialWorkspace(DirectSpatialWorkspace&&) = default;
  DirectSpatialWorkspace& operator=(DirectSpatialWorkspace&&) = default;

  /// Copy a complete interleaved configuration into fixed input storage.
  void setPositions(GeometryPositionView positions)
  {
    if (positions.size() != geometry_.electronCount())
      throw std::invalid_argument("PsiFormer spatial workspace received the wrong electron count");
    for (std::size_t electron = 0; electron < positions.size(); ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        electron_positions_[3 * electron + dimension] = positions(electron, dimension);
  }

  /// Overwrite one Cartesian coordinate without changing workspace capacity.
  void setPosition(std::size_t electron, std::size_t dimension, double value)
  {
    if (electron >= geometry_.electronCount() || dimension >= 3)
      throw std::out_of_range("PsiFormer spatial workspace coordinate index is out of range");
    electron_positions_[3 * electron + dimension] = value;
  }

  /// Return the derivative storage mode fixed at workspace construction.
  DirectSpatialMode mode() const noexcept { return mode_; }

  /// Return the parameter version used by the most recent completed evaluation.
  std::size_t observedParameterVersion() const noexcept { return observed_parameter_version_; }

  /** Hash every backing address and capacity for a no-growth regression check.
   *
   * This is a storage-identity diagnostic, not a persistent model fingerprint.
   */
  std::size_t storageFingerprint() const noexcept
  {
    std::size_t hash = 1469598103934665603ULL;
    auto mix = [&hash](const std::vector<double>& buffer) {
      hash ^= reinterpret_cast<std::uintptr_t>(buffer.data());
      hash *= 1099511628211ULL;
      hash ^= buffer.capacity();
      hash *= 1099511628211ULL;
    };
    mix(electron_positions_);
    hash ^= geometry_.storageFingerprint();
    hash *= 1099511628211ULL;
    auto mix_jet = [&mix](const DirectSpatialJetBuffer& buffer) {
      mix(buffer.value);
      mix(buffer.gradient);
      mix(buffer.laplacian);
    };
    auto jet_fingerprint = [](const DirectSpatialJetBuffer& buffer) {
      std::size_t jet_hash = 1469598103934665603ULL;
      auto mix_buffer = [&jet_hash](const std::vector<double>& values) {
        jet_hash ^= reinterpret_cast<std::uintptr_t>(values.data());
        jet_hash *= 1099511628211ULL;
        jet_hash ^= values.capacity();
        jet_hash *= 1099511628211ULL;
      };
      mix_buffer(buffer.value);
      mix_buffer(buffer.gradient);
      mix_buffer(buffer.laplacian);
      return jet_hash;
    };
    mix_jet(raw_features_);
    // Attention layers swap the two ping-pong buffer objects.  Hash their
    // allocation identities as an unordered pair so odd block counts do not
    // report a false allocation change after every successful evaluation.
    const std::size_t feature_a_fingerprint = jet_fingerprint(features_a_);
    const std::size_t feature_b_fingerprint = jet_fingerprint(features_b_);
    hash ^= std::min(feature_a_fingerprint, feature_b_fingerprint);
    hash *= 1099511628211ULL;
    hash ^= std::max(feature_a_fingerprint, feature_b_fingerprint);
    hash *= 1099511628211ULL;
    mix_jet(query_);
    mix_jet(key_);
    mix_jet(projected_value_);
    mix_jet(attention_);
    mix_jet(attended_);
    mix_jet(hidden_);
    mix_jet(orbital_matrices_);
    hash ^= determinant_workspace_.storageFingerprint();
    hash *= 1099511628211ULL;
    mix(scalar_gradient_scratch_);
    mix(scalar_laplacian_scratch_);
    mix(output_gradient_);
    mix(output_lap_log_);
    mix(output_lap_ratio_);
    return hash;
  }

  /// Return bytes reserved by every workspace buffer, including geometry tables.
  std::size_t vectorStorageBytes() const noexcept
  {
    std::size_t scalar_capacity = 0;
    auto add = [&scalar_capacity](const std::vector<double>& buffer) {
      scalar_capacity += buffer.capacity();
    };
    auto add_jet = [&add](const DirectSpatialJetBuffer& buffer) {
      add(buffer.value);
      add(buffer.gradient);
      add(buffer.laplacian);
    };
    add(electron_positions_);
    add_jet(raw_features_);
    add_jet(features_a_);
    add_jet(features_b_);
    add_jet(query_);
    add_jet(key_);
    add_jet(projected_value_);
    add_jet(attention_);
    add_jet(attended_);
    add_jet(hidden_);
    add_jet(orbital_matrices_);
    add(scalar_gradient_scratch_);
    add(scalar_laplacian_scratch_);
    add(output_gradient_);
    add(output_lap_log_);
    add(output_lap_ratio_);
    return scalar_capacity * sizeof(double) + geometry_.storageBytes() +
        determinant_workspace_.storageBytes();
  }

  /// Expose geometry-cache allocation accounting for workspace diagnostics.
  std::size_t geometryStorageBytes() const noexcept
  { return geometry_.storageBytes(); }

  /// Expose geometry allocation identity for focused storage diagnostics.
  std::size_t geometryStorageFingerprint() const noexcept
  { return geometry_.storageFingerprint(); }

private:
  friend class DirectSpatialExecutor;
  friend class DirectBatchExecutor;

  DirectSpatialMode mode_;
  std::size_t gradient_lanes_;
  std::size_t laplacian_lanes_;
  std::size_t active_electron_ = 0;
  std::vector<double> electron_positions_;
  PsiFormerGeometryCache geometry_;
  DirectSpatialJetBuffer raw_features_;
  DirectSpatialJetBuffer features_a_;
  DirectSpatialJetBuffer features_b_;
  DirectSpatialJetBuffer query_;
  DirectSpatialJetBuffer key_;
  DirectSpatialJetBuffer projected_value_;
  DirectSpatialJetBuffer attention_;
  DirectSpatialJetBuffer attended_;
  DirectSpatialJetBuffer hidden_;
  DirectSpatialJetBuffer orbital_matrices_;
  qmcplusplus::psiformer::determinant::RealOpenDeterminantWorkspace determinant_workspace_;
  std::vector<double> scalar_gradient_scratch_;
  std::vector<double> scalar_laplacian_scratch_;
  std::vector<double> output_gradient_;
  std::vector<double> output_lap_log_;
  std::vector<double> output_lap_ratio_;
  std::size_t observed_parameter_version_ = std::numeric_limits<std::size_t>::max();
};

/** Return direct spatial outputs as non-owning views into a reusable workspace. */
struct DirectSpatialResultView
{
  double sign   = 1;
  double logabs = 0;
  double value  = 0;
  DirectSpatialMode mode = DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT;
  DirectConstDoubleView gradient;
  DirectConstDoubleView lap_log;
  DirectConstDoubleView lap_ratio;
  std::size_t parameter_version = 0;
};

/** Evaluate real open-boundary PsiFormer spatial observables with direct kernels. */
class DirectSpatialExecutor
{
public:
  /// Resolve immutable parameter descriptors and retain the versioned value store.
  DirectSpatialExecutor(const PsiFormer& model,
                        const DirectValueExecutor& value_executor,
                        const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan)
      : parameters_(model.p),
        layout_(value_executor.layout()),
        nuclei_(model.cfg.nuclei.x),
        boundary_(directGeometryBoundary(plan.environment().boundary)),
        value_executor_identity_(&value_executor)
  {
    if (layout_->parameterCount() != plan.parameterCount() ||
        layout_->electronCount() != plan.modelShape().electrons())
      throw std::invalid_argument("PsiFormer spatial executor received an incompatible value layout");
    if (value_executor.parameterStoreIdentity() != &model.p)
      throw std::invalid_argument(
          "PsiFormer spatial executor and value executor use different parameter stores");
    const GeometryPositionView value_nuclei = value_executor.nuclearPositions();
    if (value_nuclei.size() * 3 != nuclei_.size())
      throw std::invalid_argument(
          "PsiFormer spatial executor and value executor use different nuclei");
    for (std::size_t nucleus = 0; nucleus < value_nuclei.size(); ++nucleus)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        if (value_nuclei(nucleus, dimension) != nuclei_[3 * nucleus + dimension])
          throw std::invalid_argument(
              "PsiFormer spatial executor and value executor use different nuclei");
    const GeometryBoundary& value_boundary = value_executor.boundary();
    if (value_boundary.kind != boundary_.kind ||
        value_boundary.lattice_vectors != boundary_.lattice_vectors ||
        value_boundary.periodic_axes != boundary_.periodic_axes)
      throw std::invalid_argument(
          "PsiFormer spatial executor and value executor use different boundaries");
  }

  /// Construct one clone-local workspace for a fixed spatial output family.
  std::unique_ptr<DirectSpatialWorkspace> makeWorkspace(DirectSpatialMode mode) const
  {
    return std::make_unique<DirectSpatialWorkspace>(
        *layout_, GeometryPositionView::interleaved(nuclei_.data(), layout_->nucleusCount()), mode, boundary_);
  }

  /// Expose the immutable descriptor identity used to bind batch executors safely.
  const std::shared_ptr<const DirectValueParameterLayout>& layout() const noexcept
  { return layout_; }

  /// Return the exact value executor whose immutable model state is shared here.
  const DirectValueExecutor* valueExecutorIdentity() const noexcept
  { return value_executor_identity_; }

  /// Evaluate complete log-gradient and per-electron log-Laplacian data.
  DirectSpatialResultView evaluateFull(DirectSpatialWorkspace& workspace) const
  {
    if (workspace.mode_ != DirectSpatialMode::FULL_VGL)
      throw std::invalid_argument("PsiFormer full VGL requires a FULL_VGL workspace");
    workspace.active_electron_ = 0;
    return evaluateImpl(workspace);
  }

  /// Evaluate only the three Cartesian log-gradient components of one electron.
  DirectSpatialResultView evaluateActive(DirectSpatialWorkspace& workspace,
                                         std::size_t active_electron) const
  {
    if (workspace.mode_ != DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT)
      throw std::invalid_argument("PsiFormer active gradient requires an active-gradient workspace");
    if (active_electron >= layout_->electronCount())
      throw std::out_of_range("PsiFormer active electron is out of range");
    workspace.active_electron_ = active_electron;
    return evaluateImpl(workspace);
  }

private:
  friend class DirectBatchExecutor;

  /// Return the beginning of one pre-resolved immutable parameter interval.
  static const double* tensor(const double* parameters,
                              const DirectParameterTensor& descriptor) noexcept
  {
    return parameters + descriptor.begin;
  }

  /// Return the physical electron associated with one Cartesian gradient lane.
  static std::size_t laneElectron(const DirectSpatialWorkspace& workspace,
                                  std::size_t lane) noexcept
  {
    return workspace.mode_ == DirectSpatialMode::FULL_VGL ? lane / 3 : workspace.active_electron_;
  }

  /// Return the x/y/z component associated with one Cartesian gradient lane.
  static std::size_t laneDimension(const DirectSpatialWorkspace& workspace,
                                   std::size_t lane) noexcept
  {
    return workspace.mode_ == DirectSpatialMode::FULL_VGL ? lane % 3 : lane;
  }

  /// Index one lane-major gradient element.
  static std::size_t gradientIndex(const DirectSpatialJetBuffer& buffer,
                                   std::size_t lane,
                                   std::size_t element) noexcept
  {
    return lane * buffer.value.size() + element;
  }

  /// Index one electron-major Laplacian element.
  static std::size_t laplacianIndex(const DirectSpatialJetBuffer& buffer,
                                    std::size_t electron,
                                    std::size_t element) noexcept
  {
    return electron * buffer.value.size() + element;
  }

  /// Execute a complete direct forward pass in the derivative mode of the workspace.
  DirectSpatialResultView evaluateImpl(DirectSpatialWorkspace& workspace) const;

  /// Refresh fixed-size open-boundary pair tables from the current input positions.
  void refreshGeometry(DirectSpatialWorkspace& workspace) const;

  /// Build analytic electron-nucleus features and their spatial derivatives.
  void buildEmbedding(const double* parameters, DirectSpatialWorkspace& workspace) const;

  /// Populate the embedding input jet without applying its shared dense projection.
  void buildEmbeddingFeatures(DirectSpatialWorkspace& workspace) const;

  /// Apply one attention/residual block to value, gradient, and Laplacian lanes.
  void applyAttentionBlock(const double* parameters,
                           const DirectValueParameterLayout::DirectAttentionLayer& layer,
                           DirectSpatialWorkspace& workspace) const;

  /// Form determinant orbital matrices including spatial derivative products.
  void buildOrbitalMatrices(const double* parameters, DirectSpatialWorkspace& workspace) const;

  /// Add analytic electron-electron cusp value and spatial derivatives in place.
  double accumulateCusp(const double* parameters, DirectSpatialWorkspace& workspace) const;

  /// Apply one row-major dense layer independently to every derivative lane.
  static void denseJet(const DirectSpatialJetBuffer& source,
                       const double* weight,
                       const double* bias,
                       std::size_t rows,
                       std::size_t input_width,
                       std::size_t output_width,
                       std::size_t gradient_lanes,
                       std::size_t laplacian_lanes,
                       DirectSpatialJetBuffer& target);

  /// Project one feature tensor into query, key, and value jets in one traversal.
  static void denseQKVJet(const DirectSpatialJetBuffer& source,
                          const double* query_weight,
                          const double* key_weight,
                          const double* value_weight,
                          std::size_t rows,
                          std::size_t width,
                          std::size_t gradient_lanes,
                          std::size_t laplacian_lanes,
                          DirectSpatialJetBuffer& query,
                          DirectSpatialJetBuffer& key,
                          DirectSpatialJetBuffer& value);

  /// Apply tanh and its gradient/Laplacian chain rules in fixed storage.
  static void tanhJetInPlace(DirectSpatialJetBuffer& buffer,
                             std::size_t gradient_lanes,
                             std::size_t laplacian_lanes);

  /// Build stable softmax attention values and derivative traces.
  void buildAttentionWeights(DirectSpatialWorkspace& workspace) const;

  /// Contract attention weights with value heads using spatial product rules.
  void buildAttentionContext(DirectSpatialWorkspace& workspace) const;

  const Parameters& parameters_;
  std::shared_ptr<const DirectValueParameterLayout> layout_;
  std::vector<double> nuclei_;
  GeometryBoundary boundary_;
  const DirectValueExecutor* value_executor_identity_;
};

inline void DirectSpatialExecutor::refreshGeometry(DirectSpatialWorkspace& workspace) const
{
  workspace.geometry_.update(
      GeometryPositionView::interleaved(workspace.electron_positions_.data(), layout_->electronCount()));
}

inline void DirectSpatialExecutor::denseJet(const DirectSpatialJetBuffer& source,
                                            const double* weight,
                                            const double* bias,
                                            std::size_t rows,
                                            std::size_t input_width,
                                            std::size_t output_width,
                                            std::size_t gradient_lanes,
                                            std::size_t laplacian_lanes,
                                            DirectSpatialJetBuffer& target)
{
  target.clear();
  for (std::size_t row = 0; row < rows; ++row)
  {
    double* output_row = target.value.data() + row * output_width;
    if (bias)
      std::copy(bias, bias + output_width, output_row);

    for (std::size_t input = 0; input < input_width; ++input)
    {
      const double* weight_row = weight + input * output_width;
      const std::size_t source_element = row * input_width + input;
      for (std::size_t output = 0; output < output_width; ++output)
        output_row[output] += source.value[source_element] * weight_row[output];
    }
  }

  // Derivative planes are lane-major; contracting one complete plane at a time
  // gives each innermost loop contiguous target and weight access.
  for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t input = 0; input < input_width; ++input)
      {
        const double source_derivative =
            source.gradient[lane * source.value.size() + row * input_width + input];
        const double* weight_row = weight + input * output_width;
        double* target_row = target.gradient.data() + lane * target.value.size() + row * output_width;
        for (std::size_t output = 0; output < output_width; ++output)
          target_row[output] += source_derivative * weight_row[output];
      }

  for (std::size_t electron = 0; electron < laplacian_lanes; ++electron)
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t input = 0; input < input_width; ++input)
      {
        const double source_derivative =
            source.laplacian[electron * source.value.size() + row * input_width + input];
        const double* weight_row = weight + input * output_width;
        double* target_row =
            target.laplacian.data() + electron * target.value.size() + row * output_width;
        for (std::size_t output = 0; output < output_width; ++output)
          target_row[output] += source_derivative * weight_row[output];
      }
}

inline void DirectSpatialExecutor::denseQKVJet(const DirectSpatialJetBuffer& source,
                                               const double* query_weight,
                                               const double* key_weight,
                                               const double* value_weight,
                                               std::size_t rows,
                                               std::size_t width,
                                               std::size_t gradient_lanes,
                                               std::size_t laplacian_lanes,
                                               DirectSpatialJetBuffer& query,
                                               DirectSpatialJetBuffer& key,
                                               DirectSpatialJetBuffer& value)
{
  query.clear();
  key.clear();
  value.clear();
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t input = 0; input < width; ++input)
    {
      const std::size_t source_element = row * width + input;
      const std::size_t weight_begin   = input * width;
      for (std::size_t output = 0; output < width; ++output)
      {
        const std::size_t target_element = row * width + output;
        const std::size_t parameter      = weight_begin + output;
        query.value[target_element] += source.value[source_element] * query_weight[parameter];
        key.value[target_element] += source.value[source_element] * key_weight[parameter];
        value.value[target_element] += source.value[source_element] * value_weight[parameter];
      }
    }

  for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t input = 0; input < width; ++input)
      {
        const double source_derivative =
            source.gradient[lane * source.value.size() + row * width + input];
        const std::size_t weight_begin = input * width;
        const std::size_t target_begin = lane * query.value.size() + row * width;
        for (std::size_t output = 0; output < width; ++output)
        {
          query.gradient[target_begin + output] += source_derivative * query_weight[weight_begin + output];
          key.gradient[target_begin + output] += source_derivative * key_weight[weight_begin + output];
          value.gradient[target_begin + output] += source_derivative * value_weight[weight_begin + output];
        }
      }

  for (std::size_t electron = 0; electron < laplacian_lanes; ++electron)
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t input = 0; input < width; ++input)
      {
        const double source_derivative =
            source.laplacian[electron * source.value.size() + row * width + input];
        const std::size_t weight_begin = input * width;
        const std::size_t target_begin = electron * query.value.size() + row * width;
        for (std::size_t output = 0; output < width; ++output)
        {
          query.laplacian[target_begin + output] += source_derivative * query_weight[weight_begin + output];
          key.laplacian[target_begin + output] += source_derivative * key_weight[weight_begin + output];
          value.laplacian[target_begin + output] += source_derivative * value_weight[weight_begin + output];
        }
      }
}

inline void DirectSpatialExecutor::buildEmbeddingFeatures(
    DirectSpatialWorkspace& workspace) const
{
  DirectSpatialJetBuffer& raw       = workspace.raw_features_;
  const GeometryPairTable& pairs    = workspace.geometry_.electronNucleusPairs();
  const auto& displacements         = pairs.displacements();
  const auto& distances             = pairs.distances();
  const auto& factors               = pairs.softenedRadialFactors();
  const std::size_t electron_count  = layout_->electronCount();
  const std::size_t nucleus_count   = layout_->nucleusCount();
  const std::size_t input_width     = layout_->inputWidth();
  raw.clear();

  for (std::size_t electron = 0; electron < electron_count; ++electron)
  {
    const std::size_t row_begin = electron * input_width;
    for (std::size_t nucleus = 0; nucleus < nucleus_count; ++nucleus)
    {
      const std::size_t pair = electron * nucleus_count + nucleus;
      const double radius    = distances[pair];
      if (radius == 0)
        throw std::runtime_error("PsiFormer spatial derivatives are undefined at an electron-nucleus coalescence");
      const double inverse_radius = 1.0 / radius;
      const GeometryPosition& displacement = displacements[pair];
      const SoftenedRadialFactors& radial  = factors[pair];
      const std::size_t radial_element     = row_begin + 4 * nucleus;
      raw.value[radial_element]            = radial.log1p_radius;
      for (std::size_t component = 0; component < 3; ++component)
        raw.value[radial_element + 1 + component] =
            displacement[component] * radial.log1p_over_radius;

      for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
      {
        if (laneElectron(workspace, lane) != electron)
          continue;
        const std::size_t derivative_dimension = laneDimension(workspace, lane);
        const double unit_component = displacement[derivative_dimension] * inverse_radius;
        raw.gradient[gradientIndex(raw, lane, radial_element)] =
            radial.log1p_first * unit_component;
        for (std::size_t component = 0; component < 3; ++component)
        {
          const double kronecker = component == derivative_dimension ? 1.0 : 0.0;
          raw.gradient[gradientIndex(raw, lane, radial_element + 1 + component)] =
              kronecker * radial.log1p_over_radius +
              displacement[component] * radial.log1p_over_radius_first * unit_component;
        }
      }

      if (workspace.mode_ == DirectSpatialMode::FULL_VGL)
      {
        raw.laplacian[laplacianIndex(raw, electron, radial_element)] =
            radial.log1p_second + 2.0 * radial.log1p_first * inverse_radius;
        const double directional_laplacian_factor =
            radial.log1p_over_radius_second + 4.0 * radial.log1p_over_radius_first * inverse_radius;
        for (std::size_t component = 0; component < 3; ++component)
          raw.laplacian[laplacianIndex(raw, electron, radial_element + 1 + component)] =
              displacement[component] * directional_laplacian_factor;
      }
    }
    raw.value[row_begin + input_width - 1] = electron < layout_->spinUpCount() ? 1.0 : -1.0;
  }

}

inline void DirectSpatialExecutor::buildEmbedding(
    const double* parameters,
    DirectSpatialWorkspace& workspace) const
{
  buildEmbeddingFeatures(workspace);
  denseJet(workspace.raw_features_, tensor(parameters, layout_->embedding_), nullptr,
           layout_->electronCount(), layout_->inputWidth(), layout_->featureWidth(),
           workspace.gradient_lanes_, workspace.laplacian_lanes_,
           workspace.features_a_);
}

inline void DirectSpatialExecutor::tanhJetInPlace(DirectSpatialJetBuffer& buffer,
                                                  std::size_t gradient_lanes,
                                                  std::size_t laplacian_lanes)
{
  for (std::size_t element = 0; element < buffer.value.size(); ++element)
  {
    const double output            = std::tanh(buffer.value[element]);
    const double first_derivative  = 1.0 - output * output;
    const double second_derivative = -2.0 * output * first_derivative;
    buffer.value[element]          = output;

    // Laplacians consume the unscaled Cartesian gradients, so update them first.
    for (std::size_t electron = 0; electron < laplacian_lanes; ++electron)
    {
      double squared_gradient = 0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double derivative =
            buffer.gradient[(3 * electron + dimension) * buffer.value.size() + element];
        squared_gradient += derivative * derivative;
      }
      const std::size_t index = electron * buffer.value.size() + element;
      buffer.laplacian[index] =
          first_derivative * buffer.laplacian[index] + second_derivative * squared_gradient;
    }
    for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
      buffer.gradient[lane * buffer.value.size() + element] *= first_derivative;
  }
}

inline void DirectSpatialExecutor::buildAttentionWeights(DirectSpatialWorkspace& workspace) const
{
  DirectSpatialJetBuffer& attention = workspace.attention_;
  const DirectSpatialJetBuffer& query = workspace.query_;
  const DirectSpatialJetBuffer& key   = workspace.key_;
  const std::size_t electron_count    = layout_->electronCount();
  const std::size_t head_count        = layout_->headCount();
  const std::size_t head_width        = layout_->headWidth();
  const std::size_t feature_width     = layout_->featureWidth();
  const double scale                  = 1.0 / std::sqrt(static_cast<double>(head_width));
  attention.clear();

  auto feature_index = [=](std::size_t electron, std::size_t head, std::size_t feature) {
    return electron * feature_width + head * head_width + feature;
  };
  auto attention_index = [=](std::size_t head, std::size_t output_electron, std::size_t input_electron) {
    return (head * electron_count + output_electron) * electron_count + input_electron;
  };

  // Form scaled Q*K^T with the product rules for every stored derivative trace.
  for (std::size_t head = 0; head < head_count; ++head)
    for (std::size_t output_electron = 0; output_electron < electron_count; ++output_electron)
      for (std::size_t input_electron = 0; input_electron < electron_count; ++input_electron)
      {
        const std::size_t output = attention_index(head, output_electron, input_electron);
        for (std::size_t feature = 0; feature < head_width; ++feature)
        {
          const std::size_t query_element = feature_index(output_electron, head, feature);
          const std::size_t key_element   = feature_index(input_electron, head, feature);
          attention.value[output] += scale * query.value[query_element] * key.value[key_element];
          for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
          {
            const double query_gradient = query.gradient[gradientIndex(query, lane, query_element)];
            const double key_gradient   = key.gradient[gradientIndex(key, lane, key_element)];
            attention.gradient[gradientIndex(attention, lane, output)] +=
                scale * (query_gradient * key.value[key_element] +
                         query.value[query_element] * key_gradient);
          }
          for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
          {
            double gradient_dot = 0;
            for (std::size_t dimension = 0; dimension < 3; ++dimension)
              gradient_dot += query.gradient[gradientIndex(query, 3 * electron + dimension, query_element)] *
                  key.gradient[gradientIndex(key, 3 * electron + dimension, key_element)];
            attention.laplacian[laplacianIndex(attention, electron, output)] +=
                scale * (query.laplacian[laplacianIndex(query, electron, query_element)] * key.value[key_element] +
                         2.0 * gradient_dot + query.value[query_element] *
                             key.laplacian[laplacianIndex(key, electron, key_element)]);
          }
        }
      }

  // Apply stable row-wise softmax.  Laplacians are transformed before gradients
  // because their formula requires the original logit gradients.
  for (std::size_t head = 0; head < head_count; ++head)
    for (std::size_t output_electron = 0; output_electron < electron_count; ++output_electron)
    {
      const std::size_t row_begin = attention_index(head, output_electron, 0);
      double row_maximum = -std::numeric_limits<double>::infinity();
      for (std::size_t input_electron = 0; input_electron < electron_count; ++input_electron)
        row_maximum = std::max(row_maximum, attention.value[row_begin + input_electron]);

      double normalization = 0;
      for (std::size_t input_electron = 0; input_electron < electron_count; ++input_electron)
      {
        double& probability = attention.value[row_begin + input_electron];
        probability         = std::exp(probability - row_maximum);
        normalization += probability;
      }
      for (std::size_t input_electron = 0; input_electron < electron_count; ++input_electron)
        attention.value[row_begin + input_electron] /= normalization;

      for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
      {
        double mean_laplacian = 0;
        double mean_gradient[3]{};
        for (std::size_t input_electron = 0; input_electron < electron_count; ++input_electron)
        {
          const std::size_t element = row_begin + input_electron;
          const double probability  = attention.value[element];
          mean_laplacian += probability * attention.laplacian[laplacianIndex(attention, electron, element)];
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
            mean_gradient[dimension] += probability *
                attention.gradient[gradientIndex(attention, 3 * electron + dimension, element)];
        }

        double mean_squared_deviation = 0;
        for (std::size_t input_electron = 0; input_electron < electron_count; ++input_electron)
        {
          const std::size_t element = row_begin + input_electron;
          double squared_deviation  = 0;
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
          {
            const double deviation =
                attention.gradient[gradientIndex(attention, 3 * electron + dimension, element)] -
                mean_gradient[dimension];
            squared_deviation += deviation * deviation;
          }
          mean_squared_deviation += attention.value[element] * squared_deviation;
        }

        for (std::size_t input_electron = 0; input_electron < electron_count; ++input_electron)
        {
          const std::size_t element = row_begin + input_electron;
          double squared_deviation  = 0;
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
          {
            const double deviation =
                attention.gradient[gradientIndex(attention, 3 * electron + dimension, element)] -
                mean_gradient[dimension];
            squared_deviation += deviation * deviation;
          }
          const std::size_t lap_index = laplacianIndex(attention, electron, element);
          attention.laplacian[lap_index] = attention.value[element] *
              (attention.laplacian[lap_index] - mean_laplacian + squared_deviation -
               mean_squared_deviation);
        }
      }

      for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
      {
        double mean_gradient = 0;
        for (std::size_t input_electron = 0; input_electron < electron_count; ++input_electron)
        {
          const std::size_t element = row_begin + input_electron;
          mean_gradient += attention.value[element] * attention.gradient[gradientIndex(attention, lane, element)];
        }
        for (std::size_t input_electron = 0; input_electron < electron_count; ++input_electron)
        {
          const std::size_t element = row_begin + input_electron;
          const std::size_t index   = gradientIndex(attention, lane, element);
          attention.gradient[index] =
              attention.value[element] * (attention.gradient[index] - mean_gradient);
        }
      }
    }
}

inline void DirectSpatialExecutor::buildAttentionContext(DirectSpatialWorkspace& workspace) const
{
  DirectSpatialJetBuffer& output        = workspace.attended_;
  const DirectSpatialJetBuffer& weights = workspace.attention_;
  const DirectSpatialJetBuffer& values  = workspace.projected_value_;
  const std::size_t electron_count      = layout_->electronCount();
  const std::size_t head_count          = layout_->headCount();
  const std::size_t head_width          = layout_->headWidth();
  const std::size_t feature_width       = layout_->featureWidth();
  output.clear();

  for (std::size_t output_electron = 0; output_electron < electron_count; ++output_electron)
    for (std::size_t head = 0; head < head_count; ++head)
      for (std::size_t input_electron = 0; input_electron < electron_count; ++input_electron)
      {
        const std::size_t weight_element =
            (head * electron_count + output_electron) * electron_count + input_electron;
        for (std::size_t feature = 0; feature < head_width; ++feature)
        {
          const std::size_t output_element = output_electron * feature_width + head * head_width + feature;
          const std::size_t value_element  = input_electron * feature_width + head * head_width + feature;
          output.value[output_element] += weights.value[weight_element] * values.value[value_element];
          for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
            output.gradient[gradientIndex(output, lane, output_element)] +=
                weights.gradient[gradientIndex(weights, lane, weight_element)] * values.value[value_element] +
                weights.value[weight_element] * values.gradient[gradientIndex(values, lane, value_element)];
          for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
          {
            double gradient_dot = 0;
            for (std::size_t dimension = 0; dimension < 3; ++dimension)
              gradient_dot +=
                  weights.gradient[gradientIndex(weights, 3 * electron + dimension, weight_element)] *
                  values.gradient[gradientIndex(values, 3 * electron + dimension, value_element)];
            output.laplacian[laplacianIndex(output, electron, output_element)] +=
                weights.laplacian[laplacianIndex(weights, electron, weight_element)] *
                    values.value[value_element] +
                2.0 * gradient_dot + weights.value[weight_element] *
                    values.laplacian[laplacianIndex(values, electron, value_element)];
          }
        }
      }
}

inline void DirectSpatialExecutor::applyAttentionBlock(
    const double* parameters,
    const DirectValueParameterLayout::DirectAttentionLayer& layer,
    DirectSpatialWorkspace& workspace) const
{
  const std::size_t electron_count = layout_->electronCount();
  const std::size_t feature_width  = layout_->featureWidth();
  denseQKVJet(workspace.features_a_, tensor(parameters, layer.query), tensor(parameters, layer.key),
              tensor(parameters, layer.value), electron_count, feature_width,
              workspace.gradient_lanes_, workspace.laplacian_lanes_, workspace.query_, workspace.key_,
              workspace.projected_value_);
  buildAttentionWeights(workspace);
  buildAttentionContext(workspace);

  denseJet(workspace.attended_, tensor(parameters, layer.projection), nullptr, electron_count,
           feature_width, feature_width, workspace.gradient_lanes_, workspace.laplacian_lanes_,
           workspace.features_b_);
  for (std::size_t element = 0; element < workspace.features_b_.value.size(); ++element)
    workspace.features_b_.value[element] += workspace.features_a_.value[element];
  for (std::size_t element = 0; element < workspace.features_b_.gradient.size(); ++element)
    workspace.features_b_.gradient[element] += workspace.features_a_.gradient[element];
  for (std::size_t element = 0; element < workspace.features_b_.laplacian.size(); ++element)
    workspace.features_b_.laplacian[element] += workspace.features_a_.laplacian[element];

  denseJet(workspace.features_b_, tensor(parameters, layer.mlp_weight_0),
           tensor(parameters, layer.mlp_bias_0), electron_count, feature_width, feature_width,
           workspace.gradient_lanes_, workspace.laplacian_lanes_, workspace.hidden_);
  tanhJetInPlace(workspace.hidden_, workspace.gradient_lanes_, workspace.laplacian_lanes_);
  denseJet(workspace.hidden_, tensor(parameters, layer.mlp_weight_1),
           tensor(parameters, layer.mlp_bias_1), electron_count, feature_width, feature_width,
           workspace.gradient_lanes_, workspace.laplacian_lanes_, workspace.attended_);
  tanhJetInPlace(workspace.attended_, workspace.gradient_lanes_, workspace.laplacian_lanes_);

  for (std::size_t element = 0; element < workspace.features_b_.value.size(); ++element)
    workspace.features_b_.value[element] += workspace.attended_.value[element];
  for (std::size_t element = 0; element < workspace.features_b_.gradient.size(); ++element)
    workspace.features_b_.gradient[element] += workspace.attended_.gradient[element];
  for (std::size_t element = 0; element < workspace.features_b_.laplacian.size(); ++element)
    workspace.features_b_.laplacian[element] += workspace.attended_.laplacian[element];
  std::swap(workspace.features_a_, workspace.features_b_);
}

inline void DirectSpatialExecutor::buildOrbitalMatrices(const double* parameters,
                                                        DirectSpatialWorkspace& workspace) const
{
  DirectSpatialJetBuffer& orbitals    = workspace.orbital_matrices_;
  const DirectSpatialJetBuffer& features = workspace.features_a_;
  const GeometryPairTable& pairs      = workspace.geometry_.electronNucleusPairs();
  const auto& displacements           = pairs.displacements();
  const auto& distances               = pairs.distances();
  const std::size_t electron_count    = layout_->electronCount();
  const std::size_t nucleus_count     = layout_->nucleusCount();
  const std::size_t determinant_count = layout_->determinantCount();
  const std::size_t feature_width     = layout_->featureWidth();
  orbitals.clear();

  for (std::size_t row_electron = 0; row_electron < electron_count; ++row_electron)
  {
    const bool spin_up = row_electron < layout_->spinUpCount();
    const double* backflow = tensor(parameters, spin_up ? layout_->backflow_up_ : layout_->backflow_down_);
    const double* pi       = tensor(parameters, spin_up ? layout_->pi_up_ : layout_->pi_down_);
    const double* zeta     = tensor(parameters, spin_up ? layout_->zeta_up_ : layout_->zeta_down_);
    const std::size_t feature_begin = row_electron * feature_width;

    for (std::size_t determinant = 0; determinant < determinant_count; ++determinant)
      for (std::size_t orbital = 0; orbital < electron_count; ++orbital)
      {
        const std::size_t channel = determinant * electron_count + orbital;
        const std::size_t matrix_element =
            (determinant * electron_count + row_electron) * electron_count + orbital;
        double backflow_value = 0;
        for (std::size_t feature = 0; feature < feature_width; ++feature)
          backflow_value += features.value[feature_begin + feature] *
              backflow[feature * determinant_count * electron_count + channel];

        double envelope_value = 0;
        std::fill(workspace.scalar_gradient_scratch_.begin(), workspace.scalar_gradient_scratch_.end(), 0.0);
        std::fill(workspace.scalar_laplacian_scratch_.begin(), workspace.scalar_laplacian_scratch_.end(), 0.0);
        for (std::size_t nucleus = 0; nucleus < nucleus_count; ++nucleus)
        {
          const std::size_t parameter = channel * nucleus_count + nucleus;
          const std::size_t pair      = row_electron * nucleus_count + nucleus;
          const double radius         = distances[pair];
          if (radius == 0)
            throw std::runtime_error(
                "PsiFormer spatial derivatives are undefined at an electron-nucleus coalescence");
          const double inverse_radius = 1.0 / radius;
          const double decay_rate     = std::abs(zeta[parameter]);
          const double exponential    = std::exp(-decay_rate * radius);
          const double weighted_value = pi[parameter] * exponential;
          const double radial_first   = -decay_rate * weighted_value;
          const double radial_second  = decay_rate * decay_rate * weighted_value;
          envelope_value += weighted_value;

          for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
            if (laneElectron(workspace, lane) == row_electron)
            {
              const std::size_t dimension = laneDimension(workspace, lane);
              workspace.scalar_gradient_scratch_[lane] +=
                  radial_first * displacements[pair][dimension] * inverse_radius;
            }
          if (workspace.mode_ == DirectSpatialMode::FULL_VGL)
            workspace.scalar_laplacian_scratch_[row_electron] +=
                radial_second + 2.0 * radial_first * inverse_radius;
        }

        orbitals.value[matrix_element] = backflow_value * envelope_value;
        for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
        {
          double backflow_gradient = 0;
          for (std::size_t feature = 0; feature < feature_width; ++feature)
            backflow_gradient += features.gradient[gradientIndex(features, lane, feature_begin + feature)] *
                backflow[feature * determinant_count * electron_count + channel];
          orbitals.gradient[gradientIndex(orbitals, lane, matrix_element)] =
              backflow_gradient * envelope_value +
              backflow_value * workspace.scalar_gradient_scratch_[lane];
        }

        for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
        {
          double backflow_laplacian = 0;
          double gradient_dot       = 0;
          for (std::size_t feature = 0; feature < feature_width; ++feature)
          {
            const double weight = backflow[feature * determinant_count * electron_count + channel];
            backflow_laplacian +=
                features.laplacian[laplacianIndex(features, electron, feature_begin + feature)] * weight;
            for (std::size_t dimension = 0; dimension < 3; ++dimension)
              gradient_dot += features.gradient[
                                  gradientIndex(features, 3 * electron + dimension,
                                                feature_begin + feature)] *
                  weight * workspace.scalar_gradient_scratch_[3 * electron + dimension];
          }
          orbitals.laplacian[laplacianIndex(orbitals, electron, matrix_element)] =
              backflow_laplacian * envelope_value + 2.0 * gradient_dot +
              backflow_value * workspace.scalar_laplacian_scratch_[electron];
        }
      }
  }
}

inline double DirectSpatialExecutor::accumulateCusp(const double* parameters,
                                                    DirectSpatialWorkspace& workspace) const
{
  std::fill(workspace.scalar_gradient_scratch_.begin(), workspace.scalar_gradient_scratch_.end(), 0.0);
  std::fill(workspace.scalar_laplacian_scratch_.begin(), workspace.scalar_laplacian_scratch_.end(), 0.0);
  const double same_alpha = layout_->same_alpha_.size == 0
      ? 1.0
      : tensor(parameters, layout_->same_alpha_)[0];
  const double opposite_alpha = tensor(parameters, layout_->anti_alpha_)[0];
  const auto& identities      = workspace.geometry_.electronPairs();
  const GeometryPairTable& pair_table = workspace.geometry_.electronElectronPairs();
  const auto& displacements = pair_table.displacements();
  const auto& distances     = pair_table.distances();
  double cusp_value         = 0;

  for (std::size_t pair_index = 0; pair_index < identities.size(); ++pair_index)
  {
    const ElectronPair pair = identities[pair_index];
    const bool first_up     = pair.first < layout_->spinUpCount();
    const bool second_up    = pair.second < layout_->spinUpCount();
    const bool same_spin    = first_up == second_up;
    const double alpha      = same_spin ? same_alpha : opposite_alpha;
    const double factor     = same_spin ? 0.25 : 0.5;
    const double radius     = distances[pair_index];
    if (radius == 0)
      throw std::runtime_error("PsiFormer spatial derivatives are undefined at an electron-electron coalescence");
    const double denominator = alpha + radius;
    const double numerator   = factor * alpha * alpha;
    const double radial_first = numerator / (denominator * denominator);
    const double radial_second = -2.0 * numerator / (denominator * denominator * denominator);
    cusp_value -= numerator / denominator;

    for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
    {
      const std::size_t electron = laneElectron(workspace, lane);
      if (electron != pair.first && electron != pair.second)
        continue;
      const double incidence_sign = electron == pair.first ? 1.0 : -1.0;
      const std::size_t dimension = laneDimension(workspace, lane);
      workspace.scalar_gradient_scratch_[lane] += incidence_sign * radial_first *
          displacements[pair_index][dimension] / radius;
    }
    if (workspace.mode_ == DirectSpatialMode::FULL_VGL)
    {
      const double pair_laplacian = radial_second + 2.0 * radial_first / radius;
      workspace.scalar_laplacian_scratch_[pair.first] += pair_laplacian;
      workspace.scalar_laplacian_scratch_[pair.second] += pair_laplacian;
    }
  }
  return cusp_value;
}

inline DirectSpatialResultView DirectSpatialExecutor::evaluateImpl(
    DirectSpatialWorkspace& workspace) const
{
  layout_->validateParameterStore(parameters_);
  if (workspace.geometry_.electronCount() != layout_->electronCount() ||
      workspace.geometry_.nucleusCount() != layout_->nucleusCount())
    throw std::invalid_argument("PsiFormer spatial workspace belongs to a different model shape");

  const double* parameter_values = parameters_.flat_values().data();
  refreshGeometry(workspace);
  buildEmbedding(parameter_values, workspace);
  for (std::size_t layer = 0; layer < layout_->layers_.size(); ++layer)
    applyAttentionBlock(parameter_values, layout_->layers_[layer], workspace);
  buildOrbitalMatrices(parameter_values, workspace);
  namespace determinant = qmcplusplus::psiformer::determinant;
  const determinant::RealDeterminantResult determinant_result =
      workspace.determinant_workspace_.evaluateSpatial(
          workspace.orbital_matrices_.value.data(),
          workspace.orbital_matrices_.gradient.data(), workspace.gradient_lanes_,
          workspace.orbital_matrices_.laplacian.data(), workspace.laplacian_lanes_,
          workspace.output_gradient_.data(), workspace.output_lap_log_.data(),
          workspace.output_lap_ratio_.data());
  if (determinant_result.amplitude.isZero())
    throw std::runtime_error("PsiFormer spatial evaluator reached an exact determinant node");

  const double cusp_value = accumulateCusp(parameter_values, workspace);
  for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
    workspace.output_gradient_[lane] += workspace.scalar_gradient_scratch_[lane];

  for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
  {
    double total_squared_gradient = 0;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const std::size_t lane = 3 * electron + dimension;
      total_squared_gradient += workspace.output_gradient_[lane] * workspace.output_gradient_[lane];
    }
    workspace.output_lap_log_[electron] += workspace.scalar_laplacian_scratch_[electron];
    workspace.output_lap_ratio_[electron] =
        workspace.output_lap_log_[electron] + total_squared_gradient;
  }

  const double logabs = determinant_result.amplitude.log_abs + cusp_value;
  if (!is_finite_parameter_value(logabs))
    throw std::runtime_error("PsiFormer spatial evaluator produced a non-finite log amplitude");
  for (double component : workspace.output_gradient_)
    if (!is_finite_parameter_value(component))
      throw std::runtime_error("PsiFormer spatial evaluator produced a non-finite gradient");
  for (double component : workspace.output_lap_log_)
    if (!is_finite_parameter_value(component))
      throw std::runtime_error("PsiFormer spatial evaluator produced a non-finite Laplacian");

  workspace.observed_parameter_version_ = parameters_.version();
  const double sign = determinant_result.amplitude.phase;
  return {sign,
          logabs,
          determinant::realValue(determinant_result.amplitude, cusp_value),
          workspace.mode_,
          {workspace.output_gradient_.data(), workspace.output_gradient_.size()},
          {workspace.output_lap_log_.data(), workspace.output_lap_log_.size()},
          {workspace.output_lap_ratio_.data(), workspace.output_lap_ratio_.size()},
          workspace.observed_parameter_version_};
}

} // namespace pf

#endif // QMCPLUSPLUS_PSIFORMER_SPATIAL_EXECUTOR_H
