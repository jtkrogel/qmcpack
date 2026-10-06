//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerValueExecutor.h
 * @brief Allocation-free direct value-only evaluator for molecular PsiFormer models.
 *
 * This executor is the optimized inference counterpart to PsiFormerNative.h.  The
 * native evaluator remains the derivative-capable oracle; this file evaluates only
 * sign and log|psi| by direct dense, attention, envelope, determinant, and cusp
 * kernels.  Parameter names and tensor shapes are resolved once at construction,
 * while all mutable scratch storage belongs to a clone-local workspace.
 *
 * Geometry is kept in the real-valued PsiFormerGeometryCache and the constructor
 * accepts its boundary descriptor.  Only open molecular boundaries and real-valued
 * wavefunctions are implemented.  Keeping geometry/boundary handling outside the
 * algebra kernels, and keeping the result in sign/log-magnitude form, provides the
 * intended seams for later periodic displacement and complex phase policies without
 * claiming support for either here.
 */

//////////////////////////////////////////////////////////////////////////////////////
// INCLUSION RESTRICTION
// PsiFormerNative.h defines non-inline functions and is included by exactly one QMCPACK
// translation unit.  Include this helper only after PsiFormerNative.h in that same
// translation unit (or in a standalone one-translation-unit test).
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_PSIFORMER_NATIVE_H
#error "Include PsiFormerNative.h before PsiFormerValueExecutor.h"
#endif

#ifndef QMCPLUSPLUS_PSIFORMER_VALUE_EXECUTOR_H
#define QMCPLUSPLUS_PSIFORMER_VALUE_EXECUTOR_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerGeometry.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDenseKernels.h"
#include "PsiFormerDeterminant.h"
#include "PsiFormerStorageRequirements.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pf
{

/** Describe one immutable interval in the canonical flattened parameter vector.
 *
 * Evaluations use only these offsets; Haiku module strings are never searched on a
 * hot path.  The owning Parameters object may replace values between evaluations
 * because offsets and shapes are immutable across optimizer versions.
 */
struct DirectParameterTensor
{
  std::size_t begin = 0;
  std::size_t size  = 0;
};

/// Hold all parameter intervals and fixed dimensions required by direct inference.
class DirectValueParameterLayout
{
public:
  /// Copy hot-path intervals from the shared, fully validated execution plan.
  DirectValueParameterLayout(const PsiFormer& model,
                             const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan)
      : parameter_count_(plan.parameterCount()),
        electron_count_(plan.modelShape().electrons()),
        spin_up_count_(plan.modelShape().spin_up_electrons),
        spin_down_count_(plan.modelShape().spin_down_electrons),
        nucleus_count_(plan.modelShape().nuclei),
        determinant_count_(plan.modelShape().determinants),
        feature_width_(plan.modelShape().feature_dimension),
        head_count_(plan.modelShape().attention_heads),
        head_width_(feature_width_ / head_count_),
        input_width_(plan.parameter(qmcplusplus::psiformer::ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0]),
        layers_(plan.modelShape().attention_blocks)
  {
    if (electron_count_ == 0 || spin_up_count_ + spin_down_count_ != electron_count_ ||
        nucleus_count_ == 0 || determinant_count_ == 0 || feature_width_ == 0 ||
        head_count_ == 0 || feature_width_ % head_count_ != 0 ||
        parameter_count_ != model.p.size())
      throw std::invalid_argument("PsiFormer direct evaluator received inconsistent model dimensions");

    using qmcplusplus::psiformer::BoundaryCondition;
    using qmcplusplus::psiformer::ParameterRole;
    using qmcplusplus::psiformer::ScalarDomain;
    const auto& environment = plan.environment();
    if (environment.parameter_scalar_domain != ScalarDomain::REAL ||
        environment.compute_scalar_domain != ScalarDomain::REAL ||
        environment.amplitude_scalar_domain != ScalarDomain::REAL || !environment.fixed_nuclei)
      throw std::invalid_argument("PsiFormer direct value execution supports only fixed-ion real models");

    anti_alpha_ = interval(plan.parameter(ParameterRole::CUSP_OPPOSITE_ALPHA));
    if (const auto* same_alpha = plan.optionalParameter(ParameterRole::CUSP_SAME_ALPHA))
      same_alpha_ = interval(*same_alpha);
    pi_down_    = interval(plan.parameter(ParameterRole::ENVELOPE_PI_DOWN));
    pi_up_      = interval(plan.parameter(ParameterRole::ENVELOPE_PI_UP));
    zeta_down_  = interval(plan.parameter(ParameterRole::ENVELOPE_ZETA_DOWN));
    zeta_up_    = interval(plan.parameter(ParameterRole::ENVELOPE_ZETA_UP));
    backflow_up_   = interval(plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT));
    backflow_down_ = interval(plan.parameter(ParameterRole::BACKFLOW_DOWN_WEIGHT));
    embedding_     = interval(plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT));

    for (std::size_t layer = 0; layer < layers_.size(); ++layer)
    {
      DirectAttentionLayer& descriptor = layers_[layer];
      descriptor.query       = interval(plan.parameter(ParameterRole::ATTENTION_QUERY_WEIGHT, layer));
      descriptor.key         = interval(plan.parameter(ParameterRole::ATTENTION_KEY_WEIGHT, layer));
      descriptor.value       = interval(plan.parameter(ParameterRole::ATTENTION_VALUE_WEIGHT, layer));
      descriptor.projection  = interval(plan.parameter(ParameterRole::ATTENTION_OUTPUT_WEIGHT, layer));
      descriptor.mlp_weight_0 = interval(plan.parameter(ParameterRole::UPDATE_HIDDEN_WEIGHT, layer));
      descriptor.mlp_bias_0   = interval(plan.parameter(ParameterRole::UPDATE_HIDDEN_BIAS, layer));
      descriptor.mlp_weight_1 = interval(plan.parameter(ParameterRole::UPDATE_OUTPUT_WEIGHT, layer));
      descriptor.mlp_bias_1   = interval(plan.parameter(ParameterRole::UPDATE_OUTPUT_BIAS, layer));
    }
  }

  /// Verify that a mutable parameter store still has the immutable layout we resolved.
  void validateParameterStore(const Parameters& parameters) const
  {
    if (parameters.size() != parameter_count_)
      throw std::logic_error("PsiFormer direct evaluator parameter layout changed after construction");
  }

  std::size_t parameterCount() const noexcept { return parameter_count_; }
  std::size_t electronCount() const noexcept { return electron_count_; }
  std::size_t spinUpCount() const noexcept { return spin_up_count_; }
  std::size_t spinDownCount() const noexcept { return spin_down_count_; }
  std::size_t nucleusCount() const noexcept { return nucleus_count_; }
  std::size_t determinantCount() const noexcept { return determinant_count_; }
  std::size_t featureWidth() const noexcept { return feature_width_; }
  std::size_t headCount() const noexcept { return head_count_; }
  std::size_t headWidth() const noexcept { return head_width_; }
  std::size_t inputWidth() const noexcept { return input_width_; }

private:
  friend class DirectValueExecutor;
  friend class DirectSpatialExecutor;
  friend class DirectBatchExecutor;

  /// Group the eight parameter tensors consumed by one attention/residual block.
  struct DirectAttentionLayer
  {
    DirectParameterTensor query;
    DirectParameterTensor key;
    DirectParameterTensor value;
    DirectParameterTensor projection;
    DirectParameterTensor mlp_weight_0;
    DirectParameterTensor mlp_bias_0;
    DirectParameterTensor mlp_weight_1;
    DirectParameterTensor mlp_bias_1;
  };

  /// Convert one shared typed descriptor to the minimal hot-path interval.
  static DirectParameterTensor interval(
      const qmcplusplus::psiformer::ParameterTensorDescriptor& descriptor)
  {
    return {descriptor.begin, descriptor.size()};
  }

  std::size_t parameter_count_;
  std::size_t electron_count_;
  std::size_t spin_up_count_;
  std::size_t spin_down_count_;
  std::size_t nucleus_count_;
  std::size_t determinant_count_;
  std::size_t feature_width_;
  std::size_t head_count_;
  std::size_t head_width_;
  std::size_t input_width_;

  DirectParameterTensor anti_alpha_;
  DirectParameterTensor same_alpha_;
  DirectParameterTensor pi_down_;
  DirectParameterTensor pi_up_;
  DirectParameterTensor zeta_down_;
  DirectParameterTensor zeta_up_;
  DirectParameterTensor backflow_up_;
  DirectParameterTensor backflow_down_;
  DirectParameterTensor embedding_;
  std::vector<DirectAttentionLayer> layers_;
};

/** Own every mutable buffer used by one evaluator clone.
 *
 * Vector sizes never change after construction.  A workspace is intentionally not
 * shared between walkers or threads; only the immutable layout and parameter store
 * are shared.
 */
class DirectValueWorkspace
{
public:
  /// Allocate fixed-size geometry and algebra buffers for one model shape.
  DirectValueWorkspace(const DirectValueParameterLayout& layout,
                       GeometryPositionView nuclei,
                       GeometryBoundary boundary = {})
      : electron_positions_(3 * layout.electronCount()),
        geometry_(layout.electronCount(), nuclei, boundary),
        raw_features_(layout.electronCount() * layout.inputWidth()),
        features_a_(layout.electronCount() * layout.featureWidth()),
        features_b_(layout.electronCount() * layout.featureWidth()),
        query_(layout.electronCount() * layout.featureWidth()),
        key_(layout.electronCount() * layout.featureWidth()),
        value_(layout.electronCount() * layout.featureWidth()),
        attention_(layout.headCount() * layout.electronCount() * layout.electronCount()),
        attended_(layout.electronCount() * layout.featureWidth()),
        hidden_(layout.electronCount() * layout.featureWidth()),
        orbital_matrices_(layout.determinantCount() * layout.electronCount() * layout.electronCount()),
        determinant_workspace_(layout.determinantCount(), layout.electronCount())
  {}

  DirectValueWorkspace(const DirectValueWorkspace&) = delete;
  DirectValueWorkspace& operator=(const DirectValueWorkspace&) = delete;
  DirectValueWorkspace(DirectValueWorkspace&&) = default;
  DirectValueWorkspace& operator=(DirectValueWorkspace&&) = default;

  /// Overwrite one Cartesian input coordinate without changing workspace capacity.
  void setPosition(std::size_t electron, std::size_t dimension, double value)
  {
    if (electron >= geometry_.electronCount() || dimension >= 3)
      throw std::out_of_range("PsiFormer direct workspace coordinate index is out of range");
    electron_positions_[3 * electron + dimension] = value;
  }

  /// Copy a complete interleaved configuration into the fixed input buffer.
  void setPositions(GeometryPositionView positions)
  {
    if (positions.size() != geometry_.electronCount())
      throw std::invalid_argument("PsiFormer direct workspace received the wrong electron count");
    for (std::size_t electron = 0; electron < positions.size(); ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        electron_positions_[3 * electron + dimension] = positions(electron, dimension);
  }

  /// Return the parameter version used by the most recent completed evaluation.
  std::size_t observedParameterVersion() const noexcept { return observed_parameter_version_; }

  /** Hash every backing address and capacity for warmed-call stability tests.
   *
   * This process-local diagnostic detects accidental workspace growth; it is not
   * a model or checkpoint fingerprint.
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
    mix(raw_features_);
    mix(features_a_);
    mix(features_b_);
    mix(query_);
    mix(key_);
    mix(value_);
    mix(attention_);
    mix(attended_);
    mix(hidden_);
    mix(orbital_matrices_);
    hash ^= determinant_workspace_.storageFingerprint();
    hash *= 1099511628211ULL;
    return hash;
  }

  /// Return bytes reserved by all numeric buffers, including geometry and determinants.
  std::size_t vectorStorageBytes() const
  {
    std::size_t bytes = 0;
    auto add = [&bytes](const std::vector<double>& buffer) {
      addStorageBytes(bytes,
                      checkedStorageBytes<double>(
                          buffer.capacity(),
                          "PsiFormer value storage bytes overflowed"),
                      "PsiFormer value storage bytes overflowed");
    };
    add(electron_positions_);
    add(raw_features_);
    add(features_a_);
    add(features_b_);
    add(query_);
    add(key_);
    add(value_);
    add(attention_);
    add(attended_);
    add(hidden_);
    add(orbital_matrices_);
    addStorageBytes(bytes, geometry_.storageBytes(),
                    "PsiFormer value storage bytes overflowed");
    addStorageBytes(bytes, determinant_workspace_.storageBytes(),
                    "PsiFormer value storage bytes overflowed");
    return bytes;
  }

  /// Expose the geometry contribution for focused inclusive-accounting tests.
  std::size_t geometryStorageBytes() const
  { return geometry_.storageBytes(); }

  /// Return the checked constructor-time requirement for this fixed model shape.
  std::size_t requiredStorageBytes() const
  {
    const std::size_t electrons = geometry_.electronCount();
    const std::size_t electron_square = checkedStorageProduct(
        electrons, electrons, "PsiFormer value shape overflowed");
    return valueWorkspaceStorageRequirement(
        {electrons, geometry_.nucleusCount(), determinant_workspace_.channels(),
         features_a_.size() / electrons,
         attention_.size() / electron_square,
         raw_features_.size() / electrons, 0, 0});
  }

private:
  friend class DirectValueExecutor;

  std::vector<double> electron_positions_;
  PsiFormerGeometryCache geometry_;
  std::vector<double> raw_features_;
  std::vector<double> features_a_;
  std::vector<double> features_b_;
  std::vector<double> query_;
  std::vector<double> key_;
  std::vector<double> value_;
  std::vector<double> attention_;
  std::vector<double> attended_;
  std::vector<double> hidden_;
  std::vector<double> orbital_matrices_;
  qmcplusplus::psiformer::determinant::RealOpenDeterminantWorkspace determinant_workspace_;
  std::size_t observed_parameter_version_ = std::numeric_limits<std::size_t>::max();
};

/** Return the real molecular wavefunction in sign/log-magnitude representation.
 *
 * The representation avoids overflow in ratios and leaves a clear extension point
 * for a future complex phase.  This implementation deliberately carries no phase
 * beyond the real sign.
 */
struct DirectValueResult
{
  double sign   = 1;
  double logabs = 0;
  double value  = 0;
  std::size_t parameter_version = 0;
};

/// Map the model-plan boundary axis to the geometry cache without relying on enum ordinals.
inline GeometryBoundary directGeometryBoundary(
    const qmcplusplus::psiformer::ExecutionEnvironment& environment)
{
  using qmcplusplus::psiformer::BoundaryCondition;
  switch (environment.boundary)
  {
  case BoundaryCondition::OPEN:
    return {GeometryBoundaryKind::OPEN};
  case BoundaryCondition::PERIODIC:
    return {GeometryBoundaryKind::PERIODIC, environment.lattice_vectors,
            environment.periodic_axes};
  }
  throw std::invalid_argument("Unknown PsiFormer execution-plan boundary condition");
}

/** Evaluate the real PsiFormer forward path under the selected geometry policy. */
class DirectValueExecutor
{
public:
  /// Resolve immutable descriptors and retain the versioned parameter value store.
  DirectValueExecutor(const PsiFormer& model,
                      const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan)
      : parameters_(model.p),
        layout_(std::make_shared<const DirectValueParameterLayout>(model, plan)),
        nuclei_(model.cfg.nuclei.x),
        boundary_(directGeometryBoundary(plan.environment()))
  {}

  /// Construct one independently mutable workspace suitable for a component clone.
  std::unique_ptr<DirectValueWorkspace> makeWorkspace() const
  {
    return std::make_unique<DirectValueWorkspace>(
        *layout_, GeometryPositionView::interleaved(nuclei_.data(), layout_->nucleusCount()), boundary_);
  }

  /// Expose immutable descriptors for diagnostics and clone-workspace construction.
  const std::shared_ptr<const DirectValueParameterLayout>& layout() const noexcept { return layout_; }

  /// Return the immutable parameter-store identity used to bind companion executors.
  const Parameters* parameterStoreIdentity() const noexcept { return &parameters_; }

  /// Expose immutable nuclear coordinates to bounded batch-workspace construction.
  GeometryPositionView nuclearPositions() const noexcept
  { return GeometryPositionView::interleaved(nuclei_.data(), layout_->nucleusCount()); }

  /// Expose the validated geometry policy used by direct batch scratch.
  const GeometryBoundary& boundary() const noexcept { return boundary_; }

  /// Evaluate sign and log|psi| without allocating or building differentiation nodes.
  DirectValueResult evaluate(DirectValueWorkspace& workspace) const
  {
    layout_->validateParameterStore(parameters_);
    if (workspace.geometry_.electronCount() != layout_->electronCount() ||
        workspace.geometry_.nucleusCount() != layout_->nucleusCount())
      throw std::invalid_argument("PsiFormer direct workspace belongs to a different model shape");

    const double* parameter_values = parameters_.flat_values().data();
    refreshGeometry(workspace);
    // Same-spin particles at the same physical point, including distinct
    // lattice images, are an exact fermionic node.
    if (hasExactSameSpinCoalescence(workspace))
      return exactNodeResult(workspace);
    buildEmbedding(parameter_values, workspace);
    for (std::size_t layer = 0; layer < layout_->layers_.size(); ++layer)
      applyAttentionBlock(parameter_values, layout_->layers_[layer], workspace);
    buildOrbitalMatrices(parameter_values, workspace);

    namespace determinant = qmcplusplus::psiformer::determinant;
    const determinant::RealDeterminantResult determinant_result =
        workspace.determinant_workspace_.evaluateValue(workspace.orbital_matrices_.data());
    if (determinant_result.amplitude.isZero())
    {
      // Value-only proposals may land exactly on a fermionic node.  Preserve
      // the signed-log zero so public ratio paths can reject that proposal with
      // ratio zero; derivative executors still reject nodes where log
      // derivatives are undefined.
      return exactNodeResult(workspace);
    }
    const double cusp   = cuspValue(parameter_values, workspace);
    const double logabs = determinant_result.amplitude.log_abs + cusp;
    if (!is_finite_parameter_value(logabs))
      throw std::runtime_error("PsiFormer direct evaluator produced a non-finite wavefunction value");

    workspace.observed_parameter_version_ = parameters_.version();
    const double sign = determinant_result.amplitude.phase;
    return {sign,
            logabs,
            determinant::realValue(determinant_result.amplitude, cusp),
            workspace.observed_parameter_version_};
  }

private:
  friend class DirectBatchExecutor;

  /// Return true when two same-spin electrons have zero boundary-aware separation.
  bool hasExactSameSpinCoalescence(const DirectValueWorkspace& workspace) const noexcept
  {
    const auto& identities = workspace.geometry_.electronPairs();
    const auto& distances  = workspace.geometry_.electronElectronPairs().distances();
    for (std::size_t pair = 0; pair < identities.size(); ++pair)
      if ((identities[pair].first < layout_->spinUpCount()) ==
              (identities[pair].second < layout_->spinUpCount()) &&
          distances[pair] == 0)
        return true;
    return false;
  }

  /// Materialize the canonical exact-zero representation for value-only paths.
  DirectValueResult exactNodeResult(DirectValueWorkspace& workspace) const noexcept
  {
    workspace.observed_parameter_version_ = parameters_.version();
    return {0.0,
            -std::numeric_limits<double>::infinity(),
            0.0,
            workspace.observed_parameter_version_};
  }

  /// Return the beginning of one pre-resolved parameter tensor.
  static const double* tensor(const double* parameters, const DirectParameterTensor& descriptor) noexcept
  {
    return parameters + descriptor.begin;
  }

  /// Refresh all open-boundary pair tables from the workspace input positions.
  void refreshGeometry(DirectValueWorkspace& workspace) const
  {
    workspace.geometry_.update(
        GeometryPositionView::interleaved(workspace.electron_positions_.data(), layout_->electronCount()));
  }

  /// Compute row-major target = source*weight + bias with cache-contiguous weights.
  static void dense(const double* source,
                    const double* weight,
                    const double* bias,
                    std::size_t rows,
                    std::size_t input_width,
                    std::size_t output_width,
                    double* target)
  {
    qmcplusplus::psiformer::dense::productReal(
        source, weight, bias, rows, input_width, output_width, target);
  }

  /// Project shared features into query, key, and value buffers in one traversal.
  static void denseQKV(const double* source,
                       const double* query_weight,
                       const double* key_weight,
                       const double* value_weight,
                       std::size_t rows,
                       std::size_t width,
                       double* query,
                       double* key,
                       double* value)
  {
    qmcplusplus::psiformer::dense::projectQkvReal(
        source, query_weight, key_weight, value_weight, rows, width, query, key, value);
  }

  /// Form stable row-wise softmax(Q*K^T/sqrt(head_width)) in the attention buffer.
  void buildAttentionWeights(DirectValueWorkspace& workspace) const
  {
    qmcplusplus::psiformer::dense::attentionWeightsReal(
        workspace.query_.data(), layout_->featureWidth(), workspace.key_.data(),
        layout_->featureWidth(), layout_->electronCount(), layout_->headCount(),
        layout_->headWidth(), workspace.attention_.data());
  }

  /// Contract softmax attention with projected values into electron-major features.
  void buildAttentionContext(DirectValueWorkspace& workspace) const
  {
    qmcplusplus::psiformer::dense::attentionContextReal(
        workspace.attention_.data(), workspace.value_.data(), layout_->featureWidth(),
        layout_->electronCount(), layout_->headCount(), layout_->headWidth(),
        workspace.attended_.data());
  }

  /// Build electron-nucleus radial/directional features and apply the embedding matrix.
  void buildEmbedding(const double* parameters, DirectValueWorkspace& workspace) const
  {
    const GeometryPairTable& pairs = workspace.geometry_.electronNucleusPairs();
    const auto& displacements      = pairs.displacements();
    const auto& complementary      = pairs.complementaryDisplacements();
    const auto& radial_factors     = pairs.softenedRadialFactors();
    const std::size_t ne           = layout_->electronCount();
    const std::size_t na           = layout_->nucleusCount();
    const std::size_t input_width  = layout_->inputWidth();
    const bool periodic = workspace.geometry_.boundary().kind == GeometryBoundaryKind::PERIODIC;
    const std::size_t pair_width = periodic ? 7 : 4;

    for (std::size_t electron = 0; electron < ne; ++electron)
    {
      double* feature_row = workspace.raw_features_.data() + electron * input_width;
      for (std::size_t nucleus = 0; nucleus < na; ++nucleus)
      {
        const std::size_t pair = electron * na + nucleus;
        const std::size_t feature_begin = pair_width * nucleus;
        feature_row[feature_begin] = radial_factors[pair].log1p_radius;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          feature_row[feature_begin + 1 + dimension] =
              displacements[pair][dimension] * radial_factors[pair].log1p_over_radius;
        if (periodic)
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
            feature_row[feature_begin + 4 + dimension] =
                complementary[pair][dimension] * radial_factors[pair].log1p_over_radius;
      }
      feature_row[input_width - 1] = electron < layout_->spinUpCount() ? 1.0 : -1.0;
    }

    dense(workspace.raw_features_.data(), tensor(parameters, layout_->embedding_), nullptr, ne, input_width,
          layout_->featureWidth(), workspace.features_a_.data());
  }

  /// Apply one direct multi-head-attention and two-layer residual MLP block.
  void applyAttentionBlock(const double* parameters,
                           const DirectValueParameterLayout::DirectAttentionLayer& layer,
                           DirectValueWorkspace& workspace) const
  {
    const std::size_t ne    = layout_->electronCount();
    const std::size_t width = layout_->featureWidth();
    denseQKV(workspace.features_a_.data(), tensor(parameters, layer.query), tensor(parameters, layer.key),
             tensor(parameters, layer.value), ne, width, workspace.query_.data(), workspace.key_.data(),
             workspace.value_.data());
    buildAttentionWeights(workspace);
    buildAttentionContext(workspace);

    dense(workspace.attended_.data(), tensor(parameters, layer.projection), nullptr, ne, width, width,
          workspace.features_b_.data());
    for (std::size_t element = 0; element < workspace.features_b_.size(); ++element)
      workspace.features_b_[element] += workspace.features_a_[element];

    dense(workspace.features_b_.data(), tensor(parameters, layer.mlp_weight_0),
          tensor(parameters, layer.mlp_bias_0), ne, width, width, workspace.hidden_.data());
    for (double& element : workspace.hidden_)
      element = std::tanh(element);
    dense(workspace.hidden_.data(), tensor(parameters, layer.mlp_weight_1),
          tensor(parameters, layer.mlp_bias_1), ne, width, width, workspace.attended_.data());
    for (std::size_t element = 0; element < workspace.attended_.size(); ++element)
      workspace.features_b_[element] += std::tanh(workspace.attended_[element]);
    workspace.features_a_.swap(workspace.features_b_);
  }

  /// Form all determinant orbital matrices from spin backflow and exponential envelopes.
  void buildOrbitalMatrices(const double* parameters, DirectValueWorkspace& workspace) const
  {
    const std::size_t ne    = layout_->electronCount();
    const std::size_t na    = layout_->nucleusCount();
    const std::size_t ndet  = layout_->determinantCount();
    const std::size_t width = layout_->featureWidth();
    const auto& distances   = workspace.geometry_.electronNucleusPairs().distances();

    for (std::size_t electron = 0; electron < ne; ++electron)
    {
      const bool spin_up = electron < layout_->spinUpCount();
      const double* backflow = tensor(parameters, spin_up ? layout_->backflow_up_ : layout_->backflow_down_);
      const double* pi       = tensor(parameters, spin_up ? layout_->pi_up_ : layout_->pi_down_);
      const double* zeta     = tensor(parameters, spin_up ? layout_->zeta_up_ : layout_->zeta_down_);
      const double* features = workspace.features_a_.data() + electron * width;

      for (std::size_t determinant = 0; determinant < ndet; ++determinant)
        for (std::size_t orbital = 0; orbital < ne; ++orbital)
        {
          const std::size_t channel = determinant * ne + orbital;
          double backflow_value     = 0;
          for (std::size_t feature = 0; feature < width; ++feature)
            backflow_value += features[feature] * backflow[feature * ndet * ne + channel];

          double envelope_value = 0;
          for (std::size_t nucleus = 0; nucleus < na; ++nucleus)
          {
            const std::size_t parameter = channel * na + nucleus;
            const double radius         = distances[electron * na + nucleus];
            envelope_value += pi[parameter] * std::exp(-std::abs(zeta[parameter] * radius));
          }
          workspace.orbital_matrices_[(determinant * ne + electron) * ne + orbital] =
              backflow_value * envelope_value;
        }
    }
  }

  /// Accumulate same- and opposite-spin analytic electron-electron cusp terms.
  double cuspValue(const double* parameters, const DirectValueWorkspace& workspace) const
  {
    const double same_alpha = layout_->same_alpha_.size == 0
        ? 1.0
        : tensor(parameters, layout_->same_alpha_)[0];
    const double anti_alpha = tensor(parameters, layout_->anti_alpha_)[0];
    const auto& pairs       = workspace.geometry_.electronPairs();
    const auto& distances   = workspace.geometry_.electronElectronPairs().distances();
    double cusp             = 0;
    for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index)
    {
      const ElectronPair pair = pairs[pair_index];
      const bool first_up     = pair.first < layout_->spinUpCount();
      const bool second_up    = pair.second < layout_->spinUpCount();
      const bool same_spin    = first_up == second_up;
      const double alpha      = same_spin ? same_alpha : anti_alpha;
      const double factor     = same_spin ? 0.25 : 0.5;
      cusp -= factor * alpha * alpha / (alpha + distances[pair_index]);
    }
    return cusp;
  }

  const Parameters& parameters_;
  std::shared_ptr<const DirectValueParameterLayout> layout_;
  std::vector<double> nuclei_;
  GeometryBoundary boundary_;
};

} // namespace pf

#endif // QMCPLUSPLUS_PSIFORMER_VALUE_EXECUTOR_H
