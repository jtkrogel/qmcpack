//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerSpatialExecutor.h
 * @brief Shared real open/periodic spatial trunk and guarded complete real evaluator.
 *
 * The executor propagates Cartesian gradients and per-electron Laplacian traces
 * directly through the embedding, attention, orbital, determinant, and cusp layers.
 * It never forms an electron Hessian: a full evaluation stores 3*Ne first-derivative
 * lanes and Ne Laplacian lanes, while an active-electron evaluation stores only three
 * first-derivative lanes. Explicit target VGL reserves 3*maxK/maxK derivative lanes
 * over the complete physical configuration. All storage is clone-local and fixed.
 *
 * Real geometry supports both open and periodic policies. Full real preparation
 * retains the existing orbital/determinant/output arrays; explicit trunk-only
 * preparation exposes borrowed real feature/cusp jets without a real head. This
 * seam does not advertise a complex spatial wavefunction or public XML capability.
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

#include "PsiFormerDenseKernels.h"
#include "PsiFormerOrbitalProjection.h"
#include "QMCWaveFunctions/FrozenReferenceSumMath.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace pf
{

/// Select active gradients, full VGL, or explicitly bounded target VGL.
enum class DirectSpatialMode
{
  ACTIVE_ELECTRON_GRADIENT,
  FULL_VGL,
  TARGET_VGL
};

/// Explicit developer selection; preserve the scalar head until paired timings.
enum class DirectOrbitalProjection
{
  SCALAR,
  GEMM
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

/** Own shared real geometry/trunk jets and cusp scratch, without real head storage.
 *
 * FULL_VGL workspaces carry a 3-vector gradient and scalar Laplacian for every
 * differentiating electron.  ACTIVE_ELECTRON_GRADIENT workspaces omit every
 * second-derivative allocation and can be reused for any active electron.
 */
class DirectSpatialTrunkWorkspace
{
public:
  /// Prepare only real spatial trunk buffers for FULL or ACTIVE differentiation.
  DirectSpatialTrunkWorkspace(const DirectValueParameterLayout& layout,
                         GeometryPositionView nuclei,
                         DirectSpatialMode mode,
                         GeometryBoundary boundary = {})
      : DirectSpatialTrunkWorkspace(requireLegacyMode(layout, mode), nuclei, mode, boundary, 0)
  {}

protected:
  /// Allocate fixed derivative capacity after the caller validates its preparation.
  DirectSpatialTrunkWorkspace(const DirectValueParameterLayout& layout,
                         GeometryPositionView nuclei,
                         DirectSpatialMode mode,
                         GeometryBoundary boundary,
                         std::size_t max_targets)
      : layout_identity_(&layout), mode_(mode),
        target_capacity_(mode == DirectSpatialMode::TARGET_VGL ? max_targets :
            (mode == DirectSpatialMode::FULL_VGL ? layout.electronCount() : 1)),
        target_count_(target_capacity_),
        gradient_lanes_(3 * target_capacity_),
        laplacian_lanes_(mode == DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT ? 0 : target_capacity_),
        electron_positions_(3 * layout.electronCount()),
        geometry_(qmcplusplus::psiformer::periodic_feature::makeGeometry(
            layout.electronCount(), nuclei, boundary, layout.periodicProfile())),
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
        scalar_gradient_scratch_(gradient_lanes_),
        scalar_laplacian_scratch_(laplacian_lanes_)
  {}

public:
  /// Prepared mutable jets cannot be shared by copying.
  DirectSpatialTrunkWorkspace(const DirectSpatialTrunkWorkspace&) = delete;

  /// Preserve exclusive clone-local buffer ownership.
  DirectSpatialTrunkWorkspace& operator=(const DirectSpatialTrunkWorkspace&) = delete;

  /// Transfer the complete real trunk preparation.
  DirectSpatialTrunkWorkspace(DirectSpatialTrunkWorkspace&&) = default;

  /// Transfer ownership together with the immutable layout identity.
  DirectSpatialTrunkWorkspace& operator=(DirectSpatialTrunkWorkspace&&) = default;

  /// Support safe deletion of a complete workspace through its public trunk base.
  virtual ~DirectSpatialTrunkWorkspace() = default;

  /// Copy a complete interleaved configuration into fixed input storage.
  void setPositions(GeometryPositionView positions)
  {
    invalidateResult();
    if (positions.size() != geometry_.electronCount())
      throw std::invalid_argument("PsiFormer spatial workspace received the wrong electron count");
    for (std::size_t electron = 0; electron < positions.size(); ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        electron_positions_[3 * electron + dimension] = positions(electron, dimension);
  }

  /// Overwrite one Cartesian coordinate without changing workspace capacity.
  void setPosition(std::size_t electron, std::size_t dimension, double value)
  {
    invalidateResult();
    if (electron >= geometry_.electronCount() || dimension >= 3)
      throw std::out_of_range("PsiFormer spatial workspace coordinate index is out of range");
    electron_positions_[3 * electron + dimension] = value;
  }

  /// Return the derivative storage mode fixed at workspace construction.
  DirectSpatialMode mode() const noexcept { return mode_; }

  /// Return the parameter version used by the most recent completed evaluation.
  std::size_t observedParameterVersion() const noexcept { return observed_parameter_version_; }

  /// Every edit or attempted evaluation expires the preceding borrowed trunk view.
  std::size_t evaluationGeneration() const noexcept { return evaluation_generation_; }

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
    mix(scalar_gradient_scratch_);
    mix(scalar_laplacian_scratch_);
    return hash;
  }

  /// Report overlap with any retained spatial-workspace allocation.
  bool overlapsStorage(const void* data, std::size_t bytes) const noexcept
  {
    return overlapsVector(electron_positions_, data, bytes) || geometry_.overlapsStorage(data, bytes) ||
        overlapsJet(raw_features_, data, bytes) || overlapsJet(features_a_, data, bytes) ||
        overlapsJet(features_b_, data, bytes) || overlapsJet(query_, data, bytes) ||
        overlapsJet(key_, data, bytes) || overlapsJet(projected_value_, data, bytes) ||
        overlapsJet(attention_, data, bytes) || overlapsJet(attended_, data, bytes) ||
        overlapsJet(hidden_, data, bytes) || overlapsVector(scalar_gradient_scratch_, data, bytes) ||
        overlapsVector(scalar_laplacian_scratch_, data, bytes);
  }

  /// Return bytes reserved by every workspace buffer, including geometry tables.
  std::size_t vectorStorageBytes() const
  {
    std::size_t bytes = 0;
    auto add = [&bytes](const std::vector<double>& buffer) {
      addStorageBytes(bytes,
                      checkedStorageBytes<double>(
                          buffer.capacity(),
                          "PsiFormer spatial storage bytes overflowed"),
                      "PsiFormer spatial storage bytes overflowed");
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
    add(scalar_gradient_scratch_);
    add(scalar_laplacian_scratch_);
    addStorageBytes(bytes, geometry_.storageBytes(),
                    "PsiFormer spatial storage bytes overflowed");
    return bytes;
  }

  /// Expose geometry-cache allocation accounting for workspace diagnostics.
  std::size_t geometryStorageBytes() const
  { return geometry_.storageBytes(); }

  /// Expose geometry allocation identity for focused storage diagnostics.
  std::size_t geometryStorageFingerprint() const noexcept
  { return geometry_.storageFingerprint(); }

  /// Return the checked constructor-time requirement for this fixed model shape.
  std::size_t requiredStorageBytes() const
  {
    std::size_t bytes = geometryStorageRequirement(geometry_.electronCount(), geometry_.nucleusCount());
    auto add = [&bytes](const std::vector<double>& buffer) {
      addStorageBytes(bytes, checkedStorageBytes<double>(buffer.size(), "PsiFormer trunk jet bytes overflow"),
                      "PsiFormer trunk jet bytes overflow");
    };
    add(electron_positions_);
    add(scalar_gradient_scratch_);
    add(scalar_laplacian_scratch_);
    for (const auto* jet : {&raw_features_, &features_a_, &features_b_, &query_, &key_,
                           &projected_value_, &attention_, &attended_, &hidden_})
      for (const auto* buffer : {&jet->value, &jet->gradient, &jet->laplacian})
        add(*buffer);
    return bytes;
  }

protected:
  friend class DirectSpatialExecutor;
  friend class DirectBatchExecutor;

  /// Close the legacy and public trunk constructors before any numeric allocation.
  static const DirectValueParameterLayout& requireLegacyMode(const DirectValueParameterLayout& layout,
                                                            DirectSpatialMode mode)
  {
    if (mode != DirectSpatialMode::FULL_VGL && mode != DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT)
      throw std::invalid_argument("PsiFormer spatial mode requires explicit target preparation");
    return layout;
  }

  /// Invalidate publication before any mutation, including a failed request.
  void invalidateResult()
  {
    observed_parameter_version_ = std::numeric_limits<std::size_t>::max();
    if (evaluation_generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("PsiFormer spatial trunk generation exhausted");
    ++evaluation_generation_;
  }

  /// Preserve fail-closed overlap checks for both shared trunk and derived head buffers.
  static bool overlapsVector(const std::vector<double>& values, const void* data, std::size_t bytes) noexcept
  {
    if (!bytes)
      return false;
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data);
    if (!data || begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
      return true;
    if (!values.capacity())
      return false;
    if (values.capacity() > std::numeric_limits<std::size_t>::max() / sizeof(double))
      return true;
    const auto storage = reinterpret_cast<std::uintptr_t>(values.data());
    const std::size_t size = values.capacity() * sizeof(double);
    return !values.data() || storage > std::numeric_limits<std::uintptr_t>::max() - size ||
        (begin < storage + size && storage < begin + bytes);
  }

  /// Check the three separately allocated value/gradient/Laplacian planes.
  static bool overlapsJet(const DirectSpatialJetBuffer& jet, const void* data, std::size_t bytes) noexcept
  {
    return overlapsVector(jet.value, data, bytes) || overlapsVector(jet.gradient, data, bytes) ||
        overlapsVector(jet.laplacian, data, bytes);
  }

  const DirectValueParameterLayout* layout_identity_;
  DirectSpatialMode mode_;
  std::size_t target_capacity_;
  std::size_t target_begin_ = 0;
  std::size_t target_count_;
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
  std::vector<double> scalar_gradient_scratch_;
  std::vector<double> scalar_laplacian_scratch_;
  std::size_t observed_parameter_version_ = std::numeric_limits<std::size_t>::max();
  std::size_t evaluation_generation_ = 0;
};

/** Retain the complete legacy real orbital/determinant/output buffers over the trunk. */
class DirectSpatialWorkspace : public DirectSpatialTrunkWorkspace
{
public:
  /// Require full real preparation before reserving any real head buffers.
  DirectSpatialWorkspace(const DirectValueParameterLayout& layout, GeometryPositionView nuclei,
      DirectSpatialMode mode, GeometryBoundary boundary = {})
      : DirectSpatialTrunkWorkspace(requireFullLayout(layout), nuclei, mode, boundary),
        orbital_matrices_(layout.determinantCount() * layout.electronCount() * layout.electronCount(),
                          gradient_lanes_, laplacian_lanes_),
        determinant_workspace_(layout.determinantCount(), layout.electronCount(), gradient_lanes_, laplacian_lanes_),
        output_gradient_(gradient_lanes_), output_lap_log_(laplacian_lanes_), output_lap_ratio_(laplacian_lanes_)
  {}

  /// Admit the complete fixed target payload before the base allocates any buffers.
  DirectSpatialWorkspace(const DirectValueParameterLayout& layout, GeometryPositionView nuclei,
      std::size_t max_targets, std::size_t byte_budget, GeometryBoundary boundary = {})
      : DirectSpatialTrunkWorkspace(admitTargetLayout(layout, nuclei, max_targets, byte_budget), nuclei,
            DirectSpatialMode::TARGET_VGL, boundary, max_targets),
        orbital_matrices_(layout.determinantCount() * layout.electronCount() * layout.electronCount(),
                          gradient_lanes_, laplacian_lanes_),
        determinant_workspace_(layout.determinantCount(), layout.electronCount(), gradient_lanes_, laplacian_lanes_),
        output_gradient_(gradient_lanes_), output_lap_log_(laplacian_lanes_), output_lap_ratio_(laplacian_lanes_)
  {}

  /// Select live physical targets without resizing any prepared derivative plane.
  void setTargetRange(std::size_t begin, std::size_t count)
  {
    invalidateResult();
    if (mode_ != DirectSpatialMode::TARGET_VGL)
      throw std::invalid_argument("PsiFormer target range requires a target workspace");
    if (count == 0 || count > target_capacity_ || begin >= geometry_.electronCount() ||
        count > geometry_.electronCount() - begin)
      throw std::out_of_range("PsiFormer target range exceeds prepared capacity or physical electrons");
    target_begin_ = begin;
    target_count_ = count;
    gradient_lanes_ = 3 * count;
    laplacian_lanes_ = count;
  }

  /// Complete real workspaces remain exclusively owned.
  DirectSpatialWorkspace(const DirectSpatialWorkspace&) = delete;

  /// Prevent sharing mutable head scratch through assignment.
  DirectSpatialWorkspace& operator=(const DirectSpatialWorkspace&) = delete;

  /// Transfer all prepared real buffers together.
  DirectSpatialWorkspace(DirectSpatialWorkspace&&) = default;

  /// Transfer the complete real preparation.
  DirectSpatialWorkspace& operator=(DirectSpatialWorkspace&&) = default;

  /// Include head/output allocations in the process-local storage identity.
  std::size_t storageFingerprint() const noexcept
  {
    std::size_t hash = DirectSpatialTrunkWorkspace::storageFingerprint();
    for (const auto* buffer : {&orbital_matrices_.value, &orbital_matrices_.gradient, &orbital_matrices_.laplacian,
                              &output_gradient_, &output_lap_log_, &output_lap_ratio_})
      qmcplusplus::psiformer::determinant::detail::mixStorage(hash, *buffer);
    hash ^= determinant_workspace_.storageFingerprint();
    hash *= 1099511628211ULL;
    return hash;
  }

  /// Preserve complete real-head overlap detection used by existing consumers.
  bool overlapsStorage(const void* data, std::size_t bytes) const noexcept
  {
    return DirectSpatialTrunkWorkspace::overlapsStorage(data, bytes) || overlapsJet(orbital_matrices_, data, bytes) ||
        determinant_workspace_.overlapsStorage(data, bytes) || overlapsVector(output_gradient_, data, bytes) ||
        overlapsVector(output_lap_log_, data, bytes) || overlapsVector(output_lap_ratio_, data, bytes);
  }

  /// Preserve the exact inclusive numeric storage total of the legacy full preparation.
  std::size_t vectorStorageBytes() const
  {
    std::size_t bytes = DirectSpatialTrunkWorkspace::vectorStorageBytes();
    for (const auto* buffer : {&orbital_matrices_.value, &orbital_matrices_.gradient, &orbital_matrices_.laplacian,
                              &output_gradient_, &output_lap_log_, &output_lap_ratio_})
      addStorageBytes(bytes, checkedStorageBytes<double>(buffer->capacity(), "PsiFormer spatial bytes overflow"),
                      "PsiFormer spatial bytes overflow");
    addStorageBytes(bytes, determinant_workspace_.storageBytes(), "PsiFormer spatial bytes overflow");
    return bytes;
  }

  /// Report reserved capacity independently of the current live target range.
  std::size_t requiredStorageBytes() const
  {
    const std::size_t n = geometry_.electronCount();
    return spatialWorkspaceStorageRequirement({n, geometry_.nucleusCount(), determinant_workspace_.channels(),
        features_a_.value.size() / n, attention_.value.size() / checkedStorageProduct(n, n,
        "PsiFormer spatial shape overflowed"), raw_features_.value.size() / n, 0, 0},
        3 * target_capacity_, mode_ == DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT ? 0 : target_capacity_);
  }

private:
  friend class DirectSpatialExecutor;
  friend class DirectBatchExecutor;

  /// Reject a trunk-only parameter layout before constructing the base allocations.
  static const DirectValueParameterLayout& requireFullLayout(const DirectValueParameterLayout& layout)
  {
    if (layout.preparation() != DirectValuePreparation::REAL_WAVEFUNCTION)
      throw std::invalid_argument("PsiFormer real spatial workspace requires full real preparation");
    return layout;
  }

  /// Check exact numeric payload and a positive finite cap before base construction.
  static const DirectValueParameterLayout& admitTargetLayout(const DirectValueParameterLayout& layout,
      GeometryPositionView nuclei, std::size_t max_targets, std::size_t byte_budget)
  {
    requireFullLayout(layout);
    if (nuclei.size() != layout.nucleusCount())
      throw std::invalid_argument("PsiFormer target workspace received the wrong nucleus count");
    if (byte_budget == 0 || byte_budget == std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("PsiFormer target byte budget must be positive and finite");
    const std::size_t bytes = targetSpatialWorkspaceStorageRequirement(
        {layout.electronCount(), layout.nucleusCount(), layout.determinantCount(), layout.featureWidth(),
         layout.headCount(), layout.inputWidth(), 0, 0}, max_targets);
    if (bytes > byte_budget)
      throw std::length_error("PsiFormer target workspace exceeds its byte budget");
    return layout;
  }

  DirectSpatialJetBuffer orbital_matrices_;
  qmcplusplus::psiformer::determinant::RealOpenDeterminantWorkspace determinant_workspace_;
  std::vector<double> output_gradient_, output_lap_log_, output_lap_ratio_;
};

/** Borrow real feature/cusp jets until the next workspace edit or evaluation.
 * Match layout identity, current parameter version and evaluation generation before
 * reusing the view. Geometry and positions refer to the same physical configuration.
 */
struct DirectSpatialTrunkResultView
{
  const DirectSpatialJetBuffer* features;
  std::size_t electron_count, feature_width;
  DirectSpatialMode mode;
  std::size_t active_electron;
  double cusp;
  DirectConstDoubleView cusp_gradient, cusp_laplacian;
  const PsiFormerGeometryCache* geometry;
  const double* packed_positions;
  std::size_t parameter_version, evaluation_generation;
};

/** Borrow spatial outputs until the next edit/evaluation; match parameter version
 * and evaluation generation before reuse. Target ranges name physical electrons.
 */
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
  std::size_t target_begin = 0, target_count = 0;
  std::size_t evaluation_generation = 0;
};

/** Evaluate real PsiFormer spatial observables with boundary-aware direct kernels. */
class DirectSpatialExecutor
{
public:
  /// Resolve immutable parameter descriptors and retain the versioned value store.
  DirectSpatialExecutor(const PsiFormer& model,
                        const DirectValueExecutor& value_executor,
                        const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan,
                        DirectOrbitalProjection orbital_projection = DirectOrbitalProjection::SCALAR,
                        DirectValuePreparation preparation = DirectValuePreparation::REAL_WAVEFUNCTION)
      : parameters_(model.p),
        layout_(preparation == DirectValuePreparation::REAL_TRUNK_ONLY ?
                    value_executor.trunkLayout() : value_executor.layout()),
        nuclei_(model.cfg.nuclei.x),
        boundary_(directGeometryBoundary(plan.environment())),
        value_executor_identity_(&value_executor),
        orbital_projection_(orbital_projection), preparation_(preparation)
  {
    if (preparation != DirectValuePreparation::REAL_WAVEFUNCTION &&
        preparation != DirectValuePreparation::REAL_TRUNK_ONLY)
      throw std::invalid_argument("Unknown PsiFormer spatial preparation profile");
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
    requireRealWavefunction();
    return std::make_unique<DirectSpatialWorkspace>(
        *layout_, GeometryPositionView::interleaved(nuclei_.data(), layout_->nucleusCount()), mode, boundary_);
  }

  /// Prepare a bounded real owner with all physical rows and only target derivative planes.
  std::unique_ptr<DirectSpatialWorkspace> makeTargetWorkspace(std::size_t max_targets,
                                                           std::size_t byte_budget) const
  {
    requireRealWavefunction();
    return std::make_unique<DirectSpatialWorkspace>(*layout_,
        GeometryPositionView::interleaved(nuclei_.data(), layout_->nucleusCount()),
        max_targets, byte_budget, boundary_);
  }

  /// Expose the immutable descriptor identity used to bind batch executors safely.
  const std::shared_ptr<const DirectValueParameterLayout>& layout() const
  {
    requireRealWavefunction();
    return layout_;
  }

  /// Expose immutable real trunk metadata through an explicit mixed-head-aware seam.
  const std::shared_ptr<const DirectValueParameterLayout>& trunkLayout() const noexcept { return layout_; }

  /// Prepare FULL or ACTIVE real jets without legacy orbital/determinant allocations.
  std::unique_ptr<DirectSpatialTrunkWorkspace> makeTrunkWorkspace(DirectSpatialMode mode) const
  {
    return std::make_unique<DirectSpatialTrunkWorkspace>(*layout_,
        GeometryPositionView::interleaved(nuclei_.data(), layout_->nucleusCount()), mode, boundary_);
  }

  /// Admit and allocate only compact target trunk jets, retaining all physical rows.
  std::unique_ptr<DirectSpatialTrunkWorkspace> makeTargetTrunkWorkspace(std::size_t max_targets,
                                                                     std::size_t byte_budget) const
  {
    if (!max_targets || max_targets > layout_->electronCount() || !byte_budget ||
        byte_budget == std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("PsiFormer target trunk requires valid targets and a finite byte cap");
    const PsiFormerStorageShape shape{layout_->electronCount(), layout_->nucleusCount(), layout_->determinantCount(),
        layout_->featureWidth(), layout_->headCount(), layout_->inputWidth(), 0, 0};
    if (spatialTrunkStorageRequirement(shape, checkedStorageProduct(3, max_targets,
            "PsiFormer target trunk gradient overflow"), max_targets) > byte_budget)
      throw std::length_error("PsiFormer target trunk exceeds its byte cap before allocation");
    return std::unique_ptr<DirectSpatialTrunkWorkspace>(new DirectSpatialTrunkWorkspace(*layout_,
        GeometryPositionView::interleaved(nuclei_.data(), layout_->nucleusCount()),
        DirectSpatialMode::TARGET_VGL, boundary_, max_targets));
  }

  /// Retarget one fixed trunk bank without changing any numeric allocation.
  DirectSpatialTrunkResultView evaluateTrunkTarget(DirectSpatialTrunkWorkspace& workspace,
                                                   std::size_t begin, std::size_t count) const
  {
    workspace.invalidateResult();
    if (workspace.mode_ != DirectSpatialMode::TARGET_VGL || !count || count > workspace.target_capacity_ ||
        begin >= layout_->electronCount() || count > layout_->electronCount()-begin)
      throw std::invalid_argument("PsiFormer target trunk interval exceeds its prepared capacity");
    workspace.target_begin_ = begin;
    workspace.target_count_ = count;
    workspace.gradient_lanes_ = 3*count;
    workspace.laplacian_lanes_ = count;
    return evaluateTrunkImpl(workspace);
  }

  /// Publish all physical-electron feature/cusp jets after complete real trunk success.
  DirectSpatialTrunkResultView evaluateTrunkFull(DirectSpatialTrunkWorkspace& workspace) const
  {
    workspace.invalidateResult();
    if (workspace.mode_ != DirectSpatialMode::FULL_VGL)
      throw std::invalid_argument("PsiFormer full trunk requires a FULL_VGL workspace");
    workspace.active_electron_ = 0;
    return evaluateTrunkImpl(workspace);
  }

  /// Publish only one physical electron's three first-derivative lanes.
  DirectSpatialTrunkResultView evaluateTrunkActive(DirectSpatialTrunkWorkspace& workspace,
                                                 std::size_t active_electron) const
  {
    workspace.invalidateResult();
    if (workspace.mode_ != DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT || active_electron >= layout_->electronCount())
      throw std::invalid_argument("PsiFormer active trunk request has invalid mode or physical electron");
    workspace.active_electron_ = active_electron;
    return evaluateTrunkImpl(workspace);
  }

  /// Return the exact value executor whose immutable model state is shared here.
  const DirectValueExecutor* valueExecutorIdentity() const noexcept
  { return value_executor_identity_; }

  /// Evaluate complete log-gradient and per-electron log-Laplacian data.
  DirectSpatialResultView evaluateFull(DirectSpatialWorkspace& workspace) const
  {
    workspace.invalidateResult();
    requireRealWavefunction();
    if (workspace.mode_ != DirectSpatialMode::FULL_VGL)
      throw std::invalid_argument("PsiFormer full VGL requires a FULL_VGL workspace");
    workspace.active_electron_ = 0;
    return evaluateImpl(workspace);
  }

  /** Return scaled amplitude derivatives without dividing by an isolated neural value.
   * The bounded small-system path is used only by a frozen-reference CPU sum. */
  qmcplusplus::residual::ScaledSpatialJet evaluateAmplitudeFull(DirectSpatialWorkspace& workspace) const;

  /// Evaluate one admitted physical target interval using the existing complete algebra.
  DirectSpatialResultView evaluateTarget(DirectSpatialWorkspace& workspace,
                                        std::size_t begin, std::size_t count) const
  {
    workspace.setTargetRange(begin, count);
    requireRealWavefunction();
    return evaluateImpl(workspace);
  }

  /// Evaluate only the three Cartesian log-gradient components of one electron.
  DirectSpatialResultView evaluateActive(DirectSpatialWorkspace& workspace,
                                         std::size_t active_electron) const
  {
    workspace.invalidateResult();
    requireRealWavefunction();
    if (workspace.mode_ != DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT)
      throw std::invalid_argument("PsiFormer active gradient requires an active-gradient workspace");
    if (active_electron >= layout_->electronCount())
      throw std::out_of_range("PsiFormer active electron is out of range");
    workspace.active_electron_ = active_electron;
    return evaluateImpl(workspace);
  }

private:
  friend class DirectBatchExecutor;

  /// Keep complete real head consumers closed to explicitly trunk-only preparation.
  void requireRealWavefunction() const
  {
    if (preparation_ != DirectValuePreparation::REAL_WAVEFUNCTION)
      throw std::invalid_argument("PsiFormer real spatial consumer cannot use trunk-only preparation");
  }

  /// Run the same embedding/attention loop for either real head or borrowed trunk jets.
  void buildTrunk(const double* parameters, DirectSpatialTrunkWorkspace& workspace) const
  {
    buildEmbedding(parameters, workspace);
    for (const auto& layer : layout_->layers_)
      applyAttentionBlock(parameters, layer, workspace);
  }

  /// Bind identity/geometry and publish only finite real feature/cusp jets.
  DirectSpatialTrunkResultView evaluateTrunkImpl(DirectSpatialTrunkWorkspace& workspace) const
  {
    layout_->validateParameterStore(parameters_);
    const auto& boundary = workspace.geometry_.boundary();
    if (workspace.layout_identity_ != layout_.get() || workspace.geometry_.electronCount() != layout_->electronCount() ||
        workspace.geometry_.nucleusCount() != layout_->nucleusCount() || boundary.kind != boundary_.kind ||
        boundary.lattice_vectors != boundary_.lattice_vectors || boundary.periodic_axes != boundary_.periodic_axes)
      throw std::invalid_argument("PsiFormer spatial trunk workspace has a different identity or geometry");
    for (std::size_t nucleus = 0; nucleus < layout_->nucleusCount(); ++nucleus)
      for (std::size_t dim = 0; dim < 3; ++dim)
        if (workspace.geometry_.nuclei()[nucleus][dim] != nuclei_[3 * nucleus + dim])
          throw std::invalid_argument("PsiFormer spatial trunk workspace has different nuclei");
    const double* parameters = parameters_.flat_values().data();
    refreshGeometry(workspace);
    buildTrunk(parameters, workspace);
    const double cusp = accumulateCusp(parameters, workspace);
    if (!is_finite_parameter_value(cusp))
      throw std::runtime_error("PsiFormer spatial trunk cusp is nonfinite");
    for (const auto* buffer : {&workspace.features_a_.value, &workspace.features_a_.gradient,
        &workspace.features_a_.laplacian, &workspace.scalar_gradient_scratch_, &workspace.scalar_laplacian_scratch_})
      for (double value : *buffer)
        if (!is_finite_parameter_value(value))
          throw std::runtime_error("PsiFormer spatial trunk jets are nonfinite");
    workspace.observed_parameter_version_ = parameters_.version();
    return {&workspace.features_a_, layout_->electronCount(), layout_->featureWidth(), workspace.mode_,
        workspace.active_electron_, cusp, {workspace.scalar_gradient_scratch_.data(), workspace.gradient_lanes_},
        {workspace.scalar_laplacian_scratch_.data(), workspace.laplacian_lanes_}, &workspace.geometry_,
        workspace.electron_positions_.data(), workspace.observed_parameter_version_, workspace.evaluation_generation_};
  }

  /// Return the beginning of one pre-resolved immutable parameter interval.
  static const double* tensor(const double* parameters,
                              const DirectParameterTensor& descriptor) noexcept
  {
    return parameters + descriptor.begin;
  }

  /// Return the physical electron associated with one Cartesian gradient lane.
  static std::size_t laneElectron(const DirectSpatialTrunkWorkspace& workspace,
                                  std::size_t lane) noexcept
  {
    return workspace.mode_ == DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT ? workspace.active_electron_ :
        workspace.target_begin_ + lane / 3;
  }

  /// Return the x/y/z component associated with one Cartesian gradient lane.
  static std::size_t laneDimension(const DirectSpatialTrunkWorkspace& workspace,
                                   std::size_t lane) noexcept
  {
    return workspace.mode_ == DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT ? lane : lane % 3;
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
  void refreshGeometry(DirectSpatialTrunkWorkspace& workspace) const;

  /// Build analytic electron-nucleus features and their spatial derivatives.
  void buildEmbedding(const double* parameters, DirectSpatialTrunkWorkspace& workspace) const;

  /// Populate the embedding input jet without applying its shared dense projection.
  void buildEmbeddingFeatures(DirectSpatialTrunkWorkspace& workspace) const;

  /// Apply one attention/residual block to value, gradient, and Laplacian lanes.
  void applyAttentionBlock(const double* parameters,
                           const DirectValueParameterLayout::DirectAttentionLayer& layer,
                           DirectSpatialTrunkWorkspace& workspace) const;

  /// Form determinant orbital matrices including spatial derivative products.
  void buildOrbitalMatrices(const double* parameters, DirectSpatialWorkspace& workspace) const;

  /// Project directly into the existing orbital value/gradient/Laplacian vectors.
  void projectBackflowMatrices(const double* parameters, DirectSpatialWorkspace& workspace) const;

  /// Add analytic electron-electron cusp value and spatial derivatives in place.
  double accumulateCusp(const double* parameters, DirectSpatialTrunkWorkspace& workspace) const;

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
  void buildAttentionWeights(const double* parameters,
      const DirectValueParameterLayout::DirectAttentionLayer& layer, DirectSpatialTrunkWorkspace& workspace) const;

  /// Contract attention weights with value heads using spatial product rules.
  void buildAttentionContext(DirectSpatialTrunkWorkspace& workspace) const;

  const Parameters& parameters_;
  std::shared_ptr<const DirectValueParameterLayout> layout_;
  std::vector<double> nuclei_;
  GeometryBoundary boundary_;
  const DirectValueExecutor* value_executor_identity_;
  DirectOrbitalProjection orbital_projection_;
  DirectValuePreparation preparation_;
};

inline void DirectSpatialExecutor::refreshGeometry(DirectSpatialTrunkWorkspace& workspace) const
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
    DirectSpatialTrunkWorkspace& workspace) const
{
  DirectSpatialJetBuffer& raw       = workspace.raw_features_;
  const GeometryPairTable& pairs    = workspace.geometry_.electronNucleusPairs();
  const auto& displacements         = pairs.displacements();
  const auto& complementary         = pairs.complementaryDisplacements();
  const auto& displacement_jacobians = pairs.displacementJacobians();
  const auto& complementary_jacobians = pairs.complementaryDisplacementJacobians();
  const auto& displacement_laplacians = pairs.displacementLaplacians();
  const auto& complementary_laplacians = pairs.complementaryDisplacementLaplacians();
  const auto& distances             = pairs.distances();
  const auto& distance_gradients    = pairs.distanceGradients();
  const auto& distance_gradient_norms = pairs.distanceGradientNormsSquared();
  const auto& distance_laplacians   = pairs.distanceLaplacians();
  const auto& factors               = pairs.softenedRadialFactors();
  const std::size_t electron_count  = layout_->electronCount();
  const std::size_t nucleus_count   = layout_->nucleusCount();
  const std::size_t input_width     = layout_->inputWidth();
  const bool periodic = workspace.geometry_.boundary().kind == GeometryBoundaryKind::PERIODIC;
  const std::size_t pair_width = periodic ? 7 : 4;
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
      if (layout_->periodic_profile_)
      {
        const auto jet = qmcplusplus::psiformer::periodic_feature::evaluate(pairs, pair,
            workspace.geometry_.electronIonBoundary(), 1, layout_->periodic_profile_->feature_transform, 1);
        if (!jet.differentiable)
          throw std::runtime_error("Periodic feature spatial jets are undefined at a non-smooth coordinate plane");
        const auto arity = qmcplusplus::psiformer::periodicFeatureArity(layout_->periodic_profile_->feature_transform);
        const auto begin = row_begin + arity * nucleus;
        std::copy_n(jet.value.data(), arity, raw.value.data() + begin);
        for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
          if (laneElectron(workspace, lane) == electron)
            for (std::size_t f = 0; f < arity; ++f)
              raw.gradient[gradientIndex(raw, lane, begin + f)] = jet.gradient[f][laneDimension(workspace, lane)];
        if (electron >= workspace.target_begin_ && electron - workspace.target_begin_ < workspace.laplacian_lanes_)
          for (std::size_t f = 0; f < arity; ++f)
            raw.laplacian[laplacianIndex(raw, electron - workspace.target_begin_, begin + f)] = jet.laplacian[f];
        continue;
      }
      const GeometryPosition& displacement = displacements[pair];
      const SoftenedRadialFactors& radial  = factors[pair];
      const std::size_t radial_element     = row_begin + pair_width * nucleus;
      raw.value[radial_element]            = radial.log1p_radius;
      for (std::size_t component = 0; component < 3; ++component)
        raw.value[radial_element + 1 + component] =
            displacement[component] * radial.log1p_over_radius;
      if (periodic)
        for (std::size_t component = 0; component < 3; ++component)
          raw.value[radial_element + 4 + component] =
              complementary[pair][component] * radial.log1p_over_radius;

      for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
      {
        if (laneElectron(workspace, lane) != electron)
          continue;
        const std::size_t derivative_dimension = laneDimension(workspace, lane);
        const double radius_gradient = distance_gradients[pair][derivative_dimension];
        raw.gradient[gradientIndex(raw, lane, radial_element)] =
            radial.log1p_first * radius_gradient;
        for (std::size_t component = 0; component < 3; ++component)
        {
          raw.gradient[gradientIndex(raw, lane, radial_element + 1 + component)] =
              displacement_jacobians[pair][component][derivative_dimension] *
                  radial.log1p_over_radius +
              displacement[component] * radial.log1p_over_radius_first * radius_gradient;
          if (periodic)
            raw.gradient[gradientIndex(raw, lane, radial_element + 4 + component)] =
                complementary_jacobians[pair][component][derivative_dimension] *
                    radial.log1p_over_radius +
                complementary[pair][component] * radial.log1p_over_radius_first *
                    radius_gradient;
        }
      }

      if (electron >= workspace.target_begin_ &&
          electron - workspace.target_begin_ < workspace.laplacian_lanes_)
      {
        const std::size_t target = electron - workspace.target_begin_;
        raw.laplacian[laplacianIndex(raw, target, radial_element)] =
            radial.log1p_second * distance_gradient_norms[pair] +
            radial.log1p_first * distance_laplacians[pair];
        for (std::size_t component = 0; component < 3; ++component)
        {
          double displacement_gradient_dot_radius_gradient = 0;
          double complementary_gradient_dot_radius_gradient = 0;
          for (std::size_t derivative = 0; derivative < 3; ++derivative)
          {
            displacement_gradient_dot_radius_gradient +=
                displacement_jacobians[pair][component][derivative] *
                distance_gradients[pair][derivative];
            complementary_gradient_dot_radius_gradient +=
                complementary_jacobians[pair][component][derivative] *
                distance_gradients[pair][derivative];
          }
          const double radial_trace =
              radial.log1p_over_radius_second * distance_gradient_norms[pair] +
              radial.log1p_over_radius_first * distance_laplacians[pair];
          raw.laplacian[laplacianIndex(raw, target, radial_element + 1 + component)] =
              displacement_laplacians[pair][component] * radial.log1p_over_radius +
              2 * radial.log1p_over_radius_first * displacement_gradient_dot_radius_gradient +
              displacement[component] * radial_trace;
          if (periodic)
            raw.laplacian[laplacianIndex(raw, target, radial_element + 4 + component)] =
                complementary_laplacians[pair][component] * radial.log1p_over_radius +
                2 * radial.log1p_over_radius_first * complementary_gradient_dot_radius_gradient +
                complementary[pair][component] * radial_trace;
        }
      }
    }
    raw.value[row_begin + input_width - 1] = electron < layout_->spinUpCount() ? 1.0 : -1.0;
  }

}

inline void DirectSpatialExecutor::buildEmbedding(
    const double* parameters,
    DirectSpatialTrunkWorkspace& workspace) const
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

inline void DirectSpatialExecutor::buildAttentionWeights(const double* parameters,
    const DirectValueParameterLayout::DirectAttentionLayer& layer, DirectSpatialTrunkWorkspace& workspace) const
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

  if (layout_->periodic_profile_)
  {
    const auto* weights = tensor(parameters, layer.pair_weight);
    const auto* bias = tensor(parameters, layer.pair_bias);
    for (std::size_t pair = 0; pair < workspace.geometry_.electronPairs().size(); ++pair)
    {
      const auto& identity = workspace.geometry_.electronPairs()[pair];
      const double spin = (identity.first < layout_->spinUpCount()) ==
          (identity.second < layout_->spinUpCount()) ? 1 : -1;
      for (int orientation : {1, -1})
      {
        const auto jet = qmcplusplus::psiformer::periodic_feature::evaluate(
            workspace.geometry_.electronElectronPairs(), pair, workspace.geometry_.boundary(), orientation,
            layout_->periodic_profile_->feature_transform, spin);
        if (!jet.differentiable)
          throw std::runtime_error("Pair-attention spatial jets are undefined at non-smooth geometry");
        const auto row = orientation == 1 ? identity.first : identity.second;
        const auto column = orientation == 1 ? identity.second : identity.first;
        for (std::size_t head = 0; head < head_count; ++head)
        {
          const auto output = attention_index(head, row, column);
          double value = bias[head], laplacian = 0;
          GeometryPosition gradient{};
          for (std::size_t f = 0; f < layout_->periodic_profile_->pair_feature_dimension; ++f)
          {
            const auto weight = weights[f * head_count + head];
            value += weight * jet.value[f];
            laplacian += weight * jet.laplacian[f];
            for (std::size_t d = 0; d < 3; ++d)
              gradient[d] += weight * jet.gradient[f][d];
          }
          attention.value[output] += value;
          for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
          {
            const auto target = laneElectron(workspace, lane);
            const double sign = target == row ? 1 : (target == column ? -1 : 0);
            attention.gradient[gradientIndex(attention, lane, output)] += sign * gradient[laneDimension(workspace, lane)];
          }
          for (std::size_t target = 0; target < workspace.laplacian_lanes_; ++target)
            if (workspace.target_begin_ + target == row || workspace.target_begin_ + target == column)
              attention.laplacian[laplacianIndex(attention, target, output)] += laplacian;
        }
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

inline void DirectSpatialExecutor::buildAttentionContext(DirectSpatialTrunkWorkspace& workspace) const
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
    DirectSpatialTrunkWorkspace& workspace) const
{
  const std::size_t electron_count = layout_->electronCount();
  const std::size_t feature_width  = layout_->featureWidth();
  denseQKVJet(workspace.features_a_, tensor(parameters, layer.query), tensor(parameters, layer.key),
              tensor(parameters, layer.value), electron_count, feature_width,
              workspace.gradient_lanes_, workspace.laplacian_lanes_, workspace.query_, workspace.key_,
              workspace.projected_value_);
  buildAttentionWeights(parameters, layer, workspace);
  buildAttentionContext(workspace);

  denseJet(workspace.attended_, tensor(parameters, layer.projection), nullptr, electron_count,
           feature_width, feature_width, workspace.gradient_lanes_, workspace.laplacian_lanes_,
           workspace.features_b_);
  for (std::size_t element = 0; element < workspace.features_b_.value.size(); ++element)
    workspace.features_b_.value[element] += workspace.features_a_.value[element];
  for (std::size_t element = 0; element < workspace.gradient_lanes_ * workspace.features_b_.value.size(); ++element)
    workspace.features_b_.gradient[element] += workspace.features_a_.gradient[element];
  for (std::size_t element = 0; element < workspace.laplacian_lanes_ * workspace.features_b_.value.size(); ++element)
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
  for (std::size_t element = 0; element < workspace.gradient_lanes_ * workspace.features_b_.value.size(); ++element)
    workspace.features_b_.gradient[element] += workspace.attended_.gradient[element];
  for (std::size_t element = 0; element < workspace.laplacian_lanes_ * workspace.features_b_.value.size(); ++element)
    workspace.features_b_.laplacian[element] += workspace.attended_.laplacian[element];
  std::swap(workspace.features_a_, workspace.features_b_);
}

// Each source plane and destination plane live in separate prepared vectors.
// The value-only descriptor supplies the same spin/channel slice schedule for
// every plane, without treating the vectors as one contiguous allocation.
inline void DirectSpatialExecutor::projectBackflowMatrices(const double* parameters,
                                                           DirectSpatialWorkspace& workspace) const
{
  namespace p = qmcplusplus::psiformer;
  const auto projection = p::makeOpenOrbitalJetLayout(1, layout_->determinantCount(),
      layout_->electronCount(), layout_->nucleusCount(), layout_->featureWidth(),
      layout_->spinUpCount(), p::SpatialJetMode::VALUE_ONLY);
  const auto& source = workspace.features_a_;
  auto& target = workspace.orbital_matrices_;
  const double* up = tensor(parameters, layout_->backflow_up_);
  const double* down = tensor(parameters, layout_->backflow_down_);

  // Reuse the row-major BLAS ABI with padded weight rows and D-major outputs.
  p::forEachOpenOrbitalProjection(projection, [&](const p::OpenOrbitalProjectionSlice& slice) {
    const auto& d = slice.dense;
    const auto dimension = [](std::size_t extent) {
      return p::dense::checkedBlasDimension(extent, "PsiFormer orbital projection exceeds BLAS integer ABI");
    };
    const int rows = dimension(d.rows), input = dimension(d.input_width), output = dimension(d.output_width);
    const int source_stride = dimension(d.source_row_stride);
    const int weight_stride = dimension(d.weight_row_stride), target_stride = dimension(d.target_row_stride);
    const double* weight = (slice.spin_up ? up : down) + slice.weight_offset;
    const auto apply = [&](const double* input_plane, double* output_plane) {
      BLAS::gemm('N', 'N', output, rows, input, 1.0, weight, weight_stride,
          input_plane + slice.source_offset, source_stride, 0.0,
          output_plane + slice.target_offset, target_stride);
    };
    apply(source.value.data(), target.value.data());
    for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
      apply(source.gradient.data() + lane * source.value.size(),
          target.gradient.data() + lane * target.value.size());
    for (std::size_t lane = 0; lane < workspace.laplacian_lanes_; ++lane)
      apply(source.laplacian.data() + lane * source.value.size(),
          target.laplacian.data() + lane * target.value.size());
  });
}

inline void DirectSpatialExecutor::buildOrbitalMatrices(const double* parameters,
                                                        DirectSpatialWorkspace& workspace) const
{
  DirectSpatialJetBuffer& orbitals    = workspace.orbital_matrices_;
  const DirectSpatialJetBuffer& features = workspace.features_a_;
  const GeometryPairTable& pairs      = workspace.geometry_.electronNucleusPairs();
  const auto& distances               = pairs.distances();
  const auto& distance_gradients      = pairs.distanceGradients();
  const auto& distance_gradient_norms = pairs.distanceGradientNormsSquared();
  const auto& distance_laplacians     = pairs.distanceLaplacians();
  const std::size_t electron_count    = layout_->electronCount();
  const std::size_t nucleus_count     = layout_->nucleusCount();
  const std::size_t determinant_count = layout_->determinantCount();
  const std::size_t feature_width     = layout_->featureWidth();
  orbitals.clear();
  const bool projected = orbital_projection_ == DirectOrbitalProjection::GEMM;
  if (projected)
    projectBackflowMatrices(parameters, workspace);

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
        double backflow_value = projected ? orbitals.value[matrix_element] : 0.0;
        for (std::size_t feature = 0; !projected && feature < feature_width; ++feature)
          backflow_value += features.value[feature_begin + feature] *
              backflow[feature * determinant_count * electron_count + channel];

        double envelope_value = 0;
        std::fill(workspace.scalar_gradient_scratch_.begin(), workspace.scalar_gradient_scratch_.end(), 0.0);
        std::fill(workspace.scalar_laplacian_scratch_.begin(), workspace.scalar_laplacian_scratch_.end(), 0.0);
        for (std::size_t nucleus = 0; nucleus < nucleus_count; ++nucleus)
        {
          const std::size_t parameter = channel * nucleus_count + nucleus;
          const std::size_t pair      = row_electron * nucleus_count + nucleus;
          const auto selected = qmcplusplus::psiformer::periodic_feature::selectedRadius(
              pairs, pair, workspace.geometry_.electronIonBoundary(), layout_->periodic_profile_);
          const double radius = selected.value;
          if (!selected.differentiable)
            throw std::runtime_error(
                "PsiFormer spatial derivatives are undefined at an electron-nucleus coalescence");
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
                  radial_first * selected.gradient[dimension];
            }
          if (row_electron >= workspace.target_begin_ &&
              row_electron - workspace.target_begin_ < workspace.laplacian_lanes_)
            workspace.scalar_laplacian_scratch_[row_electron - workspace.target_begin_] +=
                radial_second * selected.gradient_norm_squared + radial_first * selected.laplacian;
        }

        // Consume all projected gradients in the Laplacian product before the
        // gradient loop overwrites them. Only the existing buffers are needed.
        for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
        {
          double backflow_laplacian = projected
              ? orbitals.laplacian[laplacianIndex(orbitals, electron, matrix_element)] : 0.0;
          double gradient_dot       = 0;
          for (std::size_t feature = 0; !projected && feature < feature_width; ++feature)
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
          if (projected)
            for (std::size_t dimension = 0; dimension < 3; ++dimension)
              gradient_dot += orbitals.gradient[
                  gradientIndex(orbitals, 3 * electron + dimension, matrix_element)] *
                  workspace.scalar_gradient_scratch_[3 * electron + dimension];
          orbitals.laplacian[laplacianIndex(orbitals, electron, matrix_element)] =
              backflow_laplacian * envelope_value + 2.0 * gradient_dot +
              backflow_value * workspace.scalar_laplacian_scratch_[electron];
        }

        // The affine value stays local until all derivative product terms finish.
        for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
        {
          double backflow_gradient = projected
              ? orbitals.gradient[gradientIndex(orbitals, lane, matrix_element)] : 0.0;
          for (std::size_t feature = 0; !projected && feature < feature_width; ++feature)
            backflow_gradient += features.gradient[gradientIndex(features, lane, feature_begin + feature)] *
                backflow[feature * determinant_count * electron_count + channel];
          orbitals.gradient[gradientIndex(orbitals, lane, matrix_element)] =
              backflow_gradient * envelope_value +
              backflow_value * workspace.scalar_gradient_scratch_[lane];
        }
        orbitals.value[matrix_element] = backflow_value * envelope_value;
      }
  }
}

inline double DirectSpatialExecutor::accumulateCusp(const double* parameters,
                                                    DirectSpatialTrunkWorkspace& workspace) const
{
  std::fill(workspace.scalar_gradient_scratch_.begin(), workspace.scalar_gradient_scratch_.end(), 0.0);
  std::fill(workspace.scalar_laplacian_scratch_.begin(), workspace.scalar_laplacian_scratch_.end(), 0.0);
  if (layout_->periodic_profile_ && !layout_->periodic_profile_->electron_electron_cusp)
    return 0;
  const double same_alpha = layout_->same_alpha_.size == 0
      ? 1.0
      : tensor(parameters, layout_->same_alpha_)[0];
  const double opposite_alpha = tensor(parameters, layout_->anti_alpha_)[0];
  const auto& identities      = workspace.geometry_.electronPairs();
  const GeometryPairTable& pair_table = workspace.geometry_.electronElectronPairs();
  const auto& distances     = pair_table.distances();
  const auto& distance_gradients = pair_table.distanceGradients();
  const auto& distance_gradient_norms = pair_table.distanceGradientNormsSquared();
  const auto& distance_laplacians = pair_table.distanceLaplacians();
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
    if (layout_->residual_pair_)
    {
      const auto& positions = workspace.geometry_.electrons();
      const auto displacement = layout_->residual_pair_->displacement(positions[pair.first], positions[pair.second]);
      const auto jet = layout_->residual_pair_->evaluate(displacement, alpha, same_spin);
      cusp_value += jet.value;
      for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
      {
        const auto electron = laneElectron(workspace, lane);
        if (electron == pair.first || electron == pair.second)
          workspace.scalar_gradient_scratch_[lane] += (electron == pair.first ? 1 : -1) *
              jet.radial_first_over_r * displacement[laneDimension(workspace, lane)];
      }
      for (const std::size_t electron : {pair.first, pair.second})
        if (electron >= workspace.target_begin_ &&
            electron - workspace.target_begin_ < workspace.laplacian_lanes_)
          workspace.scalar_laplacian_scratch_[electron - workspace.target_begin_] +=
              jet.radial_second + 2 * jet.radial_first_over_r;
      continue;
    }
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
          distance_gradients[pair_index][dimension];
    }
    if (workspace.laplacian_lanes_ != 0)
    {
      const double pair_laplacian =
          radial_second * distance_gradient_norms[pair_index] +
          radial_first * distance_laplacians[pair_index];
      for (const std::size_t electron : {pair.first, pair.second})
        if (electron >= workspace.target_begin_ &&
            electron - workspace.target_begin_ < workspace.laplacian_lanes_)
          workspace.scalar_laplacian_scratch_[electron - workspace.target_begin_] += pair_laplacian;
    }
  }
  return cusp_value;
}

/** Evaluate the ordinary trunk/head but replace logarithmic determinant algebra by jets.
 * Division-free small determinants retain derivatives through individual channel nodes.
 */
inline qmcplusplus::residual::ScaledSpatialJet DirectSpatialExecutor::evaluateAmplitudeFull(
    DirectSpatialWorkspace& workspace) const
{
  namespace residual = qmcplusplus::residual;
  workspace.invalidateResult();
  requireRealWavefunction();
  const std::size_t electrons = layout_->electronCount();
  if (workspace.mode_ != DirectSpatialMode::FULL_VGL || electrons == 0 || electrons > 12)
    throw std::invalid_argument("Residual neural amplitude requires a CPU FULL workspace with 1 through 12 electrons");
  layout_->validateParameterStore(parameters_);
  if (workspace.geometry_.electronCount() != electrons ||
      workspace.geometry_.nucleusCount() != layout_->nucleusCount())
    throw std::invalid_argument("Residual neural amplitude workspace belongs to a different model");
  workspace.active_electron_ = 0;
  const double* values = parameters_.flat_values().data();
  refreshGeometry(workspace);
  buildTrunk(values, workspace);
  buildOrbitalMatrices(values, workspace);

  // All channels share the same electron-space derivative lanes and physical cusp.
  const auto& orbitals = workspace.orbital_matrices_;
  const std::size_t matrix_elements = electrons * electrons;
  const std::size_t all_elements = orbitals.value.size();
  residual::ScaledSpatialJet amplitude{0, residual::SpatialJet(electrons)};
  for (std::size_t channel = 0; channel < layout_->determinantCount(); ++channel)
  {
    std::vector<residual::SpatialJet> matrix(matrix_elements, residual::SpatialJet(electrons));
    for (std::size_t element = 0; element < matrix_elements; ++element)
    {
      const std::size_t index = channel * matrix_elements + element;
      auto& jet = matrix[element];
      jet.value = orbitals.value[index];
      for (std::size_t lane = 0; lane < 3 * electrons; ++lane)
        jet.gradient[lane] = orbitals.gradient[lane * all_elements + index];
      for (std::size_t electron = 0; electron < electrons; ++electron)
        jet.laplacian[electron] = orbitals.laplacian[electron * all_elements + index];
    }
    amplitude = residual::sum(amplitude, residual::determinant(std::move(matrix), electrons), 1);
  }

  // exp(cusp) contributes g_cusp and Delta(cusp)+g_cusp.g_cusp to primal derivatives.
  const double cusp = accumulateCusp(values, workspace);
  residual::SpatialJet cusp_jet(electrons, 1);
  for (std::size_t lane = 0; lane < 3 * electrons; ++lane)
    cusp_jet.gradient[lane] = workspace.scalar_gradient_scratch_[lane];
  for (std::size_t electron = 0; electron < electrons; ++electron)
  {
    cusp_jet.laplacian[electron] = workspace.scalar_laplacian_scratch_[electron];
    for (std::size_t d = 0; d < 3; ++d)
      cusp_jet.laplacian[electron] += cusp_jet.gradient[3 * electron + d] * cusp_jet.gradient[3 * electron + d];
  }
  amplitude = residual::multiply(amplitude, {cusp, std::move(cusp_jet)});
  amplitude.jet.validate();
  workspace.observed_parameter_version_ = parameters_.version();
  return amplitude;
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
  buildTrunk(parameter_values, workspace);
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
  for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
    if (!is_finite_parameter_value(workspace.output_gradient_[lane]))
      throw std::runtime_error("PsiFormer spatial evaluator produced a non-finite gradient");
  for (std::size_t target = 0; target < workspace.laplacian_lanes_; ++target)
    if (!is_finite_parameter_value(workspace.output_lap_log_[target]))
      throw std::runtime_error("PsiFormer spatial evaluator produced a non-finite Laplacian");

  workspace.observed_parameter_version_ = parameters_.version();
  const double sign = determinant_result.amplitude.phase;
  return {sign,
          logabs,
          determinant::realValue(determinant_result.amplitude, cusp_value),
          workspace.mode_,
          {workspace.output_gradient_.data(), workspace.gradient_lanes_},
          {workspace.output_lap_log_.data(), workspace.laplacian_lanes_},
          {workspace.output_lap_ratio_.data(), workspace.laplacian_lanes_},
          workspace.observed_parameter_version_,
          workspace.mode_ == DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT ? workspace.active_electron_ :
              workspace.target_begin_, workspace.target_count_, workspace.evaluation_generation_};
}

} // namespace pf

#endif // QMCPLUSPLUS_PSIFORMER_SPATIAL_EXECUTOR_H
