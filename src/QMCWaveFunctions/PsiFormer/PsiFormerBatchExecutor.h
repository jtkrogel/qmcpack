//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerBatchExecutor.h
 * @brief Bounded tiled multi-configuration PsiFormer execution.
 *
 * Logical positions and results are configuration-major and grow with the requested
 * batch.  Expensive geometry, activation, orbital, and determinant scratch grows only
 * to an explicitly bounded tile capacity.  Value and spatial modes execute shared-weight
 * dense kernels over all configuration/electron rows in a tile.  Spatial attention,
 * orbital envelopes, stable determinant reduction, and cusp terms remain
 * configuration-local while their learned projections share the tile kernel seam.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_SPATIAL_EXECUTOR_H
#error "Include PsiFormerSpatialExecutor.h before PsiFormerBatchExecutor.h"
#endif

#ifndef QMCPLUSPLUS_PSIFORMER_BATCH_EXECUTOR_H
#define QMCPLUSPLUS_PSIFORMER_BATCH_EXECUTOR_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerBatchKernels.h"

#include <algorithm>
#include <array>
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

/// Select the independently shaped output family needed by a batch call.
enum class DirectBatchMode
{
  VALUE_ONLY,
  FULL_VGL,
  ACTIVE_ELECTRON_GRADIENT
};

/// Select dense complete configurations or sparse reference-plus-replacement VALUE input.
enum class DirectBatchValueInput
{
  DENSE_CONFIGURATIONS,
  SPARSE_REPLACEMENTS
};

/// Deterministic structural counters for the most recent completed evaluation.
using DirectBatchExecutionStatistics = qmcplusplus::psiformer::batch::ExecutionStatistics;

/// Non-owning sign/log/value outputs in configuration-major order.
struct DirectBatchValueResultView
{
  std::size_t size = 0;
  const double* sign = nullptr;
  const double* logabs = nullptr;
  const double* value = nullptr;
  const std::size_t* parameter_version = nullptr;
};

/** Non-owning spatial outputs in configuration-major order.
 *
 * ``gradient_stride`` is ``3*Ne`` for FULL_VGL and three for the active path.
 * Laplacian pointers are null for the active-gradient path.
 */
struct DirectBatchSpatialResultView : DirectBatchValueResultView
{
  DirectSpatialMode mode = DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT;
  std::size_t gradient_stride = 0;
  std::size_t laplacian_stride = 0;
  const double* gradient = nullptr;
  const double* lap_log = nullptr;
  const double* lap_ratio = nullptr;
};

/** Own logical batch storage and a grow-only bounded execution arena. */
class DirectBatchWorkspace
{
public:
  static constexpr std::size_t default_tile_capacity = 4;

  DirectBatchWorkspace(const DirectValueExecutor& value_executor,
                       const DirectSpatialExecutor& spatial_executor)
      : value_executor_(&value_executor),
        spatial_executor_(&spatial_executor),
        electron_count_(value_executor.layout()->electronCount()),
        tile_capacity_(default_tile_capacity)
  {
    if (value_executor.layout().get() != spatial_executor.layout().get() ||
        spatial_executor.valueExecutorIdentity() != &value_executor)
      throw std::invalid_argument(
          "PsiFormer batch executors do not share one parameter layout");
  }

  DirectBatchWorkspace(const DirectBatchWorkspace&) = delete;
  DirectBatchWorkspace& operator=(const DirectBatchWorkspace&) = delete;
  DirectBatchWorkspace(DirectBatchWorkspace&&) = default;
  DirectBatchWorkspace& operator=(DirectBatchWorkspace&&) = default;

  /** Select a nonzero execution tile capacity and prepare the current request.
   * Scratch remains grow-only, while a smaller selection takes effect immediately.
   */
  void prepareTileCapacity(std::size_t capacity)
  {
    if (capacity == 0)
      throw std::invalid_argument("PsiFormer batch tile capacity must be positive");
    // A policy capacity larger than the current logical batch is harmless.  Only
    // the effective occupancy participates in BLAS dimensions or allocation.
    if (active_size_ != 0)
    {
      const std::size_t effective_capacity = std::min(active_size_, capacity);
      validateScratchExtents(active_mode_, effective_capacity);
      if (active_value_input_ == DirectBatchValueInput::SPARSE_REPLACEMENTS)
        (void)sparseTilePositionElements(effective_capacity);
      prepareScratch(active_mode_, effective_capacity);
      if (active_value_input_ == DirectBatchValueInput::SPARSE_REPLACEMENTS)
        growVector(sparse_tile_positions_, sparseTilePositionElements(effective_capacity));
    }
    tile_capacity_ = capacity;
  }

  /// Grow logical storage, prepare bounded scratch, and begin a packing transaction.
  void resize(DirectBatchMode mode, std::size_t size)
  {
    namespace batch = qmcplusplus::psiformer::batch;
    const std::size_t position_count = batch::checkedProduct(
        batch::checkedProduct(size, electron_count_,
                              "PsiFormer batch position extent overflowed"),
        3, "PsiFormer batch position extent overflowed");
    const std::size_t gradient_stride = mode == DirectBatchMode::FULL_VGL
        ? batch::checkedProduct(electron_count_, 3,
                                "PsiFormer batch gradient stride overflowed")
        : 3;
    const std::size_t gradient_count = mode == DirectBatchMode::VALUE_ONLY
        ? 0
        : batch::checkedProduct(size, gradient_stride,
                                "PsiFormer batch gradient extent overflowed");
    const std::size_t laplacian_count = mode == DirectBatchMode::FULL_VGL
        ? batch::checkedProduct(size, electron_count_,
                                "PsiFormer batch Laplacian extent overflowed")
        : 0;
    const std::size_t effective_capacity = std::min(size, tile_capacity_);

    // Reject impossible packed products and BLAS dimensions before changing
    // either logical storage or a retained scratch family.
    validateScratchExtents(mode, effective_capacity);

    growVector(electron_positions_, position_count);
    growVector(position_ready_, position_count);
    reserveOutputs(size, gradient_count, laplacian_count);
    prepareScratch(mode, effective_capacity);

    active_mode_ = mode;
    active_value_input_ = DirectBatchValueInput::DENSE_CONFIGURATIONS;
    active_size_ = size;
    active_reference_count_ = 0;
    active_replacement_count_ = 0;
    active_dense_coordinate_bytes_avoided_ = 0;
    mode_capacity_[modeIndex(mode)] = std::max(mode_capacity_[modeIndex(mode)], size);
    std::fill_n(position_ready_.begin(), position_count, static_cast<unsigned char>(0));
  }

  /** Grow sparse VALUE input and begin a reference-plus-replacement transaction.
   *
   * Results retain the ordinary value view and are ordered as all ``R`` reference
   * configurations followed by all ``Q`` replacement configurations.
   */
  void resizeSparseValues(std::size_t reference_count,
                          std::size_t replacement_count)
  {
    namespace batch = qmcplusplus::psiformer::batch;
    if (reference_count == 0 && replacement_count != 0)
      throw std::invalid_argument(
          "PsiFormer sparse batch replacements require a reference configuration");

    const std::size_t size = batch::checkedSum(
        reference_count, replacement_count,
        "PsiFormer sparse batch configuration extent overflowed");
    const std::size_t reference_position_count = batch::checkedProduct(
        batch::checkedProduct(reference_count, electron_count_,
                              "PsiFormer sparse reference extent overflowed"),
        3, "PsiFormer sparse reference extent overflowed");
    const std::size_t replacement_position_count = batch::checkedProduct(
        replacement_count, 3,
        "PsiFormer sparse replacement extent overflowed");
    const std::size_t effective_capacity = std::min(size, tile_capacity_);
    const std::size_t sparse_tile_position_count =
        sparseTilePositionElements(effective_capacity);
    const std::size_t avoided_coordinate_scalars = electron_count_ == 0
        ? 0
        : batch::checkedProduct(
              batch::checkedProduct(replacement_count, electron_count_ - 1,
                                    "PsiFormer avoided coordinate extent overflowed"),
              3, "PsiFormer avoided coordinate extent overflowed");
    const std::size_t avoided_coordinate_bytes = batch::checkedProduct(
        avoided_coordinate_scalars, sizeof(double),
        "PsiFormer avoided coordinate byte extent overflowed");

    // Preflight every packed extent and BLAS dimension before changing the
    // current ready transaction or any retained high-water allocation.
    validateScratchExtents(DirectBatchMode::VALUE_ONLY, effective_capacity);

    growVector(reference_positions_, reference_position_count);
    growVector(reference_ready_, reference_position_count);
    growVector(replacement_references_, replacement_count);
    growVector(replacement_electrons_, replacement_count);
    growVector(replacement_positions_, replacement_position_count);
    growVector(replacement_ready_, replacement_count);
    reserveOutputs(size, 0, 0);
    prepareScratch(DirectBatchMode::VALUE_ONLY, effective_capacity);
    growVector(sparse_tile_positions_, sparse_tile_position_count);

    active_mode_ = DirectBatchMode::VALUE_ONLY;
    active_value_input_ = DirectBatchValueInput::SPARSE_REPLACEMENTS;
    active_size_ = size;
    active_reference_count_ = reference_count;
    active_replacement_count_ = replacement_count;
    active_dense_coordinate_bytes_avoided_ = avoided_coordinate_bytes;
    mode_capacity_[modeIndex(DirectBatchMode::VALUE_ONLY)] =
        std::max(mode_capacity_[modeIndex(DirectBatchMode::VALUE_ONLY)], size);
    std::fill_n(reference_ready_.begin(), reference_position_count,
                static_cast<unsigned char>(0));
    std::fill_n(replacement_ready_.begin(), replacement_count,
                static_cast<unsigned char>(0));
  }

  /// Set one finite Cartesian coordinate in the active logical request.
  void setPosition(std::size_t configuration,
                   std::size_t electron,
                   std::size_t dimension,
                   double value)
  {
    if (active_value_input_ != DirectBatchValueInput::DENSE_CONFIGURATIONS)
      throw std::logic_error(
          "PsiFormer dense coordinate setter cannot modify sparse batch input");
    if (configuration >= active_size_)
      throw std::out_of_range("PsiFormer batch configuration index is out of range");
    if (electron >= electron_count_ || dimension >= 3)
      throw std::out_of_range("PsiFormer batch coordinate index is out of range");
    if (!qmcplusplus::psiformer::determinant::isFiniteReal(value))
      throw std::invalid_argument("PsiFormer batch coordinate is non-finite");
    const std::size_t offset = (configuration * electron_count_ + electron) * 3 + dimension;
    electron_positions_[offset] = value;
    position_ready_[offset] = 1;
  }

  /// Copy one complete configuration into contiguous logical input storage.
  void setPositions(std::size_t configuration, GeometryPositionView positions)
  {
    if (active_value_input_ != DirectBatchValueInput::DENSE_CONFIGURATIONS)
      throw std::logic_error(
          "PsiFormer dense configuration setter cannot modify sparse batch input");
    if (configuration >= active_size_)
      throw std::out_of_range("PsiFormer batch configuration index is out of range");
    if (positions.size() != electron_count_)
      throw std::invalid_argument("PsiFormer batch received the wrong electron count");

    // Validate the complete view before mutating an already packed configuration.
    // This keeps a failed bulk replacement from leaving a ready mixed transaction.
    for (std::size_t electron = 0; electron < electron_count_; ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        if (!qmcplusplus::psiformer::determinant::isFiniteReal(
                positions(electron, dimension)))
          throw std::invalid_argument("PsiFormer batch coordinate is non-finite");

    for (std::size_t electron = 0; electron < electron_count_; ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        setPosition(configuration, electron, dimension, positions(electron, dimension));
  }

  /// Copy one complete finite reference configuration into sparse logical storage.
  void setReferenceConfiguration(std::size_t reference,
                                 GeometryPositionView positions)
  {
    if (active_value_input_ != DirectBatchValueInput::SPARSE_REPLACEMENTS)
      throw std::logic_error(
          "PsiFormer sparse reference setter requires sparse VALUE input");
    if (reference >= active_reference_count_)
      throw std::out_of_range("PsiFormer sparse reference index is out of range");
    if (positions.size() != electron_count_)
      throw std::invalid_argument(
          "PsiFormer sparse reference has the wrong electron count");

    for (std::size_t electron = 0; electron < electron_count_; ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        if (!qmcplusplus::psiformer::determinant::isFiniteReal(
                positions(electron, dimension)))
          throw std::invalid_argument(
              "PsiFormer sparse reference coordinate is non-finite");

    double* target = reference_positions_.data() + reference * electron_count_ * 3;
    for (std::size_t electron = 0; electron < electron_count_; ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        target[electron * 3 + dimension] = positions(electron, dimension);
    std::fill_n(reference_ready_.begin() + reference * electron_count_ * 3,
                electron_count_ * 3, static_cast<unsigned char>(1));
  }

  /** Set one finite reference coordinate without assuming ParticleSet storage layout.
   * This permits adapters to convert scalar precision coordinate-by-coordinate
   * without materializing another complete reference array.
   */
  void setReferencePosition(std::size_t reference,
                            std::size_t electron,
                            std::size_t dimension,
                            double value)
  {
    if (active_value_input_ != DirectBatchValueInput::SPARSE_REPLACEMENTS)
      throw std::logic_error(
          "PsiFormer sparse reference setter requires sparse VALUE input");
    if (reference >= active_reference_count_)
      throw std::out_of_range("PsiFormer sparse reference index is out of range");
    if (electron >= electron_count_ || dimension >= 3)
      throw std::out_of_range(
          "PsiFormer sparse reference coordinate index is out of range");
    if (!qmcplusplus::psiformer::determinant::isFiniteReal(value))
      throw std::invalid_argument(
          "PsiFormer sparse reference coordinate is non-finite");
    const std::size_t offset =
        (reference * electron_count_ + electron) * 3 + dimension;
    reference_positions_[offset] = value;
    reference_ready_[offset] = 1;
  }

  /// Store one finite absolute replacement tuple in caller-provided order.
  void setVirtualReplacement(std::size_t replacement,
                             std::size_t reference,
                             std::size_t electron,
                             const GeometryPosition& position)
  {
    if (active_value_input_ != DirectBatchValueInput::SPARSE_REPLACEMENTS)
      throw std::logic_error(
          "PsiFormer sparse replacement setter requires sparse VALUE input");
    if (replacement >= active_replacement_count_)
      throw std::out_of_range("PsiFormer sparse replacement index is out of range");
    if (reference >= active_reference_count_)
      throw std::out_of_range("PsiFormer sparse replacement reference is out of range");
    if (electron >= electron_count_)
      throw std::out_of_range("PsiFormer sparse replacement electron is out of range");
    for (double coordinate : position)
      if (!qmcplusplus::psiformer::determinant::isFiniteReal(coordinate))
        throw std::invalid_argument(
            "PsiFormer sparse replacement coordinate is non-finite");

    replacement_references_[replacement] = reference;
    replacement_electrons_[replacement] = electron;
    std::copy(position.begin(), position.end(),
              replacement_positions_.begin() + replacement * 3);
    replacement_ready_[replacement] = 1;
  }

  std::size_t size() const noexcept { return active_size_; }
  std::size_t electronCount() const noexcept { return electron_count_; }
  std::size_t tileCapacity() const noexcept { return tile_capacity_; }
  DirectBatchValueInput valueInput() const noexcept { return active_value_input_; }
  std::size_t referenceCount() const noexcept { return active_reference_count_; }
  std::size_t replacementCount() const noexcept { return active_replacement_count_; }

  /// Return the largest allocated tile high-water mark across retained scratch.
  std::size_t allocatedTileCapacity() const noexcept
  {
    return std::max({value_tile_capacity_, full_workspaces_.size(), active_workspaces_.size()});
  }

  /// Return the logical high-water configuration count requested for one mode.
  std::size_t capacity(DirectBatchMode mode) const noexcept
  { return mode_capacity_[modeIndex(mode)]; }

  const DirectBatchExecutionStatistics& executionStatistics() const noexcept
  { return statistics_; }

  /** Hash backing addresses and capacities for warmed-call allocation tests. */
  std::size_t storageFingerprint(DirectBatchMode mode) const noexcept
  {
    std::size_t hash = 1469598103934665603ULL;
    auto mix_value = [&hash](std::uintptr_t value) {
      hash ^= value;
      hash *= 1099511628211ULL;
    };
    auto mix_vector = [&mix_value](const auto& values) {
      mix_value(reinterpret_cast<std::uintptr_t>(values.data()));
      mix_value(values.capacity());
    };

    mix_vector(electron_positions_);
    mix_vector(position_ready_);
    mix_vector(reference_positions_);
    mix_vector(reference_ready_);
    mix_vector(replacement_references_);
    mix_vector(replacement_electrons_);
    mix_vector(replacement_positions_);
    mix_vector(replacement_ready_);
    mix_vector(sign_);
    mix_vector(logabs_);
    mix_vector(value_);
    mix_vector(parameter_version_);
    mix_vector(pending_sign_);
    mix_vector(pending_logabs_);
    mix_vector(pending_value_);
    mix_vector(pending_parameter_version_);
    mix_vector(gradient_);
    mix_vector(pending_gradient_);
    mix_vector(lap_log_);
    mix_vector(lap_ratio_);
    mix_vector(pending_lap_log_);
    mix_vector(pending_lap_ratio_);

    switch (mode)
    {
    case DirectBatchMode::VALUE_ONLY:
      mix_vector(sparse_tile_positions_);
      mix_vector(value_geometries_);
      mix_vector(raw_features_);
      mix_vector(features_a_);
      mix_vector(features_b_);
      mix_vector(query_);
      mix_vector(key_);
      mix_vector(projected_value_);
      mix_vector(attention_);
      mix_vector(attended_);
      mix_vector(hidden_);
      mix_vector(spin_features_);
      mix_vector(backflow_values_);
      mix_vector(orbital_matrices_);
      for (const auto& geometry : value_geometries_)
        mix_value(geometry.storageFingerprint());
      for (const auto& determinant : determinant_workspaces_)
      {
        mix_value(reinterpret_cast<std::uintptr_t>(determinant.get()));
        mix_value(determinant->storageFingerprint());
      }
      break;
    case DirectBatchMode::FULL_VGL:
      mix_vector(spatial_dense_source_);
      mix_vector(spatial_dense_target_);
      for (const auto& spatial : full_workspaces_)
      {
        mix_value(reinterpret_cast<std::uintptr_t>(spatial.get()));
        mix_value(spatial->storageFingerprint());
      }
      break;
    case DirectBatchMode::ACTIVE_ELECTRON_GRADIENT:
      mix_vector(spatial_dense_source_);
      mix_vector(spatial_dense_target_);
      for (const auto& spatial : active_workspaces_)
      {
        mix_value(reinterpret_cast<std::uintptr_t>(spatial.get()));
        mix_value(spatial->storageFingerprint());
      }
      break;
    }
    return hash;
  }

  /// Return bytes retained by logical inputs, outputs, readiness, and staging arrays.
  std::size_t logicalStorageBytes() const noexcept
  {
    std::size_t bytes = 0;
    bytes += electron_positions_.capacity() * sizeof(double);
    bytes += position_ready_.capacity() * sizeof(unsigned char);
    bytes += sparseInputStorageBytes();
    bytes += sign_.capacity() * sizeof(double);
    bytes += logabs_.capacity() * sizeof(double);
    bytes += value_.capacity() * sizeof(double);
    bytes += parameter_version_.capacity() * sizeof(std::size_t);
    bytes += pending_sign_.capacity() * sizeof(double);
    bytes += pending_logabs_.capacity() * sizeof(double);
    bytes += pending_value_.capacity() * sizeof(double);
    bytes += pending_parameter_version_.capacity() * sizeof(std::size_t);
    bytes += gradient_.capacity() * sizeof(double);
    bytes += pending_gradient_.capacity() * sizeof(double);
    bytes += lap_log_.capacity() * sizeof(double);
    bytes += lap_ratio_.capacity() * sizeof(double);
    bytes += pending_lap_log_.capacity() * sizeof(double);
    bytes += pending_lap_ratio_.capacity() * sizeof(double);
    return bytes;
  }

  /// Return retained sparse logical input bytes, excluding results and tile scratch.
  std::size_t sparseInputStorageBytes() const noexcept
  {
    std::size_t bytes = 0;
    bytes += reference_positions_.capacity() * sizeof(double);
    bytes += reference_ready_.capacity() * sizeof(unsigned char);
    bytes += replacement_references_.capacity() * sizeof(std::size_t);
    bytes += replacement_electrons_.capacity() * sizeof(std::size_t);
    bytes += replacement_positions_.capacity() * sizeof(double);
    bytes += replacement_ready_.capacity() * sizeof(unsigned char);
    return bytes;
  }

  /// Return retained sparse tile-position bytes, bounded by the tile capacity.
  std::size_t sparseTilePositionBytes() const noexcept
  { return sparse_tile_positions_.capacity() * sizeof(double); }

  /// Return expensive numeric scratch retained at the tile high-water mark.
  std::size_t tileScratchBytes() const noexcept
  {
    std::size_t scalar_capacity = 0;
    auto add = [&scalar_capacity](const std::vector<double>& buffer) {
      scalar_capacity += buffer.capacity();
    };
    add(raw_features_);
    add(features_a_);
    add(features_b_);
    add(query_);
    add(key_);
    add(projected_value_);
    add(attention_);
    add(attended_);
    add(hidden_);
    add(spin_features_);
    add(backflow_values_);
    add(orbital_matrices_);
    add(sparse_tile_positions_);
    add(spatial_dense_source_);
    add(spatial_dense_target_);
    std::size_t bytes = scalar_capacity * sizeof(double);
    for (const auto& geometry : value_geometries_)
      bytes += geometry.storageBytes();
    for (const auto& determinant : determinant_workspaces_)
      bytes += determinant->storageBytes();
    for (const auto& spatial : full_workspaces_)
      bytes += spatial->vectorStorageBytes();
    for (const auto& spatial : active_workspaces_)
      bytes += spatial->vectorStorageBytes();
    return bytes;
  }

  /// Return the shared packed source/target storage used by spatial tile kernels.
  std::size_t spatialTileKernelBytes() const noexcept
  {
    return (spatial_dense_source_.capacity() + spatial_dense_target_.capacity()) *
        sizeof(double);
  }

  std::size_t vectorStorageBytes() const noexcept
  { return logicalStorageBytes() + tileScratchBytes(); }

private:
  friend class DirectBatchExecutor;

  static std::size_t modeIndex(DirectBatchMode mode) noexcept
  {
    switch (mode)
    {
    case DirectBatchMode::VALUE_ONLY:
      return 0;
    case DirectBatchMode::FULL_VGL:
      return 1;
    case DirectBatchMode::ACTIVE_ELECTRON_GRADIENT:
      return 2;
    }
    return 0;
  }

  template<class T>
  static void growVector(std::vector<T>& buffer, std::size_t required_size)
  {
    if (buffer.size() < required_size)
      buffer.resize(required_size);
  }

  template<class Workspace, class Factory>
  static void growWorkspaces(std::vector<std::unique_ptr<Workspace>>& workspaces,
                             std::size_t capacity,
                             Factory&& factory)
  {
    workspaces.reserve(std::max(workspaces.capacity(), capacity));
    while (workspaces.size() < capacity)
      workspaces.emplace_back(factory());
  }

  void reserveOutputs(std::size_t size,
                      std::size_t gradient_count,
                      std::size_t laplacian_count)
  {
    growVector(sign_, size);
    growVector(logabs_, size);
    growVector(value_, size);
    growVector(parameter_version_, size);
    growVector(pending_sign_, size);
    growVector(pending_logabs_, size);
    growVector(pending_value_, size);
    growVector(pending_parameter_version_, size);
    growVector(gradient_, gradient_count);
    growVector(pending_gradient_, gradient_count);
    growVector(lap_log_, laplacian_count);
    growVector(lap_ratio_, laplacian_count);
    growVector(pending_lap_log_, laplacian_count);
    growVector(pending_lap_ratio_, laplacian_count);
  }

  void prepareScratch(DirectBatchMode mode, std::size_t capacity)
  {
    switch (mode)
    {
    case DirectBatchMode::VALUE_ONLY:
      growValueScratch(capacity);
      break;
    case DirectBatchMode::FULL_VGL:
    {
      const std::size_t packed_elements =
          spatialDenseScratchElements(capacity, DirectSpatialMode::FULL_VGL);
      growWorkspaces(full_workspaces_, capacity, [this]() {
        return spatial_executor_->makeWorkspace(DirectSpatialMode::FULL_VGL);
      });
      growVector(spatial_dense_source_, packed_elements);
      growVector(spatial_dense_target_, packed_elements);
      break;
    }
    case DirectBatchMode::ACTIVE_ELECTRON_GRADIENT:
    {
      const std::size_t packed_elements = spatialDenseScratchElements(
          capacity, DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT);
      growWorkspaces(active_workspaces_, capacity, [this]() {
        return spatial_executor_->makeWorkspace(DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT);
      });
      growVector(spatial_dense_source_, packed_elements);
      growVector(spatial_dense_target_, packed_elements);
      break;
    }
    }
  }

  /// Validate scratch products and BLAS ABI dimensions without changing state.
  void validateScratchExtents(DirectBatchMode mode, std::size_t capacity) const
  {
    switch (mode)
    {
    case DirectBatchMode::VALUE_ONLY:
      (void)valueScratchExtents(capacity);
      break;
    case DirectBatchMode::FULL_VGL:
      (void)spatialDenseScratchElements(capacity, DirectSpatialMode::FULL_VGL);
      break;
    case DirectBatchMode::ACTIVE_ELECTRON_GRADIENT:
      (void)spatialDenseScratchElements(
          capacity, DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT);
      break;
    }
  }

  /// Return the bounded sparse position-arena extent after checked arithmetic.
  std::size_t sparseTilePositionElements(std::size_t capacity) const
  {
    namespace batch = qmcplusplus::psiformer::batch;
    return batch::checkedProduct(
        batch::checkedProduct(capacity, electron_count_,
                              "PsiFormer sparse tile position extent overflowed"),
        3, "PsiFormer sparse tile position extent overflowed");
  }

  /// Validate and return packed spatial elements before any slot or vector growth.
  std::size_t spatialDenseScratchElements(std::size_t capacity,
                                          DirectSpatialMode mode) const
  {
    const auto& layout = *value_executor_->layout();
    namespace batch = qmcplusplus::psiformer::batch;
    const std::size_t gradient_lanes =
        mode == DirectSpatialMode::FULL_VGL
        ? batch::checkedProduct(electron_count_, 3,
                                "PsiFormer spatial tile gradient extent overflowed")
        : 3;
    const std::size_t laplacian_lanes =
        mode == DirectSpatialMode::FULL_VGL ? electron_count_ : 0;
    const std::size_t planes = 1 + gradient_lanes + laplacian_lanes;
    const std::size_t orbital_channels = batch::checkedProduct(
        layout.determinantCount(), electron_count_,
        "PsiFormer spatial tile orbital extent overflowed");
    const std::size_t maximum_width =
        std::max({layout.inputWidth(), layout.featureWidth(), orbital_channels});
    const std::size_t packed_rows = batch::checkedProduct(
        batch::checkedProduct(capacity, planes,
                              "PsiFormer spatial tile plane extent overflowed"),
        electron_count_, "PsiFormer spatial tile row extent overflowed");
    (void)qmcplusplus::psiformer::dense::checkedBlasDimension(
        packed_rows,
        "PsiFormer spatial tile rows exceed the BLAS integer ABI");
    (void)qmcplusplus::psiformer::dense::checkedBlasDimension(
        maximum_width,
        "PsiFormer spatial tile width exceeds the BLAS integer ABI");
    const std::size_t packed_elements = batch::checkedProduct(
        packed_rows, maximum_width,
        "PsiFormer spatial tile dense scratch extent overflowed");
    return packed_elements;
  }

  struct ValueScratchExtents
  {
    std::size_t raw;
    std::size_t feature;
    std::size_t attention;
    std::size_t backflow;
    std::size_t orbital;
  };

  /// Validate and return every value-only scratch extent without allocating.
  ValueScratchExtents valueScratchExtents(std::size_t capacity) const
  {
    const auto& layout = *value_executor_->layout();
    namespace batch = qmcplusplus::psiformer::batch;
    const std::size_t electron_rows = batch::checkedProduct(
        capacity, electron_count_, "PsiFormer value tile electron extent overflowed");
    (void)qmcplusplus::psiformer::dense::checkedBlasDimension(
        electron_rows,
        "PsiFormer value tile rows exceed the BLAS integer ABI");
    const std::size_t feature_elements = batch::checkedProduct(
        electron_rows, layout.featureWidth(), "PsiFormer value tile feature extent overflowed");
    const std::size_t raw_elements = batch::checkedProduct(
        electron_rows, layout.inputWidth(), "PsiFormer value tile embedding extent overflowed");
    const std::size_t attention_per_configuration = batch::checkedProduct(
        batch::checkedProduct(layout.headCount(), electron_count_,
                              "PsiFormer value tile attention extent overflowed"),
        electron_count_, "PsiFormer value tile attention extent overflowed");
    const std::size_t attention_elements = batch::checkedProduct(
        capacity, attention_per_configuration,
        "PsiFormer value tile attention extent overflowed");
    const std::size_t orbital_channels = batch::checkedProduct(
        layout.determinantCount(), electron_count_,
        "PsiFormer value tile orbital extent overflowed");
    (void)qmcplusplus::psiformer::dense::checkedBlasDimension(
        std::max({layout.inputWidth(), layout.featureWidth(), orbital_channels}),
        "PsiFormer value tile width exceeds the BLAS integer ABI");
    const std::size_t backflow_elements = batch::checkedProduct(
        electron_rows, orbital_channels,
        "PsiFormer value tile backflow extent overflowed");
    // Both packed backflow outputs and final matrices contain
    // T * Ne * (D * Ne) scalar entries; only their index ordering differs.
    const std::size_t orbital_elements = backflow_elements;

    return {raw_elements, feature_elements, attention_elements,
            backflow_elements, orbital_elements};
  }

  /// Grow every value-only tile buffer after all element-count products are checked.
  void growValueScratch(std::size_t capacity)
  {
    if (capacity <= value_tile_capacity_)
      return;
    const auto& layout = *value_executor_->layout();
    const ValueScratchExtents extents = valueScratchExtents(capacity);

    value_geometries_.reserve(capacity);
    while (value_geometries_.size() < capacity)
      value_geometries_.emplace_back(
          electron_count_, value_executor_->nuclearPositions(), value_executor_->boundary());
    determinant_workspaces_.reserve(capacity);
    while (determinant_workspaces_.size() < capacity)
      determinant_workspaces_.push_back(
          std::make_unique<qmcplusplus::psiformer::determinant::RealOpenDeterminantWorkspace>(
              layout.determinantCount(), electron_count_));

    growVector(raw_features_, extents.raw);
    growVector(features_a_, extents.feature);
    growVector(features_b_, extents.feature);
    growVector(query_, extents.feature);
    growVector(key_, extents.feature);
    growVector(projected_value_, extents.feature);
    growVector(attention_, extents.attention);
    growVector(attended_, extents.feature);
    growVector(hidden_, extents.feature);
    growVector(spin_features_, extents.feature);
    growVector(backflow_values_, extents.backflow);
    growVector(orbital_matrices_, extents.orbital);
    value_tile_capacity_ = capacity;
  }

  GeometryPositionView positionView(std::size_t configuration) const noexcept
  {
    return GeometryPositionView::interleaved(
        electron_positions_.data() + configuration * electron_count_ * 3, electron_count_);
  }

  void requireCompletePositions() const
  {
    const std::size_t required = active_size_ * electron_count_ * 3;
    for (std::size_t coordinate = 0; coordinate < required; ++coordinate)
      if (position_ready_[coordinate] == 0)
      {
        const std::size_t configuration = coordinate / (electron_count_ * 3);
        throw std::logic_error(
            "PsiFormer batch configuration " + std::to_string(configuration) +
            " has incomplete coordinates");
      }
  }

  /// Validate every sparse reference and replacement before tile scratch changes.
  void requireCompleteSparseInput() const
  {
    for (std::size_t reference = 0; reference < active_reference_count_; ++reference)
    {
      const double* positions = reference_positions_.data() +
          reference * electron_count_ * 3;
      for (std::size_t coordinate = 0; coordinate < electron_count_ * 3;
           ++coordinate)
      {
        if (reference_ready_[reference * electron_count_ * 3 + coordinate] == 0)
          throw std::logic_error(
              "PsiFormer sparse reference " + std::to_string(reference) +
              " has incomplete coordinates");
        if (!qmcplusplus::psiformer::determinant::isFiniteReal(
                positions[coordinate]))
          throw std::invalid_argument(
              "PsiFormer sparse reference coordinate is non-finite");
      }
    }

    for (std::size_t replacement = 0;
         replacement < active_replacement_count_; ++replacement)
    {
      if (replacement_ready_[replacement] == 0)
        throw std::logic_error(
            "PsiFormer sparse replacement " + std::to_string(replacement) +
            " is incomplete");
      if (replacement_references_[replacement] >= active_reference_count_)
        throw std::out_of_range(
            "PsiFormer sparse replacement reference is out of range");
      if (replacement_electrons_[replacement] >= electron_count_)
        throw std::out_of_range(
            "PsiFormer sparse replacement electron is out of range");
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        if (!qmcplusplus::psiformer::determinant::isFiniteReal(
                replacement_positions_[replacement * 3 + dimension]))
          throw std::invalid_argument(
              "PsiFormer sparse replacement coordinate is non-finite");
    }
  }

  /// Gather one sparse tile in result order: references, then replacements.
  void materializeSparseTile(std::size_t tile_begin, std::size_t tile_size)
  {
    const std::size_t position_stride = electron_count_ * 3;
    for (std::size_t local = 0; local < tile_size; ++local)
    {
      const std::size_t configuration = tile_begin + local;
      const bool is_reference = configuration < active_reference_count_;
      const std::size_t replacement = is_reference
          ? 0
          : configuration - active_reference_count_;
      const std::size_t reference = is_reference
          ? configuration
          : replacement_references_[replacement];
      double* target = sparse_tile_positions_.data() + local * position_stride;
      std::copy_n(reference_positions_.data() + reference * position_stride,
                  position_stride, target);
      if (!is_reference)
      {
        const std::size_t electron = replacement_electrons_[replacement];
        std::copy_n(replacement_positions_.data() + replacement * 3, 3,
                    target + electron * 3);
      }
    }
  }

  const DirectValueExecutor* value_executor_;
  const DirectSpatialExecutor* spatial_executor_;
  const std::size_t electron_count_;
  DirectBatchMode active_mode_ = DirectBatchMode::VALUE_ONLY;
  DirectBatchValueInput active_value_input_ =
      DirectBatchValueInput::DENSE_CONFIGURATIONS;
  std::size_t active_size_ = 0;
  std::size_t active_reference_count_ = 0;
  std::size_t active_replacement_count_ = 0;
  std::size_t active_dense_coordinate_bytes_avoided_ = 0;
  std::size_t tile_capacity_ = default_tile_capacity;
  std::size_t value_tile_capacity_ = 0;
  std::array<std::size_t, 3> mode_capacity_{};
  DirectBatchExecutionStatistics statistics_{};

  std::vector<double> electron_positions_;
  std::vector<unsigned char> position_ready_;
  std::vector<double> reference_positions_;
  std::vector<unsigned char> reference_ready_;
  std::vector<std::size_t> replacement_references_;
  std::vector<std::size_t> replacement_electrons_;
  std::vector<double> replacement_positions_;
  std::vector<unsigned char> replacement_ready_;
  std::vector<double> sparse_tile_positions_;
  std::vector<PsiFormerGeometryCache> value_geometries_;
  std::vector<double> raw_features_;
  std::vector<double> features_a_;
  std::vector<double> features_b_;
  std::vector<double> query_;
  std::vector<double> key_;
  std::vector<double> projected_value_;
  std::vector<double> attention_;
  std::vector<double> attended_;
  std::vector<double> hidden_;
  std::vector<double> spin_features_;
  std::vector<double> backflow_values_;
  std::vector<double> orbital_matrices_;
  std::vector<std::unique_ptr<qmcplusplus::psiformer::determinant::RealOpenDeterminantWorkspace>>
      determinant_workspaces_;
  std::vector<std::unique_ptr<DirectSpatialWorkspace>> full_workspaces_;
  std::vector<std::unique_ptr<DirectSpatialWorkspace>> active_workspaces_;
  std::vector<double> spatial_dense_source_;
  std::vector<double> spatial_dense_target_;

  std::vector<double> sign_;
  std::vector<double> logabs_;
  std::vector<double> value_;
  std::vector<std::size_t> parameter_version_;
  std::vector<double> pending_sign_;
  std::vector<double> pending_logabs_;
  std::vector<double> pending_value_;
  std::vector<std::size_t> pending_parameter_version_;
  std::vector<double> gradient_;
  std::vector<double> pending_gradient_;
  std::vector<double> lap_log_;
  std::vector<double> lap_ratio_;
  std::vector<double> pending_lap_log_;
  std::vector<double> pending_lap_ratio_;
};

/** Execute configuration-major batches against one immutable model and plan. */
class DirectBatchExecutor
{
public:
  DirectBatchExecutor(const DirectValueExecutor& value_executor,
                      const DirectSpatialExecutor& spatial_executor)
      : value_executor_(value_executor), spatial_executor_(spatial_executor)
  {
    if (value_executor.layout().get() != spatial_executor.layout().get() ||
        spatial_executor.valueExecutorIdentity() != &value_executor)
      throw std::invalid_argument(
          "PsiFormer batch executors do not share one parameter layout");
  }

  std::unique_ptr<DirectBatchWorkspace> makeWorkspace() const
  { return std::make_unique<DirectBatchWorkspace>(value_executor_, spatial_executor_); }

  /** Evaluate a true tiled value batch without invoking the scalar value executor. */
  DirectBatchValueResultView evaluateValues(DirectBatchWorkspace& workspace) const
  {
    requireWorkspace(workspace);
    requireMode(workspace, DirectBatchMode::VALUE_ONLY);
    if (workspace.active_value_input_ ==
        DirectBatchValueInput::SPARSE_REPLACEMENTS)
      workspace.requireCompleteSparseInput();
    else
      workspace.requireCompletePositions();
    const auto& layout = *value_executor_.layout_;
    layout.validateParameterStore(value_executor_.parameters_);

    DirectBatchExecutionStatistics statistics;
    if (workspace.active_value_input_ ==
        DirectBatchValueInput::SPARSE_REPLACEMENTS)
    {
      statistics.reference_configurations = workspace.active_reference_count_;
      statistics.replacement_configurations = workspace.active_replacement_count_;
      statistics.reference_evaluations = workspace.active_reference_count_;
      statistics.dense_coordinate_bytes_avoided =
          workspace.active_dense_coordinate_bytes_avoided_;
    }
    const std::size_t parameter_version = value_executor_.parameters_.version();
    const double* parameters = value_executor_.parameters_.flat_values().data();
    const std::size_t tile_limit = workspace.tile_capacity_;
    for (std::size_t tile_begin = 0; tile_begin < workspace.active_size_;
         tile_begin += tile_limit)
    {
      const std::size_t tile_size = std::min(tile_limit, workspace.active_size_ - tile_begin);
      ++statistics.tiles_executed;
      statistics.max_tile_occupancy = std::max(statistics.max_tile_occupancy, tile_size);
      const double* tile_positions;
      if (workspace.active_value_input_ ==
          DirectBatchValueInput::SPARSE_REPLACEMENTS)
      {
        workspace.materializeSparseTile(tile_begin, tile_size);
        tile_positions = workspace.sparse_tile_positions_.data();
      }
      else
        tile_positions = workspace.electron_positions_.data() +
            tile_begin * workspace.electron_count_ * 3;
      evaluateValueTile(workspace, tile_begin, tile_size, tile_positions, parameters,
                        parameter_version, statistics);
    }

    if (value_executor_.parameters_.version() != parameter_version)
      throw std::runtime_error("PsiFormer parameters changed during batch evaluation");
    std::copy_n(workspace.pending_sign_.begin(), workspace.active_size_, workspace.sign_.begin());
    std::copy_n(workspace.pending_logabs_.begin(), workspace.active_size_, workspace.logabs_.begin());
    std::copy_n(workspace.pending_value_.begin(), workspace.active_size_, workspace.value_.begin());
    std::copy_n(workspace.pending_parameter_version_.begin(), workspace.active_size_,
                workspace.parameter_version_.begin());
    workspace.statistics_ = statistics;
    return valueView(workspace);
  }

  /// Evaluate full VGL with shared dense work across every tile jet plane.
  DirectBatchSpatialResultView evaluateFull(DirectBatchWorkspace& workspace) const
  {
    requireWorkspace(workspace);
    requireMode(workspace, DirectBatchMode::FULL_VGL);
    requireDenseInput(workspace);
    return evaluateSpatialBatch(workspace, DirectSpatialMode::FULL_VGL, nullptr);
  }

  /// Evaluate independently selected active-electron gradients with shared tile kernels.
  DirectBatchSpatialResultView evaluateActive(DirectBatchWorkspace& workspace,
                                               const std::size_t* active_electrons) const
  {
    requireWorkspace(workspace);
    requireMode(workspace, DirectBatchMode::ACTIVE_ELECTRON_GRADIENT);
    requireDenseInput(workspace);
    return evaluateSpatialBatch(
        workspace, DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT, active_electrons);
  }

private:
  static const double* tensor(const double* parameters,
                              const DirectParameterTensor& descriptor) noexcept
  { return parameters + descriptor.begin; }

  static void requireMode(const DirectBatchWorkspace& workspace, DirectBatchMode expected)
  {
    if (workspace.active_mode_ != expected)
      throw std::logic_error(
          "PsiFormer batch workspace mode does not match the requested evaluation");
  }

  static void requireDenseInput(const DirectBatchWorkspace& workspace)
  {
    if (workspace.active_value_input_ !=
        DirectBatchValueInput::DENSE_CONFIGURATIONS)
      throw std::logic_error(
          "PsiFormer spatial batch evaluation requires dense configuration input");
  }

  void requireWorkspace(const DirectBatchWorkspace& workspace) const
  {
    if (workspace.value_executor_ != &value_executor_ ||
        workspace.spatial_executor_ != &spatial_executor_)
      throw std::invalid_argument("PsiFormer batch workspace belongs to another executor");
  }

  static std::vector<std::unique_ptr<DirectSpatialWorkspace>>& spatialWorkspaces(
      DirectBatchWorkspace& workspace,
      DirectSpatialMode mode) noexcept
  {
    return mode == DirectSpatialMode::FULL_VGL ? workspace.full_workspaces_
                                                : workspace.active_workspaces_;
  }

  /** Pack [configuration,plane,row,feature] into the shared dense source matrix. */
  template<class Source>
  static std::size_t packSpatialTile(
      DirectBatchWorkspace& batch_workspace,
      std::vector<std::unique_ptr<DirectSpatialWorkspace>>& workspaces,
      std::size_t tile_size,
      std::size_t rows,
      std::size_t input_width,
      Source&& source)
  {
    if (tile_size == 0)
      return 0;
    const std::size_t plane_elements = rows * input_width;
    const std::size_t gradient_lanes = workspaces.front()->gradient_lanes_;
    const std::size_t laplacian_lanes = workspaces.front()->laplacian_lanes_;
    const std::size_t planes = 1 + gradient_lanes + laplacian_lanes;
    for (std::size_t local = 0; local < tile_size; ++local)
    {
      const DirectSpatialJetBuffer& jet = source(*workspaces[local]);
      if (jet.value.size() != plane_elements ||
          jet.gradient.size() != gradient_lanes * plane_elements ||
          jet.laplacian.size() != laplacian_lanes * plane_elements)
        throw std::logic_error("PsiFormer spatial batch jet shape is inconsistent");
      double* configuration = batch_workspace.spatial_dense_source_.data() +
          local * planes * plane_elements;
      std::copy_n(jet.value.data(), plane_elements, configuration);
      std::copy_n(jet.gradient.data(), gradient_lanes * plane_elements,
                  configuration + plane_elements);
      std::copy_n(jet.laplacian.data(), laplacian_lanes * plane_elements,
                  configuration + (1 + gradient_lanes) * plane_elements);
    }
    return tile_size * planes * rows;
  }

  /** Scatter one dense result back to the fixed per-configuration jet buffers. */
  template<class Target>
  static void unpackSpatialTile(
      DirectBatchWorkspace& batch_workspace,
      std::vector<std::unique_ptr<DirectSpatialWorkspace>>& workspaces,
      std::size_t tile_size,
      std::size_t rows,
      std::size_t output_width,
      const double* bias,
      Target&& target)
  {
    if (tile_size == 0)
      return;
    const std::size_t plane_elements = rows * output_width;
    const std::size_t gradient_lanes = workspaces.front()->gradient_lanes_;
    const std::size_t laplacian_lanes = workspaces.front()->laplacian_lanes_;
    const std::size_t planes = 1 + gradient_lanes + laplacian_lanes;
    for (std::size_t local = 0; local < tile_size; ++local)
    {
      DirectSpatialJetBuffer& jet = target(*workspaces[local]);
      if (jet.value.size() != plane_elements ||
          jet.gradient.size() != gradient_lanes * plane_elements ||
          jet.laplacian.size() != laplacian_lanes * plane_elements)
        throw std::logic_error("PsiFormer spatial batch target shape is inconsistent");
      const double* configuration = batch_workspace.spatial_dense_target_.data() +
          local * planes * plane_elements;
      std::copy_n(configuration, plane_elements, jet.value.data());
      std::copy_n(configuration + plane_elements,
                  gradient_lanes * plane_elements, jet.gradient.data());
      std::copy_n(configuration + (1 + gradient_lanes) * plane_elements,
                  laplacian_lanes * plane_elements, jet.laplacian.data());
      if (bias)
        for (std::size_t row = 0; row < rows; ++row)
          for (std::size_t output = 0; output < output_width; ++output)
            jet.value[row * output_width + output] += bias[output];
    }
  }

  /// Apply one shared-weight dense product to every tile configuration and jet plane.
  template<class Source, class Target>
  static void denseSpatialTile(
      DirectBatchWorkspace& batch_workspace,
      std::vector<std::unique_ptr<DirectSpatialWorkspace>>& workspaces,
      std::size_t tile_size,
      const double* weight,
      const double* bias,
      std::size_t rows,
      std::size_t input_width,
      std::size_t output_width,
      Source&& source,
      Target&& target,
      DirectBatchExecutionStatistics& statistics)
  {
    const std::size_t packed_rows = packSpatialTile(
        batch_workspace, workspaces, tile_size, rows, input_width, source);
    qmcplusplus::psiformer::batch::productReal(
        batch_workspace.spatial_dense_source_.data(), weight, nullptr,
        packed_rows, input_width, output_width,
        batch_workspace.spatial_dense_target_.data(), statistics);
    unpackSpatialTile(batch_workspace, workspaces, tile_size, rows, output_width,
                      bias, target);
  }

  /// Apply Q, K, and V weights to one packed tile jet without repacking its source.
  template<class Source, class Query, class Key, class Value>
  static void denseSpatialQkvTile(
      DirectBatchWorkspace& batch_workspace,
      std::vector<std::unique_ptr<DirectSpatialWorkspace>>& workspaces,
      std::size_t tile_size,
      const double* query_weight,
      const double* key_weight,
      const double* value_weight,
      std::size_t rows,
      std::size_t width,
      Source&& source,
      Query&& query,
      Key&& key,
      Value&& value,
      DirectBatchExecutionStatistics& statistics)
  {
    const std::size_t packed_rows = packSpatialTile(
        batch_workspace, workspaces, tile_size, rows, width, source);
    auto project = [&](const double* weight, auto&& target) {
      qmcplusplus::psiformer::batch::productReal(
          batch_workspace.spatial_dense_source_.data(), weight, nullptr,
          packed_rows, width, width, batch_workspace.spatial_dense_target_.data(),
          statistics);
      unpackSpatialTile(batch_workspace, workspaces, tile_size, rows, width,
                        nullptr, target);
    };
    project(query_weight, query);
    project(key_weight, key);
    project(value_weight, value);
  }

  static void addSpatialJet(DirectSpatialJetBuffer& target,
                            const DirectSpatialJetBuffer& source)
  {
    for (std::size_t element = 0; element < target.value.size(); ++element)
      target.value[element] += source.value[element];
    for (std::size_t element = 0; element < target.gradient.size(); ++element)
      target.gradient[element] += source.gradient[element];
    for (std::size_t element = 0; element < target.laplacian.size(); ++element)
      target.laplacian[element] += source.laplacian[element];
  }

  /// Build embedding and attention/MLP jets with tile-stacked shared projections.
  void buildSpatialNetworkTile(
      DirectBatchWorkspace& batch_workspace,
      std::vector<std::unique_ptr<DirectSpatialWorkspace>>& workspaces,
      std::size_t tile_begin,
      std::size_t tile_size,
      const std::size_t* active_electrons,
      const double* parameters,
      DirectBatchExecutionStatistics& statistics) const
  {
    const auto& layout = *spatial_executor_.layout_;
    const std::size_t electrons = layout.electronCount();
    const std::size_t width = layout.featureWidth();
    for (std::size_t local = 0; local < tile_size; ++local)
    {
      const std::size_t configuration = tile_begin + local;
      DirectSpatialWorkspace& spatial = *workspaces[local];
      spatial.setPositions(batch_workspace.positionView(configuration));
      spatial.active_electron_ = spatial.mode_ == DirectSpatialMode::FULL_VGL
          ? 0
          : active_electrons[configuration];
      try
      {
        spatial_executor_.refreshGeometry(spatial);
        spatial_executor_.buildEmbeddingFeatures(spatial);
      }
      catch (const std::exception& error)
      {
        throw std::runtime_error(
            "PsiFormer batch configuration " + std::to_string(configuration) +
            " embedding failed: " + error.what());
      }
    }

    denseSpatialTile(
        batch_workspace, workspaces, tile_size,
        tensor(parameters, layout.embedding_), nullptr, electrons,
        layout.inputWidth(), width,
        [](DirectSpatialWorkspace& spatial) -> DirectSpatialJetBuffer& {
          return spatial.raw_features_;
        },
        [](DirectSpatialWorkspace& spatial) -> DirectSpatialJetBuffer& {
          return spatial.features_a_;
        },
        statistics);

    for (const auto& layer : layout.layers_)
    {
      denseSpatialQkvTile(
          batch_workspace, workspaces, tile_size,
          tensor(parameters, layer.query), tensor(parameters, layer.key),
          tensor(parameters, layer.value), electrons, width,
          [](DirectSpatialWorkspace& spatial) -> DirectSpatialJetBuffer& {
            return spatial.features_a_;
          },
          [](DirectSpatialWorkspace& spatial) -> DirectSpatialJetBuffer& {
            return spatial.query_;
          },
          [](DirectSpatialWorkspace& spatial) -> DirectSpatialJetBuffer& {
            return spatial.key_;
          },
          [](DirectSpatialWorkspace& spatial) -> DirectSpatialJetBuffer& {
            return spatial.projected_value_;
          },
          statistics);

      for (std::size_t local = 0; local < tile_size; ++local)
      {
        const std::size_t configuration = tile_begin + local;
        try
        {
          spatial_executor_.buildAttentionWeights(*workspaces[local]);
          spatial_executor_.buildAttentionContext(*workspaces[local]);
        }
        catch (const std::exception& error)
        {
          throw std::runtime_error(
              "PsiFormer batch configuration " + std::to_string(configuration) +
              " attention failed: " + error.what());
        }
      }

      denseSpatialTile(
          batch_workspace, workspaces, tile_size,
          tensor(parameters, layer.projection), nullptr, electrons, width, width,
          [](DirectSpatialWorkspace& spatial) -> DirectSpatialJetBuffer& {
            return spatial.attended_;
          },
          [](DirectSpatialWorkspace& spatial) -> DirectSpatialJetBuffer& {
            return spatial.features_b_;
          },
          statistics);
      for (std::size_t local = 0; local < tile_size; ++local)
        addSpatialJet(workspaces[local]->features_b_,
                      workspaces[local]->features_a_);

      denseSpatialTile(
          batch_workspace, workspaces, tile_size,
          tensor(parameters, layer.mlp_weight_0),
          tensor(parameters, layer.mlp_bias_0), electrons, width, width,
          [](DirectSpatialWorkspace& spatial) -> DirectSpatialJetBuffer& {
            return spatial.features_b_;
          },
          [](DirectSpatialWorkspace& spatial) -> DirectSpatialJetBuffer& {
            return spatial.hidden_;
          },
          statistics);
      for (std::size_t local = 0; local < tile_size; ++local)
        DirectSpatialExecutor::tanhJetInPlace(
            workspaces[local]->hidden_, workspaces[local]->gradient_lanes_,
            workspaces[local]->laplacian_lanes_);

      denseSpatialTile(
          batch_workspace, workspaces, tile_size,
          tensor(parameters, layer.mlp_weight_1),
          tensor(parameters, layer.mlp_bias_1), electrons, width, width,
          [](DirectSpatialWorkspace& spatial) -> DirectSpatialJetBuffer& {
            return spatial.hidden_;
          },
          [](DirectSpatialWorkspace& spatial) -> DirectSpatialJetBuffer& {
            return spatial.attended_;
          },
          statistics);
      for (std::size_t local = 0; local < tile_size; ++local)
      {
        DirectSpatialExecutor::tanhJetInPlace(
            workspaces[local]->attended_, workspaces[local]->gradient_lanes_,
            workspaces[local]->laplacian_lanes_);
        addSpatialJet(workspaces[local]->features_b_,
                      workspaces[local]->attended_);
        std::swap(workspaces[local]->features_a_, workspaces[local]->features_b_);
      }
    }
  }

  /// Pack and project one spin block, then combine backflow and envelope jets.
  void buildSpatialOrbitalSpinTile(
      DirectBatchWorkspace& batch_workspace,
      std::vector<std::unique_ptr<DirectSpatialWorkspace>>& workspaces,
      std::size_t tile_begin,
      std::size_t tile_size,
      bool spin_up,
      const double* parameters,
      DirectBatchExecutionStatistics& statistics) const
  {
    const auto& layout = *spatial_executor_.layout_;
    const std::size_t electrons = layout.electronCount();
    const std::size_t spin_begin = spin_up ? 0 : layout.spinUpCount();
    const std::size_t spin_count =
        spin_up ? layout.spinUpCount() : layout.spinDownCount();
    if (spin_count == 0)
      return;
    const std::size_t width = layout.featureWidth();
    const std::size_t determinant_count = layout.determinantCount();
    const std::size_t channels = determinant_count * electrons;
    const std::size_t gradient_lanes = workspaces.front()->gradient_lanes_;
    const std::size_t laplacian_lanes = workspaces.front()->laplacian_lanes_;
    const std::size_t planes = 1 + gradient_lanes + laplacian_lanes;

    for (std::size_t local = 0; local < tile_size; ++local)
    {
      const DirectSpatialJetBuffer& features = workspaces[local]->features_a_;
      for (std::size_t plane = 0; plane < planes; ++plane)
      {
        const double* plane_values = plane == 0
            ? features.value.data()
            : (plane <= gradient_lanes
                   ? features.gradient.data() + (plane - 1) * features.value.size()
                   : features.laplacian.data() +
                       (plane - 1 - gradient_lanes) * features.value.size());
        for (std::size_t spin_electron = 0; spin_electron < spin_count;
             ++spin_electron)
        {
          const std::size_t electron = spin_begin + spin_electron;
          const std::size_t packed_row =
              (local * planes + plane) * spin_count + spin_electron;
          std::copy_n(plane_values + electron * width, width,
                      batch_workspace.spatial_dense_source_.data() +
                          packed_row * width);
        }
      }
    }

    const std::size_t packed_rows = tile_size * planes * spin_count;
    qmcplusplus::psiformer::batch::productReal(
        batch_workspace.spatial_dense_source_.data(),
        tensor(parameters,
               spin_up ? layout.backflow_up_ : layout.backflow_down_),
        nullptr, packed_rows, width, channels,
        batch_workspace.spatial_dense_target_.data(), statistics);

    const double* pi = tensor(parameters, spin_up ? layout.pi_up_ : layout.pi_down_);
    const double* zeta =
        tensor(parameters, spin_up ? layout.zeta_up_ : layout.zeta_down_);
    const std::size_t nuclei = layout.nucleusCount();
    auto projected = [&](std::size_t local, std::size_t plane,
                         std::size_t spin_electron,
                         std::size_t channel) -> double {
      const std::size_t row =
          (local * planes + plane) * spin_count + spin_electron;
      return batch_workspace.spatial_dense_target_[row * channels + channel];
    };

    for (std::size_t local = 0; local < tile_size; ++local)
    {
      DirectSpatialWorkspace& spatial = *workspaces[local];
      DirectSpatialJetBuffer& orbitals = spatial.orbital_matrices_;
      const GeometryPairTable& pairs = spatial.geometry_.electronNucleusPairs();
      const auto& displacements = pairs.displacements();
      const auto& distances = pairs.distances();
      for (std::size_t spin_electron = 0; spin_electron < spin_count;
           ++spin_electron)
      {
        const std::size_t electron = spin_begin + spin_electron;
        for (std::size_t determinant_index = 0;
             determinant_index < determinant_count; ++determinant_index)
          for (std::size_t orbital = 0; orbital < electrons; ++orbital)
          {
            const std::size_t channel = determinant_index * electrons + orbital;
            const std::size_t matrix_element =
                (determinant_index * electrons + electron) * electrons + orbital;
            const double backflow_value = projected(local, 0, spin_electron, channel);
            double envelope_value = 0;
            std::fill(spatial.scalar_gradient_scratch_.begin(),
                      spatial.scalar_gradient_scratch_.end(), 0.0);
            std::fill(spatial.scalar_laplacian_scratch_.begin(),
                      spatial.scalar_laplacian_scratch_.end(), 0.0);
            for (std::size_t nucleus = 0; nucleus < nuclei; ++nucleus)
            {
              const std::size_t parameter = channel * nuclei + nucleus;
              const std::size_t pair = electron * nuclei + nucleus;
              const double radius = distances[pair];
              if (radius == 0)
                throw std::runtime_error(
                    "PsiFormer batch configuration " +
                    std::to_string(tile_begin + local) +
                    " has undefined electron-nucleus spatial derivatives");
              const double inverse_radius = 1.0 / radius;
              const double decay_rate = std::abs(zeta[parameter]);
              const double exponential = std::exp(-decay_rate * radius);
              const double weighted_value = pi[parameter] * exponential;
              const double radial_first = -decay_rate * weighted_value;
              const double radial_second = decay_rate * decay_rate * weighted_value;
              envelope_value += weighted_value;

              for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
                if (DirectSpatialExecutor::laneElectron(spatial, lane) == electron)
                {
                  const std::size_t dimension =
                      DirectSpatialExecutor::laneDimension(spatial, lane);
                  spatial.scalar_gradient_scratch_[lane] +=
                      radial_first * displacements[pair][dimension] * inverse_radius;
                }
              if (spatial.mode_ == DirectSpatialMode::FULL_VGL)
                spatial.scalar_laplacian_scratch_[electron] +=
                    radial_second + 2.0 * radial_first * inverse_radius;
            }

            orbitals.value[matrix_element] = backflow_value * envelope_value;
            for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
              orbitals.gradient[lane * orbitals.value.size() + matrix_element] =
                  projected(local, 1 + lane, spin_electron, channel) *
                      envelope_value +
                  backflow_value * spatial.scalar_gradient_scratch_[lane];

            for (std::size_t laplacian_electron = 0;
                 laplacian_electron < laplacian_lanes;
                 ++laplacian_electron)
            {
              double gradient_dot = 0;
              for (std::size_t dimension = 0; dimension < 3; ++dimension)
              {
                const std::size_t lane = 3 * laplacian_electron + dimension;
                gradient_dot +=
                    projected(local, 1 + lane, spin_electron, channel) *
                    spatial.scalar_gradient_scratch_[lane];
              }
              orbitals.laplacian[
                  laplacian_electron * orbitals.value.size() + matrix_element] =
                  projected(local, 1 + gradient_lanes + laplacian_electron,
                            spin_electron, channel) * envelope_value +
                  2.0 * gradient_dot + backflow_value *
                      spatial.scalar_laplacian_scratch_[laplacian_electron];
            }
          }
      }
    }
  }

  /// Form all determinant-matrix jets from two tile-stacked spin projections.
  void buildSpatialOrbitalsTile(
      DirectBatchWorkspace& batch_workspace,
      std::vector<std::unique_ptr<DirectSpatialWorkspace>>& workspaces,
      std::size_t tile_begin,
      std::size_t tile_size,
      const double* parameters,
      DirectBatchExecutionStatistics& statistics) const
  {
    for (std::size_t local = 0; local < tile_size; ++local)
      workspaces[local]->orbital_matrices_.clear();
    buildSpatialOrbitalSpinTile(batch_workspace, workspaces, tile_begin,
                                tile_size, true, parameters, statistics);
    buildSpatialOrbitalSpinTile(batch_workspace, workspaces, tile_begin,
                                tile_size, false, parameters, statistics);
  }

  /** Complete one configuration's stable determinant and analytic cusp reduction.
   *
   * Results are written only to the pending arrays.  The public result arrays are
   * committed after every tile succeeds and the parameter version is rechecked.
   */
  void finishSpatialConfiguration(DirectBatchWorkspace& batch_workspace,
                                  DirectSpatialWorkspace& spatial,
                                  std::size_t configuration,
                                  const double* parameters,
                                  std::size_t parameter_version) const
  {
    namespace determinant = qmcplusplus::psiformer::determinant;
    try
    {
      const determinant::RealDeterminantResult determinant_result =
          spatial.determinant_workspace_.evaluateSpatial(
              spatial.orbital_matrices_.value.data(),
              spatial.orbital_matrices_.gradient.data(), spatial.gradient_lanes_,
              spatial.orbital_matrices_.laplacian.data(), spatial.laplacian_lanes_,
              spatial.output_gradient_.data(), spatial.output_lap_log_.data(),
              spatial.output_lap_ratio_.data());
      if (determinant_result.amplitude.isZero())
        throw std::runtime_error("reached an exact determinant node");

      const double cusp_value = spatial_executor_.accumulateCusp(parameters, spatial);
      for (std::size_t lane = 0; lane < spatial.gradient_lanes_; ++lane)
        spatial.output_gradient_[lane] += spatial.scalar_gradient_scratch_[lane];

      for (std::size_t electron = 0; electron < spatial.laplacian_lanes_; ++electron)
      {
        double total_squared_gradient = 0;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
        {
          const std::size_t lane = 3 * electron + dimension;
          total_squared_gradient +=
              spatial.output_gradient_[lane] * spatial.output_gradient_[lane];
        }
        spatial.output_lap_log_[electron] +=
            spatial.scalar_laplacian_scratch_[electron];
        spatial.output_lap_ratio_[electron] =
            spatial.output_lap_log_[electron] + total_squared_gradient;
      }

      const double logabs = determinant_result.amplitude.log_abs + cusp_value;
      if (!determinant::isFiniteReal(logabs))
        throw std::runtime_error("produced a non-finite log amplitude");
      for (const double component : spatial.output_gradient_)
        if (!determinant::isFiniteReal(component))
          throw std::runtime_error("produced a non-finite gradient");
      for (const double component : spatial.output_lap_log_)
        if (!determinant::isFiniteReal(component))
          throw std::runtime_error("produced a non-finite logarithmic Laplacian");
      for (const double component : spatial.output_lap_ratio_)
        if (!determinant::isFiniteReal(component))
          throw std::runtime_error("produced a non-finite Laplacian ratio");

      spatial.observed_parameter_version_ = parameter_version;
      storePendingValue(
          batch_workspace, configuration, determinant_result.amplitude.phase,
          logabs, determinant::realValue(determinant_result.amplitude, cusp_value),
          parameter_version);
      const std::size_t gradient_stride = spatial.gradient_lanes_;
      std::copy_n(spatial.output_gradient_.begin(), gradient_stride,
                  batch_workspace.pending_gradient_.begin() +
                      configuration * gradient_stride);
      if (spatial.laplacian_lanes_ != 0)
      {
        std::copy_n(spatial.output_lap_log_.begin(), spatial.laplacian_lanes_,
                    batch_workspace.pending_lap_log_.begin() +
                        configuration * spatial.laplacian_lanes_);
        std::copy_n(spatial.output_lap_ratio_.begin(), spatial.laplacian_lanes_,
                    batch_workspace.pending_lap_ratio_.begin() +
                        configuration * spatial.laplacian_lanes_);
      }
    }
    catch (const std::exception& error)
    {
      throw std::runtime_error(
          "PsiFormer batch configuration " + std::to_string(configuration) +
          " spatial finalization failed: " + error.what());
    }
  }

  static void commitSpatialOutputs(DirectBatchWorkspace& workspace,
                                   std::size_t gradient_stride,
                                   std::size_t laplacian_stride)
  {
    const std::size_t configurations = workspace.active_size_;
    std::copy_n(workspace.pending_sign_.begin(), configurations,
                workspace.sign_.begin());
    std::copy_n(workspace.pending_logabs_.begin(), configurations,
                workspace.logabs_.begin());
    std::copy_n(workspace.pending_value_.begin(), configurations,
                workspace.value_.begin());
    std::copy_n(workspace.pending_parameter_version_.begin(), configurations,
                workspace.parameter_version_.begin());
    std::copy_n(workspace.pending_gradient_.begin(),
                configurations * gradient_stride, workspace.gradient_.begin());
    if (laplacian_stride != 0)
    {
      std::copy_n(workspace.pending_lap_log_.begin(),
                  configurations * laplacian_stride, workspace.lap_log_.begin());
      std::copy_n(workspace.pending_lap_ratio_.begin(),
                  configurations * laplacian_stride, workspace.lap_ratio_.begin());
    }
  }

  DirectBatchSpatialResultView evaluateSpatialBatch(
      DirectBatchWorkspace& workspace,
      DirectSpatialMode mode,
      const std::size_t* active_electrons) const
  {
    workspace.requireCompletePositions();
    if (mode == DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT)
    {
      if (workspace.active_size_ != 0 && active_electrons == nullptr)
        throw std::invalid_argument(
            "PsiFormer active batch has no electron-index array");
      for (std::size_t configuration = 0;
           configuration < workspace.active_size_; ++configuration)
        if (active_electrons[configuration] >= workspace.electron_count_)
          throw std::out_of_range(
              "PsiFormer active batch electron index is out of range");
    }

    const auto& layout = *spatial_executor_.layout_;
    layout.validateParameterStore(spatial_executor_.parameters_);
    const std::size_t parameter_version = spatial_executor_.parameters_.version();
    const double* parameters = spatial_executor_.parameters_.flat_values().data();
    const std::size_t gradient_stride =
        mode == DirectSpatialMode::FULL_VGL ? 3 * workspace.electron_count_ : 3;
    const std::size_t laplacian_stride =
        mode == DirectSpatialMode::FULL_VGL ? workspace.electron_count_ : 0;
    std::vector<std::unique_ptr<DirectSpatialWorkspace>>& workspaces =
        spatialWorkspaces(workspace, mode);

    DirectBatchExecutionStatistics statistics;
    const std::size_t tile_limit = workspace.tile_capacity_;
    for (std::size_t tile_begin = 0; tile_begin < workspace.active_size_;
         tile_begin += tile_limit)
    {
      const std::size_t tile_size =
          std::min(tile_limit, workspace.active_size_ - tile_begin);
      ++statistics.tiles_executed;
      statistics.max_tile_occupancy =
          std::max(statistics.max_tile_occupancy, tile_size);
      buildSpatialNetworkTile(workspace, workspaces, tile_begin, tile_size,
                              active_electrons, parameters, statistics);
      buildSpatialOrbitalsTile(workspace, workspaces, tile_begin, tile_size,
                               parameters, statistics);
      for (std::size_t local = 0; local < tile_size; ++local)
        finishSpatialConfiguration(workspace, *workspaces[local],
                                   tile_begin + local, parameters,
                                   parameter_version);
    }

    if (spatial_executor_.parameters_.version() != parameter_version)
      throw std::runtime_error(
          "PsiFormer parameters changed during batch evaluation");
    commitSpatialOutputs(workspace, gradient_stride, laplacian_stride);
    workspace.statistics_ = statistics;
    return spatialView(workspace, mode, gradient_stride, laplacian_stride);
  }

  static bool hasExactSameSpinCoalescence(const double* positions,
                                          const DirectValueParameterLayout& layout) noexcept
  {
    for (std::size_t first = 0; first < layout.electronCount(); ++first)
      for (std::size_t second = first + 1; second < layout.electronCount(); ++second)
      {
        const bool first_up = first < layout.spinUpCount();
        const bool second_up = second < layout.spinUpCount();
        if (first_up != second_up)
          continue;
        bool equal = true;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          equal = equal && positions[3 * first + dimension] ==
              positions[3 * second + dimension];
        if (equal)
          return true;
      }
    return false;
  }

  static double cuspValue(const double* parameters,
                          const DirectValueParameterLayout& layout,
                          const PsiFormerGeometryCache& geometry)
  {
    const double same_alpha = layout.same_alpha_.size == 0
        ? 1.0
        : tensor(parameters, layout.same_alpha_)[0];
    const double anti_alpha = tensor(parameters, layout.anti_alpha_)[0];
    const auto& pairs = geometry.electronPairs();
    const auto& distances = geometry.electronElectronPairs().distances();
    double cusp = 0;
    for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index)
    {
      const ElectronPair pair = pairs[pair_index];
      const bool same_spin =
          (pair.first < layout.spinUpCount()) == (pair.second < layout.spinUpCount());
      const double alpha = same_spin ? same_alpha : anti_alpha;
      const double factor = same_spin ? 0.25 : 0.5;
      cusp -= factor * alpha * alpha / (alpha + distances[pair_index]);
    }
    return cusp;
  }

  void evaluateValueTile(DirectBatchWorkspace& workspace,
                         std::size_t tile_begin,
                         std::size_t tile_size,
                         const double* tile_positions,
                         const double* parameters,
                         std::size_t parameter_version,
                         DirectBatchExecutionStatistics& statistics) const
  {
    const auto& layout = *value_executor_.layout_;
    const std::size_t ne = layout.electronCount();
    const std::size_t na = layout.nucleusCount();
    const std::size_t ndet = layout.determinantCount();
    const std::size_t width = layout.featureWidth();
    const std::size_t input_width = layout.inputWidth();
    const std::size_t heads = layout.headCount();
    const std::size_t head_width = layout.headWidth();
    const std::size_t feature_elements = tile_size * ne * width;
    const std::size_t electron_rows = tile_size * ne;
    const std::size_t orbital_channels = ndet * ne;

    for (std::size_t local = 0; local < tile_size; ++local)
    {
      workspace.value_geometries_[local].update(
          GeometryPositionView::interleaved(tile_positions + local * ne * 3, ne));
      const GeometryPairTable& pairs =
          workspace.value_geometries_[local].electronNucleusPairs();
      const auto& displacements = pairs.displacements();
      const auto& radial_factors = pairs.softenedRadialFactors();
      for (std::size_t electron = 0; electron < ne; ++electron)
      {
        double* row = workspace.raw_features_.data() +
            (local * ne + electron) * input_width;
        for (std::size_t nucleus = 0; nucleus < na; ++nucleus)
        {
          const std::size_t pair = electron * na + nucleus;
          row[4 * nucleus] = radial_factors[pair].log1p_radius;
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
            row[4 * nucleus + 1 + dimension] =
                displacements[pair][dimension] * radial_factors[pair].log1p_over_radius;
        }
        row[input_width - 1] = electron < layout.spinUpCount() ? 1.0 : -1.0;
      }
    }

    namespace batch = qmcplusplus::psiformer::batch;
    batch::productReal(workspace.raw_features_.data(),
                       tensor(parameters, layout.embedding_), nullptr,
                       electron_rows, input_width, width,
                       workspace.features_a_.data(), statistics);

    double* features = workspace.features_a_.data();
    double* updated_features = workspace.features_b_.data();
    for (const auto& layer : layout.layers_)
    {
      batch::projectQkvReal(features, tensor(parameters, layer.query),
                            tensor(parameters, layer.key),
                            tensor(parameters, layer.value), electron_rows, width,
                            workspace.query_.data(), workspace.key_.data(),
                            workspace.projected_value_.data(), statistics);
      batch::attentionReal(workspace.query_.data(), workspace.key_.data(),
                           workspace.projected_value_.data(), tile_size, ne, heads,
                           head_width, workspace.attention_.data(),
                           workspace.attended_.data());
      batch::productReal(workspace.attended_.data(),
                         tensor(parameters, layer.projection), nullptr,
                         electron_rows, width, width, updated_features, statistics);
      for (std::size_t element = 0; element < feature_elements; ++element)
        updated_features[element] += features[element];

      batch::productReal(updated_features, tensor(parameters, layer.mlp_weight_0),
                         tensor(parameters, layer.mlp_bias_0), electron_rows, width,
                         width, workspace.hidden_.data(), statistics);
      for (std::size_t element = 0; element < feature_elements; ++element)
        workspace.hidden_[element] = std::tanh(workspace.hidden_[element]);
      batch::productReal(workspace.hidden_.data(),
                         tensor(parameters, layer.mlp_weight_1),
                         tensor(parameters, layer.mlp_bias_1), electron_rows, width,
                         width, workspace.attended_.data(), statistics);
      for (std::size_t element = 0; element < feature_elements; ++element)
        updated_features[element] += std::tanh(workspace.attended_[element]);
      std::swap(features, updated_features);
    }

    const std::size_t up_rows = tile_size * layout.spinUpCount();
    const std::size_t down_rows = tile_size * layout.spinDownCount();
    for (std::size_t local = 0; local < tile_size; ++local)
      for (std::size_t electron = 0; electron < ne; ++electron)
      {
        const bool spin_up = electron < layout.spinUpCount();
        const std::size_t spin_row = spin_up
            ? local * layout.spinUpCount() + electron
            : up_rows + local * layout.spinDownCount() +
                (electron - layout.spinUpCount());
        std::copy_n(features + (local * ne + electron) * width, width,
                    workspace.spin_features_.data() + spin_row * width);
      }
    if (up_rows != 0)
      batch::productReal(workspace.spin_features_.data(),
                         tensor(parameters, layout.backflow_up_), nullptr, up_rows,
                         width, orbital_channels, workspace.backflow_values_.data(),
                         statistics);
    if (down_rows != 0)
      batch::productReal(workspace.spin_features_.data() + up_rows * width,
                         tensor(parameters, layout.backflow_down_), nullptr, down_rows,
                         width, orbital_channels,
                         workspace.backflow_values_.data() + up_rows * orbital_channels,
                         statistics);

    const std::size_t matrices_per_configuration = ndet * ne * ne;
    for (std::size_t local = 0; local < tile_size; ++local)
    {
      const auto& distances =
          workspace.value_geometries_[local].electronNucleusPairs().distances();
      for (std::size_t electron = 0; electron < ne; ++electron)
      {
        const bool spin_up = electron < layout.spinUpCount();
        const std::size_t spin_row = spin_up
            ? local * layout.spinUpCount() + electron
            : up_rows + local * layout.spinDownCount() +
                (electron - layout.spinUpCount());
        const double* backflow = workspace.backflow_values_.data() +
            spin_row * orbital_channels;
        const double* pi = tensor(parameters, spin_up ? layout.pi_up_ : layout.pi_down_);
        const double* zeta =
            tensor(parameters, spin_up ? layout.zeta_up_ : layout.zeta_down_);
        for (std::size_t determinant = 0; determinant < ndet; ++determinant)
          for (std::size_t orbital = 0; orbital < ne; ++orbital)
          {
            const std::size_t channel = determinant * ne + orbital;
            double envelope = 0;
            for (std::size_t nucleus = 0; nucleus < na; ++nucleus)
            {
              const std::size_t parameter = channel * na + nucleus;
              envelope += pi[parameter] *
                  std::exp(-std::abs(zeta[parameter] *
                                     distances[electron * na + nucleus]));
            }
            workspace.orbital_matrices_[local * matrices_per_configuration +
                (determinant * ne + electron) * ne + orbital] =
                backflow[channel] * envelope;
          }
      }

      const std::size_t configuration = tile_begin + local;
      const double* positions = tile_positions + local * ne * 3;
      if (hasExactSameSpinCoalescence(positions, layout))
      {
        storePendingValue(workspace, configuration, 0.0,
                          -std::numeric_limits<double>::infinity(), 0.0,
                          parameter_version);
        continue;
      }

      namespace determinant = qmcplusplus::psiformer::determinant;
      const determinant::RealDeterminantResult determinant_result =
          workspace.determinant_workspaces_[local]->evaluateValue(
              workspace.orbital_matrices_.data() + local * matrices_per_configuration);
      if (determinant_result.amplitude.isZero())
      {
        storePendingValue(workspace, configuration, 0.0,
                          -std::numeric_limits<double>::infinity(), 0.0,
                          parameter_version);
        continue;
      }
      const double cusp = cuspValue(
          parameters, layout, workspace.value_geometries_[local]);
      const double logabs = determinant_result.amplitude.log_abs + cusp;
      if (!determinant::isFiniteReal(logabs))
        throw std::runtime_error(
            "PsiFormer batch configuration " + std::to_string(configuration) +
            " produced a non-finite wavefunction value");
      storePendingValue(workspace, configuration,
                        determinant_result.amplitude.phase, logabs,
                        determinant::realValue(determinant_result.amplitude, cusp),
                        parameter_version);
    }
  }

  static void storePendingValue(DirectBatchWorkspace& workspace,
                                std::size_t configuration,
                                double sign,
                                double logabs,
                                double value,
                                std::size_t parameter_version)
  {
    workspace.pending_sign_[configuration] = sign;
    workspace.pending_logabs_[configuration] = logabs;
    workspace.pending_value_[configuration] = value;
    workspace.pending_parameter_version_[configuration] = parameter_version;
  }

  static DirectBatchValueResultView valueView(const DirectBatchWorkspace& workspace)
  {
    return {workspace.active_size_, workspace.sign_.data(), workspace.logabs_.data(),
            workspace.value_.data(), workspace.parameter_version_.data()};
  }

  static DirectBatchSpatialResultView spatialView(const DirectBatchWorkspace& workspace,
                                                  DirectSpatialMode mode,
                                                  std::size_t gradient_stride,
                                                  std::size_t laplacian_stride)
  {
    DirectBatchSpatialResultView result;
    static_cast<DirectBatchValueResultView&>(result) = valueView(workspace);
    result.mode = mode;
    result.gradient_stride = gradient_stride;
    result.laplacian_stride = laplacian_stride;
    result.gradient = workspace.gradient_.data();
    result.lap_log = laplacian_stride == 0 ? nullptr : workspace.lap_log_.data();
    result.lap_ratio = laplacian_stride == 0 ? nullptr : workspace.lap_ratio_.data();
    return result;
  }

  const DirectValueExecutor& value_executor_;
  const DirectSpatialExecutor& spatial_executor_;
};

} // namespace pf

#endif // QMCPLUSPLUS_PSIFORMER_BATCH_EXECUTOR_H
