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
 * to an explicitly bounded tile capacity.  VALUE_ONLY executes shared-weight dense
 * kernels over all configuration/electron rows in a tile.  The spatial modes retain
 * the validated scalar algebra temporarily, but reuse at most one scalar workspace per
 * tile slot; they are the next replacement boundary and are reported explicitly by
 * execution statistics.
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
  {}

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
    if (active_size_ != 0)
      prepareScratch(active_mode_, std::min(active_size_, capacity));
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

    growVector(electron_positions_, position_count);
    growVector(position_ready_, position_count);
    reserveOutputs(size, gradient_count, laplacian_count);
    prepareScratch(mode, std::min(size, tile_capacity_));

    active_mode_ = mode;
    active_size_ = size;
    mode_capacity_[modeIndex(mode)] = std::max(mode_capacity_[modeIndex(mode)], size);
    std::fill_n(position_ready_.begin(), position_count, static_cast<unsigned char>(0));
  }

  /// Set one finite Cartesian coordinate in the active logical request.
  void setPosition(std::size_t configuration,
                   std::size_t electron,
                   std::size_t dimension,
                   double value)
  {
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

  std::size_t size() const noexcept { return active_size_; }
  std::size_t electronCount() const noexcept { return electron_count_; }
  std::size_t tileCapacity() const noexcept { return tile_capacity_; }

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
    mix_vector(sign_);
    mix_vector(logabs_);
    mix_vector(value_);
    mix_vector(parameter_version_);
    mix_vector(pending_sign_);
    mix_vector(pending_logabs_);
    mix_vector(pending_value_);
    mix_vector(pending_parameter_version_);
    mix_vector(gradient_);
    mix_vector(lap_log_);
    mix_vector(lap_ratio_);

    switch (mode)
    {
    case DirectBatchMode::VALUE_ONLY:
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
      for (const auto& spatial : full_workspaces_)
        mix_value(reinterpret_cast<std::uintptr_t>(spatial.get()));
      break;
    case DirectBatchMode::ACTIVE_ELECTRON_GRADIENT:
      for (const auto& spatial : active_workspaces_)
        mix_value(reinterpret_cast<std::uintptr_t>(spatial.get()));
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
    bytes += sign_.capacity() * sizeof(double);
    bytes += logabs_.capacity() * sizeof(double);
    bytes += value_.capacity() * sizeof(double);
    bytes += parameter_version_.capacity() * sizeof(std::size_t);
    bytes += pending_sign_.capacity() * sizeof(double);
    bytes += pending_logabs_.capacity() * sizeof(double);
    bytes += pending_value_.capacity() * sizeof(double);
    bytes += pending_parameter_version_.capacity() * sizeof(std::size_t);
    bytes += gradient_.capacity() * sizeof(double);
    bytes += lap_log_.capacity() * sizeof(double);
    bytes += lap_ratio_.capacity() * sizeof(double);
    return bytes;
  }

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
    growVector(lap_log_, laplacian_count);
    growVector(lap_ratio_, laplacian_count);
  }

  void prepareScratch(DirectBatchMode mode, std::size_t capacity)
  {
    switch (mode)
    {
    case DirectBatchMode::VALUE_ONLY:
      growValueScratch(capacity);
      break;
    case DirectBatchMode::FULL_VGL:
      growWorkspaces(full_workspaces_, capacity, [this]() {
        return spatial_executor_->makeWorkspace(DirectSpatialMode::FULL_VGL);
      });
      break;
    case DirectBatchMode::ACTIVE_ELECTRON_GRADIENT:
      growWorkspaces(active_workspaces_, capacity, [this]() {
        return spatial_executor_->makeWorkspace(DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT);
      });
      break;
    }
  }

  /// Grow every value-only tile buffer after all element-count products are checked.
  void growValueScratch(std::size_t capacity)
  {
    if (capacity <= value_tile_capacity_)
      return;
    const auto& layout = *value_executor_->layout();
    namespace batch = qmcplusplus::psiformer::batch;
    const std::size_t electron_rows = batch::checkedProduct(
        capacity, electron_count_, "PsiFormer value tile electron extent overflowed");
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
    const std::size_t backflow_elements = batch::checkedProduct(
        electron_rows, orbital_channels,
        "PsiFormer value tile backflow extent overflowed");
    // Both packed backflow outputs and final matrices contain
    // T * Ne * (D * Ne) scalar entries; only their index ordering differs.
    const std::size_t orbital_elements = backflow_elements;

    value_geometries_.reserve(capacity);
    while (value_geometries_.size() < capacity)
      value_geometries_.emplace_back(
          electron_count_, value_executor_->nuclearPositions(), value_executor_->boundary());
    determinant_workspaces_.reserve(capacity);
    while (determinant_workspaces_.size() < capacity)
      determinant_workspaces_.push_back(
          std::make_unique<qmcplusplus::psiformer::determinant::RealOpenDeterminantWorkspace>(
              layout.determinantCount(), electron_count_));

    growVector(raw_features_, raw_elements);
    growVector(features_a_, feature_elements);
    growVector(features_b_, feature_elements);
    growVector(query_, feature_elements);
    growVector(key_, feature_elements);
    growVector(projected_value_, feature_elements);
    growVector(attention_, attention_elements);
    growVector(attended_, feature_elements);
    growVector(hidden_, feature_elements);
    growVector(spin_features_, feature_elements);
    growVector(backflow_values_, backflow_elements);
    growVector(orbital_matrices_, orbital_elements);
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

  const DirectValueExecutor* value_executor_;
  const DirectSpatialExecutor* spatial_executor_;
  const std::size_t electron_count_;
  DirectBatchMode active_mode_ = DirectBatchMode::VALUE_ONLY;
  std::size_t active_size_ = 0;
  std::size_t tile_capacity_ = default_tile_capacity;
  std::size_t value_tile_capacity_ = 0;
  std::array<std::size_t, 3> mode_capacity_{};
  DirectBatchExecutionStatistics statistics_{};

  std::vector<double> electron_positions_;
  std::vector<unsigned char> position_ready_;
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

  std::vector<double> sign_;
  std::vector<double> logabs_;
  std::vector<double> value_;
  std::vector<std::size_t> parameter_version_;
  std::vector<double> pending_sign_;
  std::vector<double> pending_logabs_;
  std::vector<double> pending_value_;
  std::vector<std::size_t> pending_parameter_version_;
  std::vector<double> gradient_;
  std::vector<double> lap_log_;
  std::vector<double> lap_ratio_;
};

/** Execute configuration-major batches against one immutable model and plan. */
class DirectBatchExecutor
{
public:
  DirectBatchExecutor(const DirectValueExecutor& value_executor,
                      const DirectSpatialExecutor& spatial_executor)
      : value_executor_(value_executor), spatial_executor_(spatial_executor)
  {}

  std::unique_ptr<DirectBatchWorkspace> makeWorkspace() const
  { return std::make_unique<DirectBatchWorkspace>(value_executor_, spatial_executor_); }

  /** Evaluate a true tiled value batch without invoking the scalar value executor. */
  DirectBatchValueResultView evaluateValues(DirectBatchWorkspace& workspace) const
  {
    requireWorkspace(workspace);
    requireMode(workspace, DirectBatchMode::VALUE_ONLY);
    workspace.requireCompletePositions();
    const auto& layout = *value_executor_.layout_;
    layout.validateParameterStore(value_executor_.parameters_);

    DirectBatchExecutionStatistics statistics;
    const std::size_t parameter_version = value_executor_.parameters_.version();
    const double* parameters = value_executor_.parameters_.flat_values().data();
    const std::size_t tile_limit = workspace.tile_capacity_;
    for (std::size_t tile_begin = 0; tile_begin < workspace.active_size_;
         tile_begin += tile_limit)
    {
      const std::size_t tile_size = std::min(tile_limit, workspace.active_size_ - tile_begin);
      ++statistics.tiles_executed;
      statistics.max_tile_occupancy = std::max(statistics.max_tile_occupancy, tile_size);
      evaluateValueTile(workspace, tile_begin, tile_size, parameters,
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

  /** Evaluate full VGL through bounded transitional scalar tile scratch. */
  DirectBatchSpatialResultView evaluateFull(DirectBatchWorkspace& workspace) const
  {
    requireWorkspace(workspace);
    requireMode(workspace, DirectBatchMode::FULL_VGL);
    workspace.requireCompletePositions();
    DirectBatchExecutionStatistics statistics;
    const std::size_t gradient_stride = 3 * workspace.electron_count_;
    for (std::size_t tile_begin = 0; tile_begin < workspace.active_size_;
         tile_begin += workspace.tile_capacity_)
    {
      const std::size_t tile_size =
          std::min(workspace.tile_capacity_, workspace.active_size_ - tile_begin);
      ++statistics.tiles_executed;
      statistics.max_tile_occupancy = std::max(statistics.max_tile_occupancy, tile_size);
      for (std::size_t local = 0; local < tile_size; ++local)
      {
        const std::size_t configuration = tile_begin + local;
        workspace.full_workspaces_[local]->setPositions(workspace.positionView(configuration));
        const DirectSpatialResultView result =
            spatial_executor_.evaluateFull(*workspace.full_workspaces_[local]);
        ++statistics.scalar_executor_calls;
        storeValue(workspace, configuration, result.sign, result.logabs, result.value,
                   result.parameter_version);
        std::copy(result.gradient.begin(), result.gradient.end(),
                  workspace.gradient_.begin() + configuration * gradient_stride);
        std::copy(result.lap_log.begin(), result.lap_log.end(),
                  workspace.lap_log_.begin() + configuration * workspace.electron_count_);
        std::copy(result.lap_ratio.begin(), result.lap_ratio.end(),
                  workspace.lap_ratio_.begin() + configuration * workspace.electron_count_);
      }
    }
    workspace.statistics_ = statistics;
    return spatialView(workspace, DirectSpatialMode::FULL_VGL, gradient_stride,
                       workspace.electron_count_);
  }

  /** Evaluate active gradients through bounded transitional scalar tile scratch. */
  DirectBatchSpatialResultView evaluateActive(DirectBatchWorkspace& workspace,
                                               const std::size_t* active_electrons) const
  {
    requireWorkspace(workspace);
    requireMode(workspace, DirectBatchMode::ACTIVE_ELECTRON_GRADIENT);
    workspace.requireCompletePositions();
    if (workspace.active_size_ != 0 && active_electrons == nullptr)
      throw std::invalid_argument("PsiFormer active batch has no electron-index array");
    for (std::size_t configuration = 0; configuration < workspace.active_size_; ++configuration)
      if (active_electrons[configuration] >= workspace.electron_count_)
        throw std::out_of_range("PsiFormer active batch electron index is out of range");

    DirectBatchExecutionStatistics statistics;
    constexpr std::size_t gradient_stride = 3;
    for (std::size_t tile_begin = 0; tile_begin < workspace.active_size_;
         tile_begin += workspace.tile_capacity_)
    {
      const std::size_t tile_size =
          std::min(workspace.tile_capacity_, workspace.active_size_ - tile_begin);
      ++statistics.tiles_executed;
      statistics.max_tile_occupancy = std::max(statistics.max_tile_occupancy, tile_size);
      for (std::size_t local = 0; local < tile_size; ++local)
      {
        const std::size_t configuration = tile_begin + local;
        workspace.active_workspaces_[local]->setPositions(workspace.positionView(configuration));
        const DirectSpatialResultView result = spatial_executor_.evaluateActive(
            *workspace.active_workspaces_[local], active_electrons[configuration]);
        ++statistics.scalar_executor_calls;
        storeValue(workspace, configuration, result.sign, result.logabs, result.value,
                   result.parameter_version);
        std::copy(result.gradient.begin(), result.gradient.end(),
                  workspace.gradient_.begin() + configuration * gradient_stride);
      }
    }
    workspace.statistics_ = statistics;
    return spatialView(workspace, DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT,
                       gradient_stride, 0);
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

  void requireWorkspace(const DirectBatchWorkspace& workspace) const
  {
    if (workspace.value_executor_ != &value_executor_ ||
        workspace.spatial_executor_ != &spatial_executor_)
      throw std::invalid_argument("PsiFormer batch workspace belongs to another executor");
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
      const std::size_t configuration = tile_begin + local;
      workspace.value_geometries_[local].update(workspace.positionView(configuration));
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
      const double* positions = workspace.electron_positions_.data() +
          configuration * ne * 3;
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

  static void storeValue(DirectBatchWorkspace& workspace,
                         std::size_t configuration,
                         double sign,
                         double logabs,
                         double value,
                         std::size_t parameter_version)
  {
    workspace.sign_[configuration] = sign;
    workspace.logabs_[configuration] = logabs;
    workspace.value_[configuration] = value;
    workspace.parameter_version_[configuration] = parameter_version;
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
