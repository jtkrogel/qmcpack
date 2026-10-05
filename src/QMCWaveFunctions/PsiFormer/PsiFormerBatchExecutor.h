//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerBatchExecutor.h
 * @brief Batch-capacity storage and direct multi-configuration PsiFormer APIs.
 *
 * The first multiwalker implementation deliberately keeps the already validated
 * single-configuration algebra kernels.  It moves ownership, input packing, result
 * layout, and parameter-version synchronization to one batch API so component calls
 * no longer dispatch through the serialized WaveFunctionComponent fallback.  The
 * configuration loop in this class is the intended replacement point for grouped or
 * strided-batched dense and attention kernels; callers and ResourceCollection
 * ownership do not need to change when those kernels arrive.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_SPATIAL_EXECUTOR_H
#error "Include PsiFormerSpatialExecutor.h before PsiFormerBatchExecutor.h"
#endif

#ifndef QMCPLUSPLUS_PSIFORMER_BATCH_EXECUTOR_H
#define QMCPLUSPLUS_PSIFORMER_BATCH_EXECUTOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace pf
{

/// Select the independently sized scratch family needed by a batch call.
enum class DirectBatchMode
{
  VALUE_ONLY,
  FULL_VGL,
  ACTIVE_ELECTRON_GRADIENT
};

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

/** Own independently growing value, full-VGL, and active-gradient batch storage.
 *
 * Capacity growth is explicit and occurs before input packing.  Once a capacity is
 * sufficient, evaluations in that mode neither allocate nor resize.  The class is
 * movable but not copyable so mutable scratch cannot accidentally be shared between
 * a clone and a crowd resource.
 */
class DirectBatchWorkspace
{
public:
  /// Bind batch scratch to immutable value and spatial executors for one model.
  DirectBatchWorkspace(const DirectValueExecutor& value_executor,
                       const DirectSpatialExecutor& spatial_executor)
      : value_executor_(&value_executor),
        spatial_executor_(&spatial_executor),
        electron_count_(value_executor.layout()->electronCount())
  {}

  /// Mutable batch scratch must never be shared by copying.
  DirectBatchWorkspace(const DirectBatchWorkspace&) = delete;

  /// Copy assignment is disabled for the same ownership reason.
  DirectBatchWorkspace& operator=(const DirectBatchWorkspace&) = delete;

  /// Moving transfers exclusive ownership of every mode-specific buffer.
  DirectBatchWorkspace(DirectBatchWorkspace&&) = default;

  /// Move assignment likewise transfers exclusive scratch ownership.
  DirectBatchWorkspace& operator=(DirectBatchWorkspace&&) = default;

  /// Grow one mode to at least ``capacity`` configurations and select its active size.
  void resize(DirectBatchMode mode, std::size_t size)
  {
    switch (mode)
    {
    case DirectBatchMode::VALUE_ONLY:
      grow(value_workspaces_, size, [this]() { return value_executor_->makeWorkspace(); });
      break;
    case DirectBatchMode::FULL_VGL:
      grow(full_workspaces_, size, [this]() {
        return spatial_executor_->makeWorkspace(DirectSpatialMode::FULL_VGL);
      });
      break;
    case DirectBatchMode::ACTIVE_ELECTRON_GRADIENT:
      grow(active_workspaces_, size, [this]() {
        return spatial_executor_->makeWorkspace(DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT);
      });
      break;
    }
    active_mode_ = mode;
    active_size_ = size;
    reserveOutputs(mode, size);
  }

  /// Set one Cartesian coordinate in the currently selected scratch family.
  void setPosition(std::size_t configuration,
                   std::size_t electron,
                   std::size_t dimension,
                   double value)
  {
    if (configuration >= active_size_)
      throw std::out_of_range("PsiFormer batch configuration index is out of range");
    switch (active_mode_)
    {
    case DirectBatchMode::VALUE_ONLY:
      value_workspaces_[configuration]->setPosition(electron, dimension, value);
      break;
    case DirectBatchMode::FULL_VGL:
      full_workspaces_[configuration]->setPosition(electron, dimension, value);
      break;
    case DirectBatchMode::ACTIVE_ELECTRON_GRADIENT:
      active_workspaces_[configuration]->setPosition(electron, dimension, value);
      break;
    }
  }

  /// Return the number of configurations selected by the most recent resize.
  std::size_t size() const noexcept { return active_size_; }

  /// Return the fixed electron count inherited from the model layout.
  std::size_t electronCount() const noexcept { return electron_count_; }

  /// Return allocated configuration capacity for one mode.
  std::size_t capacity(DirectBatchMode mode) const noexcept
  {
    switch (mode)
    {
    case DirectBatchMode::VALUE_ONLY:
      return value_workspaces_.size();
    case DirectBatchMode::FULL_VGL:
      return full_workspaces_.size();
    case DirectBatchMode::ACTIVE_ELECTRON_GRADIENT:
      return active_workspaces_.size();
    }
    return 0;
  }

  /** Hash workspace identities and output capacities for resource-lifetime tests.
   * This is a process-local diagnostic and is not a persistent model fingerprint.
   */
  std::size_t storageFingerprint(DirectBatchMode mode) const noexcept
  {
    std::size_t hash = 1469598103934665603ULL;
    auto mix = [&hash](std::uintptr_t value) {
      hash ^= value;
      hash *= 1099511628211ULL;
    };
    auto mix_vector = [&mix](const auto& values) {
      mix(reinterpret_cast<std::uintptr_t>(values.data()));
      mix(values.capacity());
    };
    switch (mode)
    {
    case DirectBatchMode::VALUE_ONLY:
      for (const auto& workspace : value_workspaces_)
        mix(reinterpret_cast<std::uintptr_t>(workspace.get()));
      break;
    case DirectBatchMode::FULL_VGL:
      for (const auto& workspace : full_workspaces_)
        mix(reinterpret_cast<std::uintptr_t>(workspace.get()));
      break;
    case DirectBatchMode::ACTIVE_ELECTRON_GRADIENT:
      for (const auto& workspace : active_workspaces_)
        mix(reinterpret_cast<std::uintptr_t>(workspace.get()));
      break;
    }
    mix_vector(sign_);
    mix_vector(logabs_);
    mix_vector(value_);
    mix_vector(parameter_version_);
    mix_vector(gradient_);
    mix_vector(lap_log_);
    mix_vector(lap_ratio_);
    return hash;
  }

  /** Return bytes reserved by all currently prepared mode workspaces and outputs.
   *
   * A batch workspace may retain high-water marks for more than one mode.  The
   * returned total therefore describes its complete resident numeric storage,
   * which is the useful quantity for crowd-size benchmark manifests.
   */
  std::size_t vectorStorageBytes() const noexcept
  {
    std::size_t bytes = 0;
    for (const auto& workspace : value_workspaces_)
      bytes += workspace->vectorStorageBytes();
    for (const auto& workspace : full_workspaces_)
      bytes += workspace->vectorStorageBytes();
    for (const auto& workspace : active_workspaces_)
      bytes += workspace->vectorStorageBytes();

    bytes += sign_.capacity() * sizeof(double);
    bytes += logabs_.capacity() * sizeof(double);
    bytes += value_.capacity() * sizeof(double);
    bytes += parameter_version_.capacity() * sizeof(std::size_t);
    bytes += gradient_.capacity() * sizeof(double);
    bytes += lap_log_.capacity() * sizeof(double);
    bytes += lap_ratio_.capacity() * sizeof(double);
    return bytes;
  }

private:
  friend class DirectBatchExecutor;

  template<class Workspace, class Factory>
  /// Construct mode-specific single-configuration workspaces up to a requested capacity.
  static void grow(std::vector<std::unique_ptr<Workspace>>& workspaces,
                   std::size_t capacity,
                   Factory&& factory)
  {
    workspaces.reserve(std::max(workspaces.capacity(), capacity));
    while (workspaces.size() < capacity)
      workspaces.emplace_back(factory());
  }

  template<class T>
  /// Grow a contiguous output buffer without shrinking warmed capacity.
  static void growOutput(std::vector<T>& output, std::size_t size)
  {
    if (output.size() < size)
      output.resize(size);
  }

  /// Ensure every output required by one mode can hold the active batch.
  void reserveOutputs(DirectBatchMode mode, std::size_t size)
  {
    growOutput(sign_, size);
    growOutput(logabs_, size);
    growOutput(value_, size);
    growOutput(parameter_version_, size);
    const std::size_t gradient_stride = mode == DirectBatchMode::FULL_VGL ? 3 * electron_count_ : 3;
    if (mode != DirectBatchMode::VALUE_ONLY)
      growOutput(gradient_, size * gradient_stride);
    if (mode == DirectBatchMode::FULL_VGL)
    {
      growOutput(lap_log_, size * electron_count_);
      growOutput(lap_ratio_, size * electron_count_);
    }
  }

  const DirectValueExecutor* value_executor_;
  const DirectSpatialExecutor* spatial_executor_;
  const std::size_t electron_count_;
  DirectBatchMode active_mode_ = DirectBatchMode::VALUE_ONLY;
  std::size_t active_size_ = 0;
  std::vector<std::unique_ptr<DirectValueWorkspace>> value_workspaces_;
  std::vector<std::unique_ptr<DirectSpatialWorkspace>> full_workspaces_;
  std::vector<std::unique_ptr<DirectSpatialWorkspace>> active_workspaces_;
  std::vector<double> sign_;
  std::vector<double> logabs_;
  std::vector<double> value_;
  std::vector<std::size_t> parameter_version_;
  std::vector<double> gradient_;
  std::vector<double> lap_log_;
  std::vector<double> lap_ratio_;
};

/** Execute configuration-major batches against one immutable model and plan.
 *
 * The current implementation loops over validated direct kernels inside this native
 * batch boundary.  It is not a WaveFunctionComponent serialization fallback: model
 * dispatch, storage ownership, output packing, and version observation happen once
 * per batch.
 */
class DirectBatchExecutor
{
public:
  /// Bind the batch boundary to the validated scalar direct kernels.
  DirectBatchExecutor(const DirectValueExecutor& value_executor,
                      const DirectSpatialExecutor& spatial_executor)
      : value_executor_(value_executor), spatial_executor_(spatial_executor)
  {}

  /// Allocate an initially empty workspace suitable for clone or crowd ownership.
  std::unique_ptr<DirectBatchWorkspace> makeWorkspace() const
  { return std::make_unique<DirectBatchWorkspace>(value_executor_, spatial_executor_); }

  /// Evaluate a value-only batch and return a non-owning configuration-major view.
  DirectBatchValueResultView evaluateValues(DirectBatchWorkspace& workspace) const
  {
    requireMode(workspace, DirectBatchMode::VALUE_ONLY);
    for (std::size_t configuration = 0; configuration < workspace.active_size_; ++configuration)
    {
      const DirectValueResult result = value_executor_.evaluate(*workspace.value_workspaces_[configuration]);
      storeValue(workspace, configuration, result.sign, result.logabs, result.value,
                 result.parameter_version);
    }
    return valueView(workspace);
  }

  /// Evaluate complete VGL data for every active configuration.
  DirectBatchSpatialResultView evaluateFull(DirectBatchWorkspace& workspace) const
  {
    requireMode(workspace, DirectBatchMode::FULL_VGL);
    const std::size_t gradient_stride = 3 * workspace.electron_count_;
    for (std::size_t configuration = 0; configuration < workspace.active_size_; ++configuration)
    {
      const DirectSpatialResultView result =
          spatial_executor_.evaluateFull(*workspace.full_workspaces_[configuration]);
      storeValue(workspace, configuration, result.sign, result.logabs, result.value,
                 result.parameter_version);
      std::copy(result.gradient.begin(), result.gradient.end(),
                workspace.gradient_.begin() + configuration * gradient_stride);
      std::copy(result.lap_log.begin(), result.lap_log.end(),
                workspace.lap_log_.begin() + configuration * workspace.electron_count_);
      std::copy(result.lap_ratio.begin(), result.lap_ratio.end(),
                workspace.lap_ratio_.begin() + configuration * workspace.electron_count_);
    }
    return spatialView(workspace, DirectSpatialMode::FULL_VGL, gradient_stride,
                       workspace.electron_count_);
  }

  /// Evaluate one independently selected electron gradient per configuration.
  DirectBatchSpatialResultView evaluateActive(DirectBatchWorkspace& workspace,
                                               const std::size_t* active_electrons) const
  {
    requireMode(workspace, DirectBatchMode::ACTIVE_ELECTRON_GRADIENT);
    if (workspace.active_size_ != 0 && active_electrons == nullptr)
      throw std::invalid_argument("PsiFormer active batch has no electron-index array");
    constexpr std::size_t gradient_stride = 3;
    for (std::size_t configuration = 0; configuration < workspace.active_size_; ++configuration)
    {
      const DirectSpatialResultView result = spatial_executor_.evaluateActive(
          *workspace.active_workspaces_[configuration], active_electrons[configuration]);
      storeValue(workspace, configuration, result.sign, result.logabs, result.value,
                 result.parameter_version);
      std::copy(result.gradient.begin(), result.gradient.end(),
                workspace.gradient_.begin() + configuration * gradient_stride);
    }
    return spatialView(workspace, DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT,
                       gradient_stride, 0);
  }

private:
  /// Reject accidental reuse of a workspace prepared for another request family.
  static void requireMode(const DirectBatchWorkspace& workspace, DirectBatchMode expected)
  {
    if (workspace.active_mode_ != expected)
      throw std::logic_error("PsiFormer batch workspace mode does not match the requested evaluation");
  }

  /// Store common amplitude metadata for one configuration.
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

  /// Expose the active prefix of value outputs without copying.
  static DirectBatchValueResultView valueView(const DirectBatchWorkspace& workspace)
  {
    return {workspace.active_size_, workspace.sign_.data(), workspace.logabs_.data(),
            workspace.value_.data(), workspace.parameter_version_.data()};
  }

  /// Expose one spatial result layout with explicit gradient/Laplacian strides.
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

#endif
