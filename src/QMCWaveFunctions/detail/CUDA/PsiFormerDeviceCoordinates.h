//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceCoordinates.h
 * @brief Prepared packed-coordinate ownership, independent of parameter images.
 * Sources currently upload from the host. Borrowed views validate source lifetime
 * and generation; a synchronous copy into inference-owned storage ends the borrow.
 * This primitive does not retain samples in production or generate proposals.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COORDINATES_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COORDINATES_H

#include "PsiFormerDeviceResources.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"
#include "Platforms/CUDA/QueueCUDA.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace qmcplusplus::psiformer::device
{
// Forward declarations keep the coordinate owner independent of graph definitions.
class PreparedDeviceModelMetadata;

class PreparedDeviceModel;

class PreparedDeviceInference;

class PreparedDeviceCoordinates;

class PreparedDeviceMetropolis;

namespace coordinate_detail
{
/** Keep allocation and mutation synchronized without retaining parameter values. */
struct SourceState
{
  /// Check preparation extents before allocating the one coordinate buffer.
  SourceState(std::shared_ptr<const PreparedDeviceModelMetadata> family_in,
              std::size_t electrons_in, std::size_t capacity_in)
      : family(std::move(family_in)), electrons(electrons_in), capacity(capacity_in)
  {
    if (!family || electrons == 0 || capacity == 0 ||
        electrons > std::numeric_limits<std::size_t>::max() / 3 / capacity)
      throw std::invalid_argument("Device coordinate source has invalid preparation extents");
    positions.reserve(3 * electrons * capacity);
  }

  std::shared_ptr<const PreparedDeviceModelMetadata> family;
  const std::size_t electrons, capacity;
  DeviceBuffer<double> positions;
  std::mutex mutex;
  std::size_t generation = 0, live = 0;
  bool ready = false;
};
} // namespace coordinate_detail

/** An owner-issued contiguous [configuration][physical electron][xyz] borrow.
 * No raw-pointer constructor exists. Mutation revokes every earlier generation;
 * destroying the owner expires unconsumed views. A consumer holds the state lock
 * through queue completion, so replacement/destruction cannot race its D2D copy.
 */
class PackedDeviceCoordinatesView
{
public:
  /// Represent no source; inference rejects this default view before launching.
  PackedDeviceCoordinatesView() = default;

  /// Return the requested live configuration count, never source capacity.
  std::size_t batchCount() const noexcept { return count_; }

  /// Return the physical electron extent of one packed configuration.
  std::size_t physicalElectrons() const noexcept { return electrons_; }

  /// Return the complete checked coordinate-element extent of this borrow.
  std::size_t elements() const noexcept { return count_ * electrons_ * 3; }

  /// Identify the device that owns the source allocation.
  int deviceOrdinal() const noexcept { return device_; }

  /// Identify the source replacement epoch independently of model versions.
  std::size_t sourceGeneration() const noexcept { return generation_; }

  /// Identify one source lifetime for diagnostics, not permission to dereference it.
  const void* sourceIdentity() const noexcept { return identity_; }

private:
  friend class PreparedDeviceCoordinates;
  friend class PreparedDeviceECPObservation;
  friend class PreparedDeviceInference;

  /// Snapshot immutable extent/identity fields while the owner holds its lock.
  PackedDeviceCoordinatesView(const std::shared_ptr<coordinate_detail::SourceState>& state,
                              std::size_t begin, std::size_t count)
      : state_(state), identity_(state.get()), generation_(state->generation), begin_(begin), count_(count),
        electrons_(state->electrons), device_(state->positions.deviceOrdinal())
  {}

  /// Validate the complete borrow before copying; hold mutation exclusion to sync.
  void copyTo(DeviceBuffer<double>& destination, compute::Queue<PlatformKind::CUDA>& queue,
              const std::shared_ptr<const PreparedDeviceModelMetadata>& family, std::size_t electrons) const
  {
    const auto state = state_.lock();
    if (!state)
      throw std::invalid_argument("Device coordinate source lifetime expired");
    std::lock_guard lock(state->mutex);
    if (!state->ready || state.get() != identity_ || state->generation != generation_ ||
        state->family.get() != family.get() || electrons_ != electrons || state->electrons != electrons ||
        count_ == 0 || begin_ > state->live || count_ > state->live - begin_ ||
        device_ != state->positions.deviceOrdinal() || device_ != destination.deviceOrdinal() ||
        elements() > destination.capacity())
      throw std::invalid_argument("Device coordinate borrow has stale identity, foreign layout/device or invalid extent");
    int current = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&current), "coordinate copy current device");
#else
    checkRuntime(cudaGetDevice(&current), "coordinate copy current device");
#endif
    if (current != device_)
      throw std::invalid_argument("Device coordinate borrow belongs to another current device");
    const auto bytes = checkedDeviceBytes<double>(elements());
    const auto* source = state->positions.data() + begin_ * electrons_ * 3;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemcpyAsync(destination.data(), source, bytes, hipMemcpyDeviceToDevice, queue.getNative()),
                 "coordinate D2D copy");
#else
    checkRuntime(cudaMemcpyAsync(destination.data(), source, bytes, cudaMemcpyDeviceToDevice, queue.getNative()),
                 "coordinate D2D copy");
#endif
    queue.sync();
    deviceMemoryMetrics(device_).copied(bytes);
  }

  std::weak_ptr<coordinate_detail::SourceState> state_;
  const void* identity_ = nullptr;
  std::size_t generation_ = 0, begin_ = 0, count_ = 0, electrons_ = 0;
  int device_ = -1;
};

/** Prepare one bounded source with no neural graph or canonical parameter buffer.
 * Only the metadata family is retained. Successor parameter images can consume
 * the same coordinates; a source's generation is not a parameter/sample version.
 * Concurrent calls serialize on the state mutex, including complete D2D borrows.
 */
class PreparedDeviceCoordinates
{
public:
  /// Coordinate source ownership is unique even though copies hold state alive.
  PreparedDeviceCoordinates(const PreparedDeviceCoordinates&) = delete;

  /// Never replace an address-stable source through implicit ownership assignment.
  PreparedDeviceCoordinates& operator=(const PreparedDeviceCoordinates&) = delete;

  /// Revoke remaining views after every currently borrowing copy has completed.
  ~PreparedDeviceCoordinates()
  {
    std::lock_guard lock(state_->mutex);
    state_->ready = false;
  }

  /// Replace the live packed population, revoking old views even on failure.
  void upload(const std::vector<double>& positions)
  {
    std::lock_guard lock(state_->mutex);
    state_->ready = false;
    state_->live = 0;
    if (state_->generation == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Device coordinate source generation exhausted");
    ++state_->generation;
    const auto stride = 3 * state_->electrons;
    if (positions.empty() || positions.size() > state_->positions.capacity() || positions.size() % stride != 0)
      throw std::invalid_argument("Device coordinate upload exceeds capacity or has an incomplete configuration");
    for (const double value : positions)
      if (!device_math::isFiniteBinary64(value))
        throw std::invalid_argument("Device coordinate upload contains nonfinite values");
    int current = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&current), "coordinate upload current device");
#else
    checkRuntime(cudaGetDevice(&current), "coordinate upload current device");
#endif
    if (current != state_->positions.deviceOrdinal())
      throw std::invalid_argument("Device coordinate upload belongs to another current device");
    state_->positions.upload(positions.data(), positions.size());
    state_->live = positions.size() / stride;
    state_->ready = true;
  }

  /// Borrow a checked contiguous subpopulation without allocation or transfer.
  PackedDeviceCoordinatesView view(std::size_t begin, std::size_t count) const
  {
    std::lock_guard lock(state_->mutex);
    if (!state_->ready || count == 0 || begin > state_->live || count > state_->live - begin)
      throw std::invalid_argument("Device coordinate view is not a live source interval");
    return PackedDeviceCoordinatesView(state_, begin, count);
  }

  /// Account only this source's coordinate allocation, excluding shared metadata.
  std::size_t deviceWorkspaceBytes() const { return checkedDeviceBytes<double>(state_->positions.capacity()); }

private:
  friend class PreparedDeviceModel;
  friend class PreparedDeviceMetropolis;

  /// Bind trusted physical/layout metadata through the model's named preparation.
  PreparedDeviceCoordinates(std::shared_ptr<const PreparedDeviceModelMetadata> family,
                            std::size_t electrons, std::size_t capacity)
      : state_(std::make_shared<coordinate_detail::SourceState>(std::move(family), electrons, capacity))
  {}

  std::shared_ptr<coordinate_detail::SourceState> state_;
};
} // namespace qmcplusplus::psiformer::device
#endif
