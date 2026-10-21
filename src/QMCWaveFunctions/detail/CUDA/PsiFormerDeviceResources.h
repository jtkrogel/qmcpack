//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceResources.h
 * @brief Checked CUDA/HIP buffer ownership for prepared PsiFormer workspaces.
 *
 * Allocation is explicit preparation work, never hidden inside a kernel wrapper.
 * The caller keeps the owning device current and synchronizes outstanding users
 * before growing or destroying storage. This class does not publish model versions,
 * own streams, or establish end-to-end DEVICE capability.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_RESOURCES_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_RESOURCES_H

#include "PsiFormerDeviceRuntime.h"
#include "PsiFormerDeviceMetrics.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace qmcplusplus::psiformer::device
{

/// Translate vendor runtime failures into an operation-specific host exception.
inline void checkRuntime(Error error, const char* operation)
{
  if (error == success)
    return;
#ifdef QMC_CUDA2HIP
  const char* description = hipGetErrorString(error);
#else
  const char* description = cudaGetErrorString(error);
#endif
  throw std::runtime_error(std::string(operation) + ": " + description);
}

/// Resolve the process-frozen admission before initial image upload on this device.
inline void reportCurrentDeviceMemoryAdmission(std::ostream& out)
{
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "hipGetDevice for aggregate admission");
#else
  checkRuntime(cudaGetDevice(&ordinal), "cudaGetDevice for aggregate admission");
#endif
  deviceMemoryMetrics(ordinal).reportAdmission(out, ordinal);
}

/// Observe completed native ownership at an opt-in driver boundary, without synchronization.
inline void reportCurrentDeviceMemorySnapshot(std::ostream& out, int communicator_rank,
                                              std::string_view boundary)
{
  if (!configuredDeviceMemorySummaryEnabled())
    return;
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "hipGetDevice for native memory summary");
#else
  checkRuntime(cudaGetDevice(&ordinal), "cudaGetDevice for native memory summary");
#endif
  deviceMemoryMetrics(ordinal).reportSnapshot(out, ordinal, communicator_rank, boundary);
}

/// Reject byte-count overflow before any allocation or transfer is submitted.
template<class T>
inline std::size_t checkedDeviceBytes(std::size_t count)
{
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    throw std::overflow_error("PsiFormer device extent overflow");
  return count * sizeof(T);
}

/** Own one reusable typed device allocation with explicit capacity.
 *
 * Growth discards old contents; failed allocation leaves old storage intact.
 * Host transfers are synchronous validation/preparation operations, not a claimed
 * asynchronous production pipeline. Only trivially copyable elements are accepted.
 */
template<class T>
class DeviceBuffer
{
  static_assert(std::is_trivially_copyable_v<T>);

public:
  explicit DeviceBuffer(std::size_t capacity = 0) { reserve(capacity); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(DeviceBuffer&&) = delete;

  /// Transfer unique ownership without allocating or consulting the runtime.
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), capacity_(std::exchange(other.capacity_, 0)),
        device_ordinal_(std::exchange(other.device_ordinal_, -1)), metrics_(std::exchange(other.metrics_, nullptr))
  {}

  /// Release storage without throwing during exception unwinding.
  ~DeviceBuffer()
  {
    if (data_)
    {
#ifdef QMC_CUDA2HIP
      const auto result = hipFree(data_);
#else
      const auto result = cudaFree(data_);
#endif
      if (result == success)
        metrics_->freed(capacity_ * sizeof(T));
      else
        metrics_->failedFree();
    }
  }

  /// Grow only during preparation; requests within capacity perform no runtime call.
  void reserve(std::size_t capacity)
  {
    if (capacity <= capacity_)
      return;
    const std::size_t bytes = checkedDeviceBytes<T>(capacity);
    int device_ordinal = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&device_ordinal), "hipGetDevice for buffer preparation");
#else
    checkRuntime(cudaGetDevice(&device_ordinal), "cudaGetDevice for buffer preparation");
#endif
    if (data_ && device_ordinal != device_ordinal_)
      throw std::logic_error("PsiFormer device buffer cannot grow on a different owning device");
    auto& metrics = deviceMemoryMetrics(device_ordinal);
    // Charge old and replacement capacities together. Failed malloc rolls back
    // the reservation; a failed old-buffer free keeps both allocations charged.
    auto reservation = metrics.reserveAllocation(bytes);
    void* replacement = nullptr;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMalloc(&replacement, bytes), "hipMalloc");
#else
    checkRuntime(cudaMalloc(&replacement, bytes), "cudaMalloc");
#endif
    reservation.commit();
    // Publish new ownership only after allocation succeeds.
    T* previous = data_;
    const auto previous_bytes = capacity_ * sizeof(T);
    data_       = static_cast<T*>(replacement);
    capacity_   = capacity;
    device_ordinal_ = device_ordinal;
    metrics_ = &metrics;
    if (previous)
    {
#ifdef QMC_CUDA2HIP
      const auto result = hipFree(previous);
#else
      const auto result = cudaFree(previous);
#endif
      if (result == success)
        metrics.freed(previous_bytes);
      else
      {
        metrics.failedFree();
        checkRuntime(result, "free previous PsiFormer device buffer after growth");
      }
    }
  }

  /// Copy a bounded host interval into already prepared storage.
  void upload(const T* source, std::size_t count)
  {
    validateTransfer(source, count);
    if (count == 0)
      return;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemcpy(data_, source, checkedDeviceBytes<T>(count), hipMemcpyHostToDevice), "hipMemcpy H2D");
#else
    checkRuntime(cudaMemcpy(data_, source, checkedDeviceBytes<T>(count), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
#endif
    metrics_->uploaded(checkedDeviceBytes<T>(count));
  }

  /// Copy a bounded device interval into caller-owned host storage.
  void download(T* destination, std::size_t count, std::size_t source_offset = 0) const
  {
    if (source_offset > capacity_ || count > capacity_ - source_offset)
      throw std::length_error("PsiFormer device download interval exceeds capacity");
    validateTransfer(destination, count);
    if (count == 0)
      return;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemcpy(destination, data_ + source_offset, checkedDeviceBytes<T>(count), hipMemcpyDeviceToHost), "hipMemcpy D2H");
#else
    checkRuntime(cudaMemcpy(destination, data_ + source_offset, checkedDeviceBytes<T>(count), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
#endif
    metrics_->downloaded(checkedDeviceBytes<T>(count));
  }

  /// Copy admitted device storage without crossing a HOST mirror boundary.
  void copyFromDevice(const DeviceBuffer& source, std::size_t count)
  {
    validateTransfer(source.data(), count);
    if (count > source.capacity_ || (count && device_ordinal_ != source.device_ordinal_))
      throw std::invalid_argument("PsiFormer device copy extent or owning device mismatch");
    if (!count)
      return;
    int current = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&current), "Device copy current device");
#else
    checkRuntime(cudaGetDevice(&current), "Device copy current device");
#endif
    if (current != device_ordinal_)
      throw std::invalid_argument("PsiFormer device copy used a foreign current device");
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemcpy(data_, source.data(), checkedDeviceBytes<T>(count), hipMemcpyDeviceToDevice), "Device copy");
#else
    checkRuntime(cudaMemcpy(data_, source.data(), checkedDeviceBytes<T>(count), cudaMemcpyDeviceToDevice), "Device copy");
#endif
    metrics_->copied(checkedDeviceBytes<T>(count));
  }

  /// Preserve synchronous host lifetime while ordering only the caller's queue.
  void uploadOnStream(const T* source, std::size_t count, Stream stream)
  {
    validateTransfer(source, count);
    if (!count)
      return;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemcpyAsync(data_, source, checkedDeviceBytes<T>(count), hipMemcpyHostToDevice, stream), "hipMemcpyAsync H2D");
    checkRuntime(hipStreamSynchronize(stream), "hipStreamSynchronize H2D");
#else
    checkRuntime(cudaMemcpyAsync(data_, source, checkedDeviceBytes<T>(count), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D");
    checkRuntime(cudaStreamSynchronize(stream), "cudaStreamSynchronize H2D");
#endif
    metrics_->uploaded(checkedDeviceBytes<T>(count));
  }

  /// Download from an independent inference queue without default-stream fences.
  void downloadOnStream(T* destination, std::size_t count, Stream stream, std::size_t source_offset = 0) const
  {
    if (source_offset > capacity_ || count > capacity_ - source_offset)
      throw std::length_error("PsiFormer device download interval exceeds capacity");
    validateTransfer(destination, count);
    if (!count)
      return;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemcpyAsync(destination, data_ + source_offset, checkedDeviceBytes<T>(count), hipMemcpyDeviceToHost, stream), "hipMemcpyAsync D2H");
    checkRuntime(hipStreamSynchronize(stream), "hipStreamSynchronize D2H");
#else
    checkRuntime(cudaMemcpyAsync(destination, data_ + source_offset, checkedDeviceBytes<T>(count), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H");
    checkRuntime(cudaStreamSynchronize(stream), "cudaStreamSynchronize D2H");
#endif
    metrics_->downloaded(checkedDeviceBytes<T>(count));
  }

  /// Expose the stable prepared pointer without allocation.
  T* data() noexcept { return data_; }

  /// Expose a read-only prepared pointer without allocation.
  const T* data() const noexcept { return data_; }

  /// Report the allocated element capacity, not a logical tensor shape.
  std::size_t capacity() const noexcept { return capacity_; }

  /// Identify the preparation-time owner without another runtime query.
  int deviceOrdinal() const noexcept { return device_ordinal_; }

private:
  /// Check destination bounds and null host pointers before any vendor call.
  void validateTransfer(const void* pointer, std::size_t count) const
  {
    if (count > capacity_)
      throw std::length_error("PsiFormer device transfer exceeds capacity");
    if (count != 0 && pointer == nullptr)
      throw std::invalid_argument("PsiFormer device transfer has null host storage");
  }

  T* data_              = nullptr;
  std::size_t capacity_ = 0;
  int device_ordinal_ = -1;
  DeviceMemoryMetrics* metrics_ = nullptr;
};

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_RESOURCES_H
