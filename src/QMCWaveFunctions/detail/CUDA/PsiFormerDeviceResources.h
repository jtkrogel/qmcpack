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
      : data_(std::exchange(other.data_, nullptr)), capacity_(std::exchange(other.capacity_, 0))
  {}

  /// Release storage without throwing during exception unwinding.
  ~DeviceBuffer()
  {
    if (data_)
    {
#ifdef QMC_CUDA2HIP
      (void)hipFree(data_);
#else
      (void)cudaFree(data_);
#endif
    }
  }

  /// Grow only during preparation; requests within capacity perform no runtime call.
  void reserve(std::size_t capacity)
  {
    if (capacity <= capacity_)
      return;
    const std::size_t bytes = checkedDeviceBytes<T>(capacity);
    void* replacement = nullptr;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMalloc(&replacement, bytes), "hipMalloc");
#else
    checkRuntime(cudaMalloc(&replacement, bytes), "cudaMalloc");
#endif
    // Publish new ownership only after allocation succeeds.
    T* previous = data_;
    data_       = static_cast<T*>(replacement);
    capacity_   = capacity;
    if (previous)
    {
#ifdef QMC_CUDA2HIP
      checkRuntime(hipFree(previous), "hipFree after buffer growth");
#else
      checkRuntime(cudaFree(previous), "cudaFree after buffer growth");
#endif
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
  }

  /// Copy a bounded device interval into caller-owned host storage.
  void download(T* destination, std::size_t count) const
  {
    validateTransfer(destination, count);
    if (count == 0)
      return;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemcpy(destination, data_, checkedDeviceBytes<T>(count), hipMemcpyDeviceToHost), "hipMemcpy D2H");
#else
    checkRuntime(cudaMemcpy(destination, data_, checkedDeviceBytes<T>(count), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
#endif
  }

  /// Expose the stable prepared pointer without allocation.
  T* data() noexcept { return data_; }

  /// Expose a read-only prepared pointer without allocation.
  const T* data() const noexcept { return data_; }

  /// Report the allocated element capacity, not a logical tensor shape.
  std::size_t capacity() const noexcept { return capacity_; }

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
};

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_RESOURCES_H
