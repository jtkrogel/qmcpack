//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceMetrics.h
 * @brief Per-device evidence for explicitly owned PsiFormer allocations/transfers.
 *
 * These counters exclude vendor-private BLAS/runtime storage and kernel memory
 * traffic. A snapshot/reset is a coherent measurement only at a quiescent point;
 * atomic fields prevent races but do not constitute a global stream barrier.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_METRICS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_METRICS_H

#include <atomic>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace qmcplusplus::psiformer::device
{

/** Describe explicit live/peak ownership and successful interval operations. */
struct DeviceMemorySnapshot
{
  std::size_t allocation_calls = 0, free_calls = 0, failed_frees = 0;
  std::size_t live_bytes = 0, peak_bytes = 0;
  std::size_t upload_calls = 0, uploaded_bytes = 0;
  std::size_t download_calls = 0, downloaded_bytes = 0;
  std::size_t device_copy_calls = 0, device_copied_bytes = 0;
};

/** Count buffer ownership independently of scalar type and execution crowd. */
class DeviceMemoryMetrics
{
public:
  /// Record a successful allocation before publishing the new buffer owner.
  void allocated(std::size_t bytes) noexcept
  {
    allocation_calls_.fetch_add(1, std::memory_order_relaxed);
    const auto live = live_bytes_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    auto peak = peak_bytes_.load(std::memory_order_relaxed);
    while (peak < live && !peak_bytes_.compare_exchange_weak(peak, live, std::memory_order_relaxed))
    {}
  }

  /// Remove ownership only after the vendor confirms a successful free.
  void freed(std::size_t bytes) noexcept
  {
    free_calls_.fetch_add(1, std::memory_order_relaxed);
    live_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
  }

  /// Preserve outstanding bytes when an unwinding destructor cannot free them.
  void failedFree() noexcept { failed_frees_.fetch_add(1, std::memory_order_relaxed); }

  /// Count a successful host upload, not the device's subsequent compute traffic.
  void uploaded(std::size_t bytes) noexcept
  {
    upload_calls_.fetch_add(1, std::memory_order_relaxed);
    uploaded_bytes_.fetch_add(bytes, std::memory_order_relaxed);
  }

  /// Count a successful device download into caller-owned host storage.
  void downloaded(std::size_t bytes) noexcept
  {
    download_calls_.fetch_add(1, std::memory_order_relaxed);
    downloaded_bytes_.fetch_add(bytes, std::memory_order_relaxed);
  }

  /// Count a completed explicit D2D transfer, excluding kernel memory traffic.
  void copied(std::size_t bytes) noexcept
  {
    device_copy_calls_.fetch_add(1, std::memory_order_relaxed);
    device_copied_bytes_.fetch_add(bytes, std::memory_order_relaxed);
  }

  /// Read scalar counters without modifying allocation or stream state.
  DeviceMemorySnapshot snapshot() const noexcept
  {
    return {allocation_calls_.load(std::memory_order_relaxed), free_calls_.load(std::memory_order_relaxed),
            failed_frees_.load(std::memory_order_relaxed), live_bytes_.load(std::memory_order_relaxed),
            peak_bytes_.load(std::memory_order_relaxed), upload_calls_.load(std::memory_order_relaxed),
            uploaded_bytes_.load(std::memory_order_relaxed), download_calls_.load(std::memory_order_relaxed),
            downloaded_bytes_.load(std::memory_order_relaxed), device_copy_calls_.load(std::memory_order_relaxed),
            device_copied_bytes_.load(std::memory_order_relaxed)};
  }

  /// Start a quiescent measurement interval without forgetting resident images.
  void resetInterval() noexcept
  {
    allocation_calls_.store(0, std::memory_order_relaxed);
    free_calls_.store(0, std::memory_order_relaxed);
    failed_frees_.store(0, std::memory_order_relaxed);
    upload_calls_.store(0, std::memory_order_relaxed);
    uploaded_bytes_.store(0, std::memory_order_relaxed);
    download_calls_.store(0, std::memory_order_relaxed);
    downloaded_bytes_.store(0, std::memory_order_relaxed);
    device_copy_calls_.store(0, std::memory_order_relaxed);
    device_copied_bytes_.store(0, std::memory_order_relaxed);
    peak_bytes_.store(live_bytes_.load(std::memory_order_relaxed), std::memory_order_relaxed);
  }

private:
  std::atomic<std::size_t> allocation_calls_{0}, free_calls_{0}, failed_frees_{0};
  std::atomic<std::size_t> live_bytes_{0}, peak_bytes_{0};
  std::atomic<std::size_t> upload_calls_{0}, uploaded_bytes_{0};
  std::atomic<std::size_t> download_calls_{0}, downloaded_bytes_{0};
  std::atomic<std::size_t> device_copy_calls_{0}, device_copied_bytes_{0};
};

/** Resolve an address-stable metrics owner during buffer preparation only.
 * Device ordinals are vendor-local; one executable uses one compiled vendor.
 * Borrowed pointers remain valid until process shutdown, never one per iteration.
 */
inline DeviceMemoryMetrics& deviceMemoryMetrics(int device_ordinal)
{
  if (device_ordinal < 0)
    throw std::invalid_argument("PsiFormer memory metrics require a nonnegative device ordinal");
  static std::mutex registry_mutex;
  static std::map<int, std::unique_ptr<DeviceMemoryMetrics>> registry;
  std::lock_guard lock(registry_mutex);
  auto& metrics = registry[device_ordinal];
  if (!metrics)
    metrics = std::make_unique<DeviceMemoryMetrics>();
  return *metrics;
}

} // namespace qmcplusplus::psiformer::device

#endif
