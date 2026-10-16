//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceMetrics.h
 * @brief Per-device evidence and admission for native PsiFormer allocations.
 *
 * These counters exclude vendor-private BLAS/runtime storage and kernel memory
 * traffic. A snapshot/reset is a coherent measurement only at a quiescent point;
 * atomic fields prevent races but do not constitute a global stream barrier.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_METRICS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_METRICS_H

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

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

/** Rank-local native allocation admission, independently applied to each device.
 * The reserve charges untracked vendor/external usage explicitly; it is not a
 * measurement or enforcement of that usage. HOST allocations are excluded.
 */
struct DeviceMemoryAdmission
{
  std::optional<std::size_t> limit_bytes;
  std::size_t external_reserve_bytes = 0;
};

/// Parse decimal bytes without consulting the runtime (zero is allowed for reserve).
inline std::size_t parseDeviceMemoryAdmissionBytes(const char* value)
{
  if (!value || !*value)
    throw std::invalid_argument("PsiFormer aggregate device admission requires decimal bytes");
  std::size_t bytes = 0;
  for (const char* digit = value; *digit; ++digit)
  {
    if (*digit < '0' || *digit > '9' ||
        bytes > (std::numeric_limits<std::size_t>::max() - (*digit - '0')) / 10)
      throw std::invalid_argument("PsiFormer aggregate device admission bytes are invalid or overflow");
    bytes = 10 * bytes + (*digit - '0');
  }
  return bytes;
}

/// Freeze one process policy; different devices receive independent copies.
inline const DeviceMemoryAdmission& configuredDeviceMemoryAdmission()
{
  static const auto policy = [] {
    DeviceMemoryAdmission result;
    const char* limit = std::getenv("PSIFORMER_GPU_AGGREGATE_DEVICE_BYTES");
    const char* reserve = std::getenv("PSIFORMER_GPU_EXTERNAL_RESERVE_BYTES");
    if (reserve && !limit)
      throw std::invalid_argument("PSIFORMER_GPU_EXTERNAL_RESERVE_BYTES requires PSIFORMER_GPU_AGGREGATE_DEVICE_BYTES");
    if (limit)
      result.limit_bytes = parseDeviceMemoryAdmissionBytes(limit);
    if (reserve)
      result.external_reserve_bytes = parseDeviceMemoryAdmissionBytes(reserve);
    return result;
  }();
  return policy;
}

/** Count buffer ownership independently of scalar type and execution crowd. */
class DeviceMemoryMetrics
{
public:
  /** Hold capacity before malloc; destruction rolls back a failed preparation.
   * A committed allocation stays charged until a confirmed successful free.
   * This reservation is capacity ownership, not an asynchronous execution lease.
   */
  class AllocationReservation
  {
  public:
    AllocationReservation(const AllocationReservation&) = delete;
    AllocationReservation& operator=(const AllocationReservation&) = delete;
    AllocationReservation& operator=(AllocationReservation&&) = delete;
    AllocationReservation(AllocationReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), bytes_(other.bytes_) {}
    ~AllocationReservation()
    {
      if (owner_)
        owner_->cancelReservation(bytes_);
    }
    void commit() noexcept
    {
      if (owner_)
      {
        owner_->commitReservation(bytes_);
        owner_ = nullptr;
      }
    }

  private:
    friend class DeviceMemoryMetrics;
    AllocationReservation(DeviceMemoryMetrics& owner, std::size_t bytes) : owner_(&owner), bytes_(bytes) {}
    DeviceMemoryMetrics* owner_;
    std::size_t bytes_;
  };

  /// Configure once, before allocation; identical requests are harmless.
  void configureAdmission(DeviceMemoryAdmission policy)
  {
    std::lock_guard lock(admission_mutex_);
    if ((!policy.limit_bytes && policy.external_reserve_bytes) ||
        (policy.limit_bytes && (!*policy.limit_bytes || *policy.limit_bytes == std::numeric_limits<std::size_t>::max() ||
                               policy.external_reserve_bytes > *policy.limit_bytes)))
      throw std::invalid_argument("PsiFormer aggregate device limit must be positive finite bytes and cover its reserve");
    if (admission_configured_)
    {
      if (admission_.limit_bytes != policy.limit_bytes ||
          admission_.external_reserve_bytes != policy.external_reserve_bytes)
        throw std::logic_error("PsiFormer aggregate device admission cannot be reconfigured per component");
      return;
    }
    if (allocation_calls_.load(std::memory_order_relaxed) || live_bytes_.load(std::memory_order_relaxed) || pending_bytes_)
      throw std::logic_error("PsiFormer aggregate device admission must be configured before allocation");
    admission_ = policy;
    admission_configured_ = true;
  }

  /// Admit all live native owners plus pending requests before the vendor call.
  AllocationReservation reserveAllocation(std::size_t bytes)
  {
    std::lock_guard lock(admission_mutex_);
    const auto live = live_bytes_.load(std::memory_order_relaxed);
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (pending_bytes_ > maximum - live || bytes > maximum - live - pending_bytes_)
      throw std::overflow_error("PsiFormer aggregate native device ownership overflows");
    const auto charged = live + pending_bytes_ + bytes;
    if (admission_.limit_bytes && charged > *admission_.limit_bytes - admission_.external_reserve_bytes)
      throw std::length_error("PsiFormer aggregate native device admission rejected allocation before malloc: live_bytes=" +
          std::to_string(live) + ", pending_bytes=" + std::to_string(pending_bytes_) +
          ", request_bytes=" + std::to_string(bytes) + ", limit_bytes=" + std::to_string(*admission_.limit_bytes) +
          ", external_reserve_bytes=" + std::to_string(admission_.external_reserve_bytes));
    pending_bytes_ += bytes;
    return AllocationReservation(*this, bytes);
  }

  /// Report this rank/device's covered capacity and the deliberate exclusions.
  void reportAdmission(std::ostream& out, int device_ordinal) const
  {
    std::lock_guard lock(admission_mutex_);
    if (!admission_.limit_bytes)
      return;
    out << "  PsiFormer aggregate native device admission: vendor_local_device=" << device_ordinal
        << ", scope=one_process_one_device, limit_bytes=" << *admission_.limit_bytes
        << ", external_reserve_bytes=" << admission_.external_reserve_bytes
        << ", live_bytes=" << live_bytes_.load(std::memory_order_relaxed) << ", pending_bytes=" << pending_bytes_
        << "; covers DeviceBuffer images, scratch, retained capacity and allocation overlap"
        << "; excludes HOST, other ranks, unwrapped QMCPACK/offload, vendor/runtime and external allocations"
        << "; aggregate_process_fit=no\n";
  }

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
  void cancelReservation(std::size_t bytes) noexcept
  {
    std::lock_guard lock(admission_mutex_);
    pending_bytes_ -= bytes;
  }
  void commitReservation(std::size_t bytes) noexcept
  {
    std::lock_guard lock(admission_mutex_);
    allocated(bytes);
    pending_bytes_ -= bytes;
  }
  mutable std::mutex admission_mutex_;
  DeviceMemoryAdmission admission_;
  bool admission_configured_ = false;
  std::size_t pending_bytes_ = 0;
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
  {
    auto candidate = std::make_unique<DeviceMemoryMetrics>();
    candidate->configureAdmission(configuredDeviceMemoryAdmission());
    metrics = std::move(candidate);
  }
  return *metrics;
}

} // namespace qmcplusplus::psiformer::device

#endif
