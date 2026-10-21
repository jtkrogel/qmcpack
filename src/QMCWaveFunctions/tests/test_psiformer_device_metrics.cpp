//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include "catch2/catch_test_macros.hpp"
#include "config.h"
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceMetrics.h"
#ifdef ENABLE_CUDA
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceResources.h"
#endif

#include <condition_variable>
#include <array>
#include <sstream>
#include <thread>
#include <vector>

using namespace qmcplusplus::psiformer::device;

/// Reporting observes the existing interval and never changes ownership or counters.
TEST_CASE("PsiFormer native memory snapshot formatting", "[psiformer][accelerator][memory_summary]")
{
  DeviceMemoryMetrics metrics;
  metrics.allocated(16);
  metrics.allocated(32);
  metrics.freed(16);
  metrics.failedFree();
  metrics.uploaded(24);
  metrics.downloaded(8);
  metrics.copied(12);
  std::ostringstream out;
  metrics.reportSnapshot(out, 3, 7, "post_qmc_section");
  const auto text = out.str();
  for (const auto field : {"boundary=post_qmc_section", "communicator_rank=7", "vendor_local_device=3",
                          "scope=one_process_one_device", "counters=since_last_explicit_reset",
                          "allocation_calls=2", "free_calls=1", "failed_frees=1", "tracked_live_bytes=32",
                          "tracked_peak_bytes=48", "upload_calls=1", "uploaded_bytes=24", "download_calls=1",
                          "downloaded_bytes=8", "device_copy_calls=1", "device_copied_bytes=12",
                          "excludes HOST", "runtime_HBM_peak=no", "reset_performed=no"})
    CHECK(text.find(field) != std::string::npos);
  const auto unchanged = metrics.snapshot();
  CHECK(unchanged.live_bytes == 32);
  CHECK(unchanged.peak_bytes == 48);
  CHECK(unchanged.allocation_calls == 2);
  CHECK(unchanged.uploaded_bytes == 24);
  metrics.allocated(8);
  std::ostringstream second;
  metrics.reportSnapshot(second, 3, 7, "next_section");
  CHECK(second.str().find("allocation_calls=3") != std::string::npos);
  CHECK(second.str().find("tracked_peak_bytes=48") != std::string::npos);
}

/// Only the exact opt-in enables summaries; checking the policy needs no GPU.
TEST_CASE("PsiFormer memory summary opt in", "[psiformer][accelerator][memory_summary]")
{
  const char* previous = std::getenv("PSIFORMER_GPU_MEMORY_SUMMARY");
  const std::optional<std::string> saved = previous ? std::optional<std::string>(previous) : std::nullopt;
  // Keep the caller's profiling policy unchanged after this environment-only test.
  struct Restore
  {
    const std::optional<std::string>& saved;
    // Restore even when a test assertion exits this scope early.
    ~Restore()
    {
      if (saved)
      {
        setenv("PSIFORMER_GPU_MEMORY_SUMMARY", saved->c_str(), 1);
      }
      else
      {
        unsetenv("PSIFORMER_GPU_MEMORY_SUMMARY");
      }
    }
  } restore{saved};
  unsetenv("PSIFORMER_GPU_MEMORY_SUMMARY");
  CHECK_FALSE(configuredDeviceMemorySummaryEnabled());
  for (const char* value : {"", "0", "true", "01", "1x"})
  {
    setenv("PSIFORMER_GPU_MEMORY_SUMMARY", value, 1);
    CHECK_FALSE(configuredDeviceMemorySummaryEnabled());
  }
  setenv("PSIFORMER_GPU_MEMORY_SUMMARY", "1", 1);
  CHECK(configuredDeviceMemorySummaryEnabled());
}

/// Constrain ownership/interval semantics without a GPU or vendor runtime.
TEST_CASE("PsiFormer explicit device memory accounting", "[psiformer][accelerator]")
{
  DeviceMemoryMetrics metrics;
  metrics.allocated(16);
  metrics.allocated(32);
  metrics.freed(16);
  metrics.uploaded(24);
  metrics.downloaded(8);
  auto state = metrics.snapshot();
  CHECK(state.allocation_calls == 2);
  CHECK(state.free_calls == 1);
  CHECK(state.live_bytes == 32);
  CHECK(state.peak_bytes == 48);
  CHECK(state.upload_calls == 1);
  CHECK(state.uploaded_bytes == 24);
  CHECK(state.download_calls == 1);
  CHECK(state.downloaded_bytes == 8);

  // Starting a timing interval must retain the already resident model footprint.
  metrics.resetInterval();
  state = metrics.snapshot();
  CHECK(state.allocation_calls == 0);
  CHECK(state.free_calls == 0);
  CHECK(state.live_bytes == 32);
  CHECK(state.peak_bytes == 32);
  CHECK(state.upload_calls == 0);
  CHECK(state.download_calls == 0);
  metrics.failedFree();
  CHECK(metrics.snapshot().failed_frees == 1);
  CHECK(metrics.snapshot().live_bytes == 32);
  metrics.freed(32);
  CHECK(metrics.snapshot().live_bytes == 0);
}

/// Keep stable identities per vendor-local device without consulting hardware.
TEST_CASE("PsiFormer memory counters separate device ordinals", "[psiformer][accelerator]")
{
  auto& first = deviceMemoryMetrics(0);
  auto& second = deviceMemoryMetrics(1);
  CHECK(&first == &deviceMemoryMetrics(0));
  CHECK(&first != &second);
  CHECK_THROWS_AS(deviceMemoryMetrics(-1), std::invalid_argument);
}

TEST_CASE("PsiFormer aggregate device admission rolls back before allocation", "[psiformer][accelerator]")
{
  DeviceMemoryMetrics metrics;
  metrics.configureAdmission({100, 20});
  metrics.configureAdmission({100, 20});
  CHECK_THROWS_AS(metrics.configureAdmission({101, 20}), std::logic_error);
  {
    auto pending = metrics.reserveAllocation(80);
    CHECK_THROWS_AS(metrics.reserveAllocation(1), std::length_error);
    CHECK(metrics.snapshot().allocation_calls == 0);
    // An exceptional malloc leaves no live allocation and releases its reservation.
  }
  auto exact = metrics.reserveAllocation(80);
  exact.commit();
  CHECK(metrics.snapshot().live_bytes == 80);
  CHECK_THROWS_AS(metrics.reserveAllocation(1), std::length_error);
  metrics.resetInterval();
  CHECK_THROWS_AS(metrics.reserveAllocation(1), std::length_error);
  metrics.freed(80);
  auto reused = metrics.reserveAllocation(80);
  reused.commit();
  CHECK(metrics.snapshot().peak_bytes == 80);
  metrics.freed(80);
}

TEST_CASE("PsiFormer aggregate device admission retains replacement and failed-free capacity",
          "[psiformer][accelerator]")
{
  DeviceMemoryMetrics metrics;
  metrics.configureAdmission({100, 0});
  metrics.reserveAllocation(60).commit();
  CHECK_THROWS_AS(metrics.reserveAllocation(80), std::length_error);
  CHECK(metrics.snapshot().live_bytes == 60);
  metrics.reserveAllocation(40).commit();
  metrics.failedFree();
  CHECK(metrics.snapshot().live_bytes == 100);
  CHECK(metrics.snapshot().peak_bytes == 100);
  CHECK_THROWS_AS(metrics.reserveAllocation(1), std::length_error);
  metrics.freed(60);
  metrics.freed(40);
  CHECK(metrics.snapshot().live_bytes == 0);
}

TEST_CASE("PsiFormer aggregate device admission follows shared generation lifetimes",
          "[psiformer][accelerator]")
{
  struct Image
  {
    DeviceMemoryMetrics& metrics;
    explicit Image(DeviceMemoryMetrics& owner) : metrics(owner) { metrics.reserveAllocation(40).commit(); }
    ~Image() { metrics.freed(40); }
  };
  DeviceMemoryMetrics metrics;
  metrics.configureAdmission({80, 0});
  auto active = std::make_shared<Image>(metrics);
  auto crowd = active;
  auto second_crowd = active;
  CHECK(metrics.snapshot().live_bytes == 40);
  active = std::make_shared<Image>(metrics);
  CHECK(metrics.snapshot().live_bytes == 80);
  CHECK_THROWS_AS(std::make_shared<Image>(metrics), std::length_error);
  crowd.reset();
  CHECK_THROWS_AS(std::make_shared<Image>(metrics), std::length_error);
  second_crowd.reset();
  CHECK(metrics.snapshot().live_bytes == 40);
  active = std::make_shared<Image>(metrics);
  CHECK(metrics.snapshot().live_bytes == 40);
  CHECK(metrics.snapshot().peak_bytes == 80);
  active.reset();
  CHECK(metrics.snapshot().live_bytes == 0);
}

TEST_CASE("PsiFormer aggregate device reservations cannot race through the cap",
          "[psiformer][accelerator]")
{
  DeviceMemoryMetrics metrics;
  metrics.configureAdmission({32, 0});
  std::mutex mutex;
  std::condition_variable condition;
  std::size_t arrived = 0, admitted = 0;
  bool release = false;
  std::vector<std::thread> workers;
  const auto rendezvous = [&](bool success) {
    std::unique_lock lock(mutex);
    admitted += success;
    ++arrived;
    condition.notify_all();
    condition.wait(lock, [&] { return release; });
  };
  for (int i = 0; i < 8; ++i)
    workers.emplace_back([&] {
      try
      {
        auto pending = metrics.reserveAllocation(8);
        rendezvous(true);
      }
      catch (const std::length_error&)
      {
        rendezvous(false);
      }
    });
  {
    std::unique_lock lock(mutex);
    condition.wait(lock, [&] { return arrived == workers.size(); });
    CHECK(admitted == 4);
    release = true;
    condition.notify_all();
  }
  for (auto& worker : workers)
    worker.join();
  metrics.reserveAllocation(32).commit();
  CHECK(metrics.snapshot().live_bytes == 32);
  metrics.freed(32);
}

TEST_CASE("PsiFormer aggregate device admission validates scope and arithmetic",
          "[psiformer][accelerator]")
{
  DeviceMemoryMetrics first, second, invalid;
  CHECK_THROWS_AS(invalid.configureAdmission({0, 0}), std::invalid_argument);
  CHECK_THROWS_AS(invalid.configureAdmission({std::numeric_limits<std::size_t>::max(), 0}), std::invalid_argument);
  CHECK_THROWS_AS(invalid.configureAdmission({100, 101}), std::invalid_argument);
  CHECK_THROWS_AS(invalid.configureAdmission({std::nullopt, 1}), std::invalid_argument);
  first.configureAdmission({20, 1});
  second.configureAdmission({20, 1});
  first.reserveAllocation(19).commit();
  second.reserveAllocation(19).commit();
  CHECK_THROWS_AS(first.reserveAllocation(1), std::length_error);
  CHECK(second.snapshot().live_bytes == 19);
  first.freed(19);
  second.freed(19);

  DeviceMemoryMetrics late;
  late.reserveAllocation(16).commit();
  late.resetInterval();
  CHECK_THROWS_AS(late.configureAdmission({100, 0}), std::logic_error);
  late.freed(16);
  late.resetInterval();
  CHECK_THROWS_AS(late.configureAdmission({100, 0}), std::logic_error);

  DeviceMemoryMetrics unbounded;
  auto pending = unbounded.reserveAllocation(std::numeric_limits<std::size_t>::max());
  CHECK_THROWS_AS(unbounded.reserveAllocation(1), std::overflow_error);
  CHECK_THROWS_AS(unbounded.configureAdmission({100, 0}), std::logic_error);
  auto moved = std::move(pending);
  moved.commit();
  moved.commit();
  CHECK(unbounded.snapshot().live_bytes == std::numeric_limits<std::size_t>::max());
  CHECK_THROWS_AS(unbounded.reserveAllocation(1), std::overflow_error);
  unbounded.freed(std::numeric_limits<std::size_t>::max());

  std::ostringstream report;
  first.reportAdmission(report, 7);
  CHECK(report.str().find("vendor_local_device=7") != std::string::npos);
  CHECK(report.str().find("aggregate_process_fit=no") != std::string::npos);
  CHECK(report.str().find("excludes HOST") != std::string::npos);
  std::ostringstream disabled;
  unbounded.reportAdmission(disabled, 0);
  CHECK(disabled.str().empty());
}

TEST_CASE("PsiFormer aggregate device admission decimal parsing", "[psiformer][accelerator]")
{
  CHECK(parseDeviceMemoryAdmissionBytes("0") == 0);
  CHECK(parseDeviceMemoryAdmissionBytes("4096") == 4096);
  CHECK_THROWS_AS(parseDeviceMemoryAdmissionBytes(nullptr), std::invalid_argument);
  for (const auto* value : {"", "-1", "+1", "1.0", " 1", "1MiB", "99999999999999999999999999999999999"})
    CHECK_THROWS_AS(parseDeviceMemoryAdmissionBytes(value), std::invalid_argument);
}

#ifdef ENABLE_CUDA
/// Run alone with aggregate limit=128/reserve=0 so no unrelated test owns buffers.
TEST_CASE("PsiFormer aggregate device admission rejects before native malloc",
          "[psiformer][aggregate_device_runtime]")
{
  const auto policy = configuredDeviceMemoryAdmission();
  if (policy.limit_bytes != std::optional<std::size_t>(128) || policy.external_reserve_bytes)
  {
    WARN("NOT_RUN: isolated runtime admission test requires PSIFORMER_GPU_AGGREGATE_DEVICE_BYTES=128 and zero reserve");
    return;
  }
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  REQUIRE(hipGetDeviceCount(&devices) == hipSuccess);
  REQUIRE(devices > 0);
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  REQUIRE(cudaGetDeviceCount(&devices) == cudaSuccess);
  REQUIRE(devices > 0);
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  REQUIRE(metrics.snapshot().live_bytes == 0);
  metrics.resetInterval();
  {
    DeviceBuffer<double> first(8);
    const auto* original = first.data();
    first.reserve(8);
    CHECK(metrics.snapshot().allocation_calls == 1);
    // A 128-byte replacement overlaps the original 64-byte allocation.
    CHECK_THROWS_AS(first.reserve(16), std::length_error);
    CHECK(first.data() == original);
    CHECK(first.capacity() == 8);
    CHECK(metrics.snapshot().allocation_calls == 1);
    {
      DeviceBuffer<double> second(8);
      CHECK(metrics.snapshot().live_bytes == 128);
      CHECK_THROWS_AS(DeviceBuffer<double>(1), std::length_error);
      CHECK(metrics.snapshot().allocation_calls == 2);
    }
    CHECK(metrics.snapshot().live_bytes == 64);
    const std::array<double, 8> input{1, 2, 3, 4, 5, 6, 7, 8};
    std::array<double, 8> output{};
    first.upload(input.data(), input.size());
    first.download(output.data(), output.size());
    CHECK(output == input);
  }
  CHECK(metrics.snapshot().live_bytes == 0);
  CHECK(metrics.snapshot().allocation_calls == 2);
  CHECK(metrics.snapshot().free_calls == 2);
  CHECK(metrics.snapshot().peak_bytes == 128);
}
#endif
