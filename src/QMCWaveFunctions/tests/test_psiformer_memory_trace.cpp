//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
// A tiny deterministic CPU runtime exercises the actual DeviceBuffer ownership code.
#ifdef QMC_CUDA2HIP
#undef QMC_CUDA2HIP // This test translation unit always uses the CPU CUDA-shaped adapter.
#endif
#define QMCPLUSPLUS_PSIFORMER_DEVICE_RUNTIME_H
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <vector>
#include <thread>

namespace qmcplusplus::psiformer::device
{
using Error = int;
using Stream = int;
inline constexpr Error success = 0;
}
namespace
{
bool fail_malloc = false, fail_free = false;
std::vector<void*> leaked;
int cudaGetDevice(int* device)
{
  *device = 0;
  return 0;
}

const char* cudaGetErrorString(int) { return "injected runtime failure"; }

int cudaMalloc(void** p, std::size_t bytes)
{
  if (fail_malloc)
    return 1;
  *p = std::malloc(bytes);
  return *p ? 0 : 1;
}

int cudaFree(void* p)
{
  if (fail_free)
  {
    leaked.push_back(p);
    return 1;
  }
  std::free(p);
  return 0;
}

enum { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };

int cudaMemcpy(void* to, const void* from, std::size_t bytes, int)
{
  std::memcpy(to, from, bytes);
  return 0;
}

[[maybe_unused]] int cudaMemcpyAsync(void* to, const void* from, std::size_t bytes, int kind, int)
{
  return cudaMemcpy(to, from, bytes, kind);
}

int cudaStreamSynchronize(int) { return 0; }

void require(bool value)
{
  if (!value)
    throw std::runtime_error("memory trace test failed");
}

template<class F> void throws(F&& f)
{
  bool caught = false;
  try
  {
    f();
  }
  catch (const std::exception&)
  {
    caught = true;
  }
  require(caught);
}
}
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceResources.h"

// Constructed before the sink; optional late destruction tests process-lifetime safety.
std::unique_ptr<qmcplusplus::psiformer::device::DeviceBuffer<double>> late_buffer;

struct TracedHostPayload
{
  qmcplusplus::psiformer::memory::HostAllocation trace;
  std::vector<double> values;

  explicit TracedHostPayload(std::size_t count)
  {
    trace.requestAt(__FILE__, __LINE__, "test::host_cycle", count * sizeof(double), "float64",
                    "[8]", reinterpret_cast<std::uintptr_t>(this));
    values.resize(count);
    trace.allocationSucceeded(values.capacity() * sizeof(double));
  }

  void releaseStorageEarly()
  {
    std::vector<double>().swap(values);
    trace.storageReleased();
  }
};

int main()
{
  namespace mem = qmcplusplus::psiformer::memory;
  using namespace qmcplusplus::psiformer::device;
  if (std::getenv("PSIFORMER_MEMORY_TEST_IO_FAILURE"))
  {
    const std::string path = std::string(std::getenv("PSIFORMER_MEMORY_TRACE")) +
        ".rankunknown.pid" + std::to_string(getpid()) + ".jsonl";
    require(symlink("/dev/full", path.c_str()) == 0);
  }
  setenv("PSIFORMER_GPU_AGGREGATE_DEVICE_BYTES", "512", 1);
  unsetenv("PSIFORMER_GPU_EXTERNAL_RESERVE_BYTES");
  auto& metrics = deviceMemoryMetrics(0);
  {
    mem::Phase outer("outer");
    DeviceBuffer<double> first(8, "test::first", __FILE__, __LINE__);
    const auto id = first.memoryTraceIdentity();
    DeviceBuffer<double> moved(std::move(first));
    require(moved.memoryTraceIdentity() == id);
    require(first.memoryTraceIdentity() == 0);
    {
      mem::Phase inner("inner");
      moved.reserveNamed(16, "test::first", __FILE__, __LINE__);
      moved.traceView("test::alias", 0, 8, "8");
      require(metrics.snapshot().peak_bytes == 192);
      throws([&] { moved.reserveNamed(64, "test::rejected", __FILE__, __LINE__); });
      require(metrics.snapshot().live_bytes == 128);
      fail_malloc = true;
      throws([&] { moved.reserveNamed(32, "test::failed_malloc", __FILE__, __LINE__); });
      fail_malloc = false;
      fail_free = true;
      throws([&] { moved.reserveNamed(32, "test::failed_free", __FILE__, __LINE__); });
      fail_free = false;
      require(metrics.snapshot().live_bytes == 384);
      require(metrics.snapshot().failed_frees == 1);
    }
  }
  require(metrics.snapshot().live_bytes == 128);
  {
    auto image = std::make_shared<DeviceBuffer<double>>(8, "test::shared", __FILE__, __LINE__);
    auto alias = image;
    const auto before = metrics.snapshot().live_bytes;
    image.reset();
    require(metrics.snapshot().live_bytes == before);
    alias.reset();
    require(metrics.snapshot().live_bytes == 128);
  }
  throws([] { checkedDeviceBytes<double>(std::numeric_limits<std::size_t>::max()); });
  {
    mem::HostCapacity capacity;
    capacity.snapshot("host::arena", 64);
    mem::HostCapacity moved(std::move(capacity));
    moved.snapshot("host::arena", 128);
    mem::HostCapacity replacement;
    replacement.snapshot("host::replacement", 32);
    replacement = std::move(moved);
  }
  {
    auto old_payload = std::make_unique<TracedHostPayload>(8);
    auto replacement = std::make_unique<TracedHostPayload>(8);
    replacement->releaseStorageEarly(); // Explicit capacity release precedes owner destruction.
    replacement.reset();
    old_payload.reset();
    auto recreated = std::make_unique<TracedHostPayload>(8); // Same-size later lifetime is visible as churn.
    recreated->releaseStorageEarly();
  }
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i)
    threads.emplace_back([] {
      mem::Phase phase("concurrent");
      DeviceBuffer<double> buffer(1, "test::concurrent", __FILE__, __LINE__);
      for (int j = 0; j < 8; ++j)
        mem::emit({"boundary"});
    });
  for (auto& thread : threads)
    thread.join();
  if (std::getenv("PSIFORMER_MEMORY_TEST_DESTRUCTOR_FAILURE"))
  {
    {
      DeviceBuffer<double> buffer(8, "test::destructor_failure", __FILE__, __LINE__);
      fail_free = true;
    }
    fail_free = false;
    require(metrics.snapshot().live_bytes == 192);
  }
  for (void* p : leaked)
    std::free(p);
  mem::emit({"boundary", 0, "quoted\"owner\\with\nnewline"});
  if (std::getenv("PSIFORMER_MEMORY_TEST_LATE"))
    late_buffer = std::make_unique<DeviceBuffer<double>>(8, "test::late", __FILE__, __LINE__);
  std::cout << "memory trace ownership tests passed; enabled=" << mem::enabled() << '\n';
}
