//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** Opt-in actual-device validation of two real native inference contexts.
 * No trained files are required. Profile with nsys for a kernel/stream timeline;
 * host permit overlap alone is deliberately not presented as GPU kernel overlap.
 */
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerKineticExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "PsiFormerDeviceBatchBackend.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>

namespace
{
using namespace qmcplusplus::psiformer;
using namespace qmcplusplus::psiformer::device;

std::vector<double> copyResult(const pf::DirectBatchTileResultView& result, std::size_t electrons)
{
  std::vector<double> values;
  for (std::size_t b = 0; b < result.size; ++b)
  {
    values.push_back(result.sign[b]);
    values.push_back(result.logabs[b * result.value_stride]);
    for (std::size_t lane = 0; lane < 3 * electrons; ++lane)
      values.push_back(result.gradient[b * result.gradient_configuration_stride + lane * result.gradient_lane_stride]);
    for (std::size_t e = 0; e < electrons; ++e)
    {
      values.push_back(result.lap_log[b * result.lap_log_configuration_stride + e * result.lap_log_lane_stride]);
      values.push_back(result.lap_ratio[b * result.lap_ratio_configuration_stride + e]);
    }
  }
  return values;
}

void compare(const std::vector<double>& actual, const std::vector<double>& expected)
{
  if (actual.size() != expected.size())
    throw std::runtime_error("Concurrent inference result extent differs");
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (!device_math::isFiniteBinary64(actual[i]) ||
        std::abs(actual[i] - expected[i]) > 2e-10 * (1 + std::abs(expected[i])))
      throw std::runtime_error("Concurrent inference result differs from serialized GPU reference");
}

void run(int ordinal)
{
  const ModelShape shape{1, 1, 2, 16, 256, 4, 4};
  const auto initialized = initializePsiFormerParameters(shape, 17);
  const auto plan = PsiFormerExecutionPlan::fromParameters(initialized, shape);
  auto image = std::make_shared<PreparedDeviceModel>(plan, initialized.values,
      std::vector<double>{0, 0, 0, 1.4, 0, 0});
  const std::array<std::vector<double>, 2> positions{{
      {.2, .3, -.1, 1.2, -.1, .2, .4, -.2, .1, 1.5, .2, -.3},
      {-.3, .2, .4, 1.6, -.2, .1, .1, -.4, .2, 1.3, .3, -.2}}};
  std::array<std::vector<double>, 2> expected;
  {
    PreparedDeviceBatchBackend serial(image);
    auto workspace = serial.prepare(2);
    for (std::size_t lane = 0; lane < 2; ++lane)
      expected[lane] = copyResult(serial.evaluate(*workspace, pf::DirectBatchMode::FULL_VGL,
          positions[lane].data(), 2, 0), shape.electrons());
    if (serial.peakAdmittedInferenceCalls() != 1)
      throw std::runtime_error("Default inference did not preserve serialized admission");
  }
  DeviceExecutionPreparation preparation;
  preparation.active_inference_crowds = 2;
  PreparedDeviceBatchBackend concurrent(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE, preparation);
  std::array<std::unique_ptr<pf::DirectBatchTileBackendWorkspace>, 2> workspaces{
      concurrent.prepare(2), concurrent.prepare(2)};
  auto& metrics = deviceMemoryMetrics(ordinal);
  const auto warm = metrics.snapshot();
  std::array<std::exception_ptr, 2> failures{};
  std::array<std::vector<std::array<double, 2>>, 2> timeline;
  std::mutex mutex;
  std::condition_variable start;
  std::size_t ready = 0;
  bool release = false;
  const auto origin = std::chrono::steady_clock::now();
  const auto milliseconds = [&] {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - origin).count();
  };
  std::array<std::thread, 2> callers;
  for (std::size_t lane = 0; lane < 2; ++lane)
    callers[lane] = std::thread([&, lane] {
      try
      {
#ifdef QMC_CUDA2HIP
        checkRuntime(hipSetDevice(ordinal), "validation worker device");
#else
        checkRuntime(cudaSetDevice(ordinal), "validation worker device");
#endif
        timeline[lane].reserve(3);
        {
          std::unique_lock lock(mutex);
          ++ready;
          start.notify_all();
          start.wait(lock, [&] { return release; });
        }
        for (int repeat = 0; repeat < 3; ++repeat)
        {
          const auto begin = milliseconds();
          compare(copyResult(concurrent.evaluate(*workspaces[lane], pf::DirectBatchMode::FULL_VGL,
              positions[lane].data(), 2, 0), shape.electrons()), expected[lane]);
          timeline[lane].push_back({begin, milliseconds()});
        }
      }
      catch (...)
      {
        failures[lane] = std::current_exception();
        std::lock_guard lock(mutex);
        ++ready;
        start.notify_all();
      }
    });
  {
    std::unique_lock lock(mutex);
    start.wait(lock, [&] { return ready >= 2; });
    release = true;
    start.notify_all();
  }
  for (auto& caller : callers)
    caller.join();
  for (const auto& failure : failures)
    if (failure)
      std::rethrow_exception(failure);
  if (concurrent.peakAdmittedInferenceCalls() != 2)
    throw std::runtime_error("Independent inference submissions did not overlap");
  if (metrics.snapshot().allocation_calls != warm.allocation_calls || metrics.snapshot().free_calls != warm.free_calls)
    throw std::runtime_error("Concurrent warm inference allocated or freed native device storage");

  const auto borrowed = concurrent.evaluate(*workspaces[0], pf::DirectBatchMode::FULL_VGL, positions[0].data(), 2, 0);
  (void)concurrent.evaluate(*workspaces[1], pf::DirectBatchMode::FULL_VGL, positions[1].data(), 2, 0);
  compare(copyResult(borrowed, shape.electrons()), expected[0]);
  concurrent.publishParameters(initialized.values, 1, [] {});
  compare(copyResult(borrowed, shape.electrons()), expected[0]);
  for (std::size_t lane = 0; lane < 2; ++lane)
    compare(copyResult(concurrent.evaluate(*workspaces[lane], pf::DirectBatchMode::FULL_VGL,
        positions[lane].data(), 2, 1), shape.electrons()), expected[lane]);
  auto changed = initialized.values;
  changed[plan.parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin] *= 1.02;
  std::array<std::vector<double>, 2> changed_expected;
  {
    PreparedDeviceBatchBackend serial_changed(image->makeUpdatedImage(changed, 2));
    auto workspace = serial_changed.prepare(2);
    for (std::size_t lane = 0; lane < 2; ++lane)
      changed_expected[lane] = copyResult(serial_changed.evaluate(*workspace, pf::DirectBatchMode::FULL_VGL,
          positions[lane].data(), 2, 2), shape.electrons());
  }
  bool actual_change = false;
  for (std::size_t i = 0; i < expected[0].size(); ++i)
    actual_change = actual_change || expected[0][i] != changed_expected[0][i];
  if (!actual_change)
    throw std::runtime_error("Changed-weight publication fixture did not change observables");
  bool native_committed = false;
  concurrent.publishParameters(changed, 2, [&] { native_committed = true; });
  if (!native_committed)
    throw std::runtime_error("Changed-weight publication omitted the authoritative commit");
  compare(copyResult(borrowed, shape.electrons()), expected[0]);
  bool stale_rejected = false;
  try { (void)concurrent.evaluate(*workspaces[0], pf::DirectBatchMode::FULL_VGL, positions[0].data(), 2, 1); }
  catch (const std::invalid_argument&) { stale_rejected = true; }
  if (!stale_rejected)
    throw std::runtime_error("Changed-weight publication accepted stale parameter generation");
  for (std::size_t lane = 0; lane < 2; ++lane)
    compare(copyResult(concurrent.evaluate(*workspaces[lane], pf::DirectBatchMode::FULL_VGL,
        positions[lane].data(), 2, 2), shape.electrons()), changed_expected[lane]);
  bool rejected = false;
  try { (void)concurrent.prepareProductSession(); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Concurrent inference incorrectly admitted training products");
  for (std::size_t lane = 0; lane < 2; ++lane)
    for (const auto& interval : timeline[lane])
      std::cout << "host_call lane=" << lane << " begin_ms=" << interval[0] << " end_ms=" << interval[1] << '\n';
  const auto after = metrics.snapshot();
  std::cout << "PASS two-context FP64 inference; max_admitted_calls=" << concurrent.peakAdmittedInferenceCalls()
            << ", live_native_bytes=" << after.live_bytes << ", peak_native_bytes=" << after.peak_bytes
            << "; host intervals are not kernel-overlap or speedup evidence\n";
}
} // namespace

int main()
{
  try
  {
    int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
    if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
      return 77;
    checkRuntime(hipGetDevice(&ordinal), "validation main device");
#else
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
      return 77;
    checkRuntime(cudaGetDevice(&ordinal), "validation main device");
#endif
    reportCurrentDeviceMemoryAdmission(std::cout);
    auto& metrics = deviceMemoryMetrics(ordinal);
    const auto before = metrics.snapshot();
    run(ordinal);
    if (metrics.snapshot().live_bytes != before.live_bytes || metrics.snapshot().failed_frees != before.failed_frees)
      throw std::runtime_error("Concurrent inference left native allocations or failed frees");
    std::cout << "PASS native cleanup\n";
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
