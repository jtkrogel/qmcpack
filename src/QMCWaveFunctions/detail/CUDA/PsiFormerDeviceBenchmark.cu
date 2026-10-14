//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceBenchmark.cu
 * @brief Warm, fixed-capacity inference/parameter-product timing and copy evidence.
 *
 * Every operation includes its required forward pass and host-visible result.
 * This developer benchmark is not a VMC sampler or scientific acceptance test.
 */
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "PsiFormerDeviceInference.h"
#include "PsiFormerDeviceScoreReverse.h"
#include "PsiFormerDeviceScoreJvp.h"
#include "PsiFormerDeviceKineticReverse.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>

using namespace qmcplusplus::psiformer;
using namespace qmcplusplus::psiformer::device;

/** Retain logical population/architecture separately from measured repetitions. */
struct BenchmarkOptions
{
  std::size_t walkers = 3, repetitions = 5;
  std::size_t determinants = 16, width = 256, heads = 4, blocks = 4;
  bool mixed_values = false;
  bool mixed_vgl = false;
};

/// Read only explicit positive sizes; reject trailing/unknown command-line text.
std::size_t readSize(const char* text)
{
  const std::string input(text);
  if (input.empty() || input.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("Benchmark sizes must be positive decimal integers");
  const auto value = std::stoull(input);
  if (value == 0 || value > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("Benchmark size is zero or out of range");
  return static_cast<std::size_t>(value);
}

/// Prepare one model, warm every product, then report timed reusable intervals.
int main(int argc, char** argv)
{
  try
  {
    if (argc < 4)
      throw std::invalid_argument("Usage: benchmark parameters.h5 configurations.h5 report.csv "
                                  "[--walkers N] [--repetitions R] [--shape D WIDTH HEADS BLOCKS] "
                                  "[--mixed-values] [--mixed-vgl]");
    BenchmarkOptions options;
    for (int argument = 4; argument < argc; ++argument)
    {
      const std::string flag(argv[argument]);
      if (flag == "--mixed-values")
        options.mixed_values = true;
      else if (flag == "--mixed-vgl")
        options.mixed_vgl = true;
      else if (flag == "--walkers" && argument + 1 < argc)
        options.walkers = readSize(argv[++argument]);
      else if (flag == "--repetitions" && argument + 1 < argc)
        options.repetitions = readSize(argv[++argument]);
      else if (flag == "--shape" && argument + 4 < argc)
      {
        options.determinants = readSize(argv[++argument]);
        options.width = readSize(argv[++argument]);
        options.heads = readSize(argv[++argument]);
        options.blocks = readSize(argv[++argument]);
      }
      else
        throw std::invalid_argument("Unknown or incomplete benchmark option: " + flag);
    }

    // Input loading is deliberately separate from device preparation and warmup.
    pf::PsiFormer native(pf::Parameters(argv[1]), pf::ConfigData(argv[2]),
                         options.determinants, options.width, options.heads, options.blocks);
    const auto plan = PsiFormerExecutionPlan::fromParameters(native.p,
        {native.cfg.nup, native.cfg.ndown, native.cfg.nuclei.shape[0], native.ndet,
         native.dim, native.heads, native.blocks});
    if (native.cfg.nconfig == 0)
      throw std::invalid_argument("Benchmark requires at least one supplied configuration");
    const auto coordinate_count = spatial_detail::checkedProduct(options.walkers,
        spatial_detail::checkedProduct(3, native.ne, "Benchmark coordinate overflow"), "Benchmark population overflow");
    std::vector<double> positions(coordinate_count);
    for (std::size_t walker = 0; walker < options.walkers; ++walker)
      std::copy_n(native.cfg.electrons.x.data() + (walker % native.cfg.nconfig) * 3 * native.ne,
                   3 * native.ne, positions.data() + walker * 3 * native.ne);

    const auto prepare_start = std::chrono::steady_clock::now();
    qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
    AcceleratorBlasHandle blas(queue);
    auto image = std::make_shared<PreparedDeviceModel>(plan, native.p.flat_values(), native.cfg.nuclei.x);
    PreparedDeviceInference value(image, options.walkers, queue, blas, SpatialJetMode::VALUE_ONLY);
    PreparedDeviceInference spatial(image, options.walkers, queue, blas);
    PreparedDeviceInference value_tape(image, options.walkers, queue, blas, SpatialJetMode::VALUE_ONLY, true);
    PreparedDeviceInference spatial_tape(image, options.walkers, queue, blas, SpatialJetMode::FULL_VGL, false, true);
    PreparedDeviceScoreReverse score(image, options.walkers, queue, blas);
    PreparedDeviceScoreJvp jvp(image, options.walkers, queue, blas);
    PreparedDeviceKineticReverse kinetic(image, options.walkers, queue, blas);
    queue.sync();
    const double prepare_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - prepare_start).count();
    // Extra precision caches/graphs are opt-in, and their complete cold cost is
    // separate from the established all-FP64 benchmark pack preparation.
    std::unique_ptr<PreparedDeviceMixedValueInference> mixed;
    std::unique_ptr<PreparedDeviceMixedInference> mixed_spatial;
    double mixed_prepare_seconds = 0;
    std::size_t mixed_pack_bytes = 0, mixed_parameter_cache_bytes = 0, mixed_retry_calls = 0;
    double mixed_vgl_prepare_seconds = 0;
    std::size_t mixed_vgl_pack_bytes = 0, mixed_vgl_parameter_cache_bytes = 0;
    if (options.mixed_values)
    {
      const auto mixed_start = std::chrono::steady_clock::now();
      mixed = std::make_unique<PreparedDeviceMixedValueInference>(image, options.walkers, queue, blas);
      queue.sync();
      mixed_prepare_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - mixed_start).count();
      mixed_pack_bytes = mixed->deviceWorkspaceBytes();
      // This cache preserves canonical offsets; sensitive entries are unused,
      // but the complete reserved array must still be charged to device memory.
      mixed_parameter_cache_bytes = mixed_pack_bytes == value.deviceWorkspaceBytes()
          ? 0 : checkedDeviceBytes<float>(plan.parameterCount());
      std::cout << "Mixed-value experiment: additional_pack_bytes=" << mixed_pack_bytes
                << " parameter_cache_bytes=" << mixed_parameter_cache_bytes
                << " cache_and_graph_prepare_seconds=" << mixed_prepare_seconds
                << "; all rows' live_bytes include every prepared FP64/mixed graph" << '\n';
    }
    if (options.mixed_vgl)
    {
      const auto mixed_start = std::chrono::steady_clock::now();
      mixed_spatial = std::make_unique<PreparedDeviceMixedInference>(
          image, options.walkers, queue, blas, SpatialJetMode::FULL_VGL);
      queue.sync();
      mixed_vgl_prepare_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - mixed_start).count();
      mixed_vgl_pack_bytes = mixed_spatial->deviceWorkspaceBytes();
      mixed_vgl_parameter_cache_bytes = mixed_vgl_pack_bytes == spatial.deviceWorkspaceBytes()
          ? 0 : checkedDeviceBytes<float>(plan.parameterCount());
      std::cout << "Matched mixed-VGL experiment: additional_pack_bytes=" << mixed_vgl_pack_bytes
                << " parameter_cache_bytes=" << mixed_vgl_parameter_cache_bytes
                << " cache_and_graph_prepare_seconds=" << mixed_vgl_prepare_seconds
                << "; every spatial plane uses the selected affine precision" << '\n';
    }
    int device_ordinal = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&device_ordinal), "benchmark hipGetDevice");
#else
    checkRuntime(cudaGetDevice(&device_ordinal), "benchmark cudaGetDevice");
#endif
    auto& metrics = deviceMemoryMetrics(device_ordinal);

    // Canonical accumulators/directions are O(P); results are O(B), never B by P.
    std::vector<double> phases(options.walkers), logs(options.walkers), empty_ratios;
    std::vector<double> jets(spatialJetSpanElements(spatial.resultLayout()));
    std::vector<double> ratios(options.walkers * native.ne), weights(options.walkers, 1.0 / options.walkers);
    std::vector<double> response(plan.parameterCount()), direction(plan.parameterCount()), products(options.walkers);
    for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
      direction[parameter] = (static_cast<int>(parameter % 11) - 5) * .01 / std::sqrt(direction.size());
    jvp.prepareDirection(direction);
    std::vector<std::pair<std::string, std::function<void()>>> operations;
    operations.emplace_back("value", [&]() {
      value.evaluate(positions);
      value.downloadResults(phases, logs, empty_ratios);
    });
    if (mixed)
      operations.emplace_back("mixed_value", [&]() {
        mixed->evaluate(positions);
        mixed->downloadResults(phases, logs, empty_ratios);
        mixed_retry_calls += mixed->usedFp64Retry();
      });
    operations.emplace_back("vgl", [&]() {
      spatial.evaluate(positions);
      spatial.downloadResults(phases, jets, ratios);
    });
    if (mixed_spatial)
      operations.emplace_back("mixed_vgl", [&]() {
        mixed_spatial->evaluate(positions);
        mixed_spatial->downloadResults(phases, jets, ratios);
        mixed_retry_calls += mixed_spatial->usedFp64Retry();
      });
    operations.emplace_back("score_vjp", [&]() {
      value_tape.evaluate(positions);
      score.reverse(value_tape, weights);
      score.downloadResponse(response);
    });
    operations.emplace_back("score_jvp", [&]() {
      value_tape.evaluate(positions);
      jvp.evaluate(value_tape, direction);
      jvp.downloadProducts(products);
    });
    operations.emplace_back("score_jvp_resident", [&]() {
      value_tape.evaluate(positions);
      jvp.evaluatePrepared(value_tape);
      jvp.downloadProducts(products);
    });
    operations.emplace_back("kinetic_vjp", [&]() {
      spatial_tape.evaluate(positions);
      kinetic.evaluate(spatial_tape, weights);
      kinetic.downloadResponse(response);
    });

    std::ofstream report(argv[3]);
    if (!report)
      throw std::runtime_error("Cannot open benchmark report");
    report << "operation,walkers,repetitions,prepare_seconds,min_seconds,median_seconds,"
              "allocation_calls,free_calls,live_bytes,peak_bytes,upload_calls,uploaded_bytes,"
              "download_calls,downloaded_bytes,mixed_cache_and_graph_prepare_seconds,"
              "mixed_pack_bytes,mixed_parameter_cache_bytes,mixed_retry_calls,"
              "mixed_projections_per_eval,mixed_input_cast_elements_per_eval,"
              "mixed_output_cast_elements_per_eval,mixed_cast_roundtrip_bytes_per_eval\n";
    report << std::setprecision(17);
    for (const auto& operation : operations)
    {
      operation.second();
      queue.sync();
      mixed_retry_calls = 0;
      metrics.resetInterval();
      std::vector<double> timings;
      timings.reserve(options.repetitions);
      for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition)
      {
        const auto start = std::chrono::steady_clock::now();
        operation.second();
        queue.sync();
        timings.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
      }
      const auto memory = metrics.snapshot();
      if (memory.allocation_calls != 0 || memory.free_calls != 0 || memory.failed_frees != 0 ||
          memory.live_bytes != memory.peak_bytes)
        throw std::runtime_error("Warmed fixed-capacity operation allocated or lost explicitly owned device storage");
      std::sort(timings.begin(), timings.end());
      const auto middle = timings.size() / 2;
      const double median = timings.size() % 2 ? timings[middle] : .5 * (timings[middle - 1] + timings[middle]);
      report << operation.first << ',' << options.walkers << ',' << options.repetitions << ','
             << prepare_seconds << ',' << timings.front() << ',' << median << ',' << memory.allocation_calls << ','
             << memory.free_calls << ',' << memory.live_bytes << ',' << memory.peak_bytes << ','
             << memory.upload_calls << ',' << memory.uploaded_bytes << ','
             << memory.download_calls << ',' << memory.downloaded_bytes;
      const bool is_mixed = operation.first == "mixed_value" || operation.first == "mixed_vgl";
      const auto* measured_mixed = operation.first == "mixed_vgl" ? mixed_spatial.get() : mixed.get();
      if (is_mixed)
      {
        const auto& casts = measured_mixed->statistics();
        report << ',' << (operation.first == "mixed_vgl" ? mixed_vgl_prepare_seconds : mixed_prepare_seconds)
               << ',' << (operation.first == "mixed_vgl" ? mixed_vgl_pack_bytes : mixed_pack_bytes)
               << ',' << (operation.first == "mixed_vgl" ? mixed_vgl_parameter_cache_bytes : mixed_parameter_cache_bytes)
               << ',' << mixed_retry_calls << ',' << casts.projections << ',' << casts.input_cast_elements
               << ',' << casts.output_cast_elements << ',' << casts.roundTripBytes();
      }
      else
        report << ",0,0,0,0,0,0,0,0";
      report << '\n';
      report.flush();
      std::cout << "PASS " << operation.first << ": walkers=" << options.walkers
                << " median_seconds=" << median << " owned_allocations=0 live_device_bytes=" << memory.live_bytes << '\n';
      if (is_mixed)
        std::cout << "  explicit whole-graph FP64 retries=" << mixed_retry_calls << '/' << options.repetitions
                  << " cast_roundtrip_bytes_per_eval=" << measured_mixed->statistics().roundTripBytes()
                  << " last_retry_reason=" << measured_mixed->retryReason() << '\n';
    }
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
