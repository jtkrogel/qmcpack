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
#include "PsiFormerDeviceScoreContraction.h"
#include "PsiFormerDeviceKineticReverse.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
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
  bool serial_score_contractions = false;
  bool batched_score_tiles = false;
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
                                  "[--mixed-values] [--mixed-vgl] [--serial-score-contractions] [--batched-score-tiles]");
    BenchmarkOptions options;
    for (int argument = 4; argument < argc; ++argument)
    {
      const std::string flag(argv[argument]);
      if (flag == "--mixed-values")
        options.mixed_values = true;
      else if (flag == "--mixed-vgl")
        options.mixed_vgl = true;
      else if (flag == "--serial-score-contractions")
        options.serial_score_contractions = true;
      else if (flag == "--batched-score-tiles")
        options.serial_score_contractions = options.batched_score_tiles = true;
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

    // Optional production-shaped comparison: identical serial forward/reverse
    // work, either B real response downloads or C complex contracted downloads.
    // The established score_vjp row above/below is already a batched reverse and
    // must not be presented as the former production host-contraction baseline.
    std::unique_ptr<PreparedDeviceInference> serial_tape;
    std::unique_ptr<PreparedDeviceScoreReverse> serial_reverse;
    std::unique_ptr<PreparedDeviceScoreContraction> serial_contraction;
    std::unique_ptr<PreparedDeviceInference> tile_tape;
    std::unique_ptr<PreparedDeviceScoreReverse> tile_reverse;
    std::vector<double> serial_positions(3 * native.ne), serial_unit_weight(1, 1.0);
    std::array<std::vector<std::complex<double>>, 2> serial_sums;
    std::vector<double> tile_positions, tile_weights;
    if (options.serial_score_contractions)
    {
      const auto serial_start = std::chrono::steady_clock::now();
      serial_tape = std::make_unique<PreparedDeviceInference>(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
      serial_reverse = std::make_unique<PreparedDeviceScoreReverse>(image, 1, queue, blas);
      serial_contraction = std::make_unique<PreparedDeviceScoreContraction>(plan.parameterCount(), 2, queue);
      for (auto& sums : serial_sums)
        sums.resize(plan.parameterCount());
      queue.sync();
      std::cout << "Serial-score comparison: additional_pack_bytes="
                << serial_tape->deviceWorkspaceBytes() + serial_reverse->deviceWorkspaceBytes() +
                    serial_contraction->deviceWorkspaceBytes()
                << " contraction_pack_bytes=" << serial_contraction->deviceWorkspaceBytes()
                << " cold_prepare_seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - serial_start).count()
                << "; all rows' live_bytes include this optional pack; C=1/2 complex channels" << '\n';
    }
    if (options.batched_score_tiles)
    {
      const auto tile_start = std::chrono::steady_clock::now();
      tile_tape = std::make_unique<PreparedDeviceInference>(image, 4, queue, blas, SpatialJetMode::VALUE_ONLY, true);
      tile_reverse = std::make_unique<PreparedDeviceScoreReverse>(image, 4, queue, blas);
      tile_positions.reserve(4 * 3 * native.ne);
      tile_weights.reserve(4);
      queue.sync();
      const auto single_bytes = serial_tape->deviceWorkspaceBytes() + serial_reverse->deviceWorkspaceBytes();
      const auto tile_bytes = tile_tape->deviceWorkspaceBytes() + tile_reverse->deviceWorkspaceBytes();
      std::cout << "True score tiles: prepared_capacity=4 single_graph_bytes=" << single_bytes
                << " tile_graph_bytes=" << tile_bytes << " additional_capacity_bytes=" << tile_bytes - single_bytes
                << " cold_prepare_seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - tile_start).count()
                << "; optional comparison retains both graphs; parameter storage per graph is independent of tile capacity" << '\n';
    }
    auto serial_score_vjp = [&](bool resident, std::size_t channels, bool real_weights = false) {
      if (resident)
        serial_contraction->begin(channels);
      else
        for (std::size_t channel = 0; channel < channels; ++channel)
          std::fill(serial_sums[channel].begin(), serial_sums[channel].end(), std::complex<double>{});
      for (std::size_t sample = 0; sample < options.walkers; ++sample)
      {
        std::copy_n(positions.data() + sample * 3 * native.ne, serial_positions.size(), serial_positions.begin());
        serial_tape->evaluate(serial_positions);
        const double* device_response = serial_reverse->reverse(*serial_tape, serial_unit_weight);
        const std::array<std::complex<double>, 2> coefficients{{
            {weights[sample], real_weights ? 0.0 : .125 * weights[sample]},
            {-.7 * weights[sample], .21 * weights[sample]}}};
        if (resident)
          serial_contraction->accumulate(device_response, coefficients.data(), channels);
        else
        {
          serial_reverse->downloadResponse(response);
          for (std::size_t channel = 0; channel < channels; ++channel)
            for (std::size_t parameter = 0; parameter < response.size(); ++parameter)
              serial_sums[channel][parameter] += coefficients[channel] * response[parameter];
        }
      }
      if (resident)
      {
        serial_contraction->finish();
        for (std::size_t channel = 0; channel < channels; ++channel)
          serial_contraction->download(channel, serial_sums[channel].data(), serial_sums[channel].size());
      }
    };
    auto batched_score_vjp = [&]() {
      serial_contraction->begin(1);
      for (std::size_t offset = 0; offset < options.walkers; offset += 4)
      {
        const auto count = std::min(std::size_t{4}, options.walkers - offset);
        tile_positions.resize(count * 3 * native.ne);
        tile_weights.resize(count);
        std::copy_n(positions.data() + offset * 3 * native.ne, tile_positions.size(), tile_positions.begin());
        std::copy_n(weights.data() + offset, count, tile_weights.begin());
        tile_tape->evaluate(tile_positions);
        const auto* device_response = tile_reverse->reverse(*tile_tape, tile_weights);
        const std::complex<double> unit(1, 0);
        serial_contraction->accumulate(device_response, &unit, 1);
      }
      serial_contraction->finish();
      serial_contraction->download(0, serial_sums[0].data(), serial_sums[0].size());
    };
    if (serial_contraction)
    {
      serial_score_vjp(false, 2);
      const auto host_reference = serial_sums;
      serial_score_vjp(true, 2);
      for (std::size_t channel = 0; channel < 2; ++channel)
        for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
          if (std::abs(serial_sums[channel][parameter] - host_reference[channel][parameter]) >
              1.e-9 * std::max(1.0, std::abs(host_reference[channel][parameter])))
            throw std::runtime_error("Serial resident/host score contraction parity failed");
    }
    if (tile_tape)
    {
      serial_score_vjp(true, 1, true);
      const auto reference = serial_sums[0];
      batched_score_vjp();
      for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
        if (std::abs(serial_sums[0][parameter] - reference[parameter]) >
            1.e-9 * std::max(1.0, std::abs(reference[parameter])))
          throw std::runtime_error("Real C1 serial-resident/true-batch score parity failed");
      std::cout << "PASS real C1 serial-resident/true-batch all-parameter parity; configurations=" << options.walkers
                << " weighted_tiles=" << (options.walkers + 3) / 4
                << " final_tile_count=" << (options.walkers - 1) % 4 + 1 << '\n';
    }
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
    if (serial_contraction)
      for (const auto channels : {std::size_t{1}, std::size_t{2}})
      {
        operations.emplace_back("serial_score_vjp_host_c" + std::to_string(channels),
                                 [&, channels]() { serial_score_vjp(false, channels); });
        operations.emplace_back("serial_score_vjp_resident_c" + std::to_string(channels),
                                 [&, channels]() { serial_score_vjp(true, channels); });
      }
    if (tile_tape)
    {
      operations.emplace_back("serial_score_vjp_resident_real_c1", [&]() { serial_score_vjp(true, 1, true); });
      operations.emplace_back("batched_score_vjp_resident_real_c1_t4", batched_score_vjp);
    }
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
