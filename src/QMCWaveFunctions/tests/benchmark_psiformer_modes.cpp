//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file benchmark_psiformer_modes.cpp
 * @brief Non-gating direct/oracle benchmark manifest for every PsiFormer mode.
 *
 * The executable deliberately requires developer-supplied HDF5 exports and is
 * never registered with CTest.  Ordinary CI remains external-data-free.
 */

#define PSIFORMER_LIBRARY
#include "config.h"
#include "git-rev.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerBatchExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerKineticExecutor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif

namespace
{
using Clock = std::chrono::steady_clock;

#define PSIFORMER_STRINGIFY_DETAIL(value) #value
#define PSIFORMER_STRINGIFY(value) PSIFORMER_STRINGIFY_DETAIL(value)

volatile double benchmark_sink = 0.0;

/** Command-line controls for one reproducible developer benchmark. */
struct Options
{
  std::string parameter_path;
  std::string configuration_path;
  std::string output_path;
  int repeats                         = 5;
  std::size_t configuration_limit    = 0;
  std::size_t kinetic_configurations = 1;
  std::size_t active_electron        = 0;
  std::vector<std::size_t> batch_sizes{1, 2, 4};
};

/** Minimal observable returned from a timed call without copying large arrays. */
struct Observation
{
  double sign     = 1.0;
  double logabs   = 0.0;
  double checksum = 0.0;
};

/** Timing and storage metadata for one backend/request pair. */
struct ModeSummary
{
  std::string mode;
  std::string backend;
  std::size_t batch_size                = 1;
  std::size_t calls_per_repeat          = 0;
  std::size_t configurations_per_repeat = 0;
  std::size_t workspace_bytes           = 0;
  std::size_t logical_output_bytes      = 0;
  double warmup_seconds                 = 0.0;
  double peak_resident_mib_after_mode   = 0.0;
  Observation warmup;
  std::vector<double> raw_seconds;
};

/** Print the supported option surface and exit through normal error handling. */
[[noreturn]] void usage(const char* executable, const std::string& error = {})
{
  if (!error.empty())
    std::cerr << "error: " << error << '\n';
  std::cerr << "usage: " << executable
            << " PARAMETERS.h5 CONFIGURATIONS.h5 [--repeats N]"
               " [--configuration-limit N] [--kinetic-configurations N]"
               " [--active-electron N] [--batch-sizes 1,2,4]"
               " [--output FILE.json]\n";
  std::exit(EXIT_FAILURE);
}

/** Parse a comma-separated list of positive direct-batch capacities. */
std::vector<std::size_t> parseBatchSizes(const std::string& value)
{
  std::vector<std::size_t> sizes;
  std::istringstream input(value);
  std::string token;
  while (std::getline(input, token, ','))
  {
    if (token.empty())
      throw std::invalid_argument("empty batch size");
    const std::size_t size = std::stoull(token);
    if (size == 0)
      throw std::invalid_argument("batch sizes must be positive");
    sizes.push_back(size);
  }
  if (sizes.empty())
    throw std::invalid_argument("at least one batch size is required");
  return sizes;
}

/** Parse command-line benchmark controls. */
Options parseOptions(int argc, char** argv)
{
  if (argc < 3)
    usage(argv[0]);

  Options options;
  options.parameter_path     = argv[1];
  options.configuration_path = argv[2];
  for (int argument = 3; argument < argc; ++argument)
  {
    const std::string option = argv[argument];
    if (argument + 1 >= argc)
      usage(argv[0], "missing value after " + option);
    const std::string value = argv[++argument];
    try
    {
      if (option == "--repeats")
        options.repeats = std::stoi(value);
      else if (option == "--configuration-limit")
        options.configuration_limit = std::stoull(value);
      else if (option == "--kinetic-configurations")
        options.kinetic_configurations = std::stoull(value);
      else if (option == "--active-electron")
        options.active_electron = std::stoull(value);
      else if (option == "--batch-sizes")
        options.batch_sizes = parseBatchSizes(value);
      else if (option == "--output")
        options.output_path = value;
      else
        usage(argv[0], "unknown option " + option);
    }
    catch (const std::exception&)
    {
      usage(argv[0], "invalid value for " + option + ": " + value);
    }
  }
  if (options.repeats <= 0)
    usage(argv[0], "--repeats must be positive");
  if (options.kinetic_configurations == 0)
    usage(argv[0], "--kinetic-configurations must be positive");
  return options;
}

/// Return elapsed wall-clock seconds from a steady-clock timestamp.
double elapsedSeconds(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// Return cumulative process peak resident memory in MiB on Linux.
double peakResidentMiB()
{
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    throw std::runtime_error("getrusage failed");
  return static_cast<double>(usage.ru_maxrss) / 1024.0;
}

/// Compute a median while preserving raw timing order in the manifest.
double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
}

/// Return an environment value or an explicit marker when it is unset.
std::string environmentValue(const char* name)
{
  const char* value = std::getenv(name);
  return value ? value : "<unset>";
}

/// Escape a string for the small JSON manifest emitted by this benchmark.
std::string jsonString(const std::string& value)
{
  std::ostringstream output;
  output << '"';
  for (const char character : value)
  {
    if (character == '"' || character == '\\')
      output << '\\' << character;
    else if (character == '\n')
      output << "\\n";
    else
      output << character;
  }
  output << '"';
  return output.str();
}

/// Return the hostname recorded with the affinity and threading manifest.
std::string hostName()
{
  char hostname[256]{};
  if (gethostname(hostname, sizeof(hostname) - 1) != 0)
    return "<unknown>";
  return hostname;
}

/// Count CPUs available through the current Linux affinity mask when possible.
std::size_t affinityCpuCount()
{
#ifdef __linux__
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
    return CPU_COUNT(&affinity);
#endif
  return std::thread::hardware_concurrency();
}

/** Time one warmed request without imposing noisy absolute thresholds. */
template<class Evaluate>
ModeSummary benchmarkMode(std::string mode,
                          std::string backend,
                          std::size_t batch_size,
                          std::size_t calls_per_repeat,
                          std::size_t workspace_bytes,
                          std::size_t logical_output_bytes,
                          int repeats,
                          Evaluate&& evaluate)
{
  ModeSummary summary;
  summary.mode                      = std::move(mode);
  summary.backend                   = std::move(backend);
  summary.batch_size                = batch_size;
  summary.calls_per_repeat          = calls_per_repeat;
  summary.configurations_per_repeat = calls_per_repeat * batch_size;
  summary.workspace_bytes           = workspace_bytes;
  summary.logical_output_bytes      = logical_output_bytes;

  auto start             = Clock::now();
  summary.warmup         = evaluate(0);
  summary.warmup_seconds = elapsedSeconds(start);
  benchmark_sink += summary.warmup.checksum;

  summary.raw_seconds.reserve(repeats);
  for (int repeat = 0; repeat < repeats; ++repeat)
  {
    start = Clock::now();
    for (std::size_t call = 0; call < calls_per_repeat; ++call)
      benchmark_sink += evaluate(call).checksum;
    summary.raw_seconds.push_back(elapsedSeconds(start));
  }
  summary.peak_resident_mib_after_mode = peakResidentMiB();

  std::cerr << std::left << std::setw(22) << summary.mode << std::setw(9) << summary.backend
            << " batch=" << std::setw(3) << summary.batch_size
            << " median_s/config=" << std::scientific
            << median(summary.raw_seconds) / summary.configurations_per_repeat
            << " workspace_mib=" << std::fixed << std::setprecision(3)
            << static_cast<double>(summary.workspace_bytes) / (1024.0 * 1024.0) << '\n';
  return summary;
}

/// Compare direct and oracle warmup observables without making timing gating decisions.
void validatePair(const ModeSummary& direct, const ModeSummary& oracle, double tolerance)
{
  if (direct.mode != oracle.mode || direct.warmup.sign != oracle.warmup.sign)
    throw std::runtime_error("PsiFormer direct/oracle mode or sign mismatch for " + direct.mode);
  auto close = [tolerance](double first, double second) {
    return std::abs(first - second) <= tolerance * (1.0 + std::max(std::abs(first), std::abs(second)));
  };
  if (!close(direct.warmup.logabs, oracle.warmup.logabs) ||
      !close(direct.warmup.checksum, oracle.warmup.checksum))
    throw std::runtime_error("PsiFormer direct/oracle warmup mismatch for " + direct.mode);
}

/// Accumulate a few deterministic output entries so every requested product is observable.
template<class View>
double endpointChecksum(const View& values)
{
  if (values.size() == 0)
    return 0.0;
  return values[0] + values[values.size() - 1];
}

/// Observe the score view, whose size is a public field for adapter efficiency.
double endpointChecksum(const pf::DirectParameterScoreView& values)
{
  if (values.size == 0)
    return 0.0;
  return values[0] + values[values.size - 1];
}

/// Observe a native result without traversing its potentially million-element outputs.
Observation observeNative(const pf::Result& result)
{
  double checksum = result.sign + result.logabs + result.value;
  auto endpoints = [&checksum](const std::vector<double>& values) {
    if (!values.empty())
      checksum += values.front() + values.back();
  };
  endpoints(result.gradient);
  endpoints(result.active_gradient);
  endpoints(result.lap_log);
  endpoints(result.lap_ratio);
  endpoints(result.param_gradient);
  endpoints(result.local_energy_param_gradient);
  return {result.sign, result.logabs, checksum};
}

/// Create a native evaluator request with standalone Coulomb validation disabled.
pf::EvaluationRequest request(pf::SpatialDerivativeRequest spatial,
                              pf::ParameterDerivativeRequest parameter,
                              int active_electron = -1)
{
  pf::EvaluationRequest result;
  result.spatial_derivatives    = spatial;
  result.parameter_derivatives  = parameter;
  result.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  result.active_electron        = active_electron;
  return result;
}

/// Write one stable JSON object describing a timed mode.
void writeMode(std::ostream& output, const ModeSummary& summary)
{
  const double middle = median(summary.raw_seconds);
  output << "    {\"mode\":" << jsonString(summary.mode)
         << ",\"backend\":" << jsonString(summary.backend)
         << ",\"batch_size\":" << summary.batch_size
         << ",\"calls_per_repeat\":" << summary.calls_per_repeat
         << ",\"configurations_per_repeat\":" << summary.configurations_per_repeat
         << ",\"workspace_bytes\":" << summary.workspace_bytes
         << ",\"logical_output_bytes_per_configuration\":" << summary.logical_output_bytes
         << ",\"warmup_seconds\":" << summary.warmup_seconds
         << ",\"raw_seconds\":[";
  for (std::size_t sample = 0; sample < summary.raw_seconds.size(); ++sample)
  {
    if (sample != 0)
      output << ',';
    output << summary.raw_seconds[sample];
  }
  output << "],\"median_seconds\":" << middle
         << ",\"median_seconds_per_configuration\":"
         << middle / summary.configurations_per_repeat
         << ",\"peak_resident_mib_after_mode\":" << summary.peak_resident_mib_after_mode
         << ",\"warmup_sign\":" << summary.warmup.sign
         << ",\"warmup_logabs\":" << summary.warmup.logabs << '}';
}

/// Emit one self-describing benchmark manifest suitable for paired-run archiving.
void writeReport(std::ostream& output,
                 const Options& options,
                 const pf::PsiFormer& model,
                 double setup_seconds,
                 double baseline_peak_mib,
                 const std::vector<ModeSummary>& modes)
{
  output << std::setprecision(12) << "{\n"
         << "  \"schema\":\"qmcpack.psiformer.mode_benchmark.v1\",\n"
         << "  \"qmcpack_version\":"
         << jsonString(std::to_string(QMCPACK_VERSION_MAJOR) + "." +
                       std::to_string(QMCPACK_VERSION_MINOR) + "." +
                       std::to_string(QMCPACK_VERSION_PATCH)) << ",\n"
         << "  \"qmcpack_git_revision\":"
         << jsonString(PSIFORMER_STRINGIFY(GIT_HASH_RAW)) << ",\n"
         << "  \"compiler\":" << jsonString(__VERSION__) << ",\n"
#ifdef QMC_COMPLEX
         << "  \"qmcpack_build_complex\":true,\n"
#else
         << "  \"qmcpack_build_complex\":false,\n"
#endif
         << "  \"parameter_file\":" << jsonString(options.parameter_path) << ",\n"
         << "  \"configuration_file\":" << jsonString(options.configuration_path) << ",\n"
         << "  \"parameter_count\":" << model.p.size() << ",\n"
         << "  \"electron_count\":" << model.ne << ",\n"
         << "  \"available_configuration_count\":" << model.cfg.nconfig << ",\n"
         << "  \"repeats\":" << options.repeats << ",\n"
         << "  \"active_electron\":" << options.active_electron << ",\n"
         << "  \"model_and_workspace_setup_seconds\":" << setup_seconds << ",\n"
         << "  \"baseline_peak_resident_mib\":" << baseline_peak_mib << ",\n"
         << "  \"host\":" << jsonString(hostName()) << ",\n"
         << "  \"hardware_concurrency\":" << std::thread::hardware_concurrency() << ",\n"
         << "  \"affinity_cpu_count\":" << affinityCpuCount() << ",\n"
         << "  \"thread_environment\":{"
         << "\"OMP_NUM_THREADS\":" << jsonString(environmentValue("OMP_NUM_THREADS")) << ','
         << "\"OMP_MAX_ACTIVE_LEVELS\":"
         << jsonString(environmentValue("OMP_MAX_ACTIVE_LEVELS")) << ','
         << "\"OMP_PROC_BIND\":" << jsonString(environmentValue("OMP_PROC_BIND")) << ','
         << "\"OMP_PLACES\":" << jsonString(environmentValue("OMP_PLACES")) << ','
         << "\"OPENBLAS_NUM_THREADS\":" << jsonString(environmentValue("OPENBLAS_NUM_THREADS")) << ','
         << "\"MKL_NUM_THREADS\":" << jsonString(environmentValue("MKL_NUM_THREADS")) << ','
         << "\"BLIS_NUM_THREADS\":" << jsonString(environmentValue("BLIS_NUM_THREADS")) << ','
         << "\"VECLIB_MAXIMUM_THREADS\":"
         << jsonString(environmentValue("VECLIB_MAXIMUM_THREADS")) << "},\n"
         << "  \"notes\":["
         << jsonString("direct timings include coordinate packing into warmed workspaces") << ','
         << jsonString("oracle timings include native result allocation") << ','
         << jsonString("no absolute performance threshold is applied") << "],\n"
         << "  \"modes\":[\n";
  for (std::size_t index = 0; index < modes.size(); ++index)
  {
    writeMode(output, modes[index]);
    output << (index + 1 == modes.size() ? "\n" : ",\n");
  }
  output << "  ],\n  \"sink\":" << benchmark_sink << "\n}\n";
}

} // namespace

/** Load one export, time every optimized/oracle mode, and write one manifest. */
int main(int argc, char** argv)
{
  try
  {
    const Options options = parseOptions(argc, argv);
    auto setup_start      = Clock::now();
    pf::PsiFormer model(options.parameter_path, options.configuration_path);
    const qmcplusplus::psiformer::ModelShape shape{
        model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet, model.dim, model.heads, 4};
    const auto plan = qmcplusplus::psiformer::PsiFormerExecutionPlan::fromParameters(model.p, shape);

    pf::DirectValueExecutor value_executor(model, plan);
    pf::DirectSpatialExecutor spatial_executor(model, value_executor, plan);
    pf::DirectBatchExecutor batch_executor(value_executor, spatial_executor);
    pf::DirectScoreExecutor score_executor(model, plan);
    pf::DirectKineticExecutor kinetic_executor(model, plan);
    auto value_workspace   = value_executor.makeWorkspace();
    auto full_workspace    = spatial_executor.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
    auto active_workspace  = spatial_executor.makeWorkspace(pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT);
    auto score_workspace   = score_executor.makeWorkspace();
    auto kinetic_workspace = kinetic_executor.makeWorkspace();

    const double setup_seconds = elapsedSeconds(setup_start);
    const double baseline_peak_mib = peakResidentMiB();
    const std::size_t configuration_count = options.configuration_limit == 0
        ? model.cfg.nconfig
        : std::min(options.configuration_limit, model.cfg.nconfig);
    if (configuration_count == 0)
      throw std::runtime_error("no electron configurations selected");
    if (options.active_electron >= model.ne)
      throw std::out_of_range("--active-electron is outside the imported model");
    const std::size_t kinetic_count = std::min(options.kinetic_configurations, configuration_count);

    std::vector<pf::Tensor> configurations;
    configurations.reserve(configuration_count);
    for (std::size_t configuration = 0; configuration < configuration_count; ++configuration)
      configurations.push_back(model.cfg.configuration(configuration));
    auto positions = [&](std::size_t configuration) {
      const pf::Tensor& tensor = configurations[configuration % configuration_count];
      return pf::GeometryPositionView::interleaved(tensor.x.data(), model.ne);
    };

    const auto value_request = request(pf::SpatialDerivativeRequest::NONE,
                                       pf::ParameterDerivativeRequest::NONE);
    const auto full_request = request(pf::SpatialDerivativeRequest::FULL_VGL,
                                      pf::ParameterDerivativeRequest::NONE);
    const auto active_request = request(pf::SpatialDerivativeRequest::ACTIVE_ELECTRON_GRADIENT,
                                        pf::ParameterDerivativeRequest::NONE,
                                        static_cast<int>(options.active_electron));
    const auto score_request = request(pf::SpatialDerivativeRequest::NONE,
                                       pf::ParameterDerivativeRequest::LOG_ONLY);
    const auto kinetic_request = request(pf::SpatialDerivativeRequest::FULL_VGL,
                                         pf::ParameterDerivativeRequest::LOG_AND_KINETIC);

    std::vector<ModeSummary> modes;
    auto add_pair = [&](ModeSummary direct, ModeSummary oracle, double tolerance) {
      validatePair(direct, oracle, tolerance);
      modes.push_back(std::move(direct));
      modes.push_back(std::move(oracle));
    };

    add_pair(
        benchmarkMode("value", "direct", 1, configuration_count,
                      value_workspace->vectorStorageBytes(), 3 * sizeof(double), options.repeats,
                      [&](std::size_t configuration) {
                        value_workspace->setPositions(positions(configuration));
                        const auto result = value_executor.evaluate(*value_workspace);
                        return Observation{result.sign, result.logabs,
                                           result.sign + result.logabs + result.value};
                      }),
        benchmarkMode("value", "oracle", 1, configuration_count, 0,
                      3 * sizeof(double), options.repeats,
                      [&](std::size_t configuration) {
                        return observeNative(model.evaluate(
                            configurations[configuration % configuration_count], value_request));
                      }),
        3.0e-10);

    const std::size_t full_output_doubles = 3 + 5 * model.ne;
    add_pair(
        benchmarkMode("full_vgl", "direct", 1, configuration_count,
                      full_workspace->vectorStorageBytes(), full_output_doubles * sizeof(double),
                      options.repeats,
                      [&](std::size_t configuration) {
                        full_workspace->setPositions(positions(configuration));
                        const auto result = spatial_executor.evaluateFull(*full_workspace);
                        const double checksum = result.sign + result.logabs + result.value +
                            endpointChecksum(result.gradient) + endpointChecksum(result.lap_log) +
                            endpointChecksum(result.lap_ratio);
                        return Observation{result.sign, result.logabs, checksum};
                      }),
        benchmarkMode("full_vgl", "oracle", 1, configuration_count, 0,
                      full_output_doubles * sizeof(double), options.repeats,
                      [&](std::size_t configuration) {
                        return observeNative(model.evaluate(
                            configurations[configuration % configuration_count], full_request));
                      }),
        3.0e-7);

    add_pair(
        benchmarkMode("active_gradient", "direct", 1, configuration_count,
                      active_workspace->vectorStorageBytes(), 6 * sizeof(double), options.repeats,
                      [&](std::size_t configuration) {
                        active_workspace->setPositions(positions(configuration));
                        const auto result = spatial_executor.evaluateActive(
                            *active_workspace, options.active_electron);
                        return Observation{result.sign, result.logabs,
                                           result.sign + result.logabs + result.value +
                                               endpointChecksum(result.gradient)};
                      }),
        benchmarkMode("active_gradient", "oracle", 1, configuration_count, 0,
                      6 * sizeof(double), options.repeats,
                      [&](std::size_t configuration) {
                        return observeNative(model.evaluate(
                            configurations[configuration % configuration_count], active_request));
                      }),
        2.0e-7);

    add_pair(
        benchmarkMode("score", "direct", 1, configuration_count,
                      score_workspace->vectorStorageBytes(),
                      (3 + model.p.size()) * sizeof(double), options.repeats,
                      [&](std::size_t configuration) {
                        score_workspace->setPositions(positions(configuration));
                        const auto result = score_executor.evaluate(*score_workspace);
                        return Observation{result.sign, result.logabs,
                                           result.sign + result.logabs + result.value +
                                               endpointChecksum(result.parameter_score)};
                      }),
        benchmarkMode("score", "oracle", 1, configuration_count, 0,
                      (3 + model.p.size()) * sizeof(double), options.repeats,
                      [&](std::size_t configuration) {
                        return observeNative(model.evaluate(
                            configurations[configuration % configuration_count], score_request));
                      }),
        3.0e-8);

    const std::size_t kinetic_output_doubles = 3 + 5 * model.ne + 2 * model.p.size();
    add_pair(
        benchmarkMode("score_and_kinetic", "direct", 1, kinetic_count,
                      kinetic_workspace->vectorStorageBytes(), kinetic_output_doubles * sizeof(double),
                      options.repeats,
                      [&](std::size_t configuration) {
                        kinetic_workspace->setPositions(positions(configuration));
                        const auto result = kinetic_executor.evaluate(*kinetic_workspace);
                        const double checksum = result.sign + result.logabs + result.value +
                            endpointChecksum(result.gradient) + endpointChecksum(result.lap_log) +
                            endpointChecksum(result.lap_ratio) + endpointChecksum(result.parameter_score) +
                            endpointChecksum(result.kinetic_parameter_response);
                        return Observation{result.sign, result.logabs, checksum};
                      }),
        benchmarkMode("score_and_kinetic", "oracle", 1, kinetic_count, 0,
                      kinetic_output_doubles * sizeof(double), options.repeats,
                      [&](std::size_t configuration) {
                        return observeNative(model.evaluate(
                            configurations[configuration % configuration_count], kinetic_request));
                      }),
        3.0e-6);

    // Batch entries exercise the production crowd boundary for the three modes
    // currently supported by DirectBatchExecutor.  Score/kinetic crowd work is
    // reported by scalar workspaces until a true grouped reverse pass exists.
    for (const std::size_t batch_size : options.batch_sizes)
    {
      if (batch_size > configuration_count)
        continue;
      for (const auto mode : {pf::DirectBatchMode::VALUE_ONLY,
                              pf::DirectBatchMode::FULL_VGL,
                              pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT})
      {
        auto workspace = batch_executor.makeWorkspace();
        workspace->resize(mode, batch_size);
        for (std::size_t configuration = 0; configuration < batch_size; ++configuration)
          for (std::size_t electron = 0; electron < model.ne; ++electron)
            for (std::size_t dimension = 0; dimension < 3; ++dimension)
              workspace->setPosition(configuration, electron, dimension,
                                     configurations[configuration].x[3 * electron + dimension]);

        const std::size_t workspace_bytes = workspace->vectorStorageBytes();
        if (mode == pf::DirectBatchMode::VALUE_ONLY)
          modes.push_back(benchmarkMode(
              "value_batch", "direct", batch_size, 1, workspace_bytes,
              3 * sizeof(double), options.repeats, [&](std::size_t) {
                const auto result = batch_executor.evaluateValues(*workspace);
                return Observation{result.sign[0], result.logabs[0],
                                   result.sign[0] + result.logabs[0] + result.value[0]};
              }));
        else if (mode == pf::DirectBatchMode::FULL_VGL)
          modes.push_back(benchmarkMode(
              "full_vgl_batch", "direct", batch_size, 1, workspace_bytes,
              full_output_doubles * sizeof(double), options.repeats, [&](std::size_t) {
                const auto result = batch_executor.evaluateFull(*workspace);
                const double checksum = result.sign[0] + result.logabs[0] + result.value[0] +
                    result.gradient[0] + result.gradient[result.gradient_stride - 1] +
                    result.lap_log[0] + result.lap_ratio[0];
                return Observation{result.sign[0], result.logabs[0], checksum};
              }));
        else
        {
          std::vector<std::size_t> active_electrons(batch_size, options.active_electron);
          modes.push_back(benchmarkMode(
              "active_gradient_batch", "direct", batch_size, 1, workspace_bytes,
              6 * sizeof(double), options.repeats, [&](std::size_t) {
                const auto result = batch_executor.evaluateActive(*workspace, active_electrons.data());
                const double checksum = result.sign[0] + result.logabs[0] + result.value[0] +
                    result.gradient[0] + result.gradient[result.gradient_stride - 1];
                return Observation{result.sign[0], result.logabs[0], checksum};
              }));
        }
      }
    }

    if (options.output_path.empty())
      writeReport(std::cout, options, model, setup_seconds, baseline_peak_mib, modes);
    else
    {
      std::ofstream output(options.output_path);
      if (!output)
        throw std::runtime_error("could not open benchmark output " + options.output_path);
      writeReport(output, options, model, setup_seconds, baseline_peak_mib, modes);
      std::cerr << "Wrote " << options.output_path << '\n';
    }
    return EXIT_SUCCESS;
  }
  catch (const std::exception& error)
  {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
