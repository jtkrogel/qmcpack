//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file benchmark_psiformer_native.cpp
 * @brief Non-gating benchmark for native PsiFormer evaluation request modes.
 *
 * The benchmark loads the same exported HDF5 model and configurations used by
 * the standalone/JAX comparison driver, but times each request mode separately.
 * It intentionally reports timings rather than imposing performance thresholds
 * on correctness CI.
 */

#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"

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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/resource.h>

namespace
{
using Clock = std::chrono::steady_clock;

// Make each returned result observable without traversing the large parameter
// vectors and folding that traversal into the evaluator timing.
volatile double benchmark_sink = 0.0;

/** Command-line settings for one benchmark invocation. */
struct Options
{
  std::string parameter_path;
  std::string configuration_path;
  std::string output_path;
  int repeats                         = 5;
  std::size_t configuration_limit    = 0;
  std::size_t kinetic_configurations = 1;
};

/** Logical sizes of the data returned by one evaluator request. */
struct OutputSizes
{
  std::size_t gradient                        = 0;
  std::size_t active_gradient                 = 0;
  std::size_t lap_log                         = 0;
  std::size_t lap_ratio                       = 0;
  std::size_t potential                       = 0;
  std::size_t parameter_gradient              = 0;
  std::size_t local_energy_parameter_gradient = 0;
  std::size_t total_double_count              = 0;
  std::size_t logical_numeric_bytes           = 0;
  bool has_local_energy                       = false;
};

/** Timings and result metadata collected for one request mode. */
struct ModeSummary
{
  std::string name;
  double warmup_seconds = 0.0;
  std::vector<double> raw_seconds;
  std::size_t configurations_per_repeat = 0;
  double peak_resident_mib_after_mode    = 0.0;
  double sign                            = 0.0;
  double logabs                          = 0.0;
  double value                           = 0.0;
  OutputSizes output_sizes;
};

/** Print usage and terminate through the caller's normal error handling. */
[[noreturn]] void usage(const char* executable, const std::string& error = {})
{
  if (!error.empty())
    std::cerr << "error: " << error << '\n';
  std::cerr << "usage: " << executable
            << " PARAMETERS.h5 CONFIGURATIONS.h5 [--repeats N]"
               " [--configuration-limit N] [--kinetic-configurations N]"
               " [--output FILE.json]\n";
  std::exit(EXIT_FAILURE);
}

/** Parse the deliberately small option surface used by this developer benchmark. */
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

/** Return elapsed wall-clock seconds since a recorded start time. */
double elapsedSeconds(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/** Report cumulative process peak resident memory in MiB on Linux. */
double peakResidentMiB()
{
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    throw std::runtime_error("getrusage failed");
  return static_cast<double>(usage.ru_maxrss) / 1024.0;
}

/** Compute the median without changing the raw samples retained for JSON output. */
double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
}

/** Prevent the compiler from treating an evaluator result as unused. */
void consume(const pf::Result& result)
{
  double sample = result.sign + result.logabs + result.value;
  if (result.has_local_energy)
    sample += result.local_energy;

  auto consume_front = [&sample](const std::vector<double>& values) {
    if (!values.empty())
      sample += values.front();
  };
  consume_front(result.gradient);
  consume_front(result.active_gradient);
  consume_front(result.lap_log);
  consume_front(result.lap_ratio);
  consume_front(result.potential);
  consume_front(result.param_gradient);
  consume_front(result.local_energy_param_gradient);
  benchmark_sink = benchmark_sink + sample;
}

/** Describe returned logical data without conflating vector capacity with useful output. */
OutputSizes measureOutputSizes(const pf::Result& result)
{
  OutputSizes sizes;
  sizes.gradient                        = result.gradient.size();
  sizes.active_gradient                 = result.active_gradient.size();
  sizes.lap_log                         = result.lap_log.size();
  sizes.lap_ratio                       = result.lap_ratio.size();
  sizes.potential                       = result.potential.size();
  sizes.parameter_gradient              = result.param_gradient.size();
  sizes.local_energy_parameter_gradient = result.local_energy_param_gradient.size();
  sizes.has_local_energy                = result.has_local_energy;

  // sign, logabs, and value are always returned; local_energy is meaningful
  // only when has_local_energy is true.
  sizes.total_double_count = 3 + (sizes.has_local_energy ? 1 : 0) + sizes.gradient + sizes.active_gradient +
      sizes.lap_log + sizes.lap_ratio + sizes.potential + sizes.parameter_gradient +
      sizes.local_energy_parameter_gradient;
  sizes.logical_numeric_bytes = sizes.total_double_count * sizeof(double);
  return sizes;
}

/** Require output fields to match the advertised evaluator request. */
void validateOutputContract(const std::string& name,
                            const pf::Result& result,
                            std::size_t electron_count,
                            std::size_t parameter_count)
{
  const bool value_only   = name == "value_only" || name == "direct_value_only";
  const bool full_vgl     = name == "full_vgl";
  const bool score_only   = name == "score_only";
  const bool kinetic      = name == "score_and_kinetic";
  const bool legacy_score = name == "historical_full_jet_score";
  const bool historical   = name == "historical_standalone_full";

  const std::size_t expected_gradient =
      (full_vgl || kinetic || legacy_score || historical) ? 3 * electron_count : 0;
  const std::size_t expected_laplacian =
      (full_vgl || kinetic || legacy_score || historical) ? electron_count : 0;
  const std::size_t expected_score =
      (score_only || kinetic || legacy_score || historical) ? parameter_count : 0;
  const std::size_t expected_kinetic   = (kinetic || historical) ? parameter_count : 0;
  const std::size_t expected_potential = (legacy_score || historical) ? 3 : 0;

  if ((!value_only && !full_vgl && !score_only && !kinetic && !legacy_score && !historical) ||
      result.gradient.size() != expected_gradient || !result.active_gradient.empty() ||
      result.lap_log.size() != expected_laplacian || result.lap_ratio.size() != expected_laplacian ||
      result.param_gradient.size() != expected_score ||
      result.local_energy_param_gradient.size() != expected_kinetic || result.potential.size() != expected_potential ||
      result.has_local_energy != (legacy_score || historical))
    throw std::runtime_error("unexpected PsiFormer output fields for benchmark mode " + name);
}

/** Time one request mode after one separately reported warmup evaluation. */
template<class Evaluate>
ModeSummary benchmarkMode(const std::string& name,
                          Evaluate&& evaluate,
                          std::size_t configurations,
                          int repeats,
                          std::size_t electron_count,
                          std::size_t parameter_count)
{
  ModeSummary summary;
  summary.name                       = name;
  summary.configurations_per_repeat = configurations;

  auto start        = Clock::now();
  pf::Result warmup = evaluate(0);
  summary.warmup_seconds = elapsedSeconds(start);
  consume(warmup);

  validateOutputContract(name, warmup, electron_count, parameter_count);
  summary.sign         = warmup.sign;
  summary.logabs       = warmup.logabs;
  summary.value        = warmup.value;
  summary.output_sizes = measureOutputSizes(warmup);

  summary.raw_seconds.reserve(repeats);
  for (int repeat = 0; repeat < repeats; ++repeat)
  {
    start = Clock::now();
    for (std::size_t configuration = 0; configuration < configurations; ++configuration)
    {
      const pf::Result result = evaluate(configuration);
      consume(result);
    }
    summary.raw_seconds.push_back(elapsedSeconds(start));
  }
  summary.peak_resident_mib_after_mode = peakResidentMiB();

  std::cerr << std::left << std::setw(29) << name << " median_seconds_per_configuration=" << std::scientific
            << median(summary.raw_seconds) / configurations << " peak_mib=" << std::fixed << std::setprecision(3)
            << summary.peak_resident_mib_after_mode << '\n';
  return summary;
}

/** Construct an explicit request with standalone Hamiltonian work disabled. */
pf::EvaluationRequest makeRequest(pf::ParameterDerivativeRequest parameter_derivatives,
                                  pf::SpatialDerivativeRequest spatial_derivatives)
{
  pf::EvaluationRequest request;
  request.parameter_derivatives  = parameter_derivatives;
  request.spatial_derivatives    = spatial_derivatives;
  request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  return request;
}

/** Ensure all modes evaluate the same wavefunction scalar during warmup. */
void validateCommonValue(const std::vector<ModeSummary>& summaries)
{
  const ModeSummary& reference = summaries.front();
  auto close = [](double first, double second) {
    const double scale = std::max({1.0, std::abs(first), std::abs(second)});
    return std::abs(first - second) <= 64 * std::numeric_limits<double>::epsilon() * scale;
  };
  for (const ModeSummary& summary : summaries)
    if (summary.sign != reference.sign || !close(summary.logabs, reference.logabs) ||
        !close(summary.value, reference.value))
      throw std::runtime_error("request modes disagree on the warmup wavefunction value");
}

/** Write one output-size object in stable JSON field order. */
void writeOutputSizes(std::ostream& output, const OutputSizes& sizes)
{
  output << "{\"gradient\":" << sizes.gradient << ",\"active_gradient\":" << sizes.active_gradient
         << ",\"lap_log\":" << sizes.lap_log << ",\"lap_ratio\":" << sizes.lap_ratio
         << ",\"potential\":" << sizes.potential << ",\"parameter_gradient\":"
         << sizes.parameter_gradient << ",\"local_energy_parameter_gradient\":"
         << sizes.local_energy_parameter_gradient << ",\"has_local_energy\":"
         << (sizes.has_local_energy ? "true" : "false") << ",\"total_double_count\":"
         << sizes.total_double_count << ",\"logical_numeric_bytes\":" << sizes.logical_numeric_bytes << '}';
}

/** Write timing samples and result metadata for one mode as JSON. */
void writeModeSummary(std::ostream& output, const ModeSummary& summary)
{
  const double median_seconds = median(summary.raw_seconds);
  output << "{\n      \"warmup_seconds\":" << summary.warmup_seconds << ",\n"
         << "      \"raw_seconds\":[";
  for (std::size_t sample = 0; sample < summary.raw_seconds.size(); ++sample)
  {
    if (sample != 0)
      output << ',';
    output << summary.raw_seconds[sample];
  }
  output << "],\n      \"configurations_per_repeat\":" << summary.configurations_per_repeat << ",\n"
         << "      \"median_seconds\":" << median_seconds << ",\n"
         << "      \"median_seconds_per_configuration\":"
         << median_seconds / summary.configurations_per_repeat << ",\n"
         << "      \"peak_resident_mib_after_mode\":" << summary.peak_resident_mib_after_mode << ",\n"
         << "      \"output_sizes\":";
  writeOutputSizes(output, summary.output_sizes);
  output << "\n    }";
}

/** Write the complete, machine-readable benchmark report. */
void writeReport(std::ostream& output,
                 const Options& options,
                 const pf::PsiFormer& model,
                 double model_load_seconds,
                 double baseline_peak_resident_mib,
                 const std::vector<ModeSummary>& summaries)
{
  output << std::setprecision(12) << "{\n"
         << "  \"benchmark\":\"QMCPACK/PsiFormerNative request modes\",\n"
         << "  \"parameter_count\":" << model.p.size() << ",\n"
         << "  \"electron_count\":" << model.ne << ",\n"
         << "  \"available_configuration_count\":" << model.cfg.nconfig << ",\n"
         << "  \"repeats\":" << options.repeats << ",\n"
         << "  \"model_load_seconds\":" << model_load_seconds << ",\n"
         << "  \"baseline_peak_resident_mib\":" << baseline_peak_resident_mib << ",\n"
         << "  \"allocation_counting\":\"not instrumented; peak RSS and logical output bytes are reported\",\n"
         << "  \"modes\":{\n";
  for (std::size_t mode = 0; mode < summaries.size(); ++mode)
  {
    output << "    \"" << summaries[mode].name << "\":";
    writeModeSummary(output, summaries[mode]);
    output << (mode + 1 == summaries.size() ? "\n" : ",\n");
  }
  output << "  },\n  \"sink\":" << benchmark_sink << "\n}\n";
}

} // namespace

/** Load one model, benchmark every evaluator mode, and emit a JSON report. */
int main(int argc, char** argv)
{
  try
  {
    const Options options = parseOptions(argc, argv);

    auto start = Clock::now();
    pf::PsiFormer model(options.parameter_path, options.configuration_path);
    const qmcplusplus::psiformer::ModelShape model_shape{
        model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet, model.dim, model.heads, 4};
    const auto execution_plan =
        qmcplusplus::psiformer::PsiFormerExecutionPlan::fromParameters(model.p, model_shape);
    pf::DirectValueExecutor direct_value_executor(model, execution_plan);
    std::unique_ptr<pf::DirectValueWorkspace> direct_value_workspace = direct_value_executor.makeWorkspace();
    const double model_load_seconds          = elapsedSeconds(start);
    const double baseline_peak_resident_mib  = peakResidentMiB();
    const std::size_t regular_configurations = options.configuration_limit == 0
        ? model.cfg.nconfig
        : std::min(options.configuration_limit, model.cfg.nconfig);
    const std::size_t kinetic_configurations = std::min(options.kinetic_configurations, regular_configurations);
    if (regular_configurations == 0)
      throw std::runtime_error("no electron configurations selected");

    const pf::EvaluationRequest value_only =
        makeRequest(pf::ParameterDerivativeRequest::NONE, pf::SpatialDerivativeRequest::NONE);
    const pf::EvaluationRequest full_vgl =
        makeRequest(pf::ParameterDerivativeRequest::NONE, pf::SpatialDerivativeRequest::FULL_VGL);
    const pf::EvaluationRequest score_only =
        makeRequest(pf::ParameterDerivativeRequest::LOG_ONLY, pf::SpatialDerivativeRequest::NONE);
    const pf::EvaluationRequest score_and_kinetic =
        makeRequest(pf::ParameterDerivativeRequest::LOG_AND_KINETIC, pf::SpatialDerivativeRequest::FULL_VGL);
    pf::EvaluationRequest historical_score;
    historical_score.parameter_derivatives = pf::ParameterDerivativeRequest::LOG_ONLY;

    std::vector<ModeSummary> summaries;
    summaries.push_back(benchmarkMode(
        "direct_value_only",
        [&](std::size_t configuration) {
          const pf::Tensor positions = model.cfg.configuration(configuration);
          direct_value_workspace->setPositions(
              pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
          const pf::DirectValueResult direct = direct_value_executor.evaluate(*direct_value_workspace);
          pf::Result result;
          result.sign   = direct.sign;
          result.logabs = direct.logabs;
          result.value  = direct.value;
          return result;
        },
        regular_configurations, options.repeats, model.ne, model.p.size()));
    summaries.push_back(benchmarkMode("value_only",
                                      [&](std::size_t configuration) {
                                        return model.evaluate(model.cfg.configuration(configuration), value_only);
                                      },
                                      regular_configurations, options.repeats, model.ne, model.p.size()));
    summaries.push_back(benchmarkMode("full_vgl",
                                      [&](std::size_t configuration) {
                                        return model.evaluate(model.cfg.configuration(configuration), full_vgl);
                                      },
                                      regular_configurations, options.repeats, model.ne, model.p.size()));
    summaries.push_back(benchmarkMode("score_only",
                                      [&](std::size_t configuration) {
                                        return model.evaluate(model.cfg.configuration(configuration), score_only);
                                      },
                                      regular_configurations, options.repeats, model.ne, model.p.size()));
    summaries.push_back(benchmarkMode(
        "historical_full_jet_score",
        [&](std::size_t configuration) {
          return model.evaluate(model.cfg.configuration(configuration), historical_score);
        },
        regular_configurations, options.repeats, model.ne, model.p.size()));
    summaries.push_back(benchmarkMode(
        "score_and_kinetic",
        [&](std::size_t configuration) { return model.evaluate(model.cfg.configuration(configuration), score_and_kinetic); },
        kinetic_configurations, options.repeats, model.ne, model.p.size()));
    summaries.push_back(benchmarkMode(
        "historical_standalone_full",
        [&](std::size_t configuration) { return model.evaluate(model.cfg.configuration(configuration)); },
        kinetic_configurations, options.repeats, model.ne, model.p.size()));

    validateCommonValue(summaries);

    if (options.output_path.empty())
      writeReport(std::cout, options, model, model_load_seconds, baseline_peak_resident_mib, summaries);
    else
    {
      std::ofstream output(options.output_path);
      if (!output)
        throw std::runtime_error("could not open benchmark output " + options.output_path);
      writeReport(output, options, model, model_load_seconds, baseline_peak_resident_mib, summaries);
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
