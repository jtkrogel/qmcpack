//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file benchmark_psiformer_memory.cpp
 * @brief Fresh-process memory attribution for production PsiFormer evaluation paths.
 *
 * Each invocation measures exactly one scenario.  Use the companion Python
 * launcher to generate the deterministic LiH fixture in a separate process and
 * invoke this executable once per scenario.  The benchmark is deliberately not
 * a CTest target: RSS depends on the allocator and host, so its JSON report is a
 * diagnostic record rather than a pass/fail performance threshold.
 */

#include <stdexcept>

// Reuse the deterministic fixture recipe without linking a Catch2 runner.  The
// prefix option leaves the utility's REQUIRE spelling available for this small
// throwing replacement, so HDF5 writes are still checked.
#define CATCH_CONFIG_PREFIX_ALL
#define REQUIRE(expression)                                                                                         \
  do                                                                                                                \
  {                                                                                                                 \
    if (!(expression))                                                                                              \
      throw std::runtime_error("generated PsiFormer fixture write failed: " #expression);                           \
  } while (false)
#include "psiformer_test_utils.h"
#undef REQUIRE

#include "config.h"
#include "git-rev.h"
#ifdef PSIFORMER_MEMORY_COMPONENT
#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "ResourceCollection.h"
#else
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerBatchExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerKineticExecutor.h"
#endif

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif

#ifdef PSIFORMER_MEMORY_COMPONENT
namespace qmcplusplus::testing
{
/** Expose the production-owned numeric scratch accounting to this benchmark. */
class TestPsiFormerWF
{
public:
  static PsiFormerWorkspaceDiagnostics directWorkspaceDiagnostics(const PsiFormerWF& component)
  {
    return component.directWorkspaceDiagnosticsForTesting();
  }
};
} // namespace qmcplusplus::testing
#endif

namespace
{
#ifdef PSIFORMER_MEMORY_COMPONENT
using qmcplusplus::ParticleSet;
using qmcplusplus::PsiFormerWF;
using qmcplusplus::RefVector;
using qmcplusplus::RefVectorWithLeader;
using qmcplusplus::ResourceCollection;
using qmcplusplus::ResourceCollectionTeamLock;
using qmcplusplus::SimulationCell;
using qmcplusplus::WaveFunctionComponent;
#else
using qmcplusplus::psiformer::ModelShape;
#endif
using qmcplusplus::testing::psiformer::GeneratedFiles;
using qmcplusplus::testing::psiformer::Geometry;
using qmcplusplus::testing::psiformer::generateFiles;
using qmcplusplus::testing::psiformer::makeGeometry;

#define PSIFORMER_STRINGIFY_DETAIL(value) #value
#define PSIFORMER_STRINGIFY(value) PSIFORMER_STRINGIFY_DETAIL(value)

volatile double benchmark_sink = 0.0;

/** Shape and payload metadata read without materializing the model arrays. */
struct ModelMetadata
{
  std::size_t parameters     = 0;
  std::size_t configurations = 0;
  std::size_t electrons      = 0;
  std::size_t nuclei         = 0;
  std::size_t spin_up        = 0;
  std::size_t spin_down      = 0;
  std::size_t configuration_numeric_bytes = 0;

  /// Count the two persistent parameter payloads plus imported geometry arrays.
  std::size_t accountedModelNumericBytes() const
  {
    // Parameters keeps one canonical flat vector and one value tensor per
    // named leaf.  Layout strings, maps, vector objects, and allocator metadata
    // are intentionally left to RSS rather than guessed here.
    return 2 * parameters * sizeof(double) + configuration_numeric_bytes;
  }
};

/** User controls for one isolated memory scenario. */
struct Options
{
  std::string scenario;
  std::string parameter_path;
  std::string configuration_path;
  std::string output_path;
  std::size_t walkers = 4;
  int warm_calls      = 1;
};

/** One phase boundary with process counters and explicitly accounted storage. */
struct Phase
{
  std::string name;
  std::uint64_t current_rss_bytes = 0;
  std::uint64_t peak_rss_bytes    = 0;
  std::uint64_t accounted_bytes   = 0;
  std::map<std::string, std::uint64_t> accounted_categories;
};

/** Print accepted forms before terminating on invalid command-line input. */
[[noreturn]] void usage(const char* executable, const std::string& error = {})
{
  if (!error.empty())
    std::cerr << "error: " << error << '\n';
  std::cerr
      << "usage:\n  " << executable << " --generate-fixture SYSTEM DIRECTORY\n  "
      << executable
      << " --scenario NAME --parameters FILE --configuration FILE [--walkers N]"
         " [--warm-calls N] [--output FILE]\n"
         "scenarios: model_load, clone_population, value, full_vgl, active_gradient,"
         " crowd_value, crowd_full_vgl, crowd_active_gradient, score, score_and_kinetic\n";
  std::exit(EXIT_FAILURE);
}

/** Parse one scenario invocation; fixture generation is handled before this call. */
Options parseOptions(int argc, char** argv)
{
  Options options;
  for (int argument = 1; argument < argc; ++argument)
  {
    const std::string key = argv[argument];
    if (argument + 1 >= argc)
      usage(argv[0], "missing value after " + key);
    const std::string value = argv[++argument];
    try
    {
      if (key == "--scenario")
        options.scenario = value;
      else if (key == "--parameters")
        options.parameter_path = value;
      else if (key == "--configuration")
        options.configuration_path = value;
      else if (key == "--walkers")
        options.walkers = std::stoull(value);
      else if (key == "--warm-calls")
        options.warm_calls = std::stoi(value);
      else if (key == "--output")
        options.output_path = value;
      else
        usage(argv[0], "unknown option " + key);
    }
    catch (const std::exception&)
    {
      usage(argv[0], "invalid value for " + key + ": " + value);
    }
  }

  if (options.scenario.empty() || options.parameter_path.empty() || options.configuration_path.empty())
    usage(argv[0], "scenario, parameter file, and configuration file are required");
  if (options.walkers == 0 || options.warm_calls < 0)
    usage(argv[0], "walker count must be positive and warm-call count nonnegative");
  return options;
}

/** Return current resident bytes from Linux procfs, or zero when unavailable. */
std::uint64_t currentResidentBytes()
{
#ifdef __linux__
  std::ifstream statm("/proc/self/statm");
  std::uint64_t virtual_pages = 0;
  std::uint64_t resident_pages = 0;
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size > 0 && statm >> virtual_pages >> resident_pages)
    return resident_pages * static_cast<std::uint64_t>(page_size);
#endif
  return 0;
}

/** Return the cumulative process peak resident byte count. */
std::uint64_t peakResidentBytes()
{
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    throw std::runtime_error("getrusage failed");
#ifdef __APPLE__
  return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
  return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
#endif
}

/** Return a JSON-safe quoted string. */
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
    else if (character == '\r')
      output << "\\r";
    else if (character == '\t')
      output << "\\t";
    else
      output << character;
  }
  output << '"';
  return output.str();
}

/** Return an environment variable or an explicit unset marker. */
std::string environmentValue(const char* name)
{
  const char* value = std::getenv(name);
  return value ? value : "<unset>";
}

/** Return the current host name for benchmark provenance. */
std::string hostName()
{
  char hostname[256]{};
  return gethostname(hostname, sizeof(hostname) - 1) == 0 ? hostname : "<unknown>";
}

/** Count CPUs allowed by the process affinity mask when Linux exposes it. */
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

/** Read a rank-one dataset extent without copying its payload. */
std::size_t datasetElementCount(hid_t file, const char* path)
{
  const hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
  if (dataset < 0)
    throw std::runtime_error(std::string("missing HDF5 dataset ") + path);
  const hid_t space = H5Dget_space(dataset);
  const int rank    = H5Sget_simple_extent_ndims(space);
  std::vector<hsize_t> extents(rank);
  H5Sget_simple_extent_dims(space, extents.data(), nullptr);
  H5Sclose(space);
  H5Dclose(dataset);
  std::size_t count = 1;
  for (const hsize_t extent : extents)
    count *= extent;
  return count;
}

/** Read one signed scalar file attribute. */
std::size_t integerAttribute(hid_t file, const char* name)
{
  const hid_t attribute = H5Aopen(file, name, H5P_DEFAULT);
  if (attribute < 0)
    throw std::runtime_error(std::string("missing HDF5 attribute ") + name);
  std::int64_t value = 0;
  const herr_t status = H5Aread(attribute, H5T_NATIVE_LLONG, &value);
  H5Aclose(attribute);
  if (status < 0 || value < 0)
    throw std::runtime_error(std::string("invalid HDF5 attribute ") + name);
  return static_cast<std::size_t>(value);
}

/** Inspect the export shapes while leaving the large arrays in the fixture file. */
ModelMetadata readMetadata(const Options& options)
{
  ModelMetadata metadata;
  hid_t file = H5Fopen(options.parameter_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
    throw std::runtime_error("unable to inspect parameter file " + options.parameter_path);
  metadata.parameters = datasetElementCount(file, "/values");
  H5Fclose(file);

  file = H5Fopen(options.configuration_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
    throw std::runtime_error("unable to inspect configuration file " + options.configuration_path);
  metadata.spin_up   = integerAttribute(file, "n_up");
  metadata.spin_down = integerAttribute(file, "n_down");
  metadata.electrons = metadata.spin_up + metadata.spin_down;
  const std::size_t electron_values = datasetElementCount(file, "/electron_positions");
  const std::size_t nucleus_values  = datasetElementCount(file, "/nuclear_positions");
  const std::size_t charge_values   = datasetElementCount(file, "/nuclear_charges");
  metadata.configurations = electron_values / (3 * metadata.electrons);
  metadata.nuclei         = charge_values;
  metadata.configuration_numeric_bytes =
      (electron_values + nucleus_values + charge_values) * sizeof(double);
  H5Fclose(file);
  return metadata;
}

/** Collect phase snapshots and emit one self-describing JSON document. */
class MemoryReport
{
public:
  MemoryReport(Options options, ModelMetadata metadata)
      : options_(std::move(options)), metadata_(metadata)
  {}

  /// Record RSS and peak RSS at one explicit lifetime boundary.
  void snapshot(std::string name, std::map<std::string, std::uint64_t> categories = {})
  {
    std::uint64_t accounted = 0;
    for (const auto& [category, bytes] : categories)
      accounted += bytes;
    const std::uint64_t current = currentResidentBytes();
    const std::uint64_t peak    = std::max(current, peakResidentBytes());
    phases_.push_back({std::move(name), current, peak, accounted, std::move(categories)});
  }

  /// Emit the report to stdout or the requested file.
  void write() const
  {
    if (options_.output_path.empty())
      writeJson(std::cout);
    else
    {
      std::ofstream output(options_.output_path);
      if (!output)
        throw std::runtime_error("unable to open report " + options_.output_path);
      writeJson(output);
      std::cerr << "Wrote " << options_.output_path << '\n';
    }
  }

private:
  /// Write stable, diffable JSON without introducing a benchmark dependency.
  void writeJson(std::ostream& output) const
  {
    output << "{\n"
           << "  \"schema\":\"qmcpack.psiformer.memory_benchmark.v1\",\n"
           << "  \"scenario\":" << jsonString(options_.scenario) << ",\n"
#ifdef PSIFORMER_MEMORY_COMPONENT
           << "  \"executable_kind\":\"component\",\n"
#else
           << "  \"executable_kind\":\"native\",\n"
#endif
           << "  \"parameter_file\":" << jsonString(options_.parameter_path) << ",\n"
           << "  \"configuration_file\":" << jsonString(options_.configuration_path) << ",\n"
           << "  \"requested_walkers\":" << options_.walkers << ",\n"
#ifdef PSIFORMER_MEMORY_COMPONENT
           << "  \"workload_walkers\":" << (options_.scenario == "model_load" ? 0 : options_.walkers) << ",\n"
#else
           << "  \"workload_walkers\":1,\n"
#endif
           << "  \"warm_calls\":" << options_.warm_calls << ",\n"
           << "  \"model\":{\"parameters\":" << metadata_.parameters
           << ",\"configurations\":" << metadata_.configurations
           << ",\"electrons\":" << metadata_.electrons
           << ",\"nuclei\":" << metadata_.nuclei
           << ",\"spin_up\":" << metadata_.spin_up
           << ",\"spin_down\":" << metadata_.spin_down << "},\n"
           << "  \"provenance\":{\n"
           << "    \"qmcpack_version\":"
           << jsonString(std::to_string(QMCPACK_VERSION_MAJOR) + "." +
                         std::to_string(QMCPACK_VERSION_MINOR) + "." +
                         std::to_string(QMCPACK_VERSION_PATCH)) << ",\n"
           << "    \"qmcpack_git_revision\":" << jsonString(PSIFORMER_STRINGIFY(GIT_HASH_RAW)) << ",\n"
           << "    \"compiler\":" << jsonString(__VERSION__) << ",\n"
           << "    \"build_complex\":"
#ifdef QMC_COMPLEX
           << "true,\n"
#else
           << "false,\n"
#endif
#ifdef PSIFORMER_MEMORY_COMPONENT
           << "    \"sizeof_real\":" << sizeof(qmcplusplus::QMCTraits::RealType) << ",\n"
           << "    \"sizeof_value\":" << sizeof(qmcplusplus::QMCTraits::ValueType) << ",\n"
#else
           << "    \"sizeof_real\":" << sizeof(double) << ",\n"
           << "    \"sizeof_value\":" << sizeof(double) << ",\n"
#endif
           << "    \"host\":" << jsonString(hostName()) << ",\n"
           << "    \"pid\":" << getpid() << ",\n"
           << "    \"hardware_concurrency\":" << std::thread::hardware_concurrency() << ",\n"
           << "    \"affinity_cpu_count\":" << affinityCpuCount() << ",\n"
           << "    \"thread_environment\":{"
           << "\"OMP_NUM_THREADS\":" << jsonString(environmentValue("OMP_NUM_THREADS")) << ','
           << "\"OMP_MAX_ACTIVE_LEVELS\":" << jsonString(environmentValue("OMP_MAX_ACTIVE_LEVELS")) << ','
           << "\"OMP_PROC_BIND\":" << jsonString(environmentValue("OMP_PROC_BIND")) << ','
           << "\"OMP_PLACES\":" << jsonString(environmentValue("OMP_PLACES")) << ','
           << "\"OPENBLAS_NUM_THREADS\":" << jsonString(environmentValue("OPENBLAS_NUM_THREADS")) << ','
           << "\"MKL_NUM_THREADS\":" << jsonString(environmentValue("MKL_NUM_THREADS")) << ','
           << "\"BLIS_NUM_THREADS\":" << jsonString(environmentValue("BLIS_NUM_THREADS")) << ','
           << "\"VECLIB_MAXIMUM_THREADS\":" << jsonString(environmentValue("VECLIB_MAXIMUM_THREADS"))
           << "}\n  },\n"
           << "  \"accounting_notes\":[\n"
           << "    \"accounted bytes are explicit numeric capacities plus named shallow objects, not an allocator census\",\n"
           << "    \"RSS additionally includes code, shared libraries, HDF5, BLAS, allocator arenas, maps, strings, and object control blocks\",\n"
           << "    \"the model estimate includes the canonical flat vector, named leaf tensor payloads, and imported configuration tensors\",\n"
           << "    \"component reports include production-accounted clone scratch; crowd-resource scratch remains visible only through RSS\",\n"
           << "    \"current RSS is Linux procfs resident memory and is zero when procfs is unavailable\",\n"
           << "    \"peak RSS is cumulative within this one-scenario process\"\n"
           << "  ],\n"
           << "  \"phases\":[\n";

    for (std::size_t index = 0; index < phases_.size(); ++index)
    {
      const Phase& phase = phases_[index];
      const std::uint64_t baseline_rss = phases_.front().current_rss_bytes;
      const std::uint64_t previous_rss = index == 0 ? phase.current_rss_bytes : phases_[index - 1].current_rss_bytes;
      const std::uint64_t baseline_accounted = phases_.front().accounted_bytes;
      const std::uint64_t previous_accounted = index == 0 ? phase.accounted_bytes : phases_[index - 1].accounted_bytes;
      auto signedDelta = [](std::uint64_t value, std::uint64_t reference) {
        return static_cast<std::int64_t>(value) - static_cast<std::int64_t>(reference);
      };
      output << "    {\"name\":" << jsonString(phase.name)
             << ",\"current_rss_bytes\":" << phase.current_rss_bytes
             << ",\"peak_rss_bytes\":" << phase.peak_rss_bytes
             << ",\"rss_delta_from_baseline_bytes\":" << signedDelta(phase.current_rss_bytes, baseline_rss)
             << ",\"rss_delta_from_previous_bytes\":" << signedDelta(phase.current_rss_bytes, previous_rss)
             << ",\"accounted_bytes\":" << phase.accounted_bytes
             << ",\"accounted_delta_from_baseline_bytes\":"
             << signedDelta(phase.accounted_bytes, baseline_accounted)
             << ",\"accounted_delta_from_previous_bytes\":"
             << signedDelta(phase.accounted_bytes, previous_accounted)
             << ",\"accounted_categories\":{";
      bool first = true;
      for (const auto& [category, bytes] : phase.accounted_categories)
      {
        if (!first)
          output << ',';
        first = false;
        output << jsonString(category) << ':' << bytes;
      }
      output << "}}" << (index + 1 == phases_.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"sink\":" << std::setprecision(17) << benchmark_sink << "\n}\n";
  }

  Options options_;
  ModelMetadata metadata_;
  std::vector<Phase> phases_;
};

/** Write the standard generated export to a persistent caller-owned directory. */
int generateFixture(const std::string& system, const std::filesystem::path& directory)
{
  std::filesystem::create_directories(directory);
  GeneratedFiles generated = generateFiles(system);
  std::filesystem::copy_file(generated.parameters, directory / "parameters.h5",
                             std::filesystem::copy_options::overwrite_existing);
  std::filesystem::copy_file(generated.configuration, directory / "configuration.h5",
                             std::filesystem::copy_options::overwrite_existing);
  std::cout << (directory / "parameters.h5") << '\n'
            << (directory / "configuration.h5") << '\n';
  return EXIT_SUCCESS;
}

#ifndef PSIFORMER_MEMORY_COMPONENT
/** Form the immutable typed execution plan shared by direct executors. */
qmcplusplus::psiformer::PsiFormerExecutionPlan makePlan(const pf::PsiFormer& model)
{
  return qmcplusplus::psiformer::PsiFormerExecutionPlan::fromParameters(
      model.p, ModelShape{model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0],
                          model.ndet, model.dim, model.heads, 4});
}
#endif

#ifdef PSIFORMER_MEMORY_COMPONENT
/** Create a perturbed LiH walker while preserving the generated spin ordering. */
std::unique_ptr<ParticleSet> makeWalker(const SimulationCell& simulation_cell, std::size_t walker)
{
  const Geometry geometry = makeGeometry("lih");
  auto particles          = std::make_unique<ParticleSet>(simulation_cell);
  particles->setName("e" + std::to_string(walker));
  particles->create({2, 2});
  const double walker_shift = 0.0007 * static_cast<double>(walker % 17);
  for (int electron = 0; electron < particles->getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      particles->R[electron][dimension] = geometry.electrons[3 * electron + dimension] +
          walker_shift * static_cast<double>((electron + 1) * (dimension + 1));
  particles->update();
  return particles;
}

/** Return the explicit logical bytes owned by one benchmark ParticleSet population. */
std::uint64_t walkerLogicalBytes(std::size_t walkers, std::size_t electrons)
{
  // R, G, and L are the relevant walker-sized arrays.  ParticleSet distance
  // tables and vector capacities remain represented only in measured RSS.
  return walkers * electrons *
      (sizeof(ParticleSet::PosType) + sizeof(ParticleSet::GradType) + sizeof(ParticleSet::ValueType));
}

/** Sum production-reported evaluator scratch across component clones. */
std::uint64_t cloneWorkspaceBytes(const std::vector<PsiFormerWF*>& components)
{
  std::uint64_t bytes = 0;
  for (const PsiFormerWF* component : components)
    bytes += qmcplusplus::testing::TestPsiFormerWF::directWorkspaceDiagnostics(*component).accountedBytes();
  return bytes;
}
#endif

#ifndef PSIFORMER_MEMORY_COMPONENT
/** Measure an isolated native value/spatial/score/kinetic workspace scenario. */
void runNativeScenario(const Options& options, const ModelMetadata& metadata, MemoryReport& report)
{
  report.snapshot("baseline");

  pf::PsiFormer model(options.parameter_path, options.configuration_path);
  auto plan = makePlan(model);
  const std::uint64_t model_bytes = metadata.accountedModelNumericBytes();
  report.snapshot("model_loaded", {{"model_numeric", model_bytes}});

  const pf::Tensor positions = model.cfg.configuration(0);
  const std::uint64_t position_bytes = positions.x.capacity() * sizeof(double);
  const auto position_view = pf::GeometryPositionView::interleaved(positions.x.data(), model.ne);

  if (options.scenario == "value")
  {
    pf::DirectValueExecutor executor(model, plan);
    auto workspace = executor.makeWorkspace();
    const std::uint64_t workspace_bytes = workspace->vectorStorageBytes();
    report.snapshot("prepared", {{"model_numeric", model_bytes}, {"positions", position_bytes},
                                  {"workspace_numeric", workspace_bytes}});
    workspace->setPositions(position_view);
    auto result = executor.evaluate(*workspace);
    benchmark_sink += result.sign + result.logabs + result.value;
    report.snapshot("first_touch", {{"model_numeric", model_bytes}, {"positions", position_bytes},
                                     {"workspace_numeric", workspace_bytes}});
    for (int call = 0; call < options.warm_calls; ++call)
    {
      workspace->setPositions(position_view);
      result = executor.evaluate(*workspace);
      benchmark_sink += result.logabs;
    }
    report.snapshot("warmed", {{"model_numeric", model_bytes}, {"positions", position_bytes},
                                {"workspace_numeric", workspace_bytes}});
  }
  else if (options.scenario == "full_vgl" || options.scenario == "active_gradient")
  {
    pf::DirectValueExecutor value_executor(model, plan);
    pf::DirectSpatialExecutor executor(model, value_executor, plan);
    const pf::DirectSpatialMode mode = options.scenario == "full_vgl"
        ? pf::DirectSpatialMode::FULL_VGL
        : pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT;
    auto workspace = executor.makeWorkspace(mode);
    const std::uint64_t workspace_bytes = workspace->vectorStorageBytes();
    report.snapshot("prepared", {{"model_numeric", model_bytes}, {"positions", position_bytes},
                                  {"workspace_numeric", workspace_bytes}});
    auto evaluate = [&]() {
      workspace->setPositions(position_view);
      if (mode == pf::DirectSpatialMode::FULL_VGL)
      {
        const auto result = executor.evaluateFull(*workspace);
        benchmark_sink += result.sign + result.logabs + result.value + result.gradient[0] + result.lap_log[0];
      }
      else
      {
        const auto result = executor.evaluateActive(*workspace, 0);
        benchmark_sink += result.sign + result.logabs + result.value + result.gradient[0];
      }
    };
    evaluate();
    report.snapshot("first_touch", {{"model_numeric", model_bytes}, {"positions", position_bytes},
                                     {"workspace_numeric", workspace_bytes}});
    for (int call = 0; call < options.warm_calls; ++call)
      evaluate();
    report.snapshot("warmed", {{"model_numeric", model_bytes}, {"positions", position_bytes},
                                {"workspace_numeric", workspace_bytes}});
  }
  else if (options.scenario == "score")
  {
    pf::DirectScoreExecutor executor(model, plan);
    auto workspace = executor.makeWorkspace();
    const std::uint64_t workspace_bytes = workspace->vectorStorageBytes();
    report.snapshot("prepared", {{"model_numeric", model_bytes}, {"positions", position_bytes},
                                  {"workspace_numeric", workspace_bytes}});
    auto evaluate = [&]() {
      workspace->setPositions(position_view);
      const auto result = executor.evaluate(*workspace);
      benchmark_sink += result.sign + result.logabs + result.value + result.parameter_score[0];
    };
    evaluate();
    report.snapshot("first_touch", {{"model_numeric", model_bytes}, {"positions", position_bytes},
                                     {"workspace_numeric", workspace_bytes}});
    for (int call = 0; call < options.warm_calls; ++call)
      evaluate();
    report.snapshot("warmed", {{"model_numeric", model_bytes}, {"positions", position_bytes},
                                {"workspace_numeric", workspace_bytes}});
  }
  else if (options.scenario == "score_and_kinetic")
  {
    pf::DirectKineticExecutor executor(model, plan);
    auto workspace = executor.makeWorkspace();
    const std::uint64_t workspace_bytes = workspace->vectorStorageBytes();
    report.snapshot("prepared", {{"model_numeric", model_bytes}, {"positions", position_bytes},
                                  {"workspace_numeric", workspace_bytes}});
    auto evaluate = [&]() {
      workspace->setPositions(position_view);
      const auto result = executor.evaluate(*workspace);
      benchmark_sink += result.sign + result.logabs + result.value + result.parameter_score[0] +
          result.kinetic_parameter_response[0];
    };
    evaluate();
    report.snapshot("first_touch", {{"model_numeric", model_bytes}, {"positions", position_bytes},
                                     {"workspace_numeric", workspace_bytes}});
    for (int call = 0; call < options.warm_calls; ++call)
      evaluate();
    report.snapshot("warmed", {{"model_numeric", model_bytes}, {"positions", position_bytes},
                                {"workspace_numeric", workspace_bytes}});
  }
  else
    throw std::invalid_argument("not a native scenario: " + options.scenario);
}
#endif

#ifdef PSIFORMER_MEMORY_COMPONENT
/** Measure model sharing, clone population, and production crowd entry points. */
void runComponentScenario(const Options& options, const ModelMetadata& metadata, MemoryReport& report)
{
  report.snapshot("baseline");
  const SimulationCell simulation_cell;
  auto leader = std::make_unique<PsiFormerWF>(
      "pf_memory", options.parameter_path, options.configuration_path);
  const std::uint64_t model_bytes = metadata.accountedModelNumericBytes();
  report.snapshot("model_loaded", {{"model_numeric", model_bytes},
                                    {"component_shallow", sizeof(PsiFormerWF)}});

  if (options.scenario == "model_load")
  {
    report.snapshot("warmed", {{"model_numeric", model_bytes},
                                {"component_shallow", sizeof(PsiFormerWF)}});
    return;
  }

  std::vector<std::unique_ptr<ParticleSet>> walkers;
  std::vector<std::unique_ptr<WaveFunctionComponent>> clone_storage;
  std::vector<PsiFormerWF*> components;
  walkers.reserve(options.walkers);
  clone_storage.reserve(options.walkers > 0 ? options.walkers - 1 : 0);
  components.reserve(options.walkers);
  walkers.push_back(makeWalker(simulation_cell, 0));
  components.push_back(leader.get());
  for (std::size_t walker = 1; walker < options.walkers; ++walker)
  {
    walkers.push_back(makeWalker(simulation_cell, walker));
    clone_storage.push_back(leader->makeClone(*walkers.back()));
    components.push_back(static_cast<PsiFormerWF*>(clone_storage.back().get()));
  }

  const std::uint64_t component_bytes = options.walkers * sizeof(PsiFormerWF);
  const std::uint64_t particle_bytes  = options.walkers * sizeof(ParticleSet);
  const std::uint64_t walker_bytes    = walkerLogicalBytes(options.walkers, metadata.electrons);
  auto clone_categories = std::map<std::string, std::uint64_t>{{"model_numeric", model_bytes},
      {"component_shallow", component_bytes}, {"particle_shallow", particle_bytes},
      {"walker_logical_arrays", walker_bytes},
      {"clone_workspace_numeric", cloneWorkspaceBytes(components)}};
  report.snapshot("clone_population", clone_categories);
  if (options.scenario == "clone_population")
  {
    report.snapshot("warmed", clone_categories);
    return;
  }

  RefVectorWithLeader<WaveFunctionComponent> component_list(*leader);
  auto particle_list = std::make_unique<RefVectorWithLeader<ParticleSet>>(*walkers.front());
  for (std::size_t walker = 0; walker < options.walkers; ++walker)
  {
    component_list.push_back(*components[walker]);
    particle_list->push_back(*walkers[walker]);
  }

  ResourceCollection resource_template("psiformer_memory_template");
  leader->createResource(resource_template);
  ResourceCollection crowd_resource(resource_template);

  std::vector<ParticleSet::ParticleGradient> gradients;
  std::vector<ParticleSet::ParticleLaplacian> laplacians;
  RefVector<ParticleSet::ParticleGradient> gradient_list;
  RefVector<ParticleSet::ParticleLaplacian> laplacian_list;
  std::vector<PsiFormerWF::GradType> active_gradients(options.walkers);
  std::vector<bool> recompute(options.walkers, true);
  std::uint64_t output_bytes = 0;
  if (options.scenario == "crowd_full_vgl")
  {
    gradients.resize(options.walkers);
    laplacians.resize(options.walkers);
    for (std::size_t walker = 0; walker < options.walkers; ++walker)
    {
      gradients[walker].resize(metadata.electrons);
      laplacians[walker].resize(metadata.electrons);
      gradient_list.push_back(gradients[walker]);
      laplacian_list.push_back(laplacians[walker]);
    }
    output_bytes = options.walkers * metadata.electrons *
        (sizeof(ParticleSet::GradType) + sizeof(ParticleSet::ValueType));
  }
  else if (options.scenario == "crowd_active_gradient")
    output_bytes = options.walkers * sizeof(PsiFormerWF::GradType);
  else if (options.scenario != "crowd_value")
    throw std::invalid_argument("not a component scenario: " + options.scenario);

  auto prepared_categories = clone_categories;
  prepared_categories["caller_output"] = output_bytes;
  report.snapshot("prepared", prepared_categories);

  auto evaluate = [&]() {
    if (options.scenario == "crowd_value")
      leader->mw_recompute(component_list, *particle_list, recompute);
    else if (options.scenario == "crowd_full_vgl")
      leader->mw_evaluateLog(component_list, *particle_list, gradient_list, laplacian_list);
    else
      leader->mw_evalGrad(component_list, *particle_list, 0, active_gradients);
    benchmark_sink += std::real(components.front()->get_log_value());
  };

  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource, component_list);
    evaluate();
    prepared_categories["clone_workspace_numeric"] = cloneWorkspaceBytes(components);
    report.snapshot("first_touch", prepared_categories);
    for (int call = 0; call < options.warm_calls; ++call)
      evaluate();
    prepared_categories["clone_workspace_numeric"] = cloneWorkspaceBytes(components);
    report.snapshot("warmed", prepared_categories);
  }
}
#endif

/** Generate a fixture or execute one isolated scenario. */
int run(int argc, char** argv)
{
  if (argc == 4 && std::string(argv[1]) == "--generate-fixture")
    return generateFixture(argv[2], argv[3]);

  const Options options        = parseOptions(argc, argv);
  const ModelMetadata metadata = readMetadata(options);
  MemoryReport report(options, metadata);
#ifdef PSIFORMER_MEMORY_COMPONENT
  runComponentScenario(options, metadata, report);
#else
  runNativeScenario(options, metadata, report);
#endif
  report.write();
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv)
{
  try
  {
    return run(argc, argv);
  }
  catch (const std::exception& error)
  {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
