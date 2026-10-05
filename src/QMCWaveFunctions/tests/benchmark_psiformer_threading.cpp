//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file benchmark_psiformer_threading.cpp
 * @brief Non-gating production-crowd threading benchmark for PsiFormer.
 *
 * One invocation measures one uniform crowd layout.  Run separate processes
 * for one, two, and four crowds so RSS and allocator state remain attributable.
 * Timings are diagnostic only and intentionally have no CTest registration or
 * performance threshold.
 */

#include <stdexcept>

// Reuse the deterministic generated fixture without linking a Catch2 runner.
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
#include "Concurrency/ParallelExecutor.hpp"
#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "ResourceCollection.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
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

#ifndef PSIFORMER_VPL_ID_MANIFEST
#define PSIFORMER_VPL_ID_MANIFEST "<unknown>"
#endif
#ifndef PSIFORMER_BLA_VENDOR_MANIFEST
#define PSIFORMER_BLA_VENDOR_MANIFEST "<unknown>"
#endif
#ifndef PSIFORMER_BLAS_LIBRARIES_MANIFEST
#define PSIFORMER_BLAS_LIBRARIES_MANIFEST "<unknown>"
#endif
#ifndef PSIFORMER_VPL_OMP_MANIFEST
#define PSIFORMER_VPL_OMP_MANIFEST 0
#endif

namespace qmcplusplus::testing
{
/** Narrow read-only access to the acquired crowd diagnostic. */
class TestPsiFormerWF
{
public:
  static PsiFormerCrowdWorkspaceDiagnostics crowdWorkspaceDiagnostics(
      const PsiFormerWF& component,
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list)
  {
    return component.crowdWorkspaceDiagnosticsForTesting(wfc_list);
  }
};
} // namespace qmcplusplus::testing

namespace
{
using Clock = std::chrono::steady_clock;
using qmcplusplus::ParticleSet;
using qmcplusplus::PsiFormerWF;
using qmcplusplus::RefVector;
using qmcplusplus::RefVectorWithLeader;
using qmcplusplus::ResourceCollection;
using qmcplusplus::ResourceCollectionTeamLock;
using qmcplusplus::SimulationCell;
using qmcplusplus::WaveFunctionComponent;
using qmcplusplus::testing::PsiFormerCrowdWorkspaceDiagnostics;
using qmcplusplus::testing::psiformer::GeneratedFiles;
using qmcplusplus::testing::psiformer::Geometry;
using qmcplusplus::testing::psiformer::generateFiles;
using qmcplusplus::testing::psiformer::makeGeometry;

#define PSIFORMER_STRINGIFY_DETAIL(value) #value
#define PSIFORMER_STRINGIFY(value) PSIFORMER_STRINGIFY_DETAIL(value)

volatile double benchmark_sink = 0.0;

/** Controls for one fresh-process uniform crowd layout. */
struct Options
{
  std::string system      = "lih";
  std::string output_path;
  std::size_t crowds              = 1;
  std::size_t walkers_per_crowd   = 4;
  int warmup_calls                = 1;
  int repeats                     = 3;
};

/** Timing and checksum observations for one scheduling policy. */
struct TimingSummary
{
  std::vector<double> raw_seconds;
  std::vector<double> checksums;
};

[[noreturn]] void usage(const char* executable, const std::string& error = {})
{
  if (!error.empty())
    std::cerr << "error: " << error << '\n';
  std::cerr << "usage: " << executable
            << " [--system lih|lih_pair|lih_pp] [--crowds N]"
               " [--walkers-per-crowd N] [--warmup-calls N]"
               " [--repeats N] [--output FILE.json]\n";
  std::exit(error.empty() ? EXIT_SUCCESS : EXIT_FAILURE);
}

Options parseOptions(int argc, char** argv)
{
  Options options;
  for (int argument = 1; argument < argc; ++argument)
  {
    const std::string key = argv[argument];
    if (key == "--help")
      usage(argv[0]);
    if (argument + 1 >= argc)
      usage(argv[0], "missing value after " + key);
    const std::string value = argv[++argument];
    try
    {
      if (key == "--system")
        options.system = value;
      else if (key == "--crowds")
        options.crowds = std::stoull(value);
      else if (key == "--walkers-per-crowd")
        options.walkers_per_crowd = std::stoull(value);
      else if (key == "--warmup-calls")
        options.warmup_calls = std::stoi(value);
      else if (key == "--repeats")
        options.repeats = std::stoi(value);
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

  if (options.system != "lih" && options.system != "lih_pair" && options.system != "lih_pp")
    usage(argv[0], "unsupported generated system " + options.system);
  if (options.crowds == 0 || options.walkers_per_crowd == 0 ||
      options.warmup_calls < 0 || options.repeats <= 0)
    usage(argv[0], "crowds, walkers, and repeats must be positive; warmups must be nonnegative");
  if (options.walkers_per_crowd >
      std::numeric_limits<std::size_t>::max() / options.crowds)
    usage(argv[0], "requested walker population overflows size_t");
  return options;
}

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

std::string environmentValue(const char* name)
{
  const char* value = std::getenv(name);
  return value ? value : "<unset>";
}

void requireDirectBackends()
{
  constexpr std::array<const char*, 4> variables{
      "PSIFORMER_VALUE_BACKEND", "PSIFORMER_SPATIAL_BACKEND",
      "PSIFORMER_SCORE_BACKEND", "PSIFORMER_KINETIC_BACKEND"};
  for (const char* variable : variables)
  {
    const char* value = std::getenv(variable);
    if (value && std::string(value) != "direct")
      throw std::runtime_error(std::string(variable) +
          " must be unset or direct for the production batching benchmark");
  }
}

std::string hostName()
{
  char hostname[256]{};
  return gethostname(hostname, sizeof(hostname) - 1) == 0 ? hostname : "<unknown>";
}

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

std::uint64_t currentResidentBytes()
{
#ifdef __linux__
  std::ifstream statm("/proc/self/statm");
  std::uint64_t virtual_pages  = 0;
  std::uint64_t resident_pages = 0;
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size > 0 && statm >> virtual_pages >> resident_pages)
    return resident_pages * static_cast<std::uint64_t>(page_size);
#endif
  return 0;
}

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

double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
}

std::string pointerIdentity(const void* pointer)
{
  if (!pointer)
    return "<not-allocated>";
  std::ostringstream output;
  output << pointer;
  return output.str();
}

template<class Values>
void writeDoubleArray(std::ostream& output, const Values& values)
{
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index)
  {
    if (index != 0)
      output << ',';
    output << std::setprecision(17) << values[index];
  }
  output << ']';
}

std::unique_ptr<ParticleSet> makeWalker(const SimulationCell& simulation_cell,
                                        const std::string& system,
                                        std::size_t walker)
{
  const Geometry geometry = makeGeometry(system);
  auto particles          = std::make_unique<ParticleSet>(simulation_cell);
  particles->setName("thread_benchmark_walker_" + std::to_string(walker));
  particles->create({static_cast<int>(geometry.nup),
                     static_cast<int>(geometry.electrons.size() / 3 - geometry.nup)});
  auto& species = particles->getSpeciesSet();
  species.addSpecies("u");
  species.addSpecies("d");
  const int mass = species.addAttribute("mass");
  species(mass, 0) = 1.0;
  species(mass, 1) = 1.0;
  particles->resetGroups();
  const double shift = 0.0007 * static_cast<double>(walker % 17);
  for (int electron = 0; electron < particles->getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      particles->R[electron][dimension] = geometry.electrons[3 * electron + dimension] +
          shift * static_cast<double>((electron + 1) * (dimension + 1));
  particles->update();
  return particles;
}

/** Own all mutable data used by one production-shaped crowd task. */
class CrowdWorkload
{
public:
  CrowdWorkload(PsiFormerWF& leader,
                const std::vector<PsiFormerWF*>& components,
                const std::vector<std::unique_ptr<ParticleSet>>& walkers,
                std::size_t first_walker,
                std::size_t walker_count,
                const ResourceCollection& resource_template)
      : leader_(leader),
        components_(leader),
        particles_(*walkers[first_walker]),
        gradients_(walker_count),
        laplacians_(walker_count),
        resource_(resource_template)
  {
    for (std::size_t local_walker = 0; local_walker < walker_count; ++local_walker)
    {
      const std::size_t walker = first_walker + local_walker;
      components_.push_back(*components[walker]);
      particles_.push_back(*walkers[walker]);
      const std::size_t electrons = walkers[walker]->getTotalNum();
      gradients_[local_walker].resize(electrons);
      laplacians_[local_walker].resize(electrons);
      gradient_refs_.push_back(gradients_[local_walker]);
      laplacian_refs_.push_back(laplacians_[local_walker]);
    }
  }

  double evaluate()
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(resource_, components_);
    return evaluateAcquired();
  }

  PsiFormerCrowdWorkspaceDiagnostics warmAndDiagnose(int warmup_calls)
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(resource_, components_);
    for (int call = 0; call <= warmup_calls; ++call)
      benchmark_sink += evaluateAcquired();
    return qmcplusplus::testing::TestPsiFormerWF::crowdWorkspaceDiagnostics(leader_, components_);
  }

private:
  double evaluateAcquired()
  {
    for (std::size_t walker = 0; walker < components_.size(); ++walker)
    {
      gradients_[walker]  = PsiFormerWF::ValueType(0);
      laplacians_[walker] = PsiFormerWF::ValueType(0);
    }
    leader_.mw_evaluateLog(components_, particles_, gradient_refs_, laplacian_refs_);

    double checksum = 0.0;
    for (std::size_t walker = 0; walker < components_.size(); ++walker)
    {
      const double walker_scale = static_cast<double>(walker + 1);
      const auto log_value = components_.getCastedElement<PsiFormerWF>(walker).get_log_value();
      checksum += walker_scale * (std::real(log_value) + 0.125 * std::imag(log_value));
      for (int electron = 0; electron < particles_[walker].getTotalNum(); ++electron)
      {
        const double electron_scale = walker_scale * static_cast<double>(electron + 1);
        for (int dimension = 0; dimension < 3; ++dimension)
        {
          const auto value = gradients_[walker][electron][dimension];
          checksum += electron_scale * (0.03125 * std::real(value) + 0.015625 * std::imag(value));
        }
        const auto value = laplacians_[walker][electron];
        checksum += electron_scale * (0.0078125 * std::real(value) + 0.00390625 * std::imag(value));
      }
    }
    return checksum;
  }

  PsiFormerWF& leader_;
  RefVectorWithLeader<WaveFunctionComponent> components_;
  RefVectorWithLeader<ParticleSet> particles_;
  std::vector<ParticleSet::ParticleGradient> gradients_;
  std::vector<ParticleSet::ParticleLaplacian> laplacians_;
  RefVector<ParticleSet::ParticleGradient> gradient_refs_;
  RefVector<ParticleSet::ParticleLaplacian> laplacian_refs_;
  ResourceCollection resource_;
};

double orderedChecksum(const std::vector<double>& crowd_checksums)
{
  double checksum = 0.0;
  for (std::size_t crowd = 0; crowd < crowd_checksums.size(); ++crowd)
    checksum += static_cast<double>(crowd + 1) * crowd_checksums[crowd];
  return checksum;
}

void validateChecksum(double actual, double expected, const char* policy)
{
  const double tolerance = 1.0e-10 * (1.0 + std::abs(expected));
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
    throw std::runtime_error(std::string(policy) + " checksum does not match the warmed baseline");
}

void validateDiagnostics(const std::vector<PsiFormerCrowdWorkspaceDiagnostics>& diagnostics)
{
  const auto& first = diagnostics.front();
  std::set<const void*> resource_identities;
  std::set<const void*> batch_identities;
  for (const auto& diagnostic : diagnostics)
  {
    if (!diagnostic.shared_model_identity ||
        diagnostic.shared_model_identity != first.shared_model_identity ||
        diagnostic.persistent_model_identity != first.persistent_model_identity ||
        diagnostic.parameter_version != first.parameter_version ||
        diagnostic.backend_modes != first.backend_modes)
      throw std::runtime_error("crowds do not share one model/version/backend configuration");
    if (!diagnostic.resource_identity || !diagnostic.batch_workspace_identity ||
        diagnostic.batch_bytes == 0 || diagnostic.batch_bytes != first.batch_bytes)
      throw std::runtime_error("crowd batch workspace ownership or uniform capacity is invalid");
    resource_identities.insert(diagnostic.resource_identity);
    batch_identities.insert(diagnostic.batch_workspace_identity);
  }
  if (resource_identities.size() != diagnostics.size() ||
      batch_identities.size() != diagnostics.size())
    throw std::runtime_error("crowds unexpectedly share mutable resource scratch");
}

TimingSummary timeSerial(const std::vector<std::unique_ptr<CrowdWorkload>>& workloads,
                         int repeats,
                         double expected_checksum)
{
  TimingSummary summary;
  summary.raw_seconds.reserve(repeats);
  summary.checksums.reserve(repeats);
  std::vector<double> crowd_checksums(workloads.size());
  for (int repeat = 0; repeat < repeats; ++repeat)
  {
    const auto start = Clock::now();
    for (std::size_t crowd = 0; crowd < workloads.size(); ++crowd)
      crowd_checksums[crowd] = workloads[crowd]->evaluate();
    summary.raw_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
    summary.checksums.push_back(orderedChecksum(crowd_checksums));
    validateChecksum(summary.checksums.back(), expected_checksum, "serial");
  }
  return summary;
}

TimingSummary timeParallel(const std::vector<std::unique_ptr<CrowdWorkload>>& workloads,
                           int repeats,
                           double expected_checksum)
{
  TimingSummary summary;
  summary.raw_seconds.reserve(repeats);
  summary.checksums.reserve(repeats);
  std::vector<double> crowd_checksums(workloads.size());
  std::vector<std::exception_ptr> failures(workloads.size());
  qmcplusplus::ParallelExecutor<qmcplusplus::Executor::OPENMP> executor;

  for (int repeat = 0; repeat < repeats; ++repeat)
  {
    std::fill(failures.begin(), failures.end(), std::exception_ptr{});
    const auto start = Clock::now();
    executor(static_cast<int>(workloads.size()), [&](int crowd) {
      try
      {
        crowd_checksums[crowd] = workloads[crowd]->evaluate();
      }
      catch (...)
      {
        failures[crowd] = std::current_exception();
      }
    });
    summary.raw_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
    for (const std::exception_ptr& failure : failures)
      if (failure)
        std::rethrow_exception(failure);
    summary.checksums.push_back(orderedChecksum(crowd_checksums));
    validateChecksum(summary.checksums.back(), expected_checksum, "parallel");
  }
  return summary;
}

void writeManifest(std::ostream& output,
                   const Options& options,
                   const PsiFormerWF& leader,
                   const std::vector<PsiFormerCrowdWorkspaceDiagnostics>& diagnostics,
                   const TimingSummary& serial,
                   const TimingSummary& parallel,
                   double expected_checksum,
                   std::uint64_t baseline_rss,
                   std::uint64_t warmed_rss,
                   std::uint64_t final_rss,
                   std::uint64_t peak_rss)
{
  std::set<const void*> model_identities;
  std::set<const void*> resource_identities;
  std::set<const void*> batch_identities;
  for (const auto& diagnostic : diagnostics)
  {
    model_identities.insert(diagnostic.shared_model_identity);
    resource_identities.insert(diagnostic.resource_identity);
    batch_identities.insert(diagnostic.batch_workspace_identity);
  }

  const std::size_t configurations = options.crowds * options.walkers_per_crowd;
  const double serial_median       = median(serial.raw_seconds);
  const double parallel_median     = median(parallel.raw_seconds);
  const auto& schema               = leader.parameterSchema();
  output << "{\n"
         << "  \"schema\":\"qmcpack.psiformer.threading_benchmark.v1\",\n"
         << "  \"non_gating\":true,\n"
         << "  \"workload\":{\"system\":" << jsonString(options.system)
         << ",\"crowds\":" << options.crowds
         << ",\"walkers_per_crowd\":" << options.walkers_per_crowd
         << ",\"configurations_per_repeat\":" << configurations
         << ",\"first_touch_calls_per_crowd\":1"
         << ",\"additional_warmup_calls_per_crowd\":" << options.warmup_calls
         << ",\"checksum_baseline_calls_per_crowd\":1"
         << ",\"repeats\":" << options.repeats
         << ",\"operation\":\"mw_evaluateLog_full_vgl\"},\n"
         << "  \"model\":{\"shared_identity_count\":" << model_identities.size()
         << ",\"persistent_identity\":" << diagnostics.front().persistent_model_identity
         << ",\"parameter_version\":" << diagnostics.front().parameter_version
         << ",\"parameter_count\":" << schema.parameterCount()
         << ",\"schema_fingerprint\":" << jsonString(schema.fingerprint()) << "},\n"
         << "  \"backends\":{\"value\":" << jsonString(diagnostics.front().backend_modes[0])
         << ",\"spatial\":" << jsonString(diagnostics.front().backend_modes[1])
         << ",\"score\":" << jsonString(diagnostics.front().backend_modes[2])
         << ",\"kinetic\":" << jsonString(diagnostics.front().backend_modes[3]) << "},\n"
         << "  \"ownership\":{\"resource_identity_count\":" << resource_identities.size()
         << ",\"batch_workspace_identity_count\":" << batch_identities.size()
         << ",\"crowds\":[\n";
  for (std::size_t crowd = 0; crowd < diagnostics.size(); ++crowd)
  {
    const auto& diagnostic = diagnostics[crowd];
    output << "    {\"crowd\":" << crowd
           << ",\"model_identity\":" << jsonString(pointerIdentity(diagnostic.shared_model_identity))
           << ",\"resource_identity\":" << jsonString(pointerIdentity(diagnostic.resource_identity))
           << ",\"batch_workspace_identity\":"
           << jsonString(pointerIdentity(diagnostic.batch_workspace_identity))
           << ",\"score_workspace_identity\":"
           << jsonString(pointerIdentity(diagnostic.score_workspace_identity))
           << ",\"kinetic_workspace_identity\":"
           << jsonString(pointerIdentity(diagnostic.kinetic_workspace_identity))
           << ",\"batch_bytes\":" << diagnostic.batch_bytes
           << ",\"score_bytes\":" << diagnostic.score_bytes
           << ",\"kinetic_bytes\":" << diagnostic.kinetic_bytes
           << ",\"transient_bytes\":" << diagnostic.transient_bytes
           << ",\"accounted_bytes\":" << diagnostic.accountedBytes() << "}"
           << (crowd + 1 == diagnostics.size() ? "\n" : ",\n");
  }
  output << "  ]},\n"
         << "  \"timing\":{\"scope\":\"resource_acquire_full_vgl_release\","
         << "\"serial_raw_seconds\":";
  writeDoubleArray(output, serial.raw_seconds);
  output << ",\"parallel_raw_seconds\":";
  writeDoubleArray(output, parallel.raw_seconds);
  output << ",\"serial_configurations_per_second\":"
         << std::setprecision(17) << static_cast<double>(configurations) / serial_median
         << ",\"parallel_configurations_per_second\":"
         << static_cast<double>(configurations) / parallel_median
         << ",\"serial_checksums\":";
  writeDoubleArray(output, serial.checksums);
  output << ",\"parallel_checksums\":";
  writeDoubleArray(output, parallel.checksums);
  output << ",\"warmed_expected_checksum\":" << expected_checksum << "},\n"
         << "  \"memory\":{\"baseline_rss_bytes\":" << baseline_rss
         << ",\"warmed_rss_bytes\":" << warmed_rss
         << ",\"final_rss_bytes\":" << final_rss
         << ",\"peak_rss_bytes\":" << peak_rss << "},\n"
         << "  \"provenance\":{\"qmcpack_version\":"
         << jsonString(std::to_string(QMCPACK_VERSION_MAJOR) + "." +
                       std::to_string(QMCPACK_VERSION_MINOR) + "." +
                       std::to_string(QMCPACK_VERSION_PATCH))
         << ",\"qmcpack_git_revision\":" << jsonString(PSIFORMER_STRINGIFY(GIT_HASH_RAW))
         << ",\"compiler\":" << jsonString(__VERSION__)
         << ",\"build_complex\":"
#ifdef QMC_COMPLEX
         << "true"
#else
         << "false"
#endif
         << ",\"host\":" << jsonString(hostName())
         << ",\"pid\":" << getpid()
         << ",\"hardware_concurrency\":" << std::thread::hardware_concurrency()
         << ",\"affinity_cpu_count\":" << affinityCpuCount()
         << ",\"openmp_max_threads\":" << omp_get_max_threads()
         << ",\"parallel_executor\":\"OPENMP\""
         << ",\"vpl_id\":" << jsonString(PSIFORMER_VPL_ID_MANIFEST)
         << ",\"bla_vendor\":" << jsonString(PSIFORMER_BLA_VENDOR_MANIFEST)
         << ",\"blas_libraries\":" << jsonString(PSIFORMER_BLAS_LIBRARIES_MANIFEST)
         << ",\"vpl_omp\":" << (PSIFORMER_VPL_OMP_MANIFEST ? "true" : "false")
         << ",\"thread_environment\":{"
         << "\"OMP_NUM_THREADS\":" << jsonString(environmentValue("OMP_NUM_THREADS")) << ','
         << "\"OMP_DYNAMIC\":" << jsonString(environmentValue("OMP_DYNAMIC")) << ','
         << "\"OMP_MAX_ACTIVE_LEVELS\":" << jsonString(environmentValue("OMP_MAX_ACTIVE_LEVELS")) << ','
         << "\"OMP_PROC_BIND\":" << jsonString(environmentValue("OMP_PROC_BIND")) << ','
         << "\"OMP_PLACES\":" << jsonString(environmentValue("OMP_PLACES")) << ','
         << "\"OMP_WAIT_POLICY\":" << jsonString(environmentValue("OMP_WAIT_POLICY")) << ','
         << "\"OPENBLAS_NUM_THREADS\":" << jsonString(environmentValue("OPENBLAS_NUM_THREADS")) << ','
         << "\"GOTO_NUM_THREADS\":" << jsonString(environmentValue("GOTO_NUM_THREADS")) << ','
         << "\"MKL_NUM_THREADS\":" << jsonString(environmentValue("MKL_NUM_THREADS")) << ','
         << "\"MKL_DYNAMIC\":" << jsonString(environmentValue("MKL_DYNAMIC")) << ','
         << "\"MKL_DOMAIN_NUM_THREADS\":" << jsonString(environmentValue("MKL_DOMAIN_NUM_THREADS")) << ','
         << "\"BLIS_NUM_THREADS\":" << jsonString(environmentValue("BLIS_NUM_THREADS")) << ','
         << "\"VECLIB_MAXIMUM_THREADS\":" << jsonString(environmentValue("VECLIB_MAXIMUM_THREADS")) << ','
         << "\"ARMPL_NUM_THREADS\":" << jsonString(environmentValue("ARMPL_NUM_THREADS"))
         << "}},\n"
         << "  \"notes\":["
         << jsonString("timings are diagnostic and carry no regression threshold") << ','
         << jsonString("serial and parallel checksums use the same warmed clone family and stable "
                       "crowd-order reduction") << ','
         << jsonString("accounted bytes are explicit numeric capacities, while RSS includes libraries "
                       "and allocator state")
         << "]\n}\n";
}

int run(int argc, char** argv)
{
  const Options options = parseOptions(argc, argv);
  requireDirectBackends();
  if (options.crowds > static_cast<std::size_t>(omp_get_max_threads()))
    throw std::runtime_error("requested crowd count exceeds omp_get_max_threads; set OMP_NUM_THREADS");

  const std::uint64_t baseline_rss = currentResidentBytes();
  GeneratedFiles files             = generateFiles(options.system);
  const SimulationCell simulation_cell;
  const std::size_t walker_count = options.crowds * options.walkers_per_crowd;

  std::vector<std::unique_ptr<ParticleSet>> walkers;
  walkers.reserve(walker_count);
  for (std::size_t walker = 0; walker < walker_count; ++walker)
    walkers.push_back(makeWalker(simulation_cell, options.system, walker));

  auto leader = std::make_unique<PsiFormerWF>(
      "pf_thread_benchmark", files.parameters.string(), files.configuration.string());
  std::vector<std::unique_ptr<WaveFunctionComponent>> clone_storage;
  std::vector<PsiFormerWF*> components;
  clone_storage.reserve(walker_count - 1);
  components.reserve(walker_count);
  components.push_back(leader.get());
  for (std::size_t walker = 1; walker < walker_count; ++walker)
  {
    clone_storage.push_back(leader->makeClone(*walkers[walker]));
    components.push_back(static_cast<PsiFormerWF*>(clone_storage.back().get()));
  }

  ResourceCollection resource_template("psiformer_thread_benchmark_template");
  leader->createResource(resource_template);
  std::vector<std::unique_ptr<CrowdWorkload>> workloads;
  workloads.reserve(options.crowds);
  for (std::size_t crowd = 0; crowd < options.crowds; ++crowd)
  {
    const std::size_t first_walker = crowd * options.walkers_per_crowd;
    workloads.push_back(std::make_unique<CrowdWorkload>(
        *components[first_walker], components, walkers, first_walker,
        options.walkers_per_crowd, resource_template));
  }

  std::vector<PsiFormerCrowdWorkspaceDiagnostics> diagnostics;
  std::vector<double> warm_checksums;
  diagnostics.reserve(options.crowds);
  warm_checksums.reserve(options.crowds);
  for (auto& workload : workloads)
  {
    diagnostics.push_back(workload->warmAndDiagnose(options.warmup_calls));
    warm_checksums.push_back(workload->evaluate());
  }
  validateDiagnostics(diagnostics);
  const double expected_checksum   = orderedChecksum(warm_checksums);
  const std::uint64_t warmed_rss   = currentResidentBytes();
  const TimingSummary serial       = timeSerial(workloads, options.repeats, expected_checksum);
  const TimingSummary parallel     = timeParallel(workloads, options.repeats, expected_checksum);
  const std::uint64_t final_rss    = currentResidentBytes();
  const std::uint64_t peak_rss     = std::max(final_rss, peakResidentBytes());
  benchmark_sink += serial.checksums.back() + parallel.checksums.back();

  if (options.output_path.empty())
    writeManifest(std::cout, options, *leader, diagnostics, serial, parallel,
                  expected_checksum, baseline_rss, warmed_rss, final_rss, peak_rss);
  else
  {
    std::ofstream output(options.output_path);
    if (!output)
      throw std::runtime_error("unable to open report " + options.output_path);
    writeManifest(output, options, *leader, diagnostics, serial, parallel,
                  expected_checksum, baseline_rss, warmed_rss, final_rss, peak_rss);
    std::cerr << "Wrote " << options.output_path << '\n';
  }
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
