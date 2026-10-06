//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file benchmark_psiformer_ecp.cpp
 * @brief Non-gating benchmark for flattened, outer-tiled PsiFormer ECP work.
 *
 * The benchmark exercises the production NonLocalECPotential crowd boundary with
 * a deterministic generated pseudo-LiH model.  It sweeps test-only outer-tile
 * capacities so developers can distinguish sparse packing, PsiFormer execution,
 * and weighted score costs without introducing a runtime tuning input.  Timings
 * and retained capacities are diagnostics; this executable is not a CTest.
 */

#include <stdexcept>

// Reuse the deterministic fixture recipe without linking a Catch2 runner.
#define CATCH_CONFIG_PREFIX_ALL
#define REQUIRE(expression)                                                                                         \
  do                                                                                                                \
  {                                                                                                                 \
    if (!(expression))                                                                                              \
      throw std::runtime_error("generated PsiFormer fixture write failed: " #expression);                          \
  } while (false)
#include "QMCWaveFunctions/tests/psiformer_test_utils.h"
#undef REQUIRE

#include "config.h"
#include "git-rev.h"
#include "Message/Communicate.h"
#include "Particle/ParticleSet.h"
#include "Platforms/Host/OutputManager.h"
#include "QMCHamiltonians/ECPComponentBuilder.h"
#include "QMCHamiltonians/NLPPVirtualBatch.h"
#include "QMCHamiltonians/NonLocalECPotential.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "ResourceCollection.h"
#include "Utilities/FakeRandom.h"
#include "Utilities/RuntimeOptions.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
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

#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif

namespace qmcplusplus::testing
{
/** Public copy of the private operator statistics used by this developer tool. */
struct PsiFormerECPBenchmarkDiagnostics
{
  std::size_t logical_jobs                      = 0;
  std::size_t logical_knots                     = 0;
  std::size_t candidate_entries                 = 0;
  std::size_t tiles_packed                      = 0;
  std::size_t split_job_continuations           = 0;
  std::size_t tail_tiles                        = 0;
  std::size_t max_tile_occupancy                = 0;
  std::size_t virtual_batch_storage_fingerprint = 0;
  std::size_t derivative_staging_size           = 0;
  std::size_t derivative_staging_capacity       = 0;
  std::size_t bounded_weight_size               = 0;
  std::size_t bounded_weight_capacity           = 0;
};

/** Narrow friend access to the outer-tile capacity and classified storage. */
class TestNonLocalECPotential
{
public:
  static void setOuterTileCapacity(NonLocalECPotential& potential, std::size_t capacity)
  {
    potential.setOuterTileCapacityForTesting(capacity);
  }

  static PsiFormerECPBenchmarkDiagnostics diagnostics(const NonLocalECPotential& potential)
  {
    const auto statistics = potential.multiWalkerDerivativeStatisticsForTesting();
    return {statistics.logical_jobs,
            statistics.logical_knots,
            0,
            statistics.tiles_packed,
            statistics.split_job_continuations,
            statistics.tail_tiles,
            statistics.max_tile_occupancy,
            statistics.virtual_batch_storage_fingerprint,
            statistics.derivative_staging_size,
            statistics.derivative_staging_capacity,
            statistics.bounded_weight_size,
            statistics.bounded_weight_capacity};
  }

  static std::size_t candidateCount(const NonLocalECPotential& potential)
  {
    return potential.tmove_xy_all_.size();
  }
};

/** Narrow read-only access to the acquired PsiFormer crowd resource. */
class TestPsiFormerWF
{
public:
  static PsiFormerCrowdWorkspaceDiagnostics crowdWorkspaceDiagnostics(
      const PsiFormerWF& component,
      const RefVectorWithLeader<WaveFunctionComponent>& components)
  {
    return component.crowdWorkspaceDiagnosticsForTesting(components);
  }
};
} // namespace qmcplusplus::testing

namespace
{
using Clock = std::chrono::steady_clock;
using qmcplusplus::ECPComponentBuilder;
using qmcplusplus::NonLocalECPotential;
using qmcplusplus::OperatorBase;
using qmcplusplus::OptVariables;
using qmcplusplus::ParticleSet;
using qmcplusplus::PsiFormerWF;
using qmcplusplus::RecordArray;
using qmcplusplus::RefVectorWithLeader;
using qmcplusplus::ResourceCollection;
using qmcplusplus::ResourceCollectionTeamLock;
using qmcplusplus::RuntimeOptions;
using qmcplusplus::SimulationCell;
using qmcplusplus::TrialWaveFunction;
using qmcplusplus::WaveFunctionComponent;
using qmcplusplus::testing::PsiFormerCrowdWorkspaceDiagnostics;
using qmcplusplus::testing::PsiFormerECPBenchmarkDiagnostics;
using qmcplusplus::testing::psiformer::GeneratedFiles;
using qmcplusplus::testing::psiformer::Geometry;
using qmcplusplus::testing::psiformer::generateFiles;
using qmcplusplus::testing::psiformer::makeGeometry;
using Value = qmcplusplus::QMCTraits::ValueType;

#define PSIFORMER_STRINGIFY_DETAIL(value) #value
#define PSIFORMER_STRINGIFY(value) PSIFORMER_STRINGIFY_DETAIL(value)

volatile double benchmark_sink = 0.0;

enum class Mode
{
  ENERGY,
  TMOVE,
  DERIVATIVE
};

struct Options
{
  std::size_t walkers = 2;
  int warmup_calls    = 2;
  int repeats         = 3;
  std::vector<std::size_t> outer_tile_sizes{3, 12, 256};
  std::vector<Mode> modes{Mode::ENERGY, Mode::TMOVE, Mode::DERIVATIVE};
  std::string pseudopotential_path = "Na.BFD.xml";
  std::string output_path;
};

struct TimingResult
{
  Mode mode;
  std::vector<double> raw_seconds;
  std::vector<double> checksums;
  PsiFormerECPBenchmarkDiagnostics ecp;
  PsiFormerCrowdWorkspaceDiagnostics psiformer;
};

struct TileResult
{
  std::size_t outer_tile_capacity = 0;
  std::size_t parameter_count     = 0;
  std::size_t active_parameters   = 0;
  std::vector<TimingResult> modes;
};

[[noreturn]] void usage(const char* executable, const std::string& error = {})
{
  if (!error.empty())
    std::cerr << "error: " << error << '\n';
  std::cerr << "usage: " << executable
            << " [--walkers N] [--outer-tile-sizes N,N,...]"
               " [--modes energy,tmove,derivative] [--warmup-calls N]"
               " [--repeats N] [--pseudopotential FILE] [--output FILE.json]\n";
  std::exit(error.empty() ? EXIT_SUCCESS : EXIT_FAILURE);
}

std::vector<std::string> splitList(const std::string& value)
{
  std::vector<std::string> fields;
  std::size_t begin = 0;
  while (begin <= value.size())
  {
    const std::size_t end = value.find(',', begin);
    const std::string field = value.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (field.empty())
      throw std::invalid_argument("empty comma-separated field");
    fields.push_back(field);
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
  return fields;
}

std::vector<std::size_t> parseTileSizes(const std::string& value)
{
  std::vector<std::size_t> capacities;
  std::set<std::size_t> unique;
  for (const std::string& field : splitList(value))
  {
    const std::size_t capacity = std::stoull(field);
    if (capacity == 0)
      throw std::invalid_argument("outer tile capacities must be positive");
    if (unique.insert(capacity).second)
      capacities.push_back(capacity);
  }
  return capacities;
}

Mode parseMode(const std::string& value)
{
  if (value == "energy")
    return Mode::ENERGY;
  if (value == "tmove")
    return Mode::TMOVE;
  if (value == "derivative")
    return Mode::DERIVATIVE;
  throw std::invalid_argument("unknown mode " + value);
}

std::vector<Mode> parseModes(const std::string& value)
{
  std::vector<Mode> modes;
  std::set<int> unique;
  for (const std::string& field : splitList(value))
  {
    const Mode mode = parseMode(field);
    if (unique.insert(static_cast<int>(mode)).second)
      modes.push_back(mode);
  }
  return modes;
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
      if (key == "--walkers")
        options.walkers = std::stoull(value);
      else if (key == "--outer-tile-sizes")
        options.outer_tile_sizes = parseTileSizes(value);
      else if (key == "--modes")
        options.modes = parseModes(value);
      else if (key == "--warmup-calls")
        options.warmup_calls = std::stoi(value);
      else if (key == "--repeats")
        options.repeats = std::stoi(value);
      else if (key == "--pseudopotential")
        options.pseudopotential_path = value;
      else if (key == "--output")
        options.output_path = value;
      else
        usage(argv[0], "unknown option " + key);
    }
    catch (const std::exception& exception)
    {
      usage(argv[0], "invalid value for " + key + ": " + value + " (" + exception.what() + ")");
    }
  }

  if (options.walkers == 0 || options.warmup_calls < 0 || options.repeats <= 0 ||
      options.outer_tile_sizes.empty() || options.modes.empty() || options.pseudopotential_path.empty())
    usage(argv[0], "walkers, repeats, tile sizes, modes, and pseudopotential are required; warmups are nonnegative");
  return options;
}

const char* modeName(Mode mode)
{
  switch (mode)
  {
  case Mode::ENERGY:
    return "locality_energy";
  case Mode::TMOVE:
    return "tmove_candidates";
  case Mode::DERIVATIVE:
    return "weighted_parameter_derivatives";
  }
  return "unknown";
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

/** Return the hostname associated with this benchmark process. */
std::string hostName()
{
  char hostname[256]{};
  return gethostname(hostname, sizeof(hostname) - 1) == 0 ? hostname : "<unknown>";
}

/** Count processors admitted by the current affinity mask when available. */
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

void writeDoubleArray(std::ostream& output, const std::vector<double>& values)
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

double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
}

ParticleSet makeIons(const SimulationCell& simulation_cell)
{
  const Geometry geometry = makeGeometry("lih_pp");
  ParticleSet ions(simulation_cell);
  ions.setName("ion0");
  ions.create({2});
  auto& species          = ions.getSpeciesSet();
  const int sodium       = species.addSpecies("Na");
  const int charge       = species.addAttribute("charge");
  const int atomic_number = species.addAttribute("atomic_number");
  species(charge, sodium)       = 1.0;
  species(atomic_number, sodium) = 11.0;
  for (int ion = 0; ion < ions.getTotalNum(); ++ion)
    for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
      ions.R[ion][dimension] = geometry.nuclei[OHMMS_DIM * ion + dimension];
  ions.resetGroups();
  ions.update();
  return ions;
}

std::unique_ptr<ParticleSet> makeWalker(const SimulationCell& simulation_cell, std::size_t walker)
{
  const Geometry geometry = makeGeometry("lih_pp");
  auto electrons = std::make_unique<ParticleSet>(simulation_cell);
  electrons->setName("e" + std::to_string(walker));
  electrons->create({1, 1});
  auto& species        = electrons->getSpeciesSet();
  const int up         = species.addSpecies("u");
  const int down       = species.addSpecies("d");
  const int charge     = species.addAttribute("charge");
  const int mass       = species.addAttribute("mass");
  species(charge, up)   = -1.0;
  species(charge, down) = -1.0;
  species(mass, up)     = 1.0;
  species(mass, down)   = 1.0;
  for (int electron = 0; electron < electrons->getTotalNum(); ++electron)
    for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
      electrons->R[electron][dimension] = geometry.electrons[OHMMS_DIM * electron + dimension] +
          0.009 * static_cast<double>(walker) * static_cast<double>((electron + 1) * (dimension + 1));
  electrons->resetGroups();
  electrons->update();
  return electrons;
}

void requireDirectBackends()
{
  for (const char* variable : {"PSIFORMER_VALUE_BACKEND", "PSIFORMER_SCORE_BACKEND"})
  {
    const char* value = std::getenv(variable);
    if (value && std::string(value) != "direct")
      throw std::runtime_error(std::string(variable) + " must be unset or direct for this benchmark");
  }
}

TileResult runTileCase(const Options& options,
                       const GeneratedFiles& files,
                       std::size_t outer_tile_capacity)
{
  const SimulationCell simulation_cell;
  ParticleSet ions = makeIons(simulation_cell);

  std::vector<std::unique_ptr<ParticleSet>> walkers;
  walkers.reserve(options.walkers);
  for (std::size_t walker = 0; walker < options.walkers; ++walker)
    walkers.push_back(makeWalker(simulation_cell, walker));

  RuntimeOptions runtime_options;
  auto wavefunction = std::make_unique<TrialWaveFunction>(runtime_options, "psiformer_ecp_benchmark");
  auto psiformer = std::make_unique<PsiFormerWF>(
      "pf_ecp_benchmark", files.parameters.string(), files.configuration.string(), true,
      std::vector<std::size_t>{0, 127});
  psiformer->validateSystem(*walkers.front(), ions, "pseudopotential");
  wavefunction->addComponent(std::move(psiformer));

  OptVariables active;
  wavefunction->checkInVariables(active);
  active.resetIndex();
  wavefunction->checkOutVariables(active);
  if (active.size_of_active() != 2)
    throw std::runtime_error("generated benchmark expected two selected PsiFormer parameters");

  std::vector<std::unique_ptr<TrialWaveFunction>> wavefunction_clones;
  std::vector<TrialWaveFunction*> wavefunction_ptrs{wavefunction.get()};
  wavefunction_clones.reserve(options.walkers - 1);
  wavefunction_ptrs.reserve(options.walkers);
  for (std::size_t walker = 1; walker < options.walkers; ++walker)
  {
    wavefunction_clones.push_back(wavefunction->makeClone(*walkers[walker]));
    wavefunction_ptrs.push_back(wavefunction_clones.back().get());
  }

  auto potential = std::make_unique<NonLocalECPotential>(
      ions, *walkers.front(), false /* enable_DLA */, true /* use_VP */);
  qmcplusplus::testing::TestNonLocalECPotential::setOuterTileCapacity(
      *potential, outer_tile_capacity);
  ECPComponentBuilder ecp_builder("psiformer_ecp_benchmark", OHMMS::Controller);
  if (!ecp_builder.read_pp_file(options.pseudopotential_path) || !ecp_builder.pp_nonloc)
    throw std::runtime_error("unable to read nonlocal pseudopotential " + options.pseudopotential_path);
  potential->addComponent(0, std::move(ecp_builder.pp_nonloc));

  std::vector<std::unique_ptr<OperatorBase>> potential_clones;
  std::vector<OperatorBase*> potential_ptrs{potential.get()};
  potential_clones.reserve(options.walkers - 1);
  potential_ptrs.reserve(options.walkers);
  for (std::size_t walker = 1; walker < options.walkers; ++walker)
  {
    potential_clones.push_back(potential->makeClone(*walkers[walker], *wavefunction_ptrs[walker]));
    potential_ptrs.push_back(potential_clones.back().get());
  }

  using FullPrecReal = qmcplusplus::QMCTraits::FullPrecRealType;
  std::vector<std::unique_ptr<qmcplusplus::FakeRandom<FullPrecReal>>> random_generators;
  random_generators.reserve(options.walkers);
  for (std::size_t walker = 0; walker < options.walkers; ++walker)
  {
    random_generators.push_back(std::make_unique<qmcplusplus::FakeRandom<FullPrecReal>>());
    random_generators.back()->set_value(0.371);
    potential_ptrs[walker]->setRandomGenerator(random_generators.back().get());
    walkers[walker]->update();
  }

  RefVectorWithLeader<ParticleSet> particle_list(*walkers.front());
  RefVectorWithLeader<TrialWaveFunction> wavefunction_list(*wavefunction);
  RefVectorWithLeader<OperatorBase> potential_list(*potential);
  auto& component_leader = *wavefunction->getOrbitals().front();
  RefVectorWithLeader<WaveFunctionComponent> component_list(component_leader);
  particle_list.reserve(options.walkers);
  wavefunction_list.reserve(options.walkers);
  potential_list.reserve(options.walkers);
  component_list.reserve(options.walkers);
  for (std::size_t walker = 0; walker < options.walkers; ++walker)
  {
    particle_list.push_back(*walkers[walker]);
    wavefunction_list.push_back(*wavefunction_ptrs[walker]);
    potential_list.push_back(*potential_ptrs[walker]);
    component_list.push_back(*wavefunction_ptrs[walker]->getOrbitals().front());
  }

  ResourceCollection particle_resources("psiformer_ecp_benchmark_particles");
  ResourceCollection wavefunction_resources("psiformer_ecp_benchmark_wavefunctions");
  ResourceCollection potential_resources("psiformer_ecp_benchmark_potential");
  walkers.front()->createResource(particle_resources);
  wavefunction->createResource(wavefunction_resources);
  potential->createResource(potential_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particle_list);
  ResourceCollectionTeamLock<TrialWaveFunction> wavefunction_lock(wavefunction_resources, wavefunction_list);
  ResourceCollectionTeamLock<OperatorBase> potential_lock(potential_resources, potential_list);

  ParticleSet::mw_update(particle_list);
  TrialWaveFunction::mw_evaluateLog(wavefunction_list, particle_list);

  const std::size_t active_count = static_cast<std::size_t>(active.size_of_active());
  RecordArray<Value> scores(options.walkers, active_count);
  RecordArray<Value> derivatives(options.walkers, active_count);
  std::fill(scores.begin(), scores.end(), Value(0));

  auto invoke = [&](Mode mode) {
    if (mode == Mode::ENERGY)
      potential->mw_evaluate(potential_list, wavefunction_list, particle_list);
    else if (mode == Mode::TMOVE)
      potential->mw_evaluateWithToperator(potential_list, wavefunction_list, particle_list);
    else
      potential->mw_evaluateWithParameterDerivatives(
          potential_list, wavefunction_list, particle_list, active, scores, derivatives);

    double checksum = 0.0;
    for (std::size_t walker = 0; walker < options.walkers; ++walker)
      checksum += static_cast<double>(walker + 1) * potential_ptrs[walker]->getValue();
    if (mode == Mode::DERIVATIVE)
      for (std::size_t walker = 0; walker < options.walkers; ++walker)
        for (std::size_t parameter = 0; parameter < active_count; ++parameter)
        {
          const std::size_t index = walker * active_count + parameter;
          checksum += (1.0 + static_cast<double>(index) / derivatives.size()) *
              std::real(derivatives[walker][parameter]);
        }
    if (!std::isfinite(checksum))
      throw std::runtime_error(std::string("non-finite benchmark checksum for ") + modeName(mode));
    return checksum;
  };

  TileResult tile_result;
  tile_result.outer_tile_capacity = outer_tile_capacity;
  tile_result.active_parameters   = active_count;
  const auto& component = dynamic_cast<const PsiFormerWF&>(component_leader);
  tile_result.parameter_count = component.parameterSchema().parameterCount();

  for (const Mode mode : options.modes)
  {
    for (int warmup = 0; warmup < options.warmup_calls; ++warmup)
    {
      if (mode == Mode::DERIVATIVE)
        std::fill(derivatives.begin(), derivatives.end(), Value(0));
      benchmark_sink += invoke(mode);
    }

    TimingResult result;
    result.mode = mode;
    result.raw_seconds.reserve(options.repeats);
    result.checksums.reserve(options.repeats);
    for (int repeat = 0; repeat < options.repeats; ++repeat)
    {
      if (mode == Mode::DERIVATIVE)
        std::fill(derivatives.begin(), derivatives.end(), Value(0));
      const auto start = Clock::now();
      const double checksum = invoke(mode);
      result.raw_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
      result.checksums.push_back(checksum);
      benchmark_sink += checksum;
    }
    result.ecp = qmcplusplus::testing::TestNonLocalECPotential::diagnostics(*potential);
    for (const OperatorBase* base_potential : potential_ptrs)
    {
      const auto& ecp_potential = dynamic_cast<const NonLocalECPotential&>(*base_potential);
      if (mode == Mode::TMOVE)
        result.ecp.candidate_entries +=
            qmcplusplus::testing::TestNonLocalECPotential::candidateCount(ecp_potential);
    }
    result.psiformer = qmcplusplus::testing::TestPsiFormerWF::crowdWorkspaceDiagnostics(
        component, component_list);
    tile_result.modes.push_back(std::move(result));
  }
  return tile_result;
}

void writeECPDiagnostics(std::ostream& output, const PsiFormerECPBenchmarkDiagnostics& diagnostic)
{
  output << "{\"logical_jobs\":" << diagnostic.logical_jobs
         << ",\"logical_knots\":" << diagnostic.logical_knots
         << ",\"candidate_entries\":" << diagnostic.candidate_entries
         << ",\"tiles_packed\":" << diagnostic.tiles_packed
         << ",\"split_job_continuations\":" << diagnostic.split_job_continuations
         << ",\"tail_tiles\":" << diagnostic.tail_tiles
         << ",\"max_tile_occupancy\":" << diagnostic.max_tile_occupancy
         << ",\"virtual_batch_storage_fingerprint\":"
         << diagnostic.virtual_batch_storage_fingerprint
         << ",\"derivative_staging_size\":" << diagnostic.derivative_staging_size
         << ",\"derivative_staging_capacity\":" << diagnostic.derivative_staging_capacity
         << ",\"bounded_weight_size\":" << diagnostic.bounded_weight_size
         << ",\"bounded_weight_capacity\":" << diagnostic.bounded_weight_capacity << '}';
}

void writePsiFormerDiagnostics(std::ostream& output, const PsiFormerCrowdWorkspaceDiagnostics& diagnostic)
{
  output << "{\"persistent_model_identity\":" << diagnostic.persistent_model_identity
         << ",\"parameter_version\":" << diagnostic.parameter_version
         << ",\"batch_bytes\":" << diagnostic.batch_bytes
         << ",\"score_bytes\":" << diagnostic.score_bytes
         << ",\"kinetic_bytes\":" << diagnostic.kinetic_bytes
         << ",\"transient_bytes\":" << diagnostic.transient_bytes
         << ",\"accounted_bytes\":" << diagnostic.accountedBytes()
         << ",\"reference_configurations\":" << diagnostic.reference_configurations
         << ",\"replacement_configurations\":" << diagnostic.replacement_configurations
         << ",\"reference_evaluations\":" << diagnostic.reference_evaluations
         << ",\"dense_coordinate_bytes_avoided\":" << diagnostic.dense_coordinate_bytes_avoided
         << ",\"weighted_reference_configurations\":" << diagnostic.weighted_reference_configurations
         << ",\"weighted_replacement_configurations\":" << diagnostic.weighted_replacement_configurations
         << ",\"weighted_active_parameters\":" << diagnostic.weighted_active_parameters
         << ",\"weighted_derivative_staging_bytes\":" << diagnostic.weighted_derivative_staging_bytes
         << ",\"value_backend\":" << jsonString(diagnostic.backend_modes[0])
         << ",\"score_backend\":" << jsonString(diagnostic.backend_modes[2]) << '}';
}

void writeManifest(std::ostream& output,
                   const Options& options,
                   const std::vector<TileResult>& tile_results)
{
  output << "{\n"
         << "  \"schema\":\"qmcpack.psiformer.ecp_benchmark.v1\",\n"
         << "  \"non_gating\":true,\n"
         << "  \"workload\":{\"system\":\"lih_pp\",\"walkers\":" << options.walkers
         << ",\"fixture_recipe_version\":"
         << qmcplusplus::testing::psiformer::FIXTURE_RECIPE_VERSION
         << ",\"pseudopotential\":" << jsonString(options.pseudopotential_path)
         << ",\"warmup_calls\":" << options.warmup_calls
         << ",\"repeats\":" << options.repeats << "},\n"
         << "  \"execution\":{\"forward\":\"flattened sparse outer tiles\","
         << "\"weighted_reduction\":\"flattened direct sink\","
         << "\"reverse\":\"serialized resource-owned score tape; not a grouped batched reverse kernel\","
         << "\"timing_scope\":\"NonLocalECPotential crowd call; reference VGL and output reset excluded\"},\n"
         << "  \"tile_cases\":[\n";
  for (std::size_t tile = 0; tile < tile_results.size(); ++tile)
  {
    const TileResult& tile_result = tile_results[tile];
    output << "    {\"outer_tile_capacity\":" << tile_result.outer_tile_capacity
           << ",\"parameter_count\":" << tile_result.parameter_count
           << ",\"active_parameters\":" << tile_result.active_parameters
           << ",\"modes\":[\n";
    for (std::size_t mode = 0; mode < tile_result.modes.size(); ++mode)
    {
      const TimingResult& result = tile_result.modes[mode];
      const double median_seconds = median(result.raw_seconds);
      output << "      {\"name\":" << jsonString(modeName(result.mode))
             << ",\"raw_seconds\":";
      writeDoubleArray(output, result.raw_seconds);
      output << ",\"median_seconds\":" << std::setprecision(17) << median_seconds
             << ",\"median_jobs_per_second\":"
             << (median_seconds > 0.0 ? result.ecp.logical_jobs / median_seconds : 0.0)
             << ",\"median_knots_per_second\":"
             << (median_seconds > 0.0 ? result.ecp.logical_knots / median_seconds : 0.0)
             << ",\"checksums\":";
      writeDoubleArray(output, result.checksums);
      output << ",\"ecp_storage\":";
      writeECPDiagnostics(output, result.ecp);
      output << ",\"psiformer_resource\":";
      writePsiFormerDiagnostics(output, result.psiformer);
      output << '}' << (mode + 1 == tile_result.modes.size() ? "\n" : ",\n");
    }
    output << "    ]}" << (tile + 1 == tile_results.size() ? "\n" : ",\n");
  }
  output << "  ],\n"
         << "  \"provenance\":{\"qmcpack_version\":"
         << jsonString(std::to_string(QMCPACK_VERSION_MAJOR) + "." +
                       std::to_string(QMCPACK_VERSION_MINOR) + "." +
                       std::to_string(QMCPACK_VERSION_PATCH))
         << ",\"qmcpack_git_revision\":" << jsonString(PSIFORMER_STRINGIFY(GIT_HASH_RAW))
         << ",\"compiler\":" << jsonString(__VERSION__)
         << ",\"host\":" << jsonString(hostName())
         << ",\"build_complex\":"
#ifdef QMC_COMPLEX
         << "true"
#else
         << "false"
#endif
         << ",\"pid\":" << getpid()
         << ",\"real_type_bytes\":" << sizeof(qmcplusplus::QMCTraits::RealType)
         << ",\"value_type_bytes\":" << sizeof(Value)
         << ",\"full_precision_real_bytes\":"
         << sizeof(qmcplusplus::QMCTraits::FullPrecRealType)
         << ",\"hardware_concurrency\":" << std::thread::hardware_concurrency()
         << ",\"affinity_cpu_count\":" << affinityCpuCount()
         << ",\"thread_environment\":{\"OMP_NUM_THREADS\":"
         << jsonString(environmentValue("OMP_NUM_THREADS"))
         << ",\"OMP_MAX_ACTIVE_LEVELS\":"
         << jsonString(environmentValue("OMP_MAX_ACTIVE_LEVELS"))
         << ",\"OMP_PROC_BIND\":" << jsonString(environmentValue("OMP_PROC_BIND"))
         << ",\"OMP_PLACES\":" << jsonString(environmentValue("OMP_PLACES"))
         << ",\"OPENBLAS_NUM_THREADS\":" << jsonString(environmentValue("OPENBLAS_NUM_THREADS"))
         << ",\"GOTO_NUM_THREADS\":" << jsonString(environmentValue("GOTO_NUM_THREADS"))
         << ",\"MKL_NUM_THREADS\":" << jsonString(environmentValue("MKL_NUM_THREADS"))
         << ",\"MKL_DYNAMIC\":" << jsonString(environmentValue("MKL_DYNAMIC"))
         << ",\"BLIS_NUM_THREADS\":" << jsonString(environmentValue("BLIS_NUM_THREADS"))
         << ",\"VECLIB_MAXIMUM_THREADS\":"
         << jsonString(environmentValue("VECLIB_MAXIMUM_THREADS")) << "}},\n"
         << "  \"notes\":["
         << jsonString("outer tile capacities are a diagnostic sweep, not a production input knob") << ','
         << jsonString("reported capacities and byte counts classify retained storage and are not process RSS") << ','
         << jsonString("timings carry no absolute threshold and are not registered with CTest") << "]\n"
         << "}\n";
}

int run(int argc, char** argv)
{
  const Options options = parseOptions(argc, argv);
  requireDirectBackends();
  GeneratedFiles files = generateFiles("lih_pp");
  std::vector<TileResult> tile_results;
  tile_results.reserve(options.outer_tile_sizes.size());
  for (const std::size_t capacity : options.outer_tile_sizes)
  {
    TileResult combined;
    combined.outer_tile_capacity = capacity;
    for (const Mode mode : options.modes)
    {
      // Build each mode's crowd independently so retained high-water storage
      // and lazy score tapes cannot leak from an earlier mode into its report.
      Options isolated_options = options;
      isolated_options.modes   = {mode};
      TileResult isolated      = runTileCase(isolated_options, files, capacity);
      if (combined.modes.empty())
      {
        combined.parameter_count   = isolated.parameter_count;
        combined.active_parameters = isolated.active_parameters;
      }
      combined.modes.push_back(std::move(isolated.modes.front()));
    }
    tile_results.push_back(std::move(combined));
  }

  if (options.output_path.empty())
    writeManifest(std::cout, options, tile_results);
  else
  {
    std::ofstream output(options.output_path);
    if (!output)
      throw std::runtime_error("unable to open report " + options.output_path);
    writeManifest(output, options, tile_results);
    std::cerr << "Wrote " << options.output_path << '\n';
  }
  return EXIT_SUCCESS;
}
} // namespace

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
  mpi3::environment environment(argc, argv);
  OHMMS::Controller = new Communicate(environment.world());
#endif
  // Keep stdout a valid JSON stream when --output is omitted. Diagnostics and
  // failures below use std::cerr and remain visible.
  outputManager.shutOff();
  int result = EXIT_FAILURE;
  try
  {
    if (OHMMS::Controller->size() != 1)
      throw std::runtime_error("benchmark_psiformer_ecp must be run with one MPI rank");
    result = run(argc, argv);
  }
  catch (const std::exception& exception)
  {
    std::cerr << "error: " << exception.what() << '\n';
  }
#ifdef HAVE_MPI
  OHMMS::Controller->finalize();
#endif
  return result;
}
