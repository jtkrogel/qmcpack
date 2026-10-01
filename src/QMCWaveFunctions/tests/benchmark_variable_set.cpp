//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file benchmark_variable_set.cpp
 * @brief Non-gating development benchmark for large parameter registration.
 */
#include "Message/Communicate.h"
#include "OhmmsData/Libxml2Doc.h"
#include "QMCDrivers/Optimizers/DescentEngine.h"
#include "QMCWaveFunctions/VariableSet.h"
#include "io/hdf/hdf_archive.h"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <sys/resource.h>

namespace
{
using Clock = std::chrono::steady_clock;
using VariableSet = optimize::VariableSet;

/// Return elapsed seconds since start.
double elapsedSeconds(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// Report the process peak resident set in MiB on Linux.
double peakResidentMiB()
{
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    throw std::runtime_error("getrusage failed");
  return static_cast<double>(usage.ru_maxrss) / 1024.0;
}

/// Generate compact deterministic names resembling PsiFormer scalar names.
std::vector<VariableSet::pair_type> makeVariables(std::size_t count)
{
  std::vector<VariableSet::pair_type> variables;
  variables.reserve(count);
  for (std::size_t variable_index = 0; variable_index < count; ++variable_index)
    variables.emplace_back("pf_pf_" + std::to_string(variable_index),
                           static_cast<VariableSet::real_type>(variable_index) * 1e-8);
  return variables;
}

/// Print one timing and the cumulative peak resident set after that phase.
void report(const std::string& phase, double seconds)
{
  std::cout << std::left << std::setw(24) << phase << " seconds=" << std::right << std::setw(12)
            << std::setprecision(6) << std::fixed << seconds << " peak_mib=" << std::setw(12)
            << std::setprecision(3) << peakResidentMiB() << '\n';
}
} // namespace

/// Exercise the full scalar registration/copy/serialization path once.
int main(int argc, char** argv)
{
  const std::size_t parameter_count = argc > 1 ? std::stoull(argv[1]) : 100000;
  const bool benchmark_hdf         = argc > 2 && std::string(argv[2]) == "--hdf";

  std::cout << "VariableSet scalable-registration benchmark\n"
            << "parameters=" << parameter_count << " hdf=" << (benchmark_hdf ? "yes" : "no") << '\n';

  auto start     = Clock::now();
  auto variables = makeVariables(parameter_count);
  report("prepare_names", elapsedSeconds(start));

  VariableSet local;
  start = Clock::now();
  local.insertBulk(std::move(variables), true, optimize::OTHER_P);
  report("bulk_registration", elapsedSeconds(start));

  VariableSet global;
  global.insert("ordinary_before", -1.0);
  start = Clock::now();
  global.insertFrom(local);
  global.insert("ordinary_after", 1.0);
  report("global_insert_from", elapsedSeconds(start));

  start = Clock::now();
  global.resetIndex();
  report("global_reset_index", elapsedSeconds(start));

  start = Clock::now();
  local.getIndex(global);
  report("local_checkout", elapsedSeconds(start));

  start = Clock::now();
  VariableSet initial_variables(global);
  report("init_variables_copy", elapsedSeconds(start));

  // Exercise the real descent setup path because it materializes its own
  // ordered names, types, and value vectors from the global VariableSet.
  Libxml2Document optimizer_document;
  if (!optimizer_document.parseFromString("<optimize/>"))
    throw std::runtime_error("could not construct benchmark optimizer XML");
  qmcplusplus::DescentEngine descent_engine(OHMMS::Controller, optimizer_document.getRoot());
  start = Clock::now();
  descent_engine.setupUpdate(global);
  report("descent_setup_update", elapsedSeconds(start));
  if (descent_engine.retrieveNewParams().size() != global.size_of_active())
    throw std::runtime_error("descent setup did not retain every active parameter");

  if (parameter_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("benchmark parameter count exceeds DescentEngine's integer interface");
  start = Clock::now();
  descent_engine.prepareStorage(1, static_cast<int>(global.size_of_active()));
  report("descent_storage_r1", elapsedSeconds(start));

  if (benchmark_hdf)
  {
    const std::filesystem::path output_path =
        std::filesystem::temp_directory_path() /
        ("qmcpack_variableset_benchmark_" + std::to_string(parameter_count) + ".h5");
    std::error_code error;
    std::filesystem::remove(output_path, error);

    qmcplusplus::hdf_archive output;
    start = Clock::now();
    global.writeToHDF(output_path.string(), output);
    output.close();
    report("hdf_output", elapsedSeconds(start));

    std::filesystem::remove(output_path, error);
    if (error)
      std::cerr << "warning: could not remove benchmark file " << output_path << ": " << error.message() << '\n';
  }

  std::cout << "checksum=" << local.size_of_active() + global.size_of_active() + initial_variables.size_of_active()
            << '\n';
  return EXIT_SUCCESS;
}
