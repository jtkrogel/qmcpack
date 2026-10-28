//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file psiformer_complex_cpu_probe.cpp
 * @brief Opt-in physical-owner and complete complex spatial CPU component probe.
 * This is not registered as a timing test and performs no statistical trajectory.
 */
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexSpatialExecutor.h"
#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>

namespace
{
namespace ps = qmcplusplus::psiformer;
using C = std::complex<double>;
using Clock = std::chrono::steady_clock;

/// Strict sizes keep accidental large or unbounded probe requests explicit.
std::size_t positive(const char* text)
{
  std::size_t result = 0;
  const auto end = text+std::strlen(text);
  const auto parsed = std::from_chars(text, end, result);
  if (parsed.ec != std::errc{} || parsed.ptr != end || !result || result == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("probe sizes must be positive finite decimal integers");
  return result;
}

/// Process RSS complements exact owner capacities; it includes metadata and allocator overhead.
std::size_t resident(const std::string& key)
{
  std::ifstream input("/proc/self/status");
  std::string line;
  while (std::getline(input, line))
    if (line.rfind(key+":", 0) == 0)
    {
      std::istringstream values(line.substr(key.size()+1));
      std::size_t kib = 0;
      values >> kib;
      return kib*1024;
    }
  return 0;
}

double elapsed(Clock::time_point begin) { return std::chrono::duration<double>(Clock::now()-begin).count(); }

/// Count persistent native double arrays once, independently of derivative owners.
std::size_t modelArrays(const pf::PsiFormer& model)
{
  std::size_t count = model.p.values.capacity()+model.cfg.electrons.x.capacity()+
      model.cfg.nuclei.x.capacity()+model.cfg.charges.x.capacity();
  std::set<const pf::Node*> visited;
  for (const auto& entry : model.p.nodes)
    if (visited.insert(entry.second.get()).second)
      count += entry.second->value.x.capacity()+entry.second->d1.x.capacity()+entry.second->d2.x.capacity();
  return pf::checkedStorageBytes<double>(count, "probe model array extent overflow");
}

/// Materialize only complete outputs for comparison; these O(Ne) reference arrays are reported.
struct Products
{
  ps::determinant::ComplexLogAmplitude amplitude;
  std::vector<C> gradient, lap_log, lap_ratio;
};
Products copy(const pf::DirectComplexSpatialResult& result)
{
  Products output{result.amplitude,
      {result.gradient, result.gradient+result.gradient_count},
      {result.laplacian_log, result.laplacian_log+result.laplacian_count},
      {result.laplacian_ratio, result.laplacian_ratio+result.laplacian_count}};
  for (const auto* values : {&output.gradient, &output.lap_log, &output.lap_ratio})
    for (const auto value : *values)
      if (!pf::is_finite_parameter_value(value.real()) || !pf::is_finite_parameter_value(value.imag()))
        throw std::runtime_error("probe encountered nonfinite complete spatial products");
  return output;
}

/// Report absolute and scaled errors without discarding phase or ratio-Laplacian components.
std::pair<double,double> difference(const Products& actual, const Products& reference)
{
  double absolute = std::abs(actual.amplitude.log_abs-reference.amplitude.log_abs);
  double scaled = absolute/(1+std::abs(reference.amplitude.log_abs));
  const auto add = [&](C a, C b) {
    absolute = std::max(absolute, std::abs(a-b));
    scaled = std::max(scaled, std::abs(a-b)/(1+std::abs(b)));
  };
  add(actual.amplitude.phase, reference.amplitude.phase);
  for (const auto pair : {std::pair{&actual.gradient,&reference.gradient},
      std::pair{&actual.lap_log,&reference.lap_log}, std::pair{&actual.lap_ratio,&reference.lap_ratio}})
  {
    if (pair.first->size() != pair.second->size()) throw std::logic_error("probe output extent mismatch");
    for (std::size_t i = 0; i < pair.first->size(); ++i) add((*pair.first)[i], (*pair.second)[i]);
  }
  return {absolute,scaled};
}
} // namespace

int main(int argc, char** argv)
{
  try
  {
    if (argc != 6) throw std::invalid_argument("usage: probe fixture.h5 targets repetitions compare|memory|plan owner_byte_cap");
    const auto targets = positive(argv[2]), repetitions = positive(argv[3]), cap = positive(argv[5]);
    const std::string mode = argv[4];
    if (mode != "compare" && mode != "memory" && mode != "plan") throw std::invalid_argument("invalid probe operation");
    const auto setup_begin = Clock::now();
    pf::ConfigData configuration(argv[1]);
    const hid_t file = H5Fopen(argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("probe fixture cannot be opened");
    const auto lattice = pf::read_double(file, "/lattice");
    const auto twist = pf::read_double(file, "/cartesian_twist");
    H5Fclose(file);
    if (lattice.size() != 9 || twist.size() != 3) throw std::invalid_argument("probe fixture lattice/twist extent mismatch");
    ps::ExecutionEnvironment environment;
    environment.boundary = ps::BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = ps::GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.periodic_axes = {true,true,true};
    environment.head_representation = ps::HeadRepresentation::PAIRED_REAL_COMPLEX;
    environment.amplitude_scalar_domain = ps::ScalarDomain::COMPLEX;
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j) environment.lattice_vectors[i][j] = lattice[3*i+j];
    const ps::ModelShape shape{configuration.nup, configuration.ndown, configuration.nuclei.shape[0], 16, 256, 4, 4};
    auto initialized = ps::initializePsiFormerParameters(shape, 17, ps::PSIFORMER_COMPLEX_PERIODIC_V1, environment);
    std::vector<pf::Layout> layouts;
    for (const auto& layout : initialized.layouts)
      layouts.push_back({layout.module, layout.name, layout.shape, layout.begin, layout.end});
    pf::PsiFormer model(pf::Parameters(std::move(initialized.values), std::move(layouts)), std::move(configuration), 16, 256, 4, 4);
    const auto plan = ps::PsiFormerExecutionPlan::fromParameters(model.p, shape, environment);
    pf::DirectComplexSpatialExecutor executor(model, plan);
    const pf::PsiFormerStorageShape storage{model.ne, shape.nuclei, 16, 256, 4,
        plan.parameter(ps::ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0], 4, model.p.size()};
    const auto full_bytes = pf::DirectComplexSpatialWorkspace::storageRequirement(storage, pf::DirectSpatialMode::FULL_VGL);
    const auto stream_bytes = pf::DirectComplexStreamedSpatialWorkspace::storageRequirement(storage, targets);
    std::cout << std::setprecision(17) << "{\"event\":\"admission\",\"electrons\":" << model.ne
              << ",\"nuclei\":" << shape.nuclei << ",\"determinants\":16,\"features\":256,\"heads\":4,\"blocks\":4"
              << ",\"targets\":" << targets << ",\"parameter_count\":" << model.p.size()
              << ",\"model_double_array_bytes\":" << modelArrays(model)
              << ",\"full_owner_bytes\":" << full_bytes << ",\"streamed_owner_bytes\":" << stream_bytes
              << ",\"owner_cap_bytes\":" << cap << ",\"rss_bytes\":" << resident("VmRSS")
              << ",\"setup_seconds\":" << elapsed(setup_begin) << ",\"mode\":" << std::quoted(mode) << "}\n" << std::flush;
    if (mode == "plan") return 0;
    if (full_bytes > cap || stream_bytes > cap) throw std::length_error("probe physical owner exceeds its byte cap before allocation");
    const auto positions = pf::GeometryPositionView::interleaved(model.cfg.electrons.x.data(), model.ne);
    Products reference;
    auto start = Clock::now();
    auto full = executor.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
    if (full->vectorStorageBytes() != full_bytes) throw std::logic_error("probe FULL accounting mismatch");
    full->setPositions(positions);
    std::cout << "{\"event\":\"full_allocated\",\"allocation_seconds\":" << elapsed(start)
              << ",\"physical_owner_bytes\":" << full->vectorStorageBytes() << ",\"rss_bytes\":" << resident("VmRSS") << "}\n" << std::flush;
    const auto full_fingerprint = full->storageFingerprint();
    if (mode == "compare")
      for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
      {
        start = Clock::now();
        const auto result = executor.evaluateFull(*full);
        const auto seconds = elapsed(start);
        reference = copy(result);
        if (full->storageFingerprint() != full_fingerprint) throw std::logic_error("FULL owner changed allocation on warm call");
        std::cout << "{\"event\":\"full_evaluated\",\"repeat\":" << repeat << ",\"seconds\":" << seconds
                  << ",\"parameter_version\":" << result.parameter_version << ",\"gradient_count\":" << result.gradient_count
                  << ",\"laplacian_count\":" << result.laplacian_count << "}\n" << std::flush;
      }
    full.reset();
    start = Clock::now();
    auto stream = executor.makeStreamedWorkspace(targets, stream_bytes);
    if (stream->vectorStorageBytes() != stream_bytes) throw std::logic_error("probe streamed accounting mismatch");
    stream->setPositions(positions);
    std::cout << "{\"event\":\"streamed_allocated\",\"allocation_seconds\":" << elapsed(start)
              << ",\"physical_owner_bytes\":" << stream->vectorStorageBytes() << ",\"rss_bytes\":" << resident("VmRSS") << "}\n" << std::flush;
    const auto stream_fingerprint = stream->storageFingerprint();
    if (mode == "compare")
      for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
      {
        start = Clock::now();
        const auto result = executor.evaluateFull(*stream);
        const auto seconds = elapsed(start);
        const auto output = copy(result);
        const auto error = difference(output, reference);
        if (!pf::is_finite_parameter_value(error.second) || error.second > 2e-9)
          throw std::runtime_error("probe FULL/streamed complete-product comparison failed");
        if (stream->storageFingerprint() != stream_fingerprint) throw std::logic_error("streamed owner changed allocation on warm call");
        C kinetic{};
        for (std::size_t e = 0; e < model.ne; ++e)
        {
          C ratio = output.lap_log[e];
          for (std::size_t d = 0; d < 3; ++d)
          {
            const C physical_gradient = output.gradient[3*e+d]+C{0,twist[d]};
            ratio += physical_gradient*physical_gradient;
          }
          kinetic -= .5*ratio;
        }
        std::cout << "{\"event\":\"streamed_evaluated\",\"repeat\":" << repeat << ",\"seconds\":" << seconds
                  << ",\"max_absolute_error\":" << error.first << ",\"max_scaled_error\":" << error.second
                  << ",\"complete_vgl_finite\":true,\"physical_kinetic_real\":" << kinetic.real()
                  << ",\"physical_kinetic_imag\":" << kinetic.imag() << ",\"parameter_version\":" << result.parameter_version << "}\n" << std::flush;
      }
    std::cout << "{\"event\":\"complete\",\"peak_rss_bytes\":" << resident("VmHWM")
              << ",\"comparison_reference_bytes\":" << (reference.gradient.capacity()+reference.lap_log.capacity()+reference.lap_ratio.capacity())*sizeof(C)
              << ",\"views_allocate_bytes\":0}\n";
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "{\"event\":\"error\",\"message\":" << std::quoted(error.what()) << "}\n";
    return 1;
  }
}
