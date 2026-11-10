//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_ORDINARY_TRACE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_ORDINARY_TRACE_VALIDATION_H

#include "PsiFormerDeviceMixedPeriodicGeometryValidation.h"

namespace qmcplusplus::psiformer::device
{
/// Compare complete ordinary VGL, tails, image changes and failures at fixed coordinates.
inline void validateOrdinaryRealFullParallelTraces(compute::Queue<PlatformKind::CUDA>& queue)
{
  const char* key = "PSIFORMER_GPU_REAL_FULL_PARALLEL_TRACES";
  const char* old = std::getenv(key);
  const std::optional<std::string> previous = old ? std::optional<std::string>(old) : std::nullopt;
  struct Restore
  {
    const char* key;
    const std::optional<std::string>& previous;
    ~Restore() { if (previous) setenv(key, previous->c_str(), 1); else unsetenv(key); }
  } restore{key, previous};
  unsetenv(key);
  if (configuredRealFullParallelTraces()) throw std::runtime_error("Parallel ordinary traces changed default");
  setenv(key, "invalid", 1);
  bool invalid_rejected = false;
  try { (void)configuredRealFullParallelTraces(); }
  catch (const std::invalid_argument&) { invalid_rejected = true; }
  if (!invalid_rejected) throw std::runtime_error("Invalid ordinary trace selector accepted");

  AcceleratorBlasHandle blas(queue);
  for (const auto shape_pair : {std::pair<std::size_t, std::size_t>{22, 16}, {22, 64}, {22, 160},
                               {44, 16}, {66, 16}, {88, 16}})
  {
    const auto n = shape_pair.first, determinants = shape_pair.second;
    ExecutionEnvironment environment;
    environment.boundary = BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.lattice_vectors = {{{80., .3, .1}, {0., 80., .2}, {.1, 0., 80.}}};
    environment.periodic_axes = {true, true, true};
    const auto up = 13 * n / 22;
    const ModelShape shape{up, n - up, n, determinants, 16, 4, 1};
    auto parameters = mixed_periodic_validation::parameters(environment, shape);
    const auto plan = PsiFormerExecutionPlan::fromParameters(parameters, shape, environment);
    std::vector<double> nuclei(3 * n), positions(6 * n);
    for (std::size_t e = 0; e < n; ++e)
      for (std::size_t d = 0; d < 3; ++d)
      {
        const auto grid = d == 0 ? e % 5 : d == 1 ? (e / 5) % 5 : e / 25;
        nuclei[3 * e + d] = 1. + 6. * grid;
        for (std::size_t b = 0; b < 2; ++b)
          positions[b * 3 * n + 3 * e + d] = nuclei[3 * e + d] + .06 + .02 * d + .01 * b;
      }
    auto image = std::make_shared<PreparedDeviceModel>(plan, parameters.flat_values(), nuclei, 0);
    setenv(key, "0", 1);
    PreparedDeviceInference reference(image, 2, queue, blas, SpatialJetMode::FULL_VGL);
    setenv(key, "1", 1);
    const auto extra = planOrdinaryRealFullTraces(plan, 2).total_device_bytes;
    PreparedDeviceInference candidate(image, 2, queue, blas, SpatialJetMode::FULL_VGL);
    PreparedDeviceInference value(image, 2, queue, blas, SpatialJetMode::VALUE_ONLY);
    if (reference.usesOrdinaryParallelTraces() || !candidate.usesOrdinaryParallelTraces() ||
        value.usesOrdinaryParallelTraces() || candidate.ordinaryParallelTraceBytes() != extra ||
        candidate.deviceWorkspaceBytes() - reference.deviceWorkspaceBytes() != extra ||
        candidate.hostWorkspaceBytes() != reference.hostWorkspaceBytes())
      throw std::runtime_error("Ordinary parallel trace prepared accounting/selection differs");
    setenv(key, "0", 1); // Prepared owners must retain their cold choice.
    double maximum_error = 0.;
    const auto compare = [&](std::size_t live) {
      const std::vector<double> coordinates(positions.begin(), positions.begin() + live * 3 * n);
      reference.evaluate(coordinates);
      candidate.evaluate(coordinates);
      const auto read = [](PreparedDeviceInference& graph) {
        mixed_periodic_validation::Output out;
        const auto& layout = graph.resultLayout();
        out.phases.resize(layout.configuration_count);
        out.jets.resize(spatialJetSpanElements(layout));
        out.ratios.resize(layout.configuration_count * layout.laplacian_lanes);
        graph.downloadResults(out.phases, out.jets, out.ratios);
        return out;
      };
      const auto a = read(reference), b = read(candidate);
      const auto check = [&](const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size()) throw std::runtime_error("Ordinary parallel trace result shape differs");
        for (std::size_t i = 0; i < x.size(); ++i)
        {
          const auto error = std::abs(x[i] - y[i]);
          maximum_error = std::max(maximum_error, error);
          if (!device_math::isFiniteBinary64(x[i]) || !device_math::isFiniteBinary64(y[i]) ||
              error > 1e-10 + 5e-10 * std::max(std::abs(x[i]), std::abs(y[i])))
            throw std::runtime_error("Ordinary parallel traces changed complete VGL outputs");
        }
      };
      check(a.phases, b.phases); check(a.jets, b.jets); check(a.ratios, b.ratios);
    };
    compare(2);
    int ordinal = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&ordinal), "ordinary trace validation device");
#else
    checkRuntime(cudaGetDevice(&ordinal), "ordinary trace validation device");
#endif
    const auto warm = deviceMemoryMetrics(ordinal).snapshot();
    compare(1); compare(2);
    const auto after = deviceMemoryMetrics(ordinal).snapshot();
    if (after.allocation_calls != warm.allocation_calls || after.live_bytes != warm.live_bytes)
      throw std::runtime_error("Ordinary parallel traces allocated during warm evaluation/tails");
    auto successor = image->makeUpdatedImage(parameters.flat_values(), 1);
    reference.rebindModel(successor); candidate.rebindModel(successor);
    compare(2);
    auto bad = positions;
    bad[0] = std::numeric_limits<double>::infinity();
    for (auto* graph : {&reference, &candidate})
    {
      bool rejected = false;
      try { graph->evaluate(bad); }
      catch (const std::exception&) { rejected = true; }
      if (!rejected) throw std::runtime_error("Ordinary VGL accepted nonfinite coordinates");
      rejected = false;
      try { (void)graph->completedSpatialFinals(); }
      catch (const std::logic_error&) { rejected = true; }
      if (!rejected) throw std::runtime_error("Failed ordinary VGL published stale results");
    }
    compare(1);
    std::cout << "ORDINARY_PARALLEL_TRACES_PASS N=" << n << " D=" << determinants
              << " max_abs_error=" << maximum_error << " added_device_bytes=" << extra
              << " tail_rebind_failure_recovery=yes warm_allocations=0\n";
  }
}
} // namespace qmcplusplus::psiformer::device
#endif
