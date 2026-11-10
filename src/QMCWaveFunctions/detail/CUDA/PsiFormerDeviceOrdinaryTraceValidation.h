//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_ORDINARY_TRACE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_ORDINARY_TRACE_VALIDATION_H

#include <initializer_list>
#include "PsiFormerDeviceMixedPeriodicGeometryValidation.h"

namespace qmcplusplus::psiformer::device
{
/// Compare complete ordinary VGL, tails, image changes and failures at fixed coordinates.
inline void validateOrdinaryRealFullParallelTraces(compute::Queue<PlatformKind::CUDA>& queue)
{
  const char* key = "PSIFORMER_GPU_REAL_FULL_PARALLEL_TRACES";
  struct Restore
  {
    const char* key;
    std::optional<std::string> previous;
    explicit Restore(const char* name) : key(name)
    {
      if (const char* old = std::getenv(key)) previous = old;
    }
    ~Restore() { if (previous) setenv(key, previous->c_str(), 1); else unsetenv(key); }
  } restore(key), cooperative("PSIFORMER_GPU_COOPERATIVE_SPATIAL_TRACES"),
      targets("PSIFORMER_GPU_SMALL_MATRIX_SPATIAL_TARGETS");
  // Use the qualified production reference, rather than an accidentally serial
  // one-thread-per-configuration oracle at the larger endpoint sizes.
  setenv(cooperative.key, "1", 1);
  setenv(targets.key, "1", 1);
  unsetenv(key);
  if (!configuredRealFullParallelTraces()) throw std::runtime_error("Parallel ordinary traces must default ON");
  setenv(key, "invalid", 1);
  bool invalid_rejected = false;
  try { (void)configuredRealFullParallelTraces(); }
  catch (const std::invalid_argument&) { invalid_rejected = true; }
  if (!invalid_rejected) throw std::runtime_error("Invalid ordinary trace selector accepted");
  setenv(key, "0", 1);

  AcceleratorBlasHandle blas(queue);
  for (const auto boundary : {BoundaryCondition::PERIODIC, BoundaryCondition::OPEN})
  for (const auto shape_pair : std::initializer_list<std::pair<std::size_t, std::size_t>>{std::pair<std::size_t, std::size_t>{22, 16}, {22, 64}, {22, 160},
                               {44, 16}, {66, 16}, {88, 16}})
  {
    const auto n = shape_pair.first, determinants = shape_pair.second;
    if (boundary == BoundaryCondition::OPEN && (n != 22 || determinants != 16)) continue;
    ExecutionEnvironment environment;
    environment.boundary = boundary;
    environment.geometry_feature_policy = boundary == BoundaryCondition::OPEN
        ? GeometryFeaturePolicy::OPEN_EUCLIDEAN_V1 : GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.lattice_vectors = {{{80., .3, .1}, {0., 80., .2}, {.1, 0., 80.}}};
    environment.periodic_axes = boundary == BoundaryCondition::PERIODIC
        ? std::array<bool, 3>{true, true, true} : std::array<bool, 3>{false, false, false};
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
    if (boundary == BoundaryCondition::PERIODIC && n == 22 && determinants == 16)
    {
      auto complex_environment = environment;
      complex_environment.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
      complex_environment.amplitude_scalar_domain = ScalarDomain::COMPLEX;
      auto complex_parameters = mixed_periodic_validation::parameters(complex_environment, shape);
      const auto complex_plan = PsiFormerExecutionPlan::fromParameters(complex_parameters, shape, complex_environment);
      unsetenv(key);
      if (planOrdinaryRealFullTraces(complex_plan, 2).total_device_bytes)
        throw std::runtime_error("Automatic ordinary REAL trace planner admitted paired-complex outputs");
    }
    setenv(key, "0", 1);
    PreparedDeviceInference reference(image, 2, queue, blas, SpatialJetMode::FULL_VGL);
    setenv(key, "1", 1);
    const auto extra = planOrdinaryRealFullTraces(plan, 2).total_device_bytes;
    PreparedDeviceInference candidate(image, 2, queue, blas, SpatialJetMode::FULL_VGL);
    PreparedDeviceInference value(image, 2, queue, blas, SpatialJetMode::VALUE_ONLY);
    unsetenv(key);
    PreparedDeviceInference default_candidate(image, 2, queue, blas, SpatialJetMode::FULL_VGL);
    PreparedDeviceInference mixed_owner(image, 2, queue, blas, SpatialJetMode::FULL_VGL,
        false, false, configuredInferenceStatusPolicy(), {}, OrdinaryTracePreparation::EXCLUDE_FOR_MIXED);
    if (!default_candidate.usesOrdinaryParallelTraces() || mixed_owner.usesOrdinaryParallelTraces() ||
        default_candidate.deviceWorkspaceBytes() != candidate.deviceWorkspaceBytes())
      throw std::runtime_error("Default or mixed-owner trace admission differs");
    if (planOrdinaryRealFullTraces(plan, 100000).total_device_bytes)
      throw std::runtime_error("Automatic trace preference did not fall back for oversized metadata");
    setenv(key, "1", 1);
    bool budget_rejected = false;
    try { (void)planOrdinaryRealFullTraces(plan, 100000); }
    catch (const std::length_error&) { budget_rejected = true; }
    if (!budget_rejected) throw std::runtime_error("Explicit trace budget violation was not rejected");
    bool mixed_rejected = false;
    try
    {
      PreparedDeviceInference explicit_mixed(image, 2, queue, blas, SpatialJetMode::FULL_VGL,
          false, false, configuredInferenceStatusPolicy(), {}, OrdinaryTracePreparation::EXCLUDE_FOR_MIXED);
    }
    catch (const std::invalid_argument&) { mixed_rejected = true; }
    if (!mixed_rejected) throw std::runtime_error("Explicit incompatible trace/mixed selection accepted");
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
      default_candidate.evaluate(coordinates);
      const auto read = [](PreparedDeviceInference& graph) {
        mixed_periodic_validation::Output out;
        const auto& layout = graph.resultLayout();
        out.phases.resize(layout.configuration_count);
        out.jets.resize(spatialJetSpanElements(layout));
        out.ratios.resize(layout.configuration_count * layout.laplacian_lanes);
        graph.downloadResults(out.phases, out.jets, out.ratios);
        return out;
      };
      const auto a = read(reference), b = read(candidate), automatic = read(default_candidate);
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
      check(b.phases, automatic.phases); check(b.jets, automatic.jets); check(b.ratios, automatic.ratios);
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
    reference.rebindModel(successor); candidate.rebindModel(successor); default_candidate.rebindModel(successor);
    compare(2);
    auto bad = positions;
    bad[0] = std::numeric_limits<double>::infinity();
    for (auto* graph : {&reference, &candidate, &default_candidate})
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
    if (boundary == BoundaryCondition::OPEN)
    {
      unsetenv(key);
      {
        PreparedDeviceInference score_owner(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
        PreparedDeviceInference spatial_owner(image, 1, queue, blas, SpatialJetMode::FULL_VGL, false, true);
        TargetVglPreparation target{1, 1024 * 1024 * 1024, 1000000000000ULL, 0, false};
        PreparedDeviceInference target_owner(image, 1, queue, blas, SpatialJetMode::FULL_VGL,
            false, false, configuredInferenceStatusPolicy(), target);
        if (score_owner.usesOrdinaryParallelTraces() || spatial_owner.usesOrdinaryParallelTraces() ||
            target_owner.usesOrdinaryParallelTraces())
          throw std::runtime_error("Automatic ordinary traces entered an excluded retained/target owner");
      }
      auto mixed_cache = std::make_shared<PreparedDeviceMixedDenseParameters>(successor, queue);
      auto mixed_adapter = std::make_shared<PreparedDeviceMixedValueDense>(mixed_cache,
          default_candidate.mixedDenseRowCapacity(), queue, blas);
      const auto before_mixed = default_candidate.deviceWorkspaceBytes();
      default_candidate.setMixedDense(mixed_adapter);
      if (default_candidate.usesOrdinaryParallelTraces() ||
          before_mixed - default_candidate.deviceWorkspaceBytes() != extra)
        throw std::runtime_error("Automatic mixed attachment retained optional real trace scratch");
      default_candidate.setMixedDense({});
      bool explicit_attachment_rejected = false;
      try { candidate.setMixedDense(mixed_adapter); }
      catch (const std::invalid_argument&) { explicit_attachment_rejected = true; }
      if (!explicit_attachment_rejected || !candidate.usesOrdinaryParallelTraces())
        throw std::runtime_error("Explicit trace/mixed attachment diagnostic changed");
      compare(1); // Detaching mixed must retain valid legacy FP64 execution.
      std::cout << "ORDINARY_PARALLEL_TRACE_EXCLUSIONS_PASS retained_target_mixed=yes\n";
      // Reserve logical capacity, not physical memory, so the baseline fits
      // exactly and only the default's extra trace scratch is denied.
      const auto policy = configuredDeviceMemoryAdmission();
      auto& metrics = deviceMemoryMetrics(ordinal);
      const auto live = metrics.snapshot().live_bytes;
      const auto baseline_bytes = reference.deviceWorkspaceBytes();
      if (policy.limit_bytes && *policy.limit_bytes > policy.external_reserve_bytes &&
          *policy.limit_bytes - policy.external_reserve_bytes > live + baseline_bytes)
      {
        const auto hold_bytes = *policy.limit_bytes - policy.external_reserve_bytes - live - baseline_bytes;
        auto hold = metrics.reserveAllocation(hold_bytes);
        const auto before_fallback = metrics.snapshot().live_bytes;
        unsetenv(key);
        {
          PreparedDeviceInference fallback(image, 2, queue, blas, SpatialJetMode::FULL_VGL);
          if (fallback.usesOrdinaryParallelTraces() || fallback.deviceWorkspaceBytes() != baseline_bytes)
            throw std::runtime_error("Automatic aggregate trace admission did not retain baseline");
        }
        if (metrics.snapshot().live_bytes != before_fallback)
          throw std::runtime_error("Automatic trace fallback leaked device ownership");
        setenv(key, "1", 1);
        bool explicit_budget_rejected = false;
        try { PreparedDeviceInference denied(image, 2, queue, blas, SpatialJetMode::FULL_VGL); }
        catch (const std::length_error&) { explicit_budget_rejected = true; }
        if (!explicit_budget_rejected || metrics.snapshot().live_bytes != before_fallback)
          throw std::runtime_error("Explicit aggregate trace failure changed ownership or was not rejected");
        std::cout << "ORDINARY_PARALLEL_TRACE_AGGREGATE_FALLBACK_PASS baseline_bytes="
                  << baseline_bytes << " logical_reservation_bytes=" << hold_bytes << std::endl;
      }
      else
        std::cout << "ORDINARY_PARALLEL_TRACE_AGGREGATE_FALLBACK_NOT_RUN reason=no_finite_admission_margin\n";
      unsetenv(key);
    }
    std::cout << "ORDINARY_PARALLEL_TRACES_PASS boundary="
              << (boundary == BoundaryCondition::OPEN ? "OPEN" : "PERIODIC") << " N=" << n << " D=" << determinants
              << " max_abs_error=" << maximum_error << " added_device_bytes=" << extra
              << " tail_rebind_failure_recovery=yes default_equivalence=yes budget_fallback=yes mixed_exclusion=yes warm_allocations=0" << std::endl;
  }
}
} // namespace qmcplusplus::psiformer::device
#endif
