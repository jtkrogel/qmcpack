//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_TARGET_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_TARGET_VALIDATION_H
#include "PsiFormerDeviceComplexSpatial.h"
#include "PsiFormerDeviceTargetCostValidation.h"

namespace qmcplusplus::psiformer::device
{
/// Test-only existing target phase access, not a production fault API.
struct ComplexSpatialTargetValidationAccess
{
  /// Inspect omissions and inject failure on the actual last target after earlier success.
  static void failLate(PreparedDeviceComplexSpatial& owner, const std::vector<double>& positions)
  {
    auto& t = *owner.trunk_;
    if (!t.value_tape_.empty() || t.orbital_jets_.capacity() || t.factors_.capacity() || t.full_output_.capacity())
      throw std::runtime_error("Forward paired target graph allocated tapes or real-head/full fallback");
    owner.invalidate();
    t.beginCoordinateEvaluation();
    t.positions_.upload(positions.data(), positions.size());
    std::size_t completed = 0;
    bool injected = false, rejected = false;
    try
    {
      for (std::size_t begin = 0; begin < t.ne_;)
      {
        const auto count = std::min(t.target_preparation_.max_targets, t.ne_ - begin);
        owner.prepareTarget(begin, count);
        if (begin + count == t.ne_)
        {
          const auto invalid = make_cuDoubleComplex(std::numeric_limits<double>::quiet_NaN(), 0);
          owner.inverse_.upload(&invalid, 1);
          injected = true;
        }
        owner.finishTarget(begin);
        ++completed;
        begin += count;
      }
    }
    catch (const std::runtime_error&) { rejected = true; }
    if (!injected || !rejected || !completed)
      throw std::runtime_error("Forward paired target late physical-tail failure not exercised");
  }
};
} // namespace qmcplusplus::psiformer::device

/// Reuse metadata-only canonical descriptors; no numeric P/model/driver allocation.
void validateComplexSpatialTargetCosts()
{
  using namespace qmcplusplus::psiformer::device;
  const auto before = deviceMemoryMetrics(0).snapshot();
  const auto work_budget = std::numeric_limits<std::size_t>::max();
  for (std::size_t n : {std::size_t(100), std::size_t(200), std::size_t(300)})
  {
    const auto plan = makeTargetCostPlan(n, true, true);
    const auto full = estimateComplexSpatialCost(plan, 1);
    std::size_t previous = 0;
    for (std::size_t k : {std::size_t(1), std::size_t(3), std::size_t(32), n})
    {
      const auto cost = estimateComplexSpatialTargetCost(plan, k);
      const auto prefix = estimateHeadFreeTargetPrefixCost(plan, k);
      if (cost.device_bytes <= previous || (k == n && (cost.device_bytes != full.device_bytes || cost.host_bytes != full.host_bytes + 48)))
        throw std::runtime_error("Paired target metadata costs are not monotone/exact FULL compatible");
      previous = cost.device_bytes;
      for (std::size_t gib : {std::size_t(40), std::size_t(80)})
      {
        const auto budget = gib * (std::size_t(1) << 30);
        std::cout << "complex_target_cost N=" << n << " ions=" << n / 5 << " D=16 F=256 H=4 L=4 K=" << k
                  << " owner_device_bytes=" << cost.device_bytes << " host_numeric_bytes=" << cost.host_bytes
                  << " response_P_bytes=0 retained_tape_bytes=0 budget_GiB=" << gib
                  << " memory_admit=" << (cost.device_bytes <= budget) << " matrix_work_sites=" << prefix.charged_scalar_work_units
                  << " work_budget=" << work_budget << " mode=memory_only omitted=image,vendor,crowds,solver,object_metadata\n";
      }
    }
  }
  if (deviceMemoryMetrics(0).snapshot().allocation_calls != before.allocation_calls)
    throw std::runtime_error("Metadata-only complex target cost check allocated device memory");
  std::cout << "PASS complex target metadata-only costs: numeric_P_model_graph_allocations=0 runtime_claim=none\n";
}

/// Target forward-only oracle/legacy parity, physical outputs, generations and budgets.
void validateComplexSpatialTargets()
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  using C = std::complex<double>;
  const auto host = [](cuDoubleComplex z) { return C(cuCreal(z), cuCimag(z)); };
  ExecutionEnvironment environment;
  environment.boundary = BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
  environment.lattice_vectors = {{{4.0, 0.4, -0.2}, {0.7, 5.0, 0.3}, {-0.1, 0.8, 6.0}}};
  environment.periodic_axes = {true, true, true};
  environment.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
  environment.amplitude_scalar_domain = ScalarDomain::COMPLEX;
  pf::PsiFormer model(smallParameters(environment), configurations(environment), 4, 16, 2, 2);
  const auto plan = PsiFormerExecutionPlan::fromParameters(model.p, ModelShape{2, 2, 2, 4, 16, 2, 2}, environment);
  auto parameters = model.p.flat_values();
  for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
  {
    const auto& interval = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
    for (std::size_t p = interval.begin; p < interval.end; ++p) parameters[p] = 0.09 * std::sin(double(p - interval.begin + 1));
  }
  model.p.set_flat_values(parameters);
  pf::DirectComplexSpatialExecutor oracle(model, plan);
  auto workspace = oracle.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, parameters, model.cfg.nuclei.x);
  auto full = PreparedDeviceComplexSpatial::prepare(image, 1, queue, blas, estimateComplexSpatialCost(plan, 1).device_bytes);
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "target spatial metrics");
#else
  checkRuntime(cudaGetDevice(&ordinal), "target spatial metrics");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  for (std::size_t invalid : {std::size_t(0), std::size_t(5)})
  {
    const auto allocations = metrics.snapshot().allocation_calls;
    bool rejected = false;
    try { (void)PreparedDeviceComplexSpatial::prepare(image, queue, blas, {invalid, 1, 1}); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected || metrics.snapshot().allocation_calls != allocations)
      throw std::runtime_error("Target spatial invalid K admission allocated");
  }
  std::vector<ComplexValueResult> amplitude(1), full_amplitude(1);
  std::vector<cuDoubleComplex> g(12), l(4), r(4), fg(12), fl(4), fr(4);
  const std::vector<double> base(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 12);
  const double inverse_mass[4]{1, 1 / 1.3, 1 / 0.8, 1 / 1.7}, kappa[3]{0.13, -0.07, 0.04};
  double max_g = 0, max_l = 0, max_kinetic = 0, max_full = 0;
  std::size_t checks = 0, version = 0;
  for (std::size_t k : {std::size_t(1), std::size_t(3), std::size_t(4)})
  {
    const auto cost = estimateComplexSpatialTargetCost(plan, k);
    const auto work = estimateHeadFreeTargetPrefixCost(plan, k).charged_scalar_work_units;
    const auto before = metrics.snapshot();
    for (bool bytes : {false, true})
    {
      bool rejected = false;
      try { (void)PreparedDeviceComplexSpatial::prepare(image, queue, blas, {k, cost.device_bytes - bytes, work - !bytes}); }
      catch (const std::length_error&) { rejected = true; }
      if (!rejected || metrics.snapshot().allocation_calls != before.allocation_calls)
        throw std::runtime_error("Target spatial byte/work rejection allocated");
    }
    auto gpu = PreparedDeviceComplexSpatial::prepare(image, queue, blas, {k, cost.device_bytes, work});
    if (metrics.snapshot().live_bytes - before.live_bytes != cost.device_bytes)
      throw std::runtime_error("Target spatial actual bytes differ from exact preflight");
    const auto compare = [&](const std::vector<double>& positions) {
      gpu->evaluate(positions);
      gpu->download(gpu->resultView(), amplitude, g, l, r);
      full->evaluate(positions);
      full->download(full->resultView(), full_amplitude, fg, fl, fr);
      workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data(), 4));
      const auto expected = oracle.evaluateFull(*workspace);
      near(amplitude[0].log_abs, expected.amplitude.log_abs, "target spatial amplitude", 2e-11);
      near(std::abs(host(amplitude[0].phase) - expected.amplitude.phase), 0, "target spatial phase", 2e-11);
      C kinetic{}, reference{};
      for (std::size_t e = 0; e < 4; ++e)
      {
        C square{}, expected_square{};
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
          const auto lane = 3 * e + axis;
          max_g = std::max(max_g, std::abs(host(g[lane]) - expected.gradient[lane]));
          max_full = std::max(max_full, std::abs(host(g[lane]) - host(fg[lane])));
          near(std::abs(host(g[lane]) - expected.gradient[lane]), 0, "target G", 2e-11);
          const auto total = host(g[lane]) + C(0, kappa[axis]), ex = expected.gradient[lane] + C(0, kappa[axis]);
          square += total * total;
          expected_square += ex * ex;
        }
        max_l = std::max({max_l, std::abs(host(l[e]) - expected.laplacian_log[e]), std::abs(host(r[e]) - expected.laplacian_ratio[e])});
        max_full = std::max({max_full, std::abs(host(l[e]) - host(fl[e])), std::abs(host(r[e]) - host(fr[e]))});
        near(std::abs(host(l[e]) - expected.laplacian_log[e]), 0, "target Llog", 2e-11);
        near(std::abs(host(r[e]) - expected.laplacian_ratio[e]), 0, "target Lratio", 2e-11);
        kinetic -= 0.5 * inverse_mass[e] * (host(l[e]) + square);
        reference -= 0.5 * inverse_mass[e] * (expected.laplacian_log[e] + expected_square);
      }
      max_kinetic = std::max(max_kinetic, std::abs(kinetic - reference));
      near(std::abs(kinetic - reference), 0, "target physical kinetic consumer", 2e-11);
      near(max_full, 0, "target legacy FULL", 2e-11);
      ++checks;
    };
    compare(base);
    const auto stale = gpu->resultView();
    const auto warm = metrics.snapshot();
    for (std::size_t sample = 0; sample < 3; ++sample)
      compare(std::vector<double>(model.cfg.electrons.x.begin() + 12 * sample, model.cfg.electrons.x.begin() + 12 * (sample + 1)));
    if (metrics.snapshot().allocation_calls != warm.allocation_calls || metrics.snapshot().free_calls != warm.free_calls)
      throw std::runtime_error("Target spatial warmed DEVICE allocation/free");
    bool rejected = false;
    try { gpu->download(stale, amplitude, g, l, r); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Target spatial admitted stale generation");
    if (k != 4)
    {
      ComplexSpatialTargetValidationAccess::failLate(*gpu, base);
      rejected = false;
      try { (void)gpu->resultView(); }
      catch (const std::logic_error&) { rejected = true; }
      if (!rejected) throw std::runtime_error("Target spatial published late-failed output");
      compare(base);
    }
    const auto replace = [&](const std::vector<double>& values) {
      model.p.set_flat_values(values);
      image = image->makeUpdatedImage(values, ++version);
      const auto allocations = metrics.snapshot().allocation_calls;
      gpu->rebindModel(image);
      full->rebindModel(image);
      if (metrics.snapshot().allocation_calls != allocations) throw std::runtime_error("Target spatial rebind allocated");
    };
    auto zero = parameters;
    for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
    {
      const auto& interval = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
      std::fill(zero.begin() + interval.begin, zero.begin() + interval.end, 0);
    }
    const auto old_image = gpu->resultView();
    replace(zero);
    rejected = false;
    try { gpu->download(old_image, amplitude, g, l, r); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Target spatial admitted old-image view");
    compare(base);
    auto singular = zero;
    const auto& real = plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT);
    std::fill(singular.begin() + real.begin, singular.begin() + real.end, 0);
    replace(singular);
    rejected = false;
    try { gpu->evaluate(base); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Target spatial admitted singular canonical inverse");
    rejected = false;
    try { (void)gpu->resultView(); }
    catch (const std::logic_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Target spatial published singular-request output");
    replace(parameters);
    compare(base);
    std::cout << "complex_spatial_target K=" << k << " predicted_device_bytes=" << cost.device_bytes
              << " actual_device_bytes=" << cost.device_bytes << " host_numeric_bytes=" << cost.host_bytes
              << " retained_tape_bytes=0 response_P_bytes=0 warm_device_allocations=0 scope=device_wide\n";
  }
  std::cout << "PASS complex target spatial checks=" << checks << " max_G=" << max_g << " max_L=" << max_l
            << " max_kinetic=" << max_kinetic << " max_legacy_FULL=" << max_full << " late_tail_rollback=1\n";
}
#endif
