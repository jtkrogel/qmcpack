//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_TARGET_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_TARGET_VALIDATION_H
#include "PsiFormerDeviceComplexSpatial.h"
#include "PsiFormerDeviceTargetCostValidation.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include <cmath>
#include <iomanip>

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
    t.setBatchCount(positions.size() / (3 * t.ne_));
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
          auto* last_inverse = owner.inverse_.data() +
              (t.batch_ - 1) * t.shape_.determinants * t.ne_ * t.ne_;
#ifdef QMC_CUDA2HIP
          checkRuntime(hipMemcpyAsync(last_inverse, &invalid, sizeof(invalid), hipMemcpyHostToDevice,
              t.queue_.getNative()), "late batched target fault injection");
#else
          checkRuntime(cudaMemcpyAsync(last_inverse, &invalid, sizeof(invalid), cudaMemcpyHostToDevice,
              t.queue_.getNative()), "late batched target fault injection");
#endif
          t.queue_.sync();
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

/// Exercise the existing head bank with compact K/B tails and complete target products.
inline void validateBatchedHeadTargetOwner()
{
  constexpr std::size_t n = 64, capacity = 3, k = 7, trace_targets = 3;
  const auto descriptors = makeTargetCostPlan(n, true, true);
  const auto initialized = initializePsiFormerParameters(descriptors.modelShape(), 1729,
      PSIFORMER_COMPLEX_PERIODIC_V1, descriptors.environment());
  const PsiFormerExecutionPlan plan(initialized.model_shape, initialized.layouts, initialized.environment);
  const auto old_cost = estimateComplexSpatialTargetCost(plan, k, capacity, trace_targets);
  const auto cost = estimateComplexSpatialTargetCost(plan, k, capacity, trace_targets, true);
  const auto pointer_bytes = 2 * orbitalProjectionPointerBytes(capacity, 1 + 4 * k,
      plan.modelShape().determinants, 2);
  if (cost.projection_pointer_bytes != pointer_bytes ||
      cost.device_bytes - old_cost.device_bytes != pointer_bytes || cost.host_bytes != old_cost.host_bytes)
    throw std::runtime_error("Target head bank per-domain preflight differs from its exact capacity");

  std::vector<double> nuclei(3 * plan.modelShape().nuclei), positions(capacity * 3 * n);
  for (std::size_t i = 0; i < nuclei.size(); ++i) nuclei[i] = .23 + .071 * i;
  const double irrational[3]{.6180339887498948, .4142135623730951, .7320508075688772};
  for (std::size_t b = 0; b < capacity; ++b)
    for (std::size_t e = 0; e < n; ++e)
      for (std::size_t d = 0; d < 3; ++d)
      {
        const double fraction = (e + 1) * irrational[d] + .13 * (d + 1) + .017 * b;
        positions[b * 3 * n + 3 * e + d] = 8 * (fraction - std::floor(fraction));
      }
  compute::Queue<PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, initialized.values, nuclei);
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "target head owner device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "target head owner device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  const auto work = estimateHeadFreeTargetPrefixCost(plan, k, capacity).charged_scalar_work_units;
  const auto denied = metrics.snapshot();
  for (bool bytes : {false, true})
  {
    bool rejected = false;
    try { (void)PreparedDeviceComplexSpatial::prepare(image, queue, blas,
        {k, cost.device_bytes - bytes, work - !bytes}, capacity, trace_targets, true); }
    catch (const std::length_error&) { rejected = true; }
    if (!rejected || metrics.snapshot().allocation_calls != denied.allocation_calls)
      throw std::runtime_error("Target head cost/work-minus-one allocated numeric storage");
  }
  auto ordinary = PreparedDeviceComplexSpatial::prepare(image, queue, blas,
      {k, old_cost.device_bytes, work}, capacity, trace_targets);
  const auto cold = metrics.snapshot();
  auto batched = PreparedDeviceComplexSpatial::prepare(image, queue, blas,
      {k, cost.device_bytes, work}, capacity, trace_targets, true);
  if (metrics.snapshot().live_bytes - cold.live_bytes != cost.device_bytes)
    throw std::runtime_error("Target head actual owner bytes differ from admission");
  const auto fingerprint = batched->traceStorageFingerprint();
  auto full = PreparedDeviceComplexSpatial::prepare(image, 1, queue, blas,
      estimateComplexSpatialCost(plan, 1, trace_targets).device_bytes, trace_targets);

  double max_head = 0, max_full = 0;
  std::size_t comparisons = 0, nonexact_head = 0;
  const auto scalar = [&](double reference, double actual, double& maximum, bool head) {
    const double error = std::abs(reference - actual);
    maximum = std::max(maximum, error);
    if (!std::isfinite(reference) || !std::isfinite(actual) || error > 2e-11 * (1 + std::abs(reference)))
      throw std::runtime_error("Target head component exceeds existing spatial FP64 contract");
    nonexact_head += head && reference != actual;
    ++comparisons;
  };
  const auto complex = [&](cuDoubleComplex reference, cuDoubleComplex actual, double& maximum, bool head) {
    scalar(cuCreal(reference), cuCreal(actual), maximum, head);
    scalar(cuCimag(reference), cuCimag(actual), maximum, head);
  };
  const auto compare = [&](std::size_t live, bool check_full) {
    const std::vector<double> input(positions.begin(), positions.begin() + live * 3 * n);
    const auto warm = metrics.snapshot();
    ordinary->evaluate(input);
    const auto middle = metrics.snapshot();
    batched->evaluate(input);
    const auto after = metrics.snapshot();
    if (warm.allocation_calls != after.allocation_calls || warm.free_calls != after.free_calls ||
        middle.uploaded_bytes - warm.uploaded_bytes != after.uploaded_bytes - middle.uploaded_bytes ||
        middle.downloaded_bytes - warm.downloaded_bytes != after.downloaded_bytes - middle.downloaded_bytes ||
        batched->traceStorageFingerprint() != fingerprint)
      throw std::runtime_error("Target head warm path allocated or added HOST transfers");
    std::vector<ComplexValueResult> a(live), b(live);
    std::vector<cuDoubleComplex> ag(live * 3 * n), bg(ag.size()), al(live * n), bl(al.size()), ar(al.size()), br(al.size());
    ordinary->download(ordinary->resultView(), a, ag, al, ar);
    batched->download(batched->resultView(), b, bg, bl, br);
    for (std::size_t i = 0; i < live; ++i)
    {
      if (a[i].status || b[i].status)
        throw std::runtime_error("Target head comparison requires successful physical owners");
      scalar(a[i].log_abs, b[i].log_abs, max_head, true);
      complex(a[i].phase, b[i].phase, max_head, true);
    }
    for (std::size_t i = 0; i < ag.size(); ++i) complex(ag[i], bg[i], max_head, true);
    for (std::size_t i = 0; i < al.size(); ++i)
    {
      complex(al[i], bl[i], max_head, true);
      complex(ar[i], br[i], max_head, true);
    }
    if (check_full)
    {
      full->evaluate(std::vector<double>(input.begin(), input.begin() + 3 * n));
      std::vector<ComplexValueResult> f(1);
      std::vector<cuDoubleComplex> fg(3 * n), fl(n), fr(n);
      full->download(full->resultView(), f, fg, fl, fr);
      if (f[0].status) throw std::runtime_error("Target head FULL reference failed");
      scalar(f[0].log_abs, b[0].log_abs, max_full, false);
      complex(f[0].phase, b[0].phase, max_full, false);
      for (std::size_t i = 0; i < fg.size(); ++i) complex(fg[i], bg[i], max_full, false);
      for (std::size_t i = 0; i < n; ++i)
      {
        complex(fl[i], bl[i], max_full, false);
        complex(fr[i], br[i], max_full, false);
      }
    }
  };
  for (std::size_t live : {3u, 1u, 2u}) compare(live, live == 1);
  auto parameters = initialized.values;
  for (const auto& tensor : plan.parameterTensors())
    if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT) parameters[tensor.begin] += .0001;
  auto successor = image->makeUpdatedImage(parameters, 1);
  ordinary->rebindModel(successor);
  batched->rebindModel(successor);
  full->rebindModel(successor);
  compare(3, true);

  const auto stale = batched->resultView();
  ComplexSpatialTargetValidationAccess::failLate(*batched, positions);
  std::vector<ComplexValueResult> amplitude(capacity);
  for (auto& value : amplitude) value.status = 777;
  const auto marker = make_cuDoubleComplex(13, -9);
  std::vector<cuDoubleComplex> gradient(capacity * 3 * n, marker), lap_log(capacity * n, marker), lap_ratio(lap_log);
  bool stale_rejected = false;
  try { batched->download(stale, amplitude, gradient, lap_log, lap_ratio); }
  catch (const std::invalid_argument&) { stale_rejected = true; }
  const auto unchanged = [&](const auto& values) {
    return std::all_of(values.begin(), values.end(), [&](const auto value) {
      return cuCreal(value) == 13 && cuCimag(value) == -9;
    });
  };
  if (!stale_rejected || !std::all_of(amplitude.begin(), amplitude.end(), [](const auto& value) { return value.status == 777; }) ||
      !unchanged(gradient) || !unchanged(lap_log) || !unchanged(lap_ratio))
    throw std::runtime_error("Target head late failure published stale or partial products");
  compare(1, false);
  std::cout << std::setprecision(17) << "PASS target batched head Ne64 B3 K7 T3 comparisons=" << comparisons
            << " max_head=" << max_head << " nonexact_head_fields=" << nonexact_head << " max_FULL=" << max_full
            << " pointer_device_bytes=" << pointer_bytes << " pointer_host_bytes=0 cost_minus_one_noalloc=1"
            << " liveB=3,1,2 K_tail=1 active_head_rebind=1 warm_allocations=0 added_HOST_transfers=0 late_rollback=1\n";
}
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
void validateComplexSpatialTargets(std::size_t concurrent_targets = 0)
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
    const auto cost = estimateComplexSpatialTargetCost(plan, k, 1, concurrent_targets);
    const auto work = estimateHeadFreeTargetPrefixCost(plan, k).charged_scalar_work_units;
    const auto before = metrics.snapshot();
    for (bool bytes : {false, true})
    {
      bool rejected = false;
      try { (void)PreparedDeviceComplexSpatial::prepare(image, queue, blas,
          {k, cost.device_bytes - bytes, work - !bytes}, 1, concurrent_targets); }
      catch (const std::length_error&) { rejected = true; }
      if (!rejected || metrics.snapshot().allocation_calls != before.allocation_calls)
        throw std::runtime_error("Target spatial byte/work rejection allocated");
    }
    auto gpu = PreparedDeviceComplexSpatial::prepare(image, queue, blas,
        {k, cost.device_bytes, work}, 1, concurrent_targets);
    const auto trace_fingerprint = gpu->traceStorageFingerprint();
    if (gpu->concurrentTraceTargets() != concurrent_targets)
      throw std::runtime_error("Target spatial lost its frozen trace profile");
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
    if (metrics.snapshot().allocation_calls != warm.allocation_calls || metrics.snapshot().free_calls != warm.free_calls ||
        gpu->traceStorageFingerprint() != trace_fingerprint)
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
      if (metrics.snapshot().allocation_calls != allocations || gpu->traceStorageFingerprint() != trace_fingerprint ||
          gpu->concurrentTraceTargets() != concurrent_targets)
        throw std::runtime_error("Target spatial rebind allocated or changed its immutable trace profile");
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
    std::cout << "complex_spatial_target K=" << k << " trace_targets=" << concurrent_targets
              << " predicted_device_bytes=" << cost.device_bytes
              << " actual_device_bytes=" << cost.device_bytes << " host_numeric_bytes=" << cost.host_bytes
              << " retained_tape_bytes=0 response_P_bytes=0 warm_device_allocations=0 scope=device_wide\n";
  }
  std::cout << "PASS complex target spatial checks=" << checks << " max_G=" << max_g << " max_L=" << max_l
            << " max_kinetic=" << max_kinetic << " max_legacy_FULL=" << max_full << " late_tail_rollback=1\n";

  // Distinct configurations and independent B/K tails must never share final root strides.
  constexpr std::size_t capacity = 3, k = 3;
  const auto cost = estimateComplexSpatialTargetCost(plan, k, capacity, concurrent_targets);
  const auto work = estimateHeadFreeTargetPrefixCost(plan, k, capacity).charged_scalar_work_units;
  const auto before = metrics.snapshot();
  for (bool bytes : {false, true})
  {
    bool rejected = false;
    try { (void)PreparedDeviceComplexSpatial::prepare(image, queue, blas,
        {k, cost.device_bytes - bytes, work - !bytes}, capacity, concurrent_targets); }
    catch (const std::length_error&) { rejected = true; }
    if (!rejected || metrics.snapshot().allocation_calls != before.allocation_calls)
      throw std::runtime_error("Batched target byte/work rejection allocated");
  }
  auto batched = PreparedDeviceComplexSpatial::prepare(image, queue, blas,
      {k, cost.device_bytes, work}, capacity, concurrent_targets);
  const auto batch_trace_fingerprint = batched->traceStorageFingerprint();
  if (metrics.snapshot().live_bytes - before.live_bytes != cost.device_bytes)
    throw std::runtime_error("Batched target actual bytes differ from preflight");
  auto full_batch = PreparedDeviceComplexSpatial::prepare(image, capacity, queue, blas,
      estimateComplexSpatialCost(plan, capacity, concurrent_targets).device_bytes, concurrent_targets);
  const std::vector<double> packed(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 36);
  double max_batched = 0;
  const auto compare_batch = [&](std::size_t live) {
    const std::vector<double> positions(packed.begin(), packed.begin() + 12 * live);
    amplitude.resize(live); full_amplitude.resize(live);
    g.resize(12 * live); fg.resize(12 * live);
    l.resize(4 * live); r.resize(4 * live); fl.resize(4 * live); fr.resize(4 * live);
    batched->evaluate(positions);
    batched->download(batched->resultView(), amplitude, g, l, r);
    full_batch->evaluate(positions);
    full_batch->download(full_batch->resultView(), full_amplitude, fg, fl, fr);
    for (std::size_t b = 0; b < live; ++b)
    {
      workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data() + 12 * b, 4));
      const auto expected = oracle.evaluateFull(*workspace);
      near(amplitude[b].log_abs, expected.amplitude.log_abs, "batched target log", 2e-11);
      near(std::abs(host(amplitude[b].phase) - expected.amplitude.phase), 0, "batched target phase", 2e-11);
      near(amplitude[b].log_abs, full_amplitude[b].log_abs, "batched target FULL log", 2e-11);
      for (std::size_t e = 0; e < 4; ++e)
      {
        for (std::size_t mu = 0; mu < 3; ++mu)
        {
          const auto index = 12 * b + 3 * e + mu;
          const auto error = std::abs(host(g[index]) - expected.gradient[3 * e + mu]);
          max_batched = std::max(max_batched, error);
          near(error, 0, "batched target CPU G", 2e-11);
          near(std::abs(host(g[index]) - host(fg[index])), 0, "batched target FULL G", 2e-11);
        }
        const auto index = 4 * b + e;
        for (const auto error : {std::abs(host(l[index]) - expected.laplacian_log[e]),
                                std::abs(host(r[index]) - expected.laplacian_ratio[e]),
                                std::abs(host(l[index]) - host(fl[index])),
                                std::abs(host(r[index]) - host(fr[index]))})
        {
          max_batched = std::max(max_batched, error);
          near(error, 0, "batched target CPU/FULL L", 2e-11);
        }
      }
    }
  };
  compare_batch(3);
  bool extent_rejected = false;
  try { batched->evaluate(std::vector<double>(48, 0)); }
  catch (const std::invalid_argument&) { extent_rejected = true; }
  if (!extent_rejected) throw std::runtime_error("Batched target admitted oversized live batch");
  compare_batch(3);
  const auto old_view = batched->resultView();
  const auto warm_batch = metrics.snapshot();
  for (const auto live : {std::size_t(1), std::size_t(2), std::size_t(3)}) compare_batch(live);
  if (metrics.snapshot().allocation_calls != warm_batch.allocation_calls ||
      metrics.snapshot().free_calls != warm_batch.free_calls ||
      batched->traceStorageFingerprint() != batch_trace_fingerprint)
    throw std::runtime_error("Batched target warm allocation/free");
  bool stale_rejected = false;
  try { batched->download(old_view, amplitude, g, l, r); }
  catch (const std::invalid_argument&) { stale_rejected = true; }
  if (!stale_rejected) throw std::runtime_error("Batched target admitted stale view");
  ComplexSpatialTargetValidationAccess::failLate(*batched, packed);
  bool failure_rejected = false;
  try { (void)batched->resultView(); }
  catch (const std::logic_error&) { failure_rejected = true; }
  if (!failure_rejected) throw std::runtime_error("Batched target published late nonzero-walker failure");
  compare_batch(3);
  auto successor = image->makeUpdatedImage(parameters, ++version);
  batched->rebindModel(successor); full_batch->rebindModel(successor);
  if (batched->traceStorageFingerprint() != batch_trace_fingerprint ||
      batched->concurrentTraceTargets() != concurrent_targets)
    throw std::runtime_error("Batched target successor changed its frozen trace profile");
  compare_batch(2);
  std::cout << "PASS complex batched target B3 live3/1/2/3 K3-tail CPU/FULL max_error=" << max_batched
            << " trace_targets=" << concurrent_targets
            << " warm_allocations=0 late_nonzero_walker_failure=1 recovery=1 rebind=1\n";
}
#endif
