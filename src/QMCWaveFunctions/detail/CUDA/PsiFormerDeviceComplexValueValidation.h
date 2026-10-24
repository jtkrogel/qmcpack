//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_VALUE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_VALUE_VALIDATION_H
#include "PsiFormerDeviceComplexValue.h"
#include "PsiFormerDeviceComplexScore.h"
#include "PsiFormerDeviceTargetCostValidation.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexValueExecutor.h"
#include <cstring>

/// Exercise the actual eligible owner, exact admission, successor and retained-role inventory.
void validateComplexValuePairOwner()
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  const auto descriptors = makeTargetCostPlan(64, true, true);
  const auto& environment = descriptors.environment();
  const auto initialized = initializePsiFormerParameters(descriptors.modelShape(), 1729,
      PSIFORMER_COMPLEX_PERIODIC_V1, environment);
  const PsiFormerExecutionPlan plan(descriptors.modelShape(), initialized.layouts, environment);
  std::vector<double> nuclei(3 * plan.modelShape().nuclei), positions(2 * 3 * 64);
  for (std::size_t i = 0; i < nuclei.size(); ++i) nuclei[i] = 0.23 + 0.071 * i;
  for (std::size_t i = 0; i < positions.size(); ++i)
    positions[i] = 2.0 + 1.7 * std::sin(0.37 * (i + 1));
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, initialized.values, nuclei);
  const auto cost = estimateComplexValueCost(plan, 2);
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "pair owner device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "pair owner device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  const auto before = metrics.snapshot();
  bool denied = false;
  try { (void)PreparedDeviceComplexValue::prepare(image, 2, queue, blas, cost.device_bytes - 1); }
  catch (const std::length_error&) { denied = true; }
  if (!denied || metrics.snapshot().allocation_calls != before.allocation_calls)
    throw std::runtime_error("Eligible pair owner cost-minus-one allocated storage");
  auto owner = PreparedDeviceComplexValue::prepare(image, 2, queue, blas, cost.device_bytes);
  if (metrics.snapshot().live_bytes - before.live_bytes != cost.device_bytes)
    throw std::runtime_error("Eligible pair owner differs from exact device cost");
  const auto warm = metrics.snapshot();
  std::vector<ComplexValueResult> results(2);
  for (std::size_t live : {2u, 1u, 2u})
  {
    owner->evaluate(std::vector<double>(positions.begin(), positions.begin() + 3 * 64 * live));
    results.resize(live);
    owner->download(owner->resultView(), results);
  }
  if (warm.allocation_calls != metrics.snapshot().allocation_calls || warm.free_calls != metrics.snapshot().free_calls)
    throw std::runtime_error("Eligible pair owner warmed evaluation allocated device storage");
  const auto stale = owner->resultView();
  auto successor = image->makeUpdatedImage(initialized.values, 1);
  const auto rebind = metrics.snapshot();
  owner->rebindModel(successor);
  if (rebind.allocation_calls != metrics.snapshot().allocation_calls)
    throw std::runtime_error("Eligible pair owner successor allocated device storage");
  bool revoked = false;
  try { owner->download(stale, results); }
  catch (const std::invalid_argument&) { revoked = true; }
  if (!revoked) throw std::runtime_error("Eligible pair owner retained an old-image result");
  owner->evaluate(positions);
  // The retained score constructor checks its actual full inventory; no pair buffer is charged or owned.
  const auto score_cost = estimateComplexScoreCost(plan);
  const auto score_before = metrics.snapshot();
  auto score = PreparedDeviceComplexScore::prepare(successor, queue, blas, score_cost.device_bytes);
  if (score_cost.forward.pair_event_bytes || metrics.snapshot().live_bytes - score_before.live_bytes != score_cost.device_bytes)
    throw std::runtime_error("Eligible retained score inherited ordinary pair ownership");
  std::cout << "PASS eligible complex VALUE pair owner: Ne=64 capacity=2 event_device_owner_bytes="
            << cost.pair_event_bytes << " exact_cost=1 cost_minus_one_denied=1 warm_allocations=0 rebind=1 retained_score_excluded=1\n";
}

/// Compare complete old/new reduction bits and independently enumerate event priority.
void validateComplexValuePairEvents()
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  const auto cell = periodic_geometry::makeCell({{{8., 0., 0.}, {0., 10., 0.}, {0., 0., 12.}}}, {true, true, true});
  std::size_t comparisons = 0;
  for (std::size_t n : {63u, 64u, 65u, 100u, 200u})
  {
    const auto plan = makeTargetCostPlan(n, true, true);
    const auto old_cost = estimateComplexValueCost(plan, 3, false);
    const auto cost = estimateComplexValueCost(plan, 3);
    const auto bytes = configuredComplexValuePairCheck() && n >= 64 ? 3 * sizeof(std::uint64_t) : 0;
    if (old_cost.pair_event_bytes || cost.pair_event_bytes != bytes ||
        cost.device_bytes != old_cost.device_bytes + bytes || cost.host_bytes != old_cost.host_bytes ||
        complex_detail::baseCost(plan, 3, SpatialJetMode::FULL_VGL).pair_event_bytes)
      throw std::runtime_error("Ordinary pair record cost leaked into retained/derivative role or wrong domain");
  }
  for (std::size_t n : {63u, 64u, 65u, 100u, 200u})
  {
    constexpr std::size_t capacity = 3, determinants = 2;
    const auto spin = n / 2;
    DeviceBuffer<double> positions(3 * capacity * n), cusp(capacity);
    DeviceBuffer<ComplexValueChannel> channels(capacity * determinants);
    DeviceBuffer<ComplexValueResult> old_result(capacity), new_result(capacity);
    DeviceBuffer<std::uint64_t> events(capacity), old_status(1), new_status(1);
    const auto before = deviceMemoryMetrics(positions.deviceOrdinal()).snapshot();
    for (std::size_t live : {3u, 1u, 2u})
      for (std::size_t variant = 0; variant != 12; ++variant)
      {
        std::vector<double> r(3 * capacity * n), c(capacity, 0.125);
        std::vector<ComplexValueChannel> ch(capacity * determinants, {make_cuDoubleComplex(1, 0), 0, 0});
        std::vector<std::uint64_t> expected(capacity, UINT64_MAX), actual(capacity, UINT64_MAX);
        for (std::size_t b = 0; b < capacity; ++b)
        {
          auto* p = r.data() + 3 * b * n;
          for (std::size_t i = 0; i < n; ++i)
            for (std::size_t d = 0; d < 3; ++d)
              p[3 * i + d] = 0.1 * b + (0.011 + 0.006 * d) * (i + 1);
          if (variant == 1 || variant == 2 || variant == 4 || variant == 8 || variant == 9)
            std::copy_n(p, 3, p + 3);
          if (variant == 2)
          {
            p[0] = p[3] = 0;
            p[3] += 8;
          }
          if (variant == 3) std::copy_n(p, 3, p + 3 * spin);
          if (variant == 4 || variant == 6 || variant == 7 || variant == 11)
            p[3 * (n - 1)] = std::numeric_limits<double>::quiet_NaN();
          if (variant == 5)
          {
            p[0] = std::numeric_limits<double>::quiet_NaN();
            std::copy_n(p + 3 * (n - 2), 3, p + 3 * (n - 1));
          }
          if (variant == 6)
            ch[2 * b].status = ch[2 * b + 1].status = 1;
          if (variant == 7) ch[2 * b + 1].phase = make_cuDoubleComplex(-1, 0);
          if (variant == 8) ch[2 * b].phase = make_cuDoubleComplex(std::numeric_limits<double>::quiet_NaN(), 0);
          if (variant == 9 || variant == 10) c[b] = std::numeric_limits<double>::quiet_NaN();
          for (std::size_t i = 0; i < n && expected[b] == UINT64_MAX; ++i)
            for (std::size_t j = i + 1; j < n; ++j)
              if ((i < spin) == (j < spin))
              {
                periodic_geometry::Pair pair;
                if (!periodic_geometry::evaluatePair(cell, p + 3 * i, p + 3 * j, pair))
                { expected[b] = 2 * (i * n + j) + 1; break; }
                if (pair.radius == 0)
                { expected[b] = 2 * (i * n + j); break; }
              }
        }
        positions.upload(r.data(), r.size()); cusp.upload(c.data(), c.size()); channels.upload(ch.data(), ch.size());
        if (((variant == 1 || variant == 2 || variant == 4 || variant == 8 || variant == 9) && expected[0] != 2) ||
            (variant == 3 && expected[0] != UINT64_MAX) || (variant == 5 && expected[0] != 3))
          throw std::runtime_error("Pair-event fixture lost exact periodic node or event-order distinction");
        const std::uint64_t clear = UINT64_MAX;
        old_status.upload(&clear, 1); new_status.upload(&clear, 1); events.upload(actual.data(), actual.size());
        checkRuntime(launchComplexValuePairCheck(queue.getNative(), live, n, spin, cell, positions.data(), events.data()),
                     "pair-event validation precheck");
        queue.sync();
        events.download(actual.data(), capacity);
        std::uint64_t old_failure, new_failure;
        new_status.download(&new_failure, 1);
        if (new_failure != clear) throw std::runtime_error("Pair precheck changed numerical status early");
        for (std::size_t b = 0; b < capacity; ++b)
          if (actual[b] != (b < live ? expected[b] : clear))
            throw std::runtime_error("Pair precheck ordinal/type or live tail differs from independent scalar scan");
        checkRuntime(launchComplexValueReduce(queue.getNative(), live, determinants, n, spin, cell, positions.data(),
            channels.data(), cusp.data(), old_result.data(), old_status.data()), "old pair reduction");
        checkRuntime(launchComplexValueReduce(queue.getNative(), live, determinants, n, spin, cell, positions.data(),
            channels.data(), cusp.data(), new_result.data(), new_status.data(), 1, events.data()), "new pair reduction");
        queue.sync();
        std::vector<ComplexValueResult> a(live), b(live);
        old_result.download(a.data(), live); new_result.download(b.data(), live);
        old_status.download(&old_failure, 1); new_status.download(&new_failure, 1);
        if (old_failure != new_failure || std::memcmp(a.data(), b.data(), live * sizeof(ComplexValueResult)))
          throw std::runtime_error("Pair precheck changed complete reduction bits or first failure");
        ++comparisons;
      }
    const auto after = deviceMemoryMetrics(positions.deviceOrdinal()).snapshot();
    if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls)
      throw std::runtime_error("Pair precheck warmed evaluation allocated device storage");
  }
  std::cout << "PASS complex VALUE pair events: comparisons=" << comparisons
            << " complete_result_status_bits=exact independent_event_scan=exact warm_device_allocations=0\n";
}

/// Numerical-only complete VALUE gate; fixture builders are the periodic validator's.
void validateComplexValue()
{
  validateComplexValuePairOwner();
  validateComplexValuePairEvents();
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  ExecutionEnvironment environment;
  environment.boundary = BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
  environment.lattice_vectors = {{{4.0, 0.4, -0.2}, {0.7, 5.0, 0.3}, {-0.1, 0.8, 6.0}}};
  environment.periodic_axes = {true, true, true};
  environment.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
  environment.amplitude_scalar_domain = ScalarDomain::COMPLEX;
  const ModelShape shape{2, 2, 2, 4, 16, 2, 2};
  pf::PsiFormer model(smallParameters(environment), configurations(environment), 4, 16, 2, 2);
  const auto plan = PsiFormerExecutionPlan::fromParameters(model.p, shape, environment);
  auto parameters = model.p.flat_values();
  for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
  {
    const auto& imaginary = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
    for (std::size_t i = imaginary.begin; i < imaginary.end; ++i)
      parameters[i] = 0.09 * std::sin(double(i - imaginary.begin + 1));
  }
  model.p.set_flat_values(parameters);
  pf::DirectComplexValueExecutor oracle(model, plan);
  auto workspace = oracle.makeWorkspace();
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, parameters, model.cfg.nuclei.x);
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "complex VALUE device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "complex VALUE device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  const auto cost = estimateComplexValueCost(plan, 3);
  const auto before = metrics.snapshot();
  bool rejected = false;
  try { (void)PreparedDeviceComplexValue::prepare(image, 3, queue, blas, cost.device_bytes - 1); }
  catch (const std::length_error&) { rejected = true; }
  if (!rejected || metrics.snapshot().allocation_calls != before.allocation_calls)
    throw std::runtime_error("Complex VALUE budget rejection allocated storage");
  rejected = false;
  try { (void)estimateComplexValueCost(plan, std::numeric_limits<std::size_t>::max()); }
  catch (const std::exception&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Complex VALUE overflow extent was admitted");
  auto gpu = PreparedDeviceComplexValue::prepare(image, 3, queue, blas, cost.device_bytes);
  if (metrics.snapshot().live_bytes - before.live_bytes != cost.device_bytes)
    throw std::runtime_error("Complex VALUE exact owner bytes mismatch");
  rejected = false;
  try { PreparedDeviceInference legacy(image, 3, queue, blas); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Legacy real inference admitted a complex image");

  double max_log = 0, max_phase = 0;
  std::size_t nodes = 0, comparisons = 0;
  std::vector<ComplexValueResult> result(3);
  // Every oracle comparison uses the complete CPU complex owner, never CPU fallback.
  const auto compare = [&](const std::vector<double>& positions) {
    gpu->evaluate(positions);
    result.resize(positions.size() / 12);
    gpu->download(gpu->resultView(), result);
    for (std::size_t b = 0; b < result.size(); ++b)
    {
      workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data() + 12 * b, 4));
      const auto& expected = oracle.evaluate(*workspace);
      const auto phase = std::complex<double>(cuCreal(result[b].phase), cuCimag(result[b].phase));
      if (expected.amplitude.isZero())
      {
        std::uint64_t bits;
        std::memcpy(&bits, &result[b].log_abs, sizeof(bits));
        if (phase != std::complex<double>{} || bits != UINT64_C(0xfff0000000000000) || result[b].status != 1)
          throw std::runtime_error("Complex VALUE node contract mismatch");
        ++nodes;
      }
      else
      {
        max_log = std::max(max_log, std::abs(result[b].log_abs - expected.amplitude.log_abs));
        max_phase = std::max(max_phase, std::abs(phase - expected.amplitude.phase));
        near(result[b].log_abs, expected.amplitude.log_abs, "complex log amplitude", 2e-10);
        near(phase.real(), expected.amplitude.phase.real(), "complex phase real", 2e-10);
        near(phase.imag(), expected.amplitude.phase.imag(), "complex phase imaginary", 2e-10);
        near(result[b].cancellation_ratio, expected.cancellation_ratio, "complex cancellation", 2e-10);
        if (result[b].status || result[b].singular_channels != expected.singular_channels)
          throw std::runtime_error("Complex VALUE channel admission mismatch");
      }
      ++comparisons;
    }
  };
  compare(model.cfg.electrons.x);
  const auto initial_phase = std::complex<double>(cuCreal(result[0].phase), cuCimag(result[0].phase));
  if (std::abs(initial_phase.imag()) < 1e-8 ||
      std::abs(initial_phase - std::complex<double>(cuCreal(result[1].phase), cuCimag(result[1].phase))) < 1e-8)
    throw std::runtime_error("Complex VALUE fixture lacks a nonconstant complex phase");
  auto stale = gpu->resultView();
  const auto warm = metrics.snapshot();
  for (std::size_t count : {std::size_t(1), std::size_t(2), std::size_t(3)})
    compare(std::vector<double>(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 12 * count));
  const auto after_warm = metrics.snapshot();
  if (after_warm.allocation_calls != warm.allocation_calls || after_warm.free_calls != warm.free_calls)
    throw std::runtime_error("Complex VALUE warmed evaluation allocated device storage");

  // Each observation replaces ONE row of the same base, including index transitions.
  const std::vector<double> base(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 12);
  std::vector<PreparedDeviceComplexValue::Replacement> replacements;
  for (std::size_t electron : {0u, 2u, 1u, 2u})
    replacements.push_back({electron, {base[3 * electron] + 0.13,
        base[3 * electron + 1] - 0.07, base[3 * electron + 2] + 0.04}});
  const auto streamed_before = metrics.snapshot();
  std::vector<ComplexValueResult> observed(1);
  const auto check_replacement = [&](std::size_t point, const PreparedDeviceComplexValue::ResultView& view) {
    gpu->download(view, observed);
    auto explicit_positions = base;
    std::copy(replacements[point].position.begin(), replacements[point].position.end(),
        explicit_positions.begin() + 3 * replacements[point].electron);
    workspace->setPositions(pf::GeometryPositionView::interleaved(explicit_positions.data(), 4));
    const auto& expected = oracle.evaluate(*workspace);
    near(observed[0].log_abs, expected.amplitude.log_abs, "streamed complex log", 2e-10);
    near(cuCreal(observed[0].phase), expected.amplitude.phase.real(), "streamed complex real", 2e-10);
    near(cuCimag(observed[0].phase), expected.amplitude.phase.imag(), "streamed complex imag", 2e-10);
  };
  const auto at = [&](std::size_t point) { return replacements[point]; };
  const auto run = [](auto&& evaluation) { evaluation(); };
  gpu->evaluateReplacementStream(base, replacements.size(), at, check_replacement, run);
  const auto streamed_after = metrics.snapshot();
  if (streamed_after.uploaded_bytes - streamed_before.uploaded_bytes != 3 * sizeof(double) * (4 + 4 + 3) ||
      streamed_after.allocation_calls != streamed_before.allocation_calls)
    throw std::runtime_error("Complex VALUE replacement stream did not reuse its coordinate owner");
  auto stream_view = gpu->resultView();
  rejected = false;
  try
  {
    gpu->evaluateReplacementStream(base, replacements.size(), at,
        [&](std::size_t point, const PreparedDeviceComplexValue::ResultView& view) {
          check_replacement(point, view);
          if (point + 1 == replacements.size())
            throw std::runtime_error("test late observer failure");
        }, run);
  }
  catch (const std::runtime_error&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Complex VALUE observer failure was swallowed");
  rejected = false;
  try { gpu->download(stream_view, observed); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Complex VALUE failed stream retained an old result");
  rejected = false;
  try { (void)gpu->resultView(); }
  catch (const std::logic_error&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Complex VALUE failed stream published a partial result");
  const auto invalid_before = metrics.snapshot();
  const auto valid_last = replacements.back();
  replacements.back().electron = 4;
  rejected = false;
  try { gpu->evaluateReplacementStream(base, replacements.size(), at, check_replacement, run); }
  catch (const std::invalid_argument&) { rejected = true; }
  replacements.back() = valid_last;
  if (!rejected || metrics.snapshot().uploaded_bytes != invalid_before.uploaded_bytes)
    throw std::runtime_error("Complex VALUE stream did not preflight every replacement");
  gpu->evaluateReplacementStream(base, replacements.size(), at, check_replacement, run);
  compare(model.cfg.electrons.x);
  auto coincident = model.cfg.electrons.x;
  std::copy_n(coincident.data(), 3, coincident.data() + 3);
  compare(coincident);
  if (result[0].status != 1 || result[1].status != 0)
    throw std::runtime_error("Complex VALUE one physical node poisoned the population");
  rejected = false;
  try { gpu->download(stale, result); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Complex VALUE reused an expired generation");

  // Successor images share metadata, not the previous parameter allocation.
  std::size_t version = 0;
  const auto replace = [&](const std::vector<double>& next) {
    const auto old_view = gpu->resultView();
    model.p.set_flat_values(next);
    image = image->makeUpdatedImage(next, ++version);
    const auto before_rebind = metrics.snapshot();
    gpu->rebindModel(image);
    if (metrics.snapshot().allocation_calls != before_rebind.allocation_calls)
      throw std::runtime_error("Complex VALUE successor rebind allocated device storage");
    bool stale_rejected = false;
    try { gpu->download(old_view, result); }
    catch (const std::invalid_argument&) { stale_rejected = true; }
    if (!stale_rejected)
      throw std::runtime_error("Complex VALUE successor exposed an earlier image result");
  };
  auto zero_imaginary = parameters;
  for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
  {
    const auto& imaginary = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
    std::fill(zero_imaginary.begin() + imaginary.begin, zero_imaginary.begin() + imaginary.end, 0);
  }
  replace(zero_imaginary);
  compare(model.cfg.electrons.x);
  for (const auto& r : result)
    if (cuCimag(r.phase) != 0)
      throw std::runtime_error("Complex VALUE zero-imaginary limit is not real");

  // Clone d0's complete envelope/head into d1, permitting exact controlled mixtures.
  auto paired = parameters;
  for (const auto& tensor : plan.parameterTensors())
    if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT)
      for (std::size_t f = 0; f < 16; ++f)
        std::copy_n(paired.data() + tensor.begin + f * 16, 4, paired.data() + tensor.begin + f * 16 + 4);
    else if (tensor.role == ParameterRole::ENVELOPE_PI_UP || tensor.role == ParameterRole::ENVELOPE_PI_DOWN ||
             tensor.role == ParameterRole::ENVELOPE_ZETA_UP || tensor.role == ParameterRole::ENVELOPE_ZETA_DOWN)
      std::copy_n(paired.data() + tensor.begin, 8, paired.data() + tensor.begin + 8);
  for (const auto& tensor : plan.parameterTensors())
    if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT)
      for (std::size_t f = 0; f < 16; ++f)
        std::fill_n(paired.data() + tensor.begin + f * 16 + 8, 8, 0);
  auto mixed = paired;
  for (const auto& tensor : plan.parameterTensors())
    if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT)
      for (std::size_t f = 0; f < 16; ++f)
        std::fill_n(mixed.data() + tensor.begin + f * 16 + 4, 4, 0);
  replace(mixed);
  compare(model.cfg.electrons.x);
  if (result[0].singular_channels != 3)
    throw std::runtime_error("Complex VALUE mixed-singular fixture lost its regular channel");
  for (double factor : {-0.999, -1.0})
  {
    auto cancelling = paired;
    for (const auto& tensor : plan.parameterTensors())
      if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT)
        for (std::size_t f = 0; f < 16; ++f)
          cancelling[tensor.begin + f * 16 + 4] *= factor;
    replace(cancelling);
    compare(model.cfg.electrons.x);
  }
  auto node = paired;
  for (const auto& tensor : plan.parameterTensors())
    if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT)
      std::fill(node.begin() + tensor.begin, node.begin() + tensor.end, 0);
  replace(node);
  compare(model.cfg.electrons.x);

  // A failed logical population revokes even the prior valid node publication.
  auto invalid = model.cfg.electrons.x;
  invalid.back() = std::numeric_limits<double>::quiet_NaN();
  rejected = false;
  try { gpu->evaluate(invalid); }
  catch (const std::runtime_error&) { rejected = true; }
  bool hidden = false;
  try { (void)gpu->resultView(); }
  catch (const std::logic_error&) { hidden = true; }
  if (!rejected || !hidden)
    throw std::runtime_error("Complex VALUE failure published a partial/stale population");
  model.p.set_flat_values(parameters);
  image = image->makeUpdatedImage(parameters, ++version);
  gpu->rebindModel(image);
  compare(model.cfg.electrons.x);
  std::cout << "PASS numerical complex GPU VALUE: comparisons=" << comparisons << " nodes=" << nodes
            << " max_log_error=" << max_log << " max_phase_error=" << max_phase
            << " trunk_bytes=" << cost.trunk_bytes << " owner_device_bytes=" << cost.device_bytes
            << " owner_host_bytes=" << cost.host_bytes
            << " warm_device_allocations=0 scope=device_wide shared_image_vendor_excluded=1\n";
}
#endif
