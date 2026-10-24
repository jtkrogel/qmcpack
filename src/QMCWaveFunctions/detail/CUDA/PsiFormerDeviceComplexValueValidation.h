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
#include <iomanip>
#include <exception>

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
  // Ordinary VALUE flags never admit SCORE events; only its separate explicit policy can own them.
  const auto score_cost = estimateComplexScoreCost(plan);
  const auto score_before = metrics.snapshot();
  auto score = PreparedDeviceComplexScore::prepare(successor, queue, blas, score_cost.device_bytes);
  const auto score_pair_bytes = configuredComplexScorePairCheck() && complexScorePairCheckEligible(plan) ?
      sizeof(std::uint64_t) : 0;
  if (score_cost.forward.pair_event_bytes != score_pair_bytes ||
      metrics.snapshot().live_bytes - score_before.live_bytes != score_cost.device_bytes)
    throw std::runtime_error("Eligible retained score inherited ordinary pair ownership");
  std::cout << "PASS eligible complex VALUE pair owner: Ne=64 capacity=2 event_device_owner_bytes="
            << cost.pair_event_bytes << " exact_cost=1 cost_minus_one_denied=1 warm_allocations=0 rebind=1 retained_score_excluded="
            << (score_pair_bytes == 0) << '\n';
}

/// Compare complete old/new reduction bits and independently enumerate event priority.
void validateComplexValuePairEvents(bool incoming_status_case = false)
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
        const std::uint64_t incoming = incoming_status_case && n == 64 && live == 1 && variant == 5 ? 7 : clear;
        old_status.upload(&incoming, 1); new_status.upload(&incoming, 1); events.upload(actual.data(), actual.size());
        checkRuntime(launchComplexValuePairCheck(queue.getNative(), live, n, spin, cell, positions.data(), events.data()),
                     "pair-event validation precheck");
        queue.sync();
        events.download(actual.data(), capacity);
        std::uint64_t old_failure, new_failure;
        new_status.download(&new_failure, 1);
        if (new_failure != incoming) throw std::runtime_error("Pair precheck changed numerical status early");
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

/// An invalid private base must fall back without changing canonical point failure priority.
void validateComplexVirtualEnvelopeFallback()
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  constexpr std::size_t n = 64, count = 2, elements = n * n;
  const auto layout = makeOpenOrbitalJetLayout(count, 1, n, 2, 1, n / 2, SpatialJetMode::VALUE_ONLY);
  const auto cell = periodic_geometry::makeCell({{{8., 0., 0.}, {0., 10., 0.}, {0., 0., 12.}}},
                                               {true, true, true});
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  DeviceBuffer<double> positions(3 * count * n), nuclei(6), pi(2 * n), zeta(2 * n);
  DeviceBuffer<double> anchor(elements), ordinary(count * elements), reused(count * elements);
  DeviceBuffer<std::size_t> moved(count);
  DeviceBuffer<std::uint64_t> anchor_status(1), ordinary_status(1), reused_status(1);
  std::vector<double> coordinates(3 * count * n, 2.0), nuclear_positions(6, 0.0);
  std::vector<double> coefficients(2 * n, .75 * std::numeric_limits<double>::max()), exponents(2 * n, 100.0);
  nuclei.upload(nuclear_positions.data(), nuclear_positions.size());
  pi.upload(coefficients.data(), coefficients.size());
  zeta.upload(exponents.data(), exponents.size());
  const std::uint64_t clear = UINT64_MAX;
  std::vector<double> expected(count * elements), actual(count * elements);
  for (bool repaired : {true, false})
  {
    std::fill(coordinates.begin(), coordinates.end(), 2.0);
    const std::size_t row = repaired ? 0 : 1;
    if (!repaired)
      for (std::size_t b = 0; b < count; ++b)
        std::fill_n(coordinates.begin() + b * 3 * n, 3, 0.0);
    const std::size_t rows[count]{row, row};
    positions.upload(coordinates.data(), coordinates.size());
    moved.upload(rows, count);
    anchor_status.upload(&clear, 1);
    ordinary_status.upload(&clear, 1);
    reused_status.upload(&clear, 1);
    // For repaired=true, only the private base's row0 is at the overflowing nucleus.
    checkRuntime(launchComplexValueEnvelopeAnchor(queue.getNative(), layout, cell, positions.data(), nuclei.data(),
        pi.data(), pi.data(), zeta.data(), zeta.data(), row, 0, 0, 0, anchor.data(), anchor_status.data()),
        "invalid private envelope anchor");
    checkRuntime(launchComplexValueEnvelopes(queue.getNative(), layout, cell, positions.data(), nuclei.data(),
        pi.data(), pi.data(), zeta.data(), zeta.data(), ordinary.data(), ordinary_status.data()),
        "canonical fallback envelope");
    checkRuntime(launchComplexValueVirtualEnvelopes(queue.getNative(), layout, cell, positions.data(), nuclei.data(),
        pi.data(), pi.data(), zeta.data(), zeta.data(), anchor.data(), moved.data(), anchor_status.data(),
        reused.data(), reused_status.data()), "private fallback envelope");
    queue.sync();
    std::uint64_t private_failure = clear, expected_failure = clear, actual_failure = clear;
    anchor_status.download(&private_failure, 1);
    ordinary_status.download(&expected_failure, 1);
    reused_status.download(&actual_failure, 1);
    ordinary.download(expected.data(), expected.size());
    reused.download(actual.data(), actual.size());
    if (private_failure == clear || expected_failure != actual_failure ||
        (repaired ? expected_failure != clear : expected_failure == clear))
      throw std::runtime_error("Private envelope fallback introduced or reordered a scientific failure");
    // Failed canonical elements need not publish a value; compare all regular elements only.
    for (std::size_t i = 0; i < actual.size(); ++i)
      if (repaired || (i / n) % n != 0)
        if (std::memcmp(&expected[i], &actual[i], sizeof(double)))
          throw std::runtime_error("Private envelope fallback changed canonical regular element bits");
  }
  std::cout << "PASS virtual envelope fallback: repaired_base=1 unrepaired_first_failure=exact old_GPU_bits=exact\n";
}

/// Check the eligible request owner against complete CPU and old GPU VALUE, including epoch and lifetime boundaries.
void validateComplexVirtualEnvelopeOwner()
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  const auto descriptors = makeTargetCostPlan(64, true, true);
  const auto initialized = initializePsiFormerParameters(descriptors.modelShape(), 1729,
      PSIFORMER_COMPLEX_PERIODIC_V1, descriptors.environment());
  const PsiFormerExecutionPlan plan(initialized.model_shape, initialized.layouts, initialized.environment);
  const auto n = plan.modelShape().electrons(), ni = plan.modelShape().nuclei;
  std::vector<double> nuclei(3 * ni), base(3 * n);
  for (std::size_t i = 0; i < nuclei.size(); ++i) nuclei[i] = .23 + .071 * i;
  for (std::size_t i = 0; i < base.size(); ++i) base[i] = 2.0 + 1.7 * std::sin(.37 * (i + 1));
  std::vector<pf::Layout> layouts;
  for (const auto& tensor : initialized.layouts)
    layouts.push_back({tensor.module, tensor.name, tensor.shape, tensor.begin, tensor.end});
  pf::ConfigData configuration(pf::Tensor({1, n, 3}, base), pf::Tensor({ni, 3}, nuclei),
      pf::Tensor({ni}, std::vector<double>(ni, 1)), n / 2, n - n / 2, plan.modelShape().determinants);
  pf::PsiFormer model(pf::Parameters(initialized.values, layouts), std::move(configuration));
  pf::DirectComplexValueExecutor cpu(model, plan);
  auto workspace = cpu.makeWorkspace();
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, initialized.values, nuclei);
  const auto ordinary = estimateComplexValueCost(plan, 3);
  const auto cost = estimateComplexValueCost(plan, 3, true, true);
  if (cost.envelope_anchor_bytes != 8 * plan.modelShape().determinants * n * n ||
      cost.device_bytes - ordinary.device_bytes != cost.envelope_anchor_bytes + 3 * sizeof(std::size_t) + 8 ||
      cost.host_bytes - ordinary.host_bytes != 3 * n * sizeof(double) + 3 * sizeof(std::size_t))
    throw std::runtime_error("Virtual envelope checked per-domain cost is inconsistent");
  const auto reject = [](auto&& operation, const char* message) {
    bool rejected = false;
    try { operation(); }
    catch (const std::exception&) { rejected = true; }
    if (!rejected) throw std::runtime_error(message);
  };
  reject([&] { (void)estimateComplexValueCost(makeTargetCostPlan(32, true, true), 3, true, true); },
         "Virtual envelope admitted Ne<64");
  reject([&] { (void)estimateComplexValueCost(plan, 3, false, true); },
         "Virtual envelope admitted retained-role costing");
  if (estimateComplexScoreCost(plan).forward.envelope_anchor_bytes)
    throw std::runtime_error("Retained score inherited virtual envelope storage");
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "virtual envelope owner device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "virtual envelope owner device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  const auto before_denial = metrics.snapshot();
  reject([&] { (void)PreparedDeviceComplexValue::prepare(image, 3, queue, blas, cost.device_bytes - 1, true); },
         "Virtual envelope cost-minus-one allocated an owner");
  if (metrics.snapshot().allocation_calls != before_denial.allocation_calls)
    throw std::runtime_error("Virtual envelope budget denial allocated storage");
  auto reference = PreparedDeviceComplexValue::prepare(image, 3, queue, blas, ordinary.device_bytes);
  const auto before_owner = metrics.snapshot();
  auto cached = PreparedDeviceComplexValue::prepare(image, 3, queue, blas, cost.device_bytes, true);
  if (metrics.snapshot().live_bytes - before_owner.live_bytes != cost.device_bytes)
    throw std::runtime_error("Virtual envelope actual DEVICE owner differs from checked cost");
  cached->evaluate(base);
  const auto preserved = cached->resultView();
  std::vector<ComplexValueResult> accepted(1), observed(1);
  cached->download(preserved, accepted);
  const auto before_begin = metrics.snapshot();
  auto request = cached->beginVirtualEnvelopeBase(base.data(), base.size());
  if (metrics.snapshot().uploaded_bytes != before_begin.uploaded_bytes ||
      metrics.snapshot().downloaded_bytes != before_begin.downloaded_bytes ||
      cached->resultView().generation != preserved.generation || cached->resultView().count != preserved.count)
    throw std::runtime_error("Virtual envelope begin changed ordinary epoch/count or transferred storage");
  cached->download(preserved, observed);
  if (std::memcmp(accepted.data(), observed.data(), sizeof(ComplexValueResult)))
    throw std::runtime_error("Virtual envelope begin changed the published ordinary amplitude");
  auto invalid_base = base;
  invalid_base.back() = std::numeric_limits<double>::quiet_NaN();
  reject([&] { (void)cached->beginVirtualEnvelopeBase(invalid_base.data(), invalid_base.size()); },
         "Virtual envelope accepted a nonfinite base");
  reject([&] { (void)cached->beginVirtualEnvelopeBase(base.data(), base.size() - 1); },
         "Virtual envelope accepted a truncated base");
  cached->download(preserved, observed);
  if (std::memcmp(accepted.data(), observed.data(), sizeof(ComplexValueResult)))
    throw std::runtime_error("Invalid virtual envelope begin revoked or changed an ordinary amplitude");
  request = cached->beginVirtualEnvelopeBase(base.data(), base.size());
  const auto warm = metrics.snapshot();
  const auto fingerprint = cached->virtualEnvelopeStorageFingerprint();
  std::size_t comparisons = 0;
  double max_cpu_log = 0, max_cpu_phase = 0;
  std::exception_ptr first_cpu_failure;
  const auto compare = [&](const std::vector<std::size_t>& rows) {
    std::vector<double> positions;
    for (std::size_t b = 0; b < rows.size(); ++b)
    {
      positions.insert(positions.end(), base.begin(), base.end());
      if (rows[b] == n)
        for (std::size_t i = 0; i < base.size(); ++i)
          positions[b * base.size() + i] += .041 + .003 * b;
      else
        positions[b * base.size() + 3 * rows[b]] += .031 + .002 * b;
    }
    reference->evaluate(positions);
    std::vector<ComplexValueResult> expected(rows.size()), actual(rows.size());
    reference->download(reference->resultView(), expected);
    cached->evaluateVirtualTile(request, positions, [&](std::size_t b) { return rows[b]; });
    cached->download(cached->resultView(), actual);
    if (std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(ComplexValueResult)))
      throw std::runtime_error("Virtual envelope complete GPU VALUE differs from ordinary GPU bits");
    for (std::size_t b = 0; b < rows.size(); ++b)
    {
      workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data() + b * base.size(), n));
      const auto& result = cpu.evaluate(*workspace);
      if (result.amplitude.isZero())
      {
        if (actual[b].status != 1 || cuCreal(actual[b].phase) || cuCimag(actual[b].phase))
          throw std::runtime_error("Virtual envelope CPU node contract differs");
      }
      else
      {
        const auto phase = std::complex<double>(cuCreal(actual[b].phase), cuCimag(actual[b].phase));
        max_cpu_log = std::max(max_cpu_log, std::abs(actual[b].log_abs - result.amplitude.log_abs));
        max_cpu_phase = std::max(max_cpu_phase, std::abs(phase - result.amplitude.phase));
        if (std::abs(actual[b].log_abs - result.amplitude.log_abs) > 2e-10 * (1 + std::abs(result.amplitude.log_abs)) ||
            std::abs(phase.real() - result.amplitude.phase.real()) > 2e-10 * (1 + std::abs(result.amplitude.phase.real())) ||
            std::abs(phase.imag() - result.amplitude.phase.imag()) > 2e-10 * (1 + std::abs(result.amplitude.phase.imag())))
          std::cout << std::setprecision(17) << "VIRTUAL_ENVELOPE_STRICT_CPU_FAILURE b=" << b
                    << " moved_row=" << rows[b] << " old_log=" << expected[b].log_abs
                    << " cache_log=" << actual[b].log_abs << " cpu_log=" << result.amplitude.log_abs
                    << " old_phase=" << cuCreal(expected[b].phase) << ',' << cuCimag(expected[b].phase)
                    << " cache_phase=" << phase << " cpu_phase=" << result.amplitude.phase
                    << " old_cpu_log_abs=" << std::abs(expected[b].log_abs - result.amplitude.log_abs)
                    << " cache_cpu_log_abs=" << std::abs(actual[b].log_abs - result.amplitude.log_abs)
                    << " old_cpu_phase_abs=" << std::abs(std::complex<double>(cuCreal(expected[b].phase),
                         cuCimag(expected[b].phase)) - result.amplitude.phase)
                    << " cache_cpu_phase_abs=" << std::abs(phase - result.amplitude.phase)
                    << " old_cancel=" << expected[b].cancellation_ratio
                    << " cache_cancel=" << actual[b].cancellation_ratio << " cpu_cancel=" << result.cancellation_ratio
                    << " singular_channels=" << actual[b].singular_channels << " old_cache_bits=exact\n";
        try
        {
          near(actual[b].log_abs, result.amplitude.log_abs, "virtual envelope CPU log", 2e-10);
          near(phase.real(), result.amplitude.phase.real(), "virtual envelope CPU phase real", 2e-10);
          near(phase.imag(), result.amplitude.phase.imag(), "virtual envelope CPU phase imaginary", 2e-10);
        }
        catch (...)
        {
          // Continue independent exact-GPU/lifetime checks, but retain the first strict CPU failure.
          if (!first_cpu_failure) first_cpu_failure = std::current_exception();
        }
      }
      ++comparisons;
    }
  };
  compare({0, n / 2 - 1, n - 1});
  compare({n / 2});
  compare({n - 1, 0});
  compare({0, n, n - 1});
  compare({n - 1, n});
  const auto last = cached->resultView();
  cached->endVirtualEnvelopeBase();
  if (cached->resultView().generation != last.generation)
    throw std::runtime_error("Virtual envelope end changed the published amplitude epoch");
  reject([&] { cached->evaluateVirtualTile(request, base, [](std::size_t) { return 0; }); },
         "Virtual envelope consumed an ended request");
  request = cached->beginVirtualEnvelopeBase(base.data(), base.size());
  auto invalid_metadata = base;
  invalid_metadata.insert(invalid_metadata.end(), base.begin(), base.end());
  reject([&] { cached->evaluateVirtualTile(request, invalid_metadata,
      [&](std::size_t b) { return b ? n + 1 : 0; }); }, "Virtual envelope accepted metadata above its fallback sentinel");
  request = cached->beginVirtualEnvelopeBase(base.data(), base.size());
  reject([&] { cached->evaluateVirtualTile(request, base, [&](std::size_t) { return n; }); },
         "Virtual envelope admitted a first-point fallback without a named anchor row");
  request = cached->beginVirtualEnvelopeBase(base.data(), base.size());
  auto wrong_row = base;
  wrong_row[3] += .07;
  reject([&] { cached->evaluateVirtualTile(request, wrong_row, [](std::size_t) { return 0; }); },
         "Virtual envelope reused a changed unnamed row");
  request = cached->beginVirtualEnvelopeBase(base.data(), base.size());
  compare({0, n - 1});
  auto foreign_request = request;
  foreign_request.owner = reference.get();
  reject([&] { cached->evaluateVirtualTile(foreign_request, base, [](std::size_t) { return 0; }); },
         "Virtual envelope consumed a foreign owner token");
  request = cached->beginVirtualEnvelopeBase(base.data(), base.size());
  auto node = base;
  std::copy_n(base.data() + 3, 3, node.data());
  reference->evaluate(node);
  std::vector<ComplexValueResult> old_node(1), new_node(1);
  reference->download(reference->resultView(), old_node);
  cached->evaluateVirtualTile(request, node, [](std::size_t) { return 0; });
  cached->download(cached->resultView(), new_node);
  if (std::memcmp(old_node.data(), new_node.data(), sizeof(ComplexValueResult)) || new_node[0].status != 1)
    throw std::runtime_error("Virtual envelope changed an exact coincident-row node");
  if (metrics.snapshot().allocation_calls != warm.allocation_calls ||
      metrics.snapshot().free_calls != warm.free_calls || cached->virtualEnvelopeStorageFingerprint() != fingerprint)
    throw std::runtime_error("Virtual envelope warmed request changed storage");
  const auto stale = request;
  auto successor = image->makeUpdatedImage(initialized.values, 1);
  cached->validateModelRebind(successor);
  const auto adoption = metrics.snapshot();
  cached->adoptPreparedModel(successor);
  reference->rebindModel(successor);
  reject([&] { cached->evaluateVirtualTile(stale, base, [](std::size_t) { return 0; }); },
         "Virtual envelope retained an adopted-image request");
  if (metrics.snapshot().allocation_calls != adoption.allocation_calls)
    throw std::runtime_error("Virtual envelope adoption allocated request storage");
  request = cached->beginVirtualEnvelopeBase(base.data(), base.size());
  compare({0});
  auto successor2 = successor->makeUpdatedImage(initialized.values, 2);
  cached->rebindModel(successor2);
  reference->rebindModel(successor2);
  reject([&] { cached->evaluateVirtualTile(request, base, [](std::size_t) { return 0; }); },
         "Virtual envelope retained a rebound request");
  request = cached->beginVirtualEnvelopeBase(base.data(), base.size());
  compare({n - 1});
  for (int pass : {0, 1})
    for (int action : {0, 1, 2})
    {
      request = cached->beginVirtualEnvelopeBase(base.data(), base.size());
      int calls = 0;
      reject([&] {
        cached->evaluateVirtualTile(request, base, [&](std::size_t) {
          if (calls++ == pass)
          {
            if (action == 0) cached->endVirtualEnvelopeBase();
            else if (action == 1) (void)cached->beginVirtualEnvelopeBase(base.data(), base.size());
            else cached->rebindModel(successor2);
          }
          return 0;
        });
      }, "Virtual envelope accepted accessor reentry during authority validation");
      request = cached->beginVirtualEnvelopeBase(base.data(), base.size());
      compare({0});
    }
  cached->endVirtualEnvelopeBase();
  if (first_cpu_failure)
  {
    std::cout << "VIRTUAL_ENVELOPE_GPU_LIFECYCLE_PASS CPU_GATE_FAILED Ne=" << n
              << " comparisons=" << comparisons << " old_GPU_bits=exact warm_allocations=0"
              << " max_cpu_log=" << max_cpu_log << " max_cpu_phase=" << max_cpu_phase << '\n';
    std::rethrow_exception(first_cpu_failure);
  }
  std::cout << "PASS virtual envelope owner: Ne=" << n << " capacity=3 CPU_comparisons=" << comparisons
            << " max_cpu_log=" << max_cpu_log << " max_cpu_phase=" << max_cpu_phase
            << " old_GPU_bits=exact begin_ordinary_epoch_unchanged=1 invalid_begin_preserved=1"
            << " warm_allocations=0 exact_cost=1 stale_adopt_rebind=1\n";
}

/// Numerical-only complete VALUE gate; fixture builders are the periodic validator's.
void validateComplexValue()
{
  validateComplexValuePairOwner();
  validateComplexValuePairEvents();
  validateComplexVirtualEnvelopeFallback();
  validateComplexVirtualEnvelopeOwner();
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
