//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_COMPLEX_SPATIAL_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_COMPLEX_SPATIAL_VALIDATION_H
#include "PsiFormerDeviceMixedComplexSpatial.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexSpatialExecutor.h"
#include <complex>
#include <cstring>
#include <map>

/// Optional inverse diagnostics must not perturb one bit of canonical inverse/LU state.
inline void validateMixedComplexSpatialInverseDiagnostics()
{
  using namespace qmcplusplus::psiformer::device;
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  constexpr std::size_t channels = 2, n = 3, elements = channels * n * n;
  std::vector<cuDoubleComplex> input(elements, make_cuDoubleComplex(0, 0));
  for (std::size_t c = 0; c < channels; ++c)
    for (std::size_t i = 0; i < n; ++i)
      input[(c * n + i) * n + i] = make_cuDoubleComplex((c && i == 2) ? 1e-7 : 1. + i, .125);
  DeviceBuffer<cuDoubleComplex> a(elements), b(elements), ia(elements), ib(elements), sa(channels * n), sb(channels * n);
  DeviceBuffer<std::size_t> pa(channels * n), pb(channels * n);
  DeviceBuffer<ComplexValueChannel> ca(channels), cb(channels);
  DeviceBuffer<double> pivots(channels);
  DeviceBuffer<std::uint64_t> sta(1), stb(1);
  const std::uint64_t clear = UINT64_MAX;
  a.upload(input.data(), input.size()); b.upload(input.data(), input.size());
  sta.upload(&clear, 1); stb.upload(&clear, 1);
  checkRuntime(launchComplexValueFactor(queue.getNative(), channels, n, a.data(), ca.data(), sta.data(),
      ia.data(), pa.data(), sa.data()), "default complex inverse oracle");
  checkRuntime(launchComplexSpatialFactorWithDiagnostics(queue.getNative(), channels, n, b.data(), cb.data(), stb.data(),
      ib.data(), pb.data(), sb.data(), pivots.data()), "diagnostic complex inverse oracle");
  queue.sync();
  const auto identical = [](const auto& x, const auto& y, std::size_t count) {
    using T = typename std::remove_pointer<decltype(x.data())>::type;
    std::vector<typename std::remove_const<T>::type> lhs(count), rhs(count);
    x.download(lhs.data(), count); y.download(rhs.data(), count);
    return std::memcmp(lhs.data(), rhs.data(), count * sizeof(lhs[0])) == 0;
  };
  if (!identical(a, b, elements) || !identical(ia, ib, elements) || !identical(pa, pb, channels * n) ||
      !identical(sa, sb, channels * n) || !identical(ca, cb, channels) || !identical(sta, stb, 1))
    throw std::runtime_error("Optional complex inverse diagnostics changed canonical state bits");
  std::vector<double> observed(channels); pivots.download(observed.data(), channels);
  for (const auto p : observed)
    if (!(p > 0 && std::isfinite(p))) throw std::runtime_error("Inverse diagnostic pivot missing");
  std::cout << "PASS mixed FULL inverse diagnostics: LU_inverse_permutation_solve_metadata_status_bits=exact\n";
}

/// Force complete-field retries; exact nodes must preserve the canonical hard failure boundary.
inline void validateMixedComplexSpatialFullReplays(pf::PsiFormer& model,
    const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan, std::size_t concurrent_targets = 0)
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  const auto n = plan.modelShape().electrons(), d = plan.modelShape().determinants;
  const auto original = model.p.flat_values();
  const std::vector<double> positions(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * n);
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  struct Fields
  {
    std::vector<ComplexValueResult> amplitude;
    std::vector<cuDoubleComplex> gradient, lap_log, lap_ratio;
  };
  const auto fetch = [&](const auto& owner) {
    Fields f{std::vector<ComplexValueResult>(1), std::vector<cuDoubleComplex>(3 * n),
             std::vector<cuDoubleComplex>(n), std::vector<cuDoubleComplex>(n)};
    owner.download(owner.resultView(), f.amplitude, f.gradient, f.lap_log, f.lap_ratio);
    return f;
  };
  const auto equal = [](const Fields& a, const Fields& b) {
    const auto bits = [](const auto& x, const auto& y) {
      return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])) == 0;
    };
    return bits(a.amplitude, b.amplitude) && bits(a.gradient, b.gradient) &&
        bits(a.lap_log, b.lap_log) && bits(a.lap_ratio, b.lap_ratio);
  };
  for (const char* kind : {"conversion", "activation", "pivot", "cancellation"})
  {
    auto values = original;
    const auto& embedding = plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
    if (!std::strcmp(kind, "conversion")) values[embedding.begin] = 1e-300;
    else if (!std::strcmp(kind, "activation"))
    {
      const auto cell = periodic_geometry::makeCell(plan.environment().lattice_vectors, plan.environment().periodic_axes);
      const auto layout = makePeriodicFeatureJetLayout(1, n, model.cfg.nuclei.x.size() / 3,
          plan.modelShape().spin_up_electrons, SpatialJetMode::VALUE_ONLY);
      std::vector<double> raw(layout.output.configuration_stride);
      if (open_spatial::buildOpenFeatureJetsForConfiguration(layout, positions.data(), model.cfg.nuclei.x.data(),
              0, 0, raw.data(), &cell) != OpenSpatialStatus::REGULAR)
        throw std::runtime_error("FULL activation fixture raw row is inadmissible");
      double sum = 0; for (std::size_t row = 0; row < layout.feature_width; ++row) sum += std::abs(raw[row]);
      if (!(sum > 2)) throw std::runtime_error("FULL activation fixture lacks analytical overflow margin");
      // Finite float coefficients, known aligned row: dot > .75*FLT_MAX*2,
      // exercising executed activation overflow rather than conversion rejection.
      const double coefficient = static_cast<double>(.75f * std::numeric_limits<float>::max());
      for (std::size_t row = 0; row < layout.feature_width; ++row)
        values[embedding.begin + row * embedding.shape[1]] = std::copysign(coefficient, raw[row]);
    }
    else
      for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
        for (auto component : {HeadTensorComponent::REAL, HeadTensorComponent::IMAG})
        {
          const auto& weight = plan.parameter(role, NO_ATTENTION_BLOCK, component);
          for (std::size_t row = 0; row < plan.modelShape().feature_dimension; ++row)
            for (std::size_t determinant = 0; determinant < d; ++determinant)
              for (std::size_t column = 0; column < n; ++column)
              {
                const auto index = weight.begin + row * d * n + determinant * n + column;
                if (!std::strcmp(kind, "pivot")) { if (!determinant) values[index] = 0; }
                else values[index] = original[weight.begin + row * d * n + column] *
                    (determinant % 2 && column == 0 ? -1. : 1.);
              }
        }
    auto image = PreparedDeviceModel::preparePairedComplexValue(plan, values, model.cfg.nuclei.x);
    PreparedDeviceMixedComplexSpatial mixed(image, 1, queue, blas,
        PreparedDeviceMixedComplexSpatial::estimateDeviceBytes(plan, 1, concurrent_targets), true, concurrent_targets);
    auto reference = PreparedDeviceComplexSpatial::prepare(image, 1, queue, blas,
        estimateComplexSpatialCost(plan, 1, concurrent_targets).device_bytes, concurrent_targets);
    std::string mixed_failure, reference_failure;
    try { mixed.evaluate(positions); } catch (const std::exception& e) { mixed_failure = e.what(); }
    try { reference->evaluate(positions); } catch (const std::exception& e) { reference_failure = e.what(); }
    if (!mixed.usedFp64Retry() || mixed.retryReason().empty())
      throw std::runtime_error(std::string("Forced FULL precision hazard did not execute complete retry: ") + kind);
    const auto retry_reason = mixed.retryReason();
    if (std::strcmp(kind, "cancellation") && std::strcmp(kind, "pivot"))
    {
      if (!mixed_failure.empty() || !reference_failure.empty() || !equal(fetch(mixed), fetch(*reference)))
        throw std::runtime_error(std::string("Complete FULL retry fields differ: ") + kind + "; " + mixed_failure);
      std::cout << "PASS mixed FULL complete retry: kind=" << kind << " all_amplitude_G_L_fields_bits=exact";
    }
    else
    {
      // FULL needs each determinant inverse: an exactly singular determinant
      // or total amplitude node has no admitted FULL fields. Preserve FP64 failure.
      if (mixed_failure.empty() || mixed_failure != reference_failure)
        throw std::runtime_error(std::string("FULL singular/node failure differs: ") + kind + "; mixed=" +
                                 mixed_failure + "; reference=" + reference_failure);
      bool unavailable = false;
      try { (void)mixed.resultView(); } catch (const std::logic_error&) { unavailable = true; }
      if (!unavailable) throw std::runtime_error("FULL node published derivative authority");
      auto regular = image->makeUpdatedImage(original, 1);
      mixed.rebindModel(regular, mixed.successorWorkspaceBytes());
      const auto prior_projections = mixed.statistics().projections;
      mixed.evaluate(positions);
      if (mixed.statistics().projections <= prior_projections)
        throw std::runtime_error("FULL failed-replay restoration silently left the affine adapter detached");
      PreparedDeviceMixedComplexSpatial fresh(regular, 1, queue, blas,
          PreparedDeviceMixedComplexSpatial::estimateDeviceBytes(plan, 1, concurrent_targets), true, concurrent_targets);
      fresh.evaluate(positions);
      if (!equal(fetch(mixed), fetch(fresh))) throw std::runtime_error("FULL replay failure did not restore adapter on rebind");
      std::cout << "PASS mixed FULL complete retry: kind=" << kind
                << " canonical_failure=exact no_publication=1 restored_adapter=1";
    }
    std::cout << " concurrent_trace_targets=" << concurrent_targets << " reason=" << retry_reason << '\n';
  }
}

/// First FULL numerical evidence: finite errors are measured, not assigned VALUE's tolerance.
template<class ParameterFactory, class ConfigurationFactory>
inline void measureMixedComplexSpatialNumerical(ParameterFactory parameters, ConfigurationFactory configurations)
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  using C = std::complex<double>;
  const auto complex = [](cuDoubleComplex z) { return C(cuCreal(z), cuCimag(z)); };
  std::cout.precision(17);
  validateMixedComplexSpatialInverseDiagnostics();
  // Compare replay bits to the SAME trace layout; t0/t8 are not assumed bit-identical.
  for (std::size_t concurrent_targets : {0u, 8u})
  for (std::size_t up : {2u, 1u})
  {
    ExecutionEnvironment environment;
    environment.boundary = BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.lattice_vectors = {{{4., .4, -.2}, {.7, 5., .3}, {-.1, .8, 6.}}};
    environment.periodic_axes = {true, true, true};
    environment.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
    environment.amplitude_scalar_domain = ScalarDomain::COMPLEX;
    auto configuration = configurations(environment);
    configuration.nup = up; configuration.ndown = 4 - up;
    pf::PsiFormer model(parameters(environment), std::move(configuration), 4, 16, 2, 2);
    const auto plan = PsiFormerExecutionPlan::fromParameters(model.p, ModelShape{up, 4 - up, 2, 4, 16, 2, 2}, environment);
    auto values = model.p.flat_values();
    for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
    {
      const auto& imaginary = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
      for (std::size_t i = imaginary.begin; i < imaginary.end; ++i)
        values[i] = .09 * std::sin(double(i - imaginary.begin + 1));
    }
    model.p.set_flat_values(values);
    auto image = PreparedDeviceModel::preparePairedComplexValue(plan, values, model.cfg.nuclei.x);
    qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
    AcceleratorBlasHandle blas(queue);
    int ordinal = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&ordinal), "mixed FULL validation device");
#else
    checkRuntime(cudaGetDevice(&ordinal), "mixed FULL validation device");
#endif
    auto& metrics = deviceMemoryMetrics(ordinal);
    const auto budget = PreparedDeviceMixedComplexSpatial::estimateDeviceBytes(plan, 3, concurrent_targets);
    const auto before = metrics.snapshot();
    bool rejected = false;
    try { PreparedDeviceMixedComplexSpatial denied(image, 3, queue, blas, budget - 1, true, concurrent_targets); }
    catch (const std::length_error&) { rejected = true; }
    if (!rejected || metrics.snapshot().allocation_calls != before.allocation_calls)
      throw std::runtime_error("Mixed FULL cost-minus-one allocated device storage");
    PreparedDeviceMixedComplexSpatial mixed(image, 3, queue, blas, budget, true, concurrent_targets);
    if (metrics.snapshot().live_bytes - before.live_bytes != budget ||
        metrics.snapshot().peak_bytes > std::max(before.peak_bytes, before.live_bytes + budget))
      throw std::runtime_error("Mixed FULL exact/cold peak inventory differs from checked budget");
    auto fp64 = PreparedDeviceComplexSpatial::prepare(image, 3, queue, blas,
        estimateComplexSpatialCost(plan, 3, concurrent_targets).device_bytes, concurrent_targets);
    auto oracle = std::make_unique<pf::DirectComplexSpatialExecutor>(model, plan);
    auto workspace = oracle->makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
    double amplitude_error = 0, phase_error = 0, log_error = 0, gradient_error = 0;
    double lap_log_error = 0, lap_ratio_error = 0, kinetic_error = 0, normalized_field_error = 0;
    // [maximum absolute error, reference magnitude there, maximum normalized
    // error, reference magnitude there]; maxima need not occur at the same point.
    std::map<std::string, std::array<double, 4>> field_errors;
    std::size_t requests = 0, replays = 0, comparisons = 0;
    const auto compare = [&](const std::vector<double>& positions) {
      const auto count = positions.size() / 12;
      std::vector<ComplexValueResult> a(count), b(count);
      std::vector<cuDoubleComplex> g(12 * count), l(4 * count), r(4 * count);
      auto bg = g, bl = l, br = r;
      mixed.evaluate(positions); fp64->evaluate(positions);
      mixed.download(mixed.resultView(), a, g, l, r);
      fp64->download(fp64->resultView(), b, bg, bl, br);
      ++requests; replays += mixed.usedFp64Retry();
      for (std::size_t c = 0; c < count; ++c)
      {
        workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data() + 12 * c, 4));
        const auto expected = oracle->evaluateFull(*workspace);
        const auto observe = [&](C actual, C reference, const char* field) {
          if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
              !std::isfinite(reference.real()) || !std::isfinite(reference.imag()))
            throw std::runtime_error("Mixed FULL error evidence contains a nonfinite admitted field");
          const auto error = std::abs(actual - reference);
          const auto magnitude = std::abs(reference), normalized = error / (1 + magnitude);
          auto& maxima = field_errors[field];
          if (error > maxima[0]) { maxima[0] = error; maxima[1] = magnitude; }
          if (normalized > maxima[2]) { maxima[2] = normalized; maxima[3] = magnitude; }
          normalized_field_error = std::max(normalized_field_error, normalized);
          return error;
        };
        const auto fp64_near = [&](C actual, C reference) {
          if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
              !std::isfinite(reference.real()) || !std::isfinite(reference.imag()) ||
              std::abs(actual - reference) > 2e-10 * (1 + std::abs(reference)))
            throw std::runtime_error("Independent complete FP64 FULL oracle mismatch");
        };
        fp64_near(C(b[c].log_abs), C(expected.amplitude.log_abs));
        fp64_near(complex(b[c].phase), expected.amplitude.phase);
        log_error = std::max(log_error, observe(C(a[c].log_abs), C(expected.amplitude.log_abs), "log_amplitude"));
        phase_error = std::max(phase_error, observe(complex(a[c].phase), expected.amplitude.phase, "phase"));
        amplitude_error = std::max(amplitude_error, observe(std::exp(a[c].log_abs - expected.amplitude.log_abs) *
            complex(a[c].phase) / expected.amplitude.phase, C(1), "normalized_amplitude"));
        C kinetic{}, expected_kinetic{};
        for (std::size_t e = 0; e < 4; ++e)
        {
          for (std::size_t mu = 0; mu < 3; ++mu)
          {
            const auto i = 12 * c + 3 * e + mu;
            fp64_near(complex(bg[i]), expected.gradient[3 * e + mu]);
            gradient_error = std::max(gradient_error, observe(complex(g[i]), expected.gradient[3 * e + mu], "G"));
          }
          const auto i = 4 * c + e;
          fp64_near(complex(bl[i]), expected.laplacian_log[e]);
          fp64_near(complex(br[i]), expected.laplacian_ratio[e]);
          lap_log_error = std::max(lap_log_error, observe(complex(l[i]), expected.laplacian_log[e], "Llog"));
          lap_ratio_error = std::max(lap_ratio_error, observe(complex(r[i]), expected.laplacian_ratio[e], "Lratio"));
          kinetic -= .5 * complex(r[i]); expected_kinetic -= .5 * expected.laplacian_ratio[e];
        }
        kinetic_error = std::max(kinetic_error, observe(kinetic, expected_kinetic, "kinetic"));
        ++comparisons;
      }
    };
    compare(std::vector<double>(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 12));
    const auto old = mixed.resultView();
    const auto warm = metrics.snapshot();
    for (std::size_t count : {3u, 1u, 3u})
      compare(std::vector<double>(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 12 * count));
    const auto after = metrics.snapshot();
    if (after.allocation_calls != warm.allocation_calls || after.free_calls != warm.free_calls)
      throw std::runtime_error("Mixed FULL warmed evaluation allocated numeric storage");
    rejected = false;
    std::vector<ComplexValueResult> a(1);
    std::vector<cuDoubleComplex> g(12), l(4), r(4);
    try { mixed.download(old, a, g, l, r); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Mixed FULL admitted stale wrapper authority");
    // Change one converted affine and one FP64 head coefficient while retaining
    // the original image; an identical-value successor would hide stale caches.
    const auto& embedding = plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
    values[embedding.begin] += .125;
    const auto& head = plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK, HeadTensorComponent::REAL);
    values[head.begin] += .03125;
    model.p.set_flat_values(values);
    auto successor = image->makeUpdatedImage(values, 1);
    const auto pending = mixed.successorWorkspaceBytes();
    const auto accepted = mixed.resultView();
    rejected = false;
    try { mixed.rebindModel(successor, pending - 1); }
    catch (const std::length_error&) { rejected = true; }
    if (!rejected || mixed.resultView().generation != accepted.generation)
      throw std::runtime_error("Failed mixed FULL successor admission revoked accepted authority");
    mixed.rebindModel(successor, pending);
    rejected = false;
    try { (void)mixed.resultView(); }
    catch (const std::logic_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Mixed FULL successor published without evaluation");
    fp64->rebindModel(successor);
    workspace.reset();
    oracle = std::make_unique<pf::DirectComplexSpatialExecutor>(model, plan);
    workspace = oracle->makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
    const std::vector<double> rebound_positions(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 12);
    compare(rebound_positions);
    PreparedDeviceMixedComplexSpatial fresh(successor, 3, queue, blas, budget, true, concurrent_targets);
    fresh.evaluate(rebound_positions);
    std::vector<ComplexValueResult> rebound(1), fresh_amplitude(1);
    std::vector<cuDoubleComplex> rebound_g(12), rebound_l(4), rebound_r(4), fresh_g(12), fresh_l(4), fresh_r(4);
    mixed.download(mixed.resultView(), rebound, rebound_g, rebound_l, rebound_r);
    fresh.download(fresh.resultView(), fresh_amplitude, fresh_g, fresh_l, fresh_r);
    if (std::memcmp(rebound.data(), fresh_amplitude.data(), sizeof(rebound[0])) ||
        std::memcmp(rebound_g.data(), fresh_g.data(), rebound_g.size() * sizeof(rebound_g[0])) ||
        std::memcmp(rebound_l.data(), fresh_l.data(), rebound_l.size() * sizeof(rebound_l[0])) ||
        std::memcmp(rebound_r.data(), fresh_r.data(), rebound_r.size() * sizeof(rebound_r[0])))
      throw std::runtime_error("Changed-weight mixed FULL rebind differs from fresh complete owner");
    auto invalid_positions = rebound_positions;
    invalid_positions[0] = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try { mixed.evaluate(invalid_positions); }
    catch (const InferenceStatusError& e) { rejected = e.stage() == 0; }
    if (!rejected || mixed.usedFp64Retry())
      throw std::runtime_error("Physical FULL failure was hidden by precision replay");
    rejected = false;
    try { (void)mixed.resultView(); }
    catch (const std::logic_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Failed FULL physical input published partial authority");
    compare(rebound_positions);
    std::cout << "EVIDENCE paired mixed FULL: up=" << up << " down=" << 4 - up
              << " concurrent_trace_targets=" << concurrent_targets
              << " graph_workspaces=1 exact_device_bytes=" << budget << " warm_allocations=0 comparisons=" << comparisons
              << " actual_mixed_requests=" << requests - replays << " complete_FP64_replays=" << replays
              << " maximum_normalized_amplitude_error=" << amplitude_error << " maximum_phase_error=" << phase_error
              << " maximum_log_error=" << log_error << " maximum_G_error=" << gradient_error
              << " maximum_Llog_error=" << lap_log_error << " maximum_Lratio_error=" << lap_ratio_error
              << " maximum_kinetic_error=" << kinetic_error << " maximum_normalized_field_error=" << normalized_field_error
              << " changed_weight_rebind_fresh_bits=exact physical_failure_replay=0"
              << "; MEASUREMENT_ONLY_NO_FULL_ACCURACY_ADMISSION\n";
    for (const auto& item : field_errors)
      std::cout << "FIELD paired mixed FULL: up=" << up << " name=" << item.first
                << " max_absolute=" << item.second[0] << " reference_magnitude_at_absolute=" << item.second[1]
                << " max_normalized=" << item.second[2] << " reference_magnitude_at_normalized=" << item.second[3] << '\n';
    if (up == 2) validateMixedComplexSpatialFullReplays(model, plan, concurrent_targets);
  }
}
#endif
