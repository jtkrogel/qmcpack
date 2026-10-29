//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_COMPLEX_SPATIAL_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_COMPLEX_SPATIAL_VALIDATION_H
#include "PsiFormerDeviceMixedComplexSpatial.h"
#include "PsiFormerDeviceMixedComplexInference.h"
#include "PsiFormerDeviceComplexProducts.h"
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

/// Exercise the composing native owner's cache accounting and two-leaf publication boundary.
inline void validateMixedComplexNativeOwners(
    const std::shared_ptr<const qmcplusplus::psiformer::device::PreparedDeviceModel>& image,
    const std::vector<double>& positions, std::vector<double> values,
    qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA>& queue,
    qmcplusplus::psiformer::device::AcceleratorBlasHandle& blas)
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  using Owner = PreparedDeviceMixedComplexInference;
  const auto& plan = image->plan();
  const auto owned = Owner::required(plan, 1), borrowed = Owner::required(plan, 1, 0, false);
  const auto cache_bytes = checkedDeviceBytes<float>(plan.parameterCount());
  if (owned.device_bytes - borrowed.device_bytes != cache_bytes || borrowed.pending_device_bytes ||
      owned.host_bytes - borrowed.host_bytes != sizeof(PsiFormerParameterConversionDiagnostics))
    throw std::runtime_error("Native mixed owner shared-cache accounting is inconsistent");
  const auto budget = spatial_detail::checkedAdd(owned.device_bytes, owned.pending_device_bytes,
      "Native mixed test budget overflow");
  bool rejected = false;
  try { (void)Owner::admittedCost(plan, 1, budget - 1); }
  catch (const std::length_error&) { rejected = true; }
  if (!rejected) throw std::runtime_error("Native mixed owner accepted cap-minus-one");
  Owner owner(image, 1, queue, blas, budget);
  const auto n = plan.modelShape().electrons();
  struct Fields
  {
    std::vector<ComplexValueResult> value, full;
    std::vector<cuDoubleComplex> gradient, laplacian, ratio;
  };
  const auto fetch = [&](Owner& selected) {
    Fields f{{ComplexValueResult{}}, {ComplexValueResult{}},
        std::vector<cuDoubleComplex>(3 * n), std::vector<cuDoubleComplex>(n), std::vector<cuDoubleComplex>(n)};
    selected.downloadValue(f.value);
    selected.downloadFull(f.full, f.gradient, f.laplacian, f.ratio);
    return f;
  };
  const auto equal = [](const Fields& a, const Fields& b) {
    const auto bits = [](const auto& x, const auto& y) {
      return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])) == 0;
    };
    return bits(a.value, b.value) && bits(a.full, b.full) && bits(a.gradient, b.gradient) &&
        bits(a.laplacian, b.laplacian) && bits(a.ratio, b.ratio);
  };
  owner.evaluateValue(positions);
  owner.evaluateFull(positions);
  const auto accepted = fetch(owner);
  Owner stale_crowd(image, 1, queue, blas, borrowed.device_bytes, 0, owner.parameterCache());
  std::weak_ptr<const PreparedDeviceMixedDenseParameters> old_cache = owner.parameterCache();
  const auto& embedding = plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
  values[embedding.begin] += .125;
  auto successor = image->makeUpdatedImage(values, image->version() + 1);
  // A cache for the accepted image cannot authorize either successor leaf.
  rejected = false;
  try { owner.validatePreparedRebind(successor, owner.parameterCache()); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected || !equal(accepted, fetch(owner)))
    throw std::runtime_error("Rejected native mixed publication changed accepted VALUE/FULL bits");
  auto next = std::make_shared<PreparedDeviceMixedDenseParameters>(successor, queue,
      MixedTrunkHeadPolicy::PAIRED_COMPLEX_FP64_HEAD, true);
  // Retain both old immutable authorities until all no-throw leaf adoptions complete.
  auto retired_image = image;
  auto retired_cache = owner.parameterCache();
  owner.validatePreparedRebind(successor, next);
  owner.adoptPreparedRebind(successor, next);
  rejected = false;
  try { (void)fetch(owner); } catch (const std::logic_error&) { rejected = true; }
  if (!rejected) throw std::runtime_error("Native mixed publication retained an old output authority");
  owner.evaluateValue(positions);
  owner.evaluateFull(positions);
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "native mixed stale-cache accounting device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "native mixed stale-cache accounting device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  const auto before_retirement = metrics.snapshot();
  retired_cache.reset();
  if (old_cache.expired() || metrics.snapshot().live_bytes != before_retirement.live_bytes)
    throw std::runtime_error("Publication released a cache still owned by an idle crowd");
  stale_crowd.validatePreparedRebind(successor, next);
  stale_crowd.adoptPreparedRebind(successor, next);
  if (!old_cache.expired() || metrics.snapshot().live_bytes + cache_bytes != before_retirement.live_bytes)
    throw std::runtime_error("Last stale crowd did not release exactly one shared FP32 generation");
  Owner fresh(successor, 1, queue, blas, borrowed.device_bytes, 0, next);
  fresh.evaluateValue(positions);
  fresh.evaluateFull(positions);
  if (fresh.parameterCache().get() != next.get() || !equal(fetch(owner), fetch(fresh)))
    throw std::runtime_error("Native mixed changed-weight rebind differs from fresh shared-cache owner");
  owner.evaluateValue(positions, PsiFormerGpuTrainingEpochMode::FP64_REPLAY);
  owner.evaluateFull(positions, PsiFormerGpuTrainingEpochMode::FP64_REPLAY);
  const auto forced = fetch(owner);
  auto canonical_value = PreparedDeviceComplexValue::prepare(successor, 1, queue, blas,
      estimateComplexValueCost(plan, 1, true, false).device_bytes);
  auto canonical_full = PreparedDeviceComplexSpatial::prepare(successor, 1, queue, blas,
      estimateComplexSpatialCost(plan, 1).device_bytes);
  canonical_value->evaluate(positions);
  canonical_full->evaluate(positions);
  Fields canonical{{ComplexValueResult{}}, {ComplexValueResult{}},
      std::vector<cuDoubleComplex>(3 * n), std::vector<cuDoubleComplex>(n), std::vector<cuDoubleComplex>(n)};
  canonical_value->download(canonical_value->resultView(), canonical.value);
  canonical_full->download(canonical_full->resultView(), canonical.full, canonical.gradient,
      canonical.laplacian, canonical.ratio);
  if (!equal(forced, canonical) || owner.counters().value_epoch_replays != 1 ||
      owner.counters().full_epoch_replays != 1)
    throw std::runtime_error("Explicit paired epoch replay differs from complete canonical FP64 owners");
  // Paired JVP is the declared score-induced operator, not a rounded forward-directional network.
  using C = std::complex<double>;
  auto score = PreparedDeviceComplexScore::prepareMixed(next, queue, blas,
      estimateMixedComplexScoreCost(plan).device_bytes);
  score->evaluate(positions);
  std::vector<cuDoubleComplex> canonical_score(plan.parameterCount());
  ComplexValueResult score_amplitude;
  score->download(score->resultView(), canonical_score, score_amplitude);
  const auto chunk = std::min<std::size_t>(257, plan.parameterCount());
  const auto product_cost = estimateMixedComplexProductsCost(plan, n, 1, 1, chunk);
  auto products = PreparedDeviceComplexProducts::prepareMixed(next, n, 1, 1, chunk, queue, blas,
      product_cost.device_bytes, product_cost.host_bytes, 1);
  std::vector<C> direction(plan.parameterCount());
  C expected_dot{};
  for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
  {
    direction[parameter] = .003 * std::sin(double(parameter + 1));
    expected_dot += C(cuCreal(canonical_score[parameter]), cuCimag(canonical_score[parameter])) * direction[parameter];
  }
  products->evaluateScoreJVP(positions, direction.data(), direction.size());
  const auto jvp = products->scoreJvpView();
  products->validate(jvp);
  const auto dot_error = std::abs(jvp.data[0] - expected_dot);
  if (dot_error > 2e-5 * (1 + std::abs(expected_dot)))
    throw std::runtime_error("Paired mixed score-induced JVP differs from independent CPU score dot");
  const std::vector<double> masses(n, 1.0);
  const C coefficient{.625, -.375};
  const std::vector<PreparedDeviceComplexProducts::Channel> channels{
      {PreparedDeviceComplexProducts::Role::SCORE, &coefficient, 1}};
  double adjoint_error = 0;
  for (const bool hermitian : {false, true})
  {
    products->evaluate(positions, masses, {0, 0, 0}, channels, hermitian);
    const auto result = products->resultView();
    products->validate(result);
    C contracted{};
    for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
    {
      const C o{cuCreal(canonical_score[parameter]), cuCimag(canonical_score[parameter])};
      const auto expected = coefficient * (hermitian ? std::conj(o) : o);
      const auto error = std::abs(result.data[parameter] - expected);
      adjoint_error = std::max(adjoint_error, error);
      if (error > 2e-5 * (1 + std::abs(expected)))
        throw std::runtime_error("Paired mixed transpose/Hermitian response differs from CPU score oracle");
      contracted += result.data[parameter] * direction[parameter];
    }
    const auto expected = coefficient * (hermitian ? std::conj(expected_dot) : expected_dot);
    if (std::abs(contracted - expected) > 2e-5 * (1 + std::abs(expected)))
      throw std::runtime_error("Paired mixed score-induced JVP/adjoint contraction is inconsistent");
  }
  std::cout << "PASS paired mixed score-induced recipe=score_induced_fp64_dot_v1"
            << " direct_forward_directional=0 dot_abs_error=" << dot_error
            << " transpose_hermitian_max_abs_error=" << adjoint_error << '\n';
  // The mixed producer and its FP64 fallback preflight together before a warm update.
  auto update_values = values;
  update_values[embedding.begin] += .0625;
  auto update_image = successor->makeUpdatedImage(update_values, successor->version() + 1);
  auto update_cache = std::make_shared<PreparedDeviceMixedDenseParameters>(update_image, queue,
      MixedTrunkHeadPolicy::PAIRED_COMPLEX_FP64_HEAD, true);
  const auto fallback_cost = estimateComplexProductsCost(plan, n, 1, 1, chunk,
      ComplexProductsHostOutput::RETAINED, ComplexProductsScope::SCORE_ECP_ONLY);
  auto fallback = PreparedDeviceComplexProducts::prepare(successor, n, 1, 1, chunk, queue, blas,
      fallback_cost.device_bytes, fallback_cost.host_bytes, 1,
      ComplexProductsHostOutput::RETAINED, ComplexProductsScope::SCORE_ECP_ONLY);
  products->validateScoreModelRebind(update_image, update_cache);
  fallback->validateScoreModelRebind(update_image);
  products->adoptPreparedScoreModel(update_image, update_cache);
  fallback->adoptPreparedScoreModel(update_image);
  auto fresh_products = PreparedDeviceComplexProducts::prepareMixed(update_cache, n, 1, 1, chunk, queue, blas,
      product_cost.device_bytes, product_cost.host_bytes, 1);
  products->evaluate(positions, masses, {0, 0, 0}, channels);
  fresh_products->evaluate(positions, masses, {0, 0, 0}, channels);
  if (std::memcmp(products->resultView().data, fresh_products->resultView().data,
      direction.size() * sizeof(C)) != 0)
    throw std::runtime_error("Mixed paired warm product image differs from a fresh selected-cache owner");
  fallback->evaluate(positions, masses, {0, 0, 0}, channels);
  auto fresh_fallback = PreparedDeviceComplexProducts::prepare(update_image, n, 1, 1, chunk, queue, blas,
      fallback_cost.device_bytes, fallback_cost.host_bytes, 1,
      ComplexProductsHostOutput::RETAINED, ComplexProductsScope::SCORE_ECP_ONLY);
  fresh_fallback->evaluate(positions, masses, {0, 0, 0}, channels);
  if (std::memcmp(fallback->resultView().data, fresh_fallback->resultView().data,
      direction.size() * sizeof(C)) != 0)
    throw std::runtime_error("FP64 paired fallback warm image differs from a fresh canonical owner");
  // Strict training must propagate conversion rejection before either leaf can replay locally.
  auto strict_values = values;
  strict_values[embedding.begin] = 1e-300;
  auto strict_image = successor->makeUpdatedImage(strict_values, successor->version() + 1);
  auto strict_cache = std::make_shared<PreparedDeviceMixedDenseParameters>(strict_image, queue,
      MixedTrunkHeadPolicy::PAIRED_COMPLEX_FP64_HEAD, true);
  if (strict_cache->usable()) throw std::runtime_error("Strict replay fixture did not realize conversion rejection");
  Owner strict_owner(strict_image, 1, queue, blas, borrowed.device_bytes, 0, strict_cache);
  rejected = false;
  try { strict_owner.evaluateValue(positions, PsiFormerGpuTrainingEpochMode::STRICT_MIXED); }
  catch (const MixedPrecisionRetryRequired&) { rejected = true; }
  if (!rejected || strict_owner.counters().value_requests || strict_owner.counters().value_replays)
    throw std::runtime_error("Strict VALUE silently replayed or published a failed mixed request");
  rejected = false;
  try { strict_owner.evaluateFull(positions, PsiFormerGpuTrainingEpochMode::STRICT_MIXED); }
  catch (const MixedPrecisionRetryRequired&) { rejected = true; }
  if (!rejected || strict_owner.counters().full_requests || strict_owner.counters().full_replays)
    throw std::runtime_error("Strict FULL silently replayed or published a failed mixed request");
  strict_owner.evaluateValue(positions, PsiFormerGpuTrainingEpochMode::FP64_REPLAY);
  strict_owner.evaluateFull(positions, PsiFormerGpuTrainingEpochMode::FP64_REPLAY);
  (void)fetch(strict_owner);
  std::cout << "PASS paired mixed native owners: shared_cache_bytes=" << cache_bytes
            << " owned_live_bytes=" << owned.device_bytes << " borrowed_live_bytes=" << borrowed.device_bytes
            << " pending_cache_bytes=" << owned.pending_device_bytes
            << " rejected_publication_bits=exact changed_weight_fresh_bits=exact"
            << " stale_cache_last_owner_charge=exact forced_epoch_canonical_bits=exact"
            << " warm_mixed_and_FP64_products_fresh_bits=exact"
            << " strict_conversion_rejection_no_local_replay=1 native_admission=unqualified\n";
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
    if (up == 2 && concurrent_targets == 0)
      validateMixedComplexNativeOwners(image,
          std::vector<double>(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 12), values, queue, blas);
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
