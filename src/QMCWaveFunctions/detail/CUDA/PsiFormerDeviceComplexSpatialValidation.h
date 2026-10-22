//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_VALIDATION_H
#include "PsiFormerDeviceComplexSpatial.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexSpatialExecutor.h"

/// A matrix with more entries than block threads constrains product-stride tails.
void validateComplexSpatialProductTails(qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA>& queue)
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  using C = std::complex<double>;
  constexpr std::size_t n = 17, channels = 2, nn = n * n, m = channels * nn;
  auto layout = makeOpenOrbitalJetLayout(1, channels, n, 1, 1, n / 2, SpatialJetMode::FULL_VGL);
  layout.features = withSpatialTargetRange(layout.features, n - 1, 1);
  layout.orbitals = withSpatialTargetRange(layout.orbitals, n - 1, 1);
  std::vector<C> matrices(m), inverses(m), jets(5 * m), expected_g(3), expected_l(1), expected_r(1);
  for (std::size_t channel = 0; channel < channels; ++channel)
    for (std::size_t row = 0; row < n; ++row)
      matrices[channel * nn + row * n + row] = inverses[channel * nn + row * n + row] = 1;
  matrices[nn] = C(0, 1);
  inverses[nn] = C(0, -1);
  std::copy(matrices.begin(), matrices.end(), jets.begin());
  for (std::size_t entry = m; entry < jets.size(); ++entry)
    jets[entry] = C(.03 * std::sin(double(entry + 1)), .02 * std::cos(double(2 * entry + 3)));
  // Ordinary signed products must differ from a conjugated square norm.
  jets[m + 1] = C(2, 1);
  jets[m + n] = C(-3, .5);
  determinant::ComplexDeterminantWorkspace oracle(channels, n, 0, 3, 1);
  oracle.evaluateSpatial(matrices.data(), jets.data() + m, 3, jets.data() + 4 * m, 1,
      expected_g.data(), expected_l.data(), expected_r.data());
  const std::vector<double> cusp{0, .1, -.2, .3, -.04};
  expected_l[0] += cusp[4];
  expected_r[0] = expected_l[0];
  for (std::size_t mu = 0; mu < 3; ++mu)
  {
    expected_g[mu] += cusp[1 + mu];
    expected_r[0] += expected_g[mu] * expected_g[mu];
  }
  const auto pack = [](C z) { return make_cuDoubleComplex(z.real(), z.imag()); };
  std::vector<cuDoubleComplex> packed_jets(jets.size()), packed_inverse(inverses.size());
  std::transform(jets.begin(), jets.end(), packed_jets.begin(), pack);
  std::transform(inverses.begin(), inverses.end(), packed_inverse.begin(), pack);
  DeviceBuffer<cuDoubleComplex> head(jets.size()), inverse(inverses.size()), product(nn), g(3), l(1), r(1);
  DeviceBuffer<ComplexValueChannel> records(channels);
  DeviceBuffer<ComplexValueResult> amplitude(1);
  DeviceBuffer<double> device_cusp(cusp.size());
  DeviceBuffer<std::uint64_t> status(1);
  const std::vector<ComplexValueChannel> channel_records{{pack(C(1, 0)), 0, 0}, {pack(C(0, 1)), 0, 0}};
  const ComplexValueResult root{pack(C(1, 1) / std::sqrt(2.0)), .5 * std::log(2.0), 1 / std::sqrt(2.0), 0, 0};
  const auto clear = std::numeric_limits<std::uint64_t>::max();
  head.upload(packed_jets.data(), packed_jets.size());
  inverse.upload(packed_inverse.data(), packed_inverse.size());
  records.upload(channel_records.data(), channels);
  amplitude.upload(&root, 1);
  device_cusp.upload(cusp.data(), cusp.size());
  status.upload(&clear, 1);
  checkRuntime(launchComplexSpatialTraces(queue.getNative(), layout, records.data(), amplitude.data(),
      inverse.data(), head.data(), device_cusp.data(), product.data(), g.data(), l.data(), r.data(), status.data()),
      "complex product stride tails");
  queue.sync();
  std::uint64_t result_status;
  status.download(&result_status, 1);
  if (result_status != clear)
    throw std::runtime_error("Complex product-tail fixture rejected");
  std::vector<cuDoubleComplex> actual_g(3), actual_l(1), actual_r(1);
  g.download(actual_g.data(), 3); l.download(actual_l.data(), 1); r.download(actual_r.data(), 1);
  const auto compare = [](cuDoubleComplex actual, C expected, const char* field) {
    near(std::abs(C(cuCreal(actual), cuCimag(actual)) - expected), 0, field, 2e-11);
  };
  for (std::size_t mu = 0; mu < 3; ++mu)
    compare(actual_g[mu], expected_g[mu], "complex product-tail gradient");
  compare(actual_l[0], expected_l[0], "complex product-tail log Laplacian");
  compare(actual_r[0], expected_r[0], "complex product-tail ratio Laplacian");
  std::cout << "PASS complex spatial product tails: N17 K1 signed_nonsymmetric=1 complex_weights=1 cusp=1\n";
}

/// Tiny normalized weights must retain finite representable spatial terms, never overflow admission.
void validateComplexSpatialWeightedLeaves(qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA>& queue)
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  using C = std::complex<double>;
  const auto pack = [](C z) { return make_cuDoubleComplex(z.real(), z.imag()); };
  const auto host = [](cuDoubleComplex z) { return C(cuCreal(z), cuCimag(z)); };
  DeviceBuffer<cuDoubleComplex> matrix(8), inverse(8), solve(4), head(72), product(4), g(6), l(2), r(2);
  DeviceBuffer<std::size_t> permutation(4);
  DeviceBuffer<ComplexValueChannel> records(2);
  DeviceBuffer<ComplexValueResult> amplitude(1);
  DeviceBuffer<std::uint64_t> status(1);
  DeviceBuffer<double> cusp(9);
  const std::vector<double> zero_cusp(9, 0);
  cusp.upload(zero_cusp.data(), 9);
  const auto clear = std::numeric_limits<std::uint64_t>::max();
  const ComplexValueResult regular{make_cuDoubleComplex(1, 0), 0, 1, 0, 0};
  double max_scaled = 0;
  const auto launch = [&] {
    status.upload(&clear, 1);
    checkRuntime(launchComplexSpatialTraces(queue.getNative(), 1, 2, 2, records.data(), amplitude.data(),
        inverse.data(), head.data(), cusp.data(), product.data(), g.data(), l.data(), r.data(), status.data()),
        "finite weighted spatial leaf");
    queue.sync();
    std::uint64_t actual;
    status.download(&actual, 1);
    return actual == clear;
  };
  for (bool phase : {false, true})
  {
    const C a = std::polar(1e-200, phase ? 0.3 : 0), b = std::polar(1e-200, phase ? -0.7 : 0);
    const std::vector<C> matrices{1, 0, 0, 1, a, 0, 0, b};
    std::vector<C> jets(72), expected_g(6), expected_l(2), expected_r(2);
    std::copy(matrices.begin(), matrices.end(), jets.begin());
    jets[8 + 4] = C(1e-50, phase ? 0.2e-50 : 0);
    jets[7 * 8 + 4] = C(0.4e-50, phase ? -0.1e-50 : 0);
    determinant::ComplexDeterminantWorkspace oracle(2, 2, 0, 6, 2);
    oracle.evaluateSpatial(matrices.data(), jets.data() + 8, 6, jets.data() + 56, 2,
        expected_g.data(), expected_l.data(), expected_r.data());
    std::vector<cuDoubleComplex> packed(72), raw_inverse(8), after_inverse(8);
    std::transform(jets.begin(), jets.end(), packed.begin(), pack);
    matrix.upload(packed.data(), 8);
    head.upload(packed.data(), 72);
    amplitude.upload(&regular, 1);
    status.upload(&clear, 1);
    checkRuntime(launchComplexValueFactor(queue.getNative(), 2, 2, matrix.data(), records.data(), status.data(),
        inverse.data(), permutation.data(), solve.data()), "finite weighted spatial inverse");
    queue.sync();
    std::uint64_t factor_status;
    status.download(&factor_status, 1);
    if (factor_status != clear)
      throw std::runtime_error("Weighted spatial leaf inverse rejected");
    inverse.download(raw_inverse.data(), 8);
    std::vector<ComplexValueChannel> channels(2);
    records.download(channels.data(), 2);
    if (std::exp(channels[1].log_abs - channels[0].log_abs) != 0 || !launch())
      throw std::runtime_error("Weighted spatial fixture did not exercise an admitted underflowed weight");
    std::vector<cuDoubleComplex> actual_g(6), actual_l(2), actual_r(2);
    g.download(actual_g.data(), 6); l.download(actual_l.data(), 2); r.download(actual_r.data(), 2);
    inverse.download(after_inverse.data(), 8);
    if (std::memcmp(raw_inverse.data(), after_inverse.data(), 8 * sizeof(cuDoubleComplex)))
      throw std::runtime_error("Stable spatial weighting consumed raw inverse storage");
    const auto compare = [&](C actual, C expected) {
      const auto error = std::abs(actual / 1e-250 - expected / 1e-250);
      max_scaled = std::max(max_scaled, error);
      near(error, 0, "finite weighted spatial scaled CPU Wide", 2e-11);
      if (expected != C{} && actual == C{})
        throw std::runtime_error("Representable weighted spatial term was silently zeroed");
    };
    for (std::size_t i = 0; i < 6; ++i) compare(host(actual_g[i]), expected_g[i]);
    for (std::size_t i = 0; i < 2; ++i)
    {
      compare(host(actual_l[i]), expected_l[i]);
      compare(host(actual_r[i]), expected_r[i]);
    }
    // A tiny weight must NOT rescue an overflowing unweighted trace.
    packed[8 + 4] = make_cuDoubleComplex(1e200, 0);
    head.upload(packed.data(), 72);
    if (launch())
      throw std::runtime_error("Tiny weight admitted an overflowing unweighted spatial term");
    packed[8 + 4] = pack(jets[8 + 4]);
    head.upload(packed.data(), 72);
    const auto saved = channels;
    channels[0].log_abs = channels[1].log_abs = 0;
    channels[0].phase = make_cuDoubleComplex(1, 0);
    channels[1].phase = make_cuDoubleComplex(-1, 0);
    records.upload(channels.data(), 2);
    if (launch())
      throw std::runtime_error("Zero spatial denominator admitted");
    records.upload(saved.data(), 2);
    auto node = regular;
    node.phase = make_cuDoubleComplex(0, 0);
    node.log_abs = -std::numeric_limits<double>::infinity();
    amplitude.upload(&node, 1);
    if (launch())
      throw std::runtime_error("Expected amplitude node admitted spatial products");
  }
  std::cout << "PASS finite weighted complex spatial: tiny_G_L_nonzero=1 max_scaled_error=" << max_scaled
      << " complex_phase=1 raw_inverse_unchanged=1 unweighted_overflow_rejected=1 node_denominator_rejected=1\n";
  validateComplexSpatialProductTails(queue);
}

/// Numerical FULL oracle, ordinary complex kinetic consumer and lifecycle gate.
void validateComplexSpatial(bool selected_two_cell = false)
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
  selectPairedTwoCellValidation(environment, selected_two_cell);
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
  pf::DirectComplexSpatialExecutor oracle(model, plan);
  auto workspace = oracle.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  pf::DirectComplexValueExecutor values(model, plan);
  auto value_workspace = values.makeWorkspace();
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  validateComplexSpatialWeightedLeaves(queue);
  AcceleratorBlasHandle blas(queue);
  auto image = selected_two_cell
      ? PreparedDeviceModel::preparePairedTwoCellProfile(plan, parameters, model.cfg.nuclei.x)
      : PreparedDeviceModel::preparePairedComplexValue(plan, parameters, model.cfg.nuclei.x);
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "complex FULL device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "complex FULL device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  const auto cost = estimateComplexSpatialCost(plan, 3);
  const auto before = metrics.snapshot();
  bool rejected = false;
  try { (void)PreparedDeviceComplexSpatial::prepare(image, 3, queue, blas, cost.device_bytes - 1); }
  catch (const std::length_error&) { rejected = true; }
  if (!rejected || metrics.snapshot().allocation_calls != before.allocation_calls)
    throw std::runtime_error("Complex FULL budget admission allocated device storage");
  rejected = false;
  try { (void)estimateComplexSpatialCost(plan, std::numeric_limits<std::size_t>::max()); }
  catch (const std::exception&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Complex FULL overflow extent admitted");
  auto gpu = PreparedDeviceComplexSpatial::prepare(image, 3, queue, blas, cost.device_bytes);
  if (metrics.snapshot().live_bytes - before.live_bytes != cost.device_bytes)
    throw std::runtime_error("Complex FULL exact owner bytes mismatch");
  std::vector<ComplexValueResult> amplitude(3);
  std::vector<cuDoubleComplex> gradient(36), lap_log(12), lap_ratio(12);
  const double masses[4]{1, 1.3, 0.8, 1.7};
  // This separate HOST kinetic-consumer probe uses an external drift, not a model/ParticleSet twist.
  // Complete selected kinetic-owner/products checks retain Gamma kappa=0 below their own admission.
  const std::array<double, 3> k{.13, -.07, .04};
  double max_gradient = 0, max_lap = 0, max_kinetic = 0, max_fd = 0;
  std::size_t comparisons = 0;
  // Compare actual complex components, never branch-cut angles or squared moduli.
  const auto compare = [&](const std::vector<double>& positions) {
    const auto count = positions.size() / 12;
    amplitude.resize(count); gradient.resize(12 * count); lap_log.resize(4 * count); lap_ratio.resize(4 * count);
    gpu->evaluate(positions);
    gpu->download(gpu->resultView(), amplitude, gradient, lap_log, lap_ratio);
    for (std::size_t b = 0; b < count; ++b)
    {
      workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data() + 12 * b, 4));
      const auto expected = oracle.evaluateFull(*workspace);
      near(amplitude[b].log_abs, expected.amplitude.log_abs, "complex FULL log amplitude", 2e-11);
      near(host(amplitude[b].phase).real(), expected.amplitude.phase.real(), "complex FULL phase real", 2e-11);
      near(host(amplitude[b].phase).imag(), expected.amplitude.phase.imag(), "complex FULL phase imag", 2e-11);
      C kinetic{}, reference_kinetic{};
      for (std::size_t e = 0; e < 4; ++e)
      {
        C ordinary_square{}, shifted_square{}, reference_shifted{};
        for (std::size_t mu = 0; mu < 3; ++mu)
        {
          const auto g = host(gradient[12 * b + 3 * e + mu]), expected_g = expected.gradient[3 * e + mu];
          max_gradient = std::max(max_gradient, std::abs(g - expected_g));
          near(g.real(), expected_g.real(), "complex FULL G real", 2e-11);
          near(g.imag(), expected_g.imag(), "complex FULL G imaginary", 2e-11);
          ordinary_square += g * g;
          shifted_square += (g + C(0, k[mu])) * (g + C(0, k[mu]));
          reference_shifted += (expected_g + C(0, k[mu])) * (expected_g + C(0, k[mu]));
        }
        const auto l = host(lap_log[4 * b + e]), r = host(lap_ratio[4 * b + e]);
        max_lap = std::max({max_lap, std::abs(l - expected.laplacian_log[e]), std::abs(r - expected.laplacian_ratio[e])});
        near(l.real(), expected.laplacian_log[e].real(), "complex FULL Llog real", 2e-11);
        near(l.imag(), expected.laplacian_log[e].imag(), "complex FULL Llog imaginary", 2e-11);
        near(r.real(), expected.laplacian_ratio[e].real(), "complex FULL ratio real", 2e-11);
        near(r.imag(), expected.laplacian_ratio[e].imag(), "complex FULL ratio imaginary", 2e-11);
        near(std::abs(r - l - ordinary_square), 0, "ordinary complex ratio identity", 2e-11);
        kinetic -= 0.5 / masses[e] * (l + shifted_square);
        reference_kinetic -= 0.5 / masses[e] * (expected.laplacian_log[e] + reference_shifted);
      }
      max_kinetic = std::max(max_kinetic, std::abs(kinetic - reference_kinetic));
      near(kinetic.real(), reference_kinetic.real(), "complex unequal-mass twisted kinetic real", 2e-11);
      near(kinetic.imag(), reference_kinetic.imag(), "complex unequal-mass twisted kinetic imaginary", 2e-11);
      if (std::abs(reference_kinetic.imag()) < 1e-8)
        throw std::runtime_error("Complex kinetic fixture lost its phase cross terms");
      ++comparisons;
    }
  };
  compare(model.cfg.electrons.x);
  near(amplitude[2].log_abs, amplitude[0].log_abs, "complex FULL periodic image log", 2e-11);
  for (std::size_t lane = 0; lane < 12; ++lane)
    near(std::abs(host(gradient[lane]) - host(gradient[24 + lane])), 0, "complex FULL periodic image G", 2e-11);
  const auto old = gpu->resultView();
  const auto warm = metrics.snapshot();
  for (std::size_t count : {std::size_t(1), std::size_t(2), std::size_t(3)})
    compare(std::vector<double>(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 12 * count));
  const auto after_warm = metrics.snapshot();
  if (after_warm.allocation_calls != warm.allocation_calls || after_warm.free_calls != warm.free_calls)
    throw std::runtime_error("Complex FULL warmed device allocation/free");
  rejected = false;
  try { gpu->download(old, amplitude, gradient, lap_log, lap_ratio); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Complex FULL expired generation admitted");

  // Independent value-only finite differences visit both physical spin sectors.
  constexpr double h = 1e-5;
  for (std::size_t lane : {std::size_t(1), std::size_t(8)})
  {
    auto position = std::vector<double>(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 12);
    const auto center_phase = host(amplitude[0].phase);
    C sides[2];
    for (std::size_t side = 0; side < 2; ++side)
    {
      position[lane] += side == 0 ? h : -2 * h;
      value_workspace->setPositions(pf::GeometryPositionView::interleaved(position.data(), 4));
      const auto v = values.evaluate(*value_workspace).amplitude;
      sides[side] = v.phase * std::exp(v.log_abs - amplitude[0].log_abs);
    }
    const auto fd = (sides[0] - sides[1]) / (2 * h * center_phase);
    max_fd = std::max(max_fd, std::abs(fd - host(gradient[lane])));
    near(fd.real(), host(gradient[lane]).real(), "complex value finite difference real", 2e-8);
    near(fd.imag(), host(gradient[lane]).imag(), "complex value finite difference imaginary", 2e-8);
  }
  std::size_t version = 0;
  const auto replace = [&](const std::vector<double>& next) {
    model.p.set_flat_values(next);
    image = image->makeUpdatedImage(next, ++version);
    const auto before_rebind = metrics.snapshot();
    gpu->rebindModel(image);
    if (metrics.snapshot().allocation_calls != before_rebind.allocation_calls)
      throw std::runtime_error("Complex FULL successor rebind allocated device storage");
  };
  auto zero = parameters;
  for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
  {
    const auto& imaginary = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
    std::fill(zero.begin() + imaginary.begin, zero.begin() + imaginary.end, 0);
  }
  const auto before_image = gpu->resultView();
  replace(zero);
  rejected = false;
  try { gpu->download(before_image, amplitude, gradient, lap_log, lap_ratio); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Complex FULL successor exposed old-image output");
  compare(model.cfg.electrons.x);
  for (auto z : gradient)
    if (cuCimag(z) != 0)
      throw std::runtime_error("Complex FULL zero-imaginary limit not real");

  // A singular canonical1 channel must reject even when its normalized weight is0.
  auto singular = parameters;
  for (const auto& tensor : plan.parameterTensors())
    if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT)
      for (std::size_t f = 0; f < 16; ++f)
        std::fill_n(singular.data() + tensor.begin + f * 16 + 12, 4, 0);
  replace(singular);
  value_workspace->setPositions(pf::GeometryPositionView::interleaved(model.cfg.electrons.x.data(), 4));
  const auto regular_value = values.evaluate(*value_workspace);
  if (regular_value.amplitude.isZero() || regular_value.singular_channels != 1)
    throw std::runtime_error("Complex FULL singular fixture lacks its regular total VALUE");
  const auto reject_population = [&](const std::vector<double>& positions) {
    bool failed = false, hidden = false;
    try { gpu->evaluate(positions); }
    catch (const std::runtime_error&) { failed = true; }
    try { (void)gpu->resultView(); }
    catch (const std::logic_error&) { hidden = true; }
    if (!failed || !hidden)
      throw std::runtime_error("Complex FULL rejected request exposed partial/stale output");
  };
  reject_population(model.cfg.electrons.x);
  auto cancel = parameters;
  for (const auto& tensor : plan.parameterTensors())
    if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT)
      for (std::size_t f = 0; f < 16; ++f)
        for (std::size_t d = 1; d < 4; ++d)
        {
          std::copy_n(cancel.data() + tensor.begin + f * 16, 4, cancel.data() + tensor.begin + f * 16 + 4 * d);
          if (d % 2)
            cancel[tensor.begin + f * 16 + 4 * d] *= -1;
        }
    else if (tensor.role == ParameterRole::ENVELOPE_PI_UP || tensor.role == ParameterRole::ENVELOPE_PI_DOWN ||
             tensor.role == ParameterRole::ENVELOPE_ZETA_UP || tensor.role == ParameterRole::ENVELOPE_ZETA_DOWN)
      for (std::size_t d = 1; d < 4; ++d)
        std::copy_n(cancel.data() + tensor.begin, 8, cancel.data() + tensor.begin + 8 * d);
  replace(cancel);
  value_workspace->setPositions(pf::GeometryPositionView::interleaved(model.cfg.electrons.x.data(), 4));
  const auto node = values.evaluate(*value_workspace);
  if (!node.amplitude.isZero() || node.singular_channels != 0)
    throw std::runtime_error("Complex FULL cancellation fixture must have invertible channels");
  reject_population(model.cfg.electrons.x);
  replace(parameters);
  compare(model.cfg.electrons.x);
  auto invalid = model.cfg.electrons.x;
  invalid.back() = std::numeric_limits<double>::quiet_NaN();
  reject_population(invalid);
  auto coalescent = model.cfg.electrons.x;
  std::copy_n(model.cfg.nuclei.x.data(), 3, coalescent.data());
  reject_population(coalescent);
  compare(model.cfg.electrons.x);
  std::cout << "PASS numerical complex GPU FULL: comparisons=" << comparisons << " max_G_error=" << max_gradient
            << " max_L_error=" << max_lap << " max_kinetic_error=" << max_kinetic << " max_FD_error=" << max_fd
            << " trunk_bytes=" << cost.trunk_bytes << " owner_device_bytes=" << cost.device_bytes
            << " owner_host_bytes=" << cost.host_bytes
            << " warm_device_allocations=0 scope=device_wide outer_k=test_consumer_only"
            << " external_drift=HOST_math_probe native_outer_twist_admission=not_implied\n";
}
#endif
