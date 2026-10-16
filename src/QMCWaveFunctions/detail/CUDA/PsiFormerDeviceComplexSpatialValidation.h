//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SPATIAL_VALIDATION_H
#include "PsiFormerDeviceComplexSpatial.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexSpatialExecutor.h"

/// Numerical FULL oracle, ordinary complex kinetic consumer and lifecycle gate.
void validateComplexSpatial()
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
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, parameters, model.cfg.nuclei.x);
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
  const double masses[4]{1, 1.3, 0.8, 1.7}, k[3]{0.13, -0.07, 0.04};
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
            << " warm_device_allocations=0 scope=device_wide outer_k=test_consumer_only\n";
}
#endif
