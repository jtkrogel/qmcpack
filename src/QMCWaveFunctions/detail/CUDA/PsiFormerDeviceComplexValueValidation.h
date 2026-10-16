//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_VALUE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_VALUE_VALIDATION_H
#include "PsiFormerDeviceComplexValue.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexValueExecutor.h"
#include <cstring>

/// Numerical-only complete VALUE gate; fixture builders are the periodic validator's.
void validateComplexValue()
{
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
