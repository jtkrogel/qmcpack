//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SCORE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SCORE_VALIDATION_H
#include "PsiFormerDeviceComplexScore.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexScoreExecutor.h"

namespace qmcplusplus::psiformer::device
{
/// Test-only access to the SAME production objective phases, never a public fault API.
struct ComplexScoreValidationAccess
{
  /// Complete a real reverse, then make the second actual trunk reverse fail finite admission.
  static void failSecond(PreparedDeviceComplexScore& owner, const std::vector<double>& positions)
  {
    owner.prepareForward(positions);
    const auto generation = owner.forward_->resultView().generation;
    owner.prepareObjective(0);
    owner.finishObjective(0);
    owner.prepareObjective(1);
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    owner.feature_.upload(&invalid, 1);
    bool failed = false;
    try { owner.finishObjective(1); }
    catch (const std::runtime_error&) { failed = true; }
    if (!failed || owner.forward_->resultView().generation != generation)
      throw std::runtime_error("Complex score second-pass failure did not reuse the exact tape");
  }

};
} // namespace qmcplusplus::psiformer::device

/// Nonvacuous tiny weighted inverses and finite-component/overflowing-norm leaf table.
void validateComplexScoreWeightedLeaves(qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA>& queue)
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  using C = std::complex<double>;
  DeviceBuffer<cuDoubleComplex> matrix(8), inverse(8), solve(4);
  DeviceBuffer<std::size_t> permutation(4);
  DeviceBuffer<ComplexValueChannel> records(2);
  DeviceBuffer<ComplexValueResult> amplitude(1);
  DeviceBuffer<std::uint64_t> status(1);
  DeviceBuffer<double> cusp(1);
  const double zero = 0, delta = 1e-200;
  const auto clear = std::numeric_limits<std::uint64_t>::max();
  cusp.upload(&zero, 1);
  const auto admit = [&] {
    queue.sync();
    std::uint64_t result;
    status.download(&result, 1);
    if (result != clear)
      throw std::runtime_error("Complex score weighted-inverse leaf rejected a representable product");
  };
  double max_scaled = 0;
  for (bool complex_phase : {false, true})
  {
    const C a = std::polar(delta, complex_phase ? 0.3 : 0), b = std::polar(delta, complex_phase ? -0.7 : 0);
    std::vector<C> matrices{1, 0, 0, 1, a, 0, 0, b}, expected(8);
    determinant::ComplexDeterminantWorkspace oracle(2, 2);
    oracle.evaluate(matrices.data());
    oracle.fillLogMatrixAdjoints(expected.data());
    std::vector<cuDoubleComplex> packed(8), actual(8);
    for (std::size_t i = 0; i < 8; ++i)
      packed[i] = make_cuDoubleComplex(matrices[i].real(), matrices[i].imag());
    matrix.upload(packed.data(), 8);
    status.upload(&clear, 1);
    const ComplexValueResult admitted{make_cuDoubleComplex(1, 0), 0, 1, 0, 0};
    amplitude.upload(&admitted, 1);
    checkRuntime(launchComplexValueFactor(queue.getNative(), 2, 2, matrix.data(), records.data(), status.data(),
        inverse.data(), permutation.data(), solve.data()), "weighted score leaf inverse");
    admit();
    std::vector<ComplexValueChannel> channels(2);
    records.download(channels.data(), 2);
    if (std::exp(channels[1].log_abs - channels[0].log_abs) != 0)
      throw std::runtime_error("Tiny score fixture did not underflow a prematurely narrowed weight");
    checkRuntime(launchComplexScoreWeightedInverses(queue.getNative(), 2, 2, records.data(), inverse.data(),
        cusp.data(), amplitude.data(), status.data()), "stable weighted score leaf");
    admit();
    inverse.download(actual.data(), 8);
    for (std::size_t channel = 0; channel < 2; ++channel)
      for (std::size_t row = 0; row < 2; ++row)
        for (std::size_t column = 0; column < 2; ++column)
        {
          const auto i = 4 * channel + 2 * row + column;
          const C g(cuCreal(actual[i]), cuCimag(actual[i]));
          const auto e = expected[4 * channel + 2 * column + row];
          const auto scale = channel ? delta : 1;
          max_scaled = std::max(max_scaled, std::abs(g / scale - e / scale));
          near((g / scale).real(), (e / scale).real(), "tiny H/delta real", 2e-11);
          near((g / scale).imag(), (e / scale).imag(), "tiny H/delta imaginary", 2e-11);
          if (channel && row == column && std::abs(g) == 0)
            throw std::runtime_error("Representable tiny matrix seed was silently zeroed");
        }
  }
  // Both finite inverse components fit even though their complex norm overflows.
  const auto maximum = std::numeric_limits<double>::max();
  const auto large = make_cuDoubleComplex(0.9 * maximum, 0.9 * maximum);
  const ComplexValueChannel unit{make_cuDoubleComplex(1, 0), 0, 0};
  const ComplexValueResult admitted{make_cuDoubleComplex(1, 0), 0, 1, 0, 0};
  records.upload(&unit, 1); inverse.upload(&large, 1); amplitude.upload(&admitted, 1); status.upload(&clear, 1);
  checkRuntime(launchComplexScoreWeightedInverses(queue.getNative(), 1, 1, records.data(), inverse.data(), cusp.data(),
      amplitude.data(), status.data()), "finite component weighted score leaf");
  admit();
  cuDoubleComplex result;
  inverse.download(&result, 1);
  near(cuCreal(result) / maximum, 0.9, "finite weighted component real", 2e-11);
  near(cuCimag(result) / maximum, 0.9, "finite weighted component imaginary", 2e-11);
  // Two individually overflowing weighted terms can cancel to finite components.
  const C v0 = std::polar(13.0 / 9, 35.0 * std::acos(-1.0) / 180), v1 = C(1) - v0;
  const ComplexValueChannel cancelling[2]{{make_cuDoubleComplex(v0.real() / std::abs(v0), v0.imag() / std::abs(v0)), std::log(std::abs(v0)), 0},
      {make_cuDoubleComplex(v1.real() / std::abs(v1), v1.imag() / std::abs(v1)), std::log(std::abs(v1)), 0}};
  const cuDoubleComplex entries[2]{make_cuDoubleComplex(0.9 * maximum, (0.2 * 9 / 13) * maximum), make_cuDoubleComplex(1, 0)};
  using Wide = std::complex<long double>;
  const Wide expected = Wide(v0) / (Wide(v0) + Wide(v1)) * Wide(cuCreal(entries[0]), cuCimag(entries[0]));
  records.upload(cancelling, 2); inverse.upload(entries, 2); amplitude.upload(&admitted, 1); status.upload(&clear, 1);
  checkRuntime(launchComplexScoreWeightedInverses(queue.getNative(), 2, 1, records.data(), inverse.data(), cusp.data(),
      amplitude.data(), status.data()), "cancelling weighted product leaf");
  admit();
  inverse.download(&result, 1);
  near(cuCreal(result) / maximum, double(expected.real() / maximum), "finite cancelled weighted component real", 2e-11);
  near(cuCimag(result) / maximum, double(expected.imag() / maximum), "finite cancelled weighted component imaginary", 2e-11);
  std::cout << "PASS stable complex weighted inverse: tiny_H_nonzero=1 max_H_over_delta_error=" << max_scaled
            << " complex_phase=1 finite_components_overflowing_norm=1 cancelling_weighted_products=1\n";
}

/// Complete all-canonical CPU/Wide/FD parity and atomic two-objective publication.
void validateComplexScore(bool selected_two_cell = false)
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
  pf::DirectComplexScoreExecutor oracle(model, plan);
  auto workspace = oracle.makeWorkspace();
  pf::DirectComplexValueExecutor value(model, plan);
  auto value_workspace = value.makeWorkspace();
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  validateComplexScoreWeightedLeaves(queue);
  auto image = selected_two_cell
      ? PreparedDeviceModel::preparePairedTwoCellProfile(plan, parameters, model.cfg.nuclei.x)
      : PreparedDeviceModel::preparePairedComplexValue(plan, parameters, model.cfg.nuclei.x);
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "complex score device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "complex score device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  const auto cost = estimateComplexScoreCost(plan);
  const auto before = metrics.snapshot();
  bool rejected = false;
  try { (void)PreparedDeviceComplexScore::prepare(image, queue, blas, cost.device_bytes - 1); }
  catch (const std::length_error&) { rejected = true; }
  if (!rejected || metrics.snapshot().allocation_calls != before.allocation_calls || cost.canonical_bytes != 24 * plan.parameterCount())
    throw std::runtime_error("Complex score budget/24P admission failed");
  auto gpu = PreparedDeviceComplexScore::prepare(image, queue, blas, cost.device_bytes);
  if (metrics.snapshot().live_bytes - before.live_bytes != cost.device_bytes)
    throw std::runtime_error("Complex score actual owner bytes differ from exact preflight");
  std::vector<cuDoubleComplex> response(plan.parameterCount());
  ComplexValueResult amplitude;
  double max_error = 0, max_fd = 0;
  std::size_t comparisons = 0, version = 0;
  const auto base = std::vector<double>(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 12);
  const auto compare = [&](const std::vector<double>& positions) {
    gpu->evaluate(positions);
    gpu->download(gpu->resultView(), response, amplitude);
    workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data(), 4));
    const auto expected = oracle.evaluate(*workspace);
    near(amplitude.log_abs, expected.amplitude.log_abs, "complex score log amplitude", 2e-11);
    near(cuCreal(amplitude.phase), expected.amplitude.phase.real(), "complex score phase real", 2e-11);
    near(cuCimag(amplitude.phase), expected.amplitude.phase.imag(), "complex score phase imaginary", 2e-11);
    for (std::size_t p = 0; p < response.size(); ++p)
    {
      const auto actual = host(response[p]);
      max_error = std::max(max_error, std::abs(actual - expected.parameter_score[p]));
      near(actual.real(), expected.parameter_score[p].real(), "complex canonical score real", 2e-11);
      near(actual.imag(), expected.parameter_score[p].imag(), "complex canonical score imaginary", 2e-11);
    }
    for (auto role : {ParameterRole::CUSP_SAME_ALPHA, ParameterRole::CUSP_OPPOSITE_ALPHA})
      if (const auto* interval = plan.optionalParameter(role))
        if (cuCimag(response[interval->begin]) != 0)
          throw std::runtime_error("Common real cusp contaminated the imaginary score");
    ++comparisons;
  };
  const auto replace = [&](const std::vector<double>& next) {
    model.p.set_flat_values(next);
    image = image->makeUpdatedImage(next, ++version);
    const auto memory = metrics.snapshot();
    gpu->rebindModel(image);
    if (metrics.snapshot().allocation_calls != memory.allocation_calls)
      throw std::runtime_error("Complex score image rebind allocated device storage");
  };
  compare(base);
  const auto stale = gpu->resultView();
  const auto warm = metrics.snapshot();
  for (std::size_t sample = 0; sample < 3; ++sample)
    compare(std::vector<double>(model.cfg.electrons.x.begin() + 12 * sample, model.cfg.electrons.x.begin() + 12 * (sample + 1)));
  if (metrics.snapshot().allocation_calls != warm.allocation_calls || metrics.snapshot().free_calls != warm.free_calls)
    throw std::runtime_error("Complex score warmed DEVICE allocation/free");
  rejected = false;
  try { gpu->download(stale, response, amplitude); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Complex score accepted an expired generation");
  compare(base);
  std::vector<double> direction(response.size());
  C dot{};
  for (std::size_t p = 0; p < direction.size(); ++p)
  {
    direction[p] = (int(p % 13) - 6) * 0.01 / std::sqrt(double(direction.size()));
    dot += host(response[p]) * direction[p];
  }
  constexpr double h = 1e-4;
  C sides[2];
  for (std::size_t side = 0; side < 2; ++side)
  {
    auto shifted = parameters;
    for (std::size_t p = 0; p < direction.size(); ++p)
      shifted[p] += (side ? -h : h) * direction[p];
    model.p.set_flat_values(shifted);
    value_workspace->setPositions(pf::GeometryPositionView::interleaved(base.data(), 4));
    const auto a = value.evaluate(*value_workspace).amplitude;
    sides[side] = a.phase * std::exp(a.log_abs - amplitude.log_abs);
  }
  model.p.set_flat_values(parameters);
  const auto fd = (sides[0] - sides[1]) / (2 * h * host(amplitude.phase));
  max_fd = std::abs(fd - dot);
  near(fd.real(), dot.real(), "complex canonical signed-direction VALUE FD real", 2e-8);
  near(fd.imag(), dot.imag(), "complex canonical signed-direction VALUE FD imaginary", 2e-8);
  const auto finite_difference = [&](std::size_t parameter) {
    const auto center = model.p.flat_values();
    C shifted_values[2];
    for (std::size_t side = 0; side < 2; ++side)
    {
      auto shifted = center;
      shifted[parameter] += side ? -h : h;
      model.p.set_flat_values(shifted);
      value_workspace->setPositions(pf::GeometryPositionView::interleaved(base.data(), 4));
      const auto a = value.evaluate(*value_workspace).amplitude;
      shifted_values[side] = a.phase * std::exp(a.log_abs - amplitude.log_abs);
    }
    model.p.set_flat_values(center);
    const auto numerical = (shifted_values[0] - shifted_values[1]) / (2 * h * host(amplitude.phase));
    max_fd = std::max(max_fd, std::abs(numerical - host(response[parameter])));
    near(numerical.real(), cuCreal(response[parameter]), "selected canonical VALUE FD real", 2e-8);
    near(numerical.imag(), cuCimag(response[parameter]), "selected canonical VALUE FD imaginary", 2e-8);
  };
  finite_difference(plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).begin);
  finite_difference(plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG).begin);
  finite_difference(plan.parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin);
  auto zero = parameters;
  for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
  {
    const auto& interval = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
    std::fill(zero.begin() + interval.begin, zero.begin() + interval.end, 0);
  }
  const auto old_image = gpu->resultView();
  replace(zero);
  rejected = false;
  try { gpu->download(old_image, response, amplitude); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Complex score exposed an old-image response");
  compare(base);
  bool imaginary_head_nonzero = false;
  for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
  {
    const auto& tensor = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
    for (std::size_t p = tensor.begin; p < tensor.end; ++p)
    {
      near(cuCreal(response[p]), 0, "zero-Wi imaginary-head real part", 2e-11);
      imaginary_head_nonzero = imaginary_head_nonzero || std::abs(cuCimag(response[p])) > 1e-8;
    }
  }
  if (!imaginary_head_nonzero)
    throw std::runtime_error("Zero imaginary weights erased their phase parameter derivatives");
  auto zero_envelope = parameters;
  const auto pi = plan.parameter(ParameterRole::ENVELOPE_PI_UP).begin;
  zero_envelope[pi] = zero_envelope[pi + 1] = 0;
  replace(zero_envelope);
  compare(base);
  if (std::abs(host(response[pi])) < 1e-8)
    throw std::runtime_error("True zero envelope lost its nonzero canonical parameter score");
  finite_difference(pi);
  replace(parameters);
  compare(base);
  ComplexScoreValidationAccess::failSecond(*gpu, base);
  bool hidden = false;
  try { (void)gpu->resultView(); }
  catch (const std::logic_error&) { hidden = true; }
  if (!hidden)
    throw std::runtime_error("Second objective failure exposed the completed first objective");
  compare(base);
  auto invalid = base;
  invalid.back() = std::numeric_limits<double>::quiet_NaN();
  rejected = false;
  try { gpu->evaluate(invalid); }
  catch (const std::runtime_error&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Complex score admitted a nonfinite coordinate");
  auto singular = parameters;
  for (const auto& tensor : plan.parameterTensors())
    if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT)
      for (std::size_t f = 0; f < 16; ++f)
        std::fill_n(singular.data() + tensor.begin + f * 16 + 12, 4, 0);
  replace(singular);
  rejected = false;
  try { gpu->evaluate(base); }
  catch (const std::runtime_error&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Canonical1 singular channel was ignored at zero weight");
  replace(parameters);
  auto node = base;
  std::copy_n(node.data(), 3, node.data() + 3);
  rejected = false;
  try { gpu->evaluate(node); }
  catch (const std::runtime_error&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Complex score admitted a same-spin node");
  compare(base);
  auto coalescent = base;
  std::copy_n(model.cfg.nuclei.x.data(), 3, coalescent.data());
  compare(coalescent);
  std::cout << "PASS numerical complex GPU score: comparisons=" << comparisons << " P=" << response.size()
            << " max_score_error=" << max_error << " max_VALUE_FD_error=" << max_fd
            << " owner_device_bytes=" << cost.device_bytes << " owner_host_bytes=" << cost.host_bytes
            << " canonical_bytes=" << cost.canonical_bytes << " canonical_bytes_per_P=24"
            << " warm_device_allocations=0 scope=device_wide two_objectives_one_tape=1\n";
}
#endif
