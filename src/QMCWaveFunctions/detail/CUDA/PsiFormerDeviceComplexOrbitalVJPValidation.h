//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_ORBITAL_VJP_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_ORBITAL_VJP_VALIDATION_H
#include "PsiFormerDeviceComplexScore.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexScoreExecutor.h"

namespace qmcplusplus::psiformer::device
{
struct ComplexOrbitalVJPValidationAccess
{
  static bool noInverse(const PreparedDeviceComplexScore& owner)
  { return owner.hasNoOrbitalInverseStorage(); }

  /// Fail the actual existing trunk reverse after successful finite head seeding.
  static void failReverse(PreparedDeviceComplexScore& owner,
      const PreparedDeviceComplexScore::OrbitalView& view, const std::vector<cuDoubleComplex>& seeds)
  {
    owner.ready_ = false;
    owner.advanceGeneration();
    owner.validateOrbitalView(view);
    owner.orbital_cotangents_.upload(seeds.data(), seeds.size());
    owner.prepareObjective(0, owner.orbital_cotangents_.data());
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    owner.feature_.upload(&invalid, 1);
    owner.finishObjective(0, false);
  }
};
}

/// Independent CPU raw-MSE all-P and raw-VALUE finite-difference qualification.
void validateComplexOrbitalVJP()
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  using C = std::complex<double>;
  using Normalization = qmcplusplus::wftrain::OrbitalLossNormalization;
  ExecutionEnvironment environment;
  environment.boundary = BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
  environment.lattice_vectors = {{{4.0, 0.4, -0.2}, {0.7, 5.0, 0.3}, {-0.1, 0.8, 6.0}}};
  environment.periodic_axes = {true, true, true};
  environment.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
  environment.amplitude_scalar_domain = ScalarDomain::COMPLEX;
  const ModelShape shape{1, 3, 2, 4, 16, 2, 2};
  auto configuration = configurations(environment);
  configuration.nup = 1;
  configuration.ndown = 3;
  pf::PsiFormer model(smallParameters(environment), std::move(configuration), 4, 16, 2, 2);
  const auto plan = PsiFormerExecutionPlan::fromParameters(model.p, shape, environment);
  auto parameters = model.p.flat_values();
  for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
  {
    const auto& imaginary = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
    for (std::size_t i = imaginary.begin; i < imaginary.end; ++i)
      parameters[i] = .09 * std::sin(double(i - imaginary.begin + 1));
  }
  model.p.set_flat_values(parameters);
  pf::DirectComplexScoreExecutor oracle(model, plan);
  auto cpu = oracle.makeWorkspace();
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, parameters, model.cfg.nuclei.x);
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "raw orbital VJP device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "raw orbital VJP device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  const auto cost = estimateComplexOrbitalVJPCost(plan);
  const auto before = metrics.snapshot();
  bool rejected = false;
  try { (void)PreparedDeviceComplexScore::prepareOrbitalVJP(image, queue, blas, cost.device_bytes - 1); }
  catch (const std::length_error&) { rejected = true; }
  if (!rejected || metrics.snapshot().allocation_calls != before.allocation_calls || cost.canonical_bytes != 8 * plan.parameterCount())
    throw std::runtime_error("Raw orbital VJP preflight/realP admission failed");
  auto gpu = PreparedDeviceComplexScore::prepareOrbitalVJP(image, queue, blas, cost.device_bytes);
  if (metrics.snapshot().live_bytes - before.live_bytes != cost.device_bytes || !ComplexOrbitalVJPValidationAccess::noInverse(*gpu))
    throw std::runtime_error("Raw orbital VJP retained an inverse/complexP or wrong numeric inventory");
  const auto m = cost.forward.matrices, p = plan.parameterCount();
  std::vector<C> targets(m);
  std::vector<cuDoubleComplex> orbitals(m), cotangents(m);
  std::vector<double> response(p), reference(p);
  for (std::size_t i = 0; i < m; ++i)
    targets[i] = {.04 * std::sin(.3 * (i + 1)), .03 * std::cos(.7 * (i + 1))};
  const std::vector<double> base(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 12);
  double maximum = 0, maximum_fd = 0;
  std::size_t comparisons = 0, version = 0;
  const auto compare = [&](const std::vector<double>& positions, Normalization normalization) {
    cpu->setPositions(pf::GeometryPositionView::interleaved(positions.data(), 4));
    const auto expected = oracle.evaluateOrbitalMSE(*cpu, {targets.data(), targets.size(), reference.data(), reference.size(), normalization});
    gpu->evaluateOrbitals(positions);
    const auto view = gpu->orbitalView();
    gpu->downloadOrbitals(view, orbitals);
    for (std::size_t i = 0; i < m; ++i)
    {
      const C actual(cuCreal(orbitals[i]), cuCimag(orbitals[i]));
      near(actual.real(), expected.orbital_matrices[i].real(), "raw orbital real", 2e-11);
      near(actual.imag(), expected.orbital_matrices[i].imag(), "raw orbital imaginary", 2e-11);
      const auto row = (i / 4) % 4;
      const auto denominator = normalization == Normalization::FULL_MATRIX_MEAN ? 4 : (row < 1 ? 1 : 3);
      const auto seed = (2.0 / (4 * denominator * 4)) * std::conj(actual - targets[i]);
      cotangents[i] = make_cuDoubleComplex(seed.real(), seed.imag());
    }
    gpu->reverseOrbitals(view, cotangents);
    gpu->downloadOrbitalVJP(gpu->orbitalVJPView(), response);
    for (std::size_t parameter = 0; parameter < p; ++parameter)
    {
      maximum = std::max(maximum, std::abs(response[parameter] - reference[parameter]));
      near(response[parameter], reference[parameter], "all-canonical raw orbital MSE VJP", 2e-11);
    }
    for (auto role : {ParameterRole::CUSP_SAME_ALPHA, ParameterRole::CUSP_OPPOSITE_ALPHA})
      if (const auto* interval = plan.optionalParameter(role))
        for (std::size_t parameter = interval->begin; parameter < interval->end; ++parameter)
          if (response[parameter] != 0)
            throw std::runtime_error("Raw orbital objective acquired a cusp derivative");
    ++comparisons;
  };
  compare(base, Normalization::SPIN_SECTOR_SUM);
  const auto warm = metrics.snapshot();
  compare(base, Normalization::FULL_MATRIX_MEAN);
  compare(std::vector<double>(model.cfg.electrons.x.begin() + 12, model.cfg.electrons.x.begin() + 24), Normalization::SPIN_SECTOR_SUM);
  compare(base, Normalization::SPIN_SECTOR_SUM);
  if (metrics.snapshot().allocation_calls != warm.allocation_calls || metrics.snapshot().free_calls != warm.free_calls)
    throw std::runtime_error("Warmed raw orbital VJP allocated/freed device storage");

  // Repeated cotangents reuse this exact raw head epoch, independently of any target.
  const auto head = gpu->orbitalView();
  for (std::size_t i = 0; i < m; ++i)
    cotangents[i] = make_cuDoubleComplex(.02 * std::cos(i + 1.), -.03 * std::sin(.4 * (i + 1)));
  gpu->reverseOrbitals(head, cotangents);
  gpu->downloadOrbitalVJP(gpu->orbitalVJPView(), response);
  std::vector<double> direction(p);
  double dot = 0;
  for (std::size_t i = 0; i < p; ++i)
  {
    direction[i] = (int(i % 13) - 6) * .01 / std::sqrt(double(p));
    dot += direction[i] * response[i];
  }
  const auto linear_value = [&](const std::vector<double>& values) {
    model.p.set_flat_values(values);
    cpu->setPositions(pf::GeometryPositionView::interleaved(base.data(), 4));
    const auto raw = oracle.evaluateOrbitalMSE(*cpu, {targets.data(), m, reference.data(), p});
    double result = 0;
    for (std::size_t i = 0; i < m; ++i)
      result += (C(cuCreal(cotangents[i]), cuCimag(cotangents[i])) * raw.orbital_matrices[i]).real();
    return result;
  };
  constexpr double h = 1e-4;
  auto plus = parameters, minus = parameters;
  for (std::size_t i = 0; i < p; ++i) { plus[i] += h * direction[i]; minus[i] -= h * direction[i]; }
  const auto directional = (linear_value(plus) - linear_value(minus)) / (2 * h);
  maximum_fd = std::abs(directional - dot);
  near(directional, dot, "raw orbital signed-direction finite difference", 2e-8);
  for (const auto parameter : {plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).begin,
       plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG).begin,
       plan.parameter(ParameterRole::ENVELOPE_PI_UP).begin, plan.parameter(ParameterRole::ENVELOPE_ZETA_DOWN).begin})
  {
    plus = minus = parameters;
    plus[parameter] += h; minus[parameter] -= h;
    const auto numerical = (linear_value(plus) - linear_value(minus)) / (2 * h);
    maximum_fd = std::max(maximum_fd, std::abs(numerical - response[parameter]));
    near(numerical, response[parameter], "raw orbital selected-parameter finite difference", 2e-8);
  }
  model.p.set_flat_values(parameters);

  auto stale_response = gpu->orbitalVJPView();
  std::vector<cuDoubleComplex> zero(m, make_cuDoubleComplex(0, 0));
  gpu->reverseOrbitals(head, zero);
  gpu->downloadOrbitalVJP(gpu->orbitalVJPView(), response);
  for (const auto value : response) if (value != 0) throw std::runtime_error("Zero raw cotangent produced a gradient");
  rejected = false;
  try { gpu->downloadOrbitalVJP(stale_response, response); } catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) throw std::runtime_error("New cotangents did not revoke the old gradient view");
  auto invalid = cotangents;
  invalid.back() = make_cuDoubleComplex(std::numeric_limits<double>::quiet_NaN(), 0);
  rejected = false;
  try { gpu->reverseOrbitals(head, invalid); } catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) throw std::runtime_error("Raw cotangents admitted a nonfinite component");
  rejected = false;
  try { (void)gpu->orbitalVJPView(); } catch (const std::logic_error&) { rejected = true; }
  if (!rejected) throw std::runtime_error("Failed raw seed exposed a prior P response");
  rejected = false;
  try { ComplexOrbitalVJPValidationAccess::failReverse(*gpu, head, cotangents); } catch (const std::runtime_error&) { rejected = true; }
  if (!rejected) throw std::runtime_error("Injected actual trunk reverse did not fail finite admission");
  rejected = false;
  try { (void)gpu->orbitalVJPView(); } catch (const std::logic_error&) { rejected = true; }
  if (!rejected) throw std::runtime_error("Failed raw reverse exposed a partial P response");
  compare(base, Normalization::SPIN_SECTOR_SUM);
  auto foreign = gpu->orbitalView(); foreign.owner = nullptr;
  rejected = false;
  try { gpu->downloadOrbitals(foreign, orbitals); } catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) throw std::runtime_error("Raw orbital download accepted a foreign owner");
  const auto old_head = gpu->orbitalView();
  gpu->evaluateOrbitals(base);
  rejected = false;
  try { gpu->reverseOrbitals(old_head, cotangents); } catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) throw std::runtime_error("Raw orbital VJP accepted an expired head generation");

  const auto replace = [&](const std::vector<double>& next) {
    model.p.set_flat_values(next);
    image = image->makeUpdatedImage(next, ++version);
    const auto snapshot = metrics.snapshot();
    gpu->rebindModel(image);
    if (metrics.snapshot().allocation_calls != snapshot.allocation_calls)
      throw std::runtime_error("Raw orbital owner allocated during image rebind");
  };
  auto node = parameters;
  for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
    for (auto component : {HeadTensorComponent::REAL, HeadTensorComponent::IMAG})
    {
      const auto& interval = plan.parameter(role, NO_ATTENTION_BLOCK, component);
      std::fill(node.begin() + interval.begin, node.begin() + interval.end, 0);
    }
  replace(node);
  compare(base, Normalization::SPIN_SECTOR_SUM);
  bool nonzero = false;
  for (const auto value : response) nonzero = nonzero || std::abs(value) > 1e-8;
  if (!nonzero) throw std::runtime_error("Zero raw head lost its nonzero supervised head derivative");
  auto zero_envelope = parameters;
  const auto pi = plan.parameter(ParameterRole::ENVELOPE_PI_UP).begin;
  zero_envelope[pi] = zero_envelope[pi + 1] = 0;
  replace(zero_envelope);
  compare(base, Normalization::SPIN_SECTOR_SUM);
  if (std::abs(response[pi]) < 1e-8) throw std::runtime_error("Zero envelope lost its raw pi derivative");
  replace(parameters);
  compare(base, Normalization::SPIN_SECTOR_SUM);
  auto invalid_positions = base; invalid_positions.back() = std::numeric_limits<double>::quiet_NaN();
  rejected = false;
  try { gpu->evaluateOrbitals(invalid_positions); } catch (const std::exception&) { rejected = true; }
  if (!rejected) throw std::runtime_error("Raw orbital forward accepted nonfinite geometry");
  compare(base, Normalization::SPIN_SECTOR_SUM);
  std::cout << "PASS paired raw orbital VJP: comparisons=" << comparisons << " P=" << p
            << " max_all_P_error=" << maximum << " max_raw_VALUE_FD_error=" << maximum_fd
            << " owner_device_bytes=" << cost.device_bytes << " canonical_bytes_per_P=8"
            << " inverse_bytes=0 cusp_gradient=0 warm_device_allocations=0; GPU_XML_pretraining=not_admitted\n";
}
#endif
