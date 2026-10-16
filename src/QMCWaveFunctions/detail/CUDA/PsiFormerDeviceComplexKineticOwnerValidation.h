//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_KINETIC_OWNER_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_KINETIC_OWNER_VALIDATION_H
#include "PsiFormerDeviceComplexKinetic.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexKineticExecutor.h"

namespace qmcplusplus::psiformer::device
{
/// Focused private-phase adapter; no production fault flags/callbacks/data exposure.
struct ComplexKineticOwnerValidationAccess
{
  /// Fail the actual final-tile imaginary reverse AFTER its successful real reverse.
  static void failLate(PreparedDeviceComplexKinetic& owner, const std::vector<double>& positions,
      const std::vector<double>& masses, const std::array<double, 3>& kappa)
  {
    owner.beginRequest(positions, masses, kappa);
    std::size_t real_passes = 0, imaginary_passes = 0;
    bool injected = false, rejected = false;
    try
    {
      owner.trunk_->withTargets(positions, [&](const auto& ticket) {
        const auto generation = ticket.generation, ordinal = ticket.ordinal;
        const bool last = ticket.features.target_electron_begin + ticket.features.laplacian_lanes == masses.size();
        for (std::size_t component = 0; component < 2; ++component)
        {
          owner.trunk_->reverseSeeded(ticket, [&](const auto& current, double* feature, double* cusp, double* response) {
            if (current.generation != generation || current.ordinal != ordinal)
              throw std::runtime_error("Complex kinetic second objective changed K tape");
            owner.seed(current, component, feature, cusp, response);
            if (last && component == 1)
            {
              const double invalid = std::numeric_limits<double>::quiet_NaN();
#ifdef QMC_CUDA2HIP
              checkRuntime(hipMemcpyAsync(feature, &invalid, 8, hipMemcpyHostToDevice, owner.queue_.getNative()), "late kinetic seed");
#else
              checkRuntime(cudaMemcpyAsync(feature, &invalid, 8, cudaMemcpyHostToDevice, owner.queue_.getNative()), "late kinetic seed");
#endif
              owner.queue_.sync();
              injected = true;
            }
          }, true);
          owner.finishPass(component);
          component ? ++imaginary_passes : ++real_passes;
        }
      });
    }
    catch (const std::runtime_error&) { rejected = true; }
    if (!injected || !rejected || real_passes != imaginary_passes + 1 || !imaginary_passes)
      throw std::runtime_error("Complex kinetic late second-pass rollback was not exercised");
  }

  /// Packed-only kernels must reject generic legal padding before they can launch.
  static void malformedLayouts(PreparedDeviceComplexKinetic& owner)
  {
    const auto rejects = [&](auto launch) {
      bool rejected = false;
      try { launch(); }
      catch (const std::invalid_argument&) { rejected = true; }
      if (!rejected) throw std::runtime_error("Complex kinetic admitted a nonpacked private layout");
    };
    const auto trace = [&](OpenOrbitalJetLayout layout) {
      launchComplexSpatialTraces(owner.queue_.getNative(), layout, owner.channels_.data(), owner.amplitude_.data(),
          owner.inverse_.data(), owner.head_.data(), owner.masses_.data(), owner.scratch_.data(), owner.outputs_.data(),
          owner.outputs_.data(), owner.outputs_.data(), owner.status_.data());
    };
    auto padded = owner.layout_;
    padded.orbitals.configuration_stride += padded.orbitals.element_count;
    rejects([&] { trace(padded); });
    padded = owner.layout_;
    ++padded.orbital_row_stride;
    rejects([&] { trace(padded); });
    padded = owner.layout_;
    ++padded.orbital_determinant_stride;
    rejects([&] { trace(padded); });
    padded = owner.layout_;
    ++padded.orbitals.element_count;
    padded.orbitals.plane_stride = padded.orbitals.element_count;
    padded.orbitals.configuration_stride = padded.orbitals.plane_count * padded.orbitals.element_count;
    rejects([&] { trace(padded); });
    auto cusp = makeSpatialJetLayout(1, 1, owner.masses_.capacity(), SpatialJetMode::FULL_VGL);
    cusp.plane_stride = 2;
    cusp.configuration_stride = 2 * cusp.plane_count;
    rejects([&] { launchComplexKineticCoefficients(owner.queue_.getNative(), cusp, owner.outputs_.data(), owner.outputs_.data(),
        owner.masses_.data(), owner.masses_.data(), owner.kappa_.data(), owner.coefficients_.data(), owner.coefficients_.data(),
        owner.energy_.data(), owner.status_.data()); });
  }
};
} // namespace qmcplusplus::psiformer::device

/// Independent CPU Wide/allP and spatial-energy FD over K1/nondivisor/full/tails.
void validateComplexKineticOwner()
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  using C = std::complex<double>;
  const auto host = [](cuDoubleComplex value) { return C(cuCreal(value), cuCimag(value)); };
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
    const auto& interval = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
    for (std::size_t i = interval.begin; i < interval.end; ++i)
      parameters[i] = 0.09 * std::sin(double(i - interval.begin + 1));
  }
  model.p.set_flat_values(parameters);
  const std::vector<double> inverse_masses{1, 1 / 1.3, 1 / 0.8, 1 / 1.7};
  const std::array<double, 3> kappa{0.13, -0.07, 0.04};
  pf::DirectComplexKineticExecutor oracle(model, plan, kappa);
  auto workspace = oracle.makeWorkspace(oracle.requiredStorageBytes());
  pf::DirectComplexSpatialExecutor spatial(model, plan);
  auto spatial_workspace = spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, parameters, model.cfg.nuclei.x);
  int device = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&device), "complex kinetic device");
#else
  checkRuntime(cudaGetDevice(&device), "complex kinetic device");
#endif
  auto& metrics = deviceMemoryMetrics(device);
  for (std::size_t invalid : {std::size_t(0), std::size_t(5)})
  {
    const auto allocations = metrics.snapshot().allocation_calls;
    bool rejected = false;
    try { (void)estimateComplexKineticCost(plan, invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected || metrics.snapshot().allocation_calls != allocations)
      throw std::runtime_error("Complex kinetic invalid K admission failed");
  }
  std::vector<cuDoubleComplex> response(plan.parameterCount());
  const std::vector<double> base(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 12);
  ComplexValueResult amplitude;
  cuDoubleComplex kinetic;
  double max_response = 0, max_energy = 0, max_fd = 0;
  std::size_t checks = 0, version = 0;
  for (std::size_t targets : {std::size_t(1), std::size_t(3), std::size_t(4)})
  {
    const auto cost = estimateComplexKineticCost(plan, targets);
    const auto before = metrics.snapshot();
    for (bool bytes : {false, true})
    {
      bool rejected = false;
      try { (void)PreparedDeviceComplexKinetic::prepare(image, targets, queue, blas,
          cost.device_bytes - bytes, cost.matrix_work - !bytes); }
      catch (const std::length_error&) { rejected = true; }
      if (!rejected || metrics.snapshot().allocation_calls != before.allocation_calls)
        throw std::runtime_error("Complex kinetic combined preallocation guard failed");
    }
    auto gpu = PreparedDeviceComplexKinetic::prepare(image, targets, queue, blas, cost.device_bytes, cost.matrix_work);
    if (metrics.snapshot().live_bytes - before.live_bytes != cost.device_bytes || cost.canonical_bytes != 24 * response.size())
      throw std::runtime_error("Complex kinetic exact owner/24P bytes differ");
    ComplexKineticOwnerValidationAccess::malformedLayouts(*gpu);
    const auto compare = [&](const std::vector<double>& positions) {
      gpu->evaluate(positions, inverse_masses, kappa);
      gpu->download(gpu->resultView(), response, kinetic, amplitude);
      workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data(), 4));
      const auto expected = oracle.evaluate(*workspace, inverse_masses.data(), 4);
      near(amplitude.log_abs, expected.amplitude.log_abs, "complex kinetic log amplitude", 2e-11);
      near(cuCreal(amplitude.phase), expected.amplitude.phase.real(), "complex kinetic phase real", 2e-11);
      near(cuCimag(amplitude.phase), expected.amplitude.phase.imag(), "complex kinetic phase imag", 2e-11);
      max_energy = std::max(max_energy, std::abs(host(kinetic) - expected.kinetic));
      near(cuCreal(kinetic), expected.kinetic.real(), "complex kinetic energy real", 2e-11);
      near(cuCimag(kinetic), expected.kinetic.imag(), "complex kinetic energy imag", 2e-11);
      for (std::size_t p = 0; p < response.size(); ++p)
      {
        max_response = std::max(max_response, std::abs(host(response[p]) - expected.parameter_response[p]));
        near(cuCreal(response[p]), expected.parameter_response[p].real(), "complex kinetic allP real", 2e-11);
        near(cuCimag(response[p]), expected.parameter_response[p].imag(), "complex kinetic allP imaginary", 2e-11);
      }
      ++checks;
    };
    const auto unavailable = [&] {
      bool rejected = false;
      try { (void)gpu->resultView(); }
      catch (const std::logic_error&) { rejected = true; }
      if (!rejected) throw std::runtime_error("Complex kinetic exposed partial/failed output");
    };
    compare(base);
    auto foreign = gpu->resultView();
    foreign.owner = nullptr;
    bool foreign_rejected = false;
    try { gpu->download(foreign, response, kinetic, amplitude); }
    catch (const std::invalid_argument&) { foreign_rejected = true; }
    if (!foreign_rejected) throw std::runtime_error("Complex kinetic accepted foreign owner");
    const auto stale = gpu->resultView();
    const auto warm = metrics.snapshot();
    for (std::size_t sample = 0; sample < 3; ++sample)
      compare(std::vector<double>(model.cfg.electrons.x.begin() + 12 * sample, model.cfg.electrons.x.begin() + 12 * (sample + 1)));
    if (metrics.snapshot().allocation_calls != warm.allocation_calls || metrics.snapshot().free_calls != warm.free_calls)
      throw std::runtime_error("Complex kinetic warmed DEVICE allocation/free");
    bool rejected = false;
    try { gpu->download(stale, response, kinetic, amplitude); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Complex kinetic accepted stale generation");
    compare(base);

    // Signed canonical directional FD uses independent CPU spatial G/L, not dK.
    C dot{};
    std::vector<double> direction(response.size());
    for (std::size_t p = 0; p < response.size(); ++p)
    {
      direction[p] = (int(p % 13) - 6) * 0.01 / std::sqrt(double(response.size()));
      dot += host(response[p]) * direction[p];
    }
    C sides[2];
    constexpr double h = 1e-4;
    for (std::size_t side = 0; side < 2; ++side)
    {
      auto shifted = parameters;
      for (std::size_t p = 0; p < shifted.size(); ++p) shifted[p] += (side ? -h : h) * direction[p];
      model.p.set_flat_values(shifted);
      spatial_workspace->setPositions(pf::GeometryPositionView::interleaved(base.data(), 4));
      const auto value = spatial.evaluateFull(*spatial_workspace);
      C energy{};
      for (std::size_t e = 0; e < 4; ++e)
      {
        C square{};
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
          const auto total = value.gradient[3 * e + axis] + C(0, kappa[axis]);
          square += total * total;
        }
        energy -= 0.5 * inverse_masses[e] * (value.laplacian_log[e] + square);
      }
      sides[side] = energy;
    }
    model.p.set_flat_values(parameters);
    const auto fd = (sides[0] - sides[1]) / (2 * h);
    max_fd = std::max(max_fd, std::abs(fd - dot));
    near(fd.real(), dot.real(), "complex kinetic spatial-energy FD real", 2e-8);
    near(fd.imag(), dot.imag(), "complex kinetic spatial-energy FD imaginary", 2e-8);

    // Late physical tail/imaginary failure must revoke amplitude, K and P together.
    if (targets != 4)
    {
      ComplexKineticOwnerValidationAccess::failLate(*gpu, base, inverse_masses, kappa);
      unavailable();
      compare(base);
    }
    auto invalid_masses = inverse_masses;
    invalid_masses.back() = -1;
    rejected = false;
    try { gpu->evaluate(base, invalid_masses, kappa); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Complex kinetic accepted invalid physical mass");
    unavailable();
    compare(base);
    const auto old_image = gpu->resultView();
    auto zero = parameters;
    for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
    {
      const auto& interval = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
      std::fill(zero.begin() + interval.begin, zero.begin() + interval.end, 0);
    }
    const auto replace = [&](const std::vector<double>& values) {
      model.p.set_flat_values(values);
      image = image->makeUpdatedImage(values, ++version);
      const auto allocations = metrics.snapshot().allocation_calls;
      gpu->rebindModel(image);
      if (metrics.snapshot().allocation_calls != allocations) throw std::runtime_error("Complex kinetic rebind allocated");
    };
    replace(zero);
    rejected = false;
    try { gpu->download(old_image, response, kinetic, amplitude); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Complex kinetic accepted stale image");
    compare(base);
    bool imaginary_response = false;
    for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
    {
      const auto& interval = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
      for (std::size_t p = interval.begin; p < interval.end; ++p)
        imaginary_response |= std::abs(host(response[p])) > 1e-12;
    }
    if (!imaginary_response) throw std::runtime_error("Zero-imaginary head lost its imaginary-parameter response");
    auto singular = zero;
    const auto& real_up = plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT);
    std::fill(singular.begin() + real_up.begin, singular.begin() + real_up.end, 0);
    replace(singular);
    rejected = false;
    try { gpu->evaluate(base, inverse_masses, kappa); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Complex kinetic admitted singular canonical inverses");
    unavailable();
    replace(parameters);
    compare(base);
    std::cout << "complex_kinetic_cost K=" << targets << " owner_device_bytes=" << cost.device_bytes
              << " host_numeric_bytes=" << cost.host_bytes << " canonical_bytes=" << cost.canonical_bytes
              << " matrix_work_sites=" << cost.matrix_work << " actual_device_bytes=" << cost.device_bytes
              << " mutable_layout_metadata_bytes=" << sizeof(OpenOrbitalJetLayout)
              << " warm_device_allocations=0 omitted=image,vendor,object_metadata\n";
  }
  std::cout << "PASS complex kinetic owner checks=" << checks << " max_cpu_response=" << max_response
            << " max_cpu_kinetic=" << max_energy << " max_spatial_energy_fd=" << max_fd
            << " same_tape_two_pass=1 late_tail_rollback=1 exact_24P=1\n";
}
#endif
