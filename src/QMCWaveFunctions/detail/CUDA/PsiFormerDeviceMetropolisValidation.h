//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_METROPOLIS_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_METROPOLIS_VALIDATION_H
#include "PsiFormerDeviceMetropolis.h"
#include "PsiFormerDeviceBatchBackend.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCDrivers/GreenFunctionModifiers/DriftModifierUNR.h"
#include "Particle/ParticleBase/RandomSeqGenerator.h"
#include "Utilities/RandomGenerator.h"
#include <iostream>
#include <sstream>

namespace qmcplusplus::psiformer::device
{
/// Compare the production adapter's masks/draw mapping with copied native engines.
template<class Model>
void validateNativeMetropolisAdapter(Model& model, const std::shared_ptr<const PreparedDeviceModel>& image,
                                     bool use_drift = false)
{
  const auto ne = image->plan().modelShape().electrons(), stride = 3 * ne;
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "Native sampler validation device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "Native sampler validation device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  if (use_drift && image->plan().environment().boundary == BoundaryCondition::PERIODIC)
  {
    DeviceExecutionPreparation preparation;
    preparation.metropolis_workspace_budget_bytes = 4096;
    preparation.metropolis_drift_budget_bytes = 4096;
    const auto before = metrics.snapshot();
    bool rejected = false;
    try { PreparedDeviceBatchBackend unsupported(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE, preparation); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected || metrics.snapshot().allocation_calls != before.allocation_calls)
      throw std::runtime_error("Periodic production drift was not rejected before device allocation");
    std::cout << "PASS periodic production drift rejection before allocation/native draws\n";
    return;
  }
  DriftModifierUNR policy(.37); // Exercise the actual nondefault native limiter, not a copied formula.
  const NativeDriftCallback drift = [&](double variance, const double* gradient, double* result) {
    DriftModifierUNR::GradType g;
    DriftModifierUNR::PosType d;
    for (std::size_t xyz = 0; xyz < 3; ++xyz)
      g[xyz] = gradient[xyz];
    policy.getDrift(variance, g, d);
    for (std::size_t xyz = 0; xyz < 3; ++xyz)
      result[xyz] = d[xyz];
  };
  for (const double scale : {0., 1.e-20, 100.})
  {
    const double g[3]{scale, -scale, scale};
    double d[3];
    drift(.02, g, d);
    for (const auto value : d)
      if (!device_math::isFiniteBinary64(value) || (scale == 0 && value != 0))
        throw std::runtime_error("Native UNR zero/small/large-gradient callback failed");
  }
  for (const auto population : {std::size_t{1}, std::size_t{5}})
  {
    std::vector<double> positions(population * stride), masses(ne, 1), exported(positions.size());
    for (std::size_t walker = 0; walker < population; ++walker)
      std::copy_n(model.cfg.electrons.x.data() + (walker % model.cfg.nconfig) * stride, stride,
                  positions.data() + walker * stride);
    for (std::size_t electron = 0; electron < ne; ++electron)
      masses[electron] = .8 + .13 * electron;
    const auto cost = estimateMetropolisCost(population, ne);
    DeviceExecutionPreparation preparation;
    const auto drift_cost = estimateMetropolisDriftCost(image->plan(), 2, population);
    preparation.metropolis_workspace_budget_bytes = cost.device_bytes - !use_drift;
    preparation.metropolis_drift_budget_bytes = use_drift ? drift_cost.device_workspace_bytes - 1 : 0;
    PreparedDeviceBatchBackend under_budget(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE, preparation);
    auto rejected_workspace = under_budget.prepare(2);
    std::size_t draw_calls = 0;
    const auto gaussian_probe = [&](std::vector<double>&) { ++draw_calls; };
    const auto uniform_probe = [&]() { ++draw_calls; return .5; };
    bool rejected = false;
    const auto rejected_before = metrics.snapshot();
    try
    {
      under_budget.advanceMetropolis(*rejected_workspace, positions, masses, .02, 2, image->version(),
                                    gaussian_probe, uniform_probe, exported, {}, use_drift, drift);
    }
    catch (const std::length_error&) { rejected = true; }
    if (!rejected || draw_calls || metrics.snapshot().allocation_calls != rejected_before.allocation_calls)
      throw std::runtime_error("Production sampler budget rejection consumed native draws");
    preparation.metropolis_workspace_budget_bytes = cost.device_bytes;
    preparation.metropolis_drift_budget_bytes = use_drift ? drift_cost.device_workspace_bytes : 0;
    PreparedDeviceBatchBackend backend(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE, preparation);
    auto workspace = backend.prepare(2);
    Model reference(model);
    auto cpu = std::make_unique<pf::DirectValueExecutor>(reference, image->plan());
    auto cpu_workspace = cpu->makeWorkspace();
    const auto gradient_at = [&](const double* coordinates) {
      pf::DirectSpatialExecutor spatial(reference, *cpu, image->plan());
      auto workspace = spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
      workspace->setPositions(pf::GeometryPositionView::interleaved(coordinates, ne));
      const auto result = spatial.evaluateFull(*workspace);
      return std::vector<double>(result.gradient.data(), result.gradient.data() + stride);
    };
    auto expected_positions = positions;
    std::vector<double> expected_proposal(positions.size());
    std::vector<std::uint32_t> expected_mask(population);
    RandomGenerator cpu_rng(413), gpu_rng(cpu_rng);
    std::size_t cpu_calls = 0, gpu_calls = 0, compared_masks = 0, accepted_masks = 0, rejected_masks = 0;
    auto cpu_draw = [&]() { ++cpu_calls; return cpu_rng(); };
    auto gpu_draw = [&]() { ++gpu_calls; return gpu_rng(); };
    const double tau = .02;
    const auto gaussian = [&](std::vector<double>& actual) {
      std::vector<TinyVector<double, 3>> native_gaussian(population * ne);
      makeGaussRandomWithEngine(native_gaussian, cpu_draw);
      makeGaussRandomWithEngine(actual, gpu_draw);
      std::vector<double> old_gradient(positions.size()), displacement(positions.size());
      if (use_drift)
        for (std::size_t walker = 0; walker < population; ++walker)
        {
          const auto g = gradient_at(expected_positions.data() + walker * stride);
          std::copy(g.begin(), g.end(), old_gradient.begin() + walker * stride);
        }
      for (std::size_t coordinate = 0; coordinate < actual.size(); ++coordinate)
      {
        if (actual[coordinate] != native_gaussian[coordinate / 3][coordinate % 3])
          throw std::runtime_error("Production sampler changed native Gaussian flattening");
        displacement[coordinate] = std::sqrt(tau * masses[(coordinate / 3) % ne]) * actual[coordinate];
        if (use_drift)
        {
          const auto begin = coordinate / 3 * 3;
          double d[3];
          drift(tau * masses[(coordinate / 3) % ne], old_gradient.data() + begin, d);
          displacement[coordinate] += d[coordinate % 3];
        }
        expected_proposal[coordinate] = expected_positions[coordinate] + displacement[coordinate];
      }
      for (std::size_t walker = 0; walker < population; ++walker)
      {
        cpu_workspace->setPositions(pf::GeometryPositionView::interleaved(expected_positions.data() + walker * stride, ne));
        const auto old_value = cpu->evaluate(*cpu_workspace);
        cpu_workspace->setPositions(pf::GeometryPositionView::interleaved(expected_proposal.data() + walker * stride, ne));
        const auto proposed = cpu->evaluate(*cpu_workspace);
        double gf = 0, gb = 0;
        if (use_drift)
        {
          const auto g = gradient_at(expected_proposal.data() + walker * stride);
          for (std::size_t electron = 0; electron < ne; ++electron)
          {
            const auto index = walker * stride + electron * 3;
            const double variance = tau * masses[electron];
            TinyVector<double, 3> reverse, gaussian_displacement;
            double d[3];
            drift(variance, g.data() + electron * 3, d);
            for (std::size_t xyz = 0; xyz < 3; ++xyz)
            {
              gaussian_displacement[xyz] = std::sqrt(variance) * actual[index + xyz];
              reverse[xyz] = displacement[index + xyz] + d[xyz];
            }
            gf -= dot(gaussian_displacement, gaussian_displacement) / (2 * variance);
            gb -= dot(reverse, reverse) / (2 * variance);
          }
        }
        const double alpha = use_drift ? 2 * (proposed.logabs - old_value.logabs) + gb - gf
                                       : 2 * (proposed.logabs - old_value.logabs);
        bool take = false;
        if (proposed.sign != 0 && device_math::isFiniteBinary64(alpha))
        {
          const auto u = cpu_draw();
          take = u == 0 || std::log(u) < alpha;
        }
        expected_mask[walker] = take ? METROPOLIS_ACCEPTED : 0;
        if (take)
          std::copy_n(expected_proposal.data() + walker * stride, stride, expected_positions.data() + walker * stride);
      }
    };
    const auto observe = [&](const std::vector<std::uint32_t>& actual) {
      std::ostringstream cpu_state, gpu_state;
      cpu_rng.write(cpu_state);
      gpu_rng.write(gpu_state);
      if (actual != expected_mask || cpu_calls != gpu_calls || cpu_state.str() != gpu_state.str())
        throw std::runtime_error("Production sampler masks/draws/final native engine differ");
      for (const auto mask : actual)
        (mask & METROPOLIS_ACCEPTED ? accepted_masks : rejected_masks)++;
      ++compared_masks;
    };
    // Cold seed, then warm replay with persistent positions and the same graph.
    for (std::size_t request = 0; request < 2; ++request)
    {
      const auto before = metrics.snapshot();
      backend.advanceMetropolis(*workspace, positions, masses, tau, 2, image->version(), gaussian, gpu_draw, exported,
                                observe, use_drift, drift);
      const auto after = metrics.snapshot();
      if (request && (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls))
        throw std::runtime_error("Production sampler warm replay allocated device storage");
      if (!request && after.live_bytes - before.live_bytes != cost.device_bytes +
          (use_drift ? drift_cost.device_workspace_bytes : 0))
        throw std::runtime_error("Production sampler exact admission disagrees with actual allocation delta");
      for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate)
        if (std::abs(exported[coordinate] - expected_positions[coordinate]) > 2.e-14)
          throw std::runtime_error("Production sampler accepted physical coordinates differ");
      positions = exported;
    }
    auto values = reference.p.flat_values();
    values[image->plan().parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin] += .02;
    backend.publishParameters(values, image->version() + 1, [&]() { reference.p.set_flat_values(values); });
    cpu = std::make_unique<pf::DirectValueExecutor>(reference, image->plan());
    cpu_workspace = cpu->makeWorkspace();
    backend.advanceMetropolis(*workspace, positions, masses, tau, 2, image->version() + 1,
                             gaussian, gpu_draw, exported, observe, use_drift, drift);
    if (use_drift)
    {
      // Recreate derived device caches from accepted host coordinates, retaining native RNG state.
      positions = exported;
      workspace.reset();
      workspace = backend.prepare(2);
      backend.advanceMetropolis(*workspace, positions, masses, tau, 2, image->version() + 1,
                                gaussian, gpu_draw, exported, observe, true, drift);
    }
    if (compared_masks != (use_drift ? 8 : 6))
      throw std::runtime_error("Production sampler did not qualify every substep mask");
    if (use_drift)
    {
      // A late native-policy failure must not publish caller coordinates or draw uniforms.
      const auto saved = exported;
      std::size_t policy_calls = 0;
      draw_calls = 0;
      const auto zeros = [&](std::vector<double>& g) { ++draw_calls; std::fill(g.begin(), g.end(), 0); };
      const NativeDriftCallback failing_policy = [&](double variance, const double* g, double* d) {
        if (++policy_calls > population * ne)
          throw std::runtime_error("Injected late native drift failure");
        drift(variance, g, d);
      };
      bool failed = false;
      try { backend.advanceMetropolis(*workspace, positions, masses, tau, 1, image->version() + 1,
          zeros, uniform_probe, exported, {}, true, failing_policy); }
      catch (const std::runtime_error&) { failed = true; }
      if (!failed || exported != saved || draw_calls != 1)
        throw std::runtime_error("Drift failure partially published or consumed uniforms");

      // An unavailable seed gradient fails before any native Gaussian/uniform draw.
      auto coalesced = positions;
      std::copy_n(model.cfg.nuclei.x.data(), 3, coalesced.data());
      draw_calls = 0;
      failed = false;
      try { backend.advanceMetropolis(*workspace, coalesced, masses, tau, 1, image->version() + 1,
          zeros, uniform_probe, exported, {}, true, drift); }
      catch (const std::runtime_error&) { failed = true; }
      if (!failed || draw_calls || exported != saved)
        throw std::runtime_error("Unavailable seed gradient was not a before-draw failclosed rejection");
      backend.advanceMetropolis(*workspace, positions, masses, tau, 1, image->version() + 1,
                                zeros, uniform_probe, exported, {}, true, drift);
    }
    std::cout << "PASS native-engine sampler adapter: B=" << population
              << " usedrift=" << use_drift << " drift_UNR_a=.37"
              << " accepted=" << accepted_masks << " rejected=" << rejected_masks
              << " VALUE_capacity=2 substeps=2 requests=3 masks=" << compared_masks
              << " native_draws=" << gpu_calls << " final_engine=identical successor_refresh=yes\n";
  }
}

/// Compare checked selected proposals and native drift against complete CPU configurations.
template<class Model>
void validateSelectedMetropolis(Model& model, const std::shared_ptr<const PreparedDeviceModel>& image,
                               compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
{
  const auto ne = image->plan().modelShape().electrons(), stride = 3 * ne;
  const std::size_t population = 3;
  std::vector<std::size_t> counts{1, ne / 2, ne - 1};
  std::sort(counts.begin(), counts.end());
  counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
  pf::DirectValueExecutor value(model, image->plan());
  pf::DirectSpatialExecutor spatial(model, value, image->plan());
  auto reference = spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  DriftModifierUNR policy(.37);
  const NativeDriftCallback drift = [&](double variance, const double* gradient, double* result) {
    DriftModifierUNR::GradType g;
    DriftModifierUNR::PosType d;
    for (int xyz = 0; xyz < 3; ++xyz) g[xyz] = gradient[xyz];
    policy.getDrift(variance, g, d);
    for (int xyz = 0; xyz < 3; ++xyz) result[xyz] = d[xyz];
  };
  for (const bool use_drift : {false, true})
    for (const auto count : counts)
    {
      if (!count || count >= ne) continue;
      std::vector<double> seed(population * stride), masses(ne), gaussian(population * count * 3),
          expected(seed.size()), exported(seed.size()), phases(population), logs(population), uniforms(population);
      std::vector<std::size_t> selected(population * count);
      std::vector<std::uint32_t> expected_mask(population);
      for (std::size_t electron = 0; electron < ne; ++electron) masses[electron] = .7 + .17 * electron;
      for (std::size_t walker = 0; walker < population; ++walker)
      {
        std::copy_n(model.cfg.electrons.x.data() + walker % model.cfg.nconfig * stride, stride,
                    seed.data() + walker * stride);
        if (walker == 2 && image->plan().environment().boundary == BoundaryCondition::PERIODIC)
          for (std::size_t electron = 0; electron < ne; ++electron)
            for (int xyz = 0; xyz < 3; ++xyz)
              seed[walker * stride + electron * 3 + xyz] +=
                  image->plan().environment().lattice_vectors[0][xyz] -
                  2 * image->plan().environment().lattice_vectors[2][xyz];
        for (std::size_t local = 0; local < count; ++local)
        {
          selected[walker * count + local] = (walker ? ne - count : 0) + local;
          for (int d = 0; d < 3; ++d)
            gaussian[(walker * count + local) * 3 + d] = walker == 1 ? 20 : .03 * (1 + local + d);
        }
        uniforms[walker] = walker == 1 ? std::nextafter(1., 0.) : 0;
      }
      expected = seed;
      const double tau = .03;
      for (std::size_t walker = 0; walker < population; ++walker)
      {
        reference->setPositions(pf::GeometryPositionView::interleaved(seed.data() + walker * stride, ne));
        const auto old = spatial.evaluateFull(*reference);
        const double old_log = old.logabs;
        std::vector<double> old_gradient(old.gradient.data(), old.gradient.data() + stride);
        std::vector<double> displacement(count * 3);
        double forward = 0, backward = 0;
        for (std::size_t local = 0; local < count; ++local)
        {
          const auto electron = selected[walker * count + local];
          const double variance = tau * masses[electron];
          double d[3]{};
          if (use_drift) drift(variance, old_gradient.data() + 3 * electron, d);
          for (int xyz = 0; xyz < 3; ++xyz)
          {
            const auto index = local * 3 + xyz;
            const double g = std::sqrt(variance) * gaussian[walker * count * 3 + index];
            displacement[index] = g + d[xyz];
            expected[walker * stride + 3 * electron + xyz] += displacement[index];
            forward -= g * g / (2 * variance);
          }
        }
        reference->setPositions(pf::GeometryPositionView::interleaved(expected.data() + walker * stride, ne));
        const auto proposed = spatial.evaluateFull(*reference);
        if (use_drift)
          for (std::size_t local = 0; local < count; ++local)
          {
            const auto electron = selected[walker * count + local];
            const double variance = tau * masses[electron];
            double d[3]; drift(variance, proposed.gradient.data() + 3 * electron, d);
            for (int xyz = 0; xyz < 3; ++xyz)
            {
              const auto residual = displacement[local * 3 + xyz] + d[xyz];
              backward -= residual * residual / (2 * variance);
            }
          }
        const auto alpha = 2 * (proposed.logabs - old_log) + (use_drift ? backward - forward : 0);
        const bool take = proposed.sign != 0 && device_math::isFiniteBinary64(alpha) &&
            (uniforms[walker] == 0 || std::log(uniforms[walker]) < alpha);
        expected_mask[walker] = take ? METROPOLIS_ACCEPTED : 0;
        if (!take) std::copy_n(seed.data() + walker * stride, stride, expected.data() + walker * stride);
      }
      PreparedDeviceInference graph(image, 2, queue, blas, SpatialJetMode::VALUE_ONLY, use_drift);
      const auto cost = estimateMetropolisCost(population, ne, true);
      auto owner = use_drift
          ? PreparedDeviceMetropolis::prepareDrift(graph, queue, seed, masses, cost.device_bytes,
              estimateMetropolisDriftCost(image->plan(), 2, population).device_workspace_bytes, drift, true)
          : PreparedDeviceMetropolis::prepare(graph, queue, seed, masses, cost.device_bytes, true);
      int ordinal = -1;
#ifdef QMC_CUDA2HIP
      checkRuntime(hipGetDevice(&ordinal), "Selected Metropolis validation device");
#else
      checkRuntime(cudaGetDevice(&ordinal), "Selected Metropolis validation device");
#endif
      auto& metrics = deviceMemoryMetrics(ordinal);
      const auto before = metrics.snapshot();
      for (int fault = 0; fault < 3; ++fault)
      {
        auto invalid = selected;
        if (fault == 0) invalid[0] = ne;
        else if (fault == 1) invalid.clear();
        else if (count > 1) invalid[1] = invalid[0];
        else continue;
        bool rejected = false;
        try { (void)owner->propose(gaussian, tau, image->version(), invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || metrics.snapshot().upload_calls != before.upload_calls)
          throw std::runtime_error("Malformed selected descriptor reached device staging");
      }
      const auto token = owner->propose(gaussian, tau, image->version(), selected);
      const auto mask = owner->resolve(token, uniforms);
      const auto after = metrics.snapshot();
      if (mask != expected_mask || after.allocation_calls != before.allocation_calls || after.free_calls != before.free_calls)
        throw std::runtime_error("Selected Metropolis joint masks or warm allocation contract differ");
      owner->exportAccepted(exported);
      owner->downloadAcceptedValues(phases, logs);
      for (std::size_t walker = 0; walker < population; ++walker)
      {
        for (std::size_t electron = 0; electron < ne; ++electron)
          for (int d = 0; d < 3; ++d)
          {
            const auto index = walker * stride + 3 * electron + d;
            const bool moved = std::binary_search(selected.begin() + walker * count,
                selected.begin() + (walker + 1) * count, electron);
            if ((!moved && exported[index] != seed[index]) || std::abs(exported[index] - expected[index]) > 2e-14)
              throw std::runtime_error("Selected Metropolis changed unselected or incorrect accepted coordinates");
          }
        reference->setPositions(pf::GeometryPositionView::interleaved(expected.data() + walker * stride, ne));
        const auto result = spatial.evaluateFull(*reference);
        if (phases[walker] != result.sign || std::abs(logs[walker] - result.logabs) > 2e-9 * (1 + std::abs(result.logabs)))
          throw std::runtime_error("Selected Metropolis complete accepted value differs from CPU");
      }
      std::cout << "PASS selected device Metropolis: B3/tail Ne=" << ne << " n=" << count
                << " native_drift=" << use_drift << " CPU_joint_masks=yes unselected_exact=yes warm_allocations=0\n";
    }
}

/// Qualify the numerical owner with injected draws, not a new production sampler.
template<class Model>
void validateDeviceMetropolis(Model& model, const std::shared_ptr<const PreparedDeviceModel>& image,
                              compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
{
  const auto ne = image->plan().modelShape().electrons(), stride = ne * 3;
  pf::DirectValueExecutor cpu(model, image->plan());
  auto cpu_workspace = cpu.makeWorkspace();
  if (metropolisLogComparison(-1, -1) || !metropolisLogComparison(-2, -1) ||
      !metropolisLogComparison(-INFINITY, -1))
    throw std::runtime_error("Metropolis strict/zero-uniform comparison changed");
  double maximum_error = 0;
  std::size_t total_copies = 0, total_upload_bytes = 0, total_download_bytes = 0;
  // Compare finite signed-log values and exact coordinate/mask publication.
  const auto close = [&](double actual, double expected) {
    const auto error = std::abs(actual - expected);
    if (!device_math::isFiniteBinary64(error) || error > 2.e-9 * (1 + std::abs(expected)))
      throw std::runtime_error("Injected device Metropolis differs from the CPU reference");
    maximum_error = std::max(maximum_error, error);
  };
  for (const auto population : {std::size_t{1}, std::size_t{5}})
  {
    std::vector<double> accepted(population * stride), gaussian(accepted.size()), inverse_mass(ne),
        uniforms(population), proposed(accepted.size()), old_phase(population), old_log(population),
        phase(population), logabs(population), expected_phase(population), expected_log(population);
    std::vector<bool> expected_accept(population);
    for (std::size_t walker = 0; walker < population; ++walker)
      std::copy_n(model.cfg.electrons.x.data() + (walker % model.cfg.nconfig) * stride, stride,
                  accepted.data() + walker * stride);
    for (std::size_t electron = 0; electron < ne; ++electron)
      inverse_mass[electron] = .7 + .17 * electron;
    const auto cost = estimateMetropolisCost(population, ne);
    PreparedDeviceInference graph(image, 2, queue, blas, SpatialJetMode::VALUE_ONLY);
    int ordinal = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&ordinal), "Metropolis validation device");
#else
    checkRuntime(cudaGetDevice(&ordinal), "Metropolis validation device");
#endif
    auto& metrics = deviceMemoryMetrics(ordinal);
    const auto before = metrics.snapshot();
    bool rejected = false;
    try { (void)PreparedDeviceMetropolis::prepare(graph, queue, accepted, inverse_mass, cost.device_bytes - 1); }
    catch (const std::length_error&) { rejected = true; }
    if (!rejected || metrics.snapshot().allocation_calls != before.allocation_calls)
      throw std::runtime_error("Metropolis budget rejection allocated numeric storage");
    auto owner = PreparedDeviceMetropolis::prepare(graph, queue, accepted, inverse_mass, cost.device_bytes);
    const auto after = metrics.snapshot();
    if (after.live_bytes - before.live_bytes != cost.device_bytes || owner->cost().host_bytes != 4 * population ||
        after.upload_calls - before.upload_calls != 2 ||
        after.uploaded_bytes - before.uploaded_bytes != sizeof(double) * (accepted.size() + ne))
      throw std::runtime_error("Metropolis exact preflight/cold transfers disagree with actual ownership");
    std::cout << "Metropolis owner cost: B=" << population << " Ne=" << ne
              << " device_bytes=" << cost.device_bytes << " host_mask_bytes=" << cost.host_bytes
              << " borrowed_VALUE_graph_bytes=" << graph.deviceWorkspaceBytes() << " graph_excluded_from_owner=yes\n";
    owner->downloadAcceptedValues(old_phase, old_log);
    for (std::size_t walker = 0; walker < population; ++walker)
    {
      cpu_workspace->setPositions(pf::GeometryPositionView::interleaved(accepted.data() + walker * stride, ne));
      const auto result = cpu.evaluate(*cpu_workspace);
      close(old_phase[walker], result.sign);
      close(old_log[walker], result.logabs);
      old_phase[walker] = result.sign;
      old_log[walker] = result.logabs;
    }
    const double tau = .13;
    for (std::size_t coordinate = 0; coordinate < gaussian.size(); ++coordinate)
      gaussian[coordinate] = .07 * (static_cast<int>(coordinate % 7) - 3);
    for (std::size_t coordinate = 0; coordinate < proposed.size(); ++coordinate)
      proposed[coordinate] = accepted[coordinate] +
          std::sqrt(tau * inverse_mass[(coordinate / 3) % ne]) * gaussian[coordinate];
    for (std::size_t walker = 0; walker < population; ++walker)
    {
      cpu_workspace->setPositions(pf::GeometryPositionView::interleaved(proposed.data() + walker * stride, ne));
      const auto result = cpu.evaluate(*cpu_workspace);
      const double alpha = 2 * (result.logabs - old_log[walker]);
      uniforms[walker] = walker % 2 == 0 ? 0 : std::nextafter(1.0, 0.0);
      const bool take = result.sign != 0 && device_math::isFiniteBinary64(alpha) &&
          (uniforms[walker] == 0 || std::log(uniforms[walker]) < alpha);
      expected_accept[walker] = take;
      expected_phase[walker] = take ? result.sign : old_phase[walker];
      expected_log[walker] = take ? result.logabs : old_log[walker];
      if (!take)
        std::copy_n(accepted.data() + walker * stride, stride, proposed.data() + walker * stride);
    }
    const auto old_accepted_view = owner->acceptedView(0, 1);
    const auto warm_before = metrics.snapshot();
    const auto token = owner->propose(gaussian, tau, image->version());
    const auto old_proposed_view = owner->proposedView(token, 0, 1);
    const auto accepted_mask = owner->resolve(token, uniforms);
    const auto warm_after = metrics.snapshot();
    if (warm_after.allocation_calls != warm_before.allocation_calls || warm_after.free_calls != warm_before.free_calls ||
        warm_after.upload_calls - warm_before.upload_calls != 2 ||
        warm_after.uploaded_bytes - warm_before.uploaded_bytes != sizeof(double) * (gaussian.size() + population) ||
        warm_after.device_copy_calls - warm_before.device_copy_calls != (population + 1) / 2 ||
        warm_after.device_copied_bytes - warm_before.device_copied_bytes != accepted.size() * sizeof(double))
      throw std::runtime_error("Metropolis warmed draws/microtiles allocated or changed exact transfers");
    total_copies += warm_after.device_copy_calls - warm_before.device_copy_calls;
    total_upload_bytes += warm_after.uploaded_bytes - warm_before.uploaded_bytes;
    total_download_bytes += warm_after.downloaded_bytes - warm_before.downloaded_bytes;
    owner->exportAccepted(accepted);
    owner->downloadAcceptedValues(phase, logabs);
    for (std::size_t coordinate = 0; coordinate < accepted.size(); ++coordinate)
      close(accepted[coordinate], proposed[coordinate]);
    for (std::size_t walker = 0; walker < population; ++walker)
    {
      close(phase[walker], expected_phase[walker]);
      close(logabs[walker], expected_log[walker]);
      if ((accepted_mask[walker] == METROPOLIS_ACCEPTED) != expected_accept[walker])
        throw std::runtime_error("Metropolis accepted mask differs from injected CPU decisions");
    }
    // Old accepted and proposal source epochs must both be revoked by publication.
    for (const auto& view : {old_accepted_view, old_proposed_view})
    {
      rejected = false;
      try { graph.evaluateDeviceCoordinates(view); }
      catch (const std::invalid_argument&) { rejected = true; }
      if (!rejected)
        throw std::runtime_error("Metropolis resolution left an old coordinate view current");
    }
    rejected = false;
    try { owner->resolve(token, uniforms); }
    catch (const std::exception&) { rejected = true; }
    if (!rejected)
      throw std::runtime_error("Metropolis resolved the same proposal twice");

    // Open features overflow here; periodic features reduce the same finite input.
    // Both geometries retain the failed-resolution/version rollback checks below.
    if (population == 5 && image->plan().environment().boundary == BoundaryCondition::OPEN)
    {
      auto bad_gaussian = gaussian;
      bad_gaussian[4 * stride] = std::numeric_limits<double>::max() / 4;
      const auto generation = graph.evaluationGeneration();
      rejected = false;
      try { (void)owner->propose(bad_gaussian, .01, image->version()); }
      catch (const std::runtime_error&) { rejected = true; }
      if (!rejected || graph.evaluationGeneration() - generation != 3)
        throw std::runtime_error("Metropolis late-failure fixture did not exercise the final VALUE microtile");
      owner->exportAccepted(proposed);
      owner->downloadAcceptedValues(expected_phase, expected_log);
      if (proposed != accepted || expected_phase != phase || expected_log != logabs)
        throw std::runtime_error("Late Metropolis failure changed accepted coordinates/cache");
    }
    const auto pending = owner->propose(gaussian, tau, image->version());
    std::vector<double> bad_uniform(population, 1.0);
    rejected = false;
    try { owner->resolve(pending, bad_uniform); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected)
      throw std::runtime_error("Metropolis accepted an invalid eligible uniform");
    owner->exportAccepted(proposed);
    if (proposed != accepted)
      throw std::runtime_error("Failed Metropolis resolution changed accepted positions");

    // Rebinding invalidates old cache/proposal versions, not physical coordinates.
    const auto stale = owner->propose(gaussian, tau, image->version());
    auto values = model.p.flat_values();
    values[image->plan().parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin] += .02;
    const auto successor = image->makeUpdatedImage(values, image->version() + 1);
    graph.rebindModel(successor);
    rejected = false;
    try { owner->resolve(stale, uniforms); }
    catch (const std::exception&) { rejected = true; }
    if (!rejected)
      throw std::runtime_error("Metropolis accepted a proposal from retired parameters");
    rejected = false;
    try { (void)owner->propose(gaussian, tau, successor->version()); }
    catch (const std::logic_error&) { rejected = true; }
    if (!rejected)
      throw std::runtime_error("Metropolis reused an unrefreshed accepted cache");
    owner->refreshAccepted(successor->version());
    owner->downloadAcceptedValues(phase, logabs);
    Model changed(model);
    changed.p.set_flat_values(values);
    pf::DirectValueExecutor changed_cpu(changed, image->plan());
    auto changed_workspace = changed_cpu.makeWorkspace();
    for (std::size_t walker = 0; walker < population; ++walker)
    {
      changed_workspace->setPositions(pf::GeometryPositionView::interleaved(accepted.data() + walker * stride, ne));
      const auto result = changed_cpu.evaluate(*changed_workspace);
      close(phase[walker], result.sign);
      close(logabs[walker], result.logabs);
    }
    const auto recovered = owner->propose(gaussian, tau, successor->version());
    owner->resolve(recovered, std::vector<double>(population, 0.0));
    const auto expired = owner->acceptedView(0, 1);
    const auto before_release = metrics.snapshot();
    owner.reset();
    if (before_release.live_bytes - metrics.snapshot().live_bytes != cost.device_bytes)
      throw std::runtime_error("Metropolis destruction leaked numeric ownership");
    rejected = false;
    try { graph.evaluateDeviceCoordinates(expired); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected)
      throw std::runtime_error("Metropolis accepted an expired owner view");
  }
  // Dyadic coordinates land exactly on one same-spin node and one odd permutation.
  // The first VALUE microtile mixes regular/node lanes; the following lane stays regular.
  if (image->plan().modelShape().spin_up_electrons >= 2)
  {
    std::vector<double> seed(3 * stride), gaussian(seed.size(), 0), inverse_mass(ne, 1), exported(seed.size()),
        phase(3), logabs(3), old_phase(3), old_log(3);
    for (std::size_t walker = 0; walker < 3; ++walker)
      for (std::size_t electron = 0; electron < ne; ++electron)
      {
        seed[walker * stride + 3 * electron] = .25 * (electron + 1) + .125 * walker;
        seed[walker * stride + 3 * electron + 1] = electron % 2 ? .125 : -.125;
        seed[walker * stride + 3 * electron + 2] = .25 * (static_cast<int>(electron % 3) - 1);
      }
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      gaussian[stride + 3 + dimension] = seed[stride + dimension] - seed[stride + 3 + dimension];
      gaussian[2 * stride + dimension] = seed[2 * stride + 3 + dimension] - seed[2 * stride + dimension];
      gaussian[2 * stride + 3 + dimension] = -gaussian[2 * stride + dimension];
    }
    PreparedDeviceInference graph(image, 2, queue, blas, SpatialJetMode::VALUE_ONLY);
    auto owner = PreparedDeviceMetropolis::prepare(graph, queue, seed, inverse_mass, estimateMetropolisCost(3, ne).device_bytes);
    owner->downloadAcceptedValues(old_phase, old_log);
    const auto token = owner->propose(gaussian, 1, image->version());
    const auto eligible = owner->eligibility(token);
    if (eligible[0] != METROPOLIS_ELIGIBLE || eligible[1] != 0 || eligible[2] != METROPOLIS_ELIGIBLE)
      throw std::runtime_error("One nodal proposal poisoned other VALUE walkers or consumed a uniform");
    const auto result = owner->resolve(token, {0, std::numeric_limits<double>::quiet_NaN(), 0});
    if (result[0] != METROPOLIS_ACCEPTED || result[1] != 0 || result[2] != METROPOLIS_ACCEPTED)
      throw std::runtime_error("Mixed-node Metropolis acceptance or unused-uniform handling changed");
    owner->exportAccepted(exported);
    owner->downloadAcceptedValues(phase, logabs);
    for (std::size_t coordinate = stride; coordinate < 2 * stride; ++coordinate)
      if (exported[coordinate] != seed[coordinate])
        throw std::runtime_error("Rejected node changed accepted coordinates");
    close(phase[1], old_phase[1]);
    close(logabs[1], old_log[1]);
    close(phase[2], -old_phase[2]);
    close(logabs[2], old_log[2]);
  }
  bool overflow_rejected = false;
  try { (void)estimateMetropolisCost(std::numeric_limits<std::size_t>::max(), ne); }
  catch (const std::length_error&) { overflow_rejected = true; }
  if (!overflow_rejected)
    throw std::runtime_error("Metropolis unchecked extent overflow");
  std::cout << "PASS injected device Metropolis: B1/5 VALUE microtile2/tails, nonunit mass/tau, u0, "
               "CPU law/cache parity, view epochs, stale/double/failed resolution, "
            << (image->plan().environment().boundary == BoundaryCondition::OPEN
                    ? "late numerical rollback, " : "periodic finite-feature reduction, ")
            << "model refresh, mixed node/regular isolation, unused uniform/signflip, exact bytes, "
               "zero warmed device allocations/fixed host-mask capacity; "
            << "max_abs_error=" << maximum_error
            << " measured_D2D_calls=" << total_copies << " measured_draw_H2D_bytes=" << total_upload_bytes
            << " measured_status_mask_D2H_bytes=" << total_download_bytes
            << " transfer_counter_scope=device_wide; XML/RNG/Hamiltonian NOT_INTEGRATED\n";
  validateNativeMetropolisAdapter(model, image);
  validateNativeMetropolisAdapter(model, image, true);
  validateSelectedMetropolis(model, image, queue, blas);
}
} // namespace qmcplusplus::psiformer::device
#endif
