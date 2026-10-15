//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_METROPOLIS_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_METROPOLIS_VALIDATION_H
#include "PsiFormerDeviceMetropolis.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include <iostream>

namespace qmcplusplus::psiformer::device
{
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

    // A late feature overflow follows earlier valid microtiles, but publishes no prefix.
    if (population == 5)
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
               "CPU law/cache parity, view epochs, stale/double/failed resolution, late numerical rollback, "
               "model refresh, mixed node/regular isolation, unused uniform/signflip, exact bytes, "
               "zero warmed device allocations/fixed host-mask capacity; "
            << "max_abs_error=" << maximum_error
            << " measured_D2D_calls=" << total_copies << " measured_draw_H2D_bytes=" << total_upload_bytes
            << " measured_status_mask_D2H_bytes=" << total_download_bytes
            << " transfer_counter_scope=device_wide; XML/RNG/Hamiltonian NOT_INTEGRATED\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
