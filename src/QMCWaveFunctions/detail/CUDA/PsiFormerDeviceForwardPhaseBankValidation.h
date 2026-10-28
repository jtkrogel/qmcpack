//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_FORWARD_PHASE_BANK_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_FORWARD_PHASE_BANK_VALIDATION_H

#include "PsiFormerDeviceMixedPeriodicGeometryValidation.h"

namespace qmcplusplus::psiformer::device
{
/// A synthetic localized Ne100 head exercises capacity, live tails and failed successors.
struct ForwardPhaseBankValidationAccess
{
  /// Check exact allocation/output contracts across tails, failure and image replacement.
  static void run(compute::Queue<PlatformKind::CUDA>& queue)
  {
    const char* key = "PSIFORMER_GPU_FORWARD_PHASE_BANK";
    const char* previous_value = std::getenv(key);
    const std::optional<std::string> previous = previous_value ? std::optional<std::string>(previous_value) : std::nullopt;
    // Restore only this validator's cold selector; prepared owners retain their choice.
    struct Restore
    {
      const char* key;
      const std::optional<std::string>& value;
      ~Restore() { if (value) setenv(key, value->c_str(), 1); else unsetenv(key); }
    } restore{key, previous};
    unsetenv(key);
    if (configuredForwardPhaseBank()) throw std::runtime_error("Forward phase bank default changed");
    setenv(key, "invalid", 1);
    bool rejected = false;
    try { (void)configuredForwardPhaseBank(); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Forward phase bank accepted an invalid selector");

    ExecutionEnvironment environment;
    environment.boundary = BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.lattice_vectors = {{{80., .3, .1}, {0., 80., .2}, {.1, 0., 80.}}};
    environment.periodic_axes = {true, true, true};
    const ModelShape shape{50, 50, 100, 2, 16, 4, 1};
    auto parameters = mixed_periodic_validation::parameters(environment, shape);
    const auto plan = PsiFormerExecutionPlan::fromParameters(parameters, shape, environment);
    const auto ne = shape.electrons();
    std::vector<double> nuclei(3 * ne), positions(6 * ne);
    for (std::size_t e = 0; e < ne; ++e)
      for (std::size_t d = 0; d < 3; ++d)
      {
        const auto grid = d == 0 ? e % 5 : d == 1 ? (e / 5) % 5 : e / 25;
        nuclei[3 * e + d] = 1. + 6. * grid;
        for (std::size_t b = 0; b < 2; ++b)
          positions[b * 3 * ne + 3 * e + d] = nuclei[3 * e + d] + .06 + .02 * d + .01 * b;
      }
    auto image = std::make_shared<PreparedDeviceModel>(plan, parameters.flat_values(), nuclei, 0);
    AcceleratorBlasHandle blas(queue);
    setenv(key, "0", 1);
    PreparedDeviceInference reference(image, 2, queue, blas, SpatialJetMode::FULL_VGL);
    auto& metrics = deviceMemoryMetrics(reference.raw_.deviceOrdinal());
    const auto before_candidate = metrics.snapshot();
    setenv(key, "1", 1);
    PreparedDeviceInference candidate(image, 2, queue, blas, SpatialJetMode::FULL_VGL);
    const auto after_candidate = metrics.snapshot();
    const auto saved = checkedDeviceBytes<double>(std::min(spatialJetSpanElements(candidate.features_.output),
        spatialJetSpanElements(candidate.orbitals_.orbitals)));
    if (!candidate.usesForwardPhaseBank() || reference.usesForwardPhaseBank() ||
        candidate.forwardPhaseBankSavedBytes() != saved || candidate.orbital_jets_.capacity() ||
        reference.deviceWorkspaceBytes() - candidate.deviceWorkspaceBytes() != saved ||
        after_candidate.live_bytes - before_candidate.live_bytes != candidate.deviceWorkspaceBytes())
      throw std::runtime_error("Forward phase bank differs from exact prepared capacity");
    const auto planned = estimateOrdinaryFullVglCost(plan, 2, true);
    if (planned.device_workspace_bytes != candidate.deviceWorkspaceBytes() ||
        planned.host_workspace_bytes != candidate.hostWorkspaceBytes() ||
        estimateOrdinaryFullVglCost(plan, 2).device_workspace_bytes != reference.deviceWorkspaceBytes())
      throw std::runtime_error("Ordinary FULL planned bytes differ from prepared capacity");
    DeviceMemoryMetrics limited;
    limited.configureAdmission({planned.device_workspace_bytes - 1, 0});
    bool cap_rejected = false;
    try { auto reservation = limited.reserveAllocation(planned.device_workspace_bytes); }
    catch (const std::length_error&) { cap_rejected = true; }
    if (!cap_rejected || limited.snapshot().allocation_calls || limited.snapshot().live_bytes)
      throw std::runtime_error("Forward phase-bank cap-minus-one admitted a device allocation");
    setenv(key, "0", 1); // A prepared graph must not reread its selector.
    if (!candidate.usesForwardPhaseBank()) throw std::runtime_error("Forward phase bank reread its selector");
    const auto download = [](PreparedDeviceInference& graph) {
      mixed_periodic_validation::Output out;
      const auto& layout = graph.resultLayout();
      out.phases.resize(layout.configuration_count);
      out.jets.resize(spatialJetSpanElements(layout));
      out.ratios.resize(layout.configuration_count * layout.laplacian_lanes);
      graph.downloadResults(out.phases, out.jets, out.ratios);
      return out;
    };
    const auto compare = [&](std::size_t live) {
      const std::vector<double> prefix(positions.begin(), positions.begin() + live * 3 * ne);
      reference.evaluate(prefix);
      candidate.evaluate(prefix);
      mixed_periodic_validation::identical(download(reference), download(candidate));
      (void)candidate.completedSpatialFinals();
    };
    compare(2);
    bool tape_rejected = false;
    try { (void)candidate.spatialHeadTape(); }
    catch (const std::logic_error&) { tape_rejected = true; }
    if (!tape_rejected) throw std::runtime_error("Forward phase bank exposed an unprepared reverse tape");
    auto successor = image->makeUpdatedImage(parameters.flat_values(), 1);
    const auto warm = metrics.snapshot();
    compare(1);
    compare(2);
    auto broken = positions;
    broken[3 * ne + 2] = std::numeric_limits<double>::infinity();
    bool failed = false;
    try { candidate.evaluate(broken); }
    catch (const std::runtime_error&) { failed = true; }
    if (!failed) throw std::runtime_error("Forward phase bank accepted nonfinite successor coordinates");
    bool stale_rejected = false;
    try { (void)candidate.completedSpatialFinals(); }
    catch (const std::logic_error&) { stale_rejected = true; }
    if (!stale_rejected) throw std::runtime_error("Failed phase-bank evaluation retained completed finals");
    compare(2);
    reference.rebindModel(successor);
    candidate.rebindModel(successor);
    stale_rejected = false;
    try { (void)candidate.completedSpatialFinals(); }
    catch (const std::logic_error&) { stale_rejected = true; }
    if (!stale_rejected) throw std::runtime_error("Phase-bank image rebind retained old finals");
    compare(1);
    const auto after_warm = metrics.snapshot();
    if (after_warm.allocation_calls != warm.allocation_calls || after_warm.live_bytes != warm.live_bytes)
      throw std::runtime_error("Phase-bank tails/recovery/image rebind allocated warm scratch");
    // Production D16 reverses the relative extents: orbital jets now own the larger bank.
    const ModelShape wide_head{50, 50, 100, 16, 16, 4, 1};
    auto wide_parameters = mixed_periodic_validation::parameters(environment, wide_head);
    const auto wide_plan = PsiFormerExecutionPlan::fromParameters(wide_parameters, wide_head, environment);
    auto wide_image = std::make_shared<PreparedDeviceModel>(wide_plan, wide_parameters.flat_values(), nuclei, 0);
    setenv(key, "0", 1);
    PreparedDeviceInference wide_reference(wide_image, 1, queue, blas, SpatialJetMode::FULL_VGL);
    setenv(key, "1", 1);
    PreparedDeviceInference wide_bank(wide_image, 1, queue, blas, SpatialJetMode::FULL_VGL);
    const auto raw_span = spatialJetSpanElements(wide_bank.features_.output);
    const auto orbital_span = spatialJetSpanElements(wide_bank.orbitals_.orbitals);
    if (orbital_span <= raw_span || wide_bank.raw_.capacity() != orbital_span ||
        wide_reference.deviceWorkspaceBytes() - wide_bank.deviceWorkspaceBytes() != checkedDeviceBytes<double>(raw_span))
      throw std::runtime_error("D16 orbital-dominant phase bank capacity is incorrect");
    const std::vector<double> first(positions.begin(), positions.begin() + 3 * ne);
    wide_reference.evaluate(first);
    wide_bank.evaluate(first);
    mixed_periodic_validation::identical(download(wide_reference), download(wide_bank));
    std::cout << "Forward phase bank D16 PASS saved_device_bytes=" << wide_bank.forwardPhaseBankSavedBytes()
              << " planned_device_bytes=" << estimateOrdinaryFullVglCost(wide_plan, 1, true).device_workspace_bytes
              << "; exact outputs; orbital_span>raw_span\n";
    std::cout << "Forward phase bank PASS Ne=" << ne << " B=2 D=2 saved_device_bytes=" << saved
              << " reference_device_bytes=" << reference.deviceWorkspaceBytes()
              << " bank_device_bytes=" << candidate.deviceWorkspaceBytes()
              << "; exact outputs, tails, failure recovery, image rebind, warm allocations=0\n";
  }
};
}
#endif
