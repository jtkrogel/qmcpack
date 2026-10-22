//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_RETAINED_TRACE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_RETAINED_TRACE_VALIDATION_H

#include "PsiFormerDeviceTargetKinetic.h"
#include "PsiFormerDeviceTargetCostValidation.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerKineticExecutor.h"
#include <cstring>

namespace qmcplusplus::psiformer::device
{
/// Test-only access preserves the production tape ABI and immediate-borrow authority.
struct RetainedTraceValidationAccess
{
  struct Tile
  {
    std::size_t begin, count;
    std::vector<double> gradient, ratio, lap_log, root;
    device_determinant::DerivativeStatus status;
  };

  static std::unique_ptr<PreparedDeviceInference> prepare(
      const std::shared_ptr<const PreparedDeviceModel>& image, std::size_t targets,
      std::size_t trace_bytes, compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
  {
    const auto cost = estimateRetainedTargetForwardCost(image->plan(), targets, trace_bytes);
    return std::unique_ptr<PreparedDeviceInference>(new PreparedDeviceInference(image, 1, queue, blas,
        SpatialJetMode::FULL_VGL, false, true, configuredInferenceStatusPolicy(),
        {targets, cost.device_workspace_bytes, cost.charged_scalar_work_units, trace_bytes, false}, true));
  }

  static std::vector<Tile> snapshot(PreparedDeviceInference& graph, const std::vector<double>& positions)
  {
    constexpr double sentinel = -937.25;
    const auto k = graph.target_preparation_.max_targets;
    const auto prime = [&] {
      const std::vector<double> gradient(3 * k, sentinel), laplacian(k, sentinel);
      graph.det_gradient_.upload(gradient.data(), gradient.size());
      graph.det_lap_ratio_.upload(laplacian.data(), laplacian.size());
      graph.det_lap_log_.upload(laplacian.data(), laplacian.size());
    };
    prime();
    std::vector<Tile> tiles;
    std::optional<PreparedDeviceInference::SpatialTargetTapeView> saved;
    graph.evaluateRetainedTargets(positions, [&](const auto& view) {
      graph.validateTargetTape(view);
      saved = view;
      Tile tile{view.head.root_layout.target_electron_begin, view.head.root_layout.laplacian_lanes,
          std::vector<double>(3 * k), std::vector<double>(k), std::vector<double>(k),
          std::vector<double>(view.head.root_layout.plane_count)};
      graph.queue_.sync();
      graph.det_gradient_.download(tile.gradient.data(), tile.gradient.size());
      graph.det_lap_ratio_.download(tile.ratio.data(), tile.ratio.size());
      graph.det_lap_log_.download(tile.lap_log.data(), tile.lap_log.size());
      graph.det_status_.download(&tile.status, 1);
      graph.output_.download(tile.root.data(), tile.root.size());
      if (view.head.determinant_log_gradient != graph.det_gradient_.data() ||
          view.head.determinant_log_laplacian != graph.det_lap_log_.data())
        throw std::runtime_error("Retained trace changed the consumed determinant tape ABI");
      for (std::size_t lane = 3 * tile.count; lane < 3 * k; ++lane)
        if (tile.gradient[lane] != sentinel)
          throw std::runtime_error("Retained trace overwrote a ragged gradient tail");
      for (std::size_t lane = tile.count; lane < k; ++lane)
        if (tile.ratio[lane] != sentinel || tile.lap_log[lane] != sentinel)
          throw std::runtime_error("Retained trace overwrote a ragged Laplacian tail");
      tiles.push_back(std::move(tile));
      prime();
    });
    bool stale = false;
    try { graph.validateTargetTape(*saved); }
    catch (const std::logic_error&) { stale = true; }
    if (!stale)
      throw std::runtime_error("Retained trace admitted a saved completed-tile view");
    if (graph.parallel_traces_)
    {
      bool invalidated = false;
      try { (void)graph.parallel_traces_->result(); }
      catch (const std::logic_error&) { invalidated = true; }
      if (!invalidated)
        throw std::runtime_error("Retained trace left a borrowed leaf result admitted");
    }
    return tiles;
  }
};

/// Require canonical GPU tape/response bits; retain the independent CPU response bound.
inline void validateRetainedParallelTraces(pf::PsiFormer& model,
    const std::shared_ptr<const PreparedDeviceModel>& image,
    compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
{
  constexpr std::size_t trace_bytes = 1024 * 1024;
  const auto& plan = image->plan();
  const auto n = plan.modelShape().electrons(), p = plan.parameterCount();
  const std::vector<double> positions(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * n);
  int device = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&device), "retained trace metrics device");
#else
  checkRuntime(cudaGetDevice(&device), "retained trace metrics device");
#endif
  auto& metrics = deviceMemoryMetrics(device);
  const auto exact_gpu = [](const std::vector<double>& actual, const std::vector<double>& expected) {
    if (actual.size() != expected.size())
      throw std::runtime_error("Retained trace changed canonical GPU output extent");
    for (std::size_t i = 0; i < actual.size(); ++i)
      if (!device_math::isFiniteBinary64(actual[i]) || !device_math::isFiniteBinary64(expected[i]) ||
          std::memcmp(&actual[i], &expected[i], sizeof(double)) != 0)
        throw std::runtime_error("Retained trace differs from finite canonical GPU bits at offset " +
            std::to_string(i));
  };
  std::vector<double> masses(n), drift(3 * n), response(p), reference_response(p);
  for (std::size_t i = 0; i < n; ++i) masses[i] = .8 + .17 * i;
  for (std::size_t i = 0; i < 3 * n; ++i) drift[i] = .03 * (int(i % 7) - 3);
  pf::DirectKineticExecutor cpu(model, plan);
  auto workspace = cpu.makeWorkspace();
  workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data(), n));
  const auto cpu_result = cpu.evaluate(*workspace, drift.data(), drift.size(), masses.data(), masses.size());
  const std::vector<double> expected(cpu_result.kinetic_parameter_response.begin(),
      cpu_result.kinetic_parameter_response.end());
  const auto near_response = [&](double weight) {
    for (std::size_t i = 0; i < p; ++i)
      if (!device_math::isFiniteBinary64(response[i]) || !device_math::isFiniteBinary64(expected[i]) ||
          std::abs(response[i] - weight * expected[i]) > 2e-7 * (1 + std::abs(weight * expected[i])))
        throw std::runtime_error("Retained parallel trace parameter response exceeds the existing Model bound");
  };
  for (std::size_t k : {std::size_t(1), n > 2 ? n - 1 : n})
  {
    const auto plain = estimateTargetKineticCost(plan, k, 1);
    const auto cost = estimateTargetKineticCost(plan, k, 1, trace_bytes);
    const auto leaf = planSpatialTraceWorkspace({1, plan.modelShape().determinants, n, 3 * k, k}, trace_bytes);
    if (cost.device_workspace_bytes - plain.device_workspace_bytes != leaf.total_device_bytes ||
        cost.host_workspace_bytes != plain.host_workspace_bytes ||
        cost.charged_scalar_work_units != plain.charged_scalar_work_units)
      throw std::runtime_error("Retained trace combined cost did not charge exactly the bounded leaf");
    for (bool short_leaf : {false, true})
    {
      const auto before = metrics.snapshot();
      bool rejected = false;
      try
      {
        PreparedDeviceTargetKinetic invalid(image,
            {k, cost.device_workspace_bytes - !short_leaf, cost.charged_scalar_work_units, 1,
             short_leaf ? 1 : trace_bytes}, queue, blas);
      }
      catch (const std::length_error&) { rejected = true; }
      if (!rejected || metrics.snapshot().allocation_calls != before.allocation_calls)
        throw std::runtime_error("Retained trace budget rejection allocated device storage");
    }
    auto scalar = RetainedTraceValidationAccess::prepare(image, k, 0, queue, blas);
    auto parallel = RetainedTraceValidationAccess::prepare(image, k, trace_bytes, queue, blas);
    const auto reference = RetainedTraceValidationAccess::snapshot(*scalar, positions);
    const auto actual = RetainedTraceValidationAccess::snapshot(*parallel, positions);
    if (reference.size() != actual.size())
      throw std::runtime_error("Retained trace changed physical target coverage");
    for (std::size_t t = 0; t < actual.size(); ++t)
    {
      if (actual[t].begin != reference[t].begin || actual[t].count != reference[t].count ||
          actual[t].status != reference[t].status)
        throw std::runtime_error("Retained trace changed target/status admission");
      for (const auto pair : {std::pair{&actual[t].gradient, &reference[t].gradient},
          std::pair{&actual[t].ratio, &reference[t].ratio}, std::pair{&actual[t].lap_log, &reference[t].lap_log},
          std::pair{&actual[t].root, &reference[t].root}})
        exact_gpu(*pair.first, *pair.second);
    }
    // The zero-budget public owner is the unchanged old-GPU kinetic product oracle.
    PreparedDeviceTargetKinetic reference_target(image,
        {k, plain.device_workspace_bytes, plain.charged_scalar_work_units, 1, 0}, queue, blas);
    const auto before_prepare = metrics.snapshot();
    PreparedDeviceTargetKinetic target(image,
        {k, cost.device_workspace_bytes, cost.charged_scalar_work_units, 1, trace_bytes}, queue, blas);
    if (target.deviceWorkspaceBytes() != cost.device_workspace_bytes ||
        target.hostWorkspaceBytes() != cost.host_workspace_bytes ||
        metrics.snapshot().live_bytes - before_prepare.live_bytes != cost.device_workspace_bytes)
      throw std::runtime_error("Retained trace predicted and actual owners differ");
    for (double weight : {.7, -.25, 0.0})
    {
      reference_target.evaluate(positions, {weight}, masses, drift);
      reference_target.downloadResponse(reference_response);
      target.evaluate(positions, {weight}, masses, drift);
      target.downloadResponse(response);
      exact_gpu(response, reference_response);
      near_response(weight);
    }
    const auto before = metrics.snapshot();
    target.evaluate(positions, {.7}, masses, drift);
    const auto after = metrics.snapshot();
    if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls ||
        before.live_bytes != after.live_bytes || target.admittedTargetTiles() != cost.tile_count)
      throw std::runtime_error("Retained trace warmed request changed owners or target coverage");
    auto bad_positions = positions;
    std::copy_n(model.cfg.nuclei.x.data(), 3, bad_positions.data());
    bool failed = false, unavailable = false;
    try { target.evaluate(bad_positions, {1.0}); }
    catch (const std::runtime_error&) { failed = true; }
    try { target.downloadResponse(response); }
    catch (const std::logic_error&) { unavailable = true; }
    if (!failed || !unavailable)
      throw std::runtime_error("Retained parallel trace failure exposed a partial parameter response");
    failed = unavailable = false;
    try { reference_target.evaluate(bad_positions, {1.0}); }
    catch (const std::runtime_error&) { failed = true; }
    try { reference_target.downloadResponse(reference_response); }
    catch (const std::logic_error&) { unavailable = true; }
    if (!failed || !unavailable)
      throw std::runtime_error("Canonical GPU failure admitted a partial parameter response");
    reference_target.evaluate(positions, {1.0}, masses, drift);
    reference_target.downloadResponse(reference_response);
    target.evaluate(positions, {1.0}, masses, drift);
    target.downloadResponse(response);
    exact_gpu(response, reference_response);
    near_response(1.0);
    const auto rebound = image->makeUpdatedImage(model.p.flat_values(), image->version() + 1);
    target.rebindModel(rebound);
    reference_target.rebindModel(rebound);
    unavailable = false;
    try { target.downloadResponse(response); }
    catch (const std::logic_error&) { unavailable = true; }
    if (!unavailable) throw std::runtime_error("Retained trace rebind left an old response admitted");
    reference_target.evaluate(positions, {1.0}, masses, drift);
    reference_target.downloadResponse(reference_response);
    target.evaluate(positions, {1.0}, masses, drift);
    target.downloadResponse(response);
    exact_gpu(response, reference_response);
    near_response(1.0);
  }
  std::cout << "PASS retained parallel spatial traces: exact GPU tape/root/all-P response/status/tails, "
               "independent CPU response bound=2e-7, exact owners, "
               "failure/rebind/recovery, warmed allocations=0\n";
}

/// Reuse the existing deterministic native fixture construction at four electrons,
/// so K=3 exercises a real final ragged tile independently of external H2 files.
inline void validateRetainedParallelTraceFixtures(
    compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
{
  // Large-shape accounting uses metadata only, never parameter or graph storage.
  for (const auto n : {std::size_t(100), std::size_t(200)})
  {
    const auto plan = makeTargetCostPlan(n, false);
    const auto base = estimateTargetKineticCost(plan, 1, 1);
    const auto trace = estimateTargetKineticCost(plan, 1, 1, 1024 * 1024);
    const auto expected = n == 100 ? 1040941 : 960941;
    if (trace.device_workspace_bytes - base.device_workspace_bytes != expected ||
        trace.host_workspace_bytes != base.host_workspace_bytes)
      throw std::runtime_error("Large retained-trace metadata charge differs from exact leaf storage");
    DeviceExecutionPreparation preparation;
    preparation.kinetic = {1, trace.device_workspace_bytes, trace.charged_scalar_work_units, 1, 1024 * 1024};
    validateDeviceExecutionPreparation(plan, preparation);
    bool rejected = false;
    try { validateDeviceExecutionPreparation(makeTargetCostPlan(n, true), preparation); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Retained parallel trace admitted unsupported periodic geometry");
    preparation.kinetic.parallel_trace_workspace_budget_bytes = 0;
    preparation.kinetic.device_workspace_budget_bytes = base.device_workspace_bytes;
    preparation.vgl = {1, std::numeric_limits<std::size_t>::max(),
        std::numeric_limits<std::size_t>::max(), 1024 * 1024, false};
    validateDeviceExecutionPreparation(plan, preparation);
    if (preparation.kinetic.parallel_trace_workspace_budget_bytes)
      throw std::runtime_error("Ordinary VGL trace preparation activated the retained trace owner");
  }
  const ExecutionEnvironment environment;
  const ModelShape production{2, 2, 2, 16, 256, 4, 4};
  const auto initialized = initializePsiFormerParameters(production, 1729);
  const PsiFormerExecutionPlan production_plan(production, initialized.layouts, environment);
  std::vector<pf::Layout> layouts;
  std::vector<double> values;
  for (const auto& tensor : production_plan.parameterTensors())
  {
    if (tensor.attention_block != NO_ATTENTION_BLOCK && tensor.attention_block >= 2) continue;
    auto dimensions = tensor.shape;
    std::size_t count = 1;
    for (auto& extent : dimensions)
    {
      if (extent == 256 || extent == 64) extent = 16;
      count *= extent;
    }
    const auto begin = values.size();
    const bool rescale = tensor.role != ParameterRole::ELECTRON_EMBEDDING_WEIGHT &&
        (tensor.shape.size() == 1 || (tensor.shape.size() == 2 && tensor.shape[0] == 256));
    for (std::size_t i = 0; i < count; ++i)
      values.push_back(initialized.values[tensor.begin + i] * (rescale ? 4.0 : 1.0));
    layouts.push_back({tensor.module, tensor.name, dimensions, begin, values.size()});
  }
  pf::ConfigData configurations(
      pf::Tensor({1, 4, 3}, {.4, .7, .9, -.2, 1.3, -.7, 1.1, -.6, .3, -.8, -.4, 1.2}),
      pf::Tensor({2, 3}, {.1, .2, -.1, .8, -.3, .7}), pf::Tensor({2}, {3.0, 1.0}), 2, 2, 4);
  pf::PsiFormer model(pf::Parameters(std::move(values), std::move(layouts)), configurations, 4, 16, 2, 2);
  const auto plan = PsiFormerExecutionPlan::fromParameters(model.p, {2, 2, 2, 4, 16, 2, 2}, environment);
  const auto image = std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), model.cfg.nuclei.x);
  validateRetainedParallelTraces(model, image, queue, blas);
}
} // namespace qmcplusplus::psiformer::device
#endif
