//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_VGL_OWNERSHIP_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_VGL_OWNERSHIP_VALIDATION_H

#include "PsiFormerDeviceInference.h"
#include <chrono>
#include <iomanip>
#include <ostream>

namespace qmcplusplus::psiformer::device
{

/// Refuse an unbounded measurement before any numerical DEVICE owner exists.
inline void validateVglOwnershipAdmission()
{
  const auto& policy = configuredDeviceMemoryAdmission();
  constexpr std::size_t gib = std::size_t(1024) * 1024 * 1024;
  if (!policy.limit_bytes || *policy.limit_bytes > 3 * gib ||
      policy.external_reserve_bytes < gib / 4 || policy.external_reserve_bytes >= *policy.limit_bytes)
    throw std::invalid_argument("VGL ownership requires aggregate limit <=3GiB including reserve >=256MiB");
}

/// Measure one complete B1 request, never eager co-owned reverse/score tapes.
/// Each FULL/K1 comparison uses a fresh process; CPU oracles run separately.
inline void measureVglOwnership(pf::PsiFormer& model, const PsiFormerExecutionPlan& plan,
                               bool target_k1, std::ostream& output)
{
  validateVglOwnershipAdmission();
  int ordinal = -1;
  std::size_t free_bytes = 0, total_bytes = 0;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "VGL ownership device");
  checkRuntime(hipMemGetInfo(&free_bytes, &total_bytes), "VGL ownership headroom");
#else
  checkRuntime(cudaGetDevice(&ordinal), "VGL ownership device");
  checkRuntime(cudaMemGetInfo(&free_bytes, &total_bytes), "VGL ownership headroom");
#endif
  const auto& policy = configuredDeviceMemoryAdmission();
  if (*policy.limit_bytes > total_bytes - total_bytes / 5 || free_bytes < *policy.limit_bytes)
    throw std::length_error("VGL ownership requires 20 percent physical headroom and free aggregate allowance");
  auto& metrics = deviceMemoryMetrics(ordinal);
  if (metrics.snapshot().live_bytes != 0)
    throw std::logic_error("VGL ownership requires a fresh process without live numerical DEVICE owners");
  metrics.reportAdmission(std::cout, ordinal);
  std::cout << "VGL_OWNERSHIP device=" << ordinal << " initial_free_bytes=" << free_bytes
            << " physical_total_bytes=" << total_bytes << '\n';
  const auto ne = plan.modelShape().electrons();
  if (!ne || model.cfg.electrons.x.size() < 3 * ne)
    throw std::invalid_argument("VGL ownership requires one complete configuration");
  const std::vector<double> positions(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * ne);
  TargetVglPreparation preparation;
  if (target_k1)
  {
    const auto cost = estimateTargetVglCost(plan, 1, 1);
    preparation = {1, cost.device_workspace_bytes, cost.charged_scalar_work_units};
  }
  compute::Queue<PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), model.cfg.nuclei.x, model.p.version());
  const auto image_bytes = metrics.snapshot().live_bytes;
  std::cout << "VGL_OWNERSHIP before_scratch mode=" << (target_k1 ? "K1" : "FULL")
            << " Ne=" << ne << " image_device_bytes=" << image_bytes << std::endl;
  try
  {
    PreparedDeviceInference owner(image, 1, queue, blas, SpatialJetMode::FULL_VGL,
                                 false, false, configuredInferenceStatusPolicy(), preparation);
    const auto& layout = owner.resultLayout();
    std::vector<double> phase(1), jets(spatialJetSpanElements(layout)), ratios(ne);
    const auto download = [&] { owner.downloadResults(phase, jets, ratios); };
    owner.evaluate(positions); queue.sync(); download(); // Warm exact complete request.
    const auto before = metrics.snapshot();
    const auto start = std::chrono::steady_clock::now();
    owner.evaluate(positions); queue.sync();
    const auto computed = std::chrono::steady_clock::now();
    download();
    const auto completed = std::chrono::steady_clock::now();
    const auto after = metrics.snapshot();
    if (after.allocation_calls != before.allocation_calls || after.free_calls != before.free_calls)
      throw std::runtime_error("Warmed complete VGL request changed DEVICE ownership");
    if (after.live_bytes != image_bytes + owner.deviceWorkspaceBytes())
      throw std::runtime_error("VGL ownership contains unreported numerical DEVICE owners");
    double kinetic = 0;
    for (const auto value : ratios) kinetic -= 0.5 * value;
    const auto finite = [](double value) {
      if (!device_math::isFiniteBinary64(value)) throw std::runtime_error("Nonfinite complete VGL ownership output");
    };
    finite(phase[0]); finite(kinetic);
    for (const auto value : jets) finite(value);
    for (const auto value : ratios) finite(value);
    output << std::setprecision(17) << "# index sign logabs kinetic gradient... lap_log... lap_ratio...\n"
           << "0 " << phase[0] << ' ' << jets[layout.uncheckedValueOffset(0, 0)] << ' ' << kinetic;
    for (std::size_t i = 0; i < 3 * ne; ++i) output << ' ' << jets[layout.uncheckedGradientOffset(0, i, 0)];
    for (std::size_t i = 0; i < ne; ++i) output << ' ' << jets[layout.uncheckedLaplacianOffset(0, i, 0)];
    for (const auto value : ratios) output << ' ' << value;
    output << '\n';
    std::cout << std::setprecision(17) << "VGL_OWNERSHIP mode=" << (target_k1 ? "K1" : "FULL")
              << " Ne=" << ne << " P=" << plan.parameterCount() << " W=1 B=1 targets_completed=" << ne
              << " K=" << (target_k1 ? 1 : ne) << " Nup=" << model.cfg.nup << " Ndown=" << model.cfg.ndown
              << " D=" << model.ndet << " width=" << model.dim << " heads=" << model.heads << " blocks=" << model.blocks
              << " image_device_bytes=" << image_bytes << " scratch_device_bytes=" << owner.deviceWorkspaceBytes()
              << " scratch_host_bytes=" << owner.hostWorkspaceBytes() << " native_live_bytes=" << after.live_bytes
              << " native_peak_bytes=" << after.peak_bytes << " device_compute_seconds="
              << std::chrono::duration<double>(computed - start).count() << " host_complete_seconds="
              << std::chrono::duration<double>(completed - start).count()
              << " warm_allocations=0 H2D=" << after.uploaded_bytes - before.uploaded_bytes
              << " D2H=" << after.downloaded_bytes - before.downloaded_bytes
              << " D2D=" << after.device_copied_bytes - before.device_copied_bytes
              << " vendor_private=EXCLUDED_RESERVED native_CPU_model_and_output_HOST=EXCLUDED\n";
  }
  catch (...)
  {
    const auto partial = metrics.snapshot();
    std::cerr << "VGL_OWNERSHIP CENSORED incomplete_request native_peak_bytes=" << partial.peak_bytes
              << " native_live_bytes=" << partial.live_bytes << '\n';
    throw;
  }
  queue.sync();
  if (metrics.snapshot().live_bytes != image_bytes)
    throw std::runtime_error("VGL scratch did not retire before image release");
  image.reset();
  if (metrics.snapshot().live_bytes != 0)
    throw std::runtime_error("VGL measurement retained an unexpected DEVICE owner");
  if (!output) throw std::runtime_error("Cannot save complete VGL ownership output");
  std::cout << "PASS complete VGL ownership; retired_native_bytes=0; no VMC/optimizer/whole-process claim\n";
}

} // namespace qmcplusplus::psiformer::device
#endif
