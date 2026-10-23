//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_ROOTS_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_ROOTS_VALIDATION_H
#include "PsiFormerDeviceComplexKineticValidation.h"
#include <chrono>
#include <cstring>

/// Reuse independent CPU Wide/FD/domain tests before one bounded synthetic root timing.
void validateCooperativeComplexRoots(bool large_only = false)
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  if (!large_only)
    validateComplexKineticLeaves();
  const std::size_t n = large_only ? 100 : 33, d = large_only ? 16 : 4, k = 1;
  const auto nn = n * n, m = d * nn, span = 5 * m;
  auto layout = makeOpenOrbitalJetLayout(1, d, n, 1, 3, n / 2, SpatialJetMode::FULL_VGL);
  layout.features = withSpatialTargetRange(layout.features, n - 1, k);
  layout.orbitals = withSpatialTargetRange(layout.orbitals, n - 1, k);
  std::vector<cuDoubleComplex> inverse(m), head(span), roots(span);
  std::vector<ComplexValueChannel> records(d);
  for (std::size_t channel = 0; channel < d; ++channel)
  {
    records[channel] = {make_cuDoubleComplex(std::cos(.13 * channel), std::sin(.13 * channel)), -.1 * channel, 0};
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
      {
        const auto index = channel * nn + i * n + j;
        inverse[index] = head[index] = make_cuDoubleComplex(i == j ? 1 : 0, 0);
        for (std::size_t plane = 1; plane < 5; ++plane)
          head[plane * m + index] = make_cuDoubleComplex(.001 * std::sin(double(index + 3 * plane)),
              .0007 * std::cos(double(2 * index + plane)));
      }
  }
  const std::vector<cuDoubleComplex> gradient(3, make_cuDoubleComplex(.03, -.02));
  const std::vector<cuDoubleComplex> laplacian(1, make_cuDoubleComplex(-.04, .01));
  const std::vector<cuDoubleComplex> qg(3, make_cuDoubleComplex(.2, -.03));
  const std::vector<cuDoubleComplex> ql(1, make_cuDoubleComplex(-.5, 0));
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  DeviceBuffer<cuDoubleComplex> inv(m), jets(span), output(span), scratch(3 * nn + 2);
  DeviceBuffer<cuDoubleComplex> dg(3), dl(1), dqg(3), dql(1);
  DeviceBuffer<ComplexValueChannel> channels(d);
  DeviceBuffer<ComplexValueResult> amplitude(1);
  DeviceBuffer<std::uint64_t> status(1);
  inv.upload(inverse.data(), m);
  jets.upload(head.data(), span);
  channels.upload(records.data(), d);
  dg.upload(gradient.data(), 3);
  dl.upload(laplacian.data(), 1);
  dqg.upload(qg.data(), 3);
  dql.upload(ql.data(), 1);
  const ComplexValueResult regular{make_cuDoubleComplex(1, 0), 0, 1, 0, 0};
  amplitude.upload(&regular, 1);
  const auto canary = make_cuDoubleComplex(19, -23);
  std::vector<cuDoubleComplex> guarded(3 * nn + 2, canary);
  scratch.upload(guarded.data(), guarded.size());
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "complex root device");
  using Event = hipEvent_t;
  const auto create = [](Event* e) { return hipEventCreate(e); };
  const auto record = [](Event e, Stream s) { return hipEventRecord(e, s); };
  const auto elapsed = [](float* t, Event a, Event b) { return hipEventElapsedTime(t, a, b); };
  const auto destroy = [](Event e) { return hipEventDestroy(e); };
#else
  checkRuntime(cudaGetDevice(&ordinal), "complex root device");
  using Event = cudaEvent_t;
  const auto create = [](Event* e) { return cudaEventCreate(e); };
  const auto record = [](Event e, Stream s) { return cudaEventRecord(e, s); };
  const auto elapsed = [](float* t, Event a, Event b) { return cudaEventElapsedTime(t, a, b); };
  const auto destroy = [](Event e) { return cudaEventDestroy(e); };
#endif
  Event begin, end;
  checkRuntime(create(&begin), "complex root timing begin");
  checkRuntime(create(&end), "complex root timing end");
  auto& metrics = deviceMemoryMetrics(ordinal);
  const auto before = metrics.snapshot();
  for (std::size_t repetition = 0; repetition < 3; ++repetition)
  {
    const auto clear = std::numeric_limits<std::uint64_t>::max();
    status.upload(&clear, 1);
    checkRuntime(record(begin, queue.getNative()), "complex root timing event");
    const auto start = std::chrono::steady_clock::now();
    checkRuntime(launchComplexKineticMatrixRoots(queue.getNative(), layout, channels.data(), amplitude.data(),
        inv.data(), jets.data(), dg.data(), dl.data(), dqg.data(), dql.data(), output.data(), scratch.data() + 1,
        status.data()), "bounded complex matrix roots");
    checkRuntime(record(end, queue.getNative()), "complex root timing event");
    queue.sync();
    const auto wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    float milliseconds = 0;
    checkRuntime(elapsed(&milliseconds, begin, end), "complex root elapsed");
    std::uint64_t code;
    status.download(&code, 1);
    if (code != clear)
      throw std::runtime_error("Bounded complex root timing failed finite admission");
    output.download(roots.data(), span);
    std::uint64_t fingerprint = 1469598103934665603ull;
    const auto* bytes = reinterpret_cast<const unsigned char*>(roots.data());
    for (std::size_t i = 0; i < span * sizeof(cuDoubleComplex); ++i)
      fingerprint = (fingerprint ^ bytes[i]) * 1099511628211ull;
    std::cout << "PASS bounded complex matrix roots Ne=" << n << " D=" << d << " K=1 repetition=" << repetition
              << " measured=" << (repetition != 0) << " gpu_ms=" << milliseconds << " wall_seconds=" << wall
              << " root_fingerprint=" << fingerprint << " scratch_bytes=" << 48 * nn
              << " synthetic_leaf_not_physical_sample=1\n";
  }
  if (metrics.snapshot().allocation_calls != before.allocation_calls || metrics.snapshot().free_calls != before.free_calls)
    throw std::runtime_error("Bounded complex root leaf allocated warmed DEVICE storage");
  std::vector<cuDoubleComplex> after(m);
  inv.download(after.data(), m);
  if (std::memcmp(inverse.data(), after.data(), m * sizeof(cuDoubleComplex)))
    throw std::runtime_error("Cooperative matrix roots modified the original inverse");
  scratch.download(guarded.data(), guarded.size());
  if (std::memcmp(&canary, guarded.data(), sizeof(canary)) ||
      std::memcmp(&canary, guarded.data() + guarded.size() - 1, sizeof(canary)))
    throw std::runtime_error("Cooperative matrix roots exceeded exact three-matrix scratch");
  checkRuntime(destroy(begin), "complex root timing destroy");
  checkRuntime(destroy(end), "complex root timing destroy");
}
#endif
