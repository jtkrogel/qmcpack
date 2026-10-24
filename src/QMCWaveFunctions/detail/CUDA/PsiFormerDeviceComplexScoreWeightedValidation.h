//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SCORE_WEIGHTED_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SCORE_WEIGHTED_VALIDATION_H
#include "PsiFormerDeviceComplexScore.h"
#include "PsiFormerDeviceComplexProducts.h"
#include "PsiFormerDeviceTargetCostValidation.h"
#include <array>
#include <cstring>
#include <iomanip>

namespace qmcplusplus::psiformer::device
{
/// Compare primitive buffers, including exact failed partial writes, without a large model.
inline void validateComplexScoreWeightedInverses(bool grid = false)
{
  compute::Queue<PlatformKind::CUDA> queue;
  std::size_t comparisons = 0;
  const auto clear = std::numeric_limits<std::uint64_t>::max();
  for (const auto dimensions : {std::array<std::size_t, 2>{63, 1}, {64, 1}, {65, 3},
                                {100, 16}, {200, 16}, {300, 16}, {64, 17}})
  {
    const auto n = dimensions[0], d = dimensions[1], count = d * n * n;
    DeviceBuffer<cuDoubleComplex> old_inverse(count), new_inverse(count);
    DeviceBuffer<ComplexValueChannel> channels(d);
    DeviceBuffer<ComplexValueResult> old_amplitude(1), new_amplitude(1);
    DeviceBuffer<double> cusp(1);
    DeviceBuffer<std::uint64_t> old_status(1), new_status(1);
    DeviceBuffer<ComplexScoreWeightedGridState> state(grid ? 3 : 0);
    auto& metrics = deviceMemoryMetrics(old_inverse.deviceOrdinal());
    for (std::size_t variant = 0; variant < (grid && n == 100 && d == 16 ? 18 : 16); ++variant)
    {
      if (variant == 12 && d == 1) continue;
      std::vector<cuDoubleComplex> input(count), a(count), b(count);
      std::vector<ComplexValueChannel> metadata(d);
      for (std::size_t i = 0; i < count; ++i)
        input[i] = make_cuDoubleComplex(0.03125 * (1 + i % 17), -0.015625 * (i % 7));
      for (std::size_t c = 0; c < d; ++c)
        metadata[c] = {make_cuDoubleComplex(0.75, 0.125 * (c % 3)), -0.0625 * c, 0};
      ComplexValueResult initial{make_cuDoubleComplex(0.25, -0.5), 19.0, 0.375, 7, 0};
      double cusp_value = 0.125;
      const auto nan = std::numeric_limits<double>::quiet_NaN();
      const auto inf = std::numeric_limits<double>::infinity();
      if (variant == 1)
        for (std::size_t i = 0; i < count; ++i)
          input[i] = make_cuDoubleComplex(i % 2 ? -0.0 : 0.0, i % 3 ? -0.0 : 0.0);
      if (variant == 2)
        for (std::size_t i = 0; i < count; ++i)
          input[i] = make_cuDoubleComplex(std::numeric_limits<double>::denorm_min() * (1 + i % 7), -0.0);
      if (variant == 3)
        for (auto& c : metadata) c.log_abs = -740.0;
      if (variant == 4)
        for (auto& c : metadata) c.phase = make_cuDoubleComplex(0, 0);
      if (variant == 5 || variant == 15) metadata.back().status = 1;
      if (variant == 6) metadata.back().phase = make_cuDoubleComplex(nan, 0);
      if (variant == 7) metadata.back().log_abs = nan;
      if (variant == 8) initial.status = 1;
      if (variant == 9 || variant == 15) input[0] = make_cuDoubleComplex(nan, 0);
      if (variant == 10) input[count / 2] = make_cuDoubleComplex(0, inf);
      if (variant == 11) input.back() = make_cuDoubleComplex(nan, 0);
      if (variant == 12 || variant == 17)
      {
        // A representable near-cancelling denominator yields a finite huge weight.
        for (auto& c : metadata) c = {make_cuDoubleComplex(0, 0), 0, 0};
        metadata[0].phase = make_cuDoubleComplex(1, 0);
        metadata[1].phase = make_cuDoubleComplex(-1 + 0x1p-52, 0);
        input[n * n / 2] = make_cuDoubleComplex(std::numeric_limits<double>::max(), 0);
      }
      if (variant == 13) cusp_value = nan;
      if (variant == 14)
      {
        cusp_value = std::numeric_limits<double>::max();
        for (auto& c : metadata) c.log_abs = std::numeric_limits<double>::max();
      }
      channels.upload(metadata.data(), d); cusp.upload(&cusp_value, 1);
      old_inverse.upload(input.data(), count); new_inverse.upload(input.data(), count);
      old_amplitude.upload(&initial, 1); new_amplitude.upload(&initial, 1);
      const auto incoming = variant == 16 ? std::uint64_t(7) : variant == 17 ? std::uint64_t(0) : clear;
      old_status.upload(&incoming, 1); new_status.upload(&incoming, 1);
      std::array<ComplexScoreWeightedGridState, 3> before_state, after_state;
      if (grid)
      {
        std::memset(before_state.data(), 0xa5, sizeof(before_state));
        state.upload(before_state.data(), before_state.size());
        if (n == 100 && d == 16 && variant == 0)
        {
          const auto invalid = [&](ComplexScoreWeightedGridState* pointer, std::size_t size) {
            bool denied = false;
            try
            {
              (void)launchComplexScoreWeightedInversesGrid(queue.getNative(), d, n, channels.data(),
                  new_inverse.data(), cusp.data(), new_amplitude.data(), new_status.data(), pointer, size);
            }
            catch (const std::invalid_argument&) { denied = true; }
            if (!denied) throw std::runtime_error("Weighted grid accepted missing/aliased private state");
          };
          invalid(nullptr, 1);
          invalid(state.data() + 1, 0);
          invalid(reinterpret_cast<ComplexScoreWeightedGridState*>(new_inverse.data()), 1);
        }
      }
      const auto warm = metrics.snapshot();
      checkRuntime(launchComplexScoreWeightedInverses(queue.getNative(), d, n, channels.data(), old_inverse.data(),
          cusp.data(), old_amplitude.data(), old_status.data()), "scalar weighted inverse oracle");
      if (grid)
        checkRuntime(launchComplexScoreWeightedInversesGrid(queue.getNative(), d, n, channels.data(),
            new_inverse.data(), cusp.data(), new_amplitude.data(), new_status.data(), state.data() + 1, 1),
            "grid weighted inverse");
      else
        checkRuntime(launchComplexScoreWeightedInversesCooperative(queue.getNative(), d, n, channels.data(),
            new_inverse.data(), cusp.data(), new_amplitude.data(), new_status.data()), "cooperative weighted inverse");
      queue.sync();
      const auto after = metrics.snapshot();
      if (warm.allocation_calls != after.allocation_calls || warm.free_calls != after.free_calls)
        throw std::runtime_error("Weighted inverse primitive allocated warm storage");
      std::uint64_t old_code, new_code;
      ComplexValueResult old_result, new_result;
      old_status.download(&old_code, 1); new_status.download(&new_code, 1);
      old_amplitude.download(&old_result, 1); new_amplitude.download(&new_result, 1);
      old_inverse.download(a.data(), count); new_inverse.download(b.data(), count);
      if (grid)
      {
        state.download(after_state.data(), after_state.size());
        if (std::memcmp(&before_state[0], &after_state[0], sizeof(before_state[0])) ||
            std::memcmp(&before_state[2], &after_state[2], sizeof(before_state[2])))
          throw std::runtime_error("Weighted grid escaped its private state span");
      }
      if (old_code != new_code || std::memcmp(&old_result, &new_result, sizeof(old_result)) ||
          std::memcmp(a.data(), b.data(), count * sizeof(a[0])))
        throw std::runtime_error("Weighted inverse changed complete bits/failure prefix: Ne=" +
            std::to_string(n) + " D=" + std::to_string(d) + " variant=" + std::to_string(variant));
      if (variant < 16 && ((variant <= 3 && old_code != clear) || (variant >= 4 && old_code == clear)))
        throw std::runtime_error("Weighted inverse fixture lost its expected success/failure");
      if ((variant == 16 && old_code != 7) || (variant == 17 && old_code != 0))
        throw std::runtime_error("Weighted grid cleared or overwrote incoming public status");
      const auto expected_inverse_index = variant == 9 ? 0 : variant == 10 ? count / 2 : count - 1;
      if ((variant >= 9 && variant <= 11 && old_code != ((expected_inverse_index << 8) | 1)) ||
          (variant == 15 && old_code != (((d - 1) << 8) | 1)))
        throw std::runtime_error("Weighted inverse lost lowest-index or metadata-first failure priority");
      ++comparisons;
    }
  }
  std::cout << (grid ? "PASS weighted inverse grid primitive comparisons=" : "PASS weighted inverse primitive comparisons=")
            << comparisons
            << " all_inverse_amplitude_status_bits=exact failed_partial_bits=exact warm_allocations=0\n";
}

/// Price one frozen state through both native Products scopes without allocating a graph.
inline void validateComplexScoreWeightedGridCost()
{
  const auto plan = makeTargetCostPlan(100, true, true);
  const auto old = estimateComplexScoreCost(plan, false, false);
  const auto grid = estimateComplexScoreCost(plan, false, true);
  if (grid.device_bytes - old.device_bytes != sizeof(ComplexScoreWeightedGridState) ||
      grid.host_bytes != old.host_bytes || grid.canonical_bytes != old.canonical_bytes ||
      grid.weighted_grid_bytes != sizeof(ComplexScoreWeightedGridState) ||
      estimateComplexOrbitalVJPCost(plan).weighted_grid_bytes)
    throw std::runtime_error("Weighted grid changed canonical/raw cost or exact private-state bytes");
  for (const auto scope : {ComplexProductsScope::COMPLETE, ComplexProductsScope::SCORE_ECP_ONLY})
  {
    const auto products = estimateComplexProductsCost(plan, 1, 4, 1, 4096, ComplexProductsHostOutput::CHUNKED, scope);
    if (products.score.weighted_grid_bytes !=
        (configuredComplexScoreWeightedGrid() ? sizeof(ComplexScoreWeightedGridState) : 0) ||
        (scope == ComplexProductsScope::SCORE_ECP_ONLY && (products.kinetic.device_bytes || products.matrix_work)))
      throw std::runtime_error("Weighted grid did not reach exact native Products scope admission");
  }
  for (const auto n : {99, 100, 300, 301})
  {
    const auto candidate = makeTargetCostPlan(n, true, true);
    if (complexScoreWeightedGridEligible(candidate) != (n >= 100 && n <= 300) ||
        estimateComplexScoreCost(candidate, false, true).weighted_grid_bytes !=
            (n >= 100 && n <= 300 ? sizeof(ComplexScoreWeightedGridState) : 0))
      throw std::runtime_error("Weighted grid range or cold fallback cost widened");
  }
  if (complexScoreWeightedGridEligible(makeTargetCostPlan(100, false)) ||
      complexScoreWeightedGridEligible(makeTargetCostPlan(100, true)))
    throw std::runtime_error("Weighted grid admitted open/real plans");
  std::cout << "PASS weighted grid exact_device_delta=" << sizeof(ComplexScoreWeightedGridState)
            << " host_delta=0 aggregate_complete_and_ecp=1 raw_and_fallback_zero=1\n";
}

/// Validate one small actual score owner; large-N checks above allocate only primitive buffers.
inline void validateComplexScoreWeightedOwner()
{
  const auto descriptor = makeTargetCostPlan(64, true, true);
  const auto initialized = initializePsiFormerParameters(descriptor.modelShape(), 1729,
      PSIFORMER_COMPLEX_PERIODIC_V1, descriptor.environment());
  const PsiFormerExecutionPlan plan(descriptor.modelShape(), initialized.layouts, descriptor.environment());
  std::vector<double> nuclei(3 * plan.modelShape().nuclei, 0.25);
  compute::Queue<PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, initialized.values, nuclei);
  const auto cost = estimateComplexScoreCost(plan);
  auto& metrics = deviceMemoryMetrics(0);
  const auto before = metrics.snapshot();
  bool denied = false;
  try { (void)PreparedDeviceComplexScore::prepare(image, queue, blas, cost.device_bytes - 1); }
  catch (const std::length_error&) { denied = true; }
  if (!denied || metrics.snapshot().allocation_calls != before.allocation_calls)
    throw std::runtime_error("Weighted score cost-minus-one allocated");
  auto owner = PreparedDeviceComplexScore::prepare(image, queue, blas, cost.device_bytes);
  if (metrics.snapshot().live_bytes - before.live_bytes != cost.device_bytes ||
      owner->usesCooperativeWeightedInverses() != configuredComplexScoreWeightedInversesCooperative())
    throw std::runtime_error("Weighted score cold selector/inventory differs from admission");
  std::cout << "PASS weighted inverse owner Ne=64 exact_device_bytes=" << cost.device_bytes
            << " additional_numeric_owners=0 cost_minus_one=1 selected_cooperative="
            << owner->usesCooperativeWeightedInverses() << '\n';
}
/// Bound timing to three primitive shapes; uploads/restoration are outside GPU events.
inline void measureComplexScoreWeightedInverses()
{
  compute::Queue<PlatformKind::CUDA> queue;
#ifdef QMC_CUDA2HIP
  using Event = hipEvent_t;
  const auto create = [](Event* e) { return hipEventCreate(e); };
  const auto record = [](Event e, Stream s) { return hipEventRecord(e, s); };
  const auto elapsed = [](float* t, Event a, Event b) { return hipEventElapsedTime(t, a, b); };
  const auto destroy = [](Event e) { return hipEventDestroy(e); };
#else
  using Event = cudaEvent_t;
  const auto create = [](Event* e) { return cudaEventCreate(e); };
  const auto record = [](Event e, Stream s) { return cudaEventRecord(e, s); };
  const auto elapsed = [](float* t, Event a, Event b) { return cudaEventElapsedTime(t, a, b); };
  const auto destroy = [](Event e) { return cudaEventDestroy(e); };
#endif
  Event begin, end;
  checkRuntime(create(&begin), "weighted inverse timing begin");
  checkRuntime(create(&end), "weighted inverse timing end");
  for (const auto dimensions : {std::array<std::size_t, 2>{64, 1}, {100, 16}, {200, 16}})
  {
    const auto n = dimensions[0], d = dimensions[1], count = d * n * n;
    std::vector<cuDoubleComplex> input(count, make_cuDoubleComplex(0.375, -0.125));
    std::vector<ComplexValueChannel> metadata(d, {make_cuDoubleComplex(1, 0), 0, 0});
    DeviceBuffer<cuDoubleComplex> inverse(count);
    DeviceBuffer<ComplexValueChannel> channels(d);
    DeviceBuffer<ComplexValueResult> amplitude(1);
    DeviceBuffer<double> cusp(1);
    DeviceBuffer<std::uint64_t> status(1);
    const double zero = 0;
    channels.upload(metadata.data(), d); cusp.upload(&zero, 1);
    for (int repetition = 0; repetition < 4; ++repetition)
      for (bool cooperative : {false, true})
      {
        const auto clear = std::numeric_limits<std::uint64_t>::max();
        const ComplexValueResult initial{};
        inverse.upload(input.data(), count); status.upload(&clear, 1); amplitude.upload(&initial, 1);
        const auto launch = cooperative ? launchComplexScoreWeightedInversesCooperative :
            launchComplexScoreWeightedInverses;
        checkRuntime(record(begin, queue.getNative()), "weighted inverse timing record");
        checkRuntime(launch(queue.getNative(), d, n, channels.data(), inverse.data(), cusp.data(),
            amplitude.data(), status.data()), "timed weighted inverse");
        checkRuntime(record(end, queue.getNative()), "weighted inverse timing record");
        queue.sync();
        float ms = 0;
        checkRuntime(elapsed(&ms, begin, end), "weighted inverse GPU elapsed");
        std::uint64_t code;
        status.download(&code, 1);
        if (code != clear) throw std::runtime_error("Timed weighted inverse failed admission");
        std::cout << std::setprecision(9) << "MEASURE weighted_inverse Ne=" << n << " D=" << d
                  << " cooperative=" << cooperative << " repetition=" << repetition
                  << " measured=" << (repetition != 0) << " gpu_ms=" << ms
                  << " scope=kernel_including_launch uploads_excluded=1\n";
      }
  }
  checkRuntime(destroy(begin), "weighted inverse timing destroy");
  checkRuntime(destroy(end), "weighted inverse timing destroy");
}
} // namespace qmcplusplus::psiformer::device
#endif
