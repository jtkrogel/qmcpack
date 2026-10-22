//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_CUSP_STATE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_CUSP_STATE_VALIDATION_H

#include "PsiFormerDeviceResources.h"
#include "PsiFormerDeviceSpatial.h"
#include "Platforms/CUDA/QueueCUDA.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmcplusplus::psiformer::device
{
/// Compare native target launches with cached gathers, including failure replay and padding.
inline void validateCuspStateGather(compute::Queue<PlatformKind::CUDA>& queue)
{
  const auto cell = periodic_geometry::makeCell(
      {{{5.1, .2, .1}, {.4, 4.8, .3}, {.2, .5, 5.3}}}, {{true, true, true}});
  std::size_t cases = 0;
  const auto exercise = [&](const OpenCuspJetLayout& full, const std::vector<double>& positions,
                            const periodic_geometry::Cell* geometry, double alpha,
                            std::size_t begin, std::size_t count, bool expect_failed_full) {
    auto target = full;
    target.output = withSpatialTargetRange(full.output, begin, count);
    const auto batch = full.output.configuration_count;
    const auto nf = spatialJetSpanElements(full.output) + 1;
    const auto nt = spatialJetSpanElements(target.output) + 1;
    DeviceBuffer<double> coordinates(positions.size()), cache(nf), canonical(nt), gathered(nt);
    DeviceBuffer<std::size_t> active(batch);
    DeviceBuffer<OpenSpatialStatus> cache_status(batch), canonical_status(batch), gather_status(batch);
    coordinates.upload(positions.data(), positions.size());
    const std::vector<std::size_t> zero(batch, 0);
    active.upload(zero.data(), batch);
    const std::vector<double> full_sentinel(nf, 791.0), target_sentinel(nt, 791.0);
    cache.upload(full_sentinel.data(), nf);
    canonical.upload(target_sentinel.data(), nt);
    const auto launch = [&](const OpenCuspJetLayout& layout, double* output, OpenSpatialStatus* status) {
      return geometry ? launchPeriodicCuspJets(queue.getNative(), layout, *geometry,
          coordinates.data(), active.data(), alpha, alpha, output, status)
          : launchOpenCuspJets(queue.getNative(), layout, coordinates.data(), active.data(),
              alpha, alpha, output, status);
    };
    checkRuntime(launch(full, cache.data(), cache_status.data()), "cusp validation full");
    checkRuntime(launch(target, canonical.data(), canonical_status.data()), "cusp validation target");
    queue.sync();
    std::vector<double> expected(nt), actual(nt), full_cache(nf);
    std::vector<std::size_t> numeric(nt, 0);
    for (std::size_t b = 0; b < batch; ++b)
      for (std::size_t plane = 0; plane < target.output.plane_count; ++plane)
        numeric[target.output.uncheckedPlaneOffset(b, plane, 0)] = b + 1;
    std::vector<OpenSpatialStatus> full_status(batch), expected_status(batch), actual_status(batch);
    canonical.download(expected.data(), nt);
    canonical_status.download(expected_status.data(), batch);
    cache_status.download(full_status.data(), batch);
    cache.download(full_cache.data(), nf);
    for (std::size_t b = 0; b < batch; ++b)
      if (expected_status[b] != OpenSpatialStatus::REGULAR ||
          (full_status[b] == OpenSpatialStatus::REGULAR) == expect_failed_full)
        throw std::runtime_error("Cusp validation fixture has unexpected native status");
    for (const bool force_replay : {false, true})
    {
      if (force_replay)
      {
        const std::vector<OpenSpatialStatus> failed(batch, OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT);
        cache_status.upload(failed.data(), batch);
      }
      gathered.upload(target_sentinel.data(), nt);
      checkRuntime(launchGatherCuspTargets(queue.getNative(), full, target, coordinates.data(),
          active.data(), alpha, alpha, cache.data(), cache_status.data(), gathered.data(),
          gather_status.data(), geometry), "cusp validation gather");
      queue.sync();
      gathered.download(actual.data(), nt);
      gather_status.download(actual_status.data(), batch);
      if (actual_status != expected_status)
        throw std::runtime_error("Cusp cached gather differs from canonical target status");
      for (std::size_t i = 0; i < nt; ++i)
        if (numeric[i] ? (!device_math::isFiniteBinary64(actual[i]) ||
              ((force_replay || full_status[numeric[i] - 1] != OpenSpatialStatus::REGULAR)
                  ? std::abs(actual[i] - expected[i]) > 2e-12 * (1 + std::abs(expected[i]))
                  : actual[i] != expected[i]))
                       : actual[i] != expected[i])
          throw std::runtime_error("Cusp cached gather differs from canonical native target/padding");
      // Canonical replay is compiled in another kernel context and may differ by
      // a few final-rounding bits. Cache gathers themselves must copy exact bits.
      if (!force_replay)
        for (std::size_t b = 0; b < batch; ++b)
          if (full_status[b] == OpenSpatialStatus::REGULAR)
          {
            if (actual[target.output.uncheckedValueOffset(b, 0)] != full_cache[full.output.uncheckedValueOffset(b, 0)])
              throw std::runtime_error("Cusp gather changed cached value");
            for (std::size_t local = 0; local < count; ++local)
            {
              for (std::size_t d = 0; d < 3; ++d)
                if (actual[target.output.uncheckedGradientOffset(b, 3 * local + d, 0)] !=
                    full_cache[full.output.uncheckedGradientOffset(b, 3 * (begin + local) + d, 0)])
                  throw std::runtime_error("Cusp gather changed cached gradient");
              if (actual[target.output.uncheckedLaplacianOffset(b, local, 0)] !=
                  full_cache[full.output.uncheckedLaplacianOffset(b, begin + local, 0)])
                throw std::runtime_error("Cusp gather changed cached Laplacian");
            }
          }
      ++cases;
    }
  };
  const auto full = makeOpenCuspJetLayout(2, 5, 2, SpatialJetMode::FULL_VGL, 4, 24, 3, 70);
  std::vector<double> positions(48, 913.0);
  for (std::size_t b = 0; b < 2; ++b)
    for (std::size_t e = 0; e < 5; ++e)
      for (std::size_t d = 0; d < 3; ++d)
        positions[24 * b + 4 * e + d] = .27 + .31 * e + .13 * b + .17 * d;
  for (const bool periodic : {false, true})
    for (const auto interval : {std::pair<std::size_t, std::size_t>{0, 2}, {2, 2}, {4, 1}, {1, 1}, {0, 5}})
      exercise(full, positions, periodic ? &cell : nullptr, .8, interval.first, interval.second, false);
  // The unrequested near pair's radial second derivative is nonfinite, while
  // the original requested electron2 value/derivatives are still regular.
  exercise(makeOpenCuspJetLayout(1, 3, 1, SpatialJetMode::FULL_VGL),
      {0, 0, 0, 1e-150, 0, 0, 1, .2, .3}, nullptr, 1e-200, 2, 1, true);
  std::cout << "PASS " << cases << " native cusp gather cases: padding, ranges, replay and failure domain\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
