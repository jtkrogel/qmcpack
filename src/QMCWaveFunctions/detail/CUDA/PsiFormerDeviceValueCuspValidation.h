//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_VALUE_CUSP_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_VALUE_CUSP_VALIDATION_H

#include "PsiFormerDeviceResources.h"
#include "PsiFormerDeviceSpatial.h"
#include "Platforms/CUDA/QueueCUDA.hpp"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOpenSpatial.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qmcplusplus::psiformer::device
{
/// Native scalar/device parity, canonical hazards and exact padding without a global pair tape.
inline void validateCooperativeValueCusp(compute::Queue<PlatformKind::CUDA>& queue)
{
  std::size_t cases = 0;
  std::size_t legacy_radius_discrepancies = 0;
  for (const auto mode : {SpatialJetMode::VALUE_ONLY, SpatialJetMode::ACTIVE, SpatialJetMode::FULL_VGL})
    for (const std::size_t ne : {63, 64, 100, 129, 200})
    {
      if (mode != SpatialJetMode::VALUE_ONLY && ne != 64)
        continue;
      constexpr std::size_t batch = 2;
      const auto planes = mode == SpatialJetMode::VALUE_ONLY ? 1 :
          mode == SpatialJetMode::ACTIVE ? 4 : 1 + 4 * ne;
      const auto layout = makeOpenCuspJetLayout(batch, ne, ne / 2, mode,
          5, 5 * ne + 11, 3, planes * 3 + 19);
      const auto span = spatialJetSpanElements(layout.output) + 7;
      std::vector<double> original(batch * layout.position_configuration_stride, 23.0);
      for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t e = 0; e < ne; ++e)
          for (std::size_t d = 0; d < 3; ++d)
            original[b * layout.position_configuration_stride + 5 * e + d] = .4 + .025 * e + .1 * d + .03 * b;
      std::vector<unsigned char> logical(span, 0);
      for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t plane = 0; plane < layout.output.plane_count; ++plane)
          logical[layout.output.uncheckedPlaneOffset(b, plane, 0)] = 1;
      std::vector<double> positions = original, initial(span, 97.0), actual(span), scalar(span), host(span);
      std::vector<std::size_t> active{ne - 1, 0};
      std::vector<OpenSpatialStatus> actual_status(batch), scalar_status(batch), host_status(batch);
      DeviceBuffer<double> coordinates(positions.size()), candidate_output(span), scalar_output(span);
      DeviceBuffer<std::size_t> active_device(batch);
      DeviceBuffer<OpenSpatialStatus> candidate_status(batch), old_status(batch);
      for (int variant = 0; variant < 14; ++variant)
      {
        positions = original;
        active = {ne - 1, 0};
        double same = .7, opposite = 1.3;
        if (variant == 1) active[1] = ne;
        if (variant == 2) positions[5 * 3 + 1] = std::numeric_limits<double>::quiet_NaN();
        if (variant == 3) positions[5 * 3 + 1] = std::numeric_limits<double>::infinity();
        if (variant == 4 || variant == 5)
          for (std::size_t d = 0; d < 3; ++d) positions[5 * (ne - 1) + d] = positions[d];
        if (variant == 5 || variant == 8) same = opposite = 0.0;
        if (variant == 6) same = std::numeric_limits<double>::quiet_NaN();
        if (variant == 7) opposite = std::numeric_limits<double>::infinity();
        if (variant == 9) { same = -.07; opposite = -.11; }
        if (variant == 10) same = opposite = 1e308;
        // Finite coordinates may overflow the radius, yet the native VALUE term remains finite.
        if (variant == 11) positions[0] = 1e308;
        if (variant == 13) { same = 1e-200; opposite = 1e-150; }
        coordinates.upload(positions.data(), positions.size());
        active_device.upload(active.data(), active.size());
        candidate_output.upload(initial.data(), initial.size());
        scalar_output.upload(initial.data(), initial.size());
        host = initial;
        for (std::size_t b = 0; b < batch; ++b)
          host_status[b] = open_spatial::buildOpenCuspJetsForConfiguration(
              layout, positions.data(), b, active[b], same, opposite, host.data());
        const auto before = deviceMemoryMetrics(0).snapshot();
        checkRuntime(launchOpenCuspJets(queue.getNative(), layout, coordinates.data(), active_device.data(),
            same, opposite, scalar_output.data(), old_status.data()), "scalar VALUE cusp oracle");
        checkRuntime(launchOpenCuspValuesCooperative(queue.getNative(), layout, coordinates.data(), active_device.data(),
            same, opposite, candidate_output.data(), candidate_status.data()), "cooperative VALUE cusp");
        queue.sync();
        const auto after = deviceMemoryMetrics(0).snapshot();
        if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls ||
            before.live_bytes != after.live_bytes)
          throw std::runtime_error("Cooperative VALUE cusp changed tracked owners");
        candidate_output.download(actual.data(), actual.size());
        scalar_output.download(scalar.data(), scalar.size());
        candidate_status.download(actual_status.data(), batch);
        old_status.download(scalar_status.data(), batch);
        if (actual_status != scalar_status)
          throw std::runtime_error("Cooperative VALUE cusp canonical GPU status differs");
        for (std::size_t b = 0; b < batch; ++b)
          if (scalar_status[b] != host_status[b])
          {
            // Unchanged HIP scalar production rejects this radius-overflow fixture,
            // while the native host accepts it. Preserve the GPU domain, not a new fix.
            if (variant != 11 || b != 0 || scalar_status[b] != OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT ||
                host_status[b] != OpenSpatialStatus::REGULAR)
              throw std::runtime_error("VALUE cusp host status differs outside legacy radius overflow");
            ++legacy_radius_discrepancies;
            std::cout << "LEGACY native_cusp_radius_overflow GPU_rejects_HOST_accepts Ne=" << ne
                      << " mode=" << static_cast<int>(mode) << " candidate_reference_status_bits_exact\n";
          }
        if (variant == 1 && actual_status[1] != OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT)
          throw std::runtime_error("VALUE cusp invalid-active fixture lost premise");
        if (mode == SpatialJetMode::VALUE_ONLY && (variant == 4 || variant == 8) &&
            actual_status[0] != OpenSpatialStatus::REGULAR)
          throw std::runtime_error("Legal zero-distance/zero-alpha VALUE cusp was rejected");
        for (std::size_t i = 0; i < span; ++i)
        {
          bool failed = false;
          for (std::size_t b = 0; b < batch; ++b)
            if (i >= b * layout.output.configuration_stride &&
                i < b * layout.output.configuration_stride +
                    (layout.output.plane_count - 1) * layout.output.plane_stride + 1)
              failed = actual_status[b] != OpenSpatialStatus::REGULAR;
          if (!logical[i] && (std::memcmp(&actual[i], &initial[i], sizeof(double)) ||
              std::memcmp(&scalar[i], &initial[i], sizeof(double))))
            throw std::runtime_error("VALUE cusp modified output padding");
          const bool exact = !logical[i] || failed || ne < 64 || mode != SpatialJetMode::VALUE_ONLY || scalar[i] == 0.0;
          if (exact)
          {
            if (std::memcmp(&actual[i], &scalar[i], sizeof(double)))
              throw std::runtime_error("VALUE cusp canonical fallback/partial/signed-zero bits differ");
          }
          else if (!device_math::isFiniteBinary64(actual[i]) || !device_math::isFiniteBinary64(scalar[i]) ||
              std::abs(actual[i] - scalar[i]) > 2e-12 * (1 + std::abs(scalar[i])))
            throw std::runtime_error("VALUE cusp scalar device parity differs");
          if (!failed && (!device_math::isFiniteBinary64(host[i]) ||
              std::abs(actual[i] - host[i]) > 2e-12 * (1 + std::abs(host[i]))))
            throw std::runtime_error("VALUE cusp native host parity differs");
        }
        ++cases;
      }
    }
  if (cases != 98)
    throw std::runtime_error("VALUE cusp native case count differs");
  std::cout << "PASS cooperative_VALUE_cusp native_cases=" << cases
            << " scalar_pair_sum_order VALUE_zero_radius finite_radius_overflow fallback_modes"
            << " exact_status_partial_padding_signed_zero recovery zero_tracked_alloc"
            << " legacy_radius_overflow_gpu_host_discrepancies=" << legacy_radius_discrepancies << "\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
