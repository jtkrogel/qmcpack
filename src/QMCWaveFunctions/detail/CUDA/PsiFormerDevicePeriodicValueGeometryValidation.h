//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_PERIODIC_VALUE_GEOMETRY_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_PERIODIC_VALUE_GEOMETRY_VALIDATION_H

#include "PsiFormerDeviceResources.h"
#include "PsiFormerDeviceSpatial.h"
#include "Platforms/CUDA/QueueCUDA.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qmcplusplus::psiformer::device
{
/// Full-buffer scalar GPU parity includes failed partial scratch, padding and recovery.
inline void validatePeriodicValueGeometry(compute::Queue<PlatformKind::CUDA>& queue)
{
  const auto cell = periodic_geometry::makeCell({{{9., .3, .1}, {0., 10., .2}, {.1, 0., 11.}}},
                                               {{true, true, true}});
  std::size_t cases = 0;
  double max_host_error = 0.;
  for (const std::size_t ne : {63, 64, 100, 200})
    for (const std::size_t batch : {1, 4})
    {
      const auto mode = SpatialJetMode::VALUE_ONLY;
      const auto feature = makePeriodicFeatureJetLayout(batch, ne, ne, ne / 2, mode,
          5, 5 * ne + 11, 7, ne * (7 * ne + 1) + 13, ne * (7 * ne + 1) + 31);
      const auto cusp = makeOpenCuspJetLayout(batch, ne, ne / 2, mode,
          5, 5 * ne + 11, 3, 19);
      std::vector<double> base(batch * feature.position_configuration_stride, 17.);
      std::vector<double> nuclei(ne * 7, 23.);
      for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t e = 0; e < ne; ++e)
          for (std::size_t d = 0; d < 3; ++d)
            base[b * feature.position_configuration_stride + 5 * e + d] =
                .13 + .037 * e + .17 * d + .03 * b;
      for (std::size_t n = 0; n < ne; ++n)
        for (std::size_t d = 0; d < 3; ++d)
          nuclei[7 * n + d] = .09 + .051 * n + .23 * d;
      DeviceBuffer<double> positions(base.size()), ions(nuclei.size());
      ions.upload(nuclei.data(), nuclei.size());
      DeviceBuffer<std::size_t> active_device(batch);
      DeviceBuffer<OpenSpatialStatus> old_status(batch), new_status(batch);
      for (bool features : {true, false})
      {
        const auto& output = features ? feature.output : cusp.output;
        const auto span = spatialJetSpanElements(output) + 7;
        std::vector<double> initial(span, 97.), old(span), candidate(span), host(span);
        DeviceBuffer<double> old_output(span), new_output(span);
        std::vector<OpenSpatialStatus> old_codes(batch), new_codes(batch), host_codes(batch);
        for (int variant = 0; variant < 9; ++variant)
        {
          auto coordinates = base;
          std::vector<std::size_t> active(batch, ne - 1);
          double same = .7, opposite = 1.3;
          if (variant == 1) active.back() = ne;
          if (variant == 2) coordinates[5 * 3 + 1] = std::numeric_limits<double>::quiet_NaN();
          if (variant == 3) coordinates[5 * (ne - 1)] = std::numeric_limits<double>::infinity();
          if (variant == 4)
            for (std::size_t d = 0; d < 3; ++d)
              if (features)
                coordinates[d] = nuclei[d]; // Exact EI radial limit.
              else
                coordinates[5 + d] = coordinates[d]; // Exact EE radial limit.
          if (variant == 5) same = std::numeric_limits<double>::infinity();
          if (variant == 6) same = opposite = 1e308;
          if (variant == 7)
            for (std::size_t d = 0; d < 3; ++d)
              coordinates[d] += cell.lattice[0][d];
          // The final good variant proves recovery after every hazard.
          positions.upload(coordinates.data(), coordinates.size());
          active_device.upload(active.data(), active.size());
          old_output.upload(initial.data(), span);
          new_output.upload(initial.data(), span);
          host = initial;
          for (std::size_t b = 0; b < batch; ++b)
            host_codes[b] = features ? open_spatial::buildOpenFeatureJetsForConfiguration(
                feature, coordinates.data(), nuclei.data(), b, active[b], host.data(), &cell) :
                open_spatial::buildOpenCuspJetsForConfiguration(
                    cusp, coordinates.data(), b, active[b], same, opposite, host.data(), &cell);
          const auto before = deviceMemoryMetrics(0).snapshot();
          if (features)
          {
            checkRuntime(launchPeriodicFeatureJets(queue.getNative(), feature, cell, positions.data(),
                ions.data(), active_device.data(), old_output.data(), old_status.data()), "periodic scalar features");
            checkRuntime(launchPeriodicFeatureValuesCooperative(queue.getNative(), feature, cell, positions.data(),
                ions.data(), active_device.data(), new_output.data(), new_status.data()), "periodic cooperative features");
          }
          else
          {
            checkRuntime(launchPeriodicCuspJets(queue.getNative(), cusp, cell, positions.data(), active_device.data(),
                same, opposite, old_output.data(), old_status.data()), "periodic scalar cusp");
            checkRuntime(launchPeriodicCuspValuesCooperative(queue.getNative(), cusp, cell, positions.data(),
                active_device.data(), same, opposite, new_output.data(), new_status.data()), "periodic cooperative cusp");
          }
          queue.sync();
          const auto after = deviceMemoryMetrics(0).snapshot();
          if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls ||
              before.live_bytes != after.live_bytes)
            throw std::runtime_error("Periodic VALUE geometry changed tracked owners");
          old_output.download(old.data(), span);
          new_output.download(candidate.data(), span);
          old_status.download(old_codes.data(), batch);
          new_status.download(new_codes.data(), batch);
          if (old_codes != new_codes || old_codes != host_codes)
            throw std::runtime_error("Periodic VALUE geometry status oracle differs");
          if (std::memcmp(old.data(), candidate.data(), span * sizeof(double)))
            throw std::runtime_error("Periodic VALUE geometry scalar GPU bits/padding/partial scratch differ");
          for (std::size_t b = 0; b < batch; ++b)
            if (old_codes[b] == OpenSpatialStatus::REGULAR)
              for (std::size_t e = 0; e < output.element_count; ++e)
              {
                const auto i = output.uncheckedValueOffset(b, e);
                const double error = std::abs(candidate[i] - host[i]);
                max_host_error = std::max(max_host_error, error);
                if (!device_math::isFiniteBinary64(candidate[i]) || error > 3e-12 * (1. + std::abs(host[i])))
                  throw std::runtime_error("Periodic VALUE geometry CPU oracle differs");
              }
          ++cases;
        }
      }
    }
  std::cout << "PASS periodic_VALUE_geometry cases=" << cases << " CPU_max_abs=" << max_host_error
            << " scalar_GPU_full_buffer_bits_exact padding failure_replay recovery warm_alloc0\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
