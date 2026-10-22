//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_VALUE_ORBITAL_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_VALUE_ORBITAL_VALIDATION_H
#include "PsiFormerDeviceResources.h"
#include "PsiFormerDeviceSpatial.h"
#include "Platforms/CUDA/QueueCUDA.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qmcplusplus::psiformer::device
{
/// Compare tiled/native scalar heads, exact hazard scratch and padding, without tracked allocations.
inline void validateTiledValueOrbitals(compute::Queue<PlatformKind::CUDA>& queue)
{
  std::size_t cases = 0, helper_guards = 0;
  for (const auto mode : {SpatialJetMode::VALUE_ONLY, SpatialJetMode::ACTIVE, SpatialJetMode::FULL_VGL})
    for (const std::size_t ne : {63, 64, 65, 100, 129, 200})
    {
      if (mode != SpatialJetMode::VALUE_ONLY && ne != 64) continue;
      for (const std::size_t spin : {std::size_t(0), std::size_t(1), ne / 2, ne})
      {
        if (mode != SpatialJetMode::VALUE_ONLY && spin != ne / 2) continue;
        constexpr std::size_t batch = 2, dets = 2, nuclei_count = 3;
        // Non-packed fixtures exercise both spin banks, tile tails and independent plane strides.
        const std::size_t width = ne == 100 ? 256 : ne == 63 ? 1 : ne == 64 ? 7 : ne == 65 ? 8 : ne == 129 ? 9 : 17;
        auto layout = makeOpenOrbitalJetLayout(batch, dets, ne, nuclei_count, width, spin, mode,
            width + 3, ne + 2, ne * (ne + 2) + 5, ne + 3, dets * (ne + 3) + 7,
            nuclei_count + 2, ne * (nuclei_count + 2) + 7, 5, ne * 5 + 13, 5);
        layout.features.plane_stride += 7;
        layout.features.configuration_stride = layout.features.plane_count * layout.features.plane_stride + 11;
        layout.orbitals.plane_stride += 13;
        layout.orbitals.configuration_stride = layout.orbitals.plane_count * layout.orbitals.plane_stride + 17;
        validateOpenOrbitalJetLayout(layout);
        std::vector<double> feature(spatialJetSpanElements(layout.features) + 7, 31.0);
        for (std::size_t b = 0; b < batch; ++b)
          for (std::size_t p = 0; p < layout.features.plane_count; ++p)
            for (std::size_t row = 0; row < ne; ++row)
              for (std::size_t f = 0; f < width; ++f)
                feature[layout.features.uncheckedPlaneOffset(b, p, row * layout.feature_row_stride + f)] =
                    .001 * (1 + (f + row + b + p) % 19);
        std::vector<double> positions(batch * layout.position_configuration_stride, 43.0), nuclei(5 * nuclei_count, 47.0);
        for (std::size_t b = 0; b < batch; ++b)
          for (std::size_t row = 0; row < ne; ++row)
            for (std::size_t d = 0; d < 3; ++d)
              positions[b * layout.position_configuration_stride + 5 * row + d] = .3 + .01 * row + .1 * d + .03 * b;
        for (std::size_t n = 0; n < nuclei_count; ++n)
          for (std::size_t d = 0; d < 3; ++d) nuclei[5 * n + d] = -.2 - .05 * n - .02 * d;
        std::vector<double> up(openOrbitalBackflowSpan(layout), 59.0), down(up.size(), 61.0);
        std::vector<double> pi_up(openOrbitalEnvelopeSpan(layout), 67.0), pi_down(pi_up.size(), 71.0),
            zeta_up(pi_up.size(), 73.0), zeta_down(pi_up.size(), 79.0);
        for (std::size_t determinant = 0; determinant < dets; ++determinant)
          for (std::size_t orbital = 0; orbital < ne; ++orbital)
          {
            for (std::size_t f = 0; f < width; ++f)
            {
              const auto i = open_orbital::weightOffset(layout, f, determinant, orbital);
              up[i] = .002 * (int((f + orbital) % 13) - 6);
              down[i] = -.003 * (int((f + 2 * orbital) % 11) - 5);
            }
            for (std::size_t n = 0; n < nuclei_count; ++n)
            {
              const auto i = open_orbital::envelopeOffset(layout, determinant, orbital, n);
              pi_up[i] = .2 + .01 * n; pi_down[i] = -.1 - .02 * n;
              zeta_up[i] = -.3 - .01 * orbital; zeta_down[i] = .4 + .02 * n;
            }
          }
        const auto span = spatialJetSpanElements(layout.orbitals) + 7;
        std::vector<double> initial(span, 83.0), actual(span), scalar(span), host(span);
        const auto count = batch * dets * ne * ne;
        std::vector<int> owner(span, -1);
        for (std::size_t b = 0; b < batch; ++b)
          for (std::size_t det = 0; det < dets; ++det)
            for (std::size_t row = 0; row < ne; ++row)
              for (std::size_t col = 0; col < ne; ++col)
                for (std::size_t p = 0; p < layout.orbitals.plane_count; ++p)
                  owner[layout.orbitals.uncheckedPlaneOffset(b, p, open_orbital::outputElement(layout, det, row, col))] =
                      int(((b * dets + det) * ne + row) * ne + col);
        DeviceBuffer<double> df(feature.size()), dp(positions.size()), dn(nuclei.size()), du(up.size()), dd(down.size()),
            dpu(pi_up.size()), dpd(pi_down.size()), dzu(zeta_up.size()), dzd(zeta_down.size()),
            candidate(span), reference(span);
        DeviceBuffer<std::size_t> da(batch);
        DeviceBuffer<OpenSpatialStatus> cs(count), rs(count);
        std::vector<OpenSpatialStatus> a_status(count), r_status(count), h_status(count);
        // Each variant starts from regular data; the final regular case proves recovery.
        for (int variant = 0; variant < 13; ++variant)
        {
          if (variant >= 10 && (mode != SpatialJetMode::VALUE_ONLY || ne != 64 || spin != 0)) continue;
          auto f = feature, pos = positions, u = up, d = down, pu = pi_up, pd = pi_down, zu = zeta_up;
          std::vector<std::size_t> active{ne - 1, 0};
          if (variant == 1) active[1] = ne;
          if (variant == 2)
            f[layout.features.uncheckedValueOffset(0, 3 * layout.feature_row_stride)] =
                std::numeric_limits<double>::quiet_NaN();
          if (variant == 3) u[open_orbital::weightOffset(layout, 0, 0, 0)] = std::numeric_limits<double>::infinity();
          if (variant == 4) pd[open_orbital::envelopeOffset(layout, 0, 0, 0)] = std::numeric_limits<double>::quiet_NaN();
          if (variant == 5) for (std::size_t k = 0; k < 3; ++k) pos[k] = nuclei[k];
          if (variant == 6) zu[open_orbital::envelopeOffset(layout, 0, 0, 0)] = std::numeric_limits<double>::infinity();
          if (variant == 7) for (auto& v : f) v = 0.0;
          if (variant == 8) pos[1] = std::numeric_limits<double>::infinity();
          if (variant >= 10)
          {
            // A few explicit cancellation/underflow fixtures, not another shape sweep.
            for (auto& v : f) v = variant == 10 ? 1.0 : std::numeric_limits<double>::min();
            for (auto& v : d) v = 0.0;
            for (std::size_t det = 0; det < dets; ++det)
              for (std::size_t col = 0; col < ne; ++col)
              {
                d[open_orbital::weightOffset(layout, 0, det, col)] = variant == 10 ? 1.0 :
                    variant == 11 ? .25 : std::numeric_limits<double>::min();
                if (variant == 10) d[open_orbital::weightOffset(layout, 1, det, col)] = -1.0;
              }
          }
          df.upload(f.data(), f.size()); dp.upload(pos.data(), pos.size()); dn.upload(nuclei.data(), nuclei.size());
          du.upload(u.data(), u.size()); dd.upload(d.data(), d.size());
          dpu.upload(pu.data(), pu.size()); dpd.upload(pd.data(), pd.size());
          dzu.upload(zu.data(), zu.size()); dzd.upload(zeta_down.data(), zeta_down.size()); da.upload(active.data(), batch);
          candidate.upload(initial.data(), span); reference.upload(initial.data(), span); host = initial;
          // The native host and unchanged scalar GPU map are independent numerical oracles.
          for (std::size_t b = 0; b < batch; ++b)
            for (std::size_t det = 0; det < dets; ++det)
              for (std::size_t row = 0; row < ne; ++row)
                for (std::size_t col = 0; col < ne; ++col)
                  h_status[((b * dets + det) * ne + row) * ne + col] = open_orbital::buildElement(layout,
                      f.data(), pos.data(), nuclei.data(), u.data(), d.data(), pu.data(), pd.data(),
                      zu.data(), zeta_down.data(),
                      b, active[b], det, row, col, host.data());
          const auto before = deviceMemoryMetrics(0).snapshot();
          checkRuntime(launchOpenOrbitalJets(queue.getNative(), layout, df.data(), dp.data(), dn.data(),
              du.data(), dd.data(), dpu.data(), dpd.data(), dzu.data(), dzd.data(), da.data(),
              reference.data(), rs.data()), "native scalar orbital oracle");
          checkRuntime(launchOpenOrbitalValuesTiled(queue.getNative(), layout, df.data(), dp.data(), dn.data(),
              du.data(), dd.data(), dpu.data(), dpd.data(), dzu.data(), dzd.data(), da.data(),
              candidate.data(), cs.data()), "tiled VALUE orbital");
          queue.sync();
          const auto after = deviceMemoryMetrics(0).snapshot();
          if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls ||
              before.live_bytes != after.live_bytes)
            throw std::runtime_error("Tiled VALUE orbital changed tracked owners");
          candidate.download(actual.data(), span); reference.download(scalar.data(), span);
          cs.download(a_status.data(), count); rs.download(r_status.data(), count);
          if (a_status != r_status || a_status != h_status) throw std::runtime_error("Tiled VALUE orbital status differs");
          // Failures/fallbacks/zeros are bitwise contracts; only regular recomputations use the native bound.
          for (std::size_t i = 0; i < span; ++i)
          {
            const bool failed = owner[i] >= 0 && a_status[owner[i]] != OpenSpatialStatus::REGULAR;
            const bool exact = owner[i] < 0 || failed || ne < 64 || mode != SpatialJetMode::VALUE_ONLY || scalar[i] == 0.0;
            if (owner[i] < 0 && (std::memcmp(&actual[i], &initial[i], sizeof(double)) ||
                std::memcmp(&scalar[i], &initial[i], sizeof(double))))
              throw std::runtime_error("Tiled VALUE orbital modified padding");
            if (exact && std::memcmp(&actual[i], &scalar[i], sizeof(double)))
            {
              unsigned long long actual_bits = 0, scalar_bits = 0;
              std::memcpy(&actual_bits, &actual[i], sizeof(double));
              std::memcpy(&scalar_bits, &scalar[i], sizeof(double));
              std::cerr << "ORBITAL_DIAGNOSTIC Ne=" << ne << " spin=" << spin << " mode=" << int(mode)
                        << " variant=" << variant << " offset=" << i << " owner=" << owner[i]
                        << " status=" << (owner[i] < 0 ? -1 : int(a_status[owner[i]]))
                        << " actual_bits=" << std::hex << actual_bits << " scalar_bits=" << scalar_bits << std::dec << '\n';
              throw std::runtime_error("Tiled VALUE orbital canonical partial/fallback bits differ");
            }
            if (!exact && (!device_math::isFiniteBinary64(actual[i]) ||
                std::abs(actual[i] - scalar[i]) > 2e-12 * (1 + std::abs(scalar[i]))))
              throw std::runtime_error("Tiled VALUE orbital scalar finite parity differs");
            if (!failed && (!device_math::isFiniteBinary64(host[i]) ||
                std::abs(actual[i] - host[i]) > 2e-12 * (1 + std::abs(host[i]))))
              throw std::runtime_error("Tiled VALUE orbital host finite parity differs");
          }
          // Unsupported precontracted-helper combinations must still clear every logical plane.
          if (variant == 0)
          {
            const double precontracted = .3;
            for (int conflict = 0; conflict < 3; ++conflict)
            {
              if (mode == SpatialJetMode::VALUE_ONLY && conflict == 0) continue;
              auto scratch = initial;
              const auto status = open_orbital::buildElement(layout, f.data(), pos.data(), nuclei.data(),
                  u.data(), d.data(), pu.data(), pd.data(),
                  zu.data(), zeta_down.data(), 0, 0, 0, 0, 0, scratch.data(), nullptr, nullptr, nullptr,
                  conflict == 1 ? host.data() : nullptr, conflict == 2, &precontracted);
              if (status != OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT)
                throw std::runtime_error("Precontracted orbital helper guard differs");
              for (std::size_t p = 0; p < layout.orbitals.plane_count; ++p)
                if (scratch[layout.orbitals.uncheckedPlaneOffset(0, p, 0)] != 0.0)
                  throw std::runtime_error("Precontracted helper lost output clearing");
              ++helper_guards;
            }
          }
          ++cases;
        }
      }
    }
  if (cases != 263 || helper_guards != 54) throw std::runtime_error("Tiled orbital native case count differs");
  std::cout << "PASS tiled_VALUE_orbitals native_cases=" << cases << " helper_guards=" << helper_guards
            << " spin_crossing_polarized feature_row_column_tails exact_status_partial_padding_signed_zero recovery zero_tracked_alloc\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
