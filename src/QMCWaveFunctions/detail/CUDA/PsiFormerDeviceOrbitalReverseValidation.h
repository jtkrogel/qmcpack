//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_ORBITAL_REVERSE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_ORBITAL_REVERSE_VALIDATION_H

#include "PsiFormerDeviceOpenKineticReverse.h"
#include "PsiFormerDeviceResources.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace qmcplusplus::psiformer::device
{

/// Compare every logical output and padding lane against the shared CPU leaf.
inline void validateParallelOrbitalReverse(compute::Queue<PlatformKind::CUDA>& queue,
    const periodic_geometry::Cell* cell)
{
  OpenOrbitalKineticReverseLayout layout;
  layout.forward = makeOpenOrbitalJetLayout(1, 2, 3, 2, 4, 2, SpatialJetMode::FULL_VGL);
  for (auto* jets : {&layout.forward.features, &layout.forward.orbitals})
  {
    *jets = withSpatialTargetRange(*jets, 1, 2);
    jets->plane_stride += 7;
    jets->configuration_stride = jets->plane_stride * jets->plane_count;
  }
  const auto weights = openOrbitalBackflowSpan(layout.forward);
  const auto envelopes = openOrbitalEnvelopeSpan(layout.forward);
  layout.backflow_up_response_offset = 5;
  layout.backflow_down_response_offset = layout.backflow_up_response_offset + weights + 3;
  layout.envelope_pi_up_response_offset = layout.backflow_down_response_offset + weights + 3;
  layout.envelope_pi_down_response_offset = layout.envelope_pi_up_response_offset + envelopes + 3;
  layout.envelope_zeta_up_response_offset = layout.envelope_pi_down_response_offset + envelopes + 3;
  layout.envelope_zeta_down_response_offset = layout.envelope_zeta_up_response_offset + envelopes + 3;
  layout.response_count = layout.envelope_zeta_down_response_offset + envelopes + 5;
  validateOpenOrbitalKineticReverseLayout(layout);
  const auto features = layout.forward.features.configuration_stride;
  const auto matrices = layout.forward.orbitals.configuration_stride;
  std::vector<double> feature(features), backflow(matrices), envelope(matrices), adjoint(matrices);
  std::vector<double> weight(weights), pi(envelopes), zeta(envelopes);
  for (std::size_t i = 0; i < features; ++i) feature[i] = 0.03 * (int(i % 11) - 5);
  for (std::size_t i = 0; i < matrices; ++i)
  {
    backflow[i] = 0.2 + 0.013 * (i % 13);
    envelope[i] = 0.1 + 0.021 * (i % 7);
    adjoint[i] = 0.017 * (int(i % 9) - 4);
  }
  for (std::size_t i = 0; i < weights; ++i) weight[i] = 0.011 * (int(i % 7) - 3);
  for (std::size_t i = 0; i < envelopes; ++i)
  {
    pi[i] = 0.3 + 0.01 * i;
    zeta[i] = i % 3 == 0 ? 0.0 : (i % 2 ? -0.7 : 0.6);
  }
  std::vector<double> positions{0.4, 0.2, 0.6, 0.8, -0.4, 0.3, 1.2, 0.7, -0.2};
  const std::vector<double> nuclei{0, 0, 0, 2, 1, 1};
  DeviceBuffer<double> df, db, de, da, dw, dpi, dz, dp, dn;
  DeviceBuffer<double> bbar(matrices), ebar(matrices), fbar(features), response(layout.response_count);
  DeviceBuffer<unsigned int> failure(1);
  DeviceBuffer<OpenKineticReverseStatus> status(1);
  const auto upload = [](DeviceBuffer<double>& buffer, const std::vector<double>& values) {
    buffer.reserve(values.size()); buffer.upload(values.data(), values.size());
  };
  upload(df, feature); upload(de, envelope); upload(dw, weight);
  upload(dpi, pi); upload(dz, zeta); upload(dn, nuclei);
  double max_error = 0.0;
  for (int fault = 0; fault < 4; ++fault)
  {
    auto input_backflow = backflow, input_adjoint = adjoint, input_positions = positions;
    if (fault == 1) input_backflow[0] = std::numeric_limits<double>::quiet_NaN();
    if (fault == 2) std::copy_n(nuclei.begin(), 3, input_positions.begin());
    if (fault == 3)
    {
      input_adjoint[0] = std::numeric_limits<double>::max();
      input_backflow[0] = 2.0;
    }
    std::vector<double> expected_b(matrices, 913), expected_e(matrices, 913), expected_f(features, 913);
    std::vector<double> expected_response(layout.response_count, 0.37);
    upload(bbar, expected_b); upload(ebar, expected_e); upload(fbar, expected_f); upload(response, expected_response);
    upload(db, input_backflow); upload(da, input_adjoint); upload(dp, input_positions);
    OpenKineticReverseStatus expected_status;
    open_kinetic_reverse::reverseOpenOrbitalKineticSerial(layout, feature.data(), input_positions.data(),
        nuclei.data(), weight.data(), weight.data(), pi.data(), pi.data(), zeta.data(), zeta.data(),
        input_backflow.data(), envelope.data(), input_adjoint.data(), expected_b.data(), expected_e.data(),
        expected_f.data(), expected_response.data(), &expected_status, cell);

    // No allocation or host synchronization is needed inside the numerical launch.
    checkRuntime(launchOpenOrbitalKineticReverse(queue.getNative(), layout, df.data(), dp.data(), dn.data(),
        dw.data(), dw.data(), dpi.data(), dpi.data(), dz.data(), dz.data(), db.data(), de.data(), da.data(),
        bbar.data(), ebar.data(), fbar.data(), response.data(), status.data(), cell, failure.data()),
        "focused parallel orbital reverse");
    queue.sync();
    OpenKineticReverseStatus actual_status;
    status.download(&actual_status, 1);
    if (actual_status != expected_status || (fault != 0 && actual_status == OpenKineticReverseStatus::REGULAR))
      throw std::runtime_error("Parallel orbital reverse single-fault status differs from CPU");
    const auto compare = [&](DeviceBuffer<double>& buffer, const std::vector<double>& expected) {
      std::vector<double> actual(expected.size()); buffer.download(actual.data(), actual.size());
      for (std::size_t i = 0; i < actual.size(); ++i)
      {
        const auto error = std::abs(actual[i] - expected[i]);
        max_error = std::max(max_error, error);
        if (!device_math::isFiniteBinary64(actual[i]) || error > 2e-12 * (1 + std::abs(expected[i])))
          throw std::runtime_error("Parallel orbital reverse CPU/padding/unrelated-P mismatch");
      }
    };
    compare(bbar, expected_b); compare(ebar, expected_e); compare(fbar, expected_f);
    compare(response, expected_response);
  }
  std::cout << "PASS parallel orbital leaf: CPU all outputs, K2 physical mapping, padding/unrelated P, "
               "zero zeta, nonfinite/coalescence/overflow; max_abs_difference=" << max_error << '\n';
}

} // namespace qmcplusplus::psiformer::device
#endif
