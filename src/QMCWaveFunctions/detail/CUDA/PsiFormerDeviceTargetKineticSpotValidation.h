//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceTargetKineticSpotValidation.h
 * @brief One-configuration trained TARGET scalar/GEMM parity without CPU P traces.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_TARGET_KINETIC_SPOT_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_TARGET_KINETIC_SPOT_VALIDATION_H

#include "PsiFormerDeviceTargetKinetic.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace qmcplusplus::psiformer::device
{

/** Compare every canonical response lane, retaining only one host reference.
 * Owners are sequential, K=1 is explicit, and CPU/full-reference owners never
 * exist here. Compact tests separately establish independent CPU correctness.
 */
inline void validateTargetKineticSpot(const std::shared_ptr<const PreparedDeviceModel>& image,
    const std::vector<double>& positions, compute::Queue<PlatformKind::CUDA>& queue,
    AcceleratorBlasHandle& blas, std::ostream& output)
{
  const auto parameters = image->plan().parameterCount();
  std::vector<double> reference(parameters), chunk(std::min(std::size_t(4096), parameters));
  output << std::setprecision(17)
         << "# qualification=ONE_CONFIGURATION_TARGET_LEAF_BACKEND_PARITY; no_CPU_P_trace; K=1\n"
         << "channels,parameters,device_bytes,host_bytes,warm_seconds,max_absolute,max_relative\n";
  for (std::size_t slots : {std::size_t(0), std::size_t(1)})
  {
    const auto cost = estimateTargetKineticCost(image->plan(), 1, slots);
    PreparedDeviceTargetKinetic owner(image,
        {1, cost.device_workspace_bytes, cost.charged_scalar_work_units, slots}, queue, blas);

    // Warm this exact selected path, then time one complete retained request.
    owner.evaluate(positions, {1.0});
    queue.sync();
    const auto start = std::chrono::steady_clock::now();
    owner.evaluate(positions, {1.0});
    queue.sync();
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (slots == 0)
      owner.downloadResponse(reference);

    // Stream the candidate without a second P-sized host or device destination.
    double max_absolute = 0.0, max_relative = 0.0;
    for (std::size_t begin = 0; begin < parameters; begin += chunk.size())
    {
      const auto count = std::min(chunk.size(), parameters - begin);
      owner.downloadResponseChunk(begin, chunk.data(), count);
      for (std::size_t lane = 0; lane < count; ++lane)
      {
        const auto expected = reference[begin + lane], actual = chunk[lane];
        const auto difference = std::abs(actual - expected);
        max_absolute = std::max(max_absolute, difference);
        max_relative = std::max(max_relative, difference / std::max(std::abs(expected), 1e-12));
        if (!device_math::isFiniteBinary64(expected) || !device_math::isFiniteBinary64(actual) ||
            difference > 2e-7 * (1 + std::abs(expected)))
          throw std::runtime_error("Trained TARGET scalar/GEMM canonical response mismatch");
      }
    }
    output << slots << ',' << parameters << ',' << owner.deviceWorkspaceBytes() << ','
           << owner.hostWorkspaceBytes() << ',' << seconds << ',' << max_absolute << ',' << max_relative << '\n';
    std::cout << "TARGET kinetic spot: K=1 lifted_channel_slots=" << slots << " P=" << parameters
              << " device_bytes=" << owner.deviceWorkspaceBytes() << " host_bytes=" << owner.hostWorkspaceBytes()
              << " warm_seconds=" << seconds << " max_absolute=" << max_absolute
              << " max_relative=" << max_relative << '\n';
  }
  std::cout << "PASS one-configuration TARGET scalar/GEMM all-P trained spot; no full-VMC speedup claim\n";
}

} // namespace qmcplusplus::psiformer::device
#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_TARGET_KINETIC_SPOT_VALIDATION_H
