//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_PERIODIC_VALUE_PAIR_PRODUCER_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_PERIODIC_VALUE_PAIR_PRODUCER_H
#include "PsiFormerDeviceSpatial.h"

namespace qmcplusplus::psiformer::device::periodic_value_pairs
{
// Distinct instantiations preserve each consumer translation unit's established
// floating-point compilation policy without a cross-policy linker selection.
struct RealOrbitalMath {};
struct ComplexEnvelopeMath {};

/// Evaluate and validate the COMPLETE canonical Pair even though only radius is retained.
template<typename MathPolicy>
__global__ void preparePairs(OpenOrbitalJetLayout layout, periodic_geometry::Cell cell,
    const double* positions, const double* nuclei, double* radii, std::uint8_t* valid, std::size_t pairs)
{
  const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pairs)
    return;
  const auto nucleus = index % layout.nucleus_count;
  const auto row = (index / layout.nucleus_count) % layout.features.electron_count;
  const auto configuration = index / (layout.nucleus_count * layout.features.electron_count);
  periodic_geometry::Pair pair;
  const bool regular = periodic_geometry::evaluatePair(cell,
      positions + configuration * layout.position_configuration_stride + row * layout.position_row_stride,
      nuclei + nucleus * layout.nucleus_row_stride, pair, true);
  radii[index] = regular ? pair.radius : 0.0;
  valid[index] = regular ? 1 : 0;
}

} // namespace qmcplusplus::psiformer::device::periodic_value_pairs
#endif
