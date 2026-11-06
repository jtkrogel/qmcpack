//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerDevicePeriodicValuePairs.h
 * @brief Request-local canonical Pair reuse for ordinary untaped VALUE.
 *
 * Include ONLY from PsiFormerDeviceSpatial.cu: this header defines global kernels
 * in that single translation unit, not a separately instantiated public API.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_PERIODIC_VALUE_PAIRS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_PERIODIC_VALUE_PAIRS_H

#include "PsiFormerDeviceResources.h"
#include "PsiFormerDeviceSpatial.h"
#include "PsiFormerDevicePeriodicValuePairProducer.h"
#include <cstdint>

namespace qmcplusplus::psiformer::device::periodic_value_pairs
{
/// Borrow immutable request inputs for one same-stream producer/consumer pair.
struct Inputs
{
  const double *features, *positions, *nuclei, *up, *down, *pi_up, *pi_down, *zeta_up, *zeta_down;
  const std::size_t* active;
};

/// Preserve the original per-output anchor proof, capture order and row counter.
__global__ void consumePairs(OpenOrbitalJetLayout layout, periodic_geometry::Cell cell, Inputs input,
    const double* radii, const std::uint8_t* valid, double* output, OpenSpatialStatus* status,
    SparseEnvelopeAnchorView anchor, std::size_t count, const double* projected_backflow)
{
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (logical >= count)
    return;
  const auto ne = layout.features.electron_count, per = layout.determinant_count * ne * ne;
  const auto configuration = logical / per, remainder = logical % per, determinant = remainder / (ne * ne);
  const auto matrix = remainder % (ne * ne), row = matrix / ne, column = matrix % ne;
  const auto bank_element = determinant * ne * ne + matrix;
  bool reuse = anchor.ready && anchor.references && anchor.changed_electrons && anchor.envelope && anchor.positions &&
      anchor.references[configuration] == 0 && anchor.changed_electrons[configuration] < ne &&
      row != anchor.changed_electrons[configuration];
  const double* position = input.positions + configuration * layout.position_configuration_stride +
      row * layout.position_row_stride;
  for (std::size_t dimension = 0; reuse && dimension < 3; ++dimension)
  {
    union Bits
    {
      double value;
      unsigned long long bits;
    } current{position[dimension]}, original{anchor.positions[row * 3 + dimension]};
    reuse = current.bits == original.bits;
  }
  const bool capture = anchor.capture && configuration == 0 && anchor.envelope && anchor.positions && anchor.references &&
      anchor.changed_electrons && anchor.references[0] == 0 && anchor.changed_electrons[0] == ne;
  if (capture && determinant == 0 && column == 0)
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      anchor.positions[row * 3 + dimension] = position[dimension];
  status[logical] = open_orbital::buildElement(layout, input.features, input.positions, input.nuclei,
      input.up, input.down, input.pi_up, input.pi_down, input.zeta_up, input.zeta_down,
      configuration, input.active[configuration], determinant, row, column, output, nullptr, nullptr,
      &cell, projected_backflow, false, nullptr, reuse ? anchor.envelope + bank_element : nullptr,
      capture ? anchor.envelope + bank_element : nullptr,
      {radii, valid, layout.features.configuration_count * ne * layout.nucleus_count});
  if (reuse && determinant == 0 && column == 0 && anchor.counters && status[logical] == OpenSpatialStatus::REGULAR)
  {
    auto previous = atomicCAS(anchor.counters, 0ULL, 0ULL);
    while (previous != ~0ULL)
    {
      const auto observed = atomicCAS(anchor.counters, previous, previous + 1ULL);
      if (observed == previous)
        break;
      previous = observed;
    }
  }
}

} // namespace qmcplusplus::psiformer::device::periodic_value_pairs
#endif
