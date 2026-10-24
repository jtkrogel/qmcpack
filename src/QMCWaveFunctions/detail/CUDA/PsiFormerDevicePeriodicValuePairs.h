//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerDevicePeriodicValuePairs.h
 * @brief Request-local canonical Pair reuse for ordinary untaped REAL VALUE.
 *
 * Include ONLY from PsiFormerDeviceSpatial.cu: this header defines global kernels
 * in that single translation unit, not a separately instantiated public API.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_PERIODIC_VALUE_PAIRS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_PERIODIC_VALUE_PAIRS_H

#include "PsiFormerDeviceResources.h"
#include "PsiFormerDeviceSpatial.h"
#include <cstdint>

namespace qmcplusplus::psiformer::device::periodic_value_pairs
{
/// Borrow immutable request inputs for one same-stream producer/consumer pair.
struct Inputs
{
  const double *features, *positions, *nuclei, *up, *down, *pi_up, *pi_down, *zeta_up, *zeta_down;
  const std::size_t* active;
};

/// Evaluate and validate the COMPLETE canonical Pair even though only radius is retained.
__global__ void preparePairs(OpenOrbitalJetLayout layout, periodic_geometry::Cell cell,
    Inputs input, double* radii, std::uint8_t* valid, std::size_t pairs)
{
  const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pairs)
    return;
  const auto nucleus = index % layout.nucleus_count;
  const auto row = (index / layout.nucleus_count) % layout.features.electron_count;
  const auto configuration = index / (layout.nucleus_count * layout.features.electron_count);
  periodic_geometry::Pair pair;
  const bool regular = periodic_geometry::evaluatePair(cell,
      input.positions + configuration * layout.position_configuration_stride + row * layout.position_row_stride,
      input.nuclei + nucleus * layout.nucleus_row_stride, pair, true);
  radii[index] = regular ? pair.radius : 0.0;
  valid[index] = regular ? 1 : 0;
}

/// Match the canonical VALUE operation order; only the per-ion Pair source varies.
template<bool banked>
__device__ OpenSpatialStatus valueElement(const OpenOrbitalJetLayout& layout, Inputs input,
    const periodic_geometry::Cell& cell, const double* radii, const std::uint8_t* valid,
    std::size_t configuration, std::size_t determinant, std::size_t row, std::size_t column,
    double* output, const double* cached_envelope, double* captured_envelope)
{
  const auto element = open_orbital::outputElement(layout, determinant, row, column);
  for (std::size_t plane = 0; plane < layout.orbitals.plane_count; ++plane)
    output[layout.orbitals.uncheckedPlaneOffset(configuration, plane, element)] = 0.0;
  if (input.active[configuration] >= layout.features.electron_count)
    return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
  const bool spin_up = row < layout.spin_up_count;
  const double* backflow = spin_up ? input.up : input.down;
  const double* pi = spin_up ? input.pi_up : input.pi_down;
  const double* zeta = spin_up ? input.zeta_up : input.zeta_down;
  double backflow_value = 0.0;
  const auto feature_begin = row * layout.feature_row_stride;
  for (std::size_t feature = 0; feature < layout.feature_width; ++feature)
  {
    const double value = input.features[layout.features.uncheckedValueOffset(configuration, feature_begin + feature)];
    const double weight = backflow[open_orbital::weightOffset(layout, feature, determinant, column)];
    if (!device_math::isFiniteBinary64(value) || !device_math::isFiniteBinary64(weight))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
    backflow_value += value * weight;
  }
  double envelope_value = cached_envelope ? *cached_envelope : 0.0;
  const double* position = input.positions + configuration * layout.position_configuration_stride +
      row * layout.position_row_stride;
  for (std::size_t nucleus = 0; !cached_envelope && nucleus < layout.nucleus_count; ++nucleus)
  {
    const double* ion = input.nuclei + nucleus * layout.nucleus_row_stride;
    double displacement[3], radius_squared = 0.0;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      if (!device_math::isFiniteBinary64(position[dimension]) || !device_math::isFiniteBinary64(ion[dimension]))
        return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
      displacement[dimension] = position[dimension] - ion[dimension];
      radius_squared += displacement[dimension] * displacement[dimension];
    }
    double radius = ::sqrt(radius_squared);
    if constexpr (banked)
    {
      const auto index = (configuration * layout.features.electron_count + row) * layout.nucleus_count + nucleus;
      if (!valid[index])
        return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
      radius = radii[index];
    }
    else
    {
      periodic_geometry::Pair pair;
      if (!periodic_geometry::evaluatePair(cell, position, ion, pair, true))
        return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
      radius = pair.radius;
    }
    // VALUE coalescence remains valid. Coefficient checks follow geometry exactly.
    const auto parameter = open_orbital::envelopeOffset(layout, determinant, column, nucleus);
    const double coefficient = pi[parameter], decay = ::fabs(zeta[parameter]);
    if (!device_math::isFiniteBinary64(coefficient) || !device_math::isFiniteBinary64(decay))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
    // The canonical runtime-mode leaf shares weighted with derivative branches:
    // preserve its DMUL then DADD, rather than specializing these into a DFMA.
    const double weighted = __dmul_rn(coefficient, ::exp(-decay * radius));
    envelope_value += weighted;
  }
  const double value = backflow_value * envelope_value;
  if (!device_math::isFiniteBinary64(value))
    return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
  output[layout.orbitals.uncheckedValueOffset(configuration, element)] = value;
  if (captured_envelope)
    *captured_envelope = envelope_value;
  return OpenSpatialStatus::REGULAR;
}

/// Preserve the original per-output anchor proof, capture order and row counter.
template<bool banked>
__global__ void consumePairs(OpenOrbitalJetLayout layout, periodic_geometry::Cell cell, Inputs input,
    const double* radii, const std::uint8_t* valid, double* output, OpenSpatialStatus* status,
    SparseEnvelopeAnchorView anchor, std::size_t count)
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
  status[logical] = valueElement<banked>(layout, input, cell, radii, valid, configuration, determinant, row, column, output,
      reuse ? anchor.envelope + bank_element : nullptr, capture ? anchor.envelope + bank_element : nullptr);
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
