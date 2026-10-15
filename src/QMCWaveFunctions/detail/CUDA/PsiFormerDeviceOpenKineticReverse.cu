//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceOpenKineticReverse.cu
 * @brief Deterministic CUDA/HIP wrappers for open-boundary lifted reverse leaves.
 */

#include "PsiFormerDeviceOpenKineticReverse.h"

#include <stdexcept>
#include <algorithm>
#include <limits>

namespace qmcplusplus::psiformer::device
{
namespace
{

Error peekAtLastError()
{
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

// Each numerical destination has one writer. Only the failure word is atomic.
struct OrbitalReversePointers
{
  const double* features;
  const double* positions;
  const double* nuclei;
  const double* up;
  const double* down;
  const double* pi_up;
  const double* pi_down;
  const double* zeta_up;
  const double* zeta_down;
  const double* backflow;
  const double* envelope;
  const double* orbital_bar;
  double* backflow_bar;
  double* envelope_bar;
  double* feature_bar;
  double* response;
};

/// Integer fault bits are reduced independently of every numerical destination.
enum OrbitalFailure : unsigned int { NONFINITE_ORBITAL = 1, COALESCENT_ORBITAL = 2 };

/// Initialize failure metadata before any work kernel can report a fault.
__global__ void initializeOrbitalReverse(unsigned int* failure, OpenKineticReverseStatus* status)
{
  *failure = 0;
  *status = OpenKineticReverseStatus::REGULAR;
}

/// Each thread reverses all local jet planes of one physical matrix element.
__global__ void orbitalProductReverse(OpenOrbitalKineticReverseLayout layout,
    OrbitalReversePointers pointers, unsigned int* failure)
{
  const auto ne = layout.forward.features.electron_count;
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= layout.forward.determinant_count * ne * ne)
    return;
  const auto result = open_kinetic_reverse::reverseOrbitalProductElement(layout,
      index / (ne * ne), index / ne % ne, index % ne, pointers.backflow, pointers.envelope,
      pointers.orbital_bar, pointers.backflow_bar, pointers.envelope_bar);
  if (result != OpenKineticReverseStatus::REGULAR)
    atomicOr(failure, NONFINITE_ORBITAL);
}

/// Publish a read-only product admission gate for the following contractions.
__global__ void admitOrbitalProduct(const unsigned int* failure, OpenKineticReverseStatus* status)
{
  if (*failure)
    *status = OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT;
}

/// Gather determinant/column contributions in reference order into one feature lane.
__global__ void orbitalFeatureReverse(OpenOrbitalKineticReverseLayout layout,
    OrbitalReversePointers pointers, const OpenKineticReverseStatus* status, unsigned int* failure)
{
  if (*status != OpenKineticReverseStatus::REGULAR)
    return;
  const auto& f = layout.forward;
  const auto ne = f.features.electron_count;
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= f.features.plane_count * ne * f.feature_width)
    return;
  const auto feature = index % f.feature_width, row = index / f.feature_width % ne;
  const auto plane = index / (f.feature_width * ne);
  const double* weights = row < f.spin_up_count ? pointers.up : pointers.down;
  double sum = 0.0;
  for (std::size_t determinant = 0; determinant < f.determinant_count; ++determinant)
    for (std::size_t orbital = 0; orbital < ne; ++orbital)
    {
      const auto element = open_orbital::outputElement(f, determinant, row, orbital);
      const auto weight = weights[open_orbital::weightOffset(f, feature, determinant, orbital)];
      const auto upstream = pointers.backflow_bar[f.orbitals.uncheckedPlaneOffset(0, plane, element)];
      sum += upstream * weight;
      if (!open_kinetic_reverse::finite(weight) || !open_kinetic_reverse::finite(upstream) ||
          (plane == 0 && !open_kinetic_reverse::finite(sum)))
        atomicOr(failure, NONFINITE_ORBITAL);
    }
  if (!open_kinetic_reverse::finite(sum))
    atomicOr(failure, NONFINITE_ORBITAL);
  pointers.feature_bar[f.features.uncheckedPlaneOffset(0, plane, row * f.feature_row_stride + feature)] = sum;
}

/// Each weight owner preserves per-row plane summation and additive request order.
__global__ void orbitalWeightReverse(OpenOrbitalKineticReverseLayout layout,
    OrbitalReversePointers pointers, const OpenKineticReverseStatus* status, unsigned int* failure)
{
  if (*status != OpenKineticReverseStatus::REGULAR)
    return;
  const auto& f = layout.forward;
  const auto ne = f.features.electron_count, span = f.feature_width * f.determinant_count * ne;
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= 2 * span)
    return;
  const bool up = index < span;
  const auto logical = index % span;
  const auto orbital = logical % ne, determinant = logical / ne % f.determinant_count;
  const auto feature = logical / (ne * f.determinant_count);
  const auto parameter = open_orbital::weightOffset(f, feature, determinant, orbital);
  const auto offset = up ? layout.backflow_up_response_offset : layout.backflow_down_response_offset;
  double& destination = pointers.response[offset + parameter];
  for (std::size_t row = up ? 0 : f.spin_up_count; row < (up ? f.spin_up_count : ne); ++row)
  {
    const auto element = open_orbital::outputElement(f, determinant, row, orbital);
    double parameter_bar = 0.0;
    for (std::size_t plane = 0; plane < f.features.plane_count; ++plane)
    {
      const auto value = pointers.features[f.features.uncheckedPlaneOffset(0, plane,
          row * f.feature_row_stride + feature)];
      const auto upstream = pointers.backflow_bar[f.orbitals.uncheckedPlaneOffset(0, plane, element)];
      if (!open_kinetic_reverse::finite(value) || !open_kinetic_reverse::finite(upstream))
        atomicOr(failure, NONFINITE_ORBITAL);
      parameter_bar += value * upstream;
    }
    if (!open_kinetic_reverse::finite(parameter_bar) || !open_kinetic_reverse::finite(destination) ||
        !open_kinetic_reverse::finite(destination + parameter_bar))
      atomicOr(failure, NONFINITE_ORBITAL);
    destination += parameter_bar;
  }
}

/// A unique pi/zeta pair gathers physical rows using the shared boundary algebra.
template<bool periodic>
__global__ void orbitalEnvelopeReverse(OpenOrbitalKineticReverseLayout layout,
    OrbitalReversePointers pointers, const OpenKineticReverseStatus* status,
    unsigned int* failure, periodic_geometry::Cell cell)
{
  if (*status != OpenKineticReverseStatus::REGULAR)
    return;
  const auto& f = layout.forward;
  const auto ne = f.features.electron_count, span = f.determinant_count * ne * f.nucleus_count;
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= 2 * span)
    return;
  const bool up = index < span;
  const auto parameter = index % span;
  const auto nucleus = parameter % f.nucleus_count, orbital = parameter / f.nucleus_count % ne;
  const auto determinant = parameter / (f.nucleus_count * ne);
  for (std::size_t row = up ? 0 : f.spin_up_count; row < (up ? f.spin_up_count : ne); ++row)
  {
    const auto result = open_kinetic_reverse::reverseEnvelopeParameterRow(layout, row, determinant, orbital,
        nucleus, pointers.positions, pointers.nuclei, up ? pointers.pi_up : pointers.pi_down,
        up ? pointers.zeta_up : pointers.zeta_down, pointers.envelope_bar, pointers.response,
        up ? layout.envelope_pi_up_response_offset : layout.envelope_pi_down_response_offset,
        up ? layout.envelope_zeta_up_response_offset : layout.envelope_zeta_down_response_offset,
        periodic ? &cell : nullptr);
    if (result != OpenKineticReverseStatus::REGULAR)
      atomicOr(failure, result == OpenKineticReverseStatus::COALESCENCE ? COALESCENT_ORBITAL : NONFINITE_ORBITAL);
  }
}

/// Failure cleanup reads an immutable word; it never races a work kernel's report.
/// Mixed-invalid inputs deterministically prefer nonfinite over coalescence.
__global__ void finishOrbitalReverse(OpenOrbitalKineticReverseLayout layout,
    OrbitalReversePointers pointers, const unsigned int* failure, OpenKineticReverseStatus* status)
{
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto bits = *failure;
  if (index == 0)
    *status = bits & NONFINITE_ORBITAL ? OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT :
        (bits & COALESCENT_ORBITAL ? OpenKineticReverseStatus::COALESCENCE : OpenKineticReverseStatus::REGULAR);
  if (!bits)
    return;
  const auto& f = layout.forward;
  const auto ne = f.features.electron_count;
  if (index < f.orbitals.plane_count * f.determinant_count * ne * ne)
  {
    const auto logical = index % (f.determinant_count * ne * ne);
    const auto element = open_orbital::outputElement(f, logical / (ne * ne), logical / ne % ne, logical % ne);
    const auto at = f.orbitals.uncheckedPlaneOffset(0, index / (f.determinant_count * ne * ne), element);
    pointers.backflow_bar[at] = pointers.envelope_bar[at] = 0.0;
  }
  if (index < f.features.plane_count * ne * f.feature_width)
  {
    const auto element = index % (ne * f.feature_width);
    const auto at = f.features.uncheckedPlaneOffset(0, index / (ne * f.feature_width),
        element / f.feature_width * f.feature_row_stride + element % f.feature_width);
    pointers.feature_bar[at] = 0.0;
  }
  const auto weights = f.feature_width * f.determinant_count * ne;
  if (index < weights)
  {
    const auto parameter = open_orbital::weightOffset(f, index / (f.determinant_count * ne),
        index / ne % f.determinant_count, index % ne);
    pointers.response[layout.backflow_up_response_offset + parameter] = 0.0;
    pointers.response[layout.backflow_down_response_offset + parameter] = 0.0;
  }
  const auto envelopes = f.determinant_count * ne * f.nucleus_count;
  if (index < envelopes)
  {
    const auto parameter = open_orbital::envelopeOffset(f, index / (ne * f.nucleus_count),
        index / f.nucleus_count % ne, index % f.nucleus_count);
    const std::size_t offsets[]{layout.envelope_pi_up_response_offset, layout.envelope_pi_down_response_offset,
                               layout.envelope_zeta_up_response_offset, layout.envelope_zeta_down_response_offset};
    for (auto offset : offsets)
      pointers.response[offset + parameter] = 0.0;
  }
}

/// Specialize boundary geometry while retaining the single-owner reverse reduction.
template<bool periodic>
__global__ void openOrbitalKineticReverseKernel(
    OpenOrbitalKineticReverseLayout layout,
    const double* features,
    const double* positions,
    const double* nuclei,
    const double* backflow_up,
    const double* backflow_down,
    const double* pi_up,
    const double* pi_down,
    const double* zeta_up,
    const double* zeta_down,
    const double* backflow,
    const double* envelope,
    const double* orbital_adjoint,
    double* backflow_adjoint,
    double* envelope_adjoint,
    double* feature_adjoint,
    double* response,
    OpenKineticReverseStatus* status, periodic_geometry::Cell cell)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
    open_kinetic_reverse::reverseOpenOrbitalKineticSerial(
        layout, features, positions, nuclei, backflow_up, backflow_down, pi_up,
        pi_down, zeta_up, zeta_down, backflow, envelope, orbital_adjoint,
        backflow_adjoint, envelope_adjoint, feature_adjoint, response, status,
        periodic ? &cell : nullptr);
}

/// Reverse cusp alpha jets using the same prepared boundary policy as forward.
template<bool periodic>
__global__ void openCuspKineticReverseKernel(
    OpenCuspKineticReverseLayout layout,
    const double* positions,
    double same_alpha,
    double opposite_alpha,
    const double* cusp_adjoint,
    double* response,
    OpenKineticReverseStatus* status, periodic_geometry::Cell cell)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
    open_kinetic_reverse::reverseOpenCuspKineticSerial(
        layout, positions, same_alpha, opposite_alpha, cusp_adjoint, response,
        status, periodic ? &cell : nullptr);
}

} // namespace

Error launchOpenOrbitalKineticReverse(
    Stream stream,
    const OpenOrbitalKineticReverseLayout& layout,
    const double* features,
    const double* positions,
    const double* nuclei,
    const double* backflow_up,
    const double* backflow_down,
    const double* pi_up,
    const double* pi_down,
    const double* zeta_up,
    const double* zeta_down,
    const double* backflow,
    const double* envelope,
    const double* orbital_adjoint,
    double* backflow_adjoint,
    double* envelope_adjoint,
    double* feature_adjoint,
    double* response,
    OpenKineticReverseStatus* status, const periodic_geometry::Cell* periodic_cell,
    unsigned int* parallel_failure)
{
  validateOpenOrbitalKineticReverseLayout(layout);
  if (isZeroWork(layout))
    return success;
  if (periodic_cell)
    periodic_geometry::validateCell(*periodic_cell);
  if (!features || !positions || !nuclei || !backflow_up || !backflow_down ||
      !pi_up || !pi_down || !zeta_up || !zeta_down || !backflow || !envelope ||
      !orbital_adjoint || !backflow_adjoint || !envelope_adjoint ||
      !feature_adjoint || !response || !status)
    throw std::invalid_argument("PsiFormer open orbital kinetic reverse storage is null");
  validateOpenOrbitalKineticReverseNoAlias(
      features, positions, nuclei, backflow_up, backflow_down, pi_up, pi_down,
      zeta_up, zeta_down, backflow, envelope, orbital_adjoint, backflow_adjoint,
      envelope_adjoint, feature_adjoint, response, status);

  if (parallel_failure)
  {
    // Admit launch extents and exact-base aliases before any output is written.
    const void* occupied[]{features, positions, nuclei, backflow_up, backflow_down, pi_up, pi_down,
        zeta_up, zeta_down, backflow, envelope, orbital_adjoint, backflow_adjoint, envelope_adjoint,
        feature_adjoint, response, status};
    for (const auto* pointer : occupied)
      if (pointer == parallel_failure)
        throw std::invalid_argument("Parallel orbital failure storage aliases numerical storage");
    const auto mul = [](std::size_t a, std::size_t b) {
      return spatial_detail::checkedProduct(a, b, "Parallel orbital reverse launch extent overflow");
    };
    const auto& f = layout.forward;
    const auto ne = f.features.electron_count;
    const auto matrices = mul(f.determinant_count, mul(ne, ne));
    const auto feature_count = mul(f.features.plane_count, mul(ne, f.feature_width));
    const auto weights = mul(f.feature_width, mul(f.determinant_count, ne));
    const auto envelopes = mul(f.determinant_count, mul(ne, f.nucleus_count));
    const auto blocks = [](std::size_t count) {
      const auto value = count / 128 + (count % 128 != 0);
      if (value > std::size_t(std::numeric_limits<int>::max()))
        throw std::length_error("Parallel orbital reverse grid exceeds launch ABI");
      return static_cast<unsigned int>(value);
    };
    const auto clear_count = std::max({mul(f.orbitals.plane_count, matrices), feature_count, weights, envelopes});
    const auto matrix_blocks = blocks(matrices), feature_blocks = blocks(feature_count);
    const auto weight_blocks = blocks(mul(2, weights)), envelope_blocks = blocks(mul(2, envelopes));
    const auto clear_blocks = blocks(clear_count);
    const OrbitalReversePointers pointers{features, positions, nuclei, backflow_up, backflow_down,
        pi_up, pi_down, zeta_up, zeta_down, backflow, envelope, orbital_adjoint, backflow_adjoint,
        envelope_adjoint, feature_adjoint, response};

    // Stream boundaries make status read-only during every reporting work kernel.
    initializeOrbitalReverse<<<1, 1, 0, stream>>>(parallel_failure, status);
    orbitalProductReverse<<<matrix_blocks, 128, 0, stream>>>(layout, pointers, parallel_failure);
    admitOrbitalProduct<<<1, 1, 0, stream>>>(parallel_failure, status);
    orbitalFeatureReverse<<<feature_blocks, 128, 0, stream>>>(layout, pointers, status, parallel_failure);
    orbitalWeightReverse<<<weight_blocks, 128, 0, stream>>>(layout, pointers, status, parallel_failure);
    if (periodic_cell)
      orbitalEnvelopeReverse<true><<<envelope_blocks, 128, 0, stream>>>(layout, pointers, status, parallel_failure, *periodic_cell);
    else
      orbitalEnvelopeReverse<false><<<envelope_blocks, 128, 0, stream>>>(layout, pointers, status, parallel_failure, {});
    finishOrbitalReverse<<<clear_blocks, 128, 0, stream>>>(layout, pointers, parallel_failure, status);
    return peekAtLastError();
  }
  if (periodic_cell)
    openOrbitalKineticReverseKernel<true><<<1, 1, 0, stream>>>(
        layout, features, positions, nuclei, backflow_up, backflow_down, pi_up,
        pi_down, zeta_up, zeta_down, backflow, envelope, orbital_adjoint,
        backflow_adjoint, envelope_adjoint, feature_adjoint, response, status, *periodic_cell);
  else
    openOrbitalKineticReverseKernel<false><<<1, 1, 0, stream>>>(
        layout, features, positions, nuclei, backflow_up, backflow_down, pi_up,
        pi_down, zeta_up, zeta_down, backflow, envelope, orbital_adjoint,
        backflow_adjoint, envelope_adjoint, feature_adjoint, response, status, {});
  return peekAtLastError();
}

Error launchOpenCuspKineticReverse(
    Stream stream,
    const OpenCuspKineticReverseLayout& layout,
    const double* positions,
    double same_alpha,
    double opposite_alpha,
    const double* cusp_adjoint,
    double* response,
    OpenKineticReverseStatus* status, const periodic_geometry::Cell* periodic_cell)
{
  validateOpenCuspKineticReverseLayout(layout);
  if (isZeroWork(layout))
    return success;
  if (periodic_cell)
    periodic_geometry::validateCell(*periodic_cell);
  if (!positions || !cusp_adjoint || !response || !status)
    throw std::invalid_argument("PsiFormer open cusp kinetic reverse storage is null");
  validateOpenCuspKineticReverseNoAlias(
      positions, cusp_adjoint, response, status);
  if (periodic_cell)
    openCuspKineticReverseKernel<true><<<1, 1, 0, stream>>>(
        layout, positions, same_alpha, opposite_alpha, cusp_adjoint, response, status, *periodic_cell);
  else
    openCuspKineticReverseKernel<false><<<1, 1, 0, stream>>>(
        layout, positions, same_alpha, opposite_alpha, cusp_adjoint, response, status, {});
  return peekAtLastError();
}

} // namespace qmcplusplus::psiformer::device
