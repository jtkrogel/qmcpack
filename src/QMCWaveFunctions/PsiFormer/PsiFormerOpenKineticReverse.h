//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerOpenKineticReverse.h
 * @brief Allocation-free lifted reverse of real orbital and cusp spatial jets.
 *
 * These leaves cover exactly one FULL_VGL configuration.  All reductions have one
 * serial owner, all storage is caller owned, and successful parameter publication is
 * additive into one canonical response vector.  The six orbital response offsets are
 * independent: no block is located by arithmetic from another block. An optional
 * immutable torus cell selects periodic radial geometry; null preserves the
 * original open-boundary arithmetic and all existing entry points.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_OPEN_KINETIC_REVERSE_H
#define QMCPLUSPLUS_PSIFORMER_OPEN_KINETIC_REVERSE_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerOpenOrbital.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QMC_PF_OPEN_REVERSE_HOST_DEVICE __host__ __device__
#else
#define QMC_PF_OPEN_REVERSE_HOST_DEVICE
#endif

namespace qmcplusplus::psiformer
{

enum class OpenKineticReverseStatus : std::uint8_t
{
  REGULAR,
  COALESCENCE,
  NONFINITE_INPUT_OR_RESULT
};

/** Layout of the lifted reverse of A=B*E and its parameter contractions. */
struct OpenOrbitalKineticReverseLayout
{
  OpenOrbitalJetLayout forward;
  std::size_t response_count                    = 0;
  std::size_t backflow_up_response_offset       = 0;
  std::size_t backflow_down_response_offset     = 0;
  std::size_t envelope_pi_up_response_offset    = 0;
  std::size_t envelope_pi_down_response_offset  = 0;
  std::size_t envelope_zeta_up_response_offset  = 0;
  std::size_t envelope_zeta_down_response_offset = 0;
};

/** Layout of the lifted reverse of the analytic open-boundary cusp. */
struct OpenCuspKineticReverseLayout
{
  OpenCuspJetLayout forward;
  std::size_t response_count              = 0;
  std::size_t same_response_offset        = 0;
  std::size_t opposite_response_offset    = 0;
  bool has_same_alpha                     = false;
};

static_assert(std::is_standard_layout_v<OpenOrbitalKineticReverseLayout>);
static_assert(std::is_trivially_copyable_v<OpenOrbitalKineticReverseLayout>);
static_assert(std::is_standard_layout_v<OpenCuspKineticReverseLayout>);
static_assert(std::is_trivially_copyable_v<OpenCuspKineticReverseLayout>);

QMC_PF_OPEN_REVERSE_HOST_DEVICE inline bool isZeroWork(
    const OpenOrbitalKineticReverseLayout& layout) noexcept
{
  return layout.forward.features.configuration_count == 0 &&
      layout.forward.orbitals.configuration_count == 0 && layout.response_count == 0;
}

QMC_PF_OPEN_REVERSE_HOST_DEVICE inline bool isZeroWork(
    const OpenCuspKineticReverseLayout& layout) noexcept
{
  return layout.forward.output.configuration_count == 0 && layout.response_count == 0;
}

namespace open_kinetic_reverse_detail
{

struct Interval
{
  std::size_t begin;
  std::size_t end;
};

inline Interval checkedInterval(std::size_t begin,
                                std::size_t extent,
                                std::size_t bound,
                                const char* description)
{
  const std::size_t end = spatial_detail::checkedAdd(begin, extent, description);
  if (end > bound)
    throw std::out_of_range(description);
  return {begin, end};
}

inline bool overlaps(const Interval& left, const Interval& right) noexcept
{
  return left.begin < right.end && right.begin < left.end;
}

template<std::size_t ReadCount, std::size_t OutputCount>
inline void rejectAliasedStorage(const void* const (&reads)[ReadCount],
                                 const void* const (&outputs)[OutputCount],
                                 const char* description)
{
  for (std::size_t output = 0; output < OutputCount; ++output)
  {
    for (std::size_t read = 0; read < ReadCount; ++read)
      if (outputs[output] == reads[read])
        throw std::invalid_argument(description);
    for (std::size_t other = output + 1; other < OutputCount; ++other)
      if (outputs[output] == outputs[other])
        throw std::invalid_argument(description);
  }
}

} // namespace open_kinetic_reverse_detail

inline void validateOpenOrbitalKineticReverseLayout(
    const OpenOrbitalKineticReverseLayout& layout)
{
  if (isZeroWork(layout))
    return;
  validateOpenOrbitalJetLayout(layout.forward);
  if (layout.forward.features.configuration_count != 1 ||
      layout.forward.orbitals.configuration_count != 1 ||
      layout.forward.features.mode != SpatialJetMode::FULL_VGL ||
      layout.forward.orbitals.mode != SpatialJetMode::FULL_VGL)
    throw std::invalid_argument(
        "PsiFormer open orbital kinetic reverse requires one FULL_VGL configuration");

  const std::size_t backflow_span = openOrbitalBackflowSpan(layout.forward);
  const std::size_t envelope_span = openOrbitalEnvelopeSpan(layout.forward);
  using open_kinetic_reverse_detail::Interval;
  const Interval ranges[6]{
      open_kinetic_reverse_detail::checkedInterval(
          layout.backflow_up_response_offset, backflow_span, layout.response_count,
          "PsiFormer open orbital backflow-up response exceeds P"),
      open_kinetic_reverse_detail::checkedInterval(
          layout.backflow_down_response_offset, backflow_span, layout.response_count,
          "PsiFormer open orbital backflow-down response exceeds P"),
      open_kinetic_reverse_detail::checkedInterval(
          layout.envelope_pi_up_response_offset, envelope_span, layout.response_count,
          "PsiFormer open orbital pi-up response exceeds P"),
      open_kinetic_reverse_detail::checkedInterval(
          layout.envelope_pi_down_response_offset, envelope_span, layout.response_count,
          "PsiFormer open orbital pi-down response exceeds P"),
      open_kinetic_reverse_detail::checkedInterval(
          layout.envelope_zeta_up_response_offset, envelope_span, layout.response_count,
          "PsiFormer open orbital zeta-up response exceeds P"),
      open_kinetic_reverse_detail::checkedInterval(
          layout.envelope_zeta_down_response_offset, envelope_span, layout.response_count,
          "PsiFormer open orbital zeta-down response exceeds P")};
  for (std::size_t left = 0; left < 6; ++left)
    for (std::size_t right = left + 1; right < 6; ++right)
      if (open_kinetic_reverse_detail::overlaps(ranges[left], ranges[right]))
        throw std::invalid_argument(
            "PsiFormer open orbital canonical response regions overlap");
}

inline void validateOpenCuspKineticReverseLayout(
    const OpenCuspKineticReverseLayout& layout)
{
  if (isZeroWork(layout))
    return;
  validateOpenCuspJetLayout(layout.forward);
  if (layout.forward.output.configuration_count != 1 ||
      layout.forward.output.mode != SpatialJetMode::FULL_VGL)
    throw std::invalid_argument(
        "PsiFormer open cusp kinetic reverse requires one FULL_VGL configuration");
  if (layout.opposite_response_offset >= layout.response_count ||
      (layout.has_same_alpha && layout.same_response_offset >= layout.response_count))
    throw std::out_of_range("PsiFormer open cusp response exceeds P");
  if (layout.has_same_alpha &&
      layout.same_response_offset == layout.opposite_response_offset)
    throw std::invalid_argument("PsiFormer open cusp response regions overlap");
}

/** Reject exact-base destructive aliases at the open-orbital launch boundary. */
inline void validateOpenOrbitalKineticReverseNoAlias(
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
    OpenKineticReverseStatus* status)
{
  const void* const reads[]{features,      positions, nuclei,    backflow_up,
                            backflow_down, pi_up,     pi_down,    zeta_up,
                            zeta_down,     backflow,  envelope,   orbital_adjoint};
  const void* const outputs[]{backflow_adjoint, envelope_adjoint, feature_adjoint,
                              response, status};
  open_kinetic_reverse_detail::rejectAliasedStorage(
      reads, outputs,
      "PsiFormer open orbital kinetic reverse input/output storage aliases");
}

/** Reject exact-base destructive aliases at the open-cusp launch boundary. */
inline void validateOpenCuspKineticReverseNoAlias(
    const double* positions,
    const double* cusp_adjoint,
    double* response,
    OpenKineticReverseStatus* status)
{
  const void* const reads[]{positions, cusp_adjoint};
  const void* const outputs[]{response, status};
  open_kinetic_reverse_detail::rejectAliasedStorage(
      reads, outputs,
      "PsiFormer open cusp kinetic reverse input/output storage aliases");
}

namespace open_kinetic_reverse
{

QMC_PF_OPEN_REVERSE_HOST_DEVICE inline bool finite(double value) noexcept
{
  return device_math::isFiniteBinary64(value);
}

QMC_PF_OPEN_REVERSE_HOST_DEVICE inline void clearOrbitalJetLogical(
    const OpenOrbitalJetLayout& layout, double* jet) noexcept
{
  if (!jet)
    return;
  for (std::size_t plane = 0; plane < layout.orbitals.plane_count; ++plane)
    for (std::size_t determinant = 0; determinant < layout.determinant_count;
         ++determinant)
      for (std::size_t row = 0; row < layout.features.electron_count; ++row)
        for (std::size_t orbital = 0; orbital < layout.features.electron_count;
             ++orbital)
          jet[layout.orbitals.uncheckedPlaneOffset(
              0, plane,
              open_orbital::outputElement(layout, determinant, row, orbital))] = 0.0;
}

QMC_PF_OPEN_REVERSE_HOST_DEVICE inline void clearFeatureJetLogical(
    const OpenOrbitalJetLayout& layout, double* jet) noexcept
{
  if (!jet)
    return;
  for (std::size_t plane = 0; plane < layout.features.plane_count; ++plane)
    for (std::size_t row = 0; row < layout.features.electron_count; ++row)
      for (std::size_t feature = 0; feature < layout.feature_width; ++feature)
        jet[layout.features.uncheckedPlaneOffset(
            0, plane, row * layout.feature_row_stride + feature)] = 0.0;
}

QMC_PF_OPEN_REVERSE_HOST_DEVICE inline void clearBackflowResponse(
    const OpenOrbitalKineticReverseLayout& layout,
    std::size_t offset,
    double* response) noexcept
{
  if (!response)
    return;
  for (std::size_t feature = 0; feature < layout.forward.feature_width; ++feature)
    for (std::size_t determinant = 0; determinant < layout.forward.determinant_count;
         ++determinant)
      for (std::size_t orbital = 0; orbital < layout.forward.features.electron_count;
           ++orbital)
        response[offset + open_orbital::weightOffset(
                              layout.forward, feature, determinant, orbital)] = 0.0;
}

QMC_PF_OPEN_REVERSE_HOST_DEVICE inline void clearEnvelopeResponse(
    const OpenOrbitalKineticReverseLayout& layout,
    std::size_t offset,
    double* response) noexcept
{
  if (!response)
    return;
  for (std::size_t determinant = 0; determinant < layout.forward.determinant_count;
       ++determinant)
    for (std::size_t orbital = 0; orbital < layout.forward.features.electron_count;
         ++orbital)
      for (std::size_t nucleus = 0; nucleus < layout.forward.nucleus_count; ++nucleus)
        response[offset + open_orbital::envelopeOffset(
                              layout.forward, determinant, orbital, nucleus)] = 0.0;
}

QMC_PF_OPEN_REVERSE_HOST_DEVICE inline void clearOrbitalResponses(
    const OpenOrbitalKineticReverseLayout& layout, double* response) noexcept
{
  clearBackflowResponse(layout, layout.backflow_up_response_offset, response);
  clearBackflowResponse(layout, layout.backflow_down_response_offset, response);
  clearEnvelopeResponse(layout, layout.envelope_pi_up_response_offset, response);
  clearEnvelopeResponse(layout, layout.envelope_pi_down_response_offset, response);
  clearEnvelopeResponse(layout, layout.envelope_zeta_up_response_offset, response);
  clearEnvelopeResponse(layout, layout.envelope_zeta_down_response_offset, response);
}

QMC_PF_OPEN_REVERSE_HOST_DEVICE inline OpenKineticReverseStatus failOrbital(
    const OpenOrbitalKineticReverseLayout& layout,
    OpenKineticReverseStatus failure,
    double* backflow_adjoint,
    double* envelope_adjoint,
    double* feature_adjoint,
    double* response,
    OpenKineticReverseStatus* status) noexcept
{
  clearOrbitalJetLogical(layout.forward, backflow_adjoint);
  clearOrbitalJetLogical(layout.forward, envelope_adjoint);
  clearFeatureJetLogical(layout.forward, feature_adjoint);
  clearOrbitalResponses(layout, response);
  if (status)
    *status = failure;
  return failure;
}

/// Reverse one orbital product element without shared destinations or failure cleanup.
QMC_PF_OPEN_REVERSE_HOST_DEVICE inline OpenKineticReverseStatus reverseOrbitalProductElement(
    const OpenOrbitalKineticReverseLayout& layout, std::size_t determinant, std::size_t row,
    std::size_t orbital, const double* backflow, const double* envelope,
    const double* orbital_adjoint, double* backflow_adjoint, double* envelope_adjoint) noexcept
{
  const auto& orbital_jets = layout.forward.orbitals;
  const std::size_t element =
      open_orbital::outputElement(layout.forward, determinant, row, orbital);
  const double b_value = backflow[orbital_jets.uncheckedValueOffset(0, element)];
  const double e_value = envelope[orbital_jets.uncheckedValueOffset(0, element)];
  const double a_value_bar =
      orbital_adjoint[orbital_jets.uncheckedValueOffset(0, element)];
  if (!finite(b_value) || !finite(e_value) || !finite(a_value_bar))
    return OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT;
  double b_value_bar = a_value_bar * e_value;
  double e_value_bar = a_value_bar * b_value;
  for (std::size_t lane = 0; lane < orbital_jets.gradient_lanes; ++lane)
  {
    const std::size_t gradient_offset =
        orbital_jets.uncheckedGradientOffset(0, lane, element);
    const std::size_t electron = lane / 3;
    const double b_gradient = backflow[gradient_offset];
    const double e_gradient = envelope[gradient_offset];
    const double a_gradient_bar = orbital_adjoint[gradient_offset];
    const double a_laplacian_bar = orbital_adjoint[
        orbital_jets.uncheckedLaplacianOffset(0, electron, element)];
    if (!finite(b_gradient) || !finite(e_gradient) ||
        !finite(a_gradient_bar) || !finite(a_laplacian_bar))
      return OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT;
    b_value_bar += a_gradient_bar * e_gradient;
    e_value_bar += a_gradient_bar * b_gradient;
    const double b_gradient_bar =
        a_gradient_bar * e_value + 2.0 * a_laplacian_bar * e_gradient;
    const double e_gradient_bar =
        a_gradient_bar * b_value + 2.0 * a_laplacian_bar * b_gradient;
    if (!finite(b_gradient_bar) || !finite(e_gradient_bar))
      return OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT;
    backflow_adjoint[gradient_offset] = b_gradient_bar;
    envelope_adjoint[gradient_offset] = e_gradient_bar;
  }
  for (std::size_t electron = 0; electron < orbital_jets.laplacian_lanes;
       ++electron)
  {
    const std::size_t laplacian_offset =
        orbital_jets.uncheckedLaplacianOffset(0, electron, element);
    const double b_laplacian = backflow[laplacian_offset];
    const double e_laplacian = envelope[laplacian_offset];
    const double a_laplacian_bar = orbital_adjoint[laplacian_offset];
    if (!finite(b_laplacian) || !finite(e_laplacian) ||
        !finite(a_laplacian_bar))
      return OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT;
    b_value_bar += a_laplacian_bar * e_laplacian;
    e_value_bar += a_laplacian_bar * b_laplacian;
    const double b_laplacian_bar = a_laplacian_bar * e_value;
    const double e_laplacian_bar = a_laplacian_bar * b_value;
    if (!finite(b_laplacian_bar) || !finite(e_laplacian_bar))
      return OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT;
    backflow_adjoint[laplacian_offset] = b_laplacian_bar;
    envelope_adjoint[laplacian_offset] = e_laplacian_bar;
  }
  if (!finite(b_value_bar) || !finite(e_value_bar))
    return OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT;
  backflow_adjoint[orbital_jets.uncheckedValueOffset(0, element)] = b_value_bar;
  envelope_adjoint[orbital_jets.uncheckedValueOffset(0, element)] = e_value_bar;
  return OpenKineticReverseStatus::REGULAR;
}

/// Accumulate one physical row's pi/zeta response, using the forward boundary policy.
/// Value cotangents use all physical rows; derivative terms use only local targets.
QMC_PF_OPEN_REVERSE_HOST_DEVICE inline OpenKineticReverseStatus reverseEnvelopeParameterRow(
    const OpenOrbitalKineticReverseLayout& layout, std::size_t row, std::size_t determinant,
    std::size_t orbital, std::size_t nucleus, const double* positions, const double* nuclei,
    const double* pi, const double* zeta, const double* envelope_adjoint, double* response,
    std::size_t pi_offset, std::size_t zeta_offset, const periodic_geometry::Cell* periodic_cell) noexcept
{
  const auto& orbital_jets = layout.forward.orbitals;
  const auto element = open_orbital::outputElement(layout.forward, determinant, row, orbital);
  const double* electron_position = positions + row * layout.forward.position_row_stride;
  const std::size_t parameter =
      open_orbital::envelopeOffset(layout.forward, determinant, orbital, nucleus);
  const double* nucleus_position =
      nuclei + nucleus * layout.forward.nucleus_row_stride;
  double displacement[3];
  periodic_geometry::Pair periodic_pair;
  double radius_squared = 0.0;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
  {
    if (!finite(electron_position[dimension]) ||
        !finite(nucleus_position[dimension]))
      return OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT;
    displacement[dimension] =
        electron_position[dimension] - nucleus_position[dimension];
    radius_squared += displacement[dimension] * displacement[dimension];
  }
  double radius = ::sqrt(radius_squared);
  if (periodic_cell)
  {
    if (!periodic_geometry::evaluatePair(*periodic_cell, electron_position, nucleus_position, periodic_pair))
      return OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT;
    radius = periodic_pair.radius;
  }
  if (!(radius > 0.0))
    return OpenKineticReverseStatus::COALESCENCE;
  const double coefficient = pi[parameter];
  const double zeta_value = zeta[parameter];
  if (!finite(coefficient) || !finite(zeta_value))
    return OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT;
  const double abs_zeta = zeta_value < 0.0 ? -zeta_value : zeta_value;
  const double sign = zeta_value > 0.0 ? 1.0 : (zeta_value < 0.0 ? -1.0 : 0.0);
  const double decay = ::exp(-abs_zeta * radius);
  const double weighted = coefficient * decay;
  const double pi_radial_first = -abs_zeta * decay;
  const double pi_radial_second = abs_zeta * abs_zeta * decay;
  const double zeta_value_partial = -sign * radius * weighted;
  const double zeta_radial_first =
      sign * weighted * (abs_zeta * radius - 1.0);
  const double zeta_radial_second =
      sign * weighted * (-radius * abs_zeta * abs_zeta + 2.0 * abs_zeta);
  double pi_bar = envelope_adjoint[
      orbital_jets.uncheckedValueOffset(0, element)] * decay;
  double zeta_bar = envelope_adjoint[
      orbital_jets.uncheckedValueOffset(0, element)] * zeta_value_partial;
  // Value cotangents span every physical row on each tile. Only the
  // row-local derivative terms use the compact target interval.
  if (orbital_jets.containsTargetElectron(row))
  {
    const std::size_t local_row = row - orbital_jets.target_electron_begin;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const std::size_t lane = 3 * local_row + dimension;
      const double direction = periodic_cell ? periodic_pair.radius_gradient[dimension]
                                             : displacement[dimension] / radius;
      const double upstream = envelope_adjoint[
          orbital_jets.uncheckedGradientOffset(0, lane, element)];
      pi_bar += upstream * pi_radial_first * direction;
      zeta_bar += upstream * zeta_radial_first * direction;
    }
    const double laplacian_upstream = envelope_adjoint[
        orbital_jets.uncheckedLaplacianOffset(0, local_row, element)];
    pi_bar += laplacian_upstream *
        (periodic_cell ? pi_radial_second * periodic_pair.radius_gradient_norm_squared +
                             pi_radial_first * periodic_pair.radius_laplacian
                       : pi_radial_second + 2.0 * pi_radial_first / radius);
    zeta_bar += laplacian_upstream *
        (periodic_cell ? zeta_radial_second * periodic_pair.radius_gradient_norm_squared +
                             zeta_radial_first * periodic_pair.radius_laplacian
                       : zeta_radial_second + 2.0 * zeta_radial_first / radius);
  }
  double& pi_destination = response[pi_offset + parameter];
  double& zeta_destination = response[zeta_offset + parameter];
  if (!finite(pi_bar) || !finite(zeta_bar) || !finite(pi_destination) ||
      !finite(zeta_destination) || !finite(pi_destination + pi_bar) ||
      !finite(zeta_destination + zeta_bar))
    return OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT;
  pi_destination += pi_bar;
  zeta_destination += zeta_bar;
  return OpenKineticReverseStatus::REGULAR;
}

/** Reverse one retained A=B*E FULL_VGL tape and its open envelope parameters.
 *
 * ``backflow`` and ``envelope`` are retained forward jets.  Their adjoints and the
 * feature adjoint are owned outputs and are assigned from zero.  ``response`` is
 * updated with ``+=`` on success.  A detected runtime failure clears only these
 * logical outputs and the six logical response tensors, leaving all padding intact.
 */
QMC_PF_OPEN_REVERSE_HOST_DEVICE inline OpenKineticReverseStatus reverseOpenOrbitalKineticSerial(
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
    OpenKineticReverseStatus* status,
    const periodic_geometry::Cell* periodic_cell = nullptr) noexcept
{
  if (isZeroWork(layout))
  {
    if (status)
      *status = OpenKineticReverseStatus::REGULAR;
    return OpenKineticReverseStatus::REGULAR;
  }
  if (!features || !positions || !nuclei || !backflow_up || !backflow_down ||
      !pi_up || !pi_down || !zeta_up || !zeta_down || !backflow || !envelope ||
      !orbital_adjoint || !backflow_adjoint || !envelope_adjoint ||
      !feature_adjoint || !response || !status)
    return failOrbital(layout, OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT,
                       backflow_adjoint, envelope_adjoint, feature_adjoint, response,
                       status);

  clearOrbitalJetLogical(layout.forward, backflow_adjoint);
  clearOrbitalJetLogical(layout.forward, envelope_adjoint);
  clearFeatureJetLogical(layout.forward, feature_adjoint);

  const SpatialJetLayout& orbital_jets = layout.forward.orbitals;
  const SpatialJetLayout& feature_jets = layout.forward.features;
  const std::size_t electrons          = feature_jets.electron_count;

  // Exact lifted reverse of every product jet A=B*E.
  for (std::size_t determinant = 0; determinant < layout.forward.determinant_count;
       ++determinant)
    for (std::size_t row = 0; row < electrons; ++row)
      for (std::size_t orbital = 0; orbital < electrons; ++orbital)
      {
        const auto result = reverseOrbitalProductElement(layout, determinant, row, orbital,
            backflow, envelope, orbital_adjoint, backflow_adjoint, envelope_adjoint);
        if (result != OpenKineticReverseStatus::REGULAR)
          return failOrbital(layout, result, backflow_adjoint, envelope_adjoint, feature_adjoint, response, status);
      }

  // Lift B=F*W through all value/gradient/Laplacian planes.
  for (std::size_t row = 0; row < electrons; ++row)
  {
    const bool spin_up = row < layout.forward.spin_up_count;
    const double* weights = spin_up ? backflow_up : backflow_down;
    const std::size_t response_offset = spin_up
        ? layout.backflow_up_response_offset
        : layout.backflow_down_response_offset;
    const std::size_t feature_begin = row * layout.forward.feature_row_stride;
    for (std::size_t determinant = 0; determinant < layout.forward.determinant_count;
         ++determinant)
      for (std::size_t orbital = 0; orbital < electrons; ++orbital)
      {
        const std::size_t element =
            open_orbital::outputElement(layout.forward, determinant, row, orbital);
        for (std::size_t feature = 0; feature < layout.forward.feature_width; ++feature)
        {
          const std::size_t feature_element = feature_begin + feature;
          const std::size_t parameter =
              open_orbital::weightOffset(layout.forward, feature, determinant, orbital);
          const double weight = weights[parameter];
          if (!finite(weight))
            return failOrbital(layout, OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT,
                               backflow_adjoint, envelope_adjoint, feature_adjoint,
                               response, status);
          double parameter_bar = 0.0;
          for (std::size_t plane = 0; plane < feature_jets.plane_count; ++plane)
          {
            const std::size_t feature_offset =
                feature_jets.uncheckedPlaneOffset(0, plane, feature_element);
            const std::size_t backflow_offset =
                orbital_jets.uncheckedPlaneOffset(0, plane, element);
            const double feature_value = features[feature_offset];
            const double upstream = backflow_adjoint[backflow_offset];
            if (!finite(feature_value) || !finite(upstream))
              return failOrbital(layout, OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT,
                                 backflow_adjoint, envelope_adjoint, feature_adjoint,
                                 response, status);
            feature_adjoint[feature_offset] += upstream * weight;
            parameter_bar += feature_value * upstream;
          }
          double& destination = response[response_offset + parameter];
          if (!finite(feature_adjoint[
                  feature_jets.uncheckedValueOffset(0, feature_element)]) ||
              !finite(parameter_bar) || !finite(destination) ||
              !finite(destination + parameter_bar))
            return failOrbital(layout, OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT,
                               backflow_adjoint, envelope_adjoint, feature_adjoint,
                               response, status);
          destination += parameter_bar;
        }
      }
  }

  // Exact full-jet pi/zeta derivatives. zeta==0 selects the zero subgradient.
  // Periodic radial compositions require the general Cartesian gradient norm
  // and trace; the Euclidean f''+2*f'/r shortcut applies only to the open branch.
  for (std::size_t row = 0; row < electrons; ++row)
  {
    const bool spin_up = row < layout.forward.spin_up_count;
    const double* pi = spin_up ? pi_up : pi_down;
    const double* zeta = spin_up ? zeta_up : zeta_down;
    const std::size_t pi_offset = spin_up
        ? layout.envelope_pi_up_response_offset
        : layout.envelope_pi_down_response_offset;
    const std::size_t zeta_offset = spin_up
        ? layout.envelope_zeta_up_response_offset
        : layout.envelope_zeta_down_response_offset;
    for (std::size_t determinant = 0; determinant < layout.forward.determinant_count;
         ++determinant)
      for (std::size_t orbital = 0; orbital < electrons; ++orbital)
      {
        for (std::size_t nucleus = 0; nucleus < layout.forward.nucleus_count; ++nucleus)
        {
          const auto result = reverseEnvelopeParameterRow(layout, row, determinant, orbital, nucleus,
              positions, nuclei, pi, zeta, envelope_adjoint, response, pi_offset, zeta_offset, periodic_cell);
          if (result != OpenKineticReverseStatus::REGULAR)
            return failOrbital(layout, result, backflow_adjoint, envelope_adjoint, feature_adjoint, response, status);
        }
      }
  }

  for (std::size_t plane = 0; plane < feature_jets.plane_count; ++plane)
    for (std::size_t row = 0; row < electrons; ++row)
      for (std::size_t feature = 0; feature < layout.forward.feature_width; ++feature)
        if (!finite(feature_adjoint[feature_jets.uncheckedPlaneOffset(
                0, plane, row * layout.forward.feature_row_stride + feature)]))
          return failOrbital(layout, OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT,
                             backflow_adjoint, envelope_adjoint, feature_adjoint,
                             response, status);
  *status = OpenKineticReverseStatus::REGULAR;
  return *status;
}

QMC_PF_OPEN_REVERSE_HOST_DEVICE inline OpenKineticReverseStatus failCusp(
    const OpenCuspKineticReverseLayout& layout,
    OpenKineticReverseStatus failure,
    double* response,
    OpenKineticReverseStatus* status) noexcept
{
  if (response)
  {
    if (layout.has_same_alpha)
      response[layout.same_response_offset] = 0.0;
    response[layout.opposite_response_offset] = 0.0;
  }
  if (status)
    *status = failure;
  return failure;
}

/** Reverse the complete value/gradient/Laplacian cusp jet into alpha responses. */
QMC_PF_OPEN_REVERSE_HOST_DEVICE inline OpenKineticReverseStatus reverseOpenCuspKineticSerial(
    const OpenCuspKineticReverseLayout& layout,
    const double* positions,
    double same_alpha,
    double opposite_alpha,
    const double* cusp_adjoint,
    double* response,
    OpenKineticReverseStatus* status,
    const periodic_geometry::Cell* periodic_cell = nullptr) noexcept
{
  if (isZeroWork(layout))
  {
    if (status)
      *status = OpenKineticReverseStatus::REGULAR;
    return OpenKineticReverseStatus::REGULAR;
  }
  if (!positions || !cusp_adjoint || !response || !status)
    return failCusp(layout, OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT,
                    response, status);
  if ((layout.has_same_alpha && !finite(same_alpha)) || !finite(opposite_alpha) ||
      (layout.has_same_alpha && !finite(response[layout.same_response_offset])) ||
      !finite(response[layout.opposite_response_offset]))
    return failCusp(layout, OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT,
                    response, status);

  const SpatialJetLayout& jets = layout.forward.output;
  for (std::size_t plane = 0; plane < jets.plane_count; ++plane)
    if (!finite(cusp_adjoint[jets.uncheckedPlaneOffset(0, plane, 0)]))
      return failCusp(layout, OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT,
                      response, status);

  double same_bar = 0.0;
  double opposite_bar = 0.0;
  for (std::size_t first = 0; first < jets.electron_count; ++first)
    for (std::size_t second = first + 1; second < jets.electron_count; ++second)
    {
      const bool same_spin = (first < layout.forward.spin_up_count) ==
          (second < layout.forward.spin_up_count);
      if (same_spin && !layout.has_same_alpha)
        continue;
      const double alpha = same_spin ? same_alpha : opposite_alpha;
      const double factor = same_spin ? 0.25 : 0.5;
      double displacement[3];
      periodic_geometry::Pair periodic_pair;
      double radius_squared = 0.0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double first_value = positions[
            first * layout.forward.position_row_stride + dimension];
        const double second_value = positions[
            second * layout.forward.position_row_stride + dimension];
        if (!finite(first_value) || !finite(second_value))
          return failCusp(layout, OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT,
                          response, status);
        displacement[dimension] = first_value - second_value;
        radius_squared += displacement[dimension] * displacement[dimension];
      }
      double radius = ::sqrt(radius_squared);
      if (periodic_cell)
      {
        if (!periodic_geometry::evaluatePair(*periodic_cell,
            positions + first * layout.forward.position_row_stride,
            positions + second * layout.forward.position_row_stride, periodic_pair))
          return failCusp(layout, OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT, response, status);
        radius = periodic_pair.radius;
      }
      if (!(radius > 0.0))
        return failCusp(layout, OpenKineticReverseStatus::COALESCENCE,
                        response, status);
      const double denominator = alpha + radius;
      const double denominator2 = denominator * denominator;
      const double denominator3 = denominator2 * denominator;
      const double denominator4 = denominator3 * denominator;
      const double value_partial =
          -factor * alpha * (alpha + 2.0 * radius) / denominator2;
      const double radial_first_partial =
          2.0 * factor * alpha * radius / denominator3;
      const double radial_second_partial =
          2.0 * factor * alpha * (alpha - 2.0 * radius) / denominator4;
      double contribution =
          cusp_adjoint[jets.uncheckedValueOffset(0, 0)] * value_partial;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double directional =
            periodic_cell ? radial_first_partial * periodic_pair.radius_gradient[dimension]
                          : radial_first_partial * displacement[dimension] / radius;
        if (jets.containsTargetElectron(first))
          contribution += cusp_adjoint[jets.uncheckedGradientOffset(
              0, 3 * (first - jets.target_electron_begin) + dimension, 0)] * directional;
        if (jets.containsTargetElectron(second))
          contribution -= cusp_adjoint[jets.uncheckedGradientOffset(
              0, 3 * (second - jets.target_electron_begin) + dimension, 0)] * directional;
      }
      // Each endpoint owns a local trace lane only when this tile contains it.
      double laplacian_adjoint = 0.0;
      if (jets.containsTargetElectron(first))
        laplacian_adjoint += cusp_adjoint[jets.uncheckedLaplacianOffset(
            0, first - jets.target_electron_begin, 0)];
      if (jets.containsTargetElectron(second))
        laplacian_adjoint += cusp_adjoint[jets.uncheckedLaplacianOffset(
            0, second - jets.target_electron_begin, 0)];
      contribution += laplacian_adjoint *
          (periodic_cell ? radial_second_partial * periodic_pair.radius_gradient_norm_squared +
                               radial_first_partial * periodic_pair.radius_laplacian
                         : radial_second_partial + 2.0 * radial_first_partial / radius);
      if (!finite(contribution))
        return failCusp(layout, OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT,
                        response, status);
      if (same_spin)
        same_bar += contribution;
      else
        opposite_bar += contribution;
    }

  if (!finite(same_bar) || !finite(opposite_bar) ||
      (layout.has_same_alpha &&
       !finite(response[layout.same_response_offset] + same_bar)) ||
      !finite(response[layout.opposite_response_offset] + opposite_bar))
    return failCusp(layout, OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT,
                    response, status);
  if (layout.has_same_alpha)
    response[layout.same_response_offset] += same_bar;
  response[layout.opposite_response_offset] += opposite_bar;
  *status = OpenKineticReverseStatus::REGULAR;
  return *status;
}

} // namespace open_kinetic_reverse
} // namespace qmcplusplus::psiformer

#undef QMC_PF_OPEN_REVERSE_HOST_DEVICE

#endif // QMCPLUSPLUS_PSIFORMER_OPEN_KINETIC_REVERSE_H
