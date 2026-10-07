//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerOpenOrbital.h
 * @brief Shared open-boundary PsiFormer backflow-envelope orbital jet core.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_OPEN_ORBITAL_H
#define QMCPLUSPLUS_PSIFORMER_OPEN_ORBITAL_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerOpenSpatial.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QMC_PF_ORBITAL_HOST_DEVICE __host__ __device__
#else
#define QMC_PF_ORBITAL_HOST_DEVICE
#endif

namespace qmcplusplus::psiformer
{

/** Pointer-free layout for feature jets and [B,plane,D,N,N] orbital jets.
 *
 * Backflow parameters use [feature,determinant,orbital]. Envelope pi/zeta
 * parameters use [determinant,orbital,nucleus]. Separate parameter pointers are
 * supplied for the up/down row-electron spin channels.
 */
struct OpenOrbitalJetLayout
{
  SpatialJetLayout features;
  SpatialJetLayout orbitals;
  std::size_t determinant_count           = 0;
  std::size_t nucleus_count               = 0;
  std::size_t feature_width               = 0;
  std::size_t spin_up_count               = 0;
  std::size_t feature_row_stride          = 0;
  std::size_t orbital_determinant_stride  = 0;
  std::size_t orbital_row_stride          = 0;
  std::size_t backflow_feature_stride     = 0;
  std::size_t backflow_determinant_stride = 0;
  std::size_t envelope_determinant_stride = 0;
  std::size_t envelope_orbital_stride     = 0;
  std::size_t position_row_stride         = 0;
  std::size_t position_configuration_stride = 0;
  std::size_t nucleus_row_stride          = 0;
};

static_assert(std::is_standard_layout_v<OpenOrbitalJetLayout>);
static_assert(std::is_trivially_copyable_v<OpenOrbitalJetLayout>);

inline std::size_t openOrbitalBackflowSpan(const OpenOrbitalJetLayout& layout)
{
  const std::size_t channel_span = spatial_detail::checkedSpan(
      layout.determinant_count, layout.backflow_determinant_stride,
      layout.features.electron_count,
      "PsiFormer backflow parameter extent overflow");
  return spatial_detail::checkedSpan(
      layout.feature_width, layout.backflow_feature_stride, channel_span,
      "PsiFormer backflow parameter extent overflow");
}

inline std::size_t openOrbitalEnvelopeSpan(const OpenOrbitalJetLayout& layout)
{
  const std::size_t orbital_span = spatial_detail::checkedSpan(
      layout.features.electron_count, layout.envelope_orbital_stride,
      layout.nucleus_count, "PsiFormer envelope parameter extent overflow");
  return spatial_detail::checkedSpan(
      layout.determinant_count, layout.envelope_determinant_stride, orbital_span,
      "PsiFormer envelope parameter extent overflow");
}

inline void validateOpenOrbitalJetLayout(const OpenOrbitalJetLayout& layout)
{
  validateSpatialJetLayout(layout.features);
  validateSpatialJetLayout(layout.orbitals);
  if (!haveMatchingSpatialPlanes(layout.features, layout.orbitals))
    throw std::invalid_argument("PsiFormer orbital feature/output planes do not match");
  const std::size_t electron_count = layout.features.electron_count;
  if (layout.determinant_count == 0 || layout.nucleus_count == 0 ||
      layout.feature_width == 0 || layout.spin_up_count > electron_count)
    throw std::invalid_argument("PsiFormer orbital dimensions are invalid");
  if (layout.feature_row_stride < layout.feature_width ||
      layout.features.element_count != spatial_detail::checkedSpan(
          electron_count, layout.feature_row_stride, layout.feature_width,
          "PsiFormer orbital feature extent overflow"))
    throw std::invalid_argument("PsiFormer orbital feature layout is inconsistent");
  if (layout.orbital_row_stride < electron_count ||
      layout.orbital_determinant_stride < spatial_detail::checkedSpan(
          electron_count, layout.orbital_row_stride, electron_count,
          "PsiFormer orbital matrix extent overflow") ||
      layout.orbitals.element_count != spatial_detail::checkedSpan(
          layout.determinant_count, layout.orbital_determinant_stride,
          spatial_detail::checkedSpan(
              electron_count, layout.orbital_row_stride, electron_count,
              "PsiFormer orbital matrix extent overflow"),
          "PsiFormer orbital determinant extent overflow"))
    throw std::invalid_argument("PsiFormer orbital output layout is inconsistent");
  if (layout.backflow_determinant_stride < electron_count ||
      layout.backflow_feature_stride < spatial_detail::checkedSpan(
          layout.determinant_count, layout.backflow_determinant_stride,
          electron_count, "PsiFormer backflow parameter extent overflow") ||
      layout.envelope_orbital_stride < layout.nucleus_count ||
      layout.envelope_determinant_stride < spatial_detail::checkedSpan(
          electron_count, layout.envelope_orbital_stride, layout.nucleus_count,
          "PsiFormer envelope parameter extent overflow"))
    throw std::invalid_argument("PsiFormer orbital parameter stride is too small");
  if (layout.position_row_stride < 3 || layout.nucleus_row_stride < 3 ||
      layout.position_configuration_stride < spatial_detail::checkedSpan(
          electron_count, layout.position_row_stride, 3,
          "PsiFormer orbital position extent overflow"))
    throw std::invalid_argument("PsiFormer orbital coordinate stride is too small");
  (void)openOrbitalBackflowSpan(layout);
  (void)openOrbitalEnvelopeSpan(layout);
}

inline OpenOrbitalJetLayout makeOpenOrbitalJetLayout(
    std::size_t configuration_count, std::size_t determinant_count,
    std::size_t electron_count, std::size_t nucleus_count,
    std::size_t feature_width, std::size_t spin_up_count, SpatialJetMode mode,
    std::size_t feature_row_stride = 0,
    std::size_t orbital_row_stride = 0,
    std::size_t orbital_determinant_stride = 0,
    std::size_t backflow_determinant_stride = 0,
    std::size_t backflow_feature_stride = 0,
    std::size_t envelope_orbital_stride = 0,
    std::size_t envelope_determinant_stride = 0,
    std::size_t position_row_stride = 3,
    std::size_t position_configuration_stride = 0,
    std::size_t nucleus_row_stride = 3,
    std::size_t feature_plane_stride = 0,
    std::size_t feature_configuration_stride = 0,
    std::size_t orbital_plane_stride = 0,
    std::size_t orbital_configuration_stride = 0)
{
  const std::size_t feature_row = feature_row_stride == 0 ? feature_width
                                                           : feature_row_stride;
  const std::size_t orbital_row = orbital_row_stride == 0 ? electron_count
                                                           : orbital_row_stride;
  const std::size_t packed_matrix = spatial_detail::checkedSpan(
      electron_count, orbital_row, electron_count,
      "PsiFormer orbital matrix extent overflow");
  const std::size_t orbital_det = orbital_determinant_stride == 0
      ? packed_matrix : orbital_determinant_stride;
  const std::size_t orbital_elements = spatial_detail::checkedSpan(
      determinant_count, orbital_det, packed_matrix,
      "PsiFormer orbital determinant extent overflow");
  const std::size_t backflow_det = backflow_determinant_stride == 0
      ? electron_count : backflow_determinant_stride;
  const std::size_t packed_backflow = spatial_detail::checkedSpan(
      determinant_count, backflow_det, electron_count,
      "PsiFormer backflow parameter extent overflow");
  const std::size_t backflow_feature = backflow_feature_stride == 0
      ? packed_backflow : backflow_feature_stride;
  const std::size_t envelope_orbital = envelope_orbital_stride == 0
      ? nucleus_count : envelope_orbital_stride;
  const std::size_t packed_envelope = spatial_detail::checkedSpan(
      electron_count, envelope_orbital, nucleus_count,
      "PsiFormer envelope parameter extent overflow");
  const std::size_t envelope_det = envelope_determinant_stride == 0
      ? packed_envelope : envelope_determinant_stride;
  const std::size_t position_minimum = spatial_detail::checkedSpan(
      electron_count, position_row_stride, 3,
      "PsiFormer orbital position extent overflow");
  OpenOrbitalJetLayout layout{
      makeSpatialJetLayout(
          configuration_count,
          spatial_detail::checkedSpan(electron_count, feature_row, feature_width,
                                      "PsiFormer orbital feature extent overflow"),
          electron_count, mode, feature_plane_stride,
          feature_configuration_stride),
      makeSpatialJetLayout(configuration_count, orbital_elements, electron_count,
                           mode, orbital_plane_stride,
                           orbital_configuration_stride),
      determinant_count, nucleus_count, feature_width, spin_up_count,
      feature_row, orbital_det, orbital_row, backflow_feature, backflow_det,
      envelope_det, envelope_orbital, position_row_stride,
      position_configuration_stride == 0 ? position_minimum
                                         : position_configuration_stride,
      nucleus_row_stride};
  validateOpenOrbitalJetLayout(layout);
  return layout;
}

inline std::size_t openOrbitalElementCount(const OpenOrbitalJetLayout& layout)
{
  validateOpenOrbitalJetLayout(layout);
  return spatial_detail::checkedProduct(
      layout.features.configuration_count,
      spatial_detail::checkedProduct(
          layout.determinant_count,
          spatial_detail::checkedProduct(
              layout.features.electron_count, layout.features.electron_count,
              "PsiFormer orbital element count overflow"),
          "PsiFormer orbital element count overflow"),
      "PsiFormer orbital batch extent overflow");
}

namespace open_orbital
{

QMC_PF_ORBITAL_HOST_DEVICE inline std::size_t outputElement(
    const OpenOrbitalJetLayout& layout, std::size_t determinant,
    std::size_t row_electron, std::size_t orbital) noexcept
{
  return determinant * layout.orbital_determinant_stride +
      row_electron * layout.orbital_row_stride + orbital;
}

QMC_PF_ORBITAL_HOST_DEVICE inline std::size_t weightOffset(
    const OpenOrbitalJetLayout& layout, std::size_t feature,
    std::size_t determinant, std::size_t orbital) noexcept
{
  return feature * layout.backflow_feature_stride +
      determinant * layout.backflow_determinant_stride + orbital;
}

QMC_PF_ORBITAL_HOST_DEVICE inline std::size_t envelopeOffset(
    const OpenOrbitalJetLayout& layout, std::size_t determinant,
    std::size_t orbital, std::size_t nucleus) noexcept
{
  return determinant * layout.envelope_determinant_stride +
      orbital * layout.envelope_orbital_stride + nucleus;
}

QMC_PF_ORBITAL_HOST_DEVICE inline OpenSpatialStatus buildElement(
    const OpenOrbitalJetLayout& layout, const double* features,
    const double* positions, const double* nuclei,
    const double* backflow_up, const double* backflow_down,
    const double* pi_up, const double* pi_down,
    const double* zeta_up, const double* zeta_down,
    std::size_t configuration, std::size_t active_electron,
    std::size_t determinant, std::size_t row_electron,
    std::size_t orbital, double* output) noexcept
{
  const std::size_t element = outputElement(
      layout, determinant, row_electron, orbital);
  for (std::size_t plane = 0; plane < layout.orbitals.plane_count; ++plane)
    output[layout.orbitals.uncheckedPlaneOffset(configuration, plane, element)] = 0.0;
  if (active_electron >= layout.features.electron_count)
    return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;

  const bool spin_up = row_electron < layout.spin_up_count;
  const double* backflow = spin_up ? backflow_up : backflow_down;
  const double* pi       = spin_up ? pi_up : pi_down;
  const double* zeta     = spin_up ? zeta_up : zeta_down;
  double backflow_value  = 0.0;
  const std::size_t feature_begin = row_electron * layout.feature_row_stride;
  for (std::size_t feature = 0; feature < layout.feature_width; ++feature)
  {
    const double feature_value = features[layout.features.uncheckedValueOffset(
        configuration, feature_begin + feature)];
    const double weight = backflow[weightOffset(layout, feature, determinant, orbital)];
    if (!device_math::isFiniteBinary64(feature_value) ||
        !device_math::isFiniteBinary64(weight))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
    backflow_value += feature_value * weight;
  }

  double envelope_value = 0.0;
  double envelope_gradient[3]{0.0, 0.0, 0.0};
  double envelope_laplacian = 0.0;
  const double* electron_position = positions +
      configuration * layout.position_configuration_stride +
      row_electron * layout.position_row_stride;
  for (std::size_t nucleus = 0; nucleus < layout.nucleus_count; ++nucleus)
  {
    const double* nucleus_position = nuclei + nucleus * layout.nucleus_row_stride;
    double displacement[3];
    double radius_squared = 0.0;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      if (!device_math::isFiniteBinary64(electron_position[dimension]) ||
          !device_math::isFiniteBinary64(nucleus_position[dimension]))
        return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
      displacement[dimension] = electron_position[dimension] - nucleus_position[dimension];
      radius_squared += displacement[dimension] * displacement[dimension];
    }
    const double radius = ::sqrt(radius_squared);
    if (!(radius > 0.0))
      return OpenSpatialStatus::COALESCENCE;
    const std::size_t parameter = envelopeOffset(
        layout, determinant, orbital, nucleus);
    const double coefficient = pi[parameter];
    const double decay = ::fabs(zeta[parameter]);
    if (!device_math::isFiniteBinary64(coefficient) ||
        !device_math::isFiniteBinary64(decay))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
    const double weighted = coefficient * ::exp(-decay * radius);
    const double radial_first = -decay * weighted;
    const double radial_second = decay * decay * weighted;
    envelope_value += weighted;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      envelope_gradient[dimension] += radial_first * displacement[dimension] / radius;
    envelope_laplacian += radial_second + 2.0 * radial_first / radius;
  }

  const double value = backflow_value * envelope_value;
  if (!device_math::isFiniteBinary64(value))
    return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
  output[layout.orbitals.uncheckedValueOffset(configuration, element)] = value;
  for (std::size_t lane = 0; lane < layout.features.gradient_lanes; ++lane)
  {
    double backflow_gradient = 0.0;
    for (std::size_t feature = 0; feature < layout.feature_width; ++feature)
      backflow_gradient += features[layout.features.uncheckedGradientOffset(
          configuration, lane, feature_begin + feature)] *
          backflow[weightOffset(layout, feature, determinant, orbital)];
    const std::size_t lane_electron = layout.features.mode == SpatialJetMode::ACTIVE
        ? active_electron : lane / 3;
    const std::size_t dimension = lane % 3;
    const double envelope_first = lane_electron == row_electron
        ? envelope_gradient[dimension] : 0.0;
    const double gradient = backflow_gradient * envelope_value +
        backflow_value * envelope_first;
    if (!device_math::isFiniteBinary64(gradient))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
    output[layout.orbitals.uncheckedGradientOffset(configuration, lane, element)] = gradient;
  }

  for (std::size_t electron = 0; electron < layout.features.laplacian_lanes; ++electron)
  {
    double backflow_laplacian = 0.0;
    double gradient_dot = 0.0;
    for (std::size_t feature = 0; feature < layout.feature_width; ++feature)
    {
      const double weight = backflow[weightOffset(
          layout, feature, determinant, orbital)];
      backflow_laplacian += features[layout.features.uncheckedLaplacianOffset(
          configuration, electron, feature_begin + feature)] * weight;
      if (electron == row_electron)
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          gradient_dot += features[layout.features.uncheckedGradientOffset(
              configuration, 3 * electron + dimension,
              feature_begin + feature)] * weight * envelope_gradient[dimension];
    }
    const double laplacian = backflow_laplacian * envelope_value +
        2.0 * gradient_dot +
        (electron == row_electron ? backflow_value * envelope_laplacian : 0.0);
    if (!device_math::isFiniteBinary64(laplacian))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
    output[layout.orbitals.uncheckedLaplacianOffset(
        configuration, electron, element)] = laplacian;
  }
  return OpenSpatialStatus::REGULAR;
}

} // namespace open_orbital
} // namespace qmcplusplus::psiformer

#undef QMC_PF_ORBITAL_HOST_DEVICE

#endif // QMCPLUSPLUS_PSIFORMER_OPEN_ORBITAL_H
