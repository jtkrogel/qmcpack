//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerOpenSpatial.h
 * @brief Allocation-free open-boundary spatial jet cores shared by CPU/CUDA/HIP.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_OPEN_SPATIAL_H
#define QMCPLUSPLUS_PSIFORMER_OPEN_SPATIAL_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceDeterminantMath.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialLayout.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QMC_PF_OPEN_HOST_DEVICE __host__ __device__
#else
#define QMC_PF_OPEN_HOST_DEVICE
#endif

namespace qmcplusplus::psiformer
{

enum class OpenSpatialStatus : std::uint8_t
{
  REGULAR,
  COALESCENCE,
  NONFINITE_INPUT_OR_RESULT,
  NODE,
  REQUIRED_INVERSE_UNAVAILABLE
};

struct OpenFeatureJetLayout
{
  SpatialJetLayout output;
  std::size_t nucleus_count                = 0;
  std::size_t feature_width                = 0;
  std::size_t spin_up_count                = 0;
  std::size_t position_row_stride          = 0;
  std::size_t position_configuration_stride = 0;
  std::size_t nucleus_row_stride           = 0;
};

struct OpenCuspJetLayout
{
  SpatialJetLayout output;
  std::size_t spin_up_count                = 0;
  std::size_t position_row_stride          = 0;
  std::size_t position_configuration_stride = 0;
};

struct SpatialElementwiseJetLayout
{
  SpatialJetLayout jets;
  std::size_t row_count  = 0;
  std::size_t row_width  = 0;
  std::size_t row_stride = 0;
};

struct FinalSpatialJetLayout
{
  SpatialJetLayout output;
  std::size_t determinant_gradient_configuration_stride = 0;
  std::size_t determinant_laplacian_configuration_stride = 0;
  std::size_t laplacian_ratio_configuration_stride       = 0;
};

static_assert(std::is_trivially_copyable_v<OpenFeatureJetLayout>);
static_assert(std::is_standard_layout_v<OpenFeatureJetLayout>);
static_assert(std::is_trivially_copyable_v<OpenCuspJetLayout>);
static_assert(std::is_standard_layout_v<OpenCuspJetLayout>);
static_assert(std::is_trivially_copyable_v<SpatialElementwiseJetLayout>);
static_assert(std::is_standard_layout_v<SpatialElementwiseJetLayout>);
static_assert(std::is_trivially_copyable_v<FinalSpatialJetLayout>);
static_assert(std::is_standard_layout_v<FinalSpatialJetLayout>);

inline void validateOpenFeatureJetLayout(const OpenFeatureJetLayout& layout)
{
  validateSpatialJetLayout(layout.output);
  if (layout.nucleus_count == 0 || layout.spin_up_count > layout.output.electron_count)
    throw std::invalid_argument("PsiFormer open feature dimensions are invalid");
  const std::size_t expected_width = spatial_detail::checkedAdd(
      spatial_detail::checkedProduct(4, layout.nucleus_count,
                                     "PsiFormer open feature width overflow"),
      1, "PsiFormer open feature width overflow");
  if (layout.feature_width != expected_width ||
      layout.output.element_count != spatial_detail::checkedProduct(
          layout.output.electron_count, expected_width,
          "PsiFormer open feature extent overflow"))
    throw std::invalid_argument("PsiFormer open feature output extent is inconsistent");
  if (layout.position_row_stride < 3 || layout.nucleus_row_stride < 3 ||
      layout.position_configuration_stride < spatial_detail::checkedSpan(
          layout.output.electron_count, layout.position_row_stride, 3,
          "PsiFormer open position extent overflow"))
    throw std::invalid_argument("PsiFormer open coordinate stride is too small");
}

inline OpenFeatureJetLayout makeOpenFeatureJetLayout(
    std::size_t configuration_count, std::size_t electron_count,
    std::size_t nucleus_count, std::size_t spin_up_count, SpatialJetMode mode,
    std::size_t position_row_stride = 3,
    std::size_t position_configuration_stride = 0,
    std::size_t nucleus_row_stride = 3,
    std::size_t plane_stride = 0, std::size_t output_configuration_stride = 0)
{
  const std::size_t width = spatial_detail::checkedAdd(
      spatial_detail::checkedProduct(4, nucleus_count,
                                     "PsiFormer open feature width overflow"),
      1, "PsiFormer open feature width overflow");
  const std::size_t minimum_position_stride = spatial_detail::checkedSpan(
      electron_count, position_row_stride, 3,
      "PsiFormer open position extent overflow");
  OpenFeatureJetLayout layout{
      makeSpatialJetLayout(configuration_count,
                           spatial_detail::checkedProduct(electron_count, width,
                                                          "PsiFormer open feature extent overflow"),
                           electron_count, mode, plane_stride,
                           output_configuration_stride),
      nucleus_count, width, spin_up_count, position_row_stride,
      position_configuration_stride == 0 ? minimum_position_stride
                                         : position_configuration_stride,
      nucleus_row_stride};
  validateOpenFeatureJetLayout(layout);
  return layout;
}

inline void validateOpenCuspJetLayout(const OpenCuspJetLayout& layout)
{
  validateSpatialJetLayout(layout.output);
  if (layout.output.element_count != 1 ||
      layout.spin_up_count > layout.output.electron_count ||
      layout.position_row_stride < 3 ||
      layout.position_configuration_stride < spatial_detail::checkedSpan(
          layout.output.electron_count, layout.position_row_stride, 3,
          "PsiFormer cusp position extent overflow"))
    throw std::invalid_argument("PsiFormer open cusp layout is invalid");
}

inline OpenCuspJetLayout makeOpenCuspJetLayout(
    std::size_t configuration_count, std::size_t electron_count,
    std::size_t spin_up_count, SpatialJetMode mode,
    std::size_t position_row_stride = 3,
    std::size_t position_configuration_stride = 0,
    std::size_t plane_stride = 0, std::size_t output_configuration_stride = 0)
{
  const std::size_t minimum_position_stride = spatial_detail::checkedSpan(
      electron_count, position_row_stride, 3,
      "PsiFormer cusp position extent overflow");
  OpenCuspJetLayout layout{
      makeSpatialJetLayout(configuration_count, 1, electron_count, mode,
                           plane_stride, output_configuration_stride),
      spin_up_count, position_row_stride,
      position_configuration_stride == 0 ? minimum_position_stride
                                         : position_configuration_stride};
  validateOpenCuspJetLayout(layout);
  return layout;
}

inline void validateSpatialElementwiseJetLayout(const SpatialElementwiseJetLayout& layout)
{
  validateSpatialJetLayout(layout.jets);
  if (layout.row_count == 0 || layout.row_width == 0 ||
      layout.row_stride < layout.row_width ||
      layout.jets.element_count != spatial_detail::checkedSpan(
          layout.row_count, layout.row_stride, layout.row_width,
          "PsiFormer elementwise jet extent overflow"))
    throw std::invalid_argument("PsiFormer elementwise jet layout is invalid");
}

inline SpatialElementwiseJetLayout makeSpatialElementwiseJetLayout(
    std::size_t configuration_count, std::size_t row_count,
    std::size_t row_width, std::size_t electron_count, SpatialJetMode mode,
    std::size_t row_stride = 0, std::size_t plane_stride = 0,
    std::size_t configuration_stride = 0)
{
  const std::size_t actual_row_stride = row_stride == 0 ? row_width : row_stride;
  SpatialElementwiseJetLayout layout{
      makeSpatialJetLayout(configuration_count,
                           spatial_detail::checkedSpan(
                               row_count, actual_row_stride, row_width,
                               "PsiFormer elementwise jet extent overflow"),
                           electron_count, mode, plane_stride,
                           configuration_stride),
      row_count, row_width, actual_row_stride};
  validateSpatialElementwiseJetLayout(layout);
  return layout;
}

inline void validateFinalSpatialJetLayout(const FinalSpatialJetLayout& layout)
{
  validateSpatialJetLayout(layout.output);
  if (layout.output.element_count != 1 ||
      layout.determinant_gradient_configuration_stride < layout.output.gradient_lanes ||
      layout.determinant_laplacian_configuration_stride < layout.output.laplacian_lanes ||
      layout.laplacian_ratio_configuration_stride < layout.output.laplacian_lanes)
    throw std::invalid_argument("PsiFormer final spatial layout is invalid");
}

inline FinalSpatialJetLayout makeFinalSpatialJetLayout(
    std::size_t configuration_count, std::size_t electron_count,
    SpatialJetMode mode, std::size_t determinant_gradient_configuration_stride = 0,
    std::size_t determinant_laplacian_configuration_stride = 0,
    std::size_t laplacian_ratio_configuration_stride = 0,
    std::size_t plane_stride = 0, std::size_t output_configuration_stride = 0)
{
  const SpatialJetLayout output = makeSpatialJetLayout(
      configuration_count, 1, electron_count, mode, plane_stride,
      output_configuration_stride);
  FinalSpatialJetLayout layout{
      output,
      determinant_gradient_configuration_stride == 0 ? output.gradient_lanes
                                                      : determinant_gradient_configuration_stride,
      determinant_laplacian_configuration_stride == 0 ? output.laplacian_lanes
                                                       : determinant_laplacian_configuration_stride,
      laplacian_ratio_configuration_stride == 0 ? output.laplacian_lanes
                                                : laplacian_ratio_configuration_stride};
  validateFinalSpatialJetLayout(layout);
  return layout;
}

namespace open_spatial
{

QMC_PF_OPEN_HOST_DEVICE inline void clearConfiguration(const SpatialJetLayout& layout,
                                                   std::size_t configuration,
                                                   double* output) noexcept
{
  for (std::size_t plane = 0; plane < layout.plane_count; ++plane)
    for (std::size_t element = 0; element < layout.element_count; ++element)
      output[layout.uncheckedPlaneOffset(configuration, plane, element)] = 0.0;
}

QMC_PF_OPEN_HOST_DEVICE inline bool configurationIsFinite(
    const SpatialJetLayout& layout, std::size_t configuration,
    const double* output) noexcept
{
  for (std::size_t plane = 0; plane < layout.plane_count; ++plane)
    for (std::size_t element = 0; element < layout.element_count; ++element)
      if (!device_math::isFiniteBinary64(
              output[layout.uncheckedPlaneOffset(configuration, plane, element)]))
        return false;
  return true;
}

QMC_PF_OPEN_HOST_DEVICE inline OpenSpatialStatus buildOpenFeatureJetsForConfiguration(
    const OpenFeatureJetLayout& layout, const double* positions,
    const double* nuclei, std::size_t configuration, std::size_t active_electron,
    double* output) noexcept
{
  clearConfiguration(layout.output, configuration, output);
  if (active_electron >= layout.output.electron_count)
    return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
  const double* configuration_positions =
      positions + configuration * layout.position_configuration_stride;
  for (std::size_t electron = 0; electron < layout.output.electron_count; ++electron)
  {
    const double* electron_position = configuration_positions + electron * layout.position_row_stride;
    const std::size_t row = electron * layout.feature_width;
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
      const device_math::SoftenedRadial<double> radial = device_math::softenedRadial(radius);
      const std::size_t base = row + 4 * nucleus;
      output[layout.output.uncheckedValueOffset(configuration, base)] = radial.log1p_radius;
      for (std::size_t component = 0; component < 3; ++component)
        output[layout.output.uncheckedValueOffset(configuration, base + 1 + component)] =
            displacement[component] * radial.log1p_over_radius;

      const bool differentiated = layout.output.mode == SpatialJetMode::FULL_VGL ||
          electron == active_electron;
      if (differentiated)
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
        {
          const std::size_t lane = layout.output.mode == SpatialJetMode::FULL_VGL
              ? 3 * electron + dimension : dimension;
          const double direction = displacement[dimension] / radius;
          output[layout.output.uncheckedGradientOffset(configuration, lane, base)] =
              radial.log1p_first * direction;
          for (std::size_t component = 0; component < 3; ++component)
            output[layout.output.uncheckedGradientOffset(
                configuration, lane, base + 1 + component)] =
                (component == dimension ? radial.log1p_over_radius : 0.0) +
                displacement[component] * radial.log1p_over_radius_first * direction;
        }
      if (layout.output.mode == SpatialJetMode::FULL_VGL)
      {
        output[layout.output.uncheckedLaplacianOffset(configuration, electron, base)] =
            radial.log1p_second + 2.0 * radial.log1p_first / radius;
        for (std::size_t component = 0; component < 3; ++component)
          output[layout.output.uncheckedLaplacianOffset(
              configuration, electron, base + 1 + component)] =
              2.0 * radial.log1p_over_radius_first * displacement[component] / radius +
              displacement[component] *
                  (radial.log1p_over_radius_second +
                   2.0 * radial.log1p_over_radius_first / radius);
      }
    }
    output[layout.output.uncheckedValueOffset(configuration,
                                               row + layout.feature_width - 1)] =
        electron < layout.spin_up_count ? 1.0 : -1.0;
  }
  return configurationIsFinite(layout.output, configuration, output)
      ? OpenSpatialStatus::REGULAR
      : OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
}

QMC_PF_OPEN_HOST_DEVICE inline OpenSpatialStatus biasTanhJetElement(
    const SpatialElementwiseJetLayout& layout, const double* input,
    const double* bias, std::size_t configuration, std::size_t element,
    double* output) noexcept
{
  const std::size_t feature = element % layout.row_stride;
  if (feature >= layout.row_width ||
      !device_math::isFiniteBinary64(bias[feature]))
    return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
  const double input_value = input[layout.jets.uncheckedValueOffset(configuration, element)];
  const double value = ::tanh(input_value + bias[feature]);
  const double derivative = 1.0 - value * value;
  if (!device_math::isFiniteBinary64(value))
    return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
  output[layout.jets.uncheckedValueOffset(configuration, element)] = value;
  for (std::size_t electron = 0; electron < layout.jets.laplacian_lanes; ++electron)
  {
    double gradient_squared = 0.0;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const std::size_t lane = 3 * electron + dimension;
      const double gradient = input[layout.jets.uncheckedGradientOffset(configuration, lane, element)];
      gradient_squared += gradient * gradient;
    }
    const double laplacian = input[layout.jets.uncheckedLaplacianOffset(configuration, electron, element)];
    const double output_laplacian =
        derivative * laplacian - 2.0 * value * derivative * gradient_squared;
    if (!device_math::isFiniteBinary64(output_laplacian))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
    output[layout.jets.uncheckedLaplacianOffset(configuration, electron, element)] =
        output_laplacian;
  }
  for (std::size_t lane = 0; lane < layout.jets.gradient_lanes; ++lane)
  {
    const double output_gradient = derivative *
        input[layout.jets.uncheckedGradientOffset(configuration, lane, element)];
    if (!device_math::isFiniteBinary64(output_gradient))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
    output[layout.jets.uncheckedGradientOffset(configuration, lane, element)] =
        output_gradient;
  }
  return OpenSpatialStatus::REGULAR;
}

QMC_PF_OPEN_HOST_DEVICE inline OpenSpatialStatus residualJetElement(
    const SpatialElementwiseJetLayout& layout, const double* left,
    const double* right, std::size_t configuration, std::size_t element,
    double* output) noexcept
{
  for (std::size_t plane = 0; plane < layout.jets.plane_count; ++plane)
  {
    const std::size_t offset = layout.jets.uncheckedPlaneOffset(configuration, plane, element);
    const double value = left[offset] + right[offset];
    if (!device_math::isFiniteBinary64(value))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
    output[offset] = value;
  }
  return OpenSpatialStatus::REGULAR;
}

QMC_PF_OPEN_HOST_DEVICE inline OpenSpatialStatus buildOpenCuspJetsForConfiguration(
    const OpenCuspJetLayout& layout, const double* positions,
    std::size_t configuration, std::size_t active_electron,
    double same_spin_alpha, double opposite_spin_alpha, double* output) noexcept
{
  clearConfiguration(layout.output, configuration, output);
  if (active_electron >= layout.output.electron_count ||
      !device_math::isFiniteBinary64(same_spin_alpha) ||
      !device_math::isFiniteBinary64(opposite_spin_alpha))
    return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
  const double* configuration_positions =
      positions + configuration * layout.position_configuration_stride;
  double value = 0.0;
  for (std::size_t first = 0; first < layout.output.electron_count; ++first)
    for (std::size_t second = first + 1; second < layout.output.electron_count; ++second)
    {
      double displacement[3];
      double radius_squared = 0.0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double first_value = configuration_positions[first * layout.position_row_stride + dimension];
        const double second_value = configuration_positions[second * layout.position_row_stride + dimension];
        if (!device_math::isFiniteBinary64(first_value) ||
            !device_math::isFiniteBinary64(second_value))
          return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
        displacement[dimension] = first_value - second_value;
        radius_squared += displacement[dimension] * displacement[dimension];
      }
      const double radius = ::sqrt(radius_squared);
      if (!(radius > 0.0))
        return OpenSpatialStatus::COALESCENCE;
      const bool same_spin = (first < layout.spin_up_count) ==
          (second < layout.spin_up_count);
      const double alpha = same_spin ? same_spin_alpha : opposite_spin_alpha;
      const double factor = same_spin ? 0.25 : 0.5;
      const double denominator = alpha + radius;
      const double radial_first = factor * alpha * alpha / (denominator * denominator);
      const double radial_second = -2.0 * factor * alpha * alpha /
          (denominator * denominator * denominator);
      value -= factor * alpha * alpha / denominator;
      for (std::size_t endpoint = 0; endpoint < 2; ++endpoint)
      {
        const std::size_t electron = endpoint == 0 ? first : second;
        if (layout.output.mode == SpatialJetMode::FULL_VGL || electron == active_electron)
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
          {
            const std::size_t lane = layout.output.mode == SpatialJetMode::FULL_VGL
                ? 3 * electron + dimension : dimension;
            const double sign = endpoint == 0 ? 1.0 : -1.0;
            output[layout.output.uncheckedGradientOffset(configuration, lane, 0)] +=
                sign * radial_first * displacement[dimension] / radius;
          }
        if (layout.output.mode == SpatialJetMode::FULL_VGL)
          output[layout.output.uncheckedLaplacianOffset(configuration, electron, 0)] +=
              radial_second + 2.0 * radial_first / radius;
      }
    }
  output[layout.output.uncheckedValueOffset(configuration, 0)] = value;
  return configurationIsFinite(layout.output, configuration, output)
      ? OpenSpatialStatus::REGULAR
      : OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
}

QMC_PF_OPEN_HOST_DEVICE inline OpenSpatialStatus combineFinalSpatialForConfiguration(
    const FinalSpatialJetLayout& layout,
    const device_determinant::CombinationMetadata* determinant_metadata,
    const device_determinant::DerivativeStatus* determinant_status,
    const double* determinant_gradient, const double* determinant_laplacian_log,
    const double* cusp, std::size_t configuration, double* phase,
    double* output, double* laplacian_ratio) noexcept
{
  clearConfiguration(layout.output, configuration, output);
  phase[configuration] = 0.0;
  for (std::size_t electron = 0; electron < layout.output.laplacian_lanes; ++electron)
    laplacian_ratio[configuration * layout.laplacian_ratio_configuration_stride + electron] = 0.0;
  const auto& metadata = determinant_metadata[configuration];
  if (metadata.status == device_determinant::CombinationStatus::NODE)
    return OpenSpatialStatus::NODE;
  if (determinant_status[configuration] ==
      device_determinant::DerivativeStatus::REQUIRED_INVERSE_UNAVAILABLE)
    return OpenSpatialStatus::REQUIRED_INVERSE_UNAVAILABLE;
  if (metadata.status != device_determinant::CombinationStatus::REGULAR ||
      determinant_status[configuration] != device_determinant::DerivativeStatus::AVAILABLE)
    return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
  phase[configuration] = metadata.phase;
  const double cusp_value = cusp[layout.output.uncheckedValueOffset(configuration, 0)];
  output[layout.output.uncheckedValueOffset(configuration, 0)] = metadata.log_abs + cusp_value;
  for (std::size_t lane = 0; lane < layout.output.gradient_lanes; ++lane)
    output[layout.output.uncheckedGradientOffset(configuration, lane, 0)] =
        determinant_gradient[configuration * layout.determinant_gradient_configuration_stride + lane] +
        cusp[layout.output.uncheckedGradientOffset(configuration, lane, 0)];
  for (std::size_t electron = 0; electron < layout.output.laplacian_lanes; ++electron)
  {
    const double lap_log =
        determinant_laplacian_log[configuration * layout.determinant_laplacian_configuration_stride + electron] +
        cusp[layout.output.uncheckedLaplacianOffset(configuration, electron, 0)];
    double gradient_squared = 0.0;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double gradient = output[layout.output.uncheckedGradientOffset(
          configuration, 3 * electron + dimension, 0)];
      gradient_squared += gradient * gradient;
    }
    output[layout.output.uncheckedLaplacianOffset(configuration, electron, 0)] = lap_log;
    laplacian_ratio[configuration * layout.laplacian_ratio_configuration_stride + electron] =
        lap_log + gradient_squared;
  }
  for (std::size_t plane = 0; plane < layout.output.plane_count; ++plane)
    if (!device_math::isFiniteBinary64(
            output[layout.output.uncheckedPlaneOffset(configuration, plane, 0)]))
      return OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT;
  return OpenSpatialStatus::REGULAR;
}

} // namespace open_spatial
} // namespace qmcplusplus::psiformer

#undef QMC_PF_OPEN_HOST_DEVICE

#endif // QMCPLUSPLUS_PSIFORMER_OPEN_SPATIAL_H
