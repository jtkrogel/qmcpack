//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerSpatialLayout.h
 * @brief Pointer-free checked layouts for batched PsiFormer spatial jets.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_SPATIAL_LAYOUT_H
#define QMCPLUSPLUS_PSIFORMER_SPATIAL_LAYOUT_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace qmcplusplus::psiformer
{

enum class SpatialJetMode : std::uint8_t
{
  ACTIVE,
  FULL_VGL
};

/** Canonical structure-of-arrays storage [configuration,plane,element].
 *
 * Plane zero is the value, followed by all gradient lanes and then all
 * contracted trace-Laplacian lanes. Strides are measured in FP64 elements.
 */
struct SpatialJetLayout
{
  std::size_t configuration_count = 0;
  std::size_t element_count       = 0;
  std::size_t electron_count      = 0;
  std::size_t gradient_lanes      = 0;
  std::size_t laplacian_lanes     = 0;
  std::size_t plane_count         = 0;
  std::size_t plane_stride        = 0;
  std::size_t configuration_stride = 0;
  SpatialJetMode mode             = SpatialJetMode::ACTIVE;

  std::size_t uncheckedPlaneOffset(std::size_t configuration,
                                   std::size_t plane,
                                   std::size_t element) const noexcept
  {
    return configuration * configuration_stride + plane * plane_stride + element;
  }

  std::size_t uncheckedValueOffset(std::size_t configuration,
                                   std::size_t element) const noexcept
  {
    return uncheckedPlaneOffset(configuration, 0, element);
  }

  std::size_t uncheckedGradientOffset(std::size_t configuration,
                                      std::size_t lane,
                                      std::size_t element) const noexcept
  {
    return uncheckedPlaneOffset(configuration, 1 + lane, element);
  }

  std::size_t uncheckedLaplacianOffset(std::size_t configuration,
                                       std::size_t electron,
                                       std::size_t element) const noexcept
  {
    return uncheckedPlaneOffset(configuration, 1 + gradient_lanes + electron, element);
  }
};

struct SoftmaxJetRowLayout
{
  SpatialJetLayout jets;
  std::size_t row_count  = 0;
  std::size_t row_width  = 0;
  std::size_t row_stride = 0;
};

static_assert(std::is_standard_layout_v<SpatialJetLayout>);
static_assert(std::is_trivially_copyable_v<SpatialJetLayout>);
static_assert(std::is_standard_layout_v<SoftmaxJetRowLayout>);
static_assert(std::is_trivially_copyable_v<SoftmaxJetRowLayout>);

namespace spatial_detail
{

inline std::size_t checkedAdd(std::size_t left, std::size_t right, const char* description)
{
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::length_error(description);
  return left + right;
}

inline std::size_t checkedProduct(std::size_t left, std::size_t right, const char* description)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(description);
  return left * right;
}

inline std::size_t checkedSpan(std::size_t count,
                               std::size_t stride,
                               std::size_t width,
                               const char* description)
{
  if (count == 0)
    return 0;
  return checkedAdd(checkedProduct(count - 1, stride, description), width, description);
}

} // namespace spatial_detail

inline std::size_t spatialJetSpanElements(const SpatialJetLayout& layout)
{
  const std::size_t configuration_elements = spatial_detail::checkedSpan(
      layout.plane_count, layout.plane_stride, layout.element_count,
      "PsiFormer spatial plane extent overflow");
  return spatial_detail::checkedSpan(
      layout.configuration_count, layout.configuration_stride,
      configuration_elements, "PsiFormer spatial configuration extent overflow");
}

inline void validateSpatialJetLayout(const SpatialJetLayout& layout)
{
  if (layout.configuration_count == 0 || layout.element_count == 0 ||
      layout.electron_count == 0)
    throw std::invalid_argument("PsiFormer spatial layout dimensions must be positive");

  std::size_t expected_gradient_lanes = 0;
  std::size_t expected_laplacian_lanes = 0;
  switch (layout.mode)
  {
  case SpatialJetMode::ACTIVE:
    expected_gradient_lanes = 3;
    break;
  case SpatialJetMode::FULL_VGL:
    expected_gradient_lanes = spatial_detail::checkedProduct(
        3, layout.electron_count, "PsiFormer spatial gradient-lane extent overflow");
    expected_laplacian_lanes = layout.electron_count;
    break;
  default:
    throw std::invalid_argument("PsiFormer spatial layout mode is invalid");
  }
  const std::size_t expected_plane_count = spatial_detail::checkedAdd(
      spatial_detail::checkedAdd(1, expected_gradient_lanes,
                                 "PsiFormer spatial plane count overflow"),
      expected_laplacian_lanes, "PsiFormer spatial plane count overflow");
  if (layout.gradient_lanes != expected_gradient_lanes ||
      layout.laplacian_lanes != expected_laplacian_lanes ||
      layout.plane_count != expected_plane_count)
    throw std::invalid_argument("PsiFormer spatial layout lanes do not match its mode");
  if (layout.plane_stride < layout.element_count)
    throw std::invalid_argument("PsiFormer spatial plane stride is too small");
  const std::size_t configuration_elements = spatial_detail::checkedSpan(
      layout.plane_count, layout.plane_stride, layout.element_count,
      "PsiFormer spatial plane extent overflow");
  if (layout.configuration_stride < configuration_elements)
    throw std::invalid_argument("PsiFormer spatial configuration stride is too small");
  (void)spatialJetSpanElements(layout);
}

inline SpatialJetLayout makeSpatialJetLayout(std::size_t configuration_count,
                                             std::size_t element_count,
                                             std::size_t electron_count,
                                             SpatialJetMode mode,
                                             std::size_t plane_stride = 0,
                                             std::size_t configuration_stride = 0)
{
  std::size_t gradient_lanes = 3;
  std::size_t laplacian_lanes = 0;
  if (mode == SpatialJetMode::FULL_VGL)
  {
    gradient_lanes = spatial_detail::checkedProduct(
        3, electron_count, "PsiFormer spatial gradient-lane extent overflow");
    laplacian_lanes = electron_count;
  }
  else if (mode != SpatialJetMode::ACTIVE)
    throw std::invalid_argument("PsiFormer spatial layout mode is invalid");
  const std::size_t plane_count = spatial_detail::checkedAdd(
      spatial_detail::checkedAdd(1, gradient_lanes,
                                 "PsiFormer spatial plane count overflow"),
      laplacian_lanes, "PsiFormer spatial plane count overflow");
  const std::size_t actual_plane_stride = plane_stride == 0 ? element_count : plane_stride;
  const std::size_t minimum_configuration_stride = spatial_detail::checkedSpan(
      plane_count, actual_plane_stride, element_count,
      "PsiFormer spatial configuration extent overflow");
  SpatialJetLayout layout{configuration_count, element_count, electron_count,
                          gradient_lanes, laplacian_lanes, plane_count,
                          actual_plane_stride,
                          configuration_stride == 0 ? minimum_configuration_stride
                                                    : configuration_stride,
                          mode};
  validateSpatialJetLayout(layout);
  return layout;
}

inline std::size_t checkedSpatialPlaneOffset(const SpatialJetLayout& layout,
                                             std::size_t configuration,
                                             std::size_t plane,
                                             std::size_t element)
{
  validateSpatialJetLayout(layout);
  if (configuration >= layout.configuration_count || plane >= layout.plane_count ||
      element >= layout.element_count)
    throw std::out_of_range("PsiFormer spatial jet index is out of range");
  return layout.uncheckedPlaneOffset(configuration, plane, element);
}

inline std::size_t checkedSpatialValueOffset(const SpatialJetLayout& layout,
                                             std::size_t configuration,
                                             std::size_t element)
{
  return checkedSpatialPlaneOffset(layout, configuration, 0, element);
}

inline std::size_t checkedSpatialGradientOffset(const SpatialJetLayout& layout,
                                                std::size_t configuration,
                                                std::size_t lane,
                                                std::size_t element)
{
  if (lane >= layout.gradient_lanes)
    throw std::out_of_range("PsiFormer spatial gradient lane is out of range");
  return checkedSpatialPlaneOffset(layout, configuration, 1 + lane, element);
}

inline std::size_t checkedSpatialLaplacianOffset(const SpatialJetLayout& layout,
                                                 std::size_t configuration,
                                                 std::size_t electron,
                                                 std::size_t element)
{
  if (electron >= layout.laplacian_lanes)
    throw std::out_of_range("PsiFormer spatial Laplacian lane is out of range");
  return checkedSpatialPlaneOffset(
      layout, configuration, 1 + layout.gradient_lanes + electron, element);
}

inline void validateSoftmaxJetRowLayout(const SoftmaxJetRowLayout& layout)
{
  validateSpatialJetLayout(layout.jets);
  if (layout.row_count == 0 || layout.row_width == 0)
    throw std::invalid_argument("PsiFormer softmax jet row dimensions must be positive");
  if (layout.row_stride < layout.row_width)
    throw std::invalid_argument("PsiFormer softmax jet row stride is too small");
  const std::size_t row_elements = spatial_detail::checkedSpan(
      layout.row_count, layout.row_stride, layout.row_width,
      "PsiFormer softmax jet row extent overflow");
  if (row_elements > layout.jets.element_count)
    throw std::invalid_argument("PsiFormer softmax jet rows exceed the plane extent");
  (void)spatial_detail::checkedProduct(
      layout.jets.configuration_count, layout.row_count,
      "PsiFormer softmax jet batch row count overflow");
}

inline SoftmaxJetRowLayout makeSoftmaxJetRowLayout(
    std::size_t configuration_count,
    std::size_t row_count,
    std::size_t row_width,
    std::size_t electron_count,
    SpatialJetMode mode,
    std::size_t row_stride = 0,
    std::size_t plane_stride = 0,
    std::size_t configuration_stride = 0)
{
  const std::size_t actual_row_stride = row_stride == 0 ? row_width : row_stride;
  const std::size_t element_count = spatial_detail::checkedSpan(
      row_count, actual_row_stride, row_width,
      "PsiFormer softmax jet row extent overflow");
  SoftmaxJetRowLayout layout{
      makeSpatialJetLayout(configuration_count, element_count, electron_count,
                           mode, plane_stride, configuration_stride),
      row_count, row_width, actual_row_stride};
  validateSoftmaxJetRowLayout(layout);
  return layout;
}

inline std::size_t softmaxJetBatchRowCount(const SoftmaxJetRowLayout& layout)
{
  validateSoftmaxJetRowLayout(layout);
  return spatial_detail::checkedProduct(
      layout.jets.configuration_count, layout.row_count,
      "PsiFormer softmax jet batch row count overflow");
}

/** Host oracle/API executing the exact allocation-free device row core. */
inline void stableSoftmaxJetRows(const SoftmaxJetRowLayout& layout, double* jets)
{
  validateSoftmaxJetRowLayout(layout);
  if (!jets)
    throw std::invalid_argument("PsiFormer softmax jet storage is null");
  for (std::size_t configuration = 0;
       configuration < layout.jets.configuration_count; ++configuration)
    for (std::size_t row = 0; row < layout.row_count; ++row)
    {
      const std::size_t row_offset = configuration * layout.jets.configuration_stride +
          row * layout.row_stride;
      const device_math::JetMathStatus status = device_math::softmaxJetRowInPlace(
          jets + row_offset,
          jets + row_offset + layout.jets.plane_stride,
          layout.jets.laplacian_lanes == 0
              ? nullptr
              : jets + row_offset +
                    (1 + layout.jets.gradient_lanes) * layout.jets.plane_stride,
          layout.row_width, layout.jets.plane_stride, layout.jets.gradient_lanes,
          layout.jets.laplacian_lanes);
      if (status == device_math::JetMathStatus::NONFINITE_INPUT)
        throw std::domain_error("PsiFormer softmax jet row contains non-finite input");
      if (status == device_math::JetMathStatus::NONFINITE_NORMALIZATION)
        throw std::domain_error("PsiFormer softmax jet normalization is not finite and positive");
      if (status == device_math::JetMathStatus::NONFINITE_RESULT)
        throw std::overflow_error("PsiFormer softmax jet row produced a non-finite result");
    }
}

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_SPATIAL_LAYOUT_H
