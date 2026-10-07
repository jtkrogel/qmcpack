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

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QMC_PF_SPATIAL_HOST_DEVICE __host__ __device__
#else
#define QMC_PF_SPATIAL_HOST_DEVICE
#endif

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

  QMC_PF_SPATIAL_HOST_DEVICE std::size_t uncheckedPlaneOffset(
      std::size_t configuration, std::size_t plane,
      std::size_t element) const noexcept
  {
    return configuration * configuration_stride + plane * plane_stride + element;
  }

  QMC_PF_SPATIAL_HOST_DEVICE std::size_t uncheckedValueOffset(
      std::size_t configuration, std::size_t element) const noexcept
  {
    return uncheckedPlaneOffset(configuration, 0, element);
  }

  QMC_PF_SPATIAL_HOST_DEVICE std::size_t uncheckedGradientOffset(
      std::size_t configuration, std::size_t lane,
      std::size_t element) const noexcept
  {
    return uncheckedPlaneOffset(configuration, 1 + lane, element);
  }

  QMC_PF_SPATIAL_HOST_DEVICE std::size_t uncheckedLaplacianOffset(
      std::size_t configuration, std::size_t electron,
      std::size_t element) const noexcept
  {
    return uncheckedPlaneOffset(configuration, 1 + gradient_lanes + electron, element);
  }
};

struct SoftmaxJetRowLayout
{
  SpatialJetLayout jets;
  std::size_t row_count        = 0;
  std::size_t row_width        = 0;
  std::size_t row_stride       = 0;
  std::size_t rows_per_group   = 0;
  std::size_t row_group_stride = 0;
};

struct SpatialDenseJetLayout
{
  SpatialJetLayout source;
  SpatialJetLayout target;
  std::size_t rows              = 0;
  std::size_t input_width       = 0;
  std::size_t output_width      = 0;
  std::size_t source_row_stride = 0;
  std::size_t weight_row_stride = 0;
  std::size_t target_row_stride = 0;
};

struct SpatialAttentionJetLayout
{
  SpatialJetLayout features;
  SpatialJetLayout attention;
  std::size_t rows                  = 0;
  std::size_t heads                 = 0;
  std::size_t head_width            = 0;
  std::size_t feature_row_stride    = 0;
  std::size_t attention_row_stride  = 0;
  std::size_t attention_head_stride = 0;
};

static_assert(std::is_standard_layout_v<SpatialJetLayout>);
static_assert(std::is_trivially_copyable_v<SpatialJetLayout>);
static_assert(std::is_standard_layout_v<SoftmaxJetRowLayout>);
static_assert(std::is_trivially_copyable_v<SoftmaxJetRowLayout>);
static_assert(std::is_standard_layout_v<SpatialDenseJetLayout>);
static_assert(std::is_trivially_copyable_v<SpatialDenseJetLayout>);
static_assert(std::is_standard_layout_v<SpatialAttentionJetLayout>);
static_assert(std::is_trivially_copyable_v<SpatialAttentionJetLayout>);

inline void validateSoftmaxJetRowLayout(const SoftmaxJetRowLayout& layout);

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

inline bool haveMatchingSpatialPlanes(const SpatialJetLayout& left,
                                      const SpatialJetLayout& right) noexcept
{
  return left.configuration_count == right.configuration_count &&
      left.electron_count == right.electron_count &&
      left.gradient_lanes == right.gradient_lanes &&
      left.laplacian_lanes == right.laplacian_lanes &&
      left.plane_count == right.plane_count && left.mode == right.mode;
}

inline void validateSpatialDenseJetLayout(const SpatialDenseJetLayout& layout)
{
  validateSpatialJetLayout(layout.source);
  validateSpatialJetLayout(layout.target);
  if (!haveMatchingSpatialPlanes(layout.source, layout.target))
    throw std::invalid_argument("PsiFormer spatial dense plane layouts do not match");
  if (layout.rows == 0 || layout.input_width == 0 || layout.output_width == 0)
    throw std::invalid_argument("PsiFormer spatial dense dimensions must be positive");
  if (layout.source_row_stride < layout.input_width ||
      layout.weight_row_stride < layout.output_width ||
      layout.target_row_stride < layout.output_width)
    throw std::invalid_argument("PsiFormer spatial dense row stride is too small");
  const std::size_t source_elements = spatial_detail::checkedSpan(
      layout.rows, layout.source_row_stride, layout.input_width,
      "PsiFormer spatial dense source extent overflow");
  const std::size_t target_elements = spatial_detail::checkedSpan(
      layout.rows, layout.target_row_stride, layout.output_width,
      "PsiFormer spatial dense target extent overflow");
  (void)spatial_detail::checkedSpan(
      layout.input_width, layout.weight_row_stride, layout.output_width,
      "PsiFormer spatial dense weight extent overflow");
  if (source_elements > layout.source.element_count ||
      target_elements > layout.target.element_count)
    throw std::invalid_argument("PsiFormer spatial dense rows exceed the plane extent");
}

inline SpatialDenseJetLayout makeSpatialDenseJetLayout(
    std::size_t configuration_count,
    std::size_t rows,
    std::size_t input_width,
    std::size_t output_width,
    std::size_t electron_count,
    SpatialJetMode mode,
    std::size_t source_row_stride = 0,
    std::size_t weight_row_stride = 0,
    std::size_t target_row_stride = 0,
    std::size_t source_plane_stride = 0,
    std::size_t target_plane_stride = 0,
    std::size_t source_configuration_stride = 0,
    std::size_t target_configuration_stride = 0)
{
  const std::size_t actual_source_row_stride =
      source_row_stride == 0 ? input_width : source_row_stride;
  const std::size_t actual_weight_row_stride =
      weight_row_stride == 0 ? output_width : weight_row_stride;
  const std::size_t actual_target_row_stride =
      target_row_stride == 0 ? output_width : target_row_stride;
  const std::size_t source_elements = spatial_detail::checkedSpan(
      rows, actual_source_row_stride, input_width,
      "PsiFormer spatial dense source extent overflow");
  const std::size_t target_elements = spatial_detail::checkedSpan(
      rows, actual_target_row_stride, output_width,
      "PsiFormer spatial dense target extent overflow");
  SpatialDenseJetLayout layout{
      makeSpatialJetLayout(configuration_count, source_elements, electron_count,
                           mode, source_plane_stride, source_configuration_stride),
      makeSpatialJetLayout(configuration_count, target_elements, electron_count,
                           mode, target_plane_stride, target_configuration_stride),
      rows, input_width, output_width, actual_source_row_stride,
      actual_weight_row_stride, actual_target_row_stride};
  validateSpatialDenseJetLayout(layout);
  return layout;
}

inline void validateSpatialAttentionJetLayout(const SpatialAttentionJetLayout& layout)
{
  validateSpatialJetLayout(layout.features);
  validateSpatialJetLayout(layout.attention);
  if (!haveMatchingSpatialPlanes(layout.features, layout.attention))
    throw std::invalid_argument("PsiFormer spatial attention plane layouts do not match");
  if (layout.rows == 0 || layout.heads == 0 || layout.head_width == 0)
    throw std::invalid_argument("PsiFormer spatial attention dimensions must be positive");
  const std::size_t feature_width = spatial_detail::checkedProduct(
      layout.heads, layout.head_width,
      "PsiFormer spatial attention feature width overflow");
  if (layout.feature_row_stride < feature_width ||
      layout.attention_row_stride < layout.rows)
    throw std::invalid_argument("PsiFormer spatial attention row stride is too small");
  const std::size_t feature_elements = spatial_detail::checkedSpan(
      layout.rows, layout.feature_row_stride, feature_width,
      "PsiFormer spatial attention feature extent overflow");
  const std::size_t one_head_elements = spatial_detail::checkedSpan(
      layout.rows, layout.attention_row_stride, layout.rows,
      "PsiFormer spatial attention head extent overflow");
  if (layout.attention_head_stride < one_head_elements)
    throw std::invalid_argument("PsiFormer spatial attention head stride is too small");
  const std::size_t attention_elements = spatial_detail::checkedSpan(
      layout.heads, layout.attention_head_stride, one_head_elements,
      "PsiFormer spatial attention extent overflow");
  if (feature_elements > layout.features.element_count ||
      attention_elements > layout.attention.element_count)
    throw std::invalid_argument("PsiFormer spatial attention data exceed the plane extent");
}

inline SpatialAttentionJetLayout makeSpatialAttentionJetLayout(
    std::size_t configuration_count,
    std::size_t rows,
    std::size_t heads,
    std::size_t head_width,
    std::size_t electron_count,
    SpatialJetMode mode,
    std::size_t feature_row_stride = 0,
    std::size_t attention_row_stride = 0,
    std::size_t attention_head_stride = 0,
    std::size_t feature_plane_stride = 0,
    std::size_t attention_plane_stride = 0,
    std::size_t feature_configuration_stride = 0,
    std::size_t attention_configuration_stride = 0)
{
  const std::size_t feature_width = spatial_detail::checkedProduct(
      heads, head_width, "PsiFormer spatial attention feature width overflow");
  const std::size_t actual_feature_row_stride =
      feature_row_stride == 0 ? feature_width : feature_row_stride;
  const std::size_t actual_attention_row_stride =
      attention_row_stride == 0 ? rows : attention_row_stride;
  const std::size_t feature_elements = spatial_detail::checkedSpan(
      rows, actual_feature_row_stride, feature_width,
      "PsiFormer spatial attention feature extent overflow");
  const std::size_t one_head_elements = spatial_detail::checkedSpan(
      rows, actual_attention_row_stride, rows,
      "PsiFormer spatial attention head extent overflow");
  const std::size_t actual_attention_head_stride =
      attention_head_stride == 0 ? one_head_elements : attention_head_stride;
  const std::size_t attention_elements = spatial_detail::checkedSpan(
      heads, actual_attention_head_stride, one_head_elements,
      "PsiFormer spatial attention extent overflow");
  SpatialAttentionJetLayout layout{
      makeSpatialJetLayout(configuration_count, feature_elements, electron_count,
                           mode, feature_plane_stride, feature_configuration_stride),
      makeSpatialJetLayout(configuration_count, attention_elements, electron_count,
                           mode, attention_plane_stride, attention_configuration_stride),
      rows, heads, head_width, actual_feature_row_stride,
      actual_attention_row_stride, actual_attention_head_stride};
  validateSpatialAttentionJetLayout(layout);
  return layout;
}

inline SoftmaxJetRowLayout makeAttentionSoftmaxJetRowLayout(
    const SpatialAttentionJetLayout& layout)
{
  validateSpatialAttentionJetLayout(layout);
  const std::size_t total_rows = spatial_detail::checkedProduct(
      layout.heads, layout.rows,
      "PsiFormer spatial attention softmax row count overflow");
  SoftmaxJetRowLayout softmax{layout.attention, total_rows, layout.rows,
                              layout.attention_row_stride, layout.rows,
                              layout.attention_head_stride};
  validateSoftmaxJetRowLayout(softmax);
  return softmax;
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
  if (layout.rows_per_group == 0 || layout.row_count % layout.rows_per_group != 0)
    throw std::invalid_argument("PsiFormer softmax jet row grouping is invalid");
  const std::size_t group_count = layout.row_count / layout.rows_per_group;
  const std::size_t one_group_elements = spatial_detail::checkedSpan(
      layout.rows_per_group, layout.row_stride, layout.row_width,
      "PsiFormer softmax jet row extent overflow");
  if (layout.row_group_stride < one_group_elements)
    throw std::invalid_argument("PsiFormer softmax jet row-group stride is too small");
  const std::size_t row_elements = spatial_detail::checkedSpan(
      group_count, layout.row_group_stride, one_group_elements,
      "PsiFormer softmax jet row-group extent overflow");
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
    std::size_t configuration_stride = 0,
    std::size_t rows_per_group = 0,
    std::size_t row_group_stride = 0)
{
  if (row_count == 0 || row_width == 0)
    throw std::invalid_argument("PsiFormer softmax jet row dimensions must be positive");
  const std::size_t actual_row_stride = row_stride == 0 ? row_width : row_stride;
  const std::size_t actual_rows_per_group =
      rows_per_group == 0 ? row_count : rows_per_group;
  if (row_count % actual_rows_per_group != 0)
    throw std::invalid_argument("PsiFormer softmax jet row grouping is invalid");
  const std::size_t one_group_elements = spatial_detail::checkedSpan(
      actual_rows_per_group, actual_row_stride, row_width,
      "PsiFormer softmax jet row extent overflow");
  const std::size_t actual_group_stride =
      row_group_stride == 0 ? one_group_elements : row_group_stride;
  const std::size_t element_count = spatial_detail::checkedSpan(
      row_count / actual_rows_per_group, actual_group_stride, one_group_elements,
      "PsiFormer softmax jet row-group extent overflow");
  SoftmaxJetRowLayout layout{
      makeSpatialJetLayout(configuration_count, element_count, electron_count,
                           mode, plane_stride, configuration_stride),
      row_count, row_width, actual_row_stride, actual_rows_per_group,
      actual_group_stride};
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
      const std::size_t group = row / layout.rows_per_group;
      const std::size_t row_in_group = row % layout.rows_per_group;
      const std::size_t row_offset = configuration * layout.jets.configuration_stride +
          group * layout.row_group_stride + row_in_group * layout.row_stride;
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

#undef QMC_PF_SPATIAL_HOST_DEVICE

#endif // QMCPLUSPLUS_PSIFORMER_SPATIAL_LAYOUT_H
