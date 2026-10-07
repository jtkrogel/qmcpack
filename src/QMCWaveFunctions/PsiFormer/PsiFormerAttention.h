//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerAttention.h
 * @brief Pointer-free dense/attention layouts and host-callable softmax core.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_ATTENTION_H
#define QMCPLUSPLUS_PSIFORMER_ATTENTION_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace qmcplusplus::psiformer
{

/** Row-major dense product C[rows,output]=A[rows,input]*B[input,output]. */
struct DenseForwardLayout
{
  std::size_t rows              = 0;
  std::size_t input_width       = 0;
  std::size_t output_width      = 0;
  std::size_t source_row_stride = 0;
  std::size_t weight_row_stride = 0;
  std::size_t target_row_stride = 0;

  std::size_t sourceElements() const noexcept
  { return rows == 0 ? 0 : (rows - 1) * source_row_stride + input_width; }
  std::size_t weightElements() const noexcept
  { return input_width == 0 ? 0 : (input_width - 1) * weight_row_stride + output_width; }
  std::size_t targetElements() const noexcept
  { return rows == 0 ? 0 : (rows - 1) * target_row_stride + output_width; }
};

/** Electron-major headed features and head-major attention matrices. */
struct AttentionForwardLayout
{
  std::size_t rows                  = 0;
  std::size_t heads                 = 0;
  std::size_t head_width            = 0;
  std::size_t feature_row_stride    = 0;
  std::size_t attention_row_stride  = 0;
  std::size_t attention_head_stride = 0;

  std::size_t featureWidth() const noexcept { return heads * head_width; }
  std::size_t softmaxRowCount() const noexcept { return heads * rows; }
  std::size_t featureElements() const noexcept
  { return rows == 0 ? 0 : (rows - 1) * feature_row_stride + featureWidth(); }
  std::size_t attentionElements() const noexcept
  {
    return heads == 0 || rows == 0 ? 0
        : (heads - 1) * attention_head_stride + (rows - 1) * attention_row_stride + rows;
  }
  std::size_t featureOffset(std::size_t row, std::size_t head, std::size_t feature) const noexcept
  { return row * feature_row_stride + head * head_width + feature; }
  std::size_t attentionOffset(std::size_t head, std::size_t query, std::size_t key) const noexcept
  { return head * attention_head_stride + query * attention_row_stride + key; }
};

/** Configuration-major collection of independent attention problems.
 *
 * Feature and attention configuration strides include any desired tail padding.
 * Attention never crosses a configuration boundary.
 */
struct BatchedAttentionForwardLayout
{
  std::size_t configuration_count = 0;
  AttentionForwardLayout attention;
  std::size_t feature_configuration_stride   = 0;
  std::size_t attention_configuration_stride = 0;

  std::size_t softmaxRowCount() const noexcept
  { return configuration_count * attention.softmaxRowCount(); }
  std::size_t featureElements() const noexcept
  {
    return configuration_count == 0 ? 0
        : (configuration_count - 1) * feature_configuration_stride +
            attention.featureElements();
  }
  std::size_t attentionElements() const noexcept
  {
    return configuration_count == 0 ? 0
        : (configuration_count - 1) * attention_configuration_stride +
            attention.attentionElements();
  }
  std::size_t featureOffset(std::size_t configuration,
                            std::size_t row,
                            std::size_t head,
                            std::size_t feature) const noexcept
  {
    return configuration * feature_configuration_stride +
        attention.featureOffset(row, head, feature);
  }
  std::size_t attentionOffset(std::size_t configuration,
                              std::size_t head,
                              std::size_t query,
                              std::size_t key) const noexcept
  {
    return configuration * attention_configuration_stride +
        attention.attentionOffset(head, query, key);
  }
};

/** Configuration-major value tensor used by FP32 nonlinear and cast kernels. */
struct BatchedValueLayout
{
  std::size_t configuration_count = 0;
  std::size_t rows                 = 0;
  std::size_t width                = 0;
  std::size_t row_stride           = 0;
  std::size_t configuration_stride = 0;

  std::size_t logicalElements() const noexcept
  { return configuration_count * rows * width; }
  std::size_t storageElements() const noexcept
  {
    return configuration_count == 0 || rows == 0 ? 0
        : (configuration_count - 1) * configuration_stride +
            (rows - 1) * row_stride + width;
  }
  std::size_t offset(std::size_t configuration,
                     std::size_t row,
                     std::size_t feature) const noexcept
  {
    return configuration * configuration_stride + row * row_stride + feature;
  }
};

static_assert(std::is_trivially_copyable_v<DenseForwardLayout>);
static_assert(std::is_trivially_copyable_v<AttentionForwardLayout>);
static_assert(std::is_trivially_copyable_v<BatchedAttentionForwardLayout>);
static_assert(std::is_trivially_copyable_v<BatchedValueLayout>);

namespace attention_detail
{

/** IEEE-754 finite classification that remains valid under the project's fast-math flags. */
inline bool isFiniteBinary64(double value) noexcept
{
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

inline std::size_t checkedProduct(std::size_t left, std::size_t right, const char* description)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(description);
  return left * right;
}

inline std::size_t checkedMatrixSpan(std::size_t rows,
                                     std::size_t row_stride,
                                     std::size_t width,
                                     const char* description)
{
  if (rows == 0)
    return 0;
  const std::size_t prefix = checkedProduct(rows - 1, row_stride, description);
  if (width > std::numeric_limits<std::size_t>::max() - prefix)
    throw std::length_error(description);
  return prefix + width;
}

inline void requireBlasInteger(std::size_t extent, const char* description)
{
  if (extent > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error(description);
}

} // namespace attention_detail

inline void validateDenseForwardLayout(const DenseForwardLayout& layout)
{
  if (layout.rows == 0 || layout.input_width == 0 || layout.output_width == 0)
    throw std::invalid_argument("PsiFormer dense forward dimensions must be positive");
  if (layout.source_row_stride < layout.input_width ||
      layout.weight_row_stride < layout.output_width ||
      layout.target_row_stride < layout.output_width)
    throw std::invalid_argument("PsiFormer dense forward row stride is too small");
  attention_detail::requireBlasInteger(layout.rows, "PsiFormer dense row count exceeds the BLAS ABI");
  attention_detail::requireBlasInteger(layout.input_width, "PsiFormer dense input width exceeds the BLAS ABI");
  attention_detail::requireBlasInteger(layout.output_width, "PsiFormer dense output width exceeds the BLAS ABI");
  attention_detail::requireBlasInteger(layout.source_row_stride, "PsiFormer dense source stride exceeds the BLAS ABI");
  attention_detail::requireBlasInteger(layout.weight_row_stride, "PsiFormer dense weight stride exceeds the BLAS ABI");
  attention_detail::requireBlasInteger(layout.target_row_stride, "PsiFormer dense target stride exceeds the BLAS ABI");
  (void)attention_detail::checkedMatrixSpan(layout.rows, layout.source_row_stride,
                                             layout.input_width, "PsiFormer dense source extent overflow");
  (void)attention_detail::checkedMatrixSpan(layout.input_width, layout.weight_row_stride,
                                             layout.output_width, "PsiFormer dense weight extent overflow");
  (void)attention_detail::checkedMatrixSpan(layout.rows, layout.target_row_stride,
                                             layout.output_width, "PsiFormer dense target extent overflow");
}

inline DenseForwardLayout makeDenseForwardLayout(std::size_t rows,
                                                 std::size_t input_width,
                                                 std::size_t output_width,
                                                 std::size_t source_row_stride = 0,
                                                 std::size_t weight_row_stride = 0,
                                                 std::size_t target_row_stride = 0)
{
  DenseForwardLayout layout{rows, input_width, output_width,
                            source_row_stride == 0 ? input_width : source_row_stride,
                            weight_row_stride == 0 ? output_width : weight_row_stride,
                            target_row_stride == 0 ? output_width : target_row_stride};
  validateDenseForwardLayout(layout);
  return layout;
}

inline void validateAttentionForwardLayout(const AttentionForwardLayout& layout)
{
  if (layout.rows == 0 || layout.heads == 0 || layout.head_width == 0)
    throw std::invalid_argument("PsiFormer attention dimensions must be positive");
  const std::size_t width = attention_detail::checkedProduct(
      layout.heads, layout.head_width, "PsiFormer attention feature width overflow");
  if (layout.feature_row_stride < width || layout.attention_row_stride < layout.rows)
    throw std::invalid_argument("PsiFormer attention row stride is too small");
  const std::size_t one_head = attention_detail::checkedMatrixSpan(
      layout.rows, layout.attention_row_stride, layout.rows,
      "PsiFormer attention head extent overflow");
  if (layout.attention_head_stride < one_head)
    throw std::invalid_argument("PsiFormer attention head stride is too small");
  attention_detail::requireBlasInteger(layout.rows, "PsiFormer attention row count exceeds the BLAS ABI");
  attention_detail::requireBlasInteger(layout.head_width, "PsiFormer attention head width exceeds the BLAS ABI");
  attention_detail::requireBlasInteger(layout.feature_row_stride,
                                        "PsiFormer attention feature stride exceeds the BLAS ABI");
  attention_detail::requireBlasInteger(layout.attention_row_stride,
                                        "PsiFormer attention row stride exceeds the BLAS ABI");
  (void)attention_detail::checkedProduct(layout.heads, layout.rows,
                                          "PsiFormer attention softmax row count overflow");
  (void)attention_detail::checkedMatrixSpan(layout.rows, layout.feature_row_stride, width,
                                             "PsiFormer attention feature extent overflow");
  (void)attention_detail::checkedMatrixSpan(layout.heads, layout.attention_head_stride, one_head,
                                             "PsiFormer attention storage extent overflow");
}

inline AttentionForwardLayout makeAttentionForwardLayout(std::size_t rows,
                                                         std::size_t heads,
                                                         std::size_t head_width,
                                                         std::size_t feature_row_stride = 0,
                                                         std::size_t attention_row_stride = 0,
                                                         std::size_t attention_head_stride = 0)
{
  const std::size_t width = attention_detail::checkedProduct(
      heads, head_width, "PsiFormer attention feature width overflow");
  const std::size_t row_stride = attention_row_stride == 0 ? rows : attention_row_stride;
  const std::size_t one_head = attention_detail::checkedMatrixSpan(
      rows, row_stride, rows, "PsiFormer attention head extent overflow");
  AttentionForwardLayout layout{rows, heads, head_width,
                                feature_row_stride == 0 ? width : feature_row_stride,
                                row_stride,
                                attention_head_stride == 0 ? one_head : attention_head_stride};
  validateAttentionForwardLayout(layout);
  return layout;
}

/** Validate a configuration-major attention descriptor and all padded spans. */
inline void validateBatchedAttentionForwardLayout(const BatchedAttentionForwardLayout& layout)
{
  if (layout.configuration_count == 0)
    throw std::invalid_argument("PsiFormer attention batch must contain a configuration");
  validateAttentionForwardLayout(layout.attention);
  const std::size_t minimum_feature_stride = attention_detail::checkedProduct(
      layout.attention.rows, layout.attention.feature_row_stride,
      "PsiFormer attention configuration feature stride overflow");
  const std::size_t minimum_attention_stride = attention_detail::checkedProduct(
      layout.attention.heads, layout.attention.attention_head_stride,
      "PsiFormer attention configuration matrix stride overflow");
  if (layout.feature_configuration_stride < minimum_feature_stride ||
      layout.attention_configuration_stride < minimum_attention_stride)
    throw std::invalid_argument("PsiFormer attention configuration stride is too small");
  (void)attention_detail::checkedMatrixSpan(
      layout.configuration_count, layout.feature_configuration_stride,
      layout.attention.featureElements(), "PsiFormer batched attention feature extent overflow");
  (void)attention_detail::checkedMatrixSpan(
      layout.configuration_count, layout.attention_configuration_stride,
      layout.attention.attentionElements(), "PsiFormer batched attention matrix extent overflow");
  (void)attention_detail::checkedProduct(
      layout.configuration_count, layout.attention.softmaxRowCount(),
      "PsiFormer batched attention softmax row count overflow");
}

/** Construct a checked configuration-major attention descriptor. */
inline BatchedAttentionForwardLayout makeBatchedAttentionForwardLayout(
    std::size_t configuration_count,
    const AttentionForwardLayout& attention,
    std::size_t feature_configuration_stride = 0,
    std::size_t attention_configuration_stride = 0)
{
  const std::size_t minimum_feature_stride = attention_detail::checkedProduct(
      attention.rows, attention.feature_row_stride,
      "PsiFormer attention configuration feature stride overflow");
  const std::size_t minimum_attention_stride = attention_detail::checkedProduct(
      attention.heads, attention.attention_head_stride,
      "PsiFormer attention configuration matrix stride overflow");
  BatchedAttentionForwardLayout layout{
      configuration_count, attention,
      feature_configuration_stride == 0 ? minimum_feature_stride
                                        : feature_configuration_stride,
      attention_configuration_stride == 0 ? minimum_attention_stride
                                          : attention_configuration_stride};
  validateBatchedAttentionForwardLayout(layout);
  return layout;
}

/** Validate one padded configuration-major value tensor. */
inline void validateBatchedValueLayout(const BatchedValueLayout& layout)
{
  if (layout.configuration_count == 0 || layout.rows == 0 || layout.width == 0)
    throw std::invalid_argument("PsiFormer value batch dimensions must be positive");
  if (layout.row_stride < layout.width)
    throw std::invalid_argument("PsiFormer value row stride is too small");
  const std::size_t minimum_configuration_stride = attention_detail::checkedProduct(
      layout.rows, layout.row_stride, "PsiFormer value configuration stride overflow");
  if (layout.configuration_stride < minimum_configuration_stride)
    throw std::invalid_argument("PsiFormer value configuration stride is too small");
  (void)attention_detail::checkedProduct(
      layout.configuration_count,
      attention_detail::checkedProduct(layout.rows, layout.width,
                                       "PsiFormer value logical extent overflow"),
      "PsiFormer value batch logical extent overflow");
  (void)attention_detail::checkedMatrixSpan(
      layout.configuration_count, layout.configuration_stride,
      attention_detail::checkedMatrixSpan(layout.rows, layout.row_stride, layout.width,
                                          "PsiFormer value extent overflow"),
      "PsiFormer value batch extent overflow");
}

/** Construct one checked padded configuration-major value tensor. */
inline BatchedValueLayout makeBatchedValueLayout(
    std::size_t configuration_count,
    std::size_t rows,
    std::size_t width,
    std::size_t row_stride = 0,
    std::size_t configuration_stride = 0)
{
  const std::size_t actual_row_stride = row_stride == 0 ? width : row_stride;
  const std::size_t minimum_configuration_stride = attention_detail::checkedProduct(
      rows, actual_row_stride, "PsiFormer value configuration stride overflow");
  BatchedValueLayout layout{configuration_count, rows, width, actual_row_stride,
                            configuration_stride == 0 ? minimum_configuration_stride
                                                      : configuration_stride};
  validateBatchedValueLayout(layout);
  return layout;
}

/** Stable host execution of the exact scalar softmax transformation used on device. */
inline void stableSoftmaxRow(const double* logits, std::size_t width, double* weights)
{
  if (width == 0)
    throw std::invalid_argument("PsiFormer softmax row must be nonempty");
  for (std::size_t column = 0; column < width; ++column)
    if (!attention_detail::isFiniteBinary64(logits[column]))
      throw std::domain_error("PsiFormer softmax row contains a non-finite logit");
  const double maximum = *std::max_element(logits, logits + width);
  double normalization = 0;
  for (std::size_t column = 0; column < width; ++column)
  {
    weights[column] = device_math::shiftedExponential(logits[column], maximum);
    normalization += weights[column];
  }
  if (!attention_detail::isFiniteBinary64(normalization) || normalization <= 0)
    throw std::domain_error("PsiFormer softmax normalization is not finite and positive");
  for (std::size_t column = 0; column < width; ++column)
    weights[column] = device_math::normalizeExponential(weights[column], normalization);
}

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_ATTENTION_H
