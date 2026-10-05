//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDenseKernels.h
 * @brief Allocation-free CPU BLAS kernels for the real molecular PsiFormer path.
 *
 * Imported tensors and activations are row-major.  QMCPACK's BLAS wrapper exposes
 * the conventional column-major interface, so each product is issued as the
 * algebraically equivalent transposed product without moving data.  This revision
 * implements only real arithmetic.  The execution plan remains the authority for
 * scalar domain and reverse-product convention; a later complex implementation must
 * select transpose versus conjugate-transpose from that plan rather than copying the
 * real reverse formulas.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DENSE_KERNELS_H
#define QMCPLUSPLUS_PSIFORMER_DENSE_KERNELS_H

#include "CPU/BLAS.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer::dense
{

/// Convert one BLAS dimension only after enforcing the configured integer ABI.
inline int checkedBlasDimension(std::size_t extent, const char* quantity)
{
  if (extent > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error(quantity);
  return static_cast<int>(extent);
}

/** Apply a small row-major dense product with a compiler-vectorizable loop.
 *
 * This path avoids BLAS call overhead for the narrow electron embedding.  The
 * network's 256-by-256 learned projections use `productReal`'s BLAS path.
 */
inline void productScalarReal(const double* source,
                              const double* weight,
                              const double* bias,
                              std::size_t rows,
                              std::size_t input_width,
                              std::size_t output_width,
                              double* target)
{
  for (std::size_t row = 0; row < rows; ++row)
  {
    double* output_row = target + row * output_width;
    if (bias)
      std::copy_n(bias, output_width, output_row);
    else
      std::fill_n(output_row, output_width, 0.0);

    for (std::size_t input = 0; input < input_width; ++input)
    {
      const double source_value = source[row * input_width + input];
      const double* weight_row  = weight + input * output_width;
      for (std::size_t output = 0; output < output_width; ++output)
        output_row[output] += source_value * weight_row[output];
    }
  }
}

/** Apply row-major C=A*B through QMCPACK's column-major real BLAS wrapper.
 *
 * C[M,N] in row-major storage is C^T[N,M] in column-major storage, hence the
 * operand order B then A.  No packing, allocation, or physical transpose occurs.
 */
inline void productBlasReal(const double* source,
                            const double* weight,
                            const double* bias,
                            std::size_t rows,
                            std::size_t input_width,
                            std::size_t output_width,
                            double* target)
{
  const int blas_rows = checkedBlasDimension(
      rows, "PsiFormer dense row count exceeds the BLAS integer ABI");
  const int blas_input_width = checkedBlasDimension(
      input_width, "PsiFormer dense input width exceeds the BLAS integer ABI");
  const int blas_output_width = checkedBlasDimension(
      output_width, "PsiFormer dense output width exceeds the BLAS integer ABI");
  BLAS::gemm('N', 'N', blas_output_width, blas_rows, blas_input_width, 1.0,
             weight, blas_output_width, source, blas_input_width, 0.0, target,
             blas_output_width);

  if (bias)
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t output = 0; output < output_width; ++output)
        target[row * output_width + output] += bias[output];
}

/** Dispatch a real dense product between the narrow scalar and matrix BLAS paths.
 *
 * The threshold is deliberately based on the contracted and output dimensions,
 * not solely the electron count: measurements show BLAS wins for the 256-wide
 * network layers even for LiH's four rows, while the 9- or 17-wide embedding remains
 * faster in the vectorized loop.  Model construction must ensure dimensions fit the
 * BLAS integer ABI.
 */
inline void productReal(const double* source,
                        const double* weight,
                        const double* bias,
                        std::size_t rows,
                        std::size_t input_width,
                        std::size_t output_width,
                        double* target)
{
  constexpr std::size_t blas_width_threshold = 64;
  if (input_width >= blas_width_threshold && output_width >= blas_width_threshold)
    productBlasReal(source, weight, bias, rows, input_width, output_width, target);
  else
    productScalarReal(source, weight, bias, rows, input_width, output_width, target);
}

/** Accumulate the input adjoint of a row-major dense product with real BLAS.
 *
 * For Y=X*W, this evaluates X_bar += Y_bar*W^T.  The caller owns and
 * preinitializes the destination because several PsiFormer branches may contribute
 * to the same adjoint.  Stacking independent trace-jet planes in the row dimension
 * lets one GEMM process all Cartesian or Laplacian lanes without packing.
 */
inline void accumulateInputAdjointReal(const double* target_adjoint,
                                       const double* weight,
                                       std::size_t stacked_rows,
                                       std::size_t input_width,
                                       std::size_t output_width,
                                       double* source_adjoint)
{
  if (stacked_rows == 0)
    return;

  const int blas_rows = checkedBlasDimension(
      stacked_rows, "PsiFormer adjoint row count exceeds the BLAS integer ABI");
  const int blas_input_width = checkedBlasDimension(
      input_width, "PsiFormer adjoint input width exceeds the BLAS integer ABI");
  const int blas_output_width = checkedBlasDimension(
      output_width, "PsiFormer adjoint output width exceeds the BLAS integer ABI");

  // Column-major W*Y_bar^T has the row-major storage of Y_bar*W^T.
  BLAS::gemm('T', 'N', blas_input_width, blas_rows, blas_output_width, 1.0,
             weight, blas_output_width, target_adjoint, blas_output_width, 1.0,
             source_adjoint, blas_input_width);
}

/** Accumulate the weight adjoint of a row-major dense product with real BLAS.
 *
 * For Y=X*W, this evaluates W_bar += X^T*Y_bar.  In column-major terms the
 * row-major destination is updated through W_bar^T += Y_bar^T*X.  As with the
 * input-adjoint kernel, `stacked_rows` may combine independent trace-jet planes.
 */
inline void accumulateWeightAdjointReal(const double* source,
                                        const double* target_adjoint,
                                        std::size_t stacked_rows,
                                        std::size_t input_width,
                                        std::size_t output_width,
                                        double* weight_adjoint)
{
  if (stacked_rows == 0)
    return;

  const int blas_rows = checkedBlasDimension(
      stacked_rows, "PsiFormer adjoint row count exceeds the BLAS integer ABI");
  const int blas_input_width = checkedBlasDimension(
      input_width, "PsiFormer adjoint input width exceeds the BLAS integer ABI");
  const int blas_output_width = checkedBlasDimension(
      output_width, "PsiFormer adjoint output width exceeds the BLAS integer ABI");

  BLAS::gemm('N', 'T', blas_output_width, blas_input_width, blas_rows, 1.0,
             target_adjoint, blas_output_width, source, blas_input_width, 1.0,
             weight_adjoint, blas_output_width);
}

/** Project a shared feature matrix into separate Q, K, and V buffers.
 *
 * Three products retain the imported parameter layout.  A single packed product was
 * measured 5--7% slower for the target LiH shapes and would duplicate 1.5 MiB of
 * weights per 256-wide block, so no versioned packed cache is maintained.
 */
inline void projectQkvReal(const double* source,
                           const double* query_weight,
                           const double* key_weight,
                           const double* value_weight,
                           std::size_t rows,
                           std::size_t width,
                           double* query,
                           double* key,
                           double* value)
{
  productBlasReal(source, query_weight, nullptr, rows, width, width, query);
  productBlasReal(source, key_weight, nullptr, rows, width, width, key);
  productBlasReal(source, value_weight, nullptr, rows, width, width, value);
}

/** Form stable head-major softmax(Q*K^T/sqrt(head_width)) with strided BLAS.
 *
 * Each electron row already contains contiguous features for every head.  Leading
 * dimensions preserve the electron-major stride, avoiding head packing.
 */
inline void attentionWeightsReal(const double* query,
                                 std::size_t query_stride,
                                 const double* key,
                                 std::size_t key_stride,
                                 std::size_t rows,
                                 std::size_t heads,
                                 std::size_t head_width,
                                 double* attention)
{
  const int blas_rows = checkedBlasDimension(
      rows, "PsiFormer attention row count exceeds the BLAS integer ABI");
  const int blas_head_width = checkedBlasDimension(
      head_width, "PsiFormer attention head width exceeds the BLAS integer ABI");
  const int blas_query_stride = checkedBlasDimension(
      query_stride, "PsiFormer attention query stride exceeds the BLAS integer ABI");
  const int blas_key_stride = checkedBlasDimension(
      key_stride, "PsiFormer attention key stride exceeds the BLAS integer ABI");
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_width));
  for (std::size_t head = 0; head < heads; ++head)
  {
    double* head_attention = attention + head * rows * rows;

    // Column-major K^T*Q has the row-major storage of Q*K^T.
    BLAS::gemm('T', 'N', blas_rows, blas_rows, blas_head_width, scale,
               key + head * head_width, blas_key_stride,
               query + head * head_width, blas_query_stride, 0.0,
               head_attention, blas_rows);

    for (std::size_t query_row = 0; query_row < rows; ++query_row)
    {
      double* attention_row = head_attention + query_row * rows;
      const double maximum = *std::max_element(attention_row, attention_row + rows);
      double normalization = 0.0;
      for (std::size_t key_row = 0; key_row < rows; ++key_row)
      {
        attention_row[key_row] = std::exp(attention_row[key_row] - maximum);
        normalization += attention_row[key_row];
      }
      for (std::size_t key_row = 0; key_row < rows; ++key_row)
        attention_row[key_row] /= normalization;
    }
  }
}

/** Contract head-major attention weights with V into electron-major features.
 *
 * Strided leading dimensions write each head directly into its final feature slice;
 * neither V nor the output is repacked.
 */
inline void attentionContextReal(const double* attention,
                                 const double* value,
                                 std::size_t value_stride,
                                 std::size_t rows,
                                 std::size_t heads,
                                 std::size_t head_width,
                                 double* target)
{
  const std::size_t width = heads * head_width;
  const int blas_rows = checkedBlasDimension(
      rows, "PsiFormer attention row count exceeds the BLAS integer ABI");
  const int blas_head_width = checkedBlasDimension(
      head_width, "PsiFormer attention head width exceeds the BLAS integer ABI");
  const int blas_value_stride = checkedBlasDimension(
      value_stride, "PsiFormer attention value stride exceeds the BLAS integer ABI");
  const int blas_width = checkedBlasDimension(
      width, "PsiFormer attention output stride exceeds the BLAS integer ABI");
  for (std::size_t head = 0; head < heads; ++head)
  {
    // Column-major V^T*A^T has the row-major storage of A*V.
    BLAS::gemm('N', 'N', blas_head_width, blas_rows, blas_rows, 1.0,
               value + head * head_width, blas_value_stride,
               attention + head * rows * rows, blas_rows, 0.0,
               target + head * head_width, blas_width);
  }
}

} // namespace qmcplusplus::psiformer::dense

#endif
