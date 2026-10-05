//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDirectKernels.h
 * @brief Allocation-free real-scalar kernels shared by direct PsiFormer executors.
 *
 * The kernels expose explicit row-major forward and transpose-product operations.
 * They intentionally do not own storage or know parameter names.  A future complex
 * specialization can retain this interface while replacing each transpose product
 * with the adjoint convention selected by PsiFormerExecutionPlan.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DIRECT_KERNELS_H
#define QMCPLUSPLUS_PSIFORMER_DIRECT_KERNELS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer::direct
{

/** Compute target = source * weight + bias for row-major dense matrices. */
inline void denseForward(const double* source,
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
      std::copy(bias, bias + output_width, output_row);
    else
      std::fill(output_row, output_row + output_width, 0.0);

    for (std::size_t input = 0; input < input_width; ++input)
    {
      const double source_value = source[row * input_width + input];
      const double* weight_row  = weight + input * output_width;
      for (std::size_t output = 0; output < output_width; ++output)
        output_row[output] += source_value * weight_row[output];
    }
  }
}

/**
 * Accumulate the transpose products for one dense layer.
 *
 * All output adjoints are additive so residual branches can converge into the same
 * buffers.  Passing a null bias adjoint suppresses the bias reduction.
 */
inline void denseReverse(const double* source,
                         const double* weight,
                         const double* output_adjoint,
                         std::size_t rows,
                         std::size_t input_width,
                         std::size_t output_width,
                         double* source_adjoint,
                         double* weight_adjoint,
                         double* bias_adjoint)
{
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t input = 0; input < input_width; ++input)
    {
      const double source_value = source[row * input_width + input];
      const std::size_t parameter_row = input * output_width;
      double source_gradient = 0.0;
      for (std::size_t output = 0; output < output_width; ++output)
      {
        const double upstream = output_adjoint[row * output_width + output];
        source_gradient += upstream * weight[parameter_row + output];
        weight_adjoint[parameter_row + output] += source_value * upstream;
      }
      source_adjoint[row * input_width + input] += source_gradient;
    }
    if (bias_adjoint)
      for (std::size_t output = 0; output < output_width; ++output)
        bias_adjoint[output] += output_adjoint[row * output_width + output];
  }
}

/** Form stable per-head row softmax weights from electron-major Q and K features. */
inline void attentionWeightsForward(const double* query,
                                    const double* key,
                                    std::size_t electrons,
                                    std::size_t heads,
                                    std::size_t head_width,
                                    double* weights)
{
  const std::size_t width = heads * head_width;
  const double scale      = 1.0 / std::sqrt(static_cast<double>(head_width));
  for (std::size_t head = 0; head < heads; ++head)
    for (std::size_t query_electron = 0; query_electron < electrons; ++query_electron)
    {
      double* row        = weights + (head * electrons + query_electron) * electrons;
      double row_maximum = -std::numeric_limits<double>::infinity();
      for (std::size_t key_electron = 0; key_electron < electrons; ++key_electron)
      {
        double logit = 0.0;
        const std::size_t query_begin = query_electron * width + head * head_width;
        const std::size_t key_begin   = key_electron * width + head * head_width;
        for (std::size_t feature = 0; feature < head_width; ++feature)
          logit += query[query_begin + feature] * key[key_begin + feature];
        row[key_electron] = scale * logit;
        row_maximum       = std::max(row_maximum, row[key_electron]);
      }

      double normalization = 0.0;
      for (std::size_t key_electron = 0; key_electron < electrons; ++key_electron)
      {
        row[key_electron] = std::exp(row[key_electron] - row_maximum);
        normalization += row[key_electron];
      }
      for (std::size_t key_electron = 0; key_electron < electrons; ++key_electron)
        row[key_electron] /= normalization;
    }
}

/** Contract attention weights with electron-major value features. */
inline void attentionContextForward(const double* weights,
                                    const double* value,
                                    std::size_t electrons,
                                    std::size_t heads,
                                    std::size_t head_width,
                                    double* context)
{
  const std::size_t width = heads * head_width;
  std::fill(context, context + electrons * width, 0.0);
  for (std::size_t output_electron = 0; output_electron < electrons; ++output_electron)
    for (std::size_t head = 0; head < heads; ++head)
      for (std::size_t source_electron = 0; source_electron < electrons; ++source_electron)
      {
        const double attention = weights[(head * electrons + output_electron) * electrons + source_electron];
        const std::size_t output_begin = output_electron * width + head * head_width;
        const std::size_t source_begin = source_electron * width + head * head_width;
        for (std::size_t feature = 0; feature < head_width; ++feature)
          context[output_begin + feature] += attention * value[source_begin + feature];
      }
}

/** Reverse the attention-context contraction into weight and value adjoints. */
inline void attentionContextReverse(const double* weights,
                                    const double* value,
                                    const double* context_adjoint,
                                    std::size_t electrons,
                                    std::size_t heads,
                                    std::size_t head_width,
                                    double* weight_adjoint,
                                    double* value_adjoint)
{
  const std::size_t width = heads * head_width;
  for (std::size_t output_electron = 0; output_electron < electrons; ++output_electron)
    for (std::size_t head = 0; head < heads; ++head)
      for (std::size_t source_electron = 0; source_electron < electrons; ++source_electron)
      {
        const std::size_t attention_index =
            (head * electrons + output_electron) * electrons + source_electron;
        const std::size_t output_begin = output_electron * width + head * head_width;
        const std::size_t source_begin = source_electron * width + head * head_width;
        for (std::size_t feature = 0; feature < head_width; ++feature)
        {
          const double upstream = context_adjoint[output_begin + feature];
          weight_adjoint[attention_index] += upstream * value[source_begin + feature];
          value_adjoint[source_begin + feature] += weights[attention_index] * upstream;
        }
      }
}

/** Reverse row softmax and scaled Q*K^T into electron-major Q and K adjoints. */
inline void attentionWeightsReverse(const double* query,
                                    const double* key,
                                    const double* weights,
                                    double* weight_adjoint,
                                    std::size_t electrons,
                                    std::size_t heads,
                                    std::size_t head_width,
                                    double* query_adjoint,
                                    double* key_adjoint)
{
  const std::size_t width = heads * head_width;
  const double scale      = 1.0 / std::sqrt(static_cast<double>(head_width));
  for (std::size_t head = 0; head < heads; ++head)
    for (std::size_t query_electron = 0; query_electron < electrons; ++query_electron)
    {
      double* row_adjoint       = weight_adjoint + (head * electrons + query_electron) * electrons;
      const double* weight_row  = weights + (head * electrons + query_electron) * electrons;
      double weighted_row_sum   = 0.0;
      for (std::size_t key_electron = 0; key_electron < electrons; ++key_electron)
        weighted_row_sum += row_adjoint[key_electron] * weight_row[key_electron];

      for (std::size_t key_electron = 0; key_electron < electrons; ++key_electron)
      {
        const double logit_adjoint =
            weight_row[key_electron] * (row_adjoint[key_electron] - weighted_row_sum) * scale;
        row_adjoint[key_electron] = logit_adjoint;
        const std::size_t query_begin = query_electron * width + head * head_width;
        const std::size_t key_begin   = key_electron * width + head * head_width;
        for (std::size_t feature = 0; feature < head_width; ++feature)
        {
          query_adjoint[query_begin + feature] += logit_adjoint * key[key_begin + feature];
          key_adjoint[key_begin + feature] += logit_adjoint * query[query_begin + feature];
        }
      }
    }
}

/** Compute a determinant and inverse into caller-owned buffers by pivoted elimination. */
inline double determinantInverse(const double* matrix,
                                 std::size_t matrix_size,
                                 double* work,
                                 double* inverse)
{
  const std::size_t elements = matrix_size * matrix_size;
  std::copy(matrix, matrix + elements, work);
  std::fill(inverse, inverse + elements, 0.0);
  for (std::size_t row = 0; row < matrix_size; ++row)
    inverse[row * matrix_size + row] = 1.0;

  double determinant   = 1.0;
  int permutation_sign = 1;
  for (std::size_t column = 0; column < matrix_size; ++column)
  {
    std::size_t pivot_row = column;
    for (std::size_t row = column + 1; row < matrix_size; ++row)
      if (std::abs(work[row * matrix_size + column]) >
          std::abs(work[pivot_row * matrix_size + column]))
        pivot_row = row;
    if (std::abs(work[pivot_row * matrix_size + column]) < 1.0e-14)
      throw std::runtime_error("singular PsiFormer determinant");

    if (pivot_row != column)
    {
      for (std::size_t entry = 0; entry < matrix_size; ++entry)
      {
        std::swap(work[column * matrix_size + entry], work[pivot_row * matrix_size + entry]);
        std::swap(inverse[column * matrix_size + entry], inverse[pivot_row * matrix_size + entry]);
      }
      permutation_sign = -permutation_sign;
    }

    const double pivot = work[column * matrix_size + column];
    determinant *= pivot;
    for (std::size_t entry = 0; entry < matrix_size; ++entry)
    {
      work[column * matrix_size + entry] /= pivot;
      inverse[column * matrix_size + entry] /= pivot;
    }
    for (std::size_t row = 0; row < matrix_size; ++row)
      if (row != column)
      {
        const double factor = work[row * matrix_size + column];
        for (std::size_t entry = 0; entry < matrix_size; ++entry)
        {
          work[row * matrix_size + entry] -= factor * work[column * matrix_size + entry];
          inverse[row * matrix_size + entry] -= factor * inverse[column * matrix_size + entry];
        }
      }
  }
  return determinant * static_cast<double>(permutation_sign);
}

} // namespace qmcplusplus::psiformer::direct

#endif // QMCPLUSPLUS_PSIFORMER_DIRECT_KERNELS_H
