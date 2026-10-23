//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceDense.cu
 * @brief CUDA/HIP BLAS dispatch and stable row softmax for PsiFormer attention.
 */

#include "PsiFormerDeviceDense.h"
#include "PsiFormerDeviceBlasMathMode.h"
#include "PsiFormerDeviceMixedValueDense.h"
#include "PsiFormerDevicePrecision.h"
#include "PsiFormerDeviceInference.h"
#include "Platforms/CUDA/AccelBLAS_CUDA.hpp"

#include <cfloat>
#include <cmath>
#include <limits>
#include <sstream>

namespace qmcplusplus::psiformer::device
{
namespace
{

constexpr unsigned int softmax_block_size    = 128;
constexpr unsigned int value_block_size      = 128;
constexpr unsigned int dense_bias_block_size = 128;

/** Sum one output column in a fixed row order without auxiliary storage. */
__global__ void denseBiasAdjointKernel(DenseForwardLayout layout,
                                       const double* target_adjoint,
                                       double* bias_adjoint)
{
  const std::size_t output =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output >= layout.output_width)
    return;

  double sum = 0.0;
  for (std::size_t row = 0; row < layout.rows; ++row)
    sum += target_adjoint[row * layout.target_row_stride + output];
  bias_adjoint[output] += sum;
}

__global__ void attentionSoftmaxKernel(AttentionForwardLayout layout, double* attention)
{
  const std::size_t logical_row = blockIdx.x;
  const std::size_t head        = logical_row / layout.rows;
  const std::size_t query       = logical_row - head * layout.rows;
  double* row = attention + head * layout.attention_head_stride +
      query * layout.attention_row_stride;

  __shared__ double reduction[softmax_block_size];
  double local_maximum = -DBL_MAX;
  for (std::size_t key = threadIdx.x; key < layout.rows; key += blockDim.x)
    local_maximum = fmax(local_maximum, row[key]);
  reduction[threadIdx.x] = local_maximum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2)
  {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] = fmax(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    __syncthreads();
  }
  const double maximum = reduction[0];

  // Every thread must consume the maximum before a faster thread reuses the
  // shared reduction array for its exponential sum (including idle lanes).
  __syncthreads();

  double local_sum = 0;
  for (std::size_t key = threadIdx.x; key < layout.rows; key += blockDim.x)
  {
    const double exponential = device_math::shiftedExponential(row[key], maximum);
    row[key]                  = exponential;
    local_sum += exponential;
  }
  reduction[threadIdx.x] = local_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2)
  {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  const double normalization = reduction[0];
  for (std::size_t key = threadIdx.x; key < layout.rows; key += blockDim.x)
    row[key] = device_math::normalizeExponential(row[key], normalization);
}

/// Classify binary32 without relying on fast-math finite assumptions.
__device__ bool finiteFloat(float value)
{
  return (__float_as_uint(value) & 0x7f800000U) != 0x7f800000U;
}

/// Classify binary64 without relying on fast-math finite assumptions.
__device__ bool finiteDouble(double value)
{
  return (static_cast<unsigned long long>(__double_as_longlong(value)) &
          0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

/** Normalize one row per block while retaining FP64 maximum/sum reductions. */
__global__ void attentionSoftmaxFp32Kernel(BatchedAttentionForwardLayout layout,
                                           float* attention,
                                           PsiFormerDeviceNumericalDiagnostics* diagnostics)
{
  const std::size_t rows_per_configuration =
      layout.attention.heads * layout.attention.rows;
  const std::size_t configuration = blockIdx.x / rows_per_configuration;
  const std::size_t local_row = blockIdx.x - configuration * rows_per_configuration;
  const std::size_t head  = local_row / layout.attention.rows;
  const std::size_t query = local_row - head * layout.attention.rows;
  float* row = attention + configuration * layout.attention_configuration_stride +
      head * layout.attention.attention_head_stride +
      query * layout.attention.attention_row_stride;

  __shared__ double reduction[softmax_block_size];
  __shared__ unsigned long long invalid_values[softmax_block_size];
  double local_maximum = -DBL_MAX;
  unsigned long long local_invalid = 0;
  for (std::size_t key = threadIdx.x; key < layout.attention.rows; key += blockDim.x)
  {
    const float value = row[key];
    if (finiteFloat(value))
      local_maximum = fmax(local_maximum, static_cast<double>(value));
    else
      ++local_invalid;
  }
  reduction[threadIdx.x]      = local_maximum;
  invalid_values[threadIdx.x] = local_invalid;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2)
  {
    if (threadIdx.x < stride)
    {
      reduction[threadIdx.x] = fmax(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
      invalid_values[threadIdx.x] += invalid_values[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (invalid_values[0] != 0)
  {
    for (std::size_t key = threadIdx.x; key < layout.attention.rows; key += blockDim.x)
      row[key] = 0.0F;
    if (threadIdx.x == 0)
    {
      atomicAdd(reinterpret_cast<unsigned long long*>(&diagnostics->execution.nonfinite_count),
                invalid_values[0]);
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &diagnostics->execution.invalid_softmax_count),
                1ULL);
    }
    return;
  }

  const double maximum = reduction[0];
  // FP32 normalization shares the same reduction scratch lifetime as FP64.
  // Do not overwrite it until all participating threads have read the maximum.
  __syncthreads();
  double local_sum = 0.0;
  for (std::size_t key = threadIdx.x; key < layout.attention.rows; key += blockDim.x)
  {
    const float exponential =
        static_cast<float>(exp(static_cast<double>(row[key]) - maximum));
    row[key] = exponential;
    local_sum += static_cast<double>(exponential);
  }
  reduction[threadIdx.x] = local_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2)
  {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }

  const double normalization = reduction[0];
  const bool invalid_normalization =
      !finiteDouble(normalization) || normalization <= 0.0;
  if (invalid_normalization)
  {
    for (std::size_t key = threadIdx.x; key < layout.attention.rows; key += blockDim.x)
      row[key] = 0.0F;
    if (threadIdx.x == 0)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &diagnostics->execution.invalid_softmax_count),
                1ULL);
    return;
  }

  for (std::size_t key = threadIdx.x; key < layout.attention.rows; key += blockDim.x)
    row[key] = static_cast<float>(static_cast<double>(row[key]) / normalization);
}

/// Apply FP32 bias and tanh to each logical value without touching padding.
__global__ void biasTanhValueFp32Kernel(BatchedValueLayout layout,
                                        const float* input,
                                        const float* bias,
                                        float* output,
                                        PsiFormerDeviceNumericalDiagnostics* diagnostics)
{
  const std::size_t logical =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (logical >= layout.configuration_count * layout.rows * layout.width)
    return;
  const std::size_t feature = logical % layout.width;
  const std::size_t row_index = logical / layout.width;
  const std::size_t row = row_index % layout.rows;
  const std::size_t configuration = row_index / layout.rows;
  const std::size_t offset = configuration * layout.configuration_stride +
      row * layout.row_stride + feature;
  const float preactivation = input[offset] + bias[feature];
  if (!finiteFloat(preactivation))
  {
    output[offset] = 0.0F;
    atomicAdd(reinterpret_cast<unsigned long long*>(&diagnostics->execution.nonfinite_count),
              1ULL);
    return;
  }
  output[offset] = tanhf(preactivation);
}

/// Add FP32 residual values without touching padding.
__global__ void residualValueFp32Kernel(BatchedValueLayout layout,
                                        const float* left,
                                        const float* right,
                                        float* output)
{
  const std::size_t logical =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (logical >= layout.configuration_count * layout.rows * layout.width)
    return;
  const std::size_t feature = logical % layout.width;
  const std::size_t row_index = logical / layout.width;
  const std::size_t row = row_index % layout.rows;
  const std::size_t configuration = row_index / layout.rows;
  const std::size_t offset = configuration * layout.configuration_stride +
      row * layout.row_stride + feature;
  output[offset] = left[offset] + right[offset];
}

/** Cast FP64 logical values to FP32 while diagnosing unrepresentable inputs. */
__global__ void valueFp64ToFp32Kernel(BatchedValueLayout layout,
                                      const double* source,
                                      float* target,
                                      PsiFormerDeviceNumericalDiagnostics* diagnostics)
{
  const std::size_t logical =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (logical >= layout.configuration_count * layout.rows * layout.width)
    return;
  const std::size_t feature = logical % layout.width;
  const std::size_t row_index = logical / layout.width;
  const std::size_t row = row_index % layout.rows;
  const std::size_t configuration = row_index / layout.rows;
  const std::size_t offset = configuration * layout.configuration_stride +
      row * layout.row_stride + feature;
  const double value = source[offset];
  const unsigned long long magnitude_bits =
      static_cast<unsigned long long>(__double_as_longlong(value)) &
      0x7fffffffffffffffULL;
  const unsigned long long fp32_max_bits = static_cast<unsigned long long>(
      __double_as_longlong(static_cast<double>(FLT_MAX)));
  if (!finiteDouble(value) || magnitude_bits > fp32_max_bits)
  {
    target[offset] = 0.0F;
    atomicAdd(reinterpret_cast<unsigned long long*>(&diagnostics->execution.nonfinite_count),
              1ULL);
    return;
  }
  const float converted = static_cast<float>(value);
  target[offset] = converted;
  if (value != 0.0 && converted == 0.0F)
    atomicAdd(reinterpret_cast<unsigned long long*>(&diagnostics->conversion.underflow_to_zero_count), 1ULL);
  else if (converted != 0.0F && fabs(static_cast<double>(converted)) < static_cast<double>(FLT_MIN))
    atomicAdd(reinterpret_cast<unsigned long long*>(&diagnostics->conversion.subnormal_output_count), 1ULL);
}

/// Cross the explicit value-path type barrier while leaving padding untouched.
__global__ void valueFp32ToFp64Kernel(BatchedValueLayout layout,
                                      const float* source,
                                      double* target,
                                      PsiFormerDeviceNumericalDiagnostics* diagnostics)
{
  const std::size_t logical =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (logical >= layout.configuration_count * layout.rows * layout.width)
    return;
  const std::size_t feature = logical % layout.width;
  const std::size_t row_index = logical / layout.width;
  const std::size_t row = row_index % layout.rows;
  const std::size_t configuration = row_index / layout.rows;
  const std::size_t offset = configuration * layout.configuration_stride +
      row * layout.row_stride + feature;
  const float value = source[offset];
  if (!finiteFloat(value))
  {
    target[offset] = 0.0;
    atomicAdd(reinterpret_cast<unsigned long long*>(&diagnostics->execution.nonfinite_count),
              1ULL);
    return;
  }
  target[offset] = static_cast<double>(value);
}

inline int blasExtent(std::size_t extent) noexcept
{
  return static_cast<int>(extent);
}

/// Convert a checked logical extent into a portable one-dimensional grid size.
unsigned int valueBlockCount(std::size_t count)
{
  const std::size_t blocks = count / value_block_size +
      (count % value_block_size == 0 ? 0 : 1);
  if (blocks > std::numeric_limits<unsigned int>::max())
    throw std::length_error("PsiFormer FP32 value grid exceeds the CUDA/HIP x dimension");
  return static_cast<unsigned int>(blocks);
}

/// Issue one raw SGEMM after its caller has established the scoped math mode.
void denseForwardFp32Raw(AcceleratorBlasHandle& handle,
                         const DenseForwardLayout& layout,
                         const float* source,
                         const float* weight,
                         float* target)
{
  compute::BLAS::gemm(handle, 'N', 'N', blasExtent(layout.output_width),
                      blasExtent(layout.rows), blasExtent(layout.input_width), 1.0F,
                      weight, blasExtent(layout.weight_row_stride),
                      source, blasExtent(layout.source_row_stride), 0.0F,
                      target, blasExtent(layout.target_row_stride));
}

} // namespace

void denseForward(AcceleratorBlasHandle& handle,
                  const DenseForwardLayout& layout,
                  const double* source,
                  const double* weight,
                  double* target)
{
  validateDenseForwardLayout(layout);
  compute::BLAS::gemm(handle, 'N', 'N', blasExtent(layout.output_width),
                      blasExtent(layout.rows), blasExtent(layout.input_width), 1.0,
                      weight, blasExtent(layout.weight_row_stride),
                      source, blasExtent(layout.source_row_stride), 0.0,
                      target, blasExtent(layout.target_row_stride));
}

void accumulateDenseInputAdjoint(AcceleratorBlasHandle& handle,
                                 const DenseForwardLayout& layout,
                                 const double* target_adjoint,
                                 const double* weight,
                                 double* source_adjoint)
{
  validateDenseForwardLayout(layout);
  if (!target_adjoint || !weight || !source_adjoint)
    throw std::invalid_argument("PsiFormer dense input-adjoint storage is null");

  // Row-major target_adjoint * weight^T is the column-major product
  // weight * target_adjoint^T.  Preserve every padded row stride as its BLAS LD.
  compute::BLAS::gemm(handle, 'T', 'N', blasExtent(layout.input_width),
                      blasExtent(layout.rows), blasExtent(layout.output_width), 1.0,
                      weight, blasExtent(layout.weight_row_stride),
                      target_adjoint, blasExtent(layout.target_row_stride), 1.0,
                      source_adjoint, blasExtent(layout.source_row_stride));
}

void accumulateDenseWeightAdjoint(AcceleratorBlasHandle& handle,
                                  const DenseForwardLayout& layout,
                                  const double* source,
                                  const double* target_adjoint,
                                  double* weight_adjoint)
{
  validateDenseForwardLayout(layout);
  if (!source || !target_adjoint || !weight_adjoint)
    throw std::invalid_argument("PsiFormer dense weight-adjoint storage is null");

  // Update the column-major view weight_adjoint^T with
  // target_adjoint^T * source, which is source^T * target_adjoint row-major.
  compute::BLAS::gemm(handle, 'N', 'T', blasExtent(layout.output_width),
                      blasExtent(layout.input_width), blasExtent(layout.rows), 1.0,
                      target_adjoint, blasExtent(layout.target_row_stride),
                      source, blasExtent(layout.source_row_stride), 1.0,
                      weight_adjoint, blasExtent(layout.weight_row_stride));
}

Error launchDenseBiasAdjoint(Stream stream,
                             const DenseForwardLayout& layout,
                             const double* target_adjoint,
                             double* bias_adjoint)
{
  if (layout.rows == 0 && layout.input_width == 0 && layout.output_width == 0 &&
      layout.source_row_stride == 0 && layout.weight_row_stride == 0 &&
      layout.target_row_stride == 0)
    return success;
  validateDenseForwardLayout(layout);
  if (!target_adjoint || !bias_adjoint)
    throw std::invalid_argument("PsiFormer dense bias-adjoint storage is null");

  const std::size_t blocks = layout.output_width / dense_bias_block_size +
      (layout.output_width % dense_bias_block_size == 0 ? 0 : 1);
  if (blocks > std::numeric_limits<unsigned int>::max())
    throw std::length_error(
        "PsiFormer dense bias-adjoint grid exceeds the CUDA/HIP x dimension");
  denseBiasAdjointKernel<<<static_cast<unsigned int>(blocks),
                           dense_bias_block_size, 0, stream>>>(
      layout, target_adjoint, bias_adjoint);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

void projectQkvForward(AcceleratorBlasHandle& handle,
                       const DenseForwardLayout& layout,
                       const double* source,
                       const double* query_weight,
                       const double* key_weight,
                       const double* value_weight,
                       double* query,
                       double* key,
                       double* value)
{
  denseForward(handle, layout, source, query_weight, query);
  denseForward(handle, layout, source, key_weight, key);
  denseForward(handle, layout, source, value_weight, value);
}

void attentionLogitsForward(AcceleratorBlasHandle& handle,
                            const AttentionForwardLayout& layout,
                            const double* query,
                            const double* key,
                            double* logits)
{
  validateAttentionForwardLayout(layout);
  const double scale = 1.0 / std::sqrt(static_cast<double>(layout.head_width));
  for (std::size_t head = 0; head < layout.heads; ++head)
    compute::BLAS::gemm(handle, 'T', 'N', blasExtent(layout.rows), blasExtent(layout.rows),
                        blasExtent(layout.head_width), scale,
                        key + head * layout.head_width, blasExtent(layout.feature_row_stride),
                        query + head * layout.head_width, blasExtent(layout.feature_row_stride),
                        0.0, logits + head * layout.attention_head_stride,
                        blasExtent(layout.attention_row_stride));
}

Error launchAttentionSoftmax(Stream stream,
                             const AttentionForwardLayout& layout,
                             double* logits_and_weights)
{
  if (layout.rows == 0 && layout.heads == 0 && layout.head_width == 0)
    return success;
  validateAttentionForwardLayout(layout);
  const std::size_t row_count = layout.softmaxRowCount();
  if (row_count > std::numeric_limits<unsigned int>::max())
    throw std::length_error("PsiFormer attention softmax grid exceeds the CUDA/HIP x dimension");
  attentionSoftmaxKernel<<<static_cast<unsigned int>(row_count),
                           softmax_block_size, 0, stream>>>(layout, logits_and_weights);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

void attentionContextForward(AcceleratorBlasHandle& handle,
                             const AttentionForwardLayout& layout,
                             const double* attention,
                             const double* value,
                             double* target)
{
  validateAttentionForwardLayout(layout);
  for (std::size_t head = 0; head < layout.heads; ++head)
    compute::BLAS::gemm(handle, 'N', 'N', blasExtent(layout.head_width),
                        blasExtent(layout.rows), blasExtent(layout.rows), 1.0,
                        value + head * layout.head_width, blasExtent(layout.feature_row_stride),
                        attention + head * layout.attention_head_stride,
                        blasExtent(layout.attention_row_stride), 0.0,
                        target + head * layout.head_width, blasExtent(layout.feature_row_stride));
}

void denseForwardFp32(AcceleratorBlasHandle& handle,
                      const PsiFormerBlasMathModePlan& math_mode,
                      const DenseForwardLayout& layout,
                      const float* source,
                      const float* weight,
                      float* target)
{
  validateDenseForwardLayout(layout);
  executeWithDeviceBlasMathMode(handle, math_mode, [&] {
    denseForwardFp32Raw(handle, layout, source, weight, target);
  });
}

void projectQkvForwardFp32(AcceleratorBlasHandle& handle,
                           const PsiFormerBlasMathModePlan& math_mode,
                           const DenseForwardLayout& layout,
                           const float* source,
                           const float* query_weight,
                           const float* key_weight,
                           const float* value_weight,
                           float* query,
                           float* key,
                           float* value)
{
  validateDenseForwardLayout(layout);
  executeWithDeviceBlasMathMode(handle, math_mode, [&] {
    denseForwardFp32Raw(handle, layout, source, query_weight, query);
    denseForwardFp32Raw(handle, layout, source, key_weight, key);
    denseForwardFp32Raw(handle, layout, source, value_weight, value);
  });
}

void attentionLogitsForwardFp32(AcceleratorBlasHandle& handle,
                                const PsiFormerBlasMathModePlan& math_mode,
                                const BatchedAttentionForwardLayout& layout,
                                const float* query,
                                const float* key,
                                float* logits)
{
  validateBatchedAttentionForwardLayout(layout);
  const float scale = 1.0F / std::sqrt(static_cast<float>(layout.attention.head_width));
  executeWithDeviceBlasMathMode(handle, math_mode, [&] {
    for (std::size_t configuration = 0; configuration < layout.configuration_count;
         ++configuration)
      for (std::size_t head = 0; head < layout.attention.heads; ++head)
        compute::BLAS::gemm(
            handle, 'T', 'N', blasExtent(layout.attention.rows),
            blasExtent(layout.attention.rows), blasExtent(layout.attention.head_width),
            scale,
            key + configuration * layout.feature_configuration_stride +
                head * layout.attention.head_width,
            blasExtent(layout.attention.feature_row_stride),
            query + configuration * layout.feature_configuration_stride +
                head * layout.attention.head_width,
            blasExtent(layout.attention.feature_row_stride), 0.0F,
            logits + configuration * layout.attention_configuration_stride +
                head * layout.attention.attention_head_stride,
            blasExtent(layout.attention.attention_row_stride));
  });
}

Error launchAttentionSoftmaxFp32(Stream stream,
                                 const BatchedAttentionForwardLayout& layout,
                                 float* logits_and_weights,
                                 PsiFormerDeviceNumericalDiagnostics* diagnostics)
{
  if (layout.configuration_count == 0 && layout.attention.rows == 0 &&
      layout.attention.heads == 0 && layout.attention.head_width == 0)
    return success;
  validateBatchedAttentionForwardLayout(layout);
  if (!logits_and_weights || !diagnostics)
    throw std::invalid_argument("PsiFormer FP32 attention softmax storage is null");
  const std::size_t row_count = layout.softmaxRowCount();
  if (row_count > std::numeric_limits<unsigned int>::max())
    throw std::length_error(
        "PsiFormer FP32 attention softmax grid exceeds the CUDA/HIP x dimension");
  attentionSoftmaxFp32Kernel<<<static_cast<unsigned int>(row_count),
                               softmax_block_size, 0, stream>>>(
      layout, logits_and_weights, diagnostics);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

void attentionContextForwardFp32(AcceleratorBlasHandle& handle,
                                 const PsiFormerBlasMathModePlan& math_mode,
                                 const BatchedAttentionForwardLayout& layout,
                                 const float* attention,
                                 const float* value,
                                 float* target)
{
  validateBatchedAttentionForwardLayout(layout);
  executeWithDeviceBlasMathMode(handle, math_mode, [&] {
    for (std::size_t configuration = 0; configuration < layout.configuration_count;
         ++configuration)
      for (std::size_t head = 0; head < layout.attention.heads; ++head)
        compute::BLAS::gemm(
            handle, 'N', 'N', blasExtent(layout.attention.head_width),
            blasExtent(layout.attention.rows), blasExtent(layout.attention.rows), 1.0F,
            value + configuration * layout.feature_configuration_stride +
                head * layout.attention.head_width,
            blasExtent(layout.attention.feature_row_stride),
            attention + configuration * layout.attention_configuration_stride +
                head * layout.attention.attention_head_stride,
            blasExtent(layout.attention.attention_row_stride), 0.0F,
            target + configuration * layout.feature_configuration_stride +
                head * layout.attention.head_width,
            blasExtent(layout.attention.feature_row_stride));
  });
}

Error launchBiasTanhValueFp32(Stream stream,
                              const BatchedValueLayout& layout,
                              const float* input,
                              const float* bias,
                              float* output,
                              PsiFormerDeviceNumericalDiagnostics* diagnostics)
{
  if (layout.configuration_count == 0 && layout.rows == 0 && layout.width == 0)
    return success;
  validateBatchedValueLayout(layout);
  if (!input || !bias || !output || !diagnostics)
    throw std::invalid_argument("PsiFormer FP32 bias/tanh storage is null");
  const unsigned int blocks = valueBlockCount(layout.logicalElements());
  biasTanhValueFp32Kernel<<<blocks, value_block_size, 0, stream>>>(
      layout, input, bias, output, diagnostics);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

Error launchResidualValueFp32(Stream stream,
                              const BatchedValueLayout& layout,
                              const float* left,
                              const float* right,
                              float* output)
{
  if (layout.configuration_count == 0 && layout.rows == 0 && layout.width == 0)
    return success;
  validateBatchedValueLayout(layout);
  if (!left || !right || !output)
    throw std::invalid_argument("PsiFormer FP32 residual storage is null");
  const unsigned int blocks = valueBlockCount(layout.logicalElements());
  residualValueFp32Kernel<<<blocks, value_block_size, 0, stream>>>(
      layout, left, right, output);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

Error launchValueFp64ToFp32(Stream stream,
                            const BatchedValueLayout& layout,
                            const double* source,
                            float* target,
                            PsiFormerDeviceNumericalDiagnostics* diagnostics)
{
  if (layout.configuration_count == 0 && layout.rows == 0 && layout.width == 0)
    return success;
  validateBatchedValueLayout(layout);
  if (!source || !target || !diagnostics)
    throw std::invalid_argument("PsiFormer FP64-to-FP32 value storage is null");
  const unsigned int blocks = valueBlockCount(layout.logicalElements());
  valueFp64ToFp32Kernel<<<blocks, value_block_size, 0, stream>>>(
      layout, source, target, diagnostics);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

Error launchValueFp32ToFp64(Stream stream,
                            const BatchedValueLayout& layout,
                            const float* source,
                            double* target,
                            PsiFormerDeviceNumericalDiagnostics* diagnostics)
{
  if (layout.configuration_count == 0 && layout.rows == 0 && layout.width == 0)
    return success;
  validateBatchedValueLayout(layout);
  if (!source || !target || !diagnostics)
    throw std::invalid_argument("PsiFormer FP32-to-FP64 value storage is null");
  const unsigned int blocks = valueBlockCount(layout.logicalElements());
  valueFp32ToFp64Kernel<<<blocks, value_block_size, 0, stream>>>(
      layout, source, target, diagnostics);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

namespace
{

/// Check mixed scratch/traffic arithmetic before any allocation or launch.
std::size_t mixedCheckedMultiply(std::size_t count, std::size_t width)
{
  if (width != 0 && count > std::numeric_limits<std::size_t>::max() / width)
    throw std::overflow_error("PsiFormer mixed value extent overflow");
  return count * width;
}

/// Check accumulation of mixed scratch/cast accounting values.
std::size_t mixedCheckedAdd(std::size_t left, std::size_t right)
{
  if (left > std::numeric_limits<std::size_t>::max() - right)
    throw std::overflow_error("PsiFormer mixed value accounting overflow");
  return left + right;
}

/// Keep backflow/envelope construction and every derivative outside the FP32 cache.
bool isMixedValueDenseWeight(ParameterRole role)
{
  switch (role)
  {
  case ParameterRole::ELECTRON_EMBEDDING_WEIGHT:
  case ParameterRole::ATTENTION_QUERY_WEIGHT:
  case ParameterRole::ATTENTION_KEY_WEIGHT:
  case ParameterRole::ATTENTION_VALUE_WEIGHT:
  case ParameterRole::ATTENTION_OUTPUT_WEIGHT:
  case ParameterRole::UPDATE_HIDDEN_WEIGHT:
  case ParameterRole::UPDATE_OUTPUT_WEIGHT:
    return true;
  default:
    return false;
  }
}

/// Zero complete records, including padding, on their owning ordered queue.
void clearMixedDeviceStorage(Stream stream, void* data, std::size_t bytes)
{
#ifdef QMC_CUDA2HIP
  checkRuntime(hipMemsetAsync(data, 0, bytes, stream), "PsiFormer mixed value storage initialization");
#else
  checkRuntime(cudaMemsetAsync(data, 0, bytes, stream), "PsiFormer mixed value storage initialization");
#endif
}

/// This initial policy rejects subnormals/underflow instead of relying on vendor FTZ behavior.
bool mixedConversionNeedsRetry(const PsiFormerParameterConversionDiagnostics& diagnostics)
{
  return diagnostics.hasPublicationHazard() || diagnostics.subnormal_output_count != 0 ||
      diagnostics.underflow_to_zero_count != 0;
}

/** Screen approximate determinant heads more conservatively than FP64 solve admission. */
__global__ void mixedValueConditioningKernel(const device_determinant::FactorizationMetadata* factors,
    const device_determinant::CombinationMetadata* combinations, std::size_t configurations,
    std::size_t determinants, PsiFormerDeviceNumericalDiagnostics* diagnostics)
{
  const auto configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration >= configurations)
    return;
  // These conservative screening thresholds are not a forward-error bound.
  // They keep this first approximate route away from readily detected nodes and
  // nearly singular channels; full model-error qualification is still required.
  constexpr double minimum_scaled_pivot = 1.0e-3;
  constexpr double minimum_sum_fraction = 1.0e-3;
  unsigned long long bad_pivots = 0;
  for (std::size_t determinant = 0; determinant < determinants; ++determinant)
  {
    const auto& factor = factors[configuration * determinants + determinant];
    if (factor.status != device_determinant::FactorizationStatus::REGULAR ||
        !device_determinant::isFiniteBinary64(factor.minimum_scaled_pivot) ||
        factor.minimum_scaled_pivot < minimum_scaled_pivot)
      ++bad_pivots;
  }
  if (bad_pivots)
    atomicAdd(reinterpret_cast<unsigned long long*>(&diagnostics->execution.small_determinant_pivot_count), bad_pivots);
  const auto& sum = combinations[configuration];
  if (sum.status != device_determinant::CombinationStatus::REGULAR || sum.severe_cancellation ||
      !device_determinant::isFiniteBinary64(sum.sum_abs_scaled) ||
      !device_determinant::isFiniteBinary64(sum.abs_sum_scaled) ||
      sum.abs_sum_scaled < minimum_sum_fraction * sum.sum_abs_scaled)
    atomicAdd(reinterpret_cast<unsigned long long*>(&diagnostics->execution.severe_cancellation_count), 1ULL);
}

} // namespace

PreparedDeviceMixedDenseParameters::PreparedDeviceMixedDenseParameters(
    std::shared_ptr<const PreparedDeviceModel> model, compute::Queue<PlatformKind::CUDA>& queue,
    bool allow_fp64_retry)
    : model_(std::move(model))
{
  if (!model_)
    throw std::invalid_argument("Mixed dense parameters require an immutable canonical model");
  model_->validateCurrentDevice();
  const auto& environment = model_->plan().environment();
  if (environment.boundary != BoundaryCondition::OPEN ||
      environment.parameter_scalar_domain != ScalarDomain::REAL ||
      environment.compute_scalar_domain != ScalarDomain::REAL ||
      environment.amplitude_scalar_domain != ScalarDomain::REAL || !environment.fixed_nuclei)
    throw std::invalid_argument("Mixed value affine projections currently support fixed-ion open-real models only");
  const auto count = model_->plan().parameterCount();
  weights_.reserveNamed(count, "MixedDenseParameters::fp32_weights", __FILE__, __LINE__);
  clearMixedDeviceStorage(queue.getNative(), weights_.data(), checkedDeviceBytes<float>(count));
  DeviceBuffer<PsiFormerParameterConversionDiagnostics> diagnostics(1,
      "MixedDenseParameters::conversion_diagnostics", __FILE__, __LINE__);
  clearMixedDeviceStorage(queue.getNative(), diagnostics.data(), sizeof(PsiFormerParameterConversionDiagnostics));

  // Convert typed affine matrices only. Their canonical offsets remain stable,
  // while physical parameters and nonlinear biases continue using FP64 masters.
  constexpr std::size_t tile_capacity = 65536;
  constexpr std::size_t block_size = 128;
  for (const auto& tensor : model_->plan().parameterTensors())
    if (isMixedValueDenseWeight(tensor.role))
      for (std::size_t begin = tensor.begin; begin < tensor.end;)
      {
        const auto tile_count = std::min(tile_capacity, tensor.end - begin);
        const PsiFormerParameterConversionTile tile{begin, begin, tile_count,
            {tile_count, block_size, tile_count / block_size + (tile_count % block_size != 0), tile_count % block_size}};
        checkRuntime(launchFp64ToFp32ParameterConversion(queue.getNative(), tile,
            model_->parameters(), count, weights_.data(), count, diagnostics.data()), "mixed affine weight conversion");
        begin += tile_count;
      }
  queue.sync();
  diagnostics.download(&conversion_, 1);
  usable_ = !mixedConversionNeedsRetry(conversion_);
  if (!usable_ && !allow_fp64_retry)
    throw MixedPrecisionRetryRequired("Mixed affine parameter conversion requires whole-graph FP64 execution");
}

std::size_t PreparedDeviceMixedDenseParameters::deviceStorageBytes() const
{
  return checkedDeviceBytes<float>(weights_.capacity());
}

std::size_t MixedValueDenseStatistics::roundTripBytes() const
{
  return mixedCheckedMultiply(mixedCheckedAdd(input_cast_elements, output_cast_elements),
                              sizeof(double) + sizeof(float));
}

PreparedDeviceMixedValueDense::PreparedDeviceMixedValueDense(
    std::shared_ptr<const PreparedDeviceMixedDenseParameters> parameters, std::size_t maximum_rows,
    compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
    : parameters_(std::move(parameters)), queue_(queue), blas_(blas), maximum_rows_(maximum_rows)
{
  if (!parameters_ || maximum_rows_ == 0)
    throw std::invalid_argument("Mixed value projections require immutable parameters and positive row capacity");
  modelImage()->validateCurrentDevice();
#ifdef QMC_CUDA2HIP
  constexpr auto backend = PsiFormerAcceleratorBackend::HIP;
#else
  constexpr auto backend = PsiFormerAcceleratorBackend::CUDA;
#endif
  math_mode_ = makePsiFormerBlasMathModePlan(PsiFormerBackendMathMode::FP32_STRICT, backend);
  const auto& plan = modelImage()->plan();
  const auto width = plan.modelShape().feature_dimension;
  const auto input_width = plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0];
  source_.reserveNamed(mixedCheckedMultiply(maximum_rows_, std::max(input_width, width)),
      "MixedValueDense::fp32_source", __FILE__, __LINE__);
  target_.reserveNamed(mixedCheckedMultiply(maximum_rows_, width), "MixedValueDense::fp32_target", __FILE__, __LINE__);
  device_diagnostics_.reserveNamed(1, "MixedValueDense::diagnostics", __FILE__, __LINE__);
}

void PreparedDeviceMixedValueDense::beginEvaluation(const std::shared_ptr<const PreparedDeviceModel>& image)
{
  if (active_)
    throw std::logic_error("Mixed dense evaluation is already open");
  if (!image || image.get() != modelImage().get())
    throw std::invalid_argument("Mixed dense evaluation has a stale or foreign canonical image");
  image->validateCurrentDevice();
  diagnostics_ = {};
  statistics_ = {};
  if (!parameters_->usable())
    throw MixedPrecisionRetryRequired("FP32 affine weight conversion rejected; executing complete graph in FP64");
  clearMixedDeviceStorage(queue_.getNative(), device_diagnostics_.data(), sizeof(PsiFormerDeviceNumericalDiagnostics));
  active_ = true;
}

void PreparedDeviceMixedValueDense::project(const DenseForwardLayout& layout,
                                           const double* source, const double* weight, double* target)
{
  if (!active_)
    throw std::logic_error("Mixed dense projection has no open evaluation");
  try
  {
    validateDenseForwardLayout(layout);
    if (!source || !weight || !target || layout.rows == 0 || layout.rows > maximum_rows_ ||
        layout.source_row_stride != layout.input_width || layout.weight_row_stride != layout.output_width ||
        layout.target_row_stride != layout.output_width || layout.sourceElements() > source_.capacity() ||
        layout.targetElements() > target_.capacity())
      throw std::invalid_argument("Mixed value projection requires compact rows within prepared capacity");

    // Exact canonical pointers prevent using an FP32 shadow for a bias, physical
    // parameter, wrong layer or mismatched tensor dimensions.
    const ParameterTensorDescriptor* tensor = nullptr;
    for (const auto& candidate : modelImage()->plan().parameterTensors())
      if (isMixedValueDenseWeight(candidate.role) && weight == modelImage()->parameters() + candidate.begin)
      {
        tensor = &candidate;
        break;
      }
    if (!tensor || tensor->shape.size() != 2 || tensor->shape[0] != layout.input_width ||
        tensor->shape[1] != layout.output_width)
      throw std::invalid_argument("Mixed dense weight is not a matching canonical affine matrix");
    statistics_.projections = mixedCheckedAdd(statistics_.projections, 1);
    statistics_.input_cast_elements = mixedCheckedAdd(statistics_.input_cast_elements,
        mixedCheckedMultiply(layout.rows, layout.input_width));
    statistics_.output_cast_elements = mixedCheckedAdd(statistics_.output_cast_elements,
        mixedCheckedMultiply(layout.rows, layout.output_width));

    // Each affine boundary explicitly round-trips through FP64. This is not a
    // fully FP32-resident trunk, and the measured traffic must include both casts.
    const auto input = makeBatchedValueLayout(1, layout.rows, layout.input_width);
    const auto output = makeBatchedValueLayout(1, layout.rows, layout.output_width);
    checkRuntime(launchValueFp64ToFp32(queue_.getNative(), input, source, source_.data(),
        device_diagnostics_.data()), "mixed dense activation conversion");
    denseForwardFp32(blas_, math_mode_, layout, source_.data(), parameters_->parameters() + tensor->begin, target_.data());
    checkRuntime(launchValueFp32ToFp64(queue_.getNative(), output, target_.data(), target,
        device_diagnostics_.data()), "mixed dense output promotion");
  }
  catch (...)
  {
    active_ = false;
    throw;
  }
}

void PreparedDeviceMixedValueDense::projectSpatial(const SpatialDenseJetLayout& layout,
                                                   const double* source, const double* weight, double* target)
{
  if (layout.source.mode != SpatialJetMode::VALUE_ONLY && layout.source.mode != SpatialJetMode::FULL_VGL)
    throw std::invalid_argument("Mixed affine spatial projection supports value-only or full VGL");
  // Treat casts as identity maps in the standard mixed-AD convention. Every
  // coordinate gradient/trace-Laplacian plane uses the same quantized weights
  // and explicit FP32 product as the value plane, not an unmatched FP64 affine.
  project(makeFoldedSpatialDenseLayout(layout), source, weight, target);
}

void PreparedDeviceMixedValueDense::finishEvaluation()
{
  if (!active_)
    throw std::logic_error("Mixed dense admission has no open evaluation");
  active_ = false;
  queue_.sync();
  device_diagnostics_.download(&diagnostics_, 1);
  // Keep the device-leaf library independent of the host precision-policy object
  // (some driver test targets link these leaves without that host implementation).
  const auto& execution = diagnostics_.execution;
  if (execution.nonfinite_count || execution.invalid_softmax_count || execution.small_determinant_pivot_count ||
      execution.severe_cancellation_count || execution.extreme_ecp_ratio_count ||
      mixedConversionNeedsRetry(diagnostics_.conversion))
  {
    // Report the already downloaded evidence without changing admission or retry arithmetic.
    const auto& conversion = diagnostics_.conversion;
    std::ostringstream reason;
    reason.precision(17);
    reason << "Mixed affine evaluation requires a complete requested-graph FP64 retry"
           << "; execution_nonfinite=" << execution.nonfinite_count
           << " invalid_softmax=" << execution.invalid_softmax_count
           << " small_determinant_pivot=" << execution.small_determinant_pivot_count
           << " severe_cancellation=" << execution.severe_cancellation_count
           << " extreme_ecp_ratio=" << execution.extreme_ecp_ratio_count
           << " conversion_nonfinite=" << conversion.nonfinite_input_count
           << " conversion_overflow=" << conversion.overflow_count
           << " conversion_subnormal=" << conversion.subnormal_output_count
           << " conversion_underflow_to_zero=" << conversion.underflow_to_zero_count
           << " conversion_max_absolute_error=" << conversion.maximum_absolute_error
           << " conversion_max_relative_error=" << conversion.maximum_relative_error;
    throw MixedPrecisionRetryRequired(reason.str().c_str());
  }
}

void PreparedDeviceMixedValueDense::checkValueConditioning(
    const device_determinant::FactorizationMetadata* factors,
    const device_determinant::CombinationMetadata* combinations,
    std::size_t configurations, std::size_t determinants)
{
  if (!active_ || !factors || !combinations || configurations == 0 ||
      configurations > maximum_rows_ || determinants != modelImage()->plan().modelShape().determinants)
    throw std::invalid_argument("Mixed value conditioning requires a live matching determinant batch");
  mixedValueConditioningKernel<<<valueBlockCount(configurations), value_block_size, 0, queue_.getNative()>>>(
      factors, combinations, configurations, determinants, device_diagnostics_.data());
#ifdef QMC_CUDA2HIP
  checkRuntime(hipPeekAtLastError(), "mixed value conditioning admission");
#else
  checkRuntime(cudaPeekAtLastError(), "mixed value conditioning admission");
#endif
}

void PreparedDeviceMixedValueDense::rebindParameters(
    std::shared_ptr<const PreparedDeviceMixedDenseParameters> successor)
{
  if (active_)
    throw std::logic_error("Cannot rebind mixed parameters during an open evaluation");
  if (!successor)
    throw std::invalid_argument("Mixed dense successor cache is null");
  modelImage()->validateSuccessor(successor->modelImage());
  parameters_ = std::move(successor);
  diagnostics_ = {};
  statistics_ = {};
}

const std::shared_ptr<const PreparedDeviceModel>& PreparedDeviceMixedValueDense::modelImage() const noexcept
{
  return parameters_->modelImage();
}

std::size_t PreparedDeviceMixedValueDense::deviceWorkspaceBytes() const
{
  return mixedCheckedAdd(mixedCheckedAdd(checkedDeviceBytes<float>(source_.capacity()),
      checkedDeviceBytes<float>(target_.capacity())),
      checkedDeviceBytes<PsiFormerDeviceNumericalDiagnostics>(device_diagnostics_.capacity()));
}

} // namespace qmcplusplus::psiformer::device
