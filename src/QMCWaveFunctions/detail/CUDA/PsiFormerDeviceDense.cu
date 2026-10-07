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
#include "Platforms/CUDA/AccelBLAS_CUDA.hpp"

#include <cfloat>
#include <cmath>
#include <limits>

namespace qmcplusplus::psiformer::device
{
namespace
{

constexpr unsigned int softmax_block_size = 128;
constexpr unsigned int value_block_size   = 128;

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
  target[offset] = static_cast<float>(value);
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

} // namespace qmcplusplus::psiformer::device
