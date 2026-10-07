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
#include "Platforms/CUDA/AccelBLAS_CUDA.hpp"

#include <cfloat>
#include <cmath>
#include <limits>

namespace qmcplusplus::psiformer::device
{
namespace
{

constexpr unsigned int softmax_block_size = 128;

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

inline int blasExtent(std::size_t extent) noexcept
{
  return static_cast<int>(extent);
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

} // namespace qmcplusplus::psiformer::device
