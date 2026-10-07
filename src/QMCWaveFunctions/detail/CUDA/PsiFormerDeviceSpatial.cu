//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceSpatial.cu
 * @brief Shared CUDA/HIP spatial-jet kernels without warp-size assumptions.
 */

#include "PsiFormerDeviceSpatial.h"

#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer::device
{
namespace
{

constexpr unsigned int spatial_block_size = 128;

__global__ void softmaxJetRowsKernel(SoftmaxJetRowLayout layout,
                                     double* jets,
                                     device_math::JetMathStatus* row_status)
{
  const std::size_t batch_row = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t batch_row_count =
      layout.jets.configuration_count * layout.row_count;
  if (batch_row >= batch_row_count)
    return;
  const std::size_t configuration = batch_row / layout.row_count;
  const std::size_t row           = batch_row % layout.row_count;
  const std::size_t row_offset = configuration * layout.jets.configuration_stride +
      row * layout.row_stride;
  double* laplacian = layout.jets.laplacian_lanes == 0
      ? nullptr
      : jets + row_offset +
          (1 + layout.jets.gradient_lanes) * layout.jets.plane_stride;
  row_status[batch_row] = device_math::softmaxJetRowInPlace(
      jets + row_offset, jets + row_offset + layout.jets.plane_stride,
      laplacian, layout.row_width, layout.jets.plane_stride,
      layout.jets.gradient_lanes, layout.jets.laplacian_lanes);
}

} // namespace

Error launchSoftmaxJetRows(Stream stream,
                           const SoftmaxJetRowLayout& layout,
                           double* jets,
                           device_math::JetMathStatus* row_status)
{
  if (layout.jets.configuration_count == 0 && layout.row_count == 0 &&
      layout.row_width == 0)
    return success;
  validateSoftmaxJetRowLayout(layout);
  const std::size_t batch_row_count = softmaxJetBatchRowCount(layout);
  if (batch_row_count > std::numeric_limits<unsigned int>::max())
    throw std::length_error("PsiFormer softmax jet grid exceeds the CUDA/HIP x dimension");
  if (!jets || !row_status)
    throw std::invalid_argument("PsiFormer softmax jet device storage is null");

  const unsigned int block_count = static_cast<unsigned int>(
      (batch_row_count + spatial_block_size - 1) / spatial_block_size);
  softmaxJetRowsKernel<<<block_count, spatial_block_size, 0, stream>>>(
      layout, jets, row_status);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

} // namespace qmcplusplus::psiformer::device
