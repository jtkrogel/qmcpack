//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceDeterminantKernels.cu
 * @brief One-thread-per-matrix CUDA/HIP determinant correctness baseline.
 */

#include "PsiFormerDeviceDeterminantKernels.h"

#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer::device
{
namespace
{

constexpr unsigned int determinant_block_size = 128;

__global__ void determinantFactorizationKernel(
    const double* matrices,
    std::size_t matrix_count,
    std::size_t matrix_size,
    bool prepare_inverse,
    double* lu,
    double* inverse,
    std::size_t* permutation,
    double* solve,
    device_determinant::FactorizationMetadata* metadata)
{
  const std::size_t matrix = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (matrix >= matrix_count)
    return;
  const std::size_t matrix_offset = device_determinant::matrixOffset(matrix, matrix_size);
  const std::size_t vector_offset = matrix * matrix_size;
  metadata[matrix] = device_determinant::factorizeScaledReal(
      matrices + matrix_offset, matrix_size, lu + matrix_offset,
      permutation + vector_offset, prepare_inverse,
      inverse ? inverse + matrix_offset : nullptr,
      solve ? solve + vector_offset : nullptr);
}

std::size_t checkedProduct(std::size_t left, std::size_t right, const char* description)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(description);
  return left * right;
}

} // namespace

Error launchDeterminantFactorization(
    Stream stream,
    const double* matrices,
    std::size_t configuration_count,
    std::size_t determinant_count,
    std::size_t matrix_size,
    bool prepare_inverse,
    double* lu,
    double* inverse,
    std::size_t* permutation,
    double* solve,
    device_determinant::FactorizationMetadata* metadata)
{
  const std::size_t matrix_count = checkedProduct(
      configuration_count, determinant_count,
      "PsiFormer determinant matrix-count overflow");
  if (matrix_count == 0)
    return success;
  if (matrix_size == 0)
    throw std::invalid_argument("PsiFormer determinant matrix size must be positive");
  const std::size_t matrix_elements = checkedProduct(
      matrix_size, matrix_size, "PsiFormer determinant matrix extent overflow");
  (void)checkedProduct(matrix_count, matrix_elements,
                       "PsiFormer determinant batch matrix extent overflow");
  (void)checkedProduct(matrix_count, matrix_size,
                       "PsiFormer determinant batch vector extent overflow");
  if (matrix_count > std::numeric_limits<unsigned int>::max())
    throw std::length_error("PsiFormer determinant grid exceeds the CUDA/HIP x dimension");
  if (prepare_inverse && (!inverse || !solve))
    throw std::invalid_argument("PsiFormer determinant inverse requires inverse and solve storage");

  const unsigned int block_count = static_cast<unsigned int>(
      (matrix_count + determinant_block_size - 1) / determinant_block_size);
  determinantFactorizationKernel<<<block_count, determinant_block_size, 0, stream>>>(
      matrices, matrix_count, matrix_size, prepare_inverse, lu, inverse,
      permutation, solve, metadata);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

} // namespace qmcplusplus::psiformer::device
