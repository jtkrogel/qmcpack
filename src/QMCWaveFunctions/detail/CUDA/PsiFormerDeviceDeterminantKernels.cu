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

__global__ void determinantCombinationKernel(
    const device_determinant::FactorizationMetadata* channel_metadata,
    const double* coefficients,
    std::size_t configuration_count,
    std::size_t determinant_count,
    double* term_phase,
    double* term_log_abs,
    double* scaled_terms,
    double* normalized_weights,
    device_determinant::CombinationMetadata* combination_metadata)
{
  const std::size_t configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration >= configuration_count)
    return;
  const std::size_t offset = configuration * determinant_count;
  combination_metadata[configuration] = device_determinant::combineChannelsReal(
      channel_metadata + offset, coefficients, determinant_count,
      term_phase + offset, term_log_abs + offset, scaled_terms + offset,
      normalized_weights + offset);
}

__global__ void determinantMatrixReverseSeedsKernel(
    const device_determinant::FactorizationMetadata* factorization_metadata,
    const device_determinant::CombinationMetadata* combination_metadata,
    const double* normalized_weights,
    const double* inverses,
    std::size_t configuration_count,
    std::size_t determinant_count,
    std::size_t matrix_size,
    double* reverse_seeds,
    device_determinant::DerivativeStatus* status)
{
  const std::size_t configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration >= configuration_count)
    return;
  const std::size_t matrix_elements  = matrix_size * matrix_size;
  const std::size_t channel_elements = determinant_count * matrix_elements;
  const std::size_t channel_offset   = configuration * determinant_count;
  const std::size_t matrix_offset    = configuration * channel_elements;
  status[configuration] = device_determinant::fillMatrixReverseSeeds(
      factorization_metadata + channel_offset, combination_metadata[configuration],
      normalized_weights + channel_offset, inverses + matrix_offset,
      determinant_count, matrix_size, reverse_seeds + matrix_offset);
}

__global__ void determinantSpatialTracesKernel(
    device_determinant::SpatialLayout layout,
    const device_determinant::FactorizationMetadata* factorization_metadata,
    const device_determinant::CombinationMetadata* combination_metadata,
    const double* normalized_weights,
    const double* inverses,
    const double* matrix_gradients,
    const double* matrix_laplacians,
    double* matrix_product_scratch,
    double* output_log_gradient,
    double* output_lap_ratio,
    double* output_lap_log,
    device_determinant::DerivativeStatus* status)
{
  const std::size_t configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration >= layout.configuration_count)
    return;
  const std::size_t matrix_elements  = layout.matrix_size * layout.matrix_size;
  const std::size_t channel_elements = layout.determinant_count * matrix_elements;
  const std::size_t channel_offset   = configuration * layout.determinant_count;
  const std::size_t matrix_offset    = configuration * channel_elements;
  const std::size_t gradient_offset  = configuration * layout.gradient_lanes * channel_elements;
  const std::size_t laplacian_offset = configuration * layout.electron_count * channel_elements;
  status[configuration] = device_determinant::combineSpatialTraces(
      factorization_metadata + channel_offset, combination_metadata[configuration],
      normalized_weights + channel_offset, inverses + matrix_offset,
      matrix_gradients ? matrix_gradients + gradient_offset : nullptr,
      matrix_laplacians ? matrix_laplacians + laplacian_offset : nullptr,
      layout.determinant_count, layout.matrix_size, layout.gradient_lanes,
      layout.electron_count,
      matrix_product_scratch ? matrix_product_scratch + configuration * matrix_elements : nullptr,
      output_log_gradient ? output_log_gradient + configuration * layout.gradient_lanes : nullptr,
      output_lap_ratio ? output_lap_ratio + configuration * layout.electron_count : nullptr,
      output_lap_log ? output_lap_log + configuration * layout.electron_count : nullptr);
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

Error launchDeterminantCombination(
    Stream stream,
    const device_determinant::FactorizationMetadata* channel_metadata,
    const double* coefficients,
    std::size_t configuration_count,
    std::size_t determinant_count,
    double* term_phase,
    double* term_log_abs,
    double* scaled_terms,
    double* normalized_weights,
    device_determinant::CombinationMetadata* combination_metadata)
{
  if (configuration_count == 0)
    return success;
  if (determinant_count == 0)
    throw std::invalid_argument("PsiFormer determinant combination requires channels");
  (void)checkedProduct(configuration_count, determinant_count,
                       "PsiFormer determinant combination extent overflow");
  if (configuration_count > std::numeric_limits<unsigned int>::max())
    throw std::length_error("PsiFormer determinant combination grid exceeds the CUDA/HIP x dimension");
  if (!channel_metadata || !term_phase || !term_log_abs || !scaled_terms ||
      !normalized_weights || !combination_metadata)
    throw std::invalid_argument("PsiFormer determinant combination storage is null");

  const unsigned int block_count = static_cast<unsigned int>(
      (configuration_count + determinant_block_size - 1) / determinant_block_size);
  determinantCombinationKernel<<<block_count, determinant_block_size, 0, stream>>>(
      channel_metadata, coefficients, configuration_count, determinant_count,
      term_phase, term_log_abs, scaled_terms, normalized_weights,
      combination_metadata);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

Error launchDeterminantMatrixReverseSeeds(
    Stream stream,
    const device_determinant::FactorizationMetadata* factorization_metadata,
    const device_determinant::CombinationMetadata* combination_metadata,
    const double* normalized_weights,
    const double* inverses,
    std::size_t configuration_count,
    std::size_t determinant_count,
    std::size_t matrix_size,
    double* reverse_seeds,
    device_determinant::DerivativeStatus* status)
{
  if (configuration_count == 0)
    return success;
  if (determinant_count == 0 || matrix_size == 0)
    throw std::invalid_argument("PsiFormer determinant reverse dimensions must be positive");
  const std::size_t matrix_elements = checkedProduct(
      matrix_size, matrix_size, "PsiFormer determinant reverse matrix extent overflow");
  const std::size_t channel_elements = checkedProduct(
      determinant_count, matrix_elements, "PsiFormer determinant reverse channel extent overflow");
  (void)checkedProduct(configuration_count, determinant_count,
                       "PsiFormer determinant reverse metadata extent overflow");
  (void)checkedProduct(configuration_count, channel_elements,
                       "PsiFormer determinant reverse output extent overflow");
  if (configuration_count > std::numeric_limits<unsigned int>::max())
    throw std::length_error("PsiFormer determinant reverse grid exceeds the CUDA/HIP x dimension");
  if (!factorization_metadata || !combination_metadata || !normalized_weights ||
      !inverses || !reverse_seeds || !status)
    throw std::invalid_argument("PsiFormer determinant reverse storage is null");

  const unsigned int block_count = static_cast<unsigned int>(
      (configuration_count + determinant_block_size - 1) / determinant_block_size);
  determinantMatrixReverseSeedsKernel<<<block_count, determinant_block_size, 0, stream>>>(
      factorization_metadata, combination_metadata, normalized_weights, inverses,
      configuration_count, determinant_count, matrix_size, reverse_seeds, status);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

Error launchDeterminantSpatialTraces(
    Stream stream,
    device_determinant::SpatialLayout layout,
    const device_determinant::FactorizationMetadata* factorization_metadata,
    const device_determinant::CombinationMetadata* combination_metadata,
    const double* normalized_weights,
    const double* inverses,
    const double* matrix_gradients,
    const double* matrix_laplacians,
    double* matrix_product_scratch,
    double* output_log_gradient,
    double* output_lap_ratio,
    double* output_lap_log,
    device_determinant::DerivativeStatus* status)
{
  if (layout.configuration_count == 0)
    return success;
  if (layout.determinant_count == 0 || layout.matrix_size == 0)
    throw std::invalid_argument("PsiFormer determinant spatial dimensions must be positive");
  const std::size_t required_gradient_lanes = checkedProduct(
      layout.electron_count, std::size_t{3},
      "PsiFormer determinant spatial lane count overflow");
  if (layout.electron_count != 0 && layout.gradient_lanes != required_gradient_lanes)
    throw std::invalid_argument("PsiFormer determinant spatial Laplacians require three lanes per electron");
  const std::size_t matrix_elements = checkedProduct(
      layout.matrix_size, layout.matrix_size,
      "PsiFormer determinant spatial matrix extent overflow");
  const std::size_t channel_elements = checkedProduct(
      layout.determinant_count, matrix_elements,
      "PsiFormer determinant spatial channel extent overflow");
  const std::size_t metadata_elements = checkedProduct(
      layout.configuration_count, layout.determinant_count,
      "PsiFormer determinant spatial metadata extent overflow");
  const std::size_t gradient_configuration_elements = checkedProduct(
      layout.gradient_lanes, channel_elements,
      "PsiFormer determinant spatial gradient configuration extent overflow");
  const std::size_t laplacian_configuration_elements = checkedProduct(
      layout.electron_count, channel_elements,
      "PsiFormer determinant spatial Laplacian configuration extent overflow");
  (void)metadata_elements;
  (void)checkedProduct(layout.configuration_count, channel_elements,
                       "PsiFormer determinant spatial inverse extent overflow");
  (void)checkedProduct(layout.configuration_count, gradient_configuration_elements,
                       "PsiFormer determinant spatial gradient extent overflow");
  (void)checkedProduct(layout.configuration_count, laplacian_configuration_elements,
                       "PsiFormer determinant spatial Laplacian extent overflow");
  (void)checkedProduct(layout.configuration_count, matrix_elements,
                       "PsiFormer determinant spatial scratch extent overflow");
  if (layout.configuration_count > std::numeric_limits<unsigned int>::max())
    throw std::length_error("PsiFormer determinant spatial grid exceeds the CUDA/HIP x dimension");
  if (!factorization_metadata || !combination_metadata || !normalized_weights ||
      !inverses || !status)
    throw std::invalid_argument("PsiFormer determinant spatial state storage is null");
  if (layout.gradient_lanes != 0 && (!matrix_gradients || !output_log_gradient))
    throw std::invalid_argument("PsiFormer determinant spatial gradient storage is null");
  if (layout.electron_count != 0 &&
      (!matrix_laplacians || !matrix_product_scratch || !output_lap_ratio || !output_lap_log))
    throw std::invalid_argument("PsiFormer determinant spatial Laplacian storage is null");

  const unsigned int block_count = static_cast<unsigned int>(
      (layout.configuration_count + determinant_block_size - 1) / determinant_block_size);
  determinantSpatialTracesKernel<<<block_count, determinant_block_size, 0, stream>>>(
      layout, factorization_metadata, combination_metadata, normalized_weights,
      inverses, matrix_gradients, matrix_laplacians, matrix_product_scratch,
      output_log_gradient, output_lap_ratio, output_lap_log, status);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

} // namespace qmcplusplus::psiformer::device
