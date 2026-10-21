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

/** One block per VALUE matrix, preserving scalar pivot order and per-cell arithmetic.
 * Shared control is POD; global LU/permutation/metadata are the existing owners.
 */
__global__ void determinantValueCooperativeKernel(
    const double* matrices, std::size_t matrix_size, double* all_lu,
    std::size_t* all_permutations, device_determinant::FactorizationMetadata* metadata)
{
  const std::size_t matrix = blockIdx.x;
  const std::size_t elements = matrix_size * matrix_size;
  const double* input = matrices + matrix * elements;
  double* lu = all_lu + matrix * elements;
  std::size_t* permutation = all_permutations + matrix * matrix_size;
  __shared__ double scale, phase, log_abs, min_pivot, max_pivot, pivot;
  __shared__ std::size_t pivot_row;
  __shared__ int state; // 0 regular-in-progress, 1 zero/singular, 2 nonfinite input.
  // Match the serial global scale scan, including first nonfinite-input handling.
  if (threadIdx.x == 0)
  {
    scale = 0.0;
    state = 0;
    for (std::size_t element = 0; element < elements; ++element)
    {
      if (!device_determinant::isFiniteBinary64(input[element]))
      {
        state = 2;
        break;
      }
      const double magnitude = ::fabs(input[element]);
      scale = magnitude > scale ? magnitude : scale;
    }
    if (state == 0 && scale == 0.0)
      state = 1;
    phase = 1.0;
    log_abs = state == 0 ? static_cast<double>(matrix_size) * ::log(scale) : 0.0;
    min_pivot = DBL_MAX;
    max_pivot = 0.0;
  }
  __syncthreads();
  for (std::size_t row = threadIdx.x; row < matrix_size; row += blockDim.x)
    permutation[row] = row;
  for (std::size_t element = threadIdx.x; element < elements; element += blockDim.x)
    lu[element] = state == 0 ? input[element] / scale : 0.0;
  __syncthreads();
  if (state != 0)
  {
    if (threadIdx.x == 0)
    {
      device_determinant::FactorizationMetadata result;
      if (state == 2)
        result.status = device_determinant::FactorizationStatus::NONFINITE_INPUT;
      else
        result.matrix_scale = scale;
      metadata[matrix] = result;
    }
    return; // Uniform block return after initialization and clear.
  }
  for (std::size_t column = 0; column < matrix_size; ++column)
  {
    // One thread preserves strict-greater earliest-row pivot and metadata order.
    if (threadIdx.x == 0)
    {
      pivot_row = column;
      for (std::size_t row = column + 1; row < matrix_size; ++row)
        if (::fabs(lu[row * matrix_size + column]) > ::fabs(lu[pivot_row * matrix_size + column]))
          pivot_row = row;
      if (lu[pivot_row * matrix_size + column] == 0.0)
        state = 1;
    }
    __syncthreads();
    if (state != 0)
      break; // Preserve the same partially factored LU at a zero pivot.
    if (pivot_row != column)
    {
      for (std::size_t entry = threadIdx.x; entry < matrix_size; entry += blockDim.x)
      {
        const double temporary = lu[column * matrix_size + entry];
        lu[column * matrix_size + entry] = lu[pivot_row * matrix_size + entry];
        lu[pivot_row * matrix_size + entry] = temporary;
      }
      if (threadIdx.x == 0)
      {
        const std::size_t temporary = permutation[column];
        permutation[column] = permutation[pivot_row];
        permutation[pivot_row] = temporary;
        phase = -phase;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      pivot = lu[column * matrix_size + column];
      const double pivot_abs = ::fabs(pivot);
      phase = pivot < 0.0 ? -phase : phase;
      log_abs += ::log(pivot_abs);
      min_pivot = pivot_abs < min_pivot ? pivot_abs : min_pivot;
      max_pivot = pivot_abs > max_pivot ? pivot_abs : max_pivot;
    }
    __syncthreads();
    for (std::size_t row = column + 1 + threadIdx.x; row < matrix_size; row += blockDim.x)
      lu[row * matrix_size + column] /= pivot;
    __syncthreads();
    // Each trailing cell has exactly the scalar update, after all multipliers exist.
    const std::size_t remaining = matrix_size - column - 1;
    for (std::size_t cell = threadIdx.x; cell < remaining * remaining; cell += blockDim.x)
    {
      const std::size_t row = column + 1 + cell / remaining;
      const std::size_t entry = column + 1 + cell % remaining;
      lu[row * matrix_size + entry] -= lu[row * matrix_size + column] * lu[column * matrix_size + entry];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    device_determinant::FactorizationMetadata result;
    result.matrix_scale = scale;
    result.maximum_scaled_pivot = max_pivot;
    if (state == 0)
    {
      result.phase = phase;
      result.log_abs = log_abs;
      result.minimum_scaled_pivot = min_pivot;
      result.status = device_determinant::FactorizationStatus::REGULAR;
    }
    metadata[matrix] = result;
  }
}

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

/// Compare before packing: signed zeros differ, and nonfinite inputs never reuse.
__global__ void determinantTargetFactorizationKernel(
    const double* current, std::size_t matrix_count, std::size_t determinant_count,
    std::size_t matrix_size, std::size_t configuration_stride, bool reuse_previous,
    double* packed, double* lu, double* inverse, std::size_t* permutation,
    double* solve, device_determinant::FactorizationMetadata* metadata)
{
  const std::size_t matrix = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (matrix >= matrix_count)
    return;
  const std::size_t elements = matrix_size * matrix_size;
  const std::size_t offset = matrix * elements;
  const double* source = current + (matrix / determinant_count) * configuration_stride +
      (matrix % determinant_count) * elements;
  bool identical = reuse_previous &&
      metadata[matrix].status == device_determinant::FactorizationStatus::REGULAR &&
      metadata[matrix].inverse_available;
  for (std::size_t element = 0; identical && element < elements; ++element)
  {
    // Match the binary64 union convention used by the shared determinant math.
    union Binary64 { double floating; std::uint64_t integer; };
    const Binary64 previous{packed[offset + element]}, next{source[element]};
    identical = device_determinant::isFiniteBinary64(next.floating) && previous.integer == next.integer;
  }
  if (identical)
    return;
  for (std::size_t element = 0; element < elements; ++element)
    packed[offset + element] = source[element];
  metadata[matrix] = device_determinant::factorizeScaledReal(
      packed + offset, matrix_size, lu + offset, permutation + matrix * matrix_size,
      true, inverse + offset, solve + matrix * matrix_size);
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
  const std::size_t gradient_offset  = configuration * layout.gradient_configuration_stride;
  const std::size_t laplacian_offset = configuration * layout.laplacian_configuration_stride;
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

__global__ void determinantFactorJetsKernel(
    LiftedDeterminantLayout layout,
    const device_determinant::FactorizationMetadata* factorization_metadata,
    const device_determinant::CombinationMetadata* combination_metadata,
    const double* normalized_weights,
    const double* inverses,
    const double* orbital_jets,
    const double* determinant_log_gradient,
    const double* determinant_log_laplacian,
    double* factor_jets,
    double* three_matrix_scratch,
    device_determinant::DerivativeStatus* status)
{
  const std::size_t configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration >= layout.matrices.configuration_count)
    return;
  const std::size_t matrix_elements = layout.matrix_size * layout.matrix_size;
  const std::size_t channel_elements = layout.determinant_count * matrix_elements;
  const std::size_t channel_offset = configuration * layout.determinant_count;
  status[configuration] = device_determinant::buildLiftedDeterminantFactors(
      layout.matrices, configuration, factorization_metadata + channel_offset,
      combination_metadata[configuration], normalized_weights + channel_offset,
      inverses + configuration * channel_elements, orbital_jets,
      determinant_log_gradient + configuration * layout.matrices.gradient_lanes,
      determinant_log_laplacian + configuration * layout.matrices.laplacian_lanes,
      layout.determinant_count, layout.matrix_size, factor_jets,
      three_matrix_scratch + configuration * 3 * matrix_elements);
}

__global__ void determinantJetReverseKernel(
    LiftedDeterminantLayout layout,
    const device_determinant::DerivativeStatus* factor_status,
    const double* factor_jets,
    const double* root_jets,
    double* orbital_adjoint_jets,
    device_determinant::DerivativeStatus* status)
{
  const std::size_t configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration >= layout.matrices.configuration_count)
    return;
  status[configuration] = device_determinant::reverseLiftedDeterminant(
      layout.matrices, layout.roots, configuration, factor_status[configuration],
      factor_jets, root_jets, orbital_adjoint_jets);
}

std::size_t checkedProduct(std::size_t left, std::size_t right, const char* description)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(description);
  return left * right;
}

void validateLiftedDeterminantLayout(const LiftedDeterminantLayout& layout)
{
  validateSpatialJetLayout(layout.matrices);
  validateSpatialJetLayout(layout.roots);
  if (layout.matrices.mode != SpatialJetMode::FULL_VGL ||
      layout.roots.mode != SpatialJetMode::FULL_VGL)
    throw std::invalid_argument("PsiFormer lifted determinant requires FULL_VGL layouts");
  if (!haveMatchingSpatialPlanes(layout.matrices, layout.roots))
    throw std::invalid_argument("PsiFormer lifted determinant matrix and root planes do not match");
  if (layout.determinant_count == 0 || layout.matrix_size == 0)
    throw std::invalid_argument("PsiFormer lifted determinant dimensions must be positive");
  const std::size_t matrix_elements = checkedProduct(
      layout.matrix_size, layout.matrix_size,
      "PsiFormer lifted determinant matrix extent overflow");
  const std::size_t expected_elements = checkedProduct(
      layout.determinant_count, matrix_elements,
      "PsiFormer lifted determinant channel extent overflow");
  if (layout.matrices.element_count != expected_elements ||
      layout.roots.element_count != 1)
    throw std::invalid_argument("PsiFormer lifted determinant element counts are inconsistent");
  (void)checkedProduct(layout.matrices.configuration_count,
                       layout.determinant_count,
                       "PsiFormer lifted determinant metadata extent overflow");
  (void)checkedProduct(layout.matrices.configuration_count,
                       expected_elements,
                       "PsiFormer lifted determinant inverse extent overflow");
  (void)checkedProduct(layout.matrices.configuration_count,
                       layout.matrices.gradient_lanes,
                       "PsiFormer lifted determinant gradient extent overflow");
  (void)checkedProduct(layout.matrices.configuration_count,
                       layout.matrices.laplacian_lanes,
                       "PsiFormer lifted determinant Laplacian extent overflow");
  const std::size_t scratch_per_configuration = checkedProduct(
      std::size_t{3}, matrix_elements,
      "PsiFormer lifted determinant scratch extent overflow");
  (void)checkedProduct(layout.matrices.configuration_count,
                       scratch_per_configuration,
                       "PsiFormer lifted determinant batch scratch extent overflow");
  if (layout.matrices.configuration_count >
      std::numeric_limits<unsigned int>::max())
    throw std::length_error(
        "PsiFormer lifted determinant grid exceeds the CUDA/HIP x dimension");
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

/// Validate all owners before selecting cooperative VALUE or the legacy fallback.
Error launchDeterminantValueFactorizationCooperative(
    Stream stream, const double* matrices, std::size_t configuration_count,
    std::size_t determinant_count, std::size_t matrix_size, double* lu,
    std::size_t* permutation, device_determinant::FactorizationMetadata* metadata)
{
  const auto count = checkedProduct(configuration_count, determinant_count,
      "PsiFormer cooperative VALUE matrix-count overflow");
  if (!count)
    return success;
  if (!matrix_size)
    throw std::invalid_argument("PsiFormer cooperative VALUE matrix size must be positive");
  const auto elements = checkedProduct(matrix_size, matrix_size,
      "PsiFormer cooperative VALUE matrix extent overflow");
  (void)checkedProduct(count, elements, "PsiFormer cooperative VALUE batch extent overflow");
  (void)checkedProduct(count, matrix_size, "PsiFormer cooperative VALUE permutation extent overflow");
  if (!matrices || !lu || !permutation || !metadata)
    throw std::invalid_argument("PsiFormer cooperative VALUE requires matrix/LU/permutation/metadata storage");
  // Keep small matrices and the pre-existing in-place contract on the scalar path.
  if (matrix_size < 64 || matrices == lu)
    return launchDeterminantFactorization(stream, matrices, configuration_count,
        determinant_count, matrix_size, false, lu, nullptr, permutation, nullptr, metadata);
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("PsiFormer cooperative VALUE grid exceeds the portable CUDA/HIP x dimension");
  determinantValueCooperativeKernel<<<static_cast<unsigned int>(count), determinant_block_size, 0, stream>>>(
      matrices, matrix_size, lu, permutation, metadata);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

Error launchDeterminantTargetFactorization(
    Stream stream, const double* current_matrices, std::size_t configuration_count,
    std::size_t determinant_count, std::size_t matrix_size, std::size_t configuration_stride,
    bool reuse_previous, double* packed_matrices, double* lu, double* inverse,
    std::size_t* permutation, double* solve,
    device_determinant::FactorizationMetadata* metadata)
{
  const auto count = checkedProduct(configuration_count, determinant_count,
      "PsiFormer target determinant matrix-count overflow");
  if (!count)
    return success;
  if (!matrix_size || !current_matrices || !packed_matrices || !lu || !inverse ||
      !permutation || !solve || !metadata)
    throw std::invalid_argument("PsiFormer target determinants require nonempty matrix/inverse storage");
  const auto elements = checkedProduct(matrix_size, matrix_size,
      "PsiFormer target determinant matrix extent overflow");
  const auto packed_stride = checkedProduct(determinant_count, elements,
      "PsiFormer target determinant configuration extent overflow");
  (void)checkedProduct(count, elements, "PsiFormer target determinant batch extent overflow");
  (void)checkedProduct(count, matrix_size, "PsiFormer target determinant vector extent overflow");
  const auto origin = checkedProduct(configuration_count - 1, configuration_stride,
      "PsiFormer target determinant source extent overflow");
  if (configuration_stride < packed_stride || origin > std::numeric_limits<std::size_t>::max() - packed_stride)
    throw std::invalid_argument("PsiFormer target determinant source stride is invalid");
  if (count > std::numeric_limits<unsigned int>::max())
    throw std::length_error("PsiFormer target determinant grid exceeds the CUDA/HIP x dimension");
  const auto blocks = static_cast<unsigned int>((count + determinant_block_size - 1) / determinant_block_size);
  determinantTargetFactorizationKernel<<<blocks, determinant_block_size, 0, stream>>>(
      current_matrices, count, determinant_count, matrix_size, configuration_stride, reuse_previous,
      packed_matrices, lu, inverse, permutation, solve, metadata);
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
  (void)metadata_elements;
  (void)checkedProduct(layout.configuration_count, channel_elements,
                       "PsiFormer determinant spatial inverse extent overflow");
  layout = device_determinant::resolveSpatialConfigurationStrides(layout);
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

Error launchDeterminantFactorJets(
    Stream stream,
    const LiftedDeterminantLayout& layout,
    const device_determinant::FactorizationMetadata* factorization_metadata,
    const device_determinant::CombinationMetadata* combination_metadata,
    const double* normalized_weights,
    const double* inverses,
    const double* orbital_jets,
    const double* determinant_log_gradient,
    const double* determinant_log_laplacian,
    double* factor_jets,
    double* three_matrix_scratch,
    device_determinant::DerivativeStatus* status)
{
  if (layout.matrices.configuration_count == 0)
    return success;
  validateLiftedDeterminantLayout(layout);
  if (!factorization_metadata || !combination_metadata || !normalized_weights ||
      !inverses || !orbital_jets || !determinant_log_gradient ||
      !determinant_log_laplacian || !factor_jets || !three_matrix_scratch ||
      !status)
    throw std::invalid_argument("PsiFormer lifted determinant factor storage is null");

  const unsigned int block_count = static_cast<unsigned int>(
      (layout.matrices.configuration_count + determinant_block_size - 1) /
      determinant_block_size);
  determinantFactorJetsKernel<<<block_count, determinant_block_size, 0, stream>>>(
      layout, factorization_metadata, combination_metadata, normalized_weights,
      inverses, orbital_jets, determinant_log_gradient,
      determinant_log_laplacian, factor_jets, three_matrix_scratch, status);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

Error launchDeterminantJetReverse(
    Stream stream,
    const LiftedDeterminantLayout& layout,
    const device_determinant::DerivativeStatus* factor_status,
    const double* factor_jets,
    const double* root_jets,
    double* orbital_adjoint_jets,
    device_determinant::DerivativeStatus* status)
{
  if (layout.matrices.configuration_count == 0)
    return success;
  validateLiftedDeterminantLayout(layout);
  if (!factor_status || !factor_jets || !root_jets ||
      !orbital_adjoint_jets || !status)
    throw std::invalid_argument("PsiFormer lifted determinant reverse storage is null");

  const unsigned int block_count = static_cast<unsigned int>(
      (layout.matrices.configuration_count + determinant_block_size - 1) /
      determinant_block_size);
  determinantJetReverseKernel<<<block_count, determinant_block_size, 0, stream>>>(
      layout, factor_status, factor_jets, root_jets, orbital_adjoint_jets,
      status);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

} // namespace qmcplusplus::psiformer::device
