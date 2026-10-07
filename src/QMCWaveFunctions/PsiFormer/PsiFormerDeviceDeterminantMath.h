//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceDeterminantMath.h
 * @brief Allocation-free host/device real-FP64 determinant factorization core.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_DETERMINANT_MATH_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_DETERMINANT_MATH_H

#include <cmath>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QMC_PF_DET_HOST_DEVICE __host__ __device__
#define QMC_PF_DET_FORCE_INLINE __forceinline__
#else
#define QMC_PF_DET_HOST_DEVICE
#define QMC_PF_DET_FORCE_INLINE inline
#endif

namespace qmcplusplus::psiformer::device_determinant
{

enum class FactorizationStatus : std::uint8_t
{
  REGULAR,
  SINGULAR,
  NONFINITE_INPUT
};

/** Materialize -infinity by IEEE bits without a fast-math infinity expression. */
QMC_PF_DET_HOST_DEVICE QMC_PF_DET_FORCE_INLINE double negativeInfinityBinary64() noexcept
{
  union Binary64
  {
    std::uint64_t integer;
    double floating;
  } encoded{UINT64_C(0xfff0000000000000)};
  return encoded.floating;
}

struct FactorizationMetadata
{
  double phase                       = 0.0;
  double log_abs                     = negativeInfinityBinary64();
  double matrix_scale                = 0.0;
  double minimum_scaled_pivot        = 0.0;
  double maximum_scaled_pivot        = 0.0;
  FactorizationStatus status         = FactorizationStatus::SINGULAR;
  bool inverse_available             = false;
};

static_assert(std::is_standard_layout_v<FactorizationMetadata>);
static_assert(std::is_trivially_copyable_v<FactorizationMetadata>);

static_assert(sizeof(double) == sizeof(std::uint64_t));
static_assert(std::numeric_limits<double>::is_iec559);

/** IEEE-754 finite classification unaffected by finite-math compiler assumptions. */
QMC_PF_DET_HOST_DEVICE QMC_PF_DET_FORCE_INLINE bool isFiniteBinary64(double value) noexcept
{
  union Binary64
  {
    double floating;
    std::uint64_t integer;
  } encoded{value};
  return (encoded.integer & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

QMC_PF_DET_HOST_DEVICE QMC_PF_DET_FORCE_INLINE std::size_t matrixIndex(
    std::size_t configuration, std::size_t determinant, std::size_t determinant_count) noexcept
{
  return configuration * determinant_count + determinant;
}

QMC_PF_DET_HOST_DEVICE QMC_PF_DET_FORCE_INLINE std::size_t matrixOffset(
    std::size_t matrix, std::size_t matrix_size) noexcept
{
  return matrix * matrix_size * matrix_size;
}

QMC_PF_DET_HOST_DEVICE QMC_PF_DET_FORCE_INLINE void clearMatrix(double* matrix,
                                                                 std::size_t elements) noexcept
{
  if (matrix)
    for (std::size_t element = 0; element < elements; ++element)
      matrix[element] = 0.0;
}

/** Construct A^-1 from the scaled LU and final row permutation.
 *
 * `inverse` and `solve` are caller-owned. Back substitution writes columns into
 * inverse while solve holds one forward-substitution right-hand side.
 */
QMC_PF_DET_HOST_DEVICE inline bool buildInverse(const double* lu,
                                                const std::size_t* permutation,
                                                std::size_t matrix_size,
                                                double scale,
                                                double* inverse,
                                                double* solve) noexcept
{
  const std::size_t elements = matrix_size * matrix_size;
  clearMatrix(inverse, elements);
  for (std::size_t rhs = 0; rhs < matrix_size; ++rhs)
  {
    for (std::size_t row = 0; row < matrix_size; ++row)
    {
      double value = permutation[row] == rhs ? 1.0 : 0.0;
      for (std::size_t inner = 0; inner < row; ++inner)
        value -= lu[row * matrix_size + inner] * solve[inner];
      solve[row] = value;
    }
    for (std::size_t reverse = matrix_size; reverse > 0; --reverse)
    {
      const std::size_t row = reverse - 1;
      double value          = solve[row];
      for (std::size_t inner = row + 1; inner < matrix_size; ++inner)
        value -= lu[row * matrix_size + inner] * inverse[inner * matrix_size + rhs];
      inverse[row * matrix_size + rhs] = value / lu[row * matrix_size + row];
    }
  }

  bool finite = true;
  for (std::size_t element = 0; element < elements; ++element)
  {
    inverse[element] /= scale;
    finite = finite && isFiniteBinary64(inverse[element]);
  }
  if (!finite)
    clearMatrix(inverse, elements);
  return finite;
}

/** Scale and factor one row-major matrix using deterministic partial row pivoting. */
QMC_PF_DET_HOST_DEVICE inline FactorizationMetadata factorizeScaledReal(
    const double* matrix,
    std::size_t matrix_size,
    double* lu,
    std::size_t* permutation,
    bool prepare_inverse,
    double* inverse,
    double* solve) noexcept
{
  FactorizationMetadata result;
  const std::size_t elements = matrix_size * matrix_size;
  clearMatrix(inverse, elements);
  for (std::size_t row = 0; row < matrix_size; ++row)
    permutation[row] = row;

  double scale = 0.0;
  for (std::size_t element = 0; element < elements; ++element)
  {
    if (!isFiniteBinary64(matrix[element]))
    {
      clearMatrix(lu, elements);
      result.status = FactorizationStatus::NONFINITE_INPUT;
      return result;
    }
    const double magnitude = ::fabs(matrix[element]);
    scale                  = magnitude > scale ? magnitude : scale;
  }
  result.matrix_scale = scale;
  if (scale == 0.0)
  {
    clearMatrix(lu, elements);
    return result;
  }
  for (std::size_t element = 0; element < elements; ++element)
    lu[element] = matrix[element] / scale;

  double phase     = 1.0;
  double log_abs   = static_cast<double>(matrix_size) * ::log(scale);
  double min_pivot = DBL_MAX;
  double max_pivot = 0.0;
  for (std::size_t column = 0; column < matrix_size; ++column)
  {
    std::size_t pivot_row = column;
    for (std::size_t row = column + 1; row < matrix_size; ++row)
      if (::fabs(lu[row * matrix_size + column]) >
          ::fabs(lu[pivot_row * matrix_size + column]))
        pivot_row = row;

    if (lu[pivot_row * matrix_size + column] == 0.0)
    {
      result.minimum_scaled_pivot = 0.0;
      result.maximum_scaled_pivot = max_pivot;
      return result;
    }
    if (pivot_row != column)
    {
      for (std::size_t entry = 0; entry < matrix_size; ++entry)
      {
        const double temporary                    = lu[column * matrix_size + entry];
        lu[column * matrix_size + entry]          = lu[pivot_row * matrix_size + entry];
        lu[pivot_row * matrix_size + entry]       = temporary;
      }
      const std::size_t temporary = permutation[column];
      permutation[column]         = permutation[pivot_row];
      permutation[pivot_row]      = temporary;
      phase                       = -phase;
    }

    const double pivot     = lu[column * matrix_size + column];
    const double pivot_abs = ::fabs(pivot);
    phase                  = pivot < 0.0 ? -phase : phase;
    log_abs += ::log(pivot_abs);
    min_pivot = pivot_abs < min_pivot ? pivot_abs : min_pivot;
    max_pivot = pivot_abs > max_pivot ? pivot_abs : max_pivot;
    for (std::size_t row = column + 1; row < matrix_size; ++row)
    {
      lu[row * matrix_size + column] /= pivot;
      const double multiplier = lu[row * matrix_size + column];
      for (std::size_t entry = column + 1; entry < matrix_size; ++entry)
        lu[row * matrix_size + entry] -= multiplier * lu[column * matrix_size + entry];
    }
  }

  result.phase                = phase;
  result.log_abs              = log_abs;
  result.minimum_scaled_pivot = min_pivot;
  result.maximum_scaled_pivot = max_pivot;
  result.status               = FactorizationStatus::REGULAR;
  result.inverse_available    = prepare_inverse && inverse && solve &&
      buildInverse(lu, permutation, matrix_size, scale, inverse, solve);
  return result;
}

} // namespace qmcplusplus::psiformer::device_determinant

#undef QMC_PF_DET_HOST_DEVICE
#undef QMC_PF_DET_FORCE_INLINE

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_DETERMINANT_MATH_H
