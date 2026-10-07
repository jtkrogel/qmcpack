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

enum class CombinationStatus : std::uint8_t
{
  REGULAR,
  NODE,
  NONFINITE_INPUT
};

/** Result and diagnostics for one configuration's determinant-channel sum. */
struct CombinationMetadata
{
  double phase                          = 0.0;
  double log_abs                        = negativeInfinityBinary64();
  double reduction_shift                = negativeInfinityBinary64();
  double sum_abs_scaled                 = 0.0;
  double abs_sum_scaled                 = 0.0;
  std::size_t singular_channels         = 0;
  std::size_t underflowed_nonzero_terms = 0;
  CombinationStatus status              = CombinationStatus::NODE;
  bool all_required_inverses_available  = true;
  bool normalized_weights_available     = false;
  bool severe_cancellation              = false;
};

static_assert(std::is_standard_layout_v<CombinationMetadata>);
static_assert(std::is_trivially_copyable_v<CombinationMetadata>);

enum class DerivativeStatus : std::uint8_t
{
  AVAILABLE,
  NODE,
  REQUIRED_INVERSE_UNAVAILABLE,
  NONFINITE_INPUT_OR_RESULT
};

/** Canonical spatial storage is [B,lane,D,N,N] and [B,electron,D,N,N]. */
struct SpatialLayout
{
  std::size_t configuration_count = 0;
  std::size_t determinant_count   = 0;
  std::size_t matrix_size         = 0;
  std::size_t gradient_lanes      = 0;
  std::size_t electron_count      = 0;
};

static_assert(std::is_standard_layout_v<SpatialLayout>);
static_assert(std::is_trivially_copyable_v<SpatialLayout>);

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

/** Deterministic binary64 Neumaier accumulator shared by host and device. */
struct CompensatedSumBinary64
{
  double sum        = 0.0;
  double correction = 0.0;

  QMC_PF_DET_HOST_DEVICE QMC_PF_DET_FORCE_INLINE void add(double value) noexcept
  {
    const double updated = sum + value;
    if (::fabs(sum) >= ::fabs(value))
      correction += (sum - updated) + value;
    else
      correction += (value - updated) + sum;
    sum = updated;
  }

  QMC_PF_DET_HOST_DEVICE QMC_PF_DET_FORCE_INLINE double value() const noexcept
  {
    return sum + correction;
  }
};

/** Combine one configuration's determinant channels in signed-log form.
 *
 * The caller supplies all per-channel output storage. A null coefficient pointer
 * means coefficient one for every channel. At an exact node, normalized weights
 * are cleared and marked unavailable. Device ``long double`` is not portably wider
 * than double, so this baseline deliberately uses deterministic FP64 Neumaier
 * summation. Terms lost by shifted-exponential underflow are counted explicitly.
 */
QMC_PF_DET_HOST_DEVICE inline CombinationMetadata combineChannelsReal(
    const FactorizationMetadata* channel_metadata,
    const double* coefficients,
    std::size_t channel_count,
    double* term_phase,
    double* term_log_abs,
    double* scaled_terms,
    double* normalized_weights) noexcept
{
  CombinationMetadata result;
  for (std::size_t channel = 0; channel < channel_count; ++channel)
  {
    term_phase[channel]         = 0.0;
    term_log_abs[channel]       = negativeInfinityBinary64();
    scaled_terms[channel]       = 0.0;
    normalized_weights[channel] = 0.0;
  }

  bool invalid = false;
  double shift = negativeInfinityBinary64();
  for (std::size_t channel = 0; channel < channel_count; ++channel)
  {
    const FactorizationMetadata& factorization = channel_metadata[channel];
    const double coefficient                   = coefficients ? coefficients[channel] : 1.0;
    if (factorization.status == FactorizationStatus::SINGULAR)
      ++result.singular_channels;
    if (!isFiniteBinary64(coefficient) ||
        factorization.status == FactorizationStatus::NONFINITE_INPUT)
    {
      invalid = true;
      continue;
    }
    if (coefficient != 0.0 && !factorization.inverse_available)
      result.all_required_inverses_available = false;
    if (coefficient == 0.0 || factorization.status != FactorizationStatus::REGULAR ||
        factorization.phase == 0.0)
      continue;
    if ((factorization.phase != 1.0 && factorization.phase != -1.0) ||
        !isFiniteBinary64(factorization.log_abs))
    {
      invalid = true;
      continue;
    }

    term_phase[channel] = coefficient < 0.0 ? -factorization.phase : factorization.phase;
    term_log_abs[channel] = factorization.log_abs + ::log(::fabs(coefficient));
    if (!isFiniteBinary64(term_log_abs[channel]))
    {
      invalid = true;
      continue;
    }
    shift = term_log_abs[channel] > shift ? term_log_abs[channel] : shift;
  }

  if (invalid)
  {
    for (std::size_t channel = 0; channel < channel_count; ++channel)
    {
      term_phase[channel]         = 0.0;
      term_log_abs[channel]       = negativeInfinityBinary64();
      scaled_terms[channel]       = 0.0;
      normalized_weights[channel] = 0.0;
    }
    result.status                          = CombinationStatus::NONFINITE_INPUT;
    result.all_required_inverses_available = false;
    return result;
  }

  result.reduction_shift = shift;
  CompensatedSumBinary64 signed_sum;
  CompensatedSumBinary64 absolute_sum;
  for (std::size_t channel = 0; channel < channel_count; ++channel)
  {
    if (term_phase[channel] == 0.0)
      continue;
    const double magnitude = ::exp(term_log_abs[channel] - shift);
    if (magnitude == 0.0)
      ++result.underflowed_nonzero_terms;
    scaled_terms[channel] = term_phase[channel] * magnitude;
    signed_sum.add(scaled_terms[channel]);
    absolute_sum.add(magnitude);
  }

  const double sum_scaled = signed_sum.value();
  result.sum_abs_scaled   = absolute_sum.value();
  result.abs_sum_scaled   = ::fabs(sum_scaled);
  constexpr double severe_cancellation_ratio = 64.0 * DBL_EPSILON;
  result.severe_cancellation = result.sum_abs_scaled > 0.0 &&
      (result.underflowed_nonzero_terms != 0 ||
       result.abs_sum_scaled <= severe_cancellation_ratio * result.sum_abs_scaled);

  if (sum_scaled == 0.0 || !isFiniteBinary64(shift))
    return result;

  result.phase   = sum_scaled < 0.0 ? -1.0 : 1.0;
  result.log_abs = shift + ::log(result.abs_sum_scaled);
  if (!isFiniteBinary64(result.log_abs))
  {
    result.phase                           = 0.0;
    result.log_abs                         = negativeInfinityBinary64();
    result.status                         = CombinationStatus::NONFINITE_INPUT;
    result.all_required_inverses_available = false;
    return result;
  }

  bool finite_weights = true;
  for (std::size_t channel = 0; channel < channel_count; ++channel)
  {
    normalized_weights[channel] = scaled_terms[channel] / sum_scaled;
    finite_weights = finite_weights && isFiniteBinary64(normalized_weights[channel]);
  }
  if (!finite_weights)
  {
    for (std::size_t channel = 0; channel < channel_count; ++channel)
      normalized_weights[channel] = 0.0;
    result.severe_cancellation = true;
  }
  result.normalized_weights_available = finite_weights;
  result.status                       = CombinationStatus::REGULAR;
  return result;
}

QMC_PF_DET_HOST_DEVICE inline DerivativeStatus derivativeAvailability(
    const FactorizationMetadata* factorization,
    const CombinationMetadata& combination,
    const double* normalized_weights,
    std::size_t channel_count) noexcept
{
  if (combination.status == CombinationStatus::NODE || combination.phase == 0.0)
    return DerivativeStatus::NODE;
  if (combination.status != CombinationStatus::REGULAR ||
      !combination.normalized_weights_available)
    return DerivativeStatus::NONFINITE_INPUT_OR_RESULT;
  if (!combination.all_required_inverses_available)
    return DerivativeStatus::REQUIRED_INVERSE_UNAVAILABLE;
  for (std::size_t channel = 0; channel < channel_count; ++channel)
  {
    const double weight = normalized_weights[channel];
    if (!isFiniteBinary64(weight))
      return DerivativeStatus::NONFINITE_INPUT_OR_RESULT;
    if (factorization[channel].status == FactorizationStatus::NONFINITE_INPUT)
      return DerivativeStatus::NONFINITE_INPUT_OR_RESULT;
    if (weight != 0.0 &&
        (factorization[channel].status != FactorizationStatus::REGULAR ||
         !factorization[channel].inverse_available))
      return DerivativeStatus::REQUIRED_INVERSE_UNAVAILABLE;
  }
  return DerivativeStatus::AVAILABLE;
}

/** Fill channel-major d log|Psi|/d A_k = w_k A_k^-T. */
QMC_PF_DET_HOST_DEVICE inline DerivativeStatus fillMatrixReverseSeeds(
    const FactorizationMetadata* factorization,
    const CombinationMetadata& combination,
    const double* normalized_weights,
    const double* inverses,
    std::size_t channel_count,
    std::size_t matrix_size,
    double* reverse_seeds) noexcept
{
  const std::size_t matrix_elements = matrix_size * matrix_size;
  clearMatrix(reverse_seeds, channel_count * matrix_elements);
  const DerivativeStatus availability = derivativeAvailability(
      factorization, combination, normalized_weights, channel_count);
  if (availability != DerivativeStatus::AVAILABLE)
    return availability;

  for (std::size_t channel = 0; channel < channel_count; ++channel)
  {
    const double weight = normalized_weights[channel];
    if (weight == 0.0)
      continue;
    const double* inverse = inverses + channel * matrix_elements;
    for (std::size_t element = 0; element < matrix_elements; ++element)
      if (!isFiniteBinary64(inverse[element]))
      {
        clearMatrix(reverse_seeds, channel_count * matrix_elements);
        return DerivativeStatus::NONFINITE_INPUT_OR_RESULT;
      }
    for (std::size_t row = 0; row < matrix_size; ++row)
      for (std::size_t column = 0; column < matrix_size; ++column)
      {
        const std::size_t output = channel * matrix_elements + row * matrix_size + column;
        reverse_seeds[output] = weight * inverse[column * matrix_size + row];
        if (!isFiniteBinary64(reverse_seeds[output]))
        {
          clearMatrix(reverse_seeds, channel_count * matrix_elements);
          return DerivativeStatus::NONFINITE_INPUT_OR_RESULT;
        }
      }
  }
  return DerivativeStatus::AVAILABLE;
}

QMC_PF_DET_HOST_DEVICE inline double traceProductBinary64(const double* left,
                                                          const double* right,
                                                          std::size_t matrix_size) noexcept
{
  CompensatedSumBinary64 trace;
  for (std::size_t row = 0; row < matrix_size; ++row)
    for (std::size_t column = 0; column < matrix_size; ++column)
      trace.add(left[row * matrix_size + column] * right[column * matrix_size + row]);
  return trace.value();
}

QMC_PF_DET_HOST_DEVICE inline double inverseProductSquareTraceBinary64(
    const double* inverse,
    const double* derivative,
    std::size_t matrix_size,
    double* matrix_product) noexcept
{
  for (std::size_t row = 0; row < matrix_size; ++row)
    for (std::size_t column = 0; column < matrix_size; ++column)
    {
      CompensatedSumBinary64 entry;
      for (std::size_t inner = 0; inner < matrix_size; ++inner)
        entry.add(inverse[row * matrix_size + inner] *
                  derivative[inner * matrix_size + column]);
      matrix_product[row * matrix_size + column] = entry.value();
    }

  CompensatedSumBinary64 trace;
  for (std::size_t row = 0; row < matrix_size; ++row)
    for (std::size_t column = 0; column < matrix_size; ++column)
      trace.add(matrix_product[row * matrix_size + column] *
                matrix_product[column * matrix_size + row]);
  return trace.value();
}

/** Combine determinant spatial traces for one configuration.
 *
 * Gradient planes are [lane,D,N,N], diagonal-second-derivative planes are
 * [electron,D,N,N], and matrix_product is caller-owned N*N scratch.
 */
QMC_PF_DET_HOST_DEVICE inline DerivativeStatus combineSpatialTraces(
    const FactorizationMetadata* factorization,
    const CombinationMetadata& combination,
    const double* normalized_weights,
    const double* inverses,
    const double* matrix_gradients,
    const double* matrix_laplacians,
    std::size_t channel_count,
    std::size_t matrix_size,
    std::size_t gradient_lanes,
    std::size_t electron_count,
    double* matrix_product,
    double* output_log_gradient,
    double* output_lap_ratio,
    double* output_lap_log) noexcept
{
  const std::size_t matrix_elements = matrix_size * matrix_size;
  const std::size_t channel_elements = channel_count * matrix_elements;
  clearMatrix(output_log_gradient, gradient_lanes);
  clearMatrix(output_lap_ratio, electron_count);
  clearMatrix(output_lap_log, electron_count);
  clearMatrix(matrix_product, matrix_elements);

  const DerivativeStatus availability = derivativeAvailability(
      factorization, combination, normalized_weights, channel_count);
  if (availability != DerivativeStatus::AVAILABLE)
    return availability;

  for (std::size_t channel = 0; channel < channel_count; ++channel)
    if (normalized_weights[channel] != 0.0)
      for (std::size_t element = 0; element < matrix_elements; ++element)
        if (!isFiniteBinary64(inverses[channel * matrix_elements + element]))
          return DerivativeStatus::NONFINITE_INPUT_OR_RESULT;
  for (std::size_t element = 0; element < gradient_lanes * channel_elements; ++element)
    if (!isFiniteBinary64(matrix_gradients[element]))
      return DerivativeStatus::NONFINITE_INPUT_OR_RESULT;
  for (std::size_t element = 0; element < electron_count * channel_elements; ++element)
    if (!isFiniteBinary64(matrix_laplacians[element]))
      return DerivativeStatus::NONFINITE_INPUT_OR_RESULT;

  for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
  {
    CompensatedSumBinary64 combined_trace;
    for (std::size_t channel = 0; channel < channel_count; ++channel)
    {
      const double weight = normalized_weights[channel];
      if (weight == 0.0)
        continue;
      const double* inverse = inverses + channel * matrix_elements;
      const double* gradient = matrix_gradients + lane * channel_elements +
          channel * matrix_elements;
      combined_trace.add(weight * traceProductBinary64(inverse, gradient, matrix_size));
    }
    output_log_gradient[lane] = combined_trace.value();
    if (!isFiniteBinary64(output_log_gradient[lane]))
    {
      clearMatrix(output_log_gradient, gradient_lanes);
      clearMatrix(output_lap_ratio, electron_count);
      clearMatrix(output_lap_log, electron_count);
      return DerivativeStatus::NONFINITE_INPUT_OR_RESULT;
    }
  }

  for (std::size_t electron = 0; electron < electron_count; ++electron)
  {
    CompensatedSumBinary64 combined_laplacian;
    for (std::size_t channel = 0; channel < channel_count; ++channel)
    {
      const double weight = normalized_weights[channel];
      if (weight == 0.0)
        continue;
      const double* inverse = inverses + channel * matrix_elements;
      const double* laplacian = matrix_laplacians + electron * channel_elements +
          channel * matrix_elements;
      CompensatedSumBinary64 channel_bracket;
      channel_bracket.add(traceProductBinary64(inverse, laplacian, matrix_size));
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const std::size_t lane = 3 * electron + dimension;
        const double* gradient = matrix_gradients + lane * channel_elements +
            channel * matrix_elements;
        const double trace = traceProductBinary64(inverse, gradient, matrix_size);
        channel_bracket.add(trace * trace);
        channel_bracket.add(-inverseProductSquareTraceBinary64(
            inverse, gradient, matrix_size, matrix_product));
      }
      combined_laplacian.add(weight * channel_bracket.value());
    }
    output_lap_ratio[electron] = combined_laplacian.value();
    CompensatedSumBinary64 squared_gradient;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double component = output_log_gradient[3 * electron + dimension];
      squared_gradient.add(component * component);
    }
    output_lap_log[electron] = output_lap_ratio[electron] - squared_gradient.value();
    if (!isFiniteBinary64(output_lap_ratio[electron]) ||
        !isFiniteBinary64(output_lap_log[electron]))
    {
      clearMatrix(output_log_gradient, gradient_lanes);
      clearMatrix(output_lap_ratio, electron_count);
      clearMatrix(output_lap_log, electron_count);
      return DerivativeStatus::NONFINITE_INPUT_OR_RESULT;
    }
  }
  return DerivativeStatus::AVAILABLE;
}

} // namespace qmcplusplus::psiformer::device_determinant

#undef QMC_PF_DET_HOST_DEVICE
#undef QMC_PF_DET_FORCE_INLINE

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_DETERMINANT_MATH_H
