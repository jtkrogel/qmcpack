//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeterminant.h
 * @brief Stable analytic determinant family for real, open-boundary PsiFormer.
 *
 * This header deliberately separates amplitude representation from the real kernel.
 * LogAmplitude<Phase> is the API seam for a later unit-complex phase, while this
 * implementation instantiates only double phase (+1/-1, or 0 at an exact node).
 * Geometry and boundary handling stay outside this algebra kernel.
 *
 * A workspace owns scaled pivoted-LU factors, permutations, inverses, channel
 * sign/log determinants, and reduction scratch.  evaluate(), evaluateSpatial(), and
 * fillLogAmplitudeMatrixAdjoints() reuse that state without allocating.  Channel
 * determinants are never materialized in ordinary floating-point form: the channel
 * sum is a compensated signed-log reduction, so a finite log amplitude survives raw
 * determinant overflow and underflow.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DETERMINANT_H
#define QMCPLUSPLUS_PSIFORMER_DETERMINANT_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace qmcplusplus::psiformer::determinant
{

/** Phase/log-magnitude amplitude representation.
 *
 * Phase is double for the implemented real kernel.  A future complex specialization
 * can instantiate this representation with a normalized complex phase without
 * changing value, ratio, or result ownership contracts.
 */
template<class Phase>
struct LogAmplitude
{
  Phase phase{};
  double log_abs = -std::numeric_limits<double>::infinity();

  /// Exact zeros use phase==0 and log_abs==-infinity.
  bool isZero() const noexcept { return phase == Phase{}; }
};

using RealLogAmplitude = LogAmplitude<double>;

/** Classify a binary64 value without relying on finite-math-sensitive predicates.
 *
 * QMCPACK release builds use ``-ffast-math``.  Inspecting the IEEE exponent bits
 * keeps node and overflow handling meaningful even when the compiler is allowed
 * to assume that ordinary floating-point expressions are finite.
 */
inline bool isFiniteReal(double value) noexcept
{
  static_assert(sizeof(double) == sizeof(std::uint64_t),
                "PsiFormer stable determinants require binary64 double storage");
  static_assert(std::numeric_limits<double>::is_iec559,
                "PsiFormer stable determinants require IEEE-754 double arithmetic");
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

/// Materialize a real amplitude only when a legacy caller explicitly needs it.
inline double realValue(const RealLogAmplitude& amplitude, double additive_log = 0.0) noexcept
{
  if (amplitude.isZero())
    return 0.0;
  const double magnitude = std::exp(amplitude.log_abs + additive_log);
  return std::copysign(magnitude, amplitude.phase);
}

/// Add a finite scalar exponent (for example, the analytic cusp) in log space.
inline RealLogAmplitude shiftedLogMagnitude(RealLogAmplitude amplitude, double additive_log)
{
  if (!isFiniteReal(additive_log))
    throw std::invalid_argument("PsiFormer determinant received a non-finite log-magnitude shift");
  if (!amplitude.isZero())
    amplitude.log_abs += additive_log;
  return amplitude;
}

/// State of one channel factorization.
enum class ChannelFactorizationStatus : unsigned char
{
  REGULAR,
  SINGULAR
};

/** Reusable metadata for one scaled, pivoted LU factorization. */
struct ChannelFactorization
{
  RealLogAmplitude determinant;
  ChannelFactorizationStatus status = ChannelFactorizationStatus::SINGULAR;
  double matrix_scale               = 0.0;
  double minimum_scaled_pivot       = 0.0;
  double maximum_scaled_pivot       = 0.0;
  bool inverse_available            = false;
};

/** Result of one channel reduction. */
struct RealDeterminantResult
{
  RealLogAmplitude amplitude;
  std::size_t singular_channels = 0;
  bool all_required_inverses_available = false;
};

namespace detail
{

/// Neumaier-compensated sum in the widest inexpensive built-in real type.
struct CompensatedSum
{
  long double sum        = 0.0L;
  long double correction = 0.0L;

  /// Reset both the running sum and its lost-low-bits correction.
  void clear() noexcept
  {
    sum        = 0.0L;
    correction = 0.0L;
  }

  /// Add one term while retaining the first-order floating-point residual.
  void add(long double value) noexcept
  {
    const long double updated = sum + value;
    if (std::abs(sum) >= std::abs(value))
      correction += (sum - updated) + value;
    else
      correction += (value - updated) + sum;
    sum = updated;
  }

  /// Return the corrected accumulated value.
  long double value() const noexcept { return sum + correction; }
};

/** Dynamically shifted signed-log accumulator.
 *
 * The shift is treated as locally constant by all derivative consumers, matching the
 * reference max-shift convention.  Rescaling affects only numerical representation.
 */
class SignedLogAccumulator
{
public:
  /// Reset the dynamic exponent and compensated mantissa sum.
  void clear() noexcept
  {
    shift_ = -std::numeric_limits<double>::infinity();
    scaled_.clear();
  }

  /// Add one nonzero signed-log term, shifting prior terms when necessary.
  void add(double phase, double log_abs)
  {
    if (phase == 0.0)
      return;
    if ((phase != 1.0 && phase != -1.0) || !isFiniteReal(log_abs))
      throw std::invalid_argument("invalid signed-log determinant term");

    if (!isFiniteReal(shift_))
    {
      shift_ = log_abs;
      scaled_.add(static_cast<long double>(phase));
      return;
    }
    if (log_abs > shift_)
    {
      const long double factor =
          std::exp(static_cast<long double>(shift_) - static_cast<long double>(log_abs));
      scaled_.sum *= factor;
      scaled_.correction *= factor;
      shift_ = log_abs;
    }
    scaled_.add(static_cast<long double>(phase) *
                std::exp(static_cast<long double>(log_abs) -
                         static_cast<long double>(shift_)));
  }

  /// Convert the normalized accumulator back to phase and log magnitude.
  RealLogAmplitude amplitude() const noexcept
  {
    const long double scaled_value = scaled_.value();
    if (scaled_value == 0.0L || !isFiniteReal(shift_))
      return {};
    return {scaled_value > 0.0L ? 1.0 : -1.0,
            shift_ + static_cast<double>(std::log(std::abs(scaled_value)))};
  }

  /// Return the common logarithmic shift used by the scaled terms.
  double shift() const noexcept { return shift_; }

  /// Return the signed, compensated sum after removal of the common shift.
  long double scaledValue() const noexcept { return scaled_.value(); }

private:
  double shift_ = -std::numeric_limits<double>::infinity();
  CompensatedSum scaled_;
};

/// Narrow an extended-precision derivative only after checking its range.
inline double narrowFinite(long double value, const char* quantity)
{
  const long double limit = static_cast<long double>(std::numeric_limits<double>::max());
  if (!(value >= -limit && value <= limit))
    throw std::overflow_error(quantity);
  return static_cast<double>(value);
}

template<class Vector>
/// Mix a vector's allocation identity and capacity into a storage fingerprint.
inline void mixStorage(std::size_t& hash, const Vector& buffer) noexcept
{
  hash ^= reinterpret_cast<std::uintptr_t>(buffer.data());
  hash *= 1099511628211ULL;
  hash ^= buffer.capacity();
  hash *= 1099511628211ULL;
}

} // namespace detail

/** Allocation-free real/open determinant evaluator after workspace construction.
 *
 * Matrices are a contiguous channel-major array of row-major n-by-n matrices.
 * Optional coefficients have one entry per channel; omitting them means all ones.
 * Spatial derivative arrays are lane-major over the complete matrix batch, matching
 * DirectSpatialJetBuffer.  Laplacian lane e is the trace of the matrix second
 * derivatives for electron e, and therefore requires gradient lanes 3e..3e+2.
 */
class RealOpenDeterminantWorkspace
{
public:
  /// Allocate all factorization, reduction, and requested derivative storage once.
  RealOpenDeterminantWorkspace(std::size_t channels,
                               std::size_t matrix_size,
                               std::size_t gradient_lanes = 0,
                               std::size_t laplacian_lanes = 0)
      : channels_(channels),
        matrix_size_(matrix_size),
        matrix_elements_(checkedSquare(matrix_size)),
        gradient_capacity_(gradient_lanes),
        laplacian_capacity_(laplacian_lanes),
        lu_(checkedProduct(channels, matrix_elements_)),
        inverses_(checkedProduct(channels, matrix_elements_)),
        permutations_(checkedProduct(channels, matrix_size)),
        factorization_(channels),
        coefficient_(channels, 1.0),
        term_phase_(channels, 0.0),
        term_log_abs_(channels, -std::numeric_limits<double>::infinity()),
        scaled_terms_(channels, 0.0L),
        solve_(matrix_size),
        matrix_product_(matrix_elements_),
        gradient_sums_(gradient_lanes),
        laplacian_sums_(laplacian_lanes),
        staged_gradient_(gradient_lanes),
        staged_lap_log_(laplacian_lanes),
        staged_lap_ratio_(laplacian_lanes),
        staged_matrix_adjoint_(checkedProduct(channels, matrix_elements_))
  {
    if (channels_ == 0 || matrix_size_ == 0)
      throw std::invalid_argument("PsiFormer determinant workspace dimensions must be nonzero");
    if (laplacian_lanes != 0 && gradient_lanes != 3 * laplacian_lanes)
      throw std::invalid_argument("PsiFormer determinant Laplacians require three gradient lanes per electron");
  }

  /// Factorization scratch is movable but must never be shared by copying.
  RealOpenDeterminantWorkspace(const RealOpenDeterminantWorkspace&) = delete;

  /// Copy assignment is disabled for the same exclusive-ownership reason.
  RealOpenDeterminantWorkspace& operator=(const RealOpenDeterminantWorkspace&) = delete;

  /// Move construction transfers exclusive ownership of all factorization storage.
  RealOpenDeterminantWorkspace(RealOpenDeterminantWorkspace&&) = default;

  /// Move assignment likewise transfers the complete mutable workspace.
  RealOpenDeterminantWorkspace& operator=(RealOpenDeterminantWorkspace&&) = default;

  /// Return the number of determinant channels represented by this workspace.
  std::size_t channels() const noexcept { return channels_; }

  /// Return the common row and column extent of every channel matrix.
  std::size_t matrixSize() const noexcept { return matrix_size_; }

  /// Return the number of row-major elements in one channel matrix.
  std::size_t matrixElements() const noexcept { return matrix_elements_; }

  /** Factor every channel, prepare reusable inverses, and reduce the channel sum. */
  RealDeterminantResult evaluate(const double* matrices, const double* coefficients = nullptr)
  {
    return evaluatePrepared(matrices, coefficients, true);
  }

  /** Value-specialized reduction that omits inverse solves. */
  RealDeterminantResult evaluateValue(const double* matrices,
                                      const double* coefficients = nullptr)
  {
    return evaluatePrepared(matrices, coefficients, false);
  }

private:
  /// Factor channels and form their signed-log reduction, optionally caching inverses.
  RealDeterminantResult evaluatePrepared(const double* matrices,
                                         const double* coefficients,
                                         bool prepare_inverses)
  {
    // A failed operation must not leave the preceding result observable.  The
    // factorization buffers are scratch; only evaluated_ publishes their result.
    evaluated_ = false;
    if (!matrices)
      throw std::invalid_argument("PsiFormer determinant matrices pointer is null");

    double reduction_shift = -std::numeric_limits<double>::infinity();
    std::size_t singular_channels = 0;
    bool all_required_inverses    = true;
    for (std::size_t channel = 0; channel < channels_; ++channel)
    {
      const double coefficient = coefficients ? coefficients[channel] : 1.0;
      if (!isFiniteReal(coefficient))
        throw std::invalid_argument("PsiFormer determinant coefficient is non-finite");
      coefficient_[channel] = coefficient;
      factorizeChannel(matrices + channel * matrix_elements_, channel, prepare_inverses);
      ChannelFactorization& record = factorization_[channel];
      singular_channels += record.status == ChannelFactorizationStatus::SINGULAR ? 1 : 0;

      term_phase_[channel]   = 0.0;
      term_log_abs_[channel] = -std::numeric_limits<double>::infinity();
      if (coefficient != 0.0 && !record.determinant.isZero())
      {
        term_phase_[channel] = std::signbit(coefficient) ? -record.determinant.phase
                                                         : record.determinant.phase;
        term_log_abs_[channel] = record.determinant.log_abs + std::log(std::abs(coefficient));
        if (!isFiniteReal(term_log_abs_[channel]))
          throw std::overflow_error("PsiFormer determinant channel log magnitude is non-finite");
        reduction_shift = std::max(reduction_shift, term_log_abs_[channel]);
      }
      if (coefficient != 0.0 && !record.inverse_available)
        all_required_inverses = false;
    }

    detail::CompensatedSum scaled_sum;
    last_shift_ = reduction_shift;
    for (std::size_t channel = 0; channel < channels_; ++channel)
    {
      scaled_terms_[channel] = term_phase_[channel] == 0.0
          ? 0.0L
          : static_cast<long double>(term_phase_[channel]) *
              std::exp(static_cast<long double>(term_log_abs_[channel]) -
                       static_cast<long double>(last_shift_));
      scaled_sum.add(scaled_terms_[channel]);
    }
    last_scaled_sum_ = scaled_sum.value();

    RealLogAmplitude amplitude;
    if (last_scaled_sum_ != 0.0L && isFiniteReal(last_shift_))
      amplitude = {last_scaled_sum_ > 0.0L ? 1.0 : -1.0,
                   last_shift_ + static_cast<double>(std::log(std::abs(last_scaled_sum_)))};

    last_result_ = {amplitude, singular_channels, all_required_inverses};
    evaluated_   = true;
    return last_result_;
  }

public:

  /** Evaluate determinant-only log gradients and Laplacian primitives.
   *
   * output_lap_ratio receives (nabla_e^2 Psi_det)/Psi_det.  output_lap_log
   * receives the corresponding logarithmic Laplacian.  A caller can add cusp
   * derivatives and form its final ratio without any electron Hessian.
   */
  RealDeterminantResult evaluateSpatial(const double* matrices,
                                        const double* matrix_gradients,
                                        std::size_t gradient_lanes,
                                        const double* matrix_laplacians,
                                        std::size_t laplacian_lanes,
                                        double* output_log_gradient,
                                        double* output_lap_log,
                                        double* output_lap_ratio,
                                        const double* coefficients = nullptr)
  {
    // This compound operation publishes determinant state only if both the
    // factorization and every requested derivative lane succeed.
    evaluated_ = false;
    validateSpatialArguments(matrix_gradients, gradient_lanes, matrix_laplacians,
                             laplacian_lanes, output_log_gradient, output_lap_log,
                             output_lap_ratio);
    validateSpatialInputs(matrix_gradients, gradient_lanes, matrix_laplacians,
                          laplacian_lanes);
    const RealDeterminantResult result = evaluate(matrices, coefficients);
    try
    {
      requireDerivativeState();
    }
    catch (...)
    {
      evaluated_ = false;
      throw;
    }
    evaluated_ = false;

    for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
      gradient_sums_[lane].clear();
    for (std::size_t electron = 0; electron < laplacian_lanes; ++electron)
      laplacian_sums_[electron].clear();

    const std::size_t batch_elements = channels_ * matrix_elements_;
    for (std::size_t channel = 0; channel < channels_; ++channel)
    {
      const long double channel_weight = scaled_terms_[channel] / last_scaled_sum_;
      if (channel_weight == 0.0L)
        continue;
      const double* inverse = inverses_.data() + channel * matrix_elements_;

      for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
      {
        const double* matrix_gradient = matrix_gradients + lane * batch_elements +
            channel * matrix_elements_;
        const long double trace = traceProduct(inverse, matrix_gradient);
        gradient_sums_[lane].add(channel_weight * trace);
      }

      for (std::size_t electron = 0; electron < laplacian_lanes; ++electron)
      {
        const double* matrix_laplacian = matrix_laplacians + electron * batch_elements +
            channel * matrix_elements_;
        long double bracket = traceProduct(inverse, matrix_laplacian);
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
        {
          const std::size_t lane = 3 * electron + dimension;
          const double* matrix_gradient = matrix_gradients + lane * batch_elements +
              channel * matrix_elements_;
          const long double trace = traceProduct(inverse, matrix_gradient);
          const long double square_trace = inverseProductSquareTrace(inverse, matrix_gradient);
          bracket += trace * trace - square_trace;
        }
        laplacian_sums_[electron].add(channel_weight * bracket);
      }
    }

    // Narrow the complete result into workspace-owned storage before exposing
    // any lane to the caller.  A late overflow therefore leaves destinations
    // byte-for-byte unchanged.
    for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
      staged_gradient_[lane] = detail::narrowFinite(
          gradient_sums_[lane].value(), "PsiFormer determinant gradient overflowed");

    for (std::size_t electron = 0; electron < laplacian_lanes; ++electron)
    {
      const long double lap_ratio = laplacian_sums_[electron].value();
      long double squared_gradient = 0.0L;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const long double component = gradient_sums_[3 * electron + dimension].value();
        squared_gradient += component * component;
      }
      staged_lap_ratio_[electron] = detail::narrowFinite(
          lap_ratio, "PsiFormer determinant Laplacian ratio overflowed");
      staged_lap_log_[electron] = detail::narrowFinite(
          lap_ratio - squared_gradient,
          "PsiFormer determinant logarithmic Laplacian overflowed");
    }

    std::copy_n(staged_gradient_.data(), gradient_lanes, output_log_gradient);
    std::copy_n(staged_lap_log_.data(), laplacian_lanes, output_lap_log);
    std::copy_n(staged_lap_ratio_.data(), laplacian_lanes, output_lap_ratio);
    evaluated_ = true;
    return result;
  }

  /** Fill d log|sum_k c_k det(A_k)| / d A_k in channel-major row-major order. */
  void fillLogAmplitudeMatrixAdjoints(double* matrix_adjoint) const
  {
    if (!matrix_adjoint)
      throw std::invalid_argument("PsiFormer determinant adjoint pointer is null");
    requireDerivativeState();
    for (std::size_t channel = 0; channel < channels_; ++channel)
    {
      const long double channel_weight = scaled_terms_[channel] / last_scaled_sum_;
      const double* inverse = inverses_.data() + channel * matrix_elements_;
      for (std::size_t row = 0; row < matrix_size_; ++row)
        for (std::size_t column = 0; column < matrix_size_; ++column)
          staged_matrix_adjoint_[channel * matrix_elements_ +
                                 row * matrix_size_ + column] = detail::narrowFinite(
              channel_weight * static_cast<long double>(inverse[column * matrix_size_ + row]),
              "PsiFormer determinant reverse seed overflowed");
    }
    std::copy(staged_matrix_adjoint_.begin(), staged_matrix_adjoint_.end(),
              matrix_adjoint);
  }

  /// Return metadata from the most recent successful evaluation.
  const RealDeterminantResult& result() const
  {
    if (!evaluated_)
      throw std::logic_error("PsiFormer determinant result requested before evaluation");
    return last_result_;
  }

  /// Return pivot and singularity diagnostics for one channel.
  const ChannelFactorization& factorization(std::size_t channel) const
  {
    if (channel >= channels_)
      throw std::out_of_range("PsiFormer determinant channel is out of range");
    return factorization_[channel];
  }

  /// Return the scaled in-place LU factors for one channel.
  const double* lu(std::size_t channel) const
  {
    checkChannel(channel);
    return lu_.data() + channel * matrix_elements_;
  }

  /// Return a cached inverse, or null when it was unavailable or not requested.
  const double* inverse(std::size_t channel) const
  {
    checkChannel(channel);
    return factorization_[channel].inverse_available
        ? inverses_.data() + channel * matrix_elements_
        : nullptr;
  }

  /// Return the row permutation generated by partial pivoting.
  const std::size_t* permutation(std::size_t channel) const
  {
    checkChannel(channel);
    return permutations_.data() + channel * matrix_size_;
  }

  /// Signed normalized channel contribution c_k det(A_k) / sum_j c_j det(A_j).
  long double channelWeight(std::size_t channel) const
  {
    checkChannel(channel);
    if (!evaluated_ || last_scaled_sum_ == 0.0L)
      throw std::domain_error("PsiFormer determinant channel weights are undefined at a node");
    return scaled_terms_[channel] / last_scaled_sum_;
  }

  /// Hash every backing allocation for no-growth regression tests.
  std::size_t storageFingerprint() const noexcept
  {
    std::size_t hash = 1469598103934665603ULL;
    detail::mixStorage(hash, lu_);
    detail::mixStorage(hash, inverses_);
    detail::mixStorage(hash, permutations_);
    detail::mixStorage(hash, factorization_);
    detail::mixStorage(hash, coefficient_);
    detail::mixStorage(hash, term_phase_);
    detail::mixStorage(hash, term_log_abs_);
    detail::mixStorage(hash, scaled_terms_);
    detail::mixStorage(hash, solve_);
    detail::mixStorage(hash, matrix_product_);
    detail::mixStorage(hash, gradient_sums_);
    detail::mixStorage(hash, laplacian_sums_);
    detail::mixStorage(hash, staged_gradient_);
    detail::mixStorage(hash, staged_lap_log_);
    detail::mixStorage(hash, staged_lap_ratio_);
    detail::mixStorage(hash, staged_matrix_adjoint_);
    return hash;
  }

  /// Bytes reserved by explicit determinant workspace vectors.
  std::size_t storageBytes() const
  {
    std::size_t bytes = 0;
    auto add = [&bytes](const auto& buffer) {
      using Element = typename std::decay_t<decltype(buffer)>::value_type;
      bytes = checkedSum(
          bytes,
          checkedProduct(buffer.capacity(), sizeof(Element)));
    };
    add(lu_);
    add(inverses_);
    add(permutations_);
    add(factorization_);
    add(coefficient_);
    add(term_phase_);
    add(term_log_abs_);
    add(scaled_terms_);
    add(solve_);
    add(matrix_product_);
    add(gradient_sums_);
    add(laplacian_sums_);
    add(staged_gradient_);
    add(staged_lap_log_);
    add(staged_lap_ratio_);
    add(staged_matrix_adjoint_);
    return bytes;
  }

  /// Report overlap with any retained determinant-workspace allocation.
  bool overlapsStorage(const void* data, std::size_t bytes) const noexcept
  {
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data);
    if (bytes == 0)
      return false;
    if (data == nullptr || begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
      return true;
    const std::uintptr_t end = begin + bytes;
    const auto overlaps = [begin, end](const auto& values) noexcept {
      using Element = typename std::decay_t<decltype(values)>::value_type;
      if (values.capacity() == 0)
        return false;
      if (values.capacity() > std::numeric_limits<std::size_t>::max() / sizeof(Element))
        return true;
      const std::size_t storage_bytes = values.capacity() * sizeof(Element);
      const std::uintptr_t storage_begin =
          reinterpret_cast<std::uintptr_t>(values.data());
      if (values.data() == nullptr ||
          storage_begin > std::numeric_limits<std::uintptr_t>::max() - storage_bytes)
        return true;
      const std::uintptr_t storage_end = storage_begin + storage_bytes;
      return begin < storage_end && storage_begin < end;
    };
    return overlaps(lu_) || overlaps(inverses_) || overlaps(permutations_) ||
        overlaps(factorization_) || overlaps(coefficient_) ||
        overlaps(term_phase_) || overlaps(term_log_abs_) ||
        overlaps(scaled_terms_) || overlaps(solve_) ||
        overlaps(matrix_product_) || overlaps(gradient_sums_) ||
        overlaps(laplacian_sums_) || overlaps(staged_gradient_) ||
        overlaps(staged_lap_log_) || overlaps(staged_lap_ratio_) ||
        overlaps(staged_matrix_adjoint_);
  }

private:
  /// Add storage contributions while rejecting size_t wraparound.
  static std::size_t checkedSum(std::size_t left, std::size_t right)
  {
    if (right > std::numeric_limits<std::size_t>::max() - left)
      throw std::overflow_error("PsiFormer determinant storage byte extent overflow");
    return left + right;
  }

  /// Square an extent with an explicit size_t overflow check.
  static std::size_t checkedSquare(std::size_t value)
  {
    if (value != 0 && value > std::numeric_limits<std::size_t>::max() / value)
      throw std::overflow_error("PsiFormer determinant matrix extent overflow");
    return value * value;
  }

  /// Multiply two extents with an explicit size_t overflow check.
  static std::size_t checkedProduct(std::size_t left, std::size_t right)
  {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
      throw std::overflow_error("PsiFormer determinant batch extent overflow");
    return left * right;
  }

  /// Validate a channel index before exposing workspace-owned storage.
  void checkChannel(std::size_t channel) const
  {
    if (channel >= channels_)
      throw std::out_of_range("PsiFormer determinant channel is out of range");
  }

  /// Check derivative pointers, lane counts, and workspace capacities as one contract.
  void validateSpatialArguments(const double* gradients,
                                std::size_t gradient_lanes,
                                const double* laplacians,
                                std::size_t laplacian_lanes,
                                double* output_gradient,
                                double* output_lap_log,
                                double* output_lap_ratio) const
  {
    if (gradient_lanes > gradient_capacity_ || laplacian_lanes > laplacian_capacity_)
      throw std::invalid_argument("PsiFormer determinant derivative lanes exceed workspace capacity");
    if (gradient_lanes != 0 && (!gradients || !output_gradient))
      throw std::invalid_argument("PsiFormer determinant gradient storage is null");
    if (laplacian_lanes != 0 &&
        (!laplacians || !output_lap_log || !output_lap_ratio ||
         gradient_lanes != 3 * laplacian_lanes))
      throw std::invalid_argument("PsiFormer determinant Laplacian storage is inconsistent");
  }

  /// Preflight every derivative input so evaluation never consumes NaN or infinity.
  void validateSpatialInputs(const double* gradients,
                             std::size_t gradient_lanes,
                             const double* laplacians,
                             std::size_t laplacian_lanes) const
  {
    const std::size_t batch_elements = channels_ * matrix_elements_;
    const std::size_t gradient_elements =
        checkedProduct(gradient_lanes, batch_elements);
    const std::size_t laplacian_elements =
        checkedProduct(laplacian_lanes, batch_elements);
    for (std::size_t element = 0; element < gradient_elements;
         ++element)
      if (!isFiniteReal(gradients[element]))
        throw std::invalid_argument(
            "PsiFormer determinant gradient contains a non-finite element");
    for (std::size_t element = 0; element < laplacian_elements;
         ++element)
      if (!isFiniteReal(laplacians[element]))
        throw std::invalid_argument(
            "PsiFormer determinant Laplacian contains a non-finite element");
  }

  /// Reject node or singular states where logarithmic derivatives are undefined.
  void requireDerivativeState() const
  {
    if (!evaluated_ || last_result_.amplitude.isZero())
      throw std::domain_error("PsiFormer determinant log derivatives are undefined at an exact node");
    if (!last_result_.all_required_inverses_available)
      throw std::domain_error(
          "PsiFormer determinant derivatives require nonsingular finite-inverse channels");
  }

  /// Build a scaled partial-pivot LU factorization and optional inverse for one channel.
  void factorizeChannel(const double* matrix,
                        std::size_t channel,
                        bool prepare_inverse)
  {
    double scale = 0.0;
    for (std::size_t element = 0; element < matrix_elements_; ++element)
    {
      if (!isFiniteReal(matrix[element]))
        throw std::invalid_argument("PsiFormer determinant matrix contains a non-finite element");
      scale = std::max(scale, std::abs(matrix[element]));
    }

    double* factor            = lu_.data() + channel * matrix_elements_;
    double* inverse           = inverses_.data() + channel * matrix_elements_;
    std::size_t* permutation  = permutations_.data() + channel * matrix_size_;
    ChannelFactorization& out = factorization_[channel];
    out                       = {};
    out.matrix_scale          = scale;
    std::fill(inverse, inverse + matrix_elements_, 0.0);
    for (std::size_t row = 0; row < matrix_size_; ++row)
      permutation[row] = row;

    if (scale == 0.0)
    {
      std::fill(factor, factor + matrix_elements_, 0.0);
      return;
    }
    for (std::size_t element = 0; element < matrix_elements_; ++element)
      factor[element] = matrix[element] / scale;

    double phase      = 1.0;
    double log_abs    = static_cast<double>(matrix_size_) * std::log(scale);
    double min_pivot  = std::numeric_limits<double>::infinity();
    double max_pivot  = 0.0;
    for (std::size_t column = 0; column < matrix_size_; ++column)
    {
      std::size_t pivot_row = column;
      for (std::size_t row = column + 1; row < matrix_size_; ++row)
        if (std::abs(factor[row * matrix_size_ + column]) >
            std::abs(factor[pivot_row * matrix_size_ + column]))
          pivot_row = row;

      const double pivot_candidate = factor[pivot_row * matrix_size_ + column];
      if (pivot_candidate == 0.0)
      {
        out.minimum_scaled_pivot = 0.0;
        out.maximum_scaled_pivot = max_pivot;
        return;
      }
      if (pivot_row != column)
      {
        for (std::size_t entry = 0; entry < matrix_size_; ++entry)
          std::swap(factor[column * matrix_size_ + entry],
                    factor[pivot_row * matrix_size_ + entry]);
        std::swap(permutation[column], permutation[pivot_row]);
        phase = -phase;
      }

      const double pivot = factor[column * matrix_size_ + column];
      const double pivot_abs = std::abs(pivot);
      phase = std::signbit(pivot) ? -phase : phase;
      log_abs += std::log(pivot_abs);
      min_pivot = std::min(min_pivot, pivot_abs);
      max_pivot = std::max(max_pivot, pivot_abs);
      for (std::size_t row = column + 1; row < matrix_size_; ++row)
      {
        factor[row * matrix_size_ + column] /= pivot;
        const double multiplier = factor[row * matrix_size_ + column];
        for (std::size_t entry = column + 1; entry < matrix_size_; ++entry)
          factor[row * matrix_size_ + entry] -=
              multiplier * factor[column * matrix_size_ + entry];
      }
    }

    out.determinant          = {phase, log_abs};
    out.status               = ChannelFactorizationStatus::REGULAR;
    out.minimum_scaled_pivot = min_pivot;
    out.maximum_scaled_pivot = max_pivot;
    out.inverse_available    = prepare_inverse && buildInverse(channel);
  }

  /// Solve against every basis vector to construct one inverse from cached LU factors.
  bool buildInverse(std::size_t channel)
  {
    const double* factor           = lu_.data() + channel * matrix_elements_;
    double* inverse                = inverses_.data() + channel * matrix_elements_;
    const std::size_t* permutation = permutations_.data() + channel * matrix_size_;
    const double scale             = factorization_[channel].matrix_scale;

    // First form inv(A/scale); postpone division by scale so back substitution
    // never consumes already-rescaled columns.
    std::fill(inverse, inverse + matrix_elements_, 0.0);
    for (std::size_t rhs = 0; rhs < matrix_size_; ++rhs)
    {
      for (std::size_t row = 0; row < matrix_size_; ++row)
      {
        double value = permutation[row] == rhs ? 1.0 : 0.0;
        for (std::size_t inner = 0; inner < row; ++inner)
          value -= factor[row * matrix_size_ + inner] * solve_[inner];
        solve_[row] = value;
      }
      for (std::size_t reverse = matrix_size_; reverse > 0; --reverse)
      {
        const std::size_t row = reverse - 1;
        double value          = solve_[row];
        for (std::size_t inner = row + 1; inner < matrix_size_; ++inner)
          value -= factor[row * matrix_size_ + inner] *
              inverse[inner * matrix_size_ + rhs];
        inverse[row * matrix_size_ + rhs] = value / factor[row * matrix_size_ + row];
      }
    }

    bool finite = true;
    for (std::size_t element = 0; element < matrix_elements_; ++element)
    {
      inverse[element] /= scale;
      finite = finite && isFiniteReal(inverse[element]);
    }
    if (!finite)
      std::fill(inverse, inverse + matrix_elements_, 0.0);
    return finite;
  }

  /// Compute tr(left*right) in extended precision without temporary storage.
  long double traceProduct(const double* left, const double* right) const noexcept
  {
    long double trace = 0.0L;
    for (std::size_t row = 0; row < matrix_size_; ++row)
      for (std::size_t column = 0; column < matrix_size_; ++column)
        trace += static_cast<long double>(left[row * matrix_size_ + column]) *
            static_cast<long double>(right[column * matrix_size_ + row]);
    return trace;
  }

  /// Compute tr((inverse*derivative)^2) using reusable extended-precision scratch.
  long double inverseProductSquareTrace(const double* inverse,
                                        const double* derivative)
  {
    std::fill(matrix_product_.begin(), matrix_product_.end(), 0.0L);
    for (std::size_t row = 0; row < matrix_size_; ++row)
      for (std::size_t inner = 0; inner < matrix_size_; ++inner)
        for (std::size_t column = 0; column < matrix_size_; ++column)
          matrix_product_[row * matrix_size_ + column] +=
              static_cast<long double>(inverse[row * matrix_size_ + inner]) *
              static_cast<long double>(derivative[inner * matrix_size_ + column]);

    long double trace = 0.0L;
    for (std::size_t row = 0; row < matrix_size_; ++row)
      for (std::size_t column = 0; column < matrix_size_; ++column)
        trace += matrix_product_[row * matrix_size_ + column] *
            matrix_product_[column * matrix_size_ + row];
    return trace;
  }

  std::size_t channels_;
  std::size_t matrix_size_;
  std::size_t matrix_elements_;
  std::size_t gradient_capacity_;
  std::size_t laplacian_capacity_;

  std::vector<double> lu_;
  std::vector<double> inverses_;
  std::vector<std::size_t> permutations_;
  std::vector<ChannelFactorization> factorization_;
  std::vector<double> coefficient_;
  std::vector<double> term_phase_;
  std::vector<double> term_log_abs_;
  std::vector<long double> scaled_terms_;
  std::vector<double> solve_;
  std::vector<long double> matrix_product_;
  std::vector<detail::CompensatedSum> gradient_sums_;
  std::vector<detail::CompensatedSum> laplacian_sums_;
  std::vector<double> staged_gradient_;
  std::vector<double> staged_lap_log_;
  std::vector<double> staged_lap_ratio_;
  mutable std::vector<double> staged_matrix_adjoint_;

  double last_shift_          = -std::numeric_limits<double>::infinity();
  long double last_scaled_sum_ = 0.0L;
  RealDeterminantResult last_result_;
  bool evaluated_ = false;
};

} // namespace qmcplusplus::psiformer::determinant

#endif // QMCPLUSPLUS_PSIFORMER_DETERMINANT_H
