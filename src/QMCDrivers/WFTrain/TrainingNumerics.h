//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file TrainingNumerics.h
 * @brief Fast-math-safe scalar validation shared by training transactions.
 */

#ifndef QMCPLUSPLUS_TRAINING_NUMERICS_H
#define QMCPLUSPLUS_TRAINING_NUMERICS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace qmcplusplus::wftrain
{

/// Test an IEEE-754 binary64 value without a fast-math-sensitive comparison.
inline bool isFiniteTrainingReal(double value) noexcept
{
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  static_assert(std::numeric_limits<double>::is_iec559);
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

/// Recognize the explicit no-exclusion sentinel even under finite-math assumptions.
inline bool isPositiveInfinityTrainingReal(double value) noexcept
{
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  static_assert(std::numeric_limits<double>::is_iec559);
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits == UINT64_C(0x7ff0000000000000);
}

namespace detail
{
/// Bound projected transposed dot products by their actual vector/action norms.
inline bool projectedPairIsHermitian(double left, double right,
                                    double left_action_scale, double right_action_scale,
                                    double tolerance_multiplier, std::size_t parameter_count) noexcept
{
  // S-normalized vectors can have large Euclidean norms near weak overlap
  // directions, even when their projected entries nearly cancel.
  const double action_scale = left_action_scale + right_action_scale;
  if (!isFiniteTrainingReal(left) || !isFiniteTrainingReal(right) ||
      !isFiniteTrainingReal(action_scale) || action_scale < 0.0)
    return false;
  const double accumulated_roundoff = static_cast<double>(parameter_count) * std::numeric_limits<double>::epsilon();
  const double gamma = accumulated_roundoff < 0.5 ? accumulated_roundoff / (1.0 - accumulated_roundoff) : 1.0;
  const double tolerance = tolerance_multiplier * gamma *
      std::max({1.0, std::abs(left), std::abs(right), action_scale});
  if (!isFiniteTrainingReal(tolerance))
    return false;
  return std::abs(left - right) <= tolerance;
}
} // namespace detail

} // namespace qmcplusplus::wftrain

#endif
