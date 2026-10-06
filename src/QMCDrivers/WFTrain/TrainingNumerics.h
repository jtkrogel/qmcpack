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

} // namespace qmcplusplus::wftrain

#endif
