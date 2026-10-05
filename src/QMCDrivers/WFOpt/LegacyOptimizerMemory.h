//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file LegacyOptimizerMemory.h
 * @brief Checked memory estimates and safety checks for legacy dense optimizers.
 */

#ifndef QMCPLUSPLUS_LEGACY_OPTIMIZER_MEMORY_H
#define QMCPLUSPLUS_LEGACY_OPTIMIZER_MEMORY_H

#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace qmcplusplus::optimizer_memory
{

/// Default per-rank limit for storage whose size is known before optimization starts.
inline constexpr std::size_t default_safe_byte_limit = std::size_t{2} * 1024 * 1024 * 1024;

/// Multiply two allocation dimensions and report overflow before an allocator is called.
inline std::size_t checkedMultiply(std::size_t lhs, std::size_t rhs, const char* allocation_name)
{
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    throw std::overflow_error(std::string(allocation_name) + " byte estimate overflowed size_t");
  return lhs * rhs;
}

/// Add two allocation dimensions and report overflow before an allocator is called.
inline std::size_t checkedAdd(std::size_t lhs, std::size_t rhs, const char* allocation_name)
{
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    throw std::overflow_error(std::string(allocation_name) + " byte estimate overflowed size_t");
  return lhs + rhs;
}

/** Estimate persistent sample-by-parameter derivative records.
 *
 * The parameter count must be the stored matrix dimension, not merely the
 * number of active entries in a VariableSet, because that is what the legacy
 * cost-function matrices allocate.
 */
inline std::size_t estimateDerivativeStorageBytes(std::size_t samples,
                                                  std::size_t stored_parameters,
                                                  bool stores_log_derivatives,
                                                  bool stores_energy_derivatives,
                                                  std::size_t log_derivative_size,
                                                  std::size_t energy_derivative_size)
{
  std::size_t bytes_per_entry = stores_log_derivatives ? log_derivative_size : 0;
  if (stores_energy_derivatives)
    bytes_per_entry = checkedAdd(bytes_per_entry, energy_derivative_size, "Legacy derivative storage");
  if (bytes_per_entry == 0)
    return 0;

  const std::size_t entries =
      checkedMultiply(samples, stored_parameters, "Legacy derivative storage");
  return checkedMultiply(entries, bytes_per_entry, "Legacy derivative storage");
}

/** Estimate simultaneous square matrices used by a dense linear optimizer.
 *
 * Linear-method matrices include the normalization row and column, hence the
 * ``stored_parameters + 1`` dimension.
 */
inline std::size_t estimateDenseMatrixStorageBytes(std::size_t stored_parameters,
                                                   std::size_t matrix_count,
                                                   std::size_t element_size)
{
  const std::size_t dimension = checkedAdd(stored_parameters, 1, "Legacy dense optimizer storage");
  const std::size_t entries_per_matrix =
      checkedMultiply(dimension, dimension, "Legacy dense optimizer storage");
  const std::size_t total_entries =
      checkedMultiply(entries_per_matrix, matrix_count, "Legacy dense optimizer storage");
  return checkedMultiply(total_entries, element_size, "Legacy dense optimizer storage");
}

/// Reject a known dense-matrix peak before sampling or matrix construction begins.
inline void validateDenseMatrixStorage(std::size_t stored_parameters,
                                       std::size_t matrix_count,
                                       std::size_t element_size,
                                       std::size_t safe_byte_limit = default_safe_byte_limit)
{
  const std::size_t estimated_bytes =
      estimateDenseMatrixStorageBytes(stored_parameters, matrix_count, element_size);
  if (estimated_bytes <= safe_byte_limit)
    return;

  std::ostringstream message;
  message << "Legacy dense optimizer requires approximately " << estimated_bytes
          << " bytes per rank for " << matrix_count << " parameter-quadratic matrices with "
          << stored_parameters << " stored parameters, exceeding the " << safe_byte_limit
          << " byte safety limit. Use a streaming first-order or matrix-free optimizer, reduce the parameter "
             "selection, or explicitly redesign the dense storage path.";
  throw std::runtime_error(message.str());
}

} // namespace qmcplusplus::optimizer_memory

#endif // QMCPLUSPLUS_LEGACY_OPTIMIZER_MEMORY_H
