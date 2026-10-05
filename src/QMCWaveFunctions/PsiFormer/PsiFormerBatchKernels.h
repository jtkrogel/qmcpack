//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerBatchKernels.h
 * @brief Checked CPU kernels and diagnostics for tiled PsiFormer execution.
 *
 * Shared-weight projections flatten configuration and electron rows into one dense
 * product.  Attention remains configuration-local: constructing a block-diagonal
 * (tile*electron)-square matrix would add cross-configuration work and storage.
 * These small helpers form the CPU backend seam for a future grouped accelerator
 * implementation without exposing backend details through DirectBatchWorkspace.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_BATCH_KERNELS_H
#define QMCPLUSPLUS_PSIFORMER_BATCH_KERNELS_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerDenseKernels.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer::batch
{

/// Deterministic counters used to verify that a batch did not serialize scalar executors.
struct ExecutionStatistics
{
  std::size_t tiles_executed        = 0;
  std::size_t max_tile_occupancy    = 0;
  std::size_t grouped_dense_calls   = 0;
  std::size_t max_grouped_rows      = 0;
  std::size_t scalar_executor_calls = 0;
  /// Sparse VALUE input diagnostics.  Dense and spatial calls leave these zero.
  std::size_t reference_configurations   = 0;
  std::size_t replacement_configurations = 0;
  std::size_t reference_evaluations      = 0;
  /** Coordinate bytes avoided relative to dense [R+Q,Ne,3] packing.
   * This is exactly Q*(Ne-1)*3*sizeof(double); metadata and results are excluded.
   */
  std::size_t dense_coordinate_bytes_avoided = 0;
};

/// Add two extents or reject an impossible allocation before state is changed.
inline std::size_t checkedSum(std::size_t first,
                              std::size_t second,
                              const char* quantity)
{
  if (second > std::numeric_limits<std::size_t>::max() - first)
    throw std::length_error(quantity);
  return first + second;
}

/// Multiply two extents or reject an impossible allocation before state is changed.
inline std::size_t checkedProduct(std::size_t first,
                                  std::size_t second,
                                  const char* quantity)
{
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first)
    throw std::length_error(quantity);
  return first * second;
}

/// Apply one shared-weight product and record its complete stacked row extent.
inline void productReal(const double* source,
                        const double* weight,
                        const double* bias,
                        std::size_t rows,
                        std::size_t input_width,
                        std::size_t output_width,
                        double* target,
                        ExecutionStatistics& statistics)
{
  dense::productReal(source, weight, bias, rows, input_width, output_width, target);
  ++statistics.grouped_dense_calls;
  statistics.max_grouped_rows = std::max(statistics.max_grouped_rows, rows);
}

/// Apply three shared Q/K/V projections to the same stacked configuration rows.
inline void projectQkvReal(const double* source,
                           const double* query_weight,
                           const double* key_weight,
                           const double* value_weight,
                           std::size_t rows,
                           std::size_t width,
                           double* query,
                           double* key,
                           double* value,
                           ExecutionStatistics& statistics)
{
  dense::projectQkvReal(
      source, query_weight, key_weight, value_weight, rows, width, query, key, value);
  statistics.grouped_dense_calls += 3;
  statistics.max_grouped_rows = std::max(statistics.max_grouped_rows, rows);
}

/** Evaluate attention independently inside every configuration in a tile.
 *
 * Q/K/V and target are configuration-major [tile,electron,feature].  Attention is
 * [tile,head,electron,electron].  The existing stable row softmax and strided head
 * kernels are reused verbatim, so batching cannot introduce cross-walker terms.
 */
inline void attentionReal(const double* query,
                          const double* key,
                          const double* value,
                          std::size_t tile_size,
                          std::size_t electrons,
                          std::size_t heads,
                          std::size_t head_width,
                          double* attention,
                          double* target)
{
  const std::size_t width                 = heads * head_width;
  const std::size_t feature_configuration = electrons * width;
  const std::size_t attention_configuration = heads * electrons * electrons;
  for (std::size_t configuration = 0; configuration < tile_size; ++configuration)
  {
    const std::size_t feature_offset = configuration * feature_configuration;
    double* configuration_attention =
        attention + configuration * attention_configuration;
    dense::attentionWeightsReal(query + feature_offset, width, key + feature_offset,
                                width, electrons, heads, head_width,
                                configuration_attention);
    dense::attentionContextReal(configuration_attention, value + feature_offset, width,
                                electrons, heads, head_width,
                                target + feature_offset);
  }
}

} // namespace qmcplusplus::psiformer::batch

#endif // QMCPLUSPLUS_PSIFORMER_BATCH_KERNELS_H
