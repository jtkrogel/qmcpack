//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerFoldedDense.h
 * @brief Host-testable shared-weight batching of contiguous spatial jet matrices.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_FOLDED_DENSE_H
#define QMCPLUSPLUS_PSIFORMER_FOLDED_DENSE_H

#include "PsiFormerAttention.h"
#include "PsiFormerSpatialLayout.h"

namespace qmcplusplus::psiformer
{

/** Fold [configuration,plane,row,feature] into one row-major dense product.
 *
 * The weight is identical on every plane: this changes only BLAS scheduling,
 * not the value/gradient/Laplacian algebra. Noncontiguous tensor layouts are
 * rejected instead of silently mixing padding with another configuration.
 */
inline DenseForwardLayout makeFoldedSpatialDenseLayout(const SpatialDenseJetLayout& layout)
{
  validateSpatialDenseJetLayout(layout);
  const auto multiply = [](std::size_t left, std::size_t right) {
    return spatial_detail::checkedProduct(left, right, "Folded spatial GEMM extent overflow");
  };
  const auto source_plane = multiply(layout.rows, layout.source_row_stride);
  const auto target_plane = multiply(layout.rows, layout.target_row_stride);
  if (layout.source.plane_stride != source_plane || layout.target.plane_stride != target_plane ||
      layout.source.configuration_stride != multiply(layout.source.plane_count, source_plane) ||
      layout.target.configuration_stride != multiply(layout.target.plane_count, target_plane) ||
      layout.source.plane_count != layout.target.plane_count ||
      layout.source.configuration_count != layout.target.configuration_count)
    throw std::invalid_argument("Folded spatial GEMM requires contiguous configuration/plane storage");
  DenseForwardLayout folded{
      multiply(multiply(layout.source.configuration_count, layout.source.plane_count), layout.rows),
      layout.input_width, layout.output_width, layout.source_row_stride,
      layout.weight_row_stride, layout.target_row_stride};
  validateDenseForwardLayout(folded);
  return folded;
}

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_FOLDED_DENSE_H
