//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerOrbitalProjection.h
 * @brief Allocation-free affine orbital projection into retained backflow jets.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_ORBITAL_PROJECTION_H
#define QMCPLUSPLUS_PSIFORMER_ORBITAL_PROJECTION_H

#include "PsiFormerAttention.h"
#include "PsiFormerOpenOrbital.h"

namespace qmcplusplus::psiformer
{

/** One row-major X*W slice. Offsets count doubles, never bytes.
 *
 * The destination is the existing [B,plane,D,row,orbital] retained factor.
 * Tiling the channel dimension by one determinant accommodates arbitrary
 * admitted parameter/output padding without packing or additional scratch.
 */
struct OpenOrbitalProjectionSlice
{
  DenseForwardLayout dense;
  std::size_t source_offset;
  std::size_t weight_offset;
  std::size_t target_offset;
  bool spin_up;
};

/** Visit the spin/channel/plane GEMMs, omitting empty spin groups.
 *
 * The caller supplies the existing CPU or accelerator dense adapter. No policy
 * chooses this schedule automatically: the small scalar path remains available
 * until complete-operation measurements establish a useful dispatch threshold.
 * The callback must consume each slice immediately; this function owns no tape.
 */
template<class ApplyDense>
inline void forEachOpenOrbitalProjection(const OpenOrbitalJetLayout& layout, ApplyDense&& apply)
{
  validateOpenOrbitalJetLayout(layout);
  const auto electron_count = layout.features.electron_count;

  // Spin weights are shared across all configurations and spatial planes.
  for (std::size_t spin = 0; spin < 2; ++spin)
  {
    const auto first_row = spin == 0 ? 0 : layout.spin_up_count;
    const auto rows = spin == 0 ? layout.spin_up_count : electron_count - layout.spin_up_count;
    if (rows == 0)
      continue;
    const DenseForwardLayout dense{rows, layout.feature_width, electron_count,
        layout.feature_row_stride, layout.backflow_feature_stride, layout.orbital_row_stride};
    validateDenseForwardLayout(dense);

    // A determinant is a contiguous orbital-column tile in each weight row.
    for (std::size_t determinant = 0; determinant < layout.determinant_count; ++determinant)
      for (std::size_t configuration = 0; configuration < layout.features.configuration_count; ++configuration)
        for (std::size_t plane = 0; plane < layout.features.plane_count; ++plane)
          apply(OpenOrbitalProjectionSlice{dense,
              layout.features.uncheckedPlaneOffset(configuration, plane, first_row * layout.feature_row_stride),
              determinant * layout.backflow_determinant_stride,
              layout.orbitals.uncheckedPlaneOffset(configuration, plane,
                  open_orbital::outputElement(layout, determinant, first_row, 0)), spin == 0});
  }
}

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_ORBITAL_PROJECTION_H
