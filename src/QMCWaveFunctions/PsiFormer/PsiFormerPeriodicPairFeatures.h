//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_PERIODIC_PAIR_FEATURES_H
#define QMCPLUSPLUS_PSIFORMER_PERIODIC_PAIR_FEATURES_H

#include "PsiFormerPeriodicGeometry.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QMC_PF_PAIR_HD __host__ __device__
#else
#define QMC_PF_PAIR_HD
#endif

namespace qmcplusplus::psiformer::periodic_geometry
{

/// Ordered simulation-cell descriptors and Cartesian jets for one electron pair.
struct LogComplementPairJet
{
  double value[8]{}, gradient[8][3]{}, laplacian[8]{};
  bool differentiable = false;
};

/// Form the log-complement family without a persistent pair-by-spatial-plane tape.
QMC_PF_PAIR_HD inline bool evaluateLogComplementPair(const Cell& cell, const double* target,
    const double* source, double spin_product, LogComplementPairJet& output) noexcept
{
  output = LogComplementPairJet{};
  Pair pair;
  if ((spin_product != 1 && spin_product != -1) || !evaluatePair(cell, target, source, pair))
    return false;
  const auto radial = device_math::softenedRadial(pair.radius);
  output.differentiable = pair.radius != 0;
  output.value[0] = radial.log1p_radius;
  output.value[7] = spin_product;
  output.laplacian[0] = radial.log1p_second * pair.radius_gradient_norm_squared +
      radial.log1p_first * pair.radius_laplacian;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    output.gradient[0][dimension] = radial.log1p_first * pair.radius_gradient[dimension];
  for (std::size_t vector = 0; vector < 2; ++vector)
    for (std::size_t component = 0; component < 3; ++component)
    {
      const auto feature = 1 + 3 * vector + component;
      const double value = vector == 0 ? pair.displacement[component] : pair.complementary[component];
      const auto& jacobian = vector == 0 ? pair.jacobian[component] : pair.complementary_jacobian[component];
      const double laplacian = vector == 0 ? pair.laplacian[component] : pair.complementary_laplacian[component];
      output.value[feature] = value * radial.log1p_over_radius;
      double contraction = 0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        output.gradient[feature][dimension] = jacobian[dimension] * radial.log1p_over_radius +
            value * radial.log1p_over_radius_first * pair.radius_gradient[dimension];
        contraction += jacobian[dimension] * pair.radius_gradient[dimension];
      }
      output.laplacian[feature] = laplacian * radial.log1p_over_radius +
          2 * radial.log1p_over_radius_first * contraction +
          value * (radial.log1p_over_radius_second * pair.radius_gradient_norm_squared +
                   radial.log1p_over_radius_first * pair.radius_laplacian);
    }
  for (std::size_t feature = 0; feature < 8; ++feature)
  {
    if (!device_math::isFiniteBinary64(output.value[feature]) ||
        !device_math::isFiniteBinary64(output.laplacian[feature]))
      return false;
    for (double derivative : output.gradient[feature])
      if (!device_math::isFiniteBinary64(derivative))
        return false;
  }
  return true;
}

} // namespace qmcplusplus::psiformer::periodic_geometry

#undef QMC_PF_PAIR_HD
#endif
