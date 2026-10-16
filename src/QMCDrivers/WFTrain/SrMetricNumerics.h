//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_SR_METRIC_NUMERICS_H
#define QMCPLUSPLUS_SR_METRIC_NUMERICS_H

#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace qmcplusplus::wftrain::sr_metric_detail
{
/** Preserve the length-P Hermitian contraction roundoff bound for SR diagnostics. */
inline DerivativeReal contractionRoundoffTolerance(std::size_t parameter_count,
    DerivativeValue contraction, DerivativeReal product_norm) noexcept
{
  const DerivativeReal accumulated_roundoff =
      static_cast<DerivativeReal>(parameter_count) *
      std::numeric_limits<DerivativeReal>::epsilon();
  const DerivativeReal gamma = accumulated_roundoff < 0.5
      ? accumulated_roundoff / (1.0 - accumulated_roundoff)
      : 1.0;
  return 128.0 * gamma *
      std::max({DerivativeReal{1}, std::abs(contraction.real()), product_norm});
}
}
#endif
