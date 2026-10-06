//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file ParameterGradient.h
 * @brief Minimal objective-neutral view consumed by parameter update rules.
 */

#ifndef QMCPLUSPLUS_PARAMETER_GRADIENT_H
#define QMCPLUSPLUS_PARAMETER_GRADIENT_H

#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"

#include <cstddef>
#include <string_view>

namespace qmcplusplus::wftrain
{

/** Bind one canonical real gradient to its schema, version, and reduction domain.
 *
 * The view is intentionally nonowning.  Objective owners retain scalar diagnostics
 * and the gradient vector while an optimizer consumes this short-lived interface.
 */
struct ParameterGradientView
{
  std::string_view schema_fingerprint;
  std::size_t parameter_version = 0;
  ReductionDomain reduction_domain = ReductionDomain::CROWD_LOCAL;
  DerivativeArrayView<const DerivativeReal> gradient;
};

} // namespace qmcplusplus::wftrain

#endif
