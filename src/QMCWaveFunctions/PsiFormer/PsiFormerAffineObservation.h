//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerAffineObservation.h
 * @brief Lightweight contract for transient PsiFormer affine-layer observations.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_AFFINE_OBSERVATION_H
#define QMCPLUSPLUS_PSIFORMER_AFFINE_OBSERVATION_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"

#include <cstddef>

namespace pf
{

/** Describe one nonowning affine layer boundary after the direct reverse pass. */
struct DirectAffineObservation
{
  qmcplusplus::psiformer::ParameterRole role;
  std::size_t attention_block = qmcplusplus::psiformer::NO_ATTENTION_BLOCK;
  const double* activations = nullptr;
  const double* sensitivities = nullptr;
  std::size_t row_count = 0;
  std::size_t input_width = 0;
  std::size_t output_width = 0;
};

/** Consume transient layer-local rows without retaining the score tape. */
class DirectAffineObservationSink
{
public:
  virtual ~DirectAffineObservationSink() = default;

  /// Consume one complete affine boundary before its scratch is reused.
  virtual void observe(const DirectAffineObservation& observation) = 0;
};

} // namespace pf

#endif
