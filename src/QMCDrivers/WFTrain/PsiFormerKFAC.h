//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerKFAC.h
 * @brief Typed PsiFormer registry and score-tape adapter for the bounded KFAC core.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_KFAC_H
#define QMCPLUSPLUS_PSIFORMER_KFAC_H

#include "QMCDrivers/WFTrain/KFACPreconditioner.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerAffineObservation.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"

namespace qmcplusplus::wftrain
{

/** Build the exact supported affine/diagonal block policy for one PsiFormer plan. */
KFACBlockRegistry makePsiFormerKFACRegistry(
    const psiformer::PsiFormerExecutionPlan& plan,
    const StructuredParameterSchema& schema);

/** Map typed direct-score observations into one checked KFAC accumulator.
 *
 * The adapter is allocation-free after construction.  Sample transaction ownership
 * remains with the caller so score fallback values can be added before endSample().
 */
class PsiFormerKFACObservationSink final : public pf::DirectAffineObservationSink
{
public:
  PsiFormerKFACObservationSink(const psiformer::PsiFormerExecutionPlan& plan,
                               KFACFactorAccumulator& accumulator);

  void observe(const pf::DirectAffineObservation& observation) override;

private:
  struct Binding
  {
    psiformer::ParameterRole role;
    std::size_t attention_block;
    std::size_t affine_block;
  };

  KFACFactorAccumulator& accumulator_;
  std::vector<Binding> bindings_;
};

} // namespace qmcplusplus::wftrain

#endif
