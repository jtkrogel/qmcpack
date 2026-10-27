//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_SCORE_RESIDUAL_POLICY_H
#define QMCPLUSPLUS_SCORE_RESIDUAL_POLICY_H

namespace qmcplusplus::wftrain
{

/// Sample residual transform, independent of derivative family and update scaling.
enum class ScoreResidualTransform
{
  RAW,
  SOFT_SQUEEZE,
  HARD_MEDIAN,
  HARD_MEAN,
  COMPONENTWISE_MAD,
  COMPONENTWISE_MEAN
};

/// Resolved scalar policy; no parameter products or optimizer state are retained.
struct ScoreResidualPolicy
{
  ScoreResidualTransform transform = ScoreResidualTransform::RAW;
  double width = 1.0;
  double quantile = 0.95;
  bool excludes_samples = false;
  double exclusion_width = 0.0;
};

} // namespace qmcplusplus::wftrain
#endif
