//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_PERIODIC_PAIR_ATTENTION_H
#define QMCPLUSPLUS_PSIFORMER_PERIODIC_PAIR_ATTENTION_H

#include "PsiFormerPeriodicPairFeatures.h"
#include "PsiFormerSpatialLayout.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QMC_PF_PAIR_ATTENTION_HD __host__ __device__
#else
#define QMC_PF_PAIR_ATTENTION_HD
#endif

namespace qmcplusplus::psiformer::periodic_geometry
{

/** Add one ordered pair's learned bias before softmax, preserving diagonal logits.
 * Positions are packed within one configuration. Each caller owns one complete
 * head/query/key jet, so no atomic or persistent pair-feature storage is needed.
 */
QMC_PF_PAIR_ATTENTION_HD inline bool addLogComplementAttentionJet(
    const SpatialAttentionJetLayout& layout, const Cell& cell, const double* positions,
    std::size_t spin_up_count, const double* weights, const double* bias,
    std::size_t configuration, std::size_t head, std::size_t query, std::size_t key,
    std::size_t active_electron, double* logits) noexcept
{
  if (query == key)
    return true;
  LogComplementPairJet pair;
  if (!evaluateLogComplementPair(cell, positions + 3 * query, positions + 3 * key,
          (query < spin_up_count) == (key < spin_up_count) ? 1 : -1, pair) ||
      (layout.attention.mode != SpatialJetMode::VALUE_ONLY && !pair.differentiable))
    return false;
  const auto element = head * layout.attention_head_stride + query * layout.attention_row_stride + key;
  auto offset = layout.attention.uncheckedValueOffset(configuration, element);
  if (!device_math::isFiniteBinary64(bias[head]))
    return false;
  logits[offset] += bias[head];
  for (std::size_t feature = 0; feature < 8; ++feature)
  {
    const double weight = weights[feature * layout.heads + head];
    if (!device_math::isFiniteBinary64(weight))
      return false;
    logits[offset] += weight * pair.value[feature];
    if (layout.attention.mode == SpatialJetMode::ACTIVE)
    {
      const double sign = active_electron == query ? 1 : active_electron == key ? -1 : 0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        logits[layout.attention.uncheckedGradientOffset(configuration, dimension, element)] +=
            sign * weight * pair.gradient[feature][dimension];
    }
    else if (layout.attention.mode == SpatialJetMode::FULL_VGL)
      for (std::size_t endpoint = 0; endpoint < 2; ++endpoint)
      {
        const auto target = endpoint == 0 ? query : key;
        if (layout.attention.containsTargetElectron(target))
        {
          const auto local = target - layout.attention.target_electron_begin;
          const double sign = target == query ? 1 : -1;
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
            logits[layout.attention.uncheckedGradientOffset(configuration, 3 * local + dimension, element)] +=
                sign * weight * pair.gradient[feature][dimension];
          logits[layout.attention.uncheckedLaplacianOffset(configuration, local, element)] +=
              weight * pair.laplacian[feature];
        }
      }
  }
  for (std::size_t plane = 0; plane < layout.attention.plane_count; ++plane)
    if (!device_math::isFiniteBinary64(logits[layout.attention.uncheckedPlaneOffset(configuration, plane, element)]))
      return false;
  return true;
}

/// Contract FULL_VGL target cotangents with one affine pair parameter jet.
/// VALUE score responses and ACTIVE coordinate reverse use separate contracts.
QMC_PF_PAIR_ATTENTION_HD inline double logComplementAttentionParameterAdjoint(
    const SpatialAttentionJetLayout& layout, const LogComplementPairJet& pair,
    const double* logit_adjoint, std::size_t configuration, std::size_t head,
    std::size_t query, std::size_t key, std::size_t feature) noexcept
{
  if (query == key)
    return 0;
  const auto element = head * layout.attention_head_stride + query * layout.attention_row_stride + key;
  const double value_root = logit_adjoint[layout.attention.uncheckedValueOffset(configuration, element)];
  if (feature == 8)
    return value_root;
  double result = value_root * pair.value[feature];
  for (std::size_t endpoint = 0; endpoint < 2; ++endpoint)
  {
    const auto target = endpoint == 0 ? query : key;
    if (layout.attention.containsTargetElectron(target))
    {
      const auto local = target - layout.attention.target_electron_begin;
      const double sign = endpoint == 0 ? 1 : -1;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        result += sign * pair.gradient[feature][dimension] *
            logit_adjoint[layout.attention.uncheckedGradientOffset(configuration, 3 * local + dimension, element)];
      result += pair.laplacian[feature] *
          logit_adjoint[layout.attention.uncheckedLaplacianOffset(configuration, local, element)];
    }
  }
  return result;
}

} // namespace qmcplusplus::psiformer::periodic_geometry

#undef QMC_PF_PAIR_ATTENTION_HD
#endif
