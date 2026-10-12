//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceDerivativeMath.h
 * @brief Host/device scalar cores for the ordinary real-FP64 PsiFormer reverse pass.
 *
 * These allocation-free primitives contain only the scalar algebra shared by the
 * CUDA and HIP derivative kernels.  Row reductions, indexing, accumulation, and
 * validation policy remain the responsibility of their callers.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_DERIVATIVE_MATH_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_DERIVATIVE_MATH_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"

#include <cmath>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QMC_PF_DERIVATIVE_HOST_DEVICE __host__ __device__
#define QMC_PF_DERIVATIVE_FORCE_INLINE __forceinline__
#else
#define QMC_PF_DERIVATIVE_HOST_DEVICE
#define QMC_PF_DERIVATIVE_FORCE_INLINE inline
#endif

namespace qmcplusplus::psiformer::device_derivative
{

/** Reverse tanh from its retained forward output. */
QMC_PF_DERIVATIVE_HOST_DEVICE QMC_PF_DERIVATIVE_FORCE_INLINE double tanhAdjointFromOutput(
    double tanh_output, double output_adjoint) noexcept
{
  return output_adjoint * (1.0 - tanh_output * tanh_output);
}

/** One term in sum_j probability[j] * probability_adjoint[j]. */
QMC_PF_DERIVATIVE_HOST_DEVICE QMC_PF_DERIVATIVE_FORCE_INLINE double
softmaxRowReductionContribution(double probability, double probability_adjoint) noexcept
{
  return probability * probability_adjoint;
}

/** Reverse one probability through a row softmax after its row reduction is known. */
QMC_PF_DERIVATIVE_HOST_DEVICE QMC_PF_DERIVATIVE_FORCE_INLINE double softmaxLogitAdjoint(
    double probability, double probability_adjoint, double row_reduction) noexcept
{
  return probability * (probability_adjoint - row_reduction);
}

/** One query/key feature contribution through scale * dot(query,key). */
QMC_PF_DERIVATIVE_HOST_DEVICE QMC_PF_DERIVATIVE_FORCE_INLINE double
scaledQkAdjointContribution(double logit_adjoint,
                            double opposite_feature,
                            double inverse_sqrt_head_width) noexcept
{
  return logit_adjoint * opposite_feature * inverse_sqrt_head_width;
}

struct AttentionContextReverseContributions
{
  double attention_adjoint;
  double value_adjoint;
};

static_assert(std::is_standard_layout_v<AttentionContextReverseContributions>);
static_assert(std::is_trivially_copyable_v<AttentionContextReverseContributions>);

/** Reverse one product in context[q,f] += attention[q,k] * value[k,f]. */
QMC_PF_DERIVATIVE_HOST_DEVICE QMC_PF_DERIVATIVE_FORCE_INLINE
AttentionContextReverseContributions attentionContextReverseContributions(
    double attention, double value, double context_adjoint) noexcept
{
  return {context_adjoint * value, attention * context_adjoint};
}

struct EnvelopeValuePartials
{
  double value;
  double pi_partial;
  double zeta_partial;
};

static_assert(std::is_standard_layout_v<EnvelopeValuePartials>);
static_assert(std::is_trivially_copyable_v<EnvelopeValuePartials>);

/** Value and parameter partials of pi * exp(-abs(zeta * radius)).
 *
 * At the cusp of abs(zeta*radius), the selected zeta subgradient is exactly zero,
 * matching the established ordinary score convention.
 */
QMC_PF_DERIVATIVE_HOST_DEVICE QMC_PF_DERIVATIVE_FORCE_INLINE EnvelopeValuePartials
envelopeValueAndPartials(double radius, double pi, double zeta) noexcept
{
  const double zeta_radius = zeta * radius;
  const double magnitude   = zeta_radius < 0.0 ? -zeta_radius : zeta_radius;
  const double decay       = ::exp(-magnitude);
  const double abs_derivative = zeta_radius > 0.0 ? 1.0 : (zeta_radius < 0.0 ? -1.0 : 0.0);
  return {pi * decay, decay, -pi * decay * abs_derivative * radius};
}

struct CuspValuePartial
{
  double value;
  double alpha_partial;
};

static_assert(std::is_standard_layout_v<CuspValuePartial>);
static_assert(std::is_trivially_copyable_v<CuspValuePartial>);

/** Value and alpha partial of -factor * alpha^2 / (alpha + radius). */
QMC_PF_DERIVATIVE_HOST_DEVICE QMC_PF_DERIVATIVE_FORCE_INLINE CuspValuePartial
cuspValueAndAlphaPartial(double radius, double alpha, double factor) noexcept
{
  const double denominator = alpha + radius;
  return {-factor * alpha * alpha / denominator,
          -factor * alpha * (alpha + 2.0 * radius) / (denominator * denominator)};
}

} // namespace qmcplusplus::psiformer::device_derivative

#undef QMC_PF_DERIVATIVE_HOST_DEVICE
#undef QMC_PF_DERIVATIVE_FORCE_INLINE

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_DERIVATIVE_MATH_H
