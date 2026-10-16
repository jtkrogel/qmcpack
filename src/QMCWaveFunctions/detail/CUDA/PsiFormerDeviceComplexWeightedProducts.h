//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_WEIGHTED_PRODUCTS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_WEIGHTED_PRODUCTS_H
#include "PsiFormerDeviceComplexValueKernels.h"
#include <cmath>

namespace qmcplusplus::psiformer::device::complex_detail
{
/// One signed product represented without exponent overflow or premature underflow.
struct ScaledProduct { double high, low; int exponent; bool active; };

/// Retain the bounded mantissa multiplication residual for near-cancelling components.
__device__ inline ScaledProduct product(double a, double b)
{
  if (a == 0 || b == 0)
    return {0, 0, 0, false};
  int ea, eb;
  const auto ma = frexp(a, &ea), mb = frexp(b, &eb), p = ma * mb;
  return {p, fma(ma, mb, -p), ea + eb, true};
}

/// Add two scaled signed products before the ONE final weighted-component exponential.
__device__ inline double weightedComponent(double log_weight, double a, double x, double b, double y)
{
  const ScaledProduct terms[2]{product(a, x), product(b, y)};
  if (!terms[0].active && !terms[1].active)
    return 0;
  const int exponent = !terms[0].active ? terms[1].exponent : (!terms[1].active ? terms[0].exponent :
      (terms[0].exponent > terms[1].exponent ? terms[0].exponent : terms[1].exponent));
  device_determinant::CompensatedSumBinary64 sum;
  for (const auto& term : terms)
    if (term.active)
    {
      sum.add(scalbn(term.high, term.exponent - exponent));
      sum.add(scalbn(term.low, term.exponent - exponent));
    }
  const auto t = sum.value();
  return t == 0 ? 0 : copysign(exp(log_weight + exponent * 0.69314718055994530942 + log(fabs(t))), t);
}

/// Compute exp(log_weight)*phase*z componentwise, with ordinary complex signs.
__device__ inline bool weightedProduct(double log_weight, cuDoubleComplex phase, cuDoubleComplex z, cuDoubleComplex& out)
{
  if (!device_math::isFiniteBinary64(log_weight) || !device_math::isFiniteBinary64(cuCreal(phase)) ||
      !device_math::isFiniteBinary64(cuCimag(phase)) || !device_math::isFiniteBinary64(cuCreal(z)) ||
      !device_math::isFiniteBinary64(cuCimag(z)))
    return false;
  out = make_cuDoubleComplex(weightedComponent(log_weight, cuCreal(phase), cuCreal(z), -cuCimag(phase), cuCimag(z)),
      weightedComponent(log_weight, cuCreal(phase), cuCimag(z), cuCimag(phase), cuCreal(z)));
  return device_math::isFiniteBinary64(cuCreal(out)) && device_math::isFiniteBinary64(cuCimag(out));
}
} // namespace qmcplusplus::psiformer::device::complex_detail
#endif
