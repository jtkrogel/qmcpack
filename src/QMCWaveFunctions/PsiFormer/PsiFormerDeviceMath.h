//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceMath.h
 * @brief Small host/device mathematical cores shared by CUDA and HIP PsiFormer kernels.
 *
 * These functions deliberately have no accelerator-runtime dependency.  Keeping the
 * scalar algebra host-callable lets ordinary CPU tests constrain the exact operations
 * compiled into both device backends before GPU runtime testing is available.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_MATH_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_MATH_H

#include <cmath>
#include <cstddef>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QMC_PF_HOST_DEVICE __host__ __device__
#define QMC_PF_FORCE_INLINE __forceinline__
#else
#define QMC_PF_HOST_DEVICE
#define QMC_PF_FORCE_INLINE inline
#endif

namespace qmcplusplus::psiformer::device_math
{

template<typename T>
struct SoftenedRadial
{
  T log1p_radius;
  T log1p_over_radius;
  T log1p_first;
  T log1p_second;
  T log1p_over_radius_first;
  T log1p_over_radius_second;
};

/** Stable factors for f(r)=log(1+r)/r and its first two radial derivatives.
 *
 * Callers own finite/nonnegative input validation.  The Taylor branch matches the
 * production CPU geometry policy and supplies the continuous values at r=0.
 */
template<typename T>
QMC_PF_HOST_DEVICE QMC_PF_FORCE_INLINE SoftenedRadial<T> softenedRadial(T radius) noexcept
{
  SoftenedRadial<T> factors{};
  factors.log1p_radius         = ::log1p(radius);
  const T inverse_one_plus     = T(1) / (T(1) + radius);
  factors.log1p_first          = inverse_one_plus;
  factors.log1p_second         = -inverse_one_plus * inverse_one_plus;

  if (radius < T(1.0e-2))
  {
    // Horner forms through r^10 avoid quotient cancellation near the origin.
    factors.log1p_over_radius =
        ((((((((((T(1) / T(11) * radius - T(1) / T(10)) * radius + T(1) / T(9)) * radius -
                    T(1) / T(8)) * radius + T(1) / T(7)) * radius - T(1) / T(6)) * radius +
                  T(1) / T(5)) * radius - T(1) / T(4)) * radius + T(1) / T(3)) * radius -
                T(1) / T(2)) * radius + T(1));
    factors.log1p_over_radius_first =
        (((((((((T(10) / T(11) * radius - T(9) / T(10)) * radius + T(8) / T(9)) * radius -
                   T(7) / T(8)) * radius + T(6) / T(7)) * radius - T(5) / T(6)) * radius +
                 T(4) / T(5)) * radius - T(3) / T(4)) * radius + T(2) / T(3)) * radius -
               T(1) / T(2));
    factors.log1p_over_radius_second =
        ((((((((T(90) / T(11) * radius - T(36) / T(5)) * radius + T(56) / T(9)) * radius -
                  T(21) / T(4)) * radius + T(30) / T(7)) * radius - T(10) / T(3)) * radius +
                T(12) / T(5)) * radius - T(3) / T(2)) * radius + T(2) / T(3));
    return factors;
  }

  factors.log1p_over_radius       = factors.log1p_radius / radius;
  const T numerator               = radius * inverse_one_plus - factors.log1p_radius;
  factors.log1p_over_radius_first = numerator / (radius * radius);
  factors.log1p_over_radius_second =
      -T(1) / (radius * (T(1) + radius) * (T(1) + radius)) -
      T(2) * numerator / (radius * radius * radius);
  return factors;
}

/** Assemble one cached open (4-wide) or periodic (7-wide) pair feature. */
template<typename T>
QMC_PF_HOST_DEVICE QMC_PF_FORCE_INLINE void assemblePairFeatures(const T* displacement,
                                                                 const T* complementary,
                                                                 T radius,
                                                                 bool periodic,
                                                                 T* output) noexcept
{
  const SoftenedRadial<T> factors = softenedRadial(radius);
  output[0]                       = factors.log1p_radius;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    output[1 + dimension] = displacement[dimension] * factors.log1p_over_radius;
  if (periodic)
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      output[4 + dimension] = complementary[dimension] * factors.log1p_over_radius;
}

template<typename T>
struct ScalarJet
{
  T value;
  T first;
  T second;
};

/** Apply tanh to a value and one first/diagonal-second derivative lane. */
template<typename T>
QMC_PF_HOST_DEVICE QMC_PF_FORCE_INLINE ScalarJet<T> tanhJet(ScalarJet<T> input) noexcept
{
  const T value       = ::tanh(input.value);
  const T derivative  = T(1) - value * value;
  return {value, derivative * input.first,
          derivative * input.second - T(2) * value * derivative * input.first * input.first};
}

template<typename T>
QMC_PF_HOST_DEVICE QMC_PF_FORCE_INLINE ScalarJet<T> addJets(ScalarJet<T> left,
                                                            ScalarJet<T> right) noexcept
{
  return {left.value + right.value, left.first + right.first, left.second + right.second};
}

/** One atom contribution to an exponential orbital envelope. */
template<typename T>
QMC_PF_HOST_DEVICE QMC_PF_FORCE_INLINE T envelopeContribution(T distance, T pi, T zeta) noexcept
{
  return pi * ::exp(-::fabs(zeta * distance));
}

/** One same-spin or opposite-spin analytic electron-cusp contribution. */
template<typename T>
QMC_PF_HOST_DEVICE QMC_PF_FORCE_INLINE T cuspPair(T distance, T alpha, T cusp_factor) noexcept
{
  return -cusp_factor * alpha * alpha / (alpha + distance);
}

} // namespace qmcplusplus::psiformer::device_math

#undef QMC_PF_HOST_DEVICE
#undef QMC_PF_FORCE_INLINE

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_MATH_H
