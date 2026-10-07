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
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QMC_PF_HOST_DEVICE __host__ __device__
#define QMC_PF_FORCE_INLINE __forceinline__
#else
#define QMC_PF_HOST_DEVICE
#define QMC_PF_FORCE_INLINE inline
#endif

namespace qmcplusplus::psiformer::device_math
{

enum class JetMathStatus : std::uint8_t
{
  REGULAR,
  NONFINITE_INPUT,
  NONFINITE_NORMALIZATION,
  NONFINITE_RESULT
};

/** IEEE-754 classification that remains meaningful with accelerator fast-math. */
QMC_PF_HOST_DEVICE QMC_PF_FORCE_INLINE bool isFiniteBinary64(double value) noexcept
{
  union Binary64
  {
    double floating;
    std::uint64_t integer;
  } encoded{value};
  return (encoded.integer & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

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

/** General product rule for first lanes and one contracted trace-Laplacian.
 *
 * The Laplacian is formed before any gradient output, so output gradients may
 * safely alias either input gradient array.
 */
template<typename T>
QMC_PF_HOST_DEVICE inline void productJet(T left_value,
                                          const T* left_gradient,
                                          T left_laplacian,
                                          T right_value,
                                          const T* right_gradient,
                                          T right_laplacian,
                                          std::size_t gradient_dimensions,
                                          T* output_value,
                                          T* output_gradient,
                                          T* output_laplacian) noexcept
{
  T gradient_dot = T(0);
  for (std::size_t dimension = 0; dimension < gradient_dimensions; ++dimension)
    gradient_dot += left_gradient[dimension] * right_gradient[dimension];
  *output_laplacian = left_laplacian * right_value + T(2) * gradient_dot +
      left_value * right_laplacian;
  for (std::size_t dimension = 0; dimension < gradient_dimensions; ++dimension)
    output_gradient[dimension] = left_gradient[dimension] * right_value +
        left_value * right_gradient[dimension];
  *output_value = left_value * right_value;
}

/** Stable in-place softmax of one multi-plane jet row.
 *
 * ``gradient`` addresses lane zero and ``laplacian`` addresses electron zero;
 * successive planes are separated by ``plane_stride``. Trace-Laplacians are
 * transformed before gradients because their centered rule consumes original
 * logit gradients. The row maximum is a numerical shift only and carries no jet.
 */
QMC_PF_HOST_DEVICE inline JetMathStatus softmaxJetRowInPlace(
    double* values,
    double* gradient,
    double* laplacian,
    std::size_t width,
    std::size_t plane_stride,
    std::size_t gradient_lanes,
    std::size_t laplacian_lanes) noexcept
{
  for (std::size_t column = 0; column < width; ++column)
    if (!isFiniteBinary64(values[column]))
      return JetMathStatus::NONFINITE_INPUT;
  for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
    for (std::size_t column = 0; column < width; ++column)
      if (!isFiniteBinary64(gradient[lane * plane_stride + column]))
        return JetMathStatus::NONFINITE_INPUT;
  for (std::size_t electron = 0; electron < laplacian_lanes; ++electron)
    for (std::size_t column = 0; column < width; ++column)
      if (!isFiniteBinary64(laplacian[electron * plane_stride + column]))
        return JetMathStatus::NONFINITE_INPUT;

  double maximum = values[0];
  for (std::size_t column = 1; column < width; ++column)
    maximum = values[column] > maximum ? values[column] : maximum;
  double normalization = 0.0;
  for (std::size_t column = 0; column < width; ++column)
  {
    values[column] = ::exp(values[column] - maximum);
    normalization += values[column];
  }
  if (!isFiniteBinary64(normalization) || normalization <= 0.0)
    return JetMathStatus::NONFINITE_NORMALIZATION;
  for (std::size_t column = 0; column < width; ++column)
    values[column] /= normalization;

  for (std::size_t electron = 0; electron < laplacian_lanes; ++electron)
  {
    double mean_laplacian = 0.0;
    double mean_gradient[3]{0.0, 0.0, 0.0};
    for (std::size_t column = 0; column < width; ++column)
    {
      const double probability = values[column];
      mean_laplacian += probability * laplacian[electron * plane_stride + column];
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        mean_gradient[dimension] += probability *
            gradient[(3 * electron + dimension) * plane_stride + column];
    }

    double mean_squared_deviation = 0.0;
    for (std::size_t column = 0; column < width; ++column)
    {
      double squared_deviation = 0.0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double deviation =
            gradient[(3 * electron + dimension) * plane_stride + column] -
            mean_gradient[dimension];
        squared_deviation += deviation * deviation;
      }
      mean_squared_deviation += values[column] * squared_deviation;
    }

    for (std::size_t column = 0; column < width; ++column)
    {
      double squared_deviation = 0.0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double deviation =
            gradient[(3 * electron + dimension) * plane_stride + column] -
            mean_gradient[dimension];
        squared_deviation += deviation * deviation;
      }
      const std::size_t index = electron * plane_stride + column;
      laplacian[index] = values[column] *
          (laplacian[index] - mean_laplacian + squared_deviation -
           mean_squared_deviation);
      if (!isFiniteBinary64(laplacian[index]))
        return JetMathStatus::NONFINITE_RESULT;
    }
  }

  for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
  {
    double mean_gradient = 0.0;
    for (std::size_t column = 0; column < width; ++column)
      mean_gradient += values[column] * gradient[lane * plane_stride + column];
    for (std::size_t column = 0; column < width; ++column)
    {
      const std::size_t index = lane * plane_stride + column;
      gradient[index] = values[column] * (gradient[index] - mean_gradient);
      if (!isFiniteBinary64(gradient[index]))
        return JetMathStatus::NONFINITE_RESULT;
    }
  }
  return JetMathStatus::REGULAR;
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

/** Softmax exponential after the row maximum has been removed. */
template<typename T>
QMC_PF_HOST_DEVICE QMC_PF_FORCE_INLINE T shiftedExponential(T logit, T row_maximum) noexcept
{
  return ::exp(logit - row_maximum);
}

/** Normalize one already exponentiated softmax element. */
template<typename T>
QMC_PF_HOST_DEVICE QMC_PF_FORCE_INLINE T normalizeExponential(T exponential, T row_sum) noexcept
{
  return exponential / row_sum;
}

} // namespace qmcplusplus::psiformer::device_math

#undef QMC_PF_HOST_DEVICE
#undef QMC_PF_FORCE_INLINE

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_MATH_H
