//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_FOURIER_FACTOR_H
#define QMCPLUSPLUS_PSIFORMER_FOURIER_FACTOR_H

#include "PsiFormerDeterminant.h"
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>

namespace qmcplusplus::psiformer
{
/// One row-local complex value, Cartesian gradient, and Laplacian.
struct FourierFactorJet
{
  using Complex = std::complex<double>;
  Complex value{};
  std::array<Complex, 3> gradient{};
  Complex laplacian{};
};

/** Evaluate a unit Fourier mode in an explicitly fixed Cartesian origin gauge.
 * Reciprocal membership and the immutable physical-ion origin are established
 * by the prepared descriptor, not inferred by this allocation-free helper.
 */
inline FourierFactorJet fourierModeJet(const std::array<double, 3>& position,
    const std::array<double, 3>& origin, const std::array<double, 3>& momentum)
{
  double angle = 0, norm_squared = 0;
  for (std::size_t d = 0; d < 3; ++d)
  {
    if (!determinant::isFiniteReal(position[d]) || !determinant::isFiniteReal(origin[d]) ||
        !determinant::isFiniteReal(momentum[d]))
      throw std::invalid_argument("Fourier mode requires finite position origin and momentum");
    angle += momentum[d] * (position[d] - origin[d]);
    norm_squared += momentum[d] * momentum[d];
  }
  if (!determinant::isFiniteReal(angle) || !determinant::isFiniteReal(norm_squared))
    throw std::overflow_error("Fourier mode argument or momentum norm overflowed");
  FourierFactorJet result;
  result.value = std::polar(1.0, angle);
  for (std::size_t d = 0; d < 3; ++d)
    result.gradient[d] = FourierFactorJet::Complex(0, momentum[d]) * result.value;
  result.laplacian = -norm_squared * result.value;
  return result;
}

/** Sum real-coefficient modes without retaining a mode-by-spatial-jet table.
 * Each unit mode is also exactly the derivative with respect to its real
 * coefficient. The caller supplies an already bounded prepared mode view.
 */
inline FourierFactorJet evaluateFourierFactor(const std::array<double, 3>& position,
    const std::array<double, 3>& origin, const std::array<double, 3>* momenta,
    const double* coefficients, std::size_t count)
{
  if (!count || !momenta || !coefficients)
    throw std::invalid_argument("Fourier factor requires nonempty mode and coefficient views");
  FourierFactorJet result;
  for (std::size_t mode = 0; mode < count; ++mode)
  {
    if (!determinant::isFiniteReal(coefficients[mode]))
      throw std::invalid_argument("Fourier factor coefficient must be finite");
    const auto basis = fourierModeJet(position, origin, momenta[mode]);
    result.value += coefficients[mode] * basis.value;
    result.laplacian += coefficients[mode] * basis.laplacian;
    for (std::size_t d = 0; d < 3; ++d)
      result.gradient[d] += coefficients[mode] * basis.gradient[d];
  }
  // Finite inputs can overflow weighted sums; release fast-math must not erase this check.
  for (const auto component : std::array<FourierFactorJet::Complex, 5>{result.value,
           result.gradient[0], result.gradient[1], result.gradient[2], result.laplacian})
    if (!determinant::isFiniteReal(component.real()) || !determinant::isFiniteReal(component.imag()))
      throw std::overflow_error("Fourier factor accumulated output overflowed");
  return result;
}

/// Apply the row-local product rule; zeros of either factor require no division.
inline FourierFactorJet multiplyFourierJets(const FourierFactorJet& a, const FourierFactorJet& b)
{
  FourierFactorJet result;
  result.value = a.value * b.value;
  result.laplacian = a.laplacian * b.value + a.value * b.laplacian;
  for (std::size_t d = 0; d < 3; ++d)
  {
    result.gradient[d] = a.gradient[d] * b.value + a.value * b.gradient[d];
    result.laplacian += 2.0 * a.gradient[d] * b.gradient[d];
  }
  return result;
}

/** Pull ordinary complex cotangents through multiplication by a fixed factor.
 * This is a transpose, not a Hermitian transpose. Laplacian cross terms feed
 * the gradient seeds; all derivatives of the factor feed the value seed.
 */
inline FourierFactorJet pullBackFourierFactor(const FourierFactorJet& factor,
    const FourierFactorJet& output_seed)
{
  FourierFactorJet result;
  result.value = factor.value * output_seed.value + factor.laplacian * output_seed.laplacian;
  result.laplacian = factor.value * output_seed.laplacian;
  for (std::size_t d = 0; d < 3; ++d)
  {
    result.value += factor.gradient[d] * output_seed.gradient[d];
    result.gradient[d] = factor.value * output_seed.gradient[d] +
        2.0 * factor.gradient[d] * output_seed.laplacian;
  }
  return result;
}
} // namespace qmcplusplus::psiformer
#endif
