//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file FrozenReferenceSumMath.h
 * Real amplitude jets for sums whose individual branches may have nodes.
 * Derivatives are amplitude numerators, never undefined child log derivatives.
 */
#ifndef QMCPLUSPLUS_FROZEN_REFERENCE_SUM_MATH_H
#define QMCPLUSPLUS_FROZEN_REFERENCE_SUM_MATH_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmcplusplus::residual
{

/** Value, Cartesian first derivatives and per-electron Laplacian of an amplitude. */
struct SpatialJet
{
  long double value = 0;
  std::vector<long double> gradient;
  std::vector<long double> laplacian;

  /// Allocate zero derivative numerators for the requested electron count.
  explicit SpatialJet(std::size_t electrons = 0, long double amplitude = 0)
      : value(amplitude), gradient(3 * electrons), laplacian(electrons)
  {}

  /// Validate dimensions and finite numerators before a public algebra operation.
  void validate() const
  {
    if (gradient.size() != 3 * laplacian.size())
      throw std::invalid_argument("Residual amplitude jet has inconsistent derivative extents");
    if (!std::isfinite(value) ||
        !std::all_of(gradient.begin(), gradient.end(), [](long double x) { return std::isfinite(x); }) ||
        !std::all_of(laplacian.begin(), laplacian.end(), [](long double x) { return std::isfinite(x); }))
      throw std::invalid_argument("Residual amplitude jet contains a nonfinite numerator");
  }
};

/** A positive common exponential scale multiplying every numerator in the jet. */
struct ScaledSpatialJet
{
  long double log_scale = 0;
  SpatialJet jet;
};

/// Add a signed multiple of a same-shaped jet without dividing by its value.
inline void addScaled(SpatialJet& output, const SpatialJet& input, long double factor)
{
  if (output.gradient.size() != input.gradient.size() || output.laplacian.size() != input.laplacian.size())
    throw std::invalid_argument("Residual amplitude addition has incompatible extents");
  output.value += factor * input.value;
  for (std::size_t k = 0; k < output.gradient.size(); ++k)
    output.gradient[k] += factor * input.gradient[k];
  for (std::size_t e = 0; e < output.laplacian.size(); ++e)
    output.laplacian[e] += factor * input.laplacian[e];
}

/// Apply the product rule to amplitude derivatives, including both gradient cross terms.
inline SpatialJet multiply(const SpatialJet& left, const SpatialJet& right)
{
  if (left.gradient.size() != right.gradient.size() || left.laplacian.size() != right.laplacian.size())
    throw std::invalid_argument("Residual amplitude product has incompatible extents");
  SpatialJet result(left.laplacian.size(), left.value * right.value);
  for (std::size_t k = 0; k < result.gradient.size(); ++k)
    result.gradient[k] = left.gradient[k] * right.value + left.value * right.gradient[k];
  for (std::size_t e = 0; e < result.laplacian.size(); ++e)
  {
    result.laplacian[e] = left.laplacian[e] * right.value + left.value * right.laplacian[e];
    for (std::size_t d = 0; d < 3; ++d)
      result.laplacian[e] += 2 * left.gradient[3 * e + d] * right.gradient[3 * e + d];
  }
  return result;
}

/// Multiply scaled branches without materializing their potentially extreme absolute values.
inline ScaledSpatialJet multiply(const ScaledSpatialJet& left, const ScaledSpatialJet& right)
{
  return {left.log_scale + right.log_scale, multiply(left.jet, right.jet)};
}

/** Form Phi+c*N in a common scale, preserving finite derivatives at a child node. */
inline ScaledSpatialJet sum(const ScaledSpatialJet& reference, const ScaledSpatialJet& neural,
                            long double coefficient)
{
  reference.jet.validate();
  neural.jet.validate();
  if (!std::isfinite(coefficient) || !std::isfinite(reference.log_scale) || !std::isfinite(neural.log_scale))
    throw std::invalid_argument("Residual amplitude sum requires finite coefficients and scales");
  if (coefficient == 0)
    return reference;
  if (reference.jet.gradient.size() != neural.jet.gradient.size())
    throw std::invalid_argument("Residual branches have different electron counts");

  // The coefficient enters the scale and sign once, not the child normalization.
  const long double neural_scale = neural.log_scale + std::log(std::abs(coefficient));
  const long double scale = std::max(reference.log_scale, neural_scale);
  SpatialJet result(reference.jet.laplacian.size());
  addScaled(result, reference.jet, std::exp(reference.log_scale - scale));
  addScaled(result, neural.jet, std::copysign(std::exp(neural_scale - scale), coefficient));
  result.validate();
  return {scale, std::move(result)};
}

/** Total log amplitude, gradient and Laplacian for a nonzero summed wavefunction. */
struct LogSpatialJet
{
  long double log_abs = 0;
  int sign = 1;
  std::vector<long double> gradient;
  std::vector<long double> laplacian;
};

/// Convert only the final nonzero amplitude to logarithmic spatial derivatives.
inline LogSpatialJet logarithmic(const ScaledSpatialJet& amplitude)
{
  amplitude.jet.validate();
  if (!std::isfinite(amplitude.log_scale))
    throw std::invalid_argument("Residual amplitude has a nonfinite common scale");
  if (amplitude.jet.value == 0)
    throw std::domain_error("Residual total amplitude is zero; logarithmic derivatives are undefined");
  LogSpatialJet result;
  result.log_abs = amplitude.log_scale + std::log(std::abs(amplitude.jet.value));
  result.sign = amplitude.jet.value < 0 ? -1 : 1;
  result.gradient.resize(amplitude.jet.gradient.size());
  result.laplacian.resize(amplitude.jet.laplacian.size());
  for (std::size_t k = 0; k < result.gradient.size(); ++k)
    result.gradient[k] = amplitude.jet.gradient[k] / amplitude.jet.value;
  for (std::size_t e = 0; e < result.laplacian.size(); ++e)
  {
    result.laplacian[e] = amplitude.jet.laplacian[e] / amplitude.jet.value;
    for (std::size_t d = 0; d < 3; ++d)
      result.laplacian[e] -= result.gradient[3 * e + d] * result.gradient[3 * e + d];
  }
  return result;
}

/** Small-matrix determinant with division-free spatial derivatives at singular matrices.
 * Subset expansion costs O(n*2^n) jet operations; restricted to the small CPU profile.
 */
inline ScaledSpatialJet determinant(std::vector<SpatialJet> matrix, std::size_t order)
{
  if (order == 0 || order > 12 || matrix.size() != order * order)
    throw std::invalid_argument("Residual determinant requires a square matrix of order 1 through 12");
  const std::size_t electrons = matrix.front().laplacian.size();
  long double log_scale = 0;
  for (std::size_t row = 0; row < order; ++row)
  {
    // Row scaling includes derivatives, so a zero value row with nonzero derivatives survives.
    long double row_scale = 0;
    for (std::size_t col = 0; col < order; ++col)
    {
      auto& entry = matrix[row * order + col];
      entry.validate();
      if (entry.laplacian.size() != electrons)
        throw std::invalid_argument("Residual determinant entries have different derivative extents");
      row_scale = std::max(row_scale, std::abs(entry.value));
      for (long double x : entry.gradient)
        row_scale = std::max(row_scale, std::abs(x));
      for (long double x : entry.laplacian)
        row_scale = std::max(row_scale, std::abs(x));
    }
    if (row_scale == 0)
      return {0, SpatialJet(electrons)};
    log_scale += std::log(row_scale);
    for (std::size_t col = 0; col < order; ++col)
    {
      auto& entry = matrix[row * order + col];
      entry.value /= row_scale;
      for (auto& x : entry.gradient)
        x /= row_scale;
      for (auto& x : entry.laplacian)
        x /= row_scale;
    }
  }

  // A state retains the determinant of the first popcount(mask) rows and selected columns.
  const std::size_t states = std::size_t(1) << order;
  std::vector<SpatialJet> partial(states, SpatialJet(electrons));
  partial[0].value = 1;
  for (std::size_t mask = 1; mask < states; ++mask)
  {
    std::size_t count = 0;
    for (std::size_t bits = mask; bits; bits >>= 1)
      count += bits & 1;
    std::size_t selected_column = 0;
    for (std::size_t col = 0; col < order; ++col)
      if (mask & (std::size_t(1) << col))
      {
        const auto term = multiply(partial[mask ^ (std::size_t(1) << col)], matrix[(count - 1) * order + col]);
        const long double sign = ((count - 1 + selected_column) & 1) ? -1 : 1;
        addScaled(partial[mask], term, sign);
        ++selected_column;
      }
  }
  partial.back().validate();
  return {log_scale, std::move(partial.back())};
}

} // namespace qmcplusplus::residual

#endif
