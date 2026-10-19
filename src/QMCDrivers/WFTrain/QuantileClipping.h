//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file QuantileClipping.h
 * @brief Exact distributed construction of an immutable local-energy clipping rule.
 */

#ifndef QMCPLUSPLUS_QUANTILE_CLIPPING_H
#define QMCPLUSPLUS_QUANTILE_CLIPPING_H

#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"

#include <cstddef>
#include <cstdint>

class Communicate;

namespace qmcplusplus::wftrain
{

/// Select the explicit robust scale used to form a symmetric clipping interval.
enum class EnergyClippingScaleRule : std::uint64_t
{
  MEAN_ABSOLUTE_DEVIATION = 0,
  EMPIRICAL_QUANTILE      = 1,
  MEDIAN_ABSOLUTE_DEVIATION = 2
};

/// Fully specified policy used identically by every distributed participant.
struct EnergyClippingPolicy
{
  EnergyClippingScaleRule scale_rule = EnergyClippingScaleRule::MEAN_ABSOLUTE_DEVIATION;
  DerivativeReal width_multiplier    = 5.0;
  DerivativeReal residual_quantile   = 0.5;
};

/// Stable, serializable description of one prepared population transform.
struct EnergyClippingDescriptor
{
  EnergyClippingPolicy policy;
  std::size_t population = 0;
  DerivativeReal center  = 0.0;
  DerivativeReal scale   = 0.0;
  DerivativeReal width   = 0.0;
  std::uint64_t identity = 0;
  bool componentwise = false;
  DerivativeReal imaginary_center = 0.0;
  DerivativeReal imaginary_scale = 0.0;
  DerivativeReal imaginary_width = 0.0;

  /// Compare all semantic fields bit-for-bit rather than relying on a hash alone.
  bool equivalent(const EnergyClippingDescriptor& other) const noexcept;
};

/** Immutable symmetric clipping transform prepared from a complete population.
 *
 * The ordinary policy clamps only the real part. An explicitly componentwise
 * descriptor instead supplies independently prepared real and imaginary windows.
 */
class EnergyClippingTransform
{
public:
  explicit EnergyClippingTransform(EnergyClippingDescriptor descriptor);

  /// Return the complete immutable transform descriptor.
  const EnergyClippingDescriptor& descriptor() const noexcept { return descriptor_; }

  /// Apply the explicitly prepared windows, preserving unselected channels.
  DerivativeValue apply(DerivativeValue energy) const noexcept;

  /// Report whether either selected component changes.
  bool clips(DerivativeValue energy) const noexcept;

private:
  EnergyClippingDescriptor descriptor_;
};

/** Prepare one exact-median transform over all energies in the communicator.
 *
 * Passing a null communicator selects the same single-participant algorithm. Empty
 * local partitions are valid when another rank contributes at least one sample.
 */
EnergyClippingTransform prepareEnergyClippingTransform(
    DerivativeArrayView<const DerivativeValue> local_energies,
    const EnergyClippingPolicy& policy = {},
    Communicate* communicator = nullptr);

/// Prepare independent median/MAD windows for unit-weight complex samples across ranks.
EnergyClippingTransform prepareComplexEnergyClippingTransform(
    DerivativeArrayView<const DerivativeValue> local_energies,
    DerivativeArrayView<const DerivativeReal> local_weights,
    DerivativeReal width_multiplier = 5.0,
    Communicate* communicator = nullptr);

} // namespace qmcplusplus::wftrain

#endif
