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
  EMPIRICAL_QUANTILE      = 1
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

  /// Compare all semantic fields bit-for-bit rather than relying on a hash alone.
  bool equivalent(const EnergyClippingDescriptor& other) const noexcept;
};

/** Immutable symmetric clipping transform prepared from a complete population.
 *
 * Only the real part is clamped. The imaginary component is retained so complex
 * local-energy algebra is not silently changed by this real robust-statistics rule.
 */
class EnergyClippingTransform
{
public:
  explicit EnergyClippingTransform(EnergyClippingDescriptor descriptor);

  /// Return the complete immutable transform descriptor.
  const EnergyClippingDescriptor& descriptor() const noexcept { return descriptor_; }

  /// Clamp the real component and preserve the imaginary component exactly.
  DerivativeValue apply(DerivativeValue energy) const noexcept;

  /// Report whether applying the transform changes the real component.
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

} // namespace qmcplusplus::wftrain

#endif
