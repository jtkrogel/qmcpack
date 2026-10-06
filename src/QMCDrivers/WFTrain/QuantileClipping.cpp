//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file QuantileClipping.cpp
 * @brief Bounded-memory exact selection for distributed local-energy clipping.
 */

#include "QMCDrivers/WFTrain/QuantileClipping.h"

#include "Message/CommOperators.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

constexpr std::uint64_t clipping_protocol_version = 1;
constexpr std::uint64_t sign_mask                 = UINT64_C(1) << 63;
constexpr std::uint64_t fraction_mask             = UINT64_C(0x000fffffffffffff);

/// Identify a rank-local validation failure without transporting exception strings.
enum class ConsensusReason : std::uint64_t
{
  NONE = 0,
  INVALID_POLICY,
  NONFINITE_INPUT,
  COUNT_OVERFLOW,
  RESIDUAL_OVERFLOW
};

/// Return the bit pattern of one binary64 scalar.
std::uint64_t doubleBits(double value) noexcept
{
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

/// Classify binary64 values without relying on fast-math-sensitive library predicates.
bool isFiniteBits(double value) noexcept
{
  constexpr std::uint64_t exponent_mask = UINT64_C(0x7ff0000000000000);
  return (doubleBits(value) & exponent_mask) != exponent_mask;
}

/// Convert a finite binary64 value to a key ordered by ordinary numeric comparison.
std::uint64_t orderedKey(double value) noexcept
{
  const std::uint64_t bits = doubleBits(value);
  return (bits & sign_mask) != 0 ? ~bits : bits ^ sign_mask;
}

/// Invert orderedKey after all eight radix bytes have been selected.
double valueFromOrderedKey(std::uint64_t key) noexcept
{
  const std::uint64_t bits = (key & sign_mask) != 0 ? key ^ sign_mask : ~key;
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

/// Gather one fixed-width record without depending on std::array tail padding.
template<std::size_t field_count>
std::vector<std::array<std::uint64_t, field_count>> gatherRecords(
    Communicate* communicator,
    const std::array<std::uint64_t, field_count>& local_record)
{
  const std::size_t participant_count = communicator ? communicator->size() : 1;
  std::vector<std::array<std::uint64_t, field_count>> records(participant_count);
  if (communicator)
  {
    std::vector<std::uint64_t> flat_records(participant_count * field_count);
    auto send_record = local_record;
    communicator->allgather(send_record.data(), flat_records.data(),
                            static_cast<int>(field_count));
    for (std::size_t rank = 0; rank < participant_count; ++rank)
      std::copy_n(flat_records.data() + rank * field_count, field_count,
                  records[rank].begin());
  }
  else
    records.front() = local_record;
  return records;
}

/// Throw one rank-uniform diagnostic for a failed clipping consensus.
[[noreturn]] void throwConsensusFailure(const char* stage,
                                        std::size_t rank,
                                        ConsensusReason reason)
{
  std::ostringstream message;
  message << "Distributed energy clipping " << stage << " failed on rank " << rank
          << " (reason " << static_cast<std::uint64_t>(reason) << ')';
  throw std::runtime_error(message.str());
}

/// Verify rank-local validity and identical policy before selection collectives.
std::size_t clippingPreflight(DerivativeArrayView<const DerivativeValue> energies,
                              const EnergyClippingPolicy& policy,
                              Communicate* communicator)
{
  ConsensusReason reason = ConsensusReason::NONE;
  const bool known_rule = policy.scale_rule == EnergyClippingScaleRule::MEAN_ABSOLUTE_DEVIATION ||
      policy.scale_rule == EnergyClippingScaleRule::EMPIRICAL_QUANTILE;
  if (!known_rule || !isFiniteBits(policy.width_multiplier) ||
      policy.width_multiplier < 0.0 || !isFiniteBits(policy.residual_quantile) ||
      policy.residual_quantile < 0.0 || policy.residual_quantile > 1.0)
    reason = ConsensusReason::INVALID_POLICY;
  if (reason == ConsensusReason::NONE && energies.size() != 0 && energies.data() == nullptr)
    reason = ConsensusReason::NONFINITE_INPUT;
  if (reason == ConsensusReason::NONE)
    for (const DerivativeValue energy : energies)
      if (!isFiniteBits(std::real(energy)) || !isFiniteBits(std::imag(energy)))
      {
        reason = ConsensusReason::NONFINITE_INPUT;
        break;
      }

  const std::array<std::uint64_t, 7> local_record{
      static_cast<std::uint64_t>(reason), clipping_protocol_version,
      static_cast<std::uint64_t>(policy.scale_rule), doubleBits(policy.width_multiplier),
      doubleBits(policy.residual_quantile), energies.size(), sizeof(std::size_t)};
  const auto records = gatherRecords(communicator, local_record);

  for (std::size_t rank = 0; rank < records.size(); ++rank)
    if (records[rank][0] != static_cast<std::uint64_t>(ConsensusReason::NONE))
      throwConsensusFailure("preflight", rank,
                            static_cast<ConsensusReason>(records[rank][0]));
  for (std::size_t rank = 1; rank < records.size(); ++rank)
    for (std::size_t field = 1; field < local_record.size(); ++field)
      if (field != 5 && records[rank][field] != records.front()[field])
        throw std::runtime_error("Distributed energy clipping policy metadata mismatch");

  std::uint64_t population = 0;
  for (const auto& record : records)
  {
    if (record[5] > std::numeric_limits<std::uint64_t>::max() - population)
      throwConsensusFailure("population", 0, ConsensusReason::COUNT_OVERFLOW);
    population += record[5];
  }
  if (population == 0 || population > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("Distributed energy clipping requires a nonempty population");
  return static_cast<std::size_t>(population);
}

/** Select one zero-based global order statistic with eight fixed radix passes.
 *
 * The accessor is reevaluated on each pass, avoiding storage proportional to the
 * local population. All ranks, including empty ones, execute the identical schedule.
 */
template<typename Accessor>
double selectOrderStatistic(std::size_t local_count,
                            std::size_t global_ordinal,
                            Accessor&& accessor,
                            Communicate* communicator)
{
  std::uint64_t prefix = 0;
  std::uint64_t ordinal = global_ordinal;
  for (unsigned pass = 0; pass < 8; ++pass)
  {
    std::array<std::uint64_t, 256> bins{};
    const unsigned shift = 56 - 8 * pass;
    for (std::size_t index = 0; index < local_count; ++index)
    {
      const std::uint64_t key = orderedKey(accessor(index));
      if (pass != 0 && (key >> (64 - 8 * pass)) != prefix)
        continue;
      ++bins[(key >> shift) & UINT64_C(0xff)];
    }
    if (communicator)
      communicator->allreduce_in_place(bins.data(), bins.size());

    std::size_t selected_bin = bins.size();
    for (std::size_t bin = 0; bin < bins.size(); ++bin)
      if (ordinal < bins[bin])
      {
        selected_bin = bin;
        break;
      }
      else
        ordinal -= bins[bin];
    if (selected_bin == bins.size())
      throw std::runtime_error("Distributed energy clipping radix selection lost its target");
    prefix = (prefix << 8) | selected_bin;
  }
  return valueFromOrderedKey(prefix);
}

/// Form an overflow-resistant arithmetic midpoint for the even-population median.
double finiteMidpoint(double lower, double upper)
{
  const double midpoint = lower / 2.0 + upper / 2.0;
  if (!isFiniteBits(midpoint))
    throw std::runtime_error("Distributed energy clipping median is non-finite");
  return midpoint;
}

/// Hold the exact two-word product of two unsigned 64-bit integers.
struct UInt128Product
{
  std::uint64_t high;
  std::uint64_t low;
};

/// Multiply two unsigned 64-bit integers without requiring a compiler-specific type.
UInt128Product multiplyWide(std::uint64_t left, std::uint64_t right) noexcept
{
  constexpr std::uint64_t low_word_mask = UINT64_C(0xffffffff);
  const std::uint64_t left_low   = left & low_word_mask;
  const std::uint64_t left_high  = left >> 32;
  const std::uint64_t right_low  = right & low_word_mask;
  const std::uint64_t right_high = right >> 32;

  const std::uint64_t low_low   = left_low * right_low;
  const std::uint64_t low_high  = left_low * right_high;
  const std::uint64_t high_low  = left_high * right_low;
  const std::uint64_t high_high = left_high * right_high;
  const std::uint64_t middle =
      (low_low >> 32) + (low_high & low_word_mask) + (high_low & low_word_mask);

  return {high_high + (low_high >> 32) + (high_low >> 32) + (middle >> 32),
          (middle << 32) | (low_low & low_word_mask)};
}

/// Return the low word of an exact unsigned 128-bit right shift.
std::uint64_t shiftWideRight(UInt128Product value, unsigned shift) noexcept
{
  if (shift >= 128)
    return 0;
  if (shift == 64)
    return value.high;
  if (shift > 64)
    return value.high >> (shift - 64);
  if (shift == 0)
    return value.low;
  return (value.high << (64 - shift)) | (value.low >> shift);
}

/** Compute ceil(q * population) - 1 exactly for binary64 q in (0, 1).
 *
 * Floating-point multiplication can round a product just above an integer down
 * to that integer and select the preceding sample.  Decode q as an integer over
 * a power of two and evaluate floor((q*N numerator - 1) / denominator) with a
 * portable two-word product instead.
 */
std::size_t empiricalQuantileOrdinal(double quantile, std::size_t population) noexcept
{
  if (!(quantile > 0.0))
    return 0;
  if (quantile >= 1.0)
    return population - 1;

  const std::uint64_t bits = doubleBits(quantile);
  const unsigned exponent  = static_cast<unsigned>((bits >> 52) & UINT64_C(0x7ff));
  const std::uint64_t significand =
      exponent == 0 ? bits & fraction_mask
                    : (UINT64_C(1) << 52) | (bits & fraction_mask);
  const unsigned denominator_shift = exponent == 0 ? 1074 : 1075 - exponent;

  UInt128Product numerator = multiplyWide(static_cast<std::uint64_t>(population),
                                          significand);
  if (numerator.low == 0)
  {
    --numerator.high;
    numerator.low = std::numeric_limits<std::uint64_t>::max();
  }
  else
    --numerator.low;
  return static_cast<std::size_t>(shiftWideRight(numerator, denominator_shift));
}

/// Agree that a derived local scalar is safe before entering its sum collective.
void requireFiniteResidualSum(double local_sum, Communicate* communicator)
{
  const std::array<std::uint64_t, 2> local_record{
      isFiniteBits(local_sum) ? static_cast<std::uint64_t>(ConsensusReason::NONE)
                               : static_cast<std::uint64_t>(ConsensusReason::RESIDUAL_OVERFLOW),
      clipping_protocol_version};
  const auto records = gatherRecords(communicator, local_record);
  for (std::size_t rank = 0; rank < records.size(); ++rank)
    if (records[rank][0] != static_cast<std::uint64_t>(ConsensusReason::NONE))
      throwConsensusFailure("residual sum", rank,
                            static_cast<ConsensusReason>(records[rank][0]));
}

/** Reject a non-finite residual before either scale rule consumes it.
 *
 * The quantile path can otherwise hide an overflowing residual when the selected
 * ordinal lies below it. All participants execute this fixed-record consensus so
 * no rank can enter the following radix or sum collective alone.
 */
template<typename Accessor>
void requireFiniteResiduals(std::size_t local_count,
                            Accessor&& accessor,
                            Communicate* communicator)
{
  ConsensusReason reason = ConsensusReason::NONE;
  for (std::size_t index = 0; index < local_count; ++index)
    if (!isFiniteBits(accessor(index)))
    {
      reason = ConsensusReason::RESIDUAL_OVERFLOW;
      break;
    }

  const std::array<std::uint64_t, 2> local_record{
      static_cast<std::uint64_t>(reason), clipping_protocol_version};
  const auto records = gatherRecords(communicator, local_record);
  for (std::size_t rank = 0; rank < records.size(); ++rank)
    if (records[rank][0] != static_cast<std::uint64_t>(ConsensusReason::NONE))
      throwConsensusFailure("residual construction", rank,
                            static_cast<ConsensusReason>(records[rank][0]));
}

/// Extend a stable FNV-1a descriptor identity with one fixed-width field.
void extendIdentity(std::uint64_t& identity, std::uint64_t field) noexcept
{
  for (unsigned byte = 0; byte < 8; ++byte)
  {
    identity ^= (field >> (8 * byte)) & UINT64_C(0xff);
    identity *= UINT64_C(1099511628211);
  }
}

/// Compute a stable identity after every semantic descriptor field is final.
std::uint64_t descriptorIdentity(const EnergyClippingDescriptor& descriptor) noexcept
{
  std::uint64_t identity = UINT64_C(14695981039346656037);
  extendIdentity(identity, clipping_protocol_version);
  extendIdentity(identity, static_cast<std::uint64_t>(descriptor.policy.scale_rule));
  extendIdentity(identity, doubleBits(descriptor.policy.width_multiplier));
  extendIdentity(identity, doubleBits(descriptor.policy.residual_quantile));
  extendIdentity(identity, descriptor.population);
  extendIdentity(identity, doubleBits(descriptor.center));
  extendIdentity(identity, doubleBits(descriptor.scale));
  extendIdentity(identity, doubleBits(descriptor.width));
  return identity;
}

} // namespace

bool EnergyClippingDescriptor::equivalent(const EnergyClippingDescriptor& other) const noexcept
{
  return policy.scale_rule == other.policy.scale_rule &&
      doubleBits(policy.width_multiplier) == doubleBits(other.policy.width_multiplier) &&
      doubleBits(policy.residual_quantile) == doubleBits(other.policy.residual_quantile) &&
      population == other.population && doubleBits(center) == doubleBits(other.center) &&
      doubleBits(scale) == doubleBits(other.scale) && doubleBits(width) == doubleBits(other.width) &&
      identity == other.identity;
}

EnergyClippingTransform::EnergyClippingTransform(EnergyClippingDescriptor descriptor)
    : descriptor_(std::move(descriptor))
{
  if (descriptor_.population == 0 || !isFiniteBits(descriptor_.center) ||
      !isFiniteBits(descriptor_.scale) || !isFiniteBits(descriptor_.width) ||
      descriptor_.scale < 0.0 || descriptor_.width < 0.0 ||
      descriptor_.identity != descriptorIdentity(descriptor_))
    throw std::invalid_argument("Invalid energy clipping transform descriptor");
}

DerivativeValue EnergyClippingTransform::apply(DerivativeValue energy) const noexcept
{
  const double lower = descriptor_.center - descriptor_.width;
  const double upper = descriptor_.center + descriptor_.width;
  return {std::clamp(std::real(energy), lower, upper), std::imag(energy)};
}

bool EnergyClippingTransform::clips(DerivativeValue energy) const noexcept
{
  return std::real(energy) < descriptor_.center - descriptor_.width ||
      std::real(energy) > descriptor_.center + descriptor_.width;
}

EnergyClippingTransform prepareEnergyClippingTransform(
    DerivativeArrayView<const DerivativeValue> local_energies,
    const EnergyClippingPolicy& policy,
    Communicate* communicator)
{
  const std::size_t population = clippingPreflight(local_energies, policy, communicator);
  const auto real_energy = [&](std::size_t index) { return std::real(local_energies[index]); };

  const std::size_t upper_ordinal = population / 2;
  const double upper_median =
      selectOrderStatistic(local_energies.size(), upper_ordinal, real_energy, communicator);
  const double center = population % 2 == 0
      ? finiteMidpoint(selectOrderStatistic(local_energies.size(), upper_ordinal - 1,
                                            real_energy, communicator),
                       upper_median)
      : upper_median;

  const auto residual = [&](std::size_t index) {
    return std::abs(std::real(local_energies[index]) - center);
  };
  requireFiniteResiduals(local_energies.size(), residual, communicator);

  double scale = 0.0;
  if (policy.scale_rule == EnergyClippingScaleRule::MEAN_ABSOLUTE_DEVIATION)
  {
    for (std::size_t index = 0; index < local_energies.size(); ++index)
      scale += residual(index);
    requireFiniteResidualSum(scale, communicator);
    if (communicator)
      communicator->allreduce_in_place(&scale, 1);
    scale /= static_cast<double>(population);
  }
  else
  {
    const std::size_t ordinal =
        empiricalQuantileOrdinal(policy.residual_quantile, population);
    scale = selectOrderStatistic(local_energies.size(), ordinal, residual, communicator);
  }

  const double width = policy.width_multiplier * scale;
  if (!isFiniteBits(scale) || !isFiniteBits(width) ||
      !isFiniteBits(center - width) || !isFiniteBits(center + width))
    throw std::runtime_error("Distributed energy clipping produced a non-finite width");

  EnergyClippingDescriptor descriptor{policy, population, center, scale, width, 0};
  descriptor.identity = descriptorIdentity(descriptor);
  return EnergyClippingTransform(std::move(descriptor));
}

} // namespace qmcplusplus::wftrain
