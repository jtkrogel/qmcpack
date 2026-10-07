//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerPrecisionExecution.cpp
 * @brief Checked mixed-precision storage, conversion, and publication planning.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerPrecisionExecution.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace qmcplusplus::psiformer
{
namespace
{

constexpr std::uint64_t FNV_OFFSET = UINT64_C(14695981039346656037);
constexpr std::uint64_t FNV_PRIME  = UINT64_C(1099511628211);

/// Mix one fixed-width integer into a stable little-endian FNV-1a hash.
void mixInteger(std::uint64_t& hash, std::uint64_t value) noexcept
{
  for (unsigned shift = 0; shift < 64; shift += 8)
  {
    hash ^= static_cast<std::uint8_t>((value >> shift) & UINT64_C(0xff));
    hash *= FNV_PRIME;
  }
}

/// Multiply a scalar count by its byte width without allowing wraparound.
std::size_t checkedBytes(std::size_t count, std::size_t element_bytes, const char* label)
{
  if (element_bytes != 0 && count > std::numeric_limits<std::size_t>::max() / element_bytes)
    throw std::overflow_error(std::string("PsiFormer precision execution overflow for ") + label);
  return count * element_bytes;
}

/// Add diagnostic counters without allowing a hazard to wrap back to zero.
std::uint64_t saturatingAdd(std::uint64_t lhs, std::uint64_t rhs) noexcept
{
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  return lhs > maximum - rhs ? maximum : lhs + rhs;
}

/// Validate the independent policy and vendor-math-mode identities.
void validateMathMode(PsiFormerPrecisionPolicy policy, PsiFormerBackendMathMode mode)
{
  if (policy == PsiFormerPrecisionPolicy::FP64_REFERENCE &&
      mode != PsiFormerBackendMathMode::FP64_STRICT)
    throw std::invalid_argument("PsiFormer FP64 policy requires the strict FP64 math mode");
  if (policy == PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE &&
      mode != PsiFormerBackendMathMode::FP32_STRICT)
    throw std::invalid_argument("PsiFormer FP32 policy requires the strict FP32 math mode");
  if (policy == PsiFormerPrecisionPolicy::TF32_DENSE_FP64_SENSITIVE &&
      mode != PsiFormerBackendMathMode::CUDA_TF32)
    throw std::invalid_argument("PsiFormer TF32 policy requires the explicit CUDA TF32 math mode");
}

/// Return the object representation of one IEEE binary64 value.
std::uint64_t doubleBits(double value) noexcept
{
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

/// Construct one IEEE binary32 value without host floating-point classification.
float floatFromBits(std::uint32_t bits) noexcept
{
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

/// Round a finite IEEE binary64 value to binary32 using ties-to-even integer logic.
std::uint32_t roundedFloatBits(std::uint64_t double_bits) noexcept
{
  const std::uint32_t sign = static_cast<std::uint32_t>(double_bits >> 63) << 31;
  const std::uint64_t exponent_bits = (double_bits >> 52) & UINT64_C(0x7ff);
  const std::uint64_t fraction = double_bits & UINT64_C(0x000fffffffffffff);
  if (exponent_bits == 0)
    return sign; // Every binary64 subnormal rounds below the binary32 range.

  int exponent = static_cast<int>(exponent_bits) - 1023;
  const std::uint64_t significand = (UINT64_C(1) << 52) | fraction;
  if (exponent >= -126)
  {
    constexpr unsigned shift = 29;
    std::uint64_t rounded = significand >> shift;
    const std::uint64_t remainder = significand & ((UINT64_C(1) << shift) - 1);
    const std::uint64_t halfway = UINT64_C(1) << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0))
      ++rounded;
    if (rounded == (UINT64_C(1) << 24))
    {
      rounded >>= 1;
      ++exponent;
    }
    return sign | (static_cast<std::uint32_t>(exponent + 127) << 23) |
        (static_cast<std::uint32_t>(rounded) & UINT32_C(0x007fffff));
  }

  // A binary32 subnormal is an integer multiple of 2^-149.
  const int shift = -exponent - 97;
  if (shift > 53)
    return sign;
  std::uint64_t rounded = significand >> shift;
  const std::uint64_t remainder = significand & ((UINT64_C(1) << shift) - 1);
  const std::uint64_t halfway = UINT64_C(1) << (shift - 1);
  if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0))
    ++rounded;
  return sign | static_cast<std::uint32_t>(rounded);
}

/// Reconstruct the exact binary32 value in binary64 without reading a subnormal float.
double floatBitsAsDouble(std::uint32_t bits) noexcept
{
  const double sign = (bits >> 31) == 0 ? 1.0 : -1.0;
  const std::uint32_t exponent = (bits >> 23) & UINT32_C(0xff);
  const std::uint32_t fraction = bits & UINT32_C(0x007fffff);
  if (exponent == 0)
    return sign * std::ldexp(static_cast<double>(fraction), -149);
  return sign * std::ldexp(static_cast<double>((UINT32_C(1) << 23) | fraction),
                           static_cast<int>(exponent) - 150);
}

} // namespace

void PsiFormerParameterConversionDiagnostics::merge(
    const PsiFormerParameterConversionDiagnostics& other) noexcept
{
  nonfinite_input_count = saturatingAdd(nonfinite_input_count, other.nonfinite_input_count);
  overflow_count = saturatingAdd(overflow_count, other.overflow_count);
  subnormal_output_count = saturatingAdd(subnormal_output_count, other.subnormal_output_count);
  underflow_to_zero_count = saturatingAdd(underflow_to_zero_count, other.underflow_to_zero_count);
  maximum_absolute_error = std::max(maximum_absolute_error, other.maximum_absolute_error);
  maximum_relative_error = std::max(maximum_relative_error, other.maximum_relative_error);
}

PsiFormerMixedPublicationState::PsiFormerMixedPublicationState(
    PsiFormerPrecisionPolicy policy,
    std::size_t active_version,
    std::uint64_t active_fingerprint,
    std::uint8_t active_slot)
    : policy_(policy),
      active_version_(active_version),
      active_fingerprint_(active_fingerprint),
      active_slot_(active_slot)
{
  if (active_fingerprint == 0)
    throw std::invalid_argument("PsiFormer active precision fingerprint must be nonzero");
  if (active_slot > 1)
    throw std::invalid_argument("PsiFormer precision publication slot must be zero or one");
}

void PsiFormerMixedPublicationState::begin(std::size_t source_version,
                                           std::uint64_t execution_fingerprint)
{
  if (pending_)
    throw std::logic_error("PsiFormer mixed parameter publication is already pending");
  if (source_version <= active_version_)
    throw std::invalid_argument(
        "PsiFormer mixed publication requires a strictly newer model version");
  if (execution_fingerprint == 0)
    throw std::invalid_argument("PsiFormer pending precision fingerprint must be nonzero");

  pending_ = Pending{source_version, execution_fingerprint,
                     static_cast<std::uint8_t>(1U - active_slot_), false,
                     policy_ == PsiFormerPrecisionPolicy::FP64_REFERENCE, false};
}

PsiFormerMixedPublicationState::Pending&
PsiFormerMixedPublicationState::matchingPending(std::size_t source_version)
{
  if (!pending_ || pending_->version != source_version)
    throw std::logic_error(
        "PsiFormer mixed publication milestone does not match the pending version");
  return *pending_;
}

void PsiFormerMixedPublicationState::markMasterCopied(std::size_t source_version)
{
  Pending& pending = matchingPending(source_version);
  if (pending.master_copied)
    throw std::logic_error("PsiFormer mixed publication master copy was already recorded");
  pending.master_copied = true;
}

void PsiFormerMixedPublicationState::markComputeConversionComplete(std::size_t source_version)
{
  Pending& pending = matchingPending(source_version);
  if (policy_ == PsiFormerPrecisionPolicy::FP64_REFERENCE)
    throw std::logic_error("PsiFormer FP64 publication has no compute conversion milestone");
  if (!pending.master_copied)
    throw std::logic_error("PsiFormer compute conversion completed before its master copy");
  if (pending.compute_converted)
    throw std::logic_error("PsiFormer compute conversion was already recorded");
  pending.compute_converted = true;
}

bool PsiFormerMixedPublicationState::acceptDiagnostics(
    std::size_t source_version,
    const PsiFormerParameterConversionDiagnostics& diagnostics)
{
  Pending& pending = matchingPending(source_version);
  if (!pending.master_copied || !pending.compute_converted)
    throw std::logic_error("PsiFormer conversion diagnostics arrived before conversion completion");
  if (pending.diagnostics_accepted)
    throw std::logic_error("PsiFormer conversion diagnostics were already accepted");
  if (diagnostics.hasPublicationHazard())
  {
    pending_.reset();
    return false;
  }
  pending.diagnostics_accepted = true;
  return true;
}

void PsiFormerMixedPublicationState::publish(std::size_t source_version)
{
  const Pending& pending = matchingPending(source_version);
  if (!pending.master_copied || !pending.compute_converted || !pending.diagnostics_accepted)
    throw std::logic_error("PsiFormer mixed publication is incomplete");

  active_version_     = pending.version;
  active_fingerprint_ = pending.fingerprint;
  active_slot_        = pending.slot;
  pending_.reset();
}

void PsiFormerMixedPublicationState::cancel(std::size_t source_version)
{
  matchingPending(source_version);
  pending_.reset();
}

std::optional<std::size_t> PsiFormerMixedPublicationState::pendingVersion() const noexcept
{
  return pending_ ? std::optional<std::size_t>(pending_->version) : std::nullopt;
}

std::optional<std::uint8_t> PsiFormerMixedPublicationState::pendingSlot() const noexcept
{
  return pending_ ? std::optional<std::uint8_t>(pending_->slot) : std::nullopt;
}

PsiFormerPrecisionExecutionPlan makePsiFormerPrecisionExecutionPlan(
    PsiFormerPrecisionPolicy policy,
    PsiFormerBackendMathMode math_mode,
    PsiFormerAcceleratorBackend backend,
    bool tf32_hardware_supported,
    std::size_t parameter_count,
    std::size_t cast_tile_parameters,
    std::size_t block_size,
    std::uint64_t device_layout_fingerprint,
    std::size_t source_version,
    bool full_precision_retry_available)
{
  validateMathMode(policy, math_mode);
  validatePsiFormerPrecisionBackend(policy, backend, tf32_hardware_supported);
  const PsiFormerBlasMathModePlan blas_math =
      makePsiFormerBlasMathModePlan(math_mode, backend);
  if (device_layout_fingerprint == 0)
    throw std::invalid_argument(
        "PsiFormer precision execution requires a nonzero device-layout fingerprint");
  if (block_size == 0)
    throw std::invalid_argument("PsiFormer precision conversion block size must be positive");

  const bool mixed = policy != PsiFormerPrecisionPolicy::FP64_REFERENCE;
  if (!mixed && cast_tile_parameters != 0)
    throw std::invalid_argument("PsiFormer FP64 execution does not require a conversion tile");
  if (mixed && parameter_count != 0 &&
      (cast_tile_parameters == 0 || cast_tile_parameters > parameter_count))
    throw std::invalid_argument(
        "PsiFormer mixed conversion tile must be within the parameter vector");
  if (mixed && parameter_count == 0 && cast_tile_parameters != 0)
    throw std::invalid_argument("PsiFormer empty mixed model requires an empty conversion tile");

  PsiFormerPrecisionExecutionPlan plan;
  plan.policy                       = policy;
  plan.math_mode                    = math_mode;
  plan.backend                      = backend;
  plan.blas_math                    = blas_math;
  plan.parameter_count              = parameter_count;
  plan.canonical_source_version     = source_version;
  plan.compute_copy_version         = source_version;
  plan.device_layout_fingerprint    = device_layout_fingerprint;
  plan.full_precision_retry_available = full_precision_retry_available;
  plan.master_slot_bytes = checkedBytes(parameter_count, sizeof(double), "FP64 master slot");
  plan.compute_slot_bytes = mixed
      ? checkedBytes(parameter_count, sizeof(float), "FP32 compute slot")
      : 0;
  plan.conversion_workspace_bytes = mixed
      ? checkedBytes(cast_tile_parameters, sizeof(float), "FP32 conversion tile")
      : 0;
  plan.diagnostic_bytes = mixed ? sizeof(PsiFormerDeviceNumericalDiagnostics) : 0;

  std::vector<PsiFormerDeviceArenaRequest> requests{
      {PsiFormerDeviceArenaRegion::MODEL_PARAMETERS, plan.master_slot_bytes, 256},
      {PsiFormerDeviceArenaRegion::MODEL_PARAMETERS_STAGING, plan.master_slot_bytes, 256}};
  if (mixed)
  {
    requests.push_back({PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_0,
                        plan.compute_slot_bytes, 256});
    requests.push_back({PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_1,
                        plan.compute_slot_bytes, 256});
    requests.push_back({PsiFormerDeviceArenaRegion::PRECISION_CONVERSION_WORKSPACE,
                        plan.conversion_workspace_bytes, 256});
    requests.push_back({PsiFormerDeviceArenaRegion::NUMERICAL_DIAGNOSTICS,
                        plan.diagnostic_bytes, alignof(PsiFormerDeviceNumericalDiagnostics)});
  }
  plan.arena = makePsiFormerDeviceArenaLayout(requests);

  const std::size_t schedule_capacity = mixed && parameter_count != 0
      ? cast_tile_parameters
      : 1;
  const PsiFormerTiledLaunchSchedule launches =
      makePsiFormerTiledLaunchSchedule(mixed ? parameter_count : 0,
                                       schedule_capacity, block_size);
  plan.conversion.parameter_count = mixed ? parameter_count : 0;
  plan.conversion.tile_capacity   = mixed ? cast_tile_parameters : 0;
  plan.conversion.block_size      = block_size;
  plan.conversion.tiles.reserve(launches.tiles.size());
  for (const PsiFormerLaunchTile& tile : launches.tiles)
    plan.conversion.tiles.push_back({tile.begin, tile.begin, tile.count, tile.launch});
  plan.conversion.fingerprint = launches.fingerprint;

  std::uint64_t hash = FNV_OFFSET;
  mixInteger(hash, UINT64_C(1));
  mixInteger(hash, psiFormerPrecisionPolicyFingerprint(policy));
  mixInteger(hash, static_cast<std::uint8_t>(math_mode));
  mixInteger(hash, static_cast<std::uint8_t>(backend));
  mixInteger(hash, static_cast<std::uint8_t>(blas_math.native));
  mixInteger(hash, blas_math.reduced_multiply ? 1 : 0);
  mixInteger(hash, parameter_count);
  mixInteger(hash, source_version);
  mixInteger(hash, device_layout_fingerprint);
  mixInteger(hash, plan.arena.fingerprint);
  mixInteger(hash, plan.conversion.fingerprint);
  mixInteger(hash, full_precision_retry_available ? 1 : 0);
  plan.fingerprint = hash == 0 ? 1 : hash;
  return plan;
}

void convertPsiFormerFp64ToFp32Reference(
    const double* source,
    std::size_t count,
    float* destination,
    PsiFormerParameterConversionDiagnostics& diagnostics)
{
  if (count != 0 && (source == nullptr || destination == nullptr))
    throw std::invalid_argument("PsiFormer reference conversion requires valid nonempty spans");

  const std::uint64_t fp32_max_bits =
      doubleBits(static_cast<double>(std::numeric_limits<float>::max()));
  for (std::size_t index = 0; index < count; ++index)
  {
    const double value = source[index];
    const std::uint64_t bits = doubleBits(value);
    const std::uint64_t magnitude_bits = bits & UINT64_C(0x7fffffffffffffff);
    const std::uint64_t exponent_bits = (magnitude_bits >> 52) & UINT64_C(0x7ff);
    if (exponent_bits == UINT64_C(0x7ff))
    {
      ++diagnostics.nonfinite_input_count;
      destination[index] = 0.0F;
      continue;
    }

    if (magnitude_bits > fp32_max_bits)
    {
      ++diagnostics.overflow_count;
      destination[index] = floatFromBits(
          (static_cast<std::uint32_t>(bits >> 32) & UINT32_C(0x80000000)) |
          UINT32_C(0x7f7fffff));
      continue;
    }

    const std::uint32_t converted_bits = roundedFloatBits(bits);
    const std::uint32_t converted_magnitude_bits = converted_bits & UINT32_C(0x7fffffff);
    destination[index] = floatFromBits(converted_bits);
    if (magnitude_bits != 0 && converted_magnitude_bits == 0)
      ++diagnostics.underflow_to_zero_count;
    else if (converted_magnitude_bits != 0 && converted_magnitude_bits < UINT32_C(0x00800000))
      ++diagnostics.subnormal_output_count;

    const double converted_value = floatBitsAsDouble(converted_bits);
    const double absolute_error = std::abs(converted_value - value);
    diagnostics.maximum_absolute_error =
        std::max(diagnostics.maximum_absolute_error, absolute_error);
    if (magnitude_bits != 0)
      diagnostics.maximum_relative_error =
          std::max(diagnostics.maximum_relative_error, absolute_error / std::abs(value));
  }
}

} // namespace qmcplusplus::psiformer
