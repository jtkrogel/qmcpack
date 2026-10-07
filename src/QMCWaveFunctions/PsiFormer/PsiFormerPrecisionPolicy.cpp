//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerPrecisionPolicy.cpp
 * @brief Validated operation-specific precision policies for PsiFormer.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerPrecisionPolicy.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace qmcplusplus::psiformer
{
namespace
{

/// Multiply a byte extent without allowing silent size_t wraparound.
std::size_t checkedBytes(std::size_t count, std::size_t element_bytes, const char* description)
{
  if (element_bytes != 0 && count > std::numeric_limits<std::size_t>::max() / element_bytes)
    throw std::overflow_error(std::string("PsiFormer precision storage overflow for ") + description);
  return count * element_bytes;
}

/// Add two byte extents without allowing silent size_t wraparound.
std::size_t checkedAdd(std::size_t lhs, std::size_t rhs)
{
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs)
    throw std::overflow_error("PsiFormer precision storage total overflow");
  return lhs + rhs;
}

/// Return whether an operation is one of the approved lower-precision dense classes.
bool lowerPrecisionDenseOperation(PsiFormerArithmeticOperation operation) noexcept
{
  switch (operation)
  {
  case PsiFormerArithmeticOperation::DENSE_PROJECTION:
  case PsiFormerArithmeticOperation::ATTENTION_LOGITS:
  case PsiFormerArithmeticOperation::RESIDUAL_NONLINEAR:
  case PsiFormerArithmeticOperation::ORBITAL_CONSTRUCTION:
  case PsiFormerArithmeticOperation::PARAMETER_REVERSE_DENSE:
    return true;
  default:
    return false;
  }
}

/// Extend a stable FNV-1a fingerprint with one byte interval.
template<class T>
void extendFingerprint(std::uint64_t& fingerprint, const T& value) noexcept
{
  const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
  for (std::size_t byte = 0; byte < sizeof(T); ++byte)
  {
    fingerprint ^= bytes[byte];
    fingerprint *= UINT64_C(1099511628211);
  }
}

} // namespace

std::size_t PsiFormerPrecisionStorageRequirements::totalDeviceBytes() const
{
  return checkedAdd(checkedAdd(master_parameter_bytes, compute_parameter_bytes),
                    cast_workspace_bytes);
}

bool PsiFormerNumericalDiagnostics::requiresFullPrecisionRetry() const noexcept
{
  return nonfinite_count != 0 || invalid_softmax_count != 0 ||
      small_determinant_pivot_count != 0 || severe_cancellation_count != 0 ||
      extreme_ecp_ratio_count != 0;
}

void PsiFormerNumericalDiagnostics::merge(const PsiFormerNumericalDiagnostics& other) noexcept
{
  nonfinite_count += other.nonfinite_count;
  invalid_softmax_count += other.invalid_softmax_count;
  small_determinant_pivot_count += other.small_determinant_pivot_count;
  severe_cancellation_count += other.severe_cancellation_count;
  extreme_ecp_ratio_count += other.extreme_ecp_ratio_count;
  maximum_absolute_laplacian = std::max(maximum_absolute_laplacian,
                                        other.maximum_absolute_laplacian);
  maximum_gradient_norm = std::max(maximum_gradient_norm, other.maximum_gradient_norm);
  maximum_update_norm   = std::max(maximum_update_norm, other.maximum_update_norm);
}

const char* psiFormerPrecisionPolicyName(PsiFormerPrecisionPolicy policy) noexcept
{
  switch (policy)
  {
  case PsiFormerPrecisionPolicy::FP64_REFERENCE:
    return "fp64_reference";
  case PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE:
    return "fp32_compute_fp64_reduce";
  case PsiFormerPrecisionPolicy::TF32_DENSE_FP64_SENSITIVE:
    return "tf32_dense_fp64_sensitive";
  }
  return "unknown";
}

PsiFormerPrecisionPolicy parsePsiFormerPrecisionPolicy(std::string_view value)
{
  if (value == "fp64_reference")
    return PsiFormerPrecisionPolicy::FP64_REFERENCE;
  if (value == "fp32_compute_fp64_reduce")
    return PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE;
  if (value == "tf32_dense_fp64_sensitive")
    return PsiFormerPrecisionPolicy::TF32_DENSE_FP64_SENSITIVE;
  throw std::invalid_argument(
      "PsiFormer precision must be fp64_reference, fp32_compute_fp64_reduce, or tf32_dense_fp64_sensitive");
}

PsiFormerPrecisionRule psiFormerPrecisionRule(PsiFormerPrecisionPolicy policy,
                                              PsiFormerArithmeticOperation operation)
{
  if (operation == PsiFormerArithmeticOperation::COUNT)
    throw std::invalid_argument("PsiFormer precision rule requires a concrete operation");

  if (policy == PsiFormerPrecisionPolicy::FP64_REFERENCE)
    return {};

  if (lowerPrecisionDenseOperation(operation))
  {
    const PsiFormerArithmeticPrecision compute =
        policy == PsiFormerPrecisionPolicy::TF32_DENSE_FP64_SENSITIVE
        ? PsiFormerArithmeticPrecision::TENSOR_FLOAT32
        : PsiFormerArithmeticPrecision::BINARY32;
    return {/*storage=*/PsiFormerArithmeticPrecision::BINARY32,
            /*compute=*/compute,
            /*accumulation=*/PsiFormerArithmeticPrecision::BINARY32,
            /*numerically_sensitive=*/false,
            /*allow_fast_math=*/false};
  }

  // Softmax consumes lower-precision logits but performs its stability-critical
  // maximum and normalization reductions in binary64.
  if (operation == PsiFormerArithmeticOperation::SOFTMAX_NORMALIZATION)
    return {/*storage=*/PsiFormerArithmeticPrecision::BINARY32,
            /*compute=*/PsiFormerArithmeticPrecision::BINARY32,
            /*accumulation=*/PsiFormerArithmeticPrecision::BINARY64,
            /*numerically_sensitive=*/true,
            /*allow_fast_math=*/false};

  // Geometry, cusp/envelope, determinants, spatial traces, Hamiltonian terms,
  // canonical derivative accumulators, optimizer state, and collectives retain
  // binary64 storage and arithmetic in the first mixed policies.
  return {};
}

void validatePsiFormerPrecisionBackend(PsiFormerPrecisionPolicy policy,
                                       PsiFormerAcceleratorBackend backend,
                                       bool tf32_hardware_supported)
{
  if (policy != PsiFormerPrecisionPolicy::FP64_REFERENCE &&
      backend == PsiFormerAcceleratorBackend::CPU)
    throw std::invalid_argument("PsiFormer mixed precision requires an accelerator backend");
  if (policy == PsiFormerPrecisionPolicy::TF32_DENSE_FP64_SENSITIVE)
  {
    if (backend != PsiFormerAcceleratorBackend::CUDA)
      throw std::invalid_argument("PsiFormer TF32 policy requires the CUDA backend");
    if (!tf32_hardware_supported)
      throw std::invalid_argument("PsiFormer TF32 policy requires explicit hardware support");
  }
}

PsiFormerPrecisionStorageRequirements makePsiFormerPrecisionStorageRequirements(
    PsiFormerPrecisionPolicy policy,
    std::size_t parameter_count,
    std::size_t cast_tile_parameters)
{
  PsiFormerPrecisionStorageRequirements storage;
  storage.master_parameter_bytes = checkedBytes(parameter_count, sizeof(double), "FP64 master parameters");
  if (policy == PsiFormerPrecisionPolicy::FP64_REFERENCE)
  {
    if (cast_tile_parameters != 0)
      throw std::invalid_argument("PsiFormer FP64 policy does not require a parameter cast tile");
    return storage;
  }

  if (parameter_count != 0 && cast_tile_parameters == 0)
    throw std::invalid_argument("PsiFormer mixed precision requires a positive bounded cast tile");
  if (cast_tile_parameters > parameter_count)
    throw std::invalid_argument("PsiFormer parameter cast tile exceeds the canonical parameter count");
  storage.compute_parameter_bytes = checkedBytes(parameter_count, sizeof(float), "FP32 compute parameters");
  storage.cast_workspace_bytes = checkedBytes(cast_tile_parameters, sizeof(float), "parameter cast tile");
  storage.totalDeviceBytes();
  return storage;
}

std::uint64_t psiFormerPrecisionPolicyFingerprint(PsiFormerPrecisionPolicy policy) noexcept
{
  std::uint64_t fingerprint = UINT64_C(14695981039346656037);
  extendFingerprint(fingerprint, policy);
  for (std::uint8_t index = 0;
       index < static_cast<std::uint8_t>(PsiFormerArithmeticOperation::COUNT);
       ++index)
  {
    const PsiFormerPrecisionRule rule =
        psiFormerPrecisionRule(policy, static_cast<PsiFormerArithmeticOperation>(index));
    extendFingerprint(fingerprint, rule.storage);
    extendFingerprint(fingerprint, rule.compute);
    extendFingerprint(fingerprint, rule.accumulation);
    extendFingerprint(fingerprint, rule.numerically_sensitive);
    extendFingerprint(fingerprint, rule.allow_fast_math);
  }
  return fingerprint;
}

} // namespace qmcplusplus::psiformer
