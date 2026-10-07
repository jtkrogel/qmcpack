//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerPrecisionPolicy.h
 * @brief Explicit arithmetic and storage policies for PsiFormer accelerators.
 *
 * Precision choices are named and operation-specific.  In particular, this contract
 * prevents a backend default from silently enabling TF32 or reduced-precision
 * accumulation in determinant, Laplacian, local-energy, or optimization reductions.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_PRECISION_POLICY_H
#define QMCPLUSPLUS_PSIFORMER_PRECISION_POLICY_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorPlanning.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace qmcplusplus::psiformer
{

/// Select one complete, reproducible PsiFormer accelerator precision policy.
enum class PsiFormerPrecisionPolicy : std::uint8_t
{
  FP64_REFERENCE,
  FP32_COMPUTE_FP64_REDUCE,
  TF32_DENSE_FP64_SENSITIVE
};

/// Identify operation classes whose precision is selected independently.
enum class PsiFormerArithmeticOperation : std::uint8_t
{
  GEOMETRY_FEATURES,
  DENSE_PROJECTION,
  ATTENTION_LOGITS,
  SOFTMAX_NORMALIZATION,
  RESIDUAL_NONLINEAR,
  ENVELOPE_AND_CUSP,
  ORBITAL_CONSTRUCTION,
  DETERMINANT_SOLVE,
  DETERMINANT_REDUCTION,
  SPATIAL_JETS,
  LAPLACIAN_REDUCTION,
  LOCAL_ENERGY_REDUCTION,
  PARAMETER_REVERSE_DENSE,
  PARAMETER_ACCUMULATION,
  ECP_RATIO,
  ECP_REDUCTION,
  MASTER_PARAMETERS,
  OPTIMIZER_STATE,
  DISTRIBUTED_REDUCTION,
  COUNT
};

/// Describe a concrete binary arithmetic/storage format without C++ type coupling.
enum class PsiFormerArithmeticPrecision : std::uint8_t
{
  BINARY64,
  BINARY32,
  TENSOR_FLOAT32
};

/// Record storage, multiply, and reduction precision for one operation class.
struct PsiFormerPrecisionRule
{
  PsiFormerArithmeticPrecision storage = PsiFormerArithmeticPrecision::BINARY64;
  PsiFormerArithmeticPrecision compute = PsiFormerArithmeticPrecision::BINARY64;
  PsiFormerArithmeticPrecision accumulation = PsiFormerArithmeticPrecision::BINARY64;
  bool numerically_sensitive = true;
  bool allow_fast_math       = false;

  friend bool operator==(const PsiFormerPrecisionRule& lhs,
                         const PsiFormerPrecisionRule& rhs) noexcept
  {
    return lhs.storage == rhs.storage && lhs.compute == rhs.compute &&
        lhs.accumulation == rhs.accumulation &&
        lhs.numerically_sensitive == rhs.numerically_sensitive &&
        lhs.allow_fast_math == rhs.allow_fast_math;
  }
};

/// Exact persistent and conversion storage implied by one parameter compute copy.
struct PsiFormerPrecisionStorageRequirements
{
  std::size_t master_parameter_bytes  = 0;
  std::size_t compute_parameter_bytes = 0;
  std::size_t cast_workspace_bytes    = 0;

  /// Return the checked total device storage for this policy fragment.
  std::size_t totalDeviceBytes() const;
};

/** Compact device-reduced evidence used to decide a full-precision retry.
 *
 * Counters, rather than copied intermediates, keep normal execution diagnostics
 * bounded.  Maxima are descriptive and do not themselves trigger a retry unless the
 * operation-specific kernel has incremented the corresponding hazard counter.
 */
struct PsiFormerNumericalDiagnostics
{
  std::uint64_t nonfinite_count             = 0;
  std::uint64_t invalid_softmax_count        = 0;
  std::uint64_t small_determinant_pivot_count = 0;
  std::uint64_t severe_cancellation_count    = 0;
  std::uint64_t extreme_ecp_ratio_count      = 0;
  double maximum_absolute_laplacian          = 0.0;
  double maximum_gradient_norm               = 0.0;
  double maximum_update_norm                 = 0.0;

  /// Return whether mixed execution must retry the complete batch in FP64.
  bool requiresFullPrecisionRetry() const noexcept;

  /// Merge one independent tile/crowd diagnostic with checked maxima semantics.
  void merge(const PsiFormerNumericalDiagnostics& other) noexcept;
};

/// Return the stable input/checkpoint spelling for one named policy.
const char* psiFormerPrecisionPolicyName(PsiFormerPrecisionPolicy policy) noexcept;

/// Parse one complete policy name; aliases are deliberately not accepted.
PsiFormerPrecisionPolicy parsePsiFormerPrecisionPolicy(std::string_view value);

/// Return the explicit rule for one operation under one complete policy.
PsiFormerPrecisionRule psiFormerPrecisionRule(PsiFormerPrecisionPolicy policy,
                                              PsiFormerArithmeticOperation operation);

/// Validate that a policy/backend pair can honor its advertised arithmetic mode.
void validatePsiFormerPrecisionBackend(PsiFormerPrecisionPolicy policy,
                                       PsiFormerAcceleratorBackend backend,
                                       bool tf32_hardware_supported = false);

/// Return bytes required for the FP64 master, compute copy, and bounded cast tile.
PsiFormerPrecisionStorageRequirements makePsiFormerPrecisionStorageRequirements(
    PsiFormerPrecisionPolicy policy,
    std::size_t parameter_count,
    std::size_t cast_tile_parameters);

/// Return a stable policy/rule fingerprint for execution plans and checkpoints.
std::uint64_t psiFormerPrecisionPolicyFingerprint(PsiFormerPrecisionPolicy policy) noexcept;

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_PRECISION_POLICY_H
