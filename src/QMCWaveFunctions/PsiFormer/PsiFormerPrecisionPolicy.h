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
#include <type_traits>

namespace qmcplusplus::psiformer
{

/// Select one complete, reproducible PsiFormer accelerator precision policy.
enum class PsiFormerPrecisionPolicy : std::uint8_t
{
  FP64_REFERENCE,
  FP32_COMPUTE_FP64_REDUCE,
  TF32_DENSE_FP64_SENSITIVE,
  FP32_DENSE_FP64_ISLANDS
};

/** Frozen arithmetic identity for newly supported mixed score training/restart.
 * Earlier mixed inference had no admitted training/restart route. Native mixed
 * training fingerprints must include this revision; the public label is stable.
 */
inline constexpr const char* PSIFORMER_MIXED_AFFINE_TRAINING_ARITHMETIC_V1 =
    "fp32_dense_fp64_islands_affine_ad_v1";

/// Distinguish canonical FP64 scalar/head training from the legacy mixed-head arithmetic.
inline constexpr const char* PSIFORMER_MIXED_AFFINE_TRAINING_ARITHMETIC_V2 =
    "fp32_dense_fp64_islands_canonical_head_affine_ad_v2";

/// Bind exact mixed restarts to inverse-certified canonical-head admission.
inline constexpr const char* PSIFORMER_MIXED_AFFINE_TRAINING_ARITHMETIC_V3 =
    "fp32_dense_fp64_islands_certified_canonical_head_affine_ad_v3";

/** Scope one native mixed training epoch without changing the public precision label.
 * Ordinary inference retains its existing local replay policy. Training either
 * propagates mixed rejection or selects complete same-image FP64 replay.
 */
enum class PsiFormerGpuTrainingEpochMode : std::uint8_t
{
  INFERENCE,
  STRICT_MIXED,
  FP64_REPLAY
};

/// Explicit numerical-owner authority; the existing REAL attachment stays the default.
enum class MixedTrunkHeadPolicy : std::uint8_t
{
  REAL_AMPLITUDE,
  PAIRED_COMPLEX_FP64_HEAD
};

/// The paired numerical adapter changes real affine projections, never the FP64 head.
inline bool supportsMixedPairedTrunkEnvironment(const ExecutionEnvironment& environment)
{
  return environment.fixed_nuclei &&
      environment.parameter_scalar_domain == ScalarDomain::REAL &&
      environment.compute_scalar_domain == ScalarDomain::REAL &&
      environment.amplitude_scalar_domain == ScalarDomain::COMPLEX &&
      environment.head_representation == HeadRepresentation::PAIRED_REAL_COMPLEX &&
      environment.boundary == BoundaryCondition::PERIODIC && !environment.periodic_profile &&
      environment.geometry_feature_policy == GeometryFeaturePolicy::PERIODIC_TORUS_V1 &&
      environment.periodic_axes[0] && environment.periodic_axes[1] && environment.periodic_axes[2];
}

/// Admit the existing real affine adapter only on its qualified geometry families.
inline bool supportsMixedRealInferenceEnvironment(const ExecutionEnvironment& environment)
{
  if (!environment.fixed_nuclei || environment.parameter_scalar_domain != ScalarDomain::REAL ||
      environment.compute_scalar_domain != ScalarDomain::REAL ||
      environment.amplitude_scalar_domain != ScalarDomain::REAL ||
      environment.head_representation != HeadRepresentation::REAL)
    return false;
  if (environment.boundary == BoundaryCondition::OPEN)
    return true;
  return environment.boundary == BoundaryCondition::PERIODIC &&
      environment.geometry_feature_policy == GeometryFeaturePolicy::PERIODIC_TORUS_V1 &&
      !environment.periodic_profile && environment.periodic_axes[0] && environment.periodic_axes[1] &&
      environment.periodic_axes[2];
}

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

/// Select the backend multiplication mode independently of scalar storage.
enum class PsiFormerBackendMathMode : std::uint8_t
{
  FP64_STRICT,
  FP32_STRICT,
  CUDA_TF32
};

namespace detail
{

/// Compile-time classification of the value operations approved for FP32 storage.
template<PsiFormerArithmeticOperation Operation>
inline constexpr bool lower_precision_value_operation_v =
    Operation == PsiFormerArithmeticOperation::DENSE_PROJECTION ||
    Operation == PsiFormerArithmeticOperation::ATTENTION_LOGITS ||
    Operation == PsiFormerArithmeticOperation::RESIDUAL_NONLINEAR ||
    Operation == PsiFormerArithmeticOperation::PARAMETER_REVERSE_DENSE;

} // namespace detail

/** Expose the C++ scalar types used by one pre-instantiated policy/operation pair.
 *
 * TF32 affects only the vendor multiply mode, so its C++ input, product, and output
 * scalar types remain float.  Sensitive reductions and orbital assembly remain
 * double under every policy.
 */
template<PsiFormerPrecisionPolicy Policy, PsiFormerArithmeticOperation Operation>
struct PsiFormerPrecisionTraits
{
  static constexpr bool lower_value =
      Policy == PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS
      ? (Operation == PsiFormerArithmeticOperation::DENSE_PROJECTION ||
         Operation == PsiFormerArithmeticOperation::PARAMETER_REVERSE_DENSE)
      : Policy != PsiFormerPrecisionPolicy::FP64_REFERENCE &&
      (detail::lower_precision_value_operation_v<Operation> ||
       Operation == PsiFormerArithmeticOperation::SOFTMAX_NORMALIZATION);
  static constexpr bool fp64_accumulation =
      !lower_value || Operation == PsiFormerArithmeticOperation::SOFTMAX_NORMALIZATION;

  using storage_type      = std::conditional_t<lower_value, float, double>;
  using input_type        = storage_type;
  using product_type      = storage_type;
  using accumulation_type = std::conditional_t<fp64_accumulation, double, float>;
  using output_type       = storage_type;
};

static_assert(std::is_same_v<typename PsiFormerPrecisionTraits<
                                 PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
                                 PsiFormerArithmeticOperation::ORBITAL_CONSTRUCTION>::storage_type,
                             double>);
static_assert(std::is_same_v<typename PsiFormerPrecisionTraits<
                                 PsiFormerPrecisionPolicy::TF32_DENSE_FP64_SENSITIVE,
                                 PsiFormerArithmeticOperation::DENSE_PROJECTION>::product_type,
                             float>);
static_assert(std::is_same_v<typename PsiFormerPrecisionTraits<
                                 PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
                                 PsiFormerArithmeticOperation::SOFTMAX_NORMALIZATION>::accumulation_type,
                             double>);
static_assert(std::is_same_v<typename PsiFormerPrecisionTraits<
                                 PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
                                 PsiFormerArithmeticOperation::GEOMETRY_FEATURES>::output_type,
                             double>);
static_assert(std::is_same_v<typename PsiFormerPrecisionTraits<
                                 PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
                                 PsiFormerArithmeticOperation::DETERMINANT_SOLVE>::accumulation_type,
                             double>);
static_assert(std::is_same_v<typename PsiFormerPrecisionTraits<
                                 PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
                                 PsiFormerArithmeticOperation::SPATIAL_JETS>::storage_type,
                             double>);
static_assert(std::is_same_v<typename PsiFormerPrecisionTraits<
                                 PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
                                 PsiFormerArithmeticOperation::LOCAL_ENERGY_REDUCTION>::output_type,
                             double>);
static_assert(std::is_same_v<typename PsiFormerPrecisionTraits<
                                 PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
                                 PsiFormerArithmeticOperation::PARAMETER_ACCUMULATION>::storage_type,
                             double>);
static_assert(std::is_same_v<typename PsiFormerPrecisionTraits<
                                 PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
                                 PsiFormerArithmeticOperation::MASTER_PARAMETERS>::storage_type,
                             double>);
static_assert(std::is_same_v<typename PsiFormerPrecisionTraits<
                                 PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
                                 PsiFormerArithmeticOperation::OPTIMIZER_STATE>::storage_type,
                             double>);
static_assert(std::is_same_v<typename PsiFormerPrecisionTraits<
                                 PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS,
                                 PsiFormerArithmeticOperation::DENSE_PROJECTION>::storage_type,
                             float>);
static_assert(std::is_same_v<typename PsiFormerPrecisionTraits<
                                 PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS,
                                 PsiFormerArithmeticOperation::SOFTMAX_NORMALIZATION>::storage_type,
                             double>);
static_assert(std::is_same_v<typename PsiFormerPrecisionTraits<
                                 PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS,
                                 PsiFormerArithmeticOperation::RESIDUAL_NONLINEAR>::storage_type,
                             double>);
static_assert(std::is_same_v<typename PsiFormerPrecisionTraits<
                                 PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS,
                                 PsiFormerArithmeticOperation::PARAMETER_REVERSE_DENSE>::storage_type,
                             float>);

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
  /// Number of configurations rejected by the mixed FULL sensitivity screen.
  std::uint64_t ill_conditioned_spatial_count = 0;

  /// Return whether mixed execution must retry the complete batch in FP64.
  bool requiresFullPrecisionRetry() const noexcept;

  /// Merge one independent tile/crowd diagnostic with checked maxima semantics.
  void merge(const PsiFormerNumericalDiagnostics& other) noexcept;
};

/// Return the stable input/checkpoint spelling for one named policy.
const char* psiFormerPrecisionPolicyName(PsiFormerPrecisionPolicy policy) noexcept;

/// Return the stable diagnostic spelling for a contained backend math mode.
const char* psiFormerBackendMathModeName(PsiFormerBackendMathMode mode) noexcept;

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
