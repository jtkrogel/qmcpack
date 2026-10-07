//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_precision_policy.cpp
 * @brief CPU-only tests for explicit PsiFormer accelerator precision contracts.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerPrecisionPolicy.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <limits>

namespace qmcplusplus::psiformer
{

TEST_CASE("PsiFormer precision policy names are strict and stable",
          "[wavefunction][psiformer][precision]")
{
  for (const PsiFormerPrecisionPolicy policy : {
           PsiFormerPrecisionPolicy::FP64_REFERENCE,
           PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
           PsiFormerPrecisionPolicy::TF32_DENSE_FP64_SENSITIVE})
  {
    CHECK(parsePsiFormerPrecisionPolicy(psiFormerPrecisionPolicyName(policy)) == policy);
    CHECK(psiFormerPrecisionPolicyFingerprint(policy) != 0);
  }
  CHECK(psiFormerPrecisionPolicyFingerprint(PsiFormerPrecisionPolicy::FP64_REFERENCE) !=
        psiFormerPrecisionPolicyFingerprint(PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE));
  CHECK_THROWS_AS(parsePsiFormerPrecisionPolicy("mixed"), std::invalid_argument);
  CHECK_THROWS_AS(parsePsiFormerPrecisionPolicy("tf32"), std::invalid_argument);
}

TEST_CASE("PsiFormer mixed policy lowers only audited operation classes",
          "[wavefunction][psiformer][precision]")
{
  const PsiFormerPrecisionRule dense = psiFormerPrecisionRule(
      PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
      PsiFormerArithmeticOperation::DENSE_PROJECTION);
  CHECK(dense.storage == PsiFormerArithmeticPrecision::BINARY32);
  CHECK(dense.compute == PsiFormerArithmeticPrecision::BINARY32);
  CHECK(dense.accumulation == PsiFormerArithmeticPrecision::BINARY32);
  CHECK_FALSE(dense.numerically_sensitive);
  CHECK_FALSE(dense.allow_fast_math);

  const PsiFormerPrecisionRule softmax = psiFormerPrecisionRule(
      PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
      PsiFormerArithmeticOperation::SOFTMAX_NORMALIZATION);
  CHECK(softmax.storage == PsiFormerArithmeticPrecision::BINARY32);
  CHECK(softmax.accumulation == PsiFormerArithmeticPrecision::BINARY64);
  CHECK(softmax.numerically_sensitive);

  for (const PsiFormerArithmeticOperation operation : {
           PsiFormerArithmeticOperation::GEOMETRY_FEATURES,
           PsiFormerArithmeticOperation::ENVELOPE_AND_CUSP,
           PsiFormerArithmeticOperation::DETERMINANT_SOLVE,
           PsiFormerArithmeticOperation::LAPLACIAN_REDUCTION,
           PsiFormerArithmeticOperation::LOCAL_ENERGY_REDUCTION,
           PsiFormerArithmeticOperation::PARAMETER_ACCUMULATION,
           PsiFormerArithmeticOperation::ECP_REDUCTION,
           PsiFormerArithmeticOperation::MASTER_PARAMETERS,
           PsiFormerArithmeticOperation::OPTIMIZER_STATE,
           PsiFormerArithmeticOperation::DISTRIBUTED_REDUCTION})
  {
    const PsiFormerPrecisionRule rule = psiFormerPrecisionRule(
        PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE, operation);
    CHECK(rule.storage == PsiFormerArithmeticPrecision::BINARY64);
    CHECK(rule.compute == PsiFormerArithmeticPrecision::BINARY64);
    CHECK(rule.accumulation == PsiFormerArithmeticPrecision::BINARY64);
    CHECK(rule.numerically_sensitive);
  }

  const PsiFormerPrecisionRule tf32 = psiFormerPrecisionRule(
      PsiFormerPrecisionPolicy::TF32_DENSE_FP64_SENSITIVE,
      PsiFormerArithmeticOperation::DENSE_PROJECTION);
  CHECK(tf32.compute == PsiFormerArithmeticPrecision::TENSOR_FLOAT32);
  CHECK_THROWS_AS(
      psiFormerPrecisionRule(PsiFormerPrecisionPolicy::FP64_REFERENCE,
                             PsiFormerArithmeticOperation::COUNT),
      std::invalid_argument);
}

TEST_CASE("PsiFormer precision backend validation never enables TF32 implicitly",
          "[wavefunction][psiformer][precision]")
{
  CHECK_NOTHROW(validatePsiFormerPrecisionBackend(
      PsiFormerPrecisionPolicy::FP64_REFERENCE, PsiFormerAcceleratorBackend::CPU));
  CHECK_THROWS_AS(validatePsiFormerPrecisionBackend(
                      PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
                      PsiFormerAcceleratorBackend::CPU),
                  std::invalid_argument);
  CHECK_NOTHROW(validatePsiFormerPrecisionBackend(
      PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
      PsiFormerAcceleratorBackend::CUDA));
  CHECK_THROWS_AS(validatePsiFormerPrecisionBackend(
                      PsiFormerPrecisionPolicy::TF32_DENSE_FP64_SENSITIVE,
                      PsiFormerAcceleratorBackend::SYCL, true),
                  std::invalid_argument);
  CHECK_THROWS_WITH(validatePsiFormerPrecisionBackend(
                        PsiFormerPrecisionPolicy::TF32_DENSE_FP64_SENSITIVE,
                        PsiFormerAcceleratorBackend::CUDA, false),
                    Catch::Matchers::ContainsSubstring("explicit hardware support"));
  CHECK_NOTHROW(validatePsiFormerPrecisionBackend(
      PsiFormerPrecisionPolicy::TF32_DENSE_FP64_SENSITIVE,
      PsiFormerAcceleratorBackend::CUDA, true));
}

TEST_CASE("PsiFormer precision storage is explicit and overflow checked",
          "[wavefunction][psiformer][precision]")
{
  const PsiFormerPrecisionStorageRequirements fp64 =
      makePsiFormerPrecisionStorageRequirements(
          PsiFormerPrecisionPolicy::FP64_REFERENCE, 100, 0);
  CHECK(fp64.master_parameter_bytes == 800);
  CHECK(fp64.compute_parameter_bytes == 0);
  CHECK(fp64.cast_workspace_bytes == 0);
  CHECK(fp64.totalDeviceBytes() == 800);

  const PsiFormerPrecisionStorageRequirements mixed =
      makePsiFormerPrecisionStorageRequirements(
          PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE, 100, 16);
  CHECK(mixed.master_parameter_bytes == 800);
  CHECK(mixed.compute_parameter_bytes == 400);
  CHECK(mixed.cast_workspace_bytes == 64);
  CHECK(mixed.totalDeviceBytes() == 1264);

  CHECK_THROWS_AS(makePsiFormerPrecisionStorageRequirements(
                      PsiFormerPrecisionPolicy::FP64_REFERENCE, 100, 1),
                  std::invalid_argument);
  CHECK_THROWS_AS(makePsiFormerPrecisionStorageRequirements(
                      PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE, 100, 0),
                  std::invalid_argument);
  CHECK_THROWS_AS(makePsiFormerPrecisionStorageRequirements(
                      PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE, 100, 101),
                  std::invalid_argument);
  CHECK_THROWS_AS(makePsiFormerPrecisionStorageRequirements(
                      PsiFormerPrecisionPolicy::FP64_REFERENCE,
                      std::numeric_limits<std::size_t>::max(), 0),
                  std::overflow_error);
}

TEST_CASE("PsiFormer numerical diagnostics request deterministic FP64 retry",
          "[wavefunction][psiformer][precision]")
{
  PsiFormerNumericalDiagnostics aggregate;
  aggregate.maximum_gradient_norm = 2.0;
  CHECK_FALSE(aggregate.requiresFullPrecisionRetry());

  PsiFormerNumericalDiagnostics tile;
  tile.invalid_softmax_count       = 1;
  tile.maximum_absolute_laplacian  = 17.0;
  tile.maximum_gradient_norm       = 1.0;
  tile.maximum_update_norm         = 0.25;
  aggregate.merge(tile);
  CHECK(aggregate.invalid_softmax_count == 1);
  CHECK(aggregate.maximum_absolute_laplacian == 17.0);
  CHECK(aggregate.maximum_gradient_norm == 2.0);
  CHECK(aggregate.maximum_update_norm == 0.25);
  CHECK(aggregate.requiresFullPrecisionRetry());

  aggregate.nonfinite_count = std::numeric_limits<std::uint64_t>::max();
  tile.nonfinite_count      = 1;
  aggregate.merge(tile);
  CHECK(aggregate.nonfinite_count == std::numeric_limits<std::uint64_t>::max());
}

} // namespace qmcplusplus::psiformer
