//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_initialization.cpp
 * @brief External-data-free tests for fresh PsiFormer model initialization.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qmcplusplus::psiformer
{
namespace
{

/// Return the fixed v1 architecture with the all-electron LiH spin populations.
ModelShape lithiumHydrideShape()
{
  return {/*spin_up_electrons=*/2,
          /*spin_down_electrons=*/2,
          /*nuclei=*/2,
          /*determinants=*/16,
          /*feature_dimension=*/256,
          /*attention_heads=*/4,
          /*attention_blocks=*/4};
}

/// Resolve one tensor diagnostic by typed role and optional attention block.
const TensorInitializationDiagnostic& diagnostic(const InitializationDiagnostics& diagnostics,
                                                 ParameterRole role,
                                                 std::size_t block = NO_ATTENTION_BLOCK)
{
  const auto match = std::find_if(diagnostics.tensors.begin(), diagnostics.tensors.end(),
                                  [role, block](const TensorInitializationDiagnostic& candidate) {
                                    return candidate.role == role && candidate.attention_block == block;
                                  });
  if (match == diagnostics.tensors.end())
    throw std::runtime_error("Missing initialization diagnostic");
  return *match;
}

/// Check that every value in one typed flat interval equals a constant exactly.
void checkConstant(const InitializedPsiFormerParameters& initialized,
                   const PsiFormerExecutionPlan& plan,
                   ParameterRole role,
                   double expected)
{
  const ParameterTensorDescriptor& tensor = plan.parameter(role);
  for (std::size_t index = tensor.begin; index < tensor.end; ++index)
    REQUIRE(initialized.values[index] == expected);
}

} // namespace

TEST_CASE("PsiFormer v1 initialization constructs the canonical DeepQMC layout",
          "[wavefunction][psiformer][initialization]")
{
  const InitializedPsiFormerParameters initialized =
      initializePsiFormerParameters(lithiumHydrideShape(), 17);
  const PsiFormerExecutionPlan plan(initialized.model_shape, initialized.layouts);

  CHECK(initialized.profile == DEEPQMC_PSIFORMER_V1);
  CHECK(initialized.seed == 17);
  REQUIRE(initialized.layouts.size() == 41);
  CHECK(initialized.values.size() == 1610498);
  CHECK(plan.parameterCount() == initialized.values.size());
  CHECK(plan.parameterTensors().size() == initialized.layouts.size());

  // Portable parameter exports use lexical (module, leaf-name) order and
  // contiguous row-major intervals.
  std::size_t expected_begin = 0;
  for (std::size_t tensor = 0; tensor < initialized.layouts.size(); ++tensor)
  {
    const ParameterLayoutInput& layout = initialized.layouts[tensor];
    CHECK(layout.begin == expected_begin);
    CHECK(layout.end >= layout.begin);
    expected_begin = layout.end;
    if (tensor > 0)
    {
      const ParameterLayoutInput& previous = initialized.layouts[tensor - 1];
      CHECK((previous.module < layout.module ||
             (previous.module == layout.module && previous.name < layout.name)));
    }
  }
  CHECK(expected_begin == initialized.values.size());

  CHECK(plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape ==
        std::vector<std::size_t>{9, 256});
  CHECK(plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT).shape ==
        std::vector<std::size_t>{256, 64});
  CHECK(plan.parameter(ParameterRole::ENVELOPE_PI_DOWN).shape ==
        std::vector<std::size_t>{64, 2});
  CHECK(plan.parameter(ParameterRole::ATTENTION_QUERY_WEIGHT, 3).shape ==
        std::vector<std::size_t>{256, 256});

  CHECK(initialized.diagnostics.tensor_count == 41);
  CHECK(initialized.diagnostics.parameter_count == initialized.values.size());
  CHECK(initialized.diagnostics.random_generator == PSIFORMER_INITIALIZATION_RNG_V1);
  CHECK(initialized.diagnostics.constant_parameter_count == 514);
  CHECK(initialized.diagnostics.random_parameter_count == 1609984);
  CHECK(initializationLawName(InitializationLaw::CONSTANT) == std::string("constant"));
  CHECK(initializationLawName(InitializationLaw::NORMAL) == std::string("normal"));
  CHECK(initializationLawName(InitializationLaw::TRUNCATED_NORMAL) == std::string("truncated_normal"));
}

TEST_CASE("PsiFormer v1 initialization is reproducible and seed-sensitive",
          "[wavefunction][psiformer][initialization]")
{
  const InitializedPsiFormerParameters first =
      initializePsiFormerParameters(lithiumHydrideShape(), 0x123456789ABCDEF0ULL);
  const InitializedPsiFormerParameters repeated =
      initializePsiFormerParameters(lithiumHydrideShape(), 0x123456789ABCDEF0ULL);
  REQUIRE(first.layouts.size() == repeated.layouts.size());
  REQUIRE(first.values == repeated.values);

  const InitializedPsiFormerParameters changed =
      initializePsiFormerParameters(lithiumHydrideShape(), 0x123456789ABCDEF1ULL);
  const PsiFormerExecutionPlan plan(first.model_shape, first.layouts);
  const ParameterTensorDescriptor& query = plan.parameter(ParameterRole::ATTENTION_QUERY_WEIGHT, 0);
  CHECK_FALSE(std::equal(first.values.begin() + query.begin, first.values.begin() + query.end,
                         changed.values.begin() + query.begin));

  // Seed selection changes random leaves only; analytic envelope and cusp
  // starting values are part of the profile rather than the random stream.
  const ParameterTensorDescriptor& pi = plan.parameter(ParameterRole::ENVELOPE_PI_UP);
  CHECK(std::equal(first.values.begin() + pi.begin, first.values.begin() + pi.end,
                   changed.values.begin() + pi.begin));

  // A small golden slice locks the versioned tensor-identity-to-stream mapping.
  // The tolerance admits last-bit differences in platform libm implementations
  // of the Box--Muller transcendental functions.
  const ParameterTensorDescriptor& embedding =
      plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
  const ParameterTensorDescriptor& golden_query =
      plan.parameter(ParameterRole::ATTENTION_QUERY_WEIGHT, 2);
  const ParameterTensorDescriptor& bias =
      plan.parameter(ParameterRole::UPDATE_OUTPUT_BIAS, 3);
  const ParameterTensorDescriptor& backflow =
      plan.parameter(ParameterRole::BACKFLOW_DOWN_WEIGHT);
  CHECK(first.values[embedding.begin] == Catch::Approx(0x1.8d596916b544ep-2).margin(1e-14));
  CHECK(first.values[embedding.begin + 117] == Catch::Approx(0x1.81bedf6ff0e92p-2).margin(1e-14));
  CHECK(first.values[golden_query.begin + 4097] == Catch::Approx(0x1.00a0e9530cacap-8).margin(1e-14));
  CHECK(first.values[bias.begin + 173] == Catch::Approx(0x1.6bcaf32c65745p-6).margin(1e-14));
  CHECK(first.values[backflow.begin + 8191] == Catch::Approx(-0x1.01ef5967e3bf6p-4).margin(1e-14));
}

TEST_CASE("PsiFormer v1 initialization omits an inactive same-spin cusp leaf",
          "[wavefunction][psiformer][initialization]")
{
  ModelShape two_electron_shape = lithiumHydrideShape();
  two_electron_shape.spin_up_electrons = 1;
  two_electron_shape.spin_down_electrons = 1;
  const InitializedPsiFormerParameters initialized =
      initializePsiFormerParameters(two_electron_shape, 29);
  const PsiFormerExecutionPlan plan(initialized.model_shape, initialized.layouts);

  CHECK(initialized.layouts.size() == 40);
  CHECK(initialized.values.size() == 1593857);
  CHECK(initialized.diagnostics.constant_parameter_count == 257);
  CHECK(initialized.diagnostics.random_parameter_count == 1593600);
  CHECK_THROWS_AS(plan.parameter(ParameterRole::CUSP_SAME_ALPHA), std::out_of_range);
  checkConstant(initialized, plan, ParameterRole::CUSP_OPPOSITE_ALPHA, 1);
}

TEST_CASE("PsiFormer v1 initialization matches DeepQMC role-specific scales",
          "[wavefunction][psiformer][initialization]")
{
  const InitializedPsiFormerParameters initialized =
      initializePsiFormerParameters(lithiumHydrideShape(), 3141592653589793ULL);
  const PsiFormerExecutionPlan plan(initialized.model_shape, initialized.layouts);

  checkConstant(initialized, plan, ParameterRole::ENVELOPE_PI_UP, 1);
  checkConstant(initialized, plan, ParameterRole::ENVELOPE_PI_DOWN, 1);
  checkConstant(initialized, plan, ParameterRole::ENVELOPE_ZETA_UP, 1);
  checkConstant(initialized, plan, ParameterRole::ENVELOPE_ZETA_DOWN, 1);
  checkConstant(initialized, plan, ParameterRole::CUSP_SAME_ALPHA, 1);
  checkConstant(initialized, plan, ParameterRole::CUSP_OPPOSITE_ALPHA, 1);

  const TensorInitializationDiagnostic& embedding =
      diagnostic(initialized.diagnostics, ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
  const double embedding_distribution_sigma = 1.0 / 3.0;
  const double embedding_expected_sigma = 0.87962566103423978 * embedding_distribution_sigma;
  CHECK(embedding.law == InitializationLaw::TRUNCATED_NORMAL);
  CHECK(embedding.distribution_standard_deviation == Catch::Approx(embedding_distribution_sigma));
  CHECK(embedding.expected_standard_deviation == Catch::Approx(embedding_expected_sigma));
  CHECK(std::abs(embedding.observed_mean) < 0.02);
  CHECK(embedding.observed_standard_deviation == Catch::Approx(embedding_expected_sigma).margin(0.015));
  CHECK(embedding.observed_minimum >= -2 * embedding_distribution_sigma);
  CHECK(embedding.observed_maximum <= 2 * embedding_distribution_sigma);

  const TensorInitializationDiagnostic& query =
      diagnostic(initialized.diagnostics, ParameterRole::ATTENTION_QUERY_WEIGHT, 0);
  CHECK(query.law == InitializationLaw::NORMAL);
  CHECK(query.expected_standard_deviation == Catch::Approx(0.0625));
  CHECK(std::abs(query.observed_mean) < 0.0015);
  CHECK(query.observed_standard_deviation == Catch::Approx(0.0625).margin(0.001));

  const TensorInitializationDiagnostic& bias =
      diagnostic(initialized.diagnostics, ParameterRole::UPDATE_HIDDEN_BIAS, 0);
  CHECK(bias.law == InitializationLaw::NORMAL);
  CHECK(bias.expected_standard_deviation == Catch::Approx(0.0625));
  CHECK(std::abs(bias.observed_mean) < 0.02);
  CHECK(bias.observed_standard_deviation == Catch::Approx(0.0625).margin(0.012));
}

TEST_CASE("PsiFormer v1 initialization rejects unsupported profiles and shapes",
          "[wavefunction][psiformer][initialization]")
{
  const ModelShape valid = lithiumHydrideShape();
  CHECK_THROWS_AS(initializePsiFormerParameters(valid, 0, "unversioned"), std::invalid_argument);

  ModelShape invalid = valid;
  invalid.spin_up_electrons = 0;
  CHECK_THROWS_AS(initializePsiFormerParameters(invalid, 0), std::invalid_argument);

  invalid = valid;
  invalid.spin_down_electrons = 0;
  CHECK_THROWS_AS(initializePsiFormerParameters(invalid, 0), std::invalid_argument);

  invalid = valid;
  invalid.nuclei = 0;
  CHECK_THROWS_AS(initializePsiFormerParameters(invalid, 0), std::invalid_argument);

  invalid = valid;
  invalid.determinants = 32;
  CHECK_THROWS_AS(initializePsiFormerParameters(invalid, 0), std::invalid_argument);

  invalid = valid;
  invalid.feature_dimension = 128;
  CHECK_THROWS_AS(initializePsiFormerParameters(invalid, 0), std::invalid_argument);

  invalid = valid;
  invalid.attention_heads = 8;
  CHECK_THROWS_AS(initializePsiFormerParameters(invalid, 0), std::invalid_argument);

  invalid = valid;
  invalid.attention_blocks = 2;
  CHECK_THROWS_AS(initializePsiFormerParameters(invalid, 0), std::invalid_argument);

  invalid = valid;
  invalid.spin_up_electrons = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(initializePsiFormerParameters(invalid, 0), std::overflow_error);

  invalid = valid;
  invalid.nuclei = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(initializePsiFormerParameters(invalid, 0), std::overflow_error);
}

} // namespace qmcplusplus::psiformer
