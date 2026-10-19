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
                                                 std::size_t block = NO_ATTENTION_BLOCK,
                                                 HeadTensorComponent component = HeadTensorComponent::REAL)
{
  const auto match = std::find_if(diagnostics.tensors.begin(), diagnostics.tensors.end(),
                                  [role, block, component](const TensorInitializationDiagnostic& candidate) {
                                    return candidate.role == role && candidate.attention_block == block &&
                                        candidate.head_component == component;
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

/// Reproduce the native real parameter store's stable layout-only fingerprint.
std::uint64_t layoutFingerprint(const InitializedPsiFormerParameters& initialized)
{
  std::uint64_t hash = 14695981039346656037ULL;
  auto mix_byte = [&hash](std::uint8_t byte) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  };
  auto mix_integer = [&mix_byte](std::uint64_t value) {
    for (int byte = 0; byte < 8; ++byte)
      mix_byte(static_cast<std::uint8_t>(value >> (8 * byte)));
  };
  auto mix_string = [&mix_byte, &mix_integer](const std::string& value) {
    mix_integer(value.size());
    for (unsigned char character : value)
      mix_byte(character);
  };
  mix_integer(initialized.layouts.size());
  mix_integer(initialized.values.size());
  for (const auto& layout : initialized.layouts)
  {
    mix_string(layout.module);
    mix_string(layout.name);
    mix_integer(layout.shape.size());
    for (std::size_t extent : layout.shape)
      mix_integer(extent);
    mix_integer(layout.begin);
    mix_integer(layout.end);
  }
  return hash;
}

/// Require explicit complex periodic metadata while keeping parameter/trunk domains real.
ExecutionEnvironment pairedPeriodicEnvironment()
{
  ExecutionEnvironment environment;
  environment.boundary = BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
  environment.periodic_axes = {true, true, true};
  environment.lattice_vectors = {{{8, 0, 0}, {0, 8, 0}, {0, 0, 8}}};
  environment.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
  environment.amplitude_scalar_domain = ScalarDomain::COMPLEX;
  return environment;
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
  CHECK(layoutFingerprint(initialized) == UINT64_C(0xcc53a03cb09d6fea));
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

TEST_CASE("PsiFormer complex periodic initialization uses independent normalized real head components",
          "[wavefunction][psiformer][initialization]")
{
  const auto environment = pairedPeriodicEnvironment();
  const auto initialized = initializePsiFormerParameters(lithiumHydrideShape(), 73,
      PSIFORMER_COMPLEX_PERIODIC_V1, environment);
  const auto repeated = initializePsiFormerParameters(lithiumHydrideShape(), 73,
      PSIFORMER_COMPLEX_PERIODIC_V1, environment);
  const PsiFormerExecutionPlan plan(initialized.model_shape, initialized.layouts, environment);
  CHECK(initialized.profile == PSIFORMER_COMPLEX_PERIODIC_V1);
  CHECK(initialized.values == repeated.values);
  CHECK(initialized.layouts.size() == 43);
  CHECK(initialized.values.size() == 1644802);
  CHECK(initialized.diagnostics.constant_parameter_count == 514);
  CHECK(initialized.diagnostics.random_parameter_count == 1644288);
  CHECK(PARAMETER_ROLE_COUNT == 19);
  CHECK_FALSE(PsiFormerExecutionPlan::capabilities().complex_scalars);
  for (ParameterRole role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
  {
    const auto& real = plan.parameter(role);
    const auto& imaginary = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
    const double sigma = 1.0 / std::sqrt(512.0);
    const auto& real_diagnostic = diagnostic(initialized.diagnostics, role);
    const auto& imaginary_diagnostic = diagnostic(initialized.diagnostics, role, NO_ATTENTION_BLOCK,
                                                   HeadTensorComponent::IMAG);
    CHECK(real_diagnostic.law == InitializationLaw::NORMAL);
    CHECK(imaginary_diagnostic.law == InitializationLaw::NORMAL);
    CHECK(real_diagnostic.distribution_standard_deviation == Catch::Approx(sigma));
    CHECK(imaginary_diagnostic.distribution_standard_deviation == Catch::Approx(sigma));
    CHECK(real_diagnostic.observed_standard_deviation == Catch::Approx(sigma).margin(.001));
    CHECK(imaginary_diagnostic.observed_standard_deviation == Catch::Approx(sigma).margin(.001));
    CHECK(real.shape == std::vector<std::size_t>{256, 64});
    CHECK(imaginary.shape == real.shape);
    CHECK_FALSE(std::equal(initialized.values.begin() + real.begin, initialized.values.begin() + real.end,
                           initialized.values.begin() + imaginary.begin));
    long double cross = 0, square = 0;
    for (std::size_t i = 0; i < real.size(); ++i)
    {
      const long double r = initialized.values[real.begin + i];
      cross += r * initialized.values[imaginary.begin + i];
      square += r * r;
    }
    CHECK(std::abs(cross / square) < .05L);
  }
  // Two component variances recover the original head's variance; the real trunk
  // keeps its established per-role scale and is not duplicated or rescaled.
  CHECK(diagnostic(initialized.diagnostics, ParameterRole::ATTENTION_QUERY_WEIGHT, 0).
          distribution_standard_deviation == Catch::Approx(.0625));
  checkConstant(initialized, plan, ParameterRole::ENVELOPE_PI_UP, 1);
  checkConstant(initialized, plan, ParameterRole::CUSP_OPPOSITE_ALPHA, 1);
  CHECK_THROWS_AS(initializePsiFormerParameters(lithiumHydrideShape(), 73,
      PSIFORMER_COMPLEX_PERIODIC_V1), std::invalid_argument);
  CHECK_THROWS_AS(initializePsiFormerParameters(lithiumHydrideShape(), 73,
      DEEPQMC_PSIFORMER_V1, environment), std::invalid_argument);
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

TEST_CASE("Periodic pair initialization requires explicit metadata and zero affine biases",
          "[wavefunction][psiformer][periodic_profile]")
{
  const ModelShape shape{2, 2, 1, 16, 256, 4, 4};
  auto environment = pairedPeriodicEnvironment();
  CHECK_THROWS_AS(initializePsiFormerParameters(shape, 29, PSIFORMER_PERIODIC_PAIR_V1, environment),
                  std::invalid_argument);
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
  auto metadata = std::make_shared<PeriodicProfileMetadata>();
  metadata->cells.primitive = metadata->cells.simulation =
      {pf::GeometryBoundaryKind::PERIODIC, environment.lattice_vectors, environment.periodic_axes};
  metadata->primitive_motif = {{0, 0, 0}};
  metadata->motif_species = {3};
  metadata->feature_transform = PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1;
  metadata->occupied_columns = {{{0, 0, 0}, 0, 0, 1}, {{0, 0, 0}, 1, 0, 1},
                                {{0, 0, 0}, 0, 1, 1}, {{0, 0, 0}, 1, 1, 1}};
  environment.periodic_profile = metadata;
  CHECK_THROWS_AS(initializePsiFormerParameters(shape, 29, PSIFORMER_COMPLEX_PERIODIC_V1, environment),
                  std::invalid_argument);
  const auto initialized = initializePsiFormerParameters(shape, 29, PSIFORMER_PERIODIC_PAIR_V1, environment);
  const PsiFormerExecutionPlan plan(shape, initialized.layouts, initialized.environment);
  CHECK(initialized.profile == PSIFORMER_PERIODIC_PAIR_V1);
  CHECK(initialized.environment.periodic_profile->feature_transform == PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1);
  CHECK(initialized.environment.periodic_profile.get() != metadata.get());
  for (std::size_t block = 0; block < shape.attention_blocks; ++block)
    for (auto role : {ParameterRole::PAIR_ATTENTION_WEIGHT, ParameterRole::PAIR_ATTENTION_BIAS})
    {
      const auto& tensor = plan.parameter(role, block);
      CHECK(std::all_of(initialized.values.begin() + tensor.begin, initialized.values.begin() + tensor.end,
                        [](double value) { return value == 0; }));
    }
  environment.head_representation = HeadRepresentation::REAL;
  environment.amplitude_scalar_domain = ScalarDomain::REAL;
  const auto real = initializePsiFormerParameters(shape, 29, PSIFORMER_PERIODIC_PAIR_V1, environment);
  const PsiFormerExecutionPlan real_plan(shape, real.layouts, real.environment);
  CHECK_FALSE(real_plan.hasParameter(ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG));
  CHECK(real_plan.hasParameter(ParameterRole::PAIR_ATTENTION_WEIGHT, 0));
  metadata->feature_transform = PeriodicFeatureTransform::POLYNOMIAL_V1;
  metadata->pair_feature_dimension = periodicPairFeatureArity(metadata->feature_transform);
  CHECK(periodicFeatureArity(metadata->feature_transform) == 4);
  CHECK(metadata->pair_feature_dimension == 5);
  CHECK_THROWS_AS(initializePsiFormerParameters(shape, 29, PSIFORMER_PERIODIC_PAIR_V1, environment),
                  std::invalid_argument);
  metadata->electron_electron_cusp = false;
  for (auto head : {HeadRepresentation::REAL, HeadRepresentation::PAIRED_REAL_COMPLEX})
  {
    environment.head_representation = head;
    environment.amplitude_scalar_domain = head == HeadRepresentation::REAL ? ScalarDomain::REAL : ScalarDomain::COMPLEX;
    const auto polynomial = initializePsiFormerParameters(shape, 29, PSIFORMER_PERIODIC_PAIR_V1, environment);
    const PsiFormerExecutionPlan polynomial_plan(shape, polynomial.layouts, polynomial.environment);
    CHECK(polynomial_plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape ==
          std::vector<std::size_t>{4 * shape.nuclei + 1, shape.feature_dimension});
    CHECK(polynomial_plan.parameter(ParameterRole::PAIR_ATTENTION_WEIGHT, 0).shape ==
          std::vector<std::size_t>{5, shape.attention_heads});
    CHECK_FALSE(polynomial.environment.periodic_profile->electron_electron_cusp);
  }
}

} // namespace qmcplusplus::psiformer
