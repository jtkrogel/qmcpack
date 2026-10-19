//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_execution_plan.cpp
 * @brief Unit tests for typed PsiFormer plans and reusable aligned workspaces.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWorkspace.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus::psiformer
{
namespace
{

/// Minimal stand-in with the public layout surface exposed by pf::Parameters.
struct TestParameterStore
{
  std::vector<ParameterLayoutInput> layouts;
};

/// Return a small but structurally complete PsiFormer architecture for unit tests.
ModelShape testModelShape()
{
  return {/*spin_up_electrons=*/1,
          /*spin_down_electrons=*/1,
          /*nuclei=*/1,
          /*determinants=*/2,
          /*feature_dimension=*/8,
          /*attention_heads=*/2,
          /*attention_blocks=*/2};
}

/// Append one contiguous flat-layout interval derived from its tensor shape.
void appendLayout(std::vector<ParameterLayoutInput>& layouts,
                  std::string module,
                  std::string name,
                  std::vector<std::size_t> shape)
{
  const std::size_t begin = layouts.empty() ? 0 : layouts.back().end;
  const std::size_t size  = std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
  layouts.push_back({std::move(module), std::move(name), std::move(shape), begin, begin + size});
}

/// Construct all string-keyed layouts emitted for the test architecture.
TestParameterStore makeParameterStore(ModelShape model = testModelShape(), bool include_same_alpha = true,
                                     bool periodic = false)
{
  const std::size_t electrons = model.electrons();
  const std::string prefix = "neural_network_wave_function/~/";
  TestParameterStore store;

  appendLayout(store.layouts, prefix + "electronic_cusp_asymptotic", "anti_alpha", {});
  if (include_same_alpha)
    appendLayout(store.layouts, prefix + "electronic_cusp_asymptotic", "same_alpha", {});
  appendLayout(store.layouts, prefix + "exponential_envelopes", "pi_down",
               {model.determinants * electrons, model.nuclei});
  appendLayout(store.layouts, prefix + "exponential_envelopes", "pi_up",
               {model.determinants * electrons, model.nuclei});
  appendLayout(store.layouts, prefix + "exponential_envelopes", "zetas_down",
               {model.determinants * electrons, model.nuclei});
  appendLayout(store.layouts, prefix + "exponential_envelopes", "zetas_up",
               {model.determinants * electrons, model.nuclei});
  appendLayout(store.layouts, prefix + "omni_net/~/Backflow/~/mlp/linear_0", "w",
               {model.feature_dimension, model.determinants * electrons});
  appendLayout(store.layouts, prefix + "omni_net/~/Backflow_1/~/mlp/linear_0", "w",
               {model.feature_dimension, model.determinants * electrons});
  appendLayout(store.layouts, prefix + "omni_net/~/electron_gnn/~/electron_embedding/linear", "w",
               {(periodic ? 7 : 4) * model.nuclei + 1, model.feature_dimension});

  for (std::size_t block = 0; block < model.attention_blocks; ++block)
  {
    const std::string layer = block == 0 ? "electron_gnn_layer" : "electron_gnn_layer_" + std::to_string(block);
    const std::string base = prefix + "omni_net/~/electron_gnn/~/" + layer +
        "/~/node_attention_electron_update_feature/";
    appendLayout(store.layouts, base + "mlp/linear_0", "b", {model.feature_dimension});
    appendLayout(store.layouts, base + "mlp/linear_0", "w", {model.feature_dimension, model.feature_dimension});
    appendLayout(store.layouts, base + "mlp/linear_1", "b", {model.feature_dimension});
    appendLayout(store.layouts, base + "mlp/linear_1", "w", {model.feature_dimension, model.feature_dimension});
    appendLayout(store.layouts, base + "multi_head_attention/key", "w",
                 {model.feature_dimension, model.feature_dimension});
    appendLayout(store.layouts, base + "multi_head_attention/linear", "w",
                 {model.feature_dimension, model.feature_dimension});
    appendLayout(store.layouts, base + "multi_head_attention/query", "w",
                 {model.feature_dimension, model.feature_dimension});
    appendLayout(store.layouts, base + "multi_head_attention/value", "w",
                 {model.feature_dimension, model.feature_dimension});
  }
  return store;
}

/// Metadata-only paired head environment, without claiming a public complex executor.
ExecutionEnvironment pairedEnvironment()
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

/// Append only the explicitly named imaginary real-valued head leaves.
void appendImaginaryHeads(TestParameterStore& store, bool include_down = true)
{
  const ModelShape model = testModelShape();
  const std::string prefix = "neural_network_wave_function/~/omni_net/~/";
  appendLayout(store.layouts, prefix + "Backflow/~/mlp/linear_0", "w_imag",
               {model.feature_dimension, model.determinants * model.electrons()});
  if (include_down)
    appendLayout(store.layouts, prefix + "Backflow_1/~/mlp/linear_0", "w_imag",
                 {model.feature_dimension, model.determinants * model.electrons()});
}

} // namespace

TEST_CASE("PsiFormer execution plan types imported parameter layouts", "[wavefunction][psiformer]")
{
  TestParameterStore store = makeParameterStore();
  const std::string original_query_module = store.layouts[15].module;
  const PsiFormerExecutionPlan plan = PsiFormerExecutionPlan::fromParameters(store, testModelShape());

  REQUIRE(plan.parameterTensors().size() == 25);
  REQUIRE(plan.parameterCount() == store.layouts.back().end);
  CHECK(plan.adjointConvention() == AdjointConvention::TRANSPOSE);
  CHECK(plan.environment().boundary == BoundaryCondition::OPEN);
  CHECK(plan.environment().head_representation == HeadRepresentation::REAL);
  CHECK(PARAMETER_ROLE_COUNT == 19);
  CHECK(static_cast<std::size_t>(ParameterRole::BACKFLOW_UP_WEIGHT) == 9);
  CHECK(static_cast<std::size_t>(ParameterRole::BACKFLOW_DOWN_WEIGHT) == 10);
  CHECK(static_cast<std::size_t>(ParameterRole::CUSP_OPPOSITE_ALPHA) == 16);
  CHECK(plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT).head_component == HeadTensorComponent::REAL);
  CHECK(plan.optionalParameter(ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK,
                               HeadTensorComponent::IMAG) == nullptr);

  const ParameterTensorDescriptor& query = plan.parameter(ParameterRole::ATTENTION_QUERY_WEIGHT, 0);
  CHECK(query.attention_block == 0);
  CHECK(query.shape == std::vector<std::size_t>{8, 8});
  CHECK(query.module.find("multi_head_attention/query") != std::string::npos);

  // The plan owns immutable metadata rather than pointers into the parameter
  // store, so source mutation cannot invalidate or silently change descriptors.
  store.layouts[15].module = "mutated/source/layout";
  CHECK(plan.parameterTensors()[15].module == original_query_module);

  const ModelCapabilities capabilities = PsiFormerExecutionPlan::capabilities();
  CHECK(capabilities.open_boundary);
  CHECK(capabilities.real_scalars);
  CHECK(capabilities.periodic_boundary);
  CHECK_FALSE(capabilities.complex_scalars);
}

TEST_CASE("PsiFormer execution plan rejects unsupported or malformed models", "[wavefunction][psiformer]")
{
  TestParameterStore store = makeParameterStore();

  ExecutionEnvironment periodic;
  periodic.boundary = BoundaryCondition::PERIODIC;
  periodic.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
  periodic.periodic_axes = {true, true, true};
  periodic.lattice_vectors = {{{8, 0, 0}, {0, 8, 0}, {0, 0, 8}}};
  // The molecular fixture has a 4*Nion+1 embedding and must not be silently
  // reinterpreted under the periodic 7*Nion+1 feature contract.
  CHECK_THROWS_AS(PsiFormerExecutionPlan::fromParameters(store, testModelShape(), periodic), std::invalid_argument);

  ExecutionEnvironment complex;
  complex.compute_scalar_domain = ScalarDomain::COMPLEX;
  CHECK_THROWS_AS(PsiFormerExecutionPlan::fromParameters(store, testModelShape(), complex), std::invalid_argument);

  complex = {};
  complex.parameter_scalar_domain = ScalarDomain::COMPLEX;
  CHECK_THROWS_AS(PsiFormerExecutionPlan::fromParameters(store, testModelShape(), complex), std::invalid_argument);

  complex = {};
  complex.amplitude_scalar_domain = ScalarDomain::COMPLEX;
  CHECK_THROWS_AS(PsiFormerExecutionPlan::fromParameters(store, testModelShape(), complex), std::invalid_argument);

  TestParameterStore bad_shape = makeParameterStore();
  bad_shape.layouts[0].shape   = {1};
  CHECK_THROWS_AS(PsiFormerExecutionPlan::fromParameters(bad_shape, testModelShape()), std::invalid_argument);

  TestParameterStore missing = makeParameterStore();
  missing.layouts.pop_back();
  CHECK_THROWS_AS(PsiFormerExecutionPlan::fromParameters(missing, testModelShape()), std::invalid_argument);
}

TEST_CASE("PsiFormer execution plan supports a canonical optional same-spin cusp",
          "[wavefunction][psiformer]")
{
  const ModelShape pair_free_shape = testModelShape();
  TestParameterStore canonical = makeParameterStore(pair_free_shape, false);
  const PsiFormerExecutionPlan plan =
      PsiFormerExecutionPlan::fromParameters(canonical, pair_free_shape);

  CHECK(plan.parameterTensors().size() == 24);
  CHECK_FALSE(plan.hasParameter(ParameterRole::CUSP_SAME_ALPHA));
  CHECK(plan.optionalParameter(ParameterRole::CUSP_SAME_ALPHA) == nullptr);
  CHECK(plan.hasParameter(ParameterRole::CUSP_OPPOSITE_ALPHA));
  CHECK(plan.optionalParameter(ParameterRole::CUSP_OPPOSITE_ALPHA) ==
        &plan.parameter(ParameterRole::CUSP_OPPOSITE_ALPHA));
  CHECK_THROWS_AS(plan.parameter(ParameterRole::CUSP_SAME_ALPHA), std::out_of_range);

  // An older export containing the unused scalar remains readable for a 1+1
  // model, but any spin population containing a pair requires the leaf.
  TestParameterStore legacy = makeParameterStore(pair_free_shape, true);
  const PsiFormerExecutionPlan legacy_plan =
      PsiFormerExecutionPlan::fromParameters(legacy, pair_free_shape);
  CHECK(legacy_plan.hasParameter(ParameterRole::CUSP_SAME_ALPHA));

  ModelShape same_pair_shape = pair_free_shape;
  same_pair_shape.spin_up_electrons = 2;
  TestParameterStore missing_required = makeParameterStore(same_pair_shape, false);
  CHECK_THROWS_WITH(PsiFormerExecutionPlan::fromParameters(missing_required, same_pair_shape),
                    Catch::Matchers::ContainsSubstring("same-spin cusp alpha"));
}

TEST_CASE("PsiFormer paired-real head metadata preserves legacy roles and explicit components",
          "[wavefunction][psiformer]")
{
  const ExecutionEnvironment environment = pairedEnvironment();
  TestParameterStore store = makeParameterStore(testModelShape(), true, true);
  appendImaginaryHeads(store);
  const PsiFormerExecutionPlan plan = PsiFormerExecutionPlan::fromParameters(store, testModelShape(), environment);
  CHECK(plan.parameterTensors().size() == 27);
  CHECK(plan.environment().compute_scalar_domain == ScalarDomain::REAL);
  CHECK(plan.environment().parameter_scalar_domain == ScalarDomain::REAL);
  CHECK(plan.environment().amplitude_scalar_domain == ScalarDomain::COMPLEX);
  CHECK_FALSE(PsiFormerExecutionPlan::capabilities().complex_scalars);
  const auto& real = plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT);
  const auto& imaginary = plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK,
                                         HeadTensorComponent::IMAG);
  CHECK(real.name == "w");
  CHECK(real.head_component == HeadTensorComponent::REAL);
  CHECK(imaginary.name == "w_imag");
  CHECK(imaginary.role == real.role);
  CHECK(imaginary.head_component == HeadTensorComponent::IMAG);
  CHECK(imaginary.shape == real.shape);
  CHECK(imaginary.begin != real.begin);
  CHECK_THROWS_AS(plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK,
                                  static_cast<HeadTensorComponent>(99)), std::out_of_range);
  CHECK_THROWS_AS(plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT, NO_ATTENTION_BLOCK,
                                  HeadTensorComponent::IMAG), std::invalid_argument);
  CHECK_THROWS_AS(plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT, 0, HeadTensorComponent::IMAG),
                  std::invalid_argument);

  TestParameterStore missing = makeParameterStore(testModelShape(), true, true);
  appendImaginaryHeads(missing, false);
  CHECK_THROWS_WITH(PsiFormerExecutionPlan::fromParameters(missing, testModelShape(), environment),
                    Catch::Matchers::ContainsSubstring("missing imaginary head"));
  ExecutionEnvironment legacy = environment;
  legacy.head_representation = HeadRepresentation::REAL;
  legacy.amplitude_scalar_domain = ScalarDomain::REAL;
  CHECK_THROWS_WITH(PsiFormerExecutionPlan::fromParameters(store, testModelShape(), legacy),
                    Catch::Matchers::ContainsSubstring("require paired-real head metadata"));
  ExecutionEnvironment invalid = environment;
  invalid.head_representation = static_cast<HeadRepresentation>(99);
  CHECK_THROWS_AS(PsiFormerExecutionPlan::fromParameters(store, testModelShape(), invalid), std::invalid_argument);
  invalid = environment;
  invalid.amplitude_scalar_domain = ScalarDomain::REAL;
  CHECK_THROWS_AS(PsiFormerExecutionPlan::fromParameters(store, testModelShape(), invalid), std::invalid_argument);
  invalid = environment;
  invalid.compute_scalar_domain = ScalarDomain::COMPLEX;
  CHECK_THROWS_AS(PsiFormerExecutionPlan::fromParameters(store, testModelShape(), invalid), std::invalid_argument);
  store.layouts.back().module = "neural_network_wave_function/~/electron_embedding/linear";
  CHECK_THROWS_WITH(PsiFormerExecutionPlan::fromParameters(store, testModelShape(), environment),
                    Catch::Matchers::ContainsSubstring("Unrecognized PsiFormer parameter tensor"));
}

TEST_CASE("PsiFormer workspace grows only at explicit preparation boundaries", "[wavefunction][psiformer]")
{
  const TestParameterStore store = makeParameterStore();
  const PsiFormerExecutionPlan plan = PsiFormerExecutionPlan::fromParameters(store, testModelShape());

  PsiFormerWorkspace workspace;
  const WorkspaceRequirements value_requirements =
      makeWorkspaceRequirements(plan, {EvaluationMode::VALUE_ONLY, 2, 0, 0});
  CHECK(value_requirements[WorkspaceRegion::FIRST_DERIVATIVES] == 0);
  CHECK(value_requirements[WorkspaceRegion::SECOND_DERIVATIVES] == 0);
  CHECK(value_requirements[WorkspaceRegion::PARAMETER_OUTPUT] == 0);
  REQUIRE(workspace.prepare(value_requirements));

  BasicWorkspaceView<double> first_view = workspace.view();
  CHECK(workspace.isCurrent(first_view));
  CHECK(workspace.allocationGeneration() == 1);
  for (std::size_t region = 0; region < WORKSPACE_REGION_COUNT; ++region)
    if (!first_view.regions[region].empty())
      CHECK(reinterpret_cast<std::uintptr_t>(first_view.regions[region].data()) % QMC_SIMD_ALIGNMENT == 0);

  // Re-preparing the same high-water marks performs no allocation and leaves
  // previously returned views valid.
  CHECK_FALSE(workspace.prepare(value_requirements));
  CHECK(workspace.allocationGeneration() == 1);
  CHECK(workspace.isCurrent(first_view));

  const WorkspaceRequirements score_requirements =
      makeWorkspaceRequirements(plan, {EvaluationMode::PARAMETER_SCORE, 2, 0, 0});
  CHECK(score_requirements[WorkspaceRegion::PARAMETER_OUTPUT] == 2 * plan.parameterCount());
  REQUIRE(workspace.prepare(score_requirements));
  CHECK(workspace.allocationGeneration() == 2);
  CHECK_FALSE(workspace.isCurrent(first_view));

  const WorkspaceRequirements vgl_requirements =
      makeWorkspaceRequirements(plan, {EvaluationMode::FULL_VGL, 2, 0, 3});
  CHECK(vgl_requirements[WorkspaceRegion::FIRST_DERIVATIVES] > 0);
  CHECK(vgl_requirements[WorkspaceRegion::SECOND_DERIVATIVES] > 0);
  CHECK(workspace.prepare(vgl_requirements));

  const WorkspaceRequirements weighted_vjp =
      makeWorkspaceRequirements(plan, {EvaluationMode::VIRTUAL_WEIGHTED_PARAMETER_VJP, 2, 7, 0});
  CHECK(weighted_vjp[WorkspaceRegion::PARAMETER_OUTPUT] == plan.parameterCount());
  CHECK(weighted_vjp[WorkspaceRegion::REVERSE_ADJOINTS] > score_requirements[WorkspaceRegion::REVERSE_ADJOINTS]);
}

TEST_CASE("Periodic profile metadata preserves cells occupations and pair tensor identity",
          "[wavefunction][psiformer][periodic_profile]")
{
  const auto model = testModelShape();
  auto environment = pairedEnvironment();
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
  auto profile = std::make_shared<PeriodicProfileMetadata>();
  profile->cells.simulation = {pf::GeometryBoundaryKind::PERIODIC, environment.lattice_vectors,
                               environment.periodic_axes};
  profile->cells.primitive = profile->cells.simulation;
  profile->cells.primitive.lattice_vectors[0][0] = 4;
  profile->cells.tiling[0][0] = 2;
  profile->primitive_motif = {{0.13, -0.22, 0.31}};
  profile->motif_species = {3};
  profile->occupied_columns = {{{0, 0, 0}, 0, 0, 1}, {{0, 0, 0}, 0, 1, 1}};
  environment.periodic_profile = profile;
  auto store = makeParameterStore(model, true, true);
  appendImaginaryHeads(store);
  CHECK_THROWS_AS(PsiFormerExecutionPlan(model, store.layouts, environment), std::invalid_argument);
  for (std::size_t block = 0; block < model.attention_blocks; ++block)
  {
    const auto layer = block == 0 ? std::string("electron_gnn_layer") : "electron_gnn_layer_" + std::to_string(block);
    const auto module = "periodic/" + layer + "/pair_attention";
    appendLayout(store.layouts, module, "w", {8, model.attention_heads});
    appendLayout(store.layouts, module, "b", {model.attention_heads});
  }
  const PsiFormerExecutionPlan plan(model, store.layouts, environment);
  CHECK(static_cast<std::size_t>(ParameterRole::CUSP_OPPOSITE_ALPHA) == 16);
  CHECK(static_cast<std::size_t>(ParameterRole::PAIR_ATTENTION_WEIGHT) == 17);
  CHECK(plan.parameter(ParameterRole::PAIR_ATTENTION_WEIGHT, 0).shape == std::vector<std::size_t>{8, 2});
  CHECK(plan.parameter(ParameterRole::PAIR_ATTENTION_BIAS, 1).shape == std::vector<std::size_t>{2});
  CHECK(plan.environment().periodic_profile.get() != profile.get());
  profile->feature_transform = PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1;
  CHECK(plan.environment().periodic_profile->feature_transform == PeriodicFeatureTransform::LOG_COMPLEMENT_V1);
  CHECK_NOTHROW(PsiFormerExecutionPlan(model, store.layouts, environment));
  auto invalid = environment;
  invalid.periodic_profile.reset();
  CHECK_THROWS_AS(PsiFormerExecutionPlan(model, store.layouts, invalid), std::invalid_argument);
  invalid = pairedEnvironment();
  CHECK_THROWS_AS(PsiFormerExecutionPlan(model, store.layouts, invalid), std::invalid_argument);
  profile->version = 2;
  CHECK_THROWS_AS(validatePeriodicProfileMetadata(model, environment), std::invalid_argument);
  profile->version = 1;
  profile->occupied_columns[1].spin = 0;
  CHECK_THROWS_AS(validatePeriodicProfileMetadata(model, environment), std::invalid_argument);
  profile->occupied_columns[1].spin = 1;
  profile->occupied_columns[0].occupation = 0.5;
  CHECK_THROWS_AS(validatePeriodicProfileMetadata(model, environment), std::invalid_argument);
  profile->occupied_columns[0].occupation = 1;
  profile->cells.tiling[0][0] = 1;
  CHECK_THROWS_AS(validatePeriodicProfileMetadata(model, environment), std::invalid_argument);
  profile->cells.tiling[0][0] = 2;
  auto repeated = model;
  repeated.spin_up_electrons = repeated.spin_down_electrons = 2;
  profile->occupied_columns = {{{0, 0, 0}, 0, 0, 1}, {{0, 0, 0}, 1, 0, 1},
                               {{0, 0, 0}, 0, 1, 1}, {{0, 0, 0}, 1, 1, 1}};
  CHECK_NOTHROW(validatePeriodicProfileMetadata(repeated, environment));
  profile->occupied_columns[1].band = 0;
  CHECK_THROWS_AS(validatePeriodicProfileMetadata(repeated, environment), std::invalid_argument);
  profile->occupied_columns[1].reciprocal = {1, -1, 0};
  CHECK_NOTHROW(validatePeriodicProfileMetadata(repeated, environment));
  environment.head_representation = HeadRepresentation::REAL;
  environment.amplitude_scalar_domain = ScalarDomain::REAL;
  CHECK_THROWS_AS(validatePeriodicProfileMetadata(repeated, environment), std::invalid_argument);
  profile->occupied_columns[1].band = 1;
  profile->occupied_columns[1].reciprocal = {};
  CHECK_NOTHROW(validatePeriodicProfileMetadata(repeated, environment));
}

} // namespace qmcplusplus::psiformer
