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
TestParameterStore makeParameterStore()
{
  const ModelShape model = testModelShape();
  const std::size_t electrons = model.electrons();
  const std::string prefix = "neural_network_wave_function/~/";
  TestParameterStore store;

  appendLayout(store.layouts, prefix + "electronic_cusp_asymptotic", "anti_alpha", {});
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
               {4 * model.nuclei + 1, model.feature_dimension});

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
  CHECK_FALSE(capabilities.periodic_boundary);
  CHECK_FALSE(capabilities.complex_scalars);
}

TEST_CASE("PsiFormer execution plan rejects unsupported or malformed models", "[wavefunction][psiformer]")
{
  TestParameterStore store = makeParameterStore();

  ExecutionEnvironment periodic;
  periodic.boundary = BoundaryCondition::PERIODIC;
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

} // namespace qmcplusplus::psiformer
