//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_accelerator_planning.cpp
 * @brief CPU-only tests for PsiFormer accelerator layout and publication contracts.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorPlanning.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus::psiformer
{
namespace
{

/// Append one canonical test tensor and derive its contiguous interval.
void appendLayout(std::vector<ParameterLayoutInput>& layouts,
                  std::string module,
                  std::string name,
                  std::vector<std::size_t> shape)
{
  const std::size_t begin = layouts.empty() ? 0 : layouts.back().end;
  const std::size_t size = std::accumulate(shape.begin(), shape.end(),
                                           std::size_t{1}, std::multiplies<>());
  layouts.push_back({std::move(module), std::move(name), std::move(shape), begin, begin + size});
}

/// Construct a compact complete 1+1-electron model for pure planning tests.
PsiFormerExecutionPlan makePlan()
{
  const ModelShape model{/*spin_up_electrons=*/1,
                         /*spin_down_electrons=*/1,
                         /*nuclei=*/1,
                         /*determinants=*/2,
                         /*feature_dimension=*/8,
                         /*attention_heads=*/2,
                         /*attention_blocks=*/1};
  const std::size_t electrons = model.electrons();
  const std::string prefix = "neural_network_wave_function/~/";
  std::vector<ParameterLayoutInput> layouts;
  appendLayout(layouts, prefix + "electronic_cusp_asymptotic", "anti_alpha", {});
  appendLayout(layouts, prefix + "exponential_envelopes", "pi_down",
               {model.determinants * electrons, model.nuclei});
  appendLayout(layouts, prefix + "exponential_envelopes", "pi_up",
               {model.determinants * electrons, model.nuclei});
  appendLayout(layouts, prefix + "exponential_envelopes", "zetas_down",
               {model.determinants * electrons, model.nuclei});
  appendLayout(layouts, prefix + "exponential_envelopes", "zetas_up",
               {model.determinants * electrons, model.nuclei});
  appendLayout(layouts, prefix + "omni_net/~/Backflow/~/mlp/linear_0", "w",
               {model.feature_dimension, model.determinants * electrons});
  appendLayout(layouts, prefix + "omni_net/~/Backflow_1/~/mlp/linear_0", "w",
               {model.feature_dimension, model.determinants * electrons});
  appendLayout(layouts, prefix + "omni_net/~/electron_gnn/~/electron_embedding/linear", "w",
               {4 * model.nuclei + 1, model.feature_dimension});

  const std::string block = prefix +
      "omni_net/~/electron_gnn/~/electron_gnn_layer/~/node_attention_electron_update_feature/";
  appendLayout(layouts, block + "mlp/linear_0", "b", {model.feature_dimension});
  appendLayout(layouts, block + "mlp/linear_0", "w", {model.feature_dimension, model.feature_dimension});
  appendLayout(layouts, block + "mlp/linear_1", "b", {model.feature_dimension});
  appendLayout(layouts, block + "mlp/linear_1", "w", {model.feature_dimension, model.feature_dimension});
  appendLayout(layouts, block + "multi_head_attention/key", "w",
               {model.feature_dimension, model.feature_dimension});
  appendLayout(layouts, block + "multi_head_attention/linear", "w",
               {model.feature_dimension, model.feature_dimension});
  appendLayout(layouts, block + "multi_head_attention/query", "w",
               {model.feature_dimension, model.feature_dimension});
  appendLayout(layouts, block + "multi_head_attention/value", "w",
               {model.feature_dimension, model.feature_dimension});
  return PsiFormerExecutionPlan(model, std::move(layouts));
}

} // namespace

TEST_CASE("PsiFormer accelerator backend selection fails closed", "[wavefunction][psiformer][accelerator]")
{
  const PsiFormerCompiledAcceleratorSupport none;
  CHECK(selectPsiFormerAcceleratorBackend("", none) == PsiFormerAcceleratorBackend::CPU);
  CHECK(selectPsiFormerAcceleratorBackend("AUTO", none) == PsiFormerAcceleratorBackend::CPU);
  CHECK_THROWS_WITH(selectPsiFormerAcceleratorBackend("yes", none),
                    Catch::Matchers::ContainsSubstring("requires a compiled accelerator"));
  CHECK_THROWS_WITH(selectPsiFormerAcceleratorBackend("cuda", none),
                    Catch::Matchers::ContainsSubstring("not compiled"));
  CHECK_THROWS_AS(selectPsiFormerAcceleratorBackend("mystery", none), std::invalid_argument);

  PsiFormerCompiledAcceleratorSupport support;
  support.openmp_target = true;
  support.sycl          = true;
  CHECK(selectPsiFormerAcceleratorBackend("yes", support) == PsiFormerAcceleratorBackend::SYCL);
  CHECK(selectPsiFormerAcceleratorBackend("omptarget", support) ==
        PsiFormerAcceleratorBackend::OPENMP_TARGET);
  support.cuda = true;
  CHECK(selectPsiFormerAcceleratorBackend("auto", support) == PsiFormerAcceleratorBackend::CUDA);
}

TEST_CASE("PsiFormer device layout is pointer-free and deterministic",
          "[wavefunction][psiformer][accelerator]")
{
  const PsiFormerExecutionPlan plan = makePlan();
  const PsiFormerDeviceLayout first = makePsiFormerDeviceLayout(plan);
  const PsiFormerDeviceLayout second = makePsiFormerDeviceLayout(plan);

  REQUIRE(first.tensors.size() == plan.parameterTensors().size());
  CHECK(first.parameter_count == plan.parameterCount());
  CHECK(first.fingerprint != 0);
  CHECK(first.fingerprint == second.fingerprint);
  CHECK(first.tensors == second.tensors);
  CHECK(first.tensors.front().rank == 0);
  CHECK(first.tensors.front().size() == 1);
  CHECK(first.tensors.back().rank == 2);
  CHECK(first.tensors.back().end == first.parameter_count);

  for (std::size_t tensor = 0; tensor < first.tensors.size(); ++tensor)
  {
    CHECK(first.tensors[tensor].begin == plan.parameterTensors()[tensor].begin);
    CHECK(first.tensors[tensor].end == plan.parameterTensors()[tensor].end);
    CHECK(first.tensors[tensor].role == plan.parameterTensors()[tensor].role);
  }
}

TEST_CASE("PsiFormer device publication is versioned and failure atomic",
          "[wavefunction][psiformer][accelerator]")
{
  PsiFormerDevicePublicationState publication(4);
  CHECK(publication.activeVersion() == 4);
  CHECK(publication.isReady(4));
  CHECK_FALSE(publication.publicationPending());
  CHECK_THROWS_AS(publication.beginPublication(4), std::invalid_argument);

  publication.beginPublication(7);
  CHECK(publication.activeVersion() == 4);
  REQUIRE(publication.pendingVersion());
  CHECK(*publication.pendingVersion() == 7);
  CHECK_THROWS_AS(publication.beginPublication(8), std::logic_error);
  CHECK_THROWS_AS(publication.completePublication(6), std::logic_error);

  publication.cancelPublication(7);
  CHECK(publication.activeVersion() == 4);
  CHECK_FALSE(publication.publicationPending());

  publication.beginPublication(8);
  publication.completePublication(8);
  CHECK(publication.activeVersion() == 8);
  CHECK(publication.isReady(8));
  CHECK_FALSE(publication.isReady(4));
  CHECK_FALSE(publication.publicationPending());
}

} // namespace qmcplusplus::psiformer
