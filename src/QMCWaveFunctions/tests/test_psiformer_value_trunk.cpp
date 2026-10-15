//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file test_psiformer_value_trunk.cpp
 * @brief Tiny real-trunk parity, storage and mixed-head admission tests.
 */
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "psiformer_test_utils.h"

namespace
{
namespace ps = qmcplusplus::psiformer;
namespace fixture = qmcplusplus::testing::psiformer;

/// Generate a tiny existing-architecture model without external files or new storage.
pf::PsiFormer tinyModel(bool periodic = false, bool paired = false, std::size_t width = 8)
{
  const auto geometry = fixture::makeGeometry("lih");
  auto leaves = fixture::makeLayout(4, 2, true, 2, periodic ? 7 : 4);
  for (auto& leaf : leaves)
    for (auto& extent : leaf.shape)
    {
      if (extent == 256)
        extent = width;
      else if (extent == 64)
        extent = 8;
    }
  if (paired)
  {
    std::vector<fixture::Leaf> imaginary;
    for (const auto& leaf : leaves)
      if (leaf.module.find("Backflow") != std::string::npos)
        imaginary.push_back({leaf.module, "w_imag", leaf.shape});
    leaves.insert(leaves.end(), imaginary.begin(), imaginary.end());
  }
  std::vector<pf::Layout> layouts;
  std::size_t begin = 0;
  for (const auto& leaf : leaves)
  {
    const std::size_t size = fixture::product(leaf.shape);
    layouts.push_back({leaf.module, leaf.name, leaf.shape, begin, begin + size});
    begin += size;
  }
  pf::Parameters parameters(fixture::makeParameters("lih", leaves), std::move(layouts));
  pf::ConfigData configuration(pf::Tensor({1,4,3}, geometry.electrons),
      pf::Tensor({2,3}, geometry.nuclei), pf::Tensor({2}, geometry.charges), 2, 2, 2);
  return pf::PsiFormer(std::move(parameters), std::move(configuration), 2, width, 2, 2);
}

/// Build the explicit periodic/paired metadata without changing legacy defaults.
ps::ExecutionEnvironment environment(bool periodic, bool paired)
{
  ps::ExecutionEnvironment result;
  if (periodic)
  {
    result.boundary = ps::BoundaryCondition::PERIODIC;
    result.geometry_feature_policy = ps::GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    result.periodic_axes = {true,true,true};
    result.lattice_vectors = {{{8,0,0},{0,8,0},{0,0,8}}};
  }
  if (paired)
  {
    result.head_representation = ps::HeadRepresentation::PAIRED_REAL_COMPLEX;
    result.amplitude_scalar_domain = ps::ScalarDomain::COMPLEX;
  }
  return result;
}

/// Validate the existing canonical real store through the unchanged plan boundary.
ps::PsiFormerExecutionPlan plan(const pf::PsiFormer& model, bool periodic = false, bool paired = false)
{
  return ps::PsiFormerExecutionPlan::fromParameters(model.p,
      {model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet,
       model.dim, model.heads, model.blocks}, environment(periodic, paired));
}
} // namespace

TEST_CASE("PsiFormer value trunk shares native features and preserves complete real storage",
          "[wavefunction][psiformer][value_trunk]")
{
  auto model = tinyModel();
  const auto execution_plan = plan(model);
  pf::DirectValueExecutor executor(model, execution_plan);
  auto complete = executor.makeWorkspace();
  auto trunk = executor.makeTrunkWorkspace();
  const auto electrons = model.cfg.configuration(0);
  const auto positions = pf::GeometryPositionView::interleaved(electrons.x.data(), model.ne);
  complete->setPositions(positions);
  trunk->setPositions(positions);
  CHECK(complete->vectorStorageBytes() == complete->requiredStorageBytes());
  CHECK(trunk->vectorStorageBytes() == trunk->requiredStorageBytes());
  const std::size_t head_bytes = pf::checkedStorageBytes<double>(model.ndet * model.ne * model.ne,
      "test orbital size overflow") + pf::determinantStorageRequirement(model.ndet, model.ne);
  CHECK(complete->vectorStorageBytes() - trunk->vectorStorageBytes() == head_bytes);
  const auto borrowed = executor.evaluateTrunk(*trunk);
  CHECK(borrowed.electron_count == model.ne);
  CHECK(borrowed.feature_width == model.dim);
  CHECK(borrowed.parameter_version == model.p.version());
  CHECK(borrowed.evaluation_generation == trunk->evaluationGeneration());
  CHECK(borrowed.geometry->electronCount() == model.ne);
  CHECK(borrowed.positions(1,2) == electrons.x[5]);

  auto graph = model.embedding(pf::coordinates(electrons, std::vector<std::size_t>{}));
  for (std::size_t layer = 0; layer < model.blocks; ++layer)
    graph = model.attention_block(graph, static_cast<int>(layer));
  for (std::size_t i = 0; i < model.ne * model.dim; ++i)
    CHECK(borrowed.features[i] == Catch::Approx(graph->value.x[i]).margin(2e-13));
  const auto complete_trunk = executor.evaluateTrunk(*complete);
  CHECK(std::equal(borrowed.features, borrowed.features + model.ne * model.dim, complete_trunk.features));
  CHECK(complete_trunk.cusp == borrowed.cusp);
  pf::EvaluationRequest request;
  request.spatial_derivatives = pf::SpatialDerivativeRequest::NONE;
  request.parameter_derivatives = pf::ParameterDerivativeRequest::NONE;
  request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  const auto reference = model.evaluate(electrons, request);
  const auto value = executor.evaluate(*complete);
  CHECK(value.sign == reference.sign);
  CHECK(value.logabs == Catch::Approx(reference.logabs).margin(2e-12));
  const auto fingerprint = trunk->storageFingerprint();
  for (int repeat = 0; repeat < 3; ++repeat)
    CHECK(executor.evaluateTrunk(*trunk).parameter_version == model.p.version());
  CHECK(trunk->storageFingerprint() == fingerprint);
}

TEST_CASE("PsiFormer value trunk invalidates failures and follows real parameter versions",
          "[wavefunction][psiformer][value_trunk]")
{
  auto model = tinyModel();
  const auto execution_plan = plan(model);
  pf::DirectValueExecutor executor(model, execution_plan, pf::DirectValuePreparation::REAL_TRUNK_ONLY);
  auto workspace = executor.makeTrunkWorkspace();
  const auto electrons = model.cfg.configuration(0);
  workspace->setPositions(pf::GeometryPositionView::interleaved(electrons.x.data(), model.ne));
  const auto first = executor.evaluateTrunk(*workspace);
  const std::vector<double> initial(first.features, first.features + model.ne * model.dim);
  const auto& embedding = execution_plan.parameter(ps::ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
  model.p.set_flat_value(embedding.begin, model.p.flat_values()[embedding.begin] + .01);
  const auto updated = executor.evaluateTrunk(*workspace);
  CHECK(updated.parameter_version == model.p.version());
  CHECK(updated.evaluation_generation != first.evaluation_generation);
  CHECK_FALSE(std::equal(initial.begin(), initial.end(), updated.features));
  workspace->setPosition(0, 0, std::numeric_limits<double>::quiet_NaN());
  CHECK(workspace->observedParameterVersion() == std::numeric_limits<std::size_t>::max());
  CHECK_THROWS(executor.evaluateTrunk(*workspace));
  CHECK(workspace->observedParameterVersion() == std::numeric_limits<std::size_t>::max());
  CHECK(workspace->evaluationGeneration() != updated.evaluation_generation);
  workspace->setPositions(pf::GeometryPositionView::interleaved(electrons.x.data(), model.ne));
  CHECK(executor.evaluateTrunk(*workspace).parameter_version == model.p.version());
  auto other_model = tinyModel(false, false, 16);
  pf::DirectValueExecutor other(other_model, plan(other_model));
  auto wrong_shape = other.makeTrunkWorkspace();
  CHECK_THROWS_AS(executor.evaluateTrunk(*wrong_shape), std::invalid_argument);
  auto different_nuclei = tinyModel();
  different_nuclei.cfg.nuclei.x[0] += .25;
  pf::DirectValueExecutor other_system(different_nuclei, plan(different_nuclei));
  auto same_shape_wrong_system = other_system.makeTrunkWorkspace();
  CHECK_THROWS_AS(executor.evaluateTrunk(*same_shape_wrong_system), std::invalid_argument);
}

TEST_CASE("PsiFormer paired head metadata admits only the shared real value trunk",
          "[wavefunction][psiformer][value_trunk]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  CHECK_THROWS_AS(pf::DirectValueExecutor(model, execution_plan), std::invalid_argument);
  pf::DirectValueExecutor trunk_executor(model, execution_plan, pf::DirectValuePreparation::REAL_TRUNK_ONLY);
  auto trunk = trunk_executor.makeTrunkWorkspace();
  const auto electrons = model.cfg.configuration(0);
  trunk->setPositions(pf::GeometryPositionView::interleaved(electrons.x.data(), model.ne));
  const auto borrowed = trunk_executor.evaluateTrunk(*trunk);
  CHECK(trunk->vectorStorageBytes() == trunk->requiredStorageBytes());
  CHECK(trunk_executor.trunkLayout()->preparation() == pf::DirectValuePreparation::REAL_TRUNK_ONLY);
  CHECK_THROWS_AS(trunk_executor.layout(), std::invalid_argument);
  CHECK_THROWS_AS(trunk_executor.makeWorkspace(), std::invalid_argument);
  CHECK_THROWS_AS(pf::DirectValueWorkspace(*trunk_executor.trunkLayout(), trunk_executor.nuclearPositions(),
      trunk_executor.boundary()), std::invalid_argument);

  // Appended imaginary leaves cannot influence the shared real network. The
  // companion model has exactly the same existing real leaves and periodic cell.
  auto companion = tinyModel(true, false);
  pf::DirectValueExecutor legacy(companion, plan(companion, true));
  auto real_trunk = legacy.makeTrunkWorkspace();
  real_trunk->setPositions(pf::GeometryPositionView::interleaved(electrons.x.data(), model.ne));
  const auto real = legacy.evaluateTrunk(*real_trunk);
  CHECK(std::equal(real.features, real.features + model.ne * model.dim, borrowed.features));
  CHECK(real.cusp == borrowed.cusp);
  CHECK(real_trunk->vectorStorageBytes() == trunk->vectorStorageBytes());
  auto complete = legacy.makeWorkspace();
  CHECK_THROWS_AS(trunk_executor.evaluate(*complete), std::invalid_argument);
}
