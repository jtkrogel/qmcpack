//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_training_provider.cpp
 * @brief External-data-free tests of PsiFormer's structured parameter provider.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "Particle/MCMultiParticleMoves.h"
#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "ResourceCollection.h"
#include "psiformer_test_utils.h"

#include <limits>
#include <numeric>

namespace qmcplusplus
{

TEST_CASE("PsiFormer exposes native tensors without scalar registration",
          "[wavefunction][psiformer][training]")
{
  using namespace testing::psiformer;
  GeneratedFiles files = generateFiles("lih");
  PsiFormerWF component("pf_train", files.parameters.string(), files.configuration.string());

  wftrain::StructuredParameterProvider* provider = component.structuredParameterProvider();
  REQUIRE(provider != nullptr);
  const wftrain::StructuredParameterSchema& schema = provider->parameterSchema();
  const std::vector<Leaf> expected_layout          = makeLayout(4, 2);
  CHECK(schema.providerId() == "psiformer/pf_train");
  CHECK(schema.blocks().size() == expected_layout.size());

  const std::size_t expected_count =
      std::accumulate(expected_layout.begin(), expected_layout.end(), std::size_t{0},
                      [](std::size_t count, const Leaf& leaf) { return count + product(leaf.shape); });
  CHECK(schema.parameterCount() == expected_count);
  CHECK_FALSE(component.isOptimizable());
}

TEST_CASE("PsiFormer structured publication is atomic and clone shared",
          "[wavefunction][psiformer][training]")
{
  using namespace testing::psiformer;
  GeneratedFiles files = generateFiles("lih_pp");
  PsiFormerWF component("pf_train", files.parameters.string(), files.configuration.string());
  PsiFormerWF clone(component);

  wftrain::StructuredParameterSnapshot initial = component.snapshotParameters();
  REQUIRE_FALSE(initial.values.empty());
  wftrain::StructuredParameterSnapshot candidate = initial;
  candidate.values[0] += 1.0e-5;

  const std::size_t committed_version = component.publishParameters(candidate, initial.version);
  CHECK(committed_version == initial.version + 1);
  const wftrain::StructuredParameterSnapshot clone_snapshot = clone.snapshotParameters();
  CHECK(clone_snapshot.version == committed_version);
  CHECK(clone_snapshot.values[0] == candidate.values[0]);

  CHECK_THROWS_WITH(component.publishParameters(candidate, initial.version),
                    Catch::Matchers::ContainsSubstring("stale parameter version"));

  wftrain::StructuredParameterSnapshot invalid = clone_snapshot;
  invalid.values[0] = std::numeric_limits<double>::infinity();
  CHECK_THROWS_WITH(component.publishParameters(invalid, clone_snapshot.version),
                    Catch::Matchers::ContainsSubstring("non-finite"));
  CHECK(component.snapshotParameters().version == committed_version);

  invalid = clone_snapshot;
  invalid.schema_fingerprint = "different";
  CHECK_THROWS_WITH(component.publishParameters(invalid, clone_snapshot.version),
                    Catch::Matchers::ContainsSubstring("schema fingerprint"));
}

TEST_CASE("PsiFormer structured publication invalidates a selected-electron proposal",
          "[wavefunction][psiformer][training][multiparticle]")
{
  using namespace testing::psiformer;
  GeneratedFiles files = generateFiles("lih");
  const Geometry geometry = makeGeometry("lih");
  const SimulationCell simulation_cell;
  ParticleSet particles(simulation_cell);
  particles.setName("e");
  particles.create({2, 2});
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      particles.R[electron][dimension] = geometry.electrons[3 * electron + dimension];
  particles.update();

  PsiFormerWF component("pf_train_pending", files.parameters.string(), files.configuration.string());
  ParticleSet::ParticleGradient accepted_gradient(particles.getTotalNum());
  ParticleSet::ParticleLaplacian accepted_laplacian(particles.getTotalNum());
  accepted_gradient  = QMCTraits::ValueType(0);
  accepted_laplacian = QMCTraits::ValueType(0);
  component.evaluateLog(particles, accepted_gradient, accepted_laplacian);

  RefVectorWithLeader<WaveFunctionComponent> wfc_list(component);
  wfc_list.push_back(component);
  RefVectorWithLeader<ParticleSet> p_list(particles);
  p_list.push_back(particles);
  const ParticleSet::PosType proposed_position =
      particles.R[1] + ParticleSet::PosType{0.012, -0.008, 0.005};
  const MCMultiParticleMoves<CoordsType::POS> moves({0, 1}, {1}, {proposed_position});
  std::vector<PsiFormerWF::LogValue> log_ratios(1);
  std::vector<ParticleSet::ParticleGradient> proposed_gradient(1);
  std::vector<ParticleSet::ParticleLaplacian> proposed_laplacian(1);
  proposed_gradient[0].resize(particles.getTotalNum());
  proposed_laplacian[0].resize(particles.getTotalNum());
  proposed_gradient[0]  = QMCTraits::ValueType(0);
  proposed_laplacian[0] = QMCTraits::ValueType(0);
  RefVector<ParticleSet::ParticleGradient> proposed_gradient_list;
  RefVector<ParticleSet::ParticleLaplacian> proposed_laplacian_list;
  proposed_gradient_list.push_back(proposed_gradient[0]);
  proposed_laplacian_list.push_back(proposed_laplacian[0]);

  ResourceCollection resource_template("psiformer_training_pending_template");
  component.createResource(resource_template);
  ResourceCollection resources(resource_template);
  ResourceCollectionTeamLock<WaveFunctionComponent> lock(resources, wfc_list);
  component.mw_evaluateMultiParticleMove(
      wfc_list, p_list, moves, log_ratios,
      proposed_gradient_list, proposed_laplacian_list);

  wftrain::StructuredParameterSnapshot candidate = component.snapshotParameters();
  candidate.values[0] += 1.0e-5;
  const std::size_t committed_version = component.publishParameters(candidate, candidate.version);
  CHECK(committed_version == candidate.version + 1);
  CHECK_THROWS_WITH(component.mw_accept_rejectMultiParticleMove(
                        wfc_list, p_list, moves, std::vector<bool>{false}),
                    Catch::Matchers::ContainsSubstring("no matching pending proposal"));
}

} // namespace qmcplusplus
