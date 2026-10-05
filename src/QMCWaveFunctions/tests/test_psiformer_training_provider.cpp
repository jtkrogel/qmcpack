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

#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
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

} // namespace qmcplusplus
