//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_structured_parameter_provider.cpp
 * @brief Unit tests for tensor-level training parameter metadata.
 */

#include <catch2/catch_test_macros.hpp>

#include "QMCWaveFunctions/Optimization/StructuredParameterProvider.h"

#include <limits>
#include <string>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Return a two-tensor schema containing both trainable and frozen metadata.
StructuredParameterSchema makeSchema()
{
  return StructuredParameterSchema(
      "toy/model",
      {{"dense/weight", {2, 3}, 0, 6, ParameterScalarDomain::REAL64, true, "weights"},
       {"dense/bias", {3}, 6, 3, ParameterScalarDomain::REAL64, false, "biases"}});
}

} // namespace

TEST_CASE("Structured parameter schemas validate tensor blocks", "[wavefunction][training]")
{
  const StructuredParameterSchema schema = makeSchema();
  CHECK(schema.providerId() == "toy/model");
  CHECK(schema.parameterCount() == 9);
  REQUIRE(schema.blocks().size() == 2);
  CHECK(schema.blocks()[0].trainable);
  CHECK_FALSE(schema.blocks()[1].trainable);
  CHECK(schema.fingerprint().size() == 16);

  const StructuredParameterSchema equivalent = makeSchema();
  CHECK(equivalent.fingerprint() == schema.fingerprint());

  StructuredParameterSchema changed(
      "toy/model",
      {{"dense/weight", {2, 3}, 0, 6, ParameterScalarDomain::REAL64, false, "weights"},
       {"dense/bias", {3}, 6, 3, ParameterScalarDomain::REAL64, false, "biases"}});
  CHECK(changed.fingerprint() != schema.fingerprint());
}

TEST_CASE("Structured parameter schemas reject ambiguous layouts", "[wavefunction][training]")
{
  CHECK_THROWS_AS(StructuredParameterSchema("", {{"x", {1}, 0, 1}}), std::invalid_argument);
  CHECK_THROWS_AS(StructuredParameterSchema("toy", {}), std::invalid_argument);
  CHECK_THROWS_AS(StructuredParameterSchema("toy", {{"", {1}, 0, 1}}), std::invalid_argument);
  CHECK_THROWS_AS(
      StructuredParameterSchema("toy", {{"x", {1}, 0, 1}, {"x", {1}, 1, 1}}),
      std::invalid_argument);
  CHECK_THROWS_AS(
      StructuredParameterSchema("toy", {{"x", {2}, 0, 2}, {"y", {1}, 3, 1}}),
      std::invalid_argument);
  CHECK_THROWS_AS(StructuredParameterSchema("toy", {{"x", {2}, 0, 1}}), std::invalid_argument);
  CHECK_THROWS_AS(StructuredParameterSchema("toy", {{"x", {0}, 0, 0}}), std::invalid_argument);

  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(StructuredParameterSchema("toy", {{"x", {maximum, 2}, 0, 0}}),
                  std::overflow_error);
}

} // namespace qmcplusplus::wftrain

