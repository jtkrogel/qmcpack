//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_orbital_targets.cpp
 * @brief Deterministic tests for immutable conventional-orbital pretraining targets.
 */

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOrbitalTargets.h"
#include "QMCWaveFunctions/tests/ConstantSPOSet.h"
#include "SimulationCell.h"

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace qmcplusplus::psiformer
{
namespace
{

/// Construct deterministic per-spin SPO tables with values distinct in every slot.
std::unique_ptr<ConstantSPOSet<QMCTraits::ValueType>> makeSource(
    const char* name,
    int electrons,
    int orbitals,
    double offset)
{
  using Source = ConstantSPOSet<QMCTraits::ValueType>;
  auto source = std::make_unique<Source>(name, electrons, orbitals);
  Source::ValueMatrix values(electrons, orbitals);
  for (int electron = 0; electron < electrons; ++electron)
    for (int orbital = 0; orbital < orbitals; ++orbital)
      values(electron, orbital) = offset + 10.0 * electron + orbital;
  source->setRefVals(values);
  return source;
}

} // namespace

TEST_CASE("PsiFormer single-reference orbital target repeats full spin blocks",
          "[wavefunction][psiformer][pretraining]")
{
  OrbitalTargetDescriptor descriptor =
      OrbitalTargetDescriptor::makeSingleReference("deterministic-source", 2, 1, 2);
  REQUIRE(descriptor.terms().size() == 2);
  CHECK(descriptor.terms()[0].occupations_up == std::vector<std::size_t>{0, 1});
  CHECK(descriptor.terms()[0].occupations_down == std::vector<std::size_t>{0});
  CHECK(descriptor.terms()[0].coefficient == 1.0);
  CHECK(descriptor.terms()[1].source_ordinal == 0);

  std::vector<std::unique_ptr<SPOSet>> orbitals;
  orbitals.push_back(makeSource("up", 2, 4, 1.0));
  orbitals.push_back(makeSource("down", 1, 3, 101.0));
#if defined(QMC_COMPLEX)
  CHECK_THROWS_AS(OrbitalTargetEvaluator(std::move(descriptor), std::move(orbitals)),
                  std::invalid_argument);
#else
  OrbitalTargetEvaluator evaluator(std::move(descriptor), std::move(orbitals));
  const SimulationCell cell;
  ParticleSet electrons(cell);
  electrons.create({2, 1});
  const std::size_t storage_fingerprint = evaluator.storageFingerprint();
  const std::vector<double>& target = evaluator.evaluate(electrons);
  REQUIRE(target.size() == 18);
  const std::vector<double> expected_matrix{
      1.0, 2.0, 0.0,
      11.0, 12.0, 0.0,
      0.0, 0.0, 101.0};
  for (std::size_t determinant = 0; determinant < 2; ++determinant)
    for (std::size_t element = 0; element < expected_matrix.size(); ++element)
      CHECK(target[determinant * expected_matrix.size() + element] == expected_matrix[element]);
  CHECK(evaluator.storageFingerprint() == storage_fingerprint);
  CHECK(evaluator.retainedScalarBytes() >= target.size() * sizeof(double));
#endif
}

TEST_CASE("PsiFormer DETS target selection is stable and preserves raw coefficients",
          "[wavefunction][psiformer][pretraining]")
{
  const std::vector<double> coefficients{-0.125, 0.0, 0.5, -0.5, 0.25};
  const std::vector<std::vector<std::size_t>> map{{0, 1, 2, 3, 1}, {0, 0, 1, 2, 1}};
  const std::vector<std::vector<std::vector<std::size_t>>> configurations{
      {{0, 1}, {0, 2}, {1, 3}, {2, 3}},
      {{0}, {1}, {2}}};
  OrbitalTargetDescriptor descriptor = OrbitalTargetDescriptor::makeTruncatedMultideterminant(
      "explicit-dets", 2, 1, 5, coefficients, map, configurations);
  REQUIRE(descriptor.terms().size() == 5);
  CHECK(descriptor.terms()[0].source_ordinal == 2);
  CHECK(descriptor.terms()[1].source_ordinal == 3);
  CHECK(descriptor.terms()[2].source_ordinal == 4);
  CHECK(descriptor.terms()[0].coefficient == 0.5);
  CHECK(descriptor.terms()[1].coefficient == -0.5);
  CHECK(descriptor.terms()[2].coefficient == 0.25);
  CHECK(descriptor.terms()[3].coefficient == -0.125);
  CHECK(descriptor.terms()[4].coefficient == 0.0);
  CHECK(descriptor.terms()[0].occupations_up == std::vector<std::size_t>{1, 3});
  CHECK(descriptor.terms()[1].occupations_down == std::vector<std::size_t>{2});

  const std::uint64_t fingerprint = descriptor.fingerprint();
  OrbitalTargetDescriptor changed = OrbitalTargetDescriptor::makeTruncatedMultideterminant(
      "explicit-dets", 2, 1, 5, {-0.125, 0.0, 0.5, -0.5, 0.25000000000000006}, map,
      configurations);
  CHECK(changed.fingerprint() != fingerprint);

  std::vector<std::unique_ptr<SPOSet>> orbitals;
  auto original_up = makeSource("up", 2, 4, 1.0);
  orbitals.push_back(original_up->makeClone());
  orbitals.push_back(makeSource("down", 1, 3, 101.0));
#if !defined(QMC_COMPLEX)
  OrbitalTargetEvaluator evaluator(std::move(descriptor), std::move(orbitals));
  ConstantSPOSet<QMCTraits::ValueType>::ValueMatrix mutated(2, 4);
  mutated = QMCTraits::ValueType{-999.0};
  original_up->setRefVals(mutated);
  original_up.reset();

  const SimulationCell cell;
  ParticleSet electrons(cell);
  electrons.create({2, 1});
  const std::size_t storage_fingerprint = evaluator.storageFingerprint();
  const std::vector<double>& target = evaluator.evaluate(electrons);
  const double positive_root = std::pow(0.5, 1.0 / 3.0);
  const double quarter_root = std::pow(0.25, 1.0 / 3.0);
  CHECK(target[0] == Catch::Approx(positive_root * 2.0));
  CHECK(target[1] == Catch::Approx(positive_root * 4.0));
  CHECK(target[6] == 0.0);
  CHECK(target[8] == Catch::Approx(positive_root * 102.0));
  CHECK(target[9] == Catch::Approx(-positive_root * 3.0));
  CHECK(target[11] == 0.0);
  CHECK(target[18] == Catch::Approx(quarter_root * 1.0));
  for (std::size_t element = 36; element < 45; ++element)
    CHECK(target[element] == 0.0);

  // A second call overwrites, rather than accumulates, every target slot.
  const std::vector<double> first = target;
  CHECK(evaluator.evaluate(electrons) == first);
  CHECK(evaluator.storageFingerprint() == storage_fingerprint);
#endif
}

TEST_CASE("PsiFormer orbital target rejects malformed determinant metadata",
          "[wavefunction][psiformer][pretraining]")
{
  CHECK_THROWS_AS(OrbitalTargetDescriptor::makeSingleReference("", 1, 1, 1),
                  std::invalid_argument);
  CHECK_THROWS_AS(OrbitalTargetDescriptor::makeTruncatedMultideterminant(
                      "bad", 1, 1, 2, {1.0}, {{0}, {0}}, {{{0}}, {{0}}}),
                  std::invalid_argument);
  CHECK_THROWS_AS(OrbitalTargetDescriptor::makeTruncatedMultideterminant(
                      "bad", 1, 1, 1, {1.0}, {{1}, {0}}, {{{0}}, {{0}}}),
                  std::invalid_argument);
  CHECK_THROWS_AS(OrbitalTargetDescriptor::makeTruncatedMultideterminant(
                      "bad", 2, 1, 1, {1.0}, {{0}, {0}}, {{{0, 0}}, {{0}}}),
                  std::invalid_argument);
  CHECK_THROWS_AS(OrbitalTargetDescriptor::makeTruncatedMultideterminant(
                      "bad", 1, 1, 1, {std::numeric_limits<double>::quiet_NaN()},
                      {{0}, {0}}, {{{0}}, {{0}}}),
                  std::invalid_argument);
}

TEST_CASE("PsiFormer down-only target signs the first global beta column",
          "[wavefunction][psiformer][pretraining]")
{
  OrbitalTargetDescriptor descriptor = OrbitalTargetDescriptor::makeTruncatedMultideterminant(
      "down-only", 0, 1, 1, {-1.0}, {{0}}, {{{0}}});
  std::vector<std::unique_ptr<SPOSet>> orbitals;
  orbitals.push_back(makeSource("down", 1, 1, 7.0));
#if defined(QMC_COMPLEX)
  CHECK_THROWS_AS(OrbitalTargetEvaluator(std::move(descriptor), std::move(orbitals)),
                  std::invalid_argument);
#else
  OrbitalTargetEvaluator evaluator(std::move(descriptor), std::move(orbitals));
  const SimulationCell cell;
  ParticleSet electrons(cell);
  electrons.create({1});
  REQUIRE(evaluator.evaluate(electrons).size() == 1);
  CHECK(evaluator.evaluate(electrons)[0] == -7.0);
#endif
}

} // namespace qmcplusplus::psiformer
