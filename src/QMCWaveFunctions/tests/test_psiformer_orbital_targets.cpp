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
#include "Particle/ParticleSetPool.h"
#include "OhmmsData/Libxml2Doc.h"
#include "QMCWaveFunctions/WaveFunctionFactory.h"
#include "QMCWaveFunctions/Fermion/SlaterDet.h"
#include "QMCWaveFunctions/LCAO/LCAOrbitalSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOrbitalTargets.h"
#include "QMCWaveFunctions/tests/ConstantSPOSet.h"
#include "SimulationCell.h"
#include "Utilities/RuntimeOptions.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
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

#ifdef QMC_COMPLEX
TEST_CASE("Periodic Gaussian targets freeze coefficients and occupied gauge",
          "[wavefunction][psiformer][pretraining][periodic_lcao_target]")
{
  Libxml2Document document;
  REQUIRE(document.parseFromString(R"(<qmcsystem>
    <simulationcell><parameter name="lattice">
      6.7463223 6.7463223 0  0 3.37316115 3.37316115  3.37316115 0 3.37316115
    </parameter><parameter name="bconds">p p p</parameter></simulationcell>
    <particleset name="ion0" size="4"><group name="C"><parameter name="charge">4</parameter></group>
      <attrib name="position" datatype="posArray">
      0 0 0  1.686580575 1.686580575 1.686580575  3.37316115 3.37316115 0
      5.059741725 5.059741725 1.686580575</attrib>
      <attrib name="ionid" datatype="stringArray">C C C C</attrib></particleset>
    <particleset name="e"><group name="u" size="8"><parameter name="charge">-1</parameter></group>
      <group name="d" size="8"><parameter name="charge">-1</parameter></group></particleset>
    <wavefunction name="teacher" target="e">
      <sposet_collection type="MolecularOrbital" name="LCAOBSet" source="ion0"
        transform="no" key="GTO" href="C_Diamond_2x1x1-Gaussian.h5" twist="0 0 0"
        PBCimages="5 5 5" gpu="no" source_identity="file_image" source_identity_max_bytes="524288">
        <basisset name="LCAOBSet" key="GTO" transform="no"/>
        <sposet name="occupied" size="8"><occupation mode="ground"/>
          <coefficient size="116" spindataset="0"/></sposet>
      </sposet_collection>
      <determinantset><slaterdeterminant><determinant id="up" sposet="occupied"/>
        <determinant id="down" sposet="occupied"/></slaterdeterminant></determinantset>
    </wavefunction></qmcsystem>)"));
  ParticleSetPool pool(OHMMS::Controller);
  auto* child = document.getRoot()->children;
  xmlNodePtr wavefunction = nullptr;
  for (; child; child = child->next)
    if (xmlStrEqual(child->name, BAD_CAST "simulationcell"))
      pool.readSimulationCellXML(child);
    else if (xmlStrEqual(child->name, BAD_CAST "particleset"))
      REQUIRE(pool.put(child));
    else if (xmlStrEqual(child->name, BAD_CAST "wavefunction"))
      wavefunction = child;
  auto* electrons = pool.getParticleSet("e");
  auto* ions = pool.getParticleSet("ion0");
  REQUIRE(electrons != nullptr);
  REQUIRE(ions != nullptr);
  for (std::size_t row = 0; row < electrons->R.size(); ++row)
    electrons->R[row] = {0.21 + 0.07 * row, -0.34 + 0.11 * row, 0.53 - 0.03 * row};
  WaveFunctionFactory factory(*electrons, pool.getPool(), OHMMS::Controller);
  auto state = factory.buildTWF(wavefunction, RuntimeOptions{});
  REQUIRE(state != nullptr);
  REQUIRE(state->getOrbitals().size() == 1);
  const auto* slater = dynamic_cast<const SlaterDet*>(state->getOrbitals()[0].get());
  REQUIRE(slater != nullptr);
  const auto& source = dynamic_cast<const LCAOrbitalSet&>(state->getSPOSet("occupied"));
  REQUIRE(source.fileImageIdentity() != nullptr);
  REQUIRE(source.periodicIdentity() != nullptr);
  const auto n = electrons->R.size();
  const auto cap = (n * n + 2 * source.getOrbitalSetSize() + 2 * source.C->size()) * sizeof(std::complex<double>);
  CHECK_THROWS_AS(ComplexOrbitalTargetEvaluator::fromSingleReference(*slater, *electrons, {}, 1, cap - 1),
                  std::length_error);
  auto target = ComplexOrbitalTargetEvaluator::fromSingleReference(*slater, *electrons, {}, 1, cap);
  CHECK(target.retainedScalarBytes() == cap);
  electrons->update();
  const auto frozen = target.evaluate(*electrons);
  const auto fingerprint = target.descriptor().fingerprint();
  auto reference = source.makeClone();
  SPOSet::ValueVector values(source.getOrbitalSetSize());
  for (std::size_t row = 0; row < n; ++row)
  {
    reference->evaluateValue(*electrons, row, values);
    for (std::size_t column = 0; column < n; ++column)
      CHECK(std::abs(frozen[row * n + column] -
          ((row < 8) == (column < 8) ? values[column % 8] : std::complex<double>{})) < 1e-12);
  }
  const auto accepted = source.C->data()[0];
  source.C->data()[0] += std::complex<double>{0.13, -0.09};
  CHECK(target.evaluate(*electrons) == frozen);
  auto changed = ComplexOrbitalTargetEvaluator::fromSingleReference(*slater, *electrons, {}, 1, cap);
  CHECK(changed.descriptor().fingerprint() != fingerprint);
  source.C->data()[0] = accepted;
  const auto position = electrons->R[0];
  for (int d = 0; d < 3; ++d)
    electrons->R[0][d] += electrons->getLattice().R(1, d);
  electrons->update();
  const auto wrapped = target.evaluate(*electrons);
  for (std::size_t element = 0; element < frozen.size(); ++element)
    CHECK(std::abs(wrapped[element] - frozen[element]) < 1e-10);
  electrons->R[0] = position;
  electrons->update();
  ions->R[0][0] += 0.01;
  CHECK_THROWS_AS(target.evaluate(*electrons), std::invalid_argument);
  ions->R[0][0] -= 0.01;
  electrons->setTwist(ParticleSet::PosType{0.1, 0, 0});
  CHECK_THROWS_AS(target.evaluate(*electrons), std::invalid_argument);
}
#endif

TEST_CASE("Periodic occupied SPO fixed-point VGL export",
          "[.periodic_spo_oracle][wavefunction][psiformer]")
{
  const char* xml_path = std::getenv("PSIFORMER_SPO_XML");
  const char* oracle_path = std::getenv("PSIFORMER_SPO_ORACLE");
  const char* output_path = std::getenv("PSIFORMER_SPO_OUTPUT");
  if (!xml_path || !oracle_path || !output_path)
    SKIP("Set PSIFORMER_SPO_XML, PSIFORMER_SPO_ORACLE and PSIFORMER_SPO_OUTPUT for this diagnostic");
  const char* requested_name = std::getenv("PSIFORMER_SPO_NAME");
  const char* complex_input = std::getenv("PSIFORMER_SPO_COMPLEX_ORACLE");
  const bool complex_oracle = complex_input && std::string(complex_input) == "1";
  REQUIRE((!complex_input || std::string(complex_input) == "0" || complex_oracle));
  Libxml2Document document;
  REQUIRE(document.parse(xml_path));
  auto* root = document.getRoot();
  REQUIRE(root != nullptr);
  if (xmlStrEqual(root->name, BAD_CAST "simulation"))
    for (auto* child = root->children; child; child = child->next)
      if (xmlStrEqual(child->name, BAD_CAST "qmcsystem"))
      {
        root = child;
        break;
      }
  ParticleSetPool pool(OHMMS::Controller);
  bool cell_found = false;
  for (auto* child = root->children; child; child = child->next)
    if (xmlStrEqual(child->name, BAD_CAST "simulationcell"))
    {
      pool.readSimulationCellXML(child);
      cell_found = true;
    }
  REQUIRE(cell_found);
  for (auto* child = root->children; child; child = child->next)
    if (xmlStrEqual(child->name, BAD_CAST "particleset"))
      REQUIRE(pool.put(child));
  auto* electrons = pool.getParticleSet("e");
  REQUIRE(electrons != nullptr);
  const char* target_twist_input = std::getenv("PSIFORMER_SPO_TARGET_TWIST");
  std::array<double, 3> target_twist{};
  if (target_twist_input)
  {
    std::istringstream fields(target_twist_input);
    REQUIRE(static_cast<bool>(fields >> target_twist[0] >> target_twist[1] >> target_twist[2]));
    electrons->setTwist(ParticleSet::PosType{target_twist[0], target_twist[1], target_twist[2]});
  }
  xmlNodePtr wavefunction = nullptr;
  for (auto* child = root->children; child; child = child->next)
    if (xmlStrEqual(child->name, BAD_CAST "wavefunction"))
    {
      REQUIRE(wavefunction == nullptr);
      wavefunction = child;
    }
  REQUIRE(wavefunction != nullptr);
  WaveFunctionFactory factory(*electrons, pool.getPool(), OHMMS::Controller);
  auto state = factory.buildTWF(wavefunction, RuntimeOptions{});
  REQUIRE(state != nullptr);
  auto orbitals = state->getSPOSet(requested_name ? requested_name : "spo-up").makeClone();
  REQUIRE(orbitals != nullptr);
  const auto count = orbitals->getOrbitalSetSize();
  REQUIRE(count > 0);
#ifdef QMC_COMPLEX
  std::unique_ptr<ComplexOrbitalTargetEvaluator> target;
  if (target_twist_input)
  {
    const auto* slater = dynamic_cast<const SlaterDet*>(state->getOrbitals()[0].get());
    REQUIRE(slater != nullptr);
    const auto& source = dynamic_cast<const LCAOrbitalSet&>(state->getSPOSet(requested_name ? requested_name : "spo-up"));
    const auto n = electrons->R.size();
    const auto cap = (n * n + 2 * count + 2 * source.C->size()) * sizeof(std::complex<double>);
    CHECK_THROWS_AS(ComplexOrbitalTargetEvaluator::fromSingleReference(*slater, *electrons, target_twist, 1, cap - 1),
                    std::length_error);
    target = std::make_unique<ComplexOrbitalTargetEvaluator>(
        ComplexOrbitalTargetEvaluator::fromSingleReference(*slater, *electrons, target_twist, 1, cap));
    CHECK(target->retainedScalarBytes() == cap);
  }
#else
  REQUIRE(target_twist_input == nullptr);
#endif
  SPOSet::ValueVector values(count), laplacians(count);
  SPOSet::GradVector gradients(count);
  std::ifstream oracle(oracle_path);
  REQUIRE(oracle.good());
  std::ofstream output(output_path);
  REQUIRE(output.good());
  output << std::setprecision(17) << (complex_oracle
      ? "# x y z occupied_column real_imag_value real_imag_gx real_imag_gy real_imag_gz real_imag_laplacian\n"
      : "# x y z occupied_column real_value real_gx real_gy real_gz real_laplacian\n");
  std::array<double, 5> maximum_error{};
  double maximum_imaginary_value = 0;
  std::size_t rows = 0;
  std::string line;
  while (std::getline(oracle, line))
  {
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#')
      continue;
    std::istringstream fields(line);
    ParticleSet::PosType position;
    double column_value;
    std::array<std::complex<double>, 5> expected;
    REQUIRE(static_cast<bool>(fields >> position[0] >> position[1] >> position[2] >> column_value));
    // Oracle tables may write integral column indices in scientific notation; consume the complete token.
    REQUIRE(column_value >= 0);
    REQUIRE(column_value < count);
    const int column = static_cast<int>(column_value);
    REQUIRE(static_cast<double>(column) == column_value);
    // Complex oracle rows explicitly interleave real/imaginary values for all five physical products.
    for (auto& product : expected)
    {
      double real, imaginary = 0;
      REQUIRE(static_cast<bool>(fields >> real));
      if (complex_oracle) REQUIRE(static_cast<bool>(fields >> imaginary));
      product = {real, imaginary};
    }
    REQUIRE(column >= 0);
    REQUIRE(column < count);
    for (int d = 0; d < 3; ++d)
      REQUIRE(std::isfinite(position[d]));
    electrons->R[0] = position;
    electrons->update();
    orbitals->evaluateVGL(*electrons, 0, values, gradients, laplacians);
    const std::array<QMCTraits::ValueType, 5> actual{
        values[column], gradients[column][0], gradients[column][1], gradients[column][2], laplacians[column]};
#ifdef QMC_COMPLEX
    if (target)
    {
      const auto momentum = electrons->getLattice().k_cart(electrons->getTwist());
      const double angle = -dot(momentum, position);
      const std::complex<double> phase{std::cos(angle), std::sin(angle)};
      // Remove the common row twist once, retaining the occupied-column complex gauge.
      CHECK(std::abs(target->evaluate(*electrons)[column] - phase * values[column]) < 1e-11);
    }
#endif
    output << position[0] << ' ' << position[1] << ' ' << position[2] << ' ' << column;
    for (std::size_t product = 0; product < actual.size(); ++product)
    {
      REQUIRE(std::isfinite(std::real(actual[product])));
      REQUIRE(std::isfinite(std::imag(actual[product])));
      REQUIRE(std::isfinite(expected[product].real()));
      REQUIRE(std::isfinite(expected[product].imag()));
      maximum_error[product] = std::max(maximum_error[product], std::abs(actual[product] - expected[product]));
      output << ' ' << std::real(actual[product]);
      if (complex_oracle) output << ' ' << std::imag(actual[product]);
    }
    maximum_imaginary_value = std::max(maximum_imaginary_value, std::abs(std::imag(values[column])));
    output << '\n';
    ++rows;
  }
  REQUIRE(rows > 0);
  if (complex_oracle) CHECK(maximum_imaginary_value > 1e-8);
#ifdef QMC_COMPLEX
  if (target)
  {
    const auto frozen = target->evaluate(*electrons);
    const auto fingerprint = target->descriptor().fingerprint();
    const auto& source = dynamic_cast<const LCAOrbitalSet&>(state->getSPOSet(requested_name ? requested_name : "spo-up"));
    const auto coefficient = source.C->data()[0];
    source.C->data()[0] += std::complex<double>{0.17, -0.12};
    CHECK(target->evaluate(*electrons) == frozen);
    source.C->data()[0] = coefficient;
    // The periodic-part target must wrap without a common Bloch phase.
    for (int d = 0; d < 3; ++d)
      electrons->R[0][d] += electrons->getLattice().R(1, d);
    electrons->update();
    const auto wrapped = target->evaluate(*electrons);
    for (std::size_t element = 0; element < frozen.size(); ++element)
      CHECK(std::abs(wrapped[element] - frozen[element]) < 1e-9);
    CHECK(target->descriptor().fingerprint() == fingerprint);
    electrons->setTwist(ParticleSet::PosType{target_twist[0] + 0.01, target_twist[1], target_twist[2]});
    CHECK_THROWS_AS(target->evaluate(*electrons), std::invalid_argument);
    const auto* slater = dynamic_cast<const SlaterDet*>(state->getOrbitals()[0].get());
    const std::array<double, 3> mismatched{target_twist[0] + 0.01, target_twist[1], target_twist[2]};
    CHECK_THROWS_AS(ComplexOrbitalTargetEvaluator::fromSingleReference(
        *slater, *electrons, mismatched, 1, target->retainedScalarBytes()), std::invalid_argument);
  }
#endif
  output.flush();
  REQUIRE(output.good());
  // This diagnostic reports export error; scientific acceptance uses the fixed campaign tolerances.
  app_log() << "Periodic SPO oracle rows " << rows << " max absolute errors value/gx/gy/gz/laplacian";
  for (double error : maximum_error)
    app_log() << ' ' << error;
  app_log() << std::endl;
}

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

TEST_CASE("PsiFormer general overlap target has stable frozen-source identity",
          "[wavefunction][psiformer][overlap]")
{
  const auto descriptor = DeterminantOverlapTargetDescriptor::makeGeneralFrozenTeacher(
      "frozen-trial-wavefunction", 4);
  CHECK(descriptor.mode() == DeterminantOverlapTargetMode::GENERAL_FROZEN_TEACHER);
  CHECK(descriptor.channelCount() == 4);
  CHECK(descriptor.sourceOrdinals().empty());
  CHECK(descriptor.fingerprint() ==
        DeterminantOverlapTargetDescriptor::makeGeneralFrozenTeacher(
            "frozen-trial-wavefunction", 4).fingerprint());
  CHECK(descriptor.fingerprint() !=
        DeterminantOverlapTargetDescriptor::makeGeneralFrozenTeacher(
            "different-frozen-source", 4).fingerprint());
  CHECK_THROWS_AS(DeterminantOverlapTargetDescriptor::makeGeneralFrozenTeacher("", 4),
                  std::invalid_argument);
  CHECK_THROWS_AS(DeterminantOverlapTargetDescriptor::makeGeneralFrozenTeacher("source", 0),
                  std::invalid_argument);
}

} // namespace qmcplusplus::psiformer
