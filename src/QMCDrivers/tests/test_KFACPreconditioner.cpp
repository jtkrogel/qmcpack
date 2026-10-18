//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_KFACPreconditioner.cpp
 * @brief Deterministic dense-oracle tests for bounded KFAC statistics and solves.
 */

#include "QMCDrivers/WFTrain/KFACPreconditioner.h"
#include "QMCDrivers/WFTrain/PsiFormerKFAC.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "Particle/ParticleSet.h"
#include "Utilities/for_testing/Catch2Approx.h"

#include "Message/Communicate.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Build a weight/bias/fallback/frozen schema for compact registry tests.
StructuredParameterSchema makeSchema(ParameterScalarDomain domain =
                                         ParameterScalarDomain::REAL64)
{
  return {"kfac/test",
          {{"affine/w", {2, 2}, 0, 4, domain, true, "network"},
           {"affine/b", {2}, 4, 2, domain, true, "network"},
           {"envelope", {2}, 6, 2, domain, true, "network"},
           {"frozen", {1}, 8, 1, domain, false, "network"}}};
}

/// Build the one-affine, one-diagonal-fallback registry used by most tests.
KFACBlockRegistry makeRegistry()
{
  return {makeSchema(), {{"dense", 0, 1, 2, 2}}, {2}};
}

/// Convert initialized PsiFormer tensor layouts to the production schema convention.
StructuredParameterSchema makePsiFormerSchema(
    const psiformer::InitializedPsiFormerParameters& initialized,
    std::string provider_id = "psiformer/registry-test",
    const std::vector<std::size_t>& trainable_offsets = {})
{
  std::vector<ParameterBlockDescriptor> blocks;
  blocks.reserve(initialized.layouts.size());
  for (const psiformer::ParameterLayoutInput& layout : initialized.layouts)
    blocks.push_back(
        {layout.module + "/" + layout.name, layout.shape, layout.begin,
         layout.end - layout.begin, ParameterScalarDomain::REAL64,
         trainable_offsets.empty() ||
             std::find(trainable_offsets.begin(), trainable_offsets.end(), layout.begin) !=
                 trainable_offsets.end(),
         "neural_network"});
  return {std::move(provider_id), std::move(blocks)};
}

/// Add one complete deterministic sample to an accumulator.
void addSample(KFACFactorAccumulator& accumulator,
               DerivativeReal weight,
               const std::array<DerivativeReal, 4>& activations,
               const std::array<DerivativeReal, 4>& sensitivities,
               const std::array<DerivativeReal, 9>& score)
{
  accumulator.beginSample(weight);
  accumulator.addObservation(
      {0, {activations.data(), activations.size()},
       {sensitivities.data(), sensitivities.size()}, 2});
  accumulator.addFallbackScores({score.data(), score.size()});
  accumulator.endSample();
}

/// Solve one symmetric 2-by-2 system independently of production Cholesky code.
std::array<DerivativeReal, 2> solve2(const std::array<DerivativeReal, 4>& matrix,
                                    std::array<DerivativeReal, 2> rhs)
{
  const DerivativeReal determinant = matrix[0] * matrix[3] - matrix[1] * matrix[2];
  return {(matrix[3] * rhs[0] - matrix[1] * rhs[1]) / determinant,
          (-matrix[2] * rhs[0] + matrix[0] * rhs[1]) / determinant};
}

} // namespace

TEST_CASE("KFAC registry validates complete real block coverage",
          "[drivers][wftrain][kfac]")
{
  const KFACBlockRegistry registry = makeRegistry();
  REQUIRE(registry.affineBlocks().size() == 1);
  CHECK(registry.affineBlocks()[0].bias_block == 1);
  REQUIRE(registry.fallbackBlocks().size() == 1);
  CHECK(registry.fallbackBlocks()[0] == 2);
  CHECK(registry.fingerprint().size() == 16);

  // The frozen block needs no factor and remains outside the registry.
  CHECK_NOTHROW(KFACBlockRegistry(makeSchema(), {{"dense", 0, 1, 2, 2}}, {2}));
  CHECK_THROWS(KFACBlockRegistry(makeSchema(), {{"dense", 0, 1, 2, 2}}, {}));
  CHECK_THROWS(KFACBlockRegistry(makeSchema(), {{"bad", 0, 1, 3, 2}}, {2}));
  CHECK_THROWS(KFACBlockRegistry(makeSchema(ParameterScalarDomain::COMPLEX128),
                                 {{"dense", 0, 1, 2, 2}}, {2}));
}

TEST_CASE("PsiFormer KFAC registry classifies only true affine leaves",
          "[drivers][wftrain][kfac][psiformer]")
{
  using namespace qmcplusplus::psiformer;
  const ModelShape shape{/*spin_up_electrons=*/2, /*spin_down_electrons=*/2,
                         /*nuclei=*/2, /*determinants=*/16,
                         /*feature_dimension=*/256, /*attention_heads=*/4,
                         /*attention_blocks=*/4};
  const InitializedPsiFormerParameters initialized =
      initializePsiFormerParameters(shape, 17);
  const PsiFormerExecutionPlan plan(initialized.model_shape, initialized.layouts);
  const StructuredParameterSchema schema = makePsiFormerSchema(initialized);
  const KFACBlockRegistry registry = makePsiFormerKFACRegistry(plan, schema);

  CHECK(registry.affineBlocks().size() == 27);
  CHECK(registry.fallbackBlocks().size() == 6);
  CHECK(registry.genericBlocks().size() == 2);
  const auto backflow = std::find_if(
      registry.affineBlocks().begin(), registry.affineBlocks().end(),
      [](const auto& block) { return block.id == "backflow_up_weight"; });
  REQUIRE(backflow != registry.affineBlocks().end());
  CHECK(backflow->input_width == 256);
  CHECK(backflow->output_width == 64);
  const auto biased = std::find_if(
      registry.affineBlocks().begin(), registry.affineBlocks().end(),
      [](const auto& block) { return block.id == "update_hidden_weight/block_0"; });
  REQUIRE(biased != registry.affineBlocks().end());
  CHECK(biased->bias_block != NO_KFAC_BIAS_BLOCK);
}

TEST_CASE("PsiFormer KFAC adapter maps reverse-order observations without copies",
          "[drivers][wftrain][kfac][psiformer]")
{
  using namespace qmcplusplus::psiformer;
  const ModelShape shape{/*spin_up_electrons=*/2, /*spin_down_electrons=*/2,
                         /*nuclei=*/2, /*determinants=*/16,
                         /*feature_dimension=*/256, /*attention_heads=*/4,
                         /*attention_blocks=*/4};
  const InitializedPsiFormerParameters initialized = initializePsiFormerParameters(shape, 19);
  const PsiFormerExecutionPlan plan(initialized.model_shape, initialized.layouts);
  const std::vector<std::size_t> trainable_offsets{
      plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).begin,
      plan.parameter(ParameterRole::UPDATE_OUTPUT_WEIGHT, 0).begin,
      plan.parameter(ParameterRole::UPDATE_OUTPUT_BIAS, 0).begin,
      plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT).begin};
  const StructuredParameterSchema schema =
      makePsiFormerSchema(initialized, "psiformer/adapter-test", trainable_offsets);
  KFACFactorAccumulator accumulator(makePsiFormerKFACRegistry(plan, schema), 5);
  PsiFormerKFACObservationSink sink(plan, accumulator);

  std::array<DerivativeReal, 1024> activations{};
  std::array<DerivativeReal, 1024> sensitivities{};
  auto observe = [&](ParameterRole role, std::size_t block, std::size_t rows,
                     std::size_t input, std::size_t output) {
    sink.observe({role, block, activations.data(), sensitivities.data(),
                  rows, input, output});
  };

  accumulator.beginSample(2.0);
  observe(ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK, 2, 256, 64);
  observe(ParameterRole::UPDATE_OUTPUT_WEIGHT, 0, 4, 256, 256);
  observe(ParameterRole::ELECTRON_EMBEDDING_WEIGHT, NO_ATTENTION_BLOCK, 4, 9, 256);
  accumulator.endSample();

  REQUIRE(accumulator.factors().size() == 3);
  for (const KFACFactorStatistics& factor : accumulator.factors())
  {
    CHECK(factor.sample_weight_sum == Catch::Approx(2.0));
    CHECK((factor.row_count == 2 || factor.row_count == 4));
  }

  accumulator.beginSample(1.0);
  CHECK_THROWS_AS(
      sink.observe({ParameterRole::ELECTRON_EMBEDDING_WEIGHT, NO_ATTENTION_BLOCK,
                    activations.data(), sensitivities.data(), 4, 8, 256}),
      std::invalid_argument);
  accumulator.abortSample();
  CHECK(accumulator.sampleCount() == 1);
}

TEST_CASE("Live PsiFormer provider exposes generic cusp blocks to the driver registry",
          "[drivers][wftrain][kfac][psiformer]")
{
  using namespace qmcplusplus::psiformer;
  const SimulationCell cell;
  ParticleSet electrons(cell), ions(cell);
  electrons.setName("e");
  electrons.create({2, 2});
  auto& electron_species = electrons.getSpeciesSet();
  electron_species.addSpecies("u");
  electron_species.addSpecies("d");
  const int mass = electron_species.addAttribute("mass");
  electron_species(mass, 0) = electron_species(mass, 1) = 1.0;
  electrons.resetGroups();
  ions.setName("ion0");
  ions.create({1, 1});
  auto& ion_species = ions.getSpeciesSet();
  ion_species.addSpecies("Li");
  ion_species.addSpecies("H");
  const int charge = ion_species.addAttribute("charge");
  ion_species(charge, 0) = 3.0;
  ion_species(charge, 1) = 1.0;
  ions.resetGroups();
  ions.R[0] = ParticleSet::PosType{0.0, 0.0, 0.0};
  ions.R[1] = ParticleSet::PosType{3.0, 0.0, 0.0};
  for (int i = 0; i < 4; ++i)
    electrons.R[i] = ParticleSet::PosType{0.2 * (i + 1), 0.1, 0.3};
  const ModelShape shape{2, 2, 2, 16, 256, 4, 4};
  auto initialized = initializePsiFormerParameters(shape, 31);
  const PsiFormerExecutionPlan plan(initialized.model_shape, initialized.layouts);
  PsiFormerWF component("metadata", std::move(initialized), electrons, ions);
  const auto generic = component.genericKFACParameterBlocks();
  REQUIRE(generic.size() == 2);
  const auto& schema = component.parameterSchema();
  const auto& same = plan.parameter(ParameterRole::CUSP_SAME_ALPHA);
  const auto& opposite = plan.parameter(ParameterRole::CUSP_OPPOSITE_ALPHA);
  CHECK(schema.blocks()[generic[0]].offset == same.begin);
  CHECK(schema.blocks()[generic[1]].offset == opposite.begin);
  // Exercise the same provider metadata seam used by the native driver.
  const auto affine_metadata = component.affineParameterBlocks();
  std::vector<KFACAffineBlockDescriptor> affine;
  std::vector<bool> covered(schema.blocks().size(), false);
  for (const auto& block : affine_metadata)
  {
    affine.push_back({block.id, block.weight_block, block.bias_block, block.input_width, block.output_width});
    covered[block.weight_block] = true;
    if (block.bias_block != NO_KFAC_BIAS_BLOCK)
      covered[block.bias_block] = true;
  }
  std::vector<std::size_t> fallback;
  for (std::size_t i = 0; i < covered.size(); ++i)
    if (schema.blocks()[i].trainable && !covered[i])
      fallback.push_back(i);
  const KFACBlockRegistry live_registry(schema, std::move(affine), std::move(fallback), generic);
  CHECK(live_registry.genericBlocks() == makePsiFormerKFACRegistry(plan, schema).genericBlocks());
}

TEST_CASE("PsiFormer rejects absent spin sectors before affine observation",
          "[drivers][wftrain][kfac][psiformer]")
{
  using namespace qmcplusplus::psiformer;
  const ModelShape shape{/*spin_up_electrons=*/2, /*spin_down_electrons=*/2,
                         /*nuclei=*/2, /*determinants=*/16,
                         /*feature_dimension=*/256, /*attention_heads=*/4,
                         /*attention_blocks=*/4};
  const InitializedPsiFormerParameters initialized = initializePsiFormerParameters(shape, 23);
  ModelShape invalid_shape = shape;
  invalid_shape.spin_up_electrons = shape.electrons();
  invalid_shape.spin_down_electrons = 0;
  CHECK_THROWS_AS(PsiFormerExecutionPlan(invalid_shape, initialized.layouts),
                  std::invalid_argument);
}

TEST_CASE("KFAC streams repeated rows with sample rather than row normalization",
          "[drivers][wftrain][kfac]")
{
  KFACFactorAccumulator accumulator(makeRegistry(), 7);
  const std::array<DerivativeReal, 4> activations0{1.0, 0.0, 0.0, 2.0};
  const std::array<DerivativeReal, 4> sensitivities0{2.0, 0.0, 0.0, 1.0};
  const std::array<DerivativeReal, 9> score0{0, 0, 0, 0, 0, 0, 3.0, 4.0, 99.0};
  addSample(accumulator, 1.0, activations0, sensitivities0, score0);

  const std::array<DerivativeReal, 4> activations1{2.0, 0.0, 0.0, 1.0};
  const std::array<DerivativeReal, 4> sensitivities1{1.0, 0.0, 0.0, 2.0};
  const std::array<DerivativeReal, 9> score1{0, 0, 0, 0, 0, 0, 1.0, 2.0, -81.0};
  addSample(accumulator, 3.0, activations1, sensitivities1, score1);

  REQUIRE(accumulator.sampleCount() == 2);
  REQUIRE(accumulator.factors().size() == 1);
  const KFACFactorStatistics& factor = accumulator.factors()[0];
  CHECK(factor.sample_weight_sum == Catch::Approx(4.0));
  CHECK(factor.row_weight_sum == Catch::Approx(8.0));
  CHECK(factor.row_count == 4);
  // Bias augmentation orientation is [input0,input1,one].
  const std::array<DerivativeReal, 9> expected_a{
      13.0, 0.0, 7.0,
      0.0, 7.0, 5.0,
      7.0, 5.0, 8.0};
  for (std::size_t index = 0; index < expected_a.size(); ++index)
    CHECK(factor.activation_outer_sum[index] == Catch::Approx(expected_a[index]));
  CHECK(factor.sensitivity_outer_sum[0] == Catch::Approx(7.0));
  CHECK(factor.sensitivity_outer_sum[3] == Catch::Approx(13.0));
  CHECK(factor.sensitivity_outer_sum[1] == Catch::Approx(0.0));

  const auto fallback = accumulator.fallbackDiagonalSum();
  CHECK(accumulator.fallbackWeightSum() == Catch::Approx(4.0));
  REQUIRE(fallback.size() == 2);
  CHECK(fallback[0] == Catch::Approx(12.0));
  CHECK(fallback[1] == Catch::Approx(28.0));
}

TEST_CASE("KFAC preconditioner matches a dense factored solve and repeat scaling",
          "[drivers][wftrain][kfac]")
{
  KFACFactorAccumulator accumulator(makeRegistry(), 3);
  const std::array<DerivativeReal, 4> activations{1.0, 0.0, 0.0, 2.0};
  const std::array<DerivativeReal, 4> sensitivities{2.0, 0.0, 0.0, 1.0};
  const std::array<DerivativeReal, 9> score{0, 0, 0, 0, 0, 0, 3.0, 4.0, 10.0};
  addSample(accumulator, 1.0, activations, sensitivities, score);
  reduceKFACFactorStatistics(accumulator);

  const KFACPreconditioner preconditioner(
      accumulator, {/*damping=*/0.04, /*fallback_damping=*/0.5});
  const std::vector<DerivativeValue> gradient{
      {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {123, 0}};
  std::vector<DerivativeValue> actual(gradient.size());
  const StructuredParameterVectorConstView view(
      accumulator.registry().parameterSchema(), 3, {gradient.data(), gradient.size()});
  preconditioner.apply(view, {actual.data(), actual.size()});

  // A=sum(a a^T)/B and G=sum(g g^T)/B.  With R=2, split damping is
  // sqrt(lambda*R), and the complete inverse action carries the factor R.
  const DerivativeReal split = std::sqrt(0.08);
  const std::array<DerivativeReal, 9> a{
      1.0 + split, 0.0, 1.0,
      0.0, 4.0 + split, 2.0,
      1.0, 2.0, 2.0 + split};
  const std::array<DerivativeReal, 4> g{4.0 + split, 0.0, 0.0, 1.0 + split};

  // Independent Gaussian elimination for the 3-by-3 left solve.
  auto solve3 = [](std::array<DerivativeReal, 9> matrix,
                   std::array<DerivativeReal, 3> rhs) {
    for (std::size_t column = 0; column < 3; ++column)
    {
      const DerivativeReal pivot = matrix[column * 3 + column];
      for (std::size_t entry = column; entry < 3; ++entry)
        matrix[column * 3 + entry] /= pivot;
      rhs[column] /= pivot;
      for (std::size_t row = 0; row < 3; ++row)
        if (row != column)
        {
          const DerivativeReal scale = matrix[row * 3 + column];
          for (std::size_t entry = column; entry < 3; ++entry)
            matrix[row * 3 + entry] -= scale * matrix[column * 3 + entry];
          rhs[row] -= scale * rhs[column];
        }
    }
    return rhs;
  };
  std::array<DerivativeReal, 6> left{};
  for (std::size_t output = 0; output < 2; ++output)
  {
    const auto column = solve3(a, {gradient[output].real(),
                                    gradient[2 + output].real(),
                                    gradient[4 + output].real()});
    for (std::size_t input = 0; input < 3; ++input)
      left[input * 2 + output] = column[input];
  }
  std::array<DerivativeReal, 6> expected{};
  for (std::size_t input = 0; input < 3; ++input)
  {
    const auto row = solve2(g, {left[input * 2], left[input * 2 + 1]});
    expected[input * 2] = 2.0 * row[0];
    expected[input * 2 + 1] = 2.0 * row[1];
  }
  for (std::size_t parameter = 0; parameter < 6; ++parameter)
    CHECK(actual[parameter].real() == Catch::Approx(expected[parameter]).epsilon(1e-11));

  CHECK(actual[6].real() == Catch::Approx(7.0 / (9.0 + 0.5)));
  CHECK(actual[7].real() == Catch::Approx(8.0 / (16.0 + 0.5)));
  CHECK(actual[8] == DerivativeValue{});
  const auto storage = preconditioner.storageDiagnostics();
  CHECK(storage.parameter_count == 9);
  CHECK(storage.affine_block_count == 1);
  CHECK(storage.factor_elements == 13);
}

TEST_CASE("KFAC sample and reduction failures preserve completed statistics",
          "[drivers][wftrain][kfac]")
{
  KFACFactorAccumulator accumulator(makeRegistry(), 2);
  const std::array<DerivativeReal, 4> values{1.0, 0.0, 0.0, 1.0};
  const std::array<DerivativeReal, 9> score{};
  addSample(accumulator, 1.0, values, values, score);
  const auto baseline = accumulator.factors()[0].activation_outer_sum;

  accumulator.beginSample(2.0);
  accumulator.addObservation(
      {0, {values.data(), values.size()}, {values.data(), values.size()}, 2});
  CHECK_THROWS(accumulator.endSample());
  CHECK(accumulator.sampleCount() == 1);
  CHECK(accumulator.factors()[0].activation_outer_sum == baseline);
  CHECK_FALSE(accumulator.sampleActive());

  CHECK_THROWS(accumulator.beginSample(-1.0));
  accumulator.beginSample(1.0);
  std::array<DerivativeReal, 4> nonfinite = values;
  nonfinite[0] = std::numeric_limits<DerivativeReal>::quiet_NaN();
  CHECK_THROWS(accumulator.addObservation(
      {0, {nonfinite.data(), nonfinite.size()}, {values.data(), values.size()}, 2}));
  accumulator.abortSample();
  CHECK(accumulator.factors()[0].activation_outer_sum == baseline);

  accumulator.beginSample(1.0);
  std::array<DerivativeReal, 4> overflowing{};
  overflowing.fill(std::numeric_limits<DerivativeReal>::max());
  accumulator.addObservation(
      {0, {overflowing.data(), overflowing.size()},
       {values.data(), values.size()}, 2});
  accumulator.addFallbackScores({score.data(), score.size()});
  CHECK_THROWS_AS(accumulator.endSample(), std::overflow_error);
  CHECK(accumulator.sampleCount() == 1);
  CHECK(accumulator.factors()[0].activation_outer_sum == baseline);

  reduceKFACFactorStatistics(accumulator, nullptr, 2);
  CHECK_THROWS(reduceKFACFactorStatistics(accumulator));
  const std::size_t bytes = accumulator.retainedNumericBytes();
  CHECK(bytes > 0);
}

TEST_CASE("Registered Gaussian curvature ignores objective weights and scales sensitivities",
          "[drivers][wftrain][kfac]")
{
  KFACFactorAccumulator gaussian(makeRegistry(), 4, KFACCurvatureConvention::REGISTERED_GAUSSIAN);
  KFACFactorAccumulator equal_weights(makeRegistry(), 4, KFACCurvatureConvention::REGISTERED_GAUSSIAN);
  const std::array<DerivativeReal, 4> x{1, 0, 0, 2};
  const std::array<DerivativeReal, 4> dy{2, 0, 0, 1};
  const std::array<DerivativeReal, 9> score{0, 0, 0, 0, 0, 0, 3, 4, 0};
  addSample(gaussian, 0.0, x, dy, score);
  addSample(gaussian, 100.0, x, dy, score);
  addSample(equal_weights, 1.0, x, dy, score);
  addSample(equal_weights, 1.0, x, dy, score);
  CHECK(gaussian.factors()[0].activation_outer_sum == equal_weights.factors()[0].activation_outer_sum);
  CHECK(gaussian.factors()[0].sensitivity_outer_sum == equal_weights.factors()[0].sensitivity_outer_sum);
  const auto& factor = gaussian.factors()[0];
  CHECK(factor.sample_weight_sum == Catch::Approx(2));
  CHECK(factor.row_weight_sum == Catch::Approx(4));
  CHECK(factor.activation_outer_sum[0] / factor.sample_weight_sum == Catch::Approx(1));
  CHECK(factor.sensitivity_outer_sum[0] / factor.sample_weight_sum == Catch::Approx(8));
  CHECK(factor.sensitivity_outer_sum[3] / factor.sample_weight_sum == Catch::Approx(2));
  CHECK(gaussian.fallbackDiagonalSum()[0] / gaussian.fallbackWeightSum() == Catch::Approx(18));
  CHECK(gaussian.fallbackDiagonalSum()[1] / gaussian.fallbackWeightSum() == Catch::Approx(32));
}

TEST_CASE("Registered generic diagonal squares the sample sum and preserves replication scaling",
          "[drivers][wftrain][kfac]")
{
  const StructuredParameterSchema schema{"generic/test",
      {{"envelope", {1}, 0, 1, ParameterScalarDomain::REAL64, true, "network"},
       {"generic", {1}, 1, 1, ParameterScalarDomain::REAL64, true, "network"}}};
  const KFACBlockRegistry registry(schema, {}, {0, 1}, {1});
  CHECK_THROWS(KFACBlockRegistry(schema, {}, {0, 1}, {1, 1}));
  CHECK_THROWS(KFACBlockRegistry(schema, {}, {0, 1}, {2}));
  for (int replication : {1, 3})
  {
    KFACFactorAccumulator statistics(registry, 0, KFACCurvatureConvention::REGISTERED_GAUSSIAN);
    auto add = [&](DerivativeReal envelope, DerivativeReal generic) {
      const std::array<DerivativeReal, 2> score{envelope, generic};
      statistics.beginSample(17.0);
      statistics.addFallbackScores({score.data(), score.size()});
      statistics.endSample();
    };
    for (int i = 0; i < replication; ++i)
    {
      add(3, 1);
      add(4, 2);
    }
    reduceKFACFactorStatistics(statistics);
    CHECK(statistics.fallbackDiagonalSum()[0] / statistics.fallbackWeightSum() == Catch::Approx(25));
    CHECK(statistics.fallbackDiagonalSum()[1] / statistics.fallbackWeightSum() == Catch::Approx(9 * replication));
    const KFACPreconditioner inverse(statistics, {0.1, 0.5});
    const std::array<DerivativeValue, 2> gradient{{1.0, 1.0}};
    std::array<DerivativeValue, 2> result{};
    inverse.apply({schema, 0, {gradient.data(), gradient.size()}}, {result.data(), result.size()});
    CHECK(result[0].real() == Catch::Approx(1 / 25.5));
    CHECK(result[1].real() == Catch::Approx(1 / (9.0 * replication + 0.5)));
  }
  KFACFactorAccumulator cancellation(registry, 0, KFACCurvatureConvention::REGISTERED_GAUSSIAN);
  for (DerivativeReal sign : {1.0, -1.0})
  {
    const std::array<DerivativeReal, 2> score{1.0, sign};
    cancellation.beginSample(1.0);
    cancellation.addFallbackScores({score.data(), score.size()});
    cancellation.endSample();
  }
  CHECK(cancellation.fallbackDiagonalSum()[1] == 0.0);
  const auto baseline = cancellation.fallbackDiagonalSum()[0];
  cancellation.beginSample(1.0);
  const std::array<DerivativeReal, 2> overflow{1, std::numeric_limits<DerivativeReal>::max()};
  cancellation.addFallbackScores({overflow.data(), overflow.size()});
  CHECK_THROWS_AS(cancellation.endSample(), std::overflow_error);
  CHECK(cancellation.sampleCount() == 2);
  CHECK(cancellation.fallbackDiagonalSum()[0] == baseline);
  CHECK(cancellation.fallbackDiagonalSum()[1] == 0.0);
}

TEST_CASE("Trace-balanced KFAC matches independent anisotropic two-factor solves",
          "[drivers][wftrain][kfac]")
{
  const StructuredParameterSchema schema{"trace/test",
      {{"weight", {2, 2}, 0, 4, ParameterScalarDomain::REAL64, true, "network"}}};
  KFACFactorAccumulator statistics({schema, {{"affine", 0, NO_KFAC_BIAS_BLOCK, 2, 2}}, {}}, 3);
  const std::array<DerivativeReal, 4> x{2, 0, 1, 1}, dy{1, 0, 1, 1};
  statistics.beginSample(1);
  statistics.addObservation({0, {x.data(), x.size()}, {dy.data(), dy.size()}, 2});
  statistics.endSample();
  const KFACPreconditioner inverse(statistics, {0.04, 0.04, KFACDampingPolicy::TRACE_BALANCED});
  const std::array<DerivativeValue, 4> gradient{{1., 2., 3., 4.}};
  std::array<DerivativeValue, 4> actual{};
  inverse.apply({schema, 3, {gradient.data(), gradient.size()}}, {actual.data(), actual.size()});
  // A=[[5,1],[1,1]], B=[[2,1],[1,1]], R=2. Mean traces3 and1.5;
  // effective damping .08 gives independent diagonal shifts .4 and .2.
  const std::array<DerivativeReal, 4> a{5.4, 1, 1, 1.4}, b{2.2, 1, 1, 1.2};
  std::array<DerivativeReal, 4> left{};
  for (std::size_t column = 0; column < 2; ++column)
  {
    const auto solved = solve2(a, {gradient[column].real(), gradient[2 + column].real()});
    left[column] = solved[0];
    left[2 + column] = solved[1];
  }
  for (std::size_t row = 0; row < 2; ++row)
  {
    const auto solved = solve2(b, {left[2 * row], left[2 * row + 1]});
    CHECK(actual[2 * row].real() == Catch::Approx(2 * solved[0]).epsilon(1e-10));
    CHECK(actual[2 * row + 1].real() == Catch::Approx(2 * solved[1]).epsilon(1e-10));
  }
}

TEST_CASE("Trace-balanced KFAC handles scalar and zero factors as damped represented operators",
          "[drivers][wftrain][kfac]")
{
  for (const auto dimensions : {std::array<std::size_t, 2>{1, 1}, {1, 2}, {2, 1}, {2, 2}})
    for (bool zero_factor : {false, true})
    {
      const std::size_t n = dimensions[0], m = dimensions[1];
      const StructuredParameterSchema schema{"scalar/test",
          {{"weight", {n, m}, 0, n * m, ParameterScalarDomain::REAL64, true, "network"}}};
      KFACFactorAccumulator statistics({schema, {{"affine", 0, NO_KFAC_BIAS_BLOCK, n, m}}, {}}, 0);
      std::vector<DerivativeReal> x(2 * n, 0.0), dy(2 * m, 0.0);
      // N=1, two repeated rows, diagonal factors; scalar axes reuse both rows.
      x[0] = zero_factor ? 0.0 : 2.0;
      x[n + (n == 1 ? 0 : 1)] = zero_factor ? 0.0 : 2.0;
      dy[0] = 3.0;
      dy[m + (m == 1 ? 0 : 1)] = 3.0;
      statistics.beginSample(1);
      statistics.addObservation({0, {x.data(), x.size()}, {dy.data(), dy.size()}, 2});
      statistics.endSample();
      const KFACPreconditioner inverse(statistics, {0.25, 0.25, KFACDampingPolicy::TRACE_BALANCED});
      const std::vector<DerivativeValue> gradient(n * m, 1.0);
      std::vector<DerivativeValue> actual(n * m);
      inverse.apply({schema, 0, {gradient.data(), gradient.size()}}, {actual.data(), actual.size()});
      const DerivativeReal a = n == 1 ? 8.0 : 4.0, b = m == 1 ? 18.0 : 9.0;
      // Scalar special cases give exact product+lambda. For equal diagonal
      // nonscalar factors the factored approximation retains its cross terms.
      const DerivativeReal expected = zero_factor ? 4.0 :
          (n == 1 || m == 1 ? 2.0 / (a * b + 0.5) :
           2.0 / ((a + std::sqrt(0.5 * a / b)) * (b + std::sqrt(0.5 * b / a))));
      for (const auto value : actual)
        CHECK(value.real() == Catch::Approx(expected).epsilon(1e-10));
    }
}

TEST_CASE("KFAC factor EMA normalizes each batch and retains its running weight",
          "[drivers][wftrain][kfac][kfac_state]")
{
  const StructuredParameterSchema schema{"ema/generic",
      {{"envelope", {1}, 0, 1, ParameterScalarDomain::REAL64, true, "network"},
       {"generic", {1}, 1, 1, ParameterScalarDomain::REAL64, true, "network"}}};
  const KFACBlockRegistry registry(schema, {}, {0, 1}, {1});
  const auto batch = [&](const std::vector<std::array<DerivativeReal, 2>>& scores) {
    KFACFactorAccumulator statistics(registry, 0, KFACCurvatureConvention::REGISTERED_GAUSSIAN);
    for (const auto& score : scores)
    {
      statistics.beginSample(17);
      statistics.addFallbackScores({score.data(), score.size()});
      statistics.endSample();
    }
    reduceKFACFactorStatistics(statistics);
    return statistics;
  };
  const auto first_batch = batch({{3, 1}, {4, 2}});
  const auto first = updateKFACFactorState(nullptr, first_batch, 0.5);
  CHECK(first.normalizer == 1);
  CHECK(first.update_count == 1);
  CHECK(first.values[0] == Catch::Approx(25));
  CHECK(first.values[1] == Catch::Approx(9));
  // A larger batch has the same unit estimator weight. Generic curvature is
  // averaged AFTER each batch's signed-score square, not squared across updates.
  const auto second_batch = batch({{1, 1}, {1, -1}, {1, 1}, {1, -1}});
  const auto second = updateKFACFactorState(&first, second_batch, 0.5);
  CHECK(second.normalizer == Catch::Approx(1.5));
  CHECK(second.update_count == 2);
  CHECK(second.values[0] == Catch::Approx((0.5 * 25 + 2) / 1.5));
  CHECK(second.values[1] == Catch::Approx(3));
  CHECK(first.values[0] == 25);
  CHECK(first.values[1] == 9);
  const auto third_batch = batch({{2, 2}});
  const auto third = updateKFACFactorState(&second, third_batch, 0.5);
  CHECK(third.normalizer == Catch::Approx(1.75));
  CHECK(third.values[0] == Catch::Approx((0.25 * 25 + 0.5 * 2 + 8) / 1.75));
  CHECK(third.values[1] == Catch::Approx((0.25 * 9 + 8) / 1.75));
  // A copied serialized payload resumes exactly, including the EMA normalizer.
  const KFACFactorState serialized = second;
  const auto resumed = updateKFACFactorState(&serialized, third_batch, 0.5);
  CHECK(resumed.normalizer == third.normalizer);
  CHECK(resumed.values == third.values);
  CHECK(updateKFACFactorState(&second, third_batch, 0).values ==
        updateKFACFactorState(nullptr, third_batch, 0).values);
  auto invalid = second;
  invalid.normalizer = 0;
  CHECK_THROWS(validateKFACFactorState(registry, invalid));
  invalid = second;
  invalid.values.pop_back();
  CHECK_THROWS(updateKFACFactorState(&invalid, third_batch, 0.5));
  invalid = second;
  invalid.values[0] = std::numeric_limits<DerivativeReal>::infinity();
  CHECK_THROWS(validateKFACFactorState(registry, invalid));
  CHECK_THROWS(updateKFACFactorState(&second, third_batch, -0.1));
  KFACFactorAccumulator incomplete(registry, 0, KFACCurvatureConvention::REGISTERED_GAUSSIAN);
  CHECK_THROWS(updateKFACFactorState(nullptr, incomplete, 0.5));
}

TEST_CASE("KFAC inverse snapshots rebind exactly and refresh only current diagonal blocks",
          "[drivers][wftrain][kfac][kfac_state]")
{
  const auto registry = makeRegistry();
  const std::array<DerivativeReal, 4> x{1, 0, 0, 2}, dy{2, 0, 0, 1};
  const std::array<DerivativeReal, 9> first_score{0, 0, 0, 0, 0, 0, 3, 4, 0};
  KFACFactorAccumulator first_batch(registry, 0, KFACCurvatureConvention::REGISTERED_GAUSSIAN);
  addSample(first_batch, 1, x, dy, first_score);
  reduceKFACFactorStatistics(first_batch);
  const auto factors = updateKFACFactorState(nullptr, first_batch, 0.95);
  const KFACPreconditionerControl control{0.1, 0.2, KFACDampingPolicy::TRACE_BALANCED};
  const KFACPreconditioner direct(first_batch, control);
  const KFACPreconditioner from_factors(registry, 4, factors, control);
  const auto cache = from_factors.exportInverseState();
  const KFACPreconditioner rebound(registry, 7, cache);
  const std::array<DerivativeValue, 9> gradient{{1., 2., 3., 4., 5., 6., 7., 8., 9.}};
  const auto apply = [&](const KFACPreconditioner& inverse, std::size_t version) {
    std::vector<DerivativeValue> result(gradient.size());
    inverse.apply({registry.parameterSchema(), version, {gradient.data(), gradient.size()}},
                   {result.data(), result.size()});
    return result;
  };
  CHECK(apply(direct, 0) == apply(from_factors, 4));
  CHECK(apply(from_factors, 4) == apply(rebound, 7));
  CHECK(rebound.exportInverseState().values == cache.values);
  CHECK_THROWS(apply(rebound, 4));
  KFACFactorAccumulator second_batch(registry, 1, KFACCurvatureConvention::REGISTERED_GAUSSIAN);
  const std::array<DerivativeReal, 9> second_score{0, 0, 0, 0, 0, 0, 1, 2, 0};
  const std::array<DerivativeReal, 4> changed_x{2, 0, 0, 3};
  addSample(second_batch, 1, changed_x, dy, second_score);
  reduceKFACFactorStatistics(second_batch);
  const auto next = updateKFACFactorState(&factors, second_batch, 0.95);
  KFACPreconditioner stale(registry, 8, cache);
  stale.refreshFallbackDiagonal(next, 0.7);
  const auto actual = apply(stale, 8), prior = apply(rebound, 7);
  for (std::size_t i = 0; i < 6; ++i)
    CHECK(actual[i] == prior[i]);
  CHECK(actual[6].real() == Catch::Approx(7 / ((0.95 * 18 + 2) / 1.95 + 0.7)));
  CHECK(actual[7].real() == Catch::Approx(8 / ((0.95 * 32 + 8) / 1.95 + 0.7)));
  CHECK(actual[8] == DerivativeValue{});
  const auto updated_cache = stale.exportInverseState();
  CHECK(updated_cache.fallback_damping == 0.7);
  const KFACPreconditioner resumed(registry, 8, updated_cache);
  CHECK(apply(resumed, 8) == actual);
  auto invalid = cache;
  invalid.values[0] = 0;
  CHECK_THROWS(KFACPreconditioner(registry, 0, invalid));
  invalid = cache;
  invalid.values[1] = 1; // Upper-triangle contamination cannot be a valid cache.
  CHECK_THROWS(KFACPreconditioner(registry, 0, invalid));
  invalid = cache;
  invalid.registry_fingerprint = "foreign";
  CHECK_THROWS(KFACPreconditioner(registry, 0, invalid));
  const auto unchanged = stale.exportInverseState().values;
  CHECK_THROWS(stale.refreshFallbackDiagonal(next, -1));
  CHECK(stale.exportInverseState().values == unchanged);
}

#ifdef HAVE_MPI
TEST_CASE("Distributed registered generic curvature squares the global signed sum",
          "[drivers][wftrain][kfac][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  const StructuredParameterSchema schema{"global/generic",
      {{"generic", {1}, 0, 1, ParameterScalarDomain::REAL64, true, "network"}}};
  KFACFactorAccumulator statistics({schema, {}, {0}, {0}}, 0,
                                   KFACCurvatureConvention::REGISTERED_GAUSSIAN);
  const DerivativeReal rank_score = communicator.rank() % 2 == 0 ? 1.0 : -1.0;
  const std::array<DerivativeReal, 1> score{rank_score};
  for (int sample = 0; sample <= communicator.rank(); ++sample)
  {
    statistics.beginSample(communicator.rank() + 1);
    statistics.addFallbackScores({score.data(), score.size()});
    statistics.endSample();
  }
  reduceKFACFactorStatistics(statistics, &communicator, 1);
  DerivativeReal global_sum = 0;
  for (int rank = 0; rank < communicator.size(); ++rank)
    global_sum += (rank % 2 == 0 ? 1.0 : -1.0) * (rank + 1);
  CHECK(statistics.fallbackDiagonalSum()[0] == Catch::Approx(2 * global_sum * global_sum));
  CHECK(statistics.fallbackWeightSum() ==
        Catch::Approx(communicator.size() * (communicator.size() + 1) / 2.0));
}

TEST_CASE("Distributed KFAC reduces unequal weighted populations before normalization",
          "[drivers][wftrain][kfac][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  KFACFactorAccumulator distributed(makeRegistry(), 11);
  const DerivativeReal rank_value = static_cast<DerivativeReal>(communicator.rank() + 1);
  const std::array<DerivativeReal, 4> activations{rank_value, 0.0, 0.0, 1.0};
  const std::array<DerivativeReal, 4> sensitivities{1.0, 0.0, 0.0, rank_value};
  const std::array<DerivativeReal, 9> score{0, 0, 0, 0, 0, 0,
                                           rank_value, 2.0 * rank_value, 0.0};
  addSample(distributed, rank_value, activations, sensitivities, score);
  reduceKFACFactorStatistics(distributed, &communicator, 2);

  KFACFactorAccumulator reference(makeRegistry(), 11);
  for (int rank = 0; rank < communicator.size(); ++rank)
  {
    const DerivativeReal value = static_cast<DerivativeReal>(rank + 1);
    const std::array<DerivativeReal, 4> rank_activations{value, 0.0, 0.0, 1.0};
    const std::array<DerivativeReal, 4> rank_sensitivities{1.0, 0.0, 0.0, value};
    const std::array<DerivativeReal, 9> rank_score{0, 0, 0, 0, 0, 0,
                                                  value, 2.0 * value, 0.0};
    addSample(reference, value, rank_activations, rank_sensitivities, rank_score);
  }
  reduceKFACFactorStatistics(reference, nullptr, 2);

  CHECK(distributed.sampleCount() == reference.sampleCount());
  REQUIRE(distributed.factors().size() == reference.factors().size());
  CHECK(distributed.factors()[0].sample_weight_sum ==
        Catch::Approx(reference.factors()[0].sample_weight_sum));
  CHECK(distributed.factors()[0].row_weight_sum ==
        Catch::Approx(reference.factors()[0].row_weight_sum));
  CHECK(distributed.factors()[0].activation_outer_sum ==
        reference.factors()[0].activation_outer_sum);
  CHECK(distributed.factors()[0].sensitivity_outer_sum ==
        reference.factors()[0].sensitivity_outer_sum);
  CHECK(distributed.fallbackDiagonalSum()[0] ==
        Catch::Approx(reference.fallbackDiagonalSum()[0]));
}
#endif

} // namespace qmcplusplus::wftrain
