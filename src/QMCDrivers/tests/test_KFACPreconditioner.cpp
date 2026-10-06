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

#ifdef HAVE_MPI
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
