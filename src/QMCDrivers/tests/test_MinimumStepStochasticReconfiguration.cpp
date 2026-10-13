//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_MinimumStepStochasticReconfiguration.cpp
 * @brief Dense-oracle and transactional tests for minSR and SPRING.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "QMCDrivers/WFTrain/MinimumStepStochasticReconfiguration.h"
#include "io/hdf/hdf_archive.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

using Catch::Approx;

/// Construct the two-parameter schema used by every dense oracle.
StructuredParameterSchema makeSchema()
{
  return {"minsr/toy",
          {{"weights", {2}, 0, 2, ParameterScalarDomain::REAL64, true,
            "weights"}}};
}

/// Stream three fixed score rows without retaining any optimizer-owned matrix.
class ToyScoreOperator final : public StreamingDerivativeOperator
{
public:
  ToyScoreOperator(const StructuredParameterSchema& schema, std::size_t version)
      : schema_(schema), version_(version), plan_(schema_, version_, 1)
  {}

  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities result;
    result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
        derivativeProductBit(DerivativeProduct::SCORE_JVP);
    result.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::HERMITIAN);
    result.parameter_scalar_domain = ParameterScalarDomain::REAL64;
    result.result_scalar_domain = ParameterScalarDomain::COMPLEX128;
    result.reduction_domain = ReductionDomain::CROWD_LOCAL;
    result.execution_domain = DerivativeExecutionDomain::HOST;
    result.maximum_vjp_channels = 1;
    result.maximum_parameter_chunk_size = 1;
    result.maximum_sample_tile_size = 3;
    result.block_streaming = true;
    return result;
  }

  const StructuredParameterSchema& parameterSchema() const noexcept override
  {
    return schema_;
  }
  std::size_t parameterVersion() const noexcept override { return version_; }
  std::size_t batchOrdinal() const noexcept override { return 0; }
  std::size_t sampleOffset() const noexcept override { return 0; }
  std::size_t sampleCount() const noexcept override { return scores_.size(); }
  const ParameterChunkPlan& parameterChunkPlan() const noexcept override
  {
    return plan_;
  }

protected:
  void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                    DerivativeAdjoint,
                    ParameterReductionSink& sink) const override
  {
    for (const ParameterChunkDescriptor& chunk : plan_.chunks())
    {
      DerivativeValue value{};
      for (std::size_t sample = 0; sample < scores_.size(); ++sample)
        value += channels[0].coefficients.values[sample] *
            scores_[sample][chunk.parameter_offset];
      sink.add(0, {chunk, {&value, 1}});
    }
  }

  void evaluateScoreJVP(const StructuredParameterVectorConstView& direction,
                        SampleProductSink& sink) const override
  {
    std::array<DerivativeValue, 3> products{};
    for (std::size_t sample = 0; sample < scores_.size(); ++sample)
      for (std::size_t parameter = 0; parameter < 2; ++parameter)
        products[sample] += scores_[sample][parameter] *
            direction.values()[parameter];
    const SampleProductTileDescriptor descriptor{
        schema_.providerId(), schema_.fingerprint(), version_, 0, 0, 3, 0};
    sink.add({descriptor, {products.data(), products.size()}});
  }

private:
  const StructuredParameterSchema& schema_;
  std::size_t version_;
  ParameterChunkPlan plan_;
  const std::array<std::array<double, 2>, 3> scores_{
      {{{1.0, 0.0}}, {{0.0, 1.0}}, {{1.0, 1.0}}}};
};

/// Prepare one weighted score factor with a nonuniform sample population.
std::unique_ptr<ScoreCovarianceFactor> makeFactor(
    const StructuredParameterSchema& schema,
    const ToyScoreOperator& derivative_operator,
    std::size_t version)
{
  static const std::array<DerivativeReal, 3> weights{1.0, 2.0, 1.0};
  static const std::array<DerivativeValue, 3> energies{
      DerivativeValue{1.0, 0.0}, DerivativeValue{3.0, 0.0},
      DerivativeValue{2.0, 0.0}};
  const ScoreCovarianceBatch batch{
      &derivative_operator, {weights.data(), weights.size()},
      {energies.data(), energies.size()}};
  return std::make_unique<ScoreCovarianceFactor>(
      schema, version, DerivativeArrayView<const ScoreCovarianceBatch>{&batch, 1});
}

/// Return a globally reduced objective view matching A^T residual.
ParameterGradientView objectiveView(const StructuredParameterSchema& schema,
                                    std::size_t version,
                                    const std::array<DerivativeReal, 2>& gradient)
{
  return {schema.fingerprint(), version, ReductionDomain::GLOBAL,
          {gradient.data(), gradient.size()}};
}

/// Construct a three-parameter schema for nonsymmetric PII dense oracles.
StructuredParameterSchema makePIISchema()
{
  return {"pii/dense",
          {{"weights", {3}, 0, 3, ParameterScalarDomain::REAL64, true,
            "weights"}}};
}

/** Stream deterministic pseudorandom score and local-energy Jacobian rows.
 *
 * Distinct matrices make the shifted factor nonsymmetric with respect to the
 * score factor, so a mistaken O*B^T MinPII action cannot pass by symmetry.
 */
class DensePIIDerivativeOperator final : public StreamingDerivativeOperator
{
public:
  DensePIIDerivativeOperator(const StructuredParameterSchema& schema,
                             std::size_t version,
                             std::uint32_t local_energy_term_mask =
                                 localEnergyTermBit(LocalEnergyTerm::KINETIC))
      : schema_(schema), version_(version), plan_(schema_, version_, 2),
        local_energy_term_mask_(local_energy_term_mask)
  {
    std::mt19937_64 generator(0x5eed1234ULL);
    std::uniform_real_distribution<DerivativeReal> distribution(-1.25, 1.25);
    for (auto& row : score_)
      for (DerivativeReal& value : row)
        value = distribution(generator);
    for (auto& row : local_energy_jacobian_)
      for (DerivativeReal& value : row)
        value = distribution(generator);
  }

  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities result;
    result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
        derivativeProductBit(DerivativeProduct::SCORE_JVP) |
        derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_JVP);
    result.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::HERMITIAN);
    result.parameter_scalar_domain = ParameterScalarDomain::REAL64;
    result.result_scalar_domain = ParameterScalarDomain::COMPLEX128;
    result.reduction_domain = ReductionDomain::CROWD_LOCAL;
    result.execution_domain = DerivativeExecutionDomain::HOST;
    result.local_energy_term_mask = local_energy_term_mask_;
    result.maximum_vjp_channels = 1;
    result.maximum_parameter_chunk_size = 2;
    result.maximum_sample_tile_size = sample_count;
    result.block_streaming = true;
    return result;
  }

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }
  std::size_t parameterVersion() const noexcept override { return version_; }
  std::size_t batchOrdinal() const noexcept override { return 0; }
  std::size_t sampleOffset() const noexcept override { return 0; }
  std::size_t sampleCount() const noexcept override { return sample_count; }
  const ParameterChunkPlan& parameterChunkPlan() const noexcept override { return plan_; }

  const auto& score() const noexcept { return score_; }
  const auto& localEnergyJacobian() const noexcept { return local_energy_jacobian_; }

protected:
  void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                    DerivativeAdjoint,
                    ParameterReductionSink& sink) const override
  {
    for (const ParameterChunkDescriptor& chunk : plan_.chunks())
    {
      std::vector<DerivativeValue> values(chunk.count);
      for (std::size_t local = 0; local < chunk.count; ++local)
        for (std::size_t sample = 0; sample < sample_count; ++sample)
          values[local] += channels[0].coefficients.values[sample] *
              score_[sample][chunk.parameter_offset + local];
      sink.add(0, {chunk, {values.data(), values.size()}});
    }
  }

  void evaluateScoreJVP(const StructuredParameterVectorConstView& direction,
                        SampleProductSink& sink) const override
  {
    emitProduct(score_, direction, sink);
  }

  void evaluateLocalEnergyJVP(const StructuredParameterVectorConstView& direction,
                              std::uint32_t term_mask,
                              SampleProductSink& sink) const override
  {
    if (term_mask != local_energy_term_mask_)
      throw std::logic_error("Dense PII oracle received an unexpected term mask");
    emitProduct(local_energy_jacobian_, direction, sink);
  }

private:
  void emitProduct(const std::array<std::array<DerivativeReal, 3>, 5>& matrix,
                   const StructuredParameterVectorConstView& direction,
                   SampleProductSink& sink) const
  {
    std::array<DerivativeValue, sample_count> products{};
    for (std::size_t sample = 0; sample < sample_count; ++sample)
      for (std::size_t parameter = 0; parameter < 3; ++parameter)
        products[sample] += matrix[sample][parameter] * direction.values()[parameter];
    const SampleProductTileDescriptor descriptor{
        schema_.providerId(), schema_.fingerprint(), version_, 0, 0,
        sample_count, 0};
    sink.add({descriptor, {products.data(), products.size()}});
  }

  static constexpr std::size_t sample_count = 5;
  const StructuredParameterSchema& schema_;
  std::size_t version_;
  ParameterChunkPlan plan_;
  std::uint32_t local_energy_term_mask_ = 0;
  std::array<std::array<DerivativeReal, 3>, sample_count> score_{};
  std::array<std::array<DerivativeReal, 3>, sample_count> local_energy_jacobian_{};
};

/// Solve one small dense real system by pivoted Gaussian elimination.
std::vector<DerivativeReal> solveDense(std::vector<std::vector<DerivativeReal>> matrix,
                                      std::vector<DerivativeReal> right_hand_side)
{
  const std::size_t size = matrix.size();
  for (std::size_t column = 0; column < size; ++column)
  {
    std::size_t pivot = column;
    for (std::size_t row = column + 1; row < size; ++row)
      if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column]))
        pivot = row;
    REQUIRE(std::abs(matrix[pivot][column]) > 1.0e-13);
    std::swap(matrix[pivot], matrix[column]);
    std::swap(right_hand_side[pivot], right_hand_side[column]);
    for (std::size_t row = column + 1; row < size; ++row)
    {
      const DerivativeReal multiplier = matrix[row][column] / matrix[column][column];
      for (std::size_t entry = column; entry < size; ++entry)
        matrix[row][entry] -= multiplier * matrix[column][entry];
      right_hand_side[row] -= multiplier * right_hand_side[column];
    }
  }
  std::vector<DerivativeReal> solution(size);
  for (std::size_t reverse = size; reverse > 0; --reverse)
  {
    const std::size_t row = reverse - 1;
    DerivativeReal value = right_hand_side[row];
    for (std::size_t column = row + 1; column < size; ++column)
      value -= matrix[row][column] * solution[column];
    solution[row] = value / matrix[row][row];
  }
  return solution;
}

} // namespace

TEST_CASE("Weighted score factor agrees with a dense centered oracle",
          "[wftrain][minsr]")
{
  const StructuredParameterSchema schema = makeSchema();
  constexpr std::size_t version = 4;
  ToyScoreOperator derivative_operator(schema, version);
  const auto factor = makeFactor(schema, derivative_operator, version);

  CHECK(factor->meanEnergy().real() == Approx(2.25));
  CHECK(factor->localSampleVectorSize() == 3);
  CHECK(factor->globalSampleVectorSize() == 3);
  CHECK(factor->residual()[0].real() == Approx(-1.25));
  CHECK(factor->residual()[1].real() == Approx(1.0606601717798212));
  CHECK(factor->residual()[2].real() == Approx(-0.25));

  std::array<DerivativeValue, 2> direction{{{2.0, 0.0}, {-1.0, 0.0}}};
  const StructuredParameterVectorConstView direction_view(
      schema, version, {direction.data(), direction.size()});
  std::array<DerivativeValue, 3> product{};
  factor->apply(direction_view, {product.data(), product.size()});
  CHECK(product[0].real() == Approx(0.875));
  CHECK(product[1].real() == Approx(-0.8838834764831844));
  CHECK(product[2].real() == Approx(0.375));

  std::array<DerivativeValue, 3> sample_direction{
      {{1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}}};
  std::array<DerivativeValue, 2> adjoint{};
  factor->applyAdjoint({sample_direction.data(), sample_direction.size()},
                       {adjoint.data(), adjoint.size()});
  CHECK(adjoint[0].real() == Approx(-0.7071067811865476));
  CHECK(adjoint[1].real() == Approx(-0.1464466094067262));

  const std::vector<DerivativeReal> gradient = factor->scoreCovarianceGradient();
  CHECK(gradient[0] == Approx(-0.75));
  CHECK(gradient[1] == Approx(0.625));
  const auto storage = factor->storageDiagnostics();
  CHECK(storage.complex_parameter_vectors == 2);
  CHECK(storage.local_sample_vector_size == 3);
  CHECK(storage.retained_numeric_bytes > 0);
}

TEST_CASE("Sample PCG and minSR reproduce a dense damped solve",
          "[wftrain][minsr]")
{
  const StructuredParameterSchema schema = makeSchema();
  constexpr std::size_t version = 4;
  ToyScoreOperator derivative_operator(schema, version);
  const auto factor = makeFactor(schema, derivative_operator, version);

  MinimumStepSampleOperator sample_operator(*factor, 0.1, 1.0);
  KrylovSolverControl krylov;
  krylov.relative_tolerance = 1.0e-12;
  krylov.maximum_iterations = 20;
  krylov.maximum_operator_applications = 21;
  const DistributedSampleKrylovResult solve = solveDistributedSamplePCG(
      sample_operator, factor->residual(), krylov);
  REQUIRE(solve.converged);
  std::array<DerivativeValue, 2> dense_direction{};
  factor->applyAdjoint({solve.solution.data(), solve.solution.size()},
                       {dense_direction.data(), dense_direction.size()});
  CHECK(dense_direction[0].real() == Approx(-1.6176470588235294).epsilon(1.0e-11));
  CHECK(dense_direction[1].real() == Approx(1.4705882352941178).epsilon(1.0e-11));

  MinimumStepUpdateControl control;
  control.learning_rate = 0.2;
  control.damping = 0.1;
  control.center_shift = 1.0;
  control.krylov = krylov;
  MinimumStepSRUpdateRule updater(*factor, control);
  const std::array<DerivativeReal, 2> gradient{-0.75, 0.625};
  const StructuredParameterSnapshot parameters{schema.fingerprint(), version,
                                                {0.0, 0.0}};
  const StructuredParameterSnapshot candidate = updater.propose(
      schema, parameters, objectiveView(schema, version, gradient));
  CHECK(candidate.values[0] == Approx(0.3235294117647059).epsilon(1.0e-11));
  CHECK(candidate.values[1] == Approx(-0.2941176470588236).epsilon(1.0e-11));
  updater.proposalRejected();

  const std::array<DerivativeReal, 2> foreign_gradient{-0.7, 0.625};
  CHECK_THROWS(updater.propose(
      schema, parameters, objectiveView(schema, version, foreign_gradient)));
}

TEST_CASE("SPRING mu zero equals minSR and commits history transactionally",
          "[wftrain][spring]")
{
  const StructuredParameterSchema schema = makeSchema();
  constexpr std::size_t version = 4;
  ToyScoreOperator derivative_operator(schema, version);
  const auto factor = makeFactor(schema, derivative_operator, version);
  const std::array<DerivativeReal, 2> gradient{-0.75, 0.625};
  const auto objective = objectiveView(schema, version, gradient);
  const StructuredParameterSnapshot parameters{schema.fingerprint(), version,
                                                {0.0, 0.0}};

  SPRINGUpdateControl control;
  control.history_decay = 0.0;
  control.minimum_step.learning_rate = 0.2;
  control.minimum_step.damping = 0.1;
  control.minimum_step.krylov.relative_tolerance = 1.0e-12;
  control.minimum_step.krylov.maximum_iterations = 20;
  control.minimum_step.krylov.maximum_operator_applications = 21;
  SPRINGUpdateRule spring(*factor, control);
  CHECK(spring.retainedBytes() == 4 * sizeof(DerivativeValue));
  const StructuredParameterSnapshot candidate = spring.propose(
      schema, parameters, objective);
  CHECK(candidate.values[0] == Approx(0.3235294117647059).epsilon(1.0e-11));
  CHECK(candidate.values[1] == Approx(-0.2941176470588236).epsilon(1.0e-11));
  spring.proposalRejected();
  CHECK(spring.history()[0] == DerivativeValue{});
  CHECK(spring.history()[1] == DerivativeValue{});

  spring.propose(schema, parameters, objective);
  spring.proposalAccepted(schema, parameters, objective);
  CHECK(spring.acceptedUpdateCount() == 1);
  CHECK(spring.history()[0].real() == Approx(-1.6176470588235294).epsilon(1.0e-11));
  CHECK(spring.history()[1].real() == Approx(1.4705882352941178).epsilon(1.0e-11));

  spring.unbindFactor();
  CHECK_THROWS(spring.propose(schema, parameters, objective));
  spring.bindFactor(*factor);
  CHECK_NOTHROW(spring.propose(schema, parameters, objective));
  spring.proposalRejected();
}

TEST_CASE("SPRING checkpoint payload restores committed history exactly",
          "[wftrain][spring][checkpoint]")
{
  const StructuredParameterSchema schema = makeSchema();
  constexpr std::size_t version = 4;
  ToyScoreOperator derivative_operator(schema, version);
  const auto factor = makeFactor(schema, derivative_operator, version);
  const std::array<DerivativeReal, 2> gradient{-0.75, 0.625};
  const auto objective = objectiveView(schema, version, gradient);
  const StructuredParameterSnapshot parameters{schema.fingerprint(), version,
                                                {0.0, 0.0}};
  SPRINGUpdateControl control;
  control.history_decay = 0.5;
  control.minimum_step.damping = 0.1;
  control.minimum_step.krylov.relative_tolerance = 1.0e-12;
  control.minimum_step.krylov.maximum_iterations = 20;
  control.minimum_step.krylov.maximum_operator_applications = 21;
  SPRINGUpdateRule source(*factor, control);
  source.propose(schema, parameters, objective);
  source.proposalAccepted(schema, parameters, objective);
  const StructuredParameterSnapshot second_candidate =
      source.propose(schema, parameters, objective);
  CHECK(second_candidate.values[0] == Approx(1.7830882352941178).epsilon(1.0e-11));
  CHECK(second_candidate.values[1] == Approx(-1.6544117647058822).epsilon(1.0e-11));
  source.proposalAccepted(schema, parameters, objective);

  const std::filesystem::path path = "spring_optimizer_test.h5";
  {
    hdf_archive archive;
    REQUIRE(archive.create(path));
    archive.push("optimizer");
    source.writeCheckpointPayload(archive);
    archive.pop();
    archive.close();
  }
  SPRINGUpdateRule restored(*factor, control);
  {
    hdf_archive archive;
    REQUIRE(archive.open(path, H5F_ACC_RDONLY));
    archive.push("optimizer", false);
    auto action = restored.prepareCheckpointRestore(
        archive, source.checkpointMetadata());
    archive.pop();
    archive.close();
    action->commit();
  }
  std::filesystem::remove(path);
  CHECK(restored.acceptedUpdateCount() == source.acceptedUpdateCount());
  CHECK(restored.history() == source.history());
  CHECK(restored.checkpointMetadata().state_fingerprint ==
        source.checkpointMetadata().state_fingerprint);
}

TEST_CASE("PII and MinPII match dense nonsymmetric operators and solves",
          "[wftrain][pii][minpii]")
{
  const StructuredParameterSchema schema = makePIISchema();
  constexpr std::size_t version = 9;
  constexpr std::size_t samples = 5;
  constexpr std::size_t parameters = 3;
  const std::array<DerivativeReal, samples> weights{1.0, 2.0, 0.5, 3.0, 1.5};
  const std::array<DerivativeValue, samples> energies{{
      {-1.7, 0.0}, {-0.9, 0.0}, {0.4, 0.0}, {-2.1, 0.0}, {1.2, 0.0}}};
  DensePIIDerivativeOperator derivative_operator(schema, version);
  const ScoreCovarianceBatch batch{
      &derivative_operator, {weights.data(), weights.size()},
      {energies.data(), energies.size()}};
  ScoreCovarianceFactor score_factor(
      schema, version, DerivativeArrayView<const ScoreCovarianceBatch>{&batch, 1});
  const std::uint32_t kinetic = localEnergyTermBit(LocalEnergyTerm::KINETIC);
  ProjectedInverseIterationFactor factor(score_factor, kinetic);

  constexpr DerivativeReal tau = -1.3;
  constexpr DerivativeReal damping = 0.23;
  const DerivativeReal weight_sum = 8.0;
  std::array<DerivativeReal, parameters> mean_score{};
  for (std::size_t sample = 0; sample < samples; ++sample)
    for (std::size_t parameter = 0; parameter < parameters; ++parameter)
      mean_score[parameter] += weights[sample] *
          derivative_operator.score()[sample][parameter] / weight_sum;
  std::array<std::array<DerivativeReal, parameters>, samples> score{};
  std::array<std::array<DerivativeReal, parameters>, samples> shifted{};
  for (std::size_t sample = 0; sample < samples; ++sample)
  {
    const DerivativeReal sqrt_q = std::sqrt(weights[sample] / weight_sum);
    for (std::size_t parameter = 0; parameter < parameters; ++parameter)
    {
      score[sample][parameter] = sqrt_q *
          (derivative_operator.score()[sample][parameter] - mean_score[parameter]);
      // A*v is dE_L[v] + E_L*(centered score JVP); B=A-tau*O.
      shifted[sample][parameter] = sqrt_q *
          derivative_operator.localEnergyJacobian()[sample][parameter] +
          (energies[sample].real() - tau) * score[sample][parameter];
    }
  }

  const std::array<DerivativeValue, parameters> parameter_direction{{
      {0.7, 0.0}, {-0.4, 0.0}, {1.1, 0.0}}};
  const StructuredParameterVectorConstView parameter_view(
      schema, version, {parameter_direction.data(), parameter_direction.size()});
  std::array<DerivativeValue, samples> shifted_action{};
  factor.applyShifted(parameter_view, tau,
                      {shifted_action.data(), shifted_action.size()});
  for (std::size_t sample = 0; sample < samples; ++sample)
  {
    DerivativeReal expected = 0.0;
    for (std::size_t parameter = 0; parameter < parameters; ++parameter)
      expected += shifted[sample][parameter] * parameter_direction[parameter].real();
    CHECK(shifted_action[sample].real() == Approx(expected).margin(2.0e-13));
  }

  ProjectedInverseIterationOperator full_operator(factor, tau, damping);
  std::array<DerivativeValue, parameters> full_action{};
  full_operator.apply(parameter_view, {full_action.data(), full_action.size()});
  for (std::size_t row = 0; row < parameters; ++row)
  {
    DerivativeReal expected = damping * parameter_direction[row].real();
    for (std::size_t sample = 0; sample < samples; ++sample)
      for (std::size_t column = 0; column < parameters; ++column)
        expected += score[sample][row] * shifted[sample][column] *
            parameter_direction[column].real();
    CHECK(full_action[row].real() == Approx(expected).margin(5.0e-13));
  }

  const std::array<DerivativeValue, samples> sample_direction{{
      {0.3, 0.0}, {-0.8, 0.0}, {1.4, 0.0}, {0.2, 0.0}, {-0.6, 0.0}}};
  MinimumProjectedInverseIterationOperator minimum_operator(factor, tau, damping);
  std::array<DerivativeValue, samples> minimum_action{};
  minimum_operator.apply({sample_direction.data(), sample_direction.size()},
                         {minimum_action.data(), minimum_action.size()});
  DerivativeReal rejected_orientation_difference = 0.0;
  for (std::size_t row = 0; row < samples; ++row)
  {
    DerivativeReal expected = damping * sample_direction[row].real();
    DerivativeReal rejected = damping * sample_direction[row].real();
    for (std::size_t column = 0; column < samples; ++column)
      for (std::size_t parameter = 0; parameter < parameters; ++parameter)
      {
        expected += shifted[row][parameter] * score[column][parameter] *
            sample_direction[column].real();
        rejected += score[row][parameter] * shifted[column][parameter] *
            sample_direction[column].real();
      }
    CHECK(minimum_action[row].real() == Approx(expected).margin(7.0e-13));
    rejected_orientation_difference += std::abs(expected - rejected);
  }
  CHECK(rejected_orientation_difference > 0.1);

  std::vector<std::vector<DerivativeReal>> full_matrix(
      parameters, std::vector<DerivativeReal>(parameters));
  for (std::size_t row = 0; row < parameters; ++row)
    for (std::size_t column = 0; column < parameters; ++column)
    {
      full_matrix[row][column] = row == column ? damping : 0.0;
      for (std::size_t sample = 0; sample < samples; ++sample)
        full_matrix[row][column] += score[sample][row] * shifted[sample][column];
    }
  std::vector<std::vector<DerivativeReal>> minimum_matrix(
      samples, std::vector<DerivativeReal>(samples));
  for (std::size_t row = 0; row < samples; ++row)
    for (std::size_t column = 0; column < samples; ++column)
    {
      minimum_matrix[row][column] = row == column ? damping : 0.0;
      for (std::size_t parameter = 0; parameter < parameters; ++parameter)
        minimum_matrix[row][column] += shifted[row][parameter] * score[column][parameter];
    }
  std::vector<DerivativeReal> sample_rhs(samples);
  for (std::size_t sample = 0; sample < samples; ++sample)
    sample_rhs[sample] = 0.5 * score_factor.residual()[sample].real();
  std::vector<DerivativeReal> parameter_rhs(parameters);
  for (std::size_t parameter = 0; parameter < parameters; ++parameter)
    for (std::size_t sample = 0; sample < samples; ++sample)
      parameter_rhs[parameter] += score[sample][parameter] * sample_rhs[sample];
  const std::vector<DerivativeReal> dense_full = solveDense(full_matrix, parameter_rhs);
  const std::vector<DerivativeReal> dense_sample = solveDense(minimum_matrix, sample_rhs);

  GMRESSolverControl gmres;
  gmres.restart_length = 5;
  gmres.krylov.relative_tolerance = 1.0e-12;
  gmres.krylov.maximum_iterations = 40;
  gmres.krylov.maximum_operator_applications = 50;
  std::vector<DerivativeValue> complex_parameter_rhs(parameters);
  for (std::size_t parameter = 0; parameter < parameters; ++parameter)
    complex_parameter_rhs[parameter] = {parameter_rhs[parameter], 0.0};
  const StructuredParameterVectorConstView rhs_view(
      schema, version, {complex_parameter_rhs.data(), complex_parameter_rhs.size()});
  const KrylovSolveResult full_solve = solveRestartedGMRES(
      full_operator, rhs_view, gmres);
  REQUIRE(full_solve.converged);
  std::vector<DerivativeValue> complex_sample_rhs(samples);
  for (std::size_t sample = 0; sample < samples; ++sample)
    complex_sample_rhs[sample] = {sample_rhs[sample], 0.0};
  const DistributedSampleKrylovResult sample_solve = solveDistributedSampleGMRES(
      minimum_operator, {complex_sample_rhs.data(), complex_sample_rhs.size()}, gmres);
  REQUIRE(sample_solve.converged);
  std::array<DerivativeValue, parameters> pushed_sample_solution{};
  score_factor.applyAdjoint({sample_solve.solution.data(), sample_solve.solution.size()},
                            {pushed_sample_solution.data(), pushed_sample_solution.size()});
  for (std::size_t parameter = 0; parameter < parameters; ++parameter)
  {
    CHECK(full_solve.solution[parameter].real() ==
          Approx(dense_full[parameter]).epsilon(2.0e-10));
    CHECK(pushed_sample_solution[parameter].real() ==
          Approx(dense_full[parameter]).epsilon(2.0e-10));
  }
  for (std::size_t sample = 0; sample < samples; ++sample)
    CHECK(sample_solve.solution[sample].real() ==
          Approx(dense_sample[sample]).epsilon(2.0e-10));

  CHECK_THROWS_AS(ProjectedInverseIterationFactor(
                      score_factor, localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)),
                  std::invalid_argument);

  const std::uint32_t kinetic_and_ecp =
      kinetic | localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
  DensePIIDerivativeOperator ecp_derivative_operator(
      schema, version, kinetic_and_ecp);
  const ScoreCovarianceBatch ecp_batch{
      &ecp_derivative_operator, {weights.data(), weights.size()},
      {energies.data(), energies.size()}};
  ScoreCovarianceFactor ecp_score_factor(
      schema, version, DerivativeArrayView<const ScoreCovarianceBatch>{&ecp_batch, 1});
  ProjectedInverseIterationFactor ecp_factor(ecp_score_factor, kinetic_and_ecp);
  CHECK(ecp_factor.localEnergyTermMask() == kinetic_and_ecp);
  CHECK(ecp_factor.retainedBytes() > 0);
}

TEST_CASE("PII update rules publish transactionally with shifts retries and trust bounds",
          "[wftrain][pii][minpii][transaction]")
{
  const StructuredParameterSchema schema = makePIISchema();
  constexpr std::size_t version = 9;
  const std::array<DerivativeReal, 5> weights{1.0, 2.0, 0.5, 3.0, 1.5};
  const std::array<DerivativeValue, 5> energies{{
      {-1.7, 0.0}, {-0.9, 0.0}, {0.4, 0.0}, {-2.1, 0.0}, {1.2, 0.0}}};
  DensePIIDerivativeOperator derivative_operator(schema, version);
  const ScoreCovarianceBatch batch{
      &derivative_operator, {weights.data(), weights.size()},
      {energies.data(), energies.size()}};
  ScoreCovarianceFactor score_factor(
      schema, version, DerivativeArrayView<const ScoreCovarianceBatch>{&batch, 1});
  ProjectedInverseIterationFactor factor(
      score_factor, localEnergyTermBit(LocalEnergyTerm::KINETIC));
  const std::vector<DerivativeReal> gradient = score_factor.scoreCovarianceGradient();
  const ParameterGradientView objective{
      schema.fingerprint(), version, ReductionDomain::GLOBAL,
      {gradient.data(), gradient.size()}};
  const StructuredParameterSnapshot parameters{
      schema.fingerprint(), version, {0.25, -0.5, 0.75}};

  PIIUpdateControl control;
  control.shift_mode = PIIShiftMode::EXPLICIT;
  control.explicit_shift = -1.3;
  control.damping = 0.23;
  control.learning_rate = 0.4;
  control.maximum_update_norm = 0.03;
  control.maximum_damping_retries = 1;
  control.gmres.restart_length = 5;
  control.gmres.krylov.relative_tolerance = 1.0e-12;
  control.gmres.krylov.maximum_iterations = 40;
  control.gmres.krylov.maximum_operator_applications = 50;
  ProjectedInverseIterationUpdateRule full(factor, control);
  MinimumProjectedInverseIterationUpdateRule minimum(factor, control);
  const StructuredParameterSnapshot full_candidate =
      full.propose(schema, parameters, objective);
  const StructuredParameterSnapshot minimum_candidate =
      minimum.propose(schema, parameters, objective);
  REQUIRE(full.lastDiagnostics());
  REQUIRE(minimum.lastDiagnostics());
  CHECK(full.lastDiagnostics()->shift == Approx(-1.3));
  CHECK_FALSE(full.lastDiagnostics()->minimum_sample_space);
  CHECK(minimum.lastDiagnostics()->minimum_sample_space);
  DerivativeReal displacement_norm_squared = 0.0;
  for (std::size_t parameter = 0; parameter < 3; ++parameter)
  {
    CHECK(full_candidate.values[parameter] ==
          Approx(minimum_candidate.values[parameter]).epsilon(3.0e-9));
    displacement_norm_squared += std::pow(
        full_candidate.values[parameter] - parameters.values[parameter], 2);
  }
  CHECK(std::sqrt(displacement_norm_squared) == Approx(0.03).epsilon(2.0e-10));
  CHECK_THROWS_AS(full.propose(schema, parameters, objective), std::logic_error);
  full.proposalRejected();
  CHECK_FALSE(full.hasLiveProposal());
  full.propose(schema, parameters, objective);
  full.proposalAccepted(schema, parameters, objective);
  CHECK_FALSE(full.hasLiveProposal());
  minimum.proposalRejected();

  PIIUpdateControl mean_shift_control = control;
  mean_shift_control.shift_mode = PIIShiftMode::MEAN_ENERGY_MINUS_UNDERSHOOT;
  mean_shift_control.energy_undershoot = 0.17;
  mean_shift_control.maximum_update_norm = 0.0;
  MinimumProjectedInverseIterationUpdateRule mean_shift(factor, mean_shift_control);
  mean_shift.propose(schema, parameters, objective);
  REQUIRE(mean_shift.lastDiagnostics());
  CHECK(mean_shift.lastDiagnostics()->shift ==
        Approx(score_factor.meanEnergy().real() - 0.17));
  mean_shift.proposalRejected();

  PIIUpdateControl failure_control = control;
  failure_control.maximum_damping_retries = 2;
  failure_control.gmres.krylov.maximum_iterations = 0;
  ProjectedInverseIterationUpdateRule failed(factor, failure_control);
  CHECK_THROWS(failed.propose(schema, parameters, objective));
  REQUIRE(failed.lastDiagnostics());
  CHECK(failed.lastDiagnostics()->attempts == 3);
  CHECK_FALSE(failed.hasLiveProposal());
}

} // namespace qmcplusplus::wftrain
