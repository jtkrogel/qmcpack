//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_DistributedSampleSpaceOptimizers.cpp
 * @brief MPI dense-oracle coverage for matrix-free sample-space optimizers.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "Message/Communicate.h"
#include "QMCDrivers/WFTrain/MinimumStepStochasticReconfiguration.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

using Catch::Approx;

constexpr std::size_t PARAMETER_COUNT = 3;
constexpr std::size_t SAMPLE_COUNT    = 5;
constexpr std::size_t VERSION         = 11;

using Row = std::array<DerivativeReal, PARAMETER_COUNT>;

constexpr std::array<Row, SAMPLE_COUNT> SCORES{{
    {{1.0, 0.2, -0.3}},
    {{-0.4, 1.1, 0.5}},
    {{0.7, -0.8, 1.2}},
    {{1.3, 0.4, -0.6}},
    {{-0.9, 0.6, 0.8}},
}};

constexpr std::array<Row, SAMPLE_COUNT> LOCAL_ENERGY_JACOBIAN{{
    {{0.2, -0.1, 0.5}},
    {{-0.6, 0.3, 0.4}},
    {{0.9, 0.2, -0.7}},
    {{-0.3, 1.0, 0.1}},
    {{0.5, -0.4, 0.8}},
}};

constexpr std::array<DerivativeReal, SAMPLE_COUNT> WEIGHTS{{1.0, 2.0, 0.5, 3.0, 1.5}};
constexpr std::array<DerivativeReal, SAMPLE_COUNT> ENERGIES{{-1.7, -0.9, 0.4, -2.1, 1.2}};

/// Return the fixed parameter schema shared by every rank.
StructuredParameterSchema makeSchema()
{
  return {"distributed/sample_space_optimizers",
          {{"weights", {PARAMETER_COUNT}, 0, PARAMETER_COUNT,
            ParameterScalarDomain::REAL64, true, "weights"}}};
}

/** Assign the same global samples unevenly at every supported communicator size.
 *
 * Three ranks intentionally leave rank one empty, exercising zero-length local
 * sample vectors in both distributed Krylov solvers.
 */
std::vector<std::size_t> localSampleIndices(int rank, int size)
{
  if (size == 1)
    return {0, 1, 2, 3, 4};
  if (size == 2)
    return rank == 0 ? std::vector<std::size_t>{0, 1}
                     : std::vector<std::size_t>{2, 3, 4};
  if (size == 3)
  {
    if (rank == 0)
      return {0, 1};
    if (rank == 1)
      return {};
    return {2, 3, 4};
  }
  throw std::invalid_argument("Distributed optimizer test supports one to three ranks");
}

/// Stream rank-local score and local-energy products without storing a dense factor.
class PartitionedDerivativeOperator final : public StreamingDerivativeOperator
{
public:
  PartitionedDerivativeOperator(const StructuredParameterSchema& schema,
                                int rank,
                                std::vector<std::size_t> sample_indices)
      : schema_(schema), rank_(rank), sample_indices_(std::move(sample_indices)),
        plan_(schema_, VERSION, 2)
  {}

  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities result;
    result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
        derivativeProductBit(DerivativeProduct::SCORE_JVP) |
        derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_JVP);
    result.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::HERMITIAN);
    result.parameter_scalar_domain = ParameterScalarDomain::REAL64;
    result.result_scalar_domain = ParameterScalarDomain::COMPLEX128;
    result.reduction_domain = ReductionDomain::RANK_LOCAL;
    result.execution_domain = DerivativeExecutionDomain::HOST;
    result.local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
    result.maximum_vjp_channels = 1;
    result.maximum_parameter_chunk_size = 2;
    result.maximum_sample_tile_size = std::max<std::size_t>(sample_indices_.size(), 1);
    result.block_streaming = true;
    return result;
  }

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }
  std::size_t parameterVersion() const noexcept override { return VERSION; }
  std::size_t batchOrdinal() const noexcept override { return static_cast<std::size_t>(rank_); }
  std::size_t sampleOffset() const noexcept override
  {
    return sample_indices_.empty() ? 0 : sample_indices_.front();
  }
  std::size_t sampleCount() const noexcept override { return sample_indices_.size(); }
  const ParameterChunkPlan& parameterChunkPlan() const noexcept override { return plan_; }

protected:
  void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                    DerivativeAdjoint,
                    ParameterReductionSink& sink) const override
  {
    for (const ParameterChunkDescriptor& chunk : plan_.chunks())
    {
      std::vector<DerivativeValue> values(chunk.count);
      for (std::size_t local_parameter = 0; local_parameter < chunk.count;
           ++local_parameter)
        for (std::size_t local_sample = 0; local_sample < sample_indices_.size();
             ++local_sample)
          values[local_parameter] += channels[0].coefficients.values[local_sample] *
              SCORES[sample_indices_[local_sample]][chunk.parameter_offset + local_parameter];
      sink.add(0, {chunk, {values.data(), values.size()}});
    }
  }

  void evaluateScoreJVP(const StructuredParameterVectorConstView& direction,
                        SampleProductSink& sink) const override
  {
    emitProduct(SCORES, direction, sink);
  }

  void evaluateLocalEnergyJVP(const StructuredParameterVectorConstView& direction,
                              std::uint32_t term_mask,
                              SampleProductSink& sink) const override
  {
    if (term_mask != localEnergyTermBit(LocalEnergyTerm::KINETIC))
      throw std::logic_error("Distributed PII test received an unexpected term mask");
    emitProduct(LOCAL_ENERGY_JACOBIAN, direction, sink);
  }

private:
  /// Emit one score-like matrix-vector product over this rank's sample partition.
  void emitProduct(const std::array<Row, SAMPLE_COUNT>& matrix,
                   const StructuredParameterVectorConstView& direction,
                   SampleProductSink& sink) const
  {
    std::vector<DerivativeValue> products(sample_indices_.size());
    for (std::size_t local_sample = 0; local_sample < sample_indices_.size(); ++local_sample)
      for (std::size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter)
        products[local_sample] += matrix[sample_indices_[local_sample]][parameter] *
            direction.values()[parameter];
    const SampleProductTileDescriptor descriptor{
        schema_.providerId(), schema_.fingerprint(), VERSION,
        static_cast<std::size_t>(rank_), sampleOffset(), products.size(), 0};
    sink.add({descriptor, {products.data(), products.size()}});
  }

  const StructuredParameterSchema& schema_;
  int rank_;
  std::vector<std::size_t> sample_indices_;
  ParameterChunkPlan plan_;
};

/// Solve one small dense real system with partial pivoting.
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

/// Hold dense global quantities used as an implementation-independent oracle.
struct DenseOracle
{
  std::array<Row, SAMPLE_COUNT> centered_score{};
  std::array<Row, SAMPLE_COUNT> shifted_factor{};
  std::array<DerivativeReal, PARAMETER_COUNT> gradient{};
  std::vector<DerivativeReal> minsr_direction;
  std::vector<DerivativeReal> pii_direction;
};

/// Form exact dense minSR and PII systems from the fixed global population.
DenseOracle makeDenseOracle(DerivativeReal damping, DerivativeReal shift)
{
  DenseOracle oracle;
  DerivativeReal weight_sum = 0.0;
  DerivativeReal weighted_energy = 0.0;
  Row mean_score{};
  for (std::size_t sample = 0; sample < SAMPLE_COUNT; ++sample)
  {
    weight_sum += WEIGHTS[sample];
    weighted_energy += WEIGHTS[sample] * ENERGIES[sample];
    for (std::size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter)
      mean_score[parameter] += WEIGHTS[sample] * SCORES[sample][parameter];
  }
  weighted_energy /= weight_sum;
  for (DerivativeReal& value : mean_score)
    value /= weight_sum;

  for (std::size_t sample = 0; sample < SAMPLE_COUNT; ++sample)
  {
    const DerivativeReal sqrt_q = std::sqrt(WEIGHTS[sample] / weight_sum);
    for (std::size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter)
    {
      oracle.centered_score[sample][parameter] =
          sqrt_q * (SCORES[sample][parameter] - mean_score[parameter]);
      oracle.shifted_factor[sample][parameter] =
          sqrt_q * LOCAL_ENERGY_JACOBIAN[sample][parameter] +
          (ENERGIES[sample] - shift) * oracle.centered_score[sample][parameter];
      oracle.gradient[parameter] += 2.0 * oracle.centered_score[sample][parameter] *
          sqrt_q * (ENERGIES[sample] - weighted_energy);
    }
  }

  std::vector<std::vector<DerivativeReal>> covariance(
      PARAMETER_COUNT, std::vector<DerivativeReal>(PARAMETER_COUNT));
  std::vector<std::vector<DerivativeReal>> pii_matrix(
      PARAMETER_COUNT, std::vector<DerivativeReal>(PARAMETER_COUNT));
  for (std::size_t row = 0; row < PARAMETER_COUNT; ++row)
    for (std::size_t column = 0; column < PARAMETER_COUNT; ++column)
    {
      covariance[row][column] = row == column ? damping : 0.0;
      pii_matrix[row][column] = row == column ? damping : 0.0;
      for (std::size_t sample = 0; sample < SAMPLE_COUNT; ++sample)
      {
        covariance[row][column] += oracle.centered_score[sample][row] *
            oracle.centered_score[sample][column];
        pii_matrix[row][column] += oracle.centered_score[sample][row] *
            oracle.shifted_factor[sample][column];
      }
    }

  oracle.minsr_direction = solveDense(
      covariance,
      std::vector<DerivativeReal>(oracle.gradient.begin(), oracle.gradient.end()));
  std::vector<DerivativeReal> half_gradient(oracle.gradient.begin(), oracle.gradient.end());
  for (DerivativeReal& value : half_gradient)
    value *= 0.5;
  oracle.pii_direction = solveDense(pii_matrix, std::move(half_gradient));
  return oracle;
}

/// Return the Euclidean norm of one dense real direction.
DerivativeReal vectorNorm(const std::vector<DerivativeReal>& values)
{
  DerivativeReal squared = 0.0;
  for (const DerivativeReal value : values)
    squared += value * value;
  return std::sqrt(squared);
}

/// Check a published candidate against parameters minus a scaled oracle direction.
void checkCandidate(const StructuredParameterSnapshot& candidate,
                    const StructuredParameterSnapshot& parameters,
                    const std::vector<DerivativeReal>& direction,
                    DerivativeReal scale,
                    DerivativeReal tolerance)
{
  REQUIRE(candidate.values.size() == PARAMETER_COUNT);
  for (std::size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter)
    CHECK(candidate.values[parameter] ==
          Approx(parameters.values[parameter] - scale * direction[parameter])
              .epsilon(tolerance)
              .margin(tolerance));
}

} // namespace

TEST_CASE("Distributed sample-space optimizers match one global dense oracle",
          "[drivers][training][minsr][spring][pii][minpii][mpi]")
{
  Communicate& communicator = *OHMMS::Controller;
  REQUIRE(communicator.size() >= 1);
  REQUIRE(communicator.size() <= 3);

  const StructuredParameterSchema schema = makeSchema();
  const std::vector<std::size_t> local_indices =
      localSampleIndices(communicator.rank(), communicator.size());
  PartitionedDerivativeOperator derivative_operator(schema, communicator.rank(),
                                                     local_indices);
  std::vector<DerivativeReal> local_weights(local_indices.size());
  std::vector<DerivativeValue> local_energies(local_indices.size());
  for (std::size_t local_sample = 0; local_sample < local_indices.size(); ++local_sample)
  {
    local_weights[local_sample] = WEIGHTS[local_indices[local_sample]];
    local_energies[local_sample] = {ENERGIES[local_indices[local_sample]], 0.0};
  }

  const ScoreCovarianceBatch batch{
      &derivative_operator, {local_weights.data(), local_weights.size()},
      {local_energies.data(), local_energies.size()}};
  const DerivativeArrayView<const ScoreCovarianceBatch> local_batches =
      local_indices.empty() ? DerivativeArrayView<const ScoreCovarianceBatch>{}
                            : DerivativeArrayView<const ScoreCovarianceBatch>{&batch, 1};
  ScoreCovarianceFactor score_factor(
      schema, VERSION, local_batches, ScoreSampleDomain::REAL,
      DistributedParameterReduction(communicator, {2}));

  CHECK(score_factor.localSampleVectorSize() == local_indices.size());
  CHECK(score_factor.globalSampleVectorSize() == SAMPLE_COUNT);
  const std::vector<DerivativeReal> gradient = score_factor.scoreCovarianceGradient();
  const ParameterGradientView objective{
      schema.fingerprint(), VERSION, ReductionDomain::GLOBAL,
      {gradient.data(), gradient.size()}};
  const StructuredParameterSnapshot parameters{
      schema.fingerprint(), VERSION, {0.25, -0.5, 0.75}};

  constexpr DerivativeReal damping = 0.23;
  constexpr DerivativeReal learning_rate = 0.4;
  constexpr DerivativeReal shift = -1.3;
  const DenseOracle oracle = makeDenseOracle(damping, shift);
  for (std::size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter)
    CHECK(gradient[parameter] ==
          Approx(oracle.gradient[parameter]).epsilon(2.0e-12).margin(2.0e-12));

  MinimumStepUpdateControl minimum_step_control;
  minimum_step_control.learning_rate = learning_rate;
  minimum_step_control.damping = damping;
  minimum_step_control.center_shift = 1.0;
  minimum_step_control.krylov.relative_tolerance = 1.0e-12;
  minimum_step_control.krylov.maximum_iterations = 40;
  minimum_step_control.krylov.maximum_operator_applications = 50;

  MinimumStepSRUpdateRule minsr(score_factor, minimum_step_control);
  const StructuredParameterSnapshot minsr_candidate =
      minsr.propose(schema, parameters, objective);
  checkCandidate(minsr_candidate, parameters, oracle.minsr_direction,
                 learning_rate, 2.0e-9);
  REQUIRE(minsr.lastDiagnostics());
  CHECK(minsr.lastDiagnostics()->solve.converged);
  CHECK(minsr.lastDiagnostics()->solve.operator_applications > 0);
  CHECK(minsr.lastDiagnostics()->direction_norm ==
        Approx(vectorNorm(oracle.minsr_direction)).epsilon(2.0e-9));
  CHECK(minsr.lastDiagnostics()->applied_scale == Approx(learning_rate));
  minsr.proposalAccepted(schema, parameters, objective);

  SPRINGUpdateControl spring_control;
  spring_control.minimum_step = minimum_step_control;
  spring_control.history_decay = 0.0;
  SPRINGUpdateRule spring(score_factor, spring_control);
  const StructuredParameterSnapshot spring_candidate =
      spring.propose(schema, parameters, objective);
  checkCandidate(spring_candidate, parameters, oracle.minsr_direction,
                 learning_rate, 2.0e-9);
  REQUIRE(spring.lastDiagnostics());
  CHECK(spring.lastDiagnostics()->solve.converged);
  CHECK(spring.lastDiagnostics()->solve.operator_applications > 0);
  spring.proposalAccepted(schema, parameters, objective);
  CHECK(spring.acceptedUpdateCount() == 1);
  REQUIRE(spring.history().size() == PARAMETER_COUNT);
  for (std::size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter)
    CHECK(spring.history()[parameter].real() ==
          Approx(oracle.minsr_direction[parameter]).epsilon(2.0e-9));

  ProjectedInverseIterationFactor pii_factor(
      score_factor, localEnergyTermBit(LocalEnergyTerm::KINETIC));
  PIIUpdateControl pii_control;
  pii_control.learning_rate = learning_rate;
  pii_control.shift_mode = PIIShiftMode::EXPLICIT;
  pii_control.explicit_shift = shift;
  pii_control.damping = damping;
  pii_control.maximum_damping_retries = 0;
  pii_control.gmres.restart_length = 5;
  pii_control.gmres.krylov.relative_tolerance = 1.0e-12;
  pii_control.gmres.krylov.maximum_iterations = 40;
  pii_control.gmres.krylov.maximum_operator_applications = 50;

  ProjectedInverseIterationUpdateRule pii(pii_factor, pii_control);
  const StructuredParameterSnapshot pii_candidate =
      pii.propose(schema, parameters, objective);
  checkCandidate(pii_candidate, parameters, oracle.pii_direction,
                 learning_rate, 3.0e-8);
  REQUIRE(pii.lastDiagnostics());
  CHECK_FALSE(pii.lastDiagnostics()->minimum_sample_space);
  CHECK(pii.lastDiagnostics()->shift == Approx(shift));
  CHECK(pii.lastDiagnostics()->damping == Approx(damping));
  CHECK(pii.lastDiagnostics()->attempts == 1);
  CHECK(pii.lastDiagnostics()->operator_applications > 0);
  CHECK(pii.lastDiagnostics()->direction_norm ==
        Approx(vectorNorm(oracle.pii_direction)).epsilon(3.0e-8));
  CHECK(pii.lastDiagnostics()->applied_scale == Approx(learning_rate));
  pii.proposalAccepted(schema, parameters, objective);
  CHECK_FALSE(pii.hasLiveProposal());

  MinimumProjectedInverseIterationUpdateRule minpii(pii_factor, pii_control);
  const StructuredParameterSnapshot minpii_candidate =
      minpii.propose(schema, parameters, objective);
  checkCandidate(minpii_candidate, parameters, oracle.pii_direction,
                 learning_rate, 3.0e-8);
  REQUIRE(minpii.lastDiagnostics());
  CHECK(minpii.lastDiagnostics()->minimum_sample_space);
  CHECK(minpii.lastDiagnostics()->shift == Approx(shift));
  CHECK(minpii.lastDiagnostics()->damping == Approx(damping));
  CHECK(minpii.lastDiagnostics()->attempts == 1);
  CHECK(minpii.lastDiagnostics()->operator_applications > 0);
  CHECK(minpii.lastDiagnostics()->direction_norm ==
        Approx(vectorNorm(oracle.pii_direction)).epsilon(3.0e-8));
  CHECK(minpii.lastDiagnostics()->applied_scale == Approx(learning_rate));
  minpii.proposalAccepted(schema, parameters, objective);
  CHECK_FALSE(minpii.hasLiveProposal());
}

} // namespace qmcplusplus::wftrain
