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
#include <filesystem>
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

} // namespace qmcplusplus::wftrain
