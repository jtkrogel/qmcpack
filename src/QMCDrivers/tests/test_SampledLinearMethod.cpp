//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include "QMCDrivers/WFTrain/SampledLinearMethod.h"
#include "Utilities/for_testing/Catch2Approx.h"
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{
using Row = std::array<double, 2>;
constexpr std::uint32_t kinetic = localEnergyTermBit(LocalEnergyTerm::KINETIC);
const std::array<Row, 4> scores{{{1, -1}, {2, 0.5}, {-0.5, 2}, {3, 1}}};
const std::array<Row, 4> del{{{0.2, 1}, {-1, 0.3}, {0.7, -0.4}, {1.1, 2}}};
const std::array<double, 4> weights{{1, 2, 0, 3}};
const std::array<DerivativeValue, 4> energies{{1, 3, -2, 2}};

class ExactRows final : public StreamingDerivativeOperator
{
public:
  ExactRows(const StructuredParameterSchema& schema, std::size_t ordinal)
      : schema_(schema), ordinal_(ordinal), plan_(schema, 7, 1)
  {}
  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }
  std::size_t parameterVersion() const noexcept override { return 7; }
  std::size_t batchOrdinal() const noexcept override { return ordinal_; }
  std::size_t sampleOffset() const noexcept override { return 2 * ordinal_; }
  std::size_t sampleCount() const noexcept override { return 2; }
  const ParameterChunkPlan& parameterChunkPlan() const noexcept override { return plan_; }
  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities c;
    c.product_mask = derivativeProductBit(DerivativeProduct::SCORE_JVP) |
        derivativeProductBit(DerivativeProduct::SCORE_VJP) |
        derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_JVP) |
        derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_VJP);
    c.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::HERMITIAN) |
        derivativeAdjointBit(DerivativeAdjoint::TRANSPOSE);
    c.parameter_scalar_domain = ParameterScalarDomain::REAL64;
    c.result_scalar_domain = ParameterScalarDomain::COMPLEX128;
    c.reduction_domain = ReductionDomain::CROWD_LOCAL;
    c.execution_domain = DerivativeExecutionDomain::HOST;
    c.local_energy_term_mask = kinetic;
    c.maximum_vjp_channels = 4;
    c.maximum_parameter_chunk_size = 1;
    c.maximum_sample_tile_size = 2;
    c.block_streaming = true;
    return c;
  }
protected:
  void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                    DerivativeAdjoint, ParameterReductionSink& sink) const override
  {
    for (const auto& chunk : plan_.chunks())
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
      {
        const auto& matrix = channels[channel].product == DerivativeProduct::SCORE_VJP ? scores : del;
        DerivativeValue value{};
        for (std::size_t i = 0; i < 2; ++i)
          value += channels[channel].coefficients.values[i] * matrix[sampleOffset() + i][chunk.parameter_offset];
        sink.add(channel, {chunk, {&value, 1}});
      }
  }
  void evaluateScoreJVP(const StructuredParameterVectorConstView& direction, SampleProductSink& sink) const override
  {
    emit(scores, direction, sink);
  }
  void evaluateLocalEnergyJVP(const StructuredParameterVectorConstView& direction,
                              std::uint32_t mask, SampleProductSink& sink) const override
  {
    if (mask != kinetic)
      throw std::logic_error("Unexpected local-energy term mask in sampled LM oracle");
    emit(del, direction, sink);
  }
private:
  void emit(const std::array<Row, 4>& matrix, const StructuredParameterVectorConstView& direction,
            SampleProductSink& sink) const
  {
    std::array<DerivativeValue, 2> result{};
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t p = 0; p < 2; ++p)
        result[i] += matrix[sampleOffset() + i][p] * direction.values()[p];
    SampleProductTileDescriptor descriptor{schema_.providerId(), schema_.fingerprint(), 7,
                                            ordinal_, sampleOffset(), 2, 0};
    sink.add({descriptor, {result.data(), result.size()}});
  }
  const StructuredParameterSchema& schema_;
  std::size_t ordinal_;
  ParameterChunkPlan plan_;
};

struct Fixture
{
  StructuredParameterSchema schema{"sampled_lm/test", {{"tangent", {2}, 0, 2, ParameterScalarDomain::REAL64, true, "tangent"}}};
  ExactRows first{schema, 0}, second{schema, 1};
  std::vector<ScoreCovarianceBatch> batches{
      {&first, {weights.data(), 2}, {energies.data(), 2}},
      {&second, {weights.data() + 2, 2}, {energies.data() + 2, 2}}};
  ScoreCovarianceFactor factor{schema, 7, {batches.data(), batches.size()}};
};

// Compute the dense oracle directly from sample rows, independently of factor actions.
std::array<double, 9> oracle(bool overlap, bool symmetric, double shift)
{
  std::array<double, 9> matrix{};
  Row mean{};
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t p = 0; p < 2; ++p)
      mean[p] += weights[i] * scores[i][p] / 6;
  for (std::size_t i = 0; i < 4; ++i)
  {
    const std::array<double, 3> d{{1, scores[i][0] - mean[0], scores[i][1] - mean[1]}};
    const std::array<double, 3> l{{energies[i].real(), del[i][0] + energies[i].real() * d[1],
                                  del[i][1] + energies[i].real() * d[2]}};
    for (std::size_t r = 0; r < 3; ++r)
      for (std::size_t c = 0; c < 3; ++c)
        matrix[3*r+c] += weights[i] / 6 * d[r] * (overlap ? d[c] : l[c]);
  }
  if (symmetric && !overlap)
    for (std::size_t r = 0; r < 3; ++r)
      for (std::size_t c = r + 1; c < 3; ++c)
        matrix[3*r+c] = matrix[3*c+r] = 0.5 * (matrix[3*r+c] + matrix[3*c+r]);
  matrix[4] += shift;
  matrix[8] += shift;
  return matrix;
}
std::array<DerivativeValue, 3> applySampledAction(
    const AugmentedLinearMethodOperator& op,
    const std::array<DerivativeValue, 3>& direction)
{
  std::array<DerivativeValue, 3> result{};
  op.apply({direction.data(), direction.size()}, {result.data(), result.size()});
  return result;
}
double dot(const std::array<DerivativeValue, 3>& a, const std::array<DerivativeValue, 3>& b)
{
  double result = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    result += a[i].real() * b[i].real();
  return result;
}
class Samples final : public SampleProductSink
{
public:
  std::array<DerivativeValue, 2> values{};
private:
  void consume(const SampleProductTileConstView& tile) override
  {
    for (std::size_t i = 0; i < 2; ++i)
      values[i] = tile.values()[i];
  }
};
class Parameters final : public ParameterReductionSink
{
public:
  std::array<DerivativeValue, 2> values{};
private:
  void consume(std::size_t, const ParameterChunkConstView& chunk) override
  {
    values[chunk.descriptor().parameter_offset] += chunk.values()[0];
  }
};
} // namespace

TEST_CASE("Sampled LM actions match weighted centered dense sample oracles", "[drivers][wftrain][linear_method]")
{
  Fixture f;
  for (bool overlap : {false, true})
    for (bool symmetric : {false, true})
      for (double shift : {0.0, 0.25})
      {
        INFO("overlap=" << overlap << " symmetric=" << symmetric << " shift=" << shift);
        SampledLinearMethodOperator op(f.factor, {f.batches.data(), f.batches.size()}, kinetic, overlap, symmetric, shift);
        auto dense = oracle(overlap, symmetric, shift);
        CHECK(op.descriptor().hermitian == (overlap || symmetric));
        CHECK(op.descriptor().reduction_domain == ReductionDomain::GLOBAL);
        for (std::size_t column = 0; column < 3; ++column)
        {
          std::array<DerivativeValue, 3> direction{};
          direction[column] = 1;
          auto result = applySampledAction(op, direction);
          for (std::size_t row = 0; row < 3; ++row)
          {
            CHECK(result[row].real() == Catch::Approx(dense[3*row+column]).margin(1.e-12));
            CHECK(result[row].imag() == 0);
          }
        }
        CHECK(dot(applySampledAction(op, {{0, 0, 0}}), {{1, 1, 1}}) == 0);
      }
}

TEST_CASE("Sampled LM preserves raw orientation and symmetrized adjoint duality", "[drivers][wftrain][linear_method]")
{
  Fixture f;
  SampledLinearMethodOperator raw(f.factor, {f.batches.data(), f.batches.size()}, kinetic, false, false, 0);
  SampledLinearMethodOperator sym(f.factor, {f.batches.data(), f.batches.size()}, kinetic, false, true, 0.3);
  SampledLinearMethodOperator overlap(f.factor, {f.batches.data(), f.batches.size()}, kinetic, true, false, 0.3);
  const std::array<DerivativeValue, 3> x{{0.7, -0.2, 0.9}}, y{{-0.3, 0.8, 0.4}};
  auto dense = oracle(false, false, 0);
  auto actual = applySampledAction(raw, x);
  double transpose_error = 0;
  for (std::size_t r = 0; r < 3; ++r)
  {
    double transposed = 0;
    for (std::size_t c = 0; c < 3; ++c)
      transposed += dense[3*c+r] * x[c].real();
    transpose_error += std::abs(actual[r].real() - transposed);
  }
  CHECK(transpose_error > 0.1);
  CHECK(dot(x, applySampledAction(sym, y)) ==
        Catch::Approx(dot(applySampledAction(sym, x), y)).margin(1.e-12));
  CHECK(dot(x, applySampledAction(overlap, y)) ==
        Catch::Approx(dot(applySampledAction(overlap, x), y)).margin(1.e-12));
}

TEST_CASE("Sampled LM oracle producer has exact score and local-energy JVP VJP duality", "[drivers][wftrain][linear_method]")
{
  Fixture f;
  const std::array<DerivativeValue, 2> direction{{0.4, -0.7}}, coefficients{{0.3, 1.2}};
  StructuredParameterVectorConstView view(f.schema, 7, {direction.data(), direction.size()});
  for (bool local_energy : {false, true})
  {
    Samples sample_sink;
    Parameters parameter_sink;
    if (local_energy)
      f.first.applyLocalEnergyJVP(view, kinetic, sample_sink);
    else
      f.first.applyScoreJVP(view, sample_sink);
    CoefficientView cv{f.schema.providerId(), f.schema.fingerprint(), 7, 0, 0,
                        {coefficients.data(), coefficients.size()}};
    VJPCoefficientChannel channel{"oracle", local_energy ? DerivativeProduct::LOCAL_ENERGY_VJP : DerivativeProduct::SCORE_VJP,
                                  cv, local_energy ? kinetic : 0};
    f.first.applyVJPs({&channel, 1}, DerivativeAdjoint::HERMITIAN, parameter_sink);
    double left = 0, right = 0;
    for (std::size_t i = 0; i < 2; ++i)
    {
      left += coefficients[i].real() * sample_sink.values[i].real();
      right += direction[i].real() * parameter_sink.values[i].real();
    }
    CHECK(left == Catch::Approx(right).margin(1.e-12));
  }
}
TEST_CASE("Sampled LM rejects malformed nonowning batch views", "[drivers][wftrain][linear_method]")
{
  Fixture f;
  auto batches = f.batches;
  batches[0].local_energies = {batches[0].local_energies.data(), 1};
  CHECK_THROWS(SampledLinearMethodOperator(f.factor, {batches.data(), batches.size()},
                                         kinetic, false, true, 0.01));
  batches = f.batches;
  batches[0].derivative_operator = nullptr;
  CHECK_THROWS(SampledLinearMethodOperator(f.factor, {batches.data(), batches.size()},
                                         kinetic, true, false, 0.01));
}

} // namespace qmcplusplus::wftrain
