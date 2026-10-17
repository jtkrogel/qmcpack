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
#include <limits>
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
const std::array<Row, 4> imaginary_del{{{0.9, -0.4}, {0.2, 1.3}, {-0.8, 0.6}, {0.7, -1.2}}};
const std::array<double, 4> weights{{1, 2, 0, 3}};
const std::array<DerivativeValue, 4> energies{{1, 3, -2, 2}};

class ExactRows final : public StreamingDerivativeOperator
{
public:
  ExactRows(const StructuredParameterSchema& schema, std::size_t ordinal, bool complex_energy = false)
      : schema_(schema), ordinal_(ordinal), plan_(schema, 7, 1), complex_energy_(complex_energy)
  {}
  /// Poison only the physical derivative's imaginary part to test projection admission.
  void setNonfiniteImaginaryEnergy(bool jvp, bool vjp) { bad_energy_jvp_ = jvp; bad_energy_vjp_ = vjp; }
  /// A complex score is outside the real tangent factor's supported model class.
  void setImaginaryScore() { imaginary_score_ = true; }
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
                    DerivativeAdjoint adjoint, ParameterReductionSink& sink) const override
  {
    for (const auto& chunk : plan_.chunks())
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
      {
        const auto& matrix = channels[channel].product == DerivativeProduct::SCORE_VJP ? scores : del;
        DerivativeValue value{};
        for (std::size_t i = 0; i < 2; ++i)
        {
          const auto sample = sampleOffset() + i, parameter = chunk.parameter_offset;
          DerivativeValue entry{matrix[sample][parameter], 0.0};
          if (channels[channel].product == DerivativeProduct::LOCAL_ENERGY_VJP)
            entry.imag(bad_energy_vjp_ ? std::numeric_limits<double>::quiet_NaN() :
                       (complex_energy_ ? imaginary_del[sample][parameter] : 0.0));
          if (adjoint == DerivativeAdjoint::HERMITIAN)
            entry = std::conj(entry);
          value += channels[channel].coefficients.values[i] * entry;
        }
        sink.add(channel, {chunk, {&value, 1}});
      }
  }
  void evaluateScoreJVP(const StructuredParameterVectorConstView& direction, SampleProductSink& sink) const override
  {
    emit(scores, direction, sink, false);
  }
  void evaluateLocalEnergyJVP(const StructuredParameterVectorConstView& direction,
                              std::uint32_t mask, SampleProductSink& sink) const override
  {
    if (mask != kinetic)
      throw std::logic_error("Unexpected local-energy term mask in sampled LM oracle");
    emit(del, direction, sink, true);
  }
private:
  void emit(const std::array<Row, 4>& matrix, const StructuredParameterVectorConstView& direction,
            SampleProductSink& sink, bool energy) const
  {
    std::array<DerivativeValue, 2> result{};
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t p = 0; p < 2; ++p)
      {
        const auto sample = sampleOffset() + i;
        DerivativeValue entry{matrix[sample][p], 0.0};
        if (energy)
          entry.imag(bad_energy_jvp_ ? std::numeric_limits<double>::quiet_NaN() :
                     (complex_energy_ ? imaginary_del[sample][p] : 0.0));
        else if (imaginary_score_)
          entry.imag(0.5);
        result[i] += entry * direction.values()[p];
      }
    SampleProductTileDescriptor descriptor{schema_.providerId(), schema_.fingerprint(), 7,
                                            ordinal_, sampleOffset(), 2, 0};
    sink.add({descriptor, {result.data(), result.size()}});
  }
  const StructuredParameterSchema& schema_;
  std::size_t ordinal_;
  ParameterChunkPlan plan_;
  bool complex_energy_ = false, bad_energy_jvp_ = false, bad_energy_vjp_ = false, imaginary_score_ = false;
};

struct Fixture
{
  explicit Fixture(bool complex_energy = false)
      : first(schema, 0, complex_energy), second(schema, 1, complex_energy)
  {}
  StructuredParameterSchema schema{"sampled_lm/test", {{"tangent", {2}, 0, 2, ParameterScalarDomain::REAL64, true, "tangent"}}};
  ExactRows first{schema, 0}, second{schema, 1};
  std::vector<ScoreCovarianceBatch> batches{
      {&first, {weights.data(), 2}, {energies.data(), 2}},
      {&second, {weights.data() + 2, 2}, {energies.data() + 2, 2}}};
  ScoreCovarianceFactor factor{schema, 7, {batches.data(), batches.size()}};
};

// Compute the dense oracle directly from sample rows, independently of factor actions.
std::array<double, 9> oracle(bool overlap, bool symmetric, double shift, bool complex_energy = false)
{
  std::array<double, 9> matrix{};
  Row mean{};
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t p = 0; p < 2; ++p)
      mean[p] += weights[i] * scores[i][p] / 6;
  for (std::size_t i = 0; i < 4; ++i)
  {
    const std::array<double, 3> d{{1, scores[i][0] - mean[0], scores[i][1] - mean[1]}};
    const std::array<DerivativeValue, 3> l{{energies[i],
        DerivativeValue{del[i][0], complex_energy ? imaginary_del[i][0] : 0.0} + energies[i] * d[1],
        DerivativeValue{del[i][1], complex_energy ? imaginary_del[i][1] : 0.0} + energies[i] * d[2]}};
    for (std::size_t r = 0; r < 3; ++r)
      for (std::size_t c = 0; c < 3; ++c)
        matrix[3*r+c] += (weights[i] / 6 * d[r] * (overlap ? DerivativeValue{d[c], 0.0} : l[c])).real();
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

TEST_CASE("Sampled LM explicitly restricts complex Hamiltonian products to a real tangent basis",
          "[drivers][wftrain][linear_method][complex]")
{
  Fixture f(true);
  const std::array<DerivativeValue, 2> tangent{{0.4, -0.7}};
  const StructuredParameterVectorConstView view(f.schema, 7, {tangent.data(), tangent.size()});
  Samples physical_products;
  f.first.applyLocalEnergyJVP(view, kinetic, physical_products);
  CHECK(std::abs(physical_products.values[0].imag()) > 0.1);
  CHECK(std::abs(physical_products.values[1].imag()) > 0.1);

  // The reusable PII factor retains its strict-real default. LM/ALM's explicit
  // policy must not silently broaden other optimizers or the global reducer.
  ProjectedInverseIterationFactor strict(f.factor, kinetic);
  std::vector<DerivativeValue> sample_action(f.factor.localSampleVectorSize());
  CHECK_THROWS(strict.applyShifted(view, 0.0, {sample_action.data(), sample_action.size()}));

  for (bool symmetric : {false, true})
  {
    SampledLinearMethodOperator op(f.factor, {f.batches.data(), f.batches.size()},
                                   kinetic, false, symmetric, 0.25);
    const auto dense = oracle(false, symmetric, 0.25, true);
    for (std::size_t column = 0; column < 3; ++column)
    {
      std::array<DerivativeValue, 3> direction{};
      direction[column] = 1.0;
      const auto result = applySampledAction(op, direction);
      for (std::size_t row = 0; row < 3; ++row)
      {
        CHECK(result[row].real() == Catch::Approx(dense[3 * row + column]).margin(1e-12));
        CHECK(result[row].imag() == 0.0);
      }
    }
  }
}

TEST_CASE("Sampled LM real tangent projection rejects imaginary nonfinite data and complex scores",
          "[drivers][wftrain][linear_method][complex]")
{
  const std::array<DerivativeValue, 3> direction{{0.7, -0.2, 0.9}};
  for (bool bad_jvp : {false, true})
  {
    Fixture f(true);
    f.first.setNonfiniteImaginaryEnergy(bad_jvp, !bad_jvp);
    SampledLinearMethodOperator op(f.factor, {f.batches.data(), f.batches.size()}, kinetic, false, true, 0.0);
    CHECK_THROWS(applySampledAction(op, direction));
  }
  Fixture complex_score;
  complex_score.first.setImaginaryScore();
  SampledLinearMethodOperator overlap(complex_score.factor,
      {complex_score.batches.data(), complex_score.batches.size()}, kinetic, true, false, 0.0);
  CHECK_THROWS(applySampledAction(overlap, direction));

  Fixture f;
  auto bad_energies = energies;
  bad_energies[0] = {1.0, std::numeric_limits<double>::quiet_NaN()};
  auto batches = f.batches;
  batches[0].local_energies = {bad_energies.data(), 2};
  CHECK_THROWS(SampledLinearMethodOperator(f.factor, {batches.data(), batches.size()}, kinetic, false, true, 0.0));
}

TEST_CASE("Fixed-reference linear LM retraction is projective and rejects unusable gauges",
          "[drivers][wftrain][linear_method][linear_retraction]")
{
  Fixture f;
  StructuredParameterSnapshot parameters{f.schema.fingerprint(), 7, {0.3, -0.2}};
  const auto gradient = f.factor.scoreCovarianceGradient();
  const ParameterGradientView objective{f.schema.fingerprint(), 7, ReductionDomain::GLOBAL,
                                       {gradient.data(), gradient.size()}};
  MatrixFreeLinearMethodResult solved;
  solved.provider_id = f.schema.providerId();
  solved.schema_fingerprint = f.schema.fingerprint();
  solved.parameter_version = 7;
  solved.converged = true;
  solved.stop_reason = LinearMethodStopReason::CONVERGED;
  const std::array<double, 2> mean{{14.0 / 6.0, 0.5}};
  const auto propose = [&](MatrixFreeLinearMethodResult result, double rate = 1, double xi = 0.5) {
    LinearMethodUpdateRule rule(f.factor, std::move(result), rate, xi,
        LinearMethodNormalization::FIXED_REFERENCE_LINEAR, {f.batches.data(), f.batches.size()});
    return rule.propose(f.schema, parameters, objective);
  };
  for (double reference : {1.0, 0.0}) // A nonzero physical gauge need not have nonzero centered v0.
  {
    solved.eigenvector = {reference, 0.2, -0.1};
    const double gauge = reference - mean[0] * 0.2 + mean[1] * 0.1;
    for (double scale : {1.0, -3.0, 1e-8})
      for (double rate : {1.0, 0.25})
      {
        auto rescaled = solved;
        for (auto& v : rescaled.eigenvector) v *= scale;
        for (double xi : {0.0, 0.8})
        {
          const auto actual = propose(rescaled, rate, xi);
          CHECK(actual.values[0] == Catch::Approx(parameters.values[0] + rate * 0.2 / gauge).margin(1e-12));
          CHECK(actual.values[1] == Catch::Approx(parameters.values[1] - rate * 0.1 / gauge).margin(1e-12));
        }
      }
  }
  solved.eigenvector = {mean[0] * 0.2 - mean[1] * 0.1, 0.2, -0.1};
  CHECK_THROWS_AS(propose(solved), std::invalid_argument);
  solved.eigenvector[0] += 1e-12;
  CHECK_THROWS_AS(propose(solved), std::invalid_argument);
  solved.eigenvector = {1, DerivativeValue{0.2, 0.1}, -0.1};
  CHECK_THROWS_AS(propose(solved), std::invalid_argument);
  solved.eigenvector[1] = {std::numeric_limits<double>::quiet_NaN(), 0};
  CHECK_THROWS(propose(solved));
  solved.eigenvector = {1, 0.2, -0.1};
  auto stale = solved;
  stale.parameter_version = 6;
  CHECK_THROWS_AS(propose(stale), std::invalid_argument);
  stale = solved;
  stale.schema_fingerprint = "wrong";
  CHECK_THROWS_AS(propose(stale), std::invalid_argument);

  // Default nonlinear xi normalization still uses vi/v0 and the centered metric norm.
  LinearMethodUpdateRule nonlinear(f.factor, solved, 0.25, 0.5);
  const auto actual = nonlinear.propose(f.schema, parameters, objective);
  double q = 0;
  for (std::size_t i = 0; i < weights.size(); ++i)
  {
    const double d = (scores[i][0] - mean[0]) * 0.2 - (scores[i][1] - mean[1]) * 0.1;
    q += weights[i] * d * d / 6;
  }
  const double denominator = 1 + q / (1 + std::sqrt(1 + q));
  CHECK(actual.values[0] == Catch::Approx(parameters.values[0] + 0.25 * 0.2 / denominator).margin(1e-12));
  CHECK(actual.values[1] == Catch::Approx(parameters.values[1] - 0.25 * 0.1 / denominator).margin(1e-12));
  CHECK(parameters.values == std::vector<double>{0.3, -0.2});
}

} // namespace qmcplusplus::wftrain
