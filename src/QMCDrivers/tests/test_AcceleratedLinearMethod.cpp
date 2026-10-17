//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include "QMCDrivers/WFTrain/AcceleratedLinearMethod.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"
#include "Utilities/for_testing/Catch2Approx.h"
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::wftrain
{
namespace
{
StructuredParameterSchema schema(std::size_t count)
{
  return {"alm/test", {{"tangent", {count}, 0, count, ParameterScalarDomain::REAL64, true, "tangent"}}};
}
class Dense final : public AugmentedLinearMethodOperator
{
public:
  Dense(const StructuredParameterSchema& schema, std::vector<double> matrix, bool hermitian)
      : AugmentedLinearMethodOperator(schema, 3, ReductionDomain::GLOBAL, hermitian), matrix_(std::move(matrix))
  {}
private:
  void evaluate(DerivativeArrayView<const DerivativeValue> v, DerivativeArrayView<DerivativeValue> out) const override
  {
    for (std::size_t i = 0; i < v.size(); ++i)
    {
      out[i] = {};
      for (std::size_t j = 0; j < v.size(); ++j)
        out[i] += matrix_[i * v.size() + j] * v[j];
    }
  }
  std::vector<double> matrix_;
};
MatrixFreeLinearMethodResult solve(const Dense& h, const Dense& s, AcceleratedLinearMethodControl c = {})
{
  IdentityPreconditioner p(h.tangentSchema(), 3);
  std::vector<DerivativeValue> initial(h.tangentSchema().parameterCount() + 1);
  initial[0] = 1;
  return solveAcceleratedMatrixFreeLinearMethod(h, s, p, {initial.data(), initial.size()}, c);
}
} // namespace

TEST_CASE("Accelerated LM matches a nonsymmetric generalized dense oracle", "[drivers][wftrain][linear_method]")
{
  auto model = schema(1);
  // det(H-lambda*S)=2*(lambda^2-6lambda+7); E0=2 selects 3-sqrt(2).
  Dense h(model, {2, -2, -1, 8}, false);
  Dense s(model, {1, 0, 0, 2}, true);
  AcceleratedLinearMethodControl c;
  c.outer.maximum_subspace_dimension = 2;
  c.outer.relative_residual_tolerance = 1.e-11;
  auto result = solve(h, s, c);
  INFO(linearMethodStopReasonName(result.stop_reason));
  REQUIRE(result.converged);
  CHECK(result.eigenvalue == Catch::Approx(3 - std::sqrt(2.)).epsilon(1.e-10));
  CHECK(result.relative_residual_norm < 1.e-11);
  REQUIRE(result.eigenvector.size() == 2);
  auto x = result.eigenvector[0].real();
  auto y = result.eigenvector[1].real();
  CHECK(std::hypot(2*x - 2*y - result.eigenvalue*x,
                   -x + 8*y - 2*result.eigenvalue*y) < 1.e-10);
  CHECK(result.operator_applications <= c.outer.maximum_operator_applications);
}

TEST_CASE("Accelerated LM follows reference energy rather than the lowest root", "[drivers][wftrain][linear_method]")
{
  auto model = schema(1);
  // Roots 1 and 5, reference diagonal 4; select 5.
  Dense h(model, {4, 3, 1, 2}, false);
  Dense s(model, {1, 0, 0, 1}, true);
  auto result = solve(h, s);
  REQUIRE(result.converged);
  CHECK(result.eigenvalue == Catch::Approx(5.));
}

TEST_CASE("Accelerated LM rejects complex roots and tiny reference amplitudes", "[drivers][wftrain][linear_method]")
{
  auto model = schema(1);
  Dense s(model, {1, 0, 0, 1}, true);
  SECTION("complex conjugate pair")
  {
    Dense h(model, {0, -1, 1, 0}, false);
    auto result = solve(h, s);
    CHECK_FALSE(result.converged);
    CHECK(result.stop_reason == LinearMethodStopReason::UNTRUSTED_ROOT);
  }
  SECTION("reference cutoff excludes every full-space root")
  {
    Dense h(model, {2, -1, -1, 2}, false);
    AcceleratedLinearMethodControl c;
    c.outer.minimum_reference_overlap = 0.9;
    auto result = solve(h, s, c);
    CHECK_FALSE(result.converged);
    CHECK(result.stop_reason == LinearMethodStopReason::UNTRUSTED_ROOT);
  }
}

TEST_CASE("Accelerated LM shares bounded work between outer and inner solves", "[drivers][wftrain][linear_method]")
{
  auto model = schema(1);
  Dense h(model, {2, -2, -1, 8}, false);
  Dense s(model, {1, 0, 0, 2}, true);
  SECTION("operator applications")
  {
    AcceleratedLinearMethodControl c;
    c.outer.maximum_operator_applications = 3;
    auto result = solve(h, s, c);
    CHECK_FALSE(result.converged);
    CHECK(result.stop_reason == LinearMethodStopReason::OPERATOR_APPLICATION_LIMIT);
    CHECK(result.operator_applications == 3);
  }
  SECTION("outer iteration limit")
  {
    AcceleratedLinearMethodControl c;
    c.outer.maximum_iterations = 1;
    auto result = solve(h, s, c);
    CHECK(result.stop_reason == LinearMethodStopReason::ITERATION_LIMIT);
    CHECK(result.iterations == 1);
    CHECK(result.operator_applications == 2);
  }
  SECTION("time limit")
  {
    AcceleratedLinearMethodControl c;
    c.outer.maximum_seconds = std::numeric_limits<double>::min();
    auto result = solve(h, s, c);
    CHECK(result.stop_reason == LinearMethodStopReason::TIME_LIMIT);
    CHECK(result.operator_applications == 0);
  }
}

TEST_CASE("Accelerated LM validates overlap and controls", "[drivers][wftrain][linear_method]")
{
  auto model = schema(1);
  Dense h(model, {2, -1, -1, 3}, false);
  SECTION("negative reference overlap")
  {
    Dense s(model, {-1, 0, 0, 1}, true);
    auto result = solve(h, s);
    CHECK_FALSE(result.converged);
    CHECK(result.stop_reason == LinearMethodStopReason::NON_POSITIVE_OVERLAP);
  }
  SECTION("non Hermitian overlap descriptor")
  {
    Dense s(model, {1, 0, 0, 1}, false);
    CHECK_THROWS_AS(solve(h, s), std::invalid_argument);
  }
  SECTION("invalid inner bound")
  {
    Dense s(model, {1, 0, 0, 1}, true);
    AcceleratedLinearMethodControl c;
    c.inner_restart_length = 0;
    CHECK_THROWS_AS(solve(h, s, c), std::invalid_argument);
  }
}

TEST_CASE("Accelerated LM restarts with bounded retained storage", "[drivers][wftrain][linear_method]")
{
  auto model = schema(2);
  Dense h(model, {2, -0.3, 0.2, -0.1, 4, 0.4, 0.5, 0.1, 7}, false);
  Dense s(model, {1, 0, 0, 0, 1, 0, 0, 0, 1}, true);
  AcceleratedLinearMethodControl c;
  c.outer.maximum_subspace_dimension = 2;
  c.outer.maximum_iterations = 4;
  c.outer.relative_residual_tolerance = 1.e-14;
  c.inner_iterations = 1;
  c.inner_restart_length = 1;
  auto result = solve(h, s, c);
  CHECK(result.restart_count >= 1);
  CHECK(result.peak_subspace_dimension <= 2);
  CHECK(result.peak_parameter_vectors <= 9 * 2 + 18 + 2 * 1 + 4);
  CHECK(result.operator_applications <= c.outer.maximum_operator_applications);
}

TEST_CASE("Accelerated LM projected overlap distinguishes cancellation from material asymmetry",
          "[drivers][wftrain][linear_method][alm_roundoff]")
{
  // Exact native125 pair: cancellation in an S-normalized basis with actual
  // Euclidean vector/action scales. The old entry-only test rejected this pair.
  constexpr double left = 0x1.490c2a18bf61cp-57, right = -0x1.5bc21a653482dp-40;
  constexpr double left_scale = 0x1.11c908d9f05a6p-1, right_scale = 0x1.54366a822e4f9p+2;
  const double old_bound = 256 * std::numeric_limits<double>::epsilon() * 18;
  CHECK(std::abs(left - right) > old_bound);
  CHECK(detail::projectedPairIsHermitian(left, right, left_scale, right_scale, 256, 18));
  CHECK_FALSE(detail::projectedPairIsHermitian(left, right + 1e-5, left_scale, right_scale, 256, 18));
  const double infinity = std::numeric_limits<double>::infinity(), maximum = std::numeric_limits<double>::max();
  CHECK_FALSE(detail::projectedPairIsHermitian(left, right, infinity, right_scale, 256, 18));
  CHECK_FALSE(detail::projectedPairIsHermitian(left, right, maximum, maximum, 256, 18));
  CHECK_FALSE(detail::projectedPairIsHermitian(left, right, maximum / 8, maximum / 8, maximum, 18));
  CHECK_FALSE(detail::projectedPairIsHermitian(infinity, right, left_scale, right_scale, 256, 18));
  // A true numerical S asymmetry must still fail in the actual solver, even
  // when the provider claims its descriptor is Hermitian.
  auto model = schema(1);
  Dense h(model, {2, -2, -1, 8}, false);
  Dense asymmetric_s(model, {1, 1e-3, 0, 1}, true);
  const auto result = solve(h, asymmetric_s);
  CHECK_FALSE(result.converged);
  CHECK(result.stop_reason == LinearMethodStopReason::NON_HERMITIAN_ACTION);
}
} // namespace qmcplusplus::wftrain
