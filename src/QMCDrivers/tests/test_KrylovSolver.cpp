//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_KrylovSolver.cpp
 * @brief Dense-oracle tests for bounded matrix-free conjugate gradients.
 */

#include "QMCDrivers/WFTrain/KrylovSolver.h"
#include "Utilities/for_testing/Catch2Approx.h"

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Construct one homogeneous synthetic parameter schema.
StructuredParameterSchema makeKrylovSchema(std::string provider,
                                           std::size_t count,
                                           ParameterScalarDomain domain)
{
  return {std::move(provider),
          {{"parameters", {count}, 0, count, domain, true, "parameters"}}};
}

/// Provide a deterministic dense action while recording allocation identities.
class KrylovDenseOperator final : public MatrixFreeLinearOperator
{
public:
  KrylovDenseOperator(const StructuredParameterSchema& schema,
                      std::size_t version,
                      std::vector<DerivativeValue> matrix,
                      bool hermitian = true,
                      ReductionDomain domain = ReductionDomain::GLOBAL)
      : MatrixFreeLinearOperator(schema, version, domain, hermitian),
        matrix_(std::move(matrix))
  {}

  bool outputAddressStable() const noexcept { return output_address_stable_; }
  std::size_t callCount() const noexcept { return call_count_; }

protected:
  void evaluate(const StructuredParameterVectorConstView& direction,
                DerivativeArrayView<DerivativeValue> result) const override
  {
    if (call_count_ == 0)
      first_output_address_ = result.data();
    else if (result.data() != first_output_address_)
      output_address_stable_ = false;
    ++call_count_;
    const std::size_t count = result.size();
    for (std::size_t row = 0; row < count; ++row)
    {
      result[row] = {};
      for (std::size_t column = 0; column < count; ++column)
        result[row] += matrix_[row * count + column] * direction.values()[column];
    }
  }

private:
  std::vector<DerivativeValue> matrix_;
  mutable const DerivativeValue* first_output_address_ = nullptr;
  mutable std::size_t call_count_ = 0;
  mutable bool output_address_stable_ = true;
};

/// Return a finite zero vector to force the preconditioned-inner-product breakdown.
class ZeroPreconditioner final : public MatrixFreePreconditioner
{
public:
  using MatrixFreePreconditioner::MatrixFreePreconditioner;

protected:
  void evaluate(const StructuredParameterVectorConstView&,
                DerivativeArrayView<DerivativeValue> result) const override
  {
    std::fill(result.begin(), result.end(), DerivativeValue{});
  }
};

/// Return a non-finite value so the solver must report, rather than publish, it.
class NonFinitePreconditioner final : public MatrixFreePreconditioner
{
public:
  using MatrixFreePreconditioner::MatrixFreePreconditioner;

protected:
  void evaluate(const StructuredParameterVectorConstView&,
                DerivativeArrayView<DerivativeValue> result) const override
  {
    std::fill(result.begin(), result.end(), DerivativeValue{});
    result[0] = {std::numeric_limits<double>::quiet_NaN(), 0.0};
  }
};

/// Solve one test system from a plain owning right-hand side.
KrylovSolveResult solve(const MatrixFreeLinearOperator& linear_operator,
                        const MatrixFreePreconditioner& preconditioner,
                        const std::vector<DerivativeValue>& rhs,
                        KrylovSolverControl control = {})
{
  StructuredParameterVectorConstView rhs_view(
      linear_operator.parameterSchema(), linear_operator.descriptor().parameter_version,
      {rhs.data(), rhs.size()});
  return solvePreconditionedConjugateGradient(linear_operator, preconditioner,
                                               rhs_view, control);
}

} // namespace

TEST_CASE("Preconditioned CG matches a real dense SPD oracle", "[drivers][wftrain]")
{
  const StructuredParameterSchema schema =
      makeKrylovSchema("pcg_real", 3, ParameterScalarDomain::REAL64);
  KrylovDenseOperator linear_operator(
      schema, 4, {{4.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
                  {1.0, 0.0}, {3.0, 0.0}, {1.0, 0.0},
                  {0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}});
  const std::vector<DerivativeReal> diagonal{4.0, 3.0, 2.0};
  DiagonalPreconditioner preconditioner(schema, 4,
                                        {diagonal.data(), diagonal.size()}, 1.0e-12);
  const std::vector<DerivativeValue> rhs{{6.0, 0.0}, {10.0, 0.0}, {8.0, 0.0}};
  KrylovSolverControl control;
  control.relative_tolerance = 1.0e-12;
  control.maximum_iterations = 10;
  control.maximum_operator_applications = 10;
  const KrylovSolveResult result = solve(linear_operator, preconditioner, rhs, control);
  CHECK(result.converged);
  CHECK(result.stop_reason == KrylovStopReason::CONVERGED);
  REQUIRE(result.solution.size() == 3);
  CHECK(result.solution[0].real() == Catch::Approx(1.0).epsilon(1.0e-10));
  CHECK(result.solution[1].real() == Catch::Approx(2.0).epsilon(1.0e-10));
  CHECK(result.solution[2].real() == Catch::Approx(3.0).epsilon(1.0e-10));
  CHECK(result.final_residual_norm <= 1.0e-10);

  CHECK(linear_operator.callCount() >= 2);
  CHECK(linear_operator.outputAddressStable());
}

TEST_CASE("Preconditioned CG supports complex Hermitian tangents and warm starts",
          "[drivers][wftrain]")
{
  const StructuredParameterSchema schema =
      makeKrylovSchema("pcg_complex", 2, ParameterScalarDomain::COMPLEX128);
  KrylovDenseOperator linear_operator(
      schema, 2, {{3.0, 0.0}, {1.0, 1.0},
                  {1.0, -1.0}, {2.0, 0.0}});
  IdentityPreconditioner preconditioner(schema, 2);
  const std::vector<DerivativeValue> expected{{1.0, 2.0}, {-0.5, 1.0}};
  const std::vector<DerivativeValue> rhs{{1.5, 6.5}, {2.0, 3.0}};
  StructuredParameterVectorConstView rhs_view(schema, 2, {rhs.data(), rhs.size()});
  StructuredParameterVectorConstView warm_view(schema, 2,
                                                {expected.data(), expected.size()});
  const KrylovSolveResult result = solvePreconditionedConjugateGradient(
      linear_operator, preconditioner, rhs_view, {}, &warm_view);
  CHECK(result.converged);
  CHECK(result.stop_reason == KrylovStopReason::CONVERGED);
  CHECK(result.iterations == 0);
  CHECK(result.operator_applications == 1);
  CHECK(result.solution[0].real() == Catch::Approx(1.0));
  CHECK(result.solution[0].imag() == Catch::Approx(2.0));
  CHECK(result.solution[1].real() == Catch::Approx(-0.5));
  CHECK(result.solution[1].imag() == Catch::Approx(1.0));

  const KrylovSolveResult from_zero = solve(linear_operator, preconditioner, rhs);
  CHECK(from_zero.converged);
  CHECK(from_zero.solution[0].real() == Catch::Approx(1.0).epsilon(1.0e-10));
  CHECK(from_zero.solution[0].imag() == Catch::Approx(2.0).epsilon(1.0e-10));
  CHECK(from_zero.solution[1].real() == Catch::Approx(-0.5).epsilon(1.0e-10));
  CHECK(from_zero.solution[1].imag() == Catch::Approx(1.0).epsilon(1.0e-10));
}

TEST_CASE("Preconditioned CG reports bounded stop conditions", "[drivers][wftrain]")
{
  const StructuredParameterSchema schema =
      makeKrylovSchema("pcg_status", 2, ParameterScalarDomain::REAL64);
  IdentityPreconditioner identity(schema, 0);
  const std::vector<DerivativeValue> rhs{{1.0, 0.0}, {1.0, 0.0}};

  KrylovDenseOperator positive(schema, 0, {{1.0, 0.0}, {0.0, 0.0},
                                            {0.0, 0.0}, {10.0, 0.0}});
  KrylovSolverControl operator_limit;
  operator_limit.maximum_operator_applications = 0;
  CHECK(solve(positive, identity, rhs, operator_limit).stop_reason ==
        KrylovStopReason::OPERATOR_APPLICATION_LIMIT);

  KrylovSolverControl iteration_limit;
  iteration_limit.maximum_iterations = 0;
  CHECK(solve(positive, identity, rhs, iteration_limit).stop_reason ==
        KrylovStopReason::ITERATION_LIMIT);

  KrylovSolverControl time_limit;
  time_limit.maximum_seconds = std::numeric_limits<double>::min();
  CHECK(solve(positive, identity, rhs, time_limit).stop_reason ==
        KrylovStopReason::TIME_LIMIT);

  KrylovSolverControl stagnation;
  stagnation.stagnation_window = 1;
  stagnation.minimum_relative_improvement = 2.0;
  CHECK(solve(positive, identity, rhs, stagnation).stop_reason ==
        KrylovStopReason::STAGNATED);

  KrylovDenseOperator indefinite(schema, 0, {{-1.0, 0.0}, {0.0, 0.0},
                                              {0.0, 0.0}, {1.0, 0.0}});
  CHECK(solve(indefinite, identity, rhs).stop_reason ==
        KrylovStopReason::NON_POSITIVE_CURVATURE);

  ZeroPreconditioner zero(schema, 0);
  CHECK(solve(positive, zero, rhs).stop_reason == KrylovStopReason::BREAKDOWN);

  NonFinitePreconditioner nonfinite(schema, 0);
  CHECK(solve(positive, nonfinite, rhs).stop_reason ==
        KrylovStopReason::NONFINITE_RESULT);

  KrylovDenseOperator nonfinite_action(
      schema, 0,
      {{std::numeric_limits<double>::infinity(), 0.0}, {0.0, 0.0},
       {0.0, 0.0}, {1.0, 0.0}});
  const KrylovSolveResult nonfinite_action_result =
      solve(nonfinite_action, identity, rhs);
  CHECK(nonfinite_action_result.stop_reason ==
        KrylovStopReason::NONFINITE_RESULT);
  CHECK(nonfinite_action_result.operator_applications == 1);

  const StructuredParameterSchema complex_schema =
      makeKrylovSchema("pcg_nonhermitian_action", 1,
                       ParameterScalarDomain::COMPLEX128);
  KrylovDenseOperator complex_nonhermitian(
      complex_schema, 0, {{0.0, 1.0}});
  IdentityPreconditioner complex_identity(complex_schema, 0);
  CHECK(solve(complex_nonhermitian, complex_identity,
              {{1.0, 0.0}}).stop_reason ==
        KrylovStopReason::NON_HERMITIAN_ACTION);
}

TEST_CASE("Preconditioned CG rejects incompatible operator contracts", "[drivers][wftrain]")
{
  const StructuredParameterSchema schema =
      makeKrylovSchema("pcg_contract", 2, ParameterScalarDomain::REAL64);
  IdentityPreconditioner identity(schema, 1);
  const std::vector<DerivativeValue> rhs{{1.0, 0.0}, {0.0, 0.0}};

  KrylovDenseOperator local(schema, 1, {{1.0, 0.0}, {0.0, 0.0},
                                        {0.0, 0.0}, {1.0, 0.0}}, true,
                            ReductionDomain::RANK_LOCAL);
  CHECK_THROWS(solve(local, identity, rhs));

  KrylovDenseOperator nonhermitian(schema, 1, {{1.0, 0.0}, {1.0, 0.0},
                                               {0.0, 0.0}, {1.0, 0.0}}, false);
  CHECK_THROWS(solve(nonhermitian, identity, rhs));

  const std::vector<DerivativeValue> zeros(2);
  KrylovDenseOperator identity_operator(schema, 1,
                                        {{1.0, 0.0}, {0.0, 0.0},
                                         {0.0, 0.0}, {1.0, 0.0}});
  IdentityPreconditioner stale_preconditioner(schema, 0);
  CHECK_THROWS(solve(identity_operator, stale_preconditioner, rhs));
  const KrylovSolveResult zero_result = solve(identity_operator, identity, zeros);
  CHECK(zero_result.stop_reason == KrylovStopReason::ZERO_RIGHT_HAND_SIDE);
  CHECK(zero_result.converged);

  const std::vector<DerivativeValue> nonzero_warm{{2.0, 0.0}, {-1.0, 0.0}};
  StructuredParameterVectorConstView zero_rhs_view(schema, 1,
                                                   {zeros.data(), zeros.size()});
  StructuredParameterVectorConstView warm_view(
      schema, 1, {nonzero_warm.data(), nonzero_warm.size()});
  const KrylovSolveResult zero_rhs_from_warm =
      solvePreconditionedConjugateGradient(identity_operator, identity, zero_rhs_view,
                                            {}, &warm_view);
  CHECK(zero_rhs_from_warm.stop_reason == KrylovStopReason::ZERO_RIGHT_HAND_SIDE);
  CHECK(zero_rhs_from_warm.operator_applications == 0);
  CHECK(zero_rhs_from_warm.solution[0].real() == Catch::Approx(0.0).margin(1.0e-12));
  CHECK(zero_rhs_from_warm.solution[1].real() == Catch::Approx(0.0).margin(1.0e-12));
}

} // namespace qmcplusplus::wftrain
