//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_MatrixFreeOperator.cpp
 * @brief Deterministic tests for checked matrix-free actions and preconditioners.
 */

#include "QMCDrivers/WFTrain/MatrixFreeOperator.h"
#include "Utilities/for_testing/Catch2Approx.h"

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Construct one-block test metadata for a dense synthetic tangent space.
StructuredParameterSchema makeSchema(std::string provider,
                                     std::size_t count,
                                     ParameterScalarDomain domain)
{
  return {std::move(provider),
          {{"weights", {count}, 0, count, domain, true, "weights"}}};
}

/// Apply a small row-major dense matrix through the production checked interface.
class DenseTestOperator final : public MatrixFreeLinearOperator
{
public:
  DenseTestOperator(const StructuredParameterSchema& schema,
                    std::size_t version,
                    std::vector<DerivativeValue> matrix,
                    bool hermitian = true,
                    ReductionDomain domain = ReductionDomain::GLOBAL)
      : MatrixFreeLinearOperator(schema, version, domain, hermitian),
        matrix_(std::move(matrix))
  {
    REQUIRE(matrix_.size() == schema.parameterCount() * schema.parameterCount());
  }

protected:
  void evaluate(const StructuredParameterVectorConstView& direction,
                DerivativeArrayView<DerivativeValue> result) const override
  {
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
};

/// Deliberately violate the numeric output contract for one failure-boundary test.
class NonFiniteTestOperator final : public MatrixFreeLinearOperator
{
public:
  NonFiniteTestOperator(const StructuredParameterSchema& schema, std::size_t version)
      : MatrixFreeLinearOperator(schema, version, ReductionDomain::GLOBAL, true)
  {}

protected:
  void evaluate(const StructuredParameterVectorConstView&,
                DerivativeArrayView<DerivativeValue> result) const override
  {
    std::fill(result.begin(), result.end(), DerivativeValue{});
    result[0] = {std::numeric_limits<double>::infinity(), 0.0};
  }
};

/// Write only one entry so the checked wrapper must detect an incomplete action.
class IncompleteTestOperator final : public MatrixFreeLinearOperator
{
public:
  IncompleteTestOperator(const StructuredParameterSchema& schema, std::size_t version)
      : MatrixFreeLinearOperator(schema, version, ReductionDomain::GLOBAL, true)
  {}

protected:
  void evaluate(const StructuredParameterVectorConstView&,
                DerivativeArrayView<DerivativeValue> result) const override
  {
    result[0] = {};
  }
};

} // namespace

TEST_CASE("Matrix-free vector algebra preserves scalar domains", "[drivers][wftrain]")
{
  const StructuredParameterSchema real_schema =
      makeSchema("matrix_free_real", 3, ParameterScalarDomain::REAL64);
  std::vector<DerivativeValue> x{{1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}};
  std::vector<DerivativeValue> y{{3.0, 0.0}, {-2.0, 0.0}, {4.0, 0.0}};
  CHECK(parameterVectorHermitianDot(real_schema, {x.data(), x.size()},
                                    {y.data(), y.size()}) == DerivativeValue{-5.0, 0.0});
  CHECK(parameterVectorNorm(real_schema, {x.data(), x.size()}) ==
        Catch::Approx(std::sqrt(6.0)));
  axpyParameterVector(real_schema, {2.0, 0.0}, {x.data(), x.size()},
                      {y.data(), y.size()});
  CHECK(y[0].real() == Catch::Approx(5.0));
  CHECK(y[1].real() == Catch::Approx(2.0));
  CHECK(y[2].real() == Catch::Approx(2.0));
  CHECK_THROWS(scaleParameterVector(real_schema, {1.0, 0.5}, {y.data(), y.size()}));

  // Exact aliasing is a useful in-place scale, while offset overlap would make the
  // sequential kernel order-dependent and is rejected before any value is changed.
  axpyParameterVector(real_schema, {-0.5, 0.0}, {x.data(), x.size()},
                      {x.data(), x.size()});
  CHECK(x[0].real() == Catch::Approx(0.5));
  std::vector<DerivativeValue> overlap{{1.0, 0.0}, {2.0, 0.0},
                                       {3.0, 0.0}, {4.0, 0.0}};
  CHECK_THROWS(axpyParameterVector(
      real_schema, {1.0, 0.0}, {overlap.data(), 3}, {overlap.data() + 1, 3}));
  CHECK(overlap[1].real() == Catch::Approx(2.0));

  const StructuredParameterSchema complex_schema =
      makeSchema("matrix_free_complex", 2, ParameterScalarDomain::COMPLEX128);
  std::vector<DerivativeValue> cx{{1.0, 2.0}, {-3.0, 1.0}};
  std::vector<DerivativeValue> cy{{2.0, -1.0}, {0.5, 4.0}};
  const DerivativeValue dot = parameterVectorHermitianDot(
      complex_schema, {cx.data(), cx.size()}, {cy.data(), cy.size()});
  CHECK(dot.real() == Catch::Approx(2.5));
  CHECK(dot.imag() == Catch::Approx(-17.5));
}

TEST_CASE("Matrix-free operators validate identity shift and aliasing", "[drivers][wftrain]")
{
  const StructuredParameterSchema schema =
      makeSchema("matrix_free_action", 2, ParameterScalarDomain::REAL64);
  DenseTestOperator base(schema, 7, {{2.0, 0.0}, {1.0, 0.0},
                                     {1.0, 0.0}, {3.0, 0.0}});
  ShiftedMatrixFreeOperator shifted(base, 0.5);
  std::vector<DerivativeValue> direction{{1.0, 0.0}, {-2.0, 0.0}};
  std::vector<DerivativeValue> result(2);
  StructuredParameterVectorConstView view(schema, 7,
                                          {direction.data(), direction.size()});
  shifted.apply(view, {result.data(), result.size()});
  CHECK(result[0].real() == Catch::Approx(0.5));
  CHECK(result[1].real() == Catch::Approx(-6.0));

  CHECK_THROWS(base.apply(view, {direction.data(), direction.size()}));
  CHECK_THROWS(base.apply(view, {result.data(), result.size() - 1}));
  StructuredParameterVectorConstView stale(schema, 6,
                                           {direction.data(), direction.size()});
  CHECK_THROWS(base.apply(stale, {result.data(), result.size()}));
  const StructuredParameterSchema other_schema =
      makeSchema("matrix_free_other", 2, ParameterScalarDomain::REAL64);
  StructuredParameterVectorConstView other_view(
      other_schema, 7, {direction.data(), direction.size()});
  CHECK_THROWS(base.apply(other_view, {result.data(), result.size()}));
  CHECK_THROWS(ShiftedMatrixFreeOperator(base, -0.1));

  NonFiniteTestOperator nonfinite(schema, 7);
  CHECK_THROWS_AS(nonfinite.apply(view, {result.data(), result.size()}),
                  MatrixFreeNumericalError);

  IncompleteTestOperator incomplete(schema, 7);
  CHECK_THROWS_AS(incomplete.apply(view, {result.data(), result.size()}),
                  MatrixFreeNumericalError);

  DenseTestOperator imaginary_output(
      schema, 7, {{1.0, 1.0}, {0.0, 0.0},
                  {0.0, 0.0}, {1.0, 0.0}});
  CHECK_THROWS_AS(imaginary_output.apply(view, {result.data(), result.size()}),
                  MatrixFreeNumericalError);

  // A view validates on construction, but its nonowning backing storage can later be
  // modified. The action boundary must reject that mutation before a derived action
  // can hide it behind a finite result.
  direction[0] = {std::numeric_limits<double>::quiet_NaN(), 0.0};
  CHECK_THROWS(incomplete.apply(view, {result.data(), result.size()}));
  direction[0] = {1.0, 1.0};
  CHECK_THROWS(incomplete.apply(view, {result.data(), result.size()}));
}

TEST_CASE("Matrix-free diagonal preconditioning is bounded and checked", "[drivers][wftrain]")
{
  const StructuredParameterSchema schema =
      makeSchema("matrix_free_preconditioner", 3, ParameterScalarDomain::REAL64);
  const std::vector<DerivativeReal> diagonal{4.0, 0.0, 2.0};
  DiagonalPreconditioner preconditioner(schema, 3,
                                        {diagonal.data(), diagonal.size()}, 0.5);
  std::vector<DerivativeValue> residual{{8.0, 0.0}, {1.0, 0.0}, {-4.0, 0.0}};
  std::vector<DerivativeValue> result(3);
  StructuredParameterVectorConstView view(schema, 3,
                                          {residual.data(), residual.size()});
  preconditioner.apply(view, {result.data(), result.size()});
  CHECK(result[0].real() == Catch::Approx(2.0));
  CHECK(result[1].real() == Catch::Approx(2.0));
  CHECK(result[2].real() == Catch::Approx(-2.0));
  CHECK(preconditioner.inverseDiagonal().size() == schema.parameterCount());
  CHECK_THROWS(preconditioner.apply(view, {residual.data(), residual.size()}));

  const std::vector<DerivativeReal> negative{1.0, -1.0, 2.0};
  CHECK_THROWS(DiagonalPreconditioner(schema, 3,
                                      {negative.data(), negative.size()}, 0.5));
  CHECK_THROWS(DiagonalPreconditioner(schema, 3,
                                      {diagonal.data(), diagonal.size()}, 0.0));
  CHECK_THROWS(DiagonalPreconditioner(
      schema, 3, {diagonal.data(), diagonal.size()},
      std::numeric_limits<DerivativeReal>::denorm_min()));

  residual[0] = {std::numeric_limits<double>::quiet_NaN(), 0.0};
  CHECK_THROWS(preconditioner.apply(view, {result.data(), result.size()}));
}

TEST_CASE("Matrix-free contracts own copied schema metadata", "[drivers][wftrain]")
{
  DenseTestOperator linear_operator(
      makeSchema("matrix_free_owned", 1, ParameterScalarDomain::REAL64), 5,
      {{2.0, 0.0}});
  IdentityPreconditioner preconditioner(linear_operator.parameterSchema(), 5);
  std::vector<DerivativeValue> input{{3.0, 0.0}};
  std::vector<DerivativeValue> output(1);
  StructuredParameterVectorConstView view(linear_operator.parameterSchema(), 5,
                                          {input.data(), input.size()});
  linear_operator.apply(view, {output.data(), output.size()});
  CHECK(output[0].real() == Catch::Approx(6.0));
  preconditioner.apply(view, {output.data(), output.size()});
  CHECK(output[0].real() == Catch::Approx(3.0));
}

} // namespace qmcplusplus::wftrain
