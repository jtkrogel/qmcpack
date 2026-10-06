//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_MatrixFreeLinearMethod.cpp
 * @brief Dense-oracle tests for the bounded symmetrized matrix-free LM core.
 */

#include "QMCDrivers/WFTrain/MatrixFreeLinearMethod.h"
#include "Utilities/for_testing/Catch2Approx.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Construct a synthetic tangent schema, optionally freezing its final coordinate.
StructuredParameterSchema makeLMSchema(std::string provider,
                                       std::size_t tangent_count,
                                       ParameterScalarDomain domain,
                                       bool freeze_last = false)
{
  std::vector<ParameterBlockDescriptor> blocks;
  const std::size_t trainable_count = freeze_last ? tangent_count - 1 : tangent_count;
  blocks.push_back({"trainable", {trainable_count}, 0, trainable_count, domain,
                    true, "tangent"});
  if (freeze_last)
    blocks.push_back({"frozen", {1}, trainable_count, 1, domain, false, "frozen"});
  return {std::move(provider), std::move(blocks)};
}

/// Provide one deterministic dense P+1 action behind the production checked contract.
class DenseAugmentedOperator final : public AugmentedLinearMethodOperator
{
public:
  DenseAugmentedOperator(const StructuredParameterSchema& schema,
                         std::size_t version,
                         std::vector<DerivativeValue> matrix,
                         bool hermitian = true,
                         ReductionDomain domain = ReductionDomain::GLOBAL)
      : AugmentedLinearMethodOperator(schema, version, domain, hermitian),
        matrix_(std::move(matrix))
  {
    const std::size_t dimension = schema.parameterCount() + 1;
    REQUIRE(matrix_.size() == dimension * dimension);
  }

protected:
  void evaluate(DerivativeArrayView<const DerivativeValue> direction,
                DerivativeArrayView<DerivativeValue> result) const override
  {
    const std::size_t dimension = direction.size();
    for (std::size_t row = 0; row < dimension; ++row)
    {
      result[row] = {};
      for (std::size_t column = 0; column < dimension; ++column)
        result[row] += matrix_[row * dimension + column] * direction[column];
    }
  }

private:
  std::vector<DerivativeValue> matrix_;
};

/// Produce a non-finite result to exercise failure-atomic solver handling.
class NonFiniteAugmentedOperator final : public AugmentedLinearMethodOperator
{
public:
  NonFiniteAugmentedOperator(const StructuredParameterSchema& schema, std::size_t version)
      : AugmentedLinearMethodOperator(schema, version, ReductionDomain::GLOBAL, true)
  {}

protected:
  void evaluate(DerivativeArrayView<const DerivativeValue>,
                DerivativeArrayView<DerivativeValue> result) const override
  {
    std::fill(result.begin(), result.end(), DerivativeValue{});
    result[0] = {std::numeric_limits<double>::infinity(), 0.0};
  }
};

/// Solve a tiny dense problem from the canonical reference-state initial vector.
MatrixFreeLinearMethodResult solveDenseLM(
    const AugmentedLinearMethodOperator& hamiltonian,
    const AugmentedLinearMethodOperator& overlap,
    const MatrixFreePreconditioner& preconditioner,
    MatrixFreeLinearMethodControl control = {})
{
  std::vector<DerivativeValue> initial(
      hamiltonian.tangentSchema().parameterCount() + 1);
  initial[0] = {1.0, 0.0};
  return solveSymmetrizedMatrixFreeLinearMethod(
      hamiltonian, overlap, preconditioner,
      {initial.data(), initial.size()}, control);
}

/// Return a row-major identity matrix of one requested dimension.
std::vector<DerivativeValue> identityMatrix(std::size_t dimension)
{
  std::vector<DerivativeValue> result(dimension * dimension);
  for (std::size_t index = 0; index < dimension; ++index)
    result[index * dimension + index] = {1.0, 0.0};
  return result;
}

} // namespace

TEST_CASE("Symmetrized matrix-free LM matches an analytic generalized eigenpair",
          "[drivers][wftrain][linear_method]")
{
  const StructuredParameterSchema schema =
      makeLMSchema("lm/dense", 2, ParameterScalarDomain::REAL64);

  // S=L*L^T and H=L*C*L^T, where C has analytic lowest eigenvalue
  // 2-sqrt(2).  The nondiagonal L makes this sensitive to the orientation of
  // both whitening and the L^-T back transformation.
  DenseAugmentedOperator hamiltonian(
      schema, 4, {{2.0, 0.0}, {-2.0, 0.0}, {-1.0, 0.0},
                  {-2.0, 0.0}, {8.0, 0.0}, {2.0, 0.0},
                  {-1.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}});
  DenseAugmentedOperator overlap(
      schema, 4, {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                  {0.0, 0.0}, {4.0, 0.0}, {2.0, 0.0},
                  {0.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}});
  IdentityPreconditioner preconditioner(schema, 4);

  MatrixFreeLinearMethodControl control;
  control.relative_residual_tolerance = 1.0e-11;
  control.maximum_subspace_dimension = 3;
  control.maximum_iterations = 8;
  const MatrixFreeLinearMethodResult result =
      solveDenseLM(hamiltonian, overlap, preconditioner, control);

  INFO("stop=" << linearMethodStopReasonName(result.stop_reason)
               << " iterations=" << result.iterations
               << " residual=" << result.residual_norm
               << " restarts=" << result.restart_count);
  REQUIRE(result.converged);
  CHECK(result.stop_reason == LinearMethodStopReason::CONVERGED);
  CHECK(result.eigenvalue == Catch::Approx(2.0 - std::sqrt(2.0)).epsilon(1.0e-10));
  REQUIRE(result.eigenvector.size() == 3);
  CHECK(result.eigenvector[1].real() / result.eigenvector[0].real() ==
        Catch::Approx((std::sqrt(2.0) - 1.0) / 2.0).epsilon(1.0e-9));
  CHECK(result.eigenvector[2].real() / result.eigenvector[0].real() ==
        Catch::Approx(1.0).epsilon(1.0e-9));
  CHECK(result.residual_norm < 1.0e-10);
  CHECK(result.peak_subspace_dimension == 3);
  CHECK(result.peak_parameter_vectors <= 8 + 3 * control.maximum_subspace_dimension);
}

TEST_CASE("Symmetrized matrix-free LM restarts within its configured storage bound",
          "[drivers][wftrain][linear_method]")
{
  const StructuredParameterSchema schema =
      makeLMSchema("lm/restart", 2, ParameterScalarDomain::REAL64);
  DenseAugmentedOperator hamiltonian(
      schema, 2, {{2.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0},
                  {-1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0},
                  {0.0, 0.0}, {-1.0, 0.0}, {2.0, 0.0}});
  DenseAugmentedOperator overlap(schema, 2, identityMatrix(3));
  IdentityPreconditioner preconditioner(schema, 2);

  MatrixFreeLinearMethodControl control;
  control.relative_residual_tolerance = 2.0e-6;
  control.maximum_subspace_dimension = 2;
  control.maximum_iterations = 100;
  control.maximum_operator_applications = 202;
  const MatrixFreeLinearMethodResult result =
      solveDenseLM(hamiltonian, overlap, preconditioner, control);

  INFO("stop=" << linearMethodStopReasonName(result.stop_reason)
               << " iterations=" << result.iterations
               << " residual=" << result.residual_norm
               << " restarts=" << result.restart_count);
  REQUIRE(result.converged);
  CHECK(result.eigenvalue == Catch::Approx(2.0 - std::sqrt(2.0)).epsilon(1.0e-5));
  CHECK(result.restart_count > 0);
  CHECK(result.peak_subspace_dimension == 2);
  CHECK(result.peak_parameter_vectors <= 14);
}

TEST_CASE("Linear-method update normalization is tangent-only and nonpublishing",
          "[drivers][wftrain][linear_method]")
{
  const StructuredParameterSchema schema =
      makeLMSchema("lm/update", 2, ParameterScalarDomain::REAL64, true);
  MatrixFreeLinearMethodResult result;
  result.stop_reason = LinearMethodStopReason::CONVERGED;
  result.converged = true;
  result.provider_id = schema.providerId();
  result.schema_fingerprint = schema.fingerprint();
  result.parameter_version = 9;
  result.eigenvector = {{2.0, 0.0}, {1.0, 0.0}, {-4.0, 0.0}};

  LinearMethodUpdateControl control;
  control.maximum_update_norm = 0.25;
  const LinearMethodUpdateCandidate candidate =
      formLinearMethodUpdateCandidate(schema, result, control);
  REQUIRE(candidate.direction.size() == 2);
  CHECK(candidate.unscaled_norm == Catch::Approx(0.5));
  CHECK(candidate.applied_scale == Catch::Approx(0.5));
  CHECK(candidate.direction[0].real() == Catch::Approx(0.25));
  CHECK(candidate.direction[1] == DerivativeValue{});

  result.converged = false;
  CHECK_THROWS(formLinearMethodUpdateCandidate(schema, result, control));
  result.converged = true;
  result.eigenvector[0] = {0.0, 0.0};
  CHECK_THROWS(formLinearMethodUpdateCandidate(schema, result, control));
}

TEST_CASE("Symmetrized matrix-free LM rejects invalid contracts and overlap",
          "[drivers][wftrain][linear_method]")
{
  const StructuredParameterSchema schema =
      makeLMSchema("lm/contracts", 2, ParameterScalarDomain::REAL64);
  const auto identity = identityMatrix(3);
  DenseAugmentedOperator valid_h(schema, 1, identity);
  DenseAugmentedOperator valid_s(schema, 1, identity);
  IdentityPreconditioner preconditioner(schema, 1);

  DenseAugmentedOperator nonhermitian_contract(
      schema, 1, identity, false, ReductionDomain::GLOBAL);
  CHECK_THROWS(solveDenseLM(nonhermitian_contract, valid_s, preconditioner));
  DenseAugmentedOperator local_contract(
      schema, 1, identity, true, ReductionDomain::CROWD_LOCAL);
  CHECK_THROWS(solveDenseLM(local_contract, valid_s, preconditioner));
  DenseAugmentedOperator wrong_version(schema, 2, identity);
  CHECK_THROWS(solveDenseLM(wrong_version, valid_s, preconditioner));

  std::vector<DerivativeValue> action_input{{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  CHECK_THROWS(valid_h.apply({action_input.data(), action_input.size()},
                             {action_input.data(), action_input.size()}));
  std::vector<DerivativeValue> short_output(2);
  CHECK_THROWS(valid_h.apply({action_input.data(), action_input.size()},
                             {short_output.data(), short_output.size()}));

  DenseAugmentedOperator uncentered_overlap(
      schema, 1, {{1.0, 0.0}, {0.2, 0.0}, {0.0, 0.0},
                  {0.2, 0.0}, {1.0, 0.0}, {0.0, 0.0},
                  {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}});
  CHECK(solveDenseLM(valid_h, uncentered_overlap, preconditioner).stop_reason ==
        LinearMethodStopReason::INVALID_REFERENCE_OVERLAP);

  DenseAugmentedOperator negative_overlap(
      schema, 1, {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                  {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0},
                  {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}});
  DenseAugmentedOperator coupled_h(
      schema, 1, {{2.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0},
                  {-1.0, 0.0}, {2.0, 0.0}, {0.0, 0.0},
                  {0.0, 0.0}, {0.0, 0.0}, {3.0, 0.0}});
  CHECK(solveDenseLM(coupled_h, negative_overlap, preconditioner).stop_reason ==
        LinearMethodStopReason::NON_POSITIVE_OVERLAP);

  // A positive-semidefinite full overlap is permitted, but a correction in its
  // null space must be rejected locally rather than admitted to the basis.
  DenseAugmentedOperator semidefinite_overlap(
      schema, 1, {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                  {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                  {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}});
  CHECK(solveDenseLM(coupled_h, semidefinite_overlap, preconditioner).stop_reason ==
        LinearMethodStopReason::LINEAR_DEPENDENCE);

  const StructuredParameterSchema complex_schema =
      makeLMSchema("lm/complex", 1, ParameterScalarDomain::COMPLEX128);
  CHECK_THROWS(DenseAugmentedOperator(complex_schema, 1, identityMatrix(2)));
}

TEST_CASE("Symmetrized matrix-free LM reports bounded and numerical failures",
          "[drivers][wftrain][linear_method]")
{
  const StructuredParameterSchema schema =
      makeLMSchema("lm/failures", 2, ParameterScalarDomain::REAL64);
  const auto identity = identityMatrix(3);
  DenseAugmentedOperator overlap(schema, 0, identity);
  IdentityPreconditioner preconditioner(schema, 0);
  DenseAugmentedOperator coupled_h(
      schema, 0, {{2.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0},
                  {-1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0},
                  {0.0, 0.0}, {-1.0, 0.0}, {2.0, 0.0}});

  MatrixFreeLinearMethodControl operator_limit;
  operator_limit.maximum_operator_applications = 1;
  const MatrixFreeLinearMethodResult limited =
      solveDenseLM(coupled_h, overlap, preconditioner, operator_limit);
  CHECK(limited.stop_reason == LinearMethodStopReason::OPERATOR_APPLICATION_LIMIT);
  CHECK_FALSE(limited.converged);
  CHECK(limited.eigenvector.empty());

  MatrixFreeLinearMethodControl iteration_limit;
  iteration_limit.maximum_iterations = 1;
  const MatrixFreeLinearMethodResult iterated =
      solveDenseLM(coupled_h, overlap, preconditioner, iteration_limit);
  CHECK(iterated.stop_reason == LinearMethodStopReason::ITERATION_LIMIT);
  CHECK_FALSE(iterated.converged);

  NonFiniteAugmentedOperator nonfinite_h(schema, 0);
  const MatrixFreeLinearMethodResult nonfinite =
      solveDenseLM(nonfinite_h, overlap, preconditioner);
  CHECK(nonfinite.stop_reason == LinearMethodStopReason::NONFINITE_RESULT);
  CHECK(nonfinite.eigenvector.empty());

  MatrixFreeLinearMethodControl root_trust;
  root_trust.minimum_reference_overlap = 1.0e-2;
  root_trust.relative_residual_tolerance = 1.0e-12;
  DenseAugmentedOperator tangent_root_h(
      schema, 0, {{2.0, 0.0}, {1.0e-4, 0.0}, {0.0, 0.0},
                  {1.0e-4, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                  {0.0, 0.0}, {0.0, 0.0}, {3.0, 0.0}});
  CHECK(solveDenseLM(tangent_root_h, overlap, preconditioner, root_trust).stop_reason ==
        LinearMethodStopReason::UNTRUSTED_ROOT);

  // This operator advertises Hermiticity but violates it.  The projected action
  // must be rejected rather than silently defining H_sym from the observed entries.
  DenseAugmentedOperator asymmetric_h(
      schema, 0, {{2.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                  {-1.0, 0.0}, {2.0, 0.0}, {0.0, 0.0},
                  {0.0, 0.0}, {-1.0, 0.0}, {2.0, 0.0}});
  CHECK(solveDenseLM(asymmetric_h, overlap, preconditioner).stop_reason ==
        LinearMethodStopReason::NON_HERMITIAN_ACTION);
}

} // namespace qmcplusplus::wftrain
