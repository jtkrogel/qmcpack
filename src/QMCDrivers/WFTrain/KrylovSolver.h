//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file KrylovSolver.h
 * @brief Allocation-stable Krylov solves for checked matrix-free operators.
 */

#ifndef QMCPLUSPLUS_KRYLOV_SOLVER_H
#define QMCPLUSPLUS_KRYLOV_SOLVER_H

#include "QMCDrivers/WFTrain/MatrixFreeOperator.h"

#include <cstddef>
#include <vector>

namespace qmcplusplus::wftrain
{

/// Identify why a bounded preconditioned conjugate-gradient solve stopped.
enum class KrylovStopReason
{
  CONVERGED,
  ZERO_RIGHT_HAND_SIDE,
  ITERATION_LIMIT,
  OPERATOR_APPLICATION_LIMIT,
  TIME_LIMIT,
  STAGNATED,
  NON_POSITIVE_CURVATURE,
  NON_HERMITIAN_ACTION,
  NONFINITE_RESULT,
  BREAKDOWN
};

/// Bound solver work and define scale-independent convergence diagnostics.
struct KrylovSolverControl
{
  DerivativeReal relative_tolerance = 1.0e-8;
  DerivativeReal absolute_tolerance = 0.0;
  std::size_t maximum_iterations = 100;
  std::size_t maximum_operator_applications = 101;
  double maximum_seconds = 0.0;
  std::size_t stagnation_window = 0;
  DerivativeReal minimum_relative_improvement = 1.0e-4;
  DerivativeReal curvature_tolerance = 0.0;
  DerivativeReal imaginary_tolerance = 64.0;
};

/// Scalar-only outcome of one bounded Krylov solve, independent of vector storage.
struct KrylovSolveDiagnostics
{
  KrylovStopReason stop_reason = KrylovStopReason::BREAKDOWN;
  bool converged = false;
  std::size_t iterations = 0;
  std::size_t operator_applications = 0;
  DerivativeReal initial_residual_norm = 0.0;
  DerivativeReal final_residual_norm = 0.0;
  DerivativeReal relative_residual_norm = 0.0;
  DerivativeReal last_curvature = 0.0;
  double elapsed_seconds = 0.0;
};

/** Owning host candidate and diagnostics from one bounded Krylov solve.
 * A nonconverged candidate is diagnostic only until the caller checks its stop reason.
 */
struct KrylovSolveResult : KrylovSolveDiagnostics
{
  std::vector<DerivativeValue> solution;
};

/// Bound the retained Arnoldi basis used by restarted GMRES.
struct GMRESSolverControl
{
  KrylovSolverControl krylov;
  /// Maximum Arnoldi vectors per restart cycle; must be positive.
  std::size_t restart_length = 20;
};

/** Solve A*x=b with preconditioned CG using a fixed number of O(P) vectors.
 *
 * The operator must advertise a globally replicated Hermitian action. The optional
 * warm start must match the same schema and parameter version exactly.
 */
KrylovSolveResult solvePreconditionedConjugateGradient(
    const MatrixFreeLinearOperator& linear_operator,
    const MatrixFreePreconditioner& preconditioner,
    const StructuredParameterVectorConstView& right_hand_side,
    const KrylovSolverControl& control,
    const StructuredParameterVectorConstView* warm_start = nullptr);

/** Solve a globally replicated, possibly nonsymmetric system with restarted GMRES.
 *
 * Modified Gram--Schmidt and complex Givens rotations retain at most
 * `(restart_length + 1) * P` basis values plus O(P) work vectors. The optional
 * warm start must match the operator's exact schema and parameter version.
 */
KrylovSolveResult solveRestartedGMRES(
    const MatrixFreeLinearOperator& linear_operator,
    const StructuredParameterVectorConstView& right_hand_side,
    const GMRESSolverControl& control,
    const StructuredParameterVectorConstView* warm_start = nullptr);

/// Return the stable diagnostic name of one Krylov stop reason.
const char* krylovStopReasonName(KrylovStopReason reason) noexcept;

} // namespace qmcplusplus::wftrain

#endif
