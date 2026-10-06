//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file KrylovSolver.cpp
 * @brief Bounded preconditioned conjugate-gradient implementation.
 */

#include "QMCDrivers/WFTrain/KrylovSolver.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace qmcplusplus::wftrain
{
namespace
{

using Clock = std::chrono::steady_clock;

/// Create a const interval over one owning work vector.
DerivativeArrayView<const DerivativeValue> constView(
    const std::vector<DerivativeValue>& values) noexcept
{
  return {values.data(), values.size()};
}

/// Create a mutable interval over one owning work vector.
DerivativeArrayView<DerivativeValue> mutableView(
    std::vector<DerivativeValue>& values) noexcept
{
  return {values.data(), values.size()};
}

/// Return elapsed wall time from one monotonic start point.
double elapsedSeconds(Clock::time_point start) noexcept
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// Validate one complete solver policy before allocating work vectors.
void validateControl(const KrylovSolverControl& control)
{
  if (!isFiniteTrainingReal(control.relative_tolerance) ||
      control.relative_tolerance < 0.0 ||
      !isFiniteTrainingReal(control.absolute_tolerance) ||
      control.absolute_tolerance < 0.0 ||
      !isFiniteTrainingReal(control.minimum_relative_improvement) ||
      control.minimum_relative_improvement < 0.0 ||
      !isFiniteTrainingReal(control.curvature_tolerance) ||
      control.curvature_tolerance < 0.0 ||
      !isFiniteTrainingReal(control.imaginary_tolerance) ||
      control.imaginary_tolerance <= 0.0 ||
      !isFiniteTrainingReal(control.maximum_seconds) ||
      control.maximum_seconds < 0.0)
    throw std::invalid_argument("Krylov solver controls must be finite and nonnegative");
  if (control.relative_tolerance == 0.0 && control.absolute_tolerance == 0.0)
    throw std::invalid_argument("Krylov solver requires a positive relative or absolute tolerance");
}

/// Require a vector view to match one operator's exact identity.
void validateVectorIdentity(const MatrixFreeLinearOperator& linear_operator,
                            const StructuredParameterVectorConstView& vector,
                            const char* description)
{
  const MatrixFreeOperatorDescriptor& descriptor = linear_operator.descriptor();
  if (vector.providerId() != descriptor.provider_id ||
      vector.schemaFingerprint() != descriptor.schema_fingerprint ||
      vector.parameterVersion() != descriptor.parameter_version ||
      vector.values().size() != descriptor.parameter_count)
    throw std::invalid_argument(std::string(description) +
                                " does not match the matrix-free operator identity");
}

/// Check that a Hermitian contraction has a negligible imaginary residual.
bool hermitianScalarIsReal(DerivativeValue value,
                           DerivativeReal scale,
                           DerivativeReal tolerance_multiplier,
                           std::size_t parameter_count) noexcept
{
  const DerivativeReal accumulated_roundoff =
      static_cast<DerivativeReal>(parameter_count) *
      std::numeric_limits<DerivativeReal>::epsilon();
  const DerivativeReal gamma = accumulated_roundoff < 0.5
      ? accumulated_roundoff / (1.0 - accumulated_roundoff)
      : 1.0;
  const DerivativeReal tolerance = tolerance_multiplier * gamma *
      std::max({DerivativeReal{1}, std::abs(value.real()), scale});
  return std::abs(value.imag()) <= tolerance;
}

/// Finalize common residual and timing diagnostics for one return path.
void finishResult(KrylovSolveResult& result,
                  KrylovStopReason reason,
                  Clock::time_point start) noexcept
{
  result.stop_reason = reason;
  result.converged   = reason == KrylovStopReason::CONVERGED ||
      reason == KrylovStopReason::ZERO_RIGHT_HAND_SIDE;
  result.relative_residual_norm = result.initial_residual_norm == 0.0
      ? 0.0
      : result.final_residual_norm / result.initial_residual_norm;
  result.elapsed_seconds = elapsedSeconds(start);
}

} // namespace

const char* krylovStopReasonName(KrylovStopReason reason) noexcept
{
  switch (reason)
  {
  case KrylovStopReason::CONVERGED:
    return "converged";
  case KrylovStopReason::ZERO_RIGHT_HAND_SIDE:
    return "zero_right_hand_side";
  case KrylovStopReason::ITERATION_LIMIT:
    return "iteration_limit";
  case KrylovStopReason::OPERATOR_APPLICATION_LIMIT:
    return "operator_application_limit";
  case KrylovStopReason::TIME_LIMIT:
    return "time_limit";
  case KrylovStopReason::STAGNATED:
    return "stagnated";
  case KrylovStopReason::NON_POSITIVE_CURVATURE:
    return "non_positive_curvature";
  case KrylovStopReason::NON_HERMITIAN_ACTION:
    return "non_hermitian_action";
  case KrylovStopReason::NONFINITE_RESULT:
    return "nonfinite_result";
  case KrylovStopReason::BREAKDOWN:
    return "breakdown";
  }
  return "unknown";
}

KrylovSolveResult solvePreconditionedConjugateGradient(
    const MatrixFreeLinearOperator& linear_operator,
    const MatrixFreePreconditioner& preconditioner,
    const StructuredParameterVectorConstView& right_hand_side,
    const KrylovSolverControl& control,
    const StructuredParameterVectorConstView* warm_start)
{
  validateControl(control);
  const MatrixFreeOperatorDescriptor& descriptor = linear_operator.descriptor();
  if (!descriptor.hermitian)
    throw std::invalid_argument("Preconditioned CG requires a Hermitian matrix-free operator");
  if (descriptor.reduction_domain != ReductionDomain::GLOBAL)
    throw std::invalid_argument("Preconditioned CG requires a globally replicated operator action");
  if (preconditioner.parameterSchema().fingerprint() != descriptor.schema_fingerprint ||
      preconditioner.parameterSchema().providerId() != descriptor.provider_id ||
      preconditioner.parameterVersion() != descriptor.parameter_version)
    throw std::invalid_argument("Preconditioner identity does not match the matrix-free operator");
  validateVectorIdentity(linear_operator, right_hand_side, "Krylov right-hand side");
  if (warm_start)
    validateVectorIdentity(linear_operator, *warm_start, "Krylov warm start");

  const StructuredParameterSchema& schema = linear_operator.parameterSchema();
  const std::size_t parameter_count       = descriptor.parameter_count;
  KrylovSolveResult result;
  result.solution.assign(parameter_count, DerivativeValue{});

  // These five vectors are the complete solver-owned O(P) workspace. Their capacities
  // remain fixed throughout the iteration loop.
  std::vector<DerivativeValue> residual(parameter_count);
  std::vector<DerivativeValue> preconditioned_residual(parameter_count);
  std::vector<DerivativeValue> search_direction(parameter_count);
  std::vector<DerivativeValue> action(parameter_count);

  const Clock::time_point start = Clock::now();
  try
  {
    copyParameterVector(schema, right_hand_side.values(), mutableView(residual));
    const DerivativeReal right_hand_side_norm =
        parameterVectorNorm(schema, right_hand_side.values());
    if (right_hand_side_norm == 0.0)
    {
      result.initial_residual_norm = 0.0;
      result.final_residual_norm   = 0.0;
      finishResult(result, KrylovStopReason::ZERO_RIGHT_HAND_SIDE, start);
      return result;
    }
    if (warm_start)
      copyParameterVector(schema, warm_start->values(), mutableView(result.solution));

    if (warm_start)
    {
      result.initial_residual_norm = right_hand_side_norm;
      result.final_residual_norm   = right_hand_side_norm;
      if (control.maximum_operator_applications == 0)
      {
        finishResult(result, KrylovStopReason::OPERATOR_APPLICATION_LIMIT, start);
        return result;
      }
      if (control.maximum_seconds > 0.0 && elapsedSeconds(start) >= control.maximum_seconds)
      {
        finishResult(result, KrylovStopReason::TIME_LIMIT, start);
        return result;
      }

      ++result.operator_applications;
      StructuredParameterVectorConstView solution_view(
          schema, descriptor.parameter_version, constView(result.solution));
      linear_operator.apply(solution_view, mutableView(action));
      axpyParameterVector(schema, DerivativeValue{-1.0, 0.0}, constView(action),
                          mutableView(residual));
    }

    result.initial_residual_norm = parameterVectorNorm(schema, constView(residual));
    result.final_residual_norm   = result.initial_residual_norm;
    const DerivativeReal target = std::max(
        control.absolute_tolerance,
        control.relative_tolerance * right_hand_side_norm);
    if (result.initial_residual_norm == 0.0)
    {
      finishResult(result, KrylovStopReason::CONVERGED, start);
      return result;
    }
    if (result.initial_residual_norm <= target)
    {
      finishResult(result, KrylovStopReason::CONVERGED, start);
      return result;
    }

    StructuredParameterVectorConstView residual_view(
        schema, descriptor.parameter_version, constView(residual));
    preconditioner.apply(residual_view, mutableView(preconditioned_residual));

    DerivativeValue rho_value = parameterVectorHermitianDot(
        schema, constView(residual), constView(preconditioned_residual));
    DerivativeReal rho_scale = parameterVectorNorm(schema, constView(residual)) *
        parameterVectorNorm(schema, constView(preconditioned_residual));
    if (!hermitianScalarIsReal(rho_value, rho_scale, control.imaginary_tolerance,
                               parameter_count) ||
        rho_value.real() <= 0.0)
    {
      finishResult(result, KrylovStopReason::BREAKDOWN, start);
      return result;
    }
    DerivativeReal rho = rho_value.real();
    copyParameterVector(schema, constView(preconditioned_residual),
                        mutableView(search_direction));

    DerivativeReal best_residual = result.final_residual_norm;
    std::size_t stagnant_iterations = 0;
    while (result.iterations < control.maximum_iterations)
    {
      if (result.operator_applications >= control.maximum_operator_applications)
      {
        finishResult(result, KrylovStopReason::OPERATOR_APPLICATION_LIMIT, start);
        return result;
      }
      if (control.maximum_seconds > 0.0 && elapsedSeconds(start) >= control.maximum_seconds)
      {
        finishResult(result, KrylovStopReason::TIME_LIMIT, start);
        return result;
      }

      ++result.operator_applications;
      StructuredParameterVectorConstView direction_view(
          schema, descriptor.parameter_version, constView(search_direction));
      linear_operator.apply(direction_view, mutableView(action));
      const DerivativeValue curvature_value = parameterVectorHermitianDot(
          schema, constView(search_direction), constView(action));
      const DerivativeReal curvature_scale =
          parameterVectorNorm(schema, constView(search_direction)) *
          parameterVectorNorm(schema, constView(action));
      if (!hermitianScalarIsReal(curvature_value, curvature_scale,
                                 control.imaginary_tolerance, parameter_count))
      {
        finishResult(result, KrylovStopReason::NON_HERMITIAN_ACTION, start);
        return result;
      }
      result.last_curvature = curvature_value.real();
      const DerivativeReal minimum_curvature =
          control.curvature_tolerance * std::max(DerivativeReal{1}, curvature_scale);
      if (result.last_curvature <= minimum_curvature)
      {
        finishResult(result, KrylovStopReason::NON_POSITIVE_CURVATURE, start);
        return result;
      }

      const DerivativeReal alpha = rho / result.last_curvature;
      if (!isFiniteTrainingReal(alpha))
      {
        finishResult(result, KrylovStopReason::BREAKDOWN, start);
        return result;
      }
      axpyParameterVector(schema, DerivativeValue{alpha, 0.0},
                          constView(search_direction), mutableView(result.solution));
      axpyParameterVector(schema, DerivativeValue{-alpha, 0.0}, constView(action),
                          mutableView(residual));
      ++result.iterations;

      result.final_residual_norm = parameterVectorNorm(schema, constView(residual));
      if (result.final_residual_norm <= target)
      {
        finishResult(result, KrylovStopReason::CONVERGED, start);
        return result;
      }

      const DerivativeReal required_improvement =
          control.minimum_relative_improvement * best_residual;
      if (best_residual - result.final_residual_norm > required_improvement)
      {
        best_residual = result.final_residual_norm;
        stagnant_iterations = 0;
      }
      else if (control.stagnation_window > 0 &&
               ++stagnant_iterations >= control.stagnation_window)
      {
        finishResult(result, KrylovStopReason::STAGNATED, start);
        return result;
      }

      StructuredParameterVectorConstView next_residual_view(
          schema, descriptor.parameter_version, constView(residual));
      preconditioner.apply(next_residual_view, mutableView(preconditioned_residual));
      const DerivativeValue next_rho_value = parameterVectorHermitianDot(
          schema, constView(residual), constView(preconditioned_residual));
      rho_scale = parameterVectorNorm(schema, constView(residual)) *
          parameterVectorNorm(schema, constView(preconditioned_residual));
      if (!hermitianScalarIsReal(next_rho_value, rho_scale,
                                 control.imaginary_tolerance, parameter_count) ||
          next_rho_value.real() <= 0.0)
      {
        finishResult(result, KrylovStopReason::BREAKDOWN, start);
        return result;
      }

      const DerivativeReal beta = next_rho_value.real() / rho;
      if (!isFiniteTrainingReal(beta) || beta < 0.0)
      {
        finishResult(result, KrylovStopReason::BREAKDOWN, start);
        return result;
      }
      scaleParameterVector(schema, DerivativeValue{beta, 0.0},
                           mutableView(search_direction));
      axpyParameterVector(schema, DerivativeValue{1.0, 0.0},
                          constView(preconditioned_residual),
                          mutableView(search_direction));
      rho = next_rho_value.real();
    }

    finishResult(result, KrylovStopReason::ITERATION_LIMIT, start);
    return result;
  }
  catch (const MatrixFreeNumericalError&)
  {
    finishResult(result, KrylovStopReason::NONFINITE_RESULT, start);
    return result;
  }
}

} // namespace qmcplusplus::wftrain
