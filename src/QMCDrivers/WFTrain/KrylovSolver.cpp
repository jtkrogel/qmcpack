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

KrylovSolveResult solveRestartedGMRES(
    const MatrixFreeLinearOperator& linear_operator,
    const StructuredParameterVectorConstView& right_hand_side,
    const GMRESSolverControl& control,
    const StructuredParameterVectorConstView* warm_start)
{
  validateControl(control.krylov);
  if (control.restart_length == 0)
    throw std::invalid_argument("Restarted GMRES requires a positive restart length");

  const MatrixFreeOperatorDescriptor& descriptor = linear_operator.descriptor();
  if (descriptor.reduction_domain != ReductionDomain::GLOBAL)
    throw std::invalid_argument(
        "Restarted GMRES requires a globally replicated operator action");
  validateVectorIdentity(linear_operator, right_hand_side,
                         "GMRES right-hand side");
  if (warm_start)
    validateVectorIdentity(linear_operator, *warm_start, "GMRES warm start");

  const StructuredParameterSchema& schema = linear_operator.parameterSchema();
  const std::size_t parameter_count       = descriptor.parameter_count;
  const std::size_t restart_length =
      std::min(control.restart_length, parameter_count);
  if (restart_length == 0 ||
      restart_length == std::numeric_limits<std::size_t>::max() ||
      restart_length + 1 >
          std::numeric_limits<std::size_t>::max() / parameter_count ||
      restart_length >
          std::numeric_limits<std::size_t>::max() / (restart_length + 1))
    throw std::length_error("Restarted GMRES workspace extent overflows size_t");

  KrylovSolveResult result;
  result.solution.assign(parameter_count, DerivativeValue{});
  std::vector<DerivativeValue> residual(parameter_count);
  std::vector<DerivativeValue> action(parameter_count);
  std::vector<DerivativeValue> basis(
      (restart_length + 1) * parameter_count);
  std::vector<DerivativeValue> hessenberg(
      (restart_length + 1) * restart_length);
  std::vector<DerivativeReal> rotations_cosine(restart_length);
  std::vector<DerivativeValue> rotations_sine(restart_length);
  std::vector<DerivativeValue> least_squares_rhs(restart_length + 1);
  std::vector<DerivativeValue> coefficients(restart_length);

  auto basisView = [&](std::size_t column) {
    return DerivativeArrayView<DerivativeValue>{
        basis.data() + column * parameter_count, parameter_count};
  };
  auto constBasisView = [&](std::size_t column) {
    return DerivativeArrayView<const DerivativeValue>{
        basis.data() + column * parameter_count, parameter_count};
  };
  auto hessenbergAt = [&](std::size_t row,
                          std::size_t column) -> DerivativeValue& {
    return hessenberg[column * (restart_length + 1) + row];
  };

  const Clock::time_point start = Clock::now();
  try
  {
    copyParameterVector(schema, right_hand_side.values(), mutableView(residual));
    const DerivativeReal right_hand_side_norm =
        parameterVectorNorm(schema, right_hand_side.values());
    result.initial_residual_norm = right_hand_side_norm;
    result.final_residual_norm   = right_hand_side_norm;
    if (right_hand_side_norm == 0.0)
    {
      finishResult(result, KrylovStopReason::ZERO_RIGHT_HAND_SIDE, start);
      return result;
    }

    if (warm_start)
    {
      copyParameterVector(schema, warm_start->values(),
                          mutableView(result.solution));
      if (control.krylov.maximum_operator_applications == 0)
      {
        finishResult(result, KrylovStopReason::OPERATOR_APPLICATION_LIMIT, start);
        return result;
      }
      if (control.krylov.maximum_seconds > 0.0 &&
          elapsedSeconds(start) >= control.krylov.maximum_seconds)
      {
        finishResult(result, KrylovStopReason::TIME_LIMIT, start);
        return result;
      }
      const StructuredParameterVectorConstView solution_view(
          schema, descriptor.parameter_version, constView(result.solution));
      linear_operator.apply(solution_view, mutableView(action));
      ++result.operator_applications;
      axpyParameterVector(schema, {-1.0, 0.0}, constView(action),
                          mutableView(residual));
    }

    result.initial_residual_norm =
        parameterVectorNorm(schema, constView(residual));
    result.final_residual_norm = result.initial_residual_norm;
    const DerivativeReal target = std::max(
        control.krylov.absolute_tolerance,
        control.krylov.relative_tolerance * right_hand_side_norm);
    if (result.initial_residual_norm <= target)
    {
      finishResult(result, KrylovStopReason::CONVERGED, start);
      return result;
    }
    if (control.krylov.maximum_iterations == 0)
    {
      finishResult(result, KrylovStopReason::ITERATION_LIMIT, start);
      return result;
    }

    DerivativeReal best_residual = result.final_residual_norm;
    std::size_t stagnant_iterations = 0;
    while (true)
    {
      std::fill(hessenberg.begin(), hessenberg.end(), DerivativeValue{});
      std::fill(rotations_cosine.begin(), rotations_cosine.end(),
                DerivativeReal{});
      std::fill(rotations_sine.begin(), rotations_sine.end(),
                DerivativeValue{});
      std::fill(least_squares_rhs.begin(), least_squares_rhs.end(),
                DerivativeValue{});
      std::fill(coefficients.begin(), coefficients.end(), DerivativeValue{});

      const DerivativeReal cycle_residual = result.final_residual_norm;
      DerivativeArrayView<DerivativeValue> first_basis = basisView(0);
      for (std::size_t parameter = 0; parameter < parameter_count; ++parameter)
        first_basis[parameter] = residual[parameter] / cycle_residual;
      least_squares_rhs[0] = {cycle_residual, 0.0};

      KrylovStopReason pending_reason = KrylovStopReason::BREAKDOWN;
      bool have_pending_reason = false;
      std::size_t arnoldi_columns = 0;
      for (std::size_t column = 0; column < restart_length; ++column)
      {
        if (result.operator_applications >=
            control.krylov.maximum_operator_applications)
        {
          pending_reason = KrylovStopReason::OPERATOR_APPLICATION_LIMIT;
          have_pending_reason = true;
          break;
        }
        if (control.krylov.maximum_seconds > 0.0 &&
            elapsedSeconds(start) >= control.krylov.maximum_seconds)
        {
          pending_reason = KrylovStopReason::TIME_LIMIT;
          have_pending_reason = true;
          break;
        }

        const StructuredParameterVectorConstView basis_vector(
            schema, descriptor.parameter_version, constBasisView(column));
        linear_operator.apply(basis_vector, mutableView(action));
        ++result.operator_applications;

        // Two-pass modified Gram--Schmidt limits loss of orthogonality for the
        // nonnormal operators that motivate this solver.
        for (int pass = 0; pass < 2; ++pass)
          for (std::size_t row = 0; row <= column; ++row)
          {
            const DerivativeValue projection = parameterVectorHermitianDot(
                schema, constBasisView(row), constView(action));
            hessenbergAt(row, column) += projection;
            axpyParameterVector(schema, -projection, constBasisView(row),
                                mutableView(action));
          }
        const DerivativeReal next_norm =
            parameterVectorNorm(schema, constView(action));
        hessenbergAt(column + 1, column) = {next_norm, 0.0};
        if (next_norm > 0.0)
        {
          DerivativeArrayView<DerivativeValue> next_basis =
              basisView(column + 1);
          for (std::size_t parameter = 0; parameter < parameter_count;
               ++parameter)
            next_basis[parameter] = action[parameter] / next_norm;
        }

        for (std::size_t rotation = 0; rotation < column; ++rotation)
        {
          const DerivativeValue upper = hessenbergAt(rotation, column);
          const DerivativeValue lower = hessenbergAt(rotation + 1, column);
          hessenbergAt(rotation, column) =
              rotations_cosine[rotation] * upper +
              rotations_sine[rotation] * lower;
          hessenbergAt(rotation + 1, column) =
              -std::conj(rotations_sine[rotation]) * upper +
              rotations_cosine[rotation] * lower;
        }

        const DerivativeValue diagonal = hessenbergAt(column, column);
        const DerivativeValue subdiagonal =
            hessenbergAt(column + 1, column);
        const DerivativeReal diagonal_norm = std::abs(diagonal);
        const DerivativeReal subdiagonal_norm = std::abs(subdiagonal);
        const DerivativeReal rotation_norm =
            std::hypot(diagonal_norm, subdiagonal_norm);
        if (rotation_norm == 0.0)
        {
          pending_reason = KrylovStopReason::BREAKDOWN;
          have_pending_reason = true;
          break;
        }
        if (diagonal_norm == 0.0)
        {
          rotations_cosine[column] = 0.0;
          rotations_sine[column] =
              std::conj(subdiagonal) / subdiagonal_norm;
          hessenbergAt(column, column) = {subdiagonal_norm, 0.0};
        }
        else
        {
          const DerivativeValue phase = diagonal / diagonal_norm;
          rotations_cosine[column] = diagonal_norm / rotation_norm;
          rotations_sine[column] =
              phase * std::conj(subdiagonal) / rotation_norm;
          hessenbergAt(column, column) = phase * rotation_norm;
        }
        hessenbergAt(column + 1, column) = {};

        const DerivativeValue current_rhs = least_squares_rhs[column];
        least_squares_rhs[column] =
            rotations_cosine[column] * current_rhs;
        least_squares_rhs[column + 1] =
            -std::conj(rotations_sine[column]) * current_rhs;
        ++result.iterations;
        arnoldi_columns = column + 1;
        result.final_residual_norm =
            std::abs(least_squares_rhs[column + 1]);

        if (result.final_residual_norm <= target)
        {
          pending_reason = KrylovStopReason::CONVERGED;
          have_pending_reason = true;
          break;
        }
        const DerivativeReal required_improvement =
            control.krylov.minimum_relative_improvement * best_residual;
        if (best_residual - result.final_residual_norm > required_improvement)
        {
          best_residual = result.final_residual_norm;
          stagnant_iterations = 0;
        }
        else if (control.krylov.stagnation_window > 0 &&
                 ++stagnant_iterations >= control.krylov.stagnation_window)
        {
          pending_reason = KrylovStopReason::STAGNATED;
          have_pending_reason = true;
          break;
        }
        if (next_norm == 0.0)
        {
          pending_reason = KrylovStopReason::BREAKDOWN;
          have_pending_reason = true;
          break;
        }
        if (result.iterations >= control.krylov.maximum_iterations)
        {
          pending_reason = KrylovStopReason::ITERATION_LIMIT;
          have_pending_reason = true;
          break;
        }
      }

      if (arnoldi_columns == 0)
      {
        finishResult(result,
                     have_pending_reason ? pending_reason
                                         : KrylovStopReason::BREAKDOWN,
                     start);
        return result;
      }

      for (std::size_t reverse = arnoldi_columns; reverse > 0; --reverse)
      {
        const std::size_t row = reverse - 1;
        DerivativeValue value = least_squares_rhs[row];
        for (std::size_t column = row + 1; column < arnoldi_columns;
             ++column)
          value -= hessenbergAt(row, column) * coefficients[column];
        const DerivativeValue diagonal = hessenbergAt(row, row);
        const DerivativeReal threshold =
            std::numeric_limits<DerivativeReal>::epsilon() *
            std::max(DerivativeReal{1}, std::abs(value));
        if (std::abs(diagonal) <= threshold)
        {
          finishResult(result, KrylovStopReason::BREAKDOWN, start);
          return result;
        }
        coefficients[row] = value / diagonal;
      }
      for (std::size_t column = 0; column < arnoldi_columns; ++column)
        axpyParameterVector(schema, coefficients[column], constBasisView(column),
                            mutableView(result.solution));

      if (have_pending_reason)
      {
        finishResult(result, pending_reason, start);
        return result;
      }
      if (result.iterations >= control.krylov.maximum_iterations)
      {
        finishResult(result, KrylovStopReason::ITERATION_LIMIT, start);
        return result;
      }
      if (result.operator_applications >=
          control.krylov.maximum_operator_applications)
      {
        finishResult(result, KrylovStopReason::OPERATOR_APPLICATION_LIMIT, start);
        return result;
      }
      if (control.krylov.maximum_seconds > 0.0 &&
          elapsedSeconds(start) >= control.krylov.maximum_seconds)
      {
        finishResult(result, KrylovStopReason::TIME_LIMIT, start);
        return result;
      }

      // Recompute the true residual at every restart. This prevents recursively
      // accumulated Arnoldi residual estimates from crossing cycle boundaries.
      const StructuredParameterVectorConstView solution_view(
          schema, descriptor.parameter_version, constView(result.solution));
      linear_operator.apply(solution_view, mutableView(action));
      ++result.operator_applications;
      copyParameterVector(schema, right_hand_side.values(), mutableView(residual));
      axpyParameterVector(schema, {-1.0, 0.0}, constView(action),
                          mutableView(residual));
      result.final_residual_norm =
          parameterVectorNorm(schema, constView(residual));
      if (result.final_residual_norm <= target)
      {
        finishResult(result, KrylovStopReason::CONVERGED, start);
        return result;
      }
    }
  }
  catch (const MatrixFreeNumericalError&)
  {
    finishResult(result, KrylovStopReason::NONFINITE_RESULT, start);
    return result;
  }
}

} // namespace qmcplusplus::wftrain
