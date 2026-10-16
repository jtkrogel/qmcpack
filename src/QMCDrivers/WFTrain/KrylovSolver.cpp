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
#include "QMCDrivers/WFTrain/KrylovPcgControl.h"
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

using krylov_detail::Clock;
using krylov_detail::elapsedSeconds;
using krylov_detail::finishResult;
using krylov_detail::hermitianScalarIsReal;
using krylov_detail::validateControl;
using krylov_detail::PcgSlot;

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

/** Retain the existing five host work vectors and checked numerical wrappers.
 * RHS and optional warm start remain borrowed host views for this synchronous solve.
 */
class HostPcgStorage
{
public:
  /// Borrow inputs and allocate r/z/p/Ap after the separately owned host solution.
  HostPcgStorage(const MatrixFreeLinearOperator& linear_operator,
      const MatrixFreePreconditioner& preconditioner,
      const StructuredParameterVectorConstView& rhs,
      const StructuredParameterVectorConstView* warm_start,
      std::vector<DerivativeValue>& solution)
      : linear_operator_(linear_operator), preconditioner_(preconditioner),
        schema_(linear_operator.parameterSchema()), rhs_(rhs), warm_start_(warm_start),
        solution_(solution), residual_(solution.size()), preconditioned_residual_(solution.size()),
        search_direction_(solution.size()), action_(solution.size())
  {}

  /// Preserve the optional host warm-start branch in the shared recurrence.
  bool hasWarmStart() const noexcept { return warm_start_ != nullptr; }

  /// Delegate every storage transformation to the existing checked host algebra.
  void copy(PcgSlot source, PcgSlot destination)
  { copyParameterVector(schema_, read(source), write(destination)); }

  /// Retain separate checked host scaling before any subsequent AXPY.
  void scale(DerivativeValue factor, PcgSlot values)
  { scaleParameterVector(schema_, factor, write(values)); }

  /// Preserve finite/alias admission and host multiply/add order.
  void axpy(DerivativeValue factor, PcgSlot source, PcgSlot destination)
  { axpyParameterVector(schema_, factor, read(source), write(destination)); }

  /// Preserve the host Hermitian contraction and raw-squared norm arithmetic.
  DerivativeValue dot(PcgSlot left, PcgSlot right)
  { return parameterVectorHermitianDot(schema_, read(left), read(right)); }

  /// Keep the existing raw Hermitian-square norm, including its overflow policy.
  DerivativeReal norm(PcgSlot values)
  { return parameterVectorNorm(schema_, read(values)); }

  /// Construct ordinary host schema views only at the existing operator boundary.
  void applyOperator(PcgSlot source, PcgSlot destination)
  {
    const StructuredParameterVectorConstView view(
        schema_, linear_operator_.descriptor().parameter_version, read(source));
    linear_operator_.apply(view, write(destination));
  }

  /// Keep the checked host preconditioner wrapper and distinct destination slot.
  void applyPreconditioner(PcgSlot source, PcgSlot destination)
  {
    const StructuredParameterVectorConstView view(
        schema_, linear_operator_.descriptor().parameter_version, read(source));
    preconditioner_.apply(view, write(destination));
  }

private:
  /// Expose only borrowed host intervals; external RHS/warm slots are read-only.
  DerivativeArrayView<const DerivativeValue> read(PcgSlot slot)
  {
    if (slot == PcgSlot::RHS)
      return rhs_.values();
    if (slot == PcgSlot::WARM_START)
    {
      if (!warm_start_)
        throw std::logic_error("PCG storage has no warm start");
      return warm_start_->values();
    }
    const auto values = write(slot);
    return {values.data(), values.size()};
  }

  /// Name exactly the original solution/residual/preconditioned/search/action owners.
  DerivativeArrayView<DerivativeValue> write(PcgSlot slot)
  {
    switch (slot)
    {
    case PcgSlot::X: return mutableView(solution_);
    case PcgSlot::R: return mutableView(residual_);
    case PcgSlot::Z: return mutableView(preconditioned_residual_);
    case PcgSlot::P: return mutableView(search_direction_);
    case PcgSlot::AP: return mutableView(action_);
    default: throw std::logic_error("PCG storage destination is not a mutable work slot");
    }
  }

  const MatrixFreeLinearOperator& linear_operator_;
  const MatrixFreePreconditioner& preconditioner_;
  const StructuredParameterSchema& schema_;
  const StructuredParameterVectorConstView& rhs_;
  const StructuredParameterVectorConstView* warm_start_;
  std::vector<DerivativeValue>& solution_;
  std::vector<DerivativeValue> residual_, preconditioned_residual_, search_direction_, action_;
};

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

/// Admit host identities/storage, then run the shared PCG law with an owning host result.
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

  const std::size_t parameter_count = descriptor.parameter_count;
  KrylovSolveResult result;
  result.solution.assign(parameter_count, DerivativeValue{});

  // Preserve solution/r/z/p/Ap allocation order before the shared loop starts its clock.
  HostPcgStorage storage(linear_operator, preconditioner, right_hand_side, warm_start, result.solution);
  krylov_detail::runPreconditionedConjugateGradient(storage, control, parameter_count, result);
  return result;
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
