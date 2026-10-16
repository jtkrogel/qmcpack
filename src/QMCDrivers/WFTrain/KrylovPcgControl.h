//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file KrylovPcgControl.h
 * @brief Private storage-independent PCG recurrence; no vector addresses cross this boundary.
 */
#ifndef QMCPLUSPLUS_KRYLOV_PCG_CONTROL_H
#define QMCPLUSPLUS_KRYLOV_PCG_CONTROL_H

#include "QMCDrivers/WFTrain/KrylovSolver.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::wftrain::krylov_detail
{
using Clock = std::chrono::steady_clock;

/// Name logical work vectors without imposing host or device storage ownership.
enum class PcgSlot { RHS, WARM_START, X, R, Z, P, AP };

/// Return elapsed wall time from one monotonic start point.
inline double elapsedSeconds(Clock::time_point start) noexcept
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// Validate one complete solver policy before allocating work vectors.
inline void validateControl(const KrylovSolverControl& control)
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

/// Check that a Hermitian contraction has a negligible imaginary residual.
inline bool hermitianScalarIsReal(DerivativeValue value,
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
inline void finishResult(KrylovSolveDiagnostics& result,
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

/** Execute the existing PCG control law over an admitted fixed-capacity backend.
 * Entry identity/control validation and all allocation precede this call. Storage
 * owns every slot and checked operation; diagnostics never imply host ownership.
 */
template<class Storage>
void runPreconditionedConjugateGradient(Storage& storage, const KrylovSolverControl& control,
    std::size_t parameter_count, KrylovSolveDiagnostics& result)
{
  const Clock::time_point start = Clock::now();
  try
  {
    storage.copy(PcgSlot::RHS, PcgSlot::R);
    const DerivativeReal right_hand_side_norm = storage.norm(PcgSlot::RHS);
    if (right_hand_side_norm == 0.0)
    {
      result.initial_residual_norm = 0.0;
      result.final_residual_norm   = 0.0;
      finishResult(result, KrylovStopReason::ZERO_RIGHT_HAND_SIDE, start);
      return;
    }
    if (storage.hasWarmStart())
      storage.copy(PcgSlot::WARM_START, PcgSlot::X);

    if (storage.hasWarmStart())
    {
      result.initial_residual_norm = right_hand_side_norm;
      result.final_residual_norm   = right_hand_side_norm;
      if (control.maximum_operator_applications == 0)
      {
        finishResult(result, KrylovStopReason::OPERATOR_APPLICATION_LIMIT, start);
        return;
      }
      if (control.maximum_seconds > 0.0 && elapsedSeconds(start) >= control.maximum_seconds)
      {
        finishResult(result, KrylovStopReason::TIME_LIMIT, start);
        return;
      }

      ++result.operator_applications;
      storage.applyOperator(PcgSlot::X, PcgSlot::AP);
      storage.axpy(DerivativeValue{-1.0, 0.0}, PcgSlot::AP, PcgSlot::R);
    }

    result.initial_residual_norm = storage.norm(PcgSlot::R);
    result.final_residual_norm   = result.initial_residual_norm;
    const DerivativeReal target = std::max(
        control.absolute_tolerance,
        control.relative_tolerance * right_hand_side_norm);
    if (result.initial_residual_norm == 0.0)
    {
      finishResult(result, KrylovStopReason::CONVERGED, start);
      return;
    }
    if (result.initial_residual_norm <= target)
    {
      finishResult(result, KrylovStopReason::CONVERGED, start);
      return;
    }

    storage.applyPreconditioner(PcgSlot::R, PcgSlot::Z);
    DerivativeValue rho_value = storage.dot(PcgSlot::R, PcgSlot::Z);
    DerivativeReal rho_scale = storage.norm(PcgSlot::R) * storage.norm(PcgSlot::Z);
    if (!hermitianScalarIsReal(rho_value, rho_scale, control.imaginary_tolerance,
                               parameter_count) ||
        rho_value.real() <= 0.0)
    {
      finishResult(result, KrylovStopReason::BREAKDOWN, start);
      return;
    }
    DerivativeReal rho = rho_value.real();
    storage.copy(PcgSlot::Z, PcgSlot::P);

    DerivativeReal best_residual = result.final_residual_norm;
    std::size_t stagnant_iterations = 0;
    while (result.iterations < control.maximum_iterations)
    {
      if (result.operator_applications >= control.maximum_operator_applications)
      {
        finishResult(result, KrylovStopReason::OPERATOR_APPLICATION_LIMIT, start);
        return;
      }
      if (control.maximum_seconds > 0.0 && elapsedSeconds(start) >= control.maximum_seconds)
      {
        finishResult(result, KrylovStopReason::TIME_LIMIT, start);
        return;
      }

      ++result.operator_applications;
      storage.applyOperator(PcgSlot::P, PcgSlot::AP);
      const DerivativeValue curvature_value = storage.dot(PcgSlot::P, PcgSlot::AP);
      const DerivativeReal curvature_scale = storage.norm(PcgSlot::P) * storage.norm(PcgSlot::AP);
      if (!hermitianScalarIsReal(curvature_value, curvature_scale,
                                 control.imaginary_tolerance, parameter_count))
      {
        finishResult(result, KrylovStopReason::NON_HERMITIAN_ACTION, start);
        return;
      }
      result.last_curvature = curvature_value.real();
      const DerivativeReal minimum_curvature =
          control.curvature_tolerance * std::max(DerivativeReal{1}, curvature_scale);
      if (result.last_curvature <= minimum_curvature)
      {
        finishResult(result, KrylovStopReason::NON_POSITIVE_CURVATURE, start);
        return;
      }

      const DerivativeReal alpha = rho / result.last_curvature;
      if (!isFiniteTrainingReal(alpha))
      {
        finishResult(result, KrylovStopReason::BREAKDOWN, start);
        return;
      }
      storage.axpy(DerivativeValue{alpha, 0.0}, PcgSlot::P, PcgSlot::X);
      storage.axpy(DerivativeValue{-alpha, 0.0}, PcgSlot::AP, PcgSlot::R);
      ++result.iterations;

      result.final_residual_norm = storage.norm(PcgSlot::R);
      if (result.final_residual_norm <= target)
      {
        finishResult(result, KrylovStopReason::CONVERGED, start);
        return;
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
        return;
      }

      storage.applyPreconditioner(PcgSlot::R, PcgSlot::Z);
      const DerivativeValue next_rho_value = storage.dot(PcgSlot::R, PcgSlot::Z);
      rho_scale = storage.norm(PcgSlot::R) * storage.norm(PcgSlot::Z);
      if (!hermitianScalarIsReal(next_rho_value, rho_scale,
                                 control.imaginary_tolerance, parameter_count) ||
          next_rho_value.real() <= 0.0)
      {
        finishResult(result, KrylovStopReason::BREAKDOWN, start);
        return;
      }

      const DerivativeReal beta = next_rho_value.real() / rho;
      if (!isFiniteTrainingReal(beta) || beta < 0.0)
      {
        finishResult(result, KrylovStopReason::BREAKDOWN, start);
        return;
      }
      storage.scale(DerivativeValue{beta, 0.0}, PcgSlot::P);
      storage.axpy(DerivativeValue{1.0, 0.0}, PcgSlot::Z, PcgSlot::P);
      rho = next_rho_value.real();
    }

    finishResult(result, KrylovStopReason::ITERATION_LIMIT, start);
    return;
  }
  catch (const MatrixFreeNumericalError&)
  {
    finishResult(result, KrylovStopReason::NONFINITE_RESULT, start);
    return;
  }
}

} // namespace qmcplusplus::wftrain::krylov_detail
#endif
