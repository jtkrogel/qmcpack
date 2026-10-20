//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_WFTRAIN_OPTIMIZER_INPUT_H
#define QMCPLUSPLUS_WFTRAIN_OPTIMIZER_INPUT_H

#include "QMCDrivers/WFTrain/MinimumStepStochasticReconfiguration.h"
#include "QMCDrivers/WFTrain/EnergyGradientAccumulator.h"
#include "QMCDrivers/WFTrain/WFTrainDriverInput.h"

#include <stdexcept>
#include <string_view>

namespace qmcplusplus::wftrain
{

/// Explicit score conventions select the same estimator on CPU and GPU providers.
inline bool hasExplicitKFACPolicy(const WFTrainDriverInput& input) noexcept
{
  using I = WFTrainDriverInput;
  return input.getKFACDefaults() != I::KFACDefaults::LEGACY ||
      input.getKFACGradient() != I::KFACGradient::LEGACY ||
      input.getKFACUpdateOrder() != I::KFACUpdateOrder::POST_INVERSE_RATE ||
      input.getKFACScheduleArithmetic() != I::KFACScheduleArithmetic::DIVIDE_RATE;
}

/// Preserve legacy energy-response routing unless a score convention was selected.
inline bool usesKFACScoreEstimator(const WFTrainDriverInput& input, bool paired_complex) noexcept
{
  return input.getOptimizer() == WFTrainDriverInput::Optimizer::KFAC &&
      (paired_complex || input.getKFACDefaults() != WFTrainDriverInput::KFACDefaults::LEGACY ||
       input.getKFACGradient() != WFTrainDriverInput::KFACGradient::LEGACY);
}

/// Input part of the bounded real/open hybrid contract; the driver checks physics and topology.
inline bool supportsHybridKFACPolicy(const WFTrainDriverInput& input) noexcept
{
  using I = WFTrainDriverInput;
  const auto objective = input.getKFACObjective();
  const auto gradient = input.getKFACGradient();
  return input.getOptimizer() == I::Optimizer::KFAC && input.getObjective() == I::Objective::ENERGY &&
      input.getLineSearch() == I::LineSearch::OFF && hasExplicitKFACPolicy(input) &&
      (objective == I::KFACObjective::RAW_COVARIANCE || objective == I::KFACObjective::CLIPPED_COVARIANCE) &&
      (gradient == I::KFACGradient::HALF_COVARIANCE || gradient == I::KFACGradient::PHYSICAL_COVARIANCE) &&
      input.getMaximumRetainedBytes() > 0 &&
      input.getMaximumRetainedBytes() != std::numeric_limits<std::size_t>::max();
}

/// Choose the finite-sample score convention independently of numerical placement.
inline EnergyGradientEstimator kfacScoreEstimator(const WFTrainDriverInput& input) noexcept
{
  using G = WFTrainDriverInput::KFACGradient;
  if (input.getKFACGradient() == G::HALF_COVARIANCE)
    return EnergyGradientEstimator::HALF_SCORE_COVARIANCE;
  if (input.getKFACGradient() == G::REAL_CENTERED_RESIDUAL)
    return EnergyGradientEstimator::REAL_CENTERED_SCORE_RESIDUAL;
  return EnergyGradientEstimator::SCORE_COVARIANCE;
}

/** Return the stable input spelling used in diagnostics and persisted metadata. */
inline std::string_view optimizerName(WFTrainDriverInput::Optimizer optimizer) noexcept
{
  switch (optimizer)
  {
  case WFTrainDriverInput::Optimizer::SGD: return "sgd";
  case WFTrainDriverInput::Optimizer::MOMENTUM_SGD: return "momentum_sgd";
  case WFTrainDriverInput::Optimizer::RMSPROP: return "rmsprop";
  case WFTrainDriverInput::Optimizer::AMSGRAD: return "amsgrad";
  case WFTrainDriverInput::Optimizer::LINEAR_METHOD: return "lm";
  case WFTrainDriverInput::Optimizer::ACCELERATED_LINEAR_METHOD: return "alm";
  case WFTrainDriverInput::Optimizer::KFAC: return "kfac";
  case WFTrainDriverInput::Optimizer::ADAM:
    return "adam";
  case WFTrainDriverInput::Optimizer::STOCHASTIC_RECONFIGURATION:
    return "sr";
  case WFTrainDriverInput::Optimizer::MINIMUM_STEP_SR:
    return "minsr";
  case WFTrainDriverInput::Optimizer::SPRING:
    return "spring";
  case WFTrainDriverInput::Optimizer::PROJECTED_INVERSE_ITERATION:
    return "pii";
  case WFTrainDriverInput::Optimizer::MINIMUM_PROJECTED_INVERSE_ITERATION:
    return "minpii";
  }
  return "unknown";
}

/** Report optimizers whose production derivative path requires REAL64 parameters. */
inline bool optimizerRequiresRealParameters(WFTrainDriverInput::Optimizer optimizer) noexcept
{
  return optimizer != WFTrainDriverInput::Optimizer::ADAM;
}

/** Translate strict driver input into the exact primitive PII control contract. */
inline PIIUpdateControl makePIIUpdateControl(const WFTrainDriverInput& input)
{
  const auto optimizer = input.getOptimizer();
  if (optimizer != WFTrainDriverInput::Optimizer::PROJECTED_INVERSE_ITERATION &&
      optimizer != WFTrainDriverInput::Optimizer::MINIMUM_PROJECTED_INVERSE_ITERATION)
    throw std::invalid_argument("PII controls require optimizer=\"pii\" or optimizer=\"minpii\"");

  PIIUpdateControl control;
  control.learning_rate = input.getLearningRate();
  control.shift_mode = input.getPIIShiftMode() == WFTrainDriverInput::PIIShiftMode::EXPLICIT
      ? PIIShiftMode::EXPLICIT : PIIShiftMode::MEAN_ENERGY_MINUS_UNDERSHOOT;
  control.explicit_shift = input.getPIIExplicitTau();
  control.energy_undershoot = input.getPIIEnergyUndershoot();
  control.damping = input.getDamping();
  control.damping_multiplier = input.getDampingMultiplier();
  control.maximum_damping_retries = input.getMaximumDampingRetries();
  control.maximum_update_norm = input.getMaximumUpdateNorm();
  control.objective_tolerance = input.getObjectiveTolerance();
  control.gmres.restart_length = input.getGMRESRestartLength();
  control.gmres.krylov.relative_tolerance = input.getKrylovRelativeTolerance();
  control.gmres.krylov.absolute_tolerance = input.getKrylovAbsoluteTolerance();
  control.gmres.krylov.maximum_iterations = input.getKrylovMaximumIterations();
  control.gmres.krylov.maximum_operator_applications =
      input.getKrylovMaximumOperatorApplications();
  control.gmres.krylov.maximum_seconds = input.getKrylovMaximumSeconds();
  control.gmres.krylov.stagnation_window = input.getKrylovStagnationWindow();
  control.gmres.krylov.minimum_relative_improvement =
      input.getKrylovMinimumRelativeImprovement();
  control.gmres.krylov.curvature_tolerance = input.getKrylovCurvatureTolerance();
  control.gmres.krylov.imaginary_tolerance = input.getKrylovImaginaryTolerance();
  return control;
}

} // namespace qmcplusplus::wftrain

#endif // QMCPLUSPLUS_WFTRAIN_OPTIMIZER_INPUT_H
