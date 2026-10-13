//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_WFTRAIN_OPTIMIZER_INPUT_H
#define QMCPLUSPLUS_WFTRAIN_OPTIMIZER_INPUT_H

#include "QMCDrivers/WFTrain/MinimumStepStochasticReconfiguration.h"
#include "QMCDrivers/WFTrain/WFTrainDriverInput.h"

#include <stdexcept>
#include <string_view>

namespace qmcplusplus::wftrain
{

/** Return the stable input spelling used in diagnostics and persisted metadata. */
inline std::string_view optimizerName(WFTrainDriverInput::Optimizer optimizer) noexcept
{
  switch (optimizer)
  {
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
