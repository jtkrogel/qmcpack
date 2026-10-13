//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_WFTRAIN_DRIVER_INPUT_H
#define QMCPLUSPLUS_WFTRAIN_DRIVER_INPUT_H

#include "Configuration.h"

#include <cstddef>
#include <string>

namespace qmcplusplus
{

/** Strict, typed input for the high-parameter wavefunction-training driver. */
class WFTrainDriverInput
{
public:
  enum class Objective
  {
    ENERGY
  };

  enum class Optimizer
  {
    ADAM,
    STOCHASTIC_RECONFIGURATION,
    MINIMUM_STEP_SR,
    SPRING
  };

  /** Parse exactly one nested training section from a qmc driver node. */
  void readXML(xmlNodePtr qmc_node);

  Objective getObjective() const noexcept { return objective_; }
  Optimizer getOptimizer() const noexcept { return optimizer_; }
  std::size_t getParameterChunkSize() const noexcept { return parameter_chunk_size_; }
  double getLearningRate() const noexcept { return learning_rate_; }
  double getAdamBeta1() const noexcept { return adam_beta1_; }
  double getAdamBeta2() const noexcept { return adam_beta2_; }
  double getEpsilon() const noexcept { return epsilon_; }
  double getDamping() const noexcept { return damping_; }
  double getDampingMultiplier() const noexcept { return damping_multiplier_; }
  double getMinimumRetryDamping() const noexcept { return minimum_retry_damping_; }
  std::size_t getMaximumDampingAttempts() const noexcept { return maximum_damping_attempts_; }
  double getMaximumUpdateNorm() const noexcept { return maximum_update_norm_; }
  double getMaximumMetricNorm() const noexcept { return maximum_metric_norm_; }
  double getCenterShift() const noexcept { return center_shift_; }
  double getObjectiveTolerance() const noexcept { return objective_tolerance_; }
  double getSpringHistoryDecay() const noexcept { return spring_history_decay_; }
  double getSpringMaximumUpdateNormSquared() const noexcept { return spring_maximum_update_norm_squared_; }
  double getKrylovRelativeTolerance() const noexcept { return krylov_relative_tolerance_; }
  double getKrylovAbsoluteTolerance() const noexcept { return krylov_absolute_tolerance_; }
  std::size_t getKrylovMaximumIterations() const noexcept { return krylov_maximum_iterations_; }
  std::size_t getKrylovMaximumOperatorApplications() const noexcept
  {
    return krylov_maximum_operator_applications_;
  }
  double getKrylovMaximumSeconds() const noexcept { return krylov_maximum_seconds_; }
  std::size_t getKrylovStagnationWindow() const noexcept { return krylov_stagnation_window_; }
  double getKrylovMinimumRelativeImprovement() const noexcept
  {
    return krylov_minimum_relative_improvement_;
  }
  double getKrylovCurvatureTolerance() const noexcept { return krylov_curvature_tolerance_; }
  double getKrylovImaginaryTolerance() const noexcept { return krylov_imaginary_tolerance_; }
  std::size_t getMaximumRetainedBytes() const noexcept { return maximum_retained_bytes_; }
  const std::string& getCheckpointPath() const noexcept { return checkpoint_path_; }
  const std::string& getRestartPath() const noexcept { return restart_path_; }
  const std::string& getProviderID() const noexcept { return provider_id_; }

private:
  Objective objective_{Objective::ENERGY};
  Optimizer optimizer_{Optimizer::ADAM};
  std::size_t parameter_chunk_size_{65536};
  double learning_rate_{1.0e-3};
  double adam_beta1_{0.9};
  double adam_beta2_{0.999};
  double epsilon_{1.0e-8};
  double damping_{1.0e-3};
  double damping_multiplier_{10.0};
  double minimum_retry_damping_{1.0e-8};
  std::size_t maximum_damping_attempts_{3};
  double maximum_update_norm_{0.0};
  double maximum_metric_norm_{0.0};
  double center_shift_{1.0};
  double objective_tolerance_{1.0e-9};
  double spring_history_decay_{0.99};
  double spring_maximum_update_norm_squared_{0.0};
  double krylov_relative_tolerance_{1.0e-8};
  double krylov_absolute_tolerance_{0.0};
  std::size_t krylov_maximum_iterations_{100};
  std::size_t krylov_maximum_operator_applications_{101};
  double krylov_maximum_seconds_{0.0};
  std::size_t krylov_stagnation_window_{0};
  double krylov_minimum_relative_improvement_{1.0e-4};
  double krylov_curvature_tolerance_{0.0};
  double krylov_imaginary_tolerance_{64.0};
  std::size_t maximum_retained_bytes_{0};
  std::string checkpoint_path_;
  std::string restart_path_;
  std::string provider_id_;
};

} // namespace qmcplusplus

#endif // QMCPLUSPLUS_WFTRAIN_DRIVER_INPUT_H
