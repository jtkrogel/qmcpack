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
    ENERGY,
    DETERMINANT_OVERLAP
  };

  enum class OverlapTarget
  {
    GENERAL_FROZEN_TEACHER,
    TOP_COEFFICIENT_MULTIDETERMINANT
  };

  enum class OverlapWeighting
  {
    EQUAL,
    COEFFICIENT_SQUARED
  };

  enum class Optimizer
  {
    ADAM,
    STOCHASTIC_RECONFIGURATION,
    MINIMUM_STEP_SR,
    SPRING,
    PROJECTED_INVERSE_ITERATION,
    MINIMUM_PROJECTED_INVERSE_ITERATION
  };

  enum class PIIShiftMode
  {
    EXPLICIT,
    MEAN_ENERGY_MINUS_UNDERSHOOT
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
  PIIShiftMode getPIIShiftMode() const noexcept { return pii_shift_mode_; }
  double getPIIExplicitTau() const noexcept { return pii_explicit_tau_; }
  double getPIIEnergyUndershoot() const noexcept { return pii_energy_undershoot_; }
  std::size_t getMaximumDampingRetries() const noexcept { return maximum_damping_retries_; }
  std::size_t getGMRESRestartLength() const noexcept { return gmres_restart_length_; }
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
  /** Return the postconstruction ceiling for retained and optimizer workspace bytes.
   *
   * Component batch-memory policies remain responsible for rejecting unsafe clone
   * and resource allocations before construction.  This driver-level ceiling is
   * checked once exact live-owner capacities and conservative solver peaks are
   * available; it is not a process-wide allocator limit.
   */
  std::size_t getMaximumRetainedBytes() const noexcept { return maximum_retained_bytes_; }
  const std::string& getCheckpointPath() const noexcept { return checkpoint_path_; }
  const std::string& getRestartPath() const noexcept { return restart_path_; }
  const std::string& getProviderID() const noexcept { return provider_id_; }
  const std::string& getTeacherWaveFunction() const noexcept { return teacher_wavefunction_; }
  OverlapTarget getOverlapTarget() const noexcept { return overlap_target_; }
  OverlapWeighting getOverlapWeighting() const noexcept { return overlap_weighting_; }
  const std::string& getTargetProvenance() const noexcept { return target_provenance_; }
  double getMinimumFidelity() const noexcept { return minimum_fidelity_; }
  double getMinimumEffectiveSamples() const noexcept { return minimum_effective_samples_; }

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
  PIIShiftMode pii_shift_mode_{PIIShiftMode::MEAN_ENERGY_MINUS_UNDERSHOOT};
  double pii_explicit_tau_{0.0};
  double pii_energy_undershoot_{0.1};
  std::size_t maximum_damping_retries_{2};
  std::size_t gmres_restart_length_{20};
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
  std::string teacher_wavefunction_;
  OverlapTarget overlap_target_{OverlapTarget::GENERAL_FROZEN_TEACHER};
  OverlapWeighting overlap_weighting_{OverlapWeighting::EQUAL};
  std::string target_provenance_;
  double minimum_fidelity_{1.0e-14};
  double minimum_effective_samples_{1.0};
};

} // namespace qmcplusplus

#endif // QMCPLUSPLUS_WFTRAIN_DRIVER_INPUT_H
