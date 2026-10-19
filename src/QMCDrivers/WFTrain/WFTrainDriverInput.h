//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_WFTRAIN_DRIVER_INPUT_H
#define QMCPLUSPLUS_WFTRAIN_DRIVER_INPUT_H

#include "Configuration.h"
#include "QMCWaveFunctions/Optimization/OrbitalLossNormalization.h"

#include <cstddef>
#include <limits>
#include <string>

namespace qmcplusplus
{

/** Strict training input: objective/optimizer attributes and named scalar parameters. */
class WFTrainDriverInput
{
public:
  enum class KFACObjective { RAW_COVARIANCE, CLIPPED_COVARIANCE, COMPONENTWISE_MAD_COVARIANCE, COMPONENTWISE_MEAN_RESIDUAL };

  enum class KFACDefaults { LEGACY, SOFT_COVARIANCE, MEAN_RESIDUAL, MEDIAN_COVARIANCE };

  enum class KFACGradient { LEGACY, PHYSICAL_COVARIANCE, HALF_COVARIANCE, REAL_CENTERED_RESIDUAL };

  enum class KFACUpdateOrder { POST_INVERSE_RATE, PRE_INVERSE_RATE };

  enum class KFACScheduleArithmetic { DIVIDE_RATE, MULTIPLY_RECIPROCAL };

  enum class KFACClipMode { SOFT_MEDIAN, HARD_MEDIAN, HARD_MEAN };
  enum class KFACCurvature { NATIVE_SCORE, REGISTERED_GAUSSIAN };
  enum class KFACDamping { EQUAL_SPLIT, TRACE_BALANCED };
  enum class KFACConstraint { NONE, DAMPED_KFAC };
  /// Select fixed scaling, correlated backtracking, or sampled endpoint expansion.
  enum class LineSearch
  {
    OFF,
    REFERENCE,
    ENDPOINTS
  };

  enum class Objective
  {
    ENERGY,
    DETERMINANT_OVERLAP,
    ORBITAL_MSE
  };

  /// Choose repeated occupied orbitals or the largest-coefficient CI terms.
  enum class OrbitalTarget
  {
    SINGLE_REFERENCE,
    TOP_COEFFICIENT_MULTIDETERMINANT
  };

  /// Select the fixed teacher measure or the current student's sampling measure.
  enum class OrbitalSampling { TEACHER, STUDENT };

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
    MINIMUM_PROJECTED_INVERSE_ITERATION,
    SGD,
    MOMENTUM_SGD,
    RMSPROP,
    AMSGRAD,
    LINEAR_METHOD,
    ACCELERATED_LINEAR_METHOD,
    KFAC
  };

  enum class PIIShiftMode
  {
    EXPLICIT,
    MEAN_ENERGY_MINUS_UNDERSHOOT
  };

  /** Parse exactly one nested training section from a qmc driver node. */
  void readXML(xmlNodePtr qmc_node);

  /** Total outer optimization attempts, including restored and rejected attempts. */
  std::size_t getIterations() const noexcept { return iterations_; }

  /// Completed outer attempts between native checkpoint exports.
  std::size_t getCheckpointInterval() const noexcept { return checkpoint_interval_; }

  /** Return discarded MC advances before every iteration's sample collection. */
  std::size_t getWarmupSteps() const noexcept { return warmup_steps_; }

  Objective getObjective() const noexcept { return objective_; }

  OrbitalSampling getOrbitalSampling() const noexcept { return orbital_sampling_; }

  /// Select the per-entry supervised loss convention without changing its target gauge.
  wftrain::OrbitalLossNormalization getOrbitalNormalization() const noexcept { return orbital_normalization_; }

  /// Student-measure pretraining never evaluates Hamiltonians while drawing samples.
  bool samplesOrbitalStudent() const noexcept
  { return objective_ == Objective::ORBITAL_MSE && orbital_sampling_ == OrbitalSampling::STUDENT; }
  OrbitalTarget getOrbitalTarget() const noexcept { return orbital_target_; }
  LineSearch getLineSearch() const noexcept { return line_search_; }
  double getUpdateScale() const noexcept { return update_scale_; }
  double getWeightDecay() const noexcept { return weight_decay_; }
  double getLineSearchMinimumESSFraction() const noexcept { return line_search_minimum_ess_fraction_; }
  double getLineSearchMaximumWeight() const noexcept { return line_search_maximum_weight_; }
  double getLineSearchConfidence() const noexcept { return line_search_confidence_; }
  std::size_t getLineSearchWarmupSteps() const noexcept { return line_search_warmup_steps_; }
  std::size_t getLineSearchSamplingSteps() const noexcept { return line_search_sampling_steps_; }
  Optimizer getOptimizer() const noexcept { return optimizer_; }
  double getMomentumDecay() const noexcept { return momentum_decay_; }
  double getRMSDecay() const noexcept { return rms_decay_; }
  double getLMOverlapShift() const noexcept { return lm_overlap_shift_; }
  double getLMNormalizationXi() const noexcept { return lm_normalization_xi_; }
  std::size_t getLMMaximumSubspaceDimension() const noexcept { return lm_maximum_subspace_dimension_; }
  std::size_t getLMInnerIterations() const noexcept { return lm_inner_iterations_; }
  std::size_t getLMInnerRestartLength() const noexcept { return lm_inner_restart_length_; }
  double getLMInnerRelativeTolerance() const noexcept { return lm_inner_relative_tolerance_; }
  double getKFACFallbackDamping() const noexcept { return kfac_fallback_damping_; }
  KFACObjective getKFACObjective() const noexcept { return kfac_objective_; }

  /// Identify the selected bundle independently of its explicitly overridden settings.
  KFACDefaults getKFACDefaults() const noexcept { return kfac_defaults_; }

  /// Return the finite-sample gradient convention, not a learning-rate compensation.
  KFACGradient getKFACGradient() const noexcept { return kfac_gradient_; }

  /// Locate schedule scaling relative to inverse and constraint arithmetic.
  KFACUpdateOrder getKFACUpdateOrder() const noexcept { return kfac_update_order_; }

  /// Preserve the scalar evaluation order selected for the inverse schedule.
  KFACScheduleArithmetic getKFACScheduleArithmetic() const noexcept { return kfac_schedule_arithmetic_; }

  KFACClipMode getKFACClipMode() const noexcept { return kfac_clip_mode_; }
  KFACCurvature getKFACCurvature() const noexcept { return kfac_curvature_; }
  KFACDamping getKFACDampingPolicy() const noexcept { return kfac_damping_policy_; }
  KFACConstraint getKFACConstraint() const noexcept { return kfac_constraint_; }
  double getKFACClipWidth() const noexcept { return kfac_clip_width_; }
  double getKFACComplexClipWidth() const noexcept { return kfac_complex_clip_width_; }
  double getKFACClipQuantile() const noexcept { return kfac_clip_quantile_; }
  double getKFACExcludeWidth() const noexcept { return kfac_exclude_width_; }
  double getKFACNormConstraint() const noexcept { return kfac_norm_constraint_; }
  bool getKFACPersistState() const noexcept { return kfac_persist_state_; }
  double getKFACEMADecay() const noexcept { return kfac_ema_decay_; }
  std::size_t getKFACInversePeriod() const noexcept { return kfac_inverse_period_; }
  double getKFACLearningRateDecay() const noexcept { return kfac_learning_rate_decay_; }
  bool getKFACScheduleFloat32() const noexcept { return kfac_schedule_float32_; }
  std::size_t getParameterChunkSize() const noexcept { return parameter_chunk_size_; }
  std::size_t getLegacyMaximumParameters() const noexcept { return legacy_maximum_parameters_; }
  std::size_t getLegacyDerivativeScratchBytes() const noexcept { return legacy_derivative_scratch_bytes_; }
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

  /// Explicit additional derivative/successor-generation caps; all absent keeps GPU training closed.
  struct ComplexGpuTraining
  {
    std::size_t device_bytes = 0, host_bytes = 0, kinetic_targets = 0, maximum_matrix_work = 0;
    /// All four validated fields are either absent or enabled together.
    bool enabled() const noexcept { return device_bytes != 0; }
  };

  /// Resource policy is immutable during a training stage, not a process-peak budget.
  const ComplexGpuTraining& getComplexGpuTraining() const noexcept { return complex_gpu_training_; }
  /// Optional measured process-RSS guard; distinct from numeric workspace accounting.
  std::size_t getMaximumResidentBytes() const noexcept { return maximum_resident_bytes_; }
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
  std::size_t iterations_{1};
  std::size_t checkpoint_interval_{1000};
  std::size_t warmup_steps_{0};
  double momentum_decay_{0.9};
  double rms_decay_{0.9};
  double lm_overlap_shift_{1.0e-4};
  double lm_normalization_xi_{0.5};
  std::size_t lm_maximum_subspace_dimension_{8};
  std::size_t lm_inner_iterations_{8};
  std::size_t lm_inner_restart_length_{4};
  double lm_inner_relative_tolerance_{0.1};
  double kfac_fallback_damping_{1.0e-3};
  KFACObjective kfac_objective_{KFACObjective::RAW_COVARIANCE};
  KFACDefaults kfac_defaults_{KFACDefaults::LEGACY};
  KFACGradient kfac_gradient_{KFACGradient::LEGACY};
  KFACUpdateOrder kfac_update_order_{KFACUpdateOrder::POST_INVERSE_RATE};
  KFACScheduleArithmetic kfac_schedule_arithmetic_{KFACScheduleArithmetic::DIVIDE_RATE};
  KFACClipMode kfac_clip_mode_{KFACClipMode::SOFT_MEDIAN};
  KFACCurvature kfac_curvature_{KFACCurvature::NATIVE_SCORE};
  KFACDamping kfac_damping_policy_{KFACDamping::EQUAL_SPLIT};
  KFACConstraint kfac_constraint_{KFACConstraint::NONE};
  double kfac_clip_width_{1.0};
  double kfac_complex_clip_width_{5.0};
  double kfac_clip_quantile_{0.95};
  double kfac_exclude_width_{std::numeric_limits<double>::infinity()};
  double kfac_norm_constraint_{0.0};
  bool kfac_persist_state_{false};
  double kfac_ema_decay_{0.0};
  std::size_t kfac_inverse_period_{1};
  double kfac_learning_rate_decay_{0.0};
  bool kfac_schedule_float32_{false};
  LineSearch line_search_{LineSearch::OFF};
  double update_scale_{1.0};
  double weight_decay_{0.0};
  double line_search_minimum_ess_fraction_{0.5};
  double line_search_maximum_weight_{0.25};
  double line_search_confidence_{2.0};
  std::size_t line_search_warmup_steps_{10};
  std::size_t line_search_sampling_steps_{10};
  Objective objective_{Objective::ENERGY};
  OrbitalSampling orbital_sampling_{OrbitalSampling::TEACHER};
  wftrain::OrbitalLossNormalization orbital_normalization_{wftrain::OrbitalLossNormalization::SPIN_SECTOR_SUM};
  OrbitalTarget orbital_target_{OrbitalTarget::SINGLE_REFERENCE};
  Optimizer optimizer_{Optimizer::ADAM};
  std::size_t parameter_chunk_size_{65536};
  std::size_t legacy_maximum_parameters_{0};
  std::size_t legacy_derivative_scratch_bytes_{0};
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
  ComplexGpuTraining complex_gpu_training_;
  std::size_t maximum_resident_bytes_{0};
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
