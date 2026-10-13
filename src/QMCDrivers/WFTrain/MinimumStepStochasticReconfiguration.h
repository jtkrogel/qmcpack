//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file MinimumStepStochasticReconfiguration.h
 * @brief Distributed sample-space minSR and transactional SPRING updates.
 *
 * The score factor is represented only through streamed JVP/VJP products.  Sample
 * vectors remain rank local while parameter vectors remain globally replicated;
 * neither a sample-by-parameter table nor a parameter-squared matrix is formed.
 */

#ifndef QMCPLUSPLUS_MINIMUM_STEP_STOCHASTIC_RECONFIGURATION_H
#define QMCPLUSPLUS_MINIMUM_STEP_STOCHASTIC_RECONFIGURATION_H

#include "QMCDrivers/WFTrain/DistributedParameterReduction.h"
#include "QMCDrivers/WFTrain/KrylovSolver.h"
#include "QMCDrivers/WFTrain/ParameterUpdateTransaction.h"
#include "QMCDrivers/WFTrain/TrainingCheckpoint.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace qmcplusplus::wftrain
{

/// Select the real sample representation used by the score factor.
enum class ScoreSampleDomain
{
  REAL,
  COMPLEX_REALIFIED
};

/** Bind one local sample batch to copied weights and local energies.
 *
 * The derivative producer is nonowning and must outlive the factor and every solve
 * that uses it.
 */
struct ScoreCovarianceBatch
{
  const StreamingDerivativeOperator* derivative_operator = nullptr;
  DerivativeArrayView<const DerivativeReal> weights;
  DerivativeArrayView<const DerivativeValue> local_energies;
};

/// Exact numeric storage retained by a prepared score factor.
struct ScoreCovarianceStorageDiagnostics
{
  std::size_t parameter_count = 0;
  std::size_t local_physical_sample_count = 0;
  std::size_t local_sample_vector_size = 0;
  std::size_t global_sample_vector_size = 0;
  std::size_t real_sample_values = 0;
  std::size_t complex_sample_values = 0;
  std::size_t complex_parameter_vectors = 0;
  std::size_t retained_numeric_bytes = 0;
};

/** Matrix-free weighted centered score factor A and its real transpose.
 *
 * With q_i=w_i/sum(w), A maps a real parameter direction to
 * sqrt(q_i)[J_i v - sum_j q_j J_j v].  COMPLEX_REALIFIED emits adjacent real and
 * imaginary sample coordinates.  applyAdjoint() implements the exact real transpose.
 */
class ScoreCovarianceFactor
{
public:
  ScoreCovarianceFactor(
      const StructuredParameterSchema& schema,
      std::size_t parameter_version,
      DerivativeArrayView<const ScoreCovarianceBatch> batches,
      ScoreSampleDomain sample_domain = ScoreSampleDomain::REAL,
      DistributedParameterReduction reduction = {});
  ~ScoreCovarianceFactor();

  ScoreCovarianceFactor(const ScoreCovarianceFactor&) = delete;
  ScoreCovarianceFactor& operator=(const ScoreCovarianceFactor&) = delete;

  /// Return the immutable parameter schema bound to all products.
  const StructuredParameterSchema& parameterSchema() const noexcept { return schema_; }

  /// Return the parameter version captured when the factor was prepared.
  std::size_t parameterVersion() const noexcept { return parameter_version_; }

  /// Return the chosen real sample representation.
  ScoreSampleDomain sampleDomain() const noexcept { return sample_domain_; }

  /// Return the number of rank-local scalar coordinates in sample space.
  std::size_t localSampleVectorSize() const noexcept { return local_sample_vector_size_; }

  /// Return the globally distributed scalar dimension.
  std::size_t globalSampleVectorSize() const noexcept { return global_sample_vector_size_; }

  /// Return the globally normalized weighted mean local energy.
  DerivativeValue meanEnergy() const noexcept { return mean_energy_; }

  /// Return the local score-covariance energy residual, including the factor of two.
  DerivativeArrayView<const DerivativeValue> residual() const noexcept;

  /// Apply A to one complete parameter direction.
  void apply(const StructuredParameterVectorConstView& direction,
             DerivativeArrayView<DerivativeValue> result) const;

  /// Apply A^T to one rank-local realified sample vector and globally sum the result.
  void applyAdjoint(DerivativeArrayView<const DerivativeValue> direction,
                    DerivativeArrayView<DerivativeValue> result) const;

  /// Add omega times the projector onto score-centering null modes.
  void addCenterProjector(DerivativeArrayView<const DerivativeValue> direction,
                          DerivativeReal omega,
                          DerivativeArrayView<DerivativeValue> result) const;

  /// Form the globally replicated score-covariance gradient A^T residual.
  std::vector<DerivativeReal> scoreCovarianceGradient() const;

  /// Return exact retained numeric storage owned by this factor.
  ScoreCovarianceStorageDiagnostics storageDiagnostics() const noexcept;

  /// Reduce one local sample-space inner product on all ranks.
  DerivativeValue reduceSampleScalar(DerivativeValue local_value,
                                     std::exception_ptr local_failure = {}) const;

private:
  struct BatchStorage;

  void validateParameterDirection(const StructuredParameterVectorConstView& direction) const;
  void validateSampleVector(DerivativeArrayView<const DerivativeValue> values,
                            const char* description) const;

  StructuredParameterSchema schema_;
  std::size_t parameter_version_ = 0;
  ScoreSampleDomain sample_domain_ = ScoreSampleDomain::REAL;
  DistributedParameterReduction reduction_;
  std::vector<BatchStorage> batches_;
  std::size_t local_physical_sample_count_ = 0;
  std::size_t local_sample_vector_size_ = 0;
  std::size_t global_sample_vector_size_ = 0;
  DerivativeValue mean_energy_{};
  std::vector<DerivativeValue> residual_;
  mutable std::vector<DerivativeValue> projected_direction_;
  mutable std::vector<DerivativeValue> local_parameter_action_;
};

/// Checked descriptor for one distributed rank-local sample-space action.
struct DistributedSampleOperatorDescriptor
{
  std::string provider_id;
  std::string schema_fingerprint;
  std::size_t parameter_version = 0;
  std::size_t local_size = 0;
  std::size_t global_size = 0;
  bool hermitian = false;
};

/** Checked linear action whose sample vector is partitioned across MPI ranks. */
class DistributedSampleLinearOperator
{
public:
  DistributedSampleLinearOperator(const ScoreCovarianceFactor& factor, bool hermitian);
  DistributedSampleLinearOperator(const DistributedSampleLinearOperator&) = delete;
  DistributedSampleLinearOperator& operator=(const DistributedSampleLinearOperator&) = delete;
  virtual ~DistributedSampleLinearOperator() = default;

  /// Return immutable action identity and local/global extents.
  const DistributedSampleOperatorDescriptor& descriptor() const noexcept { return descriptor_; }

  /// Apply the checked action to distinct rank-local intervals.
  void apply(DerivativeArrayView<const DerivativeValue> direction,
             DerivativeArrayView<DerivativeValue> result) const;

  /// Return the global Hermitian inner product of two rank-local vectors.
  DerivativeValue hermitianDot(DerivativeArrayView<const DerivativeValue> left,
                               DerivativeArrayView<const DerivativeValue> right) const;

  /// Return the globally reduced Euclidean norm.
  DerivativeReal norm(DerivativeArrayView<const DerivativeValue> values) const;

protected:
  const ScoreCovarianceFactor& factor() const noexcept { return factor_; }

  /// Fill a validated destination without changing vector ownership.
  virtual void evaluate(DerivativeArrayView<const DerivativeValue> direction,
                        DerivativeArrayView<DerivativeValue> result) const = 0;

private:
  const ScoreCovarianceFactor& factor_;
  DistributedSampleOperatorDescriptor descriptor_;
};

/** Apply A A^T plus Tikhonov and score-centering shifts in sample space. */
class MinimumStepSampleOperator final : public DistributedSampleLinearOperator
{
public:
  MinimumStepSampleOperator(const ScoreCovarianceFactor& factor,
                            DerivativeReal damping,
                            DerivativeReal center_shift);

  DerivativeReal damping() const noexcept { return damping_; }
  DerivativeReal centerShift() const noexcept { return center_shift_; }

protected:
  void evaluate(DerivativeArrayView<const DerivativeValue> direction,
                DerivativeArrayView<DerivativeValue> result) const override;

private:
  DerivativeReal damping_ = 0.0;
  DerivativeReal center_shift_ = 0.0;
  mutable std::vector<DerivativeValue> parameter_scratch_;
};

/// Result of one bounded distributed sample-space PCG solve.
struct DistributedSampleKrylovResult
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
  std::vector<DerivativeValue> solution;
};

/** Solve a Hermitian positive-definite distributed sample-space system with PCG. */
DistributedSampleKrylovResult solveDistributedSamplePCG(
    const DistributedSampleLinearOperator& linear_operator,
    DerivativeArrayView<const DerivativeValue> right_hand_side,
    const KrylovSolverControl& control,
    DerivativeArrayView<const DerivativeValue> warm_start = {});

/// Shared numerical controls for minSR and SPRING proposals.
struct MinimumStepUpdateControl
{
  DerivativeReal learning_rate = 1.0;
  DerivativeReal damping = 1.0e-3;
  DerivativeReal center_shift = 1.0;
  /// Zero disables the Euclidean norm bound on the published parameter update.
  DerivativeReal maximum_update_norm = 0.0;
  /// Relative tolerance for matching the supplied objective to A^T residual.
  DerivativeReal objective_tolerance = 1.0e-9;
  KrylovSolverControl krylov;
};

/// Diagnostics retained from one minSR or SPRING proposal.
struct MinimumStepUpdateDiagnostics
{
  DistributedSampleKrylovResult solve;
  DerivativeReal direction_norm = 0.0;
  DerivativeReal applied_scale = 0.0;
};

/** Stateless transactional minimum-step stochastic-reconfiguration update. */
class MinimumStepSRUpdateRule final : public TrainingUpdateRule
{
public:
  MinimumStepSRUpdateRule(const ScoreCovarianceFactor& factor,
                          MinimumStepUpdateControl control = {});

  StructuredParameterSnapshot propose(
      const StructuredParameterSchema& schema,
      const StructuredParameterSnapshot& parameters,
      ParameterGradientView objective) override;

  void proposalAccepted(const StructuredParameterSchema&,
                        const StructuredParameterSnapshot&,
                        ParameterGradientView) noexcept override;

  void proposalRejected() noexcept override;

  const std::optional<MinimumStepUpdateDiagnostics>& lastDiagnostics() const noexcept
  {
    return last_diagnostics_;
  }

private:
  const ScoreCovarianceFactor& factor_;
  MinimumStepUpdateControl control_;
  std::optional<MinimumStepUpdateDiagnostics> last_diagnostics_;
  bool proposal_live_ = false;
};

/// SPRING-specific history decay and paper norm constraint.
struct SPRINGUpdateControl
{
  MinimumStepUpdateControl minimum_step;
  DerivativeReal history_decay = 0.99;
  /// Paper constant C in ||dtheta||^2 <= C; zero disables the constraint.
  DerivativeReal maximum_update_norm_squared = 0.0;
};

/** Transactional Kaczmarz-history update with exact checkpoint continuation. */
class SPRINGUpdateRule final : public TrainingUpdateRule,
                               public OptimizerCheckpointContributor
{
public:
  SPRINGUpdateRule(const ScoreCovarianceFactor& factor,
                   SPRINGUpdateControl control = {});

  /// Bind the next immutable sample factor without discarding committed history.
  void bindFactor(const ScoreCovarianceFactor& factor);

  StructuredParameterSnapshot propose(
      const StructuredParameterSchema& schema,
      const StructuredParameterSnapshot& parameters,
      ParameterGradientView objective) override;

  void proposalAccepted(const StructuredParameterSchema& schema,
                        const StructuredParameterSnapshot& parameters,
                        ParameterGradientView objective) noexcept override;

  void proposalRejected() noexcept override;

  CheckpointContributorMetadata checkpointMetadata() const override;
  void writeCheckpointPayload(hdf_archive& archive) const override;
  std::unique_ptr<PreparedRestore> prepareCheckpointRestore(
      hdf_archive& archive,
      const CheckpointContributorMetadata& saved_metadata) override;

  std::uint64_t acceptedUpdateCount() const noexcept { return accepted_update_count_; }
  const std::vector<DerivativeValue>& history() const noexcept { return history_; }
  bool hasLiveProposal() const noexcept { return proposal_live_; }
  const std::optional<MinimumStepUpdateDiagnostics>& lastDiagnostics() const noexcept
  {
    return last_diagnostics_;
  }

private:
  class PreparedRestoreAction;

  std::string stateFingerprint(std::uint64_t update_count,
                               const std::vector<DerivativeValue>& history) const;
  void validateRestoredState(std::uint64_t update_count,
                             const std::vector<DerivativeValue>& history) const;

  const ScoreCovarianceFactor* factor_ = nullptr;
  SPRINGUpdateControl control_;
  std::string configuration_fingerprint_;
  std::vector<DerivativeValue> history_;
  std::vector<DerivativeValue> pending_history_;
  std::optional<MinimumStepUpdateDiagnostics> last_diagnostics_;
  std::uint64_t accepted_update_count_ = 0;
  bool proposal_live_ = false;
};

} // namespace qmcplusplus::wftrain

#endif
