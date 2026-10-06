//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file MatrixFreeStochasticReconfiguration.h
 * @brief Bounded score-covariance actions and transactional SR parameter updates.
 *
 * The implementation composes public score JVP and Hermitian VJP streams.  It never
 * materializes a sample-by-parameter score table or a parameter-squared covariance.
 */

#ifndef QMCPLUSPLUS_MATRIX_FREE_STOCHASTIC_RECONFIGURATION_H
#define QMCPLUSPLUS_MATRIX_FREE_STOCHASTIC_RECONFIGURATION_H

#include "QMCDrivers/WFTrain/DistributedParameterReduction.h"
#include "QMCDrivers/WFTrain/KrylovSolver.h"
#include "QMCDrivers/WFTrain/ParameterUpdateTransaction.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace qmcplusplus::wftrain
{

/** Bind one local sample batch to nonnegative weights copied by the SR operator.
 *
 * The derivative operator is nonowning and must outlive the score-covariance action.
 */
struct StochasticReconfigurationBatch
{
  const StreamingDerivativeOperator* derivative_operator = nullptr;
  DerivativeArrayView<const DerivativeReal> weights;
};

/// Exact numeric storage retained by one prepared score-covariance action.
struct StochasticReconfigurationStorageDiagnostics
{
  std::size_t parameter_count = 0;
  std::size_t local_sample_count = 0;
  std::size_t real_sample_values = 0;
  std::size_t complex_sample_values = 0;
  std::size_t complex_parameter_vectors = 0;
  std::size_t retained_numeric_bytes = 0;
};

/** Apply a globally centered weighted score covariance with O(P)+O(B) storage.
 *
 * This object owns mutable reusable scratch and is intentionally not concurrent-call
 * safe. Distinct crowds/ranks should prepare distinct operators over their local
 * batches; every successful action is globally replicated.
 */
class StochasticReconfigurationOperator final : public MatrixFreeLinearOperator
{
public:
  StochasticReconfigurationOperator(
      const StructuredParameterSchema& schema,
      std::size_t parameter_version,
      DerivativeArrayView<const StochasticReconfigurationBatch> batches,
      DistributedParameterReduction reduction = {});
  ~StochasticReconfigurationOperator() override;

  StochasticReconfigurationOperator(const StochasticReconfigurationOperator&) = delete;
  StochasticReconfigurationOperator& operator=(const StochasticReconfigurationOperator&) = delete;

  /// Return exact retained numeric storage, excluding the referenced derivative owners.
  StochasticReconfigurationStorageDiagnostics storageDiagnostics() const noexcept;

protected:
  void evaluate(const StructuredParameterVectorConstView& direction,
                DerivativeArrayView<DerivativeValue> result) const override;

private:
  struct BatchStorage;

  void validateAndProjectDirection(const StructuredParameterVectorConstView& direction) const;

  std::vector<BatchStorage> batches_;
  DistributedParameterReduction reduction_;
  ParameterScalarDomain scalar_domain_ = ParameterScalarDomain::REAL64;
  mutable std::vector<DerivativeValue> projected_direction_;
  mutable std::vector<DerivativeValue> local_action_;
};

/// Configure one bounded damped-PCG SR proposal.
struct StochasticReconfigurationUpdateControl
{
  DerivativeReal learning_rate = 1.0;
  DerivativeReal initial_damping = 1.0e-3;
  DerivativeReal damping_multiplier = 10.0;
  /// Positive shift used for the first retry when initial_damping is zero.
  DerivativeReal minimum_retry_damping = 1.0e-8;
  std::size_t maximum_damping_attempts = 3;
  /// Zero disables the Euclidean parameter-step bound.
  DerivativeReal maximum_update_norm = 0.0;
  /// Zero disables the covariance-metric parameter-step bound.
  DerivativeReal maximum_metric_norm = 0.0;
  KrylovSolverControl krylov;
};

/// Diagnostics retained from the most recent successful or failed SR proposal.
struct StochasticReconfigurationUpdateDiagnostics
{
  std::size_t damping_attempts = 0;
  DerivativeReal damping = 0.0;
  DerivativeReal direction_norm = 0.0;
  DerivativeReal metric_norm = 0.0;
  DerivativeReal applied_scale = 1.0;
  KrylovSolveResult solve;
};

/** Convert an objective gradient into a failure-atomic damped natural-gradient step.
 *
 * The rule is objective neutral and plugs directly into completeParameterUpdate().
 * It currently publishes REAL64 StructuredParameterSnapshot values; the covariance
 * operator itself also supports COMPLEX128 tangent algebra for later complex drivers.
 */
class StochasticReconfigurationUpdateRule final : public TrainingUpdateRule
{
public:
  StochasticReconfigurationUpdateRule(
      const MatrixFreeLinearOperator& covariance,
      const MatrixFreePreconditioner& preconditioner,
      StochasticReconfigurationUpdateControl control = {});

  StructuredParameterSnapshot propose(
      const StructuredParameterSchema& schema,
      const StructuredParameterSnapshot& parameters,
      ParameterGradientView objective) override;

  void proposalAccepted(const StructuredParameterSchema& schema,
                        const StructuredParameterSnapshot& parameters,
                        ParameterGradientView objective) noexcept override;

  void proposalRejected() noexcept override;

  /// Report whether one speculative candidate is awaiting publication outcome.
  bool hasLiveProposal() const noexcept { return proposal_live_; }

  /// Return diagnostics when at least one solve attempt has occurred.
  const std::optional<StochasticReconfigurationUpdateDiagnostics>&
  lastDiagnostics() const noexcept
  {
    return last_diagnostics_;
  }

private:
  void validateInputs(const StructuredParameterSchema& schema,
                      const StructuredParameterSnapshot& parameters,
                      ParameterGradientView objective) const;

  const MatrixFreeLinearOperator& covariance_;
  const MatrixFreePreconditioner& preconditioner_;
  StochasticReconfigurationUpdateControl control_;
  std::optional<StochasticReconfigurationUpdateDiagnostics> last_diagnostics_;
  bool proposal_live_ = false;
};

} // namespace qmcplusplus::wftrain

#endif
