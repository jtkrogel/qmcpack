//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file MatrixFreeLinearMethod.h
 * @brief Bounded real-Hermitian generalized eigensolves for the linear method.
 *
 * The solver consumes abstract Hamiltonian and overlap actions.  A future sampling
 * backend may expose the usual augmented [Psi, dPsi/dp] space through the operator
 * contract below.  The reference amplitude is deliberately separate from the model's
 * tangent schema; this numerical layer neither stores sample derivatives nor publishes
 * a wave-function update.
 */

#ifndef QMCPLUSPLUS_MATRIX_FREE_LINEAR_METHOD_H
#define QMCPLUSPLUS_MATRIX_FREE_LINEAR_METHOD_H

#include "QMCDrivers/WFTrain/MatrixFreeOperator.h"

#include <cstddef>
#include <string>
#include <vector>

namespace qmcplusplus::wftrain
{

/// Immutable identity of an augmented reference-plus-tangent action.
struct AugmentedLinearMethodOperatorDescriptor
{
  std::string provider_id;
  std::string schema_fingerprint;
  std::size_t parameter_version = 0;
  std::size_t tangent_count = 0;
  ReductionDomain reduction_domain = ReductionDomain::CROWD_LOCAL;
  bool hermitian = false;
};

/** Checked action over [reference scalar, P structured tangent coordinates].
 *
 * The reference scalar is not part of the copied tangent schema and cannot enter a
 * parameter optimizer or KFAC block accidentally.  Derived providers implement only
 * the numerical action after identity, extent, alias, and finite-value checks.
 */
class AugmentedLinearMethodOperator
{
public:
  AugmentedLinearMethodOperator(const StructuredParameterSchema& tangent_schema,
                                std::size_t parameter_version,
                                ReductionDomain reduction_domain,
                                bool hermitian);
  AugmentedLinearMethodOperator(const AugmentedLinearMethodOperator&) = delete;
  AugmentedLinearMethodOperator& operator=(const AugmentedLinearMethodOperator&) = delete;
  virtual ~AugmentedLinearMethodOperator() = default;

  /// Return the exact model-parameter schema underlying the tangent coordinates.
  const StructuredParameterSchema& tangentSchema() const noexcept { return tangent_schema_; }

  /// Return immutable identity and algebraic promises for this action.
  const AugmentedLinearMethodOperatorDescriptor& descriptor() const noexcept
  {
    return descriptor_;
  }

  /// Apply the action to distinct intervals of exactly P+1 real-valued entries.
  void apply(DerivativeArrayView<const DerivativeValue> direction,
             DerivativeArrayView<DerivativeValue> result) const;

protected:
  /// Fill one checked P+1 destination without retaining the caller's intervals.
  virtual void evaluate(DerivativeArrayView<const DerivativeValue> direction,
                        DerivativeArrayView<DerivativeValue> result) const = 0;

private:
  StructuredParameterSchema tangent_schema_;
  AugmentedLinearMethodOperatorDescriptor descriptor_;
};

/// Identify why a bounded generalized Davidson solve stopped.
enum class LinearMethodStopReason
{
  CONVERGED,
  ITERATION_LIMIT,
  OPERATOR_APPLICATION_LIMIT,
  TIME_LIMIT,
  LINEAR_DEPENDENCE,
  NON_POSITIVE_OVERLAP,
  INVALID_REFERENCE_OVERLAP,
  NON_HERMITIAN_ACTION,
  NONFINITE_RESULT,
  PROJECTED_SOLVE_FAILURE,
  UNTRUSTED_ROOT,
  BREAKDOWN
};

/// Bound subspace work and define numerical acceptance thresholds.
struct MatrixFreeLinearMethodControl
{
  DerivativeReal relative_residual_tolerance = 1.0e-8;
  DerivativeReal absolute_residual_tolerance = 0.0;
  std::size_t maximum_iterations = 100;
  std::size_t maximum_operator_applications = 202;
  double maximum_seconds = 0.0;
  std::size_t maximum_subspace_dimension = 12;
  DerivativeReal minimum_overlap_norm = 1.0e-12;
  DerivativeReal projected_eigensolver_tolerance = 1.0e-12;
  std::size_t maximum_projected_sweeps = 100;
  DerivativeReal hermitian_tolerance = 256.0;
  /// Minimum |c0|/sqrt(c^T S c) accepted for a converged Ritz vector.
  DerivativeReal minimum_reference_overlap = 1.0e-6;
};

/// Complete nonpublishing result and bounded-storage diagnostics from one solve.
struct MatrixFreeLinearMethodResult
{
  LinearMethodStopReason stop_reason = LinearMethodStopReason::BREAKDOWN;
  bool converged = false;
  std::size_t iterations = 0;
  std::size_t operator_applications = 0;
  std::size_t restart_count = 0;
  std::size_t peak_subspace_dimension = 0;
  std::size_t peak_parameter_vectors = 0;
  DerivativeReal eigenvalue = 0.0;
  DerivativeReal residual_norm = 0.0;
  DerivativeReal relative_residual_norm = 0.0;
  double elapsed_seconds = 0.0;
  std::string provider_id;
  std::string schema_fingerprint;
  std::size_t parameter_version = 0;
  std::vector<DerivativeValue> eigenvector;
};

/** Find the lowest generalized Ritz pair H*x=lambda*S*x.
 *
 * Both symmetrized actions must be global, Hermitian, REAL64, and bound to the same
 * tangent schema/version.  The intended sampled matrices are
 * S=D^H D and H_sym=(D^H L+L^H D)/2.  Storage is O(mP + m^2), where m is
 * maximum_subspace_dimension.  The returned vector is diagnostic until the caller
 * explicitly transforms and publishes it.
 */
MatrixFreeLinearMethodResult solveSymmetrizedMatrixFreeLinearMethod(
    const AugmentedLinearMethodOperator& hamiltonian,
    const AugmentedLinearMethodOperator& overlap,
    const MatrixFreePreconditioner& preconditioner,
    DerivativeArrayView<const DerivativeValue> initial_vector,
    const MatrixFreeLinearMethodControl& control = {});

/// Control reference normalization and optional tangent norm limiting.
struct LinearMethodUpdateControl
{
  DerivativeReal minimum_reference_magnitude = 1.0e-10;
  DerivativeReal maximum_update_norm = 0.0;
};

/// Candidate tangent formed from a converged augmented LM eigenvector.
struct LinearMethodUpdateCandidate
{
  std::vector<DerivativeValue> direction;
  DerivativeReal unscaled_norm = 0.0;
  DerivativeReal applied_scale = 1.0;
};

/** Convert a converged [reference, tangent] eigenvector into an update candidate.
 *
 * Entry zero is the separate reference coefficient.  The returned vector has exactly
 * P tangent entries, with frozen schema blocks zeroed.  This function has no
 * wave-function handle and therefore cannot publish or partially apply an update.
 */
LinearMethodUpdateCandidate formLinearMethodUpdateCandidate(
    const StructuredParameterSchema& tangent_schema,
    const MatrixFreeLinearMethodResult& result,
    const LinearMethodUpdateControl& control = {});

/// Return the stable diagnostic name of one linear-method stop reason.
const char* linearMethodStopReasonName(LinearMethodStopReason reason) noexcept;

} // namespace qmcplusplus::wftrain

#endif
