//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file MatrixFreeOperator.h
 * @brief Checked O(P) linear-operator and preconditioner contracts for training.
 *
 * These interfaces operate on one schema/version-bound parameter vector at a time.
 * They cannot represent a dense parameter matrix or a sample-by-parameter table.
 */

#ifndef QMCPLUSPLUS_MATRIX_FREE_OPERATOR_H
#define QMCPLUSPLUS_MATRIX_FREE_OPERATOR_H

#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace qmcplusplus::wftrain
{

/// Report a non-finite value produced by an otherwise valid matrix-free action.
class MatrixFreeNumericalError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

/// Describe the immutable identity and algebraic promises of one linear action.
struct MatrixFreeOperatorDescriptor
{
  std::string provider_id;
  std::string schema_fingerprint;
  std::size_t parameter_version = 0;
  std::size_t parameter_count   = 0;
  ParameterScalarDomain scalar_domain = ParameterScalarDomain::REAL64;
  ReductionDomain reduction_domain = ReductionDomain::CROWD_LOCAL;
  bool hermitian = false;
};

/** Checked matrix-free action over one canonical parameter vector.
 *
 * The public wrapper validates identity, extent, aliasing, and numeric output. On an
 * action failure the caller-owned result is deliberately unusable and must be discarded.
 * Derived classes implement only the action.
 */
class MatrixFreeLinearOperator
{
public:
  MatrixFreeLinearOperator(const StructuredParameterSchema& schema,
                           std::size_t parameter_version,
                           ReductionDomain reduction_domain,
                           bool hermitian);
  MatrixFreeLinearOperator(const MatrixFreeLinearOperator&) = delete;
  MatrixFreeLinearOperator& operator=(const MatrixFreeLinearOperator&) = delete;
  virtual ~MatrixFreeLinearOperator() = default;

  /// Return the exact schema whose tangent space is acted on.
  const StructuredParameterSchema& parameterSchema() const noexcept { return schema_; }

  /// Return immutable action metadata.
  const MatrixFreeOperatorDescriptor& descriptor() const noexcept { return descriptor_; }

  /// Apply the operator after checking the complete input/output contract.
  void apply(const StructuredParameterVectorConstView& direction,
             DerivativeArrayView<DerivativeValue> result) const;

protected:
  /// Fill a distinct, correctly sized result after common preflight succeeds.
  virtual void evaluate(const StructuredParameterVectorConstView& direction,
                        DerivativeArrayView<DerivativeValue> result) const = 0;

private:
  StructuredParameterSchema schema_;
  MatrixFreeOperatorDescriptor descriptor_;
};

/** Add a nonnegative identity shift to an existing Hermitian global action.
 *
 * The adapter owns no parameter-sized storage and delegates all workspace ownership to
 * the wrapped operator and caller.
 */
class ShiftedMatrixFreeOperator final : public MatrixFreeLinearOperator
{
public:
  /** Wrap an operator whose lifetime must exceed this nonowning adapter. */
  ShiftedMatrixFreeOperator(const MatrixFreeLinearOperator& operand, DerivativeReal shift);
  ShiftedMatrixFreeOperator(MatrixFreeLinearOperator&&, DerivativeReal) = delete;

  /// Return the fixed identity shift.
  DerivativeReal shift() const noexcept { return shift_; }

protected:
  void evaluate(const StructuredParameterVectorConstView& direction,
                DerivativeArrayView<DerivativeValue> result) const override;

private:
  const MatrixFreeLinearOperator& operand_;
  DerivativeReal shift_ = 0.0;
};

/** Checked inverse-preconditioner action bound to one schema and version. */
class MatrixFreePreconditioner
{
public:
  MatrixFreePreconditioner(const StructuredParameterSchema& schema,
                           std::size_t parameter_version);
  MatrixFreePreconditioner(const MatrixFreePreconditioner&) = delete;
  MatrixFreePreconditioner& operator=(const MatrixFreePreconditioner&) = delete;
  virtual ~MatrixFreePreconditioner() = default;

  /// Return the exact schema accepted by this preconditioner.
  const StructuredParameterSchema& parameterSchema() const noexcept { return schema_; }

  /// Return the bound parameter version.
  std::size_t parameterVersion() const noexcept { return parameter_version_; }

  /// Apply the inverse preconditioner to a distinct caller-owned destination.
  void apply(const StructuredParameterVectorConstView& residual,
             DerivativeArrayView<DerivativeValue> result) const;

protected:
  /// Fill the preconditioned residual after common validation succeeds.
  virtual void evaluate(const StructuredParameterVectorConstView& residual,
                        DerivativeArrayView<DerivativeValue> result) const = 0;

private:
  StructuredParameterSchema schema_;
  std::size_t parameter_version_ = 0;
};

/// Copy a residual unchanged while preserving the common checked interface.
class IdentityPreconditioner final : public MatrixFreePreconditioner
{
public:
  using MatrixFreePreconditioner::MatrixFreePreconditioner;

protected:
  void evaluate(const StructuredParameterVectorConstView& residual,
                DerivativeArrayView<DerivativeValue> result) const override;
};

/** Apply the inverse of a positive real diagonal with a configurable floor. */
class DiagonalPreconditioner final : public MatrixFreePreconditioner
{
public:
  DiagonalPreconditioner(const StructuredParameterSchema& schema,
                         std::size_t parameter_version,
                         DerivativeArrayView<const DerivativeReal> diagonal,
                         DerivativeReal floor);

  /// Return the retained inverse diagonal for memory accounting and diagnostics.
  DerivativeArrayView<const DerivativeReal> inverseDiagonal() const noexcept
  {
    return {inverse_diagonal_.data(), inverse_diagonal_.size()};
  }

protected:
  void evaluate(const StructuredParameterVectorConstView& residual,
                DerivativeArrayView<DerivativeValue> result) const override;

private:
  std::vector<DerivativeReal> inverse_diagonal_;
};

/// Validate and copy one complete parameter vector.
void copyParameterVector(const StructuredParameterSchema& schema,
                         DerivativeArrayView<const DerivativeValue> source,
                         DerivativeArrayView<DerivativeValue> destination);

/// Scale one parameter vector in place by a finite scalar.
void scaleParameterVector(const StructuredParameterSchema& schema,
                          DerivativeValue factor,
                          DerivativeArrayView<DerivativeValue> values);

/// Form y <- y + alpha*x without temporary storage.
void axpyParameterVector(const StructuredParameterSchema& schema,
                         DerivativeValue alpha,
                         DerivativeArrayView<const DerivativeValue> x,
                         DerivativeArrayView<DerivativeValue> y);

/// Return the Hermitian inner product x^H y after domain and finite checks.
DerivativeValue parameterVectorHermitianDot(
    const StructuredParameterSchema& schema,
    DerivativeArrayView<const DerivativeValue> x,
    DerivativeArrayView<const DerivativeValue> y);

/// Return the Euclidean norm induced by the Hermitian inner product.
DerivativeReal parameterVectorNorm(const StructuredParameterSchema& schema,
                                   DerivativeArrayView<const DerivativeValue> values);

} // namespace qmcplusplus::wftrain

#endif
