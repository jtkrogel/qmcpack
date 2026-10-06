//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file MatrixFreeOperator.cpp
 * @brief Validation and bounded vector algebra for matrix-free training operators.
 */

#include "QMCDrivers/WFTrain/MatrixFreeOperator.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::wftrain
{
namespace
{

/// Return the one scalar domain shared by every block, or reject a mixed schema.
ParameterScalarDomain homogeneousScalarDomain(const StructuredParameterSchema& schema)
{
  const ParameterScalarDomain domain = schema.blocks().front().scalar_domain;
  for (const ParameterBlockDescriptor& block : schema.blocks())
    if (block.scalar_domain != domain)
      throw std::invalid_argument("Matrix-free operators require one homogeneous parameter scalar domain");
  return domain;
}

/// Report whether both components of one contraction scalar are finite.
bool isFinite(DerivativeValue value) noexcept
{
  return isFiniteTrainingReal(value.real()) && isFiniteTrainingReal(value.imag());
}

/// Validate a one-dimensional parameter interval against one schema.
template<class T>
void validateExtent(const StructuredParameterSchema& schema,
                    DerivativeArrayView<T> values,
                    const char* description)
{
  if (values.size() != schema.parameterCount() || (!values.empty() && values.data() == nullptr))
    throw std::invalid_argument(std::string(description) + " extent does not match the parameter schema");
}

/// Reject non-finite values and imaginary tangents for a real parameter schema.
template<class T>
void validateValues(const StructuredParameterSchema& schema,
                    DerivativeArrayView<T> values,
                    const char* description,
                    bool numerical_output)
{
  const ParameterScalarDomain domain = homogeneousScalarDomain(schema);
  for (const DerivativeValue value : values)
  {
    if (!isFinite(value))
    {
      if (numerical_output)
        throw MatrixFreeNumericalError(std::string(description) + " contains a non-finite value");
      throw std::invalid_argument(std::string(description) + " contains a non-finite value");
    }
    if (domain == ParameterScalarDomain::REAL64 && value.imag() != DerivativeReal{})
    {
      if (numerical_output)
        throw MatrixFreeNumericalError(std::string(description) +
                                       " has an imaginary component for real parameters");
      throw std::invalid_argument(std::string(description) +
                                  " has an imaginary component for real parameters");
    }
  }
}

/// Reject overlapping input/output intervals without comparing unrelated pointers.
bool intervalsOverlap(const DerivativeValue* left,
                      std::size_t left_size,
                      const DerivativeValue* right,
                      std::size_t right_size) noexcept
{
  if (left_size == 0 || right_size == 0)
    return false;
  const std::uintptr_t left_begin  = reinterpret_cast<std::uintptr_t>(left);
  const std::uintptr_t right_begin = reinterpret_cast<std::uintptr_t>(right);
  const std::size_t left_bytes     = left_size * sizeof(DerivativeValue);
  const std::size_t right_bytes    = right_size * sizeof(DerivativeValue);
  return left_begin < right_begin + right_bytes && right_begin < left_begin + left_bytes;
}

/// Require a direction to match an exact operator/preconditioner identity.
void validateIdentity(const StructuredParameterSchema& schema,
                      std::size_t parameter_version,
                      const StructuredParameterVectorConstView& direction,
                      const char* description)
{
  if (direction.providerId() != schema.providerId() ||
      direction.schemaFingerprint() != schema.fingerprint() ||
      direction.parameterVersion() != parameter_version)
    throw std::invalid_argument(std::string(description) + " identity or parameter version does not match");
}

/// Require a finite scalar compatible with the parameter domain.
void validateFactor(const StructuredParameterSchema& schema,
                    DerivativeValue factor,
                    const char* description)
{
  if (!isFinite(factor))
    throw std::invalid_argument(std::string(description) + " is non-finite");
  if (homogeneousScalarDomain(schema) == ParameterScalarDomain::REAL64 &&
      factor.imag() != DerivativeReal{})
    throw std::invalid_argument(std::string(description) +
                                " is complex for a real parameter schema");
}

} // namespace

MatrixFreeLinearOperator::MatrixFreeLinearOperator(
    const StructuredParameterSchema& schema,
    std::size_t parameter_version,
    ReductionDomain reduction_domain,
    bool hermitian)
    : schema_(schema),
      descriptor_{schema_.providerId(), schema_.fingerprint(), parameter_version,
                  schema_.parameterCount(), homogeneousScalarDomain(schema_),
                  reduction_domain, hermitian}
{}

void MatrixFreeLinearOperator::apply(
    const StructuredParameterVectorConstView& direction,
    DerivativeArrayView<DerivativeValue> result) const
{
  validateIdentity(schema_, descriptor_.parameter_version, direction,
                   "Matrix-free direction");
  // The view is nonowning, so its backing storage may have changed since view
  // construction. Revalidate at the action boundary before derived code can mask an
  // invalid tangent (for example, a zero action hiding a NaN input).
  validateValues(schema_, direction.values(), "Matrix-free direction", false);
  validateExtent(schema_, result, "Matrix-free result");
  if (intervalsOverlap(direction.values().data(), direction.values().size(),
                       result.data(), result.size()))
    throw std::invalid_argument("Matrix-free input and output intervals must not overlap");

  // Poison every destination entry so a derived action that omits one element cannot
  // accidentally publish a stale value from an earlier Krylov iteration.
  const DerivativeReal sentinel = std::numeric_limits<DerivativeReal>::quiet_NaN();
  std::fill(result.begin(), result.end(), DerivativeValue{sentinel, sentinel});
  evaluate(direction, result);
  validateValues(schema_,
                 DerivativeArrayView<const DerivativeValue>{result.data(), result.size()},
                 "Matrix-free result", true);
}

ShiftedMatrixFreeOperator::ShiftedMatrixFreeOperator(
    const MatrixFreeLinearOperator& operand,
    DerivativeReal shift)
    : MatrixFreeLinearOperator(operand.parameterSchema(),
                               operand.descriptor().parameter_version,
                               operand.descriptor().reduction_domain,
                               operand.descriptor().hermitian),
      operand_(operand), shift_(shift)
{
  if (!isFiniteTrainingReal(shift_) || shift_ < 0.0)
    throw std::invalid_argument("Matrix-free identity shift must be finite and nonnegative");
}

void ShiftedMatrixFreeOperator::evaluate(
    const StructuredParameterVectorConstView& direction,
    DerivativeArrayView<DerivativeValue> result) const
{
  operand_.apply(direction, result);
  for (std::size_t index = 0; index < result.size(); ++index)
    result[index] += shift_ * direction.values()[index];
}

MatrixFreePreconditioner::MatrixFreePreconditioner(
    const StructuredParameterSchema& schema,
    std::size_t parameter_version)
    : schema_(schema), parameter_version_(parameter_version)
{
  homogeneousScalarDomain(schema_);
}

void MatrixFreePreconditioner::apply(
    const StructuredParameterVectorConstView& residual,
    DerivativeArrayView<DerivativeValue> result) const
{
  validateIdentity(schema_, parameter_version_, residual,
                   "Matrix-free preconditioner residual");
  validateValues(schema_, residual.values(),
                 "Matrix-free preconditioner residual", false);
  validateExtent(schema_, result, "Matrix-free preconditioner result");
  if (intervalsOverlap(residual.values().data(), residual.values().size(),
                       result.data(), result.size()))
    throw std::invalid_argument("Matrix-free preconditioner input and output must not overlap");

  const DerivativeReal sentinel = std::numeric_limits<DerivativeReal>::quiet_NaN();
  std::fill(result.begin(), result.end(), DerivativeValue{sentinel, sentinel});
  evaluate(residual, result);
  validateValues(schema_,
                 DerivativeArrayView<const DerivativeValue>{result.data(), result.size()},
                 "Matrix-free preconditioner result", true);
}

void IdentityPreconditioner::evaluate(
    const StructuredParameterVectorConstView& residual,
    DerivativeArrayView<DerivativeValue> result) const
{
  std::copy(residual.values().begin(), residual.values().end(), result.begin());
}

DiagonalPreconditioner::DiagonalPreconditioner(
    const StructuredParameterSchema& schema,
    std::size_t parameter_version,
    DerivativeArrayView<const DerivativeReal> diagonal,
    DerivativeReal floor)
    : MatrixFreePreconditioner(schema, parameter_version),
      inverse_diagonal_(schema.parameterCount())
{
  if (diagonal.size() != schema.parameterCount() ||
      (!diagonal.empty() && diagonal.data() == nullptr))
    throw std::invalid_argument("Diagonal preconditioner extent does not match the parameter schema");
  if (!isFiniteTrainingReal(floor) || floor <= 0.0)
    throw std::invalid_argument("Diagonal preconditioner floor must be finite and positive");

  for (std::size_t index = 0; index < diagonal.size(); ++index)
  {
    if (!isFiniteTrainingReal(diagonal[index]) || diagonal[index] < 0.0)
      throw std::invalid_argument("Diagonal preconditioner contains a negative or non-finite value");
    inverse_diagonal_[index] = DerivativeReal{1} / std::max(diagonal[index], floor);
    if (!isFiniteTrainingReal(inverse_diagonal_[index]))
      throw std::invalid_argument("Diagonal preconditioner inverse is non-finite");
  }
}

void DiagonalPreconditioner::evaluate(
    const StructuredParameterVectorConstView& residual,
    DerivativeArrayView<DerivativeValue> result) const
{
  for (std::size_t index = 0; index < result.size(); ++index)
    result[index] = inverse_diagonal_[index] * residual.values()[index];
}

void copyParameterVector(const StructuredParameterSchema& schema,
                         DerivativeArrayView<const DerivativeValue> source,
                         DerivativeArrayView<DerivativeValue> destination)
{
  validateExtent(schema, source, "Parameter-vector source");
  validateExtent(schema, destination, "Parameter-vector destination");
  validateValues(schema, source, "Parameter-vector source", false);
  if (intervalsOverlap(source.data(), source.size(), destination.data(),
                       destination.size()) && source.data() != destination.data())
    throw std::invalid_argument("Partially overlapping parameter-vector copies are not supported");
  std::copy(source.begin(), source.end(), destination.begin());
}

void scaleParameterVector(const StructuredParameterSchema& schema,
                          DerivativeValue factor,
                          DerivativeArrayView<DerivativeValue> values)
{
  validateExtent(schema, values, "Scaled parameter vector");
  validateValues(schema,
                 DerivativeArrayView<const DerivativeValue>{values.data(), values.size()},
                 "Scaled parameter vector", false);
  validateFactor(schema, factor, "Parameter-vector scale factor");
  for (DerivativeValue& value : values)
    value *= factor;
  validateValues(schema,
                 DerivativeArrayView<const DerivativeValue>{values.data(), values.size()},
                 "Scaled parameter vector", true);
}

void axpyParameterVector(const StructuredParameterSchema& schema,
                         DerivativeValue alpha,
                         DerivativeArrayView<const DerivativeValue> x,
                         DerivativeArrayView<DerivativeValue> y)
{
  validateExtent(schema, x, "AXPY source");
  validateExtent(schema, y, "AXPY destination");
  if (intervalsOverlap(x.data(), x.size(), y.data(), y.size()) &&
      x.data() != y.data())
    throw std::invalid_argument("Partially overlapping AXPY vectors are not supported");
  validateValues(schema, x, "AXPY source", false);
  validateValues(schema,
                 DerivativeArrayView<const DerivativeValue>{y.data(), y.size()},
                 "AXPY destination", false);
  validateFactor(schema, alpha, "AXPY factor");
  for (std::size_t index = 0; index < y.size(); ++index)
    y[index] += alpha * x[index];
  validateValues(schema,
                 DerivativeArrayView<const DerivativeValue>{y.data(), y.size()},
                 "AXPY result", true);
}

DerivativeValue parameterVectorHermitianDot(
    const StructuredParameterSchema& schema,
    DerivativeArrayView<const DerivativeValue> x,
    DerivativeArrayView<const DerivativeValue> y)
{
  validateExtent(schema, x, "Hermitian-dot left vector");
  validateExtent(schema, y, "Hermitian-dot right vector");
  validateValues(schema, x, "Hermitian-dot left vector", false);
  validateValues(schema, y, "Hermitian-dot right vector", false);

  DerivativeValue result{};
  for (std::size_t index = 0; index < x.size(); ++index)
    result += std::conj(x[index]) * y[index];
  if (!isFinite(result))
    throw MatrixFreeNumericalError("Hermitian parameter-vector dot product is non-finite");
  return result;
}

DerivativeReal parameterVectorNorm(
    const StructuredParameterSchema& schema,
    DerivativeArrayView<const DerivativeValue> values)
{
  const DerivativeValue squared = parameterVectorHermitianDot(schema, values, values);
  const DerivativeReal tolerance =
      std::numeric_limits<DerivativeReal>::epsilon() *
      std::max(DerivativeReal{1}, std::abs(squared.real()));
  if (std::abs(squared.imag()) > tolerance || squared.real() < -tolerance)
    throw MatrixFreeNumericalError("Hermitian parameter-vector norm is not real and nonnegative");
  return std::sqrt(std::max(DerivativeReal{}, squared.real()));
}

} // namespace qmcplusplus::wftrain
