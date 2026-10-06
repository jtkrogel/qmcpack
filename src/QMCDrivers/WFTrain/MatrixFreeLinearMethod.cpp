//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file MatrixFreeLinearMethod.cpp
 * @brief Bounded generalized Rayleigh--Ritz/Davidson implementation.
 */

#include "QMCDrivers/WFTrain/MatrixFreeLinearMethod.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace qmcplusplus::wftrain
{
namespace
{

using Clock = std::chrono::steady_clock;

/// Store one overlap-orthonormal basis and its two operator images.
struct DavidsonBasis
{
  std::vector<std::vector<DerivativeValue>> vectors;
  std::vector<std::vector<DerivativeValue>> hamiltonian_images;
  std::vector<std::vector<DerivativeValue>> overlap_images;
};

/// Return a read-only interval over one owning derivative vector.
DerivativeArrayView<const DerivativeValue> constView(
    const std::vector<DerivativeValue>& values) noexcept
{
  return {values.data(), values.size()};
}

/// Return a mutable interval over one owning derivative vector.
DerivativeArrayView<DerivativeValue> mutableView(
    std::vector<DerivativeValue>& values) noexcept
{
  return {values.data(), values.size()};
}

/// Return elapsed monotonic wall time.
double elapsedSeconds(Clock::time_point start) noexcept
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// Report whether a full complex derivative scalar is finite.
bool isFinite(DerivativeValue value) noexcept
{
  return isFiniteTrainingReal(value.real()) && isFiniteTrainingReal(value.imag());
}

/// Reject invalid extents, complex real-domain entries, and non-finite values.
template<class T>
void validateAugmentedValues(DerivativeArrayView<T> values,
                             std::size_t expected_size,
                             const char* description,
                             bool numerical_output)
{
  if (values.size() != expected_size || (!values.empty() && values.data() == nullptr))
    throw std::invalid_argument(std::string(description) + " extent is not P+1");
  for (DerivativeValue value : values)
  {
    if (!isFinite(value) || value.imag() != 0.0)
    {
      if (numerical_output)
        throw MatrixFreeNumericalError(std::string(description) +
                                       " is non-finite or non-real");
      throw std::invalid_argument(std::string(description) +
                                  " is non-finite or non-real");
    }
  }
}

/// Reject overlapping action intervals without ordering unrelated pointers.
bool intervalsOverlap(const DerivativeValue* left,
                      std::size_t left_size,
                      const DerivativeValue* right,
                      std::size_t right_size) noexcept
{
  if (left_size == 0 || right_size == 0)
    return false;
  const std::uintptr_t left_begin = reinterpret_cast<std::uintptr_t>(left);
  const std::uintptr_t right_begin = reinterpret_cast<std::uintptr_t>(right);
  const std::size_t left_bytes = left_size * sizeof(DerivativeValue);
  const std::size_t right_bytes = right_size * sizeof(DerivativeValue);
  return left_begin < right_begin + right_bytes &&
      right_begin < left_begin + left_bytes;
}

/// Return the Hermitian dot product of two augmented real vectors.
DerivativeValue augmentedDot(DerivativeArrayView<const DerivativeValue> left,
                             DerivativeArrayView<const DerivativeValue> right)
{
  DerivativeValue result{};
  for (std::size_t index = 0; index < left.size(); ++index)
    result += std::conj(left[index]) * right[index];
  return result;
}

/// Return the Euclidean norm of one augmented vector.
DerivativeReal augmentedNorm(DerivativeArrayView<const DerivativeValue> values)
{
  const DerivativeReal squared = augmentedDot(values, values).real();
  if (!isFiniteTrainingReal(squared) || squared < 0.0)
    throw MatrixFreeNumericalError("Augmented vector norm is non-finite");
  return std::sqrt(squared);
}

/// Form y <- y + alpha*x for equally sized augmented vectors.
void augmentedAxpy(DerivativeValue alpha,
                   DerivativeArrayView<const DerivativeValue> x,
                   DerivativeArrayView<DerivativeValue> y)
{
  for (std::size_t index = 0; index < y.size(); ++index)
    y[index] += alpha * x[index];
}

/// Scale one augmented vector in place.
void augmentedScale(DerivativeValue factor,
                    DerivativeArrayView<DerivativeValue> values)
{
  for (DerivativeValue& value : values)
    value *= factor;
}

/// Validate controls before any parameter-sized allocation occurs.
void validateControl(const MatrixFreeLinearMethodControl& control)
{
  if (!isFiniteTrainingReal(control.relative_residual_tolerance) ||
      control.relative_residual_tolerance < 0.0 ||
      !isFiniteTrainingReal(control.absolute_residual_tolerance) ||
      control.absolute_residual_tolerance < 0.0 ||
      !isFiniteTrainingReal(control.maximum_seconds) || control.maximum_seconds < 0.0 ||
      !isFiniteTrainingReal(control.minimum_overlap_norm) ||
      control.minimum_overlap_norm <= 0.0 ||
      !isFiniteTrainingReal(control.projected_eigensolver_tolerance) ||
      control.projected_eigensolver_tolerance <= 0.0 ||
      !isFiniteTrainingReal(control.hermitian_tolerance) ||
      control.hermitian_tolerance <= 0.0 ||
      !isFiniteTrainingReal(control.minimum_reference_overlap) ||
      control.minimum_reference_overlap < 0.0 ||
      control.minimum_reference_overlap > 1.0)
    throw std::invalid_argument("Matrix-free LM controls must be finite and nonnegative");
  if (control.relative_residual_tolerance == 0.0 &&
      control.absolute_residual_tolerance == 0.0)
    throw std::invalid_argument("Matrix-free LM requires a positive residual tolerance");
  if (control.maximum_subspace_dimension < 2)
    throw std::invalid_argument("Matrix-free LM requires a subspace dimension of at least two");
  if (control.maximum_projected_sweeps == 0)
    throw std::invalid_argument("Matrix-free LM requires at least one projected eigensolver sweep");
}

/// Require H, S, the preconditioner, and initial vector to share one real identity.
void validateProblem(const AugmentedLinearMethodOperator& hamiltonian,
                     const AugmentedLinearMethodOperator& overlap,
                     const MatrixFreePreconditioner& preconditioner,
                     DerivativeArrayView<const DerivativeValue> initial_vector)
{
  const AugmentedLinearMethodOperatorDescriptor& h = hamiltonian.descriptor();
  const AugmentedLinearMethodOperatorDescriptor& s = overlap.descriptor();
  if (!h.hermitian || !s.hermitian || h.reduction_domain != ReductionDomain::GLOBAL ||
      s.reduction_domain != ReductionDomain::GLOBAL)
    throw std::invalid_argument("Matrix-free LM requires global Hermitian H and S actions");
  if (h.provider_id != s.provider_id || h.schema_fingerprint != s.schema_fingerprint ||
      h.parameter_version != s.parameter_version || h.tangent_count != s.tangent_count)
    throw std::invalid_argument("Matrix-free LM Hamiltonian and overlap identities differ");
  if (preconditioner.parameterSchema().providerId() != h.provider_id ||
      preconditioner.parameterSchema().fingerprint() != h.schema_fingerprint ||
      preconditioner.parameterVersion() != h.parameter_version)
    throw std::invalid_argument("Matrix-free LM preconditioner identity differs from H and S");
  validateAugmentedValues(initial_vector, h.tangent_count + 1,
                          "Matrix-free LM initial vector", false);
  if (initial_vector[0] != DerivativeValue{1.0, 0.0})
    throw std::invalid_argument(
        "Matrix-free LM initial vector must be the canonical reference state");
  for (std::size_t index = 1; index < initial_vector.size(); ++index)
    if (initial_vector[index] != DerivativeValue{})
      throw std::invalid_argument(
          "Matrix-free LM initial vector must contain zero tangent coordinates");
}

/// Bound imaginary roundoff in a nominally real Hermitian contraction.
bool contractionIsReal(DerivativeValue value,
                       DerivativeReal scale,
                       DerivativeReal tolerance_multiplier,
                       std::size_t parameter_count) noexcept
{
  const DerivativeReal accumulated_roundoff =
      static_cast<DerivativeReal>(parameter_count) *
      std::numeric_limits<DerivativeReal>::epsilon();
  const DerivativeReal gamma = accumulated_roundoff < 0.5
      ? accumulated_roundoff / (1.0 - accumulated_roundoff)
      : 1.0;
  const DerivativeReal tolerance = tolerance_multiplier * gamma *
      std::max({DerivativeReal{1}, std::abs(value.real()), scale});
  return std::abs(value.imag()) <= tolerance;
}

/// Compare projected transposed entries with a scale-aware floating-point bound.
bool projectedPairIsHermitian(DerivativeReal left,
                              DerivativeReal right,
                              DerivativeReal tolerance_multiplier,
                              std::size_t parameter_count) noexcept
{
  const DerivativeReal accumulated_roundoff =
      static_cast<DerivativeReal>(parameter_count) *
      std::numeric_limits<DerivativeReal>::epsilon();
  const DerivativeReal gamma = accumulated_roundoff < 0.5
      ? accumulated_roundoff / (1.0 - accumulated_roundoff)
      : 1.0;
  const DerivativeReal tolerance = tolerance_multiplier * gamma *
      std::max({DerivativeReal{1}, std::abs(left), std::abs(right)});
  return std::abs(left - right) <= tolerance;
}

/// Diagonalize one small real symmetric matrix with bounded Jacobi sweeps.
bool smallestSymmetricEigenpair(std::vector<DerivativeReal> matrix,
                                std::size_t dimension,
                                DerivativeReal tolerance,
                                std::size_t maximum_sweeps,
                                DerivativeReal& eigenvalue,
                                std::vector<DerivativeReal>& eigenvector)
{
  std::vector<DerivativeReal> eigenvectors(dimension * dimension, 0.0);
  for (std::size_t index = 0; index < dimension; ++index)
    eigenvectors[index * dimension + index] = 1.0;

  bool converged = dimension <= 1;
  for (std::size_t sweep = 0; sweep < maximum_sweeps && !converged; ++sweep)
  {
    DerivativeReal largest_off_diagonal = 0.0;
    DerivativeReal diagonal_scale = 1.0;
    std::size_t pivot_row = 0;
    std::size_t pivot_column = 1;
    for (std::size_t row = 0; row < dimension; ++row)
    {
      diagonal_scale = std::max(diagonal_scale,
                                std::abs(matrix[row * dimension + row]));
      for (std::size_t column = row + 1; column < dimension; ++column)
        if (std::abs(matrix[row * dimension + column]) > largest_off_diagonal)
        {
          largest_off_diagonal = std::abs(matrix[row * dimension + column]);
          pivot_row = row;
          pivot_column = column;
        }
    }
    if (largest_off_diagonal <= tolerance * diagonal_scale)
    {
      converged = true;
      break;
    }

    const DerivativeReal app = matrix[pivot_row * dimension + pivot_row];
    const DerivativeReal aqq = matrix[pivot_column * dimension + pivot_column];
    const DerivativeReal apq = matrix[pivot_row * dimension + pivot_column];
    const DerivativeReal angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
    const DerivativeReal cosine = std::cos(angle);
    const DerivativeReal sine = std::sin(angle);

    for (std::size_t index = 0; index < dimension; ++index)
      if (index != pivot_row && index != pivot_column)
      {
        const DerivativeReal aip = matrix[index * dimension + pivot_row];
        const DerivativeReal aiq = matrix[index * dimension + pivot_column];
        const DerivativeReal rotated_p = cosine * aip - sine * aiq;
        const DerivativeReal rotated_q = sine * aip + cosine * aiq;
        matrix[index * dimension + pivot_row] = rotated_p;
        matrix[pivot_row * dimension + index] = rotated_p;
        matrix[index * dimension + pivot_column] = rotated_q;
        matrix[pivot_column * dimension + index] = rotated_q;
      }
    matrix[pivot_row * dimension + pivot_row] =
        cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
    matrix[pivot_column * dimension + pivot_column] =
        sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
    matrix[pivot_row * dimension + pivot_column] = 0.0;
    matrix[pivot_column * dimension + pivot_row] = 0.0;

    for (std::size_t row = 0; row < dimension; ++row)
    {
      const DerivativeReal vip = eigenvectors[row * dimension + pivot_row];
      const DerivativeReal viq = eigenvectors[row * dimension + pivot_column];
      eigenvectors[row * dimension + pivot_row] = cosine * vip - sine * viq;
      eigenvectors[row * dimension + pivot_column] = sine * vip + cosine * viq;
    }
  }
  if (!converged)
  {
    DerivativeReal largest_off_diagonal = 0.0;
    DerivativeReal diagonal_scale = 1.0;
    for (std::size_t row = 0; row < dimension; ++row)
    {
      diagonal_scale = std::max(diagonal_scale,
                                std::abs(matrix[row * dimension + row]));
      for (std::size_t column = row + 1; column < dimension; ++column)
        largest_off_diagonal = std::max(
            largest_off_diagonal,
            std::abs(matrix[row * dimension + column]));
    }
    converged = largest_off_diagonal <= tolerance * diagonal_scale;
  }
  if (!converged)
    return false;

  std::size_t lowest = 0;
  for (std::size_t index = 1; index < dimension; ++index)
    if (matrix[index * dimension + index] < matrix[lowest * dimension + lowest])
      lowest = index;
  eigenvalue = matrix[lowest * dimension + lowest];
  eigenvector.resize(dimension);
  for (std::size_t row = 0; row < dimension; ++row)
    eigenvector[row] = eigenvectors[row * dimension + lowest];
  return isFiniteTrainingReal(eigenvalue);
}

/** Solve one tiny symmetric-definite projected generalized eigenproblem.
 *
 * Cholesky whitening maps H*y=lambda*S*y to an ordinary symmetric problem.
 */
bool smallestGeneralizedEigenpair(const std::vector<DerivativeReal>& hamiltonian,
                                  const std::vector<DerivativeReal>& overlap,
                                  std::size_t dimension,
                                  const MatrixFreeLinearMethodControl& control,
                                  DerivativeReal& eigenvalue,
                                  std::vector<DerivativeReal>& coefficients)
{
  std::vector<DerivativeReal> cholesky(dimension * dimension, 0.0);
  for (std::size_t row = 0; row < dimension; ++row)
    for (std::size_t column = 0; column <= row; ++column)
    {
      DerivativeReal value = overlap[row * dimension + column];
      for (std::size_t inner = 0; inner < column; ++inner)
        value -= cholesky[row * dimension + inner] *
            cholesky[column * dimension + inner];
      if (row == column)
      {
        if (!isFiniteTrainingReal(value) || value <= control.minimum_overlap_norm)
          return false;
        cholesky[row * dimension + column] = std::sqrt(value);
      }
      else
        cholesky[row * dimension + column] =
            value / cholesky[column * dimension + column];
    }

  // Invert the tiny lower-triangular Cholesky factor one column at a time.
  std::vector<DerivativeReal> inverse(dimension * dimension, 0.0);
  for (std::size_t column = 0; column < dimension; ++column)
    for (std::size_t row = 0; row < dimension; ++row)
    {
      DerivativeReal value = row == column ? 1.0 : 0.0;
      for (std::size_t inner = 0; inner < row; ++inner)
        value -= cholesky[row * dimension + inner] *
            inverse[inner * dimension + column];
      inverse[row * dimension + column] =
          value / cholesky[row * dimension + row];
    }

  std::vector<DerivativeReal> temporary(dimension * dimension, 0.0);
  std::vector<DerivativeReal> whitened(dimension * dimension, 0.0);
  for (std::size_t row = 0; row < dimension; ++row)
    for (std::size_t column = 0; column < dimension; ++column)
      for (std::size_t inner = 0; inner < dimension; ++inner)
        temporary[row * dimension + column] +=
            inverse[row * dimension + inner] *
            hamiltonian[inner * dimension + column];
  for (std::size_t row = 0; row < dimension; ++row)
    for (std::size_t column = 0; column < dimension; ++column)
      for (std::size_t inner = 0; inner < dimension; ++inner)
        whitened[row * dimension + column] +=
            temporary[row * dimension + inner] *
            inverse[column * dimension + inner];

  // Remove only roundoff-level projected asymmetry before the Hermitian solver.
  for (std::size_t row = 0; row < dimension; ++row)
    for (std::size_t column = row + 1; column < dimension; ++column)
    {
      const DerivativeReal average = 0.5 *
          (whitened[row * dimension + column] +
           whitened[column * dimension + row]);
      whitened[row * dimension + column] = average;
      whitened[column * dimension + row] = average;
    }

  std::vector<DerivativeReal> whitened_vector;
  if (!smallestSymmetricEigenpair(std::move(whitened), dimension,
                                  control.projected_eigensolver_tolerance,
                                  control.maximum_projected_sweeps, eigenvalue,
                                  whitened_vector))
    return false;

  // Transform z back with y=L^-T*z and normalize in the projected S metric.
  coefficients.assign(dimension, 0.0);
  for (std::size_t row = 0; row < dimension; ++row)
    for (std::size_t inner = 0; inner < dimension; ++inner)
      coefficients[row] += inverse[inner * dimension + row] *
          whitened_vector[inner];
  DerivativeReal norm_squared = 0.0;
  for (std::size_t row = 0; row < dimension; ++row)
    for (std::size_t column = 0; column < dimension; ++column)
      norm_squared += coefficients[row] * overlap[row * dimension + column] *
          coefficients[column];
  if (!isFiniteTrainingReal(norm_squared) || norm_squared <= 0.0)
    return false;
  const DerivativeReal inverse_norm = 1.0 / std::sqrt(norm_squared);
  for (DerivativeReal& coefficient : coefficients)
    coefficient *= inverse_norm;
  return true;
}

/// Finalize common status fields without changing a diagnostic candidate.
void finishResult(MatrixFreeLinearMethodResult& result,
                  LinearMethodStopReason reason,
                  Clock::time_point start) noexcept
{
  result.stop_reason = reason;
  result.converged = reason == LinearMethodStopReason::CONVERGED;
  result.elapsed_seconds = elapsedSeconds(start);
}

} // namespace

AugmentedLinearMethodOperator::AugmentedLinearMethodOperator(
    const StructuredParameterSchema& tangent_schema,
    std::size_t parameter_version,
    ReductionDomain reduction_domain,
    bool hermitian)
    : tangent_schema_(tangent_schema),
      descriptor_{tangent_schema_.providerId(), tangent_schema_.fingerprint(),
                  parameter_version, tangent_schema_.parameterCount(), reduction_domain,
                  hermitian}
{
  for (const ParameterBlockDescriptor& block : tangent_schema_.blocks())
    if (block.scalar_domain != ParameterScalarDomain::REAL64)
      throw std::invalid_argument(
          "Augmented linear-method operators currently require a REAL64 tangent schema");
  if (descriptor_.tangent_count == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("Augmented linear-method dimension overflows size_t");
}

void AugmentedLinearMethodOperator::apply(
    DerivativeArrayView<const DerivativeValue> direction,
    DerivativeArrayView<DerivativeValue> result) const
{
  const std::size_t augmented_size = descriptor_.tangent_count + 1;
  validateAugmentedValues(direction, augmented_size,
                          "Augmented linear-method direction", false);
  if (result.size() != augmented_size || (!result.empty() && result.data() == nullptr))
    throw std::invalid_argument("Augmented linear-method result extent is not P+1");
  if (intervalsOverlap(direction.data(), direction.size(), result.data(), result.size()))
    throw std::invalid_argument(
        "Augmented linear-method input and output intervals must not overlap");

  const DerivativeReal sentinel = std::numeric_limits<DerivativeReal>::quiet_NaN();
  std::fill(result.begin(), result.end(), DerivativeValue{sentinel, sentinel});
  evaluate(direction, result);
  validateAugmentedValues(
      DerivativeArrayView<const DerivativeValue>{result.data(), result.size()},
      augmented_size, "Augmented linear-method result", true);
}

const char* linearMethodStopReasonName(LinearMethodStopReason reason) noexcept
{
  switch (reason)
  {
  case LinearMethodStopReason::CONVERGED:
    return "converged";
  case LinearMethodStopReason::ITERATION_LIMIT:
    return "iteration_limit";
  case LinearMethodStopReason::OPERATOR_APPLICATION_LIMIT:
    return "operator_application_limit";
  case LinearMethodStopReason::TIME_LIMIT:
    return "time_limit";
  case LinearMethodStopReason::LINEAR_DEPENDENCE:
    return "linear_dependence";
  case LinearMethodStopReason::NON_POSITIVE_OVERLAP:
    return "non_positive_overlap";
  case LinearMethodStopReason::INVALID_REFERENCE_OVERLAP:
    return "invalid_reference_overlap";
  case LinearMethodStopReason::NON_HERMITIAN_ACTION:
    return "non_hermitian_action";
  case LinearMethodStopReason::NONFINITE_RESULT:
    return "nonfinite_result";
  case LinearMethodStopReason::PROJECTED_SOLVE_FAILURE:
    return "projected_solve_failure";
  case LinearMethodStopReason::UNTRUSTED_ROOT:
    return "untrusted_root";
  case LinearMethodStopReason::BREAKDOWN:
    return "breakdown";
  }
  return "unknown";
}

MatrixFreeLinearMethodResult solveSymmetrizedMatrixFreeLinearMethod(
    const AugmentedLinearMethodOperator& hamiltonian,
    const AugmentedLinearMethodOperator& overlap,
    const MatrixFreePreconditioner& preconditioner,
    DerivativeArrayView<const DerivativeValue> initial_vector,
    const MatrixFreeLinearMethodControl& control)
{
  validateControl(control);
  validateProblem(hamiltonian, overlap, preconditioner, initial_vector);

  const StructuredParameterSchema& schema = hamiltonian.tangentSchema();
  const AugmentedLinearMethodOperatorDescriptor& descriptor = hamiltonian.descriptor();
  const std::size_t tangent_count = descriptor.tangent_count;
  const std::size_t augmented_size = tangent_count + 1;
  const Clock::time_point start = Clock::now();

  MatrixFreeLinearMethodResult result;
  result.provider_id = descriptor.provider_id;
  result.schema_fingerprint = descriptor.schema_fingerprint;
  result.parameter_version = descriptor.parameter_version;

  DavidsonBasis basis;
  basis.vectors.reserve(control.maximum_subspace_dimension);
  basis.hamiltonian_images.reserve(control.maximum_subspace_dimension);
  basis.overlap_images.reserve(control.maximum_subspace_dimension);
  std::vector<DerivativeValue> candidate(augmented_size);
  std::vector<DerivativeValue> candidate_h(augmented_size);
  std::vector<DerivativeValue> candidate_s(augmented_size);
  std::vector<DerivativeValue> residual(augmented_size);
  std::vector<DerivativeValue> correction(augmented_size);
  std::vector<DerivativeValue> ritz_h(augmented_size);
  std::vector<DerivativeValue> ritz_s(augmented_size);
  result.peak_parameter_vectors = 7;

  auto timeExpired = [&]() {
    return control.maximum_seconds > 0.0 &&
        elapsedSeconds(start) >= control.maximum_seconds;
  };

  /** Orthogonalize one candidate, apply H/S, and append it atomically to the basis. */
  auto appendCandidate = [&](std::vector<DerivativeValue>& input)
      -> LinearMethodStopReason {
    if (timeExpired())
      return LinearMethodStopReason::TIME_LIMIT;
    if (result.operator_applications >= control.maximum_operator_applications)
      return LinearMethodStopReason::OPERATOR_APPLICATION_LIMIT;
    ++result.operator_applications;
    overlap.apply(constView(input), mutableView(candidate_s));

    // The centered tangent convention requires S*[1;0]=[1;0].  Checking this
    // first action prevents a provider with an uncentered reference row from being
    // accepted merely because projected symmetrization hides the discrepancy.
    if (basis.vectors.empty() && result.iterations == 0)
    {
      const DerivativeReal accumulated_roundoff =
          static_cast<DerivativeReal>(augmented_size) *
          std::numeric_limits<DerivativeReal>::epsilon();
      const DerivativeReal gamma = accumulated_roundoff < 0.5
          ? accumulated_roundoff / (1.0 - accumulated_roundoff)
          : 1.0;
      const DerivativeReal tolerance = control.hermitian_tolerance * gamma;
      if (std::abs(candidate_s[0].real() - 1.0) > tolerance)
        return LinearMethodStopReason::INVALID_REFERENCE_OVERLAP;
      for (std::size_t index = 1; index < candidate_s.size(); ++index)
        if (std::abs(candidate_s[index].real()) > tolerance)
          return LinearMethodStopReason::INVALID_REFERENCE_OVERLAP;
    }

    // Two passes control loss of overlap-metric orthogonality as the subspace grows.
    for (int pass = 0; pass < 2; ++pass)
      for (std::size_t column = 0; column < basis.vectors.size(); ++column)
      {
        const DerivativeValue contraction = augmentedDot(
            constView(basis.vectors[column]), constView(candidate_s));
        const DerivativeReal scale = augmentedNorm(constView(basis.vectors[column])) *
            augmentedNorm(constView(candidate_s));
        if (!contractionIsReal(contraction, scale, control.hermitian_tolerance,
                               augmented_size))
          return LinearMethodStopReason::NON_HERMITIAN_ACTION;
        augmentedAxpy({-contraction.real(), 0.0},
                      constView(basis.vectors[column]), mutableView(input));
        augmentedAxpy({-contraction.real(), 0.0},
                      constView(basis.overlap_images[column]), mutableView(candidate_s));
      }

    const DerivativeValue norm_squared_value = augmentedDot(
        constView(input), constView(candidate_s));
    const DerivativeReal norm_scale = augmentedNorm(constView(input)) *
        augmentedNorm(constView(candidate_s));
    if (!contractionIsReal(norm_squared_value, norm_scale,
                           control.hermitian_tolerance, augmented_size))
      return LinearMethodStopReason::NON_HERMITIAN_ACTION;
    if (norm_squared_value.real() < 0.0)
      return LinearMethodStopReason::NON_POSITIVE_OVERLAP;
    if (std::sqrt(norm_squared_value.real()) <= control.minimum_overlap_norm)
      return LinearMethodStopReason::LINEAR_DEPENDENCE;
    const DerivativeReal inverse_norm = 1.0 / std::sqrt(norm_squared_value.real());
    augmentedScale({inverse_norm, 0.0}, mutableView(input));
    augmentedScale({inverse_norm, 0.0}, mutableView(candidate_s));

    if (timeExpired())
      return LinearMethodStopReason::TIME_LIMIT;
    if (result.operator_applications >= control.maximum_operator_applications)
      return LinearMethodStopReason::OPERATOR_APPLICATION_LIMIT;
    ++result.operator_applications;
    hamiltonian.apply(constView(input), mutableView(candidate_h));

    basis.vectors.push_back(input);
    basis.hamiltonian_images.push_back(candidate_h);
    basis.overlap_images.push_back(candidate_s);
    result.peak_subspace_dimension =
        std::max(result.peak_subspace_dimension, basis.vectors.size());
    result.peak_parameter_vectors = std::max(
        result.peak_parameter_vectors, 7 + 3 * basis.vectors.size());
    return LinearMethodStopReason::CONVERGED;
  };

  try
  {
    std::copy(initial_vector.begin(), initial_vector.end(), candidate.begin());
    LinearMethodStopReason append_status = appendCandidate(candidate);
    if (append_status != LinearMethodStopReason::CONVERGED)
    {
      finishResult(result, append_status, start);
      return result;
    }

    while (result.iterations < control.maximum_iterations)
    {
      if (timeExpired())
      {
        finishResult(result, LinearMethodStopReason::TIME_LIMIT, start);
        return result;
      }

      const std::size_t dimension = basis.vectors.size();
      std::vector<DerivativeReal> projected_h(dimension * dimension);
      std::vector<DerivativeReal> projected_s(dimension * dimension);
      for (std::size_t row = 0; row < dimension; ++row)
        for (std::size_t column = 0; column < dimension; ++column)
        {
          const DerivativeValue h_value = augmentedDot(
              constView(basis.vectors[row]),
              constView(basis.hamiltonian_images[column]));
          const DerivativeValue s_value = augmentedDot(
              constView(basis.vectors[row]),
              constView(basis.overlap_images[column]));
          const DerivativeReal h_scale =
              augmentedNorm(constView(basis.vectors[row])) *
              augmentedNorm(constView(basis.hamiltonian_images[column]));
          const DerivativeReal s_scale =
              augmentedNorm(constView(basis.vectors[row])) *
              augmentedNorm(constView(basis.overlap_images[column]));
          if (!contractionIsReal(h_value, h_scale, control.hermitian_tolerance,
                                 augmented_size) ||
              !contractionIsReal(s_value, s_scale, control.hermitian_tolerance,
                                 augmented_size))
          {
            finishResult(result, LinearMethodStopReason::NON_HERMITIAN_ACTION, start);
            return result;
          }
          projected_h[row * dimension + column] = h_value.real();
          projected_s[row * dimension + column] = s_value.real();
        }

      // Detect material asymmetry before symmetrizing roundoff for the tiny solver.
      for (std::size_t row = 0; row < dimension; ++row)
        for (std::size_t column = row + 1; column < dimension; ++column)
        {
          if (!projectedPairIsHermitian(projected_h[row * dimension + column],
                                        projected_h[column * dimension + row],
                                        control.hermitian_tolerance, augmented_size) ||
              !projectedPairIsHermitian(projected_s[row * dimension + column],
                                        projected_s[column * dimension + row],
                                        control.hermitian_tolerance, augmented_size))
          {
            finishResult(result, LinearMethodStopReason::NON_HERMITIAN_ACTION, start);
            return result;
          }
          const DerivativeReal h_average = 0.5 *
              (projected_h[row * dimension + column] +
               projected_h[column * dimension + row]);
          const DerivativeReal s_average = 0.5 *
              (projected_s[row * dimension + column] +
               projected_s[column * dimension + row]);
          projected_h[row * dimension + column] = h_average;
          projected_h[column * dimension + row] = h_average;
          projected_s[row * dimension + column] = s_average;
          projected_s[column * dimension + row] = s_average;
        }

      std::vector<DerivativeReal> coefficients;
      if (!smallestGeneralizedEigenpair(projected_h, projected_s, dimension,
                                        control, result.eigenvalue, coefficients))
      {
        finishResult(result, LinearMethodStopReason::PROJECTED_SOLVE_FAILURE, start);
        return result;
      }

      result.eigenvector.assign(augmented_size, DerivativeValue{});
      result.peak_parameter_vectors = std::max(
          result.peak_parameter_vectors, 8 + 3 * basis.vectors.size());
      std::fill(ritz_h.begin(), ritz_h.end(), DerivativeValue{});
      std::fill(ritz_s.begin(), ritz_s.end(), DerivativeValue{});
      for (std::size_t column = 0; column < dimension; ++column)
      {
        augmentedAxpy({coefficients[column], 0.0},
                      constView(basis.vectors[column]), mutableView(result.eigenvector));
        augmentedAxpy({coefficients[column], 0.0},
                      constView(basis.hamiltonian_images[column]), mutableView(ritz_h));
        augmentedAxpy({coefficients[column], 0.0},
                      constView(basis.overlap_images[column]), mutableView(ritz_s));
      }
      residual = ritz_h;
      augmentedAxpy({-result.eigenvalue, 0.0}, constView(ritz_s), mutableView(residual));
      result.residual_norm = augmentedNorm(constView(residual));
      const DerivativeReal residual_scale = std::max(
          {DerivativeReal{1}, augmentedNorm(constView(ritz_h)),
           std::abs(result.eigenvalue) *
               augmentedNorm(constView(ritz_s))});
      result.relative_residual_norm = result.residual_norm / residual_scale;
      ++result.iterations;
      const DerivativeReal target = std::max(
          control.absolute_residual_tolerance,
          control.relative_residual_tolerance * residual_scale);
      if (result.residual_norm <= target)
      {
        // The eigenvector is S-normalized, so |c0| is exactly the requested
        // reference-overlap weight.  Reject roots whose parameter update c_p/c0
        // would amplify noise even though their Ritz residual is small.
        if (std::abs(result.eigenvector[0].real()) <
            control.minimum_reference_overlap)
        {
          finishResult(result, LinearMethodStopReason::UNTRUSTED_ROOT, start);
          return result;
        }
        finishResult(result, LinearMethodStopReason::CONVERGED, start);
        return result;
      }
      if (result.iterations >= control.maximum_iterations)
      {
        finishResult(result, LinearMethodStopReason::ITERATION_LIMIT, start);
        return result;
      }

      // The reference coefficient is not a model parameter.  Apply the model
      // preconditioner only to the tangent residual and omit a c0 correction.
      correction[0] = {};
      StructuredParameterVectorConstView tangent_residual_view(
          schema, descriptor.parameter_version,
          {residual.data() + 1, tangent_count});
      preconditioner.apply(
          tangent_residual_view, {correction.data() + 1, tangent_count});
      augmentedScale({-1.0, 0.0}, mutableView(correction));

      if (basis.vectors.size() == control.maximum_subspace_dimension)
      {
        // Thick restart with the normalized Ritz vector and already available images.
        basis.vectors.clear();
        basis.hamiltonian_images.clear();
        basis.overlap_images.clear();
        basis.vectors.push_back(result.eigenvector);
        basis.hamiltonian_images.push_back(ritz_h);
        basis.overlap_images.push_back(ritz_s);
        ++result.restart_count;
      }

      append_status = appendCandidate(correction);
      if (append_status != LinearMethodStopReason::CONVERGED)
      {
        finishResult(result, append_status, start);
        return result;
      }
    }

    finishResult(result, LinearMethodStopReason::ITERATION_LIMIT, start);
    return result;
  }
  catch (const MatrixFreeNumericalError&)
  {
    result.eigenvector.clear();
    finishResult(result, LinearMethodStopReason::NONFINITE_RESULT, start);
    return result;
  }
}

LinearMethodUpdateCandidate formLinearMethodUpdateCandidate(
    const StructuredParameterSchema& tangent_schema,
    const MatrixFreeLinearMethodResult& result,
    const LinearMethodUpdateControl& control)
{
  if (!result.converged || result.stop_reason != LinearMethodStopReason::CONVERGED)
    throw std::invalid_argument("Linear-method update requires a converged solve");
  if (result.provider_id != tangent_schema.providerId() ||
      result.schema_fingerprint != tangent_schema.fingerprint() ||
      result.eigenvector.size() != tangent_schema.parameterCount() + 1)
    throw std::invalid_argument("Linear-method result does not match the tangent schema");
  if (!isFiniteTrainingReal(control.minimum_reference_magnitude) ||
      control.minimum_reference_magnitude <= 0.0 ||
      !isFiniteTrainingReal(control.maximum_update_norm) ||
      control.maximum_update_norm < 0.0)
    throw std::invalid_argument("Linear-method update controls are invalid");

  const DerivativeValue reference = result.eigenvector[0];
  if (!isFinite(reference) || reference.imag() != 0.0 ||
      std::abs(reference.real()) < control.minimum_reference_magnitude)
    throw std::invalid_argument("Linear-method reference coefficient is unusable");

  LinearMethodUpdateCandidate candidate;
  candidate.direction.resize(tangent_schema.parameterCount());
  for (const ParameterBlockDescriptor& block : tangent_schema.blocks())
    if (block.trainable)
      for (std::size_t index = block.offset; index < block.offset + block.count; ++index)
      {
        const DerivativeValue value = result.eigenvector[index + 1] / reference.real();
        if (!isFinite(value) || value.imag() != 0.0)
          throw std::invalid_argument("Linear-method tangent coefficient is invalid");
        candidate.direction[index] = value;
      }
  candidate.unscaled_norm = parameterVectorNorm(
      tangent_schema, constView(candidate.direction));
  if (control.maximum_update_norm > 0.0 &&
      candidate.unscaled_norm > control.maximum_update_norm)
  {
    candidate.applied_scale = control.maximum_update_norm / candidate.unscaled_norm;
    scaleParameterVector(tangent_schema, {candidate.applied_scale, 0.0},
                         mutableView(candidate.direction));
  }
  return candidate;
}

} // namespace qmcplusplus::wftrain
