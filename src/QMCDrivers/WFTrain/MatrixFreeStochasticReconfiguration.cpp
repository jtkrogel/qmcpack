//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file MatrixFreeStochasticReconfiguration.cpp
 * @brief Streaming centered-score covariance and damped-PCG update implementation.
 */

#include "QMCDrivers/WFTrain/MatrixFreeStochasticReconfiguration.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace qmcplusplus::wftrain
{
namespace
{

/// Report whether both components of one derivative scalar are finite.
bool isFinite(DerivativeValue value) noexcept
{
  return isFiniteTrainingReal(value.real()) && isFiniteTrainingReal(value.imag());
}

/// Return the homogeneous scalar domain or reject a mixed/unsupported schema.
ParameterScalarDomain homogeneousScalarDomain(const StructuredParameterSchema& schema)
{
  const ParameterScalarDomain domain = schema.blocks().front().scalar_domain;
  if (domain != ParameterScalarDomain::REAL64 &&
      domain != ParameterScalarDomain::COMPLEX128)
    throw std::invalid_argument("SR requires a real64 or complex128 parameter schema");
  for (const ParameterBlockDescriptor& block : schema.blocks())
    if (block.scalar_domain != domain)
      throw std::invalid_argument("SR requires one homogeneous parameter scalar domain");
  return domain;
}

/// Collect one checked JVP stream into preallocated sample storage.
class SRJVPCollector final : public SampleProductSink
{
public:
  SRJVPCollector(const StreamingDerivativeOperator& derivative_operator,
                 DerivativeArrayView<DerivativeValue> destination)
      : derivative_operator_(derivative_operator), destination_(destination)
  {}

protected:
  void onBegin(const DerivativeStreamDescriptor& descriptor) override
  {
    if (descriptor.provider_id != derivative_operator_.parameterSchema().providerId() ||
        descriptor.schema_fingerprint != derivative_operator_.parameterSchema().fingerprint() ||
        descriptor.parameter_version != derivative_operator_.parameterVersion() ||
        descriptor.batch_ordinal != derivative_operator_.batchOrdinal() ||
        descriptor.sample_offset != derivative_operator_.sampleOffset() ||
        descriptor.sample_count != destination_.size())
      throw std::invalid_argument("SR JVP stream identity does not match its local batch");
    const DerivativeReal sentinel = std::numeric_limits<DerivativeReal>::quiet_NaN();
    std::fill(destination_.begin(), destination_.end(), DerivativeValue{sentinel, sentinel});
  }

  void consume(const SampleProductTileConstView& tile) override
  {
    const std::size_t offset = tile.descriptor().sample_offset -
        derivative_operator_.sampleOffset();
    std::copy(tile.values().begin(), tile.values().end(), destination_.begin() + offset);
  }

  void onAbort() noexcept override
  {
    const DerivativeReal sentinel = std::numeric_limits<DerivativeReal>::quiet_NaN();
    std::fill(destination_.begin(), destination_.end(), DerivativeValue{sentinel, sentinel});
  }

private:
  const StreamingDerivativeOperator& derivative_operator_;
  DerivativeArrayView<DerivativeValue> destination_;
};

/// Accumulate one checked Hermitian VJP into an existing local P-vector.
class SRVJPAccumulator final : public ParameterReductionSink
{
public:
  SRVJPAccumulator(const StructuredParameterSchema& schema,
                   std::size_t parameter_version,
                   DerivativeArrayView<DerivativeValue> destination)
      : schema_(schema), parameter_version_(parameter_version), destination_(destination)
  {}

protected:
  void onBegin(const DerivativeStreamDescriptor& descriptor,
               const ParameterChunkPlan&,
               DerivativeArrayView<const VJPCoefficientChannel> channels) override
  {
    if (descriptor.provider_id != schema_.providerId() ||
        descriptor.schema_fingerprint != schema_.fingerprint() ||
        descriptor.parameter_version != parameter_version_ ||
        descriptor.adjoint != DerivativeAdjoint::HERMITIAN ||
        channels.size() != 1 ||
        channels[0].product != DerivativeProduct::SCORE_VJP)
      throw std::invalid_argument("SR VJP stream does not match its covariance action");
  }

  void consume(std::size_t channel_ordinal,
               const ParameterChunkConstView& chunk) override
  {
    if (channel_ordinal != 0)
      throw std::logic_error("SR received an unexpected VJP channel ordinal");
    const std::size_t offset = chunk.descriptor().parameter_offset;
    for (std::size_t index = 0; index < chunk.values().size(); ++index)
      destination_[offset + index] += chunk.values()[index];
  }

private:
  const StructuredParameterSchema& schema_;
  std::size_t parameter_version_;
  DerivativeArrayView<DerivativeValue> destination_;
};

/// Create coefficient metadata bound to one exact local derivative batch.
CoefficientView coefficientsFor(const StreamingDerivativeOperator& derivative_operator,
                                DerivativeArrayView<const DerivativeValue> values)
{
  return {derivative_operator.parameterSchema().providerId(),
          derivative_operator.parameterSchema().fingerprint(),
          derivative_operator.parameterVersion(),
          derivative_operator.batchOrdinal(),
          derivative_operator.sampleOffset(), values};
}

/// Return a const view over one owning derivative vector.
DerivativeArrayView<const DerivativeValue> constView(
    const std::vector<DerivativeValue>& values) noexcept
{
  return {values.data(), values.size()};
}

/// Return a mutable view over one owning derivative vector.
DerivativeArrayView<DerivativeValue> mutableView(
    std::vector<DerivativeValue>& values) noexcept
{
  return {values.data(), values.size()};
}

/// Validate one finite nonnegative optional norm bound.
void validateOptionalBound(DerivativeReal value, const char* description)
{
  if (!isFiniteTrainingReal(value) || value < 0.0)
    throw std::invalid_argument(std::string(description) + " must be finite and nonnegative");
}

/** Bound roundoff in a length-P Hermitian contraction.
 *
 * The product of the vector norms bounds the sum of absolute scalar products.
 * Combining that scale with the standard gamma(P) accumulation bound avoids
 * rejecting a positive-semidefinite metric solely because a large cancellation
 * leaves a tiny negative residual.
 */
DerivativeReal contractionRoundoffTolerance(std::size_t parameter_count,
                                             DerivativeValue contraction,
                                             DerivativeReal product_norm) noexcept
{
  const DerivativeReal accumulated_roundoff =
      static_cast<DerivativeReal>(parameter_count) *
      std::numeric_limits<DerivativeReal>::epsilon();
  const DerivativeReal gamma = accumulated_roundoff < 0.5
      ? accumulated_roundoff / (1.0 - accumulated_roundoff)
      : 1.0;
  return 128.0 * gamma *
      std::max({DerivativeReal{1}, std::abs(contraction.real()), product_norm});
}

} // namespace

/// Retain copied local weights and one reusable complex sample-product buffer.
struct StochasticReconfigurationOperator::BatchStorage
{
  const StreamingDerivativeOperator* derivative_operator = nullptr;
  std::vector<DerivativeReal> weights;
  mutable std::vector<DerivativeValue> sample_products;
};

StochasticReconfigurationOperator::StochasticReconfigurationOperator(
    const StructuredParameterSchema& schema,
    std::size_t parameter_version,
    DerivativeArrayView<const StochasticReconfigurationBatch> batches,
    DistributedParameterReduction reduction)
    : MatrixFreeLinearOperator(schema, parameter_version, ReductionDomain::GLOBAL, true),
      reduction_(std::move(reduction)),
      scalar_domain_(homogeneousScalarDomain(schema)),
      projected_direction_(schema.parameterCount()),
      local_action_(schema.parameterCount())
{
  if (batches.empty() || batches.data() == nullptr)
    throw std::invalid_argument("SR requires at least one local batch descriptor");

  batches_.reserve(batches.size());
  std::vector<std::pair<std::size_t, std::size_t>> sample_intervals;
  sample_intervals.reserve(batches.size());
  for (const StochasticReconfigurationBatch& input : batches)
  {
    if (!input.derivative_operator)
      throw std::invalid_argument("SR received a null streaming derivative operator");
    const StreamingDerivativeOperator& derivative_operator = *input.derivative_operator;
    const StreamingDerivativeCapabilities capabilities = derivative_operator.capabilities();
    if (derivative_operator.parameterSchema().providerId() != schema.providerId() ||
        derivative_operator.parameterSchema().fingerprint() != schema.fingerprint() ||
        derivative_operator.parameterVersion() != parameter_version)
      throw std::invalid_argument("SR derivative batch schema or version does not match");
    if (input.weights.size() != derivative_operator.sampleCount() ||
        (!input.weights.empty() && input.weights.data() == nullptr))
      throw std::invalid_argument("SR weight extent does not match its derivative batch");
    if (!capabilities.supports(DerivativeProduct::SCORE_JVP) ||
        !capabilities.supports(DerivativeProduct::SCORE_VJP) ||
        !capabilities.supports(DerivativeAdjoint::HERMITIAN) ||
        capabilities.parameter_scalar_domain != scalar_domain_ ||
        capabilities.result_scalar_domain != ParameterScalarDomain::COMPLEX128 ||
        capabilities.execution_domain != DerivativeExecutionDomain::HOST ||
        capabilities.reduction_domain == ReductionDomain::GLOBAL ||
        !capabilities.block_streaming ||
        capabilities.maximum_parameter_chunk_size == 0 ||
        capabilities.maximum_sample_tile_size == 0 ||
        capabilities.maximum_vjp_channels == 0)
      throw std::invalid_argument("SR derivative batch lacks a required bounded score product");
    if (derivative_operator.sampleOffset() >
        std::numeric_limits<std::size_t>::max() - derivative_operator.sampleCount())
      throw std::overflow_error("SR derivative sample interval overflows size_t");
    const std::size_t end = derivative_operator.sampleOffset() +
        derivative_operator.sampleCount();
    sample_intervals.emplace_back(derivative_operator.sampleOffset(), end);

    BatchStorage storage;
    storage.derivative_operator = &derivative_operator;
    storage.weights.assign(input.weights.begin(), input.weights.end());
    for (DerivativeReal weight : storage.weights)
      if (!isFiniteTrainingReal(weight) || weight < 0.0)
        throw std::invalid_argument("SR sample weights must be finite and nonnegative");
    storage.sample_products.resize(storage.weights.size());
    batches_.push_back(std::move(storage));
  }

  // Overlapping local sample identities commonly indicate that one crowd was added
  // twice. Zero-length intervals are harmless and may coexist on an empty rank.
  std::sort(sample_intervals.begin(), sample_intervals.end());
  for (std::size_t index = 1; index < sample_intervals.size(); ++index)
    if (sample_intervals[index - 1].first != sample_intervals[index - 1].second &&
        sample_intervals[index].first != sample_intervals[index].second &&
        sample_intervals[index].first < sample_intervals[index - 1].second)
      throw std::invalid_argument("SR local derivative sample intervals overlap");
}

StochasticReconfigurationOperator::~StochasticReconfigurationOperator() = default;

StochasticReconfigurationStorageDiagnostics
StochasticReconfigurationOperator::storageDiagnostics() const noexcept
{
  StochasticReconfigurationStorageDiagnostics result;
  result.parameter_count = descriptor().parameter_count;
  result.complex_parameter_vectors = 2;
  for (const BatchStorage& batch : batches_)
  {
    result.local_sample_count += batch.weights.size();
    result.real_sample_values += batch.weights.capacity();
    result.complex_sample_values += batch.sample_products.capacity();
  }
  result.retained_numeric_bytes =
      result.real_sample_values * sizeof(DerivativeReal) +
      result.complex_sample_values * sizeof(DerivativeValue) +
      (projected_direction_.capacity() + local_action_.capacity()) *
          sizeof(DerivativeValue);
  return result;
}

void StochasticReconfigurationOperator::validateAndProjectDirection(
    const StructuredParameterVectorConstView& direction) const
{
  std::copy(direction.values().begin(), direction.values().end(),
            projected_direction_.begin());
  for (const ParameterBlockDescriptor& block : parameterSchema().blocks())
    if (!block.trainable)
      std::fill(projected_direction_.begin() + block.offset,
                projected_direction_.begin() + block.offset + block.count,
                DerivativeValue{});
}

void StochasticReconfigurationOperator::evaluate(
    const StructuredParameterVectorConstView& direction,
    DerivativeArrayView<DerivativeValue> result) const
{
  validateAndProjectDirection(direction);
  const StructuredParameterVectorConstView projected_view(
      parameterSchema(), descriptor().parameter_version,
      constView(projected_direction_));

  DistributedWeightedSampleMoments local_moments;
  std::exception_ptr jvp_failure;
  try
  {
    for (const BatchStorage& batch : batches_)
    {
      SRJVPCollector sink(*batch.derivative_operator,
                          mutableView(batch.sample_products));
      batch.derivative_operator->applyScoreJVP(projected_view, sink);
      if (batch.weights.size() >
          std::numeric_limits<std::size_t>::max() - local_moments.sample_count)
        throw std::overflow_error("SR local sample count overflow");
      local_moments.sample_count += batch.weights.size();
      for (std::size_t sample = 0; sample < batch.weights.size(); ++sample)
      {
        local_moments.weight_sum += batch.weights[sample];
        local_moments.weighted_value_sum +=
            batch.weights[sample] * batch.sample_products[sample];
      }
    }
    if (!isFiniteTrainingReal(local_moments.weight_sum) ||
        !isFinite(local_moments.weighted_value_sum))
      throw MatrixFreeNumericalError("SR local JVP moments are non-finite");
  }
  catch (...)
  {
    jvp_failure = std::current_exception();
    local_moments = {};
  }

  const DistributedWeightedSampleMoments global_moments =
      reduction_.reduceWeightedSampleMoments(
          parameterSchema(), descriptor().parameter_version, local_moments,
          jvp_failure);
  const DerivativeValue mean_product =
      global_moments.weighted_value_sum / global_moments.weight_sum;

  std::fill(local_action_.begin(), local_action_.end(), DerivativeValue{});
  std::exception_ptr vjp_failure;
  try
  {
    for (const BatchStorage& batch : batches_)
    {
      for (std::size_t sample = 0; sample < batch.weights.size(); ++sample)
        batch.sample_products[sample] =
            (batch.weights[sample] / global_moments.weight_sum) *
            (batch.sample_products[sample] - mean_product);
      const VJPCoefficientChannel channel{
          "sr/centered_score", DerivativeProduct::SCORE_VJP,
          coefficientsFor(*batch.derivative_operator,
                          constView(batch.sample_products)), 0};
      SRVJPAccumulator sink(parameterSchema(), descriptor().parameter_version,
                            mutableView(local_action_));
      batch.derivative_operator->applyVJPs({&channel, 1},
                                           DerivativeAdjoint::HERMITIAN, sink);
    }

    // For real variational parameters the SR metric is Re[J^H W J]. Projecting
    // before the sum is algebraically identical and satisfies the real-vector
    // distributed reduction contract without retaining a second P-vector.
    if (scalar_domain_ == ParameterScalarDomain::REAL64)
      for (DerivativeValue& value : local_action_)
        value = {value.real(), 0.0};
    // Enforce the left projection independently of producer chunk policy.  A
    // diagnostic producer is permitted to expose frozen chunks, but SR must still
    // implement P_train^T S P_train and return exactly zero frozen components.
    for (const ParameterBlockDescriptor& block : parameterSchema().blocks())
      if (!block.trainable)
        std::fill(local_action_.begin() + block.offset,
                  local_action_.begin() + block.offset + block.count,
                  DerivativeValue{});
  }
  catch (...)
  {
    vjp_failure = std::current_exception();
    std::fill(local_action_.begin(), local_action_.end(), DerivativeValue{});
  }

  reduction_.reduceParameterVector(parameterSchema(),
                                   descriptor().parameter_version,
                                   mutableView(local_action_), vjp_failure);
  std::copy(local_action_.begin(), local_action_.end(), result.begin());
}

StochasticReconfigurationUpdateRule::StochasticReconfigurationUpdateRule(
    const MatrixFreeLinearOperator& covariance,
    const MatrixFreePreconditioner& preconditioner,
    StochasticReconfigurationUpdateControl control)
    : covariance_(covariance), preconditioner_(preconditioner),
      control_(std::move(control))
{
  const MatrixFreeOperatorDescriptor& descriptor = covariance_.descriptor();
  if (!descriptor.hermitian || descriptor.reduction_domain != ReductionDomain::GLOBAL ||
      descriptor.scalar_domain != ParameterScalarDomain::REAL64)
    throw std::invalid_argument(
        "SR update rule requires a global Hermitian REAL64 covariance action");
  if (preconditioner_.parameterSchema().providerId() != descriptor.provider_id ||
      preconditioner_.parameterSchema().fingerprint() != descriptor.schema_fingerprint ||
      preconditioner_.parameterVersion() != descriptor.parameter_version)
    throw std::invalid_argument("SR update preconditioner identity does not match covariance");
  if (!isFiniteTrainingReal(control_.learning_rate) || control_.learning_rate <= 0.0)
    throw std::invalid_argument("SR learning rate must be finite and positive");
  if (!isFiniteTrainingReal(control_.initial_damping) || control_.initial_damping < 0.0)
    throw std::invalid_argument("SR damping must be finite and nonnegative");
  if (!isFiniteTrainingReal(control_.damping_multiplier) ||
      control_.damping_multiplier <= 1.0)
    throw std::invalid_argument("SR damping multiplier must be finite and greater than one");
  if (!isFiniteTrainingReal(control_.minimum_retry_damping) ||
      control_.minimum_retry_damping <= 0.0)
    throw std::invalid_argument("SR minimum retry damping must be finite and positive");
  if (control_.maximum_damping_attempts == 0)
    throw std::invalid_argument(
        "SR damping requires at least one solve attempt");
  validateOptionalBound(control_.maximum_update_norm, "SR maximum update norm");
  validateOptionalBound(control_.maximum_metric_norm, "SR maximum metric norm");
}

void StochasticReconfigurationUpdateRule::validateInputs(
    const StructuredParameterSchema& schema,
    const StructuredParameterSnapshot& parameters,
    ParameterGradientView objective) const
{
  const MatrixFreeOperatorDescriptor& descriptor = covariance_.descriptor();
  if (proposal_live_)
    throw std::logic_error("SR update rule already has a live proposal");
  if (schema.providerId() != descriptor.provider_id ||
      schema.fingerprint() != descriptor.schema_fingerprint ||
      parameters.schema_fingerprint != descriptor.schema_fingerprint ||
      parameters.version != descriptor.parameter_version ||
      parameters.values.size() != descriptor.parameter_count ||
      objective.schema_fingerprint != descriptor.schema_fingerprint ||
      objective.parameter_version != descriptor.parameter_version ||
      objective.gradient.size() != descriptor.parameter_count)
    throw std::invalid_argument("SR update inputs do not match the covariance identity");
  if (objective.reduction_domain != ReductionDomain::GLOBAL)
    throw std::invalid_argument("SR update requires a globally reduced objective gradient");
  for (double value : parameters.values)
    if (!isFiniteTrainingReal(value))
      throw std::invalid_argument("SR parameters contain a non-finite value");
  for (DerivativeReal value : objective.gradient)
    if (!isFiniteTrainingReal(value))
      throw std::invalid_argument("SR objective gradient contains a non-finite value");
}

StructuredParameterSnapshot StochasticReconfigurationUpdateRule::propose(
    const StructuredParameterSchema& schema,
    const StructuredParameterSnapshot& parameters,
    ParameterGradientView objective)
{
  validateInputs(schema, parameters, objective);
  last_diagnostics_.reset();

  std::vector<DerivativeValue> right_hand_side(schema.parameterCount());
  for (const ParameterBlockDescriptor& block : schema.blocks())
    if (block.trainable)
      for (std::size_t parameter = block.offset;
           parameter < block.offset + block.count; ++parameter)
        right_hand_side[parameter] = {objective.gradient[parameter], 0.0};
  const StructuredParameterVectorConstView rhs_view(
      schema, parameters.version, constView(right_hand_side));

  DerivativeReal damping = control_.initial_damping;
  KrylovSolveResult solve;
  for (std::size_t attempt = 1; attempt <= control_.maximum_damping_attempts;
       ++attempt)
  {
    ShiftedMatrixFreeOperator shifted(covariance_, damping);
    solve = solvePreconditionedConjugateGradient(
        shifted, preconditioner_, rhs_view, control_.krylov);
    last_diagnostics_.emplace();
    last_diagnostics_->damping_attempts = attempt;
    last_diagnostics_->damping = damping;
    last_diagnostics_->solve = solve;
    if (solve.converged)
      break;
    damping = damping == 0.0 ? control_.minimum_retry_damping
                             : damping * control_.damping_multiplier;
    if (!isFiniteTrainingReal(damping))
      break;
  }
  if (!solve.converged)
    throw std::runtime_error(
        std::string("SR PCG did not converge: ") +
        krylovStopReasonName(solve.stop_reason));

  std::vector<DerivativeValue> metric_action(schema.parameterCount());
  const StructuredParameterVectorConstView solution_view(
      schema, parameters.version,
      {solve.solution.data(), solve.solution.size()});
  covariance_.apply(solution_view, mutableView(metric_action));
  const DerivativeValue metric_squared = parameterVectorHermitianDot(
      schema, constView(solve.solution), constView(metric_action));
  const DerivativeReal metric_product_norm =
      parameterVectorNorm(schema, constView(solve.solution)) *
      parameterVectorNorm(schema, constView(metric_action));
  const DerivativeReal metric_tolerance = contractionRoundoffTolerance(
      schema.parameterCount(), metric_squared, metric_product_norm);
  if (std::abs(metric_squared.imag()) > metric_tolerance ||
      metric_squared.real() < -metric_tolerance)
    throw MatrixFreeNumericalError("SR covariance metric norm is not real and nonnegative");

  const DerivativeReal direction_norm =
      parameterVectorNorm(schema, constView(solve.solution));
  const DerivativeReal metric_norm =
      std::sqrt(std::max(DerivativeReal{}, metric_squared.real()));
  DerivativeReal scale = 1.0;
  if (control_.maximum_update_norm > 0.0 && direction_norm > 0.0)
    scale = std::min(scale, control_.maximum_update_norm /
                     (control_.learning_rate * direction_norm));
  if (control_.maximum_metric_norm > 0.0 && metric_norm > 0.0)
    scale = std::min(scale, control_.maximum_metric_norm /
                     (control_.learning_rate * metric_norm));
  if (!isFiniteTrainingReal(scale) || scale < 0.0)
    throw MatrixFreeNumericalError("SR trust control produced an invalid scale");

  StructuredParameterSnapshot candidate = parameters;
  for (const ParameterBlockDescriptor& block : schema.blocks())
    if (block.trainable)
      for (std::size_t parameter = block.offset;
           parameter < block.offset + block.count; ++parameter)
      {
        candidate.values[parameter] -= control_.learning_rate * scale *
            solve.solution[parameter].real();
        if (!isFiniteTrainingReal(candidate.values[parameter]))
          throw MatrixFreeNumericalError("SR candidate contains a non-finite value");
      }

  last_diagnostics_->direction_norm = direction_norm;
  last_diagnostics_->metric_norm = metric_norm;
  last_diagnostics_->applied_scale = scale;
  last_diagnostics_->solve = std::move(solve);
  proposal_live_ = true;
  return candidate;
}

void StochasticReconfigurationUpdateRule::proposalAccepted(
    const StructuredParameterSchema&,
    const StructuredParameterSnapshot&,
    ParameterGradientView) noexcept
{
  proposal_live_ = false;
}

void StochasticReconfigurationUpdateRule::proposalRejected() noexcept
{
  proposal_live_ = false;
}

} // namespace qmcplusplus::wftrain
