//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file MinimumStepStochasticReconfiguration.cpp
 * @brief Weighted score products and matrix-free minSR, SPRING, PII, and MinPII.
 */

#include "QMCDrivers/WFTrain/MinimumStepStochasticReconfiguration.h"

#include "QMCDrivers/WFTrain/TrainingNumerics.h"
#include "io/hdf/hdf_archive.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qmcplusplus::wftrain
{
namespace
{

using Clock = std::chrono::steady_clock;
constexpr char SPRING_CONTRIBUTOR_ID[] = "qmcpack/wftrain/spring";

/// Return a read-only view over one owning vector.
DerivativeArrayView<const DerivativeValue> constView(
    const std::vector<DerivativeValue>& values) noexcept
{
  return {values.data(), values.size()};
}

/// Return a mutable view over one owning vector.
DerivativeArrayView<DerivativeValue> mutableView(
    std::vector<DerivativeValue>& values) noexcept
{
  return {values.data(), values.size()};
}

/// Report whether both components of one scalar are finite.
bool isFinite(DerivativeValue value) noexcept
{
  return isFiniteTrainingReal(value.real()) && isFiniteTrainingReal(value.imag());
}

/// Return elapsed monotonic time.
double elapsedSeconds(Clock::time_point start) noexcept
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// Collect one complete score JVP into preallocated physical-sample storage.
class ScoreJVPCollector final : public SampleProductSink
{
public:
  ScoreJVPCollector(const StreamingDerivativeOperator& derivative_operator,
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
      throw std::invalid_argument("Score-factor JVP stream identity mismatch");
    const DerivativeReal sentinel = std::numeric_limits<DerivativeReal>::quiet_NaN();
    std::fill(destination_.begin(), destination_.end(),
              DerivativeValue{sentinel, sentinel});
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
    std::fill(destination_.begin(), destination_.end(),
              DerivativeValue{sentinel, sentinel});
  }

private:
  const StreamingDerivativeOperator& derivative_operator_;
  DerivativeArrayView<DerivativeValue> destination_;
};

/// Collect one exact term-masked local-energy JVP into preallocated storage.
class LocalEnergyJVPCollector final : public SampleProductSink
{
public:
  LocalEnergyJVPCollector(const StreamingDerivativeOperator& derivative_operator,
                          std::uint32_t term_mask,
                          DerivativeArrayView<DerivativeValue> destination)
      : derivative_operator_(derivative_operator), term_mask_(term_mask),
        destination_(destination)
  {}

protected:
  void onBegin(const DerivativeStreamDescriptor& descriptor) override
  {
    if (descriptor.provider_id != derivative_operator_.parameterSchema().providerId() ||
        descriptor.schema_fingerprint != derivative_operator_.parameterSchema().fingerprint() ||
        descriptor.parameter_version != derivative_operator_.parameterVersion() ||
        descriptor.product_mask != derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_JVP) ||
        descriptor.local_energy_term_mask != term_mask_ ||
        descriptor.batch_ordinal != derivative_operator_.batchOrdinal() ||
        descriptor.sample_offset != derivative_operator_.sampleOffset() ||
        descriptor.sample_count != destination_.size())
      throw std::invalid_argument("PII local-energy JVP stream identity mismatch");
    const DerivativeReal sentinel = std::numeric_limits<DerivativeReal>::quiet_NaN();
    std::fill(destination_.begin(), destination_.end(),
              DerivativeValue{sentinel, sentinel});
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
    std::fill(destination_.begin(), destination_.end(),
              DerivativeValue{sentinel, sentinel});
  }

private:
  const StreamingDerivativeOperator& derivative_operator_;
  std::uint32_t term_mask_;
  DerivativeArrayView<DerivativeValue> destination_;
};

/// Accumulate one score VJP into a preallocated rank-local parameter vector.
class ScoreVJPAccumulator final : public ParameterReductionSink
{
public:
  ScoreVJPAccumulator(const StructuredParameterSchema& schema,
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
        descriptor.adjoint != DerivativeAdjoint::HERMITIAN || channels.size() != 1 ||
        channels[0].product != DerivativeProduct::SCORE_VJP)
      throw std::invalid_argument("Score-factor VJP stream identity mismatch");
  }

  void consume(std::size_t channel_ordinal,
               const ParameterChunkConstView& chunk) override
  {
    if (channel_ordinal != 0)
      throw std::logic_error("Score factor received an unexpected VJP channel");
    const std::size_t offset = chunk.descriptor().parameter_offset;
    for (std::size_t index = 0; index < chunk.values().size(); ++index)
      destination_[offset + index] += chunk.values()[index];
  }

private:
  const StructuredParameterSchema& schema_;
  std::size_t parameter_version_;
  DerivativeArrayView<DerivativeValue> destination_;
};

/// Construct coefficient metadata for one exact derivative batch.
CoefficientView coefficientsFor(const StreamingDerivativeOperator& derivative_operator,
                                DerivativeArrayView<const DerivativeValue> values)
{
  return {derivative_operator.parameterSchema().providerId(),
          derivative_operator.parameterSchema().fingerprint(),
          derivative_operator.parameterVersion(), derivative_operator.batchOrdinal(),
          derivative_operator.sampleOffset(), values};
}

/// Validate controls common to minSR and SPRING.
void validateControl(const MinimumStepUpdateControl& control)
{
  if (!isFiniteTrainingReal(control.learning_rate) || control.learning_rate <= 0.0 ||
      !isFiniteTrainingReal(control.damping) || control.damping <= 0.0 ||
      !isFiniteTrainingReal(control.center_shift) || control.center_shift < 0.0 ||
      !isFiniteTrainingReal(control.maximum_update_norm) ||
      control.maximum_update_norm < 0.0 ||
      !isFiniteTrainingReal(control.objective_tolerance) ||
      control.objective_tolerance < 0.0)
    throw std::invalid_argument("Minimum-step controls are invalid");
}

/// Validate bounded work, damping-retry, shift, and trust controls for PII.
void validatePIIControl(const PIIUpdateControl& control)
{
  const KrylovSolverControl& krylov = control.gmres.krylov;
  if (!isFiniteTrainingReal(control.learning_rate) || control.learning_rate <= 0.0 ||
      !isFiniteTrainingReal(control.explicit_shift) ||
      !isFiniteTrainingReal(control.energy_undershoot) || control.energy_undershoot < 0.0 ||
      !isFiniteTrainingReal(control.damping) || control.damping <= 0.0 ||
      !isFiniteTrainingReal(control.damping_multiplier) || control.damping_multiplier <= 1.0 ||
      control.maximum_damping_retries > 64 ||
      !isFiniteTrainingReal(control.maximum_update_norm) || control.maximum_update_norm < 0.0 ||
      !isFiniteTrainingReal(control.objective_tolerance) || control.objective_tolerance < 0.0 ||
      control.gmres.restart_length == 0 ||
      !isFiniteTrainingReal(krylov.relative_tolerance) || krylov.relative_tolerance < 0.0 ||
      !isFiniteTrainingReal(krylov.absolute_tolerance) || krylov.absolute_tolerance < 0.0 ||
      (krylov.relative_tolerance == 0.0 && krylov.absolute_tolerance == 0.0) ||
      !isFiniteTrainingReal(krylov.maximum_seconds) || krylov.maximum_seconds < 0.0 ||
      !isFiniteTrainingReal(krylov.minimum_relative_improvement) ||
      krylov.minimum_relative_improvement < 0.0 ||
      !isFiniteTrainingReal(krylov.curvature_tolerance) ||
      krylov.curvature_tolerance < 0.0 ||
      !isFiniteTrainingReal(krylov.imaginary_tolerance) ||
      krylov.imaginary_tolerance <= 0.0)
    throw std::invalid_argument("PII controls are invalid");
}

/// Resolve the fixed or mean-energy shift for one PII proposal.
DerivativeReal resolvePIIShift(const ScoreCovarianceFactor& factor,
                               const PIIUpdateControl& control)
{
  const DerivativeReal shift = control.shift_mode == PIIShiftMode::EXPLICIT
      ? control.explicit_shift
      : factor.meanEnergy().real() - control.energy_undershoot;
  if (!isFiniteTrainingReal(shift))
    throw MatrixFreeNumericalError("PII shift is non-finite");
  return shift;
}

/// Validate one update request and the score-covariance objective identity.
void validateUpdateInputs(const ScoreCovarianceFactor& factor,
                          bool proposal_live,
                          const StructuredParameterSchema& schema,
                          const StructuredParameterSnapshot& parameters,
                          ParameterGradientView objective,
                          DerivativeReal tolerance)
{
  if (proposal_live)
    throw std::logic_error("Minimum-step optimizer already has a live proposal");
  if (schema.providerId() != factor.parameterSchema().providerId() ||
      schema.fingerprint() != factor.parameterSchema().fingerprint() ||
      parameters.schema_fingerprint != schema.fingerprint() ||
      parameters.version != factor.parameterVersion() ||
      parameters.values.size() != schema.parameterCount() ||
      objective.schema_fingerprint != schema.fingerprint() ||
      objective.parameter_version != parameters.version ||
      objective.gradient.size() != schema.parameterCount())
    throw std::invalid_argument("Minimum-step update identity mismatch");
  if (objective.reduction_domain != ReductionDomain::GLOBAL)
    throw std::invalid_argument("Minimum-step update requires a global objective");
  for (const ParameterBlockDescriptor& block : schema.blocks())
    if (block.scalar_domain != ParameterScalarDomain::REAL64)
      throw std::invalid_argument("Minimum-step updates require REAL64 parameters");
  if (!std::all_of(parameters.values.begin(), parameters.values.end(),
                   isFiniteTrainingReal) ||
      !std::all_of(objective.gradient.begin(), objective.gradient.end(),
                   isFiniteTrainingReal))
    throw std::invalid_argument("Minimum-step update inputs are non-finite");

  const std::vector<DerivativeReal> expected = factor.scoreCovarianceGradient();
  for (std::size_t parameter = 0; parameter < expected.size(); ++parameter)
  {
    const DerivativeReal scale = std::max(
        {DerivativeReal{1}, std::abs(expected[parameter]),
         std::abs(objective.gradient[parameter])});
    if (std::abs(expected[parameter] - objective.gradient[parameter]) >
        tolerance * scale)
      throw std::invalid_argument(
          "Minimum-step update requires the score-covariance residual objective");
  }
}

/// Apply one converged direction to trainable blocks only.
StructuredParameterSnapshot formCandidate(
    const StructuredParameterSchema& schema,
    const StructuredParameterSnapshot& parameters,
    DerivativeArrayView<const DerivativeValue> direction,
    DerivativeReal scale)
{
  StructuredParameterSnapshot candidate = parameters;
  for (const ParameterBlockDescriptor& block : schema.blocks())
    if (block.trainable)
      for (std::size_t parameter = block.offset;
           parameter < block.offset + block.count; ++parameter)
      {
        candidate.values[parameter] -= scale * direction[parameter].real();
        if (!isFiniteTrainingReal(candidate.values[parameter]))
          throw MatrixFreeNumericalError("Minimum-step candidate is non-finite");
      }
  return candidate;
}

/// Extend one FNV-1a fingerprint with raw bytes.
void extendFingerprint(std::uint64_t& hash, const void* data, std::size_t bytes) noexcept
{
  const auto* input = static_cast<const unsigned char*>(data);
  for (std::size_t index = 0; index < bytes; ++index)
  {
    hash ^= input[index];
    hash *= UINT64_C(1099511628211);
  }
}

/// Extend a fingerprint with a length-delimited string.
void extendString(std::uint64_t& hash, const std::string& value) noexcept
{
  const std::uint64_t size = value.size();
  extendFingerprint(hash, &size, sizeof(size));
  extendFingerprint(hash, value.data(), value.size());
}

/// Render one fixed-width hexadecimal fingerprint.
std::string formatFingerprint(std::uint64_t hash)
{
  std::ostringstream text;
  text << std::hex << std::setw(16) << std::setfill('0') << hash;
  return text.str();
}

} // namespace

/// Copied numeric data and reusable scratch for one derivative batch.
ClippedScoreObjectiveResult prepareClippedScoreObjective(
    DerivativeArrayView<const DerivativeReal> energies,
    DerivativeArrayView<const DerivativeReal> weights,
    ClippedScoreObjectiveControl control)
{
  const bool unbounded_exclusion = isPositiveInfinityTrainingReal(control.exclude_width);

  if (energies.empty() || energies.size() != weights.size() || !energies.data() || !weights.data() ||
      !isFiniteTrainingReal(control.width) || control.width <= 0.0 ||
      !isFiniteTrainingReal(control.quantile) || control.quantile <= 0.0 || control.quantile > 1.0 ||
      (!unbounded_exclusion &&
       (!isFiniteTrainingReal(control.exclude_width) || control.exclude_width <= 0.0)))
    throw std::invalid_argument("Clipped score objective has invalid inputs or controls");
  if (control.mode != ScoreObjectiveClipMode::SOFT_MEDIAN &&
      control.mode != ScoreObjectiveClipMode::HARD_MEDIAN &&
      control.mode != ScoreObjectiveClipMode::HARD_MEAN)
    throw std::invalid_argument("Clipped score objective has an unknown transform");
  DerivativeReal weight_sum = 0.0;
  DerivativeReal energy_sum = 0.0;
  for (std::size_t i = 0; i < energies.size(); ++i)
  {
    if (!isFiniteTrainingReal(energies[i]) || !isFiniteTrainingReal(weights[i]) || weights[i] < 0.0)
      throw std::invalid_argument("Clipped score objective has nonfinite energies or invalid weights");
    weight_sum += weights[i];
    energy_sum += energies[i];
  }
  if (!isFiniteTrainingReal(weight_sum) || weight_sum <= 0.0 || !isFiniteTrainingReal(energy_sum))
    throw std::invalid_argument("Clipped score objective has invalid aggregate moments");
  const auto quantile = [](std::vector<DerivativeReal>& sorted, DerivativeReal probability) {
    std::sort(sorted.begin(), sorted.end());
    const DerivativeReal index = (sorted.size() - 1) * probability;
    const auto low = static_cast<std::size_t>(std::floor(index));
    const auto high = static_cast<std::size_t>(std::ceil(index));
    const DerivativeReal fraction = index - low;
    return (1.0 - fraction) * sorted[low] + fraction * sorted[high];
  };
  ClippedScoreObjectiveResult result;
  std::vector<DerivativeReal> scratch(energies.begin(), energies.end());
  result.center = control.mode == ScoreObjectiveClipMode::HARD_MEAN
      ? energy_sum / energies.size() : quantile(scratch, 0.5);
  DerivativeReal absolute_sum = 0.0;
  for (std::size_t i = 0; i < energies.size(); ++i)
  {
    scratch[i] = std::abs(energies[i] - result.center);
    absolute_sum += scratch[i];
  }
  result.scale = control.mode == ScoreObjectiveClipMode::SOFT_MEDIAN
      ? quantile(scratch, control.quantile) : absolute_sum / energies.size();
  const DerivativeReal half_width = control.width * result.scale;
  if (!isFiniteTrainingReal(half_width) ||
      (control.mode == ScoreObjectiveClipMode::SOFT_MEDIAN && half_width <= 0.0))
    throw std::invalid_argument("Clipped score objective has zero or nonfinite soft scale");
  result.coefficients.resize(energies.size());
  std::vector<bool> mask(energies.size());
  for (std::size_t i = 0; i < energies.size(); ++i)
  {
    const DerivativeReal delta = energies[i] - result.center;
    DerivativeReal transformed;
    if (control.mode == ScoreObjectiveClipMode::SOFT_MEDIAN)
    {
      const DerivativeReal z = delta / (2.0 * half_width);
      const DerivativeReal a = std::abs(z);
      const DerivativeReal squeezed = std::copysign(
          std::log1p((a + a * a / 2.0 + a * a * a) / (1.0 + a * a)), z);
      transformed = result.center + 2.0 * half_width * squeezed;
      mask[i] = true;
      if (!unbounded_exclusion)
        mask[i] = std::abs(delta) / result.scale < control.exclude_width;
    }
    else
    {
      transformed = std::clamp(energies[i], result.center - half_width, result.center + half_width);
      mask[i] = true;
      if (!unbounded_exclusion)
        mask[i] = std::abs(delta) < control.exclude_width;
    }
    if (!isFiniteTrainingReal(transformed))
      throw std::invalid_argument("Clipped score objective produced a nonfinite transformed energy");
    result.coefficients[i] = transformed;
    result.transformed_mean += weights[i] * transformed / weight_sum;
    result.unmasked_samples += mask[i];
  }
  if (result.unmasked_samples == 0 || !isFiniteTrainingReal(result.transformed_mean))
    throw std::invalid_argument("Clipped score objective has no unmasked samples or nonfinite mean");
  const DerivativeReal normalization =
      static_cast<DerivativeReal>(energies.size()) / weight_sum / result.unmasked_samples;
  for (std::size_t i = 0; i < energies.size(); ++i)
  {
    result.coefficients[i] = mask[i]
        ? normalization * weights[i] * (result.coefficients[i].real() - result.transformed_mean) : 0.0;
    if (!isFinite(result.coefficients[i]))
      throw std::invalid_argument("Clipped score objective produced a nonfinite residual");
  }
  return result;
}

struct ScoreCovarianceFactor::BatchStorage
{
  const StreamingDerivativeOperator* derivative_operator = nullptr;
  std::size_t sample_vector_offset = 0;
  std::vector<DerivativeReal> weights;
  std::vector<DerivativeValue> local_energies;
  std::vector<DerivativeReal> sqrt_normalized_weights;
  mutable std::vector<DerivativeValue> sample_products;
  mutable std::vector<DerivativeValue> coefficients;
};

ScoreCovarianceFactor::ScoreCovarianceFactor(
    const StructuredParameterSchema& schema,
    std::size_t parameter_version,
    DerivativeArrayView<const ScoreCovarianceBatch> batches,
    ScoreSampleDomain sample_domain,
    DistributedParameterReduction reduction,
    std::optional<ScoreObjectiveResidualOverride> objective_residual)
    : schema_(schema), parameter_version_(parameter_version),
      sample_domain_(sample_domain), reduction_(std::move(reduction)),
      projected_direction_(schema.parameterCount()),
      local_parameter_action_(schema.parameterCount())
{
  if (!batches.empty() && batches.data() == nullptr)
    throw std::invalid_argument("Score covariance factor received a null batch interval");
  for (const ParameterBlockDescriptor& block : schema_.blocks())
    if (block.scalar_domain != ParameterScalarDomain::REAL64)
      throw std::invalid_argument("Score covariance factor requires REAL64 parameters");

  DistributedWeightedSampleMoments local_moments;
  local_moments.reduction_domain = ReductionDomain::RANK_LOCAL;
  std::vector<std::pair<std::size_t, std::size_t>> intervals;
  batches_.reserve(batches.size());
  for (const ScoreCovarianceBatch& input : batches)
  {
    if (!input.derivative_operator)
      throw std::invalid_argument("Score covariance factor received a null derivative operator");
    const StreamingDerivativeOperator& derivative_operator = *input.derivative_operator;
    const StreamingDerivativeCapabilities capabilities = derivative_operator.capabilities();
    if (derivative_operator.parameterSchema().providerId() != schema_.providerId() ||
        derivative_operator.parameterSchema().fingerprint() != schema_.fingerprint() ||
        derivative_operator.parameterVersion() != parameter_version_ ||
        input.weights.size() != derivative_operator.sampleCount() ||
        input.local_energies.size() != derivative_operator.sampleCount())
      throw std::invalid_argument("Score covariance batch identity or extent mismatch");
    if (!capabilities.supports(DerivativeProduct::SCORE_JVP) ||
        !capabilities.supports(DerivativeProduct::SCORE_VJP) ||
        !capabilities.supports(DerivativeAdjoint::HERMITIAN) ||
        capabilities.parameter_scalar_domain != ParameterScalarDomain::REAL64 ||
        capabilities.result_scalar_domain != ParameterScalarDomain::COMPLEX128 ||
        capabilities.execution_domain != DerivativeExecutionDomain::HOST ||
        capabilities.reduction_domain == ReductionDomain::GLOBAL ||
        !capabilities.block_streaming || capabilities.maximum_vjp_channels == 0 ||
        capabilities.maximum_parameter_chunk_size == 0 ||
        capabilities.maximum_sample_tile_size == 0)
      throw std::invalid_argument("Score covariance batch lacks bounded score products");
    if (derivative_operator.sampleOffset() >
        std::numeric_limits<std::size_t>::max() - derivative_operator.sampleCount())
      throw std::overflow_error("Score covariance sample interval overflow");
    intervals.emplace_back(derivative_operator.sampleOffset(),
                           derivative_operator.sampleOffset() + derivative_operator.sampleCount());

    BatchStorage storage;
    storage.derivative_operator = &derivative_operator;
    storage.sample_vector_offset = local_sample_vector_size_;
    storage.weights.assign(input.weights.begin(), input.weights.end());
    storage.local_energies.assign(input.local_energies.begin(),
                                  input.local_energies.end());
    storage.sqrt_normalized_weights.resize(storage.weights.size());
    storage.sample_products.resize(storage.weights.size());
    storage.coefficients.resize(storage.weights.size());
    for (std::size_t sample = 0; sample < storage.weights.size(); ++sample)
    {
      const DerivativeReal weight = storage.weights[sample];
      const DerivativeValue energy = storage.local_energies[sample];
      if (!isFiniteTrainingReal(weight) || weight < 0.0 || !isFinite(energy) ||
          (sample_domain_ == ScoreSampleDomain::REAL && energy.imag() != 0.0))
        throw std::invalid_argument("Score covariance samples are invalid");
      local_moments.weight_sum += weight;
      local_moments.weighted_value_sum += weight * energy;
    }
    if (storage.weights.size() >
        std::numeric_limits<std::size_t>::max() - local_moments.sample_count)
      throw std::overflow_error("Score covariance local sample count overflow");
    local_moments.sample_count += storage.weights.size();
    local_physical_sample_count_ += storage.weights.size();
    const std::size_t multiplier =
        sample_domain_ == ScoreSampleDomain::REAL ? 1 : 2;
    if (storage.weights.size() >
        (std::numeric_limits<std::size_t>::max() - local_sample_vector_size_) /
            multiplier)
      throw std::overflow_error("Score covariance sample-vector extent overflow");
    local_sample_vector_size_ += multiplier * storage.weights.size();
    batches_.push_back(std::move(storage));
  }

  if (objective_residual)
  {
    const auto coefficients = objective_residual->coefficients;
    if (coefficients.size() != local_physical_sample_count_ ||
        (!coefficients.empty() && !coefficients.data()))
      throw std::invalid_argument("Objective residual extent does not match physical samples");
    for (const auto coefficient : coefficients)
      if (!isFinite(coefficient) ||
          (sample_domain_ == ScoreSampleDomain::REAL && coefficient.imag() != 0.0))
        throw std::invalid_argument("Objective residual contains an invalid coefficient");
    objective_residual_.emplace(coefficients.size());
    if (!coefficients.empty())
      std::copy(coefficients.begin(), coefficients.end(), objective_residual_->begin());
  }

  std::sort(intervals.begin(), intervals.end());
  for (std::size_t index = 1; index < intervals.size(); ++index)
    if (intervals[index - 1].first != intervals[index - 1].second &&
        intervals[index].first != intervals[index].second &&
        intervals[index].first < intervals[index - 1].second)
      throw std::invalid_argument("Score covariance local sample intervals overlap");

  const DistributedWeightedSampleMoments global =
      reduction_.reduceWeightedSampleMoments(schema_, parameter_version_, local_moments);
  mean_energy_ = global.weighted_value_sum / global.weight_sum;
  const std::size_t multiplier = sample_domain_ == ScoreSampleDomain::REAL ? 1 : 2;
  if (global.sample_count > std::numeric_limits<std::size_t>::max() / multiplier)
    throw std::overflow_error("Score covariance global sample-vector extent overflow");
  global_sample_vector_size_ = multiplier * global.sample_count;
  residual_.resize(local_sample_vector_size_);

  for (BatchStorage& batch : batches_)
    for (std::size_t sample = 0; sample < batch.weights.size(); ++sample)
    {
      const DerivativeReal sqrt_q = std::sqrt(batch.weights[sample] / global.weight_sum);
      batch.sqrt_normalized_weights[sample] = sqrt_q;
      const DerivativeValue centered_energy = batch.local_energies[sample] - mean_energy_;
      const std::size_t offset = batch.sample_vector_offset + multiplier * sample;
      residual_[offset] = {2.0 * sqrt_q * centered_energy.real(), 0.0};
      if (multiplier == 2)
        residual_[offset + 1] = {2.0 * sqrt_q * centered_energy.imag(), 0.0};
    }
}

ScoreCovarianceFactor::~ScoreCovarianceFactor() = default;

DerivativeArrayView<const DerivativeValue> ScoreCovarianceFactor::residual() const noexcept
{
  return constView(residual_);
}

void ScoreCovarianceFactor::validateParameterDirection(
    const StructuredParameterVectorConstView& direction) const
{
  if (direction.providerId() != schema_.providerId() ||
      direction.schemaFingerprint() != schema_.fingerprint() ||
      direction.parameterVersion() != parameter_version_ ||
      direction.values().size() != schema_.parameterCount())
    throw std::invalid_argument("Score covariance parameter direction identity mismatch");
}

void ScoreCovarianceFactor::validateSampleVector(
    DerivativeArrayView<const DerivativeValue> values,
    const char* description) const
{
  if (values.size() != local_sample_vector_size_ ||
      (!values.empty() && values.data() == nullptr))
    throw std::invalid_argument(std::string(description) + " extent mismatch");
  if (!std::all_of(values.begin(), values.end(), isFinite) ||
      std::any_of(values.begin(), values.end(),
                  [](DerivativeValue value) { return value.imag() != 0.0; }))
    throw std::invalid_argument(std::string(description) + " must contain finite real values");
}

void ScoreCovarianceFactor::apply(
    const StructuredParameterVectorConstView& direction,
    DerivativeArrayView<DerivativeValue> result) const
{
  validateParameterDirection(direction);
  if (result.size() != local_sample_vector_size_ ||
      (!result.empty() && result.data() == nullptr))
    throw std::invalid_argument("Score covariance result extent mismatch");
  std::copy(direction.values().begin(), direction.values().end(),
            projected_direction_.begin());
  for (const ParameterBlockDescriptor& block : schema_.blocks())
    if (!block.trainable)
      std::fill(projected_direction_.begin() + block.offset,
                projected_direction_.begin() + block.offset + block.count,
                DerivativeValue{});
  const StructuredParameterVectorConstView projected(
      schema_, parameter_version_, constView(projected_direction_));

  DistributedWeightedSampleMoments local_moments;
  local_moments.reduction_domain = ReductionDomain::RANK_LOCAL;
  std::exception_ptr failure;
  try
  {
    for (const BatchStorage& batch : batches_)
    {
      ScoreJVPCollector sink(*batch.derivative_operator,
                             mutableView(batch.sample_products));
      batch.derivative_operator->applyScoreJVP(projected, sink);
      if (sample_domain_ == ScoreSampleDomain::REAL &&
          std::any_of(batch.sample_products.begin(), batch.sample_products.end(),
                      [](DerivativeValue value) { return value.imag() != 0.0; }))
        throw MatrixFreeNumericalError("Real score factor produced an imaginary JVP");
      local_moments.sample_count += batch.weights.size();
      for (std::size_t sample = 0; sample < batch.weights.size(); ++sample)
      {
        local_moments.weight_sum += batch.weights[sample];
        local_moments.weighted_value_sum +=
            batch.weights[sample] * batch.sample_products[sample];
      }
    }
  }
  catch (...)
  {
    failure = std::current_exception();
    local_moments = {};
  }
  const DistributedWeightedSampleMoments global =
      reduction_.reduceWeightedSampleMoments(schema_, parameter_version_, local_moments,
                                             failure);
  const DerivativeValue mean = global.weighted_value_sum / global.weight_sum;
  const std::size_t multiplier = sample_domain_ == ScoreSampleDomain::REAL ? 1 : 2;
  for (const BatchStorage& batch : batches_)
    for (std::size_t sample = 0; sample < batch.weights.size(); ++sample)
    {
      const DerivativeValue value =
          batch.sqrt_normalized_weights[sample] *
          (batch.sample_products[sample] - mean);
      const std::size_t offset = batch.sample_vector_offset + multiplier * sample;
      result[offset] = {value.real(), 0.0};
      if (multiplier == 2)
        result[offset + 1] = {value.imag(), 0.0};
    }
}

DerivativeValue ScoreCovarianceFactor::reduceSampleScalar(
    DerivativeValue local_value,
    std::exception_ptr local_failure) const
{
  DistributedWeightedSampleMoments local;
  local.sample_count = local_physical_sample_count_;
  local.weight_sum = static_cast<DerivativeReal>(local_physical_sample_count_);
  local.weighted_value_sum = local_value;
  local.reduction_domain = ReductionDomain::RANK_LOCAL;
  const auto global = reduction_.reduceWeightedSampleMoments(
      schema_, parameter_version_, local, local_failure);
  return global.weighted_value_sum;
}

void ScoreCovarianceFactor::applyAdjoint(
    DerivativeArrayView<const DerivativeValue> direction,
    DerivativeArrayView<DerivativeValue> result) const
{
  validateSampleVector(direction, "Score covariance adjoint direction");
  if (result.size() != schema_.parameterCount() ||
      (!result.empty() && result.data() == nullptr))
    throw std::invalid_argument("Score covariance adjoint result extent mismatch");
  const std::size_t multiplier = sample_domain_ == ScoreSampleDomain::REAL ? 1 : 2;
  DerivativeValue local_center{};
  for (const BatchStorage& batch : batches_)
    for (std::size_t sample = 0; sample < batch.weights.size(); ++sample)
    {
      const std::size_t offset = batch.sample_vector_offset + multiplier * sample;
      const DerivativeValue realified{
          direction[offset].real(), multiplier == 2 ? direction[offset + 1].real() : 0.0};
      local_center += batch.sqrt_normalized_weights[sample] * realified;
    }
  const DerivativeValue center = reduceSampleScalar(local_center);

  std::fill(local_parameter_action_.begin(), local_parameter_action_.end(),
            DerivativeValue{});
  std::exception_ptr failure;
  try
  {
    for (const BatchStorage& batch : batches_)
    {
      for (std::size_t sample = 0; sample < batch.weights.size(); ++sample)
      {
        const std::size_t offset = batch.sample_vector_offset + multiplier * sample;
        const DerivativeValue realified{
            direction[offset].real(), multiplier == 2 ? direction[offset + 1].real() : 0.0};
        const DerivativeReal q = batch.sqrt_normalized_weights[sample] *
            batch.sqrt_normalized_weights[sample];
        batch.coefficients[sample] =
            batch.sqrt_normalized_weights[sample] * realified - q * center;
      }
      const VJPCoefficientChannel channel{
          "minsr/score_adjoint", DerivativeProduct::SCORE_VJP,
          coefficientsFor(*batch.derivative_operator, constView(batch.coefficients)), 0};
      ScoreVJPAccumulator sink(schema_, parameter_version_,
                               mutableView(local_parameter_action_));
      batch.derivative_operator->applyVJPs(
          {&channel, 1}, DerivativeAdjoint::HERMITIAN, sink);
    }
    for (DerivativeValue& value : local_parameter_action_)
      value = {value.real(), 0.0};
    for (const ParameterBlockDescriptor& block : schema_.blocks())
      if (!block.trainable)
        std::fill(local_parameter_action_.begin() + block.offset,
                  local_parameter_action_.begin() + block.offset + block.count,
                  DerivativeValue{});
  }
  catch (...)
  {
    failure = std::current_exception();
    std::fill(local_parameter_action_.begin(), local_parameter_action_.end(),
              DerivativeValue{});
  }
  reduction_.reduceParameterVector(schema_, parameter_version_,
                                   mutableView(local_parameter_action_), failure);
  std::copy(local_parameter_action_.begin(), local_parameter_action_.end(), result.begin());
}

void ScoreCovarianceFactor::addCenterProjector(
    DerivativeArrayView<const DerivativeValue> direction,
    DerivativeReal omega,
    DerivativeArrayView<DerivativeValue> result) const
{
  validateSampleVector(direction, "Score covariance projector direction");
  if (result.size() != local_sample_vector_size_ ||
      (!result.empty() && result.data() == nullptr) ||
      !isFiniteTrainingReal(omega) || omega < 0.0)
    throw std::invalid_argument("Score covariance projector input is invalid");
  if (omega == 0.0)
    return;
  const std::size_t multiplier = sample_domain_ == ScoreSampleDomain::REAL ? 1 : 2;
  for (std::size_t component = 0; component < multiplier; ++component)
  {
    DerivativeValue local_projection{};
    for (const BatchStorage& batch : batches_)
      for (std::size_t sample = 0; sample < batch.weights.size(); ++sample)
      {
        const std::size_t offset = batch.sample_vector_offset + multiplier * sample;
        local_projection += batch.sqrt_normalized_weights[sample] *
            direction[offset + component];
      }
    const DerivativeValue projection = reduceSampleScalar(local_projection);
    for (const BatchStorage& batch : batches_)
      for (std::size_t sample = 0; sample < batch.weights.size(); ++sample)
      {
        const std::size_t offset = batch.sample_vector_offset + multiplier * sample;
        result[offset + component] += omega *
            batch.sqrt_normalized_weights[sample] * projection;
      }
  }
}

std::vector<DerivativeReal> ScoreCovarianceFactor::scoreCovarianceGradient() const
{
  std::vector<DerivativeValue> complex_gradient(schema_.parameterCount());
  if (!objective_residual_)
    applyAdjoint(residual(), mutableView(complex_gradient));
  else
  {
    std::exception_ptr failure;
    try
    {
      std::size_t offset = 0;
      for (const BatchStorage& batch : batches_)
      {
        if (batch.weights.empty())
          continue;
        const VJPCoefficientChannel channel{
            "kfac/objective_residual", DerivativeProduct::SCORE_VJP,
            coefficientsFor(*batch.derivative_operator,
                {objective_residual_->data() + offset, batch.weights.size()}), 0};
        ScoreVJPAccumulator sink(schema_, parameter_version_, mutableView(complex_gradient));
        batch.derivative_operator->applyVJPs({&channel, 1}, DerivativeAdjoint::HERMITIAN, sink);
        offset += batch.weights.size();
      }
      for (DerivativeValue& value : complex_gradient)
        value = {value.real(), 0.0};
      for (const auto& block : schema_.blocks())
        if (!block.trainable)
          std::fill(complex_gradient.begin() + block.offset,
                    complex_gradient.begin() + block.offset + block.count, DerivativeValue{});
    }
    catch (...)
    {
      failure = std::current_exception();
      std::fill(complex_gradient.begin(), complex_gradient.end(), DerivativeValue{});
    }
    reduction_.reduceParameterVector(schema_, parameter_version_, mutableView(complex_gradient), failure);
  }
  std::vector<DerivativeReal> gradient(complex_gradient.size());
  for (std::size_t parameter = 0; parameter < gradient.size(); ++parameter)
    gradient[parameter] = complex_gradient[parameter].real();
  return gradient;
}

ScoreCovarianceStorageDiagnostics ScoreCovarianceFactor::storageDiagnostics() const noexcept
{
  ScoreCovarianceStorageDiagnostics result;
  result.parameter_count = schema_.parameterCount();
  result.local_physical_sample_count = local_physical_sample_count_;
  result.local_sample_vector_size = local_sample_vector_size_;
  result.global_sample_vector_size = global_sample_vector_size_;
  result.complex_parameter_vectors = 2;
  result.complex_sample_values = residual_.capacity() +
      (objective_residual_ ? objective_residual_->capacity() : 0);
  for (const BatchStorage& batch : batches_)
  {
    result.real_sample_values += batch.weights.capacity() +
        batch.sqrt_normalized_weights.capacity();
    result.complex_sample_values += batch.local_energies.capacity() +
        batch.sample_products.capacity() + batch.coefficients.capacity();
  }
  result.retained_numeric_bytes = result.real_sample_values * sizeof(DerivativeReal) +
      result.complex_sample_values * sizeof(DerivativeValue) +
      (projected_direction_.capacity() + local_parameter_action_.capacity()) *
          sizeof(DerivativeValue);
  return result;
}

ProjectedInverseIterationFactor::ProjectedInverseIterationFactor(
    const ScoreCovarianceFactor& score_factor,
    std::uint32_t local_energy_term_mask, HamiltonianSampleProjection projection)
    : score_factor_(score_factor), local_energy_term_mask_(local_energy_term_mask), projection_(projection),
      projected_direction_(score_factor.parameterSchema().parameterCount()),
      score_products_(score_factor.localSampleVectorSize()),
      local_energy_products_(score_factor.localSampleVectorSize())
{
  const std::uint32_t kinetic = localEnergyTermBit(LocalEnergyTerm::KINETIC);
  const std::uint32_t kinetic_and_ecp =
      kinetic | localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
  if (local_energy_term_mask_ != kinetic &&
      local_energy_term_mask_ != kinetic_and_ecp)
    throw std::invalid_argument(
        "PII requires kinetic or kinetic-plus-nonlocal-ECP local-energy derivatives");
  if (score_factor_.sampleDomain() != ScoreSampleDomain::REAL)
    throw std::invalid_argument("PII currently requires a real sample-space factor");
  if (projection_ != HamiltonianSampleProjection::STRICT_REAL &&
      projection_ != HamiltonianSampleProjection::REAL_TANGENT)
    throw std::invalid_argument("Unknown Hamiltonian sample projection");

  for (const ScoreCovarianceFactor::BatchStorage& batch : score_factor_.batches_)
  {
    const StreamingDerivativeCapabilities capabilities =
        batch.derivative_operator->capabilities();
    if (!capabilities.supports(DerivativeProduct::LOCAL_ENERGY_JVP) ||
        !capabilities.coversLocalEnergyTerms(local_energy_term_mask_) ||
        capabilities.execution_domain != DerivativeExecutionDomain::HOST ||
        capabilities.reduction_domain == ReductionDomain::GLOBAL ||
        !capabilities.block_streaming || capabilities.maximum_vjp_channels == 0 ||
        capabilities.maximum_parameter_chunk_size == 0 ||
        capabilities.maximum_sample_tile_size == 0)
      throw std::invalid_argument(
          "PII derivative batch lacks the requested bounded local-energy products");
  }
}

std::size_t ProjectedInverseIterationFactor::retainedBytes() const noexcept
{
  return (projected_direction_.capacity() + score_products_.capacity() +
          local_energy_products_.capacity()) * sizeof(DerivativeValue);
}

void ProjectedInverseIterationFactor::applyShifted(
    const StructuredParameterVectorConstView& direction,
    DerivativeReal tau,
    DerivativeArrayView<DerivativeValue> result) const
{
  score_factor_.validateParameterDirection(direction);
  if (!isFiniteTrainingReal(tau) || result.size() != score_products_.size() ||
      (!result.empty() && result.data() == nullptr))
    throw std::invalid_argument("PII shifted-factor input is invalid");

  std::copy(direction.values().begin(), direction.values().end(),
            projected_direction_.begin());
  for (const ParameterBlockDescriptor& block : score_factor_.schema_.blocks())
    if (!block.trainable)
      std::fill(projected_direction_.begin() + block.offset,
                projected_direction_.begin() + block.offset + block.count,
                DerivativeValue{});
  const StructuredParameterVectorConstView projected(
      score_factor_.schema_, score_factor_.parameter_version_,
      constView(projected_direction_));

  std::exception_ptr failure;
  try
  {
    for (const ScoreCovarianceFactor::BatchStorage& batch : score_factor_.batches_)
    {
      DerivativeArrayView<DerivativeValue> batch_local_energy{
          batch.sample_products.empty()
              ? nullptr
              : local_energy_products_.data() + batch.sample_vector_offset,
          batch.sample_products.size()};
      LocalEnergyJVPCollector sink(*batch.derivative_operator,
                                   local_energy_term_mask_,
                                   batch_local_energy);
      batch.derivative_operator->applyLocalEnergyJVP(
          projected, local_energy_term_mask_, sink);
      if (std::any_of(batch_local_energy.begin(), batch_local_energy.end(),
                      [this](DerivativeValue value) {
                        return !isFinite(value) ||
                            (projection_ == HamiltonianSampleProjection::STRICT_REAL && value.imag() != 0.0);
                      }))
        throw MatrixFreeNumericalError(
            "Hamiltonian factor produced a non-finite or inadmissible complex local-energy JVP");
    }
  }
  catch (...)
  {
    failure = std::current_exception();
  }
  // Reach a collective agreement on local producer failure before applying O,
  // whose score centering itself performs collectives.
  score_factor_.reduceSampleScalar({}, failure);

  score_factor_.apply(projected, mutableView(score_products_));
  for (const ScoreCovarianceFactor::BatchStorage& batch : score_factor_.batches_)
    for (std::size_t sample = 0; sample < batch.weights.size(); ++sample)
    {
      const std::size_t index = batch.sample_vector_offset + sample;
      // REAL_TANGENT restricts the Hamiltonian bilinear form to real scores and
      // coefficients: Re(D^dagger H D). The physical derivative remains complex;
      // its imaginary part is admitted/checked above, not removed at the source.
      const DerivativeReal local_energy_jvp =
          batch.sqrt_normalized_weights[sample] * local_energy_products_[index].real();
      result[index] = {local_energy_jvp +
          (batch.local_energies[sample].real() - tau) * score_products_[index].real(),
          0.0};
    }
}

ProjectedInverseIterationOperator::ProjectedInverseIterationOperator(
    const ProjectedInverseIterationFactor& factor,
    DerivativeReal tau,
    DerivativeReal damping)
    : MatrixFreeLinearOperator(factor.scoreFactor().parameterSchema(),
                               factor.scoreFactor().parameterVersion(),
                               ReductionDomain::GLOBAL, false),
      factor_(factor), tau_(tau), damping_(damping),
      sample_scratch_(factor.scoreFactor().localSampleVectorSize())
{
  if (!isFiniteTrainingReal(tau_) || !isFiniteTrainingReal(damping_) || damping_ <= 0.0)
    throw std::invalid_argument("Full PII shift or damping is invalid");
}

void ProjectedInverseIterationOperator::evaluate(
    const StructuredParameterVectorConstView& direction,
    DerivativeArrayView<DerivativeValue> result) const
{
  factor_.applyShifted(direction, tau_, mutableView(sample_scratch_));
  factor_.scoreFactor().applyAdjoint(constView(sample_scratch_), result);
  for (std::size_t parameter = 0; parameter < result.size(); ++parameter)
    result[parameter] += damping_ * direction.values()[parameter];
}

DistributedSampleLinearOperator::DistributedSampleLinearOperator(
    const ScoreCovarianceFactor& factor,
    bool hermitian)
    : factor_(factor),
      descriptor_{factor.parameterSchema().providerId(),
                  factor.parameterSchema().fingerprint(), factor.parameterVersion(),
                  factor.localSampleVectorSize(), factor.globalSampleVectorSize(), hermitian}
{}

void DistributedSampleLinearOperator::apply(
    DerivativeArrayView<const DerivativeValue> direction,
    DerivativeArrayView<DerivativeValue> result) const
{
  if (direction.size() != descriptor_.local_size || result.size() != descriptor_.local_size ||
      (!direction.empty() && (!direction.data() || !result.data())) ||
      (!direction.empty() && direction.data() == result.data()))
    throw std::invalid_argument("Distributed sample action extent or alias mismatch");
  if (!std::all_of(direction.begin(), direction.end(), isFinite))
    throw std::invalid_argument("Distributed sample direction is non-finite");
  const DerivativeReal sentinel = std::numeric_limits<DerivativeReal>::quiet_NaN();
  std::fill(result.begin(), result.end(), DerivativeValue{sentinel, sentinel});
  evaluate(direction, result);
  if (!std::all_of(result.begin(), result.end(), isFinite))
    throw MatrixFreeNumericalError("Distributed sample action is non-finite");
}

DerivativeValue DistributedSampleLinearOperator::hermitianDot(
    DerivativeArrayView<const DerivativeValue> left,
    DerivativeArrayView<const DerivativeValue> right) const
{
  if (left.size() != descriptor_.local_size || right.size() != descriptor_.local_size)
    throw std::invalid_argument("Distributed sample dot-product extent mismatch");
  DerivativeValue local{};
  for (std::size_t index = 0; index < left.size(); ++index)
    local += std::conj(left[index]) * right[index];
  return factor_.reduceSampleScalar(local);
}

DerivativeReal DistributedSampleLinearOperator::norm(
    DerivativeArrayView<const DerivativeValue> values) const
{
  const DerivativeValue squared = hermitianDot(values, values);
  const DerivativeReal tolerance = 128.0 * std::numeric_limits<DerivativeReal>::epsilon() *
      static_cast<DerivativeReal>(std::max<std::size_t>(descriptor_.global_size, 1)) *
      std::max(DerivativeReal{1}, std::abs(squared.real()));
  if (std::abs(squared.imag()) > tolerance || squared.real() < -tolerance)
    throw MatrixFreeNumericalError("Distributed sample norm is not real and nonnegative");
  return std::sqrt(std::max(DerivativeReal{}, squared.real()));
}

MinimumStepSampleOperator::MinimumStepSampleOperator(
    const ScoreCovarianceFactor& factor,
    DerivativeReal damping,
    DerivativeReal center_shift)
    : DistributedSampleLinearOperator(factor, true), damping_(damping),
      center_shift_(center_shift),
      parameter_scratch_(factor.parameterSchema().parameterCount())
{
  if (!isFiniteTrainingReal(damping_) || damping_ <= 0.0 ||
      !isFiniteTrainingReal(center_shift_) || center_shift_ < 0.0)
    throw std::invalid_argument("Minimum-step sample shifts are invalid");
}

void MinimumStepSampleOperator::evaluate(
    DerivativeArrayView<const DerivativeValue> direction,
    DerivativeArrayView<DerivativeValue> result) const
{
  factor().applyAdjoint(direction, mutableView(parameter_scratch_));
  const StructuredParameterVectorConstView parameter_direction(
      factor().parameterSchema(), factor().parameterVersion(),
      constView(parameter_scratch_));
  factor().apply(parameter_direction, result);
  for (std::size_t index = 0; index < result.size(); ++index)
    result[index] += damping_ * direction[index];
  factor().addCenterProjector(direction, center_shift_, result);
}

MinimumProjectedInverseIterationOperator::MinimumProjectedInverseIterationOperator(
    const ProjectedInverseIterationFactor& factor,
    DerivativeReal tau,
    DerivativeReal damping)
    : DistributedSampleLinearOperator(factor.scoreFactor(), false),
      pii_factor_(factor), tau_(tau), damping_(damping),
      parameter_scratch_(factor.scoreFactor().parameterSchema().parameterCount())
{
  if (!isFiniteTrainingReal(tau_) || !isFiniteTrainingReal(damping_) || damping_ <= 0.0)
    throw std::invalid_argument("MinPII shift or damping is invalid");
}

void MinimumProjectedInverseIterationOperator::evaluate(
    DerivativeArrayView<const DerivativeValue> direction,
    DerivativeArrayView<DerivativeValue> result) const
{
  // The ordered product is B O^T, not O B^T: first lift the distributed
  // sample vector through the score transpose, then apply the shifted PII factor.
  factor().applyAdjoint(direction, mutableView(parameter_scratch_));
  const StructuredParameterVectorConstView parameter_direction(
      factor().parameterSchema(), factor().parameterVersion(),
      constView(parameter_scratch_));
  pii_factor_.applyShifted(parameter_direction, tau_, result);
  for (std::size_t sample = 0; sample < result.size(); ++sample)
    result[sample] += damping_ * direction[sample];
}

DistributedSampleKrylovResult solveDistributedSamplePCG(
    const DistributedSampleLinearOperator& linear_operator,
    DerivativeArrayView<const DerivativeValue> right_hand_side,
    const KrylovSolverControl& control,
    DerivativeArrayView<const DerivativeValue> warm_start)
{
  if (!linear_operator.descriptor().hermitian)
    throw std::invalid_argument("Distributed sample PCG requires a Hermitian action");
  if (right_hand_side.size() != linear_operator.descriptor().local_size ||
      (!warm_start.empty() && warm_start.size() != right_hand_side.size()))
    throw std::invalid_argument("Distributed sample PCG vector extent mismatch");
  if ((!isFiniteTrainingReal(control.relative_tolerance) || control.relative_tolerance < 0.0) ||
      (!isFiniteTrainingReal(control.absolute_tolerance) || control.absolute_tolerance < 0.0) ||
      (control.relative_tolerance == 0.0 && control.absolute_tolerance == 0.0) ||
      !isFiniteTrainingReal(control.maximum_seconds) || control.maximum_seconds < 0.0 ||
      !isFiniteTrainingReal(control.minimum_relative_improvement) ||
      control.minimum_relative_improvement < 0.0 ||
      !isFiniteTrainingReal(control.curvature_tolerance) ||
      control.curvature_tolerance < 0.0)
    throw std::invalid_argument("Distributed sample PCG controls are invalid");

  const std::size_t size = right_hand_side.size();
  DistributedSampleKrylovResult result;
  result.solution.assign(size, DerivativeValue{});
  std::vector<DerivativeValue> residual(right_hand_side.begin(), right_hand_side.end());
  std::vector<DerivativeValue> search_direction(size);
  std::vector<DerivativeValue> action(size);
  const Clock::time_point start = Clock::now();

  auto finish = [&](KrylovStopReason reason) {
    result.stop_reason = reason;
    result.converged = reason == KrylovStopReason::CONVERGED ||
        reason == KrylovStopReason::ZERO_RIGHT_HAND_SIDE;
    result.relative_residual_norm = result.initial_residual_norm == 0.0
        ? 0.0
        : result.final_residual_norm / result.initial_residual_norm;
    result.elapsed_seconds = elapsedSeconds(start);
  };

  try
  {
    const DerivativeReal rhs_norm = linear_operator.norm(right_hand_side);
    result.initial_residual_norm = rhs_norm;
    result.final_residual_norm = rhs_norm;
    if (rhs_norm == 0.0)
    {
      finish(KrylovStopReason::ZERO_RIGHT_HAND_SIDE);
      return result;
    }
    if (!warm_start.empty())
    {
      std::copy(warm_start.begin(), warm_start.end(), result.solution.begin());
      if (control.maximum_operator_applications == 0)
      {
        finish(KrylovStopReason::OPERATOR_APPLICATION_LIMIT);
        return result;
      }
      linear_operator.apply(constView(result.solution), mutableView(action));
      ++result.operator_applications;
      for (std::size_t index = 0; index < size; ++index)
        residual[index] -= action[index];
      result.final_residual_norm = linear_operator.norm(constView(residual));
    }
    const DerivativeReal target = std::max(
        control.absolute_tolerance, control.relative_tolerance * rhs_norm);
    if (result.final_residual_norm <= target)
    {
      finish(KrylovStopReason::CONVERGED);
      return result;
    }

    std::copy(residual.begin(), residual.end(), search_direction.begin());
    DerivativeValue rho_value = linear_operator.hermitianDot(
        constView(residual), constView(residual));
    DerivativeReal rho = rho_value.real();
    DerivativeReal best_residual = result.final_residual_norm;
    std::size_t stagnant = 0;
    while (result.iterations < control.maximum_iterations)
    {
      if (result.operator_applications >= control.maximum_operator_applications)
      {
        finish(KrylovStopReason::OPERATOR_APPLICATION_LIMIT);
        return result;
      }
      if (control.maximum_seconds > 0.0 && elapsedSeconds(start) >= control.maximum_seconds)
      {
        finish(KrylovStopReason::TIME_LIMIT);
        return result;
      }
      linear_operator.apply(constView(search_direction), mutableView(action));
      ++result.operator_applications;
      const DerivativeValue curvature_value = linear_operator.hermitianDot(
          constView(search_direction), constView(action));
      const DerivativeReal curvature = curvature_value.real();
      result.last_curvature = curvature;
      if (!isFinite(curvature_value) ||
          std::abs(curvature_value.imag()) > control.imaginary_tolerance *
              std::numeric_limits<DerivativeReal>::epsilon() *
              std::max(DerivativeReal{1}, std::abs(curvature)) ||
          curvature <= control.curvature_tolerance)
      {
        finish(KrylovStopReason::NON_POSITIVE_CURVATURE);
        return result;
      }
      const DerivativeReal alpha = rho / curvature;
      if (!isFiniteTrainingReal(alpha))
      {
        finish(KrylovStopReason::BREAKDOWN);
        return result;
      }
      for (std::size_t index = 0; index < size; ++index)
      {
        result.solution[index] += alpha * search_direction[index];
        residual[index] -= alpha * action[index];
      }
      ++result.iterations;
      result.final_residual_norm = linear_operator.norm(constView(residual));
      if (result.final_residual_norm <= target)
      {
        finish(KrylovStopReason::CONVERGED);
        return result;
      }
      if (control.stagnation_window > 0)
      {
        if (result.final_residual_norm <
            best_residual * (1.0 - control.minimum_relative_improvement))
        {
          best_residual = result.final_residual_norm;
          stagnant = 0;
        }
        else if (++stagnant >= control.stagnation_window)
        {
          finish(KrylovStopReason::STAGNATED);
          return result;
        }
      }
      const DerivativeValue next_rho_value = linear_operator.hermitianDot(
          constView(residual), constView(residual));
      const DerivativeReal next_rho = next_rho_value.real();
      if (!(rho > 0.0) || !isFiniteTrainingReal(next_rho) || next_rho < 0.0)
      {
        finish(KrylovStopReason::BREAKDOWN);
        return result;
      }
      const DerivativeReal beta = next_rho / rho;
      for (std::size_t index = 0; index < size; ++index)
        search_direction[index] = residual[index] + beta * search_direction[index];
      rho = next_rho;
    }
    finish(KrylovStopReason::ITERATION_LIMIT);
    return result;
  }
  catch (const MatrixFreeNumericalError&)
  {
    finish(KrylovStopReason::NONFINITE_RESULT);
    return result;
  }
}

DistributedSampleKrylovResult solveDistributedSampleGMRES(
    const DistributedSampleLinearOperator& linear_operator,
    DerivativeArrayView<const DerivativeValue> right_hand_side,
    const GMRESSolverControl& control,
    DerivativeArrayView<const DerivativeValue> warm_start)
{
  const KrylovSolverControl& krylov = control.krylov;
  if (right_hand_side.size() != linear_operator.descriptor().local_size ||
      (!right_hand_side.empty() && right_hand_side.data() == nullptr) ||
      (!warm_start.empty() &&
       (warm_start.size() != right_hand_side.size() || warm_start.data() == nullptr)))
    throw std::invalid_argument("Distributed sample GMRES vector extent mismatch");
  if (control.restart_length == 0 ||
      !isFiniteTrainingReal(krylov.relative_tolerance) || krylov.relative_tolerance < 0.0 ||
      !isFiniteTrainingReal(krylov.absolute_tolerance) || krylov.absolute_tolerance < 0.0 ||
      (krylov.relative_tolerance == 0.0 && krylov.absolute_tolerance == 0.0) ||
      !isFiniteTrainingReal(krylov.maximum_seconds) || krylov.maximum_seconds < 0.0 ||
      !isFiniteTrainingReal(krylov.minimum_relative_improvement) ||
      krylov.minimum_relative_improvement < 0.0 ||
      !isFiniteTrainingReal(krylov.curvature_tolerance) ||
      krylov.curvature_tolerance < 0.0 ||
      !isFiniteTrainingReal(krylov.imaginary_tolerance) ||
      krylov.imaginary_tolerance <= 0.0)
    throw std::invalid_argument("Distributed sample GMRES controls are invalid");

  const std::size_t local_size = right_hand_side.size();
  const std::size_t global_size = linear_operator.descriptor().global_size;
  const std::size_t restart_length = std::min(control.restart_length, global_size);
  if (restart_length == 0 || restart_length == std::numeric_limits<std::size_t>::max() ||
      (local_size != 0 && restart_length + 1 >
           std::numeric_limits<std::size_t>::max() / local_size) ||
      restart_length > std::numeric_limits<std::size_t>::max() / (restart_length + 1))
    throw std::length_error("Distributed sample GMRES workspace extent overflows size_t");

  DistributedSampleKrylovResult result;
  result.solution.assign(local_size, DerivativeValue{});
  std::vector<DerivativeValue> residual(right_hand_side.begin(), right_hand_side.end());
  std::vector<DerivativeValue> action(local_size);
  std::vector<DerivativeValue> basis((restart_length + 1) * local_size);
  std::vector<DerivativeValue> hessenberg((restart_length + 1) * restart_length);
  std::vector<DerivativeReal> rotations_cosine(restart_length);
  std::vector<DerivativeValue> rotations_sine(restart_length);
  std::vector<DerivativeValue> least_squares_rhs(restart_length + 1);
  std::vector<DerivativeValue> coefficients(restart_length);

  auto basisView = [&](std::size_t column) {
    return DerivativeArrayView<DerivativeValue>{
        local_size == 0 ? nullptr : basis.data() + column * local_size, local_size};
  };
  auto constBasisView = [&](std::size_t column) {
    return DerivativeArrayView<const DerivativeValue>{
        local_size == 0 ? nullptr : basis.data() + column * local_size, local_size};
  };
  auto hessenbergAt = [&](std::size_t row, std::size_t column) -> DerivativeValue& {
    return hessenberg[column * (restart_length + 1) + row];
  };
  const Clock::time_point start = Clock::now();
  auto finish = [&](KrylovStopReason reason) {
    result.stop_reason = reason;
    result.converged = reason == KrylovStopReason::CONVERGED ||
        reason == KrylovStopReason::ZERO_RIGHT_HAND_SIDE;
    result.relative_residual_norm = result.initial_residual_norm == 0.0
        ? 0.0 : result.final_residual_norm / result.initial_residual_norm;
    result.elapsed_seconds = elapsedSeconds(start);
  };

  try
  {
    const DerivativeReal rhs_norm = linear_operator.norm(right_hand_side);
    result.initial_residual_norm = rhs_norm;
    result.final_residual_norm = rhs_norm;
    if (rhs_norm == 0.0)
    {
      finish(KrylovStopReason::ZERO_RIGHT_HAND_SIDE);
      return result;
    }
    if (!warm_start.empty())
    {
      std::copy(warm_start.begin(), warm_start.end(), result.solution.begin());
      if (krylov.maximum_operator_applications == 0)
      {
        finish(KrylovStopReason::OPERATOR_APPLICATION_LIMIT);
        return result;
      }
      linear_operator.apply(constView(result.solution), mutableView(action));
      ++result.operator_applications;
      for (std::size_t index = 0; index < local_size; ++index)
        residual[index] -= action[index];
      result.initial_residual_norm = linear_operator.norm(constView(residual));
      result.final_residual_norm = result.initial_residual_norm;
    }
    const DerivativeReal target = std::max(
        krylov.absolute_tolerance, krylov.relative_tolerance * rhs_norm);
    if (result.final_residual_norm <= target)
    {
      finish(KrylovStopReason::CONVERGED);
      return result;
    }
    if (krylov.maximum_iterations == 0)
    {
      finish(KrylovStopReason::ITERATION_LIMIT);
      return result;
    }

    DerivativeReal best_residual = result.final_residual_norm;
    std::size_t stagnant_iterations = 0;
    while (true)
    {
      std::fill(hessenberg.begin(), hessenberg.end(), DerivativeValue{});
      std::fill(rotations_cosine.begin(), rotations_cosine.end(), DerivativeReal{});
      std::fill(rotations_sine.begin(), rotations_sine.end(), DerivativeValue{});
      std::fill(least_squares_rhs.begin(), least_squares_rhs.end(), DerivativeValue{});
      std::fill(coefficients.begin(), coefficients.end(), DerivativeValue{});

      const DerivativeReal cycle_residual = result.final_residual_norm;
      auto first_basis = basisView(0);
      for (std::size_t index = 0; index < local_size; ++index)
        first_basis[index] = residual[index] / cycle_residual;
      least_squares_rhs[0] = {cycle_residual, 0.0};

      KrylovStopReason pending_reason = KrylovStopReason::BREAKDOWN;
      bool have_pending_reason = false;
      std::size_t arnoldi_columns = 0;
      for (std::size_t column = 0; column < restart_length; ++column)
      {
        if (result.operator_applications >= krylov.maximum_operator_applications)
        {
          pending_reason = KrylovStopReason::OPERATOR_APPLICATION_LIMIT;
          have_pending_reason = true;
          break;
        }
        if (krylov.maximum_seconds > 0.0 && elapsedSeconds(start) >= krylov.maximum_seconds)
        {
          pending_reason = KrylovStopReason::TIME_LIMIT;
          have_pending_reason = true;
          break;
        }
        linear_operator.apply(constBasisView(column), mutableView(action));
        ++result.operator_applications;

        for (int pass = 0; pass < 2; ++pass)
          for (std::size_t row = 0; row <= column; ++row)
          {
            const DerivativeValue projection = linear_operator.hermitianDot(
                constBasisView(row), constView(action));
            hessenbergAt(row, column) += projection;
            for (std::size_t index = 0; index < local_size; ++index)
              action[index] -= projection * constBasisView(row)[index];
          }
        const DerivativeReal next_norm = linear_operator.norm(constView(action));
        hessenbergAt(column + 1, column) = {next_norm, 0.0};
        if (next_norm > 0.0)
        {
          auto next_basis = basisView(column + 1);
          for (std::size_t index = 0; index < local_size; ++index)
            next_basis[index] = action[index] / next_norm;
        }

        for (std::size_t rotation = 0; rotation < column; ++rotation)
        {
          const DerivativeValue upper = hessenbergAt(rotation, column);
          const DerivativeValue lower = hessenbergAt(rotation + 1, column);
          hessenbergAt(rotation, column) = rotations_cosine[rotation] * upper +
              rotations_sine[rotation] * lower;
          hessenbergAt(rotation + 1, column) =
              -std::conj(rotations_sine[rotation]) * upper +
              rotations_cosine[rotation] * lower;
        }
        const DerivativeValue diagonal = hessenbergAt(column, column);
        const DerivativeValue subdiagonal = hessenbergAt(column + 1, column);
        const DerivativeReal diagonal_norm = std::abs(diagonal);
        const DerivativeReal subdiagonal_norm = std::abs(subdiagonal);
        const DerivativeReal rotation_norm = std::hypot(diagonal_norm, subdiagonal_norm);
        if (rotation_norm == 0.0)
        {
          pending_reason = KrylovStopReason::BREAKDOWN;
          have_pending_reason = true;
          break;
        }
        if (diagonal_norm == 0.0)
        {
          rotations_cosine[column] = 0.0;
          rotations_sine[column] = std::conj(subdiagonal) / subdiagonal_norm;
          hessenbergAt(column, column) = {subdiagonal_norm, 0.0};
        }
        else
        {
          const DerivativeValue phase = diagonal / diagonal_norm;
          rotations_cosine[column] = diagonal_norm / rotation_norm;
          rotations_sine[column] = phase * std::conj(subdiagonal) / rotation_norm;
          hessenbergAt(column, column) = phase * rotation_norm;
        }
        hessenbergAt(column + 1, column) = {};
        const DerivativeValue current_rhs = least_squares_rhs[column];
        least_squares_rhs[column] = rotations_cosine[column] * current_rhs;
        least_squares_rhs[column + 1] =
            -std::conj(rotations_sine[column]) * current_rhs;
        ++result.iterations;
        arnoldi_columns = column + 1;
        result.final_residual_norm = std::abs(least_squares_rhs[column + 1]);

        if (result.final_residual_norm <= target)
        {
          pending_reason = KrylovStopReason::CONVERGED;
          have_pending_reason = true;
          break;
        }
        const DerivativeReal required_improvement =
            krylov.minimum_relative_improvement * best_residual;
        if (best_residual - result.final_residual_norm > required_improvement)
        {
          best_residual = result.final_residual_norm;
          stagnant_iterations = 0;
        }
        else if (krylov.stagnation_window > 0 &&
                 ++stagnant_iterations >= krylov.stagnation_window)
        {
          pending_reason = KrylovStopReason::STAGNATED;
          have_pending_reason = true;
          break;
        }
        if (next_norm == 0.0)
        {
          pending_reason = KrylovStopReason::BREAKDOWN;
          have_pending_reason = true;
          break;
        }
        if (result.iterations >= krylov.maximum_iterations)
        {
          pending_reason = KrylovStopReason::ITERATION_LIMIT;
          have_pending_reason = true;
          break;
        }
      }

      if (arnoldi_columns == 0)
      {
        finish(have_pending_reason ? pending_reason : KrylovStopReason::BREAKDOWN);
        return result;
      }
      for (std::size_t reverse = arnoldi_columns; reverse > 0; --reverse)
      {
        const std::size_t row = reverse - 1;
        DerivativeValue value = least_squares_rhs[row];
        for (std::size_t column = row + 1; column < arnoldi_columns; ++column)
          value -= hessenbergAt(row, column) * coefficients[column];
        const DerivativeValue diagonal = hessenbergAt(row, row);
        const DerivativeReal threshold = std::numeric_limits<DerivativeReal>::epsilon() *
            std::max(DerivativeReal{1}, std::abs(value));
        if (std::abs(diagonal) <= threshold)
        {
          finish(KrylovStopReason::BREAKDOWN);
          return result;
        }
        coefficients[row] = value / diagonal;
      }
      for (std::size_t column = 0; column < arnoldi_columns; ++column)
        for (std::size_t index = 0; index < local_size; ++index)
          result.solution[index] += coefficients[column] * constBasisView(column)[index];

      if (have_pending_reason)
      {
        finish(pending_reason);
        return result;
      }
      if (result.iterations >= krylov.maximum_iterations)
      {
        finish(KrylovStopReason::ITERATION_LIMIT);
        return result;
      }
      if (result.operator_applications >= krylov.maximum_operator_applications)
      {
        finish(KrylovStopReason::OPERATOR_APPLICATION_LIMIT);
        return result;
      }
      if (krylov.maximum_seconds > 0.0 && elapsedSeconds(start) >= krylov.maximum_seconds)
      {
        finish(KrylovStopReason::TIME_LIMIT);
        return result;
      }

      linear_operator.apply(constView(result.solution), mutableView(action));
      ++result.operator_applications;
      std::copy(right_hand_side.begin(), right_hand_side.end(), residual.begin());
      for (std::size_t index = 0; index < local_size; ++index)
        residual[index] -= action[index];
      result.final_residual_norm = linear_operator.norm(constView(residual));
      if (result.final_residual_norm <= target)
      {
        finish(KrylovStopReason::CONVERGED);
        return result;
      }
    }
  }
  catch (const MatrixFreeNumericalError&)
  {
    finish(KrylovStopReason::NONFINITE_RESULT);
    return result;
  }
}

ProjectedInverseIterationUpdateRule::ProjectedInverseIterationUpdateRule(
    const ProjectedInverseIterationFactor& factor,
    PIIUpdateControl control)
    : factor_(factor), control_(std::move(control))
{
  validatePIIControl(control_);
}

StructuredParameterSnapshot ProjectedInverseIterationUpdateRule::propose(
    const StructuredParameterSchema& schema,
    const StructuredParameterSnapshot& parameters,
    ParameterGradientView objective)
{
  validateUpdateInputs(factor_.scoreFactor(), proposal_live_, schema, parameters,
                       objective, control_.objective_tolerance);
  last_diagnostics_.reset();
  const DerivativeReal tau = resolvePIIShift(factor_.scoreFactor(), control_);
  std::vector<DerivativeValue> right_hand_side(schema.parameterCount());
  for (std::size_t parameter = 0; parameter < right_hand_side.size(); ++parameter)
    right_hand_side[parameter] = {0.5 * objective.gradient[parameter], 0.0};
  const StructuredParameterVectorConstView rhs_view(
      schema, parameters.version, constView(right_hand_side));

  std::vector<DerivativeValue> direction;
  DerivativeReal damping = control_.damping;
  for (std::size_t attempt = 0; attempt <= control_.maximum_damping_retries; ++attempt)
  {
    ProjectedInverseIterationOperator linear_operator(factor_, tau, damping);
    KrylovSolveResult solve = solveRestartedGMRES(
        linear_operator, rhs_view, control_.gmres);
    last_diagnostics_ = PIIUpdateDiagnostics{
        false, tau, damping, attempt + 1, 0.0, 0.0, solve.stop_reason,
        solve.iterations, solve.operator_applications, solve.final_residual_norm};
    if (solve.converged)
    {
      direction = std::move(solve.solution);
      break;
    }
    if (attempt == control_.maximum_damping_retries)
      throw std::runtime_error(std::string("PII GMRES did not converge: ") +
                               krylovStopReasonName(solve.stop_reason));
    damping *= control_.damping_multiplier;
    if (!isFiniteTrainingReal(damping))
      throw MatrixFreeNumericalError("PII damping retry overflowed");
  }

  const DerivativeReal norm = parameterVectorNorm(schema, constView(direction));
  DerivativeReal scale = control_.learning_rate;
  if (control_.maximum_update_norm > 0.0 && norm > 0.0)
    scale = std::min(scale, control_.maximum_update_norm / norm);
  last_diagnostics_->direction_norm = norm;
  last_diagnostics_->applied_scale = scale;
  StructuredParameterSnapshot candidate =
      formCandidate(schema, parameters, constView(direction), scale);
  proposal_live_ = true;
  return candidate;
}

void ProjectedInverseIterationUpdateRule::proposalAccepted(
    const StructuredParameterSchema&,
    const StructuredParameterSnapshot&,
    ParameterGradientView) noexcept
{
  proposal_live_ = false;
}

void ProjectedInverseIterationUpdateRule::proposalRejected() noexcept
{
  proposal_live_ = false;
}

MinimumProjectedInverseIterationUpdateRule::MinimumProjectedInverseIterationUpdateRule(
    const ProjectedInverseIterationFactor& factor,
    PIIUpdateControl control)
    : factor_(factor), control_(std::move(control))
{
  validatePIIControl(control_);
}

StructuredParameterSnapshot MinimumProjectedInverseIterationUpdateRule::propose(
    const StructuredParameterSchema& schema,
    const StructuredParameterSnapshot& parameters,
    ParameterGradientView objective)
{
  validateUpdateInputs(factor_.scoreFactor(), proposal_live_, schema, parameters,
                       objective, control_.objective_tolerance);
  last_diagnostics_.reset();
  const DerivativeReal tau = resolvePIIShift(factor_.scoreFactor(), control_);
  std::vector<DerivativeValue> right_hand_side(
      factor_.scoreFactor().residual().begin(), factor_.scoreFactor().residual().end());
  for (DerivativeValue& value : right_hand_side)
    value *= 0.5;

  std::vector<DerivativeValue> sample_solution;
  DerivativeReal damping = control_.damping;
  for (std::size_t attempt = 0; attempt <= control_.maximum_damping_retries; ++attempt)
  {
    MinimumProjectedInverseIterationOperator linear_operator(factor_, tau, damping);
    DistributedSampleKrylovResult solve = solveDistributedSampleGMRES(
        linear_operator, constView(right_hand_side), control_.gmres);
    last_diagnostics_ = PIIUpdateDiagnostics{
        true, tau, damping, attempt + 1, 0.0, 0.0, solve.stop_reason,
        solve.iterations, solve.operator_applications, solve.final_residual_norm};
    if (solve.converged)
    {
      sample_solution = std::move(solve.solution);
      break;
    }
    if (attempt == control_.maximum_damping_retries)
      throw std::runtime_error(std::string("MinPII GMRES did not converge: ") +
                               krylovStopReasonName(solve.stop_reason));
    damping *= control_.damping_multiplier;
    if (!isFiniteTrainingReal(damping))
      throw MatrixFreeNumericalError("MinPII damping retry overflowed");
  }

  std::vector<DerivativeValue> direction(schema.parameterCount());
  factor_.scoreFactor().applyAdjoint(constView(sample_solution), mutableView(direction));
  const DerivativeReal norm = parameterVectorNorm(schema, constView(direction));
  DerivativeReal scale = control_.learning_rate;
  if (control_.maximum_update_norm > 0.0 && norm > 0.0)
    scale = std::min(scale, control_.maximum_update_norm / norm);
  last_diagnostics_->direction_norm = norm;
  last_diagnostics_->applied_scale = scale;
  StructuredParameterSnapshot candidate =
      formCandidate(schema, parameters, constView(direction), scale);
  proposal_live_ = true;
  return candidate;
}

void MinimumProjectedInverseIterationUpdateRule::proposalAccepted(
    const StructuredParameterSchema&,
    const StructuredParameterSnapshot&,
    ParameterGradientView) noexcept
{
  proposal_live_ = false;
}

void MinimumProjectedInverseIterationUpdateRule::proposalRejected() noexcept
{
  proposal_live_ = false;
}

MinimumStepSRUpdateRule::MinimumStepSRUpdateRule(
    const ScoreCovarianceFactor& factor,
    MinimumStepUpdateControl control)
    : factor_(factor), control_(std::move(control))
{
  validateControl(control_);
}

StructuredParameterSnapshot MinimumStepSRUpdateRule::propose(
    const StructuredParameterSchema& schema,
    const StructuredParameterSnapshot& parameters,
    ParameterGradientView objective)
{
  validateUpdateInputs(factor_, proposal_live_, schema, parameters, objective,
                       control_.objective_tolerance);
  last_diagnostics_.reset();
  MinimumStepSampleOperator sample_operator(factor_, control_.damping,
                                             control_.center_shift);
  DistributedSampleKrylovResult solve = solveDistributedSamplePCG(
      sample_operator, factor_.residual(), control_.krylov);
  if (!solve.converged)
    throw std::runtime_error(std::string("minSR sample PCG did not converge: ") +
                             krylovStopReasonName(solve.stop_reason));
  std::vector<DerivativeValue> direction(schema.parameterCount());
  factor_.applyAdjoint(constView(solve.solution), mutableView(direction));
  const DerivativeReal norm = parameterVectorNorm(schema, constView(direction));
  DerivativeReal scale = control_.learning_rate;
  if (control_.maximum_update_norm > 0.0 && norm > 0.0)
    scale = std::min(scale, control_.maximum_update_norm / norm);
  last_diagnostics_ = MinimumStepUpdateDiagnostics{std::move(solve), norm, scale};
  StructuredParameterSnapshot candidate =
      formCandidate(schema, parameters, constView(direction), scale);
  proposal_live_ = true;
  return candidate;
}

void MinimumStepSRUpdateRule::proposalAccepted(
    const StructuredParameterSchema&,
    const StructuredParameterSnapshot&,
    ParameterGradientView) noexcept
{
  proposal_live_ = false;
}

void MinimumStepSRUpdateRule::proposalRejected() noexcept
{
  proposal_live_ = false;
}

/// Owning no-throw restore action for committed SPRING history.
class SPRINGUpdateRule::PreparedRestoreAction final
    : public OptimizerCheckpointContributor::PreparedRestore
{
public:
  PreparedRestoreAction(SPRINGUpdateRule& owner,
                        std::uint64_t update_count,
                        std::vector<DerivativeValue> history)
      : owner_(owner), update_count_(update_count), history_(std::move(history))
  {}

  void commit() noexcept override
  {
    owner_.history_.swap(history_);
    owner_.accepted_update_count_ = update_count_;
  }

private:
  SPRINGUpdateRule& owner_;
  std::uint64_t update_count_;
  std::vector<DerivativeValue> history_;
};

SPRINGUpdateRule::SPRINGUpdateRule(const StructuredParameterSchema& schema,
                                   SPRINGUpdateControl control)
    : schema_(schema), control_(std::move(control)),
      history_(schema.parameterCount()), pending_history_(schema.parameterCount())
{
  validateControl(control_.minimum_step);
  if (!isFiniteTrainingReal(control_.history_decay) ||
      control_.history_decay < 0.0 || control_.history_decay >= 1.0 ||
      !isFiniteTrainingReal(control_.maximum_update_norm_squared) ||
      control_.maximum_update_norm_squared < 0.0)
    throw std::invalid_argument("SPRING history controls are invalid");
  std::uint64_t hash = UINT64_C(1469598103934665603);
  extendString(hash, schema.fingerprint());
  extendFingerprint(hash, &control_.minimum_step.learning_rate, sizeof(DerivativeReal));
  extendFingerprint(hash, &control_.minimum_step.damping, sizeof(DerivativeReal));
  extendFingerprint(hash, &control_.minimum_step.center_shift, sizeof(DerivativeReal));
  extendFingerprint(hash, &control_.minimum_step.maximum_update_norm,
                    sizeof(DerivativeReal));
  extendFingerprint(hash, &control_.minimum_step.objective_tolerance,
                    sizeof(DerivativeReal));
  extendFingerprint(hash, &control_.minimum_step.krylov.relative_tolerance,
                    sizeof(DerivativeReal));
  extendFingerprint(hash, &control_.minimum_step.krylov.absolute_tolerance,
                    sizeof(DerivativeReal));
  extendFingerprint(hash, &control_.minimum_step.krylov.maximum_iterations,
                    sizeof(std::size_t));
  extendFingerprint(hash, &control_.minimum_step.krylov.maximum_operator_applications,
                    sizeof(std::size_t));
  extendFingerprint(hash, &control_.minimum_step.krylov.maximum_seconds,
                    sizeof(double));
  extendFingerprint(hash, &control_.minimum_step.krylov.stagnation_window,
                    sizeof(std::size_t));
  extendFingerprint(hash, &control_.minimum_step.krylov.minimum_relative_improvement,
                    sizeof(DerivativeReal));
  extendFingerprint(hash, &control_.minimum_step.krylov.curvature_tolerance,
                    sizeof(DerivativeReal));
  extendFingerprint(hash, &control_.minimum_step.krylov.imaginary_tolerance,
                    sizeof(DerivativeReal));
  extendFingerprint(hash, &control_.history_decay, sizeof(DerivativeReal));
  extendFingerprint(hash, &control_.maximum_update_norm_squared,
                    sizeof(DerivativeReal));
  configuration_fingerprint_ = formatFingerprint(hash);
}

SPRINGUpdateRule::SPRINGUpdateRule(const ScoreCovarianceFactor& factor,
                                   SPRINGUpdateControl control)
    : SPRINGUpdateRule(factor.parameterSchema(), std::move(control))
{
  bindFactor(factor);
}

void SPRINGUpdateRule::bindFactor(const ScoreCovarianceFactor& factor)
{
  if (proposal_live_)
    throw std::logic_error("Cannot rebind SPRING with a live proposal");
  if (factor.parameterSchema().providerId() != schema_.providerId() ||
      factor.parameterSchema().fingerprint() != schema_.fingerprint())
    throw std::invalid_argument("SPRING replacement factor schema mismatch");
  factor_ = &factor;
}

void SPRINGUpdateRule::unbindFactor() noexcept
{
  assert(!proposal_live_);
  factor_ = nullptr;
}

StructuredParameterSnapshot SPRINGUpdateRule::propose(
    const StructuredParameterSchema& schema,
    const StructuredParameterSnapshot& parameters,
    ParameterGradientView objective)
{
  if (factor_ == nullptr)
    throw std::logic_error("SPRING requires a bound score-covariance factor");
  validateUpdateInputs(*factor_, proposal_live_, schema, parameters, objective,
                       control_.minimum_step.objective_tolerance);
  if (accepted_update_count_ == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("SPRING accepted-update count overflow");
  last_diagnostics_.reset();

  std::vector<DerivativeValue> decayed_history = history_;
  scaleParameterVector(schema, {control_.history_decay, 0.0},
                       mutableView(decayed_history));
  const StructuredParameterVectorConstView history_view(
      schema, parameters.version, constView(decayed_history));
  std::vector<DerivativeValue> projected_residual(factor_->localSampleVectorSize());
  factor_->apply(history_view, mutableView(projected_residual));
  for (std::size_t sample = 0; sample < projected_residual.size(); ++sample)
    projected_residual[sample] = factor_->residual()[sample] - projected_residual[sample];

  MinimumStepSampleOperator sample_operator(
      *factor_, control_.minimum_step.damping,
      control_.minimum_step.center_shift);
  DistributedSampleKrylovResult solve = solveDistributedSamplePCG(
      sample_operator, constView(projected_residual), control_.minimum_step.krylov);
  if (!solve.converged)
    throw std::runtime_error(std::string("SPRING sample PCG did not converge: ") +
                             krylovStopReasonName(solve.stop_reason));
  std::vector<DerivativeValue> correction(schema.parameterCount());
  factor_->applyAdjoint(constView(solve.solution), mutableView(correction));
  pending_history_ = std::move(decayed_history);
  axpyParameterVector(schema, {1.0, 0.0}, constView(correction),
                      mutableView(pending_history_));
  const DerivativeReal norm = parameterVectorNorm(schema, constView(pending_history_));
  DerivativeReal scale = control_.minimum_step.learning_rate;
  if (control_.maximum_update_norm_squared > 0.0 && norm > 0.0)
    scale = std::min(scale,
                     std::sqrt(control_.maximum_update_norm_squared) / norm);
  if (control_.minimum_step.maximum_update_norm > 0.0 && norm > 0.0)
    scale = std::min(scale, control_.minimum_step.maximum_update_norm / norm);
  last_diagnostics_ = MinimumStepUpdateDiagnostics{std::move(solve), norm, scale};
  StructuredParameterSnapshot candidate =
      formCandidate(schema, parameters, constView(pending_history_), scale);
  proposal_live_ = true;
  return candidate;
}

void SPRINGUpdateRule::proposalAccepted(
    const StructuredParameterSchema&,
    const StructuredParameterSnapshot&,
    ParameterGradientView) noexcept
{
  assert(proposal_live_);
  history_.swap(pending_history_);
  std::fill(pending_history_.begin(), pending_history_.end(), DerivativeValue{});
  ++accepted_update_count_;
  proposal_live_ = false;
}

void SPRINGUpdateRule::proposalRejected() noexcept
{
  std::fill(pending_history_.begin(), pending_history_.end(), DerivativeValue{});
  proposal_live_ = false;
}

std::string SPRINGUpdateRule::stateFingerprint(
    std::uint64_t update_count,
    const std::vector<DerivativeValue>& history) const
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  extendFingerprint(hash, &update_count, sizeof(update_count));
  const std::uint64_t size = history.size();
  extendFingerprint(hash, &size, sizeof(size));
  if (!history.empty())
    extendFingerprint(hash, history.data(), history.size() * sizeof(DerivativeValue));
  return formatFingerprint(hash);
}

void SPRINGUpdateRule::validateRestoredState(
    std::uint64_t update_count,
    const std::vector<DerivativeValue>& history) const
{
  if (history.size() != history_.size() ||
      update_count == std::numeric_limits<std::uint64_t>::max() ||
      !std::all_of(history.begin(), history.end(), isFinite) ||
      std::any_of(history.begin(), history.end(),
                  [](DerivativeValue value) { return value.imag() != 0.0; }))
    throw std::runtime_error("SPRING checkpoint state is invalid");
  for (const ParameterBlockDescriptor& block : schema_.blocks())
    if (!block.trainable)
      for (std::size_t parameter = block.offset;
           parameter < block.offset + block.count; ++parameter)
        if (history[parameter] != DerivativeValue{})
          throw std::runtime_error("SPRING checkpoint has nonzero frozen history");
}

CheckpointContributorMetadata SPRINGUpdateRule::checkpointMetadata() const
{
  if (proposal_live_)
    throw std::logic_error("Cannot checkpoint SPRING with a live proposal");
  return {SPRING_CONTRIBUTOR_ID, {1, 0, 0}, configuration_fingerprint_,
          stateFingerprint(accepted_update_count_, history_)};
}

void SPRINGUpdateRule::writeCheckpointPayload(hdf_archive& archive) const
{
  if (proposal_live_)
    throw std::logic_error("Cannot checkpoint SPRING with a live proposal");
  archive.write(accepted_update_count_, "accepted_update_count");
  archive.write(history_, "history");
}

std::unique_ptr<OptimizerCheckpointContributor::PreparedRestore>
SPRINGUpdateRule::prepareCheckpointRestore(
    hdf_archive& archive,
    const CheckpointContributorMetadata& saved_metadata)
{
  if (proposal_live_)
    throw std::logic_error("Cannot restore SPRING with a live proposal");
  if (saved_metadata.contributor_id != SPRING_CONTRIBUTOR_ID ||
      saved_metadata.format_version[0] != 1 ||
      saved_metadata.configuration_fingerprint != configuration_fingerprint_)
    throw std::runtime_error("SPRING checkpoint metadata mismatch");
  std::uint64_t update_count = 0;
  std::vector<DerivativeValue> history;
  archive.read(update_count, "accepted_update_count");
  archive.read(history, "history");
  validateRestoredState(update_count, history);
  if (saved_metadata.state_fingerprint != stateFingerprint(update_count, history))
    throw std::runtime_error("SPRING checkpoint fingerprint mismatch");
  return std::make_unique<PreparedRestoreAction>(
      *this, update_count, std::move(history));
}

} // namespace qmcplusplus::wftrain
