//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include "SampledLinearMethod.h"
#include "TrainingNumerics.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>

namespace qmcplusplus::wftrain
{
namespace
{
/// Accumulate one term-matched local-energy VJP into canonical scratch.
class EnergyAdjointSink final : public ParameterReductionSink
{
public:
  explicit EnergyAdjointSink(std::vector<DerivativeValue>& values) : values_(values) {}

protected:
  void onBegin(const DerivativeStreamDescriptor&, const ParameterChunkPlan&,
               DerivativeArrayView<const VJPCoefficientChannel>) override {}
  void consume(std::size_t channel, const ParameterChunkConstView& chunk) override
  {
    const auto offset = chunk.descriptor().parameter_offset;
    if (channel != 0 || offset > values_.size() || chunk.values().size() > values_.size() - offset)
      throw std::logic_error("Sampled LM local-energy adjoint extent mismatch");
    for (std::size_t i = 0; i < chunk.values().size(); ++i)
      values_[offset + i] += chunk.values()[i];
  }

private:
  std::vector<DerivativeValue>& values_;
};

/// Contract raw Jv into its statistical mean without retaining rows or a mean P-vector.
class LinearScoreMeanSink final : public SampleProductSink
{
public:
  explicit LinearScoreMeanSink(const ScoreCovarianceBatch& batch) : batch_(batch) {}
  double weighted = 0, absolute = 0, weight = 0;

protected:
  void consume(const SampleProductTileConstView& tile) override
  {
    const auto offset = tile.descriptor().sample_offset - batch_.derivative_operator->sampleOffset();
    if (offset > batch_.weights.size() || tile.values().size() > batch_.weights.size() - offset)
      throw std::invalid_argument("Linear CI score-mean tile extent mismatch");
    for (std::size_t i = 0; i < tile.values().size(); ++i)
    {
      const auto value = tile.values()[i];
      const auto w = batch_.weights[offset + i];
      if (!isFiniteTrainingReal(value.real()) || !isFiniteTrainingReal(value.imag()) || value.imag() != 0 ||
          !isFiniteTrainingReal(w) || w < 0)
        throw std::invalid_argument("Linear CI retraction requires finite real scores and nonnegative weights");
      weighted += w * value.real();
      absolute += w * std::abs(value.real());
      weight += w;
    }
  }

private:
  const ScoreCovarianceBatch& batch_;
};
}

SampledLinearMethodOperator::SampledLinearMethodOperator(const ScoreCovarianceFactor& factor,
    DerivativeArrayView<const ScoreCovarianceBatch> batches, std::uint32_t term_mask,
    bool overlap, bool symmetrize, double tangent_shift)
    : AugmentedLinearMethodOperator(factor.parameterSchema(), factor.parameterVersion(),
                                    ReductionDomain::GLOBAL, overlap || symmetrize),
      factor_(factor), batches_(batches.begin(), batches.end()), term_mask_(term_mask),
      overlap_(overlap), symmetrize_(symmetrize), shift_(tangent_shift)
{
  if (!isFiniteTrainingReal(shift_) || shift_ < 0.0 || factor.sampleDomain() != ScoreSampleDomain::REAL)
    throw std::invalid_argument("Sampled LM requires real samples and a nonnegative shift");
  double local_weight = 0.0;
  std::exception_ptr failure;
  try
  {
    // Validate the nonowning batch views before indexing them. All ranks enter
    // the same reduction even if one participant supplies malformed samples.
    for (const auto& batch : batches_)
    {
      const auto* op = batch.derivative_operator;
      if (!op || batch.weights.size() != batch.local_energies.size() ||
          op->parameterVersion() != factor.parameterVersion() ||
          op->parameterSchema().fingerprint() != factor.parameterSchema().fingerprint() ||
          op->parameterSchema().providerId() != factor.parameterSchema().providerId())
        throw std::invalid_argument("Sampled LM batch identity or extent mismatch");
      for (std::size_t i = 0; i < batch.weights.size(); ++i)
      {
        const auto energy = batch.local_energies[i];
        if (!isFiniteTrainingReal(batch.weights[i]) || batch.weights[i] < 0.0 ||
            !isFiniteTrainingReal(energy.real()) || !isFiniteTrainingReal(energy.imag()) || energy.imag() != 0.0)
          throw std::invalid_argument("Sampled LM requires finite real energies and nonnegative weights");
        local_weight += batch.weights[i];
      }
    }
    if (!isFiniteTrainingReal(local_weight))
      throw std::invalid_argument("Sampled LM statistical weight overflow");
  }
  catch (...) { failure = std::current_exception(); }
  const double weight = factor_.reduceSampleScalar({local_weight, 0.0}, failure).real();
  if (!isFiniteTrainingReal(weight) || !(weight > 0.0))
    throw std::invalid_argument("Sampled LM has invalid total statistical weight");
  for (const auto& batch : batches_)
    for (std::size_t i = 0; i < batch.weights.size(); ++i)
    {
      sqrt_q_.push_back(std::sqrt(batch.weights[i] / weight));
      energies_.push_back(batch.local_energies[i].real());
    }
  if (sqrt_q_.size() != factor.localSampleVectorSize())
    throw std::invalid_argument("Sampled LM batches differ from score-factor samples");
  d_.resize(sqrt_q_.size());
  l_.resize(sqrt_q_.size());
  coefficients_.resize(sqrt_q_.size());
  adjoint_.resize(factor.parameterSchema().parameterCount());
  local_gradient_.resize(adjoint_.size());
  if (!overlap_)
    h_factor_ = std::make_unique<ProjectedInverseIterationFactor>(
        factor, term_mask, HamiltonianSampleProjection::REAL_TANGENT);
}

std::size_t SampledLinearMethodOperator::retainedBytes() const noexcept
{
  return (sqrt_q_.capacity() + energies_.capacity()) * sizeof(double) +
      (d_.capacity() + l_.capacity() + coefficients_.capacity() +
       adjoint_.capacity() + local_gradient_.capacity()) * sizeof(DerivativeValue) +
      (h_factor_ ? h_factor_->retainedBytes() : 0);
}

void SampledLinearMethodOperator::evaluate(DerivativeArrayView<const DerivativeValue> direction,
                                           DerivativeArrayView<DerivativeValue> result) const
{
  const auto& schema = factor_.parameterSchema();
  const StructuredParameterVectorConstView tangent(schema, factor_.parameterVersion(),
                                                    {direction.data() + 1, direction.size() - 1});
  factor_.apply(tangent, {d_.data(), d_.size()});
  for (std::size_t i = 0; i < d_.size(); ++i) d_[i] += sqrt_q_[i] * direction[0];
  std::fill(result.begin(), result.end(), DerivativeValue{});

  if (overlap_)
  {
    DerivativeValue reference{};
    for (std::size_t i = 0; i < d_.size(); ++i) reference += sqrt_q_[i] * d_[i];
    result[0] = factor_.reduceSampleScalar(reference);
    factor_.applyAdjoint({d_.data(), d_.size()}, {result.data() + 1, result.size() - 1});
  }
  else
  {
    // The raw action uses only the Hamiltonian JVP followed by the score adjoint.
    h_factor_->applyShifted(tangent, 0.0, {l_.data(), l_.size()});
    DerivativeValue reference{};
    for (std::size_t i = 0; i < l_.size(); ++i)
    {
      l_[i] += sqrt_q_[i] * energies_[i] * direction[0];
      reference += sqrt_q_[i] * l_[i];
    }
    result[0] = factor_.reduceSampleScalar(reference);
    factor_.applyAdjoint({l_.data(), l_.size()}, {result.data() + 1, result.size() - 1});
    if (symmetrize_)
    {
      // L^T D uses the local-energy VJP plus O^T(E_L D); the ECP VJP
      // is bound to the same immutable accepted quadrature replay as the JVP.
      std::fill(local_gradient_.begin(), local_gradient_.end(), DerivativeValue{});
      std::size_t offset = 0;
      std::exception_ptr failure;
      try
      {
        for (const auto& batch : batches_)
        {
          const std::size_t count = batch.weights.size();
          for (std::size_t i = 0; i < count; ++i)
            coefficients_[i] = sqrt_q_[offset + i] * d_[offset + i];
          const auto& op = *batch.derivative_operator;
          const CoefficientView coefficients{schema.providerId(), schema.fingerprint(),
              factor_.parameterVersion(), op.batchOrdinal(), op.sampleOffset(),
              {coefficients_.data(), count}};
          const VJPCoefficientChannel channel{"lm-energy-adjoint",
              DerivativeProduct::LOCAL_ENERGY_VJP, coefficients, term_mask_};
          EnergyAdjointSink sink(local_gradient_);
          op.applyVJPs({&channel, 1}, DerivativeAdjoint::HERMITIAN, sink);
          offset += count;
        }
        // This LM/ALM basis has real scores and real tangent coefficients. The
        // corresponding Hamiltonian is Re(D^dagger H D), even when a fixed
        // parameter-independent phase gives complex physical energy derivatives.
        // Check both parts before projection; never hide an imaginary NaN.
        for (auto& value : local_gradient_)
        {
          if (!isFiniteTrainingReal(value.real()) || !isFiniteTrainingReal(value.imag()))
            throw MatrixFreeNumericalError("Sampled LM energy adjoint is non-finite");
          value = {value.real(), 0.0};
        }
      }
      catch (...) { failure = std::current_exception(); }
      factor_.reduceParameterVector({local_gradient_.data(), local_gradient_.size()}, failure);
      DerivativeValue transpose_reference{};
      for (std::size_t i = 0; i < d_.size(); ++i)
      {
        transpose_reference += sqrt_q_[i] * energies_[i] * d_[i];
        l_[i] = energies_[i] * d_[i];
      }
      factor_.applyAdjoint({l_.data(), l_.size()}, {adjoint_.data(), adjoint_.size()});
      result[0] = 0.5 * (result[0] + factor_.reduceSampleScalar(transpose_reference));
      for (std::size_t i = 0; i < adjoint_.size(); ++i)
        result[i + 1] = 0.5 * (result[i + 1] + adjoint_[i] + local_gradient_[i]);
    }
  }
  for (const auto& block : schema.blocks())
    for (std::size_t i = block.offset; i < block.offset + block.count; ++i)
      result[i + 1] = block.trainable ? result[i + 1] + shift_ * direction[i + 1] : DerivativeValue{};
}

LinearMethodUpdateRule::LinearMethodUpdateRule(const ScoreCovarianceFactor& factor,
    MatrixFreeLinearMethodResult result, double learning_rate, double normalization_xi,
    LinearMethodNormalization normalization, DerivativeArrayView<const ScoreCovarianceBatch> linear_batches)
    : factor_(factor), result_(std::move(result)), learning_rate_(learning_rate), xi_(normalization_xi),
      normalization_(normalization), linear_batches_(linear_batches.begin(), linear_batches.end())
{
  if (!result_.converged || !isFiniteTrainingReal(learning_rate_) || learning_rate_ <= 0.0 ||
      !isFiniteTrainingReal(xi_) || xi_ < 0.0 || xi_ > 1.0)
    throw std::invalid_argument("Live LM requires a converged solve and valid step controls");
}

StructuredParameterSnapshot LinearMethodUpdateRule::propose(const StructuredParameterSchema& schema,
    const StructuredParameterSnapshot& parameters, ParameterGradientView objective)
{
  const bool stale = parameters.version != result_.parameter_version || objective.parameter_version != parameters.version ||
      objective.schema_fingerprint != schema.fingerprint();
  bool linear_fallback = false;
  if (normalization_ == LinearMethodNormalization::MIXED_FIXED_REFERENCE)
  {
    bool nonlinear_tangent = false;
    std::vector<DerivativeValue> masked, sample;
    std::exception_ptr failure;
    try
    {
      if (stale || factor_.sampleDomain() != ScoreSampleDomain::REAL ||
          result_.stop_reason != LinearMethodStopReason::CONVERGED || result_.provider_id != schema.providerId() ||
          result_.schema_fingerprint != schema.fingerprint() || parameters.schema_fingerprint != schema.fingerprint() ||
          parameters.values.size() != schema.parameterCount() || result_.eigenvector.size() != schema.parameterCount() + 1 ||
          factor_.parameterVersion() != parameters.version || factor_.parameterSchema().fingerprint() != schema.fingerprint())
        throw std::invalid_argument("Mixed LM retraction has an incompatible solved state");
      double magnitude = 0;
      for (const auto value : result_.eigenvector)
      {
        if (!isFiniteTrainingReal(value.real()) || !isFiniteTrainingReal(value.imag()) || value.imag() != 0)
          throw std::invalid_argument("Mixed LM Ritz coefficients must be finite and real");
        magnitude = std::max(magnitude, std::abs(value.real()));
      }
      for (const auto value : parameters.values)
        if (!isFiniteTrainingReal(value))
          throw std::invalid_argument("Mixed LM accepted parameters must be finite");
      for (const auto& block : schema.blocks())
        for (std::size_t i = block.offset; i < block.offset + block.count; ++i)
        {
          const auto value = result_.eigenvector[i + 1];
          if (!block.trainable && value != DerivativeValue{})
            throw std::invalid_argument("Mixed LM retraction cannot update frozen coordinates");
          nonlinear_tangent |= block.trainable && block.update_group != "legacy_linear" && value != DerivativeValue{};
        }
      std::size_t local_samples = 0;
      for (const auto& batch : linear_batches_)
      {
        if (!batch.derivative_operator || batch.weights.size() != batch.derivative_operator->sampleCount() ||
            batch.derivative_operator->parameterVersion() != parameters.version ||
            batch.derivative_operator->parameterSchema().fingerprint() != schema.fingerprint())
          throw std::invalid_argument("Mixed LM retraction has incompatible samples");
        local_samples += batch.weights.size();
      }
      if (local_samples != factor_.localSampleVectorSize())
        throw std::invalid_argument("Mixed LM samples do not match its centered factor");
      if (nonlinear_tangent)
      {
        if (std::abs(result_.eigenvector[0].real()) <= 1e-10 * magnitude)
          throw std::invalid_argument("Mixed LM nonlinear reference chart is unusable");
        masked.resize(schema.parameterCount());
        sample.resize(factor_.localSampleVectorSize());
      }
    }
    catch (...) { failure = std::current_exception(); }
    // Agree local validation/allocation and the fallback schedule before any products.
    const double nonlinear_ranks = factor_.reduceSampleScalar({nonlinear_tangent ? 1.0 : 0.0, 0}, failure).real();
    const double ranks = factor_.reduceSampleScalar({1, 0}).real();
    if (nonlinear_ranks != 0 && nonlinear_ranks != ranks)
      throw std::invalid_argument("Mixed LM ranks disagree on the nonlinear tangent chart");
    linear_fallback = nonlinear_ranks == 0;
    if (!linear_fallback)
    {
      const double reference = result_.eigenvector[0].real();
      for (const auto& block : schema.blocks())
        if (block.trainable && block.update_group == "legacy_linear")
          for (std::size_t i = block.offset; i < block.offset + block.count; ++i)
            masked[i] = result_.eigenvector[i + 1].real() / reference;
      double weighted = 0, absolute = 0, weight = 0;
      try
      {
        const StructuredParameterVectorConstView linear(schema, parameters.version, {masked.data(), masked.size()});
        for (const auto& batch : linear_batches_)
        {
          LinearScoreMeanSink sink(batch);
          batch.derivative_operator->applyScoreJVP(linear, sink);
          weighted += sink.weighted;
          absolute += sink.absolute;
          weight += sink.weight;
        }
        if (!isFiniteTrainingReal(weighted) || !isFiniteTrainingReal(absolute) || !isFiniteTrainingReal(weight))
          throw std::overflow_error("Mixed LM linear score mean overflow");
      }
      catch (...) { failure = std::current_exception(); }
      const double total_weight = factor_.reduceSampleScalar({weight, 0}, failure).real();
      const double mean = factor_.reduceSampleScalar({weighted, 0}).real() / total_weight;
      const double absolute_mean = factor_.reduceSampleScalar({absolute, 0}).real() / total_weight;
      if (!(total_weight > 0) || !isFiniteTrainingReal(total_weight) ||
          !isFiniteTrainingReal(mean) || !isFiniteTrainingReal(absolute_mean))
        throw std::invalid_argument("Mixed LM linear score mean is invalid");
      std::fill(masked.begin(), masked.end(), DerivativeValue{});
      for (const auto& block : schema.blocks())
        if (block.trainable && block.update_group != "legacy_linear")
          for (std::size_t i = block.offset; i < block.offset + block.count; ++i)
            masked[i] = result_.eigenvector[i + 1].real() / reference;
      factor_.apply({schema, parameters.version, {masked.data(), masked.size()}}, {sample.data(), sample.size()});
      double local_norm = 0;
      try
      {
        for (const auto value : sample) local_norm += std::norm(value);
        if (!isFiniteTrainingReal(local_norm)) throw std::overflow_error("Mixed LM nonlinear norm overflow");
      }
      catch (...) { failure = std::current_exception(); }
      const double q = factor_.reduceSampleScalar({local_norm, 0}, failure).real();
      const double beta = (1 - xi_) * q / (1 - xi_ + xi_ * std::sqrt(1 + q));
      const double denominator = 1 - mean + beta;
      // One shear preserves the Ritz tangent direction; q deliberately uses only NN.
      if (!isFiniteTrainingReal(q) || q < 0 || !isFiniteTrainingReal(beta) || !isFiniteTrainingReal(denominator) ||
          std::abs(denominator) <= 1e-10 * (1 + absolute_mean + std::abs(beta)))
        throw std::invalid_argument("Mixed LM common projective denominator is unusable");
      auto candidate = parameters;
      for (const auto& block : schema.blocks())
        for (std::size_t i = block.offset; i < block.offset + block.count; ++i)
        {
          if (block.trainable)
            candidate.values[i] += learning_rate_ * (result_.eigenvector[i + 1].real() / reference) / denominator;
          if (!isFiniteTrainingReal(candidate.values[i]))
            throw std::overflow_error("Mixed LM retracted parameter is nonfinite");
        }
      return candidate;
    }
  }
  if (stale && normalization_ != LinearMethodNormalization::FIXED_REFERENCE_LINEAR && !linear_fallback)
    throw std::invalid_argument("Live LM update has a stale parameter binding");
  if (normalization_ == LinearMethodNormalization::FIXED_REFERENCE_LINEAR || linear_fallback)
  {
    double weighted = 0, absolute = 0, weight = 0;
    std::exception_ptr failure;
    try
    {
      if (stale || factor_.sampleDomain() != ScoreSampleDomain::REAL ||
          result_.stop_reason != LinearMethodStopReason::CONVERGED || result_.provider_id != schema.providerId() ||
          result_.schema_fingerprint != schema.fingerprint() || parameters.schema_fingerprint != schema.fingerprint() ||
          parameters.values.size() != schema.parameterCount() || result_.eigenvector.size() != schema.parameterCount() + 1 ||
          factor_.parameterVersion() != parameters.version || factor_.parameterSchema().fingerprint() != schema.fingerprint())
        throw std::invalid_argument("Linear CI retraction has an incompatible solved state");
      for (const auto value : result_.eigenvector)
        if (!isFiniteTrainingReal(value.real()) || !isFiniteTrainingReal(value.imag()) || value.imag() != 0)
          throw std::invalid_argument("Linear CI Ritz coefficients must be finite and real");
      for (const auto value : parameters.values)
        if (!isFiniteTrainingReal(value))
          throw std::invalid_argument("Linear CI accepted parameters must be finite");
      for (const auto& block : schema.blocks())
        if (!block.trainable)
          for (std::size_t i = block.offset; i < block.offset + block.count; ++i)
            if (result_.eigenvector[i + 1] != DerivativeValue{})
              throw std::invalid_argument("Linear CI retraction cannot update frozen coordinates");
      const StructuredParameterVectorConstView tangent(schema, parameters.version,
          {result_.eigenvector.data() + 1, schema.parameterCount()});
      std::size_t local_samples = 0;
      for (const auto& batch : linear_batches_)
      {
        if (!batch.derivative_operator || batch.weights.size() != batch.derivative_operator->sampleCount())
          throw std::invalid_argument("Linear CI retraction has incompatible samples");
        LinearScoreMeanSink sink(batch);
        batch.derivative_operator->applyScoreJVP(tangent, sink);
        weighted += sink.weighted;
        absolute += sink.absolute;
        weight += sink.weight;
        local_samples += batch.weights.size();
      }
      if (local_samples != factor_.localSampleVectorSize())
        throw std::invalid_argument("Linear CI retraction samples do not match its centered factor");
      if (!isFiniteTrainingReal(weighted) || !isFiniteTrainingReal(absolute) || !isFiniteTrainingReal(weight))
        throw std::overflow_error("Linear CI score mean overflow");
    }
    catch (...) { failure = std::current_exception(); }
    // Agree all fallible local work before subsequent reductions, including empty ranks.
    const double total_weight = factor_.reduceSampleScalar({weight, 0}, failure).real();
    const double mean = factor_.reduceSampleScalar({weighted, 0}).real() / total_weight;
    const double absolute_mean = factor_.reduceSampleScalar({absolute, 0}).real() / total_weight;
    const double reference = result_.eigenvector[0].real();
    const double gauge = reference - mean;
    if (!(total_weight > 0) || !isFiniteTrainingReal(total_weight) || !isFiniteTrainingReal(mean) || !isFiniteTrainingReal(absolute_mean) ||
        !isFiniteTrainingReal(gauge) || std::abs(gauge) <= 1e-10 * (std::abs(reference) + absolute_mean))
      throw std::invalid_argument("Linear CI fixed-reference projective denominator is unusable");
    auto candidate = parameters;
    for (const auto& block : schema.blocks())
      for (std::size_t i = block.offset; i < block.offset + block.count; ++i)
      {
        if (block.trainable)
          candidate.values[i] += learning_rate_ * (result_.eigenvector[i + 1].real() / gauge);
        if (!isFiniteTrainingReal(candidate.values[i]))
          throw std::overflow_error("Linear CI retracted parameter is nonfinite");
      }
    return candidate; // xi is deliberately nonlinear-only; shifts belong to the solved H/S actions.
  }
  const auto update = formLinearMethodUpdateCandidate(schema, result_);
  std::vector<DerivativeValue> sample(factor_.localSampleVectorSize());
  factor_.apply({schema, parameters.version, {update.direction.data(), update.direction.size()}},
                {sample.data(), sample.size()});
  double local_norm = 0.0;
  for (auto value : sample) local_norm += std::norm(value);
  const double q = factor_.reduceSampleScalar({local_norm, 0.0}).real();
  const double denominator = 1.0 + (1.0 - xi_) * q /
      (1.0 - xi_ + xi_ * std::sqrt(1.0 + q));
  auto candidate = parameters;
  for (std::size_t i = 0; i < candidate.values.size(); ++i)
    candidate.values[i] += learning_rate_ * update.direction[i].real() / denominator;
  return candidate;
}
}
