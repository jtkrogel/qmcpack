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
    h_factor_ = std::make_unique<ProjectedInverseIterationFactor>(factor, term_mask);
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
    MatrixFreeLinearMethodResult result, double learning_rate, double normalization_xi)
    : factor_(factor), result_(std::move(result)), learning_rate_(learning_rate), xi_(normalization_xi)
{
  if (!result_.converged || !isFiniteTrainingReal(learning_rate_) || learning_rate_ <= 0.0 ||
      !isFiniteTrainingReal(xi_) || xi_ < 0.0 || xi_ > 1.0)
    throw std::invalid_argument("Live LM requires a converged solve and valid step controls");
}

StructuredParameterSnapshot LinearMethodUpdateRule::propose(const StructuredParameterSchema& schema,
    const StructuredParameterSnapshot& parameters, ParameterGradientView objective)
{
  if (parameters.version != result_.parameter_version || objective.parameter_version != parameters.version ||
      objective.schema_fingerprint != schema.fingerprint())
    throw std::invalid_argument("Live LM update has a stale parameter binding");
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
