//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#include "QMCDrivers/WFTrain/KFACOptimizer.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"
#include "io/hdf/hdf_archive.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace qmcplusplus::wftrain
{
namespace
{
constexpr char contributor_id[] = "qmcpack/wftrain/kfac";
}

/// Complete committed optimizer state, independent of process-local model versions.
struct KFACOptimizer::State
{
  std::uint64_t step = 0;
  std::uint64_t data_seen = 0;
  std::uint64_t inverse_step = 0;
  std::optional<KFACFactorState> factors;
  std::optional<KFACInverseState> inverse;
};

/// Own the next recurrence and its version-bound solve until publication succeeds.
struct KFACOptimizer::Pending
{
  std::unique_ptr<State> state;
  std::unique_ptr<KFACPreconditioner> inverse;
  std::size_t parameter_version = 0;
  bool proposed = false;
};

/// Make final checkpoint installation an allocation-free, nonthrowing owner swap.
class KFACOptimizer::PreparedRestoreAction final : public OptimizerCheckpointContributor::PreparedRestore
{
public:
  PreparedRestoreAction(KFACOptimizer& owner, std::unique_ptr<State> state)
      : owner_(owner), state_(std::move(state)) {}

  void commit() noexcept override { owner_.state_.swap(state_); }

private:
  KFACOptimizer& owner_;
  std::unique_ptr<State> state_;
};

KFACOptimizer::KFACOptimizer(KFACBlockRegistry registry, KFACOptimizerControl control)
    : registry_(std::move(registry)), control_(control), state_(std::make_unique<State>())
{
  if (!isFiniteTrainingReal(control_.learning_rate) || control_.learning_rate <= 0.0 ||
      !isFiniteTrainingReal(control_.learning_rate_decay) || control_.learning_rate_decay < 0.0 ||
      !isFiniteTrainingReal(control_.ema_decay) || control_.ema_decay < 0.0 || control_.ema_decay >= 1.0 ||
      control_.inverse_period == 0 || !isFiniteTrainingReal(control_.norm_constraint) ||
      control_.norm_constraint < 0.0 || !isFiniteTrainingReal(control_.inverse.damping) ||
      control_.inverse.damping <= 0.0 || !isFiniteTrainingReal(control_.inverse.fallback_damping) ||
      control_.inverse.fallback_damping <= 0.0)
    throw std::invalid_argument("Invalid persistent KFAC controls");
  if ((control_.convention != KFACCurvatureConvention::NATIVE_SCORE &&
       control_.convention != KFACCurvatureConvention::REGISTERED_GAUSSIAN) ||
      (control_.inverse.damping_policy != KFACDampingPolicy::EQUAL_SPLIT &&
       control_.inverse.damping_policy != KFACDampingPolicy::TRACE_BALANCED) ||
      (control_.update_order != KFACUpdateOrder::POST_INVERSE_RATE &&
       control_.update_order != KFACUpdateOrder::PRE_INVERSE_RATE) ||
      (control_.schedule_arithmetic != KFACScheduleArithmetic::DIVIDE_RATE &&
       control_.schedule_arithmetic != KFACScheduleArithmetic::MULTIPLY_RECIPROCAL))
    throw std::invalid_argument("Invalid persistent KFAC curvature or damping policy");
  const std::vector<double> configuration{control_.learning_rate, control_.learning_rate_decay,
      static_cast<double>(control_.schedule_float32), control_.ema_decay,
      static_cast<double>(control_.inverse_period), control_.norm_constraint,
      static_cast<double>(control_.convention), control_.inverse.damping,
      control_.inverse.fallback_damping, static_cast<double>(control_.inverse.damping_policy)};
  configuration_fingerprint_ = registry_.fingerprint() + "/" + TrainingCheckpoint::valuesFingerprint(configuration);
  if (control_.update_order == KFACUpdateOrder::PRE_INVERSE_RATE)
    configuration_fingerprint_ += "/pre-inverse-rate-v1";
  if (control_.schedule_arithmetic == KFACScheduleArithmetic::MULTIPLY_RECIPROCAL)
    configuration_fingerprint_ += "/reciprocal-schedule-v1";
}

KFACOptimizer::~KFACOptimizer() = default;

std::uint64_t KFACOptimizer::step() const noexcept { return state_->step; }

std::uint64_t KFACOptimizer::dataSeen() const noexcept { return state_->data_seen; }

std::uint64_t KFACOptimizer::inverseStep() const noexcept { return state_->inverse_step; }

DerivativeReal KFACOptimizer::effectiveLearningRate() const noexcept
{
  if (control_.learning_rate_decay == 0.0) return control_.learning_rate;
  if (control_.schedule_float32)
  {
    // Preserve scalar dtype rounding even in builds that permit reassociation.
    volatile float progress = static_cast<float>(state_->step) / static_cast<float>(control_.learning_rate_decay);
    volatile float denominator = 1.0F + progress;
    if (control_.schedule_arithmetic == KFACScheduleArithmetic::MULTIPLY_RECIPROCAL)
    {
      volatile float reciprocal = 1.0F / denominator;
      volatile float rate = static_cast<float>(control_.learning_rate) * reciprocal;
      return rate;
    }
    volatile float rate = static_cast<float>(control_.learning_rate) / denominator;
    return rate;
  }
  const double denominator = 1.0 + static_cast<double>(state_->step) / control_.learning_rate_decay;
  return control_.schedule_arithmetic == KFACScheduleArithmetic::MULTIPLY_RECIPROCAL
      ? control_.learning_rate * (1.0 / denominator) : control_.learning_rate / denominator;
}

void KFACOptimizer::prepareStatistics(const KFACFactorAccumulator& statistics)
{
  psiformer::memory::Phase phase("kfac_factor_update_and_inverse");
  traceStorage();
  if (pending_) throw std::logic_error("KFAC already has a speculative update");
  if (statistics.registry().fingerprint() != registry_.fingerprint() ||
      statistics.curvatureConvention() != control_.convention ||
      statistics.sampleCount() == 0 || statistics.sampleActive())
    throw std::invalid_argument("KFAC statistics do not match the persistent optimizer");
  if (state_->step == std::numeric_limits<std::uint64_t>::max() ||
      statistics.sampleCount() > std::numeric_limits<std::uint64_t>::max() - state_->data_seen)
    throw std::overflow_error("KFAC progress counter exhausted");
  // Build the complete next state before exposing any proposal to publication.
  auto pending = std::make_unique<Pending>();
  pending->state = std::make_unique<State>();
  pending->parameter_version = statistics.parameterVersion();
  pending->state->step = state_->step + 1;
  pending->state->data_seen = state_->data_seen + statistics.sampleCount();
  pending->state->factors = updateKFACFactorState(state_->factors ? &*state_->factors : nullptr,
                                               statistics, control_.ema_decay);
  const bool refresh = !state_->inverse || state_->step % control_.inverse_period == 0;
  if (refresh)
  {
    pending->inverse = std::make_unique<KFACPreconditioner>(registry_, statistics.parameterVersion(),
        *pending->state->factors, control_.inverse);
    pending->state->inverse_step = state_->step;
  }
  else
  {
    // Affine caches may be stale; diagonal blocks always use current averages.
    pending->inverse = std::make_unique<KFACPreconditioner>(registry_, statistics.parameterVersion(), *state_->inverse);
    pending->inverse->refreshFallbackDiagonal(*pending->state->factors, control_.inverse.fallback_damping);
    pending->state->inverse_step = state_->inverse_step;
  }
  pending->state->inverse = pending->inverse->exportInverseState();
  pending_ = std::move(pending);
  traceStorage();
}

StructuredParameterSnapshot KFACOptimizer::propose(const StructuredParameterSchema& schema,
    const StructuredParameterSnapshot& parameters, ParameterGradientView objective)
{
  psiformer::memory::Phase phase("kfac_solve_and_candidate");
  if (!pending_ || pending_->proposed || schema.fingerprint() != registry_.parameterSchema().fingerprint() ||
      parameters.version != pending_->parameter_version || parameters.schema_fingerprint != schema.fingerprint() ||
      parameters.values.size() != schema.parameterCount() || objective.schema_fingerprint != schema.fingerprint() ||
      objective.parameter_version != parameters.version || objective.reduction_domain != ReductionDomain::GLOBAL ||
      objective.gradient.size() != schema.parameterCount())
    throw std::logic_error("KFAC proposal lacks matching prepared statistics");
  const double rate = effectiveLearningRate();
  if (!isFiniteTrainingReal(rate) || rate <= 0.0)
    throw std::runtime_error("KFAC scheduled learning rate is invalid");
  std::vector<DerivativeValue> gradient(objective.gradient.size()), direction(gradient.size());
  psiformer::memory::HostCapacity trace_vectors;
  if (psiformer::memory::enabled()) trace_vectors.snapshotAt(__FILE__, __LINE__, "KFACOptimizer::gradient_and_direction",
      (gradient.capacity() + direction.capacity()) * sizeof(DerivativeValue));
  const bool pre_scaled = control_.update_order == KFACUpdateOrder::PRE_INVERSE_RATE;
  for (std::size_t i = 0; i < gradient.size(); ++i)
    gradient[i] = (pre_scaled ? rate : 1.0) * objective.gradient[i];
  pending_->inverse->apply({schema, parameters.version, {gradient.data(), gradient.size()}},
                           {direction.data(), direction.size()});
  // Preserve the schedule's squared-rate dtype before the parameter-space product.
  double product = 0.0;
  for (std::size_t i = 0; i < gradient.size(); ++i)
    product += (std::conj(gradient[i]) * direction[i]).real();
  double rate_squared = rate * rate;
  if (!pre_scaled && control_.schedule_float32 && control_.learning_rate_decay > 0.0)
  {
    volatile float square = static_cast<float>(rate) * static_cast<float>(rate);
    rate_squared = square;
  }
  last_metric_ = pre_scaled ? product : rate_squared * product;
  if (!isFiniteTrainingReal(last_metric_) || last_metric_ < 0.0)
    throw std::runtime_error("KFAC damped gradient metric is invalid");
  last_scale_ = control_.norm_constraint > 0.0 && last_metric_ > control_.norm_constraint
      ? std::sqrt(control_.norm_constraint / last_metric_) : 1.0;
  auto candidate = parameters;
  for (std::size_t i = 0; i < direction.size(); ++i)
    candidate.values[i] -= (pre_scaled ? 1.0 : rate) * last_scale_ * direction[i].real();
  pending_->proposed = true;
  return candidate;
}

void KFACOptimizer::proposalAccepted(const StructuredParameterSchema&, const StructuredParameterSnapshot&,
                                     ParameterGradientView) noexcept
{
  if (pending_ && pending_->proposed) state_.swap(pending_->state);
  pending_.reset();
  traceStorage();
}

void KFACOptimizer::proposalRejected() noexcept
{
  pending_.reset();
  traceStorage();
}

void KFACOptimizer::traceStorage() noexcept
{
  if (!psiformer::memory::enabled()) return;
  // The same vector-capacity ownership used by retainedNumericBytes(), separated
  // by acceptance lifetime. Exported inverse banks own distinct vectors.
  // Retire pending observations before publishing accepted capacities so a
  // successful ownership transfer cannot manufacture a double-counted peak.
  trace_storage_[2].snapshotAt(__FILE__, __LINE__, "KFACOptimizer::pending_factors",
      pending_ && pending_->state->factors ? pending_->state->factors->values.capacity() * sizeof(DerivativeReal) : 0);
  trace_storage_[3].snapshotAt(__FILE__, __LINE__, "KFACOptimizer::pending_inverse",
      pending_ && pending_->state->inverse ? pending_->state->inverse->values.capacity() * sizeof(DerivativeReal) : 0);
  // The prepared preconditioner reports its own capacity, also for stateless KFAC.
  trace_storage_[0].snapshotAt(__FILE__, __LINE__, "KFACOptimizer::accepted_factors",
      state_->factors ? state_->factors->values.capacity() * sizeof(DerivativeReal) : 0);
  trace_storage_[1].snapshotAt(__FILE__, __LINE__, "KFACOptimizer::accepted_inverse",
      state_->inverse ? state_->inverse->values.capacity() * sizeof(DerivativeReal) : 0);
}

std::size_t KFACOptimizer::retainedNumericBytes() const noexcept
{
  const auto bytes = [](const State& state) {
    return ((state.factors ? state.factors->values.capacity() : 0) +
            (state.inverse ? state.inverse->values.capacity() : 0)) * sizeof(DerivativeReal);
  };
  return bytes(*state_) + (pending_ ? bytes(*pending_->state) +
      pending_->inverse->storageDiagnostics().retained_numeric_bytes : 0);
}

std::string KFACOptimizer::stateFingerprint(const State& state) const
{
  std::vector<double> scalars;
  if (state.factors) scalars.push_back(state.factors->normalizer);
  if (state.inverse)
  {
    scalars.push_back(state.inverse->damping);
    scalars.push_back(state.inverse->fallback_damping);
  }
  return std::to_string(state.step) + ":" + std::to_string(state.data_seen) + ":" +
      std::to_string(state.inverse_step) + ":" +
      (state.factors ? std::to_string(state.factors->update_count) + ":" +
          TrainingCheckpoint::valuesFingerprint(state.factors->values) : "empty") + ":" +
      (state.inverse ? TrainingCheckpoint::valuesFingerprint(state.inverse->values) : "empty") + ":" +
      TrainingCheckpoint::valuesFingerprint(scalars);
}

CheckpointContributorMetadata KFACOptimizer::checkpointMetadata() const
{
  if (pending_) throw std::logic_error("Cannot checkpoint speculative KFAC state");
  return {contributor_id, {1, 0, 0}, configuration_fingerprint_, stateFingerprint(*state_)};
}

void KFACOptimizer::writeCheckpointPayload(hdf_archive& archive) const
{
  if (pending_) throw std::logic_error("Cannot checkpoint speculative KFAC state");
  archive.write(state_->step, "step");
  archive.write(state_->data_seen, "data_seen");
  archive.write(state_->inverse_step, "inverse_step");
  if (!state_->factors) return;
  archive.write(state_->factors->normalizer, "factor_normalizer");
  archive.write(state_->factors->update_count, "factor_update_count");
  archive.write(state_->factors->values, "factor_values");
  archive.write(state_->inverse->values, "inverse_values");
  archive.write(state_->inverse->damping, "inverse_damping");
  archive.write(state_->inverse->fallback_damping, "inverse_fallback_damping");
}

std::unique_ptr<OptimizerCheckpointContributor::PreparedRestore> KFACOptimizer::prepareCheckpointRestore(
    hdf_archive& archive, const CheckpointContributorMetadata& metadata)
{
  if (pending_) throw std::logic_error("Cannot restore speculative KFAC state");
  if (metadata.contributor_id != contributor_id || metadata.format_version != std::array<int, 3>{1, 0, 0} ||
      metadata.configuration_fingerprint != configuration_fingerprint_)
    throw std::runtime_error("KFAC checkpoint configuration or format differs");
  // Required fields, semantic identities and payload hashes precede any live swap.
  auto state = std::make_unique<State>();
  if (!archive.readEntry(state->step, "step") || !archive.readEntry(state->data_seen, "data_seen") ||
      !archive.readEntry(state->inverse_step, "inverse_step") ||
      state->step == std::numeric_limits<std::uint64_t>::max())
    throw std::runtime_error("KFAC checkpoint progress fields are invalid");
  if (state->step != 0)
  {
    state->factors.emplace();
    state->inverse.emplace();
    auto& factors = *state->factors;
    auto& inverse = *state->inverse;
    factors.registry_fingerprint = inverse.registry_fingerprint = registry_.fingerprint();
    factors.convention = inverse.convention = control_.convention;
    inverse.damping_policy = control_.inverse.damping_policy;
    if (!archive.readEntry(factors.normalizer, "factor_normalizer") ||
        !archive.readEntry(factors.update_count, "factor_update_count") ||
        !archive.readEntry(factors.values, "factor_values") || !archive.readEntry(inverse.values, "inverse_values") ||
        !archive.readEntry(inverse.damping, "inverse_damping") ||
        !archive.readEntry(inverse.fallback_damping, "inverse_fallback_damping"))
      throw std::runtime_error("KFAC checkpoint recurrence payload is missing");
    validateKFACFactorState(registry_, factors);
    KFACPreconditioner checked_inverse(registry_, 0, inverse);
    if (factors.update_count != state->step || state->data_seen < state->step ||
        state->inverse_step != ((state->step - 1) / control_.inverse_period) * control_.inverse_period ||
        inverse.damping != control_.inverse.damping || inverse.fallback_damping != control_.inverse.fallback_damping)
      throw std::runtime_error("KFAC checkpoint recurrence counters or damping differ");
  }
  else if (state->data_seen != 0 || state->inverse_step != 0)
    throw std::runtime_error("KFAC checkpoint has progress without accepted updates");
  if (metadata.state_fingerprint != stateFingerprint(*state))
    throw std::runtime_error("KFAC checkpoint state fingerprint mismatch");
  return std::make_unique<PreparedRestoreAction>(*this, std::move(state));
}
}
