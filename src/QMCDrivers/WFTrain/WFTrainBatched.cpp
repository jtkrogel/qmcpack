//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include "Estimators/EstimatorInputDelegates.h"
#include "WFTrainBatched.h"

#include "Concurrency/ParallelExecutor.hpp"
#include "MemoryUsage.h"
#include "Message/CommOperators.h"
#include "Message/UniformCommunicateError.h"
#include "QMCDrivers/WalkerLogManager.h"
#include "QMCDrivers/WFTrain/DistributedTrainingCheckpoint.h"
#include "QMCDrivers/WFTrain/FirstOrderOptimizer.h"
#include "QMCDrivers/WFTrain/HighParameterTraining.h"
#include "QMCDrivers/WFTrain/MatrixFreeOperator.h"
#include "QMCDrivers/WFTrain/MatrixFreeStochasticReconfiguration.h"
#include "QMCDrivers/WFTrain/MinimumStepStochasticReconfiguration.h"
#include "QMCDrivers/WFTrain/PreparedEnergyGradientProducer.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"
#include "QMCDrivers/WFTrain/WFTrainSamplerCheckpoint.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCWaveFunctions/Optimization/StructuredParameterProvider.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "Utilities/RunTimeManager.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace qmcplusplus
{
namespace
{

/** Record publication without performing fallible work in the transaction callback. */
class DirtyCacheObserver final : public wftrain::ParameterUpdateObserver
{
public:
  void parametersPublished(std::size_t new_version) noexcept override
  {
    dirty_             = true;
    published_version_ = new_version;
  }

  bool dirty() const noexcept { return dirty_; }
  std::size_t publishedVersion() const noexcept { return published_version_; }
  void clear() noexcept { dirty_ = false; }

private:
  bool dirty_ = false;
  std::size_t published_version_ = 0;
};

/** Ensure a transient SPRING factor binding never outlives its prepared owners. */
class SPRINGFactorBindingGuard
{
public:
  explicit SPRINGFactorBindingGuard(wftrain::SPRINGUpdateRule& optimizer) noexcept
      : optimizer_(optimizer)
  {}

  ~SPRINGFactorBindingGuard() { optimizer_.unbindFactor(); }

  SPRINGFactorBindingGuard(const SPRINGFactorBindingGuard&)            = delete;
  SPRINGFactorBindingGuard& operator=(const SPRINGFactorBindingGuard&) = delete;

private:
  wftrain::SPRINGUpdateRule& optimizer_;
};

/** Build one Adam learning-rate entry for every trainable schema group. */
wftrain::FirstOrderOptimizerOptions makeAdamOptions(const wftrain::StructuredParameterSchema& schema,
                                                    const WFTrainDriverInput& input)
{
  wftrain::FirstOrderOptimizerOptions options;
  options.method     = wftrain::FirstOrderMethod::ADAM;
  options.adam_beta1 = input.getAdamBeta1();
  options.adam_beta2 = input.getAdamBeta2();
  options.epsilon    = input.getEpsilon();

  std::set<std::string> update_groups;
  for (const wftrain::ParameterBlockDescriptor& block : schema.blocks())
    if (block.trainable)
      update_groups.insert(block.update_group);
  for (const std::string& update_group : update_groups)
    options.learning_rates.push_back({update_group, input.getLearningRate()});
  return options;
}

/** Copy live crowd metadata and bind it to a version-checked derivative owner. */
wftrain::PreparedEnergyGradientBatch prepareCrowdBatch(
    Crowd& crowd,
    std::string_view provider_id,
    std::size_t batch_ordinal,
    std::size_t sample_offset,
    std::size_t maximum_parameter_chunk_size)
{
  const auto& twfs      = crowd.get_walker_twfs();
  const auto& electrons = crowd.get_walker_elecs();
  const auto& walkers   = crowd.get_walkers();
  RefVectorWithLeader<TrialWaveFunction> twf_batch(twfs[0], twfs);
  RefVectorWithLeader<ParticleSet> electron_batch(electrons[0], electrons);

  wftrain::PreparedEnergyGradientBatch batch;
  batch.derivative_operator = TrialWaveFunction::makeStructuredStreamingDerivativeOperator(
      twf_batch, electron_batch, provider_id, batch_ordinal, sample_offset,
      maximum_parameter_chunk_size);
  batch.weights.assign(walkers.size(), 1.0);
  batch.local_energies.reserve(walkers.size());
  for (const auto& walker : walkers)
    batch.local_energies.emplace_back(walker.get().Properties(WalkerProperties::LOCALENERGY));
  batch.local_energy_term_mask =
      wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC);
  return batch;
}

/** Return the Euclidean norm of a real parameter gradient. */
double gradientNorm(const std::vector<wftrain::DerivativeReal>& gradient)
{
  double squared_norm = 0.0;
  for (const double value : gradient)
    squared_norm += value * value;
  return std::sqrt(squared_norm);
}

/** Translate the driver input into the common bounded Krylov contract. */
wftrain::KrylovSolverControl makeKrylovControl(const WFTrainDriverInput& input)
{
  wftrain::KrylovSolverControl control;
  control.relative_tolerance             = input.getKrylovRelativeTolerance();
  control.absolute_tolerance             = input.getKrylovAbsoluteTolerance();
  control.maximum_iterations             = input.getKrylovMaximumIterations();
  control.maximum_operator_applications  = input.getKrylovMaximumOperatorApplications();
  control.maximum_seconds                = input.getKrylovMaximumSeconds();
  control.stagnation_window              = input.getKrylovStagnationWindow();
  control.minimum_relative_improvement   = input.getKrylovMinimumRelativeImprovement();
  control.curvature_tolerance            = input.getKrylovCurvatureTolerance();
  control.imaginary_tolerance            = input.getKrylovImaginaryTolerance();
  return control;
}

/** Translate typed input into the parameter-space SR update controls. */
wftrain::StochasticReconfigurationUpdateControl makeSRControl(const WFTrainDriverInput& input)
{
  wftrain::StochasticReconfigurationUpdateControl control;
  control.learning_rate            = input.getLearningRate();
  control.initial_damping          = input.getDamping();
  control.damping_multiplier       = input.getDampingMultiplier();
  control.minimum_retry_damping    = input.getMinimumRetryDamping();
  control.maximum_damping_attempts = input.getMaximumDampingAttempts();
  control.maximum_update_norm      = input.getMaximumUpdateNorm();
  control.maximum_metric_norm      = input.getMaximumMetricNorm();
  control.krylov                   = makeKrylovControl(input);
  return control;
}

/** Translate typed input into the shared sample-space minimum-step controls. */
wftrain::MinimumStepUpdateControl makeMinimumStepControl(const WFTrainDriverInput& input)
{
  wftrain::MinimumStepUpdateControl control;
  control.learning_rate       = input.getLearningRate();
  control.damping             = input.getDamping();
  control.center_shift        = input.getCenterShift();
  control.maximum_update_norm = input.getMaximumUpdateNorm();
  control.objective_tolerance = input.getObjectiveTolerance();
  control.krylov              = makeKrylovControl(input);
  return control;
}

/** Translate typed input into the persistent SPRING recurrence controls. */
wftrain::SPRINGUpdateControl makeSPRINGControl(const WFTrainDriverInput& input)
{
  wftrain::SPRINGUpdateControl control;
  control.minimum_step               = makeMinimumStepControl(input);
  control.history_decay              = input.getSpringHistoryDecay();
  control.maximum_update_norm_squared = input.getSpringMaximumUpdateNormSquared();
  return control;
}

/** Expose nonowning parameter-space SR batches while the prepared owner is live. */
std::vector<wftrain::StochasticReconfigurationBatch> makeSRBatches(
    const std::vector<wftrain::PreparedEnergyGradientBatch>& batches)
{
  std::vector<wftrain::StochasticReconfigurationBatch> result;
  result.reserve(batches.size());
  for (const auto& batch : batches)
    result.push_back({batch.derivative_operator.get(),
                      {batch.weights.data(), batch.weights.size()}});
  return result;
}

/** Expose nonowning sample-space batches while the prepared owners remain live. */
std::vector<wftrain::ScoreCovarianceBatch> makeScoreBatches(
    const std::vector<wftrain::PreparedEnergyGradientBatch>& batches)
{
  std::vector<wftrain::ScoreCovarianceBatch> result;
  result.reserve(batches.size());
  for (const auto& batch : batches)
    result.push_back({batch.derivative_operator.get(),
                      {batch.weights.data(), batch.weights.size()},
                      {batch.local_energies.data(), batch.local_energies.size()}});
  return result;
}

/** Count exact numeric storage held by raw prepared batches. */
wftrain::PreparedEnergyGradientStorageDiagnostics preparedStorageDiagnostics(
    const std::vector<wftrain::PreparedEnergyGradientBatch>& batches)
{
  wftrain::PreparedEnergyGradientStorageDiagnostics result;
  result.batch_count = batches.size();
  for (const auto& batch : batches)
  {
    if (!batch.derivative_operator)
      throw std::invalid_argument(
          "WFTrainBatched: prepared batch has no derivative operator");
    result.sample_count += batch.weights.size();
    result.derivative_owner_bytes +=
        batch.derivative_operator->storageDiagnostics().retained_numeric_bytes;
    result.sample_scalar_bytes += batch.weights.capacity() * sizeof(wftrain::DerivativeReal) +
        batch.local_energies.capacity() * sizeof(wftrain::DerivativeValue);
    if (batch.additive_local_energy_vjp)
      result.sample_scalar_bytes += batch.additive_local_energy_vjp->weighted_vjp.capacity() *
          sizeof(wftrain::DerivativeValue);
  }
  result.retained_numeric_bytes =
      result.derivative_owner_bytes + result.sample_scalar_bytes;
  return result;
}

/** Reject a rank-local memory-budget violation on every rank before evaluation. */
void enforceMemoryBudget(Communicate& communicator,
                         std::size_t retained_bytes,
                         std::size_t maximum_retained_bytes)
{
  if (maximum_retained_bytes == 0)
    return;
  int budget_exceeded = retained_bytes > maximum_retained_bytes ? 1 : 0;
  communicator.allreduce(budget_exceeded);
  if (budget_exceeded != 0)
    throw UniformCommunicateError(
        "WFTrainBatched: retained training storage exceeds maximum_retained_bytes on at least one rank");
}

/** Validate the score products before any rank enters a factor constructor collective. */
void validateScoreBatches(const wftrain::StructuredParameterSchema& schema,
                          std::size_t parameter_version,
                          const std::vector<wftrain::PreparedEnergyGradientBatch>& batches)
{
  for (const auto& batch : batches)
  {
    if (!batch.derivative_operator)
      throw std::invalid_argument("WFTrainBatched: prepared a null score derivative operator");
    const auto& derivative_operator = *batch.derivative_operator;
    const auto capabilities         = derivative_operator.capabilities();
    if (derivative_operator.parameterSchema().providerId() != schema.providerId() ||
        derivative_operator.parameterSchema().fingerprint() != schema.fingerprint() ||
        derivative_operator.parameterVersion() != parameter_version ||
        derivative_operator.sampleCount() != batch.weights.size() ||
        derivative_operator.sampleCount() != batch.local_energies.size())
      throw std::invalid_argument("WFTrainBatched: prepared score-batch identity or extent mismatch");
    if (!capabilities.supports(wftrain::DerivativeProduct::SCORE_JVP) ||
        !capabilities.supports(wftrain::DerivativeProduct::SCORE_VJP) ||
        !capabilities.supports(wftrain::DerivativeAdjoint::HERMITIAN) ||
        capabilities.parameter_scalar_domain != wftrain::ParameterScalarDomain::REAL64 ||
        capabilities.result_scalar_domain != wftrain::ParameterScalarDomain::COMPLEX128 ||
        capabilities.execution_domain != wftrain::DerivativeExecutionDomain::HOST ||
        capabilities.reduction_domain == wftrain::ReductionDomain::GLOBAL ||
        !capabilities.block_streaming || capabilities.maximum_vjp_channels == 0 ||
        capabilities.maximum_parameter_chunk_size == 0 ||
        capabilities.maximum_sample_tile_size == 0)
      throw std::invalid_argument("WFTrainBatched: prepared batch lacks bounded score products");
    if (derivative_operator.sampleOffset() >
        std::numeric_limits<std::size_t>::max() - derivative_operator.sampleCount())
      throw std::overflow_error("WFTrainBatched: prepared score-batch interval overflows size_t");
    for (std::size_t sample = 0; sample < batch.weights.size(); ++sample)
      if (!wftrain::isFiniteTrainingReal(batch.weights[sample]) || batch.weights[sample] < 0.0 ||
          !wftrain::isFiniteTrainingReal(batch.local_energies[sample].real()) ||
          !wftrain::isFiniteTrainingReal(batch.local_energies[sample].imag()) ||
          batch.local_energies[sample].imag() != 0.0)
        throw std::invalid_argument("WFTrainBatched: prepared real score samples are invalid");
  }
}

/** Complete a direct factor-gradient update with failure-atomic state publication. */
wftrain::TrainingIterationResult runScoreFactorIteration(
    wftrain::StructuredParameterProvider& provider,
    const wftrain::StructuredParameterSnapshot& parameters,
    const wftrain::ScoreCovarianceFactor& factor,
    wftrain::TrainingUpdateRule& update_rule,
    wftrain::TrainingIterationState& state,
    const wftrain::DistributedParameterReduction& reduction,
    wftrain::ParameterUpdateObserver& observer,
    wftrain::DerivativeReal weight_sum,
    wftrain::DerivativeReal energy_variance)
{
  const auto& schema = provider.parameterSchema();
  if (state.completed_iterations != 0 &&
      (state.schema_fingerprint != schema.fingerprint() ||
       state.parameter_version != parameters.version))
    throw std::invalid_argument("WFTrainBatched: training state does not match the parameter provider");
  if (state.completed_iterations == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("WFTrainBatched: training iteration count overflow");

  wftrain::TrainingIterationState pending_state;
  pending_state.completed_iterations = state.completed_iterations + 1;
  pending_state.parameter_version    = parameters.version;
  pending_state.schema_fingerprint   = schema.fingerprint();

  wftrain::EnergyGradientResult objective;
  objective.schema_fingerprint = schema.fingerprint();
  objective.parameter_version  = parameters.version;
  objective.reduction_domain   = wftrain::ReductionDomain::GLOBAL;
  objective.estimator          = wftrain::EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN;
  objective.sample_count       = factor.globalSampleVectorSize();
  objective.weight_sum         = weight_sum;
  objective.mean_energy        = factor.meanEnergy();
  objective.energy_variance    = energy_variance;
  objective.gradient           = factor.scoreCovarianceGradient();

  const std::size_t committed_version = wftrain::completeParameterUpdate(
      provider, parameters, objective.parameterGradient(), update_rule, reduction);
  pending_state.parameter_version = committed_version;
  state.completed_iterations       = pending_state.completed_iterations;
  state.parameter_version          = pending_state.parameter_version;
  state.schema_fingerprint.swap(pending_state.schema_fingerprint);
  observer.parametersPublished(committed_version);
  return {state.completed_iterations, committed_version, std::move(objective)};
}

/** Globally reduced energy moments retained for direct factor diagnostics. */
struct EnergyMoments
{
  wftrain::DerivativeReal weight_sum = 0.0;
  wftrain::DerivativeReal variance   = 0.0;
};

/** Return globally weighted energy moments for live sample-space batches. */
EnergyMoments energyMoments(
    Communicate& communicator,
    const std::vector<wftrain::PreparedEnergyGradientBatch>& batches,
    wftrain::DerivativeReal mean_energy)
{
  wftrain::DerivativeReal local_weight = 0.0;
  wftrain::DerivativeReal local_sum    = 0.0;
  for (const auto& batch : batches)
    for (std::size_t sample = 0; sample < batch.weights.size(); ++sample)
    {
      const wftrain::DerivativeReal delta = batch.local_energies[sample].real() - mean_energy;
      local_weight += batch.weights[sample];
      local_sum += batch.weights[sample] * delta * delta;
    }
  std::vector<wftrain::DerivativeReal> moments{local_weight, local_sum};
  communicator.allreduce(moments);
  if (!wftrain::isFiniteTrainingReal(moments[0]) || moments[0] <= 0.0 ||
      !wftrain::isFiniteTrainingReal(moments[1]))
    throw UniformCommunicateError(
        "WFTrainBatched: direct score-factor diagnostics require positive finite global weight");
  return {moments[0], moments[1] / moments[0]};
}

/** Return the stable diagnostic name of one live optimizer. */
const char* optimizerName(WFTrainDriverInput::Optimizer optimizer) noexcept
{
  switch (optimizer)
  {
  case WFTrainDriverInput::Optimizer::ADAM:
    return "adam";
  case WFTrainDriverInput::Optimizer::STOCHASTIC_RECONFIGURATION:
    return "sr";
  case WFTrainDriverInput::Optimizer::MINIMUM_STEP_SR:
    return "minsr";
  case WFTrainDriverInput::Optimizer::SPRING:
    return "spring";
  }
  return "unknown";
}

/// Extend the stable stage fingerprint with a trivially represented scalar.
template<class T>
void extendStageFingerprint(std::uint64_t& hash, const T& value) noexcept
{
  static_assert(std::is_trivially_copyable_v<T>);
  const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
  for (std::size_t index = 0; index < sizeof(T); ++index)
  {
    hash ^= bytes[index];
    hash *= UINT64_C(1099511628211);
  }
}

/// Extend the stable stage fingerprint with an explicitly bounded string.
void extendStageFingerprint(std::uint64_t& hash, std::string_view value) noexcept
{
  const std::size_t size = value.size();
  extendStageFingerprint(hash, size);
  for (const unsigned char byte : value)
  {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
}

/// Route owned strings through the explicit byte/string overload.
void extendStageFingerprint(std::uint64_t& hash, const std::string& value) noexcept
{
  extendStageFingerprint(hash, std::string_view(value));
}

/** Bind checkpoint continuation to training and Markov-chain semantics.
 *
 * Output cadence, memory ceilings, and the requested final block count are
 * intentionally excluded so they may be changed for a resumed run.
 */
std::string stageConfigurationFingerprint(const WFTrainDriverInput& training,
                                          const QMCDriverInput& qmc,
                                          const VMCDriverInput& vmc,
                                          const wftrain::StructuredParameterSchema& schema,
                                          std::size_t steps_per_block)
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  extendStageFingerprint(hash, schema.fingerprint());
  const auto objective = static_cast<std::underlying_type_t<WFTrainDriverInput::Objective>>(
      training.getObjective());
  const auto optimizer = static_cast<std::underlying_type_t<WFTrainDriverInput::Optimizer>>(
      training.getOptimizer());
  extendStageFingerprint(hash, objective);
  extendStageFingerprint(hash, optimizer);
  extendStageFingerprint(hash, training.getParameterChunkSize());
  extendStageFingerprint(hash, training.getLearningRate());
  extendStageFingerprint(hash, training.getAdamBeta1());
  extendStageFingerprint(hash, training.getAdamBeta2());
  extendStageFingerprint(hash, training.getEpsilon());
  extendStageFingerprint(hash, training.getDamping());
  extendStageFingerprint(hash, training.getDampingMultiplier());
  extendStageFingerprint(hash, training.getMinimumRetryDamping());
  extendStageFingerprint(hash, training.getMaximumDampingAttempts());
  extendStageFingerprint(hash, training.getMaximumUpdateNorm());
  extendStageFingerprint(hash, training.getMaximumMetricNorm());
  extendStageFingerprint(hash, training.getCenterShift());
  extendStageFingerprint(hash, training.getObjectiveTolerance());
  extendStageFingerprint(hash, training.getSpringHistoryDecay());
  extendStageFingerprint(hash, training.getSpringMaximumUpdateNormSquared());
  extendStageFingerprint(hash, training.getKrylovRelativeTolerance());
  extendStageFingerprint(hash, training.getKrylovAbsoluteTolerance());
  extendStageFingerprint(hash, training.getKrylovMaximumIterations());
  extendStageFingerprint(hash, training.getKrylovMaximumOperatorApplications());
  extendStageFingerprint(hash, training.getKrylovMaximumSeconds());
  extendStageFingerprint(hash, training.getKrylovStagnationWindow());
  extendStageFingerprint(hash, training.getKrylovMinimumRelativeImprovement());
  extendStageFingerprint(hash, training.getKrylovCurvatureTolerance());
  extendStageFingerprint(hash, training.getKrylovImaginaryTolerance());
  extendStageFingerprint(hash, qmc.get_tau());
  extendStageFingerprint(hash, qmc.get_spin_mass());
  extendStageFingerprint(hash, qmc.get_sub_steps());
  extendStageFingerprint(hash, qmc.get_warmup_steps());
  extendStageFingerprint(hash, qmc.get_recalculate_properties_period());
  extendStageFingerprint(hash, qmc.get_blocks_between_recompute());
  extendStageFingerprint(hash, steps_per_block);
  extendStageFingerprint(hash, vmc.get_use_drift());
  const auto move_kind = static_cast<std::underlying_type_t<VMCDriverInput::MoveKind>>(
      vmc.get_move_kind());
  const auto selection = static_cast<std::underlying_type_t<VMCDriverInput::ElectronSelection>>(
      vmc.get_electron_selection());
  extendStageFingerprint(hash, move_kind);
  extendStageFingerprint(hash, vmc.get_electrons_per_move());
  extendStageFingerprint(hash, selection);
  std::ostringstream text;
  text << std::hex << std::setw(16) << std::setfill('0') << hash;
  return text.str();
}

} // namespace

WFTrainBatched::WFTrainBatched(const ProjectData& project_data,
                               QMCDriverInput&& qmcdriver_input,
                               UPtr<EstimatorManagerNew>&& estimator_manager,
                               VMCDriverInput&& vmc_input,
                               WFTrainDriverInput&& training_input,
                               WalkerConfigurations& walker_configurations,
                               MCPopulation&& population,
                               const RefVector<RandomBase<FullPrecRealType>>& random_generators,
                               Communicate* communicator)
    : QMCDriverNew(project_data,
                   std::move(qmcdriver_input),
                   std::move(estimator_manager),
                   walker_configurations,
                   std::move(population),
                   random_generators,
                   "WFTrainBatched::",
                   communicator,
                   "WFTrainBatched"),
      vmcdriver_input_(std::move(vmc_input)),
      training_input_(std::move(training_input))
{}

void WFTrainBatched::process(xmlNodePtr node)
{
  ScopedTimer startup_timer(timers_.startup_timer);
  print_mem("WFTrainBatched before initialization", app_log());

  try
  {
    VMCBatchedSampling::validateMoveConfiguration(vmcdriver_input_, population_.get_golden_electrons(),
                                                  population_.get_golden_twf(), serializing_crowd_walkers_);

    if (population_.get_golden_electrons().isSpinor())
      throw UniformCommunicateError("WFTrainBatched: spinor training is not supported");
    validateHamiltonianForTraining();

    const auto providers = population_.get_golden_twf().extractStructuredParameterProviders();
    const std::string& requested_provider = training_input_.getProviderID();
    if (requested_provider.empty())
    {
      if (providers.size() != 1)
        throw UniformCommunicateError(
            "WFTrainBatched: an omitted provider requires exactly one structured parameter provider");
      selected_provider_id_ = providers.front().get().parameterSchema().providerId();
      selected_provider_    = &providers.front().get();
    }
    else
    {
      for (const auto& provider : providers)
        if (provider.get().parameterSchema().providerId() == requested_provider)
        {
          selected_provider_id_ = requested_provider;
          selected_provider_    = &provider.get();
          break;
        }
      if (selected_provider_id_.empty())
        throw UniformCommunicateError("WFTrainBatched: no structured parameter provider named " +
                                      requested_provider);
    }

    const AdjustedWalkerCounts walker_counts = adjustGlobalWalkerCount(
        *myComm, walker_configs_ref_.getActiveWalkers(), qmcdriver_input_.get_total_walkers(),
        qmcdriver_input_.get_walkers_per_rank(), 1.0,
        determineNumCrowds(qmcdriver_input_.get_num_crowds(), rngs_.size()));
    steps_per_block_ = determineStepsPerBlock(walker_counts.global_walkers, qmcdriver_input_.get_requested_samples(),
                                              qmcdriver_input_.get_requested_steps(),
                                              qmcdriver_input_.get_max_blocks());
    initPopulationAndCrowds(walker_counts);
    createStepContexts(crowds_.size());
  }
  catch (const UniformCommunicateError& error)
  {
    myComm->barrier_and_abort(error.what());
  }

  if (qmcdriver_input_.get_measure_imbalance())
    measureImbalance("Startup");
}

void WFTrainBatched::run()
{
  if (selected_provider_ == nullptr)
    throw UniformCommunicateError("WFTrainBatched: process() must select a provider before run()");

  const IndexType num_blocks = qmcdriver_input_.get_max_blocks();
  estimator_manager_->startDriverRun();

  WalkerLogManager wlog_manager(walker_logs_input, allow_walker_logs, get_root_name(), myComm);
  for (auto& crowd : crowds_)
    crowd->setWalkerLogCollector(wlog_manager.makeCollector());
  wlog_manager.startRun(Crowd::getWalkerLogCollectorRefs(crowds_));

  VMCBatchedSampling::State sampling_state(qmcdriver_input_, vmcdriver_input_, *drift_modifier_, population_,
                                            steps_per_block_, serializing_crowd_walkers_);
  VMCBatchedSampling::Timers sampling_timers{
      timers_.resource_timer, timers_.movepbyp_timer, timers_.buffer_timer,
      timers_.hamiltonian_timer, timers_.collectables_timer, timers_.estimators_timer,
      [this](Crowd& crowd, std::string_view location) {
        checkLogAndGL(crowd, location, serializing_crowd_walkers_);
      }};

  wftrain::DistributedReductionPolicy reduction_policy;
  reduction_policy.maximum_chunk_size = training_input_.getParameterChunkSize();
  wftrain::DistributedParameterReduction reduction(*myComm, reduction_policy);
  wftrain::TrainingCapabilities requirements{
      wftrain::TrainingCapability::VALUE_EVALUATION};
  wftrain::HighParameterTraining trainer(
      requirements, wftrain::EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN,
      reduction);
  const auto initial_parameters = selected_provider_->snapshotParameters();
  if (initial_parameters.schema_fingerprint != selected_provider_->parameterSchema().fingerprint() ||
      initial_parameters.values.size() != selected_provider_->parameterSchema().parameterCount())
    throw UniformCommunicateError(
        "WFTrainBatched: selected provider returned an inconsistent initial snapshot");
  std::unique_ptr<wftrain::FirstOrderOptimizer> adam_optimizer;
  if (training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::ADAM)
    adam_optimizer = std::make_unique<wftrain::FirstOrderOptimizer>(
        selected_provider_->parameterSchema(),
        makeAdamOptions(selected_provider_->parameterSchema(), training_input_));
  std::unique_ptr<wftrain::SPRINGUpdateRule> spring_optimizer;
  if (training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::SPRING)
    spring_optimizer = std::make_unique<wftrain::SPRINGUpdateRule>(
        selected_provider_->parameterSchema(), makeSPRINGControl(training_input_));
  wftrain::OptimizerCheckpointContributor* optimizer_checkpoint = nullptr;
  if (adam_optimizer)
    optimizer_checkpoint = adam_optimizer.get();
  else if (spring_optimizer)
    optimizer_checkpoint = spring_optimizer.get();

  wftrain::TrainingIterationState training_state{
      0, initial_parameters.version, selected_provider_->parameterSchema().fingerprint()};
  wftrain::TrainingStageState stage_state{
      "all_electron_energy", 0, 0,
      stageConfigurationFingerprint(training_input_, qmcdriver_input_, vmcdriver_input_,
                                    selected_provider_->parameterSchema(), steps_per_block_)};
  DirtyCacheObserver cache_observer;

  LoopTimer<> training_loop;
  RunTimeControl<> runtime_control(run_time_manager, project_data_.getMaxCPUSeconds(),
                                   project_data_.getTitle(), myComm->rank() == 0);
  ParallelExecutor<> crowd_task;

  int global_step = 0;
  wftrain::WFTrainSamplerCheckpoint sampler_checkpoint(
      population_, rngs_, global_step, static_cast<std::size_t>(myComm->size()),
      static_cast<std::size_t>(myComm->rank()), crowds_.size());
  const bool resumed = !training_input_.getRestartPath().empty();
  if (resumed)
  {
    wftrain::DistributedTrainingCheckpoint::restore(
        training_input_.getRestartPath(), *myComm, *selected_provider_, training_state,
        stage_state, optimizer_checkpoint, sampler_checkpoint, sampler_checkpoint.topology());
    if (stage_state.completed_stage_iterations != training_state.completed_iterations ||
        stage_state.completed_stage_iterations > static_cast<std::size_t>(num_blocks))
      throw UniformCommunicateError(
          "WFTrainBatched: restored training progress is inconsistent with requested blocks");
    if (myComm->rank() == 0)
      app_log() << "WFTrain restored " << stage_state.completed_stage_iterations
                << " completed update(s) from " << training_input_.getRestartPath()
                << std::endl;
  }

  {
    ScopedTimer initialization_timer(timers_.init_walkers_timer);
    auto step_context_refs = getContextForStepsRefs();
    crowd_task(crowds_.size(), initialLogEvaluation, crowds_, step_context_refs,
               serializing_crowd_walkers_);
    print_mem("WFTrainBatched after initialLogEvaluation", app_summary());
  }

  ScopedTimer production_timer(timers_.production_timer);
  if (!resumed)
    for (int step = 0; step < qmcdriver_input_.get_warmup_steps(); ++step)
    {
      ScopedTimer step_timer(timers_.run_steps_timer);
      crowd_task(crowds_.size(), VMCBatchedSampling::runWarmupStep, sampling_state,
                 sampling_timers, rngs_, crowds_);
    }
  myComm->barrier();

  int completed_blocks = 0;
  for (int block = static_cast<int>(stage_state.completed_stage_iterations);
       block < num_blocks; ++block)
  {
    ScopeGuard<LoopTimer<>> block_timer(training_loop);
    sampling_state.recalculate_properties_period =
        qmc_driver_mode_[QMC_UPDATE_MODE]
        ? qmcdriver_input_.get_recalculate_properties_period()
        : 0;
    sampling_state.is_recomputing_block = qmcdriver_input_.get_blocks_between_recompute()
        ? (1 + block) % qmcdriver_input_.get_blocks_between_recompute() == 0
        : false;

    estimator_manager_->startBlock(steps_per_block_);
    for (auto& crowd : crowds_)
      crowd->startBlock(steps_per_block_);

    for (int step = 0; step < steps_per_block_; ++step, ++global_step)
    {
      ScopedTimer step_timer(timers_.run_steps_timer);
      sampling_state.step        = step;
      sampling_state.global_step = global_step;
      crowd_task(crowds_.size(), VMCBatchedSampling::runVMCStep, sampling_state,
                 sampling_timers, rngs_, crowds_);
    }

    // Finalize estimator statistics while they still describe the sampled version.
    endBlock();

    wftrain::TrainingIterationResult iteration_result;
    wftrain::PreparedEnergyGradientStorageDiagnostics storage_diagnostics;
    std::size_t retained_numeric_bytes = 0;
    std::optional<std::size_t> krylov_iterations;
    std::optional<std::size_t> krylov_operator_applications;
    std::optional<wftrain::DerivativeReal> applied_scale;
    {
      std::vector<wftrain::PreparedEnergyGradientBatch> batches;
      std::exception_ptr preparation_failure;
      try
      {
        batches.reserve(crowds_.size());
        std::size_t sample_offset = 0;
        std::size_t batch_ordinal = 0;
        for (auto& crowd : crowds_)
          if (crowd->size() != 0)
          {
            batches.push_back(prepareCrowdBatch(
                *crowd, selected_provider_id_, batch_ordinal++, sample_offset,
                training_input_.getParameterChunkSize()));
            sample_offset += static_cast<std::size_t>(crowd->size());
          }
        storage_diagnostics = preparedStorageDiagnostics(batches);
      }
      catch (...)
      {
        preparation_failure = std::current_exception();
      }

      int preparation_failed = preparation_failure ? 1 : 0;
      myComm->allreduce(preparation_failed);
      if (preparation_failed != 0)
      {
        if (preparation_failure)
          std::rethrow_exception(preparation_failure);
        throw UniformCommunicateError(
            "WFTrainBatched: another rank failed while preparing a live training batch");
      }

      const auto optimizer_kind = training_input_.getOptimizer();
      if (optimizer_kind == WFTrainDriverInput::Optimizer::ADAM ||
          optimizer_kind == WFTrainDriverInput::Optimizer::STOCHASTIC_RECONFIGURATION)
      {
        int empty_rank = batches.empty() ? 1 : 0;
        myComm->allreduce(empty_rank);
        if (empty_rank != 0)
          throw UniformCommunicateError(
              "WFTrainBatched: Adam and parameter-space SR require at least one walker on every rank");
      }
      if (optimizer_kind == WFTrainDriverInput::Optimizer::ADAM)
      {
        std::unique_ptr<wftrain::PreparedEnergyGradientProducer> producer;
        std::exception_ptr producer_failure;
        try
        {
          producer = std::make_unique<wftrain::PreparedEnergyGradientProducer>(
              std::move(batches));
        }
        catch (...)
        {
          producer_failure = std::current_exception();
        }
        int producer_failed = producer_failure ? 1 : 0;
        myComm->allreduce(producer_failed);
        if (producer_failed != 0)
        {
          if (producer_failure)
            std::rethrow_exception(producer_failure);
          throw UniformCommunicateError(
              "WFTrainBatched: another rank failed while preparing the Adam objective");
        }
        storage_diagnostics = producer->storageDiagnostics();
        retained_numeric_bytes = storage_diagnostics.retained_numeric_bytes +
            adam_optimizer->retainedBytes();
        enforceMemoryBudget(*myComm, retained_numeric_bytes,
                            training_input_.getMaximumRetainedBytes());
        iteration_result = trainer.runIteration(*selected_provider_, *producer,
                                                *adam_optimizer, training_state,
                                                &cache_observer);
      }
      else if (optimizer_kind ==
               WFTrainDriverInput::Optimizer::STOCHASTIC_RECONFIGURATION)
      {
        std::unique_ptr<wftrain::PreparedEnergyGradientProducer> producer;
        std::vector<wftrain::StochasticReconfigurationBatch> sr_batches;
        std::unique_ptr<wftrain::StochasticReconfigurationOperator> covariance;
        std::unique_ptr<wftrain::IdentityPreconditioner> preconditioner;
        std::unique_ptr<wftrain::StochasticReconfigurationUpdateRule> update_rule;
        std::exception_ptr sr_preparation_failure;
        try
        {
          producer = std::make_unique<wftrain::PreparedEnergyGradientProducer>(
              std::move(batches));
          sr_batches = makeSRBatches(producer->batches());
          const auto parameters = selected_provider_->snapshotParameters();
          covariance = std::make_unique<wftrain::StochasticReconfigurationOperator>(
              selected_provider_->parameterSchema(), parameters.version,
              wftrain::DerivativeArrayView<const wftrain::StochasticReconfigurationBatch>(
                  sr_batches.data(), sr_batches.size()),
              reduction);
          preconditioner = std::make_unique<wftrain::IdentityPreconditioner>(
              selected_provider_->parameterSchema(), parameters.version);
          update_rule = std::make_unique<wftrain::StochasticReconfigurationUpdateRule>(
              *covariance, *preconditioner, makeSRControl(training_input_));
        }
        catch (...)
        {
          sr_preparation_failure = std::current_exception();
        }
        int sr_preparation_failed = sr_preparation_failure ? 1 : 0;
        myComm->allreduce(sr_preparation_failed);
        if (sr_preparation_failed != 0)
        {
          if (sr_preparation_failure)
            std::rethrow_exception(sr_preparation_failure);
          throw UniformCommunicateError(
              "WFTrainBatched: another rank failed while preparing parameter-space SR");
        }

        storage_diagnostics = producer->storageDiagnostics();
        retained_numeric_bytes = storage_diagnostics.retained_numeric_bytes +
            covariance->storageDiagnostics().retained_numeric_bytes;
        enforceMemoryBudget(*myComm, retained_numeric_bytes,
                            training_input_.getMaximumRetainedBytes());
        iteration_result = trainer.runIteration(*selected_provider_, *producer,
                                                *update_rule, training_state,
                                                &cache_observer);
        if (const auto& diagnostics = update_rule->lastDiagnostics())
        {
          krylov_iterations            = diagnostics->solve.iterations;
          krylov_operator_applications = diagnostics->solve.operator_applications;
          applied_scale                = diagnostics->applied_scale;
        }
      }
      else
      {
        const auto& schema = selected_provider_->parameterSchema();
        std::optional<wftrain::StructuredParameterSnapshot> parameters;
        std::exception_ptr factor_preflight_failure;
        try
        {
          parameters.emplace(selected_provider_->snapshotParameters());
          if (parameters->schema_fingerprint != schema.fingerprint() ||
              parameters->values.size() != schema.parameterCount())
            throw std::logic_error(
                "WFTrainBatched: structured provider returned an inconsistent snapshot");
          if (training_state.completed_iterations != 0 &&
              (training_state.schema_fingerprint != schema.fingerprint() ||
               training_state.parameter_version != parameters->version))
            throw std::logic_error(
                "WFTrainBatched: training state does not match the parameter provider");
          validateScoreBatches(schema, parameters->version, batches);
        }
        catch (...)
        {
          factor_preflight_failure = std::current_exception();
        }
        reduction.preflight(schema, parameters ? &*parameters : nullptr,
                            wftrain::EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN,
                            factor_preflight_failure);

        // The distributed sample-space factor deliberately accepts an empty local
        // batch list; zero-walker ranks still execute every global reduction.
        const auto score_batches = makeScoreBatches(batches);
        wftrain::ScoreCovarianceFactor factor(
            schema, parameters->version,
            {score_batches.data(), score_batches.size()},
            wftrain::ScoreSampleDomain::REAL, reduction);
        std::optional<SPRINGFactorBindingGuard> spring_factor_binding;
        if (optimizer_kind == WFTrainDriverInput::Optimizer::SPRING)
        {
          spring_optimizer->bindFactor(factor);
          spring_factor_binding.emplace(*spring_optimizer);
        }
        const auto factor_storage = factor.storageDiagnostics();
        retained_numeric_bytes = storage_diagnostics.retained_numeric_bytes +
            factor_storage.retained_numeric_bytes;
        if (spring_optimizer)
          retained_numeric_bytes += spring_optimizer->retainedBytes();
        enforceMemoryBudget(*myComm, retained_numeric_bytes,
                            training_input_.getMaximumRetainedBytes());
        const EnergyMoments energy_moments =
            energyMoments(*myComm, batches, factor.meanEnergy().real());

        if (optimizer_kind == WFTrainDriverInput::Optimizer::MINIMUM_STEP_SR)
        {
          wftrain::MinimumStepSRUpdateRule update_rule(
              factor, makeMinimumStepControl(training_input_));
          iteration_result = runScoreFactorIteration(
              *selected_provider_, *parameters, factor, update_rule,
              training_state, reduction, cache_observer,
              energy_moments.weight_sum, energy_moments.variance);
          if (const auto& diagnostics = update_rule.lastDiagnostics())
          {
            krylov_iterations            = diagnostics->solve.iterations;
            krylov_operator_applications = diagnostics->solve.operator_applications;
            applied_scale                = diagnostics->applied_scale;
          }
        }
        else
        {
          iteration_result = runScoreFactorIteration(
              *selected_provider_, *parameters, factor, *spring_optimizer,
              training_state, reduction, cache_observer,
              energy_moments.weight_sum, energy_moments.variance);
          if (const auto& diagnostics = spring_optimizer->lastDiagnostics())
          {
            krylov_iterations            = diagnostics->solve.iterations;
            krylov_operator_applications = diagnostics->solve.operator_applications;
            applied_scale                = diagnostics->applied_scale;
          }
        }
      }
    }

    // All derivative owners for the old version are gone before cache replacement.
    if (!cache_observer.dirty() ||
        cache_observer.publishedVersion() != iteration_result.parameter_version)
      throw UniformCommunicateError(
          "WFTrainBatched: parameter publication did not arm the cache-refresh barrier");
    refreshPublishedParameters(iteration_result.parameter_version);
    cache_observer.clear();
    stage_state.completed_stage_iterations = iteration_result.completed_iteration;

    if (myComm->rank() == 0)
    {
      const auto& objective = iteration_result.objective;
      app_log() << std::setprecision(12)
                << "WFTrain iteration " << iteration_result.completed_iteration
                << " version " << iteration_result.parameter_version
                << " samples " << objective.sample_count
                << " mean_energy " << std::real(objective.mean_energy)
                << " variance " << objective.energy_variance
                << " gradient_norm " << gradientNorm(objective.gradient)
                << " retained_bytes " << retained_numeric_bytes
                << " optimizer " << optimizerName(training_input_.getOptimizer());
      if (krylov_iterations)
        app_log() << " krylov_iterations " << *krylov_iterations
                  << " krylov_operator_applications "
                  << *krylov_operator_applications
                  << " applied_scale " << *applied_scale;
      app_log() << std::endl;
    }

    wlog_manager.writeBuffers();
    recordBlock(block);
    ++completed_blocks;

    bool stop_requested = false;
    if (myComm->rank() == 0)
      stop_requested = runtime_control.checkStop(training_loop);
    myComm->bcast(stop_requested);
    if (myComm->rank() == 0)
      app_log() << runtime_control.generateProgressMessage("WFTrainBatched", block, num_blocks);
    const int checkpoint_period = qmcdriver_input_.get_check_point_period();
    const bool periodic_checkpoint =
        checkpoint_period > 0 && (block + 1) % checkpoint_period == 0;
    const bool final_block = block + 1 == num_blocks;
    if (!training_input_.getCheckpointPath().empty() &&
        (periodic_checkpoint || final_block || stop_requested))
    {
      ScopedTimer checkpoint_timer(timers_.checkpoint_timer);
      wftrain::DistributedTrainingCheckpoint::saveAtomic(
          training_input_.getCheckpointPath(), *myComm, *selected_provider_,
          training_state, stage_state, optimizer_checkpoint, sampler_checkpoint,
          sampler_checkpoint.topology());
      if (myComm->rank() == 0)
        app_log() << "WFTrain checkpointed " << stage_state.completed_stage_iterations
                  << " completed update(s) to " << training_input_.getCheckpointPath()
                  << std::endl;
    }
    if (stop_requested)
    {
      if (myComm->rank() == 0)
        app_log() << runtime_control.generateStopMessage("WFTrainBatched", block);
      run_time_manager.markStop();
      break;
    }
  }

  if (myComm->rank() == 0)
    app_log() << "WFTrain completed " << completed_blocks << " block(s) and "
              << training_state.completed_iterations << " committed update(s)" << std::endl;
  print_mem("WFTrainBatched ends", app_log());
  wlog_manager.stopRun();
  estimator_manager_->stopDriverRun();
  finalize(completed_blocks, true);
}

RefVector<QMCDriverNew::ContextForSteps> WFTrainBatched::getContextForStepsRefs() const
{
  RefVector<ContextForSteps> refs;
  refs.reserve(step_contexts_.size());
  for (auto& context : step_contexts_)
    refs.push_back(*context);
  return refs;
}

void WFTrainBatched::createStepContexts(int num_crowds)
{
  assert(num_crowds <= rngs_.size());
  step_contexts_.resize(num_crowds);
  for (int crowd = 0; crowd < num_crowds; ++crowd)
    step_contexts_[crowd] = std::make_unique<ContextForSteps>(rngs_[crowd]);
}

void WFTrainBatched::validateHamiltonianForTraining()
{
  QMCHamiltonian& hamiltonian = population_.get_golden_hamiltonian();
  if (hamiltonian.hasPhysicalNLPP())
    throw UniformCommunicateError(
        "WFTrainBatched: nonlocal pseudopotentials are not supported by the initial all-electron route");
  if (hamiltonian.has_L2())
    throw UniformCommunicateError(
        "WFTrainBatched: L2 Hamiltonian terms are not supported by the initial all-electron route");

  for (int component = 0; component < hamiltonian.size(); ++component)
    if (hamiltonian.getComponent(component)->getClassName().find("ECP") != std::string::npos)
      throw UniformCommunicateError(
          "WFTrainBatched: ECP Hamiltonian terms are not supported by the initial all-electron route");

  const RefVector<OperatorBase> dependent_components = hamiltonian.getTWFDependentComponents();
  if (dependent_components.size() != 1 ||
      dependent_components.front().get().getClassName() != "BareKineticEnergy")
    throw UniformCommunicateError(
        "WFTrainBatched: the initial route requires BareKineticEnergy as its sole "
        "wavefunction-dependent Hamiltonian term");
}

void WFTrainBatched::refreshPublishedParameters(std::size_t parameter_version)
{
  // Publication modifies shared model state. Fence ranks before any clone evaluates it.
  myComm->barrier();

  std::exception_ptr refresh_failure;
  try
  {
    ParallelExecutor<> refresh_task;
    auto step_context_refs = getContextForStepsRefs();
    refresh_task(crowds_.size(), initialLogEvaluation, crowds_, step_context_refs,
                 serializing_crowd_walkers_);
  }
  catch (...)
  {
    refresh_failure = std::current_exception();
  }

  int refresh_failed = refresh_failure ? 1 : 0;
  myComm->allreduce(refresh_failed);
  if (refresh_failed != 0)
  {
    if (refresh_failure)
      std::rethrow_exception(refresh_failure);
    throw UniformCommunicateError(
        "WFTrainBatched: another rank failed while refreshing published parameter version " +
        std::to_string(parameter_version));
  }
  if (myComm->rank() == 0)
    app_log() << "WFTrain refreshed parameter version " << parameter_version
              << " across " << crowds_.size() << " crowd(s)" << std::endl;
}

} // namespace qmcplusplus
