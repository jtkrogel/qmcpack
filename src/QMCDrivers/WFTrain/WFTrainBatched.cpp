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
#include "QMCDrivers/WFTrain/FirstOrderOptimizer.h"
#include "QMCDrivers/WFTrain/HighParameterTraining.h"
#include "QMCDrivers/WFTrain/PreparedEnergyGradientProducer.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCWaveFunctions/Optimization/StructuredParameterProvider.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "Utilities/RunTimeManager.h"

#include <cmath>
#include <exception>
#include <iomanip>
#include <memory>
#include <set>
#include <string_view>
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

    if (!training_input_.getCheckpointPath().empty() || !training_input_.getRestartPath().empty())
      throw UniformCommunicateError(
          "WFTrainBatched: training checkpoint/restart is not enabled in the initial Adam route");
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
  wftrain::TrainingCapabilities requirements{
      wftrain::TrainingCapability::VALUE_EVALUATION};
  wftrain::HighParameterTraining trainer(
      requirements, wftrain::EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN,
      wftrain::DistributedParameterReduction(*myComm, reduction_policy));
  wftrain::FirstOrderOptimizer optimizer(
      selected_provider_->parameterSchema(),
      makeAdamOptions(selected_provider_->parameterSchema(), training_input_));
  wftrain::TrainingIterationState training_state;
  DirtyCacheObserver cache_observer;

  LoopTimer<> training_loop;
  RunTimeControl<> runtime_control(run_time_manager, project_data_.getMaxCPUSeconds(),
                                   project_data_.getTitle(), myComm->rank() == 0);
  ParallelExecutor<> crowd_task;

  {
    ScopedTimer initialization_timer(timers_.init_walkers_timer);
    auto step_context_refs = getContextForStepsRefs();
    crowd_task(crowds_.size(), initialLogEvaluation, crowds_, step_context_refs,
               serializing_crowd_walkers_);
    print_mem("WFTrainBatched after initialLogEvaluation", app_summary());
  }

  ScopedTimer production_timer(timers_.production_timer);
  for (int step = 0; step < qmcdriver_input_.get_warmup_steps(); ++step)
  {
    ScopedTimer step_timer(timers_.run_steps_timer);
    crowd_task(crowds_.size(), VMCBatchedSampling::runWarmupStep, sampling_state,
               sampling_timers, rngs_, crowds_);
  }
  myComm->barrier();

  int global_step = 0;
  int completed_blocks = 0;
  for (int block = 0; block < num_blocks; ++block)
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
    {
      std::unique_ptr<wftrain::PreparedEnergyGradientProducer> producer;
      std::exception_ptr preparation_failure;
      try
      {
        std::vector<wftrain::PreparedEnergyGradientBatch> batches;
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
        producer = std::make_unique<wftrain::PreparedEnergyGradientProducer>(std::move(batches));
        storage_diagnostics = producer->storageDiagnostics();
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

      iteration_result = trainer.runIteration(*selected_provider_, *producer, optimizer,
                                              training_state, &cache_observer);
    }

    // All derivative owners for the old version are gone before cache replacement.
    if (!cache_observer.dirty() ||
        cache_observer.publishedVersion() != iteration_result.parameter_version)
      throw UniformCommunicateError(
          "WFTrainBatched: parameter publication did not arm the cache-refresh barrier");
    refreshPublishedParameters(iteration_result.parameter_version);
    cache_observer.clear();

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
                << " retained_bytes " << storage_diagnostics.retained_numeric_bytes
                << std::endl;
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
