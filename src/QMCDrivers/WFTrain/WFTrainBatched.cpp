//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include "Estimators/EstimatorInputDelegates.h"
#include "WFTrainBatched.h"

#include "MemoryUsage.h"
#include "Message/UniformCommunicateError.h"
#include "QMCWaveFunctions/Optimization/StructuredParameterProvider.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"

namespace qmcplusplus
{

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

    const auto providers = population_.get_golden_twf().extractStructuredParameterProviders();
    const std::string& requested_provider = training_input_.getProviderID();
    if (requested_provider.empty())
    {
      if (providers.size() != 1)
        throw UniformCommunicateError(
            "WFTrainBatched: an omitted provider requires exactly one structured parameter provider");
      selected_provider_id_ = providers.front().get().parameterSchema().providerId();
    }
    else
    {
      for (const auto& provider : providers)
        if (provider.get().parameterSchema().providerId() == requested_provider)
        {
          selected_provider_id_ = requested_provider;
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
  throw UniformCommunicateError(
      "WFTrainBatched: energy-gradient production is not enabled in the stage-29.1/29.2 driver scaffold");
}

} // namespace qmcplusplus
