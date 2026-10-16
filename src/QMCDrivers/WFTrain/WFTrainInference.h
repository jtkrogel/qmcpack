//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_WFTRAIN_INFERENCE_H
#define QMCPLUSPLUS_WFTRAIN_INFERENCE_H

#include "QMCDrivers/QMCDriverNew.h"
#include "QMCDrivers/WFTrain/WFTrainInferenceInput.h"
#include "QMCDrivers/VMC/VMCBatchedSampling.h"
#include "QMCDrivers/Crowd.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "Estimators/EstimatorManagerNew.h"
#include "Concurrency/ParallelExecutor.hpp"
#include "Utilities/RunTimeManager.h"
#include "Utilities/NativeTimerScope.h"
#include <fstream>

namespace qmcplusplus
{
/** Fresh synchronous VMC context. Never finalizes into production configurations/RNGs. */
class WFTrainInference : public QMCDriverNew
{
public:
  WFTrainInference(const ProjectData& project, const WFTrainInferenceInput& input,
                   WalkerConfigurations& walkers, MCPopulation&& population,
                   const RefVector<RandomBase<FullPrecRealType>>& rngs, Communicate* comm,
                   std::string prefix, std::size_t iteration, std::size_t version)
      : QMCDriverNew(project, QMCDriverInput(input.sampling), nullptr, walkers, std::move(population), rngs,
                     "WFTrainInference::", comm, "WFTrainInference"),
        input_(input), prefix_(std::move(prefix)), iteration_(iteration), version_(version)
  {}

  void process(xmlNodePtr) override
  {
    VMCBatchedSampling::validateMoveConfiguration(input_.vmc, population_.get_golden_electrons(),
                                                population_.get_golden_twf(), serializing_crowd_walkers_);
    const auto counts = adjustGlobalWalkerCount(*myComm, 0, qmcdriver_input_.get_total_walkers(),
        qmcdriver_input_.get_walkers_per_rank(), 1.0,
        determineNumCrowds(qmcdriver_input_.get_num_crowds(), rngs_.size()));
    steps_per_block_ = determineStepsPerBlock(counts.global_walkers, qmcdriver_input_.get_requested_samples(),
        qmcdriver_input_.get_requested_steps(), qmcdriver_input_.get_max_blocks());
    initPopulationAndCrowds(counts);
    for (int i = 0; i < crowds_.size(); ++i)
      contexts_.push_back(std::make_unique<ContextForSteps>(rngs_[i]));
  }

  void run() override
  {
    estimator_manager_->startDriverRun({prefix_, true});
    ParallelExecutor<> task;
    RefVector<ContextForSteps> refs;
    for (auto& context : contexts_)
      refs.push_back(*context);
    task(crowds_.size(), initialLogEvaluation, crowds_, refs, serializing_crowd_walkers_);
    VMCBatchedSampling::State state(qmcdriver_input_, input_.vmc, *drift_modifier_, population_,
                                    steps_per_block_, serializing_crowd_walkers_);
    VMCBatchedSampling::Timers timers{timers_.resource_timer, timers_.movepbyp_timer, timers_.buffer_timer,
        timers_.hamiltonian_timer, timers_.collectables_timer, timers_.estimators_timer,
        [this](Crowd& crowd, std::string_view location) {
          checkLogAndGL(crowd, location, serializing_crowd_walkers_);
        }};
    LoopTimer<> loop;
    // Do not create another RunTimeControl: its default stop-file cleanup must
    // not clear the outer training driver's stop request.
    RunTimeControl<> control(run_time_manager, project_data_.getMaxCPUSeconds(), project_data_.getTitle(), false);
    auto stopped = [&]() {
      bool stop = false;
      if (myComm->rank() == 0)
        stop = control.checkStop(loop);
      myComm->bcast(stop);
      return stop;
    };
    bool stop = stopped();
    for (int warmup = 0; warmup < qmcdriver_input_.get_warmup_steps() && !stop; ++warmup)
    {
      loop.start();
      task(crowds_.size(), VMCBatchedSampling::runWarmupStep, state, timers, rngs_, crowds_);
      loop.stop();
      stop = stopped();
    }
    int global_step = 0;
    for (int block = 0; block < qmcdriver_input_.get_max_blocks() && !stop; ++block)
    {
      loop.start();
      state.recalculate_properties_period = input_.vmc.get_move_kind() == VMCDriverInput::MoveKind::PBYP
          ? qmcdriver_input_.get_recalculate_properties_period() : 0;
      state.is_recomputing_block = qmcdriver_input_.get_blocks_between_recompute()
          ? (block + 1) % qmcdriver_input_.get_blocks_between_recompute() == 0 : false;
      estimator_manager_->startBlock(steps_per_block_);
      for (auto& crowd : crowds_)
        crowd->startBlock(steps_per_block_);
      for (int step = 0; step < steps_per_block_; ++step, ++global_step)
      {
        state.step = step;
        state.global_step = global_step;
        task(crowds_.size(), VMCBatchedSampling::runVMCStep, state, timers, rngs_, crowds_);
      }
      endBlock();
      ++completed_blocks_;
      loop.stop();
      stop = stopped();
    }
    complete_ = completed_blocks_ == qmcdriver_input_.get_max_blocks();
    estimator_manager_->stopDriverRun();
    bool failed = false;
    if (myComm->rank() == 0)
    {
      std::ofstream scalar(prefix_ + ".scalar.dat", std::ios::app);
      scalar << "# WFTrain inference iteration " << iteration_ << " parameter_version " << version_
             << " completed_blocks " << completed_blocks_ << " requested_blocks "
             << qmcdriver_input_.get_max_blocks() << " complete " << (complete_ ? "yes" : "no") << '\n';
      scalar.flush();
      failed = !scalar;
      app_log() << "WFTrain inference iteration " << iteration_ << " parameter_version " << version_
                << " completed_blocks " << completed_blocks_ << " complete " << (complete_ ? "yes" : "no") << std::endl;
    }
    myComm->bcast(failed);
    if (failed)
      throw UniformCommunicateError("WFTrain inference: failed to record scalar completion");
  }

  bool complete() const { return complete_; }

private:
  QMCRunType getRunType() override { return QMCRunType::VMC_BATCH; }
  const WFTrainInferenceInput& input_;
  std::string prefix_;
  std::size_t iteration_, version_;
  UPtrVector<ContextForSteps> contexts_;
  int completed_blocks_ = 0;
  bool complete_ = false;
};
}
#endif
