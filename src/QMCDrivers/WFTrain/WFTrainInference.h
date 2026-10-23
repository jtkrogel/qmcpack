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
#include "Message/CommOperators.h"
#include <fstream>
#include <functional>

namespace qmcplusplus
{
/** Fresh synchronous VMC context. Never finalizes into production configurations/RNGs. */
class WFTrainInference : public QMCDriverNew
{
public:
  WFTrainInference(const ProjectData& project, const WFTrainInferenceInput& input,
                   WalkerConfigurations& walkers, MCPopulation&& population,
                   const RefVector<RandomBase<FullPrecRealType>>& rngs, Communicate* comm,
                   std::string prefix, std::size_t iteration, std::size_t version,
                   std::function<void(Crowd&)> orbital_observer = {},
                   std::function<std::string()> orbital_block = {}, std::string orbital_header = {})
      : QMCDriverNew(project, QMCDriverInput(input.sampling), nullptr, walkers, std::move(population), rngs,
                     "WFTrainInference::", comm, "WFTrainInference"),
        input_(input), prefix_(std::move(prefix)), iteration_(iteration), version_(version),
        orbital_observer_(std::move(orbital_observer)), orbital_block_(std::move(orbital_block)),
        orbital_header_(std::move(orbital_header))
  {
    if (static_cast<bool>(orbital_observer_) != static_cast<bool>(orbital_block_) ||
        (!input_.evaluate_energy && !orbital_observer_))
      throw std::invalid_argument("Objective-only inference requires a complete orbital observer");
  }

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
    std::ofstream orbital_scalar;
    if (orbital_observer_)
    {
      int failed = 0;
      if (myComm->rank() == 0)
      {
        orbital_scalar.open(prefix_ + ".scalar.dat");
        orbital_scalar << "# index " << orbital_header_ << '\n';
        failed = !orbital_scalar;
      }
      myComm->bcast(failed);
      if (failed)
        throw UniformCommunicateError("WFTrain inference: cannot open orbital scalar output");
    }
    else
      estimator_manager_->startDriverRun({prefix_, true});
    ParallelExecutor<> task;
    RefVector<ContextForSteps> refs;
    for (auto& context : contexts_)
      refs.push_back(*context);
    const auto initialize = input_.evaluate_energy ? initialLogEvaluation : initialLogEvaluationWavefunctionOnly;
    task(crowds_.size(), initialize, crowds_, refs, serializing_crowd_walkers_);
    VMCBatchedSampling::State state(qmcdriver_input_, input_.vmc, *drift_modifier_, population_,
                                    steps_per_block_, serializing_crowd_walkers_);
    state.evaluate_energy = input_.evaluate_energy;
    if (input_.vmc.get_move_kind() == VMCDriverInput::MoveKind::N_ELECTRON)
      state.subset_cursors.resize(crowds_.size(), 0);
    state.accumulate_estimators = !orbital_observer_;
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
      if (!orbital_observer_)
      {
        estimator_manager_->startBlock(steps_per_block_);
        for (auto& crowd : crowds_)
          crowd->startBlock(steps_per_block_);
      }
      for (int step = 0; step < steps_per_block_; ++step, ++global_step)
      {
        state.step = step;
        state.global_step = global_step;
        task(crowds_.size(), VMCBatchedSampling::runVMCStep, state, timers, rngs_, crowds_);
        if (orbital_observer_)
        {
          // One sequential observer reuses target scratch; fail together before reductions.
          int failed = 0;
          try { for (auto& crowd : crowds_) orbital_observer_(*crowd); }
          catch (...) { failed = 1; }
          myComm->allreduce(failed);
          if (failed)
            throw UniformCommunicateError("WFTrain inference: orbital observation failed");
        }
      }
      if (orbital_observer_)
      {
        const auto row = orbital_block_(); // All ranks merge raw moments before normalization.
        int failed = 0;
        if (myComm->rank() == 0)
        {
          orbital_scalar << block << ' ' << row << '\n';
          orbital_scalar.flush();
          failed = !orbital_scalar;
        }
        myComm->bcast(failed);
        if (failed)
          throw UniformCommunicateError("WFTrain inference: failed to write orbital scalar block");
      }
      else
        endBlock();
      ++completed_blocks_;
      loop.stop();
      stop = stopped();
    }
    complete_ = completed_blocks_ == qmcdriver_input_.get_max_blocks();
    if (orbital_observer_)
      orbital_scalar.close();
    else
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
  /// Orbital-only scalar policy; ordinary energy inference retains native estimators.
  std::function<void(Crowd&)> orbital_observer_;
  std::function<std::string()> orbital_block_;
  std::string orbital_header_;
  UPtrVector<ContextForSteps> contexts_;
  int completed_blocks_ = 0;
  bool complete_ = false;
};
}
#endif
