//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2020 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File refactored from: VMC.cpp
//////////////////////////////////////////////////////////////////////////////////////

#include "VMCBatched.h"
#include "EstimatorInputDelegates.h"
#include "Concurrency/ParallelExecutor.hpp"
#include "Concurrency/Info.hpp"
#include "Message/UniformCommunicateError.h"
#include "Message/CommOperators.h"
#include "TrialWaveFunction.h"
#include "QMCWaveFunctions/Fermion/MultiSlaterDetTableMethod.h"
#include "Utilities/RunTimeManager.h"
#include "Particle/MCSample.h"
#include "MemoryUsage.h"
#include "WalkerLogManager.h"

namespace qmcplusplus
{
/** Constructor maintains proper ownership of input parameters
   */
VMCBatched::VMCBatched(const ProjectData& project_data,
                       QMCDriverInput&& qmcdriver_input,
                       UPtr<EstimatorManagerNew>&& estimator_manager,
                       VMCDriverInput&& input,
                       WalkerConfigurations& wc,
                       MCPopulation&& pop,
                       const RefVector<RandomBase<FullPrecRealType>>& rng_refs,
                       SampleStack& samples,
                       Communicate* comm,
                       bool imported_md_orbital_ecp)
    : QMCDriverNew(project_data,
                   std::move(qmcdriver_input),
                   std::move(estimator_manager),
                   wc,
                   std::move(pop),
                   rng_refs,
                   "VMCBatched::",
                   comm,
                   "VMCBatched"),
      vmcdriver_input_(input),
      imported_md_orbital_ecp_(imported_md_orbital_ecp),
      samples_(samples),
      collect_samples_(false)
{}


void VMCBatched::process(xmlNodePtr node)
{
  ScopedTimer local_timer(timers_.startup_timer);
  print_mem("VMCBatched before initialization", app_log());

  try
  {
    VMCBatchedSampling::validateMoveConfiguration(vmcdriver_input_, population_.get_golden_electrons(),
                                                  population_.get_golden_twf(), serializing_crowd_walkers_);

    QMCDriverNew::AdjustedWalkerCounts awc =
        adjustGlobalWalkerCount(*myComm, walker_configs_ref_.getActiveWalkers(), qmcdriver_input_.get_total_walkers(),
                                qmcdriver_input_.get_walkers_per_rank(), 1.0,
                                determineNumCrowds(qmcdriver_input_.get_num_crowds(), rngs_.size()));

    steps_per_block_ =
        determineStepsPerBlock(awc.global_walkers, qmcdriver_input_.get_requested_samples(),
                               qmcdriver_input_.get_requested_steps(), qmcdriver_input_.get_max_blocks());

    initPopulationAndCrowds(awc);
    if (imported_md_orbital_ecp_)
      for (auto& wavefunction : population_.get_twfs())
      {
        if (wavefunction.get() == &population_.get_golden_twf())
          throw std::logic_error("Imported orbital ECP population aliases its external golden wavefunction");
        for (auto& determinant : wavefunction->findMSD())
          determinant.get().enableAcceptedReferenceOrbitalVirtualRatios();
      }
    createStepContexts(crowds_.size());
  }
  catch (const UniformCommunicateError& ue)
  {
    myComm->barrier_and_abort(ue.what());
  }

  if (qmcdriver_input_.get_measure_imbalance())
    measureImbalance("Startup");
}

size_t VMCBatched::compute_samples_per_rank(const size_t num_blocks,
                                            const size_t samples_per_block,
                                            const size_t local_walkers)
{
  return num_blocks * samples_per_block * local_walkers;
}


/** Runs the actual VMC section
 *
 *  Dependent on base class state machine
 *  Assumes state already updated from the following calls:
 *  1. QMCDriverNew::setStatus
 *  2. QMCDriverNew::putWalkers
 *  3. QMCDriverNew::process
 *
 *  At the moment I don't care about 1st touch, prove it matters
 *  If does consider giving more to the thread by value that should
 *  end up thread local. (I think)
 */
void VMCBatched::run()
{
  IndexType num_blocks = qmcdriver_input_.get_max_blocks();
  //start the main estimator
  estimator_manager_->startDriverRun();

  //initialize WalkerLogManager and collectors
  WalkerLogManager wlog_manager(walker_logs_input, allow_walker_logs, get_root_name(), myComm);
  for (auto& crowd : crowds_)
    crowd->setWalkerLogCollector(wlog_manager.makeCollector());
  //register walker log collectors into the manager
  wlog_manager.startRun(Crowd::getWalkerLogCollectorRefs(crowds_));

  StateForThread vmc_state(qmcdriver_input_, vmcdriver_input_, *drift_modifier_, population_, steps_per_block_,
                           serializing_crowd_walkers_);
  VMCBatchedSampling::Timers sampling_timers{
      timers_.resource_timer,     timers_.movepbyp_timer,    timers_.buffer_timer,
      timers_.hamiltonian_timer,  timers_.collectables_timer, timers_.estimators_timer,
      [this](Crowd& crowd, std::string_view location) {
        checkLogAndGL(crowd, location, serializing_crowd_walkers_);
      }};

  LoopTimer<> vmc_loop;
  RunTimeControl<> runtimeControl(run_time_manager, project_data_.getMaxCPUSeconds(), project_data_.getTitle(),
                                  myComm->rank() == 0);

  { // walker initialization
    ScopedTimer local_timer(timers_.init_walkers_timer);
    ParallelExecutor<> section_start_task;
    auto step_contexts_refs = getContextForStepsRefs();
    section_start_task(crowds_.size(), initialLogEvaluation, crowds_, step_contexts_refs, serializing_crowd_walkers_);
    print_mem("VMCBatched after initialLogEvaluation", app_summary());
    if (qmcdriver_input_.get_measure_imbalance())
      measureImbalance("InitialLogEvaluation");
  }

  ScopedTimer local_timer(timers_.production_timer);
  ParallelExecutor<> crowd_task;

  if (qmcdriver_input_.get_warmup_steps() > 0)
  {
    // Run warm-up steps
    Timer warmup_timer;
    for (int step = 0; step < qmcdriver_input_.get_warmup_steps(); ++step)
    {
      ScopedTimer local_timer(timers_.run_steps_timer);
      crowd_task(crowds_.size(), VMCBatchedSampling::runWarmupStep, vmc_state, sampling_timers, rngs_, crowds_);
    }

    print_mem("VMCBatched after Warmup", app_log());
    if (qmcdriver_input_.get_measure_imbalance())
      measureImbalance("Warmup");

    app_log() << "VMC Warmup completed in " << std::setprecision(4) << warmup_timer.elapsed() << " secs" << std::endl;
  }

  // this barrier fences all previous load imbalance. Avoid block 0 timing pollution.
  myComm->barrier();

  int global_step = 0;
  for (int block = 0; block < num_blocks; ++block)
  {
    {
      ScopeGuard<LoopTimer<>> vmc_local_timer(vmc_loop);
      vmc_state.recalculate_properties_period =
          (qmc_driver_mode_[QMC_UPDATE_MODE]) ? qmcdriver_input_.get_recalculate_properties_period() : 0;
      vmc_state.is_recomputing_block = qmcdriver_input_.get_blocks_between_recompute()
          ? (1 + block) % qmcdriver_input_.get_blocks_between_recompute() == 0
          : false;

      estimator_manager_->startBlock(steps_per_block_);

      for (auto& crowd : crowds_)
        crowd->startBlock(steps_per_block_);

      for (int step = 0; step < steps_per_block_; ++step, ++global_step)
      {
        ScopedTimer local_timer(timers_.run_steps_timer);
        vmc_state.step        = step;
        vmc_state.global_step = global_step;
        crowd_task(crowds_.size(), VMCBatchedSampling::runVMCStep, vmc_state, sampling_timers, rngs_, crowds_);

        if (collect_samples_)
        {
          const auto& elec_psets = population_.get_elec_particle_sets();
          for (const auto& walker : elec_psets)
          {
            samples_.appendSample(MCSample(*walker));
          }
        }
      }

      print_mem("VMCBatched after a block", app_debug_stream());
      if (qmcdriver_input_.get_measure_imbalance())
        measureImbalance("Block " + std::to_string(block));
      endBlock();
      wlog_manager.writeBuffers();
      recordBlock(block);
    }

    bool stop_requested = false;
    // Rank 0 decides whether the time limit was reached
    if (!myComm->rank())
      stop_requested = runtimeControl.checkStop(vmc_loop);
    myComm->bcast(stop_requested);
    // Progress messages before possibly stopping
    if (!myComm->rank())
      app_log() << runtimeControl.generateProgressMessage("VMCBatched", block, num_blocks);
    if (stop_requested)
    {
      if (!myComm->rank())
        app_log() << runtimeControl.generateStopMessage("VMCBatched", block);
      run_time_manager.markStop();
      break;
    }
  }
  // This is confusing logic from VMC.cpp want this functionality write documentation of this
  // and clean it up
  // bool wrotesamples = qmcdriver_input_.get_dump_config();
  // if (qmcdriver_input_.get_dump_config())
  // {
  //wrotesamples = W.dumpEnsemble(wClones, wOut, myComm->size(), nBlocks);
  //if (wrotesamples)
  //  app_log() << "  samples are written to the config.h5" << std::endl;
  // }

  // second argument was !wrotesample so if W.dumpEnsemble returns false or
  // dump_config is false from input then dump_walkers
  {
    std::ostringstream o;
    FullPrecRealType ene, var;
    estimator_manager_->getApproximateEnergyVariance(ene, var);
    o << "====================================================";
    o << "\n  End of a VMC section";
    o << "\n    QMC counter        = " << project_data_.getSeriesIndex();
    o << "\n    time step          = " << qmcdriver_input_.get_tau();
    o << "\n    reference energy   = " << ene;
    o << "\n    reference variance = " << var;
    o << "\n====================================================";
    app_log() << o.str() << std::endl;
  }

  print_mem("VMCBatched ends", app_log());

  wlog_manager.stopRun();
  estimator_manager_->stopDriverRun();

  finalize(num_blocks, true);
}

RefVector<QMCDriverNew::ContextForSteps> VMCBatched::getContextForStepsRefs() const
{
  RefVector<ContextForSteps> refs;
  refs.reserve(step_contexts_.size());
  for (auto& one_context : step_contexts_)
    refs.push_back(*one_context);
  return refs;
}

void VMCBatched::createStepContexts(int num_crowds)
{
  assert(num_crowds <= rngs_.size());
  step_contexts_.resize(num_crowds);
  for (int i = 0; i < num_crowds; ++i)
    step_contexts_[i] = std::make_unique<ContextForSteps>(rngs_[i]);
}

void VMCBatched::enable_sample_collection()
{
  assert(steps_per_block_ > 0 && "VMCBatched::enable_sample_collection steps_per_block_ must be positive!");
  auto samples = compute_samples_per_rank(qmcdriver_input_.get_max_blocks(), steps_per_block_,
                                          population_.get_num_local_walkers());
  samples_.setMaxSamples(samples);
  collect_samples_ = true;

  auto total_samples = samples * population_.get_num_ranks();
  app_log() << "VMCBatched Driver collecting samples, samples per rank = " << samples << std::endl
            << "                                      total samples    = " << total_samples << std::endl
            << std::endl;
}

} // namespace qmcplusplus
