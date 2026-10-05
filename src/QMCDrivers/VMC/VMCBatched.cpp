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
#include "Utilities/RunTimeManager.h"
#include "ParticleBase/RandomSeqGenerator.h"
#include "Particle/MCSample.h"
#include "Particle/MCMultiParticleMoves.h"
#include "MemoryUsage.h"
#include "QMCWaveFunctions/TWFGrads.hpp"
#include "QMCDrivers/VMC/ElectronSubsetSelector.h"
#include <PSdispatcher.h>
#include <TWFdispatcher.h>
#include <Hdispatcher.h>
#include "TauParams.hpp"
#include "WalkerLogManager.h"

#include <cmath>
#include <numeric>

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
                       Communicate* comm)
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
      samples_(samples),
      collect_samples_(false)
{}

template<CoordsType CT>
void VMCBatched::advanceWalkers(const StateForThread& sft,
                                Crowd& crowd,
                                QMCDriverNew::DriverTimers& timers,
                                ContextForSteps& step_context,
                                bool recompute,
                                bool accumulate_this_step)
{
  if (crowd.size() == 0)
    return;
  const PSdispatcher ps_dispatcher(!sft.serializing_crowd_walkers);
  const TWFdispatcher twf_dispatcher(!sft.serializing_crowd_walkers);
  const Hdispatcher ham_dispatcher(!sft.serializing_crowd_walkers);
  auto& walkers = crowd.get_walkers();
  const RefVectorWithLeader<ParticleSet> walker_elecs(crowd.get_walker_elecs()[0], crowd.get_walker_elecs());
  const RefVectorWithLeader<TrialWaveFunction> walker_twfs(crowd.get_walker_twfs()[0], crowd.get_walker_twfs());

  // This is really a waste the resources can be acquired outside of the run steps loop in VMCD!
  // I don't see an  easy way to measure the release without putting the weight of tons of timer_manager calls in
  // ResourceCollectionTeamLock's constructor.
  timers.resource_timer.start();
  ResourceCollectionTeamLock<ParticleSet> pset_res_lock(crowd.getSharedResource().pset_res, walker_elecs);
  ResourceCollectionTeamLock<TrialWaveFunction> twfs_res_lock(crowd.getSharedResource().twf_res, walker_twfs);
  timers.resource_timer.stop();
  if (sft.qmcdrv_input.get_debug_checks() & DriverDebugChecks::CHECKGL_AFTER_LOAD)
    checkLogAndGL(crowd, "checkGL_after_load", sft.serializing_crowd_walkers);

  {
    ScopedTimer pbyp_local_timer(timers.movepbyp_timer);
    const int num_walkers   = crowd.size();
    auto& walker_leader     = walker_elecs.getLeader();
    const int num_particles = walker_leader.getTotalNum();
    const bool use_drift    = sft.vmcdrv_input.get_use_drift();

    const VMCDriverInput::MoveKind move_kind = sft.vmcdrv_input.get_move_kind();

    if (move_kind != VMCDriverInput::MoveKind::PBYP)
    {
      if constexpr (CT == CoordsType::POS_SPIN)
        throw std::runtime_error("Collective VMC electron moves do not support spin coordinates.");
      else
      {
        if (sft.serializing_crowd_walkers)
          throw std::runtime_error("Collective VMC electron moves require crowd-level execution.");
        if (!walker_twfs.getLeader().supportsMultiParticleMoves())
          throw std::runtime_error(
              "The requested collective VMC move is not supported by every wavefunction component.");

        const int particles_per_move = move_kind == VMCDriverInput::MoveKind::ALL_ELECTRON
            ? num_particles
            : sft.vmcdrv_input.get_electrons_per_move();
        if (particles_per_move <= 0 || particles_per_move > num_particles)
          throw std::runtime_error("electrons_per_move exceeds the number of electrons in the simulation.");

        ElectronSubsetSelector selector(num_particles, particles_per_move,
                                        sft.vmcdrv_input.get_electron_selection());
        std::vector<ParticleSet::PosType> gaussian_displacements(num_walkers * particles_per_move);
        std::vector<ParticleSet::PosType> physical_displacements(num_walkers * particles_per_move);
        std::vector<ParticleSet::PosType> proposed_positions(num_walkers * particles_per_move);
        std::vector<ParticleSet::IndexType> selected_indices;
        selected_indices.reserve(num_walkers * particles_per_move);
        std::vector<ParticleSet::IndexType> all_electrons(num_particles);
        std::iota(all_electrons.begin(), all_electrons.end(), ParticleSet::IndexType{0});
        std::vector<std::size_t> walker_offsets(num_walkers + 1);
        std::vector<TrialWaveFunction::LogValue> log_ratios(num_walkers);
        std::vector<RealType> log_gf(num_walkers);
        std::vector<RealType> log_gb(num_walkers);
        std::vector<bool> accepted(num_walkers);

        for (int sub_step = 0; sub_step < sft.qmcdrv_input.get_sub_steps(); ++sub_step)
        {
          makeGaussRandomWithEngine(gaussian_displacements, step_context.get_random_gen());
          selected_indices.clear();
          walker_offsets[0] = 0;

          for (int iw = 0; iw < num_walkers; ++iw)
          {
            const std::vector<ParticleSet::IndexType>* selection = nullptr;
            if (move_kind == VMCDriverInput::MoveKind::ALL_ELECTRON)
              selection = &all_electrons;
            else
              selection = &selector.select(step_context.get_random_gen());

            const ParticleSet& particles = walker_elecs[iw];
            for (int local_index = 0; local_index < particles_per_move; ++local_index)
            {
              const int flat_index = iw * particles_per_move + local_index;
              const int iat        = (*selection)[local_index];
              const int group      = particles.getGroupID(iat);
              const RealType tau_over_mass =
                  sft.qmcdrv_input.get_tau() / walker_leader.get_mass_by_group()[group];

              gaussian_displacements[flat_index] *= std::sqrt(tau_over_mass);
              ParticleSet::PosType drift;
              if (use_drift)
                sft.drift_modifier.getDrift(tau_over_mass, particles.G[iat], drift);
              else
                drift = ParticleSet::PosType{};

              physical_displacements[flat_index] = gaussian_displacements[flat_index] + drift;
              proposed_positions[flat_index]      = particles.R[iat] + physical_displacements[flat_index];
              selected_indices.push_back(iat);
            }
            walker_offsets[iw + 1] = selected_indices.size();
          }

          const MCMultiParticleMoves<CoordsType::POS> moves(walker_offsets, selected_indices, proposed_positions);
          TrialWaveFunction::mw_evaluateMultiParticleMove(walker_twfs, walker_elecs, moves, log_ratios);

          std::fill(log_gf.begin(), log_gf.end(), RealType{});
          std::fill(log_gb.begin(), log_gb.end(), RealType{});
          for (int iw = 0; iw < num_walkers; ++iw)
          {
            const auto move_slice     = moves.slice(iw);
            const auto& proposed_grad = walker_twfs[iw].multiParticleProposalGradient();
            for (int local_index = 0; local_index < particles_per_move; ++local_index)
            {
              const int flat_index = iw * particles_per_move + local_index;
              const int iat        = move_slice.particleIndex(local_index);
              const int group      = walker_elecs[iw].getGroupID(iat);
              const RealType tau_over_mass =
                  sft.qmcdrv_input.get_tau() / walker_leader.get_mass_by_group()[group];
              ParticleSet::PosType reverse_drift;
              if (use_drift)
                sft.drift_modifier.getDrift(tau_over_mass, proposed_grad[iat], reverse_drift);
              else
                reverse_drift = ParticleSet::PosType{};

              log_gf[iw] -= dot(gaussian_displacements[flat_index], gaussian_displacements[flat_index]) /
                  (RealType{2} * tau_over_mass);
              const ParticleSet::PosType reverse_residual = physical_displacements[flat_index] + reverse_drift;
              log_gb[iw] -= dot(reverse_residual, reverse_residual) / (RealType{2} * tau_over_mass);
            }
          }

          std::vector<bool> position_valid;
          try
          {
            ParticleSet::mw_makeMoveSelectedParticles(walker_elecs, moves, position_valid);
          }
          catch (...)
          {
            // The wavefunction proposal exists before ParticleSet installs the
            // trial coordinates. Unwind it if particle-side validation fails.
            const std::vector<bool> reject_all(num_walkers, false);
            TrialWaveFunction::mw_accept_rejectMultiParticleMove(walker_twfs, walker_elecs, moves, reject_all);
            throw;
          }
          for (int iw = 0; iw < num_walkers; ++iw)
          {
            const auto move_slice = moves.slice(iw);
            bool walker_valid     = true;
            for (std::size_t local_index = 0; local_index < move_slice.size(); ++local_index)
              walker_valid = walker_valid && position_valid[move_slice.flatOffset() + local_index];

            const RealType log_acceptance =
                RealType{2} * std::real(log_ratios[iw]) + log_gb[iw] - log_gf[iw];
            accepted[iw] = walker_valid && std::isfinite(log_acceptance) &&
                std::log(step_context.get_random_gen()()) < log_acceptance;
            if (accepted[iw])
              crowd.incAccept();
            else
              crowd.incReject();
          }

          TrialWaveFunction::mw_accept_rejectMultiParticleMove(walker_twfs, walker_elecs, moves, accepted);
          ParticleSet::mw_accept_rejectMoveSelectedParticles(walker_elecs, accepted);
        }
      }
    }
    else
    {
      std::vector<bool> are_valid(num_walkers);
      std::vector<TrialWaveFunction::PsiValue> ratios(num_walkers);
      std::vector<RealType> log_gf(num_walkers);
      std::vector<RealType> log_gb(num_walkers);
      std::vector<RealType> prob(num_walkers);

      // local list to handle accept/reject
      std::vector<bool> isAccepted;
      std::vector<std::reference_wrapper<TrialWaveFunction>> twf_accept_list, twf_reject_list;
      isAccepted.reserve(num_walkers);

      MCCoords<CT> drifts(num_walkers), drifts_reverse(num_walkers);
      MCCoords<CT> walker_deltas(num_walkers * num_particles), deltas(num_walkers);
      TWFGrads<CT> grads_now(num_walkers), grads_new(num_walkers);

      for (int sub_step = 0; sub_step < sft.qmcdrv_input.get_sub_steps(); sub_step++)
      {
        //This generates an entire steps worth of deltas.
        makeGaussRandomWithEngine(walker_deltas, step_context.get_random_gen());

        // up and down electrons are "species" within qmpack
        for (int ig = 0; ig < walker_leader.groups(); ++ig) //loop over species
        {
          TauParams<RealType, CT> taus(sft.qmcdrv_input.get_tau(), 1.0 / walker_leader.get_mass_by_group()[ig],
                                       sft.qmcdrv_input.get_spin_mass());

          twf_dispatcher.flex_prepareGroup(walker_twfs, walker_elecs, ig);

          for (int iat = walker_leader.first(ig); iat < walker_leader.last(ig); ++iat)
          {
            //get deltas for this particle (iat) for all walkers
            walker_deltas.getSubset(iat * num_walkers, num_walkers, deltas);
            scaleBySqrtTau(taus, deltas);

            if (use_drift)
            {
              twf_dispatcher.flex_evalGrad(walker_twfs, walker_elecs, iat, grads_now);
              sft.drift_modifier.getDrifts(taus, grads_now, drifts);
              drifts += deltas;
            }
            else
              drifts = deltas;

            ps_dispatcher.flex_makeMove(walker_elecs, iat, drifts, are_valid);

            // This is inelegant
            if (use_drift)
            {
              twf_dispatcher.flex_calcRatioGrad(walker_twfs, walker_elecs, iat, ratios, grads_new);

              computeLogGreensFunction(deltas, taus, log_gf);

              sft.drift_modifier.getDrifts(taus, grads_new, drifts_reverse);

              drifts_reverse += drifts;

              computeLogGreensFunction(drifts_reverse, taus, log_gb);
            }
            else
              twf_dispatcher.flex_calcRatio(walker_twfs, walker_elecs, iat, ratios);

            std::transform(ratios.begin(), ratios.end(), prob.begin(), [](auto ratio) { return std::norm(ratio); });

            isAccepted.clear();

            for (int i_accept = 0; i_accept < num_walkers; ++i_accept)
              if (are_valid[i_accept] && prob[i_accept] >= std::numeric_limits<RealType>::epsilon() &&
                  step_context.get_random_gen()() < prob[i_accept] * std::exp(log_gb[i_accept] - log_gf[i_accept]))
              {
                crowd.incAccept();
                isAccepted.push_back(true);
              }
              else
              {
                crowd.incReject();
                isAccepted.push_back(false);
              }

            twf_dispatcher.flex_accept_rejectMove(walker_twfs, walker_elecs, iat, isAccepted, true);

            ps_dispatcher.flex_accept_rejectMove<CT>(walker_elecs, iat, isAccepted);
          }
        }
        twf_dispatcher.flex_completeUpdates(walker_twfs);
      }

      ps_dispatcher.flex_donePbyP(walker_elecs);
    }
  }

  {
    ScopedTimer buffer_local_timer(timers.buffer_timer);
    twf_dispatcher.flex_evaluateGL(walker_twfs, walker_elecs, recompute);
    if (sft.qmcdrv_input.get_debug_checks() & DriverDebugChecks::CHECKGL_AFTER_MOVES)
      checkLogAndGL(crowd, "checkGL_after_moves", sft.serializing_crowd_walkers);
    ps_dispatcher.flex_saveWalker(walker_elecs, walkers);
  }

  const RefVectorWithLeader<QMCHamiltonian> walker_hamiltonians(crowd.get_walker_hamiltonians()[0],
                                                                crowd.get_walker_hamiltonians());
  {
    ScopedTimer hamiltonian_local_timer(timers.hamiltonian_timer);
    ResourceCollectionTeamLock<QMCHamiltonian> hams_res_lock(crowd.getSharedResource().ham_res, walker_hamiltonians);
    std::vector<QMCHamiltonian::FullPrecRealType> local_energies(
        ham_dispatcher.flex_evaluate(walker_hamiltonians, walker_twfs, walker_elecs));

    auto resetSigNLocalEnergy = [](MCPWalker& walker, TrialWaveFunction& twf, auto& local_energy) {
      walker.resetProperty(twf.getLogPsi(), twf.getPhase(), local_energy);
    };
    for (int iw = 0; iw < crowd.size(); ++iw)
      resetSigNLocalEnergy(walkers[iw], walker_twfs[iw], local_energies[iw]);
  }

  {
    ScopedTimer collectables_local_timer(timers.collectables_timer);
    auto evaluateNonPhysicalHamiltonianElements = [](QMCHamiltonian& ham, TrialWaveFunction& psi, ParticleSet& pset,
                                                     MCPWalker& walker) { ham.auxHevaluate(psi, pset, walker); };
    for (int iw = 0; iw < crowd.size(); ++iw)
      evaluateNonPhysicalHamiltonianElements(walker_hamiltonians[iw], walker_twfs[iw], walker_elecs[iw], walkers[iw]);

    auto savePropertiesIntoWalker = [](QMCHamiltonian& ham, MCPWalker& walker) {
      ham.saveProperty(walker.getPropertyBase());
    };
    for (int iw = 0; iw < crowd.size(); ++iw)
      savePropertiesIntoWalker(walker_hamiltonians[iw], walkers[iw]);
  }

  if (accumulate_this_step)
  {
    ScopedTimer est_timer(timers.estimators_timer);
    crowd.accumulate(step_context.get_random_gen());
  }

  // collect walker logs
  crowd.collectStepWalkerLog(sft.global_step);

  // TODO:
  //  check if all moves failed
}

template void VMCBatched::advanceWalkers<CoordsType::POS>(const StateForThread& sft,
                                                          Crowd& crowd,
                                                          QMCDriverNew::DriverTimers& timers,
                                                          ContextForSteps& step_context,
                                                          bool recompute,
                                                          bool accumulate_this_step);

template void VMCBatched::advanceWalkers<CoordsType::POS_SPIN>(const StateForThread& sft,
                                                               Crowd& crowd,
                                                               QMCDriverNew::DriverTimers& timers,
                                                               ContextForSteps& step_context,
                                                               bool recompute,
                                                               bool accumulate_this_step);

/** Thread body for VMC step
 *
 */
void VMCBatched::runVMCStep(int crowd_id,
                            const StateForThread& sft,
                            DriverTimers& timers,
                            UPtrVector<ContextForSteps>& context_for_steps,
                            UPtrVector<Crowd>& crowds)
{
  Crowd& crowd = *(crowds[crowd_id]);
  crowd.setRNGForHamiltonian(context_for_steps[crowd_id]->get_random_gen());
  const IndexType step = sft.step;
  // Are we entering the the last step of a block to recompute at?
  const bool recompute_this_step = (sft.is_recomputing_block && (step + 1) == sft.steps_per_block);
  // For VMC we don't call this method for warmup steps.
  const bool accumulate_this_step = (step % sft.qmcdrv_input.get_estimator_measurement_period() == 0);
  const bool spin_move            = sft.population.get_golden_electrons().isSpinor();
  if (spin_move)
    advanceWalkers<CoordsType::POS_SPIN>(sft, crowd, timers, *context_for_steps[crowd_id], recompute_this_step,
                                         accumulate_this_step);
  else
    advanceWalkers<CoordsType::POS>(sft, crowd, timers, *context_for_steps[crowd_id], recompute_this_step,
                                    accumulate_this_step);
}

void VMCBatched::process(xmlNodePtr node)
{
  ScopedTimer local_timer(timers_.startup_timer);
  print_mem("VMCBatched before initialization", app_log());

  try
  {
    if (vmcdriver_input_.get_move_kind() != VMCDriverInput::MoveKind::PBYP)
    {
      const ParticleSet& electrons = population_.get_golden_electrons();
      const TrialWaveFunction& wavefunction = population_.get_golden_twf();
      if (serializing_crowd_walkers_)
        throw UniformCommunicateError(
            "Collective VMC electron moves require multi-walker crowd execution.");
      if (electrons.isSpinor())
        throw UniformCommunicateError(
            "Collective VMC electron moves do not yet support spin coordinates.");
      if (!wavefunction.supportsMultiParticleMoves())
        throw UniformCommunicateError(
            "Every wavefunction component must support collective electron moves.");
      if (vmcdriver_input_.get_move_kind() == VMCDriverInput::MoveKind::N_ELECTRON &&
          vmcdriver_input_.get_electrons_per_move() >= electrons.getTotalNum())
        throw UniformCommunicateError(
            "n_electron VMC moves require electrons_per_move to be smaller than the electron count; "
            "use move=\"alle\" for a full-system proposal.");
    }

    QMCDriverNew::AdjustedWalkerCounts awc =
        adjustGlobalWalkerCount(*myComm, walker_configs_ref_.getActiveWalkers(), qmcdriver_input_.get_total_walkers(),
                                qmcdriver_input_.get_walkers_per_rank(), 1.0,
                                determineNumCrowds(qmcdriver_input_.get_num_crowds(), rngs_.size()));

    steps_per_block_ =
        determineStepsPerBlock(awc.global_walkers, qmcdriver_input_.get_requested_samples(),
                               qmcdriver_input_.get_requested_steps(), qmcdriver_input_.get_max_blocks());

    initPopulationAndCrowds(awc);
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
    auto runWarmupStep = [](int crowd_id, StateForThread& sft, DriverTimers& timers,
                            UPtrVector<ContextForSteps>& context_for_steps, UPtrVector<Crowd>& crowds) {
      Crowd& crowd                    = *(crowds[crowd_id]);
      const bool recompute            = false;
      const bool accumulate_this_step = false;
      const bool spin_move            = sft.population.get_golden_electrons().isSpinor();
      if (spin_move)
        advanceWalkers<CoordsType::POS_SPIN>(sft, crowd, timers, *context_for_steps[crowd_id], recompute,
                                             accumulate_this_step);
      else
        advanceWalkers<CoordsType::POS>(sft, crowd, timers, *context_for_steps[crowd_id], recompute,
                                        accumulate_this_step);
    };

    for (int step = 0; step < qmcdriver_input_.get_warmup_steps(); ++step)
    {
      ScopedTimer local_timer(timers_.run_steps_timer);
      crowd_task(crowds_.size(), runWarmupStep, vmc_state, timers_, step_contexts_, crowds_);
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
        crowd_task(crowds_.size(), runVMCStep, vmc_state, timers_, step_contexts_, crowds_);

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
