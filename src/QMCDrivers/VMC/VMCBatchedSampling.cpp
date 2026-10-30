//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include "VMCBatchedSampling.h"

#include "Estimators/EstimatorInputDelegates.h"
#include "Message/UniformCommunicateError.h"
#include "ParticleBase/RandomSeqGenerator.h"
#include "Particle/MCMultiParticleMoves.h"
#include "QMCDrivers/Crowd.h"
#include "QMCDrivers/GreenFunctionModifiers/DriftModifierBase.h"
#include "QMCDrivers/MCPopulation.h"
#include "QMCDrivers/QMCDriverInput.h"
#include "QMCDrivers/WalkerProperties.h"
#include "QMCDrivers/VMC/ElectronSubsetSelector.h"
#include "QMCDrivers/VMC/VMCDriverInput.h"
#include "QMCDrivers/WFTrain/TrainingWorkCounters.h"
#include "QMCWaveFunctions/TWFGrads.hpp"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "Utilities/NativeTimerScope.h"
#include "TauParams.hpp"
#include <Hdispatcher.h>
#include <PSdispatcher.h>
#include <TWFdispatcher.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

namespace qmcplusplus
{
using MCPWalker = MCPopulation::MCPWalker;

namespace
{

class OptionalNativeTimerScope
{
public:
  explicit OptionalNativeTimerScope(NewTimer* timer)
  {
    if (timer)
      scope_.emplace(*timer);
  }

private:
  std::optional<NativeTimerScope> scope_;
};

void recordEvaluationProducts(wftrain::TrainingWorkCounters* counters,
                              WaveFunctionEvaluationRequest request)
{
  if (!counters)
    return;
  switch (request)
  {
  case WaveFunctionEvaluationRequest::VALUE_ONLY:
    counters->addRequestedProducts(true, false, false);
    counters->addComputedProducts(true, false, false);
    // Generic wavefunction components may implement a reduced request with FULL products.
    counters->addUnknownComputedProducts();
    break;
  case WaveFunctionEvaluationRequest::GRADIENT_ONLY:
    counters->addRequestedProducts(true, true, false);
    counters->addComputedProducts(true, true, false);
    counters->addUnknownComputedProducts();
    break;
  case WaveFunctionEvaluationRequest::FULL_SPATIAL:
    counters->addRequestedProducts(true, true, true);
    counters->addComputedProducts(true, true, true);
    break;
  }
}

/// Discover the one explicit PsiFormer-only route without changing default moves.
const PsiFormerWF* deviceSamplingComponent(const TrialWaveFunction& wavefunction)
{
  const PsiFormerWF* selected = nullptr;
  for (const auto& orbital : wavefunction.getOrbitals())
    if (const auto* component = dynamic_cast<const PsiFormerWF*>(orbital.get());
        component && component->deviceSamplingEnabled())
      selected = component;
  if (selected && wavefunction.getOrbitals().size() != 1)
    throw UniformCommunicateError("Explicit device sampling requires exactly one PsiFormer wavefunction factor.");
  return selected;
}

/// Requested walker logging cannot consume intentionally unmaterialized HOST state.
void validateResidentWalkerLogs(const VMCBatchedSampling::State& state, const UPtrVector<Crowd>& crowds)
{
  if (!state.resident_device_observation)
    return;
  for (const WalkerLogCollector& collector : Crowd::getWalkerLogCollectorRefs(crowds))
    if (collector.logsActive())
      throw UniformCommunicateError("Resident device observation excludes active walker logging.");
}

/** Scale proposed coordinate displacements by the species time steps. */
template<typename RT, CoordsType CT>
void scaleBySqrtTau(const TauParams<RT, CT>& taus, MCCoords<CT>& coordinates)
{
  for (auto& position : coordinates.positions)
    position *= taus.sqrttau;
  if constexpr (CT == CoordsType::POS_SPIN)
    for (auto& spin : coordinates.spins)
      spin *= taus.spin_sqrttau;
}

/** Evaluate the forward or reverse Gaussian Green-function exponent. */
template<typename RT, CoordsType CT>
void computeLogGreensFunction(const MCCoords<CT>& coordinates,
                              const TauParams<RT, CT>& taus,
                              std::vector<QMCTraits::RealType>& log_green)
{
  assert(coordinates.positions.size() == log_green.size());
  std::transform(coordinates.positions.begin(), coordinates.positions.end(), log_green.begin(),
                 [half_over_tau = taus.oneover2tau](const QMCTraits::PosType& position) {
                   return -half_over_tau * dot(position, position);
                 });
  if constexpr (CT == CoordsType::POS_SPIN)
    std::transform(coordinates.spins.begin(), coordinates.spins.end(), log_green.begin(), log_green.begin(),
                   [half_over_tau = taus.spin_oneover2tau](const QMCTraits::FullPrecRealType& spin,
                                                          const QMCTraits::RealType& value) {
                     return value - half_over_tau * spin * spin;
                   });
}

} // namespace

bool VMCBatchedSampling::canReuseRejectedDeviceFullState(const TrialWaveFunction& wavefunction,
                                                       const ParticleSet& particles,
                                                       std::size_t accepted_moves, bool recompute)
{
#ifdef QMC_COMPLEX
  return false;
#else
  if (accepted_moves != 0 || recompute || wavefunction.getOrbitals().size() != 1 ||
      particles.isSpinor() || particles.getLattice().getSuperCellEnum() != SUPERCELL_OPEN)
    return false;
  const auto* component = deviceSamplingComponent(wavefunction);
  return component && component->canReuseFullSpatialReplay(particles, particles, component->parameterVersion(),
                                                          wavefunction.getLogPsi(), wavefunction.getPhase());
#endif
}

void VMCBatchedSampling::validateMoveConfiguration(const VMCDriverInput& input,
                                                   const ParticleSet& electrons,
                                                   const TrialWaveFunction& wavefunction,
                                                   bool serializing_crowd_walkers)
{
  if (const auto* sampler = deviceSamplingComponent(wavefunction); sampler &&
      ((input.get_move_kind() != VMCDriverInput::MoveKind::ALL_ELECTRON &&
        input.get_move_kind() != VMCDriverInput::MoveKind::N_ELECTRON) ||
       input.get_use_drift() != sampler->deviceSamplingUsesDrift() ||
       electrons.isSpinor() || serializing_crowd_walkers))
    throw UniformCommunicateError("Explicit device sampling requires joint alle/n_electron moves, matching usedrift profile, POS and batched crowds.");
  if (input.get_move_kind() == VMCDriverInput::MoveKind::PBYP)
    return;
  if (serializing_crowd_walkers)
    throw UniformCommunicateError("Collective VMC electron moves require multi-walker crowd execution.");
  if (electrons.isSpinor())
    throw UniformCommunicateError("Collective VMC electron moves do not yet support spin coordinates.");
  if (!wavefunction.supportsMultiParticleMoves())
    throw UniformCommunicateError("Every wavefunction component must support collective electron moves.");
  if (input.get_move_kind() == VMCDriverInput::MoveKind::N_ELECTRON &&
      input.get_electrons_per_move() >= electrons.getTotalNum())
    throw UniformCommunicateError(
        "n_electron VMC moves require electrons_per_move to be smaller than the electron count; "
        "use move=\"alle\" for a full-system proposal.");
}

template<CoordsType CT>
void VMCBatchedSampling::advanceWalkers(const State& sft,
                                         Crowd& crowd,
                                         Timers& timers,
                                         RandomBase<FullPrecRealType>& random_generator,
                                         bool recompute,
                                         bool accumulate_this_step,
                                         bool warmup,
                                         IndexType* subset_cursor)
{
  if (crowd.size() == 0)
    return;
  if (warmup && sft.resident_device_observation && !sft.resident_device_mark_host_dirty)
    throw UniformCommunicateError("Resident warmup requires a host-dirty notification.");
  // Reject HOST/debug consumers before any random draws or accepted-state mutation.
  if (sft.resident_device_observation &&
      (!sft.resident_device_observe || sft.qmcdrv_input.get_debug_checks() != DriverDebugChecks::ALL_OFF ||
       CT != CoordsType::POS || sft.vmcdrv_input.get_use_drift() ||
       !deviceSamplingComponent(crowd.get_walker_twfs()[0])))
    throw UniformCommunicateError(
        "Resident device observation requires an explicit callback and debug-free POS/noDrift device sampling.");
  const PSdispatcher ps_dispatcher(!sft.serializing_crowd_walkers);
  const TWFdispatcher twf_dispatcher(!sft.serializing_crowd_walkers);
  const Hdispatcher ham_dispatcher(!sft.serializing_crowd_walkers);
  auto& walkers = crowd.get_walkers();
  const auto accepted_before = crowd.get_accept();
  const auto rejected_before = crowd.get_reject();
  const RefVectorWithLeader<ParticleSet> walker_elecs(crowd.get_walker_elecs()[0], crowd.get_walker_elecs());
  const RefVectorWithLeader<TrialWaveFunction> walker_twfs(crowd.get_walker_twfs()[0], crowd.get_walker_twfs());

  // This is really a waste the resources can be acquired outside of the run steps loop in VMCD!
  // I don't see an  easy way to measure the release without putting the weight of tons of timer_manager calls in
  // ResourceCollectionTeamLock's constructor.
  timers.resource.start();
  ResourceCollectionTeamLock<ParticleSet> pset_res_lock(crowd.getSharedResource().pset_res, walker_elecs);
  ResourceCollectionTeamLock<TrialWaveFunction> twfs_res_lock(crowd.getSharedResource().twf_res, walker_twfs);
  timers.resource.stop();
  const bool use_drift = sft.vmcdrv_input.get_use_drift();
  const bool needs_full_products = !warmup ||
      (sft.qmcdrv_input.get_debug_checks() & DriverDebugChecks::CHECKGL_AFTER_LOAD);
  const bool missing_full_products = needs_full_products &&
      std::any_of(walker_twfs.begin(), walker_twfs.end(),
                  [](const TrialWaveFunction& wavefunction) {
                    return !wavefunction.hasValidGradients() ||
                        !wavefunction.hasValidLaplacians();
                  });
  if (missing_full_products)
  {
    {
      OptionalNativeTimerScope spatial_timer(timers.spatial_evaluation_api);
      twf_dispatcher.flex_evaluateGL(walker_twfs, walker_elecs, false);
    }
    recordEvaluationProducts(timers.work_counters, WaveFunctionEvaluationRequest::FULL_SPATIAL);
    if (!warmup && timers.work_counters)
      timers.work_counters->addPreMoveFullSpatialHandoffBatch();
  }
  if (sft.qmcdrv_input.get_debug_checks() & DriverDebugChecks::CHECKGL_AFTER_LOAD)
    timers.check_log_and_gl(crowd, "checkGL_after_load");

  bool device_moves_completed = false;
  std::size_t device_accepted_moves = 0;

  {
    ScopedTimer pbyp_local_timer(timers.move);
    const int num_walkers   = crowd.size();
    auto& walker_leader     = walker_elecs.getLeader();
    const int num_particles = walker_leader.getTotalNum();
    const bool use_drift    = sft.vmcdrv_input.get_use_drift();

    const VMCDriverInput::MoveKind move_kind = sft.vmcdrv_input.get_move_kind();

    if (const auto* sampler = deviceSamplingComponent(walker_twfs.getLeader()))
    {
      validateMoveConfiguration(sft.vmcdrv_input, walker_leader, walker_twfs.getLeader(),
                                sft.serializing_crowd_walkers);
      RefVector<WaveFunctionComponent> components;
      for (const TrialWaveFunction& wavefunction : walker_twfs)
      {
        if (!deviceSamplingComponent(wavefunction))
          throw UniformCommunicateError("Explicit device sampling crowd contains a disabled or foreign factor.");
        components.push_back(*wavefunction.getOrbitals()[0]);
      }
      const RefVectorWithLeader<WaveFunctionComponent> component_list(components[0], components);
      const auto particles_per_move = move_kind == VMCDriverInput::MoveKind::N_ELECTRON
          ? sft.vmcdrv_input.get_electrons_per_move() : 0;
      std::optional<ElectronSubsetSelector> selector;
      std::function<void(std::vector<std::size_t>&)> select_indices;
      if (particles_per_move)
      {
        if (!subset_cursor)
          throw std::logic_error("Selected device sampling requires a crowd cursor.");
        selector.emplace(num_particles, particles_per_move, sft.vmcdrv_input.get_electron_selection());
        selector->restoreCyclicCursor(*subset_cursor);
        select_indices = [&](std::vector<std::size_t>& indices) {
          for (int walker = 0; walker < num_walkers; ++walker)
          {
            const auto& selected = selector->selectForWalker(random_generator, walker);
            std::copy(selected.begin(), selected.end(), indices.begin() + walker * particles_per_move);
          }
        };
      }
      const auto counts = sampler->mw_advanceDeviceSampling(component_list, walker_elecs,
          sft.qmcdrv_input.get_tau(), sft.qmcdrv_input.get_sub_steps(),
          [&](std::vector<double>& gaussian) { makeGaussRandomWithEngine(gaussian, random_generator); },
          [&]() { return random_generator(); }, use_drift,
          [&](double variance, const double* gradient, double* result) {
            ParticleSet::PosType drift;
            QMCTraits::GradType native_gradient;
            for (std::size_t d = 0; d < 3; ++d)
              native_gradient[d] = gradient[d];
            sft.drift_modifier.getDrift(variance, native_gradient, drift);
            for (std::size_t d = 0; d < 3; ++d)
              result[d] = drift[d];
          }, sft.resident_device_observation, particles_per_move, select_indices);
      if (timers.work_counters)
      {
        timers.work_counters->addRatioApiBatches();
        timers.work_counters->addRequestedProducts(true, use_drift, false);
        timers.work_counters->addComputedProducts(true, use_drift, false);
        if (use_drift)
          timers.work_counters->addDriftGradientApiBatches();
      }
      if (selector)
        *subset_cursor = selector->cyclicCursor();
      for (std::size_t count = 0; count < counts.first; ++count)
        crowd.incAccept();
      for (std::size_t count = 0; count < counts.second; ++count)
        crowd.incReject();
      device_accepted_moves = counts.first;
      device_moves_completed = true;
    }
    else if (move_kind != VMCDriverInput::MoveKind::PBYP)
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
        if (move_kind == VMCDriverInput::MoveKind::N_ELECTRON)
        {
          if (!subset_cursor)
            throw std::logic_error("Selected electron sampling requires a crowd cursor.");
          selector.restoreCyclicCursor(*subset_cursor);
        }
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
          OptionalNativeTimerScope middle_substep(sub_step == 1
              ? &(warmup ? timers.warmup_middle_substep_host : timers.production_middle_substep_host)
              : nullptr);
          makeGaussRandomWithEngine(gaussian_displacements, random_generator);
          selected_indices.clear();
          walker_offsets[0] = 0;

          for (int iw = 0; iw < num_walkers; ++iw)
          {
            const std::vector<ParticleSet::IndexType>* selection = nullptr;
            if (move_kind == VMCDriverInput::MoveKind::ALL_ELECTRON)
              selection = &all_electrons;
            else
              selection = &selector.selectForWalker(random_generator, iw);

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
          const auto request = warmup && !use_drift
              ? WaveFunctionEvaluationRequest::VALUE_ONLY : WaveFunctionEvaluationRequest::FULL_SPATIAL;
          {
            OptionalNativeTimerScope spatial_timer(timers.spatial_evaluation_api);
            TrialWaveFunction::mw_evaluateMultiParticleMove(
                walker_twfs, walker_elecs, moves, log_ratios, request);
          }
          if (timers.work_counters)
          {
            timers.work_counters->addRatioApiBatches();
            recordEvaluationProducts(timers.work_counters, request);
          }

          std::fill(log_gf.begin(), log_gf.end(), RealType{});
          std::fill(log_gb.begin(), log_gb.end(), RealType{});
          for (int iw = 0; iw < num_walkers; ++iw)
          {
            const auto move_slice     = moves.slice(iw);
            for (int local_index = 0; local_index < particles_per_move; ++local_index)
            {
              const int flat_index = iw * particles_per_move + local_index;
              const int iat        = move_slice.particleIndex(local_index);
              const int group      = walker_elecs[iw].getGroupID(iat);
              const RealType tau_over_mass =
                  sft.qmcdrv_input.get_tau() / walker_leader.get_mass_by_group()[group];
              ParticleSet::PosType reverse_drift;
              if (use_drift)
              {
                const auto& proposed_grad = walker_twfs[iw].multiParticleProposalGradient();
                sft.drift_modifier.getDrift(tau_over_mass, proposed_grad[iat], reverse_drift);
              }
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
                std::log(random_generator()) < log_acceptance;
            if (accepted[iw])
              crowd.incAccept();
            else
              crowd.incReject();
          }

          TrialWaveFunction::mw_accept_rejectMultiParticleMove(walker_twfs, walker_elecs, moves, accepted);
          ParticleSet::mw_accept_rejectMoveSelectedParticles(walker_elecs, accepted);
          if (subset_cursor && move_kind == VMCDriverInput::MoveKind::N_ELECTRON)
            *subset_cursor = selector.cyclicCursor();
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
        OptionalNativeTimerScope middle_substep(sub_step == 1
            ? &(warmup ? timers.warmup_middle_substep_host : timers.production_middle_substep_host)
            : nullptr);
        //This generates an entire steps worth of deltas.
        makeGaussRandomWithEngine(walker_deltas, random_generator);

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
              {
                OptionalNativeTimerScope spatial_timer(timers.spatial_evaluation_api);
                twf_dispatcher.flex_evalGrad(walker_twfs, walker_elecs, iat, grads_now);
              }
              if (timers.work_counters)
              {
                timers.work_counters->addDriftGradientApiBatches();
                timers.work_counters->addRequestedProducts(false, true, false);
                timers.work_counters->addComputedProducts(false, true, false);
              }
              sft.drift_modifier.getDrifts(taus, grads_now, drifts);
              drifts += deltas;
            }
            else
              drifts = deltas;

            ps_dispatcher.flex_makeMove(walker_elecs, iat, drifts, are_valid);

            // This is inelegant
            if (use_drift)
            {
              {
                OptionalNativeTimerScope spatial_timer(timers.spatial_evaluation_api);
                twf_dispatcher.flex_calcRatioGrad(walker_twfs, walker_elecs, iat, ratios, grads_new);
              }
              if (timers.work_counters)
              {
                timers.work_counters->addRatioApiBatches();
                timers.work_counters->addDriftGradientApiBatches();
                timers.work_counters->addRequestedProducts(true, true, false);
                timers.work_counters->addComputedProducts(true, true, false);
              }

              computeLogGreensFunction(deltas, taus, log_gf);

              sft.drift_modifier.getDrifts(taus, grads_new, drifts_reverse);

              drifts_reverse += drifts;

              computeLogGreensFunction(drifts_reverse, taus, log_gb);
            }
            else
            {
              {
                OptionalNativeTimerScope spatial_timer(timers.spatial_evaluation_api);
                twf_dispatcher.flex_calcRatio(walker_twfs, walker_elecs, iat, ratios);
              }
              if (timers.work_counters)
              {
                timers.work_counters->addRatioApiBatches();
                timers.work_counters->addRequestedProducts(true, false, false);
                timers.work_counters->addComputedProducts(true, false, false);
              }
            }

            std::transform(ratios.begin(), ratios.end(), prob.begin(), [](auto ratio) { return std::norm(ratio); });

            isAccepted.clear();

            for (int i_accept = 0; i_accept < num_walkers; ++i_accept)
              if (are_valid[i_accept] && prob[i_accept] >= std::numeric_limits<RealType>::epsilon() &&
                  random_generator() < prob[i_accept] * std::exp(log_gb[i_accept] - log_gf[i_accept]))
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

  if (timers.work_counters)
  {
    timers.work_counters->addProposalRounds(static_cast<std::uint64_t>(sft.qmcdrv_input.get_sub_steps()));
    timers.work_counters->addMoveDecisions(crowd.get_accept() - accepted_before,
                                           crowd.get_reject() - rejected_before);
  }

  if (sft.resident_device_observation)
  {
    if (!device_moves_completed)
      throw UniformCommunicateError("Resident device observation did not complete device moves.");
    if (warmup)
    {
      // A resident observation callback is a production consumer, even when
      // scalar accumulation is disabled. Keep its host energy unavailable.
      sft.resident_device_mark_host_dirty();
      for (int iw = 0; iw < crowd.size(); ++iw)
        walkers[iw].get().Properties(WalkerProperties::LOCALENERGY) =
            std::numeric_limits<QMCTraits::FullPrecRealType>::quiet_NaN();
      return;
    }
    // The admitted driver owns scalar properties/statistics and the later cold
    // fence. Never invoke generic estimators, native H, FULL or logging here.
    sft.resident_device_observe(crowd, accumulate_this_step);
    return;
  }

  {
    ScopedTimer buffer_local_timer(timers.buffer);
    if (device_moves_completed)
    {
      // Keep native tables current even when an all-rejected window left its
      // already complete FULL cache untouched. Any failed proof takes full replay.
      ps_dispatcher.flex_update(walker_elecs);
      bool reuse_full = device_accepted_moves == 0 && !recompute;
      for (std::size_t walker = 0; walker < walker_twfs.size() && reuse_full; ++walker)
        reuse_full = canReuseRejectedDeviceFullState(walker_twfs[walker], walker_elecs[walker],
                                                    device_accepted_moves, recompute);
      if (!reuse_full)
      {
        {
          OptionalNativeTimerScope spatial_timer(timers.spatial_evaluation_api);
          twf_dispatcher.flex_evaluateLog(walker_twfs, walker_elecs);
        }
        recordEvaluationProducts(timers.work_counters, WaveFunctionEvaluationRequest::FULL_SPATIAL);
      }
    }
    else if (!(warmup && !sft.vmcdrv_input.get_use_drift() &&
               sft.qmcdrv_input.get_debug_checks() == DriverDebugChecks::ALL_OFF))
    {
      {
        OptionalNativeTimerScope spatial_timer(timers.spatial_evaluation_api);
        twf_dispatcher.flex_evaluateGL(walker_twfs, walker_elecs, recompute);
      }
      recordEvaluationProducts(timers.work_counters, WaveFunctionEvaluationRequest::FULL_SPATIAL);
    }
    if (sft.qmcdrv_input.get_debug_checks() & DriverDebugChecks::CHECKGL_AFTER_MOVES)
      timers.check_log_and_gl(crowd, "checkGL_after_moves");
    ps_dispatcher.flex_saveWalker(walker_elecs, walkers);
  }

  if (warmup || !sft.evaluate_energy)
  {
    // Sampling still publishes current positions/logs, but no fictitious local energy.
    for (int iw = 0; iw < crowd.size(); ++iw)
      walkers[iw].get().resetProperty(walker_twfs[iw].getLogPsi(), walker_twfs[iw].getPhase(),
                               std::numeric_limits<QMCTraits::FullPrecRealType>::quiet_NaN());
    return;
  }

  const RefVectorWithLeader<QMCHamiltonian> walker_hamiltonians(crowd.get_walker_hamiltonians()[0],
                                                                crowd.get_walker_hamiltonians());
  {
    ScopedTimer hamiltonian_local_timer(timers.hamiltonian);
    ResourceCollectionTeamLock<QMCHamiltonian> hams_res_lock(crowd.getSharedResource().ham_res, walker_hamiltonians);
    std::vector<QMCHamiltonian::FullPrecRealType> local_energies;
    {
      OptionalNativeTimerScope hamiltonian_timer(timers.hamiltonian_evaluation_api);
      local_energies = ham_dispatcher.flex_evaluate(walker_hamiltonians, walker_twfs, walker_elecs);
    }
    if (timers.work_counters)
    {
      const auto ecp_walkers = std::count_if(walker_hamiltonians.begin(), walker_hamiltonians.end(),
          [](const QMCHamiltonian& hamiltonian) { return hamiltonian.hasPhysicalNLPP(); });
      timers.work_counters->addHamiltonianEvaluations(local_energies.size(), ecp_walkers);
    }

    auto resetSigNLocalEnergy = [](MCPWalker& walker, TrialWaveFunction& twf, auto& local_energy) {
      walker.resetProperty(twf.getLogPsi(), twf.getPhase(), local_energy);
    };
    for (int iw = 0; iw < crowd.size(); ++iw)
      resetSigNLocalEnergy(walkers[iw], walker_twfs[iw], local_energies[iw]);
  }

  {
    ScopedTimer collectables_local_timer(timers.collectables);
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

  if (accumulate_this_step && sft.accumulate_estimators)
  {
    ScopedTimer est_timer(timers.estimators);
    crowd.accumulate(random_generator);
  }

  // collect walker logs
  crowd.collectStepWalkerLog(sft.global_step);

  // TODO:
  //  check if all moves failed
}

template void VMCBatchedSampling::advanceWalkers<CoordsType::POS>(const State& state,
                                                                  Crowd& crowd,
                                                                  Timers& timers,
                                                                  RandomBase<FullPrecRealType>& random_generator,
                                                                  bool recompute,
                                                                  bool accumulate_this_step,
                                                                  bool warmup,
                                                                  IndexType* subset_cursor);

template void VMCBatchedSampling::advanceWalkers<CoordsType::POS_SPIN>(const State& state,
                                                                       Crowd& crowd,
                                                                       Timers& timers,
                                                                       RandomBase<FullPrecRealType>& random_generator,
                                                                       bool recompute,
                                                                       bool accumulate_this_step,
                                                                       bool warmup,
                                                                       IndexType* subset_cursor);

void VMCBatchedSampling::runVMCStep(int crowd_id,
                                    const State& state,
                                    Timers& timers,
                                    const RefVector<RandomBase<FullPrecRealType>>& random_generators,
                                    UPtrVector<Crowd>& crowds)
{
  if (!state.evaluate_energy && state.resident_device_observation)
    throw std::invalid_argument("Objective-only sampling requires the CPU observation path");
  validateResidentWalkerLogs(state, crowds);
  Crowd& crowd = *crowds[crowd_id];
  RandomBase<FullPrecRealType>& random_generator = random_generators[crowd_id];
  crowd.setRNGForHamiltonian(random_generator);
  const bool recompute  = state.is_recomputing_block && state.step + 1 == state.steps_per_block;
  const bool accumulate = state.step % state.qmcdrv_input.get_estimator_measurement_period() == 0;
  IndexType* cursor = state.subset_cursors.empty() ? nullptr : &state.subset_cursors.at(crowd_id);
  if (state.population.get_golden_electrons().isSpinor())
    advanceWalkers<CoordsType::POS_SPIN>(state, crowd, timers, random_generator, recompute, accumulate, false, cursor);
  else
    advanceWalkers<CoordsType::POS>(state, crowd, timers, random_generator, recompute, accumulate, false, cursor);
}

void VMCBatchedSampling::runWarmupStep(int crowd_id,
                                       const State& state,
                                       Timers& timers,
                                       const RefVector<RandomBase<FullPrecRealType>>& random_generators,
                                       UPtrVector<Crowd>& crowds)
{
  validateResidentWalkerLogs(state, crowds);
  Crowd& crowd = *crowds[crowd_id];
  RandomBase<FullPrecRealType>& random_generator = random_generators[crowd_id];
  IndexType* cursor = state.subset_cursors.empty() ? nullptr : &state.subset_cursors.at(crowd_id);
  if (state.population.get_golden_electrons().isSpinor())
    advanceWalkers<CoordsType::POS_SPIN>(state, crowd, timers, random_generator, false, false, true, cursor);
  else
    advanceWalkers<CoordsType::POS>(state, crowd, timers, random_generator, false, false, true, cursor);
}

} // namespace qmcplusplus
