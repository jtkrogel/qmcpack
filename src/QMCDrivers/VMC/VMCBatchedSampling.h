//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_VMC_BATCHED_SAMPLING_H
#define QMCPLUSPLUS_VMC_BATCHED_SAMPLING_H

#include "Configuration.h"
#include "Particle/MCCoords.hpp"
#include "Utilities/RandomGenerator.h"
#include "Utilities/TimerManager.h"
#include "type_traits/template_types.hpp"

#include <functional>
#include <string_view>

namespace qmcplusplus
{

class ParticleSet;
class TrialWaveFunction;
class Crowd;
class DriftModifierBase;
class MCPopulation;
class QMCDriverInput;
class VMCDriverInput;

/** Shared batched VMC proposal/evaluation step used by sampling-based drivers. */
class VMCBatchedSampling
{
public:
  using FullPrecRealType = QMCTraits::FullPrecRealType;
  using IndexType        = QMCTraits::IndexType;
  using RealType         = QMCTraits::RealType;

  /** Mutable section-step state plus immutable inputs needed by crowd tasks. */
  struct State
  {
    const QMCDriverInput& qmcdrv_input;
    const VMCDriverInput& vmcdrv_input;
    const DriftModifierBase& drift_modifier;
    const MCPopulation& population;
    IndexType recalculate_properties_period;
    const std::size_t steps_per_block;
    IndexType step            = -1;
    IndexType global_step     = -1;
    bool is_recomputing_block = false;
    const bool serializing_crowd_walkers;
    // Explicit driver-owned scalar observer; native HOST particle/FULL state is dirty.
    bool resident_device_observation = false;
    std::function<void(Crowd&, bool)> resident_device_observe;

    State(const QMCDriverInput& qmc_input,
          const VMCDriverInput& vmc_input,
          const DriftModifierBase& drift,
          const MCPopulation& population_in,
          std::size_t block_steps,
          bool serialize_walkers)
        : qmcdrv_input(qmc_input),
          vmcdrv_input(vmc_input),
          drift_modifier(drift),
          population(population_in),
          steps_per_block(block_steps),
          serializing_crowd_walkers(serialize_walkers)
    {}
  };

  /** Narrow view of driver timers and the driver's protected debug callback. */
  struct Timers
  {
    NewTimer& resource;
    NewTimer& move;
    NewTimer& buffer;
    NewTimer& hamiltonian;
    NewTimer& collectables;
    NewTimer& estimators;
    std::function<void(Crowd&, std::string_view)> check_log_and_gl;
  };

  /** Reject unsupported collective-move combinations before walker creation. */
  static void validateMoveConfiguration(const VMCDriverInput& input,
                                        const ParticleSet& electrons,
                                        const TrialWaveFunction& wavefunction,
                                        bool serializing_crowd_walkers);

  /// Prove unchanged complete spatial state; never replaces a Hamiltonian observation.
  static bool canReuseRejectedDeviceFullState(const TrialWaveFunction& wavefunction,
                                             const ParticleSet& particles,
                                             std::size_t accepted_moves, bool recompute);

  /** Advance and evaluate one production step for one crowd. */
  static void runVMCStep(int crowd_id,
                         const State& state,
                         Timers& timers,
                         const RefVector<RandomBase<FullPrecRealType>>& random_generators,
                         UPtrVector<Crowd>& crowds);

  /** Advance and evaluate one warm-up step for one crowd without accumulation. */
  static void runWarmupStep(int crowd_id,
                            const State& state,
                            Timers& timers,
                            const RefVector<RandomBase<FullPrecRealType>>& random_generators,
                            UPtrVector<Crowd>& crowds);

private:
  template<CoordsType CT>
  static void advanceWalkers(const State& state,
                             Crowd& crowd,
                             Timers& timers,
                             RandomBase<FullPrecRealType>& random_generator,
                             bool recompute,
                             bool accumulate_this_step);
};

} // namespace qmcplusplus

#endif // QMCPLUSPLUS_VMC_BATCHED_SAMPLING_H
