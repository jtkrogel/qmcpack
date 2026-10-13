//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2020 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File refactored from VMC.h
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_VMCBATCHED_H
#define QMCPLUSPLUS_VMCBATCHED_H

#include "QMCDrivers/QMCDriverNew.h"
#include "QMCDrivers/VMC/VMCDriverInput.h"
#include "QMCDrivers/VMC/VMCBatchedSampling.h"
#include "QMCDrivers/MCPopulation.h"
#include "QMCDrivers/GreenFunctionModifiers/DriftModifierBase.h"

#include "Utilities/Timer.h"

namespace qmcplusplus
{
namespace testing
{
class VMCBatchedTest;
}

/** @ingroup QMCDrivers  ParticleByParticle
 * @brief Implements a VMC using particle-by-particle move. Threaded execution.
 */
class VMCBatched : public QMCDriverNew
{
public:
  using FullPrecRealType  = QMCTraits::FullPrecRealType;
  using PosType           = QMCTraits::PosType;
  using ParticlePositions = PtclOnLatticeTraits::ParticlePos;

  using StateForThread = VMCBatchedSampling::State;

public:
  /// Constructor.
  VMCBatched(const ProjectData& project_data,
             QMCDriverInput&& qmcdriver_input,
             UPtr<EstimatorManagerNew>&& estimator_manager,
             VMCDriverInput&& input,
             WalkerConfigurations& wc,
             MCPopulation&& pop,
             const RefVector<RandomBase<FullPrecRealType>>& rng_refs,
             SampleStack& samples_,
             Communicate* comm);
  /// Copy constructor
  VMCBatched(const VMCBatched&) = delete;
  /// Copy operator (disabled).
  VMCBatched& operator=(const VMCBatched&) = delete;

  void process(xmlNodePtr node) override;

  void run() override;

  /** transitional interface on the way to better walker count adjustment handling.
   *  returns a closure taking walkers per rank and accomplishing what calc_default_local_walkers does.
   */
  auto getCDLW();

  /** Enable collecting samples during the VMC run
   *
   *  strong assumption that VMCBatched driver has passed through process phase of
   *  initialization.
   *  A side effect of VMCBatched::process is that MCPopulation has created local walkers.
   */
  void enable_sample_collection();

private:
  VMCDriverInput vmcdriver_input_;
  /// Per crowd, driver-specific move contexts
  UPtrVector<ContextForSteps> step_contexts_;

  /// obtain reference vector of step contexts
  RefVector<ContextForSteps> getContextForStepsRefs() const;

  QMCRunType getRunType() override { return QMCRunType::VMC_BATCH; }
  /// Storage for samples (later used in optimizer)
  SampleStack& samples_;
  /// Sample collection flag
  bool collect_samples_;

  // create Rngs and StepContests
  void createStepContexts(int num_crowds);

  /** function to calculate samples per MPI rank
   */
  static size_t compute_samples_per_rank(const size_t num_blocks,
                                         const size_t samples_per_block,
                                         const size_t local_walkers);

  friend class qmcplusplus::testing::VMCBatchedTest;
};

extern std::ostream& operator<<(std::ostream& o_stream, const VMCBatched& vmc_batched);
} // namespace qmcplusplus

#endif
