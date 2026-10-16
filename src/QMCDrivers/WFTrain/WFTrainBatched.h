//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_WFTRAIN_BATCHED_H
#define QMCPLUSPLUS_WFTRAIN_BATCHED_H

#include "QMCDrivers/QMCDriverNew.h"
#include "QMCDrivers/VMC/VMCBatchedSampling.h"
#include "QMCDrivers/VMC/VMCDriverInput.h"
#include "QMCDrivers/WFTrain/WFTrainDriverInput.h"
#include "QMCDrivers/WFTrain/TrainingTimers.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
class DeterminantOverlapObjective;
class StructuredParameterProvider;
}

namespace qmcplusplus
{
class PsiFormerComplexWF;

namespace psiformer
{
class OrbitalTargetEvaluator;
}

/** Batched high-parameter wavefunction-training driver. */
class WFTrainBatched : public QMCDriverNew
{
public:
  WFTrainBatched(const ProjectData& project_data,
                 QMCDriverInput&& qmcdriver_input,
                 UPtr<EstimatorManagerNew>&& estimator_manager,
                 VMCDriverInput&& vmc_input,
                 WFTrainDriverInput&& training_input,
                 WalkerConfigurations& walker_configurations,
                 TrialWaveFunction& student_prototype,
                 MCPopulation&& population,
                 const RefVector<RandomBase<FullPrecRealType>>& random_generators,
                 Communicate* communicator);

  WFTrainBatched(const WFTrainBatched&)            = delete;
  WFTrainBatched& operator=(const WFTrainBatched&) = delete;
  ~WFTrainBatched() override;

  /** Validate the all-electron training contract and establish native walkers. */
  void process(xmlNodePtr node) override;

  /** Pool fixed-version statistical blocks, then publish one update per iteration. */
  void run() override;

private:
  wftrain::TrainingTimers phase_timers_;

  QMCRunType getRunType() override { return QMCRunType::WF_TRAIN_BATCH; }

  VMCDriverInput vmcdriver_input_;
  WFTrainDriverInput training_input_;
  TrialWaveFunction& student_prototype_;
  std::string selected_provider_id_;
  wftrain::StructuredParameterProvider* selected_provider_ = nullptr;
  /// Driver-only opt-in authority; generic GPU component discovery/publication remains closed.
  std::unique_ptr<wftrain::StructuredParameterProvider> complex_gpu_provider_;
  /// Actual numerical component remains distinct from its driver-owned provider adapter.
  PsiFormerComplexWF* complex_training_component_ = nullptr;
  /// Resolved once during pre-draw process admission, never re-read mid-run.
  std::size_t resident_adam_budget_ = 0, resident_norm_bytes_ = 0;
  /// Complete SR cap pair and incremental peak, resolved before sampling.
  std::pair<std::size_t, std::size_t> resident_sr_budget_{0, 0};
  std::size_t resident_sr_incremental_bytes_ = 0;

  /// Student clones aligned one-for-one with the frozen-teacher crowd ParticleSets.
  std::vector<UPtrVector<TrialWaveFunction>> overlap_student_twfs_;
  std::unique_ptr<wftrain::DeterminantOverlapObjective> determinant_overlap_objective_;

  /// Frozen conventional SPO targets, evaluated without the teacher's Jastrow factors.
  std::unique_ptr<psiformer::OrbitalTargetEvaluator> orbital_target_;

  /// Per-crowd RNG contexts used by full initialization and post-update refreshes.
  UPtrVector<ContextForSteps> step_contexts_;

  /** Construct the reference view consumed by the shared full-refresh routine. */
  RefVector<ContextForSteps> getContextForStepsRefs() const;

  /** Allocate one stable RNG context for every initialized crowd. */
  void createStepContexts(int num_crowds);

  /** Clone the student onto every teacher configuration and freeze target semantics. */
  void initializeDeterminantOverlap();

  /// Freeze occupied or top-coefficient conventional orbitals for orbital-MSE training.
  void initializeOrbitalPretraining();

  /** Reject Hamiltonians outside the first all-electron kinetic-energy route. */
  void validateHamiltonianForTraining();

  /** Refresh all crowd wavefunction, drift, and local-energy caches after publication. */
  void refreshPublishedParameters(std::size_t parameter_version);
};

} // namespace qmcplusplus

#endif // QMCPLUSPLUS_WFTRAIN_BATCHED_H
