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
#include "QMCDrivers/WFTrain/WFTrainInferenceInput.h"
#include "QMCDrivers/WFTrain/TrainingTimers.h"

#include <cstddef>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
class DeterminantOverlapObjective;
class StructuredParameterProvider;
class TrainingWaveFunctionBinding;
}

namespace qmcplusplus
{
class PsiFormerComplexWF;
class PsiFormerWF;

namespace wftrain
{
/// Validate and fingerprint only the selected real pair native AE Ewald physics row.
std::string realPairCoulombIdentity(PsiFormerWF& component, ParticleSet& electrons, QMCHamiltonian& hamiltonian);
#ifdef QMC_COMPLEX
/// Fingerprint the selected paired AE full-cell Ewald physics after exact native-source validation.
std::string pairedPairCoulombIdentity(const PsiFormerComplexWF& component, ParticleSet& electrons, QMCHamiltonian& hamiltonian);
#endif
}

namespace psiformer
{
class OrbitalTargetEvaluator;
class ComplexOrbitalTargetEvaluator;
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
                 Communicate* communicator,
                 const TrialWaveFunction* frozen_teacher);

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
  WFTrainInferenceInput inference_input_;
  std::uint64_t inference_seed_ = 0;
  NewTimer* inference_timer_ = nullptr;
  TrialWaveFunction& student_prototype_;
  /// Pool-owned immutable target, independent of the selected sampling measure.
  const TrialWaveFunction* frozen_teacher_;
  std::string selected_provider_id_;
  wftrain::StructuredParameterProvider* selected_provider_ = nullptr;
  std::unique_ptr<wftrain::TrainingWaveFunctionBinding> conventional_binding_;
  /// Driver-only opt-in authority; generic GPU component discovery/publication remains closed.
  std::unique_ptr<wftrain::StructuredParameterProvider> complex_gpu_provider_;
  /// Actual numerical component remains distinct from its driver-owned provider adapter.
  PsiFormerComplexWF* complex_training_component_ = nullptr;
  /// Resolved once during pre-draw process admission, never re-read mid-run.
  std::size_t resident_adam_budget_ = 0, resident_norm_bytes_ = 0;
  bool resident_device_authority_ = false;
  /// Complete resident observation is a separate sole-AE W2 opt-in, not implied by parameter authority.
  bool resident_open_observation_ = false;
  std::array<int, 4> resident_open_scalar_slots_{}; // K, ee, ei, ii native observable columns
  const ParticleSet* resident_open_ions_ = nullptr; // immutable selected Hamiltonian source, owned by the pool
  std::size_t resident_snapshot_cap_ = 0, resident_successor_cap_ = 0, resident_authority_extra_bytes_ = 0;
  /// Complete SR cap pair and incremental peak, resolved before sampling.
  std::pair<std::size_t, std::size_t> resident_sr_budget_{0, 0};
  std::size_t resident_sr_incremental_bytes_ = 0;

  /// Student clones aligned one-for-one with the frozen-teacher crowd ParticleSets.
  std::vector<UPtrVector<TrialWaveFunction>> overlap_student_twfs_;
  std::unique_ptr<wftrain::DeterminantOverlapObjective> determinant_overlap_objective_;

  /// Frozen conventional SPO targets, evaluated without the teacher's Jastrow factors.
  std::unique_ptr<psiformer::OrbitalTargetEvaluator> orbital_target_;
#ifdef QMC_COMPLEX
  std::unique_ptr<psiformer::ComplexOrbitalTargetEvaluator> complex_orbital_target_;
#endif
  /// Frozen target metadata and separately charged explicit score-only CPU owner.
  std::uint64_t orbital_target_fingerprint_ = 0;
  std::size_t orbital_target_bytes_ = 0, complex_orbital_score_bytes_ = 0;

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

  /** Sample accepted parameters in a fresh isolated native VMC population. */
  bool runInference(std::size_t iteration, std::size_t parameter_version);
};

} // namespace qmcplusplus

#endif // QMCPLUSPLUS_WFTRAIN_BATCHED_H
