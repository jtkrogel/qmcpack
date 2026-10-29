//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include "Estimators/EstimatorInputDelegates.h"
#include "WFTrainBatched.h"

#include "Concurrency/ParallelExecutor.hpp"
#include "MemoryUsage.h"
#include "Message/CommOperators.h"
#include "Message/UniformCommunicateError.h"
#include "QMCDrivers/WalkerLogManager.h"
#include "QMCDrivers/WFTrain/DeterminantOverlapObjective.h"
#include "QMCDrivers/WFTrain/DistributedTrainingCheckpoint.h"
#include "QMCDrivers/WFTrain/EnergyGradientAccumulator.h"
#include "QMCDrivers/WFTrain/FirstOrderOptimizer.h"
#include "QMCDrivers/WFTrain/HighParameterTraining.h"
#include "QMCDrivers/WFTrain/GpuTrainingEpochReplay.h"
#include "QMCDrivers/WFTrain/GpuTrainingEpochObjective.h"
#include "QMCDrivers/WFTrain/JastrowNeuralTrainingBinding.h"
#include "QMCDrivers/WFTrain/AcceleratorRuntimePlacement.h"
#include "QMCDrivers/WFTrain/MatrixFreeOperator.h"
#include "QMCDrivers/WFTrain/SampledLinearMethod.h"
#include "QMCDrivers/WFTrain/AcceleratedLinearMethod.h"
#include "QMCDrivers/WFTrain/KFACPreconditioner.h"
#include "QMCDrivers/WFTrain/KFACOptimizer.h"
#include "QMCDrivers/WFTrain/KFACDenseSolver.h"
#include "QMCDrivers/WFTrain/MatrixFreeStochasticReconfiguration.h"
#include "QMCDrivers/WFTrain/KrylovPcgControl.h"
#include "QMCDrivers/WFTrain/ResidentEnergySRDiagnostics.h"
#include "QMCDrivers/WFTrain/MinimumStepStochasticReconfiguration.h"
#include "QMCDrivers/WFTrain/ParameterUpdateTransaction.h"
#include "QMCDrivers/WFTrain/ProposalSelection.h"
#include "QMCDrivers/WFTrain/NativeLineSearch.h"
#include "QMCDrivers/WFTrain/PreparedEnergyGradientProducer.h"
#include "QMCDrivers/WFTrain/NonLocalEnergyAccumulator.h"
#include "QMCDrivers/WFTrain/OrbitalPretraining.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"
#include "QMCDrivers/WFTrain/TrainingWaveFunctionBinding.h"
#include "QMCDrivers/WFTrain/WFTrainOptimizerInput.h"
#include "QMCDrivers/WFTrain/WFTrainSamplerCheckpoint.h"
#include "QMCDrivers/WFTrain/WFTrainInference.h"
#include "QMCDrivers/WFTrain/OrbitalInferenceMoments.h"
#include "QMCDrivers/WFTrain/ComplexOrbitalInferenceMoments.h"
#include "QMCHamiltonians/NonLocalECPotential.h"
#include "QMCHamiltonians/LocalECPotential.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCHamiltonians/CoulombPotential.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "Estimators/LocalEnergyEstimator.h"
#include "QMCHamiltonians/CoulombPBCAA.h"
#include "QMCHamiltonians/CoulombPBCAB.h"
#include "QMCWaveFunctions/Optimization/StructuredParameterProvider.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/FrozenReferenceResidualWF.h"
#include "QMCWaveFunctions/detail/PsiFormerMemoryTrace.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorPlanning.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOrbitalTargets.h"
#include "QMCWaveFunctions/Fermion/SlaterDet.h"
#include "QMCWaveFunctions/Fermion/MultiSlaterDetTableMethod.h"
#include "Utilities/RunTimeManager.h"
#ifdef ENABLE_CUDA
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceInference.h"
#ifdef QMC_COMPLEX
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceComplexValue.h"
#endif
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <unordered_map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef __linux__
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace qmcplusplus
{
namespace
{

std::size_t checkedMemoryAdd(std::size_t lhs, std::size_t rhs);
std::size_t checkedMemoryMultiply(std::size_t lhs, std::size_t rhs);

/// Collective control signal used only before a native mixed epoch can publish any update.
class NativeGpuEpochRetry final : public std::runtime_error
{
public:
  NativeGpuEpochRetry() : std::runtime_error("Native mixed numerical epoch requires complete FP64 replay") {}
};

/// Identify only declared numerical precision failures; geometry/vendor/source failures remain fatal.
bool isMixedNumericalRejection(const std::exception_ptr& failure)
{
  if (!failure) return false;
  try { std::rethrow_exception(failure); }
#ifdef ENABLE_CUDA
  catch (const psiformer::device::MixedPrecisionRetryRequired&) { return true; }
#ifdef QMC_COMPLEX
  catch (const psiformer::device::MixedComplexValueFailure& error)
  { return error.stage() != psiformer::device::ComplexValueFailureStage::PHYSICAL_ENVELOPE; }
#endif
  catch (const psiformer::device::InferenceStatusError& error) { return error.stage() != 0; }
#endif
  catch (...) { return false; }
}

/// Every rank agrees on retry versus fatal failure before residual/factor reduction or publication.
void requireNativeEpochReadiness(Communicate& comm, const std::exception_ptr& failure, bool strict_mixed)
{
  int retry = failure && isMixedNumericalRejection(failure) ? 1 : 0;
  int fatal = failure && !retry ? 1 : 0;
  comm.allreduce(retry);
  comm.allreduce(fatal);
  if (retry && !fatal && strict_mixed)
    throw NativeGpuEpochRetry();
  if (retry || fatal)
  {
    if (failure) std::rethrow_exception(failure);
    throw UniformCommunicateError("WFTrainBatched: another rank rejected the pending native precision epoch");
  }
}

/** Retain one crowd snapshot without retaining a samples-by-parameters matrix.
 *
 * Particle copies preserve positions, masses and total log gradients. Frozen
 * teacher clones are needed only for overlap objectives; neural parameters remain
 * shared and read-only until the iteration's update transaction.
 */
struct RetainedCrowdSamples
{
  std::size_t crowd_index = 0;
  std::vector<std::unique_ptr<ParticleSet>> particles;
  std::vector<std::unique_ptr<TrialWaveFunction>> teachers;
  std::vector<double> local_energies;
  /// Optional current-version/value evidence only for the admitted resident single snapshot.
  std::optional<std::size_t> replay_version;
  std::vector<std::array<double, 2>> replay_log_phase;
  psiformer::memory::HostCapacity trace_scalar_capacity;
  psiformer::memory::HostCapacity trace_particle_capacity;
};

#ifdef QMC_COMPLEX
using ImmediateReplayComponent = PsiFormerComplexWF;
#else
using ImmediateReplayComponent = PsiFormerWF;
#endif

/// Only the copied fresh derivatives are new; retained snapshots already own the physical metadata.
struct ImmediateFullReplayLane
{
  const ImmediateReplayComponent* component;
  const TrialWaveFunction* wavefunction;
  const ParticleSet *live, *saved;
  std::size_t version;
  double log, phase;
  int boundary;
  std::array<double, 12> cell_twist;
  std::vector<WaveFunctionComponent::ValueType> gradient, laplacian;
#ifdef QMC_COMPLEX
  PsiFormerComplexWF::ImmediateFullReplayContext context;
#endif
};

/// Synchronous one-use HOST evidence, never a persistent cache or a ParticleSet clone.
struct ImmediateFullReplayReceipt
{
  const Crowd* crowd;
  const ParticleSet* source;
  int source_boundary;
  std::array<double, 12> source_cell_twist;
  std::vector<ParticleSet::PosType> source_positions;
  std::vector<double> source_charges;
  std::vector<int> source_groups;
  std::vector<std::string> source_species;
  std::vector<ImmediateFullReplayLane> lanes;
  bool consumed = false;

  /// Conservative owned HOST capacity charge; no allocator/RSS or aggregate HOST-fit claim.
  std::size_t retainedBytes() const
  {
    auto bytes = checkedMemoryAdd(sizeof(*this), checkedMemoryMultiply(lanes.capacity(), sizeof(lanes[0])));
    for (const auto& lane : lanes)
      bytes = checkedMemoryAdd(bytes, checkedMemoryMultiply(
          checkedMemoryAdd(lane.gradient.capacity(), lane.laplacian.capacity()), sizeof(lane.gradient[0])));
    bytes = checkedMemoryAdd(bytes, checkedMemoryMultiply(source_positions.capacity(), sizeof(source_positions[0])));
    bytes = checkedMemoryAdd(bytes, checkedMemoryMultiply(source_charges.capacity(), sizeof(source_charges[0])));
    bytes = checkedMemoryAdd(bytes, checkedMemoryMultiply(source_groups.capacity(), sizeof(source_groups[0])));
    bytes = checkedMemoryAdd(bytes, checkedMemoryMultiply(source_species.capacity(), sizeof(source_species[0])));
    // Counting SSO capacity again is conservative metadata accounting, not measured physical memory.
    for (const auto& name : source_species)
      bytes = checkedMemoryAdd(bytes, checkedMemoryAdd(name.capacity(), 1));
    return bytes;
  }
};

/// SimulationCell is shared by ParticleSet copies: capture numeric values explicitly.
std::array<double, 12> immediateReplayCell(const ParticleSet& particles)
{
  std::array<double, 12> result;
  for (int axis = 0; axis < 3; ++axis)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      result[3 * axis + dimension] = particles.getLattice().R(axis, dimension);
    result[9 + axis] = particles.getTwist()[axis];
  }
  return result;
}

/// The component-specific proof is read-only; no temporary image handles escape this call.
bool immediateReplayMatches(const ImmediateFullReplayLane& lane, const ParticleSet& live, const ParticleSet* source)
{
#ifdef QMC_COMPLEX
  return lane.context.source == source && lane.context.version == lane.version &&
      lane.component->canReuseImmediateFullSpatialReplay(live, *lane.saved, lane.gradient, lane.laplacian,
                                                         lane.context, lane.log, lane.phase);
#else
  return lane.component->canReuseImmediateFullSpatialReplay(live, *lane.saved, lane.gradient, lane.laplacian,
                                                           source, lane.version, lane.log, lane.phase);
#endif
}

/// Capture only after fresh replay, before any accepted-grid draw; unproved or oversized evidence is a miss.
std::optional<ImmediateFullReplayReceipt> captureImmediateFullReplay(
    Crowd& crowd, const RetainedCrowdSamples& samples, std::size_t version, std::size_t available_bytes)
{
  const auto& electrons = crowd.get_walker_elecs();
  const auto& wavefunctions = crowd.get_walker_twfs();
  if (!crowd.size() || samples.particles.size() != crowd.size() || !available_bytes)
    return std::nullopt;
  const auto& components = wavefunctions[0].get().getOrbitals();
  const auto* leader = components.size() == 1 ? dynamic_cast<const ImmediateReplayComponent*>(components[0].get()) : nullptr;
#ifdef QMC_COMPLEX
  const auto* source = leader ? &leader->sourceParticles() : nullptr;
#else
  const auto* source = leader ? leader->validatedReplaySource() : nullptr;
#endif
  if (!source)
    return std::nullopt;
  const auto ions = static_cast<std::size_t>(source->getTotalNum());
  const auto& species = source->getSpeciesSet();
  const auto charge = species.findAttribute("charge");
  if (charge < 0 || source->R.size() != ions || source->GroupID.size() != ions)
    return std::nullopt;
  auto required = checkedMemoryAdd(sizeof(ImmediateFullReplayReceipt),
      checkedMemoryMultiply(crowd.size(), sizeof(ImmediateFullReplayLane)));
  required = checkedMemoryAdd(required, checkedMemoryMultiply(ions,
      sizeof(ParticleSet::PosType) + sizeof(double) + sizeof(int)));
  for (int group = 0; group < species.getTotalNum(); ++group)
    required = checkedMemoryAdd(required, checkedMemoryAdd(sizeof(std::string),
        // Include the supported string ABI's empty/SSO capacity before allocating.
        // Actual retained capacities are checked again before the receipt is usable.
        checkedMemoryAdd(std::max(species.getSpeciesName(group).capacity(), std::string{}.capacity()), 1)));
  for (const auto& particles : electrons)
    required = checkedMemoryAdd(required, checkedMemoryMultiply(
        static_cast<std::size_t>(particles.get().getTotalNum()), 4 * sizeof(WaveFunctionComponent::ValueType)));
  if (required > available_bytes)
    return std::nullopt;

  ImmediateFullReplayReceipt receipt{&crowd, source, source->getLattice().getSuperCellEnum(),
                                    immediateReplayCell(*source)};
  receipt.source_positions.reserve(ions);
  receipt.source_charges.reserve(ions);
  receipt.source_groups.reserve(ions);
  receipt.source_species.reserve(species.getTotalNum());
  receipt.lanes.reserve(crowd.size());
  for (int group = 0; group < species.getTotalNum(); ++group)
    receipt.source_species.push_back(species.getSpeciesName(group));
  for (std::size_t ion = 0; ion < ions; ++ion)
  {
    const auto group = source->GroupID[ion];
    if (group < 0 || group >= species.getTotalNum())
      return std::nullopt;
    receipt.source_positions.push_back(source->R[ion]);
    receipt.source_groups.push_back(group);
    receipt.source_charges.push_back(species(charge, group));
  }
  for (std::size_t sample = 0; sample < crowd.size(); ++sample)
  {
    const auto& live = electrons[sample].get();
    const auto& twf = wavefunctions[sample].get();
    const auto& orbitals = twf.getOrbitals();
    const auto* component = orbitals.size() == 1 ? dynamic_cast<const ImmediateReplayComponent*>(orbitals[0].get()) : nullptr;
    if (!component)
      return std::nullopt;
    ImmediateFullReplayLane lane{component, &twf, &live, samples.particles[sample].get(), version,
        twf.getLogPsi(), twf.getPhase(), live.getLattice().getSuperCellEnum(), immediateReplayCell(live)};
#ifdef QMC_COMPLEX
    lane.context = component->immediateFullReplayContext();
#endif
    const auto count = static_cast<std::size_t>(live.getTotalNum());
    if (live.G.size() != count || live.L.size() != count)
      return std::nullopt;
    lane.gradient.reserve(3 * count);
    lane.laplacian.reserve(count);
    for (std::size_t electron = 0; electron < count; ++electron)
    {
      for (int dimension = 0; dimension < 3; ++dimension)
        lane.gradient.push_back(live.G[electron][dimension]);
      lane.laplacian.push_back(live.L[electron]);
    }
    if (!immediateReplayMatches(lane, live, source))
      return std::nullopt;
    receipt.lanes.push_back(std::move(lane));
  }
  if (receipt.retainedBytes() > available_bytes)
    return std::nullopt;
  return std::optional<ImmediateFullReplayReceipt>(std::move(receipt));
}

/// Recheck the complete crowd after mw_update; a miss consumes the evidence and uses canonical refresh.
bool consumeImmediateFullReplay(Crowd& crowd, ImmediateFullReplayReceipt& receipt)
{
  if (std::exchange(receipt.consumed, true) || receipt.crowd != &crowd || receipt.lanes.size() != crowd.size())
    return false;
  const auto& source = *receipt.source;
  const auto& species = source.getSpeciesSet();
  const auto charge = species.findAttribute("charge");
  if (charge < 0 || source.getLattice().getSuperCellEnum() != receipt.source_boundary ||
      immediateReplayCell(source) != receipt.source_cell_twist ||
      source.R.size() != receipt.source_positions.size() || source.GroupID.size() != receipt.source_groups.size() ||
      species.getTotalNum() != receipt.source_species.size())
    return false;
  for (int group = 0; group < species.getTotalNum(); ++group)
    if (species.getSpeciesName(group) != receipt.source_species[group])
      return false;
  for (std::size_t ion = 0; ion < receipt.source_positions.size(); ++ion)
  {
    const auto group = source.GroupID[ion];
    if (group != receipt.source_groups[ion] || group < 0 || group >= species.getTotalNum() ||
        species(charge, group) != receipt.source_charges[ion])
      return false;
    for (int dimension = 0; dimension < 3; ++dimension)
      if (source.R[ion][dimension] != receipt.source_positions[ion][dimension])
        return false;
  }
  for (std::size_t sample = 0; sample < crowd.size(); ++sample)
  {
    const auto& lane = receipt.lanes[sample];
    const auto& live = crowd.get_walker_elecs()[sample].get();
    const auto& twf = crowd.get_walker_twfs()[sample].get();
    const auto& components = twf.getOrbitals();
    if (lane.live != &live || lane.wavefunction != &twf || components.size() != 1 ||
        components[0].get() != lane.component || twf.getLogPsi() != lane.log || twf.getPhase() != lane.phase ||
        live.getLattice().getSuperCellEnum() != lane.boundary || immediateReplayCell(live) != lane.cell_twist ||
        !immediateReplayMatches(lane, live, receipt.source))
      return false;
  }
  return true;
}

/// The unchanged scalar accumulator/temporary overlap is reserved before receipt allocation.
std::size_t scoreOnlyECPCaptureNamedBytes(const wftrain::StreamingDerivativeOperator& score, std::size_t samples)
{
  const auto temporary = checkedMemoryMultiply(samples, 3 * sizeof(wftrain::DerivativeValue) + 2 * sizeof(double));
  return checkedMemoryAdd(score.storageDiagnostics().retained_numeric_bytes, temporary);
}

/// Prepare exactly the ordinary score-only owner; the caller holds both crowd resource guards.
std::unique_ptr<wftrain::StreamingDerivativeOperator> makeRealGpuScoreBatchUnderLocks(
    Crowd& crowd, std::size_t batch, std::size_t offset, std::size_t chunk)
{
  const auto& twfs = crowd.get_walker_twfs();
  const auto& electrons = crowd.get_walker_elecs();
  RefVectorWithLeader<ParticleSet> particles(electrons[0], electrons);
  const auto& leader_orbitals = twfs[0].get().getOrbitals();
  const auto* real = leader_orbitals.size() == 1 ? dynamic_cast<const PsiFormerWF*>(leader_orbitals[0].get()) : nullptr;
  if (!real)
    throw std::invalid_argument("GPU score-only ECP requires one real component per walker");
  RefVectorWithLeader<WaveFunctionComponent> components(*leader_orbitals[0]);
  for (const auto& wavefunction : twfs)
  {
    if (wavefunction.get().getOrbitals().size() != 1)
      throw std::invalid_argument("GPU score-only ECP requires one real component per walker");
    components.push_back(*wavefunction.get().getOrbitals()[0]);
  }
  return real->makeGpuScoreStreamingDerivativeOperator(components, particles, batch, offset, chunk);
}

/// The early factory copies fresh replay state but performs no derivatives or accepted-grid RNG.
std::unique_ptr<wftrain::StreamingDerivativeOperator> prepareImmediateGpuScoreBatch(
    Crowd& crowd, std::size_t batch, std::size_t offset, std::size_t chunk)
{
  const auto& electrons = crowd.get_walker_elecs();
  const auto& twfs = crowd.get_walker_twfs();
  RefVectorWithLeader<ParticleSet> particles(electrons[0], electrons);
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(twfs[0], twfs);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(crowd.getSharedResource().pset_res, particles);
  ResourceCollectionTeamLock<TrialWaveFunction> wavefunction_lock(crowd.getSharedResource().twf_res, wavefunctions);
  return makeRealGpuScoreBatchUnderLocks(crowd, batch, offset, chunk);
}

/// One scalar private evaluator reused by every retained conventional replay view.
struct ConventionalReplayWorker
{
  std::unique_ptr<ParticleSet> particles;
  std::unique_ptr<TrialWaveFunction> wavefunction;
  ConventionalReplayWorker(const wftrain::TrainingWaveFunctionBinding& binding, const ParticleSet& source)
      : particles(std::make_unique<ParticleSet>(source)), wavefunction(binding.makePrivateClone(*particles))
  {}
};

class ConventionalReplay final : public wftrain::LegacyWaveFunctionReplay
{
public:
  ConventionalReplay(std::shared_ptr<ConventionalReplayWorker> worker, const RetainedCrowdSamples& samples)
      : worker_(std::move(worker)), samples_(samples)
  {}
  std::size_t sampleCount() const noexcept override { return samples_.particles.size(); }
  wftrain::LegacyWaveFunctionReplaySample prepareSample(std::size_t index) override
  {
    worker_->particles->R = samples_.particles.at(index)->R;
    worker_->particles->update();
    return {*worker_->wavefunction, *worker_->particles};
  }
private:
  std::shared_ptr<ConventionalReplayWorker> worker_;
  const RetainedCrowdSamples& samples_;
};

/** Copy an observed population before later MC moves overwrite its positions. */
RetainedCrowdSamples retainCrowdSamples(Crowd& crowd, std::size_t index, bool overlap,
                                       bool exact_energy_capacity = false,
                                       std::optional<std::size_t> replay_version = std::nullopt)
{
  RetainedCrowdSamples result;
  result.crowd_index = index;
  result.replay_version = replay_version;
  const auto& electrons = crowd.get_walker_elecs();
  const auto& wavefunctions = crowd.get_walker_twfs();
  const auto& walkers = crowd.get_walkers();
  if (exact_energy_capacity)
    result.local_energies.reserve(crowd.size());
  if (replay_version)
    result.replay_log_phase.reserve(crowd.size());
  for (std::size_t sample = 0; sample < crowd.size(); ++sample)
  {
    result.particles.push_back(std::make_unique<ParticleSet>(electrons[sample]));
    result.particles.back()->update();
    result.local_energies.push_back(walkers[sample].get().Properties(WalkerProperties::LOCALENERGY));
    if (replay_version)
      result.replay_log_phase.push_back({walkers[sample].get().Properties(WalkerProperties::LOGPSI),
                                         walkers[sample].get().Properties(WalkerProperties::SIGN)});
    if (overlap)
      result.teachers.push_back(wavefunctions[sample].get().makeClone(*result.particles.back()));
  }
  if (psiformer::memory::enabled())
    result.trace_scalar_capacity.snapshotAt(__FILE__, __LINE__, "RetainedCrowdSamples::scalar_vectors",
        result.local_energies.capacity() * sizeof(double) +
        result.replay_log_phase.capacity() * sizeof(result.replay_log_phase[0]),
        reinterpret_cast<std::uintptr_t>(&crowd), index);
  if (psiformer::memory::enabled())
  {
    std::size_t bytes = 0;
    for (const auto& particles : result.particles)
      bytes += particles->R.capacity() * sizeof(particles->R[0]) +
          particles->G.capacity() * sizeof(particles->G[0]) + particles->L.capacity() * sizeof(particles->L[0]);
    result.trace_particle_capacity.snapshotAt(__FILE__, __LINE__, "RetainedCrowdSamples::particle_R_G_L", bytes,
        reinterpret_cast<std::uintptr_t>(&crowd), index);
  }
  return result;
}

/** Stream raw-orbital targets and VJPs without retaining a samples-by-parameters matrix.
 * Frozen CPU SPO targets and the student's selected CPU/GPU backend share only positions. */
template<class Student, class Target>
class SampledOrbitalPretrainingProducer final : public wftrain::OrbitalPretrainingProducer
{
public:
  SampledOrbitalPretrainingProducer(Student& student,
                                    Target& target,
                                    const std::vector<RetainedCrowdSamples>& samples,
                                    wftrain::OrbitalLossNormalization normalization =
                                        wftrain::OrbitalLossNormalization::SPIN_SECTOR_SUM)
      : student_(student), target_(target), samples_(samples),
        gradient_(student.parameterSchema().parameterCount()), normalization_(normalization)
  {}

  wftrain::TrainingCapabilities capabilities() const noexcept override
  {
    return {wftrain::TrainingCapability::REAL_PARAMETERS,
            wftrain::TrainingCapability::ORBITAL_MSE_VJP};
  }

  void accumulate(const wftrain::StructuredParameterSnapshot& parameters,
                  wftrain::OrbitalPretrainingAccumulator& accumulator) override
  {
    // One target matrix and one canonical gradient are reused for all retained samples.
    for (const auto& crowd : samples_)
      for (const auto& particles : crowd.particles)
      {
        double loss;
#ifdef QMC_COMPLEX
        if constexpr (std::is_same_v<Student, PsiFormerComplexWF>)
          loss = student_.evaluateOrbitalMSE(
              *particles, target_.evaluate(*particles), gradient_, parameters.version, normalization_);
        else
#endif
          loss = student_.evaluateOrbitalMSE(
              *particles, target_.evaluate(*particles), gradient_, parameters.version);
        accumulator.addSample(loss, parameters.version, {gradient_.data(), gradient_.size()});
      }
  }

  /// Explicit producer scratch; the accumulator and result are charged separately.
  std::size_t retainedBytes() const noexcept { return gradient_.capacity() * sizeof(double); }

private:
  Student& student_;
  Target& target_;
  const std::vector<RetainedCrowdSamples>& samples_;
  std::vector<double> gradient_;
  wftrain::OrbitalLossNormalization normalization_;
};

/** Replay a retained snapshot on resident clones and reconstruct coherent VGL.
 * Iterating snapshots in collection order leaves the live chains at their final
 * positions. No MC proposals or parameter publications occur during replay.
 * An exact current resident snapshot may reuse FULL; all cheap replay operations remain.
 */
bool replayCrowdSamples(Crowd& crowd, const RetainedCrowdSamples& samples,
                       std::optional<std::size_t> reuse_version = std::nullopt)
{
  const auto& electrons = crowd.get_walker_elecs();
  const auto& wavefunctions = crowd.get_walker_twfs();
  const auto& walkers = crowd.get_walkers();
  RefVectorWithLeader<ParticleSet> particle_batch(electrons[0], electrons);
  RefVectorWithLeader<TrialWaveFunction> wavefunction_batch(wavefunctions[0], wavefunctions);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(crowd.getSharedResource().pset_res, particle_batch);
  ResourceCollectionTeamLock<TrialWaveFunction> wavefunction_lock(crowd.getSharedResource().twf_res, wavefunction_batch);
  // Only the admitted resident single snapshot can still be the live chain.
  bool reuse_full = reuse_version && samples.replay_version == reuse_version &&
      samples.particles.size() == crowd.size() && samples.local_energies.size() == crowd.size() &&
      samples.replay_log_phase.size() == crowd.size();
  for (std::size_t sample = 0; reuse_full && sample < crowd.size(); ++sample)
  {
    const auto& live = electrons[sample].get();
    const auto& saved = *samples.particles[sample];
    reuse_full = live.R.size() == saved.R.size() && live.spins.size() == live.R.size() &&
        live.spins.size() == saved.spins.size();
    for (std::size_t electron = 0; reuse_full && electron < live.R.size(); ++electron)
    {
      reuse_full = wftrain::isFiniteTrainingReal(live.spins[electron]) &&
          wftrain::isFiniteTrainingReal(saved.spins[electron]) && live.spins[electron] == saved.spins[electron];
      for (int dimension = 0; reuse_full && dimension < OHMMS_DIM; ++dimension)
        reuse_full = wftrain::isFiniteTrainingReal(live.R[electron][dimension]) &&
            wftrain::isFiniteTrainingReal(saved.R[electron][dimension]) &&
            live.R[electron][dimension] == saved.R[electron][dimension];
    }
  }
  for (std::size_t sample = 0; sample < crowd.size(); ++sample)
    electrons[sample].get().R = samples.particles[sample]->R;
  ParticleSet::mw_update(particle_batch);
  for (std::size_t sample = 0; reuse_full && sample < crowd.size(); ++sample)
  {
    const auto& twf = wavefunctions[sample].get();
    const auto& orbitals = twf.getOrbitals();
    const auto* component = orbitals.size() == 1 ? dynamic_cast<const PsiFormerWF*>(orbitals.front().get()) : nullptr;
    const auto& captured = samples.replay_log_phase[sample];
    reuse_full = component && wftrain::isFiniteTrainingReal(captured[0]) && wftrain::isFiniteTrainingReal(captured[1]) &&
        wftrain::isFiniteTrainingReal(twf.getLogPsi()) && wftrain::isFiniteTrainingReal(twf.getPhase()) &&
        wftrain::isFiniteTrainingReal(walkers[sample].get().Properties(WalkerProperties::LOGPSI)) &&
        wftrain::isFiniteTrainingReal(walkers[sample].get().Properties(WalkerProperties::SIGN)) &&
        twf.getLogPsi() == captured[0] && twf.getPhase() == captured[1] &&
        walkers[sample].get().Properties(WalkerProperties::LOGPSI) == captured[0] &&
        walkers[sample].get().Properties(WalkerProperties::SIGN) == captured[1] &&
        wftrain::isFiniteTrainingReal(samples.local_energies[sample]) &&
        wftrain::isFiniteTrainingReal(walkers[sample].get().Properties(WalkerProperties::LOCALENERGY)) &&
        walkers[sample].get().Properties(WalkerProperties::LOCALENERGY) == samples.local_energies[sample] &&
        component->canReuseFullSpatialReplay(electrons[sample], *samples.particles[sample],
                                             *reuse_version, captured[0], captured[1]);
  }
  if (!reuse_full)
    TrialWaveFunction::mw_evaluateLog(wavefunction_batch, particle_batch);
  if (reuse_version)
    app_log() << "Resident ENERGY replay full_elided=" << reuse_full
              << " full_evaluations=" << !reuse_full << " version=" << *reuse_version << std::endl;
  for (std::size_t sample = 0; sample < crowd.size(); ++sample)
  {
    electrons[sample].get().saveWalker(walkers[sample]);
    walkers[sample].get().resetProperty(wavefunctions[sample].get().getLogPsi(),
                                       wavefunctions[sample].get().getPhase(),
                                       samples.local_energies[sample]);
  }
  return !reuse_full;
}

/** Record publication without performing fallible work in the transaction callback. */
class DirtyCacheObserver final : public wftrain::ParameterUpdateObserver
{
public:
  void parametersPublished(std::size_t new_version) noexcept override
  {
    dirty_             = true;
    published_version_ = new_version;
  }

  bool dirty() const noexcept { return dirty_; }
  std::size_t publishedVersion() const noexcept { return published_version_; }
  void clear() noexcept { dirty_ = false; }

private:
  bool dirty_ = false;
  std::size_t published_version_ = 0;
};

/** Ensure a transient SPRING factor binding never outlives its prepared owners. */
class SPRINGFactorBindingGuard
{
public:
  explicit SPRINGFactorBindingGuard(wftrain::SPRINGUpdateRule& optimizer) noexcept
      : optimizer_(optimizer)
  {}

  ~SPRINGFactorBindingGuard() { optimizer_.unbindFactor(); }

  SPRINGFactorBindingGuard(const SPRINGFactorBindingGuard&)            = delete;
  SPRINGFactorBindingGuard& operator=(const SPRINGFactorBindingGuard&) = delete;

private:
  wftrain::SPRINGUpdateRule& optimizer_;
};

/// Feed provider-owned affine rows into the generic KFAC factor accumulator.
class LiveAffineObserver final : public wftrain::AffineScoreObserver
{
public:
  explicit LiveAffineObserver(wftrain::KFACFactorAccumulator& accumulator) : accumulator_(accumulator) {}

  void observe(const wftrain::AffineScoreObservation& observation) override
  {
    const auto& block = accumulator_.registry().affineBlocks().at(observation.affine_block);
    if (observation.row_count > std::numeric_limits<std::size_t>::max() /
            std::max(block.input_width, block.output_width))
      throw std::overflow_error("Live KFAC observation extent overflow");
    accumulator_.addObservation({observation.affine_block,
        {observation.activations, observation.row_count * block.input_width},
        {observation.sensitivities, observation.row_count * block.output_width},
        observation.row_count,
        observation.imaginary_sensitivities
            ? wftrain::DerivativeArrayView<const wftrain::DerivativeReal>{observation.imaginary_sensitivities,
                observation.row_count * block.output_width}
            : wftrain::DerivativeArrayView<const wftrain::DerivativeReal>{}});
  }

private:
  wftrain::KFACFactorAccumulator& accumulator_;
};

/// Keep optimizer calls in the driver library while DEVICE collection owns its banks.
class LiveDeviceAffineSink final : public PsiFormerWF::DeviceAffineStatisticsSink
{
public:
  explicit LiveDeviceAffineSink(wftrain::KFACFactorAccumulator& statistics) : statistics_(statistics)
  {
    if (statistics_.sampleActive() || statistics_.sampleCount() || statistics_.globallyReduced() ||
        statistics_.affineStatisticsComplete())
      throw std::invalid_argument("Device affine sink requires fresh precomputed statistics");
  }

  void beginSample() override { statistics_.beginSample(1); }

  void addFallbackScore(const std::vector<double>& score) override
  {
    statistics_.addFallbackScores({score.data(), score.size()});
  }

  /// Preserve both HOST fallback channels while affine sums remain provider-owned.
  void addComplexFallbackScore(const std::vector<double>& real, const std::vector<double>& imaginary) override
  {
    statistics_.addFallbackScores({real.data(), real.size()}, {imaginary.data(), imaginary.size()});
  }

  void endSample() override { statistics_.endSample(); }

  void abortSample() noexcept override { statistics_.abortSample(); }

  void import(std::vector<wftrain::KFACFactorStatistics> factors) override
  {
    statistics_.importAffineStatistics(statistics_.registry().fingerprint(), statistics_.parameterVersion(),
        std::move(factors));
  }

private:
  wftrain::KFACFactorAccumulator& statistics_;
};

wftrain::KFACBlockRegistry makeLiveKFACRegistry(const wftrain::StructuredParameterProvider& provider)
{
  const auto& schema = provider.parameterSchema();
  const auto metadata = provider.affineParameterBlocks();
  if (metadata.empty()) throw std::invalid_argument("Provider lacks live KFAC affine observations");
  std::vector<wftrain::KFACAffineBlockDescriptor> affine;
  std::vector<bool> covered(schema.blocks().size(), false);
  for (const auto& block : metadata)
  {
    affine.push_back({block.id, block.weight_block, block.bias_block, block.input_width, block.output_width});
    covered.at(block.weight_block) = true;
    if (block.bias_block != wftrain::NO_AFFINE_BIAS_BLOCK) covered.at(block.bias_block) = true;
  }
  std::vector<std::size_t> fallback;
  for (std::size_t i = 0; i < covered.size(); ++i)
    if (schema.blocks()[i].trainable && !covered[i]) fallback.push_back(i);
  return {schema, std::move(affine), std::move(fallback), provider.genericKFACParameterBlocks()};
}

/// Form a stateless Kronecker-preconditioned gradient proposal.
class LiveKFACUpdateRule final : public wftrain::TrainingUpdateRule
{
public:
  LiveKFACUpdateRule(const wftrain::KFACPreconditioner& inverse, double rate, double norm_constraint = 0.0)
      : inverse_(inverse), rate_(rate), norm_constraint_(norm_constraint) {}

  wftrain::StructuredParameterSnapshot propose(const wftrain::StructuredParameterSchema& schema,
      const wftrain::StructuredParameterSnapshot& parameters, wftrain::ParameterGradientView objective) override
  {
    std::vector<wftrain::DerivativeValue> gradient(parameters.values.size()), direction(gradient.size());
    for (std::size_t i = 0; i < gradient.size(); ++i) gradient[i] = objective.gradient[i];
    inverse_.apply({schema, parameters.version, {gradient.data(), gradient.size()}},
                   {direction.data(), direction.size()});
    if (norm_constraint_ > 0.0)
    {
      double product = 0.0;
      for (std::size_t i = 0; i < gradient.size(); ++i)
        product += (std::conj(gradient[i]) * direction[i]).real();
      metric_ = rate_ * rate_ * product;
      if (!wftrain::isFiniteTrainingReal(metric_) || metric_ < 0.0)
        throw std::runtime_error("KFAC damped gradient metric is invalid");
      scale_ = metric_ > norm_constraint_ ? std::sqrt(norm_constraint_ / metric_) : 1.0;
    }
    auto candidate = parameters;
    for (std::size_t i = 0; i < direction.size(); ++i)
      candidate.values[i] -= rate_ * scale_ * direction[i].real();
    return candidate;
  }

  double constraintMetric() const noexcept { return metric_; }
  double constraintScale() const noexcept { return scale_; }

private:
  const wftrain::KFACPreconditioner& inverse_;
  double rate_;
  double norm_constraint_;
  double metric_ = 0.0;
  double scale_ = 1.0;
};

/// Identify the first-order family using the common live gradient producer.
bool isFirstOrder(WFTrainDriverInput::Optimizer optimizer)
{
  using O = WFTrainDriverInput::Optimizer;
  return optimizer == O::ADAM || optimizer == O::AMSGRAD || optimizer == O::SGD ||
      optimizer == O::MOMENTUM_SGD || optimizer == O::RMSPROP;
}

/// Collective sampling requires the component's complete selected-proposal contract.
bool supportsNativeEnergyMove(const VMCDriverInput& input, const WaveFunctionComponent& component)
{
  return input.get_move_kind() == VMCDriverInput::MoveKind::PBYP ||
      ((input.get_move_kind() == VMCDriverInput::MoveKind::ALL_ELECTRON ||
        input.get_move_kind() == VMCDriverInput::MoveKind::N_ELECTRON) && component.supportsMultiParticleMoves());
}

/// Fixed first-order sample weights permit an inline ECP response without future replay.
bool usesInlinePairedECPResponse(const WFTrainDriverInput& input)
{
  return isFirstOrder(input.getOptimizer()) && input.getObjective() == WFTrainDriverInput::Objective::ENERGY &&
      input.getLineSearch() == WFTrainDriverInput::LineSearch::OFF && !wftrain::usesEnergyScoreEstimator(input, true);
}

/** Build a first-order learning-rate entry for every trainable schema group. */
wftrain::FirstOrderOptimizerOptions makeFirstOrderOptions(const wftrain::StructuredParameterSchema& schema,
                                                    const WFTrainDriverInput& input)
{
  wftrain::FirstOrderOptimizerOptions options;
  options.method     = wftrain::FirstOrderMethod::ADAM;
  switch (input.getOptimizer())
  {
  case WFTrainDriverInput::Optimizer::SGD: options.method = wftrain::FirstOrderMethod::SGD; break;
  case WFTrainDriverInput::Optimizer::MOMENTUM_SGD: options.method = wftrain::FirstOrderMethod::MOMENTUM_SGD; break;
  case WFTrainDriverInput::Optimizer::RMSPROP: options.method = wftrain::FirstOrderMethod::RMSPROP; break;
  case WFTrainDriverInput::Optimizer::AMSGRAD: options.method = wftrain::FirstOrderMethod::AMSGRAD; break;
  default: break;
  }
  options.momentum_decay = input.getMomentumDecay();
  options.rms_decay = input.getRMSDecay();
  options.adam_beta1 = input.getAdamBeta1();
  options.adam_beta2 = input.getAdamBeta2();
  options.epsilon    = input.getEpsilon();

  std::set<std::string> update_groups;
  for (const wftrain::ParameterBlockDescriptor& block : schema.blocks())
    if (block.trainable)
      update_groups.insert(block.update_group);
  for (const std::string& update_group : update_groups)
    options.learning_rates.push_back({update_group, input.getLearningRate()});
  return options;
}

#ifdef QMC_COMPLEX
/// Narrow driver authority: no parameter copy/lane and no change to generic GPU component discovery.
class ComplexGpuTrainingProvider final : public wftrain::StructuredParameterProvider,
                                         public wftrain::PreparedParameterPublicationContributor
{
public:
  /// Publication receives only the exact admitted successor-generation caps, not derivative ownership.
  ComplexGpuTrainingProvider(PsiFormerComplexWF& component, std::size_t device, std::size_t host)
      : component_(component), device_(device), host_(host)
  {}

  /// Keep the actual component's canonical REAL schema and lifetime.
  const wftrain::StructuredParameterSchema& parameterSchema() const noexcept override
  {
    return component_.parameterSchema();
  }

  /// Snapshot the same atomic model used by actual walker clones.
  wftrain::StructuredParameterSnapshot snapshotParameters() const override
  {
    return component_.snapshotParameters();
  }

  /// Numerical products run on GPU even though completed complex chunks are emitted on HOST.
  bool usesAcceleratorParameterProducts() const noexcept override
  {
    return true;
  }

  /// Preserve the model's typed weight selection, including imaginary head matrices.
  std::vector<std::string> neuralWeightBlocks() const override
  {
    return component_.neuralWeightBlocks();
  }

  /// Factors use the explicitly prepared HOST affine tape of this same model.
  std::vector<wftrain::AffineParameterBlock> affineParameterBlocks() const override
  {
    return component_.hybridAffineParameterBlocks();
  }

  /// Preserve exact diagonal curvature roles in the component's real schema.
  std::vector<std::size_t> genericKFACParameterBlocks() const override
  {
    return component_.genericKFACParameterBlocks();
  }

  /// Read both score channels under the same model authority used by publication.
  void observeComplexAffineScores(const ParticleSet& particles, wftrain::AffineScoreObserver& observer,
      std::vector<double>& real_score, std::vector<double>& imaginary_score) const override
  {
    component_.observeHybridComplexAffineScores(particles, observer, real_score, imaginary_score);
  }

  /// Reuse the qualified old-version transaction and complete new-generation publication.
  std::size_t publishParameters(const wftrain::StructuredParameterSnapshot& candidate, std::size_t version) override
  {
    return component_.publishBudgetedGpuParameters(candidate, version, device_, host_);
  }

  /// Resolve the component-owned successor before the shared transaction admits commit.
  std::unique_ptr<wftrain::PreparedParameterPublication> prepareParameterPublication(
      const wftrain::StructuredParameterSnapshot& candidate, std::size_t version) override
  {
    auto publication = component_.prepareBudgetedGpuParameterPublication(candidate, version, device_, host_);
    if (!publication)
      throw std::logic_error("Paired GPU component returned no prepared publication");
    return std::make_unique<PreparedPublication>(std::move(publication));
  }

private:
  /// Layering-only owner: component code has no dependency on the driver protocol.
  class PreparedPublication final : public wftrain::PreparedParameterPublication
  {
  public:
    explicit PreparedPublication(std::unique_ptr<PsiFormerComplexWF::PreparedGpuParameterPublication> publication)
        : publication_(std::move(publication))
    {}

    std::size_t commit() noexcept override { return publication_->commit(); }

  private:
    std::unique_ptr<PsiFormerComplexWF::PreparedGpuParameterPublication> publication_;
  };

  PsiFormerComplexWF& component_;
  const std::size_t device_, host_;
};
#endif

/** Replay capture must never publish a parameter response. */
std::size_t checkedMemoryAdd(std::size_t lhs, std::size_t rhs);
std::size_t checkedMemoryMultiply(std::size_t lhs, std::size_t rhs);

class ReplayOnlyECPSink final : public wftrain::ParameterReductionSink
{
  void consume(std::size_t, const wftrain::ParameterChunkConstView&) override
  {
    throw std::logic_error("WFTrainBatched: replay-only ECP capture emitted a derivative");
  }
};

#ifdef QMC_COMPLEX
/// Conservative pre-cutoff bound from the actual ionic species and built PP grids.
std::size_t pairedECPPointCapacity(const QMCHamiltonian& ham, const ParticleSet& electrons, std::size_t samples)
{
  std::size_t points = 0;
  for (int i = 0; i < ham.size(); ++i)
    if (const auto* nl = dynamic_cast<const NonLocalECPotential*>(ham.getComponent(i)))
    {
      const auto state = nl->captureQuadratureState();
      const auto& ions = nl->sourceParticles();
      for (int ion = 0; ion < ions.getTotalNum(); ++ion)
      {
        const auto species = static_cast<std::size_t>(ions.GroupID[ion]);
        if (species >= state.knots_per_species.size())
          throw std::invalid_argument("Paired ECP ionic species has no built quadrature metadata");
        points = checkedMemoryAdd(points, state.knots_per_species[species]);
      }
    }
  return checkedMemoryMultiply(samples, checkedMemoryMultiply(electrons.getTotalNum(), points));
}
#endif

/** Flush allocation intent before entering an owner/factory; no MPI here. */
void logTrainingAllocation(const char* owner, std::size_t samples, std::size_t parameters,
                           std::optional<std::size_t> planned_numeric_bytes = std::nullopt)
{
  app_log() << "WFTrain allocation before owner=" << owner << " scope=rank0_local"
            << " samples=" << samples << " parameters=" << parameters << " planned_numeric_bytes=";
  if (planned_numeric_bytes)
    app_log() << *planned_numeric_bytes;
  else
    app_log() << "unknown";
  app_log() << " (numeric_intent_not_RSS_or_peak)" << std::endl;
}

/** Copy live crowd metadata and bind it to a version-checked derivative owner. */
wftrain::PreparedEnergyGradientBatch prepareCrowdBatch(
    Crowd& crowd,
    std::string_view provider_id,
    std::size_t batch_ordinal,
    std::size_t sample_offset,
    std::size_t maximum_parameter_chunk_size,
    bool has_nonlocal_ecp,
    bool capture_ecp_jvp_replay,
    bool score_only,
    bool complex_training = false,
    const WFTrainDriverInput::ComplexGpuTraining* gpu_training = nullptr,
    wftrain::PairedFixedGridNativeEnergy* fixed_grid = nullptr,
    std::size_t scalar_ecp_budget = 0,
    bool capture_inline_ecp_vjp = false,
    ImmediateFullReplayReceipt* full_handoff = nullptr,
    std::unique_ptr<wftrain::StreamingDerivativeOperator> prepared_score = nullptr)
{
  const auto& twfs      = crowd.get_walker_twfs();
  const auto& electrons = crowd.get_walker_elecs();
  const auto& walkers   = crowd.get_walkers();
  RefVectorWithLeader<TrialWaveFunction> twf_batch(twfs[0], twfs);
  RefVectorWithLeader<ParticleSet> electron_batch(electrons[0], electrons);

  wftrain::PreparedEnergyGradientBatch batch;
  batch.weights.assign(walkers.size(), 1.0);
  batch.local_energy_term_mask =
      wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC);

#ifdef QMC_COMPLEX
  if (complex_training)
  {
    const auto& hamiltonians = crowd.get_walker_hamiltonians();
    RefVectorWithLeader<QMCHamiltonian> hamiltonian_batch(hamiltonians[0], hamiltonians);
    ResourceCollectionTeamLock<ParticleSet> particle_lock(crowd.getSharedResource().pset_res, electron_batch);
    ResourceCollectionTeamLock<TrialWaveFunction> wavefunction_lock(crowd.getSharedResource().twf_res, twf_batch);
    ResourceCollectionTeamLock<QMCHamiltonian> hamiltonian_lock(crowd.getSharedResource().ham_res, hamiltonian_batch);
    ParticleSet::mw_update(electron_batch);
    const bool reuse_full = full_handoff && gpu_training && score_only && has_nonlocal_ecp &&
        !capture_ecp_jvp_replay && !capture_inline_ecp_vjp &&
        consumeImmediateFullReplay(crowd, *full_handoff);
    if (!reuse_full)
      TrialWaveFunction::mw_evaluateLog(twf_batch, electron_batch);
    if (full_handoff)
      app_log() << "Paired immediate ECP FULL handoff: full_elided=" << reuse_full
                << " full_evaluations=" << !reuse_full << " batch=" << batch_ordinal << " offset=" << sample_offset
                << " host_capacity_bytes=" << full_handoff->retainedBytes()
                << "; physical accepted grid and Hamiltonian unchanged" << std::endl;
    // Resolve actual replay batch leaders/clones; the primary prototype cannot stand in for them.
    auto componentAt = [&](std::size_t sample) -> PsiFormerComplexWF& {
      const auto& components = twfs[sample].get().getOrbitals();
      auto* component = components.size() == 1 ? dynamic_cast<PsiFormerComplexWF*>(components[0].get()) : nullptr;
      if (!component || component->gpuInferenceEnabled() != (gpu_training != nullptr) ||
          component->parameterSchema().providerId() != provider_id)
        throw std::invalid_argument("WFTrainBatched: paired batch lost its sole component/provider/mode binding");
      return *component;
    };
    auto& leader = componentAt(0);
    RefVectorWithLeader<WaveFunctionComponent> component_batch(leader);
    if (gpu_training && has_nonlocal_ecp && !score_only && !capture_inline_ecp_vjp)
    {
      const auto samples = walkers.size();
      if (fixed_grid)
      {
        if (!fixed_grid->references.empty() || samples < 2)
          throw std::logic_error("Paired fixed-grid selection requires one stable multiwalker batch");
        fixed_grid->sources.reserve(samples);
        fixed_grid->references.reserve(samples);
        if (fixed_grid->sources.capacity() != samples || fixed_grid->references.capacity() != samples)
          throw std::length_error("Paired fixed-grid binding capacities exceed preallocation admission");
      }
      const auto points = pairedECPPointCapacity(hamiltonians[0], electrons[0], samples);
      const auto capacity = checkedMemoryAdd(samples, points); // Adam/VJP: B accepted plus Q virtual.
      const auto chunk = std::min(maximum_parameter_chunk_size, leader.parameterSchema().parameterCount());
      const auto required = leader.gpuDerivativeRequirements(samples, capacity, gpu_training->kinetic_targets, chunk);
      const auto replay_bytes = leader.gpuNonLocalReplayRequirements(samples, points);
      batch.derivative_operator = leader.makeBudgetedGpuDerivativeOperator(twf_batch, electron_batch,
          batch_ordinal, sample_offset, capacity, gpu_training->kinetic_targets, chunk, required.device_bytes,
          checkedMemoryAdd(required.host_bytes, replay_bytes.bound_host_extra_bytes), gpu_training->maximum_matrix_work);
      auto consumer = leader.makeBudgetedGpuNonLocalReplayConsumer(twf_batch, electron_batch,
          *batch.derivative_operator, replay_bytes.capture_host_bytes);
      const auto& schema = batch.derivative_operator->parameterSchema();
      std::vector<wftrain::DerivativeValue> ones(samples, {1.0, 0.0});
      const wftrain::CoefficientView coefficients{schema.providerId(), schema.fingerprint(),
          batch.derivative_operator->parameterVersion(), batch_ordinal, sample_offset, {ones.data(), ones.size()}};
      const auto ecp_mask = wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP);
      const wftrain::VJPCoefficientChannel channel{wftrain::EnergyGradientAccumulator::WEIGHTED_LOCAL_ENERGY_CHANNEL,
          wftrain::DerivativeProduct::LOCAL_ENERGY_VJP, coefficients, ecp_mask};
      ReplayOnlyECPSink sink;
      const auto native_cap = fixed_grid ? hamiltonians[0].get().nativeCaptureRequirements(samples).total_host_bytes : 0;
      const auto result = QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
          hamiltonian_batch, twf_batch, electron_batch, *consumer, {&channel, 1}, sink, native_cap);
      auto replay = consumer->releaseJVPReplay();
      if (replay->gridFingerprint() != result.grid_fingerprint)
        throw std::logic_error("Paired ECP accepted-grid identity changed before binding");
      const auto actual_points = replay->points().size();
      const auto replay_host_bytes = fixed_grid ? replay->storageDiagnostics().retained_numeric_bytes : 0;
      batch.derivative_operator->bindNonLocalECPJVPReplay(std::move(replay));
      batch.local_energy_term_mask |= ecp_mask;
      double maximum_imaginary_nonlocal = 0;
      for (std::size_t sample = 0; sample < samples; ++sample)
      {
        // Native REAL potential already contains Re(Vnl): restore ONLY its imaginary part.
        const auto observation = componentAt(sample).observeCapturedLocalEnergy(electrons[sample],
            *batch.derivative_operator, sample, hamiltonians[sample].get().getLocalPotential());
        batch.local_energies.push_back(observation.local_energy);
        maximum_imaginary_nonlocal = std::max(maximum_imaginary_nonlocal, std::abs(observation.nonlocal.imag()));
        if (fixed_grid)
        {
          const auto requirement = componentAt(sample).fixedGridEnergyReferenceRequirements();
          const auto cap = checkedMemoryAdd(checkedMemoryAdd(requirement.new_reference_host_bytes,
              requirement.shared_replay_descriptor_bytes), checkedMemoryAdd(replay_host_bytes, native_cap));
          fixed_grid->sources.emplace_back(componentAt(sample));
          fixed_grid->references.push_back(componentAt(sample).makeBudgetedFixedGridEnergyReference(
              electrons[sample], *batch.derivative_operator, sample, result.native_capture, cap));
        }
      }
      app_log() << std::setprecision(17) << "WFTrain paired accepted-grid ECP: Q=" << actual_points
                << " grid=" << result.grid_fingerprint << " version=" << batch.derivative_operator->parameterVersion()
                << " samples=" << samples << " max_abs_ImVnl=" << maximum_imaginary_nonlocal << std::endl;
      return batch;
    }
    if (has_nonlocal_ecp)
    {
      // Score capture and fixed-weight full response share the same native scalar-energy checks.
      for (std::size_t sample = 0; sample < walkers.size(); ++sample)
        component_batch.push_back(componentAt(sample));
      if (gpu_training)
      {
        if (!capture_inline_ecp_vjp) batch.local_energy_term_mask = 0;
        const auto scope = capture_inline_ecp_vjp ? PsiFormerComplexWF::GpuDerivativeScope::COMPLETE :
            PsiFormerComplexWF::GpuDerivativeScope::SCORE_ECP_ONLY;
        const auto chunk = std::min(maximum_parameter_chunk_size, leader.parameterSchema().parameterCount());
        const auto required = leader.gpuDerivativeRequirements(
            walkers.size(), walkers.size(), gpu_training->kinetic_targets, chunk, scope);
        batch.derivative_operator = leader.makeBudgetedGpuDerivativeOperator(twf_batch, electron_batch,
            batch_ordinal, sample_offset, walkers.size(), gpu_training->kinetic_targets, chunk,
            required.device_bytes, required.host_bytes, gpu_training->maximum_matrix_work, scope);
      }
      else
      {
        if (score_only) batch.local_energy_term_mask = 0;
        const auto cpu_scope = score_only ? PsiFormerComplexWF::CpuDerivativeScope::SCORE_ONLY
                                         : PsiFormerComplexWF::CpuDerivativeScope::COMPLETE;
        const auto required = leader.streamingDerivativeRequiredBytes(walkers.size(), cpu_scope);
        batch.derivative_operator = leader.makeBudgetedStreamingDerivativeOperator(
            component_batch, electron_batch, batch_ordinal, sample_offset, maximum_parameter_chunk_size, required, cpu_scope);
      }
      const auto& schema = batch.derivative_operator->parameterSchema();
      const auto version = batch.derivative_operator->parameterVersion();
      const auto energy_bytes = capture_inline_ecp_vjp ? leader.nonLocalVJPCaptureRequiredBytes(walkers.size()) :
          wftrain::NonLocalEnergyAccumulator::requiredBytes(walkers.size());
      const auto ecp_mask = wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP);
      std::unique_ptr<wftrain::PreparedLocalEnergyVJPCapture> response;
      std::unique_ptr<wftrain::NonLocalEnergyAccumulator> energy;
      std::unique_ptr<wftrain::NonLocalECPDerivativeConsumer> inline_consumer;
      if (capture_inline_ecp_vjp)
      {
        if (score_only || capture_ecp_jvp_replay || fixed_grid)
          throw std::invalid_argument("Inline ECP response requires fixed first-order full-response weights without replay");
        response = std::make_unique<wftrain::PreparedLocalEnergyVJPCapture>(schema, version, ecp_mask);
        inline_consumer = leader.makeBudgetedNonLocalVJPConsumer(twf_batch, electron_batch,
            *batch.derivative_operator, energy_bytes);
      }
      else
        energy = std::make_unique<wftrain::NonLocalEnergyAccumulator>(schema, version, batch_ordinal, sample_offset,
            walkers.size(), electrons[0].get().getTotalNum(), energy_bytes);
      std::vector<wftrain::DerivativeValue> ones(walkers.size(), {1, 0});
      const wftrain::VJPCoefficientChannel channel{wftrain::EnergyGradientAccumulator::WEIGHTED_LOCAL_ENERGY_CHANNEL,
          wftrain::DerivativeProduct::LOCAL_ENERGY_VJP,
          {schema.providerId(), schema.fingerprint(), version, batch_ordinal, sample_offset,
              {ones.data(), ones.size()}}, wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP)};
      ReplayOnlyECPSink scalar_sink;
      wftrain::ParameterReductionSink& sink = response ? static_cast<wftrain::ParameterReductionSink&>(*response) : scalar_sink;
      auto& consumer = inline_consumer ? *inline_consumer : static_cast<wftrain::NonLocalECPDerivativeConsumer&>(*energy);
      const auto result = QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
          hamiltonian_batch, twf_batch, electron_batch, consumer, {&channel, 1}, sink);
      if (energy && energy->gridFingerprint() != result.grid_fingerprint)
        throw std::logic_error("Scalar nonlocal energy accepted-grid identity changed");
      const auto captured_energies = inline_consumer ? leader.capturedNonLocalEnergies(*inline_consumer) : energy->energies();
      double maximum_imaginary_nonlocal = 0;
      batch.local_energies.reserve(walkers.size());
      for (std::size_t sample = 0; sample < walkers.size(); ++sample)
      {
        auto& hamiltonian = hamiltonians[sample].get();
        const auto captured = captured_energies[sample];
        double reported_nonlocal = 0;
        for (int term = 0; term < hamiltonian.size(); ++term)
          if (const auto* nonlocal = dynamic_cast<const NonLocalECPotential*>(hamiltonian.getComponent(term)))
            reported_nonlocal += nonlocal->getValue();
        const auto tolerance = 512 * std::numeric_limits<double>::epsilon() *
            (1 + std::abs(reported_nonlocal) + std::abs(captured.real()));
        if (!wftrain::isFiniteTrainingReal(reported_nonlocal) ||
            std::abs(reported_nonlocal - captured.real()) > tolerance)
          throw std::logic_error("Scalar complex nonlocal energy differs from the native accepted real potential");
        // Native V already contains Re(Vnl). Restore only Im(Vnl), together with the actual complex kinetic value.
        const auto observed = componentAt(sample).observeCoulombEnergy(
            electrons[sample], hamiltonian.getLocalPotential());
        const auto complete = observed.local_energy + wftrain::DerivativeValue{0, captured.imag()};
        if (!wftrain::isFiniteTrainingReal(complete.real()) || !wftrain::isFiniteTrainingReal(complete.imag()))
          throw std::overflow_error("Scalar complete complex nonlocal energy is nonfinite");
        batch.local_energies.push_back(complete);
        maximum_imaginary_nonlocal = std::max(maximum_imaginary_nonlocal, std::abs(captured.imag()));
      }
      if (response) batch.additive_local_energy_vjp.emplace(response->release());
      app_log() << std::setprecision(17);
      if (capture_inline_ecp_vjp) app_log() << "WFTrain paired inline full-response ECP:";
      else app_log() << "WFTrain " << (gpu_training ? "GPU score-only" : "CPU") << " accepted-grid ECP:";
      app_log() << " grid=" << result.grid_fingerprint
                << " version=" << version << " samples=" << walkers.size()
                << " max_abs_ImVnl=" << maximum_imaginary_nonlocal
                << " capture_numeric_bytes=" << energy_bytes
                << " ECP_derivative_response_mask=" << (capture_inline_ecp_vjp ? ecp_mask : 0)
                << " retained_grid_bytes=0" << std::endl;
      return batch;
    }
    batch.local_energies.reserve(walkers.size());
    for (std::size_t sample = 0; sample < walkers.size(); ++sample)
    {
      auto& component = componentAt(sample);
      component_batch.push_back(component);
      auto& hamiltonian = hamiltonians[sample].get();
      auto& p = electrons[sample].get();
      hamiltonian.evaluate(twfs[sample].get(), p);
      // The actual admitted same-sample Coulomb H supplies V; REAL walker properties are reporting only.
      batch.local_energies.push_back(component.observeCoulombEnergy(p, hamiltonian.getLocalPotential()).local_energy);
    }
    if (gpu_training)
    {
      const auto chunk = std::min(maximum_parameter_chunk_size, leader.parameterSchema().parameterCount());
      const auto required = leader.gpuDerivativeRequirements(
          walkers.size(), walkers.size(), gpu_training->kinetic_targets, chunk);
      batch.derivative_operator = leader.makeBudgetedGpuDerivativeOperator(twf_batch, electron_batch,
          batch_ordinal, sample_offset, walkers.size(), gpu_training->kinetic_targets, chunk,
          required.device_bytes, required.host_bytes, gpu_training->maximum_matrix_work);
    }
    else
    {
      if (score_only) batch.local_energy_term_mask = 0;
      const auto cpu_scope = score_only ? PsiFormerComplexWF::CpuDerivativeScope::SCORE_ONLY
                                       : PsiFormerComplexWF::CpuDerivativeScope::COMPLETE;
      const auto required = leader.streamingDerivativeRequiredBytes(walkers.size(), cpu_scope);
      batch.derivative_operator = leader.makeBudgetedStreamingDerivativeOperator(
          component_batch, electron_batch, batch_ordinal, sample_offset, maximum_parameter_chunk_size, required, cpu_scope);
    }
    return batch;
  }
#endif

  // Score-covariance updates differentiate log(parent), not the local energy or the ECP grid.
  // Sample collection already evaluated the actual complete Hamiltonian at these saved positions.
  const auto& parent_components = twfs[0].get().getOrbitals();
  const auto* frozen_sum = parent_components.size() == 1
      ? dynamic_cast<const FrozenReferenceResidualWF*>(parent_components[0].get()) : nullptr;
  if (frozen_sum && score_only)
  {
    if (capture_ecp_jvp_replay)
      throw std::invalid_argument("Frozen reference sum does not expose local-energy parameter response");
    batch.local_energy_term_mask = 0;
    batch.derivative_operator = TrialWaveFunction::makeStructuredStreamingDerivativeOperator(
        twf_batch, electron_batch, provider_id, batch_ordinal, sample_offset, maximum_parameter_chunk_size);
    batch.local_energies.reserve(walkers.size());
    for (const auto& walker : walkers)
      batch.local_energies.emplace_back(walker.get().Properties(WalkerProperties::LOCALENERGY));
    return batch;
  }

  if (!has_nonlocal_ecp)
  {
    batch.derivative_operator =
        TrialWaveFunction::makeStructuredStreamingDerivativeOperator(
            twf_batch, electron_batch, provider_id, batch_ordinal,
            sample_offset, maximum_parameter_chunk_size);
    batch.local_energies.reserve(walkers.size());
    for (const auto& walker : walkers)
      batch.local_energies.emplace_back(
          walker.get().Properties(WalkerProperties::LOCALENERGY));
    return batch;
  }

  // The ordinary-locality ECP energy and its weighted parameter response must
  // share one accepted stochastic quadrature grid.  Re-evaluate the complete
  // local energy here and capture the isolated ECP VJP during that traversal.
  const auto& hamiltonians = crowd.get_walker_hamiltonians();
  RefVectorWithLeader<QMCHamiltonian> hamiltonian_batch(
      hamiltonians[0], hamiltonians);

  ResourceCollectionTeamLock<ParticleSet> particle_lock(
      crowd.getSharedResource().pset_res, electron_batch);
  ResourceCollectionTeamLock<TrialWaveFunction> wavefunction_lock(
      crowd.getSharedResource().twf_res, twf_batch);
  ResourceCollectionTeamLock<QMCHamiltonian> hamiltonian_lock(
      crowd.getSharedResource().ham_res, hamiltonian_batch);
  ParticleSet::mw_update(electron_batch);
  bool reuse_immediate_full = false;
#ifndef QMC_COMPLEX
  if (full_handoff && score_only && !capture_ecp_jvp_replay)
  {
    const auto& components = twfs[0].get().getOrbitals();
    const auto* real = components.size() == 1 ? dynamic_cast<const PsiFormerWF*>(components[0].get()) : nullptr;
    reuse_immediate_full = real && real->usesAcceleratorParameterProducts() &&
        real->isPseudopotentialSystem() && consumeImmediateFullReplay(crowd, *full_handoff);
  }
#endif
  if (!reuse_immediate_full)
  {
    // An old-version or stale-context owner must die BEFORE canonical refresh/recreation.
    prepared_score.reset();
    TrialWaveFunction::mw_evaluateLog(twf_batch, electron_batch);
  }
  if (full_handoff)
    app_log() << "Immediate ECP FULL handoff: full_elided=" << reuse_immediate_full
              << " full_evaluations=" << !reuse_immediate_full
              << " batch=" << batch_ordinal << " offset=" << sample_offset
              << " host_capacity_bytes=" << full_handoff->retainedBytes()
              << "; accepted ECP grid traversal and RNG unchanged" << std::endl;

  // Construct both derivative owners only after canonical value/VGL refresh so
  // the kinetic tape and accepted-grid ECP response see identical reference
  // coordinates, gradients, and parameter version.
#ifndef QMC_COMPLEX
  const auto* real = dynamic_cast<const PsiFormerWF*>(twfs[0].get().getOrbitals()[0].get());
  if (score_only && real && real->usesAcceleratorParameterProducts())
  {
    if (capture_ecp_jvp_replay || !real->isPseudopotentialSystem())
      throw std::invalid_argument("GPU score-only ECP requires physical PP batches without derivative replay");
    if (!provider_id.empty() && provider_id != real->parameterSchema().providerId())
      throw std::invalid_argument("GPU score-only ECP provider identity differs from the selected provider");
    // Score covariance needs physical EL, but no local-energy parameter response.
    // Select its immutable profile before any complete kinetic owner is prepared.
    batch.derivative_operator = prepared_score ? std::move(prepared_score) :
        makeRealGpuScoreBatchUnderLocks(crowd, batch_ordinal, sample_offset, maximum_parameter_chunk_size);
    // This mask describes derivative coverage, not the complete physical EL below.
    batch.local_energy_term_mask = 0;
    const auto samples = walkers.size();
    const auto scalar_bytes = wftrain::NonLocalEnergyAccumulator::requiredBytes(samples);
    const auto named = scoreOnlyECPCaptureNamedBytes(*batch.derivative_operator, samples);
    if (!scalar_ecp_budget || scalar_ecp_budget == std::numeric_limits<std::size_t>::max() || named > scalar_ecp_budget)
      throw std::length_error("GPU score-only ECP numeric owners exceed remaining maximum_retained_bytes");
    logTrainingAllocation("score_ecp_scalar_capture", samples,
        batch.derivative_operator->parameterSchema().parameterCount(), named);
    const auto& schema = batch.derivative_operator->parameterSchema();
    const auto version = batch.derivative_operator->parameterVersion();
    wftrain::NonLocalEnergyAccumulator energy(schema, version, batch_ordinal, sample_offset,
        samples, electrons[0].get().getTotalNum(), scalar_bytes);
    std::vector<wftrain::DerivativeValue> ones(samples, {1, 0});
    const wftrain::VJPCoefficientChannel channel{wftrain::EnergyGradientAccumulator::WEIGHTED_LOCAL_ENERGY_CHANNEL,
        wftrain::DerivativeProduct::LOCAL_ENERGY_VJP,
        {schema.providerId(), schema.fingerprint(), version, batch_ordinal, sample_offset,
            {ones.data(), ones.size()}}, wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP)};
    ReplayOnlyECPSink sink;
    const auto result = QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
        hamiltonian_batch, twf_batch, electron_batch, energy, {&channel, 1}, sink);
    if (energy.gridFingerprint() != result.grid_fingerprint || result.local_energies.size() != samples)
      throw std::logic_error("GPU score-only ECP accepted energy/grid identity mismatch");
    batch.local_energies.reserve(samples);
    for (std::size_t sample = 0; sample < samples; ++sample)
    {
      double reported = 0;
      const auto& h = hamiltonians[sample].get();
      for (int term = 0; term < h.size(); ++term)
        if (const auto* nonlocal = dynamic_cast<const NonLocalECPotential*>(h.getComponent(term)))
          reported += nonlocal->getValue();
      const auto captured = energy.energies()[sample];
      const auto tolerance = 512 * std::numeric_limits<double>::epsilon() * (1 + std::abs(reported) + std::abs(captured));
      if (!wftrain::isFiniteTrainingReal(reported) || std::abs(captured.imag()) > tolerance ||
          std::abs(reported - captured.real()) > tolerance ||
          !wftrain::isFiniteTrainingReal(result.local_energies[sample]))
        throw std::logic_error("GPU score-only ECP differs from the actual native nonlocal energy");
      batch.local_energies.emplace_back(result.local_energies[sample], 0);
    }
    if (real->residentECPObservationEnabled())
    {
      RefVectorWithLeader<WaveFunctionComponent> components(*twfs[0].get().getOrbitals()[0]);
      for (const auto& wf : twfs) components.push_back(*wf.get().getOrbitals()[0]);
      const auto physical = wftrain::realECPPhysicalIdentity(const_cast<PsiFormerWF&>(*real),
          electrons[0].get(), hamiltonians[0].get());
      std::vector<double> kinetic(samples);
      for (std::size_t sample = 0; sample < samples; ++sample)
        kinetic[sample] = hamiltonians[sample].get().getKineticEnergy();
      real->bindResidentECPObservation(components, *batch.derivative_operator, result.grid_fingerprint,
          physical.fingerprint, result.local_energies, kinetic);
      if (scoreOnlyECPCaptureNamedBytes(*batch.derivative_operator, samples) > scalar_ecp_budget)
        throw std::length_error("GPU ECP physical observation plus score owners exceed retained numeric budget");
    }
    // Physical EL includes ECP; this owner intentionally advertises no ECP dEL.
    app_log() << "WFTrain GPU score-only accepted-grid ECP: grid=" << result.grid_fingerprint
              << " version=" << version << " samples=" << samples
              << " parameter_response_mask=0 retained_grid_bytes=0; named budget excludes native tile/vendor storage"
              << std::endl;
    return batch;
  }
#endif
  batch.derivative_operator =
      TrialWaveFunction::makeStructuredStreamingDerivativeOperator(
          twf_batch, electron_batch, provider_id, batch_ordinal, sample_offset,
          maximum_parameter_chunk_size);
  bool bounded_replay = false;
  for (const auto& component : twfs[0].get().getOrbitals())
    if (const auto* psiformer = dynamic_cast<const PsiFormerWF*>(component.get()))
      if (psiformer->parameterSchema().providerId() == provider_id)
      {
        bounded_replay = !psiformer->usesAcceleratorParameterProducts();
#ifndef QMC_COMPLEX
        // Real GPU replay contracts accepted virtual/reference scores in shared
        // DEVICE scratch; complex-build aggregate compatibility is unchanged.
        bounded_replay = true;
#endif
      }
  logTrainingAllocation("accepted_ecp_grid_consumer", walkers.size(),
      batch.derivative_operator->parameterSchema().parameterCount());
  app_log() << "WFTrain ECP capture batch=" << batch_ordinal << " sample_offset=" << sample_offset
            << " replay_only=" << bounded_replay
            << " planned_parameter_workspace_bytes=" << (bounded_replay ? "0" : "provider_defined")
            << " planned_grid_bytes=unknown_until_accepted_traversal" << std::endl;
  auto ecp_consumer =
      TrialWaveFunction::makeStructuredNonLocalECPDerivativeConsumer(
          twf_batch, electron_batch, provider_id,
          maximum_parameter_chunk_size, capture_ecp_jvp_replay || bounded_replay,
          bounded_replay);

  const wftrain::StructuredParameterSchema& schema =
      batch.derivative_operator->parameterSchema();
  const std::size_t parameter_version =
      batch.derivative_operator->parameterVersion();
  std::vector<wftrain::DerivativeValue> ecp_coefficients(
      walkers.size(), wftrain::DerivativeValue{1.0, 0.0});
  const wftrain::CoefficientView coefficient_view{
      schema.providerId(), schema.fingerprint(), parameter_version,
      batch_ordinal, sample_offset,
      {ecp_coefficients.data(), ecp_coefficients.size()}};
  const std::uint32_t ecp_mask =
      wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP);
  const wftrain::VJPCoefficientChannel ecp_channel{
      wftrain::EnergyGradientAccumulator::WEIGHTED_LOCAL_ENERGY_CHANNEL,
      wftrain::DerivativeProduct::LOCAL_ENERGY_VJP, coefficient_view, ecp_mask};
  // Bounded preparation retains accepted grids, never one P-vector per snapshot.
  // Arbitrary weighted/clipped VJPs and JVPs later reuse the streaming owner's
  // shared scratch on this same grid. Complex-build GPU compatibility is unchanged.
  std::unique_ptr<wftrain::PreparedLocalEnergyVJPCapture> ecp_capture;
  ReplayOnlyECPSink replay_sink;
  if (!bounded_replay)
    ecp_capture = std::make_unique<wftrain::PreparedLocalEnergyVJPCapture>(
        schema, parameter_version, ecp_mask);
  wftrain::ParameterReductionSink& ecp_sink = bounded_replay
      ? static_cast<wftrain::ParameterReductionSink&>(replay_sink) : *ecp_capture;

  const auto ecp_result =
      QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
          hamiltonian_batch, twf_batch, electron_batch, *ecp_consumer,
          {&ecp_channel, 1}, ecp_sink);
  batch.local_energies.reserve(ecp_result.local_energies.size());
  for (const auto energy : ecp_result.local_energies)
    batch.local_energies.emplace_back(energy, 0.0);
  if (ecp_capture)
    batch.additive_local_energy_vjp.emplace(ecp_capture->release());

  if (capture_ecp_jvp_replay || bounded_replay)
  {
    auto replay = ecp_consumer->releaseJVPReplay();
    if (replay->gridFingerprint() != ecp_result.grid_fingerprint)
      throw std::logic_error(
          "WFTrainBatched: accepted ECP replay grid identity changed before binding");
    batch.derivative_operator->bindNonLocalECPJVPReplay(std::move(replay));
    if (bounded_replay)
      batch.local_energy_term_mask |= ecp_mask;
  }
  return batch;
}

/** Capture conventional energy and accepted ECP grid in one existing traversal.
 * Retained collection energies belong to older grids and must be replaced along
 * with the replay. Derivative products keep one shared private iteration worker.
 */
wftrain::PreparedEnergyGradientBatch prepareConventionalECPBatch(
    Crowd& crowd, RetainedCrowdSamples& samples, wftrain::TrainingWaveFunctionBinding& binding,
    const std::shared_ptr<ConventionalReplayWorker>& worker,
    const std::shared_ptr<const wftrain::StructuredParameterSnapshot>& parameters,
    std::size_t batch_ordinal, std::size_t sample_offset)
{
  replayCrowdSamples(crowd, samples);
  const auto& twfs = crowd.get_walker_twfs();
  const auto& electrons = crowd.get_walker_elecs();
  const auto& hamiltonians = crowd.get_walker_hamiltonians();
  RefVectorWithLeader<TrialWaveFunction> twf_batch(twfs[0], twfs);
  RefVectorWithLeader<ParticleSet> particle_batch(electrons[0], electrons);
  RefVectorWithLeader<QMCHamiltonian> hamiltonian_batch(hamiltonians[0], hamiltonians);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(crowd.getSharedResource().pset_res, particle_batch);
  ResourceCollectionTeamLock<TrialWaveFunction> wavefunction_lock(crowd.getSharedResource().twf_res, twf_batch);
  ResourceCollectionTeamLock<QMCHamiltonian> hamiltonian_lock(crowd.getSharedResource().ham_res, hamiltonian_batch);
  ParticleSet::mw_update(particle_batch);
  TrialWaveFunction::mw_evaluateLog(twf_batch, particle_batch);

  auto replay_samples = std::make_shared<ConventionalReplay>(worker, samples);
  wftrain::PreparedEnergyGradientBatch batch;
  batch.weights.assign(samples.particles.size(), 1.0);
  batch.derivative_operator = binding.adapter().makeDerivativeOperator(
      replay_samples, parameters, batch_ordinal, sample_offset);
  logTrainingAllocation("conventional_accepted_ecp_grid", samples.particles.size(),
                        binding.parameterSchema().parameterCount());
  auto consumer = binding.adapter().makeNonLocalECPDerivativeConsumer(
      replay_samples, parameters, batch_ordinal, sample_offset);
  std::vector<wftrain::DerivativeValue> coefficients(samples.particles.size(), {1.0, 0.0});
  const wftrain::CoefficientView coefficient_view{
      binding.parameterSchema().providerId(), binding.parameterSchema().fingerprint(), parameters->version,
      batch_ordinal, sample_offset, {coefficients.data(), coefficients.size()}};
  const auto ecp_mask = wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP);
  const wftrain::VJPCoefficientChannel channel{
      wftrain::EnergyGradientAccumulator::WEIGHTED_LOCAL_ENERGY_CHANNEL,
      wftrain::DerivativeProduct::LOCAL_ENERGY_VJP, coefficient_view, ecp_mask};
  ReplayOnlyECPSink sink;
  const auto result = QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
      hamiltonian_batch, twf_batch, particle_batch, *consumer, {&channel, 1}, sink);
  auto replay = consumer->releaseJVPReplay();
  if (replay->gridFingerprint() != result.grid_fingerprint ||
      result.local_energies.size() != samples.particles.size())
    throw std::logic_error("Conventional ECP accepted energy/grid identity mismatch");
  batch.derivative_operator->bindNonLocalECPJVPReplay(std::move(replay));
  batch.local_energy_term_mask = wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC) | ecp_mask;
  batch.local_energies.reserve(result.local_energies.size());
  for (double energy : result.local_energies)
    batch.local_energies.emplace_back(energy, 0.0);
  samples.local_energies.assign(result.local_energies.begin(), result.local_energies.end());
  for (std::size_t sample = 0; sample < crowd.size(); ++sample)
  {
    electrons[sample].get().saveWalker(crowd.get_walkers()[sample]);
    crowd.get_walkers()[sample].get().resetProperty(
        twfs[sample].get().getLogPsi(), twfs[sample].get().getPhase(), result.local_energies[sample]);
  }
  return batch;
}

/** Validate and return the expected complete Hamiltonian-term mask. */
std::uint32_t completeLocalEnergyTermMask(
    const std::vector<wftrain::PreparedEnergyGradientBatch>& batches,
    bool has_nonlocal_ecp)
{
  std::uint32_t expected_mask =
      wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC);
  if (has_nonlocal_ecp)
    expected_mask |=
        wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP);
  for (const auto& batch : batches)
  {
    std::uint32_t mask = batch.local_energy_term_mask;
    if (batch.additive_local_energy_vjp)
      mask |= batch.additive_local_energy_vjp->local_energy_term_mask;
    if (mask != expected_mask)
      throw std::invalid_argument(
          "WFTrainBatched: prepared batches have inconsistent Hamiltonian-term coverage");
  }
  return expected_mask;
}

/// Scalar-only diagnostics are distinct from a host parameter-gradient objective.
struct ResidentEnergyDiagnostics
{
  std::size_t samples = 0;
  double mean = 0, variance = 0, gradient_norm = 0;
};

/// Check the actual selected operators both before sampling and before a resident update.
void validateResidentCoulomb(QMCHamiltonian& hamiltonian, bool device_ecp = false)
{
  if (hamiltonian.size() < 2)
    throw UniformCommunicateError("Resident ENERGY requires an actual Coulomb Hamiltonian");
  std::size_t kinetic_count = 0, nonlocal_count = 0, local_ecp_count = 0;
  for (int term = 0; term < hamiltonian.size(); ++term)
  {
    const auto name = hamiltonian.getComponent(term)->getClassName();
    if (name == "BareKineticEnergy")
      ++kinetic_count;
    else if (device_ecp && name == "LocalECPotential")
      ++local_ecp_count;
    else if (device_ecp && dynamic_cast<NonLocalECPotential*>(hamiltonian.getComponent(term)))
    {
      const auto& potential = *static_cast<NonLocalECPotential*>(hamiltonian.getComponent(term));
      if (potential.usesDLA() || !potential.usesVirtualParticles())
        throw UniformCommunicateError("Resident DEVICE ECP requires ordinary locality and virtual particles");
      ++nonlocal_count;
    }
    else if (name != "CoulombPotential")
      throw UniformCommunicateError("Resident ENERGY supports only the actual selected open Coulomb Hamiltonian");
  }
  if (kinetic_count != 1)
    throw UniformCommunicateError("Resident ENERGY requires exactly one physical kinetic operator");
  if (nonlocal_count > 1 || local_ecp_count > 1 ||
      (local_ecp_count && nonlocal_count != 1) ||
      (device_ecp && hamiltonian.hasPhysicalNLPP() != (nonlocal_count == 1)))
    throw UniformCommunicateError("Resident DEVICE ECP requires one consistently registered ordinary nonlocal operator");
}

/// Identify the exact molecular operator/source identities, not merely Coulomb class names.
std::array<int, 4> residentOpenScalarSlots(QMCHamiltonian& h, ParticleSet& electrons,
    PsiFormerWF& provider, const ParticleSet*& selected_ions)
{
  if (h.size() != 4 || h.sizeOfObservables() != 4 || h.sizeOfCollectables() != 0 ||
      electrons.getTotalNum() != 4 || electrons.groups() != 2 ||
      electrons.groupsize(0) != 2 || electrons.groupsize(1) != 2 ||
      electrons.getLattice().SuperCellEnum != SUPERCELL_OPEN)
    throw UniformCommunicateError("Resident open observation requires the actual Ne4(2/2) four-term molecular Hamiltonian");
  for (int e = 0; e < electrons.getTotalNum(); ++e)
    if (!wftrain::isFiniteTrainingReal(electrons.Z[e]) || electrons.Z[e] != -1.0)
      throw UniformCommunicateError("Resident molecular Coulomb requires actual unit negative electron charges");
  std::array<int, 4> slots{{-1, -1, -1, -1}};
  const ParticleSet* ions = nullptr;
  const ParticleSet* ion_pair = nullptr;
  for (int term = 0; term < h.size(); ++term)
  {
    auto* op = h.getComponent(term);
    int kind = -1;
    if (const auto* kinetic = dynamic_cast<const BareKineticEnergy*>(op))
    {
      if (&kinetic->getTargetPSet() != &electrons)
        throw UniformCommunicateError("Resident kinetic target differs from the sampled electrons");
      kind = 0;
    }
    else if (const auto* coulomb = dynamic_cast<const CoulombPotential*>(op))
    {
      if (coulomb->computesForces())
        throw UniformCommunicateError("Resident molecular observation excludes force consumers");
      const auto* source = &coulomb->getSourcePSet();
      const auto* target = &coulomb->getTargetPSet();
      if (coulomb->isSameParticleSet() && source == &electrons && target == &electrons)
      {
        if (!coulomb->evaluatesCurrentTarget())
          throw UniformCommunicateError("Resident electron Coulomb must evaluate the current target");
        kind = 1;
      }
      else if (!coulomb->isSameParticleSet() && target == &electrons && source != &electrons)
      {
        if (!coulomb->evaluatesCurrentTarget())
          throw UniformCommunicateError("Resident electron/ion Coulomb must evaluate the current target");
        kind = 2; ions = source;
      }
      else if (coulomb->isSameParticleSet() && source == target && source != &electrons)
      { kind = 3; ion_pair = source; }
    }
    const auto column = h.getObservable(op->getName());
    if (kind < 0 || slots[kind] != -1 || column < 0)
      throw UniformCommunicateError("Resident molecular observation has an unsupported, duplicate or unmapped operator");
    slots[kind] = column;
  }
  if (!ions || ions != ion_pair || ions->getTotalNum() != 2 ||
      ions->getLattice().SuperCellEnum != SUPERCELL_OPEN)
    throw UniformCommunicateError("Resident molecular electron/ion and ion/ion operators must share the actual two-ion source");
  // Reuse the existing exact ordered positions/charges/lattice check against the immutable model.
  provider.validateSystem(electrons, *ions, "all_electron");
  selected_ions = ions;
  return slots;
}

/// Compare the same undamped host/resident diagnostics without changing the CSV schema.
void logSRDiagnostics(const char* route, const wftrain::KrylovSolveDiagnostics& solve,
    double direction_norm, double metric_norm, double damping, double scale)
{
  const auto precision = app_log().precision();
  app_log() << std::setprecision(17) << "SR diagnostics route=" << route << " direction_norm=" << direction_norm
            << " metric_norm=" << metric_norm << " damping=" << damping << " applied_scale=" << scale
            << " residual_norm=" << solve.final_residual_norm << " relative_residual=" << solve.relative_residual_norm
            << " stop=" << wftrain::krylovStopReasonName(solve.stop_reason) << std::endl;
  app_log().precision(precision);
}

/** Return the Euclidean norm of a real parameter gradient. */
double gradientNorm(const std::vector<wftrain::DerivativeReal>& gradient)
{
  double squared_norm = 0.0;
  for (const double value : gradient)
    squared_norm += value * value;
  return std::sqrt(squared_norm);
}

/** Translate the driver input into the common bounded Krylov contract. */
wftrain::KrylovSolverControl makeKrylovControl(const WFTrainDriverInput& input)
{
  wftrain::KrylovSolverControl control;
  control.relative_tolerance             = input.getKrylovRelativeTolerance();
  control.absolute_tolerance             = input.getKrylovAbsoluteTolerance();
  control.maximum_iterations             = input.getKrylovMaximumIterations();
  control.maximum_operator_applications  = input.getKrylovMaximumOperatorApplications();
  control.maximum_seconds                = input.getKrylovMaximumSeconds();
  control.stagnation_window              = input.getKrylovStagnationWindow();
  control.minimum_relative_improvement   = input.getKrylovMinimumRelativeImprovement();
  control.curvature_tolerance            = input.getKrylovCurvatureTolerance();
  control.imaginary_tolerance            = input.getKrylovImaginaryTolerance();
  return control;
}

/** Translate typed input into the parameter-space SR update controls. */
wftrain::StochasticReconfigurationUpdateControl makeSRControl(const WFTrainDriverInput& input)
{
  wftrain::StochasticReconfigurationUpdateControl control;
  control.learning_rate            = input.getLearningRate();
  control.initial_damping          = input.getDamping();
  control.damping_multiplier       = input.getDampingMultiplier();
  control.minimum_retry_damping    = input.getMinimumRetryDamping();
  control.maximum_damping_attempts = input.getMaximumDampingAttempts();
  control.maximum_update_norm      = input.getMaximumUpdateNorm();
  control.maximum_metric_norm      = input.getMaximumMetricNorm();
  control.krylov                   = makeKrylovControl(input);
  return control;
}

/** Translate typed input into the shared sample-space minimum-step controls. */
wftrain::MinimumStepUpdateControl makeMinimumStepControl(const WFTrainDriverInput& input)
{
  wftrain::MinimumStepUpdateControl control;
  control.learning_rate       = input.getLearningRate();
  control.damping             = input.getDamping();
  control.center_shift        = input.getCenterShift();
  control.maximum_update_norm = input.getMaximumUpdateNorm();
  control.objective_tolerance = input.getObjectiveTolerance();
  control.krylov              = makeKrylovControl(input);
  return control;
}

/** Translate typed input into the persistent SPRING recurrence controls. */
wftrain::SPRINGUpdateControl makeSPRINGControl(const WFTrainDriverInput& input)
{
  wftrain::SPRINGUpdateControl control;
  control.minimum_step               = makeMinimumStepControl(input);
  control.history_decay              = input.getSpringHistoryDecay();
  control.maximum_update_norm_squared = input.getSpringMaximumUpdateNormSquared();
  return control;
}

/** Expose nonowning parameter-space SR batches while the prepared owner is live. */
std::vector<wftrain::StochasticReconfigurationBatch> makeSRBatches(
    const std::vector<wftrain::PreparedEnergyGradientBatch>& batches)
{
  std::vector<wftrain::StochasticReconfigurationBatch> result;
  result.reserve(batches.size());
  for (const auto& batch : batches)
    result.push_back({batch.derivative_operator.get(),
                      {batch.weights.data(), batch.weights.size()}});
  return result;
}

/** Expose nonowning sample-space batches while the prepared owners remain live. */
std::vector<wftrain::ScoreCovarianceBatch> makeScoreBatches(
    const std::vector<wftrain::PreparedEnergyGradientBatch>& batches)
{
  std::vector<wftrain::ScoreCovarianceBatch> result;
  result.reserve(batches.size());
  for (const auto& batch : batches)
    result.push_back({batch.derivative_operator.get(),
                      {batch.weights.data(), batch.weights.size()},
                      {batch.local_energies.data(), batch.local_energies.size()}});
  return result;
}

std::size_t checkedMemoryAdd(std::size_t lhs, std::size_t rhs);

/** Admit actual numeric storage plus explicit unmaterialized fallback reservations. */
wftrain::PreparedEnergyGradientStorageDiagnostics preparedStorageDiagnostics(
    const std::vector<wftrain::PreparedEnergyGradientBatch>& batches)
{
  wftrain::PreparedEnergyGradientStorageDiagnostics result;
  std::unordered_map<std::uintptr_t, std::size_t> shared_workspace_max;
  result.batch_count = batches.size();
  for (const auto& batch : batches)
  {
    if (!batch.derivative_operator)
      throw std::invalid_argument(
          "WFTrainBatched: prepared batch has no derivative operator");
    result.sample_count += batch.weights.size();
    const auto storage = batch.derivative_operator->storageDiagnostics();
    result.derivative_owner_bytes = checkedMemoryAdd(result.derivative_owner_bytes,
        checkedMemoryAdd(storage.retained_numeric_bytes, storage.shared_workspace_reserved_bytes));
    if (storage.shared_workspace_identity)
    {
      auto& previous_max = shared_workspace_max[storage.shared_workspace_identity];
      const auto shared_bytes = checkedMemoryAdd(storage.shared_workspace_bytes, storage.shared_workspace_reserved_bytes);
      result.derivative_owner_bytes -= std::min(previous_max, shared_bytes);
      previous_max = std::max(previous_max, shared_bytes);
    }
    result.sample_scalar_bytes += batch.weights.capacity() * sizeof(wftrain::DerivativeReal) +
        batch.local_energies.capacity() * sizeof(wftrain::DerivativeValue);
    if (batch.additive_local_energy_vjp)
      result.sample_scalar_bytes += batch.additive_local_energy_vjp->weighted_vjp.capacity() *
          sizeof(wftrain::DerivativeValue);
  }
  result.retained_numeric_bytes =
      result.derivative_owner_bytes + result.sample_scalar_bytes;
  return result;
}

/** Measure process RSS including opaque heap owners; zero means unavailable. */
std::size_t currentResidentBytes()
{
#ifdef __linux__
  std::ifstream status("/proc/self/statm");
  std::size_t virtual_pages = 0, resident_pages = 0;
  const long page_bytes = sysconf(_SC_PAGESIZE);
  if (status >> virtual_pages >> resident_pages && page_bytes > 0 &&
      resident_pages <= std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(page_bytes))
    return resident_pages * static_cast<std::size_t>(page_bytes);
#endif
  return 0;
}

/** Collective phase snapshots, not a continuous job peak or numeric owner count. */
void logTrainingMemoryPhase(Communicate& communicator, std::size_t iteration, const char* phase)
{
  double local[2]{static_cast<double>(currentResidentBytes()), 0};
  if (psiformer::memory::enabled())
  {
    psiformer::memory::Event event{"boundary", 0, phase, "rss_sample", "host", -1,
                                  static_cast<std::size_t>(local[0])};
    event.generation = iteration;
    psiformer::memory::emit(event);
  }
#ifdef __linux__
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    local[1] = static_cast<double>(usage.ru_maxrss) * 1024;
#endif
  std::vector<double> ranks(2 * communicator.size());
  communicator.allgather(local, ranks.data(), 2);
  if (communicator.rank() == 0)
  {
    double sum = 0, maximum = 0, high_water_sum = 0, high_water_max = 0;
    for (int rank = 0; rank < communicator.size(); ++rank)
    {
      sum += ranks[2 * rank];
      maximum = std::max(maximum, ranks[2 * rank]);
      high_water_sum += ranks[2 * rank + 1];
      high_water_max = std::max(high_water_max, ranks[2 * rank + 1]);
    }
    app_log() << "WFTrain memory iteration=" << iteration << " phase=" << phase
              << " ranks=" << communicator.size() << " rank0_rss_bytes=" << local[0]
              << " sum_current_rss_bytes=" << sum << " max_current_rss_bytes=" << maximum
              << " sum_rank_high_water_bytes=" << high_water_sum
              << " max_rank_high_water_bytes=" << high_water_max << std::endl;
  }
}

/** Enforce a measured per-process boundary guard including opaque heap owners.
 * It is not an allocator limit: transient peaks between checks can exceed it.
 */
void enforceResidentBudget(Communicate& communicator, std::size_t maximum_bytes)
{
  if (!maximum_bytes)
    return;
  const auto measured = currentResidentBytes();
  int failed = measured == 0 || measured > maximum_bytes;
  communicator.allreduce(failed);
  if (failed)
    throw UniformCommunicateError(
        "WFTrainBatched: process RSS exceeds maximum_resident_bytes or cannot be measured");
}

/** Reject a rank-local numeric-budget violation before derivative evaluation. */
void enforceMemoryBudget(Communicate& communicator,
                         std::size_t retained_bytes,
                         std::size_t maximum_retained_bytes)
{
  if (maximum_retained_bytes == 0)
    return;
  int budget_exceeded = retained_bytes > maximum_retained_bytes ? 1 : 0;
  communicator.allreduce(budget_exceeded);
  if (budget_exceeded != 0)
    throw UniformCommunicateError(
        "WFTrainBatched: retained training storage exceeds maximum_retained_bytes on at least one rank");
}

/** Add byte extents without allowing accounting overflow to wrap the budget. */
std::size_t checkedMemoryAdd(std::size_t lhs, std::size_t rhs)
{
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    throw std::overflow_error("WFTrainBatched: training memory estimate overflowed");
  return lhs + rhs;
}

/** Multiply byte extents without allowing accounting overflow to wrap the budget. */
std::size_t checkedMemoryMultiply(std::size_t lhs, std::size_t rhs)
{
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    throw std::overflow_error("WFTrainBatched: training memory estimate overflowed");
  return lhs * rhs;
}

/// Charge transactional factor arrays before construction; optimizer/inverse allowance is separate.
std::size_t kfacStatisticsBytes(const wftrain::KFACBlockRegistry& registry)
{
  std::size_t elements = 0;
  for (const auto& block : registry.affineBlocks())
  {
    const auto activation = checkedMemoryAdd(block.input_width,
        block.bias_block == wftrain::NO_KFAC_BIAS_BLOCK ? 0 : 1);
    elements = checkedMemoryAdd(elements, checkedMemoryMultiply(2, checkedMemoryAdd(
        checkedMemoryMultiply(activation, activation), checkedMemoryMultiply(block.output_width, block.output_width))));
  }
  for (auto block : registry.fallbackBlocks())
    elements = checkedMemoryAdd(elements, checkedMemoryMultiply(registry.parameterSchema().blocks()[block].count,
        registry.genericBlocks().empty() ? 2 : 6));
  return checkedMemoryMultiply(elements, sizeof(wftrain::DerivativeReal));
}

/// Prepare one population-wide scalar weighting after all ranks finish product preparation.
wftrain::EnergyClippingTransform makeScoreResidualTransform(Communicate& communicator,
    const std::vector<wftrain::PreparedEnergyGradientBatch>& batches,
    const WFTrainDriverInput& input, std::size_t already_retained_bytes)
{
  std::size_t count = 0;
  for (const auto& batch : batches)
    count = checkedMemoryAdd(count, batch.weights.size());
  enforceMemoryBudget(communicator, checkedMemoryAdd(already_retained_bytes,
      checkedMemoryMultiply(count, sizeof(wftrain::DerivativeValue) + sizeof(wftrain::DerivativeReal))),
      input.getMaximumRetainedBytes());
  std::vector<wftrain::DerivativeValue> energies;
  std::vector<wftrain::DerivativeReal> weights;
  std::exception_ptr failure;
  try
  {
    energies.reserve(count);
    weights.reserve(count);
    for (const auto& batch : batches)
    {
      energies.insert(energies.end(), batch.local_energies.begin(), batch.local_energies.end());
      weights.insert(weights.end(), batch.weights.begin(), batch.weights.end());
    }
  }
  catch (...) { failure = std::current_exception(); }
  int failed = failure ? 1 : 0;
  communicator.allreduce(failed);
  if (failed)
  {
    if (failure)
      std::rethrow_exception(failure);
    throw UniformCommunicateError("Another rank failed preparing score residual samples");
  }
  const auto clipping = [&]() {
    using T = wftrain::ScoreResidualTransform;
    const auto& policy = input.getScoreResidualPolicy();
    if (policy.transform == T::COMPONENTWISE_MEAN)
      return wftrain::prepareMeanResidualEnergyClippingTransform({energies.data(), energies.size()},
          {weights.data(), weights.size()}, policy.width, &communicator);
    if (policy.transform == T::SOFT_SQUEEZE ||
        input.getKFACObjective() == WFTrainDriverInput::KFACObjective::CLIPPED_COVARIANCE)
    {
      int nonuniform = std::any_of(weights.begin(), weights.end(), [](double weight) { return weight != 1; });
      communicator.allreduce(nonuniform);
      if (nonuniform)
        throw UniformCommunicateError("Soft score weighting requires unit statistical weights on every rank");
      return wftrain::prepareEnergyClippingTransform({energies.data(), energies.size()},
          {wftrain::EnergyClippingScaleRule::LINEAR_QUANTILE_SOFT_SQUEEZE,
           policy.width, policy.quantile}, &communicator);
    }
    return wftrain::prepareComplexEnergyClippingTransform({energies.data(), energies.size()},
        {weights.data(), weights.size()}, policy.width, &communicator);
  }();
  if (communicator.rank() == 0)
  {
    const auto& descriptor = clipping.descriptor();
    app_log() << "WFTrain score weighting: real_center=" << descriptor.center
              << " real_scale=" << descriptor.scale << " imaginary_center=" << descriptor.imaginary_center
              << " imaginary_scale=" << descriptor.imaginary_scale
              << " width_multiplier=" << descriptor.policy.width_multiplier
              << " statistics=global_energy_only_host raw_energy_reporting=unchanged" << std::endl;
  }
  return clipping;
}

/** Gather scalar energies/weights only; parameter products remain streamed. */
std::vector<wftrain::DerivativeValue> makeClippedKFACObjective(
    Communicate& communicator,
    const std::vector<wftrain::PreparedEnergyGradientBatch>& batches,
    const WFTrainDriverInput& input,
    std::size_t already_retained_bytes)
{
  std::uint64_t local_count = 0;
  for (const auto& batch : batches)
    local_count += batch.weights.size();
  std::vector<std::uint64_t> rank_counts(communicator.size());
  communicator.allgather(&local_count, rank_counts.data(), 1);
  std::vector<int> counts(rank_counts.size()), offsets(rank_counts.size());
  std::size_t total = 0;
  for (std::size_t rank = 0; rank < counts.size(); ++rank)
  {
    if (rank_counts[rank] > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) - total)
      throw std::overflow_error("KFAC scalar objective population exceeds transport count limit");
    offsets[rank] = static_cast<int>(total);
    counts[rank] = static_cast<int>(rank_counts[rank]);
    total += counts[rank];
  }
  if (total == 0)
    throw std::invalid_argument("KFAC scalar objective requires samples");
  // Include gathered inputs, transform scratch/output and the returned local copy.
  enforceMemoryBudget(communicator,
      checkedMemoryAdd(already_retained_bytes, checkedMemoryMultiply(total, 80)),
      input.getMaximumRetainedBytes());
  std::vector<double> local_energies, local_weights, energies, weights;
  std::exception_ptr preparation_failure;
  try
  {
    local_energies.resize(std::max<std::size_t>(local_count, 1));
    local_weights.resize(local_energies.size());
    energies.resize(total);
    weights.resize(total);
    std::size_t offset = 0;
    for (const auto& batch : batches)
      for (std::size_t i = 0; i < batch.weights.size(); ++i, ++offset)
      {
        local_energies[offset] = batch.local_energies[i].real();
        local_weights[offset] = batch.weights[i];
      }
  }
  catch (...) { preparation_failure = std::current_exception(); }
  int failed = preparation_failure ? 1 : 0;
  communicator.allreduce(failed);
  if (failed)
  {
    if (preparation_failure) std::rethrow_exception(preparation_failure);
    throw UniformCommunicateError("Another rank failed preparing KFAC objective scalar gather");
  }
  communicator.gatherv(local_energies.data(), energies.data(), static_cast<int>(local_count), counts, offsets);
  communicator.gatherv(local_weights.data(), weights.data(), static_cast<int>(local_count), counts, offsets);
  communicator.bcast(energies.data(), static_cast<int>(total));
  communicator.bcast(weights.data(), static_cast<int>(total));
  std::vector<wftrain::DerivativeValue> local_residual;
  try
  {
    wftrain::ClippedScoreObjectiveControl control;
    using M = WFTrainDriverInput::KFACClipMode;
    control.mode = input.getKFACClipMode() == M::SOFT_MEDIAN ? wftrain::ScoreObjectiveClipMode::SOFT_MEDIAN
        : input.getKFACClipMode() == M::HARD_MEDIAN ? wftrain::ScoreObjectiveClipMode::HARD_MEDIAN
                                                : wftrain::ScoreObjectiveClipMode::HARD_MEAN;
    control.width = input.getKFACClipWidth();
    control.quantile = input.getKFACClipQuantile();
    control.exclude_width = input.getKFACExcludeWidth();
    const auto result = wftrain::prepareClippedScoreObjective(
        {energies.data(), energies.size()}, {weights.data(), weights.size()}, control);
    const auto first = result.coefficients.begin() + offsets[communicator.rank()];
    local_residual.assign(first, first + local_count);
    if (communicator.rank() == 0)
      app_log() << "WFTrain KFAC objective transform center=" << result.center << " scale=" << result.scale
                << " transformed_mean=" << result.transformed_mean << " unmasked=" << result.unmasked_samples
                << " global_samples=" << total << " gradient_factor=1 mask_denominator=count" << std::endl;
  }
  catch (...) { preparation_failure = std::current_exception(); }
  failed = preparation_failure ? 1 : 0;
  communicator.allreduce(failed);
  if (failed)
  {
    if (preparation_failure) std::rethrow_exception(preparation_failure);
    throw UniformCommunicateError("Another rank failed preparing clipped KFAC objective residual");
  }
  return local_residual;
}

/** Conservatively charge transient restarted-GMRES and proposal storage.
 *
 * This estimate intentionally overlaps some short-lived vectors.  A hard memory
 * ceiling must fail before entering a solve rather than relying on allocator
 * failure, and conservative O(mP) or O(mB_local+P) accounting is preferable to
 * understating the peak.
 */
std::size_t piiWorkspaceBytes(
    WFTrainDriverInput::Optimizer optimizer,
    const wftrain::ScoreCovarianceFactor& factor,
    std::size_t restart_length)
{
  const bool minimum = optimizer ==
      WFTrainDriverInput::Optimizer::MINIMUM_PROJECTED_INVERSE_ITERATION;
  const std::size_t parameter_count = factor.parameterSchema().parameterCount();
  const std::size_t local_extent =
      minimum ? factor.localSampleVectorSize() : parameter_count;
  const std::size_t global_extent =
      minimum ? factor.globalSampleVectorSize() : parameter_count;
  const std::size_t restart = std::min(restart_length, global_extent);
  const std::size_t restart_plus_one = checkedMemoryAdd(restart, 1);

  std::size_t complex_values =
      checkedMemoryMultiply(restart_plus_one, local_extent);
  complex_values = checkedMemoryAdd(
      complex_values, checkedMemoryMultiply(restart_plus_one, restart));
  complex_values = checkedMemoryAdd(
      complex_values, checkedMemoryMultiply(3, local_extent));
  complex_values = checkedMemoryAdd(
      complex_values, checkedMemoryAdd(checkedMemoryMultiply(3, restart), 1));
  complex_values = checkedMemoryAdd(
      complex_values,
      minimum
          ? checkedMemoryAdd(checkedMemoryMultiply(2, local_extent),
                             checkedMemoryMultiply(2, parameter_count))
          : checkedMemoryAdd(checkedMemoryMultiply(3, parameter_count),
                             factor.localSampleVectorSize()));
  return checkedMemoryAdd(
      checkedMemoryMultiply(complex_values, sizeof(wftrain::DerivativeValue)),
      checkedMemoryMultiply(restart, sizeof(wftrain::DerivativeReal)));
}

/** Conservatively charge parameter-space SR solve and proposal transients. */
std::size_t parameterSpaceSRWorkspaceBytes(std::size_t parameter_count)
{
  // PCG owns five P-vectors including its returned solution.  The update rule
  // simultaneously retains the RHS, metric action, candidate, and transaction
  // direction at its peak.
  return checkedMemoryMultiply(
      checkedMemoryMultiply(9, parameter_count),
      sizeof(wftrain::DerivativeValue));
}

/** Conservatively charge minSR/SPRING sample-space solve transients. */
std::size_t minimumStepWorkspaceBytes(
    WFTrainDriverInput::Optimizer optimizer,
    const wftrain::ScoreCovarianceFactor& factor)
{
  const std::size_t sample_values = checkedMemoryMultiply(
      5, factor.localSampleVectorSize());
  const std::size_t parameter_vectors =
      optimizer == WFTrainDriverInput::Optimizer::SPRING ? 4 : 3;
  const std::size_t parameter_values = checkedMemoryMultiply(
      parameter_vectors, factor.parameterSchema().parameterCount());
  return checkedMemoryMultiply(
      checkedMemoryAdd(sample_values, parameter_values),
      sizeof(wftrain::DerivativeValue));
}

/** Validate the score products before any rank enters a factor constructor collective. */
void validateScoreBatches(const wftrain::StructuredParameterSchema& schema,
                          std::size_t parameter_version,
                          const std::vector<wftrain::PreparedEnergyGradientBatch>& batches)
{
  for (const auto& batch : batches)
  {
    if (!batch.derivative_operator)
      throw std::invalid_argument("WFTrainBatched: prepared a null score derivative operator");
    const auto& derivative_operator = *batch.derivative_operator;
    const auto capabilities         = derivative_operator.capabilities();
    if (derivative_operator.parameterSchema().providerId() != schema.providerId() ||
        derivative_operator.parameterSchema().fingerprint() != schema.fingerprint() ||
        derivative_operator.parameterVersion() != parameter_version ||
        derivative_operator.sampleCount() != batch.weights.size() ||
        derivative_operator.sampleCount() != batch.local_energies.size())
      throw std::invalid_argument("WFTrainBatched: prepared score-batch identity or extent mismatch");
    if (!capabilities.supports(wftrain::DerivativeProduct::SCORE_JVP) ||
        !capabilities.supports(wftrain::DerivativeProduct::SCORE_VJP) ||
        !capabilities.supports(wftrain::DerivativeAdjoint::HERMITIAN) ||
        capabilities.parameter_scalar_domain != wftrain::ParameterScalarDomain::REAL64 ||
        capabilities.result_scalar_domain != wftrain::ParameterScalarDomain::COMPLEX128 ||
        capabilities.execution_domain != wftrain::DerivativeExecutionDomain::HOST ||
        capabilities.reduction_domain == wftrain::ReductionDomain::GLOBAL ||
        !capabilities.block_streaming || capabilities.maximum_vjp_channels == 0 ||
        capabilities.maximum_parameter_chunk_size == 0 ||
        capabilities.maximum_sample_tile_size == 0)
      throw std::invalid_argument("WFTrainBatched: prepared batch lacks bounded score products");
    if (derivative_operator.sampleOffset() >
        std::numeric_limits<std::size_t>::max() - derivative_operator.sampleCount())
      throw std::overflow_error("WFTrainBatched: prepared score-batch interval overflows size_t");
    for (std::size_t sample = 0; sample < batch.weights.size(); ++sample)
      if (!wftrain::isFiniteTrainingReal(batch.weights[sample]) || batch.weights[sample] < 0.0 ||
          !wftrain::isFiniteTrainingReal(batch.local_energies[sample].real()) ||
          !wftrain::isFiniteTrainingReal(batch.local_energies[sample].imag()) ||
          batch.local_energies[sample].imag() != 0.0)
        throw std::invalid_argument("WFTrainBatched: prepared real score samples are invalid");
  }
}

/** Complete a direct factor-gradient update with failure-atomic state publication. */
wftrain::TrainingIterationResult runScoreFactorIteration(
    wftrain::StructuredParameterProvider& provider,
    const wftrain::StructuredParameterSnapshot& parameters,
    const wftrain::ScoreCovarianceFactor& factor,
    wftrain::TrainingUpdateRule& update_rule,
    wftrain::TrainingIterationState& state,
    const wftrain::DistributedParameterReduction& reduction,
    wftrain::ParameterUpdateObserver& observer,
    wftrain::DerivativeReal weight_sum,
    wftrain::DerivativeReal energy_variance)
{
  const auto& schema = provider.parameterSchema();
  if (state.completed_iterations != 0 &&
      (state.schema_fingerprint != schema.fingerprint() ||
       state.parameter_version != parameters.version))
    throw std::invalid_argument("WFTrainBatched: training state does not match the parameter provider");
  if (state.completed_iterations == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("WFTrainBatched: training iteration count overflow");

  wftrain::TrainingIterationState pending_state;
  pending_state.completed_iterations = state.completed_iterations + 1;
  pending_state.parameter_version    = parameters.version;
  pending_state.schema_fingerprint   = schema.fingerprint();

  wftrain::EnergyGradientResult objective;
  objective.schema_fingerprint = schema.fingerprint();
  objective.parameter_version  = parameters.version;
  objective.reduction_domain   = wftrain::ReductionDomain::GLOBAL;
  objective.estimator          = wftrain::EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN;
  objective.sample_count       = factor.globalSampleVectorSize();
  objective.weight_sum         = weight_sum;
  objective.mean_energy        = factor.meanEnergy();
  objective.energy_variance    = energy_variance;
  objective.gradient           = factor.scoreCovarianceGradient();

  const std::size_t committed_version = wftrain::completeParameterUpdate(
      provider, parameters, objective.parameterGradient(), update_rule, reduction);
  pending_state.parameter_version = committed_version;
  state.completed_iterations       = pending_state.completed_iterations;
  state.parameter_version          = pending_state.parameter_version;
  state.schema_fingerprint.swap(pending_state.schema_fingerprint);
  observer.parametersPublished(committed_version);
  return {state.completed_iterations, committed_version, std::move(objective)};
}

/** Globally reduced energy moments retained for direct factor diagnostics. */
struct EnergyMoments
{
  wftrain::DerivativeReal weight_sum = 0.0;
  wftrain::DerivativeReal variance   = 0.0;
};

/** Return globally weighted energy moments for live sample-space batches. */
EnergyMoments energyMoments(
    Communicate& communicator,
    const std::vector<wftrain::PreparedEnergyGradientBatch>& batches,
    wftrain::DerivativeReal mean_energy)
{
  wftrain::DerivativeReal local_weight = 0.0;
  wftrain::DerivativeReal local_sum    = 0.0;
  for (const auto& batch : batches)
    for (std::size_t sample = 0; sample < batch.weights.size(); ++sample)
    {
      const wftrain::DerivativeReal delta = batch.local_energies[sample].real() - mean_energy;
      local_weight += batch.weights[sample];
      local_sum += batch.weights[sample] * delta * delta;
    }
  std::vector<wftrain::DerivativeReal> moments{local_weight, local_sum};
  communicator.allreduce(moments);
  if (!wftrain::isFiniteTrainingReal(moments[0]) || moments[0] <= 0.0 ||
      !wftrain::isFiniteTrainingReal(moments[1]))
    throw UniformCommunicateError(
        "WFTrainBatched: direct score-factor diagnostics require positive finite global weight");
  return {moments[0], moments[1] / moments[0]};
}

/// Extend the stable stage fingerprint with a trivially represented scalar.
template<class T>
void extendStageFingerprint(std::uint64_t& hash, const T& value) noexcept
{
  static_assert(std::is_trivially_copyable_v<T>);
  const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
  for (std::size_t index = 0; index < sizeof(T); ++index)
  {
    hash ^= bytes[index];
    hash *= UINT64_C(1099511628211);
  }
}

/// Extend the stable stage fingerprint with an explicitly bounded string.
void extendStageFingerprint(std::uint64_t& hash, std::string_view value) noexcept
{
  const std::size_t size = value.size();
  extendStageFingerprint(hash, size);
  for (const unsigned char byte : value)
  {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
}

/// Route owned strings through the explicit byte/string overload.
void extendStageFingerprint(std::uint64_t& hash, const std::string& value) noexcept
{
  extendStageFingerprint(hash, std::string_view(value));
}

/** Bind checkpoint continuation to training and Markov-chain semantics.
 *
 * Output cadence, memory ceilings, and the requested final block count are
 * intentionally excluded so they may be changed for a resumed run.
 */
std::string stageConfigurationFingerprint(const WFTrainDriverInput& training,
                                          const QMCDriverInput& qmc,
                                          const VMCDriverInput& vmc,
                                          const wftrain::StructuredParameterSchema& schema,
                                          std::size_t steps_per_block,
                                          std::uint64_t resolved_overlap_fingerprint,
                                          const std::string& complex_physical_identity = {},
                                          bool resident_energy_adam = false, bool resident_energy_sr = false)
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  extendStageFingerprint(hash, schema.fingerprint());
  if (resident_energy_adam)
    extendStageFingerprint(hash, std::string(training.getGradientEstimator() == WFTrainDriverInput::GradientEstimator::SCORE_COVARIANCE
        ? "resident-score-c2-adam-host-v1" : "resident-energy-c3-adam-v1"));
  if (resident_energy_sr)
    extendStageFingerprint(hash, std::string("resident-energy-c3-sr-v1"));
  if (training.getComplexGpuTraining().enabled())
  {
    extendStageFingerprint(hash, std::string("paired-gpu-pathwise-first-order-v1"));
    extendStageFingerprint(hash, training.getComplexGpuTraining().kinetic_targets);
  }
  if (!complex_physical_identity.empty())
    extendStageFingerprint(hash, complex_physical_identity);
  const auto objective = static_cast<std::underlying_type_t<WFTrainDriverInput::Objective>>(
      training.getObjective());
  const auto optimizer = static_cast<std::underlying_type_t<WFTrainDriverInput::Optimizer>>(
      training.getOptimizer());
  extendStageFingerprint(hash, objective);
  // Preserve historical fingerprints when no independent estimator was selected.
  if (training.getGradientEstimator() != WFTrainDriverInput::GradientEstimator::LEGACY)
  {
    extendStageFingerprint(hash, std::string("energy-gradient-estimator-v1"));
    extendStageFingerprint(hash, static_cast<int>(training.getGradientEstimator()));
  }
  if (training.getOptimizer() != WFTrainDriverInput::Optimizer::KFAC &&
      training.getScoreResidualPolicy().transform != wftrain::ScoreResidualTransform::RAW)
  {
    extendStageFingerprint(hash, std::string("score-residual-policy-v1"));
    extendStageFingerprint(hash, static_cast<int>(training.getScoreResidualPolicy().transform));
    extendStageFingerprint(hash, training.getScoreResidualPolicy().width);
    extendStageFingerprint(hash, training.getScoreResidualPolicy().quantile);
  }
  if (training.samplesOrbitalStudent())
    extendStageFingerprint(hash, std::string("orbital-student-measure-energy-off-v1"));
  extendStageFingerprint(hash, optimizer);
  if (training.getKFACDefaults() != WFTrainDriverInput::KFACDefaults::LEGACY ||
      training.getKFACGradient() != WFTrainDriverInput::KFACGradient::LEGACY ||
      training.getKFACUpdateOrder() != WFTrainDriverInput::KFACUpdateOrder::POST_INVERSE_RATE ||
      training.getKFACScheduleArithmetic() != WFTrainDriverInput::KFACScheduleArithmetic::DIVIDE_RATE)
  {
    extendStageFingerprint(hash, std::string("typed-kfac-policy-v1"));
    extendStageFingerprint(hash, static_cast<int>(training.getKFACDefaults()));
    extendStageFingerprint(hash, static_cast<int>(training.getKFACGradient()));
    extendStageFingerprint(hash, static_cast<int>(training.getKFACUpdateOrder()));
    extendStageFingerprint(hash, static_cast<int>(training.getKFACScheduleArithmetic()));
  }
  extendStageFingerprint(hash, training.getParameterChunkSize());
  extendStageFingerprint(hash, training.getLearningRate());
  extendStageFingerprint(hash, training.getMomentumDecay());
  extendStageFingerprint(hash, training.getRMSDecay());
  extendStageFingerprint(hash, training.getLMOverlapShift());
  extendStageFingerprint(hash, training.getLMNormalizationXi());
  extendStageFingerprint(hash, training.getLMMaximumSubspaceDimension());
  extendStageFingerprint(hash, training.getLMInnerIterations());
  extendStageFingerprint(hash, training.getLMInnerRestartLength());
  extendStageFingerprint(hash, training.getLMInnerRelativeTolerance());
  extendStageFingerprint(hash, training.getKFACFallbackDamping());
  if (training.getKFACPersistState())
  {
    extendStageFingerprint(hash, std::string("kfac-persistent-v1"));
    extendStageFingerprint(hash, training.getKFACEMADecay());
    extendStageFingerprint(hash, training.getKFACInversePeriod());
    extendStageFingerprint(hash, training.getKFACLearningRateDecay());
    extendStageFingerprint(hash, training.getKFACScheduleFloat32());
  }
  if (training.getOptimizer() == WFTrainDriverInput::Optimizer::KFAC &&
      (training.getKFACObjective() != WFTrainDriverInput::KFACObjective::RAW_COVARIANCE ||
       training.getKFACCurvature() != WFTrainDriverInput::KFACCurvature::NATIVE_SCORE ||
       training.getKFACDampingPolicy() != WFTrainDriverInput::KFACDamping::EQUAL_SPLIT ||
       training.getKFACConstraint() != WFTrainDriverInput::KFACConstraint::NONE))
  {
    extendStageFingerprint(hash, std::string("kfac-objective-curvature-v1"));
    extendStageFingerprint(hash, static_cast<int>(training.getKFACObjective()));
    extendStageFingerprint(hash, static_cast<int>(training.getKFACClipMode()));
    extendStageFingerprint(hash, training.getKFACClipWidth());
    extendStageFingerprint(hash, training.getKFACClipQuantile());
    extendStageFingerprint(hash, training.getKFACExcludeWidth());
    extendStageFingerprint(hash, static_cast<int>(training.getKFACCurvature()));
    extendStageFingerprint(hash, static_cast<int>(training.getKFACDampingPolicy()));
    extendStageFingerprint(hash, static_cast<int>(training.getKFACConstraint()));
    extendStageFingerprint(hash, training.getKFACNormConstraint());
    if (training.getKFACObjective() == WFTrainDriverInput::KFACObjective::COMPONENTWISE_MAD_COVARIANCE ||
        training.getKFACObjective() == WFTrainDriverInput::KFACObjective::COMPONENTWISE_MEAN_RESIDUAL)
    {
      extendStageFingerprint(hash, std::string("complex-componentwise-mad-covariance-v1"));
      extendStageFingerprint(hash, training.getKFACComplexClipWidth());
    }
  }
  extendStageFingerprint(hash, static_cast<int>(training.getLineSearch()));
  extendStageFingerprint(hash, training.getUpdateScale());
  extendStageFingerprint(hash, training.getWeightDecay());
  extendStageFingerprint(hash, training.getLineSearchMinimumESSFraction());
  extendStageFingerprint(hash, training.getLineSearchMaximumWeight());
  extendStageFingerprint(hash, training.getLineSearchConfidence());
  extendStageFingerprint(hash, training.getLineSearchWarmupSteps());
  extendStageFingerprint(hash, training.getLineSearchSamplingSteps());
  extendStageFingerprint(hash, training.getAdamBeta1());
  extendStageFingerprint(hash, training.getAdamBeta2());
  extendStageFingerprint(hash, training.getEpsilon());
  extendStageFingerprint(hash, training.getDamping());
  extendStageFingerprint(hash, training.getDampingMultiplier());
  extendStageFingerprint(hash, training.getMinimumRetryDamping());
  extendStageFingerprint(hash, training.getMaximumDampingAttempts());
  extendStageFingerprint(hash, training.getMaximumUpdateNorm());
  extendStageFingerprint(hash, training.getMaximumMetricNorm());
  extendStageFingerprint(hash, training.getCenterShift());
  extendStageFingerprint(hash, training.getObjectiveTolerance());
  const auto pii_shift_mode =
      static_cast<std::underlying_type_t<WFTrainDriverInput::PIIShiftMode>>(
          training.getPIIShiftMode());
  extendStageFingerprint(hash, pii_shift_mode);
  extendStageFingerprint(hash, training.getPIIExplicitTau());
  extendStageFingerprint(hash, training.getPIIEnergyUndershoot());
  extendStageFingerprint(hash, training.getMaximumDampingRetries());
  extendStageFingerprint(hash, training.getGMRESRestartLength());
  extendStageFingerprint(hash, training.getSpringHistoryDecay());
  extendStageFingerprint(hash, training.getSpringMaximumUpdateNormSquared());
  extendStageFingerprint(hash, training.getKrylovRelativeTolerance());
  extendStageFingerprint(hash, training.getKrylovAbsoluteTolerance());
  extendStageFingerprint(hash, training.getKrylovMaximumIterations());
  extendStageFingerprint(hash, training.getKrylovMaximumOperatorApplications());
  extendStageFingerprint(hash, training.getKrylovMaximumSeconds());
  extendStageFingerprint(hash, training.getKrylovStagnationWindow());
  extendStageFingerprint(hash, training.getKrylovMinimumRelativeImprovement());
  extendStageFingerprint(hash, training.getKrylovCurvatureTolerance());
  extendStageFingerprint(hash, training.getKrylovImaginaryTolerance());
  if (training.getObjective() != WFTrainDriverInput::Objective::ENERGY)
  {
    extendStageFingerprint(hash, training.getTeacherWaveFunction());
    extendStageFingerprint(hash, training.getTargetProvenance());
    const auto overlap_target =
        static_cast<std::underlying_type_t<WFTrainDriverInput::OverlapTarget>>(
            training.getOverlapTarget());
    const auto overlap_weighting =
        static_cast<std::underlying_type_t<WFTrainDriverInput::OverlapWeighting>>(
            training.getOverlapWeighting());
    extendStageFingerprint(hash, overlap_target);
    extendStageFingerprint(hash, overlap_weighting);
    extendStageFingerprint(hash, training.getMinimumFidelity());
    extendStageFingerprint(hash, training.getMinimumEffectiveSamples());
    extendStageFingerprint(hash, resolved_overlap_fingerprint);
    if (training.getObjective() == WFTrainDriverInput::Objective::ORBITAL_MSE)
    {
      extendStageFingerprint(hash, static_cast<int>(training.getOrbitalTarget()));
      // Omitted/default normalization preserves all existing stage identities.
      if (training.getOrbitalNormalization() != wftrain::OrbitalLossNormalization::SPIN_SECTOR_SUM)
        extendStageFingerprint(hash, std::string("orbital-full-matrix-mean-v1"));
    }
  }
  extendStageFingerprint(hash, qmc.get_tau());
  extendStageFingerprint(hash, qmc.get_spin_mass());
  extendStageFingerprint(hash, qmc.get_sub_steps());
  extendStageFingerprint(hash, qmc.get_warmup_steps());
  extendStageFingerprint(hash, qmc.get_recalculate_properties_period());
  extendStageFingerprint(hash, qmc.get_blocks_between_recompute());
  extendStageFingerprint(hash, steps_per_block);
  extendStageFingerprint(hash, qmc.get_max_blocks());
  extendStageFingerprint(hash, training.getWarmupSteps());
  // Iteration limit may be extended on restart; the sampling schedule may not.
  extendStageFingerprint(hash, vmc.get_use_drift());
  const auto move_kind = static_cast<std::underlying_type_t<VMCDriverInput::MoveKind>>(
      vmc.get_move_kind());
  const auto selection = static_cast<std::underlying_type_t<VMCDriverInput::ElectronSelection>>(
      vmc.get_electron_selection());
  extendStageFingerprint(hash, move_kind);
  extendStageFingerprint(hash, vmc.get_electrons_per_move());
  extendStageFingerprint(hash, selection);
  std::ostringstream text;
  text << std::hex << std::setw(16) << std::setfill('0') << hash;
  return text.str();
}

/// Validate the native periodic PP operator partition, without touching quadrature RNGs.
std::string periodicScoreECPIdentity(ParticleSet& electrons, const ParticleSet& ions,
    QMCHamiltonian& hamiltonian, const std::string& model_identity,
    const char* identity_prefix = "legacy-torus-score-ecp-v1/")
{
  const auto& provenance = hamiltonian.ecpModelProvenance();
  if (!provenance || !provenance->ordinary_locality_vp || provenance->scientific_definition.empty())
    throw std::invalid_argument("Periodic score ECP requires physical ordinary-locality provenance");
  const CoulombPBCAA *ee = nullptr, *ii = nullptr;
  std::size_t kinetic = 0, local = 0, nonlocal = 0;
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (int term = 0; term < hamiltonian.size(); ++term)
  {
    const auto* current = hamiltonian.getComponent(term);
    if (const auto* k = dynamic_cast<const BareKineticEnergy*>(current))
    {
      if (&k->getTargetPSet() != &electrons)
        throw std::invalid_argument("Periodic score ECP kinetic target differs");
      ++kinetic;
    }
    else if (const auto* aa = dynamic_cast<const CoulombPBCAA*>(current))
    {
      auto*& selected = aa->is_active ? ee : ii;
      if (selected || aa->usesOffload() || aa->ewaldModelDefinition().empty())
        throw std::invalid_argument("Periodic score ECP requires unique native 3D Ewald ee/ii");
      selected = aa;
      extendStageFingerprint(hash, aa->ewaldModelDefinition());
    }
    else if (const auto* l = dynamic_cast<const CoulombPBCAB*>(current))
    {
      if (&l->getSourcePSet() != &ions || &l->Peln != &electrons || l->ewaldModelDefinition().empty())
        throw std::invalid_argument("Periodic score local ECP sources differ");
      bool occupied_pp = false;
      for (int ion = 0; ion < ions.getTotalNum(); ++ion)
      {
        if (static_cast<std::size_t>(ions.GroupID[ion]) >= l->Vspec.size())
          throw std::invalid_argument("Periodic score local ECP species channel is missing");
        // A mixed PP/AE source retains the native bare Coulomb spline for AE species.
        const bool pp_channel = static_cast<bool>(l->Vspec[ions.GroupID[ion]]);
        occupied_pp = occupied_pp || pp_channel;
        extendStageFingerprint(hash, pp_channel);
      }
      if (!occupied_pp)
        throw std::invalid_argument("Periodic score local ECP has no occupied PP channel");
      extendStageFingerprint(hash, l->ewaldModelDefinition());
      ++local;
    }
    else if (const auto* nl = dynamic_cast<const NonLocalECPotential*>(current))
    {
      if (&nl->sourceParticles() != &ions || nl->virtualTargetParticles() != &electrons ||
          nl->usesDLA() || !nl->usesVirtualParticles())
        throw std::invalid_argument("Periodic score nonlocal ECP requires the same ordinary VP sources");
      ++nonlocal;
    }
    else
      throw std::invalid_argument("Periodic score ECP requires kinetic, native Ewald ee/ii and local/nonlocal PP only");
  }
  if (kinetic != 1 || local != 1 || nonlocal != 1 || !ee || !ii ||
      &ee->Ps != &electrons || &ii->Ps != &ions)
    throw std::invalid_argument("Periodic score ECP operator partition or physical sources differ");
  const auto& es = electrons.getSpeciesSet();
  const int charge = es.findAttribute("charge");
  const auto& masses = electrons.get_mass_by_group();
  if (charge < 0 || masses.size() < static_cast<std::size_t>(electrons.groups()))
    throw std::invalid_argument("Periodic score ECP electron charges/masses are missing");
  for (int group = 0; group < electrons.groups(); ++group)
  {
    if (es(charge, group) != -1 || !wftrain::isFiniteTrainingReal(masses[group]) ||
        std::abs(masses[group] - 1) > 64 * std::numeric_limits<double>::epsilon())
      throw std::invalid_argument("Periodic score ECP requires unit-mass charge-minus-one electrons");
  }
  extendStageFingerprint(hash, model_identity);
  extendStageFingerprint(hash, provenance->scientific_definition);
  const auto& species = ions.getSpeciesSet();
  const int ion_charge = species.findAttribute("charge");
  if (ion_charge < 0)
    throw std::invalid_argument("Periodic score ECP ionic charges are missing");
  extendStageFingerprint(hash, ions.getTotalNum());
  for (int ion = 0; ion < ions.getTotalNum(); ++ion)
  {
    extendStageFingerprint(hash, species.speciesName.at(ions.GroupID[ion]));
    const auto z = species(ion_charge, ions.GroupID[ion]);
    if (!wftrain::isFiniteTrainingReal(z))
      throw std::invalid_argument("Periodic score ECP ionic charge is nonfinite");
    extendStageFingerprint(hash, z);
    for (int d = 0; d < 3; ++d)
    {
      if (!wftrain::isFiniteTrainingReal(ions.R[ion][d]))
        throw std::invalid_argument("Periodic score ECP ionic position is nonfinite");
      extendStageFingerprint(hash, ions.R[ion][d]);
    }
  }
  std::ostringstream identity;
  identity << identity_prefix << std::hex << hash;
  return identity.str();
}

/// Preserve the historical real score identity while validating its PP metadata.
std::string realScoreECPIdentity(PsiFormerWF& component, ParticleSet& electrons, QMCHamiltonian& hamiltonian)
{
  const auto& provenance = hamiltonian.ecpModelProvenance();
  if (!component.isPseudopotentialSystem() || !hamiltonian.hasPhysicalNLPP() ||
      !provenance || !provenance->ordinary_locality_vp || provenance->scientific_definition.empty())
    throw std::invalid_argument("GPU score-only ECP requires validated physical ordinary-locality PP metadata");
  const NonLocalECPotential* nonlocal = nullptr;
  for (int term = 0; term < hamiltonian.size(); ++term)
    if (const auto* current = dynamic_cast<const NonLocalECPotential*>(hamiltonian.getComponent(term)))
    {
      if (nonlocal || current->usesDLA() || !current->usesVirtualParticles())
        throw std::invalid_argument("GPU score-only ECP requires one ordinary virtual-particle operator");
      nonlocal = current;
    }
  if (!nonlocal)
    throw std::invalid_argument("GPU score-only ECP has no actual nonlocal operator");
  const auto& ions = nonlocal->sourceParticles();
  component.validateSystem(electrons, ions, "pseudopotential");
  if (electrons.getLattice().SuperCellEnum != SUPERCELL_OPEN)
  {
    const bool cpu_pair = !component.usesAcceleratorParameterProducts() &&
        component.matchesRealDevicePairTarget(electrons);
    if (!component.matchesLegacyTorusPseudopotentialTarget(electrons) && !cpu_pair)
      throw std::invalid_argument("Periodic real score ECP requires legacy torus Gamma or qualified CPU pair metadata");
    return periodicScoreECPIdentity(electrons, ions, hamiltonian, component.trainingConfigurationFingerprint(),
        cpu_pair ? "real-cpu-pair-score-ecp-v1/" : "legacy-torus-score-ecp-v1/");
  }
  const auto& species = ions.getSpeciesSet();
  const int charge = species.findAttribute("charge");
  if (charge < 0)
    throw std::invalid_argument("GPU score-only ECP source charges are missing");
  std::uint64_t hash = UINT64_C(14695981039346656037);
  extendStageFingerprint(hash, provenance->scientific_definition);
  extendStageFingerprint(hash, ions.getTotalNum());
  for (int ion = 0; ion < ions.getTotalNum(); ++ion)
  {
    if (!wftrain::isFiniteTrainingReal(species(charge, ions.GroupID[ion])))
      throw std::invalid_argument("GPU score-only ECP source charge is nonfinite");
    extendStageFingerprint(hash, species.speciesName.at(ions.GroupID[ion]));
    extendStageFingerprint(hash, species(charge, ions.GroupID[ion]));
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      if (!wftrain::isFiniteTrainingReal(ions.R[ion][dimension]))
        throw std::invalid_argument("GPU score-only ECP source position is nonfinite");
      extendStageFingerprint(hash, ions.R[ion][dimension]);
    }
  }
  std::ostringstream identity;
  identity << "real-gpu-score-ecp-v1/" << std::hex << hash;
  return identity.str();
}

} // namespace

namespace
{
/// Reuse only typed full-cell Ewald checks; component-specific source validation stays explicit.
template<class ValidateSource>
std::string pairCoulombIdentity(ParticleSet& electrons, QMCHamiltonian& hamiltonian,
    const std::string& model_identity, const char* prefix, ValidateSource validate_source)
{
  const auto& species = electrons.getSpeciesSet();
  const int charge = species.findAttribute("charge");
  const auto& masses = electrons.get_mass_by_group();
  if (charge < 0 || species.getTotalNum() < electrons.groups() ||
      masses.size() < static_cast<std::size_t>(electrons.groups()))
    throw std::invalid_argument("Real pair GPU training requires initialized electron charges and masses");
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (int group = 0; group < electrons.groups(); ++group)
  {
    if (species(charge, group) != -1 || !wftrain::isFiniteTrainingReal(masses[group]) ||
        std::abs(masses[group] - 1) > 64 * std::numeric_limits<double>::epsilon())
      throw std::invalid_argument("Real pair GPU training requires unit-mass electrons of charge minus one");
    extendStageFingerprint(hash, species(charge, group));
    extendStageFingerprint(hash, masses[group]);
  }
  const CoulombPBCAA* ee = nullptr;
  const CoulombPBCAA* ii = nullptr;
  const CoulombPBCAB* ei = nullptr;
  std::size_t kinetic = 0;
  for (int term = 0; term < hamiltonian.size(); ++term)
  {
    const auto* current = hamiltonian.getComponent(term);
    if (const auto* kinetic_term = dynamic_cast<const BareKineticEnergy*>(current))
    {
      if (&kinetic_term->getTargetPSet() != &electrons)
        throw std::invalid_argument("Real pair GPU training kinetic target differs from its electron set");
      ++kinetic;
      extendStageFingerprint(hash, std::string("unit-mass-kinetic"));
    }
    else if (const auto* aa = dynamic_cast<const CoulombPBCAA*>(current))
    {
      auto*& selected = aa->is_active ? ee : ii;
      if (selected || aa->usesOffload() || aa->ewaldModelDefinition().empty())
        throw std::invalid_argument("Real pair GPU training requires unique native 3D Ewald ee/ii operators");
      selected = aa;
      extendStageFingerprint(hash, aa->ewaldModelDefinition());
    }
    else if (const auto* ab = dynamic_cast<const CoulombPBCAB*>(current))
    {
      if (ei || &ab->Peln != &electrons || ab->ewaldModelDefinition().empty())
        throw std::invalid_argument("Real pair GPU training requires one native 3D Ewald electron-ion operator");
      ei = ab;
      extendStageFingerprint(hash, ab->ewaldModelDefinition());
    }
    else
      throw std::invalid_argument("Real pair GPU training initially supports only kinetic plus 3D Ewald Coulomb physics");
  }
  if (kinetic != 1 || !ee || !ii || !ei || &ee->Ps != &electrons || &ii->Ps != &ei->getSourcePSet())
    throw std::invalid_argument("Real pair GPU training requires actual matching electron/ion Ewald sources");
  validate_source(ei->getSourcePSet());
  extendStageFingerprint(hash, model_identity);
  std::ostringstream identity;
  identity << prefix << std::hex << hash;
  return identity.str();
}
} // namespace

/// Validate real PP/bare sources without evaluating energies or rotating grids.
wftrain::RealECPPhysicalIdentity wftrain::realECPPhysicalIdentity(
    PsiFormerWF& component, ParticleSet& electrons, QMCHamiltonian& hamiltonian)
{
  const auto previous_identity = realScoreECPIdentity(component, electrons, hamiltonian);
  if (electrons.getLattice().SuperCellEnum != SUPERCELL_OPEN)
    return {previous_identity, false};

  const LocalECPotential* local = nullptr;
  const NonLocalECPotential* nonlocal = nullptr;
  const CoulombPotential* ee = nullptr;
  const CoulombPotential* ii = nullptr;
  const CoulombPotential* bare = nullptr;
  std::size_t kinetic = 0;
  for (int term = 0; term < hamiltonian.size(); ++term)
  {
    const auto* current = hamiltonian.getComponent(term);
    if (const auto* k = dynamic_cast<const BareKineticEnergy*>(current))
    {
      if (&k->getTargetPSet() != &electrons)
        throw std::invalid_argument("OPEN ECP kinetic target differs");
      ++kinetic;
    }
    else if (const auto* l = dynamic_cast<const LocalECPotential*>(current))
    {
      if (local)
        throw std::invalid_argument("OPEN ECP has duplicate local PP operators");
      local = l;
    }
    else if (const auto* nl = dynamic_cast<const NonLocalECPotential*>(current))
    {
      if (nonlocal)
        throw std::invalid_argument("OPEN ECP has duplicate nonlocal PP operators");
      nonlocal = nl;
    }
    else if (const auto* c = dynamic_cast<const CoulombPotential*>(current))
    {
      const bool electron_pair = c->isSameParticleSet() && &c->getSourcePSet() == &electrons;
      auto*& selected = c->isSameParticleSet() ? (electron_pair ? ee : ii) : bare;
      if (selected || c->computesForces())
        throw std::invalid_argument("OPEN ECP Coulomb partition is duplicated or unsupported");
      selected = c;
    }
    else
      throw std::invalid_argument("OPEN ECP requires kinetic, native ee/ii, local/nonlocal PP and optional bare-ion Coulomb only");
  }
  if (kinetic != 1 || !local || !nonlocal || !ee || !ii)
    throw std::invalid_argument("OPEN ECP operator partition is incomplete");
  const auto& ions = nonlocal->sourceParticles();
  if (&local->IonConfig != &ions || &local->Pion != &ions || &local->Peln != &electrons ||
      nonlocal->virtualTargetParticles() != &electrons ||
      &ee->getTargetPSet() != &electrons || !ee->evaluatesCurrentTarget() ||
      &ii->getSourcePSet() != &ions || &ii->getTargetPSet() != &ions || ii->evaluatesCurrentTarget() ||
      local->PP.size() != static_cast<std::size_t>(ions.getTotalNum()) || local->Zeff.size() != local->PP.size())
    throw std::invalid_argument("OPEN ECP actual operator sources or local species channels differ");
  const auto& species = ions.getSpeciesSet();
  const int charge = species.findAttribute("charge");
  const auto& electron_species = electrons.getSpeciesSet();
  const int electron_charge = electron_species.findAttribute("charge");
  const auto& masses = electrons.get_mass_by_group();
  if (charge < 0 || electron_charge < 0 || masses.size() < static_cast<std::size_t>(electrons.groups()))
    throw std::invalid_argument("OPEN ECP charges or electron masses are missing");
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (int group = 0; group < electrons.groups(); ++group)
  {
    if (electron_species(electron_charge, group) != -1 || !wftrain::isFiniteTrainingReal(masses[group]) ||
        std::abs(masses[group] - 1) > 64 * std::numeric_limits<double>::epsilon())
      throw std::invalid_argument("OPEN ECP requires unit-mass charge-minus-one electrons");
    extendStageFingerprint(hash, electrons.groupsize(group));
    extendStageFingerprint(hash, electron_species(electron_charge, group));
    extendStageFingerprint(hash, masses[group]);
  }
  extendStageFingerprint(hash, previous_identity);
  extendStageFingerprint(hash, component.trainingConfigurationFingerprint());
  std::vector<std::size_t> missing;
  for (std::size_t ion = 0; ion < local->PP.size(); ++ion)
  {
    const bool pp = local->PP[ion] != nullptr;
    extendStageFingerprint(hash, pp);
    if (pp)
    {
      if (!wftrain::isFiniteTrainingReal(local->Zeff[ion]) || local->Zeff[ion] != species(charge, ions.GroupID[ion]))
        throw std::invalid_argument("OPEN ECP local effective charge differs from its physical ion");
      extendStageFingerprint(hash, local->Zeff[ion]);
    }
    else
      missing.push_back(ion);
  }
  if (missing.size() == local->PP.size() || (missing.empty() != (bare == nullptr)))
    throw std::invalid_argument("OPEN ECP bare-ion Coulomb coverage is missing or double counted");
  if (bare)
  {
    const auto& auxiliary = bare->getSourcePSet();
    const auto& auxiliary_species = auxiliary.getSpeciesSet();
    const int auxiliary_charge = auxiliary_species.findAttribute("charge");
    if (&bare->getTargetPSet() != &electrons || !bare->evaluatesCurrentTarget() ||
        auxiliary.getLattice().SuperCellEnum != SUPERCELL_OPEN || auxiliary_charge < 0 ||
        auxiliary.getTotalNum() != static_cast<int>(missing.size()))
      throw std::invalid_argument("OPEN ECP bare-ion source/target or occupied extent differs");
    std::vector<bool> matched(missing.size(), false);
    for (int site = 0; site < auxiliary.getTotalNum(); ++site)
    {
      if (!wftrain::isFiniteTrainingReal(auxiliary_species(auxiliary_charge, auxiliary.GroupID[site])))
        throw std::invalid_argument("OPEN ECP bare source charge must be finite");
      for (int dimension = 0; dimension < 3; ++dimension)
        if (!wftrain::isFiniteTrainingReal(auxiliary.R[site][dimension]))
          throw std::invalid_argument("OPEN ECP bare source coordinates must be finite");
      std::size_t match = missing.size();
      for (std::size_t candidate = 0; candidate < missing.size(); ++candidate)
      {
        const auto ion = missing[candidate];
        if (!matched[candidate] && auxiliary_species(auxiliary_charge, auxiliary.GroupID[site]) ==
            species(charge, ions.GroupID[ion]) && auxiliary.R[site] == ions.R[ion])
        {
          match = candidate;
          break;
        }
      }
      if (match == missing.size())
        throw std::invalid_argument("OPEN ECP bare source must cover each missing physical ion exactly once");
      matched[match] = true;
      extendStageFingerprint(hash, missing[match]);
      extendStageFingerprint(hash, auxiliary_species(auxiliary_charge, auxiliary.GroupID[site]));
      for (int dimension = 0; dimension < 3; ++dimension)
        extendStageFingerprint(hash, auxiliary.R[site][dimension]);
    }
  }
  std::ostringstream identity;
  identity << "real-open-physical-ecp-v1/" << std::hex << hash;
  return {identity.str(), !missing.empty()};
}

/// Preserve the existing real profile's physical checks and restart identity exactly.
std::string wftrain::realPairCoulombIdentity(PsiFormerWF& component, ParticleSet& electrons, QMCHamiltonian& hamiltonian)
{
  if (!component.matchesRealDevicePairTarget(electrons) || component.isPseudopotentialSystem() ||
      hamiltonian.hasPhysicalNLPP() || hamiltonian.ecpModelProvenance())
    throw std::invalid_argument("Real pair GPU training requires selected Gamma all-electron two-cell metadata");
  return pairCoulombIdentity(electrons, hamiltonian, component.trainingConfigurationFingerprint(),
      "real-gpu-pair-coulomb-v1/", [&](const ParticleSet& ions) {
        component.validateSystem(electrons, ions, "all_electron");
      });
}

#ifdef QMC_COMPLEX
/// Bind paired selected training to the actual target and full physical ionic source.
std::string wftrain::pairedPairCoulombIdentity(const PsiFormerComplexWF& component, ParticleSet& electrons,
    QMCHamiltonian& hamiltonian)
{
  if (!component.matchesPairedDevicePairTarget(electrons) || component.isPseudopotentialSystem() ||
      hamiltonian.hasPhysicalNLPP() || hamiltonian.ecpModelProvenance() || &component.targetParticles() != &electrons)
    throw std::invalid_argument("Paired pair GPU training requires selected Gamma all-electron two-cell metadata");
  return pairCoulombIdentity(electrons, hamiltonian, component.trainingConfigurationFingerprint(),
      "paired-gpu-pair-coulomb-v1/", [&](const ParticleSet& ions) {
        if (&component.sourceParticles() != &ions)
          throw std::invalid_argument("Paired pair GPU Ewald source differs from its actual fixed source");
      });
}
#endif

WFTrainBatched::WFTrainBatched(const ProjectData& project_data,
                               QMCDriverInput&& qmcdriver_input,
                               UPtr<EstimatorManagerNew>&& estimator_manager,
                               VMCDriverInput&& vmc_input,
                               WFTrainDriverInput&& training_input,
                               WalkerConfigurations& walker_configurations,
                               TrialWaveFunction& student_prototype,
                               MCPopulation&& population,
                               const RefVector<RandomBase<FullPrecRealType>>& random_generators,
                               Communicate* communicator,
                               const TrialWaveFunction* frozen_teacher)
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
      training_input_(std::move(training_input)),
      student_prototype_(student_prototype),
      frozen_teacher_(frozen_teacher)
{}

WFTrainBatched::~WFTrainBatched() = default;

void WFTrainBatched::process(xmlNodePtr node)
{
  ScopedTimer startup_timer(timers_.startup_timer);
  print_mem("WFTrainBatched before initialization", app_log());

  try
  {
    inference_input_.readXML(node, training_input_.getObjective() != WFTrainDriverInput::Objective::ORBITAL_MSE);
    if (inference_input_.enabled && !inference_input_.evaluate_energy &&
        training_input_.getObjective() != WFTrainDriverInput::Objective::ORBITAL_MSE)
      throw UniformCommunicateError("Energy-off inference requires orbital_mse");
    if (inference_input_.enabled)
    {
      inference_timer_ = &createGlobalTimer("WFTrain::inference", timer_level_medium);
      // Read only: deriving a separate domain never advances production RNGs.
      std::vector<RandomBase<FullPrecRealType>::uint_type> initial_state;
      rngs_[0].get().save(initial_state);
      inference_seed_ = UINT64_C(14695981039346656037);
      for (const auto word : initial_state)
      {
        inference_seed_ ^= word;
        inference_seed_ *= UINT64_C(1099511628211);
      }
    }
    VMCBatchedSampling::validateMoveConfiguration(vmcdriver_input_, population_.get_golden_electrons(),
                                                  population_.get_golden_twf(), serializing_crowd_walkers_);
    const auto parsed_move = vmcdriver_input_.get_move_kind();
    app_log() << "  WFTrain parsed electron move="
              << (parsed_move == VMCDriverInput::MoveKind::ALL_ELECTRON ? "alle" :
                  parsed_move == VMCDriverInput::MoveKind::PBYP ? "pbyp" : "n_electron")
              << ", particles_per_joint_proposal="
              << (parsed_move == VMCDriverInput::MoveKind::ALL_ELECTRON
                  ? population_.get_golden_electrons().getTotalNum()
                  : parsed_move == VMCDriverInput::MoveKind::PBYP ? 1 : vmcdriver_input_.get_electrons_per_move())
              << "; parsed move kind is included in checkpoint stage identity" << std::endl;

    if (population_.get_golden_electrons().isSpinor())
      throw UniformCommunicateError("WFTrainBatched: spinor training is not supported");
#ifdef QMC_COMPLEX
    bool legacy_paired_mpi2_kfac = false;
    bool legacy_paired_score_ecp = false;
    bool legacy_paired_inline_ecp = false;
    for (const auto& orbital : student_prototype_.getOrbitals())
      if (auto* complex = dynamic_cast<PsiFormerComplexWF*>(orbital.get()))
      {
        complex_training_component_ = complex;
        using O = WFTrainDriverInput::Optimizer;
        const auto cap = training_input_.getMaximumRetainedBytes();
        const auto optimizer = training_input_.getOptimizer();
        const auto& provenance = population_.get_golden_hamiltonian().ecpModelProvenance();
        const bool supported_paired_move = supportsNativeEnergyMove(vmcdriver_input_, *complex) &&
            (vmcdriver_input_.get_move_kind() == VMCDriverInput::MoveKind::PBYP || !vmcdriver_input_.get_use_drift());
        // Supervised raw-orbital VJPs use the explicit GPU publication wrapper, not ENERGY adjoints.
        const bool gpu_orbital_mse = complex->gpuInferenceEnabled() && myComm->size() == 1 &&
            training_input_.getObjective() == WFTrainDriverInput::Objective::ORBITAL_MSE &&
            training_input_.getOrbitalTarget() == WFTrainDriverInput::OrbitalTarget::SINGLE_REFERENCE &&
            optimizer == O::ADAM && training_input_.getLineSearch() == WFTrainDriverInput::LineSearch::OFF &&
            !inference_input_.enabled && qmcdriver_input_.get_max_blocks() == 1 &&
            training_input_.getUpdateScale() == 1 && training_input_.getWeightDecay() == 0 &&
            training_input_.getMaximumUpdateNorm() == 0 && training_input_.getMaximumMetricNorm() == 0 &&
            supported_paired_move && !vmcdriver_input_.get_use_drift() &&
            ((complex->matchesLegacyTorusAllElectronTarget(population_.get_golden_electrons()) &&
              !population_.get_golden_hamiltonian().hasPhysicalNLPP() && !provenance) ||
             (training_input_.samplesOrbitalStudent() &&
              complex->matchesLegacyTorusPseudopotentialTarget(population_.get_golden_electrons()) &&
              population_.get_golden_hamiltonian().hasPhysicalNLPP() && provenance && provenance->ordinary_locality_vp &&
              !provenance->scientific_definition.empty()));
        legacy_paired_inline_ecp = optimizer == O::ADAM && usesInlinePairedECPResponse(training_input_) &&
            !inference_input_.enabled && training_input_.getUpdateScale() == 1 && training_input_.getWeightDecay() == 0 &&
            training_input_.getMaximumUpdateNorm() == 0 &&
            complex->matchesLegacyTorusPseudopotentialTarget(population_.get_golden_electrons()) &&
            population_.get_golden_hamiltonian().hasPhysicalNLPP() && provenance && provenance->ordinary_locality_vp &&
            !provenance->scientific_definition.empty() &&
            supported_paired_move && !vmcdriver_input_.get_use_drift();
        legacy_paired_score_ecp = complex->gpuInferenceEnabled() &&
            complex->matchesLegacyTorusPseudopotentialTarget(population_.get_golden_electrons()) &&
            wftrain::usesEnergyScoreEstimator(training_input_, true) &&
            (optimizer == O::ADAM || (optimizer == O::KFAC && wftrain::supportsHybridKFACPolicy(training_input_))) &&
            training_input_.getLineSearch() == WFTrainDriverInput::LineSearch::OFF &&
            population_.get_golden_hamiltonian().hasPhysicalNLPP() && provenance && provenance->ordinary_locality_vp &&
            !provenance->scientific_definition.empty() &&
            supported_paired_move && !vmcdriver_input_.get_use_drift();
        const bool hip_paired_reference_ecp = complex->gpuInferenceEnabled() &&
            psiformer::compiledPsiFormerAcceleratorSupport().hip && myComm->size() == 1 &&
            complex->matchesLegacyTorusPseudopotentialTarget(population_.get_golden_electrons()) &&
            training_input_.getObjective() == WFTrainDriverInput::Objective::ENERGY && optimizer == O::ADAM &&
            training_input_.getLineSearch() == WFTrainDriverInput::LineSearch::REFERENCE &&
            !inference_input_.enabled && training_input_.getUpdateScale() == 1 && training_input_.getWeightDecay() == 0 &&
            training_input_.getMaximumUpdateNorm() == 0 && cap && cap != std::numeric_limits<std::size_t>::max() &&
            supported_paired_move && !vmcdriver_input_.get_use_drift() &&
            population_.get_golden_hamiltonian().hasPhysicalNLPP() && provenance && provenance->ordinary_locality_vp &&
            !provenance->scientific_definition.empty();
        if (complex->gpuInferenceEnabled() && complex->isPseudopotentialSystem() &&
            psiformer::compiledPsiFormerAcceleratorSupport().hip && !legacy_paired_score_ecp &&
            !legacy_paired_inline_ecp && !hip_paired_reference_ecp && !gpu_orbital_mse)
          throw UniformCommunicateError("WFTrainBatched: paired HIP ECP requires legacy torus score covariance/OFF, "
              "inline full-response Adam/OFF, or MPI1 ENERGY Adam/REFERENCE with default update controls");
        // Widen only the already qualified legacy AE score-covariance lane.
        // Its complex chunks and affine factors remain HOST-staged raw sums.
        legacy_paired_mpi2_kfac = complex->gpuInferenceEnabled() && myComm->size() == 2 &&
            complex->matchesLegacyTorusAllElectronTarget(population_.get_golden_electrons()) &&
            wftrain::supportsHybridKFACPolicy(training_input_) &&
            !population_.get_golden_hamiltonian().hasPhysicalNLPP() &&
            !population_.get_golden_hamiltonian().ecpModelProvenance() &&
            supported_paired_move && !vmcdriver_input_.get_use_drift();
        const bool cpu_orbital_mse = !complex->gpuInferenceEnabled() &&
            training_input_.getObjective() == WFTrainDriverInput::Objective::ORBITAL_MSE &&
            training_input_.getOrbitalTarget() == WFTrainDriverInput::OrbitalTarget::SINGLE_REFERENCE &&
            (optimizer == O::ADAM || optimizer == O::SGD);
        const bool supported_optimizer = optimizer == O::ADAM || optimizer == O::SGD ||
            optimizer == O::STOCHASTIC_RECONFIGURATION ||
            (optimizer == O::KFAC && training_input_.getLineSearch() == WFTrainDriverInput::LineSearch::OFF &&
             (!complex->gpuInferenceEnabled() ||
              (wftrain::supportsHybridKFACPolicy(training_input_) &&
               ((!complex->isPseudopotentialSystem() && !population_.get_golden_hamiltonian().ecpModelProvenance()) ||
                legacy_paired_score_ecp) && !vmcdriver_input_.get_use_drift()))) ||
            (complex->gpuInferenceEnabled() && isFirstOrder(optimizer));
        const bool cpu_scalar_ecp = !complex->gpuInferenceEnabled() &&
            supported_paired_move &&
            complex->isPseudopotentialSystem() && population_.get_golden_hamiltonian().ecpModelProvenance() &&
            population_.get_golden_hamiltonian().ecpModelProvenance()->ordinary_locality_vp;
        const bool cpu_score_ecp = cpu_scalar_ecp && wftrain::usesEnergyScoreEstimator(training_input_, true) &&
            training_input_.getObjective() == WFTrainDriverInput::Objective::ENERGY &&
            training_input_.getLineSearch() == WFTrainDriverInput::LineSearch::OFF;
        // Supervised PP fitting needs raw orbitals only; it does not admit energy Adam or PP dEL.
        const bool cpu_mse_ecp = cpu_scalar_ecp && cpu_orbital_mse && optimizer == O::ADAM &&
            training_input_.samplesOrbitalStudent() &&
            training_input_.getLineSearch() == WFTrainDriverInput::LineSearch::OFF;
        if (population_.get_golden_hamiltonian().hasPhysicalNLPP() && !cpu_score_ecp && !cpu_mse_ecp && !gpu_orbital_mse &&
            !legacy_paired_score_ecp && !legacy_paired_inline_ecp &&
            (!complex->gpuInferenceEnabled() || optimizer != O::ADAM ||
             training_input_.getObjective() != WFTrainDriverInput::Objective::ENERGY ||
             (training_input_.getLineSearch() != WFTrainDriverInput::LineSearch::OFF &&
              training_input_.getLineSearch() != WFTrainDriverInput::LineSearch::REFERENCE) ||
             inference_input_.enabled ||
             training_input_.getUpdateScale() != 1 || training_input_.getWeightDecay() != 0 ||
             training_input_.getMaximumUpdateNorm() != 0))
          throw UniformCommunicateError("Paired ECP training requires CPU score-only ENERGY/OFF, qualified GPU score covariance/OFF, or student single-reference "
              "orbital_mse Adam/OFF with ordinary scalar VP "
              "or GPU ENERGY Adam/OFF or REFERENCE with default update controls");
        // GPU reference candidates require the separate budgeted private lane/image route.
        const bool energy_reference =
            training_input_.getObjective() == WFTrainDriverInput::Objective::ENERGY &&
            training_input_.getLineSearch() == WFTrainDriverInput::LineSearch::REFERENCE &&
            (!complex->gpuInferenceEnabled() ||
             ((optimizer == O::ADAM || optimizer == O::SGD || optimizer == O::STOCHASTIC_RECONFIGURATION) &&
              training_input_.getUpdateScale() == 1 && training_input_.getWeightDecay() == 0 &&
              training_input_.getMaximumUpdateNorm() == 0));
        if (complex->gpuInferenceEnabled() &&
            (!training_input_.getComplexGpuTraining().enabled() ||
             (myComm->size() != 1 && !legacy_paired_mpi2_kfac && !legacy_paired_score_ecp && !legacy_paired_inline_ecp) ||
             !supported_paired_move ||
             !supported_optimizer))
          throw UniformCommunicateError(
              "WFTrainBatched: paired GPU requires explicit caps, MPI1 (or legacy AE score KFAC MPI2 or legacy PP score/inline Adam), "
              "PBYP or supported all-electron/no-drift ENERGY host first-order, "
              "AE covariance KFAC/OFF without drift, or positively damped host-controlled SR");
        if (student_prototype_.getOrbitals().size() != 1 ||
            (!cpu_orbital_mse && !gpu_orbital_mse && training_input_.getObjective() != WFTrainDriverInput::Objective::ENERGY) ||
            !supported_optimizer ||
            (training_input_.getOptimizer() == O::STOCHASTIC_RECONFIGURATION &&
             (!wftrain::isFiniteTrainingReal(training_input_.getDamping()) || training_input_.getDamping() <= 0)) ||
            (training_input_.getLineSearch() != WFTrainDriverInput::LineSearch::OFF && !energy_reference) ||
            (!complex->gpuInferenceEnabled() &&
             (training_input_.getUpdateScale() != 1 || training_input_.getWeightDecay() != 0 ||
              training_input_.getMaximumUpdateNorm() != 0)) || training_input_.getMaximumMetricNorm() != 0 ||
            !cap || cap == std::numeric_limits<std::size_t>::max())
          throw UniformCommunicateError("WFTrainBatched: paired training requires sole ENERGY Adam/plainSGD/covariance KFAC-OFF "
              "or positively damped SR (GPU also momentumSGD/RMSProp/AMSGrad), no metric-norm/selection controls "
              "except ENERGY reference search with default update controls, "
              "shared scale/shrinkage/Euclidean bounds are GPU-only, "
              "and positive finite maximum_retained_bytes; CPU single-reference orbital_mse admits Adam/plainSGD");
        if (complex->gpuInferenceEnabled() && training_input_.getWeightDecay() != 0 &&
            complex->neuralWeightBlocks().empty())
          throw UniformCommunicateError("WFTrainBatched: paired GPU shrinkage requires typed nn_weights metadata");
      }
#endif
    if (training_input_.getKFACObjective() == WFTrainDriverInput::KFACObjective::COMPONENTWISE_MAD_COVARIANCE &&
        ((!complex_training_component_ && training_input_.getKFACDefaults() == WFTrainDriverInput::KFACDefaults::LEGACY &&
          training_input_.getKFACGradient() == WFTrainDriverInput::KFACGradient::LEGACY) ||
         training_input_.getComplexGpuTraining().enabled()))
      throw UniformCommunicateError("Componentwise median KFAC clipping requires the CPU complex energy profile");
    if (training_input_.getComplexGpuTraining().enabled())
    {
#ifdef QMC_COMPLEX
      if (!complex_training_component_ || !complex_training_component_->gpuInferenceEnabled())
#endif
        throw UniformCommunicateError("WFTrainBatched: complex GPU controls require a sole paired GPU profile");
    }
    if (training_input_.getObjective() == WFTrainDriverInput::Objective::ENERGY)
      validateHamiltonianForTraining();

    auto providers = student_prototype_.extractStructuredParameterProviders();
    for (auto& provider : providers)
      if (auto* neural = dynamic_cast<PsiFormerWF*>(&provider.get()); neural && neural->isJastrowAugmented())
      {
        using I = WFTrainDriverInput;
        const bool pretrain = training_input_.getObjective() == I::Objective::ORBITAL_MSE;
        const bool energy = training_input_.getObjective() == I::Objective::ENERGY;
        if (providers.size() != 1 || vmcdriver_input_.get_move_kind() != VMCDriverInput::MoveKind::PBYP ||
            training_input_.getLineSearch() != I::LineSearch::OFF ||
            (pretrain && (training_input_.getOptimizer() != I::Optimizer::ADAM || !training_input_.samplesOrbitalStudent())) ||
            (energy && ((training_input_.getOptimizer() != I::Optimizer::ADAM && training_input_.getOptimizer() != I::Optimizer::KFAC) ||
                         !wftrain::usesEnergyScoreEstimator(training_input_, false))) || (!pretrain && !energy) ||
            !training_input_.getMaximumRetainedBytes() ||
            training_input_.getMaximumRetainedBytes() == std::numeric_limits<std::size_t>::max())
          throw UniformCommunicateError("Jastrow product training requires CPU PBYP, finite retained cap, line_search=off, "
              "student-measure orbital_mse Adam or score-covariance ENERGY Adam/KFAC");
        jastrow_binding_ = std::make_unique<wftrain::JastrowNeuralTrainingBinding>(student_prototype_,
            population_.get_golden_electrons(), *myComm, pretrain);
        std::size_t pair_parameters = 0;
        for (const auto& block : jastrow_binding_->parameterSchema().blocks())
          if (block.update_group == "neural_pair_residual") pair_parameters += block.count;
        const auto traditional_selected = jastrow_binding_->jastrowParameterCount();
        const auto traditional_total = jastrow_binding_->jastrowTotalParameterCount();
        const auto determinant_parameters = jastrow_binding_->parameterSchema().parameterCount()-pair_parameters-traditional_selected;
        app_log() << "WFTrain Jastrow product counts: determinant=" << determinant_parameters << " neural_pair=" << pair_parameters
                  << " traditional_total=" << traditional_total
                  << " objective_active=" << determinant_parameters+(pretrain ? 0 : pair_parameters+traditional_selected)
                  << " objective_frozen=" << (pretrain ? pair_parameters+traditional_total : traditional_total-traditional_selected)
                  << std::endl;
        app_log() << "WFTrain Jastrow product masks: determinant=active neural_pair=" << (pretrain ? "frozen_zero" : "active")
                  << " traditional_selected=" << jastrow_binding_->jastrowParameterCount()
                  << " traditional_objective=" << (pretrain ? "frozen" : "coefficient_optimize_flags")
                  << " max_coefficients=" << wftrain::JastrowNeuralTrainingBinding::MAXIMUM_JASTROW_PARAMETERS
                  << " dense_metric_bytes=" << jastrow_binding_->jastrowParameterCount()*jastrow_binding_->jastrowParameterCount()*sizeof(double)
                  << "; KFAC uses complete-configuration coefficient scores; cross-block curvature omitted" << std::endl;
      }
    if (jastrow_binding_)
    {
      providers.clear();
      providers.emplace_back(*jastrow_binding_);
    }
    if (providers.empty() && !complex_training_component_)
    {
      using O = WFTrainDriverInput::Optimizer;
      const auto optimizer = training_input_.getOptimizer();
      if (optimizer == O::KFAC)
        throw UniformCommunicateError("Conventional-only training has no affine observations required by KFAC");
      if (training_input_.getObjective() != WFTrainDriverInput::Objective::ENERGY ||
          (!isFirstOrder(optimizer) && optimizer != O::STOCHASTIC_RECONFIGURATION &&
           optimizer != O::MINIMUM_STEP_SR && optimizer != O::SPRING &&
           optimizer != O::LINEAR_METHOD && optimizer != O::ACCELERATED_LINEAR_METHOD &&
           optimizer != O::PROJECTED_INVERSE_ITERATION &&
           optimizer != O::MINIMUM_PROJECTED_INVERSE_ITERATION))
        throw UniformCommunicateError("Conventional training requires ENERGY and a supported optimizer; model-specific CPU/ECP qualifications are checked separately");
      if (!training_input_.getLegacyMaximumParameters() || !training_input_.getLegacyDerivativeScratchBytes())
        throw UniformCommunicateError("Conventional training requires positive legacy_maximum_parameters and legacy_derivative_scratch_bytes");
      if (training_input_.getWeightDecay() != 0)
        throw UniformCommunicateError("Conventional training has no neural-weight decay blocks; weight_decay must be zero");
      if (population_.get_golden_electrons().getCoordinates().getKind() != DynamicCoordinateKind::DC_POS)
        throw UniformCommunicateError("Conventional training currently requires CPU particle coordinates");
#ifdef QMC_COMPLEX
      // Binding admits only fixed Gamma HEG14/FreeOrbital/J2 with real
      // coordinates; this does not enable complex or phase parameters.
      if (optimizer != O::ADAM || training_input_.getLineSearch() != WFTrainDriverInput::LineSearch::OFF ||
          training_input_.getUpdateScale() != 1 || training_input_.getMaximumUpdateNorm() != 0 ||
          training_input_.getMaximumMetricNorm() != 0)
        throw UniformCommunicateError("Conventional complex training requires Gamma HEG14 ENERGY Adam "
                                      "with line_search off and default update controls");
#endif
      const auto single_determinants = student_prototype_.findSD();
      const auto multideterminants = student_prototype_.findMSD();
      if (single_determinants.size() + multideterminants.size() != 1)
        throw UniformCommunicateError("Conventional training requires one single-Slater or fixed-orbital multideterminant component");
      if (!multideterminants.empty() &&
          (population_.get_golden_electrons().groups() != 2 || multideterminants[0].get().getDetSize() != 2))
        throw UniformCommunicateError("Conventional coefficient training currently requires two collinear spin groups");
      conventional_binding_ = std::make_unique<wftrain::TrainingWaveFunctionBinding>(
          student_prototype_, population_.get_golden_electrons(),
          wftrain::LegacyWaveFunctionAdapterOptions{training_input_.getLegacyMaximumParameters(),
              training_input_.getLegacyDerivativeScratchBytes(), training_input_.getParameterChunkSize()}, *myComm);
      conventional_binding_->setHamiltonian(population_.get_golden_hamiltonian());
      const auto& variables = conventional_binding_->adapter().registrationVariables();
      const bool linear_method = optimizer == O::LINEAR_METHOD || optimizer == O::ACCELERATED_LINEAR_METHOD;
      const bool inverse_iteration = optimizer == O::PROJECTED_INVERSE_ITERATION ||
          optimizer == O::MINIMUM_PROJECTED_INVERSE_ITERATION;
      std::size_t active_variables = 0, active_linear = 0;
      for (std::size_t variable = 0; variable < variables.size(); ++variable)
        if (variables.where(variable) >= 0)
        {
          ++active_variables;
          active_linear += variables.getType(variable) == optimize::LINEAR_P;
        }
      // Fixed-CI MD orbital/J2 has no qualified mixed reference shear consumer.
      if (linear_method && !active_linear && multideterminants.size() == 1 &&
          conventional_binding_->hasOrbitalRotations() && population_.get_golden_hamiltonian().hasPhysicalNLPP())
        throw UniformCommunicateError("Conventional fixed-CI orbital ECP LM/ALM mixed updates remain unqualified");
      if (active_linear && (linear_method || inverse_iteration))
      {
        const bool ci_j2_layout = active_linear != active_variables &&
            multideterminants.size() == 1 &&
            std::all_of(student_prototype_.getOrbitals().begin(), student_prototype_.getOrbitals().end(),
                [](const auto& component) {
                  return component->getClassName() == "MultiSlaterDetTableMethod" ||
                      component->getClassName() == "TwoBodyJastrow";
                });
        const bool joint_ci_j2 = inverse_iteration && ci_j2_layout;
        const bool mixed_lm_ae = linear_method && ci_j2_layout && !conventional_binding_->hasOrbitalRotations() &&
            population_.get_golden_electrons().getLattice().SuperCellEnum == SUPERCELL_OPEN &&
            !population_.get_golden_hamiltonian().hasPhysicalNLPP() &&
            !population_.get_golden_hamiltonian().ecpModelProvenance() &&
            std::any_of(student_prototype_.getOrbitals().begin(), student_prototype_.getOrbitals().end(),
                [](const auto& component) { return component->getClassName() == "TwoBodyJastrow"; });
        const bool joint_ci_orbital_ecp = (joint_ci_j2 ||
            (linear_method && ci_j2_layout &&
             population_.get_golden_electrons().getLattice().SuperCellEnum == SUPERCELL_OPEN &&
             population_.get_golden_hamiltonian().ecpModelProvenance())) && conventional_binding_->hasOrbitalRotations() &&
            population_.get_golden_hamiltonian().hasPhysicalNLPP() &&
            (linear_method || !multideterminants[0].get().hasCSFExpansion()) &&
            std::any_of(student_prototype_.getOrbitals().begin(), student_prototype_.getOrbitals().end(),
                [](const auto& component) { return component->getClassName() == "TwoBodyJastrow"; });
        if (joint_ci_orbital_ecp)
          multideterminants[0].get().validateAcceptedReferenceOrbitalProfile();
        const bool mixed_lm_ecp = linear_method && ci_j2_layout && !conventional_binding_->hasOrbitalRotations() &&
            population_.get_golden_electrons().getLattice().SuperCellEnum == SUPERCELL_OPEN &&
            population_.get_golden_hamiltonian().hasPhysicalNLPP() &&
            population_.get_golden_hamiltonian().ecpModelProvenance();
        if ((active_linear != active_variables && !joint_ci_j2 && !mixed_lm_ae && !mixed_lm_ecp && !joint_ci_orbital_ecp) ||
            (conventional_binding_->hasOrbitalRotations() && !joint_ci_orbital_ecp) ||
            multideterminants.size() != 1)
          throw UniformCommunicateError("Conventional LM/ALM admit pure coefficients, qualified fixed-orbital CI/J2 AE/ECP or joint DETS/CSF orbital/J2 locality-ECP; other mixed updates remain unqualified");
        // Keep this locality-ECP boundary explicit beside the general mixed guard.
        if (!inverse_iteration &&
            (population_.get_golden_hamiltonian().ecpModelProvenance() || population_.get_golden_hamiltonian().hasPhysicalNLPP()) &&
            ((active_linear != active_variables && !mixed_lm_ecp && !joint_ci_orbital_ecp) ||
             (conventional_binding_->hasOrbitalRotations() && !joint_ci_orbital_ecp) || multideterminants.size() != 1))
          throw UniformCommunicateError("Conventional locality-ECP LM/ALM require pure coefficients, qualified fixed-orbital DETS/CSF J2 or joint DETS/CSF orbital/J2");
        // The frozen CSF reference precedes its fixed determinant expansion.
        const auto reference = multideterminants[0].get().fixedCoefficientReference();
        if (reference == QMCTraits::ValueType{} || !wftrain::isFiniteTrainingReal(std::real(reference)) ||
            !wftrain::isFiniteTrainingReal(std::imag(reference)))
          throw UniformCommunicateError("Conventional CI updates require a finite nonzero frozen first coefficient");
      }
      // Fail before population initialization/sampling for unqualified fixed assets.
      conventional_binding_->modelIdentity();
      app_log() << "WFTrain conventional derivative scratch cap="
                << training_input_.getLegacyDerivativeScratchBytes()
                << " scope='adapter rows/channels only; excludes component-owned orbital/J3 P*Ne caches, "
                   "private clones, retained configurations'" << std::endl;
      providers.emplace_back(*conventional_binding_);
    }
#ifdef QMC_COMPLEX
    if (complex_training_component_ && complex_training_component_->gpuInferenceEnabled())
    {
      const auto q = complex_training_component_->gpuParameterPublicationRequirements();
      complex_gpu_provider_ = std::make_unique<ComplexGpuTrainingProvider>(*complex_training_component_,
                                                                         q.device_bytes, q.host_bytes);
      providers.emplace_back(*complex_gpu_provider_);
    }
#endif
    const std::string& requested_provider = training_input_.getProviderID();
    if (requested_provider.empty())
    {
      if (providers.size() != 1)
        throw UniformCommunicateError(
            "WFTrainBatched: an omitted provider requires exactly one structured parameter provider");
      selected_provider_id_ = providers.front().get().parameterSchema().providerId();
      selected_provider_    = &providers.front().get();
    }
    else
    {
      for (const auto& provider : providers)
        if (provider.get().parameterSchema().providerId() == requested_provider)
        {
          selected_provider_id_ = requested_provider;
          selected_provider_    = &provider.get();
          break;
        }
      if (selected_provider_id_.empty())
        throw UniformCommunicateError("WFTrainBatched: no structured parameter provider named " +
                                      requested_provider);
    }

    if (const auto* parent = dynamic_cast<const FrozenReferenceResidualWF*>(selected_provider_))
    {
      using O = WFTrainDriverInput::Optimizer;
      const auto optimizer = training_input_.getOptimizer();
      if (student_prototype_.getOrbitals().size() != 1 ||
          training_input_.getObjective() != WFTrainDriverInput::Objective::ENERGY ||
          (optimizer != O::ADAM && optimizer != O::KFAC) ||
          !wftrain::usesEnergyScoreEstimator(training_input_, false) ||
          training_input_.getLineSearch() != WFTrainDriverInput::LineSearch::OFF ||
          parent->frozenModelIdentity().empty())
        throw UniformCommunicateError("Frozen reference sum requires a sole CPU ENERGY Adam/KFAC provider, "
            "score covariance, line_search=off and reproducible frozen reference content");
    }

    // Visible device ordinals may differ under rank-local GPU masks. Agree on
    // physical identities before sampling or entering optimizer collectives.
    const bool explicit_kfac_policy = wftrain::hasExplicitKFACPolicy(training_input_);
    const bool score_policy = complex_training_component_ || explicit_kfac_policy;
    bool hybrid_score_kfac = false;
    bool selected_pair_accelerator = false;
#if !defined(QMC_COMPLEX) && !defined(MIXED_PRECISION)
    const auto* real_provider = dynamic_cast<const PsiFormerWF*>(selected_provider_);
    const auto& hamiltonian = population_.get_golden_hamiltonian();
    const auto& provenance = hamiltonian.ecpModelProvenance();
    const bool ae = real_provider && !real_provider->isPseudopotentialSystem() &&
        !hamiltonian.hasPhysicalNLPP() && !provenance;
    const bool scalar_pp = real_provider && real_provider->isPseudopotentialSystem() &&
        hamiltonian.hasPhysicalNLPP() && provenance && provenance->ordinary_locality_vp &&
        !provenance->scientific_definition.empty();
    selected_pair_accelerator = real_provider && selected_provider_->usesAcceleratorParameterProducts() &&
        real_provider->matchesRealDevicePairTarget(population_.get_golden_electrons());
    hybrid_score_kfac = wftrain::supportsHybridKFACPolicy(training_input_) && selected_provider_->usesAcceleratorParameterProducts() &&
        real_provider && student_prototype_.getOrbitals().size() == 1 &&
        (population_.get_golden_electrons().getLattice().SuperCellEnum == SUPERCELL_OPEN ||
         (scalar_pp && real_provider->matchesLegacyTorusPseudopotentialTarget(population_.get_golden_electrons())) ||
         (ae && myComm->size() == 1 && selected_pair_accelerator &&
          vmcdriver_input_.get_move_kind() == VMCDriverInput::MoveKind::PBYP)) &&
        (ae || scalar_pp) &&
        supportsNativeEnergyMove(vmcdriver_input_, *real_provider) && !vmcdriver_input_.get_use_drift();
#elif defined(QMC_COMPLEX) && !defined(MIXED_PRECISION)
    selected_pair_accelerator = complex_gpu_provider_ &&
        complex_training_component_->matchesPairedDevicePairTarget(population_.get_golden_electrons());
    hybrid_score_kfac = complex_gpu_provider_ && wftrain::supportsHybridKFACPolicy(training_input_) &&
        (legacy_paired_score_ecp || (!complex_training_component_->isPseudopotentialSystem() &&
        !population_.get_golden_hamiltonian().hasPhysicalNLPP() &&
        !population_.get_golden_hamiltonian().ecpModelProvenance())) &&
        (myComm->size() == 1 || legacy_paired_mpi2_kfac || legacy_paired_score_ecp) &&
        supportsNativeEnergyMove(vmcdriver_input_, *complex_training_component_) && !vmcdriver_input_.get_use_drift();
#endif
    int policy_failure = (selected_pair_accelerator && !hybrid_score_kfac) ||
        (explicit_kfac_policy && ((selected_provider_->usesAcceleratorParameterProducts() && !hybrid_score_kfac) ||
        (!dynamic_cast<PsiFormerWF*>(selected_provider_) && !jastrow_binding_ &&
         !dynamic_cast<FrozenReferenceResidualWF*>(selected_provider_) && !complex_training_component_) ||
        training_input_.getLineSearch() != WFTrainDriverInput::LineSearch::OFF));
    int hybrid_ranks = hybrid_score_kfac ? 1 : 0;
    // Every rank agrees admission before placement and later optimizer collectives.
    myComm->allreduce(policy_failure);
    myComm->allreduce(hybrid_ranks);
    if (policy_failure || (hybrid_ranks != 0 && hybrid_ranks != myComm->size()))
      throw UniformCommunicateError("Explicit KFAC score policies require a native CPU provider or a qualified sole AE/PP FP64 "
          "GPU PBYP or supported all-electron/no-drift covariance profile with finite retained cap on every rank; line_search=off is required");
    if (score_policy && training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::KFAC &&
        training_input_.getKFACObjective() == WFTrainDriverInput::KFACObjective::CLIPPED_COVARIANCE &&
        (training_input_.getKFACClipMode() != WFTrainDriverInput::KFACClipMode::SOFT_MEDIAN ||
         !wftrain::isPositiveInfinityTrainingReal(training_input_.getKFACExcludeWidth())))
      throw UniformCommunicateError("Score-only KFAC soft weighting does not support hard windows or exclusion");
    wftrain::agreeAcceleratorRuntimePlacement(*myComm, selected_provider_->usesAcceleratorParameterProducts());

    const AdjustedWalkerCounts walker_counts = adjustGlobalWalkerCount(
        *myComm, walker_configs_ref_.getActiveWalkers(), qmcdriver_input_.get_total_walkers(),
        qmcdriver_input_.get_walkers_per_rank(), 1.0,
        determineNumCrowds(qmcdriver_input_.get_num_crowds(), rngs_.size()));
    if (qmcdriver_input_.get_requested_samples() != 0)
      throw UniformCommunicateError(
          "WFTrainBatched: samples is unsupported; use walkers, blocks and steps per iteration");
    if (training_input_.getIterations() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        training_input_.getIterations() > static_cast<std::size_t>(std::numeric_limits<int>::max()) /
        static_cast<std::size_t>(std::max(1, qmcdriver_input_.get_max_blocks())))
      throw UniformCommunicateError("WFTrainBatched: iterations times blocks exceeds the driver block range");
    steps_per_block_ = determineStepsPerBlock(walker_counts.global_walkers, qmcdriver_input_.get_requested_samples(),
                                              qmcdriver_input_.get_requested_steps(),
                                              qmcdriver_input_.get_max_blocks());
    // Target spatial tapes do not replace score/complete local-energy products.
    // Bound native energy training independently of the spatial owner budget.
    const auto* target_provider = dynamic_cast<const PsiFormerWF*>(selected_provider_);
    const bool target_training = target_provider && target_provider->usesCpuTargetVGL();
    int target_ranks = target_training ? 1 : 0;
    myComm->allreduce(target_ranks);
    if (target_ranks != 0 && target_ranks != myComm->size())
      throw UniformCommunicateError("CPU target VGL training must select the same owner profile on every rank");
    if (target_training)
    {
      const bool admitted_optimizer = training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::ADAM ||
          (training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::KFAC &&
           wftrain::supportsHybridKFACPolicy(training_input_));
      const auto& hamiltonian = population_.get_golden_hamiltonian();
      const auto& provenance = hamiltonian.ecpModelProvenance();
      const bool ae = !target_provider->isPseudopotentialSystem() && !hamiltonian.hasPhysicalNLPP() && !provenance;
      const bool pp = target_provider->isPseudopotentialSystem() && hamiltonian.hasPhysicalNLPP() &&
          provenance && provenance->ordinary_locality_vp && !provenance->scientific_definition.empty();
      const bool extended = myComm->size() != 1 || walker_counts.walkers_per_crowd.size() != 1 || !ae ||
          population_.get_golden_electrons().getLattice().SuperCellEnum != SUPERCELL_OPEN;
      const auto cap = training_input_.getMaximumRetainedBytes();
      int invalid_target = !admitted_optimizer ||
          training_input_.getObjective() != WFTrainDriverInput::Objective::ENERGY ||
          student_prototype_.getOrbitals().size() != 1 || steps_per_block_ != 1 || (!ae && !pp) ||
          !std::all_of(walker_counts.walkers_per_crowd.begin(), walker_counts.walkers_per_crowd.end(),
                       [](auto count) { return count > 0; }) ||
          (extended && (training_input_.getLineSearch() != WFTrainDriverInput::LineSearch::OFF ||
                        !cap || cap == std::numeric_limits<std::size_t>::max())) ||
          !supportsNativeEnergyMove(vmcdriver_input_, *target_provider) || vmcdriver_input_.get_use_drift();
      myComm->allreduce(invalid_target);
      if (invalid_target)
        throw UniformCommunicateError("CPU target VGL training requires sole REAL FP64 OPEN or legacy torus Gamma "
            "AE/ordinary-VP PP energy Adam or explicit-policy KFAC, nonempty crowds/step1 and PBYP or supported all-electron/no drift; "
            "extended rank/crowd/PP profiles require finite retained cap and line_search=off on every rank");
    }
    // A request receipt precedes population allocation. Admission below can still
    // reject it; do not present requested crowds as concurrent derivative lanes.
    if (selected_provider_->usesAcceleratorParameterProducts())
    {
      const auto* real_request = dynamic_cast<const PsiFormerWF*>(selected_provider_);
      bool mixed_request = real_request && real_request->usesMixedAffineTraining();
#ifdef QMC_COMPLEX
      mixed_request = mixed_request ||
          (complex_training_component_ && complex_training_component_->usesMixedAffineTraining());
#endif
      app_log() << "WFTrain execution request: schema=" << selected_provider_->parameterSchema().fingerprint()
                << " scalar=" << (complex_training_component_ ? "paired_complex" : "real")
                << " geometry=" << (population_.get_golden_electrons().getLattice().SuperCellEnum == SUPERCELL_OPEN
                    ? "open" : "periodic")
                << " physical_ecp=" << population_.get_golden_hamiltonian().hasPhysicalNLPP()
                << " score_only=" << wftrain::usesEnergyScoreEstimator(training_input_, bool(complex_training_component_))
                << " ranks=" << myComm->size() << " global_walkers=" << walker_counts.global_walkers
                << " nominal_crowds=" << walker_counts.walkers_per_crowd.size()
                << " nonempty_crowds=" << std::count_if(walker_counts.walkers_per_crowd.begin(),
                    walker_counts.walkers_per_crowd.end(), [](auto n) { return n > 0; })
                << " precision=" << psiformer::psiFormerPrecisionPolicyName(mixed_request
                    ? psiformer::PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS
                    : psiformer::PsiFormerPrecisionPolicy::FP64_REFERENCE)
                << " admission=pending" << std::endl;
    }
    initPopulationAndCrowds(walker_counts);
    // Only physical ECP score Adam/KFAC may use the new independent C2 owners.
    // All other operations retain their original topology restrictions.
    bool ecp_score_c2 = false;
    int score_lane_failure = 0;
    if (selected_provider_->usesAcceleratorParameterProducts() &&
        population_.get_golden_hamiltonian().hasPhysicalNLPP() &&
        wftrain::usesEnergyScoreEstimator(training_input_, bool(complex_training_component_)) &&
        training_input_.getLineSearch() == WFTrainDriverInput::LineSearch::OFF &&
        (training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::ADAM || hybrid_score_kfac))
    {
      try
      {
        std::size_t lanes = 0;
        if (const auto* real = dynamic_cast<const PsiFormerWF*>(selected_provider_))
          lanes = real->activeAcceleratorInferenceCrowdCapacity();
#ifdef QMC_COMPLEX
        else if (complex_gpu_provider_ && legacy_paired_score_ecp)
          lanes = complex_training_component_->activeAcceleratorInferenceCrowdCapacity();
#endif
        const bool nonempty = std::all_of(crowds_.begin(), crowds_.end(),
            [](const auto& crowd) { return crowd->size() != 0; });
        ecp_score_c2 = wftrain::supportsGpuScoreECPCrowds(crowds_.size(), nonempty, lanes);
      }
      catch (...) { score_lane_failure = 1; }
    }
    myComm->allreduce(score_lane_failure);
    if (score_lane_failure)
      throw UniformCommunicateError("GPU ECP score lane capacity discovery failed before sampling");
    // Independent score selection must retain the qualified scalar ECP topology.
    int invalid_score_ecp = 0;
    if (!complex_training_component_ && wftrain::usesEnergyScoreEstimator(training_input_, false) &&
        selected_provider_->usesAcceleratorParameterProducts() &&
        population_.get_golden_hamiltonian().hasPhysicalNLPP())
    {
      const auto* real = dynamic_cast<const PsiFormerWF*>(selected_provider_);
      const auto& metadata = population_.get_golden_hamiltonian().ecpModelProvenance();
      const auto cap = training_input_.getMaximumRetainedBytes();
      invalid_score_ecp = !real || !real->isPseudopotentialSystem() ||
          student_prototype_.getOrbitals().size() != 1 || !metadata || !metadata->ordinary_locality_vp ||
          metadata->scientific_definition.empty() || !cap || cap == std::numeric_limits<std::size_t>::max() ||
          (crowds_.size() != 1 && !ecp_score_c2) || crowds_.front()->size() == 0 || qmcdriver_input_.get_max_blocks() != 1 ||
          steps_per_block_ != 1 ||
          (population_.get_golden_electrons().getLattice().SuperCellEnum != SUPERCELL_OPEN &&
           !real->matchesLegacyTorusPseudopotentialTarget(population_.get_golden_electrons())) ||
          !supportsNativeEnergyMove(vmcdriver_input_, *real) || vmcdriver_input_.get_use_drift();
    }
    myComm->allreduce(invalid_score_ecp);
    if (invalid_score_ecp)
      throw UniformCommunicateError("GPU score-only ECP requires a sole physical OPEN or legacy torus Gamma PP provider, finite cap, nonempty C1 or independent C2/block1/step1 and PBYP or supported all-electron/no drift on every rank");
    if (hybrid_score_kfac)
    {
      bool real_open_ae = false;
      std::size_t active_gpu_capacity = 0;
      int capacity_failure = 0;
#if !defined(QMC_COMPLEX) && !defined(MIXED_PRECISION)
      real_open_ae = ae && real_provider &&
          population_.get_golden_electrons().getLattice().SuperCellEnum == SUPERCELL_OPEN;
      if (real_open_ae && (myComm->size() == 1 || myComm->size() == 2) &&
          (crowds_.size() == 2 || crowds_.size() == 4))
      {
        // Vote a local validation failure before any rank leaves population admission.
        try
        {
          active_gpu_capacity = real_provider->activeAcceleratorInferenceCrowdCapacity();
        }
        catch (...)
        {
          capacity_failure = 1;
        }
      }
#endif
      const bool all_nonempty = std::all_of(crowds_.begin(), crowds_.end(),
          [](const auto& crowd) { return crowd->size() != 0; });
      int invalid_population = capacity_failure || (!ecp_score_c2 && !wftrain::supportsHybridKFACPopulation(real_open_ae,
          static_cast<std::size_t>(myComm->size()), crowds_.size(), all_nonempty, active_gpu_capacity));
      myComm->allreduce(invalid_population);
      if (invalid_population)
        throw UniformCommunicateError("Hybrid score KFAC requires nonempty crowds; C2/C4 require "
            "REAL OPEN AE FP64 MPI1/MPI2 and one prepared GPU inference lease per rank, "
            "or ECP C2 with two independent score lanes before sampling");
      if (population_.get_golden_hamiltonian().hasPhysicalNLPP() &&
          (qmcdriver_input_.get_max_blocks() != 1 || steps_per_block_ != 1))
        throw UniformCommunicateError("GPU score-only ECP requires block1/step1 before sampling");
      if (selected_pair_accelerator &&
          (myComm->size() != 1 || qmcdriver_input_.get_max_blocks() != 1 || steps_per_block_ != 1))
        throw UniformCommunicateError("Selected pair GPU KFAC requires MPI1/block1/step1 before sampling");
      app_log() << "WFTrain hybrid score KFAC: evaluation/score VJPs/publication=GPU, affine observations/factors/inverse/state=HOST; "
                   "parameter authority=HOST, local energy=native Hamiltonian, reduction=HOST raw sums, ranks="
                << myComm->size() << std::endl;
    }
    // Each statistical block retains its own samples. C1 shares its Products
    // lane across snapshots; ECP C2 has one independent score owner per crowd.
    if (complex_gpu_provider_ && ((crowds_.size() != 1 && !ecp_score_c2) || steps_per_block_ != 1))
      throw UniformCommunicateError("WFTrainBatched: paired GPU requires C1 or independent ECP score C2, steps1 before sampling");
    if (complex_gpu_provider_ && population_.get_golden_hamiltonian().hasPhysicalNLPP() &&
        qmcdriver_input_.get_max_blocks() != 1)
      throw UniformCommunicateError("WFTrainBatched: paired ECP requires one block before sampling");
    if (selected_provider_->usesAcceleratorParameterProducts() &&
        wftrain::usesEnergyScoreEstimator(training_input_, bool(complex_training_component_)))
      app_log() << "WFTrain effective score execution: derivative_lanes=" << (ecp_score_c2 ? 2 : 1)
                << " derivative_dispatch=serialized_batches independent_queues=" << ecp_score_c2
                << " retained_crowds=" << crowds_.size()
                << " statistics=rank_global_then_mpi updates_per_iteration=1"
                << " local_energy=native parameter_authority=host"
                << " residency=host_mediated fallback_reason="
                << (ecp_score_c2 ? "host_batch_dispatch" : "shared_derivative_lane") << std::endl;
    const auto* device_provider = dynamic_cast<const PsiFormerWF*>(selected_provider_);
    resident_adam_budget_ = device_provider ? device_provider->residentEnergyAdamBudget() : 0;
    resident_score_adam_ = resident_adam_budget_ &&
        training_input_.getGradientEstimator() == WFTrainDriverInput::GradientEstimator::SCORE_COVARIANCE;
    resident_sr_budget_ = device_provider ? device_provider->residentEnergySRBudget() : std::pair<std::size_t, std::size_t>{0, 0};
    resident_device_authority_ = device_provider && device_provider->residentDeviceAuthorityRequested();
    if (std::getenv("PSIFORMER_GPU_RESIDENT_ECP_OBSERVATION_BYTES") && !resident_score_adam_)
      throw UniformCommunicateError("Resident ECP physical observation requires explicit score Adam C2 bridge before sampling");
    if (resident_score_adam_ && resident_device_authority_)
      throw UniformCommunicateError("Score Adam C2 bridge requires HOST parameter authority until physical PP DEVICE observations are implemented");
    if (const auto* optin = std::getenv("PSIFORMER_GPU_RESIDENT_OPEN_OBSERVATION"))
    {
      if (std::string(optin) != "1")
        throw UniformCommunicateError("PSIFORMER_GPU_RESIDENT_OPEN_OBSERVATION requires explicit 1 or must be absent");
      resident_open_observation_ = true;
      if (!resident_device_authority_ || !resident_adam_budget_ || resident_sr_budget_.first ||
          crowds_.size() != 1 || crowds_.front()->size() != 2 ||
          population_.get_golden_hamiltonian().hasPhysicalNLPP() || walker_logs_input.present ||
          qmcdriver_input_.get_estimator_measurement_period() != 1)
        throw UniformCommunicateError("Resident open observation requires sole AE W2 DEVICE-authority Adam and no walker logs");
      auto* provider = dynamic_cast<PsiFormerWF*>(selected_provider_);
      resident_open_scalar_slots_ = residentOpenScalarSlots(population_.get_golden_hamiltonian(),
          population_.get_golden_electrons(), *provider, resident_open_ions_);
      auto& estimators = crowds_.front()->get_estimator_manager_crowd();
      if (!dynamic_cast<LocalEnergyEstimator*>(&estimators.get_main_estimator()) ||
          !estimators.get_scalar_estimators().empty() || !estimators.get_operator_estimators().empty())
        throw UniformCommunicateError("Resident open observation admits only the standard local-energy scalar estimator");
    }
    if (resident_device_authority_ &&
        ((!resident_adam_budget_ && !resident_sr_budget_.first) || !device_provider->deviceSamplingEnabled() ||
         (vmcdriver_input_.get_move_kind() != VMCDriverInput::MoveKind::ALL_ELECTRON &&
          vmcdriver_input_.get_move_kind() != VMCDriverInput::MoveKind::N_ELECTRON) ||
         vmcdriver_input_.get_use_drift() || serializing_crowd_walkers_ || inference_input_.enabled ||
         qmcdriver_input_.get_debug_checks() != DriverDebugChecks::ALL_OFF))
      throw UniformCommunicateError("DEVICE authority requires explicit resident Adam or SR, DEVICE joint alle/n_electron moves, no drift, batched refresh and no debug GL/inference");
    if (resident_sr_budget_.first)
    {
      if (resident_adam_budget_ || device_provider->deviceAdamBudget() ||
          training_input_.getOptimizer() != WFTrainDriverInput::Optimizer::STOCHASTIC_RECONFIGURATION ||
          !wftrain::isFiniteTrainingReal(training_input_.getDamping()) || training_input_.getDamping() <= 0 ||
          !wftrain::isFiniteTrainingReal(training_input_.getLearningRate()) || training_input_.getLearningRate() <= 0 ||
          training_input_.getMaximumDampingAttempts() != 1)
        throw UniformCommunicateError("Resident ENERGY SR requires exclusive fixed positive damping/LR and one attempt");
      wftrain::krylov_detail::validateControl(makeKrylovControl(training_input_));
      if (resident_device_authority_)
      {
        const char* reuse = std::getenv("PSIFORMER_GPU_RESIDENT_ENERGY_SR_REUSE");
        if (!reuse || std::string(reuse) != "on")
          throw UniformCommunicateError("DEVICE SR authority requires explicit retained solver reuse=on");
      }
    }
    if (resident_adam_budget_ || resident_sr_budget_.first)
    {
      if (myComm->size() != 1 || crowds_.size() != 1 || crowds_.front()->size() < (resident_device_authority_ ? 1 : 7) ||
          qmcdriver_input_.get_max_blocks() != 1 || steps_per_block_ != 1 ||
          student_prototype_.getOrbitals().size() != 1 ||
          training_input_.getObjective() != WFTrainDriverInput::Objective::ENERGY ||
          (resident_adam_budget_ && training_input_.getOptimizer() != WFTrainDriverInput::Optimizer::ADAM) ||
          training_input_.getLineSearch() != WFTrainDriverInput::LineSearch::OFF ||
          training_input_.getUpdateScale() != 1 || training_input_.getWeightDecay() != 0 ||
          training_input_.getMaximumUpdateNorm() != 0 || training_input_.getMaximumMetricNorm() != 0 ||
          (resident_adam_budget_ && !device_provider->deviceAdamBudget()))
        throw UniformCommunicateError("Resident ENERGY requires sole real/open/FP64 GPU, rank/crowd/blocks/steps=1, B>=7 (DEVICE authority B>0), fixed unmodified step and explicit caps");
      if (resident_score_adam_)
      {
        if (!population_.get_golden_hamiltonian().hasPhysicalNLPP())
          throw UniformCommunicateError("Score Adam C2 driver bridge currently requires an actual open ECP Hamiltonian");
        auto& provider = *dynamic_cast<PsiFormerWF*>(selected_provider_);
        (void)wftrain::realECPPhysicalIdentity(provider, population_.get_golden_electrons(), population_.get_golden_hamiltonian());
      }
      else
        validateResidentCoulomb(population_.get_golden_hamiltonian(), resident_device_authority_);
      if (resident_adam_budget_)
        resident_norm_bytes_ = resident_score_adam_
            ? device_provider->prepareResidentScoreAdam(resident_adam_budget_)
            : device_provider->prepareResidentEnergyAdam(resident_adam_budget_);
      else
        resident_sr_incremental_bytes_ = device_provider->prepareResidentEnergySR(crowds_.front()->size(), resident_sr_budget_);
      if (resident_score_adam_)
      {
        auto& provider = *dynamic_cast<PsiFormerWF*>(selected_provider_);
        const auto physical = wftrain::realECPPhysicalIdentity(provider,
            population_.get_golden_electrons(), population_.get_golden_hamiltonian());
        const auto observation_bytes = provider.prepareResidentECPObservation(crowds_.front()->size(), physical.fingerprint);
        enforceMemoryBudget(*myComm, observation_bytes, training_input_.getMaximumRetainedBytes());
      }
      // Authority activation borrows the admitted C3 lane; it allocates no new image.
      if (resident_device_authority_)
      {
        std::size_t resident_open_bytes = 0;
        if (resident_open_observation_)
        {
          const auto cost = device_provider->residentOpenObservationRequirements(2);
          resident_open_bytes = checkedMemoryAdd(cost.device_bytes,
              checkedMemoryAdd(cost.host_bytes, cost.receipt_host_bytes));
          enforceMemoryBudget(*myComm, resident_open_bytes, training_input_.getMaximumRetainedBytes());
          device_provider->prepareResidentOpenObservation(population_.get_golden_electrons(), *resident_open_ions_,
              cost.device_bytes, cost.host_bytes);
        }
        const auto bytes = device_provider->prepareResidentDeviceAuthority();
        resident_snapshot_cap_ = bytes.snapshot_device_bytes;
        resident_successor_cap_ = bytes.successor_device_bytes;
        const auto cold_moments = resident_adam_budget_ ? checkedMemoryMultiply(
            selected_provider_->parameterSchema().parameterCount(), 2 * sizeof(double)) : 0;
        resident_authority_extra_bytes_ = checkedMemoryAdd(resident_open_bytes, checkedMemoryAdd(bytes.cold_host_bytes,
            checkedMemoryAdd(cold_moments, checkedMemoryAdd(resident_snapshot_cap_, resident_successor_cap_))));
        enforceMemoryBudget(*myComm, checkedMemoryAdd(resident_authority_extra_bytes_, resident_sr_incremental_bytes_),
            training_input_.getMaximumRetainedBytes());
        app_log() << "Resident DEVICE authority additional admitted bytes: snapshot DEVICE=" << resident_snapshot_cap_
                  << ", successor DEVICE=" << resident_successor_cap_ << ", cold HOST mirror=" << bytes.cold_host_bytes
                  << ", cold HOST moments=" << cold_moments
                  << "; default HOST path unchanged; warm publication uses metadata only" << std::endl;
      }
    }
    // ContextForSteps only borrows an RNG reference; all MC draws follow admission.
    createStepContexts(crowds_.size());
    if (conventional_binding_)
    {
      conventional_binding_->attachLive(population_.get_golden_twf(), population_.get_golden_electrons(),
                                        &population_.get_golden_hamiltonian());
      for (std::size_t i = 0; i < population_.get_twfs().size(); ++i)
      {
        // These TWFs belong exclusively to this driver's MCPopulation. Never
        // enable optimizer-private VP evaluation on the external golden/pool TWF.
        auto& walker_wf = *population_.get_twfs()[i];
        if (&walker_wf == &student_prototype_ || &walker_wf == &population_.get_golden_twf())
          throw std::logic_error("Conventional population unexpectedly aliases its external prototype");
        if (population_.get_golden_hamiltonian().hasPhysicalNLPP())
          for (auto& determinant : walker_wf.findMSD())
            if (conventional_binding_->hasOrbitalRotations())
              determinant.get().enableAcceptedReferenceOrbitalVirtualRatios();
            else
              determinant.get().enableAcceptedReferenceVirtualRatios();
        conventional_binding_->attachLive(*population_.get_twfs()[i], *population_.get_elec_particle_sets()[i],
                                          population_.get_hamiltonians()[i].get());
      }
    }
    if (jastrow_binding_)
    {
      jastrow_binding_->attachLive(population_.get_golden_twf(), population_.get_golden_electrons(),
          &population_.get_golden_hamiltonian());
      for (std::size_t i = 0; i < population_.get_twfs().size(); ++i)
        jastrow_binding_->attachLive(*population_.get_twfs()[i], *population_.get_elec_particle_sets()[i],
            population_.get_hamiltonians()[i].get());
    }
    initializeDeterminantOverlap();
    // Target construction clones SPOs and may allocate differently on each rank.
    // Agree on failure before any participant enters the training collectives.
    std::exception_ptr target_failure;
    try
    {
      initializeOrbitalPretraining();
      if (inference_input_.enabled && training_input_.getObjective() == WFTrainDriverInput::Objective::ORBITAL_MSE)
      {
        bool admitted = orbital_target_ && (dynamic_cast<PsiFormerWF*>(selected_provider_) || jastrow_binding_);
#ifdef QMC_COMPLEX
        admitted = admitted || (complex_orbital_target_ && complex_training_component_ &&
                                 !complex_training_component_->gpuInferenceEnabled());
#endif
        const bool gpu = selected_provider_->usesAcceleratorParameterProducts();
        if (!admitted || (gpu && (!dynamic_cast<PsiFormerWF*>(selected_provider_) ||
                                 !wftrain::supportsRealGpuOrbitalInference(training_input_))))
          throw UniformCommunicateError("Orbital inference requires a matching CPU provider or bounded real GPU "
              "teacher-measure single-reference Adam/spin-sector loss and frozen target");
        if (gpu)
        {
          const auto& descriptor = orbital_target_->descriptor();
          const auto n = checkedMemoryAdd(descriptor.spinUpElectrons(), descriptor.spinDownElectrons());
          const auto count = checkedMemoryMultiply(descriptor.terms().size(), checkedMemoryMultiply(n, n));
          const auto output_bytes = checkedMemoryMultiply(count, sizeof(double));
          if (checkedMemoryAdd(orbital_target_bytes_, output_bytes) > training_input_.getMaximumRetainedBytes())
            throw std::length_error("GPU orbital inference target and output exceed maximum_retained_bytes");
          app_log() << "WFTrain GPU orbital inference: caller output numeric_bytes=" << output_bytes
                    << "; existing product lane reused, no reverse/parameter response download; "
                       "excludes inference clones/model/vendor/RSS" << std::endl;
        }
      }
    }
    catch (...)
    {
      target_failure = std::current_exception();
    }
    int target_failed = target_failure ? 1 : 0;
    myComm->allreduce(target_failed);
    if (target_failed)
    {
      if (target_failure)
        std::rethrow_exception(target_failure);
      throw UniformCommunicateError("WFTrainBatched: another rank failed to construct its orbital target");
    }
  }
  catch (const std::exception& error)
  {
    myComm->barrier_and_abort(error.what());
  }

  if (qmcdriver_input_.get_measure_imbalance())
    measureImbalance("Startup");
}

void WFTrainBatched::run()
{
  if (selected_provider_ == nullptr)
    throw UniformCommunicateError("WFTrainBatched: process() must select a provider before run()");

  // Declare before every iteration owner: snapshots and callbacks die before these driver-phase anchors.
  std::vector<std::shared_ptr<PsiFormerGpuScoreLaneAnchor>> score_lane_anchors;
  std::vector<std::shared_ptr<PsiFormerComplexGpuScoreLaneAnchor>> complex_score_lane_anchors;
  auto* device_provider = dynamic_cast<PsiFormerWF*>(selected_provider_);
  bool native_mixed_training = device_provider && device_provider->usesMixedAffineTraining();
#ifdef QMC_COMPLEX
  native_mixed_training = native_mixed_training ||
      (complex_training_component_ && complex_training_component_->usesMixedAffineTraining());
#endif
  int mixed_participants = native_mixed_training ? 1 : 0;
  myComm->allreduce(mixed_participants);
  if (mixed_participants && mixed_participants != myComm->size())
    throw UniformCommunicateError("WFTrainBatched: mixed native training policy differs across ranks");
  const auto resident_adam_budget = resident_adam_budget_, resident_norm_bytes = resident_norm_bytes_;
  const bool resident_score_adam = resident_score_adam_;
  const auto cold_moment_allowance = resident_device_authority_ && resident_adam_budget
      ? checkedMemoryMultiply(selected_provider_->parameterSchema().parameterCount(), 2 * sizeof(double)) : 0;
  const auto resident_sr_budget = resident_sr_budget_;
  /// Destroy after iteration sources/leases, on normal and exceptional run exit.
  struct ResidentSRRelease
  {
    const PsiFormerWF* provider;
    ~ResidentSRRelease() noexcept
    {
      if (provider)
        provider->releaseResidentEnergySRWorkspace();
    }
  } resident_sr_release{resident_sr_budget.first ? device_provider : nullptr};
  const IndexType num_blocks = qmcdriver_input_.get_max_blocks();
  const int num_iterations = static_cast<int>(training_input_.getIterations());
  bool complex_training = false;
  std::size_t complex_affine_score_bytes = 0;
  bool complex_device_affine = false;
  std::size_t complex_device_affine_bytes = 0;
  std::size_t complex_affine_device_bytes = 0, complex_affine_host_bytes = 0;
  std::size_t complex_expected_prepared_bytes = 0;
  std::size_t complex_handoff_preflight_bytes = 0;
  std::size_t complex_fixed_grid_host_bytes = 0;
  std::string complex_physical_identity;
#ifdef QMC_COMPLEX
  if (const auto* complex = complex_training_component_)
  {
    complex_training = true;
    std::exception_ptr budget_failure;
    std::size_t prospective = 0, allowance = 0;
    try
    {
      // One snapshot per walker/block/STEP, not per electron or VMC substep.
      std::size_t walkers = 0, largest_batch = 0;
      for (const auto& crowd : crowds_)
      {
        walkers = checkedMemoryAdd(walkers, crowd->size());
        largest_batch = std::max(largest_batch, static_cast<std::size_t>(crowd->size()));
      }
      if (!walkers || !num_blocks || !steps_per_block_)
        throw std::invalid_argument("WFTrainBatched: paired training requires samples on every rank");
      const auto samples = checkedMemoryMultiply(walkers,
          checkedMemoryMultiply(static_cast<std::size_t>(num_blocks), steps_per_block_));
      const auto& target = population_.get_golden_electrons();
      const auto n = static_cast<std::size_t>(target.getTotalNum());
      const auto p = selected_provider_->parameterSchema().parameterCount();
      const bool ecp = population_.get_golden_hamiltonian().hasPhysicalNLPP();
      if (complex_gpu_provider_ && training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::KFAC &&
          complex->usesDeviceAffineStatistics())
      {
        if (!ecp || !wftrain::usesEnergyScoreEstimator(training_input_, true) ||
            training_input_.getLineSearch() != WFTrainDriverInput::LineSearch::OFF)
          throw std::invalid_argument("Paired DEVICE affine KFAC requires legacy ECP score curvature and line search OFF");
        complex_device_affine = true;
        const auto requirements = complex->deviceAffineStatisticsRequirements(makeLiveKFACRegistry(*selected_provider_));
        complex_affine_device_bytes = requirements.device_bytes;
        complex_affine_host_bytes = requirements.host_bytes;
        complex_device_affine_bytes = checkedMemoryAdd(requirements.device_bytes, requirements.host_bytes);
      }
      const bool inline_ecp = ecp && usesInlinePairedECPResponse(training_input_);
      const auto prepared_batch_bound = checkedMemoryMultiply(crowds_.size(),
          checkedMemoryMultiply(static_cast<std::size_t>(num_blocks), steps_per_block_));
      const auto additive_bytes = inline_ecp ? checkedMemoryMultiply(prepared_batch_bound,
          checkedMemoryMultiply(p, sizeof(wftrain::DerivativeValue))) : 0;
      const auto inline_capture_bytes = inline_ecp ? checkedMemoryAdd(complex->nonLocalVJPCaptureRequiredBytes(largest_batch),
          checkedMemoryMultiply(largest_batch, sizeof(wftrain::DerivativeValue))) : 0;
      // R(S) counts the weak-shared family lane ONCE and all disjoint immutable coordinates/masses.
      std::size_t derivative_bytes = 0, successor_bytes = 0;
      if (training_input_.getObjective() == WFTrainDriverInput::Objective::ORBITAL_MSE)
      {
        derivative_bytes = complex->orbitalMSEStorage().prepared_bytes;
        if (complex_gpu_provider_)
        {
          const auto q = complex->gpuParameterPublicationRequirements();
          successor_bytes = checkedMemoryAdd(q.device_bytes, q.host_bytes);
        }
      }
      else if (complex_gpu_provider_)
      {
        const auto& limits = training_input_.getComplexGpuTraining();
        const bool scalar_ecp = ecp && wftrain::usesEnergyScoreEstimator(training_input_, true);
        const auto scope = scalar_ecp ? PsiFormerComplexWF::GpuDerivativeScope::SCORE_ECP_ONLY :
            PsiFormerComplexWF::GpuDerivativeScope::COMPLETE;
        const auto points = ecp && !scalar_ecp && !inline_ecp ?
            pairedECPPointCapacity(population_.get_golden_hamiltonian(), target, largest_batch) : 0;
        auto r = complex->gpuDerivativeRequirements(samples, checkedMemoryAdd(largest_batch, points), limits.kinetic_targets,
            std::min(p, training_input_.getParameterChunkSize()), scope);
        if (scalar_ecp && crowds_.size() == 2 && complex->activeAcceleratorInferenceCrowdCapacity() == 2)
        {
          // C2 owns one score tape/queue per crowd. Sum exact crowd capacities,
          // including uneven tails, before sampling; immutable weights stay shared.
          r = {};
          for (const auto& crowd : crowds_)
          {
            const auto lane = complex->gpuDerivativeRequirements(crowd->size(), crowd->size(),
                limits.kinetic_targets, std::min(p, training_input_.getParameterChunkSize()), scope);
            r.device_bytes = checkedMemoryAdd(r.device_bytes, lane.device_bytes);
            r.host_bytes = checkedMemoryAdd(r.host_bytes, lane.host_bytes);
            r.matrix_work = std::max(r.matrix_work, lane.matrix_work);
          }
        }
        const auto replay_bytes = ecp && !scalar_ecp && !inline_ecp ?
            complex->gpuNonLocalReplayRequirements(largest_batch, points).bound_host_extra_bytes : 0;
        if (ecp && training_input_.getLineSearch() == WFTrainDriverInput::LineSearch::REFERENCE)
        {
          if (largest_batch < 2)
            throw std::invalid_argument("Paired ECP reference selection requires at least two walker clusters");
          const auto native = population_.get_golden_hamiltonian().nativeCaptureRequirements(largest_batch);
          const auto reference = complex->fixedGridEnergyReferenceRequirements();
          // Replay arrays are already in derivative admission; only these NEW owners join its ledger.
          complex_fixed_grid_host_bytes = checkedMemoryAdd(native.total_host_bytes,
              checkedMemoryAdd(reference.shared_replay_descriptor_bytes,
                  checkedMemoryAdd(sizeof(wftrain::PairedFixedGridNativeEnergy), checkedMemoryMultiply(largest_batch,
                      checkedMemoryAdd(reference.new_reference_host_bytes,
                          sizeof(std::reference_wrapper<const PsiFormerComplexWF>) +
                          sizeof(std::shared_ptr<const PsiFormerComplexWF::FixedGridEnergyReference>))))));
        }
        const auto q = complex->gpuParameterPublicationRequirements();
        auto generation_device = q.device_bytes, generation_host = q.host_bytes;
        if (training_input_.getLineSearch() == WFTrainDriverInput::LineSearch::REFERENCE)
        {
          const auto isolation = complex->gpuCandidateIsolationRequirements();
          // Private current I1 plus next I2 coexist; source I0 is shared, not copied.
          generation_device = checkedMemoryAdd(isolation.device_bytes, checkedMemoryMultiply(2, q.device_bytes));
          // Isolation owns flat/leaves; caller snapshot and prepared native candidate add two P vectors.
          generation_host = checkedMemoryAdd(isolation.host_bytes, checkedMemoryMultiply(2, q.host_bytes));
        }
        const auto device = checkedMemoryAdd(checkedMemoryAdd(r.device_bytes, generation_device),
            complex_affine_device_bytes);
        const auto host = checkedMemoryAdd(checkedMemoryAdd(checkedMemoryAdd(
            checkedMemoryAdd(r.host_bytes, checkedMemoryAdd(additive_bytes, inline_capture_bytes)), replay_bytes),
            generation_host), checkedMemoryAdd(complex_fixed_grid_host_bytes, complex_affine_host_bytes));
        if (device > limits.device_bytes || host > limits.host_bytes || r.matrix_work > limits.maximum_matrix_work)
          throw std::length_error(
              "WFTrainBatched: paired GPU derivative+successor device/HOST/work cap exceeded before sampling");
        derivative_bytes = checkedMemoryAdd(checkedMemoryAdd(r.device_bytes, r.host_bytes), replay_bytes);
        successor_bytes = checkedMemoryAdd(generation_device, generation_host);
        app_log() << "WFTrain paired GPU admitted additional device=" << device << " HOST=" << host
                  << " derivative=" << derivative_bytes << " successor_or_private_selector=" << successor_bytes
                  << " K=" << limits.kinetic_targets << " matrix_work=" << r.matrix_work
                  << "; reference includes private scratch/current+successor images and snapshot/prepared HOST; "
                     "OFF includes one successor/native candidate; excludes source inference generation, "
                     "native/image model HOST and vendor storage" << std::endl;
      }
      else
        derivative_bytes = complex->streamingDerivativeRequiredBytes(samples,
            wftrain::usesEnergyScoreEstimator(training_input_, true) ? PsiFormerComplexWF::CpuDerivativeScope::SCORE_ONLY
                                                                   : PsiFormerComplexWF::CpuDerivativeScope::COMPLETE);
      derivative_bytes = checkedMemoryAdd(derivative_bytes, additive_bytes);
      std::size_t complex_kfac_allowance = 0;
      if (training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::KFAC)
      {
        complex_affine_score_bytes = complex_device_affine ? 0 : complex->affineScoreStorage().required_bytes;
        const auto statistics_bytes = kfacStatisticsBytes(makeLiveKFACRegistry(*selected_provider_));
        // Include committed/pending factor states, inverse caches and solve scratch.
        complex_kfac_allowance = checkedMemoryAdd(checkedMemoryAdd(complex_affine_score_bytes,
            complex_device_affine_bytes), checkedMemoryMultiply(8, statistics_bytes));
        app_log() << "WFTrain complex KFAC before allocation: affine_score_bytes=" << complex_affine_score_bytes
                  << " transactional_factor_bytes=" << statistics_bytes
                  << " factor_inverse_allowance=" << complex_kfac_allowance << std::endl;
        if (complex_device_affine)
          app_log() << "WFTrain paired DEVICE affine numeric device=" << complex_affine_device_bytes
                    << " HOST=" << complex_affine_host_bytes << " combined=" << complex_device_affine_bytes
                    << "; exact per-domain admission, combined once in retained allowance; not aggregate HOST fit; "
                       "actual DEVICE banks use aggregate HBM reservations; separate DEVICE score pass, HOST solve" << std::endl;
      }
      complex_expected_prepared_bytes = checkedMemoryAdd(derivative_bytes,
          training_input_.getObjective() == WFTrainDriverInput::Objective::ORBITAL_MSE ? 0 :
          checkedMemoryMultiply(samples, sizeof(wftrain::DerivativeReal) + sizeof(wftrain::DerivativeValue)));
      const auto configuration_bytes = checkedMemoryMultiply(samples, checkedMemoryAdd(sizeof(double),
          checkedMemoryMultiply(n, sizeof(target.R[0]) + sizeof(target.G[0]) + sizeof(target.L[0]))));
      // Name actual persistent host recurrence storage BEFORE allocating it.
      // Snapshot/selection copies belong to the separate conservative allowance.
      std::size_t optimizer_vectors = 0;
      using O = WFTrainDriverInput::Optimizer;
      switch (training_input_.getOptimizer())
      {
      case O::SGD: break;
      case O::MOMENTUM_SGD:
      case O::RMSPROP: optimizer_vectors = 1; break;
      case O::ADAM: optimizer_vectors = 2; break;
      case O::AMSGRAD: optimizer_vectors = 3; break;
      default: break;
      }
      const auto optimizer_bytes = checkedMemoryMultiply(checkedMemoryMultiply(optimizer_vectors, p), sizeof(double));
      app_log() << "WFTrain paired host recurrence before allocation: vectors=" << optimizer_vectors
                << " named_numeric_bytes=" << optimizer_bytes << std::endl;
      // Explicit conservative allowance, NOT measured retained storage or a hard process-peak claim.
      allowance = checkedMemoryAdd(checkedMemoryMultiply(p,
          12 * sizeof(double) + 3 * sizeof(wftrain::DerivativeValue)),
          checkedMemoryMultiply(checkedMemoryMultiply(2, largest_batch), sizeof(wftrain::DerivativeValue)));
      std::size_t covariance_bytes = 0;
      if (training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::STOCHASTIC_RECONFIGURATION)
      {
        // Existing real-tangent covariance retains two complex P-vectors and copied weights/JVP samples.
        covariance_bytes = checkedMemoryAdd(checkedMemoryMultiply(checkedMemoryMultiply(2, p),
            sizeof(wftrain::DerivativeValue)), checkedMemoryMultiply(samples,
            sizeof(wftrain::DerivativeReal) + sizeof(wftrain::DerivativeValue)));
        allowance = checkedMemoryAdd(allowance, parameterSpaceSRWorkspaceBytes(p));
      }
      prospective = checkedMemoryAdd(checkedMemoryAdd(complex_expected_prepared_bytes, configuration_bytes),
          checkedMemoryAdd(covariance_bytes, checkedMemoryAdd(optimizer_bytes, allowance)));
      prospective = checkedMemoryAdd(prospective, successor_bytes);
      prospective = checkedMemoryAdd(prospective, complex_fixed_grid_host_bytes);
      prospective = checkedMemoryAdd(prospective, complex_kfac_allowance);
      if (inline_ecp)
      {
        prospective = checkedMemoryAdd(prospective, inline_capture_bytes);
        app_log() << "WFTrain inline full-response ECP before allocation: retained_additive_bytes=" << additive_bytes
                  << " temporary_capture_bytes=" << inline_capture_bytes
                  << " virtual_capacity=0; primary response=complete kinetic; native VP order preserved" << std::endl;
      }
      else if ((!complex_gpu_provider_ || training_input_.getObjective() != WFTrainDriverInput::Objective::ORBITAL_MSE) &&
          (!complex_gpu_provider_ || wftrain::usesEnergyScoreEstimator(training_input_, true)) &&
          population_.get_golden_hamiltonian().hasPhysicalNLPP())
      {
        // One temporary B-energy vector and one B-coefficient vector coexist only during accepted-grid capture.
        const auto nonlocal_capture_bytes = checkedMemoryMultiply(2,
            wftrain::NonLocalEnergyAccumulator::requiredBytes(largest_batch));
        prospective = checkedMemoryAdd(prospective, nonlocal_capture_bytes);
        app_log() << "WFTrain scalar nonlocal energy before allocation: temporary_numeric_bytes="
                  << nonlocal_capture_bytes << "; no ECP parameter derivatives or retained quadrature replay" << std::endl;
      }
      if (training_input_.getObjective() == WFTrainDriverInput::Objective::ORBITAL_MSE)
      {
        prospective = checkedMemoryAdd(prospective, orbital_target_bytes_);
        if (inference_input_.enabled)
        {
          const auto n = static_cast<std::size_t>(population_.get_golden_electrons().getTotalNum());
          const auto columns = checkedMemoryMultiply(complex->orbitalMSEChannelCount(), n);
          const auto matrix_bytes = checkedMemoryMultiply(checkedMemoryMultiply(columns, n), sizeof(std::complex<double>));
          const auto statistics_bytes = checkedMemoryMultiply(2 * sizeof(double),
              checkedMemoryAdd(5, checkedMemoryMultiply(4, columns)));
          const auto observation_bytes = checkedMemoryAdd(matrix_bytes, statistics_bytes);
          prospective = checkedMemoryAdd(prospective, observation_bytes);
          app_log() << "WFTrain complex orbital observer before allocation: numeric_bytes=" << observation_bytes
                    << "; existing VALUE workspace, no parameter products" << std::endl;
        }
      }
      complex_physical_identity = complex->trainingConfigurationFingerprint();
      if (complex_gpu_provider_ && complex->matchesPairedDevicePairTarget(population_.get_golden_electrons()))
        complex_physical_identity = wftrain::pairedPairCoulombIdentity(*complex,
            population_.get_golden_electrons(), population_.get_golden_hamiltonian());
      if (const auto& provenance = population_.get_golden_hamiltonian().ecpModelProvenance())
      {
        complex_physical_identity += provenance->scientific_definition;
        const auto& ions = complex->sourceParticles();
        // Equal effective charges do not identify which PP species occupies each site.
        for (int ion = 0; ion < ions.getTotalNum(); ++ion)
        {
          const auto& name = ions.getSpeciesSet().speciesName.at(ions.GroupID[ion]);
          complex_physical_identity += ";ordered_pp_species=" + std::to_string(name.size()) + ":" + name;
        }
        for (int i = 0; i < population_.get_golden_hamiltonian().size(); ++i)
        {
          const auto* term = population_.get_golden_hamiltonian().getComponent(i);
          if (const auto* aa = dynamic_cast<const CoulombPBCAA*>(term))
            complex_physical_identity += aa->ewaldModelDefinition();
          else if (const auto* ab = dynamic_cast<const CoulombPBCAB*>(term))
            complex_physical_identity += ab->ewaldModelDefinition();
        }
      }
      if (((complex_gpu_provider_ && wftrain::usesEnergyScoreEstimator(training_input_, true)) || inline_ecp) &&
          population_.get_golden_hamiltonian().hasPhysicalNLPP())
      {
        if (!complex->matchesLegacyTorusPseudopotentialTarget(target))
          throw std::invalid_argument("Paired GPU score ECP requires matching legacy torus PP metadata");
        complex_physical_identity = periodicScoreECPIdentity(population_.get_golden_electrons(),
            complex->sourceParticles(), population_.get_golden_hamiltonian(),
            complex->trainingConfigurationFingerprint());
      }
    }
    catch (...) { budget_failure = std::current_exception(); }
    int failed = budget_failure ? 1 : 0;
    myComm->allreduce(failed);
    if (failed)
    {
      if (budget_failure)
        std::rethrow_exception(budget_failure);
      throw UniformCommunicateError("WFTrainBatched: another rank failed paired training preallocation admission");
    }
    enforceMemoryBudget(*myComm, prospective, training_input_.getMaximumRetainedBytes());
    // This existing conservative admission includes the paired derivative/capture and optimizer coexistence.
    complex_handoff_preflight_bytes = prospective;
    if (complex_affine_score_bytes)
    {
      std::exception_ptr affine_preparation_failure;
      try
      {
        if (complex_gpu_provider_)
          complex_training_component_->prepareHybridAffineScores(complex_affine_score_bytes);
        else
          complex_training_component_->prepareAffineScores(complex_affine_score_bytes);
      }
      catch (...)
      {
        affine_preparation_failure = std::current_exception();
      }
      // A failed rank-local HOST tape allocation must not strand peers in the
      // subsequent model/gradient/checkpoint collectives.
      int affine_preparation_failed = affine_preparation_failure ? 1 : 0;
      myComm->allreduce(affine_preparation_failed);
      if (affine_preparation_failed)
      {
        if (affine_preparation_failure)
          std::rethrow_exception(affine_preparation_failure);
        throw UniformCommunicateError("WFTrainBatched: another rank failed paired affine preparation");
      }
    }
    app_log() << "WFTrain paired " << (complex_gpu_provider_ ? "GPU" : "CPU")
              << " prospective named numeric bytes plus allowance=" << prospective
              << ", conservative snapshot/solve allowance=" << allowance
              << "; excludes shared model, inference clones, opaque ParticleSet caches, allocator/RSS" << std::endl;
  }
#endif

  // Older real CPU/legacy fingerprints remain unchanged. New GPU score profiles
  // bind their actual operators; paired identity was resolved above. Agree all
  // physical identities before sampling, independently of device ordinal masks.
  std::exception_ptr physical_identity_failure;
  std::vector<std::uint64_t> physical_identity_records;
  std::uint64_t physical_identity_hash = 0;
  try
  {
    physical_identity_records.resize(static_cast<std::size_t>(myComm->size()));
    std::string physical_agreement_identity;
#ifndef QMC_COMPLEX
    std::optional<wftrain::RealECPPhysicalIdentity> real_ecp_identity;
    if (device_provider && device_provider->isPseudopotentialSystem() &&
        training_input_.getObjective() == WFTrainDriverInput::Objective::ENERGY &&
        student_prototype_.getOrbitals().size() == 1)
    {
      real_ecp_identity = wftrain::realECPPhysicalIdentity(*device_provider,
          population_.get_golden_electrons(), population_.get_golden_hamiltonian());
      physical_agreement_identity = real_ecp_identity->fingerprint;
      // Preserve historical rows; bind the new CPU pair row to actual PP content too.
      if (real_ecp_identity->mixed_open_sources ||
          (!device_provider->usesAcceleratorParameterProducts() &&
           device_provider->matchesRealDevicePairTarget(population_.get_golden_electrons())))
        complex_physical_identity = physical_agreement_identity;
    }
    if (device_provider && device_provider->usesAcceleratorParameterProducts() &&
        wftrain::usesEnergyScoreEstimator(training_input_, false))
    {
      if (population_.get_golden_hamiltonian().hasPhysicalNLPP())
        complex_physical_identity = real_ecp_identity && real_ecp_identity->mixed_open_sources
            ? real_ecp_identity->fingerprint
            : realScoreECPIdentity(*device_provider, population_.get_golden_electrons(), population_.get_golden_hamiltonian());
      else if (wftrain::supportsHybridKFACPolicy(training_input_) &&
               device_provider->matchesRealDevicePairTarget(population_.get_golden_electrons()))
        complex_physical_identity = wftrain::realPairCoulombIdentity(*device_provider, population_.get_golden_electrons(),
            population_.get_golden_hamiltonian());
    }
#endif
    if (const auto* parent = dynamic_cast<const FrozenReferenceResidualWF*>(selected_provider_))
    {
      const auto& hamiltonian = population_.get_golden_hamiltonian();
      std::ostringstream identity;
      identity << parent->frozenModelIdentity();
      if (const auto& provenance = hamiltonian.ecpModelProvenance())
        identity << ";ordinary_ecp=" << provenance->ordinary_locality_vp << ";ecp="
                 << provenance->scientific_definition;
      // Preserve actual Ewald settings and physical operator topology across restart.
      for (int i = 0; i < hamiltonian.size(); ++i)
      {
        const auto* term = hamiltonian.getComponent(i);
        identity << ';' << term->getClassName() << ':' << term->getName();
        if (const auto* aa = dynamic_cast<const CoulombPBCAA*>(term))
          identity << ':' << aa->ewaldModelDefinition();
        if (const auto* ab = dynamic_cast<const CoulombPBCAB*>(term))
          identity << ':' << ab->ewaldModelDefinition();
      }
      complex_physical_identity = identity.str();
      physical_agreement_identity = complex_physical_identity;
    }
    if (jastrow_binding_)
    {
      std::ostringstream identity;
      identity << jastrow_binding_->modelIdentity();
      const auto& hamiltonian = population_.get_golden_hamiltonian();
      if (const auto& provenance = hamiltonian.ecpModelProvenance())
        identity << ";ordinary_ecp=" << provenance->ordinary_locality_vp << ";ecp=" << provenance->scientific_definition;
      for (int i = 0; i < hamiltonian.size(); ++i)
      {
        const auto* term = hamiltonian.getComponent(i);
        identity << ';' << term->getClassName() << ':' << term->getName();
        if (const auto* aa = dynamic_cast<const CoulombPBCAA*>(term)) identity << ':' << aa->ewaldModelDefinition();
        if (const auto* ab = dynamic_cast<const CoulombPBCAB*>(term)) identity << ':' << ab->ewaldModelDefinition();
      }
      complex_physical_identity = identity.str();
      physical_agreement_identity = complex_physical_identity;
    }
    if (physical_agreement_identity.empty())
      physical_agreement_identity = complex_physical_identity;
    if (!physical_agreement_identity.empty())
    {
      physical_identity_hash = UINT64_C(14695981039346656037);
      extendStageFingerprint(physical_identity_hash, physical_agreement_identity);
    }
  }
  catch (...) { physical_identity_failure = std::current_exception(); }
  int physical_identity_failed = physical_identity_failure ? 1 : 0;
  myComm->allreduce(physical_identity_failed);
  if (physical_identity_failed)
  {
    if (physical_identity_failure)
      std::rethrow_exception(physical_identity_failure);
    throw UniformCommunicateError("WFTrainBatched: another rank failed GPU physical identity preflight");
  }
  myComm->allgather(&physical_identity_hash, physical_identity_records.data(), 1);
  if (!std::all_of(physical_identity_records.begin(), physical_identity_records.end(),
                   [&](std::uint64_t identity) { return identity == physical_identity_hash; }))
    throw UniformCommunicateError("WFTrainBatched: GPU physical operator or ordered-ion identity differs across ranks before sampling");

  WalkerLogManager wlog_manager(walker_logs_input, allow_walker_logs, get_root_name(), myComm);
  for (auto& crowd : crowds_)
    crowd->setWalkerLogCollector(wlog_manager.makeCollector());
  wlog_manager.startRun(Crowd::getWalkerLogCollectorRefs(crowds_));

  VMCBatchedSampling::State sampling_state(qmcdriver_input_, vmcdriver_input_, *drift_modifier_, population_,
                                            steps_per_block_, serializing_crowd_walkers_);
  sampling_state.evaluate_energy = !training_input_.samplesOrbitalStudent();
  if (vmcdriver_input_.get_move_kind() == VMCDriverInput::MoveKind::N_ELECTRON)
    sampling_state.subset_cursors.resize(crowds_.size(), 0);
  std::unique_ptr<PsiFormerWF::ResidentOpenObservationBatch> resident_observations;
  bool resident_host_dirty = false;
  if (resident_open_observation_)
  {
    sampling_state.resident_device_observation = true;
    sampling_state.resident_device_observe = [&](Crowd& crowd, bool accumulate) {
      resident_host_dirty = true; // Native R/G/L, walker geometry and H are not published here.
      if (!accumulate)
        return;
      RefVector<WaveFunctionComponent> components;
      for (TrialWaveFunction& wf : crowd.get_walker_twfs())
        components.push_back(*wf.getOrbitals()[0]);
      const auto& leader = dynamic_cast<const PsiFormerWF&>(components[0].get());
      resident_observations = leader.observeResidentOpenAccepted({components[0], components});
      const auto precision = app_log().precision();
      app_log() << std::setprecision(std::numeric_limits<double>::max_digits10);
      for (std::size_t sample = 0; sample < resident_observations->sampleCount(); ++sample)
      {
        const auto& energy = resident_observations->energy(sample);
        auto* properties = crowd.get_walkers()[sample].get().getPropertyBase();
        properties[WP::LOCALENERGY] = energy[4];
        properties[WP::LOCALPOTENTIAL] = energy[1] + energy[2] + energy[3];
        const auto& h = crowd.get_walker_hamiltonians()[sample].get();
        for (std::size_t term = 0; term < 4; ++term)
          properties[h.startIndex() + resident_open_scalar_slots_[term]] = energy[term];
        app_log() << "Resident open native observation: parameter_version=" << resident_observations->sourceVersion()
                  << ", sample=" << sample << ", samples=2, K=" << energy[0] << ", ee=" << energy[1]
                  << ", ei=" << energy[2] << ", ii=" << energy[3] << ", total=" << energy[4]
                  << ", host_spatial_cache=dirty, complete_H=yes, coordinate_ingress=DEVICE" << std::endl;
      }
      app_log().precision(precision);
      // Pre-draw admission established exactly one LocalEnergyEstimator and no
      // other scalar/operator/listener consumers. It reads ONLY these properties.
      crowd.accumulate(rngs_[0]);
    };
  }
  VMCBatchedSampling::Timers sampling_timers{
      timers_.resource_timer, timers_.movepbyp_timer, timers_.buffer_timer,
      timers_.hamiltonian_timer, timers_.collectables_timer, timers_.estimators_timer,
      [this](Crowd& crowd, std::string_view location) {
        checkLogAndGL(crowd, location, serializing_crowd_walkers_);
      }};

  wftrain::DistributedReductionPolicy reduction_policy;
  reduction_policy.maximum_chunk_size = training_input_.getParameterChunkSize();
  wftrain::DistributedParameterReduction reduction(*myComm, reduction_policy);
  wftrain::TrainingCapabilities requirements{
      wftrain::TrainingCapability::VALUE_EVALUATION};
  const bool score_training = wftrain::usesEnergyScoreEstimator(training_input_, complex_training);
  const bool score_kfac_training = wftrain::usesKFACScoreEstimator(training_input_, complex_training);
  if (native_mixed_training && (crowds_.size() != 1 ||
      !population_.get_golden_hamiltonian().hasPhysicalNLPP() || !score_training ||
      (training_input_.getOptimizer() != WFTrainDriverInput::Optimizer::ADAM &&
       training_input_.getOptimizer() != WFTrainDriverInput::Optimizer::KFAC) ||
      training_input_.getLineSearch() != WFTrainDriverInput::LineSearch::OFF ||
      training_input_.getMaximumMetricNorm() > 0 || resident_adam_budget || resident_sr_budget.first))
    throw UniformCommunicateError("Mixed native training requires ECP score Adam/KFAC, one crowd, line search OFF, "
        "and no resident optimizer or post-epoch metric query");
  // Admit only a scope that can still be the live chain; replay proves FULL again.
  const char* kfac_replay_optin = std::getenv("PSIFORMER_GPU_KFAC_SINGLE_REPLAY");
  const bool kfac_single_replay = kfac_replay_optin && std::string(kfac_replay_optin) == "1" &&
      score_kfac_training && !complex_training && device_provider &&
      device_provider->usesAcceleratorParameterProducts() &&
      population_.get_golden_electrons().getLattice().SuperCellEnum == SUPERCELL_OPEN &&
      crowds_.size() == 1 && num_blocks == 1 && steps_per_block_ == 1;
  const char* immediate_optin = std::getenv("PSIFORMER_GPU_KFAC_ECP_FULL_HANDOFF");
  if (immediate_optin && std::string(immediate_optin) != "0" && std::string(immediate_optin) != "1")
    throw UniformCommunicateError("PSIFORMER_GPU_KFAC_ECP_FULL_HANDOFF must be 0 or 1");
#if !defined(QMC_COMPLEX) && !defined(MIXED_PRECISION)
  constexpr bool immediate_real_fp64_build = true;
#else
  constexpr bool immediate_real_fp64_build = false;
#endif
  const bool real_kfac_ecp_full_handoff = immediate_optin && std::string(immediate_optin) == "1" &&
      immediate_real_fp64_build && score_kfac_training && !complex_training && !resident_device_authority_ &&
      std::is_same_v<WaveFunctionComponent::ValueType, double> && device_provider &&
      device_provider->usesAcceleratorParameterProducts() &&
      device_provider->matchesLegacyTorusPseudopotentialTarget(population_.get_golden_electrons()) &&
      population_.get_golden_hamiltonian().hasPhysicalNLPP() &&
      crowds_.size() == 1 && num_blocks == 1 && steps_per_block_ == 1;
  bool paired_kfac_ecp_full_handoff = false;
#if defined(QMC_COMPLEX) && !defined(MIXED_PRECISION)
  paired_kfac_ecp_full_handoff = immediate_optin && std::string(immediate_optin) == "1" &&
      complex_training && complex_gpu_provider_ && score_kfac_training && !resident_device_authority_ &&
      complex_training_component_->gpuInferenceEnabled() &&
      complex_training_component_->matchesLegacyTorusPseudopotentialTarget(population_.get_golden_electrons()) &&
      population_.get_golden_hamiltonian().hasPhysicalNLPP() &&
      crowds_.size() == 1 && num_blocks == 1 && steps_per_block_ == 1;
#endif
  const bool kfac_ecp_full_handoff = real_kfac_ecp_full_handoff || paired_kfac_ecp_full_handoff;
  const char* score_handoff_optin = std::getenv("PSIFORMER_GPU_SCORE_ECP_FULL_HANDOFF");
  if (score_handoff_optin && std::string(score_handoff_optin) != "0" && std::string(score_handoff_optin) != "1")
    throw UniformCommunicateError("PSIFORMER_GPU_SCORE_ECP_FULL_HANDOFF must be 0 or 1");
  // Every receipt is captured immediately after its own replay. It owns copied
  // derivative evidence, never mutable FULL tapes or a previous accepted grid.
  const bool score_ecp_full_handoff = score_handoff_optin && std::string(score_handoff_optin) == "1" &&
      score_training && training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::ADAM &&
      !resident_device_authority_ && !jastrow_binding_ && crowds_.size() == 1 &&
      population_.get_golden_hamiltonian().hasPhysicalNLPP() &&
      ((!complex_training && immediate_real_fp64_build && device_provider &&
        device_provider->usesAcceleratorParameterProducts() &&
        (population_.get_golden_electrons().getLattice().SuperCellEnum == SUPERCELL_OPEN ||
         device_provider->matchesLegacyTorusPseudopotentialTarget(population_.get_golden_electrons())))
#ifdef QMC_COMPLEX
       || (complex_training && complex_gpu_provider_ && complex_training_component_ &&
           complex_training_component_->gpuInferenceEnabled() &&
           complex_training_component_->matchesLegacyTorusPseudopotentialTarget(population_.get_golden_electrons()))
#endif
      );
  if (score_handoff_optin && std::string(score_handoff_optin) == "1")
    app_log() << "Score Adam ECP FULL handoff: eligible=" << score_ecp_full_handoff
              << "; per-replay proof, native Hamiltonian/grid/RNG unchanged" << std::endl;
  if (immediate_optin && std::string(immediate_optin) == "1")
    app_log() << "KFAC immediate ECP FULL handoff: eligible=" << kfac_ecp_full_handoff
              << "; only the second synchronous FULL may be elided" << std::endl;
  if (kfac_replay_optin && std::string(kfac_replay_optin) == "1")
    app_log() << "KFAC single-snapshot FULL replay: eligible=" << kfac_single_replay
              << "; exact current version/coordinate/spatial cache checks remain required" << std::endl;
  const auto kfac_estimator = wftrain::kfacScoreEstimator(training_input_);
  const auto energy_estimator = wftrain::energyGradientEstimator(training_input_, complex_training);
  if (training_input_.getObjective() == WFTrainDriverInput::Objective::ENERGY && myComm->rank() == 0)
    app_log() << "WFTrain energy gradient estimator="
              << (energy_estimator == wftrain::EnergyGradientEstimator::SCORE_COVARIANCE ? "score_covariance" :
                  energy_estimator == wftrain::EnergyGradientEstimator::HALF_SCORE_COVARIANCE ? "half_score_covariance" :
                  energy_estimator == wftrain::EnergyGradientEstimator::REAL_CENTERED_SCORE_RESIDUAL ? "real_centered_score_residual" :
                  energy_estimator == wftrain::EnergyGradientEstimator::PATHWISE_LOCAL_ENERGY ? "pathwise_local_energy" :
                  "symmetrized_hamiltonian") << std::endl;
  if (training_input_.hasExplicitScoreResidualPolicy() && myComm->rank() == 0)
    app_log() << "WFTrain score residual policy="
              << (training_input_.getScoreResidualPolicy().transform == wftrain::ScoreResidualTransform::RAW
                      ? "raw" : "soft_squeeze")
              << " width=" << training_input_.getScoreResidualPolicy().width
              << " quantile=" << training_input_.getScoreResidualPolicy().quantile
              << " derivative_family=score parameter_response_mask=0" << std::endl;
  wftrain::HighParameterTraining trainer(
      requirements, energy_estimator,
      reduction);
  logTrainingAllocation("initial_parameter_snapshot", 0, selected_provider_->parameterSchema().parameterCount(),
      checkedMemoryMultiply(selected_provider_->parameterSchema().parameterCount(), sizeof(double)));
  const auto initial_version = [&] {
    const auto initial_parameters = selected_provider_->snapshotParameters();
    if (initial_parameters.schema_fingerprint != selected_provider_->parameterSchema().fingerprint() ||
        initial_parameters.values.size() != selected_provider_->parameterSchema().parameterCount())
      throw UniformCommunicateError(
          "WFTrainBatched: selected provider returned an inconsistent initial snapshot");
    return initial_parameters.version;
  }(); // Release the cold P vector before warm iterations.
  std::unique_ptr<wftrain::FirstOrderOptimizer> first_order_optimizer;
  if (!device_provider && selected_provider_->usesAcceleratorParameterProducts())
    if (const char* value = std::getenv("PSIFORMER_GPU_ADAM_BYTES"))
    {
      bool enabled = false;
      for (const char* digit = value; *digit; ++digit)
      {
        if (*digit < '0' || *digit > '9')
          throw UniformCommunicateError("PSIFORMER_GPU_ADAM_BYTES must be a base-10 byte limit or zero/off");
        enabled = enabled || *digit != '0';
      }
      if (enabled)
        throw UniformCommunicateError("Explicit GPU Adam requires a selected PsiFormer provider");
    }
  const auto device_adam_budget = device_provider ? device_provider->deviceAdamBudget() : 0;
  if (device_adam_budget && training_input_.getOptimizer() != WFTrainDriverInput::Optimizer::ADAM)
    throw UniformCommunicateError("Explicit GPU Adam budget requires the selected real Adam optimizer");
  if (isFirstOrder(training_input_.getOptimizer()))
    first_order_optimizer = std::make_unique<wftrain::FirstOrderOptimizer>(
        selected_provider_->parameterSchema(),
        makeFirstOrderOptions(selected_provider_->parameterSchema(), training_input_));
  if (device_adam_budget)
    first_order_optimizer->prepareDeviceAdam(device_adam_budget);
  if (resident_adam_budget)
  {
    first_order_optimizer->prepareResidentAdamNorm(resident_norm_bytes);
    app_log() << (resident_score_adam ? "Resident score Adam bridge device cap(C2+norm only)="
                                   : "Resident ENERGY Adam bridge device cap(C3+norm only)=") << resident_adam_budget
              << ", norm device bytes=" << resident_norm_bytes << ", norm host bytes=" << resident_norm_bytes
              << "; existing derivative tapes/host compatibility channels retained once, not eliminated; not a process feasibility budget" << std::endl;
  }
  std::unique_ptr<wftrain::SPRINGUpdateRule> spring_optimizer;
  if (training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::SPRING)
    spring_optimizer = std::make_unique<wftrain::SPRINGUpdateRule>(
        selected_provider_->parameterSchema(), makeSPRINGControl(training_input_));
  std::shared_ptr<wftrain::KFACDenseSolver> kfac_dense_solver;
  if (training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::KFAC)
  {
    // Backend is execution policy, not curvature/checkpoint identity. Agree it
    // before any rank can enter a device-only preparation or sample collection.
    std::uint64_t requested = training_input_.getKFACDenseBackend() ==
        WFTrainDriverInput::KFACDenseBackend::DEVICE ? 1 : 0;
    const bool device_factor_authority = training_input_.getKFACFactorAuthority() ==
        WFTrainDriverInput::KFACFactorAuthority::DEVICE;
    requested |= device_factor_authority ? 2 : 0;
    std::vector<std::uint64_t> requests(myComm->size());
    myComm->allgather(&requested, requests.data(), 1);
    if (!std::all_of(requests.begin(), requests.end(), [&](auto value) { return value == requested; }))
      throw UniformCommunicateError("WFTrainBatched: KFAC dense backend/factor authority differs across ranks");
    if (device_factor_authority && !(requested & 1))
      throw UniformCommunicateError("WFTrainBatched: device factor authority requires the device dense backend");
    if (requested & 1)
    {
      std::exception_ptr failure;
      try
      {
#if defined(ENABLE_CUDA) && !defined(MIXED_PRECISION)
        if (!training_input_.getKFACPersistState() || !selected_provider_->usesAcceleratorParameterProducts() ||
            !wftrain::supportsHybridKFACPolicy(training_input_) || resident_device_authority_)
          throw std::invalid_argument("Device KFAC dense backend requires supported persistent GPU FP64 score KFAC");
        kfac_dense_solver = wftrain::KFACDenseSolver::prepare(makeLiveKFACRegistry(*selected_provider_),
            0, training_input_.getMaximumRetainedBytes(),
            device_factor_authority ? wftrain::KFACDenseCapacity::RESIDENT : wftrain::KFACDenseCapacity::AUTO,
            device_factor_authority);
#else
        throw std::invalid_argument("Device KFAC dense backend requires a FP64 CUDA/HIP build");
#endif
      }
      catch (...) { failure = std::current_exception(); }
      int failed = failure ? 1 : 0;
      myComm->allreduce(failed);
      if (failed)
      {
        if (failure) std::rethrow_exception(failure);
        throw UniformCommunicateError("WFTrainBatched: another rank failed KFAC device admission before collection");
      }
      const std::array<std::uint64_t, 5> policy{static_cast<std::uint64_t>(kfac_dense_solver->capacity()),
          kfac_dense_solver->workspaceBytes(), kfac_dense_solver->deviceBytes(), kfac_dense_solver->hostBytes(),
          static_cast<std::uint64_t>(kfac_dense_solver->deviceAuthority())};
      std::vector<std::uint64_t> policies(static_cast<std::size_t>(myComm->size()) * policy.size());
      auto send = policy;
      myComm->allgather(send.data(), policies.data(), static_cast<int>(policy.size()));
      for (int rank = 0; rank < myComm->size(); ++rank)
        if (!std::equal(policy.begin(), policy.end(), policies.begin() + rank * policy.size()))
          throw UniformCommunicateError("WFTrainBatched: KFAC device capacity/workspace policy differs across ranks");
      app_log() << "WFTrain KFAC dense backend=device capacity="
                << (kfac_dense_solver->capacity() == wftrain::KFACDenseCapacity::RESIDENT ? "resident" : "one_block_streamed")
                << " DEVICE_numeric_bytes=" << kfac_dense_solver->deviceBytes()
                << " HOST_metadata_bound=" << kfac_dense_solver->hostBytes()
                << " workspace_bytes=" << kfac_dense_solver->workspaceBytes()
                << " factor_authority=" << (device_factor_authority ? "device" : "host")
                << " HOST_factor_materialization=" << (device_factor_authority ? "checkpoint_or_explicit_HOST" : "eager")
                << "; HOST reduction/EMA/canonical checkpoint retained; not fully resident KFAC" << std::endl;
    }
  }
  std::unique_ptr<wftrain::KFACOptimizer> kfac_optimizer;
  if (training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::KFAC && training_input_.getKFACPersistState())
  {
    wftrain::KFACOptimizerControl control;
    control.learning_rate = training_input_.getLearningRate();
    control.learning_rate_decay = training_input_.getKFACLearningRateDecay();
    control.schedule_float32 = training_input_.getKFACScheduleFloat32();
    control.update_order = training_input_.getKFACUpdateOrder() == WFTrainDriverInput::KFACUpdateOrder::PRE_INVERSE_RATE
        ? wftrain::KFACUpdateOrder::PRE_INVERSE_RATE : wftrain::KFACUpdateOrder::POST_INVERSE_RATE;
    control.schedule_arithmetic = training_input_.getKFACScheduleArithmetic() == WFTrainDriverInput::KFACScheduleArithmetic::MULTIPLY_RECIPROCAL
        ? wftrain::KFACScheduleArithmetic::MULTIPLY_RECIPROCAL : wftrain::KFACScheduleArithmetic::DIVIDE_RATE;
    control.ema_decay = training_input_.getKFACEMADecay();
    control.inverse_period = training_input_.getKFACInversePeriod();
    control.norm_constraint = training_input_.getKFACConstraint() == WFTrainDriverInput::KFACConstraint::DAMPED_KFAC
        ? training_input_.getKFACNormConstraint() : 0.0;
    control.convention = training_input_.getKFACCurvature() == WFTrainDriverInput::KFACCurvature::REGISTERED_GAUSSIAN
        ? wftrain::KFACCurvatureConvention::REGISTERED_GAUSSIAN : wftrain::KFACCurvatureConvention::NATIVE_SCORE;
    control.inverse = {training_input_.getDamping(), training_input_.getKFACFallbackDamping(),
        training_input_.getKFACDampingPolicy() == WFTrainDriverInput::KFACDamping::TRACE_BALANCED
            ? wftrain::KFACDampingPolicy::TRACE_BALANCED : wftrain::KFACDampingPolicy::EQUAL_SPLIT};
    kfac_optimizer = std::make_unique<wftrain::KFACOptimizer>(makeLiveKFACRegistry(*selected_provider_), control,
        kfac_dense_solver);
  }
  wftrain::OptimizerCheckpointContributor* optimizer_checkpoint = nullptr;
  if (first_order_optimizer)
    optimizer_checkpoint = first_order_optimizer.get();
  else if (spring_optimizer)
    optimizer_checkpoint = spring_optimizer.get();
  else if (kfac_optimizer)
    optimizer_checkpoint = kfac_optimizer.get();

  wftrain::TrainingIterationState training_state{
      0, initial_version, selected_provider_->parameterSchema().fingerprint()};
  const bool determinant_overlap =
      training_input_.getObjective() ==
      WFTrainDriverInput::Objective::DETERMINANT_OVERLAP;
  const bool orbital_mse = training_input_.getObjective() == WFTrainDriverInput::Objective::ORBITAL_MSE;
  bool has_orbital_target = orbital_target_ != nullptr;
#ifdef QMC_COMPLEX
  has_orbital_target = has_orbital_target || complex_orbital_target_ != nullptr;
#endif
  if (orbital_mse && !has_orbital_target)
    throw UniformCommunicateError("WFTrainBatched: orbital target was not initialized by process()");
  if (determinant_overlap && !determinant_overlap_objective_)
    throw UniformCommunicateError(
        "WFTrainBatched: determinant-overlap state was not initialized by process()");
  const std::uint64_t overlap_fingerprint = determinant_overlap
      ? determinant_overlap_objective_->configurationFingerprint()
      : (orbital_mse ? orbital_target_fingerprint_ : 0);
  const bool has_nonlocal_ecp =
      population_.get_golden_hamiltonian().hasPhysicalNLPP();
  wftrain::TrainingStageState stage_state{
      orbital_mse ? wftrain::OrbitalPretrainingCoordinator::STAGE_ID : determinant_overlap ? "determinant_overlap"
                          : (has_nonlocal_ecp ? "pseudopotential_energy"
                                              : "all_electron_energy"),
      0, 0,
      stageConfigurationFingerprint(training_input_, qmcdriver_input_, vmcdriver_input_,
                                    selected_provider_->parameterSchema(), steps_per_block_,
                                    overlap_fingerprint, complex_physical_identity, resident_adam_budget != 0, resident_sr_budget.first != 0)};
  if (conventional_binding_)
    stage_state.configuration_fingerprint += ":" + conventional_binding_->modelIdentity();
  if (const auto* parent = dynamic_cast<const FrozenReferenceResidualWF*>(selected_provider_))
    stage_state.configuration_fingerprint += ":" + parent->frozenModelIdentity();
  if (resident_device_authority_)
    stage_state.configuration_fingerprint += ":resident-device-authority-v1";
  DirtyCacheObserver cache_observer;

  LoopTimer<> training_loop;
  RunTimeControl<> runtime_control(run_time_manager, project_data_.getMaxCPUSeconds(),
                                   project_data_.getTitle(), myComm->rank() == 0);
  ParallelExecutor<> crowd_task;

  int global_step = 0;
  wftrain::WFTrainSamplerCheckpoint sampler_checkpoint(
      population_, rngs_, global_step, static_cast<std::size_t>(myComm->size()),
      static_cast<std::size_t>(myComm->rank()), crowds_.size(),
      vmcdriver_input_.get_move_kind() == VMCDriverInput::MoveKind::N_ELECTRON &&
              vmcdriver_input_.get_electron_selection() == VMCDriverInput::ElectronSelection::CYCLIC
          ? &sampling_state.subset_cursors : nullptr);
  const bool resumed = !training_input_.getRestartPath().empty();
  if (!training_input_.getCheckpointPath().empty() && myComm->rank() == 0)
    app_warning() << "WFTrain checkpoint_path is deprecated; native checkpoints use "
                  << get_root_name() << ".training/<iteration>.checkpoint/ and latest pointer "
                  << get_root_name() << ".last_checkpoint.h5" << std::endl;
  // Scalar blocks remain statistical output. This separate history records one
  // complete optimization attempt, including its sampled and published versions.
  std::ofstream iteration_history;
  bool history_failed = false;
  if (resumed)
  {
    if (myComm->rank() == 0)
      app_log() << "WFTrain exact restart requires unchanged physical-system and "
                   "Hamiltonian input; stage validation covers the parameter schema, "
                   "training controls, and sampler topology."
                << std::endl;
    wftrain::DistributedTrainingCheckpoint::restore(
        training_input_.getRestartPath(), *myComm, *selected_provider_, training_state,
        stage_state, optimizer_checkpoint, sampler_checkpoint, sampler_checkpoint.topology());
    if (stage_state.completed_stage_iterations != training_state.completed_iterations ||
        stage_state.completed_stage_iterations > static_cast<std::size_t>(num_iterations))
      throw UniformCommunicateError(
          "WFTrainBatched: restored training progress is inconsistent with requested iterations");
    if (myComm->rank() == 0)
      app_log() << "WFTrain restored " << stage_state.completed_stage_iterations
                << " completed iteration(s) from " << training_input_.getRestartPath()
                << "; runtime parameter generation " << training_state.parameter_version
                << " (checkpoint source generations are rebased to this process)" << std::endl;
  }

  if (resident_device_authority_ && resident_adam_budget)
    first_order_optimizer->seedResidentDeviceAdam(cold_moment_allowance); // Seed BEFORE vacating initial/restart HOST state.

  // Validate the entire existing output family before sampling or opening files.
  // An older checkpoint may branch under a new root, but cannot replace future data.
  const std::filesystem::path training_directory(get_root_name() + ".training");
  if (myComm->rank() == 0)
    try
    {
      const auto completed = training_state.completed_iterations;
      const std::filesystem::path history_path(get_root_name() + ".training_info.csv");
      const bool has_history = std::filesystem::exists(history_path);
      if (has_history)
      {
        std::ifstream previous(history_path);
        std::string line;
        if (!std::getline(previous, line) || line.rfind("iteration,sample_version,parameter_version,", 0) != 0)
          throw std::runtime_error("Invalid training history header");
        std::optional<std::size_t> next_iteration;
        while (std::getline(previous, line))
        {
          const auto comma = line.find(',');
          const auto index = comma == std::string::npos ? std::string{} : line.substr(0, comma);
          if (index.empty() || index.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("Noncontiguous training history");
          const auto iteration_index = std::stoull(index);
          if (next_iteration && iteration_index != *next_iteration)
            throw std::runtime_error("Noncontiguous training history");
          next_iteration = iteration_index + 1;
        }
        if (!resumed || (next_iteration ? *next_iteration != completed : completed != 0))
          throw std::runtime_error("Training history conflicts with restart progress");
      }
      const auto reject_future = [&](const std::filesystem::path& directory) {
        if (!std::filesystem::exists(directory))
          return;
        const std::string prefix = std::filesystem::path(get_root_name()).filename().string() + ".i";
        for (const auto& entry : std::filesystem::directory_iterator(directory))
        {
          const auto name = entry.path().filename().string();
          if (name.rfind(prefix, 0) != 0)
            continue;
          // Ordinary containers such as .inference share the .i prefix. Only
          // a digit starts a numbered artifact; malformed numbered names still fail below.
          if (name.size() > prefix.size() && (name[prefix.size()] < '0' || name[prefix.size()] > '9'))
            continue;
          const auto end = name.find('.', prefix.size());
          const auto digits = name.substr(prefix.size(), end - prefix.size());
          if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos ||
              std::stoull(digits) >= completed)
            throw std::runtime_error("Existing training iteration conflicts with restart: " + name);
        }
      };
      reject_future(training_directory);
      if (inference_input_.enabled)
        reject_future(get_root_name() + ".inference");
      const auto parent = std::filesystem::path(get_root_name()).parent_path();
      reject_future(parent.empty() ? std::filesystem::path(".") : parent);
      std::filesystem::create_directories(training_directory);
      iteration_history.open(history_path, has_history ? std::ios::app : std::ios::out);
      if (!has_history)
        iteration_history << "iteration,sample_version,parameter_version,objective,optimizer,samples,"
                             "mean_energy,variance,gradient_norm,overlap_loss,selected_alpha,committed,"
                             "sampling_seconds,total_seconds,retained_numeric_bytes,resident_bytes,"
                             "krylov_iterations,krylov_operator_applications,orbital_loss\n";
      iteration_history.flush();
      history_failed = !iteration_history;
    }
    catch (const std::exception& error)
    {
      app_error() << "WFTrain output validation: " << error.what() << std::endl;
      history_failed = true;
    }
  myComm->bcast(history_failed);
  if (history_failed)
    throw UniformCommunicateError("WFTrainBatched: conflicting output family or inaccessible training history");

  if (score_training && training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::ADAM &&
      population_.get_golden_hamiltonian().hasPhysicalNLPP())
  {
    if (device_provider && device_provider->usesAcceleratorParameterProducts())
      for (std::size_t ordinal = 0; ordinal < crowds_.size(); ++ordinal)
        score_lane_anchors.push_back(device_provider->retainGpuScoreLane(ordinal));

#ifdef QMC_COMPLEX
    else if (complex_gpu_provider_)
    {
      const auto* complex = complex_training_component_;
      if (!complex || !complex->gpuInferenceEnabled())
        throw UniformCommunicateError("Complex score anchors require the admitted GPU component behind the provider");
      const auto& limits = training_input_.getComplexGpuTraining();
      for (std::size_t ordinal = 0; ordinal < crowds_.size(); ++ordinal)
        complex_score_lane_anchors.push_back(complex->retainGpuScoreLane(ordinal, crowds_[ordinal]->size(),
            limits.kinetic_targets, std::min(selected_provider_->parameterSchema().parameterCount(),
                training_input_.getParameterChunkSize()), limits.device_bytes, limits.host_bytes,
            limits.maximum_matrix_work));
    }
#endif
  }

  {
    ScopedTimer initialization_timer(timers_.init_walkers_timer);
    auto step_context_refs = getContextForStepsRefs();
    const auto initialize = training_input_.samplesOrbitalStudent() ? initialLogEvaluationWavefunctionOnly
        : resumed ? initialLogEvaluationDeterministic : initialLogEvaluation;
    crowd_task(crowds_.size(), initialize, crowds_, step_context_refs,
               serializing_crowd_walkers_);
    print_mem("WFTrainBatched after initialLogEvaluation", app_summary());
  }
  enforceResidentBudget(*myComm, training_input_.getMaximumResidentBytes());

  ScopedTimer production_timer(timers_.production_timer);
  if (!resumed)
    for (int step = 0; step < qmcdriver_input_.get_warmup_steps(); ++step)
    {
      ScopedTimer step_timer(timers_.run_steps_timer);
      crowd_task(crowds_.size(), VMCBatchedSampling::runWarmupStep, sampling_state,
                 sampling_timers, rngs_, crowds_);
    }
  myComm->barrier();

  int completed_blocks = 0;
  std::size_t accepted_updates = 0;
  if (myComm->rank() == 0)
    app_log() << "WFTrain sampling schedule: iterations " << num_iterations
              << " blocks_per_iteration " << num_blocks << " steps_per_block " << steps_per_block_
              << " substeps " << qmcdriver_input_.get_sub_steps()
              << " iteration_warmupsteps " << training_input_.getWarmupSteps() << std::endl;

  if (myComm->rank() == 0 && training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::KFAC)
  {
    app_log() << "WFTrain KFAC controls: defaults="
              << (training_input_.getKFACDefaults() == WFTrainDriverInput::KFACDefaults::LEGACY ? "legacy" :
                  training_input_.getKFACDefaults() == WFTrainDriverInput::KFACDefaults::SOFT_COVARIANCE ? "deepqmc" :
                  training_input_.getKFACDefaults() == WFTrainDriverInput::KFACDefaults::MEAN_RESIDUAL ? "deepsolid" : "jaqmc")
              << " policy_version=1 gradient_convention="
              << (score_kfac_training ?
                  (kfac_estimator == wftrain::EnergyGradientEstimator::HALF_SCORE_COVARIANCE ? "half_covariance" :
                   kfac_estimator == wftrain::EnergyGradientEstimator::REAL_CENTERED_SCORE_RESIDUAL ? "real_centered_residual" :
                   "physical_covariance") : "legacy_real_score")
              << " update_order="
              << (training_input_.getKFACUpdateOrder() == WFTrainDriverInput::KFACUpdateOrder::PRE_INVERSE_RATE
                    ? "pre_inverse_rate" : "post_inverse_rate")
              << " objective="
              << (training_input_.getKFACObjective() == WFTrainDriverInput::KFACObjective::RAW_COVARIANCE
                      ? "raw_covariance" :
                  training_input_.getKFACObjective() == WFTrainDriverInput::KFACObjective::COMPONENTWISE_MAD_COVARIANCE
                      ? "componentwise_mad_covariance" :
                  training_input_.getKFACObjective() == WFTrainDriverInput::KFACObjective::COMPONENTWISE_MEAN_RESIDUAL
                      ? "componentwise_mean_residual" : "clipped_covariance")
              << " clip_mode="
              << (training_input_.getKFACObjective() == WFTrainDriverInput::KFACObjective::COMPONENTWISE_MAD_COVARIANCE
                      ? "componentwise_mad" :
                  training_input_.getKFACObjective() == WFTrainDriverInput::KFACObjective::COMPONENTWISE_MEAN_RESIDUAL
                      ? "componentwise_mean_residual" :
                  training_input_.getKFACClipMode() == WFTrainDriverInput::KFACClipMode::SOFT_MEDIAN
                      ? "soft_median" : training_input_.getKFACClipMode() == WFTrainDriverInput::KFACClipMode::HARD_MEDIAN
                      ? "hard_median" : "hard_mean")
              << " clip_width=" << (training_input_.getKFACObjective() ==
                  WFTrainDriverInput::KFACObjective::COMPONENTWISE_MAD_COVARIANCE ||
                  training_input_.getKFACObjective() == WFTrainDriverInput::KFACObjective::COMPONENTWISE_MEAN_RESIDUAL
                      ? training_input_.getKFACComplexClipWidth() : training_input_.getKFACClipWidth())
              << " clip_quantile=" << training_input_.getKFACClipQuantile()
              << " schedule_arithmetic="
              << (training_input_.getKFACScheduleArithmetic() == WFTrainDriverInput::KFACScheduleArithmetic::MULTIPLY_RECIPROCAL
                    ? "multiply_reciprocal" : "divide_rate")
              << " exclude_width=";
    // Keep the unbounded policy out of a finite-math scalar logging argument.
    if (wftrain::isPositiveInfinityTrainingReal(training_input_.getKFACExcludeWidth()))
      app_log() << "inf";
    else
      app_log() << training_input_.getKFACExcludeWidth();
    app_log() << " curvature="
              << (training_input_.getKFACCurvature() == WFTrainDriverInput::KFACCurvature::NATIVE_SCORE
                      ? "native_score" : "registered_gaussian")
              << " damping_policy="
              << (training_input_.getKFACDampingPolicy() == WFTrainDriverInput::KFACDamping::EQUAL_SPLIT
                      ? "equal_split" : "trace_balanced")
              << " damping=" << training_input_.getDamping()
              << " fallback_damping=" << training_input_.getKFACFallbackDamping()
              << " learning_rate=" << training_input_.getLearningRate()
              << " constraint="
              << (training_input_.getKFACConstraint() == WFTrainDriverInput::KFACConstraint::NONE
                      ? "none" : "damped_kfac")
              << " norm_constraint=" << training_input_.getKFACNormConstraint()
              << " persist_state=" << (training_input_.getKFACPersistState() ? "yes" : "no")
              << " weight_decay=" << training_input_.getWeightDecay()
              << " update_scale=" << training_input_.getUpdateScale()
              << " line_search=" << (training_input_.getLineSearch() == WFTrainDriverInput::LineSearch::OFF
                  ? "off" : training_input_.getLineSearch() == WFTrainDriverInput::LineSearch::REFERENCE ? "reference" : "endpoints")
              << " constraint_units=LR^2*g_dot_preconditioned_g"
              << " euclidean_update_cap=" << training_input_.getMaximumUpdateNorm()
              << " registered_curvature_reference_devices=1" << std::endl;
  }
  if (myComm->rank() == 0 && kfac_optimizer)
    app_log() << "WFTrain persistent KFAC: step=" << kfac_optimizer->step()
              << " data_seen=" << kfac_optimizer->dataSeen()
              << " ema_decay=" << training_input_.getKFACEMADecay()
              << " affine_inverse_period=" << training_input_.getKFACInversePeriod()
              << " curvature_period=1 momentum=0 burn_in=0"
              << " learning_rate_decay=" << training_input_.getKFACLearningRateDecay()
              << " schedule_exponent=1"
              << " schedule_precision=" << (training_input_.getKFACScheduleFloat32() ? "float32" : "float64")
              << " effective_learning_rate=" << kfac_optimizer->effectiveLearningRate() << std::endl;

  for (int iteration = static_cast<int>(stage_state.completed_stage_iterations);
       iteration < num_iterations; ++iteration)
  {
    ScopeGuard<LoopTimer<>> iteration_timer(training_loop);
    Timer iteration_wall;
    const auto sample_version = training_state.parameter_version;
    if (resident_device_authority_)
    {
      if (resident_adam_budget)
      {
        first_order_optimizer->seedResidentDeviceAdam(cold_moment_allowance);
        app_log() << "Resident DEVICE Adam HOST moments before sampling: actual_bytes="
                  << first_order_optimizer->hostMomentMirrorBytes() << ", cold_allowance=" << cold_moment_allowance << std::endl;
      }
      device_provider->residentDeviceAuthorityTransferWindow("sampling_through_refresh", true);
    }
    logTrainingMemoryPhase(*myComm, iteration, "before_sampling");
    psiformer::memory::Phase memory_phase("sampling");
    app_log() << "WFTrain allocation before owner=retained_sample_snapshots iteration=" << iteration
              << " scope=rank0_local blocks=" << num_blocks << " steps_per_block=" << steps_per_block_
              << " crowds=" << crowds_.size() << " parameters=" << selected_provider_->parameterSchema().parameterCount()
              << " planned_numeric_bytes=unknown (sample_coordinates_and_scalar_metadata)" << std::endl;
    std::ostringstream iteration_name;
    iteration_name << std::filesystem::path(get_root_name()).filename().string()
                   << ".i" << std::setfill('0') << std::setw(7) << iteration;
    std::exception_ptr output_failure;
    try
    {
      if (sampling_state.evaluate_energy)
        estimator_manager_->startDriverRun({(training_directory / iteration_name.str()).string(), true});
    }
    catch (...) { output_failure = std::current_exception(); }
    int output_failed = output_failure ? 1 : 0;
    myComm->allreduce(output_failed);
    if (output_failed)
      throw UniformCommunicateError("WFTrainBatched: cannot start scalar-only iteration output");
    // This equilibration is separate from the initial qmc warmup. Warmup moves
    // honor substeps but contribute neither retained samples nor scalar statistics.
    for (std::size_t warmup = 0; warmup < training_input_.getWarmupSteps(); ++warmup)
    {
      ScopedTimer step_timer(timers_.run_steps_timer);
      crowd_task(crowds_.size(), VMCBatchedSampling::runWarmupStep, sampling_state,
                 sampling_timers, rngs_, crowds_);
      ++global_step;
    }

    std::vector<RetainedCrowdSamples> retained_samples;
    std::size_t retained_configuration_bytes = checkedMemoryAdd(complex_affine_score_bytes,
        checkedMemoryAdd(resident_authority_extra_bytes_, complex_fixed_grid_host_bytes));
    for (int block = 0; block < num_blocks; ++block)
    {
      const int statistical_block = iteration * num_blocks + block;
      if (myComm->rank() == 0)
        app_log() << "WFTrain statistical block " << statistical_block
                  << " iteration " << iteration
                  << " version " << training_state.parameter_version << std::endl;
      sampling_state.recalculate_properties_period = qmc_driver_mode_[QMC_UPDATE_MODE]
          ? qmcdriver_input_.get_recalculate_properties_period() : 0;
      sampling_state.is_recomputing_block = qmcdriver_input_.get_blocks_between_recompute()
          ? (1 + statistical_block) % qmcdriver_input_.get_blocks_between_recompute() == 0 : false;
      if (sampling_state.evaluate_energy)
      {
        estimator_manager_->startBlock(steps_per_block_);
        for (auto& crowd : crowds_)
          crowd->startBlock(steps_per_block_);
      }
      for (int step = 0; step < steps_per_block_; ++step, ++global_step)
      {
        ScopedTimer step_timer(timers_.run_steps_timer);
        sampling_state.step = step;
        sampling_state.global_step = global_step;
        crowd_task(crowds_.size(), VMCBatchedSampling::runVMCStep, sampling_state,
                   sampling_timers, rngs_, crowds_);
        // Fail collectively before endBlock reductions if any rank cannot
        // retain a snapshot. Charge copied numerical arrays, not allocator/RSS
        // overhead or opaque teacher-clone caches, to the numeric storage budget.
        std::exception_ptr retention_failure;
        try
        {
          for (std::size_t crowd_index = 0; !resident_open_observation_ && crowd_index < crowds_.size(); ++crowd_index)
            if (crowds_[crowd_index]->size())
            {
              retained_samples.push_back(retainCrowdSamples(
                  *crowds_[crowd_index], crowd_index, determinant_overlap, complex_training,
                  (resident_adam_budget || resident_sr_budget.first || kfac_single_replay)
                      ? std::optional<std::size_t>(training_state.parameter_version) : std::nullopt));
              const auto& snapshot = retained_samples.back();
              if (snapshot.replay_version)
                retained_configuration_bytes = checkedMemoryAdd(retained_configuration_bytes,
                    checkedMemoryAdd(sizeof(*snapshot.replay_version), checkedMemoryMultiply(
                        snapshot.replay_log_phase.capacity(), sizeof(snapshot.replay_log_phase[0]))));
              retained_configuration_bytes = checkedMemoryAdd(retained_configuration_bytes,
                  checkedMemoryMultiply(snapshot.local_energies.capacity(), sizeof(double)));
              for (const auto& particles : snapshot.particles)
                retained_configuration_bytes = checkedMemoryAdd(retained_configuration_bytes,
                    checkedMemoryMultiply(particles->R.size(),
                        sizeof(particles->R[0]) + sizeof(particles->G[0]) + sizeof(particles->L[0])));
            }
        }
        catch (...) { retention_failure = std::current_exception(); }
        int retention_failed = retention_failure ? 1 : 0;
        myComm->allreduce(retention_failed);
        if (retention_failed)
        {
          if (retention_failure)
            std::rethrow_exception(retention_failure);
          throw UniformCommunicateError("WFTrainBatched: another rank failed to retain iteration samples");
        }
        enforceMemoryBudget(*myComm, retained_configuration_bytes, training_input_.getMaximumRetainedBytes());
        enforceResidentBudget(*myComm, training_input_.getMaximumResidentBytes());
      }
      // Statistical rows refer only to this iteration's fixed parameter version.
      if (sampling_state.evaluate_energy)
        endBlock();
      wlog_manager.writeBuffers();
      ++completed_blocks;
    }

    int scalar_write_failed = 0;
    try { if (sampling_state.evaluate_energy) estimator_manager_->stopDriverRun(); }
    catch (...) { scalar_write_failed = 1; }
    myComm->allreduce(scalar_write_failed);
    if (scalar_write_failed)
      throw UniformCommunicateError("WFTrainBatched: failed to write scalar iteration output");
    const double sampling_seconds = iteration_wall.elapsed();
    std::unique_ptr<wftrain::GpuTrainingEpochReplay> native_epoch_replay;
    if (native_mixed_training)
    {
      std::exception_ptr snapshot_failure;
      try
      {
        std::vector<NonLocalECPotential*> potentials;
        for (const auto& crowd : crowds_)
          for (QMCHamiltonian& ham : crowd->get_walker_hamiltonians())
            for (int component = 0; component < ham.size(); ++component)
              if (auto* nonlocal = dynamic_cast<NonLocalECPotential*>(ham.getComponent(component)))
                potentials.push_back(nonlocal);
        native_epoch_replay = std::make_unique<wftrain::GpuTrainingEpochReplay>(rngs_, potentials);
        retained_configuration_bytes = checkedMemoryAdd(retained_configuration_bytes, native_epoch_replay->retainedBytes());
        if (retained_configuration_bytes > training_input_.getMaximumRetainedBytes())
          throw std::length_error("Native mixed predecessor RNG/grid snapshots exceed retained budget");
      }
      catch (...) { snapshot_failure = std::current_exception(); }
      requireNativeEpochReadiness(*myComm, snapshot_failure, false);
    }
    logTrainingMemoryPhase(*myComm, iteration, "after_sampling_before_preparation");
    memory_phase.transition("retained_configuration_and_product_preparation");
    logTrainingAllocation("prepared_derivative_owners_and_shared_workspace", retained_samples.size(),
        selected_provider_->parameterSchema().parameterCount());
    wftrain::TrainingIterationResult iteration_result;
    std::optional<ResidentEnergyDiagnostics> resident_diagnostics;
    wftrain::PreparedEnergyGradientStorageDiagnostics storage_diagnostics;
    std::size_t retained_numeric_bytes = 0;
    std::optional<std::size_t> krylov_iterations;
    std::optional<std::size_t> krylov_operator_applications;
    std::optional<wftrain::DerivativeReal> applied_scale;
    std::optional<wftrain::DeterminantOverlapObjectiveResult> overlap_diagnostics;
    std::optional<wftrain::OrbitalPretrainingResult> orbital_diagnostics;
    double selected_alpha = 1.0;
    double selected_displacement_norm = 0.0;
#ifdef QMC_COMPLEX
    std::optional<wftrain::PairedFixedGridNativeEnergy> fixed_grid;
    if (complex_fixed_grid_host_bytes)
      fixed_grid.emplace();
#endif
    auto decorate = [&](wftrain::TrainingUpdateRule& optimizer) {
      wftrain::ProposalSelectionControl control;
      control.update_scale = training_input_.getUpdateScale();
      control.weight_decay = training_input_.getWeightDecay();
      control.maximum_update_norm = training_input_.getMaximumUpdateNorm();
      if (control.weight_decay != 0.0)
      {
        control.decay_blocks = selected_provider_->neuralWeightBlocks();
        if (control.decay_blocks.empty())
          throw UniformCommunicateError("Selected provider has no typed nn_weights decay selection");
      }
      return std::make_unique<wftrain::SelectedUpdateRule>(optimizer, std::move(control),
          [&](const auto& old, const auto& full) {
            double norm = 0.0;
            for (std::size_t i = 0; i < old.values.size(); ++i)
              norm = std::hypot(norm, full.values[i] - old.values[i]);
            double maximum_alpha = 2.0;
            if (training_input_.getMaximumUpdateNorm() > 0.0 && norm > 0.0)
              maximum_alpha = std::min(maximum_alpha, training_input_.getMaximumUpdateNorm() / norm);
            if (training_input_.getMaximumMetricNorm() > 0.0)
            {
              std::vector<std::unique_ptr<wftrain::StreamingDerivativeOperator>> owners;
              std::vector<std::vector<double>> weights;
              std::vector<std::vector<wftrain::DerivativeValue>> energies;
              std::vector<wftrain::ScoreCovarianceBatch> metric_batches;
              for (auto& samples : retained_samples)
                {
                  const auto& twfs = crowds_[samples.crowd_index]->get_walker_twfs();
                  RefVectorWithLeader<TrialWaveFunction> wf_batch(twfs[0], twfs);
                  RefVectorWithLeader<ParticleSet> p_batch(*samples.particles.front());
                  for (auto& particles : samples.particles)
                    p_batch.push_back(*particles);
                  owners.push_back(TrialWaveFunction::makeStructuredStreamingDerivativeOperator(
                      wf_batch, p_batch, selected_provider_id_, owners.size(), 0,
                      training_input_.getParameterChunkSize()));
                  weights.emplace_back(samples.particles.size(), 1.0);
                  energies.emplace_back(samples.particles.size(), 0.0);
                  metric_batches.push_back({owners.back().get(),
                      {weights.back().data(), weights.back().size()},
                      {energies.back().data(), energies.back().size()}});
                }
              wftrain::ScoreCovarianceFactor factor(selected_provider_->parameterSchema(), old.version,
                  {metric_batches.data(), metric_batches.size()}, wftrain::ScoreSampleDomain::REAL, reduction);
              std::vector<wftrain::DerivativeValue> direction(old.values.size());
              for (std::size_t i = 0; i < direction.size(); ++i)
                direction[i] = full.values[i] - old.values[i];
              std::vector<wftrain::DerivativeValue> product(factor.localSampleVectorSize());
              factor.apply({selected_provider_->parameterSchema(), old.version,
                            {direction.data(), direction.size()}}, {product.data(), product.size()});
              double squared = 0.0;
              for (auto value : product) squared += std::norm(value);
              myComm->allreduce(squared);
              if (squared > 0.0)
                maximum_alpha = std::min(maximum_alpha, training_input_.getMaximumMetricNorm() / std::sqrt(squared));
            }
            if (training_input_.getLineSearch() == WFTrainDriverInput::LineSearch::OFF)
            {
              selected_alpha = (training_input_.getUpdateScale() == 0.0 &&
                                training_input_.getWeightDecay() == 0.0)
                  ? 0.0 : std::min(1.0, maximum_alpha);
              selected_displacement_norm = selected_alpha * norm;
              return selected_alpha;
            }
            std::vector<std::reference_wrapper<ParticleSet>> configurations;
            std::vector<std::reference_wrapper<TrialWaveFunction>> teachers;
            std::vector<std::size_t> chain_ids;
            std::vector<std::size_t> crowd_chain_offsets(crowds_.size(), 0);
            for (std::size_t crowd_index = 1; crowd_index < crowds_.size(); ++crowd_index)
              crowd_chain_offsets[crowd_index] = crowd_chain_offsets[crowd_index - 1] +
                  crowds_[crowd_index - 1]->size();
            for (auto& samples : retained_samples)
              for (std::size_t i = 0; i < samples.particles.size(); ++i)
              {
                configurations.emplace_back(*samples.particles[i]);
                chain_ids.push_back(crowd_chain_offsets[samples.crowd_index] + i);
                teachers.emplace_back(determinant_overlap ? *samples.teachers[i]
                    : crowds_[samples.crowd_index]->get_walker_twfs()[i].get());
              }
            NativeTimerScope candidate_timer(phase_timers_.candidate);
            psiformer::memory::Phase memory_candidate_timer("candidate");
            selected_alpha = wftrain::selectNativeStep(training_input_, *myComm,
                student_prototype_, population_.get_golden_hamiltonian(), selected_provider_id_,
                configurations, chain_ids, teachers, rngs_[0], qmcdriver_input_.get_tau(), old, full,
                determinant_overlap ? determinant_overlap_objective_.get() : nullptr, maximum_alpha,
                [&](double alpha, double loss, double difference, double error,
                    double ess, double weight, bool valid) {
                  if (myComm->rank() == 0)
                    app_log() << "WFTrain candidate alpha " << alpha << " loss " << loss
                              << " loss_difference " << difference
                              << " difference_error " << error << " ess " << ess
                              << " maximum_weight " << weight << " valid " << valid << std::endl;
                }, conventional_binding_.get()
#ifdef QMC_COMPLEX
                , fixed_grid ? &*fixed_grid : nullptr
#endif
                );
            selected_displacement_norm = selected_alpha * norm;
            return selected_alpha;
          });
    };
    {
      std::unique_ptr<PsiFormerWF::GpuTrainingEpoch> native_real_epoch;
#ifdef QMC_COMPLEX
      std::unique_ptr<PsiFormerComplexWF::GpuTrainingEpoch> native_complex_epoch;
#endif
      bool strict_native_epoch = true;
      std::size_t native_epoch_objective_peak_bytes = 0;
      // Actual production is timed here; the unchanged trainer later times only the merge/update.
      auto completeNativeObjective = [&](wftrain::GradientProducer& source,
          const wftrain::StructuredParameterSnapshot* parameters, std::size_t other_retained)
          -> std::unique_ptr<wftrain::GpuTrainingEpochObjective> {
        if (!native_mixed_training) return {};
        std::unique_ptr<wftrain::GpuTrainingEpochObjective> completed;
        std::exception_ptr failure;
        try
        {
          std::optional<wftrain::StructuredParameterSnapshot> private_parameters;
          if (!parameters)
          {
            private_parameters.emplace(selected_provider_->snapshotParameters());
            parameters = &*private_parameters;
          }
          const auto p = selected_provider_->parameterSchema().parameterCount();
          const auto raw_bytes = checkedMemoryMultiply(p, 3 * sizeof(wftrain::DerivativeValue));
          native_epoch_objective_peak_bytes = checkedMemoryAdd(checkedMemoryMultiply(2, raw_bytes),
              checkedMemoryMultiply(p, sizeof(double))); // Includes trainer destination and one parameter snapshot.
          const auto combined = checkedMemoryAdd(retained_configuration_bytes,
              checkedMemoryAdd(other_retained, native_epoch_objective_peak_bytes));
          if (combined > training_input_.getMaximumRetainedBytes())
            throw std::length_error("Mixed native staged objective and trainer peak exceed retained budget");
          NativeTimerScope timer(phase_timers_.objective);
          psiformer::memory::Phase memory_objective("native_epoch_objective");
          completed = std::make_unique<wftrain::GpuTrainingEpochObjective>(
              selected_provider_->parameterSchema(), *parameters, source, energy_estimator);
          if (completed->retainedBytes() != raw_bytes)
            throw std::logic_error("Native staged objective differs from its planned raw-sum inventory");
        }
        catch (...) { failure = std::current_exception(); }
        requireNativeEpochReadiness(*myComm, failure, strict_native_epoch);
        native_real_epoch.reset();
#ifdef QMC_COMPLEX
        native_complex_epoch.reset();
#endif
        app_log() << "WFTrain admitted native precision objective: raw_sum_bytes=" << completed->retainedBytes()
                  << " staged_and_trainer_peak_bound=" << native_epoch_objective_peak_bytes
                  << " actual_production_passes=1 final_trainer_derivative_passes=0" << std::endl;
        return completed;
      };
      // Reuse the existing ENERGY implementation unchanged for every ordinary FP64 call.
      auto executeEnergyEpoch = [&]() {
        const auto optimizer_kind = training_input_.getOptimizer();
        const bool capture_ecp_jvp_replay =
            optimizer_kind == WFTrainDriverInput::Optimizer::LINEAR_METHOD ||
            optimizer_kind == WFTrainDriverInput::Optimizer::ACCELERATED_LINEAR_METHOD ||
            optimizer_kind ==
                WFTrainDriverInput::Optimizer::PROJECTED_INVERSE_ITERATION ||
            optimizer_kind ==
                WFTrainDriverInput::Optimizer::MINIMUM_PROJECTED_INVERSE_ITERATION;
        const bool has_nonlocal_ecp =
            population_.get_golden_hamiltonian().hasPhysicalNLPP();
        std::vector<wftrain::PreparedEnergyGradientBatch> batches;
        std::exception_ptr preparation_failure;
        try
        {
          NativeTimerScope preparation_timer(phase_timers_.preparation);
          psiformer::memory::Phase memory_preparation_timer("preparation");
          batches.reserve(retained_samples.size());
          std::size_t sample_offset = 0;
          std::size_t batch_ordinal = 0;
          std::shared_ptr<ConventionalReplayWorker> conventional_worker;
          std::shared_ptr<const wftrain::StructuredParameterSnapshot> conventional_parameters;
          if (conventional_binding_)
          {
            conventional_worker = std::make_shared<ConventionalReplayWorker>(
                *conventional_binding_, population_.get_golden_electrons());
            conventional_parameters = std::make_shared<const wftrain::StructuredParameterSnapshot>(
                selected_provider_->snapshotParameters());
          }
          // Prepare immutable score/kinetic owners only after collection. ECP
          // energies and responses are paired on their accepted replay grids.
          for (auto& samples : retained_samples)
          {
            if (jastrow_binding_)
            {
              wftrain::PreparedEnergyGradientBatch batch;
              batch.weights.assign(samples.particles.size(), 1.0);
              for (double energy : samples.local_energies) batch.local_energies.emplace_back(energy);
              batch.local_energy_term_mask = 0;
              RefVector<ParticleSet> positions;
              for (auto& particles : samples.particles) positions.push_back(*particles);
              batch.derivative_operator = jastrow_binding_->makeScoreOperator(positions, batch_ordinal++, sample_offset,
                  training_input_.getParameterChunkSize());
              sample_offset += samples.particles.size();
              batches.push_back(std::move(batch));
              continue;
            }
            if (conventional_binding_)
            {
              if (has_nonlocal_ecp)
              {
                batches.push_back(prepareConventionalECPBatch(
                    *crowds_[samples.crowd_index], samples, *conventional_binding_, conventional_worker,
                    conventional_parameters, batch_ordinal++, sample_offset));
                sample_offset += samples.particles.size();
                continue;
              }
              wftrain::PreparedEnergyGradientBatch batch;
              batch.weights.assign(samples.particles.size(), 1.0);
              for (double energy : samples.local_energies)
                batch.local_energies.emplace_back(energy);
              batch.local_energy_term_mask = wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC);
              batch.derivative_operator = conventional_binding_->adapter().makeDerivativeOperator(
                  std::make_shared<ConventionalReplay>(conventional_worker, samples),
                  conventional_parameters, batch_ordinal++, sample_offset);
              sample_offset += samples.particles.size();
              batches.push_back(std::move(batch));
              continue;
            }
            Crowd& crowd = *crowds_[samples.crowd_index];
            // KFAC reuses only FULL wavefunction state; ECP capture below still
            // evaluates its complete Hamiltonian and draws its accepted grid.
            const bool resident_single_snapshot = (kfac_single_replay ||
                ((resident_adam_budget || resident_sr_budget.first) &&
                 (!has_nonlocal_ecp || resident_device_authority_))) && !complex_training &&
                retained_samples.size() == 1 && crowds_.size() == 1 &&
                num_blocks == 1 && steps_per_block_ == 1;
            const bool fresh_full = replayCrowdSamples(crowd, samples, resident_single_snapshot
                ? std::optional<std::size_t>(training_state.parameter_version) : std::nullopt);
            std::size_t scalar_ecp_budget = 0;
            if (!complex_training && score_training && has_nonlocal_ecp &&
                selected_provider_->usesAcceleratorParameterProducts())
            {
              auto live = checkedMemoryAdd(retained_configuration_bytes,
                  kfac_optimizer ? kfac_optimizer->retainedNumericBytes() : 0);
              live = checkedMemoryAdd(live, first_order_optimizer ? first_order_optimizer->retainedBytes() : 0);
              // Earlier crowds remain live until the global producer finishes.
              // Charge their shared/private numerical owners before the next lane.
              live = checkedMemoryAdd(live, preparedStorageDiagnostics(batches).retained_numeric_bytes);
              if (live >= training_input_.getMaximumRetainedBytes())
                throw std::length_error("GPU score-only ECP batches exceed remaining numeric budget");
              scalar_ecp_budget = training_input_.getMaximumRetainedBytes() - live;
            }
            std::optional<ImmediateFullReplayReceipt> full_handoff;
            std::unique_ptr<wftrain::StreamingDerivativeOperator> prepared_score;
            if ((kfac_ecp_full_handoff || score_ecp_full_handoff) && fresh_full &&
                score_training && has_nonlocal_ecp && !capture_ecp_jvp_replay)
            {
              if (complex_training)
              {
                auto live = checkedMemoryAdd(retained_configuration_bytes,
                    kfac_optimizer ? kfac_optimizer->retainedNumericBytes() : 0);
                live = checkedMemoryAdd(live, first_order_optimizer ? first_order_optimizer->retainedBytes() : 0);
                live = checkedMemoryAdd(live, complex_expected_prepared_bytes);
                live = checkedMemoryAdd(live, checkedMemoryMultiply(2,
                    wftrain::NonLocalEnergyAccumulator::requiredBytes(crowd.size())));
                const auto reserved = std::max(live, complex_handoff_preflight_bytes);
                if (reserved < training_input_.getMaximumRetainedBytes())
                  full_handoff = captureImmediateFullReplay(crowd, samples, training_state.parameter_version,
                      training_input_.getMaximumRetainedBytes() - reserved);
              }
              else
              {
                prepared_score = prepareImmediateGpuScoreBatch(crowd, batch_ordinal, sample_offset,
                                                               training_input_.getParameterChunkSize());
                const auto named = scoreOnlyECPCaptureNamedBytes(*prepared_score, crowd.size());
                if (named < scalar_ecp_budget)
                  full_handoff = captureImmediateFullReplay(crowd, samples, training_state.parameter_version,
                                                            scalar_ecp_budget - named);
                if (full_handoff)
                  scalar_ecp_budget -= full_handoff->retainedBytes();
                else
                  prepared_score.reset();
              }
            }
            batches.push_back(prepareCrowdBatch(
                crowd, selected_provider_id_, batch_ordinal++, sample_offset,
                training_input_.getParameterChunkSize(), has_nonlocal_ecp,
                capture_ecp_jvp_replay, score_training, complex_training,
                complex_gpu_provider_ ? &training_input_.getComplexGpuTraining() : nullptr,
#ifdef QMC_COMPLEX
                fixed_grid ? &*fixed_grid : nullptr,
#else
                nullptr,
#endif
                scalar_ecp_budget, usesInlinePairedECPResponse(training_input_),
                full_handoff ? &*full_handoff : nullptr, std::move(prepared_score)));
            for (std::size_t sample = 0; sample < crowd.size(); ++sample)
            {
              crowd.get_walker_elecs()[sample].get().saveWalker(crowd.get_walkers()[sample]);
              crowd.get_walkers()[sample].get().resetProperty(
                  crowd.get_walker_twfs()[sample].get().getLogPsi(),
                  crowd.get_walker_twfs()[sample].get().getPhase(),
                  batches.back().local_energies[sample].real());
            }
            sample_offset += crowd.size();
          }
          storage_diagnostics = preparedStorageDiagnostics(batches);
          if (conventional_parameters)
          {
            const auto snapshot_bytes = checkedMemoryMultiply(
                conventional_parameters->values.capacity(), sizeof(double));
            // This outer shared owner survives producer/factor construction;
            // charge it once alongside other retained outer-owner allocations.
            retained_configuration_bytes = checkedMemoryAdd(retained_configuration_bytes, snapshot_bytes);
          }
          // PP capture owns variable actual Q within the explicit worst-case bound;
          // the unchanged AE path still requires exact realized preflight equality.
          if (complex_training && (has_nonlocal_ecp
                  ? storage_diagnostics.retained_numeric_bytes > complex_expected_prepared_bytes
                  : storage_diagnostics.retained_numeric_bytes != complex_expected_prepared_bytes))
            throw std::logic_error("WFTrainBatched: paired prepared capacity differs from exact preallocation admission");
          // RSS covers ParticleSet/teacher caches that numeric diagnostics omit.
        }
        catch (...)
        {
          preparation_failure = std::current_exception();
        }

        if (native_mixed_training)
          requireNativeEpochReadiness(*myComm, preparation_failure, strict_native_epoch);
        int preparation_failed = preparation_failure ? 1 : 0;
        myComm->allreduce(preparation_failed);
        if (preparation_failed != 0)
        {
          if (preparation_failure)
            std::rethrow_exception(preparation_failure);
          throw UniformCommunicateError(
              "WFTrainBatched: another rank failed while preparing a live training batch");
        }
        enforceResidentBudget(*myComm, training_input_.getMaximumResidentBytes());

        logTrainingMemoryPhase(*myComm, iteration, "after_preparation_before_update");
        memory_phase.transition("objective_gradients_and_optimizer_update");
        logTrainingAllocation("optimizer_reductions_and_proposal", storage_diagnostics.sample_count,
            selected_provider_->parameterSchema().parameterCount());

        if (isFirstOrder(optimizer_kind) ||
            optimizer_kind == WFTrainDriverInput::Optimizer::STOCHASTIC_RECONFIGURATION)
        {
          int empty_rank = batches.empty() ? 1 : 0;
          myComm->allreduce(empty_rank);
          if (empty_rank != 0)
            throw UniformCommunicateError(
                "WFTrainBatched: First-order methods and parameter-space SR require at least one walker on every rank");
        }
        if (resident_adam_budget || resident_sr_budget.first)
        {
          if (resident_score_adam)
          {
            auto& provider = *dynamic_cast<PsiFormerWF*>(selected_provider_);
            (void)wftrain::realECPPhysicalIdentity(provider, population_.get_golden_electrons(), population_.get_golden_hamiltonian());
          }
          else
            validateResidentCoulomb(population_.get_golden_hamiltonian(), resident_device_authority_);
          const auto resident_mask = resident_score_adam ? 0 :
              wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC) |
              (resident_device_authority_ && has_nonlocal_ecp
                  ? wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP) : 0);
          std::optional<wftrain::StructuredParameterSnapshot> parameters;
          if (!resident_device_authority_)
            parameters = selected_provider_->snapshotParameters();
          const auto source_version = resident_device_authority_ ? device_provider->parameterVersion() : parameters->version;
          const auto& schema = selected_provider_->parameterSchema();
          if (student_prototype_.getOrbitals().size() != 1 || batches.size() != 1 ||
              batches[0].weights.size() < (resident_device_authority_ ? 1 : 7) ||
              (resident_sr_budget.first && batches[0].weights.size() != crowds_.front()->size()) || batches[0].additive_local_energy_vjp ||
              batches[0].local_energy_term_mask != resident_mask ||
              training_state.schema_fingerprint != schema.fingerprint() || training_state.parameter_version != source_version ||
              training_state.completed_iterations == std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("Resident ENERGY requires one current complete physical kinetic/ECP batch/state");
          const auto& batch = batches.front();
          for (const auto energy : batch.local_energies)
            if (energy.imag() != 0)
              throw std::invalid_argument("Resident ENERGY requires real physical local energies");
          std::optional<wftrain::EnergyClippingTransform> clipping;
          if (resident_score_adam && training_input_.getScoreResidualPolicy().transform ==
                  wftrain::ScoreResidualTransform::SOFT_SQUEEZE)
            clipping = makeScoreResidualTransform(*myComm, batches, training_input_,
                checkedMemoryAdd(first_order_optimizer->retainedBytes(), checkedMemoryAdd(
                    retained_configuration_bytes, storage_diagnostics.retained_numeric_bytes)));
          const auto coefficients = wftrain::prepareEnergyCoefficients(*batch.derivative_operator,
              {batch.weights.data(), batch.weights.size()}, {batch.local_energies.data(), batch.local_energies.size()},
              resident_score_adam ? wftrain::EnergyGradientEstimator::SCORE_COVARIANCE
                                  : wftrain::EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN,
              clipping ? &*clipping : nullptr);
          if (coefficients.weight_sum <= 0)
            throw std::invalid_argument("Resident ENERGY requires positive total sample weight");
          const auto generic = coefficients.channels(*batch.derivative_operator, batch.local_energy_term_mask);
          const std::array<wftrain::VJPCoefficientChannel, 3> channels = resident_score_adam
              ? std::array<wftrain::VJPCoefficientChannel, 3>{{generic[1], generic[0], {}}}
              : std::array<wftrain::VJPCoefficientChannel, 3>{{generic[2], generic[1], generic[0]}};
          const std::size_t channel_count = resident_score_adam ? 2 : 3;
          const double score_center = clipping ? coefficients.clipped_weighted_energy_sum.real() / coefficients.weight_sum
                                              : coefficients.weighted_energy_sum.real() / coefficients.weight_sum;
          auto pending = training_state; // Fallible fingerprint copy precedes proposal/publication.
          ++pending.completed_iterations;
          resident_diagnostics.emplace(ResidentEnergyDiagnostics{batch.weights.size(),
              (coefficients.weighted_energy_sum / coefficients.weight_sum).real(), 0, 0});
          resident_diagnostics->variance = std::max(0.0, coefficients.weighted_energy_norm_sum / coefficients.weight_sum -
              std::norm(coefficients.weighted_energy_sum / coefficients.weight_sum));
          if (!wftrain::isFiniteTrainingReal(resident_diagnostics->mean) ||
              !wftrain::isFiniteTrainingReal(resident_diagnostics->variance))
            throw std::invalid_argument("Resident ENERGY scalar diagnostics are nonfinite");
          storage_diagnostics = preparedStorageDiagnostics(batches);
          // The mandatory cold allowance counts the HOST pair once whether it
          // is currently present or vacant; actual retainedBytes stays truthful.
          const auto optimizer_bytes = resident_adam_budget ? first_order_optimizer->retainedBytes() : 0;
          const auto cold_covered_mirrors = resident_device_authority_ && resident_adam_budget
              ? first_order_optimizer->hostMomentMirrorBytes() : 0;
          retained_numeric_bytes = checkedMemoryAdd(storage_diagnostics.retained_numeric_bytes,
              resident_adam_budget ? optimizer_bytes - cold_covered_mirrors : resident_sr_incremental_bytes_);
          enforceMemoryBudget(*myComm, checkedMemoryAdd(retained_numeric_bytes, retained_configuration_bytes), training_input_.getMaximumRetainedBytes());
          bool resident_committed = false;
          try
          {
            NativeTimerScope proposal_timer(phase_timers_.proposal);
            psiformer::memory::Phase memory_proposal_timer("proposal");
            if (resident_device_authority_ && resident_sr_budget.first)
            {
              // The component owns all SR vectors and the finite shared successor;
              // only scalar receipts cross this driver/publication boundary.
              std::unique_ptr<PsiFormerWF::PreparedResidentDeviceSRPublication> publication;
              std::exception_ptr failure;
              try
              {
                auto candidate = device_provider->consumeResidentEnergySRDevice(*batch.derivative_operator,
                    {channels.data(), channel_count}, batch.weights, coefficients.weight_sum,
                    resident_diagnostics->mean, training_input_.getDamping(), training_input_.getLearningRate(),
                    makeKrylovControl(training_input_), resident_sr_budget, resident_snapshot_cap_);
                if (!candidate || candidate->sourceVersion() != source_version)
                  throw std::invalid_argument("Resident DEVICE SR candidate source is incompatible");
                const auto& diagnostics = candidate->diagnostics();
                resident_diagnostics->gradient_norm = diagnostics.gradient_norm;
                selected_displacement_norm = candidate->displacementNorm();
                if (!wftrain::isFiniteTrainingReal(resident_diagnostics->gradient_norm) ||
                    resident_diagnostics->gradient_norm < 0 ||
                    !wftrain::isFiniteTrainingReal(selected_displacement_norm) || selected_displacement_norm < 0)
                  throw std::invalid_argument("Resident DEVICE SR scalar norms are invalid");
                krylov_iterations = diagnostics.solve.iterations;
                krylov_operator_applications = diagnostics.solve.operator_applications;
                applied_scale = 1.0;
                logSRDiagnostics("resident_device", diagnostics.solve, diagnostics.direction_norm,
                    diagnostics.metric_norm, training_input_.getDamping(), 1.0);
                publication = device_provider->prepareResidentDeviceSRPublication(
                    std::move(candidate), source_version, resident_successor_cap_);
                if (!publication)
                  throw std::logic_error("Resident DEVICE SR publication returned no owning action");
              }
              catch (...) { failure = std::current_exception(); }
              reduction.validatePublicationReadiness(source_version, true, failure);
              NativeTimerScope publication_timer(phase_timers_.publication);
              psiformer::memory::Phase memory_publication_timer("publication");
              const auto version = publication->commit();
              resident_committed = true;
              pending.parameter_version = reduction.completePublication(version);
            }
            else if (resident_device_authority_)
            {
              std::unique_ptr<PsiFormerWF::PreparedResidentDeviceAdamPublication> publication;
              std::exception_ptr failure;
              try
              {
                std::unique_ptr<PsiFormerWF::ResidentDeviceAdamProposal> candidate;
                first_order_optimizer->proposeResidentDeviceAdam(schema, source_version,
                  [&](const auto& adam, double b1, double b2, double c1, double c2, double epsilon) {
                    candidate = device_provider->consumeResidentEnergyAdamDevice(*batch.derivative_operator,
                        {channels.data(), channel_count}, adam, coefficients.weight_sum,
                        resident_diagnostics->mean, b1, b2, c1, c2, epsilon, resident_snapshot_cap_);
                    if (!candidate || candidate->sourceVersion() != source_version)
                      throw std::invalid_argument("Resident DEVICE candidate source is incompatible");
                    return candidate->gradientNorm();
                  }, resident_diagnostics->gradient_norm);
                selected_displacement_norm = candidate->displacementNorm();
                if (!wftrain::isFiniteTrainingReal(selected_displacement_norm) || selected_displacement_norm < 0)
                  throw std::invalid_argument("Resident DEVICE displacement norm is invalid");
                publication = device_provider->prepareResidentDeviceAdamPublication(
                    std::move(candidate), source_version, resident_successor_cap_);
                if (!publication)
                  throw std::logic_error("Resident DEVICE publication returned no owning action");
              }
              catch (...) { failure = std::current_exception(); }
              reduction.validatePublicationReadiness(source_version, true, failure);
              NativeTimerScope publication_timer(phase_timers_.publication);
              psiformer::memory::Phase memory_publication_timer("publication");
              const auto version = publication->commit();
              resident_committed = true;
              first_order_optimizer->acceptResidentDeviceAdam(version);
              pending.parameter_version = reduction.completePublication(version);
            }
            else
            {
              wftrain::StructuredParameterSnapshot candidate;
              if (resident_sr_budget.first)
              {
                candidate = *parameters;
                const auto diagnostics = device_provider->consumeResidentEnergySR(*batch.derivative_operator,
                    {channels.data(), channel_count}, batch.weights, parameters->values, coefficients.weight_sum,
                    resident_diagnostics->mean, training_input_.getDamping(), training_input_.getLearningRate(),
                    makeKrylovControl(training_input_), resident_sr_budget, candidate.values);
                resident_diagnostics->gradient_norm = diagnostics.gradient_norm;
                krylov_iterations = diagnostics.solve.iterations;
                krylov_operator_applications = diagnostics.solve.operator_applications;
                applied_scale = 1.0;
                logSRDiagnostics("resident", diagnostics.solve, diagnostics.direction_norm,
                    diagnostics.metric_norm, training_input_.getDamping(), 1.0);
              }
              else
                candidate = first_order_optimizer->proposeResidentAdam(schema, *parameters,
                  [&](auto& adam, double b1, double b2, double c1, double c2, double epsilon, auto& values) {
                    if (resident_score_adam)
                      return device_provider->consumeResidentScoreAdam(*batch.derivative_operator,
                          {channels.data(), channel_count}, adam, parameters->values, coefficients.weight_sum,
                          score_center, b1, b2, c1, c2, epsilon, values);
                    return device_provider->consumeResidentEnergyAdam(*batch.derivative_operator,
                        {channels.data(), channel_count}, adam, parameters->values, coefficients.weight_sum,
                        resident_diagnostics->mean, b1, b2, c1, c2, epsilon, values);
                  }, resident_diagnostics->gradient_norm);
              // Legacy routes retain their owning HOST candidate and full validation.
              wftrain::validateTrainingCandidate(schema, *parameters, candidate);
              for (std::size_t p = 0; p < candidate.values.size(); ++p)
                selected_displacement_norm = std::hypot(selected_displacement_norm, candidate.values[p] - parameters->values[p]);
              if (!wftrain::isFiniteTrainingReal(selected_displacement_norm))
                throw std::overflow_error("Resident ENERGY candidate displacement norm is nonfinite");
              {
                NativeTimerScope publication_timer(phase_timers_.publication);
                psiformer::memory::Phase memory_publication_timer("publication");
                pending.parameter_version = selected_provider_->publishParameters(candidate, parameters->version);
              }
            }
          }
          catch (...)
          {
            if (resident_adam_budget && !resident_committed)
              first_order_optimizer->proposalRejected();
            throw;
          }
          if (resident_adam_budget && !resident_device_authority_)
            first_order_optimizer->acceptResidentAdam();
          training_state.completed_iterations = pending.completed_iterations;
          training_state.parameter_version = pending.parameter_version;
          training_state.schema_fingerprint.swap(pending.schema_fingerprint);
          cache_observer.parametersPublished(training_state.parameter_version);
          iteration_result.completed_iteration = training_state.completed_iterations;
          iteration_result.parameter_version = training_state.parameter_version;
        }
        else if (score_kfac_training)
        {
          const auto& schema = selected_provider_->parameterSchema();
          const auto* affine_provider = dynamic_cast<const PsiFormerWF*>(selected_provider_);
          bool device_affine = false;
          std::size_t device_affine_budget = 0;
          std::optional<wftrain::StructuredParameterSnapshot> parameters;
          std::unique_ptr<wftrain::KFACFactorAccumulator> statistics;
          psiformer::memory::HostCapacity memory_statistics;
          std::unique_ptr<wftrain::PreparedEnergyGradientProducer> producer;
          std::exception_ptr preparation_failure;
          try
          {
            NativeTimerScope timer(phase_timers_.affine_observation);
            psiformer::memory::Phase memory_timer("affine_observation");
            parameters.emplace(selected_provider_->snapshotParameters());
            auto registry = makeLiveKFACRegistry(*selected_provider_);
            device_affine = affine_provider && affine_provider->usesDeviceAffineStatistics() &&
                !registry.affineBlocks().empty();
#ifdef QMC_COMPLEX
            device_affine = device_affine || complex_device_affine;
            if (complex_device_affine)
            {
              device_affine_budget = complex_device_affine_bytes;
              auto live = checkedMemoryAdd(retained_configuration_bytes, storage_diagnostics.retained_numeric_bytes);
              live = checkedMemoryAdd(live, checkedMemoryAdd(device_affine_budget, kfacStatisticsBytes(registry)));
              if (kfac_optimizer) live = checkedMemoryAdd(live, kfac_optimizer->retainedNumericBytes());
              if (live > training_input_.getMaximumRetainedBytes())
                throw std::length_error("Paired DEVICE affine statistics exceed maximum_retained_bytes");
              logTrainingAllocation("paired_kfac_device_affine_combined_bound", 1,
                  schema.parameterCount(), device_affine_budget);
            }
#endif
            if (const auto* real = dynamic_cast<const PsiFormerWF*>(selected_provider_);
                real && real->usesAcceleratorParameterProducts())
            {
              // Admit the selected transient affine owner beside retained products
              // and the existing HOST parameter/snapshot channels before allocation.
              const auto affine_bytes = device_affine ? real->deviceAffineStatisticsStorageBytes(registry) :
                  real->affineScoreStorageBytes();
              device_affine_budget = device_affine ? affine_bytes : 0;
              const auto vectors = checkedMemoryMultiply(schema.parameterCount(), 3 * sizeof(double));
              auto live = checkedMemoryAdd(retained_configuration_bytes, storage_diagnostics.retained_numeric_bytes);
              live = checkedMemoryAdd(live, checkedMemoryAdd(affine_bytes, vectors));
              live = checkedMemoryAdd(live, kfacStatisticsBytes(registry));
              if (kfac_optimizer)
                live = checkedMemoryAdd(live, kfac_optimizer->retainedNumericBytes());
              // Local failures join the common preflight below; never enter a
              // collective from this rank-local allocation/observation try block.
              if (live > training_input_.getMaximumRetainedBytes())
                throw std::length_error("Hybrid KFAC affine storage exceeds maximum_retained_bytes");
              logTrainingAllocation(device_affine ? "hybrid_kfac_device_affine_statistics" : "hybrid_kfac_host_affine",
                  1, schema.parameterCount(), affine_bytes);
            }
            logTrainingAllocation("complex_kfac_transactional_factors", 0, schema.parameterCount(),
                                  kfacStatisticsBytes(registry));
            statistics = std::make_unique<wftrain::KFACFactorAccumulator>(std::move(registry), parameters->version,
                training_input_.getKFACCurvature() == WFTrainDriverInput::KFACCurvature::REGISTERED_GAUSSIAN
                    ? wftrain::KFACCurvatureConvention::REGISTERED_GAUSSIAN
                    : wftrain::KFACCurvatureConvention::NATIVE_SCORE,
                device_affine ? wftrain::KFACAffineStatisticsSource::PRECOMPUTED_SUMS :
                    wftrain::KFACAffineStatisticsSource::OBSERVATION_ROWS);
            if (psiformer::memory::enabled())
              memory_statistics.snapshotAt(__FILE__, __LINE__, "KFAC::factor_observation_statistics",
                  statistics->retainedNumericBytes(), reinterpret_cast<std::uintptr_t>(statistics.get()), -1, parameters->version);
            if (device_affine)
            {
              std::vector<const ParticleSet*> samples;
              for (const auto& retained : retained_samples)
                for (const auto& particles : retained.particles)
                  samples.push_back(particles.get());
              LiveDeviceAffineSink sink(*statistics);
              if (affine_provider)
                affine_provider->accumulateDeviceAffineStatistics(samples, statistics->registry(),
                  statistics->parameterVersion(),
                  statistics->curvatureConvention() == wftrain::KFACCurvatureConvention::REGISTERED_GAUSSIAN,
                  sink, device_affine_budget);
#ifdef QMC_COMPLEX
              else if (complex_device_affine)
                complex_training_component_->accumulateDeviceAffineStatistics(samples, statistics->registry(),
                    statistics->parameterVersion(),
                    statistics->curvatureConvention() == wftrain::KFACCurvatureConvention::REGISTERED_GAUSSIAN,
                    sink, device_affine_budget);
#endif
              else
                throw std::logic_error("DEVICE affine dispatch lacks its selected provider");
              // Precomputed descriptors acquire their HOST numeric bank only at import.
              if (psiformer::memory::enabled())
                memory_statistics.snapshotAt(__FILE__, __LINE__, "KFAC::factor_observation_statistics",
                    statistics->retainedNumericBytes(), reinterpret_cast<std::uintptr_t>(statistics.get()),
                    -1, parameters->version);
            }
            else
            {
              LiveAffineObserver observer(*statistics);
              std::vector<double> real_score, imaginary_score;
              for (const auto& samples : retained_samples)
                for (const auto& particles : samples.particles)
                {
                  statistics->beginSample(1);
                  selected_provider_->observeComplexAffineScores(*particles, observer, real_score, imaginary_score);
                  statistics->addFallbackScores({real_score.data(), real_score.size()},
                                                 {imaginary_score.data(), imaginary_score.size()});
                  statistics->endSample();
                }
            }
            // Physical local energies and score products share the same accepted samples.
            // These explicit score-only estimators need no local-energy parameter responses.
          }
          catch (...) { preparation_failure = std::current_exception(); }
          if (native_mixed_training)
            requireNativeEpochReadiness(*myComm, preparation_failure, strict_native_epoch);
          reduction.preflight(schema, parameters ? &*parameters : nullptr,
                              kfac_estimator, preparation_failure);
          wftrain::reduceKFACFactorStatistics(*statistics, myComm, training_input_.getParameterChunkSize());
          std::optional<wftrain::EnergyClippingTransform> clipping;
          if (training_input_.getKFACObjective() != WFTrainDriverInput::KFACObjective::RAW_COVARIANCE)
            clipping = makeScoreResidualTransform(*myComm, batches, training_input_,
                checkedMemoryAdd(retained_configuration_bytes, checkedMemoryAdd(
                    storage_diagnostics.retained_numeric_bytes, statistics->retainedNumericBytes())));
          preparation_failure = {};
          try { producer = std::make_unique<wftrain::PreparedEnergyGradientProducer>(
              std::move(batches), std::move(clipping), score_training); }
          catch (...) { preparation_failure = std::current_exception(); }
          reduction.preflight(schema, &*parameters, kfac_estimator, preparation_failure);
          storage_diagnostics = producer->storageDiagnostics();
          auto admitted_objective = completeNativeObjective(*producer, &*parameters,
              checkedMemoryAdd(checkedMemoryAdd(storage_diagnostics.retained_numeric_bytes,
                  statistics->retainedNumericBytes()), kfac_optimizer ? kfac_optimizer->retainedNumericBytes() : 0));
          wftrain::GradientProducer& update_producer = admitted_objective ?
              static_cast<wftrain::GradientProducer&>(*admitted_objective) : *producer;
          retained_numeric_bytes = checkedMemoryAdd(native_epoch_objective_peak_bytes,
              checkedMemoryAdd(storage_diagnostics.retained_numeric_bytes, statistics->retainedNumericBytes()));
          if (kfac_optimizer)
          {
            std::exception_ptr inverse_failure;
            try
            {
              NativeTimerScope timer(phase_timers_.kfac_inverse);
              psiformer::memory::Phase memory_timer("kfac_inverse");
              kfac_optimizer->prepareStatistics(*statistics);
            }
            catch (...) { inverse_failure = std::current_exception(); }
            try
            {
              reduction.preflight(schema, &*parameters, kfac_estimator,
                                  inverse_failure);
              retained_numeric_bytes = checkedMemoryAdd(retained_numeric_bytes, kfac_optimizer->retainedNumericBytes());
              enforceMemoryBudget(*myComm, checkedMemoryAdd(retained_configuration_bytes, retained_numeric_bytes),
                                  training_input_.getMaximumRetainedBytes());
              applied_scale = kfac_optimizer->effectiveLearningRate();
              iteration_result = trainer.runIteration(*selected_provider_, update_producer, *decorate(*kfac_optimizer),
                                                       training_state, &cache_observer);
              if (native_mixed_training) native_epoch_replay->acceptPublishedEpoch();
              if (myComm->rank() == 0)
                app_log() << "WFTrain persistent KFAC accepted step=" << kfac_optimizer->step()
                          << " data_seen=" << kfac_optimizer->dataSeen()
                          << " affine_inverse_step=" << kfac_optimizer->inverseStep()
                          << " learning_rate=" << *applied_scale << " constraint_metric=" << kfac_optimizer->constraintMetric()
                          << " constraint_scale=" << kfac_optimizer->constraintScale()
                          << " score_only=yes parameter_response_mask=0" << std::endl;
            }
            catch (...) { kfac_optimizer->proposalRejected(); throw; }
          }
          else
          {
            std::unique_ptr<wftrain::KFACPreconditioner> inverse;
            std::exception_ptr inverse_failure;
            try
            {
              NativeTimerScope timer(phase_timers_.kfac_inverse);
              psiformer::memory::Phase memory_timer("kfac_inverse");
              inverse = std::make_unique<wftrain::KFACPreconditioner>(*statistics,
                  wftrain::KFACPreconditionerControl{training_input_.getDamping(), training_input_.getKFACFallbackDamping(),
                      training_input_.getKFACDampingPolicy() == WFTrainDriverInput::KFACDamping::TRACE_BALANCED
                          ? wftrain::KFACDampingPolicy::TRACE_BALANCED : wftrain::KFACDampingPolicy::EQUAL_SPLIT});
            }
            catch (...) { inverse_failure = std::current_exception(); }
            reduction.preflight(schema, &*parameters, kfac_estimator, inverse_failure);
            retained_numeric_bytes = checkedMemoryAdd(retained_numeric_bytes, inverse->storageDiagnostics().retained_numeric_bytes);
            enforceMemoryBudget(*myComm, checkedMemoryAdd(retained_configuration_bytes, retained_numeric_bytes),
                                training_input_.getMaximumRetainedBytes());
            LiveKFACUpdateRule update_rule(*inverse, training_input_.getLearningRate(),
                training_input_.getKFACConstraint() == WFTrainDriverInput::KFACConstraint::DAMPED_KFAC
                    ? training_input_.getKFACNormConstraint() : 0);
            iteration_result = trainer.runIteration(*selected_provider_, update_producer, *decorate(update_rule),
                                                     training_state, &cache_observer);
            if (native_mixed_training) native_epoch_replay->acceptPublishedEpoch();
            applied_scale = update_rule.constraintScale();
          }
        }
        else if (isFirstOrder(optimizer_kind))
        {
          std::optional<wftrain::EnergyClippingTransform> clipping;
          if (score_training && training_input_.getScoreResidualPolicy().transform ==
                  wftrain::ScoreResidualTransform::SOFT_SQUEEZE)
            clipping = makeScoreResidualTransform(*myComm, batches, training_input_,
                checkedMemoryAdd(first_order_optimizer->retainedBytes(), checkedMemoryAdd(
                    retained_configuration_bytes, storage_diagnostics.retained_numeric_bytes)));
          std::unique_ptr<wftrain::PreparedEnergyGradientProducer> producer;
          std::exception_ptr producer_failure;
          try
          {
            producer = std::make_unique<wftrain::PreparedEnergyGradientProducer>(
                std::move(batches), std::move(clipping), score_training);
          }
          catch (...)
          {
            producer_failure = std::current_exception();
          }
          int producer_failed = producer_failure ? 1 : 0;
          myComm->allreduce(producer_failed);
          if (producer_failed != 0)
          {
            if (producer_failure)
              std::rethrow_exception(producer_failure);
            throw UniformCommunicateError(
                "WFTrainBatched: another rank failed while preparing the first-order objective");
          }
          storage_diagnostics = producer->storageDiagnostics();
          std::unique_ptr<wftrain::GpuTrainingEpochObjective> admitted_objective;
          if (native_mixed_training)
            admitted_objective = completeNativeObjective(*producer, nullptr,
                checkedMemoryAdd(storage_diagnostics.retained_numeric_bytes, first_order_optimizer->retainedBytes()));
          wftrain::GradientProducer& update_producer = admitted_objective ?
              static_cast<wftrain::GradientProducer&>(*admitted_objective) : *producer;
          retained_numeric_bytes = checkedMemoryAdd(native_epoch_objective_peak_bytes,
              checkedMemoryAdd(storage_diagnostics.retained_numeric_bytes, first_order_optimizer->retainedBytes()));
          const auto aggregate = checkedMemoryAdd(retained_numeric_bytes, retained_configuration_bytes);
          enforceMemoryBudget(*myComm, aggregate,
                              training_input_.getMaximumRetainedBytes());
          iteration_result = trainer.runIteration(*selected_provider_, update_producer,
                                                  *decorate(*first_order_optimizer), training_state,
                                                  &cache_observer);
          if (native_mixed_training) native_epoch_replay->acceptPublishedEpoch();
        }
        else if (optimizer_kind ==
                 WFTrainDriverInput::Optimizer::STOCHASTIC_RECONFIGURATION)
        {
          std::unique_ptr<wftrain::PreparedEnergyGradientProducer> producer;
          std::vector<wftrain::StochasticReconfigurationBatch> sr_batches;
          std::unique_ptr<wftrain::StochasticReconfigurationOperator> covariance;
          std::unique_ptr<wftrain::IdentityPreconditioner> preconditioner;
          std::unique_ptr<wftrain::StochasticReconfigurationUpdateRule> update_rule;
          std::exception_ptr sr_preparation_failure;
          try
          {
            producer = std::make_unique<wftrain::PreparedEnergyGradientProducer>(
                std::move(batches));
            sr_batches = makeSRBatches(producer->batches());
            const auto parameters = selected_provider_->snapshotParameters();
            covariance = std::make_unique<wftrain::StochasticReconfigurationOperator>(
                selected_provider_->parameterSchema(), parameters.version,
                wftrain::DerivativeArrayView<const wftrain::StochasticReconfigurationBatch>(
                    sr_batches.data(), sr_batches.size()),
                reduction);
            preconditioner = std::make_unique<wftrain::IdentityPreconditioner>(
                selected_provider_->parameterSchema(), parameters.version);
            update_rule = std::make_unique<wftrain::StochasticReconfigurationUpdateRule>(
                *covariance, *preconditioner, makeSRControl(training_input_));
          }
          catch (...)
          {
            sr_preparation_failure = std::current_exception();
          }
          int sr_preparation_failed = sr_preparation_failure ? 1 : 0;
          myComm->allreduce(sr_preparation_failed);
          if (sr_preparation_failed != 0)
          {
            if (sr_preparation_failure)
              std::rethrow_exception(sr_preparation_failure);
            throw UniformCommunicateError(
                "WFTrainBatched: another rank failed while preparing parameter-space SR");
          }

          storage_diagnostics = producer->storageDiagnostics();
          retained_numeric_bytes = checkedMemoryAdd(
              storage_diagnostics.retained_numeric_bytes,
              covariance->storageDiagnostics().retained_numeric_bytes);
          retained_numeric_bytes = checkedMemoryAdd(
              retained_numeric_bytes,
              parameterSpaceSRWorkspaceBytes(
                  selected_provider_->parameterSchema().parameterCount()));
          enforceMemoryBudget(*myComm, checkedMemoryAdd(retained_numeric_bytes, retained_configuration_bytes),
                              training_input_.getMaximumRetainedBytes());
          iteration_result = trainer.runIteration(*selected_provider_, *producer,
                                                  *decorate(*update_rule), training_state,
                                                  &cache_observer);
          if (const auto& diagnostics = update_rule->lastDiagnostics())
          {
            krylov_iterations            = diagnostics->solve.iterations;
            krylov_operator_applications = diagnostics->solve.operator_applications;
            applied_scale                = diagnostics->applied_scale;
            logSRDiagnostics("host", diagnostics->solve, diagnostics->direction_norm,
                diagnostics->metric_norm, diagnostics->damping, diagnostics->applied_scale);
          }
        }
        else
        {
          const auto& schema = selected_provider_->parameterSchema();
          std::optional<wftrain::StructuredParameterSnapshot> parameters;
          std::exception_ptr factor_preflight_failure;
          try
          {
            parameters.emplace(selected_provider_->snapshotParameters());
            if (parameters->schema_fingerprint != schema.fingerprint() ||
                parameters->values.size() != schema.parameterCount())
              throw std::logic_error(
                  "WFTrainBatched: structured provider returned an inconsistent snapshot");
            if (training_state.completed_iterations != 0 &&
                (training_state.schema_fingerprint != schema.fingerprint() ||
                 training_state.parameter_version != parameters->version))
              throw std::logic_error(
                  "WFTrainBatched: training state does not match the parameter provider");
            validateScoreBatches(schema, parameters->version, batches);
          }
          catch (...)
          {
            factor_preflight_failure = std::current_exception();
          }
          reduction.preflight(schema, parameters ? &*parameters : nullptr,
                              wftrain::EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN,
                              factor_preflight_failure);

          // The distributed sample-space factor deliberately accepts an empty local
          // batch list; zero-walker ranks still execute every global reduction.
          const auto score_batches = makeScoreBatches(batches);
          std::optional<std::vector<wftrain::DerivativeValue>> objective_residual;
          std::optional<wftrain::ScoreObjectiveResidualOverride> objective_override;
          if (optimizer_kind == WFTrainDriverInput::Optimizer::KFAC &&
              training_input_.getKFACObjective() == WFTrainDriverInput::KFACObjective::CLIPPED_COVARIANCE)
          {
            objective_residual = makeClippedKFACObjective(*myComm, batches, training_input_,
                checkedMemoryAdd(storage_diagnostics.retained_numeric_bytes, retained_configuration_bytes));
            objective_override = wftrain::ScoreObjectiveResidualOverride{
                {objective_residual->data(), objective_residual->size()}};
          }
          wftrain::ScoreCovarianceFactor factor(
              schema, parameters->version,
              {score_batches.data(), score_batches.size()},
              wftrain::ScoreSampleDomain::REAL, reduction, objective_override);
          objective_residual.reset();
          std::optional<SPRINGFactorBindingGuard> spring_factor_binding;
          if (optimizer_kind == WFTrainDriverInput::Optimizer::SPRING)
          {
            spring_optimizer->bindFactor(factor);
            spring_factor_binding.emplace(*spring_optimizer);
          }
          const auto factor_storage = factor.storageDiagnostics();
          retained_numeric_bytes = checkedMemoryAdd(
              storage_diagnostics.retained_numeric_bytes,
              factor_storage.retained_numeric_bytes);
          if (spring_optimizer)
            retained_numeric_bytes = checkedMemoryAdd(
                retained_numeric_bytes, spring_optimizer->retainedBytes());
          if (optimizer_kind == WFTrainDriverInput::Optimizer::MINIMUM_STEP_SR ||
              optimizer_kind == WFTrainDriverInput::Optimizer::SPRING)
            retained_numeric_bytes = checkedMemoryAdd(
                retained_numeric_bytes,
                minimumStepWorkspaceBytes(optimizer_kind, factor));
          enforceMemoryBudget(*myComm, checkedMemoryAdd(retained_numeric_bytes, retained_configuration_bytes),
                              training_input_.getMaximumRetainedBytes());
          const EnergyMoments energy_moments =
              energyMoments(*myComm, batches, factor.meanEnergy().real());

          if (optimizer_kind == WFTrainDriverInput::Optimizer::LINEAR_METHOD ||
              optimizer_kind == WFTrainDriverInput::Optimizer::ACCELERATED_LINEAR_METHOD)
          {
            const bool accelerated = optimizer_kind == WFTrainDriverInput::Optimizer::ACCELERATED_LINEAR_METHOD;
            const auto mask = completeLocalEnergyTermMask(batches, has_nonlocal_ecp);
            wftrain::SampledLinearMethodOperator h(factor, {score_batches.data(), score_batches.size()},
                mask, false, !accelerated, training_input_.getDamping());
            wftrain::SampledLinearMethodOperator s(factor, {score_batches.data(), score_batches.size()},
                mask, true, false, training_input_.getLMOverlapShift());
            wftrain::IdentityPreconditioner preconditioner(schema, parameters->version);
            wftrain::MatrixFreeLinearMethodControl lm_control;
            lm_control.relative_residual_tolerance = training_input_.getKrylovRelativeTolerance();
            lm_control.absolute_residual_tolerance = training_input_.getKrylovAbsoluteTolerance();
            lm_control.maximum_iterations = training_input_.getKrylovMaximumIterations();
            lm_control.maximum_operator_applications = training_input_.getKrylovMaximumOperatorApplications();
            lm_control.maximum_seconds = training_input_.getKrylovMaximumSeconds();
            lm_control.maximum_subspace_dimension = training_input_.getLMMaximumSubspaceDimension();
            std::vector<wftrain::DerivativeValue> initial(schema.parameterCount() + 1);
            initial[0] = 1.0;
            retained_numeric_bytes = checkedMemoryAdd(retained_numeric_bytes,
                checkedMemoryAdd(h.retainedBytes(), s.retainedBytes()));
            const auto vectors = checkedMemoryAdd(checkedMemoryMultiply(9, lm_control.maximum_subspace_dimension),
                checkedMemoryAdd(30, checkedMemoryMultiply(2, training_input_.getLMInnerRestartLength())));
            const bool fixed_ci = conventional_binding_ && !schema.blocks().empty() &&
                std::all_of(schema.blocks().begin(), schema.blocks().end(),
                    [](const auto& block) { return block.update_group == "legacy_linear"; });
            const bool mixed_ci = conventional_binding_ &&
                std::any_of(schema.blocks().begin(), schema.blocks().end(),
                    [](const auto& block) { return block.trainable && block.update_group == "legacy_linear"; }) &&
                std::any_of(schema.blocks().begin(), schema.blocks().end(),
                    [](const auto& block) { return block.trainable && block.update_group != "legacy_linear"; });
            retained_numeric_bytes = checkedMemoryAdd(retained_numeric_bytes,
                checkedMemoryMultiply(checkedMemoryMultiply(vectors, initial.size()), sizeof(wftrain::DerivativeValue)));
            // Explicit conservative reservation beyond the existing generous LM vector estimate.
            if (mixed_ci)
              retained_numeric_bytes = checkedMemoryAdd(retained_numeric_bytes,
                  checkedMemoryMultiply(checkedMemoryAdd(schema.parameterCount(), factor.localSampleVectorSize()),
                                        sizeof(wftrain::DerivativeValue)));
            // ALM's projected overlap check owns two FP64 subspace-norm arrays.
            // Name their numeric reservation, not an allocator/vendor/RSS peak.
            if (accelerated)
              retained_numeric_bytes = checkedMemoryAdd(retained_numeric_bytes,
                  checkedMemoryMultiply(checkedMemoryMultiply(2, lm_control.maximum_subspace_dimension), sizeof(double)));
            enforceMemoryBudget(*myComm, checkedMemoryAdd(retained_numeric_bytes, retained_configuration_bytes), training_input_.getMaximumRetainedBytes());
            wftrain::MatrixFreeLinearMethodResult result;
            if (accelerated)
            {
              wftrain::AcceleratedLinearMethodControl control;
              control.outer = lm_control;
              control.inner_iterations = training_input_.getLMInnerIterations();
              control.inner_restart_length = training_input_.getLMInnerRestartLength();
              control.inner_relative_tolerance = training_input_.getLMInnerRelativeTolerance();
              result = wftrain::solveAcceleratedMatrixFreeLinearMethod(h, s, preconditioner,
                  {initial.data(), initial.size()}, control);
            }
            else
              result = wftrain::solveSymmetrizedMatrixFreeLinearMethod(h, s, preconditioner,
                  {initial.data(), initial.size()}, lm_control);
            if (!result.converged)
            {
              std::string failure = std::string("Live LM solve failed: ") +
                  wftrain::linearMethodStopReasonName(result.stop_reason);
              const auto& roots = result.ritz_rejections;
              if (accelerated && roots.projected_solves > 0)
                failure += "; ALM Ritz filters: projected_solves=" + std::to_string(roots.projected_solves) +
                    " last_dimension=" + std::to_string(roots.last_dimension) +
                    " last_examined=" + std::to_string(roots.last_examined) +
                    " last_rejected_spectrum=" + std::to_string(roots.last_rejected_spectrum) +
                    " last_rejected_overlap=" + std::to_string(roots.last_rejected_overlap) +
                    " last_rejected_reference=" + std::to_string(roots.last_rejected_reference) +
                    " last_accepted=" + std::to_string(roots.last_accepted);
              throw UniformCommunicateError(failure);
            }
            krylov_iterations = result.iterations;
            krylov_operator_applications = result.operator_applications;
            applied_scale = training_input_.getLearningRate();
            wftrain::LinearMethodUpdateRule update_rule(factor, std::move(result),
                training_input_.getLearningRate(), training_input_.getLMNormalizationXi(),
                mixed_ci ? wftrain::LinearMethodNormalization::MIXED_FIXED_REFERENCE
                         : fixed_ci ? wftrain::LinearMethodNormalization::FIXED_REFERENCE_LINEAR
                                    : wftrain::LinearMethodNormalization::NONLINEAR,
                fixed_ci || mixed_ci ? wftrain::DerivativeArrayView<const wftrain::ScoreCovarianceBatch>{score_batches.data(), score_batches.size()}
                         : wftrain::DerivativeArrayView<const wftrain::ScoreCovarianceBatch>{});
            iteration_result = runScoreFactorIteration(*selected_provider_, *parameters, factor,
                *decorate(update_rule), training_state, reduction, cache_observer,
                energy_moments.weight_sum, energy_moments.variance);
          }
          else if (optimizer_kind == WFTrainDriverInput::Optimizer::KFAC)
          {
            std::unique_ptr<wftrain::KFACFactorAccumulator> statistics;
            psiformer::memory::HostCapacity memory_statistics;
            std::exception_ptr observation_failure;
            try
            {
              NativeTimerScope timer(phase_timers_.affine_observation);
              psiformer::memory::Phase memory_timer("affine_observation");
              statistics = std::make_unique<wftrain::KFACFactorAccumulator>(
                  makeLiveKFACRegistry(*selected_provider_), parameters->version,
                  training_input_.getKFACCurvature() == WFTrainDriverInput::KFACCurvature::REGISTERED_GAUSSIAN
                      ? wftrain::KFACCurvatureConvention::REGISTERED_GAUSSIAN
                      : wftrain::KFACCurvatureConvention::NATIVE_SCORE);
              if (psiformer::memory::enabled())
                memory_statistics.snapshotAt(__FILE__, __LINE__, "KFAC::factor_observation_statistics",
                    statistics->retainedNumericBytes(), reinterpret_cast<std::uintptr_t>(statistics.get()), -1, parameters->version);
              LiveAffineObserver observer(*statistics);
              std::vector<double> score;
              for (const auto& samples : retained_samples)
                for (const auto& particles : samples.particles)
                {
                  statistics->beginSample(1.0);
                  selected_provider_->observeAffineScores(*particles, observer, score);
                  statistics->addFallbackScores({score.data(), score.size()});
                  statistics->endSample();
                }
            }
            catch (...) { observation_failure = std::current_exception(); }
            reduction.preflight(schema, &*parameters,
                wftrain::EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN, observation_failure);
            wftrain::reduceKFACFactorStatistics(*statistics, myComm, training_input_.getParameterChunkSize());
            if (kfac_optimizer)
            {
              std::exception_ptr preparation_failure;
              try
              {
                NativeTimerScope timer(phase_timers_.kfac_inverse);
                psiformer::memory::Phase memory_timer("kfac_inverse");
                kfac_optimizer->prepareStatistics(*statistics);
              }
              catch (...) { preparation_failure = std::current_exception(); }
              try
              {
                reduction.preflight(schema, &*parameters,
                    wftrain::EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN, preparation_failure);
                enforceMemoryBudget(*myComm, checkedMemoryAdd(retained_configuration_bytes,
                    checkedMemoryAdd(retained_numeric_bytes, checkedMemoryAdd(statistics->retainedNumericBytes(),
                        kfac_optimizer->retainedNumericBytes()))), training_input_.getMaximumRetainedBytes());
                const auto rate = kfac_optimizer->effectiveLearningRate();
                iteration_result = runScoreFactorIteration(*selected_provider_, *parameters, factor,
                    *decorate(*kfac_optimizer), training_state, reduction, cache_observer,
                    energy_moments.weight_sum, energy_moments.variance);
                if (myComm->rank() == 0)
                  app_log() << "WFTrain persistent KFAC accepted step=" << kfac_optimizer->step()
                            << " data_seen=" << kfac_optimizer->dataSeen()
                            << " affine_inverse_step=" << kfac_optimizer->inverseStep()
                            << " learning_rate=" << rate << " constraint_metric=" << kfac_optimizer->constraintMetric()
                            << " constraint_scale=" << kfac_optimizer->constraintScale() << std::endl;
              }
              catch (...) { kfac_optimizer->proposalRejected(); throw; }
              retained_numeric_bytes = checkedMemoryAdd(retained_numeric_bytes,
                  checkedMemoryAdd(statistics->retainedNumericBytes(), kfac_optimizer->retainedNumericBytes()));
            }
            else
            {
            wftrain::KFACPreconditionerControl inverse_control{
                training_input_.getDamping(), training_input_.getKFACFallbackDamping()};
            inverse_control.damping_policy =
                training_input_.getKFACDampingPolicy() == WFTrainDriverInput::KFACDamping::TRACE_BALANCED
                    ? wftrain::KFACDampingPolicy::TRACE_BALANCED : wftrain::KFACDampingPolicy::EQUAL_SPLIT;
            auto inverse = [&]() {
              NativeTimerScope timer(phase_timers_.kfac_inverse);
              psiformer::memory::Phase memory_timer("kfac_inverse");
              return wftrain::KFACPreconditioner(*statistics, inverse_control);
            }();
            retained_numeric_bytes = checkedMemoryAdd(retained_numeric_bytes,
                checkedMemoryAdd(statistics->retainedNumericBytes(), inverse.storageDiagnostics().retained_numeric_bytes));
            enforceMemoryBudget(*myComm, checkedMemoryAdd(retained_numeric_bytes, retained_configuration_bytes), training_input_.getMaximumRetainedBytes());
            LiveKFACUpdateRule update_rule(inverse, training_input_.getLearningRate(),
                training_input_.getKFACConstraint() == WFTrainDriverInput::KFACConstraint::DAMPED_KFAC
                    ? training_input_.getKFACNormConstraint() : 0.0);
            iteration_result = runScoreFactorIteration(*selected_provider_, *parameters, factor,
                *decorate(update_rule), training_state, reduction, cache_observer,
                energy_moments.weight_sum, energy_moments.variance);
            if (myComm->rank() == 0)
              app_log() << "WFTrain KFAC constraint metric=" << update_rule.constraintMetric()
                        << " applied_scale=" << update_rule.constraintScale()
                        << " euclidean_cap=" << training_input_.getMaximumUpdateNorm() << std::endl;
            }
          }
          else if (optimizer_kind == WFTrainDriverInput::Optimizer::MINIMUM_STEP_SR)
          {
            wftrain::MinimumStepSRUpdateRule update_rule(
                factor, makeMinimumStepControl(training_input_));
            iteration_result = runScoreFactorIteration(
                *selected_provider_, *parameters, factor, *decorate(update_rule),
                training_state, reduction, cache_observer,
                energy_moments.weight_sum, energy_moments.variance);
            if (const auto& diagnostics = update_rule.lastDiagnostics())
            {
              krylov_iterations            = diagnostics->solve.iterations;
              krylov_operator_applications = diagnostics->solve.operator_applications;
              applied_scale                = diagnostics->applied_scale;
            }
          }
          else if (optimizer_kind == WFTrainDriverInput::Optimizer::SPRING)
          {
            iteration_result = runScoreFactorIteration(
                *selected_provider_, *parameters, factor, *decorate(*spring_optimizer),
                training_state, reduction, cache_observer,
                energy_moments.weight_sum, energy_moments.variance);
            if (const auto& diagnostics = spring_optimizer->lastDiagnostics())
            {
              krylov_iterations            = diagnostics->solve.iterations;
              krylov_operator_applications = diagnostics->solve.operator_applications;
              applied_scale                = diagnostics->applied_scale;
            }
          }
          else if (optimizer_kind ==
                       WFTrainDriverInput::Optimizer::PROJECTED_INVERSE_ITERATION ||
                   optimizer_kind ==
                       WFTrainDriverInput::Optimizer::MINIMUM_PROJECTED_INVERSE_ITERATION)
          {
            wftrain::ProjectedInverseIterationFactor pii_factor(
                factor, completeLocalEnergyTermMask(batches, has_nonlocal_ecp));
            const wftrain::PIIUpdateControl control =
                wftrain::makePIIUpdateControl(training_input_);
            retained_numeric_bytes = checkedMemoryAdd(
                retained_numeric_bytes, pii_factor.retainedBytes());
            retained_numeric_bytes = checkedMemoryAdd(
                retained_numeric_bytes,
                piiWorkspaceBytes(optimizer_kind, factor,
                                  control.gmres.restart_length));
            enforceMemoryBudget(*myComm, checkedMemoryAdd(retained_numeric_bytes, retained_configuration_bytes),
                                training_input_.getMaximumRetainedBytes());

            if (optimizer_kind ==
                WFTrainDriverInput::Optimizer::PROJECTED_INVERSE_ITERATION)
            {
              wftrain::ProjectedInverseIterationUpdateRule update_rule(
                  pii_factor, control);
              iteration_result = runScoreFactorIteration(
                  *selected_provider_, *parameters, factor, *decorate(update_rule),
                  training_state, reduction, cache_observer,
                  energy_moments.weight_sum, energy_moments.variance);
              if (const auto& diagnostics = update_rule.lastDiagnostics())
              {
                krylov_iterations            = diagnostics->iterations;
                krylov_operator_applications = diagnostics->operator_applications;
                applied_scale                = diagnostics->applied_scale;
              }
            }
            else
            {
              wftrain::MinimumProjectedInverseIterationUpdateRule update_rule(
                  pii_factor, control);
              iteration_result = runScoreFactorIteration(
                  *selected_provider_, *parameters, factor, *decorate(update_rule),
                  training_state, reduction, cache_observer,
                  energy_moments.weight_sum, energy_moments.variance);
              if (const auto& diagnostics = update_rule.lastDiagnostics())
              {
                krylov_iterations            = diagnostics->iterations;
                krylov_operator_applications = diagnostics->operator_applications;
                applied_scale                = diagnostics->applied_scale;
              }
            }
          }
          else
            throw std::logic_error(
                "WFTrainBatched: unsupported optimizer reached score-factor execution");
        }
            };

      if (resident_open_observation_)
      {
        if (!resident_observations || resident_observations->sourceVersion() != training_state.parameter_version ||
            resident_observations->sampleCount() != 2)
          throw UniformCommunicateError("Resident open iteration has no current W2 complete observations");
        const auto source_version = training_state.parameter_version;
        auto pending = training_state;
        if (pending.completed_iterations == std::numeric_limits<std::size_t>::max())
          throw UniformCommunicateError("Resident open completed iteration count overflow");
        ++pending.completed_iterations;
        const double e0 = resident_observations->energy(0)[4], e1 = resident_observations->energy(1)[4];
        const double mean = (e0 + e1) / 2.0;
        const double variance = ((e0 - mean) * (e0 - mean) + (e1 - mean) * (e1 - mean)) / 2.0;
        if (!wftrain::isFiniteTrainingReal(mean) || !wftrain::isFiniteTrainingReal(variance))
          throw std::invalid_argument("Resident open iteration energy moments are nonfinite");
        resident_diagnostics.emplace(ResidentEnergyDiagnostics{2, mean, variance, 0.0});
        std::unique_ptr<PsiFormerWF::PreparedResidentDeviceAdamPublication> publication;
        std::exception_ptr failure;
        try
        {
          std::unique_ptr<PsiFormerWF::ResidentDeviceAdamProposal> candidate;
          first_order_optimizer->proposeResidentDeviceAdam(selected_provider_->parameterSchema(), source_version,
              [&](const auto& adam, double b1, double b2, double c1, double c2, double epsilon) {
                candidate = device_provider->consumeResidentOpenAdamDevice(*resident_observations, adam,
                    b1, b2, c1, c2, epsilon, resident_snapshot_cap_);
                if (!candidate || candidate->sourceVersion() != source_version)
                  throw std::invalid_argument("Resident open Adam candidate is foreign or stale");
                return candidate->gradientNorm();
              }, resident_diagnostics->gradient_norm);
          selected_displacement_norm = candidate->displacementNorm();
          if (!wftrain::isFiniteTrainingReal(mean) || !wftrain::isFiniteTrainingReal(variance) ||
              !wftrain::isFiniteTrainingReal(selected_displacement_norm) || selected_displacement_norm < 0)
            throw std::invalid_argument("Resident open iteration scalar result is nonfinite");
          publication = device_provider->prepareResidentDeviceAdamPublication(
              std::move(candidate), source_version, resident_successor_cap_);
          if (!publication)
            throw std::logic_error("Resident open Adam publication returned no owning action");
        }
        catch (...) { failure = std::current_exception(); }
        try
        {
          reduction.validatePublicationReadiness(source_version, true, failure);
          const auto version = publication->commit();
          first_order_optimizer->acceptResidentDeviceAdam(version);
          pending.parameter_version = reduction.completePublication(version);
        }
        catch (...)
        {
          first_order_optimizer->proposalRejected();
          throw;
        }
        training_state.completed_iterations = pending.completed_iterations;
        training_state.parameter_version = pending.parameter_version;
        training_state.schema_fingerprint.swap(pending.schema_fingerprint);
        cache_observer.parametersPublished(training_state.parameter_version);
        iteration_result.completed_iteration = training_state.completed_iterations;
        iteration_result.parameter_version = training_state.parameter_version;
        resident_observations.reset(); // Retire old sample/image receipts before the next accepted epoch.
      }
      else if (orbital_mse)
      {
        std::unique_ptr<wftrain::OrbitalPretrainingProducer> producer;
        std::size_t producer_retained_bytes = 0;
        std::unique_ptr<wftrain::OrbitalPretrainingCoordinator> coordinator;
        std::unique_ptr<wftrain::SelectedUpdateRule> update;
        std::optional<wftrain::TrainingStageState> coordinator_stage;
        std::optional<wftrain::StructuredParameterSnapshot> parameters;
        std::exception_ptr preparation_failure;
        try
        {
          NativeTimerScope preparation_timer(phase_timers_.preparation);
          psiformer::memory::Phase memory_preparation_timer("preparation");
          parameters.emplace(selected_provider_->snapshotParameters());
#ifdef QMC_COMPLEX
          if (complex_orbital_target_)
          {
            auto typed = std::make_unique<SampledOrbitalPretrainingProducer<PsiFormerComplexWF,
                psiformer::ComplexOrbitalTargetEvaluator>>(*complex_training_component_, *complex_orbital_target_, retained_samples,
                    training_input_.getOrbitalNormalization());
            producer_retained_bytes = typed->retainedBytes();
            producer = std::move(typed);
          }
          else
#endif
          {
            auto* student = dynamic_cast<PsiFormerWF*>(selected_provider_);
            if (jastrow_binding_)
            {
              auto typed = std::make_unique<SampledOrbitalPretrainingProducer<wftrain::JastrowNeuralTrainingBinding,
                  psiformer::OrbitalTargetEvaluator>>(*jastrow_binding_, *orbital_target_, retained_samples);
              producer_retained_bytes = typed->retainedBytes();
              producer = std::move(typed);
            }
            else
            {
              if (!student)
                throw UniformCommunicateError("WFTrainBatched: orbital_mse requires a PsiFormer provider");
              auto typed = std::make_unique<SampledOrbitalPretrainingProducer<PsiFormerWF,
                  psiformer::OrbitalTargetEvaluator>>(*student, *orbital_target_, retained_samples);
              producer_retained_bytes = typed->retainedBytes();
              producer = std::move(typed);
            }
          }
          coordinator = std::make_unique<wftrain::OrbitalPretrainingCoordinator>(
              orbital_target_fingerprint_, reduction);
          coordinator_stage.emplace(coordinator->makeStageState(stage_state.stage_ordinal));
          coordinator_stage->completed_stage_iterations = stage_state.completed_stage_iterations;
          update = decorate(*first_order_optimizer);
        }
        catch (...)
        {
          preparation_failure = std::current_exception();
        }
        reduction.preflightOrbital(selected_provider_->parameterSchema(),
            parameters ? &*parameters : nullptr, orbital_target_fingerprint_,
            wftrain::OrbitalPretrainingCoordinator::LOSS_FINGERPRINT, preparation_failure);
        // The outer checkpoint binds all driver controls; the coordinator independently
        // validates its exact target/loss identity and owns failure-atomic publication.
        retained_numeric_bytes = checkedMemoryAdd(first_order_optimizer->retainedBytes(),
            checkedMemoryAdd(producer_retained_bytes,
                checkedMemoryAdd(orbital_target_bytes_, complex_orbital_score_bytes_)));
        retained_numeric_bytes = checkedMemoryAdd(retained_numeric_bytes,
            checkedMemoryMultiply(selected_provider_->parameterSchema().parameterCount(), 4 * sizeof(double)));
        enforceMemoryBudget(*myComm, checkedMemoryAdd(retained_numeric_bytes, retained_configuration_bytes),
                            training_input_.getMaximumRetainedBytes());
        parameters.reset();
        auto result = coordinator->runIteration(*selected_provider_, *producer,
            *update, training_state, *coordinator_stage, &cache_observer);
        stage_state.completed_stage_iterations = coordinator_stage->completed_stage_iterations;
        iteration_result.completed_iteration = result.completed_iteration;
        iteration_result.parameter_version = result.parameter_version;
        orbital_diagnostics.emplace(std::move(result.objective));
      }
      else if (determinant_overlap)
      {
        const auto& schema = selected_provider_->parameterSchema();
        std::optional<wftrain::StructuredParameterSnapshot> parameters;
        std::vector<std::unique_ptr<psiformer::DeterminantChannelBatch>> batch_owners;
        std::vector<wftrain::DeterminantOverlapBatchView> overlap_batches;
        std::exception_ptr preparation_failure;
        try
        {
          parameters.emplace(selected_provider_->snapshotParameters());
          if (parameters->schema_fingerprint != schema.fingerprint() ||
              parameters->values.size() != schema.parameterCount())
            throw std::logic_error(
                "WFTrainBatched: structured provider returned an inconsistent snapshot");
          if (training_state.completed_iterations != 0 &&
              (training_state.schema_fingerprint != schema.fingerprint() ||
               training_state.parameter_version != parameters->version))
            throw std::logic_error(
                "WFTrainBatched: training state does not match the parameter provider");
          if (training_state.completed_iterations ==
              std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("WFTrainBatched: training iteration count overflow");
          if (overlap_student_twfs_.size() != crowds_.size())
            throw std::logic_error(
                "WFTrainBatched: student-clone crowd topology changed after initialization");

          batch_owners.reserve(retained_samples.size());
          overlap_batches.reserve(retained_samples.size());
          for (auto& samples : retained_samples)
          {
            auto& students = overlap_student_twfs_[samples.crowd_index];
            RefVectorWithLeader<TrialWaveFunction> student_batch(*students.front());
            RefVectorWithLeader<ParticleSet> particle_batch(*samples.particles.front());
            for (std::size_t sample = 0; sample < students.size(); ++sample)
            {
              student_batch.push_back(*students[sample]);
              particle_batch.push_back(*samples.particles[sample]);
            }
            batch_owners.push_back(
                TrialWaveFunction::makeStructuredDeterminantChannelBatch(
                    student_batch, particle_batch, selected_provider_id_));

            wftrain::DeterminantOverlapBatchView view;
            view.student = batch_owners.back().get();
            view.teachers.reserve(students.size());
            view.electrons.reserve(students.size());
            for (std::size_t sample = 0; sample < students.size(); ++sample)
            {
              view.teachers.emplace_back(*samples.teachers[sample]);
              view.electrons.emplace_back(*samples.particles[sample]);
            }
            overlap_batches.push_back(std::move(view));
          }
        }
        catch (...)
        {
          preparation_failure = std::current_exception();
        }

        reduction.preflightDeterminantOverlap(
            schema, parameters ? &*parameters : nullptr,
            determinant_overlap_objective_->targetDescriptor().fingerprint(),
            determinant_overlap_objective_->configurationFingerprint(),
            preparation_failure);

        wftrain::DeterminantOverlapObjectiveResult objective =
            determinant_overlap_objective_->evaluate(*myComm, overlap_batches);
        if (objective.parameter_version != parameters->version ||
            objective.parameter_gradient.size() != schema.parameterCount() ||
            objective.target_fingerprint !=
                determinant_overlap_objective_->targetDescriptor().fingerprint() ||
            objective.target_provenance != training_input_.getTargetProvenance())
          throw UniformCommunicateError(
              "WFTrainBatched: determinant-overlap objective returned inconsistent identity");
        if (objective.retained_numeric_bytes >
            std::numeric_limits<std::size_t>::max() - first_order_optimizer->retainedBytes())
          throw std::overflow_error(
              "WFTrainBatched: determinant-overlap retained-byte count overflowed");
        retained_numeric_bytes =
            objective.retained_numeric_bytes + first_order_optimizer->retainedBytes();
        enforceMemoryBudget(*myComm, checkedMemoryAdd(retained_numeric_bytes, retained_configuration_bytes),
                            training_input_.getMaximumRetainedBytes());

        const wftrain::ParameterGradientView gradient{
            schema.fingerprint(), parameters->version, wftrain::ReductionDomain::GLOBAL,
            {objective.parameter_gradient.data(), objective.parameter_gradient.size()}};
        const std::size_t committed_version = wftrain::completeParameterUpdate(
            *selected_provider_, *parameters, gradient, *decorate(*first_order_optimizer), reduction);
        ++training_state.completed_iterations;
        training_state.parameter_version = committed_version;
        training_state.schema_fingerprint = schema.fingerprint();
        cache_observer.parametersPublished(committed_version);
        iteration_result.completed_iteration = training_state.completed_iterations;
        iteration_result.parameter_version = committed_version;
        overlap_diagnostics.emplace(std::move(objective));
      }
      else
      {
        if (!native_mixed_training)
          executeEnergyEpoch();
        else
          for (std::size_t attempt = 0; attempt < 2; ++attempt)
          {
            strict_native_epoch = attempt == 0;
            const auto mode = strict_native_epoch ? psiformer::PsiFormerGpuTrainingEpochMode::STRICT_MIXED
                                                 : psiformer::PsiFormerGpuTrainingEpochMode::FP64_REPLAY;
            const auto predecessor_completed_iterations = training_state.completed_iterations;
            try
            {
              std::exception_ptr entry_failure;
              try
              {
#ifdef QMC_COMPLEX
                if (complex_training_component_)
                  native_complex_epoch = complex_training_component_->enterGpuTrainingEpoch(training_state.parameter_version, mode);
                else
#endif
                  native_real_epoch = device_provider->enterGpuTrainingEpoch(training_state.parameter_version, mode);
              }
              catch (...) { entry_failure = std::current_exception(); }
              requireNativeEpochReadiness(*myComm, entry_failure, strict_native_epoch);
              executeEnergyEpoch();
              native_real_epoch.reset();
#ifdef QMC_COMPLEX
              native_complex_epoch.reset();
#endif
              native_epoch_replay->acceptPublishedEpoch();
              break;
            }
            catch (const NativeGpuEpochRetry&)
            {
              native_real_epoch.reset();
#ifdef QMC_COMPLEX
              native_complex_epoch.reset();
#endif
              if (attempt != 0) throw;
              native_epoch_replay->rewindForFp64Replay();
              native_epoch_objective_peak_bytes = 0;
              storage_diagnostics = {};
              retained_numeric_bytes = 0;
              memory_phase.transition("mixed_epoch_FP64_replay");
              app_log() << "WFTrain complete native precision replay: source_version=" << training_state.parameter_version
                        << " scope=post_MC_pre_EL retained_configs=yes all_ranks=yes extra_MC_moves=0"
                           " pending_factors_and_objective_discarded=yes" << std::endl;
            }
            catch (...)
            {
              // A successful transaction must never be followed by predecessor RNG rollback.
              if (training_state.completed_iterations != predecessor_completed_iterations)
                native_epoch_replay->acceptPublishedEpoch();
              throw;
            }
          }
      }
    }

    memory_phase.transition("candidate_publication_and_refresh");
    // All derivative owners for the old version are gone before cache replacement.
    if (!cache_observer.dirty() ||
        cache_observer.publishedVersion() != iteration_result.parameter_version)
      throw UniformCommunicateError(
          "WFTrainBatched: parameter publication did not arm the cache-refresh barrier");
    if (selected_alpha != 0.0)
    {
      if (!resident_open_observation_)
        refreshPublishedParameters(iteration_result.parameter_version);
      ++accepted_updates;
    }
    if (resident_device_authority_)
      device_provider->residentDeviceAuthorityTransferWindow("sampling_through_refresh", false);
    // Rejection leaves every live parameter/cache valid. In particular, avoid
    // consuming production quadrature RNG merely to refresh the unchanged ECP state.
    cache_observer.clear();
    stage_state.completed_stage_iterations = iteration_result.completed_iteration;
    logTrainingMemoryPhase(*myComm, iteration, "after_update");
    enforceResidentBudget(*myComm, training_input_.getMaximumResidentBytes());

    if (myComm->rank() == 0)
    {
      app_log() << std::setprecision(12)
                << "WFTrain iteration " << iteration
                << " version " << iteration_result.parameter_version;
      if (orbital_diagnostics)
        app_log() << " samples " << orbital_diagnostics->sample_count
                  << " orbital_loss " << orbital_diagnostics->mean_loss
                  << " gradient_norm " << gradientNorm(orbital_diagnostics->gradient);
      else if (overlap_diagnostics)
      {
        double minimum_fidelity = 1.0;
        double minimum_target_effective_samples =
            std::numeric_limits<double>::infinity();
        double minimum_student_effective_samples =
            std::numeric_limits<double>::infinity();
        for (const auto& channel : overlap_diagnostics->channels)
        {
          minimum_fidelity = std::min(minimum_fidelity, channel.fidelity);
          minimum_target_effective_samples = std::min(
              minimum_target_effective_samples, channel.target_effective_samples);
          minimum_student_effective_samples = std::min(
              minimum_student_effective_samples, channel.student_effective_samples);
        }
        app_log() << " samples " << overlap_diagnostics->sample_count
                  << " overlap_loss " << overlap_diagnostics->loss
                  << " minimum_fidelity " << minimum_fidelity
                  << " minimum_target_effective_samples "
                  << minimum_target_effective_samples
                  << " minimum_student_effective_samples "
                  << minimum_student_effective_samples
                  << " gradient_norm "
                  << gradientNorm(overlap_diagnostics->parameter_gradient);
      }
      else if (resident_diagnostics)
        app_log() << " samples " << resident_diagnostics->samples << " mean_energy " << resident_diagnostics->mean
                  << " variance " << resident_diagnostics->variance << " gradient_norm " << resident_diagnostics->gradient_norm;
      else
      {
        const auto& objective = iteration_result.objective;
        app_log() << " samples " << objective.sample_count
                  << " mean_energy " << std::real(objective.mean_energy)
                  << " variance " << objective.energy_variance
                  << " gradient_norm " << gradientNorm(objective.gradient);
      }
      app_log() << " retained_bytes " << checkedMemoryAdd(retained_numeric_bytes, retained_configuration_bytes)
                << " update_scale " << training_input_.getUpdateScale()
                << " selected_alpha " << selected_alpha
                << " effective_scale " << selected_alpha * training_input_.getUpdateScale()
                << " weight_decay " << training_input_.getWeightDecay()
                << " displacement_norm " << selected_displacement_norm
                << " selection_reason " << (selected_alpha == 0.0 ? "rejected" :
                    (training_input_.getLineSearch() == WFTrainDriverInput::LineSearch::OFF
                         ? "fixed_scale" : "reliable_improvement"))
                << " optimizer " << wftrain::optimizerName(training_input_.getOptimizer());
      if (krylov_iterations)
        app_log() << " krylov_iterations " << *krylov_iterations
                  << " krylov_operator_applications "
                  << *krylov_operator_applications
                  << " applied_scale " << *applied_scale;
      app_log() << std::endl;
    }

    // Runtime and exact checkpoints are checked only at complete update boundaries.
    history_failed = false;
    if (myComm->rank() == 0)
    {
      if (!sampling_state.evaluate_energy)
      {
        // The objective is evaluated once on the complete retained population.
        std::ofstream scalar(training_directory / (iteration_name.str() + ".scalar.dat"));
        scalar << "# index OrbitalMSE SampleCount\n" << std::setprecision(17)
               << 0 << ' ' << orbital_diagnostics->mean_loss << ' '
               << orbital_diagnostics->sample_count << '\n';
        scalar.flush();
        history_failed = !scalar;
      }
      iteration_history << std::setprecision(17) << iteration << ','
                        << sample_version << ',' << iteration_result.parameter_version << ','
                        << (orbital_mse ? "orbital_mse" : determinant_overlap ? "determinant_overlap" : "energy") << ','
                        << wftrain::optimizerName(training_input_.getOptimizer()) << ',';
      if (orbital_diagnostics)
        iteration_history << orbital_diagnostics->sample_count << ",,,"
                          << gradientNorm(orbital_diagnostics->gradient) << ',';
      else if (overlap_diagnostics)
        iteration_history << overlap_diagnostics->sample_count << ",,,,"
                          << overlap_diagnostics->loss;
      else if (resident_diagnostics)
        iteration_history << resident_diagnostics->samples << ',' << resident_diagnostics->mean << ','
                          << resident_diagnostics->variance << ',' << resident_diagnostics->gradient_norm << ',';
      else
        iteration_history << iteration_result.objective.sample_count << ','
                          << std::real(iteration_result.objective.mean_energy) << ','
                          << iteration_result.objective.energy_variance << ','
                          << gradientNorm(iteration_result.objective.gradient) << ',';
      iteration_history << ',' << selected_alpha << ',' << (selected_alpha != 0.0) << ','
                        << sampling_seconds << ',' << iteration_wall.elapsed() << ','
                        << checkedMemoryAdd(retained_numeric_bytes, retained_configuration_bytes) << ','
                        << currentResidentBytes() << ',';
      if (krylov_iterations)
        iteration_history << *krylov_iterations;
      iteration_history << ',';
      if (krylov_operator_applications)
        iteration_history << *krylov_operator_applications;
      iteration_history << ',';
      if (orbital_diagnostics)
        iteration_history << orbital_diagnostics->mean_loss;
      iteration_history << '\n';
      iteration_history.flush();
      history_failed = history_failed || !iteration_history;
    }
    myComm->bcast(history_failed);
    if (history_failed)
      throw UniformCommunicateError("WFTrainBatched: failed to write iteration history");
    bool stop_requested = false;
    if (myComm->rank() == 0)
      stop_requested = runtime_control.checkStop(training_loop);
    myComm->bcast(stop_requested);
    if (myComm->rank() == 0)
      app_log() << runtime_control.generateProgressMessage("WFTrainBatched", iteration, num_iterations);
    const bool periodic_checkpoint =
        training_state.completed_iterations % training_input_.getCheckpointInterval() == 0;
    const bool final_block = iteration + 1 == num_iterations;
    if (inference_input_.due(training_state.completed_iterations, final_block))
    {
      if (stop_requested)
      {
        if (myComm->rank() == 0)
          app_log() << "WFTrain inference iteration " << iteration << " parameter_version "
                    << iteration_result.parameter_version << " complete no skipped runtime_stop" << std::endl;
      }
      else
      {
        if (!runInference(static_cast<std::size_t>(iteration), iteration_result.parameter_version))
          stop_requested = true;
        if (myComm->rank() == 0)
          stop_requested = stop_requested || runtime_control.checkStop(training_loop);
        myComm->bcast(stop_requested);
      }
    }
    if (periodic_checkpoint || final_block || stop_requested)
    {
      ScopedTimer checkpoint_timer(timers_.checkpoint_timer);
      psiformer::memory::Phase memory_checkpoint("checkpoint");
      if (resident_open_observation_ && resident_host_dirty)
      {
        // Explicit cold boundary: coordinates first, then complete native tables,
        // FULL, physical H and walker properties before any checkpoint/export reader.
        auto& crowd = *crowds_.front();
        const auto& particles = crowd.get_walker_elecs();
        const auto& wavefunctions = crowd.get_walker_twfs();
        const auto& hamiltonians = crowd.get_walker_hamiltonians();
        RefVectorWithLeader<ParticleSet> p_list(particles[0], particles);
        RefVectorWithLeader<TrialWaveFunction> wf_list(wavefunctions[0], wavefunctions);
        RefVectorWithLeader<QMCHamiltonian> h_list(hamiltonians[0], hamiltonians);
        RefVector<WaveFunctionComponent> components;
        for (TrialWaveFunction& wf : wavefunctions)
          components.push_back(*wf.getOrbitals()[0]);
        ResourceCollectionTeamLock<ParticleSet> p_lock(crowd.getSharedResource().pset_res, p_list);
        ResourceCollectionTeamLock<TrialWaveFunction> wf_lock(crowd.getSharedResource().twf_res, wf_list);
        ResourceCollectionTeamLock<QMCHamiltonian> h_lock(crowd.getSharedResource().ham_res, h_list);
        const auto& leader = dynamic_cast<const PsiFormerWF&>(components[0].get());
        leader.materializeResidentOpenAccepted({components[0], components}, p_list);
        app_log() << "Resident open cold fence: parameter_version=" << training_state.parameter_version
                  << ", samples=2, coordinates=HOST, full_and_hamiltonian_refresh=required" << std::endl;
        for (std::size_t sample = 0; sample < particles.size(); ++sample)
        {
          auto& p = particles[sample].get();
          auto& wf = wavefunctions[sample].get();
          auto& h = hamiltonians[sample].get();
          auto& walker = crowd.get_walkers()[sample].get();
          p.update();
          wf.evaluateLog(p);
          const double energy = h.evaluate(wf, p);
          p.saveWalker(walker);
          walker.resetProperty(wf.getLogPsi(), wf.getPhase(), energy);
          h.saveProperty(walker.getPropertyBase());
          walker.Weight = 1.0;
          walker.wasTouched = false;
        }
        resident_host_dirty = false;
        app_log() << "Resident open cold refresh complete: parameter_version="
                  << training_state.parameter_version << std::endl;
      }
      const wftrain::StructuredParameterProvider* artifact_provider = selected_provider_;
#ifdef QMC_COMPLEX
      // The wrapper delegates the same canonical schema/state/version. Only CONST
      // bundle save/export uses the model owner; training and restore keep the
      // wrapper's explicit budgeted publication authority.
      if (complex_gpu_provider_)
        artifact_provider = complex_training_component_;
#endif
      wftrain::DistributedTrainingCheckpoint::saveIterationBundle(
          get_root_name(), static_cast<std::size_t>(iteration), *myComm, *artifact_provider,
          training_state, stage_state, optimizer_checkpoint, sampler_checkpoint,
          sampler_checkpoint.topology());
      if (myComm->rank() == 0)
        app_log() << "WFTrain checkpointed " << stage_state.completed_stage_iterations
                  << " completed iteration(s) under " << get_root_name()
                  << std::endl;
    }
    if (stop_requested)
    {
      if (myComm->rank() == 0)
        app_log() << runtime_control.generateStopMessage("WFTrainBatched", iteration);
      run_time_manager.markStop();
      break;
    }
  }

  if (myComm->rank() == 0)
    app_log() << "WFTrain completed " << completed_blocks << " block(s) and "
              << accepted_updates << " committed update(s), "
              << training_state.completed_iterations << " completed iteration(s)" << std::endl;
  print_mem("WFTrainBatched ends", app_log());
  wlog_manager.stopRun();
  if (sampling_state.evaluate_energy)
    estimator_manager_->stopDriverRun();
  finalize(completed_blocks, false, false);
}

bool WFTrainBatched::runInference(std::size_t iteration, std::size_t parameter_version)
{
  NativeTimerScope phase(*inference_timer_);
  Timer elapsed;
  const auto directory = std::filesystem::path(get_root_name() + ".inference");
  std::ostringstream name;
  name << std::filesystem::path(get_root_name()).filename().string() << ".i"
       << std::setw(7) << std::setfill('0') << iteration;
  const std::string prefix = (directory / name.str()).string();
  bool failed = false;
  if (myComm->rank() == 0)
    try
    {
      std::filesystem::create_directories(directory);
      if (std::filesystem::exists(prefix + ".scalar.dat"))
        throw std::runtime_error("inference scalar already exists");
    }
    catch (const std::exception& error)
    {
      app_error() << "WFTrain inference output: " << error.what() << std::endl;
      failed = true;
    }
  myComm->bcast(failed);
  if (failed)
    throw UniformCommunicateError("WFTrain inference: conflicting or inaccessible scalar output");

  auto electrons = std::make_unique<ParticleSet>(population_.get_golden_electrons());
  const bool private_md_orbital_ecp = conventional_binding_ && conventional_binding_->hasOrbitalRotations() &&
      !student_prototype_.findMSD().empty() && population_.get_golden_hamiltonian().hasPhysicalNLPP();
  auto wavefunction = private_md_orbital_ecp ? conventional_binding_->makePrivateClone(*electrons)
                                          : student_prototype_.makeClone(*electrons);
  auto hamiltonian = population_.get_golden_hamiltonian().makeClone(*electrons, *wavefunction);
  WalkerConfigurations configurations;
  MCPopulation population(myComm->size(), myComm->rank(), *electrons, *wavefunction, *hamiltonian);
  UPtrVector<RandomBase<FullPrecRealType>> generators;
  RefVector<RandomBase<FullPrecRealType>> refs;
  auto mix = [](std::uint64_t value) {
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
  };
  const auto seed = mix(inference_seed_ ^ UINT64_C(0x696e666572656e63) ^ mix(iteration) ^
                        mix(static_cast<std::uint64_t>(myComm->rank())));
  for (std::size_t crowd = 0; crowd < rngs_.size(); ++crowd)
  {
    generators.push_back(std::make_unique<RandomGenerator>(mix(seed ^ mix(crowd + 1))));
    refs.push_back(*generators.back());
  }
  std::unique_ptr<wftrain::OrbitalInferenceMoments> moments;
  std::vector<double> orbital_values;
#ifdef QMC_COMPLEX
  std::unique_ptr<wftrain::ComplexOrbitalInferenceMoments> complex_moments;
  std::vector<std::complex<double>> complex_orbital_values;
#endif
  std::function<void(Crowd&)> observer;
  std::function<std::string()> block_result;
  std::string header;
  if (training_input_.getObjective() == WFTrainDriverInput::Objective::ORBITAL_MSE)
  {
#ifdef QMC_COMPLEX
    if (complex_orbital_target_)
    {
      const auto& descriptor = complex_orbital_target_->descriptor();
      complex_moments = std::make_unique<wftrain::ComplexOrbitalInferenceMoments>(descriptor.spinUpElectrons(),
          descriptor.spinDownElectrons(), descriptor.terms().size(), inference_input_.evaluate_energy,
          training_input_.getOrbitalNormalization());
      const auto n = checkedMemoryAdd(descriptor.spinUpElectrons(), descriptor.spinDownElectrons());
      complex_orbital_values.reserve(checkedMemoryMultiply(descriptor.terms().size(), checkedMemoryMultiply(n, n)));
      auto* student = complex_training_component_;
      observer = [&, student](Crowd& crowd) {
        for (std::size_t i = 0; i < crowd.size(); ++i)
        {
          const auto& particles = crowd.get_walker_elecs()[i].get();
          student->evaluateOrbitalValues(particles, complex_orbital_values, parameter_version);
          std::optional<double> energy;
          if (inference_input_.evaluate_energy)
            energy = crowd.get_walkers()[i].get().Properties(WalkerProperties::LOCALENERGY);
          complex_moments->addSample(complex_orbital_values, complex_orbital_target_->evaluate(particles), energy);
        }
      };
    }
    else
#endif
    {
      const auto& descriptor = orbital_target_->descriptor();
      moments = std::make_unique<wftrain::OrbitalInferenceMoments>(descriptor.spinUpElectrons(),
          descriptor.spinDownElectrons(), descriptor.terms().size(), inference_input_.evaluate_energy);
      auto* student = jastrow_binding_ ? &jastrow_binding_->neural() : dynamic_cast<PsiFormerWF*>(selected_provider_);
      observer = [&, student](Crowd& crowd) {
        for (std::size_t i = 0; i < crowd.size(); ++i)
        {
          const auto& particles = crowd.get_walker_elecs()[i].get();
          student->evaluateOrbitalValues(particles, orbital_values, parameter_version);
          std::optional<double> energy;
          if (inference_input_.evaluate_energy)
            energy = crowd.get_walkers()[i].get().Properties(WalkerProperties::LOCALENERGY);
          moments->addSample(orbital_values, orbital_target_->evaluate(particles), energy);
        }
      };
    }
    block_result = [&]() {
      wftrain::OrbitalInferenceResult result;
#ifdef QMC_COMPLEX
      if (complex_moments)
      {
        auto sums = complex_moments->sums();
        myComm->allreduce(sums);
        result = complex_moments->summarize(sums);
        complex_moments->reset();
      }
      else
#endif
      {
        auto sums = moments->sums();
        myComm->allreduce(sums);
        result = moments->summarize(sums);
        moments->reset();
      }
      std::ostringstream row;
      row << std::setprecision(17) << result.mean_mse << ' ' << result.mse_variance << ' ';
      if (wftrain::isFiniteTrainingReal(result.occupied_column_cosine_squared))
        row << result.occupied_column_cosine_squared;
      else
        row << "nan";
      row << ' ' << result.undefined_columns << ' ' << result.samples;
      if (inference_input_.evaluate_energy)
        row << ' ' << result.energy << ' ' << result.energy_squared;
      return row.str();
    };
    header = "OrbitalMSE OrbitalMSEVariance EmpiricalOccupiedColumnCosineSquared UndefinedOrbitalColumns SampleCount";
    if (inference_input_.evaluate_energy)
      header += " LocalEnergy LocalEnergy_sq";
  }
  WFTrainInference inference(project_data_, inference_input_, configurations, std::move(population), refs,
                             myComm, prefix, iteration, parameter_version,
                             std::move(observer), std::move(block_result), std::move(header));
  inference.process(nullptr);
  inference.run();
  if (myComm->rank() == 0)
    app_log() << "WFTrain inference iteration " << iteration << " inference_seconds " << elapsed.elapsed() << std::endl;
  return inference.complete();
}

RefVector<QMCDriverNew::ContextForSteps> WFTrainBatched::getContextForStepsRefs() const
{
  RefVector<ContextForSteps> refs;
  refs.reserve(step_contexts_.size());
  for (auto& context : step_contexts_)
    refs.push_back(*context);
  return refs;
}

void WFTrainBatched::createStepContexts(int num_crowds)
{
  assert(num_crowds <= rngs_.size());
  step_contexts_.resize(num_crowds);
  for (int crowd = 0; crowd < num_crowds; ++crowd)
    step_contexts_[crowd] = std::make_unique<ContextForSteps>(rngs_[crowd]);
}

void WFTrainBatched::initializeDeterminantOverlap()
{
  if (training_input_.getObjective() !=
      WFTrainDriverInput::Objective::DETERMINANT_OVERLAP)
    return;
  if (&student_prototype_ == &population_.get_golden_twf())
    throw UniformCommunicateError(
        "WFTrainBatched: determinant overlap requires a distinct frozen teacher wavefunction");

  // Discover M through the generic TrialWaveFunction/component seam.  This works
  // identically on a rank with no live walkers and avoids a concrete PsiFormer cast.
  ParticleSet& golden_electrons = population_.get_golden_electrons();
  auto probe_student = student_prototype_.makeClone(golden_electrons);
  RefVectorWithLeader<TrialWaveFunction> probe_wavefunctions(*probe_student);
  probe_wavefunctions.push_back(*probe_student);
  RefVectorWithLeader<ParticleSet> probe_particles(golden_electrons);
  probe_particles.push_back(golden_electrons);
  auto probe_batch = TrialWaveFunction::makeStructuredDeterminantChannelBatch(
      probe_wavefunctions, probe_particles, selected_provider_id_);
  const std::size_t channel_count = probe_batch->channelCount();

  const auto target_mode =
      training_input_.getOverlapTarget() ==
          WFTrainDriverInput::OverlapTarget::GENERAL_FROZEN_TEACHER
      ? psiformer::DeterminantOverlapTargetMode::GENERAL_FROZEN_TEACHER
      : psiformer::DeterminantOverlapTargetMode::TOP_COEFFICIENT_MULTIDETERMINANT;
  auto target = psiformer::DeterminantOverlapTargetEvaluator::fromFrozenTeacher(
      population_.get_golden_twf(), target_mode,
      training_input_.getTargetProvenance(), channel_count);
  std::vector<double> weights =
      training_input_.getOverlapWeighting() == WFTrainDriverInput::OverlapWeighting::EQUAL
      ? wftrain::makeEqualOverlapWeights(channel_count)
      : wftrain::makeCoefficientSquaredOverlapWeights(target.descriptor());
  psiformer::DeterminantOverlapPolicy policy;
  policy.minimum_fidelity = training_input_.getMinimumFidelity();
  policy.minimum_effective_samples = training_input_.getMinimumEffectiveSamples();
  determinant_overlap_objective_ = std::make_unique<wftrain::DeterminantOverlapObjective>(
      std::move(target), std::move(weights), policy);

  overlap_student_twfs_.resize(crowds_.size());
  for (std::size_t crowd_index = 0; crowd_index < crowds_.size(); ++crowd_index)
  {
    const auto& electrons = crowds_[crowd_index]->get_walker_elecs();
    auto& students = overlap_student_twfs_[crowd_index];
    students.reserve(electrons.size());
    for (ParticleSet& particle_set : electrons)
      students.push_back(student_prototype_.makeClone(particle_set));
  }
}

/** Freeze a conventional determinant source independently of teacher Jastrow factors. */
void WFTrainBatched::initializeOrbitalPretraining()
{
  if (training_input_.getObjective() != WFTrainDriverInput::Objective::ORBITAL_MSE)
    return;
  if (!frozen_teacher_ || frozen_teacher_ == &student_prototype_)
    throw UniformCommunicateError("WFTrainBatched: orbital_mse requires a distinct frozen teacher");
  bool paired = false;
#ifdef QMC_COMPLEX
  auto* complex = complex_training_component_;
  paired = complex != nullptr;
#endif
  if (!paired && !dynamic_cast<PsiFormerWF*>(selected_provider_) && !jastrow_binding_)
    throw UniformCommunicateError("WFTrainBatched: orbital_mse requires a PsiFormer provider");
  if (training_input_.getOrbitalNormalization() != wftrain::OrbitalLossNormalization::SPIN_SECTOR_SUM)
  {
    bool supported = false;
#ifdef QMC_COMPLEX
    supported = complex != nullptr;
#endif
    if (!supported)
      throw UniformCommunicateError("Full-matrix orbital normalization requires the paired provider");
    app_log() << "WFTrain orbital_mse normalization=full_matrix_mean" << std::endl;
  }
  if (training_input_.samplesOrbitalStudent())
  {
    const auto* real_student = jastrow_binding_ ? &jastrow_binding_->neural() : dynamic_cast<PsiFormerWF*>(selected_provider_);
    bool supported_student = real_student && !real_student->usesAcceleratorParameterProducts();
#ifdef QMC_COMPLEX
    supported_student = supported_student || complex != nullptr;
#endif
    if (!supported_student)
      throw UniformCommunicateError("Student-measure orbital_mse requires a supported real CPU or paired provider");
    app_log() << "WFTrain orbital_mse sampling=student target=frozen energy_evaluation=off" << std::endl;
  }

  // Query channel count through the existing determinant-channel seam, without
  // opening native tensor metadata or depending on an imported parameter file.
  ParticleSet& electrons = population_.get_golden_electrons();
  std::size_t count = 0;
#ifdef QMC_COMPLEX
  if (complex)
    count = complex->orbitalMSEChannelCount();
  else
#endif
  {
    auto probe = student_prototype_.makeClone(electrons);
    RefVectorWithLeader<TrialWaveFunction> wavefunctions(*probe);
    wavefunctions.push_back(*probe);
    RefVectorWithLeader<ParticleSet> particles(electrons);
    particles.push_back(electrons);
    count = TrialWaveFunction::makeStructuredDeterminantChannelBatch(
        wavefunctions, particles, jastrow_binding_ ? jastrow_binding_->neural().parameterSchema().providerId() : selected_provider_id_)->channelCount();
  }

  const SlaterDet* single = nullptr;
  const MultiSlaterDetTableMethod* multi = nullptr;
  std::size_t determinant_components = 0;
  for (const auto& component : frozen_teacher_->getOrbitals())
  {
    if (auto* determinant = dynamic_cast<const SlaterDet*>(component.get()))
    {
      single = determinant;
      ++determinant_components;
    }
    else if (auto* determinant = dynamic_cast<const MultiSlaterDetTableMethod*>(component.get()))
    {
      multi = determinant;
      ++determinant_components;
    }
  }
  if (determinant_components != 1)
    throw UniformCommunicateError(
        "WFTrainBatched: orbital_mse teacher must contain exactly one conventional determinant component");
  const bool top = training_input_.getOrbitalTarget() ==
      WFTrainDriverInput::OrbitalTarget::TOP_COEFFICIENT_MULTIDETERMINANT;
#ifdef QMC_COMPLEX
  if (complex)
  {
    // The fixed-gauge single-reference teacher remains CPU, independently of the student backend.
    if (top || !single)
      throw UniformCommunicateError("Paired orbital_mse requires a CPU single-reference periodic orbital teacher");
    complex_orbital_target_ = std::make_unique<psiformer::ComplexOrbitalTargetEvaluator>(
        psiformer::ComplexOrbitalTargetEvaluator::fromSingleReference(*single, electrons, complex->reducedTwist(),
            count, training_input_.getMaximumRetainedBytes()));
    orbital_target_fingerprint_ = complex_orbital_target_->descriptor().fingerprint();
    orbital_target_bytes_ = complex_orbital_target_->retainedScalarBytes();
    const auto caller_bytes = checkedMemoryMultiply(selected_provider_->parameterSchema().parameterCount(), sizeof(double));
    const auto owner_bytes = complex->orbitalMSEStorage().required_bytes;
    auto required = checkedMemoryAdd(owner_bytes,
        checkedMemoryAdd(orbital_target_bytes_, caller_bytes));
    if (complex->gpuInferenceEnabled())
    {
      const auto q = complex->gpuParameterPublicationRequirements();
      const auto& limits = training_input_.getComplexGpuTraining();
      if (checkedMemoryAdd(owner_bytes, q.device_bytes) > limits.device_bytes ||
          checkedMemoryAdd(owner_bytes, q.host_bytes) > limits.host_bytes)
        throw std::length_error("Paired GPU orbital-MSE combined owner/successor cap exceeded before preparation");
      required = checkedMemoryAdd(required, checkedMemoryAdd(q.device_bytes, q.host_bytes));
      app_log() << "WFTrain paired GPU orbital-MSE combined DEVICE+HOST owner bound=" << owner_bytes
                << "; conservatively checked under each independent GPU device/HOST cap; "
                   "successor device=" << q.device_bytes << " HOST=" << q.host_bytes << std::endl;
    }
    if (required > training_input_.getMaximumRetainedBytes())
      throw std::length_error("Paired orbital-MSE score/target/caller-gradient cap exceeded before preparation");
    complex->prepareOrbitalMSE(training_input_.getMaximumRetainedBytes() - orbital_target_bytes_ - caller_bytes);
    complex_orbital_score_bytes_ = complex->orbitalMSEStorage().prepared_bytes;
    app_log() << "WFTrain paired " << (complex->gpuInferenceEnabled() ? "GPU" : "CPU")
              << " orbital-MSE score owner=" << complex_orbital_score_bytes_
              << " target/scratch=" << orbital_target_bytes_ << " caller gradient=" << caller_bytes
              << "; excludes inference/model/SPO metadata and allocator/RSS" << std::endl;
    return;
  }
#endif
  if (top && multi)
    orbital_target_ = std::make_unique<psiformer::OrbitalTargetEvaluator>(
        psiformer::OrbitalTargetEvaluator::fromMultideterminant(
            *multi, training_input_.getTargetProvenance(), count));
  else if (!top && single)
    orbital_target_ = std::make_unique<psiformer::OrbitalTargetEvaluator>(
        psiformer::OrbitalTargetEvaluator::fromSingleReference(
            *single, training_input_.getTargetProvenance(), count));
  else
    throw UniformCommunicateError(
        "WFTrainBatched: single_reference requires SlaterDet; top_coefficients requires MultiSlaterDetTableMethod");
  orbital_target_fingerprint_ = orbital_target_->descriptor().fingerprint();
  orbital_target_bytes_ = orbital_target_->retainedScalarBytes();
}

void WFTrainBatched::validateHamiltonianForTraining()
{
  QMCHamiltonian& hamiltonian = population_.get_golden_hamiltonian();
  if (hamiltonian.has_L2())
    throw UniformCommunicateError(
        "WFTrainBatched: L2 Hamiltonian terms are not supported by energy training");

  std::size_t kinetic_count = 0;
  std::size_t nonlocal_ecp_count = 0;
  for (int component = 0; component < hamiltonian.size(); ++component)
  {
    OperatorBase* operator_component = hamiltonian.getComponent(component);
    if (operator_component->getClassName() == "BareKineticEnergy")
    {
      ++kinetic_count;
      continue;
    }
    if (auto* nonlocal_ecp =
            dynamic_cast<NonLocalECPotential*>(operator_component))
    {
      ++nonlocal_ecp_count;
      if (nonlocal_ecp->usesDLA() || !nonlocal_ecp->usesVirtualParticles())
        throw UniformCommunicateError(
            "WFTrainBatched: ECP training requires ordinary locality with "
            "virtual-particle batching");
      continue;
    }
    if (operator_component->isNonLocal())
      throw UniformCommunicateError(
          "WFTrainBatched: energy training encountered an unsupported nonlocal Hamiltonian term");
    if (operator_component->dependsOnWaveFunction())
      throw UniformCommunicateError(
          "WFTrainBatched: energy training encountered an unsupported "
          "wavefunction-dependent Hamiltonian term");
  }

  if (kinetic_count != 1 || nonlocal_ecp_count > 1 ||
      hamiltonian.hasPhysicalNLPP() != (nonlocal_ecp_count == 1))
    throw UniformCommunicateError(
        "WFTrainBatched: energy training requires one kinetic operator and "
        "at most one consistently registered nonlocal ECP operator");
}

void WFTrainBatched::refreshPublishedParameters(std::size_t parameter_version)
{
  NativeTimerScope timer(phase_timers_.refresh);
  psiformer::memory::Phase memory_timer("refresh");
  // Publication modifies shared model state. Fence ranks before any clone evaluates it.
  myComm->barrier();

  // Frozen-teacher sampling caches do not depend on student parameters.  The
  // next determinant-channel owner reads the shared student model version while
  // copying the current teacher configurations, so no ParticleSet G/L state is
  // overwritten by evaluating a student TrialWaveFunction here. Student-measure
  // orbital sampling instead refreshes current wavefunction caches without energy.
  if (training_input_.getObjective() != WFTrainDriverInput::Objective::ENERGY &&
      !training_input_.samplesOrbitalStudent())
  {
    if (myComm->rank() == 0)
      app_log() << "WFTrain published frozen-target student version "
                << parameter_version << " across " << crowds_.size()
                << " crowd(s)" << std::endl;
    return;
  }

  std::exception_ptr refresh_failure;
  try
  {
    ParallelExecutor<> refresh_task;
    auto step_context_refs = getContextForStepsRefs();
    const auto refresh = training_input_.samplesOrbitalStudent()
        ? initialLogEvaluationWavefunctionOnly : initialLogEvaluation;
    refresh_task(crowds_.size(), refresh, crowds_, step_context_refs,
                 serializing_crowd_walkers_);
  }
  catch (...)
  {
    refresh_failure = std::current_exception();
  }

  int refresh_failed = refresh_failure ? 1 : 0;
  myComm->allreduce(refresh_failed);
  if (refresh_failed != 0)
  {
    if (refresh_failure)
      std::rethrow_exception(refresh_failure);
    throw UniformCommunicateError(
        "WFTrainBatched: another rank failed while refreshing published parameter version " +
        std::to_string(parameter_version));
  }
  if (myComm->rank() == 0)
    app_log() << "WFTrain refreshed parameter version " << parameter_version
              << " across " << crowds_.size() << " crowd(s)" << std::endl;
}

} // namespace qmcplusplus
