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
#include "QMCDrivers/WFTrain/MatrixFreeOperator.h"
#include "QMCDrivers/WFTrain/SampledLinearMethod.h"
#include "QMCDrivers/WFTrain/AcceleratedLinearMethod.h"
#include "QMCDrivers/WFTrain/KFACPreconditioner.h"
#include "QMCDrivers/WFTrain/MatrixFreeStochasticReconfiguration.h"
#include "QMCDrivers/WFTrain/MinimumStepStochasticReconfiguration.h"
#include "QMCDrivers/WFTrain/ParameterUpdateTransaction.h"
#include "QMCDrivers/WFTrain/ProposalSelection.h"
#include "QMCDrivers/WFTrain/NativeLineSearch.h"
#include "QMCDrivers/WFTrain/PreparedEnergyGradientProducer.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"
#include "QMCDrivers/WFTrain/WFTrainOptimizerInput.h"
#include "QMCDrivers/WFTrain/WFTrainSamplerCheckpoint.h"
#include "QMCHamiltonians/NonLocalECPotential.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCWaveFunctions/Optimization/StructuredParameterProvider.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "Utilities/RunTimeManager.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef __linux__
#include <unistd.h>
#endif

namespace qmcplusplus
{
namespace
{

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
};

/** Copy an observed population before later MC moves overwrite its positions. */
RetainedCrowdSamples retainCrowdSamples(Crowd& crowd, std::size_t index, bool overlap)
{
  RetainedCrowdSamples result;
  result.crowd_index = index;
  const auto& electrons = crowd.get_walker_elecs();
  const auto& wavefunctions = crowd.get_walker_twfs();
  const auto& walkers = crowd.get_walkers();
  for (std::size_t sample = 0; sample < crowd.size(); ++sample)
  {
    result.particles.push_back(std::make_unique<ParticleSet>(electrons[sample]));
    result.particles.back()->update();
    result.local_energies.push_back(walkers[sample].get().Properties(WalkerProperties::LOCALENERGY));
    if (overlap)
      result.teachers.push_back(wavefunctions[sample].get().makeClone(*result.particles.back()));
  }
  return result;
}

/** Replay a retained snapshot on resident clones and reconstruct coherent VGL.
 * Iterating snapshots in collection order leaves the live chains at their final
 * positions. No MC proposals or parameter publications occur during replay.
 */
void replayCrowdSamples(Crowd& crowd, const RetainedCrowdSamples& samples)
{
  const auto& electrons = crowd.get_walker_elecs();
  const auto& wavefunctions = crowd.get_walker_twfs();
  const auto& walkers = crowd.get_walkers();
  RefVectorWithLeader<ParticleSet> particle_batch(electrons[0], electrons);
  RefVectorWithLeader<TrialWaveFunction> wavefunction_batch(wavefunctions[0], wavefunctions);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(crowd.getSharedResource().pset_res, particle_batch);
  ResourceCollectionTeamLock<TrialWaveFunction> wavefunction_lock(crowd.getSharedResource().twf_res, wavefunction_batch);
  for (std::size_t sample = 0; sample < crowd.size(); ++sample)
    electrons[sample].get().R = samples.particles[sample]->R;
  ParticleSet::mw_update(particle_batch);
  TrialWaveFunction::mw_evaluateLog(wavefunction_batch, particle_batch);
  for (std::size_t sample = 0; sample < crowd.size(); ++sample)
  {
    electrons[sample].get().saveWalker(walkers[sample]);
    walkers[sample].get().resetProperty(wavefunctions[sample].get().getLogPsi(),
                                       wavefunctions[sample].get().getPhase(),
                                       samples.local_energies[sample]);
  }
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
        observation.row_count});
  }

private:
  wftrain::KFACFactorAccumulator& accumulator_;
};

/// Form a stateless Kronecker-preconditioned gradient proposal.
class LiveKFACUpdateRule final : public wftrain::TrainingUpdateRule
{
public:
  LiveKFACUpdateRule(const wftrain::KFACPreconditioner& inverse, double rate)
      : inverse_(inverse), rate_(rate) {}

  wftrain::StructuredParameterSnapshot propose(const wftrain::StructuredParameterSchema& schema,
      const wftrain::StructuredParameterSnapshot& parameters, wftrain::ParameterGradientView objective) override
  {
    std::vector<wftrain::DerivativeValue> gradient(parameters.values.size()), direction(gradient.size());
    for (std::size_t i = 0; i < gradient.size(); ++i) gradient[i] = objective.gradient[i];
    inverse_.apply({schema, parameters.version, {gradient.data(), gradient.size()}},
                   {direction.data(), direction.size()});
    auto candidate = parameters;
    for (std::size_t i = 0; i < direction.size(); ++i) candidate.values[i] -= rate_ * direction[i].real();
    return candidate;
  }

private:
  const wftrain::KFACPreconditioner& inverse_;
  double rate_;
};

/// Identify the first-order family using the common live gradient producer.
bool isFirstOrder(WFTrainDriverInput::Optimizer optimizer)
{
  using O = WFTrainDriverInput::Optimizer;
  return optimizer == O::ADAM || optimizer == O::AMSGRAD || optimizer == O::SGD ||
      optimizer == O::MOMENTUM_SGD || optimizer == O::RMSPROP;
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

/** Copy live crowd metadata and bind it to a version-checked derivative owner. */
wftrain::PreparedEnergyGradientBatch prepareCrowdBatch(
    Crowd& crowd,
    std::string_view provider_id,
    std::size_t batch_ordinal,
    std::size_t sample_offset,
    std::size_t maximum_parameter_chunk_size,
    bool has_nonlocal_ecp,
    bool capture_ecp_jvp_replay)
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
  TrialWaveFunction::mw_evaluateLog(twf_batch, electron_batch);

  // Construct both derivative owners only after canonical value/VGL refresh so
  // the kinetic tape and accepted-grid ECP response see identical reference
  // coordinates, gradients, and parameter version.
  batch.derivative_operator =
      TrialWaveFunction::makeStructuredStreamingDerivativeOperator(
          twf_batch, electron_batch, provider_id, batch_ordinal, sample_offset,
          maximum_parameter_chunk_size);
  auto ecp_consumer =
      TrialWaveFunction::makeStructuredNonLocalECPDerivativeConsumer(
          twf_batch, electron_batch, provider_id,
          maximum_parameter_chunk_size, capture_ecp_jvp_replay);

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
  wftrain::PreparedLocalEnergyVJPCapture ecp_capture(
      schema, parameter_version, ecp_mask);

  const auto ecp_result =
      QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
          hamiltonian_batch, twf_batch, electron_batch, *ecp_consumer,
          {&ecp_channel, 1}, ecp_capture);
  batch.local_energies.reserve(ecp_result.local_energies.size());
  for (const auto energy : ecp_result.local_energies)
    batch.local_energies.emplace_back(energy, 0.0);
  batch.additive_local_energy_vjp.emplace(ecp_capture.release());

  if (capture_ecp_jvp_replay)
  {
    auto replay = ecp_consumer->releaseJVPReplay();
    if (replay->gridFingerprint() != ecp_result.grid_fingerprint)
      throw std::logic_error(
          "WFTrainBatched: accepted ECP replay grid identity changed before binding");
    batch.derivative_operator->bindNonLocalECPJVPReplay(std::move(replay));
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

/** Count exact numeric storage held by raw prepared batches. */
wftrain::PreparedEnergyGradientStorageDiagnostics preparedStorageDiagnostics(
    const std::vector<wftrain::PreparedEnergyGradientBatch>& batches)
{
  wftrain::PreparedEnergyGradientStorageDiagnostics result;
  result.batch_count = batches.size();
  for (const auto& batch : batches)
  {
    if (!batch.derivative_operator)
      throw std::invalid_argument(
          "WFTrainBatched: prepared batch has no derivative operator");
    result.sample_count += batch.weights.size();
    const auto storage = batch.derivative_operator->storageDiagnostics();
    result.derivative_owner_bytes += storage.retained_numeric_bytes;
    if (storage.shared_workspace_identity)
      for (const auto& previous : batches)
      {
        if (&previous == &batch)
          break;
        if (previous.derivative_operator->storageDiagnostics().shared_workspace_identity ==
            storage.shared_workspace_identity)
        {
          result.derivative_owner_bytes -= storage.shared_workspace_bytes;
          break;
        }
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
                                          std::uint64_t resolved_overlap_fingerprint)
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  extendStageFingerprint(hash, schema.fingerprint());
  const auto objective = static_cast<std::underlying_type_t<WFTrainDriverInput::Objective>>(
      training.getObjective());
  const auto optimizer = static_cast<std::underlying_type_t<WFTrainDriverInput::Optimizer>>(
      training.getOptimizer());
  extendStageFingerprint(hash, objective);
  extendStageFingerprint(hash, optimizer);
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
  if (training.getObjective() == WFTrainDriverInput::Objective::DETERMINANT_OVERLAP)
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

} // namespace

WFTrainBatched::WFTrainBatched(const ProjectData& project_data,
                               QMCDriverInput&& qmcdriver_input,
                               UPtr<EstimatorManagerNew>&& estimator_manager,
                               VMCDriverInput&& vmc_input,
                               WFTrainDriverInput&& training_input,
                               WalkerConfigurations& walker_configurations,
                               TrialWaveFunction& student_prototype,
                               MCPopulation&& population,
                               const RefVector<RandomBase<FullPrecRealType>>& random_generators,
                               Communicate* communicator)
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
      student_prototype_(student_prototype)
{}

WFTrainBatched::~WFTrainBatched() = default;

void WFTrainBatched::process(xmlNodePtr node)
{
  ScopedTimer startup_timer(timers_.startup_timer);
  print_mem("WFTrainBatched before initialization", app_log());

  try
  {
    VMCBatchedSampling::validateMoveConfiguration(vmcdriver_input_, population_.get_golden_electrons(),
                                                  population_.get_golden_twf(), serializing_crowd_walkers_);

    if (population_.get_golden_electrons().isSpinor())
      throw UniformCommunicateError("WFTrainBatched: spinor training is not supported");
    if (training_input_.getObjective() == WFTrainDriverInput::Objective::ENERGY)
      validateHamiltonianForTraining();

    const auto providers = student_prototype_.extractStructuredParameterProviders();
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
    initPopulationAndCrowds(walker_counts);
    createStepContexts(crowds_.size());
    initializeDeterminantOverlap();
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

  const IndexType num_blocks = qmcdriver_input_.get_max_blocks();
  const int num_iterations = static_cast<int>(training_input_.getIterations());
  estimator_manager_->startDriverRun();

  WalkerLogManager wlog_manager(walker_logs_input, allow_walker_logs, get_root_name(), myComm);
  for (auto& crowd : crowds_)
    crowd->setWalkerLogCollector(wlog_manager.makeCollector());
  wlog_manager.startRun(Crowd::getWalkerLogCollectorRefs(crowds_));

  VMCBatchedSampling::State sampling_state(qmcdriver_input_, vmcdriver_input_, *drift_modifier_, population_,
                                            steps_per_block_, serializing_crowd_walkers_);
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
  wftrain::HighParameterTraining trainer(
      requirements, wftrain::EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN,
      reduction);
  const auto initial_parameters = selected_provider_->snapshotParameters();
  if (initial_parameters.schema_fingerprint != selected_provider_->parameterSchema().fingerprint() ||
      initial_parameters.values.size() != selected_provider_->parameterSchema().parameterCount())
    throw UniformCommunicateError(
        "WFTrainBatched: selected provider returned an inconsistent initial snapshot");
  std::unique_ptr<wftrain::FirstOrderOptimizer> first_order_optimizer;
  if (isFirstOrder(training_input_.getOptimizer()))
    first_order_optimizer = std::make_unique<wftrain::FirstOrderOptimizer>(
        selected_provider_->parameterSchema(),
        makeFirstOrderOptions(selected_provider_->parameterSchema(), training_input_));
  std::unique_ptr<wftrain::SPRINGUpdateRule> spring_optimizer;
  if (training_input_.getOptimizer() == WFTrainDriverInput::Optimizer::SPRING)
    spring_optimizer = std::make_unique<wftrain::SPRINGUpdateRule>(
        selected_provider_->parameterSchema(), makeSPRINGControl(training_input_));
  wftrain::OptimizerCheckpointContributor* optimizer_checkpoint = nullptr;
  if (first_order_optimizer)
    optimizer_checkpoint = first_order_optimizer.get();
  else if (spring_optimizer)
    optimizer_checkpoint = spring_optimizer.get();

  wftrain::TrainingIterationState training_state{
      0, initial_parameters.version, selected_provider_->parameterSchema().fingerprint()};
  const bool determinant_overlap =
      training_input_.getObjective() ==
      WFTrainDriverInput::Objective::DETERMINANT_OVERLAP;
  if (determinant_overlap && !determinant_overlap_objective_)
    throw UniformCommunicateError(
        "WFTrainBatched: determinant-overlap state was not initialized by process()");
  const std::uint64_t overlap_fingerprint = determinant_overlap
      ? determinant_overlap_objective_->configurationFingerprint()
      : 0;
  const bool has_nonlocal_ecp =
      population_.get_golden_hamiltonian().hasPhysicalNLPP();
  wftrain::TrainingStageState stage_state{
      determinant_overlap ? "determinant_overlap"
                          : (has_nonlocal_ecp ? "pseudopotential_energy"
                                              : "all_electron_energy"),
      0, 0,
      stageConfigurationFingerprint(training_input_, qmcdriver_input_, vmcdriver_input_,
                                    selected_provider_->parameterSchema(), steps_per_block_,
                                    overlap_fingerprint)};
  DirtyCacheObserver cache_observer;

  LoopTimer<> training_loop;
  RunTimeControl<> runtime_control(run_time_manager, project_data_.getMaxCPUSeconds(),
                                   project_data_.getTitle(), myComm->rank() == 0);
  ParallelExecutor<> crowd_task;

  int global_step = 0;
  wftrain::WFTrainSamplerCheckpoint sampler_checkpoint(
      population_, rngs_, global_step, static_cast<std::size_t>(myComm->size()),
      static_cast<std::size_t>(myComm->rank()), crowds_.size());
  const bool resumed = !training_input_.getRestartPath().empty();
  // Scalar blocks remain statistical output. This separate history records one
  // complete optimization attempt, including its sampled and published versions.
  std::ofstream iteration_history;
  bool history_failed = false;
  if (myComm->rank() == 0)
  {
    const std::string history_path = get_root_name() + ".training.csv";
    std::ifstream previous(history_path);
    const bool has_header = resumed && previous.peek() != std::ifstream::traits_type::eof();
    iteration_history.open(history_path, resumed ? std::ios::app : std::ios::trunc);
    if (!has_header)
      iteration_history << "iteration,sample_version,parameter_version,objective,optimizer,samples,"
                           "mean_energy,variance,gradient_norm,overlap_loss,selected_alpha,committed,"
                           "sampling_seconds,total_seconds,retained_numeric_bytes,resident_bytes,"
                           "krylov_iterations,krylov_operator_applications\n";
    iteration_history.flush();
    history_failed = !iteration_history;
  }
  myComm->bcast(history_failed);
  if (history_failed)
    throw UniformCommunicateError("WFTrainBatched: cannot open iteration history");
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
                << " completed update(s) from " << training_input_.getRestartPath()
                << std::endl;
  }

  {
    ScopedTimer initialization_timer(timers_.init_walkers_timer);
    auto step_context_refs = getContextForStepsRefs();
    const auto initialize = resumed ? initialLogEvaluationDeterministic : initialLogEvaluation;
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

  for (int iteration = static_cast<int>(stage_state.completed_stage_iterations);
       iteration < num_iterations; ++iteration)
  {
    ScopeGuard<LoopTimer<>> iteration_timer(training_loop);
    Timer iteration_wall;
    const auto sample_version = training_state.parameter_version;
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
    std::size_t retained_configuration_bytes = 0;
    for (int block = 0; block < num_blocks; ++block)
    {
      const int statistical_block = iteration * num_blocks + block;
      if (myComm->rank() == 0)
        app_log() << "WFTrain statistical block " << statistical_block
                  << " iteration " << iteration + 1
                  << " version " << training_state.parameter_version << std::endl;
      sampling_state.recalculate_properties_period = qmc_driver_mode_[QMC_UPDATE_MODE]
          ? qmcdriver_input_.get_recalculate_properties_period() : 0;
      sampling_state.is_recomputing_block = qmcdriver_input_.get_blocks_between_recompute()
          ? (1 + statistical_block) % qmcdriver_input_.get_blocks_between_recompute() == 0 : false;
      estimator_manager_->startBlock(steps_per_block_);
      for (auto& crowd : crowds_)
        crowd->startBlock(steps_per_block_);
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
          for (std::size_t crowd_index = 0; crowd_index < crowds_.size(); ++crowd_index)
            if (crowds_[crowd_index]->size())
            {
              retained_samples.push_back(retainCrowdSamples(
                  *crowds_[crowd_index], crowd_index, determinant_overlap));
              const auto& snapshot = retained_samples.back();
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
      endBlock();
      wlog_manager.writeBuffers();
      recordBlock(statistical_block);
      ++completed_blocks;
    }

    const double sampling_seconds = iteration_wall.elapsed();
    wftrain::TrainingIterationResult iteration_result;
    wftrain::PreparedEnergyGradientStorageDiagnostics storage_diagnostics;
    std::size_t retained_numeric_bytes = 0;
    std::optional<std::size_t> krylov_iterations;
    std::optional<std::size_t> krylov_operator_applications;
    std::optional<wftrain::DerivativeReal> applied_scale;
    std::optional<wftrain::DeterminantOverlapObjectiveResult> overlap_diagnostics;
    double selected_alpha = 1.0;
    double selected_displacement_norm = 0.0;
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
                });
            selected_displacement_norm = selected_alpha * norm;
            return selected_alpha;
          });
    };
    {
      if (determinant_overlap)
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
          batches.reserve(retained_samples.size());
          std::size_t sample_offset = 0;
          std::size_t batch_ordinal = 0;
          // Prepare immutable score/kinetic owners only after collection. ECP
          // energies and responses are paired on their accepted replay grids.
          for (auto& samples : retained_samples)
          {
            Crowd& crowd = *crowds_[samples.crowd_index];
            replayCrowdSamples(crowd, samples);
            batches.push_back(prepareCrowdBatch(
                crowd, selected_provider_id_, batch_ordinal++, sample_offset,
                training_input_.getParameterChunkSize(), has_nonlocal_ecp,
                capture_ecp_jvp_replay));
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
          // RSS covers ParticleSet/teacher caches that numeric diagnostics omit.
        }
        catch (...)
        {
          preparation_failure = std::current_exception();
        }

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

        if (isFirstOrder(optimizer_kind) ||
            optimizer_kind == WFTrainDriverInput::Optimizer::STOCHASTIC_RECONFIGURATION)
        {
          int empty_rank = batches.empty() ? 1 : 0;
          myComm->allreduce(empty_rank);
          if (empty_rank != 0)
            throw UniformCommunicateError(
                "WFTrainBatched: First-order methods and parameter-space SR require at least one walker on every rank");
        }
        if (isFirstOrder(optimizer_kind))
        {
          std::unique_ptr<wftrain::PreparedEnergyGradientProducer> producer;
          std::exception_ptr producer_failure;
          try
          {
            producer = std::make_unique<wftrain::PreparedEnergyGradientProducer>(
                std::move(batches));
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
          retained_numeric_bytes = storage_diagnostics.retained_numeric_bytes +
              first_order_optimizer->retainedBytes();
          enforceMemoryBudget(*myComm, checkedMemoryAdd(retained_numeric_bytes, retained_configuration_bytes),
                              training_input_.getMaximumRetainedBytes());
          iteration_result = trainer.runIteration(*selected_provider_, *producer,
                                                  *decorate(*first_order_optimizer), training_state,
                                                  &cache_observer);
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
          wftrain::ScoreCovarianceFactor factor(
              schema, parameters->version,
              {score_batches.data(), score_batches.size()},
              wftrain::ScoreSampleDomain::REAL, reduction);
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
            retained_numeric_bytes = checkedMemoryAdd(retained_numeric_bytes,
                checkedMemoryMultiply(checkedMemoryMultiply(vectors, initial.size()), sizeof(wftrain::DerivativeValue)));
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
              throw UniformCommunicateError(std::string("Live LM solve failed: ") +
                  wftrain::linearMethodStopReasonName(result.stop_reason));
            krylov_iterations = result.iterations;
            krylov_operator_applications = result.operator_applications;
            applied_scale = training_input_.getLearningRate();
            wftrain::LinearMethodUpdateRule update_rule(factor, std::move(result),
                training_input_.getLearningRate(), training_input_.getLMNormalizationXi());
            iteration_result = runScoreFactorIteration(*selected_provider_, *parameters, factor,
                *decorate(update_rule), training_state, reduction, cache_observer,
                energy_moments.weight_sum, energy_moments.variance);
          }
          else if (optimizer_kind == WFTrainDriverInput::Optimizer::KFAC)
          {
            std::unique_ptr<wftrain::KFACFactorAccumulator> statistics;
            std::exception_ptr observation_failure;
            try
            {
              const auto metadata = selected_provider_->affineParameterBlocks();
              if (metadata.empty()) throw std::invalid_argument("Provider lacks live KFAC affine observations");
              std::vector<wftrain::KFACAffineBlockDescriptor> affine;
              std::vector<bool> covered(schema.blocks().size(), false);
              for (const auto& block : metadata)
              {
                affine.push_back({block.id, block.weight_block, block.bias_block,
                                  block.input_width, block.output_width});
                covered.at(block.weight_block) = true;
                if (block.bias_block != wftrain::NO_AFFINE_BIAS_BLOCK) covered.at(block.bias_block) = true;
              }
              std::vector<std::size_t> fallback;
              for (std::size_t i = 0; i < covered.size(); ++i)
                if (schema.blocks()[i].trainable && !covered[i]) fallback.push_back(i);
              statistics = std::make_unique<wftrain::KFACFactorAccumulator>(
                  wftrain::KFACBlockRegistry(schema, std::move(affine), std::move(fallback)), parameters->version);
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
            wftrain::KFACPreconditioner inverse(*statistics,
                {training_input_.getDamping(), training_input_.getKFACFallbackDamping()});
            retained_numeric_bytes = checkedMemoryAdd(retained_numeric_bytes,
                checkedMemoryAdd(statistics->retainedNumericBytes(), inverse.storageDiagnostics().retained_numeric_bytes));
            enforceMemoryBudget(*myComm, checkedMemoryAdd(retained_numeric_bytes, retained_configuration_bytes), training_input_.getMaximumRetainedBytes());
            LiveKFACUpdateRule update_rule(inverse, training_input_.getLearningRate());
            iteration_result = runScoreFactorIteration(*selected_provider_, *parameters, factor,
                *decorate(update_rule), training_state, reduction, cache_observer,
                energy_moments.weight_sum, energy_moments.variance);
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
      }
    }

    // All derivative owners for the old version are gone before cache replacement.
    if (!cache_observer.dirty() ||
        cache_observer.publishedVersion() != iteration_result.parameter_version)
      throw UniformCommunicateError(
          "WFTrainBatched: parameter publication did not arm the cache-refresh barrier");
    if (selected_alpha != 0.0)
    {
      refreshPublishedParameters(iteration_result.parameter_version);
      ++accepted_updates;
    }
    // Rejection leaves every live parameter/cache valid. In particular, avoid
    // consuming production quadrature RNG merely to refresh the unchanged ECP state.
    cache_observer.clear();
    stage_state.completed_stage_iterations = iteration_result.completed_iteration;
    enforceResidentBudget(*myComm, training_input_.getMaximumResidentBytes());

    if (myComm->rank() == 0)
    {
      app_log() << std::setprecision(12)
                << "WFTrain iteration " << iteration_result.completed_iteration
                << " version " << iteration_result.parameter_version;
      if (overlap_diagnostics)
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
      iteration_history << std::setprecision(17) << iteration_result.completed_iteration << ','
                        << sample_version << ',' << iteration_result.parameter_version << ','
                        << (determinant_overlap ? "determinant_overlap" : "energy") << ','
                        << wftrain::optimizerName(training_input_.getOptimizer()) << ',';
      if (overlap_diagnostics)
        iteration_history << overlap_diagnostics->sample_count << ",,,,"
                          << overlap_diagnostics->loss;
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
      iteration_history << '\n';
      iteration_history.flush();
      history_failed = !iteration_history;
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
    const int checkpoint_period = qmcdriver_input_.get_check_point_period();
    const bool periodic_checkpoint =
        checkpoint_period > 0 &&
        ((iteration + 1) * num_blocks / checkpoint_period > iteration * num_blocks / checkpoint_period);
    const bool final_block = iteration + 1 == num_iterations;
    if (!training_input_.getCheckpointPath().empty() &&
        (periodic_checkpoint || final_block || stop_requested))
    {
      ScopedTimer checkpoint_timer(timers_.checkpoint_timer);
      wftrain::DistributedTrainingCheckpoint::saveAtomic(
          training_input_.getCheckpointPath(), *myComm, *selected_provider_,
          training_state, stage_state, optimizer_checkpoint, sampler_checkpoint,
          sampler_checkpoint.topology());
      if (myComm->rank() == 0)
        app_log() << "WFTrain checkpointed " << stage_state.completed_stage_iterations
                  << " completed update(s) to " << training_input_.getCheckpointPath()
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
  estimator_manager_->stopDriverRun();
  finalize(completed_blocks, true);
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
  // Publication modifies shared model state. Fence ranks before any clone evaluates it.
  myComm->barrier();

  // Frozen-teacher sampling caches do not depend on student parameters.  The
  // next determinant-channel owner reads the shared student model version while
  // copying the current teacher configurations, so no ParticleSet G/L state is
  // overwritten by evaluating a student TrialWaveFunction here.
  if (training_input_.getObjective() ==
      WFTrainDriverInput::Objective::DETERMINANT_OVERLAP)
  {
    if (myComm->rank() == 0)
      app_log() << "WFTrain published determinant-overlap student version "
                << parameter_version << " across " << crowds_.size()
                << " crowd(s)" << std::endl;
    return;
  }

  std::exception_ptr refresh_failure;
  try
  {
    ParallelExecutor<> refresh_task;
    auto step_context_refs = getContextForStepsRefs();
    refresh_task(crowds_.size(), initialLogEvaluation, crowds_, step_context_refs,
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
