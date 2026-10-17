//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file NativeLineSearch.cpp
 * @brief Isolated native inference and paired, guarded stochastic step selection.
 */
#include "NativeLineSearch.h"
#include "DeterminantOverlapObjective.h"
#include "TrainingNumerics.h"
#include "TrainingWaveFunctionBinding.h"
#include "Message/CommOperators.h"
#include "Message/Communicate.h"
#include "Particle/ParticleSet.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexWF.h"
#include "QMCWaveFunctions/Optimization/StructuredParameterProvider.h"
#include "Utilities/RandomBase.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace qmcplusplus::wftrain
{
namespace
{

/// Locate a unique provider on a cloned complete wavefunction.
StructuredParameterProvider& providerOf(TrialWaveFunction& wavefunction, const std::string& id)
{
  for (auto provider : wavefunction.extractStructuredParameterProviders())
    if (provider.get().parameterSchema().providerId() == id)
      return provider.get();
#ifdef QMC_COMPLEX
  // GPU discovery stays closed globally; only this sole-component private route resolves it.
  if (wavefunction.getOrbitals().size() == 1)
    if (auto* paired = dynamic_cast<PsiFormerComplexWF*>(wavefunction.getOrbitals().front().get());
        paired && paired->gpuInferenceEnabled() && paired->parameterSchema().providerId() == id)
      return *paired;
#endif
  throw std::invalid_argument("Candidate wavefunction lacks selected provider");
}

/// Synchronize fallible local inference before any statistical collective.
bool allReady(Communicate& comm, bool ready)
{
  int failed = ready ? 0 : 1;
  comm.allreduce(failed);
  return failed == 0;
}

/// Obtain a replicated maximum for stable log-weight normalization.
double globalMaximum(Communicate& comm, double value)
{
#ifdef HAVE_MPI
  if (comm.size() > 1)
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MAX, comm.getMPI());
#endif
  return value;
}

/// Candidate loss and paired uncertainty on one fixed reference population.
struct Assessment
{
  double loss = 0.0;
  double difference = 0.0;
  double standard_error = 0.0;
  double ess = 0.0;
  double maximum_weight = 0.0;
  bool valid = false;
};

/// Reduce log-space importance weights and walker-clustered paired influences.
Assessment assessEnergy(Communicate& comm, const std::vector<double>& logs,
                        const std::vector<double>& energies,
                        const std::vector<double>& reference,
                        const std::vector<std::size_t>& clusters, std::size_t local_clusters)
{
  Assessment result;
  std::size_t count = logs.size();
  comm.allreduce(count);
  if (count < 2) return result;
  double shift = -std::numeric_limits<double>::infinity();
  for (double value : logs) shift = std::max(shift, value);
  shift = globalMaximum(comm, shift);
  std::vector<double> weights(logs.size());
  double sum = 0.0, squared = 0.0, weighted_energy = 0.0, reference_sum = 0.0;
  for (std::size_t i = 0; i < logs.size(); ++i)
  {
    weights[i] = std::exp(logs[i] - shift);
    sum += weights[i];
    squared += weights[i] * weights[i];
    weighted_energy += weights[i] * energies[i];
    reference_sum += reference[i];
  }
  comm.allreduce(sum);
  comm.allreduce(squared);
  comm.allreduce(weighted_energy);
  comm.allreduce(reference_sum);
  result.loss = weighted_energy / sum;
  const double baseline = reference_sum / count;
  result.difference = result.loss - baseline;
  result.ess = sum * sum / squared;
  result.maximum_weight = 1.0 / sum;
  std::vector<double> influence(local_clusters, 0.0);
  for (std::size_t i = 0; i < logs.size(); ++i)
    influence[clusters[i]] += weights[i] / sum * (energies[i] - result.loss) -
        (reference[i] - baseline) / count;
  double variance = 0.0;
  for (double value : influence) variance += value * value;
  std::size_t cluster_count = local_clusters;
  comm.allreduce(cluster_count);
  comm.allreduce(variance);
  result.standard_error = cluster_count > 1
      ? std::sqrt(variance * cluster_count / (cluster_count - 1.0))
      : std::numeric_limits<double>::infinity();
  result.valid = isFiniteTrainingReal(result.loss) &&
      isFiniteTrainingReal(result.standard_error);
  return result;
}

/// Symmetric all-particle proposals used only by isolated endpoint sampling.
void sampleEndpoint(ParticleSet& particles, TrialWaveFunction& wavefunction,
                    RandomBase<double>& rng, double timestep, std::size_t steps)
{
  double logpsi = wavefunction.evaluateLog(particles);
  for (std::size_t step = 0; step < steps; ++step)
  {
    const auto old = particles.R;
    for (int i = 0; i < particles.getTotalNum(); ++i)
      for (int d = 0; d < 3; ++d)
      {
        const double u = std::max(rng(), std::numeric_limits<double>::min());
        const double gaussian = std::sqrt(-2.0 * std::log(u)) *
            std::cos(6.2831853071795864769 * rng());
        particles.R[i][d] += std::sqrt(timestep) * gaussian;
      }
    particles.update();
    const double proposed = wavefunction.evaluateLog(particles);
    const double log_acceptance = 2.0 * (proposed - logpsi);
    if (isFiniteTrainingReal(proposed) &&
        std::log(std::max(rng(), std::numeric_limits<double>::min())) < log_acceptance)
      logpsi = proposed;
    else
    {
      particles.R = old;
      particles.update();
      wavefunction.evaluateLog(particles);
    }
  }
}

} // namespace

double selectNativeStep(const WFTrainDriverInput& input, Communicate& comm,
                        TrialWaveFunction& prototype, QMCHamiltonian& hamiltonian,
                        const std::string& provider_id,
                        const std::vector<std::reference_wrapper<ParticleSet>>& configurations,
                        const std::vector<std::size_t>& configuration_chain_ids,
                        const std::vector<std::reference_wrapper<TrialWaveFunction>>& teachers,
                        RandomBase<double>& production_rng, double timestep,
                        const StructuredParameterSnapshot& old,
                        const StructuredParameterSnapshot& full,
                        const DeterminantOverlapObjective* overlap,
                        double maximum_alpha,
                        const std::function<void(double,double,double,double,double,double,bool)>& report,
                        const TrainingWaveFunctionBinding* conventional,
                        const PairedFixedGridNativeEnergy* paired_fixed_grid)
{
  // Each rank retains one private model and one scratch ParticleSet. Empty ranks
  // use no private model but participate in every candidate statistical reduction.
  std::unique_ptr<ParticleSet> particles;
  std::unique_ptr<TrialWaveFunction> wavefunction;
  std::unique_ptr<QMCHamiltonian> ham;
  auto rng = production_rng.makeClone();
  auto quadrature_rng = production_rng.makeClone();
  bool ready = true;
  std::size_t local_chain_count = 0;
  try
  {
#ifdef QMC_COMPLEX
    if (paired_fixed_grid && (conventional || overlap ||
        input.getObjective() != WFTrainDriverInput::Objective::ENERGY ||
        input.getLineSearch() != WFTrainDriverInput::LineSearch::REFERENCE ||
        paired_fixed_grid->sources.size() != configurations.size() ||
        paired_fixed_grid->references.size() != configurations.size()))
      throw std::invalid_argument("Paired fixed-grid selection requires complete ENERGY reference bindings");
#else
    if (paired_fixed_grid)
      throw std::invalid_argument("Paired fixed-grid selection requires a complex build");
#endif
    if (configuration_chain_ids.size() != configurations.size())
      throw std::invalid_argument("Line-search configurations and chain identities differ in size");
    for (const auto id : configuration_chain_ids)
    {
      if (id >= configurations.size())
        throw std::invalid_argument("Line-search chain identity is outside the sample extent");
      local_chain_count = std::max(local_chain_count, id + 1);
    }
    if (!configurations.empty())
    {
      particles = std::make_unique<ParticleSet>(configurations.front().get());
      wavefunction = prototype.makeClone(*particles);
      if (!conventional)
      {
        auto& provider = providerOf(*wavefunction, provider_id);
#ifdef QMC_COMPLEX
        if (auto* paired = dynamic_cast<PsiFormerComplexWF*>(&provider); paired && paired->gpuInferenceEnabled())
        {
          const auto q = paired->gpuCandidateIsolationRequirements();
          paired->isolateBudgetedGpuParameterState(q.device_bytes, q.host_bytes);
        }
        else
#endif
          provider.isolateParameterState();
      }
      if (!paired_fixed_grid)
        ham = hamiltonian.makeClone(*particles, *wavefunction);
    }
  }
  catch (...) { ready = false; }
  if (!allReady(comm, ready))
    throw std::runtime_error("Failed to construct isolated line-search inference state");

  auto install = [&](double alpha) {
    if (!wavefunction) return;
    if (conventional)
    {
      auto candidate = old;
      for (std::size_t i = 0; i < candidate.values.size(); ++i)
        candidate.values[i] = old.values[i] + alpha * (full.values[i] - old.values[i]);
      // Candidate installation evaluates immediately; initialize the copied distance tables first.
      particles->update();
      wavefunction = conventional->makeCandidate(*particles, candidate);
      ham = hamiltonian.makeClone(*particles, *wavefunction);
      return;
    }
    auto& provider = providerOf(*wavefunction, provider_id);
    // Private generations advance per trial; interpolation always uses fixed production endpoints.
    auto candidate = provider.snapshotParameters();
    for (std::size_t i = 0; i < candidate.values.size(); ++i)
      candidate.values[i] = old.values[i] + alpha * (full.values[i] - old.values[i]);
#ifdef QMC_COMPLEX
    if (auto* paired = dynamic_cast<PsiFormerComplexWF*>(&provider); paired && paired->gpuInferenceEnabled())
    {
      const auto q = paired->gpuParameterPublicationRequirements();
      paired->publishBudgetedGpuParameters(candidate, candidate.version, q.device_bytes, q.host_bytes);
      return;
    }
#endif
    provider.publishParameters(candidate, candidate.version);
  };
  using Positions = ParticleSet::ParticlePos;
  std::vector<Positions> positions;
  std::vector<std::size_t> clusters;
  for (std::size_t i = 0; i < configurations.size(); ++i)
  {
    positions.push_back(configurations[i].get().R);
    clusters.push_back(configuration_chain_ids[i]);
  }
  std::vector<double> reference_log, reference_energy;
  std::map<double, Assessment> assessments;

  auto overlapEvaluation = [&](double alpha) {
    ready = true;
    std::vector<std::unique_ptr<ParticleSet>> copied_particles;
    std::vector<std::unique_ptr<TrialWaveFunction>> students;
    std::vector<DeterminantOverlapBatchView> views;
    std::vector<std::unique_ptr<psiformer::DeterminantChannelBatch>> owners;
    try
    {
      install(alpha);
      for (std::size_t i = 0; i < configurations.size(); ++i)
      {
        copied_particles.push_back(std::make_unique<ParticleSet>(configurations[i].get()));
        students.push_back(wavefunction->makeClone(*copied_particles.back()));
      }
      if (!students.empty())
      {
        RefVectorWithLeader<TrialWaveFunction> twfs(*students.front());
        RefVectorWithLeader<ParticleSet> psets(*copied_particles.front());
        DeterminantOverlapBatchView view;
        for (std::size_t i = 0; i < students.size(); ++i)
        {
          twfs.push_back(*students[i]);
          psets.push_back(*copied_particles[i]);
          view.teachers.push_back(teachers[i]);
          view.electrons.push_back(*copied_particles[i]);
        }
        owners.push_back(TrialWaveFunction::makeStructuredDeterminantChannelBatch(twfs, psets, provider_id));
        view.student = owners.back().get();
        views.push_back(std::move(view));
      }
    }
    catch (...) { ready = false; }
    if (!allReady(comm, ready))
      throw std::runtime_error("Could not prepare isolated overlap candidates");
    return overlap->evaluate(comm, views, false);
  };
  std::optional<DeterminantOverlapObjectiveResult> overlap_reference;
  if (overlap)
    overlap_reference.emplace(overlapEvaluation(0.0));

  auto energyValues = [&](double alpha, std::vector<double>& logs, std::vector<double>& energies) {
    ready = true;
    try
    {
      install(alpha);
      quadrature_rng = production_rng.makeClone();
      if (ham)
        ham->setRandomGenerator(quadrature_rng.get());
      for (std::size_t sample = 0; sample < positions.size(); ++sample)
      {
        particles->R = positions[sample];
        particles->update();
        const double logpsi = wavefunction->evaluateLog(*particles);
        // Replay one private RNG stream across the fixed population, pairing
        // quadrature rotations between candidates while varying them by sample.
        double energy;
#ifdef QMC_COMPLEX
        if (paired_fixed_grid)
        {
          const auto* candidate = dynamic_cast<const PsiFormerComplexWF*>(
              &providerOf(*wavefunction, provider_id));
          if (!candidate || !candidate->gpuInferenceEnabled())
            throw std::logic_error("Paired fixed-grid selection lost its private GPU family");
          const auto observed = paired_fixed_grid->sources[sample].get().evaluateFixedGridCandidateEnergy(
              paired_fixed_grid->references[sample], *candidate, *particles);
          paired_fixed_grid->sources[sample].get().validateFixedGridCandidateEnergy(observed, *candidate);
          energy = observed.local_energy.real();
          app_log() << std::setprecision(17) << "WFTrain paired fixed-grid candidate: alpha=" << alpha
                    << " sample=" << sample << " grid=" << observed.grid_fingerprint
                    << " source_version=" << observed.source_version << " candidate_version=" << observed.candidate_version
                    << " ReEL=" << energy << " ImEL=" << observed.local_energy.imag()
                    << " no_rng_draw=yes" << std::endl;
        }
        else
#endif
          energy = ham->evaluate(*wavefunction, *particles);
        if (!isFiniteTrainingReal(logpsi) || !isFiniteTrainingReal(energy))
          throw std::runtime_error("Nonfinite candidate inference");
        logs.push_back(logpsi);
        energies.push_back(energy);
      }
    }
    catch (...) { ready = false; }
    return allReady(comm, ready);
  };
  if (!overlap && !energyValues(0.0, reference_log, reference_energy))
    throw std::runtime_error("Invalid reference line-search energy");

  auto evaluate = [&](double alpha) {
    if (const auto found = assessments.find(alpha); found != assessments.end())
      return found->second;
    Assessment result;
    if (alpha > maximum_alpha)
    {
      assessments[alpha] = result;
      return result;
    }
    if (overlap)
    {
      try
      {
        auto candidate = overlapEvaluation(alpha);
        result.loss = candidate.loss;
        result.difference = candidate.loss - overlap_reference->loss;
        result.ess = candidate.sample_count;
        result.maximum_weight = 0.0;
        // Repeated snapshots from one walker are not independent observations.
        std::vector<double> influences(local_chain_count, 0.0);
        for (std::size_t i = 0; i < candidate.sample_loss_influences.size(); ++i)
        {
          const double delta = candidate.sample_loss_influences[i] -
              overlap_reference->sample_loss_influences[i];
          influences[configuration_chain_ids[i]] += delta;
        }
        double variance = 0.0;
        for (double influence : influences)
          variance += influence * influence;
        std::size_t global_chain_count = local_chain_count;
        comm.allreduce(global_chain_count);
        comm.allreduce(variance);
        result.standard_error = global_chain_count > 1
            ? std::sqrt(variance * global_chain_count / (global_chain_count - 1.0))
            : std::numeric_limits<double>::infinity();
        result.valid = isFiniteTrainingReal(result.loss) && isFiniteTrainingReal(result.standard_error);
        for (std::size_t k = 0; k < candidate.channels.size(); ++k)
        {
          const auto& channel = candidate.channels[k];
          result.ess = std::min(result.ess, std::min(channel.student_effective_samples,
                                                     channel.target_effective_samples));
          // Guard each channel against a large deterioration hidden by the aggregate.
          if (channel.log_fidelity < overlap_reference->channels[k].log_fidelity - 0.1)
            result.valid = false;
        }
      }
      catch (...)
      {
        // Objective failures are synchronized by the overlap evaluator. A
        // singular or unreliable trial rejects this scale, permitting backtracking.
        result.valid = false;
      }
    }
    else
    {
      std::vector<double> logs, energies;
      if (energyValues(alpha, logs, energies))
      {
        for (std::size_t i = 0; i < logs.size(); ++i)
          logs[i] = 2.0 * (logs[i] - reference_log[i]);
        result = assessEnergy(comm, logs, energies, reference_energy, clusters, local_chain_count);
      }
    }
    std::size_t count = positions.size();
    comm.allreduce(count);
    result.valid = result.valid && result.ess >= input.getLineSearchMinimumESSFraction() * count &&
        result.maximum_weight <= input.getLineSearchMaximumWeight();
    report(alpha, result.loss, result.difference, result.standard_error, result.ess,
           result.maximum_weight, result.valid);
    assessments[alpha] = result;
    return result;
  };

  evaluate(0.0);
  const auto endpoint = evaluate(1.0);
  bool expand = input.getLineSearch() == WFTrainDriverInput::LineSearch::ENDPOINTS &&
      endpoint.valid && endpoint.difference + input.getLineSearchConfidence() * endpoint.standard_error < 0.0;

  // The endpoint route samples the private full-step wavefunction only when
  // expansion has evidence of improvement. Backtracking keeps reference samples.
  if (input.getLineSearch() == WFTrainDriverInput::LineSearch::ENDPOINTS && !overlap && maximum_alpha >= 1.0)
  {
    const auto original_positions = positions;
    const auto original_clusters = clusters;
    const auto original_logs = reference_log;
    const auto original_energies = reference_energy;
    positions.clear();
    clusters.clear();
    ready = true;
    try
    {
      install(1.0);
      for (std::size_t i = 0; i < configurations.size(); ++i)
      {
        particles->R = configurations[i].get().R;
        particles->update();
        sampleEndpoint(*particles, *wavefunction, *rng, timestep, input.getLineSearchWarmupSteps());
        for (std::size_t step = 0; step < input.getLineSearchSamplingSteps(); ++step)
        {
          sampleEndpoint(*particles, *wavefunction, *rng, timestep, 1);
          positions.push_back(particles->R);
          clusters.push_back(configuration_chain_ids[i]);
        }
      }
    }
    catch (...) { ready = false; }
    if (!allReady(comm, ready)) return 0.0;
    reference_log.clear();
    reference_energy.clear();
    if (!energyValues(1.0, reference_log, reference_energy)) return 0.0;
    assessments.clear();
    // The endpoint population is also used to reassess the unchanged state.
    const auto old_on_endpoint = evaluate(0.0);
    const auto full_on_endpoint = evaluate(1.0);
    expand = old_on_endpoint.valid && full_on_endpoint.valid &&
        old_on_endpoint.difference - input.getLineSearchConfidence() * old_on_endpoint.standard_error > 0.0;
    if (!expand)
    {
      positions = original_positions;
      clusters = original_clusters;
      reference_log = original_logs;
      reference_energy = original_energies;
      assessments.clear();
      evaluate(0.0);
      evaluate(1.0);
    }
  }
  const double backtrack_base = std::min(1.0, maximum_alpha);
  const std::vector<double> scales = expand
      ? std::vector<double>{0.5, 1.5, 2.0}
      : std::vector<double>{0.125 * backtrack_base, 0.25 * backtrack_base,
                            0.5 * backtrack_base, 0.75 * backtrack_base, backtrack_base};
  for (double alpha : scales) evaluate(alpha);
  double selected = 0.0;
  double best = 0.0;
  const auto baseline = assessments.at(0.0);
  if (!baseline.valid) return 0.0;
  for (const auto& [alpha, assessment] : assessments)
  {
    // For endpoint samples use a conservative sum of uncertainty bounds when
    // comparing two reweighted candidates; reference mode uses paired differences.
    const double difference = assessment.loss - baseline.loss;
    const double uncertainty = assessment.standard_error +
        (expand && !overlap ? baseline.standard_error : 0.0);
    const double score = difference + input.getLineSearchConfidence() * uncertainty;
    if (alpha != 0.0 && assessment.valid && score < best)
    {
      best = score;
      selected = alpha;
    }
  }
  return selected;
}

} // namespace qmcplusplus::wftrain
