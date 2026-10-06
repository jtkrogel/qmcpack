//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file HighParameterTraining.cpp
 * @brief Failure-atomic local training iteration orchestration.
 */

#include "QMCDrivers/WFTrain/HighParameterTraining.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace qmcplusplus::wftrain
{
namespace
{

/// Install a completely prepared iteration state without any throwing operation.
void installTrainingState(TrainingIterationState& state,
                          TrainingIterationState& pending_state) noexcept
{
  state.completed_iterations = pending_state.completed_iterations;
  state.parameter_version    = pending_state.parameter_version;
  state.schema_fingerprint.swap(pending_state.schema_fingerprint);
}

} // namespace

void LocalTrainingReduction::preflight() const
{
  if (participant_count != 1)
    throw std::invalid_argument(
        "Local high-parameter reduction supports exactly one participant; "
        "distributed transport is not available in Task 10");
}

void LocalTrainingReduction::complete(EnergyGradientAccumulator& accumulator) const
{
  preflight();
  accumulator.completeSingleParticipantReduction();
}

HighParameterTraining::HighParameterTraining(
    TrainingCapabilities requirements,
    EnergyGradientEstimator estimator,
    LocalTrainingReduction reduction)
    : requirements_(requirements), estimator_(estimator), reduction_(reduction)
{
  // Every energy-gradient iteration requires both elementary products regardless
  // of additional model- or Hamiltonian-specific requirements supplied by the caller.
  requirements_.add(TrainingCapability::REAL_PARAMETERS);
  requirements_.add(TrainingCapability::SCORE_VJP);
  requirements_.add(TrainingCapability::LOCAL_ENERGY_VJP);
}

TrainingIterationResult HighParameterTraining::runIteration(
    StructuredParameterProvider& provider,
    GradientProducer& producer,
    TrainingUpdateRule& update_rule,
    TrainingIterationState& state,
    ParameterUpdateObserver* observer) const
{
  // Capability and transport failures must precede derivative workspace allocation
  // and sampling.
  requireTrainingCapabilities(producer.capabilities(), requirements_,
                              "High-parameter gradient producer");
  reduction_.preflight();

  const StructuredParameterSchema& schema       = provider.parameterSchema();
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  if (parameters.schema_fingerprint != schema.fingerprint() ||
      parameters.values.size() != schema.parameterCount())
    throw std::logic_error("Structured provider returned an inconsistent snapshot");

  if (state.completed_iterations != 0)
  {
    if (state.schema_fingerprint != schema.fingerprint())
      throw std::invalid_argument("Training state schema does not match the parameter provider");
    if (state.parameter_version != parameters.version)
      throw std::invalid_argument("Training state parameter version does not match the provider");
  }

  // Prepare every potentially throwing state operation before producer work and
  // parameter publication. The committed version itself is a nonthrowing scalar
  // assignment once publishParameters() succeeds.
  if (state.completed_iterations == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("High-parameter training iteration count overflow");
  TrainingIterationState pending_state;
  pending_state.completed_iterations = state.completed_iterations + 1;
  pending_state.parameter_version    = parameters.version;
  pending_state.schema_fingerprint   = schema.fingerprint();

  EnergyGradientAccumulator accumulator(schema, parameters.version, estimator_);
  producer.accumulate(parameters, accumulator);
  reduction_.complete(accumulator);
  EnergyGradientResult objective = accumulator.finalize();
  StructuredParameterSnapshot candidate =
      update_rule.propose(schema, parameters, objective);

  if (candidate.schema_fingerprint != schema.fingerprint() ||
      candidate.version != parameters.version ||
      candidate.values.size() != parameters.values.size())
    throw std::invalid_argument("Training update rule returned an incompatible candidate snapshot");

  // Frozen tensors are an invariant of the schema rather than an update-rule convention.
  for (const ParameterBlockDescriptor& block : schema.blocks())
    if (!block.trainable)
      for (std::size_t parameter = block.offset; parameter < block.offset + block.count; ++parameter)
        if (candidate.values[parameter] != parameters.values[parameter])
          throw std::invalid_argument("Training update rule modified frozen parameter block " + block.id);

  const std::size_t committed_version =
      provider.publishParameters(candidate, parameters.version);

  // All post-publication state operations are nonthrowing. The observer is
  // notified only after both the provider and iteration state expose the commit.
  pending_state.parameter_version = committed_version;
  installTrainingState(state, pending_state);
  if (observer)
    observer->parametersPublished(committed_version);

  return {state.completed_iterations, committed_version, std::move(objective)};
}

} // namespace qmcplusplus::wftrain
