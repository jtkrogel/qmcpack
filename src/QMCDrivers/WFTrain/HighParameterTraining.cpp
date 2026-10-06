//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file HighParameterTraining.cpp
 * @brief Failure-atomic replicated training iteration orchestration.
 */

#include "QMCDrivers/WFTrain/HighParameterTraining.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
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

HighParameterTraining::HighParameterTraining(
    TrainingCapabilities requirements,
    EnergyGradientEstimator estimator,
    DistributedParameterReduction reduction)
    : requirements_(requirements), estimator_(estimator), reduction_(std::move(reduction))
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
  const StructuredParameterSchema& schema = provider.parameterSchema();
  std::optional<StructuredParameterSnapshot> parameters_storage;
  TrainingIterationState pending_state;
  std::unique_ptr<EnergyGradientAccumulator> accumulator;
  std::exception_ptr preflight_failure;
  try
  {
    // Complete all rank-local checks and O(P) allocations before the pre-sampling
    // manifest. A rank that fails here still participates in that fixed control record.
    requireTrainingCapabilities(producer.capabilities(), requirements_,
                                "High-parameter gradient producer");
    parameters_storage.emplace(provider.snapshotParameters());
    const StructuredParameterSnapshot& parameters = *parameters_storage;
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

    if (state.completed_iterations == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("High-parameter training iteration count overflow");
    pending_state.completed_iterations = state.completed_iterations + 1;
    pending_state.parameter_version    = parameters.version;
    pending_state.schema_fingerprint   = schema.fingerprint();
    accumulator =
        std::make_unique<EnergyGradientAccumulator>(schema, parameters.version, estimator_);
  }
  catch (...)
  {
    preflight_failure = std::current_exception();
  }
  reduction_.preflight(schema,
                       parameters_storage ? &*parameters_storage : nullptr,
                       estimator_, preflight_failure);
  const StructuredParameterSnapshot& parameters = *parameters_storage;

  // A local producer exception is converted into a fixed readiness record; no rank
  // enters the scalar or parameter all-reduces until every producer is complete.
  std::exception_ptr production_failure;
  try
  {
    producer.accumulate(parameters, *accumulator);
  }
  catch (...)
  {
    production_failure = std::current_exception();
  }
  reduction_.reduce(*accumulator, production_failure);

  std::optional<EnergyGradientResult> objective;
  std::optional<StructuredParameterSnapshot> candidate;
  std::exception_ptr update_failure;
  try
  {
    objective.emplace(accumulator->finalize());
    candidate.emplace(update_rule.propose(schema, parameters, *objective));
    if (candidate->schema_fingerprint != schema.fingerprint() ||
        candidate->version != parameters.version ||
        candidate->values.size() != parameters.values.size())
      throw std::invalid_argument("Training update rule returned an incompatible candidate snapshot");
    if (!std::all_of(candidate->values.begin(), candidate->values.end(),
                     [](double value) { return std::isfinite(value); }))
      throw std::invalid_argument("Training update rule returned a non-finite candidate snapshot");

    // Frozen tensors are a schema invariant rather than an update-rule convention.
    for (const ParameterBlockDescriptor& block : schema.blocks())
      if (!block.trainable)
        for (std::size_t parameter = block.offset; parameter < block.offset + block.count;
             ++parameter)
          if (candidate->values[parameter] != parameters.values[parameter])
            throw std::invalid_argument("Training update rule modified frozen parameter block " +
                                        block.id);
  }
  catch (...)
  {
    update_failure = std::current_exception();
  }
  reduction_.validateCandidate(schema, parameters,
                               candidate ? &*candidate : nullptr, update_failure);

  std::size_t local_committed_version = parameters.version;
  std::exception_ptr publication_failure;
  try
  {
    local_committed_version = provider.publishParameters(*candidate, parameters.version);
  }
  catch (...)
  {
    publication_failure = std::current_exception();
  }
  const std::size_t committed_version =
      reduction_.completePublication(local_committed_version, publication_failure);

  // All post-publication state operations are nonthrowing. The observer is
  // notified only after both the provider and iteration state expose the commit.
  pending_state.parameter_version = committed_version;
  installTrainingState(state, pending_state);
  if (observer)
    observer->parametersPublished(committed_version);

  return {state.completed_iterations, committed_version, std::move(*objective)};
}

} // namespace qmcplusplus::wftrain
