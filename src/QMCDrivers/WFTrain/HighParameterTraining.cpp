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
  objective.emplace(accumulator->finalize());
  const std::size_t committed_version = completeParameterUpdate(
      provider, parameters, objective->parameterGradient(), update_rule, reduction_);

  // The shared transaction has committed optimizer and provider state.  Publish
  // coordinator state before notifying sampler-side caches.
  pending_state.parameter_version = committed_version;
  installTrainingState(state, pending_state);
  if (observer)
    observer->parametersPublished(committed_version);

  return {state.completed_iterations, committed_version, std::move(*objective)};
}

} // namespace qmcplusplus::wftrain
