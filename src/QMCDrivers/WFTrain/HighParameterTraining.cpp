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

#include <stdexcept>

namespace qmcplusplus::wftrain
{

TrainingIterationResult HighParameterTraining::runIteration(
    StructuredParameterProvider& provider,
    GradientProducer& producer,
    TrainingUpdateRule& update_rule,
    TrainingIterationState& state,
    ParameterUpdateObserver* observer) const
{
  // Capability failure must precede derivative workspace allocation and sampling.
  requireTrainingCapabilities(producer.capabilities(), requirements_,
                              "High-parameter gradient producer");

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

  EnergyGradientAccumulator accumulator(schema, convention_);
  producer.accumulate(parameters, accumulator);
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

  // State becomes visible only after model publication succeeds. The observer
  // is noexcept so sampler cache refresh cannot leave an ambiguous commit state.
  state.completed_iterations += 1;
  state.parameter_version    = committed_version;
  state.schema_fingerprint   = schema.fingerprint();
  if (observer)
    observer->parametersPublished(committed_version);

  return {state.completed_iterations, committed_version, std::move(objective)};
}

} // namespace qmcplusplus::wftrain

