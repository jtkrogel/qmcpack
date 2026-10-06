//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file ParameterUpdateTransaction.cpp
 * @brief Shared candidate validation and atomic publication transaction.
 */

#include "QMCDrivers/WFTrain/ParameterUpdateTransaction.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"

#include <algorithm>
#include <exception>
#include <optional>
#include <stdexcept>

namespace qmcplusplus::wftrain
{

std::size_t completeParameterUpdate(
    StructuredParameterProvider& provider,
    const StructuredParameterSnapshot& parameters,
    ParameterGradientView objective,
    TrainingUpdateRule& update_rule,
    const DistributedParameterReduction& reduction)
{
  const StructuredParameterSchema& schema = provider.parameterSchema();
  std::optional<StructuredParameterSnapshot> candidate;
  std::exception_ptr update_failure;
  bool proposal_started = false;
  bool proposal_live = false;
  try
  {
    if (parameters.schema_fingerprint != schema.fingerprint() ||
        parameters.values.size() != schema.parameterCount())
      throw std::invalid_argument("Training parameter snapshot is incompatible with the provider");
    if (!std::all_of(parameters.values.begin(), parameters.values.end(),
                     isFiniteTrainingReal))
      throw std::invalid_argument("Training parameter snapshot is non-finite");
    if (objective.schema_fingerprint != schema.fingerprint() ||
        objective.parameter_version != parameters.version ||
        objective.gradient.size() != schema.parameterCount() ||
        (objective.gradient.size() != 0 && !objective.gradient.data()))
      throw std::invalid_argument("Training parameter gradient is incompatible with the provider");
    if (objective.reduction_domain != ReductionDomain::GLOBAL)
      throw std::invalid_argument("Training parameter gradient is not globally reduced");
    if (!std::all_of(objective.gradient.begin(), objective.gradient.end(),
                     isFiniteTrainingReal))
      throw std::invalid_argument("Training parameter gradient is non-finite");

    proposal_started = true;
    candidate.emplace(update_rule.propose(schema, parameters, objective));
    proposal_live = true;
    if (candidate->schema_fingerprint != schema.fingerprint() ||
        candidate->version != parameters.version ||
        candidate->values.size() != schema.parameterCount())
      throw std::invalid_argument("Training update rule returned an incompatible candidate snapshot");
    if (!std::all_of(candidate->values.begin(), candidate->values.end(),
                     isFiniteTrainingReal))
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
    if (proposal_started)
      update_rule.proposalRejected();
    proposal_live = false;
    update_failure = std::current_exception();
  }

  try
  {
    reduction.validateCandidate(schema, parameters,
                                candidate ? &*candidate : nullptr, update_failure);
  }
  catch (...)
  {
    if (proposal_live)
      update_rule.proposalRejected();
    throw;
  }

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

  std::size_t committed_version = parameters.version;
  try
  {
    committed_version =
        reduction.completePublication(local_committed_version, publication_failure);
  }
  catch (...)
  {
    update_rule.proposalRejected();
    throw;
  }

  update_rule.proposalAccepted(schema, parameters, objective);
  return committed_version;
}

} // namespace qmcplusplus::wftrain
