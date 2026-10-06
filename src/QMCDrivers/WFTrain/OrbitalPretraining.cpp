//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file OrbitalPretraining.cpp
 * @brief Target-bound, bounded PsiFormer orbital-pretraining iterations.
 */

#include "QMCDrivers/WFTrain/OrbitalPretraining.h"

#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qmcplusplus::wftrain
{
namespace
{

/// Render target/loss identities without locale-dependent decimal formatting.
std::string makeConfigurationFingerprint(std::uint64_t target, std::uint64_t loss)
{
  std::ostringstream text;
  text << "orbital-pretraining-v1:" << std::hex << std::setw(16) << std::setfill('0')
       << target << ':' << std::setw(16) << loss;
  return text.str();
}

/// Install prepared coordinator state without allocation after model publication.
void installIterationState(TrainingIterationState& state,
                           TrainingIterationState& pending) noexcept
{
  state.completed_iterations = pending.completed_iterations;
  state.parameter_version    = pending.parameter_version;
  state.schema_fingerprint.swap(pending.schema_fingerprint);
}

} // namespace

OrbitalPretrainingCoordinator::OrbitalPretrainingCoordinator(
    std::uint64_t target_fingerprint,
    DistributedParameterReduction reduction)
    : target_fingerprint_(target_fingerprint),
      configuration_fingerprint_(
          makeConfigurationFingerprint(target_fingerprint, LOSS_FINGERPRINT)),
      reduction_(std::move(reduction))
{}

TrainingStageState OrbitalPretrainingCoordinator::makeStageState(
    std::size_t stage_ordinal) const
{
  return {STAGE_ID, stage_ordinal, 0, configuration_fingerprint_};
}

OrbitalPretrainingIterationResult OrbitalPretrainingCoordinator::runIteration(
    StructuredParameterProvider& provider,
    OrbitalPretrainingProducer& producer,
    TrainingUpdateRule& update_rule,
    TrainingIterationState& state,
    TrainingStageState& stage_state,
    ParameterUpdateObserver* observer) const
{
  const StructuredParameterSchema& schema = provider.parameterSchema();
  std::optional<StructuredParameterSnapshot> parameters_storage;
  TrainingIterationState pending_state;
  std::size_t pending_stage_iterations = 0;
  std::unique_ptr<OrbitalPretrainingAccumulator> accumulator;
  std::exception_ptr preflight_failure;
  try
  {
    const TrainingCapabilities requirements{
        TrainingCapability::REAL_PARAMETERS, TrainingCapability::ORBITAL_MSE_VJP};
    requireTrainingCapabilities(producer.capabilities(), requirements,
                                "Orbital pretraining producer");
    parameters_storage.emplace(provider.snapshotParameters());
    const StructuredParameterSnapshot& parameters = *parameters_storage;
    if (parameters.schema_fingerprint != schema.fingerprint() ||
        parameters.values.size() != schema.parameterCount())
      throw std::logic_error("Structured provider returned an inconsistent snapshot");
    if (stage_state.stage_id != STAGE_ID ||
        stage_state.configuration_fingerprint != configuration_fingerprint_)
      throw std::invalid_argument("Orbital pretraining stage identity does not match its target");
    if (state.completed_iterations != 0 &&
        (state.schema_fingerprint != schema.fingerprint() ||
         state.parameter_version != parameters.version))
      throw std::invalid_argument("Orbital pretraining state does not match the provider");
    if (state.completed_iterations == std::numeric_limits<std::size_t>::max() ||
        stage_state.completed_stage_iterations == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Orbital pretraining iteration count overflow");

    pending_state = {state.completed_iterations + 1, parameters.version,
                     schema.fingerprint()};
    pending_stage_iterations = stage_state.completed_stage_iterations + 1;
    accumulator = std::make_unique<OrbitalPretrainingAccumulator>(
        schema, parameters.version, target_fingerprint_, LOSS_FINGERPRINT);
  }
  catch (...)
  {
    preflight_failure = std::current_exception();
  }
  reduction_.preflightOrbital(schema, parameters_storage ? &*parameters_storage : nullptr,
                              target_fingerprint_, LOSS_FINGERPRINT, preflight_failure);
  const StructuredParameterSnapshot& parameters = *parameters_storage;

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
  OrbitalPretrainingResult objective = accumulator->finalize();

  const std::size_t committed_version = completeParameterUpdate(
      provider, parameters, objective.parameterGradient(), update_rule, reduction_);
  pending_state.parameter_version = committed_version;
  installIterationState(state, pending_state);
  stage_state.completed_stage_iterations = pending_stage_iterations;
  if (observer)
    observer->parametersPublished(committed_version);

  return {state.completed_iterations, stage_state.completed_stage_iterations,
          committed_version, std::move(objective)};
}

} // namespace qmcplusplus::wftrain
