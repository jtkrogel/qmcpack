//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file HighParameterTraining.h
 * @brief Driver-independent atomic iteration coordinator for scalable training.
 */

#ifndef QMCPLUSPLUS_HIGH_PARAMETER_TRAINING_H
#define QMCPLUSPLUS_HIGH_PARAMETER_TRAINING_H

#include "QMCDrivers/WFTrain/EnergyGradientAccumulator.h"
#include "QMCDrivers/WFTrain/TrainingCapabilities.h"

#include <cstddef>
#include <string>
#include <vector>

namespace qmcplusplus::wftrain
{

/// Minimal deterministic state carried between committed training iterations.
struct TrainingIterationState
{
  std::size_t completed_iterations = 0;
  std::size_t parameter_version    = 0;
  std::string schema_fingerprint;
};

/// Produce one iteration's scalar statistics and derivative contractions.
class GradientProducer
{
public:
  virtual ~GradientProducer() = default;
  virtual TrainingCapabilities capabilities() const noexcept = 0;
  virtual void accumulate(const StructuredParameterSnapshot& parameters,
                          EnergyGradientAccumulator& accumulator) = 0;
};

/// Convert one finalized gradient into a complete candidate parameter snapshot.
class TrainingUpdateRule
{
public:
  virtual ~TrainingUpdateRule() = default;
  virtual StructuredParameterSnapshot propose(
      const StructuredParameterSchema& schema,
      const StructuredParameterSnapshot& parameters,
      const EnergyGradientResult& objective) = 0;
};

/// Refresh sampler-side value/drift caches after atomic parameter publication.
class ParameterUpdateObserver
{
public:
  virtual ~ParameterUpdateObserver() = default;
  virtual void parametersPublished(std::size_t new_version) noexcept = 0;
};

/// Result returned only after a complete parameter update has committed.
struct TrainingIterationResult
{
  std::size_t completed_iteration = 0;
  std::size_t parameter_version   = 0;
  EnergyGradientResult objective;
};

/** Enforce preflight, reduction, update, publication, and cache-refresh order.
 *
 * This class deliberately owns no sampler, model, optimizer, or checkpoint
 * implementation.  Later tasks plug those roles into the stable barriers.
 */
class HighParameterTraining
{
public:
  explicit HighParameterTraining(
      TrainingCapabilities requirements,
      EnergyGradientConvention convention = EnergyGradientConvention::REAL_VMC)
      : requirements_(requirements), convention_(convention)
  {}

  /// Execute one failure-atomic local training iteration.
  TrainingIterationResult runIteration(StructuredParameterProvider& provider,
                                       GradientProducer& producer,
                                       TrainingUpdateRule& update_rule,
                                       TrainingIterationState& state,
                                       ParameterUpdateObserver* observer = nullptr) const;

private:
  TrainingCapabilities requirements_;
  EnergyGradientConvention convention_;
};

} // namespace qmcplusplus::wftrain

#endif

