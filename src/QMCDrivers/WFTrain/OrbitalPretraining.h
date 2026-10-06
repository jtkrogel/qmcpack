//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file OrbitalPretraining.h
 * @brief Driver-independent coordinator for PsiFormer orbital pretraining.
 */

#ifndef QMCPLUSPLUS_ORBITAL_PRETRAINING_H
#define QMCPLUSPLUS_ORBITAL_PRETRAINING_H

#include "QMCDrivers/WFTrain/HighParameterTraining.h"
#include "QMCDrivers/WFTrain/OrbitalPretrainingAccumulator.h"
#include "QMCDrivers/WFTrain/TrainingCheckpoint.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace qmcplusplus::wftrain
{

/** Stream one iteration's sampled orbital losses and direct VJPs into a bounded sink. */
class OrbitalPretrainingProducer
{
public:
  virtual ~OrbitalPretrainingProducer() = default;
  virtual TrainingCapabilities capabilities() const noexcept = 0;
  virtual void accumulate(const StructuredParameterSnapshot& parameters,
                          OrbitalPretrainingAccumulator& accumulator) = 0;
};

/// Result returned only after a complete orbital-pretraining update commits.
struct OrbitalPretrainingIterationResult
{
  std::size_t completed_iteration = 0;
  std::size_t completed_stage_iteration = 0;
  std::size_t parameter_version = 0;
  OrbitalPretrainingResult objective;
};

/** Coordinate target-bound production, reduction, and atomic parameter publication. */
class OrbitalPretrainingCoordinator
{
public:
  /// Stable stage identifier persisted by the Task 15 checkpoint envelope.
  static constexpr const char* STAGE_ID = "psiformer/orbital-pretraining";

  /** Stable fingerprint of the Stage A full-matrix, spin-normalized real loss. */
  static constexpr std::uint64_t LOSS_FINGERPRINT = UINT64_C(0x50464f52424d5345);

  OrbitalPretrainingCoordinator(std::uint64_t target_fingerprint,
                                DistributedParameterReduction reduction = {});

  /// Construct the exact checkpoint stage identity expected by this coordinator.
  TrainingStageState makeStageState(std::size_t stage_ordinal = 0) const;

  /// Run one failure-atomic replicated pretraining iteration.
  OrbitalPretrainingIterationResult runIteration(
      StructuredParameterProvider& provider,
      OrbitalPretrainingProducer& producer,
      TrainingUpdateRule& update_rule,
      TrainingIterationState& state,
      TrainingStageState& stage_state,
      ParameterUpdateObserver* observer = nullptr) const;

  /// Return the exact target identity bound to this coordinator.
  std::uint64_t targetFingerprint() const noexcept { return target_fingerprint_; }

  /// Return the checkpoint configuration identity derived from target and loss.
  const std::string& configurationFingerprint() const noexcept
  {
    return configuration_fingerprint_;
  }

private:
  std::uint64_t target_fingerprint_;
  std::string configuration_fingerprint_;
  DistributedParameterReduction reduction_;
};

} // namespace qmcplusplus::wftrain

#endif
