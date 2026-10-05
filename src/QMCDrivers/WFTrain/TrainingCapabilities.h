//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file TrainingCapabilities.h
 * @brief Fail-fast capability negotiation for the high-parameter training route.
 */

#ifndef QMCPLUSPLUS_TRAINING_CAPABILITIES_H
#define QMCPLUSPLUS_TRAINING_CAPABILITIES_H

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace qmcplusplus::wftrain
{

/// Enumerate independently negotiable model and execution capabilities.
enum class TrainingCapability : std::uint8_t
{
  REAL_PARAMETERS,
  COMPLEX_PARAMETERS,
  VALUE_EVALUATION,
  SPATIAL_DERIVATIVES,
  SCORE_VJP,
  SCORE_JVP,
  LOCAL_ENERGY_VJP,
  NONLOCAL_ECP,
  ALL_ELECTRON_MOVES,
  SUBSET_ELECTRON_MOVES,
  MULTIWALKER_BATCHING,
  SHARED_MODEL_THREADING,
  DISTRIBUTED_REDUCTION,
  DEVICE_EXECUTION
};

/// Store a compact set of training capabilities and produce readable diagnostics.
class TrainingCapabilities
{
public:
  TrainingCapabilities() = default;
  TrainingCapabilities(std::initializer_list<TrainingCapability> capabilities);

  /// Add one supported or required capability.
  void add(TrainingCapability capability) noexcept;

  /// Return whether one capability is present.
  bool contains(TrainingCapability capability) const noexcept;

  /// Return required capabilities absent from this set.
  std::vector<TrainingCapability> missing(const TrainingCapabilities& required) const;

  /// Return the raw mask for persistence and tests.
  std::uint64_t mask() const noexcept { return mask_; }

private:
  std::uint64_t mask_ = 0;
};

/// Return the stable input/diagnostic name of one capability.
const char* trainingCapabilityName(TrainingCapability capability) noexcept;

/// Throw one actionable error listing every unmet requirement.
void requireTrainingCapabilities(const TrainingCapabilities& offered,
                                 const TrainingCapabilities& required,
                                 const std::string& context);

} // namespace qmcplusplus::wftrain

#endif

