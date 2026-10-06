//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file TrainingCapabilities.cpp
 * @brief Training capability names and validation.
 */

#include "QMCDrivers/WFTrain/TrainingCapabilities.h"

#include <sstream>
#include <stdexcept>

namespace qmcplusplus::wftrain
{
namespace
{

/// Convert one enum value to its mask bit.
std::uint64_t capabilityBit(TrainingCapability capability) noexcept
{
  return UINT64_C(1) << static_cast<std::uint8_t>(capability);
}

} // namespace

TrainingCapabilities::TrainingCapabilities(
    std::initializer_list<TrainingCapability> capabilities)
{
  for (const TrainingCapability capability : capabilities)
    add(capability);
}

void TrainingCapabilities::add(TrainingCapability capability) noexcept
{
  mask_ |= capabilityBit(capability);
}

bool TrainingCapabilities::contains(TrainingCapability capability) const noexcept
{
  return (mask_ & capabilityBit(capability)) != 0;
}

std::vector<TrainingCapability> TrainingCapabilities::missing(
    const TrainingCapabilities& required) const
{
  std::vector<TrainingCapability> result;
  for (std::uint8_t index = 0;
       index <= static_cast<std::uint8_t>(TrainingCapability::DEVICE_EXECUTION); ++index)
  {
    const TrainingCapability capability = static_cast<TrainingCapability>(index);
    if (required.contains(capability) && !contains(capability))
      result.push_back(capability);
  }
  return result;
}

const char* trainingCapabilityName(TrainingCapability capability) noexcept
{
  switch (capability)
  {
  case TrainingCapability::REAL_PARAMETERS:
    return "real_parameters";
  case TrainingCapability::COMPLEX_PARAMETERS:
    return "complex_parameters";
  case TrainingCapability::VALUE_EVALUATION:
    return "value_evaluation";
  case TrainingCapability::SPATIAL_DERIVATIVES:
    return "spatial_derivatives";
  case TrainingCapability::SCORE_VJP:
    return "score_vjp";
  case TrainingCapability::SCORE_JVP:
    return "score_jvp";
  case TrainingCapability::LOCAL_ENERGY_VJP:
    return "local_energy_vjp";
  case TrainingCapability::NONLOCAL_ECP:
    return "nonlocal_ecp";
  case TrainingCapability::ALL_ELECTRON_MOVES:
    return "all_electron_moves";
  case TrainingCapability::SUBSET_ELECTRON_MOVES:
    return "subset_electron_moves";
  case TrainingCapability::MULTIWALKER_BATCHING:
    return "multiwalker_batching";
  case TrainingCapability::SHARED_MODEL_THREADING:
    return "shared_model_threading";
  case TrainingCapability::DISTRIBUTED_REDUCTION:
    return "distributed_reduction";
  case TrainingCapability::ORBITAL_MSE_VJP:
    return "orbital_mse_vjp";
  case TrainingCapability::DEVICE_EXECUTION:
    return "device_execution";
  }
  return "unknown";
}

void requireTrainingCapabilities(const TrainingCapabilities& offered,
                                 const TrainingCapabilities& required,
                                 const std::string& context)
{
  const std::vector<TrainingCapability> missing = offered.missing(required);
  if (missing.empty())
    return;

  std::ostringstream message;
  message << context << " is missing training capabilities:";
  for (const TrainingCapability capability : missing)
    message << ' ' << trainingCapabilityName(capability);
  throw std::invalid_argument(message.str());
}

} // namespace qmcplusplus::wftrain
