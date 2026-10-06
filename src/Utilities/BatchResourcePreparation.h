//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: QMCPACK developers
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_BATCH_RESOURCE_PREPARATION_H
#define QMCPLUSPLUS_BATCH_RESOURCE_PREPARATION_H

#include "BatchExecutionMemory.h"

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace qmcplusplus
{

/** Identifies the immutable section plan and crowd whose cloned resources are being prepared.
 *
 * A null plan is the explicit no-policy state.  It is still forwarded to resources so a
 * reused resource can clear policy-specific state without needing a leader walker.
 */
struct BatchResourcePreparationContext
{
  std::shared_ptr<const BatchExecutionPlan> plan;
  std::size_t crowd_index = 0;

  /** Validate the crowd identity before any resource clone is mutated. */
  void validate() const
  {
    if (!plan)
      return;

    const BatchExecutionTopology& topology = plan->topology();
    validateBatchExecutionTopology(topology);
    const std::size_t crowd_count = topology.initial_walkers_per_crowd.size();
    if (crowd_index >= crowd_count)
      throw std::out_of_range("Batch resource preparation crowd index is outside the planned topology");
  }

  /** Return the number of initially living walkers assigned to this crowd. */
  std::size_t initialWalkerCapacity() const
  {
    validate();
    if (!plan)
      throw std::logic_error("A no-policy batch resource context has no initial walker capacity");
    return plan->topology().initial_walkers_per_crowd[crowd_index];
  }

  /** Return the admitted reserve envelope, falling back to the initial topology when absent. */
  std::size_t reserveWalkerCapacity() const
  {
    validate();
    if (!plan)
      throw std::logic_error("A no-policy batch resource context has no reserve walker capacity");

    const BatchExecutionTopology& topology = plan->topology();
    return topology.reserve_walkers_per_crowd.empty() ? topology.initial_walkers_per_crowd[crowd_index]
                                                      : topology.reserve_walkers_per_crowd[crowd_index];
  }
};

} // namespace qmcplusplus

#endif // QMCPLUSPLUS_BATCH_RESOURCE_PREPARATION_H
