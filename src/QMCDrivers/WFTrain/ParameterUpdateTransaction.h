//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file ParameterUpdateTransaction.h
 * @brief Objective-neutral failure-atomic parameter update publication.
 */

#ifndef QMCPLUSPLUS_PARAMETER_UPDATE_TRANSACTION_H
#define QMCPLUSPLUS_PARAMETER_UPDATE_TRANSACTION_H

#include "QMCDrivers/WFTrain/DistributedParameterReduction.h"
#include "QMCDrivers/WFTrain/ParameterGradient.h"

#include <cstddef>

namespace qmcplusplus::wftrain
{

/// Convert one finalized gradient into a complete candidate parameter snapshot.
class TrainingUpdateRule
{
public:
  virtual ~TrainingUpdateRule() = default;

  /// Construct one speculative candidate without changing committed optimizer state.
  virtual StructuredParameterSnapshot propose(
      const StructuredParameterSchema& schema,
      const StructuredParameterSnapshot& parameters,
      ParameterGradientView objective) = 0;

  /** Commit recurrence state after candidate parameters are globally published. */
  virtual void proposalAccepted(const StructuredParameterSchema& schema,
                                const StructuredParameterSnapshot& parameters,
                                ParameterGradientView objective) noexcept
  {}

  /// Discard a speculative proposal after any recoverable post-proposal failure.
  virtual void proposalRejected() noexcept {}
};

/// Refresh sampler-side value/drift caches after atomic parameter publication.
class ParameterUpdateObserver
{
public:
  virtual ~ParameterUpdateObserver() = default;
  virtual void parametersPublished(std::size_t new_version) noexcept = 0;
};

/** Validate, publish, and accept one objective-neutral update transaction.
 *
 * The optimizer recurrence is committed only after every rank reports successful
 * provider publication. Candidate failures and uniformly failed publications reject
 * the proposal. A mixed cross-rank publication outcome is diagnosed as fatal because
 * an already-published provider cannot in general be rolled back.
 */
std::size_t completeParameterUpdate(
    StructuredParameterProvider& provider,
    const StructuredParameterSnapshot& parameters,
    ParameterGradientView objective,
    TrainingUpdateRule& update_rule,
    const DistributedParameterReduction& reduction);

} // namespace qmcplusplus::wftrain

#endif
