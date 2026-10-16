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
#include <memory>

namespace qmcplusplus::wftrain
{
/// Shared candidate-only admission; never fabricates or validates a host gradient.
void validateTrainingCandidate(const StructuredParameterSchema& schema,
    const StructuredParameterSnapshot& parameters, const StructuredParameterSnapshot& candidate);


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

  /// A selector may reject an update while returning the unchanged valid snapshot.
  virtual bool candidateAccepted() const noexcept { return true; }
};

/// Refresh sampler-side value/drift caches after atomic parameter publication.
class ParameterUpdateObserver
{
public:
  virtual ~ParameterUpdateObserver() = default;
  virtual void parametersPublished(std::size_t new_version) noexcept = 0;
};

/** Own a fully prepared model replacement and its publication authority.
 *
 * Preparation must resolve every allocation, finite check, device wait and model
 * leaf before returning this action. Until commit, destruction only discards the
 * candidate. Commit is nonallocating/no-fail and returns expected_version + 1;
 * the owner must exclude intervening publication/proposals through this boundary.
 * Irrecoverable device/communication loss still requires process abort/restart,
 * not an exception-based rollback of an already committed model.
 */
class PreparedParameterPublication
{
public:
  virtual ~PreparedParameterPublication() = default;
  virtual std::size_t commit() noexcept = 0;
};

/// Optional provider role; legacy single-phase providers retain their current path.
class PreparedParameterPublicationContributor
{
public:
  virtual ~PreparedParameterPublicationContributor() = default;
  virtual std::unique_ptr<PreparedParameterPublication> prepareParameterPublication(
      const StructuredParameterSnapshot& candidate, std::size_t expected_version) = 0;
};

/** Validate, publish, and accept one objective-neutral update transaction.
 *
 * The optimizer recurrence is committed only after every rank reports successful
 * provider publication. Optional prepared providers agree readiness before any
 * no-fail commit; preparation failure discards the candidate on every rank. Legacy
 * providers still publish in one phase. Postcommit failure is diagnosed as fatal
 * because an already-published provider cannot in general be rolled back.
 */
std::size_t completeParameterUpdate(
    StructuredParameterProvider& provider,
    const StructuredParameterSnapshot& parameters,
    ParameterGradientView objective,
    TrainingUpdateRule& update_rule,
    const DistributedParameterReduction& reduction);

} // namespace qmcplusplus::wftrain

#endif
