//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file ProposalSelection.h
 * @brief Objective-independent scaling, shrinkage, and speculative step selection.
 */
#ifndef QMCPLUSPLUS_PROPOSAL_SELECTION_H
#define QMCPLUSPLUS_PROPOSAL_SELECTION_H

#include "QMCDrivers/WFTrain/ParameterUpdateTransaction.h"
#include <functional>
#include <vector>

namespace qmcplusplus::wftrain
{

/// Controls applied to a complete optimizer displacement before publication.
struct ProposalSelectionControl
{
  double update_scale = 1.0;
  double weight_decay = 0.0;
  double maximum_update_norm = 0.0;
  std::vector<std::string> decay_blocks;
};

/** Decorate an optimizer without changing its recurrence during candidate trials.
 * The selector receives the old parameters and combined full-step candidate and
 * returns a relative scale. Zero rejects the proposal without provider publication.
 */
class SelectedUpdateRule final : public TrainingUpdateRule
{
public:
  using Selector = std::function<double(const StructuredParameterSnapshot&,
                                        const StructuredParameterSnapshot&)>;

  /// Bind a live optimizer, explicit decay selection, and optional candidate evaluator.
  SelectedUpdateRule(TrainingUpdateRule& optimizer, ProposalSelectionControl control,
                     Selector selector = {});

  /// Form the combined displacement, constrain it, and select one candidate.
  StructuredParameterSnapshot propose(const StructuredParameterSchema& schema,
                                      const StructuredParameterSnapshot& parameters,
                                      ParameterGradientView objective) override;

  /// Commit the underlying recurrence only for an accepted nonzero step.
  void proposalAccepted(const StructuredParameterSchema& schema,
                        const StructuredParameterSnapshot& parameters,
                        ParameterGradientView objective) noexcept override;

  /// Discard the underlying speculative recurrence after failure or rejection.
  void proposalRejected() noexcept override;

  /// Tell the publication transaction whether selection accepted a nonzero step.
  bool candidateAccepted() const noexcept override { return alpha_ != 0.0; }

  double selectedAlpha() const noexcept { return alpha_; }
  double displacementNorm() const noexcept { return displacement_norm_; }

private:
  TrainingUpdateRule& optimizer_;
  ProposalSelectionControl control_;
  Selector selector_;
  double alpha_ = 1.0;
  double displacement_norm_ = 0.0;
};

} // namespace qmcplusplus::wftrain
#endif
