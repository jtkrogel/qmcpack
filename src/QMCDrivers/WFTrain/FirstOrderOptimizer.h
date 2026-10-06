//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file FirstOrderOptimizer.h
 * @brief Bounded SGD-family update rules for the high-parameter training route.
 */

#ifndef QMCPLUSPLUS_FIRST_ORDER_OPTIMIZER_H
#define QMCPLUSPLUS_FIRST_ORDER_OPTIMIZER_H

#include "QMCDrivers/WFTrain/TrainingCheckpoint.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qmcplusplus::wftrain
{

/// Select one supported first-order recurrence.
enum class FirstOrderMethod : std::uint8_t
{
  SGD,
  MOMENTUM_SGD,
  RMSPROP,
  ADAM
};

/// Assign one positive constant learning rate to a structured update group.
struct UpdateGroupLearningRate
{
  std::string update_group;
  double learning_rate = 0.0;
};

/// Configure one bounded first-order optimizer instance.
struct FirstOrderOptimizerOptions
{
  FirstOrderMethod method = FirstOrderMethod::ADAM;
  std::vector<UpdateGroupLearningRate> learning_rates;
  double momentum_decay = 0.9;
  double rms_decay      = 0.9;
  double adam_beta1     = 0.9;
  double adam_beta2     = 0.999;
  double epsilon        = 1.0e-8;
};

/** Apply SGD, momentum SGD, RMSProp, or Adam to a structured real parameter vector.
 *
 * Candidate construction is transactional: committed recurrence vectors change
 * only in proposalAccepted(), after the coordinator publishes model parameters.
 */
class FirstOrderOptimizer final : public TrainingUpdateRule,
                                  public OptimizerCheckpointContributor
{
public:
  FirstOrderOptimizer(const StructuredParameterSchema& schema,
                      FirstOrderOptimizerOptions options);

  StructuredParameterSnapshot propose(
      const StructuredParameterSchema& schema,
      const StructuredParameterSnapshot& parameters,
      ParameterGradientView objective) override;

  /// Preserve direct source compatibility for callers retaining energy diagnostics.
  StructuredParameterSnapshot propose(
      const StructuredParameterSchema& schema,
      const StructuredParameterSnapshot& parameters,
      const EnergyGradientResult& objective)
  {
    return propose(schema, parameters, objective.parameterGradient());
  }

  void proposalAccepted(const StructuredParameterSchema& schema,
                        const StructuredParameterSnapshot& parameters,
                        ParameterGradientView objective) noexcept override;

  /// Preserve direct source compatibility for callers retaining energy diagnostics.
  void proposalAccepted(const StructuredParameterSchema& schema,
                        const StructuredParameterSnapshot& parameters,
                        const EnergyGradientResult& objective) noexcept
  {
    proposalAccepted(schema, parameters, objective.parameterGradient());
  }

  void proposalRejected() noexcept override;

  CheckpointContributorMetadata checkpointMetadata() const override;

  void writeCheckpointPayload(hdf_archive& archive) const override;

  std::unique_ptr<PreparedRestore> prepareCheckpointRestore(
      hdf_archive& archive,
      const CheckpointContributorMetadata& saved_metadata) override;

  /// Return the number of successfully published optimizer updates.
  std::uint64_t acceptedUpdateCount() const noexcept { return accepted_update_count_; }

  /// Return numeric recurrence storage retained by this optimizer.
  std::size_t retainedBytes() const noexcept;

  /// Expose the committed first moment for deterministic validation.
  const std::vector<double>& firstMoment() const noexcept { return first_moment_; }

  /// Expose the committed second moment for deterministic validation.
  const std::vector<double>& secondMoment() const noexcept { return second_moment_; }

  /// Report whether one candidate currently awaits acceptance or rejection.
  bool hasLiveProposal() const noexcept { return proposal_live_; }

private:
  /// Compact per-block update metadata avoids per-parameter configuration storage.
  struct BlockUpdate
  {
    std::size_t offset    = 0;
    std::size_t count     = 0;
    bool trainable        = false;
    double learning_rate  = 0.0;
  };

  class PreparedRestoreAction;

  void validateInputs(const StructuredParameterSchema& schema,
                      const StructuredParameterSnapshot& parameters,
                      ParameterGradientView objective) const;
  void commitRecurrence(ParameterGradientView objective) noexcept;
  void validateRestoredState(std::uint64_t update_count,
                             const std::vector<double>& first_moment,
                             const std::vector<double>& second_moment) const;
  std::string stateFingerprint(std::uint64_t update_count,
                               const std::vector<double>& first_moment,
                               const std::vector<double>& second_moment) const;

  FirstOrderOptimizerOptions options_;
  std::string provider_id_;
  std::string schema_fingerprint_;
  std::string configuration_fingerprint_;
  std::size_t parameter_count_ = 0;
  std::vector<BlockUpdate> blocks_;
  std::vector<double> first_moment_;
  std::vector<double> second_moment_;
  std::uint64_t accepted_update_count_ = 0;
  bool proposal_live_ = false;
};

} // namespace qmcplusplus::wftrain

#endif
