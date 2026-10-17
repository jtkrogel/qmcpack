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
#include <functional>

namespace qmcplusplus::psiformer::device
{
/// Optional real-Adam recurrence owner; never a parameter publication authority.
class PreparedDeviceAdam;
}

namespace qmcplusplus::wftrain
{

/// Select one supported first-order recurrence.
enum class FirstOrderMethod : std::uint8_t
{
  SGD,
  MOMENTUM_SGD,
  RMSPROP,
  ADAM,
  AMSGRAD
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

/** Apply SGD, momentum SGD, RMSProp, Adam, or AMSGrad to real parameters.
 *
 * Candidate construction is transactional: committed recurrence vectors change
 * only in proposalAccepted(), after the coordinator publishes model parameters.
 * AMSGrad retains Adam's ordinary second-moment EMA v and an independent
 * v_max = max(v_max, v). Its denominator is sqrt(v_max / (1-beta2^t)) + epsilon,
 * and its numerator is m / (1-beta1^t), using accepted update count t.
 */
class FirstOrderOptimizer final : public TrainingUpdateRule,
                                  public OptimizerCheckpointContributor
{
public:
  /// Scoped numeric callback; its producer leases must end before this returns.
  using ResidentAdamConsumer = std::function<double(psiformer::device::PreparedDeviceAdam&,
      double, double, double, double, double, std::vector<double>&)>;

  /// Admit the optional small device/host norm capacities before sampling.
  void prepareResidentAdamNorm(std::size_t bytes);

  /// A distinct proposal contract, never an empty or synthetic host gradient.
  StructuredParameterSnapshot proposeResidentAdam(const StructuredParameterSchema& schema,
      const StructuredParameterSnapshot& parameters, const ResidentAdamConsumer& consume, double& norm);

  /// Commit only after successful atomic provider publication.
  void acceptResidentAdam() noexcept;

  using ResidentDeviceAdamConsumer = std::function<double(
      const std::shared_ptr<psiformer::device::PreparedDeviceAdam>&, double, double, double, double, double)>;
  /// Explicit initial/restart moment upload, before the warm DEVICE-only window.
  void seedResidentDeviceAdam();
  /// Metadata-only proposal; the callback owns its DEVICE candidate and moment owner.
  void proposeResidentDeviceAdam(const StructuredParameterSchema& schema, std::size_t source_version,
                                const ResidentDeviceAdamConsumer& consume, double& norm);
  /// Numerical moments/model were committed together; only scalar bookkeeping remains.
  void acceptResidentDeviceAdam(std::size_t published_version) noexcept;

  FirstOrderOptimizer(const StructuredParameterSchema& schema,
                      FirstOrderOptimizerOptions options);

  /// Destroy optional device scratch where its implementation is complete.
  ~FirstOrderOptimizer() override;

  /// Admit optional real-Adam scratch once; zero preserves the CPU recurrence.
  void prepareDeviceAdam(std::size_t device_budget_bytes);

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

  /// Return host mirrors plus owned device recurrence bytes, counted once.
  std::size_t retainedBytes() const noexcept;

  /// Expose the committed first moment for deterministic validation.
  const std::vector<double>& firstMoment() const;

  /// Expose the committed second moment for deterministic validation.
  const std::vector<double>& secondMoment() const;

  /// Expose the independent AMSGrad running maximum of the second-moment EMA.
  const std::vector<double>& maximumSecondMoment() const noexcept { return maximum_second_moment_; }

  /// Report whether one candidate currently awaits acceptance or rejection.
  bool hasLiveProposal() const noexcept { return proposal_live_; }

private:
  /// Shared snapshot/schema/counter admission without any fabricated gradient.
  void validateSnapshot(const StructuredParameterSchema& schema,
                        const StructuredParameterSnapshot& parameters) const;

  /// Compact per-block update metadata avoids per-parameter configuration storage.
  struct BlockUpdate
  {
    std::size_t offset    = 0;
    std::size_t count     = 0;
    bool trainable        = false;
    double learning_rate  = 0.0;
  };

  class PreparedRestoreAction;

  /// Refresh checked host mirrors before inspection/fingerprint/payload, never commit.
  void synchronizeMoments() const;

  void validateInputs(const StructuredParameterSchema& schema,
                      const StructuredParameterSnapshot& parameters,
                      ParameterGradientView objective) const;
  void commitRecurrence(ParameterGradientView objective) noexcept;
  void validateRestoredState(std::uint64_t update_count,
                             const std::vector<double>& first_moment,
                             const std::vector<double>& second_moment,
                             const std::vector<double>& maximum_second_moment) const;
  std::string stateFingerprint(std::uint64_t update_count,
                               const std::vector<double>& first_moment,
                               const std::vector<double>& second_moment,
                               const std::vector<double>& maximum_second_moment) const;

  FirstOrderOptimizerOptions options_;
  std::string provider_id_;
  std::string schema_fingerprint_;
  std::string configuration_fingerprint_;
  std::size_t parameter_count_ = 0;
  std::vector<BlockUpdate> blocks_;
  mutable std::vector<double> first_moment_;
  mutable std::vector<double> second_moment_;
  std::vector<double> maximum_second_moment_;
  std::uint64_t accepted_update_count_ = 0;
  bool proposal_live_ = false;
  bool resident_device_proposal_ = false;
  std::size_t resident_source_version_ = 0;
  mutable bool moment_mirrors_stale_ = false;
#ifdef ENABLE_CUDA
  std::shared_ptr<psiformer::device::PreparedDeviceAdam> device_adam_;
#endif
};

} // namespace qmcplusplus::wftrain

#endif
