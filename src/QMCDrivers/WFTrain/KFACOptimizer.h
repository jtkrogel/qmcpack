//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_KFAC_OPTIMIZER_H
#define QMCPLUSPLUS_KFAC_OPTIMIZER_H

#include "QMCDrivers/WFTrain/KFACPreconditioner.h"
#include "QMCDrivers/WFTrain/ParameterUpdateTransaction.h"
#include "QMCDrivers/WFTrain/TrainingCheckpoint.h"
#include <cstdint>
#include <memory>

namespace qmcplusplus::wftrain
{
/// Select whether the schedule scalar enters before or after inverse/constraint arithmetic.
enum class KFACUpdateOrder { POST_INVERSE_RATE, PRE_INVERSE_RATE };

/// Selected fixed-damping recurrence, schedule, inverse cadence and step controls.
struct KFACOptimizerControl
{
  DerivativeReal learning_rate = 1e-3;
  DerivativeReal learning_rate_decay = 0.0;
  bool schedule_float32 = false;
  DerivativeReal ema_decay = 0.0;
  std::size_t inverse_period = 1;
  DerivativeReal norm_constraint = 0.0;
  KFACCurvatureConvention convention = KFACCurvatureConvention::NATIVE_SCORE;
  KFACPreconditionerControl inverse;
  KFACUpdateOrder update_order = KFACUpdateOrder::POST_INVERSE_RATE;
};

/** Persistent KFAC recurrence committed only with a published model update. */
class KFACOptimizer final : public TrainingUpdateRule, public OptimizerCheckpointContributor
{
public:
  /// Bind immutable schema/block semantics and validate the selected controls.
  KFACOptimizer(KFACBlockRegistry registry, KFACOptimizerControl control);

  ~KFACOptimizer() override;

  /// Prepare speculative factor averages and inverse caches without publishing them.
  void prepareStatistics(const KFACFactorAccumulator& statistics);

  /// Apply the prepared inverse, current schedule and optional surrogate constraint.
  StructuredParameterSnapshot propose(const StructuredParameterSchema& schema,
      const StructuredParameterSnapshot& parameters, ParameterGradientView objective) override;

  /// Install prepared recurrence state after successful parameter publication.
  void proposalAccepted(const StructuredParameterSchema&, const StructuredParameterSnapshot&,
                        ParameterGradientView) noexcept override;

  /// Discard the complete speculative recurrence after a failed/rejected update.
  void proposalRejected() noexcept override;

  /// Return accepted optimizer updates, distinct from historical parameter labels.
  std::uint64_t step() const noexcept;

  /// Return physical samples represented by accepted optimizer updates.
  std::uint64_t dataSeen() const noexcept;

  /// Return the pre-update step that last refreshed affine inverse factors.
  std::uint64_t inverseStep() const noexcept;

  /// Evaluate the pre-update inverse learning-rate schedule in its declared dtype.
  DerivativeReal effectiveLearningRate() const noexcept;

  /// Return the last unconstrained damped surrogate, including squared rate.
  DerivativeReal constraintMetric() const noexcept { return last_metric_; }

  /// Return the last scale applied to the preconditioned gradient.
  DerivativeReal constraintScale() const noexcept { return last_scale_; }

  /// Charge committed, speculative and prepared-inverse numeric storage.
  std::size_t retainedNumericBytes() const noexcept;

  /// Identify a complete committed state; speculative checkpoints are rejected.
  CheckpointContributorMetadata checkpointMetadata() const override;

  /// Serialize exact factor/inverse state and accepted progress counters.
  void writeCheckpointPayload(hdf_archive&) const override;

  /// Validate an owning restore before the outer checkpoint publishes any state.
  std::unique_ptr<OptimizerCheckpointContributor::PreparedRestore> prepareCheckpointRestore(
      hdf_archive&, const CheckpointContributorMetadata&) override;

private:
  struct State;

  struct Pending;

  class PreparedRestoreAction;

  /// Fingerprint counters and exact numeric payload independently of object addresses.
  std::string stateFingerprint(const State&) const;
  KFACBlockRegistry registry_;
  KFACOptimizerControl control_;
  std::string configuration_fingerprint_;
  std::unique_ptr<State> state_;
  std::unique_ptr<Pending> pending_;
  DerivativeReal last_metric_ = 0.0;
  DerivativeReal last_scale_ = 1.0;
};
}
#endif
