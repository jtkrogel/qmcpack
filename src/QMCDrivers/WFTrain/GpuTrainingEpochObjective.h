//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_WFTRAIN_GPU_TRAINING_EPOCH_OBJECTIVE_H
#define QMCPLUSPLUS_WFTRAIN_GPU_TRAINING_EPOCH_OBJECTIVE_H

#include "HighParameterTraining.h"
#include <memory>
#include <stdexcept>
#include <string>

namespace qmcplusplus::wftrain
{
/** One admitted native mixed/FP64 epoch's rank-local objective, before global publication.
 * The native driver times the actual production below and resolves collective
 * retry readiness before passing this completed producer to the unchanged trainer.
 * No derivative pass or MC traversal occurs during the final merge.
 */
class GpuTrainingEpochObjective final : public GradientProducer
{
public:
  /// Execute the one actual derivative pass into disposable private O(P) raw sums.
  GpuTrainingEpochObjective(const StructuredParameterSchema& schema,
      const StructuredParameterSnapshot& parameters, GradientProducer& source,
      EnergyGradientEstimator estimator)
      : capabilities_(source.capabilities()), fingerprint_(schema.fingerprint()), version_(parameters.version),
        objective_(std::make_unique<EnergyGradientAccumulator>(schema, parameters.version, estimator))
  {
    if (parameters.schema_fingerprint != fingerprint_ || parameters.values.size() != schema.parameterCount())
      throw std::invalid_argument("Native epoch objective has incompatible parameter identity");
    source.accumulate(parameters, *objective_);
    if (!objective_->hasCompleteContribution() || objective_->reductionDomain() != ReductionDomain::CROWD_LOCAL)
      throw std::logic_error("Native epoch objective requires a complete unreduced rank-local contribution");
  }

  /// Preserve the admitted producer's objective requirements for ordinary trainer preflight.
  TrainingCapabilities capabilities() const noexcept override { return capabilities_; }

  /// Transfer complete raw sums once; this method never evaluates wavefunctions or derivatives.
  void accumulate(const StructuredParameterSnapshot& parameters, EnergyGradientAccumulator& accumulator) override
  {
    if (!objective_)
      throw std::logic_error("Native epoch objective can be consumed only once");
    if (parameters.version != version_ || parameters.schema_fingerprint != fingerprint_)
      throw std::invalid_argument("Native epoch objective is stale or belongs to another parameter schema");
    accumulator.merge(*objective_);
    objective_.reset();
  }

  /// Charge private raw sums alongside the trainer's destination accumulator at their simultaneous peak.
  std::size_t retainedBytes() const noexcept { return objective_ ? objective_->retainedBytes() : 0; }

private:
  const TrainingCapabilities capabilities_;
  const std::string fingerprint_;
  const std::size_t version_;
  std::unique_ptr<EnergyGradientAccumulator> objective_;
};
} // namespace qmcplusplus::wftrain
#endif
