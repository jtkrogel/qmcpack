//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PreparedEnergyGradientProducer.h
 * @brief Version-bound live-batch owner for streamed energy-gradient production.
 */

#ifndef QMCPLUSPLUS_PREPARED_ENERGY_GRADIENT_PRODUCER_H
#define QMCPLUSPLUS_PREPARED_ENERGY_GRADIENT_PRODUCER_H

#include "QMCDrivers/WFTrain/HighParameterTraining.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace qmcplusplus::wftrain
{

/** Own one isolated, sample-weighted local-energy response in canonical order. */
struct PreparedAdditiveLocalEnergyVJP
{
  std::string provider_id;
  std::string schema_fingerprint;
  std::size_t parameter_version = 0;
  std::size_t batch_ordinal     = 0;
  std::size_t sample_offset     = 0;
  std::size_t sample_count      = 0;
  std::uint32_t local_energy_term_mask = 0;
  std::vector<DerivativeValue> weighted_vjp;
};

/** Capture exactly one bounded local-energy channel without publishing partial data. */
class PreparedLocalEnergyVJPCapture final : public ParameterReductionSink
{
public:
  /// Bind the capture to one immutable parameter schema and model version.
  PreparedLocalEnergyVJPCapture(const StructuredParameterSchema& schema,
                                std::size_t parameter_version,
                                std::uint32_t local_energy_term_mask);

  /// Move the completed response into a prepared batch exactly once.
  PreparedAdditiveLocalEnergyVJP release();

  /// Hash fixed response storage and capacity for warmed-call allocation checks.
  std::size_t storageFingerprint() const noexcept;

  /// Return exact retained numeric bytes.
  std::size_t retainedBytes() const noexcept
  {
    return staging_.capacity() * sizeof(DerivativeValue);
  }

protected:
  void onBegin(const DerivativeStreamDescriptor& descriptor,
               const ParameterChunkPlan& plan,
               DerivativeArrayView<const VJPCoefficientChannel> channels) override;
  void consume(std::size_t channel_ordinal,
               const ParameterChunkConstView& chunk) override;
  void onEnd() override;
  void onAbort() noexcept override;
  void onReset() noexcept override;

private:
  std::string provider_id_;
  std::string schema_fingerprint_;
  std::size_t parameter_version_ = 0;
  std::uint32_t local_energy_term_mask_ = 0;
  std::size_t batch_ordinal_ = 0;
  std::size_t sample_offset_ = 0;
  std::size_t sample_count_  = 0;
  std::vector<DerivativeValue> staging_;
  bool transaction_complete_ = false;
  bool released_             = false;
};

/** Own one crowd's immutable samples and derivative operator for an iteration.
 *
 * Local energies and weights use the exact sample ordering copied by the operator.
 * The owner is move-only so a prepared parameter version cannot be duplicated
 * accidentally across update boundaries.
 */
struct PreparedEnergyGradientBatch
{
  std::unique_ptr<StreamingDerivativeOperator> derivative_operator;
  std::vector<DerivativeReal> weights;
  std::vector<DerivativeValue> local_energies;
  /// Terms produced by derivative_operator; additive coverage must be disjoint.
  std::uint32_t local_energy_term_mask = 0;
  std::optional<PreparedAdditiveLocalEnergyVJP> additive_local_energy_vjp;
};

/// Exact numeric storage retained by one prepared live-batch producer.
struct PreparedEnergyGradientStorageDiagnostics
{
  std::size_t batch_count = 0;
  std::size_t sample_count = 0;
  std::size_t derivative_owner_bytes = 0;
  std::size_t sample_scalar_bytes = 0;
  std::size_t retained_numeric_bytes = 0;
};

/** Merge deterministic crowd-local streamed contributions into one rank objective.
 *
 * The producer owns no sampler and performs no parameter publication.  A production
 * driver prepares it only after all live VGL, local-energy, and stochastic-grid state
 * is fixed for the current model version.
 */
class PreparedEnergyGradientProducer final : public GradientProducer
{
public:
  /// Explicit score mode needs no dEL owner and cannot be consumed as a complete-response producer.
  PreparedEnergyGradientProducer(
      std::vector<PreparedEnergyGradientBatch> batches,
      std::optional<EnergyClippingTransform> clipping = std::nullopt,
      bool score_only = false);

  TrainingCapabilities capabilities() const noexcept override;

  void accumulate(const StructuredParameterSnapshot& parameters,
                  EnergyGradientAccumulator& accumulator) override;

  /// Return exact retained numeric storage, including every crowd derivative owner.
  PreparedEnergyGradientStorageDiagnostics storageDiagnostics() const noexcept;

  /// Expose immutable prepared batches for a same-version matrix-free optimizer.
  const std::vector<PreparedEnergyGradientBatch>& batches() const noexcept
  {
    return batches_;
  }

private:
  std::vector<PreparedEnergyGradientBatch> batches_;
  std::optional<EnergyClippingTransform> clipping_;
  TrainingCapabilities capabilities_;
  std::string schema_fingerprint_;
  std::size_t parameter_version_ = 0;
  std::uint32_t local_energy_term_mask_ = 0;
  bool consumed_ = false;
  bool score_only_ = false;
};

} // namespace qmcplusplus::wftrain

#endif
