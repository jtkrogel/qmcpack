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
#include <memory>
#include <optional>
#include <vector>

namespace qmcplusplus::wftrain
{

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
  std::uint32_t local_energy_term_mask = 0;
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
  PreparedEnergyGradientProducer(
      std::vector<PreparedEnergyGradientBatch> batches,
      std::optional<EnergyClippingTransform> clipping = std::nullopt);

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
};

} // namespace qmcplusplus::wftrain

#endif
