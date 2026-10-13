//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PreparedEnergyGradientProducer.cpp
 * @brief Deterministic composition of prepared crowd energy-gradient streams.
 */

#include "QMCDrivers/WFTrain/PreparedEnergyGradientProducer.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace qmcplusplus::wftrain
{
PreparedEnergyGradientProducer::PreparedEnergyGradientProducer(
    std::vector<PreparedEnergyGradientBatch> batches,
    std::optional<EnergyClippingTransform> clipping)
    : batches_(std::move(batches)), clipping_(std::move(clipping))
{
  if (batches_.empty())
    throw std::invalid_argument(
        "Prepared energy-gradient producer requires at least one batch owner");

  std::set<std::size_t> batch_ordinals;
  std::vector<std::pair<std::size_t, std::size_t>> sample_intervals;
  sample_intervals.reserve(batches_.size());
  bool supports_score_jvp = true;
  for (const PreparedEnergyGradientBatch& batch : batches_)
  {
    if (!batch.derivative_operator)
      throw std::invalid_argument(
          "Prepared energy-gradient batch has no derivative operator");
    const StreamingDerivativeOperator& derivative = *batch.derivative_operator;
    const StreamingDerivativeCapabilities offered = derivative.capabilities();
    if (!offered.supports(DerivativeProduct::SCORE_VJP) ||
        !offered.supports(DerivativeProduct::LOCAL_ENERGY_VJP))
      throw std::invalid_argument(
          "Prepared energy-gradient batch lacks a required VJP product");
    if (offered.parameter_scalar_domain != ParameterScalarDomain::REAL64)
      throw std::invalid_argument(
          "Prepared energy-gradient production currently requires real parameters");
    if (!offered.coversLocalEnergyTerms(batch.local_energy_term_mask) ||
        batch.local_energy_term_mask == 0)
      throw std::invalid_argument(
          "Prepared energy-gradient batch has incomplete local-energy coverage");
    if (batch.weights.size() != derivative.sampleCount() ||
        batch.local_energies.size() != derivative.sampleCount())
      throw std::invalid_argument(
          "Prepared energy-gradient sample arrays do not match their derivative owner");

    if (schema_fingerprint_.empty())
    {
      schema_fingerprint_        = derivative.parameterSchema().fingerprint();
      parameter_version_         = derivative.parameterVersion();
      local_energy_term_mask_    = batch.local_energy_term_mask;
    }
    else if (schema_fingerprint_ != derivative.parameterSchema().fingerprint() ||
             parameter_version_ != derivative.parameterVersion() ||
             local_energy_term_mask_ != batch.local_energy_term_mask)
      throw std::invalid_argument(
          "Prepared energy-gradient batches have inconsistent schema, version, or Hamiltonian coverage");

    if (!batch_ordinals.insert(derivative.batchOrdinal()).second)
      throw std::invalid_argument(
          "Prepared energy-gradient batches have duplicate ordinals");
    if (derivative.sampleOffset() >
        std::numeric_limits<std::size_t>::max() - derivative.sampleCount())
      throw std::overflow_error(
          "Prepared energy-gradient sample interval overflow");
    sample_intervals.emplace_back(
        derivative.sampleOffset(),
        derivative.sampleOffset() + derivative.sampleCount());
    supports_score_jvp = supports_score_jvp &&
        offered.supports(DerivativeProduct::SCORE_JVP);
  }

  std::sort(sample_intervals.begin(), sample_intervals.end());
  for (std::size_t interval = 1; interval < sample_intervals.size(); ++interval)
    if (sample_intervals[interval].first < sample_intervals[interval - 1].second)
      throw std::invalid_argument(
          "Prepared energy-gradient sample intervals overlap");

  capabilities_.add(TrainingCapability::REAL_PARAMETERS);
  capabilities_.add(TrainingCapability::VALUE_EVALUATION);
  capabilities_.add(TrainingCapability::SCORE_VJP);
  capabilities_.add(TrainingCapability::LOCAL_ENERGY_VJP);
  if (supports_score_jvp)
    capabilities_.add(TrainingCapability::SCORE_JVP);
  if ((local_energy_term_mask_ &
       localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)) != 0)
    capabilities_.add(TrainingCapability::NONLOCAL_ECP);
}

TrainingCapabilities PreparedEnergyGradientProducer::capabilities() const noexcept
{
  return capabilities_;
}

void PreparedEnergyGradientProducer::accumulate(
    const StructuredParameterSnapshot& parameters,
    EnergyGradientAccumulator& accumulator)
{
  if (consumed_)
    throw std::logic_error(
        "Prepared energy-gradient producer can be consumed only once");
  if (parameters.schema_fingerprint != schema_fingerprint_ ||
      parameters.version != parameter_version_)
    throw std::invalid_argument(
        "Prepared energy-gradient producer is bound to a different parameter snapshot");

  const EnergyClippingTransform* clipping = clipping_ ? &*clipping_ : nullptr;
  bool first = true;
  for (const PreparedEnergyGradientBatch& batch : batches_)
  {
    if (first)
    {
      accumulateEnergyGradientBatch(
          *batch.derivative_operator,
          {batch.weights.data(), batch.weights.size()},
          {batch.local_energies.data(), batch.local_energies.size()},
          batch.local_energy_term_mask, accumulator, DerivativeAdjoint::TRANSPOSE,
          clipping);
      first = false;
    }
    else
    {
      EnergyGradientAccumulator partial(
          batch.derivative_operator->parameterSchema(), parameter_version_,
          accumulator.estimator());
      accumulateEnergyGradientBatch(
          *batch.derivative_operator,
          {batch.weights.data(), batch.weights.size()},
          {batch.local_energies.data(), batch.local_energies.size()},
          batch.local_energy_term_mask, partial, DerivativeAdjoint::TRANSPOSE,
          clipping);
      accumulator.merge(partial);
    }
  }
  consumed_ = true;
}

PreparedEnergyGradientStorageDiagnostics
PreparedEnergyGradientProducer::storageDiagnostics() const noexcept
{
  PreparedEnergyGradientStorageDiagnostics diagnostics;
  diagnostics.batch_count = batches_.size();
  for (const PreparedEnergyGradientBatch& batch : batches_)
  {
    diagnostics.sample_count += batch.weights.size();
    diagnostics.derivative_owner_bytes +=
        batch.derivative_operator->storageDiagnostics().retained_numeric_bytes;
    diagnostics.sample_scalar_bytes +=
        batch.weights.capacity() * sizeof(DerivativeReal) +
        batch.local_energies.capacity() * sizeof(DerivativeValue);
  }
  diagnostics.retained_numeric_bytes = diagnostics.derivative_owner_bytes +
      diagnostics.sample_scalar_bytes;
  return diagnostics;
}

} // namespace qmcplusplus::wftrain
