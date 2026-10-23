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
#include "QMCDrivers/WFTrain/TrainingNumerics.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace qmcplusplus::wftrain
{
namespace
{

/// Report whether both parts of one contraction scalar are finite.
bool isFinite(DerivativeValue value) noexcept
{
  return isFiniteTrainingReal(value.real()) &&
      isFiniteTrainingReal(value.imag());
}

} // namespace

PreparedLocalEnergyVJPCapture::PreparedLocalEnergyVJPCapture(
    const StructuredParameterSchema& schema,
    std::size_t parameter_version,
    std::uint32_t local_energy_term_mask)
    : provider_id_(schema.providerId()),
      schema_fingerprint_(schema.fingerprint()),
      parameter_version_(parameter_version),
      local_energy_term_mask_(local_energy_term_mask),
      staging_(schema.parameterCount())
{
  const std::uint32_t known_terms =
      localEnergyTermBit(LocalEnergyTerm::KINETIC) |
      localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
  if (local_energy_term_mask_ == 0 ||
      (local_energy_term_mask_ & ~known_terms) != 0)
    throw std::invalid_argument(
        "Prepared local-energy capture requires named Hamiltonian-term coverage");
}

void PreparedLocalEnergyVJPCapture::onBegin(
    const DerivativeStreamDescriptor& descriptor,
    const ParameterChunkPlan&,
    DerivativeArrayView<const VJPCoefficientChannel> channels)
{
  if (released_)
    throw std::logic_error("Prepared local-energy capture was already released");
  if (descriptor.provider_id != provider_id_ ||
      descriptor.schema_fingerprint != schema_fingerprint_ ||
      descriptor.parameter_version != parameter_version_)
    throw std::invalid_argument(
        "Prepared local-energy capture does not match the parameter snapshot");
  if (descriptor.adjoint != DerivativeAdjoint::TRANSPOSE || channels.size() != 1 ||
      channels[0].product != DerivativeProduct::LOCAL_ENERGY_VJP ||
      channels[0].local_energy_term_mask != local_energy_term_mask_)
    throw std::invalid_argument(
        "Prepared local-energy capture requires one exact transpose VJP channel");
  std::fill(staging_.begin(), staging_.end(), DerivativeValue{});
  batch_ordinal_       = descriptor.batch_ordinal;
  sample_offset_       = descriptor.sample_offset;
  sample_count_        = descriptor.sample_count;
  transaction_complete_ = false;
}

void PreparedLocalEnergyVJPCapture::consume(
    std::size_t channel_ordinal,
    const ParameterChunkConstView& chunk)
{
  if (channel_ordinal != 0)
    throw std::invalid_argument(
        "Prepared local-energy capture received an unexpected channel");
  const auto values = chunk.values();
  for (DerivativeValue value : values)
    if (!isFinite(value))
      throw std::invalid_argument(
          "Prepared local-energy capture received a non-finite contraction");
  const std::size_t begin = chunk.descriptor().parameter_offset;
  std::copy(values.begin(), values.end(), staging_.begin() + begin);
}

void PreparedLocalEnergyVJPCapture::onEnd()
{
  transaction_complete_ = true;
}

void PreparedLocalEnergyVJPCapture::onAbort() noexcept
{
  std::fill(staging_.begin(), staging_.end(), DerivativeValue{});
  transaction_complete_ = false;
}

void PreparedLocalEnergyVJPCapture::onReset() noexcept
{
  onAbort();
}

PreparedAdditiveLocalEnergyVJP PreparedLocalEnergyVJPCapture::release()
{
  if (state() != DerivativeSinkState::COMPLETE || !transaction_complete_)
    throw std::logic_error(
        "Prepared local-energy capture is not complete");
  if (released_)
    throw std::logic_error(
        "Prepared local-energy capture can be released only once");
  released_ = true;
  return {provider_id_, schema_fingerprint_, parameter_version_, batch_ordinal_,
          sample_offset_, sample_count_, local_energy_term_mask_,
          std::move(staging_)};
}

std::size_t PreparedLocalEnergyVJPCapture::storageFingerprint() const noexcept
{
  std::size_t hash = 1469598103934665603ULL;
  hash ^= reinterpret_cast<std::uintptr_t>(staging_.data());
  hash *= 1099511628211ULL;
  hash ^= staging_.capacity();
  return hash * 1099511628211ULL;
}

PreparedEnergyGradientProducer::PreparedEnergyGradientProducer(
    std::vector<PreparedEnergyGradientBatch> batches,
    std::optional<EnergyClippingTransform> clipping, bool score_only)
    : batches_(std::move(batches)), clipping_(std::move(clipping)), score_only_(score_only)
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
        (!score_only_ && !offered.supports(DerivativeProduct::LOCAL_ENERGY_VJP)))
      throw std::invalid_argument(
          "Prepared energy-gradient batch lacks a required VJP product");
    if (offered.parameter_scalar_domain != ParameterScalarDomain::REAL64)
      throw std::invalid_argument(
          "Prepared energy-gradient production currently requires real parameters");
    if (!offered.coversLocalEnergyTerms(batch.local_energy_term_mask) ||
        (!score_only_ && batch.local_energy_term_mask == 0))
      throw std::invalid_argument(
          "Prepared energy-gradient batch has incomplete local-energy coverage");
    if (batch.weights.size() != derivative.sampleCount() ||
        batch.local_energies.size() != derivative.sampleCount())
      throw std::invalid_argument(
          "Prepared energy-gradient sample arrays do not match their derivative owner");

    std::uint32_t complete_term_mask = batch.local_energy_term_mask;
    if (batch.additive_local_energy_vjp)
    {
      if (score_only_)
        throw std::invalid_argument("Score-only prepared production does not consume additive local-energy responses");
      const PreparedAdditiveLocalEnergyVJP& additive =
          *batch.additive_local_energy_vjp;
      if ((batch.local_energy_term_mask & additive.local_energy_term_mask) != 0)
        throw std::invalid_argument(
            "Prepared energy-gradient additive and primary term masks overlap");
      const std::uint32_t known_terms =
          localEnergyTermBit(LocalEnergyTerm::KINETIC) |
          localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
      if (additive.provider_id != derivative.parameterSchema().providerId() ||
          additive.schema_fingerprint != derivative.parameterSchema().fingerprint() ||
          additive.parameter_version != derivative.parameterVersion() ||
          additive.batch_ordinal != derivative.batchOrdinal() ||
          additive.sample_offset != derivative.sampleOffset() ||
          additive.sample_count != derivative.sampleCount() ||
          additive.local_energy_term_mask == 0 ||
          (additive.local_energy_term_mask & ~known_terms) != 0 ||
          additive.weighted_vjp.size() != derivative.parameterSchema().parameterCount())
        throw std::invalid_argument(
            "Prepared energy-gradient additive response has incompatible identity or extent");
      for (DerivativeValue value : additive.weighted_vjp)
        if (!isFinite(value))
          throw std::invalid_argument(
              "Prepared energy-gradient additive response is non-finite");
      complete_term_mask |= additive.local_energy_term_mask;
    }

    if (schema_fingerprint_.empty())
    {
      schema_fingerprint_        = derivative.parameterSchema().fingerprint();
      parameter_version_         = derivative.parameterVersion();
      local_energy_term_mask_    = complete_term_mask;
    }
    else if (schema_fingerprint_ != derivative.parameterSchema().fingerprint() ||
             parameter_version_ != derivative.parameterVersion() ||
             local_energy_term_mask_ != complete_term_mask)
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
  if (!score_only_)
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
  if (score_only_ && !isScoreOnlyEstimator(accumulator.estimator()))
    throw std::invalid_argument("Score-only prepared production requires a score-only accumulator");
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
      if (batch.additive_local_energy_vjp)
        accumulator.addLocalEnergyVJP(
            {batch.additive_local_energy_vjp->weighted_vjp.data(),
             batch.additive_local_energy_vjp->weighted_vjp.size()},
            batch.additive_local_energy_vjp->local_energy_term_mask);
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
      if (batch.additive_local_energy_vjp)
        partial.addLocalEnergyVJP(
            {batch.additive_local_energy_vjp->weighted_vjp.data(),
             batch.additive_local_energy_vjp->weighted_vjp.size()},
            batch.additive_local_energy_vjp->local_energy_term_mask);
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
    const auto storage = batch.derivative_operator->storageDiagnostics();
    diagnostics.derivative_owner_bytes += storage.retained_numeric_bytes;
    // This allocation-free identity scan is safe inside the noexcept diagnostic.
    if (storage.shared_workspace_identity)
      for (const auto& previous : batches_)
      {
        if (&previous == &batch)
          break;
        if (previous.derivative_operator->storageDiagnostics().shared_workspace_identity ==
            storage.shared_workspace_identity)
        {
          diagnostics.derivative_owner_bytes -= storage.shared_workspace_bytes;
          break;
        }
      }
    diagnostics.sample_scalar_bytes +=
        batch.weights.capacity() * sizeof(DerivativeReal) +
        batch.local_energies.capacity() * sizeof(DerivativeValue);
    if (batch.additive_local_energy_vjp)
      diagnostics.sample_scalar_bytes +=
          batch.additive_local_energy_vjp->weighted_vjp.capacity() *
          sizeof(DerivativeValue);
  }
  diagnostics.retained_numeric_bytes = diagnostics.derivative_owner_bytes +
      diagnostics.sample_scalar_bytes;
  return diagnostics;
}

} // namespace qmcplusplus::wftrain
