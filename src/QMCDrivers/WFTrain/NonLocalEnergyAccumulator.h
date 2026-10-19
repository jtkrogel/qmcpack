//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_NONLOCALENERGYACCUMULATOR_H
#define QMCPLUSPLUS_NONLOCALENERGYACCUMULATOR_H

#include "QMCDrivers/WFTrain/TrainingNumerics.h"
#include "QMCWaveFunctions/Optimization/NonLocalECPDerivative.h"
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qmcplusplus::wftrain
{

/// Reduce accepted ordinary-locality tiles to complex energies without parameter products or replay.
class NonLocalEnergyAccumulator final : public NonLocalECPDerivativeConsumer
{
public:
  /// Admit exactly one complex scalar per sample before allocating any numerical storage.
  static std::size_t requiredBytes(std::size_t samples)
  {
    if (!samples || samples > std::numeric_limits<std::size_t>::max() / sizeof(DerivativeValue))
      throw std::length_error("Nonlocal energy sample extent is empty or overflows");
    return samples * sizeof(DerivativeValue);
  }

  /// Pin an ordered sample batch; the finite cap excludes caller-owned transient native tiles.
  NonLocalEnergyAccumulator(const StructuredParameterSchema& schema, std::size_t version,
                            std::size_t batch, std::size_t offset, std::size_t samples,
                            std::size_t electrons, std::size_t cap)
      : provider_(schema.providerId()), schema_(schema.fingerprint()), version_(version),
        batch_(batch), offset_(offset), electrons_(electrons)
  {
    const auto bytes = requiredBytes(samples);
    if (!electrons || cap == std::numeric_limits<std::size_t>::max() || bytes > cap)
      throw std::length_error("Nonlocal energy accumulator exceeds finite host admission");
    sums_.resize(samples);
    if (sums_.capacity() > cap / sizeof(DerivativeValue))
      throw std::length_error("Nonlocal energy actual capacity exceeds host admission");
  }

  /// Reject mismatched identity, localization or channels before the Hamiltonian draws a grid.
  void preflight(const NonLocalECPDerivativeContext& context,
                 DerivativeArrayView<const VJPCoefficientChannel> channels,
                 const ParameterReductionSink& sink) const override
  {
    if (state_ != DerivativeSinkState::IDLE)
      throw std::logic_error("Nonlocal energy accumulator is a one-shot transaction");
    if (context.provider_id != provider_ || context.schema_fingerprint != schema_ ||
        context.parameter_version != version_ || context.batch_ordinal != batch_ ||
        context.sample_offset != offset_ || context.sample_count != sums_.size() ||
        context.localization != NonLocalECPLocalization::ORDINARY_LOCALITY ||
        !context.uses_virtual_particles || !context.scalar_relativistic || context.used_dense_derivative_fallback ||
        channels.size() != 1 || sink.state() != DerivativeSinkState::IDLE)
      throw std::invalid_argument("Nonlocal energy requires matching scalar ordinary-locality metadata");
    const auto& channel = channels[0];
    const auto& coefficients = channel.coefficients;
    if (channel.product != DerivativeProduct::LOCAL_ENERGY_VJP ||
        channel.local_energy_term_mask != localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP) ||
        coefficients.provider_id != provider_ || coefficients.schema_fingerprint != schema_ ||
        coefficients.parameter_version != version_ || coefficients.batch_ordinal != batch_ ||
        coefficients.sample_offset != offset_ || coefficients.values.size() != sums_.size())
      throw std::invalid_argument("Nonlocal energy channel identity is incompatible");
    for (const auto value : coefficients.values) requireFinite(value);
  }

  /// Bind the actual accepted grid without activating or writing the derivative sink.
  void begin(const NonLocalECPDerivativeContext& context,
             DerivativeArrayView<const VJPCoefficientChannel> channels, ParameterReductionSink& sink) override
  {
    preflight(context, channels, sink);
    if (!context.grid_fingerprint)
      throw std::invalid_argument("Nonlocal energy requires an accepted grid identity");
    grid_ = context.grid_fingerprint;
    expected_points_ = context.expected_point_count;
    state_ = DerivativeSinkState::ACTIVE;
  }

  /// Contract each complete ratio before the shared native tile buffers are reused.
  void consume(const NonLocalECPDerivativeTile& tile) override
  {
    try
    {
      const auto& batch = tile.virtual_particles;
      if (state_ != DerivativeSinkState::ACTIVE || tile.grid_fingerprint != grid_ ||
          tile.tile_ordinal != tiles_ || batch.walkerCount() != sums_.size() ||
          tile.bare_weights.size() != batch.size() || tile.complete_ratios.size() != batch.size() ||
          batch.size() > expected_points_ - points_)
        throw std::invalid_argument("Nonlocal energy tile identity or extent is invalid");
      for (std::size_t segment = 0; segment < batch.segmentCount(); ++segment)
      {
        const auto slice = batch.slice(segment);
        if (slice.walkerId() < 0 || static_cast<std::size_t>(slice.walkerId()) >= sums_.size() ||
            slice.electronId() < 0 || static_cast<std::size_t>(slice.electronId()) >= electrons_)
          throw std::invalid_argument("Nonlocal energy segment index is invalid");
        for (std::size_t point = 0; point < slice.size(); ++point)
        {
          const auto index = slice.flatOffset() + point;
          const DerivativeValue weight = tile.bare_weights[index];
          const DerivativeValue ratio = tile.complete_ratios[index];
          requireFinite(weight); requireFinite(ratio);
          const auto contribution = weight * ratio;
          requireFinite(contribution);
          auto& sum = sums_[slice.walkerId()];
          requireFinite(sum + contribution);
          sum += contribution;
        }
      }
      points_ += batch.size();
      ++tiles_;
    }
    catch (...) { abort(); throw; }
  }

  /// Expose results only after the accepted traversal has delivered every point exactly once.
  void end() override
  {
    if (state_ != DerivativeSinkState::ACTIVE || points_ != expected_points_)
    {
      abort();
      throw std::logic_error("Nonlocal energy traversal is incomplete");
    }
    state_ = DerivativeSinkState::COMPLETE;
  }

  /// Hide all partial sums on native traversal failure without touching model state or RNG.
  void abort() noexcept override { state_ = DerivativeSinkState::POISONED; }

  /// Energy-only reductions intentionally never retain accepted-grid replay data.
  std::unique_ptr<NonLocalECPJVPReplay> releaseJVPReplay() override
  { throw std::logic_error("Nonlocal energy accumulation does not retain replay"); }

  /// Borrow completed per-sample energies while the one-shot accumulator remains alive.
  DerivativeArrayView<const DerivativeValue> energies() const
  {
    if (state_ != DerivativeSinkState::COMPLETE)
      throw std::logic_error("Nonlocal energy result is not complete");
    return {sums_.data(), sums_.size()};
  }

  /// Identify the exact accepted traversal associated with the completed energies.
  std::uint64_t gridFingerprint() const
  {
    (void)energies();
    return grid_;
  }

  /// Report the fixed O(samples) allocation; no virtual-point or parameter-sized arrays exist.
  StreamingDerivativeStorageDiagnostics storageDiagnostics() const override
  {
    StreamingDerivativeStorageDiagnostics result;
    result.sample_count = sums_.size();
    result.sample_auxiliary_bytes = sums_.capacity() * sizeof(DerivativeValue);
    result.retained_numeric_bytes = result.sample_auxiliary_bytes;
    result.allocation_generation = 1;
    result.storage_fingerprint = reinterpret_cast<std::size_t>(sums_.data());
    return result;
  }

private:
  /// Keep finite checks valid under the optimized build's fast-math flags.
  static void requireFinite(DerivativeValue value)
  {
    if (!isFiniteTrainingReal(value.real()) || !isFiniteTrainingReal(value.imag()))
      throw std::overflow_error("Nonlocal energy arithmetic is nonfinite");
  }

  const std::string provider_, schema_;
  const std::size_t version_, batch_, offset_, electrons_;
  std::vector<DerivativeValue> sums_;
  DerivativeSinkState state_ = DerivativeSinkState::IDLE;
  std::size_t expected_points_ = 0, points_ = 0, tiles_ = 0;
  std::uint64_t grid_ = 0;
};

} // namespace qmcplusplus::wftrain
#endif
