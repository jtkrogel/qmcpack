//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file EnergyGradientAccumulator.h
 * @brief O(parameter-count) streaming statistics for an energy objective.
 */

#ifndef QMCPLUSPLUS_ENERGY_GRADIENT_ACCUMULATOR_H
#define QMCPLUSPLUS_ENERGY_GRADIENT_ACCUMULATOR_H

#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qmcplusplus::wftrain
{

class DistributedParameterReduction;

/// Select one explicitly named finite-sample estimator of the energy gradient.
enum class EnergyGradientEstimator
{
  SYMMETRIZED_HAMILTONIAN,
  PATHWISE_LOCAL_ENERGY
};

/// Final scalar moments and one real, parameter-shaped energy gradient.
struct EnergyGradientResult
{
  std::string schema_fingerprint;
  std::size_t parameter_version = 0;
  ReductionDomain reduction_domain = ReductionDomain::CROWD_LOCAL;
  EnergyGradientEstimator estimator = EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN;
  std::uint32_t local_energy_term_mask = 0;
  std::size_t sample_count           = 0;
  DerivativeReal weight_sum          = 0.0;
  DerivativeValue mean_energy        = 0.0;
  DerivativeReal energy_variance     = 0.0;
  std::vector<DerivativeReal> gradient;
};

/** Accumulate checked VJP chunks and scalar moments without retaining sample rows.
 *
 * One transaction consumes three named channels: weighted score, energy-weighted
 * score, and weighted local-energy response. Retained storage is three O(P)
 * contraction vectors regardless of the number of samples.
 */
class EnergyGradientAccumulator final : public ParameterReductionSink
{
public:
  /// Stable channel names used by the energy-objective composition helper.
  static constexpr const char* WEIGHTED_SCORE_CHANNEL        = "energy/weighted_score";
  static constexpr const char* ENERGY_WEIGHTED_SCORE_CHANNEL = "energy/energy_weighted_score";
  static constexpr const char* WEIGHTED_LOCAL_ENERGY_CHANNEL = "energy/weighted_local_energy";

  EnergyGradientAccumulator(
      const StructuredParameterSchema& schema,
      std::size_t parameter_version,
      EnergyGradientEstimator estimator = EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN,
      DerivativeAdjoint adjoint = DerivativeAdjoint::TRANSPOSE);

  /// Add scalar raw sums for the exact sample batch consumed by the VJP transaction.
  void addScalarSums(std::size_t sample_count,
                     DerivativeReal weight_sum,
                     DerivativeValue weighted_energy_sum,
                     DerivativeReal weighted_energy_norm_sum);

  /// Merge one complete compatible partial result in deterministic caller order.
  void merge(const EnergyGradientAccumulator& other);

  /// Mark a local result global when it is the only reduction participant.
  void completeSingleParticipantReduction();

  /// Normalize raw sums exactly once and construct the selected estimator.
  EnergyGradientResult finalize();

  /// Return retained numeric capacity, excluding a separately returned result.
  std::size_t retainedBytes() const noexcept;

  /// Return the number of scalar parameters represented by each contraction vector.
  std::size_t parameterCount() const noexcept { return weighted_score_sum_.size(); }

  /// Return the selected finite-sample estimator.
  EnergyGradientEstimator estimator() const noexcept { return estimator_; }

  /// Return the exact Hamiltonian-term coverage of the local-energy VJP.
  std::uint32_t localEnergyTermMask() const noexcept { return local_energy_term_mask_; }

  /// Report whether complete derivative and scalar contributions are available.
  bool hasCompleteContribution() const noexcept
  {
    return derivative_complete_ && scalar_sums_complete_;
  }

  /// Return the reduction domain currently attached to the raw sums.
  ReductionDomain reductionDomain() const noexcept { return reduction_domain_; }

protected:
  /// Validate channel identities and bind this accumulator to one checked stream.
  void onBegin(const DerivativeStreamDescriptor& descriptor,
               const ParameterChunkPlan& plan,
               DerivativeArrayView<const VJPCoefficientChannel> channels) override;

  /// Add one already validated canonical channel/chunk pair.
  void consume(std::size_t channel_ordinal, const ParameterChunkConstView& chunk) override;

  /// Make a completely delivered VJP transaction visible to scalar accumulation.
  void onEnd() override;

  /// Erase partial raw sums after a producer or validation failure.
  void onAbort() noexcept override;

  /// Reinitialize all objective state for an explicit retry.
  void onReset() noexcept override;

private:
  friend class DistributedParameterReduction;

  enum class ChannelKind
  {
    WEIGHTED_SCORE,
    ENERGY_WEIGHTED_SCORE,
    WEIGHTED_LOCAL_ENERGY
  };

  void clearRawSums() noexcept;
  void requireUsableContribution() const;

  std::string provider_id_;
  std::string schema_fingerprint_;
  std::size_t parameter_version_ = 0;
  EnergyGradientEstimator estimator_;
  DerivativeAdjoint adjoint_;
  std::size_t sample_count_ = 0;
  DerivativeReal weight_sum_ = 0.0;
  DerivativeValue weighted_energy_sum_ = 0.0;
  DerivativeReal weighted_energy_norm_sum_ = 0.0;
  std::vector<DerivativeValue> weighted_score_sum_;
  std::vector<DerivativeValue> weighted_energy_score_sum_;
  std::vector<DerivativeValue> weighted_energy_derivative_sum_;
  std::vector<ChannelKind> channel_kinds_;
  std::uint32_t local_energy_term_mask_ = 0;
  std::size_t stream_sample_count_ = 0;
  ReductionDomain reduction_domain_ = ReductionDomain::CROWD_LOCAL;
  bool derivative_complete_ = false;
  bool scalar_sums_complete_ = false;
  bool finalized_ = false;
};

/** Compose the three checked VJP channels required by one energy-gradient batch.
 *
 * Coefficient scratch is O(samples); the operator can emit only bounded parameter
 * chunks through the sink. No sample-by-parameter derivative representation exists.
 */
void accumulateEnergyGradientBatch(
    const StreamingDerivativeOperator& derivative_operator,
    DerivativeArrayView<const DerivativeReal> weights,
    DerivativeArrayView<const DerivativeValue> local_energies,
    std::uint32_t local_energy_term_mask,
    EnergyGradientAccumulator& accumulator,
    DerivativeAdjoint adjoint = DerivativeAdjoint::TRANSPOSE);

} // namespace qmcplusplus::wftrain

#endif
