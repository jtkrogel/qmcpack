//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file OrbitalPretrainingAccumulator.h
 * @brief O(parameter-count) streaming statistics for orbital pretraining.
 */

#ifndef QMCPLUSPLUS_ORBITAL_PRETRAINING_ACCUMULATOR_H
#define QMCPLUSPLUS_ORBITAL_PRETRAINING_ACCUMULATOR_H

#include "QMCDrivers/WFTrain/ParameterGradient.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qmcplusplus::wftrain
{

class DistributedParameterReduction;

/** Final mean orbital loss and its canonical mean parameter gradient. */
struct OrbitalPretrainingResult
{
  std::string schema_fingerprint;
  std::size_t parameter_version = 0;
  ReductionDomain reduction_domain = ReductionDomain::CROWD_LOCAL;
  std::uint64_t target_fingerprint = 0;
  std::uint64_t loss_fingerprint = 0;
  std::size_t sample_count = 0;
  DerivativeReal mean_loss = 0.0;
  std::vector<DerivativeReal> gradient;

  /// Return the objective-neutral portion consumed by an update rule.
  ParameterGradientView parameterGradient() const noexcept
  {
    return {schema_fingerprint, parameter_version, reduction_domain,
            {gradient.data(), gradient.size()}};
  }
};

/** Accumulate per-sample orbital-MSE values and gradients without retaining rows. */
class OrbitalPretrainingAccumulator
{
public:
  OrbitalPretrainingAccumulator(const StructuredParameterSchema& schema,
                                std::size_t parameter_version,
                                std::uint64_t target_fingerprint,
                                std::uint64_t loss_fingerprint);

  /// Add one complete sample contribution directly to raw scalar/vector sums.
  void addSample(DerivativeReal loss,
                 std::size_t parameter_version,
                 DerivativeArrayView<const DerivativeReal> gradient);

  /// Merge one compatible crowd-local partial contribution in caller order.
  void merge(const OrbitalPretrainingAccumulator& other);

  /// Promote one nonempty local contribution when no communicator is present.
  void completeSingleParticipantReduction();

  /// Normalize the globally summed objective exactly once.
  OrbitalPretrainingResult finalize();

  /// Return retained numeric capacity, excluding a separately returned result.
  std::size_t retainedBytes() const noexcept;

  /// Hash the retained vector address and capacity for warmed-memory checks.
  std::size_t storageFingerprint() const noexcept;

  /// Return the raw sample count (which may be zero before global reduction).
  std::size_t sampleCount() const noexcept { return sample_count_; }

  /// Return the number of canonical parameters represented by the raw vector.
  std::size_t parameterCount() const noexcept { return gradient_sum_.size(); }

  /// Return the current reduction domain.
  ReductionDomain reductionDomain() const noexcept { return reduction_domain_; }

private:
  friend class DistributedParameterReduction;

  void requireCompatible(const OrbitalPretrainingAccumulator& other) const;
  void poison() noexcept;

  std::string provider_id_;
  std::string schema_fingerprint_;
  std::size_t parameter_version_ = 0;
  std::uint64_t target_fingerprint_ = 0;
  std::uint64_t loss_fingerprint_ = 0;
  std::size_t sample_count_ = 0;
  DerivativeReal loss_sum_ = 0.0;
  std::vector<DerivativeReal> gradient_sum_;
  ReductionDomain reduction_domain_ = ReductionDomain::CROWD_LOCAL;
  bool finalized_ = false;
  bool poisoned_ = false;
};

} // namespace qmcplusplus::wftrain

#endif
