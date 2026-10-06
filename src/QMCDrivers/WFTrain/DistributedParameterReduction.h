//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file DistributedParameterReduction.h
 * @brief Synchronous replicated reduction of bounded training statistics.
 *
 * The context deliberately implements one conservative collective schedule. It owns
 * no parameter-sized staging buffer: the three raw accumulator vectors are reduced
 * in place, in canonical chunks, after all ranks agree that production completed.
 */

#ifndef QMCPLUSPLUS_DISTRIBUTED_PARAMETER_REDUCTION_H
#define QMCPLUSPLUS_DISTRIBUTED_PARAMETER_REDUCTION_H

#include "QMCDrivers/WFTrain/EnergyGradientAccumulator.h"
#include "QMCDrivers/WFTrain/OrbitalPretrainingAccumulator.h"

#include <cstddef>
#include <exception>

class Communicate;

namespace qmcplusplus::wftrain
{

/** One weighted sample scalar reduced before a matrix-free parameter action.
 *
 * Local producers submit RANK_LOCAL raw moments.  A successful distributed
 * reduction returns the exact global sample count and replicated GLOBAL sums.
 */
struct DistributedWeightedSampleMoments
{
  std::size_t sample_count = 0;
  DerivativeReal weight_sum = 0.0;
  DerivativeValue weighted_value_sum{};
  ReductionDomain reduction_domain = ReductionDomain::RANK_LOCAL;
};

/// Configure the only tunable property of the synchronous replicated reduction.
struct DistributedReductionPolicy
{
  std::size_t maximum_chunk_size = 65536;
};

/** Coordinate fixed-record consensus and in-place replicated parameter reductions.
 *
 * A default-constructed context is a one-participant local context. Binding a
 * communicator enables the same protocol across its ranks without changing result
 * ownership: every successful rank receives the complete globally summed vectors.
 */
class DistributedParameterReduction
{
public:
  /// Construct a single-participant reduction context.
  DistributedParameterReduction(DistributedReductionPolicy policy = {});

  /// Bind the reduction context to an existing communicator with nonowning lifetime.
  DistributedParameterReduction(Communicate& communicator,
                                DistributedReductionPolicy policy = {});

  /// Agree on immutable iteration metadata before derivative production begins.
  void preflight(const StructuredParameterSchema& schema,
                 const StructuredParameterSnapshot* parameters,
                 EnergyGradientEstimator estimator,
                 std::exception_ptr local_failure = {}) const;

  /** Reduce one complete local accumulator or report a uniform producer failure.
   *
   * Raw scalar moments and the three P-vectors are summed before the accumulator is
   * promoted to GLOBAL. Zero-sample ranks execute the identical collective schedule.
   */
  void reduce(EnergyGradientAccumulator& accumulator,
              std::exception_ptr local_failure = {}) const;

  /// Agree on model and target identity before orbital-MSE production begins.
  void preflightOrbital(const StructuredParameterSchema& schema,
                        const StructuredParameterSnapshot* parameters,
                        std::uint64_t target_fingerprint,
                        std::uint64_t loss_fingerprint,
                        std::exception_ptr local_failure = {}) const;

  /// Reduce one complete orbital objective, allowing zero-sample local ranks.
  void reduce(OrbitalPretrainingAccumulator& accumulator,
              std::exception_ptr local_failure = {}) const;

  /** Sum one rank-local parameter-vector contribution on every participant.
   *
   * The input is replaced in place by a replicated global vector. All ranks execute
   * one fixed metadata/failure consensus before entering the chunked data collectives.
   */
  ReductionDomain reduceParameterVector(
      const StructuredParameterSchema& schema,
      std::size_t parameter_version,
      DerivativeArrayView<DerivativeValue> values,
      std::exception_ptr local_failure = {}) const;

  /** Reduce rank-local weighted scalar moments for a matrix-free sample action.
   *
   * Different ranks may contribute different (including zero) sample counts.  The
   * fixed-record consensus precedes both scalar collectives, so a rank-local JVP
   * failure cannot strand peers in an unmatched all-reduce.
   */
  DistributedWeightedSampleMoments reduceWeightedSampleMoments(
      const StructuredParameterSchema& schema,
      std::size_t parameter_version,
      const DistributedWeightedSampleMoments& local_moments,
      std::exception_ptr local_failure = {}) const;

  /// Verify that every rank prepared the same complete update before publication.
  void validateCandidate(const StructuredParameterSchema& schema,
                         const StructuredParameterSnapshot& parameters,
                         const StructuredParameterSnapshot* candidate,
                         std::exception_ptr local_failure = {}) const;

  /// Agree on publication success and return the common committed version.
  std::size_t completePublication(std::size_t local_version,
                                  std::exception_ptr local_failure = {}) const;

  /// Return the number of participants in the bound context.
  std::size_t participantCount() const noexcept;

  /// Return the configured logical-parameter chunk limit.
  std::size_t maximumChunkSize() const noexcept { return policy_.maximum_chunk_size; }

private:
  Communicate* communicator_ = nullptr;
  DistributedReductionPolicy policy_;
};

} // namespace qmcplusplus::wftrain

#endif
