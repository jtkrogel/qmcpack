//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file KFACPreconditioner.h
 * @brief Bounded Kronecker-factored statistics and matrix-free inverse application.
 *
 * The classes in this file retain layer-sized factors, never a sample-by-parameter
 * table or a parameter-squared matrix.  Model-specific code supplies a checked block
 * registry and streams nonowning activation/sensitivity rows into the accumulator.
 */

#ifndef QMCPLUSPLUS_KFAC_PRECONDITIONER_H
#define QMCPLUSPLUS_KFAC_PRECONDITIONER_H

#include "QMCDrivers/WFTrain/MatrixFreeOperator.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

class Communicate;

namespace qmcplusplus::wftrain
{

/// Sentinel for an affine KFAC block that has no associated bias tensor.
inline constexpr std::size_t NO_KFAC_BIAS_BLOCK =
    std::numeric_limits<std::size_t>::max();

/** Map one affine layer to canonical schema blocks and factor dimensions. */
struct KFACAffineBlockDescriptor
{
  std::string id;
  std::size_t weight_block = 0;
  std::size_t bias_block   = NO_KFAC_BIAS_BLOCK;
  std::size_t input_width  = 0;
  std::size_t output_width = 0;
};

/** Classify every trainable schema block as factored or scalar fallback. */
class KFACBlockRegistry
{
public:
  KFACBlockRegistry(const StructuredParameterSchema& schema,
                    std::vector<KFACAffineBlockDescriptor> affine_blocks,
                    std::vector<std::size_t> fallback_blocks);

  /// Return the copied schema that defines the canonical tangent space.
  const StructuredParameterSchema& parameterSchema() const noexcept { return schema_; }

  /// Return affine blocks in their deterministic observation order.
  const std::vector<KFACAffineBlockDescriptor>& affineBlocks() const noexcept
  {
    return affine_blocks_;
  }

  /// Return schema-block ordinals handled by the scalar fallback.
  const std::vector<std::size_t>& fallbackBlocks() const noexcept
  {
    return fallback_blocks_;
  }

  /// Return a stable identity for factor snapshots and distributed consensus.
  const std::string& fingerprint() const noexcept { return fingerprint_; }

private:
  StructuredParameterSchema schema_;
  std::vector<KFACAffineBlockDescriptor> affine_blocks_;
  std::vector<std::size_t> fallback_blocks_;
  std::string fingerprint_;
};

/** Nonowning row-major activation and output-sensitivity observation. */
struct KFACLayerObservation
{
  std::size_t affine_block = 0;
  DerivativeArrayView<const DerivativeReal> activations;
  DerivativeArrayView<const DerivativeReal> sensitivities;
  std::size_t row_count = 0;
};

/** Raw symmetric factor sums for one affine block. */
struct KFACFactorStatistics
{
  std::size_t activation_dimension = 0;
  std::size_t sensitivity_dimension = 0;
  DerivativeReal sample_weight_sum = 0.0;
  DerivativeReal row_weight_sum = 0.0;
  std::size_t row_count = 0;
  std::vector<DerivativeReal> activation_outer_sum;
  std::vector<DerivativeReal> sensitivity_outer_sum;
};

/** Stream complete samples into layer-sized Kronecker factors.
 *
 * A sample transaction must observe every affine block exactly once.  Failed or
 * incomplete transactions are discarded without modifying the published sums.
 */
class KFACFactorAccumulator
{
public:
  KFACFactorAccumulator(KFACBlockRegistry registry,
                        std::size_t parameter_version);

  /// Begin one sample with a finite nonnegative statistical weight.
  void beginSample(DerivativeReal sample_weight);

  /// Add one block's repeated dense rows to the current sample transaction.
  void addObservation(const KFACLayerObservation& observation);

  /// Accumulate score-squared diagonal statistics for every fallback interval.
  void addFallbackScores(DerivativeArrayView<const DerivativeReal> score);

  /// Atomically merge the current sample into the published factor sums.
  void endSample();

  /// Discard an incomplete current sample while retaining completed statistics.
  void abortSample() noexcept;

  /// Return the immutable registry that controls observation order and dimensions.
  const KFACBlockRegistry& registry() const noexcept { return registry_; }

  /// Return the parameter version to which these factors are bound.
  std::size_t parameterVersion() const noexcept { return parameter_version_; }

  /// Return the number of complete samples represented by the local statistics.
  std::size_t sampleCount() const noexcept { return sample_count_; }

  /// Return completed factor sums in registry order.
  const std::vector<KFACFactorStatistics>& factors() const noexcept { return factors_; }

  /// Return score-squared sums packed in fallback-block/canonical-entry order.
  DerivativeArrayView<const DerivativeReal> fallbackDiagonalSum() const noexcept
  {
    return {fallback_diagonal_sum_.data(), fallback_diagonal_sum_.size()};
  }

  /// Return the completed statistical weight used by fallback diagonal estimates.
  DerivativeReal fallbackWeightSum() const noexcept { return fallback_weight_sum_; }

  /// Return whether a sample transaction is currently open.
  bool sampleActive() const noexcept { return sample_active_; }

  /// Return exact retained numeric bytes, including transactional scratch.
  std::size_t retainedNumericBytes() const noexcept;

private:
  friend void reduceKFACFactorStatistics(KFACFactorAccumulator&,
                                         Communicate*,
                                         std::size_t);

  KFACBlockRegistry registry_;
  std::size_t parameter_version_ = 0;
  std::size_t sample_count_ = 0;
  DerivativeReal sample_weight_ = 0.0;
  std::vector<KFACFactorStatistics> factors_;
  std::vector<KFACFactorStatistics> pending_;
  std::vector<bool> observed_;
  std::vector<DerivativeReal> fallback_diagonal_sum_;
  std::vector<DerivativeReal> pending_fallback_diagonal_;
  DerivativeReal fallback_weight_sum_ = 0.0;
  bool fallback_observed_ = false;
  bool sample_active_ = false;
  bool globally_reduced_ = false;
};

/** Replicate raw factor sums across ranks using deterministic block/chunk order.
 *
 * A null communicator is the serial identity.  Distributed calls first compare a
 * fixed registry/version record and then all-reduce scalar counts and factor arrays.
 */
void reduceKFACFactorStatistics(KFACFactorAccumulator& accumulator,
                                Communicate* communicator = nullptr,
                                std::size_t maximum_chunk_size = 65536);

/// Configure split damping and explicit behavior for unfactored tensors.
struct KFACPreconditionerControl
{
  DerivativeReal damping = 1.0e-3;
  DerivativeReal fallback_damping = 1.0e-3;
};

/// Exact retained state for a prepared KFAC inverse.
struct KFACStorageDiagnostics
{
  std::size_t parameter_count = 0;
  std::size_t affine_block_count = 0;
  std::size_t factor_elements = 0;
  std::size_t solve_scratch_elements = 0;
  std::size_t retained_numeric_bytes = 0;
};

/** Apply split-damped Kronecker inverses as a checked Task-19 preconditioner. */
class KFACPreconditioner final : public MatrixFreePreconditioner
{
public:
  KFACPreconditioner(const KFACFactorAccumulator& statistics,
                     KFACPreconditionerControl control = {});
  ~KFACPreconditioner() override;

  /// Return retained layer-factor and solve-work storage.
  KFACStorageDiagnostics storageDiagnostics() const noexcept;

  /// Return the immutable registry copied into this prepared inverse.
  const KFACBlockRegistry& registry() const noexcept { return registry_; }

protected:
  void evaluate(const StructuredParameterVectorConstView& residual,
                DerivativeArrayView<DerivativeValue> result) const override;

private:
  struct PreparedBlock;

  KFACBlockRegistry registry_;
  KFACPreconditionerControl control_;
  std::vector<PreparedBlock> blocks_;
  /// Packed in the same fallback-block/canonical-entry order as the registry.
  std::vector<DerivativeReal> fallback_inverse_diagonal_;
  mutable std::vector<DerivativeReal> matrix_scratch_;
  mutable std::vector<DerivativeReal> solve_scratch_;
};

} // namespace qmcplusplus::wftrain

#endif
