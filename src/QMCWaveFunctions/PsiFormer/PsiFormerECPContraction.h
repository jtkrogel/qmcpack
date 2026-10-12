//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerECPContraction.h
 * @brief Bounded offline contraction of flattened nonlocal-ECP score responses.
 *
 * This surface is deliberately below public DEVICE dispatch.  It copies a live
 * Hamiltonian-owned tile into the planned ECP_TILE byte layout, then serializes
 * replacement-score evaluations through one real response[P].  No Q-by-P object is
 * representable here.  Hamiltonian energy reduction and publication remain owned by
 * NonLocalECPotential.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_ECP_CONTRACTION_H
#define QMCPLUSPLUS_PSIFORMER_ECP_CONTRACTION_H

#include "QMCWaveFunctions/Optimization/NonLocalECPDerivative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDerivativeOrchestration.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace qmcplusplus::psiformer
{

/// Canonical sentinel stored for an off-sphere segment with no source center.
inline constexpr std::size_t PSIFORMER_ECP_NO_SOURCE = static_cast<std::size_t>(-1);

enum PsiFormerECPSegmentFlag : std::uint8_t
{
  ECP_SEGMENT_ON_SPHERE = UINT8_C(1) << 0
};

/** Transaction identity and exact bounded extent known before the first tile.
 *
 * EvaluationStamp is intentionally opaque.  The Hamiltonian/provider boundary must
 * translate its already validated stamp vector into ordered records and identify the
 * PsiFormer producer by a nonzero process-local fingerprint.  This adapter never
 * guesses private stamp fields.
 */
struct PsiFormerECPContractionContext
{
  std::uint64_t grid_fingerprint = 0;
  std::uint64_t producer_fingerprint = 0;
  std::size_t expected_point_count = 0;
  std::size_t sample_count = 0;
  std::size_t channel_count = 0;
  wftrain::DerivativeAdjoint adjoint = wftrain::DerivativeAdjoint::TRANSPOSE;
};

/** Copy-return inspection of byte-serialized ECP_TILE storage.
 *
 * No method forms a typed pointer into the raw allocation.  The view and its backing
 * contraction/storage must remain alive for each accessor call.
 */
class PsiFormerCopiedECPTileView
{
public:
  PsiFormerECPTileMetadata metadata() const;
  std::size_t segmentOffset(std::size_t index) const;
  std::size_t segmentWalkerId(std::size_t index) const;
  std::size_t segmentElectronId(std::size_t index) const;
  std::size_t segmentSourceCenterId(std::size_t index) const;
  std::uint8_t segmentFlags(std::size_t index) const;
  double referencePosition(std::size_t index) const;
  std::array<double, 3> replacementPosition(std::size_t point) const;
  double bareWeight(std::size_t point) const;
  wftrain::DerivativeValue completeRatio(std::size_t point) const;
  PsiFormerECPEvaluationStampRecord evaluationStamp(std::size_t index) const;
  wftrain::DerivativeValue coefficient(std::size_t index) const;
  wftrain::DerivativeValue sampleWeightSum(std::size_t sample) const;
  std::uint32_t pointStatus(std::size_t point) const;

private:
  friend class PsiFormerECPContraction;
  PsiFormerCopiedECPTileView(const std::byte* storage,
                             PsiFormerECPTileLayout layout) noexcept
      : storage_(storage), layout_(layout)
  {}
  void copyAt(void* destination,
              std::size_t bytes,
              std::size_t offset,
              const char* description) const;

  const std::byte* storage_;
  PsiFormerECPTileLayout layout_;
};

/** Allocation-free score seam used by the CPU oracle and a future device executor.
 *
 * `replacement_position == nullptr` selects the reference configuration.  Otherwise
 * exactly `replaced_electron` is replaced.  The implementation must overwrite all P
 * entries of response and must not retain any input pointer.
 */
class PsiFormerECPScoreEvaluator
{
public:
  PsiFormerECPScoreEvaluator() = default;
  PsiFormerECPScoreEvaluator(const PsiFormerECPScoreEvaluator&) = delete;
  PsiFormerECPScoreEvaluator& operator=(const PsiFormerECPScoreEvaluator&) = delete;
  virtual ~PsiFormerECPScoreEvaluator() = default;

  virtual void evaluate(const double* reference_positions,
                        std::size_t electron_count,
                        std::size_t replaced_electron,
                        const double* replacement_position,
                        double* response,
                        std::size_t parameter_count) = 0;
};

/** Allocation-free transaction over one reusable ECP_TILE and one response[P].
 *
 * `ecp_tile_storage` must be aligned, lifetime-free raw storage and remains strictly
 * byte serialization: this class never starts typed object lifetimes inside it.  The
 * caller must not independently construct overlapping objects or read it through
 * typed pointers, and no object teardown in this storage is required.  Prepared
 * typed host owners hold the O(B*N), O(C*B), and O(B) transaction state required
 * across calls.
 *
 * begin() copies reference positions and channel-major [channel,sample]
 * coefficients, clears the complex accumulators and B weight sums, and binds the
 * traversal identity.  consume() copies the complete live descriptor before doing
 * arithmetic.  finish() evaluates each reference once, subtracts its weighted score,
 * and returns the A7.3 completion token used by PsiFormerDerivativeHostPublisher.
 * Caller inputs to begin()/consume() must not alias ECP_TILE, response, or accumulator
 * storage; the implementation checks this before copying or arithmetic.
 */
class PsiFormerECPContraction
{
public:
  PsiFormerECPContraction(const PsiFormerDerivativePlan& plan,
                          void* ecp_tile_storage,
                          std::size_t ecp_tile_storage_bytes,
                          double* response,
                          std::size_t response_count,
                          double* accumulators,
                          std::size_t accumulator_fp64_elements,
                          PsiFormerECPScoreEvaluator& evaluator);
  PsiFormerECPContraction(const PsiFormerECPContraction&) = delete;
  PsiFormerECPContraction& operator=(const PsiFormerECPContraction&) = delete;
  PsiFormerECPContraction(PsiFormerECPContraction&&) = delete;
  PsiFormerECPContraction& operator=(PsiFormerECPContraction&&) = delete;

  void begin(
      const PsiFormerECPContractionContext& context,
      wftrain::DerivativeArrayView<const double> reference_positions,
      wftrain::DerivativeArrayView<const wftrain::DerivativeValue> coefficients);

  /** Copy and immediately contract one live tile.
   *
   * `validated_stamp_records` is the host-provided, order-preserving translation of
   * tile.evaluation_stamps after the Hamiltonian/provider has validated the opaque
   * identities.  The records are copied verbatim and must include this transaction's
   * producer/version pair.
   */
  PsiFormerECPTileLaunchRecord consume(
      const wftrain::NonLocalECPDerivativeTile& tile,
      wftrain::DerivativeArrayView<const PsiFormerECPEvaluationStampRecord>
          validated_stamp_records);

  PsiFormerDerivativeCompletionToken finish();

  const PsiFormerECPContractionContext& context() const noexcept { return context_; }
  std::size_t consumedPointCount() const noexcept { return consumed_point_count_; }
  const double* accumulators() const noexcept { return accumulators_; }
  std::size_t accumulatorFP64Elements() const noexcept
  {
    return accumulator_fp64_elements_;
  }
  /// Inspect the constructed ECP_TILE storage without permitting arbitrary raw bytes.
  PsiFormerCopiedECPTileView copiedTileView() const;

private:
  enum class State : std::uint8_t
  {
    IDLE,
    ACTIVE,
    COMPLETE,
    POISONED
  };

  void requireActive(const char* operation) const;
  void poison() noexcept { state_ = State::POISONED; }
  void evaluateAndFold(std::size_t sample,
                       std::size_t replaced_electron,
                       const double* replacement_position,
                       wftrain::DerivativeValue weight);

  PsiFormerDerivativePlan plan_;
  std::byte* ecp_tile_storage_ = nullptr;
  std::size_t ecp_tile_storage_bytes_ = 0;
  double* response_ = nullptr;
  std::size_t response_count_ = 0;
  double* accumulators_ = nullptr;
  std::size_t accumulator_fp64_elements_ = 0;
  PsiFormerECPScoreEvaluator& evaluator_;
  PsiFormerECPContractionContext context_;
  std::vector<double> reference_positions_;
  std::vector<wftrain::DerivativeValue> coefficients_;
  std::vector<wftrain::DerivativeValue> sample_weight_sums_;
  std::size_t expected_tile_ordinal_ = 0;
  std::size_t consumed_point_count_ = 0;
  State state_ = State::IDLE;
};

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_ECP_CONTRACTION_H
