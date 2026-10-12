//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDerivativeOrchestration.h
 * @brief Prepared folding and host publication for streamed device derivatives.
 *
 * This surface remains below public device dispatch.  A future accelerator executor
 * supplies the asynchronous copies and synchronization through ``HostStager``; the
 * checked public sinks continue to receive only host-accessible values.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DERIVATIVE_ORCHESTRATION_H
#define QMCPLUSPLUS_PSIFORMER_DERIVATIVE_ORCHESTRATION_H

#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDerivativePlanning.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace qmcplusplus::psiformer
{

/// Identify which real reverse pass supplies one component of a complex response.
enum class PsiFormerDerivativeResponsePart : std::uint8_t
{
  REAL,
  IMAGINARY
};

/// Scalar coefficient passed to the real-response/interleaved-complex AXPY leaf.
struct PsiFormerComplexAxpyCoefficient
{
  double real = 0.0;
  double imaginary = 0.0;
};

/// Distinguish completion identities that may be exposed to different public sinks.
enum class PsiFormerDerivativePublicationKind : std::uint8_t
{
  VJP,
  SCORE_JVP
};

/** Completion identity required before any pending derivative result is public.
 *
 * ``status == 0`` is legal only after the backend has completed all launches and
 * finite scans for the response, accumulator, or JVP scalar being exposed.  Any
 * launch, version, or non-finite diagnostic is aggregated into a nonzero status.
 * A VJP token binds the adjoint and channel count; exact channel IDs and coefficient
 * batch provenance remain preflight responsibilities of ``StreamingDerivativeOperator``.
 */
struct PsiFormerDerivativeCompletionToken
{
  std::uint64_t plan_fingerprint = 0;
  std::uint64_t device_layout_fingerprint = 0;
  std::uint64_t device_arena_fingerprint = 0;
  std::size_t parameter_version = 0;
  PsiFormerDerivativePublicationKind kind = PsiFormerDerivativePublicationKind::VJP;
  wftrain::DerivativeAdjoint adjoint = wftrain::DerivativeAdjoint::TRANSPOSE;
  std::size_t channel_count = 0;
  std::size_t batch_ordinal = 0;
  std::size_t sample_offset = 0;
  std::size_t tile_ordinal = 0;
  std::uint32_t status = 0;
};

static_assert(std::is_standard_layout_v<PsiFormerComplexAxpyCoefficient>);
static_assert(std::is_trivially_copyable_v<PsiFormerComplexAxpyCoefficient>);
static_assert(std::is_standard_layout_v<PsiFormerDerivativeCompletionToken>);
static_assert(std::is_trivially_copyable_v<PsiFormerDerivativeCompletionToken>);

/** Derive the exact AXPY coefficient for one real or imaginary reverse response.
 *
 * The derivative represented by an imaginary response is ``+i response`` for a
 * transpose product and ``-i response`` for a Hermitian product.  The caller's
 * complex sample coefficient is applied after that convention change.
 */
PsiFormerComplexAxpyCoefficient makePsiFormerComplexAxpyCoefficient(
    wftrain::DerivativeValue coefficient,
    PsiFormerDerivativeResponsePart response_part,
    wftrain::DerivativeAdjoint adjoint);

/** CPU oracle for the device real-to-complex AXPY leaf.
 *
 * ``accumulators`` is interleaved complex FP64 storage with the canonical stride
 * from ``plan.accumulators``.  The function allocates no memory and touches exactly
 * the requested channel prefixes.
 */
void foldPsiFormerRealResponse(
    const PsiFormerDerivativePlan& plan,
    const double* response,
    std::size_t response_count,
    wftrain::DerivativeArrayView<const wftrain::DerivativeValue> coefficients,
    PsiFormerDerivativeResponsePart response_part,
    wftrain::DerivativeAdjoint adjoint,
    double* accumulators,
    std::size_t accumulator_fp64_elements);

/** Construct a VJP completion identity from one checked prepared plan.
 *
 * Channel IDs and coefficient batch identity have already been validated by the
 * outer ``StreamingDerivativeOperator`` transaction and are not duplicated here.
 */
PsiFormerDerivativeCompletionToken makePsiFormerVJPCompletionToken(
    const PsiFormerDerivativePlan& plan,
    wftrain::DerivativeAdjoint adjoint,
    std::size_t channel_count,
    std::uint32_t status = 0);

/** Construct a VJP token after the owning preparation boundary validated plan.
 *
 * This warmed path checks the POD request but does not rebuild vector-backed plan
 * layouts and allocates no memory.
 */
PsiFormerDerivativeCompletionToken
makePsiFormerVJPCompletionTokenFromPrevalidatedPlan(
    const PsiFormerDerivativePlan& plan,
    wftrain::DerivativeAdjoint adjoint,
    std::size_t channel_count,
    std::uint32_t status = 0);

/// Construct a one-scalar score-JVP completion identity.
PsiFormerDerivativeCompletionToken makePsiFormerJVPCompletionToken(
    const PsiFormerDerivativePlan& plan,
    std::size_t batch_ordinal,
    std::size_t sample_offset,
    std::size_t tile_ordinal,
    std::uint32_t status = 0);

/** Backend seam for one reusable pinned-host staging interval.
 *
 * Implementations may enqueue an asynchronous device-to-host copy in either copy
 * method.  ``synchronize`` must make the immediately preceding copy host-visible.
 * No method may retain the destination after the next call begins.
 */
class PsiFormerDerivativeHostStager
{
public:
  PsiFormerDerivativeHostStager() = default;
  PsiFormerDerivativeHostStager(const PsiFormerDerivativeHostStager&) = delete;
  PsiFormerDerivativeHostStager& operator=(const PsiFormerDerivativeHostStager&) = delete;
  virtual ~PsiFormerDerivativeHostStager() = default;

  /// Stage one interleaved-complex accumulator interval.
  virtual void copyComplexChunk(const double* source_interleaved,
                                std::size_t count,
                                wftrain::DerivativeValue* destination) = 0;

  /// Stage one real score-JVP scalar as a complex value with zero imaginary part.
  virtual void copyRealJVPScalar(const double* source,
                                 wftrain::DerivativeValue* destination) = 0;

  /// Wait until the most recently staged interval is host-visible.
  virtual void synchronize() = 0;
};

/** Prepared publication boundary for one derivative plan.
 *
 * Construction validates and copies the allocation-bearing public
 * ``ParameterChunkPlan``.  After construction, the publisher-owned work on a
 * successful warmed call performs no allocation; virtual stager/sink implementations
 * and exception paths are outside that claim.  VJP delivery is chunk-major/channel-
 * minor, and staging is not reused until synchronous ``sink.add`` returns.
 *
 * The A8 executor or owning ``StreamingDerivativeOperator`` must obtain ``plan`` and
 * ``chunk_plan`` from the same PsiFormer structured-parameter owner and schema.
 * Numeric tensor-interval validation deliberately permits omission of frozen blocks;
 * it cannot authenticate a foreign schema whose intervals happen to be numerically
 * identical and aligned.
 */
class PsiFormerDerivativeHostPublisher
{
public:
  /** Prepare one publisher from co-owned device and structured-parameter plans.
   *
   * The caller is responsible for the common-owner/schema provenance described by
   * the class contract.  Construction copies ``chunk_plan`` and may allocate.
   */
  PsiFormerDerivativeHostPublisher(const PsiFormerDerivativePlan& plan,
                                    wftrain::ParameterChunkPlan chunk_plan);

  /// Stage and publish every selected canonical parameter chunk.
  void publishParameterChunks(
      const PsiFormerDerivativeCompletionToken& completion,
      const double* accumulators,
      std::size_t accumulator_fp64_elements,
      wftrain::DerivativeValue* host_staging,
      std::size_t host_staging_capacity,
      PsiFormerDerivativeHostStager& stager,
      wftrain::ParameterReductionSink& sink) const;

  /** Stage and publish one score-JVP sample scalar.
   *
   * The checked ``SampleProductSink::add`` call remains authoritative for the
   * monotonic sample cursor and tile-order transition.
   */
  void publishJVPScalar(
      const PsiFormerDerivativeCompletionToken& completion,
      const double* device_result,
      wftrain::DerivativeValue* host_staging,
      std::size_t host_staging_capacity,
      PsiFormerDerivativeHostStager& stager,
      wftrain::SampleProductSink& sink) const;

private:
  void validateCompletion(const PsiFormerDerivativeCompletionToken& completion) const;
  void validateParameterSink(const PsiFormerDerivativeCompletionToken& completion,
                             const wftrain::ParameterReductionSink& sink) const;
  void validateSampleSink(const PsiFormerDerivativeCompletionToken& completion,
                          const wftrain::SampleProductSink& sink) const;

  wftrain::ParameterChunkPlan chunk_plan_;
  std::uint64_t plan_fingerprint_ = 0;
  std::uint64_t device_layout_fingerprint_ = 0;
  std::uint64_t device_arena_fingerprint_ = 0;
  std::size_t parameter_version_ = 0;
  std::size_t parameter_count_ = 0;
  std::size_t accumulator_count_ = 0;
  std::size_t accumulator_stride_fp64_elements_ = 0;
  std::size_t accumulator_total_fp64_elements_ = 0;
  std::size_t host_chunk_capacity_ = 0;
  std::size_t host_jvp_capacity_ = 0;
};

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_DERIVATIVE_ORCHESTRATION_H
