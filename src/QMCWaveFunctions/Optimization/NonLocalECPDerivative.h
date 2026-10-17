//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file NonLocalECPDerivative.h
 * @brief Bounded inline-consumer contract for nonlocal-ECP parameter derivatives.
 *
 * VirtualParticleBatch views are transient tile storage.  This interface therefore
 * requires consumers to contract each tile while its positions, bare weights, and
 * complete TrialWaveFunction ratios are live; retaining the view is forbidden.
 */

#ifndef QMCPLUSPLUS_NONLOCALECPDERIVATIVE_H
#define QMCPLUSPLUS_NONLOCALECPDERIVATIVE_H

#include "Particle/VirtualParticleBatch.h"
#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace qmcplusplus { class QMCHamiltonian; }

namespace qmcplusplus::wftrain
{

/// Native-only seal of one accepted ordinary-locality traversal, not caller-supplied potential data.
class NonLocalECPNativeCapture
{
public:
  /// Borrow the immutable scientific identity pinned by the native Hamiltonian.
  std::string_view scientificDefinition() const noexcept { return scientific_definition_; }
  /// Match this native result to the exact one-shot replay producer.
  const std::shared_ptr<const void>& captureIdentity() const noexcept { return capture_identity_; }
  /// Identify the completed grid and its per-sample REAL potential, including Re(nonlocal).
  std::uint64_t gridFingerprint() const noexcept { return grid_; }
  DerivativeArrayView<const double> reportedPotentials() const noexcept
  { return {potentials_.data(), potentials_.size()}; }
  /// Account newly retained receipt storage separately from the shared immutable PP owner.
  std::size_t hostBytes() const noexcept { return host_bytes_; }

private:
  friend class qmcplusplus::QMCHamiltonian;
  NonLocalECPNativeCapture() = default;
  std::shared_ptr<const void> provenance_owner_, capture_identity_;
  std::string_view scientific_definition_;
  std::vector<double> potentials_;
  std::size_t host_bytes_ = 0;
  std::uint64_t grid_ = 0;
};

/// Identify the localization rule used to construct one live ECP traversal.
enum class NonLocalECPLocalization
{
  ORDINARY_LOCALITY,
  DLA,
  TMDLA
};

/** Describe one complete inline ECP contraction transaction.
 *
 * The fingerprint identifies the accepted stochastic or fixed grid traversal;
 * each tile repeats it so accidental mixing of traversals fails before arithmetic.
 */
struct NonLocalECPDerivativeContext
{
  std::string_view provider_id;
  std::string_view schema_fingerprint;
  std::size_t parameter_version = 0;
  std::size_t batch_ordinal     = 0;
  std::size_t sample_offset     = 0;
  std::size_t sample_count      = 0;
  std::size_t expected_point_count = 0;
  std::uint64_t grid_fingerprint = 0;
  NonLocalECPLocalization localization = NonLocalECPLocalization::ORDINARY_LOCALITY;
  bool uses_virtual_particles          = true;
  bool scalar_relativistic             = true;
  bool used_dense_derivative_fallback  = false;
};

/** Nonowning data for one live flattened quadrature tile.
 *
 * `bare_weights * complete_ratios` is the exact coefficient used by the
 * ordinary-locality energy.  Fermionic-only ratios are not valid here.
 */
struct NonLocalECPDerivativeTile
{
  const VirtualParticleBatch& virtual_particles;
  DerivativeArrayView<const QMCTraits::ValueType> bare_weights;
  DerivativeArrayView<const QMCTraits::ValueType> complete_ratios;
  DerivativeArrayView<const WaveFunctionComponent::EvaluationStamp> evaluation_stamps;
  std::size_t tile_ordinal       = 0;
  std::uint64_t grid_fingerprint = 0;
};

/** One immutable quadrature point retained for exact parameter-direction replay. */
struct NonLocalECPReplayPoint
{
  std::size_t sample = 0;
  std::size_t electron = 0;
  std::array<double, 3> replacement_position{};
  DerivativeValue bare_weight{};
  DerivativeValue complete_ratio{};
};

/** Preserve one accepted virtual-particle segment and its flat point interval. */
struct NonLocalECPReplaySegment
{
  std::size_t point_offset = 0;
  std::size_t point_count = 0;
  std::size_t sample = 0;
  std::size_t electron = 0;
  bool on_sphere = false;
  std::size_t source_center = static_cast<std::size_t>(-1);
};

/** Preserve exact tile boundaries and opaque value-evaluation provenance.
 *
 * Evaluation stamps are retained for restart/audit identity.  Replay arithmetic
 * uses the already accepted positions and complete ratios and never reevaluates
 * the stamped TrialWaveFunction ratio path.
 */
struct NonLocalECPReplayTile
{
  std::size_t tile_ordinal = 0;
  std::size_t point_offset = 0;
  std::size_t point_count = 0;
  std::size_t segment_offset = 0;
  std::size_t segment_count = 0;
  std::size_t stamp_offset = 0;
  std::size_t stamp_count = 0;
  std::uint64_t virtual_batch_fingerprint = 0;
};

/// Exact retained-storage accounting for one immutable accepted-grid replay.
struct NonLocalECPReplayStorageDiagnostics
{
  std::size_t point_count = 0;
  std::size_t segment_count = 0;
  std::size_t tile_count = 0;
  std::size_t evaluation_stamp_count = 0;
  std::size_t reference_position_bytes = 0;
  std::size_t point_bytes = 0;
  std::size_t provenance_bytes = 0;
  std::size_t retained_numeric_bytes = 0;
  std::size_t storage_fingerprint = 0;
};

/** Immutable O(Q) record of one accepted ordinary-locality ECP traversal. */
class NonLocalECPJVPReplay
{
public:
  NonLocalECPJVPReplay() = default;
  NonLocalECPJVPReplay(const NonLocalECPJVPReplay&) = delete;
  NonLocalECPJVPReplay& operator=(const NonLocalECPJVPReplay&) = delete;
  virtual ~NonLocalECPJVPReplay() = default;

  virtual std::string_view providerId() const noexcept = 0;
  virtual std::string_view schemaFingerprint() const noexcept = 0;
  virtual std::size_t parameterVersion() const noexcept = 0;
  virtual std::size_t batchOrdinal() const noexcept = 0;
  virtual std::size_t sampleOffset() const noexcept = 0;
  virtual std::size_t sampleCount() const noexcept = 0;
  virtual std::size_t electronCount() const noexcept = 0;
  virtual std::uint32_t localEnergyTermMask() const noexcept = 0;
  virtual std::uint64_t gridFingerprint() const noexcept = 0;
  virtual DerivativeArrayView<const double> referencePositions() const noexcept = 0;
  virtual DerivativeArrayView<const NonLocalECPReplayPoint> points() const noexcept = 0;
  virtual DerivativeArrayView<const NonLocalECPReplaySegment> segments() const noexcept = 0;
  virtual DerivativeArrayView<const NonLocalECPReplayTile> tiles() const noexcept = 0;
  virtual DerivativeArrayView<const WaveFunctionComponent::EvaluationStamp>
      evaluationStamps() const noexcept = 0;
  virtual NonLocalECPReplayStorageDiagnostics storageDiagnostics() const noexcept = 0;
};

/** Consume live ECP tiles without a virtual-point-by-parameter staging matrix.
 *
 * Implementations own at most a fixed number of O(P) contraction vectors and
 * O(B) scalar reductions.  Implementations may optionally retain an O(Q)
 * accepted-grid replay when explicitly requested by the factory; ordinary VJP
 * consumers need not pay this cost.  A begun transaction must finish with
 * end() or abort().
 */
class NonLocalECPDerivativeConsumer
{
public:
  NonLocalECPDerivativeConsumer() = default;
  NonLocalECPDerivativeConsumer(const NonLocalECPDerivativeConsumer&) = delete;
  NonLocalECPDerivativeConsumer& operator=(const NonLocalECPDerivativeConsumer&) = delete;
  virtual ~NonLocalECPDerivativeConsumer() = default;

  /// Optional one-shot replay identity; ordinary response consumers retain their existing empty contract.
  virtual std::shared_ptr<const void> nativeCaptureIdentity() const noexcept { return {}; }

  /** Validate immutable identity and channel metadata before any grid draw.
   *
   * The preflight context deliberately has a zero grid fingerprint and point
   * count; begin() later binds the accepted rotated traversal.
   */
  virtual void preflight(
      const NonLocalECPDerivativeContext& context,
      DerivativeArrayView<const VJPCoefficientChannel> channels,
      const ParameterReductionSink& sink) const = 0;

  /// Begin one ordinary-locality transaction and activate its checked sink.
  virtual void begin(const NonLocalECPDerivativeContext& context,
                     DerivativeArrayView<const VJPCoefficientChannel> channels,
                     ParameterReductionSink& sink) = 0;

  /// Contract one transient descriptor tile before its backing storage is reused.
  virtual void consume(const NonLocalECPDerivativeTile& tile) = 0;

  /// Fold reference scores, publish canonical chunks, and complete the sink.
  virtual void end() = 0;

  /// Abandon partial arithmetic and poison the active sink without throwing.
  virtual void abort() noexcept = 0;

  /** Transfer an explicitly requested completed accepted-grid replay once.
   *
   * The transaction is one-shot even after transfer; callers must construct a
   * new consumer for another accepted grid.  Implementations reject this call
   * when replay capture was not requested.
   */
  virtual std::unique_ptr<NonLocalECPJVPReplay> releaseJVPReplay() = 0;

  /// Return exact retained-storage diagnostics for bounded-memory tests.
  virtual StreamingDerivativeStorageDiagnostics storageDiagnostics() const = 0;
};

} // namespace qmcplusplus::wftrain

#endif
