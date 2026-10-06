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
#include <string_view>

namespace qmcplusplus::wftrain
{

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

/** Consume live ECP tiles without a virtual-point-by-parameter staging matrix.
 *
 * Implementations own at most a fixed number of O(P) contraction vectors and
 * O(B) scalar reductions.  A begun transaction must finish with end() or abort().
 */
class NonLocalECPDerivativeConsumer
{
public:
  NonLocalECPDerivativeConsumer() = default;
  NonLocalECPDerivativeConsumer(const NonLocalECPDerivativeConsumer&) = delete;
  NonLocalECPDerivativeConsumer& operator=(const NonLocalECPDerivativeConsumer&) = delete;
  virtual ~NonLocalECPDerivativeConsumer() = default;

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

  /// Return exact retained-storage diagnostics for bounded-memory tests.
  virtual StreamingDerivativeStorageDiagnostics storageDiagnostics() const = 0;
};

} // namespace qmcplusplus::wftrain

#endif
