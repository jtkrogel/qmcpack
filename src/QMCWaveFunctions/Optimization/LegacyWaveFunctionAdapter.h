//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file LegacyWaveFunctionAdapter.h
 * @brief Bounded compatibility adapter from scalar legacy variables to structured training.
 *
 * This bridge intentionally retains the scalar legacy derivative implementation.  It
 * provides O(P) streaming contractions for ordinary, modest parameter sets, but does
 * not claim neural-scale batching or device capabilities. Accepted-grid ECP
 * products are separately restricted to real CPU fixed-Slater/J1/J2 profiles.
 */

#ifndef QMCPLUSPLUS_LEGACY_WAVEFUNCTION_ADAPTER_H
#define QMCPLUSPLUS_LEGACY_WAVEFUNCTION_ADAPTER_H

#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"
#include "QMCWaveFunctions/Optimization/NonLocalECPDerivative.h"
#include "QMCWaveFunctions/VariableSet.h"
#include "type_traits/template_types.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace qmcplusplus
{
class ParticleSet;
class TrialWaveFunction;

namespace wftrain
{
struct LegacyDerivativeWorkspace;

/// A bounded, exclusively owned evaluator loaded with one frozen sample.
struct LegacyWaveFunctionReplaySample
{
  TrialWaveFunction& wavefunction;
  ParticleSet& particles;
};

/** Replay sample coordinates into private evaluator state, independent of blocks.
 * prepareSample must restore all configuration-dependent state, including distance
 * tables. sampleCount must remain constant for the lifetime of every view.
 * The adapter installs its immutable parameter snapshot and full VGL. Replay
 * workers may be shared only among views from the same adapter/workspace lane.
 */
class LegacyWaveFunctionReplay
{
public:
  virtual ~LegacyWaveFunctionReplay() = default;
  virtual std::size_t sampleCount() const noexcept = 0;
  virtual LegacyWaveFunctionReplaySample prepareSample(std::size_t index) = 0;
};

/// Hard resource limits required for every legacy compatibility adapter.
struct LegacyWaveFunctionAdapterOptions
{
  std::size_t maximum_parameter_count         = 0;
  std::size_t maximum_derivative_scratch_bytes = 0;
  std::size_t maximum_parameter_chunk_size     = 0;
};

/** Own a detached, versioned copy of one legacy wavefunction's active variables.
 *
 * Publication never invokes legacy reset callbacks.  A snapshot is installed only
 * into caller-owned, disposable evaluators while constructing a derivative operator,
 * so a failing legacy callback cannot partially mutate the authoritative state.
 */
class LegacyWaveFunctionAdapter final : public StructuredParameterProvider
{
public:
  LegacyWaveFunctionAdapter(std::string provider_id,
                            TrialWaveFunction& registration_source,
                            LegacyWaveFunctionAdapterOptions options);

  const StructuredParameterSchema& parameterSchema() const noexcept override { return *schema_; }

  StructuredParameterSnapshot snapshotParameters() const override;

  std::size_t publishParameters(const StructuredParameterSnapshot& candidate,
                                std::size_t expected_version) override;

  /** Activate a prevalidated, privately prepared snapshot without allocation.
   * The binding must establish exclusive publication, matching schema/extent,
   * finite values, candidate.version == current version, and no version overflow
   * before collective readiness. This is the no-fail half of that transaction.
   */
  std::size_t commitPreparedParameters(StructuredParameterSnapshot&& candidate) noexcept;

  const optimize::VariableSet& registrationVariables() const noexcept { return *registration_template_; }

  /// Reuse the exact product-profile admission in the native physical binding.
  void validateNonLocalECPProfile(TrialWaveFunction& evaluator, ParticleSet& particles) const;

  /// Reset only caller-owned private component state; do not evaluate or mutate particles.
  void resetPrivateParameters(TrialWaveFunction& evaluator,
                              const StructuredParameterSnapshot& snapshot) const;

  /// Install a matching snapshot and recompute complete VGL in private evaluator state.
  void installSnapshot(TrialWaveFunction& evaluator,
                       ParticleSet& particles,
                       const StructuredParameterSnapshot& snapshot) const;

  /** Capture one accepted ordinary-locality grid without parameter contractions.
   * Real CPU fixed-Slater/J1/J2 only, with active Jastrows and fixed orbitals.
   * The supplied sink stays idle. Energy and replay must come from the same
   * streaming Hamiltonian traversal. No parameter rows are retained per grid.
   */
  std::unique_ptr<NonLocalECPDerivativeConsumer> makeNonLocalECPDerivativeConsumer(
      std::shared_ptr<LegacyWaveFunctionReplay> replay,
      std::shared_ptr<const StructuredParameterSnapshot> snapshot,
      std::size_t batch_ordinal,
      std::size_t sample_offset) const;

  /** Share one iteration snapshot and one bounded replay worker across block views.
   * All operators from this adapter lease the same scratch workspace. Actions are
   * serialized; retained block views do not allocate additional parameter rows.
   * The binding owns/accounts the shared iteration snapshot and replay worker;
   * storageDiagnostics reports the shared derivative workspace once by identity.
   */
  std::unique_ptr<StreamingDerivativeOperator> makeDerivativeOperator(
      std::shared_ptr<LegacyWaveFunctionReplay> replay,
      std::shared_ptr<const StructuredParameterSnapshot> snapshot,
      std::size_t batch_ordinal,
      std::size_t sample_offset) const;

  /** Bind one immutable snapshot to an exclusive evaluator per sample.
   *
   * The caller must keep every evaluator and particle set alive and must not use or
   * mutate them until the returned operator is destroyed.  Failed construction leaves
   * the adapter state unchanged; the supplied evaluators should then be discarded.
   */
  std::unique_ptr<StreamingDerivativeOperator> makeDerivativeOperator(
      const RefVector<TrialWaveFunction>& evaluators,
      const RefVector<ParticleSet>& particle_sets,
      std::size_t batch_ordinal,
      std::size_t sample_offset) const;

private:
  std::shared_ptr<const StructuredParameterSchema> schema_;
  // Shared with ephemeral operators so each operator does not duplicate another
  // full legacy registration table in addition to its derivative scratch rows.
  std::shared_ptr<const optimize::VariableSet> registration_template_;
  LegacyWaveFunctionAdapterOptions options_;
  std::shared_ptr<LegacyDerivativeWorkspace> workspace_;

  mutable std::mutex state_mutex_;
  std::size_t parameter_version_ = 0;
  std::vector<double> parameter_values_;
};

} // namespace wftrain
} // namespace qmcplusplus

#endif
