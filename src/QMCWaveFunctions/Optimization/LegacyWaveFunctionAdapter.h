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
 * not claim neural-scale batching, threading, device, or nonlocal-ECP capabilities.
 */

#ifndef QMCPLUSPLUS_LEGACY_WAVEFUNCTION_ADAPTER_H
#define QMCPLUSPLUS_LEGACY_WAVEFUNCTION_ADAPTER_H

#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"
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

  mutable std::mutex state_mutex_;
  std::size_t parameter_version_ = 0;
  std::vector<double> parameter_values_;
};

} // namespace wftrain
} // namespace qmcplusplus

#endif
