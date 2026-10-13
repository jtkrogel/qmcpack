//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file DeterminantOverlapObjective.h
 * @brief Distributed two-pass determinant-overlap objective for native walkers.
 */

#ifndef QMCPLUSPLUS_DETERMINANT_OVERLAP_OBJECTIVE_H
#define QMCPLUSPLUS_DETERMINANT_OVERLAP_OBJECTIVE_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerDeterminantChannelBatch.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOrbitalTargets.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

class Communicate;

namespace qmcplusplus
{
class ParticleSet;
class TrialWaveFunction;

namespace wftrain
{

/** Bind one copied student batch to the same live teacher walker configurations. */
struct DeterminantOverlapBatchView
{
  psiformer::DeterminantChannelBatch* student = nullptr;
  std::vector<std::reference_wrapper<TrialWaveFunction>> teachers;
  std::vector<std::reference_wrapper<ParticleSet>> electrons;
};

/** Return globally reduced overlap diagnostics and one canonical student gradient. */
struct DeterminantOverlapObjectiveResult
{
  double loss = 0.0;
  std::size_t sample_count = 0;
  std::size_t parameter_version = 0;
  std::uint64_t target_fingerprint = 0;
  std::string target_provenance;
  std::vector<psiformer::DeterminantOverlapChannelDiagnostics> channels;
  std::vector<double> parameter_gradient;
  std::size_t retained_numeric_bytes = 0;
};

/** Evaluate a normalized-overlap loss and gradient without B-by-M or M-by-P storage. */
class DeterminantOverlapObjective
{
public:
  DeterminantOverlapObjective(psiformer::DeterminantOverlapTargetEvaluator target,
                              std::vector<double> objective_weights,
                              psiformer::DeterminantOverlapPolicy policy = {});

  DeterminantOverlapObjective(const DeterminantOverlapObjective&) = delete;
  DeterminantOverlapObjective& operator=(const DeterminantOverlapObjective&) = delete;
  DeterminantOverlapObjective(DeterminantOverlapObjective&&) = default;
  DeterminantOverlapObjective& operator=(DeterminantOverlapObjective&&) = default;

  /// Execute two deterministic passes over unchanged native walker configurations.
  DeterminantOverlapObjectiveResult evaluate(
      Communicate& communicator,
      const std::vector<DeterminantOverlapBatchView>& batches) const;

  /// Return the immutable target mapping and provenance identity.
  const psiformer::DeterminantOverlapTargetDescriptor& targetDescriptor() const noexcept
  {
    return target_.descriptor();
  }

  /// Return a stable identity for target mapping, channel weights, and numerical floors.
  std::uint64_t configurationFingerprint() const noexcept { return configuration_fingerprint_; }

private:
  psiformer::DeterminantOverlapTargetEvaluator target_;
  std::vector<double> objective_weights_;
  psiformer::DeterminantOverlapPolicy policy_;
  std::uint64_t configuration_fingerprint_ = 0;
};

/** Construct equal normalized channel weights. */
std::vector<double> makeEqualOverlapWeights(std::size_t channel_count);

/** Construct normalized |c|^2 weights from a paired multideterminant descriptor. */
std::vector<double> makeCoefficientSquaredOverlapWeights(
    const psiformer::DeterminantOverlapTargetDescriptor& descriptor);

} // namespace wftrain
} // namespace qmcplusplus

#endif // QMCPLUSPLUS_DETERMINANT_OVERLAP_OBJECTIVE_H
