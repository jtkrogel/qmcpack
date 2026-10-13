//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeterminantChannelBatch.h
 * @brief Bounded production seam for determinant-resolved PsiFormer objectives.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DETERMINANT_CHANNEL_BATCH_H
#define QMCPLUSPLUS_PSIFORMER_DETERMINANT_CHANNEL_BATCH_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerDeterminantOverlap.h"

#include <cstddef>
#include <string>

namespace qmcplusplus::psiformer
{

/** Own one immutable-version batch of configurations and determinant-channel scratch.
 *
 * Implementations retain O(B N_e + M + P) storage.  They must evaluate all M
 * amplitudes without retaining B-by-M data and apply an M-vector of channel seeds
 * through one shared-trunk reverse pass for the requested sample.
 */
class DeterminantChannelBatch
{
public:
  virtual ~DeterminantChannelBatch() = default;

  /// Return the structured provider identity associated with the canonical gradient.
  virtual const std::string& providerId() const noexcept = 0;

  /// Return the structured parameter-schema fingerprint.
  virtual const std::string& schemaFingerprint() const noexcept = 0;

  /// Return the immutable parameter version captured by this batch.
  virtual std::size_t parameterVersion() const noexcept = 0;

  /// Return the number of copied configurations.
  virtual std::size_t sampleCount() const noexcept = 0;

  /// Return the number of independent determinant channels.
  virtual std::size_t channelCount() const noexcept = 0;

  /// Return the canonical parameter-gradient extent.
  virtual std::size_t parameterCount() const noexcept = 0;

  /// Evaluate one sample's determinant channels into caller-owned signed/log storage.
  virtual void evaluateAmplitudes(std::size_t sample,
                                  RealSignedLogAmplitude* output,
                                  std::size_t output_count) const = 0;

  /** Evaluate one shared reverse pass and overwrite the canonical gradient output. */
  virtual void evaluateWeightedScore(std::size_t sample,
                                     const double* channel_weights,
                                     std::size_t channel_count,
                                     double* output,
                                     std::size_t output_count) const = 0;

  /// Return exact retained numeric bytes for memory-budget diagnostics.
  virtual std::size_t retainedNumericBytes() const noexcept = 0;
};

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_DETERMINANT_CHANNEL_BATCH_H
