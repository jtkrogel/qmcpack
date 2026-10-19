//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file StructuredParameterProvider.h
 * @brief Tensor-level parameter metadata and atomic update interface for scalable training.
 */

#ifndef QMCPLUSPLUS_STRUCTURED_PARAMETER_PROVIDER_H
#define QMCPLUSPLUS_STRUCTURED_PARAMETER_PROVIDER_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <stdexcept>
#include <vector>

namespace qmcplusplus
{
class ParticleSet;
}

namespace qmcplusplus::wftrain
{

/// Identify the scalar representation owned by one parameter block.
enum class ParameterScalarDomain
{
  REAL64,
  COMPLEX128
};

/** Describe one contiguous tensor in a provider's canonical flat ordering.
 *
 * Stable identifiers address tensors, not individual scalars.  This keeps the
 * metadata proportional to the number of model tensors for neural networks.
 */
struct ParameterBlockDescriptor
{
  std::string id;
  std::vector<std::size_t> shape;
  std::size_t offset = 0;
  std::size_t count  = 0;
  ParameterScalarDomain scalar_domain = ParameterScalarDomain::REAL64;
  bool trainable                         = true;
  std::string update_group;
};

/** Immutable, validated description of a provider's canonical parameter vector. */
class StructuredParameterSchema
{
public:
  /// Validate and take ownership of a complete, contiguous block layout.
  StructuredParameterSchema(std::string provider_id, std::vector<ParameterBlockDescriptor> blocks);

  /// Return the stable identity of the owning model component.
  const std::string& providerId() const noexcept { return provider_id_; }

  /// Return all blocks in canonical flat-vector order.
  const std::vector<ParameterBlockDescriptor>& blocks() const noexcept { return blocks_; }

  /// Return the total number of scalar entries covered by the schema.
  std::size_t parameterCount() const noexcept { return parameter_count_; }

  /// Return a deterministic identity for layout and training metadata.
  const std::string& fingerprint() const noexcept { return fingerprint_; }

private:
  std::string provider_id_;
  std::vector<ParameterBlockDescriptor> blocks_;
  std::size_t parameter_count_ = 0;
  std::string fingerprint_;
};

/// Own one self-consistent copy of a versioned real parameter vector.
struct StructuredParameterSnapshot
{
  std::string schema_fingerprint;
  std::size_t version = 0;
  std::vector<double> values;
};

inline constexpr std::size_t NO_AFFINE_BIAS_BLOCK = std::numeric_limits<std::size_t>::max();

/// Describe one trainable affine weight and optional bias in schema block order.
struct AffineParameterBlock
{
  std::string id;
  std::size_t weight_block = 0;
  std::size_t bias_block = NO_AFFINE_BIAS_BLOCK;
  std::size_t input_width = 0;
  std::size_t output_width = 0;
};

/// Row-major activation and score-sensitivity views valid only during observe().
struct AffineScoreObservation
{
  std::size_t affine_block = 0;
  const double* activations = nullptr;
  const double* sensitivities = nullptr;
  std::size_t row_count = 0;
  /// Optional phase response for real parameters; null denotes an identically zero channel.
  const double* imaginary_sensitivities = nullptr;
};

/// Consume transient affine rows without retaining a model tape.
class AffineScoreObserver
{
public:
  virtual ~AffineScoreObserver() = default;
  virtual void observe(const AffineScoreObservation& observation) = 0;
};

/** Abstract synchronization boundary between a model and a training driver.
 *
 * Implementations copy values while holding their own lock and publish complete
 * replacements atomically.  No mutable view may outlive the provider lock.
 */
class StructuredParameterProvider
{
public:
  virtual ~StructuredParameterProvider() = default;

  /// Return immutable tensor metadata whose lifetime matches the provider.
  virtual const StructuredParameterSchema& parameterSchema() const noexcept = 0;

  /// Copy one version-consistent parameter state.
  virtual StructuredParameterSnapshot snapshotParameters() const = 0;

  /// Identify actual numerical device products, not the memory domain of emitted chunks.
  virtual bool usesAcceleratorParameterProducts() const noexcept { return false; }

  /// Detach a speculative clone from shared production parameters before trial publication.
  virtual void isolateParameterState()
  { throw std::runtime_error("Provider does not support isolated candidate evaluation"); }

  /// Return explicitly classified neural-network weight tensor identifiers for decay.
  virtual std::vector<std::string> neuralWeightBlocks() const { return {}; }

  /// Return supported trainable affine boundaries in stable observation order.
  virtual std::vector<AffineParameterBlock> affineParameterBlocks() const { return {}; }

  /// Return trainable generic-diagonal KFAC block ordinals in the parameter schema.
  virtual std::vector<std::size_t> genericKFACParameterBlocks() const { return {}; }

  /// Stream affine rows and copy the canonical real parameter score for one sample.
  virtual void observeAffineScores(const ParticleSet&, AffineScoreObserver&, std::vector<double>&) const
  { throw std::runtime_error("Provider does not support affine score observations"); }

  /// Stream both response channels for real parameters; an empty imaginary score denotes zero.
  virtual void observeComplexAffineScores(const ParticleSet& particles, AffineScoreObserver& observer,
                                          std::vector<double>& real_score,
                                          std::vector<double>& imaginary_score) const
  {
    observeAffineScores(particles, observer, real_score);
    imaginary_score.clear();
  }

  /** Atomically publish a complete candidate if the expected version matches.
   *
   * Returns the newly committed version.  Implementations must reject schema,
   * shape, stale-version, and non-finite-value errors before mutation.
   */
  virtual std::size_t publishParameters(const StructuredParameterSnapshot& candidate,
                                        std::size_t expected_version) = 0;
};

} // namespace qmcplusplus::wftrain

#endif
