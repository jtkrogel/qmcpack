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
#include <string>
#include <vector>

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

