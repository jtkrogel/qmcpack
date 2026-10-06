//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerOrbitalTargets.h
 * @brief Immutable conventional-orbital targets for PsiFormer pretraining.
 *
 * The descriptor freezes determinant selection, occupations, coefficients, and
 * source identity.  The evaluator separately owns cloned SPO sets and constructs
 * full spin-block-diagonal target matrices at one electron configuration.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_ORBITAL_TARGETS_H
#define QMCPLUSPLUS_PSIFORMER_ORBITAL_TARGETS_H

#include "Configuration.h"
#include "QMCWaveFunctions/SPOSet.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qmcplusplus
{
class MultiSlaterDetTableMethod;
class ParticleSet;
class SlaterDet;

namespace psiformer
{

/// Select how conventional determinant targets are assigned to PsiFormer channels.
enum class OrbitalTargetMode
{
  SINGLE_REFERENCE,
  TRUNCATED_MULTIDETERMINANT
};

/// Freeze one selected determinant's source ordinal, occupations, and raw coefficient.
struct OrbitalTargetTerm
{
  std::size_t source_ordinal = 0;
  std::vector<std::size_t> occupations_up;
  std::vector<std::size_t> occupations_down;
  double coefficient = 1.0;
};

/** Own the immutable identity and determinant data for an orbital pretraining target. */
class OrbitalTargetDescriptor
{
public:
  /// Return the serialized semantic version of this target convention.
  static constexpr std::uint64_t formatVersion() noexcept { return 1; }

  /// Confirm that selected CI coefficients are retained without normalization.
  static constexpr bool renormalizesCoefficients() noexcept { return false; }

  /// Confirm that targets always use full N_e by N_e matrices with zero off-spin blocks.
  static constexpr bool usesFullSpinBlockMatrix() noexcept { return true; }

  /// Repeat the conventional ground-state occupations for every PsiFormer channel.
  static OrbitalTargetDescriptor makeSingleReference(std::string source_identity,
                                                      std::size_t spin_up_electrons,
                                                      std::size_t spin_down_electrons,
                                                      std::size_t determinant_count);

  /** Select explicit DETS terms by stable descending coefficient magnitude. */
  static OrbitalTargetDescriptor makeTruncatedMultideterminant(
      std::string source_identity,
      std::size_t spin_up_electrons,
      std::size_t spin_down_electrons,
      std::size_t determinant_count,
      const std::vector<double>& coefficients,
      const std::vector<std::vector<std::size_t>>& determinant_to_unique,
      const std::vector<std::vector<std::vector<std::size_t>>>& unique_occupations);

  /// Return the immutable target selection mode.
  OrbitalTargetMode mode() const noexcept { return mode_; }

  /**
   * Return the caller-supplied source identity included in checkpoint identity.
   *
   * Generic SPOSet does not expose a portable content hash, so callers must use
   * a stable configuration/content identity rather than a transient object name.
   */
  const std::string& sourceIdentity() const noexcept { return source_identity_; }

  /// Return the fixed alpha electron count.
  std::size_t spinUpElectrons() const noexcept { return spin_up_electrons_; }

  /// Return the fixed beta electron count.
  std::size_t spinDownElectrons() const noexcept { return spin_down_electrons_; }

  /// Return the full electron count.
  std::size_t electronCount() const noexcept { return spin_up_electrons_ + spin_down_electrons_; }

  /// Return the selected determinant targets in PsiFormer channel order.
  const std::vector<OrbitalTargetTerm>& terms() const noexcept { return terms_; }

  /// Return a deterministic bit-exact identity for target/restart validation.
  std::uint64_t fingerprint() const noexcept { return fingerprint_; }

private:
  /// Validate and freeze one fully specified target descriptor.
  OrbitalTargetDescriptor(OrbitalTargetMode mode,
                          std::string source_identity,
                          std::size_t spin_up_electrons,
                          std::size_t spin_down_electrons,
                          std::vector<OrbitalTargetTerm> terms);

  OrbitalTargetMode mode_;
  std::string source_identity_;
  std::size_t spin_up_electrons_;
  std::size_t spin_down_electrons_;
  std::vector<OrbitalTargetTerm> terms_;
  std::uint64_t fingerprint_;
};

/** Evaluate one immutable target through private SPO clones and fixed scratch. */
class OrbitalTargetEvaluator
{
public:
  /// Freeze a conventional single-determinant source and repeat it across channels.
  static OrbitalTargetEvaluator fromSingleReference(const SlaterDet& source,
                                                    std::string source_identity,
                                                    std::size_t determinant_count);

  /// Freeze and rank an explicit DETS source without coefficient renormalization.
  static OrbitalTargetEvaluator fromMultideterminant(const MultiSlaterDetTableMethod& source,
                                                     std::string source_identity,
                                                     std::size_t determinant_count);

  /// Construct from an already validated descriptor and one SPO clone per spin sector.
  OrbitalTargetEvaluator(OrbitalTargetDescriptor descriptor,
                         std::vector<std::unique_ptr<SPOSet>>&& source_orbitals);

  OrbitalTargetEvaluator(const OrbitalTargetEvaluator&) = delete;
  OrbitalTargetEvaluator& operator=(const OrbitalTargetEvaluator&) = delete;
  OrbitalTargetEvaluator(OrbitalTargetEvaluator&&) = default;
  OrbitalTargetEvaluator& operator=(OrbitalTargetEvaluator&&) = default;

  /// Evaluate and return D x N_e x N_e row-major target matrices.
  const std::vector<double>& evaluate(const ParticleSet& electrons);

  /// Return the immutable target/checkpoint identity.
  const OrbitalTargetDescriptor& descriptor() const noexcept { return descriptor_; }

  /** Return bytes retained by the evaluator's explicit reusable numeric buffers.
   *
   * Opaque storage owned internally by the cloned SPO sets is intentionally not
   * included because the generic SPOSet interface cannot report it.
   */
  std::size_t retainedScalarBytes() const noexcept;

  /// Hash explicit reusable-buffer addresses and capacities for warmed-call checks.
  std::size_t storageFingerprint() const noexcept;

private:
  OrbitalTargetDescriptor descriptor_;
  std::vector<std::unique_ptr<SPOSet>> source_orbitals_;
  std::vector<SPOSet::ValueVector> orbital_scratch_;
  std::vector<double> coefficient_roots_;
  std::vector<double> target_matrices_;
};

} // namespace psiformer
} // namespace qmcplusplus

#endif // QMCPLUSPLUS_PSIFORMER_ORBITAL_TARGETS_H
