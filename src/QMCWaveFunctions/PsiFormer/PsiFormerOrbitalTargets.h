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
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeterminantOverlap.h"
#include "QMCWaveFunctions/SPOSet.h"

#include <cstddef>
#include <array>
#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qmcplusplus
{
class MultiSlaterDetTableMethod;
class ParticleSet;
class SlaterDet;
class TrialWaveFunction;

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

#ifdef QMC_COMPLEX
/** Frozen FreeOrbital single-reference targets in fixed-gauge periodic-part form. */
class ComplexOrbitalTargetEvaluator
{
public:
  /// Freeze actual occupied FreeOrbital momenta, spin ranges and the matching Bloch convention.
  static ComplexOrbitalTargetEvaluator fromSingleReference(const SlaterDet& source,
      const ParticleSet& electrons, std::array<double, 3> expected_reduced_twist, std::size_t channels);
  /// Fill one reusable periodic-part target after validating the current sample metadata.
  const std::vector<std::complex<double>>& evaluate(const ParticleSet& electrons);
  /// Return immutable occupation/channel and exact scientific identity metadata.
  const OrbitalTargetDescriptor& descriptor() const noexcept { return descriptor_; }
  /// Explicit target/scratch only; cloned SPO and immutable descriptor metadata are excluded.
  std::size_t retainedScalarBytes() const noexcept;

private:
  ComplexOrbitalTargetEvaluator(OrbitalTargetDescriptor descriptor,
      std::array<std::unique_ptr<SPOSet>, 2>&& orbitals, std::array<double, 9> lattice,
      std::array<double, 3> reduced, std::array<double, 3> cartesian);
  OrbitalTargetDescriptor descriptor_;
  std::array<std::unique_ptr<SPOSet>, 2> orbitals_;
  std::array<SPOSet::ValueVector, 2> scratch_;
  const std::array<double, 9> lattice_;
  const std::array<double, 3> reduced_twist_, cartesian_twist_;
  std::vector<std::complex<double>> target_;
};
#endif

/// Select whether overlap channels repeat a teacher or pair with selected CI terms.
enum class DeterminantOverlapTargetMode
{
  GENERAL_FROZEN_TEACHER,
  TOP_COEFFICIENT_MULTIDETERMINANT
};

/** Freeze the identity and channel mapping of a determinant-overlap teacher. */
class DeterminantOverlapTargetDescriptor
{
public:
  /// Repeat one arbitrary named frozen teacher for every student determinant.
  static DeterminantOverlapTargetDescriptor makeGeneralFrozenTeacher(
      std::string source_identity, std::size_t determinant_count);

  /// Pair channels with stable descending-|c| nonzero terms of one DETS expansion.
  static DeterminantOverlapTargetDescriptor makeTopCoefficientMultideterminant(
      const MultiSlaterDetTableMethod& source,
      std::string source_identity,
      std::size_t determinant_count);

  /// Return the target construction convention.
  DeterminantOverlapTargetMode mode() const noexcept { return mode_; }

  /// Return the stable externally supplied identity of the frozen teacher.
  const std::string& sourceIdentity() const noexcept { return source_identity_; }

  /// Return selected source CI ordinals, empty for repeated general targets.
  const std::vector<std::size_t>& sourceOrdinals() const noexcept { return source_ordinals_; }

  /// Return the selected raw CI coefficients, empty for repeated general targets.
  const std::vector<double>& selectedCoefficients() const noexcept { return selected_coefficients_; }

  /// Return the number of PsiFormer determinant channels.
  std::size_t channelCount() const noexcept { return channel_count_; }

  /// Return a deterministic identity for checkpoint/restart validation.
  std::uint64_t fingerprint() const noexcept { return fingerprint_; }

private:
  DeterminantOverlapTargetDescriptor(DeterminantOverlapTargetMode mode,
                                     std::string source_identity,
                                     std::size_t channel_count,
                                     std::vector<std::size_t> source_ordinals,
                                     std::vector<double> selected_coefficients);

  DeterminantOverlapTargetMode mode_;
  std::string source_identity_;
  std::size_t channel_count_;
  std::vector<std::size_t> source_ordinals_;
  std::vector<double> selected_coefficients_;
  std::uint64_t fingerprint_;
};

/** Evaluate signed/log targets from an externally owned, frozen TrialWaveFunction.
 *
 * Ownership and name lookup remain driver concerns.  For selected multideterminant
 * targets, every nonfermionic component (notably a common Jastrow) is retained by
 * replacing only the teacher's full MSD component with one selected CI term.
 */
class DeterminantOverlapTargetEvaluator
{
public:
  /** Freeze a general or top-coefficient target directly from a named teacher. */
  static DeterminantOverlapTargetEvaluator fromFrozenTeacher(
      const TrialWaveFunction& teacher,
      DeterminantOverlapTargetMode mode,
      std::string source_identity,
      std::size_t channel_count);

  /// Construct an allocation-stable evaluator for one immutable descriptor.
  explicit DeterminantOverlapTargetEvaluator(DeterminantOverlapTargetDescriptor descriptor);

  /// Evaluate the full teacher and all target channels at one configuration.
  void evaluate(TrialWaveFunction& frozen_teacher,
                ParticleSet& electrons,
                RealSignedLogAmplitude& teacher,
                RealSignedLogAmplitude* targets,
                std::size_t target_count) const;

  /// Return the immutable target mapping and restart identity.
  const DeterminantOverlapTargetDescriptor& descriptor() const noexcept { return descriptor_; }

private:
  DeterminantOverlapTargetDescriptor descriptor_;
};

} // namespace psiformer
} // namespace qmcplusplus

#endif // QMCPLUSPLUS_PSIFORMER_ORBITAL_TARGETS_H
