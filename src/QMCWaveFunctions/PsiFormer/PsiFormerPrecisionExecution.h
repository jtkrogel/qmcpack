//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerPrecisionExecution.h
 * @brief Pointer-free mixed-precision storage, conversion, and publication contracts.
 *
 * These records deliberately contain no device pointers or runtime events.  A backend
 * owns those resources and may publish a slot pair only after this state machine has
 * observed the completed copy, conversion, and accepted diagnostics milestones.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_PRECISION_EXECUTION_H
#define QMCPLUSPLUS_PSIFORMER_PRECISION_EXECUTION_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorSchedule.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPrecisionPolicy.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace qmcplusplus::psiformer
{

/** Fixed-layout evidence collected while converting one FP64 parameter image.
 *
 * Non-finite input and overflow are publication hazards.  Subnormal and rounded-to-
 * zero values are recorded for later hardware acceptance but do not independently
 * reject an offline-developed compute image.
 */
struct PsiFormerParameterConversionDiagnostics
{
  std::uint64_t nonfinite_input_count = 0;
  std::uint64_t overflow_count        = 0;
  std::uint64_t subnormal_output_count = 0;
  std::uint64_t underflow_to_zero_count = 0;
  double maximum_absolute_error = 0.0;
  double maximum_relative_error = 0.0;

  /// Return whether this conversion is unsafe to publish.
  bool hasPublicationHazard() const noexcept
  {
    return nonfinite_input_count != 0 || overflow_count != 0;
  }

  /// Merge an independent tile using saturating counters and maximum errors.
  void merge(const PsiFormerParameterConversionDiagnostics& other) noexcept;
};

/// Describe one checked contiguous FP64-to-FP32 parameter conversion tile.
struct PsiFormerParameterConversionTile
{
  std::size_t source_begin      = 0;
  std::size_t destination_begin = 0;
  std::size_t count             = 0;
  PsiFormerLinearLaunch launch;

  /// Return the first source element after this tile.
  std::size_t sourceEnd() const noexcept { return source_begin + count; }

  /// Return the first destination element after this tile.
  std::size_t destinationEnd() const noexcept { return destination_begin + count; }

  friend bool operator==(const PsiFormerParameterConversionTile& lhs,
                         const PsiFormerParameterConversionTile& rhs) noexcept
  {
    return lhs.source_begin == rhs.source_begin &&
        lhs.destination_begin == rhs.destination_begin && lhs.count == rhs.count &&
        lhs.launch == rhs.launch;
  }
};

/// Hold the exact bounded conversion schedule for one canonical parameter vector.
struct PsiFormerParameterConversionSchedule
{
  std::size_t parameter_count = 0;
  std::size_t tile_capacity   = 0;
  std::size_t block_size      = 0;
  std::vector<PsiFormerParameterConversionTile> tiles;
  std::uint64_t fingerprint = 0;
};

/** Immutable storage and conversion identity for one canonical model version. */
struct PsiFormerPrecisionExecutionPlan
{
  PsiFormerPrecisionPolicy policy = PsiFormerPrecisionPolicy::FP64_REFERENCE;
  PsiFormerBackendMathMode math_mode = PsiFormerBackendMathMode::FP64_STRICT;
  std::size_t parameter_count = 0;
  std::size_t canonical_source_version = 0;
  std::size_t compute_copy_version      = 0;
  std::size_t master_slot_bytes         = 0;
  std::size_t compute_slot_bytes        = 0;
  std::size_t conversion_workspace_bytes = 0;
  std::size_t diagnostic_bytes          = 0;
  std::uint64_t device_layout_fingerprint = 0;
  PsiFormerDeviceArenaLayout arena;
  PsiFormerParameterConversionSchedule conversion;
  bool full_precision_retry_available = false;
  std::uint64_t fingerprint = 0;
};

/** Track two failure-atomic FP64-master/FP32-compute slot pairs.
 *
 * Runtime events must be observed before calling the corresponding milestone method.
 * The class itself is intentionally host-side and externally synchronized; publish()
 * performs one logical switch only after all milestones have completed.
 */
class PsiFormerMixedPublicationState
{
public:
  /// Construct one synchronized active slot pair.
  explicit PsiFormerMixedPublicationState(PsiFormerPrecisionPolicy policy,
                                           std::size_t active_version = 0,
                                           std::uint64_t active_fingerprint = 1,
                                           std::uint8_t active_slot = 0);

  /// Reserve the inactive slot pair for a strictly newer complete model.
  void begin(std::size_t source_version, std::uint64_t execution_fingerprint);

  /// Record completion of the host/canonical-to-device FP64 master copy.
  void markMasterCopied(std::size_t source_version);

  /// Record completion of every FP64-to-FP32 conversion tile.
  void markComputeConversionComplete(std::size_t source_version);

  /** Validate bounded conversion evidence.
   *
   * A hard hazard cancels the staged publication and returns false, preserving the
   * old active slot pair.  Benign underflow evidence is retained and returns true.
   */
  bool acceptDiagnostics(std::size_t source_version,
                         const PsiFormerParameterConversionDiagnostics& diagnostics);

  /// Atomically expose the complete inactive slot pair as the new active model.
  void publish(std::size_t source_version);

  /// Discard a matching in-flight publication at any milestone.
  void cancel(std::size_t source_version);

  std::size_t activeVersion() const noexcept { return active_version_; }
  std::uint64_t activeFingerprint() const noexcept { return active_fingerprint_; }
  std::uint8_t activeSlot() const noexcept { return active_slot_; }
  std::optional<std::size_t> pendingVersion() const noexcept;
  std::optional<std::uint8_t> pendingSlot() const noexcept;
  bool publicationPending() const noexcept { return pending_.has_value(); }

  /// Return whether an exact version/plan pair is safe for evaluation.
  bool isReady(std::size_t version, std::uint64_t fingerprint) const noexcept
  {
    return version == active_version_ && fingerprint == active_fingerprint_;
  }

private:
  struct Pending
  {
    std::size_t version = 0;
    std::uint64_t fingerprint = 0;
    std::uint8_t slot = 0;
    bool master_copied = false;
    bool compute_converted = false;
    bool diagnostics_accepted = false;
  };

  Pending& matchingPending(std::size_t source_version);

  PsiFormerPrecisionPolicy policy_;
  std::size_t active_version_ = 0;
  std::uint64_t active_fingerprint_ = 1;
  std::uint8_t active_slot_ = 0;
  std::optional<Pending> pending_;
};

/// Construct one exact two-slot mixed-precision arena and conversion schedule.
PsiFormerPrecisionExecutionPlan makePsiFormerPrecisionExecutionPlan(
    PsiFormerPrecisionPolicy policy,
    PsiFormerBackendMathMode math_mode,
    std::size_t parameter_count,
    std::size_t cast_tile_parameters,
    std::size_t block_size,
    std::uint64_t device_layout_fingerprint,
    std::size_t source_version,
    bool full_precision_retry_available);

/// Convert one host tile and accumulate the same diagnostics required by the device wrapper.
void convertPsiFormerFp64ToFp32Reference(const double* source,
                                         std::size_t count,
                                         float* destination,
                                         PsiFormerParameterConversionDiagnostics& diagnostics);

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_PRECISION_EXECUTION_H
