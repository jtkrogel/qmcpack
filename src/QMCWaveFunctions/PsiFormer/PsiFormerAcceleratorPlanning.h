//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerAcceleratorPlanning.h
 * @brief Backend-neutral layout and publication contracts for PsiFormer accelerators.
 *
 * This layer is intentionally free of CUDA, HIP, SYCL, and OpenMP runtime types.
 * CPU-only builds can therefore validate every layout and state transition used by
 * later accelerator executors.  Device pointers, queues, and allocation ownership
 * belong to backend resource classes rather than this immutable planning surface.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_ACCELERATOR_PLANNING_H
#define QMCPLUSPLUS_PSIFORMER_ACCELERATOR_PLANNING_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace qmcplusplus::psiformer
{

/// Identify the execution backend without exposing vendor runtime types.
enum class PsiFormerAcceleratorBackend : std::uint8_t
{
  CPU,
  OPENMP_TARGET,
  CUDA,
  HIP,
  SYCL
};

/// Report accelerator features compiled into the current executable.
struct PsiFormerCompiledAcceleratorSupport
{
  bool openmp_target = false;
  bool cuda          = false;
  bool hip           = false;
  bool sycl          = false;

  /// Return whether one explicitly requested backend is available.
  bool supports(PsiFormerAcceleratorBackend backend) const noexcept;
};

/** Fixed-rank device copy of one canonical parameter tensor descriptor.
 *
 * Current PsiFormer leaves are scalars, vectors, or matrices.  A fixed extent array
 * avoids a device-side graph of independently allocated shape vectors.
 */
struct PsiFormerDeviceTensorDescriptor
{
  ParameterRole role = ParameterRole::COUNT;
  std::size_t attention_block = NO_ATTENTION_BLOCK;
  std::uint8_t rank = 0;
  std::array<std::size_t, 2> extents{};
  std::size_t begin = 0;
  std::size_t end   = 0;

  /// Return the canonical number of scalar values in this tensor.
  std::size_t size() const noexcept { return end - begin; }

  friend bool operator==(const PsiFormerDeviceTensorDescriptor& lhs,
                         const PsiFormerDeviceTensorDescriptor& rhs) noexcept
  {
    return lhs.role == rhs.role && lhs.attention_block == rhs.attention_block &&
        lhs.rank == rhs.rank && lhs.extents == rhs.extents &&
        lhs.begin == rhs.begin && lhs.end == rhs.end;
  }
};

/** Pointer-free accelerator layout derived from an immutable execution plan.
 *
 * The descriptor vector is host-owned preparation metadata.  Its contiguous payload
 * may be copied to a device once; descriptors contain only values and canonical flat
 * offsets, never host addresses.
 */
struct PsiFormerDeviceLayout
{
  ModelShape model_shape;
  BoundaryCondition boundary = BoundaryCondition::OPEN;
  GeometryFeaturePolicy geometry_feature_policy = GeometryFeaturePolicy::OPEN_EUCLIDEAN_V1;
  ScalarDomain parameter_scalar_domain = ScalarDomain::REAL;
  ScalarDomain compute_scalar_domain   = ScalarDomain::REAL;
  ScalarDomain amplitude_scalar_domain = ScalarDomain::REAL;
  std::size_t parameter_count = 0;
  std::vector<PsiFormerDeviceTensorDescriptor> tensors;
  std::uint64_t fingerprint = 0;
};

/** Track one failure-atomic asynchronous parameter publication.
 *
 * A backend starts a copy, records its completion outside this class, and calls
 * completePublication only after the corresponding event has completed.  Evaluation
 * may use activeVersion throughout a pending copy; the staged version is never
 * exposed as active prematurely.
 */
class PsiFormerDevicePublicationState
{
public:
  /// Construct a synchronized device mirror at the supplied model version.
  explicit PsiFormerDevicePublicationState(std::size_t active_version = 0) noexcept
      : active_version_(active_version)
  {}

  /// Begin copying a strictly newer complete canonical model.
  void beginPublication(std::size_t source_version);

  /// Publish the staged model after its asynchronous copy has completed.
  void completePublication(std::size_t source_version);

  /// Discard a failed staged copy without modifying the active model.
  void cancelPublication(std::size_t source_version);

  /// Return the version safe for evaluation.
  std::size_t activeVersion() const noexcept { return active_version_; }

  /// Return the in-flight source version, if a copy is pending.
  std::optional<std::size_t> pendingVersion() const noexcept { return pending_version_; }

  /// Return whether a copy is currently in flight.
  bool publicationPending() const noexcept { return pending_version_.has_value(); }

  /// Return whether the device mirror can evaluate the requested exact version.
  bool isReady(std::size_t requested_version) const noexcept
  {
    return active_version_ == requested_version;
  }

private:
  std::size_t active_version_ = 0;
  std::optional<std::size_t> pending_version_;
};

/// Return build-time accelerator support without probing runtime devices.
PsiFormerCompiledAcceleratorSupport compiledPsiFormerAcceleratorSupport() noexcept;

/// Return the stable input and diagnostic spelling for one backend.
const char* psiFormerAcceleratorBackendName(PsiFormerAcceleratorBackend backend) noexcept;

/** Parse an explicit backend request and reject unavailable accelerator paths.
 *
 * Empty, ``no``, and ``cpu`` select CPU.  ``yes`` and ``auto`` select the preferred
 * compiled accelerator in CUDA/HIP, SYCL, then OpenMP-target order, falling back to
 * CPU only for ``auto``.  A strict ``yes`` request fails when no accelerator exists.
 */
PsiFormerAcceleratorBackend selectPsiFormerAcceleratorBackend(
    std::string_view request,
    PsiFormerCompiledAcceleratorSupport support = compiledPsiFormerAcceleratorSupport());

/// Translate validated host metadata into the pointer-free accelerator layout.
PsiFormerDeviceLayout makePsiFormerDeviceLayout(const PsiFormerExecutionPlan& plan);

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_ACCELERATOR_PLANNING_H
