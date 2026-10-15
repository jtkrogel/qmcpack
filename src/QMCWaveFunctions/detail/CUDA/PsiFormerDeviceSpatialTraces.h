//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceSpatialTraces.h
 * @brief Bounded parallel compensated determinant spatial traces.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_SPATIAL_TRACES_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_SPATIAL_TRACES_H

#include "PsiFormerDeviceResources.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceDeterminantMath.h"

#include <vector>

namespace qmcplusplus::psiformer::device
{

/** Borrow packed matrix planes with optional configuration gaps.
 * Physical matrix size N is independent of the local derivative target count K.
 * Input allocations must cover layout's checked spans and outlive the stream.
 */
struct SpatialTraceInputs
{
  const device_determinant::FactorizationMetadata* factorizations = nullptr;
  const device_determinant::CombinationMetadata* combinations = nullptr;
  const double* normalized_weights = nullptr;
  const double* inverses = nullptr;
  const double* gradients = nullptr;
  const double* laplacians = nullptr;
};

/** Explicit device buffer charge; input/model and vendor allocations are excluded.
 * The custom product uses no BLAS workspace and no dynamic host staging.
 */
struct SpatialTraceWorkspacePlan
{
  device_determinant::SpatialLayout capacity;
  std::size_t product_slots = 0;
  std::size_t product_bytes = 0;
  std::size_t metadata_bytes = 0;
  std::size_t total_device_bytes = 0;
  std::size_t shared_bytes_per_block = 0;
};

/// Check every extent and the complete owned-buffer budget without a GPU call.
SpatialTraceWorkspacePlan planSpatialTraceWorkspace(
    device_determinant::SpatialLayout capacity, std::size_t budget_bytes = 8 * 1024 * 1024);

/** Borrow one submitted result, valid until replacement/invalidation/destruction.
 * Outputs are tightly packed [B,3K] and [B,K]. Completion remains ordered on the
 * supplied stream; callers must admit typed configuration statuses before use.
 */
struct SpatialTraceDeviceResult
{
  const double* log_gradient = nullptr;
  const double* lap_ratio = nullptr;
  const double* lap_log = nullptr;
  const device_determinant::DerivativeStatus* status = nullptr;
  device_determinant::SpatialLayout layout;
  std::size_t generation = 0;
};

/** Reuse a fixed product microbatch independently of forward target tile size.
 * The caller keeps the owning device current, holds the outer exclusive lease
 * and keeps the borrowed stream alive through destruction. This numerical leaf
 * does not own a model image, publication or coordinate-admission authority.
 */
class PreparedDeviceSpatialTraces
{
public:
  PreparedDeviceSpatialTraces(device_determinant::SpatialLayout capacity, Stream stream,
                               std::size_t budget_bytes = 8 * 1024 * 1024);

  ~PreparedDeviceSpatialTraces();

  PreparedDeviceSpatialTraces(const PreparedDeviceSpatialTraces&) = delete;
  PreparedDeviceSpatialTraces& operator=(const PreparedDeviceSpatialTraces&) = delete;

  /// Enqueue a complete request. Any rejected replacement revokes prior results.
  SpatialTraceDeviceResult evaluate(device_determinant::SpatialLayout layout, SpatialTraceInputs inputs);

  /// Revoke borrowed result admission without allocating or changing capacity.
  void invalidate() noexcept { submitted_ = false; }

  /// Return only this evaluator's most recently submitted generation.
  SpatialTraceDeviceResult result() const;

  /// Synchronize, then copy defined outputs and typed statuses for every target.
  /// Numerical failure returns a failure status and zero configuration outputs.
  void downloadResults(std::vector<double>& gradient, std::vector<double>& lap_ratio,
                        std::vector<double>& lap_log,
                        std::vector<device_determinant::DerivativeStatus>& status);

  const SpatialTraceWorkspacePlan& workspacePlan() const noexcept { return plan_; }

  std::size_t deviceWorkspaceBytes() const noexcept { return plan_.total_device_bytes; }

  std::size_t hostWorkspaceBytes() const noexcept { return 0; }

private:
  void validateCurrentDevice() const;

  SpatialTraceWorkspacePlan plan_;
  Stream stream_;
  int device_ordinal_ = -1;
  // Stats store [gradient xyz, square-trace xyz, Laplacian] per [B,K,D].
  DeviceBuffer<double> products_, gradient_, lap_ratio_, lap_log_, stats_;
  DeviceBuffer<unsigned int> failure_;
  DeviceBuffer<device_determinant::DerivativeStatus> status_;
  device_determinant::SpatialLayout live_{};
  std::size_t generation_ = 0;
  bool submitted_ = false;
};

} // namespace qmcplusplus::psiformer::device

#endif
