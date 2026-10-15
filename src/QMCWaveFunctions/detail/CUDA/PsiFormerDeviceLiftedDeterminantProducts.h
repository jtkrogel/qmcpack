//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceLiftedDeterminantProducts.h
 * @brief Bounded-channel FP64 matrix products for lifted inverse factors.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_LIFTED_DETERMINANT_PRODUCTS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_LIFTED_DETERMINANT_PRODUCTS_H

#include "PsiFormerDeviceDense.h"
#include "PsiFormerDeviceDeterminantKernels.h"
#include "PsiFormerDeviceResources.h"
#include "Platforms/CUDA/QueueCUDA.hpp"

namespace qmcplusplus::psiformer::device
{

/** Checked owned-buffer charge, excluding BLAS library/vendor storage. Only
 * channel slots grow scratch; target count and caller-owned factor outputs
 * never become additional retained storage.
 */
struct LiftedDeterminantProductPlan
{
  std::size_t matrix_size, channel_slots;
  std::size_t matrix_elements, matrix_storage_elements;
  std::size_t device_workspace_bytes;
};

/// Compute every owned byte and reject an insufficient explicit budget before allocation.
LiftedDeterminantProductPlan planLiftedDeterminantProducts(std::size_t matrix_size,
    std::size_t channel_slots, std::size_t budget_bytes);

/** Borrow one complete configuration of ordinary orbital and determinant jets.
 * The layout distinguishes physical N from local targets K. Its matrix planes
 * are packed [plane,D,N,N], with optional plane padding; roots have one element.
 * All inputs and the output remain alive through the synchronous evaluation.
 */
struct LiftedDeterminantProductInputs
{
  LiftedDeterminantLayout layout;
  const device_determinant::FactorizationMetadata* factorizations;
  const device_determinant::CombinationMetadata* combination;
  const double* weights;
  const double* inverses;
  const double* orbitals;
  const double* log_gradient;
  const double* log_laplacian;
};

/** Reuse a small channel microbatch for fourteen products per target/channel.
 * This optional numerical owner does not alter default kinetic execution. The
 * caller owns factor_jets in the existing admitted layout; no P buffer exists.
 * The reference's ordered (I*G)*(I*G)*I Laplacian terms remain separate.
 * This leaf masks zero-weight channels like the factor reference. A kinetic
 * caller must still enforce its separate ALL-channel inverse admission first.
 */
class PreparedDeviceLiftedDeterminantProducts
{
public:
  /// Prepare six product matrices per slot, one masked zero matrix and metadata.
  /// The external BLAS handle must be bound to the supplied queue's stream.
  PreparedDeviceLiftedDeterminantProducts(std::size_t matrix_size, std::size_t channel_slots,
      std::size_t budget_bytes, compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas);

  /// Fill all factor planes, returning the reference's typed admission status.
  /// Failure clears the entire logical output; padding is never modified.
  device_determinant::DerivativeStatus evaluate(const LiftedDeterminantProductInputs& input,
      double* factor_jets);

  /// Expose the exact admitted extents and owned numerical storage descriptor.
  const LiftedDeterminantProductPlan& workspacePlan() const noexcept { return plan_; }

  /// Count only explicitly owned device buffers, excluding inputs, output and BLAS.
  std::size_t deviceWorkspaceBytes() const noexcept { return plan_.device_workspace_bytes; }

  /// This owner retains no host numerical staging arrays.
  std::size_t hostWorkspaceBytes() const noexcept { return 0; }

private:
  /// Apply one row-major batch using the existing column-major BLAS adapter.
  void product(std::size_t batches);

  LiftedDeterminantProductPlan plan_;
  compute::Queue<PlatformKind::CUDA>& queue_;
  AcceleratorBlasHandle& blas_;
  int device_ordinal_ = -1;
  DeviceBuffer<double> matrices_, statistics_;
  DeviceBuffer<const double*> left_, right_;
  DeviceBuffer<double*> target_;
  DeviceBuffer<unsigned int> failure_;
  DeviceBuffer<device_determinant::DerivativeStatus> status_;
};

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_LIFTED_DETERMINANT_PRODUCTS_H
