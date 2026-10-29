//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_CANONICAL_HEAD_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_CANONICAL_HEAD_H

#include "PsiFormerDeviceCoordinates.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceDeterminantMath.h"

namespace qmcplusplus::psiformer::device
{
class PreparedDeviceCanonicalRealHead;
class PreparedDeviceCanonicalComplexHead;

/** Retain cold storage while allowing its producer to revoke every issued view.
 * Existing parent execution permits serialize mutation and consumption; this is
 * provenance, not an additional concurrent scheduling primitive.
 */
class CanonicalHeadState
{
private:
  friend class PreparedDeviceCanonicalRealHead;
  friend class PreparedDeviceCanonicalComplexHead;
  friend class CanonicalHeadIdentity;

  CanonicalHeadState() = default;
  std::shared_ptr<const PreparedDeviceModel> image;
  std::shared_ptr<void> storage;
  PackedDeviceCoordinatesView coordinates;
  const void* queue = nullptr;
  std::size_t model_version = 0, generation = 0, count = 0, electrons = 0, determinants = 0;
  bool ready = false, alive = true;
};

/** Only a complete full-FP64 VALUE producer can issue this same-image/R borrow. */
class CanonicalHeadIdentity
{
public:
  /// A default identity is unusable and fails before numerical work.
  CanonicalHeadIdentity() = default;

  /// Require the exact immutable model, queue, live shape and current producer generation.
  void validate(const std::shared_ptr<const PreparedDeviceModel>& image, std::size_t model_version,
      const void* queue, std::size_t count, std::size_t electrons, std::size_t determinants) const
  {
    if (!state_ || !state_->alive || !state_->ready || state_->generation != generation_ ||
        !state_->storage || state_->image.get() != image.get() || image_.get() != image.get() || state_->model_version != model_version ||
        state_->queue != queue || state_->count != count || state_->electrons != electrons ||
        state_->determinants != determinants || count == 0 ||
        !coordinates_.sameSourceRange(state_->coordinates) ||
        coordinates_.batchCount() != count || coordinates_.physicalElectrons() != electrons)
      throw std::invalid_argument("Canonical FP64 head is stale, foreign, incomplete or on another queue");
  }

  /// Consumers receive this producer-owned coordinate borrow, never an independent R argument.
  const PackedDeviceCoordinatesView& coordinates() const noexcept { return coordinates_; }

  /// Report the exact issued logical population.
  std::size_t batchCount() const noexcept { return count_; }

  /// Identify one producer evaluation independently of model versions.
  std::size_t generation() const noexcept { return generation_; }

  /// Identify storage provenance without granting access through a raw pointer.
  const void* producerIdentity() const noexcept { return state_.get(); }

private:
  friend class PreparedDeviceCanonicalRealHead;
  friend class PreparedDeviceCanonicalComplexHead;

  /// Snapshot only a producer's admitted full-FP64 VALUE state.
  explicit CanonicalHeadIdentity(const std::shared_ptr<CanonicalHeadState>& state)
      : state_(state), image_(state->image), coordinates_(state->coordinates), generation_(state->generation), count_(state->count)
  {}

  std::shared_ptr<const CanonicalHeadState> state_;
  std::shared_ptr<const PreparedDeviceModel> image_;
  PackedDeviceCoordinatesView coordinates_;
  std::size_t generation_ = 0, count_ = 0;
};

/** Borrow REAL determinant coefficients and canonical primal output without copying matrices. */
class RealCanonicalHeadView
{
public:
  /// A default view has no publication or numerical authority.
  RealCanonicalHeadView() = default;

  /// Validate provenance before exposing its complete coefficient set to a consumer.
  void validate(const std::shared_ptr<const PreparedDeviceModel>& image, std::size_t version,
      const void* queue, std::size_t count, std::size_t electrons, std::size_t determinants) const
  {
    identity_.validate(image, version, queue, count, electrons, determinants);
    if (!matrices_ || !inverse_ || !weights_ || !factors_ || !combinations_ || !phase_ || !log_ || !log_stride_)
      throw std::invalid_argument("Canonical REAL head is missing an admitted coefficient field");
  }

  /// Retain provenance together with every coefficient address.
  const CanonicalHeadIdentity& identity() const noexcept { return identity_; }

  /// Supply the only coordinates accepted by canonical-head consumers.
  const PackedDeviceCoordinatesView& coordinates() const noexcept { return identity_.coordinates(); }

  /// Borrow row-major [configuration][determinant][row][column] primal matrices.
  const double* matrices() const noexcept { return matrices_; }

  /// Borrow row-major inverses B satisfying A*B=I.
  const double* inverses() const noexcept { return inverse_; }

  /// Borrow signed determinant weights normalized by the complete canonical sum.
  const double* normalizedWeights() const noexcept { return weights_; }

  /// Borrow original regularity and pivot diagnostics for the coefficients actually used.
  const device_determinant::FactorizationMetadata* factorizations() const noexcept { return factors_; }

  /// Borrow original finite/cancellation metadata for the canonical determinant sum.
  const device_determinant::CombinationMetadata* combinations() const noexcept { return combinations_; }

  /// Borrow the complete canonical physical REAL phase.
  const double* phases() const noexcept { return phase_; }

  /// Borrow canonical log amplitudes, including the physical cusp exactly once.
  const double* logValues() const noexcept { return log_; }

  /// Report the canonical VALUE output's configuration stride.
  std::size_t logStride() const noexcept { return log_stride_; }

private:
  friend class PreparedDeviceCanonicalRealHead;
  CanonicalHeadIdentity identity_;
  const double* matrices_ = nullptr;
  const double* inverse_ = nullptr;
  const double* weights_ = nullptr;
  const device_determinant::FactorizationMetadata* factors_ = nullptr;
  const device_determinant::CombinationMetadata* combinations_ = nullptr;
  const double* phase_ = nullptr;
  const double* log_ = nullptr;
  std::size_t log_stride_ = 0;
};
} // namespace qmcplusplus::psiformer::device
#endif
