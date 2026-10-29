//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceMixedValueDense.h
 * @brief Explicit FP32 affine projections between FP64 value-graph islands.
 *
 * Spatial planes use the same selected affine arithmetic as values. This does not
 * qualify parameter derivatives or production VMC. A retry replays the whole graph,
 * never substituting individual GEMMs into already approximated activations.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_VALUE_DENSE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_VALUE_DENSE_H

#include "PsiFormerDeviceDense.h"
#include "PsiFormerDeviceResources.h"
#include "Platforms/CUDA/QueueCUDA.hpp"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceDeterminantMath.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerFoldedDense.h"

#include <memory>
#include <stdexcept>

namespace qmcplusplus::psiformer { class PsiFormerExecutionPlan; }

namespace qmcplusplus::psiformer::device
{

class PreparedDeviceModel;
struct ComplexValueChannel;
struct ComplexValueResult;

/** A bounded mixed request failed numerical admission and needs a whole-graph FP64 retry. */
class MixedPrecisionRetryRequired final : public std::runtime_error
{
public:
  explicit MixedPrecisionRetryRequired(const char* reason) : std::runtime_error(reason) {}
};

/** Immutable shared FP32 copies of affine weights, retaining their canonical FP64 image. */
class PreparedDeviceMixedDenseParameters
{
public:
  /// Convert only typed embedding/attention/MLP matrices and admit diagnostics before use.
  PreparedDeviceMixedDenseParameters(std::shared_ptr<const PreparedDeviceModel> model,
                                     compute::Queue<PlatformKind::CUDA>& queue,
                                     bool allow_fp64_retry = false);

  /// A paired numerical owner explicitly authorizes only its shared real affine trunk.
  PreparedDeviceMixedDenseParameters(std::shared_ptr<const PreparedDeviceModel> model,
                                     compute::Queue<PlatformKind::CUDA>& queue,
                                     MixedTrunkHeadPolicy policy, bool allow_fp64_retry = false);

  /// Derive immutable family authority from the admitted image, not a mutable flag.
  MixedTrunkHeadPolicy trunkHeadPolicy() const noexcept;

  /// An inadmissible conversion remains versioned but may only execute through FP64 replay.
  bool usable() const noexcept { return usable_; }

  /// Return the exact immutable canonical image represented by the compute copy.
  const std::shared_ptr<const PreparedDeviceModel>& modelImage() const noexcept { return model_; }

  /// Expose read-only canonical-offset FP32 weights; sensitive parameter slots are unused.
  const float* parameters() const noexcept { return weights_.data(); }

  /// Return conversion evidence collected during this immutable cache's preparation.
  const PsiFormerParameterConversionDiagnostics& conversionDiagnostics() const noexcept { return conversion_; }

  /// Return persistent compute-copy bytes, excluding its shared FP64 model owner.
  std::size_t deviceStorageBytes() const;

private:
  std::shared_ptr<const PreparedDeviceModel> model_;
  DeviceBuffer<float> weights_;
  PsiFormerParameterConversionDiagnostics conversion_{};
  bool usable_ = true;
};

/** Logical cast traffic for one attempted value evaluation, not vendor memory traffic. */
struct MixedValueDenseStatistics
{
  std::size_t projections = 0;
  std::size_t input_cast_elements = 0;
  std::size_t output_cast_elements = 0;

  /// Count FP64-read/FP32-write plus FP32-read/FP64-write bytes without overflow.
  std::size_t roundTripBytes() const;
};

/** Prepared single-queue FP32 affine value/spatial planes between FP64 boundaries. */
class PreparedDeviceMixedValueDense
{
public:
  /// Prepare bounded input/output casts once; a shared immutable cache avoids per-crowd shadows.
  PreparedDeviceMixedValueDense(std::shared_ptr<const PreparedDeviceMixedDenseParameters> parameters,
                                std::size_t maximum_rows,
                                compute::Queue<PlatformKind::CUDA>& queue,
                                AcceleratorBlasHandle& blas);

  /// Bind this attempt to exactly the graph's canonical image and clear sticky diagnostics.
  void beginEvaluation(const std::shared_ptr<const PreparedDeviceModel>& image);

  /// Cast, multiply in strict FP32, and promote one compact value-only affine projection.
  void project(const DenseForwardLayout& layout, const double* source, const double* weight, double* target);

  /// Apply identical mixed affine arithmetic to every value/full-VGL plane (casts differentiate as identity).
  void projectSpatial(const SpatialDenseJetLayout& layout,
                       const double* source, const double* weight, double* target);

  /// Return prepared folded-row capacity, including all spatial planes when requested.
  std::size_t maximumRows() const noexcept { return maximum_rows_; }

  /// Record conservative near-singular/cancellation hazards while metadata stays on device.
  void checkValueConditioning(const device_determinant::FactorizationMetadata* factors,
                              const device_determinant::CombinationMetadata* combinations,
                              std::size_t configurations, std::size_t determinants);

  /// Screen typed paired metadata without reinterpreting REAL determinant records.
  void checkComplexValueConditioning(const double* minimum_scaled_pivots,
                                    const ComplexValueChannel* channels, const ComplexValueResult* results,
                                    std::size_t configurations, std::size_t determinants);

  /// Admit cast/activation diagnostics before a paired FP64 head; keep the attempt open.
  void admitAffinePrefix();

  /// Admit all casts/GEMMs before any approximate value result becomes visible.
  void finishEvaluation();

  /// Abandon a failed graph attempt; its owner retains responsibility for queue recovery/fencing.
  void cancelEvaluation() noexcept { active_ = false; }

  /// Install a separately prepared same-family successor without allocation or device work.
  void rebindParameters(std::shared_ptr<const PreparedDeviceMixedDenseParameters> successor);
  /// Preflight all leaves before any member of a native publication transaction mutates.
  void validateParameterRebind(const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& successor) const;
  /// Requires completed preflight under the same idle lease; performs no device work.
  void adoptPreparedParameters(const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& successor) noexcept;


  /// Return the canonical image to which this adapter is currently bound.
  const std::shared_ptr<const PreparedDeviceModel>& modelImage() const noexcept;

  /// Share the exact immutable compute copy used by this adapter.
  const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& parameterCache() const noexcept { return parameters_; }

  /// Return diagnostics for the last completed admission, including a rejected attempt.
  const PsiFormerDeviceNumericalDiagnostics& diagnostics() const noexcept { return diagnostics_; }

  /// Return cast counts for the most recent attempted evaluation.
  const MixedValueDenseStatistics& statistics() const noexcept { return statistics_; }

  /// Return graph-local bytes, excluding the separately shared compute-parameter image.
  static std::size_t estimateDeviceWorkspaceBytes(const PsiFormerExecutionPlan& plan, std::size_t maximum_rows);
  std::size_t hostWorkspaceBytes() const noexcept { return sizeof(diagnostics_); }
  std::size_t deviceWorkspaceBytes() const;

private:
  /// Download and validate diagnostics without choosing publication or transaction state.
  void admitDiagnostics();

  std::shared_ptr<const PreparedDeviceMixedDenseParameters> parameters_;
  compute::Queue<PlatformKind::CUDA>& queue_;
  AcceleratorBlasHandle& blas_;
  PsiFormerBlasMathModePlan math_mode_;
  std::size_t maximum_rows_;
  DeviceBuffer<float> source_, target_;
  DeviceBuffer<PsiFormerDeviceNumericalDiagnostics> device_diagnostics_;
  PsiFormerDeviceNumericalDiagnostics diagnostics_{};
  MixedValueDenseStatistics statistics_{};
  bool active_ = false;
};

/** Promoted operands actually consumed by the most recent mixed weight VJP.
 * Unadmitted until finishEvaluation succeeds; callbacks must stage privately.
 * Borrowed on the owner's stream until its next projection or rebind. The bias
 * metric uses the rounded adjoint; the physical bias gradient still sums FP64 A.
 */
struct MixedAffineObservationView
{
  const double* source;
  const double* adjoint;
};

/** Cumulative arithmetic-epoch counters, including rejected private work.
 * Logical conversion bytes exclude FP64 accumulator reads and vendor workspace traffic.
 * Completion admits this adapter's arithmetic only, never a native sample/global sum.
 */
struct MixedAffineDerivativeStatistics
{
  std::size_t attempts = 0, completed = 0, rejected = 0;
  std::size_t fp32_gemms = 0, cast_elements = 0, promoted_elements = 0;

  /// Count logical conversion reads/writes, excluding response accumulation.
  std::size_t logicalTransferBytes() const;
};

/** Prepared algorithmic affine VJP/JVP for one immutable mixed master image.
 * Casts differentiate as identity. Each derivative SGEMM has FP32 operands,
 * accumulation and private output; promotion and response accumulation are FP64.
 * Failed diagnostics require whole-sample replay before publishing any response
 * or observation. This numerical owner never chooses native retry/publication.
 */
class PreparedDeviceMixedAffineDerivatives
{
public:

  /// Allocate the fixed derivative and optional observation capacities once.
  PreparedDeviceMixedAffineDerivatives(
      std::shared_ptr<const PreparedDeviceMixedDenseParameters> parameters,
      std::size_t maximum_rows, compute::Queue<PlatformKind::CUDA>& queue,
      AcceleratorBlasHandle& blas);

  /// Start one same-image private arithmetic epoch.
  void beginEvaluation(const std::shared_ptr<const PreparedDeviceModel>& image);

  /// Reject cast/product hazards before outputs or factor rows become accepted.
  void finishEvaluation();

  /// Revoke an incomplete epoch without changing its image or allocated storage.
  void cancelEvaluation() noexcept
  {
    if (active_)
    {
      active_ = false;
      ++statistics_.rejected;
    }
  }

  /// Rebind an idle owner through the same validate/adopt contract used by publication.
  void rebindParameters(std::shared_ptr<const PreparedDeviceMixedDenseParameters> successor);

  /// Preflight all leaves before any member of a native publication transaction mutates.
  void validateParameterRebind(const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& successor) const;

  /// Requires completed preflight under the same idle lease; performs no device work.
  void adoptPreparedParameters(const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& successor) noexcept;

  /// Return the immutable master image represented by the bound compute cache.
  const std::shared_ptr<const PreparedDeviceModel>& modelImage() const noexcept;

  /// Borrow the exact cache identity used to match a completed primal tape.
  const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& parameters() const noexcept { return parameters_; }

  /// Exact prepared device bytes; immutable FP32/master caches are separately owned.
  static std::size_t estimateDeviceWorkspaceBytes(const PsiFormerExecutionPlan& plan, std::size_t maximum_rows);

  /// Retained numeric diagnostics only; model/cache metadata and counters are excluded.
  std::size_t hostWorkspaceBytes() const noexcept { return sizeof(diagnostics_); }

  /// Read cumulative arithmetic-epoch and logical conversion counters.
  const MixedAffineDerivativeStatistics& statistics() const noexcept { return statistics_; }

  /// Report the fixed folded-row capacity.
  std::size_t maximumRows() const noexcept { return maximum_rows_; }

  /// Count actual device capacities, excluding separately shared parameter images.
  std::size_t deviceWorkspaceBytes() const;

  /// Read the last completed or rejected arithmetic diagnostics.
  const PsiFormerDeviceNumericalDiagnostics& diagnostics() const noexcept { return diagnostics_; }

  /// Accumulate a privately rounded product into a compact FP64 input response.
  void accumulateInputAdjoint(const DenseForwardLayout& layout, const double* adjoint,
      const double* canonical_weight, double* input_response);

  /// Accumulate the weight response; only an explicitly requested pending sink gets operands.
  MixedAffineObservationView accumulateWeightAdjoint(const DenseForwardLayout& layout,
      const double* source, const double* adjoint, const double* canonical_weight, double* weight_response,
      bool materialize_observations = false);

  /// Form dX W + X dW; null source_direction denotes the fixed-geometry zero tangent.
  void projectDirection(const DenseForwardLayout& layout, const double* source,
      const double* source_direction, const double* canonical_weight,
      const double* weight_direction, double* target);

private:

  /// Admit the typed canonical matrix and the compact prepared row extents.
  const float* validateProjection(const DenseForwardLayout& layout, const double* canonical_weight) const;

  /// Convert only the logical source operands and retain any numerical hazard.
  void cast(const double* source, float* target, std::size_t rows, std::size_t width);

  /// Cross back to the FP64 response boundary with optional accumulation.
  void promote(const float* source, double* target, std::size_t count, bool accumulate);

  std::shared_ptr<const PreparedDeviceMixedDenseParameters> parameters_;
  std::size_t maximum_rows_, maximum_input_, width_;
  compute::Queue<PlatformKind::CUDA>& queue_;
  AcceleratorBlasHandle& blas_;
  PsiFormerBlasMathModePlan math_mode_;
  DeviceBuffer<float> source_, adjoint_, product_, direction_;
  DeviceBuffer<double> observation_source_, observation_adjoint_;
  DeviceBuffer<PsiFormerDeviceNumericalDiagnostics> device_diagnostics_;
  PsiFormerDeviceNumericalDiagnostics diagnostics_{};
  MixedAffineDerivativeStatistics statistics_{};
  bool active_ = false;
};

/// Cancel private arithmetic state on every exception; publication remains external.
class MixedAffineDerivativeAttempt
{
public:

  /// Open only the selected mixed arithmetic epoch.
  MixedAffineDerivativeAttempt(PreparedDeviceMixedAffineDerivatives* adapter,
      const std::shared_ptr<const PreparedDeviceModel>& image) : adapter_(adapter)
  {
    if (adapter_)
      adapter_->beginEvaluation(image);
  }

  /// Cancel an unadmitted epoch on every exceptional owner exit.
  ~MixedAffineDerivativeAttempt()
  {
    if (adapter_)
      adapter_->cancelEvaluation();
  }

  /// Admit the arithmetic before releasing this guard's cancellation responsibility.
  void finish()
  {
    if (adapter_)
    {
      adapter_->finishEvaluation();
      adapter_ = nullptr;
    }
  }

private:
  PreparedDeviceMixedAffineDerivatives* adapter_;
};

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_VALUE_DENSE_H
