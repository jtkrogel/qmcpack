//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_CANONICAL_HEAD_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_CANONICAL_HEAD_H
#include "PsiFormerDeviceCanonicalHead.h"
#include "PsiFormerDeviceComplexValue.h"
#include "PsiFormerDeviceInverseCertificate.h"

namespace qmcplusplus::psiformer::device
{
/** Same-image FP64 paired CORE head; the native Bloch carrier is applied separately. */
class ComplexCanonicalHeadView
{
public:
  /// A default borrow has no numerical or publication authority.
  ComplexCanonicalHeadView() = default;

  /// Admit only the exact live model, queue, shape and producer coordinate generation.
  void validate(const std::shared_ptr<const PreparedDeviceModel>& image, std::size_t version,
      const void* queue, std::size_t count, std::size_t electrons, std::size_t determinants) const
  {
    identity_.validate(image, version, queue, count, electrons, determinants);
    if (!inverse_ || !channels_ || !amplitudes_)
      throw std::invalid_argument("Canonical paired head has incomplete coefficient storage");
  }

  /// Retain generation and immutable-image authority with all coefficient addresses.
  const CanonicalHeadIdentity& identity() const noexcept { return identity_; }

  /// Supply the sole coordinate population admitted by this head.
  const PackedDeviceCoordinatesView& coordinates() const noexcept { return identity_.coordinates(); }

  /// Borrow actual unscaled row-major inverses satisfying A*B=I.
  const cuDoubleComplex* inverses() const noexcept { return inverse_; }

  /// Borrow canonical determinant phases, magnitudes and regularity metadata.
  const ComplexValueChannel* channels() const noexcept { return channels_; }

  /// Borrow the physical CORE amplitude with cusp included once.
  const ComplexValueResult* amplitudes() const noexcept { return amplitudes_; }
private:
  friend class PreparedDeviceCanonicalComplexHead;
  CanonicalHeadIdentity identity_;
  const cuDoubleComplex* inverse_ = nullptr;
  const ComplexValueChannel* channels_ = nullptr;
  const ComplexValueResult* amplitudes_ = nullptr;
};

/** Cold full-FP64 VALUE and inverse owner, independent of mixed derivative tapes. */
class PreparedDeviceCanonicalComplexHead
{
public:
  /// Report complete numeric ownership before any allocation.
  struct Cost
  {
    std::size_t device_bytes, host_bytes;
  };
  /// Count the inverse-only VALUE graph, owned R snapshot and conditioning records.
  static Cost required(const PsiFormerExecutionPlan& plan, std::size_t capacity)
  {
    const auto graph = estimateComplexCanonicalValueCost(plan, capacity);
    const auto mul = [](auto a, auto b) {
      return spatial_detail::checkedProduct(a, b, "Canonical paired head extent overflow");
    };
    const auto add = [](auto a, auto b) {
      return spatial_detail::checkedAdd(a, b, "Canonical paired head bytes overflow");
    };
    const auto channels = mul(capacity, plan.modelShape().determinants);
    auto device = add(graph.device_bytes, checkedDeviceBytes<double>(channels));
    device = add(device, checkedDeviceBytes<double>(mul(mul(3, capacity), plan.modelShape().electrons())));
    device = add(device, checkedDeviceBytes<inverse_certificate::Result>(channels));
    auto host = graph.host_bytes;
    for (auto bytes : {checkedDeviceBytes<ComplexValueChannel>(channels), checkedDeviceBytes<double>(channels),
         checkedDeviceBytes<ComplexValueResult>(capacity), checkedDeviceBytes<inverse_certificate::Result>(channels)})
      host = add(host, bytes);
    return {device, host};
  }

  /// Reject malformed extents and insufficient finite budgets before preparing numeric storage.
  static std::unique_ptr<PreparedDeviceCanonicalComplexHead> prepare(
      std::shared_ptr<const PreparedDeviceModel> image, std::size_t capacity,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, std::size_t budget)
  {
    if (!image)
      throw std::invalid_argument("Canonical paired head image is absent");
    const auto cost = required(image->plan(), capacity);
    if (!budget || budget == std::numeric_limits<std::size_t>::max() || cost.device_bytes > budget)
      throw std::length_error("Canonical paired head exceeds its explicit owner admission");
    return std::unique_ptr<PreparedDeviceCanonicalComplexHead>(
        new PreparedDeviceCanonicalComplexHead(std::move(image), capacity, queue, blas, cost));
  }

  /// Revoke authority while outstanding identities keep their numeric storage alive.
  ~PreparedDeviceCanonicalComplexHead()
  {
    state_->ready = false;
    state_->alive = false;
  }

  /// Upload one population and admit its complete canonical inverse head.
  ComplexCanonicalHeadView evaluate(const std::vector<double>& positions)
  {
    invalidate();
    storage_->coordinates->upload(positions);
    return evaluateCurrent(storage_->coordinates->view(0, positions.size() / (3 * state_->electrons)));
  }

  /// Import the current population without a HOST coordinate readback.
  ComplexCanonicalHeadView evaluateDeviceCoordinates(const PackedDeviceCoordinatesView& positions)
  {
    invalidate();
    storage_->coordinates->copyFrom(positions, queue_);
    return evaluateCurrent(storage_->coordinates->view(0, positions.batchCount()));
  }

  /// Scalar MC/ECP proposals use this same FP64 primal, including valid nodes, without minting inverse authority.
  void evaluateScalar(const std::vector<double>& positions)
  {
    invalidate();
    storage_->coordinates->upload(positions);
    const auto count = positions.size() / (3 * state_->electrons);
    storage_->graph->evaluateCanonicalDeviceCoordinates(storage_->coordinates->view(0, count), nullptr, true);
    state_->count = count;
    scalar_ready_ = true;
  }

  /// Copy only the most recent scalar publication; a later head request revokes it.
  void downloadScalar(std::vector<ComplexValueResult>& amplitudes) const
  {
    if (!scalar_ready_ || !state_->count || amplitudes.size() != state_->count)
      throw std::invalid_argument("Canonical paired scalar output is unavailable or has another extent");
    storage_->graph->download(storage_->graph->resultView(), amplitudes);
  }

  /// Mint a borrow only after all used canonical coefficients pass admission.
  ComplexCanonicalHeadView view() const
  {
    ComplexCanonicalHeadView result;
    result.identity_ = CanonicalHeadIdentity(state_);
    result.inverse_ = storage_->graph->inverse_.data();
    result.channels_ = storage_->graph->channels_.data();
    result.amplitudes_ = storage_->graph->results_.data();
    result.validate(state_->image, state_->model_version, &queue_, state_->count, state_->electrons, state_->determinants);
    return result;
  }

  /// Preflight a successor without invalidating the accepted state.
  void validateModelRebind(const std::shared_ptr<const PreparedDeviceModel>& successor) const
  {
    if (state_.use_count() != 1)
      throw std::logic_error("Canonical paired head cannot rebind with an outstanding borrow");
    if (state_->generation == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Canonical paired head generation exhausted");
    storage_->graph->validateModelRebind(successor);
  }

  /// Adopt only after all composing leaves preflight under exclusive publication authority.
  void adoptPreparedModel(const std::shared_ptr<const PreparedDeviceModel>& successor) noexcept
  {
    state_->ready = scalar_ready_ = false;
    ++state_->generation;
    storage_->graph->adoptPreparedModel(successor);
    state_->image = successor;
    state_->model_version = successor->version();
    state_->count = 0;
  }

  /// Preserve the ordinary preflight/adoption convenience operation.
  void rebindModel(const std::shared_ptr<const PreparedDeviceModel>& successor)
  {
    validateModelRebind(successor);
    adoptPreparedModel(successor);
  }

  /// Count the checked actual graph inventory and separately owned coordinates/pivots.
  std::size_t deviceWorkspaceBytes() const
  {
    return spatial_detail::checkedAdd(graph_cost_.device_bytes,
        spatial_detail::checkedAdd(storage_->coordinates->deviceWorkspaceBytes(),
            spatial_detail::checkedAdd(checkedDeviceBytes<double>(storage_->pivots.capacity()),
                checkedDeviceBytes<inverse_certificate::Result>(storage_->inverse_quality.capacity()),
                "Canonical paired certificate bytes overflow"), "Canonical paired device bytes overflow"),
        "Canonical paired device bytes overflow");
  }

  /// Count numeric HOST staging separately from shared metadata and process RSS.
  std::size_t hostWorkspaceBytes() const
  {
    auto bytes = graph_cost_.host_bytes;
    for (const auto amount : {checkedDeviceBytes<double>(storage_->host_pivots.capacity()),
         checkedDeviceBytes<ComplexValueResult>(storage_->amplitudes.capacity()),
         checkedDeviceBytes<ComplexValueChannel>(storage_->channels.capacity()),
         checkedDeviceBytes<inverse_certificate::Result>(storage_->host_inverse_quality.capacity())})
      bytes = spatial_detail::checkedAdd(bytes, amount, "Canonical paired HOST bytes overflow");
    return bytes;
  }

  /// Expose immutable model identity for composing publication checks.
  const std::shared_ptr<const PreparedDeviceModel>& modelImage() const noexcept { return state_->image; }

  /// Report cold capacity independently of the latest live population.
  std::size_t capacity() const noexcept { return capacity_; }
private:
  /// Retain every cold numeric allocation through the lifetime of issued identities.
  struct Storage
  {
    std::unique_ptr<PreparedDeviceComplexValue> graph;
    std::unique_ptr<PreparedDeviceCoordinates> coordinates;
    DeviceBuffer<double> pivots;
    std::vector<double> host_pivots;
    std::vector<ComplexValueResult> amplitudes;
    std::vector<ComplexValueChannel> channels;
    DeviceBuffer<inverse_certificate::Result> inverse_quality;
    std::vector<inverse_certificate::Result> host_inverse_quality;
  };

  /// Construct exactly the complete admitted owner, with no retained neural reverse tape.
  PreparedDeviceCanonicalComplexHead(std::shared_ptr<const PreparedDeviceModel> image, std::size_t capacity,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, Cost cost)
      : queue_(queue), capacity_(capacity), graph_cost_(estimateComplexCanonicalValueCost(image->plan(), capacity)),
        storage_(std::make_shared<Storage>()), state_(std::shared_ptr<CanonicalHeadState>(new CanonicalHeadState))
  {
    const auto& shape = image->plan().modelShape();
    storage_->coordinates = image->prepareCoordinates(capacity, checkedDeviceBytes<double>(3 * shape.electrons() * capacity));
    storage_->graph.reset(new PreparedDeviceComplexValue(image, capacity, queue, blas, graph_cost_,
        false, true, false, false, false, false, true));
    const auto channels = spatial_detail::checkedProduct(capacity, shape.determinants,
        "Canonical paired certificate extent overflow");
    storage_->pivots.reserveNamed(channels, "CanonicalComplexHead::pivots", __FILE__, __LINE__);
    storage_->host_pivots.resize(channels);
    storage_->amplitudes.resize(capacity);
    storage_->channels.resize(channels);
    storage_->inverse_quality.reserveNamed(channels, "CanonicalComplexHead::inverse_quality", __FILE__, __LINE__);
    storage_->host_inverse_quality.resize(channels);
    state_->image = std::move(image);
    state_->storage = storage_;
    state_->queue = &queue;
    state_->model_version = state_->image->version();
    state_->electrons = shape.electrons();
    state_->determinants = shape.determinants;
    if (deviceWorkspaceBytes() != cost.device_bytes || hostWorkspaceBytes() != cost.host_bytes)
      throw std::logic_error("Canonical paired head constructor differs from cold inventory");
  }

  /// Revoke scalar and inverse authority before every replacement, including failed input.
  void invalidate()
  {
    state_->ready = scalar_ready_ = false;
    state_->count = 0;
    if (state_->generation == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Canonical paired head generation exhausted");
    ++state_->generation;
  }

  /// Certify the reconstructed FP64 A/inverse while retaining finite factors and the unchanged cancellation gate.
  ComplexCanonicalHeadView evaluateCurrent(const PackedDeviceCoordinatesView& coordinates)
  {
    storage_->graph->evaluateCanonicalDeviceCoordinates(coordinates, storage_->pivots.data());
    const auto count = coordinates.batchCount();
    const auto channels = spatial_detail::checkedProduct(count, state_->determinants,
        "Canonical paired live certificate extent overflow");
    storage_->pivots.download(storage_->host_pivots.data(), channels);
    storage_->graph->results_.download(storage_->amplitudes.data(), count);
    storage_->graph->channels_.download(storage_->channels.data(), channels);
    std::size_t bad_factors = 0, bad_inverse = 0, small_pivots = 0, bad_sums = 0;
    bool inverse_checked = false;
    double worst_inverse_rho = 0.0, worst_inverse_relative = 0.0;
    for (std::size_t i = 0; i < channels; ++i)
    {
      const auto& channel = storage_->channels[i];
      if (channel.status || !device_math::isFiniteBinary64(channel.log_abs) ||
          !device_math::isFiniteBinary64(cuCreal(channel.phase)) ||
          !device_math::isFiniteBinary64(cuCimag(channel.phase)) ||
          !device_math::isFiniteBinary64(storage_->host_pivots[i]))
        ++bad_factors;
      if (storage_->host_pivots[i] < 1.0e-3) ++small_pivots; // Diagnostic only for this full-FP64 producer.
    }
    for (std::size_t i = 0; i < count; ++i)
    {
      const auto& a = storage_->amplitudes[i];
      if (a.status || !device_math::isFiniteBinary64(a.log_abs) ||
          !device_math::isFiniteBinary64(cuCreal(a.phase)) || !device_math::isFiniteBinary64(cuCimag(a.phase)) ||
          !device_math::isFiniteBinary64(a.cancellation_ratio) || a.cancellation_ratio < 1.0e-3)
        ++bad_sums;
    }
    const auto require_admitted = [&] {
      if (bad_factors || bad_inverse || bad_sums)
      {
        std::ostringstream reason;
        reason.precision(std::numeric_limits<double>::max_digits10);
        reason << "Canonical paired FP64 head requires a complete physical FP64 retry; invalid_factor=" << bad_factors
               << " uncertified_fp64_inverse=" << bad_inverse << " small_determinant_pivot_diagnostic="
               << small_pivots << " severe_cancellation=" << bad_sums << " ill_conditioned_spatial=0"
               << " inverse_certificate_checked=" << inverse_checked << " inverse_rho_upper_max="
               << worst_inverse_rho << " relative_inverse_error_upper_max=" << worst_inverse_relative
               << " relative_inverse_error_budget=" << inverse_certificate::relative_budget;
        throw MixedPrecisionRetryRequired(reason.str().c_str());
      }
    };
    // Invalid factors may leave inverse storage unwritten; never read B on that path.
    require_admitted();
    const auto* matrix = storage_->graph->repackCanonicalMatrixForCertificate();
    checkRuntime(inverse_certificate::launchComplex(queue_.getNative(), matrix, storage_->graph->inverse_.data(),
        channels, state_->electrons, storage_->inverse_quality.data()), "canonical paired inverse certificate");
    storage_->inverse_quality.downloadOnStream(storage_->host_inverse_quality.data(), channels, queue_.getNative());
    inverse_checked = true;
    for (std::size_t i = 0; i < channels; ++i)
    {
      const auto& quality = storage_->host_inverse_quality[i];
      if (!device_math::isFiniteBinary64(quality.rho_upper) || quality.rho_upper > worst_inverse_rho)
        worst_inverse_rho = quality.rho_upper;
      if (!device_math::isFiniteBinary64(quality.relative_inverse_error_upper) ||
          quality.relative_inverse_error_upper > worst_inverse_relative)
        worst_inverse_relative = quality.relative_inverse_error_upper;
      if (!inverse_certificate::isCertified(quality)) ++bad_inverse;
    }
    require_admitted();
    state_->coordinates = coordinates;
    state_->count = count;
    state_->ready = true;
    return view();
  }
  compute::Queue<PlatformKind::CUDA>& queue_;
  const std::size_t capacity_;
  bool scalar_ready_ = false;
  const ComplexValueCost graph_cost_;
  std::shared_ptr<Storage> storage_;
  std::shared_ptr<CanonicalHeadState> state_;
};
} // namespace qmcplusplus::psiformer::device
#endif
