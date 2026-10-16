//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceSrMetric.h
 * @brief Scoped real/open undamped parameter covariance, without a host P product.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_SR_METRIC_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_SR_METRIC_H

#include "PsiFormerDeviceProductSession.h"

namespace qmcplusplus::psiformer::device
{

/** Own fixed-population numerical scratch, not an optimizer or a model image.
 * Host positions are copied in configuration-major interleaved Cartesian order.
 * The session must outlive this owner. Every operation holds its control mutex;
 * no device pointer/view escapes. Image publication makes this owner stale.
 */
class PreparedDeviceSrMetric
{
public:
  using FrozenInterval = std::pair<std::size_t, std::size_t>; // begin, count
  /// Separate exact device and host numeric payloads for explicit admission.
  struct Cost { std::size_t device_bytes, host_bytes; };

  /** Opaque owner/generation receipt; neither input nor action exposes storage. */
  class Receipt
  {
    friend class PreparedDeviceSrMetric;
    const PreparedDeviceSrMetric* owner_;
    std::size_t generation_;
    bool action_;
    Receipt(const PreparedDeviceSrMetric* owner, std::size_t generation, bool action)
        : owner_(owner), generation_(generation), action_(action) {}
  };

  /// Exact numeric payload including fixed C contraction, excluding existing session/model/allocator.
  static Cost requiredBytes(std::size_t parameters, std::size_t population, std::size_t electrons,
      std::size_t channels, std::size_t frozen_intervals = 0)
  {
    if (!parameters || !population || !electrons || channels == 0 || channels > 3)
      throw std::invalid_argument("Device SR metric requires positive P/B/Ne and fixed C in [1,3]");
    const auto q = parameters / 4096 + (parameters % 4096 != 0);
    const auto cp = mul(mul(parameters, channels), 2);
    const auto cq = cp / 4096 + (cp % 4096 != 0);
    return {add(mul(16, parameters), add(mul(sizeof(DerivativeFiniteStatus), q),
                PreparedDeviceScoreContraction::requiredBytes(parameters, channels))),
        add(mul(8, add(mul(mul(3, population), electrons), mul(2, population))),
            add(mul(sizeof(DerivativeFiniteStatus), add(q, cq)),
                mul(sizeof(FrozenInterval), frozen_intervals)))};
  }

  /// Admit both explicit payload caps before any new numeric buffers or coordinate copies.
  static std::unique_ptr<PreparedDeviceSrMetric> prepare(PreparedDeviceProductSession& session,
      const std::vector<double>& positions, const std::vector<double>& weights,
      const std::vector<FrozenInterval>& frozen, std::size_t version,
      std::size_t device_cap, std::size_t host_cap)
  {
    std::lock_guard lock(session.mutex_);
    session.validateContractionControl(version);
    const auto& image = session.image_;
    if (image->periodicCell() || image->plan().environment().amplitude_scalar_domain != ScalarDomain::REAL)
      throw std::invalid_argument("Device SR metric requires ordinary real/open FP64 products");
    const auto p = image->plan().parameterCount();
    const auto c = std::max<std::size_t>(1, session.scoreContractionCapacity());
    const auto cost = requiredBytes(p, weights.size(), image->plan().modelShape().electrons(), c, frozen.size());
    const auto finite_cap = [](std::size_t cap) { return cap && cap != std::numeric_limits<std::size_t>::max(); };
    if (!finite_cap(device_cap) || !finite_cap(host_cap) ||
        cost.device_bytes > device_cap || cost.host_bytes > host_cap)
      throw std::length_error("Device SR metric exceeds explicit numeric payload caps");
    if (positions.size() != mul(mul(3, weights.size()), image->plan().modelShape().electrons()))
      throw std::invalid_argument("Device SR metric coordinate population mismatch");
    double weight = 0.0;
    for (double w : weights)
    {
      if (!device_math::isFiniteBinary64(w) || w < 0.0)
        throw std::invalid_argument("Device SR metric weights must be finite nonnegative");
      weight += w;
    }
    if (!(weight > 0.0) || !device_math::isFiniteBinary64(weight))
      throw std::invalid_argument("Device SR metric total weight must be finite positive");
    for (double coordinate : positions)
      if (!device_math::isFiniteBinary64(coordinate))
        throw std::invalid_argument("Device SR metric coordinates must be finite");
    std::size_t end = 0;
    for (const auto& interval : frozen)
    {
      if (!interval.second || interval.first < end || interval.first > p || interval.second > p - interval.first)
        throw std::invalid_argument("Device SR frozen intervals must be ordered disjoint canonical ranges");
      end = interval.first + interval.second;
    }
    auto result = std::unique_ptr<PreparedDeviceSrMetric>(
        new PreparedDeviceSrMetric(session, positions, weights, frozen, weight, cost));
    if (!session.score_contraction_)
    {
      session.score_contraction_ = std::make_unique<PreparedDeviceScoreContraction>(p, 1, session.queue_);
      session.contraction_device_bytes_.store(session.score_contraction_->deviceWorkspaceBytes(), std::memory_order_release);
      session.contraction_host_bytes_.store(session.score_contraction_->hostWorkspaceBytes(), std::memory_order_release);
      session.contraction_capacity_.store(1, std::memory_order_release);
      session.contraction_generation_.fetch_add(1, std::memory_order_release);
    }
    result->contraction_capacity_ = session.scoreContractionCapacity();
    return result;
  }

  /// Explicit cold/test ingress; all-P device finiteness is admitted by apply before projection.
  Receipt uploadDirection(const std::vector<double>& direction)
  {
    std::lock_guard lock(session_.mutex_);
    validate();
    input_ready_ = action_ready_ = false;
    if (direction.size() != direction_.capacity())
      throw std::invalid_argument("Device SR direction extent mismatch");
    advance(input_generation_);
    direction_.upload(direction.data(), direction.size());
    input_ready_ = true;
    return Receipt(this, input_generation_, false);
  }

  /** Apply P_train J^T diag(w/W) (J P_train v - mean_w) in deterministic sample order.
   * O(B) products/coefficients, cusp scalars and finite statuses may cross host.
   * There is no P upload, P download, damping, optimizer step or model publication.
   */
  Receipt apply(const Receipt& direction)
  {
    std::lock_guard lock(session_.mutex_);
    action_ready_ = false;
    try
    {
      validate();
      checkReceipt(direction, false);
      advance(action_generation_);
      scan(direction_); // All canonical entries, including frozen ones, precede projection.
      project(direction_);
      session_.jvp_.prepareDirectionDevice(direction_.data());
      const auto stride = 3 * image_->plan().modelShape().electrons();
      for (std::size_t begin = 0; begin < weights_.size(); begin += session_.score_capacity_)
      {
        const auto count = std::min(session_.score_capacity_, weights_.size() - begin);
        session_.executeUnderControl(PsiFormerAcceleratorOperation::PARAMETER_SCORE,
            positions_.data() + begin * stride, image_->version(), [&]() {
          session_.values_.evaluate(session_.positions_);
          session_.jvp_.evaluatePrepared(session_.values_);
          session_.products_.resize(count);
          session_.jvp_.downloadProducts(session_.products_);
          std::copy_n(session_.products_.begin(), count, products_.begin() + begin);
        }, count);
      }
      double mean = 0.0;
      for (std::size_t s = 0; s < weights_.size(); ++s)
        mean += weights_[s] * products_[s];
      mean /= weight_;
      if (!device_math::isFiniteBinary64(mean))
        throw std::runtime_error("Device SR metric weighted mean is nonfinite");
      session_.invalidateScoreContraction();
      auto& contraction = *session_.score_contraction_;
      contraction.begin(1);
      for (std::size_t s = 0; s < weights_.size(); ++s)
      {
        const std::complex<double> coefficient(weights_[s] / weight_ * (products_[s] - mean), 0.0);
        session_.executeUnderControl(PsiFormerAcceleratorOperation::PARAMETER_SCORE,
            positions_.data() + s * stride, image_->version(), [&]() {
          session_.values_.evaluate(session_.positions_);
          contraction.accumulate(session_.score_.reverse(session_.values_, session_.weights_), &coefficient, 1);
        });
      }
      contraction.finish(); // Scans the entire fixed C capacity, even with live C1.
      checkRuntime(launchDerivativeCopy(session_.queue_.getNative(), {action_.capacity(), 0, 2, 0, 1},
          contraction.sums_.data(), action_.data()), "SR resident C1 action");
      project(action_);
      session_.queue_.sync();
      session_.invalidateScoreContraction();
      action_ready_ = true;
      return Receipt(this, action_generation_, true);
    }
    catch (...)
    {
      session_.invalidateScoreContraction();
      session_.jvp_.clearPreparedDirection();
      input_ready_ = false;
      throw;
    }
  }

  /// Explicit inspection boundary, outside the warm resident-action transfer window.
  void downloadAction(const Receipt& action, std::vector<double>& destination)
  {
    std::lock_guard lock(session_.mutex_);
    validate();
    checkReceipt(action, true);
    if (destination.size() != action_.capacity())
      throw std::invalid_argument("Device SR action destination extent mismatch");
    action_.download(destination.data(), destination.size());
  }

  /// Fixed admitted numeric capacities; C is the prepared capacity, never live C1.
  Cost cost() const noexcept { return cost_; }

  /// Count concrete numeric capacities, including the shared fixed C owner once.
  Cost allocatedBytes()
  {
    std::lock_guard lock(session_.mutex_);
    validate();
    return {add(mul(8, add(direction_.capacity(), action_.capacity())),
                add(mul(sizeof(DerivativeFiniteStatus), status_.capacity()), session_.score_contraction_->deviceWorkspaceBytes())),
        add(mul(8, add(positions_.capacity(), add(weights_.capacity(), products_.capacity()))),
            add(mul(sizeof(FrozenInterval), frozen_.capacity()),
                add(mul(sizeof(DerivativeFiniteStatus), host_status_.capacity()), session_.score_contraction_->hostWorkspaceBytes())))};
  }

private:
  PreparedDeviceSrMetric(PreparedDeviceProductSession& session, const std::vector<double>& positions,
      const std::vector<double>& weights, const std::vector<FrozenInterval>& frozen, double weight, Cost cost)
      : session_(session), image_(session.image_), positions_(positions), weights_(weights),
        products_(weights.size()), frozen_(frozen), weight_(weight), cost_(cost),
        host_status_(image_->plan().parameterCount() / 4096 + (image_->plan().parameterCount() % 4096 != 0))
  {
    direction_.reserve(image_->plan().parameterCount());
    action_.reserve(direction_.capacity());
    status_.reserve(host_status_.size());
  }

  /// Checked sizes are shared by admission and concrete payload construction.
  static std::size_t mul(std::size_t a, std::size_t b)
  { return spatial_detail::checkedProduct(a, b, "Device SR metric payload overflow"); }

  static std::size_t add(std::size_t a, std::size_t b)
  { return spatial_detail::checkedAdd(a, b, "Device SR metric payload overflow"); }

  /// Publication/device changes invalidate the entire stamped fixed population.
  void validate()
  {
    session_.validateContractionControl(image_->version());
    if (session_.image_.get() != image_.get() || session_.scoreContractionCapacity() != contraction_capacity_)
      throw std::invalid_argument("Device SR metric image or fixed contraction capacity is stale");
  }

  /// Reject foreign, replaced or failed input/action generations before accessing storage.
  void checkReceipt(const Receipt& receipt, bool action) const
  {
    if (receipt.owner_ != this || receipt.action_ != action ||
        receipt.generation_ != (action ? action_generation_ : input_generation_) ||
        !(action ? action_ready_ : input_ready_))
      throw std::invalid_argument("Device SR metric receipt is foreign, stale or unavailable");
  }

  /// Advance one owner-local receipt generation without wraparound aliasing.
  static void advance(std::size_t& generation)
  {
    if (generation == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Device SR metric generation exhausted");
    ++generation;
  }

  /// Reuse finite admission over every P entry before frozen projection.
  void scan(const DeviceBuffer<double>& values)
  {
    checkRuntime(launchDerivativeFiniteScan(session_.queue_.getNative(),
        {values.capacity(), 4096, status_.capacity(), 0, 1}, values.data(), status_.data()), "SR direction finite scan");
    session_.queue_.sync();
    status_.download(host_status_.data(), host_status_.size());
    for (const auto& status : host_status_)
      if (status.nonfinite_count)
        throw std::runtime_error("Device SR direction is nonfinite, including frozen entries");
  }

  /// Zero only admitted canonical frozen intervals, on either side of the metric.
  void project(DeviceBuffer<double>& values)
  {
    for (const auto& interval : frozen_)
      checkRuntime(launchDerivativeZero(session_.queue_.getNative(),
          {interval.second, 0, 1, interval.first, 1}, values.data()), "SR frozen projection");
  }

  PreparedDeviceProductSession& session_;
  std::shared_ptr<const PreparedDeviceModel> image_;
  std::vector<double> positions_, weights_, products_;
  std::vector<FrozenInterval> frozen_;
  double weight_;
  Cost cost_;
  DeviceBuffer<double> direction_, action_;
  DeviceBuffer<DerivativeFiniteStatus> status_;
  std::vector<DerivativeFiniteStatus> host_status_;
  std::size_t input_generation_ = 0, action_generation_ = 0;
  std::size_t contraction_capacity_ = 0;
  bool input_ready_ = false, action_ready_ = false;
};

} // namespace qmcplusplus::psiformer::device
#endif
