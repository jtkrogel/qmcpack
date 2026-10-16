//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerDevicePcg.h
 * @brief Concrete real/open device storage for the qualified host-controlled PCG law.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_PCG_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_PCG_H
#include "PsiFormerDeviceSrMetric.h"
#include "QMCDrivers/WFTrain/KrylovPcgControl.h"

namespace qmcplusplus::psiformer::device
{
/** Retain one immutable population/image and RHS, not an optimizer candidate.
 * Session/backend must outlive this owner and its opaque receipts. All operations
 * use the session mutex; explicit host ingress/inspection are not warm residency.
 */
class PreparedDevicePcg
{
  using Metric = PreparedDeviceSrMetric;
  using Slot = wftrain::krylov_detail::PcgSlot;
  template<class Storage>
  friend void wftrain::krylov_detail::runPreconditionedConjugateGradient(
      Storage&, const wftrain::KrylovSolverControl&, std::size_t, wftrain::KrylovSolveDiagnostics&);
public:
  /// Receipts bind this address; only the owning unique_ptr may move, not its pointee.
  PreparedDevicePcg(const PreparedDevicePcg&) = delete;
  PreparedDevicePcg& operator=(const PreparedDevicePcg&) = delete;
  PreparedDevicePcg(PreparedDevicePcg&&) = delete;
  PreparedDevicePcg& operator=(PreparedDevicePcg&&) = delete;

  using FrozenInterval = Metric::FrozenInterval;
  using Cost = Metric::Cost;

  /// Bind an uploaded RHS to this owner and one nonwrapping replacement generation.
  class RhsReceipt
  {
    friend class PreparedDevicePcg;
    const PreparedDevicePcg* owner_;
    std::size_t generation_;
    RhsReceipt(const PreparedDevicePcg* owner, std::size_t generation) : owner_(owner), generation_(generation) {}
  };

  /// Identify inspectable diagnostic x, never accepted optimizer publication state.
  class IterateReceipt
  {
    friend class PreparedDevicePcg;
    const PreparedDevicePcg* owner_;
    std::size_t generation_;
    IterateReceipt(const PreparedDevicePcg* owner, std::size_t generation) : owner_(owner), generation_(generation) {}
  };

  /// Scalar diagnostics do not pretend that a device iterate is a host vector.
  struct Result { wftrain::KrylovSolveDiagnostics diagnostics; IterateReceipt iterate; };

  /// Include the entire metric/fixed C owner, persistent RHS and distinct PCG scratch.
  static Cost requiredBytes(std::size_t p, std::size_t b, std::size_t ne, std::size_t c, std::size_t f = 0)
  {
    const auto base = Metric::requiredBytes(p, b, ne, c, f), extra = additionalBytes(p);
    return {Metric::add(base.device_bytes, extra.device_bytes), Metric::add(base.host_bytes, extra.host_bytes)};
  }

  /// Admit both complete numeric payload caps before even preparing the metric.
  static std::unique_ptr<PreparedDevicePcg> prepare(PreparedDeviceProductSession& session,
      const std::vector<double>& positions, const std::vector<double>& weights,
      const std::vector<FrozenInterval>& frozen, std::size_t version, double damping,
      std::size_t device_cap, std::size_t host_cap)
  {
    std::lock_guard lock(session.mutex_);
    session.validateContractionControl(version);
    if (!device_math::isFiniteBinary64(damping) || damping <= 0.0)
      throw std::invalid_argument("Device PCG requires fixed finite positive damping");
    const auto p = session.image_->plan().parameterCount();
    const auto ne = session.image_->plan().modelShape().electrons();
    const auto c = std::max<std::size_t>(1, session.scoreContractionCapacity());
    const auto cost = requiredBytes(p, weights.size(), ne, c, frozen.size());
    const auto finite_cap = [](std::size_t cap) { return cap && cap != std::numeric_limits<std::size_t>::max(); };
    if (!finite_cap(device_cap) || !finite_cap(host_cap) || cost.device_bytes > device_cap || cost.host_bytes > host_cap)
      throw std::length_error("Device PCG exceeds explicit numeric payload caps");
    const auto base = Metric::requiredBytes(p, weights.size(), ne, c, frozen.size());
    auto metric = Metric::prepareUnderControl(session, positions, weights, frozen, version,
        base.device_bytes, base.host_bytes);
    return std::unique_ptr<PreparedDevicePcg>(new PreparedDevicePcg(session, std::move(metric), damping, cost));
  }

  /// Cold real host ingress; retain immutable RHS across explicit zero-start solves.
  RhsReceipt uploadRhs(const std::vector<double>& rhs, std::size_t version)
  {
    std::lock_guard lock(session_.mutex_);
    rhs_ready_ = iterate_ready_ = false;
    validate();
    if (version != metric_->image_->version() || rhs.size() != parameters_)
      throw std::invalid_argument("Device PCG RHS has stale version or wrong canonical extent");
    for (double value : rhs)
      if (!device_math::isFiniteBinary64(value))
        throw std::invalid_argument("Device PCG RHS is nonfinite, including frozen lanes");
    Metric::advance(rhs_generation_);
    try
    {
      rhs_.upload(rhs.data(), rhs.size());
      scan(rhs_);
    }
    catch (...)
    {
      failed_ = true;
      throw;
    }
    rhs_ready_ = true;
    return RhsReceipt(this, rhs_generation_);
  }

  /// Keep the entire shared recurrence under one model/session lock; no fallback.
  Result solve(const RhsReceipt& rhs, const wftrain::KrylovSolverControl& control)
  {
    std::lock_guard lock(session_.mutex_);
    iterate_ready_ = false;
    Metric::advance(iterate_generation_);
    validate();
    if (!rhs_ready_ || rhs.owner_ != this || rhs.generation_ != rhs_generation_)
      throw std::invalid_argument("Device PCG RHS receipt is foreign, stale or unavailable");
    wftrain::krylov_detail::validateControl(control);
    try
    {
      checkRuntime(launchDerivativeZero(session_.queue_.getNative(), {parameters_, 0, 1, 0, 1}, x_.data()), "PCG zero start");
      session_.queue_.sync();
      wftrain::KrylovSolveDiagnostics diagnostics;
      wftrain::krylov_detail::runPreconditionedConjugateGradient(*this, control, parameters_, diagnostics);
      session_.queue_.sync();
      iterate_ready_ = true;
      return {diagnostics, IterateReceipt(this, iterate_generation_)};
    }
    catch (...)
    {
      failed_ = true;
      rhs_ready_ = false;
      metric_->input_ready_ = metric_->action_ready_ = false;
      throw;
    }
  }

  /// Explicit diagnostic inspection also permits nonconverged x; not a candidate API.
  void downloadIterate(const IterateReceipt& iterate, std::vector<double>& output)
  {
    std::lock_guard lock(session_.mutex_);
    validate();
    if (!iterate_ready_ || iterate.owner_ != this || iterate.generation_ != iterate_generation_ || output.size() != parameters_)
      throw std::invalid_argument("Device PCG iterate receipt or host destination is invalid");
    try
    {
      x_.download(output.data(), output.size());
    }
    catch (...)
    {
      failed_ = true;
      rhs_ready_ = iterate_ready_ = false;
      throw;
    }
  }

  /// Report admitted fixed numeric capacity, excluding the existing session/model.
  Cost cost() const noexcept { return cost_; }

  /// Independently count actual buffers, including the fixed C metric source once.
  Cost allocatedBytes()
  {
    std::lock_guard lock(session_.mutex_);
    validate();
    auto actual = metric_->allocatedBytesUnderControl();
    for (const auto* buffer : {&x_, &r_, &z_, &p_, &rhs_, &partials_, &dot_})
      actual.device_bytes = Metric::add(actual.device_bytes, checkedDeviceBytes<double>(buffer->capacity()));
    for (const auto* buffer : {&partial_status_, &finite_status_, &dot_status_})
      actual.device_bytes = Metric::add(actual.device_bytes, checkedDeviceBytes<DerivativeFiniteStatus>(buffer->capacity()));
    actual.host_bytes = Metric::add(actual.host_bytes,
        Metric::add(checkedDeviceBytes<DerivativeFiniteStatus>(host_status_.capacity()), sizeof(host_dot_) + sizeof(host_dot_status_)));
    return actual;
  }

private:
  /// Prepare only fixed numeric buffers after complete factory admission.
  PreparedDevicePcg(PreparedDeviceProductSession& session, std::unique_ptr<Metric> metric, double damping, Cost cost)
      : session_(session), metric_(std::move(metric)), damping_(damping), cost_(cost),
        parameters_(metric_->image_->plan().parameterCount()),
        q_(parameters_ / 4096 + (parameters_ % 4096 != 0)), host_status_(q_)
  {
    for (auto* buffer : {&x_, &r_, &z_, &p_, &rhs_})
      buffer->reserve(parameters_);
    partials_.reserve(q_);
    partial_status_.reserve(q_);
    finite_status_.reserve(q_);
    dot_.reserve(1);
    dot_status_.reserve(1);
  }

  /// Charge five realP slots and independent finite/dot/status scratch exactly.
  static Cost additionalBytes(std::size_t p)
  {
    const auto q = p / 4096 + (p % 4096 != 0);
    const auto scalar = sizeof(double) + sizeof(DerivativeFiniteStatus);
    return {Metric::add(Metric::mul(5 * sizeof(double), p),
                Metric::add(Metric::mul(sizeof(double) + 2 * sizeof(DerivativeFiniteStatus), q), scalar)),
        Metric::add(Metric::mul(sizeof(DerivativeFiniteStatus), q), scalar)};
  }

  /// A hard runtime failure requires re-preparation; current image/device/C stay stamped.
  void validate()
  {
    if (failed_)
      throw std::logic_error("Device PCG owner failed; re-prepare before further use");
    metric_->validate();
  }

  /// Resolve only fixed canonical slots; no host view ever receives this pointer.
  DeviceBuffer<double>& buffer(Slot slot)
  {
    switch (slot)
    {
    case Slot::RHS: return rhs_;
    case Slot::X: return x_;
    case Slot::R: return r_;
    case Slot::Z: return z_;
    case Slot::P: return p_;
    case Slot::AP: return metric_->action_;
    default: throw std::logic_error("Device PCG has no warm-start slot");
    }
  }

  /// Keep the first concrete backend identity-preconditioned and zero-start only.
  bool hasWarmStart() const noexcept { return false; }

  /// Copy exact finite entries; an additional result scan cannot add information.
  void copy(Slot source, Slot destination)
  {
    scan(buffer(source));
    checkRuntime(launchDerivativeCopy(session_.queue_.getNative(), {parameters_, 0, 1, 0, 1},
        buffer(source).data(), buffer(destination).data()), "PCG resident copy");
  }

  /// Preserve the separate checked host scale before the subsequent AXPY.
  void scale(wftrain::DerivativeValue factor, Slot values)
  {
    scan(buffer(values));
    checkRuntime(launchDerivativeScale(session_.queue_.getNative(), {parameters_, 0, 1, 0, 1},
        realFactor(factor), buffer(values).data()), "PCG resident scale");
    scan(buffer(values));
  }

  /// Retain all-P input/output checks and the qualified separate-product AXPY law.
  void axpy(wftrain::DerivativeValue factor, Slot source, Slot destination)
  {
    scan(buffer(source));
    scan(buffer(destination));
    checkRuntime(launchDerivativeAxpy(session_.queue_.getNative(), {parameters_, 0, 1, 0, 1},
        realFactor(factor), buffer(source).data(), buffer(destination).data()), "PCG resident AXPY");
    scan(buffer(destination));
  }

  /// Keep r/z distinct even for the identity preconditioner.
  void applyPreconditioner(Slot source, Slot destination) { copy(source, destination); }

  /// Preserve original p while metric work is projected, then shift EVERY lane.
  void applyOperator(Slot source, Slot destination)
  {
    if (source != Slot::P || destination != Slot::AP)
      throw std::logic_error("Device PCG operator requires the zero-start P/AP slots");
    const auto direction = metric_->stageResidentDirectionUnderControl(p_.data(), parameters_);
    (void)metric_->applyUnderControl(direction);
    metric_->action_ready_ = false;
    axpy({damping_, 0.0}, source, destination);
  }

  /// Deterministic checked device reduction downloads only one scalar/status pair.
  wftrain::DerivativeValue dot(Slot left, Slot right)
  {
    checkRuntime(launchResponseDotDirection(session_.queue_.getNative(),
        {parameters_, 4096, q_, 0, 1, 0, 1}, buffer(left).data(), buffer(right).data(),
        partials_.data(), partial_status_.data(), dot_.data(), dot_status_.data()), "PCG resident dot");
    session_.queue_.sync();
    dot_.download(&host_dot_, 1);
    dot_status_.download(&host_dot_status_, 1);
    if (host_dot_status_.nonfinite_count)
      throw wftrain::MatrixFreeNumericalError("Device PCG dot is nonfinite");
    return {host_dot_, 0.0};
  }

  /// Retain the host raw-square norm domain, not the Adam stable-L2 variant.
  wftrain::DerivativeReal norm(Slot values)
  {
    const auto squared = dot(values, values);
    const double tolerance = std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(squared.real()));
    if (std::abs(squared.imag()) > tolerance || squared.real() < -tolerance)
      throw wftrain::MatrixFreeNumericalError("Device PCG norm is not real nonnegative");
    return std::sqrt(std::max(0.0, squared.real()));
  }

  /// Known numerical statuses use the common solver's NONFINITE_RESULT path.
  void scan(const DeviceBuffer<double>& values)
  {
    checkRuntime(launchDerivativeFiniteScan(session_.queue_.getNative(),
        {parameters_, 4096, q_, 0, 1}, values.data(), finite_status_.data()), "PCG all-P finite scan");
    session_.queue_.sync();
    finite_status_.download(host_status_.data(), host_status_.size());
    for (const auto& status : host_status_)
      if (status.nonfinite_count)
        throw wftrain::MatrixFreeNumericalError("Device PCG vector is nonfinite, including frozen lanes");
  }

  /// Reject unsupported imaginary or nonfinite scalars before any real update.
  static double realFactor(wftrain::DerivativeValue factor)
  {
    if (!device_math::isFiniteBinary64(factor.real()) || !device_math::isFiniteBinary64(factor.imag()) || factor.imag() != 0.0)
      throw std::invalid_argument("Device PCG vector factor must be finite real");
    return factor.real();
  }

  PreparedDeviceProductSession& session_;
  std::unique_ptr<Metric> metric_;
  double damping_;
  Cost cost_;
  const std::size_t parameters_, q_;
  DeviceBuffer<double> x_, r_, z_, p_, rhs_, partials_, dot_;
  DeviceBuffer<DerivativeFiniteStatus> partial_status_, finite_status_, dot_status_;
  std::vector<DerivativeFiniteStatus> host_status_;
  double host_dot_ = 0;
  DerivativeFiniteStatus host_dot_status_{};
  std::size_t rhs_generation_ = 0, iterate_generation_ = 0;
  bool rhs_ready_ = false, iterate_ready_ = false, failed_ = false;
};
} // namespace qmcplusplus::psiformer::device
#endif
