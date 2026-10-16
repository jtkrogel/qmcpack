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
#include "QMCDrivers/WFTrain/SrMetricNumerics.h"
#include <algorithm>

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

  /// Only candidate_ready makes the final undamped norms and caller candidate valid.
  struct EnergyResult
  {
    wftrain::KrylovSolveDiagnostics solve;
    double gradient_norm = 0, direction_norm = 0, metric_norm = 0;
    bool candidate_ready = false;
  };

  /// Include the entire metric/fixed C owner, persistent RHS and distinct PCG scratch.
  static Cost requiredBytes(std::size_t p, std::size_t b, std::size_t ne, std::size_t c, std::size_t f = 0)
  {
    const auto base = Metric::requiredBytes(p, b, ne, c, f), extra = additionalBytes(p);
    return {Metric::add(base.device_bytes, extra.device_bytes), Metric::add(base.host_bytes, extra.host_bytes)};
  }

  /// Charge optional stable full-gradient partials/status, without another P buffer.
  static Cost energyRequiredBytes(std::size_t p, std::size_t b, std::size_t ne, std::size_t c, std::size_t f = 0)
  {
    if (c != 3)
      throw std::invalid_argument("Device ENERGY SR requires fixed C3 preparation");
    const auto base = requiredBytes(p, b, ne, c, f);
    const auto extra = Metric::add(PreparedDeviceAdam::energyGradientRequiredBytes(p), sizeof(std::uint64_t));
    return {Metric::add(base.device_bytes, extra), Metric::add(base.host_bytes, extra)};
  }

  /// Admit both complete numeric payload caps before even preparing the metric.
  static std::unique_ptr<PreparedDevicePcg> prepare(PreparedDeviceProductSession& session,
      const std::vector<double>& positions, const std::vector<double>& weights,
      const std::vector<FrozenInterval>& frozen, std::size_t version, double damping,
      std::size_t device_cap, std::size_t host_cap)
  {
    std::lock_guard lock(session.mutex_);
    return prepareUnderControl(session, positions, weights, frozen, version, damping, device_cap, host_cap, false);
  }

  /// Opt in to exact ENERGY completion storage before any ordinary PCG allocation.
  static std::unique_ptr<PreparedDevicePcg> prepareEnergy(PreparedDeviceProductSession& session,
      const std::vector<double>& positions, const std::vector<double>& weights,
      const std::vector<FrozenInterval>& frozen, std::size_t version, double damping,
      std::size_t device_cap, std::size_t host_cap)
  {
    std::lock_guard lock(session.mutex_);
    return prepareUnderControl(session, positions, weights, frozen, version, damping, device_cap, host_cap, true);
  }

  /// Retain exact storage, not an image: the first supplied population is real, never dummy.
  static std::unique_ptr<PreparedDevicePcg> prepareRetainedEnergy(PreparedDeviceProductSession& session,
      const std::vector<double>& positions, const std::vector<double>& weights,
      const std::vector<FrozenInterval>& frozen, std::size_t version, double damping,
      std::size_t device_cap, std::size_t host_cap)
  {
    std::lock_guard lock(session.mutex_);
    try
    {
      auto owner = prepareUnderControl(session, positions, weights, frozen, version, damping, device_cap, host_cap, true);
      owner->retained_ = true;
      owner->retained_device_cap_ = device_cap;
      owner->retained_host_cap_ = host_cap;
      owner->retireUnderControl();
      return owner;
    }
    catch (...)
    {
      session.invalidateScoreContraction();
      throw;
    }
  }

private:
  /// Both public factories own the control mutex and admit the complete payload first.
  static std::unique_ptr<PreparedDevicePcg> prepareUnderControl(PreparedDeviceProductSession& session,
      const std::vector<double>& positions, const std::vector<double>& weights,
      const std::vector<FrozenInterval>& frozen, std::size_t version, double damping,
      std::size_t device_cap, std::size_t host_cap, bool energy)
  {
    session.validateContractionControl(version);
    if (!device_math::isFiniteBinary64(damping) || damping <= 0.0)
      throw std::invalid_argument("Device PCG requires fixed finite positive damping");
    const auto p = session.image_->plan().parameterCount();
    const auto ne = session.image_->plan().modelShape().electrons();
    const auto c = std::max<std::size_t>(1, session.scoreContractionCapacity());
    const auto cost = energy ? energyRequiredBytes(p, weights.size(), ne, c, frozen.size()) :
        requiredBytes(p, weights.size(), ne, c, frozen.size());
    const auto finite_cap = [](std::size_t cap) { return cap && cap != std::numeric_limits<std::size_t>::max(); };
    if (!finite_cap(device_cap) || !finite_cap(host_cap) || cost.device_bytes > device_cap || cost.host_bytes > host_cap)
      throw std::length_error("Device PCG exceeds explicit numeric payload caps");
    const auto base = Metric::requiredBytes(p, weights.size(), ne, c, frozen.size());
    auto metric = Metric::prepareUnderControl(session, positions, weights, frozen, version,
        base.device_bytes, base.host_bytes);
    return std::unique_ptr<PreparedDevicePcg>(new PreparedDevicePcg(session, std::move(metric), damping, cost, energy));
  }

public:
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
    return solveUnderControl(rhs, control);
  }

private:
  /// Delegate unchanged zero-start control while the concrete ENERGY caller keeps the lock.
  Result solveUnderControl(const RhsReceipt& rhs, const wftrain::KrylovSolverControl& control)
  {
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

public:
  /// Consume tagged physical ENERGY once, with all leases retained until candidate download.
  EnergyResult consumeEnergy(std::size_t version, std::size_t generation,
      const std::vector<double>& parameters, double weight, double energy, double learning_rate,
      const wftrain::KrylovSolverControl& control, std::vector<double>& candidate)
  {
    std::lock_guard lock(session_.mutex_);
    return consumeEnergyUnderControl(version, generation, parameters, weight, energy, learning_rate, control, candidate);
  }

  /// Bind/consume/retire under one lock; neither failure nor publication may pin old P.
  EnergyResult consumeRetainedEnergy(const std::vector<double>& positions, const std::vector<double>& weights,
      const std::vector<FrozenInterval>& frozen, std::size_t version, std::size_t generation,
      const std::vector<double>& parameters, double weight, double energy, double damping, double learning_rate,
      const wftrain::KrylovSolverControl& control, std::size_t device_cap, std::size_t host_cap,
      std::vector<double>& candidate)
  {
    std::lock_guard lock(session_.mutex_);
    if (!retained_)
      throw std::invalid_argument("Retained ENERGY consume requires its explicit retained factory");
    Retirement retirement{*this};
    try
    {
      bindEnergyUnderControl(positions, weights, frozen, version, damping, device_cap, host_cap);
      const auto result = consumeEnergyUnderControl(version, generation, parameters, weight, energy, learning_rate, control, candidate);
      if (!result.solve.converged || !result.candidate_ready)
        failed_ = true;
      else
        consumed_ = true;
      return result;
    }
    catch (...)
    {
      failed_ = true;
      session_.invalidateScoreContraction();
      throw;
    }
  }

private:
  /// Preserve the ordinary ENERGY body/order while its two explicit wrappers own the lock.
  EnergyResult consumeEnergyUnderControl(std::size_t version, std::size_t generation,
      const std::vector<double>& parameters, double weight, double energy, double learning_rate,
      const wftrain::KrylovSolverControl& control, std::vector<double>& candidate)
  {
    rhs_ready_ = iterate_ready_ = false;
    try
    {
      validate();
      auto* source = session_.score_contraction_.get();
      if (version != metric_->image_->version() || session_.energy_image_ != metric_->image_.get() ||
          generation != session_.contraction_transaction_ || !source || !source->ready_ ||
          source->channels_ != 3 || source->parameters_ != parameters_ ||
          parameters.size() != parameters_ || candidate.size() != parameters_ ||
          energy_host_.size() != PreparedDeviceAdam::energyGradientRequiredBytes(parameters_) / sizeof(double) ||
          energy_partials_.capacity() != energy_host_.size() || energy_status_.capacity() != 1 ||
          energy_status_host_.size() != 1 || weight != metric_->weight_ ||
          !device_math::isFiniteBinary64(energy) || !device_math::isFiniteBinary64(learning_rate) || learning_rate <= 0)
        throw std::invalid_argument("Device ENERGY SR inputs, C3 provenance or preparation are invalid");
      wftrain::krylov_detail::validateControl(control);
      Metric::advance(rhs_generation_);
      try
      {
        x_.upload(parameters.data(), parameters_);
#ifdef QMC_CUDA2HIP
        checkRuntime(hipMemsetAsync(energy_status_.data(), 0xff, sizeof(std::uint64_t), session_.queue_.getNative()), "SR ENERGY status reset");
#else
        checkRuntime(cudaMemsetAsync(energy_status_.data(), 0xff, sizeof(std::uint64_t), session_.queue_.getNative()), "SR ENERGY status reset");
#endif
        checkRuntime(launchAdamEnergyGradient(session_.queue_.getNative(), parameters_, source->sums_.data(),
            x_.data(), metric_->image_->parameters(), weight, energy, rhs_.data(), energy_partials_.data(),
            energy_status_.data()), "SR resident ENERGY gradient");
        session_.queue_.sync();
        energy_status_.download(energy_status_host_.data(), 1);
        if (energy_status_host_[0] != std::numeric_limits<std::uint64_t>::max())
          throw std::overflow_error("Device ENERGY SR gradient or canonical old parameters are invalid");
        energy_partials_.download(energy_host_.data(), energy_host_.size());
        EnergyResult result;
        for (std::size_t i = 0; i < energy_host_.size(); i += 2)
          result.gradient_norm = std::hypot(result.gradient_norm, energy_host_[i] * std::sqrt(energy_host_[i + 1]));
        if (!device_math::isFiniteBinary64(result.gradient_norm))
          throw std::overflow_error("Device ENERGY SR full gradient norm is nonfinite");
        metric_->project(rhs_); // Full-gradient finite/stable norm precedes frozen projection.
        rhs_ready_ = true;
        session_.invalidateScoreContraction(); // C1 metric work may now overwrite fixed C3.
        result.solve = solveUnderControl(RhsReceipt(this, rhs_generation_), control).diagnostics;
        if (!result.solve.converged)
          return result;
        const auto direction = metric_->stageResidentDirectionUnderControl(x_.data(), parameters_);
        (void)metric_->applyUnderControl(direction); // EXTRA UNDAMPED action, never shifted AP.
        metric_->action_ready_ = false;
        const auto squared = dot(Slot::X, Slot::AP);
        const auto product_norm = norm(Slot::X) * norm(Slot::AP);
        const auto tolerance = wftrain::sr_metric_detail::contractionRoundoffTolerance(parameters_, squared, product_norm);
        if (std::abs(squared.imag()) > tolerance || squared.real() < -tolerance)
          throw wftrain::MatrixFreeNumericalError("SR covariance metric norm is not real and nonnegative");
        result.direction_norm = norm(Slot::X);
        result.metric_norm = std::sqrt(std::max(0.0, squared.real()));
        checkRuntime(launchDerivativeCopy(session_.queue_.getNative(), {parameters_, 0, 1, 0, 1},
            metric_->image_->parameters(), r_.data()), "SR canonical candidate seed");
        axpy({-learning_rate, 0.0}, Slot::X, Slot::R);
        for (const auto& interval : metric_->frozen_)
          checkRuntime(launchDerivativeCopy(session_.queue_.getNative(),
              {interval.second, interval.first, 1, interval.first, 1}, metric_->image_->parameters(), r_.data()),
              "SR frozen candidate preservation");
        r_.download(candidate.data(), parameters_);
        result.candidate_ready = true;
        return result;
      }
      catch (...)
      {
        failed_ = true;
        rhs_ready_ = iterate_ready_ = false;
        throw;
      }
    }
    catch (...)
    {
      session_.invalidateScoreContraction();
      throw;
    }
  }

public:
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
    // A detached retained owner permits capacity inspection, never numerical/image access.
    Cost actual;
    if (retained_ && !metric_->image_)
    {
      if (failed_)
        throw std::logic_error("Device PCG owner failed; re-prepare before further use");
      actual = metric_->countAllocatedBytesUnderControl();
    }
    else
    {
      validate();
      actual = metric_->allocatedBytesUnderControl();
    }
    for (const auto* buffer : {&x_, &r_, &z_, &p_, &rhs_, &partials_, &dot_})
      actual.device_bytes = Metric::add(actual.device_bytes, checkedDeviceBytes<double>(buffer->capacity()));
    for (const auto* buffer : {&partial_status_, &finite_status_, &dot_status_})
      actual.device_bytes = Metric::add(actual.device_bytes, checkedDeviceBytes<DerivativeFiniteStatus>(buffer->capacity()));
    actual.host_bytes = Metric::add(actual.host_bytes,
        Metric::add(checkedDeviceBytes<DerivativeFiniteStatus>(host_status_.capacity()), sizeof(host_dot_) + sizeof(host_dot_status_)));
    actual.device_bytes = Metric::add(actual.device_bytes,
        Metric::add(checkedDeviceBytes<double>(energy_partials_.capacity()), checkedDeviceBytes<std::uint64_t>(energy_status_.capacity())));
    actual.host_bytes = Metric::add(actual.host_bytes,
        Metric::add(checkedDeviceBytes<double>(energy_host_.capacity()), checkedDeviceBytes<std::uint64_t>(energy_status_host_.capacity())));
    return actual;
  }

private:
  /// Prepare only fixed numeric buffers after complete factory admission.
  PreparedDevicePcg(PreparedDeviceProductSession& session, std::unique_ptr<Metric> metric, double damping, Cost cost, bool energy)
      : session_(session), metric_(std::move(metric)), damping_(damping), cost_(cost),
        parameters_(metric_->image_->plan().parameterCount()),
        q_(parameters_ / 4096 + (parameters_ % 4096 != 0)),
        electrons_(metric_->image_->plan().modelShape().electrons()), score_capacity_(session.score_capacity_),
        host_status_(q_), binding_version_(metric_->image_->version())
  {
    for (auto* buffer : {&x_, &r_, &z_, &p_, &rhs_})
      buffer->reserve(parameters_);
    partials_.reserve(q_);
    partial_status_.reserve(q_);
    finite_status_.reserve(q_);
    dot_.reserve(1);
    dot_status_.reserve(1);
    if (energy)
    {
      energy_host_.resize(PreparedDeviceAdam::energyGradientRequiredBytes(parameters_) / sizeof(double));
      energy_status_host_.resize(1);
      energy_partials_.reserve(energy_host_.size());
      energy_status_.reserve(1);
    }
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

  /// Scope cleanup never waits/rebinds/launches: Session still owns any bound image.
  void retireUnderControl() noexcept
  {
    rhs_ready_ = iterate_ready_ = false;
    metric_->input_ready_ = metric_->action_ready_ = false;
    metric_->image_.reset();
  }

  /// Destroy before unlocking, including failed bind/preflight/consume; never mask an error.
  struct Retirement
  {
    PreparedDevicePcg& owner;
    ~Retirement() noexcept
    {
      owner.session_.invalidateScoreContraction();
      owner.retireUnderControl();
    }
  };

  /// Only the producer may have caught up the healthy fixed-layout Session, BEFORE C3.
  void bindEnergyUnderControl(const std::vector<double>& positions, const std::vector<double>& weights,
      const std::vector<FrozenInterval>& frozen, std::size_t version, double damping,
      std::size_t device_cap, std::size_t host_cap)
  {
    if (!retained_ || failed_ || metric_->image_ || session_.image_.get() != session_.active_image_.get() ||
        version != session_.image_->version() || damping != damping_ ||
        (consumed_ && (binding_version_ == std::numeric_limits<std::size_t>::max() || version != binding_version_ + 1)) ||
        (!consumed_ && version != binding_version_) ||
        session_.image_->plan().parameterCount() != parameters_ ||
        session_.image_->plan().modelShape().electrons() != electrons_ || session_.score_capacity_ != score_capacity_ ||
        session_.scoreContractionCapacity() != 3 || metric_->contraction_capacity_ != 3 ||
        positions.size() != metric_->positions_.size() || weights.size() != metric_->weights_.size() || frozen != metric_->frozen_)
      throw std::invalid_argument("Retained ENERGY SR requires its healthy detached fixed-shape successor batch");
    session_.validateContractionControl(version); // Already-current image means NO catch-up/C3 invalidation.
    const auto required = energyRequiredBytes(parameters_, weights.size(), electrons_, 3, frozen.size());
    if (device_cap != retained_device_cap_ || host_cap != retained_host_cap_ ||
        required.device_bytes > device_cap || required.host_bytes > host_cap)
      throw std::length_error("Retained ENERGY SR requires unchanged complete device/host caps");
    double total = 0;
    for (double w : weights)
    {
      if (!device_math::isFiniteBinary64(w) || w < 0)
        throw std::invalid_argument("Retained ENERGY SR weights must be finite nonnegative");
      total += w;
    }
    if (!(total > 0) || !device_math::isFiniteBinary64(total))
      throw std::invalid_argument("Retained ENERGY SR requires finite positive total weight");
    for (double coordinate : positions)
      if (!device_math::isFiniteBinary64(coordinate))
        throw std::invalid_argument("Retained ENERGY SR coordinates must be finite");
    std::copy(positions.begin(), positions.end(), metric_->positions_.begin());
    std::copy(weights.begin(), weights.end(), metric_->weights_.begin());
    metric_->weight_ = total;
    metric_->image_ = session_.image_; // Same Session's qualified validateSuccessor proves schema/layout stability.
    binding_version_ = version;
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
  const std::size_t parameters_, q_, electrons_, score_capacity_;
  DeviceBuffer<double> x_, r_, z_, p_, rhs_, partials_, dot_;
  DeviceBuffer<double> energy_partials_;
  DeviceBuffer<std::uint64_t> energy_status_;
  std::vector<double> energy_host_;
  std::vector<std::uint64_t> energy_status_host_;
  DeviceBuffer<DerivativeFiniteStatus> partial_status_, finite_status_, dot_status_;
  std::vector<DerivativeFiniteStatus> host_status_;
  double host_dot_ = 0;
  DerivativeFiniteStatus host_dot_status_{};
  std::size_t rhs_generation_ = 0, iterate_generation_ = 0;
  std::size_t binding_version_, retained_device_cap_ = 0, retained_host_cap_ = 0;
  bool retained_ = false, consumed_ = false;
  bool rhs_ready_ = false, iterate_ready_ = false, failed_ = false;
};
} // namespace qmcplusplus::psiformer::device
#endif
