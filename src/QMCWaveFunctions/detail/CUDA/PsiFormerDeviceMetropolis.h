//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerDeviceMetropolis.h
 * @brief Injected-draw real/open joint Metropolis, with opt-in native host drift.
 * The borrowed VALUE graph and queue MUST outlive this noncopyable owner. Callers
 * hold their existing model/crowd execution lease, exclude competing graph work
 * and publication during a request, and keep the owning device current. This is
 * not a production sampler, RNG, Hamiltonian or parameter-publication authority.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_METROPOLIS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_METROPOLIS_H

#include "PsiFormerDeviceInference.h"
#include "PsiFormerDeviceMetropolisKernels.h"
#include "PsiFormerDeviceCoordinateGradient.h"

#include <functional>

namespace qmcplusplus::psiformer::device
{
/// Borrow the native configured host drift policy; arguments are variance, G, drift.
using NativeDriftCallback = std::function<void(double, const double*, double*)>;

/// Drift additions charge retained VALUE, no-P reverse and device Green correction.
inline CoordinateGradientCost estimateMetropolisDriftCost(const PsiFormerExecutionPlan& plan,
    std::size_t capacity, std::size_t population)
{
  auto cost = estimateCoordinateGradientCost(plan, capacity);
  const auto add = [](std::size_t a, std::size_t b) {
    return spatial_detail::checkedAdd(a, b, "Metropolis drift byte overflow");
  };
  const auto mul = [](std::size_t a, std::size_t b) {
    return spatial_detail::checkedProduct(a, b, "Metropolis drift extent overflow");
  };
  if (!population)
    throw std::invalid_argument("Metropolis drift requires a population");
  cost.device_workspace_bytes = add(cost.device_workspace_bytes,
      add(estimateRetainedValueIncrementBytes(plan, capacity), checkedDeviceBytes<double>(mul(2, population))));
  // Two population G caches, displacement, microtile G scratch, masses and Green sums.
  cost.host_workspace_bytes = add(cost.host_workspace_bytes, checkedDeviceBytes<double>(add(
      mul(add(mul(3, population), capacity), mul(3, plan.modelShape().electrons())),
      add(mul(3, population), plan.modelShape().electrons()))));
  return cost;
}

/// Exact standalone numeric storage; borrowed graph/model/vendor bytes are excluded.
struct MetropolisCost
{
  std::size_t population, electrons, coordinates, device_bytes, host_bytes;
  bool selected = false;
};

/// Check every extent/byte before any owner state or device allocation.
inline MetropolisCost estimateMetropolisCost(std::size_t population, std::size_t electrons, bool selected = false)
{
  using spatial_detail::checkedAdd;
  using spatial_detail::checkedProduct;
  if (!population || !electrons)
    throw std::invalid_argument("Device Metropolis needs a nonempty physical population");
  const auto coordinates = checkedProduct(checkedProduct(population, electrons, "Metropolis population overflow"),
                                         3, "Metropolis Cartesian extent overflow");
  // The first-error status reserves its low eight bits for the failure code.
  if (coordinates - 1 > (std::numeric_limits<std::uint64_t>::max() >> 8))
    throw std::overflow_error("Metropolis status index encoding overflow");
  const auto doubles = checkedAdd(checkedAdd(checkedProduct(3, coordinates, "Metropolis coordinate buffers overflow"),
      electrons, "Metropolis masses overflow"), checkedProduct(6, population, "Metropolis scalar rows overflow"),
      "Metropolis double storage overflow");
  const auto masks = checkedDeviceBytes<std::uint32_t>(population);
  const auto selection = selected ? checkedDeviceBytes<std::uint32_t>(coordinates / 3) : 0;
  const auto host_draws = selected ? checkedDeviceBytes<double>(coordinates) : 0;
  return {population, electrons, coordinates,
      checkedAdd(checkedAdd(checkedDeviceBytes<double>(doubles), masks, "Metropolis device bytes overflow"),
                 checkedAdd(sizeof(std::uint64_t), selection, "Metropolis selection bytes overflow"),
                 "Metropolis status bytes overflow"),
      checkedAdd(checkedAdd(masks, selection, "Metropolis host mask bytes overflow"), host_draws,
                 "Metropolis host selection draws overflow"), selected};
}

/// Owner-issued pending source epoch; weak identity prevents expired-owner reuse.
class MetropolisProposal
{
public:
  /// Represent no pending proposal for rejection tests or empty caller state.
  MetropolisProposal() = default;

private:
  friend class PreparedDeviceMetropolis;
  std::weak_ptr<coordinate_detail::SourceState> source_;
  std::size_t generation_ = 0, version_ = 0;
};

/** Own accepted/proposed coordinates and caches, never a parameter image.
 * Joint all-electron or selected moves are independent of AE versus ECP Hamiltonians.
 * Draws upload from the host; status/exports download explicitly. No RNG state
 * or opaque derived TWF/Hamiltonian cache is maintained here.
 */
class PreparedDeviceMetropolis
{
  /// Keep the two cache generations address-stable while swapping their ownership.
  struct ValueCache
  {
    /// Allocate just one phase/log pair for the complete population.
    explicit ValueCache(std::size_t population) : phase(population), logabs(population) {}

    DeviceBuffer<double> phase, logabs;
  };

public:
  /// Admit the entire owner before allocation; VALUE microtile capacity is independent of B.
  static std::unique_ptr<PreparedDeviceMetropolis> prepare(PreparedDeviceInference& graph,
      compute::Queue<PlatformKind::CUDA>& queue, const std::vector<double>& positions,
      const std::vector<double>& inverse_mass, std::size_t workspace_budget_bytes, bool selected = false)
  {
    graph.model_->validateCurrentDevice();
    const auto ne = graph.plan_.modelShape().electrons();
    if (graph.mode_ != SpatialJetMode::VALUE_ONLY || graph.mixed_dense_ || graph.retained_targets_ ||
        graph.target_preparation_.max_targets ||
        graph.plan_.environment().boundary != BoundaryCondition::OPEN || &graph.queue_ != &queue ||
        positions.empty() || positions.size() % (3 * ne) != 0 || inverse_mass.size() != ne)
      throw std::invalid_argument("Device Metropolis needs one borrowed FP64/open VALUE graph and packed physical inputs");
    const auto cost = estimateMetropolisCost(positions.size() / (3 * ne), ne, selected);
    if (cost.device_bytes > workspace_budget_bytes)
      throw std::length_error("Device Metropolis exceeds its standalone owner budget");
    for (const auto value : positions)
      if (!device_math::isFiniteBinary64(value))
        throw std::invalid_argument("Device Metropolis seed coordinates are nonfinite");
    for (const auto value : inverse_mass)
      if (!device_math::isFiniteBinary64(value) || value <= 0)
        throw std::invalid_argument("Device Metropolis inverse masses must be finite and positive");
    return std::unique_ptr<PreparedDeviceMetropolis>(new PreparedDeviceMetropolis(graph, queue, cost, positions, inverse_mass));
  }

  /// Admit additions before reverse/cache allocation; caller already admitted the VALUE upgrade.
  static std::unique_ptr<PreparedDeviceMetropolis> prepareDrift(PreparedDeviceInference& graph,
      compute::Queue<PlatformKind::CUDA>& queue, const std::vector<double>& positions,
      const std::vector<double>& inverse_mass, std::size_t owner_budget, std::size_t drift_budget,
      NativeDriftCallback drift, bool selected = false)
  {
    const auto cost = estimateMetropolisDriftCost(graph.plan_, graph.batch_capacity_,
        positions.size() / (3 * graph.plan_.modelShape().electrons()));
    if (!drift || !drift_budget || drift_budget == std::numeric_limits<std::size_t>::max() ||
        !graph.retained_backflow_.capacity())
      throw std::invalid_argument("Explicit drift needs a native policy, finite cap and retained VALUE graph");
    if (cost.device_workspace_bytes > drift_budget)
      throw std::length_error("Metropolis drift additions exceed per-crowd device budget");
    auto result = prepare(graph, queue, positions, inverse_mass, owner_budget, selected);
    result->gradient_ = PreparedDeviceCoordinateGradient::prepare(graph.model_, graph.batch_capacity_, queue, graph.blas_,
        estimateCoordinateGradientCost(graph.plan_, graph.batch_capacity_).device_workspace_bytes);
    result->gradient_version_ = graph.model_->version();
    result->drift_ = std::move(drift);
    result->drift_masses_ = inverse_mass;
    for (auto* buffer : {&result->accepted_gradient_, &result->staged_gradient_, &result->displacement_})
      buffer->resize(positions.size());
    result->gradient_scratch_.resize(graph.batch_capacity_ * 3 * graph.plan_.modelShape().electrons());
    result->forward_green_.resize(result->cost_.population);
    result->host_green_.resize(2 * result->cost_.population);
    result->green_.reserve(2 * result->cost_.population);
    auto host_bytes = result->gradient_->hostWorkspaceBytes();
    for (const auto* buffer : {&result->accepted_gradient_, &result->staged_gradient_, &result->displacement_,
        &result->gradient_scratch_, &result->drift_masses_, &result->forward_green_, &result->host_green_})
      host_bytes = spatial_detail::checkedAdd(host_bytes, checkedDeviceBytes<double>(buffer->capacity()),
                                             "Metropolis drift host capacity overflow");
    if (host_bytes != cost.host_workspace_bytes)
      throw std::logic_error("Metropolis drift host capacities differ from exact preparation accounting");
    result->refreshAccepted(graph.model_->version());
    return result;
  }

  /// Preserve borrowed references and source identity by forbidding copies.
  PreparedDeviceMetropolis(const PreparedDeviceMetropolis&) = delete;

  /// Never replace an owner with unresolved coordinate/cache state.
  PreparedDeviceMetropolis& operator=(const PreparedDeviceMetropolis&) = delete;

  /// Report checked standalone allocation and host-mask capacities.
  const MetropolisCost& cost() const noexcept { return cost_; }

  /// Bind only the current request's native callback; never retain an older caller closure for use.
  void bindNativeDriftPolicy(NativeDriftCallback drift)
  {
    std::lock_guard lock(mutex_);
    if (!gradient_ || !drift || pending_)
      throw std::invalid_argument("Native drift policy requires a prepared coordinate reverse owner");
    drift_ = std::move(drift);
  }

  /// Recompute accepted values after an explicitly rebound parameter image.
  /// Coordinates survive a failed refresh, but no old cache is current afterward.
  void refreshAccepted(std::size_t version)
  {
    std::lock_guard lock(mutex_);
    abortPending();
    cache_valid_ = false;
    validateGraph(version);
    if (gradient_ && gradient_version_ != version)
    {
      gradient_->rebindModel(graph_.model_);
      gradient_version_ = version;
    }
    stageValues(*accepted_, true);
    std::swap(accepted_cache_, staged_cache_);
    if (gradient_)
      accepted_gradient_.swap(staged_gradient_);
    cache_version_ = version;
    cache_generation_ = accepted_->state_->generation;
    cache_valid_ = true;
  }

  /// Stage all proposed VALUE microtiles without changing accepted coordinates/cache.
  MetropolisProposal propose(const std::vector<double>& gaussian, double tau, std::size_t version,
                               const std::vector<std::size_t>& selected_indices = {})
  {
    std::lock_guard lock(mutex_);
    const bool was_pending = pending_;
    abortPending();
    if (was_pending)
      throw std::logic_error("Resolve the pending Metropolis proposal before replacing it");
    requireCache(version);
    const auto draw_count = cost_.selected
        ? spatial_detail::checkedProduct(selected_indices.size(), 3, "Metropolis selected draw extent overflow")
        : cost_.coordinates;
    if (gaussian.size() != draw_count || !device_math::isFiniteBinary64(tau) || tau <= 0)
      throw std::invalid_argument("Device Metropolis Gaussian extent or timestep is invalid");
    if (cost_.selected)
    {
      if (selected_indices.empty() || selected_indices.size() % cost_.population ||
          selected_indices.size() / cost_.population >= cost_.electrons)
        throw std::invalid_argument("Device Metropolis selected extent must be a proper nonempty subset per walker");
      const auto count = selected_indices.size() / cost_.population;
      for (std::size_t walker = 0; walker < cost_.population; ++walker)
        for (std::size_t local = 0; local < count; ++local)
        {
          const auto index = walker * count + local;
          if (selected_indices[index] >= cost_.electrons ||
              (local && selected_indices[index - 1] >= selected_indices[index]))
            throw std::invalid_argument("Device Metropolis selected indices must be sorted, unique and in range");
        }
      std::fill(host_selected_.begin(), host_selected_.end(), 0);
      std::fill(dense_gaussian_.begin(), dense_gaussian_.end(), 0);
      for (std::size_t walker = 0; walker < cost_.population; ++walker)
        for (std::size_t local = 0; local < count; ++local)
        {
          const auto index = walker * count + local;
          const auto electron = walker * cost_.electrons + selected_indices[index];
          host_selected_[electron] = 1;
          std::copy_n(gaussian.data() + 3 * index, 3, dense_gaussian_.data() + 3 * electron);
        }
    }
    else if (!selected_indices.empty())
      throw std::invalid_argument("Device Metropolis selected proposals need an admitted selection workspace");
    for (const auto value : gaussian)
      if (!device_math::isFiniteBinary64(value))
        throw std::invalid_argument("Device Metropolis Gaussian draw is nonfinite");
    try
    {
      const auto& physical_gaussian = cost_.selected ? dense_gaussian_ : gaussian;
      if (gradient_)
        formDisplacement(physical_gaussian, tau);
      gaussian_.upload(gradient_ ? displacement_.data() : physical_gaussian.data(), cost_.coordinates);
      if (cost_.selected)
        selected_.upload(host_selected_.data(), host_selected_.size());
      {
        std::scoped_lock sources(accepted_->state_->mutex, proposed_->state_->mutex);
        beginMutation(*proposed_->state_);
        resetStatus();
        checkRuntime(launchMetropolisProposal(queue_.getNative(), cost_.population, cost_.electrons,
            accepted_->state_->positions.data(), gaussian_.data(), inverse_mass_.data(), tau,
            proposed_->state_->positions.data(), status_.data(), bool(gradient_),
            cost_.selected ? selected_.data() : nullptr), "Metropolis proposal");
        admitStatus();
        proposed_->state_->live = cost_.population;
        proposed_->state_->ready = true;
      }
      stageValues(*proposed_, false);
      if (gradient_)
      {
        formGreenCorrection(tau);
        green_.upload(host_green_.data(), host_green_.size());
        resetStatus();
        checkRuntime(launchMetropolisStageValues(queue_.getNative(), cost_.population, 0, false,
            staged_cache_->phase.data(), staged_cache_->logabs.data(), accepted_cache_->logabs.data(),
            staged_cache_->phase.data(), staged_cache_->logabs.data(), log_acceptance_.data(), mask_.data(),
            status_.data(), green_.data()), "Metropolis native Green correction");
        admitStatus();
      }
      mask_.download(host_mask_.data(), cost_.population);
      pending_ = true;
      MetropolisProposal token;
      token.source_ = proposed_->state_;
      token.generation_ = proposed_->state_->generation;
      token.version_ = version;
      return token;
    }
    catch (...)
    {
      abortPending();
      throw;
    }
  }

  /// Return native uniform eligibility; consume immediately before the next owner call.
  const std::vector<std::uint32_t>& eligibility(const MetropolisProposal& token) const
  {
    std::lock_guard lock(mutex_);
    validateToken(token);
    return host_mask_;
  }

  /// Resolve only eligible uniforms, with zero handled explicitly by the strict log law.
  /// Merge into staging first, then atomically publish the complete next population.
  const std::vector<std::uint32_t>& resolve(const MetropolisProposal& token, const std::vector<double>& uniforms)
  {
    std::lock_guard lock(mutex_);
    try
    {
      validateToken(token);
      if (uniforms.size() != cost_.population)
        throw std::invalid_argument("Device Metropolis uniform extent mismatch");
      for (std::size_t walker = 0; walker < uniforms.size(); ++walker)
        if ((host_mask_[walker] & METROPOLIS_ELIGIBLE) &&
            (!device_math::isFiniteBinary64(uniforms[walker]) || uniforms[walker] < 0 || uniforms[walker] >= 1))
          throw std::invalid_argument("Device Metropolis eligible uniform must lie in [0,1)");
      uniforms_.upload(uniforms.data(), uniforms.size());
      std::scoped_lock sources(accepted_->state_->mutex, proposed_->state_->mutex);
      if (accepted_->state_->generation == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Device Metropolis accepted generation exhausted");
      beginMutation(*proposed_->state_);
      resetStatus();
      checkRuntime(launchMetropolisResolve(queue_.getNative(), cost_.population, cost_.electrons,
          accepted_->state_->positions.data(), proposed_->state_->positions.data(),
          accepted_cache_->phase.data(), accepted_cache_->logabs.data(), staged_cache_->phase.data(),
          staged_cache_->logabs.data(), uniforms_.data(), log_acceptance_.data(), mask_.data(), status_.data()),
          "Metropolis staged resolution");
      admitStatus();
      mask_.download(host_mask_.data(), cost_.population);
      // No fallible numerical work follows this publication boundary.
      if (gradient_)
        for (std::size_t walker = 0; walker < cost_.population; ++walker)
          if (host_mask_[walker] & METROPOLIS_ACCEPTED)
            std::copy_n(staged_gradient_.data() + walker * 3 * cost_.electrons, 3 * cost_.electrons,
                        accepted_gradient_.data() + walker * 3 * cost_.electrons);
      accepted_->state_->ready = false;
      ++accepted_->state_->generation;
      proposed_->state_->ready = true;
      std::swap(accepted_, proposed_);
      std::swap(accepted_cache_, staged_cache_);
      cache_generation_ = accepted_->state_->generation;
      pending_ = false;
      return host_mask_;
    }
    catch (...)
    {
      abortPending();
      throw;
    }
  }

  /// Borrow current accepted physical coordinates, independently of parameter caches.
  PackedDeviceCoordinatesView acceptedView(std::size_t begin, std::size_t count) const
  {
    std::lock_guard lock(mutex_);
    return accepted_->view(begin, count);
  }

  /// Borrow pending proposal coordinates only under their exact transaction token.
  PackedDeviceCoordinatesView proposedView(const MetropolisProposal& token, std::size_t begin, std::size_t count) const
  {
    std::lock_guard lock(mutex_);
    validateToken(token);
    return proposed_->view(begin, count);
  }

  /// Export coordinates explicitly for host Hamiltonians/checkpoints, not every inference.
  void exportAccepted(std::vector<double>& positions) const
  {
    std::lock_guard lock(mutex_);
    if (positions.size() != cost_.coordinates)
      throw std::invalid_argument("Metropolis accepted export extent mismatch");
    std::lock_guard source_lock(accepted_->state_->mutex);
    accepted_->state_->positions.download(positions.data(), positions.size());
  }

  /// Download only current accepted signed-log caches for validation or explicit consumers.
  void downloadAcceptedValues(std::vector<double>& phase, std::vector<double>& logabs) const
  {
    std::lock_guard lock(mutex_);
    requireCache(graph_.model_->version());
    if (phase.size() != cost_.population || logabs.size() != cost_.population)
      throw std::invalid_argument("Metropolis accepted value extent mismatch");
    accepted_cache_->phase.download(phase.data(), phase.size());
    accepted_cache_->logabs.download(logabs.data(), logabs.size());
  }

private:
  /// Apply the native host policy and retain forward Green sums in physical electron order.
  void formDisplacement(const std::vector<double>& gaussian, double tau)
  {
    std::fill(forward_green_.begin(), forward_green_.end(), 0);
    for (std::size_t walker = 0; walker < cost_.population; ++walker)
      for (std::size_t electron = 0; electron < cost_.electrons; ++electron)
      {
        if (cost_.selected && !host_selected_[walker * cost_.electrons + electron])
          continue;
        const auto index = (walker * cost_.electrons + electron) * 3;
        const double variance = tau * drift_masses_[electron];
        if (!device_math::isFiniteBinary64(variance) || variance <= 0)
          throw std::invalid_argument("Metropolis drift variance is not finite and positive");
        double drift[3], g[3];
        drift_(variance, accepted_gradient_.data() + index, drift);
        for (std::size_t d = 0; d < 3; ++d)
        {
          g[d] = std::sqrt(variance) * gaussian[index + d];
          displacement_[index + d] = g[d] + drift[d];
        }
        forward_green_[walker] -= (g[0]*g[0] + g[1]*g[1] + g[2]*g[2]) / (2 * variance);
      }
  }

  /// Reverse native Green sum uses D+d_proposed, never a reimplemented drift limiter.
  void formGreenCorrection(double tau)
  {
    for (std::size_t walker = 0; walker < cost_.population; ++walker)
    {
      double backward = 0;
      for (std::size_t electron = 0; electron < cost_.electrons; ++electron)
      {
        if (cost_.selected && !host_selected_[walker * cost_.electrons + electron])
          continue;
        const auto index = (walker * cost_.electrons + electron) * 3;
        const double variance = tau * drift_masses_[electron];
        double drift[3], residual[3];
        drift_(variance, staged_gradient_.data() + index, drift);
        for (std::size_t d = 0; d < 3; ++d)
          residual[d] = displacement_[index + d] + drift[d];
        backward -= (residual[0]*residual[0] + residual[1]*residual[1] + residual[2]*residual[2]) / (2 * variance);
      }
      host_green_[walker] = backward;
      host_green_[cost_.population + walker] = forward_green_[walker];
      if (!device_math::isFiniteBinary64(backward) || !device_math::isFiniteBinary64(forward_green_[walker]))
        throw std::runtime_error("Metropolis native Green correction is nonfinite");
    }
  }

  /// Allocate only after full admission; seed cache work is cold, not a hidden proposal.
  PreparedDeviceMetropolis(PreparedDeviceInference& graph, compute::Queue<PlatformKind::CUDA>& queue,
      MetropolisCost cost, const std::vector<double>& positions, const std::vector<double>& inverse_mass)
      : cost_(cost), graph_(graph), queue_(queue),
        accepted_(graph.model_->prepareCoordinates(cost.population, checkedDeviceBytes<double>(cost.coordinates))),
        proposed_(graph.model_->prepareCoordinates(cost.population, checkedDeviceBytes<double>(cost.coordinates))),
        accepted_cache_(std::make_unique<ValueCache>(cost.population)),
        staged_cache_(std::make_unique<ValueCache>(cost.population)), gaussian_(cost.coordinates),
        inverse_mass_(cost.electrons), uniforms_(cost.population), log_acceptance_(cost.population),
        mask_(cost.population), selected_(cost.selected ? cost.coordinates / 3 : 0),
        status_(1), host_mask_(cost.population),
        host_selected_(cost.selected ? cost.coordinates / 3 : 0),
        dense_gaussian_(cost.selected ? cost.coordinates : 0)
  {
    std::size_t bytes = accepted_->deviceWorkspaceBytes() + proposed_->deviceWorkspaceBytes();
    for (const auto* buffer : {&accepted_cache_->phase, &accepted_cache_->logabs, &staged_cache_->phase,
        &staged_cache_->logabs, &gaussian_, &inverse_mass_, &uniforms_, &log_acceptance_})
      bytes += checkedDeviceBytes<double>(buffer->capacity());
    bytes += checkedDeviceBytes<std::uint32_t>(mask_.capacity() + selected_.capacity()) +
        checkedDeviceBytes<std::uint64_t>(status_.capacity());
    const auto host_bytes = checkedDeviceBytes<std::uint32_t>(host_mask_.capacity() + host_selected_.capacity()) +
        checkedDeviceBytes<double>(dense_gaussian_.capacity());
    if (bytes != cost.device_bytes || host_bytes != cost.host_bytes)
      throw std::logic_error("Device Metropolis allocations disagree with exact preflight");
    accepted_->upload(positions);
    inverse_mass_.upload(inverse_mass.data(), inverse_mass.size());
    refreshAccepted(graph.model_->version());
  }

  /// Validate the current numerical image, not a source reference parameter version.
  void validateGraph(std::size_t version) const
  {
    graph_.model_->validateCurrentDevice();
    if (version != graph_.model_->version() || graph_.mixed_dense_ ||
        graph_.model_->metadataFamily().get() != accepted_->state_->family.get())
      throw std::invalid_argument("Device Metropolis graph has stale or foreign model provenance");
  }

  /// Guard the accepted cache against coordinate mutation and explicit model rebinding.
  void requireCache(std::size_t version) const
  {
    validateGraph(version);
    if (!cache_valid_ || cache_version_ != version || cache_generation_ != accepted_->state_->generation)
      throw std::logic_error("Refresh accepted Metropolis values before proposing under this image");
  }

  /// Reject stale, foreign, expired or already resolved pending transactions.
  void validateToken(const MetropolisProposal& token) const
  {
    requireCache(token.version_);
    if (!pending_ || token.source_.lock().get() != proposed_->state_.get() ||
        token.generation_ != proposed_->state_->generation)
      throw std::invalid_argument("Device Metropolis proposal token is stale or foreign");
  }

  /// Revoke old views before any write; the caller holds the real source mutex.
  static void beginMutation(coordinate_detail::SourceState& source)
  {
    source.ready = false;
    if (source.generation == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Device Metropolis proposal generation exhausted");
    ++source.generation;
  }

  /// Discard failed/pending scratch without changing accepted state or throwing on overflow.
  void abortPending() noexcept
  {
    pending_ = false;
    std::lock_guard lock(proposed_->state_->mutex);
    proposed_->state_->ready = false;
    if (proposed_->state_->generation != std::numeric_limits<std::size_t>::max())
      ++proposed_->state_->generation;
  }

  /// Clear ordered scalar diagnostics without a host-to-device status transfer.
  void resetStatus()
  {
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemsetAsync(status_.data(), 0xff, sizeof(std::uint64_t), queue_.getNative()), "Metropolis status reset");
#else
    checkRuntime(cudaMemsetAsync(status_.data(), 0xff, sizeof(std::uint64_t), queue_.getNative()), "Metropolis status reset");
#endif
  }

  /// Admit all queued numerical work before publishing source/cache changes.
  void admitStatus()
  {
    queue_.sync();
    std::uint64_t status;
    status_.download(&status, 1);
    if (status != std::numeric_limits<std::uint64_t>::max())
      throw std::runtime_error("Device Metropolis numerical admission failed, status=" + std::to_string(status));
  }

  /// Reuse a bounded VALUE graph and immediately stage each live signed-log prefix.
  void stageValues(PreparedDeviceCoordinates& source, bool seed)
  {
    resetStatus();
    for (std::size_t offset = 0; offset < cost_.population; offset += graph_.batch_capacity_)
    {
      const auto count = std::min(graph_.batch_capacity_, cost_.population - offset);
      graph_.evaluateDeviceCoordinates(source.view(offset, count));
      if (gradient_)
      {
        gradient_->reverse(graph_); // Any unavailable channel aborts; no node isolation/fallback.
        gradient_scratch_.resize(count * 3 * cost_.electrons);
        gradient_->downloadGradient(gradient_scratch_);
        std::copy(gradient_scratch_.begin(), gradient_scratch_.end(),
                  staged_gradient_.begin() + offset * 3 * cost_.electrons);
      }
      if (!graph_.result_valid_ || graph_.batch_ != count)
        throw std::logic_error("Metropolis VALUE microtile did not publish its complete live result");
      checkRuntime(launchMetropolisStageValues(queue_.getNative(), count, offset, seed,
          graph_.phase_.data(), graph_.output_.data(), accepted_cache_->logabs.data(), staged_cache_->phase.data(),
          staged_cache_->logabs.data(), log_acceptance_.data(), mask_.data(), status_.data()), "Metropolis VALUE staging");
    }
    admitStatus();
  }

  const MetropolisCost cost_;
  PreparedDeviceInference& graph_;
  compute::Queue<PlatformKind::CUDA>& queue_;
  std::unique_ptr<PreparedDeviceCoordinates> accepted_, proposed_;
  std::unique_ptr<ValueCache> accepted_cache_, staged_cache_;
  DeviceBuffer<double> gaussian_, inverse_mass_, uniforms_, log_acceptance_;
  DeviceBuffer<std::uint32_t> mask_, selected_;
  DeviceBuffer<std::uint64_t> status_;
  std::vector<std::uint32_t> host_mask_;
  std::vector<std::uint32_t> host_selected_;
  std::vector<double> dense_gaussian_;
  std::unique_ptr<PreparedDeviceCoordinateGradient> gradient_;
  NativeDriftCallback drift_;
  std::size_t gradient_version_ = 0;
  DeviceBuffer<double> green_;
  std::vector<double> accepted_gradient_, staged_gradient_, gradient_scratch_, displacement_, drift_masses_;
  std::vector<double> forward_green_, host_green_;
  mutable std::mutex mutex_;
  std::size_t cache_version_ = 0, cache_generation_ = 0;
  bool cache_valid_ = false, pending_ = false;
};
} // namespace qmcplusplus::psiformer::device
#endif
