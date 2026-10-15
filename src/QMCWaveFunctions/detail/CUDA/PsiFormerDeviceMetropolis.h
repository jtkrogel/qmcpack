//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerDeviceMetropolis.h
 * @brief Injected-draw, real/open drift-free whole-system Metropolis state.
 * The borrowed VALUE graph and queue MUST outlive this noncopyable owner. Callers
 * hold their existing model/crowd execution lease, exclude competing graph work
 * and publication during a request, and keep the owning device current. This is
 * not a production sampler, RNG, Hamiltonian or parameter-publication authority.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_METROPOLIS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_METROPOLIS_H

#include "PsiFormerDeviceInference.h"
#include "PsiFormerDeviceMetropolisKernels.h"

namespace qmcplusplus::psiformer::device
{
/// Exact standalone numeric storage; borrowed graph/model/vendor bytes are excluded.
struct MetropolisCost
{
  std::size_t population, electrons, coordinates, device_bytes, host_bytes;
};

/// Check every extent/byte before any owner state or device allocation.
inline MetropolisCost estimateMetropolisCost(std::size_t population, std::size_t electrons)
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
  return {population, electrons, coordinates,
      checkedAdd(checkedAdd(checkedDeviceBytes<double>(doubles), masks, "Metropolis device bytes overflow"),
                 sizeof(std::uint64_t), "Metropolis status bytes overflow"), masks};
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
 * Whole-system moves are independent of all-electron versus ECP Hamiltonians.
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
      const std::vector<double>& inverse_mass, std::size_t workspace_budget_bytes)
  {
    graph.model_->validateCurrentDevice();
    const auto ne = graph.plan_.modelShape().electrons();
    if (graph.mode_ != SpatialJetMode::VALUE_ONLY || graph.mixed_dense_ || graph.retained_targets_ ||
        graph.target_preparation_.max_targets ||
        graph.plan_.environment().boundary != BoundaryCondition::OPEN || &graph.queue_ != &queue ||
        positions.empty() || positions.size() % (3 * ne) != 0 || inverse_mass.size() != ne)
      throw std::invalid_argument("Device Metropolis needs one borrowed FP64/open VALUE graph and packed physical inputs");
    const auto cost = estimateMetropolisCost(positions.size() / (3 * ne), ne);
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

  /// Preserve borrowed references and source identity by forbidding copies.
  PreparedDeviceMetropolis(const PreparedDeviceMetropolis&) = delete;

  /// Never replace an owner with unresolved coordinate/cache state.
  PreparedDeviceMetropolis& operator=(const PreparedDeviceMetropolis&) = delete;

  /// Report checked standalone allocation and host-mask capacities.
  const MetropolisCost& cost() const noexcept { return cost_; }

  /// Recompute accepted values after an explicitly rebound parameter image.
  /// Coordinates survive a failed refresh, but no old cache is current afterward.
  void refreshAccepted(std::size_t version)
  {
    std::lock_guard lock(mutex_);
    abortPending();
    cache_valid_ = false;
    validateGraph(version);
    stageValues(*accepted_, true);
    std::swap(accepted_cache_, staged_cache_);
    cache_version_ = version;
    cache_generation_ = accepted_->state_->generation;
    cache_valid_ = true;
  }

  /// Stage all proposed VALUE microtiles without changing accepted coordinates/cache.
  MetropolisProposal propose(const std::vector<double>& gaussian, double tau, std::size_t version)
  {
    std::lock_guard lock(mutex_);
    const bool was_pending = pending_;
    abortPending();
    if (was_pending)
      throw std::logic_error("Resolve the pending Metropolis proposal before replacing it");
    requireCache(version);
    if (gaussian.size() != cost_.coordinates || !device_math::isFiniteBinary64(tau) || tau <= 0)
      throw std::invalid_argument("Device Metropolis Gaussian extent or timestep is invalid");
    for (const auto value : gaussian)
      if (!device_math::isFiniteBinary64(value))
        throw std::invalid_argument("Device Metropolis Gaussian draw is nonfinite");
    try
    {
      gaussian_.upload(gaussian.data(), gaussian.size());
      {
        std::scoped_lock sources(accepted_->state_->mutex, proposed_->state_->mutex);
        beginMutation(*proposed_->state_);
        resetStatus();
        checkRuntime(launchMetropolisProposal(queue_.getNative(), cost_.population, cost_.electrons,
            accepted_->state_->positions.data(), gaussian_.data(), inverse_mass_.data(), tau,
            proposed_->state_->positions.data(), status_.data()), "Metropolis proposal");
        admitStatus();
        proposed_->state_->live = cost_.population;
        proposed_->state_->ready = true;
      }
      stageValues(*proposed_, false);
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
  /// Allocate only after full admission; seed cache work is cold, not a hidden proposal.
  PreparedDeviceMetropolis(PreparedDeviceInference& graph, compute::Queue<PlatformKind::CUDA>& queue,
      MetropolisCost cost, const std::vector<double>& positions, const std::vector<double>& inverse_mass)
      : cost_(cost), graph_(graph), queue_(queue),
        accepted_(graph.model_->prepareCoordinates(cost.population, checkedDeviceBytes<double>(cost.coordinates))),
        proposed_(graph.model_->prepareCoordinates(cost.population, checkedDeviceBytes<double>(cost.coordinates))),
        accepted_cache_(std::make_unique<ValueCache>(cost.population)),
        staged_cache_(std::make_unique<ValueCache>(cost.population)), gaussian_(cost.coordinates),
        inverse_mass_(cost.electrons), uniforms_(cost.population), log_acceptance_(cost.population),
        mask_(cost.population), status_(1), host_mask_(cost.population)
  {
    std::size_t bytes = accepted_->deviceWorkspaceBytes() + proposed_->deviceWorkspaceBytes();
    for (const auto* buffer : {&accepted_cache_->phase, &accepted_cache_->logabs, &staged_cache_->phase,
        &staged_cache_->logabs, &gaussian_, &inverse_mass_, &uniforms_, &log_acceptance_})
      bytes += checkedDeviceBytes<double>(buffer->capacity());
    bytes += checkedDeviceBytes<std::uint32_t>(mask_.capacity()) + checkedDeviceBytes<std::uint64_t>(status_.capacity());
    if (bytes != cost.device_bytes || host_mask_.capacity() * sizeof(std::uint32_t) != cost.host_bytes)
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
  DeviceBuffer<std::uint32_t> mask_;
  DeviceBuffer<std::uint64_t> status_;
  std::vector<std::uint32_t> host_mask_;
  mutable std::mutex mutex_;
  std::size_t cache_version_ = 0, cache_generation_ = 0;
  bool cache_valid_ = false, pending_ = false;
};
} // namespace qmcplusplus::psiformer::device
#endif
