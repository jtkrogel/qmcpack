//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceNonlocalResponse.h
 * @brief Stream regular-node nonlocal-ECP parameter responses through score VJPs.
 *
 * For complete W_q=bare_q*Psi_total(virtual_q)/Psi_total(reference_q), contract
 * sum_q W_q*(O_virtual_q-O_reference_q). O is this component's canonical score;
 * complete weights must include other fixed wavefunction components' ratios.
 * Quadrature construction and rotations remain caller/CPU Hamiltonian work.
 * Base/virtual total nodes or singular channels are unsupported, never silently
 * treated as zero times an undefined score. Production binding remains separate.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_NONLOCAL_RESPONSE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_NONLOCAL_RESPONSE_H

#include "PsiFormerDeviceScoreReverse.h"

namespace qmcplusplus::psiformer::device
{

/** Own a bounded virtual-tile transaction and O(P) canonical accumulators. */
class PreparedDeviceNonlocalResponse
{
public:
  /// Prepare shared score scratch and fixed base/virtual input capacities.
  PreparedDeviceNonlocalResponse(std::shared_ptr<const PreparedDeviceModel> model,
                                 std::size_t base_capacity, std::size_t virtual_capacity,
                                 compute::Queue<PlatformKind::CUDA>& queue,
                                 AcceleratorBlasHandle& blas)
      : model_(std::move(model)), base_capacity_(base_capacity), virtual_capacity_(virtual_capacity),
        queue_(queue), score_(model_, std::max(base_capacity, virtual_capacity), queue, blas)
  {
    if (base_capacity_ == 0 || virtual_capacity_ == 0)
      throw std::invalid_argument("Nonlocal response requires nonzero base/virtual capacities");
    response_.reserve(model_->plan().parameterCount());
    pending_.reserve(response_.capacity());
    sample_weights_.reserve(base_capacity_);
    base_coefficients_.reserve(base_capacity_);
    virtual_weights_.reserve(virtual_capacity_);
    finite_layout_ = {response_.capacity(), 4096,
        response_.capacity() / 4096 + (response_.capacity() % 4096 != 0), 0, 1};
    finite_status_.reserve(finite_layout_.status_capacity);
    host_finite_status_.resize(finite_layout_.status_capacity);
  }

  /** Start a new captured-grid contraction; caller retains base through finish.
   * Base coordinates may not change during the transaction. Sample weights are
   * arbitrary finite signed loss contractions, independent of complete W_q.
   */
  void begin(const PreparedDeviceInference& base, const std::vector<double>& sample_weights)
  {
    ready_ = false;
    open_ = false;
    const auto count = base.valueTapeBatchCount();
    if (base.modelImage().get() != model_.get() || count > base_capacity_ || sample_weights.size() != count)
      throw std::invalid_argument("Nonlocal response base model/capacity/weight mismatch");
    for (double weight : sample_weights)
      if (!device_math::isFiniteBinary64(weight))
        throw std::invalid_argument("Nonlocal sample contraction weight is nonfinite");
    model_->validateCurrentDevice();
    sample_weights_.assign(sample_weights.begin(), sample_weights.end());
    base_coefficients_.assign(count, 0.0);
    active_pending_ = false;
    checkRuntime(launchDerivativeZero(queue_.getNative(), {response_.capacity(), 0, 1, 0, 1}, response_.data()),
                 "nonlocal response zero");
    base_ = &base;
    base_generation_ = base.evaluationGeneration();
    open_ = true;
  }

  /** Add one bounded virtual tile with explicit reference mapping and complete W_q.
   * Virtual configurations may be reordered and may map repeatedly to one base.
   * Reference-score coefficients accumulate until finish, avoiding base replay
   * per tile and avoiding any Q-by-P table.
   */
  void accumulateVirtualTile(const PreparedDeviceInference& virtual_forward,
                            const std::vector<std::size_t>& reference_indices,
                            const std::vector<double>& complete_weights)
  {
    requireOpen();
    try
    {
      validateBase();
      const auto count = virtual_forward.valueTapeBatchCount();
      if (virtual_forward.modelImage().get() != model_.get() || count > virtual_capacity_ ||
          reference_indices.size() != count || complete_weights.size() != count)
        throw std::invalid_argument("Nonlocal virtual model/capacity/input extent mismatch");
      virtual_weights_.resize(count);
      for (std::size_t point = 0; point < count; ++point)
      {
        const auto reference = reference_indices[point];
        if (reference >= sample_weights_.size() || !device_math::isFiniteBinary64(complete_weights[point]))
          throw std::invalid_argument("Nonlocal complete weight or reference mapping is invalid");
        const double weighted = sample_weights_[reference] * complete_weights[point];
        if (!device_math::isFiniteBinary64(weighted))
          throw std::overflow_error("Nonlocal weighted contraction overflow");
        virtual_weights_[point] = weighted;
        base_coefficients_[reference] -= weighted;
        if (!device_math::isFiniteBinary64(base_coefficients_[reference]))
          throw std::overflow_error("Nonlocal base coefficient overflow");
      }
      add(score_.reverse(virtual_forward, virtual_weights_));
    }
    catch (...)
    {
      open_ = false;
      ready_ = false;
      throw;
    }
  }

  /** Finish with one base-score contraction and publish only a finite result. */
  const double* finish()
  {
    requireOpen();
    try
    {
      validateBase();
      add(score_.reverse(*base_, base_coefficients_));
      checkRuntime(launchDerivativeFiniteScan(queue_.getNative(), finite_layout_, currentResponse().data(), finite_status_.data()),
                   "nonlocal response finite scan");
      queue_.sync();
      finite_status_.download(host_finite_status_.data(), host_finite_status_.size());
      for (const auto& status : host_finite_status_)
        if (status.nonfinite_count != 0)
          throw std::runtime_error("Nonlocal canonical response is nonfinite");
      open_ = false;
      ready_ = true;
      return currentResponse().data();
    }
    catch (...)
    {
      open_ = false;
      ready_ = false;
      throw;
    }
  }

  /// Copy a completed canonical response only after a successful transaction.
  void downloadResponse(std::vector<double>& destination) const
  {
    if (!ready_)
      throw std::logic_error("Nonlocal response has no completed transaction");
    if (destination.size() != response_.capacity())
      throw std::invalid_argument("Nonlocal response destination extent mismatch");
    currentResponse().download(destination.data(), destination.size());
  }

  /// Rebind only outside a captured-grid transaction; old results become unavailable.
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> successor)
  {
    if (open_)
      throw std::logic_error("Cannot rebind an open nonlocal response transaction");
    model_->validateSuccessor(successor);
    score_.rebindModel(successor);
    model_ = std::move(successor);
    base_ = nullptr;
    ready_ = false;
  }

  /// Stream a canonical interval only after a successfully completed transaction.
  void downloadResponseChunk(std::size_t begin, double* destination, std::size_t count) const
  {
    if (!ready_)
      throw std::logic_error("Nonlocal response has no completed transaction");
    if (begin > response_.capacity() || count > response_.capacity() - begin)
      throw std::out_of_range("Nonlocal response chunk exceeds canonical extent");
    currentResponse().download(destination, count, begin);
  }

  /// Account explicit device scratch without the shared image or forward tapes.
  std::size_t deviceWorkspaceBytes() const
  {
    auto bytes = spatial_detail::checkedAdd(score_.deviceWorkspaceBytes(),
        checkedDeviceBytes<double>(response_.capacity()), "Nonlocal response storage overflow");
    bytes = spatial_detail::checkedAdd(bytes, checkedDeviceBytes<double>(pending_.capacity()),
                                      "Nonlocal pending storage overflow");
    return spatial_detail::checkedAdd(bytes, checkedDeviceBytes<DerivativeFiniteStatus>(finite_status_.capacity()),
                                      "Nonlocal finite status storage overflow");
  }

private:
  /// Reject incomplete/reused transaction calls before touching device scratch.
  void requireOpen() const
  {
    if (!open_)
      throw std::logic_error("Nonlocal response transaction is not open");
  }

  /// Reject coordinate replacement even when the parameter model remains identical.
  void validateBase() const
  {
    if (!base_ || base_->evaluationGeneration() != base_generation_ ||
        base_->valueTapeBatchCount() != sample_weights_.size())
      throw std::logic_error("Nonlocal base snapshot changed during captured-grid contraction");
  }

  /// Add one completed score contraction with alternating prepared destinations.
  void add(const double* contribution)
  {
    auto& destination = active_pending_ ? response_ : pending_;
    checkRuntime(launchAddParameterResponses(queue_.getNative(), response_.capacity(), currentResponse().data(),
        contribution, destination.data()), "nonlocal response accumulation");
    active_pending_ = !active_pending_;
  }

  /// Select the published side without moving address-stable prepared buffers.
  const DeviceBuffer<double>& currentResponse() const noexcept
  { return active_pending_ ? pending_ : response_; }

  std::shared_ptr<const PreparedDeviceModel> model_;
  std::size_t base_capacity_, virtual_capacity_, base_generation_ = 0;
  compute::Queue<PlatformKind::CUDA>& queue_;
  PreparedDeviceScoreReverse score_;
  const PreparedDeviceInference* base_ = nullptr;
  DeviceBuffer<double> response_, pending_;
  std::vector<double> sample_weights_, base_coefficients_, virtual_weights_;
  DerivativeFiniteScanLayout finite_layout_;
  DeviceBuffer<DerivativeFiniteStatus> finite_status_;
  std::vector<DerivativeFiniteStatus> host_finite_status_;
  bool open_ = false, ready_ = false;
  bool active_pending_ = false;
};

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_NONLOCAL_RESPONSE_H
