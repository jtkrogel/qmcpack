//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_ADAM_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_ADAM_H
#include "PsiFormerDeviceAdamKernels.h"
#include "PsiFormerDeviceResources.h"
#include "PsiFormerDeviceScoreContraction.h"
#include "Platforms/CUDA/QueueCUDA.hpp"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"
#include <cassert>
#include <cmath>
#include <memory>
#include <vector>

namespace qmcplusplus::psiformer::device
{
/// Check exact standalone device storage before queue/State/buffer construction.
inline std::size_t estimateAdamBytes(std::size_t parameters, std::size_t blocks)
{
  const auto metadata = checkedDeviceBytes<AdamBlock>(blocks);
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (!parameters || !blocks || metadata > maximum - 8 || parameters > (maximum - metadata - 8) / 48 ||
      parameters - 1 > (std::numeric_limits<std::uint64_t>::max() >> 8) ||
      (parameters - 1) / 128 >= std::size_t(std::numeric_limits<int>::max()))
    throw std::overflow_error("Device Adam storage/index extent is invalid");
  return 48 * parameters + metadata + 8;
}

/** Committed device moment authority, never a parameter/model publication owner.
 * Host global gradient/old parameters upload, raw candidate downloads. The
 * caller keeps this owning device current; accept/reject never execute GPU work.
 * Internal FirstOrderOptimizer adapter: finite seed/parameters/gradient and
 * validated beta/correction/epsilon scalars are its caller's preconditions.
 */
class PreparedDeviceAdam
{
  friend class PreparedDeviceProductSession;
  /// Keep each moment pair address-stable while swapping host ownership.
  struct Moments
  {
    /// Allocate exactly one first/second pair.
    explicit Moments(std::size_t count) : first(count), second(count) {}

    DeviceBuffer<double> first, second;
  };

public:
  /// Lazy ENERGY-only norm capacity; existing status and P scratch are reused.
  static std::size_t energyGradientRequiredBytes(std::size_t count)
  {
    estimateAdamBytes(count, 1);
    return checkedDeviceBytes<double>(2 * ((count - 1) / 128 + 1));
  }

  /// Explicit admission leaves ordinary host-gradient preparation unchanged.
  void prepareEnergyGradient(std::size_t budget)
  {
    validateDevice();
    const auto bytes = energyGradientRequiredBytes(count_);
    if (bytes > budget)
      throw std::length_error("Device Adam ENERGY norm scratch exceeds its byte budget");
    norm_host_.resize(bytes / sizeof(double));
    norm_partials_.reserve(norm_host_.size());
  }

  /// Report only the optional norm owner's exact allocated device payload.
  std::size_t energyGradientDeviceBytes() const noexcept
  {
    return sizeof(double) * norm_partials_.capacity();
  }

  /// Report the separately charged small host norm-reduction staging capacity.
  std::size_t energyGradientHostBytes() const noexcept
  {
    return sizeof(double) * norm_host_.capacity();
  }

  /// Validate compact complete schema and exact budget before any device owner.
  static std::unique_ptr<PreparedDeviceAdam> prepare(std::size_t count,
      const std::vector<AdamBlock>& blocks, std::size_t budget)
  {
    const auto bytes = estimateAdamBytes(count, blocks.size());
    if (bytes > budget)
      throw std::length_error("Device Adam exceeds its explicit per-optimizer byte budget");
    std::size_t end = 0;
    for (const auto& block : blocks)
    {
      if (block.begin != end || !block.count || block.count > count - end || block.trainable > 1 ||
          !device_math::isFiniteBinary64(block.learning_rate) ||
          (block.trainable ? block.learning_rate <= 0 : block.learning_rate != 0))
        throw std::invalid_argument("Device Adam blocks must exactly cover the ordered schema");
      end += block.count;
    }
    if (end != count)
      throw std::invalid_argument("Device Adam blocks do not cover every parameter");
    return std::unique_ptr<PreparedDeviceAdam>(new PreparedDeviceAdam(count, blocks, bytes));
  }

  /// Seed validated host checkpoint/initial mirrors into existing moment storage.
  void seed(const std::vector<double>& first, const std::vector<double>& second)
  {
    validateDevice();
    seeded_ = pending_ = false;
    if (first.size() != count_ || second.size() != count_)
      throw std::invalid_argument("Device Adam seed moment extent mismatch");
    committed_->first.upload(first.data(), count_);
    committed_->second.upload(second.data(), count_);
    seeded_ = true;
  }

  /// Host corrections are the existing exact -expm1 result, not device pow.
  void propose(const std::vector<double>& parameters, const double* gradient,
      double beta1, double beta2, double correction1, double correction2, double epsilon,
      std::vector<double>& candidate)
  {
    validateDevice();
    pending_ = false;
    if (!seeded_ || parameters.size() != count_ || candidate.size() != count_ || !gradient)
      throw std::invalid_argument("Device Adam inputs/seed are incomplete");
    // Two explicit P uploads: host canonical parameters and global host gradient.
    parameters_.upload(parameters.data(), count_);
    gradient_candidate_.upload(gradient, count_);
    completeProposal(beta1, beta2, correction1, correction2, epsilon, candidate);
  }

private:
  /// Only a locked, current ProductSession can supply resident ordered C3 storage.
  double proposeEnergy(const PreparedDeviceScoreContraction& source, const double* image_parameters,
      const std::vector<double>& parameters, double weight, double energy, double beta1, double beta2,
      double correction1, double correction2, double epsilon, std::vector<double>& candidate)
  {
    validateDevice();
    pending_ = false;
    if (!seeded_ || parameters.size() != count_ || candidate.size() != count_ || !image_parameters ||
        !source.ready_ || source.channels_ != 3 || source.parameters_ != count_ ||
        norm_host_.size() != energyGradientRequiredBytes(count_) / sizeof(double) ||
        norm_partials_.capacity() != norm_host_.size() ||
        !device_math::isFiniteBinary64(weight) || weight <= 0 || !device_math::isFiniteBinary64(energy) ||
        !device_math::isFiniteBinary64(beta1) || beta1 < 0 || beta1 >= 1 ||
        !device_math::isFiniteBinary64(beta2) || beta2 < 0 || beta2 >= 1 ||
        !device_math::isFiniteBinary64(correction1) || correction1 <= 0 || correction1 > 1 ||
        !device_math::isFiniteBinary64(correction2) || correction2 <= 0 || correction2 > 1 ||
        !device_math::isFiniteBinary64(epsilon) || epsilon <= 0)
      throw std::invalid_argument("Device ENERGY Adam inputs, C3 provenance or preparation are invalid");
    parameters_.upload(parameters.data(), count_);
    resetStatus();
    checkRuntime(launchAdamEnergyGradient(queue_.getNative(), count_, source.sums_.data(), parameters_.data(),
        image_parameters, weight, energy, gradient_candidate_.data(), norm_partials_.data(), status_.data()),
        "Device ENERGY gradient");
    admitStatus();
    norm_partials_.download(norm_host_.data(), norm_host_.size());
    double norm = 0;
    for (std::size_t index = 0; index < norm_host_.size(); index += 2)
      norm = std::hypot(norm, norm_host_[index] * std::sqrt(norm_host_[index + 1]));
    if (!device_math::isFiniteBinary64(norm))
      throw std::overflow_error("Device ENERGY gradient norm is nonfinite");
    completeProposal(beta1, beta2, correction1, correction2, epsilon, candidate);
    return norm;
  }

  /// Share precisely the old recurrence, status admission and candidate transfer.
  void completeProposal(double beta1, double beta2, double correction1, double correction2,
      double epsilon, std::vector<double>& candidate)
  {
    resetStatus();
    checkRuntime(launchAdamCandidate(queue_.getNative(), count_, blocks_.data(), blocks_.capacity(),
        parameters_.data(), gradient_candidate_.data(), committed_->first.data(), committed_->second.data(),
        staged_->first.data(), staged_->second.data(), beta1, beta2, correction1, correction2, epsilon,
        status_.data()), "Device Adam candidate");
    admitStatus();
    gradient_candidate_.download(candidate.data(), count_);
    pending_ = true;
  }

  /// The same reusable status word covers gradient and recurrence admission.
  void resetStatus()
  {
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemsetAsync(status_.data(), 0xff, 8, queue_.getNative()), "Adam status reset");
#else
    checkRuntime(cudaMemsetAsync(status_.data(), 0xff, 8, queue_.getNative()), "Adam status reset");
#endif
  }

  /// Drain the owning queue before inspecting its compact failure word.
  void admitStatus()
  {
    queue_.sync();
    std::uint64_t status;
    status_.download(&status, 1);
    if (status != std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("Device Adam pending recurrence failed, status=" + std::to_string(status));
  }

public:
  /// Publish already admitted moments with no runtime call, copy or allocation.
  void accept() noexcept
  {
    assert(pending_);
    committed_.swap(staged_);
    pending_ = false;
  }

  /// Discard only speculative scratch after selection/publication rejection.
  void reject() noexcept { pending_ = false; }

  /// Restore commits only invalidate the seed; next propose uploads host mirrors.
  void invalidateSeed() noexcept { seeded_ = pending_ = false; }

  /// Report whether a restored host payload still needs its cold upload.
  bool seeded() const noexcept { return seeded_; }

  /// Synchronize committed moments only at explicit inspection/checkpoint boundaries.
  void download(std::vector<double>& first, std::vector<double>& second) const
  {
    validateDevice();
    if (!seeded_ || first.size() != count_ || second.size() != count_)
      throw std::invalid_argument("Device Adam committed mirror extent/seed mismatch");
    committed_->first.download(first.data(), count_);
    committed_->second.download(second.data(), count_);
  }

  /// Report the unchanged base footprint; optional ENERGY norm bytes are separate.
  std::size_t deviceBytes() const noexcept { return bytes_; }

private:
  /// Allocate admitted buffers and upload only compact immutable metadata.
  PreparedDeviceAdam(std::size_t count, const std::vector<AdamBlock>& blocks, std::size_t bytes)
      : count_(count), bytes_(bytes), committed_(std::make_unique<Moments>(count)),
        staged_(std::make_unique<Moments>(count)), parameters_(count), gradient_candidate_(count),
        blocks_(blocks.size()), status_(1)
  {
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&device_), "Adam owning device");
#else
    checkRuntime(cudaGetDevice(&device_), "Adam owning device");
#endif
    blocks_.upload(blocks.data(), blocks.size());
  }

  /// Reject a changed current device before touching owner buffers or its queue.
  void validateDevice() const
  {
    int current = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&current), "Adam current device");
#else
    checkRuntime(cudaGetDevice(&current), "Adam current device");
#endif
    if (current != device_)
      throw std::invalid_argument("Device Adam owner belongs to a different current device");
  }

  const std::size_t count_, bytes_;
  compute::Queue<PlatformKind::CUDA> queue_;
  std::unique_ptr<Moments> committed_, staged_;
  DeviceBuffer<double> parameters_, gradient_candidate_;
  DeviceBuffer<AdamBlock> blocks_;
  DeviceBuffer<std::uint64_t> status_;
  DeviceBuffer<double> norm_partials_;
  std::vector<double> norm_host_;
  bool seeded_ = false, pending_ = false;
  int device_ = -1;
};
} // namespace qmcplusplus::psiformer::device
#endif
