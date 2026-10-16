//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceScoreContraction.h
 * @brief Fixed-capacity complex contractions of serial real device score responses.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_SCORE_CONTRACTION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_SCORE_CONTRACTION_H

#include "PsiFormerDeviceScoreReverse.h"

#include <complex>

namespace qmcplusplus::psiformer::device
{

/** Own O(CP) storage, with C fixed at preparation, never a sample Jacobian.
 * The product session owns model/version admission and its exclusive scratch
 * lease spans begin, every ordered accumulation, finish and all downloads.
 */
class PreparedDeviceScoreContraction
{
  friend class PreparedDeviceProductSession;
  friend class PreparedDeviceAdam;
  friend class PreparedDeviceSrMetric;
public:
  /// Exact numeric device capacity, excluding queue/model and allocator bookkeeping.
  static std::size_t requiredBytes(std::size_t parameters, std::size_t channels)
  {
    if (!parameters || !channels)
      throw std::invalid_argument("Device score contraction requires positive capacity");
    const auto values = spatial_detail::checkedProduct(parameters, channels,
        "Score contraction extent overflow");
    const auto complex_values = spatial_detail::checkedProduct(values, 2, "Score contraction extent overflow");
    return spatial_detail::checkedAdd(checkedDeviceBytes<double>(complex_values),
        checkedDeviceBytes<DerivativeFiniteStatus>(complex_values / 4096 + (complex_values % 4096 != 0)),
        "Score contraction storage overflow");
  }

  PreparedDeviceScoreContraction(std::size_t parameters, std::size_t channels,
                                 compute::Queue<PlatformKind::CUDA>& queue)
      : parameters_(parameters), capacity_(channels), queue_(queue)
  {
    if (parameters == 0 || channels == 0)
      throw std::invalid_argument("Device score contraction requires positive capacity");
    const auto extent = spatial_detail::checkedProduct(parameters, channels,
        "Device score contraction channel extent overflow");
    sums_.reserve(spatial_detail::checkedProduct(extent, 2, "Device score contraction complex extent overflow"));
    finite_layout_ = {sums_.capacity(), 4096,
        sums_.capacity() / 4096 + (sums_.capacity() % 4096 != 0), 0, 1};
    finite_status_.reserve(finite_layout_.status_capacity);
    host_status_.resize(finite_layout_.status_capacity);
  }

  /// Revoke both partial and completed results without reallocating storage.
  void invalidate() noexcept
  {
    channels_ = samples_ = 0;
    accumulating_ = ready_ = false;
  }

  /// Start a fresh transaction, clearing every prepared channel before reuse.
  void begin(std::size_t channels)
  {
    invalidate();
    if (channels == 0 || channels > capacity_)
      throw std::invalid_argument("Device score contraction channel count exceeds preparation");
    checkRuntime(launchDerivativeZero(queue_.getNative(), {sums_.capacity(), 0, 1, 0, 1}, sums_.data()),
                 "score contraction zero");
    channels_ = channels;
    accumulating_ = true;
  }

  /// Add algebraic complex coefficients times one completed real score.
  void accumulate(const double* response, const std::complex<double>* coefficients, std::size_t count)
  {
    if (!accumulating_ || !response || !coefficients || count != channels_)
      throw std::logic_error("Device score contraction has no matching active transaction");
    for (std::size_t channel = 0; channel < count; ++channel)
      if (!device_math::isFiniteBinary64(coefficients[channel].real()) ||
          !device_math::isFiniteBinary64(coefficients[channel].imag()))
        throw std::invalid_argument("Device score contraction coefficient is nonfinite");
    for (std::size_t channel = 0; channel < count; ++channel)
      checkRuntime(launchRealToComplexAxpy(queue_.getNative(),
          {parameters_, 0, 1, 2 * channel * parameters_, 2},
          coefficients[channel].real(), coefficients[channel].imag(), response, sums_.data()),
          "score complex contraction");
    ++samples_;
  }

  /// Admit all channels together; finite input scores can still overflow a sum.
  void finish()
  {
    ready_ = false;
    if (!accumulating_ || samples_ == 0)
      throw std::logic_error("Device score contraction has no accumulated samples");
    accumulating_ = false;
    checkRuntime(launchDerivativeFiniteScan(queue_.getNative(), finite_layout_, sums_.data(), finite_status_.data()),
                 "score contraction finite scan");
    queue_.sync();
    finite_status_.download(host_status_.data(), host_status_.size());
    for (const auto& status : host_status_)
      if (status.nonfinite_count != 0)
        throw std::runtime_error("Device complex score contraction is nonfinite");
    ready_ = true;
  }

  /// Download once per channel after admission, with no response-size host staging.
  void download(std::size_t channel, std::complex<double>* destination, std::size_t count) const
  {
    if (!ready_)
      throw std::logic_error("Device score contraction has no completed result");
    if (!destination || count != parameters_ || channel >= channels_)
      throw std::invalid_argument("Device score contraction download extent mismatch");
    // C++ guarantees interleaved real/imaginary array access for std::complex<T>.
    sums_.download(reinterpret_cast<double*>(destination), 2 * count, 2 * channel * parameters_);
  }

  std::size_t deviceWorkspaceBytes() const
  {
    return spatial_detail::checkedAdd(checkedDeviceBytes<double>(sums_.capacity()),
        checkedDeviceBytes<DerivativeFiniteStatus>(finite_status_.capacity()), "Score contraction storage overflow");
  }

  std::size_t hostWorkspaceBytes() const
  {
    return checkedDeviceBytes<DerivativeFiniteStatus>(host_status_.capacity());
  }

  std::size_t channelCapacity() const noexcept { return capacity_; }

private:
  const std::size_t parameters_, capacity_;
  compute::Queue<PlatformKind::CUDA>& queue_;
  DeviceBuffer<double> sums_;
  DeviceBuffer<DerivativeFiniteStatus> finite_status_;
  std::vector<DerivativeFiniteStatus> host_status_;
  DerivativeFiniteScanLayout finite_layout_;
  std::size_t channels_ = 0, samples_ = 0;
  bool accumulating_ = false, ready_ = false;
};

} // namespace qmcplusplus::psiformer::device

#endif
