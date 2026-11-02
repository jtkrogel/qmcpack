//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_COMPLETED_SEQUENCE_TIMER_H
#define QMCPLUSPLUS_COMPLETED_SEQUENCE_TIMER_H

#include "Message/Communicate.h"
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#ifdef ENABLE_CUDA
#include "Platforms/CUDA/CUDAruntime.hpp"
#endif

namespace qmcplusplus
{
/** Optional complete-unit sequence wall, independent of native timer thresholds.
 * Configure identical warm/measured counts on every rank. Only the two window
 * boundaries complete device work and synchronize ranks. Intermediate native
 * logging/collectives remain in the interval; final artifact export is outside.
 */
class CompletedSequenceTimer
{
public:
  CompletedSequenceTimer(Communicate& comm, const char* mode, std::size_t steps_per_block,
                         std::size_t available_units)
      : comm_(comm), mode_(mode)
  {
    const char* warm = std::getenv("QMC_COMPLETED_SEQUENCE_WARMUP");
    const char* measured = std::getenv("QMC_COMPLETED_SEQUENCE_MEASURED");
    if (!warm && !measured)
      return;
    if (!warm || !measured)
      throw std::invalid_argument("Completed sequence timing requires both WARMUP and MEASURED counts");
    warm_ = parseCount(warm);
    measured_ = parseCount(measured);
    if (!measured_ || warm_ > std::numeric_limits<std::size_t>::max() - measured_ ||
        available_units < warm_ + measured_ || steps_per_block != 1)
      throw std::invalid_argument("Completed sequence timing requires sufficient single-step units and a positive measured count");
#ifdef ENABLE_SYCL
    throw std::invalid_argument("Completed sequence timing supports host or CUDA/HIP completion only");
#endif
    enabled_ = true;
  }

  void beginUnit()
  {
    if (!enabled_ || started_ || completed_ != warm_)
      return;
    completeDevice();
    comm_.barrier();
    start_ = Clock::now();
    started_ = true;
  }

  /// Call after core work, native scalar/history reporting and health collectives.
  void completeUnit()
  {
    if (!enabled_)
      return;
    ++completed_;
    if (!started_ || reported_ || completed_ != warm_ + measured_)
      return;
    completeDevice();
    comm_.barrier();
    double seconds = std::chrono::duration<double>(Clock::now() - start_).count();
#ifdef HAVE_MPI
    if (comm_.size() > 1 &&
        MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, MPI_DOUBLE, MPI_MAX, comm_.getMPI()) != MPI_SUCCESS)
      throw std::runtime_error("Completed sequence timing maximum reduction failed");
#endif
    reported_ = true;
    if (comm_.rank() == 0)
    {
      std::ostringstream receipt;
      receipt << std::setprecision(17) << "Completed sequence wall: mode=" << mode_
              << " warmup_units=" << warm_ << " measured_units=" << measured_
              << " ranks=" << comm_.size() << " max_rank_seconds=" << seconds
              << " mean_unit_seconds=" << seconds / measured_
              << " complete=yes scope=core_and_native_reporting final_artifacts=excluded"
              << " boundary_device_completions=2 boundary_mpi_barriers=2\n";
      std::cout << receipt.str() << std::flush;
    }
  }

  /// An early native stop must never masquerade as a completed window.
  void reportIncomplete() const
  {
    if (enabled_ && !reported_ && comm_.rank() == 0)
      std::cout << "Completed sequence wall: mode=" << mode_ << " complete=no completed_units="
                << completed_ << " warmup_units=" << warm_ << " measured_units=" << measured_ << std::endl;
  }

private:
  using Clock = std::chrono::steady_clock;
  static std::size_t parseCount(const char* text)
  {
    const std::string_view value(text);
    std::size_t count = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), count);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
      throw std::invalid_argument("Completed sequence timing count must be a nonnegative integer");
    return count;
  }
  static void completeDevice()
  {
#ifdef ENABLE_CUDA
    cudaErrorCheck(cudaDeviceSynchronize(), "Completed sequence boundary device completion failed");
#endif
  }
  Communicate& comm_;
  const char* mode_;
  std::size_t warm_ = 0, measured_ = 0, completed_ = 0;
  bool enabled_ = false, started_ = false, reported_ = false;
  Clock::time_point start_;
};
}
#endif
