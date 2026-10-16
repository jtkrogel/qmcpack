//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_PSIFORMER_INFERENCE_CONCURRENCY_H
#define QMCPLUSPLUS_PSIFORMER_INFERENCE_CONCURRENCY_H

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <utility>

namespace qmcplusplus::psiformer
{

/** Bound concurrent native crowd calls without owning resources or scheduling work.
 * A permit protects one independently prepared context through host result staging.
 * Publication waits outside the backend control mutex for every reader to leave.
 * The caller still owns its native crowd resource until consuming borrowed results.
 */
class PsiFormerInferenceConcurrency
{
public:
  class Lease
  {
  public:
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease& operator=(Lease&&) = delete;
    Lease(Lease&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), publication_(std::move(other.publication_)) {}
    ~Lease()
    {
      if (owner_)
      {
        publication_.unlock();
        owner_->release();
      }
    }

  private:
    friend class PsiFormerInferenceConcurrency;
    Lease(PsiFormerInferenceConcurrency& owner, std::shared_lock<std::shared_mutex> publication)
        : owner_(&owner), publication_(std::move(publication)) {}
    PsiFormerInferenceConcurrency* owner_;
    std::shared_lock<std::shared_mutex> publication_;
  };

  explicit PsiFormerInferenceConcurrency(std::size_t maximum = 1) : maximum_(maximum)
  {
    if (!maximum_ || maximum_ == std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("PsiFormer active inference crowds must be positive and finite");
  }

  Lease acquire()
  {
    {
      std::unique_lock lock(mutex_);
      available_.wait(lock, [&] { return active_ < maximum_; });
      ++active_;
      peak_ = std::max(peak_, active_);
    }
    try
    {
      return Lease(*this, std::shared_lock<std::shared_mutex>(publication_));
    }
    catch (...)
    {
      release();
      throw;
    }
  }

  std::unique_lock<std::shared_mutex> publicationBoundary()
  { return std::unique_lock<std::shared_mutex>(publication_); }

  std::size_t maximum() const noexcept { return maximum_; }
  std::size_t peakAdmittedCalls() const
  {
    std::lock_guard lock(mutex_);
    return peak_;
  }

private:
  void release() noexcept
  {
    std::lock_guard lock(mutex_);
    --active_;
    available_.notify_one();
  }
  const std::size_t maximum_;
  mutable std::mutex mutex_;
  std::condition_variable available_;
  std::size_t active_ = 0, peak_ = 0;
  std::shared_mutex publication_;
};

} // namespace qmcplusplus::psiformer
#endif
