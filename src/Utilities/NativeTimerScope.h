//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_NATIVE_TIMER_SCOPE_H
#define QMCPLUSPLUS_NATIVE_TIMER_SCOPE_H

#include "Utilities/NewTimer.h"
#include <thread>

namespace qmcplusplus
{
/// Bind production timer registration to its controlling thread before launching workers.
inline const std::thread::id& nativeTimerRecordingThread()
{
  static const std::thread::id recording_thread = std::this_thread::get_id();
  return recording_thread;
}

/** Preserve native stack semantics without touching its global stack from async workers.
 * OpenMP workers retain NewTimer's existing true-master filtering. No lock or wait is added.
 */
class NativeTimerScope
{
public:
  explicit NativeTimerScope(NewTimer& timer)
  {
#ifdef ENABLE_TIMERS
    if (std::this_thread::get_id() == nativeTimerRecordingThread())
    {
      timer_ = &timer;
      timer_->start();
    }
#else
    (void)timer;
#endif
  }

  NativeTimerScope(const NativeTimerScope&) = delete;
  NativeTimerScope& operator=(const NativeTimerScope&) = delete;

  ~NativeTimerScope()
  {
    if (timer_)
      timer_->stop();
  }

private:
  NewTimer* timer_ = nullptr;
};
}
#endif
