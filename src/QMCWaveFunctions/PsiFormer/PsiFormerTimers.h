//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_TIMERS_H
#define QMCPLUSPLUS_PSIFORMER_TIMERS_H

#include "Utilities/TimerManager.h"
#include "Utilities/NativeTimerScope.h"

namespace qmcplusplus::psiformer
{
/** Production-owner timers; standalone numerical headers do not depend on these.
 * Native timers follow the existing true-OpenMP-master recording convention.
 * GPU scopes measure host calls, including only waits already required by those calls.
 */
struct PsiFormerTimers
{
  PsiFormerTimers() { (void)nativeTimerRecordingThread(); }
  PsiFormerTimers(const PsiFormerTimers&) : PsiFormerTimers() {}

  NewTimer& preparation = createGlobalTimer("PsiFormer::prepare", timer_level_medium);
  NewTimer& value = createGlobalTimer("PsiFormer::VALUE", timer_level_medium);
  NewTimer& gradient = createGlobalTimer("PsiFormer::amplitude_gradient", timer_level_medium);
  NewTimer& vgl = createGlobalTimer("PsiFormer::VGL", timer_level_medium);
  NewTimer& batch_value = createGlobalTimer("PsiFormer::batch_VALUE", timer_level_medium);
  NewTimer& batch_gradient = createGlobalTimer("PsiFormer::batch_amplitude_gradient", timer_level_medium);
  NewTimer& batch_vgl = createGlobalTimer("PsiFormer::batch_VGL", timer_level_medium);
  NewTimer& score_jvp = createGlobalTimer("PsiFormer::score_JVP", timer_level_medium);
  NewTimer& parameter_vjp = createGlobalTimer("PsiFormer::score_local_energy_VJP", timer_level_medium);
  NewTimer& energy_jvp = createGlobalTimer("PsiFormer::local_energy_JVP", timer_level_medium);
  NewTimer& publication = createGlobalTimer("PsiFormer::parameter_publication", timer_level_medium);
  NewTimer& resource = createGlobalTimer("PsiFormer::resource_acquire", timer_level_medium);
  NewTimer& cpu_spatial = createGlobalTimer("PsiFormer::CPU_spatial_complete", timer_level_fine);
  NewTimer& cpu_reverse = createGlobalTimer("PsiFormer::CPU_parameter_reverse_complete", timer_level_fine);
  NewTimer& cpu_kinetic = createGlobalTimer("PsiFormer::CPU_kinetic_parameter_complete", timer_level_fine);
  NewTimer& gpu_call = createGlobalTimer("PsiFormer::GPU_HOST_call", timer_level_fine);
  NewTimer& gpu_evaluate = createGlobalTimer("PsiFormer::GPU_HOST_evaluate_complete", timer_level_fine);
  NewTimer& gpu_download = createGlobalTimer("PsiFormer::GPU_HOST_download_complete", timer_level_fine);
};
}
#endif
