//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_TRAINING_TIMERS_H
#define QMCPLUSPLUS_TRAINING_TIMERS_H

#include "Utilities/TimerManager.h"
#include "Utilities/NativeTimerScope.h"

namespace qmcplusplus::wftrain
{
/// Register fixed names at owner construction, never during a product or collective.
struct TrainingTimers
{
  TrainingTimers() { (void)nativeTimerRecordingThread(); }
  TrainingTimers(const TrainingTimers&) : TrainingTimers() {}

  NewTimer& preparation = createGlobalTimer("WFTrain::product_preparation", timer_level_medium);
  NewTimer& objective = createGlobalTimer("WFTrain::objective_gradient", timer_level_medium);
  NewTimer& affine_observation = createGlobalTimer("WFTrain::affine_observation_factors", timer_level_fine);
  NewTimer& reduction = createGlobalTimer("WFTrain::parameter_metric_reduction", timer_level_medium);
  NewTimer& proposal = createGlobalTimer("WFTrain::optimizer_proposal_solve", timer_level_medium);
  NewTimer& candidate = createGlobalTimer("WFTrain::candidate_line_search", timer_level_medium);
  NewTimer& publication = createGlobalTimer("WFTrain::parameter_publication", timer_level_medium);
  NewTimer& refresh = createGlobalTimer("WFTrain::cache_refresh", timer_level_medium);
};
}
#endif
