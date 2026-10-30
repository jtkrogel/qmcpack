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

  // These scopes split driver setup/warmup from recurring work without
  // changing run_steps or training_info.csv meanings.
  NewTimer& initial_evaluation = createGlobalTimer("WFTrain::initial_evaluation", timer_level_fine);
  NewTimer& initial_warmup = createGlobalTimer("WFTrain::initial_warmup", timer_level_fine);
  NewTimer& iteration_warmup = createGlobalTimer("WFTrain::iteration_warmup", timer_level_fine);
  NewTimer& completed_iteration_wall = createGlobalTimer("WFTrain::completed_iteration_wall", timer_level_fine);

  NewTimer& preparation = createGlobalTimer("WFTrain::product_preparation", timer_level_medium);
  NewTimer& objective = createGlobalTimer("WFTrain::objective_gradient", timer_level_medium);
  NewTimer& affine_observation = createGlobalTimer("WFTrain::affine_observation_factors", timer_level_fine);
  NewTimer& kfac_inverse = createGlobalTimer("WFTrain::kfac_factor_inverse_preparation", timer_level_fine);
  NewTimer& reduction = createGlobalTimer("WFTrain::parameter_metric_reduction", timer_level_medium);
  NewTimer& proposal = createGlobalTimer("WFTrain::optimizer_proposal_solve", timer_level_medium);
  NewTimer& candidate = createGlobalTimer("WFTrain::candidate_line_search", timer_level_medium);
  NewTimer& publication = createGlobalTimer("WFTrain::parameter_publication", timer_level_medium);
  NewTimer& refresh = createGlobalTimer("WFTrain::cache_refresh", timer_level_medium);
};
}
#endif
