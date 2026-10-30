//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include "QMCDrivers/WFTrain/TrainingWorkCounters.h"

#include "catch.hpp"

namespace qmcplusplus::wftrain
{
TEST_CASE("WFTrain work counters preserve requested and computed product counts", "[wftrain][counters]")
{
  TrainingWorkCounters counters;
  counters.addProposalRounds(3);
  counters.addRatioApiBatches(5);
  counters.addDriftGradientApiBatches(2);
  counters.addMoveDecisions(7, 4);
  counters.addRequestedProducts(true, false, false);
  counters.addComputedProducts(true, true, true); // Compatibility FULL work stays visible.
  counters.addUnknownComputedProducts();
  counters.addHamiltonianEvaluations(12, 8);
  counters.addRetainedConfigurations(10);

  const auto counts = counters.snapshotAndReset();
  CHECK(counts.proposal_rounds == 3);
  CHECK(counts.ratio_api_batches == 5);
  CHECK(counts.drift_gradient_api_batches == 2);
  CHECK(counts.accepted_move_decisions == 7);
  CHECK(counts.rejected_move_decisions == 4);
  CHECK(counts.requested_value_product_batches == 1);
  CHECK(counts.requested_gradient_product_batches == 0);
  CHECK(counts.computed_value_product_batches == 1);
  CHECK(counts.computed_gradient_product_batches == 1);
  CHECK(counts.computed_laplacian_product_batches == 1);
  CHECK(counts.computed_product_status_unknown_batches == 1);
  CHECK(counts.hamiltonian_walker_evaluations == 12);
  CHECK(counts.ecp_bearing_hamiltonian_walker_evaluations == 8);
  CHECK(counts.retained_configurations == 10);
  const auto empty = counters.snapshotAndReset();
  CHECK(empty.proposal_rounds == 0);
  CHECK(empty.computed_laplacian_product_batches == 0);
}
}
