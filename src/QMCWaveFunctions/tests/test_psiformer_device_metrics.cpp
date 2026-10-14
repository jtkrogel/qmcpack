//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include "catch2/catch_test_macros.hpp"
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceMetrics.h"

using namespace qmcplusplus::psiformer::device;

/// Constrain ownership/interval semantics without a GPU or vendor runtime.
TEST_CASE("PsiFormer explicit device memory accounting", "[psiformer][accelerator]")
{
  DeviceMemoryMetrics metrics;
  metrics.allocated(16);
  metrics.allocated(32);
  metrics.freed(16);
  metrics.uploaded(24);
  metrics.downloaded(8);
  auto state = metrics.snapshot();
  CHECK(state.allocation_calls == 2);
  CHECK(state.free_calls == 1);
  CHECK(state.live_bytes == 32);
  CHECK(state.peak_bytes == 48);
  CHECK(state.upload_calls == 1);
  CHECK(state.uploaded_bytes == 24);
  CHECK(state.download_calls == 1);
  CHECK(state.downloaded_bytes == 8);

  // Starting a timing interval must retain the already resident model footprint.
  metrics.resetInterval();
  state = metrics.snapshot();
  CHECK(state.allocation_calls == 0);
  CHECK(state.free_calls == 0);
  CHECK(state.live_bytes == 32);
  CHECK(state.peak_bytes == 32);
  CHECK(state.upload_calls == 0);
  CHECK(state.download_calls == 0);
  metrics.failedFree();
  CHECK(metrics.snapshot().failed_frees == 1);
  CHECK(metrics.snapshot().live_bytes == 32);
  metrics.freed(32);
  CHECK(metrics.snapshot().live_bytes == 0);
}

/// Keep stable identities per vendor-local device without consulting hardware.
TEST_CASE("PsiFormer memory counters separate device ordinals", "[psiformer][accelerator]")
{
  auto& first = deviceMemoryMetrics(0);
  auto& second = deviceMemoryMetrics(1);
  CHECK(&first == &deviceMemoryMetrics(0));
  CHECK(&first != &second);
  CHECK_THROWS_AS(deviceMemoryMetrics(-1), std::invalid_argument);
}
