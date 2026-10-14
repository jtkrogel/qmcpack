//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_AcceleratorDeviceCollective.cpp
 * @brief Opt-in actual CUDA/HIP buffer and pinned-host MPI transport validation.
 * Run one MPI rank per physical GPU; the startup guard rejects oversubscription.
 */

#include "QMCDrivers/WFTrain/AcceleratorRuntimePlacement.h"
#include "QMCDrivers/WFTrain/AcceleratorCollectiveDeviceCopy.h"
#include "QMCDrivers/WFTrain/HighParameterTraining.h"
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceResources.h"
#include "Message/Communicate.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <limits>

namespace qmcplusplus::wftrain
{

TEST_CASE("Actual device host-staged transport preserves complex sums and recovers",
          "[drivers][wftrain][accelerator][device_runtime][mpi]")
{
  auto& communicator = *OHMMS::Controller;
  agreeAcceleratorRuntimePlacement(communicator, true);
  const auto rank = static_cast<std::size_t>(communicator.rank());
  const auto participants = static_cast<std::size_t>(communicator.size());

  // Physical uniqueness was checked above. Anonymous distinct placement keys
  // now suffice for transport planning; they do not select or schedule devices.
  AcceleratorRankPlanInput local{rank, participants, rank, 0,
      participants > 1 && rank == 1 ? 0 : rank + 1,
      std::nullopt, 5 * sizeof(DerivativeValue), 2, 7, 101, 202, false};
  DistributedParameterReduction preflight(communicator, {2});
  const auto plan = preflight.preflightAcceleratorPlan(local, 5, sizeof(DerivativeValue),
      AcceleratorTransportRequest::HOST_STAGED_MPI, 2);
  compute::Queue<PlatformKind::CUDA> device_queue;
  CudaHipAcceleratorCollectiveCopyQueue copy_queue(device_queue);
  AcceleratorPinnedHostSlots pinned(plan);
  HostStagedAcceleratorReduction reduction(communicator, plan, copy_queue, pinned.descriptors());
  psiformer::device::DeviceBuffer<DerivativeValue> device_values(5);
  std::array<DerivativeValue, 5> values{};

  const double local_factor = participants > 1 && rank == 1 ? 0.0 : static_cast<double>(rank + 1);
  double global_factor = 0;
  for (std::size_t remote = 0; remote < participants; ++remote)
    if (!(participants > 1 && remote == 1))
      global_factor += remote + 1;
  const auto fill_local = [&]() {
    for (std::size_t parameter = 0; parameter < values.size(); ++parameter)
      values[parameter] = local_factor * DerivativeValue(parameter + 1, -0.125 * (parameter + 1));
    device_values.upload(values.data(), values.size());
  };

  // Five elements and chunk size two exercise both pinned slots and a tail.
  fill_local();
  reduction.reduceComplexVector(device_values.data(), values.size(), plan.fingerprint, plan.parameter_version);
  device_values.download(values.data(), values.size());
  for (std::size_t parameter = 0; parameter < values.size(); ++parameter)
  {
    CHECK(values[parameter].real() == Catch::Approx(global_factor * (parameter + 1)));
    CHECK(values[parameter].imag() == Catch::Approx(-0.125 * global_factor * (parameter + 1)));
  }

  // A nonfinite device contribution must reject uniformly after D2H admission,
  // drain the queue, and allow a regenerated complete input to be retried.
  fill_local();
  if (rank == participants - 1)
  {
    values.back() = DerivativeValue(std::numeric_limits<double>::quiet_NaN(), 0);
    device_values.upload(values.data(), values.size());
  }
  CHECK_THROWS(reduction.reduceComplexVector(device_values.data(), values.size(), plan.fingerprint,
                                              plan.parameter_version));
  fill_local();
  CHECK_NOTHROW(reduction.reduceComplexVector(device_values.data(), values.size(), plan.fingerprint,
                                               plan.parameter_version));
  device_values.download(values.data(), values.size());
  CHECK(values.back().real() == Catch::Approx(global_factor * values.size()));
}

} // namespace qmcplusplus::wftrain
