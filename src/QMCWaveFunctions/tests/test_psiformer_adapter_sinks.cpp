//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_adapter_sinks.cpp
 * @brief End-to-end allocation and backend-equivalence tests for scalar PsiFormer adapters.
 */

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "psiformer_test_utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace psiformer_adapter_allocation_audit
{
enum class Form : std::size_t
{
  ORDINARY,
  ARRAY,
  NOTHROW,
  NOTHROW_ARRAY,
  ALIGNED,
  ALIGNED_ARRAY,
  ALIGNED_NOTHROW,
  ALIGNED_NOTHROW_ARRAY,
  COUNT
};

std::atomic<bool> enabled{false};
std::array<std::atomic<std::size_t>, static_cast<std::size_t>(Form::COUNT)> counts{};

void record(Form form) noexcept
{
  if (enabled.load(std::memory_order_relaxed))
    counts[static_cast<std::size_t>(form)].fetch_add(1, std::memory_order_relaxed);
}

void* allocate(std::size_t bytes)
{
  if (void* pointer = std::malloc(bytes == 0 ? 1 : bytes))
    return pointer;
  throw std::bad_alloc();
}

void* allocateAligned(std::size_t bytes, std::size_t alignment)
{
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, bytes == 0 ? 1 : bytes) == 0)
    return pointer;
  throw std::bad_alloc();
}
} // namespace psiformer_adapter_allocation_audit

void* operator new(std::size_t bytes)
{
  psiformer_adapter_allocation_audit::record(
      psiformer_adapter_allocation_audit::Form::ORDINARY);
  return psiformer_adapter_allocation_audit::allocate(bytes);
}

void* operator new[](std::size_t bytes)
{
  psiformer_adapter_allocation_audit::record(
      psiformer_adapter_allocation_audit::Form::ARRAY);
  return psiformer_adapter_allocation_audit::allocate(bytes);
}

void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept
{
  psiformer_adapter_allocation_audit::record(
      psiformer_adapter_allocation_audit::Form::NOTHROW);
  try
  {
    return psiformer_adapter_allocation_audit::allocate(bytes);
  }
  catch (...)
  {
    return nullptr;
  }
}

void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept
{
  psiformer_adapter_allocation_audit::record(
      psiformer_adapter_allocation_audit::Form::NOTHROW_ARRAY);
  try
  {
    return psiformer_adapter_allocation_audit::allocate(bytes);
  }
  catch (...)
  {
    return nullptr;
  }
}

void* operator new(std::size_t bytes, std::align_val_t alignment)
{
  psiformer_adapter_allocation_audit::record(
      psiformer_adapter_allocation_audit::Form::ALIGNED);
  return psiformer_adapter_allocation_audit::allocateAligned(
      bytes, static_cast<std::size_t>(alignment));
}

void* operator new[](std::size_t bytes, std::align_val_t alignment)
{
  psiformer_adapter_allocation_audit::record(
      psiformer_adapter_allocation_audit::Form::ALIGNED_ARRAY);
  return psiformer_adapter_allocation_audit::allocateAligned(
      bytes, static_cast<std::size_t>(alignment));
}

void* operator new(std::size_t bytes,
                   std::align_val_t alignment,
                   const std::nothrow_t&) noexcept
{
  psiformer_adapter_allocation_audit::record(
      psiformer_adapter_allocation_audit::Form::ALIGNED_NOTHROW);
  try
  {
    return psiformer_adapter_allocation_audit::allocateAligned(
        bytes, static_cast<std::size_t>(alignment));
  }
  catch (...)
  {
    return nullptr;
  }
}

void* operator new[](std::size_t bytes,
                     std::align_val_t alignment,
                     const std::nothrow_t&) noexcept
{
  psiformer_adapter_allocation_audit::record(
      psiformer_adapter_allocation_audit::Form::ALIGNED_NOTHROW_ARRAY);
  try
  {
    return psiformer_adapter_allocation_audit::allocateAligned(
        bytes, static_cast<std::size_t>(alignment));
  }
  catch (...)
  {
    return nullptr;
  }
}

void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, const std::nothrow_t&) noexcept { std::free(pointer); }
void operator delete[](void* pointer, const std::nothrow_t&) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept
{
  std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept
{
  std::free(pointer);
}

namespace qmcplusplus
{
namespace
{
using namespace testing::psiformer;
using Value = QMCTraits::ValueType;

struct AllocationSnapshot
{
  std::array<std::size_t,
             static_cast<std::size_t>(psiformer_adapter_allocation_audit::Form::COUNT)> counts{};
};

/// Count every replaceable C++ allocation form only while one warmed call runs.
template<class Function>
AllocationSnapshot auditAllocations(Function&& function)
{
  using namespace psiformer_adapter_allocation_audit;
  for (auto& count : counts)
    count.store(0, std::memory_order_relaxed);
  enabled.store(true, std::memory_order_seq_cst);
  try
  {
    std::forward<Function>(function)();
  }
  catch (...)
  {
    enabled.store(false, std::memory_order_seq_cst);
    throw;
  }
  enabled.store(false, std::memory_order_seq_cst);

  AllocationSnapshot snapshot;
  for (std::size_t form = 0; form < snapshot.counts.size(); ++form)
    snapshot.counts[form] = counts[form].load(std::memory_order_relaxed);
  return snapshot;
}

void checkNoAllocations(const AllocationSnapshot& snapshot)
{
  using Form = psiformer_adapter_allocation_audit::Form;
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::ORDINARY)] == 0);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::ARRAY)] == 0);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::NOTHROW)] == 0);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::NOTHROW_ARRAY)] == 0);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::ALIGNED)] == 0);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::ALIGNED_ARRAY)] == 0);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::ALIGNED_NOTHROW)] == 0);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::ALIGNED_NOTHROW_ARRAY)] == 0);
}

ParticleSet makeElectrons(const SimulationCell& simulation_cell)
{
  const Geometry geometry = makeGeometry("lih");
  ParticleSet electrons(simulation_cell);
  electrons.setName("e");
  electrons.create({2, 2});
  SpeciesSet& species = electrons.getSpeciesSet();
  species.addSpecies("u");
  species.addSpecies("d");
  const int mass = species.addAttribute("mass");
  species(mass, 0) = 1.0;
  species(mass, 1) = 1.0;
  electrons.resetGroups();
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      electrons.R[electron][dimension] = geometry.electrons[3 * electron + dimension];
  electrons.update();
  return electrons;
}

void setBackend(const char* mode)
{
  REQUIRE(setenv("PSIFORMER_VALUE_BACKEND", mode, 1) == 0);
  REQUIRE(setenv("PSIFORMER_SPATIAL_BACKEND", mode, 1) == 0);
}

struct ScalarObservation
{
  double real;
  double imag;
};

template<class T>
ScalarObservation observe(const T& value)
{
  return {std::real(value), std::imag(value)};
}

struct PublicAdapterSnapshot
{
  ScalarObservation accepted_log;
  std::vector<ScalarObservation> gradient;
  std::vector<ScalarObservation> laplacian;
  std::array<ScalarObservation, 3> active_gradient;
  ScalarObservation ratio;
  ScalarObservation ratio_grad;
  std::array<ScalarObservation, 3> proposed_gradient;
  ScalarObservation committed_log;
  ScalarObservation refreshed_log;
  ScalarObservation rejected_log;
};

/// Exercise direct, oracle, or compare through only public component entry points.
PublicAdapterSnapshot evaluatePublicAdapters(const GeneratedFiles& files, const char* backend)
{
  setBackend(backend);
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeElectrons(simulation_cell);
  PsiFormerWF component(
      std::string("pf_adapter_") + backend, files.parameters.string(), files.configuration.string());

  PublicAdapterSnapshot snapshot;
  electrons.G = Value(0);
  electrons.L = Value(0);
  snapshot.accepted_log = observe(component.evaluateLog(electrons, electrons.G, electrons.L));
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      snapshot.gradient.push_back(observe(electrons.G[electron][dimension]));
    snapshot.laplacian.push_back(observe(electrons.L[electron]));
  }

  constexpr int moved_electron = 1;
  const PsiFormerWF::GradType active_gradient = component.evalGrad(electrons, moved_electron);
  for (int dimension = 0; dimension < 3; ++dimension)
    snapshot.active_gradient[dimension] = observe(active_gradient[dimension]);

  const ParticleSet::SingleParticlePos displacement{0.08, -0.03, 0.02};
  electrons.makeMove(moved_electron, displacement);
  snapshot.ratio = observe(component.ratio(electrons, moved_electron));
  component.restore(moved_electron);
  electrons.rejectMove(moved_electron);

  electrons.makeMove(moved_electron, displacement);
  PsiFormerWF::GradType proposed_gradient;
  for (int dimension = 0; dimension < 3; ++dimension)
    proposed_gradient[dimension] = Value(0.125 * (dimension + 1));
  snapshot.ratio_grad = observe(component.ratioGrad(electrons, moved_electron, proposed_gradient));
  for (int dimension = 0; dimension < 3; ++dimension)
    snapshot.proposed_gradient[dimension] = observe(proposed_gradient[dimension]);
  component.acceptMove(electrons, moved_electron, true);
  electrons.acceptMove(moved_electron);
  snapshot.committed_log = observe(component.get_log_value());

  electrons.G = Value(0);
  electrons.L = Value(0);
  snapshot.refreshed_log = observe(component.evaluateLog(electrons, electrons.G, electrons.L));

  const ParticleSet::SingleParticlePos rejected_displacement{-0.025, 0.014, -0.009};
  electrons.makeMove(moved_electron, rejected_displacement);
  component.ratio(electrons, moved_electron);
  component.restore(moved_electron);
  electrons.rejectMove(moved_electron);
  snapshot.rejected_log = observe(component.get_log_value());
  return snapshot;
}

void checkObservation(const ScalarObservation& actual,
                      const ScalarObservation& expected,
                      double tolerance = 3.0e-9)
{
  CHECK(actual.real == Catch::Approx(expected.real).epsilon(tolerance).margin(tolerance));
  CHECK(actual.imag == Catch::Approx(expected.imag).epsilon(tolerance).margin(tolerance));
}

void checkSnapshot(const PublicAdapterSnapshot& actual,
                   const PublicAdapterSnapshot& expected)
{
  checkObservation(actual.accepted_log, expected.accepted_log);
  REQUIRE(actual.gradient.size() == expected.gradient.size());
  REQUIRE(actual.laplacian.size() == expected.laplacian.size());
  for (std::size_t coordinate = 0; coordinate < actual.gradient.size(); ++coordinate)
    checkObservation(actual.gradient[coordinate], expected.gradient[coordinate], 2.0e-8);
  for (std::size_t electron = 0; electron < actual.laplacian.size(); ++electron)
    checkObservation(actual.laplacian[electron], expected.laplacian[electron], 3.0e-8);
  for (int dimension = 0; dimension < 3; ++dimension)
  {
    checkObservation(actual.active_gradient[dimension], expected.active_gradient[dimension], 2.0e-8);
    checkObservation(actual.proposed_gradient[dimension], expected.proposed_gradient[dimension], 2.0e-8);
  }
  checkObservation(actual.ratio, expected.ratio);
  checkObservation(actual.ratio_grad, expected.ratio_grad);
  checkObservation(actual.committed_log, expected.committed_log);
  checkObservation(actual.refreshed_log, expected.refreshed_log);
  checkObservation(actual.rejected_log, expected.rejected_log);
  checkObservation(actual.committed_log, actual.refreshed_log);
  checkObservation(actual.rejected_log, actual.refreshed_log);
}

volatile double allocation_sink = 0.0;

template<class Function>
double medianNanosecondsPerCall(Function&& function)
{
  constexpr int samples = 15;
  constexpr int calls_per_sample = 10;
  std::array<double, samples> timings;
  std::forward<Function>(function)();
  for (double& timing : timings)
  {
    const auto start = std::chrono::steady_clock::now();
    for (int call = 0; call < calls_per_sample; ++call)
      std::forward<Function>(function)();
    timing = std::chrono::duration<double, std::nano>(
                 std::chrono::steady_clock::now() - start)
                 .count() /
        calls_per_sample;
  }
  std::sort(timings.begin(), timings.end());
  return timings[timings.size() / 2];
}
} // namespace

TEST_CASE("PsiFormer warmed scalar direct adapters allocate no owning results",
          "[wavefunction][psiformer][allocation]")
{
  GeneratedFiles files = generateFiles("lih");
  setBackend("direct");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeElectrons(simulation_cell);
  PsiFormerWF component("pf_adapter_alloc", files.parameters.string(), files.configuration.string());

  constexpr int moved_electron = 1;
  const ParticleSet::SingleParticlePos displacement{0.08, -0.03, 0.02};

  // Warm every executor, accepted-state sink, and proposal-state sink before
  // opening any allocation audit window.
  electrons.G = Value(0);
  electrons.L = Value(0);
  component.evaluateLog(electrons, electrons.G, electrons.L);
  component.evalGrad(electrons, moved_electron);
  electrons.makeMove(moved_electron, displacement);
  component.ratio(electrons, moved_electron);
  component.restore(moved_electron);
  electrons.rejectMove(moved_electron);
  electrons.makeMove(moved_electron, displacement);
  PsiFormerWF::GradType warm_gradient;
  warm_gradient = Value(0);
  component.ratioGrad(electrons, moved_electron, warm_gradient);
  component.restore(moved_electron);
  electrons.rejectMove(moved_electron);

  electrons.G = Value(0);
  electrons.L = Value(0);
  const AllocationSnapshot evaluate_log_allocations = auditAllocations([&] {
    for (int repeat = 0; repeat < 3; ++repeat)
      allocation_sink += std::real(component.evaluateLog(electrons, electrons.G, electrons.L));
  });
  checkNoAllocations(evaluate_log_allocations);

  const AllocationSnapshot eval_grad_allocations = auditAllocations([&] {
    for (int repeat = 0; repeat < 3; ++repeat)
      allocation_sink += std::real(component.evalGrad(electrons, moved_electron)[0]);
  });
  checkNoAllocations(eval_grad_allocations);

  electrons.makeMove(moved_electron, displacement);
  const AllocationSnapshot ratio_allocations = auditAllocations([&] {
    for (int repeat = 0; repeat < 3; ++repeat)
      allocation_sink += std::real(component.ratio(electrons, moved_electron));
  });
  component.restore(moved_electron);
  electrons.rejectMove(moved_electron);
  checkNoAllocations(ratio_allocations);

  electrons.makeMove(moved_electron, displacement);
  PsiFormerWF::GradType proposed_gradient;
  proposed_gradient = Value(0);
  const AllocationSnapshot ratio_grad_allocations = auditAllocations([&] {
    for (int repeat = 0; repeat < 3; ++repeat)
      allocation_sink +=
          std::real(component.ratioGrad(electrons, moved_electron, proposed_gradient));
  });
  component.restore(moved_electron);
  electrons.rejectMove(moved_electron);
  checkNoAllocations(ratio_grad_allocations);
  CHECK(std::isfinite(allocation_sink));
}

TEST_CASE("PsiFormer scalar adapter sinks preserve direct oracle and compare results",
          "[wavefunction][psiformer][allocation]")
{
  GeneratedFiles files = generateFiles("lih");
  const PublicAdapterSnapshot oracle = evaluatePublicAdapters(files, "oracle");
  const PublicAdapterSnapshot direct = evaluatePublicAdapters(files, "direct");
  const PublicAdapterSnapshot compare = evaluatePublicAdapters(files, "compare");
  checkSnapshot(direct, oracle);
  checkSnapshot(compare, oracle);
}

// Explicit-only developer measurement used to compare this overlay against its
// Stage-9 baseline.  It is hidden from the ordinary allocation/correctness CTest.
TEST_CASE("PsiFormer scalar adapter sink latency", "[.benchmark][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  setBackend("direct");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeElectrons(simulation_cell);
  PsiFormerWF component("pf_adapter_latency", files.parameters.string(), files.configuration.string());
  constexpr int moved_electron = 1;
  const ParticleSet::SingleParticlePos displacement{0.08, -0.03, 0.02};

  electrons.G = Value(0);
  electrons.L = Value(0);
  const double evaluate_log_ns = medianNanosecondsPerCall([&] {
    allocation_sink += std::real(component.evaluateLog(electrons, electrons.G, electrons.L));
  });
  const double eval_grad_ns = medianNanosecondsPerCall([&] {
    allocation_sink += std::real(component.evalGrad(electrons, moved_electron)[0]);
  });

  electrons.makeMove(moved_electron, displacement);
  const double ratio_ns = medianNanosecondsPerCall([&] {
    allocation_sink += std::real(component.ratio(electrons, moved_electron));
  });
  component.restore(moved_electron);
  electrons.rejectMove(moved_electron);

  electrons.makeMove(moved_electron, displacement);
  PsiFormerWF::GradType proposed_gradient;
  proposed_gradient = Value(0);
  const double ratio_grad_ns = medianNanosecondsPerCall([&] {
    allocation_sink += std::real(component.ratioGrad(electrons, moved_electron, proposed_gradient));
  });
  component.restore(moved_electron);
  electrons.rejectMove(moved_electron);

  std::cout << "{\"evaluateLog_ns\":" << evaluate_log_ns
            << ",\"evalGrad_ns\":" << eval_grad_ns
            << ",\"ratio_ns\":" << ratio_ns
            << ",\"ratioGrad_ns\":" << ratio_grad_ns
            << ",\"sink\":" << allocation_sink << "}\n";
  CHECK(std::isfinite(allocation_sink));
}

} // namespace qmcplusplus
