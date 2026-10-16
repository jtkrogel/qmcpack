//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include "QMCWaveFunctions/PsiFormer/PsiFormerInferenceConcurrency.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorExecutor.h"
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <exception>
#include <thread>
#include <vector>

using namespace qmcplusplus::psiformer;

TEST_CASE("PsiFormer inference permits preserve defaults and exceptional release",
          "[psiformer][inference_concurrency]")
{
  CHECK_THROWS_AS(PsiFormerInferenceConcurrency(0), std::invalid_argument);
  CHECK_THROWS_AS(PsiFormerInferenceConcurrency(std::numeric_limits<std::size_t>::max()), std::invalid_argument);
  PsiFormerInferenceConcurrency gate;
  CHECK(gate.maximum() == 1);
  CHECK(gate.peakAdmittedCalls() == 0);
  CHECK_THROWS_AS(([&] {
    auto first = gate.acquire();
    auto moved = std::move(first);
    throw std::runtime_error("injected numerical failure");
  })(), std::runtime_error);
  {
    auto reused = gate.acquire();
    CHECK(gate.peakAdmittedCalls() == 1);
  }
  auto writer = gate.publicationBoundary();
  CHECK(writer.owns_lock());
}

TEST_CASE("PsiFormer inference permits bound independent callers",
          "[psiformer][inference_concurrency]")
{
  PsiFormerInferenceConcurrency gate(2);
  std::mutex mutex;
  std::condition_variable condition;
  std::size_t inside = 0, peak = 0, entered = 0;
  bool release = false;
  std::vector<std::thread> workers;
  for (int i = 0; i < 6; ++i)
    workers.emplace_back([&] {
      auto permit = gate.acquire();
      std::unique_lock lock(mutex);
      ++inside;
      ++entered;
      peak = std::max(peak, inside);
      condition.notify_all();
      condition.wait(lock, [&] { return release; });
      --inside;
    });
  {
    std::unique_lock lock(mutex);
    condition.wait(lock, [&] { return entered == 2; });
    CHECK(inside == 2);
    release = true;
    condition.notify_all();
  }
  for (auto& worker : workers)
    worker.join();
  CHECK(entered == 6);
  CHECK(inside == 0);
  CHECK(peak == 2);
  CHECK(gate.peakAdmittedCalls() == 2);
}

TEST_CASE("PsiFormer publication excludes admitted inference readers",
          "[psiformer][inference_concurrency]")
{
  PsiFormerInferenceConcurrency gate(2);
  std::mutex mutex;
  std::condition_variable condition;
  std::size_t readers = 0;
  bool release = false, attempted = false, published = false;
  std::vector<std::thread> workers;
  for (int i = 0; i < 2; ++i)
    workers.emplace_back([&] {
      auto permit = gate.acquire();
      std::unique_lock lock(mutex);
      ++readers;
      condition.notify_all();
      condition.wait(lock, [&] { return release; });
    });
  {
    std::unique_lock lock(mutex);
    condition.wait(lock, [&] { return readers == 2; });
  }
  std::thread writer([&] {
    {
      std::lock_guard lock(mutex);
      attempted = true;
      condition.notify_all();
    }
    auto boundary = gate.publicationBoundary();
    std::lock_guard lock(mutex);
    published = true;
  });
  {
    std::unique_lock lock(mutex);
    condition.wait(lock, [&] { return attempted; });
    CHECK_FALSE(published);
    release = true;
    condition.notify_all();
  }
  for (auto& worker : workers)
    worker.join();
  writer.join();
  CHECK(published);
}

TEST_CASE("PsiFormer concurrent executor readers preserve lease and publication provenance",
          "[psiformer][inference_concurrency]")
{
  class Recorder final : public PsiFormerAcceleratorHooks
  {
  public:
    void submit(const PsiFormerAcceleratorEvent&) override { ++events; }
    std::size_t events = 0;
  };
  constexpr std::uint64_t layout = 1234;
  const auto operations = acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES);
  PsiFormerAcceleratorSharedModel model(PsiFormerAcceleratorBackend::CUDA, layout, 1);
  PsiFormerInferenceConcurrency gate(2);
  std::mutex control, errors_mutex;
  std::exception_ptr error;
  std::atomic<std::size_t> completed{0}, wrong{0};
  const auto retain_error = [&] {
    std::lock_guard lock(errors_mutex);
    error = std::current_exception();
  };
  std::vector<std::thread> workers;
  for (std::uint64_t identity = 1; identity <= 2; ++identity)
    workers.emplace_back([&, identity] {
      try
      {
        Recorder recorder;
        PsiFormerAcceleratorExecutor executor(PsiFormerAcceleratorBackend::CUDA, operations);
        PsiFormerAcceleratorCrowdState crowd(identity);
        executor.prepare(crowd, layout, 1, recorder);
        for (std::size_t iteration = 1; iteration <= 500; ++iteration)
        {
          auto permit = gate.acquire();
          {
            std::lock_guard lock(control);
            executor.acquire(model, crowd, recorder);
          }
          const auto result = executor.evaluate(model, crowd, operations, 1, iteration, recorder);
          executor.wait(crowd, result.result_generation, recorder);
          const auto& ready = crowd.completedResult(result.result_generation, iteration);
          if (ready.model.model_version != result.model.model_version || ready.crowd_identity != identity)
            wrong.fetch_add(1);
          {
            std::lock_guard lock(control);
            executor.release(model, crowd, recorder);
          }
          completed.fetch_add(1);
        }
      }
      catch (...)
      {
        retain_error();
      }
    });
  std::thread writer([&] {
    try
    {
      Recorder recorder;
      for (std::size_t version = 2; version <= 11; ++version)
      {
        auto boundary = gate.publicationBoundary();
        std::lock_guard lock(control);
        model.stage(layout, version, recorder);
        model.complete(version, recorder);
      }
    }
    catch (...)
    {
      retain_error();
    }
  });
  for (auto& worker : workers)
    worker.join();
  writer.join();
  REQUIRE_FALSE(error);
  CHECK(completed.load() == 1000);
  CHECK(wrong.load() == 0);
  CHECK(model.activeProvenance().model_version == 11);
}
