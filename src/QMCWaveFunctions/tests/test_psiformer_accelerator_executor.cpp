//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_accelerator_executor.cpp
 * @brief Deterministic tests for the PsiFormer accelerator production control plane.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorExecutor.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <thread>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace qmcplusplus::psiformer
{
namespace
{

/// Record a bounded operation stream without allocating in warmed control-plane calls.
class BoundedRecorder final : public PsiFormerAcceleratorHooks
{
public:
  /// Append one event, optionally injecting a backend failure at a selected boundary.
  void submit(const PsiFormerAcceleratorEvent& event) override
  {
    if (failure_kind_ && *failure_kind_ == event.kind)
      throw std::runtime_error("injected accelerator hook failure");
    if (count_ == events_.size())
      throw std::length_error("test accelerator event recorder is full");
    events_[count_++] = event;
  }

  /// Arrange for the next matching event kind to throw before it is recorded.
  void failOn(PsiFormerAcceleratorEventKind kind) noexcept { failure_kind_ = kind; }

  /// Disable failure injection.
  void clearFailure() noexcept { failure_kind_.reset(); }

  /// Remove recorded events while retaining fixed-capacity storage.
  void clear() noexcept { count_ = 0; }

  /// Return the number of submitted events.
  std::size_t size() const noexcept { return count_; }

  /// Return one recorded event.
  const PsiFormerAcceleratorEvent& operator[](std::size_t index) const
  {
    if (index >= count_)
      throw std::out_of_range("accelerator event index is out of range");
    return events_[index];
  }

private:
  std::array<PsiFormerAcceleratorEvent, 64> events_{};
  std::size_t count_ = 0;
  std::optional<PsiFormerAcceleratorEventKind> failure_kind_;
};

constexpr std::uint64_t LAYOUT = UINT64_C(0x4d4f44454c);

} // namespace

TEST_CASE("PsiFormer independent lanes preserve every concurrent image lease",
          "[wavefunction][psiformer][accelerator][concurrency]")
{
  constexpr auto backend = PsiFormerAcceleratorBackend::CUDA;
  PsiFormerAcceleratorExecutor executor(backend, acceleratorOperationBit(PsiFormerAcceleratorOperation::PARAMETER_SCORE));
  // Repeated bursts exercise acquire/release contention without publishing while
  // readers hold shared authority. Each lane has independent hook/crowd state.
  for (int burst = 0; burst < 64; ++burst)
  {
    PsiFormerAcceleratorSharedModel model(backend, LAYOUT, 1);
    std::array<std::thread, 8> workers;
    std::atomic<std::size_t> acquired{0}, failures{0};
    std::atomic<bool> start{false}, release{false};
    for (std::size_t i = 0; i < workers.size(); ++i)
      workers[i] = std::thread([&, i] {
        BoundedRecorder hooks;
        PsiFormerAcceleratorCrowdState crowd(i + 1);
        try
        {
          executor.prepare(crowd, LAYOUT, 1, hooks);
          while (!start.load()) std::this_thread::yield();
          executor.acquire(model, crowd, hooks);
        }
        catch (...) { ++failures; }
        ++acquired;
        while (!release.load()) std::this_thread::yield();
        try { executor.release(model, crowd, hooks); }
        catch (...) { ++failures; }
      });
    start = true;
    while (acquired.load() != workers.size()) std::this_thread::yield();
    release = true;
    for (auto& worker : workers) worker.join();
    CHECK(failures.load() == 0);
    // Every lease has drained, so both image slots must be reusable.
    BoundedRecorder publication;
    model.stage(LAYOUT, 2, publication);
    model.complete(2, publication);
    CHECK_NOTHROW(model.stage(LAYOUT, 3, publication));
  }
}

TEST_CASE("PsiFormer accelerator state owners are address stable",
          "[wavefunction][psiformer][accelerator]")
{
  CHECK_FALSE(std::is_copy_constructible_v<PsiFormerAcceleratorSharedModel>);
  CHECK_FALSE(std::is_move_constructible_v<PsiFormerAcceleratorSharedModel>);
  CHECK_FALSE(std::is_copy_assignable_v<PsiFormerAcceleratorSharedModel>);
  CHECK_FALSE(std::is_move_assignable_v<PsiFormerAcceleratorSharedModel>);
  CHECK_FALSE(std::is_copy_constructible_v<PsiFormerAcceleratorCrowdState>);
  CHECK_FALSE(std::is_move_constructible_v<PsiFormerAcceleratorCrowdState>);
  CHECK_FALSE(std::is_copy_assignable_v<PsiFormerAcceleratorCrowdState>);
  CHECK_FALSE(std::is_move_assignable_v<PsiFormerAcceleratorCrowdState>);
  CHECK_FALSE(std::is_copy_constructible_v<PsiFormerAcceleratorExecutor>);
  CHECK_FALSE(std::is_move_constructible_v<PsiFormerAcceleratorExecutor>);
  CHECK_FALSE(std::is_copy_assignable_v<PsiFormerAcceleratorExecutor>);
  CHECK_FALSE(std::is_move_assignable_v<PsiFormerAcceleratorExecutor>);
  CHECK_THROWS_AS(PsiFormerAcceleratorCrowdState(0), std::invalid_argument);
}

TEST_CASE("PsiFormer shared accelerator publication is double buffered and failure atomic",
          "[wavefunction][psiformer][accelerator]")
{
  BoundedRecorder recorder;
  PsiFormerAcceleratorSharedModel model(PsiFormerAcceleratorBackend::CUDA, LAYOUT, 4);

  CHECK(model.activeProvenance().model_version == 4);
  CHECK(model.activeProvenance().active_slot == 0);
  CHECK(model.stagingSlot() == 1);

  recorder.failOn(PsiFormerAcceleratorEventKind::MODEL_STAGE);
  CHECK_THROWS_AS(model.stage(LAYOUT, 5, recorder), std::runtime_error);
  CHECK_FALSE(model.publicationPending());
  CHECK(model.activeProvenance().model_version == 4);

  recorder.clearFailure();
  model.stage(LAYOUT, 5, recorder);
  REQUIRE(model.stagedVersion());
  CHECK(*model.stagedVersion() == 5);
  CHECK(model.activeProvenance().model_version == 4);

  BoundedRecorder wrong_publication_hooks;
  CHECK_THROWS_AS(model.complete(5, wrong_publication_hooks), std::logic_error);
  CHECK(model.publicationPending());
  CHECK(wrong_publication_hooks.size() == 0);

  recorder.failOn(PsiFormerAcceleratorEventKind::MODEL_COMPLETE);
  CHECK_THROWS_AS(model.complete(5, recorder), std::runtime_error);
  CHECK(model.publicationPending());
  CHECK(model.activeProvenance().model_version == 4);

  recorder.clearFailure();
  model.complete(5, recorder);
  CHECK_FALSE(model.publicationPending());
  CHECK(model.activeProvenance().model_version == 5);
  CHECK(model.activeProvenance().publication_generation == 1);
  CHECK(model.activeProvenance().active_slot == 1);
  CHECK(model.stagingSlot() == 0);

  model.stage(LAYOUT, 6, recorder);
  recorder.failOn(PsiFormerAcceleratorEventKind::MODEL_CANCEL);
  CHECK_THROWS_AS(model.cancel(6, recorder), std::runtime_error);
  CHECK(model.publicationPending());
  recorder.clearFailure();
  model.cancel(6, recorder);
  CHECK_FALSE(model.publicationPending());
  CHECK(model.activeProvenance().model_version == 5);
  CHECK(model.activeProvenance().publication_generation == 1);

  CHECK_THROWS_AS(model.stage(LAYOUT + 1, 7, recorder), std::invalid_argument);
  CHECK_THROWS_AS(model.stage(LAYOUT, 5, recorder), std::invalid_argument);
}

TEST_CASE("PsiFormer accelerator executor isolates two crowds sharing one model",
          "[wavefunction][psiformer][accelerator]")
{
  BoundedRecorder recorder;
  const PsiFormerAcceleratorOperationMask coverage =
      acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES) |
      acceleratorOperationBit(PsiFormerAcceleratorOperation::VGL) |
      acceleratorOperationBit(PsiFormerAcceleratorOperation::PARAMETER_SCORE);
  PsiFormerAcceleratorExecutor executor(PsiFormerAcceleratorBackend::HIP, coverage);
  PsiFormerAcceleratorSharedModel model(PsiFormerAcceleratorBackend::HIP, LAYOUT, 11);
  PsiFormerAcceleratorCrowdState first(101);
  PsiFormerAcceleratorCrowdState second(102);

  executor.prepare(first, LAYOUT, 32, recorder);
  executor.prepare(second, LAYOUT, 8, recorder);
  CHECK(first.resourceGeneration() == 1);
  CHECK(second.resourceGeneration() == 1);
  CHECK(first.crowdIdentity() != second.crowdIdentity());
  CHECK(recorder[0].crowd_identity == first.crowdIdentity());
  CHECK(recorder[1].crowd_identity == second.crowdIdentity());

  executor.acquire(model, first, recorder);
  executor.acquire(model, second, recorder);
  CHECK(first.acquired());
  CHECK(second.acquired());
  CHECK(first.acquisitionGeneration() == 1);
  CHECK(second.acquisitionGeneration() == 1);

  const PsiFormerAcceleratorResultProvenance first_result = executor.evaluate(
      model, first,
      acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES) |
          acceleratorOperationBit(PsiFormerAcceleratorOperation::PARAMETER_SCORE),
      17, 101, recorder);
  const PsiFormerAcceleratorResultProvenance second_result = executor.evaluate(
      model, second, acceleratorOperationBit(PsiFormerAcceleratorOperation::VGL),
      8, 202, recorder);
  CHECK(first_result.result_generation == 1);
  CHECK(second_result.result_generation == 1);
  CHECK(first_result.crowd_identity == first.crowdIdentity());
  CHECK(second_result.crowd_identity == second.crowdIdentity());
  CHECK(first_result.model == second_result.model);
  CHECK(first.resultPending());
  CHECK(second.resultPending());

  CHECK_THROWS_AS(executor.release(model, first, recorder), std::logic_error);
  CHECK_THROWS_AS(executor.wait(first, 2, recorder), std::logic_error);
  executor.wait(first, first_result.result_generation, recorder);
  CHECK_FALSE(first.resultPending());
  CHECK(first.completedResult(1, 101).item_count == 17);
  CHECK_THROWS_AS(first.completedResult(1, 102), std::logic_error);
  CHECK_THROWS_AS(first.completedResult(2, 101), std::logic_error);
  executor.release(model, first, recorder);

  // The second crowd remains independently acquired and pending.
  CHECK(second.acquired());
  CHECK(second.resultPending());
  executor.wait(second, second_result.result_generation, recorder);
  executor.release(model, second, recorder);

  // Repreparation advances resource provenance without resetting result generations.
  executor.prepare(first, LAYOUT, 64, recorder);
  CHECK(first.resourceGeneration() == 2);
  CHECK(first.resultGeneration() == 1);
}

TEST_CASE("PsiFormer accelerator executor rejects stale provenance and incomplete coverage",
          "[wavefunction][psiformer][accelerator]")
{
  BoundedRecorder recorder;
  PsiFormerAcceleratorExecutor executor(
      PsiFormerAcceleratorBackend::CUDA,
      acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES));
  PsiFormerAcceleratorSharedModel model(PsiFormerAcceleratorBackend::CUDA, LAYOUT, 2);
  PsiFormerAcceleratorCrowdState crowd(201);

  CHECK_THROWS_AS(executor.prepare(crowd, LAYOUT, 0, recorder), std::invalid_argument);
  executor.prepare(crowd, LAYOUT, 4, recorder);
  executor.acquire(model, crowd, recorder);
  CHECK_THROWS_AS(executor.evaluate(
                      model, crowd,
                      acceleratorOperationBit(PsiFormerAcceleratorOperation::VGL),
                      1, 1, recorder),
                  std::runtime_error);
  CHECK_THROWS_AS(executor.evaluate(
                      model, crowd,
                      acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES),
                      5, 1, recorder),
                  std::invalid_argument);

  // One publication preserves an older leased slot, but the next overwrite is blocked.
  model.stage(LAYOUT, 3, recorder);
  model.complete(3, recorder);
  const auto old_result = executor.evaluate(
      model, crowd, acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES),
      1, 7, recorder);
  CHECK(old_result.model.model_version == 2);
  executor.wait(crowd, old_result.result_generation, recorder);
  CHECK_THROWS_AS(model.stage(LAYOUT, 4, recorder), std::logic_error);
  executor.release(model, crowd, recorder);

  model.stage(LAYOUT, 4, recorder);
  model.complete(4, recorder);
  executor.acquire(model, crowd, recorder);
  const auto result = executor.evaluate(
      model, crowd, acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES),
      4, 8, recorder);
  CHECK(result.model.model_version == 4);
  CHECK(result.model.publication_generation == 2);
  executor.wait(crowd, result.result_generation, recorder);
  executor.release(model, crowd, recorder);

  PsiFormerAcceleratorCrowdState wrong_layout(202);
  executor.prepare(wrong_layout, LAYOUT + 1, 1, recorder);
  CHECK_THROWS_AS(executor.acquire(model, wrong_layout, recorder), std::invalid_argument);
}

TEST_CASE("PsiFormer accelerator operation stream has canonical high-level order",
          "[wavefunction][psiformer][accelerator]")
{
  BoundedRecorder recorder;
  PsiFormerAcceleratorExecutor executor(PsiFormerAcceleratorBackend::CUDA,
                                         allPsiFormerAcceleratorOperations());
  PsiFormerAcceleratorSharedModel model(PsiFormerAcceleratorBackend::CUDA, LAYOUT, 9);
  PsiFormerAcceleratorCrowdState crowd(301);

  executor.prepare(crowd, LAYOUT, 3, recorder);
  executor.acquire(model, crowd, recorder);
  recorder.clear();

  CHECK_THROWS_AS(executor.evaluate(
                      model, crowd,
                      acceleratorOperationBit(
                          PsiFormerAcceleratorOperation::PARAMETER_PUBLICATION),
                      3, 54, recorder),
                  std::invalid_argument);
  CHECK(recorder.size() == 0);

  const PsiFormerAcceleratorOperationMask requested =
      acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES) |
      acceleratorOperationBit(PsiFormerAcceleratorOperation::VGL) |
      acceleratorOperationBit(PsiFormerAcceleratorOperation::PARAMETER_SCORE) |
      acceleratorOperationBit(PsiFormerAcceleratorOperation::KINETIC_RESPONSE) |
      acceleratorOperationBit(PsiFormerAcceleratorOperation::NONLOCAL_ECP);
  const auto result = executor.evaluate(model, crowd, requested, 3, 55, recorder);

  REQUIRE(recorder.size() == 7);
  CHECK(recorder[0].kind == PsiFormerAcceleratorEventKind::INPUT_UPLOAD);
  for (std::size_t index = 1; index < 6; ++index)
    CHECK(recorder[index].kind == PsiFormerAcceleratorEventKind::OPERATION);
  CHECK(recorder[1].operation == PsiFormerAcceleratorOperation::VALUES);
  CHECK(recorder[2].operation == PsiFormerAcceleratorOperation::VGL);
  CHECK(recorder[3].operation == PsiFormerAcceleratorOperation::PARAMETER_SCORE);
  CHECK(recorder[4].operation == PsiFormerAcceleratorOperation::KINETIC_RESPONSE);
  CHECK(recorder[5].operation == PsiFormerAcceleratorOperation::NONLOCAL_ECP);
  CHECK(recorder[6].kind == PsiFormerAcceleratorEventKind::RESULT_RECORD);
  for (std::size_t index = 0; index < recorder.size(); ++index)
  {
    CHECK(recorder[index].model_version == 9);
    CHECK(recorder[index].result_generation == result.result_generation);
    CHECK(recorder[index].coordinate_generation == 55);
    CHECK(recorder[index].item_count == 3);
  }

  recorder.clear();
  executor.wait(crowd, result.result_generation, recorder);
  executor.release(model, crowd, recorder);
  REQUIRE(recorder.size() == 2);
  CHECK(recorder[0].kind == PsiFormerAcceleratorEventKind::RESULT_WAIT);
  CHECK(recorder[1].kind == PsiFormerAcceleratorEventKind::CROWD_RELEASE);
}

TEST_CASE("PsiFormer accelerator control plane rejects cross-owner routing",
          "[wavefunction][psiformer][accelerator]")
{
  BoundedRecorder recorder;
  BoundedRecorder wrong_hooks;
  const auto values = acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES);
  PsiFormerAcceleratorExecutor executor(PsiFormerAcceleratorBackend::CUDA, values);
  PsiFormerAcceleratorExecutor same_backend_executor(PsiFormerAcceleratorBackend::CUDA, values);
  PsiFormerAcceleratorExecutor hip_executor(PsiFormerAcceleratorBackend::HIP, values);
  PsiFormerAcceleratorSharedModel model(PsiFormerAcceleratorBackend::CUDA, LAYOUT, 1);
  PsiFormerAcceleratorSharedModel identical_model(PsiFormerAcceleratorBackend::CUDA, LAYOUT, 1);
  PsiFormerAcceleratorCrowdState crowd(350);

  CHECK(model.activeProvenance().model_identity !=
        identical_model.activeProvenance().model_identity);
  executor.prepare(crowd, LAYOUT, 2, recorder);
  CHECK_THROWS_AS(executor.acquire(model, crowd, wrong_hooks), std::logic_error);
  CHECK(wrong_hooks.size() == 0);
  executor.acquire(model, crowd, recorder);
  CHECK(recorder[recorder.size() - 1].model_identity ==
        model.activeProvenance().model_identity);

  const std::size_t submitted = recorder.size();
  CHECK_THROWS_AS(executor.evaluate(identical_model, crowd, values, 1, 1, recorder),
                  std::logic_error);
  CHECK_THROWS_AS(executor.release(identical_model, crowd, recorder), std::logic_error);
  CHECK_THROWS_AS(same_backend_executor.evaluate(model, crowd, values, 1, 1, recorder),
                  std::logic_error);
  CHECK_THROWS_AS(hip_executor.evaluate(model, crowd, values, 1, 1, recorder),
                  std::logic_error);
  CHECK_THROWS_AS(same_backend_executor.wait(crowd, 1, recorder), std::logic_error);
  CHECK_THROWS_AS(same_backend_executor.release(model, crowd, recorder), std::logic_error);
  CHECK_THROWS_AS(hip_executor.release(model, crowd, recorder), std::logic_error);
  CHECK(recorder.size() == submitted);

  recorder.failOn(PsiFormerAcceleratorEventKind::OPERATION);
  CHECK_THROWS_AS(executor.evaluate(model, crowd, values, 1, 2, recorder),
                  std::runtime_error);
  CHECK(crowd.poisoned());
  recorder.clearFailure();
  const std::size_t before_wrong_recovery = recorder.size();
  CHECK_THROWS_AS(executor.recover(identical_model, crowd, recorder), std::logic_error);
  CHECK(crowd.poisoned());
  CHECK(recorder.size() == before_wrong_recovery);
  executor.recover(model, crowd, recorder);
  CHECK_FALSE(crowd.poisoned());
}

TEST_CASE("PsiFormer accelerator crowd fails closed after partial submission",
          "[wavefunction][psiformer][accelerator]")
{
  BoundedRecorder recorder;
  PsiFormerAcceleratorExecutor executor(
      PsiFormerAcceleratorBackend::CUDA,
      acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES));
  PsiFormerAcceleratorSharedModel model(PsiFormerAcceleratorBackend::CUDA, LAYOUT, 1);
  PsiFormerAcceleratorCrowdState crowd(401);

  executor.prepare(crowd, LAYOUT, 2, recorder);
  executor.acquire(model, crowd, recorder);

  // INPUT_UPLOAD succeeds before the injected OPERATION failure.
  recorder.failOn(PsiFormerAcceleratorEventKind::OPERATION);
  CHECK_THROWS_AS(executor.evaluate(
                      model, crowd,
                      acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES),
                      2, 40, recorder),
                  std::runtime_error);
  CHECK(crowd.poisoned());
  CHECK_FALSE(crowd.resultPending());

  recorder.clearFailure();
  CHECK_THROWS_AS(executor.evaluate(
                      model, crowd,
                      acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES),
                      2, 41, recorder),
                  std::logic_error);
  CHECK_THROWS_AS(executor.wait(crowd, 1, recorder), std::logic_error);
  CHECK_THROWS_AS(executor.release(model, crowd, recorder), std::logic_error);
  CHECK_THROWS_AS(executor.prepare(crowd, LAYOUT, 2, recorder), std::logic_error);

  // The old slot remains protected until explicit backend recovery succeeds.
  model.stage(LAYOUT, 2, recorder);
  model.complete(2, recorder);
  CHECK_THROWS_AS(model.stage(LAYOUT, 3, recorder), std::logic_error);
  recorder.failOn(PsiFormerAcceleratorEventKind::CROWD_RECOVER);
  CHECK_THROWS_AS(executor.recover(model, crowd, recorder), std::runtime_error);
  CHECK(crowd.poisoned());
  CHECK_THROWS_AS(model.stage(LAYOUT, 3, recorder), std::logic_error);
  recorder.clearFailure();
  executor.recover(model, crowd, recorder);
  CHECK_FALSE(crowd.poisoned());
  CHECK_FALSE(crowd.prepared());
  CHECK_FALSE(crowd.acquired());
  model.stage(LAYOUT, 3, recorder);
  model.complete(3, recorder);
  executor.prepare(crowd, LAYOUT, 2, recorder);

  PsiFormerAcceleratorCrowdState acquire_failure(402);
  recorder.clearFailure();
  executor.prepare(acquire_failure, LAYOUT, 1, recorder);
  recorder.failOn(PsiFormerAcceleratorEventKind::MODEL_WAIT);
  CHECK_THROWS_AS(executor.acquire(model, acquire_failure, recorder), std::runtime_error);
  CHECK(acquire_failure.poisoned());
  CHECK_FALSE(acquire_failure.acquired());
  recorder.clearFailure();
  executor.recover(model, acquire_failure, recorder);
  CHECK_FALSE(acquire_failure.poisoned());
  executor.prepare(acquire_failure, LAYOUT, 1, recorder);
}

} // namespace qmcplusplus::psiformer
