//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerAcceleratorExecutor.h
 * @brief Backend-neutral production control plane for PsiFormer accelerators.
 *
 * This layer deliberately owns no vendor handles or pointers.  It validates the
 * model-publication and crowd-resource protocol and emits a bounded sequence of
 * recording-hook calls that a later CUDA/HIP adapter will implement.  Construction
 * and preparation may allocate in their callers; acquire/evaluate/wait/release do
 * not allocate within this state machine.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_ACCELERATOR_EXECUTOR_H
#define QMCPLUSPLUS_PSIFORMER_ACCELERATOR_EXECUTOR_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorPlanning.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace qmcplusplus::psiformer
{

class PsiFormerAcceleratorSharedModel;
class PsiFormerAcceleratorExecutor;

/// Identify observable control-plane boundaries in their required submission order.
enum class PsiFormerAcceleratorEventKind : std::uint8_t
{
  MODEL_STAGE,
  MODEL_COMPLETE,
  MODEL_CANCEL,
  CROWD_PREPARE,
  CROWD_ACQUIRE,
  MODEL_WAIT,
  INPUT_UPLOAD,
  OPERATION,
  RESULT_RECORD,
  RESULT_WAIT,
  CROWD_RELEASE,
  CROWD_RECOVER
};

/// Describe one backend-neutral queue/resource action without dereferenceable pointers.
struct PsiFormerAcceleratorEvent
{
  PsiFormerAcceleratorEventKind kind = PsiFormerAcceleratorEventKind::CROWD_PREPARE;
  PsiFormerAcceleratorBackend backend = PsiFormerAcceleratorBackend::CPU;
  PsiFormerAcceleratorOperation operation = PsiFormerAcceleratorOperation::VALUES;
  std::uint64_t layout_fingerprint = 0;
  std::size_t model_version = 0;
  std::size_t publication_generation = 0;
  std::size_t resource_generation = 0;
  std::size_t acquisition_generation = 0;
  std::size_t result_generation = 0;
  std::size_t coordinate_generation = 0;
  std::size_t item_count = 0;
  std::uintptr_t model_identity = 0;
  std::uint64_t crowd_identity = 0;
  std::uint8_t model_slot = 0;
};

/** Receive control-plane actions from the backend-neutral executor.
 *
 * A production hook will launch copies, kernels, and events.  Tests use a bounded
 * recorder.  Hook implementations must not retain references to the event object.
 * Throwing from one submit must mean that event was not submitted.  A hook that can
 * fail after submission must absorb that failure and represent it through a later
 * event.  The executor poisons a crowd when a later event in a multi-submit sequence
 * fails after an earlier event succeeded.
 */
class PsiFormerAcceleratorHooks
{
public:
  virtual ~PsiFormerAcceleratorHooks() = default;

  /// Submit or record one fully validated control-plane action.
  virtual void submit(const PsiFormerAcceleratorEvent& event) = 0;
};

/// Identify the active immutable model image used by a crowd acquisition.
struct PsiFormerAcceleratorModelProvenance
{
  const PsiFormerAcceleratorSharedModel* owner = nullptr;
  std::uintptr_t model_identity = 0;
  PsiFormerAcceleratorBackend backend = PsiFormerAcceleratorBackend::CPU;
  std::uint64_t layout_fingerprint = 0;
  std::size_t model_version = 0;
  std::size_t publication_generation = 0;
  std::uint8_t active_slot = 0;

  friend bool operator==(const PsiFormerAcceleratorModelProvenance& lhs,
                         const PsiFormerAcceleratorModelProvenance& rhs) noexcept
  {
    return lhs.owner == rhs.owner && lhs.model_identity == rhs.model_identity &&
        lhs.backend == rhs.backend &&
        lhs.layout_fingerprint == rhs.layout_fingerprint &&
        lhs.model_version == rhs.model_version &&
        lhs.publication_generation == rhs.publication_generation &&
        lhs.active_slot == rhs.active_slot;
  }
};

/** Track double-buffered parameters shared by all crowds using one model/device.
 *
 * Stage, complete, and cancel are failure atomic with respect to hook exceptions:
 * the corresponding state transition occurs only after the hook returns.  A staged
 * image never changes the active provenance until completion succeeds.
 * Publication and lease mutations require external serialization by the owning
 * PsiFormerSharedState mutex. Atomic lease counts permit concurrent reads during
 * independent crowd submissions only while publication is excluded externally.
 * The shared model and publication hook adapter must outlive every pending
 * publication, and the shared model must outlive every crowd lease naming it.
 */
class PsiFormerAcceleratorSharedModel
{
public:
  PsiFormerAcceleratorSharedModel(PsiFormerAcceleratorBackend backend,
                                  std::uint64_t layout_fingerprint,
                                  std::size_t initial_version);

  PsiFormerAcceleratorSharedModel(const PsiFormerAcceleratorSharedModel&) = delete;
  PsiFormerAcceleratorSharedModel& operator=(const PsiFormerAcceleratorSharedModel&) = delete;
  PsiFormerAcceleratorSharedModel(PsiFormerAcceleratorSharedModel&&) = delete;
  PsiFormerAcceleratorSharedModel& operator=(PsiFormerAcceleratorSharedModel&&) = delete;

  /// Begin copying a newer full parameter image into the inactive slot.
  void stage(std::uint64_t layout_fingerprint,
             std::size_t source_version,
             PsiFormerAcceleratorHooks& hooks);

  /// Atomically publish the staged slot after its backend completion event.
  void complete(std::size_t source_version, PsiFormerAcceleratorHooks& hooks);

  /// Cancel the staged image while leaving the active slot and generation unchanged.
  void cancel(std::size_t source_version, PsiFormerAcceleratorHooks& hooks);

  /// Return the model image currently safe for evaluation.
  PsiFormerAcceleratorModelProvenance activeProvenance() const noexcept;

  /// Return whether a parameter copy is pending in the inactive slot.
  bool publicationPending() const noexcept { return publication_pending_; }

  /// Return the staged version when publication is pending.
  std::optional<std::size_t> stagedVersion() const noexcept;

  /// Return the inactive slot targeted by the next or current publication.
  std::uint8_t stagingSlot() const noexcept { return static_cast<std::uint8_t>(1U - active_slot_); }

private:
  friend class PsiFormerAcceleratorExecutor;

  /// Register one exact active slot/version as in use by a crowd.
  void acquireLease(const PsiFormerAcceleratorModelProvenance& provenance);

  /// Release one exact crowd lease without changing publication state.
  void releaseLease(const PsiFormerAcceleratorModelProvenance& provenance);

  /// Return whether one exact leased image remains resident and protected.
  bool hasLease(const PsiFormerAcceleratorModelProvenance& provenance) const noexcept;

  PsiFormerAcceleratorEvent publicationEvent(PsiFormerAcceleratorEventKind kind,
                                              std::size_t version) const noexcept;

  PsiFormerAcceleratorBackend backend_;
  PsiFormerAcceleratorHooks* publication_hooks_owner_ = nullptr;
  std::uint64_t layout_fingerprint_;
  std::size_t active_version_ = 0;
  std::size_t publication_generation_ = 0;
  std::size_t staged_version_ = 0;
  std::array<std::size_t, 2> slot_versions_{};
  std::array<std::size_t, 2> slot_publication_generations_{};
  std::array<std::atomic<std::size_t>, 2> slot_leases_{{0, 0}};
  std::uint8_t active_slot_ = 0;
  bool publication_pending_ = false;
};

/// Describe one completed or pending result without exposing backend storage.
struct PsiFormerAcceleratorResultProvenance
{
  PsiFormerAcceleratorModelProvenance model;
  std::uint64_t crowd_identity = 0;
  std::size_t resource_generation = 0;
  std::size_t acquisition_generation = 0;
  std::size_t result_generation = 0;
  std::size_t item_count = 0;
  std::size_t coordinate_generation = 0;
  PsiFormerAcceleratorOperationMask operations = 0;
};

/** Hold one crowd's prepared capacity and warmed execution generations.
 *
 * This object stores scalar control state only.  The backend Resource paired with
 * it owns its queue, arena, handles, and result storage.  Every mutation requires
 * exclusive ownership of that borrowed crowd resource; no method is concurrently
 * safe on the same instance.  The executor, hook adapter, and shared model must
 * outlive every prepared or acquired crowd that is bound to them.
 */
class PsiFormerAcceleratorCrowdState
{
public:
  /** Construct an unprepared crowd with a stable adapter-routing identity.
   *
   * The nonzero identity must be unique within the lifetime and routing domain of
   * the hook adapter to which this crowd will be bound.
   */
  explicit PsiFormerAcceleratorCrowdState(std::uint64_t crowd_identity);
  PsiFormerAcceleratorCrowdState(const PsiFormerAcceleratorCrowdState&) = delete;
  PsiFormerAcceleratorCrowdState& operator=(const PsiFormerAcceleratorCrowdState&) = delete;
  PsiFormerAcceleratorCrowdState(PsiFormerAcceleratorCrowdState&&) = delete;
  PsiFormerAcceleratorCrowdState& operator=(PsiFormerAcceleratorCrowdState&&) = delete;

  /// Return whether backend storage has been prepared for this crowd.
  bool prepared() const noexcept { return prepared_; }

  /// Return whether this crowd currently holds its paired backend resource.
  bool acquired() const noexcept { return acquired_; }

  /// Return whether a submitted result still awaits synchronization.
  bool resultPending() const noexcept { return pending_result_.has_value(); }

  /// Return whether partial submission made this crowd unsafe until explicit recovery.
  bool poisoned() const noexcept { return poisoned_; }

  /// Return the stable routing identity of this crowd resource.
  std::uint64_t crowdIdentity() const noexcept { return crowd_identity_; }

  /// Return the current prepared-resource generation.
  std::size_t resourceGeneration() const noexcept { return resource_generation_; }

  /// Return the current acquisition generation.
  std::size_t acquisitionGeneration() const noexcept { return acquisition_generation_; }

  /// Return the most recently assigned result generation.
  std::size_t resultGeneration() const noexcept { return result_generation_; }

  /// Return the pending result provenance, when present.
  const std::optional<PsiFormerAcceleratorResultProvenance>& pendingResult() const noexcept
  {
    return pending_result_;
  }

  /// Return one exact completed result or reject a stale generation request.
  const PsiFormerAcceleratorResultProvenance& completedResult(
      std::size_t expected_generation,
      std::size_t expected_coordinate_generation) const;

private:
  friend class PsiFormerAcceleratorExecutor;

  PsiFormerAcceleratorBackend backend_ = PsiFormerAcceleratorBackend::CPU;
  const std::uint64_t crowd_identity_;
  const PsiFormerAcceleratorExecutor* executor_owner_ = nullptr;
  PsiFormerAcceleratorHooks* hooks_owner_ = nullptr;
  std::uint64_t layout_fingerprint_ = 0;
  std::size_t capacity_ = 0;
  std::size_t resource_generation_ = 0;
  std::size_t acquisition_generation_ = 0;
  std::size_t result_generation_ = 0;
  bool prepared_ = false;
  bool acquired_ = false;
  bool poisoned_ = false;
  std::optional<PsiFormerAcceleratorModelProvenance> bound_model_;
  std::optional<PsiFormerAcceleratorResultProvenance> pending_result_;
  std::optional<PsiFormerAcceleratorResultProvenance> completed_result_;
};

/** Validate and emit the production-shaped PsiFormer accelerator operation stream.
 *
 * Coverage is supplied explicitly by the end-to-end adapter.  Compiled leaf kernels
 * do not implicitly enable any operation bit or public DEVICE capability.
 */
class PsiFormerAcceleratorExecutor
{
public:
  PsiFormerAcceleratorExecutor(PsiFormerAcceleratorBackend backend,
                               PsiFormerAcceleratorOperationMask operation_coverage);

  PsiFormerAcceleratorExecutor(const PsiFormerAcceleratorExecutor&) = delete;
  PsiFormerAcceleratorExecutor& operator=(const PsiFormerAcceleratorExecutor&) = delete;
  PsiFormerAcceleratorExecutor(PsiFormerAcceleratorExecutor&&) = delete;
  PsiFormerAcceleratorExecutor& operator=(PsiFormerAcceleratorExecutor&&) = delete;

  /// Prepare or reprepare one released crowd resource; allocation belongs in its hook.
  void prepare(PsiFormerAcceleratorCrowdState& crowd,
               std::uint64_t layout_fingerprint,
               std::size_t capacity,
               PsiFormerAcceleratorHooks& hooks) const;

  /// Acquire a prepared crowd resource and bind it to the model's active provenance.
  void acquire(PsiFormerAcceleratorSharedModel& model,
               PsiFormerAcceleratorCrowdState& crowd,
               PsiFormerAcceleratorHooks& hooks) const;

  /// Submit one canonical operation sequence and return its pending provenance.
  PsiFormerAcceleratorResultProvenance evaluate(
      const PsiFormerAcceleratorSharedModel& model,
      PsiFormerAcceleratorCrowdState& crowd,
      PsiFormerAcceleratorOperationMask operations,
      std::size_t item_count,
      std::size_t coordinate_generation,
      PsiFormerAcceleratorHooks& hooks) const;

  /// Synchronize one exact pending result generation.
  void wait(PsiFormerAcceleratorCrowdState& crowd,
            std::size_t result_generation,
            PsiFormerAcceleratorHooks& hooks) const;

  /// Release a crowd only after all pending result work has completed.
  void release(PsiFormerAcceleratorSharedModel& model,
               PsiFormerAcceleratorCrowdState& crowd,
               PsiFormerAcceleratorHooks& hooks) const;

  /** Recover one poisoned crowd only after backend synchronization and cleanup.
   *
   * A successful CROWD_RECOVER hook must guarantee that no earlier operation can
   * access the bound model slot or crowd storage.  Only then is the exact lease
   * released and the crowd's preparation invalidated.  Hook failure leaves all
   * poison, ownership, and lease state unchanged.
   */
  void recover(PsiFormerAcceleratorSharedModel& model,
               PsiFormerAcceleratorCrowdState& crowd,
               PsiFormerAcceleratorHooks& hooks) const;

  /// Return the end-to-end operation coverage explicitly supplied at construction.
  PsiFormerAcceleratorOperationMask operationCoverage() const noexcept
  {
    return operation_coverage_;
  }

private:
  /// Reject an executor, backend, or hook other than those bound during preparation.
  void validateCrowdOwners(const PsiFormerAcceleratorCrowdState& crowd,
                           const PsiFormerAcceleratorHooks& hooks) const;

  PsiFormerAcceleratorEvent crowdEvent(PsiFormerAcceleratorEventKind kind,
                                       const PsiFormerAcceleratorCrowdState& crowd,
                                       const PsiFormerAcceleratorModelProvenance* model,
                                       std::size_t result_generation,
                                       std::size_t item_count,
                                       std::size_t coordinate_generation = 0) const noexcept;

  PsiFormerAcceleratorBackend backend_;
  PsiFormerAcceleratorOperationMask operation_coverage_;
};

/// Return the stable diagnostic spelling for one accelerator operation.
const char* psiFormerAcceleratorOperationName(PsiFormerAcceleratorOperation operation) noexcept;

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_ACCELERATOR_EXECUTOR_H
