//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerAcceleratorExecutor.cpp
 * @brief Checked implementation of the PsiFormer accelerator control plane.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorExecutor.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer
{
namespace
{

/// List operations in the invariant production submission order.
constexpr std::array<PsiFormerAcceleratorOperation, 6> OPERATION_ORDER{
    PsiFormerAcceleratorOperation::VALUES,
    PsiFormerAcceleratorOperation::VGL,
    PsiFormerAcceleratorOperation::GRADIENTS_AND_MOVES,
    PsiFormerAcceleratorOperation::PARAMETER_SCORE,
    PsiFormerAcceleratorOperation::KINETIC_RESPONSE,
    PsiFormerAcceleratorOperation::NONLOCAL_ECP};

/// Increment a generation counter without permitting wraparound aliasing.
std::size_t nextGeneration(std::size_t generation)
{
  if (generation == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("PsiFormer accelerator generation overflowed");
  return generation + 1;
}

/// Reject CPU and accelerator families without a Task 26 production executor.
void validateProductionBackend(PsiFormerAcceleratorBackend backend)
{
  if (backend != PsiFormerAcceleratorBackend::CUDA &&
      backend != PsiFormerAcceleratorBackend::HIP)
    throw std::invalid_argument("PsiFormer production accelerator executor supports only CUDA or HIP");
}

} // namespace

PsiFormerAcceleratorSharedModel::PsiFormerAcceleratorSharedModel(
    PsiFormerAcceleratorBackend backend,
    std::uint64_t layout_fingerprint,
    std::size_t initial_version)
    : backend_(backend), layout_fingerprint_(layout_fingerprint), active_version_(initial_version)
{
  validateProductionBackend(backend_);
  if (layout_fingerprint_ == 0)
    throw std::invalid_argument("PsiFormer accelerator model requires a nonzero layout fingerprint");
  slot_versions_[active_slot_] = initial_version;
  slot_publication_generations_[active_slot_] = publication_generation_;
}

void PsiFormerAcceleratorSharedModel::stage(std::uint64_t layout_fingerprint,
                                            std::size_t source_version,
                                            PsiFormerAcceleratorHooks& hooks)
{
  if (publication_hooks_owner_ && publication_hooks_owner_ != &hooks)
    throw std::logic_error("PsiFormer accelerator model belongs to a different publication hook adapter");
  if (publication_pending_)
    throw std::logic_error("PsiFormer accelerator model publication is already pending");
  if (layout_fingerprint != layout_fingerprint_)
    throw std::invalid_argument("PsiFormer accelerator publication has a stale layout fingerprint");
  if (source_version <= active_version_)
    throw std::invalid_argument("PsiFormer accelerator publication requires a newer model version");
  if (slot_leases_[stagingSlot()].load(std::memory_order_relaxed) != 0)
    throw std::logic_error("PsiFormer accelerator staging slot is still leased by an older crowd");

  hooks.submit(publicationEvent(PsiFormerAcceleratorEventKind::MODEL_STAGE, source_version));
  publication_hooks_owner_ = &hooks;
  staged_version_ = source_version;
  publication_pending_ = true;
}

void PsiFormerAcceleratorSharedModel::complete(std::size_t source_version,
                                               PsiFormerAcceleratorHooks& hooks)
{
  if (publication_hooks_owner_ != &hooks)
    throw std::logic_error("PsiFormer accelerator model belongs to a different publication hook adapter");
  if (!publication_pending_ || staged_version_ != source_version)
    throw std::logic_error("PsiFormer accelerator completion does not match the staged version");

  const std::size_t next_publication_generation = nextGeneration(publication_generation_);
  hooks.submit(publicationEvent(PsiFormerAcceleratorEventKind::MODEL_COMPLETE, source_version));
  active_slot_ = stagingSlot();
  active_version_ = source_version;
  publication_generation_ = next_publication_generation;
  slot_versions_[active_slot_] = source_version;
  slot_publication_generations_[active_slot_] = publication_generation_;
  publication_pending_ = false;
  staged_version_ = 0;
}

PsiFormerAcceleratorSharedModel::PreparedCompletion
PsiFormerAcceleratorSharedModel::prepareCompletion(std::size_t version, const PsiFormerAcceleratorHooks& hooks) const
{
  if (publication_hooks_owner_ != &hooks || !publication_pending_ || staged_version_ != version)
    throw std::logic_error("Prepared device completion has foreign or stale staged authority");
  return {this, version, nextGeneration(publication_generation_), stagingSlot()};
}

void PsiFormerAcceleratorSharedModel::completePrepared(const PreparedCompletion& prepared) noexcept
{
  if (prepared.owner != this || !publication_pending_ || staged_version_ != prepared.version ||
      prepared.generation != publication_generation_ + 1 || prepared.slot != stagingSlot())
    std::terminate();
  active_slot_ = prepared.slot;
  active_version_ = prepared.version;
  publication_generation_ = prepared.generation;
  slot_versions_[active_slot_] = prepared.version;
  slot_publication_generations_[active_slot_] = prepared.generation;
  publication_pending_ = false;
  staged_version_ = 0;
}

void PsiFormerAcceleratorSharedModel::cancel(std::size_t source_version,
                                             PsiFormerAcceleratorHooks& hooks)
{
  if (publication_hooks_owner_ != &hooks)
    throw std::logic_error("PsiFormer accelerator model belongs to a different publication hook adapter");
  if (!publication_pending_ || staged_version_ != source_version)
    throw std::logic_error("PsiFormer accelerator cancellation does not match the staged version");

  hooks.submit(publicationEvent(PsiFormerAcceleratorEventKind::MODEL_CANCEL, source_version));
  publication_pending_ = false;
  staged_version_ = 0;
}

PsiFormerAcceleratorModelProvenance
PsiFormerAcceleratorSharedModel::activeProvenance() const noexcept
{
  return {this, reinterpret_cast<std::uintptr_t>(this), backend_, layout_fingerprint_,
          active_version_, publication_generation_, active_slot_};
}

std::optional<std::size_t> PsiFormerAcceleratorSharedModel::stagedVersion() const noexcept
{
  return publication_pending_ ? std::optional<std::size_t>{staged_version_} : std::nullopt;
}

PsiFormerAcceleratorEvent PsiFormerAcceleratorSharedModel::publicationEvent(
    PsiFormerAcceleratorEventKind kind,
    std::size_t version) const noexcept
{
  PsiFormerAcceleratorEvent event;
  event.kind = kind;
  event.backend = backend_;
  event.layout_fingerprint = layout_fingerprint_;
  event.model_version = version;
  event.publication_generation = publication_generation_;
  event.model_identity = reinterpret_cast<std::uintptr_t>(this);
  event.model_slot = stagingSlot();
  return event;
}

void PsiFormerAcceleratorSharedModel::acquireLease(
    const PsiFormerAcceleratorModelProvenance& provenance)
{
  if (!(activeProvenance() == provenance))
    throw std::logic_error("PsiFormer accelerator lease requested for stale active provenance");
  auto& count = slot_leases_[provenance.active_slot];
  count.store(nextGeneration(count.load(std::memory_order_relaxed)), std::memory_order_relaxed);
}

void PsiFormerAcceleratorSharedModel::releaseLease(
    const PsiFormerAcceleratorModelProvenance& provenance)
{
  if (!hasLease(provenance))
    throw std::logic_error("PsiFormer accelerator lease release has stale model provenance");
  slot_leases_[provenance.active_slot].fetch_sub(1, std::memory_order_relaxed);
}

bool PsiFormerAcceleratorSharedModel::hasLease(
    const PsiFormerAcceleratorModelProvenance& provenance) const noexcept
{
  if (provenance.owner != this || provenance.backend != backend_ ||
      provenance.model_identity != reinterpret_cast<std::uintptr_t>(this) ||
      provenance.layout_fingerprint != layout_fingerprint_ ||
      provenance.active_slot > 1)
    return false;
  return slot_versions_[provenance.active_slot] == provenance.model_version &&
      slot_publication_generations_[provenance.active_slot] == provenance.publication_generation &&
      slot_leases_[provenance.active_slot].load(std::memory_order_relaxed) != 0;
}

const PsiFormerAcceleratorResultProvenance&
PsiFormerAcceleratorCrowdState::completedResult(
    std::size_t expected_generation,
    std::size_t expected_coordinate_generation) const
{
  if (!completed_result_ || completed_result_->result_generation != expected_generation ||
      completed_result_->coordinate_generation != expected_coordinate_generation)
    throw std::logic_error("PsiFormer accelerator completed-result generation is stale");
  return *completed_result_;
}

PsiFormerAcceleratorCrowdState::PsiFormerAcceleratorCrowdState(
    std::uint64_t crowd_identity)
    : crowd_identity_(crowd_identity)
{
  if (crowd_identity_ == 0)
    throw std::invalid_argument("PsiFormer accelerator crowd identity must be nonzero");
}

PsiFormerAcceleratorExecutor::PsiFormerAcceleratorExecutor(
    PsiFormerAcceleratorBackend backend,
    PsiFormerAcceleratorOperationMask operation_coverage)
    : backend_(backend), operation_coverage_(operation_coverage)
{
  validateProductionBackend(backend_);
  if ((operation_coverage_ & ~allPsiFormerAcceleratorOperations()) != 0)
    throw std::invalid_argument("PsiFormer accelerator executor contains unknown coverage bits");
}

void PsiFormerAcceleratorExecutor::prepare(PsiFormerAcceleratorCrowdState& crowd,
                                            std::uint64_t layout_fingerprint,
                                            std::size_t capacity,
                                            PsiFormerAcceleratorHooks& hooks) const
{
  if (crowd.executor_owner_ && crowd.executor_owner_ != this)
    throw std::logic_error("PsiFormer accelerator crowd belongs to a different executor");
  if (crowd.hooks_owner_ && crowd.hooks_owner_ != &hooks)
    throw std::logic_error("PsiFormer accelerator crowd belongs to a different hook adapter");
  if (crowd.executor_owner_ && crowd.backend_ != backend_)
    throw std::logic_error("PsiFormer accelerator crowd belongs to a different backend");
  if (crowd.poisoned_)
    throw std::logic_error("PsiFormer accelerator crowd is poisoned by partial backend submission");
  if (layout_fingerprint == 0)
    throw std::invalid_argument("PsiFormer accelerator crowd requires a nonzero layout fingerprint");
  if (capacity == 0)
    throw std::invalid_argument("PsiFormer accelerator crowd capacity must be positive");
  if (crowd.acquired_ || crowd.pending_result_)
    throw std::logic_error("PsiFormer accelerator crowd cannot be prepared while active");

  const std::size_t generation = nextGeneration(crowd.resource_generation_);
  PsiFormerAcceleratorEvent event;
  event.kind = PsiFormerAcceleratorEventKind::CROWD_PREPARE;
  event.backend = backend_;
  event.layout_fingerprint = layout_fingerprint;
  event.resource_generation = generation;
  event.item_count = capacity;
  event.crowd_identity = crowd.crowd_identity_;
  hooks.submit(event);

  crowd.backend_ = backend_;
  crowd.executor_owner_ = this;
  crowd.hooks_owner_ = &hooks;
  crowd.layout_fingerprint_ = layout_fingerprint;
  crowd.capacity_ = capacity;
  crowd.resource_generation_ = generation;
  crowd.prepared_ = true;
  crowd.completed_result_.reset();
}

void PsiFormerAcceleratorExecutor::acquire(PsiFormerAcceleratorSharedModel& model,
                                            PsiFormerAcceleratorCrowdState& crowd,
                                            PsiFormerAcceleratorHooks& hooks) const
{
  validateCrowdOwners(crowd, hooks);
  if (crowd.poisoned_)
    throw std::logic_error("PsiFormer accelerator crowd is poisoned by partial backend submission");
  if (!crowd.prepared_)
    throw std::logic_error("PsiFormer accelerator crowd must be prepared before acquisition");
  if (crowd.acquired_ || crowd.pending_result_)
    throw std::logic_error("PsiFormer accelerator crowd is already active");

  const PsiFormerAcceleratorModelProvenance provenance = model.activeProvenance();
  if (crowd.backend_ != backend_ || provenance.backend != backend_)
    throw std::invalid_argument("PsiFormer accelerator crowd/model backend mismatch");
  if (crowd.layout_fingerprint_ != provenance.layout_fingerprint)
    throw std::invalid_argument("PsiFormer accelerator crowd/model layout fingerprint mismatch");

  const std::size_t acquisition = nextGeneration(crowd.acquisition_generation_);
  PsiFormerAcceleratorEvent event = crowdEvent(PsiFormerAcceleratorEventKind::CROWD_ACQUIRE,
                                                crowd, &provenance, 0, 0);
  event.acquisition_generation = acquisition;
  hooks.submit(event);
  event.kind = PsiFormerAcceleratorEventKind::MODEL_WAIT;
  try
  {
    hooks.submit(event);
    model.acquireLease(provenance);
  }
  catch (...)
  {
    crowd.poisoned_ = true;
    throw;
  }

  crowd.acquisition_generation_ = acquisition;
  crowd.bound_model_ = provenance;
  crowd.acquired_ = true;
}

PsiFormerAcceleratorResultProvenance PsiFormerAcceleratorExecutor::evaluate(
    const PsiFormerAcceleratorSharedModel& model,
    PsiFormerAcceleratorCrowdState& crowd,
    PsiFormerAcceleratorOperationMask operations,
    std::size_t item_count,
    std::size_t coordinate_generation,
    PsiFormerAcceleratorHooks& hooks) const
{
  validateCrowdOwners(crowd, hooks);
  if (crowd.poisoned_)
    throw std::logic_error("PsiFormer accelerator crowd is poisoned by partial backend submission");
  if (!crowd.acquired_ || !crowd.bound_model_)
    throw std::logic_error("PsiFormer accelerator crowd must be acquired before evaluation");
  if (crowd.pending_result_)
    throw std::logic_error("PsiFormer accelerator crowd already has a pending result");
  if (operations == 0 ||
      (operations & ~psiFormerAcceleratorEvaluationOperations()) != 0)
    throw std::invalid_argument("PsiFormer accelerator evaluation has invalid operation bits");
  if ((operations & operation_coverage_) != operations)
    throw std::runtime_error("PsiFormer accelerator executor lacks end-to-end operation coverage");
  if (item_count == 0 || item_count > crowd.capacity_)
    throw std::invalid_argument("PsiFormer accelerator evaluation exceeds prepared crowd capacity");
  if (coordinate_generation == 0)
    throw std::invalid_argument("PsiFormer accelerator evaluation requires a nonzero coordinate generation");
  if (crowd.bound_model_->owner != &model || !model.hasLease(*crowd.bound_model_))
    throw std::logic_error("PsiFormer accelerator crowd is bound to stale model provenance");

  const std::size_t result_generation = nextGeneration(crowd.result_generation_);
  PsiFormerAcceleratorResultProvenance result{*crowd.bound_model_,
                                               crowd.crowd_identity_,
                                               crowd.resource_generation_,
                                               crowd.acquisition_generation_,
                                               result_generation,
                                               item_count,
                                               coordinate_generation,
                                               operations};

  std::size_t submitted = 0;
  try
  {
    hooks.submit(crowdEvent(PsiFormerAcceleratorEventKind::INPUT_UPLOAD, crowd,
                            &*crowd.bound_model_, result_generation, item_count,
                            coordinate_generation));
    ++submitted;
    for (const PsiFormerAcceleratorOperation operation : OPERATION_ORDER)
      if ((operations & acceleratorOperationBit(operation)) != 0)
      {
        PsiFormerAcceleratorEvent event = crowdEvent(PsiFormerAcceleratorEventKind::OPERATION,
                                                      crowd, &*crowd.bound_model_,
                                                      result_generation, item_count,
                                                      coordinate_generation);
        event.operation = operation;
        hooks.submit(event);
        ++submitted;
      }
    hooks.submit(crowdEvent(PsiFormerAcceleratorEventKind::RESULT_RECORD, crowd,
                            &*crowd.bound_model_, result_generation, item_count,
                            coordinate_generation));
  }
  catch (...)
  {
    if (submitted != 0)
      crowd.poisoned_ = true;
    throw;
  }

  crowd.result_generation_ = result_generation;
  crowd.pending_result_ = result;
  return result;
}

void PsiFormerAcceleratorExecutor::wait(PsiFormerAcceleratorCrowdState& crowd,
                                         std::size_t result_generation,
                                         PsiFormerAcceleratorHooks& hooks) const
{
  validateCrowdOwners(crowd, hooks);
  if (crowd.poisoned_)
    throw std::logic_error("PsiFormer accelerator crowd is poisoned by partial backend submission");
  if (!crowd.acquired_ || !crowd.pending_result_ ||
      crowd.pending_result_->result_generation != result_generation)
    throw std::logic_error("PsiFormer accelerator wait has a stale result generation");

  hooks.submit(crowdEvent(PsiFormerAcceleratorEventKind::RESULT_WAIT, crowd,
                          &crowd.pending_result_->model, result_generation,
                          crowd.pending_result_->item_count,
                          crowd.pending_result_->coordinate_generation));
  crowd.completed_result_ = crowd.pending_result_;
  crowd.pending_result_.reset();
}

void PsiFormerAcceleratorExecutor::release(PsiFormerAcceleratorSharedModel& model,
                                            PsiFormerAcceleratorCrowdState& crowd,
                                            PsiFormerAcceleratorHooks& hooks) const
{
  validateCrowdOwners(crowd, hooks);
  if (crowd.poisoned_)
    throw std::logic_error("PsiFormer accelerator crowd is poisoned by partial backend submission");
  if (!crowd.acquired_ || !crowd.bound_model_)
    throw std::logic_error("PsiFormer accelerator crowd is not acquired");
  if (crowd.pending_result_)
    throw std::logic_error("PsiFormer accelerator crowd cannot be released with a pending result");
  if (crowd.bound_model_->owner != &model)
    throw std::logic_error("PsiFormer accelerator crowd belongs to a different shared model");

  hooks.submit(crowdEvent(PsiFormerAcceleratorEventKind::CROWD_RELEASE, crowd,
                          &*crowd.bound_model_, crowd.result_generation_, 0));
  try
  {
    model.releaseLease(*crowd.bound_model_);
  }
  catch (...)
  {
    crowd.poisoned_ = true;
    throw;
  }
  crowd.bound_model_.reset();
  crowd.acquired_ = false;
}

void PsiFormerAcceleratorExecutor::recover(PsiFormerAcceleratorSharedModel& model,
                                            PsiFormerAcceleratorCrowdState& crowd,
                                            PsiFormerAcceleratorHooks& hooks) const
{
  validateCrowdOwners(crowd, hooks);
  if (!crowd.poisoned_)
    throw std::logic_error("PsiFormer accelerator recovery requires a poisoned crowd");

  const PsiFormerAcceleratorModelProvenance* provenance =
      crowd.bound_model_ ? &*crowd.bound_model_ : nullptr;
  if (provenance && provenance->owner != &model)
    throw std::logic_error("PsiFormer accelerator crowd belongs to a different shared model");
  hooks.submit(crowdEvent(PsiFormerAcceleratorEventKind::CROWD_RECOVER, crowd,
                          provenance, crowd.result_generation_, 0));

  if (crowd.bound_model_ && model.hasLease(*crowd.bound_model_))
    model.releaseLease(*crowd.bound_model_);

  crowd.layout_fingerprint_ = 0;
  crowd.capacity_ = 0;
  crowd.prepared_ = false;
  crowd.acquired_ = false;
  crowd.poisoned_ = false;
  crowd.bound_model_.reset();
  crowd.pending_result_.reset();
  crowd.completed_result_.reset();
}

void PsiFormerAcceleratorExecutor::validateCrowdOwners(
    const PsiFormerAcceleratorCrowdState& crowd,
    const PsiFormerAcceleratorHooks& hooks) const
{
  if (crowd.executor_owner_ != this)
    throw std::logic_error("PsiFormer accelerator crowd belongs to a different executor");
  if (crowd.hooks_owner_ != &hooks)
    throw std::logic_error("PsiFormer accelerator crowd belongs to a different hook adapter");
  if (crowd.backend_ != backend_)
    throw std::logic_error("PsiFormer accelerator crowd belongs to a different backend");
}

PsiFormerAcceleratorEvent PsiFormerAcceleratorExecutor::crowdEvent(
    PsiFormerAcceleratorEventKind kind,
    const PsiFormerAcceleratorCrowdState& crowd,
    const PsiFormerAcceleratorModelProvenance* model,
    std::size_t result_generation,
    std::size_t item_count,
    std::size_t coordinate_generation) const noexcept
{
  PsiFormerAcceleratorEvent event;
  event.kind = kind;
  event.backend = backend_;
  event.layout_fingerprint = crowd.layout_fingerprint_;
  event.resource_generation = crowd.resource_generation_;
  event.acquisition_generation = crowd.acquisition_generation_;
  event.result_generation = result_generation;
  event.coordinate_generation = coordinate_generation;
  event.item_count = item_count;
  event.crowd_identity = crowd.crowd_identity_;
  if (model)
  {
    event.model_version = model->model_version;
    event.publication_generation = model->publication_generation;
    event.model_identity = model->model_identity;
    event.model_slot = model->active_slot;
  }
  return event;
}

const char* psiFormerAcceleratorOperationName(PsiFormerAcceleratorOperation operation) noexcept
{
  switch (operation)
  {
  case PsiFormerAcceleratorOperation::VALUES:
    return "values";
  case PsiFormerAcceleratorOperation::VGL:
    return "vgl";
  case PsiFormerAcceleratorOperation::GRADIENTS_AND_MOVES:
    return "gradients_and_moves";
  case PsiFormerAcceleratorOperation::PARAMETER_SCORE:
    return "parameter_score";
  case PsiFormerAcceleratorOperation::KINETIC_RESPONSE:
    return "kinetic_response";
  case PsiFormerAcceleratorOperation::NONLOCAL_ECP:
    return "nonlocal_ecp";
  case PsiFormerAcceleratorOperation::PARAMETER_PUBLICATION:
    return "parameter_publication";
  }
  return "unknown";
}

} // namespace qmcplusplus::psiformer
