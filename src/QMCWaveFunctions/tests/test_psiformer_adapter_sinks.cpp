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

#include "Particle/MCMultiParticleMoves.h"
#include "Particle/ParticleSet.h"
#include "Particle/VirtualParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerMemoryPolicy.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "ResourceCollection.h"
#include "Utilities/BatchResourcePreparation.h"
#include "psiformer_test_utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
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
  DEALLOCATION,
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

void recordDeallocation(void* pointer) noexcept
{
  psiformer_adapter_allocation_audit::record(
      psiformer_adapter_allocation_audit::Form::DEALLOCATION);
  std::free(pointer);
}

void operator delete(void* pointer) noexcept { recordDeallocation(pointer); }
void operator delete[](void* pointer) noexcept { recordDeallocation(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { recordDeallocation(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { recordDeallocation(pointer); }
void operator delete(void* pointer, const std::nothrow_t&) noexcept { recordDeallocation(pointer); }
void operator delete[](void* pointer, const std::nothrow_t&) noexcept { recordDeallocation(pointer); }
void operator delete(void* pointer, std::align_val_t) noexcept { recordDeallocation(pointer); }
void operator delete[](void* pointer, std::align_val_t) noexcept { recordDeallocation(pointer); }
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept { recordDeallocation(pointer); }
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept { recordDeallocation(pointer); }
void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept
{
  recordDeallocation(pointer);
}
void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept
{
  recordDeallocation(pointer);
}

namespace qmcplusplus
{
namespace testing
{
/** Fixed-storage view used only by the allocation-gate executable. */
struct PsiFormerAllocationCloneStorage
{
  std::array<const void*, 4> data{};
  std::array<std::size_t, 4> sizes{};
  std::array<std::size_t, 4> capacities{};
  bool exact_marker = false;
};

/** Complete persistent clone state frozen around a scalar allocation window. */
struct PsiFormerScalarAllocationState
{
  PsiFormerWF::LogValue log_value;
  std::size_t observed_parameter_version = 0;
  bool restore_validation_pending = false;
  bool accepted_value_valid = false;
  ParticleSet::ParticleGradient accepted_gradient;
  ParticleSet::ParticleLaplacian accepted_laplacian;
  std::uint64_t accepted_configuration_identity = 0;
  std::size_t accepted_parameter_version = 0;
  std::uint64_t accepted_state_requirement = 0;
  double current_sign = 0.0;
  double proposed_sign = 0.0;
  PsiFormerWF::LogValue proposed_log_value;
  ParticleSet::ParticleGradient proposed_gradient;
  ParticleSet::ParticleLaplacian proposed_laplacian;
  std::uint64_t proposed_configuration_identity = 0;
  std::uint64_t proposed_descriptor_fingerprint = 0;
  std::size_t proposed_parameter_version = 0;
  int proposed_particle = -1;
  std::uint64_t proposal_origin = 0;
  bool has_proposal = false;
};

/** Narrow friend seam for the component-only hard-plan allocation gate. */
class TestPsiFormerVirtualBatch
{
public:
  /// Public spelling of the two planned one-electron producer domains.
  enum class SingleProposalOrigin
  {
    CALC_RATIO,
    RATIO_GRADIENT
  };

  /// Exact immutable token required to cancel one planned proposal crowd.
  struct SingleProposalEvidence
  {
    SingleProposalOrigin origin;
    std::size_t active_electron;
    std::size_t parameter_version;
    std::uint64_t transaction_fingerprint;
  };

  static void bindParticleSet(PsiFormerWF& component,
                              const ParticleSet& particles)
  {
    component.bound_particle_set_ = &particles;
  }

  static void useCompleteBatchMemoryAccounting(PsiFormerWF& component)
  {
    component.complete_batch_memory_accounting_for_testing_ = true;
  }

  static PsiFormerAllocationCloneStorage preparedCloneStorage(
      const PsiFormerWF& component)
  {
    return {{component.accepted_gradient_.data(),
             component.accepted_laplacian_.data(),
             component.proposed_gradient_.data(),
             component.proposed_laplacian_.data()},
            {component.accepted_gradient_.size(),
             component.accepted_laplacian_.size(),
             component.proposed_gradient_.size(),
             component.proposed_laplacian_.size()},
            {component.accepted_gradient_.capacity(),
             component.accepted_laplacian_.capacity(),
             component.proposed_gradient_.capacity(),
             component.proposed_laplacian_.capacity()},
            component.hasPreparedBatchExecutionClone(
                component.batch_execution_plan_)};
  }

  static PsiFormerScalarAllocationState scalarAllocationState(
      const PsiFormerWF& component)
  {
    return {component.log_value_,
            component.observed_parameter_version_,
            component.restore_validation_pending_,
            component.accepted_value_valid_,
            component.accepted_gradient_,
            component.accepted_laplacian_,
            component.accepted_configuration_identity_,
            component.accepted_parameter_version_,
            static_cast<std::uint64_t>(
                component.accepted_state_requirement_),
            component.current_sign_,
            component.proposed_sign_,
            component.proposed_log_value_,
            component.proposed_gradient_,
            component.proposed_laplacian_,
            component.proposed_configuration_identity_,
            component.proposed_descriptor_fingerprint_,
            component.proposed_parameter_version_,
            component.proposed_particle_,
            static_cast<std::uint64_t>(component.proposal_origin_),
            component.has_proposal_};
  }

  static PsiFormerWorkspaceDiagnostics cloneWorkspaceDiagnostics(
      const PsiFormerWF& component)
  {
    return component.directWorkspaceDiagnosticsForTesting();
  }

  static PsiFormerCrowdWorkspaceDiagnostics crowdWorkspaceDiagnostics(
      const PsiFormerWF& component,
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list)
  {
    return component.crowdWorkspaceDiagnosticsForTesting(wfc_list);
  }

  static PsiFormerSelectedProposalMapDiagnostics selectedCompactMap(
      const PsiFormerWF& component,
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      std::size_t live_walkers, std::size_t evaluated_rows)
  {
    return component.selectedProposalMapDiagnosticsForTesting(
        wfc_list, live_walkers, evaluated_rows);
  }

  static bool hasCurrentFullAcceptedState(const PsiFormerWF& component,
                                          const ParticleSet& particles)
  {
    const std::size_t version = component.parameterVersion();
    return component.observed_parameter_version_ == version &&
        component.acceptedStateMatches(
            particles, version,
            PsiFormerWF::AcceptedStateRequirement::FULL_SPATIAL);
  }

  static bool hasCurrentAcceptedValue(const PsiFormerWF& component,
                                      const ParticleSet& particles)
  {
    const std::size_t version = component.parameterVersion();
    return component.observed_parameter_version_ == version &&
        component.acceptedStateMatches(
            particles, version,
            PsiFormerWF::AcceptedStateRequirement::VALUE_ONLY);
  }

  static bool hasProposal(const PsiFormerWF& component) noexcept
  {
    return component.has_proposal_;
  }

  static std::size_t plannedSelectedTransactionCount(
      const PsiFormerWF& component) noexcept
  {
    return component.plannedSelectedTransactionCountForTesting();
  }

  static std::size_t plannedSingleTransactionCount(
      const PsiFormerWF& component) noexcept
  {
    return component.plannedSingleTransactionCountForTesting();
  }

  /// Read the exact token published by one successful planned producer.
  static SingleProposalEvidence singleProposalEvidence(
      const PsiFormerWF& component)
  {
    SingleProposalOrigin origin;
    switch (component.proposal_origin_)
    {
    case PsiFormerWF::ProposalOrigin::MW_CALC_RATIO_VALUE:
      origin = SingleProposalOrigin::CALC_RATIO;
      break;
    case PsiFormerWF::ProposalOrigin::MW_RATIO_GRADIENT_ACTIVE:
      origin = SingleProposalOrigin::RATIO_GRADIENT;
      break;
    default:
      throw std::logic_error(
          "PsiFormer allocation test expected a planned one-electron proposal");
    }
    return {origin,
            static_cast<std::size_t>(component.proposed_particle_),
            component.proposed_parameter_version_,
            component.proposed_descriptor_fingerprint_};
  }

  /// Cancel a proposal using only the exact token captured after publication.
  static void cancelPlannedSingleProposal(
      const PsiFormerWF& component,
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const SingleProposalEvidence& evidence)
  {
    const PsiFormerWF::ProposalOrigin origin =
        evidence.origin == SingleProposalOrigin::CALC_RATIO
        ? PsiFormerWF::ProposalOrigin::MW_CALC_RATIO_VALUE
        : PsiFormerWF::ProposalOrigin::MW_RATIO_GRADIENT_ACTIVE;
    component.cancelPlannedSingleProposal(
        wfc_list, p_list, evidence.active_electron, origin,
        evidence.parameter_version, evidence.transaction_fingerprint);
  }

  static void cancelPlannedSelectedProposal(
      const PsiFormerWF& component,
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const MCMultiParticleMoves<CoordsType::POS>& moves,
      std::size_t expected_proposal_version)
  {
    component.cancelPlannedSelectedProposal(
        wfc_list, p_list, moves, expected_proposal_version);
  }
};
} // namespace testing

namespace
{
using namespace testing::psiformer;
using Value = QMCTraits::ValueType;

struct AllocationSnapshot
{
  std::array<std::size_t,
             static_cast<std::size_t>(psiformer_adapter_allocation_audit::Form::COUNT)> counts{};
};

/** Count every replaceable C++ allocation and deallocation form in one warmed call.
 * Direct calls to malloc are outside this language-level interposition seam;
 * pointer/capacity/fingerprint and selected-byte checks below cover the owning
 * PsiFormer storage that such a call could otherwise replace.
 */
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

void checkNoAllocations(const AllocationSnapshot& snapshot,
                        const char* scope = "allocation audit")
{
  using Form = psiformer_adapter_allocation_audit::Form;
  INFO("audited scope: " << scope);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::ORDINARY)] == 0);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::ARRAY)] == 0);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::NOTHROW)] == 0);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::NOTHROW_ARRAY)] == 0);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::ALIGNED)] == 0);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::ALIGNED_ARRAY)] == 0);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::ALIGNED_NOTHROW)] == 0);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::ALIGNED_NOTHROW_ARRAY)] == 0);
  CHECK(snapshot.counts[static_cast<std::size_t>(Form::DEALLOCATION)] == 0);
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

std::unique_ptr<ParticleSet> makeAllocationWalker(
    const SimulationCell& simulation_cell, std::size_t walker_index)
{
  const Geometry geometry = makeGeometry("lih");
  auto particles = std::make_unique<ParticleSet>(simulation_cell);
  particles->setName("allocation_e" + std::to_string(walker_index));
  particles->create({2, 2});
  SpeciesSet& species = particles->getSpeciesSet();
  species.addSpecies("u");
  species.addSpecies("d");
  const int mass = species.addAttribute("mass");
  species(mass, 0) = 1.0;
  species(mass, 1) = 1.0;
  particles->resetGroups();
  for (int electron = 0; electron < particles->getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      particles->R[electron][dimension] =
          geometry.electrons[3 * electron + dimension] +
          0.007 * static_cast<double>(walker_index) *
              static_cast<double>((electron + 1) * (dimension + 1));
  particles->update();
  return particles;
}

/** Minimal component-only crowd used by the hard-plan allocation gate. */
struct PlannedAllocationCrowd
{
  PlannedAllocationCrowd(const GeneratedFiles& files,
                         const SimulationCell& simulation_cell,
                         std::size_t walker_count)
      : leader("pf_planned_allocation", files.parameters.string(),
               files.configuration.string()),
        wfc_list(leader)
  {
    walkers.reserve(walker_count);
    clone_storage.reserve(walker_count > 0 ? walker_count - 1 : 0);
    components.reserve(walker_count);
    walkers.push_back(makeAllocationWalker(simulation_cell, 0));
    components.push_back(&leader);
    for (std::size_t walker = 1; walker < walker_count; ++walker)
    {
      walkers.push_back(makeAllocationWalker(simulation_cell, walker));
      clone_storage.push_back(leader.makeClone(*walkers.back()));
      components.push_back(
          static_cast<PsiFormerWF*>(clone_storage.back().get()));
    }

    p_list = std::make_unique<RefVectorWithLeader<ParticleSet>>(
        *walkers.front());
    for (std::size_t walker = 0; walker < walker_count; ++walker)
    {
      p_list->push_back(*walkers[walker]);
      wfc_list.push_back(*components[walker]);
      testing::TestPsiFormerVirtualBatch::bindParticleSet(
          *components[walker], *walkers[walker]);
      testing::TestPsiFormerVirtualBatch::useCompleteBatchMemoryAccounting(
          *components[walker]);
    }
  }

  PsiFormerWF leader;
  std::vector<std::unique_ptr<ParticleSet>> walkers;
  std::vector<std::unique_ptr<WaveFunctionComponent>> clone_storage;
  std::vector<PsiFormerWF*> components;
  RefVectorWithLeader<WaveFunctionComponent> wfc_list;
  std::unique_ptr<RefVectorWithLeader<ParticleSet>> p_list;
};

std::shared_ptr<const BatchExecutionPlan> makeAllocationPlan(
    PsiFormerWF& component, std::size_t walker_count,
    const std::string& participant_id,
    std::size_t reserve_walker_count = 0,
    bool include_scalar_value = false,
    bool include_lifecycle = false)
{
  if (reserve_walker_count == 0)
    reserve_walker_count = walker_count;

  BatchExecutionRequirements requirements;
  component.contributeBatchExecutionRequirements(requirements);
  requirements.require(BatchExecutionMode::VALUE);
  requirements.require(BatchExecutionMode::ACTIVE_GRADIENT);
  if (include_scalar_value)
    requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  if (include_lifecycle)
  {
    requirements.require(BatchExecutionMode::PREPARE_GROUP);
    requirements.require(BatchExecutionMode::COMPLETE_UPDATES);
  }

  BatchExecutionSelectionInput selection;
  selection.requirements                       = requirements;
  selection.topology.initial_walkers_per_crowd = {walker_count};
  selection.topology.reserve_walkers_per_crowd = {reserve_walker_count};
  selection.topology.run_kind = "psiformer-hard-plan-allocation-gate";
  selection.particle_count    = 4;
  selection.target_coordinate = BatchExecutionTargetCoordinate::POS_ONLY;
  selection.preference.id     = "psiformer-hard-plan-allocation-v1";
  selection.preference.preferred = {
      reserve_walker_count, reserve_walker_count, 1, reserve_walker_count};
  selection.logical_maximum = component.batchExecutionLogicalMaximum(
      {selection.requirements, selection.topology, selection.particle_count,
       selection.active_parameter_count, selection.parameter_derivative_width,
       selection.target_coordinate});

  return std::make_shared<const BatchExecutionPlan>(selectBatchExecutionPlan(
      selection,
      [&component, &participant_id](
          const BatchExecutionPlanningContext& candidate) {
        BatchMemoryContribution contribution =
            component.estimateBatchExecutionMemory(candidate);
        return std::vector<BatchMemoryParticipantContribution>{
            {participant_id, std::move(contribution)}};
      }));
}

void bindAndPrepareAllocationCrowd(
    PlannedAllocationCrowd& crowd,
    const std::shared_ptr<const BatchExecutionPlan>& plan,
    const std::string& participant_id)
{
  const BatchExecutionParticipantPlan participant_plan =
      makeBatchExecutionParticipantPlan(plan, participant_id);
  for (PsiFormerWF* component : crowd.components)
    component->validateBatchExecutionPlanBinding(participant_plan);
  for (PsiFormerWF* component : crowd.components)
    component->bindBatchExecutionPlan(participant_plan);
  for (PsiFormerWF* component : crowd.components)
    component->prepareBatchExecutionClone(participant_plan);
}

struct PlannedAllocationFreeze
{
  std::vector<testing::PsiFormerAllocationCloneStorage> clone_storage;
  std::vector<testing::PsiFormerWorkspaceDiagnostics> clone_workspaces;
  testing::PsiFormerCrowdWorkspaceDiagnostics resource;
  std::size_t collection_cursor = 0;
  std::size_t outstanding_loans = 0;
};

PlannedAllocationFreeze capturePlannedAllocationFreeze(
    PlannedAllocationCrowd& crowd, const ResourceCollection& collection)
{
  PlannedAllocationFreeze snapshot;
  snapshot.clone_storage.reserve(crowd.components.size());
  snapshot.clone_workspaces.reserve(crowd.components.size());
  for (const PsiFormerWF* component : crowd.components)
  {
    snapshot.clone_storage.push_back(
        testing::TestPsiFormerVirtualBatch::preparedCloneStorage(*component));
    snapshot.clone_workspaces.push_back(
        testing::TestPsiFormerVirtualBatch::cloneWorkspaceDiagnostics(
            *component));
  }
  snapshot.resource =
      testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
          crowd.leader, crowd.wfc_list);
  snapshot.collection_cursor = collection.getCursor();
  snapshot.outstanding_loans = collection.getOutstandingLoanCount();
  return snapshot;
}

void checkCloneWorkspaceUnchanged(
    const testing::PsiFormerWorkspaceDiagnostics& actual,
    const testing::PsiFormerWorkspaceDiagnostics& expected)
{
  CHECK(actual.owns_value_workspace == expected.owns_value_workspace);
  CHECK(actual.owns_full_spatial_workspace ==
        expected.owns_full_spatial_workspace);
  CHECK(actual.owns_active_spatial_workspace ==
        expected.owns_active_spatial_workspace);
  CHECK(actual.owns_batch_workspace == expected.owns_batch_workspace);
  CHECK(actual.owns_score_workspace == expected.owns_score_workspace);
  CHECK(actual.owns_kinetic_workspace == expected.owns_kinetic_workspace);
  CHECK(actual.has_prepared_clone_plan == expected.has_prepared_clone_plan);
  CHECK(actual.value_bytes == expected.value_bytes);
  CHECK(actual.full_spatial_bytes == expected.full_spatial_bytes);
  CHECK(actual.active_spatial_bytes == expected.active_spatial_bytes);
  CHECK(actual.batch_bytes == expected.batch_bytes);
  CHECK(actual.score_bytes == expected.score_bytes);
  CHECK(actual.kinetic_bytes == expected.kinetic_bytes);
  CHECK(actual.total_log_gradient_bytes ==
        expected.total_log_gradient_bytes);
  CHECK(actual.scalar_value_publication_bytes ==
        expected.scalar_value_publication_bytes);
  CHECK(actual.accepted_spatial_bytes == expected.accepted_spatial_bytes);
  CHECK(actual.proposed_spatial_bytes == expected.proposed_spatial_bytes);
  CHECK(actual.batch_storage_fingerprint ==
        expected.batch_storage_fingerprint);
  CHECK(actual.batch_workspace_identity == expected.batch_workspace_identity);
  CHECK(actual.scalar_value_publication_identity ==
        expected.scalar_value_publication_identity);
  CHECK(actual.scalar_value_publication_size ==
        expected.scalar_value_publication_size);
  CHECK(actual.scalar_value_publication_capacity ==
        expected.scalar_value_publication_capacity);
  CHECK(actual.prepared_scalar_value_compatibility ==
        expected.prepared_scalar_value_compatibility);
  CHECK(actual.prepared_batch_workspace_identity ==
        expected.prepared_batch_workspace_identity);
  CHECK(actual.prepared_batch_storage_fingerprint ==
        expected.prepared_batch_storage_fingerprint);
  CHECK(actual.prepared_batch_bytes == expected.prepared_batch_bytes);
  CHECK(actual.prepared_scalar_value_publication_identity ==
        expected.prepared_scalar_value_publication_identity);
  CHECK(actual.prepared_scalar_value_publication_size ==
        expected.prepared_scalar_value_publication_size);
  CHECK(actual.prepared_scalar_value_publication_capacity ==
        expected.prepared_scalar_value_publication_capacity);
}

void checkResourceStorageUnchanged(
    const testing::PsiFormerCrowdWorkspaceDiagnostics& actual,
    const testing::PsiFormerCrowdWorkspaceDiagnostics& expected)
{
  CHECK(actual.shared_model_identity == expected.shared_model_identity);
  CHECK(actual.resource_identity == expected.resource_identity);
  CHECK(actual.batch_workspace_identity ==
        expected.batch_workspace_identity);
  CHECK(actual.score_workspace_identity == expected.score_workspace_identity);
  CHECK(actual.kinetic_workspace_identity ==
        expected.kinetic_workspace_identity);
  CHECK(actual.persistent_model_identity == expected.persistent_model_identity);
  CHECK(actual.parameter_version == expected.parameter_version);
  CHECK(actual.batch_bytes == expected.batch_bytes);
  CHECK(actual.score_bytes == expected.score_bytes);
  CHECK(actual.kinetic_bytes == expected.kinetic_bytes);
  CHECK(actual.transient_bytes == expected.transient_bytes);
  CHECK(actual.has_expected_plan == expected.has_expected_plan);
  CHECK(actual.has_prepared_plan == expected.has_prepared_plan);
  CHECK(actual.prepared_plan_identity == expected.prepared_plan_identity);
  CHECK(actual.prepared_plan_fingerprint ==
        expected.prepared_plan_fingerprint);
  CHECK(actual.participant_id == expected.participant_id);
  CHECK(actual.prepared_crowd_index == expected.prepared_crowd_index);
  CHECK(actual.initial_walker_capacity == expected.initial_walker_capacity);
  CHECK(actual.reserve_walker_capacity == expected.reserve_walker_capacity);
  CHECK(actual.prepared_storage_fingerprint ==
        expected.prepared_storage_fingerprint);
  CHECK(actual.current_storage_fingerprint ==
        expected.current_storage_fingerprint);
  CHECK(actual.ratio_arena == expected.ratio_arena);
  CHECK(actual.logical_sizes == expected.logical_sizes);
  CHECK(actual.expected_resource_storage ==
        expected.expected_resource_storage);
  CHECK(actual.actual_resource_storage == expected.actual_resource_storage);
  CHECK(actual.backend_modes == expected.backend_modes);
}

void checkPlannedAllocationFreeze(
    PlannedAllocationCrowd& crowd, const ResourceCollection& collection,
    const BatchExecutionPlan& plan,
    const PlannedAllocationFreeze& expected)
{
  const PlannedAllocationFreeze actual =
      capturePlannedAllocationFreeze(crowd, collection);
  REQUIRE(actual.clone_storage.size() == expected.clone_storage.size());
  REQUIRE(actual.clone_workspaces.size() == expected.clone_workspaces.size());
  for (std::size_t lane = 0; lane < actual.clone_storage.size(); ++lane)
  {
    CHECK(actual.clone_storage[lane].data ==
          expected.clone_storage[lane].data);
    CHECK(actual.clone_storage[lane].sizes ==
          expected.clone_storage[lane].sizes);
    CHECK(actual.clone_storage[lane].capacities ==
          expected.clone_storage[lane].capacities);
    CHECK(actual.clone_storage[lane].exact_marker ==
          expected.clone_storage[lane].exact_marker);
    checkCloneWorkspaceUnchanged(actual.clone_workspaces[lane],
                                 expected.clone_workspaces[lane]);
  }
  checkResourceStorageUnchanged(actual.resource, expected.resource);
  CHECK(actual.collection_cursor == expected.collection_cursor);
  CHECK(actual.outstanding_loans == expected.outstanding_loans);

  REQUIRE(plan.participantEvidence().size() == 1);
  const BatchMemoryBytes selected_clone =
      plan.participantEvidence().front().selected_per_owner.at(
          BatchMemoryCategory::FIXED_CLONE_STATE);
  std::size_t fixed_clone_bytes = 0;
  std::size_t actual_clone_bytes = 0;
  for (const auto& diagnostics : actual.clone_workspaces)
  {
    fixed_clone_bytes += diagnostics.accepted_spatial_bytes +
        diagnostics.proposed_spatial_bytes;
    actual_clone_bytes += diagnostics.accountedBytes() +
        diagnostics.accepted_spatial_bytes +
        diagnostics.proposed_spatial_bytes;
  }
  const std::size_t live_clones = actual.clone_workspaces.size();
  REQUIRE(live_clones > 0);
  REQUIRE(actual.resource.reserve_walker_capacity >= live_clones);
  REQUIRE(fixed_clone_bytes % live_clones == 0);
  REQUIRE(actual_clone_bytes % live_clones == 0);
  const std::size_t selected_fixed_clone_bytes =
      (fixed_clone_bytes / live_clones) *
      actual.resource.reserve_walker_capacity;
  const std::size_t selected_actual_clone_bytes =
      (actual_clone_bytes / live_clones) *
      actual.resource.reserve_walker_capacity;
  CHECK(selected_clone.host == selected_fixed_clone_bytes);
  CHECK(selected_clone.device == 0);

  CHECK(actual.resource.expected_resource_storage ==
        actual.resource.actual_resource_storage);
  const BatchMemoryBytes expected_total =
      actual.resource.expected_resource_storage.total();
  const BatchMemoryBytes replacement =
      actual.resource.expected_resource_storage.at(
          BatchMemoryCategory::REALLOCATION_TRANSIENT);
  REQUIRE(expected_total.host >= replacement.host);
  CHECK(expected_total.device == 0);
  CHECK(replacement.device == 0);
  CHECK(actual.resource.accountedBytes() ==
        expected_total.host - replacement.host);
  const BatchMemoryBytes selected_total =
      plan.participantEvidence().front().selected_per_owner.total();
  REQUIRE(selected_total.host >= expected_total.host);
  REQUIRE(selected_total.device >= expected_total.device);
  CHECK(selected_total.host - expected_total.host ==
        selected_actual_clone_bytes);
  CHECK(selected_total.device - expected_total.device == 0);
  CHECK(actual.resource.prepared_storage_fingerprint ==
        actual.resource.current_storage_fingerprint);
}

template<class VectorType>
bool sameVectorBits(const VectorType& actual, const VectorType& expected)
{
  return actual.size() == expected.size() &&
      (actual.size() == 0 ||
       std::memcmp(actual.data(), expected.data(),
                   actual.size() * sizeof(typename VectorType::value_type)) ==
           0);
}

std::vector<testing::PsiFormerScalarAllocationState>
captureScalarAllocationStates(const PlannedAllocationCrowd& crowd)
{
  std::vector<testing::PsiFormerScalarAllocationState> states;
  states.reserve(crowd.components.size());
  for (const PsiFormerWF* component : crowd.components)
    states.push_back(
        testing::TestPsiFormerVirtualBatch::scalarAllocationState(*component));
  return states;
}

void checkScalarAllocationStatesUnchanged(
    const PlannedAllocationCrowd& crowd,
    const std::vector<testing::PsiFormerScalarAllocationState>& expected)
{
  REQUIRE(crowd.components.size() == expected.size());
  for (std::size_t lane = 0; lane < crowd.components.size(); ++lane)
  {
    const auto actual =
        testing::TestPsiFormerVirtualBatch::scalarAllocationState(
            *crowd.components[lane]);
    const auto& reference = expected[lane];
    CHECK(actual.log_value == reference.log_value);
    CHECK(actual.observed_parameter_version ==
          reference.observed_parameter_version);
    CHECK(actual.restore_validation_pending ==
          reference.restore_validation_pending);
    CHECK(actual.accepted_value_valid == reference.accepted_value_valid);
    CHECK(sameVectorBits(actual.accepted_gradient,
                         reference.accepted_gradient));
    CHECK(sameVectorBits(actual.accepted_laplacian,
                         reference.accepted_laplacian));
    CHECK(actual.accepted_configuration_identity ==
          reference.accepted_configuration_identity);
    CHECK(actual.accepted_parameter_version ==
          reference.accepted_parameter_version);
    CHECK(actual.accepted_state_requirement ==
          reference.accepted_state_requirement);
    CHECK(actual.current_sign == reference.current_sign);
    CHECK(actual.proposed_sign == reference.proposed_sign);
    CHECK(actual.proposed_log_value == reference.proposed_log_value);
    CHECK(sameVectorBits(actual.proposed_gradient,
                         reference.proposed_gradient));
    CHECK(sameVectorBits(actual.proposed_laplacian,
                         reference.proposed_laplacian));
    CHECK(actual.proposed_configuration_identity ==
          reference.proposed_configuration_identity);
    CHECK(actual.proposed_descriptor_fingerprint ==
          reference.proposed_descriptor_fingerprint);
    CHECK(actual.proposed_parameter_version ==
          reference.proposed_parameter_version);
    CHECK(actual.proposed_particle == reference.proposed_particle);
    CHECK(actual.proposal_origin == reference.proposal_origin);
    CHECK(actual.has_proposal == reference.has_proposal);
  }
}

/** Prove every selected lifecycle hook is an allocation-free structural
 * no-op for one full or partial prepared crowd.
 */
void checkPlannedLifecycleAllocations(
    const GeneratedFiles& files, std::size_t walker_count,
    std::size_t reserve_walker_count)
{
  using Probe = testing::TestPsiFormerVirtualBatch;
  REQUIRE(walker_count > 0);
  REQUIRE(reserve_walker_count >= walker_count);

  const SimulationCell simulation_cell;
  PlannedAllocationCrowd crowd(files, simulation_cell, walker_count);
  const std::string population_label =
      std::to_string(walker_count) + "-of-" +
      std::to_string(reserve_walker_count);
  const std::string participant_id =
      "test/psiformer/planned-lifecycle-allocation-" + population_label;
  const auto plan = makeAllocationPlan(
      crowd.leader, walker_count, participant_id, reserve_walker_count,
      false, true);
  bindAndPrepareAllocationCrowd(crowd, plan, participant_id);

  ResourceCollection resource_template(
      "psiformer_planned_lifecycle_allocation_template_" +
      population_label);
  crowd.leader.createResource(resource_template);
  ResourceCollection resource(resource_template);
  resource.prepareBatchResources({plan, 0});
  ResourceCollectionTeamLock<WaveFunctionComponent> resource_lock(
      resource, crowd.wfc_list);

  // Warm each scalar and team entry, including both physical spin groups,
  // before enabling global allocation interposition.
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    crowd.components[lane]->prepareGroup(*crowd.walkers[lane], 0);
    crowd.components[lane]->completeUpdates();
    crowd.components[lane]->prepareGroup(*crowd.walkers[lane], 1);
    crowd.components[lane]->completeUpdates();
  }
  crowd.leader.mw_prepareGroup(crowd.wfc_list, *crowd.p_list, 0);
  crowd.leader.mw_completeUpdates(crowd.wfc_list);
  crowd.leader.mw_prepareGroup(crowd.wfc_list, *crowd.p_list, 1);
  crowd.leader.mw_completeUpdates(crowd.wfc_list);

  const PlannedAllocationFreeze frozen =
      capturePlannedAllocationFreeze(crowd, resource);
  const auto frozen_states = captureScalarAllocationStates(crowd);
  std::vector<ParticleSet::ParticlePos> frozen_positions;
  std::vector<ParticleSet::ParticleGradient> frozen_gradients;
  std::vector<ParticleSet::ParticleLaplacian> frozen_laplacians;
  frozen_positions.reserve(walker_count);
  frozen_gradients.reserve(walker_count);
  frozen_laplacians.reserve(walker_count);
  for (const auto& particles : crowd.walkers)
  {
    frozen_positions.push_back(particles->R);
    frozen_gradients.push_back(particles->G);
    frozen_laplacians.push_back(particles->L);
  }
  const std::size_t parameter_version = crowd.leader.parameterVersion();
  const std::size_t selected_transactions =
      Probe::plannedSelectedTransactionCount(crowd.leader);
  const std::size_t single_transactions =
      Probe::plannedSingleTransactionCount(crowd.leader);

  const auto check_frozen_state = [&] {
    checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);
    const auto current_resource =
        Probe::crowdWorkspaceDiagnostics(crowd.leader, crowd.wfc_list);
    CHECK(current_resource.successful_batch_generation ==
          frozen.resource.successful_batch_generation);
    checkScalarAllocationStatesUnchanged(crowd, frozen_states);
    CHECK(crowd.leader.parameterVersion() == parameter_version);
    CHECK(Probe::plannedSelectedTransactionCount(crowd.leader) ==
          selected_transactions);
    CHECK(Probe::plannedSingleTransactionCount(crowd.leader) ==
          single_transactions);
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      CHECK(sameVectorBits(crowd.walkers[lane]->R,
                           frozen_positions[lane]));
      CHECK(sameVectorBits(crowd.walkers[lane]->G,
                           frozen_gradients[lane]));
      CHECK(sameVectorBits(crowd.walkers[lane]->L,
                           frozen_laplacians[lane]));
    }
  };

  const AllocationSnapshot scalar_prepare_allocations =
      auditAllocations([&] {
        for (std::size_t lane = 0; lane < walker_count; ++lane)
          crowd.components[lane]->prepareGroup(*crowd.walkers[lane], 0);
      });
  checkNoAllocations(scalar_prepare_allocations,
                     "planned scalar prepareGroup");
  check_frozen_state();

  const AllocationSnapshot scalar_complete_allocations =
      auditAllocations([&] {
        for (PsiFormerWF* component : crowd.components)
          component->completeUpdates();
      });
  checkNoAllocations(scalar_complete_allocations,
                     "planned scalar completeUpdates");
  check_frozen_state();

  const AllocationSnapshot team_prepare_allocations =
      auditAllocations([&] {
        crowd.leader.mw_prepareGroup(crowd.wfc_list, *crowd.p_list, 1);
      });
  checkNoAllocations(team_prepare_allocations,
                     "planned team prepareGroup");
  check_frozen_state();

  const AllocationSnapshot team_complete_allocations =
      auditAllocations([&] {
        crowd.leader.mw_completeUpdates(crowd.wfc_list);
      });
  checkNoAllocations(team_complete_allocations,
                     "planned team completeUpdates");
  check_frozen_state();

  const AllocationSnapshot alternating_allocations = auditAllocations([&] {
    for (int repetition = 0; repetition < 3; ++repetition)
    {
      for (std::size_t lane = 0; lane < walker_count; ++lane)
      {
        crowd.components[lane]->prepareGroup(*crowd.walkers[lane],
                                              repetition % 2);
        crowd.components[lane]->completeUpdates();
      }
      crowd.leader.mw_prepareGroup(crowd.wfc_list, *crowd.p_list,
                                   repetition % 2);
      crowd.leader.mw_completeUpdates(crowd.wfc_list);
    }
  });
  checkNoAllocations(alternating_allocations,
                     "alternating scalar/team lifecycle calls");
  check_frozen_state();
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

/** Exercise every planned one-electron transaction boundary after warming all
 * executors.  Only the named component call is inside each allocation window;
 * ParticleSet proposal construction and resolution remain explicit outside it.
 */
void checkPlannedSingleTransactionAllocations(
    const GeneratedFiles& files, std::size_t walker_count,
    std::size_t reserve_walker_count)
{
  using Probe = testing::TestPsiFormerVirtualBatch;
  constexpr int active_electron = 1;
  const bool uses_reserve_prefix = reserve_walker_count > walker_count;
  REQUIRE(walker_count > 0);
  REQUIRE(reserve_walker_count >= walker_count);
  const std::string population_label = walker_count == 1
      ? "singleton"
      : uses_reserve_prefix ? "prefix" : "full";
  const std::string population_notation = walker_count == 1
      ? "b=1"
      : uses_reserve_prefix ? "b<B" : "b=B";

  const SimulationCell simulation_cell;
  PlannedAllocationCrowd crowd(files, simulation_cell, walker_count);
  const std::string participant_id =
      "test/psiformer/planned-single-allocation-" + population_label;
  const auto plan = makeAllocationPlan(
      crowd.leader, walker_count, participant_id, reserve_walker_count);
  bindAndPrepareAllocationCrowd(crowd, plan, participant_id);

  ResourceCollection particle_resource(
      "psiformer_planned_single_allocation_" + population_label +
      "_particles");
  crowd.walkers.front()->createResource(particle_resource);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(
      particle_resource, *crowd.p_list);

  ResourceCollection resource_template(
      "psiformer_planned_single_allocation_" + population_label +
      "_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection resource(resource_template);
  resource.prepareBatchResources({plan, 0});
  ResourceCollectionTeamLock<WaveFunctionComponent> resource_lock(
      resource, crowd.wfc_list);

  const std::size_t electron_count =
      static_cast<std::size_t>(crowd.walkers.front()->getTotalNum());
  std::vector<ParticleSet::ParticleGradient> accepted_gradients(walker_count);
  std::vector<ParticleSet::ParticleLaplacian> accepted_laplacians(walker_count);
  RefVector<ParticleSet::ParticleGradient> accepted_gradient_list;
  RefVector<ParticleSet::ParticleLaplacian> accepted_laplacian_list;
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    accepted_gradients[lane].resize(electron_count);
    accepted_laplacians[lane].resize(electron_count);
    accepted_gradients[lane] = Value(0);
    accepted_laplacians[lane] = Value(0);
    accepted_gradient_list.push_back(accepted_gradients[lane]);
    accepted_laplacian_list.push_back(accepted_laplacians[lane]);
  }
  crowd.leader.mw_evaluateLog(crowd.wfc_list, *crowd.p_list,
                              accepted_gradient_list,
                              accepted_laplacian_list);

  std::vector<ParticleSet::SingleParticlePos> displacements(walker_count);
  const auto make_move = [&](double scale) {
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      const double lane_scale = scale * static_cast<double>(lane + 1);
      displacements[lane] =
          ParticleSet::SingleParticlePos{lane_scale, -0.6 * lane_scale,
                                         0.4 * lane_scale};
    }
    ParticleSet::mw_makeMove(*crowd.p_list, active_electron, displacements);
  };

  std::vector<Value> ratios(walker_count, Value(0));
  std::vector<PsiFormerWF::GradType> ratio_gradients(walker_count);
  const std::vector<bool> all_rejected(walker_count, false);

  // Warm the value producer and exact cancellation path together, then clear
  // the independent ParticleSet active-move state outside any audit window.
  make_move(0.0010);
  crowd.leader.mw_calcRatio(crowd.wfc_list, *crowd.p_list, active_electron,
                            ratios);
  const auto warm_value_evidence = Probe::singleProposalEvidence(crowd.leader);
  Probe::cancelPlannedSingleProposal(
      crowd.leader, crowd.wfc_list, *crowd.p_list, warm_value_evidence);
  ParticleSet::mw_accept_rejectMove<CoordsType::POS>(
      *crowd.p_list, active_electron, all_rejected);

  // Warm the gradient producer and resolver with a rejection, preserving a
  // FULL accepted baseline for the later rejection-preservation check.
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    ratio_gradients[lane] =
        PsiFormerWF::GradType(Value(0.1 + lane), Value(-0.2), Value(0.3));
  make_move(-0.0008);
  crowd.leader.mw_ratioGrad(crowd.wfc_list, *crowd.p_list, active_electron,
                            ratios, ratio_gradients);
  crowd.leader.mw_accept_rejectMove(
      crowd.wfc_list, *crowd.p_list, active_electron, all_rejected, true);
  ParticleSet::mw_accept_rejectMove<CoordsType::POS>(
      *crowd.p_list, active_electron, all_rejected);

  REQUIRE(Probe::plannedSingleTransactionCount(crowd.leader) == 0);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    REQUIRE_FALSE(Probe::hasProposal(*crowd.components[lane]));
    REQUIRE(Probe::hasCurrentFullAcceptedState(
        *crowd.components[lane], *crowd.walkers[lane]));
  }
  const PlannedAllocationFreeze frozen =
      capturePlannedAllocationFreeze(crowd, resource);
  CHECK(frozen.resource.initial_walker_capacity == walker_count);
  CHECK(frozen.resource.reserve_walker_capacity == reserve_walker_count);
  REQUIRE(frozen.resource.ratio_arena.prepared_kind !=
          testing::PsiFormerRatioArenaKind::NONE);
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);

  // Independent unplanned scalar components provide numerical oracles without
  // sharing accepted state, proposal metadata, or prepared crowd scratch.
  std::vector<std::unique_ptr<PsiFormerWF>> references;
  std::vector<ParticleSet::ParticleGradient> reference_gradients(walker_count);
  std::vector<ParticleSet::ParticleLaplacian> reference_laplacians(
      walker_count);
  references.reserve(walker_count);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    references.push_back(std::make_unique<PsiFormerWF>(
        "pf_planned_single_allocation_reference_" + std::to_string(lane),
        files.parameters.string(), files.configuration.string()));
    reference_gradients[lane].resize(electron_count);
    reference_laplacians[lane].resize(electron_count);
  }
  const auto refresh_reference_accepted_state = [&] {
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      reference_gradients[lane] = Value(0);
      reference_laplacians[lane] = Value(0);
      references[lane]->evaluateLog(
          *crowd.walkers[lane], reference_gradients[lane],
          reference_laplacians[lane]);
    }
  };
  refresh_reference_accepted_state();

  const std::string calc_ratio_scope =
      "planned CALC_RATIO " + population_notation;
  const std::string cancellation_scope =
      "planned single cancellation " + population_notation;
  const std::string ratio_grad_scope =
      "planned RATIO_GRADIENT " + population_notation;
  const std::string resolution_scope =
      "planned single resolution " + population_notation;
  const std::string rejection_scope =
      "planned all-reject resolution " + population_notation;

  Value* const ratio_data = ratios.data();
  const std::size_t ratio_capacity = ratios.capacity();
  std::vector<Value> expected_ratios(walker_count);
  std::vector<PsiFormerWF::LogValue> accepted_logs_before_value(walker_count);
  std::fill(ratios.begin(), ratios.end(), Value(-37.0));
  make_move(0.0013);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    accepted_logs_before_value[lane] =
        crowd.components[lane]->get_log_value();
    expected_ratios[lane] =
        references[lane]->ratio(*crowd.walkers[lane], active_electron);
    references[lane]->restore(active_electron);
  }
  const AllocationSnapshot calc_ratio_allocations = auditAllocations([&] {
    crowd.leader.mw_calcRatio(crowd.wfc_list, *crowd.p_list,
                              active_electron, ratios);
  });
  checkNoAllocations(calc_ratio_allocations, calc_ratio_scope.c_str());
  REQUIRE(ratios.size() == walker_count);
  CHECK(ratios.data() == ratio_data);
  CHECK(ratios.capacity() == ratio_capacity);
  CHECK(Probe::plannedSingleTransactionCount(crowd.leader) == 1);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    checkObservation(observe(ratios[lane]), observe(expected_ratios[lane]),
                     2.0e-8);
    CHECK(Probe::hasProposal(*crowd.components[lane]));
  }
  const auto value_evidence = Probe::singleProposalEvidence(crowd.leader);
  CHECK(value_evidence.origin == Probe::SingleProposalOrigin::CALC_RATIO);
  CHECK(value_evidence.active_electron == active_electron);
  CHECK(value_evidence.parameter_version == crowd.leader.parameterVersion());
  CHECK(value_evidence.transaction_fingerprint != 0);
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);

  const AllocationSnapshot cancellation_allocations = auditAllocations([&] {
    Probe::cancelPlannedSingleProposal(
        crowd.leader, crowd.wfc_list, *crowd.p_list, value_evidence);
  });
  checkNoAllocations(cancellation_allocations, cancellation_scope.c_str());
  CHECK(Probe::plannedSingleTransactionCount(crowd.leader) == 0);
  REQUIRE(crowd.components.size() == walker_count);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    CHECK_FALSE(Probe::hasProposal(*crowd.components[lane]));
    CHECK(crowd.components[lane]->get_log_value() ==
          accepted_logs_before_value[lane]);
  }
  ParticleSet::mw_accept_rejectMove<CoordsType::POS>(
      *crowd.p_list, active_electron, all_rejected);
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);

  // The gradient output is additive.  Seed it before entering the measured
  // window and retain its identity to catch either hidden resizing or swap.
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    ratios[lane] = Value(-41.0);
    ratio_gradients[lane] =
        PsiFormerWF::GradType(Value(0.25 + lane), Value(-0.5), Value(0.75));
  }
  std::vector<PsiFormerWF::GradType> expected_ratio_gradients =
      ratio_gradients;
  std::vector<PsiFormerWF::LogValue> expected_resolved_logs(walker_count);
  std::vector<bool> resolution(walker_count, false);
  resolution[0] = true;
  PsiFormerWF::GradType* const gradient_data = ratio_gradients.data();
  const std::size_t gradient_capacity = ratio_gradients.capacity();
  make_move(-0.0011);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    expected_ratios[lane] = references[lane]->ratioGrad(
        *crowd.walkers[lane], active_electron,
        expected_ratio_gradients[lane]);
    if (resolution[lane])
      references[lane]->acceptMove(
          *crowd.walkers[lane], active_electron, true);
    else
      references[lane]->restore(active_electron);
    expected_resolved_logs[lane] = references[lane]->get_log_value();
  }
  const AllocationSnapshot ratio_grad_allocations = auditAllocations([&] {
    crowd.leader.mw_ratioGrad(crowd.wfc_list, *crowd.p_list,
                              active_electron, ratios, ratio_gradients);
  });
  checkNoAllocations(ratio_grad_allocations, ratio_grad_scope.c_str());
  REQUIRE(ratios.size() == walker_count);
  REQUIRE(ratio_gradients.size() == walker_count);
  CHECK(ratios.data() == ratio_data);
  CHECK(ratios.capacity() == ratio_capacity);
  CHECK(ratio_gradients.data() == gradient_data);
  CHECK(ratio_gradients.capacity() == gradient_capacity);
  CHECK(Probe::plannedSingleTransactionCount(crowd.leader) == 1);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    checkObservation(observe(ratios[lane]), observe(expected_ratios[lane]),
                     2.0e-8);
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      checkObservation(
          observe(ratio_gradients[lane][dimension]),
          observe(expected_ratio_gradients[lane][dimension]), 2.0e-8);
    CHECK(Probe::hasProposal(*crowd.components[lane]));
  }
  const auto gradient_evidence =
      Probe::singleProposalEvidence(crowd.leader);
  CHECK(gradient_evidence.origin ==
        Probe::SingleProposalOrigin::RATIO_GRADIENT);
  CHECK(gradient_evidence.active_electron == active_electron);
  CHECK(gradient_evidence.parameter_version ==
        crowd.leader.parameterVersion());
  CHECK(gradient_evidence.transaction_fingerprint != 0);
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);

  // For multi-lane crowds, resolve one accepted and the remaining rejected so
  // one measured call covers VALUE_ONLY promotion and exact preservation.
  const AllocationSnapshot resolution_allocations = auditAllocations([&] {
    crowd.leader.mw_accept_rejectMove(
        crowd.wfc_list, *crowd.p_list, active_electron, resolution, true);
  });
  checkNoAllocations(resolution_allocations, resolution_scope.c_str());
  CHECK(Probe::plannedSingleTransactionCount(crowd.leader) == 0);
  REQUIRE(crowd.components.size() == walker_count);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    CHECK_FALSE(Probe::hasProposal(*crowd.components[lane]));
    checkObservation(observe(crowd.components[lane]->get_log_value()),
                     observe(expected_resolved_logs[lane]), 2.0e-8);
  }
  ParticleSet::mw_accept_rejectMove<CoordsType::POS>(
      *crowd.p_list, active_electron, resolution);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    if (resolution[lane])
    {
      CHECK(Probe::hasCurrentAcceptedValue(
          *crowd.components[lane], *crowd.walkers[lane]));
      CHECK_FALSE(Probe::hasCurrentFullAcceptedState(
          *crowd.components[lane], *crowd.walkers[lane]));
    }
    else
      CHECK(Probe::hasCurrentFullAcceptedState(
          *crowd.components[lane], *crowd.walkers[lane]));
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);

  // Audit an all-reject transaction independently of the all-accept/mixed
  // resolution above.  Rejection must preserve every accepted VALUE/FULL
  // record while clearing the proposal crowd-wide without allocating.
  std::vector<PsiFormerWF::LogValue> accepted_logs_before_rejection(
      walker_count);
  make_move(0.0007);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    accepted_logs_before_rejection[lane] =
        crowd.components[lane]->get_log_value();
  crowd.leader.mw_calcRatio(crowd.wfc_list, *crowd.p_list, active_electron,
                            ratios);
  REQUIRE(Probe::plannedSingleTransactionCount(crowd.leader) == 1);
  const AllocationSnapshot rejection_allocations = auditAllocations([&] {
    crowd.leader.mw_accept_rejectMove(crowd.wfc_list, *crowd.p_list,
                                      active_electron, all_rejected, true);
  });
  checkNoAllocations(rejection_allocations, rejection_scope.c_str());
  CHECK(Probe::plannedSingleTransactionCount(crowd.leader) == 0);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    CHECK_FALSE(Probe::hasProposal(*crowd.components[lane]));
    CHECK(crowd.components[lane]->get_log_value() ==
          accepted_logs_before_rejection[lane]);
  }
  ParticleSet::mw_accept_rejectMove<CoordsType::POS>(
      *crowd.p_list, active_electron, all_rejected);
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);

  // Restore the strongest accepted-state baseline before fixture teardown.
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    accepted_gradients[lane] = Value(0);
    accepted_laplacians[lane] = Value(0);
  }
  crowd.leader.mw_evaluateLog(crowd.wfc_list, *crowd.p_list,
                              accepted_gradient_list,
                              accepted_laplacian_list);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    CHECK(Probe::hasCurrentFullAcceptedState(
        *crowd.components[lane], *crowd.walkers[lane]));
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);
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

TEST_CASE("PsiFormer warmed hard-plan transactions freeze selected storage",
          "[wavefunction][psiformer][allocation][batch_memory]")
{
  using Probe = testing::TestPsiFormerVirtualBatch;
  using Moves = MCMultiParticleMoves<CoordsType::POS>;
  constexpr std::size_t walker_count = 2;

  GeneratedFiles files = generateFiles("lih");
  setBackend("direct");
  const SimulationCell simulation_cell;
  PlannedAllocationCrowd crowd(files, simulation_cell, walker_count);
  PsiFormerWF reference("pf_hard_plan_allocation_reference",
                        files.parameters.string(),
                        files.configuration.string());
  const std::string participant_id =
      "test/psiformer/hard-plan-allocation";
  const auto plan = makeAllocationPlan(
      crowd.leader, walker_count, participant_id);
  bindAndPrepareAllocationCrowd(crowd, plan, participant_id);

  ResourceCollection particle_resource(
      "psiformer_hard_plan_allocation_particles");
  crowd.walkers.front()->createResource(particle_resource);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(
      particle_resource, *crowd.p_list);

  ResourceCollection resource_template(
      "psiformer_hard_plan_allocation_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection resource(resource_template);
  resource.prepareBatchResources({plan, 0});
  ResourceCollectionTeamLock<WaveFunctionComponent> resource_lock(
      resource, crowd.wfc_list);

  const std::size_t electron_count =
      static_cast<std::size_t>(crowd.walkers.front()->getTotalNum());
  std::vector<ParticleSet::ParticleGradient> gradients(walker_count);
  std::vector<ParticleSet::ParticleLaplacian> laplacians(walker_count);
  RefVector<ParticleSet::ParticleGradient> gradient_list;
  RefVector<ParticleSet::ParticleLaplacian> laplacian_list;
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    gradients[lane].resize(electron_count);
    laplacians[lane].resize(electron_count);
    gradient_list.push_back(gradients[lane]);
    laplacian_list.push_back(laplacians[lane]);
  }
  std::vector<PsiFormerWF::LogValue> log_ratios(walker_count);
  std::vector<PsiFormerWF::GradType> active_gradients(walker_count);
  std::vector<PsiFormerWF::GradType> expected_active_gradients(walker_count);
  std::vector<PsiFormerWF::LogValue> recompute_logs_before(walker_count);
  std::vector<PsiFormerWF::LogValue> expected_recompute_logs(walker_count);
  ParticleSet::ParticleGradient reference_gradient;
  ParticleSet::ParticleLaplacian reference_laplacian;
  reference_gradient.resize(electron_count);
  reference_laplacian.resize(electron_count);
  auto reset_outputs = [&]() {
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      gradients[lane] = Value(0);
      laplacians[lane] = Value(0);
      log_ratios[lane] = PsiFormerWF::LogValue(0);
    }
  };

  Moves::PosType moved = crowd.walkers[1]->R[0];
  moved[0] += 0.004;
  moved[1] -= 0.002;
  const std::array<Moves::PosType, walker_count> baseline_positions{
      crowd.walkers[0]->R[0], crowd.walkers[1]->R[0]};
  const Moves moves(
      {0, 1, 2}, {0, 0}, {crowd.walkers[0]->R[0], moved});
  const std::vector<bool> all_reject{false, false};
  const std::vector<bool> mixed_resolution{false, true};
  const std::vector<bool> recompute_none{false, false};
  const std::vector<bool> recompute_sparse{true, false};
  const std::vector<bool> recompute_all{true, true};

  // Warm the executor, proposal bookkeeping, both abandonment routes, and
  // resource ownership transitions before enabling global-new accounting.
  reset_outputs();
  crowd.leader.mw_evaluateLog(crowd.wfc_list, *crowd.p_list,
                              gradient_list, laplacian_list);
  reset_outputs();
  crowd.leader.mw_evaluateMultiParticleMove(
      crowd.wfc_list, *crowd.p_list, moves, log_ratios, gradient_list,
      laplacian_list);
  Probe::cancelPlannedSelectedProposal(
      crowd.leader, crowd.wfc_list, *crowd.p_list, moves,
      crowd.leader.parameterVersion());
  reset_outputs();
  crowd.leader.mw_evaluateMultiParticleMove(
      crowd.wfc_list, *crowd.p_list, moves, log_ratios, gradient_list,
      laplacian_list);
  crowd.leader.mw_accept_rejectMultiParticleMove(
      crowd.wfc_list, *crowd.p_list, moves, all_reject);
  resource.rewind(0);
  crowd.leader.releaseResource(resource, crowd.wfc_list);
  resource.rewind(0);
  crowd.leader.acquireResource(resource, crowd.wfc_list);

  // A post-reacquire FULL call establishes the exact accepted baseline used
  // by every audited selected transaction.
  reset_outputs();
  crowd.leader.mw_evaluateLog(crowd.wfc_list, *crowd.p_list,
                              gradient_list, laplacian_list);
  crowd.leader.mw_recompute(crowd.wfc_list, *crowd.p_list,
                            recompute_none);
  crowd.leader.mw_recompute(crowd.wfc_list, *crowd.p_list,
                            recompute_sparse);
  crowd.leader.mw_recompute(crowd.wfc_list, *crowd.p_list,
                            recompute_all);
  crowd.leader.mw_evalGrad(crowd.wfc_list, *crowd.p_list, 1,
                           active_gradients);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    CHECK(Probe::hasCurrentFullAcceptedState(
        *crowd.components[lane], *crowd.walkers[lane]));

  const PlannedAllocationFreeze frozen =
      capturePlannedAllocationFreeze(crowd, resource);
  REQUIRE(frozen.clone_storage.size() == walker_count);
  for (const auto& storage : frozen.clone_storage)
  {
    CHECK(storage.exact_marker);
    for (std::size_t field = 0; field < storage.data.size(); ++field)
    {
      CHECK(storage.data[field] != nullptr);
      CHECK(storage.sizes[field] == electron_count);
      CHECK(storage.capacities[field] == electron_count);
    }
  }
  CHECK(frozen.resource.initial_walker_capacity == walker_count);
  CHECK(frozen.resource.reserve_walker_capacity == walker_count);
  REQUIRE(plan->participantEvidence().size() == 1);
  CHECK(plan->participantEvidence().front().fully_accounted);
  CHECK(frozen.resource.has_expected_plan);
  CHECK(frozen.resource.has_prepared_plan);
  CHECK(frozen.resource.prepared_plan_identity == plan.get());
  CHECK(frozen.resource.prepared_plan_fingerprint != 0);
  CHECK(frozen.resource.prepared_storage_fingerprint != 0);
  CHECK(frozen.resource.prepared_storage_fingerprint ==
        frozen.resource.current_storage_fingerprint);
  CHECK(frozen.collection_cursor == 1);
  CHECK(frozen.outstanding_loans == 1);
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);

  reset_outputs();
  const AllocationSnapshot full_allocations = auditAllocations([&] {
    crowd.leader.mw_evaluateLog(crowd.wfc_list, *crowd.p_list,
                                gradient_list, laplacian_list);
  });
  checkNoAllocations(full_allocations, "planned FULL");
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    CHECK(Probe::hasCurrentFullAcceptedState(
        *crowd.components[lane], *crowd.walkers[lane]));
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);

  const auto audit_recompute = [&](const std::vector<bool>& mask,
                                   const char* scope,
                                   double coordinate_shift) {
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      recompute_logs_before[lane] = crowd.components[lane]->get_log_value();
      expected_recompute_logs[lane] = recompute_logs_before[lane];
      if (!mask[lane])
        continue;

      // Make a selected lane's accepted value stale, then obtain an
      // independent scalar-direct reference before opening the allocation
      // window.  A no-op planned recompute can no longer satisfy this test.
      crowd.walkers[lane]->R[0][0] +=
          coordinate_shift * static_cast<double>(lane + 1);
      crowd.walkers[lane]->update();
      reference_gradient = Value(0);
      reference_laplacian = Value(0);
      expected_recompute_logs[lane] = reference.evaluateLog(
          *crowd.walkers[lane], reference_gradient, reference_laplacian);
      REQUIRE(std::abs(expected_recompute_logs[lane] -
                       recompute_logs_before[lane]) > 1.0e-10);
    }

    const AllocationSnapshot allocations = auditAllocations([&] {
      crowd.leader.mw_recompute(crowd.wfc_list, *crowd.p_list, mask);
    });
    checkNoAllocations(allocations, scope);
    CHECK(Probe::plannedSelectedTransactionCount(crowd.leader) == 0);
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      if (mask[lane])
      {
        CHECK(Probe::hasCurrentAcceptedValue(
            *crowd.components[lane], *crowd.walkers[lane]));
        CHECK_FALSE(Probe::hasCurrentFullAcceptedState(
            *crowd.components[lane], *crowd.walkers[lane]));
        checkObservation(observe(crowd.components[lane]->get_log_value()),
                         observe(expected_recompute_logs[lane]));
        CHECK(std::abs(crowd.components[lane]->get_log_value() -
                       recompute_logs_before[lane]) > 1.0e-10);
      }
      else
      {
        CHECK(Probe::hasCurrentFullAcceptedState(
            *crowd.components[lane], *crowd.walkers[lane]));
        CHECK(crowd.components[lane]->get_log_value() ==
              recompute_logs_before[lane]);
      }
      CHECK_FALSE(Probe::hasProposal(*crowd.components[lane]));
    }
    checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);
  };
  audit_recompute(recompute_none, "planned RECOMPUTE_VALUE q=0", 0.0);
  audit_recompute(recompute_sparse, "planned RECOMPUTE_VALUE sparse",
                  0.003);
  audit_recompute(recompute_all, "planned RECOMPUTE_VALUE all", -0.002);

  // Restore complete accepted spatial state after the VALUE-only refreshes so
  // the following ACTIVE audit begins from the same strong baseline as the
  // rest of the selected-transaction checks.
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    crowd.walkers[lane]->R[0] = baseline_positions[lane];
    crowd.walkers[lane]->update();
  }
  reset_outputs();
  crowd.leader.mw_evaluateLog(crowd.wfc_list, *crowd.p_list,
                              gradient_list, laplacian_list);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    CHECK(Probe::hasCurrentFullAcceptedState(
        *crowd.components[lane], *crowd.walkers[lane]));
    expected_active_gradients[lane] =
        reference.evalGrad(*crowd.walkers[lane], 1);
    for (int dimension = 0; dimension < 3; ++dimension)
      active_gradients[lane][dimension] =
          Value(-1700.0 - 10.0 * static_cast<double>(lane) - dimension);
  }

  PsiFormerWF::GradType* const active_gradient_data = active_gradients.data();
  const std::size_t active_gradient_capacity = active_gradients.capacity();
  const AllocationSnapshot active_gradient_allocations = auditAllocations([&] {
    crowd.leader.mw_evalGrad(crowd.wfc_list, *crowd.p_list, 1,
                             active_gradients);
  });
  checkNoAllocations(active_gradient_allocations,
                     "planned ACTIVE_GRADIENT b=B");
  CHECK(active_gradients.data() == active_gradient_data);
  CHECK(active_gradients.size() == walker_count);
  CHECK(active_gradients.capacity() == active_gradient_capacity);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      const Value sentinel(
          -1700.0 - 10.0 * static_cast<double>(lane) - dimension);
      CHECK(active_gradients[lane][dimension] != sentinel);
      checkObservation(observe(active_gradients[lane][dimension]),
                       observe(expected_active_gradients[lane][dimension]),
                       2.0e-8);
    }
  CHECK(Probe::plannedSelectedTransactionCount(crowd.leader) == 0);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    CHECK(Probe::hasCurrentFullAcceptedState(
        *crowd.components[lane], *crowd.walkers[lane]));
    CHECK_FALSE(Probe::hasProposal(*crowd.components[lane]));
  }
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);

  // Lane one differs from its accepted coordinates, so this is a real q=1
  // direct batch rather than the accepted-state reuse fast path.
  reset_outputs();
  const AllocationSnapshot proposal_allocations = auditAllocations([&] {
    crowd.leader.mw_evaluateMultiParticleMove(
        crowd.wfc_list, *crowd.p_list, moves, log_ratios, gradient_list,
        laplacian_list);
  });
  checkNoAllocations(proposal_allocations, "selected proposal");
  CHECK(Probe::plannedSelectedTransactionCount(crowd.leader) == 1);
  for (const PsiFormerWF* component : crowd.components)
    CHECK(Probe::hasProposal(*component));
  const auto compact = Probe::selectedCompactMap(
      crowd.leader, crowd.wfc_list, walker_count, 1);
  CHECK(compact.batch_slots ==
        std::vector<std::size_t>{std::numeric_limits<std::size_t>::max(), 0});
  CHECK(compact.walker_indices == std::vector<std::size_t>{1});
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);
  Probe::cancelPlannedSelectedProposal(
      crowd.leader, crowd.wfc_list, *crowd.p_list, moves,
      crowd.leader.parameterVersion());

  // Cancellation has its own counter window; proposal construction remains
  // outside it so the result cannot hide either operation's allocations.
  reset_outputs();
  crowd.leader.mw_evaluateMultiParticleMove(
      crowd.wfc_list, *crowd.p_list, moves, log_ratios, gradient_list,
      laplacian_list);
  const std::size_t cancellation_version = crowd.leader.parameterVersion();
  const AllocationSnapshot cancellation_allocations = auditAllocations([&] {
    Probe::cancelPlannedSelectedProposal(
        crowd.leader, crowd.wfc_list, *crowd.p_list, moves,
        cancellation_version);
  });
  checkNoAllocations(cancellation_allocations, "selected cancellation");
  CHECK(Probe::plannedSelectedTransactionCount(crowd.leader) == 0);
  for (const PsiFormerWF* component : crowd.components)
    CHECK_FALSE(Probe::hasProposal(*component));
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);

  const AllocationSnapshot release_reacquire_allocations =
      auditAllocations([&] {
        resource.rewind(0);
        crowd.leader.releaseResource(resource, crowd.wfc_list);
        resource.rewind(0);
        crowd.leader.acquireResource(resource, crowd.wfc_list);
      });
  checkNoAllocations(release_reacquire_allocations, "release/reacquire");
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    CHECK(Probe::hasCurrentFullAcceptedState(
        *crowd.components[lane], *crowd.walkers[lane]));
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);

  // Install the accepted lane's ParticleSet proposal before the resolution
  // window.  The measured scope then includes accepted-state G/L promotion
  // while the rejected lane preserves its old accepted state.
  reset_outputs();
  crowd.leader.mw_evaluateMultiParticleMove(
      crowd.wfc_list, *crowd.p_list, moves, log_ratios, gradient_list,
      laplacian_list);
  std::vector<bool> position_valid;
  ParticleSet::mw_makeMoveSelectedParticles(
      *crowd.p_list, moves, position_valid);
  REQUIRE(position_valid.size() == walker_count);
  CHECK(std::all_of(position_valid.begin(), position_valid.end(),
                    [](bool valid) { return valid; }));
  const AllocationSnapshot resolution_allocations = auditAllocations([&] {
    crowd.leader.mw_accept_rejectMultiParticleMove(
        crowd.wfc_list, *crowd.p_list, moves, mixed_resolution);
  });
  checkNoAllocations(resolution_allocations, "selected resolution");
  CHECK(Probe::plannedSelectedTransactionCount(crowd.leader) == 0);
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);
  ParticleSet::mw_accept_rejectMoveSelectedParticles(
      *crowd.p_list, mixed_resolution);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    CHECK(Probe::hasCurrentFullAcceptedState(
        *crowd.components[lane], *crowd.walkers[lane]));

  // This remains only a component test seam: scalar dispatch rejects before
  // output mutation, and the inherited production publication capability
  // keeps TrialWaveFunction/ParticleSet entry closed.
  CHECK_FALSE(crowd.leader.supportsAtomicBatchPublication());
  CHECK(Probe::plannedSelectedTransactionCount(crowd.leader) == 0);
  const ParticleSet::ParticleGradient scalar_gradient_before = gradients[0];
  const ParticleSet::ParticleLaplacian scalar_laplacian_before =
      laplacians[0];
  CHECK_THROWS_AS(crowd.leader.evaluateLog(
                      *crowd.walkers[0], gradients[0], laplacians[0]),
                  std::logic_error);
  CHECK(sameVectorBits(gradients[0], scalar_gradient_before));
  CHECK(sameVectorBits(laplacians[0], scalar_laplacian_before));
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);
}

TEST_CASE("PsiFormer planned lifecycle hooks allocate no storage",
          "[wavefunction][psiformer][allocation][batch_memory][lifecycle]")
{
  GeneratedFiles files = generateFiles("lih");
  setBackend("direct");

  SECTION("singleton live crowd")
  {
    checkPlannedLifecycleAllocations(files, 1, 1);
  }

  SECTION("live crowd is a strict prepared-reserve prefix")
  {
    checkPlannedLifecycleAllocations(files, 2, 3);
  }

  SECTION("live crowd fills its prepared reserve")
  {
    checkPlannedLifecycleAllocations(files, 3, 3);
  }
}

TEST_CASE("PsiFormer warmed hard-plan active gradient freezes reserve storage",
          "[wavefunction][psiformer][allocation][batch_memory]")
{
  using Probe = testing::TestPsiFormerVirtualBatch;
  constexpr std::size_t walker_count = 2;
  constexpr std::size_t reserve_walker_count = 3;

  GeneratedFiles files = generateFiles("lih");
  setBackend("direct");
  const SimulationCell simulation_cell;
  PlannedAllocationCrowd crowd(files, simulation_cell, walker_count);
  PsiFormerWF reference("pf_hard_plan_active_gradient_reserve_reference",
                        files.parameters.string(),
                        files.configuration.string());
  const std::string participant_id =
      "test/psiformer/hard-plan-active-gradient-reserve";
  const auto plan = makeAllocationPlan(
      crowd.leader, walker_count, participant_id, reserve_walker_count);
  bindAndPrepareAllocationCrowd(crowd, plan, participant_id);

  ResourceCollection particle_resource(
      "psiformer_hard_plan_active_gradient_reserve_particles");
  crowd.walkers.front()->createResource(particle_resource);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(
      particle_resource, *crowd.p_list);

  ResourceCollection resource_template(
      "psiformer_hard_plan_active_gradient_reserve_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection resource(resource_template);
  resource.prepareBatchResources({plan, 0});
  ResourceCollectionTeamLock<WaveFunctionComponent> resource_lock(
      resource, crowd.wfc_list);

  const std::size_t electron_count =
      static_cast<std::size_t>(crowd.walkers.front()->getTotalNum());
  std::vector<ParticleSet::ParticleGradient> gradients(walker_count);
  std::vector<ParticleSet::ParticleLaplacian> laplacians(walker_count);
  RefVector<ParticleSet::ParticleGradient> gradient_list;
  RefVector<ParticleSet::ParticleLaplacian> laplacian_list;
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    gradients[lane].resize(electron_count);
    laplacians[lane].resize(electron_count);
    gradients[lane] = Value(0);
    laplacians[lane] = Value(0);
    gradient_list.push_back(gradients[lane]);
    laplacian_list.push_back(laplacians[lane]);
  }

  // Establish accepted value/spatial state and warm the active executor before
  // opening the allocation window at a live count below reserve capacity.
  crowd.leader.mw_evaluateLog(crowd.wfc_list, *crowd.p_list,
                              gradient_list, laplacian_list);
  std::vector<PsiFormerWF::GradType> active_gradients(walker_count);
  crowd.leader.mw_evalGrad(crowd.wfc_list, *crowd.p_list, 1,
                           active_gradients);
  std::vector<PsiFormerWF::GradType> expected_active_gradients(walker_count);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    expected_active_gradients[lane] =
        reference.evalGrad(*crowd.walkers[lane], 1);

  const PlannedAllocationFreeze frozen =
      capturePlannedAllocationFreeze(crowd, resource);
  CHECK(frozen.resource.initial_walker_capacity == walker_count);
  CHECK(frozen.resource.reserve_walker_capacity == reserve_walker_count);
  CHECK(frozen.resource.prepared_storage_fingerprint ==
        frozen.resource.current_storage_fingerprint);
  CHECK(frozen.collection_cursor == 1);
  CHECK(frozen.outstanding_loans == 1);
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);

  PsiFormerWF::GradType* const output_data = active_gradients.data();
  const std::size_t output_capacity = active_gradients.capacity();
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    for (int dimension = 0; dimension < 3; ++dimension)
      active_gradients[lane][dimension] =
          Value(-1900.0 - 10.0 * static_cast<double>(lane) - dimension);
  const AllocationSnapshot allocations = auditAllocations([&] {
    crowd.leader.mw_evalGrad(crowd.wfc_list, *crowd.p_list, 1,
                             active_gradients);
  });
  checkNoAllocations(allocations, "planned ACTIVE_GRADIENT b<B");
  CHECK(active_gradients.data() == output_data);
  CHECK(active_gradients.size() == walker_count);
  CHECK(active_gradients.capacity() == output_capacity);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      const Value sentinel(
          -1900.0 - 10.0 * static_cast<double>(lane) - dimension);
      CHECK(active_gradients[lane][dimension] != sentinel);
      checkObservation(observe(active_gradients[lane][dimension]),
                       observe(expected_active_gradients[lane][dimension]),
                       2.0e-8);
    }
  CHECK(Probe::plannedSelectedTransactionCount(crowd.leader) == 0);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    CHECK(Probe::hasCurrentFullAcceptedState(
        *crowd.components[lane], *crowd.walkers[lane]));
    CHECK_FALSE(Probe::hasProposal(*crowd.components[lane]));
  }
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);

  // Exercise VALUE refresh with live population below reserve capacity.  The
  // selected accepted value is deliberately stale and compared with a scalar
  // direct reference, making a no-op implementation observable.
  const PsiFormerWF::LogValue old_log =
      crowd.components[0]->get_log_value();
  crowd.walkers[0]->R[0][1] += 0.0035;
  crowd.walkers[0]->update();
  ParticleSet::ParticleGradient reference_gradient;
  ParticleSet::ParticleLaplacian reference_laplacian;
  reference_gradient.resize(electron_count);
  reference_laplacian.resize(electron_count);
  reference_gradient = Value(0);
  reference_laplacian = Value(0);
  const PsiFormerWF::LogValue expected_log = reference.evaluateLog(
      *crowd.walkers[0], reference_gradient, reference_laplacian);
  REQUIRE(std::abs(expected_log - old_log) > 1.0e-10);
  const std::vector<bool> recompute_sparse{true, false};
  const AllocationSnapshot recompute_allocations = auditAllocations([&] {
    crowd.leader.mw_recompute(crowd.wfc_list, *crowd.p_list,
                              recompute_sparse);
  });
  checkNoAllocations(recompute_allocations,
                     "planned RECOMPUTE_VALUE b<B");
  CHECK(Probe::hasCurrentAcceptedValue(
      *crowd.components[0], *crowd.walkers[0]));
  CHECK_FALSE(Probe::hasCurrentFullAcceptedState(
      *crowd.components[0], *crowd.walkers[0]));
  CHECK(Probe::hasCurrentFullAcceptedState(
      *crowd.components[1], *crowd.walkers[1]));
  checkObservation(observe(crowd.components[0]->get_log_value()),
                   observe(expected_log));
  CHECK(std::abs(crowd.components[0]->get_log_value() - old_log) >
        1.0e-10);
  CHECK(Probe::plannedSelectedTransactionCount(crowd.leader) == 0);
  for (const PsiFormerWF* component : crowd.components)
    CHECK_FALSE(Probe::hasProposal(*component));
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);
}

TEST_CASE("PsiFormer warmed planned scalar VALUE calls freeze clone storage",
          "[wavefunction][psiformer][allocation][batch_memory][scalar_value]")
{
  using Probe = testing::TestPsiFormerVirtualBatch;
  constexpr std::size_t clone_count = 2;

  GeneratedFiles files = generateFiles("lih");
  setBackend("direct");
  const SimulationCell simulation_cell;
  PlannedAllocationCrowd crowd(files, simulation_cell, clone_count);
  const std::size_t electron_count =
      static_cast<std::size_t>(crowd.walkers.front()->getTotalNum());
  REQUIRE(electron_count >= 3);

  // Establish nontrivial accepted state before plan binding.  Scalar VALUE
  // queries must preserve it bit-for-bit on every prepared clone.
  for (std::size_t lane = 0; lane < clone_count; ++lane)
  {
    crowd.walkers[lane]->G = Value(0);
    crowd.walkers[lane]->L = Value(0);
    crowd.components[lane]->evaluateLog(
        *crowd.walkers[lane], crowd.walkers[lane]->G,
        crowd.walkers[lane]->L);
  }

  const std::string participant_id =
      "test/psiformer/planned-scalar-value-allocation";
  const auto plan = makeAllocationPlan(
      crowd.leader, clone_count, participant_id, clone_count, true);
  bindAndPrepareAllocationCrowd(crowd, plan, participant_id);

  // Keep the selected crowd owner acquired so the existing reconciliation
  // helper can prove selected participant bytes equal independently measured
  // resource plus clone bytes.  Neither scalar call consumes this resource.
  ResourceCollection resource_template(
      "psiformer_planned_scalar_value_allocation_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection resource(resource_template);
  resource.prepareBatchResources({plan, 0});
  ResourceCollectionTeamLock<WaveFunctionComponent> resource_lock(
      resource, crowd.wfc_list);

  PsiFormerWF& component = *crowd.components.front();
  ParticleSet& particles = *crowd.walkers.front();
  particles.makeVirtualMoves(
      ParticleSet::SingleParticlePos{0.37, -0.22, 0.41});

  const Value sentinel(-1907.0);
  std::vector<Value> all_to_one_ratios(electron_count, sentinel);
  std::vector<std::unique_ptr<VirtualParticleSet>> virtual_particles;
  std::vector<std::vector<Value>> virtual_ratios;
  virtual_particles.reserve(electron_count + 1);
  virtual_ratios.reserve(electron_count + 1);
  for (std::size_t count = 0; count <= electron_count; ++count)
  {
    std::vector<ParticleSet::SingleParticlePos> displacements;
    displacements.reserve(count);
    for (std::size_t move = 0; move < count; ++move)
    {
      const double scale = static_cast<double>(move + 1);
      displacements.emplace_back(0.025 * scale, -0.017 * scale,
                                 0.013 * scale);
    }
    auto virtual_set = std::make_unique<VirtualParticleSet>(particles);
    virtual_set->makeMoves(particles, 1, displacements);
    virtual_particles.push_back(std::move(virtual_set));
    virtual_ratios.emplace_back(count, sentinel);
  }
  const std::size_t interior_count = electron_count / 2;
  REQUIRE(interior_count > 1);
  REQUIRE(interior_count < electron_count);

  // Warm the all-to-one envelope, every virtual logical prefix (including the
  // constructible zero prefix), and alternating dense/sparse mode reuse before
  // enabling allocation interposition.
  component.evaluateRatiosAlltoOne(particles, all_to_one_ratios);
  for (std::size_t count = 0; count <= electron_count; ++count)
    component.evaluateRatios(*virtual_particles[count],
                             virtual_ratios[count]);
  component.evaluateRatiosAlltoOne(particles, all_to_one_ratios);
  component.evaluateRatios(*virtual_particles[1], virtual_ratios[1]);
  component.evaluateRatiosAlltoOne(particles, all_to_one_ratios);
  component.evaluateRatios(*virtual_particles[interior_count],
                           virtual_ratios[interior_count]);
  component.evaluateRatios(*virtual_particles[electron_count],
                           virtual_ratios[electron_count]);

  const PlannedAllocationFreeze frozen =
      capturePlannedAllocationFreeze(crowd, resource);
  const auto frozen_states = captureScalarAllocationStates(crowd);
  REQUIRE(frozen_states.size() == clone_count);
  for (const auto& state : frozen_states)
  {
    REQUIRE(state.accepted_value_valid);
    REQUIRE_FALSE(state.has_proposal);
  }
  const std::size_t parameter_version = component.parameterVersion();
  const std::size_t selected_transactions =
      Probe::plannedSelectedTransactionCount(component);
  const std::size_t single_transactions =
      Probe::plannedSingleTransactionCount(component);
  REQUIRE(selected_transactions == 0);
  REQUIRE(single_transactions == 0);
  REQUIRE(frozen.clone_workspaces.size() == clone_count);
  for (const auto& diagnostics : frozen.clone_workspaces)
  {
    REQUIRE(diagnostics.owns_batch_workspace);
    REQUIRE(diagnostics.has_prepared_clone_plan);
    REQUIRE(diagnostics.prepared_scalar_value_compatibility);
    REQUIRE(diagnostics.batch_workspace_identity != nullptr);
    REQUIRE(diagnostics.batch_workspace_identity ==
            diagnostics.prepared_batch_workspace_identity);
    REQUIRE(diagnostics.batch_storage_fingerprint != 0);
    REQUIRE(diagnostics.batch_storage_fingerprint ==
            diagnostics.prepared_batch_storage_fingerprint);
    REQUIRE(diagnostics.batch_bytes != 0);
    REQUIRE(diagnostics.batch_bytes == diagnostics.prepared_batch_bytes);
    REQUIRE(diagnostics.scalar_value_publication_identity != nullptr);
    REQUIRE(diagnostics.scalar_value_publication_identity ==
            diagnostics.prepared_scalar_value_publication_identity);
    REQUIRE(diagnostics.scalar_value_publication_size == electron_count + 1);
    REQUIRE(diagnostics.scalar_value_publication_capacity ==
            electron_count + 1);
    REQUIRE(diagnostics.scalar_value_publication_size ==
            diagnostics.prepared_scalar_value_publication_size);
    REQUIRE(diagnostics.scalar_value_publication_capacity ==
            diagnostics.prepared_scalar_value_publication_capacity);
    REQUIRE(diagnostics.scalar_value_publication_bytes ==
            (electron_count + 1) * sizeof(Value));
    REQUIRE(diagnostics.accountedBytes() ==
            diagnostics.batch_bytes +
                diagnostics.scalar_value_publication_bytes);
  }
  REQUIRE(frozen.clone_workspaces[0].batch_workspace_identity !=
          frozen.clone_workspaces[1].batch_workspace_identity);
  REQUIRE(frozen.clone_workspaces[0].scalar_value_publication_identity !=
          frozen.clone_workspaces[1].scalar_value_publication_identity);
  checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);

  const auto check_frozen_state = [&] {
    checkPlannedAllocationFreeze(crowd, resource, *plan, frozen);
    checkScalarAllocationStatesUnchanged(crowd, frozen_states);
    CHECK(component.parameterVersion() == parameter_version);
    CHECK(Probe::plannedSelectedTransactionCount(component) ==
          selected_transactions);
    CHECK(Probe::plannedSingleTransactionCount(component) ==
          single_transactions);
    for (const PsiFormerWF* clone : crowd.components)
      CHECK_FALSE(Probe::hasProposal(*clone));
  };

  const auto check_ratios = [&](const std::vector<Value>& ratios,
                                const Value* expected_data,
                                std::size_t expected_capacity) {
    CHECK(ratios.data() == expected_data);
    CHECK(ratios.capacity() == expected_capacity);
    bool has_nonunit_ratio = false;
    for (const Value ratio : ratios)
    {
      CHECK(ratio != sentinel);
      CHECK(std::isfinite(std::real(ratio)));
      CHECK(std::isfinite(std::imag(ratio)));
      has_nonunit_ratio = has_nonunit_ratio ||
          std::abs(ratio - Value(1)) > 1.0e-10;
    }
    CHECK((ratios.empty() || has_nonunit_ratio));
  };

  const auto audit_scalar_call = [&](std::vector<Value>& ratios,
                                     const char* scope,
                                     auto&& operation) {
    std::fill(ratios.begin(), ratios.end(), sentinel);
    Value* const output_data = ratios.data();
    const std::size_t output_size = ratios.size();
    const std::size_t output_capacity = ratios.capacity();
    const AllocationSnapshot allocations = auditAllocations(
        std::forward<decltype(operation)>(operation));
    checkNoAllocations(allocations, scope);
    CHECK(ratios.size() == output_size);
    check_ratios(ratios, output_data, output_capacity);
    check_frozen_state();
  };

  audit_scalar_call(all_to_one_ratios, "planned scalar all-to-one Ne+1", [&] {
    component.evaluateRatiosAlltoOne(particles, all_to_one_ratios);
  });
  audit_scalar_call(virtual_ratios[0], "planned scalar virtual q=0", [&] {
    component.evaluateRatios(*virtual_particles[0], virtual_ratios[0]);
  });
  audit_scalar_call(virtual_ratios[1], "planned scalar virtual q=1", [&] {
    component.evaluateRatios(*virtual_particles[1], virtual_ratios[1]);
  });
  audit_scalar_call(
      virtual_ratios[interior_count], "planned scalar virtual interior prefix",
      [&] {
        component.evaluateRatios(*virtual_particles[interior_count],
                                 virtual_ratios[interior_count]);
      });
  audit_scalar_call(
      virtual_ratios[electron_count], "planned scalar virtual q=Ne", [&] {
        component.evaluateRatios(*virtual_particles[electron_count],
                                 virtual_ratios[electron_count]);
      });

  // Repeatedly alternate the two public entry points and four virtual
  // prefixes inside one window.  This catches allocation growth hidden by an
  // individually warmed mode while retaining every caller allocation.
  constexpr std::size_t alternating_output_count = 5;
  const std::array<std::vector<Value>*, alternating_output_count>
      alternating_outputs{
      &all_to_one_ratios, &virtual_ratios[0], &virtual_ratios[1],
      &virtual_ratios[interior_count], &virtual_ratios[electron_count]};
  std::array<Value*, alternating_output_count> alternating_data{};
  std::array<std::size_t, alternating_output_count> alternating_sizes{};
  std::array<std::size_t, alternating_output_count> alternating_capacities{};
  for (std::size_t output = 0; output < alternating_outputs.size(); ++output)
  {
    std::fill(alternating_outputs[output]->begin(),
              alternating_outputs[output]->end(), sentinel);
    alternating_data[output] = alternating_outputs[output]->data();
    alternating_sizes[output] = alternating_outputs[output]->size();
    alternating_capacities[output] = alternating_outputs[output]->capacity();
  }
  const AllocationSnapshot alternating_allocations = auditAllocations([&] {
    for (int repetition = 0; repetition < 3; ++repetition)
    {
      component.evaluateRatiosAlltoOne(particles, all_to_one_ratios);
      component.evaluateRatios(*virtual_particles[0], virtual_ratios[0]);
      component.evaluateRatiosAlltoOne(particles, all_to_one_ratios);
      component.evaluateRatios(*virtual_particles[1], virtual_ratios[1]);
      component.evaluateRatiosAlltoOne(particles, all_to_one_ratios);
      component.evaluateRatios(*virtual_particles[interior_count],
                               virtual_ratios[interior_count]);
      component.evaluateRatiosAlltoOne(particles, all_to_one_ratios);
      component.evaluateRatios(*virtual_particles[electron_count],
                               virtual_ratios[electron_count]);
    }
  });
  checkNoAllocations(alternating_allocations,
                     "planned scalar alternating all-to-one/virtual");
  for (std::size_t output = 0; output < alternating_outputs.size(); ++output)
  {
    CHECK(alternating_outputs[output]->size() == alternating_sizes[output]);
    check_ratios(*alternating_outputs[output], alternating_data[output],
                 alternating_capacities[output]);
  }
  check_frozen_state();
}

TEST_CASE("PsiFormer warmed planned one-electron transactions allocate no storage",
          "[wavefunction][psiformer][allocation][batch_memory]")
{
  GeneratedFiles files = generateFiles("lih");
  setBackend("direct");

  SECTION("singleton live crowd")
  {
    checkPlannedSingleTransactionAllocations(files, 1, 1);
  }

  SECTION("live crowd fills its prepared reserve")
  {
    checkPlannedSingleTransactionAllocations(files, 2, 2);
  }

  SECTION("live crowd is a strict prepared-reserve prefix")
  {
    checkPlannedSingleTransactionAllocations(files, 2, 3);
  }
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
