//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_multiwalker.cpp
 * @brief Deterministic public-API tests for PsiFormer crowd and virtual batches.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Particle/MCMultiParticleMoves.h"
#include "Particle/ParticleSet.h"
#include "Particle/VirtualParticleBatch.h"
#include "Particle/VirtualParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerMemoryPolicy.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "ResourceCollection.h"
#include "Utilities/BatchResourcePreparation.h"
#include "Utilities/RuntimeOptions.h"
#include "psiformer_test_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus
{
namespace testing
{
/** Exact clone-local state snapshot used to prove virtual evaluations are read-only. */
struct PsiFormerCloneStateSnapshot
{
  PsiFormerWF::LogValue log_value;
  std::size_t observed_parameter_version;
  bool restore_validation_pending;
  bool accepted_value_valid;
  ParticleSet::ParticleGradient accepted_gradient;
  ParticleSet::ParticleLaplacian accepted_laplacian;
  std::uint64_t accepted_configuration_identity;
  std::size_t accepted_parameter_version;
  std::uint64_t accepted_state_requirement;
  double current_sign;
  double proposed_sign;
  PsiFormerWF::LogValue proposed_log_value;
  ParticleSet::ParticleGradient proposed_gradient;
  ParticleSet::ParticleLaplacian proposed_laplacian;
  std::uint64_t proposed_configuration_identity;
  std::uint64_t proposed_descriptor_fingerprint;
  std::size_t proposed_parameter_version;
  int proposed_particle;
  std::uint64_t proposal_kind;
  bool has_proposal;
};

struct PsiFormerPreparedCloneStorage
{
  std::array<const void*, 4> data{};
  std::array<std::size_t, 4> sizes{};
  std::array<std::size_t, 4> capacities{};
  bool exact_marker = false;
};

/** Narrow friend accessor for state-isolation and crowd-workspace diagnostics. */
class TestPsiFormerVirtualBatch
{
public:
  /// Public test spelling of the private typed runtime-operation discriminator.
  enum class RuntimeOperation
  {
    FULL_VGL,
    RECOMPUTE_VALUE,
    CALC_RATIO,
    ACTIVE_GRADIENT,
    RATIO_GRADIENT,
    ACCEPT_REJECT_VALUE,
    SELECTED_PROPOSE,
    SELECTED_RESOLVE,
    ECP_VALUE,
    ECP_WEIGHTED_SCORE,
    SCORE_DERIVATIVES,
    KINETIC_DERIVATIVES,
    SCALAR_VALUE_COMPATIBILITY,
    BUFFER_READ,
    BUFFER_WRITE,
    PREPARE_GROUP,
    COMPLETE_UPDATES
  };

  /// Allocation-free dimensions supplied to the friend-only preflight probe.
  struct RuntimeRequest
  {
    RuntimeOperation operation = RuntimeOperation::FULL_VGL;
    std::size_t live_walkers = 0;
    std::size_t dense_configurations = 0;
    std::size_t sparse_references = 0;
    std::size_t sparse_replacements = 0;
    std::size_t selected_parameters = 0;
    std::size_t derivative_width = 0;
    std::optional<std::size_t> active_electron;
    std::optional<std::uint64_t> descriptor_fingerprint;
  };

  static PsiFormerCloneStateSnapshot cloneState(const PsiFormerWF& component)
  {
    return {component.log_value_,
            component.observed_parameter_version_,
            component.restore_validation_pending_,
            component.accepted_value_valid_,
            component.accepted_gradient_,
            component.accepted_laplacian_,
            component.accepted_configuration_identity_,
            component.accepted_parameter_version_,
            static_cast<std::uint64_t>(component.accepted_state_requirement_),
            component.current_sign_,
            component.proposed_sign_,
            component.proposed_log_value_,
            component.proposed_gradient_,
            component.proposed_laplacian_,
            component.proposed_configuration_identity_,
            component.proposed_descriptor_fingerprint_,
            component.proposed_parameter_version_,
            component.proposed_particle_,
            static_cast<std::uint64_t>(component.proposal_kind_),
            component.has_proposal_};
  }

  static PsiFormerPreparedCloneStorage preparedCloneStorage(
      const PsiFormerWF& component)
  {
    PsiFormerPreparedCloneStorage storage;
    storage.data = {component.accepted_gradient_.data(),
                    component.accepted_laplacian_.data(),
                    component.proposed_gradient_.data(),
                    component.proposed_laplacian_.data()};
    storage.sizes = {component.accepted_gradient_.size(),
                     component.accepted_laplacian_.size(),
                     component.proposed_gradient_.size(),
                     component.proposed_laplacian_.size()};
    storage.capacities = {component.accepted_gradient_.capacity(),
                          component.accepted_laplacian_.capacity(),
                          component.proposed_gradient_.capacity(),
                          component.proposed_laplacian_.capacity()};
    storage.exact_marker =
        component.hasPreparedBatchExecutionClone(component.batch_execution_plan_);
    return storage;
  }

  static void invalidateAcceptedState(PsiFormerWF& component)
  {
    component.accepted_value_valid_ = false;
    component.accepted_configuration_identity_ = 0;
    component.accepted_parameter_version_ = 0;
    component.accepted_state_requirement_ =
        PsiFormerWF::AcceptedStateRequirement::INVALID;
  }

  static void injectPlannedFullVGLPrepublicationFailure(
      PsiFormerWF& component, bool enabled)
  {
    component.fail_planned_full_vgl_before_publish_for_testing_ = enabled;
  }

  static bool hasCurrentFullAcceptedState(const PsiFormerWF& component,
                                          const ParticleSet& particles)
  {
    const std::size_t current_version = component.parameterVersion();
    return component.observed_parameter_version_ == current_version &&
        component.acceptedStateMatches(
            particles, current_version,
            PsiFormerWF::AcceptedStateRequirement::FULL_SPATIAL);
  }

  static bool cloneStateMatches(const PsiFormerWF& component,
                                const PsiFormerCloneStateSnapshot& snapshot)
  {
    if (component.log_value_ != snapshot.log_value ||
        component.observed_parameter_version_ != snapshot.observed_parameter_version ||
        component.restore_validation_pending_ != snapshot.restore_validation_pending ||
        component.accepted_value_valid_ != snapshot.accepted_value_valid ||
        component.accepted_configuration_identity_ != snapshot.accepted_configuration_identity ||
        component.accepted_parameter_version_ != snapshot.accepted_parameter_version ||
        static_cast<std::uint64_t>(component.accepted_state_requirement_) !=
            snapshot.accepted_state_requirement ||
        component.current_sign_ != snapshot.current_sign ||
        component.proposed_sign_ != snapshot.proposed_sign ||
        component.proposed_log_value_ != snapshot.proposed_log_value ||
        component.proposed_configuration_identity_ != snapshot.proposed_configuration_identity ||
        component.proposed_descriptor_fingerprint_ != snapshot.proposed_descriptor_fingerprint ||
        component.proposed_parameter_version_ != snapshot.proposed_parameter_version ||
        component.proposed_particle_ != snapshot.proposed_particle ||
        static_cast<std::uint64_t>(component.proposal_kind_) != snapshot.proposal_kind ||
        component.has_proposal_ != snapshot.has_proposal)
      return false;

    return sameGradient(component.accepted_gradient_, snapshot.accepted_gradient) &&
        sameLaplacian(component.accepted_laplacian_, snapshot.accepted_laplacian) &&
        sameGradient(component.proposed_gradient_, snapshot.proposed_gradient) &&
        sameLaplacian(component.proposed_laplacian_, snapshot.proposed_laplacian);
  }

  static PsiFormerCrowdWorkspaceDiagnostics crowdWorkspaceDiagnostics(
      const PsiFormerWF& component,
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list)
  {
    return component.crowdWorkspaceDiagnosticsForTesting(wfc_list);
  }

  /// Bind a directly constructed test component to its actual ParticleSet lane.
  static void bindParticleSet(PsiFormerWF& component, const ParticleSet& particles)
  { component.bound_particle_set_ = &particles; }

  /// Rebind metadata only long enough to exercise shared-identity validation.
  static void useOptimizationMetadata(PsiFormerWF& component,
                                      const PsiFormerWF& donor)
  { component.optimization_metadata_ = donor.optimization_metadata_; }

  /// Invoke only the read-only common planned-runtime boundary.
  static std::size_t requirePlannedRuntime(
      const PsiFormerWF& component,
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const RuntimeRequest& request)
  {
    PsiFormerWF::PlannedRuntimeRequest private_request;
    private_request.operation = operation(request.operation);
    private_request.live_walkers = request.live_walkers;
    private_request.dense_configurations = request.dense_configurations;
    private_request.sparse_references = request.sparse_references;
    private_request.sparse_replacements = request.sparse_replacements;
    private_request.selected_parameters = request.selected_parameters;
    private_request.derivative_width = request.derivative_width;
    private_request.active_electron = request.active_electron;
    private_request.descriptor_fingerprint = request.descriptor_fingerprint;
    return component
        .requirePlannedMultiWalkerOperation(wfc_list, p_list, private_request)
        .storage_fingerprint;
  }

  /// Install and clear one coherent pending proposal to test absence guards.
  static void installSingleProposal(PsiFormerWF& component, int electron)
  {
    component.has_proposal_ = true;
    component.proposal_kind_ = PsiFormerWF::ProposalKind::SINGLE_PARTICLE;
    component.proposed_particle_ = electron;
    component.proposed_parameter_version_ = component.observed_parameter_version_;
  }

  static void clearProposal(PsiFormerWF& component)
  { component.clearProposalState(); }

  /// Toggle complete Stage-5 ownership claims without exposing a production API.
  static void useCompleteBatchMemoryAccounting(PsiFormerWF& component,
                                               bool enabled)
  {
    component.complete_batch_memory_accounting_for_testing_ = enabled;
  }

  /// Count clone-local score tapes; flattened crowd scoring must own none.
  static std::size_t cloneScoreWorkspaceCount(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list)
  {
    std::size_t count = 0;
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      const auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      if (component.direct_score_workspace_)
        ++count;
    }
    return count;
  }

private:
  static PsiFormerWF::PlannedOperation operation(RuntimeOperation operation)
  {
    switch (operation)
    {
    case RuntimeOperation::FULL_VGL:
      return PsiFormerWF::PlannedOperation::FULL_VGL;
    case RuntimeOperation::RECOMPUTE_VALUE:
      return PsiFormerWF::PlannedOperation::RECOMPUTE_VALUE;
    case RuntimeOperation::CALC_RATIO:
      return PsiFormerWF::PlannedOperation::CALC_RATIO;
    case RuntimeOperation::ACTIVE_GRADIENT:
      return PsiFormerWF::PlannedOperation::ACTIVE_GRADIENT;
    case RuntimeOperation::RATIO_GRADIENT:
      return PsiFormerWF::PlannedOperation::RATIO_GRADIENT;
    case RuntimeOperation::ACCEPT_REJECT_VALUE:
      return PsiFormerWF::PlannedOperation::ACCEPT_REJECT_VALUE;
    case RuntimeOperation::SELECTED_PROPOSE:
      return PsiFormerWF::PlannedOperation::SELECTED_PROPOSE;
    case RuntimeOperation::SELECTED_RESOLVE:
      return PsiFormerWF::PlannedOperation::SELECTED_RESOLVE;
    case RuntimeOperation::ECP_VALUE:
      return PsiFormerWF::PlannedOperation::ECP_VALUE;
    case RuntimeOperation::ECP_WEIGHTED_SCORE:
      return PsiFormerWF::PlannedOperation::ECP_WEIGHTED_SCORE;
    case RuntimeOperation::SCORE_DERIVATIVES:
      return PsiFormerWF::PlannedOperation::SCORE_DERIVATIVES;
    case RuntimeOperation::KINETIC_DERIVATIVES:
      return PsiFormerWF::PlannedOperation::KINETIC_DERIVATIVES;
    case RuntimeOperation::SCALAR_VALUE_COMPATIBILITY:
      return PsiFormerWF::PlannedOperation::SCALAR_VALUE_COMPATIBILITY;
    case RuntimeOperation::BUFFER_READ:
      return PsiFormerWF::PlannedOperation::BUFFER_READ;
    case RuntimeOperation::BUFFER_WRITE:
      return PsiFormerWF::PlannedOperation::BUFFER_WRITE;
    case RuntimeOperation::PREPARE_GROUP:
      return PsiFormerWF::PlannedOperation::PREPARE_GROUP;
    case RuntimeOperation::COMPLETE_UPDATES:
      return PsiFormerWF::PlannedOperation::COMPLETE_UPDATES;
    }
    throw std::logic_error("Unknown PsiFormer runtime-operation test value");
  }

  static bool sameGradient(const ParticleSet::ParticleGradient& actual,
                           const ParticleSet::ParticleGradient& expected)
  {
    if (actual.size() != expected.size())
      return false;
    for (std::size_t electron = 0; electron < actual.size(); ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        if (actual[electron][dimension] != expected[electron][dimension])
          return false;
    return true;
  }

  static bool sameLaplacian(const ParticleSet::ParticleLaplacian& actual,
                            const ParticleSet::ParticleLaplacian& expected)
  {
    if (actual.size() != expected.size())
      return false;
    for (std::size_t electron = 0; electron < actual.size(); ++electron)
      if (actual[electron] != expected[electron])
        return false;
    return true;
  }
};
} // namespace testing

namespace
{
using namespace testing::psiformer;
using Value = QMCTraits::ValueType;

class ScopedEnvironmentVariable
{
public:
  ScopedEnvironmentVariable(std::string name, const char* value) : name_(std::move(name))
  {
    if (const char* previous = std::getenv(name_.c_str()))
      previous_ = previous;
    if (setenv(name_.c_str(), value, 1) != 0)
      throw std::runtime_error("Unable to set PsiFormer test environment variable");
  }

  ~ScopedEnvironmentVariable()
  {
    if (previous_)
      setenv(name_.c_str(), previous_->c_str(), 1);
    else
      unsetenv(name_.c_str());
  }

private:
  std::string name_;
  std::optional<std::string> previous_;
};

std::unique_ptr<ParticleSet> makeWalker(const SimulationCell& simulation_cell, std::size_t walker)
{
  const Geometry geometry = makeGeometry("lih");
  auto particles = std::make_unique<ParticleSet>(simulation_cell);
  particles->setName("e" + std::to_string(walker));
  particles->create({2, 2});
  for (int electron = 0; electron < particles->getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      particles->R[electron][dimension] = geometry.electrons[3 * electron + dimension] +
          0.007 * static_cast<double>(walker) * static_cast<double>((electron + 1) * (dimension + 1));
  particles->update();
  return particles;
}

void checkValue(Value actual, Value expected, double tolerance = 3.0e-9)
{
  CHECK(std::real(actual) == Catch::Approx(std::real(expected)).epsilon(tolerance).margin(tolerance));
  CHECK(std::imag(actual) == Catch::Approx(std::imag(expected)).epsilon(tolerance).margin(tolerance));
}

void checkLog(PsiFormerWF::LogValue actual,
              PsiFormerWF::LogValue expected,
              double tolerance = 3.0e-9)
{
  CHECK(std::real(actual) == Catch::Approx(std::real(expected)).epsilon(tolerance).margin(tolerance));
  CHECK(std::imag(actual) == Catch::Approx(std::imag(expected)).epsilon(tolerance).margin(tolerance));
}

void checkGrad(const PsiFormerWF::GradType& actual,
               const PsiFormerWF::GradType& expected,
               double tolerance = 3.0e-8)
{
  for (int dimension = 0; dimension < 3; ++dimension)
    checkValue(actual[dimension], expected[dimension], tolerance);
}

template<class VectorType>
bool sameVectorBits(const VectorType& actual, const VectorType& expected)
{
  return actual.size() == expected.size() &&
      (actual.size() == 0 ||
       std::memcmp(actual.data(), expected.data(),
                   actual.size() * sizeof(typename VectorType::value_type)) == 0);
}

struct Crowd
{
  Crowd(const GeneratedFiles& files,
        const SimulationCell& simulation_cell,
        std::size_t size,
        bool optimize = false,
        std::vector<std::size_t> selected_flat_indices = {})
      : leader("pf_mw", files.parameters.string(), files.configuration.string(),
               optimize, std::move(selected_flat_indices)),
        wfc_list(leader)
  {
    walkers.reserve(size);
    components.reserve(size);
    walkers.push_back(makeWalker(simulation_cell, 0));
    components.push_back(&leader);
    for (std::size_t walker = 1; walker < size; ++walker)
    {
      walkers.push_back(makeWalker(simulation_cell, walker));
      clone_storage.push_back(leader.makeClone(*walkers.back()));
      components.push_back(static_cast<PsiFormerWF*>(clone_storage.back().get()));
    }
    p_list = std::make_unique<RefVectorWithLeader<ParticleSet>>(*walkers.front());
    for (std::size_t walker = 0; walker < size; ++walker)
    {
      p_list->push_back(*walkers[walker]);
      wfc_list.push_back(*components[walker]);
      testing::TestPsiFormerVirtualBatch::bindParticleSet(
          *components[walker], *walkers[walker]);
    }
  }

  PsiFormerWF leader;
  std::vector<std::unique_ptr<ParticleSet>> walkers;
  std::vector<std::unique_ptr<WaveFunctionComponent>> clone_storage;
  std::vector<PsiFormerWF*> components;
  RefVectorWithLeader<WaveFunctionComponent> wfc_list;
  std::unique_ptr<RefVectorWithLeader<ParticleSet>> p_list;
};

/// Build a selected plan from the component's current, internally consistent evidence.
std::shared_ptr<const BatchExecutionPlan> makeCrowdPreparationTestPlan(
    PsiFormerWF& component,
    const BatchExecutionRequirements& requirements,
    std::vector<std::size_t> initial_walkers,
    std::vector<std::size_t> reserve_walkers,
    const std::string& participant_id,
    const std::string& profile_id,
    BatchTileCapacities preferred = {2, 1, 1, 2},
    std::size_t active_parameter_count = 2)
{
  BatchExecutionSelectionInput selection;
  selection.requirements                         = requirements;
  selection.topology.initial_walkers_per_crowd   = std::move(initial_walkers);
  selection.topology.reserve_walkers_per_crowd   = std::move(reserve_walkers);
  selection.topology.run_kind                    = "psiformer-crowd-preparation-test";
  selection.particle_count                       = 4;
  selection.active_parameter_count               = active_parameter_count;
  selection.target_coordinate                    = BatchExecutionTargetCoordinate::POS_ONLY;
  selection.preference.id                        = profile_id;
  selection.preference.preferred                 = preferred;
  selection.logical_maximum = component.batchExecutionLogicalMaximum(
      {requirements, selection.topology, selection.particle_count,
       active_parameter_count, selection.parameter_derivative_width,
       selection.target_coordinate});

  // ECP_OUTER is an aggregate-driver capacity. PsiFormer contributes no
  // component-local maximum even though its flattened ECP path consumes it.
  if (requirements.requires(BatchExecutionMode::ECP_OUTER))
    selection.logical_maximum.ecp_outer = preferred.ecp_outer;

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

/// Return the broad operation set needed to exercise every Stage-5 resource owner.
BatchExecutionRequirements makeCrowdPreparationRequirements(
    const PsiFormerWF& component)
{
  BatchExecutionRequirements requirements;
  component.contributeBatchExecutionRequirements(requirements);
  requirements.require(BatchExecutionMode::VALUE);
  requirements.require(BatchExecutionMode::ACTIVE_GRADIENT);
  requirements.require(BatchExecutionMode::SCORE);
  requirements.require(BatchExecutionMode::KINETIC);
  requirements.require(BatchExecutionMode::ECP_OUTER);
  requirements.require(BatchExecutionMode::ECP_WEIGHTED_SCORE);
  requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  return requirements;
}

/// Enable the friend-only complete-accounting seam for one test clone family.
void enableCrowdPreparationTestAccounting(Crowd& crowd, bool enabled = true)
{
  for (PsiFormerWF* component : crowd.components)
    testing::TestPsiFormerVirtualBatch::useCompleteBatchMemoryAccounting(
        *component, enabled);
}

/// Validate and bind the same selected participant view to one clone family.
void bindCrowdPreparationPlan(
    Crowd& crowd,
    const std::shared_ptr<const BatchExecutionPlan>& plan,
    const std::string& participant_id)
{
  const BatchExecutionParticipantPlan participant_plan =
      makeBatchExecutionParticipantPlan(plan, participant_id);
  for (PsiFormerWF* component : crowd.components)
    component->validateBatchExecutionPlanBinding(participant_plan);
  for (PsiFormerWF* component : crowd.components)
    component->bindBatchExecutionPlan(participant_plan);
}

/// Materialize every clone-local byte associated with an already-bound plan.
void prepareCrowdPreparationClones(
    Crowd& crowd,
    const std::shared_ptr<const BatchExecutionPlan>& plan,
    const std::string& participant_id)
{
  const BatchExecutionParticipantPlan participant_plan =
      makeBatchExecutionParticipantPlan(plan, participant_id);
  for (PsiFormerWF* component : crowd.components)
    component->prepareBatchExecutionClone(participant_plan);
}

/// Snapshot every state category that a read-only runtime preflight must preserve.
struct RuntimePreflightSnapshot
{
  std::size_t parameter_version = 0;
  std::vector<testing::PsiFormerCloneStateSnapshot> clones;
  std::vector<ParticleSet::ParticlePos> positions;
  std::vector<ParticleSet::ParticleScalar> spins;
  std::vector<ParticleSet::ParticleIndex> group_ids;
  std::vector<std::vector<ParticleSet::PosType>> soa_positions;
  std::vector<ParticleSet::Index_t> active_particles;
  std::vector<ParticleSet::PosType> active_positions;
  std::vector<ParticleSet::RealType> active_spins;
  std::vector<ParticleSet::ParticleGradient> gradients;
  std::vector<ParticleSet::ParticleLaplacian> laplacians;
  testing::PsiFormerCrowdWorkspaceDiagnostics resource;
  std::size_t collection_cursor = 0;
  std::size_t outstanding_loans = 0;
  std::vector<Value> caller_output;
};

RuntimePreflightSnapshot captureRuntimePreflightState(
    Crowd& crowd,
    ResourceCollection& collection,
    const std::vector<Value>& caller_output)
{
  RuntimePreflightSnapshot snapshot;
  snapshot.parameter_version = crowd.leader.parameterVersion();
  snapshot.clones.reserve(crowd.components.size());
  snapshot.positions.reserve(crowd.walkers.size());
  snapshot.spins.reserve(crowd.walkers.size());
  snapshot.group_ids.reserve(crowd.walkers.size());
  snapshot.soa_positions.reserve(crowd.walkers.size());
  snapshot.active_particles.reserve(crowd.walkers.size());
  snapshot.active_positions.reserve(crowd.walkers.size());
  snapshot.active_spins.reserve(crowd.walkers.size());
  snapshot.gradients.reserve(crowd.walkers.size());
  snapshot.laplacians.reserve(crowd.walkers.size());
  for (std::size_t lane = 0; lane < crowd.components.size(); ++lane)
  {
    snapshot.clones.push_back(
        testing::TestPsiFormerVirtualBatch::cloneState(*crowd.components[lane]));
    snapshot.positions.push_back(crowd.walkers[lane]->R);
    snapshot.spins.push_back(crowd.walkers[lane]->spins);
    snapshot.group_ids.push_back(crowd.walkers[lane]->GroupID);
    std::vector<ParticleSet::PosType> soa;
    soa.reserve(crowd.walkers[lane]->getTotalNum());
    for (int electron = 0; electron < crowd.walkers[lane]->getTotalNum(); ++electron)
      soa.push_back(crowd.walkers[lane]->getCoordinates().getAllParticlePos()[electron]);
    snapshot.soa_positions.push_back(std::move(soa));
    snapshot.active_particles.push_back(crowd.walkers[lane]->getActivePtcl());
    snapshot.active_positions.push_back(crowd.walkers[lane]->getActivePos());
    snapshot.active_spins.push_back(crowd.walkers[lane]->getActiveSpinVal());
    snapshot.gradients.push_back(crowd.walkers[lane]->G);
    snapshot.laplacians.push_back(crowd.walkers[lane]->L);
  }
  snapshot.resource =
      testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
          crowd.leader, crowd.wfc_list);
  snapshot.collection_cursor = collection.getCursor();
  snapshot.outstanding_loans = collection.getOutstandingLoanCount();
  snapshot.caller_output = caller_output;
  return snapshot;
}

void checkRuntimePreflightState(
    Crowd& crowd,
    ResourceCollection& collection,
    const std::vector<Value>& caller_output,
    const RuntimePreflightSnapshot& expected)
{
  const auto same_real = [](ParticleSet::RealType actual,
                            ParticleSet::RealType reference) {
    return std::memcmp(&actual, &reference, sizeof(actual)) == 0;
  };
  CHECK(crowd.leader.parameterVersion() == expected.parameter_version);
  REQUIRE(crowd.components.size() == expected.clones.size());
  for (std::size_t lane = 0; lane < crowd.components.size(); ++lane)
  {
    CHECK(testing::TestPsiFormerVirtualBatch::cloneStateMatches(
        *crowd.components[lane], expected.clones[lane]));
    REQUIRE(crowd.walkers[lane]->R.size() == expected.positions[lane].size());
    REQUIRE(crowd.walkers[lane]->spins.size() == expected.spins[lane].size());
    REQUIRE(crowd.walkers[lane]->GroupID.size() == expected.group_ids[lane].size());
    REQUIRE(crowd.walkers[lane]->getCoordinates().getAllParticlePos().size() ==
            expected.soa_positions[lane].size());
    CHECK(crowd.walkers[lane]->getActivePtcl() == expected.active_particles[lane]);
    CHECK(same_real(crowd.walkers[lane]->getActiveSpinVal(),
                    expected.active_spins[lane]));
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      CHECK(same_real(crowd.walkers[lane]->getActivePos()[dimension],
                      expected.active_positions[lane][dimension]));
    REQUIRE(crowd.walkers[lane]->G.size() == expected.gradients[lane].size());
    REQUIRE(crowd.walkers[lane]->L.size() == expected.laplacians[lane].size());
    for (std::size_t electron = 0; electron < expected.gradients[lane].size(); ++electron)
    {
      CHECK(crowd.walkers[lane]->spins[electron] == expected.spins[lane][electron]);
      CHECK(crowd.walkers[lane]->GroupID[electron] == expected.group_ids[lane][electron]);
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        CHECK(same_real(crowd.walkers[lane]->R[electron][dimension],
                        expected.positions[lane][electron][dimension]));
        CHECK(same_real(
            crowd.walkers[lane]->getCoordinates().getAllParticlePos()[electron][dimension],
            expected.soa_positions[lane][electron][dimension]));
        CHECK(crowd.walkers[lane]->G[electron][dimension] ==
              expected.gradients[lane][electron][dimension]);
      }
      CHECK(crowd.walkers[lane]->L[electron] ==
            expected.laplacians[lane][electron]);
    }
  }

  const auto actual_resource =
      testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
          crowd.leader, crowd.wfc_list);
  CHECK(actual_resource.resource_identity == expected.resource.resource_identity);
  CHECK(actual_resource.batch_workspace_identity ==
        expected.resource.batch_workspace_identity);
  CHECK(actual_resource.prepared_plan_identity ==
        expected.resource.prepared_plan_identity);
  CHECK(actual_resource.prepared_plan_fingerprint ==
        expected.resource.prepared_plan_fingerprint);
  CHECK(actual_resource.prepared_storage_fingerprint ==
        expected.resource.prepared_storage_fingerprint);
  CHECK(actual_resource.current_storage_fingerprint ==
        expected.resource.current_storage_fingerprint);
  CHECK(actual_resource.logical_sizes == expected.resource.logical_sizes);
  CHECK(actual_resource.actual_resource_storage ==
        expected.resource.actual_resource_storage);
  CHECK(collection.getCursor() == expected.collection_cursor);
  CHECK(collection.getOutstandingLoanCount() == expected.outstanding_loans);
  CHECK(caller_output == expected.caller_output);
}

/// Clear every component plan without changing shared model ownership.
void clearCrowdPreparationPlan(Crowd& crowd)
{
  const BatchExecutionParticipantPlan empty_plan;
  for (PsiFormerWF* component : crowd.components)
    component->validateBatchExecutionPlanBinding(empty_plan);
  for (PsiFormerWF* component : crowd.components)
    component->bindBatchExecutionPlan(empty_plan);
}

/// Compare exact prepared storage and its retained/setup accounting split.
void checkPreparedResourceStorage(
    const testing::PsiFormerCrowdWorkspaceDiagnostics& diagnostics)
{
  CHECK(diagnostics.has_expected_plan);
  CHECK(diagnostics.has_prepared_plan);
  CHECK(diagnostics.prepared_plan_identity != nullptr);
  CHECK(diagnostics.prepared_plan_fingerprint != 0);
  CHECK(diagnostics.prepared_storage_fingerprint != 0);
  CHECK(diagnostics.expected_resource_storage ==
        diagnostics.actual_resource_storage);

  const BatchMemoryBytes total =
      diagnostics.expected_resource_storage.total();
  const BatchMemoryBytes replacement =
      diagnostics.expected_resource_storage.at(
          BatchMemoryCategory::REALLOCATION_TRANSIENT);
  CHECK(total.device == 0);
  CHECK(replacement.device == 0);
  REQUIRE(total.host >= replacement.host);
  CHECK(diagnostics.accountedBytes() == total.host - replacement.host);
}

void checkPreparedCloneStorageUnchanged(
    const testing::PsiFormerPreparedCloneStorage& actual,
    const testing::PsiFormerPreparedCloneStorage& expected)
{
  CHECK(actual.data == expected.data);
  CHECK(actual.sizes == expected.sizes);
  CHECK(actual.capacities == expected.capacities);
  CHECK(actual.exact_marker == expected.exact_marker);
}

void checkPreparedResourceStorageUnchanged(
    const testing::PsiFormerCrowdWorkspaceDiagnostics& actual,
    const testing::PsiFormerCrowdWorkspaceDiagnostics& expected)
{
  CHECK(actual.resource_identity == expected.resource_identity);
  CHECK(actual.batch_workspace_identity == expected.batch_workspace_identity);
  CHECK(actual.prepared_plan_identity == expected.prepared_plan_identity);
  CHECK(actual.prepared_plan_fingerprint == expected.prepared_plan_fingerprint);
  CHECK(actual.prepared_storage_fingerprint ==
        expected.prepared_storage_fingerprint);
  CHECK(actual.current_storage_fingerprint ==
        expected.current_storage_fingerprint);
  CHECK(actual.logical_sizes == expected.logical_sizes);
  CHECK(actual.batch_bytes == expected.batch_bytes);
  CHECK(actual.score_bytes == expected.score_bytes);
  CHECK(actual.kinetic_bytes == expected.kinetic_bytes);
  CHECK(actual.transient_bytes == expected.transient_bytes);
  CHECK(actual.actual_resource_storage == expected.actual_resource_storage);
}

struct PlannedFullVGLSnapshot
{
  std::vector<testing::PsiFormerCloneStateSnapshot> clone_states;
  std::vector<testing::PsiFormerPreparedCloneStorage> clone_storage;
  std::vector<ParticleSet::ParticleGradient> gradients;
  std::vector<ParticleSet::ParticleLaplacian> laplacians;
  testing::PsiFormerCrowdWorkspaceDiagnostics resource;
};

PlannedFullVGLSnapshot capturePlannedFullVGLSnapshot(
    Crowd& crowd,
    const std::vector<ParticleSet::ParticleGradient>& gradients,
    const std::vector<ParticleSet::ParticleLaplacian>& laplacians)
{
  PlannedFullVGLSnapshot snapshot;
  snapshot.clone_states.reserve(crowd.components.size());
  snapshot.clone_storage.reserve(crowd.components.size());
  for (const PsiFormerWF* component : crowd.components)
  {
    snapshot.clone_states.push_back(
        testing::TestPsiFormerVirtualBatch::cloneState(*component));
    snapshot.clone_storage.push_back(
        testing::TestPsiFormerVirtualBatch::preparedCloneStorage(*component));
  }
  snapshot.gradients = gradients;
  snapshot.laplacians = laplacians;
  snapshot.resource =
      testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
          crowd.leader, crowd.wfc_list);
  return snapshot;
}

void checkPlannedFullVGLSnapshot(
    Crowd& crowd,
    const std::vector<ParticleSet::ParticleGradient>& gradients,
    const std::vector<ParticleSet::ParticleLaplacian>& laplacians,
    const PlannedFullVGLSnapshot& expected)
{
  REQUIRE(crowd.components.size() == expected.clone_states.size());
  REQUIRE(crowd.components.size() == expected.clone_storage.size());
  REQUIRE(gradients.size() == expected.gradients.size());
  REQUIRE(laplacians.size() == expected.laplacians.size());
  for (std::size_t lane = 0; lane < crowd.components.size(); ++lane)
  {
    CHECK(testing::TestPsiFormerVirtualBatch::cloneStateMatches(
        *crowd.components[lane], expected.clone_states[lane]));
    checkPreparedCloneStorageUnchanged(
        testing::TestPsiFormerVirtualBatch::preparedCloneStorage(
            *crowd.components[lane]),
        expected.clone_storage[lane]);
    CHECK(sameVectorBits(gradients[lane], expected.gradients[lane]));
    CHECK(sameVectorBits(laplacians[lane], expected.laplacians[lane]));
  }
  checkPreparedResourceStorageUnchanged(
      testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
          crowd.leader, crowd.wfc_list),
      expected.resource);
}

/// Own one compatibility VirtualParticleSet scratch object per reference walker.
struct VirtualScratchCrowd
{
  explicit VirtualScratchCrowd(const Crowd& crowd)
  {
    storage.reserve(crowd.walkers.size());
    for (const auto& walker : crowd.walkers)
      storage.push_back(std::make_unique<VirtualParticleSet>(*walker));
    list = std::make_unique<RefVectorWithLeader<VirtualParticleSet>>(*storage.front());
    for (const auto& scratch : storage)
      list->push_back(*scratch);
  }

  std::vector<std::unique_ptr<VirtualParticleSet>> storage;
  std::unique_ptr<RefVectorWithLeader<VirtualParticleSet>> list;
};

/// Append one off-sphere segment using deterministic displacements from its reference electron.
void appendVirtualSegment(
    const Crowd& crowd,
    std::size_t walker,
    int electron,
    std::initializer_list<ParticleSet::PosType> displacements,
    std::vector<std::size_t>& offsets,
    std::vector<VirtualParticleBatch::Segment>& segments,
    std::vector<ParticleSet::PosType>& positions)
{
  segments.emplace_back(static_cast<int>(walker), electron);
  for (const ParticleSet::PosType& displacement : displacements)
    positions.push_back(crowd.walkers[walker]->R[electron] + displacement);
  offsets.push_back(positions.size());
}

/// Evaluate the unchanged scalar virtual interface segment by segment as an independent oracle.
std::vector<Value> evaluateScalarVirtualBatch(Crowd& crowd,
                                              const VirtualParticleBatch& batch)
{
  std::vector<Value> ratios(batch.size());
  for (std::size_t segment_index = 0; segment_index < batch.segmentCount();
       ++segment_index)
  {
    const VirtualParticleBatch::Slice slice = batch.slice(segment_index);
    const std::size_t walker = static_cast<std::size_t>(slice.walkerId());
    VirtualParticleSet scratch(*crowd.walkers[walker]);
    scratch.makeMovesAbsolute(*crowd.walkers[walker], slice.electronId(),
                              slice.positions(), slice.isOnSphere(),
                              slice.sourceCenterId());
    std::vector<Value> segment_ratios(slice.size());
    crowd.components[walker]->evaluateRatios(scratch, segment_ratios);
    std::copy(segment_ratios.begin(), segment_ratios.end(),
              ratios.begin() + slice.flatOffset());
  }
  return ratios;
}

/// Build a global optimizer map whose PsiFormer destinations are sparse and reordered.
OptVariables configureSparseSelectedMapping(PsiFormerWF& component)
{
  OptVariables selected;
  component.checkInVariablesExclusive(selected);
  REQUIRE(selected.size() == 3);

  OptVariables active;
  active.insert("ordinary_padding_0", -1.0, true, optimize::LINEAR_P);
  active.insert(selected.name(2), selected[2]);
  active.insert("ordinary_padding_1", 2.0, true, optimize::LOGLINEAR_P);
  active.insert(selected.name(0), selected[0]);
  active.insert("ordinary_padding_2", 3.0, true, optimize::SPO_P);
  active.insert(selected.name(1), selected[1]);
  active.resetIndex();
  component.checkOutVariables(active);

  CHECK(active.getIndex(selected.name(0)) == 3);
  CHECK(active.getIndex(selected.name(1)) == 5);
  CHECK(active.getIndex(selected.name(2)) == 1);
  return active;
}

/// Map only the first selected PsiFormer variable behind one unrelated global slot.
OptVariables configureFirstSelectedOnly(PsiFormerWF& component,
                                        std::size_t expected_selected_count)
{
  OptVariables selected;
  component.checkInVariablesExclusive(selected);
  REQUIRE(selected.size() == expected_selected_count);

  OptVariables active;
  active.insert("ordinary_partial_padding", -2.0, true, optimize::LINEAR_P);
  active.insert(selected.name(0), selected[0]);
  active.resetIndex();
  component.checkOutVariables(active);
  CHECK(active.getIndex(selected.name(0)) == 1);
  return active;
}

/// Construct a real or genuinely complex quadrature coefficient for both builds.
Value makeWeight(double real_part, double imaginary_part = 0.0)
{
#ifdef QMC_COMPLEX
  return Value(real_part, imaginary_part);
#else
  static_cast<void>(imaginary_part);
  return Value(real_part);
#endif
}

/** Materialized scalar result used as the established compatibility oracle for the
 * compact flattened weighted implementation. */
struct MaterializedWeightedOracle
{
  std::vector<Value> ratios;
  std::vector<Value> total_weights;
  std::vector<std::vector<Value>> derivatives;
};

/// Contract scalar derivative-ratio matrices segment by segment as an oracle.
MaterializedWeightedOracle evaluateMaterializedWeightedOracle(
    Crowd& crowd,
    const VirtualParticleBatch& batch,
    const OptVariables& active,
    const std::vector<Value>& bare_weights,
    const std::vector<std::vector<Value>>& initial_derivatives)
{
  REQUIRE(bare_weights.size() == batch.size());
  REQUIRE(initial_derivatives.size() == crowd.walkers.size());

  MaterializedWeightedOracle oracle;
  oracle.ratios.resize(batch.size());
  oracle.total_weights.resize(batch.size());
  oracle.derivatives = initial_derivatives;
  for (std::size_t segment_index = 0; segment_index < batch.segmentCount();
       ++segment_index)
  {
    const VirtualParticleBatch::Slice slice = batch.slice(segment_index);
    const std::size_t walker = static_cast<std::size_t>(slice.walkerId());
    VirtualParticleSet scratch(*crowd.walkers[walker]);
    scratch.makeMovesAbsolute(*crowd.walkers[walker], slice.electronId(),
                              slice.positions(), slice.isOnSphere(),
                              slice.sourceCenterId());
    std::vector<Value> segment_ratios(slice.size());
    Matrix<Value> derivative_ratios(slice.size(), active.size());
    derivative_ratios = Value(0);
    crowd.components[walker]->evaluateDerivRatios(
        scratch, active, segment_ratios, derivative_ratios);

    for (std::size_t local_index = 0; local_index < slice.size(); ++local_index)
    {
      const std::size_t flat_index = slice.flatOffset() + local_index;
      oracle.ratios[flat_index] = segment_ratios[local_index];
      oracle.total_weights[flat_index] =
          bare_weights[flat_index] * segment_ratios[local_index];
      for (std::size_t parameter = 0; parameter < active.size(); ++parameter)
        oracle.derivatives[walker][parameter] +=
            oracle.total_weights[flat_index] *
            derivative_ratios(local_index, parameter);
    }
  }
  return oracle;
}

/// Create non-owning derivative rows for a vector-backed test destination.
std::vector<WaveFunctionComponent::ParameterDerivativeView> makeDerivativeViews(
    std::vector<std::vector<Value>>& derivatives)
{
  std::vector<WaveFunctionComponent::ParameterDerivativeView> views;
  views.reserve(derivatives.size());
  for (std::vector<Value>& row : derivatives)
    views.push_back({row.empty() ? nullptr : row.data(), row.size()});
  return views;
}

} // namespace

TEST_CASE("PsiFormer crowd APIs match scalar paths for batches 1 2 and 4",
          "[wavefunction][psiformer][multiwalker]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;

  for (const std::size_t batch_size : {std::size_t{1}, std::size_t{2}, std::size_t{4}})
  {
    DYNAMIC_SECTION("batch size " << batch_size)
    {
      Crowd crowd(files, simulation_cell, batch_size);
      constexpr int moved_electron = 1;
      const std::size_t electrons = crowd.walkers.front()->getTotalNum();

      std::vector<ParticleSet::ParticleGradient> scalar_g(batch_size);
      std::vector<ParticleSet::ParticleLaplacian> scalar_l(batch_size);
      std::vector<PsiFormerWF::LogValue> scalar_log(batch_size);
      std::vector<ParticleSet::ParticleGradient> batch_g(batch_size);
      std::vector<ParticleSet::ParticleLaplacian> batch_l(batch_size);
      RefVector<ParticleSet::ParticleGradient> batch_g_list;
      RefVector<ParticleSet::ParticleLaplacian> batch_l_list;
      for (std::size_t walker = 0; walker < batch_size; ++walker)
      {
        scalar_g[walker].resize(electrons);
        scalar_l[walker].resize(electrons);
        batch_g[walker].resize(electrons);
        batch_l[walker].resize(electrons);
        scalar_g[walker] = Value(0.125 * (walker + 1));
        scalar_l[walker] = Value(-0.25 * (walker + 1));
        batch_g[walker] = scalar_g[walker];
        batch_l[walker] = scalar_l[walker];
        scalar_log[walker] = crowd.components[walker]->evaluateLog(
            *crowd.walkers[walker], scalar_g[walker], scalar_l[walker]);
        batch_g_list.push_back(batch_g[walker]);
        batch_l_list.push_back(batch_l[walker]);
      }

      ResourceCollection resource_template("psiformer_resource_template");
      crowd.leader.createResource(resource_template);
      ResourceCollection crowd_resource(resource_template);
      {
        ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource, crowd.wfc_list);
        crowd.leader.mw_evaluateLog(crowd.wfc_list, *crowd.p_list, batch_g_list, batch_l_list);

        for (std::size_t walker = 0; walker < batch_size; ++walker)
        {
          checkLog(crowd.components[walker]->get_log_value(), scalar_log[walker]);
          for (std::size_t electron = 0; electron < electrons; ++electron)
          {
            checkGrad(batch_g[walker][electron], scalar_g[walker][electron]);
            checkValue(batch_l[walker][electron], scalar_l[walker][electron], 3.0e-7);
          }
        }

        std::vector<PsiFormerWF::GradType> scalar_active(batch_size);
        std::vector<PsiFormerWF::GradType> batch_active(batch_size);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
          scalar_active[walker] = crowd.components[walker]->evalGrad(
              *crowd.walkers[walker], moved_electron);
        crowd.leader.mw_evalGrad(
            crowd.wfc_list, *crowd.p_list, moved_electron, batch_active);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
          checkGrad(batch_active[walker], scalar_active[walker]);

        for (std::size_t walker = 0; walker < batch_size; ++walker)
        {
          const ParticleSet::SingleParticlePos displacement{
              0.012 * (walker + 1), -0.009 * (walker + 1), 0.006 * (walker + 1)};
          crowd.walkers[walker]->makeMove(moved_electron, displacement);
        }

        std::vector<Value> scalar_ratios(batch_size);
        std::vector<Value> batch_ratios(batch_size);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
        {
          scalar_ratios[walker] = crowd.components[walker]->ratio(
              *crowd.walkers[walker], moved_electron);
          crowd.components[walker]->restore(moved_electron);
        }
        crowd.leader.mw_calcRatio(
            crowd.wfc_list, *crowd.p_list, moved_electron, batch_ratios);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
          checkValue(batch_ratios[walker], scalar_ratios[walker]);

        std::vector<PsiFormerWF::GradType> scalar_ratio_grads(batch_size);
        std::vector<PsiFormerWF::GradType> batch_ratio_grads(batch_size);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
        {
          const PsiFormerWF::GradType seed(Value(0.31 + walker), Value(-0.17), Value(0.23));
          scalar_ratio_grads[walker] = seed;
          batch_ratio_grads[walker] = seed;
          scalar_ratios[walker] = crowd.components[walker]->ratioGrad(
              *crowd.walkers[walker], moved_electron, scalar_ratio_grads[walker]);
          crowd.components[walker]->restore(moved_electron);
        }
        crowd.leader.mw_ratioGrad(crowd.wfc_list, *crowd.p_list, moved_electron,
                                  batch_ratios, batch_ratio_grads);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
        {
          checkValue(batch_ratios[walker], scalar_ratios[walker]);
          checkGrad(batch_ratio_grads[walker], scalar_ratio_grads[walker]);
        }

        std::vector<PsiFormerWF::LogValue> old_logs(batch_size);
        std::vector<PsiFormerWF::LogValue> proposed_logs(batch_size);
        std::vector<bool> accepted(batch_size);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
        {
          old_logs[walker] = scalar_log[walker];
          const double current_sign = std::abs(std::imag(old_logs[walker])) > 1.0 ? -1.0 : 1.0;
          const double ratio_sign = std::real(batch_ratios[walker]) < 0.0 ? -1.0 : 1.0;
          proposed_logs[walker] = PsiFormerWF::LogValue(
              std::real(old_logs[walker]) + std::log(std::abs(std::real(batch_ratios[walker]))),
              current_sign * ratio_sign < 0.0 ? M_PI : 0.0);
          accepted[walker] = walker % 2 == 0;
        }
        crowd.leader.mw_accept_rejectMove(
            crowd.wfc_list, *crowd.p_list, moved_electron, accepted, true);
        for (std::size_t walker = 0; walker < batch_size; ++walker)
          checkLog(crowd.components[walker]->get_log_value(),
                   accepted[walker] ? proposed_logs[walker] : old_logs[walker]);
      }

      // The copied ResourceCollection owns independent scratch and release clears
      // the leader handle, so a direct crowd call outside a team lock fails early.
      std::vector<PsiFormerWF::GradType> gradients(batch_size);
      CHECK_THROWS_AS(crowd.leader.mw_evalGrad(
                          crowd.wfc_list, *crowd.p_list, moved_electron, gradients),
                      std::logic_error);
    }
  }
}

TEST_CASE("PsiFormer all-to-one and ragged virtual batches are state isolated",
          "[wavefunction][psiformer][multiwalker][ecp]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;

  // Scalar all-to-one has no framework multiwalker entry point, so it uses the
  // clone-local native batch workspace and must not populate proposal state.
  auto particles = makeWalker(simulation_cell, 0);
  PsiFormerWF all_to_one("pf_all_to_one", files.parameters.string(), files.configuration.string());
  ParticleSet::ParticleGradient gradient(particles->getTotalNum());
  ParticleSet::ParticleLaplacian laplacian(particles->getTotalNum());
  gradient = Value(0);
  laplacian = Value(0);
  const PsiFormerWF::LogValue reference_log =
      all_to_one.evaluateLog(*particles, gradient, laplacian);
  const ParticleSet::SingleParticlePos common_position{0.37, -0.22, 0.41};
  particles->makeVirtualMoves(common_position);
  std::vector<Value> all_ratios(particles->getTotalNum());
  all_to_one.evaluateRatiosAlltoOne(*particles, all_ratios);
  checkLog(all_to_one.get_log_value(), reference_log);
  all_to_one.acceptMove(*particles, 0);
  checkLog(all_to_one.get_log_value(), reference_log);

  PsiFormerWF all_to_one_oracle("pf_all_to_one_oracle", files.parameters.string(),
                                files.configuration.string());
  for (int electron = 0; electron < particles->getTotalNum(); ++electron)
  {
    auto moved = makeWalker(simulation_cell, 0);
    moved->R[electron] = common_position;
    moved->update();
    ParticleSet::ParticleGradient moved_g(moved->getTotalNum());
    ParticleSet::ParticleLaplacian moved_l(moved->getTotalNum());
    moved_g = Value(0);
    moved_l = Value(0);
    const auto moved_log = all_to_one_oracle.evaluateLog(*moved, moved_g, moved_l);
    checkValue(all_ratios[electron], Value(std::real(std::exp(moved_log - reference_log))));
  }

  Crowd crowd(files, simulation_cell, 4);
  std::vector<std::unique_ptr<VirtualParticleSet>> virtual_storage;
  std::vector<std::vector<ParticleSet::SingleParticlePos>> displacements(4);
  for (std::size_t walker = 0; walker < 4; ++walker)
  {
    for (std::size_t move = 0; move < walker + 1; ++move)
      displacements[walker].push_back(ParticleSet::SingleParticlePos{
          0.01 * (move + 1), -0.013 * (walker + 1), 0.008 * (move + walker + 1)});
    virtual_storage.push_back(std::make_unique<VirtualParticleSet>(*crowd.walkers[walker]));
    virtual_storage.back()->makeMoves(*crowd.walkers[walker], static_cast<int>(walker % 4),
                                      displacements[walker]);
  }

  RefVectorWithLeader<const VirtualParticleSet> virtual_list(*virtual_storage.front());
  std::vector<std::vector<Value>> expected(4);
  std::vector<std::vector<Value>> actual(4);
  std::vector<PsiFormerWF::LogValue> state_before(4);
  for (std::size_t walker = 0; walker < 4; ++walker)
  {
    virtual_list.push_back(*virtual_storage[walker]);
    expected[walker].resize(displacements[walker].size());
    actual[walker].resize(displacements[walker].size());
    ParticleSet::ParticleGradient g(crowd.walkers[walker]->getTotalNum());
    ParticleSet::ParticleLaplacian l(crowd.walkers[walker]->getTotalNum());
    g = Value(0);
    l = Value(0);
    state_before[walker] = crowd.components[walker]->evaluateLog(*crowd.walkers[walker], g, l);
    crowd.components[walker]->evaluateRatios(*virtual_storage[walker], expected[walker]);
  }

  ResourceCollection resource_template("psiformer_virtual_template");
  crowd.leader.createResource(resource_template);
  for (int resource_clone = 0; resource_clone < 2; ++resource_clone)
  {
    ResourceCollection crowd_resource(resource_template);
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource, crowd.wfc_list);
    crowd.leader.mw_evaluateRatios(crowd.wfc_list, virtual_list, actual);
    for (std::size_t walker = 0; walker < 4; ++walker)
    {
      REQUIRE(actual[walker].size() == expected[walker].size());
      for (std::size_t move = 0; move < actual[walker].size(); ++move)
        checkValue(actual[walker][move], expected[walker][move]);
      checkLog(crowd.components[walker]->get_log_value(), state_before[walker]);
    }
    RefVector<std::pair<WaveFunctionComponent::ValueVector, WaveFunctionComponent::ValueVector>>
        unused_spin_multipliers;
    CHECK_THROWS_AS(crowd.leader.mw_evaluateSpinorRatios(
                        crowd.wfc_list, virtual_list, unused_spin_multipliers, actual),
                    std::invalid_argument);
  }
}

TEST_CASE("PsiFormer flattened virtual batches share sparse references and preserve state",
          "[wavefunction][psiformer][multiwalker][ecp][sparse]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  Crowd crowd(files, simulation_cell, 4);

  // Deliberately order segments independently of walker order, name walker 2
  // twice with different electrons, and leave walker 3 empty.
  std::vector<std::size_t> offsets{0};
  std::vector<VirtualParticleBatch::Segment> segments;
  std::vector<ParticleSet::PosType> positions;
  appendVirtualSegment(crowd, 2, 3,
                       {{0.012, -0.007, 0.005}, {-0.009, 0.011, 0.004}},
                       offsets, segments, positions);
  appendVirtualSegment(crowd, 0, 1, {{0.006, 0.003, -0.008}},
                       offsets, segments, positions);
  appendVirtualSegment(crowd, 2, 0,
                       {{-0.004, 0.008, 0.013}, {0.015, -0.006, -0.002}},
                       offsets, segments, positions);
  appendVirtualSegment(crowd, 1, 2,
                       {{0.007, -0.014, 0.009}, {-0.011, 0.005, 0.012}},
                       offsets, segments, positions);
  const VirtualParticleBatch batch(crowd.walkers.size(), offsets, segments,
                                   positions);
  const std::vector<Value> expected = evaluateScalarVirtualBatch(crowd, batch);
  VirtualScratchCrowd scratch(crowd);

  // Populate complete accepted caches, then leave one ordinary proposal live.
  for (std::size_t walker = 0; walker < crowd.walkers.size(); ++walker)
  {
    crowd.walkers[walker]->G = Value(0);
    crowd.walkers[walker]->L = Value(0);
    crowd.components[walker]->evaluateLog(
        *crowd.walkers[walker], crowd.walkers[walker]->G,
        crowd.walkers[walker]->L);
  }
  constexpr int proposed_electron = 1;
  crowd.walkers[2]->makeMove(
      proposed_electron, ParticleSet::PosType{0.003, -0.005, 0.007});
  const Value pending_ratio = crowd.components[2]->ratio(
      *crowd.walkers[2], proposed_electron);
  CHECK(std::isfinite(std::real(pending_ratio)));

  std::vector<testing::PsiFormerCloneStateSnapshot> states_before;
  for (const PsiFormerWF* component : crowd.components)
    states_before.push_back(testing::TestPsiFormerVirtualBatch::cloneState(*component));

  ResourceCollection resource_template("psiformer_flattened_virtual_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection crowd_resource(resource_template);
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource,
                                                            crowd.wfc_list);

    // Shape failure is detected before resource or component state publication.
    std::vector<Value> wrong_extent(batch.size() - 1, Value(-17));
    CHECK_THROWS_AS(crowd.leader.mw_evaluateVirtualRatios(
                        crowd.wfc_list, *crowd.p_list, *scratch.list, batch,
                        wrong_extent),
                    std::invalid_argument);
    CHECK(std::all_of(wrong_extent.begin(), wrong_extent.end(),
                      [](Value value) { return value == Value(-17); }));

    std::vector<Value> actual(batch.size(), Value(-23));
    const WaveFunctionComponent::EvaluationStamp first_stamp =
        crowd.leader.mw_evaluateVirtualRatios(
            crowd.wfc_list, *crowd.p_list, *scratch.list, batch, actual);
    REQUIRE(first_stamp.isVersioned());
    for (std::size_t virtual_index = 0; virtual_index < batch.size();
         ++virtual_index)
      checkValue(actual[virtual_index], expected[virtual_index]);

    const testing::PsiFormerCrowdWorkspaceDiagnostics first_diagnostics =
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            crowd.leader, crowd.wfc_list);
    CHECK(first_diagnostics.reference_configurations == 3);
    CHECK(first_diagnostics.replacement_configurations == batch.size());
    CHECK(first_diagnostics.reference_evaluations == 3);
    CHECK(first_diagnostics.dense_coordinate_bytes_avoided ==
          batch.size() * (crowd.walkers.front()->getTotalNum() - 1) * 3 *
              sizeof(double));

    std::fill(actual.begin(), actual.end(), Value(-31));
    const WaveFunctionComponent::EvaluationStamp repeated_stamp =
        crowd.leader.mw_evaluateVirtualRatios(
            crowd.wfc_list, *crowd.p_list, *scratch.list, batch, actual);
    CHECK(repeated_stamp == first_stamp);
    for (std::size_t virtual_index = 0; virtual_index < batch.size();
         ++virtual_index)
      checkValue(actual[virtual_index], expected[virtual_index]);

    const testing::PsiFormerCrowdWorkspaceDiagnostics repeated_diagnostics =
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            crowd.leader, crowd.wfc_list);
    CHECK(repeated_diagnostics.batch_workspace_identity ==
          first_diagnostics.batch_workspace_identity);
    CHECK(repeated_diagnostics.batch_bytes == first_diagnostics.batch_bytes);
    CHECK(repeated_diagnostics.transient_bytes ==
          first_diagnostics.transient_bytes);

    // Empty work still reports the model version needed by an outer tiled caller.
    const std::vector<std::size_t> empty_offsets{0};
    const std::vector<VirtualParticleBatch::Segment> empty_segments;
    const std::vector<ParticleSet::PosType> empty_positions;
    const VirtualParticleBatch empty_batch(
        crowd.walkers.size(), empty_offsets, empty_segments, empty_positions);
    std::vector<Value> empty_ratios;
    const WaveFunctionComponent::EvaluationStamp empty_stamp =
        crowd.leader.mw_evaluateVirtualRatios(
            crowd.wfc_list, *crowd.p_list, *scratch.list, empty_batch,
            empty_ratios);
    CHECK(empty_stamp == first_stamp);
    CHECK(empty_ratios.empty());

    // Same-version flattened evaluation must not consume or overwrite accepted
    // values, VGL products, or the deliberately pending proposal.
    for (std::size_t walker = 0; walker < crowd.components.size(); ++walker)
      CHECK(testing::TestPsiFormerVirtualBatch::cloneStateMatches(
          *crowd.components[walker], states_before[walker]));

    // A genuine publication changes the opaque stamp.  The following sparse
    // call performs the established lazy invalidation of stale clone caches.
    wftrain::StructuredParameterSnapshot candidate =
        crowd.leader.snapshotParameters();
    candidate.values.at(127) += 1.0e-4;
    crowd.leader.publishParameters(candidate, candidate.version);
    const WaveFunctionComponent::EvaluationStamp changed_stamp =
        crowd.leader.mw_evaluateVirtualRatios(
            crowd.wfc_list, *crowd.p_list, *scratch.list, batch, actual);
    CHECK(changed_stamp != first_stamp);
    for (Value value : actual)
      CHECK(std::isfinite(std::real(value)));
    const WaveFunctionComponent::EvaluationStamp stable_changed_stamp =
        crowd.leader.mw_evaluateVirtualRatios(
            crowd.wfc_list, *crowd.p_list, *scratch.list, batch, actual);
    CHECK(stable_changed_stamp == changed_stamp);
  }

  crowd.walkers[2]->rejectMove(proposed_electron);
}

TEST_CASE("PsiFormer flattened virtual batches publish atomically after ratio failure",
          "[wavefunction][psiformer][multiwalker][ecp][sparse]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  Crowd crowd(files, simulation_cell, 2);
  for (std::size_t walker = 0; walker < crowd.walkers.size(); ++walker)
  {
    crowd.walkers[walker]->G = Value(0);
    crowd.walkers[walker]->L = Value(0);
    crowd.components[walker]->evaluateLog(
        *crowd.walkers[walker], crowd.walkers[walker]->G,
        crowd.walkers[walker]->L);
  }

  const ParticleSet::PosType saved_position = crowd.walkers[1]->R[1];
  crowd.walkers[1]->R[1] = crowd.walkers[1]->R[0];
  crowd.walkers[1]->update();

  std::vector<std::size_t> offsets{0};
  std::vector<VirtualParticleBatch::Segment> segments;
  std::vector<ParticleSet::PosType> positions;
  appendVirtualSegment(crowd, 1, 0, {{0.09, -0.04, 0.03}}, offsets,
                       segments, positions);
  const VirtualParticleBatch batch(crowd.walkers.size(), offsets, segments,
                                   positions);
  VirtualScratchCrowd scratch(crowd);

  std::vector<testing::PsiFormerCloneStateSnapshot> states_before;
  for (const PsiFormerWF* component : crowd.components)
    states_before.push_back(testing::TestPsiFormerVirtualBatch::cloneState(*component));

  ResourceCollection resource_template("psiformer_flattened_failure_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection crowd_resource(resource_template);
  ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource,
                                                          crowd.wfc_list);

  // The exact same-spin reference node is accepted by the value evaluator;
  // forming a finite moved/reference ratio then fails after all sparse outputs exist.
  std::vector<Value> ratios(batch.size(), Value(-41));
  CHECK_THROWS_AS(crowd.leader.mw_evaluateVirtualRatios(
                      crowd.wfc_list, *crowd.p_list, *scratch.list, batch,
                      ratios),
                  std::runtime_error);
  CHECK(ratios.front() == Value(-41));
  for (std::size_t walker = 0; walker < crowd.components.size(); ++walker)
    CHECK(testing::TestPsiFormerVirtualBatch::cloneStateMatches(
        *crowd.components[walker], states_before[walker]));

  // Correcting the reference makes the same resource immediately reusable.
  crowd.walkers[1]->R[1] = saved_position;
  crowd.walkers[1]->update();
  const std::vector<Value> expected = evaluateScalarVirtualBatch(crowd, batch);
  const WaveFunctionComponent::EvaluationStamp retry_stamp =
      crowd.leader.mw_evaluateVirtualRatios(
          crowd.wfc_list, *crowd.p_list, *scratch.list, batch, ratios);
  CHECK(retry_stamp.isVersioned());
  checkValue(ratios.front(), expected.front());
  for (std::size_t walker = 0; walker < crowd.components.size(); ++walker)
    CHECK(testing::TestPsiFormerVirtualBatch::cloneStateMatches(
        *crowd.components[walker], states_before[walker]));
}

TEST_CASE("PsiFormer flattened virtual batches honor oracle and compare backends",
          "[wavefunction][psiformer][multiwalker][ecp][sparse]")
{
  const SimulationCell simulation_cell;
  for (const char* backend : {"oracle", "compare"})
  {
    DYNAMIC_SECTION("backend " << backend)
    {
      ScopedEnvironmentVariable backend_mode("PSIFORMER_VALUE_BACKEND", backend);
      GeneratedFiles files = generateFiles("lih");
      Crowd crowd(files, simulation_cell, 2);
      std::vector<std::size_t> offsets{0};
      std::vector<VirtualParticleBatch::Segment> segments;
      std::vector<ParticleSet::PosType> positions;
      appendVirtualSegment(crowd, 1, 2,
                           {{0.007, -0.006, 0.005}, {-0.004, 0.009, 0.003}},
                           offsets, segments, positions);
      appendVirtualSegment(crowd, 0, 0, {{0.011, 0.002, -0.008}},
                           offsets, segments, positions);
      const VirtualParticleBatch batch(crowd.walkers.size(), offsets, segments,
                                       positions);
      const std::vector<Value> expected = evaluateScalarVirtualBatch(crowd, batch);
      VirtualScratchCrowd scratch(crowd);

      std::vector<testing::PsiFormerCloneStateSnapshot> states_before;
      for (const PsiFormerWF* component : crowd.components)
        states_before.push_back(testing::TestPsiFormerVirtualBatch::cloneState(*component));

      ResourceCollection resource_template("psiformer_flattened_backend_template");
      crowd.leader.createResource(resource_template);
      ResourceCollection crowd_resource(resource_template);
      ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource,
                                                              crowd.wfc_list);
      std::vector<Value> actual(batch.size(), Value(-53));
      const WaveFunctionComponent::EvaluationStamp stamp =
          crowd.leader.mw_evaluateVirtualRatios(
              crowd.wfc_list, *crowd.p_list, *scratch.list, batch, actual);
      CHECK(stamp.isVersioned());
      for (std::size_t virtual_index = 0; virtual_index < batch.size();
           ++virtual_index)
        checkValue(actual[virtual_index], expected[virtual_index]);
      for (std::size_t walker = 0; walker < crowd.components.size(); ++walker)
        CHECK(testing::TestPsiFormerVirtualBatch::cloneStateMatches(
            *crowd.components[walker], states_before[walker]));

      const testing::PsiFormerCrowdWorkspaceDiagnostics diagnostics =
          testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
              crowd.leader, crowd.wfc_list);
      if (std::string(backend) == "compare")
      {
        CHECK(diagnostics.reference_configurations == 2);
        CHECK(diagnostics.replacement_configurations == batch.size());
        CHECK(diagnostics.reference_evaluations == 2);
      }
      else
      {
        CHECK(diagnostics.reference_configurations == 0);
        CHECK(diagnostics.replacement_configurations == 0);
        CHECK(diagnostics.reference_evaluations == 0);
      }
    }
  }
}

TEST_CASE("PsiFormer flattened weighted derivatives match materialized sparse scores",
          "[wavefunction][psiformer][multiwalker][ecp][weighted]")
{
  ScopedEnvironmentVariable score_backend("PSIFORMER_SCORE_BACKEND", "direct");
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  const std::vector<std::size_t> selected_flat_indices{0, 1, 127};
  Crowd crowd(files, simulation_cell, 4, true, selected_flat_indices);
  Crowd oracle_crowd(files, simulation_cell, 4, true, selected_flat_indices);
  const OptVariables active = configureSparseSelectedMapping(crowd.leader);
  const OptVariables oracle_active =
      configureSparseSelectedMapping(oracle_crowd.leader);
  REQUIRE(active.size() == oracle_active.size());

  // Segments are not walker ordered, walker 2 moves two different electrons,
  // and walker 3 is deliberately absent from the descriptor.
  std::vector<std::size_t> offsets{0};
  std::vector<VirtualParticleBatch::Segment> segments;
  std::vector<ParticleSet::PosType> positions;
  appendVirtualSegment(crowd, 2, 3,
                       {{0.012, -0.007, 0.005}, {-0.009, 0.011, 0.004}},
                       offsets, segments, positions);
  appendVirtualSegment(crowd, 0, 1, {{0.006, 0.003, -0.008}}, offsets,
                       segments, positions);
  appendVirtualSegment(crowd, 2, 0,
                       {{-0.004, 0.008, 0.013}, {0.015, -0.006, -0.002}},
                       offsets, segments, positions);
  appendVirtualSegment(crowd, 1, 2,
                       {{0.007, -0.014, 0.009}, {-0.011, 0.005, 0.012}},
                       offsets, segments, positions);
  const VirtualParticleBatch batch(crowd.walkers.size(), offsets, segments,
                                   positions);

  const std::vector<Value> bare_weights{
      makeWeight(0.17, 0.03),  makeWeight(-0.09, 0.02),
      makeWeight(0.13, -0.04), makeWeight(0.21, 0.01),
      makeWeight(-0.08, 0.05), makeWeight(0.11, -0.02),
      makeWeight(-0.06, -0.03)};
  REQUIRE(bare_weights.size() == batch.size());

  // Oversized rows exercise sparse destinations 1, 3, and 5 while protecting
  // padding and trailing entries from accidental dense writes.
  std::vector<std::vector<Value>> initial_derivatives(crowd.walkers.size());
  for (std::size_t walker = 0; walker < initial_derivatives.size(); ++walker)
    initial_derivatives[walker].assign(active.size() + 2,
                                       Value(10.0 + walker));
  const MaterializedWeightedOracle oracle =
      evaluateMaterializedWeightedOracle(oracle_crowd, batch, oracle_active,
                                         bare_weights, initial_derivatives);

  // A second descriptor keeps the same active walkers and parameters while
  // increasing Q, so diagnostics can verify retained derivative staging is Q-independent.
  std::vector<std::size_t> larger_offsets{0};
  std::vector<VirtualParticleBatch::Segment> larger_segments;
  std::vector<ParticleSet::PosType> larger_positions;
  appendVirtualSegment(crowd, 2, 3,
                       {{0.012, -0.007, 0.005}, {-0.009, 0.011, 0.004},
                        {0.003, 0.006, -0.010}, {-0.013, -0.002, 0.007}},
                       larger_offsets, larger_segments, larger_positions);
  appendVirtualSegment(crowd, 0, 1,
                       {{0.006, 0.003, -0.008}, {-0.005, 0.010, 0.004},
                        {0.009, -0.004, 0.006}},
                       larger_offsets, larger_segments, larger_positions);
  appendVirtualSegment(crowd, 1, 2,
                       {{0.007, -0.014, 0.009}, {-0.011, 0.005, 0.012},
                        {0.004, 0.008, -0.006}},
                       larger_offsets, larger_segments, larger_positions);
  const VirtualParticleBatch larger_batch(
      crowd.walkers.size(), larger_offsets, larger_segments, larger_positions);
  const std::vector<Value> larger_bare_weights{
      Value(0.03), Value(-0.04), Value(0.05), Value(-0.06), Value(0.07),
      Value(-0.08), Value(0.09), Value(-0.10), Value(0.11), Value(-0.12)};
  REQUIRE(larger_bare_weights.size() == larger_batch.size());
  const MaterializedWeightedOracle larger_oracle =
      evaluateMaterializedWeightedOracle(
          oracle_crowd, larger_batch, oracle_active, larger_bare_weights,
          initial_derivatives);

  // Populate accepted VGL state and leave one ordinary proposal pending.  The
  // flattened value and score paths must be completely read-only at this version.
  for (std::size_t walker = 0; walker < crowd.walkers.size(); ++walker)
  {
    crowd.walkers[walker]->G = Value(0);
    crowd.walkers[walker]->L = Value(0);
    crowd.components[walker]->evaluateLog(
        *crowd.walkers[walker], crowd.walkers[walker]->G,
        crowd.walkers[walker]->L);
  }
  constexpr int proposed_electron = 1;
  crowd.walkers[2]->makeMove(
      proposed_electron, ParticleSet::PosType{0.003, -0.005, 0.007});
  CHECK(std::isfinite(std::real(
      crowd.components[2]->ratio(*crowd.walkers[2], proposed_electron))));

  std::vector<testing::PsiFormerCloneStateSnapshot> states_before;
  for (const PsiFormerWF* component : crowd.components)
    states_before.push_back(
        testing::TestPsiFormerVirtualBatch::cloneState(*component));

  VirtualScratchCrowd scratch(crowd);
  std::vector<std::vector<ParticleSet::PosType>> scratch_positions_before;
  for (const auto& virtual_particles : scratch.storage)
    scratch_positions_before.emplace_back(virtual_particles->R.begin(),
                                          virtual_particles->R.end());

  ResourceCollection resource_template("psiformer_flattened_weighted_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection crowd_resource(resource_template);
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource,
                                                            crowd.wfc_list);
    std::vector<Value> actual_ratios(batch.size(), Value(-71));
    const WaveFunctionComponent::EvaluationStamp value_stamp =
        crowd.leader.mw_evaluateVirtualRatios(
            crowd.wfc_list, *crowd.p_list, *scratch.list, batch,
            actual_ratios);

    std::vector<std::vector<Value>> actual_derivatives = initial_derivatives;
    std::vector<WaveFunctionComponent::ParameterDerivativeView> derivative_views =
        makeDerivativeViews(actual_derivatives);
    const WaveFunctionComponent::EvaluationStamp derivative_stamp =
        crowd.leader.mw_evaluateVirtualDerivRatiosWeighted(
            crowd.wfc_list, *crowd.p_list, *scratch.list, batch, active,
            oracle.total_weights, derivative_views);
    REQUIRE(value_stamp.isVersioned());
    CHECK(derivative_stamp == value_stamp);

    for (std::size_t virtual_index = 0; virtual_index < batch.size();
         ++virtual_index)
      checkValue(actual_ratios[virtual_index], oracle.ratios[virtual_index]);
    for (std::size_t walker = 0; walker < crowd.walkers.size(); ++walker)
    {
      for (std::size_t parameter = 0; parameter < actual_derivatives[walker].size();
           ++parameter)
        checkValue(actual_derivatives[walker][parameter],
                   oracle.derivatives[walker][parameter], 5.0e-8);
      for (std::size_t padding : {std::size_t{0}, std::size_t{2},
                                  std::size_t{4}, std::size_t{6},
                                  std::size_t{7}})
        CHECK(actual_derivatives[walker][padding] ==
              initial_derivatives[walker][padding]);
    }

    const testing::PsiFormerCrowdWorkspaceDiagnostics first_diagnostics =
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            crowd.leader, crowd.wfc_list);
    CHECK(first_diagnostics.score_workspace_identity != nullptr);
    CHECK(first_diagnostics.weighted_reference_configurations == 3);
    CHECK(first_diagnostics.weighted_replacement_configurations == batch.size());
    CHECK(first_diagnostics.weighted_active_parameters == 3);
    CHECK(first_diagnostics.weighted_derivative_staging_bytes >=
          3 * 3 * sizeof(Value));
    CHECK(testing::TestPsiFormerVirtualBatch::cloneScoreWorkspaceCount(
              crowd.wfc_list) == 0);

    // A warmed call reuses the single score tape and compact staging capacity.
    std::vector<std::vector<Value>> repeated_derivatives = initial_derivatives;
    derivative_views = makeDerivativeViews(repeated_derivatives);
    const WaveFunctionComponent::EvaluationStamp repeated_stamp =
        crowd.leader.mw_evaluateVirtualDerivRatiosWeighted(
            crowd.wfc_list, *crowd.p_list, *scratch.list, batch, active,
            oracle.total_weights, derivative_views);
    CHECK(repeated_stamp == derivative_stamp);
    for (std::size_t walker = 0; walker < crowd.walkers.size(); ++walker)
      for (std::size_t parameter = 0;
           parameter < repeated_derivatives[walker].size(); ++parameter)
        checkValue(repeated_derivatives[walker][parameter],
                   oracle.derivatives[walker][parameter], 5.0e-8);

    const testing::PsiFormerCrowdWorkspaceDiagnostics repeated_diagnostics =
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            crowd.leader, crowd.wfc_list);
    CHECK(repeated_diagnostics.score_workspace_identity ==
          first_diagnostics.score_workspace_identity);
    CHECK(repeated_diagnostics.score_bytes == first_diagnostics.score_bytes);
    CHECK(repeated_diagnostics.weighted_derivative_staging_bytes ==
          first_diagnostics.weighted_derivative_staging_bytes);
    CHECK(repeated_diagnostics.transient_bytes == first_diagnostics.transient_bytes);

    std::vector<std::vector<Value>> larger_derivatives = initial_derivatives;
    derivative_views = makeDerivativeViews(larger_derivatives);
    crowd.leader.mw_evaluateVirtualDerivRatiosWeighted(
        crowd.wfc_list, *crowd.p_list, *scratch.list, larger_batch, active,
        larger_oracle.total_weights, derivative_views);
    for (std::size_t walker = 0; walker < crowd.walkers.size(); ++walker)
      for (std::size_t parameter = 0;
           parameter < larger_derivatives[walker].size(); ++parameter)
        checkValue(larger_derivatives[walker][parameter],
                   larger_oracle.derivatives[walker][parameter], 5.0e-8);
    const testing::PsiFormerCrowdWorkspaceDiagnostics larger_diagnostics =
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            crowd.leader, crowd.wfc_list);
    CHECK(larger_diagnostics.weighted_reference_configurations == 3);
    CHECK(larger_diagnostics.weighted_replacement_configurations ==
          larger_batch.size());
    CHECK(larger_diagnostics.weighted_derivative_staging_bytes ==
          first_diagnostics.weighted_derivative_staging_bytes);
    CHECK(larger_diagnostics.score_workspace_identity ==
          first_diagnostics.score_workspace_identity);
    CHECK(larger_diagnostics.transient_bytes ==
          repeated_diagnostics.transient_bytes);
  }

  for (std::size_t walker = 0; walker < crowd.components.size(); ++walker)
  {
    CHECK(testing::TestPsiFormerVirtualBatch::cloneStateMatches(
        *crowd.components[walker], states_before[walker]));
    CHECK(std::vector<ParticleSet::PosType>(scratch.storage[walker]->R.begin(),
                                            scratch.storage[walker]->R.end()) ==
          scratch_positions_before[walker]);
  }
  crowd.walkers[2]->rejectMove(proposed_electron);
}

TEST_CASE("PsiFormer flattened weighted scratch follows mapped active parameters",
          "[wavefunction][psiformer][multiwalker][ecp][weighted]")
{
  ScopedEnvironmentVariable score_backend("PSIFORMER_SCORE_BACKEND", "direct");
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  std::vector<std::size_t> many_selected(128);
  std::iota(many_selected.begin(), many_selected.end(), std::size_t{0});

  Crowd many_crowd(files, simulation_cell, 2, true, many_selected);
  Crowd one_crowd(files, simulation_cell, 2, true, {0});
  const OptVariables many_active =
      configureFirstSelectedOnly(many_crowd.leader, many_selected.size());
  const OptVariables one_active =
      configureFirstSelectedOnly(one_crowd.leader, 1);
  REQUIRE(many_active.size() == one_active.size());

  std::vector<std::size_t> offsets{0};
  std::vector<VirtualParticleBatch::Segment> segments;
  std::vector<ParticleSet::PosType> positions;
  appendVirtualSegment(many_crowd, 0, 1,
                       {{0.006, 0.003, -0.008}, {-0.005, 0.010, 0.004}},
                       offsets, segments, positions);
  appendVirtualSegment(many_crowd, 1, 2, {{0.007, -0.014, 0.009}}, offsets,
                       segments, positions);
  const VirtualParticleBatch batch(2, offsets, segments, positions);
  const std::vector<Value> weights{Value(0.14), Value(-0.08), Value(0.11)};

  auto evaluate = [&](Crowd& crowd, const OptVariables& active,
                      const std::string& resource_name) {
    VirtualScratchCrowd scratch(crowd);
    std::vector<std::vector<Value>> derivatives(
        2, std::vector<Value>(active.size(), Value(3.5)));
    std::vector<WaveFunctionComponent::ParameterDerivativeView> views =
        makeDerivativeViews(derivatives);
    ResourceCollection resource_template(resource_name);
    crowd.leader.createResource(resource_template);
    ResourceCollection resource(resource_template);
    {
      ResourceCollectionTeamLock<WaveFunctionComponent> lock(resource,
                                                              crowd.wfc_list);
      crowd.leader.mw_evaluateVirtualDerivRatiosWeighted(
          crowd.wfc_list, *crowd.p_list, *scratch.list, batch, active, weights,
          views);
      return std::pair{
          std::move(derivatives),
          testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
              crowd.leader, crowd.wfc_list)};
    }
  };

  auto [many_derivatives, many_diagnostics] =
      evaluate(many_crowd, many_active, "psiformer_many_selected_partial");
  auto [one_derivatives, one_diagnostics] =
      evaluate(one_crowd, one_active, "psiformer_one_selected_partial");
  for (std::size_t walker = 0; walker < many_derivatives.size(); ++walker)
  {
    CHECK(many_derivatives[walker][0] == Value(3.5));
    CHECK(one_derivatives[walker][0] == Value(3.5));
    checkValue(many_derivatives[walker][1], one_derivatives[walker][1],
               5.0e-8);
  }
  CHECK(many_diagnostics.weighted_active_parameters == 1);
  CHECK(one_diagnostics.weighted_active_parameters == 1);
  CHECK(many_diagnostics.weighted_derivative_staging_bytes ==
        one_diagnostics.weighted_derivative_staging_bytes);
  CHECK(many_diagnostics.transient_bytes == one_diagnostics.transient_bytes);
  CHECK(many_diagnostics.score_bytes == one_diagnostics.score_bytes);

  // A large selected set with no mapped PsiFormer variables must not create
  // either the full score tape or any active-parameter staging.
  Crowd inactive_crowd(files, simulation_cell, 2, true, many_selected);
  OptVariables unrelated_active;
  unrelated_active.insert("ordinary_only", 1.0, true, optimize::LINEAR_P);
  unrelated_active.resetIndex();
  inactive_crowd.leader.checkOutVariables(unrelated_active);
  VirtualScratchCrowd inactive_scratch(inactive_crowd);
  std::vector<std::vector<Value>> inactive_derivatives(
      2, std::vector<Value>(1, Value(7.0)));
  std::vector<WaveFunctionComponent::ParameterDerivativeView> inactive_views =
      makeDerivativeViews(inactive_derivatives);
  ResourceCollection inactive_template("psiformer_many_selected_inactive");
  inactive_crowd.leader.createResource(inactive_template);
  ResourceCollection inactive_resource(inactive_template);
  ResourceCollectionTeamLock<WaveFunctionComponent> inactive_lock(
      inactive_resource, inactive_crowd.wfc_list);
  inactive_crowd.leader.mw_evaluateVirtualDerivRatiosWeighted(
      inactive_crowd.wfc_list, *inactive_crowd.p_list, *inactive_scratch.list,
      batch, unrelated_active, weights, inactive_views);
  CHECK(inactive_derivatives ==
        std::vector<std::vector<Value>>(2, std::vector<Value>(1, Value(7.0))));
  const testing::PsiFormerCrowdWorkspaceDiagnostics inactive_diagnostics =
      testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
          inactive_crowd.leader, inactive_crowd.wfc_list);
  CHECK(inactive_diagnostics.score_workspace_identity == nullptr);
  CHECK(inactive_diagnostics.weighted_active_parameters == 0);
  CHECK(inactive_diagnostics.weighted_derivative_staging_bytes == 0);
}

TEST_CASE("TrialWaveFunction flattened weighted dispatch accepts PsiFormer version stamps",
          "[wavefunction][psiformer][multiwalker][ecp][weighted][trialwf]")
{
  ScopedEnvironmentVariable score_backend("PSIFORMER_SCORE_BACKEND", "direct");
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  const std::vector<std::size_t> selected_flat_indices{0, 1, 127};

  RuntimeOptions runtime_options;
  auto walker0 = makeWalker(simulation_cell, 0);
  auto walker1 = makeWalker(simulation_cell, 1);
  TrialWaveFunction wavefunction0(runtime_options, "pf_weighted_twf");
  auto component = std::make_unique<PsiFormerWF>(
      "pf_mw", files.parameters.string(), files.configuration.string(), true,
      selected_flat_indices);
  PsiFormerWF* component0 = component.get();
  wavefunction0.addComponent(std::move(component));
  const OptVariables active = configureSparseSelectedMapping(*component0);
  wavefunction0.checkOutVariables(active);

  std::unique_ptr<TrialWaveFunction> wavefunction1 =
      wavefunction0.makeClone(*walker1);
  wavefunction1->checkOutVariables(active);
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(wavefunction0);
  wavefunctions.push_back(wavefunction0);
  wavefunctions.push_back(*wavefunction1);
  RefVectorWithLeader<ParticleSet> particles(*walker0);
  particles.push_back(*walker0);
  particles.push_back(*walker1);

  // Use a distinct component crowd for the materialized scalar oracle so the
  // integration fixture begins without clone-local score tapes.
  Crowd oracle_crowd(files, simulation_cell, 2, true, selected_flat_indices);
  const OptVariables oracle_active =
      configureSparseSelectedMapping(oracle_crowd.leader);
  std::vector<std::size_t> offsets{0};
  std::vector<VirtualParticleBatch::Segment> segments;
  std::vector<ParticleSet::PosType> positions;
  appendVirtualSegment(oracle_crowd, 1, 2,
                       {{0.007, -0.006, 0.005}, {-0.004, 0.009, 0.003}},
                       offsets, segments, positions);
  appendVirtualSegment(oracle_crowd, 0, 0, {{0.011, 0.002, -0.008}},
                       offsets, segments, positions);
  appendVirtualSegment(oracle_crowd, 1, 3, {{-0.006, 0.004, 0.010}},
                       offsets, segments, positions);
  const VirtualParticleBatch batch(2, offsets, segments, positions);
  const std::vector<Value> bare_weights{
      makeWeight(0.19, 0.03), makeWeight(-0.12, -0.02),
      makeWeight(0.08, 0.01), makeWeight(0.16, -0.04)};
  std::vector<std::vector<Value>> initial_derivatives(
      2, std::vector<Value>(active.size(), Value(4.25)));
  const MaterializedWeightedOracle oracle =
      evaluateMaterializedWeightedOracle(
          oracle_crowd, batch, oracle_active, bare_weights,
          initial_derivatives);

  auto scratch0 = std::make_unique<VirtualParticleSet>(*walker0);
  auto scratch1 = std::make_unique<VirtualParticleSet>(*walker1);
  RefVectorWithLeader<VirtualParticleSet> scratch(*scratch0);
  scratch.push_back(*scratch0);
  scratch.push_back(*scratch1);

  std::vector<Value> ratios(batch.size(), Value(-79));
  std::vector<std::vector<Value>> derivatives = initial_derivatives;
  std::vector<TrialWaveFunction::ParameterDerivativeView> derivative_views =
      makeDerivativeViews(derivatives);
  std::vector<TrialWaveFunction::EvaluationStamp> stamps;

  ResourceCollection resource_collection("psiformer_weighted_twf_resources");
  wavefunction0.createResource(resource_collection);
  {
    ResourceCollectionTeamLock<TrialWaveFunction> lock(resource_collection,
                                                        wavefunctions);
    TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted(
        wavefunctions, particles, scratch, batch, active, bare_weights, ratios,
        derivative_views, stamps);
  }

  for (std::size_t virtual_index = 0; virtual_index < batch.size();
       ++virtual_index)
    checkValue(ratios[virtual_index], oracle.ratios[virtual_index]);
  for (std::size_t walker = 0; walker < derivatives.size(); ++walker)
    for (std::size_t parameter = 0; parameter < derivatives[walker].size();
         ++parameter)
      checkValue(derivatives[walker][parameter],
                 oracle.derivatives[walker][parameter], 5.0e-8);
  REQUIRE(stamps.size() == 1);
  CHECK(stamps.front().isVersioned());
}

TEST_CASE("PsiFormer flattened weighted no-work paths retain version semantics",
          "[wavefunction][psiformer][multiwalker][ecp][weighted]")
{
  ScopedEnvironmentVariable score_backend("PSIFORMER_SCORE_BACKEND", "direct");
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;

  SECTION("fixed component with virtual positions")
  {
    Crowd crowd(files, simulation_cell, 2);
    std::vector<std::size_t> offsets{0};
    std::vector<VirtualParticleBatch::Segment> segments;
    std::vector<ParticleSet::PosType> positions;
    appendVirtualSegment(crowd, 1, 2,
                         {{0.007, -0.006, 0.005}, {-0.004, 0.009, 0.003}},
                         offsets, segments, positions);
    const VirtualParticleBatch batch(2, offsets, segments, positions);
    VirtualScratchCrowd scratch(crowd);
    const OptVariables no_parameters;
    std::vector<Value> ratios(batch.size(), Value(-83));
    std::vector<Value> total_weights(batch.size(), Value(0.125));
    std::vector<std::vector<Value>> derivatives(2);
    std::vector<WaveFunctionComponent::ParameterDerivativeView> views =
        makeDerivativeViews(derivatives);

    ResourceCollection resource_template("psiformer_fixed_weighted_template");
    crowd.leader.createResource(resource_template);
    ResourceCollection resource(resource_template);
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(resource,
                                                            crowd.wfc_list);
    const WaveFunctionComponent::EvaluationStamp value_stamp =
        crowd.leader.mw_evaluateVirtualRatios(
            crowd.wfc_list, *crowd.p_list, *scratch.list, batch, ratios);
    const WaveFunctionComponent::EvaluationStamp weighted_stamp =
        crowd.leader.mw_evaluateVirtualDerivRatiosWeighted(
            crowd.wfc_list, *crowd.p_list, *scratch.list, batch,
            no_parameters, total_weights, views);
    REQUIRE(value_stamp.isVersioned());
    CHECK(weighted_stamp == value_stamp);
    CHECK(derivatives[0].empty());
    CHECK(derivatives[1].empty());

    const testing::PsiFormerCrowdWorkspaceDiagnostics diagnostics =
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            crowd.leader, crowd.wfc_list);
    CHECK(diagnostics.score_workspace_identity == nullptr);
    CHECK(diagnostics.weighted_reference_configurations == 0);
    CHECK(diagnostics.weighted_replacement_configurations == 0);
    CHECK(diagnostics.weighted_active_parameters == 0);
  }

  SECTION("empty active descriptor before and after parameter publication")
  {
    Crowd crowd(files, simulation_cell, 2, true, {0, 1, 127});
    const OptVariables active = configureSparseSelectedMapping(crowd.leader);
    const std::vector<std::size_t> offsets{0};
    const std::vector<VirtualParticleBatch::Segment> segments;
    const std::vector<ParticleSet::PosType> positions;
    const VirtualParticleBatch empty_batch(2, offsets, segments, positions);
    VirtualScratchCrowd scratch(crowd);
    std::vector<Value> ratios;
    const std::vector<Value> weights;
    std::vector<std::vector<Value>> derivatives(
        2, std::vector<Value>(active.size(), Value(6.5)));
    const std::vector<std::vector<Value>> original_derivatives = derivatives;
    std::vector<WaveFunctionComponent::ParameterDerivativeView> views =
        makeDerivativeViews(derivatives);

    ResourceCollection resource_template("psiformer_empty_weighted_template");
    crowd.leader.createResource(resource_template);
    ResourceCollection resource(resource_template);
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(resource,
                                                            crowd.wfc_list);
    const WaveFunctionComponent::EvaluationStamp first_value_stamp =
        crowd.leader.mw_evaluateVirtualRatios(
            crowd.wfc_list, *crowd.p_list, *scratch.list, empty_batch, ratios);
    const WaveFunctionComponent::EvaluationStamp first_weighted_stamp =
        crowd.leader.mw_evaluateVirtualDerivRatiosWeighted(
            crowd.wfc_list, *crowd.p_list, *scratch.list, empty_batch, active,
            weights, views);
    CHECK(first_weighted_stamp == first_value_stamp);
    CHECK(derivatives == original_derivatives);

    wftrain::StructuredParameterSnapshot candidate =
        crowd.leader.snapshotParameters();
    candidate.values.at(127) += 1.0e-4;
    crowd.leader.publishParameters(candidate, candidate.version);
    const WaveFunctionComponent::EvaluationStamp changed_value_stamp =
        crowd.leader.mw_evaluateVirtualRatios(
            crowd.wfc_list, *crowd.p_list, *scratch.list, empty_batch, ratios);
    const WaveFunctionComponent::EvaluationStamp changed_weighted_stamp =
        crowd.leader.mw_evaluateVirtualDerivRatiosWeighted(
            crowd.wfc_list, *crowd.p_list, *scratch.list, empty_batch, active,
            weights, views);
    CHECK(changed_value_stamp != first_value_stamp);
    CHECK(changed_weighted_stamp == changed_value_stamp);
    CHECK(derivatives == original_derivatives);

    const testing::PsiFormerCrowdWorkspaceDiagnostics diagnostics =
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            crowd.leader, crowd.wfc_list);
    CHECK(diagnostics.score_workspace_identity == nullptr);
    CHECK(diagnostics.weighted_reference_configurations == 0);
    CHECK(diagnostics.weighted_replacement_configurations == 0);
    CHECK(diagnostics.weighted_active_parameters == 3);
    CHECK(diagnostics.weighted_derivative_staging_bytes == 0);
  }

  SECTION("zero-walker descriptor")
  {
    PsiFormerWF leader("pf_mw", files.parameters.string(),
                       files.configuration.string(), true, {0, 1, 127});
    const OptVariables active = configureSparseSelectedMapping(leader);
    auto reference = makeWalker(simulation_cell, 0);
    VirtualParticleSet scratch_object(*reference);
    RefVectorWithLeader<WaveFunctionComponent> components(leader);
    RefVectorWithLeader<ParticleSet> particles(*reference);
    RefVectorWithLeader<VirtualParticleSet> scratch(scratch_object);
    const std::vector<std::size_t> offsets{0};
    const std::vector<VirtualParticleBatch::Segment> segments;
    const std::vector<ParticleSet::PosType> positions;
    const VirtualParticleBatch empty_batch(0, offsets, segments, positions);
    std::vector<Value> ratios;
    const std::vector<Value> weights;
    const std::vector<WaveFunctionComponent::ParameterDerivativeView> views;

    ResourceCollection resource_template("psiformer_zero_walker_template");
    leader.createResource(resource_template);
    ResourceCollection resource(resource_template);
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(resource, components);
    const WaveFunctionComponent::EvaluationStamp value_stamp =
        leader.mw_evaluateVirtualRatios(components, particles, scratch,
                                        empty_batch, ratios);
    const WaveFunctionComponent::EvaluationStamp weighted_stamp =
        leader.mw_evaluateVirtualDerivRatiosWeighted(
            components, particles, scratch, empty_batch, active, weights,
            views);
    REQUIRE(value_stamp.isVersioned());
    CHECK(weighted_stamp == value_stamp);
  }
}

TEST_CASE("PsiFormer flattened weighted derivatives honor score backends",
          "[wavefunction][psiformer][multiwalker][ecp][weighted]")
{
  const SimulationCell simulation_cell;
  const std::vector<std::size_t> selected_flat_indices{0, 1, 127};
  for (const char* backend : {"oracle", "compare"})
  {
    DYNAMIC_SECTION("backend " << backend)
    {
      ScopedEnvironmentVariable backend_mode("PSIFORMER_SCORE_BACKEND", backend);
      GeneratedFiles files = generateFiles("lih");
      Crowd crowd(files, simulation_cell, 2, true, selected_flat_indices);
      Crowd oracle_crowd(files, simulation_cell, 2, true,
                         selected_flat_indices);
      const OptVariables active = configureSparseSelectedMapping(crowd.leader);
      const OptVariables oracle_active =
          configureSparseSelectedMapping(oracle_crowd.leader);

      std::vector<std::size_t> offsets{0};
      std::vector<VirtualParticleBatch::Segment> segments;
      std::vector<ParticleSet::PosType> positions;
      appendVirtualSegment(crowd, 1, 2,
                           {{0.007, -0.006, 0.005}, {-0.004, 0.009, 0.003}},
                           offsets, segments, positions);
      appendVirtualSegment(crowd, 0, 0, {{0.011, 0.002, -0.008}}, offsets,
                           segments, positions);
      const VirtualParticleBatch batch(2, offsets, segments, positions);
      const std::vector<Value> bare_weights{
          makeWeight(0.17, 0.02), makeWeight(-0.09, -0.03),
          makeWeight(0.13, 0.04)};
      std::vector<std::vector<Value>> initial_derivatives(
          2, std::vector<Value>(active.size(), Value(2.75)));
      const MaterializedWeightedOracle oracle =
          evaluateMaterializedWeightedOracle(
              oracle_crowd, batch, oracle_active, bare_weights,
              initial_derivatives);
      VirtualScratchCrowd scratch(crowd);

      ResourceCollection resource_template(
          std::string("psiformer_weighted_backend_") + backend);
      crowd.leader.createResource(resource_template);
      ResourceCollection resource(resource_template);
      ResourceCollectionTeamLock<WaveFunctionComponent> lock(resource,
                                                              crowd.wfc_list);
      std::vector<std::vector<Value>> derivatives = initial_derivatives;
      std::vector<WaveFunctionComponent::ParameterDerivativeView> views =
          makeDerivativeViews(derivatives);
      const WaveFunctionComponent::EvaluationStamp stamp =
          crowd.leader.mw_evaluateVirtualDerivRatiosWeighted(
              crowd.wfc_list, *crowd.p_list, *scratch.list, batch, active,
              oracle.total_weights, views);
      REQUIRE(stamp.isVersioned());
      for (std::size_t walker = 0; walker < derivatives.size(); ++walker)
        for (std::size_t parameter = 0; parameter < derivatives[walker].size();
             ++parameter)
          checkValue(derivatives[walker][parameter],
                     oracle.derivatives[walker][parameter], 5.0e-8);

      const testing::PsiFormerCrowdWorkspaceDiagnostics diagnostics =
          testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
              crowd.leader, crowd.wfc_list);
      CHECK(diagnostics.backend_modes[2] == backend);
      CHECK((diagnostics.score_workspace_identity != nullptr) ==
            (std::string(backend) == "compare"));
      CHECK(testing::TestPsiFormerVirtualBatch::cloneScoreWorkspaceCount(
                crowd.wfc_list) == 0);
      CHECK(diagnostics.weighted_reference_configurations == 2);
      CHECK(diagnostics.weighted_replacement_configurations == batch.size());
    }
  }
}

TEST_CASE("PsiFormer flattened weighted derivatives publish atomically after score failure",
          "[wavefunction][psiformer][multiwalker][ecp][weighted]")
{
  ScopedEnvironmentVariable score_backend("PSIFORMER_SCORE_BACKEND", "direct");
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  const std::vector<std::size_t> selected_flat_indices{0, 1, 127};
  Crowd crowd(files, simulation_cell, 2, true, selected_flat_indices);
  Crowd oracle_crowd(files, simulation_cell, 2, true, selected_flat_indices);
  const OptVariables active = configureSparseSelectedMapping(crowd.leader);
  const OptVariables oracle_active =
      configureSparseSelectedMapping(oracle_crowd.leader);

  std::vector<std::size_t> offsets{0};
  std::vector<VirtualParticleBatch::Segment> segments;
  std::vector<ParticleSet::PosType> positions;
  appendVirtualSegment(crowd, 0, 1, {{0.006, 0.003, -0.008}}, offsets,
                       segments, positions);
  appendVirtualSegment(crowd, 1, 0, {{0.09, -0.04, 0.03}}, offsets, segments,
                       positions);
  const VirtualParticleBatch batch(2, offsets, segments, positions);
  const std::vector<Value> bare_weights{Value(0.18), Value(-0.11)};
  std::vector<std::vector<Value>> initial_derivatives(
      2, std::vector<Value>(active.size(), Value(9.0)));
  const MaterializedWeightedOracle oracle =
      evaluateMaterializedWeightedOracle(
          oracle_crowd, batch, oracle_active, bare_weights,
          initial_derivatives);
  VirtualScratchCrowd scratch(crowd);

  // Collapse two same-spin electrons in the later reference configuration.
  const ParticleSet::PosType saved_position = crowd.walkers[1]->R[1];
  crowd.walkers[1]->R[1] = crowd.walkers[1]->R[0];
  crowd.walkers[1]->update();
  std::vector<testing::PsiFormerCloneStateSnapshot> states_before;
  for (const PsiFormerWF* component : crowd.components)
    states_before.push_back(
        testing::TestPsiFormerVirtualBatch::cloneState(*component));

  ResourceCollection resource_template("psiformer_weighted_failure_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection resource(resource_template);
  ResourceCollectionTeamLock<WaveFunctionComponent> lock(resource,
                                                          crowd.wfc_list);
  std::vector<std::vector<Value>> derivatives = initial_derivatives;
  std::vector<WaveFunctionComponent::ParameterDerivativeView> views =
      makeDerivativeViews(derivatives);
  CHECK_THROWS_AS(crowd.leader.mw_evaluateVirtualDerivRatiosWeighted(
                      crowd.wfc_list, *crowd.p_list, *scratch.list, batch,
                      active, oracle.total_weights, views),
                  std::domain_error);
  CHECK(derivatives == initial_derivatives);
  for (std::size_t walker = 0; walker < crowd.components.size(); ++walker)
    CHECK(testing::TestPsiFormerVirtualBatch::cloneStateMatches(
        *crowd.components[walker], states_before[walker]));
  const testing::PsiFormerCrowdWorkspaceDiagnostics failed_diagnostics =
      testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
          crowd.leader, crowd.wfc_list);
  CHECK(failed_diagnostics.weighted_reference_configurations == 0);
  CHECK(failed_diagnostics.weighted_replacement_configurations == 0);

  // Repair the reference and reuse the same acquired resource immediately.
  crowd.walkers[1]->R[1] = saved_position;
  crowd.walkers[1]->update();
  const WaveFunctionComponent::EvaluationStamp retry_stamp =
      crowd.leader.mw_evaluateVirtualDerivRatiosWeighted(
          crowd.wfc_list, *crowd.p_list, *scratch.list, batch, active,
          oracle.total_weights, views);
  REQUIRE(retry_stamp.isVersioned());
  for (std::size_t walker = 0; walker < derivatives.size(); ++walker)
    for (std::size_t parameter = 0; parameter < derivatives[walker].size();
         ++parameter)
      checkValue(derivatives[walker][parameter],
                 oracle.derivatives[walker][parameter], 5.0e-8);
  const testing::PsiFormerCrowdWorkspaceDiagnostics retry_diagnostics =
      testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
          crowd.leader, crowd.wfc_list);
  CHECK(retry_diagnostics.weighted_reference_configurations == 2);
  CHECK(retry_diagnostics.weighted_replacement_configurations == batch.size());
}

TEST_CASE("PsiFormer selected-electron proposals are atomic full-VGL transactions",
          "[wavefunction][psiformer][multiwalker][multiparticle]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  constexpr std::size_t walker_count = 3;
  Crowd crowd(files, simulation_cell, walker_count);
  const std::size_t electron_count = crowd.walkers.front()->getTotalNum();

  std::vector<ParticleSet::ParticleGradient> initial_gradient(walker_count);
  std::vector<ParticleSet::ParticleLaplacian> initial_laplacian(walker_count);
  std::vector<PsiFormerWF::LogValue> initial_log(walker_count);
  std::vector<std::vector<ParticleSet::PosType>> initial_positions(walker_count);
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    initial_gradient[walker].resize(electron_count);
    initial_laplacian[walker].resize(electron_count);
    initial_gradient[walker]  = Value(0);
    initial_laplacian[walker] = Value(0);
    initial_log[walker] = crowd.components[walker]->evaluateLog(
        *crowd.walkers[walker], initial_gradient[walker], initial_laplacian[walker]);
    initial_positions[walker].assign(crowd.walkers[walker]->R.begin(),
                                     crowd.walkers[walker]->R.end());
  }

  using Moves = MCMultiParticleMoves<CoordsType::POS>;
  const std::vector<std::size_t> offsets{0, 2, 3, 5};
  const std::vector<Moves::IndexType> indices{0, 2, 1, 0, 3};
  std::vector<Moves::PosType> positions{
      crowd.walkers[0]->R[0] + Moves::PosType{0.021, -0.014, 0.009},
      crowd.walkers[0]->R[2] + Moves::PosType{-0.017, 0.011, 0.006},
      // An exact no-op replacement exercises safe reuse of accepted full VGL state.
      crowd.walkers[1]->R[1],
      crowd.walkers[2]->R[0] + Moves::PosType{0.013, 0.019, -0.008},
      crowd.walkers[2]->R[3] + Moves::PosType{-0.015, 0.007, 0.012}};
  const Moves moves(offsets, indices, positions);

  std::vector<ParticleSet::ParticleGradient> expected_gradient(walker_count);
  std::vector<ParticleSet::ParticleLaplacian> expected_laplacian(walker_count);
  std::vector<PsiFormerWF::LogValue> expected_log(walker_count);
  PsiFormerWF oracle("pf_selected_oracle", files.parameters.string(), files.configuration.string());
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    auto proposed = makeWalker(simulation_cell, walker);
    const auto slice = moves.slice(walker);
    for (std::size_t selected = 0; selected < slice.size(); ++selected)
      proposed->R[slice.particleIndex(selected)] = slice.proposedPosition(selected);
    proposed->update();
    expected_gradient[walker].resize(electron_count);
    expected_laplacian[walker].resize(electron_count);
    expected_gradient[walker]  = Value(0);
    expected_laplacian[walker] = Value(0);
    expected_log[walker] = oracle.evaluateLog(
        *proposed, expected_gradient[walker], expected_laplacian[walker]);
  }

  const Value gradient_seed(0.125);
  const Value laplacian_seed(-0.375);
  std::vector<ParticleSet::ParticleGradient> proposed_gradient(walker_count);
  std::vector<ParticleSet::ParticleLaplacian> proposed_laplacian(walker_count);
  RefVector<ParticleSet::ParticleGradient> proposed_gradient_list;
  RefVector<ParticleSet::ParticleLaplacian> proposed_laplacian_list;
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    proposed_gradient[walker].resize(electron_count);
    proposed_laplacian[walker].resize(electron_count);
    proposed_gradient[walker]  = gradient_seed;
    proposed_laplacian[walker] = laplacian_seed;
    proposed_gradient_list.push_back(proposed_gradient[walker]);
    proposed_laplacian_list.push_back(proposed_laplacian[walker]);
  }
  std::vector<PsiFormerWF::LogValue> log_ratios(walker_count, PsiFormerWF::LogValue(19.0));

  ResourceCollection wf_template("psiformer_selected_template");
  crowd.leader.createResource(wf_template);
  ResourceCollection wf_resources(wf_template);
  ResourceCollection particle_resources("psiformer_selected_particles");
  crowd.walkers.front()->createResource(particle_resources);
  CHECK_THROWS_AS(crowd.leader.mw_evaluateMultiParticleMove(
                      crowd.wfc_list, *crowd.p_list, moves, log_ratios,
                      proposed_gradient_list, proposed_laplacian_list),
                  std::logic_error);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, *crowd.p_list);
  ResourceCollectionTeamLock<WaveFunctionComponent> wf_lock(wf_resources, crowd.wfc_list);

  REQUIRE(crowd.leader.supportsMultiParticleMoves());
  proposed_laplacian.back().resize(electron_count - 1);
  CHECK_THROWS_AS(crowd.leader.mw_evaluateMultiParticleMove(
                      crowd.wfc_list, *crowd.p_list, moves, log_ratios,
                      proposed_gradient_list, proposed_laplacian_list),
                  std::invalid_argument);
  proposed_laplacian.back().resize(electron_count);
  proposed_laplacian.back() = laplacian_seed;
  crowd.leader.mw_evaluateMultiParticleMove(
      crowd.wfc_list, *crowd.p_list, moves, log_ratios,
      proposed_gradient_list, proposed_laplacian_list);

  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    // Evaluation consumes descriptor-owned absolute coordinates and leaves P accepted.
    for (std::size_t electron = 0; electron < electron_count; ++electron)
      for (int dimension = 0; dimension < 3; ++dimension)
        CHECK(crowd.walkers[walker]->R[electron][dimension] ==
              initial_positions[walker][electron][dimension]);
    checkLog(crowd.components[walker]->get_log_value(), initial_log[walker]);
    checkLog(log_ratios[walker], expected_log[walker] - initial_log[walker]);
    for (std::size_t electron = 0; electron < electron_count; ++electron)
    {
      for (int dimension = 0; dimension < 3; ++dimension)
        checkValue(proposed_gradient[walker][electron][dimension],
                   gradient_seed + expected_gradient[walker][electron][dimension], 3.0e-8);
      checkValue(proposed_laplacian[walker][electron],
                 laplacian_seed + expected_laplacian[walker][electron], 3.0e-7);
    }
  }

  // A different descriptor cannot consume the pending proposal, and failure is
  // crowd-atomic so the original transaction remains resolvable.
  std::vector<Moves::PosType> mismatched_positions = positions;
  mismatched_positions.back()[0] += 1.0e-4;
  const Moves mismatched_moves(offsets, indices, std::move(mismatched_positions));
  CHECK_THROWS_AS(crowd.leader.mw_accept_rejectMultiParticleMove(
                      crowd.wfc_list, *crowd.p_list, mismatched_moves,
                      std::vector<bool>(walker_count, false)),
                  std::logic_error);
  PsiFormerWF::WFBufferType pending_buffer;
  CHECK_THROWS_AS(crowd.components.front()->registerData(
                      *crowd.walkers.front(), pending_buffer),
                  std::logic_error);

  std::vector<bool> valid;
  ParticleSet::mw_makeMoveSelectedParticles(*crowd.p_list, moves, valid);
  CHECK(std::all_of(valid.begin(), valid.end(), [](bool value) { return value; }));
  const std::vector<bool> accepted{true, false, true};
  crowd.leader.mw_accept_rejectMultiParticleMove(
      crowd.wfc_list, *crowd.p_list, moves, accepted);
  ParticleSet::mw_accept_rejectMoveSelectedParticles(*crowd.p_list, accepted);

  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    const auto& final_gradient = accepted[walker] ? expected_gradient[walker] : initial_gradient[walker];
    const auto& final_laplacian = accepted[walker] ? expected_laplacian[walker] : initial_laplacian[walker];
    checkLog(crowd.components[walker]->get_log_value(),
             accepted[walker] ? expected_log[walker] : initial_log[walker]);

    // updateBuffer(false) must be able to reuse the promoted complete spatial cache.
    PsiFormerWF::WFBufferType buffer;
    crowd.components[walker]->registerData(*crowd.walkers[walker], buffer);
    buffer.allocate();
    buffer.rewind();
    crowd.walkers[walker]->G = Value(0);
    crowd.walkers[walker]->L = Value(0);
    checkLog(crowd.components[walker]->updateBuffer(
                 *crowd.walkers[walker], buffer, false),
             accepted[walker] ? expected_log[walker] : initial_log[walker]);
    for (std::size_t electron = 0; electron < electron_count; ++electron)
    {
      checkGrad(crowd.walkers[walker]->G[electron], final_gradient[electron]);
      checkValue(crowd.walkers[walker]->L[electron], final_laplacian[electron], 3.0e-7);
    }
  }
}

TEST_CASE("PsiFormer selected-electron proposals honor oracle and compare backends",
          "[wavefunction][psiformer][multiwalker][multiparticle][threading]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  constexpr std::size_t walker_count = 3;

  for (const char* backend : {"oracle", "compare"})
  {
    DYNAMIC_SECTION("spatial backend " << backend)
    {
      ScopedEnvironmentVariable backend_mode("PSIFORMER_SPATIAL_BACKEND", backend);
      Crowd crowd(files, simulation_cell, walker_count);
      const std::size_t electron_count = crowd.walkers.front()->getTotalNum();

      std::vector<ParticleSet::ParticleGradient> accepted_gradient(walker_count);
      std::vector<ParticleSet::ParticleLaplacian> accepted_laplacian(walker_count);
      std::vector<PsiFormerWF::LogValue> accepted_log(walker_count);
      for (std::size_t walker = 0; walker < walker_count; ++walker)
      {
        accepted_gradient[walker].resize(electron_count);
        accepted_laplacian[walker].resize(electron_count);
        accepted_gradient[walker]  = Value(0);
        accepted_laplacian[walker] = Value(0);
        accepted_log[walker] = crowd.components[walker]->evaluateLog(
            *crowd.walkers[walker], accepted_gradient[walker],
            accepted_laplacian[walker]);
      }

      using Moves = MCMultiParticleMoves<CoordsType::POS>;
      const std::vector<std::size_t> offsets{0, 1, 2, 4};
      const std::vector<Moves::IndexType> indices{0, 1, 0, 3};
      const std::vector<Moves::PosType> positions{
          crowd.walkers[0]->R[0] + Moves::PosType{0.013, -0.008, 0.005},
          // Exact replacement keeps one row on the accepted full-VGL reuse path.
          crowd.walkers[1]->R[1],
          crowd.walkers[2]->R[0] + Moves::PosType{-0.009, 0.012, 0.004},
          crowd.walkers[2]->R[3] + Moves::PosType{0.007, -0.006, 0.011}};
      const Moves moves(offsets, indices, positions);

      std::vector<ParticleSet::ParticleGradient> expected_gradient(walker_count);
      std::vector<ParticleSet::ParticleLaplacian> expected_laplacian(walker_count);
      std::vector<PsiFormerWF::LogValue> expected_log(walker_count);
      PsiFormerWF scalar("pf_selected_backend_scalar", files.parameters.string(),
                         files.configuration.string());
      for (std::size_t walker = 0; walker < walker_count; ++walker)
      {
        auto proposed = makeWalker(simulation_cell, walker);
        const auto selected = moves.slice(walker);
        for (std::size_t move = 0; move < selected.size(); ++move)
          proposed->R[selected.particleIndex(move)] = selected.proposedPosition(move);
        proposed->update();
        expected_gradient[walker].resize(electron_count);
        expected_laplacian[walker].resize(electron_count);
        expected_gradient[walker]  = Value(0);
        expected_laplacian[walker] = Value(0);
        expected_log[walker] = scalar.evaluateLog(
            *proposed, expected_gradient[walker], expected_laplacian[walker]);
      }

      std::vector<ParticleSet::ParticleGradient> proposed_gradient(walker_count);
      std::vector<ParticleSet::ParticleLaplacian> proposed_laplacian(walker_count);
      RefVector<ParticleSet::ParticleGradient> proposed_gradient_list;
      RefVector<ParticleSet::ParticleLaplacian> proposed_laplacian_list;
      for (std::size_t walker = 0; walker < walker_count; ++walker)
      {
        proposed_gradient[walker].resize(electron_count);
        proposed_laplacian[walker].resize(electron_count);
        proposed_gradient[walker]  = Value(0);
        proposed_laplacian[walker] = Value(0);
        proposed_gradient_list.push_back(proposed_gradient[walker]);
        proposed_laplacian_list.push_back(proposed_laplacian[walker]);
      }
      std::vector<PsiFormerWF::LogValue> log_ratios(walker_count);

      ResourceCollection resource_template("psiformer_selected_backend_template");
      crowd.leader.createResource(resource_template);
      ResourceCollection crowd_resource(resource_template);
      ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource,
                                                              crowd.wfc_list);
      crowd.leader.mw_evaluateMultiParticleMove(
          crowd.wfc_list, *crowd.p_list, moves, log_ratios,
          proposed_gradient_list, proposed_laplacian_list);

      for (std::size_t walker = 0; walker < walker_count; ++walker)
      {
        checkLog(log_ratios[walker], expected_log[walker] - accepted_log[walker]);
        for (std::size_t electron = 0; electron < electron_count; ++electron)
        {
          checkGrad(proposed_gradient[walker][electron],
                    expected_gradient[walker][electron]);
          checkValue(proposed_laplacian[walker][electron],
                     expected_laplacian[walker][electron], 3.0e-7);
        }
      }

      crowd.leader.mw_accept_rejectMultiParticleMove(
          crowd.wfc_list, *crowd.p_list, moves,
          std::vector<bool>(walker_count, false));
    }
  }
}

TEST_CASE("PsiFormer prepares exact planned crowd storage for uneven reserves",
          "[wavefunction][psiformer][multiwalker][resource][batch_memory]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  Crowd crowd(files, simulation_cell, 2, true, {0, 1});
  enableCrowdPreparationTestAccounting(crowd);

  const BatchExecutionRequirements requirements =
      makeCrowdPreparationRequirements(crowd.leader);
  const std::string participant_id = "test/psiformer/crowd-resource";
  const auto plan = makeCrowdPreparationTestPlan(
      crowd.leader, requirements, {2, 0, 0}, {3, 0, 2},
      participant_id, "crowd-resource-v1");
  bindCrowdPreparationPlan(crowd, plan, participant_id);

  ResourceCollection resource_template("psiformer_planned_template");
  crowd.leader.createResource(resource_template);
  CHECK(resource_template.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::UNPREPARED);

  // The first crowd has both initially living walkers and spare reserve.
  ResourceCollection first_resource(resource_template);
  first_resource.prepareBatchResources({plan, 0});
  CHECK(first_resource.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::PREPARED);
  CHECK(first_resource.getBatchResourcePreparationProvenance().plan == plan);
  CHECK(first_resource.getBatchResourcePreparationProvenance().crowd_index == 0);

  std::size_t prepared_storage_fingerprint = 0;
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(first_resource,
                                                            crowd.wfc_list);
    const auto diagnostics =
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            crowd.leader, crowd.wfc_list);
    checkPreparedResourceStorage(diagnostics);
    CHECK(diagnostics.participant_id == participant_id);
    CHECK(diagnostics.prepared_plan_identity == plan.get());
    CHECK(diagnostics.prepared_plan_fingerprint == plan->fingerprint());
    CHECK(diagnostics.prepared_crowd_index == 0);
    CHECK(diagnostics.initial_walker_capacity == 2);
    CHECK(diagnostics.reserve_walker_capacity == 3);
    CHECK(diagnostics.batch_workspace_identity != nullptr);
    CHECK(diagnostics.score_workspace_identity != nullptr);
    CHECK(diagnostics.kinetic_workspace_identity != nullptr);
    CHECK(diagnostics.batch_bytes > 0);
    CHECK(diagnostics.score_bytes > 0);
    CHECK(diagnostics.kinetic_bytes > 0);
    prepared_storage_fingerprint = diagnostics.prepared_storage_fingerprint;
  }

  // Publishing a new parameter vector invalidates values, not resource
  // allocation identities or capacities.
  wftrain::StructuredParameterSnapshot candidate =
      crowd.leader.snapshotParameters();
  REQUIRE_FALSE(candidate.values.empty());
  const std::size_t old_parameter_version = candidate.version;
  candidate.values.front() += 1.0e-10;
  const std::size_t new_parameter_version =
      crowd.leader.publishParameters(candidate, old_parameter_version);
  CHECK(new_parameter_version > old_parameter_version);
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(first_resource,
                                                            crowd.wfc_list);
    const auto diagnostics =
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            crowd.leader, crowd.wfc_list);
    checkPreparedResourceStorage(diagnostics);
    CHECK(diagnostics.parameter_version == new_parameter_version);
    CHECK(diagnostics.prepared_storage_fingerprint ==
          prepared_storage_fingerprint);
  }

  // A topology record with neither living nor reserve walkers remains a
  // prepared, canonically empty resource with a real batch-workspace identity.
  ResourceCollection zero_resource(resource_template);
  zero_resource.prepareBatchResources({plan, 1});
  RefVectorWithLeader<WaveFunctionComponent> empty_components(crowd.leader);
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(zero_resource,
                                                            empty_components);
    const auto diagnostics =
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            crowd.leader, empty_components);
    checkPreparedResourceStorage(diagnostics);
    CHECK(diagnostics.prepared_crowd_index == 1);
    CHECK(diagnostics.initial_walker_capacity == 0);
    CHECK(diagnostics.reserve_walker_capacity == 0);
    CHECK(diagnostics.batch_workspace_identity != nullptr);
    CHECK(diagnostics.accountedBytes() == 0);
  }

  // A crowd with no initially living walkers may later occupy its admitted
  // reserve without changing the exact prepared owner.
  ResourceCollection reserve_only_resource(resource_template);
  reserve_only_resource.prepareBatchResources({plan, 2});
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(
        reserve_only_resource, crowd.wfc_list);
    const auto diagnostics =
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            crowd.leader, crowd.wfc_list);
    checkPreparedResourceStorage(diagnostics);
    CHECK(diagnostics.prepared_crowd_index == 2);
    CHECK(diagnostics.initial_walker_capacity == 0);
    CHECK(diagnostics.reserve_walker_capacity == 2);
    CHECK(diagnostics.accountedBytes() > 0);
  }
}

TEST_CASE("PsiFormer planned resource copies require clear and support replanning",
          "[wavefunction][psiformer][multiwalker][resource][batch_memory]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  Crowd crowd(files, simulation_cell, 1, true, {0, 1});
  enableCrowdPreparationTestAccounting(crowd);

  const BatchExecutionRequirements requirements =
      makeCrowdPreparationRequirements(crowd.leader);
  const std::string participant_id = "test/psiformer/resource-replan";
  const auto first_plan = makeCrowdPreparationTestPlan(
      crowd.leader, requirements, {1}, {2}, participant_id,
      "resource-replan-v1");
  bindCrowdPreparationPlan(crowd, first_plan, participant_id);

  ResourceCollection resource_template("psiformer_replan_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection prepared(resource_template);
  prepared.prepareBatchResources({first_plan, 0});

  // A copy of prepared storage carries aggregate provenance but receives an
  // empty numeric resource clone, so neither same-plan nor different-plan
  // preparation may proceed until the explicit clear transition.
  ResourceCollection derived(prepared);
  CHECK(derived.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::DERIVED_REQUIRES_CLEAR);
  CHECK(derived.getBatchResourcePreparationProvenance().plan == first_plan);
  CHECK_THROWS_AS(derived.prepareBatchResources({first_plan, 0}),
                  std::logic_error);
  derived.prepareBatchResources({nullptr, 0});
  CHECK(derived.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::UNPREPARED);
  CHECK_FALSE(derived.getBatchResourcePreparationProvenance().plan);

  // The cleared owner resumes historical lazy behavior. Exercise it before
  // proving that retained high water must itself pass through another clear.
  clearCrowdPreparationPlan(crowd);
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(derived,
                                                            crowd.wfc_list);
    std::vector<PsiFormerWF::GradType> gradients(crowd.wfc_list.size());
    crowd.leader.mw_evalGrad(crowd.wfc_list, *crowd.p_list, 0, gradients);
    for (const PsiFormerWF::GradType& gradient : gradients)
      for (int dimension = 0; dimension < 3; ++dimension)
        CHECK(std::isfinite(std::real(gradient[dimension])));
  }

  const auto second_plan = makeCrowdPreparationTestPlan(
      crowd.leader, requirements, {1}, {2}, participant_id,
      "resource-replan-v2", {1, 1, 1, 1});
  REQUIRE(second_plan->fingerprint() != first_plan->fingerprint());
  bindCrowdPreparationPlan(crowd, second_plan, participant_id);
  CHECK_THROWS_AS(derived.prepareBatchResources({second_plan, 0}),
                  std::logic_error);
  CHECK(derived.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::UNPREPARED);

  derived.prepareBatchResources({nullptr, 0});
  derived.prepareBatchResources({second_plan, 0});
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(derived,
                                                            crowd.wfc_list);
    const auto diagnostics =
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            crowd.leader, crowd.wfc_list);
    checkPreparedResourceStorage(diagnostics);
    CHECK(diagnostics.participant_id == participant_id);
    CHECK(diagnostics.prepared_plan_identity == second_plan.get());
    CHECK(diagnostics.prepared_plan_fingerprint ==
          second_plan->fingerprint());
    CHECK(diagnostics.reserve_walker_capacity == 2);
  }
}

TEST_CASE("PsiFormer resource preparation rejects stale accounting evidence",
          "[wavefunction][psiformer][multiwalker][resource][batch_memory]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  Crowd plan_crowd(files, simulation_cell, 1, true, {0, 1});
  enableCrowdPreparationTestAccounting(plan_crowd);

  const BatchExecutionRequirements requirements =
      makeCrowdPreparationRequirements(plan_crowd.leader);
  const std::string participant_id = "test/psiformer/resource-stale";
  const auto plan = makeCrowdPreparationTestPlan(
      plan_crowd.leader, requirements, {1}, {1}, participant_id,
      "resource-stale-v1");

  // Bypass aggregate validation exactly as a direct component caller could.
  // The second same-shape crowd retains the production-incomplete claims, so
  // resource preflight must reject the otherwise compatible selected evidence.
  Crowd default_claims_crowd(files, simulation_cell, 1, true, {0, 1});
  const BatchExecutionParticipantPlan participant_plan =
      makeBatchExecutionParticipantPlan(plan, participant_id);
  for (PsiFormerWF* component : default_claims_crowd.components)
    component->bindBatchExecutionPlan(participant_plan);

  ResourceCollection resource_template("psiformer_stale_template");
  default_claims_crowd.leader.createResource(resource_template);
  ResourceCollection resource(resource_template);
  CHECK_THROWS_WITH(
      resource.prepareBatchResources({plan, 0}),
      Catch::Matchers::ContainsSubstring("accounting evidence is stale"));
  CHECK(resource.getBatchResourcePreparationProvenance().state ==
        BatchResourcePreparationState::UNPREPARED);
  CHECK_FALSE(resource.getBatchResourcePreparationProvenance().plan);
  CHECK(resource.getCursor() == 0);
  CHECK(resource.getOutstandingLoanCount() == 0);
}

TEST_CASE("PsiFormer planned acquisition failures roll back direct loans",
          "[wavefunction][psiformer][multiwalker][resource][batch_memory]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  Crowd crowd(files, simulation_cell, 2, true, {0, 1});
  enableCrowdPreparationTestAccounting(crowd);

  const BatchExecutionRequirements requirements =
      makeCrowdPreparationRequirements(crowd.leader);
  const std::string participant_id = "test/psiformer/resource-rollback";
  const auto plan = makeCrowdPreparationTestPlan(
      crowd.leader, requirements, {1}, {1}, participant_id,
      "resource-rollback-v1");
  bindCrowdPreparationPlan(crowd, plan, participant_id);

  ResourceCollection resource_template("psiformer_rollback_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection resource(resource_template);
  resource.prepareBatchResources({plan, 0});

  RefVectorWithLeader<WaveFunctionComponent> singleton(crowd.leader);
  singleton.push_back(crowd.leader);
  auto check_successful_reuse = [&]() {
    resource.rewind();
    crowd.leader.acquireResource(resource, singleton);
    CHECK(resource.getCursor() == 1);
    CHECK(resource.getOutstandingLoanCount() == 1);
    resource.rewind();
    crowd.leader.releaseResource(resource, singleton);
    CHECK(resource.getCursor() == 1);
    CHECK(resource.getOutstandingLoanCount() == 0);
    resource.rewind();
  };

  // Reserve overflow happens after lending; acquisition must return the exact
  // candidate before propagating the failure.
  resource.rewind();
  CHECK_THROWS_AS(crowd.leader.acquireResource(resource, crowd.wfc_list),
                  std::length_error);
  CHECK(resource.getCursor() == 0);
  CHECK(resource.getOutstandingLoanCount() == 0);
  check_successful_reuse();

  Crowd other_model(files, simulation_cell, 1, true, {0, 1});
  enableCrowdPreparationTestAccounting(other_model);
  const auto other_plan = makeCrowdPreparationTestPlan(
      other_model.leader, requirements, {1}, {1}, participant_id,
      "resource-rollback-v2");
  bindCrowdPreparationPlan(other_model, other_plan, participant_id);

  // Aggregate plan mismatch is rejected before lending.
  resource.rewind();
  CHECK_THROWS_AS(
      other_model.leader.acquireResource(resource, other_model.wfc_list),
      std::logic_error);
  CHECK(resource.getCursor() == 0);
  CHECK(resource.getOutstandingLoanCount() == 0);
  check_successful_reuse();

  // With matching aggregate provenance, the distinct shared model is rejected
  // after lending and exercises the same takeback rollback path.
  clearCrowdPreparationPlan(other_model);
  bindCrowdPreparationPlan(other_model, plan, participant_id);
  resource.rewind();
  CHECK_THROWS_AS(
      other_model.leader.acquireResource(resource, other_model.wfc_list),
      std::logic_error);
  CHECK(resource.getCursor() == 0);
  CHECK(resource.getOutstandingLoanCount() == 0);
  check_successful_reuse();
}

TEST_CASE("PsiFormer resource mismatch leaves both crowds immediately reusable",
          "[wavefunction][psiformer][multiwalker][resource][threading]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  Crowd crowd_a(files, simulation_cell, 1);
  Crowd crowd_b(files, simulation_cell, 1);

  ResourceCollection template_a("psiformer_model_a_template");
  ResourceCollection template_b("psiformer_model_b_template");
  crowd_a.leader.createResource(template_a);
  crowd_b.leader.createResource(template_b);
  ResourceCollection resource_a(template_a);
  ResourceCollection resource_b(template_b);

  // A same-typed resource from a distinct shared model must be rejected without
  // publishing a leader handle or consuming the collection cursor.
  CHECK_THROWS_AS(ResourceCollectionTeamLock<WaveFunctionComponent>(resource_a, crowd_b.wfc_list),
                  std::logic_error);

  auto evaluate_one = [](Crowd& crowd, ResourceCollection& resource) {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(resource, crowd.wfc_list);
    std::vector<PsiFormerWF::GradType> gradients(crowd.wfc_list.size());
    crowd.leader.mw_evalGrad(crowd.wfc_list, *crowd.p_list, 0, gradients);
    for (const auto& gradient : gradients)
      for (int dimension = 0; dimension < 3; ++dimension)
        CHECK(std::isfinite(std::real(gradient[dimension])));
  };

  // Both the rejected leader and the mismatched collection remain usable.
  evaluate_one(crowd_b, resource_b);
  evaluate_one(crowd_a, resource_a);

  // Heterogeneous component lists fail before lending any resource.
  RefVectorWithLeader<WaveFunctionComponent> mixed_components(crowd_a.leader);
  mixed_components.push_back(crowd_a.leader);
  mixed_components.push_back(crowd_b.leader);
  CHECK_THROWS_AS(ResourceCollectionTeamLock<WaveFunctionComponent>(resource_a, mixed_components),
                  std::invalid_argument);
  evaluate_one(crowd_a, resource_a);
}

TEST_CASE("PsiFormer planned FULL_VGL matches the legacy oracle and refreshes caches",
          "[wavefunction][psiformer][multiwalker][full_vgl][batch_memory]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  constexpr std::size_t walker_count = 2;
  Crowd planned(files, simulation_cell, walker_count, true, {0, 1});
  ScopedEnvironmentVariable oracle_backend("PSIFORMER_SPATIAL_BACKEND",
                                            "oracle");
  Crowd oracle(files, simulation_cell, walker_count, true, {0, 1});
  enableCrowdPreparationTestAccounting(planned);

  const BatchExecutionRequirements requirements =
      makeCrowdPreparationRequirements(planned.leader);
  const std::string participant_id = "test/psiformer/planned-full-vgl";
  const auto plan = makeCrowdPreparationTestPlan(
      planned.leader, requirements, {walker_count}, {3}, participant_id,
      "planned-full-vgl-v1");
  bindCrowdPreparationPlan(planned, plan, participant_id);
  prepareCrowdPreparationClones(planned, plan, participant_id);

  ResourceCollection planned_template("psiformer_planned_full_vgl_template");
  planned.leader.createResource(planned_template);
  ResourceCollection planned_resource(planned_template);
  planned_resource.prepareBatchResources({plan, 0});

  ResourceCollection oracle_template("psiformer_legacy_full_vgl_template");
  oracle.leader.createResource(oracle_template);
  ResourceCollection oracle_resource(oracle_template);

  std::vector<testing::PsiFormerPreparedCloneStorage> clone_storage_before;
  testing::PsiFormerCrowdWorkspaceDiagnostics resource_before;
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> planned_lock(
        planned_resource, planned.wfc_list);
    ResourceCollectionTeamLock<WaveFunctionComponent> oracle_lock(
        oracle_resource, oracle.wfc_list);

    const std::size_t electrons = planned.walkers.front()->getTotalNum();
    std::vector<ParticleSet::ParticleGradient> planned_gradients(walker_count);
    std::vector<ParticleSet::ParticleLaplacian> planned_laplacians(walker_count);
    std::vector<ParticleSet::ParticleGradient> oracle_gradients(walker_count);
    std::vector<ParticleSet::ParticleLaplacian> oracle_laplacians(walker_count);
    RefVector<ParticleSet::ParticleGradient> planned_gradient_list;
    RefVector<ParticleSet::ParticleLaplacian> planned_laplacian_list;
    RefVector<ParticleSet::ParticleGradient> oracle_gradient_list;
    RefVector<ParticleSet::ParticleLaplacian> oracle_laplacian_list;
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      planned_gradients[lane].resize(electrons);
      planned_laplacians[lane].resize(electrons);
      oracle_gradients[lane].resize(electrons);
      oracle_laplacians[lane].resize(electrons);
      planned_gradient_list.push_back(planned_gradients[lane]);
      planned_laplacian_list.push_back(planned_laplacians[lane]);
      oracle_gradient_list.push_back(oracle_gradients[lane]);
      oracle_laplacian_list.push_back(oracle_laplacians[lane]);
    }

    clone_storage_before.reserve(walker_count);
    for (const PsiFormerWF* component : planned.components)
    {
      clone_storage_before.push_back(
          testing::TestPsiFormerVirtualBatch::preparedCloneStorage(*component));
      REQUIRE(clone_storage_before.back().exact_marker);
    }
    resource_before =
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            planned.leader, planned.wfc_list);
    checkPreparedResourceStorage(resource_before);
    REQUIRE(resource_before.initial_walker_capacity == walker_count);
    REQUIRE(resource_before.reserve_walker_capacity == 3);

    auto reset_outputs = [&]() {
      for (std::size_t lane = 0; lane < walker_count; ++lane)
        for (std::size_t electron = 0; electron < electrons; ++electron)
        {
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
          {
            const Value seed = makeWeight(
                0.017 * static_cast<double>((lane + 1) * (electron + 1) *
                                           (dimension + 1)),
                -0.003 * static_cast<double>((lane + 1) * (dimension + 1)));
            planned_gradients[lane][electron][dimension] = seed;
            oracle_gradients[lane][electron][dimension] = seed;
          }
          const Value seed = makeWeight(
              -0.029 * static_cast<double>((lane + 1) * (electron + 1)),
              0.004 * static_cast<double>(electron + 1));
          planned_laplacians[lane][electron] = seed;
          oracle_laplacians[lane][electron] = seed;
        }
    };

    auto evaluate_and_compare = [&](bool through_gl, bool from_scratch = true) {
      reset_outputs();
      if (through_gl)
      {
        oracle.leader.mw_evaluateGL(oracle.wfc_list, *oracle.p_list,
                                    oracle_gradient_list,
                                    oracle_laplacian_list, from_scratch);
        planned.leader.mw_evaluateGL(planned.wfc_list, *planned.p_list,
                                     planned_gradient_list,
                                     planned_laplacian_list, from_scratch);
      }
      else
      {
        oracle.leader.mw_evaluateLog(oracle.wfc_list, *oracle.p_list,
                                     oracle_gradient_list,
                                     oracle_laplacian_list);
        planned.leader.mw_evaluateLog(planned.wfc_list, *planned.p_list,
                                      planned_gradient_list,
                                      planned_laplacian_list);
      }

      for (std::size_t lane = 0; lane < walker_count; ++lane)
      {
        checkLog(planned.components[lane]->get_log_value(),
                 oracle.components[lane]->get_log_value());
        CHECK(testing::TestPsiFormerVirtualBatch::hasCurrentFullAcceptedState(
            *planned.components[lane], *planned.walkers[lane]));
        for (std::size_t electron = 0; electron < electrons; ++electron)
        {
          checkGrad(planned_gradients[lane][electron],
                    oracle_gradients[lane][electron]);
          checkValue(planned_laplacians[lane][electron],
                     oracle_laplacians[lane][electron], 3.0e-7);
        }
      }
    };

    // Invalid accepted caches are legal inputs: a successful FULL_VGL replaces
    // them with current complete spatial state.
    for (PsiFormerWF* component : planned.components)
      testing::TestPsiFormerVirtualBatch::invalidateAcceptedState(*component);
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      CHECK_FALSE(testing::TestPsiFormerVirtualBatch::hasCurrentFullAcceptedState(
          *planned.components[lane], *planned.walkers[lane]));
    evaluate_and_compare(false);

    // A new accepted configuration must replace the now-different cached one.
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      const ParticleSet::RealType delta =
          0.002 * static_cast<ParticleSet::RealType>(lane + 1);
      planned.walkers[lane]->R[1][lane] += delta;
      oracle.walkers[lane]->R[1][lane] += delta;
      planned.walkers[lane]->update();
      oracle.walkers[lane]->update();
      CHECK_FALSE(testing::TestPsiFormerVirtualBatch::hasCurrentFullAcceptedState(
          *planned.components[lane], *planned.walkers[lane]));
    }
    evaluate_and_compare(false);

    // Publishing on each leader leaves a mixture of explicitly invalid leader
    // state and stale clone state.  mw_evaluateGL delegates to the same planned
    // FULL transaction and refreshes every lane without a scalar synchronize.
    auto publish_delta = [](PsiFormerWF& component) {
      wftrain::StructuredParameterSnapshot candidate =
          component.snapshotParameters();
      REQUIRE_FALSE(candidate.values.empty());
      candidate.values.front() += 1.0e-6;
      return component.publishParameters(candidate, candidate.version);
    };
    const std::size_t planned_version = publish_delta(planned.leader);
    const std::size_t oracle_version = publish_delta(oracle.leader);
    REQUIRE(planned_version == oracle_version);
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      CHECK_FALSE(testing::TestPsiFormerVirtualBatch::hasCurrentFullAcceptedState(
          *planned.components[lane], *planned.walkers[lane]));
    evaluate_and_compare(true);
    evaluate_and_compare(true, false);

    // Exact same-lane ParticleSet G/L destinations are the one intentional
    // output alias.  They retain normal additive semantics in both branches.
    RefVector<ParticleSet::ParticleGradient> planned_particle_gradients;
    RefVector<ParticleSet::ParticleLaplacian> planned_particle_laplacians;
    RefVector<ParticleSet::ParticleGradient> oracle_particle_gradients;
    RefVector<ParticleSet::ParticleLaplacian> oracle_particle_laplacians;
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      for (std::size_t electron = 0; electron < electrons; ++electron)
      {
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
        {
          const Value seed = makeWeight(
              0.011 * static_cast<double>((lane + 1) * (dimension + 1)),
              0.002 * static_cast<double>(electron + 1));
          planned.walkers[lane]->G[electron][dimension] = seed;
          oracle.walkers[lane]->G[electron][dimension] = seed;
        }
        const Value seed = makeWeight(-0.013 * static_cast<double>(lane + 1),
                                      -0.001 * static_cast<double>(electron + 1));
        planned.walkers[lane]->L[electron] = seed;
        oracle.walkers[lane]->L[electron] = seed;
      }
      planned_particle_gradients.push_back(planned.walkers[lane]->G);
      planned_particle_laplacians.push_back(planned.walkers[lane]->L);
      oracle_particle_gradients.push_back(oracle.walkers[lane]->G);
      oracle_particle_laplacians.push_back(oracle.walkers[lane]->L);
    }
    oracle.leader.mw_evaluateLog(oracle.wfc_list, *oracle.p_list,
                                 oracle_particle_gradients,
                                 oracle_particle_laplacians);
    planned.leader.mw_evaluateLog(planned.wfc_list, *planned.p_list,
                                  planned_particle_gradients,
                                  planned_particle_laplacians);
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      checkLog(planned.components[lane]->get_log_value(),
               oracle.components[lane]->get_log_value());
      for (std::size_t electron = 0; electron < electrons; ++electron)
      {
        checkGrad(planned.walkers[lane]->G[electron],
                  oracle.walkers[lane]->G[electron]);
        checkValue(planned.walkers[lane]->L[electron],
                   oracle.walkers[lane]->L[electron], 3.0e-7);
      }
    }

    for (std::size_t lane = 0; lane < walker_count; ++lane)
      checkPreparedCloneStorageUnchanged(
          testing::TestPsiFormerVirtualBatch::preparedCloneStorage(
              *planned.components[lane]),
          clone_storage_before[lane]);
    checkPreparedResourceStorageUnchanged(
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            planned.leader, planned.wfc_list),
        resource_before);
  }

  CHECK(planned_resource.getOutstandingLoanCount() == 0);
  CHECK(oracle_resource.getOutstandingLoanCount() == 0);
  planned_resource.rewind();
  CHECK(planned_resource.getCursor() == 0);
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> planned_lock(
        planned_resource, planned.wfc_list);
    checkPreparedResourceStorageUnchanged(
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            planned.leader, planned.wfc_list),
        resource_before);
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      checkPreparedCloneStorageUnchanged(
          testing::TestPsiFormerVirtualBatch::preparedCloneStorage(
              *planned.components[lane]),
          clone_storage_before[lane]);

    const std::size_t electrons = planned.walkers.front()->getTotalNum();
    std::vector<ParticleSet::ParticleGradient> retry_gradients(walker_count);
    std::vector<ParticleSet::ParticleLaplacian> retry_laplacians(walker_count);
    RefVector<ParticleSet::ParticleGradient> retry_gradient_list;
    RefVector<ParticleSet::ParticleLaplacian> retry_laplacian_list;
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      retry_gradients[lane].resize(electrons);
      retry_laplacians[lane].resize(electrons);
      retry_gradients[lane] = Value(0);
      retry_laplacians[lane] = Value(0);
      retry_gradient_list.push_back(retry_gradients[lane]);
      retry_laplacian_list.push_back(retry_laplacians[lane]);
    }
    planned.leader.mw_evaluateLog(planned.wfc_list, *planned.p_list,
                                  retry_gradient_list, retry_laplacian_list);
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      CHECK(testing::TestPsiFormerVirtualBatch::hasCurrentFullAcceptedState(
          *planned.components[lane], *planned.walkers[lane]));
      checkPreparedCloneStorageUnchanged(
          testing::TestPsiFormerVirtualBatch::preparedCloneStorage(
              *planned.components[lane]),
          clone_storage_before[lane]);
    }
    checkPreparedResourceStorageUnchanged(
        testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
            planned.leader, planned.wfc_list),
        resource_before);
  }

  CHECK(planned_resource.getOutstandingLoanCount() == 0);
}

TEST_CASE("PsiFormer planned FULL_VGL rejects bad destinations atomically",
          "[wavefunction][psiformer][multiwalker][full_vgl][atomic]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  constexpr std::size_t walker_count = 2;
  Crowd crowd(files, simulation_cell, walker_count, true, {0, 1});
  enableCrowdPreparationTestAccounting(crowd);

  const BatchExecutionRequirements requirements =
      makeCrowdPreparationRequirements(crowd.leader);
  const std::string participant_id = "test/psiformer/planned-full-atomic";
  const auto plan = makeCrowdPreparationTestPlan(
      crowd.leader, requirements, {walker_count}, {3}, participant_id,
      "planned-full-atomic-v1");
  bindCrowdPreparationPlan(crowd, plan, participant_id);
  prepareCrowdPreparationClones(crowd, plan, participant_id);

  ResourceCollection resource_template("psiformer_planned_full_atomic_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection resource(resource_template);
  resource.prepareBatchResources({plan, 0});
  ResourceCollectionTeamLock<WaveFunctionComponent> lock(resource,
                                                          crowd.wfc_list);

  const std::size_t electrons = crowd.walkers.front()->getTotalNum();
  std::vector<ParticleSet::ParticleGradient> gradients(walker_count);
  std::vector<ParticleSet::ParticleLaplacian> laplacians(walker_count);
  RefVector<ParticleSet::ParticleGradient> gradient_list;
  RefVector<ParticleSet::ParticleLaplacian> laplacian_list;
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    gradients[lane].resize(electrons);
    laplacians[lane].resize(electrons);
    gradient_list.push_back(gradients[lane]);
    laplacian_list.push_back(laplacians[lane]);
  }

  auto reset_outputs = [&]() {
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      for (std::size_t electron = 0; electron < electrons; ++electron)
      {
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          gradients[lane][electron][dimension] = makeWeight(
              0.01 * static_cast<double>((lane + 1) * (dimension + 1)),
              -0.002 * static_cast<double>(electron + 1));
        laplacians[lane][electron] = makeWeight(
            -0.02 * static_cast<double>((lane + 1) * (electron + 1)),
            0.003 * static_cast<double>(lane + 1));
      }
  };
  auto retry = [&]() {
    reset_outputs();
    crowd.leader.mw_evaluateLog(crowd.wfc_list, *crowd.p_list,
                                gradient_list, laplacian_list);
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      CHECK(testing::TestPsiFormerVirtualBatch::hasCurrentFullAcceptedState(
          *crowd.components[lane], *crowd.walkers[lane]));
  };

  std::vector<testing::PsiFormerPreparedCloneStorage> clone_storage_before;
  clone_storage_before.reserve(walker_count);
  for (const PsiFormerWF* component : crowd.components)
    clone_storage_before.push_back(
        testing::TestPsiFormerVirtualBatch::preparedCloneStorage(*component));
  const auto resource_before =
      testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
          crowd.leader, crowd.wfc_list);
  REQUIRE(resource_before.reserve_walker_capacity == 3);
  const std::vector<Value> caller_sentinel{
      makeWeight(7.0, -0.5), makeWeight(-3.0, 0.25)};
  retry();

  gradients.back().resize(electrons - 1);
  const PlannedFullVGLSnapshot malformed =
      capturePlannedFullVGLSnapshot(crowd, gradients, laplacians);
  const RuntimePreflightSnapshot malformed_runtime =
      captureRuntimePreflightState(crowd, resource, caller_sentinel);
  CHECK_THROWS_AS(crowd.leader.mw_evaluateLog(
                      crowd.wfc_list, *crowd.p_list, gradient_list,
                      laplacian_list),
                  std::invalid_argument);
  checkPlannedFullVGLSnapshot(crowd, gradients, laplacians, malformed);
  checkRuntimePreflightState(crowd, resource, caller_sentinel,
                             malformed_runtime);
  gradients.back().resize(electrons);
  retry();

  gradients.back()[electrons - 1][2] =
      makeWeight(std::numeric_limits<double>::quiet_NaN(), 0.0);
  const PlannedFullVGLSnapshot nonfinite =
      capturePlannedFullVGLSnapshot(crowd, gradients, laplacians);
  const RuntimePreflightSnapshot nonfinite_runtime =
      captureRuntimePreflightState(crowd, resource, caller_sentinel);
  CHECK_THROWS_AS(crowd.leader.mw_evaluateLog(
                      crowd.wfc_list, *crowd.p_list, gradient_list,
                      laplacian_list),
                  std::invalid_argument);
  checkPlannedFullVGLSnapshot(crowd, gradients, laplacians, nonfinite);
  checkRuntimePreflightState(crowd, resource, caller_sentinel,
                             nonfinite_runtime);
  retry();

  RefVector<ParticleSet::ParticleGradient> aliased_gradient_list;
  aliased_gradient_list.push_back(gradients.front());
  aliased_gradient_list.push_back(gradients.front());
  const PlannedFullVGLSnapshot aliased =
      capturePlannedFullVGLSnapshot(crowd, gradients, laplacians);
  const RuntimePreflightSnapshot aliased_runtime =
      captureRuntimePreflightState(crowd, resource, caller_sentinel);
  CHECK_THROWS_AS(crowd.leader.mw_evaluateLog(
                      crowd.wfc_list, *crowd.p_list,
                      aliased_gradient_list, laplacian_list),
                  std::invalid_argument);
  checkPlannedFullVGLSnapshot(crowd, gradients, laplacians, aliased);
  checkRuntimePreflightState(crowd, resource, caller_sentinel,
                             aliased_runtime);
  retry();

  // Real builds can attach a Laplacian destination directly to the distinct
  // SoA coordinate allocation. Reject it before publication so a successful
  // call can never leave ParticleSet's AoS and SoA positions inconsistent.
#ifndef QMC_COMPLEX
  {
    const auto& soa_positions =
        crowd.walkers.front()->getCoordinates().getAllParticlePos();
    ParticleSet::ParticleLaplacian soa_laplacian;
    soa_laplacian.attachReference(
        const_cast<ParticleSet::RealType*>(soa_positions.data()), electrons);
    RefVector<ParticleSet::ParticleLaplacian> soa_laplacian_list;
    soa_laplacian_list.push_back(soa_laplacian);
    soa_laplacian_list.push_back(laplacians.back());
    const RuntimePreflightSnapshot soa_runtime =
        captureRuntimePreflightState(crowd, resource, caller_sentinel);

    CHECK_THROWS_AS(crowd.leader.mw_evaluateLog(
                        crowd.wfc_list, *crowd.p_list, gradient_list,
                        soa_laplacian_list),
                    std::invalid_argument);
    checkRuntimePreflightState(crowd, resource, caller_sentinel, soa_runtime);
    retry();
  }
#endif

  // Force the latest throwing boundary after direct evaluation and all
  // identity revalidation. No accepted cache or caller destination may have
  // been published when this injected failure is observed.
  reset_outputs();
  testing::TestPsiFormerVirtualBatch::injectPlannedFullVGLPrepublicationFailure(
      crowd.leader, true);
  const PlannedFullVGLSnapshot late_failure =
      capturePlannedFullVGLSnapshot(crowd, gradients, laplacians);
  const RuntimePreflightSnapshot late_failure_runtime =
      captureRuntimePreflightState(crowd, resource, caller_sentinel);
  CHECK_THROWS_AS(crowd.leader.mw_evaluateLog(
                      crowd.wfc_list, *crowd.p_list, gradient_list,
                      laplacian_list),
                  std::overflow_error);
  checkPlannedFullVGLSnapshot(crowd, gradients, laplacians, late_failure);
  checkRuntimePreflightState(crowd, resource, caller_sentinel,
                             late_failure_runtime);
  testing::TestPsiFormerVirtualBatch::injectPlannedFullVGLPrepublicationFailure(
      crowd.leader, false);
  retry();

  for (std::size_t lane = 0; lane < walker_count; ++lane)
    checkPreparedCloneStorageUnchanged(
        testing::TestPsiFormerVirtualBatch::preparedCloneStorage(
            *crowd.components[lane]),
        clone_storage_before[lane]);
  checkPreparedResourceStorageUnchanged(
      testing::TestPsiFormerVirtualBatch::crowdWorkspaceDiagnostics(
          crowd.leader, crowd.wfc_list),
      resource_before);
}

TEST_CASE("PsiFormer planned runtime preflight is exact and read only",
          "[wavefunction][psiformer][multiwalker][batch_memory][preflight]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  Crowd crowd(files, simulation_cell, 2, true, {0, 1});
  enableCrowdPreparationTestAccounting(crowd);

  const BatchExecutionRequirements requirements =
      makeCrowdPreparationRequirements(crowd.leader);
  const std::string participant_id = "test/psiformer/runtime-preflight";
  const auto plan = makeCrowdPreparationTestPlan(
      crowd.leader, requirements, {2}, {2}, participant_id,
      "runtime-preflight-v1");
  bindCrowdPreparationPlan(crowd, plan, participant_id);
  prepareCrowdPreparationClones(crowd, plan, participant_id);

  ResourceCollection resource_template("psiformer_runtime_preflight_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection resource(resource_template);
  resource.prepareBatchResources({plan, 0});
  ResourceCollectionTeamLock<WaveFunctionComponent> lock(resource, crowd.wfc_list);

  using Probe = testing::TestPsiFormerVirtualBatch;
  Probe::RuntimeRequest full_request;
  full_request.operation = Probe::RuntimeOperation::FULL_VGL;
  full_request.live_walkers = 2;
  full_request.dense_configurations = 2;
  std::vector<Value> caller_output{Value(7.0), Value(-3.0)};
  std::size_t preflight_invocation = 0;

  auto require_unchanged = [&](const auto& wfc_list, const auto& p_list,
                               const Probe::RuntimeRequest& request,
                               bool should_throw) {
    const std::size_t invocation = preflight_invocation++;
    CAPTURE(invocation);
    CAPTURE(static_cast<int>(request.operation), request.live_walkers,
            request.dense_configurations, request.sparse_references,
            request.sparse_replacements, request.selected_parameters,
            request.derivative_width, request.active_electron.has_value(),
            request.descriptor_fingerprint.has_value());
    const RuntimePreflightSnapshot before = captureRuntimePreflightState(
        crowd, resource, caller_output);
    if (should_throw)
      CHECK_THROWS(Probe::requirePlannedRuntime(
          crowd.leader, wfc_list, p_list, request));
    else
      CHECK(Probe::requirePlannedRuntime(
                crowd.leader, wfc_list, p_list, request) ==
            before.resource.current_storage_fingerprint);
    checkRuntimePreflightState(crowd, resource, caller_output, before);
  };

  // A complete VGL crowd and masked value recomputation both validate without
  // opening a model transaction or changing any caller/component storage.
  require_unchanged(crowd.wfc_list, *crowd.p_list, full_request, false);
  Probe::RuntimeRequest masked_recompute;
  masked_recompute.operation = Probe::RuntimeOperation::RECOMPUTE_VALUE;
  masked_recompute.live_walkers = 2;
  masked_recompute.dense_configurations = 1;
  require_unchanged(crowd.wfc_list, *crowd.p_list, masked_recompute, false);

  Probe::RuntimeRequest malformed = full_request;
  malformed.live_walkers = 1;
  require_unchanged(crowd.wfc_list, *crowd.p_list, malformed, true);

  malformed = full_request;
  malformed.dense_configurations = 1;
  require_unchanged(crowd.wfc_list, *crowd.p_list, malformed, true);

  malformed = full_request;
  malformed.sparse_references = 1;
  require_unchanged(crowd.wfc_list, *crowd.p_list, malformed, true);

  malformed = full_request;
  malformed.selected_parameters = 1;
  require_unchanged(crowd.wfc_list, *crowd.p_list, malformed, true);

  malformed = full_request;
  malformed.active_electron = 0;
  require_unchanged(crowd.wfc_list, *crowd.p_list, malformed, true);

  malformed = full_request;
  malformed.descriptor_fingerprint = UINT64_C(17);
  require_unchanged(crowd.wfc_list, *crowd.p_list, malformed, true);

  // Duplicate and reordered lanes fail before resource or proposal state can change.
  RefVectorWithLeader<WaveFunctionComponent> duplicate_components(crowd.leader);
  duplicate_components.push_back(crowd.leader);
  duplicate_components.push_back(crowd.leader);
  require_unchanged(duplicate_components, *crowd.p_list, full_request, true);

  RefVectorWithLeader<ParticleSet> duplicate_particles(*crowd.walkers[0]);
  duplicate_particles.push_back(*crowd.walkers[0]);
  duplicate_particles.push_back(*crowd.walkers[0]);
  require_unchanged(crowd.wfc_list, duplicate_particles, full_request, true);

  RefVectorWithLeader<ParticleSet> reordered_particles(*crowd.walkers[1]);
  reordered_particles.push_back(*crowd.walkers[1]);
  reordered_particles.push_back(*crowd.walkers[0]);
  require_unchanged(crowd.wfc_list, reordered_particles, full_request, true);

  Crowd foreign_crowd(files, simulation_cell, 1, true, {0, 1});
  RefVectorWithLeader<WaveFunctionComponent> mixed_components(crowd.leader);
  mixed_components.push_back(crowd.leader);
  mixed_components.push_back(foreign_crowd.leader);
  RefVectorWithLeader<ParticleSet> mixed_particles(*crowd.walkers[0]);
  mixed_particles.push_back(*crowd.walkers[0]);
  mixed_particles.push_back(*foreign_crowd.walkers[0]);
  require_unchanged(mixed_components, mixed_particles, full_request, true);

  Probe::useOptimizationMetadata(*crowd.components[1], foreign_crowd.leader);
  require_unchanged(crowd.wfc_list, *crowd.p_list, full_request, true);
  Probe::useOptimizationMetadata(*crowd.components[1], crowd.leader);

  // Target-coordinate, coordinate-value, and proposal-absence failures are
  // similarly nonmutating, including clone-local observed parameter versions.
  crowd.walkers[1]->setSpinor(true);
  require_unchanged(crowd.wfc_list, *crowd.p_list, full_request, true);
  crowd.walkers[1]->setSpinor(false);

  const int saved_group = crowd.walkers[1]->GroupID[0];
  crowd.walkers[1]->GroupID[0] = 1;
  require_unchanged(crowd.wfc_list, *crowd.p_list, full_request, true);
  crowd.walkers[1]->GroupID[0] = saved_group;

  const auto saved_finite_position = crowd.walkers[1]->R[0][0];
  crowd.walkers[1]->R[0][0] += 0.125;
  require_unchanged(crowd.wfc_list, *crowd.p_list, full_request, true);
  crowd.walkers[1]->R[0][0] = saved_finite_position;

  const auto saved_position = crowd.walkers[1]->R[0][0];
  crowd.walkers[1]->R[0][0] = std::numeric_limits<ParticleSet::RealType>::quiet_NaN();
  require_unchanged(crowd.wfc_list, *crowd.p_list, full_request, true);
  crowd.walkers[1]->R[0][0] = saved_position;

  Probe::installSingleProposal(crowd.leader, 0);
  require_unchanged(crowd.wfc_list, *crowd.p_list, full_request, true);
  Probe::clearProposal(crowd.leader);

  // A one-electron operation additionally requires the matching live
  // ParticleSet proposal, not merely an in-range electron index.
  Probe::RuntimeRequest ratio_request;
  ratio_request.operation = Probe::RuntimeOperation::RATIO_GRADIENT;
  ratio_request.live_walkers = 2;
  ratio_request.dense_configurations = 2;
  ratio_request.active_electron = 0;
  require_unchanged(crowd.wfc_list, *crowd.p_list, ratio_request, true);

  for (auto& walker : crowd.walkers)
    walker->makeMove(0, ParticleSet::PosType(0.002, -0.001, 0.003));
  require_unchanged(crowd.wfc_list, *crowd.p_list, ratio_request, false);
  for (auto& walker : crowd.walkers)
    walker->rejectMove(0);

  for (auto& walker : crowd.walkers)
    walker->makeMove(
        0, ParticleSet::PosType(
               std::numeric_limits<ParticleSet::RealType>::quiet_NaN(), 0.0, 0.0));
  require_unchanged(crowd.wfc_list, *crowd.p_list, ratio_request, true);
  for (auto& walker : crowd.walkers)
    walker->rejectMove(0);

  // Runtime preflight rechecks the exact post-acquire collection cursor.
  resource.rewind(0);
  require_unchanged(crowd.wfc_list, *crowd.p_list, full_request, true);
  resource.rewind(1);
}

TEST_CASE("PsiFormer planned runtime rejects unprepared clones before mutation",
          "[wavefunction][psiformer][multiwalker][batch_memory][preflight]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  Crowd crowd(files, simulation_cell, 2, true, {0, 1});
  enableCrowdPreparationTestAccounting(crowd);

  const BatchExecutionRequirements requirements =
      makeCrowdPreparationRequirements(crowd.leader);
  const std::string participant_id = "test/psiformer/unprepared-runtime";
  const auto plan = makeCrowdPreparationTestPlan(
      crowd.leader, requirements, {2}, {2}, participant_id,
      "unprepared-runtime-v1");
  bindCrowdPreparationPlan(crowd, plan, participant_id);
  const BatchExecutionParticipantPlan participant_plan =
      makeBatchExecutionParticipantPlan(plan, participant_id);
  crowd.leader.prepareBatchExecutionClone(participant_plan);

  ResourceCollection resource_template("psiformer_unprepared_runtime_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection resource(resource_template);
  resource.prepareBatchResources({plan, 0});
  ResourceCollectionTeamLock<WaveFunctionComponent> lock(resource, crowd.wfc_list);

  testing::TestPsiFormerVirtualBatch::RuntimeRequest request;
  request.operation = testing::TestPsiFormerVirtualBatch::RuntimeOperation::FULL_VGL;
  request.live_walkers = 2;
  request.dense_configurations = 2;
  std::vector<Value> caller_output{Value(11.0)};
  const RuntimePreflightSnapshot before = captureRuntimePreflightState(
      crowd, resource, caller_output);
  CHECK_THROWS(testing::TestPsiFormerVirtualBatch::requirePlannedRuntime(
      crowd.leader, crowd.wfc_list, *crowd.p_list, request));
  checkRuntimePreflightState(crowd, resource, caller_output, before);
}

TEST_CASE("PsiFormer planned runtime retains exact acquired lane order",
          "[wavefunction][psiformer][multiwalker][batch_memory][preflight]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  Crowd crowd(files, simulation_cell, 3, true, {0, 1});
  enableCrowdPreparationTestAccounting(crowd);

  const BatchExecutionRequirements requirements =
      makeCrowdPreparationRequirements(crowd.leader);
  const std::string participant_id = "test/psiformer/acquired-lane-order";
  const auto plan = makeCrowdPreparationTestPlan(
      crowd.leader, requirements, {3}, {3}, participant_id,
      "acquired-lane-order-v1");
  bindCrowdPreparationPlan(crowd, plan, participant_id);
  prepareCrowdPreparationClones(crowd, plan, participant_id);

  ResourceCollection resource_template("psiformer_acquired_lane_template");
  crowd.leader.createResource(resource_template);
  ResourceCollection resource(resource_template);
  resource.prepareBatchResources({plan, 0});
  resource.rewind();
  crowd.leader.acquireResource(resource, crowd.wfc_list);

  using Probe = testing::TestPsiFormerVirtualBatch;
  Probe::RuntimeRequest request;
  request.operation = Probe::RuntimeOperation::FULL_VGL;
  request.live_walkers = 3;
  request.dense_configurations = 3;
  const std::size_t storage_fingerprint = Probe::requirePlannedRuntime(
      crowd.leader, crowd.wfc_list, *crowd.p_list, request);

  // Permute components and their correctly bound ParticleSets together.  All
  // pairwise identities and shapes remain valid, so only the exact ephemeral
  // acquisition-lane markers can reject this otherwise plausible crowd.
  RefVectorWithLeader<WaveFunctionComponent> permuted_components(crowd.leader);
  permuted_components.push_back(crowd.leader);
  permuted_components.push_back(*crowd.components[2]);
  permuted_components.push_back(*crowd.components[1]);
  RefVectorWithLeader<ParticleSet> permuted_particles(*crowd.walkers[0]);
  permuted_particles.push_back(*crowd.walkers[0]);
  permuted_particles.push_back(*crowd.walkers[2]);
  permuted_particles.push_back(*crowd.walkers[1]);

  std::vector<Value> caller_output{Value(19.0)};
  const RuntimePreflightSnapshot before = captureRuntimePreflightState(
      crowd, resource, caller_output);
  CHECK_THROWS(Probe::requirePlannedRuntime(
      crowd.leader, permuted_components, permuted_particles, request));
  checkRuntimePreflightState(crowd, resource, caller_output, before);

  // A wrong-order release fails before takeback or marker clearing.  Restoring
  // the collection cursor makes the original crowd immediately usable again.
  resource.rewind(0);
  CHECK_THROWS(crowd.leader.releaseResource(resource, permuted_components));
  CHECK(resource.getCursor() == 0);
  CHECK(resource.getOutstandingLoanCount() == 1);
  resource.rewind(1);
  CHECK(Probe::requirePlannedRuntime(
            crowd.leader, crowd.wfc_list, *crowd.p_list, request) ==
        storage_fingerprint);

  resource.rewind(0);
  crowd.leader.releaseResource(resource, crowd.wfc_list);
  CHECK(resource.getCursor() == 1);
  CHECK(resource.getOutstandingLoanCount() == 0);

  // A second complete loan proves successful release cleared every fixed
  // clone-local marker rather than leaving stale crowd identity behind.
  resource.rewind(0);
  crowd.leader.acquireResource(resource, crowd.wfc_list);
  CHECK(Probe::requirePlannedRuntime(
            crowd.leader, crowd.wfc_list, *crowd.p_list, request) ==
        storage_fingerprint);
  resource.rewind(0);
  crowd.leader.releaseResource(resource, crowd.wfc_list);
  CHECK(resource.getOutstandingLoanCount() == 0);
}

} // namespace qmcplusplus
