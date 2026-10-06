//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2020 QMCPACK developers.
//
// File developed by: Bryan Clark, bclark@Princeton.edu, Princeton University
//                    Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Raymond Clay III, j.k.rofling@gmail.com, Lawrence Livermore National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//                    Anouar Benali, abenali.sci@gmail.com, Qubit Pharmaceuticals
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>

#include "TrialWaveFunction.h"
#include "Particle/MCMultiParticleMoves.h"
#include "QMCWaveFunctions/Optimization/StructuredParameterProvider.h"
#include "QMCWaveFunctions/TrialWaveFunctionMemoryPolicy.h"
#include "Utilities/BatchResourcePreparation.h"
#include "ResourceCollection.h"
#include "Utilities/IteratorUtility.h"
#include "Concurrency/Info.hpp"
#include "type_traits/ConvertToReal.h"
#include "NaNguard.h"
#include "Fermion/SlaterDet.h"
#include "Fermion/MultiSlaterDetTableMethod.h"
#include "QMCWaveFunctions/TWFFastDerivWrapper.h"

namespace qmcplusplus
{
namespace
{
// Return the destination span required by possibly sparse global mappings.
std::size_t requiredDerivativeExtent(const OptVariables& optvars)
{
  std::size_t required_extent = 0;
  for (std::size_t local_index = 0; local_index < optvars.size(); ++local_index)
  {
    const int global_index = optvars.where(local_index);
    if (global_index >= 0)
      required_extent = std::max(required_extent, static_cast<std::size_t>(global_index) + 1);
  }
  return required_extent;
}

/** Build the stable structural identity used in planning evidence and binding.
 * Component-provided text is escaped one path segment at a time so separators,
 * percent signs, control bytes, and non-ASCII bytes cannot alias topology.
 */
std::string batchParticipantId(std::size_t index,
                               const WaveFunctionComponent& component)
{
  return "twf/component/" + std::to_string(index) + "/" +
      escapeBatchParticipantIdSegment(component.getClassName()) + "/" +
      escapeBatchParticipantIdSegment(component.getName());
}

/// Mix one byte into the fixed-size component-topology fingerprint.
void mixTopologyByte(std::uint64_t& hash, unsigned char byte) noexcept
{
  constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;
  hash ^= static_cast<std::uint64_t>(byte);
  hash *= FNV_PRIME;
}

/// Mix an integer in a fixed-width, platform-independent byte order.
void mixTopologyInteger(std::uint64_t& hash, std::size_t value) noexcept
{
  const std::uint64_t fixed_width_value = value;
  for (std::size_t byte = 0; byte < sizeof(fixed_width_value); ++byte)
    mixTopologyByte(hash, static_cast<unsigned char>((fixed_width_value >> (8 * byte)) & 0xffU));
}

/// Mix length-delimited text so adjacent component fields cannot alias.
void mixTopologyText(std::uint64_t& hash, std::string_view text) noexcept
{
  mixTopologyInteger(hash, text.size());
  for (const unsigned char byte : text)
    mixTopologyByte(hash, byte);
}

/** Return an allocation-free retained fingerprint of component structure.
 * getClassName() currently returns a temporary string, but no text or dynamic
 * container is retained by TrialWaveFunction after this snapshot is formed.
 */
std::uint64_t batchParticipantTopologyFingerprint(
    const std::vector<std::unique_ptr<WaveFunctionComponent>>& components)
{
  constexpr std::uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
  std::uint64_t hash                       = FNV_OFFSET_BASIS;
  mixTopologyInteger(hash, components.size());
  for (std::size_t index = 0; index < components.size(); ++index)
  {
    mixTopologyInteger(hash, index);
    const std::string class_name = components[index]->getClassName();
    mixTopologyText(hash, class_name);
    mixTopologyText(hash, components[index]->getName());
  }
  return hash;
}

/// Return whether one participant envelope fits within the aggregate envelope.
bool capacitiesFitWithin(const BatchTileCapacities& capacities,
                         const BatchTileCapacities& envelope) noexcept
{
  return capacities.value <= envelope.value && capacities.full_vgl <= envelope.full_vgl &&
      capacities.active_gradient <= envelope.active_gradient && capacities.ecp_outer <= envelope.ecp_outer;
}

/// Return the exact aggregate type widths used by this executable.
TrialWaveFunctionMemoryTypeSizes makeTrialWaveFunctionMemoryTypeSizesForBuild() noexcept
{
  return makeTrialWaveFunctionMemoryTypeSizes<
      TrialWaveFunction::ValueType, ParticleSet::ParticleGradient::value_type,
      ParticleSet::ParticleLaplacian::value_type, std::reference_wrapper<WaveFunctionComponent>,
      std::reference_wrapper<ParticleSet::ParticleGradient>,
      std::reference_wrapper<ParticleSet::ParticleLaplacian>, TrialWaveFunction::ParameterDerivativeView,
      TrialWaveFunction::EvaluationStamp, unsigned char>();
}

/** Build aggregate policy input from current object and sole-child evidence.
 * The accounting override is private test state; production claims stay false
 * until every aggregate runtime owner is migrated to prepared storage.
 */
TrialWaveFunctionMemoryPolicyInput makeTrialWaveFunctionMemoryPolicyInput(
    std::size_t component_count,
    bool use_tasking,
    bool fallback_path_reachable,
    bool complete_accounting_for_testing,
    const BatchMemoryContribution* sole_child,
    bool sole_child_atomic_publication)
{
  TrialWaveFunctionMemoryPolicyInput input;
  input.type_sizes = makeTrialWaveFunctionMemoryTypeSizesForBuild();
  input.accounting_claims = complete_accounting_for_testing
      ? TrialWaveFunctionMemoryAccountingClaims::complete()
      : TrialWaveFunctionMemoryAccountingClaims{};
  input.component_count          = component_count;
  input.use_tasking              = use_tasking;
  input.fallback_path_reachable  = fallback_path_reachable;
  if (sole_child)
    input.sole_child = {sole_child->owner_multiplicity, sole_child->fully_accounted,
                        sole_child_atomic_publication};
  return input;
}
} // namespace

typedef enum
{
  V_TIMER = 0,
  VGL_TIMER,
  ACCEPT_TIMER,
  NL_TIMER,
  RECOMPUTE_TIMER,
  BUFFER_TIMER,
  DERIVS_TIMER,
  PREPAREGROUP_TIMER,
  TIMER_SKIP
} TimerEnum;

static const std::vector<std::string> suffixes{"V",         "VGL",    "accept", "NLratio",
                                               "recompute", "buffer", "derivs", "preparegroup"};

/** Exact crowd-owned aggregate views and staging for one TrialWaveFunction team.
 *
 * Copies retain only immutable filler and planning provenance.  Prepared
 * storage is rebuilt by ResourceCollection for the destination crowd, so a
 * copied prepared collection remains storage-empty until explicitly cleared.
 */
struct TrialWaveFunction::TrialWaveFunctionMultiWalkerResource : public Resource
{
  using ComponentView = RefVectorWithLeader<WaveFunctionComponent>;
  using GradientView  = RefVectorWithLeader<ParticleSet::ParticleGradient>;
  using LaplacianView = RefVectorWithLeader<ParticleSet::ParticleLaplacian>;

  /// Construct an empty template retaining only stable filler and plan facts.
  TrialWaveFunctionMultiWalkerResource(
      WaveFunctionComponent* component_filler,
      TrialWaveFunctionMemoryPolicyInput policy_input,
      BatchExecutionParticipantPlan expected_plan = {})
      : Resource("TrialWaveFunctionMultiWalkerResource"),
        component_filler_(component_filler),
        policy_input_(std::move(policy_input)),
        expected_plan_(std::move(expected_plan))
  {}

  /// Copy immutable template provenance without prepared storage or bindings.
  TrialWaveFunctionMultiWalkerResource(
      const TrialWaveFunctionMultiWalkerResource& other)
      : TrialWaveFunctionMultiWalkerResource(other.component_filler_,
                                               other.policy_input_,
                                               other.expected_plan_)
  {}

  /// Recreate an empty resource for a copied ResourceCollection.
  std::unique_ptr<Resource> makeClone() const override
  {
    return std::make_unique<TrialWaveFunctionMultiWalkerResource>(*this);
  }

  /// Validate one preparation context without mutating resource state.
  void validateBatchResourcePreparation(
      const BatchResourcePreparationContext& context) const override
  {
    context.validate();
    if (!context.plan)
      return;
    if (prepared_plan_)
      throw std::logic_error(
          "TrialWaveFunction aggregate resource replanning requires an explicit null clear");
    if (retainedBytes() != 0)
      throw std::logic_error(
          "TrialWaveFunction aggregate resource retained storage before preparation");
    if (policy_input_.component_count != 1 || policy_input_.use_tasking ||
        policy_input_.fallback_path_reachable || !component_filler_)
      throw std::invalid_argument(
          "TrialWaveFunction planned aggregate resource requires one direct component");

    const BatchExecutionParticipantPlan selected =
        makeBatchExecutionParticipantPlan(
            context.plan, TRIAL_WAVEFUNCTION_MEMORY_PARTICIPANT_ID);
    if (expected_plan_ && !expected_plan_.sameBinding(selected))
      throw std::invalid_argument(
          "TrialWaveFunction aggregate resource received the wrong plan");
    validateEvidence(selected);
    const auto plans = makeCrowdPlans(selected);
    if (context.crowd_index >= plans.size())
      throw std::out_of_range(
          "TrialWaveFunction aggregate resource crowd index is outside its plan");
  }

  /// Materialize exact crowd storage transactionally, or clear it on null.
  void prepareBatchResource(
      const BatchResourcePreparationContext& context) override
  {
    validateBatchResourcePreparation(context);
    if (!context.plan)
    {
      clearToNoPolicy();
      return;
    }

    BatchExecutionParticipantPlan selected =
        makeBatchExecutionParticipantPlan(
            context.plan, TRIAL_WAVEFUNCTION_MEMORY_PARTICIPANT_ID);
    auto plans = makeCrowdPlans(selected);
    TrialWaveFunctionCrowdMemoryPlan crowd_plan =
        std::move(plans.at(context.crowd_index));
    TrialWaveFunctionMultiWalkerResource candidate(
        component_filler_, policy_input_, selected);
    candidate.materialize(std::move(selected), context.crowd_index,
                          std::move(crowd_plan));
    publish(std::move(candidate));
  }

  /** Verify prepared provenance and storage before any runtime view is rebound. */
  void validateAcquiredBinding(const BatchExecutionParticipantPlan& plan,
                               std::optional<std::size_t> crowd_index,
                               std::size_t live_walkers) const
  {
    if (!plan)
    {
      if (expected_plan_ || prepared_plan_ || prepared_crowd_plan_)
        throw std::logic_error(
            "TrialWaveFunction no-policy acquisition received a planned aggregate resource");
      return;
    }
    if (!expected_plan_.sameBinding(plan) || !prepared_plan_.sameBinding(plan) ||
        !prepared_crowd_plan_)
      throw std::logic_error(
          "TrialWaveFunction aggregate resource was not prepared for this plan");
    if (prepared_plan_fingerprint_ != plan.plan().fingerprint())
      throw std::logic_error(
          "TrialWaveFunction aggregate resource has stale plan provenance");
    if (crowd_index && *crowd_index != prepared_crowd_index_)
      throw std::logic_error(
          "TrialWaveFunction aggregate resource has the wrong crowd identity");
    if (live_walkers > prepared_crowd_plan_->reserve_walkers)
      throw std::length_error(
          "TrialWaveFunction live crowd exceeds the prepared reserve");
    if (prepared_storage_fingerprint_ == 0 ||
        prepared_storage_fingerprint_ != storageFingerprint())
      throw std::logic_error(
          "TrialWaveFunction aggregate resource allocation identity changed");
    if (!matchesPreparedCapacities())
      throw std::logic_error(
          "TrialWaveFunction aggregate resource capacity changed");
  }

  /** Verify that release presents the identical leader and ordered lane list. */
  bool sameBoundTeam(
      const RefVectorWithLeader<TrialWaveFunction>& wavefunctions) const noexcept
  {
    if (!component_refs_ || !gradient_refs_ || !laplacian_refs_ ||
        &component_refs_->getLeader() !=
            wavefunctions.getLeader().Z.front().get() ||
        &gradient_refs_->getLeader() != &wavefunctions.getLeader().G ||
        &laplacian_refs_->getLeader() != &wavefunctions.getLeader().L ||
        component_refs_->size() != wavefunctions.size() ||
        gradient_refs_->size() != wavefunctions.size() ||
        laplacian_refs_->size() != wavefunctions.size())
      return false;
    for (std::size_t walker = 0; walker < wavefunctions.size(); ++walker)
      if (&(*component_refs_)[walker] != wavefunctions[walker].Z.front().get() ||
          &(*gradient_refs_)[walker] != &wavefunctions[walker].G ||
          &(*laplacian_refs_)[walker] != &wavefunctions[walker].L)
        return false;
    return true;
  }

  /** Rebind fixed-capacity reference views to the current live team. */
  void bindViews(const RefVectorWithLeader<TrialWaveFunction>& wavefunctions)
  {
    const std::size_t live = wavefunctions.size();
    const std::size_t reserve = prepared_crowd_plan_->reserve_walkers;
    restoreReferenceExtent(reserve);
    restoreScratchExtent();

    component_refs_->rebindLeader(*wavefunctions.getLeader().Z.front());
    gradient_refs_->rebindLeader(wavefunctions.getLeader().G);
    laplacian_refs_->rebindLeader(wavefunctions.getLeader().L);
    for (std::size_t walker = 0; walker < live; ++walker)
    {
      component_refs_->rebindElement(walker, *wavefunctions[walker].Z.front());
      gradient_refs_->rebindElement(walker, wavefunctions[walker].G);
      laplacian_refs_->rebindElement(walker, wavefunctions[walker].L);
    }
    component_refs_->erase(component_refs_->begin() + live,
                           component_refs_->end());
    gradient_refs_->erase(gradient_refs_->begin() + live,
                          gradient_refs_->end());
    laplacian_refs_->erase(laplacian_refs_->begin() + live,
                           laplacian_refs_->end());
    if (!weighted_derivative_views_.empty())
      weighted_derivative_views_.resize(live);
    transaction_flags_.resize(live);
    std::fill(transaction_flags_.begin(), transaction_flags_.end(), 0);
  }

  /** Restore stable filler references without touching large numeric staging. */
  void resetViews() noexcept
  {
    if (!prepared_crowd_plan_)
      return;
    component_refs_->rebindLeader(*component_filler_);
    gradient_refs_->rebindLeader(gradient_filler_);
    laplacian_refs_->rebindLeader(laplacian_filler_);
    for (std::size_t walker = 0; walker < component_refs_->size(); ++walker)
    {
      component_refs_->rebindElement(walker, *component_filler_);
      gradient_refs_->rebindElement(walker, gradient_filler_);
      laplacian_refs_->rebindElement(walker, laplacian_filler_);
    }
    // Numeric staging remains at its admitted extent.  Its contents are dead
    // outside a loan, so release avoids an O(B*P) clear; only the small live
    // transaction mask is invalidated before the resource is returned.
    std::fill(transaction_flags_.begin(), transaction_flags_.end(), 0);
  }

  /// Expose the sole-component lane view to TrialWaveFunction runtime paths.
  ComponentView& componentView() { return *component_refs_; }
  const ComponentView& componentView() const { return *component_refs_; }

  /// Expose the aggregate gradient lane view to TrialWaveFunction runtime paths.
  GradientView& gradientView() { return *gradient_refs_; }
  const GradientView& gradientView() const { return *gradient_refs_; }

  /// Expose the aggregate Laplacian lane view to TrialWaveFunction runtime paths.
  LaplacianView& laplacianView() { return *laplacian_refs_; }
  const LaplacianView& laplacianView() const { return *laplacian_refs_; }

  /// Return private weighted-ECP ratios retained for the admitted outer tile.
  std::vector<ValueType>& privateRatios() noexcept { return private_ratios_; }

  /// Return accumulated weighted-ECP walker weights.
  std::vector<ValueType>& totalWeights() noexcept { return total_weights_; }

  /// Return flat parameter-derivative delta staging.
  std::vector<ValueType>& derivativeDelta() noexcept
  { return weighted_derivative_delta_; }

  /// Return walker views into flat parameter-derivative delta staging.
  std::vector<ParameterDerivativeView>& derivativeViews() noexcept
  { return weighted_derivative_views_; }

  /// Return accepted-value stamps used by weighted evaluations.
  std::vector<EvaluationStamp>& valueStamps() noexcept { return value_stamps_; }

  /// Return per-walker transaction-state flags.
  std::vector<unsigned char>& transactionFlags() noexcept
  { return transaction_flags_; }

  /// Report whether this crowd resource has exact prepared provenance.
  bool isPrepared() const noexcept { return static_cast<bool>(prepared_plan_); }

  /// Return the retained allocation-identity fingerprint for tests.
  std::size_t storageFingerprintForTesting() const noexcept
  { return prepared_storage_fingerprint_; }

  /// Return the exact crowd descriptor retained after preparation.
  const std::optional<TrialWaveFunctionCrowdMemoryPlan>&
  crowdPlanForTesting() const noexcept
  { return prepared_crowd_plan_; }

  /// Return measured category bytes retained after preparation.
  const BatchMemoryEstimate& actualStorageForTesting() const noexcept
  { return actual_resource_storage_; }

  /// Return the participant plan view used to prepare this crowd resource.
  const BatchExecutionParticipantPlan& preparedPlanForTesting() const noexcept
  { return prepared_plan_; }

  /// Return the prepared crowd index for diagnostics.
  std::size_t preparedCrowdIndexForTesting() const noexcept
  { return prepared_crowd_index_; }

  /// Corrupt the retained crowd index for a focused fail-closed test.
  void setPreparedCrowdIndexForTesting(std::size_t crowd_index) noexcept
  { prepared_crowd_index_ = crowd_index; }

  /// Measure typed allocation capacities for focused exactness tests.
  TrialWaveFunctionResourceStorageRequirement measuredRequirementForTesting() const
  { return measureRequirement(); }

  /** Verify every idle reference view targets this resource's own fillers. */
  bool placeholdersMatchDestinationFillersForTesting() const noexcept
  {
    if (!component_refs_ || !gradient_refs_ || !laplacian_refs_ ||
        &component_refs_->getLeader() != component_filler_ ||
        &gradient_refs_->getLeader() != &gradient_filler_ ||
        &laplacian_refs_->getLeader() != &laplacian_filler_ ||
        component_refs_->size() != gradient_refs_->size() ||
        component_refs_->size() != laplacian_refs_->size())
      return false;
    for (std::size_t slot = 0; slot < component_refs_->size(); ++slot)
      if (&(*component_refs_)[slot] != component_filler_ ||
          &(*gradient_refs_)[slot] != &gradient_filler_ ||
          &(*laplacian_refs_)[slot] != &laplacian_filler_)
        return false;
    return true;
  }

private:
  /// Allocate a vector whose capacity exactly matches one byte descriptor.
  template<class T>
  static std::vector<T> makeExactVector(std::size_t bytes,
                                        const char* description)
  {
    if (bytes % sizeof(T) != 0)
      throw std::length_error(std::string(description) +
                              " is not divisible by its element size");
    std::vector<T> result(bytes / sizeof(T));
    if (result.capacity() * sizeof(T) != bytes)
      throw std::length_error(std::string(description) +
                              " exceeded its exact admitted capacity");
    return result;
  }

  /// Measure one vector's owned allocation capacity in bytes.
  template<class T>
  static std::size_t vectorBytes(const std::vector<T>& values,
                                 const char* description)
  {
    return checkedBatchMemoryMultiply(values.capacity(), sizeof(T), description);
  }

  /// Release all capacity retained by one typed vector.
  template<class T>
  static void freeVector(std::vector<T>& values)
  {
    std::vector<T>().swap(values);
  }

  /// Reconstruct stable per-crowd descriptors for one validated plan view.
  std::vector<TrialWaveFunctionCrowdMemoryPlan> makeCrowdPlans(
      const BatchExecutionParticipantPlan& plan) const
  {
    const BatchExecutionPlan& selected = plan.plan();
    return makeTrialWaveFunctionCrowdMemoryPlans(
        policy_input_,
        {selected.requirements(), selected.topology(), selected.logicalMaximum(),
         selected.selectedCapacities(), selected.particleCount(),
         selected.activeParameterCount(), selected.parameterDerivativeWidth()});
  }

  /// Recompute selected and minimum evidence from immutable resource facts.
  void validateEvidence(const BatchExecutionParticipantPlan& participant) const
  {
    const BatchExecutionPlan& plan = participant.plan();
    const auto& evidence = participant.evidence();
    if (evidence.participant_id !=
        TRIAL_WAVEFUNCTION_MEMORY_PARTICIPANT_ID)
      throw std::invalid_argument(
          "TrialWaveFunction aggregate resource participant identity is stale");
    BatchExecutionPlanningContext context{
        plan.requirements(), plan.topology(), plan.logicalMaximum(),
        plan.selectedCapacities(), plan.particleCount(),
        plan.activeParameterCount(), plan.parameterDerivativeWidth()};
    const BatchMemoryContribution selected =
        estimateTrialWaveFunctionBatchMemory(policy_input_, context);
    if (!(selected.logical_maximum == evidence.logical_maximum) ||
        selected.owner_multiplicity != evidence.owner_multiplicity ||
        !(selected.per_owner == evidence.selected_per_owner) ||
        selected.fully_accounted != evidence.fully_accounted ||
        !selected.fully_accounted)
      throw std::invalid_argument(
          "TrialWaveFunction aggregate resource selected evidence is stale");
    context.candidate_capacities = plan.minimumCapacities();
    const BatchMemoryContribution minimum =
        estimateTrialWaveFunctionBatchMemory(policy_input_, context);
    if (!(minimum.logical_maximum == evidence.logical_maximum) ||
        minimum.owner_multiplicity != evidence.owner_multiplicity ||
        !(minimum.per_owner == evidence.fixed_minimum_per_owner) ||
        minimum.fully_accounted != evidence.fully_accounted)
      throw std::invalid_argument(
          "TrialWaveFunction aggregate resource minimum evidence is stale");
  }

  /// Compare every field of two typed resource-storage descriptors.
  static bool sameRequirement(
      const TrialWaveFunctionResourceStorageRequirement& lhs,
      const TrialWaveFunctionResourceStorageRequirement& rhs) noexcept
  {
    return lhs.component_reference_slots == rhs.component_reference_slots &&
        lhs.aggregate_gradient_reference_slots ==
            rhs.aggregate_gradient_reference_slots &&
        lhs.aggregate_laplacian_reference_slots ==
            rhs.aggregate_laplacian_reference_slots &&
        lhs.weighted_ecp_private_ratios == rhs.weighted_ecp_private_ratios &&
        lhs.weighted_ecp_total_weights == rhs.weighted_ecp_total_weights &&
        lhs.weighted_parameter_derivative_delta ==
            rhs.weighted_parameter_derivative_delta &&
        lhs.weighted_parameter_derivative_views ==
            rhs.weighted_parameter_derivative_views &&
        lhs.weighted_value_stamps == rhs.weighted_value_stamps &&
        lhs.transaction_flags == rhs.transaction_flags;
  }

  /// Measure all typed allocation capacities by their runtime purpose.
  TrialWaveFunctionResourceStorageRequirement measureRequirement() const
  {
    TrialWaveFunctionResourceStorageRequirement measured;
    measured.component_reference_slots = component_refs_
        ? vectorBytes(static_cast<const ComponentView::BaseVec&>(*component_refs_),
                      "TWF component-reference storage")
        : 0;
    measured.aggregate_gradient_reference_slots = gradient_refs_
        ? vectorBytes(static_cast<const GradientView::BaseVec&>(*gradient_refs_),
                      "TWF gradient-reference storage")
        : 0;
    measured.aggregate_laplacian_reference_slots = laplacian_refs_
        ? vectorBytes(static_cast<const LaplacianView::BaseVec&>(*laplacian_refs_),
                      "TWF Laplacian-reference storage")
        : 0;
    measured.weighted_ecp_private_ratios =
        vectorBytes(private_ratios_, "TWF private-ratio storage");
    measured.weighted_ecp_total_weights =
        vectorBytes(total_weights_, "TWF total-weight storage");
    measured.weighted_parameter_derivative_delta = vectorBytes(
        weighted_derivative_delta_, "TWF derivative-delta storage");
    measured.weighted_parameter_derivative_views = vectorBytes(
        weighted_derivative_views_, "TWF derivative-view storage");
    measured.weighted_value_stamps =
        vectorBytes(value_stamps_, "TWF value-stamp storage");
    measured.transaction_flags =
        vectorBytes(transaction_flags_, "TWF transaction-flag storage");
    return measured;
  }

  /// Convert the typed capacity measurement to batch-memory categories.
  BatchMemoryEstimate measureStorage() const
  {
    const auto requirement = measureRequirement();
    BatchMemoryEstimate measured;
    measured.add(BatchMemoryCategory::LOGICAL_INPUT_OUTPUT,
                 {requirement.logicalInputOutputBytes(), 0});
    measured.add(BatchMemoryCategory::OUTER_TILE_SCRATCH,
                 {requirement.outerTileScratchBytes(), 0});
    measured.add(BatchMemoryCategory::PUBLICATION_STAGING,
                 {requirement.publicationStagingBytes(), 0});
    measured.add(BatchMemoryCategory::ECP_METADATA,
                 {requirement.ecpMetadataBytes(), 0});
    return measured;
  }

  /// Compare capacities without checked-arithmetic diagnostics after a loan.
  bool matchesPreparedCapacities() const noexcept
  {
    if (!prepared_crowd_plan_ || !component_refs_ || !gradient_refs_ ||
        !laplacian_refs_)
      return false;
    const auto& required = prepared_crowd_plan_->resource_storage;
    return component_refs_->capacity() *
                sizeof(std::reference_wrapper<WaveFunctionComponent>) ==
            required.component_reference_slots &&
        gradient_refs_->capacity() *
                sizeof(std::reference_wrapper<ParticleSet::ParticleGradient>) ==
            required.aggregate_gradient_reference_slots &&
        laplacian_refs_->capacity() *
                sizeof(std::reference_wrapper<ParticleSet::ParticleLaplacian>) ==
            required.aggregate_laplacian_reference_slots &&
        private_ratios_.capacity() * sizeof(ValueType) ==
            required.weighted_ecp_private_ratios &&
        total_weights_.capacity() * sizeof(ValueType) ==
            required.weighted_ecp_total_weights &&
        weighted_derivative_delta_.capacity() * sizeof(ValueType) ==
            required.weighted_parameter_derivative_delta &&
        weighted_derivative_views_.capacity() *
                sizeof(ParameterDerivativeView) ==
            required.weighted_parameter_derivative_views &&
        value_stamps_.capacity() * sizeof(EvaluationStamp) ==
            required.weighted_value_stamps &&
        transaction_flags_.capacity() * sizeof(unsigned char) ==
            required.transaction_flags;
  }

  /// Allocate and verify every typed buffer for one crowd descriptor.
  void materialize(BatchExecutionParticipantPlan plan,
                   std::size_t crowd_index,
                   TrialWaveFunctionCrowdMemoryPlan crowd_plan)
  {
    const auto& required = crowd_plan.resource_storage;
    component_refs_.emplace(*component_filler_);
    gradient_refs_.emplace(gradient_filler_);
    laplacian_refs_.emplace(laplacian_filler_);
    const std::size_t reserve = crowd_plan.reserve_walkers;
    const std::size_t component_slots = checkedBatchMemoryMultiply(
        reserve, crowd_plan.component_count,
        "TWF aggregate component-reference extent");
    component_refs_->reserve(component_slots);
    gradient_refs_->reserve(reserve);
    laplacian_refs_->reserve(reserve);
    for (std::size_t slot = 0; slot < reserve; ++slot)
    {
      component_refs_->push_back(*component_filler_);
      gradient_refs_->push_back(gradient_filler_);
      laplacian_refs_->push_back(laplacian_filler_);
    }
    private_ratios_ = makeExactVector<ValueType>(
        required.weighted_ecp_private_ratios, "TWF private ratios");
    total_weights_ = makeExactVector<ValueType>(
        required.weighted_ecp_total_weights, "TWF total weights");
    weighted_derivative_delta_ = makeExactVector<ValueType>(
        required.weighted_parameter_derivative_delta,
        "TWF weighted derivative delta");
    weighted_derivative_views_ = makeExactVector<ParameterDerivativeView>(
        required.weighted_parameter_derivative_views,
        "TWF weighted derivative views");
    value_stamps_ = makeExactVector<EvaluationStamp>(
        required.weighted_value_stamps, "TWF value stamps");
    transaction_flags_ = makeExactVector<unsigned char>(
        required.transaction_flags, "TWF transaction flags");
    bindDerivativeViews(crowd_plan.parameter_derivative_width);

    const auto measured_requirement = measureRequirement();
    if (!sameRequirement(measured_requirement, required))
      throw std::length_error(
          "TrialWaveFunction aggregate resource typed capacities differ from policy");
    const BatchMemoryEstimate measured = measureStorage();
    if (!(measured == crowd_plan.expected_resource_storage))
      throw std::length_error(
          "TrialWaveFunction aggregate resource categorized bytes differ from policy");

    expected_plan_                = plan;
    prepared_plan_                = std::move(plan);
    prepared_crowd_plan_          = std::move(crowd_plan);
    prepared_crowd_index_         = crowd_index;
    prepared_plan_fingerprint_    = prepared_plan_.plan().fingerprint();
    actual_resource_storage_      = measured;
    prepared_storage_fingerprint_ = storageFingerprint();
    if (prepared_storage_fingerprint_ == 0)
      throw std::logic_error(
          "TrialWaveFunction aggregate resource fingerprint is invalid");
  }

  /// Publish a completely materialized candidate using noexcept moves.
  void publish(TrialWaveFunctionMultiWalkerResource&& candidate) noexcept
  {
    static_assert(std::is_nothrow_move_assignable_v<
                  std::optional<ComponentView>>);
    static_assert(std::is_nothrow_move_assignable_v<
                  std::optional<GradientView>>);
    static_assert(std::is_nothrow_move_assignable_v<
                  std::optional<LaplacianView>>);
    static_assert(std::is_nothrow_move_assignable_v<std::vector<ValueType>>);
    static_assert(std::is_nothrow_move_assignable_v<
                  std::vector<ParameterDerivativeView>>);
    static_assert(std::is_nothrow_move_assignable_v<
                  std::vector<EvaluationStamp>>);
    static_assert(std::is_nothrow_move_assignable_v<
                  std::vector<unsigned char>>);
    static_assert(std::is_nothrow_move_assignable_v<
                  BatchExecutionParticipantPlan>);
    static_assert(std::is_nothrow_move_assignable_v<
                  std::optional<TrialWaveFunctionCrowdMemoryPlan>>);
    static_assert(std::is_nothrow_copy_assignable_v<BatchMemoryEstimate>);
    component_refs_                    = std::move(candidate.component_refs_);
    gradient_refs_                     = std::move(candidate.gradient_refs_);
    laplacian_refs_                    = std::move(candidate.laplacian_refs_);
    private_ratios_                    = std::move(candidate.private_ratios_);
    total_weights_                     = std::move(candidate.total_weights_);
    weighted_derivative_delta_         = std::move(candidate.weighted_derivative_delta_);
    weighted_derivative_views_         = std::move(candidate.weighted_derivative_views_);
    value_stamps_                      = std::move(candidate.value_stamps_);
    transaction_flags_                 = std::move(candidate.transaction_flags_);
    expected_plan_                     = std::move(candidate.expected_plan_);
    prepared_plan_                     = std::move(candidate.prepared_plan_);
    prepared_crowd_plan_               = std::move(candidate.prepared_crowd_plan_);
    prepared_crowd_index_              = candidate.prepared_crowd_index_;
    prepared_plan_fingerprint_         = candidate.prepared_plan_fingerprint_;
    prepared_storage_fingerprint_      = candidate.prepared_storage_fingerprint_;
    actual_resource_storage_           = candidate.actual_resource_storage_;

    // Reference-wrapper storage was constructed against the temporary
    // candidate's owned fillers.  Rebind it to this destination resource
    // before candidate destruction; this touches only O(B) reference slots.
    resetViews();
  }

  /// Release all prepared storage and return to reusable no-policy state.
  void clearToNoPolicy()
  {
    component_refs_.reset();
    gradient_refs_.reset();
    laplacian_refs_.reset();
    freeVector(private_ratios_);
    freeVector(total_weights_);
    freeVector(weighted_derivative_delta_);
    freeVector(weighted_derivative_views_);
    freeVector(value_stamps_);
    freeVector(transaction_flags_);
    expected_plan_ = {};
    prepared_plan_ = {};
    prepared_crowd_plan_.reset();
    prepared_crowd_index_ = 0;
    prepared_plan_fingerprint_ = 0;
    prepared_storage_fingerprint_ = 0;
    actual_resource_storage_ = {};
  }

  /// Restore full reference-view extents using already admitted capacity.
  void restoreReferenceExtent(std::size_t reserve)
  {
    component_refs_->resize(reserve, std::ref(*component_filler_));
    gradient_refs_->resize(reserve, std::ref(gradient_filler_));
    laplacian_refs_->resize(reserve, std::ref(laplacian_filler_));
  }

  /// Restore exact logical staging extents without exceeding prepared capacity.
  void restoreScratchExtent()
  {
    const auto& required = prepared_crowd_plan_->resource_storage;
    private_ratios_.resize(required.weighted_ecp_private_ratios /
                           sizeof(ValueType));
    total_weights_.resize(required.weighted_ecp_total_weights /
                          sizeof(ValueType));
    weighted_derivative_delta_.resize(
        required.weighted_parameter_derivative_delta / sizeof(ValueType));
    weighted_derivative_views_.resize(
        required.weighted_parameter_derivative_views /
        sizeof(ParameterDerivativeView));
    value_stamps_.resize(required.weighted_value_stamps /
                         sizeof(EvaluationStamp));
    transaction_flags_.resize(required.transaction_flags /
                              sizeof(unsigned char));
    bindDerivativeViews(prepared_crowd_plan_->parameter_derivative_width);
  }

  /// Point each derivative row view into its stable slice of the flat buffer.
  void bindDerivativeViews(std::size_t width) noexcept
  {
    for (std::size_t walker = 0; walker < weighted_derivative_views_.size(); ++walker)
      weighted_derivative_views_[walker] = {
          width ? weighted_derivative_delta_.data() + walker * width : nullptr,
          width};
  }

  /// Return every currently retained heap byte across typed buffers.
  std::size_t retainedBytes() const
  {
    return measureRequirement().totalBytes();
  }

  /// Hash allocation addresses and capacities for post-publication validation.
  std::size_t storageFingerprint() const noexcept
  {
    std::size_t hash = 1469598103934665603ULL;
    auto mix = [&hash](std::uintptr_t value) {
      hash ^= value;
      hash *= 1099511628211ULL;
    };
    auto mix_vector = [&mix](const auto& values) {
      mix(reinterpret_cast<std::uintptr_t>(values.data()));
      mix(values.capacity());
    };
    if (component_refs_)
      mix_vector(static_cast<const ComponentView::BaseVec&>(*component_refs_));
    if (gradient_refs_)
      mix_vector(static_cast<const GradientView::BaseVec&>(*gradient_refs_));
    if (laplacian_refs_)
      mix_vector(static_cast<const LaplacianView::BaseVec&>(*laplacian_refs_));
    mix_vector(private_ratios_);
    mix_vector(total_weights_);
    mix_vector(weighted_derivative_delta_);
    mix_vector(weighted_derivative_views_);
    mix_vector(value_stamps_);
    mix_vector(transaction_flags_);
    return hash;
  }

  WaveFunctionComponent* component_filler_;
  ParticleSet::ParticleGradient gradient_filler_;
  ParticleSet::ParticleLaplacian laplacian_filler_;
  TrialWaveFunctionMemoryPolicyInput policy_input_;
  BatchExecutionParticipantPlan expected_plan_;
  BatchExecutionParticipantPlan prepared_plan_;
  std::optional<TrialWaveFunctionCrowdMemoryPlan> prepared_crowd_plan_;
  std::size_t prepared_crowd_index_ = 0;
  std::uint64_t prepared_plan_fingerprint_ = 0;
  std::size_t prepared_storage_fingerprint_ = 0;
  BatchMemoryEstimate actual_resource_storage_;

  std::optional<ComponentView> component_refs_;
  std::optional<GradientView> gradient_refs_;
  std::optional<LaplacianView> laplacian_refs_;
  std::vector<ValueType> private_ratios_;
  std::vector<ValueType> total_weights_;
  std::vector<ValueType> weighted_derivative_delta_;
  std::vector<ParameterDerivativeView> weighted_derivative_views_;
  std::vector<EvaluationStamp> value_stamps_;
  std::vector<unsigned char> transaction_flags_;
};

// Snapshot exact aggregate allocation and live team bindings for friend tests.
TrialWaveFunction::AggregateResourceDiagnostics
TrialWaveFunction::aggregateResourceDiagnosticsForTesting() const
{
  if (!aggregate_mw_resource_handle_)
    throw std::logic_error(
        "TrialWaveFunction aggregate resource diagnostics require a live loan");
  const auto& resource = aggregate_mw_resource_handle_.getResource();
  AggregateResourceDiagnostics diagnostics;
  diagnostics.prepared = resource.isPrepared();
  diagnostics.storage_fingerprint = resource.storageFingerprintForTesting();
  if (!resource.crowdPlanForTesting())
    return diagnostics;

  const auto& crowd = *resource.crowdPlanForTesting();
  const auto measured = resource.measuredRequirementForTesting();
  diagnostics.plan_identity =
      &resource.preparedPlanForTesting().plan();
  diagnostics.crowd_index = resource.preparedCrowdIndexForTesting();
  diagnostics.reserve_walkers = crowd.reserve_walkers;
  diagnostics.expected_bytes =
      crowd.expected_resource_storage.total().host;
  diagnostics.actual_bytes =
      resource.actualStorageForTesting().total().host;
  diagnostics.component_reference_bytes = measured.component_reference_slots;
  diagnostics.gradient_reference_bytes =
      measured.aggregate_gradient_reference_slots;
  diagnostics.laplacian_reference_bytes =
      measured.aggregate_laplacian_reference_slots;
  diagnostics.private_ratio_bytes = measured.weighted_ecp_private_ratios;
  diagnostics.total_weight_bytes = measured.weighted_ecp_total_weights;
  diagnostics.derivative_delta_bytes =
      measured.weighted_parameter_derivative_delta;
  diagnostics.derivative_view_bytes =
      measured.weighted_parameter_derivative_views;
  diagnostics.value_stamp_bytes = measured.weighted_value_stamps;
  diagnostics.transaction_flag_bytes = measured.transaction_flags;
  const auto& components = static_cast<const TrialWaveFunctionMultiWalkerResource::ComponentView::BaseVec&>(
      resource.componentView());
  const auto& gradients = static_cast<const TrialWaveFunctionMultiWalkerResource::GradientView::BaseVec&>(
      resource.gradientView());
  const auto& laplacians = static_cast<const TrialWaveFunctionMultiWalkerResource::LaplacianView::BaseVec&>(
      resource.laplacianView());
  diagnostics.component_reference_data = components.data();
  diagnostics.gradient_reference_data = gradients.data();
  diagnostics.laplacian_reference_data = laplacians.data();
  diagnostics.component_leader = &resource.componentView().getLeader();
  diagnostics.gradient_leader = &resource.gradientView().getLeader();
  diagnostics.laplacian_leader = &resource.laplacianView().getLeader();
  if (!resource.componentView().empty())
  {
    diagnostics.first_component = &resource.componentView()[0];
    diagnostics.first_gradient = &resource.gradientView()[0];
    diagnostics.first_laplacian = &resource.laplacianView()[0];
  }
  return diagnostics;
}

// Corrupt retained crowd provenance for a focused fail-closed test.
void TrialWaveFunction::setAggregateResourceCrowdForTesting(
    ResourceCollection& collection, std::size_t crowd_index)
{
  const std::size_t entry_cursor = collection.getCursor();
  auto resource =
      collection.lendResource<TrialWaveFunctionMultiWalkerResource>();
  resource.getResource().setPreparedCrowdIndexForTesting(crowd_index);
  collection.rewind(entry_cursor);
  collection.takebackResource(resource);
  collection.rewind(entry_cursor);
}

// Check idle placeholder ownership without publishing a runtime resource loan.
bool TrialWaveFunction::aggregateResourcePlaceholdersMatchFillersForTesting(
    ResourceCollection& collection)
{
  const std::size_t entry_cursor = collection.getCursor();
  auto resource =
      collection.lendResource<TrialWaveFunctionMultiWalkerResource>();
  const bool matches =
      resource.getResource().placeholdersMatchDestinationFillersForTesting();
  collection.rewind(entry_cursor);
  collection.takebackResource(resource);
  collection.rewind(entry_cursor);
  return matches;
}

static TimerNameList_t<TimerEnum> create_names(std::string_view myName)
{
  TimerNameList_t<TimerEnum> timer_names;
  std::string prefix = std::string("WaveFunction:").append(myName).append("::");
  for (std::size_t i = 0; i < suffixes.size(); ++i)
    timer_names.push_back({static_cast<TimerEnum>(i), prefix + suffixes[i]});
  return timer_names;
}

TrialWaveFunction::TrialWaveFunction(const RuntimeOptions& runtime_options, const std::string_view aname, bool tasking)
    : runtime_options_(runtime_options),
      myNode_(NULL),
      spomap_(std::make_shared<SPOMap>()),
      myName(aname),
      BufferCursor(0),
      BufferCursor_scalar(0),
      PhaseValue(0.0),
      PhaseDiff(0.0),
      log_real_(0.0),
      use_tasking_(tasking),
      TWF_timers_(getGlobalTimerManager(), create_names(aname), timer_level_medium)
{
  if (suffixes.size() != TIMER_SKIP)
    throw std::runtime_error("TrialWaveFunction::TrialWaveFunction mismatched timer enums and suffixes");
}

/** Destructor
*
*@warning Have not decided whether Z is cleaned up by TrialWaveFunction
*  or not. It will depend on I/O implementation.
*/
TrialWaveFunction::~TrialWaveFunction()
{
  if (myNode_ != NULL)
    xmlFreeNode(myNode_);
}

/** Takes owndership of aterm
 */
void TrialWaveFunction::addComponent(std::unique_ptr<WaveFunctionComponent>&& aterm)
{
  if (resource_acquired_)
    throw std::logic_error(
        "Cannot change TrialWaveFunction component topology while resources are acquired");
  if (batch_execution_plan_)
    throw std::logic_error(
        "Clear the TrialWaveFunction batch execution plan before changing component topology");
  if (!aterm)
    throw std::invalid_argument(
        "TrialWaveFunction cannot add a null WaveFunctionComponent");

  std::string aname = aterm->getClassName();
  if (!aterm->getName().empty())
    aname += ":" + aterm->getName();

  if (aterm->isFermionic())
    app_log() << "  Added a fermionic WaveFunctionComponent " << aname << std::endl;

  for (auto& suffix : suffixes)
    WFC_timers_.push_back(createGlobalTimer(aname + "::" + suffix));

  Z.emplace_back(std::move(aterm));
}

void TrialWaveFunction::contributeBatchExecutionRequirements(
    BatchExecutionRequirements& requirements) const
{
  for (const auto& component : Z)
    component->contributeBatchExecutionRequirements(requirements);
}

BatchTileCapacities TrialWaveFunction::batchExecutionLogicalMaximum(
    const BatchExecutionWorkloadContext& context) const
{
  BatchTileCapacities maximum = trialWaveFunctionBatchLogicalMaximum(context);
  for (const auto& component : Z)
    includeBatchExecutionLogicalMaximum(
        maximum, component->batchExecutionLogicalMaximum(context));
  return maximum;
}

std::vector<BatchMemoryParticipantContribution>
TrialWaveFunction::estimateBatchExecutionMemory(
    const BatchExecutionPlanningContext& context) const
{
  // Evaluate each child exactly once for this candidate.  The aggregate's
  // completeness decision consumes that same live sole-child evidence.
  std::vector<BatchMemoryParticipantContribution> child_contributions;
  child_contributions.reserve(Z.size());
  for (std::size_t index = 0; index < Z.size(); ++index)
    child_contributions.push_back(
        {batchParticipantId(index, *Z[index]),
         Z[index]->estimateBatchExecutionMemory(context)});

  const BatchMemoryContribution* sole_child =
      child_contributions.size() == 1 ? &child_contributions.front().contribution : nullptr;
  const bool sole_child_atomic_publication =
      Z.size() == 1 && Z.front()->supportsAtomicBatchPublication();
  const TrialWaveFunctionMemoryPolicyInput aggregate_input =
      makeTrialWaveFunctionMemoryPolicyInput(
          Z.size(), use_tasking_, static_cast<bool>(twf_fastderiv_),
          complete_batch_memory_accounting_for_testing_, sole_child,
          sole_child_atomic_publication);

  std::vector<BatchMemoryParticipantContribution> contributions;
  contributions.reserve(child_contributions.size() + 1);
  contributions.push_back(
      {std::string(TRIAL_WAVEFUNCTION_MEMORY_PARTICIPANT_ID),
       estimateTrialWaveFunctionBatchMemory(aggregate_input, context)});
  contributions.insert(contributions.end(),
                       std::make_move_iterator(child_contributions.begin()),
                       std::make_move_iterator(child_contributions.end()));
  return contributions;
}

void TrialWaveFunction::validateAggregateBatchExecutionPlanBinding(
    const BatchExecutionParticipantPlan& participant_plan) const
{
  if (!participant_plan)
    return;

  const BatchExecutionPlan& plan = participant_plan.plan();
  const BatchMemoryParticipantEvidence& evidence = participant_plan.evidence();
  if (evidence.participant_id != TRIAL_WAVEFUNCTION_MEMORY_PARTICIPANT_ID)
    throw std::invalid_argument(
        "TrialWaveFunction aggregate batch participant has the wrong identity");

  const auto makeAggregateContribution = [this](const BatchExecutionPlanningContext& context) {
    BatchMemoryContribution child;
    const BatchMemoryContribution* sole_child = nullptr;
    bool sole_child_atomic_publication        = false;
    if (Z.size() == 1)
    {
      child = Z.front()->estimateBatchExecutionMemory(context);
      sole_child = &child;
      sole_child_atomic_publication = Z.front()->supportsAtomicBatchPublication();
    }
    const TrialWaveFunctionMemoryPolicyInput input =
        makeTrialWaveFunctionMemoryPolicyInput(
            Z.size(), use_tasking_, static_cast<bool>(twf_fastderiv_),
            complete_batch_memory_accounting_for_testing_, sole_child,
            sole_child_atomic_publication);
    return estimateTrialWaveFunctionBatchMemory(input, context);
  };

  const BatchExecutionPlanningContext selected_context{
      plan.requirements(), plan.topology(), plan.logicalMaximum(), plan.selectedCapacities(),
      plan.particleCount(), plan.activeParameterCount(), plan.parameterDerivativeWidth()};
  const BatchMemoryContribution selected = makeAggregateContribution(selected_context);
  if (!capacitiesFitWithin(selected.logical_maximum, plan.logicalMaximum()))
    throw std::invalid_argument(
        "TrialWaveFunction aggregate logical maximum exceeds the batch plan envelope");
  if (!(evidence.logical_maximum == selected.logical_maximum))
    throw std::invalid_argument(
        "TrialWaveFunction aggregate logical-maximum evidence is stale");
  if (selected.owner_multiplicity != 1 ||
      evidence.owner_multiplicity != selected.owner_multiplicity)
    throw std::invalid_argument(
        "TrialWaveFunction aggregate must have one exact rank-local owner");
  if (!(evidence.selected_per_owner == selected.per_owner))
    throw std::invalid_argument(
        "TrialWaveFunction aggregate selected memory evidence is stale");

  BatchExecutionPlanningContext minimum_context = selected_context;
  minimum_context.candidate_capacities           = plan.minimumCapacities();
  const BatchMemoryContribution minimum = makeAggregateContribution(minimum_context);
  if (!(minimum.logical_maximum == selected.logical_maximum) ||
      minimum.owner_multiplicity != selected.owner_multiplicity)
    throw std::invalid_argument(
        "TrialWaveFunction aggregate invariant evidence changed at the minimum capacity");
  if (!(evidence.fixed_minimum_per_owner == minimum.per_owner))
    throw std::invalid_argument(
        "TrialWaveFunction aggregate minimum memory evidence is stale");
  if (evidence.fully_accounted != selected.fully_accounted ||
      minimum.fully_accounted != selected.fully_accounted)
    throw std::invalid_argument(
        "TrialWaveFunction aggregate accounting evidence is stale");
  if (!selected.fully_accounted)
    throw std::logic_error(
        "TrialWaveFunction aggregate planned execution is not fully storage-accounted");
}

TrialWaveFunction::InlineBatchTopologyState TrialWaveFunction::captureBatchTopologyState(
    const std::shared_ptr<const BatchExecutionPlan>& plan) const
{
  InlineBatchTopologyState state;
  state.component_count          = Z.size();
  state.participant_fingerprint  = batchParticipantTopologyFingerprint(Z);
  state.engaged                  = true;

  if (!plan)
    return state;
  if (Z.size() != 1)
    throw std::logic_error(
        "A planned TrialWaveFunction topology must contain exactly one component");

  state.aggregate_plan = makeBatchExecutionParticipantPlan(
      plan, TRIAL_WAVEFUNCTION_MEMORY_PARTICIPANT_ID);
  state.sole_component_plan = makeBatchExecutionParticipantPlan(
      plan, batchParticipantId(0, *Z.front()));
  return state;
}

void TrialWaveFunction::validateRetainedBatchExecutionBinding() const
{
  if (!batch_execution_plan_)
  {
    if (!bound_batch_topology_.sameState({}))
      throw std::logic_error(
          "TrialWaveFunction retained participant bindings without a batch plan");
    return;
  }

  if (!bound_batch_topology_.engaged || bound_batch_topology_.component_count != Z.size() ||
      bound_batch_topology_.participant_fingerprint != batchParticipantTopologyFingerprint(Z))
    throw std::logic_error(
        "TrialWaveFunction component topology changed after batch plan binding");

  const InlineBatchTopologyState expected = captureBatchTopologyState(batch_execution_plan_);
  if (!bound_batch_topology_.sameState(expected))
    throw std::logic_error(
        "TrialWaveFunction participants no longer share the bound plan identity");
}

void TrialWaveFunction::bindBatchExecutionPlan(
    std::shared_ptr<const BatchExecutionPlan> plan)
{
  const bool changes_binding = plan.get() != batch_execution_plan_.get();
  if (changes_binding && multi_particle_proposal_pending_)
    throw std::logic_error(
        "Cannot change a TrialWaveFunction batch execution plan while a proposal is pending");
  if (changes_binding && plan && batch_execution_plan_)
    throw std::logic_error(
        "TrialWaveFunction nonempty batch replanning requires an explicit null-plan clear");

  if (resource_acquired_)
  {
    if (plan.get() == batch_execution_plan_.get())
    {
      validateRetainedBatchExecutionBinding();
      const InlineBatchTopologyState current = captureBatchTopologyState(batch_execution_plan_);
      if (acquired_batch_topology_.sameState(current))
        return;
    }
    throw std::logic_error(
        "Cannot rebind a TrialWaveFunction batch execution plan while resources are acquired");
  }

  const BatchExecutionParticipantPlan aggregate_plan =
      makeBatchExecutionParticipantPlan(
          plan, TRIAL_WAVEFUNCTION_MEMORY_PARTICIPANT_ID);
  validateAggregateBatchExecutionPlanBinding(aggregate_plan);

  // A nonempty aggregate plan is structurally gated to C==1 before its child
  // view is formed.  A null plan still visits every legacy component so stale
  // child state is cleared without retaining an arbitrary-size ID array.
  const InlineBatchTopologyState candidate = plan ? captureBatchTopologyState(plan)
                                                   : InlineBatchTopologyState{};
  if (plan)
    Z.front()->validateBatchExecutionPlanBinding(candidate.sole_component_plan);
  else
    for (const auto& component : Z)
      component->validateBatchExecutionPlanBinding({});

  static_assert(std::is_nothrow_copy_assignable_v<InlineBatchTopologyState>);
  static_assert(noexcept(std::declval<WaveFunctionComponent&>().bindBatchExecutionPlan(
      std::declval<BatchExecutionParticipantPlan>())));

  // G/L are accepted physical wavefunction state and must survive null binding.
  // Retain the exact proposed arrays as intrinsic fixed clone state as well;
  // they are reusable by a same-shape replan and cannot alias a pending move
  // because plan changes were rejected above.
  if (changes_binding)
  {
    prepared_aggregate_batch_execution_plan_ = {};
    prepared_aggregate_accepted_gradient_data_  = nullptr;
    prepared_aggregate_accepted_laplacian_data_ = nullptr;
    prepared_aggregate_proposed_gradient_data_  = nullptr;
    prepared_aggregate_proposed_laplacian_data_ = nullptr;
  }
  bound_batch_topology_ = candidate;
  if (plan)
    Z.front()->bindBatchExecutionPlan(candidate.sole_component_plan);
  else
    for (const auto& component : Z)
      component->bindBatchExecutionPlan({});
  batch_execution_plan_ = std::move(plan);
}

void TrialWaveFunction::prepareBatchExecutionClone(
    const BatchExecutionParticipantPlan& aggregate_plan)
{
  if (resource_acquired_)
    throw std::logic_error(
        "Cannot prepare TrialWaveFunction clone storage while resources are acquired");
  if (multi_particle_proposal_pending_)
    throw std::logic_error(
        "Cannot prepare TrialWaveFunction clone storage while a proposal is pending");
  if (!aggregate_plan || !batch_execution_plan_ ||
      !bound_batch_topology_.aggregate_plan.sameBinding(aggregate_plan))
    throw std::logic_error(
        "TrialWaveFunction clone preparation received the wrong aggregate participant plan");

  validateRetainedBatchExecutionBinding();
  if (Z.size() != 1)
    throw std::logic_error(
        "Planned TrialWaveFunction clone preparation requires exactly one component");

  const std::size_t particle_count = aggregate_plan.plan().particleCount();
  if (particle_count == 0)
    throw std::invalid_argument(
        "TrialWaveFunction clone preparation requires a positive particle count");

  const TrialWaveFunctionCloneStorageRequirement expected =
      trialWaveFunctionCloneStorageRequirement(
          1, particle_count, makeTrialWaveFunctionMemoryTypeSizesForBuild());

  // Measure allocation capacity rather than logical size: Ohmms vectors retain
  // high water when shrunk, and attached storage is not an aggregate-owned byte.
  const auto exact_storage_bytes = [particle_count](const auto& storage,
                                                     std::size_t element_size,
                                                     std::string_view description) {
    if (storage.isAttached())
      throw std::logic_error(
          std::string("TrialWaveFunction prepared ") + std::string(description) +
          " must own its storage");
    if (storage.size() != particle_count || storage.capacity() != particle_count)
      throw std::length_error(
          std::string("TrialWaveFunction prepared ") + std::string(description) +
          " does not have the exact admitted particle capacity");
    return checkedBatchMemoryMultiply(
        storage.capacity(), element_size,
        std::string("TWF prepared ") + std::string(description) + " bytes");
  };
  const auto validate_exact_storage = [&]() {
    const std::size_t accepted_gradient_bytes = exact_storage_bytes(
        G, sizeof(ParticleSet::ParticleGradient::value_type), "accepted gradient storage");
    const std::size_t proposed_gradient_bytes = exact_storage_bytes(
        multi_particle_proposed_gradient_,
        sizeof(ParticleSet::ParticleGradient::value_type),
        "proposed gradient storage");
    const std::size_t accepted_laplacian_bytes = exact_storage_bytes(
        L, sizeof(ParticleSet::ParticleLaplacian::value_type),
        "accepted Laplacian storage");
    const std::size_t proposed_laplacian_bytes = exact_storage_bytes(
        multi_particle_proposed_laplacian_,
        sizeof(ParticleSet::ParticleLaplacian::value_type),
        "proposed Laplacian storage");
    if (accepted_gradient_bytes != expected.accepted_gradients ||
        proposed_gradient_bytes != expected.proposed_gradients ||
        accepted_laplacian_bytes != expected.accepted_laplacians ||
        proposed_laplacian_bytes != expected.proposed_laplacians)
      throw std::length_error(
          "TrialWaveFunction prepared aggregate clone state differs from its admitted storage");
  };

  if (prepared_aggregate_batch_execution_plan_)
  {
    if (!prepared_aggregate_batch_execution_plan_.sameBinding(aggregate_plan))
      throw std::logic_error(
          "Cannot replace a prepared TrialWaveFunction aggregate clone plan");
    validate_exact_storage();
    if (G.data() != prepared_aggregate_accepted_gradient_data_ ||
        L.data() != prepared_aggregate_accepted_laplacian_data_ ||
        multi_particle_proposed_gradient_.data() !=
            prepared_aggregate_proposed_gradient_data_ ||
        multi_particle_proposed_laplacian_.data() !=
            prepared_aggregate_proposed_laplacian_data_)
      throw std::logic_error(
          "TrialWaveFunction prepared aggregate clone allocation identity changed");
    return;
  }

  // Validate the complete binding before allowing the first component to grow
  // clone-local storage.  Individual component preparation owns its retry
  // contract; bounded aggregate high water remains reusable on a child failure.
  validateAggregateBatchExecutionPlanBinding(aggregate_plan);
  Z.front()->validateBatchExecutionPlanBinding(bound_batch_topology_.sole_component_plan);

  // Existing accepted state may already hold a physical configuration.  It is
  // safe to allocate an empty array, but a nonempty wrong extent or retained
  // excess capacity cannot be canonicalized without risking those values.
  const auto validate_accepted_storage = [particle_count](const auto& storage,
                                                           std::string_view description) {
    if (storage.size() != 0 && storage.size() != particle_count)
      throw std::logic_error(
          std::string("TrialWaveFunction accepted ") + std::string(description) +
          " has the wrong particle extent");
    if (storage.size() == particle_count && storage.capacity() != particle_count)
      throw std::length_error(
          std::string("TrialWaveFunction accepted ") + std::string(description) +
          " retains storage beyond the admitted particle extent");
  };
  validate_accepted_storage(G, "gradient storage");
  validate_accepted_storage(L, "Laplacian storage");
  if (G.isAttached() || L.isAttached() ||
      multi_particle_proposed_gradient_.isAttached() ||
      multi_particle_proposed_laplacian_.isAttached())
    throw std::logic_error(
        "TrialWaveFunction aggregate clone preparation requires owned spatial storage");

  // Ohmms Vector cannot publish an off-side candidate allocation.  Empty or
  // retry-retained selected high water is therefore materialized in place;
  // the aggregate marker remains empty until every child also succeeds.
  if (G.size() == 0)
  {
    G.free();
    G.resize(particle_count);
  }
  if (L.size() == 0)
  {
    L.free();
    L.resize(particle_count);
  }
  if (multi_particle_proposed_gradient_.size() != particle_count ||
      multi_particle_proposed_gradient_.capacity() != particle_count)
  {
    multi_particle_proposed_gradient_.free();
    multi_particle_proposed_gradient_.resize(particle_count);
  }
  if (multi_particle_proposed_laplacian_.size() != particle_count ||
      multi_particle_proposed_laplacian_.capacity() != particle_count)
  {
    multi_particle_proposed_laplacian_.free();
    multi_particle_proposed_laplacian_.resize(particle_count);
  }

  validate_exact_storage();

  Z.front()->prepareBatchExecutionClone(bound_batch_topology_.sole_component_plan);
  static_assert(std::is_nothrow_copy_assignable_v<BatchExecutionParticipantPlan>);
  prepared_aggregate_accepted_gradient_data_  = G.data();
  prepared_aggregate_accepted_laplacian_data_ = L.data();
  prepared_aggregate_proposed_gradient_data_  = multi_particle_proposed_gradient_.data();
  prepared_aggregate_proposed_laplacian_data_ = multi_particle_proposed_laplacian_.data();
  // The participant view is the validity marker and is deliberately published last.
  prepared_aggregate_batch_execution_plan_ = aggregate_plan;
}

void TrialWaveFunction::validatePreparedBatchExecutionClone(
    const BatchExecutionParticipantPlan& aggregate_plan) const
{
  if (!aggregate_plan ||
      !prepared_aggregate_batch_execution_plan_.sameBinding(aggregate_plan))
    throw std::logic_error(
        "TrialWaveFunction aggregate clone is not prepared for the active plan");
  if (multi_particle_proposal_pending_)
    throw std::logic_error(
        "TrialWaveFunction planned acquisition cannot overlap a pending proposal");

  const std::size_t particle_count = aggregate_plan.plan().particleCount();
  const auto validate_storage = [particle_count](const auto& storage,
                                                  const void* identity,
                                                  std::string_view name) {
    if (storage.isAttached() || storage.size() != particle_count ||
        storage.capacity() != particle_count || storage.data() != identity)
      throw std::logic_error(std::string("TrialWaveFunction prepared ") +
                             std::string(name) +
                             " allocation changed before resource acquisition");
  };
  validate_storage(G, prepared_aggregate_accepted_gradient_data_,
                   "accepted gradient");
  validate_storage(L, prepared_aggregate_accepted_laplacian_data_,
                   "accepted Laplacian");
  validate_storage(multi_particle_proposed_gradient_,
                   prepared_aggregate_proposed_gradient_data_,
                   "proposed gradient");
  validate_storage(multi_particle_proposed_laplacian_,
                   prepared_aggregate_proposed_laplacian_data_,
                   "proposed Laplacian");
}

void TrialWaveFunction::prepareBatchExecutionClones()
{
  if (!batch_execution_plan_)
    return;
  prepareBatchExecutionClone(bound_batch_topology_.aggregate_plan);
}

const SPOSet& TrialWaveFunction::getSPOSet(const std::string& name) const
{
  auto spoit = spomap_->find(name);
  if (spoit == spomap_->end())
    throw std::runtime_error("SPOSet " + name + " cannot be found!");
  return *spoit->second;
}

RefVector<SlaterDet> TrialWaveFunction::findSD() const
{
  RefVector<SlaterDet> refs;
  for (auto& component : Z)
    if (auto* comp_ptr = dynamic_cast<SlaterDet*>(component.get()); comp_ptr)
      refs.push_back(*comp_ptr);
  return refs;
}

RefVector<MultiSlaterDetTableMethod> TrialWaveFunction::findMSD() const
{
  RefVector<MultiSlaterDetTableMethod> refs;
  for (auto& component : Z)
    if (auto* comp_ptr = dynamic_cast<MultiSlaterDetTableMethod*>(component.get()); comp_ptr)
      refs.push_back(*comp_ptr);
  return refs;
}

/** return log(|psi|)
*
* PhaseValue is the phase for the complex wave function
*/
TrialWaveFunction::RealType TrialWaveFunction::evaluateLog(ParticleSet& P)
{
  ScopedTimer local_timer(TWF_timers_[RECOMPUTE_TIMER]);
  P.G = 0.0;
  P.L = 0.0;
  LogValue logpsi(0.0);
  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer z_timer(WFC_timers_[RECOMPUTE_TIMER + TIMER_SKIP * i]);
#ifndef NDEBUG
    // Best way I've found yet to quickly see if WFC made it over the wire successfully
    auto subterm = Z[i]->evaluateLog(P, P.G, P.L);
    // std::cerr << "evaluate log Z element:" <<  i << "  value: " << subterm << '\n';
    logpsi += subterm;
#else
    logpsi += Z[i]->evaluateLog(P, P.G, P.L);
#endif
  }

  G = P.G;
  L = P.L;

  log_real_  = std::real(logpsi);
  PhaseValue = std::imag(logpsi);
  return log_real_;
}

void TrialWaveFunction::mw_evaluateLog(const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                       const RefVectorWithLeader<ParticleSet>& p_list)
{
  auto& wf_leader = wf_list.getLeader();
  auto& p_leader  = p_list.getLeader();
  ScopedTimer local_timer(wf_leader.TWF_timers_[RECOMPUTE_TIMER]);

  constexpr RealType czero(0);
  const auto g_list(TrialWaveFunction::extractGRefList(wf_list));
  const auto l_list(TrialWaveFunction::extractLRefList(wf_list));

  // due to historic design issue, ParticleSet holds G and L instead of TrialWaveFunction.
  // TrialWaveFunction now also holds G and L to move forward but they need to be copied to P.G and P.L
  // to be compatible with legacy use pattern.
  const int num_particles = p_leader.getTotalNum();
  auto initGandL          = [num_particles, czero](TrialWaveFunction& twf, ParticleSet::ParticleGradient& grad,
                                                   ParticleSet::ParticleLaplacian& lapl) {
    grad.resize(num_particles);
    lapl.resize(num_particles);
    grad           = czero;
    lapl           = czero;
    twf.log_real_  = czero;
    twf.PhaseValue = czero;
  };
  for (int iw = 0; iw < wf_list.size(); iw++)
    initGandL(wf_list[iw], g_list[iw], l_list[iw]);

  auto& wavefunction_components = wf_leader.Z;
  const int num_wfc             = wf_leader.Z.size();
  for (int i = 0; i < num_wfc; ++i)
  {
    ScopedTimer z_timer(wf_leader.WFC_timers_[RECOMPUTE_TIMER + TIMER_SKIP * i]);
    const auto wfc_list(extractWFCRefList(wf_list, i));
    wavefunction_components[i]->mw_evaluateLog(wfc_list, p_list, g_list, l_list);
  }

  for (int iw = 0; iw < wf_list.size(); iw++)
  {
    ParticleSet& pset      = p_list[iw];
    TrialWaveFunction& twf = wf_list[iw];

    for (int i = 0; i < num_wfc; ++i)
    {
      twf.log_real_ += std::real(twf.Z[i]->get_log_value());
      twf.PhaseValue += std::imag(twf.Z[i]->get_log_value());
    }

    // Ye: temporal workaround to have P.G/L always defined.
    // remove when KineticEnergy use WF.G/L instead of P.G/L
    pset.G = twf.G;
    pset.L = twf.L;
  }
}

void TrialWaveFunction::recompute(const ParticleSet& P)
{
  ScopedTimer local_timer(TWF_timers_[RECOMPUTE_TIMER]);
  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer z_timer(WFC_timers_[RECOMPUTE_TIMER + TIMER_SKIP * i]);
    Z[i]->recompute(P);
  }
}

void TrialWaveFunction::mw_recompute(const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                     const RefVectorWithLeader<ParticleSet>& p_list,
                                     const std::vector<bool>& recompute)
{
  auto& wf_leader = wf_list.getLeader();
  auto& p_leader  = p_list.getLeader();
  ScopedTimer local_timer(wf_leader.TWF_timers_[RECOMPUTE_TIMER]);

  auto& wavefunction_components = wf_leader.Z;
  const int num_wfc             = wf_leader.Z.size();
  for (int i = 0; i < num_wfc; ++i)
  {
    ScopedTimer z_timer(wf_leader.WFC_timers_[RECOMPUTE_TIMER + TIMER_SKIP * i]);
    const auto wfc_list(extractWFCRefList(wf_list, i));
    wavefunction_components[i]->mw_recompute(wfc_list, p_list, recompute);
  }
}

TrialWaveFunction::RealType TrialWaveFunction::evaluateDeltaLog(ParticleSet& P, bool recomputeall)
{
  ScopedTimer local_timer(TWF_timers_[RECOMPUTE_TIMER]);
  P.G = 0.0;
  P.L = 0.0;
  LogValue logpsi(0.0);
  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer z_timer(WFC_timers_[RECOMPUTE_TIMER + TIMER_SKIP * i]);
    if (Z[i]->isOptimizable())
      logpsi += Z[i]->evaluateLog(P, P.G, P.L);
  }
  log_real_  = std::real(logpsi);
  PhaseValue = std::imag(logpsi);

  //In case we need to recompute orbitals, initialize dummy vectors for G and L.
  //evaluateLog dumps into these variables, and logPsi contribution is discarded.
  //Only called for non-optimizable orbitals.
  if (recomputeall)
  {
    ParticleSet::ParticleGradient dummyG(P.G);
    ParticleSet::ParticleLaplacian dummyL(P.L);

    for (int i = 0; i < Z.size(); ++i)
    {
      //update orbitals if its not flagged optimizable, AND recomputeall is true
      if (!Z[i]->isOptimizable())
        Z[i]->evaluateLog(P, dummyG, dummyL);
    }
  }
  return log_real_;
}

void TrialWaveFunction::evaluateDeltaLogSetup(ParticleSet& P,
                                              RealType& logpsi_fixed_r,
                                              RealType& logpsi_opt_r,
                                              ParticleSet::ParticleGradient& fixedG,
                                              ParticleSet::ParticleLaplacian& fixedL)
{
  ScopedTimer local_timer(TWF_timers_[RECOMPUTE_TIMER]);
  P.G    = 0.0;
  P.L    = 0.0;
  fixedL = 0.0;
  fixedG = 0.0;
  LogValue logpsi_fixed(0.0);
  LogValue logpsi_opt(0.0);

  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer z_timer(WFC_timers_[RECOMPUTE_TIMER + TIMER_SKIP * i]);
    if (Z[i]->isOptimizable())
      logpsi_opt += Z[i]->evaluateLog(P, P.G, P.L);
    else
      logpsi_fixed += Z[i]->evaluateLog(P, fixedG, fixedL);
  }
  P.G += fixedG;
  P.L += fixedL;
  convertToReal(logpsi_fixed, logpsi_fixed_r);
  convertToReal(logpsi_opt, logpsi_opt_r);
}


void TrialWaveFunction::mw_evaluateDeltaLogSetup(const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                                 const RefVectorWithLeader<ParticleSet>& p_list,
                                                 std::vector<RealType>& logpsi_fixed_list,
                                                 std::vector<RealType>& logpsi_opt_list,
                                                 RefVector<ParticleSet::ParticleGradient>& fixedG_list,
                                                 RefVector<ParticleSet::ParticleLaplacian>& fixedL_list)
{
  auto& wf_leader = wf_list.getLeader();
  auto& p_leader  = p_list.getLeader();
  ScopedTimer local_timer(wf_leader.TWF_timers_[RECOMPUTE_TIMER]);
  constexpr RealType czero(0);
  const int num_particles = p_leader.getTotalNum();
  const auto g_list(TrialWaveFunction::extractGRefList(wf_list));
  const auto l_list(TrialWaveFunction::extractLRefList(wf_list));

  auto initGandL = [num_particles, czero](TrialWaveFunction& twf, ParticleSet::ParticleGradient& grad,
                                          ParticleSet::ParticleLaplacian& lapl) {
    grad.resize(num_particles);
    lapl.resize(num_particles);
    grad           = czero;
    lapl           = czero;
    twf.log_real_  = czero;
    twf.PhaseValue = czero;
  };
  for (int iw = 0; iw < wf_list.size(); iw++)
    initGandL(wf_list[iw], g_list[iw], l_list[iw]);
  auto& wavefunction_components = wf_leader.Z;
  const int num_wfc             = wf_leader.Z.size();
  for (int i = 0; i < num_wfc; ++i)
  {
    ScopedTimer z_timer(wf_leader.WFC_timers_[RECOMPUTE_TIMER + TIMER_SKIP * i]);
    const auto wfc_list(extractWFCRefList(wf_list, i));
    if (wavefunction_components[i]->isOptimizable())
    {
      wavefunction_components[i]->mw_evaluateLog(wfc_list, p_list, g_list, l_list);
      for (int iw = 0; iw < wf_list.size(); iw++)
        logpsi_opt_list[iw] += std::real(wfc_list[iw].get_log_value());
    }
    else
    {
      wavefunction_components[i]->mw_evaluateLog(wfc_list, p_list, fixedG_list, fixedL_list);
      for (int iw = 0; iw < wf_list.size(); iw++)
        logpsi_fixed_list[iw] += std::real(wfc_list[iw].get_log_value());
    }
  }

  // Temporary workaround to have P.G/L always defined.
  // remove when KineticEnergy use WF.G/L instead of P.G/L
  auto addAndCopyToP = [](ParticleSet& pset, TrialWaveFunction& twf, ParticleSet::ParticleGradient& grad,
                          ParticleSet::ParticleLaplacian& lapl) {
    pset.G = twf.G + grad;
    pset.L = twf.L + lapl;
  };
  for (int iw = 0; iw < wf_list.size(); iw++)
    addAndCopyToP(p_list[iw], wf_list[iw], fixedG_list[iw], fixedL_list[iw]);
}


void TrialWaveFunction::mw_evaluateDeltaLog(const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                            const RefVectorWithLeader<ParticleSet>& p_list,
                                            std::vector<RealType>& logpsi_list,
                                            RefVector<ParticleSet::ParticleGradient>& dummyG_list,
                                            RefVector<ParticleSet::ParticleLaplacian>& dummyL_list,
                                            bool recompute)
{
  auto& p_leader  = p_list.getLeader();
  auto& wf_leader = wf_list.getLeader();
  ScopedTimer local_timer(wf_leader.TWF_timers_[RECOMPUTE_TIMER]);
  constexpr RealType czero(0);
  int num_particles = p_leader.getTotalNum();
  const auto g_list(TrialWaveFunction::extractGRefList(wf_list));
  const auto l_list(TrialWaveFunction::extractLRefList(wf_list));

  // Initialize various members of the wavefunction, grad, and laplacian
  auto initGandL = [num_particles, czero](TrialWaveFunction& twf, ParticleSet::ParticleGradient& grad,
                                          ParticleSet::ParticleLaplacian& lapl) {
    grad.resize(num_particles);
    lapl.resize(num_particles);
    grad           = czero;
    lapl           = czero;
    twf.log_real_  = czero;
    twf.PhaseValue = czero;
  };
  for (int iw = 0; iw < wf_list.size(); iw++)
    initGandL(wf_list[iw], g_list[iw], l_list[iw]);

  // Get wavefunction components (assumed the same for every WF in the list)
  auto& wavefunction_components = wf_leader.Z;
  const int num_wfc             = wf_leader.Z.size();

  // Loop over the wavefunction components
  for (int i = 0; i < num_wfc; ++i)
    if (wavefunction_components[i]->isOptimizable())
    {
      ScopedTimer z_timer(wf_leader.WFC_timers_[RECOMPUTE_TIMER + TIMER_SKIP * i]);
      const auto wfc_list(extractWFCRefList(wf_list, i));
      wavefunction_components[i]->mw_evaluateLog(wfc_list, p_list, g_list, l_list);
      for (int iw = 0; iw < wf_list.size(); iw++)
        logpsi_list[iw] += std::real(wfc_list[iw].get_log_value());
    }

  // Temporary workaround to have P.G/L always defined.
  // remove when KineticEnergy use WF.G/L instead of P.G/L
  auto copyToP = [](ParticleSet& pset, TrialWaveFunction& twf) {
    pset.G = twf.G;
    pset.L = twf.L;
  };
  for (int iw = 0; iw < wf_list.size(); iw++)
    copyToP(p_list[iw], wf_list[iw]);

  // Recompute is usually used to prepare the wavefunction for NLPP derivatives.
  // (e.g compute the matrix inverse for determinants)
  // Call mw_evaluateLog for the wavefunction components that were skipped previously.
  // Ignore logPsi, G and L.
  if (recompute)
    for (int i = 0; i < num_wfc; ++i)
      if (!wavefunction_components[i]->isOptimizable())
      {
        ScopedTimer z_timer(wf_leader.WFC_timers_[RECOMPUTE_TIMER + TIMER_SKIP * i]);
        const auto wfc_list(extractWFCRefList(wf_list, i));
        wavefunction_components[i]->mw_evaluateLog(wfc_list, p_list, dummyG_list, dummyL_list);
      }
}


/*void TrialWaveFunction::evaluateHessian(ParticleSet & P, int iat, HessType& grad_grad_psi)
{
  std::vector<WaveFunctionComponent*>::iterator it(Z.begin());
  std::vector<WaveFunctionComponent*>::iterator it_end(Z.end());
  
  grad_grad_psi=0.0;
  
  for (; it!=it_end; ++it)
  {	
	  HessType tmp_hess;
	  (*it)->evaluateHessian(P, iat, tmp_hess);
	  grad_grad_psi+=tmp_hess;
  }
}*/

void TrialWaveFunction::evaluateHessian(ParticleSet& P, HessVector& grad_grad_psi)
{
  grad_grad_psi.resize(P.getTotalNum());

  for (int i = 0; i < Z.size(); i++)
  {
    HessVector tmp_hess(grad_grad_psi);
    tmp_hess = 0.0;
    Z[i]->evaluateHessian(P, tmp_hess);
    grad_grad_psi += tmp_hess;
    //  app_log()<<"TrialWavefunction::tmp_hess = "<<tmp_hess<< std::endl;
    //  app_log()<< std::endl<< std::endl;
  }
  // app_log()<<" TrialWavefunction::Hessian = "<<grad_grad_psi<< std::endl;
}

TrialWaveFunction::ValueType TrialWaveFunction::calcRatio(ParticleSet& P, int iat, ComputeType ct)
{
  ScopedTimer local_timer(TWF_timers_[V_TIMER]);
  PsiValue r(1.0);
  for (int i = 0; i < Z.size(); i++)
    if (ct == ComputeType::ALL || (Z[i]->isFermionic() && ct == ComputeType::FERMIONIC) ||
        (!Z[i]->isFermionic() && ct == ComputeType::NONFERMIONIC))
    {
      ScopedTimer z_timer(WFC_timers_[V_TIMER + TIMER_SKIP * i]);
      r *= Z[i]->ratio(P, iat);
    }

  NaNguard::checkOneParticleRatio(r, "TWF::calcRatio at particle " + std::to_string(iat));
  return static_cast<ValueType>(r);
}

void TrialWaveFunction::mw_calcRatio(const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                     const RefVectorWithLeader<ParticleSet>& p_list,
                                     int iat,
                                     std::vector<PsiValue>& ratios,
                                     ComputeType ct)
{
  const int num_wf = wf_list.size();
  ratios.resize(num_wf);
  std::fill(ratios.begin(), ratios.end(), PsiValue(1));

  auto& wf_leader = wf_list.getLeader();
  ScopedTimer local_timer(wf_leader.TWF_timers_[V_TIMER]);
  const int num_wfc             = wf_leader.Z.size();
  auto& wavefunction_components = wf_leader.Z;

  std::vector<PsiValue> ratios_z(num_wf);
  for (int i = 0; i < num_wfc; i++)
  {
    if (ct == ComputeType::ALL || (wavefunction_components[i]->isFermionic() && ct == ComputeType::FERMIONIC) ||
        (!wavefunction_components[i]->isFermionic() && ct == ComputeType::NONFERMIONIC))
    {
      ScopedTimer z_timer(wf_leader.WFC_timers_[V_TIMER + TIMER_SKIP * i]);
      const auto wfc_list(extractWFCRefList(wf_list, i));
      wavefunction_components[i]->mw_calcRatio(wfc_list, p_list, iat, ratios_z);
      for (int iw = 0; iw < wf_list.size(); iw++)
        ratios[iw] *= ratios_z[iw];
    }
  }

  for (int iw = 0; iw < wf_list.size(); iw++)
  {
    NaNguard::checkOneParticleRatio(ratios[iw], "TWF::mw_calcRatio at particle " + std::to_string(iat));
    wf_list[iw].PhaseDiff = std::arg(ratios[iw]);
  }
}

void TrialWaveFunction::prepareGroup(ParticleSet& P, int ig)
{
  ScopedTimer local_timer(TWF_timers_[PREPAREGROUP_TIMER]);
  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer z_timer(WFC_timers_[PREPAREGROUP_TIMER + TIMER_SKIP * i]);
    Z[i]->prepareGroup(P, ig);
  }
}

void TrialWaveFunction::mw_prepareGroup(const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                        const RefVectorWithLeader<ParticleSet>& p_list,
                                        int ig)
{
  auto& wf_leader = wf_list.getLeader();
  ScopedTimer local_timer(wf_leader.TWF_timers_[PREPAREGROUP_TIMER]);
  const int num_wfc             = wf_leader.Z.size();
  auto& wavefunction_components = wf_leader.Z;

  for (int i = 0; i < num_wfc; i++)
  {
    ScopedTimer z_timer(wf_leader.WFC_timers_[PREPAREGROUP_TIMER + TIMER_SKIP * i]);
    const auto wfc_list(extractWFCRefList(wf_list, i));
    wavefunction_components[i]->mw_prepareGroup(wfc_list, p_list, ig);
  }
}

TrialWaveFunction::GradType TrialWaveFunction::evalGrad(ParticleSet& P, int iat)
{
  ScopedTimer local_timer(TWF_timers_[VGL_TIMER]);
  GradType grad_iat;
  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer z_timer(WFC_timers_[VGL_TIMER + TIMER_SKIP * i]);
    grad_iat += Z[i]->evalGrad(P, iat);
  }
  NaNguard::checkOneParticleGradients(grad_iat, "TWF::evalGrad at particle " + std::to_string(iat));
  return grad_iat;
}

TrialWaveFunction::GradType TrialWaveFunction::evalGradWithSpin(ParticleSet& P, int iat, ComplexType& spingrad)
{
  ScopedTimer local_timer(TWF_timers_[VGL_TIMER]);
  GradType grad_iat;
  spingrad = 0;
  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer z_timer(WFC_timers_[VGL_TIMER + TIMER_SKIP * i]);
    grad_iat += Z[i]->evalGradWithSpin(P, iat, spingrad);
  }
  NaNguard::checkOneParticleGradients(grad_iat, "TWF::evalGradWithSpin at particle " + std::to_string(iat));
  return grad_iat;
}

template<CoordsType CT>
void TrialWaveFunction::mw_evalGrad(const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                    const RefVectorWithLeader<ParticleSet>& p_list,
                                    int iat,
                                    TWFGrads<CT>& grads)
{
  const int num_wf = wf_list.size();
  grads            = TWFGrads<CT>(num_wf); //ensure elements are set to zero

  auto& wf_leader = wf_list.getLeader();
  ScopedTimer local_timer(wf_leader.TWF_timers_[VGL_TIMER]);
  // Right now mw_evalGrad can only be called through an concrete instance of a wavefunctioncomponent
  const int num_wfc             = wf_leader.Z.size();
  auto& wavefunction_components = wf_leader.Z;

  TWFGrads<CT> grads_z(num_wf);
  for (int i = 0; i < num_wfc; i++)
  {
    ScopedTimer localtimer(wf_leader.WFC_timers_[VGL_TIMER + TIMER_SKIP * i]);
    const auto wfc_list(extractWFCRefList(wf_list, i));
    wavefunction_components[i]->mw_evalGrad(wfc_list, p_list, iat, grads_z);
    grads += grads_z;
  }

  for (const GradType& grads : grads.grads_positions)
    NaNguard::checkOneParticleGradients(grads, "TWF::mw_evalGrad at particle " + std::to_string(iat));
}

// Evaluates the gradient w.r.t. to the source of the Laplacian
// w.r.t. to the electrons of the wave function.
TrialWaveFunction::GradType TrialWaveFunction::evalGradSource(ParticleSet& P, ParticleSet& source, int iat)
{
  GradType grad_iat = GradType();
  for (int i = 0; i < Z.size(); ++i)
    grad_iat += Z[i]->evalGradSource(P, source, iat);
  return grad_iat;
}

TrialWaveFunction::GradType TrialWaveFunction::evalGradSource(
    ParticleSet& P,
    ParticleSet& source,
    int iat,
    TinyVector<ParticleSet::ParticleGradient, OHMMS_DIM>& grad_grad,
    TinyVector<ParticleSet::ParticleLaplacian, OHMMS_DIM>& lapl_grad)
{
  GradType grad_iat = GradType();
  for (int dim = 0; dim < OHMMS_DIM; dim++)
    for (int i = 0; i < grad_grad[0].size(); i++)
    {
      grad_grad[dim][i] = GradType();
      lapl_grad[dim][i] = 0.0;
    }
  for (int i = 0; i < Z.size(); ++i)
    grad_iat += Z[i]->evalGradSource(P, source, iat, grad_grad, lapl_grad);
  return grad_iat;
}

TrialWaveFunction::ValueType TrialWaveFunction::calcRatioGrad(ParticleSet& P, int iat, GradType& grad_iat)
{
  ScopedTimer local_timer(TWF_timers_[VGL_TIMER]);
  grad_iat = 0.0;
  PsiValue r(1.0);
  if (use_tasking_)
  {
    std::vector<GradType> grad_components(Z.size(), GradType(0.0));
    std::vector<PsiValue> ratio_components(Z.size(), 0.0);
    PRAGMA_OMP_TASKLOOP("omp taskloop default(shared)")
    for (int i = 0; i < Z.size(); ++i)
    {
      ScopedTimer z_timer(WFC_timers_[VGL_TIMER + TIMER_SKIP * i]);
      ratio_components[i] = Z[i]->ratioGrad(P, iat, grad_components[i]);
    }

    for (int i = 0; i < Z.size(); ++i)
    {
      grad_iat += grad_components[i];
      r *= ratio_components[i];
    }
  }
  else
    for (int i = 0; i < Z.size(); ++i)
    {
      ScopedTimer z_timer(WFC_timers_[VGL_TIMER + TIMER_SKIP * i]);
      r *= Z[i]->ratioGrad(P, iat, grad_iat);
    }

  NaNguard::checkOneParticleRatio(r, "TWF::calcRatioGrad at particle " + std::to_string(iat));
  if (r != PsiValue(0)) // grad_iat is meaningful only when r is strictly non-zero
    NaNguard::checkOneParticleGradients(grad_iat, "TWF::calcRatioGrad at particle " + std::to_string(iat));
  LogValue logratio = convertValueToLog(r);
  PhaseDiff         = std::imag(logratio);
  return static_cast<ValueType>(r);
}

TrialWaveFunction::ValueType TrialWaveFunction::calcRatioGradWithSpin(ParticleSet& P,
                                                                      int iat,
                                                                      GradType& grad_iat,
                                                                      ComplexType& spingrad_iat)
{
  ScopedTimer local_timer(TWF_timers_[VGL_TIMER]);
  grad_iat     = 0.0;
  spingrad_iat = 0.0;
  PsiValue r(1.0);
  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer z_timer(WFC_timers_[VGL_TIMER + TIMER_SKIP * i]);
    r *= Z[i]->ratioGradWithSpin(P, iat, grad_iat, spingrad_iat);
  }

  NaNguard::checkOneParticleRatio(r, "TWF::calcRatioGradWithSpin at particle " + std::to_string(iat));
  if (r != PsiValue(0)) // grad_iat is meaningful only when r is strictly non-zero
    NaNguard::checkOneParticleGradients(grad_iat, "TWF::calcRatioGradWithSpin at particle " + std::to_string(iat));
  LogValue logratio = convertValueToLog(r);
  PhaseDiff         = std::imag(logratio);
  return static_cast<ValueType>(r);
}

template<CoordsType CT>
void TrialWaveFunction::mw_calcRatioGrad(const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                         const RefVectorWithLeader<ParticleSet>& p_list,
                                         int iat,
                                         std::vector<PsiValue>& ratios,
                                         TWFGrads<CT>& grad_new)
{
  const int num_wf = wf_list.size();
  ratios.resize(num_wf);
  std::fill(ratios.begin(), ratios.end(), PsiValue(1));
  grad_new = TWFGrads<CT>(num_wf);

  auto& wf_leader = wf_list.getLeader();
  ScopedTimer local_timer(wf_leader.TWF_timers_[VGL_TIMER]);
  const int num_wfc             = wf_leader.Z.size();
  auto& wavefunction_components = wf_leader.Z;

  if (wf_leader.use_tasking_)
  {
    std::vector<std::vector<PsiValue>> ratios_components(num_wfc, std::vector<PsiValue>(wf_list.size()));
    std::vector<TWFGrads<CT>> grads_components(num_wfc, TWFGrads<CT>(num_wf));
    PRAGMA_OMP_TASKLOOP("omp taskloop default(shared)")
    for (int i = 0; i < num_wfc; ++i)
    {
      ScopedTimer z_timer(wf_leader.WFC_timers_[VGL_TIMER + TIMER_SKIP * i]);
      const auto wfc_list(extractWFCRefList(wf_list, i));
      wavefunction_components[i]->mw_ratioGrad(wfc_list, p_list, iat, ratios_components[i], grads_components[i]);
    }

    for (int i = 0; i < num_wfc; ++i)
    {
      grad_new += grads_components[i];
      for (int iw = 0; iw < wf_list.size(); iw++)
        ratios[iw] *= ratios_components[i][iw];
    }
  }
  else
  {
    std::vector<PsiValue> ratios_z(wf_list.size());
    for (int i = 0; i < num_wfc; ++i)
    {
      ScopedTimer z_timer(wf_leader.WFC_timers_[VGL_TIMER + TIMER_SKIP * i]);
      const auto wfc_list(extractWFCRefList(wf_list, i));
      wavefunction_components[i]->mw_ratioGrad(wfc_list, p_list, iat, ratios_z, grad_new);
      for (int iw = 0; iw < wf_list.size(); iw++)
        ratios[iw] *= ratios_z[iw];
    }
  }

  for (int iw = 0; iw < wf_list.size(); iw++)
  {
    wf_list[iw].PhaseDiff = std::arg(ratios[iw]);
    NaNguard::checkOneParticleRatio(ratios[iw], "TWF::mw_calcRatioGrad at particle " + std::to_string(iat));
    if (ratios[iw] != PsiValue(0))
      NaNguard::checkOneParticleGradients(grad_new.grads_positions[iw],
                                          "TWF::mw_calcRatioGrad at particle " + std::to_string(iat));
  }
}

void TrialWaveFunction::printGL(ParticleSet::ParticleGradient& G, ParticleSet::ParticleLaplacian& L, std::string tag)
{
  std::ostringstream o;
  o << "---  reporting " << tag << std::endl << "  ---" << std::endl;
  for (int iat = 0; iat < L.size(); iat++)
    o << "index: " << std::fixed << iat << std::scientific << "   G: " << G[iat][0] << "  " << G[iat][1] << "  "
      << G[iat][2] << "   L: " << L[iat] << std::endl;
  o << "---  end  ---" << std::endl;
  std::cout << o.str();
}

/** restore to the original state
 * @param iat index of the particle with a trial move
 *
 * The proposed move of the iath particle is rejected.
 * All the temporary data should be restored to the state prior to the move.
 */
void TrialWaveFunction::rejectMove(int iat)
{
  for (int i = 0; i < Z.size(); i++)
    Z[i]->restore(iat);
  PhaseDiff = 0;
}

/** update the state with the new data
 * @param P ParticleSet
 * @param iat index of the particle with a trial move
 *
 * The proposed move of the iath particle is accepted.
 * All the temporary data should be incorporated so that the next move is valid.
 */
void TrialWaveFunction::acceptMove(ParticleSet& P, int iat, bool safe_to_delay)
{
  ScopedTimer local_timer(TWF_timers_[ACCEPT_TIMER]);
  PRAGMA_OMP_TASKLOOP("omp taskloop default(shared) if (use_tasking_)")
  for (int i = 0; i < Z.size(); i++)
  {
    ScopedTimer z_timer(WFC_timers_[ACCEPT_TIMER + TIMER_SKIP * i]);
    Z[i]->acceptMove(P, iat, safe_to_delay);
  }
  PhaseValue += PhaseDiff;
  PhaseDiff = 0.0;
  log_real_ = 0;
  for (int i = 0; i < Z.size(); i++)
    log_real_ += std::real(Z[i]->get_log_value());
}

void TrialWaveFunction::mw_accept_rejectMove(const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                             const RefVectorWithLeader<ParticleSet>& p_list,
                                             int iat,
                                             const std::vector<bool>& isAccepted,
                                             bool safe_to_delay)
{
  auto& wf_leader = wf_list.getLeader();
  ScopedTimer local_timer(wf_leader.TWF_timers_[ACCEPT_TIMER]);
  const int num_wfc             = wf_leader.Z.size();
  auto& wavefunction_components = wf_leader.Z;

  for (int iw = 0; iw < wf_list.size(); iw++)
    if (isAccepted[iw])
    {
      wf_list[iw].log_real_  = 0;
      wf_list[iw].PhaseValue = 0;
    }

  PRAGMA_OMP_TASKLOOP("omp taskloop default(shared) if (wf_leader.use_tasking_)")
  for (int i = 0; i < num_wfc; i++)
  {
    ScopedTimer z_timer(wf_leader.WFC_timers_[ACCEPT_TIMER + TIMER_SKIP * i]);
    const auto wfc_list(extractWFCRefList(wf_list, i));
    wavefunction_components[i]->mw_accept_rejectMove(wfc_list, p_list, iat, isAccepted, safe_to_delay);
    for (int iw = 0; iw < wf_list.size(); iw++)
      if (isAccepted[iw])
      {
        wf_list[iw].log_real_ += std::real(wfc_list[iw].get_log_value());
        wf_list[iw].PhaseValue += std::imag(wfc_list[iw].get_log_value());
      }
  }
}

bool TrialWaveFunction::supportsMultiParticleMoves() const noexcept
{
  return std::all_of(Z.begin(), Z.end(),
                     [](const auto& component) { return component->supportsMultiParticleMoves(); });
}

void TrialWaveFunction::mw_evaluateMultiParticleMove(
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const MCMultiParticleMoves<CoordsType::POS>& moves,
    std::vector<LogValue>& log_ratios)
{
  if (wf_list.size() != p_list.size() || moves.walkerCount() != wf_list.size())
    throw std::invalid_argument(
        "Selected-electron wavefunction proposal has inconsistent walker counts");
  if (log_ratios.size() != wf_list.size())
    throw std::invalid_argument(
        "Selected-electron wavefunction proposal has the wrong log-ratio count");
  moves.validateFor(p_list);

  const std::uint64_t proposal_fingerprint = moves.fingerprint();
  for (std::size_t walker = 0; walker < wf_list.size(); ++walker)
  {
    TrialWaveFunction& wavefunction = wf_list[walker];
    if (wavefunction.multi_particle_proposal_pending_)
      throw std::logic_error(
          "Cannot start a selected-electron wavefunction proposal before resolving the previous one");
    if (!wavefunction.supportsMultiParticleMoves())
      throw std::invalid_argument(
          "TrialWaveFunction contains a component without selected-electron move support");

    const std::size_t electron_count = p_list[walker].getTotalNum();
    wavefunction.multi_particle_proposed_gradient_.resize(electron_count);
    wavefunction.multi_particle_proposed_laplacian_.resize(electron_count);
    wavefunction.multi_particle_proposed_gradient_  = ValueType(0);
    wavefunction.multi_particle_proposed_laplacian_ = ValueType(0);
    wavefunction.multi_particle_proposed_log_ratio_ = LogValue(0);
    wavefunction.multi_particle_proposal_fingerprint_ = proposal_fingerprint;
    log_ratios[walker] = LogValue(0);
  }

  RefVector<ParticleSet::ParticleGradient> proposed_gradient_list;
  RefVector<ParticleSet::ParticleLaplacian> proposed_laplacian_list;
  proposed_gradient_list.reserve(wf_list.size());
  proposed_laplacian_list.reserve(wf_list.size());
  for (TrialWaveFunction& wavefunction : wf_list)
  {
    proposed_gradient_list.push_back(wavefunction.multi_particle_proposed_gradient_);
    proposed_laplacian_list.push_back(wavefunction.multi_particle_proposed_laplacian_);
  }

  auto& leader = wf_list.getLeader();
  std::size_t completed_components = 0;
  try
  {
    for (std::size_t component_index = 0; component_index < leader.Z.size(); ++component_index)
    {
      const auto component_list = extractWFCRefList(wf_list, component_index);
      std::vector<LogValue> component_log_ratios(wf_list.size(), LogValue(0));
      leader.Z[component_index]->mw_evaluateMultiParticleMove(
          component_list, p_list, moves, component_log_ratios,
          proposed_gradient_list, proposed_laplacian_list);
      for (std::size_t walker = 0; walker < wf_list.size(); ++walker)
        log_ratios[walker] += component_log_ratios[walker];
      ++completed_components;
    }
  }
  catch (...)
  {
    const std::vector<bool> reject_all(wf_list.size(), false);
    for (std::size_t component_index = 0; component_index < completed_components;
         ++component_index)
    {
      const auto component_list = extractWFCRefList(wf_list, component_index);
      leader.Z[component_index]->mw_accept_rejectMultiParticleMove(
          component_list, p_list, moves, reject_all);
    }
    for (TrialWaveFunction& wavefunction : wf_list)
    {
      wavefunction.multi_particle_proposal_fingerprint_ = 0;
      wavefunction.multi_particle_proposed_log_ratio_    = LogValue(0);
    }
    throw;
  }

  for (std::size_t walker = 0; walker < wf_list.size(); ++walker)
  {
    wf_list[walker].multi_particle_proposed_log_ratio_ = log_ratios[walker];
    wf_list[walker].multi_particle_proposal_pending_   = true;
  }
}

void TrialWaveFunction::mw_accept_rejectMultiParticleMove(
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const MCMultiParticleMoves<CoordsType::POS>& moves,
    const std::vector<bool>& accepted)
{
  if (wf_list.size() != p_list.size() || moves.walkerCount() != wf_list.size() ||
      accepted.size() != wf_list.size())
    throw std::invalid_argument(
        "Selected-electron wavefunction resolution has inconsistent walker counts");
  moves.validateFor(p_list);
  const std::uint64_t proposal_fingerprint = moves.fingerprint();
  for (const TrialWaveFunction& wavefunction : wf_list)
    if (!wavefunction.multi_particle_proposal_pending_ ||
        wavefunction.multi_particle_proposal_fingerprint_ != proposal_fingerprint)
      throw std::logic_error(
          "Selected-electron wavefunction resolution does not match the pending proposal");

  auto& leader = wf_list.getLeader();
  for (std::size_t component_index = 0; component_index < leader.Z.size(); ++component_index)
  {
    const auto component_list = extractWFCRefList(wf_list, component_index);
    leader.Z[component_index]->mw_accept_rejectMultiParticleMove(
        component_list, p_list, moves, accepted);
  }

  for (std::size_t walker = 0; walker < wf_list.size(); ++walker)
  {
    TrialWaveFunction& wavefunction = wf_list[walker];
    if (accepted[walker])
    {
      wavefunction.G = wavefunction.multi_particle_proposed_gradient_;
      wavefunction.L = wavefunction.multi_particle_proposed_laplacian_;
      p_list[walker].G = wavefunction.G;
      p_list[walker].L = wavefunction.L;
      wavefunction.log_real_  = 0;
      wavefunction.PhaseValue = 0;
      for (const auto& component : wavefunction.Z)
      {
        wavefunction.log_real_ += std::real(component->get_log_value());
        wavefunction.PhaseValue += std::imag(component->get_log_value());
      }
    }
    wavefunction.PhaseDiff = 0;
    wavefunction.multi_particle_proposal_pending_ = false;
    wavefunction.multi_particle_proposal_fingerprint_ = 0;
    wavefunction.multi_particle_proposed_log_ratio_ = LogValue(0);
  }
}

const ParticleSet::ParticleGradient& TrialWaveFunction::multiParticleProposalGradient() const
{
  if (!multi_particle_proposal_pending_)
    throw std::logic_error("TrialWaveFunction has no pending selected-electron proposal");
  return multi_particle_proposed_gradient_;
}

void TrialWaveFunction::completeUpdates()
{
  ScopedTimer local_timer(TWF_timers_[ACCEPT_TIMER]);
  for (int i = 0; i < Z.size(); i++)
  {
    ScopedTimer z_timer(WFC_timers_[ACCEPT_TIMER + TIMER_SKIP * i]);
    Z[i]->completeUpdates();
  }
}

void TrialWaveFunction::mw_completeUpdates(const RefVectorWithLeader<TrialWaveFunction>& wf_list)
{
  auto& wf_leader = wf_list.getLeader();
  ScopedTimer local_timer(wf_leader.TWF_timers_[ACCEPT_TIMER]);
  const int num_wfc             = wf_leader.Z.size();
  auto& wavefunction_components = wf_leader.Z;

  for (int i = 0; i < num_wfc; i++)
  {
    ScopedTimer z_timer(wf_leader.WFC_timers_[ACCEPT_TIMER + TIMER_SKIP * i]);
    const auto wfc_list(extractWFCRefList(wf_list, i));
    wavefunction_components[i]->mw_completeUpdates(wfc_list);
  }
}

TrialWaveFunction::LogValue TrialWaveFunction::evaluateGL(ParticleSet& P, bool fromscratch)
{
  ScopedTimer local_timer(TWF_timers_[BUFFER_TIMER]);
  P.G = 0.0;
  P.L = 0.0;
  LogValue logpsi(0.0);
  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer z_timer(WFC_timers_[BUFFER_TIMER + TIMER_SKIP * i]);
    logpsi += Z[i]->evaluateGL(P, P.G, P.L, fromscratch);
  }

  // Ye: temporal workaround to have WF.G/L always defined.
  // remove when KineticEnergy use WF.G/L instead of P.G/L
  G          = P.G;
  L          = P.L;
  log_real_  = std::real(logpsi);
  PhaseValue = std::imag(logpsi);
  return logpsi;
}

void TrialWaveFunction::mw_evaluateGL(const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                      const RefVectorWithLeader<ParticleSet>& p_list,
                                      bool fromscratch)
{
  auto& p_leader  = p_list.getLeader();
  auto& wf_leader = wf_list.getLeader();
  ScopedTimer local_timer(wf_leader.TWF_timers_[BUFFER_TIMER]);

  constexpr RealType czero(0);
  const auto g_list(TrialWaveFunction::extractGRefList(wf_list));
  const auto l_list(TrialWaveFunction::extractLRefList(wf_list));

  const int num_particles = p_leader.getTotalNum();
  for (TrialWaveFunction& wfs : wf_list)
  {
    wfs.G.resize(num_particles);
    wfs.L.resize(num_particles);
    wfs.G          = czero;
    wfs.L          = czero;
    wfs.log_real_  = czero;
    wfs.PhaseValue = czero;
  }

  auto& wavefunction_components = wf_leader.Z;
  const int num_wfc             = wf_leader.Z.size();

  for (int i = 0; i < num_wfc; ++i)
  {
    ScopedTimer z_timer(wf_leader.WFC_timers_[BUFFER_TIMER + TIMER_SKIP * i]);
    const auto wfc_list(extractWFCRefList(wf_list, i));
    wavefunction_components[i]->mw_evaluateGL(wfc_list, p_list, g_list, l_list, fromscratch);
  }

  for (int iw = 0; iw < wf_list.size(); iw++)
  {
    ParticleSet& pset      = p_list[iw];
    TrialWaveFunction& twf = wf_list[iw];

    for (int i = 0; i < num_wfc; ++i)
    {
      twf.log_real_ += std::real(twf.Z[i]->get_log_value());
      twf.PhaseValue += std::imag(twf.Z[i]->get_log_value());
    }

    // Ye: temporal workaround to have P.G/L always defined.
    // remove when KineticEnergy use WF.G/L instead of P.G/L
    pset.G = twf.G;
    pset.L = twf.L;
  }
}

UniqueOptObjRefs TrialWaveFunction::extractOptimizableObjectRefs()
{
  UniqueOptObjRefs opt_obj_refs;
  for (int i = 0; i < Z.size(); i++)
    Z[i]->extractOptimizableObjectRefs(opt_obj_refs);
  return opt_obj_refs;
}

std::vector<std::reference_wrapper<wftrain::StructuredParameterProvider>>
TrialWaveFunction::extractStructuredParameterProviders()
{
  std::vector<std::reference_wrapper<wftrain::StructuredParameterProvider>> providers;
  std::set<std::string> provider_ids;
  for (const auto& component : Z)
    if (wftrain::StructuredParameterProvider* provider = component->structuredParameterProvider())
    {
      const std::string& provider_id = provider->parameterSchema().providerId();
      if (!provider_ids.insert(provider_id).second)
        throw std::invalid_argument("Distinct structured parameter providers have duplicate identity " +
                                    provider_id);
      providers.emplace_back(*provider);
    }
  return providers;
}

void TrialWaveFunction::checkInVariables(OptVariables& active)
{
  auto opt_obj_refs = extractOptimizableObjectRefs();
  for (OptimizableObject& obj : opt_obj_refs)
    obj.checkInVariablesExclusive(active);
}

void TrialWaveFunction::checkOutVariables(const OptVariables& active)
{
  for (int i = 0; i < Z.size(); i++)
    if (Z[i]->isOptimizable())
      Z[i]->checkOutVariables(active);
}

void TrialWaveFunction::resetParameters(const OptVariables& active)
{
  auto opt_obj_refs = extractOptimizableObjectRefs();
  for (OptimizableObject& obj : opt_obj_refs)
    obj.resetParametersExclusive(active);
}

void TrialWaveFunction::reportStatus(std::ostream& os)
{
  auto opt_obj_refs = extractOptimizableObjectRefs();
  for (OptimizableObject& obj : opt_obj_refs)
    obj.reportStatus(os);
}

void TrialWaveFunction::getLogs(std::vector<RealType>& lvals)
{
  lvals.resize(Z.size(), 0);
  for (int i = 0; i < Z.size(); i++)
  {
    lvals[i] = std::real(Z[i]->get_log_value());
  }
}

void TrialWaveFunction::getPhases(std::vector<RealType>& pvals)
{
  pvals.resize(Z.size(), 0);
  for (int i = 0; i < Z.size(); i++)
  {
    pvals[i] = std::imag(Z[i]->get_log_value());
  }
}

void TrialWaveFunction::registerData(ParticleSet& P, WFBufferType& buf)
{
  ScopedTimer local_timer(TWF_timers_[BUFFER_TIMER]);
  //save the current position
  BufferCursor        = buf.current();
  BufferCursor_scalar = buf.current_scalar();
  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer z_timer(WFC_timers_[BUFFER_TIMER + TIMER_SKIP * i]);
    Z[i]->registerData(P, buf);
  }
  buf.add(PhaseValue);
  buf.add(log_real_);
}

void TrialWaveFunction::debugOnlyCheckBuffer(WFBufferType& buffer)
{
#ifndef NDEBUG
  if (buffer.size() < buffer.current() + buffer.current_scalar() * sizeof(FullPrecRealType))
  {
    std::ostringstream assert_message;
    assert_message << "On thread:" << Concurrency::getWorkerId<>() << "  buf_list[iw].get().size():" << buffer.size()
                   << " < buf_list[iw].get().current():" << buffer.current()
                   << " + buf.current_scalar():" << buffer.current_scalar()
                   << " * sizeof(FullPrecRealType):" << sizeof(FullPrecRealType) << '\n';
    throw std::runtime_error(assert_message.str());
  }
#endif
}

TrialWaveFunction::RealType TrialWaveFunction::updateBuffer(ParticleSet& P, WFBufferType& buf, bool fromscratch)
{
  ScopedTimer local_timer(TWF_timers_[BUFFER_TIMER]);
  P.G = 0.0;
  P.L = 0.0;
  buf.rewind(BufferCursor, BufferCursor_scalar);
  LogValue logpsi(0.0);
  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer z_timer(WFC_timers_[BUFFER_TIMER + TIMER_SKIP * i]);
    logpsi += Z[i]->updateBuffer(P, buf, fromscratch);
  }

  G = P.G;
  L = P.L;

  log_real_  = std::real(logpsi);
  PhaseValue = std::imag(logpsi);
  //printGL(P.G,P.L);
  buf.put(PhaseValue);
  buf.put(log_real_);
  // Ye: temperal added check, to be removed
  debugOnlyCheckBuffer(buf);
  return log_real_;
}

void TrialWaveFunction::copyFromBuffer(ParticleSet& P, WFBufferType& buf)
{
  ScopedTimer local_timer(TWF_timers_[BUFFER_TIMER]);
  buf.rewind(BufferCursor, BufferCursor_scalar);
  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer z_timer(WFC_timers_[BUFFER_TIMER + TIMER_SKIP * i]);
    Z[i]->copyFromBuffer(P, buf);
  }
  //get the gradients and laplacians from the buffer
  buf.get(PhaseValue);
  buf.get(log_real_);
  debugOnlyCheckBuffer(buf);
}

void TrialWaveFunction::evaluateRatios(const VirtualParticleSet& VP, std::vector<ValueType>& ratios, ComputeType ct)
{
  ScopedTimer local_timer(TWF_timers_[NL_TIMER]);
  assert(VP.getTotalNum() == ratios.size());
  std::vector<ValueType> t(ratios.size());
  std::fill(ratios.begin(), ratios.end(), 1.0);
  for (int i = 0; i < Z.size(); ++i)
    if (ct == ComputeType::ALL || (Z[i]->isFermionic() && ct == ComputeType::FERMIONIC) ||
        (!Z[i]->isFermionic() && ct == ComputeType::NONFERMIONIC))
    {
      ScopedTimer z_timer(WFC_timers_[NL_TIMER + TIMER_SKIP * i]);
      Z[i]->evaluateRatios(VP, t);
      for (int j = 0; j < ratios.size(); ++j)
        ratios[j] *= t[j];
    }
}

void TrialWaveFunction::evaluateSpinorRatios(const VirtualParticleSet& VP,
                                             const std::pair<ValueVector, ValueVector>& spinor_multiplier,
                                             std::vector<ValueType>& ratios) const
{
  ScopedTimer local_timer(TWF_timers_[NL_TIMER]);
  assert(VP.getTotalNum() == ratios.size());
  std::vector<ValueType> t(ratios.size());
  std::fill(ratios.begin(), ratios.end(), 1.0);
  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer z_timer(WFC_timers_[NL_TIMER + TIMER_SKIP * i]);
    Z[i]->evaluateSpinorRatios(VP, spinor_multiplier, t);
    for (int j = 0; j < ratios.size(); ++j)
      ratios[j] *= t[j];
  }
}

void TrialWaveFunction::mw_evaluateRatios(const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                          const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
                                          const RefVector<std::vector<ValueType>>& ratios_list,
                                          ComputeType ct)
{
  auto& wf_leader = wf_list.getLeader();
  ScopedTimer local_timer(wf_leader.TWF_timers_[NL_TIMER]);
  auto& wavefunction_components = wf_leader.Z;
  std::vector<std::vector<ValueType>> t(ratios_list.size());
  for (int iw = 0; iw < wf_list.size(); iw++)
  {
    std::vector<ValueType>& ratios = ratios_list[iw];
    assert(vp_list[iw].getTotalNum() == ratios.size());
    std::fill(ratios.begin(), ratios.end(), 1.0);
    t[iw].resize(ratios.size());
  }

  for (int i = 0; i < wavefunction_components.size(); i++)
    if (ct == ComputeType::ALL || (wavefunction_components[i]->isFermionic() && ct == ComputeType::FERMIONIC) ||
        (!wavefunction_components[i]->isFermionic() && ct == ComputeType::NONFERMIONIC))
    {
      ScopedTimer z_timer(wf_leader.WFC_timers_[NL_TIMER + TIMER_SKIP * i]);
      const auto wfc_list(extractWFCRefList(wf_list, i));
      wavefunction_components[i]->mw_evaluateRatios(wfc_list, vp_list, t);
      for (int iw = 0; iw < wf_list.size(); iw++)
      {
        std::vector<ValueType>& ratios = ratios_list[iw];
        for (int j = 0; j < ratios.size(); ++j)
          ratios[j] *= t[iw][j];
      }
    }
}

void TrialWaveFunction::mw_evaluateSpinorRatios(
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
    const RefVector<std::pair<ValueVector, ValueVector>>& spinor_multiplier_list,
    const RefVector<std::vector<ValueType>>& ratios_list)
{
  auto& wf_leader = wf_list.getLeader();
  ScopedTimer local_timer(wf_leader.TWF_timers_[NL_TIMER]);
  auto& wavefunction_components = wf_leader.Z;
  std::vector<std::vector<ValueType>> t(ratios_list.size());
  for (int iw = 0; iw < wf_list.size(); iw++)
  {
    std::vector<ValueType>& ratios = ratios_list[iw];
    assert(vp_list[iw].getTotalNum() == ratios.size());
    std::fill(ratios.begin(), ratios.end(), 1.0);
    t[iw].resize(ratios.size());
  }

  for (int i = 0; i < wavefunction_components.size(); i++)
  {
    ScopedTimer z_timer(wf_leader.WFC_timers_[NL_TIMER + TIMER_SKIP * i]);
    const auto wfc_list(extractWFCRefList(wf_list, i));
    wavefunction_components[i]->mw_evaluateSpinorRatios(wfc_list, vp_list, spinor_multiplier_list, t);
    for (int iw = 0; iw < wf_list.size(); iw++)
    {
      std::vector<ValueType>& ratios = ratios_list[iw];
      for (int j = 0; j < ratios.size(); ++j)
        ratios[j] *= t[iw][j];
    }
  }
}

void TrialWaveFunction::mw_evaluateVirtualRatios(
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
    const VirtualParticleBatch& batch,
    std::vector<ValueType>& ratios,
    std::vector<EvaluationStamp>& evaluation_stamps,
    ComputeType ct)
{
  if (wf_list.size() != batch.walkerCount() || p_list.size() != batch.walkerCount() ||
      vp_scratch_list.size() != batch.walkerCount())
    throw std::invalid_argument(
        "TrialWaveFunction::mw_evaluateVirtualRatios list sizes do not match the descriptor walker count.");
  batch.validateOutputExtent(ratios.size());
  batch.validateFor(p_list);

  switch (ct)
  {
  case ComputeType::ALL:
  case ComputeType::FERMIONIC:
  case ComputeType::NONFERMIONIC:
    break;
  default:
    throw std::invalid_argument("TrialWaveFunction::mw_evaluateVirtualRatios received an invalid ComputeType.");
  }

  TrialWaveFunction& wf_leader = wf_list.getLeader();
  const std::size_t component_count = wf_leader.Z.size();
  for (std::size_t walker = 0; walker < batch.walkerCount(); ++walker)
  {
    for (std::size_t other = 0; other < walker; ++other)
    {
      if (std::addressof(wf_list[walker]) == std::addressof(wf_list[other]))
        throw std::invalid_argument(
            "TrialWaveFunction::mw_evaluateVirtualRatios requires one distinct wavefunction clone per walker.");
      if (std::addressof(vp_scratch_list[walker]) == std::addressof(vp_scratch_list[other]))
        throw std::invalid_argument(
            "TrialWaveFunction::mw_evaluateVirtualRatios requires one distinct scratch object per walker.");
    }
    if (wf_list[walker].Z.size() != component_count)
      throw std::invalid_argument(
          "TrialWaveFunction::mw_evaluateVirtualRatios wavefunction clones have different component counts.");

    const ParticleSet* scratch_as_particles = static_cast<const ParticleSet*>(std::addressof(vp_scratch_list[walker]));
    for (std::size_t reference = 0; reference < batch.walkerCount(); ++reference)
      if (scratch_as_particles == std::addressof(p_list[reference]))
        throw std::invalid_argument(
            "TrialWaveFunction::mw_evaluateVirtualRatios scratch objects must not alias reference walkers.");
    if (vp_scratch_list[walker].isSpinor() != p_list[walker].isSpinor())
      throw std::invalid_argument(
          "TrialWaveFunction::mw_evaluateVirtualRatios reference and scratch spinor modes do not match.");
  }

  for (std::size_t component = 0; component < component_count; ++component)
    for (std::size_t walker = 0; walker < batch.walkerCount(); ++walker)
      if (typeid(*wf_list[walker].Z[component]) != typeid(*wf_leader.Z[component]) ||
          wf_list[walker].Z[component]->isFermionic() != wf_leader.Z[component]->isFermionic())
        throw std::invalid_argument(
            "TrialWaveFunction::mw_evaluateVirtualRatios wavefunction clones have incompatible component topology.");

  ScopedTimer local_timer(wf_leader.TWF_timers_[NL_TIMER]);
  std::vector<ValueType> staged_ratios(batch.size(), ValueType(1));
  std::vector<EvaluationStamp> staged_stamps;
  staged_stamps.reserve(component_count);
  std::vector<ValueType> component_ratios(batch.size());

  for (std::size_t component = 0; component < component_count; ++component)
  {
    const WaveFunctionComponent& component_leader = *wf_leader.Z[component];
    const bool selected = ct == ComputeType::ALL ||
        (component_leader.isFermionic() && ct == ComputeType::FERMIONIC) ||
        (!component_leader.isFermionic() && ct == ComputeType::NONFERMIONIC);
    if (!selected)
      continue;

    ScopedTimer component_timer(wf_leader.WFC_timers_[NL_TIMER + TIMER_SKIP * component]);
    const RefVectorWithLeader<WaveFunctionComponent> wfc_list = extractWFCRefList(wf_list, component);
    const EvaluationStamp stamp = component_leader.mw_evaluateVirtualRatios(
        wfc_list, p_list, vp_scratch_list, batch, component_ratios);
    if (component_ratios.size() != batch.size())
      throw std::runtime_error(
          "WaveFunctionComponent::mw_evaluateVirtualRatios changed the flattened output extent.");

    if (stamp.isVersioned())
    {
      for (const EvaluationStamp& prior_stamp : staged_stamps)
        if (prior_stamp.source_identity_ == stamp.source_identity_ && prior_stamp.version_ != stamp.version_)
          throw std::runtime_error(
              "TrialWaveFunction::mw_evaluateVirtualRatios observed conflicting versions of one shared state.");
      staged_stamps.push_back(stamp);
    }

    for (std::size_t virtual_index = 0; virtual_index < batch.size(); ++virtual_index)
      staged_ratios[virtual_index] *= component_ratios[virtual_index];
  }

  ratios.swap(staged_ratios);
  evaluation_stamps.swap(staged_stamps);
}

void TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted(
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
    const VirtualParticleBatch& batch,
    const OptVariables& optvars,
    const std::vector<ValueType>& bare_weights,
    std::vector<ValueType>& ratios,
    const std::vector<ParameterDerivativeView>& weighted_derivatives,
    std::vector<EvaluationStamp>& evaluation_stamps,
    ComputeType ct)
{
  const std::size_t walker_count = batch.walkerCount();
  if (wf_list.size() != walker_count || p_list.size() != walker_count ||
      vp_scratch_list.size() != walker_count || weighted_derivatives.size() != walker_count)
    throw std::invalid_argument(
        "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted list sizes do not match the descriptor walker "
        "count.");
  batch.validateOutputExtent(bare_weights.size());
  batch.validateOutputExtent(ratios.size());
  batch.validateFor(p_list);

  switch (ct)
  {
  case ComputeType::ALL:
  case ComputeType::FERMIONIC:
  case ComputeType::NONFERMIONIC:
    break;
  default:
    throw std::invalid_argument(
        "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted received an invalid ComputeType.");
  }

  const std::size_t derivative_width = weighted_derivatives.empty() ? 0 : weighted_derivatives.front().size;
  if (!weighted_derivatives.empty() && derivative_width < requiredDerivativeExtent(optvars))
    throw std::invalid_argument(
        "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted derivative rows are too short.");
  if (derivative_width != 0 && walker_count > std::numeric_limits<std::size_t>::max() / derivative_width)
    throw std::length_error(
        "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted derivative staging extent overflows.");
  if (derivative_width > std::numeric_limits<std::size_t>::max() / sizeof(ValueType))
    throw std::length_error(
        "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted derivative destination extent overflows.");

  TrialWaveFunction& wf_leader       = wf_list.getLeader();
  const std::size_t component_count  = wf_leader.Z.size();
  const std::size_t derivative_bytes = derivative_width * sizeof(ValueType);
  if (batch.size() > std::numeric_limits<std::size_t>::max() / sizeof(ValueType))
    throw std::length_error(
        "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted flat value extent overflows.");
  const std::size_t flat_value_bytes = batch.size() * sizeof(ValueType);
  const auto checked_end = [](const ValueType* begin, std::size_t bytes) {
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(begin);
    if (address > std::numeric_limits<std::uintptr_t>::max() - bytes)
      throw std::length_error(
          "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted caller storage range overflows.");
    return address + bytes;
  };
  const std::uintptr_t ratio_begin = reinterpret_cast<std::uintptr_t>(ratios.data());
  const std::uintptr_t ratio_end   = checked_end(ratios.data(), flat_value_bytes);
  const std::uintptr_t weight_begin = reinterpret_cast<std::uintptr_t>(bare_weights.data());
  const std::uintptr_t weight_end   = checked_end(bare_weights.data(), flat_value_bytes);
  const auto ranges_overlap = [](std::uintptr_t first_begin,
                                 std::uintptr_t first_end,
                                 std::uintptr_t second_begin,
                                 std::uintptr_t second_end) {
    return first_begin < second_end && second_begin < first_end;
  };
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    const ParameterDerivativeView destination = weighted_derivatives[walker];
    if (destination.size != derivative_width || (derivative_width != 0 && destination.data == nullptr))
      throw std::invalid_argument(
          "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted derivative rows have inconsistent shapes.");
    if (derivative_width != 0)
    {
      const std::uintptr_t destination_begin = reinterpret_cast<std::uintptr_t>(destination.data);
      if (destination_begin > std::numeric_limits<std::uintptr_t>::max() - derivative_bytes)
        throw std::length_error(
            "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted derivative destination range overflows.");
      const std::uintptr_t destination_end = destination_begin + derivative_bytes;
      if (ranges_overlap(destination_begin, destination_end, ratio_begin, ratio_end) ||
          ranges_overlap(destination_begin, destination_end, weight_begin, weight_end))
        throw std::invalid_argument(
            "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted derivative destinations must not overlap "
            "flat value inputs or outputs.");
      for (std::size_t other = 0; other < walker; ++other)
      {
        const std::uintptr_t other_begin =
            reinterpret_cast<std::uintptr_t>(weighted_derivatives[other].data);
        const std::uintptr_t other_end = other_begin + derivative_bytes;
        if (destination_begin < other_end && other_begin < destination_end)
          throw std::invalid_argument(
              "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted requires non-overlapping derivative "
              "destinations.");
      }
    }

    for (std::size_t other = 0; other < walker; ++other)
    {
      if (std::addressof(wf_list[walker]) == std::addressof(wf_list[other]))
        throw std::invalid_argument(
            "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted requires one distinct wavefunction clone "
            "per walker.");
      if (std::addressof(vp_scratch_list[walker]) == std::addressof(vp_scratch_list[other]))
        throw std::invalid_argument(
            "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted requires one distinct scratch object per "
            "walker.");
    }
    if (wf_list[walker].Z.size() != component_count)
      throw std::invalid_argument(
          "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted wavefunction clones have different component "
          "counts.");

    const ParticleSet* scratch_as_particles = static_cast<const ParticleSet*>(std::addressof(vp_scratch_list[walker]));
    for (std::size_t reference = 0; reference < walker_count; ++reference)
      if (scratch_as_particles == std::addressof(p_list[reference]))
        throw std::invalid_argument(
            "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted scratch objects must not alias reference "
            "walkers.");
    if (vp_scratch_list[walker].isSpinor() != p_list[walker].isSpinor())
      throw std::invalid_argument(
          "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted reference and scratch spinor modes do not "
          "match.");
  }

  for (std::size_t component = 0; component < component_count; ++component)
    for (std::size_t walker = 0; walker < walker_count; ++walker)
      if (typeid(*wf_list[walker].Z[component]) != typeid(*wf_leader.Z[component]) ||
          wf_list[walker].Z[component]->isFermionic() != wf_leader.Z[component]->isFermionic())
        throw std::invalid_argument(
            "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted wavefunction clones have incompatible "
            "component topology.");

  // Both phases are private until every component has succeeded.  In
  // particular, a late weighted-score exception cannot expose the already
  // completed ratio product or any earlier component's derivative delta.
  ScopedTimer local_timer(wf_leader.TWF_timers_[NL_TIMER]);
  std::vector<ValueType> staged_ratios(batch.size(), ValueType(1));
  std::vector<ValueType> component_ratios(batch.size());
  std::vector<EvaluationStamp> value_stamps(component_count);
  std::vector<bool> selected_components(component_count, false);
  std::vector<EvaluationStamp> staged_stamps;
  staged_stamps.reserve(component_count);

  for (std::size_t component = 0; component < component_count; ++component)
  {
    const WaveFunctionComponent& component_leader = *wf_leader.Z[component];
    const bool selected = ct == ComputeType::ALL ||
        (component_leader.isFermionic() && ct == ComputeType::FERMIONIC) ||
        (!component_leader.isFermionic() && ct == ComputeType::NONFERMIONIC);
    selected_components[component] = selected;
    if (!selected)
      continue;

    ScopedTimer component_timer(wf_leader.WFC_timers_[NL_TIMER + TIMER_SKIP * component]);
    const RefVectorWithLeader<WaveFunctionComponent> wfc_list = extractWFCRefList(wf_list, component);
    const EvaluationStamp stamp = component_leader.mw_evaluateVirtualRatios(
        wfc_list, p_list, vp_scratch_list, batch, component_ratios);
    if (component_ratios.size() != batch.size())
      throw std::runtime_error(
          "WaveFunctionComponent::mw_evaluateVirtualRatios changed the flattened output extent.");
    value_stamps[component] = stamp;

    if (stamp.isVersioned())
    {
      for (const EvaluationStamp& prior_stamp : staged_stamps)
        if (prior_stamp.source_identity_ == stamp.source_identity_ && prior_stamp.version_ != stamp.version_)
          throw std::runtime_error(
              "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted observed conflicting value versions of "
              "one shared state.");
      staged_stamps.push_back(stamp);
    }

    for (std::size_t virtual_index = 0; virtual_index < batch.size(); ++virtual_index)
      staged_ratios[virtual_index] *= component_ratios[virtual_index];
  }

  std::vector<ValueType> total_weights(batch.size());
  for (std::size_t virtual_index = 0; virtual_index < batch.size(); ++virtual_index)
    total_weights[virtual_index] = bare_weights[virtual_index] * staged_ratios[virtual_index];

  const std::size_t staged_extent = walker_count * derivative_width;
  std::vector<ValueType> staged_derivatives(staged_extent, ValueType(0));
  std::vector<ParameterDerivativeView> staged_views;
  staged_views.reserve(walker_count);
  for (std::size_t walker = 0; walker < walker_count; ++walker)
    staged_views.push_back(
        {derivative_width == 0 ? nullptr : staged_derivatives.data() + walker * derivative_width, derivative_width});

  for (std::size_t component = 0; component < component_count; ++component)
  {
    if (!selected_components[component])
      continue;

    const WaveFunctionComponent& component_leader = *wf_leader.Z[component];
    ScopedTimer component_timer(wf_leader.WFC_timers_[DERIVS_TIMER + TIMER_SKIP * component]);
    const RefVectorWithLeader<WaveFunctionComponent> wfc_list = extractWFCRefList(wf_list, component);
    const EvaluationStamp weighted_stamp = component_leader.mw_evaluateVirtualDerivRatiosWeighted(
        wfc_list, p_list, vp_scratch_list, batch, optvars, total_weights, staged_views);
    if (weighted_stamp != value_stamps[component])
      throw std::runtime_error(
          "TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted observed different component versions in "
          "the value and weighted phases.");
  }

  // Vector swaps are nonthrowing for these standard-allocator vectors, and
  // ValueType addition is nonthrowing.  All potentially failing work has
  // therefore completed before the first caller-visible publication.
  ratios.swap(staged_ratios);
  evaluation_stamps.swap(staged_stamps);
  for (std::size_t walker = 0; walker < walker_count; ++walker)
    for (std::size_t parameter = 0; parameter < derivative_width; ++parameter)
      weighted_derivatives[walker][parameter] += staged_views[walker][parameter];
}

void TrialWaveFunction::evaluateDerivRatios(const VirtualParticleSet& VP,
                                            const OptVariables& optvars,
                                            std::vector<ValueType>& ratios,
                                            Matrix<ValueType>& dratio)
{
  std::fill(ratios.begin(), ratios.end(), 1.0);
  std::fill(dratio.begin(), dratio.end(), 0.0);
  std::vector<ValueType> t(ratios.size());
  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer z_timer(WFC_timers_[DERIVS_TIMER + TIMER_SKIP * i]);
    Z[i]->evaluateDerivRatios(VP, optvars, t, dratio);
    for (int j = 0; j < ratios.size(); ++j)
      ratios[j] *= t[j];
  }
}

void TrialWaveFunction::evaluateDerivRatiosWeighted(const VirtualParticleSet& VP,
                                                    const OptVariables& optvars,
                                                    const std::vector<ValueType>& bare_weights,
                                                    std::vector<ValueType>& ratios,
                                                    ParameterDerivativeView weighted_derivatives,
                                                    ComputeType ct)
{
  const std::size_t virtual_count = VP.getTotalNum();
  if (bare_weights.size() != virtual_count || ratios.size() != virtual_count ||
      weighted_derivatives.size < optvars.size_of_active() ||
      (weighted_derivatives.size != 0 && weighted_derivatives.data == nullptr))
    throw std::invalid_argument("TrialWaveFunction weighted derivative-ratio inputs have inconsistent shapes");

  // Ratios must be formed for the complete selected product before any
  // component derivative is reduced.  Using a component-local ratio here
  // would omit cross-component factors from d(V_NL Psi / Psi)/d alpha.
  evaluateRatios(VP, ratios, ct);
  std::vector<ValueType> total_weights(virtual_count);
  for (std::size_t virtual_index = 0; virtual_index < virtual_count; ++virtual_index)
    total_weights[virtual_index] = bare_weights[virtual_index] * ratios[virtual_index];

  for (int component = 0; component < Z.size(); ++component)
    if (ct == ComputeType::ALL || (Z[component]->isFermionic() && ct == ComputeType::FERMIONIC) ||
        (!Z[component]->isFermionic() && ct == ComputeType::NONFERMIONIC))
    {
      ScopedTimer component_timer(WFC_timers_[DERIVS_TIMER + TIMER_SKIP * component]);
      Z[component]->evaluateDerivRatiosWeighted(VP, optvars, total_weights, weighted_derivatives);
    }
}

void TrialWaveFunction::mw_evaluateDerivRatiosWeighted(
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
    const OptVariables& optvars,
    const RefVector<const std::vector<ValueType>>& bare_weights,
    const RefVector<std::vector<ValueType>>& ratios,
    const std::vector<ParameterDerivativeView>& weighted_derivatives,
    ComputeType ct)
{
  const std::size_t walker_count = wf_list.size();
  if (vp_list.size() != walker_count || bare_weights.size() != walker_count || ratios.size() != walker_count ||
      weighted_derivatives.size() != walker_count)
    throw std::invalid_argument("TrialWaveFunction batched weighted reductions have inconsistent walker counts");

  auto& leader = wf_list.getLeader();

  // The existing ratio dispatcher is already component-major and respects the
  // fermionic/nonfermionic partition.  Its results establish the total-product
  // weights consumed by every component reverse pass below.
  mw_evaluateRatios(wf_list, vp_list, ratios, ct);
  std::vector<std::vector<ValueType>> total_weights_storage(walker_count);
  RefVector<const std::vector<ValueType>> total_weights;
  total_weights.reserve(walker_count);
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    const auto& walker_bare_weights = bare_weights[walker].get();
    auto& walker_ratios             = ratios[walker].get();
    if (walker_bare_weights.size() != walker_ratios.size() ||
        walker_ratios.size() != static_cast<std::size_t>(vp_list[walker].getTotalNum()) ||
        weighted_derivatives[walker].size < optvars.size_of_active() ||
        (weighted_derivatives[walker].size != 0 && weighted_derivatives[walker].data == nullptr))
      throw std::invalid_argument("TrialWaveFunction batched weighted reduction has an invalid walker shape");

    auto& walker_total_weights = total_weights_storage[walker];
    walker_total_weights.resize(walker_ratios.size());
    for (std::size_t virtual_index = 0; virtual_index < walker_ratios.size(); ++virtual_index)
      walker_total_weights[virtual_index] = walker_bare_weights[virtual_index] * walker_ratios[virtual_index];
    total_weights.push_back(std::cref(walker_total_weights));
  }

  auto& components = leader.Z;
  for (int component = 0; component < components.size(); ++component)
    if (ct == ComputeType::ALL || (components[component]->isFermionic() && ct == ComputeType::FERMIONIC) ||
        (!components[component]->isFermionic() && ct == ComputeType::NONFERMIONIC))
    {
      ScopedTimer component_timer(leader.WFC_timers_[DERIVS_TIMER + TIMER_SKIP * component]);
      const auto wfc_list(extractWFCRefList(wf_list, component));
      components[component]->mw_evaluateDerivRatiosWeighted(wfc_list, vp_list, optvars, total_weights,
                                                            weighted_derivatives);
    }
}

void TrialWaveFunction::evaluateSpinorDerivRatios(const VirtualParticleSet& VP,
                                                  const std::pair<ValueVector, ValueVector>& spinor_multiplier,
                                                  const OptVariables& optvars,
                                                  std::vector<ValueType>& ratios,
                                                  Matrix<ValueType>& dratio)
{
  std::fill(ratios.begin(), ratios.end(), 1.0);
  std::fill(dratio.begin(), dratio.end(), 0.0);
  std::vector<ValueType> t(ratios.size());
  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer z_timer(WFC_timers_[DERIVS_TIMER + TIMER_SKIP * i]);
    Z[i]->evaluateSpinorDerivRatios(VP, spinor_multiplier, optvars, t, dratio);
    for (int j = 0; j < ratios.size(); ++j)
      ratios[j] *= t[j];
  }
}

bool TrialWaveFunction::put(xmlNodePtr cur) { return true; }

std::unique_ptr<TrialWaveFunction> TrialWaveFunction::makeClone(ParticleSet& tqp) const
{
  if (resource_acquired_)
    throw std::logic_error(
        "Cannot clone a TrialWaveFunction while its resources are acquired");
  validateRetainedBatchExecutionBinding();

  auto myclone                 = std::make_unique<TrialWaveFunction>(runtime_options_, myName, use_tasking_);
  myclone->BufferCursor        = BufferCursor;
  myclone->BufferCursor_scalar = BufferCursor_scalar;
  myclone->complete_batch_memory_accounting_for_testing_ =
      complete_batch_memory_accounting_for_testing_;
  for (int i = 0; i < Z.size(); ++i)
    myclone->addComponent(Z[i]->makeClone(tqp));
  // The clone receives immutable participant identity only.  Constructor-empty
  // aggregate G/L and preparation provenance are materialized independently at
  // the later resident-clone boundary.
  // Visit children even for a null aggregate plan: a legacy clone constructor
  // may have copied stale internal binding state that explicit no-policy bind
  // must clear before the clone enters a new section.
  myclone->bindBatchExecutionPlan(batch_execution_plan_);
  return myclone;
}

/** evaluate derivatives of KE wrt optimizable varibles
 *
 * @todo WaveFunctionComponent objects should take the mass into account.
 */
void TrialWaveFunction::evaluateDerivatives(ParticleSet& P,
                                            const OptVariables& optvars,
                                            Vector<ValueType>& dlogpsi,
                                            Vector<ValueType>& dhpsioverpsi)
{
  //     // First, zero out derivatives
  //  This should only be done for some variables.
  //     for (int j=0; j<dlogpsi.size(); j++)
  //       dlogpsi[j] = dhpsioverpsi[j] = 0.0;
  for (int i = 0; i < Z.size(); i++)
  {
    ScopedTimer z_timer(WFC_timers_[DERIVS_TIMER + TIMER_SKIP * i]);
    Z[i]->evaluateDerivatives(P, optvars, dlogpsi, dhpsioverpsi);
  }
}

void TrialWaveFunction::mw_evaluateParameterDerivatives(const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                                        const RefVectorWithLeader<ParticleSet>& p_list,
                                                        const OptVariables& optvars,
                                                        RecordArray<ValueType>& dlogpsi,
                                                        RecordArray<ValueType>& dhpsioverpsi)
{
  auto& leader = wf_list.getLeader();
  if (wf_list.size() != p_list.size() || dlogpsi.getNumOfEntries() != wf_list.size() ||
      dhpsioverpsi.getNumOfEntries() != wf_list.size() ||
      dlogpsi.getNumOfParams() != dhpsioverpsi.getNumOfParams())
    throw std::invalid_argument("TrialWaveFunction batched derivative inputs have inconsistent shapes");

  // Dispatch component-major so an optimized component can process the complete
  // walker batch while legacy components retain the serialized virtual default.
  for (int component = 0; component < leader.Z.size(); ++component)
  {
    ScopedTimer component_timer(leader.WFC_timers_[DERIVS_TIMER + TIMER_SKIP * component]);
    const auto wfc_list(extractWFCRefList(wf_list, component));
    leader.Z[component]->mw_evaluateParameterDerivatives(wfc_list, p_list, optvars, dlogpsi, dhpsioverpsi);
  }
}


void TrialWaveFunction::evaluateDerivativesWF(ParticleSet& P, const OptVariables& optvars, Vector<ValueType>& dlogpsi)
{
  for (int i = 0; i < Z.size(); i++)
  {
    ScopedTimer z_timer(WFC_timers_[DERIVS_TIMER + TIMER_SKIP * i]);
    Z[i]->evaluateDerivativesWF(P, optvars, dlogpsi);
  }
}

void TrialWaveFunction::mw_evaluateParameterDerivativesWF(const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                                          const RefVectorWithLeader<ParticleSet>& p_list,
                                                          const OptVariables& optvars,
                                                          RecordArray<ValueType>& dlogpsi)
{
  auto& leader = wf_list.getLeader();
  if (wf_list.size() != p_list.size() || dlogpsi.getNumOfEntries() != wf_list.size())
    throw std::invalid_argument("TrialWaveFunction batched score inputs have inconsistent shapes");

  for (int component = 0; component < leader.Z.size(); ++component)
  {
    ScopedTimer component_timer(leader.WFC_timers_[DERIVS_TIMER + TIMER_SKIP * component]);
    const auto wfc_list(extractWFCRefList(wf_list, component));
    leader.Z[component]->mw_evaluateParameterDerivativesWF(wfc_list, p_list, optvars, dlogpsi);
  }
}

TrialWaveFunction::RealType TrialWaveFunction::KECorrection() const
{
  RealType sum = 0.0;
  for (int i = 0; i < Z.size(); ++i)
    sum += Z[i]->KECorrection();
  return sum;
}

void TrialWaveFunction::evaluateRatiosAlltoOne(ParticleSet& P, std::vector<ValueType>& ratios)
{
  ScopedTimer local_timer(TWF_timers_[V_TIMER]);
  std::fill(ratios.begin(), ratios.end(), 1.0);
  std::vector<ValueType> t(ratios.size());
  for (int i = 0; i < Z.size(); ++i)
  {
    ScopedTimer local_timer(WFC_timers_[V_TIMER + TIMER_SKIP * i]);
    Z[i]->evaluateRatiosAlltoOne(P, t);
    for (int j = 0; j < t.size(); ++j)
      ratios[j] *= t[j];
  }
}

void TrialWaveFunction::createResource(ResourceCollection& collection) const
{
  if (batch_execution_plan_)
  {
    BatchMemoryContribution sole_child;
    const BatchMemoryContribution* sole_child_ptr = nullptr;
    bool sole_child_atomic = false;
    if (Z.size() == 1)
    {
      const auto& evidence = bound_batch_topology_.sole_component_plan.evidence();
      sole_child.logical_maximum    = evidence.logical_maximum;
      sole_child.owner_multiplicity = evidence.owner_multiplicity;
      sole_child.fully_accounted    = evidence.fully_accounted;
      sole_child_ptr                = &sole_child;
      sole_child_atomic             = Z.front()->supportsAtomicBatchPublication();
    }
    const TrialWaveFunctionMemoryPolicyInput policy_input =
        makeTrialWaveFunctionMemoryPolicyInput(
            Z.size(), use_tasking_, static_cast<bool>(twf_fastderiv_),
            complete_batch_memory_accounting_for_testing_, sole_child_ptr,
            sole_child_atomic);
    // Production accounting claims remain false.  In particular, target spin
    // capability is not yet fingerprinted here; the complete-claims override is
    // confined to friend tests until that later driver boundary supplies it.
    collection.addResource(std::make_unique<TrialWaveFunctionMultiWalkerResource>(
        Z.empty() ? nullptr : Z.front().get(), policy_input,
        bound_batch_topology_.aggregate_plan));
  }

  for (int i = 0; i < Z.size(); ++i)
    Z[i]->createResource(collection);

  // Delegate to TWFFastDerivWrapper where the definition is visible
  if (twf_fastderiv_)
    TWFFastDerivWrapper::createResource(collection);
}

void TrialWaveFunction::acquireResource(ResourceCollection& collection,
                                        const RefVectorWithLeader<TrialWaveFunction>& wf_list)
{
  auto& wf_leader = wf_list.getLeader();
  if (wf_leader.resource_acquired_)
    throw std::logic_error(
        "TrialWaveFunction resources are already acquired for the leader");
  if (wf_leader.aggregate_mw_resource_handle_)
    throw std::logic_error(
        "TrialWaveFunction leader retained a stale aggregate resource handle");
  if (wf_leader.acquired_resource_collection_)
    throw std::logic_error(
        "TrialWaveFunction leader retained stale ResourceCollection provenance");
  if (!wf_leader.acquired_batch_topology_.sameState({}))
    throw std::logic_error(
        "TrialWaveFunction leader retained a stale resource-acquisition topology");

  const BatchResourcePreparationProvenance& provenance =
      collection.getBatchResourcePreparationProvenance();
  if (wf_leader.batch_execution_plan_)
  {
    const InlineBatchTopologyState leader_topology =
        wf_leader.bound_batch_topology_;
    if (provenance.state != BatchResourcePreparationState::PREPARED ||
        provenance.plan.get() != wf_leader.batch_execution_plan_.get())
      throw std::logic_error(
          "TrialWaveFunction planned acquisition requires its prepared ResourceCollection");
    if (wf_leader.Z.size() != 1 || wf_leader.use_tasking_ ||
        wf_leader.twf_fastderiv_ || !leader_topology.engaged ||
        leader_topology.component_count != 1 ||
        !leader_topology.aggregate_plan ||
        !leader_topology.sole_component_plan ||
        &leader_topology.aggregate_plan.plan() !=
            wf_leader.batch_execution_plan_.get() ||
        &leader_topology.sole_component_plan.plan() !=
            wf_leader.batch_execution_plan_.get())
      throw std::logic_error(
          "TrialWaveFunction planned acquisition requires the direct C==1 topology");
    const auto& reserves = trialWaveFunctionReserveWalkersPerCrowd(
        provenance.plan->topology());
    if (provenance.crowd_index >= reserves.size() ||
        wf_list.size() > reserves[provenance.crowd_index])
      throw std::length_error(
          "TrialWaveFunction live crowd exceeds its planned reserve");

    wf_leader.validatePreparedBatchExecutionClone(
        leader_topology.aggregate_plan);
    if (!wf_leader.Z.front()->hasBatchExecutionPlanBinding(
            leader_topology.sole_component_plan) ||
        !wf_leader.Z.front()->hasPreparedBatchExecutionClone(
            leader_topology.sole_component_plan))
      throw std::logic_error(
          "TrialWaveFunction leader child lacks exact planned preparation");
    for (TrialWaveFunction& wavefunction : wf_list)
    {
      if (wavefunction.resource_acquired_ ||
          !wavefunction.acquired_batch_topology_.sameState({}))
        throw std::logic_error(
            "TrialWaveFunction clone already owns resources or stale acquisition state");
      if (wavefunction.batch_execution_plan_.get() !=
              wf_leader.batch_execution_plan_.get() ||
          !wavefunction.bound_batch_topology_.sameState(leader_topology) ||
          wavefunction.Z.size() != 1 || wavefunction.use_tasking_ ||
          wavefunction.twf_fastderiv_)
        throw std::invalid_argument(
            "TrialWaveFunction planned resource lanes have incompatible provenance");
      wavefunction.validatePreparedBatchExecutionClone(
          leader_topology.aggregate_plan);
      if (!wavefunction.Z.front()->hasBatchExecutionPlanBinding(
              leader_topology.sole_component_plan) ||
          !wavefunction.Z.front()->hasPreparedBatchExecutionClone(
              leader_topology.sole_component_plan))
        throw std::logic_error(
            "TrialWaveFunction resource lane child lacks exact planned preparation");
    }

    const std::size_t aggregate_slot = collection.getCursor();
    auto aggregate =
        collection.lendResource<TrialWaveFunctionMultiWalkerResource>();
    try
    {
      aggregate.getResource().validateAcquiredBinding(
          leader_topology.aggregate_plan, provenance.crowd_index,
          wf_list.size());
      aggregate.getResource().bindViews(wf_list);
      wf_leader.Z.front()->acquireResource(
          collection, aggregate.getResource().componentView());
    }
    catch (...)
    {
      const std::exception_ptr failure = std::current_exception();
      aggregate.getResource().resetViews();
      collection.rewind(aggregate_slot);
      collection.takebackResource(aggregate);
      collection.rewind(aggregate_slot);
      std::rethrow_exception(failure);
    }

    // No throwing operation follows child acquisition: either every owner is
    // live or no lane publishes acquisition state.
    static_assert(std::is_nothrow_copy_assignable_v<InlineBatchTopologyState>);
    static_assert(std::is_nothrow_move_assignable_v<
                  ResourceHandle<TrialWaveFunctionMultiWalkerResource>>);
    wf_leader.aggregate_resource_cursor_ = aggregate_slot;
    wf_leader.child_resource_cursor_     = aggregate_slot + 1;
    wf_leader.final_resource_cursor_     = collection.getCursor();
    wf_leader.acquired_resource_outstanding_loans_ =
        collection.getOutstandingLoanCount();
    wf_leader.acquired_resource_collection_ = &collection;
    wf_leader.aggregate_mw_resource_handle_ = std::move(aggregate);
    wf_leader.acquired_batch_topology_       = leader_topology;
    wf_leader.resource_acquired_             = true;
    for (TrialWaveFunction& wavefunction : wf_list)
    {
      wavefunction.acquired_batch_topology_ = leader_topology;
      wavefunction.resource_acquired_       = true;
    }
    return;
  }

  const InlineBatchTopologyState leader_topology =
      wf_leader.captureBatchTopologyState(wf_leader.batch_execution_plan_);
  wf_leader.validateRetainedBatchExecutionBinding();

  if (provenance.state != BatchResourcePreparationState::UNPREPARED ||
      provenance.plan)
    throw std::logic_error(
        "TrialWaveFunction no-policy acquisition received planned ResourceCollection storage");

  // Collect fixed-size acquisition snapshots before the first resource loan.
  // Publishing them after successful acquisition is then allocation-free and
  // noexcept; no component-name vector becomes persistent object state.
  std::vector<std::pair<TrialWaveFunction*, InlineBatchTopologyState>> acquired_topologies;
  acquired_topologies.reserve(wf_list.size() + 1);
  acquired_topologies.emplace_back(&wf_leader, leader_topology);

  std::set<const TrialWaveFunction*> distinct_wavefunctions;
  for (TrialWaveFunction& wavefunction : wf_list)
  {
    // Batch entries are lanes rather than ownership identities.  In
    // particular, compatibility callers may repeat one TrialWaveFunction for
    // multiple ParticleSets, so validate every lane but publish acquisition
    // state only once per distinct object.
    const bool first_occurrence =
        distinct_wavefunctions.insert(&wavefunction).second;
    if (wavefunction.resource_acquired_)
      throw std::logic_error(
          "TrialWaveFunction resources are already acquired for a clone");
    if (!wavefunction.acquired_batch_topology_.sameState({}))
      throw std::logic_error(
          "TrialWaveFunction clone retained a stale resource-acquisition topology");
    if (wavefunction.batch_execution_plan_.get() !=
        wf_leader.batch_execution_plan_.get())
      throw std::invalid_argument(
          "TrialWaveFunction resource clones do not share one batch execution plan identity");
    if (!wavefunction.bound_batch_topology_.aggregate_plan.sameBinding(
            wf_leader.bound_batch_topology_.aggregate_plan) ||
        !wavefunction.bound_batch_topology_.sole_component_plan.sameBinding(
            wf_leader.bound_batch_topology_.sole_component_plan))
      throw std::invalid_argument(
          "TrialWaveFunction resource clones do not share one planned topology binding");
    if (wavefunction.Z.size() != wf_leader.Z.size())
      throw std::invalid_argument(
          "TrialWaveFunction resource clones have different component counts");
    if (static_cast<bool>(wavefunction.twf_fastderiv_) !=
        static_cast<bool>(wf_leader.twf_fastderiv_))
      throw std::invalid_argument(
          "TrialWaveFunction resource clones have inconsistent fast-derivative wrappers");
    const InlineBatchTopologyState topology =
        wavefunction.captureBatchTopologyState(wavefunction.batch_execution_plan_);
    if (!topology.sameState(leader_topology))
      throw std::invalid_argument(
          "TrialWaveFunction resource clones have incompatible component topology");
    wavefunction.validateRetainedBatchExecutionBinding();
    if (first_occurrence && &wavefunction != &wf_leader)
      acquired_topologies.emplace_back(&wavefunction, topology);
  }

  // Revalidate each component's retained participant view before lending any
  // resource.  This catches a stale or independently rebound component while
  // collection and acquisition state are still untouched.
  for (const auto& [wavefunction, topology] : acquired_topologies)
  {
    wavefunction->validateAggregateBatchExecutionPlanBinding(topology.aggregate_plan);
    if (wavefunction->batch_execution_plan_)
      wavefunction->Z.front()->validateBatchExecutionPlanBinding(topology.sole_component_plan);
    else
      for (const auto& component : wavefunction->Z)
        component->validateBatchExecutionPlanBinding({});
  }

  // Build every reference list before lending the first resource.  Besides
  // making all lane-shape validation precede mutation, retaining these lists
  // guarantees the rollback path itself performs no vector allocation.
  std::vector<RefVectorWithLeader<WaveFunctionComponent>> wfc_lists;
  wfc_lists.reserve(wf_leader.Z.size());
  for (int component = 0; component < wf_leader.Z.size(); ++component)
    wfc_lists.push_back(extractWFCRefList(wf_list, component));

  std::optional<RefVectorWithLeader<TWFFastDerivWrapper>> wrapper_list;
  if (wf_leader.twf_fastderiv_)
  {
    wrapper_list.emplace(*wf_leader.twf_fastderiv_);
    wrapper_list->reserve(wf_list.size());
    for (TrialWaveFunction& wavefunction : wf_list)
      wrapper_list->push_back(*wavefunction.twf_fastderiv_);
  }

  const size_t cursor_begin = collection.getCursor();
  int acquired_components   = 0;
  bool acquired_wrapper     = false;

  try
  {
    // First handle WFC resources
    for (int i = 0; i < wf_leader.Z.size(); ++i)
    {
      wf_leader.Z[i]->acquireResource(collection, wfc_lists[i]);
      ++acquired_components;
    }

    // Handle wrapper resources if they exist
    if (wrapper_list)
    {
      wf_leader.twf_fastderiv_->acquireResource(collection, *wrapper_list);
      acquired_wrapper = true;
    }

    for (const auto& [wavefunction, topology] : acquired_topologies)
    {
      wavefunction->acquired_batch_topology_ = topology;
      wavefunction->resource_acquired_       = true;
    }
  }
  catch (...)
  {
    const std::exception_ptr acquisition_failure = std::current_exception();
    collection.rewind(cursor_begin);
    try
    {
      // ResourceCollection takeback traverses from the rewound cursor in the
      // same order as acquisition, rather than in stack order.
      for (int i = 0; i < acquired_components; ++i)
        wf_leader.Z[i]->releaseResource(collection, wfc_lists[i]);
      if (acquired_wrapper)
        wf_leader.twf_fastderiv_->releaseResource(collection, *wrapper_list);
    }
    catch (...)
    {
      collection.rewind(cursor_begin);
      throw;
    }
    collection.rewind(cursor_begin);
    std::rethrow_exception(acquisition_failure);
  }
}

void TrialWaveFunction::releaseResource(ResourceCollection& collection,
                                        const RefVectorWithLeader<TrialWaveFunction>& wf_list)
{
  auto& wf_leader = wf_list.getLeader();

  if (!wf_leader.resource_acquired_)
    throw std::logic_error(
        "TrialWaveFunction resources are not acquired for the leader");
  if (wf_leader.batch_execution_plan_)
  {
    if (wf_leader.acquired_resource_collection_ != &collection ||
        collection.getOutstandingLoanCount() !=
            wf_leader.acquired_resource_outstanding_loans_)
      throw std::logic_error(
          "TrialWaveFunction release received the wrong or changed ResourceCollection");
    if (!wf_leader.aggregate_mw_resource_handle_)
      throw std::logic_error(
          "TrialWaveFunction aggregate resource handle is not acquired");
    const InlineBatchTopologyState leader_topology =
        wf_leader.bound_batch_topology_;
    if (!wf_leader.acquired_batch_topology_.sameState(leader_topology) ||
        wf_leader.multi_particle_proposal_pending_)
      throw std::logic_error(
          "TrialWaveFunction planned release found changed state or a pending proposal");
    wf_leader.validatePreparedBatchExecutionClone(
        leader_topology.aggregate_plan);
    if (!wf_leader.Z.front()->hasBatchExecutionPlanBinding(
            leader_topology.sole_component_plan) ||
        !wf_leader.Z.front()->hasPreparedBatchExecutionClone(
            leader_topology.sole_component_plan))
      throw std::logic_error(
          "TrialWaveFunction planned release found stale leader-child provenance");
    for (TrialWaveFunction& wavefunction : wf_list)
    {
      if (!wavefunction.resource_acquired_ ||
          !wavefunction.acquired_batch_topology_.sameState(leader_topology) ||
          !wavefunction.bound_batch_topology_.sameState(leader_topology) ||
          wavefunction.batch_execution_plan_.get() !=
              wf_leader.batch_execution_plan_.get() ||
          wavefunction.multi_particle_proposal_pending_)
        throw std::logic_error(
            "TrialWaveFunction planned release found incompatible lane state");
      wavefunction.validatePreparedBatchExecutionClone(
          leader_topology.aggregate_plan);
      if (!wavefunction.Z.front()->hasBatchExecutionPlanBinding(
              leader_topology.sole_component_plan) ||
          !wavefunction.Z.front()->hasPreparedBatchExecutionClone(
              leader_topology.sole_component_plan))
        throw std::logic_error(
            "TrialWaveFunction planned release found stale lane-child provenance");
    }

    auto& aggregate =
        wf_leader.aggregate_mw_resource_handle_.getResource();
    if (!aggregate.sameBoundTeam(wf_list))
      throw std::invalid_argument(
          "TrialWaveFunction release requires the exact acquired lane order");
    const std::size_t aggregate_slot = wf_leader.aggregate_resource_cursor_;
    const std::size_t child_slot     = wf_leader.child_resource_cursor_;
    const std::size_t expected_final = wf_leader.final_resource_cursor_;
    collection.rewind(child_slot);
    wf_leader.Z.front()->releaseResource(collection,
                                         aggregate.componentView());
    const std::size_t final_cursor = collection.getCursor();
    if (final_cursor != expected_final)
      throw std::logic_error(
          "TrialWaveFunction child release did not consume its exact resource segment");
    aggregate.resetViews();
    collection.rewind(aggregate_slot);
    collection.takebackResource(wf_leader.aggregate_mw_resource_handle_);
    collection.rewind(final_cursor);

    wf_leader.acquired_batch_topology_.clear();
    wf_leader.resource_acquired_                    = false;
    wf_leader.aggregate_resource_cursor_            = 0;
    wf_leader.child_resource_cursor_                = 0;
    wf_leader.final_resource_cursor_                = 0;
    wf_leader.acquired_resource_outstanding_loans_ = 0;
    wf_leader.acquired_resource_collection_        = nullptr;
    for (TrialWaveFunction& wavefunction : wf_list)
    {
      wavefunction.acquired_batch_topology_.clear();
      wavefunction.resource_acquired_ = false;
    }
    return;
  }

  if (wf_leader.aggregate_mw_resource_handle_)
    throw std::logic_error(
        "TrialWaveFunction no-policy release retained an aggregate resource handle");

  const InlineBatchTopologyState leader_topology =
      wf_leader.captureBatchTopologyState(wf_leader.batch_execution_plan_);
  if (!wf_leader.acquired_batch_topology_.sameState(leader_topology))
    throw std::logic_error(
        "TrialWaveFunction leader topology changed while resources were acquired");
  wf_leader.validateRetainedBatchExecutionBinding();
  std::vector<TrialWaveFunction*> acquired_wavefunctions;
  acquired_wavefunctions.reserve(wf_list.size() + 1);
  acquired_wavefunctions.push_back(&wf_leader);

  std::set<const TrialWaveFunction*> distinct_wavefunctions;
  for (TrialWaveFunction& wavefunction : wf_list)
  {
    const bool first_occurrence =
        distinct_wavefunctions.insert(&wavefunction).second;
    if (!wavefunction.resource_acquired_)
      throw std::logic_error(
          "TrialWaveFunction resources are not acquired for a clone");
    if (wavefunction.batch_execution_plan_.get() !=
        wf_leader.batch_execution_plan_.get())
      throw std::logic_error(
          "TrialWaveFunction acquired clones no longer share one batch execution plan identity");
    if (!wavefunction.bound_batch_topology_.aggregate_plan.sameBinding(
            wf_leader.bound_batch_topology_.aggregate_plan) ||
        !wavefunction.bound_batch_topology_.sole_component_plan.sameBinding(
            wf_leader.bound_batch_topology_.sole_component_plan))
      throw std::logic_error(
          "TrialWaveFunction acquired clones no longer share one planned topology binding");
    if (wavefunction.Z.size() != wf_leader.Z.size())
      throw std::logic_error(
          "TrialWaveFunction acquired clones no longer share component topology");
    const InlineBatchTopologyState topology =
        wavefunction.captureBatchTopologyState(wavefunction.batch_execution_plan_);
    if (!topology.sameState(leader_topology) ||
        !wavefunction.acquired_batch_topology_.sameState(topology))
      throw std::logic_error(
          "TrialWaveFunction component topology changed while resources were acquired");
    wavefunction.validateRetainedBatchExecutionBinding();
    if (static_cast<bool>(wavefunction.twf_fastderiv_) !=
        static_cast<bool>(wf_leader.twf_fastderiv_))
      throw std::logic_error(
          "TrialWaveFunction acquired clones no longer share wrapper topology");
    if (first_occurrence && &wavefunction != &wf_leader)
      acquired_wavefunctions.push_back(&wavefunction);
  }

  if (wf_leader.multi_particle_proposal_pending_)
    throw std::logic_error(
        "Cannot release TrialWaveFunction resources with a pending selected-electron proposal");
  for (const TrialWaveFunction& wavefunction : wf_list)
    if (wavefunction.multi_particle_proposal_pending_)
      throw std::logic_error(
          "Cannot release TrialWaveFunction resources with a pending selected-electron proposal");

  // Materialize every component/wrapper view before the first takeback so an
  // allocation failure cannot leave a partially released wavefunction team.
  std::vector<RefVectorWithLeader<WaveFunctionComponent>> wfc_lists;
  wfc_lists.reserve(wf_leader.Z.size());
  for (int component = 0; component < wf_leader.Z.size(); ++component)
    wfc_lists.push_back(extractWFCRefList(wf_list, component));

  std::optional<RefVectorWithLeader<TWFFastDerivWrapper>> wrapper_list;
  if (wf_leader.twf_fastderiv_)
  {
    wrapper_list.emplace(*wf_leader.twf_fastderiv_);
    wrapper_list->reserve(wf_list.size());
    for (TrialWaveFunction& wavefunction : wf_list)
      wrapper_list->push_back(*wavefunction.twf_fastderiv_);
  }

  // Preserve the historical no-policy choreography: callers position the
  // collection cursor before release and components return their own slots.
  for (int i = 0; i < wf_leader.Z.size(); ++i)
    wf_leader.Z[i]->releaseResource(collection, wfc_lists[i]);

  // Release wrapper resources if they exist
  if (wrapper_list)
    wf_leader.twf_fastderiv_->releaseResource(collection, *wrapper_list);

  for (TrialWaveFunction* wavefunction : acquired_wavefunctions)
  {
    wavefunction->acquired_batch_topology_.clear();
    wavefunction->resource_acquired_ = false;
  }
}


RefVectorWithLeader<WaveFunctionComponent> TrialWaveFunction::extractWFCRefList(
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    int id)
{
  RefVectorWithLeader<WaveFunctionComponent> wfc_list(*wf_list.getLeader().Z[id]);
  wfc_list.reserve(wf_list.size());
  for (TrialWaveFunction& wf : wf_list)
    wfc_list.push_back(*wf.Z[id]);
  return wfc_list;
}

std::vector<WaveFunctionComponent*> TrialWaveFunction::extractWFCPtrList(const UPtrVector<TrialWaveFunction>& g, int id)
{
  std::vector<WaveFunctionComponent*> WFC_list;
  WFC_list.reserve(g.size());
  for (auto& WF : g)
    WFC_list.push_back(WF->Z[id].get());
  return WFC_list;
}

RefVector<ParticleSet::ParticleGradient> TrialWaveFunction::extractGRefList(
    const RefVectorWithLeader<TrialWaveFunction>& wf_list)
{
  RefVector<ParticleSet::ParticleGradient> g_list;
  for (TrialWaveFunction& wf : wf_list)
    g_list.push_back(wf.G);
  return g_list;
}

RefVector<ParticleSet::ParticleLaplacian> TrialWaveFunction::extractLRefList(
    const RefVectorWithLeader<TrialWaveFunction>& wf_list)
{
  RefVector<ParticleSet::ParticleLaplacian> l_list;
  for (TrialWaveFunction& wf : wf_list)
    l_list.push_back(wf.L);
  return l_list;
}

void TrialWaveFunction::initializeTWFFastDerivWrapper(const ParticleSet& P, TWFFastDerivWrapper& twf) const
{
  for (int i = 0; i < Z.size(); ++i)
  {
    if (Z[i]->isFermionic())
    {
      Z[i]->registerTWFFastDerivWrapper(P, twf);
    }
    else
      twf.addJastrow(Z[i].get());
  }
}


// Lazily build the fast-derivative topology only before any plan or loan fixes it.
TWFFastDerivWrapper& TrialWaveFunction::getOrCreateTWFFastDerivWrapper(const ParticleSet& P)
{
  if (!twf_fastderiv_)
  {
    if (resource_acquired_ || batch_execution_plan_)
      throw std::logic_error(
          "Cannot add a fast-derivative wrapper while resources or a batch plan are active");
    twf_fastderiv_ = std::make_unique<TWFFastDerivWrapper>();
    initializeTWFFastDerivWrapper(P, *twf_fastderiv_);
  }
  return *twf_fastderiv_;
}


//explicit instantiations
template void TrialWaveFunction::mw_evalGrad<CoordsType::POS>(const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                                              const RefVectorWithLeader<ParticleSet>& p_list,
                                                              int iat,
                                                              TWFGrads<CoordsType::POS>& grads);
template void TrialWaveFunction::mw_evalGrad<CoordsType::POS_SPIN>(
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int iat,
    TWFGrads<CoordsType::POS_SPIN>& grads);
template void TrialWaveFunction::mw_calcRatioGrad<CoordsType::POS>(
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int iat,
    std::vector<PsiValue>& ratios,
    TWFGrads<CoordsType::POS>& grads);
template void TrialWaveFunction::mw_calcRatioGrad<CoordsType::POS_SPIN>(
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int iat,
    std::vector<PsiValue>& ratios,
    TWFGrads<CoordsType::POS_SPIN>& grads);

} // namespace qmcplusplus
