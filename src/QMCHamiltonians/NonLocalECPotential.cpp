//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//                    Peter W. Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include "NonLocalECPotential.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <DistanceTable.h>
#include <IteratorUtility.h>
#include <ResourceCollection.h>
#include "NonLocalECPComponent.h"
#include "NonLocalTOperator.h"
#include "NLPPJob.h"
#include "NLPPVirtualBatch.h"

namespace qmcplusplus
{

struct NonLocalECPotential::NonLocalECPotentialMultiWalkerResource : public Resource
{
  NonLocalECPotentialMultiWalkerResource(
      std::shared_ptr<const MultiWalkerResourceIdentity> identity,
      MultiWalkerResourceSchema schema)
      : Resource("NonLocalECPotential"), identity(std::move(identity)), schema(schema)
  {
    virtual_batch = std::make_unique<NLPPVirtualBatchStorage>(schema.outer_tile_capacity);
  }

  NonLocalECPotentialMultiWalkerResource(const NonLocalECPotentialMultiWalkerResource& other)
      : Resource("NonLocalECPotential"),
        identity(other.identity),
        schema(other.schema),
        collection(other.collection)
  {
    virtual_batch = std::make_unique<NLPPVirtualBatchStorage>(schema.outer_tile_capacity);
  }

  std::unique_ptr<Resource> makeClone() const override
  { return std::make_unique<NonLocalECPotentialMultiWalkerResource>(*this); }

  /// Operator-family identity and immutable resource shape copied without scratch.
  const std::shared_ptr<const MultiWalkerResourceIdentity> identity;
  const MultiWalkerResourceSchema schema;
  ResourceCollection collection{"NLPPcollection"};
  /// Bounded virtual-knot tile and final per-walker candidate staging.
  std::unique_ptr<NLPPVirtualBatchStorage> virtual_batch;
  /// Private public-job replacements, indexed by walker then electron group.
  std::vector<std::vector<std::vector<NLPPJob<Real>>>> staged_jobs;
  /// Private bidirectional neighbor-list replacements, one per walker.
  std::vector<NeighborListsForPseudo::OwnedLists> staged_neighbor_lists;
  /// Whole-request energy results, published only after every tile validates.
  std::vector<Real> staged_values;
  /// Reused caller-owned component arithmetic scratch.
  std::vector<Real> radial_scratch;
  std::vector<Real> legendre_scratch;
  /// First-tile and current-tile optimistic model-version stamps.
  std::vector<TrialWaveFunction::EvaluationStamp> reference_stamps;
  std::vector<TrialWaveFunction::EvaluationStamp> tile_stamps;
  /// a crowds worth of per particle nonlocal ecp potential values
  Matrix<Real> ve_samples;
  Matrix<Real> vi_samples;
};

/** constructor
 *\param ions the positions of the ions
 *\param els the positions of the electrons
 *\param psi trial wavefunction
 */
NonLocalECPotential::NonLocalECPotential(ParticleSet& ions, ParticleSet& els, bool enable_DLA, bool use_VP)
    : ForceBase(ions, els),
      myRNG(nullptr),
      IonConfig(ions),
      use_DLA(enable_DLA),
      mw_resource_identity_(std::make_shared<MultiWalkerResourceIdentity>()),
      vp_(use_VP ? std::make_unique<VirtualParticleSet>(els) : nullptr),
      Peln(els),
      neighbor_lists(els.getTotalNum(), ions.getTotalNum(), PP)
{
  setEnergyDomain(POTENTIAL);
  twoBodyQuantumDomain(IonConfig, els);
  myTableIndex  = els.addTable(IonConfig);
  auto num_ions = IonConfig.getTotalNum();
  PP.resize(num_ions, nullptr);
  prefix_ = "FNL";
  PPset.resize(IonConfig.getSpeciesSet().getTotalNum());
  PulayTerm.resize(num_ions);
  update_mode_.set(NONLOCAL, 1);
  nlpp_jobs.resize(els.groups());
  for (size_t ig = 0; ig < els.groups(); ig++)
  {
    // this should be enough in most calculations assuming that every electron cannot be in more than two pseudo regions.
    nlpp_jobs[ig].reserve(2 * els.groupsize(ig));
  }
}

NonLocalECPotential::NonLocalECPotential(const NonLocalECPotential& nlpp, ParticleSet& els)
    : ForceBase(nlpp.IonConfig, els),
      myRNG(nullptr),
      IonConfig(nlpp.IonConfig),
      use_DLA(nlpp.use_DLA),
      mw_resource_identity_(nlpp.mw_resource_identity_),
      outer_tile_capacity_(nlpp.outer_tile_capacity_),
      vp_(nlpp.vp_ ? std::make_unique<VirtualParticleSet>(els, nlpp.vp_->getNumDistTables()) : nullptr),
      Peln(els),
      neighbor_lists(els.getTotalNum(), nlpp.IonConfig.getTotalNum(), PP)
{
  setEnergyDomain(POTENTIAL);
  twoBodyQuantumDomain(IonConfig, els);
  myTableIndex  = els.addTable(IonConfig);
  auto num_ions = IonConfig.getTotalNum();
  PP.resize(num_ions, nullptr);
  prefix_ = "FNL";
  PPset.resize(IonConfig.getSpeciesSet().getTotalNum());
  PulayTerm.resize(num_ions);
  update_mode_.set(NONLOCAL, 1);
  nlpp_jobs.resize(els.groups());
  for (size_t ig = 0; ig < els.groups(); ig++)
  {
    // this should be enough in most calculations assuming that every electron cannot be in more than two pseudo regions.
    nlpp_jobs[ig].reserve(2 * els.groupsize(ig));
  }
  for (int ig = 0; ig < nlpp.PPset.size(); ++ig)
    if (nlpp.PPset[ig])
      addComponent(ig, std::make_unique<NonLocalECPComponent>(*nlpp.PPset[ig], els));
}

NonLocalECPotential::~NonLocalECPotential() = default;

NonLocalECPotential::MultiWalkerResourceSchema NonLocalECPotential::multiWalkerResourceSchema() const noexcept
{
  return {bool(vp_), vp_ ? static_cast<std::size_t>(vp_->getNumDistTables()) : 0,
          static_cast<std::size_t>(Peln.getTotalNum()),
          static_cast<std::size_t>(Peln.groups()), static_cast<std::size_t>(IonConfig.getTotalNum()),
          outer_tile_capacity_};
}

void NonLocalECPotential::setOuterTileCapacityForTesting(std::size_t capacity)
{
  if (capacity == 0)
    throw std::invalid_argument("NonLocalECPotential outer tile capacity must be nonzero.");
  if (mw_res_handle_)
    throw std::logic_error("NonLocalECPotential outer tile capacity cannot change while a resource is acquired.");
  outer_tile_capacity_ = capacity;
}

void NonLocalECPotential::resizeMultiWalkerListenerScratchForTesting(std::size_t walkers,
                                                                     std::size_t electrons,
                                                                     std::size_t ions)
{
  auto& resource = mw_res_handle_.getResource();
  resource.ve_samples.resize(walkers, electrons);
  resource.vi_samples.resize(walkers, ions);
}

std::pair<std::size_t, std::size_t> NonLocalECPotential::multiWalkerListenerScratchSizesForTesting() const
{
  const auto& resource = mw_res_handle_.getResource();
  return {resource.ve_samples.size(), resource.vi_samples.size()};
}

#if !defined(REMOVE_TRACEMANAGER)
void NonLocalECPotential::contributeParticleQuantities() { request_.contribute_array(name_); }

void NonLocalECPotential::checkoutParticleQuantities(TraceManager& tm)
{
  streaming_particles_ = request_.streaming_array(name_);
  if (streaming_particles_)
  {
    Ve_sample = tm.checkout_real<1>(name_, Peln);
    Vi_sample = tm.checkout_real<1>(name_, IonConfig);
  }
}

void NonLocalECPotential::deleteParticleQuantities()
{
  if (streaming_particles_)
  {
    delete Ve_sample;
    delete Vi_sample;
  }
}
#endif

NonLocalECPotential::Return_t NonLocalECPotential::evaluate(TrialWaveFunction& psi, ParticleSet& P)
{
  evaluateImpl(psi, P, false);
  return value_;
}

NonLocalECPotential::Return_t NonLocalECPotential::evaluateDeterministic(TrialWaveFunction& psi, ParticleSet& P)
{
  evaluateImpl(psi, P, false, true);
  return value_;
}

void NonLocalECPotential::mw_evaluate(const RefVectorWithLeader<OperatorBase>& o_list,
                                      const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                      const RefVectorWithLeader<ParticleSet>& p_list) const
{ mw_evaluateImpl(o_list, wf_list, p_list, false, std::nullopt); }

void NonLocalECPotential::mw_evaluateWithParameterDerivatives(
    const RefVectorWithLeader<OperatorBase>& o_list,
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const OptVariables& optvars,
    const RecordArray<ValueType>& dlogpsi,
    RecordArray<ValueType>& dhpsioverpsi) const
{
  auto& leader = o_list.getCastedLeader<NonLocalECPotential>();
  assert(this == &leader);
  const std::size_t walker_count = o_list.size();
  if (wf_list.size() != walker_count || p_list.size() != walker_count ||
      dlogpsi.getNumOfEntries() != walker_count || dhpsioverpsi.getNumOfEntries() != walker_count ||
      dlogpsi.getNumOfParams() != dhpsioverpsi.getNumOfParams())
    throw std::invalid_argument("NonLocalECPotential derivative batch has inconsistent shapes");
  if (walker_count == 0)
    return;

  // Direct unit-test and legacy callers may invoke this interface without the
  // normal Hamiltonian ResourceCollection lifecycle.  They must retain the
  // established serialized behavior rather than dereferencing an empty handle.
  if (!leader.mw_res_handle_)
  {
    leader.OperatorBase::mw_evaluateWithParameterDerivatives(o_list, wf_list, p_list, optvars, dlogpsi,
                                                              dhpsioverpsi);
    return;
  }

  // The legacy non-VP route mutates one accepted configuration per knot and
  // needs its reference dlogpsi row. Preserve it through OperatorBase's
  // serialized compatibility implementation.
  for (std::size_t walker = 0; walker < walker_count; ++walker)
    if (!o_list.getCastedElement<NonLocalECPotential>(walker).vp_)
    {
      leader.OperatorBase::mw_evaluateWithParameterDerivatives(o_list, wf_list, p_list, optvars, dlogpsi,
                                                                dhpsioverpsi);
      return;
    }

  const int parameter_count = dhpsioverpsi.getNumOfParams();
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    auto& potential       = o_list.getCastedElement<NonLocalECPotential>(walker);
    const ParticleSet& ps = p_list[walker];
    for (const auto& component : potential.PPset)
      if (component)
        component->rotateQuadratureGrid(generateRandomRotationMatrix(*potential.myRNG));

    const auto& distance_table = ps.getDistTableAB(potential.myTableIndex);
    for (int group = 0; group < ps.groups(); ++group)
    {
      auto& jobs = potential.nlpp_jobs[group];
      jobs.clear();
      for (int electron = ps.first(group); electron < ps.last(group); ++electron)
      {
        const auto& distances     = distance_table.getDistRow(electron);
        const auto& displacements = distance_table.getDisplRow(electron);
        for (int ion = 0; ion < potential.PP.size(); ++ion)
          if (potential.PP[ion] && distances[ion] < potential.PP[ion]->getRmax())
            jobs.emplace_back(ion, electron, distances[ion], -displacements[ion]);
      }
    }
    potential.value_ = 0.0;
  }

  const auto leader_component =
      std::find_if(leader.PPset.begin(), leader.PPset.end(), [](const auto& component) { return bool(component); });
  if (leader_component == leader.PPset.end())
    return;

  std::vector<Real> pair_potentials(walker_count);
  auto& shared_collection = leader.mw_res_handle_.getResource().collection;
  RefVector<NonLocalECPotential> potential_batch;
  RefVectorWithLeader<NonLocalECPComponent> component_batch(**leader_component);
  RefVectorWithLeader<ParticleSet> particle_batch(p_list.getLeader());
  RefVectorWithLeader<VirtualParticleSet> virtual_particle_batch(*leader.vp_);
  RefVectorWithLeader<TrialWaveFunction> wavefunction_batch(wf_list.getLeader());
  RefVector<const NLPPJob<Real>> job_batch;
  std::vector<TrialWaveFunction::ParameterDerivativeView> derivative_batch;

  potential_batch.reserve(walker_count);
  component_batch.reserve(walker_count);
  particle_batch.reserve(walker_count);
  virtual_particle_batch.reserve(walker_count);
  wavefunction_batch.reserve(walker_count);
  job_batch.reserve(walker_count);
  derivative_batch.reserve(walker_count);

  // A clone owns one VirtualParticleSet, so each compact subbatch contains at
  // most one active ion-electron job from a walker. Unequal job counts produce
  // naturally ragged batches without padding or reordering quadrature points.
  for (int group = 0; group < p_list.getLeader().groups(); ++group)
  {
    TrialWaveFunction::mw_prepareGroup(wf_list, p_list, group);
    std::size_t maximum_jobs = 0;
    for (std::size_t walker = 0; walker < walker_count; ++walker)
      maximum_jobs = std::max(maximum_jobs,
                              o_list.getCastedElement<NonLocalECPotential>(walker).nlpp_jobs[group].size());

    for (std::size_t job_index = 0; job_index < maximum_jobs; ++job_index)
    {
      potential_batch.clear();
      component_batch.clear();
      particle_batch.clear();
      virtual_particle_batch.clear();
      wavefunction_batch.clear();
      job_batch.clear();
      derivative_batch.clear();

      for (std::size_t walker = 0; walker < walker_count; ++walker)
      {
        auto& potential = o_list.getCastedElement<NonLocalECPotential>(walker);
        if (job_index >= potential.nlpp_jobs[group].size())
          continue;

        const auto& job = potential.nlpp_jobs[group][job_index];
        potential_batch.push_back(std::ref(potential));
        component_batch.push_back(std::ref(*potential.PP[job.ion_id]));
        particle_batch.push_back(std::ref(p_list[walker]));
        virtual_particle_batch.push_back(std::ref(*potential.vp_));
        wavefunction_batch.push_back(std::ref(wf_list[walker]));
        job_batch.push_back(std::cref(job));
        derivative_batch.push_back({dhpsioverpsi[walker], static_cast<std::size_t>(parameter_count)});
      }

      NonLocalECPComponent::mw_evaluateValueAndDerivatives(
          component_batch, particle_batch, virtual_particle_batch, wavefunction_batch, job_batch, optvars,
          derivative_batch, pair_potentials, shared_collection);
      for (std::size_t batch_index = 0; batch_index < potential_batch.size(); ++batch_index)
        potential_batch[batch_index].get().value_ += pair_potentials[batch_index];
    }
  }
}

NonLocalECPotential::Return_t NonLocalECPotential::evaluateWithToperator(TrialWaveFunction& psi, ParticleSet& P)
{
  evaluateImpl(psi, P, true);
  return value_;
}

void NonLocalECPotential::mw_evaluateWithToperator(const RefVectorWithLeader<OperatorBase>& o_list,
                                                   const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                                   const RefVectorWithLeader<ParticleSet>& p_list) const
{ mw_evaluateImpl(o_list, wf_list, p_list, true, std::nullopt); }

void NonLocalECPotential::mw_evaluatePerParticle(const RefVectorWithLeader<OperatorBase>& o_list,
                                                 const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                                 const RefVectorWithLeader<ParticleSet>& p_list,
                                                 const std::vector<ListenerVector<Real>>& listeners,
                                                 const std::vector<ListenerVector<Real>>& listeners_ions) const
{
  std::optional<ListenerOption<Real>> l_opt(std::in_place, listeners, listeners_ions);
  mw_evaluateImpl(o_list, wf_list, p_list, false, l_opt);
}

void NonLocalECPotential::mw_evaluatePerParticleWithToperator(
    const RefVectorWithLeader<OperatorBase>& o_list,
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const std::vector<ListenerVector<Real>>& listeners,
    const std::vector<ListenerVector<Real>>& listeners_ions) const
{
  std::optional<ListenerOption<Real>> l_opt(std::in_place, listeners, listeners_ions);
  mw_evaluateImpl(o_list, wf_list, p_list, true, l_opt);
}

void NonLocalECPotential::mw_evaluateImplFlattenedVP(
    const RefVectorWithLeader<OperatorBase>& o_list,
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    bool compute_txy_all,
    const std::optional<ListenerOption<Real>>& listeners,
    bool keep_grid)
{
  auto& leader             = o_list.getCastedLeader<NonLocalECPotential>();
  ParticleSet& pset_leader = p_list.getLeader();
  const std::size_t walker_count = o_list.size();

  if (wf_list.size() != walker_count || p_list.size() != walker_count)
    throw std::invalid_argument("NonLocalECPotential flattened crowd lists have inconsistent sizes.");
  if (!leader.vp_ || (leader.use_DLA && compute_txy_all))
    throw std::logic_error("NonLocalECPotential flattened evaluator received an unsupported localization mode.");
  if (!leader.mw_res_handle_)
    throw std::logic_error("NonLocalECPotential flattened evaluation requires an acquired crowd resource.");

  auto& resource = leader.mw_res_handle_.getResource();
  if (!resource.virtual_batch || resource.virtual_batch->tileCapacity() != leader.outer_tile_capacity_)
    throw std::logic_error("NonLocalECPotential flattened resource has an incompatible outer tile.");

  // Validate the complete crowd before rotating a grid or changing reusable
  // staging. Resource acquisition already establishes the family/schema; the
  // local checks make direct test calls fail deterministically as well.
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    const auto& potential = o_list.getCastedElement<NonLocalECPotential>(walker);
    const ParticleSet& particles = p_list[walker];
    if (!potential.vp_ || potential.use_DLA != leader.use_DLA)
      throw std::invalid_argument("NonLocalECPotential flattened crowd has incompatible localization state.");
    if (particles.groups() != pset_leader.groups() ||
        particles.getTotalNum() != pset_leader.getTotalNum() ||
        potential.nlpp_jobs.size() != static_cast<std::size_t>(particles.groups()) ||
        potential.PP.size() != static_cast<std::size_t>(leader.IonConfig.getTotalNum()))
      throw std::invalid_argument("NonLocalECPotential flattened crowd has incompatible particle shapes.");
    if (!keep_grid && potential.myRNG == nullptr)
      throw std::logic_error("NonLocalECPotential grid rotation requires a random-number generator.");
  }

  // Prepare grow-only private state before consuming an RNG value. None of
  // these allocations can expose a partial Hamiltonian result.
  resource.staged_jobs.resize(walker_count);
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    auto& walker_jobs = resource.staged_jobs[walker];
    walker_jobs.resize(static_cast<std::size_t>(p_list[walker].groups()));
    for (auto& group_jobs : walker_jobs)
      group_jobs.clear();
  }

  if (resource.staged_neighbor_lists.size() != walker_count)
  {
    std::vector<NeighborListsForPseudo::OwnedLists> rebuilt;
    rebuilt.reserve(walker_count);
    for (std::size_t walker = 0; walker < walker_count; ++walker)
      rebuilt.emplace_back(
          o_list.getCastedElement<NonLocalECPotential>(walker).neighbor_lists.makeOwnedLists());
    resource.staged_neighbor_lists.swap(rebuilt);
  }
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    o_list.getCastedElement<NonLocalECPotential>(walker).neighbor_lists.validateOwnedLists(
        resource.staged_neighbor_lists[walker]);
    resource.staged_neighbor_lists[walker].clear();
  }

  resource.staged_values.assign(walker_count, Real(0));
  resource.reference_stamps.clear();
  resource.tile_stamps.clear();
  resource.virtual_batch->reset(walker_count, compute_txy_all);

  std::size_t maximum_channels = 0;
  std::size_t maximum_legendre = 0;
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    const auto& potential = o_list.getCastedElement<NonLocalECPotential>(walker);
    for (const auto& component : potential.PPset)
      if (component)
      {
        maximum_channels = std::max(maximum_channels, static_cast<std::size_t>(component->getNchannel()));
        maximum_legendre =
            std::max(maximum_legendre, static_cast<std::size_t>(component->getLmax() + 1));
      }
  }
  resource.radial_scratch.resize(maximum_channels);
  resource.legendre_scratch.resize(maximum_legendre);

  auto& ve_samples = resource.ve_samples;
  auto& vi_samples = resource.vi_samples;
  if (listeners)
  {
    ve_samples.resize(walker_count, pset_leader.getTotalNum());
    vi_samples.resize(walker_count, leader.IonConfig.getTotalNum());
    ve_samples = Real(0);
    vi_samples = Real(0);
  }
  const std::string ion_listener_operator_name = listeners ? leader.getName() + "Ion" : std::string{};

  // Preserve the established per-walker/species grid-rotation order, then
  // build private jobs and neighbors in the scalar electron/ion order.
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    auto& potential          = o_list.getCastedElement<NonLocalECPotential>(walker);
    const ParticleSet& particles = p_list[walker];
    if (!keep_grid)
      for (const auto& component : potential.PPset)
        if (component)
          component->rotateQuadratureGrid(generateRandomRotationMatrix(*potential.myRNG));

    const auto& distance_table = particles.getDistTableAB(potential.myTableIndex);
    for (int group = 0; group < particles.groups(); ++group)
    {
      auto& group_jobs = resource.staged_jobs[walker][group];
      for (int electron = particles.first(group); electron < particles.last(group); ++electron)
      {
        const auto& distances     = distance_table.getDistRow(electron);
        const auto& displacements = distance_table.getDisplRow(electron);
        for (int ion = 0; ion < potential.PP.size(); ++ion)
          if (potential.PP[ion] && distances[ion] < potential.PP[ion]->getRmax())
          {
            resource.staged_neighbor_lists[walker].addElecIonPair(electron, ion);
            group_jobs.emplace_back(ion, electron, distances[ion], -displacements[ion]);
          }
      }
    }
  }

  // G2 owns canonical group/walker/job identity. Physical distance data stay
  // in the staged legacy jobs and are resolved by walkerJobOrdinal below.
  for (int group = 0; group < pset_leader.groups(); ++group)
    for (std::size_t walker = 0; walker < walker_count; ++walker)
    {
      const auto& potential = o_list.getCastedElement<NonLocalECPotential>(walker);
      const auto& group_jobs = resource.staged_jobs[walker][group];
      for (std::size_t ordinal = 0; ordinal < group_jobs.size(); ++ordinal)
      {
        const auto& job = group_jobs[ordinal];
        const int knot_count = potential.PP[job.ion_id]->getNknot();
        if (knot_count <= 0)
          throw std::logic_error("NonLocalECPotential encountered an empty quadrature grid.");
        resource.virtual_batch->appendJob(
            {group, static_cast<int>(walker), job.ion_id, job.electron_id, ordinal,
             static_cast<std::size_t>(knot_count)});
      }
    }
  resource.virtual_batch->seal();

  RefVectorWithLeader<VirtualParticleSet> vp_scratch_list(*leader.vp_);
  vp_scratch_list.reserve(walker_count);
  for (std::size_t walker = 0; walker < walker_count; ++walker)
    vp_scratch_list.push_back(*o_list.getCastedElement<NonLocalECPotential>(walker).vp_);
  {
    // Nested VP resources are needed only while descriptors are materialized
    // and consumed. Release them before public-state commit and callbacks.
    ResourceCollectionTeamLock<VirtualParticleSet> vp_resource_lock(resource.collection, vp_scratch_list);

    bool have_reference_stamps = false;
    bool tile_available        = resource.virtual_batch->packNextTile();
    for (int group = 0; group < pset_leader.groups(); ++group)
    {
      // Keep this boundary even for an empty group, matching the reference path.
      TrialWaveFunction::mw_prepareGroup(wf_list, p_list, group);

      while (tile_available)
      {
        const auto& segments = resource.virtual_batch->tileSegments();
        if (segments.empty())
          throw std::logic_error("NonLocalECPotential packed an empty active tile.");
        if (segments.front().groupId() > group)
          break;
        if (segments.front().groupId() != group)
          throw std::logic_error("NonLocalECPotential tile traversal left canonical group order.");

        auto& absolute_positions = resource.virtual_batch->mutableTileAbsolutePositions();
        auto& deltas             = resource.virtual_batch->mutableTileDeltas();
        auto& bare_weights       = resource.virtual_batch->mutableTileBareWeights();
        for (const auto& segment : segments)
        {
          const std::size_t walker = static_cast<std::size_t>(segment.walkerId());
          auto& potential          = o_list.getCastedElement<NonLocalECPotential>(walker);
          const auto& group_jobs   = resource.staged_jobs[walker][group];
          if (segment.walkerJobOrdinal() >= group_jobs.size())
            throw std::logic_error("NonLocalECPotential tile references an absent staged job.");
          const auto& job = group_jobs[segment.walkerJobOrdinal()];
          if (job.ion_id != segment.ionId() || job.electron_id != segment.electronId())
            throw std::logic_error("NonLocalECPotential tile identity disagrees with its staged job.");

          potential.PP[job.ion_id]->buildQuadraturePointRange(
              job.ion_elec_dist, job.ion_elec_displ, p_list[walker].R[job.electron_id],
              segment.firstKnot(), segment.knotCount(), segment.tileOffset(), deltas,
              absolute_positions, bare_weights, resource.radial_scratch, resource.legendre_scratch);
        }

        resource.virtual_batch->finalizeTileInput();
        const VirtualParticleBatch descriptor = resource.virtual_batch->makeVirtualParticleBatch();
        descriptor.validateFor(p_list, static_cast<std::size_t>(leader.IonConfig.getTotalNum()));

        auto& ratios = resource.virtual_batch->mutableTileRatios();
        resource.tile_stamps.clear();
        TrialWaveFunction::mw_evaluateVirtualRatios(
            wf_list, p_list, vp_scratch_list, descriptor, ratios, resource.tile_stamps,
            leader.use_DLA ? TrialWaveFunction::ComputeType::FERMIONIC
                           : TrialWaveFunction::ComputeType::ALL);

        if (!have_reference_stamps)
        {
          resource.reference_stamps = resource.tile_stamps;
          have_reference_stamps     = true;
        }
        else if (resource.tile_stamps != resource.reference_stamps)
          throw std::runtime_error(
              "NonLocalECPotential observed different wavefunction parameter versions across outer tiles.");

        auto& transformed_weights = resource.virtual_batch->mutableTileTransformedWeights();
        for (const auto& segment : segments)
        {
          const std::size_t walker = static_cast<std::size_t>(segment.walkerId());
          std::vector<NonLocalData>* candidates =
              compute_txy_all ? &resource.virtual_batch->walkerCandidates(walker) : nullptr;
          Real& pair_potential = resource.virtual_batch->jobPairPotential(segment.globalJobId());
          NonLocalECPComponent::reduceQuadraturePointRange(
              segment.electronId(), segment.tileOffset(), segment.knotCount(),
              resource.virtual_batch->tileDeltas(), resource.virtual_batch->tileBareWeights(), ratios,
              nullptr, segment.tileOffset(), transformed_weights, segment.walkerKnotOffset(), candidates,
              pair_potential);

          if (segment.endsJob())
          {
            resource.staged_values[walker] += pair_potential;
            if (listeners)
            {
              ve_samples(walker, segment.electronId()) += Real(0.5) * pair_potential;
              vi_samples(walker, segment.ionId()) += Real(0.5) * pair_potential;
            }
          }
        }

        tile_available = resource.virtual_batch->packNextTile();
      }
    }
    if (tile_available)
      throw std::logic_error("NonLocalECPotential tile traversal exceeded the particle-group range.");
  }

  // Validate every destination while failure can still leave public state
  // untouched. The subsequent publication contains only noexcept swaps and
  // scalar assignments.
  resource.virtual_batch->validateLogicalOutputExtents();
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    auto& potential = o_list.getCastedElement<NonLocalECPotential>(walker);
    if (potential.nlpp_jobs.size() != resource.staged_jobs[walker].size())
      throw std::logic_error("NonLocalECPotential staged job-group extent changed before publication.");
    potential.neighbor_lists.validateOwnedLists(resource.staged_neighbor_lists[walker]);
  }

  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    auto& potential = o_list.getCastedElement<NonLocalECPotential>(walker);
    for (std::size_t group = 0; group < potential.nlpp_jobs.size(); ++group)
      potential.nlpp_jobs[group].swap(resource.staged_jobs[walker][group]);
    const bool neighbor_swap_succeeded =
        potential.neighbor_lists.swapOwnedLists(resource.staged_neighbor_lists[walker]);
    assert(neighbor_swap_succeeded);
    static_cast<void>(neighbor_swap_succeeded);
    if (compute_txy_all)
      potential.tmove_xy_all_.swap(resource.virtual_batch->walkerCandidates(walker));
    potential.value_ = resource.staged_values[walker];
  }

  // Listener reporting is the external commit phase. An arbitrary callback
  // cannot be rolled back, but no callback is entered before internal commit.
  if (listeners)
  {
    const int electron_count = pset_leader.getTotalNum();
    const int ion_count      = leader.IonConfig.getTotalNum();
    for (std::size_t walker = 0; walker < walker_count; ++walker)
    {
      Vector<Real> electron_row(ve_samples.begin(walker), electron_count);
      Vector<Real> ion_row(vi_samples.begin(walker), ion_count);
      for (const ListenerVector<Real>& listener : listeners->electron_values)
        listener.report(walker, leader.getName(), electron_row);
      for (const ListenerVector<Real>& listener : listeners->ion_values)
        listener.report(walker, ion_listener_operator_name, ion_row);
    }
    ve_samples = Real(0);
    vi_samples = Real(0);
  }
}

void NonLocalECPotential::evaluateImpl(TrialWaveFunction& psi, ParticleSet& P, bool compute_txy_all, bool keep_grid)
{
  if (compute_txy_all)
    tmove_xy_all_.clear();

  value_ = 0.0;
#if !defined(REMOVE_TRACEMANAGER)
  auto& Ve_samp = *Ve_sample;
  auto& Vi_samp = *Vi_sample;
  if (streaming_particles_)
  {
    Ve_samp = 0.0;
    Vi_samp = 0.0;
  }
#endif

  if (!keep_grid)
    for (int ipp = 0; ipp < PPset.size(); ipp++)
      if (PPset[ipp])
        PPset[ipp]->rotateQuadratureGrid(generateRandomRotationMatrix(*myRNG));

  neighbor_lists.clear();
  const auto& myTable = P.getDistTableAB(myTableIndex);
  for (int ig = 0; ig < P.groups(); ++ig) //loop over species
  {
    psi.prepareGroup(P, ig);
    for (int jel = P.first(ig); jel < P.last(ig); ++jel)
    {
      const auto& dist  = myTable.getDistRow(jel);
      const auto& displ = myTable.getDisplRow(jel);
      for (int iat = 0; iat < PP.size(); iat++)
        if (PP[iat] != nullptr && dist[iat] < PP[iat]->getRmax())
        {
          Real pairpot =
              PP[iat]->evaluateOne(P, vp_ ? makeOptionalRef<VirtualParticleSet>(*vp_) : std::nullopt, iat, psi, jel,
                                   dist[iat], -displ[iat],
                                   compute_txy_all ? makeOptionalRef<std::vector<NonLocalData>>(tmove_xy_all_)
                                                   : std::nullopt,
                                   use_DLA);
          neighbor_lists.addElecIonPair(jel, iat);

          value_ += pairpot;
#if !defined(REMOVE_TRACEMANAGER)
          if (streaming_particles_)
          {
            Ve_samp(jel) += 0.5 * pairpot;
            Vi_samp(iat) += 0.5 * pairpot;
          }
#endif
        }
    }
  }

#if !defined(TRACE_CHECK) && !defined(REMOVE_TRACEMANAGER)
  if (streaming_particles_)
  {
    Return_t Vnow = value_;
    Real Visum    = Vi_sample->sum();
    Real Vesum    = Ve_sample->sum();
    Real Vsum     = Vesum + Visum;
    if (std::abs(Vsum - Vnow) > TraceManager::trace_tol)
    {
      app_log() << "accumtest: NonLocalECPotential::evaluate()" << std::endl;
      app_log() << "accumtest:   tot:" << Vnow << std::endl;
      app_log() << "accumtest:   sum:" << Vsum << std::endl;
      APP_ABORT("Trace check failed");
    }
    if (std::abs(Vesum - Visum) > TraceManager::trace_tol)
    {
      app_log() << "sharetest: NonLocalECPotential::evaluate()" << std::endl;
      app_log() << "sharetest:   e share:" << Vesum << std::endl;
      app_log() << "sharetest:   i share:" << Visum << std::endl;
      APP_ABORT("Trace check failed");
    }
  }
#endif
}

void NonLocalECPotential::mw_evaluateImpl(const RefVectorWithLeader<OperatorBase>& o_list,
                                          const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                          const RefVectorWithLeader<ParticleSet>& p_list,
                                          bool compute_txy_all,
                                          const std::optional<ListenerOption<Real>> listeners,
                                          bool keep_grid)
{
  auto& O_leader           = o_list.getCastedLeader<NonLocalECPotential>();
  ParticleSet& pset_leader = p_list.getLeader();
  const size_t nw          = o_list.size();

  // TMDLA needs independent fermionic/nonfermionic streams and remains on the
  // established wavefront until the next boundary. The non-VP path is also an
  // unchanged compatibility reference.
  if (O_leader.vp_ && !(O_leader.use_DLA && compute_txy_all))
  {
    mw_evaluateImplFlattenedVP(o_list, wf_list, p_list, compute_txy_all, listeners, keep_grid);
    return;
  }

  for (size_t iw = 0; iw < nw; iw++)
  {
    auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
    const ParticleSet& P(p_list[iw]);

    if (compute_txy_all)
      O.tmove_xy_all_.clear();

    if (!keep_grid)
      for (int ipp = 0; ipp < O.PPset.size(); ipp++)
        if (O.PPset[ipp])
          O.PPset[ipp]->rotateQuadratureGrid(generateRandomRotationMatrix(*O.myRNG));

    O.neighbor_lists.clear();
    const auto& myTable = P.getDistTableAB(O.myTableIndex);
    for (int ig = 0; ig < P.groups(); ++ig) //loop over species
    {
      auto& joblist = O.nlpp_jobs[ig];
      joblist.clear();

      for (int jel = P.first(ig); jel < P.last(ig); ++jel)
      {
        const auto& dist  = myTable.getDistRow(jel);
        const auto& displ = myTable.getDisplRow(jel);
        for (int iat = 0; iat < O.PP.size(); iat++)
          if (O.PP[iat] != nullptr && dist[iat] < O.PP[iat]->getRmax())
          {
            O.neighbor_lists.addElecIonPair(jel, iat);
            joblist.emplace_back(iat, jel, dist[iat], -displ[iat]);
          }
      }
    }

    O.value_ = 0.0;
  }

  if (listeners)
  {
    auto& ve_samples = O_leader.mw_res_handle_.getResource().ve_samples;
    auto& vi_samples = O_leader.mw_res_handle_.getResource().vi_samples;
    ve_samples.resize(nw, pset_leader.getTotalNum());
    vi_samples.resize(nw, O_leader.IonConfig.getTotalNum());
  }

  // the VP of the first NonLocalECPComponent is responsible for holding the shared resource.
  auto pp_component = std::find_if(O_leader.PPset.begin(), O_leader.PPset.end(), [](auto& ptr) { return bool(ptr); });
  assert(pp_component != std::end(O_leader.PPset));

  RefVector<NonLocalECPotential> ecp_potential_list;
  RefVectorWithLeader<NonLocalECPComponent> ecp_component_list(**pp_component);
  RefVectorWithLeader<ParticleSet> pset_list(pset_leader);
  RefVectorWithLeader<TrialWaveFunction> psi_list(wf_list.getLeader());

  RefVector<const NLPPJob<Real>> batch_list;
  RefVector<VirtualParticleSet> vp_list;
  std::vector<Real> pairpots(nw);
  std::vector<size_t> batch_walker_indices;
  RefVector<std::vector<NonLocalData>> tmove_xy_all_batch_list;

  ecp_potential_list.reserve(nw);
  ecp_component_list.reserve(nw);
  pset_list.reserve(nw);
  psi_list.reserve(nw);
  batch_list.reserve(nw);
  batch_walker_indices.reserve(nw);
  tmove_xy_all_batch_list.reserve(nw);

  for (int ig = 0; ig < pset_leader.groups(); ++ig) //loop over species
  {
    TrialWaveFunction::mw_prepareGroup(wf_list, p_list, ig);

    // find the max number of jobs of all the walkers
    size_t max_num_jobs = 0;
    for (size_t iw = 0; iw < nw; iw++)
    {
      const auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
      max_num_jobs  = std::max(max_num_jobs, O.nlpp_jobs[ig].size());
    }

    for (size_t jobid = 0; jobid < max_num_jobs; jobid++)
    {
      ecp_potential_list.clear();
      ecp_component_list.clear();
      pset_list.clear();
      psi_list.clear();
      batch_list.clear();
      batch_walker_indices.clear();
      tmove_xy_all_batch_list.clear();
      vp_list.reserve(nw);
      for (size_t iw = 0; iw < nw; iw++)
      {
        auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
        if (jobid < O.nlpp_jobs[ig].size())
        {
          const auto& job = O.nlpp_jobs[ig][jobid];
          ecp_potential_list.push_back(O);
          ecp_component_list.push_back(*O.PP[job.ion_id]);
          pset_list.push_back(p_list[iw]);
          if (O.vp_)
            vp_list.push_back(*O.vp_);
          psi_list.push_back(wf_list[iw]);
          batch_list.push_back(job);
          batch_walker_indices.push_back(iw);
          if (compute_txy_all)
            tmove_xy_all_batch_list.push_back(O.tmove_xy_all_);
        }
      }

      if (O_leader.vp_)
        NonLocalECPComponent::mw_evaluateOne(ecp_component_list, pset_list, {*O_leader.vp_, std::move(vp_list)},
                                             psi_list, batch_list, pairpots, tmove_xy_all_batch_list,
                                             O_leader.mw_res_handle_.getResource().collection, O_leader.use_DLA);
      else
        // The batch lists are compacted: a walker with no job at this jobid
        // is absent, so they can be shorter than nw. Index by batch slot,
        // matching the accumulation loop below.
        for (size_t j = 0; j < ecp_component_list.size(); j++)
          pairpots[j] =
              ecp_component_list[j].evaluateOne(pset_list[j], std::nullopt, batch_list[j].get().ion_id, psi_list[j],
                                                batch_list[j].get().electron_id, batch_list[j].get().ion_elec_dist,
                                                batch_list[j].get().ion_elec_displ,
                                                compute_txy_all ? makeOptionalRef<std::vector<NonLocalData>>(
                                                                      tmove_xy_all_batch_list[j])
                                                                : std::nullopt,
                                                O_leader.use_DLA);

      for (size_t j = 0; j < ecp_potential_list.size(); j++)
      {
        NonLocalECPotential& ecp = ecp_potential_list[j];
        ecp.value_ += pairpots[j];

        if (listeners)
        {
          auto& ve_samples = O_leader.mw_res_handle_.getResource().ve_samples;
          auto& vi_samples = O_leader.mw_res_handle_.getResource().vi_samples;
          const size_t iw  = batch_walker_indices[j];
          ve_samples(iw, batch_list[j].get().electron_id) += 0.5 * pairpots[j];
          vi_samples(iw, batch_list[j].get().ion_id) += 0.5 * pairpots[j];
        }

#ifdef DEBUG_NLPP_BATCHED
        std::vector<NonLocalData> tmove_xy_dummy;
        Real check_value =
            ecp_component_list[j].evaluateOne(pset_list[j],
                                              ecp.vp_ ? makeOptionalRef<VirtualParticleSet>(*ecp.vp_) : std::nullopt,
                                              batch_list[j].get().ion_id, psi_list[j], batch_list[j].get().electron_id,
                                              batch_list[j].get().ion_elec_dist, batch_list[j].get().ion_elec_displ,
                                              compute_txy_all
                                                  ? makeOptionalRef<std::vector<NonLocalData>>(tmove_xy_dummy)
                                                  : std::nullopt,
                                              O_leader.use_DLA);
        if (std::abs(check_value - pairpots[j]) > 1e-5)
          std::cout << "check " << check_value << " wrong " << pairpots[j] << " diff "
                    << std::abs(check_value - pairpots[j]) << std::endl;
#endif
      }
    }
  }

  if (listeners)
  {
    // Motivation for this repeated definition is to make factoring this listener code out easy
    // and making it ignorable when reading this function.
    auto& ve_samples  = O_leader.mw_res_handle_.getResource().ve_samples;
    auto& vi_samples  = O_leader.mw_res_handle_.getResource().vi_samples;
    int num_electrons = pset_leader.getTotalNum();
    const std::string ion_listener_operator_name{O_leader.getName() + "Ion"};
    for (int iw = 0; iw < nw; ++iw)
    {
      Vector<Real> ve_sample(ve_samples.begin(iw), num_electrons);
      Vector<Real> vi_sample(vi_samples.begin(iw), O_leader.IonConfig.getTotalNum());
      for (const ListenerVector<Real>& listener : listeners->electron_values)
        listener.report(iw, O_leader.getName(), ve_sample);

      for (const ListenerVector<Real>& listener : listeners->ion_values)
        listener.report(iw, ion_listener_operator_name, vi_sample);
    }
    ve_samples = 0.0;
    vi_samples = 0.0;
  }
}

void NonLocalECPotential::evaluateIonDerivs(ParticleSet& P,
                                            ParticleSet& ions,
                                            TrialWaveFunction& psi,
                                            ParticleSet::ParticlePos& hf_terms,
                                            ParticleSet::ParticlePos& pulay_terms)
{
  value_    = 0.0;
  forces_   = 0;
  PulayTerm = 0;

  const auto& myTable = P.getDistTableAB(myTableIndex);
  for (int ig = 0; ig < P.groups(); ++ig) //loop over species
  {
    psi.prepareGroup(P, ig);
    for (int jel = P.first(ig); jel < P.last(ig); ++jel)
    {
      const auto& dist  = myTable.getDistRow(jel);
      const auto& displ = myTable.getDisplRow(jel);
      for (int iat = 0; iat < PP.size(); iat++)
        if (PP[iat] != nullptr && dist[iat] < PP[iat]->getRmax())
          value_ +=
              PP[iat]->evaluateOneWithForces(P, vp_ ? makeOptionalRef<VirtualParticleSet>(*vp_) : std::nullopt, ions,
                                             iat, psi, jel, dist[iat], -displ[iat], forces_[iat], PulayTerm);
    }
  }

  hf_terms -= forces_;
  pulay_terms -= PulayTerm;
}

void NonLocalECPotential::computeOneElectronTxy(TrialWaveFunction& psi,
                                                ParticleSet& P,
                                                const int ref_elec,
                                                std::vector<NonLocalData>& tmove_xy)
{
  tmove_xy.clear();
  const auto& myTable = P.getDistTableAB(myTableIndex);
  const auto& dist    = myTable.getDistRow(ref_elec);
  const auto& displ   = myTable.getDisplRow(ref_elec);
  for (const int iat : neighbor_lists.getNeighboringIons(ref_elec))
    PP[iat]->evaluateOne(P, vp_ ? makeOptionalRef<VirtualParticleSet>(*vp_) : std::nullopt, iat, psi, ref_elec,
                         dist[iat], -displ[iat], tmove_xy, use_DLA);
}

void NonLocalECPotential::evaluateOneBodyOpMatrix(ParticleSet& P,
                                                  const TWFFastDerivWrapper& psi,
                                                  std::vector<ValueMatrix>& B)
{
  bool keepGrid = true;
  for (int ipp = 0; ipp < PPset.size(); ipp++)
    if (PPset[ipp])
      if (!keepGrid)
        PPset[ipp]->rotateQuadratureGrid(generateRandomRotationMatrix(*myRNG));

  const auto& myTable = P.getDistTableAB(myTableIndex);
  for (int ig = 0; ig < P.groups(); ++ig) //loop over species
  {
    for (int jel = P.first(ig); jel < P.last(ig); ++jel)
    {
      const auto& dist  = myTable.getDistRow(jel);
      const auto& displ = myTable.getDisplRow(jel);
      for (int iat = 0; iat < PP.size(); iat++)
        if (PP[iat] != nullptr && dist[iat] < PP[iat]->getRmax())
          PP[iat]->evaluateOneBodyOpMatrixContribution(P, iat, psi, jel, dist[iat], -displ[iat], B);
    }
  }
}

void NonLocalECPotential::evaluateOneBodyOpMatrixForceDeriv(ParticleSet& P,
                                                            ParticleSet& source,
                                                            const TWFFastDerivWrapper& psi,
                                                            const int iat_source,
                                                            std::vector<std::vector<ValueMatrix>>& Bforce)
{
  bool keepGrid = true;
  for (int ipp = 0; ipp < PPset.size(); ipp++)
    if (PPset[ipp])
      if (!keepGrid)
        PPset[ipp]->rotateQuadratureGrid(generateRandomRotationMatrix(*myRNG));

  const auto& myTable = P.getDistTableAB(myTableIndex);
  for (int ig = 0; ig < P.groups(); ++ig) //loop over species
  {
    for (int jel = P.first(ig); jel < P.last(ig); ++jel)
    {
      const auto& dist  = myTable.getDistRow(jel);
      const auto& displ = myTable.getDisplRow(jel);
      for (int iat = 0; iat < PP.size(); iat++)
        if (PP[iat] != nullptr && dist[iat] < PP[iat]->getRmax())
          PP[iat]->evaluateOneBodyOpMatrixdRContribution(P, source, iat, iat_source, psi, jel, dist[iat], -displ[iat],
                                                         Bforce);
    }
  }
}

int NonLocalECPotential::makeNonLocalMovesPbyP(TrialWaveFunction& psi, ParticleSet& P, NonLocalTOperator& move_op)
{
  auto& RandomGen(*myRNG);
  auto& nonLocalOps = move_op;

  int NonLocalMoveAccepted = 0;
  if (move_op.getMoveKind() == TmoveKind::V0)
  {
    const NonLocalData* oneTMove = nonLocalOps.selectMove(RandomGen(), tmove_xy_all_);
    //make a non-local move
    if (oneTMove)
    {
      const int iat = oneTMove->PID;
      psi.prepareGroup(P, P.getGroupID(iat));
      GradType grad_iat;
      if (P.makeMoveAndCheck(iat, oneTMove->Delta) && psi.calcRatioGrad(P, iat, grad_iat) != ValueType(0))
      {
        psi.acceptMove(P, iat, true);
        P.acceptMove(iat);
        NonLocalMoveAccepted++;
      }
    }
  }
  else if (move_op.getMoveKind() == TmoveKind::V1)
  {
    GradType grad_iat;
    std::vector<NonLocalData> tmove_xy;
    //make a non-local move per particle
    for (int ig = 0; ig < P.groups(); ++ig) //loop over species
    {
      psi.prepareGroup(P, ig);
      for (int iat = P.first(ig); iat < P.last(ig); ++iat)
      {
        computeOneElectronTxy(psi, P, iat, tmove_xy);
        const NonLocalData* oneTMove = nonLocalOps.selectMove(RandomGen(), tmove_xy);
        if (oneTMove)
        {
          if (P.makeMoveAndCheck(iat, oneTMove->Delta) && psi.calcRatioGrad(P, iat, grad_iat) != ValueType(0))
          {
            psi.acceptMove(P, iat, true);
            P.acceptMove(iat);
            NonLocalMoveAccepted++;
          }
        }
      }
    }
  }
  else if (move_op.getMoveKind() == TmoveKind::V3)
  {
    elecTMAffected.assign(P.getTotalNum(), false);
    nonLocalOps.groupByElectron(P.getTotalNum(), tmove_xy_all_);
    GradType grad_iat;
    std::vector<NonLocalData> tmove_xy;
    //make a non-local move per particle
    for (int ig = 0; ig < P.groups(); ++ig) //loop over species
    {
      psi.prepareGroup(P, ig);
      for (int iat = P.first(ig); iat < P.last(ig); ++iat)
      {
        const NonLocalData* oneTMove;
        if (elecTMAffected[iat])
        {
          // recompute Txy for the given electron effected by T-moves
          computeOneElectronTxy(psi, P, iat, tmove_xy);
          oneTMove = nonLocalOps.selectMove(RandomGen(), tmove_xy);
        }
        else
          oneTMove = nonLocalOps.selectMove(RandomGen(), iat);
        if (oneTMove)
        {
          if (P.makeMoveAndCheck(iat, oneTMove->Delta) && psi.calcRatioGrad(P, iat, grad_iat) != ValueType(0))
          {
            psi.acceptMove(P, iat, true);
            // mark all affected electrons
            neighbor_lists.markAffectedElecs(P.getDistTableAB(myTableIndex), iat, elecTMAffected);
            P.acceptMove(iat);
            NonLocalMoveAccepted++;
          }
        }
      }
    }
  }

  if (NonLocalMoveAccepted > 0)
  {
    psi.completeUpdates();
    // this step also updates electron positions on the device.
    P.donePbyP(true);
  }

  return NonLocalMoveAccepted;
}

std::vector<int> NonLocalECPotential::mw_makeNonLocalMovesPbyP(const RefVectorWithLeader<OperatorBase>& o_list,
                                                               const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                                               const RefVectorWithLeader<ParticleSet>& p_list,
                                                               NonLocalTOperator& move_op)
{
  const size_t nw = o_list.size();
  std::vector<int> num_accepted(nw, 0);

  if (move_op.getMoveKind() != TmoveKind::V1)
  {
    for (size_t iw = 0; iw < nw; iw++)
      num_accepted[iw] =
          o_list.getCastedElement<NonLocalECPotential>(iw).makeNonLocalMovesPbyP(wf_list[iw], p_list[iw], move_op);
    return num_accepted;
  }

  auto& O_leader           = o_list.getCastedLeader<NonLocalECPotential>();
  ParticleSet& pset_leader = p_list.getLeader();

  // per-walker candidate lists and single-electron job lists, rebuilt per electron
  std::vector<std::vector<NonLocalData>> tmove_xy(nw);
  std::vector<std::vector<NLPPJob<Real>>> jel_jobs(nw);

  auto pp_component = std::find_if(O_leader.PPset.begin(), O_leader.PPset.end(), [](auto& ptr) { return bool(ptr); });
  assert(pp_component != std::end(O_leader.PPset));

  // generate random numbers in the order exactly the same as serialization code path.
  // Note that: O.myRNG of the same batch are exactly identical and thus the order matters.
  // ad-hoc allocating rng_vals memory is sub-optimal and needs to be taken care.
  Matrix<RealType> rng_vals(nw, pset_leader.getTotalNum());
  for (int iw = 0; iw < rng_vals.rows(); iw++)
  {
    auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
    for (int jel = 0; jel < rng_vals.cols(); jel++)
      rng_vals[iw][jel] = (*O.myRNG)();
  }

  RefVectorWithLeader<NonLocalECPComponent> ecp_component_list(**pp_component);
  RefVectorWithLeader<ParticleSet> pset_list(pset_leader);
  RefVectorWithLeader<TrialWaveFunction> psi_list(wf_list.getLeader());
  RefVector<const NLPPJob<Real>> batch_list;
  RefVector<std::vector<NonLocalData>> tmove_xy_batch_list;
  RefVector<VirtualParticleSet> vp_list;
  std::vector<Real> pairpots(nw);

  ecp_component_list.reserve(nw);
  pset_list.reserve(nw);
  psi_list.reserve(nw);
  batch_list.reserve(nw);
  tmove_xy_batch_list.reserve(nw);

  for (int ig = 0; ig < pset_leader.groups(); ++ig) //loop over species
  {
    TrialWaveFunction::mw_prepareGroup(wf_list, p_list, ig);

    for (int jel = pset_leader.first(ig); jel < pset_leader.last(ig); ++jel)
    {
      // candidate ratio evaluations of this electron, batched across walkers.
      // Neighbor lists are the ones built at the preceding energy evaluation,
      // matching the single-walker sweep; distances are read fresh so accepts
      // of earlier electrons in this sweep are seen.
      size_t max_num_jobs = 0;
      for (size_t iw = 0; iw < nw; iw++)
      {
        auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
        const ParticleSet& P(p_list[iw]);
        auto& jobs = jel_jobs[iw];
        jobs.clear();
        tmove_xy[iw].clear();
        const auto& myTable = P.getDistTableAB(O.myTableIndex);
        const auto& dist    = myTable.getDistRow(jel);
        const auto& displ   = myTable.getDisplRow(jel);
        for (const int iat : O.neighbor_lists.getNeighboringIons(jel))
          jobs.emplace_back(iat, jel, dist[iat], -displ[iat]);
        max_num_jobs = std::max(max_num_jobs, jobs.size());
      }

      for (size_t jobid = 0; jobid < max_num_jobs; jobid++)
      {
        ecp_component_list.clear();
        pset_list.clear();
        psi_list.clear();
        batch_list.clear();
        tmove_xy_batch_list.clear();
        vp_list.clear();
        vp_list.reserve(nw);
        for (size_t iw = 0; iw < nw; iw++)
        {
          auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
          if (jobid < jel_jobs[iw].size())
          {
            const auto& job = jel_jobs[iw][jobid];
            ecp_component_list.push_back(*O.PP[job.ion_id]);
            pset_list.push_back(p_list[iw]);
            if (O.vp_)
              vp_list.push_back(*O.vp_);
            psi_list.push_back(wf_list[iw]);
            batch_list.push_back(job);
            tmove_xy_batch_list.push_back(tmove_xy[iw]);
          }
        }

        if (O_leader.vp_)
          NonLocalECPComponent::mw_evaluateOne(ecp_component_list, pset_list, {*O_leader.vp_, std::move(vp_list)},
                                               psi_list, batch_list, pairpots, tmove_xy_batch_list,
                                               O_leader.mw_res_handle_.getResource().collection, O_leader.use_DLA);
        else
          for (size_t j = 0; j < ecp_component_list.size(); j++)
            ecp_component_list[j].evaluateOne(pset_list[j], std::nullopt, batch_list[j].get().ion_id, psi_list[j],
                                              batch_list[j].get().electron_id, batch_list[j].get().ion_elec_dist,
                                              batch_list[j].get().ion_elec_displ,
                                              makeOptionalRef<std::vector<NonLocalData>>(tmove_xy_batch_list[j]),
                                              O_leader.use_DLA);
      }

      // selection and accepts per walker: identical calls, RNG draw order,
      // and accept bookkeeping as the single-walker v1 sweep
      for (size_t iw = 0; iw < nw; iw++)
      {
        auto& O                      = o_list.getCastedElement<NonLocalECPotential>(iw);
        const NonLocalData* oneTMove = move_op.selectMove(rng_vals[iw][jel], tmove_xy[iw]);
        if (oneTMove)
        {
          TrialWaveFunction& psi = wf_list[iw];
          ParticleSet& P         = p_list[iw];
          GradType grad_iat;
          if (P.makeMoveAndCheck(jel, oneTMove->Delta) && psi.calcRatioGrad(P, jel, grad_iat) != ValueType(0))
          {
            psi.acceptMove(P, jel, true);
            P.acceptMove(jel);
            num_accepted[iw]++;
          }
        }
      }
    }
  }

  for (size_t iw = 0; iw < nw; iw++)
    if (num_accepted[iw] > 0)
    {
      wf_list[iw].completeUpdates();
      // this step also updates electron positions on the device.
      p_list[iw].donePbyP(true);
    }

  return num_accepted;
}

void NonLocalECPotential::addComponent(int groupID, std::unique_ptr<NonLocalECPComponent>&& ppot)
{
  for (int iat = 0; iat < PP.size(); iat++)
    if (IonConfig.GroupID[iat] == groupID)
      PP[iat] = ppot.get();
  PPset[groupID] = std::move(ppot);
}

void NonLocalECPotential::createResource(ResourceCollection& collection) const
{
  auto new_res =
      std::make_unique<NonLocalECPotentialMultiWalkerResource>(mw_resource_identity_, multiWalkerResourceSchema());
  if (vp_)
    vp_->createResource(new_res->collection);
  collection.addResource(std::move(new_res));
}

void NonLocalECPotential::acquireResource(ResourceCollection& collection,
                                          const RefVectorWithLeader<OperatorBase>& o_list) const
{
  auto& O_leader = o_list.getCastedLeader<NonLocalECPotential>();
  if (this != std::addressof(O_leader))
    throw std::logic_error("NonLocalECPotential resource acquisition must be invoked on the crowd leader.");
  if (O_leader.mw_res_handle_)
    throw std::logic_error("NonLocalECPotential multi-walker resource is already acquired.");

  const MultiWalkerResourceSchema leader_schema = O_leader.multiWalkerResourceSchema();
  for (std::size_t iw = 0; iw < o_list.size(); ++iw)
  {
    const auto& potential = o_list.getCastedElement<NonLocalECPotential>(iw);
    if (potential.mw_resource_identity_.get() != O_leader.mw_resource_identity_.get())
      throw std::invalid_argument(
          "NonLocalECPotential crowd contains operators from different clone families.");
    if (potential.multiWalkerResourceSchema() != leader_schema)
      throw std::invalid_argument("NonLocalECPotential crowd contains incompatible resource schemas.");
  }

  const std::size_t entry_cursor = collection.getCursor();
  auto candidate_handle = collection.lendResource<NonLocalECPotentialMultiWalkerResource>();
  const NonLocalECPotentialMultiWalkerResource& candidate = candidate_handle.getResource();
  if (candidate.identity.get() != O_leader.mw_resource_identity_.get() || candidate.schema != leader_schema)
  {
    collection.rewind(entry_cursor);
    throw std::logic_error("NonLocalECPotential ResourceCollection belongs to an incompatible operator family.");
  }
  O_leader.mw_res_handle_ = std::move(candidate_handle);
}

void NonLocalECPotential::releaseResource(ResourceCollection& collection,
                                          const RefVectorWithLeader<OperatorBase>& o_list) const
{
  auto& O_leader = o_list.getCastedLeader<NonLocalECPotential>();
  if (this != std::addressof(O_leader) || !O_leader.mw_res_handle_)
    throw std::logic_error("NonLocalECPotential resource release has no acquired crowd-leader handle.");
  collection.takebackResource(O_leader.mw_res_handle_);
}

std::unique_ptr<OperatorBase> NonLocalECPotential::makeClone(ParticleSet& qp, TrialWaveFunction& psi) const
{ return std::make_unique<NonLocalECPotential>(*this, qp); }

} // namespace qmcplusplus
