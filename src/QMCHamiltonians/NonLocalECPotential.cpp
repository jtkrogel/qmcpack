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

#include <limits>
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
namespace
{
/** Return the caller-row extent required by possibly sparse global mappings. */
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
} // namespace

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
  /// Bounded real-to-ValueType conversion used by weighted derivative tiles.
  std::vector<ValueType> derivative_bare_weights;
  /// Whole-request B-by-P derivative transaction and its non-owning rows.
  std::vector<ValueType> staged_parameter_derivatives;
  std::vector<TrialWaveFunction::ParameterDerivativeView> parameter_derivative_views;
  /// Reused caller-owned component arithmetic scratch.
  std::vector<Real> radial_scratch;
  std::vector<Real> legendre_scratch;
  /// First-tile and current-tile optimistic model-version stamps.
  std::vector<TrialWaveFunction::EvaluationStamp> reference_stamps;
  std::vector<TrialWaveFunction::EvaluationStamp> tile_stamps;
  /// Independent TMDLA nonfermionic first-tile and current-tile stamps.
  std::vector<TrialWaveFunction::EvaluationStamp> nonfermionic_reference_stamps;
  std::vector<TrialWaveFunction::EvaluationStamp> nonfermionic_tile_stamps;
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

NonLocalECPotential::MultiWalkerDerivativeStatistics
NonLocalECPotential::multiWalkerDerivativeStatisticsForTesting() const
{
  const auto& resource    = mw_res_handle_.getResource();
  const auto batch_stats  = resource.virtual_batch->statistics();
  return {batch_stats.logical_jobs,
          batch_stats.logical_knots,
          batch_stats.tiles_packed,
          batch_stats.split_job_continuations,
          batch_stats.tail_tiles,
          batch_stats.max_tile_occupancy,
          resource.virtual_batch->storageFingerprint(),
          resource.staged_parameter_derivatives.size(),
          resource.staged_parameter_derivatives.capacity(),
          resource.derivative_bare_weights.size(),
          resource.derivative_bare_weights.capacity()};
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

  mw_evaluateWithParameterDerivativesFlattenedVP(o_list, wf_list, p_list, optvars, dhpsioverpsi);
}

void NonLocalECPotential::mw_consumeFlattenedVPDerivativePreparedGroup(
    NonLocalECPotentialMultiWalkerResource& resource,
    const RefVectorWithLeader<OperatorBase>& o_list,
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
    const OptVariables& optvars,
    int group,
    bool& have_reference_stamps,
    bool& tile_available)
{
  auto& leader = o_list.getCastedLeader<NonLocalECPotential>();

  while (tile_available)
  {
    const auto& segments = resource.virtual_batch->tileSegments();
    if (segments.empty())
      throw std::logic_error("NonLocalECPotential packed an empty derivative tile.");
    if (segments.front().groupId() > group)
      break;
    if (segments.front().groupId() != group)
      throw std::logic_error("NonLocalECPotential derivative tile traversal left canonical group order.");

    // Materialize only the current bounded tile. The descriptor retains the
    // original walker/job/knot identities needed by the reductions below.
    auto& absolute_positions = resource.virtual_batch->mutableTileAbsolutePositions();
    auto& deltas             = resource.virtual_batch->mutableTileDeltas();
    auto& bare_weights       = resource.virtual_batch->mutableTileBareWeights();
    for (const auto& segment : segments)
    {
      const std::size_t walker = static_cast<std::size_t>(segment.walkerId());
      auto& potential          = o_list.getCastedElement<NonLocalECPotential>(walker);
      const auto& group_jobs   = resource.staged_jobs[walker][group];
      if (segment.walkerJobOrdinal() >= group_jobs.size())
        throw std::logic_error("NonLocalECPotential derivative tile references an absent staged job.");
      const auto& job = group_jobs[segment.walkerJobOrdinal()];
      if (job.ion_id != segment.ionId() || job.electron_id != segment.electronId())
        throw std::logic_error("NonLocalECPotential derivative tile identity disagrees with its staged job.");

      potential.PP[job.ion_id]->buildQuadraturePointRange(
          job.ion_elec_dist, job.ion_elec_displ, p_list[walker].R[job.electron_id],
          segment.firstKnot(), segment.knotCount(), segment.tileOffset(), deltas,
          absolute_positions, bare_weights, resource.radial_scratch, resource.legendre_scratch);
    }

    resource.virtual_batch->finalizeTileInput();
    const VirtualParticleBatch descriptor = resource.virtual_batch->makeVirtualParticleBatch();
    descriptor.validateFor(p_list, static_cast<std::size_t>(leader.IonConfig.getTotalNum()));

    // ECP quadrature weights are real, while a complex wavefunction's score
    // contraction must retain the imaginary part of weight*ratio. Convert in
    // bounded scratch rather than reinterpreting the real tile storage.
    resource.derivative_bare_weights.resize(bare_weights.size());
    for (std::size_t knot = 0; knot < bare_weights.size(); ++knot)
      resource.derivative_bare_weights[knot] = ValueType(bare_weights[knot]);

    auto& ratios = resource.virtual_batch->mutableTileRatios();
    resource.tile_stamps.clear();
    TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted(
        wf_list, p_list, vp_scratch_list, descriptor, optvars,
        resource.derivative_bare_weights, ratios, resource.parameter_derivative_views,
        resource.tile_stamps, TrialWaveFunction::ComputeType::ALL);

    // The TrialWaveFunction call already proves value and score phases used
    // one version within this tile. Preserve that same version across every
    // outer tile before allowing any caller-visible publication.
    if (!have_reference_stamps)
    {
      resource.reference_stamps = resource.tile_stamps;
      have_reference_stamps     = true;
    }
    else if (resource.tile_stamps != resource.reference_stamps)
      throw std::runtime_error(
          "NonLocalECPotential observed different wavefunction parameter versions across derivative tiles.");

    // Energy and derivative paths consume the same complete product ratios.
    // Persistent per-job accumulators preserve knot order across split tiles.
    auto& transformed_weights = resource.virtual_batch->mutableTileTransformedWeights();
    for (const auto& segment : segments)
    {
      const std::size_t walker = static_cast<std::size_t>(segment.walkerId());
      Real& pair_potential = resource.virtual_batch->jobPairPotential(segment.globalJobId());
      NonLocalECPComponent::reduceQuadraturePointRange(
          segment.electronId(), segment.tileOffset(), segment.knotCount(),
          resource.virtual_batch->tileDeltas(), resource.virtual_batch->tileBareWeights(), ratios,
          nullptr, segment.tileOffset(), transformed_weights, segment.walkerKnotOffset(), nullptr,
          pair_potential);
      if (segment.endsJob())
        resource.staged_values[walker] += pair_potential;
    }

    tile_available = resource.virtual_batch->packNextTile();
  }
}

void NonLocalECPotential::mw_consumeFlattenedVPStreamingDerivativePreparedGroup(
    NonLocalECPotentialMultiWalkerResource& resource,
    const RefVectorWithLeader<OperatorBase>& o_list,
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
    wftrain::NonLocalECPDerivativeConsumer& consumer,
    std::uint64_t grid_fingerprint,
    std::size_t& tile_ordinal,
    int group,
    bool& have_reference_stamps,
    bool& tile_available)
{
  auto& leader = o_list.getCastedLeader<NonLocalECPotential>();

  while (tile_available)
  {
    const auto& segments = resource.virtual_batch->tileSegments();
    if (segments.empty())
      throw std::logic_error(
          "NonLocalECPotential packed an empty streaming-derivative tile.");
    if (segments.front().groupId() > group)
      break;
    if (segments.front().groupId() != group)
      throw std::logic_error(
          "NonLocalECPotential streaming-derivative traversal left canonical group order.");

    // The Hamiltonian remains the sole owner of quadrature construction.  Fill
    // the bounded tile exactly once, then use the same descriptor for the
    // complete ratio, energy reduction, and parameter-response contraction.
    auto& absolute_positions = resource.virtual_batch->mutableTileAbsolutePositions();
    auto& deltas             = resource.virtual_batch->mutableTileDeltas();
    auto& bare_weights       = resource.virtual_batch->mutableTileBareWeights();
    for (const auto& segment : segments)
    {
      const std::size_t walker = static_cast<std::size_t>(segment.walkerId());
      auto& potential          = o_list.getCastedElement<NonLocalECPotential>(walker);
      const auto& group_jobs   = resource.staged_jobs[walker][group];
      if (segment.walkerJobOrdinal() >= group_jobs.size())
        throw std::logic_error(
            "NonLocalECPotential streaming-derivative tile references an absent staged job.");
      const auto& job = group_jobs[segment.walkerJobOrdinal()];
      if (job.ion_id != segment.ionId() ||
          job.electron_id != segment.electronId())
        throw std::logic_error(
            "NonLocalECPotential streaming-derivative tile identity disagrees with its staged job.");

      potential.PP[job.ion_id]->buildQuadraturePointRange(
          job.ion_elec_dist, job.ion_elec_displ,
          p_list[walker].R[job.electron_id], segment.firstKnot(),
          segment.knotCount(), segment.tileOffset(), deltas,
          absolute_positions, bare_weights, resource.radial_scratch,
          resource.legendre_scratch);
    }

    resource.virtual_batch->finalizeTileInput();
    const VirtualParticleBatch descriptor =
        resource.virtual_batch->makeVirtualParticleBatch();
    descriptor.validateFor(
        p_list, static_cast<std::size_t>(leader.IonConfig.getTotalNum()));

    auto& ratios = resource.virtual_batch->mutableTileRatios();
    resource.tile_stamps.clear();
    TrialWaveFunction::mw_evaluateVirtualRatios(
        wf_list, p_list, vp_scratch_list, descriptor, ratios,
        resource.tile_stamps, TrialWaveFunction::ComputeType::ALL);

    // Every tile must come from one immutable wavefunction version.  This
    // cross-tile check complements the consumer's check for its own provider.
    if (!have_reference_stamps)
    {
      resource.reference_stamps = resource.tile_stamps;
      have_reference_stamps     = true;
    }
    else if (resource.tile_stamps != resource.reference_stamps)
      throw std::runtime_error(
          "NonLocalECPotential observed different wavefunction parameter versions across streaming-derivative tiles.");

    resource.derivative_bare_weights.resize(bare_weights.size());
    for (std::size_t point = 0; point < bare_weights.size(); ++point)
      resource.derivative_bare_weights[point] = ValueType(bare_weights[point]);

    // The descriptor is intentionally consumed before either the energy
    // reduction or packNextTile() may reuse its backing storage.  Its exact
    // coefficient is bare_weight * complete ALL-wavefunction ratio.
    consumer.consume({descriptor,
                      {resource.derivative_bare_weights.data(),
                       resource.derivative_bare_weights.size()},
                      {ratios.data(), ratios.size()},
                      {resource.tile_stamps.data(), resource.tile_stamps.size()},
                      tile_ordinal, grid_fingerprint});

    auto& transformed_weights =
        resource.virtual_batch->mutableTileTransformedWeights();
    for (const auto& segment : segments)
    {
      const std::size_t walker = static_cast<std::size_t>(segment.walkerId());
      Real& pair_potential =
          resource.virtual_batch->jobPairPotential(segment.globalJobId());
      NonLocalECPComponent::reduceQuadraturePointRange(
          segment.electronId(), segment.tileOffset(), segment.knotCount(),
          resource.virtual_batch->tileDeltas(),
          resource.virtual_batch->tileBareWeights(), ratios, nullptr,
          segment.tileOffset(), transformed_weights,
          segment.walkerKnotOffset(), nullptr, pair_potential);
      if (segment.endsJob())
        resource.staged_values[walker] += pair_potential;
    }

    ++tile_ordinal;
    tile_available = resource.virtual_batch->packNextTile();
  }
}

void NonLocalECPotential::mw_evaluateWithParameterDerivativesFlattenedVP(
    const RefVectorWithLeader<OperatorBase>& o_list,
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const OptVariables& optvars,
    RecordArray<ValueType>& dhpsioverpsi)
{
  auto& leader                    = o_list.getCastedLeader<NonLocalECPotential>();
  const ParticleSet& pset_leader = p_list.getLeader();
  const std::size_t walker_count = o_list.size();
  const std::size_t parameter_count = static_cast<std::size_t>(dhpsioverpsi.getNumOfParams());

  if (!leader.vp_ || !leader.mw_res_handle_)
    throw std::logic_error("NonLocalECPotential flattened derivatives require VP crowd resources.");
  if (parameter_count < requiredDerivativeExtent(optvars))
    throw std::invalid_argument("NonLocalECPotential derivative rows are too short for the optimizer mapping.");
  if (parameter_count != 0 && walker_count > std::numeric_limits<std::size_t>::max() / parameter_count)
    throw std::length_error("NonLocalECPotential derivative staging extent overflows.");
  const std::size_t derivative_extent = walker_count * parameter_count;
  if (derivative_extent > std::numeric_limits<std::size_t>::max() / sizeof(ValueType))
    throw std::length_error("NonLocalECPotential derivative staging byte extent overflows.");

  auto& resource = leader.mw_res_handle_.getResource();
  if (!resource.virtual_batch || resource.virtual_batch->tileCapacity() != leader.outer_tile_capacity_)
    throw std::logic_error("NonLocalECPotential derivative resource has an incompatible outer tile.");

  // Validate the complete crowd before consuming random rotations or changing
  // any public energy, job, or derivative destination.
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    const auto& potential       = o_list.getCastedElement<NonLocalECPotential>(walker);
    const ParticleSet& particles = p_list[walker];
    if (!potential.vp_ || potential.use_DLA != leader.use_DLA)
      throw std::invalid_argument("NonLocalECPotential derivative crowd has incompatible localization state.");
    if (particles.groups() != pset_leader.groups() ||
        particles.getTotalNum() != pset_leader.getTotalNum() ||
        potential.nlpp_jobs.size() != static_cast<std::size_t>(particles.groups()) ||
        potential.PP.size() != static_cast<std::size_t>(leader.IonConfig.getTotalNum()))
      throw std::invalid_argument("NonLocalECPotential derivative crowd has incompatible particle shapes.");
    if (potential.myRNG == nullptr)
      throw std::logic_error("NonLocalECPotential derivative grid rotation requires a random-number generator.");
  }

  // Prepare the request transaction and all fixed-size storage before grid
  // rotation. Later tile calls add only to these private derivative rows.
  resource.staged_jobs.resize(walker_count);
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    auto& walker_jobs = resource.staged_jobs[walker];
    walker_jobs.resize(static_cast<std::size_t>(p_list[walker].groups()));
    for (auto& group_jobs : walker_jobs)
      group_jobs.clear();
  }
  resource.staged_values.assign(walker_count, Real(0));
  resource.staged_parameter_derivatives.assign(derivative_extent, ValueType(0));
  resource.parameter_derivative_views.resize(walker_count);
  for (std::size_t walker = 0; walker < walker_count; ++walker)
    resource.parameter_derivative_views[walker] = {
        parameter_count == 0 ? nullptr
                             : resource.staged_parameter_derivatives.data() + walker * parameter_count,
        parameter_count};
  resource.derivative_bare_weights.clear();
  resource.derivative_bare_weights.reserve(leader.outer_tile_capacity_);
  resource.reference_stamps.clear();
  resource.tile_stamps.clear();
  resource.virtual_batch->reset(walker_count, false);

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

  // Preserve the scalar grid-rotation and electron/ion enumeration order,
  // but keep jobs private until the complete request succeeds.
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    auto& potential              = o_list.getCastedElement<NonLocalECPotential>(walker);
    const ParticleSet& particles = p_list[walker];
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
            group_jobs.emplace_back(ion, electron, distances[ion], -displacements[ion]);
      }
    }
  }

  // Flatten canonical group/walker/job order. Jobs larger than the outer
  // capacity are split only at knot boundaries by NLPPVirtualBatchStorage.
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
          throw std::logic_error("NonLocalECPotential encountered an empty derivative quadrature grid.");
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
    ResourceCollectionTeamLock<VirtualParticleSet> vp_resource_lock(resource.collection, vp_scratch_list);
    bool have_reference_stamps = false;
    bool tile_available        = resource.virtual_batch->packNextTile();
    for (int group = 0; group < pset_leader.groups(); ++group)
    {
      TrialWaveFunction::mw_prepareGroup(wf_list, p_list, group);
      mw_consumeFlattenedVPDerivativePreparedGroup(
          resource, o_list, wf_list, p_list, vp_scratch_list, optvars, group,
          have_reference_stamps, tile_available);
    }
    if (tile_available)
      throw std::logic_error("NonLocalECPotential derivative tile traversal exceeded the particle-group range.");
  }

  // Check all public destinations before the no-throw commit phase.
  resource.virtual_batch->validateLogicalOutputExtents();
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    const auto& potential = o_list.getCastedElement<NonLocalECPotential>(walker);
    if (potential.nlpp_jobs.size() != resource.staged_jobs[walker].size())
      throw std::logic_error("NonLocalECPotential staged derivative job extent changed before publication.");
  }

  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    auto& potential = o_list.getCastedElement<NonLocalECPotential>(walker);
    for (std::size_t group = 0; group < potential.nlpp_jobs.size(); ++group)
      potential.nlpp_jobs[group].swap(resource.staged_jobs[walker][group]);
    potential.value_ = resource.staged_values[walker];
    for (std::size_t parameter = 0; parameter < parameter_count; ++parameter)
      dhpsioverpsi[walker][parameter] +=
          resource.staged_parameter_derivatives[walker * parameter_count + parameter];
  }
}

void NonLocalECPotential::mw_evaluateWithStreamingParameterDerivatives(
    const RefVectorWithLeader<OperatorBase>& o_list,
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    wftrain::NonLocalECPDerivativeConsumer& consumer,
    wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> channels,
    wftrain::ParameterReductionSink& sink) const
{
  auto& leader = o_list.getCastedLeader<NonLocalECPotential>();
  if (this != &leader)
    throw std::invalid_argument(
        "NonLocalECPotential streaming derivatives must be invoked on the crowd leader.");

  const std::size_t walker_count = o_list.size();
  if (wf_list.size() != walker_count || p_list.size() != walker_count)
    throw std::invalid_argument(
        "NonLocalECPotential streaming-derivative crowd has inconsistent list sizes.");
  if (channels.empty())
    throw std::invalid_argument(
        "NonLocalECPotential streaming derivatives require a coefficient channel.");
  if (!leader.vp_ || !leader.mw_res_handle_)
    throw std::logic_error(
        "NonLocalECPotential streaming derivatives require VP crowd resources.");
  if (leader.use_DLA)
    throw std::invalid_argument(
        "NonLocalECPotential streaming derivatives do not support DLA or TMDLA localization.");

  const ParticleSet& pset_leader = p_list.getLeader();
  if (pset_leader.isSpinor())
    throw std::invalid_argument(
        "NonLocalECPotential streaming derivatives do not support spinor or spin-orbit ECP evaluation.");

  auto& resource = leader.mw_res_handle_.getResource();
  if (!resource.virtual_batch ||
      resource.virtual_batch->tileCapacity() != leader.outer_tile_capacity_)
    throw std::logic_error(
        "NonLocalECPotential streaming-derivative resource has an incompatible outer tile.");

  // Validate the complete crowd before consuming random rotations, beginning
  // the sink, or changing any public operator state.
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    const auto& potential =
        o_list.getCastedElement<NonLocalECPotential>(walker);
    const ParticleSet& particles = p_list[walker];
    if (!potential.vp_ || potential.use_DLA || particles.isSpinor())
      throw std::invalid_argument(
          "NonLocalECPotential streaming-derivative crowd contains an unsupported localization, non-VP, or spinor member.");
    if (particles.groups() != pset_leader.groups() ||
        particles.getTotalNum() != pset_leader.getTotalNum() ||
        potential.nlpp_jobs.size() !=
            static_cast<std::size_t>(particles.groups()) ||
        potential.PP.size() !=
            static_cast<std::size_t>(leader.IonConfig.getTotalNum()))
      throw std::invalid_argument(
          "NonLocalECPotential streaming-derivative crowd has incompatible particle shapes.");
    if (potential.myRNG == nullptr)
      throw std::logic_error(
          "NonLocalECPotential streaming-derivative grid rotation requires a random-number generator.");
  }

  resource.staged_jobs.resize(walker_count);
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    auto& walker_jobs = resource.staged_jobs[walker];
    walker_jobs.resize(static_cast<std::size_t>(p_list[walker].groups()));
    for (auto& group_jobs : walker_jobs)
      group_jobs.clear();
  }
  resource.staged_values.assign(walker_count, Real(0));
  resource.reference_stamps.clear();
  resource.tile_stamps.clear();
  resource.derivative_bare_weights.clear();

  // A resource may previously have served the compatibility B-by-P route.
  // Release that storage before entering the strict bounded transaction so
  // its live footprint contains no hidden walker-by-parameter allocation.
  std::vector<ValueType>().swap(resource.staged_parameter_derivatives);
  resource.parameter_derivative_views.clear();
  resource.virtual_batch->reset(walker_count, false);

  std::size_t maximum_channels = 0;
  std::size_t maximum_legendre = 0;
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    const auto& potential =
        o_list.getCastedElement<NonLocalECPotential>(walker);
    for (const auto& component : potential.PPset)
      if (component)
      {
        maximum_channels =
            std::max(maximum_channels,
                     static_cast<std::size_t>(component->getNchannel()));
        maximum_legendre =
            std::max(maximum_legendre,
                     static_cast<std::size_t>(component->getLmax() + 1));
      }
  }
  resource.radial_scratch.resize(maximum_channels);
  resource.legendre_scratch.resize(maximum_legendre);

  // Preserve the production grid-rotation and electron/ion enumeration order.
  // Jobs remain private until both the Hamiltonian and sink transactions have
  // completed successfully.
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    auto& potential = o_list.getCastedElement<NonLocalECPotential>(walker);
    const ParticleSet& particles = p_list[walker];
    for (const auto& component : potential.PPset)
      if (component)
        component->rotateQuadratureGrid(
            generateRandomRotationMatrix(*potential.myRNG));

    const auto& distance_table =
        particles.getDistTableAB(potential.myTableIndex);
    for (int group = 0; group < particles.groups(); ++group)
    {
      auto& group_jobs = resource.staged_jobs[walker][group];
      for (int electron = particles.first(group);
           electron < particles.last(group); ++electron)
      {
        const auto& distances = distance_table.getDistRow(electron);
        const auto& displacements = distance_table.getDisplRow(electron);
        for (int ion = 0; ion < potential.PP.size(); ++ion)
          if (potential.PP[ion] &&
              distances[ion] < potential.PP[ion]->getRmax())
            group_jobs.emplace_back(ion, electron, distances[ion],
                                    -displacements[ion]);
      }
    }
  }

  for (int group = 0; group < pset_leader.groups(); ++group)
    for (std::size_t walker = 0; walker < walker_count; ++walker)
    {
      const auto& potential =
          o_list.getCastedElement<NonLocalECPotential>(walker);
      const auto& group_jobs = resource.staged_jobs[walker][group];
      for (std::size_t ordinal = 0; ordinal < group_jobs.size(); ++ordinal)
      {
        const auto& job = group_jobs[ordinal];
        const int knot_count = potential.PP[job.ion_id]->getNknot();
        if (knot_count <= 0)
          throw std::logic_error(
              "NonLocalECPotential encountered an empty streaming-derivative quadrature grid.");
        resource.virtual_batch->appendJob(
            {group, static_cast<int>(walker), job.ion_id, job.electron_id,
             ordinal, static_cast<std::size_t>(knot_count)});
      }
    }
  resource.virtual_batch->seal();

  std::uint64_t grid_fingerprint = resource.virtual_batch->logicalFingerprint();
  if (grid_fingerprint == 0)
    grid_fingerprint = 1;
  const auto& coefficient_identity = channels[0].coefficients;
  wftrain::NonLocalECPDerivativeContext context;
  context.provider_id          = coefficient_identity.provider_id;
  context.schema_fingerprint   = coefficient_identity.schema_fingerprint;
  context.parameter_version    = coefficient_identity.parameter_version;
  context.batch_ordinal        = coefficient_identity.batch_ordinal;
  context.sample_offset        = coefficient_identity.sample_offset;
  context.sample_count         = walker_count;
  context.expected_point_count = resource.virtual_batch->totalKnotCount();
  context.grid_fingerprint     = grid_fingerprint;
  context.localization                   = wftrain::NonLocalECPLocalization::ORDINARY_LOCALITY;
  context.uses_virtual_particles         = true;
  context.scalar_relativistic            = true;
  context.used_dense_derivative_fallback = false;

  consumer.begin(context, channels, sink);
  try
  {
    RefVectorWithLeader<VirtualParticleSet> vp_scratch_list(*leader.vp_);
    vp_scratch_list.reserve(walker_count);
    for (std::size_t walker = 0; walker < walker_count; ++walker)
      vp_scratch_list.push_back(
          *o_list.getCastedElement<NonLocalECPotential>(walker).vp_);

    {
      ResourceCollectionTeamLock<VirtualParticleSet> vp_resource_lock(
          resource.collection, vp_scratch_list);
      bool have_reference_stamps = false;
      bool tile_available = resource.virtual_batch->packNextTile();
      std::size_t tile_ordinal = 0;
      for (int group = 0; group < pset_leader.groups(); ++group)
      {
        TrialWaveFunction::mw_prepareGroup(wf_list, p_list, group);
        mw_consumeFlattenedVPStreamingDerivativePreparedGroup(
            resource, o_list, wf_list, p_list, vp_scratch_list, consumer,
            grid_fingerprint, tile_ordinal, group, have_reference_stamps,
            tile_available);
      }
      if (tile_available)
        throw std::logic_error(
            "NonLocalECPotential streaming-derivative traversal exceeded the particle-group range.");
    }

    // Validate every Hamiltonian destination before the sink publishes its
    // completed response.  After consumer.end(), only noexcept swaps and
    // scalar assignments remain, giving one failure-atomic outer transaction.
    resource.virtual_batch->validateLogicalOutputExtents();
    for (std::size_t walker = 0; walker < walker_count; ++walker)
    {
      const auto& potential =
          o_list.getCastedElement<NonLocalECPotential>(walker);
      if (potential.nlpp_jobs.size() != resource.staged_jobs[walker].size())
        throw std::logic_error(
            "NonLocalECPotential staged streaming-derivative job extent changed before publication.");
    }

    consumer.end();
  }
  catch (...)
  {
    consumer.abort();
    throw;
  }

  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    auto& potential = o_list.getCastedElement<NonLocalECPotential>(walker);
    for (std::size_t group = 0; group < potential.nlpp_jobs.size(); ++group)
      potential.nlpp_jobs[group].swap(resource.staged_jobs[walker][group]);
    potential.value_ = resource.staged_values[walker];
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

void NonLocalECPotential::mw_consumeFlattenedVPPreparedGroup(
    NonLocalECPotentialMultiWalkerResource& resource,
    const RefVectorWithLeader<OperatorBase>& o_list,
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
    int group,
    bool compute_txy_all,
    bool accumulate_pair_results,
    const std::optional<ListenerOption<Real>>& listeners,
    bool& have_reference_stamps,
    bool& have_nonfermionic_reference_stamps,
    bool& tile_available)
{
  auto& leader = o_list.getCastedLeader<NonLocalECPotential>();

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

    auto& ratios         = resource.virtual_batch->mutableTileRatios();
    const bool use_tmdla = leader.use_DLA && compute_txy_all;
    auto evaluate_and_validate_stamps =
        [&](std::vector<ValueType>& selected_ratios,
            TrialWaveFunction::ComputeType compute_type,
            std::vector<TrialWaveFunction::EvaluationStamp>& tile_stamps,
            std::vector<TrialWaveFunction::EvaluationStamp>& reference_stamps,
            bool& have_reference,
            const char* stream_name) {
          tile_stamps.clear();
          TrialWaveFunction::mw_evaluateVirtualRatios(
              wf_list, p_list, vp_scratch_list, descriptor, selected_ratios, tile_stamps, compute_type);

          if (!have_reference)
          {
            reference_stamps = tile_stamps;
            have_reference   = true;
          }
          else if (tile_stamps != reference_stamps)
            throw std::runtime_error(
                std::string("NonLocalECPotential observed different wavefunction parameter versions across ") +
                stream_name + " outer tiles.");
        };

    std::vector<ValueType>* fermionic_ratios = nullptr;
    if (use_tmdla)
    {
      auto& selected_fermionic_ratios = resource.virtual_batch->mutableTileFermionicRatios();
      evaluate_and_validate_stamps(
          selected_fermionic_ratios, TrialWaveFunction::ComputeType::FERMIONIC,
          resource.tile_stamps, resource.reference_stamps, have_reference_stamps, "fermionic");

      auto& nonfermionic_ratios = resource.virtual_batch->mutableTileNonfermionicRatios();
      evaluate_and_validate_stamps(
          nonfermionic_ratios, TrialWaveFunction::ComputeType::NONFERMIONIC,
          resource.nonfermionic_tile_stamps, resource.nonfermionic_reference_stamps,
          have_nonfermionic_reference_stamps, "nonfermionic");

      if (ratios.size() != selected_fermionic_ratios.size() || ratios.size() != nonfermionic_ratios.size())
        throw std::logic_error("NonLocalECPotential TMDLA ratio buffers have inconsistent sizes.");
      for (std::size_t ratio = 0; ratio < ratios.size(); ++ratio)
      {
        ratios[ratio] = nonfermionic_ratios[ratio];
        ratios[ratio] *= selected_fermionic_ratios[ratio];
      }
      fermionic_ratios = &selected_fermionic_ratios;
    }
    else
      evaluate_and_validate_stamps(
          ratios,
          leader.use_DLA ? TrialWaveFunction::ComputeType::FERMIONIC
                         : TrialWaveFunction::ComputeType::ALL,
          resource.tile_stamps, resource.reference_stamps, have_reference_stamps, "selected");

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
          fermionic_ratios, segment.tileOffset(), transformed_weights, segment.walkerKnotOffset(), candidates,
          pair_potential);

      if (accumulate_pair_results && segment.endsJob())
      {
        resource.staged_values[walker] += pair_potential;
        if (listeners)
        {
          resource.ve_samples(walker, segment.electronId()) += Real(0.5) * pair_potential;
          resource.vi_samples(walker, segment.ionId()) += Real(0.5) * pair_potential;
        }
      }
    }

    tile_available = resource.virtual_batch->packNextTile();
  }
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
  if (!leader.vp_)
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
  resource.nonfermionic_reference_stamps.clear();
  resource.nonfermionic_tile_stamps.clear();
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
    auto& potential              = o_list.getCastedElement<NonLocalECPotential>(walker);
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

    bool have_reference_stamps              = false;
    bool have_nonfermionic_reference_stamps = false;
    bool tile_available                     = resource.virtual_batch->packNextTile();
    for (int group = 0; group < pset_leader.groups(); ++group)
    {
      // Keep this boundary even for an empty group, matching the reference path.
      TrialWaveFunction::mw_prepareGroup(wf_list, p_list, group);
      mw_consumeFlattenedVPPreparedGroup(
          resource, o_list, wf_list, p_list, vp_scratch_list, group, compute_txy_all,
          true, listeners, have_reference_stamps, have_nonfermionic_reference_stamps,
          tile_available);
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

  // The non-VP path remains an unchanged compatibility reference.
  if (O_leader.vp_)
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

void NonLocalECPotential::mw_prepareV1FlattenedVPResource(
    const RefVectorWithLeader<OperatorBase>& o_list,
    const RefVectorWithLeader<ParticleSet>& p_list)
{
  auto& leader = o_list.getCastedLeader<NonLocalECPotential>();
  const std::size_t walker_count = o_list.size();
  if (!leader.vp_ || !leader.mw_res_handle_)
    throw std::logic_error("NonLocalECPotential VP V1 preparation requires an acquired crowd resource.");
  if (p_list.size() != walker_count)
    throw std::invalid_argument("NonLocalECPotential VP V1 particle list has an inconsistent size.");

  auto& resource = leader.mw_res_handle_.getResource();
  resource.staged_jobs.resize(walker_count);
  std::size_t maximum_channels = 0;
  std::size_t maximum_legendre = 0;
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    auto& potential = o_list.getCastedElement<NonLocalECPotential>(walker);
    if (!potential.vp_ || potential.use_DLA != leader.use_DLA ||
        p_list[walker].groups() != p_list.getLeader().groups() ||
        p_list[walker].getTotalNum() != p_list.getLeader().getTotalNum())
      throw std::invalid_argument("NonLocalECPotential VP V1 crowd has incompatible resource shapes.");

    resource.staged_jobs[walker].resize(static_cast<std::size_t>(p_list[walker].groups()));
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
}

NLPPVirtualBatchStorage& NonLocalECPotential::mw_evaluateV1ElectronCandidates(
    const RefVectorWithLeader<OperatorBase>& o_list,
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
    int group,
    int electron)
{
  auto& leader = o_list.getCastedLeader<NonLocalECPotential>();
  const std::size_t walker_count = o_list.size();
  if (!leader.vp_ || !leader.mw_res_handle_)
    throw std::logic_error("NonLocalECPotential VP V1 evaluation requires an acquired crowd resource.");
  if (wf_list.size() != walker_count || p_list.size() != walker_count ||
      vp_scratch_list.size() != walker_count)
    throw std::invalid_argument("NonLocalECPotential VP V1 crowd lists have inconsistent sizes.");
  if (group < 0 || group >= p_list.getLeader().groups() ||
      electron < p_list.getLeader().first(group) || electron >= p_list.getLeader().last(group))
    throw std::invalid_argument("NonLocalECPotential VP V1 electron does not belong to the prepared group.");

  auto& resource = leader.mw_res_handle_.getResource();
  if (!resource.virtual_batch || resource.virtual_batch->tileCapacity() != leader.outer_tile_capacity_)
    throw std::logic_error("NonLocalECPotential VP V1 resource has an incompatible outer tile.");
  if (resource.staged_jobs.size() != walker_count)
    throw std::logic_error("NonLocalECPotential VP V1 staged-job walker extent is not prepared.");

  resource.reference_stamps.clear();
  resource.tile_stamps.clear();
  resource.nonfermionic_reference_stamps.clear();
  resource.nonfermionic_tile_stamps.clear();
  resource.virtual_batch->reset(walker_count, true);

  // V1 deliberately reuses the preceding energy evaluation's neighbor list,
  // while distances are read after all earlier-electron accepts in this sweep.
  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    auto& potential          = o_list.getCastedElement<NonLocalECPotential>(walker);
    const ParticleSet& particles = p_list[walker];
    if (!potential.vp_ || potential.use_DLA != leader.use_DLA ||
        resource.staged_jobs[walker].size() != static_cast<std::size_t>(particles.groups()) ||
        group >= particles.groups() || electron < particles.first(group) || electron >= particles.last(group))
      throw std::invalid_argument("NonLocalECPotential VP V1 crowd has incompatible particle or localization state.");

    auto& jobs = resource.staged_jobs[walker][group];
    jobs.clear();
    const auto& distance_table = particles.getDistTableAB(potential.myTableIndex);
    const auto& distances      = distance_table.getDistRow(electron);
    const auto& displacements  = distance_table.getDisplRow(electron);
    for (const int ion : potential.neighbor_lists.getNeighboringIons(electron))
    {
      if (ion < 0 || static_cast<std::size_t>(ion) >= potential.PP.size() || !potential.PP[ion])
        throw std::logic_error("NonLocalECPotential VP V1 neighbor list references an absent component.");
      jobs.emplace_back(ion, electron, distances[ion], -displacements[ion]);
    }
  }

  for (std::size_t walker = 0; walker < walker_count; ++walker)
  {
    const auto& potential = o_list.getCastedElement<NonLocalECPotential>(walker);
    const auto& jobs      = resource.staged_jobs[walker][group];
    for (std::size_t ordinal = 0; ordinal < jobs.size(); ++ordinal)
    {
      const auto& job       = jobs[ordinal];
      const int knot_count = potential.PP[job.ion_id]->getNknot();
      if (knot_count <= 0)
        throw std::logic_error("NonLocalECPotential VP V1 encountered an empty quadrature grid.");
      resource.virtual_batch->appendJob(
          {group, static_cast<int>(walker), job.ion_id, electron, ordinal,
           static_cast<std::size_t>(knot_count)});
    }
  }
  resource.virtual_batch->seal();

  {
    // Candidate storage must be complete before selection, and the nested VP
    // lock must not span selectMove, calcRatioGrad, or an accepted update.
    ResourceCollectionTeamLock<VirtualParticleSet> vp_resource_lock(resource.collection, vp_scratch_list);
    bool have_reference_stamps              = false;
    bool have_nonfermionic_reference_stamps = false;
    bool tile_available                     = resource.virtual_batch->packNextTile();
    mw_consumeFlattenedVPPreparedGroup(
        resource, o_list, wf_list, p_list, vp_scratch_list, group, true, false,
        std::nullopt, have_reference_stamps, have_nonfermionic_reference_stamps,
        tile_available);
    if (tile_available)
      throw std::logic_error("NonLocalECPotential VP V1 tile traversal crossed an electron boundary.");
  }

  resource.virtual_batch->validateLogicalOutputExtents();
  return *resource.virtual_batch;
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

  if (O_leader.vp_)
  {
    if (!O_leader.mw_res_handle_)
      throw std::logic_error("NonLocalECPotential VP V1 evaluation requires an acquired crowd resource.");

    mw_prepareV1FlattenedVPResource(o_list, p_list);

    RefVectorWithLeader<VirtualParticleSet> vp_scratch_list(*O_leader.vp_);
    vp_scratch_list.reserve(nw);
    for (std::size_t walker = 0; walker < nw; ++walker)
      vp_scratch_list.push_back(*o_list.getCastedElement<NonLocalECPotential>(walker).vp_);

    for (int group = 0; group < pset_leader.groups(); ++group)
    {
      // This remains a group boundary, not a per-electron tile operation.
      TrialWaveFunction::mw_prepareGroup(wf_list, p_list, group);

      for (int electron = pset_leader.first(group); electron < pset_leader.last(group); ++electron)
      {
        NLPPVirtualBatchStorage& candidates = mw_evaluateV1ElectronCandidates(
            o_list, wf_list, p_list, vp_scratch_list, group, electron);

        // Selection and accepts remain walker-ordered and complete before the
        // next electron's candidates observe any updated configurations.
        for (std::size_t walker = 0; walker < nw; ++walker)
        {
          const NonLocalData* selected =
              move_op.selectMove(rng_vals[walker][electron], candidates.walkerCandidates(walker));
          if (selected)
          {
            TrialWaveFunction& psi = wf_list[walker];
            ParticleSet& particles = p_list[walker];
            GradType gradient;
            if (particles.makeMoveAndCheck(electron, selected->Delta) &&
                psi.calcRatioGrad(particles, electron, gradient) != ValueType(0))
            {
              psi.acceptMove(particles, electron, true);
              particles.acceptMove(electron);
              ++num_accepted[walker];
            }
          }
        }
      }
    }

    for (std::size_t walker = 0; walker < nw; ++walker)
      if (num_accepted[walker] > 0)
      {
        wf_list[walker].completeUpdates();
        // This step also updates electron positions on the device.
        p_list[walker].donePbyP(true);
      }
    return num_accepted;
  }

  // The non-VP compatibility path retains the established one-job-per-walker
  // wavefront and its per-electron candidate vectors.
  std::vector<std::vector<NonLocalData>> tmove_xy(nw);
  std::vector<std::vector<NLPPJob<Real>>> jel_jobs(nw);
  auto pp_component =
      std::find_if(O_leader.PPset.begin(), O_leader.PPset.end(), [](auto& component) { return bool(component); });
  assert(pp_component != std::end(O_leader.PPset));

  RefVectorWithLeader<NonLocalECPComponent> ecp_component_list(**pp_component);
  RefVectorWithLeader<ParticleSet> pset_list(pset_leader);
  RefVectorWithLeader<TrialWaveFunction> psi_list(wf_list.getLeader());
  RefVector<const NLPPJob<Real>> batch_list;
  RefVector<std::vector<NonLocalData>> tmove_xy_batch_list;
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
        for (size_t iw = 0; iw < nw; iw++)
        {
          auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
          if (jobid < jel_jobs[iw].size())
          {
            const auto& job = jel_jobs[iw][jobid];
            ecp_component_list.push_back(*O.PP[job.ion_id]);
            pset_list.push_back(p_list[iw]);
            psi_list.push_back(wf_list[iw]);
            batch_list.push_back(job);
            tmove_xy_batch_list.push_back(tmove_xy[iw]);
          }
        }

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
