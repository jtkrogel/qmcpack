//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "Configuration.h"
#include "Numerics/Quadrature.h"
#include "OhmmsData/Libxml2Doc.h"
#include "Particle/ParticleSet.h"
#include "Particle/VirtualParticleSet.h"
#include "QMCHamiltonians/ECPComponentBuilder.h"
#include "QMCHamiltonians/NonLocalECPotential.h"
#include "QMCHamiltonians/NonLocalECPComponent.h"
#include "QMCHamiltonians/NonLocalTOperator.h"
#include "QMCHamiltonians/NLPPJob.h"
#include "QMCHamiltonians/NLPPVirtualBatch.h"
#include "QMCWaveFunctions/ConstantOrbital.h"
#include "QMCWaveFunctions/Jastrow/RadialJastrowBuilder.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeterminant.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/tests/psiformer_test_utils.h"
#include "ResourceCollection.h"
#include "TestListenerFunction.h"
#include "Utilities/FakeRandom.h"
#include "Utilities/StdRandom.h"
#include "Utilities/StlPrettyPrint.hpp"
#include "Utilities/RuntimeOptions.h"

namespace qmcplusplus
{
namespace testing
{
/** class to violate access control because evaluation of NonLocalECPotential uses RNG
 *  which we may not be able to control.
 */
class TestNonLocalECPotential
{
  using Real = QMCTraits::RealType;

public:
  static void copyGridUnrotatedForTest(NonLocalECPotential& nl_ecp)
  {
    nl_ecp.PPset[0]->rrotsgrid_m = nl_ecp.PPset[0]->sgridxyz_m;
  }

  static bool didGridChange(NonLocalECPotential& nl_ecp)
  {
    return nl_ecp.PPset[0]->rrotsgrid_m != nl_ecp.PPset[0]->sgridxyz_m;
  }

  static void copyComponentGridUnrotatedForTest(NonLocalECPComponent& component)
  {
    component.rrotsgrid_m = component.sgridxyz_m;
  }

  struct LegacyQuadratureData
  {
    std::vector<QMCTraits::PosType> deltas;
    std::vector<Real> bare_weights;
  };

  /** Build quadrature data through the pre-batching component scratch path. */
  static LegacyQuadratureData buildLegacyQuadrature(NonLocalECPComponent& component,
                                                     Real radius,
                                                     const QMCTraits::PosType& displacement)
  {
    LegacyQuadratureData result;
    result.deltas.resize(component.nknot);
    result.bare_weights.resize(component.nknot);
    component.buildQuadraturePointDeltaPosAndPartialPotential(
        radius, displacement, result.deltas, result.bare_weights);
    return result;
  }

  struct LegacyReduction
  {
    Real pair_potential;
    std::vector<Real> transformed_weights;
    std::vector<NonLocalData> candidates;
  };

  /** Reduce synthetic ratios through the pre-batching mutable component path. */
  static LegacyReduction reduceLegacyQuadrature(NonLocalECPComponent& component,
                                                 int electron_id,
                                                 const LegacyQuadratureData& quadrature,
                                                 const std::vector<QMCTraits::ValueType>& ratios,
                                                 const std::vector<QMCTraits::ValueType>& fermionic_ratios,
                                                 bool use_tmdla)
  {
    component.deltaV_      = quadrature.deltas;
    component.knot_pots_   = quadrature.bare_weights;
    component.psiratio     = ratios;
    component.psiratio_det = fermionic_ratios;

    LegacyReduction result;
    result.pair_potential = component.calculatePotential(component.knot_pots_, use_tmdla);
    result.transformed_weights = component.knot_pots_;
    component.contributeTxy(electron_id, result.candidates);
    return result;
  }

  struct LegacyScratchSnapshot
  {
    std::vector<QMCTraits::PosType> deltas;
    std::vector<Real> legendre;
    std::vector<Real> radial;
    std::vector<Real> knot_weights;
    std::vector<QMCTraits::ValueType> ratios;
    std::vector<QMCTraits::ValueType> fermionic_ratios;
  };

  /** Capture the legacy member arrays the caller-owned seam must not touch. */
  static LegacyScratchSnapshot snapshotLegacyScratch(const NonLocalECPComponent& component)
  {
    return {component.deltaV_, component.lpol, component.vrad, component.knot_pots_, component.psiratio,
            component.psiratio_det};
  }

  static bool legacyScratchMatches(const NonLocalECPComponent& component,
                                   const LegacyScratchSnapshot& snapshot)
  {
    return component.deltaV_ == snapshot.deltas && component.lpol == snapshot.legendre &&
        component.vrad == snapshot.radial && component.knot_pots_ == snapshot.knot_weights &&
        component.psiratio == snapshot.ratios && component.psiratio_det == snapshot.fermionic_ratios;
  }

  static void evaluateImpl(NonLocalECPotential& nl_ecp,
                           TrialWaveFunction& psi,
                           ParticleSet& P,
                           bool compute_txy_all,
                           bool keep_grid)
  {
    nl_ecp.evaluateImpl(psi, P, compute_txy_all, keep_grid);
  }

  static void mw_evaluateImpl(NonLocalECPotential& nl_ecp,
                              const RefVectorWithLeader<OperatorBase>& o_list,
                              const RefVectorWithLeader<TrialWaveFunction>& twf_list,
                              const RefVectorWithLeader<ParticleSet>& p_list,
                              bool compute_txy_all,
                              const std::optional<ListenerOption<Real>> listener_opt,
                              bool keep_grid)
  {
    nl_ecp.mw_evaluateImpl(o_list, twf_list, p_list, compute_txy_all, listener_opt, keep_grid);
  }

  static size_t numNeighboringIons(NonLocalECPotential& nl_ecp, int jel)
  {
    return nl_ecp.neighbor_lists.getNeighboringIons(jel).size();
  }

  static const std::vector<int>& neighboringIons(const NonLocalECPotential& nl_ecp, int jel)
  {
    return nl_ecp.neighbor_lists.getNeighboringIons(jel);
  }

  static const std::vector<int>& neighboringElectrons(const NonLocalECPotential& nl_ecp, int iat)
  {
    return nl_ecp.neighbor_lists.getNeighboringElectrons(iat);
  }

  static bool neighborListsBindOwnComponents(const NonLocalECPotential& nl_ecp)
  {
    return nl_ecp.neighbor_lists.isBoundTo(nl_ecp.PP);
  }

  static bool neighborListsBindComponents(const NonLocalECPotential& owner,
                                          const NonLocalECPotential& component_owner)
  {
    return owner.neighbor_lists.isBoundTo(component_owner.PP);
  }

  static NeighborListsForPseudo::OwnedLists makeNeighborListStaging(const NonLocalECPotential& nl_ecp)
  {
    return nl_ecp.neighbor_lists.makeOwnedLists();
  }

  static void validateNeighborListStaging(const NonLocalECPotential& nl_ecp,
                                          const NeighborListsForPseudo::OwnedLists& staging)
  {
    nl_ecp.neighbor_lists.validateOwnedLists(staging);
  }

  static bool publishNeighborListStaging(NonLocalECPotential& nl_ecp,
                                         NeighborListsForPseudo::OwnedLists& staging) noexcept
  {
    return nl_ecp.neighbor_lists.swapOwnedLists(staging);
  }

  static void addNeighborPair(NonLocalECPotential& nl_ecp, int electron, int ion)
  {
    nl_ecp.neighbor_lists.addElecIonPair(electron, ion);
  }

  static bool hasMultiWalkerResource(const NonLocalECPotential& nl_ecp) { return bool(nl_ecp.mw_res_handle_); }

  static void resizeListenerScratch(NonLocalECPotential& nl_ecp,
                                    std::size_t walkers,
                                    std::size_t electrons,
                                    std::size_t ions)
  { nl_ecp.resizeMultiWalkerListenerScratchForTesting(walkers, electrons, ions); }

  static std::pair<std::size_t, std::size_t> listenerScratchSizes(const NonLocalECPotential& nl_ecp)
  { return nl_ecp.multiWalkerListenerScratchSizesForTesting(); }

  static auto derivativeStatistics(const NonLocalECPotential& nl_ecp)
  { return nl_ecp.multiWalkerDerivativeStatisticsForTesting(); }

  static void setOuterTileCapacity(NonLocalECPotential& nl_ecp, std::size_t capacity)
  { nl_ecp.setOuterTileCapacityForTesting(capacity); }

  /** Report compatibility derivative-matrix storage without exposing it in production. */
  static size_t derivativeMatrixElements(const NonLocalECPotential& nl_ecp)
  {
    const auto component = std::find_if(
        nl_ecp.PPset.begin(), nl_ecp.PPset.end(), [](const auto& candidate) { return bool(candidate); });
    return component == nl_ecp.PPset.end() ? 0 : (*component)->dratio.size();
  }

  /** Report the candidates generated by locality/T-move evaluation modes. */
  static size_t tmoveCandidateCount(const NonLocalECPotential& nl_ecp)
  {
    return nl_ecp.tmove_xy_all_.size();
  }

  static const std::vector<NonLocalData>& tmoveCandidates(const NonLocalECPotential& nl_ecp)
  { return nl_ecp.tmove_xy_all_; }

  static const std::vector<std::vector<NLPPJob<Real>>>& jobs(const NonLocalECPotential& nl_ecp)
  { return nl_ecp.nlpp_jobs; }

  static int firstComponentKnotCount(const NonLocalECPotential& nl_ecp)
  {
    const auto component = std::find_if(
        nl_ecp.PPset.begin(), nl_ecp.PPset.end(), [](const auto& candidate) { return bool(candidate); });
    return component == nl_ecp.PPset.end() ? 0 : (*component)->getNknot();
  }

  /** Report the quadrature size selected for one concrete ion. */
  static int componentKnotCountForIon(const NonLocalECPotential& nl_ecp, int ion)
  {
    if (ion < 0 || static_cast<std::size_t>(ion) >= nl_ecp.PP.size() || !nl_ecp.PP[ion])
      throw std::out_of_range("NonLocalECPotential test ion has no component");
    return nl_ecp.PP[ion]->getNknot();
  }

  /** Evaluate one prepared V1 electron through the production flattened VP seam. */
  static std::vector<std::vector<NonLocalData>> evaluateFlattenedV1Candidates(
      NonLocalECPotential& leader,
      const RefVectorWithLeader<OperatorBase>& potentials,
      const RefVectorWithLeader<TrialWaveFunction>& wavefunctions,
      const RefVectorWithLeader<ParticleSet>& particles,
      int group,
      int electron)
  {
    NonLocalECPotential::mw_prepareV1FlattenedVPResource(potentials, particles);
    TrialWaveFunction::mw_prepareGroup(wavefunctions, particles, group);

    RefVectorWithLeader<VirtualParticleSet> vp_scratch_list(*leader.vp_);
    vp_scratch_list.reserve(potentials.size());
    for (std::size_t walker = 0; walker < potentials.size(); ++walker)
      vp_scratch_list.push_back(
          *potentials.getCastedElement<NonLocalECPotential>(walker).vp_);

    const NLPPVirtualBatchStorage& storage =
        NonLocalECPotential::mw_evaluateV1ElectronCandidates(
            potentials, wavefunctions, particles, vp_scratch_list, group, electron);
    std::vector<std::vector<NonLocalData>> result(potentials.size());
    for (std::size_t walker = 0; walker < potentials.size(); ++walker)
      result[walker] = storage.walkerCandidates(walker);
    return result;
  }

  /** Evaluate one V1 electron through the established scalar component path. */
  static std::vector<NonLocalData> evaluateScalarV1Candidates(NonLocalECPotential& potential,
                                                               TrialWaveFunction& wavefunction,
                                                               ParticleSet& particles,
                                                               int electron)
  {
    std::vector<NonLocalData> result;
    potential.computeOneElectronTxy(wavefunction, particles, electron, result);
    return result;
  }

  struct ListenerRows
  {
    std::vector<Real> electron;
    std::vector<Real> ion;
  };

  /** Compute the per-particle listener rows through the established scalar pair path. */
  static ListenerRows evaluateScalarListenerRows(NonLocalECPotential& nl_ecp,
                                                  TrialWaveFunction& psi,
                                                  ParticleSet& particles,
                                                  bool compute_tmove_data = false)
  {
    ListenerRows rows{std::vector<Real>(particles.getTotalNum(), 0),
                      std::vector<Real>(nl_ecp.IonConfig.getTotalNum(), 0)};
    std::vector<NonLocalData> candidates;
    const auto& distance_table = particles.getDistTableAB(nl_ecp.myTableIndex);
    for (int group = 0; group < particles.groups(); ++group)
    {
      psi.prepareGroup(particles, group);
      for (int electron = particles.first(group); electron < particles.last(group); ++electron)
      {
        const auto& distances     = distance_table.getDistRow(electron);
        const auto& displacements = distance_table.getDisplRow(electron);
        for (int ion = 0; ion < nl_ecp.PP.size(); ++ion)
          if (nl_ecp.PP[ion] && distances[ion] < nl_ecp.PP[ion]->getRmax())
          {
            const Real pair_potential = nl_ecp.PP[ion]->evaluateOne(
                particles, nl_ecp.vp_ ? makeOptionalRef<VirtualParticleSet>(*nl_ecp.vp_) : std::nullopt, ion, psi,
                electron, distances[ion], -displacements[ion],
                compute_tmove_data ? makeOptionalRef<std::vector<NonLocalData>>(candidates) : std::nullopt,
                nl_ecp.use_DLA);
            rows.electron[electron] += Real(0.5) * pair_potential;
            rows.ion[ion] += Real(0.5) * pair_potential;
          }
      }
    }
    return rows;
  }
};

} // namespace testing

TEST_CASE("NonLocalECPotential", "[hamiltonian]")
{
  using Real         = QMCTraits::RealType;
  using FullPrecReal = QMCTraits::FullPrecRealType;
  using Position     = QMCTraits::PosType;
  using testing::getParticularListener;

  Lattice lattice;
  lattice.BoxBConds = true; // periodic
  lattice.R.diagonal(20.0);
  lattice.LR_dim_cutoff = 15;
  lattice.reset();

  const SimulationCell simulation_cell(lattice);

  ParticleSet ions(simulation_cell);

  ions.setName("ion");
  ions.create({2});
  ions.R[0] = {0.0, 1.0, 0.0};
  ions.R[1] = {0.0, -1.0, 0.0};

  SpeciesSet& ion_species                         = ions.getSpeciesSet();
  int index_species                               = ion_species.addSpecies("Na");
  int index_charge                                = ion_species.addAttribute("charge");
  int index_atomic_number                         = ion_species.addAttribute("atomic_number");
  ion_species(index_charge, index_species)        = 1;
  ion_species(index_atomic_number, index_species) = 1;
  ions.createSK();
  ions.resetGroups(); // test_ecp.cpp claims this is needed
  ions.update();      // elsewhere its implied this is needed

  ParticleSet ions2(ions);
  ions2.update();

  ParticleSet elec(simulation_cell);
  elec.setName("elec");
  elec.create({2, 1});
  elec.R[0] = {0.4, 0.0, 0.0};
  elec.R[1] = {1.0, 0.0, 0.0};

  SpeciesSet& tspecies       = elec.getSpeciesSet();
  int upIdx                  = tspecies.addSpecies("u");
  int chargeIdx              = tspecies.addAttribute("charge");
  int massIdx                = tspecies.addAttribute("mass");
  tspecies(chargeIdx, upIdx) = -1;
  tspecies(massIdx, upIdx)   = 1.0;

  int dnIdx                  = tspecies.addSpecies("d");
  chargeIdx                  = tspecies.addAttribute("charge");
  massIdx                    = tspecies.addAttribute("mass");
  tspecies(chargeIdx, dnIdx) = -1;
  tspecies(massIdx, dnIdx)   = 1.0;

  elec.createSK();
  elec.resetGroups();
  elec.addTable(ions);
  elec.update();

  ParticleSet elec2(elec);
  elec2.update();

  RefVectorWithLeader<ParticleSet> p_list(elec, {elec, elec2});

  RuntimeOptions runtime_options;
  TrialWaveFunction psi(runtime_options);
  TrialWaveFunction psi2(runtime_options);
  RefVectorWithLeader<TrialWaveFunction> twf_list(psi, {psi, psi2});

  NonLocalECPotential nl_ecp(ions, elec, false /*use_DLA*/, false /*use_VP*/);

  int num_walkers = 2;
  int max_values  = 10;
  Matrix<Real> local_pots(num_walkers, max_values);
  Matrix<Real> local_pots2(num_walkers, max_values);

  ResourceCollection pset_res("test_pset_res");
  elec.createResource(pset_res);
  ResourceCollectionTeamLock<ParticleSet> pset_lock(pset_res, p_list);

  std::vector<ListenerVector<Real>> listeners;
  listeners.emplace_back("nonlocalpotential", getParticularListener(local_pots));
  listeners.emplace_back("nonlocalpotential", getParticularListener(local_pots2));

  Matrix<Real> ion_pots(num_walkers, max_values);
  Matrix<Real> ion_pots2(num_walkers, max_values);

  std::vector<ListenerVector<Real>> ion_listeners;
  ion_listeners.emplace_back("nonlocalpotential", getParticularListener(ion_pots));
  ion_listeners.emplace_back("nonlocalpotential", getParticularListener(ion_pots2));


  // This took some time to sort out from the multistage mess of put and clones
  // but this accomplishes in a straight forward way what I interpret to be done by that code.
  Communicate* comm = OHMMS::Controller;
  ECPComponentBuilder ecp_comp_builder("test_read_ecp", comm, 4, 1);

  bool okay = ecp_comp_builder.read_pp_file("Na.BFD.xml");
  REQUIRE(okay);
  UPtr<NonLocalECPComponent> nl_ecp_comp = std::move(ecp_comp_builder.pp_nonloc);
  nl_ecp.addComponent(0, std::move(nl_ecp_comp));
  UPtr<OperatorBase> nl_ecp2_ptr = nl_ecp.makeClone(elec2, psi2);
  auto& nl_ecp2                  = dynamic_cast<NonLocalECPotential&>(*nl_ecp2_ptr);

  StdRandom<FullPrecReal> rng(10101);
  StdRandom<FullPrecReal> rng2(10201);
  nl_ecp.setRandomGenerator(&rng);
  nl_ecp2.setRandomGenerator(&rng2);

  RefVectorWithLeader<OperatorBase> o_list(nl_ecp, {nl_ecp, nl_ecp2});
  ResourceCollection nl_ecp_res("test_nl_ecp_res");
  nl_ecp.createResource(nl_ecp_res);
  ResourceCollectionTeamLock<OperatorBase> nl_ecp_lock(nl_ecp_res, o_list);

  // Despite what test_ecp.cpp says this does not need to be done.
  // I think because of the pp
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(nl_ecp);
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(nl_ecp2);

  CHECK(!testing::TestNonLocalECPotential::didGridChange(nl_ecp));

  ListenerOption<Real> listener_opt{listeners, ion_listeners};
  testing::TestNonLocalECPotential::mw_evaluateImpl(nl_ecp, o_list, twf_list, p_list, false, listener_opt, true);

  // for now we'll check against the single particle API
  testing::TestNonLocalECPotential::evaluateImpl(nl_ecp, psi, elec, false, true);
  const auto value = nl_ecp.getValue();

  double total_localpots = std::accumulate(local_pots.begin(), local_pots.begin() + local_pots.cols(), 0.0);
  total_localpots += std::accumulate(ion_pots.begin(), ion_pots.begin() + ion_pots.cols(), 0.0);
  CHECK(total_localpots == Approx(value));
  double total_localpots2 = std::accumulate(local_pots[1], local_pots[1] + local_pots.cols(), 0.0);
  total_localpots2 += std::accumulate(ion_pots[1], ion_pots[1] + ion_pots.cols(), 0.0);
  CHECK(total_localpots2 == Approx(value));

  CHECK(!testing::TestNonLocalECPotential::didGridChange(nl_ecp));

  elec.R[0] = {0.5, 0.0, 2.0};
  elec.update();

  testing::TestNonLocalECPotential::mw_evaluateImpl(nl_ecp, o_list, twf_list, p_list, false, listener_opt, true);

  CHECK(!testing::TestNonLocalECPotential::didGridChange(nl_ecp));
  auto value_moved = o_list[0].evaluateDeterministic(psi, elec);

  total_localpots = std::accumulate(local_pots.begin(), local_pots.begin() + local_pots.cols(), 0.0);
  total_localpots += std::accumulate(ion_pots.begin(), ion_pots.begin() + ion_pots.cols(), 0.0);
  CHECK(total_localpots == Approx(value_moved));
  // check the second walker which will be unchanged.
  total_localpots2 = std::accumulate(local_pots[1], local_pots[1] + local_pots.cols(), 0.0);
  total_localpots2 += std::accumulate(ion_pots[1], ion_pots[1] + ion_pots.cols(), 0.0);
  CHECK(total_localpots2 == Approx(value));

  testing::TestNonLocalECPotential::mw_evaluateImpl(nl_ecp, o_list, twf_list, p_list, false, listener_opt, false);
  auto value3     = o_list[0].evaluateDeterministic(twf_list[0], p_list[0]);
  total_localpots = std::accumulate(local_pots.begin(), local_pots.begin() + local_pots.cols(), 0.0);
  total_localpots += std::accumulate(ion_pots.begin(), ion_pots.begin() + ion_pots.cols(), 0.0);

  CHECK(total_localpots == Approx(value3));
}

namespace
{
/** the two ions and the base three-electron ParticleSet shared by all three
 *  v1 T-move tests below; only electron 1's position and the RNG seed vary
 *  per walker.
 */
SimulationCell makeTmoveV1SimulationCell()
{
  Lattice lattice;
  lattice.BoxBConds = true;
  lattice.R.diagonal(20.0);
  lattice.LR_dim_cutoff = 15;
  lattice.reset();
  return SimulationCell(lattice);
}

ParticleSet makeTmoveV1Ions(const SimulationCell& simulation_cell)
{
  ParticleSet ions(simulation_cell);
  ions.setName("ion");
  ions.create({2});
  ions.R[0]                                       = {0.0, 1.0, 0.0};
  ions.R[1]                                       = {0.0, -1.0, 0.0};
  SpeciesSet& ion_species                         = ions.getSpeciesSet();
  int index_species                               = ion_species.addSpecies("Na");
  int index_charge                                = ion_species.addAttribute("charge");
  int index_atomic_number                         = ion_species.addAttribute("atomic_number");
  ion_species(index_charge, index_species)        = 1;
  ion_species(index_atomic_number, index_species) = 1;
  ions.createSK();
  ions.resetGroups();
  ions.update();
  return ions;
}

ParticleSet makeTmoveV1Elec(const SimulationCell& simulation_cell,
                            const ParticleSet& ions,
                            const QMCTraits::PosType& r0,
                            const QMCTraits::PosType& r1,
                            const QMCTraits::PosType& r2)
{
  ParticleSet elec(simulation_cell);
  elec.setName("elec");
  elec.create({2, 1});
  elec.R[0]                  = r0;
  elec.R[1]                  = r1;
  elec.R[2]                  = r2;
  SpeciesSet& tspecies       = elec.getSpeciesSet();
  int upIdx                  = tspecies.addSpecies("u");
  int chargeIdx              = tspecies.addAttribute("charge");
  int massIdx                = tspecies.addAttribute("mass");
  tspecies(chargeIdx, upIdx) = -1;
  tspecies(massIdx, upIdx)   = 1.0;
  int dnIdx                  = tspecies.addSpecies("d");
  chargeIdx                  = tspecies.addAttribute("charge");
  massIdx                    = tspecies.addAttribute("mass");
  tspecies(chargeIdx, dnIdx) = -1;
  tspecies(massIdx, dnIdx)   = 1.0;
  elec.createSK();
  elec.resetGroups();
  elec.addTable(ions);
  elec.update();
  return elec;
}

/// Read the Na.BFD.xml component with an explicitly selected angular rule.
UPtr<NonLocalECPComponent> readTmoveV1PPComponent(int quadrature_rule = 4)
{
  Communicate* comm = OHMMS::Controller;
  ECPComponentBuilder ecp_comp_builder("test_read_ecp", comm, quadrature_rule, 1);
  bool okay = ecp_comp_builder.read_pp_file("Na.BFD.xml");
  REQUIRE(okay);
  return std::move(ecp_comp_builder.pp_nonloc);
}

bool sameRealBits(QMCTraits::RealType left, QMCTraits::RealType right)
{
  return std::memcmp(&left, &right, sizeof(left)) == 0;
}

bool samePositionBits(const QMCTraits::PosType& left, const QMCTraits::PosType& right)
{
  for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
    if (!sameRealBits(left[dimension], right[dimension]))
      return false;
  return true;
}

/** Shared controls for a deterministic generic flattened-ratio test component. */
struct StampedRatioControl
{
  int scalar_calls                    = 0;
  int flattened_calls                 = 0;
  int weighted_calls                  = 0;
  int change_version_at_call          = 0;
  int weighted_change_version_at_call = 0;
  int throw_after_call                = 0;
  int weighted_throw_after_call       = 0;
  std::uint64_t version               = 7;
  QMCTraits::ValueType ratio                = QMCTraits::ValueType(1);
  QMCTraits::ValueType derivative_increment = QMCTraits::ValueType(0);
  bool derivative_uses_total_weights        = false;
  std::size_t maximum_segments               = 0;
  bool saw_repeated_walker                   = false;
  bool saw_mixed_electrons                   = false;
};

QMCTraits::ValueType makeStampedRatio(QMCTraits::RealType real_part,
                                      QMCTraits::RealType imaginary_part)
{
#if defined(QMC_COMPLEX)
  return QMCTraits::ValueType(real_part, imaginary_part);
#else
  static_cast<void>(imaginary_part);
  return QMCTraits::ValueType(real_part);
#endif
}

/** Exercise the generic WaveFunctionComponent VP fallback while returning a test stamp. */
class StampedRatioOrbital : public ConstantOrbital
{
public:
  StampedRatioOrbital(std::shared_ptr<StampedRatioControl> control, bool fermionic)
      : control_(std::move(control)), fermionic_(fermionic)
  {}

  std::string getClassName() const override { return "StampedRatioOrbital"; }
  bool isFermionic() const override { return fermionic_; }

  void evaluateRatios(const VirtualParticleSet& virtual_particles,
                      std::vector<ValueType>& ratios) override
  {
    if (ratios.size() != static_cast<std::size_t>(virtual_particles.getTotalNum()))
      throw std::invalid_argument("StampedRatioOrbital received a mismatched ratio extent.");
    ++control_->scalar_calls;
    std::fill(ratios.begin(), ratios.end(), control_->ratio);
  }

  EvaluationStamp mw_evaluateVirtualRatios(
      const RefVectorWithLeader<WaveFunctionComponent>& component_list,
      const RefVectorWithLeader<ParticleSet>& particle_list,
      const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
      const VirtualParticleBatch& batch,
      std::vector<ValueType>& ratios) const override
  {
    const int call = ++control_->flattened_calls;
    const EvaluationStamp stamp = EvaluationStamp::versioned(
        control_.get(), control_->version + (control_->change_version_at_call > 0 &&
                                              call >= control_->change_version_at_call));
    control_->maximum_segments = std::max(control_->maximum_segments, batch.segmentCount());
    for (std::size_t segment = 0; segment < batch.segmentCount(); ++segment)
      for (std::size_t prior = 0; prior < segment; ++prior)
      {
        control_->saw_repeated_walker |=
            batch.segment(segment).walkerId() == batch.segment(prior).walkerId();
        control_->saw_mixed_electrons |=
            batch.segment(segment).electronId() != batch.segment(prior).electronId();
      }
    WaveFunctionComponent::mw_evaluateVirtualRatios(component_list, particle_list, vp_scratch_list,
                                                     batch, ratios);
    if (control_->throw_after_call == call)
      throw std::runtime_error("deliberate late flattened NLPP tile failure");
    return stamp;
  }

  void evaluateDerivRatiosWeighted(
      const VirtualParticleSet& virtual_particles,
      const OptVariables&,
      const std::vector<ValueType>& total_weights,
      ParameterDerivativeView weighted_derivatives) override
  {
    if (total_weights.size() != static_cast<std::size_t>(virtual_particles.getTotalNum()))
      throw std::invalid_argument("StampedRatioOrbital received mismatched weighted-derivative input.");

    // A deterministic per-segment contribution lets the atomicity regression
    // detect even one prematurely published tile. Other stamped-ratio tests
    // retain the default zero increment.
    ValueType segment_increment;
    if (control_->derivative_uses_total_weights)
      segment_increment = control_->derivative_increment * std::accumulate(
          total_weights.begin(), total_weights.end(), ValueType(0));
    else
      segment_increment =
          control_->derivative_increment * ValueType(virtual_particles.getTotalNum());
    for (std::size_t parameter = 0; parameter < weighted_derivatives.size; ++parameter)
      weighted_derivatives[parameter] += segment_increment;
  }

  EvaluationStamp mw_evaluateVirtualDerivRatiosWeighted(
      const RefVectorWithLeader<WaveFunctionComponent>& component_list,
      const RefVectorWithLeader<ParticleSet>& particle_list,
      const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
      const VirtualParticleBatch& batch,
      const OptVariables& optvars,
      const std::vector<ValueType>& total_weights,
      const std::vector<ParameterDerivativeView>& weighted_derivatives) const override
  {
    const int call = ++control_->weighted_calls;
    WaveFunctionComponent::mw_evaluateVirtualDerivRatiosWeighted(
        component_list, particle_list, vp_scratch_list, batch, optvars, total_weights,
        weighted_derivatives);
    if (control_->weighted_throw_after_call == call)
      throw std::runtime_error("deliberate late flattened NLPP weighted-derivative failure");

    // Mirror an intentional value-version change by default so that the
    // Hamiltonian can detect it across tiles. The independent weighted knob
    // instead exercises the TrialWaveFunction's same-tile phase check.
    const bool value_version_changed =
        control_->change_version_at_call > 0 &&
        control_->flattened_calls >= control_->change_version_at_call;
    const bool weighted_version_changed =
        control_->weighted_change_version_at_call > 0 &&
        call >= control_->weighted_change_version_at_call;
    return EvaluationStamp::versioned(
        control_.get(), control_->version + value_version_changed + weighted_version_changed);
  }

  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet&) const override
  { return std::make_unique<StampedRatioOrbital>(control_, fermionic_); }

private:
  std::shared_ptr<StampedRatioControl> control_;
  bool fermionic_;
};

/** Compare complete ordered candidate vectors, including floating-point bits. */
bool sameCandidates(const std::vector<NonLocalData>& left,
                    const std::vector<NonLocalData>& right)
{
  if (left.size() != right.size())
    return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (left[index].PID != right[index].PID ||
        !sameRealBits(left[index].Weight, right[index].Weight) ||
        !samePositionBits(left[index].Delta, right[index].Delta))
      return false;
  return true;
}

/** Compare the physical fields and order of private/public legacy job lists. */
bool sameJobs(const std::vector<std::vector<NLPPJob<QMCTraits::RealType>>>& left,
              const std::vector<std::vector<NLPPJob<QMCTraits::RealType>>>& right)
{
  if (left.size() != right.size())
    return false;
  for (std::size_t group = 0; group < left.size(); ++group)
  {
    if (left[group].size() != right[group].size())
      return false;
    for (std::size_t job = 0; job < left[group].size(); ++job)
      if (left[group][job].ion_id != right[group][job].ion_id ||
          left[group][job].electron_id != right[group][job].electron_id ||
          !sameRealBits(left[group][job].ion_elec_dist, right[group][job].ion_elec_dist) ||
          !samePositionBits(left[group][job].ion_elec_displ, right[group][job].ion_elec_displ))
        return false;
  }
  return true;
}

/** Snapshot every Hamiltonian-visible destination owned by one NLPP operator. */
struct NLPPPublicSnapshot
{
  QMCTraits::RealType value;
  std::vector<NonLocalData> candidates;
  std::vector<std::vector<NLPPJob<QMCTraits::RealType>>> jobs;
  std::vector<std::vector<int>> electron_neighbors;
  std::vector<std::vector<int>> ion_neighbors;
};

NLPPPublicSnapshot snapshotNLPPPublicState(const NonLocalECPotential& potential,
                                           int electron_count,
                                           int ion_count)
{
  NLPPPublicSnapshot snapshot{potential.getValue(),
                              testing::TestNonLocalECPotential::tmoveCandidates(potential),
                              testing::TestNonLocalECPotential::jobs(potential)};
  for (int electron = 0; electron < electron_count; ++electron)
    snapshot.electron_neighbors.push_back(
        testing::TestNonLocalECPotential::neighboringIons(potential, electron));
  for (int ion = 0; ion < ion_count; ++ion)
    snapshot.ion_neighbors.push_back(
        testing::TestNonLocalECPotential::neighboringElectrons(potential, ion));
  return snapshot;
}

bool sameNLPPPublicState(const NonLocalECPotential& potential,
                         const NLPPPublicSnapshot& snapshot,
                         int electron_count,
                         int ion_count)
{
  if (!sameRealBits(potential.getValue(), snapshot.value) ||
      !sameCandidates(testing::TestNonLocalECPotential::tmoveCandidates(potential), snapshot.candidates) ||
      !sameJobs(testing::TestNonLocalECPotential::jobs(potential), snapshot.jobs))
    return false;
  for (int electron = 0; electron < electron_count; ++electron)
    if (testing::TestNonLocalECPotential::neighboringIons(potential, electron) !=
        snapshot.electron_neighbors[electron])
      return false;
  for (int ion = 0; ion < ion_count; ++ion)
    if (testing::TestNonLocalECPotential::neighboringElectrons(potential, ion) !=
        snapshot.ion_neighbors[ion])
      return false;
  return true;
}

TEST_CASE("NonLocalECPotential flattened outer tiles preserve ordinary locality outputs",
          "[hamiltonian][ecp][nlpp_flattened]")
{
  using Real = QMCTraits::RealType;
  using testing::getParticularListener;

  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet electrons = makeTmoveV1Elec(
      simulation_cell, ions, {0.4, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-0.4, 0.6, -0.3});
  ParticleSet empty_walker(electrons);
  for (int electron = 0; electron < empty_walker.getTotalNum(); ++electron)
    empty_walker.R[electron] = {8.0 + electron, 8.0, 8.0};
  empty_walker.update();

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options);
  TrialWaveFunction empty_wavefunction(runtime_options);
  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons, empty_walker});
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(
      wavefunction, {wavefunction, empty_wavefunction});

  NonLocalECPotential potential(ions, electrons, false, true);
  testing::TestNonLocalECPotential::setOuterTileCapacity(potential, 3);
  potential.addComponent(0, readTmoveV1PPComponent());
  UPtr<OperatorBase> empty_potential_storage = potential.makeClone(empty_walker, empty_wavefunction);
  auto& empty_potential = dynamic_cast<NonLocalECPotential&>(*empty_potential_storage);
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(potential);
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(empty_potential);

  REQUIRE(testing::TestNonLocalECPotential::firstComponentKnotCount(potential) > 3);

  // Build independent scalar references before the crowd resource is acquired.
  testing::TestNonLocalECPotential::evaluateImpl(potential, wavefunction, electrons, true, true);
  testing::TestNonLocalECPotential::evaluateImpl(
      empty_potential, empty_wavefunction, empty_walker, true, true);
  const Real expected_energy = potential.getValue();
  const Real expected_empty_energy = empty_potential.getValue();
  const std::vector<NonLocalData> expected_candidates =
      testing::TestNonLocalECPotential::tmoveCandidates(potential);
  const std::vector<NonLocalData> expected_empty_candidates =
      testing::TestNonLocalECPotential::tmoveCandidates(empty_potential);
  const auto expected_listener = testing::TestNonLocalECPotential::evaluateScalarListenerRows(
      potential, wavefunction, electrons);
  const auto expected_empty_listener = testing::TestNonLocalECPotential::evaluateScalarListenerRows(
      empty_potential, empty_wavefunction, empty_walker);
  std::vector<std::vector<int>> expected_electron_neighbors;
  std::vector<std::vector<int>> expected_ion_neighbors;
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    expected_electron_neighbors.push_back(
        testing::TestNonLocalECPotential::neighboringIons(potential, electron));
  for (int ion = 0; ion < ions.getTotalNum(); ++ion)
    expected_ion_neighbors.push_back(
        testing::TestNonLocalECPotential::neighboringElectrons(potential, ion));

  REQUIRE(!expected_candidates.empty());
  CHECK(expected_empty_candidates.empty());
  CHECK(sameRealBits(expected_empty_energy, Real(0)));

  Matrix<Real> electron_rows(2, electrons.getTotalNum());
  Matrix<Real> ion_rows(2, ions.getTotalNum());
  electron_rows = Real(-91);
  ion_rows      = Real(-92);
  std::vector<ListenerVector<Real>> electron_listeners;
  std::vector<ListenerVector<Real>> ion_listeners;
  electron_listeners.emplace_back("flattened_electrons", getParticularListener(electron_rows));
  ion_listeners.emplace_back("flattened_ions", getParticularListener(ion_rows));

  RefVectorWithLeader<OperatorBase> potentials(potential, {potential, empty_potential});
  ResourceCollection particle_resources("flattened_locality_particles");
  ResourceCollection potential_resources("flattened_locality_potential");
  electrons.createResource(particle_resources);
  potential.createResource(potential_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
  ResourceCollectionTeamLock<OperatorBase> potential_lock(potential_resources, potentials);

  // A callback may synchronously enter another evaluation transaction. This
  // pins that the nested VP resource lock is released before callbacks begin.
  bool callback_reentered           = false;
  bool callback_saw_committed_state = false;
  electron_listeners.emplace_back(
      "flattened_reentry",
      [&](int walker, const std::string&, const Vector<Real>&) {
        if (walker != 0 || callback_reentered)
          return;
        callback_reentered           = true;
        callback_saw_committed_state =
            sameRealBits(potential.getValue(), expected_energy) &&
            sameCandidates(testing::TestNonLocalECPotential::tmoveCandidates(potential),
                           expected_candidates);
        testing::TestNonLocalECPotential::mw_evaluateImpl(
            potential, potentials, wavefunctions, particles, true, std::nullopt, true);
      });
  const ListenerOption<Real> listener_option{electron_listeners, ion_listeners};

  testing::TestNonLocalECPotential::mw_evaluateImpl(
      potential, potentials, wavefunctions, particles, true, listener_option, true);

  CHECK(callback_reentered);
  CHECK(callback_saw_committed_state);
  CHECK(sameRealBits(potential.getValue(), expected_energy));
  CHECK(sameRealBits(empty_potential.getValue(), expected_empty_energy));
  CHECK(sameCandidates(testing::TestNonLocalECPotential::tmoveCandidates(potential),
                       expected_candidates));
  CHECK(sameCandidates(testing::TestNonLocalECPotential::tmoveCandidates(empty_potential),
                       expected_empty_candidates));
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
  {
    CHECK(testing::TestNonLocalECPotential::neighboringIons(potential, electron) ==
          expected_electron_neighbors[electron]);
    CHECK(sameRealBits(electron_rows(0, electron), expected_listener.electron[electron]));
    CHECK(sameRealBits(electron_rows(1, electron), expected_empty_listener.electron[electron]));
  }
  for (int ion = 0; ion < ions.getTotalNum(); ++ion)
  {
    CHECK(testing::TestNonLocalECPotential::neighboringElectrons(potential, ion) ==
          expected_ion_neighbors[ion]);
    CHECK(sameRealBits(ion_rows(0, ion), expected_listener.ion[ion]));
    CHECK(sameRealBits(ion_rows(1, ion), expected_empty_listener.ion[ion]));
  }
  for (const auto& group_jobs : testing::TestNonLocalECPotential::jobs(potential))
    CHECK_FALSE(group_jobs.empty());
  for (const auto& group_jobs : testing::TestNonLocalECPotential::jobs(empty_potential))
    CHECK(group_jobs.empty());

  // With every walker outside the cutoff, the same transaction must publish
  // complete empty logical outputs and zero listener rows.
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    electrons.R[electron] = {7.0 + electron, 8.0, 8.0};
  electrons.update();
  electron_rows = Real(-93);
  ion_rows      = Real(-94);
  testing::TestNonLocalECPotential::mw_evaluateImpl(
      potential, potentials, wavefunctions, particles, true, listener_option, true);
  for (std::size_t walker = 0; walker < 2; ++walker)
  {
    const auto& walker_potential =
        dynamic_cast<const NonLocalECPotential&>(potentials[walker]);
    CHECK(sameRealBits(walker_potential.getValue(), Real(0)));
    CHECK(testing::TestNonLocalECPotential::tmoveCandidates(walker_potential).empty());
    for (const auto& group_jobs : testing::TestNonLocalECPotential::jobs(walker_potential))
      CHECK(group_jobs.empty());
    for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
      CHECK(sameRealBits(electron_rows(walker, electron), Real(0)));
    for (int ion = 0; ion < ions.getTotalNum(); ++ion)
      CHECK(sameRealBits(ion_rows(walker, ion), Real(0)));
  }
}

TEST_CASE("NonLocalECPotential flattened outer tiles publish atomically",
          "[hamiltonian][ecp][nlpp_flattened][atomic]")
{
  using Real = QMCTraits::RealType;
  using testing::getParticularListener;

  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet electrons = makeTmoveV1Elec(
      simulation_cell, ions, {0.4, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-0.4, 0.6, -0.3});

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options);
  auto control = std::make_shared<StampedRatioControl>();
  wavefunction.addComponent(std::make_unique<StampedRatioOrbital>(control, true));
  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons});
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(wavefunction, {wavefunction});

  NonLocalECPotential potential(ions, electrons, false, true);
  testing::TestNonLocalECPotential::setOuterTileCapacity(potential, 3);
  potential.addComponent(0, readTmoveV1PPComponent());
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(potential);
  RefVectorWithLeader<OperatorBase> potentials(potential, {potential});

  ResourceCollection particle_resources("flattened_atomic_particles");
  ResourceCollection potential_resources("flattened_atomic_potential");
  electrons.createResource(particle_resources);
  potential.createResource(potential_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
  ResourceCollectionTeamLock<OperatorBase> potential_lock(potential_resources, potentials);

  testing::TestNonLocalECPotential::mw_evaluateImpl(
      potential, potentials, wavefunctions, particles, true, std::nullopt, true);
  REQUIRE(control->flattened_calls > 1);
  const NLPPPublicSnapshot initial = snapshotNLPPPublicState(
      potential, electrons.getTotalNum(), ions.getTotalNum());

  Matrix<Real> electron_rows(1, electrons.getTotalNum());
  Matrix<Real> ion_rows(1, ions.getTotalNum());
  std::vector<ListenerVector<Real>> electron_listeners;
  std::vector<ListenerVector<Real>> ion_listeners;
  electron_listeners.emplace_back("atomic_electrons", getParticularListener(electron_rows));
  ion_listeners.emplace_back("atomic_ions", getParticularListener(ion_rows));
  const ListenerOption<Real> listener_option{electron_listeners, ion_listeners};

  // A later tile reports a different monotonic model version. No internal or
  // external destination may observe the already reduced first tile.
  control->flattened_calls        = 0;
  control->change_version_at_call = 2;
  electron_rows                  = Real(-71);
  ion_rows                       = Real(-72);
  CHECK_THROWS_AS(testing::TestNonLocalECPotential::mw_evaluateImpl(
                      potential, potentials, wavefunctions, particles, true,
                      listener_option, true),
                  std::runtime_error);
  CHECK(control->flattened_calls == 2);
  CHECK(sameNLPPPublicState(potential, initial, electrons.getTotalNum(), ions.getTotalNum()));
  for (const Real value : electron_rows)
    CHECK(sameRealBits(value, Real(-71)));
  for (const Real value : ion_rows)
    CHECK(sameRealBits(value, Real(-72)));

  // Resource and private staging must be immediately reusable after failure.
  control->flattened_calls        = 0;
  control->change_version_at_call = 0;
  testing::TestNonLocalECPotential::mw_evaluateImpl(
      potential, potentials, wavefunctions, particles, true, listener_option, true);
  REQUIRE(control->flattened_calls > 1);
  const NLPPPublicSnapshot after_retry = snapshotNLPPPublicState(
      potential, electrons.getTotalNum(), ions.getTotalNum());
  CHECK(sameCandidates(after_retry.candidates, initial.candidates));

  // Throw only after the generic fallback has materialized and evaluated the
  // second tile. The nested VP resource lock must unwind along with all staging.
  control->flattened_calls = 0;
  control->throw_after_call = 2;
  electron_rows             = Real(-81);
  ion_rows                  = Real(-82);
  CHECK_THROWS_AS(testing::TestNonLocalECPotential::mw_evaluateImpl(
                      potential, potentials, wavefunctions, particles, true,
                      listener_option, true),
                  std::runtime_error);
  CHECK(control->flattened_calls == 2);
  CHECK(sameNLPPPublicState(potential, after_retry, electrons.getTotalNum(), ions.getTotalNum()));
  for (const Real value : electron_rows)
    CHECK(sameRealBits(value, Real(-81)));
  for (const Real value : ion_rows)
    CHECK(sameRealBits(value, Real(-82)));

  control->flattened_calls = 0;
  control->throw_after_call = 0;
  CHECK_NOTHROW(testing::TestNonLocalECPotential::mw_evaluateImpl(
      potential, potentials, wavefunctions, particles, true, std::nullopt, true));
  CHECK(sameNLPPPublicState(potential, after_retry, electrons.getTotalNum(), ions.getTotalNum()));
}

TEST_CASE("NonLocalECPotential flattened weighted derivatives publish atomically",
          "[hamiltonian][ecp][nlpp_flattened][derivatives][atomic]")
{
  using FullPrecReal = QMCTraits::FullPrecRealType;
  using ValueType    = QMCTraits::ValueType;

  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet electrons = makeTmoveV1Elec(
      simulation_cell, ions, {0.4, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-0.4, 0.6, -0.3});

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options);
  auto control   = std::make_shared<StampedRatioControl>();
  control->ratio = makeStampedRatio(0.83, 0.27);
  control->derivative_increment = makeStampedRatio(0.125, 0.0625);
  wavefunction.addComponent(std::make_unique<StampedRatioOrbital>(control, true));

  NonLocalECPotential potential(ions, electrons, false, true);
  testing::TestNonLocalECPotential::setOuterTileCapacity(potential, 3);
  potential.addComponent(0, readTmoveV1PPComponent());
  REQUIRE(testing::TestNonLocalECPotential::firstComponentKnotCount(potential) > 3);

  FakeRandom<FullPrecReal> grid_rng;
  grid_rng.set_value(0.371);
  potential.setRandomGenerator(&grid_rng);

  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons});
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(wavefunction, {wavefunction});
  RefVectorWithLeader<OperatorBase> potentials(potential, {potential});
  ResourceCollection particle_resources("flattened_derivative_atomic_particles");
  ResourceCollection potential_resources("flattened_derivative_atomic_potential");
  electrons.createResource(particle_resources);
  potential.createResource(potential_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
  ResourceCollectionTeamLock<OperatorBase> potential_lock(potential_resources, potentials);

  OptVariables active;
  active.insert("unused_guard_parameter", 0.0);
  active.resetIndex();
  RecordArray<ValueType> scores(1, 1);
  RecordArray<ValueType> derivatives(1, 1);
  scores[0][0]      = ValueType(-2.25);
  derivatives[0][0] = ValueType(1.25);

  // Establish a successful derivative result and public-state baseline using
  // multiple outer tiles. The deterministic nonzero component contribution
  // also measures the full-request delta expected from a successful retry.
  potential.mw_evaluateWithParameterDerivatives(
      potentials, wavefunctions, particles, active, scores, derivatives);
  REQUIRE(control->flattened_calls > 1);
  REQUIRE(control->weighted_calls == control->flattened_calls);
  const ValueType successful_derivative_delta = derivatives[0][0] - ValueType(1.25);
  CHECK(std::abs(successful_derivative_delta) > 0);
  const NLPPPublicSnapshot baseline = snapshotNLPPPublicState(
      potential, electrons.getTotalNum(), ions.getTotalNum());

  auto exercise_failure_and_retry = [&]() {
    control->flattened_calls = 0;
    control->weighted_calls  = 0;
    derivatives[0][0]        = ValueType(9.25);
    CHECK_THROWS_AS(potential.mw_evaluateWithParameterDerivatives(
                        potentials, wavefunctions, particles, active, scores, derivatives),
                    std::runtime_error);
    CHECK(control->flattened_calls == 2);
    CHECK(control->weighted_calls == 2);
    CHECK(scores[0][0] == ValueApprox(ValueType(-2.25)));
    CHECK(derivatives[0][0] == ValueApprox(ValueType(9.25)));
    CHECK(sameNLPPPublicState(
        potential, baseline, electrons.getTotalNum(), ions.getTotalNum()));

    // Clear every failure mode and prove the same borrowed crowd resource can
    // immediately complete and publish a new request.
    control->change_version_at_call          = 0;
    control->weighted_change_version_at_call = 0;
    control->weighted_throw_after_call       = 0;
    control->flattened_calls                 = 0;
    control->weighted_calls                  = 0;
    derivatives[0][0]                        = ValueType(-6.5);
    CHECK_NOTHROW(potential.mw_evaluateWithParameterDerivatives(
        potentials, wavefunctions, particles, active, scores, derivatives));
    CHECK(control->flattened_calls > 1);
    CHECK(control->weighted_calls == control->flattened_calls);
    CHECK(derivatives[0][0] ==
          ValueApprox(ValueType(-6.5) + successful_derivative_delta));
    CHECK(sameNLPPPublicState(
        potential, baseline, electrons.getTotalNum(), ions.getTotalNum()));
  };

  SECTION("value version changes on a later tile")
  {
    control->change_version_at_call = 2;
    exercise_failure_and_retry();
  }
  SECTION("weighted phase changes version on a later tile")
  {
    control->weighted_change_version_at_call = 2;
    exercise_failure_and_retry();
  }
  SECTION("weighted phase throws on a later tile")
  {
    control->weighted_throw_after_call = 2;
    exercise_failure_and_retry();
  }
}

TEST_CASE("NonLocalECPotential flattened DLA selects only fermionic components",
          "[hamiltonian][ecp][nlpp_flattened][dla]")
{
  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet electrons = makeTmoveV1Elec(
      simulation_cell, ions, {0.4, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-0.4, 0.6, -0.3});

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options);
  auto fermionic_control    = std::make_shared<StampedRatioControl>();
  auto nonfermionic_control = std::make_shared<StampedRatioControl>();
  wavefunction.addComponent(std::make_unique<StampedRatioOrbital>(fermionic_control, true));
  wavefunction.addComponent(std::make_unique<StampedRatioOrbital>(nonfermionic_control, false));
  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons});
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(wavefunction, {wavefunction});

  NonLocalECPotential potential(ions, electrons, true, true);
  testing::TestNonLocalECPotential::setOuterTileCapacity(potential, 3);
  potential.addComponent(0, readTmoveV1PPComponent());
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(potential);
  testing::TestNonLocalECPotential::evaluateImpl(potential, wavefunction, electrons, false, true);
  const QMCTraits::RealType scalar_energy = potential.getValue();

  RefVectorWithLeader<OperatorBase> potentials(potential, {potential});
  ResourceCollection particle_resources("flattened_dla_particles");
  ResourceCollection potential_resources("flattened_dla_potential");
  electrons.createResource(particle_resources);
  potential.createResource(potential_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
  ResourceCollectionTeamLock<OperatorBase> potential_lock(potential_resources, potentials);

  testing::TestNonLocalECPotential::mw_evaluateImpl(
      potential, potentials, wavefunctions, particles, false, std::nullopt, true);
  CHECK(sameRealBits(potential.getValue(), scalar_energy));
  CHECK(fermionic_control->flattened_calls > 1);
  CHECK(nonfermionic_control->flattened_calls == 0);
}

TEST_CASE("NonLocalECPotential flattened TMDLA preserves dual-ratio outputs",
          "[hamiltonian][ecp][nlpp_flattened][tmdla]")
{
  using Real  = QMCTraits::RealType;
  using Value = QMCTraits::ValueType;
  using testing::getParticularListener;

  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet electrons = makeTmoveV1Elec(
      simulation_cell, ions, {0.4, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-0.4, 0.6, -0.3});

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options);
  auto fermionic_control          = std::make_shared<StampedRatioControl>();
  auto nonfermionic_control       = std::make_shared<StampedRatioControl>();
  fermionic_control->ratio        = makeStampedRatio(Real(-0.75), Real(0.5));
  nonfermionic_control->ratio     = makeStampedRatio(Real(-1.25), Real(0.75));
  wavefunction.addComponent(std::make_unique<StampedRatioOrbital>(fermionic_control, true));
  wavefunction.addComponent(std::make_unique<StampedRatioOrbital>(nonfermionic_control, false));

  NonLocalECPotential potential(ions, electrons, true, true);
  testing::TestNonLocalECPotential::setOuterTileCapacity(potential, 3);
  potential.addComponent(0, readTmoveV1PPComponent());
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(potential);

  // An ordinary-locality reference exposes the sign of bare*real(full ratio),
  // which is exactly the strict branch predicate used by TMDLA.
  NonLocalECPotential ordinary_potential(ions, electrons, false, true);
  ordinary_potential.addComponent(0, readTmoveV1PPComponent());
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(ordinary_potential);
  testing::TestNonLocalECPotential::evaluateImpl(
      ordinary_potential, wavefunction, electrons, true, true);
  const auto ordinary_candidates =
      testing::TestNonLocalECPotential::tmoveCandidates(ordinary_potential);

  testing::TestNonLocalECPotential::evaluateImpl(potential, wavefunction, electrons, true, true);
  const auto expected_listener = testing::TestNonLocalECPotential::evaluateScalarListenerRows(
      potential, wavefunction, electrons, true);
  const NLPPPublicSnapshot expected = snapshotNLPPPublicState(
      potential, electrons.getTotalNum(), ions.getTotalNum());

  REQUIRE(ordinary_candidates.size() == expected.candidates.size());
  const Value full_ratio = nonfermionic_control->ratio * fermionic_control->ratio;
  REQUIRE(std::real(full_ratio) != Real(0));
  int positive_full_knots    = 0;
  int nonpositive_full_knots = 0;
  int transformed_knots      = 0;
  for (std::size_t knot = 0; knot < ordinary_candidates.size(); ++knot)
  {
    CHECK(ordinary_candidates[knot].PID == expected.candidates[knot].PID);
    CHECK(samePositionBits(ordinary_candidates[knot].Delta, expected.candidates[knot].Delta));
    if (ordinary_candidates[knot].Weight > Real(0))
    {
      ++positive_full_knots;
      const Real bare_weight = ordinary_candidates[knot].Weight / std::real(full_ratio);
      CHECK(expected.candidates[knot].Weight ==
            Approx(bare_weight * std::real(fermionic_control->ratio)).epsilon(1e-12));
      transformed_knots += !sameRealBits(expected.candidates[knot].Weight,
                                         ordinary_candidates[knot].Weight);
    }
    else
    {
      ++nonpositive_full_knots;
      CHECK(sameRealBits(expected.candidates[knot].Weight,
                         ordinary_candidates[knot].Weight));
    }
  }
  REQUIRE(positive_full_knots > 0);
  REQUIRE(nonpositive_full_knots > 0);
  REQUIRE(transformed_knots > 0);

  Matrix<Real> electron_rows(1, electrons.getTotalNum());
  Matrix<Real> ion_rows(1, ions.getTotalNum());
  electron_rows = Real(-31);
  ion_rows      = Real(-32);
  std::vector<ListenerVector<Real>> electron_listeners;
  std::vector<ListenerVector<Real>> ion_listeners;
  electron_listeners.emplace_back("tmdla_electrons", getParticularListener(electron_rows));
  ion_listeners.emplace_back("tmdla_ions", getParticularListener(ion_rows));
  const ListenerOption<Real> listener_option{electron_listeners, ion_listeners};

  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons});
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(wavefunction, {wavefunction});
  RefVectorWithLeader<OperatorBase> potentials(potential, {potential});
  ResourceCollection particle_resources("flattened_tmdla_particles");
  ResourceCollection potential_resources("flattened_tmdla_potential");
  electrons.createResource(particle_resources);
  potential.createResource(potential_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
  ResourceCollectionTeamLock<OperatorBase> potential_lock(potential_resources, potentials);

  fermionic_control->flattened_calls    = 0;
  nonfermionic_control->flattened_calls = 0;
  testing::TestNonLocalECPotential::mw_evaluateImpl(
      potential, potentials, wavefunctions, particles, true, listener_option, true);

  REQUIRE(fermionic_control->flattened_calls > 1);
  CHECK(nonfermionic_control->flattened_calls == fermionic_control->flattened_calls);
  CHECK(sameRealBits(potential.getValue(), expected.value));
  CHECK(sameCandidates(testing::TestNonLocalECPotential::tmoveCandidates(potential),
                       expected.candidates));
  for (const auto& group_jobs : testing::TestNonLocalECPotential::jobs(potential))
    CHECK_FALSE(group_jobs.empty());
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    CHECK(testing::TestNonLocalECPotential::neighboringIons(potential, electron) ==
          expected.electron_neighbors[electron]);
  for (int ion = 0; ion < ions.getTotalNum(); ++ion)
    CHECK(testing::TestNonLocalECPotential::neighboringElectrons(potential, ion) ==
          expected.ion_neighbors[ion]);
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    CHECK(sameRealBits(electron_rows(0, electron), expected_listener.electron[electron]));
  for (int ion = 0; ion < ions.getTotalNum(); ++ion)
    CHECK(sameRealBits(ion_rows(0, ion), expected_listener.ion[ion]));
}

TEST_CASE("NonLocalECPotential flattened TMDLA stamp streams publish atomically",
          "[hamiltonian][ecp][nlpp_flattened][tmdla][atomic]")
{
  using Real = QMCTraits::RealType;
  using testing::getParticularListener;

  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet electrons = makeTmoveV1Elec(
      simulation_cell, ions, {0.4, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-0.4, 0.6, -0.3});

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options);
  auto fermionic_control      = std::make_shared<StampedRatioControl>();
  auto nonfermionic_control   = std::make_shared<StampedRatioControl>();
  fermionic_control->ratio    = makeStampedRatio(Real(-0.75), Real(0.5));
  nonfermionic_control->ratio = makeStampedRatio(Real(-1.25), Real(0.75));
  wavefunction.addComponent(std::make_unique<StampedRatioOrbital>(fermionic_control, true));
  wavefunction.addComponent(std::make_unique<StampedRatioOrbital>(nonfermionic_control, false));

  NonLocalECPotential potential(ions, electrons, true, true);
  // A capacity larger than either group's complete knot extent produces one
  // tile per group.  A version change on call two therefore pins that stamp
  // baselines span the entire request rather than resetting at prepareGroup.
  testing::TestNonLocalECPotential::setOuterTileCapacity(potential, 256);
  potential.addComponent(0, readTmoveV1PPComponent());
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(potential);

  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons});
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(wavefunction, {wavefunction});
  RefVectorWithLeader<OperatorBase> potentials(potential, {potential});
  ResourceCollection particle_resources("flattened_tmdla_atomic_particles");
  ResourceCollection potential_resources("flattened_tmdla_atomic_potential");
  electrons.createResource(particle_resources);
  potential.createResource(potential_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
  ResourceCollectionTeamLock<OperatorBase> potential_lock(potential_resources, potentials);

  testing::TestNonLocalECPotential::mw_evaluateImpl(
      potential, potentials, wavefunctions, particles, true, std::nullopt, true);
  REQUIRE(fermionic_control->flattened_calls == 2);
  REQUIRE(nonfermionic_control->flattened_calls == 2);
  const NLPPPublicSnapshot initial = snapshotNLPPPublicState(
      potential, electrons.getTotalNum(), ions.getTotalNum());

  Matrix<Real> electron_rows(1, electrons.getTotalNum());
  Matrix<Real> ion_rows(1, ions.getTotalNum());
  std::vector<ListenerVector<Real>> electron_listeners;
  std::vector<ListenerVector<Real>> ion_listeners;
  electron_listeners.emplace_back("tmdla_atomic_electrons", getParticularListener(electron_rows));
  ion_listeners.emplace_back("tmdla_atomic_ions", getParticularListener(ion_rows));
  const ListenerOption<Real> listener_option{electron_listeners, ion_listeners};

  fermionic_control->flattened_calls    = 0;
  nonfermionic_control->flattened_calls = 0;
  electron_rows                         = Real(-41);
  ion_rows                              = Real(-42);

  SECTION("fermionic version changes at the next group")
  {
    fermionic_control->change_version_at_call = 2;
  }
  SECTION("nonfermionic version changes at the next group")
  {
    nonfermionic_control->change_version_at_call = 2;
  }
  SECTION("nonfermionic evaluation throws at the next group")
  {
    nonfermionic_control->throw_after_call = 2;
  }

  CHECK_THROWS_AS(testing::TestNonLocalECPotential::mw_evaluateImpl(
                      potential, potentials, wavefunctions, particles, true,
                      listener_option, true),
                  std::runtime_error);
  CHECK(fermionic_control->flattened_calls == 2);
  if (fermionic_control->change_version_at_call != 0)
    CHECK(nonfermionic_control->flattened_calls == 1);
  else
    CHECK(nonfermionic_control->flattened_calls == 2);
  CHECK(sameNLPPPublicState(potential, initial, electrons.getTotalNum(), ions.getTotalNum()));
  for (const Real value : electron_rows)
    CHECK(sameRealBits(value, Real(-41)));
  for (const Real value : ion_rows)
    CHECK(sameRealBits(value, Real(-42)));

  fermionic_control->flattened_calls         = 0;
  fermionic_control->change_version_at_call  = 0;
  fermionic_control->throw_after_call        = 0;
  nonfermionic_control->flattened_calls      = 0;
  nonfermionic_control->change_version_at_call = 0;
  nonfermionic_control->throw_after_call       = 0;
  CHECK_NOTHROW(testing::TestNonLocalECPotential::mw_evaluateImpl(
      potential, potentials, wavefunctions, particles, true, std::nullopt, true));
  CHECK(sameNLPPPublicState(potential, initial, electrons.getTotalNum(), ions.getTotalNum()));
}

TEST_CASE("NonLocalECPotential flattened V1 electron candidates match scalar bits",
          "[hamiltonian][ecp][nlpp_v1_flattened]")
{
  using Real = QMCTraits::RealType;

  for (const bool use_dla : {false, true})
  {
    const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
    ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
    ParticleSet electrons = makeTmoveV1Elec(
        simulation_cell, ions, {0.4, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-0.4, 0.6, -0.3});

    RuntimeOptions runtime_options;
    TrialWaveFunction wavefunction(runtime_options);
    auto fermionic_control      = std::make_shared<StampedRatioControl>();
    auto nonfermionic_control   = std::make_shared<StampedRatioControl>();
    fermionic_control->ratio    = makeStampedRatio(Real(-0.75), Real(0.5));
    nonfermionic_control->ratio = makeStampedRatio(Real(-1.25), Real(0.75));
    wavefunction.addComponent(std::make_unique<StampedRatioOrbital>(fermionic_control, true));
    wavefunction.addComponent(std::make_unique<StampedRatioOrbital>(nonfermionic_control, false));

    NonLocalECPotential potential(ions, electrons, use_dla, true);
    potential.addComponent(0, readTmoveV1PPComponent());
    testing::TestNonLocalECPotential::copyGridUnrotatedForTest(potential);
    const int knot_count = testing::TestNonLocalECPotential::firstComponentKnotCount(potential);
    REQUIRE(knot_count > 1);
    // The first tile contains a whole ion job plus the first knot of the next,
    // proving that several same-walker jobs share one flattened descriptor;
    // the second job also continues into a tail tile.
    testing::TestNonLocalECPotential::setOuterTileCapacity(
        potential, static_cast<std::size_t>(knot_count + 1));

    testing::TestNonLocalECPotential::evaluateImpl(
        potential, wavefunction, electrons, false, true);
    wavefunction.prepareGroup(electrons, 0);
    const auto scalar_candidates =
        testing::TestNonLocalECPotential::evaluateScalarV1Candidates(
            potential, wavefunction, electrons, 0);
    const NLPPPublicSnapshot public_before = snapshotNLPPPublicState(
        potential, electrons.getTotalNum(), ions.getTotalNum());

    RefVectorWithLeader<ParticleSet> particles(electrons, {electrons});
    RefVectorWithLeader<TrialWaveFunction> wavefunctions(wavefunction, {wavefunction});
    RefVectorWithLeader<OperatorBase> potentials(potential, {potential});
    ResourceCollection particle_resources("flattened_v1_candidate_particles");
    ResourceCollection potential_resources("flattened_v1_candidate_potential");
    electrons.createResource(particle_resources);
    potential.createResource(potential_resources);
    ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
    ResourceCollectionTeamLock<OperatorBase> potential_lock(potential_resources, potentials);

    fermionic_control->flattened_calls       = 0;
    fermionic_control->maximum_segments      = 0;
    fermionic_control->saw_repeated_walker   = false;
    fermionic_control->saw_mixed_electrons   = false;
    nonfermionic_control->flattened_calls     = 0;
    nonfermionic_control->maximum_segments    = 0;
    nonfermionic_control->saw_repeated_walker = false;
    nonfermionic_control->saw_mixed_electrons = false;
    const auto flattened_candidates =
        testing::TestNonLocalECPotential::evaluateFlattenedV1Candidates(
            potential, potentials, wavefunctions, particles, 0, 0);

    REQUIRE(flattened_candidates.size() == 1);
    CHECK(sameCandidates(flattened_candidates[0], scalar_candidates));
    CHECK(flattened_candidates[0].size() == static_cast<std::size_t>(2 * knot_count));
    CHECK(fermionic_control->flattened_calls == 2);
    CHECK(fermionic_control->maximum_segments == 2);
    CHECK(fermionic_control->saw_repeated_walker);
    CHECK_FALSE(fermionic_control->saw_mixed_electrons);
    CHECK(nonfermionic_control->flattened_calls == 2);
    CHECK(nonfermionic_control->maximum_segments == 2);
    CHECK(nonfermionic_control->saw_repeated_walker);
    CHECK_FALSE(nonfermionic_control->saw_mixed_electrons);
    CHECK(sameNLPPPublicState(
        potential, public_before, electrons.getTotalNum(), ions.getTotalNum()));
  }
}

TEST_CASE("NonLocalECPotential flattened V1 tile failure precedes selection and permits retry",
          "[hamiltonian][ecp][nlpp_v1_flattened][atomic]")
{
  using FullPrecReal = QMCTraits::FullPrecRealType;

  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet electrons = makeTmoveV1Elec(
      simulation_cell, ions, {0.4, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-0.4, 0.6, -0.3});

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options);
  auto control = std::make_shared<StampedRatioControl>();
  wavefunction.addComponent(std::make_unique<StampedRatioOrbital>(control, true));

  NonLocalECPotential potential(ions, electrons, false, true);
  potential.addComponent(0, readTmoveV1PPComponent());
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(potential);
  const int knot_count = testing::TestNonLocalECPotential::firstComponentKnotCount(potential);
  REQUIRE(knot_count > 1);
  testing::TestNonLocalECPotential::setOuterTileCapacity(
      potential, static_cast<std::size_t>(knot_count + 1));

  // Establish the public neighbor lists consumed by V1 and nontrivial public
  // candidates that a failed private per-electron rebuild must not replace.
  testing::TestNonLocalECPotential::evaluateImpl(
      potential, wavefunction, electrons, true, true);
  const NLPPPublicSnapshot public_before = snapshotNLPPPublicState(
      potential, electrons.getTotalNum(), ions.getTotalNum());
  const std::vector<QMCTraits::PosType> positions_before(electrons.R.begin(), electrons.R.end());

  StdRandom<FullPrecReal> rng(271828u);
  potential.setRandomGenerator(&rng);
  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons});
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(wavefunction, {wavefunction});
  RefVectorWithLeader<OperatorBase> potentials(potential, {potential});
  ResourceCollection particle_resources("flattened_v1_atomic_particles");
  ResourceCollection potential_resources("flattened_v1_atomic_potential");
  electrons.createResource(particle_resources);
  potential.createResource(potential_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
  ResourceCollectionTeamLock<OperatorBase> potential_lock(potential_resources, potentials);

  NonLocalTOperator move_operator(TmoveKind::V1, 0.5, 0.0, 0.0);
  control->flattened_calls        = 0;
  control->change_version_at_call = 2;
  CHECK_THROWS_AS(
      NonLocalECPotential::mw_makeNonLocalMovesPbyP(
          potentials, wavefunctions, particles, move_operator),
      std::runtime_error);
  CHECK(control->flattened_calls == 2);
  REQUIRE(electrons.R.size() == positions_before.size());
  for (std::size_t particle = 0; particle < positions_before.size(); ++particle)
    CHECK(samePositionBits(electrons.R[particle], positions_before[particle]));
  CHECK(sameNLPPPublicState(
      potential, public_before, electrons.getTotalNum(), ions.getTotalNum()));

  control->flattened_calls        = 0;
  control->change_version_at_call = 0;
  CHECK_NOTHROW(NonLocalECPotential::mw_makeNonLocalMovesPbyP(
      potentials, wavefunctions, particles, move_operator));
  CHECK(control->flattened_calls > 2);
}

TEST_CASE("NonLocalECPotential clone owns neighbor-list binding and staged publication",
          "[hamiltonian][ecp][resource]")
{
  static_assert(noexcept(std::declval<NeighborListsForPseudo&>().swapOwnedLists(
      std::declval<NeighborListsForPseudo::OwnedLists&>())));

  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet electrons = makeTmoveV1Elec(
      simulation_cell, ions, {0.4, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-0.4, 0.6, -0.3});
  ParticleSet clone_electrons(electrons);

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options);
  TrialWaveFunction clone_wavefunction(runtime_options);
  NonLocalECPotential potential(ions, electrons, false, true);
  potential.addComponent(0, readTmoveV1PPComponent());
  UPtr<OperatorBase> clone_storage = potential.makeClone(clone_electrons, clone_wavefunction);
  auto& clone = dynamic_cast<NonLocalECPotential&>(*clone_storage);

  CHECK(testing::TestNonLocalECPotential::neighborListsBindOwnComponents(potential));
  CHECK(testing::TestNonLocalECPotential::neighborListsBindOwnComponents(clone));
  CHECK_FALSE(testing::TestNonLocalECPotential::neighborListsBindComponents(clone, potential));

  testing::TestNonLocalECPotential::addNeighborPair(clone, 0, 0);
  testing::TestNonLocalECPotential::addNeighborPair(clone, 2, 1);
  auto staging = testing::TestNonLocalECPotential::makeNeighborListStaging(clone);
  staging.addElecIonPair(1, 1);
  testing::TestNonLocalECPotential::validateNeighborListStaging(clone, staging);
  CHECK(testing::TestNonLocalECPotential::publishNeighborListStaging(clone, staging));

  CHECK(testing::TestNonLocalECPotential::neighboringIons(clone, 0).empty());
  CHECK(testing::TestNonLocalECPotential::neighboringIons(clone, 1) == std::vector<int>{1});
  CHECK(testing::TestNonLocalECPotential::neighboringIons(clone, 2).empty());
  CHECK(testing::TestNonLocalECPotential::neighboringElectrons(clone, 0).empty());
  CHECK(testing::TestNonLocalECPotential::neighboringElectrons(clone, 1) == std::vector<int>{1});

  // The staging object now owns the complete former public state.  No list was
  // copied or rebuilt during the noexcept publication operation.
  CHECK(staging.getNeighboringIons(0) == std::vector<int>{0});
  CHECK(staging.getNeighboringIons(1).empty());
  CHECK(staging.getNeighboringIons(2) == std::vector<int>{1});
  CHECK(staging.getNeighboringElectrons(0) == std::vector<int>{0});
  CHECK(staging.getNeighboringElectrons(1) == std::vector<int>{2});

  std::vector<NonLocalECPComponent*> wrong_components(1, nullptr);
  NeighborListsForPseudo wrong_shape(1, 1, wrong_components);
  auto wrong_staging = wrong_shape.makeOwnedLists();
  wrong_staging.addElecIonPair(0, 0);
  const auto before_electron_one = testing::TestNonLocalECPotential::neighboringIons(clone, 1);
  const auto before_ion_one      = testing::TestNonLocalECPotential::neighboringElectrons(clone, 1);
  CHECK_THROWS_AS(testing::TestNonLocalECPotential::validateNeighborListStaging(clone, wrong_staging),
                  std::invalid_argument);
  CHECK_FALSE(testing::TestNonLocalECPotential::publishNeighborListStaging(clone, wrong_staging));
  CHECK(testing::TestNonLocalECPotential::neighboringIons(clone, 1) == before_electron_one);
  CHECK(testing::TestNonLocalECPotential::neighboringElectrons(clone, 1) == before_ion_one);
}

TEST_CASE("NonLocalECPotential multi-walker resources clone empty and recover from mismatch",
          "[hamiltonian][ecp][resource]")
{
  using FullPrecReal = QMCTraits::FullPrecRealType;
  using ValueType    = QMCTraits::ValueType;

  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet electrons = makeTmoveV1Elec(
      simulation_cell, ions, {0.4, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-0.4, 0.6, -0.3});
  ParticleSet clone_electrons(electrons);

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options);
  TrialWaveFunction clone_wavefunction(runtime_options);
  NonLocalECPotential potential(ions, electrons, false, true);
  testing::TestNonLocalECPotential::setOuterTileCapacity(potential, 5);
  potential.addComponent(0, readTmoveV1PPComponent());
  UPtr<OperatorBase> clone_storage = potential.makeClone(clone_electrons, clone_wavefunction);
  auto& clone = dynamic_cast<NonLocalECPotential&>(*clone_storage);
  electrons.update();
  clone_electrons.update();
  RefVectorWithLeader<OperatorBase> family(potential, {potential, clone});
  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons, clone_electrons});
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(
      wavefunction, {wavefunction, clone_wavefunction});

  FakeRandom<FullPrecReal> grid_rng;
  FakeRandom<FullPrecReal> clone_grid_rng;
  grid_rng.set_value(0.371);
  clone_grid_rng.set_value(0.371);
  potential.setRandomGenerator(&grid_rng);
  clone.setRandomGenerator(&clone_grid_rng);

  ResourceCollection particle_resources("nlpp_resource_ownership_particles");
  electrons.createResource(particle_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);

  OptVariables active;
  active.insert("resource_scratch_guard", 0.0);
  active.resetIndex();
  RecordArray<ValueType> scores(2, 1);
  RecordArray<ValueType> derivatives(2, 1);
  std::fill(scores.begin(), scores.end(), ValueType(0));
  std::fill(derivatives.begin(), derivatives.end(), ValueType(0));

  std::size_t warmed_storage_fingerprint = 0;
  std::size_t warmed_derivative_capacity = 0;
  std::size_t warmed_weight_capacity     = 0;
  std::size_t warmed_logical_jobs        = 0;
  std::size_t warmed_logical_knots       = 0;

  ResourceCollection resources("nlpp_resource_ownership");
  potential.createResource(resources);
  {
    ResourceCollectionTeamLock<OperatorBase> lock(resources, family);
    CHECK(testing::TestNonLocalECPotential::hasMultiWalkerResource(potential));
    potential.mw_evaluateWithParameterDerivatives(
        family, wavefunctions, particles, active, scores, derivatives);
    const auto statistics = testing::TestNonLocalECPotential::derivativeStatistics(potential);
    REQUIRE(statistics.logical_jobs > 0);
    REQUIRE(statistics.logical_knots > 0);
    REQUIRE(statistics.derivative_staging_size == 2);
    warmed_storage_fingerprint = statistics.virtual_batch_storage_fingerprint;
    warmed_derivative_capacity = statistics.derivative_staging_capacity;
    warmed_weight_capacity     = statistics.bounded_weight_capacity;
    warmed_logical_jobs        = statistics.logical_jobs;
    warmed_logical_knots       = statistics.logical_knots;
    testing::TestNonLocalECPotential::resizeListenerScratch(potential, 2, electrons.getTotalNum(),
                                                            ions.getTotalNum());
    CHECK(testing::TestNonLocalECPotential::listenerScratchSizes(potential) ==
          std::make_pair(std::size_t{6}, std::size_t{4}));
  }
  CHECK_FALSE(testing::TestNonLocalECPotential::hasMultiWalkerResource(potential));

  // Copying the collection preserves its schema and family identity but does
  // not copy live listener matrices or any future batch scratch.
  ResourceCollection copied_resources(resources);
  {
    ResourceCollectionTeamLock<OperatorBase> copied_lock(copied_resources, family);
    CHECK(testing::TestNonLocalECPotential::listenerScratchSizes(potential) ==
          std::make_pair(std::size_t{0}, std::size_t{0}));
    const auto copied_statistics =
        testing::TestNonLocalECPotential::derivativeStatistics(potential);
    CHECK(copied_statistics.logical_jobs == 0);
    CHECK(copied_statistics.logical_knots == 0);
    CHECK(copied_statistics.tiles_packed == 0);
    CHECK(copied_statistics.derivative_staging_size == 0);
    CHECK(copied_statistics.virtual_batch_storage_fingerprint !=
          warmed_storage_fingerprint);
    testing::TestNonLocalECPotential::resizeListenerScratch(potential, 1, 1, 1);
  }
  {
    ResourceCollectionTeamLock<OperatorBase> original_lock(resources, family);
    CHECK(testing::TestNonLocalECPotential::listenerScratchSizes(potential) ==
          std::make_pair(std::size_t{6}, std::size_t{4}));
    const auto retained_statistics =
        testing::TestNonLocalECPotential::derivativeStatistics(potential);
    CHECK(retained_statistics.virtual_batch_storage_fingerprint ==
          warmed_storage_fingerprint);
    CHECK(retained_statistics.derivative_staging_capacity == warmed_derivative_capacity);
    CHECK(retained_statistics.bounded_weight_capacity == warmed_weight_capacity);
    CHECK(retained_statistics.logical_jobs == warmed_logical_jobs);
    CHECK(retained_statistics.logical_knots == warmed_logical_knots);
  }

  // A shape-compatible but unrelated potential must not borrow this family's
  // resource.  The failed local-handle validation leaves both leaders empty
  // and rewinds the collection so its rightful owner can immediately acquire.
  NonLocalECPotential unrelated(ions, electrons, false, true);
  RefVectorWithLeader<OperatorBase> unrelated_family(unrelated, {unrelated});
  auto acquire_unrelated = [&]() {
    ResourceCollectionTeamLock<OperatorBase> wrong_lock(resources, unrelated_family);
  };
  CHECK_THROWS_AS(acquire_unrelated(), std::logic_error);
  CHECK_FALSE(testing::TestNonLocalECPotential::hasMultiWalkerResource(unrelated));
  CHECK_FALSE(testing::TestNonLocalECPotential::hasMultiWalkerResource(potential));
  {
    ResourceCollectionTeamLock<OperatorBase> recovered_lock(resources, family);
    CHECK(testing::TestNonLocalECPotential::hasMultiWalkerResource(potential));
  }

  // Mixing clone families is rejected before lending, with the same
  // no-partial-handle and immediate-reacquire guarantees.
  RefVectorWithLeader<OperatorBase> mixed_family(potential, {potential, unrelated});
  auto acquire_mixed = [&]() {
    ResourceCollectionTeamLock<OperatorBase> wrong_lock(resources, mixed_family);
  };
  CHECK_THROWS_AS(acquire_mixed(), std::invalid_argument);
  CHECK_FALSE(testing::TestNonLocalECPotential::hasMultiWalkerResource(potential));
  {
    ResourceCollectionTeamLock<OperatorBase> recovered_lock(resources, family);
    CHECK(testing::TestNonLocalECPotential::hasMultiWalkerResource(potential));
  }

  // A clone retains the family identity, but a different electron shape is an
  // incompatible schema and is rejected before touching the collection.
  ParticleSet short_electrons(simulation_cell);
  short_electrons.setName("short_electrons");
  short_electrons.create({1, 1});
  SpeciesSet& short_species             = short_electrons.getSpeciesSet();
  const int short_up                    = short_species.addSpecies("u");
  const int short_charge                = short_species.addAttribute("charge");
  const int short_mass                  = short_species.addAttribute("mass");
  short_species(short_charge, short_up) = -1;
  short_species(short_mass, short_up)   = 1;
  const int short_down                  = short_species.addSpecies("d");
  short_species(short_charge, short_down) = -1;
  short_species(short_mass, short_down)   = 1;
  short_electrons.resetGroups();
  short_electrons.addTable(ions);
  short_electrons.update();
  TrialWaveFunction short_wavefunction(runtime_options);
  UPtr<OperatorBase> short_clone_storage = potential.makeClone(short_electrons, short_wavefunction);
  auto& short_clone = dynamic_cast<NonLocalECPotential&>(*short_clone_storage);
  RefVectorWithLeader<OperatorBase> mismatched_schema(potential, {potential, short_clone});
  auto acquire_mismatched_schema = [&]() {
    ResourceCollectionTeamLock<OperatorBase> wrong_lock(resources, mismatched_schema);
  };
  CHECK_THROWS_AS(acquire_mismatched_schema(), std::invalid_argument);
  CHECK_FALSE(testing::TestNonLocalECPotential::hasMultiWalkerResource(potential));
  {
    ResourceCollectionTeamLock<OperatorBase> recovered_lock(resources, family);
    CHECK(testing::TestNonLocalECPotential::hasMultiWalkerResource(potential));
  }
}

TEST_CASE("NonLocalECPComponent caller-owned quadrature ranges match legacy",
          "[hamiltonian][ecp][quadrature_range]")
{
  using Real     = QMCTraits::RealType;
  using Value    = QMCTraits::ValueType;
  using Position = QMCTraits::PosType;

  UPtr<NonLocalECPComponent> component = readTmoveV1PPComponent();
  testing::TestNonLocalECPotential::copyComponentGridUnrotatedForTest(*component);

  const Position reference_position{Real(0.4), Real(-0.3), Real(0.2)};
  const Position displacement{Real(0.7), Real(-1.1), Real(0.5)};
  const Real radius = std::sqrt(dot(displacement, displacement));
  const std::size_t knot_count = static_cast<std::size_t>(component->getNknot());
  REQUIRE(knot_count > 2);

  const auto legacy = testing::TestNonLocalECPotential::buildLegacyQuadrature(
      *component, radius, displacement);
  const auto scratch_before_builder =
      testing::TestNonLocalECPotential::snapshotLegacyScratch(*component);

  std::vector<Position> deltas(knot_count);
  std::vector<Position> absolute_positions(knot_count);
  std::vector<Real> bare_weights(knot_count);
  std::vector<Real> radial_scratch(static_cast<std::size_t>(component->getNchannel()));
  std::vector<Real> legendre_scratch(static_cast<std::size_t>(component->getLmax() + 1));
  component->buildQuadraturePointRange(radius, displacement, reference_position, 0, knot_count, 0, deltas,
                                       absolute_positions, bare_weights, radial_scratch, legendre_scratch);

  for (std::size_t knot = 0; knot < knot_count; ++knot)
  {
    CHECK(samePositionBits(deltas[knot], legacy.deltas[knot]));
    CHECK(sameRealBits(bare_weights[knot], legacy.bare_weights[knot]));
    const Position expected_absolute = reference_position + legacy.deltas[knot];
    CHECK(samePositionBits(absolute_positions[knot], expected_absolute));
  }

  // Rebuilding into two ranges must reproduce the same positions and weights,
  // even though each range independently evaluates its radial-channel scratch.
  const std::size_t split = knot_count / 2;
  std::vector<Position> split_deltas(knot_count, Position(Real(9)));
  std::vector<Position> split_positions(knot_count, Position(Real(9)));
  std::vector<Real> split_weights(knot_count, Real(9));
  component->buildQuadraturePointRange(radius, displacement, reference_position, 0, split, 0, split_deltas,
                                       split_positions, split_weights, radial_scratch, legendre_scratch);
  component->buildQuadraturePointRange(radius, displacement, reference_position, split, knot_count - split, split,
                                       split_deltas, split_positions, split_weights, radial_scratch,
                                       legendre_scratch);
  for (std::size_t knot = 0; knot < knot_count; ++knot)
  {
    CHECK(samePositionBits(split_deltas[knot], deltas[knot]));
    CHECK(samePositionBits(split_positions[knot], absolute_positions[knot]));
    CHECK(sameRealBits(split_weights[knot], bare_weights[knot]));
  }

  // A nonzero source-knot offset is independent of the tile-local output
  // offset used by the caller's bounded scratch arrays.
  std::vector<Position> tail_deltas(knot_count - split);
  std::vector<Position> tail_positions(knot_count - split);
  std::vector<Real> tail_weights(knot_count - split);
  component->buildQuadraturePointRange(radius, displacement, reference_position, split, knot_count - split, 0,
                                       tail_deltas, tail_positions, tail_weights, radial_scratch,
                                       legendre_scratch);
  for (std::size_t local_knot = 0; local_knot < knot_count - split; ++local_knot)
  {
    CHECK(samePositionBits(tail_deltas[local_knot], deltas[split + local_knot]));
    CHECK(samePositionBits(tail_positions[local_knot], absolute_positions[split + local_knot]));
    CHECK(sameRealBits(tail_weights[local_knot], bare_weights[split + local_knot]));
  }

  // A zero-length tail is a true no-op and does not require arithmetic scratch.
  std::vector<Real> empty_scratch;
  CHECK_NOTHROW(component->buildQuadraturePointRange(
      radius, displacement, reference_position, knot_count, 0, knot_count, deltas, absolute_positions,
      bare_weights, empty_scratch, empty_scratch));

  // Every extent is checked before radial scratch or an output is changed.
  std::vector<Position> guarded_deltas(knot_count, Position(Real(7)));
  std::vector<Position> guarded_positions(knot_count, Position(Real(8)));
  std::vector<Real> short_weights(knot_count - 1, Real(6));
  const auto guarded_deltas_before     = guarded_deltas;
  const auto guarded_positions_before = guarded_positions;
  const auto radial_before           = radial_scratch;
  const auto legendre_before         = legendre_scratch;
  CHECK_THROWS_AS(component->buildQuadraturePointRange(
                      radius, displacement, reference_position, 0, knot_count, 0, guarded_deltas,
                      guarded_positions, short_weights, radial_scratch, legendre_scratch),
                  std::invalid_argument);
  for (std::size_t knot = 0; knot < knot_count; ++knot)
  {
    CHECK(samePositionBits(guarded_deltas[knot], guarded_deltas_before[knot]));
    CHECK(samePositionBits(guarded_positions[knot], guarded_positions_before[knot]));
  }
  CHECK(radial_scratch == radial_before);
  CHECK(legendre_scratch == legendre_before);
  CHECK(testing::TestNonLocalECPotential::legacyScratchMatches(*component, scratch_before_builder));

  auto make_value = [](Real real_part, Real imaginary_part) -> Value {
#if defined(QMC_COMPLEX)
    return Value(real_part, imaginary_part);
#else
    static_cast<void>(imaginary_part);
    return real_part;
#endif
  };

  // Alternate the sign of bare*real(full ratio), independently of the bare
  // weight sign, so the TMDLA test necessarily exercises both branches.
  std::vector<Value> full_ratios(knot_count);
  std::vector<Value> fermionic_ratios(knot_count);
  const auto zero_knot_iter = std::find_if(
      bare_weights.begin(), bare_weights.end(), [](Real weight) { return weight != Real(0); });
  REQUIRE(zero_knot_iter != bare_weights.end());
  const std::size_t zero_knot = static_cast<std::size_t>(std::distance(bare_weights.begin(), zero_knot_iter));
  int positive_full_knots    = 0;
  int nonpositive_full_knots = 0;
  for (std::size_t knot = 0; knot < knot_count; ++knot)
  {
    const Real magnitude = Real(0.5) + Real(0.03125) * static_cast<Real>(knot);
    const Real bare_sign = bare_weights[knot] < Real(0) ? Real(-1) : Real(1);
    const Real branch_sign = knot % 2 == 0 ? Real(1) : Real(-1);
    const Real full_real = knot == zero_knot ? std::copysign(Real(0), -bare_weights[knot])
                                             : bare_sign * branch_sign * magnitude;
    full_ratios[knot] = make_value(full_real,
                                   Real(0.125) + Real(0.01) * static_cast<Real>(knot));
    fermionic_ratios[knot] = make_value(Real(-0.4) + Real(0.0625) * static_cast<Real>(knot),
                                        Real(-0.25) - Real(0.02) * static_cast<Real>(knot));
    if (bare_weights[knot] * std::real(full_ratios[knot]) > Real(0))
      ++positive_full_knots;
    else
      ++nonpositive_full_knots;
  }
  REQUIRE(positive_full_knots > 0);
  REQUIRE(nonpositive_full_knots > 0);

  auto compare_reduction = [&](bool use_tmdla, bool emit_candidates) {
    const auto reference = testing::TestNonLocalECPotential::reduceLegacyQuadrature(
        *component, 1, legacy, full_ratios, fermionic_ratios, use_tmdla);
    std::vector<Real> transformed(knot_count, Real(11));
    std::vector<NonLocalData> candidates(knot_count);
    Real pair_potential = Real(0);
    const auto scratch_before_reducer =
        testing::TestNonLocalECPotential::snapshotLegacyScratch(*component);
    NonLocalECPComponent::reduceQuadraturePointRange(
        1, 0, knot_count, deltas, bare_weights, full_ratios,
        use_tmdla ? &fermionic_ratios : nullptr, 0, transformed, 0,
        emit_candidates ? &candidates : nullptr, pair_potential);
    CHECK(testing::TestNonLocalECPotential::legacyScratchMatches(*component, scratch_before_reducer));

    CHECK(sameRealBits(pair_potential, reference.pair_potential));
    for (std::size_t knot = 0; knot < knot_count; ++knot)
    {
      CHECK(sameRealBits(transformed[knot], reference.transformed_weights[knot]));
      if (emit_candidates)
      {
        CHECK(candidates[knot].PID == reference.candidates[knot].PID);
        CHECK(sameRealBits(candidates[knot].Weight, reference.candidates[knot].Weight));
        CHECK(samePositionBits(candidates[knot].Delta, reference.candidates[knot].Delta));
      }
    }
    return reference;
  };

  compare_reduction(false, true);
  const auto tmdla_reference = compare_reduction(true, true);
  const Real zero_full = bare_weights[zero_knot] * std::real(full_ratios[zero_knot]);
  REQUIRE(zero_full == Real(0));
  CHECK(sameRealBits(tmdla_reference.transformed_weights[zero_knot], zero_full));
  CHECK(!sameRealBits(tmdla_reference.transformed_weights[zero_knot],
                      bare_weights[zero_knot] * std::real(fermionic_ratios[zero_knot])));

  // DLA without T-move candidates uses its already selected fermionic ratio
  // in the ordinary transform; the reducer itself does no component filtering.
  const auto dla_reference = testing::TestNonLocalECPotential::reduceLegacyQuadrature(
      *component, 1, legacy, fermionic_ratios, full_ratios, false);
  std::vector<Real> dla_weights(knot_count);
  Real dla_energy = Real(0);
  NonLocalECPComponent::reduceQuadraturePointRange(1, 0, knot_count, deltas, bare_weights, fermionic_ratios,
                                                   nullptr, 0, dla_weights, 0, nullptr, dla_energy);
  CHECK(sameRealBits(dla_energy, dla_reference.pair_potential));
  for (std::size_t knot = 0; knot < knot_count; ++knot)
    CHECK(sameRealBits(dla_weights[knot], dla_reference.transformed_weights[knot]));

  // Tile-local transformed storage and globally staged candidate storage have
  // independent offsets.  Keep input distinct as well to pin all three maps.
  constexpr std::size_t input_prefix       = 1;
  constexpr std::size_t transformed_prefix = 2;
  constexpr std::size_t candidate_prefix   = 3;
  std::vector<Position> offset_deltas(input_prefix + knot_count, Position(Real(-5)));
  std::vector<Real> offset_bare(input_prefix + knot_count, Real(-5));
  std::vector<Value> offset_full(input_prefix + knot_count, Value(Real(-5)));
  std::vector<Value> offset_fermionic(input_prefix + knot_count, Value(Real(-5)));
  for (std::size_t knot = 0; knot < knot_count; ++knot)
  {
    offset_deltas[input_prefix + knot]    = deltas[knot];
    offset_bare[input_prefix + knot]      = bare_weights[knot];
    offset_full[input_prefix + knot]      = full_ratios[knot];
    offset_fermionic[input_prefix + knot] = fermionic_ratios[knot];
  }
  std::vector<Real> offset_transformed(transformed_prefix + knot_count, Real(-7));
  std::vector<NonLocalData> offset_candidates(
      candidate_prefix + knot_count, NonLocalData(-7, Real(-7), Position(Real(-7))));
  Real offset_energy = Real(0);
  NonLocalECPComponent::reduceQuadraturePointRange(
      1, input_prefix, knot_count, offset_deltas, offset_bare, offset_full, &offset_fermionic,
      transformed_prefix, offset_transformed, candidate_prefix, &offset_candidates, offset_energy);
  CHECK(sameRealBits(offset_energy, tmdla_reference.pair_potential));
  for (std::size_t offset = 0; offset < transformed_prefix; ++offset)
    CHECK(sameRealBits(offset_transformed[offset], Real(-7)));
  for (std::size_t offset = 0; offset < candidate_prefix; ++offset)
    CHECK(offset_candidates[offset].PID == -7);
  for (std::size_t knot = 0; knot < knot_count; ++knot)
  {
    CHECK(sameRealBits(offset_transformed[transformed_prefix + knot],
                       tmdla_reference.transformed_weights[knot]));
    CHECK(offset_candidates[candidate_prefix + knot].PID == tmdla_reference.candidates[knot].PID);
    CHECK(sameRealBits(offset_candidates[candidate_prefix + knot].Weight,
                       tmdla_reference.candidates[knot].Weight));
    CHECK(samePositionBits(offset_candidates[candidate_prefix + knot].Delta,
                           tmdla_reference.candidates[knot].Delta));
  }

  // Range calls add one knot at a time to the supplied accumulator.  Therefore
  // splitting inside a job retains both floating-point association and the
  // exact candidate scan order.
  constexpr Real initial_pair_potential = Real(0.3125);
  std::vector<Real> one_range_weights(knot_count);
  std::vector<Real> split_range_weights(knot_count);
  std::vector<NonLocalData> one_range_candidates(knot_count);
  std::vector<NonLocalData> split_range_candidates(knot_count);
  Real one_range_energy   = initial_pair_potential;
  Real split_range_energy = initial_pair_potential;
  NonLocalECPComponent::reduceQuadraturePointRange(
      1, 0, knot_count, deltas, bare_weights, full_ratios, &fermionic_ratios, 0, one_range_weights,
      0, &one_range_candidates, one_range_energy);
  NonLocalECPComponent::reduceQuadraturePointRange(
      1, 0, split, deltas, bare_weights, full_ratios, &fermionic_ratios, 0, split_range_weights,
      0, &split_range_candidates, split_range_energy);
  NonLocalECPComponent::reduceQuadraturePointRange(
      1, split, knot_count - split, deltas, bare_weights, full_ratios, &fermionic_ratios, split,
      split_range_weights, split, &split_range_candidates, split_range_energy);

  CHECK(sameRealBits(split_range_energy, one_range_energy));
  for (std::size_t knot = 0; knot < knot_count; ++knot)
  {
    CHECK(sameRealBits(split_range_weights[knot], one_range_weights[knot]));
    CHECK(split_range_candidates[knot].PID == one_range_candidates[knot].PID);
    CHECK(sameRealBits(split_range_candidates[knot].Weight, one_range_candidates[knot].Weight));
    CHECK(samePositionBits(split_range_candidates[knot].Delta, one_range_candidates[knot].Delta));
  }

  // Reducer validation precedes accumulator or output publication.
  std::vector<Real> guarded_weights(knot_count, Real(13));
  std::vector<NonLocalData> short_candidates(knot_count - 1);
  Real guarded_energy = initial_pair_potential;
  CHECK_THROWS_AS(NonLocalECPComponent::reduceQuadraturePointRange(
                      1, 0, knot_count, deltas, bare_weights, full_ratios, &fermionic_ratios, 0,
                      guarded_weights, 0, &short_candidates, guarded_energy),
                  std::invalid_argument);
  CHECK(guarded_weights == std::vector<Real>(knot_count, Real(13)));
  CHECK(sameRealBits(guarded_energy, initial_pair_potential));

  std::vector<Position> empty_deltas;
  std::vector<Real> empty_reals;
  std::vector<Value> empty_values;
  Real empty_energy = initial_pair_potential;
  CHECK_NOTHROW(NonLocalECPComponent::reduceQuadraturePointRange(
      1, 0, 0, empty_deltas, empty_reals, empty_values, nullptr, 0, empty_reals, 0, nullptr, empty_energy));
  CHECK(sameRealBits(empty_energy, initial_pair_potential));
}

// Electron-1 positions and RNG seeds shared by DLA and v1 T-move regressions.
const QMCTraits::PosType kTmoveV1Walker1Elec1Pos{1.0, 0.0, 0.0};
const QMCTraits::PosType kTmoveV1Walker2Elec1Pos{0.8, 0.3, 0.1};
constexpr unsigned kTmoveV1Walker1Seed = 10101u;
constexpr unsigned kTmoveV1Walker2Seed = 10201u;
constexpr unsigned kTmoveV1Walker3Seed = 10301u;

struct DlaEvaluationResult
{
  std::array<double, 2> energies;
  std::array<std::size_t, 2> candidate_counts;
};

/** Evaluate two walkers through the determinant-locality path, optionally
 * requesting the T-move data that activates TMDLA's split ratio evaluation. */
DlaEvaluationResult runDlaEvaluation(bool batched, bool compute_tmove_data)
{
  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet electrons = makeTmoveV1Elec(
      simulation_cell, ions, {0.4, 0.0, 0.0}, kTmoveV1Walker1Elec1Pos, {-0.4, 0.6, -0.3});
  ParticleSet electrons2(electrons);
  electrons2.R[1] = kTmoveV1Walker2Elec1Pos;
  electrons2.update();

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options);
  TrialWaveFunction wavefunction2(runtime_options);
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(
      wavefunction, {wavefunction, wavefunction2});
  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons, electrons2});

  NonLocalECPotential potential(ions, electrons, true /* enable_DLA */, true /* use_VP */);
  potential.addComponent(0, readTmoveV1PPComponent());
  UPtr<OperatorBase> potential2_ptr = potential.makeClone(electrons2, wavefunction2);
  auto& potential2 = dynamic_cast<NonLocalECPotential&>(*potential2_ptr);
  RefVectorWithLeader<OperatorBase> potentials(potential, {potential, potential2});

  ResourceCollection particle_resources("dla_particles");
  ResourceCollection potential_resources("dla_potentials");
  electrons.createResource(particle_resources);
  potential.createResource(potential_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
  ResourceCollectionTeamLock<OperatorBase> potential_lock(potential_resources, potentials);

  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(potential);
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(potential2);
  if (batched)
    testing::TestNonLocalECPotential::mw_evaluateImpl(
        potential, potentials, wavefunctions, particles, compute_tmove_data, std::nullopt, true);
  else
  {
    testing::TestNonLocalECPotential::evaluateImpl(
        potential, wavefunction, electrons, compute_tmove_data, true);
    testing::TestNonLocalECPotential::evaluateImpl(
        potential2, wavefunction2, electrons2, compute_tmove_data, true);
  }

  return {{potential.getValue(), potential2.getValue()},
          {testing::TestNonLocalECPotential::tmoveCandidateCount(potential),
           testing::TestNonLocalECPotential::tmoveCandidateCount(potential2)}};
}

struct TmoveV1Result
{
  std::vector<int> accepts;
  std::vector<QMCTraits::PosType> final_R;
  std::vector<QMCTraits::FullPrecRealType> next_rng_values;
};

/** run a two-walker v1 T-move sweep with fixed quadrature grids and
 *  per-walker seeded RNGs; batched and serial paths must agree walker by
 *  walker in accepted counts and final electron positions.
 */
TmoveV1Result runTmoveV1(bool batched, bool use_VP, bool use_DLA = false)
{
  using FullPrecReal = QMCTraits::FullPrecRealType;

  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet elec =
      makeTmoveV1Elec(simulation_cell, ions, {0.4, 0.0, 0.0}, kTmoveV1Walker1Elec1Pos, {-0.4, 0.6, -0.3});

  ParticleSet elec2(elec);
  elec2.R[1] = kTmoveV1Walker2Elec1Pos;
  elec2.update();

  RefVectorWithLeader<ParticleSet> p_list(elec, {elec, elec2});
  RuntimeOptions runtime_options;
  TrialWaveFunction psi(runtime_options);
  TrialWaveFunction psi2(runtime_options);
  RefVectorWithLeader<TrialWaveFunction> twf_list(psi, {psi, psi2});

  NonLocalECPotential nl_ecp(ions, elec, use_DLA, use_VP);
  nl_ecp.addComponent(0, readTmoveV1PPComponent());
  if (use_VP)
    testing::TestNonLocalECPotential::setOuterTileCapacity(nl_ecp, 3);
  UPtr<OperatorBase> nl_ecp2_ptr = nl_ecp.makeClone(elec2, psi2);
  auto& nl_ecp2                  = dynamic_cast<NonLocalECPotential&>(*nl_ecp2_ptr);

  StdRandom<FullPrecReal> rng(kTmoveV1Walker1Seed);
  StdRandom<FullPrecReal> rng2(kTmoveV1Walker2Seed);
  nl_ecp.setRandomGenerator(&rng);
  nl_ecp2.setRandomGenerator(&rng2);

  RefVectorWithLeader<OperatorBase> o_list(nl_ecp, {nl_ecp, nl_ecp2});
  ResourceCollection pset_res("test_pset_res");
  elec.createResource(pset_res);
  ResourceCollectionTeamLock<ParticleSet> pset_lock(pset_res, p_list);
  ResourceCollection nl_ecp_res("test_nl_ecp_res");
  nl_ecp.createResource(nl_ecp_res);
  ResourceCollectionTeamLock<OperatorBase> nl_ecp_lock(nl_ecp_res, o_list);

  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(nl_ecp);
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(nl_ecp2);

  // the energy evaluation builds the neighbor lists the v1 sweep reads
  testing::TestNonLocalECPotential::mw_evaluateImpl(nl_ecp, o_list, twf_list, p_list, false, std::nullopt, true);

  NonLocalTOperator move_op(TmoveKind::V1, 0.5 /*tau*/, 0.0 /*alpha*/, 0.0 /*gamma*/);

  TmoveV1Result res;
  if (batched)
    res.accepts = NonLocalECPotential::mw_makeNonLocalMovesPbyP(o_list, twf_list, p_list, move_op);
  else
  {
    res.accepts.resize(2);
    res.accepts[0] = nl_ecp.makeNonLocalMovesPbyP(psi, elec, move_op);
    res.accepts[1] = nl_ecp2.makeNonLocalMovesPbyP(psi2, elec2, move_op);
  }
  for (int iat = 0; iat < elec.getTotalNum(); ++iat)
    res.final_R.push_back(elec.R[iat]);
  for (int iat = 0; iat < elec2.getTotalNum(); ++iat)
    res.final_R.push_back(elec2.R[iat]);
  res.next_rng_values = {rng(), rng2()};
  return res;
}
} // namespace

TEST_CASE("NonLocalECPotential Tmove v1 batched matches serial", "[hamiltonian]")
{
  for (const bool use_VP : {false, true})
  {
    const auto serial  = runTmoveV1(false, use_VP);
    const auto batched = runTmoveV1(true, use_VP);

    REQUIRE(serial.accepts.size() == batched.accepts.size());
    CHECK(serial.accepts == batched.accepts);
    // a sweep with no accepted move would leave the accept path untested
    const int total_accepts = std::accumulate(serial.accepts.begin(), serial.accepts.end(), 0);
    CHECK(total_accepts > 0);

    REQUIRE(serial.final_R.size() == batched.final_R.size());
    for (size_t i = 0; i < serial.final_R.size(); ++i)
      CHECK(samePositionBits(serial.final_R[i], batched.final_R[i]));
    CHECK(serial.next_rng_values == batched.next_rng_values);
  }
}

TEST_CASE("NonLocalECPotential flattened TMDLA V1 seeded sweep matches scalar bits",
          "[hamiltonian][ecp][nlpp_v1_flattened]")
{
  const auto scalar    = runTmoveV1(false, true, true);
  const auto flattened = runTmoveV1(true, true, true);

  CHECK(flattened.accepts == scalar.accepts);
  REQUIRE(flattened.final_R.size() == scalar.final_R.size());
  for (std::size_t particle = 0; particle < scalar.final_R.size(); ++particle)
    CHECK(samePositionBits(flattened.final_R[particle], scalar.final_R[particle]));
  CHECK(flattened.next_rng_values == scalar.next_rng_values);
  CHECK(std::accumulate(flattened.accepts.begin(), flattened.accepts.end(), 0) > 0);
}

namespace
{
struct TmoveFallbackResult
{
  std::vector<int> accepts;
  std::vector<QMCTraits::PosType> final_R;
  std::vector<QMCTraits::FullPrecRealType> next_rng_values;
  int scalar_ratio_calls;
  int flattened_ratio_calls;
};

/** Run the unchanged V0/V3 per-walker fallback after one flattened VP energy evaluation. */
TmoveFallbackResult runTmoveFallback(TmoveKind move_kind, bool batched)
{
  using FullPrecReal = QMCTraits::FullPrecRealType;

  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet electrons = makeTmoveV1Elec(
      simulation_cell, ions, {0.4, 0.0, 0.0}, kTmoveV1Walker1Elec1Pos, {-0.4, 0.6, -0.3});
  ParticleSet electrons2(electrons);
  electrons2.R[1] = kTmoveV1Walker2Elec1Pos;
  electrons2.update();

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options);
  TrialWaveFunction wavefunction2(runtime_options);
  auto control = std::make_shared<StampedRatioControl>();
  wavefunction.addComponent(std::make_unique<StampedRatioOrbital>(control, true));
  wavefunction2.addComponent(std::make_unique<StampedRatioOrbital>(control, true));
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(
      wavefunction, {wavefunction, wavefunction2});
  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons, electrons2});

  NonLocalECPotential potential(ions, electrons, false, true);
  potential.addComponent(0, readTmoveV1PPComponent());
  testing::TestNonLocalECPotential::setOuterTileCapacity(potential, 3);
  UPtr<OperatorBase> potential2_storage = potential.makeClone(electrons2, wavefunction2);
  auto& potential2 = dynamic_cast<NonLocalECPotential&>(*potential2_storage);
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(potential);
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(potential2);

  StdRandom<FullPrecReal> rng(kTmoveV1Walker1Seed);
  StdRandom<FullPrecReal> rng2(kTmoveV1Walker2Seed);
  potential.setRandomGenerator(&rng);
  potential2.setRandomGenerator(&rng2);
  RefVectorWithLeader<OperatorBase> potentials(potential, {potential, potential2});

  ResourceCollection particle_resources("tmove_fallback_particles");
  ResourceCollection potential_resources("tmove_fallback_potentials");
  electrons.createResource(particle_resources);
  potential.createResource(potential_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
  ResourceCollectionTeamLock<OperatorBase> potential_lock(potential_resources, potentials);

  testing::TestNonLocalECPotential::mw_evaluateImpl(
      potential, potentials, wavefunctions, particles, true, std::nullopt, true);
  control->scalar_calls    = 0;
  control->flattened_calls = 0;

  // A large step makes the fixed-seed V3 fixture accept an early move and
  // therefore exercise its affected-electron scalar candidate recomputation.
  NonLocalTOperator move_operator(move_kind, 10.0, 0.0, 0.0);
  TmoveFallbackResult result;
  if (batched)
    result.accepts = NonLocalECPotential::mw_makeNonLocalMovesPbyP(
        potentials, wavefunctions, particles, move_operator);
  else
  {
    result.accepts.resize(2);
    result.accepts[0] = potential.makeNonLocalMovesPbyP(wavefunction, electrons, move_operator);
    result.accepts[1] = potential2.makeNonLocalMovesPbyP(wavefunction2, electrons2, move_operator);
  }
  for (const auto& position : electrons.R)
    result.final_R.push_back(position);
  for (const auto& position : electrons2.R)
    result.final_R.push_back(position);
  result.next_rng_values       = {rng(), rng2()};
  result.scalar_ratio_calls    = control->scalar_calls;
  result.flattened_ratio_calls = control->flattened_calls;
  return result;
}
} // namespace

TEST_CASE("NonLocalECPotential V0 and V3 remain exact scalar fallbacks",
          "[hamiltonian][ecp][nlpp_v1_flattened][fallback]")
{
  for (const TmoveKind move_kind : {TmoveKind::V0, TmoveKind::V3})
  {
    const TmoveFallbackResult scalar  = runTmoveFallback(move_kind, false);
    const TmoveFallbackResult batched = runTmoveFallback(move_kind, true);

    CHECK(batched.accepts == scalar.accepts);
    REQUIRE(batched.final_R.size() == scalar.final_R.size());
    for (std::size_t particle = 0; particle < scalar.final_R.size(); ++particle)
      CHECK(samePositionBits(batched.final_R[particle], scalar.final_R[particle]));
    CHECK(batched.next_rng_values == scalar.next_rng_values);
    CHECK(batched.scalar_ratio_calls == scalar.scalar_ratio_calls);
    CHECK(batched.flattened_ratio_calls == 0);
    CHECK(scalar.flattened_ratio_calls == 0);

    if (move_kind == TmoveKind::V0)
      CHECK(batched.scalar_ratio_calls == 0);
    else
    {
      CHECK(std::accumulate(batched.accepts.begin(), batched.accepts.end(), 0) > 0);
      CHECK(batched.scalar_ratio_calls > 0);
    }
  }
}

TEST_CASE("NonLocalECPotential batched DLA and TMDLA match scalar", "[hamiltonian]")
{
  for (const bool compute_tmove_data : {false, true})
  {
    const DlaEvaluationResult scalar  = runDlaEvaluation(false, compute_tmove_data);
    const DlaEvaluationResult batched = runDlaEvaluation(true, compute_tmove_data);
    for (int walker = 0; walker < 2; ++walker)
    {
      CHECK(batched.energies[walker] == Approx(scalar.energies[walker]).epsilon(1e-12));
      CHECK(batched.candidate_counts[walker] == scalar.candidate_counts[walker]);
      if (compute_tmove_data)
        CHECK(batched.candidate_counts[walker] > 0);
      else
        CHECK(batched.candidate_counts[walker] == 0);
    }
  }
}

namespace
{
/** run a three-walker v1 T-move sweep where the third walker's electron 2
 *  sits 2.0 from ion 0 (inside the 3.475 cutoff read from Na.BFD.xml) and
 *  4.0 from ion 1 (outside it), while walkers 1 and 2 keep electron 2 close
 *  to both ions. So walker 3's neighbor-ion list for electron 2 has one
 *  fewer entry than the other two walkers', jel_jobs[iw].size() genuinely
 *  differs across walkers within one batch, and the ragged-index guard
 *  `if (jobid < jel_jobs[iw].size())` in mw_makeNonLocalMovesPbyP is taken.
 *  Walkers 1 and 2 reuse the two-walker case's fixture and seeds unchanged.
 */
TmoveV1Result runTmoveV1Ragged(bool batched, bool use_VP)
{
  using FullPrecReal = QMCTraits::FullPrecRealType;

  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet elec =
      makeTmoveV1Elec(simulation_cell, ions, {0.4, 0.0, 0.0}, kTmoveV1Walker1Elec1Pos, {-0.4, 0.6, -0.3});

  ParticleSet elec2(elec);
  elec2.R[1] = kTmoveV1Walker2Elec1Pos;
  elec2.update();

  ParticleSet elec3(elec);
  elec3.R[2] = {0.0, 3.0, 0.0};
  elec3.update();

  RefVectorWithLeader<ParticleSet> p_list(elec, {elec, elec2, elec3});
  RuntimeOptions runtime_options;
  TrialWaveFunction psi(runtime_options);
  TrialWaveFunction psi2(runtime_options);
  TrialWaveFunction psi3(runtime_options);
  RefVectorWithLeader<TrialWaveFunction> twf_list(psi, {psi, psi2, psi3});

  NonLocalECPotential nl_ecp(ions, elec, false /*enable_DLA*/, use_VP);
  nl_ecp.addComponent(0, readTmoveV1PPComponent());
  if (use_VP)
    testing::TestNonLocalECPotential::setOuterTileCapacity(nl_ecp, 3);
  UPtr<OperatorBase> nl_ecp2_ptr = nl_ecp.makeClone(elec2, psi2);
  auto& nl_ecp2                  = dynamic_cast<NonLocalECPotential&>(*nl_ecp2_ptr);
  UPtr<OperatorBase> nl_ecp3_ptr = nl_ecp.makeClone(elec3, psi3);
  auto& nl_ecp3                  = dynamic_cast<NonLocalECPotential&>(*nl_ecp3_ptr);

  StdRandom<FullPrecReal> rng(kTmoveV1Walker1Seed);
  StdRandom<FullPrecReal> rng2(kTmoveV1Walker2Seed);
  StdRandom<FullPrecReal> rng3(kTmoveV1Walker3Seed);
  nl_ecp.setRandomGenerator(&rng);
  nl_ecp2.setRandomGenerator(&rng2);
  nl_ecp3.setRandomGenerator(&rng3);

  RefVectorWithLeader<OperatorBase> o_list(nl_ecp, {nl_ecp, nl_ecp2, nl_ecp3});
  ResourceCollection pset_res("test_pset_res");
  elec.createResource(pset_res);
  ResourceCollectionTeamLock<ParticleSet> pset_lock(pset_res, p_list);
  ResourceCollection nl_ecp_res("test_nl_ecp_res");
  nl_ecp.createResource(nl_ecp_res);
  ResourceCollectionTeamLock<OperatorBase> nl_ecp_lock(nl_ecp_res, o_list);

  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(nl_ecp);
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(nl_ecp2);
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(nl_ecp3);

  // the energy evaluation builds the neighbor lists the v1 sweep reads
  testing::TestNonLocalECPotential::mw_evaluateImpl(nl_ecp, o_list, twf_list, p_list, false, std::nullopt, true);

  // pin the raggedness itself: if a future change to the fixture or to the
  // pseudopotential cutoff makes these three counts equal, this must fail
  // rather than silently stop exercising jobid < jel_jobs[iw].size().
  REQUIRE(testing::TestNonLocalECPotential::numNeighboringIons(nl_ecp, 2) == 2);
  REQUIRE(testing::TestNonLocalECPotential::numNeighboringIons(nl_ecp2, 2) == 2);
  REQUIRE(testing::TestNonLocalECPotential::numNeighboringIons(nl_ecp3, 2) == 1);

  NonLocalTOperator move_op(TmoveKind::V1, 0.5 /*tau*/, 0.0 /*alpha*/, 0.0 /*gamma*/);

  TmoveV1Result res;
  if (batched)
    res.accepts = NonLocalECPotential::mw_makeNonLocalMovesPbyP(o_list, twf_list, p_list, move_op);
  else
  {
    res.accepts.resize(3);
    res.accepts[0] = nl_ecp.makeNonLocalMovesPbyP(psi, elec, move_op);
    res.accepts[1] = nl_ecp2.makeNonLocalMovesPbyP(psi2, elec2, move_op);
    res.accepts[2] = nl_ecp3.makeNonLocalMovesPbyP(psi3, elec3, move_op);
  }
  for (int iat = 0; iat < elec.getTotalNum(); ++iat)
    res.final_R.push_back(elec.R[iat]);
  for (int iat = 0; iat < elec2.getTotalNum(); ++iat)
    res.final_R.push_back(elec2.R[iat]);
  for (int iat = 0; iat < elec3.getTotalNum(); ++iat)
    res.final_R.push_back(elec3.R[iat]);
  res.next_rng_values = {rng(), rng2(), rng3()};
  return res;
}
} // namespace

TEST_CASE("NonLocalECPotential Tmove v1 batched matches serial, ragged candidate counts", "[hamiltonian]")
{
  for (const bool use_VP : {false, true})
  {
    const auto serial  = runTmoveV1Ragged(false, use_VP);
    const auto batched = runTmoveV1Ragged(true, use_VP);

    REQUIRE(serial.accepts.size() == batched.accepts.size());
    CHECK(serial.accepts == batched.accepts);
    // a sweep with no accepted move would leave the accept path untested
    const int total_accepts = std::accumulate(serial.accepts.begin(), serial.accepts.end(), 0);
    CHECK(total_accepts > 0);

    REQUIRE(serial.final_R.size() == batched.final_R.size());
    for (size_t i = 0; i < serial.final_R.size(); ++i)
      CHECK(samePositionBits(serial.final_R[i], batched.final_R[i]));
    CHECK(serial.next_rng_values == batched.next_rng_values);
  }
}

namespace
{
/** run a single-walker (nw == 1) v1 T-move sweep through the batched entry
 *  point and check it against the existing serial path. elec1_pos and seed
 *  select which of the two-walker case's walkers this instance reproduces
 *  running alone. Per-walker RNG draw order and count do not depend on how
 *  many other walkers share a batch (see the T-move determinism finding in
 *  the audit), so accepts and final positions here must match that
 *  walker's contribution to the two-walker case exactly.
 */
TmoveV1Result runTmoveV1SingleWalker(bool batched, bool use_VP, const QMCTraits::PosType& elec1_pos, unsigned rng_seed)
{
  using FullPrecReal = QMCTraits::FullPrecRealType;

  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet elec = makeTmoveV1Elec(simulation_cell, ions, {0.4, 0.0, 0.0}, elec1_pos, {-0.4, 0.6, -0.3});

  RefVectorWithLeader<ParticleSet> p_list(elec, {elec});
  RuntimeOptions runtime_options;
  TrialWaveFunction psi(runtime_options);
  RefVectorWithLeader<TrialWaveFunction> twf_list(psi, {psi});

  NonLocalECPotential nl_ecp(ions, elec, false /*enable_DLA*/, use_VP);
  nl_ecp.addComponent(0, readTmoveV1PPComponent());
  if (use_VP)
    testing::TestNonLocalECPotential::setOuterTileCapacity(nl_ecp, 3);

  StdRandom<FullPrecReal> rng(rng_seed);
  nl_ecp.setRandomGenerator(&rng);

  RefVectorWithLeader<OperatorBase> o_list(nl_ecp, {nl_ecp});
  ResourceCollection pset_res("test_pset_res");
  elec.createResource(pset_res);
  ResourceCollectionTeamLock<ParticleSet> pset_lock(pset_res, p_list);
  ResourceCollection nl_ecp_res("test_nl_ecp_res");
  nl_ecp.createResource(nl_ecp_res);
  ResourceCollectionTeamLock<OperatorBase> nl_ecp_lock(nl_ecp_res, o_list);

  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(nl_ecp);

  // the energy evaluation builds the neighbor lists the v1 sweep reads
  testing::TestNonLocalECPotential::mw_evaluateImpl(nl_ecp, o_list, twf_list, p_list, false, std::nullopt, true);

  NonLocalTOperator move_op(TmoveKind::V1, 0.5 /*tau*/, 0.0 /*alpha*/, 0.0 /*gamma*/);

  TmoveV1Result res;
  if (batched)
    res.accepts = NonLocalECPotential::mw_makeNonLocalMovesPbyP(o_list, twf_list, p_list, move_op);
  else
  {
    res.accepts.resize(1);
    res.accepts[0] = nl_ecp.makeNonLocalMovesPbyP(psi, elec, move_op);
  }
  for (int iat = 0; iat < elec.getTotalNum(); ++iat)
    res.final_R.push_back(elec.R[iat]);
  res.next_rng_values = {rng()};
  return res;
}
} // namespace

TEST_CASE("NonLocalECPotential Tmove v1 batched matches serial, single walker", "[hamiltonian]")
{
  // reruns each of the two-walker case's walkers alone through nw == 1.
  // Because per-walker RNG draw order and count do not depend on batch
  // size, the union of accepts across both reproduces the two-walker
  // case's own non-zero total for the same reason, rather than by chance.
  const std::vector<std::pair<QMCTraits::PosType, unsigned>> single_walker_cases =
      {{kTmoveV1Walker1Elec1Pos, kTmoveV1Walker1Seed},  // walker 1 of the two-walker case, alone
       {kTmoveV1Walker2Elec1Pos, kTmoveV1Walker2Seed}}; // walker 2 of the two-walker case, alone

  for (const bool use_VP : {false, true})
  {
    int total_accepts = 0;
    for (const auto& [elec1_pos, seed] : single_walker_cases)
    {
      const auto serial  = runTmoveV1SingleWalker(false, use_VP, elec1_pos, seed);
      const auto batched = runTmoveV1SingleWalker(true, use_VP, elec1_pos, seed);

      REQUIRE(serial.accepts.size() == batched.accepts.size());
      CHECK(serial.accepts == batched.accepts);
      total_accepts += std::accumulate(serial.accepts.begin(), serial.accepts.end(), 0);

      REQUIRE(serial.final_R.size() == batched.final_R.size());
      for (size_t i = 0; i < serial.final_R.size(); ++i)
        CHECK(samePositionBits(serial.final_R[i], batched.final_R[i]));
      CHECK(serial.next_rng_values == batched.next_rng_values);
    }
    // a sweep with no accepted move anywhere would leave the accept path untested
    CHECK(total_accepts > 0);
  }
}

TEST_CASE("NonLocalECPotential mw_evaluate ragged listener scatter", "[hamiltonian]")
{
  using Real         = QMCTraits::RealType;
  using FullPrecReal = QMCTraits::FullPrecRealType;

  Lattice lattice;
  lattice.BoxBConds = true; // periodic
  lattice.R.diagonal(20.0);
  lattice.LR_dim_cutoff = 15;
  lattice.reset();

  const SimulationCell simulation_cell(lattice);

  ParticleSet ions(simulation_cell);

  ions.setName("ion");
  ions.create({2});
  ions.R[0] = {0.0, 1.0, 0.0};
  ions.R[1] = {0.0, -1.0, 0.0};

  SpeciesSet& ion_species                         = ions.getSpeciesSet();
  int index_species                               = ion_species.addSpecies("Na");
  int index_charge                                = ion_species.addAttribute("charge");
  int index_atomic_number                         = ion_species.addAttribute("atomic_number");
  ion_species(index_charge, index_species)        = 1;
  ion_species(index_atomic_number, index_species) = 1;
  ions.createSK();
  ions.resetGroups();
  ions.update();

  ParticleSet elec(simulation_cell);
  elec.setName("elec");
  elec.create({2, 1});
  elec.R[0] = {0.4, 0.0, 0.0};
  elec.R[1] = {1.0, 0.0, 0.0};
  elec.R[2] = {0.0, 0.0, 0.0};

  SpeciesSet& tspecies       = elec.getSpeciesSet();
  int upIdx                  = tspecies.addSpecies("u");
  int chargeIdx              = tspecies.addAttribute("charge");
  int massIdx                = tspecies.addAttribute("mass");
  tspecies(chargeIdx, upIdx) = -1;
  tspecies(massIdx, upIdx)   = 1.0;

  int dnIdx                  = tspecies.addSpecies("d");
  chargeIdx                  = tspecies.addAttribute("charge");
  massIdx                    = tspecies.addAttribute("mass");
  tspecies(chargeIdx, dnIdx) = -1;
  tspecies(massIdx, dnIdx)   = 1.0;

  elec.createSK();
  elec.resetGroups();
  elec.addTable(ions);
  elec.update();

  ParticleSet elec2(elec);
  elec2.update();

  // The sparse walkers' lone down electron sits 2.0 from ion 0 and 4.0 from
  // ion 1; the Na.BFD.xml cutoff lies between, so these walkers carry one
  // fewer job than the full walkers in this species group.
  ParticleSet elec3(elec);
  elec3.R[2] = {0.0, 3.0, 0.0};
  elec3.update();
  ParticleSet elec4(elec3);
  elec4.update();

  // At the second down-electron job, only crowd walkers 1 and 3 remain.  This
  // creates both a leading and a middle hole in the compact batch.
  RefVectorWithLeader<ParticleSet> p_list(elec3, {elec3, elec, elec4, elec2});

  RuntimeOptions runtime_options;
  TrialWaveFunction psi(runtime_options);
  TrialWaveFunction psi2(runtime_options);
  TrialWaveFunction psi3(runtime_options);
  TrialWaveFunction psi4(runtime_options);
  RefVectorWithLeader<TrialWaveFunction> twf_list(psi3, {psi3, psi, psi4, psi2});

  NonLocalECPotential nl_ecp(ions, elec, false /*use_DLA*/, false /*use_VP*/);

  Communicate* comm = OHMMS::Controller;
  ECPComponentBuilder ecp_comp_builder("test_read_ecp", comm, 4, 1);

  bool okay = ecp_comp_builder.read_pp_file("Na.BFD.xml");
  REQUIRE(okay);
  UPtr<NonLocalECPComponent> nl_ecp_comp = std::move(ecp_comp_builder.pp_nonloc);
  nl_ecp.addComponent(0, std::move(nl_ecp_comp));
  UPtr<OperatorBase> nl_ecp2_ptr = nl_ecp.makeClone(elec2, psi2);
  auto& nl_ecp2                  = dynamic_cast<NonLocalECPotential&>(*nl_ecp2_ptr);
  UPtr<OperatorBase> nl_ecp3_ptr = nl_ecp.makeClone(elec3, psi3);
  auto& nl_ecp3                  = dynamic_cast<NonLocalECPotential&>(*nl_ecp3_ptr);
  UPtr<OperatorBase> nl_ecp4_ptr = nl_ecp.makeClone(elec4, psi4);
  auto& nl_ecp4                  = dynamic_cast<NonLocalECPotential&>(*nl_ecp4_ptr);

  StdRandom<FullPrecReal> rng(10101);
  StdRandom<FullPrecReal> rng2(10201);
  StdRandom<FullPrecReal> rng3(10301);
  StdRandom<FullPrecReal> rng4(10401);
  nl_ecp.setRandomGenerator(&rng);
  nl_ecp2.setRandomGenerator(&rng2);
  nl_ecp3.setRandomGenerator(&rng3);
  nl_ecp4.setRandomGenerator(&rng4);

  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(nl_ecp);
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(nl_ecp2);
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(nl_ecp3);
  testing::TestNonLocalECPotential::copyGridUnrotatedForTest(nl_ecp4);

  std::vector<testing::TestNonLocalECPotential::ListenerRows> scalar_rows;
  scalar_rows.reserve(4);
  scalar_rows.push_back(testing::TestNonLocalECPotential::evaluateScalarListenerRows(nl_ecp3, psi3, elec3));
  scalar_rows.push_back(testing::TestNonLocalECPotential::evaluateScalarListenerRows(nl_ecp, psi, elec));
  scalar_rows.push_back(testing::TestNonLocalECPotential::evaluateScalarListenerRows(nl_ecp4, psi4, elec4));
  scalar_rows.push_back(testing::TestNonLocalECPotential::evaluateScalarListenerRows(nl_ecp2, psi2, elec2));

  RefVectorWithLeader<OperatorBase> o_list(nl_ecp3, {nl_ecp3, nl_ecp, nl_ecp4, nl_ecp2});
  ResourceCollection pset_res("test_pset_res");
  elec3.createResource(pset_res);
  ResourceCollectionTeamLock<ParticleSet> pset_lock(pset_res, p_list);
  ResourceCollection nl_ecp_res("test_nl_ecp_res");
  nl_ecp3.createResource(nl_ecp_res);
  ResourceCollectionTeamLock<OperatorBase> nl_ecp_lock(nl_ecp_res, o_list);

  Matrix<Real> electron_samples(4, elec.getTotalNum());
  Matrix<Real> ion_samples(4, ions.getTotalNum());
  std::vector<ListenerVector<Real>> electron_listeners;
  electron_listeners.emplace_back("nonlocalpotential", testing::getParticularListener(electron_samples));
  std::vector<ListenerVector<Real>> ion_listeners;
  ion_listeners.emplace_back("nonlocalpotential", testing::getParticularListener(ion_samples));
  ListenerOption<Real> listener_option{electron_listeners, ion_listeners};

  testing::TestNonLocalECPotential::mw_evaluateImpl(nl_ecp3, o_list, twf_list, p_list, false, listener_option, true);

  for (size_t walker = 0; walker < scalar_rows.size(); ++walker)
  {
    for (size_t electron = 0; electron < scalar_rows[walker].electron.size(); ++electron)
      CHECK(electron_samples(walker, electron) == Approx(scalar_rows[walker].electron[electron]));
    for (size_t ion = 0; ion < scalar_rows[walker].ion.size(); ++ion)
      CHECK(ion_samples(walker, ion) == Approx(scalar_rows[walker].ion[ion]));
  }

  CHECK(testing::TestNonLocalECPotential::numNeighboringIons(nl_ecp3, 2) == 1);
  CHECK(testing::TestNonLocalECPotential::numNeighboringIons(nl_ecp, 2) == 2);
  CHECK(testing::TestNonLocalECPotential::numNeighboringIons(nl_ecp4, 2) == 1);
  CHECK(testing::TestNonLocalECPotential::numNeighboringIons(nl_ecp2, 2) == 2);

  const auto mw_value  = nl_ecp.getValue();
  const auto mw_value2 = nl_ecp2.getValue();
  const auto mw_value3 = nl_ecp3.getValue();
  const auto mw_value4 = nl_ecp4.getValue();

  // The two full and two sparse walkers agree within their shapes, while the
  // shapes differ.  A batch value bleeding across slots cannot satisfy these.
  CHECK(mw_value == Approx(mw_value2));
  CHECK(mw_value3 == Approx(mw_value4));
  CHECK(mw_value3 != Approx(mw_value));

  testing::TestNonLocalECPotential::evaluateImpl(nl_ecp, psi, elec, false, true);
  CHECK(nl_ecp.getValue() == Approx(mw_value));
  testing::TestNonLocalECPotential::evaluateImpl(nl_ecp2, psi2, elec2, false, true);
  CHECK(nl_ecp2.getValue() == Approx(mw_value2));
  testing::TestNonLocalECPotential::evaluateImpl(nl_ecp3, psi3, elec3, false, true);
  CHECK(nl_ecp3.getValue() == Approx(mw_value3));
  testing::TestNonLocalECPotential::evaluateImpl(nl_ecp4, psi4, elec4, false, true);
  CHECK(nl_ecp4.getValue() == Approx(mw_value4));

  // the T-move candidate column flows through the same compacted lists;
  // collecting it must not disturb the values
  testing::TestNonLocalECPotential::mw_evaluateImpl(nl_ecp3, o_list, twf_list, p_list, true, std::nullopt, true);
  CHECK(nl_ecp.getValue() == Approx(mw_value));
  CHECK(nl_ecp2.getValue() == Approx(mw_value2));
  CHECK(nl_ecp3.getValue() == Approx(mw_value3));
  CHECK(nl_ecp4.getValue() == Approx(mw_value4));
}

TEST_CASE("NonLocalECPotential batched weighted parameter-derivative path", "[hamiltonian]")
{
  using FullPrecReal = QMCTraits::FullPrecRealType;
  using ValueType    = QMCTraits::ValueType;

  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet electrons =
      makeTmoveV1Elec(simulation_cell, ions, {0.4, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
  ParticleSet electrons2(electrons);
  ParticleSet electrons3(electrons);
  electrons2.R[1] = {0.8, 0.3, 0.1};
  electrons2.update();
  // This location gives the final spin group fewer in-cutoff ion jobs and
  // exercises compaction below the three-walker batch size.
  electrons3.R[2] = {0.0, 3.0, 0.0};
  electrons3.update();

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options);
  TrialWaveFunction wavefunction2(runtime_options);
  TrialWaveFunction wavefunction3(runtime_options);
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(
      wavefunction, {wavefunction, wavefunction2, wavefunction3});
  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons, electrons2, electrons3});

  NonLocalECPotential potential(ions, electrons, false /* enable_DLA */, true /* use_VP */);
  testing::TestNonLocalECPotential::setOuterTileCapacity(potential, 3);
  potential.addComponent(0, readTmoveV1PPComponent());
  UPtr<OperatorBase> potential2_ptr = potential.makeClone(electrons2, wavefunction2);
  UPtr<OperatorBase> potential3_ptr = potential.makeClone(electrons3, wavefunction3);
  auto& potential2 = dynamic_cast<NonLocalECPotential&>(*potential2_ptr);
  auto& potential3 = dynamic_cast<NonLocalECPotential&>(*potential3_ptr);
  RefVectorWithLeader<OperatorBase> potentials(potential, {potential, potential2, potential3});

  StdRandom<FullPrecReal> rng1(19101);
  StdRandom<FullPrecReal> rng2(19201);
  StdRandom<FullPrecReal> rng3(19301);
  potential.setRandomGenerator(&rng1);
  potential2.setRandomGenerator(&rng2);
  potential3.setRandomGenerator(&rng3);

  ResourceCollection particle_resources("weighted_derivative_particles");
  ResourceCollection potential_resources("weighted_derivative_potentials");
  electrons.createResource(particle_resources);
  potential.createResource(potential_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
  ResourceCollectionTeamLock<OperatorBase> potential_lock(potential_resources, potentials);

  OptVariables active;
  active.insert("unused_guard_parameter", 0.0);
  active.resetIndex();
  RecordArray<ValueType> scores(3, 1);
  RecordArray<ValueType> derivatives(3, 1);
  std::fill(scores.begin(), scores.end(), ValueType(0));
  std::fill(derivatives.begin(), derivatives.end(), ValueType(2.5));

  potential.mw_evaluateWithParameterDerivatives(potentials, wavefunctions, particles, active, scores, derivatives);
  const std::array<double, 3> batched_energies{
      potential.getValue(), potential2.getValue(), potential3.getValue()};
  CHECK(testing::TestNonLocalECPotential::derivativeMatrixElements(potential) == 0);
  CHECK(testing::TestNonLocalECPotential::derivativeMatrixElements(potential2) == 0);
  CHECK(testing::TestNonLocalECPotential::derivativeMatrixElements(potential3) == 0);
  for (int walker = 0; walker < 3; ++walker)
    CHECK(derivatives[walker][0] == ValueApprox(ValueType(2.5)));

  // Reset each independent random stream so scalar evaluation uses precisely
  // the same randomly rotated grid as its corresponding batch slot.
  StdRandom<FullPrecReal> scalar_rng1(19101);
  StdRandom<FullPrecReal> scalar_rng2(19201);
  StdRandom<FullPrecReal> scalar_rng3(19301);
  potential.setRandomGenerator(&scalar_rng1);
  potential2.setRandomGenerator(&scalar_rng2);
  potential3.setRandomGenerator(&scalar_rng3);
  std::array<Vector<ValueType>, 3> scalar_derivatives{Vector<ValueType>(1), Vector<ValueType>(1),
                                                       Vector<ValueType>(1)};
  for (auto& derivative : scalar_derivatives)
    derivative = ValueType(2.5);
  Vector<ValueType> score(1);
  score = ValueType(0);

  const std::array<double, 3> scalar_energies{
      potential.evaluateValueAndDerivatives(wavefunction, electrons, active, score, scalar_derivatives[0]),
      potential2.evaluateValueAndDerivatives(wavefunction2, electrons2, active, score, scalar_derivatives[1]),
      potential3.evaluateValueAndDerivatives(wavefunction3, electrons3, active, score, scalar_derivatives[2])};
  for (int walker = 0; walker < 3; ++walker)
  {
    CHECK(batched_energies[walker] == Approx(scalar_energies[walker]).epsilon(1e-12));
    CHECK(scalar_derivatives[walker][0] == ValueApprox(ValueType(2.5)));
  }

  // The Hamiltonian preflight must honor the largest sparse global mapping,
  // not merely the number of selected variables. Padding destinations remain
  // untouched because this empty trial wavefunction has no parameter score.
  OptVariables global;
  global.insert("padding_0", 0.0);
  global.insert("guard_p0", 0.0);
  global.insert("padding_1", 0.0);
  global.insert("padding_2", 0.0);
  global.insert("guard_p2", 0.0);
  global.resetIndex();
  OptVariables sparse;
  sparse.insert("guard_p0", 0.0);
  sparse.insert("guard_p2", 0.0);
  sparse.getIndex(global);
  REQUIRE(sparse.where(0) == 1);
  REQUIRE(sparse.where(1) == 4);

  RecordArray<ValueType> short_scores(3, 2);
  RecordArray<ValueType> short_derivatives(3, 2);
  std::fill(short_scores.begin(), short_scores.end(), ValueType(-3));
  std::fill(short_derivatives.begin(), short_derivatives.end(), ValueType(7));
  CHECK_THROWS_AS(potential.mw_evaluateWithParameterDerivatives(
                      potentials, wavefunctions, particles, sparse, short_scores, short_derivatives),
                  std::invalid_argument);
  for (const ValueType& derivative : short_derivatives)
    CHECK(derivative == ValueApprox(ValueType(7)));

  RecordArray<ValueType> padded_scores(3, 5);
  RecordArray<ValueType> padded_derivatives(3, 5);
  std::fill(padded_scores.begin(), padded_scores.end(), ValueType(-3));
  std::fill(padded_derivatives.begin(), padded_derivatives.end(), ValueType(7));
  CHECK_NOTHROW(potential.mw_evaluateWithParameterDerivatives(
      potentials, wavefunctions, particles, sparse, padded_scores, padded_derivatives));
  for (const ValueType& derivative : padded_derivatives)
    CHECK(derivative == ValueApprox(ValueType(7)));

  // Zero-width derivative rows remain a valid energy-only transaction.
  OptVariables inactive;
  inactive.resetIndex();
  RecordArray<ValueType> empty_scores(3, 0);
  RecordArray<ValueType> empty_derivatives(3, 0);
  CHECK_NOTHROW(potential.mw_evaluateWithParameterDerivatives(
      potentials, wavefunctions, particles, inactive, empty_scores, empty_derivatives));
  const auto empty_stats = testing::TestNonLocalECPotential::derivativeStatistics(potential);
  CHECK(empty_stats.derivative_staging_size == 0);
  CHECK(empty_stats.max_tile_occupancy <= 3);
}

TEST_CASE("NonLocalECPotential mixed ion quadratures preserve scalar derivatives",
          "[hamiltonian][ecp][multiwalker][derivatives][mixed_species]")
{
  using FullPrecReal = QMCTraits::FullPrecRealType;
  using ValueType    = QMCTraits::ValueType;

  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions(simulation_cell);
  ions.setName("mixed_rule_ions");
  ions.create({1, 1});
  ions.R[0] = {0.0, 1.0, 0.0};
  ions.R[1] = {0.0, -1.0, 0.0};
  SpeciesSet& ion_species                  = ions.getSpeciesSet();
  const int rule_four                      = ion_species.addSpecies("Na_rule4");
  const int rule_six                       = ion_species.addSpecies("Na_rule6");
  const int ion_charge                     = ion_species.addAttribute("charge");
  const int ion_atomic_number              = ion_species.addAttribute("atomic_number");
  ion_species(ion_charge, rule_four)        = 1;
  ion_species(ion_charge, rule_six)         = 1;
  ion_species(ion_atomic_number, rule_four) = 11;
  ion_species(ion_atomic_number, rule_six)  = 11;
  ions.createSK();
  ions.resetGroups();
  ions.update();

  ParticleSet electrons = makeTmoveV1Elec(
      simulation_cell, ions, {0.35, 0.0, 0.0}, {0.9, 0.15, 0.0}, {-0.3, -0.2, 0.1});
  ParticleSet electrons2(electrons);
  electrons2.R[1] += QMCTraits::PosType{-0.08, 0.04, 0.03};
  electrons2.update();

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "mixed_rule_wavefunction");
  auto control                            = std::make_shared<StampedRatioControl>();
  control->ratio                          = makeStampedRatio(0.83, 0.19);
  control->derivative_increment           = makeStampedRatio(0.03125, 0.015625);
  control->derivative_uses_total_weights = true;
  wavefunction.addComponent(std::make_unique<StampedRatioOrbital>(control, true));
  std::unique_ptr<TrialWaveFunction> wavefunction2 = wavefunction.makeClone(electrons2);

  constexpr std::size_t outer_tile_capacity = 7;
  NonLocalECPotential potential(ions, electrons, false /* enable_DLA */, true /* use_VP */);
  testing::TestNonLocalECPotential::setOuterTileCapacity(potential, outer_tile_capacity);
  potential.addComponent(rule_four, readTmoveV1PPComponent(4));
  potential.addComponent(rule_six, readTmoveV1PPComponent(6));
  REQUIRE(testing::TestNonLocalECPotential::componentKnotCountForIon(potential, 0) == 12);
  REQUIRE(testing::TestNonLocalECPotential::componentKnotCountForIon(potential, 1) == 26);

  UPtr<OperatorBase> potential2_storage = potential.makeClone(electrons2, *wavefunction2);
  auto& potential2                       = dynamic_cast<NonLocalECPotential&>(*potential2_storage);
  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons, electrons2});
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(
      wavefunction, {wavefunction, *wavefunction2});
  RefVectorWithLeader<OperatorBase> potentials(potential, {potential, potential2});

  FakeRandom<FullPrecReal> grid_rng;
  FakeRandom<FullPrecReal> grid_rng2;
  grid_rng.set_value(0.371);
  grid_rng2.set_value(0.371);
  potential.setRandomGenerator(&grid_rng);
  potential2.setRandomGenerator(&grid_rng2);

  ResourceCollection particle_resources("mixed_rule_derivative_particles");
  ResourceCollection potential_resources("mixed_rule_derivative_potentials");
  electrons.createResource(particle_resources);
  potential.createResource(potential_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
  ResourceCollectionTeamLock<OperatorBase> potential_lock(potential_resources, potentials);

  OptVariables active;
  active.insert("mixed_rule_guard", 0.0);
  active.resetIndex();
  constexpr ValueType derivative_sentinel = ValueType(1.75);
  RecordArray<ValueType> scores(2, 1);
  RecordArray<ValueType> derivatives(2, 1);
  std::fill(scores.begin(), scores.end(), ValueType(0));
  std::fill(derivatives.begin(), derivatives.end(), derivative_sentinel);

  potential.mw_evaluateWithParameterDerivatives(
      potentials, wavefunctions, particles, active, scores, derivatives);
  const std::array<double, 2> batched_energies{potential.getValue(), potential2.getValue()};
  const std::array<ValueType, 2> batched_derivatives{derivatives[0][0], derivatives[1][0]};
  const auto statistics = testing::TestNonLocalECPotential::derivativeStatistics(potential);
  CHECK(statistics.logical_jobs == 12);
  CHECK(statistics.logical_knots == 228);
  CHECK(statistics.tiles_packed > 1);
  CHECK(statistics.split_job_continuations > 0);
  CHECK(statistics.tail_tiles > 0);
  CHECK(statistics.max_tile_occupancy <= outer_tile_capacity);

  // Re-run each walker through the public scalar operator with the same fixed
  // rotations. This checks unequal job lengths in both the energy reduction
  // and the weighted parameter destination, not just descriptor construction.
  FakeRandom<FullPrecReal> scalar_rng;
  FakeRandom<FullPrecReal> scalar_rng2;
  scalar_rng.set_value(0.371);
  scalar_rng2.set_value(0.371);
  potential.setRandomGenerator(&scalar_rng);
  potential2.setRandomGenerator(&scalar_rng2);
  Vector<ValueType> scalar_score(1);
  scalar_score = ValueType(0);
  std::array<Vector<ValueType>, 2> scalar_derivatives{Vector<ValueType>(1), Vector<ValueType>(1)};
  scalar_derivatives[0] = derivative_sentinel;
  scalar_derivatives[1] = derivative_sentinel;
  const std::array<double, 2> scalar_energies{
      potential.evaluateValueAndDerivatives(
          wavefunction, electrons, active, scalar_score, scalar_derivatives[0]),
      potential2.evaluateValueAndDerivatives(
          *wavefunction2, electrons2, active, scalar_score, scalar_derivatives[1])};

  for (int walker = 0; walker < 2; ++walker)
  {
    CHECK(batched_energies[walker] == Approx(scalar_energies[walker]).epsilon(1e-12));
    CHECK(batched_derivatives[walker] == ValueApprox(scalar_derivatives[walker][0]));
    CHECK(std::abs(batched_derivatives[walker] - derivative_sentinel) > 1e-8);
  }
}

TEST_CASE("NonLocalECPotential empty derivative workload",
          "[hamiltonian][ecp][multiwalker][derivatives]")
{
  using FullPrecReal = QMCTraits::FullPrecRealType;
  using ValueType    = QMCTraits::ValueType;

  const SimulationCell simulation_cell = makeTmoveV1SimulationCell();
  ParticleSet ions                     = makeTmoveV1Ions(simulation_cell);
  ParticleSet electrons = makeTmoveV1Elec(
      simulation_cell, ions, {7.0, 7.0, 7.0}, {6.0, 7.0, 7.0}, {7.0, 6.0, 7.0});

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options);
  NonLocalECPotential potential(ions, electrons, false /* enable_DLA */, true /* use_VP */);
  testing::TestNonLocalECPotential::setOuterTileCapacity(potential, 5);
  potential.addComponent(0, readTmoveV1PPComponent());

  FakeRandom<FullPrecReal> grid_rng;
  grid_rng.set_value(0.371);
  potential.setRandomGenerator(&grid_rng);

  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons});
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(wavefunction, {wavefunction});
  RefVectorWithLeader<OperatorBase> potentials(potential, {potential});
  ResourceCollection particle_resources("empty_derivative_particles");
  ResourceCollection potential_resources("empty_derivative_potential");
  electrons.createResource(particle_resources);
  potential.createResource(potential_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
  ResourceCollectionTeamLock<OperatorBase> potential_lock(potential_resources, potentials);

  OptVariables active;
  active.insert("empty_workload_guard", 0.0);
  active.resetIndex();
  RecordArray<ValueType> scores(1, 1);
  RecordArray<ValueType> derivatives(1, 1);
  scores[0][0]      = ValueType(0);
  derivatives[0][0] = ValueType(2.75);

  CHECK_NOTHROW(potential.mw_evaluateWithParameterDerivatives(
      potentials, wavefunctions, particles, active, scores, derivatives));
  CHECK(potential.getValue() == Approx(0.0));
  CHECK(derivatives[0][0] == ValueApprox(ValueType(2.75)));
  for (const auto& group_jobs : testing::TestNonLocalECPotential::jobs(potential))
    CHECK(group_jobs.empty());

  const auto statistics = testing::TestNonLocalECPotential::derivativeStatistics(potential);
  CHECK(statistics.logical_jobs == 0);
  CHECK(statistics.logical_knots == 0);
  CHECK(statistics.tiles_packed == 0);
  CHECK(statistics.tail_tiles == 0);
  CHECK(statistics.max_tile_occupancy == 0);
  CHECK(statistics.derivative_staging_size == 1);
  CHECK(statistics.bounded_weight_size == 0);
  CHECK(testing::TestNonLocalECPotential::derivativeMatrixElements(potential) == 0);
}

TEST_CASE("PsiFormer NonLocalECPotential multiwalker parameter derivatives",
          "[hamiltonian][psiformer][ecp][multiwalker]")
{
  using FullPrecReal = QMCTraits::FullPrecRealType;
  using ValueType    = QMCTraits::ValueType;
  using namespace testing::psiformer;

  constexpr int walker_count              = 2;
  constexpr ValueType derivative_sentinel = ValueType(0.375);

  GeneratedFiles files    = generateFiles("lih_pp");
  const Geometry geometry = makeGeometry("lih_pp");
  const SimulationCell cell;

  ParticleSet ions(cell);
  ions.setName("ion0");
  ions.create({2});
  SpeciesSet& ion_species           = ions.getSpeciesSet();
  const int sodium                  = ion_species.addSpecies("Na");
  const int ion_charge              = ion_species.addAttribute("charge");
  const int atomic_number           = ion_species.addAttribute("atomic_number");
  ion_species(ion_charge, sodium)    = 1.0;
  ion_species(atomic_number, sodium) = 11.0;
  for (int nucleus = 0; nucleus < ions.getTotalNum(); ++nucleus)
    for (int dimension = 0; dimension < 3; ++dimension)
      ions.R[nucleus][dimension] = geometry.nuclei[3 * nucleus + dimension];
  ions.resetGroups();
  ions.update();

  ParticleSet electrons(cell);
  electrons.setName("e");
  electrons.create({1, 1});
  SpeciesSet& electron_species           = electrons.getSpeciesSet();
  const int up                           = electron_species.addSpecies("u");
  const int down                         = electron_species.addSpecies("d");
  const int electron_charge              = electron_species.addAttribute("charge");
  const int electron_mass                = electron_species.addAttribute("mass");
  electron_species(electron_charge, up)   = -1.0;
  electron_species(electron_charge, down) = -1.0;
  electron_species(electron_mass, up)     = 1.0;
  electron_species(electron_mass, down)   = 1.0;
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      electrons.R[electron][dimension] = geometry.electrons[3 * electron + dimension];
  electrons.resetGroups();
  electrons.update();

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction(runtime_options, "psiformer_nlpp_e2e");
  auto psiformer = std::make_unique<PsiFormerWF>(
      "pf_e2e", files.parameters.string(), files.configuration.string(), true, std::vector<std::size_t>{0, 127});
  psiformer->validateSystem(electrons, ions, "pseudopotential");
  wavefunction.addComponent(std::move(psiformer));

  // Keep the active vector PsiFormer-only while forcing all NLPP ratios and
  // derivative weights to contain a nontrivial second wavefunction factor.
  const char* jastrow_xml = R"(<tmp>
    <jastrow name="J2_fixed_e2e" type="Two-Body" function="Bspline" print="no" gpu="no">
      <correlation speciesA="u" speciesB="d" rcut="5" size="8">
        <coefficients id="fixed_ud_e2e" type="Array" optimize="no">
          0.42 -0.31 0.24 -0.16 0.11 -0.07 0.035 -0.012
        </coefficients>
      </correlation>
    </jastrow>
  </tmp>)";
  Libxml2Document jastrow_document;
  REQUIRE(jastrow_document.parseFromString(jastrow_xml));
  RadialJastrowBuilder jastrow_builder(OHMMS::Controller, electrons);
  auto jastrow_component = jastrow_builder.buildComponent(xmlFirstElementChild(jastrow_document.getRoot()));
  WaveFunctionComponent* fixed_jastrow = jastrow_component.get();
  wavefunction.addComponent(std::move(jastrow_component));
  electrons.update();

  OptVariables active;
  wavefunction.checkInVariables(active);
  active.resetIndex();
  wavefunction.checkOutVariables(active);
  REQUIRE(active.size() == 2);
  CHECK(active.name(0) == "pf_e2e_pf_0000000");
  CHECK(active.name(1) == "pf_e2e_pf_0000127");

  ParticleSet electrons2(electrons);
  electrons2.R[0] += QMCTraits::PosType{0.11, -0.04, 0.03};
  electrons2.update();
  std::unique_ptr<TrialWaveFunction> wavefunction2 = wavefunction.makeClone(electrons2);

  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons, electrons2});
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(wavefunction, {wavefunction, *wavefunction2});

  const int parameter_count = active.size_of_active();
  RecordArray<ValueType> scores(walker_count, parameter_count);
  auto prepare_references_and_scores = [&]() {
    for (int walker = 0; walker < walker_count; ++walker)
    {
      particles[walker].G = ValueType(0);
      particles[walker].L = ValueType(0);
      wavefunctions[walker].evaluateLog(particles[walker]);
      Vector<ValueType> score_view(scores[walker], parameter_count);
      score_view = ValueType(0);
      wavefunctions[walker].evaluateDerivativesWF(particles[walker], active, score_view);
    }
  };
  prepare_references_and_scores();

  VirtualParticleSet jastrow_probe(electrons);
  const std::vector<QMCTraits::PosType> probe_displacements{{0.17, -0.08, 0.05}};
  jastrow_probe.makeMoves(electrons, 0, probe_displacements);
  std::vector<ValueType> jastrow_ratios(1);
  fixed_jastrow->evaluateRatios(jastrow_probe, jastrow_ratios);
  CHECK(std::abs(jastrow_ratios[0] - ValueType(1)) > 1e-5);

  NonLocalECPotential potential(ions, electrons, false /* enable_DLA */, true /* use_VP */);
  constexpr std::size_t outer_tile_capacity = 5;
  testing::TestNonLocalECPotential::setOuterTileCapacity(potential, outer_tile_capacity);
  ECPComponentBuilder ecp_builder("psiformer_nlpp_e2e", OHMMS::Controller);
  REQUIRE(ecp_builder.read_pp_file("Na.BFD.xml"));
  REQUIRE(ecp_builder.pp_nonloc != nullptr);
  potential.addComponent(0, std::move(ecp_builder.pp_nonloc));
  UPtr<OperatorBase> potential2_storage = potential.makeClone(electrons2, *wavefunction2);
  auto& potential2 = dynamic_cast<NonLocalECPotential&>(*potential2_storage);
  electrons.update();
  electrons2.update();
  RefVectorWithLeader<OperatorBase> potentials(potential, {potential, potential2});

  // A constant generator makes every scalar, multiwalker, and +/- parameter
  // evaluation use exactly the same non-degenerate rotated quadrature grid.
  FakeRandom<FullPrecReal> grid_rng1;
  FakeRandom<FullPrecReal> grid_rng2;
  grid_rng1.set_value(0.371);
  grid_rng2.set_value(0.371);
  potential.setRandomGenerator(&grid_rng1);
  potential2.setRandomGenerator(&grid_rng2);

  ResourceCollection particle_resources("psiformer_nlpp_e2e_particles");
  ResourceCollection wavefunction_resources("psiformer_nlpp_e2e_wavefunctions");
  ResourceCollection potential_resources("psiformer_nlpp_e2e_potentials");
  electrons.createResource(particle_resources);
  wavefunction.createResource(wavefunction_resources);
  potential.createResource(potential_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources, particles);
  ResourceCollectionTeamLock<TrialWaveFunction> wavefunction_lock(wavefunction_resources, wavefunctions);
  ResourceCollectionTeamLock<OperatorBase> potential_lock(potential_resources, potentials);

  RecordArray<ValueType> batch_derivatives(walker_count, parameter_count);
  auto run_batch = [&](RecordArray<ValueType>& output, ValueType initial_value) {
    prepare_references_and_scores();
    std::fill(output.begin(), output.end(), initial_value);
    potential.mw_evaluateWithParameterDerivatives(potentials, wavefunctions, particles, active, scores, output);
    return std::array<double, walker_count>{potential.getValue(), potential2.getValue()};
  };

  const auto batch_energies = run_batch(batch_derivatives, derivative_sentinel);
  const auto derivative_stats =
      testing::TestNonLocalECPotential::derivativeStatistics(potential);
  CHECK(derivative_stats.logical_jobs > 0);
  CHECK(derivative_stats.logical_knots > outer_tile_capacity);
  CHECK(derivative_stats.tiles_packed > 1);
  CHECK(derivative_stats.split_job_continuations > 0);
  CHECK(derivative_stats.tail_tiles > 0);
  CHECK(derivative_stats.logical_knots % outer_tile_capacity != 0);
  CHECK(derivative_stats.max_tile_occupancy <= outer_tile_capacity);
  CHECK(derivative_stats.derivative_staging_size ==
        static_cast<std::size_t>(walker_count * parameter_count));
  CHECK(derivative_stats.bounded_weight_size <= outer_tile_capacity);
  CHECK(derivative_stats.bounded_weight_capacity >= outer_tile_capacity);
  std::array<std::array<ValueType, 2>, walker_count> analytic_derivatives;
  for (int walker = 0; walker < walker_count; ++walker)
    for (int parameter = 0; parameter < parameter_count; ++parameter)
      analytic_derivatives[walker][parameter] = batch_derivatives[walker][parameter];

  // The first request primes both sides of the staged/public job-list swap.
  // Two subsequent same-shape requests must then retain the bounded virtual-
  // batch allocation identity and derivative/weight staging capacities.
  RecordArray<ValueType> warmed_derivatives(walker_count, parameter_count);
  const auto warmed_energies = run_batch(warmed_derivatives, derivative_sentinel);
  const auto warmed_stats = testing::TestNonLocalECPotential::derivativeStatistics(potential);
  RecordArray<ValueType> rewarmed_derivatives(walker_count, parameter_count);
  const auto rewarmed_energies = run_batch(rewarmed_derivatives, derivative_sentinel);
  const auto rewarmed_stats = testing::TestNonLocalECPotential::derivativeStatistics(potential);
  CHECK(rewarmed_stats.virtual_batch_storage_fingerprint ==
        warmed_stats.virtual_batch_storage_fingerprint);
  CHECK(rewarmed_stats.derivative_staging_capacity == warmed_stats.derivative_staging_capacity);
  CHECK(rewarmed_stats.bounded_weight_capacity == warmed_stats.bounded_weight_capacity);
  CHECK(warmed_stats.logical_jobs == derivative_stats.logical_jobs);
  CHECK(warmed_stats.logical_knots == derivative_stats.logical_knots);
  for (int walker = 0; walker < walker_count; ++walker)
  {
    CHECK(warmed_energies[walker] == Approx(batch_energies[walker]).epsilon(1e-13));
    CHECK(rewarmed_energies[walker] == Approx(batch_energies[walker]).epsilon(1e-13));
    for (int parameter = 0; parameter < parameter_count; ++parameter)
    {
      CHECK(warmed_derivatives[walker][parameter] ==
            ValueApprox(analytic_derivatives[walker][parameter]));
      CHECK(rewarmed_derivatives[walker][parameter] ==
            ValueApprox(analytic_derivatives[walker][parameter]));
    }
  }

  CHECK(testing::TestNonLocalECPotential::derivativeMatrixElements(potential) == 0);
  CHECK(testing::TestNonLocalECPotential::derivativeMatrixElements(potential2) == 0);

  // Compare against the public scalar operator path with the same references,
  // accumulation sentinel, wavefunctions, and exactly identical grids.
  prepare_references_and_scores();
  std::array<Vector<ValueType>, walker_count> scalar_derivatives{
      Vector<ValueType>(parameter_count), Vector<ValueType>(parameter_count)};
  std::array<double, walker_count> scalar_energies;
  for (int walker = 0; walker < walker_count; ++walker)
  {
    scalar_derivatives[walker] = derivative_sentinel;
    const Vector<ValueType> score_view(scores[walker], parameter_count);
    scalar_energies[walker] = potentials[walker].evaluateValueAndDerivatives(
        wavefunctions[walker], particles[walker], active, score_view, scalar_derivatives[walker]);
    CHECK(batch_energies[walker] == Approx(scalar_energies[walker]).epsilon(2e-11).margin(2e-11));
    for (int parameter = 0; parameter < parameter_count; ++parameter)
      CHECK(analytic_derivatives[walker][parameter] == ValueApprox(scalar_derivatives[walker][parameter]));
    CHECK(std::abs(analytic_derivatives[walker][0] - derivative_sentinel) > 1e-8);
  }

  // Differentiate the same high-level multiwalker energy, not a component or
  // pre-reduced surrogate. Both walkers validate independent output rows.
  const double original_parameter = std::real(active[0]);
  const double parameter_step     = 2e-5;
  RecordArray<ValueType> finite_difference_scratch(walker_count, parameter_count);

  active[0] = original_parameter + parameter_step;
  wavefunction.resetParameters(active);
  const auto plus_energies = run_batch(finite_difference_scratch, ValueType(0));

  active[0] = original_parameter - parameter_step;
  wavefunction.resetParameters(active);
  const auto minus_energies = run_batch(finite_difference_scratch, ValueType(0));

  active[0] = original_parameter;
  wavefunction.resetParameters(active);
  for (int walker = 0; walker < walker_count; ++walker)
  {
    const double finite_difference =
        (plus_energies[walker] - minus_energies[walker]) / (2 * parameter_step);
    const double analytic = std::real(analytic_derivatives[walker][0] - derivative_sentinel);
    CHECK(psiformer::determinant::isFiniteReal(finite_difference));
    CHECK(analytic == Catch::Approx(finite_difference).epsilon(5e-4).margin(5e-4));
  }
}
} // namespace qmcplusplus
