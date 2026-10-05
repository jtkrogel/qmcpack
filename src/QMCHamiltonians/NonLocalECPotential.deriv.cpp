//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include "QMCHamiltonians/NonLocalECPComponent.h"
#include "QMCHamiltonians/NonLocalECPotential.h"
#include "DistanceTable.h"
#include "CPU/BLAS.hpp"
#include "Utilities/Timer.h"

namespace qmcplusplus
{
NonLocalECPotential::Return_t NonLocalECPotential::evaluateValueAndDerivatives(TrialWaveFunction& psi,
                                                                               ParticleSet& P,
                                                                               const OptVariables& optvars,
                                                                               const Vector<ValueType>& dlogpsi,
                                                                               Vector<ValueType>& dhpsioverpsi)
{
  value_ = 0.0;
  for (int ipp = 0; ipp < PPset.size(); ipp++)
    if (PPset[ipp])
      PPset[ipp]->rotateQuadratureGrid(generateRandomRotationMatrix(*myRNG));

  /* evaluating TWF ratio values requires calling prepareGroup
   * In evaluate() we first loop over species and call prepareGroup before looping over all the electrons of a species
   * Here it is not necessary because TWF::evaluateLog has been called and precomputed data is up-to-date
   */
  const auto& myTable = P.getDistTableAB(myTableIndex);
  for (int jel = 0; jel < P.getTotalNum(); jel++)
  {
    const auto& dist  = myTable.getDistRow(jel);
    const auto& displ = myTable.getDisplRow(jel);
    for (int iat = 0; iat < PP.size(); iat++)
      if (PP[iat] != nullptr && dist[iat] < PP[iat]->getRmax())
        value_ +=
            PP[iat]->evaluateValueAndDerivatives(P, vp_ ? makeOptionalRef<VirtualParticleSet>(*vp_) : std::nullopt,

                                                 iat, psi, jel, dist[iat], -displ[iat], optvars, dlogpsi, dhpsioverpsi);
  }
  return value_;
}

/** evaluate the non-local potential of the iat-th ionic center
   * @param W electron configuration
   * @param iat ionic index
   * @param psi trial wavefunction
   * @param optvars optimizables 
   * @param dlogpsi derivatives of the wavefunction at W.R 
   * @param hdpsioverpsi derivatives of Vpp 
   * @param return the non-local component
   *
   * This is a temporary solution which uses TrialWaveFunction::evaluateDerivatives
   * assuming that the distance tables are fully updated for each ratio computation.
   */
NonLocalECPComponent::RealType NonLocalECPComponent::evaluateValueAndDerivatives(
    ParticleSet& W,
    const OptionalRef<VirtualParticleSet> vp,
    int iat,
    TrialWaveFunction& psi,
    int iel,
    RealType r,
    const PosType& dr,
    const OptVariables& optvars,
    const Vector<ValueType>& dlogpsi,
    Vector<ValueType>& dhpsioverpsi)
{
  const size_t num_vars = optvars.size_of_active();

  buildQuadraturePointDeltaPosAndPartialPotential(r, dr, deltaV_, knot_pots_);

  if (vp)
  {
    VirtualParticleSet& vp_set(*vp);
    // The partial potential is the bare quadrature weight. TrialWaveFunction
    // multiplies it by the complete product ratio before reducing component
    // log-ratio derivatives into the caller's existing output row.
    vp_set.makeMoves(W, iel, deltaV_, true, iat);
    for (int knot = 0; knot < nknot; ++knot)
      wvec[knot] = ValueType(knot_pots_[knot]);
    psi.evaluateDerivRatiosWeighted(vp_set, optvars, wvec, psiratio,
                                    {dhpsioverpsi.data(), static_cast<std::size_t>(dhpsioverpsi.size())});
  }
  else
  {
    // Preserve the legacy particle-by-particle compatibility route. Only this
    // non-VP fallback materializes an nknot-by-nparameter matrix.
    dratio.resize(nknot, num_vars);
    std::fill(dratio.begin(), dratio.end(), ValueType(0));
    dlogpsi_vp.resize(dlogpsi.size());
    for (int j = 0; j < nknot; j++)
    {
      W.makeMove(iel, deltaV_[j]);
      psiratio[j] = psi.calcRatio(W, iel);
      psi.acceptMove(W, iel);
      W.acceptMove(iel);

      //use existing methods
      std::fill(dlogpsi_vp.begin(), dlogpsi_vp.end(), 0.0);
      psi.evaluateDerivativesWF(W, optvars, dlogpsi_vp);
      for (int v = 0; v < dlogpsi_vp.size(); ++v)
        dratio(j, v) = dlogpsi_vp[v] - dlogpsi[v];

      W.makeMove(iel, -deltaV_[j]);
      psi.calcRatio(W, iel);
      psi.acceptMove(W, iel);
      W.acceptMove(iel);
    }
  }

  RealType pairpot(0);
  for (int j = 0; j < nknot; j++)
  {
    wvec[j] = knot_pots_[j] * psiratio[j];
    pairpot += std::real(wvec[j]);
  }

  if (!vp)
    BLAS::gemv('N', num_vars, nknot, 1.0, dratio.data(), num_vars, wvec.data(), 1, 1.0,
               dhpsioverpsi.data(), 1);

  return pairpot;
}

void NonLocalECPComponent::mw_evaluateValueAndDerivatives(
    const RefVectorWithLeader<NonLocalECPComponent>& ecp_component_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVectorWithLeader<VirtualParticleSet>& vp_list,
    const RefVectorWithLeader<TrialWaveFunction>& psi_list,
    const RefVector<const NLPPJob<RealType>>& joblist,
    const OptVariables& optvars,
    const std::vector<TrialWaveFunction::ParameterDerivativeView>& weighted_derivatives,
    std::vector<RealType>& pairpots,
    ResourceCollection& collection)
{
  const std::size_t batch_size = ecp_component_list.size();
  if (p_list.size() != batch_size || vp_list.size() != batch_size || psi_list.size() != batch_size ||
      joblist.size() != batch_size || weighted_derivatives.size() != batch_size || pairpots.size() < batch_size)
    throw std::invalid_argument("NonLocalECPComponent derivative batch has inconsistent sizes");
  if (batch_size == 0)
    return;

  RefVector<const std::vector<PosType>> displacement_list;
  RefVector<const std::vector<ValueType>> bare_weight_list;
  RefVector<std::vector<ValueType>> ratio_list;
  displacement_list.reserve(batch_size);
  bare_weight_list.reserve(batch_size);
  ratio_list.reserve(batch_size);

  // Build each ragged job's quadrature positions and partial potentials before
  // entering the component-major TrialWaveFunction batch traversal.
  for (std::size_t batch_index = 0; batch_index < batch_size; ++batch_index)
  {
    NonLocalECPComponent& component = ecp_component_list[batch_index];
    const NLPPJob<RealType>& job    = joblist[batch_index];
    component.buildQuadraturePointDeltaPosAndPartialPotential(
        job.ion_elec_dist, job.ion_elec_displ, component.deltaV_, component.knot_pots_);
    for (int knot = 0; knot < component.nknot; ++knot)
      component.wvec[knot] = ValueType(component.knot_pots_[knot]);

    displacement_list.push_back(std::cref(component.deltaV_));
    bare_weight_list.push_back(std::cref(component.wvec));
    ratio_list.push_back(std::ref(component.psiratio));
  }

  RefVectorWithLeader<const VirtualParticleSet> const_vp_list(vp_list.getLeader());
  const_vp_list.reserve(batch_size);
  for (VirtualParticleSet& virtual_particles : vp_list)
    const_vp_list.push_back(virtual_particles);

  ResourceCollectionTeamLock<VirtualParticleSet> vp_resource_lock(collection, vp_list);
  VirtualParticleSet::mw_makeMoves(vp_list, p_list, displacement_list, joblist, true);
  TrialWaveFunction::mw_evaluateDerivRatiosWeighted(psi_list, const_vp_list, optvars, bare_weight_list,
                                                    ratio_list, weighted_derivatives);

  // Energy and derivative reductions consume identical total ratios and bare
  // potentials, preventing the two observable paths from drifting apart.
  for (std::size_t batch_index = 0; batch_index < batch_size; ++batch_index)
  {
    NonLocalECPComponent& component = ecp_component_list[batch_index];
    RealType pair_potential         = 0;
    for (int knot = 0; knot < component.nknot; ++knot)
    {
      component.wvec[knot] = component.knot_pots_[knot] * component.psiratio[knot];
      pair_potential += std::real(component.wvec[knot]);
    }
    pairpots[batch_index] = pair_potential;
  }
}

} // namespace qmcplusplus
