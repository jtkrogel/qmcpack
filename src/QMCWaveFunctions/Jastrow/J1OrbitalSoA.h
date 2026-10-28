//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                  Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                  Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//                  Shiv Upadhyay, shivnupadhyay@gmail.com, University of Pittsburgh
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_ONEBODYJASTROW_OPTIMIZED_SOA_H
#define QMCPLUSPLUS_ONEBODYJASTROW_OPTIMIZED_SOA_H

#include <map>
#include <numeric>
#include "Configuration.h"
#include <ResourceHandle.h>
#include "Particle/DistanceTable.h"
#include "Particle/MCMultiParticleMoves.h"
#include "CPU/VectorOps.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "Utilities/qmc_common.h"
#include "Utilities/IteratorUtility.h"
#include "CPU/SIMD/aligned_allocator.hpp"
#include "CPU/SIMD/algorithm.hpp"

#include "BsplineFunctor.h"
#include "SplineFunctors.h"
#include "UserFunctor.h"
#include "ShortRangeCuspFunctor.h"
#include "PadeFunctors.h"

namespace qmcplusplus
{

template<typename T>
struct J1OrbitalSoAMultiWalkerMem;

/** @ingroup WaveFunctionComponent
 *  @brief Specialization for one-body Jastrow function using multiple functors
 */
template<class FT>
class J1OrbitalSoA : public WaveFunctionComponent
{
  ///alias FuncType
  using FuncType = FT;
  ///type of each component U, dU, d2U;
  using valT = typename FT::real_type;
  ///element position type
  using posT = TinyVector<valT, OHMMS_DIM>;
  ///use the same container
  using DistRow  = DistanceTable::DistRow;
  using DisplRow = DistanceTable::DisplRow;

  using GradDerivVec  = Vector<QTFull::GradType>;
  using ValueDerivVec = Vector<QTFull::ValueType>;

  // One bounded J-only staging bank per active component/walker, never per sample.
  std::unique_ptr<ParticleSet> multi_particle_particles_;
  std::unique_ptr<WaveFunctionComponent> multi_particle_component_;
  ParticleSet::ParticleGradient multi_particle_gradient_;
  ParticleSet::ParticleLaplacian multi_particle_laplacian_;
  std::uint64_t multi_particle_fingerprint_ = 0;
  bool multi_particle_pending_ = false;

  /// if true use offload
  const bool use_offload_;

  ///table index
  const int myTableID;
  ///number of ions
  const int Nions;
  ///number of electrons
  const int Nelec;
  /// number of ion groups
  const int NumGroups;
  ///reference to the sources (ions)
  const ParticleSet& Ions;
  /// the group_id of each particle
  Vector<int, OffloadPinnedAllocator<int>> grp_ids;

  ///variables handled by this orbital
  OptVariables myVars;

  valT curAt;
  valT curLap;
  posT curGrad;

  ///\f$Vat[i] = sum_(j) u_{i,j}\f$
  Vector<valT> Vat;
  aligned_vector<valT> U, dU, d2U, d3U;
  aligned_vector<valT> DistCompressed;
  aligned_vector<int> DistIndice;
  Vector<posT> Grad;
  Vector<valT> Lap;
  ///Container for \f$F[ig*NIons+jg]\f$
  std::vector<FT*> J1Functors;
  /** functor for each ion group
   * Conceptually, this is different from unique functions.
   * In a spin polarized system, two species may share the same J1 functor.
   * FIXME However this is not supported right now. Each species needs its dedicated function.
   */
  std::vector<FT*> GroupFunctors;

  std::vector<std::pair<int, int>> OffSet;
  Vector<RealType> dLogPsi;
  std::vector<GradDerivVec> gradLogPsi;
  std::vector<ValueDerivVec> lapLogPsi;

  ResourceHandle<J1OrbitalSoAMultiWalkerMem<RealType>> mw_mem_handle_;

  void resizeWFOptVectors()
  {
    dLogPsi.resize(myVars.size());
    gradLogPsi.resize(myVars.size(), GradDerivVec(Nelec));
    lapLogPsi.resize(myVars.size(), ValueDerivVec(Nelec));
  }

  /// compute G and L from internally stored data
  inline QTFull::RealType computeGL(ParticleSet::ParticleGradient& G, ParticleSet::ParticleLaplacian& L) const
  {
    for (size_t iat = 0; iat < Nelec; ++iat)
    {
      G[iat] += Grad[iat];
      L[iat] -= Lap[iat];
    }
    return -simd::accumulate_n(Vat.data(), Nelec, QTFull::RealType());
  }

  inline LogValue evaluateGL(const ParticleSet& P,
                             ParticleSet::ParticleGradient& G,
                             ParticleSet::ParticleLaplacian& L,
                             bool fromscratch = false) override
  { return log_value_ = computeGL(G, L); }

  /** compute gradient and lap
   * @return lap
   */
  inline valT accumulateGL(const valT* restrict du, const valT* restrict d2u, const DisplRow& displ, posT& grad) const
  {
    valT lap(0);
    constexpr valT lapfac = OHMMS_DIM - RealType(1);
    //#pragma omp simd reduction(+:lap)
    for (int jat = 0; jat < Nions; ++jat)
      lap += d2u[jat] + lapfac * du[jat];
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
    {
      const valT* restrict dX = displ.data(idim);
      valT s                  = valT();
      //#pragma omp simd reduction(+:s)
      for (int jat = 0; jat < Nions; ++jat)
        s += du[jat] * dX[jat];
      grad[idim] = s;
    }
    return lap;
  }

  inline valT computeU(const DistRow& dist)
  {
    valT curVat(0);
    for (int jg = 0; jg < NumGroups; ++jg)
    {
      if (J1UniqueFunctors[jg] != nullptr)
        curVat +=
            J1UniqueFunctors[jg]->evaluateV(-1, Ions.first(jg), Ions.last(jg), dist.data(), DistCompressed.data());
    }
    return curVat;
  }

  /** compute U, dU and d2U
   * @param P quantum particleset
   * @param iat the moving particle
   * @param dist starting address of the distances of the ions wrt the iat-th particle
   */
  inline void computeU3(const ParticleSet& P, int iat, const DistRow& dist)
  {
    constexpr valT czero(0);
    std::fill_n(U.data(), Nions, czero);
    std::fill_n(dU.data(), Nions, czero);
    std::fill_n(d2U.data(), Nions, czero);

    for (int jg = 0; jg < NumGroups; ++jg)
    {
      if (J1UniqueFunctors[jg] == nullptr)
        continue;
      J1UniqueFunctors[jg]->evaluateVGL(-1, Ions.first(jg), Ions.last(jg), dist.data(), U.data(), dU.data(), d2U.data(),
                                        DistCompressed.data(), DistIndice.data());
    }
  }

protected:
  ///container for the unique Jastrow functions
  std::vector<std::unique_ptr<FT>> J1UniqueFunctors;

public:
  J1OrbitalSoA(const std::string& obj_name, const ParticleSet& ions, ParticleSet& els, bool use_offload);

  J1OrbitalSoA(const J1OrbitalSoA& rhs) = delete;

  ~J1OrbitalSoA();

  std::string getClassName() const override { return "J1OrbitalSoA"; }

  /* initialize storage */
  void initialize(const ParticleSet& els)
  {
    J1Functors.resize(Nions, nullptr);
    GroupFunctors.resize(NumGroups, nullptr);
    J1UniqueFunctors.resize(Ions.getSpeciesSet().getTotalNum());
    Vat.resize(Nelec);
    Grad.resize(Nelec);
    Lap.resize(Nelec);

    U.resize(Nions);
    dU.resize(Nions);
    d2U.resize(Nions);
    d3U.resize(Nions);
    DistCompressed.resize(Nions);
    DistIndice.resize(Nions);
  }

  void addFunc(int source_type, std::unique_ptr<FT> afunc, int target_type = -1)
  {
    for (int i = 0; i < J1Functors.size(); i++)
      if (Ions.getGroupID(i) == source_type)
        J1Functors[i] = afunc.get();
    GroupFunctors[source_type]    = afunc.get();
    J1UniqueFunctors[source_type] = std::move(afunc);
  }

  void checkSanity() const override;

  const auto& getFunctors() const { return J1Functors; }

  void createResource(ResourceCollection& collection) const override;

  void acquireResource(ResourceCollection& collection,
                       const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const override;

  void releaseResource(ResourceCollection& collection,
                       const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const override;

  /// CPU real full-configuration proposals stage native J-only products privately.
  bool supportsMultiParticleMoves() const noexcept override
  {
#if !defined(QMC_COMPLEX) && !defined(MIXED_PRECISION)
    return !use_offload_;
#else
    return false;
#endif
  }
  void mw_evaluateMultiParticleMove(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list, const MCMultiParticleMoves<CoordsType::POS>& moves,
      std::vector<LogValue>& log_ratios, const RefVector<ParticleSet::ParticleGradient>& proposed_gradient_list,
      const RefVector<ParticleSet::ParticleLaplacian>& proposed_laplacian_list) const override;
  void mw_accept_rejectMultiParticleMove(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list, const MCMultiParticleMoves<CoordsType::POS>& moves,
      const std::vector<bool>& accepted) const override;

  void recompute(const ParticleSet& P) override
  {
    const auto& d_ie(P.getDistTableAB(myTableID));
    for (int iat = 0; iat < Nelec; ++iat)
    {
      computeU3(P, iat, d_ie.getDistRow(iat));
      Vat[iat] = simd::accumulate_n(U.data(), Nions, valT());
      Lap[iat] = accumulateGL(dU.data(), d2U.data(), d_ie.getDisplRow(iat), Grad[iat]);
    }
  }

  LogValue evaluateLog(const ParticleSet& P,
                       ParticleSet::ParticleGradient& G,
                       ParticleSet::ParticleLaplacian& L) override
  {
    recompute(P);
    return log_value_ = computeGL(G, L);
  }

  void evaluateHessian(ParticleSet& P, HessVector& grad_grad_psi) override
  {
    const auto& d_ie(P.getDistTableAB(myTableID));
    valT dudr, d2udr2;

    Tensor<valT, DIM> ident;
    grad_grad_psi = 0.0;
    ident.diagonal(1.0);

    for (int iel = 0; iel < Nelec; ++iel)
    {
      const auto& dist  = d_ie.getDistRow(iel);
      const auto& displ = d_ie.getDisplRow(iel);
      for (int iat = 0; iat < Nions; iat++)
      {
        int gid    = Ions.getGroupID(iat);
        auto* func = J1UniqueFunctors[gid].get();
        if (func != nullptr)
        {
          RealType r    = dist[iat];
          RealType rinv = 1.0 / r;
          PosType dr    = displ[iat];
          func->evaluate(r, dudr, d2udr2);
          grad_grad_psi[iel] -= rinv * rinv * outerProduct(dr, dr) * (d2udr2 - dudr * rinv) + ident * dudr * rinv;
        }
      }
    }
  }

  PsiValue ratio(ParticleSet& P, int iat) override
  {
    UpdateMode = ORB_PBYP_RATIO;
    curAt      = computeU(P.getDistTableAB(myTableID).getTempDists());
    return std::exp(static_cast<PsiValue>(Vat[iat] - curAt));
  }

  inline void evaluateRatios(const VirtualParticleSet& VP, std::vector<ValueType>& ratios) override
  {
    for (int k = 0; k < ratios.size(); ++k)
      ratios[k] = std::exp(Vat[VP.refPtcl] - computeU(VP.getDistTableAB(myTableID).getDistRow(k)));
  }

  void mw_evaluateRatios(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                         const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
                         std::vector<std::vector<ValueType>>& ratios) const override;

  void evaluateDerivatives(ParticleSet& P,
                           const OptVariables& active,
                           Vector<ValueType>& dlogpsi,
                           Vector<ValueType>& dhpsioverpsi) override
  {
    evaluateDerivativesWF(P, active, dlogpsi);
    bool recalculate(false);
    std::vector<bool> rcsingles(myVars.size(), false);
    for (int k = 0; k < myVars.size(); ++k)
    {
      int kk = myVars.where(k);
      if (kk < 0)
        continue;
      recalculate  = true;
      rcsingles[k] = true;
    }
    if (recalculate)
    {
      for (int k = 0; k < myVars.size(); ++k)
      {
        int kk = myVars.where(k);
        if (kk < 0)
          continue;
        if (rcsingles[k])
        {
          dhpsioverpsi[kk] = -RealType(0.5) * ValueType(Sum(lapLogPsi[k])) - ValueType(Dot(P.G, gradLogPsi[k]));
        }
      }
    }
  }

  void evaluateDerivativesWF(ParticleSet& P, const OptVariables& active, Vector<ValueType>& dlogpsi) override
  {
    resizeWFOptVectors();
    bool recalculate(false);
    std::vector<bool> rcsingles(myVars.size(), false);
    for (int k = 0; k < myVars.size(); ++k)
    {
      int kk = myVars.where(k);
      if (kk < 0)
        continue;
      recalculate  = true;
      rcsingles[k] = true;
    }
    if (recalculate)
    {
      const size_t NumVars = myVars.size();
      for (int p = 0; p < NumVars; ++p)
      {
        gradLogPsi[p] = 0.0;
        lapLogPsi[p]  = 0.0;
      }
      dLogPsi = 0.0;

      const auto& d_table = P.getDistTableAB(myTableID);
      std::vector<TinyVector<RealType, 3>> derivs(NumVars);

      constexpr RealType cone(1);
      constexpr RealType lapfac(OHMMS_DIM - cone);
      const size_t ns = d_table.sources();
      const size_t nt = P.getTotalNum();

      aligned_vector<int> iadj(nt);
      aligned_vector<RealType> dist(nt);
      std::vector<PosType> displ(nt);

      for (size_t i = 0; i < ns; ++i)
      {
        FT* func = J1Functors[i];
        if (func == nullptr)
          continue;
        int first(OffSet[i].first);
        int last(OffSet[i].second);
        bool recalcFunc(false);
        for (int rcs = first; rcs < last; rcs++)
          if (rcsingles[rcs] == true)
            recalcFunc = true;
        if (recalcFunc)
        {
          for (size_t j = 0; j < nt; ++j)
          {
            std::fill(derivs.begin(), derivs.end(), 0);
            auto dist = P.getDistTableAB(myTableID).getDistRow(j)[i];
            if (!func->evaluateDerivatives(dist, derivs))
              continue;
            RealType rinv(cone / dist);
            const PosType& dr = P.getDistTableAB(myTableID).getDisplRow(j)[i];
            for (int p = first, ip = 0; p < last; ++p, ++ip)
            {
              dLogPsi[p] -= derivs[ip][0];
              RealType dudr(rinv * derivs[ip][1]);
              gradLogPsi[p][j] += dudr * dr;
              lapLogPsi[p][j] -= derivs[ip][2] + lapfac * dudr;
            }
          }
        }
      }
      for (int k = 0; k < myVars.size(); ++k)
      {
        int kk = myVars.where(k);
        if (kk < 0)
          continue;
        if (rcsingles[k])
        {
          dlogpsi[kk] = ValueType(dLogPsi[k]);
        }
      }
    }
  }

  void evaluateRatiosAlltoOne(ParticleSet& P, std::vector<ValueType>& ratios) override
  {
    const auto& dist = P.getDistTableAB(myTableID).getTempDists();
    curAt            = valT(0);
    for (int jg = 0; jg < NumGroups; ++jg)
    {
      if (J1UniqueFunctors[jg] != nullptr)
        curAt += J1UniqueFunctors[jg]->evaluateV(-1, Ions.first(jg), Ions.last(jg), dist.data(), DistCompressed.data());
    }

    for (int i = 0; i < Nelec; ++i)
      ratios[i] = std::exp(Vat[i] - curAt);
  }

  /** compute the gradient during particle-by-particle update
   * @param P quantum particleset
   * @param iat particle index
   */
  GradType evalGrad(ParticleSet& P, int iat) override { return GradType(Grad[iat]); }

  /** compute the gradient during particle-by-particle update
   * @param P quantum particleset
   * @param iat particle index
   *
   * Using getTempDists(). curAt, curGrad and curLap are computed.
   */
  PsiValue ratioGrad(ParticleSet& P, int iat, GradType& grad_iat) override
  {
    UpdateMode = ORB_PBYP_PARTIAL;

    computeU3(P, iat, P.getDistTableAB(myTableID).getTempDists());
    curLap = accumulateGL(dU.data(), d2U.data(), P.getDistTableAB(myTableID).getTempDispls(), curGrad);
    curAt  = simd::accumulate_n(U.data(), Nions, valT());
    grad_iat += curGrad;
    return std::exp(static_cast<PsiValue>(Vat[iat] - curAt));
  }

  /** Rejected move. Nothing to do */
  inline void restore(int iat) override {}

  /** Accpted move. Update Vat[iat],Grad[iat] and Lap[iat] */
  void acceptMove(ParticleSet& P, int iat, bool safe_to_delay = false) override
  {
    if (UpdateMode == ORB_PBYP_RATIO)
    {
      computeU3(P, iat, P.getDistTableAB(myTableID).getTempDists());
      curLap = accumulateGL(dU.data(), d2U.data(), P.getDistTableAB(myTableID).getTempDispls(), curGrad);
    }

    log_value_ += Vat[iat] - curAt;
    Vat[iat]  = curAt;
    Grad[iat] = curGrad;
    Lap[iat]  = curLap;
  }


  inline void registerData(ParticleSet& P, WFBufferType& buf) override
  {
    if (Bytes_in_WFBuffer == 0)
    {
      Bytes_in_WFBuffer = buf.current();
      buf.add(Vat.begin(), Vat.end());
      buf.add(Grad.begin(), Grad.end());
      buf.add(Lap.begin(), Lap.end());
      Bytes_in_WFBuffer = buf.current() - Bytes_in_WFBuffer;
      // free local space
      Vat.free();
      Grad.free();
      Lap.free();
    }
    else
    {
      buf.forward(Bytes_in_WFBuffer);
    }
  }

  inline LogValue updateBuffer(ParticleSet& P, WFBufferType& buf, bool fromscratch = false) override
  {
    log_value_ = computeGL(P.G, P.L);
    buf.forward(Bytes_in_WFBuffer);
    return log_value_;
  }

  inline void copyFromBuffer(ParticleSet& P, WFBufferType& buf) override
  {
    Vat.attachReference(buf.lendReference<valT>(Nelec), Nelec);
    Grad.attachReference(buf.lendReference<posT>(Nelec), Nelec);
    Lap.attachReference(buf.lendReference<valT>(Nelec), Nelec);
  }

  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet& tqp) const override
  {
    auto j1copy = std::make_unique<J1OrbitalSoA<FT>>(my_name_, Ions, tqp, use_offload_);
    for (size_t i = 0, n = J1UniqueFunctors.size(); i < n; ++i)
    {
      if (J1UniqueFunctors[i] != nullptr)
      {
        auto fc = std::make_unique<FT>(*J1UniqueFunctors[i].get());
        j1copy->addFunc(i, std::move(fc));
      }
    }
    j1copy->myVars = myVars;
    j1copy->OffSet = OffSet;
    return j1copy;
  }

  /**@{ WaveFunctionComponent virtual functions that are not essential for the development */
  bool isOptimizable() const override { return true; }

  void extractOptimizableObjectRefs(UniqueOptObjRefs& opt_obj_refs) override
  {
    for (auto& functor : J1UniqueFunctors)
      if (functor)
        opt_obj_refs.push_back(*functor);
  }

  void checkOutVariables(const OptVariables& active) override
  {
    myVars.clear();
    for (int i = 0; i < J1UniqueFunctors.size(); ++i)
    {
      if (J1UniqueFunctors[i])
      {
        J1UniqueFunctors[i]->myVars.getIndex(active);
        myVars.insertFrom(J1UniqueFunctors[i]->myVars);
      }
    }
    myVars.getIndex(active);
    const size_t NumVars = myVars.size();
    if (NumVars)
    {
      OffSet.resize(J1Functors.size());
      int varoffset = myVars.Index[0];
      for (int i = 0; i < J1Functors.size(); ++i)
      {
        if (J1Functors[i] != nullptr)
        {
          OffSet[i].first  = J1Functors[i]->myVars.Index.front() - varoffset;
          OffSet[i].second = J1Functors[i]->myVars.Index.size() + OffSet[i].first;
        }
      }
    }
    for (size_t i = 0, n = J1UniqueFunctors.size(); i < n; ++i)
      if (J1UniqueFunctors[i] != nullptr)
        J1UniqueFunctors[i]->checkOutVariables(active);
  }

  /**@} */

  void evaluateDerivRatios(const VirtualParticleSet& VP,
                           const OptVariables& optvars,
                           std::vector<ValueType>& ratios,
                           Matrix<ValueType>& dratios) override
  {
    evaluateRatios(VP, ratios);
    bool recalculate(false);
    std::vector<bool> rcsingles(myVars.size(), false);
    for (int k = 0; k < myVars.size(); ++k)
    {
      const int kk = myVars.where(k);
      if (kk < 0)
        continue;
      recalculate  = true;
      rcsingles[k] = true;
    }

    if (recalculate)
    {
      const size_t NumVars = myVars.size();
      const auto& d_table  = VP.getDistTableAB(myTableID);
      std::vector<RealType> derivs_ref(NumVars);
      std::vector<RealType> derivs(NumVars);

      const size_t ns = d_table.sources();
      const size_t nt = VP.getTotalNum();

      const auto& dist_ref = VP.getRefPS().getDistTableAB(myTableID).getDistRow(VP.refPtcl);

      for (size_t i = 0; i < ns; ++i)
      {
        FT* func = J1Functors[i];
        if (func == nullptr)
          continue;
        int first(OffSet[i].first);
        int last(OffSet[i].second);
        bool recalcFunc(false);
        for (int rcs = first; rcs < last; rcs++)
          if (rcsingles[rcs] == true)
            recalcFunc = true;
        if (recalcFunc)
        {
          //first calculate the old derivatives VP.refPctl.
          std::fill(derivs_ref.begin(), derivs_ref.end(), 0);
          func->evaluateDerivatives(dist_ref[i], derivs_ref);
          for (size_t j = 0; j < nt; ++j)
          {
            std::fill(derivs.begin(), derivs.end(), 0);
            //first calculate the new derivatives
            func->evaluateDerivatives(VP.getDistTableAB(myTableID).getDistRow(j)[i], derivs);
            //compute the new derivatives - old derivatives
            for (int ip = 0, p = func->myVars.Index.front(); ip < func->myVars.Index.size(); ++ip, ++p)
              dratios[j][p] += derivs_ref[ip] - derivs[ip];
          }
        }
      }
    }
  }

  inline GradType evalGradSource(ParticleSet& P, ParticleSet& source, int isrc) override
  {
    GradType g_return(0.0);
    const auto& d_ie(P.getDistTableAB(myTableID));
    for (int iat = 0; iat < Nelec; ++iat)
    {
      const auto& dist  = d_ie.getDistRow(iat);
      const auto& displ = d_ie.getDisplRow(iat);
      int gid           = source.getGroupID(isrc);
      RealType r        = dist[isrc];
      RealType rinv     = 1.0 / r;
      PosType dr        = displ[isrc];

      if (J1UniqueFunctors[gid] != nullptr)
      {
        U[isrc] = J1UniqueFunctors[gid]->evaluate(dist[isrc], dU[isrc], d2U[isrc], d3U[isrc]);
        g_return -= dU[isrc] * rinv * dr;
      }
    }
    return g_return;
  }

  inline GradType evalGradSource(ParticleSet& P,
                                 ParticleSet& source,
                                 int isrc,
                                 TinyVector<ParticleSet::ParticleGradient, OHMMS_DIM>& grad_grad,
                                 TinyVector<ParticleSet::ParticleLaplacian, OHMMS_DIM>& lapl_grad) override
  {
    GradType g_return(0.0);
    const auto& d_ie(P.getDistTableAB(myTableID));
    for (int iat = 0; iat < Nelec; ++iat)
    {
      const auto& dist  = d_ie.getDistRow(iat);
      const auto& displ = d_ie.getDisplRow(iat);
      int gid           = source.getGroupID(isrc);
      RealType r        = dist[isrc];
      RealType rinv     = 1.0 / r;
      PosType dr        = displ[isrc];

      if (J1UniqueFunctors[gid] != nullptr)
      {
        U[isrc] = J1UniqueFunctors[gid]->evaluate(dist[isrc], dU[isrc], d2U[isrc], d3U[isrc]);
      }
      else
      {
        APP_ABORT("J1OrbitalSoa::evaluateGradSource:  J1UniqueFunctors[gid]==nullptr")
      }

      g_return -= dU[isrc] * rinv * dr;

      //The following terms depend only on the radial component r.  Thus,
      //we compute them and mix with position vectors to acquire the full
      //cartesian vector objects.
      valT grad_component = (d2U[isrc] - dU[isrc] * rinv);
      valT lapl_component = d3U[isrc] + 2 * rinv * grad_component;

      for (int idim = 0; idim < OHMMS_DIM; idim++)
      {
        grad_grad[idim][iat] += dr[idim] * dr * rinv * rinv * grad_component;
        grad_grad[idim][iat][idim] += rinv * dU[isrc];

        lapl_grad[idim][iat] -= lapl_component * rinv * dr[idim];
      }
    }
    return g_return;
  }
};

extern template class J1OrbitalSoA<BsplineFunctor<QMCTraits::RealType>>;
extern template class J1OrbitalSoA<
    CubicSplineSingle<QMCTraits::RealType, CubicBspline<QMCTraits::RealType, LINEAR_1DGRID, FIRSTDERIV_CONSTRAINTS>>>;
extern template class J1OrbitalSoA<UserFunctor<QMCTraits::RealType>>;
extern template class J1OrbitalSoA<ShortRangeCuspFunctor<QMCTraits::RealType>>;
extern template class J1OrbitalSoA<PadeFunctor<QMCTraits::RealType>>;
extern template class J1OrbitalSoA<Pade2ndOrderFunctor<QMCTraits::RealType>>;
template<class FT>
void J1OrbitalSoA<FT>::mw_evaluateMultiParticleMove(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list, const MCMultiParticleMoves<CoordsType::POS>& moves,
    std::vector<LogValue>& log_ratios, const RefVector<ParticleSet::ParticleGradient>& proposed_gradient_list,
    const RefVector<ParticleSet::ParticleLaplacian>& proposed_laplacian_list) const
{
  const auto count = wfc_list.size();
  if (!supportsMultiParticleMoves() || p_list.size() != count || moves.walkerCount() != count ||
      log_ratios.size() != count || proposed_gradient_list.size() != count || proposed_laplacian_list.size() != count)
    throw std::invalid_argument("J1OrbitalSoA CPU collective proposal has incompatible shapes or backend");
  moves.validateFor(p_list);
  for (std::size_t walker = 0; walker < count; ++walker)
  {
    const auto& component = wfc_list.getCastedElement<J1OrbitalSoA<FT>>(walker);
    if (!component.supportsMultiParticleMoves() || component.multi_particle_pending_ ||
        p_list[walker].getTotalNum() != component.Nelec ||
        proposed_gradient_list[walker].get().size() != component.Nelec ||
        proposed_laplacian_list[walker].get().size() != component.Nelec)
      throw std::logic_error("J1OrbitalSoA CPU collective proposal requires resolved matching components");
    for (const ParticleSet& accepted : p_list)
      if (proposed_gradient_list[walker].get().data() == accepted.G.data() ||
          proposed_laplacian_list[walker].get().data() == accepted.L.data())
        throw std::invalid_argument("J1OrbitalSoA collective outputs alias accepted particle products");
  }
  try
  {
    for (std::size_t walker = 0; walker < count; ++walker)
    {
      auto& component = wfc_list.getCastedElement<J1OrbitalSoA<FT>>(walker);
      if (!component.multi_particle_particles_)
        component.multi_particle_particles_ = std::make_unique<ParticleSet>(p_list[walker]);
      auto& proposed = *component.multi_particle_particles_;
      proposed.R = p_list[walker].R;
      const auto slice = moves.slice(walker);
      for (std::size_t entry = 0; entry < slice.size(); ++entry)
        proposed.R[slice.particleIndex(entry)] = slice.proposedPosition(entry);
      proposed.update(true); // Only short-range J distances are consumed, never its structure factor.
      // Refresh small conventional functors so coefficient publications/resets cannot leave stale staging.
      component.multi_particle_component_ = component.makeClone(proposed);
      component.multi_particle_gradient_.resize(component.Nelec);
      component.multi_particle_laplacian_.resize(component.Nelec);
      component.multi_particle_gradient_ = ValueType(0);
      component.multi_particle_laplacian_ = ValueType(0);
      component.multi_particle_component_->evaluateLog(proposed, component.multi_particle_gradient_,
          component.multi_particle_laplacian_);
      component.multi_particle_fingerprint_ = moves.fingerprint();
      component.multi_particle_pending_ = true;
    }
    for (std::size_t walker = 0; walker < count; ++walker)
    {
      const auto& component = wfc_list.getCastedElement<J1OrbitalSoA<FT>>(walker);
      log_ratios[walker] = component.multi_particle_component_->get_log_value() - component.log_value_;
      for (int electron = 0; electron < component.Nelec; ++electron)
      {
        proposed_gradient_list[walker].get()[electron] += component.multi_particle_gradient_[electron];
        proposed_laplacian_list[walker].get()[electron] += component.multi_particle_laplacian_[electron];
      }
    }
  }
  catch (...)
  {
    for (WaveFunctionComponent& base : wfc_list)
    {
      auto& component = static_cast<J1OrbitalSoA<FT>&>(base);
      component.multi_particle_pending_ = false;
      component.multi_particle_fingerprint_ = 0;
    }
    throw;
  }
}

template<class FT>
void J1OrbitalSoA<FT>::mw_accept_rejectMultiParticleMove(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list, const MCMultiParticleMoves<CoordsType::POS>& moves,
    const std::vector<bool>& accepted) const
{
  if (!supportsMultiParticleMoves() || wfc_list.size() != p_list.size() ||
      moves.walkerCount() != wfc_list.size() || accepted.size() != wfc_list.size())
    throw std::invalid_argument("J1OrbitalSoA CPU collective resolution has incompatible shapes or backend");
  moves.validateFor(p_list);
  const auto fingerprint = moves.fingerprint();
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    const auto& component = wfc_list.getCastedElement<J1OrbitalSoA<FT>>(walker);
    if (!component.multi_particle_pending_ || component.multi_particle_fingerprint_ != fingerprint ||
        !component.multi_particle_component_)
      throw std::logic_error("J1OrbitalSoA collective resolution has no matching pending proposal");
  }
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<J1OrbitalSoA<FT>>(walker);
    if (accepted[walker])
    {
      const auto& next = static_cast<const J1OrbitalSoA<FT>&>(*component.multi_particle_component_);
      std::copy_n(next.Vat.data(), component.Nelec, component.Vat.data());
      std::copy_n(next.Grad.data(), component.Nelec, component.Grad.data());
      std::copy_n(next.Lap.data(), component.Nelec, component.Lap.data());
      component.log_value_ = next.log_value_;
    }
    component.multi_particle_pending_ = false;
    component.multi_particle_fingerprint_ = 0;
  }
}

} // namespace qmcplusplus
#endif
