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
#include <exception>
#include <set>
#include <stdexcept>
#include <typeinfo>

#include "TrialWaveFunction.h"
#include "Particle/MCMultiParticleMoves.h"
#include "QMCWaveFunctions/Optimization/StructuredParameterProvider.h"
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
  std::string aname = aterm->getClassName();
  if (!aterm->getName().empty())
    aname += ":" + aterm->getName();

  if (aterm->isFermionic())
    app_log() << "  Added a fermionic WaveFunctionComponent " << aname << std::endl;

  for (auto& suffix : suffixes)
    WFC_timers_.push_back(createGlobalTimer(aname + "::" + suffix));

  Z.emplace_back(std::move(aterm));
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
  auto myclone                 = std::make_unique<TrialWaveFunction>(runtime_options_, myName, use_tasking_);
  myclone->BufferCursor        = BufferCursor;
  myclone->BufferCursor_scalar = BufferCursor_scalar;
  for (int i = 0; i < Z.size(); ++i)
    myclone->addComponent(Z[i]->makeClone(tqp));
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
  const size_t cursor_begin = collection.getCursor();
  int acquired_components   = 0;

  try
  {
    // First handle WFC resources
    for (int i = 0; i < wf_leader.Z.size(); ++i)
    {
      const auto wfc_list(extractWFCRefList(wf_list, i));
      wf_leader.Z[i]->acquireResource(collection, wfc_list);
      ++acquired_components;
    }

    // Handle wrapper resources if they exist
    if (wf_leader.twf_fastderiv_)
    {
      RefVectorWithLeader<TWFFastDerivWrapper> wrapper_list(*wf_leader.twf_fastderiv_);
      for (int iw = 0; iw < wf_list.size(); ++iw)
        wrapper_list.push_back(*wf_list[iw].twf_fastderiv_);
      wf_leader.twf_fastderiv_->acquireResource(collection, wrapper_list);
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
      {
        const auto wfc_list(extractWFCRefList(wf_list, i));
        wf_leader.Z[i]->releaseResource(collection, wfc_list);
      }
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

  for (const TrialWaveFunction& wavefunction : wf_list)
    if (wavefunction.multi_particle_proposal_pending_)
      throw std::logic_error(
          "Cannot release TrialWaveFunction resources with a pending selected-electron proposal");

  // Release WFC resources
  for (int i = 0; i < wf_leader.Z.size(); ++i)
  {
    const auto wfc_list(extractWFCRefList(wf_list, i));
    wf_leader.Z[i]->releaseResource(collection, wfc_list);
  }

  // Release wrapper resources if they exist
  if (wf_leader.twf_fastderiv_)
  {
    RefVectorWithLeader<TWFFastDerivWrapper> wrapper_list(*wf_leader.twf_fastderiv_);
    for (int iw = 0; iw < wf_list.size(); ++iw)
    {
      if (wf_list[iw].twf_fastderiv_)
        wrapper_list.push_back(*wf_list[iw].twf_fastderiv_);
    }
    wf_leader.twf_fastderiv_->releaseResource(collection, wrapper_list);
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


// Add debug checks in getOrCreateTWFFastDerivWrapper
TWFFastDerivWrapper& TrialWaveFunction::getOrCreateTWFFastDerivWrapper(const ParticleSet& P)
{
  if (!twf_fastderiv_)
  {
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
