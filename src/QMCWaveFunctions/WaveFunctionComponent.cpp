//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Raymond Clay III, j.k.rofling@gmail.com, Lawrence Livermore National Laboratory
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include "WaveFunctionComponent.h"

namespace qmcplusplus
{
// for return types
using PsiValue = WaveFunctionComponent::PsiValue;

WaveFunctionComponent::WaveFunctionComponent(const std::string& obj_name)
    : UpdateMode(ORB_WALKER), Bytes_in_WFBuffer(0), my_name_(obj_name), log_value_(0.0)
{}

WaveFunctionComponent::~WaveFunctionComponent() = default;

void WaveFunctionComponent::mw_evaluateLog(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                           const RefVectorWithLeader<ParticleSet>& p_list,
                                           const RefVector<ParticleSet::ParticleGradient>& G_list,
                                           const RefVector<ParticleSet::ParticleLaplacian>& L_list) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    wfc_list[iw].evaluateLog(p_list[iw], G_list[iw], L_list[iw]);
}

void WaveFunctionComponent::recompute(const ParticleSet& P)
{
  ParticleSet::ParticleGradient temp_G(P.getTotalNum());
  ParticleSet::ParticleLaplacian temp_L(P.getTotalNum());

  evaluateLog(P, temp_G, temp_L);
}

void WaveFunctionComponent::mw_recompute(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                         const RefVectorWithLeader<ParticleSet>& p_list,
                                         const std::vector<bool>& recompute) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    if (recompute[iw])
      wfc_list[iw].recompute(p_list[iw]);
}

void WaveFunctionComponent::mw_prepareGroup(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                            const RefVectorWithLeader<ParticleSet>& p_list,
                                            int ig) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    wfc_list[iw].prepareGroup(p_list[iw], ig);
}

template<CoordsType CT>
void WaveFunctionComponent::mw_evalGrad(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                        const RefVectorWithLeader<ParticleSet>& p_list,
                                        const int iat,
                                        TWFGrads<CT>& grad_now) const
{
  if constexpr (CT == CoordsType::POS_SPIN)
    mw_evalGradWithSpin(wfc_list, p_list, iat, grad_now.grads_positions, grad_now.grads_spins);
  else
    mw_evalGrad(wfc_list, p_list, iat, grad_now.grads_positions);
}

void WaveFunctionComponent::mw_evalGrad(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                        const RefVectorWithLeader<ParticleSet>& p_list,
                                        int iat,
                                        std::vector<GradType>& grad_now) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    grad_now[iw] = wfc_list[iw].evalGrad(p_list[iw], iat);
}

void WaveFunctionComponent::mw_evalGradWithSpin(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                                const RefVectorWithLeader<ParticleSet>& p_list,
                                                int iat,
                                                std::vector<GradType>& grad_now,
                                                std::vector<ComplexType>& spingrad_now) const
{
  mw_evalGrad(wfc_list, p_list, iat, grad_now);
  for (int iw = 0; iw < wfc_list.size(); iw++)
    spingrad_now[iw] = 0;
}

void WaveFunctionComponent::mw_evalGradWithSpin_serialized(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                                           const RefVectorWithLeader<ParticleSet>& p_list,
                                                           int iat,
                                                           std::vector<GradType>& grad_now,
                                                           std::vector<ComplexType>& spingrad_now) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
  {
    spingrad_now[iw] = 0;
    grad_now[iw]     = wfc_list[iw].evalGradWithSpin(p_list[iw], iat, spingrad_now[iw]);
  }
}

void WaveFunctionComponent::mw_calcRatio(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                         const RefVectorWithLeader<ParticleSet>& p_list,
                                         int iat,
                                         std::vector<PsiValue>& ratios) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    ratios[iw] = wfc_list[iw].ratio(p_list[iw], iat);
}


PsiValue WaveFunctionComponent::ratioGrad(ParticleSet& P, int iat, GradType& grad_iat)
{
  APP_ABORT("WaveFunctionComponent::ratioGrad is not implemented in " + getClassName() + " class.");
  return ValueType();
}

template<CoordsType CT>
void WaveFunctionComponent::mw_ratioGrad(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                         const RefVectorWithLeader<ParticleSet>& p_list,
                                         int iat,
                                         std::vector<PsiValue>& ratios,
                                         TWFGrads<CT>& grad_new) const
{
  if constexpr (CT == CoordsType::POS_SPIN)
    mw_ratioGradWithSpin(wfc_list, p_list, iat, ratios, grad_new.grads_positions, grad_new.grads_spins);
  else
    mw_ratioGrad(wfc_list, p_list, iat, ratios, grad_new.grads_positions);
}

void WaveFunctionComponent::mw_ratioGrad(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                         const RefVectorWithLeader<ParticleSet>& p_list,
                                         int iat,
                                         std::vector<PsiValue>& ratios,
                                         std::vector<GradType>& grad_new) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    ratios[iw] = wfc_list[iw].ratioGrad(p_list[iw], iat, grad_new[iw]);
}

void WaveFunctionComponent::mw_ratioGradWithSpin(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                                 const RefVectorWithLeader<ParticleSet>& p_list,
                                                 int iat,
                                                 std::vector<PsiValue>& ratios,
                                                 std::vector<GradType>& grad_new,
                                                 std::vector<ComplexType>& spingrad_new) const
{ mw_ratioGrad(wfc_list, p_list, iat, ratios, grad_new); }

void WaveFunctionComponent::mw_ratioGradWithSpin_serialized(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                                            const RefVectorWithLeader<ParticleSet>& p_list,
                                                            int iat,
                                                            std::vector<PsiValue>& ratios,
                                                            std::vector<GradType>& grad_new,
                                                            std::vector<ComplexType>& spingrad_new) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    ratios[iw] = wfc_list[iw].ratioGradWithSpin(p_list[iw], iat, grad_new[iw], spingrad_new[iw]);
}

void WaveFunctionComponent::mw_accept_rejectMove(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                                 const RefVectorWithLeader<ParticleSet>& p_list,
                                                 int iat,
                                                 const std::vector<bool>& isAccepted,
                                                 bool safe_to_delay) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    if (isAccepted[iw])
      wfc_list[iw].acceptMove(p_list[iw], iat, safe_to_delay);
    else
      wfc_list[iw].restore(iat);
}

void WaveFunctionComponent::mw_completeUpdates(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    wfc_list[iw].completeUpdates();
}

WaveFunctionComponent::LogValue WaveFunctionComponent::evaluateGL(const ParticleSet& P,
                                                                  ParticleSet::ParticleGradient& G,
                                                                  ParticleSet::ParticleLaplacian& L,
                                                                  bool fromscratch)
{ return evaluateLog(P, G, L); }

void WaveFunctionComponent::mw_evaluateGL(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                          const RefVectorWithLeader<ParticleSet>& p_list,
                                          const RefVector<ParticleSet::ParticleGradient>& G_list,
                                          const RefVector<ParticleSet::ParticleLaplacian>& L_list,
                                          bool fromscratch) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    wfc_list[iw].evaluateGL(p_list[iw], G_list[iw], L_list[iw], fromscratch);
}

void WaveFunctionComponent::extractOptimizableObjectRefs(UniqueOptObjRefs&)
{
  if (isOptimizable())
    throw std::logic_error("Bug!! " + getClassName() +
                           "::extractOptimizableObjectRefs "
                           "must be overloaded when the WFC is optimizable.");
}

void WaveFunctionComponent::checkOutVariables(const OptVariables& active)
{
  if (isOptimizable())
    throw std::logic_error("Bug!! " + getClassName() +
                           "::checkOutVariables "
                           "must be overloaded when the WFC is optimizable.");
}

void WaveFunctionComponent::mw_evaluateMultiParticleMove(
    const RefVectorWithLeader<WaveFunctionComponent>&,
    const RefVectorWithLeader<ParticleSet>&,
    const MCMultiParticleMoves<CoordsType::POS>&,
    std::vector<LogValue>&,
    const RefVector<ParticleSet::ParticleGradient>&,
    const RefVector<ParticleSet::ParticleLaplacian>&) const
{
  throw std::runtime_error(getClassName() +
                           " does not support atomic selected-electron proposals");
}

void WaveFunctionComponent::mw_accept_rejectMultiParticleMove(
    const RefVectorWithLeader<WaveFunctionComponent>&,
    const RefVectorWithLeader<ParticleSet>&,
    const MCMultiParticleMoves<CoordsType::POS>&,
    const std::vector<bool>&) const
{
  throw std::runtime_error(getClassName() +
                           " does not support atomic selected-electron proposals");
}

void WaveFunctionComponent::evaluateDerivativesWF(ParticleSet& P,
                                                  const OptVariables& active,
                                                  Vector<ValueType>& dlogpsi)
{ throw std::runtime_error("WaveFunctionComponent::evaluateDerivativesWF is not implemented by " + getClassName()); }

void WaveFunctionComponent::mw_evaluateParameterDerivatives(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const OptVariables& optvars,
    RecordArray<ValueType>& dlogpsi,
    RecordArray<ValueType>& dhpsioverpsi) const
{
  assert(this == &wfc_list.getLeader());
  if (wfc_list.size() != p_list.size() || dlogpsi.getNumOfEntries() != wfc_list.size() ||
      dhpsioverpsi.getNumOfEntries() != wfc_list.size() ||
      dlogpsi.getNumOfParams() != dhpsioverpsi.getNumOfParams())
    throw std::invalid_argument("WaveFunctionComponent batched derivative inputs have inconsistent shapes");

  const int parameter_count = dlogpsi.getNumOfParams();
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    Vector<ValueType> score(dlogpsi[walker], parameter_count);
    Vector<ValueType> kinetic_response(dhpsioverpsi[walker], parameter_count);
    wfc_list[walker].evaluateDerivatives(p_list[walker], optvars, score, kinetic_response);
  }
}

void WaveFunctionComponent::mw_evaluateParameterDerivativesWF(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const OptVariables& optvars,
    RecordArray<ValueType>& dlogpsi) const
{
  assert(this == &wfc_list.getLeader());
  if (wfc_list.size() != p_list.size() || dlogpsi.getNumOfEntries() != wfc_list.size())
    throw std::invalid_argument("WaveFunctionComponent batched score inputs have inconsistent shapes");

  const int parameter_count = dlogpsi.getNumOfParams();
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    Vector<ValueType> score(dlogpsi[walker], parameter_count);
    wfc_list[walker].evaluateDerivativesWF(p_list[walker], optvars, score);
  }
}

/*@todo makeClone should be a pure virtual function
 */
std::unique_ptr<WaveFunctionComponent> WaveFunctionComponent::makeClone(ParticleSet& tpq) const
{
  APP_ABORT("Implement WaveFunctionComponent::makeClone " + getClassName() + " class.");
  return std::unique_ptr<WaveFunctionComponent>();
}

WaveFunctionComponent::RealType WaveFunctionComponent::KECorrection() { return 0; }

void WaveFunctionComponent::evaluateRatiosAlltoOne(ParticleSet& P, std::vector<ValueType>& ratios)
{
  assert(P.getTotalNum() == ratios.size());
  for (int i = 0; i < P.getTotalNum(); ++i)
    ratios[i] = ratio(P, i);
}

void WaveFunctionComponent::evaluateRatios(const VirtualParticleSet& P, std::vector<ValueType>& ratios)
{
  std::ostringstream o;
  o << "WaveFunctionComponent::evaluateRatios is not implemented by " << getClassName();
  APP_ABORT(o.str());
}

void WaveFunctionComponent::evaluateSpinorRatios(const VirtualParticleSet& P,
                                                 const std::pair<ValueVector, ValueVector>& spinor_multiplier,
                                                 std::vector<ValueType>& ratios)
{ evaluateRatios(P, ratios); }

void WaveFunctionComponent::mw_evaluateRatios(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                              const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
                                              std::vector<std::vector<ValueType>>& ratios) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    wfc_list[iw].evaluateRatios(vp_list[iw], ratios[iw]);
}

void WaveFunctionComponent::mw_evaluateSpinorRatios(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
    const RefVector<std::pair<ValueVector, ValueVector>>& spinor_multiplier_list,
    std::vector<std::vector<ValueType>>& ratios) const
{ mw_evaluateRatios(wfc_list, vp_list, ratios); }

void WaveFunctionComponent::mw_evaluateSpinorRatios_serialized(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
    const RefVector<std::pair<ValueVector, ValueVector>>& spinor_multiplier_list,
    std::vector<std::vector<ValueType>>& ratios) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    wfc_list[iw].evaluateSpinorRatios(vp_list[iw], spinor_multiplier_list[iw], ratios[iw]);
}

void WaveFunctionComponent::evaluateDerivRatios(const VirtualParticleSet& VP,
                                                const OptVariables& optvars,
                                                std::vector<ValueType>& ratios,
                                                Matrix<ValueType>& dratios)
{
  //default is only ratios and zero derivatives
  evaluateRatios(VP, ratios);
}

void WaveFunctionComponent::evaluateDerivRatiosWeighted(
    const VirtualParticleSet& VP,
    const OptVariables& optvars,
    const std::vector<ValueType>& total_weights,
    ParameterDerivativeView weighted_derivatives)
{
  const std::size_t virtual_count = VP.getTotalNum();
  if (total_weights.size() != virtual_count || weighted_derivatives.size < optvars.size_of_active() ||
      (weighted_derivatives.size != 0 && weighted_derivatives.data == nullptr))
    throw std::invalid_argument("WaveFunctionComponent weighted derivative-ratio inputs have inconsistent shapes");

  // Compatibility components retain their established derivative-ratio code.
  // Only this fallback materializes the quadrature-by-parameter matrix.
  std::vector<ValueType> component_ratios(virtual_count);
  Matrix<ValueType> derivative_ratios(virtual_count, weighted_derivatives.size);
  std::fill(derivative_ratios.begin(), derivative_ratios.end(), ValueType(0));
  evaluateDerivRatios(VP, optvars, component_ratios, derivative_ratios);

  for (std::size_t virtual_index = 0; virtual_index < virtual_count; ++virtual_index)
    for (std::size_t parameter = 0; parameter < weighted_derivatives.size; ++parameter)
      weighted_derivatives[parameter] += total_weights[virtual_index] * derivative_ratios(virtual_index, parameter);
}

void WaveFunctionComponent::mw_evaluateDerivRatiosWeighted(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
    const OptVariables& optvars,
    const RefVector<const std::vector<ValueType>>& total_weights,
    const std::vector<ParameterDerivativeView>& weighted_derivatives) const
{
  assert(this == &wfc_list.getLeader());
  if (wfc_list.size() != vp_list.size() || total_weights.size() != wfc_list.size() ||
      weighted_derivatives.size() != wfc_list.size())
    throw std::invalid_argument("WaveFunctionComponent batched weighted reductions have inconsistent sizes");

  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    wfc_list[walker].evaluateDerivRatiosWeighted(vp_list[walker], optvars, total_weights[walker],
                                                 weighted_derivatives[walker]);
}

void WaveFunctionComponent::evaluateSpinorDerivRatios(const VirtualParticleSet& VP,
                                                      const std::pair<ValueVector, ValueVector>& spinor_multiplier,
                                                      const OptVariables& optvars,
                                                      std::vector<ValueType>& ratios,
                                                      Matrix<ValueType>& dratios)
{ evaluateDerivRatios(VP, optvars, ratios, dratios); }

void WaveFunctionComponent::registerTWFFastDerivWrapper(const ParticleSet& P, TWFFastDerivWrapper& twf) const
{
  std::ostringstream o;
  o << "WaveFunctionComponent::registerTWFFastDerivWrapper is not implemented by " << getClassName();
  APP_ABORT(o.str());
}

template void WaveFunctionComponent::mw_evalGrad<CoordsType::POS>(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int iat,
    TWFGrads<CoordsType::POS>& grad_now) const;
template void WaveFunctionComponent::mw_evalGrad<CoordsType::POS_SPIN>(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int iat,
    TWFGrads<CoordsType::POS_SPIN>& grad_now) const;
template void WaveFunctionComponent::mw_ratioGrad<CoordsType::POS>(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int iat,
    std::vector<PsiValue>& ratios,
    TWFGrads<CoordsType::POS>& grad_new) const;
template void WaveFunctionComponent::mw_ratioGrad<CoordsType::POS_SPIN>(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int iat,
    std::vector<PsiValue>& ratios,
    TWFGrads<CoordsType::POS_SPIN>& grad_new) const;

} // namespace qmcplusplus
