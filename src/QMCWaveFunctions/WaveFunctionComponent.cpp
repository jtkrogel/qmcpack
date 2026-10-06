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
#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <typeinfo>

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
} // namespace

// for return types
using PsiValue = WaveFunctionComponent::PsiValue;

WaveFunctionComponent::WaveFunctionComponent(const std::string& obj_name)
    : UpdateMode(ORB_WALKER), Bytes_in_WFBuffer(0), my_name_(obj_name), log_value_(0.0)
{}

WaveFunctionComponent::~WaveFunctionComponent() = default;

std::unique_ptr<wftrain::StreamingDerivativeOperator>
WaveFunctionComponent::makeStreamingDerivativeOperator(
    const RefVectorWithLeader<WaveFunctionComponent>&,
    const RefVectorWithLeader<ParticleSet>&,
    std::size_t,
    std::size_t,
    std::size_t) const
{
  throw std::runtime_error("Wavefunction component does not support streaming parameter derivatives");
}

WaveFunctionComponent::EvaluationStamp WaveFunctionComponent::EvaluationStamp::versioned(
    const void* source_identity,
    std::uint64_t version)
{
  if (source_identity == nullptr)
    throw std::invalid_argument("A versioned wavefunction evaluation stamp requires a non-null source identity.");
  return EvaluationStamp(source_identity, version);
}

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

WaveFunctionComponent::EvaluationStamp WaveFunctionComponent::mw_evaluateVirtualRatios(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
    const VirtualParticleBatch& batch,
    std::vector<ValueType>& ratios) const
{
  if (this != std::addressof(wfc_list.getLeader()))
    throw std::invalid_argument(
        "WaveFunctionComponent::mw_evaluateVirtualRatios must be invoked on the component-list leader.");
  if (wfc_list.size() != batch.walkerCount() || p_list.size() != batch.walkerCount() ||
      vp_scratch_list.size() != batch.walkerCount())
    throw std::invalid_argument(
        "WaveFunctionComponent::mw_evaluateVirtualRatios list sizes do not match the descriptor walker count.");

  batch.validateOutputExtent(ratios.size());
  batch.validateFor(p_list);

  for (std::size_t walker = 0; walker < batch.walkerCount(); ++walker)
  {
    if (typeid(wfc_list[walker]) != typeid(wfc_list.getLeader()))
      throw std::invalid_argument(
          "WaveFunctionComponent::mw_evaluateVirtualRatios component clones have different dynamic types.");
    for (std::size_t other = 0; other < walker; ++other)
    {
      if (std::addressof(wfc_list[walker]) == std::addressof(wfc_list[other]))
        throw std::invalid_argument(
            "WaveFunctionComponent::mw_evaluateVirtualRatios requires one distinct component clone per walker.");
      if (std::addressof(vp_scratch_list[walker]) == std::addressof(vp_scratch_list[other]))
        throw std::invalid_argument(
            "WaveFunctionComponent::mw_evaluateVirtualRatios requires one distinct scratch object per walker.");
    }

    const ParticleSet* scratch_as_particles = static_cast<const ParticleSet*>(std::addressof(vp_scratch_list[walker]));
    for (std::size_t reference = 0; reference < batch.walkerCount(); ++reference)
      if (scratch_as_particles == std::addressof(p_list[reference]))
        throw std::invalid_argument(
            "WaveFunctionComponent::mw_evaluateVirtualRatios scratch objects must not alias reference walkers.");
    if (vp_scratch_list[walker].isSpinor() != p_list[walker].isSpinor())
      throw std::invalid_argument(
          "WaveFunctionComponent::mw_evaluateVirtualRatios reference and scratch spinor modes do not match.");
  }

  std::vector<ValueType> staged_ratios(batch.size());
  for (std::size_t segment_index = 0; segment_index < batch.segmentCount(); ++segment_index)
  {
    const VirtualParticleBatch::Slice slice = batch.slice(segment_index);
    VirtualParticleSet& scratch              = vp_scratch_list[slice.walkerId()];
    scratch.makeMovesAbsolute(p_list[slice.walkerId()], slice.electronId(), slice.positions(), slice.isOnSphere(),
                              slice.sourceCenterId());

    std::vector<ValueType> segment_ratios(slice.size());
    wfc_list[slice.walkerId()].evaluateRatios(scratch, segment_ratios);
    if (segment_ratios.size() != slice.size())
      throw std::runtime_error(
          "WaveFunctionComponent::evaluateRatios changed the flattened virtual-ratio output extent.");
    std::copy(segment_ratios.begin(), segment_ratios.end(), staged_ratios.begin() + slice.flatOffset());
  }

  ratios.swap(staged_ratios);
  return EvaluationStamp{};
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
  if (total_weights.size() != virtual_count || weighted_derivatives.size < requiredDerivativeExtent(optvars) ||
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

WaveFunctionComponent::EvaluationStamp WaveFunctionComponent::mw_evaluateVirtualDerivRatiosWeighted(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
    const VirtualParticleBatch& batch,
    const OptVariables& optvars,
    const std::vector<ValueType>& total_weights,
    const std::vector<ParameterDerivativeView>& weighted_derivatives) const
{
  if (this != std::addressof(wfc_list.getLeader()))
    throw std::invalid_argument(
        "WaveFunctionComponent::mw_evaluateVirtualDerivRatiosWeighted must be invoked on the component-list "
        "leader.");
  if (wfc_list.size() != batch.walkerCount() || p_list.size() != batch.walkerCount() ||
      vp_scratch_list.size() != batch.walkerCount() || weighted_derivatives.size() != batch.walkerCount())
    throw std::invalid_argument(
        "WaveFunctionComponent::mw_evaluateVirtualDerivRatiosWeighted list sizes do not match the descriptor "
        "walker count.");

  batch.validateOutputExtent(total_weights.size());
  batch.validateFor(p_list);

  const std::size_t derivative_width = weighted_derivatives.empty() ? 0 : weighted_derivatives.front().size;
  if (!weighted_derivatives.empty() && derivative_width < requiredDerivativeExtent(optvars))
    throw std::invalid_argument(
        "WaveFunctionComponent::mw_evaluateVirtualDerivRatiosWeighted derivative rows are too short.");

  for (std::size_t walker = 0; walker < batch.walkerCount(); ++walker)
  {
    if (weighted_derivatives[walker].size != derivative_width ||
        (derivative_width != 0 && weighted_derivatives[walker].data == nullptr))
      throw std::invalid_argument(
          "WaveFunctionComponent::mw_evaluateVirtualDerivRatiosWeighted derivative rows have inconsistent "
          "shapes.");
    if (typeid(wfc_list[walker]) != typeid(wfc_list.getLeader()))
      throw std::invalid_argument(
          "WaveFunctionComponent::mw_evaluateVirtualDerivRatiosWeighted component clones have different dynamic "
          "types.");
    for (std::size_t other = 0; other < walker; ++other)
    {
      if (std::addressof(wfc_list[walker]) == std::addressof(wfc_list[other]))
        throw std::invalid_argument(
            "WaveFunctionComponent::mw_evaluateVirtualDerivRatiosWeighted requires one distinct component clone "
            "per walker.");
      if (std::addressof(vp_scratch_list[walker]) == std::addressof(vp_scratch_list[other]))
        throw std::invalid_argument(
            "WaveFunctionComponent::mw_evaluateVirtualDerivRatiosWeighted requires one distinct scratch object "
            "per walker.");
    }

    const ParticleSet* scratch_as_particles = static_cast<const ParticleSet*>(std::addressof(vp_scratch_list[walker]));
    for (std::size_t reference = 0; reference < batch.walkerCount(); ++reference)
      if (scratch_as_particles == std::addressof(p_list[reference]))
        throw std::invalid_argument(
            "WaveFunctionComponent::mw_evaluateVirtualDerivRatiosWeighted scratch objects must not alias "
            "reference walkers.");
    if (vp_scratch_list[walker].isSpinor() != p_list[walker].isSpinor())
      throw std::invalid_argument(
          "WaveFunctionComponent::mw_evaluateVirtualDerivRatiosWeighted reference and scratch spinor modes do not "
          "match.");
  }

  if (derivative_width != 0 && batch.walkerCount() > std::numeric_limits<std::size_t>::max() / derivative_width)
    throw std::length_error(
        "WaveFunctionComponent::mw_evaluateVirtualDerivRatiosWeighted derivative staging extent overflows.");
  const std::size_t staged_extent = batch.walkerCount() * derivative_width;
  std::vector<ValueType> staged_derivatives(staged_extent, ValueType(0));
  std::vector<ParameterDerivativeView> staged_views;
  staged_views.reserve(batch.walkerCount());
  for (std::size_t walker = 0; walker < batch.walkerCount(); ++walker)
    staged_views.push_back(
        {derivative_width == 0 ? nullptr : staged_derivatives.data() + walker * derivative_width, derivative_width});

  for (std::size_t segment_index = 0; segment_index < batch.segmentCount(); ++segment_index)
  {
    const VirtualParticleBatch::Slice slice = batch.slice(segment_index);
    VirtualParticleSet& scratch              = vp_scratch_list[slice.walkerId()];
    scratch.makeMovesAbsolute(p_list[slice.walkerId()], slice.electronId(), slice.positions(), slice.isOnSphere(),
                              slice.sourceCenterId());

    std::vector<ValueType> segment_weights(slice.size());
    std::copy_n(total_weights.begin() + slice.flatOffset(), slice.size(), segment_weights.begin());
    wfc_list[slice.walkerId()].evaluateDerivRatiosWeighted(scratch, optvars, segment_weights,
                                                           staged_views[slice.walkerId()]);
  }

  for (std::size_t walker = 0; walker < batch.walkerCount(); ++walker)
    for (std::size_t parameter = 0; parameter < derivative_width; ++parameter)
      weighted_derivatives[walker][parameter] += staged_views[walker][parameter];
  return EvaluationStamp{};
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
