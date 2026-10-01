//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerWF.cpp
 * @brief QMCPACK integration for the native PsiFormer evaluator.
 */
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace qmcplusplus
{
namespace
{
/// Build a stable, compact VariableSet name from a component and canonical flat index.
std::string makeParameterName(const std::string& component_name, std::size_t flat_index, std::size_t width)
{
  std::ostringstream name;
  name << component_name << "_pf_" << std::setfill('0') << std::setw(width) << flat_index;
  return name.str();
}
} // namespace

// Load the exported model and create the selected local-to-flat parameter map.
PsiFormerWF::PsiFormerWF(std::string name,
                         std::string parameters,
                         std::string configuration,
                         bool enable_optimization,
                         std::vector<std::size_t> selected_flat_indices)
    : WaveFunctionComponent(name),
      OptimizableObject(name),
      model_(std::make_shared<pf::PsiFormer>(parameters, configuration)),
      selected_flat_indices_(std::move(selected_flat_indices)),
      optimization_enabled_(enable_optimization)
{
  if (!optimization_enabled_ && !selected_flat_indices_.empty())
    throw std::invalid_argument("PsiFormer optimize_indices requires optimize=yes");
  if (optimization_enabled_ && selected_flat_indices_.empty())
    throw std::invalid_argument("PsiFormer selected-parameter optimization requires at least one flat index");

  std::sort(selected_flat_indices_.begin(), selected_flat_indices_.end());
  if (std::adjacent_find(selected_flat_indices_.begin(), selected_flat_indices_.end()) !=
      selected_flat_indices_.end())
    throw std::invalid_argument("PsiFormer optimize_indices contains a duplicate flat index");

  const std::size_t name_width = std::to_string(model_->p.size() - 1).size();
  for (std::size_t flat_index : selected_flat_indices_)
  {
    model_->p.layout_for_flat_index(flat_index);
    myVars.insert(makeParameterName(WaveFunctionComponent::getName(), flat_index, name_width),
                  model_->p.flat_values()[flat_index], true, optimize::OTHER_P);
  }
}

// Register this object only when the input explicitly enabled optimization.
void PsiFormerWF::extractOptimizableObjectRefs(UniqueOptObjRefs& opt_obj_refs)
{
  if (optimization_enabled_)
    opt_obj_refs.push_back(*this);
}

// Append selected local values to the optimizer's global variable collection.
void PsiFormerWF::checkInVariablesExclusive(OptVariables& active)
{
  if (optimization_enabled_)
    active.insertFrom(myVars);
}

// Cache the global active index corresponding to each selected local parameter.
void PsiFormerWF::checkOutVariables(const OptVariables& active)
{
  if (optimization_enabled_)
    myVars.getIndex(active);
}

// Apply one validated selected-parameter update and invalidate old move state.
void PsiFormerWF::resetParametersExclusive(const OptVariables& active)
{
  if (!optimization_enabled_)
    return;

  std::vector<std::size_t> changed_local_indices;
  std::vector<std::size_t> changed_flat_indices;
  std::vector<double> changed_values;
  for (std::size_t local_index = 0; local_index < selected_flat_indices_.size(); ++local_index)
  {
    const int global_index = myVars.where(local_index);
    if (global_index < 0)
      continue;
    if (global_index >= active.size())
      throw std::out_of_range("PsiFormer global optimization index is out of range");

    changed_local_indices.push_back(local_index);
    changed_flat_indices.push_back(selected_flat_indices_[local_index]);
    changed_values.push_back(std::real(active[global_index]));
  }

  if (changed_values.empty())
    return;

  model_->p.set_flat_values(changed_flat_indices, changed_values);
  for (std::size_t changed = 0; changed < changed_values.size(); ++changed)
    myVars[changed_local_indices[changed]] = changed_values[changed];

  // A proposal or cached accepted log value evaluated with the previous
  // parameter version must not participate in the next move sequence.
  current_sign_       = 1.0;
  proposed_sign_      = 1.0;
  log_value_          = LogValue(0);
  proposed_log_value_ = LogValue(0);
  has_proposal_      = false;
}

// Translate QMCPACK particle coordinates and derivative context into a native request.
pf::Result PsiFormerWF::evaluate(const ParticleSet& p,
                                 int active,
                                 bool with_parameter_gradient,
                                 bool with_kinetic_parameter_gradient) const
{
  if (static_cast<std::size_t>(p.getTotalNum()) != model_->ne)
    throw std::runtime_error("PsiFormerWF electron count differs from exported model");
  if (with_kinetic_parameter_gradient && !with_parameter_gradient)
    throw std::logic_error("PsiFormer kinetic parameter derivatives require log parameter derivatives");

  // For a particle-by-particle proposal, substitute only the active position.
  // All other positions remain at the last accepted ParticleSet coordinates.
  pf::Tensor positions({static_cast<std::size_t>(p.getTotalNum()), 3});
  for (int electron = 0; electron < p.getTotalNum(); ++electron)
  {
    const auto& position = electron == active ? p.activeR(electron) : p.R[electron];
    for (int dimension = 0; dimension < 3; ++dimension)
      positions.x[3 * electron + dimension] = position[dimension];
  }

  pf::EvaluationRequest request;
  if (with_kinetic_parameter_gradient)
    request.parameter_derivatives = pf::ParameterDerivativeRequest::LOG_AND_KINETIC;
  else if (with_parameter_gradient)
    request.parameter_derivatives = pf::ParameterDerivativeRequest::LOG_ONLY;

  std::vector<double> total_log_gradient;
  if (with_kinetic_parameter_gradient)
  {
    total_log_gradient.reserve(3 * p.getTotalNum());
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
      for (int dimension = 0; dimension < 3; ++dimension)
        total_log_gradient.push_back(std::real(p.G[electron][dimension]));
    request.total_log_gradient = &total_log_gradient;
  }

  return model_->evaluate(positions, request);
}

// Evaluate a full accepted configuration and accumulate its spatial derivatives.
PsiFormerWF::LogValue PsiFormerWF::evaluateLog(const ParticleSet& p,
                                               ParticleSet::ParticleGradient& g,
                                               ParticleSet::ParticleLaplacian& l)
{
  auto result   = evaluate(p);
  current_sign_ = result.sign;
  // QMCPACK represents a negative real wavefunction by adding pi to its complex
  // phase.
  log_value_ = LogValue(result.logabs, result.sign < 0 ? M_PI : 0.0);
  for (int electron = 0; electron < p.getTotalNum(); ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      g[electron][dimension] += result.gradient[3 * electron + dimension];
    l[electron] += result.lap_log[electron];
  }
  return log_value_;
}

// Evaluate and cache the wavefunction ratio for one proposed electron position.
PsiFormerWF::PsiValue PsiFormerWF::ratio(ParticleSet& p, int iat)
{
  // Cache proposal state so acceptMove can commit it without reevaluating the
  // network.
  auto result         = evaluate(p, iat);
  proposed_sign_      = result.sign;
  proposed_log_value_ = LogValue(result.logabs, result.sign < 0 ? M_PI : 0.0);
  has_proposal_       = true;
  return (proposed_sign_ / current_sign_) * std::exp(std::real(proposed_log_value_ - log_value_));
}

// Return one accepted electron logarithmic gradient.
PsiFormerWF::GradType PsiFormerWF::evalGrad(ParticleSet& p, int iat)
{
  auto result = evaluate(p);
  GradType gradient;
  for (int dimension = 0; dimension < 3; ++dimension)
    gradient[dimension] = result.gradient[3 * iat + dimension];
  return gradient;
}

// Evaluate a proposed ratio and gradient in one native-model traversal.
PsiFormerWF::PsiValue PsiFormerWF::ratioGrad(ParticleSet& p, int iat, GradType& gradient)
{
  // Evaluate the proposal once and return both its ratio and active-electron
  // gradient.
  auto result         = evaluate(p, iat);
  proposed_sign_      = result.sign;
  proposed_log_value_ = LogValue(result.logabs, result.sign < 0 ? M_PI : 0.0);
  has_proposal_       = true;
  for (int dimension = 0; dimension < 3; ++dimension)
    gradient[dimension] = result.gradient[3 * iat + dimension];
  return (proposed_sign_ / current_sign_) * std::exp(std::real(proposed_log_value_ - log_value_));
}

// Promote cached proposal state to accepted state after a successful move.
void PsiFormerWF::acceptMove(ParticleSet&, int, bool)
{
  if (has_proposal_)
  {
    log_value_    = proposed_log_value_;
    current_sign_ = proposed_sign_;
  }
  has_proposal_ = false;
}

// Forget cached proposal state after a rejected move.
void PsiFormerWF::restore(int) { has_proposal_ = false; }

// Re-evaluate the component because it does not maintain walker-buffer storage.
PsiFormerWF::LogValue PsiFormerWF::updateBuffer(ParticleSet& p, WFBufferType&, bool)
{
  return evaluateLog(p, p.G, p.L);
}

// Return whether a selected local parameter is present in the global active set.
bool PsiFormerWF::hasActiveParameters() const
{
  for (std::size_t local_index = 0; local_index < myVars.size(); ++local_index)
    if (myVars.where(local_index) >= 0)
      return true;
  return false;
}

// Scatter selected flat derivatives into their global QMCPACK entries.
void PsiFormerWF::addSelectedGradient(const std::vector<double>& flat_gradient, Vector<ValueType>& output) const
{
  for (std::size_t local_index = 0; local_index < selected_flat_indices_.size(); ++local_index)
  {
    const int global_index = myVars.where(local_index);
    if (global_index < 0)
      continue;
    if (global_index >= output.size())
      throw std::out_of_range("PsiFormer derivative output index is out of range");

    const std::size_t flat_index = selected_flat_indices_[local_index];
    if (flat_index >= flat_gradient.size())
      throw std::out_of_range("PsiFormer native derivative is missing a selected flat index");
    output[global_index] += ValueType(flat_gradient[flat_index]);
  }
}

// Add only score derivatives, avoiding the mixed coordinate-jet reverse used for kinetic derivatives.
void PsiFormerWF::evaluateDerivativesWF(ParticleSet& p, const OptVariables&, Vector<ValueType>& dlogpsi)
{
  if (!optimization_enabled_ || !hasActiveParameters())
    return;

  const pf::Result result = evaluate(p, -1, true, false);
  addSelectedGradient(result.param_gradient, dlogpsi);
}

// Add score and component kinetic derivatives using the complete TrialWaveFunction gradient in P.G.
void PsiFormerWF::evaluateDerivatives(ParticleSet& p,
                                      const OptVariables&,
                                      Vector<ValueType>& dlogpsi,
                                      Vector<ValueType>& dhpsioverpsi)
{
  if (!optimization_enabled_ || !hasActiveParameters())
    return;

  const pf::Result result = evaluate(p, -1, true, true);
  addSelectedGradient(result.param_gradient, dlogpsi);
  addSelectedGradient(result.local_energy_param_gradient, dhpsioverpsi);
}

// Copy move and optimizer-index state while sharing the synchronized native model.
std::unique_ptr<WaveFunctionComponent> PsiFormerWF::makeClone(ParticleSet&) const
{
  return std::make_unique<PsiFormerWF>(*this);
}

} // namespace qmcplusplus
