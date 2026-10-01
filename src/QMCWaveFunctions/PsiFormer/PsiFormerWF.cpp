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
#include "io/hdf/hdf_archive.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qmcplusplus
{

/** Shared native model protected at the optimizer/evaluator synchronization
 * boundary. Component clones retain only walker-local move state. */
class PsiFormerSharedState
{
public:
  /// Load the model that all clones of one PsiFormer component will share.
  PsiFormerSharedState(const std::string& parameters, const std::string& configuration)
      : model(parameters, configuration)
  {}

  mutable std::shared_mutex mutex;
  pf::PsiFormer model;
};

namespace
{
constexpr std::array<int, 3> PERSISTENCE_VERSION{1, 0, 0};

/// Build a stable, compact VariableSet name from a component and canonical flat index.
std::string makeParameterName(const std::string& component_name, std::size_t flat_index, std::size_t width)
{
  std::ostringstream name;
  name << component_name << "_pf_" << std::setfill('0') << std::setw(width) << flat_index;
  return name.str();
}

/// Convert canonical native indices to an explicitly sized HDF5 integer type.
std::vector<std::uint64_t> persistIndices(const std::vector<std::size_t>& indices)
{
  return std::vector<std::uint64_t>(indices.begin(), indices.end());
}

/// Read a rank-one HDF5 dataset after determining its on-disk extent.
template<class T>
std::vector<T> readVector(hdf_archive& input, const std::string& name)
{
  std::vector<int> shape;
  if (!input.getShape<T>(name, shape) || shape.size() != 1 || shape[0] < 0)
    throw std::runtime_error("PsiFormer VP dataset " + name + " is not a rank-one array");

  std::vector<T> values(shape[0]);
  input.read(values, name);
  return values;
}

/// Require exact immutable metadata equality before accepting a model payload.
template<class T>
void requireEqual(const std::vector<T>& actual, const std::vector<T>& expected, const std::string& description)
{
  if (actual != expected)
    throw std::runtime_error("PsiFormer VP " + description + " does not match the configured model");
}
} // namespace

// Load the exported model and create the selected local-to-flat parameter map.
PsiFormerWF::PsiFormerWF(std::string name,
                         std::string parameters,
                         std::string configuration,
                         bool enable_optimization,
                         std::vector<std::size_t> selected_flat_indices,
                         bool optimize_all)
    : WaveFunctionComponent(name),
      OptimizableObject(name),
      model_state_(std::make_shared<PsiFormerSharedState>(parameters, configuration)),
      selected_flat_indices_(std::move(selected_flat_indices)),
      optimization_enabled_(enable_optimization),
      optimize_all_(optimize_all)
{
  if (!optimization_enabled_ && !selected_flat_indices_.empty())
    throw std::invalid_argument("PsiFormer optimize_indices requires optimize=yes");
  if (optimize_all_ && !optimization_enabled_)
    throw std::invalid_argument("PsiFormer optimize_scope=all requires optimize=yes");
  if (optimize_all_ && !selected_flat_indices_.empty())
    throw std::invalid_argument("PsiFormer optimize_scope=all cannot be combined with optimize_indices");

  pf::Parameters& parameters_ref = model_state_->model.p;
  if (parameters_ref.size() == 0)
    throw std::invalid_argument("PsiFormer parameter export contains no scalar values");
  if (optimize_all_)
  {
    selected_flat_indices_.resize(parameters_ref.size());
    std::iota(selected_flat_indices_.begin(), selected_flat_indices_.end(), std::size_t{0});
  }
  if (optimization_enabled_ && selected_flat_indices_.empty())
    throw std::invalid_argument("PsiFormer selected-parameter optimization requires at least one flat index");

  if (!optimize_all_)
  {
    std::sort(selected_flat_indices_.begin(), selected_flat_indices_.end());
    if (std::adjacent_find(selected_flat_indices_.begin(), selected_flat_indices_.end()) !=
        selected_flat_indices_.end())
      throw std::invalid_argument("PsiFormer optimize_indices contains a duplicate flat index");
  }

  observed_parameter_version_ = parameters_ref.version();
  const std::size_t name_width = std::to_string(parameters_ref.size() - 1).size();
  std::vector<OptVariables::pair_type> selected_parameters;
  selected_parameters.reserve(selected_flat_indices_.size());
  for (std::size_t flat_index : selected_flat_indices_)
  {
    if (!optimize_all_)
      parameters_ref.layout_for_flat_index(flat_index);
    selected_parameters.emplace_back(makeParameterName(WaveFunctionComponent::getName(), flat_index, name_width),
                                     parameters_ref.flat_values()[flat_index]);
  }
  myVars.insertBulk(std::move(selected_parameters), true, optimize::OTHER_P);
}

// Copy clone-local accepted state while deliberately discarding an in-flight proposal.
PsiFormerWF::PsiFormerWF(const PsiFormerWF& other)
    : WaveFunctionComponent(other),
      OptimizableObject(other),
      model_state_(other.model_state_),
      selected_flat_indices_(other.selected_flat_indices_),
      optimization_enabled_(other.optimization_enabled_),
      optimize_all_(other.optimize_all_),
      observed_parameter_version_(other.observed_parameter_version_),
      restore_validation_pending_(other.restore_validation_pending_),
      accepted_value_valid_(other.accepted_value_valid_),
      current_sign_(other.current_sign_)
{
  std::shared_lock state_lock(model_state_->mutex);
  synchronizeParameterVersion(model_state_->model.p.version());
  for (std::size_t local_index = 0; local_index < selected_flat_indices_.size(); ++local_index)
    myVars[local_index] = model_state_->model.p.flat_values()[selected_flat_indices_[local_index]];

  proposed_sign_      = 1.0;
  proposed_log_value_ = LogValue(0);
  has_proposal_       = false;
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
  if (!optimization_enabled_)
    return;

  std::shared_lock state_lock(model_state_->mutex);
  synchronizeParameterVersion(model_state_->model.p.version());
  for (std::size_t local_index = 0; local_index < selected_flat_indices_.size(); ++local_index)
    myVars[local_index] = model_state_->model.p.flat_values()[selected_flat_indices_[local_index]];
  active.insertFrom(myVars);
}

// Cache the global active index corresponding to each selected local parameter.
void PsiFormerWF::checkOutVariables(const OptVariables& active)
{
  if (optimization_enabled_)
    myVars.getIndex(active);
}

// Clear all cached values derived from an older parameter vector.
void PsiFormerWF::invalidateParameterCaches(std::size_t parameter_version)
{
  current_sign_               = 1.0;
  proposed_sign_              = 1.0;
  log_value_                  = LogValue(0);
  proposed_log_value_         = LogValue(0);
  has_proposal_               = false;
  accepted_value_valid_       = false;
  observed_parameter_version_ = parameter_version;
}

// Lazily invalidate clone-local caches after another clone updates the model.
void PsiFormerWF::synchronizeParameterVersion(std::size_t parameter_version)
{
  if (observed_parameter_version_ != parameter_version)
    invalidateParameterCaches(parameter_version);
}

// Apply a validated selected-parameter or complete-vector update at an exclusive model barrier.
void PsiFormerWF::resetParametersExclusive(const OptVariables& active)
{
  if (!optimization_enabled_)
    return;

  // Full scope is already in canonical flat order. Avoid materializing and
  // sorting redundant local/flat index vectors for every optimizer step.
  if (optimize_all_)
  {
    std::vector<double> active_values;
    active_values.reserve(selected_flat_indices_.size());
    for (std::size_t local_index = 0; local_index < selected_flat_indices_.size(); ++local_index)
    {
      const int global_index = myVars.where(local_index);
      if (global_index < 0)
        throw std::runtime_error("PsiFormer optimize_scope=all requires every model parameter to remain active");
      if (global_index >= active.size())
        throw std::out_of_range("PsiFormer global optimization index is out of range");
      active_values.push_back(std::real(active[global_index]));
    }

    std::size_t parameter_version;
    bool model_changed;
    {
      std::unique_lock state_lock(model_state_->mutex);
      pf::Parameters& parameters = model_state_->model.p;
      if (restore_validation_pending_ && active_values != parameters.flat_values())
        throw std::runtime_error(
            "PsiFormer generic full-network values disagree with the authoritative VP model payload");
      restore_validation_pending_ = false;
      model_changed               = active_values != parameters.flat_values();
      if (model_changed)
        parameters.set_flat_values(active_values);
      parameter_version = parameters.version();
    }

    for (std::size_t parameter = 0; parameter < active_values.size(); ++parameter)
      myVars[parameter] = active_values[parameter];
    if (model_changed || observed_parameter_version_ != parameter_version)
      invalidateParameterCaches(parameter_version);
    return;
  }

  std::vector<std::size_t> active_local_indices;
  std::vector<std::size_t> active_flat_indices;
  std::vector<double> active_values;
  for (std::size_t local_index = 0; local_index < selected_flat_indices_.size(); ++local_index)
  {
    const int global_index = myVars.where(local_index);
    if (global_index < 0)
      continue;
    if (global_index >= active.size())
      throw std::out_of_range("PsiFormer global optimization index is out of range");

    active_local_indices.push_back(local_index);
    active_flat_indices.push_back(selected_flat_indices_[local_index]);
    active_values.push_back(std::real(active[global_index]));
  }

  if (active_values.empty())
    return;

  std::size_t parameter_version;
  bool model_changed = false;
  {
    std::unique_lock state_lock(model_state_->mutex);
    pf::Parameters& parameters = model_state_->model.p;

    // The complete object-specific payload is authoritative on restart. The
    // duplicated generic scalar list must agree before the normal reset path
    // is allowed to continue.
    if (restore_validation_pending_)
    {
      for (std::size_t parameter = 0; parameter < active_values.size(); ++parameter)
        if (active_values[parameter] != parameters.flat_values()[active_flat_indices[parameter]])
          throw std::runtime_error(
              "PsiFormer generic selected values disagree with the authoritative VP model payload");
      restore_validation_pending_ = false;
    }

    for (std::size_t parameter = 0; parameter < active_values.size(); ++parameter)
      model_changed =
          model_changed || active_values[parameter] != parameters.flat_values()[active_flat_indices[parameter]];

    if (model_changed)
      parameters.set_flat_values(active_flat_indices, active_values);
    parameter_version = parameters.version();
  }

  for (std::size_t parameter = 0; parameter < active_values.size(); ++parameter)
    myVars[active_local_indices[parameter]] = active_values[parameter];

  if (model_changed || observed_parameter_version_ != parameter_version)
    invalidateParameterCaches(parameter_version);
}

// Store a complete, self-identifying model payload in the optimizer VP file.
void PsiFormerWF::writeVariationalParameters(hdf_archive& output)
{
  if (!optimization_enabled_)
    return;

  std::shared_lock state_lock(model_state_->mutex);
  const pf::PsiFormer& model = model_state_->model;

  output.push("PsiFormer");
  output.push(OptimizableObject::getName());

  const std::vector<int> format_version(PERSISTENCE_VERSION.begin(), PERSISTENCE_VERSION.end());
  const std::vector<std::uint64_t> parameter_count{model.p.size()};
  const std::vector<std::uint64_t> spin_counts{model.cfg.nup, model.cfg.ndown};
  const std::vector<std::uint64_t> architecture{model.ndet, model.dim, model.heads};
  const std::vector<std::uint64_t> nuclear_shape(model.cfg.nuclei.shape.begin(), model.cfg.nuclei.shape.end());
  const std::vector<std::uint64_t> selected_indices = persistIndices(selected_flat_indices_);
  const std::string layout_fingerprint              = model.p.layout_fingerprint();

  output.write(format_version, "format_version");
  output.write(parameter_count, "parameter_count");
  output.write(layout_fingerprint, "layout_fingerprint");
  output.write(spin_counts, "spin_counts");
  output.write(architecture, "architecture");
  output.write(nuclear_shape, "nuclear_shape");
  output.write(model.cfg.nuclei.x, "nuclear_positions");
  output.write(model.cfg.charges.x, "nuclear_charges");
  output.write(selected_indices, "selected_flat_indices");
  output.write(model.p.flat_values(), "flat_values");

  output.pop();
  output.pop();
}

// Restore a complete model only after validating all identifying metadata.
void PsiFormerWF::readVariationalParameters(hdf_archive& input)
{
  if (!optimization_enabled_)
    return;
  if (!input.is_group("PsiFormer"))
    throw std::runtime_error("PsiFormer VP file has no PsiFormer object group");

  input.push("PsiFormer", false);
  if (!input.is_group(OptimizableObject::getName()))
    throw std::runtime_error("PsiFormer VP file has no group for component " + OptimizableObject::getName());
  input.push(OptimizableObject::getName(), false);

  const std::vector<int> format_version             = readVector<int>(input, "format_version");
  const std::vector<std::uint64_t> parameter_count  = readVector<std::uint64_t>(input, "parameter_count");
  const std::vector<std::uint64_t> spin_counts      = readVector<std::uint64_t>(input, "spin_counts");
  const std::vector<std::uint64_t> architecture     = readVector<std::uint64_t>(input, "architecture");
  const std::vector<std::uint64_t> nuclear_shape    = readVector<std::uint64_t>(input, "nuclear_shape");
  const std::vector<double> nuclear_positions       = readVector<double>(input, "nuclear_positions");
  const std::vector<double> nuclear_charges         = readVector<double>(input, "nuclear_charges");
  const std::vector<std::uint64_t> selected_indices =
      readVector<std::uint64_t>(input, "selected_flat_indices");
  const std::vector<double> flat_values = readVector<double>(input, "flat_values");
  std::string layout_fingerprint;
  input.read(layout_fingerprint, "layout_fingerprint");

  input.pop();
  input.pop();

  const std::vector<int> expected_version(PERSISTENCE_VERSION.begin(), PERSISTENCE_VERSION.end());
  requireEqual(format_version, expected_version, "format version");

  std::size_t parameter_version;
  {
    std::unique_lock state_lock(model_state_->mutex);
    pf::PsiFormer& model = model_state_->model;

    requireEqual(parameter_count, std::vector<std::uint64_t>{model.p.size()}, "parameter count");
    if (layout_fingerprint != model.p.layout_fingerprint())
      throw std::runtime_error("PsiFormer VP layout fingerprint does not match the configured model");
    requireEqual(spin_counts, std::vector<std::uint64_t>{model.cfg.nup, model.cfg.ndown}, "spin populations");
    requireEqual(architecture, std::vector<std::uint64_t>{model.ndet, model.dim, model.heads}, "architecture");
    requireEqual(nuclear_shape,
                 std::vector<std::uint64_t>(model.cfg.nuclei.shape.begin(), model.cfg.nuclei.shape.end()),
                 "nuclear-position shape");
    requireEqual(nuclear_positions, model.cfg.nuclei.x, "nuclear positions");
    requireEqual(nuclear_charges, model.cfg.charges.x, "nuclear charges");
    requireEqual(selected_indices, persistIndices(selected_flat_indices_), "selected flat indices");

    if (flat_values != model.p.flat_values())
      model.p.set_flat_values(flat_values);
    parameter_version = model.p.version();

    for (std::size_t local_index = 0; local_index < selected_flat_indices_.size(); ++local_index)
      myVars[local_index] = model.p.flat_values()[selected_flat_indices_[local_index]];
  }

  invalidateParameterCaches(parameter_version);
  restore_validation_pending_ = true;
}

// Return the parameter version shared by this component and all of its clones.
std::size_t PsiFormerWF::parameterVersion() const
{
  std::shared_lock state_lock(model_state_->mutex);
  return model_state_->model.p.version();
}

// Write a standalone flat parameter file without exposing mutable native storage.
void PsiFormerWF::exportParameters(const std::string& path) const
{
  std::shared_lock state_lock(model_state_->mutex);
  model_state_->model.p.write(path);
}

// Translate QMCPACK particle coordinates and derivative context into a native request.
pf::Result PsiFormerWF::evaluate(const ParticleSet& p,
                                 int active,
                                 bool with_parameter_gradient,
                                 bool with_kinetic_parameter_gradient)
{
  std::shared_lock state_lock(model_state_->mutex);
  pf::PsiFormer& model = model_state_->model;
  synchronizeParameterVersion(model.p.version());

  if (static_cast<std::size_t>(p.getTotalNum()) != model.ne)
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

  return model.evaluate(positions, request);
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
  accepted_value_valid_ = true;
  return log_value_;
}

// Evaluate and cache the wavefunction ratio for one proposed electron position.
PsiFormerWF::PsiValue PsiFormerWF::ratio(ParticleSet& p, int iat)
{
  auto result = evaluate(p, iat);
  if (!accepted_value_valid_)
    throw std::logic_error("PsiFormer ratio requested before evaluateLog for the current parameter version");

  // Cache proposal state so acceptMove can commit it without reevaluating the
  // network.
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
  auto result = evaluate(p, iat);
  if (!accepted_value_valid_)
    throw std::logic_error("PsiFormer ratioGrad requested before evaluateLog for the current parameter version");

  // Evaluate the proposal once and return both its ratio and active-electron
  // gradient.
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
  std::shared_lock state_lock(model_state_->mutex);
  synchronizeParameterVersion(model_state_->model.p.version());
  if (has_proposal_)
  {
    log_value_            = proposed_log_value_;
    current_sign_         = proposed_sign_;
    accepted_value_valid_ = true;
  }
  has_proposal_ = false;
}

// Forget cached proposal state after a rejected move.
void PsiFormerWF::restore(int)
{
  std::shared_lock state_lock(model_state_->mutex);
  synchronizeParameterVersion(model_state_->model.p.version());
  has_proposal_ = false;
}

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

// Copy optimizer mapping and accepted state while sharing the synchronized native model.
std::unique_ptr<WaveFunctionComponent> PsiFormerWF::makeClone(ParticleSet&) const
{
  return std::make_unique<PsiFormerWF>(*this);
}

} // namespace qmcplusplus
