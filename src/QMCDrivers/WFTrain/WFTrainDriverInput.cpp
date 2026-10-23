//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include "WFTrainDriverInput.h"

#include "Message/UniformCommunicateError.h"
#include "OhmmsData/XMLParsingString.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace qmcplusplus
{
namespace
{

constexpr std::array<std::string_view, 95> allowed_controls{
    "gradient_estimator",
    "defaults", "kfac_gradient_convention", "kfac_update_order", "kfac_schedule_arithmetic",
    "iterations", "checkpoint_interval", "warmupsteps", "momentum_decay", "rms_decay", "lm_overlap_shift", "lm_normalization_xi",
    "lm_maximum_subspace_dimension", "lm_inner_iterations", "lm_inner_restart_length",
    "lm_inner_relative_tolerance", "kfac_fallback_damping",
    "kfac_objective", "kfac_clip_mode", "kfac_clip_width", "kfac_clip_quantile", "kfac_exclude_width",
    "kfac_complex_clip_width",
    "kfac_curvature", "kfac_damping_policy", "kfac_constraint", "kfac_norm_constraint",
    "kfac_persist_state", "kfac_ema_decay", "kfac_inverse_period", "kfac_learning_rate_decay",
    "kfac_schedule_precision", "kfac_momentum", "kfac_burn_in", "kfac_curvature_period",
    "line_search", "update_scale", "weight_decay", "weight_decay_scope",
    "line_search_minimum_ess_fraction", "line_search_maximum_weight", "line_search_confidence",
    "line_search_warmup_steps", "line_search_sampling_steps",
    "objective", "optimizer", "parameter_chunk_size", "learning_rate", "adam_beta1",
    "adam_beta2", "epsilon", "damping", "damping_multiplier", "minimum_retry_damping",
    "maximum_damping_attempts", "maximum_update_norm", "maximum_metric_norm", "center_shift",
    "objective_tolerance", "spring_history_decay", "spring_maximum_update_norm_squared",
    "krylov_relative_tolerance", "krylov_absolute_tolerance", "krylov_maximum_iterations",
    "krylov_maximum_operator_applications", "krylov_maximum_seconds", "krylov_stagnation_window",
    "krylov_minimum_relative_improvement", "krylov_curvature_tolerance", "krylov_imaginary_tolerance",
    "maximum_retained_bytes", "maximum_resident_bytes", "checkpoint_path", "restart_path", "provider",
    "complex_gpu_device_bytes", "complex_gpu_host_bytes", "complex_gpu_kinetic_targets", "complex_gpu_maximum_matrix_work",
    "teacher", "overlap_target", "overlap_weighting", "target_provenance", "orbital_target", "orbital_sampling", "orbital_normalization",
    "minimum_fidelity", "minimum_effective_samples", "pii_shift_mode",
    "pii_explicit_tau", "pii_energy_undershoot", "maximum_damping_retries",
    "gmres_restart_length", "legacy_maximum_parameters", "legacy_derivative_scratch_bytes"};

/** Return an XML attribute value without changing its spelling or whitespace. */
std::string readAttribute(xmlNodePtr node, const char* name)
{
  xmlChar* raw_value = xmlGetProp(node, reinterpret_cast<const xmlChar*>(name));
  if (raw_value == nullptr)
    return {};
  std::string value = castXMLCharToChar(raw_value);
  xmlFree(raw_value);
  return value;
}

/** Parse a positive base-ten size while checking overflow. */
std::size_t parsePositiveSize(const std::string& text, const std::string& attribute)
{
  if (text.empty() ||
      !std::all_of(text.begin(), text.end(), [](unsigned char character) { return std::isdigit(character); }))
    throw UniformCommunicateError("WFTrainDriverInput: " + attribute + " must be a positive base-ten integer");

  std::size_t value = 0;
  for (const char digit : text)
  {
    const std::size_t numeric_digit = static_cast<std::size_t>(digit - '0');
    if (value > (std::numeric_limits<std::size_t>::max() - numeric_digit) / 10)
      throw UniformCommunicateError("WFTrainDriverInput: " + attribute + " overflows size_t");
    value = value * 10 + numeric_digit;
  }
  if (value == 0)
    throw UniformCommunicateError("WFTrainDriverInput: " + attribute + " must be positive");
  return value;
}

/** Parse a nonnegative base-ten size while checking overflow. */
std::size_t parseNonnegativeSize(const std::string& text, const std::string& attribute)
{
  if (text.empty() ||
      !std::all_of(text.begin(), text.end(), [](unsigned char character) { return std::isdigit(character); }))
    throw UniformCommunicateError("WFTrainDriverInput: " + attribute + " must be a nonnegative base-ten integer");

  std::size_t value = 0;
  for (const char digit : text)
  {
    const std::size_t numeric_digit = static_cast<std::size_t>(digit - '0');
    if (value > (std::numeric_limits<std::size_t>::max() - numeric_digit) / 10)
      throw UniformCommunicateError("WFTrainDriverInput: " + attribute + " overflows size_t");
    value = value * 10 + numeric_digit;
  }
  return value;
}

/** Parse one finite floating-point value and reject trailing text. */
double parseFiniteReal(const std::string& text, const std::string& attribute)
{
  if (text.empty())
    throw UniformCommunicateError("WFTrainDriverInput: " + attribute + " must be a finite real number");

  errno             = 0;
  char* parse_end    = nullptr;
  const double value = std::strtod(text.c_str(), &parse_end);
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  static_assert(std::numeric_limits<double>::is_iec559);
  std::uint64_t value_bits;
  std::memcpy(&value_bits, &value, sizeof(value_bits));
  constexpr std::uint64_t exponent_mask = UINT64_C(0x7ff0000000000000);
  const bool is_finite = (value_bits & exponent_mask) != exponent_mask;
  if (errno == ERANGE || parse_end != text.c_str() + text.size() || !is_finite)
    throw UniformCommunicateError("WFTrainDriverInput: " + attribute + " must be a finite real number");
  return value;
}

/** Parse a finite real constrained to the half-open unit interval. */
double parseDecay(const std::string& text, const std::string& attribute)
{
  const double value = parseFiniteReal(text, attribute);
  if (value < 0.0 || value >= 1.0)
    throw UniformCommunicateError("WFTrainDriverInput: " + attribute + " must lie in [0, 1)");
  return value;
}

/** Parse a finite nonnegative real. */
double parseNonnegativeReal(const std::string& text, const std::string& attribute)
{
  const double value = parseFiniteReal(text, attribute);
  if (value < 0.0)
    throw UniformCommunicateError("WFTrainDriverInput: " + attribute + " must be nonnegative");
  return value;
}

/** Reject controls that do not apply to the selected optimizer. */
void rejectControls(const std::unordered_set<std::string>& seen_controls,
                      std::initializer_list<std::string_view> attributes,
                      const std::string& optimizer)
{
  for (const std::string_view attribute : attributes)
    if (seen_controls.find(std::string(attribute)) != seen_controls.end())
      throw UniformCommunicateError("WFTrainDriverInput: " + std::string(attribute) +
                                    " is not valid for optimizer=\"" + optimizer + "\"");
}

/** Remove XML indentation around a parameter value, preserving internal whitespace. */
std::string trimParameterText(const std::string& text)
{
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

/** Read flat parameter children strictly, rejecting misspellings and ambiguous controls. */
void readParameterChildren(xmlNodePtr training_node,
                           std::unordered_map<std::string, std::string>& values,
                           std::unordered_set<std::string>& seen_controls)
{
  for (xmlNodePtr child = training_node->children; child != nullptr; child = child->next)
  {
    if (child->type == XML_COMMENT_NODE)
      continue;
    if (child->type == XML_TEXT_NODE || child->type == XML_CDATA_SECTION_NODE)
    {
      if (trimParameterText(castXMLCharToChar(child->content)).empty())
        continue;
      throw UniformCommunicateError("WFTrainDriverInput: unexpected text in training section");
    }
    if (child->type != XML_ELEMENT_NODE ||
        std::string_view(castXMLCharToChar(child->name)) != "parameter")
      throw UniformCommunicateError("WFTrainDriverInput: training children must be parameter elements");

    // Each parameter has exactly one attribute, its name, and only scalar text.
    for (xmlAttrPtr attribute = child->properties; attribute != nullptr; attribute = attribute->next)
      if (std::string_view(castXMLCharToChar(attribute->name)) != "name" || attribute->ns != nullptr)
        throw UniformCommunicateError("WFTrainDriverInput: training parameter accepts only a name attribute");
    const std::string name = readAttribute(child, "name");
    if (name == "objective" || name == "optimizer")
      throw UniformCommunicateError("WFTrainDriverInput: " + name + " must remain a training attribute");
    if (std::find(allowed_controls.begin(), allowed_controls.end(), name) == allowed_controls.end())
      throw UniformCommunicateError("WFTrainDriverInput: unknown training parameter " + name);
    if (!seen_controls.insert(name).second)
      throw UniformCommunicateError("WFTrainDriverInput: duplicate training parameter " + name);

    std::string text;
    for (xmlNodePtr content = child->children; content != nullptr; content = content->next)
    {
      if (content->type == XML_COMMENT_NODE)
        continue;
      if (content->type != XML_TEXT_NODE && content->type != XML_CDATA_SECTION_NODE)
        throw UniformCommunicateError("WFTrainDriverInput: nested content in parameter " + name);
      text += castXMLCharToChar(content->content);
    }
    text = trimParameterText(text);
    if (text.empty())
      throw UniformCommunicateError("WFTrainDriverInput: empty training parameter " + name);
    values.emplace(name, std::move(text));
  }
}

} // namespace

void WFTrainDriverInput::readXML(xmlNodePtr qmc_node)
{
  if (qmc_node == nullptr)
    throw UniformCommunicateError("WFTrainDriverInput: cannot parse a null qmc node");

  xmlNodePtr training_node = nullptr;
  for (xmlNodePtr child = qmc_node->children; child != nullptr; child = child->next)
  {
    if (child->type != XML_ELEMENT_NODE || std::string_view(castXMLCharToChar(child->name)) != "training")
      continue;
    if (training_node != nullptr)
      throw UniformCommunicateError("WFTrainDriverInput: duplicate training section");
    training_node = child;
  }
  if (training_node == nullptr)
    throw UniformCommunicateError("WFTrainDriverInput: method=\"wf_train\" requires one training section");

  std::unordered_set<std::string> seen_controls;
  std::unordered_map<std::string, std::string> values;
  for (xmlAttrPtr attribute = training_node->properties; attribute != nullptr; attribute = attribute->next)
  {
    const std::string name = castXMLCharToChar(attribute->name);
    if (name != "objective" && name != "optimizer")
    {
      if (std::find(allowed_controls.begin(), allowed_controls.end(), name) != allowed_controls.end())
        throw UniformCommunicateError("WFTrainDriverInput: obsolete training attribute " + name +
                                      "; use <parameter name='" + name + "'>value</parameter>. "
                                      "Set training iterations explicitly; blocks now count statistical sample blocks.");
      throw UniformCommunicateError("WFTrainDriverInput: unknown training attribute " + name);
    }
    if (attribute->ns != nullptr || !seen_controls.insert(name).second)
      throw UniformCommunicateError("WFTrainDriverInput: duplicate or namespaced training attribute " + name);
    values.emplace(name, readAttribute(training_node, name.c_str()));
  }
  readParameterChildren(training_node, values, seen_controls);

  // Common validation below treats scalar attributes and parameter values uniformly.
  const auto readControl = [&values](const char* name) -> const std::string& { return values.at(name); };

  iterations_ = 1;
  checkpoint_interval_ = 1000;
  warmup_steps_ = 0;
  objective_                        = Objective::ENERGY;
  gradient_estimator_ = GradientEstimator::LEGACY;
  orbital_target_                    = OrbitalTarget::SINGLE_REFERENCE;
  momentum_decay_ = 0.9;
  rms_decay_ = 0.9;
  lm_overlap_shift_ = 1.0e-4;
  lm_normalization_xi_ = 0.5;
  lm_maximum_subspace_dimension_ = 8;
  lm_inner_iterations_ = 8;
  lm_inner_restart_length_ = 4;
  lm_inner_relative_tolerance_ = 0.1;
  kfac_fallback_damping_ = 1.0e-3;
  kfac_objective_ = KFACObjective::RAW_COVARIANCE;
  kfac_defaults_ = KFACDefaults::LEGACY;
  kfac_gradient_ = KFACGradient::LEGACY;
  kfac_update_order_ = KFACUpdateOrder::POST_INVERSE_RATE;
  kfac_schedule_arithmetic_ = KFACScheduleArithmetic::DIVIDE_RATE;
  kfac_clip_mode_ = KFACClipMode::SOFT_MEDIAN;
  kfac_curvature_ = KFACCurvature::NATIVE_SCORE;
  kfac_damping_policy_ = KFACDamping::EQUAL_SPLIT;
  kfac_constraint_ = KFACConstraint::NONE;
  kfac_clip_width_ = 1.0;
  kfac_complex_clip_width_ = 5.0;
  kfac_clip_quantile_ = 0.95;
  kfac_exclude_width_ = std::numeric_limits<double>::infinity();
  kfac_norm_constraint_ = 0.0;
  kfac_persist_state_ = false;
  kfac_ema_decay_ = 0.0;
  kfac_inverse_period_ = 1;
  kfac_learning_rate_decay_ = 0.0;
  kfac_schedule_float32_ = false;
  line_search_ = LineSearch::OFF;
  update_scale_ = 1.0;
  weight_decay_ = 0.0;
  line_search_minimum_ess_fraction_ = 0.5;
  line_search_maximum_weight_ = 0.25;
  line_search_confidence_ = 2.0;
  line_search_warmup_steps_ = 10;
  line_search_sampling_steps_ = 10;
  optimizer_                        = Optimizer::ADAM;
  parameter_chunk_size_             = 65536;
  legacy_maximum_parameters_        = 0;
  legacy_derivative_scratch_bytes_  = 0;
  learning_rate_                    = 1.0e-3;
  adam_beta1_                       = 0.9;
  adam_beta2_                       = 0.999;
  epsilon_                          = 1.0e-8;
  damping_                          = 1.0e-3;
  damping_multiplier_               = 10.0;
  minimum_retry_damping_            = 1.0e-8;
  maximum_damping_attempts_         = 3;
  maximum_update_norm_              = 0.0;
  maximum_metric_norm_              = 0.0;
  center_shift_                     = 1.0;
  objective_tolerance_              = 1.0e-9;
  pii_shift_mode_                   = PIIShiftMode::MEAN_ENERGY_MINUS_UNDERSHOOT;
  pii_explicit_tau_                 = 0.0;
  pii_energy_undershoot_            = 0.1;
  maximum_damping_retries_          = 2;
  gmres_restart_length_             = 20;
  spring_history_decay_             = 0.99;
  spring_maximum_update_norm_squared_ = 0.0;
  krylov_relative_tolerance_         = 1.0e-8;
  krylov_absolute_tolerance_         = 0.0;
  krylov_maximum_iterations_         = 100;
  krylov_maximum_operator_applications_ = 101;
  krylov_maximum_seconds_              = 0.0;
  krylov_stagnation_window_            = 0;
  krylov_minimum_relative_improvement_ = 1.0e-4;
  krylov_curvature_tolerance_          = 0.0;
  krylov_imaginary_tolerance_          = 64.0;
  maximum_retained_bytes_              = 0;
  complex_gpu_training_                = {};
  maximum_resident_bytes_              = 0;
  checkpoint_path_.clear();
  restart_path_.clear();
  provider_id_.clear();
  teacher_wavefunction_.clear();
  orbital_sampling_ = OrbitalSampling::TEACHER;
  orbital_normalization_ = wftrain::OrbitalLossNormalization::SPIN_SECTOR_SUM;
  overlap_target_            = OverlapTarget::GENERAL_FROZEN_TEACHER;
  overlap_weighting_         = OverlapWeighting::EQUAL;
  target_provenance_.clear();
  minimum_fidelity_          = 1.0e-14;
  minimum_effective_samples_ = 1.0;
  // Expand the selected bundle before applying any explicit scalar override.
  if (seen_controls.count("defaults"))
  {
    const auto& label = readControl("defaults");
    if (label == "deepqmc") kfac_defaults_ = KFACDefaults::SOFT_COVARIANCE;
    else if (label == "deepsolid") kfac_defaults_ = KFACDefaults::MEAN_RESIDUAL;
    else if (label == "jaqmc") kfac_defaults_ = KFACDefaults::MEDIAN_COVARIANCE;
    else throw UniformCommunicateError("WFTrainDriverInput: unknown defaults label");
    learning_rate_ = .05;
    damping_ = kfac_fallback_damping_ = .001;
    kfac_curvature_ = KFACCurvature::REGISTERED_GAUSSIAN;
    kfac_damping_policy_ = KFACDamping::TRACE_BALANCED;
    kfac_persist_state_ = true;
    kfac_ema_decay_ = .95;
    kfac_inverse_period_ = 1;
    kfac_constraint_ = KFACConstraint::DAMPED_KFAC;
    kfac_norm_constraint_ = .001;
    kfac_learning_rate_decay_ = kfac_defaults_ == KFACDefaults::MEDIAN_COVARIANCE ? 2000 : 10000;
    kfac_schedule_float32_ = kfac_defaults_ == KFACDefaults::MEDIAN_COVARIANCE;
    if (kfac_defaults_ == KFACDefaults::SOFT_COVARIANCE)
    {
      kfac_objective_ = KFACObjective::CLIPPED_COVARIANCE;
      kfac_gradient_ = KFACGradient::HALF_COVARIANCE;
    }
    else if (kfac_defaults_ == KFACDefaults::MEAN_RESIDUAL)
    {
      kfac_objective_ = KFACObjective::COMPONENTWISE_MEAN_RESIDUAL;
      kfac_gradient_ = KFACGradient::REAL_CENTERED_RESIDUAL;
    }
    else
    {
      kfac_objective_ = KFACObjective::COMPONENTWISE_MAD_COVARIANCE;
      kfac_gradient_ = KFACGradient::PHYSICAL_COVARIANCE;
      kfac_update_order_ = KFACUpdateOrder::PRE_INVERSE_RATE;
      kfac_schedule_arithmetic_ = KFACScheduleArithmetic::MULTIPLY_RECIPROCAL;
    }
  }
  if (seen_controls.count("iterations"))
    iterations_ = parsePositiveSize(readControl("iterations"), "iterations");
  if (seen_controls.count("checkpoint_interval"))
    checkpoint_interval_ = parsePositiveSize(readControl("checkpoint_interval"), "checkpoint_interval");
  if (seen_controls.count("warmupsteps"))
    warmup_steps_ = parseNonnegativeSize(readControl("warmupsteps"), "warmupsteps");
  if (seen_controls.count("momentum_decay"))
    momentum_decay_ = parseNonnegativeReal(readControl("momentum_decay"), "momentum_decay");
  if (seen_controls.count("rms_decay"))
    rms_decay_ = parseNonnegativeReal(readControl("rms_decay"), "rms_decay");
  if (seen_controls.count("lm_overlap_shift"))
    lm_overlap_shift_ = parseNonnegativeReal(readControl("lm_overlap_shift"), "lm_overlap_shift");
  if (seen_controls.count("lm_normalization_xi"))
    lm_normalization_xi_ = parseNonnegativeReal(readControl("lm_normalization_xi"), "lm_normalization_xi");
  if (seen_controls.count("lm_maximum_subspace_dimension"))
    lm_maximum_subspace_dimension_ = parsePositiveSize(readControl("lm_maximum_subspace_dimension"), "lm_maximum_subspace_dimension");
  if (seen_controls.count("lm_inner_iterations"))
    lm_inner_iterations_ = parsePositiveSize(readControl("lm_inner_iterations"), "lm_inner_iterations");
  if (seen_controls.count("lm_inner_restart_length"))
    lm_inner_restart_length_ = parsePositiveSize(readControl("lm_inner_restart_length"), "lm_inner_restart_length");
  if (seen_controls.count("lm_inner_relative_tolerance"))
    lm_inner_relative_tolerance_ = parseNonnegativeReal(readControl("lm_inner_relative_tolerance"), "lm_inner_relative_tolerance");
  if (seen_controls.count("kfac_fallback_damping"))
    kfac_fallback_damping_ = parseNonnegativeReal(readControl("kfac_fallback_damping"), "kfac_fallback_damping");
  if (momentum_decay_ >= 1.0 || rms_decay_ >= 1.0 || lm_normalization_xi_ > 1.0 ||
      lm_overlap_shift_ <= 0.0 || lm_inner_relative_tolerance_ <= 0.0 ||
      lm_maximum_subspace_dimension_ < 2 || kfac_fallback_damping_ <= 0.0)
    throw UniformCommunicateError("Invalid first-order/LM/KFAC optimizer controls");

  if (seen_controls.count("line_search"))
  {
    const auto mode = readControl("line_search");
    if (mode == "off") line_search_ = LineSearch::OFF;
    else if (mode == "reference") line_search_ = LineSearch::REFERENCE;
    else if (mode == "endpoints") line_search_ = LineSearch::ENDPOINTS;
    else throw UniformCommunicateError("line_search must be off, reference, or endpoints");
  }
  if (seen_controls.count("weight_decay_scope") &&
      readControl("weight_decay_scope") != "nn_weights")
    throw UniformCommunicateError("weight_decay_scope currently supports nn_weights only");
  if (seen_controls.count("update_scale"))
    update_scale_ = parseNonnegativeReal(readControl("update_scale"), "update_scale");
  if (seen_controls.count("weight_decay"))
    weight_decay_ = parseNonnegativeReal(readControl("weight_decay"), "weight_decay");
  if (seen_controls.count("line_search_minimum_ess_fraction"))
    line_search_minimum_ess_fraction_ = parseNonnegativeReal(readControl("line_search_minimum_ess_fraction"), "line_search_minimum_ess_fraction");
  if (seen_controls.count("line_search_maximum_weight"))
    line_search_maximum_weight_ = parseNonnegativeReal(readControl("line_search_maximum_weight"), "line_search_maximum_weight");
  if (seen_controls.count("line_search_confidence"))
    line_search_confidence_ = parseNonnegativeReal(readControl("line_search_confidence"), "line_search_confidence");
  if (seen_controls.count("line_search_warmup_steps"))
    line_search_warmup_steps_ = parsePositiveSize(readControl("line_search_warmup_steps"), "line_search_warmup_steps");
  if (seen_controls.count("line_search_sampling_steps"))
    line_search_sampling_steps_ = parsePositiveSize(readControl("line_search_sampling_steps"), "line_search_sampling_steps");
  if (weight_decay_ >= 1.0 || line_search_minimum_ess_fraction_ > 1.0 ||
      line_search_minimum_ess_fraction_ == 0.0 || line_search_maximum_weight_ > 1.0 ||
      line_search_maximum_weight_ == 0.0)
    throw UniformCommunicateError("Decay must lie in [0,1), and line-search weight guards in (0,1]");

  if (seen_controls.find("objective") != seen_controls.end())
  {
    const std::string objective = readControl("objective");
    if (objective == "energy")
      objective_ = Objective::ENERGY;
    else if (objective == "determinant_overlap")
      objective_ = Objective::DETERMINANT_OVERLAP;
    else if (objective == "orbital_mse")
      objective_ = Objective::ORBITAL_MSE;
    else
      throw UniformCommunicateError("WFTrainDriverInput: unsupported objective=\"" + objective +
                                    "\"; supported objectives are energy, determinant_overlap and orbital_mse");
  }
  if (seen_controls.count("gradient_estimator"))
  {
    const auto& value = readControl("gradient_estimator");
    if (value == "legacy") gradient_estimator_ = GradientEstimator::LEGACY;
    else if (value == "score_covariance") gradient_estimator_ = GradientEstimator::SCORE_COVARIANCE;
    else if (value == "full_local_energy_response")
      gradient_estimator_ = GradientEstimator::FULL_LOCAL_ENERGY_RESPONSE;
    else
      throw UniformCommunicateError("WFTrainDriverInput: gradient_estimator must be legacy, score_covariance, or full_local_energy_response");
  }
  if (seen_controls.find("optimizer") != seen_controls.end())
  {
    const std::string optimizer = readControl("optimizer");
    if (optimizer == "adam")
      optimizer_ = Optimizer::ADAM;
    else if (optimizer == "sgd") optimizer_ = Optimizer::SGD;
    else if (optimizer == "momentum_sgd") optimizer_ = Optimizer::MOMENTUM_SGD;
    else if (optimizer == "rmsprop") optimizer_ = Optimizer::RMSPROP;
    else if (optimizer == "amsgrad") optimizer_ = Optimizer::AMSGRAD;
    else if (optimizer == "lm") optimizer_ = Optimizer::LINEAR_METHOD;
    else if (optimizer == "alm") optimizer_ = Optimizer::ACCELERATED_LINEAR_METHOD;
    else if (optimizer == "kfac") optimizer_ = Optimizer::KFAC;
    else if (optimizer == "sr")
      optimizer_ = Optimizer::STOCHASTIC_RECONFIGURATION;
    else if (optimizer == "minsr")
      optimizer_ = Optimizer::MINIMUM_STEP_SR;
    else if (optimizer == "spring")
      optimizer_ = Optimizer::SPRING;
    else if (optimizer == "pii")
      optimizer_ = Optimizer::PROJECTED_INVERSE_ITERATION;
    else if (optimizer == "minpii")
      optimizer_ = Optimizer::MINIMUM_PROJECTED_INVERSE_ITERATION;
    else
      throw UniformCommunicateError("WFTrainDriverInput: unsupported optimizer=\"" + optimizer +
                                    "\"; supported optimizers are sgd, momentum_sgd, rmsprop, adam, amsgrad, sr, minsr, spring, pii, minpii, lm, alm, kfac");
  }
  const bool pii_optimizer = optimizer_ == Optimizer::PROJECTED_INVERSE_ITERATION ||
      optimizer_ == Optimizer::MINIMUM_PROJECTED_INVERSE_ITERATION;
  const bool lm_optimizer = optimizer_ == Optimizer::LINEAR_METHOD ||
      optimizer_ == Optimizer::ACCELERATED_LINEAR_METHOD;
  const bool first_order = optimizer_ == Optimizer::ADAM || optimizer_ == Optimizer::AMSGRAD ||
      optimizer_ == Optimizer::SGD || optimizer_ == Optimizer::MOMENTUM_SGD || optimizer_ == Optimizer::RMSPROP;
  if (lm_optimizer && !seen_controls.count("krylov_relative_tolerance"))
    krylov_relative_tolerance_ = 1.0e-3;
  if (lm_optimizer && !seen_controls.count("learning_rate")) learning_rate_ = 1.0;
  if (pii_optimizer && seen_controls.find("learning_rate") == seen_controls.end())
    learning_rate_ = 1.0;
  if (seen_controls.find("parameter_chunk_size") != seen_controls.end())
    parameter_chunk_size_ = parsePositiveSize(readControl("parameter_chunk_size"),
                                              "parameter_chunk_size");
  if (seen_controls.count("legacy_maximum_parameters"))
    legacy_maximum_parameters_ = parsePositiveSize(readControl("legacy_maximum_parameters"),
                                                   "legacy_maximum_parameters");
  if (seen_controls.count("legacy_derivative_scratch_bytes"))
    legacy_derivative_scratch_bytes_ = parsePositiveSize(readControl("legacy_derivative_scratch_bytes"),
                                                        "legacy_derivative_scratch_bytes");
  if (seen_controls.find("learning_rate") != seen_controls.end())
  {
    learning_rate_ = parseFiniteReal(readControl("learning_rate"), "learning_rate");
    if (!(learning_rate_ > 0.0))
      throw UniformCommunicateError("WFTrainDriverInput: learning_rate must be positive");
  }
  if (seen_controls.find("adam_beta1") != seen_controls.end())
    adam_beta1_ = parseDecay(readControl("adam_beta1"), "adam_beta1");
  if (seen_controls.find("adam_beta2") != seen_controls.end())
    adam_beta2_ = parseDecay(readControl("adam_beta2"), "adam_beta2");
  if (seen_controls.find("epsilon") != seen_controls.end())
  {
    epsilon_ = parseFiniteReal(readControl("epsilon"), "epsilon");
    if (!(epsilon_ > 0.0))
      throw UniformCommunicateError("WFTrainDriverInput: epsilon must be positive");
  }
  if (seen_controls.find("damping") != seen_controls.end())
    damping_ = parseNonnegativeReal(readControl("damping"), "damping");
  if (seen_controls.find("damping_multiplier") != seen_controls.end())
  {
    damping_multiplier_ = parseFiniteReal(readControl("damping_multiplier"), "damping_multiplier");
    if (!(damping_multiplier_ > 1.0))
      throw UniformCommunicateError("WFTrainDriverInput: damping_multiplier must be greater than one");
  }
  if (seen_controls.find("minimum_retry_damping") != seen_controls.end())
  {
    minimum_retry_damping_ = parseFiniteReal(readControl("minimum_retry_damping"),
                                             "minimum_retry_damping");
    if (!(minimum_retry_damping_ > 0.0))
      throw UniformCommunicateError("WFTrainDriverInput: minimum_retry_damping must be positive");
  }
  if (seen_controls.find("maximum_damping_attempts") != seen_controls.end())
    maximum_damping_attempts_ = parsePositiveSize(readControl("maximum_damping_attempts"),
                                                  "maximum_damping_attempts");
  if (seen_controls.find("maximum_update_norm") != seen_controls.end())
    maximum_update_norm_ = parseNonnegativeReal(readControl("maximum_update_norm"),
                                                "maximum_update_norm");
  if (seen_controls.find("maximum_metric_norm") != seen_controls.end())
    maximum_metric_norm_ = parseNonnegativeReal(readControl("maximum_metric_norm"),
                                                "maximum_metric_norm");
  if (seen_controls.find("center_shift") != seen_controls.end())
    center_shift_ = parseNonnegativeReal(readControl("center_shift"), "center_shift");
  if (seen_controls.find("objective_tolerance") != seen_controls.end())
    objective_tolerance_ = parseNonnegativeReal(readControl("objective_tolerance"),
                                                "objective_tolerance");
  if (seen_controls.find("pii_shift_mode") != seen_controls.end())
  {
    const std::string mode = readControl("pii_shift_mode");
    if (mode == "explicit")
      pii_shift_mode_ = PIIShiftMode::EXPLICIT;
    else if (mode == "mean_energy_minus_undershoot")
      pii_shift_mode_ = PIIShiftMode::MEAN_ENERGY_MINUS_UNDERSHOOT;
    else
      throw UniformCommunicateError(
          "WFTrainDriverInput: pii_shift_mode must be explicit or mean_energy_minus_undershoot");
  }
  if (seen_controls.find("pii_explicit_tau") != seen_controls.end())
    pii_explicit_tau_ = parseFiniteReal(readControl("pii_explicit_tau"),
                                        "pii_explicit_tau");
  if (seen_controls.find("pii_energy_undershoot") != seen_controls.end())
    pii_energy_undershoot_ = parseNonnegativeReal(
        readControl("pii_energy_undershoot"), "pii_energy_undershoot");
  if (seen_controls.find("maximum_damping_retries") != seen_controls.end())
  {
    maximum_damping_retries_ = parseNonnegativeSize(
        readControl("maximum_damping_retries"), "maximum_damping_retries");
    if (maximum_damping_retries_ > 64)
      throw UniformCommunicateError(
          "WFTrainDriverInput: maximum_damping_retries must not exceed 64");
  }
  if (seen_controls.find("gmres_restart_length") != seen_controls.end())
    gmres_restart_length_ = parsePositiveSize(
        readControl("gmres_restart_length"), "gmres_restart_length");
  if (seen_controls.find("spring_history_decay") != seen_controls.end())
    spring_history_decay_ = parseDecay(readControl("spring_history_decay"),
                                       "spring_history_decay");
  if (seen_controls.find("spring_maximum_update_norm_squared") != seen_controls.end())
    spring_maximum_update_norm_squared_ = parseNonnegativeReal(
        readControl("spring_maximum_update_norm_squared"),
        "spring_maximum_update_norm_squared");
  if (seen_controls.find("krylov_relative_tolerance") != seen_controls.end())
    krylov_relative_tolerance_ = parseNonnegativeReal(
        readControl("krylov_relative_tolerance"), "krylov_relative_tolerance");
  if (seen_controls.find("krylov_absolute_tolerance") != seen_controls.end())
    krylov_absolute_tolerance_ = parseNonnegativeReal(
        readControl("krylov_absolute_tolerance"), "krylov_absolute_tolerance");
  if (seen_controls.find("krylov_maximum_iterations") != seen_controls.end())
    krylov_maximum_iterations_ = parsePositiveSize(readControl("krylov_maximum_iterations"),
                                                   "krylov_maximum_iterations");
  if (seen_controls.find("krylov_maximum_operator_applications") != seen_controls.end())
    krylov_maximum_operator_applications_ = parsePositiveSize(
        readControl("krylov_maximum_operator_applications"),
        "krylov_maximum_operator_applications");
  if (seen_controls.find("krylov_maximum_seconds") != seen_controls.end())
    krylov_maximum_seconds_ = parseNonnegativeReal(readControl("krylov_maximum_seconds"),
                                                   "krylov_maximum_seconds");
  if (seen_controls.find("krylov_stagnation_window") != seen_controls.end())
    krylov_stagnation_window_ = parseNonnegativeSize(readControl("krylov_stagnation_window"),
                                                     "krylov_stagnation_window");
  if (seen_controls.find("krylov_minimum_relative_improvement") != seen_controls.end())
    krylov_minimum_relative_improvement_ = parseNonnegativeReal(
        readControl("krylov_minimum_relative_improvement"),
        "krylov_minimum_relative_improvement");
  if (seen_controls.find("krylov_curvature_tolerance") != seen_controls.end())
    krylov_curvature_tolerance_ = parseNonnegativeReal(
        readControl("krylov_curvature_tolerance"), "krylov_curvature_tolerance");
  if (seen_controls.find("krylov_imaginary_tolerance") != seen_controls.end())
  {
    krylov_imaginary_tolerance_ = parseFiniteReal(
        readControl("krylov_imaginary_tolerance"), "krylov_imaginary_tolerance");
    if (!(krylov_imaginary_tolerance_ > 0.0))
      throw UniformCommunicateError("WFTrainDriverInput: krylov_imaginary_tolerance must be positive");
  }
  if (seen_controls.find("maximum_retained_bytes") != seen_controls.end())
    maximum_retained_bytes_ = parseNonnegativeSize(readControl("maximum_retained_bytes"),
                                                   "maximum_retained_bytes");
  if (seen_controls.count("maximum_resident_bytes"))
    maximum_resident_bytes_ = parseNonnegativeSize(readControl("maximum_resident_bytes"),
                                                  "maximum_resident_bytes");
  // These four controls form one opt-in contract; zero/SIZE_MAX never mean unlimited.
  const std::array<const char*, 4> gpu_names{"complex_gpu_device_bytes", "complex_gpu_host_bytes",
      "complex_gpu_kinetic_targets", "complex_gpu_maximum_matrix_work"};
  const std::array<std::size_t*, 4> gpu_values{&complex_gpu_training_.device_bytes, &complex_gpu_training_.host_bytes,
      &complex_gpu_training_.kinetic_targets, &complex_gpu_training_.maximum_matrix_work};
  std::size_t gpu_present = 0;
  for (std::size_t i = 0; i < gpu_names.size(); ++i)
    if (seen_controls.count(gpu_names[i]))
    {
      ++gpu_present;
      *gpu_values[i] = parsePositiveSize(readControl(gpu_names[i]), gpu_names[i]);
      if (*gpu_values[i] == std::numeric_limits<std::size_t>::max())
        throw UniformCommunicateError("WFTrainDriverInput: complex GPU controls require finite positive limits");
    }
  if (gpu_present && gpu_present != gpu_names.size())
    throw UniformCommunicateError("WFTrainDriverInput: all four complex GPU controls are required together");
  if (seen_controls.find("checkpoint_path") != seen_controls.end())
    checkpoint_path_ = readControl("checkpoint_path");
  if (seen_controls.find("restart_path") != seen_controls.end())
    restart_path_ = readControl("restart_path");
  if (seen_controls.find("provider") != seen_controls.end())
  {
    provider_id_ = readControl("provider");
    if (provider_id_.empty())
      throw UniformCommunicateError("WFTrainDriverInput: provider must not be empty when specified");
  }
  if (seen_controls.find("teacher") != seen_controls.end())
  {
    teacher_wavefunction_ = readControl("teacher");
    if (teacher_wavefunction_.empty())
      throw UniformCommunicateError("WFTrainDriverInput: teacher must not be empty when specified");
  }
  if (seen_controls.find("overlap_target") != seen_controls.end())
  {
    const std::string target = readControl("overlap_target");
    if (target == "general")
      overlap_target_ = OverlapTarget::GENERAL_FROZEN_TEACHER;
    else if (target == "top_coefficients")
      overlap_target_ = OverlapTarget::TOP_COEFFICIENT_MULTIDETERMINANT;
    else
      throw UniformCommunicateError(
          "WFTrainDriverInput: overlap_target must be general or top_coefficients");
  }
  if (seen_controls.find("overlap_weighting") != seen_controls.end())
  {
    const std::string weighting = readControl("overlap_weighting");
    if (weighting == "equal")
      overlap_weighting_ = OverlapWeighting::EQUAL;
    else if (weighting == "coefficient_squared")
      overlap_weighting_ = OverlapWeighting::COEFFICIENT_SQUARED;
    else
      throw UniformCommunicateError(
          "WFTrainDriverInput: overlap_weighting must be equal or coefficient_squared");
  }
  if (seen_controls.find("target_provenance") != seen_controls.end())
  {
    target_provenance_ = readControl("target_provenance");
    if (target_provenance_.empty())
      throw UniformCommunicateError(
          "WFTrainDriverInput: target_provenance must not be empty when specified");
  }
  if (seen_controls.find("minimum_fidelity") != seen_controls.end())
  {
    minimum_fidelity_ = parseFiniteReal(readControl("minimum_fidelity"),
                                        "minimum_fidelity");
    if (minimum_fidelity_ < 0.0 || minimum_fidelity_ > 1.0)
      throw UniformCommunicateError("WFTrainDriverInput: minimum_fidelity must lie in [0, 1]");
  }
  if (seen_controls.find("minimum_effective_samples") != seen_controls.end())
    minimum_effective_samples_ = parseNonnegativeReal(
        readControl("minimum_effective_samples"),
        "minimum_effective_samples");

  const std::initializer_list<std::string_view> overlap_attributes = {
      "teacher", "overlap_target", "overlap_weighting", "target_provenance",
      "minimum_fidelity", "minimum_effective_samples"};
  if (seen_controls.count("orbital_target"))
  {
    const std::string target = readControl("orbital_target");
    if (target == "single_reference")
      orbital_target_ = OrbitalTarget::SINGLE_REFERENCE;
    else if (target == "top_coefficients")
      orbital_target_ = OrbitalTarget::TOP_COEFFICIENT_MULTIDETERMINANT;
    else
      throw UniformCommunicateError(
          "WFTrainDriverInput: orbital_target must be single_reference or top_coefficients");
  }
  if (seen_controls.count("orbital_sampling"))
  {
    const auto sampling = readControl("orbital_sampling");
    if (sampling == "teacher") orbital_sampling_ = OrbitalSampling::TEACHER;
    else if (sampling == "student") orbital_sampling_ = OrbitalSampling::STUDENT;
    else throw UniformCommunicateError("WFTrainDriverInput: orbital_sampling must be teacher or student");
  }
  if (objective_ != Objective::ORBITAL_MSE)
    rejectControls(seen_controls, {"orbital_target", "orbital_sampling", "orbital_normalization"}, "non-orbital objective");
  if (seen_controls.count("orbital_normalization"))
  {
    const auto normalization = readControl("orbital_normalization");
    if (normalization == "spin_sector_sum")
      orbital_normalization_ = wftrain::OrbitalLossNormalization::SPIN_SECTOR_SUM;
    else if (normalization == "full_matrix_mean")
      orbital_normalization_ = wftrain::OrbitalLossNormalization::FULL_MATRIX_MEAN;
    else
      throw UniformCommunicateError("WFTrainDriverInput: orbital_normalization must be spin_sector_sum or full_matrix_mean");
  }
  if (objective_ == Objective::ENERGY)
    rejectControls(seen_controls, overlap_attributes, "energy objective");
  else if (objective_ == Objective::ORBITAL_MSE)
  {
    if (teacher_wavefunction_.empty() || target_provenance_.empty())
      throw UniformCommunicateError(
          "WFTrainDriverInput: orbital_mse requires teacher and target_provenance");
    if (!first_order)
      throw UniformCommunicateError(
          "WFTrainDriverInput: orbital_mse requires a first-order optimizer");
    if (line_search_ != LineSearch::OFF || maximum_metric_norm_ != 0.0)
      throw UniformCommunicateError(
          "WFTrainDriverInput: orbital_mse requires line_search=off and maximum_metric_norm=0");
    rejectControls(seen_controls, {"overlap_target", "overlap_weighting", "minimum_fidelity",
                                  "minimum_effective_samples"}, "orbital_mse objective");
  }
  else
  {
    if (teacher_wavefunction_.empty() || target_provenance_.empty())
      throw UniformCommunicateError(
          "WFTrainDriverInput: determinant_overlap requires teacher and target_provenance");
    if (!first_order)
      throw UniformCommunicateError(
          "WFTrainDriverInput: determinant_overlap requires a first-order optimizer");
    if (overlap_weighting_ == OverlapWeighting::COEFFICIENT_SQUARED &&
        overlap_target_ != OverlapTarget::TOP_COEFFICIENT_MULTIDETERMINANT)
      throw UniformCommunicateError(
          "WFTrainDriverInput: coefficient_squared weighting requires top_coefficients targets");
  }

  const std::initializer_list<std::string_view> adam_attributes = {"adam_beta1", "adam_beta2", "epsilon"};
  const std::initializer_list<std::string_view> matrix_free_attributes = {
      "damping", "damping_multiplier", "minimum_retry_damping", "maximum_damping_attempts",
      "maximum_metric_norm", "center_shift", "objective_tolerance",
      "spring_history_decay", "spring_maximum_update_norm_squared", "krylov_relative_tolerance",
      "krylov_absolute_tolerance", "krylov_maximum_iterations", "krylov_maximum_operator_applications",
      "krylov_maximum_seconds", "krylov_stagnation_window", "krylov_minimum_relative_improvement",
      "krylov_curvature_tolerance", "krylov_imaginary_tolerance", "pii_shift_mode",
      "pii_explicit_tau", "pii_energy_undershoot", "maximum_damping_retries",
      "gmres_restart_length"};
  const std::initializer_list<std::string_view> sr_only_attributes = {
      "damping_multiplier", "minimum_retry_damping", "maximum_damping_attempts", "maximum_metric_norm"};
  const std::initializer_list<std::string_view> sr_non_pii_attributes = {
      "minimum_retry_damping", "maximum_damping_attempts", "maximum_metric_norm"};
  const std::initializer_list<std::string_view> minimum_step_attributes = {"center_shift",
                                                                           "objective_tolerance"};
  const std::initializer_list<std::string_view> spring_only_attributes = {
      "spring_history_decay", "spring_maximum_update_norm_squared"};
  const std::initializer_list<std::string_view> pii_only_attributes = {
      "pii_shift_mode", "pii_explicit_tau", "pii_energy_undershoot",
      "maximum_damping_retries", "gmres_restart_length"};
  switch (optimizer_)
  {
  case Optimizer::SGD:
  case Optimizer::MOMENTUM_SGD:
    rejectControls(seen_controls, adam_attributes, "first-order");
    [[fallthrough]];
  case Optimizer::RMSPROP:
    if (optimizer_ == Optimizer::RMSPROP)
      rejectControls(seen_controls, {"adam_beta1", "adam_beta2"}, "rmsprop");
    [[fallthrough]];
  case Optimizer::AMSGRAD:
  case Optimizer::ADAM:
    rejectControls(seen_controls, matrix_free_attributes, "first-order");
    break;
  case Optimizer::LINEAR_METHOD:
  case Optimizer::ACCELERATED_LINEAR_METHOD:
  case Optimizer::KFAC:
    rejectControls(seen_controls, adam_attributes, "lm/alm/kfac");
    rejectControls(seen_controls, spring_only_attributes, "lm/alm/kfac");
    rejectControls(seen_controls, pii_only_attributes, "lm/alm/kfac");
    rejectControls(seen_controls, sr_only_attributes, "lm/alm/kfac");
    rejectControls(seen_controls, minimum_step_attributes, "lm/alm/kfac");
    // These solvers use residual/action/time limits, not the CG-specific
    // stagnation, curvature or imaginary-part policy of the SR solver.
    rejectControls(seen_controls, {"krylov_stagnation_window",
        "krylov_minimum_relative_improvement", "krylov_curvature_tolerance",
        "krylov_imaginary_tolerance"}, "lm/alm/kfac");
    if (optimizer_ == Optimizer::KFAC)
      rejectControls(seen_controls, {"krylov_relative_tolerance", "krylov_absolute_tolerance",
          "krylov_maximum_iterations", "krylov_maximum_operator_applications",
          "krylov_maximum_seconds"}, "kfac");
    if (damping_ <= 0.0)
      throw UniformCommunicateError("LM/ALM/KFAC require positive damping");
    break;
  case Optimizer::STOCHASTIC_RECONFIGURATION:
    rejectControls(seen_controls, adam_attributes, "sr");
    rejectControls(seen_controls, minimum_step_attributes, "sr");
    rejectControls(seen_controls, spring_only_attributes, "sr");
    rejectControls(seen_controls, pii_only_attributes, "sr");
    break;
  case Optimizer::MINIMUM_STEP_SR:
    rejectControls(seen_controls, adam_attributes, "minsr");
    rejectControls(seen_controls, sr_only_attributes, "minsr");
    rejectControls(seen_controls, spring_only_attributes, "minsr");
    rejectControls(seen_controls, pii_only_attributes, "minsr");
    if (!(damping_ > 0.0))
      throw UniformCommunicateError("WFTrainDriverInput: optimizer=\"minsr\" requires positive damping");
    break;
  case Optimizer::SPRING:
    rejectControls(seen_controls, adam_attributes, "spring");
    rejectControls(seen_controls, sr_only_attributes, "spring");
    rejectControls(seen_controls, pii_only_attributes, "spring");
    if (!(damping_ > 0.0))
      throw UniformCommunicateError("WFTrainDriverInput: optimizer=\"spring\" requires positive damping");
    break;
  case Optimizer::PROJECTED_INVERSE_ITERATION:
  case Optimizer::MINIMUM_PROJECTED_INVERSE_ITERATION:
  {
    const std::string optimizer = optimizer_ == Optimizer::PROJECTED_INVERSE_ITERATION
        ? "pii" : "minpii";
    rejectControls(seen_controls, adam_attributes, optimizer);
    rejectControls(seen_controls, sr_non_pii_attributes, optimizer);
    rejectControls(seen_controls, {"center_shift"}, optimizer);
    rejectControls(seen_controls, spring_only_attributes, optimizer);
    if (!(damping_ > 0.0))
      throw UniformCommunicateError("WFTrainDriverInput: optimizer=\"" + optimizer +
                                    "\" requires positive damping");
    if (pii_shift_mode_ == PIIShiftMode::EXPLICIT)
    {
      if (seen_controls.find("pii_explicit_tau") == seen_controls.end())
        throw UniformCommunicateError("WFTrainDriverInput: pii_shift_mode=\"explicit\" requires pii_explicit_tau");
      rejectControls(seen_controls, {"pii_energy_undershoot"}, optimizer);
    }
    else
      rejectControls(seen_controls, {"pii_explicit_tau"}, optimizer);
    break;
  }
  }
  if (!first_order && krylov_relative_tolerance_ == 0.0 &&
      krylov_absolute_tolerance_ == 0.0)
    throw UniformCommunicateError(
        "WFTrainDriverInput: matrix-free optimization requires a positive Krylov tolerance");
  if (optimizer_ != Optimizer::MOMENTUM_SGD)
    rejectControls(seen_controls, {"momentum_decay"}, "selected optimizer");
  if (optimizer_ != Optimizer::RMSPROP)
    rejectControls(seen_controls, {"rms_decay"}, "selected optimizer");
  if (!lm_optimizer)
    rejectControls(seen_controls, {"lm_overlap_shift", "lm_normalization_xi",
        "lm_maximum_subspace_dimension", "lm_inner_iterations", "lm_inner_restart_length",
        "lm_inner_relative_tolerance"}, "selected optimizer");
  else if (optimizer_ != Optimizer::ACCELERATED_LINEAR_METHOD)
    rejectControls(seen_controls, {"lm_inner_iterations", "lm_inner_restart_length",
        "lm_inner_relative_tolerance"}, "lm");
  if (optimizer_ != Optimizer::KFAC)
    rejectControls(seen_controls, {"defaults", "kfac_gradient_convention", "kfac_update_order", "kfac_schedule_arithmetic",
        "kfac_fallback_damping", "kfac_objective", "kfac_clip_mode",
        "kfac_clip_width", "kfac_clip_quantile", "kfac_exclude_width", "kfac_curvature",
        "kfac_complex_clip_width",
        "kfac_damping_policy", "kfac_constraint", "kfac_norm_constraint", "kfac_persist_state",
        "kfac_ema_decay", "kfac_inverse_period", "kfac_learning_rate_decay", "kfac_schedule_precision",
        "kfac_momentum", "kfac_burn_in", "kfac_curvature_period"}, "selected optimizer");
  else
  {
    if (seen_controls.count("kfac_gradient_convention"))
    {
      const auto& value = readControl("kfac_gradient_convention");
      if (value == "physical_covariance") kfac_gradient_ = KFACGradient::PHYSICAL_COVARIANCE;
      else if (value == "half_covariance") kfac_gradient_ = KFACGradient::HALF_COVARIANCE;
      else if (value == "real_centered_residual") kfac_gradient_ = KFACGradient::REAL_CENTERED_RESIDUAL;
      else throw UniformCommunicateError("WFTrainDriverInput: invalid kfac_gradient_convention");
    }
    if (seen_controls.count("kfac_update_order"))
    {
      const auto& value = readControl("kfac_update_order");
      if (value == "post_inverse_rate") kfac_update_order_ = KFACUpdateOrder::POST_INVERSE_RATE;
      else if (value == "pre_inverse_rate") kfac_update_order_ = KFACUpdateOrder::PRE_INVERSE_RATE;
      else throw UniformCommunicateError("WFTrainDriverInput: invalid kfac_update_order");
    }
    if (seen_controls.count("kfac_schedule_arithmetic"))
    {
      const auto& value = readControl("kfac_schedule_arithmetic");
      if (value == "divide_rate") kfac_schedule_arithmetic_ = KFACScheduleArithmetic::DIVIDE_RATE;
      else if (value == "multiply_reciprocal") kfac_schedule_arithmetic_ = KFACScheduleArithmetic::MULTIPLY_RECIPROCAL;
      else throw UniformCommunicateError("WFTrainDriverInput: invalid kfac_schedule_arithmetic");
    }
    if (seen_controls.count("kfac_objective"))
    {
      const auto& value = readControl("kfac_objective");
      if (value == "raw_covariance") kfac_objective_ = KFACObjective::RAW_COVARIANCE;
      else if (value == "clipped_covariance") kfac_objective_ = KFACObjective::CLIPPED_COVARIANCE;
      else if (value == "componentwise_mad_covariance") kfac_objective_ = KFACObjective::COMPONENTWISE_MAD_COVARIANCE;
      else if (value == "componentwise_mean_residual") kfac_objective_ = KFACObjective::COMPONENTWISE_MEAN_RESIDUAL;
      else throw UniformCommunicateError("WFTrainDriverInput: invalid kfac_objective");
    }
    if (seen_controls.count("kfac_clip_mode"))
    {
      const auto& value = readControl("kfac_clip_mode");
      if (value == "soft_median") kfac_clip_mode_ = KFACClipMode::SOFT_MEDIAN;
      else if (value == "hard_median") kfac_clip_mode_ = KFACClipMode::HARD_MEDIAN;
      else if (value == "hard_mean") kfac_clip_mode_ = KFACClipMode::HARD_MEAN;
      else throw UniformCommunicateError("WFTrainDriverInput: invalid kfac_clip_mode");
    }
    if (seen_controls.count("kfac_curvature"))
    {
      const auto& value = readControl("kfac_curvature");
      if (value == "native_score") kfac_curvature_ = KFACCurvature::NATIVE_SCORE;
      else if (value == "registered_gaussian") kfac_curvature_ = KFACCurvature::REGISTERED_GAUSSIAN;
      else throw UniformCommunicateError("WFTrainDriverInput: invalid kfac_curvature");
    }
    if (seen_controls.count("kfac_damping_policy"))
    {
      const auto& value = readControl("kfac_damping_policy");
      if (value == "equal_split") kfac_damping_policy_ = KFACDamping::EQUAL_SPLIT;
      else if (value == "trace_balanced") kfac_damping_policy_ = KFACDamping::TRACE_BALANCED;
      else throw UniformCommunicateError("WFTrainDriverInput: invalid kfac_damping_policy");
    }
    if (seen_controls.count("kfac_constraint"))
    {
      const auto& value = readControl("kfac_constraint");
      if (value == "none") kfac_constraint_ = KFACConstraint::NONE;
      else if (value == "damped_kfac") kfac_constraint_ = KFACConstraint::DAMPED_KFAC;
      else throw UniformCommunicateError("WFTrainDriverInput: invalid kfac_constraint");
    }
    if (seen_controls.count("kfac_clip_width"))
      kfac_clip_width_ = parseNonnegativeReal(readControl("kfac_clip_width"), "kfac_clip_width");
    if (seen_controls.count("kfac_clip_quantile"))
      kfac_clip_quantile_ = parseNonnegativeReal(readControl("kfac_clip_quantile"), "kfac_clip_quantile");
    if (seen_controls.count("kfac_exclude_width") && readControl("kfac_exclude_width") != "none")
      kfac_exclude_width_ = parseNonnegativeReal(readControl("kfac_exclude_width"), "kfac_exclude_width");
    if (seen_controls.count("kfac_norm_constraint"))
      kfac_norm_constraint_ = parseNonnegativeReal(readControl("kfac_norm_constraint"), "kfac_norm_constraint");
    if (kfac_clip_width_ <= 0.0 || kfac_clip_quantile_ <= 0.0 || kfac_clip_quantile_ > 1.0 ||
        kfac_exclude_width_ <= 0.0)
      throw UniformCommunicateError("WFTrainDriverInput: invalid KFAC clipping width or quantile");
    if (kfac_objective_ == KFACObjective::COMPONENTWISE_MEAN_RESIDUAL &&
        kfac_gradient_ != KFACGradient::REAL_CENTERED_RESIDUAL)
      throw UniformCommunicateError("WFTrainDriverInput: componentwise_mean_residual requires real_centered_residual convention");
    if (kfac_objective_ == KFACObjective::RAW_COVARIANCE)
      rejectControls(seen_controls, {"kfac_clip_mode", "kfac_clip_width", "kfac_clip_quantile",
          "kfac_exclude_width"}, "raw KFAC objective");
    if (kfac_clip_mode_ != KFACClipMode::SOFT_MEDIAN)
      rejectControls(seen_controls, {"kfac_clip_quantile"}, "hard KFAC clipping");
    if (kfac_objective_ == KFACObjective::COMPONENTWISE_MAD_COVARIANCE ||
        kfac_objective_ == KFACObjective::COMPONENTWISE_MEAN_RESIDUAL)
    {
      rejectControls(seen_controls, {"kfac_clip_mode", "kfac_clip_width", "kfac_clip_quantile", "kfac_exclude_width"},
                     "componentwise median clipping");
      if (seen_controls.count("kfac_complex_clip_width"))
        kfac_complex_clip_width_ = parseNonnegativeReal(readControl("kfac_complex_clip_width"), "kfac_complex_clip_width");
      if (kfac_complex_clip_width_ <= 0)
        throw UniformCommunicateError("WFTrainDriverInput: kfac_complex_clip_width must be positive");
    }
    else
      rejectControls(seen_controls, {"kfac_complex_clip_width"}, "selected KFAC objective");
    if (kfac_constraint_ == KFACConstraint::DAMPED_KFAC)
    {
      if (kfac_norm_constraint_ <= 0.0)
        throw UniformCommunicateError("WFTrainDriverInput: damped_kfac requires positive kfac_norm_constraint");
    }
    else
      rejectControls(seen_controls, {"kfac_norm_constraint"}, "unconstrained KFAC");
    if (seen_controls.count("kfac_persist_state"))
    {
      const auto& value = readControl("kfac_persist_state");
      if (value != "yes" && value != "no")
        throw UniformCommunicateError("WFTrainDriverInput: kfac_persist_state must be yes or no");
      kfac_persist_state_ = value == "yes";
    }
    if (!kfac_persist_state_)
    {
      if (kfac_update_order_ != KFACUpdateOrder::POST_INVERSE_RATE ||
          kfac_schedule_arithmetic_ != KFACScheduleArithmetic::DIVIDE_RATE)
        throw UniformCommunicateError("WFTrainDriverInput: selected update order/schedule arithmetic requires persistent KFAC");
      rejectControls(seen_controls, {"kfac_ema_decay", "kfac_inverse_period", "kfac_learning_rate_decay",
          "kfac_schedule_precision", "kfac_schedule_arithmetic", "kfac_momentum", "kfac_burn_in", "kfac_curvature_period"},
          "stateless KFAC; set kfac_persist_state=yes to carry recurrence state");
    }
    if (seen_controls.count("kfac_ema_decay"))
      kfac_ema_decay_ = parseNonnegativeReal(readControl("kfac_ema_decay"), "kfac_ema_decay");
    if (kfac_ema_decay_ >= 1.0)
      throw UniformCommunicateError("WFTrainDriverInput: kfac_ema_decay must be below one");
    if (seen_controls.count("kfac_inverse_period"))
      kfac_inverse_period_ = parsePositiveSize(readControl("kfac_inverse_period"), "kfac_inverse_period");
    if (seen_controls.count("kfac_learning_rate_decay"))
      kfac_learning_rate_decay_ = parseNonnegativeReal(readControl("kfac_learning_rate_decay"), "kfac_learning_rate_decay");
    if (seen_controls.count("kfac_schedule_precision"))
    {
      const auto& value = readControl("kfac_schedule_precision");
      if (value != "float32" && value != "float64")
        throw UniformCommunicateError("WFTrainDriverInput: kfac_schedule_precision must be float32 or float64");
      kfac_schedule_float32_ = value == "float32";
    }
    if (seen_controls.count("kfac_momentum") &&
        parseNonnegativeReal(readControl("kfac_momentum"), "kfac_momentum") != 0.0)
      throw UniformCommunicateError("WFTrainDriverInput: persistent KFAC currently supports kfac_momentum=0 only");
    if (seen_controls.count("kfac_burn_in") &&
        parseNonnegativeSize(readControl("kfac_burn_in"), "kfac_burn_in") != 0)
      throw UniformCommunicateError("WFTrainDriverInput: persistent KFAC currently supports kfac_burn_in=0 only");
    if (seen_controls.count("kfac_curvature_period") &&
        parsePositiveSize(readControl("kfac_curvature_period"), "kfac_curvature_period") != 1)
      throw UniformCommunicateError("WFTrainDriverInput: persistent KFAC requires kfac_curvature_period=1");
  }
  if (gradient_estimator_ != GradientEstimator::LEGACY)
  {
    if (objective_ != Objective::ENERGY || (!first_order && optimizer_ != Optimizer::KFAC))
      throw UniformCommunicateError("WFTrainDriverInput: explicit gradient_estimator requires energy and a first-order optimizer or KFAC");
    if (gradient_estimator_ == GradientEstimator::FULL_LOCAL_ENERGY_RESPONSE && optimizer_ == Optimizer::KFAC)
      throw UniformCommunicateError("WFTrainDriverInput: explicit full_local_energy_response requires a first-order optimizer");
    if (gradient_estimator_ == GradientEstimator::SCORE_COVARIANCE && optimizer_ == Optimizer::KFAC &&
        (kfac_defaults_ != KFACDefaults::LEGACY ||
         (kfac_gradient_ != KFACGradient::LEGACY && kfac_gradient_ != KFACGradient::PHYSICAL_COVARIANCE) ||
         kfac_objective_ != KFACObjective::RAW_COVARIANCE))
      throw UniformCommunicateError("WFTrainDriverInput: score_covariance requires raw physical covariance without a KFAC preset");
    if (line_search_ != LineSearch::OFF)
      throw UniformCommunicateError("WFTrainDriverInput: explicit gradient_estimator currently requires line_search=off");
  }
}

} // namespace qmcplusplus
