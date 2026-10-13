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
#include <unordered_set>

namespace qmcplusplus
{
namespace
{

constexpr std::array<std::string_view, 36> allowed_attributes{
    "objective", "optimizer", "parameter_chunk_size", "learning_rate", "adam_beta1",
    "adam_beta2", "epsilon", "damping", "damping_multiplier", "minimum_retry_damping",
    "maximum_damping_attempts", "maximum_update_norm", "maximum_metric_norm", "center_shift",
    "objective_tolerance", "spring_history_decay", "spring_maximum_update_norm_squared",
    "krylov_relative_tolerance", "krylov_absolute_tolerance", "krylov_maximum_iterations",
    "krylov_maximum_operator_applications", "krylov_maximum_seconds", "krylov_stagnation_window",
    "krylov_minimum_relative_improvement", "krylov_curvature_tolerance", "krylov_imaginary_tolerance",
    "maximum_retained_bytes", "checkpoint_path", "restart_path", "provider",
    "teacher", "overlap_target", "overlap_weighting", "target_provenance",
    "minimum_fidelity", "minimum_effective_samples"};

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

/** Reject attributes that do not apply to the selected optimizer. */
void rejectAttributes(const std::unordered_set<std::string>& seen_attributes,
                      std::initializer_list<std::string_view> attributes,
                      const std::string& optimizer)
{
  for (const std::string_view attribute : attributes)
    if (seen_attributes.find(std::string(attribute)) != seen_attributes.end())
      throw UniformCommunicateError("WFTrainDriverInput: " + std::string(attribute) +
                                    " is not valid for optimizer=\"" + optimizer + "\"");
}

/** Reject nested elements and non-whitespace text in the attribute-only section. */
void validateElementContent(xmlNodePtr training_node)
{
  for (xmlNodePtr child = training_node->children; child != nullptr; child = child->next)
  {
    if (child->type == XML_COMMENT_NODE)
      continue;
    if (child->type == XML_TEXT_NODE || child->type == XML_CDATA_SECTION_NODE)
    {
      const std::string text = castXMLCharToChar(child->content);
      if (std::all_of(text.begin(), text.end(), [](unsigned char character) { return std::isspace(character); }))
        continue;
    }
    throw UniformCommunicateError("WFTrainDriverInput: nested training content is not supported");
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

  validateElementContent(training_node);

  std::unordered_set<std::string> seen_attributes;
  for (xmlAttrPtr attribute = training_node->properties; attribute != nullptr; attribute = attribute->next)
  {
    const std::string name = castXMLCharToChar(attribute->name);
    if (std::find(allowed_attributes.begin(), allowed_attributes.end(), name) == allowed_attributes.end())
      throw UniformCommunicateError("WFTrainDriverInput: unknown training attribute " + name);
    if (!seen_attributes.insert(name).second)
      throw UniformCommunicateError("WFTrainDriverInput: duplicate training attribute " + name);
  }

  objective_                        = Objective::ENERGY;
  optimizer_                        = Optimizer::ADAM;
  parameter_chunk_size_             = 65536;
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
  checkpoint_path_.clear();
  restart_path_.clear();
  provider_id_.clear();
  teacher_wavefunction_.clear();
  overlap_target_            = OverlapTarget::GENERAL_FROZEN_TEACHER;
  overlap_weighting_         = OverlapWeighting::EQUAL;
  target_provenance_.clear();
  minimum_fidelity_          = 1.0e-14;
  minimum_effective_samples_ = 1.0;

  if (seen_attributes.find("objective") != seen_attributes.end())
  {
    const std::string objective = readAttribute(training_node, "objective");
    if (objective == "energy")
      objective_ = Objective::ENERGY;
    else if (objective == "determinant_overlap")
      objective_ = Objective::DETERMINANT_OVERLAP;
    else
      throw UniformCommunicateError("WFTrainDriverInput: unsupported objective=\"" + objective +
                                    "\"; supported objectives are energy and determinant_overlap");
  }
  if (seen_attributes.find("optimizer") != seen_attributes.end())
  {
    const std::string optimizer = readAttribute(training_node, "optimizer");
    if (optimizer == "adam")
      optimizer_ = Optimizer::ADAM;
    else if (optimizer == "sr")
      optimizer_ = Optimizer::STOCHASTIC_RECONFIGURATION;
    else if (optimizer == "minsr")
      optimizer_ = Optimizer::MINIMUM_STEP_SR;
    else if (optimizer == "spring")
      optimizer_ = Optimizer::SPRING;
    else
      throw UniformCommunicateError("WFTrainDriverInput: unsupported optimizer=\"" + optimizer +
                                    "\"; supported optimizers are adam, sr, minsr, and spring");
  }
  if (seen_attributes.find("parameter_chunk_size") != seen_attributes.end())
    parameter_chunk_size_ = parsePositiveSize(readAttribute(training_node, "parameter_chunk_size"),
                                              "parameter_chunk_size");
  if (seen_attributes.find("learning_rate") != seen_attributes.end())
  {
    learning_rate_ = parseFiniteReal(readAttribute(training_node, "learning_rate"), "learning_rate");
    if (!(learning_rate_ > 0.0))
      throw UniformCommunicateError("WFTrainDriverInput: learning_rate must be positive");
  }
  if (seen_attributes.find("adam_beta1") != seen_attributes.end())
    adam_beta1_ = parseDecay(readAttribute(training_node, "adam_beta1"), "adam_beta1");
  if (seen_attributes.find("adam_beta2") != seen_attributes.end())
    adam_beta2_ = parseDecay(readAttribute(training_node, "adam_beta2"), "adam_beta2");
  if (seen_attributes.find("epsilon") != seen_attributes.end())
  {
    epsilon_ = parseFiniteReal(readAttribute(training_node, "epsilon"), "epsilon");
    if (!(epsilon_ > 0.0))
      throw UniformCommunicateError("WFTrainDriverInput: epsilon must be positive");
  }
  if (seen_attributes.find("damping") != seen_attributes.end())
    damping_ = parseNonnegativeReal(readAttribute(training_node, "damping"), "damping");
  if (seen_attributes.find("damping_multiplier") != seen_attributes.end())
  {
    damping_multiplier_ = parseFiniteReal(readAttribute(training_node, "damping_multiplier"), "damping_multiplier");
    if (!(damping_multiplier_ > 1.0))
      throw UniformCommunicateError("WFTrainDriverInput: damping_multiplier must be greater than one");
  }
  if (seen_attributes.find("minimum_retry_damping") != seen_attributes.end())
  {
    minimum_retry_damping_ = parseFiniteReal(readAttribute(training_node, "minimum_retry_damping"),
                                             "minimum_retry_damping");
    if (!(minimum_retry_damping_ > 0.0))
      throw UniformCommunicateError("WFTrainDriverInput: minimum_retry_damping must be positive");
  }
  if (seen_attributes.find("maximum_damping_attempts") != seen_attributes.end())
    maximum_damping_attempts_ = parsePositiveSize(readAttribute(training_node, "maximum_damping_attempts"),
                                                  "maximum_damping_attempts");
  if (seen_attributes.find("maximum_update_norm") != seen_attributes.end())
    maximum_update_norm_ = parseNonnegativeReal(readAttribute(training_node, "maximum_update_norm"),
                                                "maximum_update_norm");
  if (seen_attributes.find("maximum_metric_norm") != seen_attributes.end())
    maximum_metric_norm_ = parseNonnegativeReal(readAttribute(training_node, "maximum_metric_norm"),
                                                "maximum_metric_norm");
  if (seen_attributes.find("center_shift") != seen_attributes.end())
    center_shift_ = parseNonnegativeReal(readAttribute(training_node, "center_shift"), "center_shift");
  if (seen_attributes.find("objective_tolerance") != seen_attributes.end())
    objective_tolerance_ = parseNonnegativeReal(readAttribute(training_node, "objective_tolerance"),
                                                "objective_tolerance");
  if (seen_attributes.find("spring_history_decay") != seen_attributes.end())
    spring_history_decay_ = parseDecay(readAttribute(training_node, "spring_history_decay"),
                                       "spring_history_decay");
  if (seen_attributes.find("spring_maximum_update_norm_squared") != seen_attributes.end())
    spring_maximum_update_norm_squared_ = parseNonnegativeReal(
        readAttribute(training_node, "spring_maximum_update_norm_squared"),
        "spring_maximum_update_norm_squared");
  if (seen_attributes.find("krylov_relative_tolerance") != seen_attributes.end())
    krylov_relative_tolerance_ = parseNonnegativeReal(
        readAttribute(training_node, "krylov_relative_tolerance"), "krylov_relative_tolerance");
  if (seen_attributes.find("krylov_absolute_tolerance") != seen_attributes.end())
    krylov_absolute_tolerance_ = parseNonnegativeReal(
        readAttribute(training_node, "krylov_absolute_tolerance"), "krylov_absolute_tolerance");
  if (seen_attributes.find("krylov_maximum_iterations") != seen_attributes.end())
    krylov_maximum_iterations_ = parsePositiveSize(readAttribute(training_node, "krylov_maximum_iterations"),
                                                   "krylov_maximum_iterations");
  if (seen_attributes.find("krylov_maximum_operator_applications") != seen_attributes.end())
    krylov_maximum_operator_applications_ = parsePositiveSize(
        readAttribute(training_node, "krylov_maximum_operator_applications"),
        "krylov_maximum_operator_applications");
  if (seen_attributes.find("krylov_maximum_seconds") != seen_attributes.end())
    krylov_maximum_seconds_ = parseNonnegativeReal(readAttribute(training_node, "krylov_maximum_seconds"),
                                                   "krylov_maximum_seconds");
  if (seen_attributes.find("krylov_stagnation_window") != seen_attributes.end())
    krylov_stagnation_window_ = parseNonnegativeSize(readAttribute(training_node, "krylov_stagnation_window"),
                                                     "krylov_stagnation_window");
  if (seen_attributes.find("krylov_minimum_relative_improvement") != seen_attributes.end())
    krylov_minimum_relative_improvement_ = parseNonnegativeReal(
        readAttribute(training_node, "krylov_minimum_relative_improvement"),
        "krylov_minimum_relative_improvement");
  if (seen_attributes.find("krylov_curvature_tolerance") != seen_attributes.end())
    krylov_curvature_tolerance_ = parseNonnegativeReal(
        readAttribute(training_node, "krylov_curvature_tolerance"), "krylov_curvature_tolerance");
  if (seen_attributes.find("krylov_imaginary_tolerance") != seen_attributes.end())
  {
    krylov_imaginary_tolerance_ = parseFiniteReal(
        readAttribute(training_node, "krylov_imaginary_tolerance"), "krylov_imaginary_tolerance");
    if (!(krylov_imaginary_tolerance_ > 0.0))
      throw UniformCommunicateError("WFTrainDriverInput: krylov_imaginary_tolerance must be positive");
  }
  if (seen_attributes.find("maximum_retained_bytes") != seen_attributes.end())
    maximum_retained_bytes_ = parseNonnegativeSize(readAttribute(training_node, "maximum_retained_bytes"),
                                                   "maximum_retained_bytes");
  if (seen_attributes.find("checkpoint_path") != seen_attributes.end())
    checkpoint_path_ = readAttribute(training_node, "checkpoint_path");
  if (seen_attributes.find("restart_path") != seen_attributes.end())
    restart_path_ = readAttribute(training_node, "restart_path");
  if (seen_attributes.find("provider") != seen_attributes.end())
  {
    provider_id_ = readAttribute(training_node, "provider");
    if (provider_id_.empty())
      throw UniformCommunicateError("WFTrainDriverInput: provider must not be empty when specified");
  }
  if (seen_attributes.find("teacher") != seen_attributes.end())
  {
    teacher_wavefunction_ = readAttribute(training_node, "teacher");
    if (teacher_wavefunction_.empty())
      throw UniformCommunicateError("WFTrainDriverInput: teacher must not be empty when specified");
  }
  if (seen_attributes.find("overlap_target") != seen_attributes.end())
  {
    const std::string target = readAttribute(training_node, "overlap_target");
    if (target == "general")
      overlap_target_ = OverlapTarget::GENERAL_FROZEN_TEACHER;
    else if (target == "top_coefficients")
      overlap_target_ = OverlapTarget::TOP_COEFFICIENT_MULTIDETERMINANT;
    else
      throw UniformCommunicateError(
          "WFTrainDriverInput: overlap_target must be general or top_coefficients");
  }
  if (seen_attributes.find("overlap_weighting") != seen_attributes.end())
  {
    const std::string weighting = readAttribute(training_node, "overlap_weighting");
    if (weighting == "equal")
      overlap_weighting_ = OverlapWeighting::EQUAL;
    else if (weighting == "coefficient_squared")
      overlap_weighting_ = OverlapWeighting::COEFFICIENT_SQUARED;
    else
      throw UniformCommunicateError(
          "WFTrainDriverInput: overlap_weighting must be equal or coefficient_squared");
  }
  if (seen_attributes.find("target_provenance") != seen_attributes.end())
  {
    target_provenance_ = readAttribute(training_node, "target_provenance");
    if (target_provenance_.empty())
      throw UniformCommunicateError(
          "WFTrainDriverInput: target_provenance must not be empty when specified");
  }
  if (seen_attributes.find("minimum_fidelity") != seen_attributes.end())
  {
    minimum_fidelity_ = parseFiniteReal(readAttribute(training_node, "minimum_fidelity"),
                                        "minimum_fidelity");
    if (minimum_fidelity_ < 0.0 || minimum_fidelity_ > 1.0)
      throw UniformCommunicateError("WFTrainDriverInput: minimum_fidelity must lie in [0, 1]");
  }
  if (seen_attributes.find("minimum_effective_samples") != seen_attributes.end())
    minimum_effective_samples_ = parseNonnegativeReal(
        readAttribute(training_node, "minimum_effective_samples"),
        "minimum_effective_samples");

  const std::initializer_list<std::string_view> overlap_attributes = {
      "teacher", "overlap_target", "overlap_weighting", "target_provenance",
      "minimum_fidelity", "minimum_effective_samples"};
  if (objective_ == Objective::ENERGY)
    rejectAttributes(seen_attributes, overlap_attributes, "energy objective");
  else
  {
    if (teacher_wavefunction_.empty() || target_provenance_.empty())
      throw UniformCommunicateError(
          "WFTrainDriverInput: determinant_overlap requires teacher and target_provenance");
    if (optimizer_ != Optimizer::ADAM)
      throw UniformCommunicateError(
          "WFTrainDriverInput: determinant_overlap currently requires optimizer=\"adam\"");
    if (overlap_weighting_ == OverlapWeighting::COEFFICIENT_SQUARED &&
        overlap_target_ != OverlapTarget::TOP_COEFFICIENT_MULTIDETERMINANT)
      throw UniformCommunicateError(
          "WFTrainDriverInput: coefficient_squared weighting requires top_coefficients targets");
  }

  const std::initializer_list<std::string_view> adam_attributes = {"adam_beta1", "adam_beta2", "epsilon"};
  const std::initializer_list<std::string_view> matrix_free_attributes = {
      "damping", "damping_multiplier", "minimum_retry_damping", "maximum_damping_attempts",
      "maximum_update_norm", "maximum_metric_norm", "center_shift", "objective_tolerance",
      "spring_history_decay", "spring_maximum_update_norm_squared", "krylov_relative_tolerance",
      "krylov_absolute_tolerance", "krylov_maximum_iterations", "krylov_maximum_operator_applications",
      "krylov_maximum_seconds", "krylov_stagnation_window", "krylov_minimum_relative_improvement",
      "krylov_curvature_tolerance", "krylov_imaginary_tolerance"};
  const std::initializer_list<std::string_view> sr_only_attributes = {
      "damping_multiplier", "minimum_retry_damping", "maximum_damping_attempts", "maximum_metric_norm"};
  const std::initializer_list<std::string_view> minimum_step_attributes = {"center_shift",
                                                                           "objective_tolerance"};
  const std::initializer_list<std::string_view> spring_only_attributes = {
      "spring_history_decay", "spring_maximum_update_norm_squared"};
  switch (optimizer_)
  {
  case Optimizer::ADAM:
    rejectAttributes(seen_attributes, matrix_free_attributes, "adam");
    break;
  case Optimizer::STOCHASTIC_RECONFIGURATION:
    rejectAttributes(seen_attributes, adam_attributes, "sr");
    rejectAttributes(seen_attributes, minimum_step_attributes, "sr");
    rejectAttributes(seen_attributes, spring_only_attributes, "sr");
    break;
  case Optimizer::MINIMUM_STEP_SR:
    rejectAttributes(seen_attributes, adam_attributes, "minsr");
    rejectAttributes(seen_attributes, sr_only_attributes, "minsr");
    rejectAttributes(seen_attributes, spring_only_attributes, "minsr");
    if (!(damping_ > 0.0))
      throw UniformCommunicateError("WFTrainDriverInput: optimizer=\"minsr\" requires positive damping");
    break;
  case Optimizer::SPRING:
    rejectAttributes(seen_attributes, adam_attributes, "spring");
    rejectAttributes(seen_attributes, sr_only_attributes, "spring");
    if (!(damping_ > 0.0))
      throw UniformCommunicateError("WFTrainDriverInput: optimizer=\"spring\" requires positive damping");
    break;
  }
  if (optimizer_ != Optimizer::ADAM && krylov_relative_tolerance_ == 0.0 &&
      krylov_absolute_tolerance_ == 0.0)
    throw UniformCommunicateError(
        "WFTrainDriverInput: matrix-free optimization requires a positive Krylov tolerance");
}

} // namespace qmcplusplus
