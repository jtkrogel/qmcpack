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
#include <limits>
#include <string_view>
#include <unordered_set>

namespace qmcplusplus
{
namespace
{

constexpr std::array<std::string_view, 10> allowed_attributes{
    "objective", "optimizer", "parameter_chunk_size", "learning_rate", "adam_beta1",
    "adam_beta2", "epsilon", "checkpoint_path", "restart_path", "provider"};

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

  objective_            = Objective::ENERGY;
  optimizer_            = Optimizer::ADAM;
  parameter_chunk_size_ = 65536;
  learning_rate_        = 1.0e-3;
  adam_beta1_           = 0.9;
  adam_beta2_           = 0.999;
  epsilon_              = 1.0e-8;
  checkpoint_path_.clear();
  restart_path_.clear();
  provider_id_.clear();

  if (seen_attributes.find("objective") != seen_attributes.end())
  {
    const std::string objective = readAttribute(training_node, "objective");
    if (objective != "energy")
      throw UniformCommunicateError("WFTrainDriverInput: unsupported objective=\"" + objective +
                                    "\"; the initial driver supports exactly energy");
  }
  if (seen_attributes.find("optimizer") != seen_attributes.end())
  {
    const std::string optimizer = readAttribute(training_node, "optimizer");
    if (optimizer != "adam")
      throw UniformCommunicateError("WFTrainDriverInput: unsupported optimizer=\"" + optimizer +
                                    "\"; the initial driver supports exactly adam");
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

  if (!checkpoint_path_.empty() && checkpoint_path_ == restart_path_)
    throw UniformCommunicateError("WFTrainDriverInput: checkpoint_path and restart_path must differ");
}

} // namespace qmcplusplus
