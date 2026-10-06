//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: QMCPACK developers
//////////////////////////////////////////////////////////////////////////////////////

#include "BatchMemoryInput.h"

#include "Message/UniformCommunicateError.h"
#include "ModernStringUtils.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace qmcplusplus
{
namespace
{

/** Parse the deliberately small IEC byte grammar with checked multiplication. */
std::size_t parseByteCount(const std::string& text, const std::string& attribute)
{
  const std::array<std::pair<std::string, std::size_t>, 4> suffixes{{
      {"GiB", std::size_t{1} << 30},
      {"MiB", std::size_t{1} << 20},
      {"KiB", std::size_t{1} << 10},
      {"B", 1},
  }};

  for (const auto& [suffix, multiplier] : suffixes)
    if (text.size() > suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0)
    {
      const std::string digits = text.substr(0, text.size() - suffix.size());
      if (!std::all_of(digits.begin(), digits.end(), [](unsigned char character) { return std::isdigit(character); }))
        break;

      std::size_t value = 0;
      for (const char digit : digits)
      {
        const std::size_t numeric_digit = static_cast<std::size_t>(digit - '0');
        if (value > (std::numeric_limits<std::size_t>::max() - numeric_digit) / 10)
          throw UniformCommunicateError("BatchMemoryInput: " + attribute + " overflows size_t");
        value = value * 10 + numeric_digit;
      }
      try
      {
        return checkedBatchMemoryMultiply(value, multiplier, attribute);
      }
      catch (const std::overflow_error& error)
      {
        throw UniformCommunicateError("BatchMemoryInput: " + std::string(error.what()));
      }
    }

  throw UniformCommunicateError("BatchMemoryInput: " + attribute +
                                " must be a nonnegative base-ten integer followed by B, KiB, MiB, or GiB");
}

/** Parse exactly "auto" or one positive base-ten integer. */
BatchTileRequest parseTileRequest(const std::string& text, const std::string& attribute)
{
  if (text == "auto")
    return BatchTileRequest::automatic();
  if (text.empty() ||
      !std::all_of(text.begin(), text.end(), [](unsigned char character) { return std::isdigit(character); }))
    throw UniformCommunicateError("BatchMemoryInput: " + attribute + " must be auto or a positive integer");

  std::size_t capacity = 0;
  for (const char digit : text)
  {
    const std::size_t numeric_digit = static_cast<std::size_t>(digit - '0');
    if (capacity > (std::numeric_limits<std::size_t>::max() - numeric_digit) / 10)
      throw UniformCommunicateError("BatchMemoryInput: " + attribute + " overflows size_t");
    capacity = capacity * 10 + numeric_digit;
  }
  if (capacity == 0)
    throw UniformCommunicateError("BatchMemoryInput: " + attribute + " must be positive");
  return BatchTileRequest::fixed(capacity);
}

/** Preserve the complete attribute spelling so whitespace and trailing text are rejected. */
std::string readCompleteValue(std::istringstream& value)
{
  return {std::istreambuf_iterator<char>(value), std::istreambuf_iterator<char>()};
}

} // namespace

BatchMemoryInput::BatchMemoryInput()
{
  section_name = "batch_memory";
  attributes   = {"host_budget", "device_budget", "value_tile", "full_vgl_tile", "active_gradient_tile",
                  "ecp_outer_tile"};
  custom       = attributes;
}

BatchMemoryInput::BatchMemoryInput(xmlNodePtr cur) : BatchMemoryInput()
{
  if (cur == nullptr)
    throw UniformCommunicateError("BatchMemoryInput: cannot parse a null XML node");

  // InputSection accepts a root "type" attribute as an alternative section
  // identifier. This section is identified only by its element name, so scan
  // explicitly to retain the promised reject-unknown-attributes contract.
  for (xmlAttrPtr attribute = cur->properties; attribute != nullptr; attribute = attribute->next)
  {
    const std::string name = lowerCase(castXMLCharToChar(attribute->name));
    if (attributes.find(name) == attributes.end())
      throw UniformCommunicateError("BatchMemoryInput: unknown attribute " + name);
  }

  for (xmlNodePtr child = cur->children; child != nullptr; child = child->next)
    if (child->type == XML_TEXT_NODE || child->type == XML_CDATA_SECTION_NODE)
    {
      const std::string text = castXMLCharToChar(child->content);
      if (std::any_of(text.begin(), text.end(), [](unsigned char character) { return !std::isspace(character); }))
        throw UniformCommunicateError("BatchMemoryInput: element content is not supported");
    }
  readXML(cur);
}

void BatchMemoryInput::setFromStreamCustom(const std::string& element_name,
                                           const std::string& name,
                                           std::istringstream& value)
{
  if (!element_name.empty())
    throw UniformCommunicateError("BatchMemoryInput: nested values are not supported");
  if (!seen_attributes_.insert(name).second)
    throw UniformCommunicateError("BatchMemoryInput: duplicate attribute " + name);

  const std::string text = readCompleteValue(value);
  if (name == "host_budget")
    policy_.host_budget = parseByteCount(text, name);
  else if (name == "device_budget")
    policy_.device_budget = parseByteCount(text, name);
  else if (name == "value_tile")
    policy_.tiles.value = parseTileRequest(text, name);
  else if (name == "full_vgl_tile")
    policy_.tiles.full_vgl = parseTileRequest(text, name);
  else if (name == "active_gradient_tile")
    policy_.tiles.active_gradient = parseTileRequest(text, name);
  else if (name == "ecp_outer_tile")
    policy_.tiles.ecp_outer = parseTileRequest(text, name);
  else
    throw UniformCommunicateError("BatchMemoryInput: unknown attribute " + name);
}

} // namespace qmcplusplus
