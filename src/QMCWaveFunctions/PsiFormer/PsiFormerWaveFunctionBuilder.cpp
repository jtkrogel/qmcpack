//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerWaveFunctionBuilder.cpp
 * @brief XML construction of the PsiFormer wavefunction component.
 */
#include "QMCWaveFunctions/PsiFormer/PsiFormerWaveFunctionBuilder.h"
#include "OhmmsData/AttributeSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace qmcplusplus
{
namespace
{
/// Parse the conventional QMCPACK yes/no spellings used for XML attributes.
bool parseOptimizationFlag(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  if (value == "yes" || value == "true" || value == "1")
    return true;
  if (value == "no" || value == "false" || value == "0")
    return false;
  throw std::invalid_argument("PsiFormer optimize must be yes/no, true/false, or 1/0");
}

/// Parse a whitespace- or comma-separated list of nonnegative canonical flat indices.
std::vector<std::size_t> parseFlatIndices(std::string values)
{
  std::replace(values.begin(), values.end(), ',', ' ');
  std::istringstream stream(values);
  std::vector<std::size_t> indices;
  long long index;
  while (stream >> index)
  {
    if (index < 0)
      throw std::invalid_argument("PsiFormer optimize_indices cannot contain a negative index");
    indices.push_back(static_cast<std::size_t>(index));
  }
  if (!stream.eof())
    throw std::invalid_argument("PsiFormer optimize_indices contains a malformed index");
  return indices;
}
} // namespace

// Validate export paths and the initial selected-index optimization input.
std::unique_ptr<WaveFunctionComponent> PsiFormerWaveFunctionBuilder::buildComponent(xmlNodePtr cur)
{
  std::string name = "psiformer", parameters, configuration;
  std::string optimize = "no", optimize_scope = "indices", optimize_indices;

  // Both files use the compact export format consumed by PsiFormerNative.h.
  // Optimization remains off unless explicitly requested, preserving existing
  // inference inputs exactly.
  OhmmsAttributeSet attributes;
  attributes.add(name, "name");
  attributes.add(parameters, "parameters");
  attributes.add(configuration, "configuration");
  attributes.add(optimize, "optimize");
  attributes.add(optimize_scope, "optimize_scope");
  attributes.add(optimize_indices, "optimize_indices");
  attributes.put(cur);
  if (parameters.empty() || configuration.empty())
    throw std::runtime_error("psiformer requires parameters and configuration HDF5 paths");

  const bool optimization_enabled = parseOptimizationFlag(optimize);
  std::vector<std::size_t> selected_indices = parseFlatIndices(optimize_indices);
  if (!optimization_enabled && !selected_indices.empty())
    throw std::invalid_argument("PsiFormer optimize_indices requires optimize=yes");
  if (optimize_scope != "indices" && optimize_scope != "all")
    throw std::invalid_argument("PsiFormer optimize_scope must be indices or all");
  if (!optimization_enabled && optimize_scope != "indices")
    throw std::invalid_argument("PsiFormer optimize_scope requires optimize=yes");
  const bool optimize_all = optimization_enabled && optimize_scope == "all";
  if (optimize_all && !selected_indices.empty())
    throw std::invalid_argument("PsiFormer optimize_scope=all cannot be combined with optimize_indices");
  if (optimization_enabled && !optimize_all && selected_indices.empty())
    throw std::invalid_argument("PsiFormer optimize=yes requires a nonempty optimize_indices list");

  return std::make_unique<PsiFormerWF>(name, parameters, configuration, optimization_enabled,
                                       std::move(selected_indices), optimize_all);
}

} // namespace qmcplusplus
