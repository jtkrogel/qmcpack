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
#include <cmath>
#include <limits>
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
  std::string name = "psiformer", parameters, configuration, source = "ion0", system = "auto";
  std::string export_parameters;
  std::string optimize = "no", optimize_scope = "indices", optimize_indices;

  // Both files use the compact export format consumed by PsiFormerNative.h.
  // Optimization remains off unless explicitly requested, preserving existing
  // inference inputs exactly.
  OhmmsAttributeSet attributes;
  attributes.add(name, "name");
  attributes.add(parameters, "parameters");
  attributes.add(configuration, "configuration");
  attributes.add(source, "source");
  attributes.add(system, "system");
  attributes.add(export_parameters, "export_parameters");
  attributes.add(optimize, "optimize");
  attributes.add(optimize_scope, "optimize_scope");
  attributes.add(optimize_indices, "optimize_indices");
  attributes.put(cur);
  if (parameters.empty() || configuration.empty())
    throw std::runtime_error("psiformer requires parameters and configuration HDF5 paths");

  const bool optimization_enabled = parseOptimizationFlag(optimize);
  if (targetPtcl.isSpinor())
    throw std::invalid_argument("PsiFormer does not support spinor electron particle sets");
  if (targetPtcl.getLattice().getSuperCellEnum() != SUPERCELL_OPEN)
    throw std::invalid_argument(
        "PsiFormer supports only open-boundary molecular particle sets; periodic execution is not implemented");
  if (optimization_enabled)
  {
    const auto& masses = targetPtcl.get_mass_by_group();
    if (masses.size() == 0 ||
        masses.size() < static_cast<std::size_t>(targetPtcl.groups()) ||
        !targetPtcl.isSameMass() ||
        std::abs(masses[0] - 1.0) > 64.0 * std::numeric_limits<double>::epsilon())
      throw std::invalid_argument(
          "PsiFormer optimization requires initialized unit electron masses; "
          "inference observables are mass independent");
  }
  if (system != "auto" && system != "all_electron" && system != "pseudopotential")
    throw std::invalid_argument("PsiFormer system must be auto, all_electron, or pseudopotential");
  if (optimization_enabled && system == "auto")
    throw std::invalid_argument(
        "PsiFormer optimization requires system=all_electron or system=pseudopotential for metadata validation");
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

  auto component = std::make_unique<PsiFormerWF>(name, parameters, configuration, optimization_enabled,
                                                 std::move(selected_indices), optimize_all, export_parameters);
  if (system != "auto")
  {
    const auto source_particle_set = particle_sets_.find(source);
    if (source_particle_set == particle_sets_.end())
      throw std::invalid_argument("PsiFormer source particle set not found: " + source);
    component->validateSystem(targetPtcl, *source_particle_set->second, system);
  }
  return component;
}

} // namespace qmcplusplus
