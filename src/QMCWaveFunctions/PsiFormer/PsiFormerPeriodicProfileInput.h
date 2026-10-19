//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_PERIODIC_PROFILE_INPUT_H
#define QMCPLUSPLUS_PSIFORMER_PERIODIC_PROFILE_INPUT_H

#include "Particle/ParticleSet.h"
#include "OhmmsData/XMLParsingString.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeterminant.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerFourierHead.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace qmcplusplus::psiformer
{

/// Locate one opt-in metadata child without changing any legacy child interpretation.
inline xmlNodePtr periodicProfileChild(xmlNodePtr parent)
{
  xmlNodePtr result = nullptr;
  for (auto* child = parent->children; child; child = child->next)
    if (child->type == XML_ELEMENT_NODE && std::string_view(castXMLCharToChar(child->name)) == "periodic_profile")
    {
      if (result) throw std::invalid_argument("PsiFormer accepts only one periodic_profile child");
      result = child;
    }
  return result;
}

/// Read the explicit head representation before resolving profile-specific builder admission.
inline HeadRepresentation periodicProfileHead(xmlNodePtr profile)
{
  if (!profile) throw std::invalid_argument("Periodic initialization requires one periodic_profile child");
  const auto value = getXMLAttributeValue(profile, "head");
  if (value == "paired_complex") return HeadRepresentation::PAIRED_REAL_COMPLEX;
  if (value == "real") return HeadRepresentation::REAL;
  throw std::invalid_argument("periodic_profile requires head=paired_complex or real");
}

/// Reject nested markup and return the exact scalar-text payload owned by one node.
inline std::string periodicProfileText(xmlNodePtr node)
{
  for (auto* child = node->children; child; child = child->next)
    if (child->type != XML_COMMENT_NODE && child->type != XML_TEXT_NODE && child->type != XML_CDATA_SECTION_NODE)
      throw std::invalid_argument("Periodic profile parameters require scalar text only");
  std::unique_ptr<xmlChar, decltype(xmlFree)> content(xmlNodeGetContent(node), xmlFree);
  return content ? std::string(castXMLCharToChar(content.get())) : std::string{};
}

/// Parse a finite numerical list, rejecting trailing tokens instead of truncating metadata.
inline std::vector<double> periodicProfileScalars(const std::string& text)
{
  std::istringstream stream(text);
  std::vector<double> result;
  double value;
  while (stream >> value)
  {
    if (!determinant::isFiniteReal(value))
      throw std::invalid_argument("Periodic profile numerical metadata must be finite");
    result.push_back(value);
  }
  if (!stream.eof() || result.empty())
    throw std::invalid_argument("Periodic profile numerical metadata is empty or malformed");
  return result;
}

/// Convert only exactly represented bounded integers to cell and occupied-index coordinates.
inline int periodicProfileInteger(double value)
{
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max() ||
      value != std::floor(value))
    throw std::invalid_argument("Periodic profile index or tiling entry must be a bounded integer");
  return static_cast<int>(value);
}

/// Build an immutable two-cell contract while keeping the actual simulation ParticleSet authoritative.
inline ExecutionEnvironment parsePeriodicProfileInput(xmlNodePtr node, const ParticleSet& electrons,
                                                       const ParticleSet& ions)
{
  if (!node || node->ns || electrons.getLattice().getSuperCellEnum() != SUPERCELL_BULK || electrons.isSpinor() ||
      ions.getLattice().getSuperCellEnum() != SUPERCELL_BULK)
    throw std::invalid_argument("Periodic profile requires explicit nonspinor bulk metadata");
  // Admit one strict schema with all fields explicit; no partial profile acquires inferred defaults.
  for (auto* attribute = node->properties; attribute; attribute = attribute->next)
  {
    const std::string name = castXMLCharToChar(attribute->name);
    if (attribute->ns || (name != "version" && name != "head" && name != "feature_transform" &&
                         name != "electron_electron_cusp" && name != "envelope_policy"))
      throw std::invalid_argument("Unknown periodic_profile attribute: " + name);
  }
  auto metadata = std::make_shared<PeriodicProfileMetadata>();
  if (getXMLAttributeValue(node, "version") != "1")
    throw std::invalid_argument("periodic_profile requires version=1");
  const auto transform = getXMLAttributeValue(node, "feature_transform");
  if (transform == "raw_trigonometric_v1") metadata->feature_transform = PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1;
  else if (transform == "log_complement_v1") metadata->feature_transform = PeriodicFeatureTransform::LOG_COMPLEMENT_V1;
  else if (transform == "polynomial_v1") metadata->feature_transform = PeriodicFeatureTransform::POLYNOMIAL_V1;
  else throw std::invalid_argument("Periodic profile requires an explicit supported feature_transform");
  metadata->pair_feature_dimension = periodicPairFeatureArity(metadata->feature_transform);
  const auto cusp = getXMLAttributeValue(node, "electron_electron_cusp");
  if (cusp != "yes" && cusp != "no")
    throw std::invalid_argument("periodic_profile requires electron_electron_cusp=yes or no");
  metadata->electron_electron_cusp = cusp == "yes";
  if (metadata->feature_transform == PeriodicFeatureTransform::POLYNOMIAL_V1 && metadata->electron_electron_cusp)
    throw std::invalid_argument("Polynomial periodic features require electron_electron_cusp=no");
  const auto envelope = getXMLAttributeValue(node, "envelope_policy");
  if (envelope == "fourier") metadata->envelope_policy = PeriodicEnvelopePolicy::FOURIER;
  else if (xmlHasProp(node, BAD_CAST "envelope_policy") && envelope != "atom_centered")
    throw std::invalid_argument("Periodic envelope_policy must be atom_centered or fourier");
  const bool fourier = metadata->envelope_policy == PeriodicEnvelopePolicy::FOURIER;
  std::vector<std::string> names{"primitive_lattice", "tiling", "primitive_motif", "motif_species", "occupied_columns"};
  if (fourier) names.push_back("fourier_modes");
  std::unordered_map<std::string, std::string> fields;
  for (auto* child = node->children; child; child = child->next)
  {
    if (child->type == XML_COMMENT_NODE) continue;
    if (child->type == XML_TEXT_NODE || child->type == XML_CDATA_SECTION_NODE)
    {
      const std::string text = castXMLCharToChar(child->content);
      if (std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isspace(c); })) continue;
    }
    if (child->type != XML_ELEMENT_NODE || child->ns || std::string_view(castXMLCharToChar(child->name)) != "parameter")
      throw std::invalid_argument("periodic_profile children must be named parameter elements");
    for (auto* attribute = child->properties; attribute; attribute = attribute->next)
      if (attribute->ns || std::string_view(castXMLCharToChar(attribute->name)) != "name")
        throw std::invalid_argument("Periodic profile parameter accepts only its name attribute");
    const auto name = getXMLAttributeValue(child, "name");
    if (std::find(names.begin(), names.end(), name) == names.end() ||
        !fields.emplace(name, periodicProfileText(child)).second)
      throw std::invalid_argument("Unknown or duplicate periodic profile parameter: " + name);
  }
  if (fields.size() != names.size())
    throw std::invalid_argument("Periodic profile requires primitive_lattice, tiling, primitive_motif, motif_species and occupied_columns" +
        std::string(fourier ? ", plus fourier_modes for Fourier envelopes" : ""));
  ExecutionEnvironment environment;
  environment.boundary = BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
  environment.head_representation = periodicProfileHead(node);
  if (fourier && environment.head_representation != HeadRepresentation::PAIRED_REAL_COMPLEX)
    throw std::invalid_argument("Fourier envelopes require paired_complex periodic heads");
  environment.amplitude_scalar_domain = environment.head_representation == HeadRepresentation::REAL ? ScalarDomain::REAL : ScalarDomain::COMPLEX;
  environment.periodic_axes = {true, true, true};
  // Physical ParticleSets define the simulation lattice; the feature cell cannot replace it.
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
    {
      environment.lattice_vectors[i][j] = electrons.getLattice().R(i, j);
      const auto expected = environment.lattice_vectors[i][j];
      if (std::abs(ions.getLattice().R(i, j) - expected) >
          128 * std::numeric_limits<double>::epsilon() * (1 + std::abs(expected)))
        throw std::invalid_argument("Periodic profile physical ion lattice differs from the simulation lattice");
    }
  metadata->cells.simulation = {pf::GeometryBoundaryKind::PERIODIC, environment.lattice_vectors, environment.periodic_axes};
  metadata->cells.primitive = metadata->cells.simulation;
  const auto lattice = periodicProfileScalars(fields.at("primitive_lattice"));
  const auto tiling = periodicProfileScalars(fields.at("tiling"));
  if (lattice.size() != 9 || tiling.size() != 9)
    throw std::invalid_argument("Periodic profile lattices and tiling require exactly nine entries");
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
    {
      metadata->cells.primitive.lattice_vectors[i][j] = lattice[3 * i + j];
      metadata->cells.tiling[i][j] = periodicProfileInteger(tiling[3 * i + j]);
    }
  pf::validatePeriodicGeometryMapping(metadata->cells);
  // Match the primitive motif against every physical replica before constructing feature tensors.
  const auto motif = periodicProfileScalars(fields.at("primitive_motif"));
  if (motif.size() % 3)
    throw std::invalid_argument("Periodic primitive motif requires complete Cartesian triples");
  for (std::size_t ion = 0; ion < motif.size() / 3; ++ion)
    metadata->primitive_motif.push_back({motif[3 * ion], motif[3 * ion + 1], motif[3 * ion + 2]});
  std::istringstream species_stream(fields.at("motif_species"));
  std::string label;
  const auto& species = ions.getSpeciesSet();
  while (species_stream >> label)
  {
    const auto found = std::find(species.speciesName.begin(), species.speciesName.end(), label);
    if (found == species.speciesName.end())
      throw std::invalid_argument("Periodic motif species does not exist in the physical source: " + label);
    metadata->motif_species.push_back(static_cast<int>(found - species.speciesName.begin()));
  }
  if (metadata->motif_species.size() != metadata->primitive_motif.size())
    throw std::invalid_argument("Periodic motif requires one explicit physical species label per feature center");
  std::vector<double> physical_positions;
  std::vector<int> physical_species;
  for (int ion = 0; ion < ions.getTotalNum(); ++ion)
  {
    for (int d = 0; d < 3; ++d) physical_positions.push_back(ions.R[ion][d]);
    physical_species.push_back(ions.GroupID[ion]);
  }
  (void)pf::mapPeriodicIonMotif(metadata->cells,
      pf::GeometryPositionView::interleaved(motif.data(), motif.size() / 3), metadata->motif_species,
      pf::GeometryPositionView::interleaved(physical_positions.data(), ions.getTotalNum()), physical_species);
  if (fourier)
  {
    // The fixed gauge uses every physical ion before any primitive-feature substitution.
    const auto modes = periodicProfileScalars(fields.at("fourier_modes"));
    if (modes.empty() || modes.size() % 3 || modes.size() / 3 > FOURIER_MODE_LIMIT)
      throw std::invalid_argument("Fourier modes require a bounded nonempty list of integer triplets");
    for (std::size_t mode = 0; mode < modes.size() / 3; ++mode)
      metadata->fourier_modes.push_back({periodicProfileInteger(modes[3 * mode]),
          periodicProfileInteger(modes[3 * mode + 1]), periodicProfileInteger(modes[3 * mode + 2])});
    metadata->fourier_origin = physicalIonFourierOrigin(
        pf::GeometryPositionView::interleaved(physical_positions.data(), ions.getTotalNum()));
    (void)FourierHeadModes(metadata->cells.simulation, metadata->fourier_modes, metadata->fourier_origin);
  }
  // Preserve explicit spin and band column order while decoding reciprocal phases and occupations.
  const auto occupied = periodicProfileScalars(fields.at("occupied_columns"));
  if (occupied.size() != 6 * static_cast<std::size_t>(electrons.getTotalNum()))
    throw std::invalid_argument("Occupied columns require one Gx Gy Gz band spin occupation row per electron");
  for (std::size_t column = 0; column < occupied.size() / 6; ++column)
  {
    const auto band = periodicProfileInteger(occupied[6 * column + 3]);
    if (band < 0) throw std::invalid_argument("Occupied band indices must be nonnegative");
    metadata->occupied_columns.push_back({{periodicProfileInteger(occupied[6 * column]),
        periodicProfileInteger(occupied[6 * column + 1]), periodicProfileInteger(occupied[6 * column + 2])},
        static_cast<std::size_t>(band), periodicProfileInteger(occupied[6 * column + 4]), occupied[6 * column + 5]});
  }
  environment.periodic_profile = std::move(metadata);
  return environment;
}

} // namespace qmcplusplus::psiformer
#endif
