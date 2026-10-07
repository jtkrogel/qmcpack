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
#include "Message/CommOperators.h"
#include "OhmmsData/AttributeSet.h"
#include "OhmmsData/XMLParsingString.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeterminant.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdlib>
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

/// Parse the full unsigned seed domain without accepting signs or trailing text.
std::uint64_t parseInitializationSeed(const std::string& value)
{
  std::uint64_t seed = 0;
  if (value.empty())
    throw std::invalid_argument("PsiFormer initialization_seed cannot be empty");
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), seed);
  if (error != std::errc{} || end != value.data() + value.size())
    throw std::invalid_argument("PsiFormer initialization_seed must be an unsigned integer");
  return seed;
}

/// Parse exactly three finite reduced-twist components from one XML attribute.
std::array<double, 3> parseReducedTwist(const std::string& value)
{
  std::array<double, 3> twist{};
  const char* cursor = value.c_str();
  const char* const end_of_value = cursor + value.size();
  for (double& component : twist)
  {
    while (cursor != end_of_value &&
           std::isspace(static_cast<unsigned char>(*cursor)))
      ++cursor;
    if (cursor == end_of_value)
      throw std::invalid_argument(
          "PsiFormer twist must contain exactly three components");

    errno = 0;
    char* parsed_end = nullptr;
    component = std::strtod(cursor, &parsed_end);
    if (parsed_end == cursor)
      throw std::invalid_argument("PsiFormer twist contains a malformed component");
    if (errno == ERANGE || !psiformer::determinant::isFiniteReal(component))
      throw std::invalid_argument("PsiFormer twist components must be finite");
    cursor = parsed_end;
  }
  while (cursor != end_of_value &&
         std::isspace(static_cast<unsigned char>(*cursor)))
    ++cursor;
  if (cursor != end_of_value)
    throw std::invalid_argument(
        "PsiFormer twist must contain exactly three components");
  return twist;
}
} // namespace

// Select HDF5 import or deterministic internal construction and validate optimizer controls.
std::unique_ptr<WaveFunctionComponent> PsiFormerWaveFunctionBuilder::buildComponent(xmlNodePtr cur)
{
  std::string name = "psiformer", parameters, configuration, source = "ion0", system = "auto";
  std::string export_parameters, initialization, initialization_seed_text = "0", feature_policy;
  std::string optimize = "no", optimize_scope = "indices", optimize_indices;
  ParticleSet::PosType reduced_twist(0.0);

  // Both files use the compact export format consumed by PsiFormerNative.h.
  // Optimization remains off unless explicitly requested, preserving existing
  // inference inputs exactly.
  OhmmsAttributeSet attributes;
  attributes.add(name, "name");
  attributes.add(parameters, "parameters");
  attributes.add(configuration, "configuration");
  attributes.add(initialization, "initialization");
  attributes.add(initialization_seed_text, "initialization_seed");
  attributes.add(feature_policy, "feature_policy");
  attributes.add(source, "source");
  attributes.add(system, "system");
  attributes.add(export_parameters, "export_parameters");
  attributes.add(optimize, "optimize");
  attributes.add(optimize_scope, "optimize_scope");
  attributes.add(optimize_indices, "optimize_indices");
  attributes.put(cur);

  // OhmmsAttributeSet tokenizes std::string values at whitespace. Read this
  // list directly so both documented separators survive XML parsing.
  if (xmlHasProp(cur, BAD_CAST "optimize_indices") != nullptr)
    optimize_indices = getXMLAttributeValue(cur, "optimize_indices");

  const bool internal_initialization = !initialization.empty();
  const bool has_parameter_path      = !parameters.empty();
  const bool has_configuration_path  = !configuration.empty();
  const bool has_initialization_seed = xmlHasProp(cur, BAD_CAST "initialization_seed") != nullptr;
  const bool has_explicit_source     = xmlHasProp(cur, BAD_CAST "source") != nullptr;
  const bool has_explicit_twist      = xmlHasProp(cur, BAD_CAST "twist") != nullptr;
  if (has_explicit_twist)
  {
    const std::array<double, 3> parsed_twist =
        parseReducedTwist(getXMLAttributeValue(cur, "twist"));
    for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
      reduced_twist[dimension] = parsed_twist[dimension];
  }
  if (internal_initialization && (has_parameter_path || has_configuration_path))
    throw std::invalid_argument(
        "PsiFormer internal initialization cannot be combined with parameters or configuration paths");
  if (!internal_initialization && (!has_parameter_path || !has_configuration_path))
    throw std::runtime_error(
        "psiformer requires parameters and configuration HDF5 paths, or initialization");
  if (!internal_initialization && has_initialization_seed)
    throw std::invalid_argument(
        "PsiFormer initialization_seed requires internal initialization");
  if (internal_initialization && initialization != psiformer::DEEPQMC_PSIFORMER_V1)
    throw std::invalid_argument("Unsupported PsiFormer initialization profile: " + initialization);
  const std::uint64_t initialization_seed = parseInitializationSeed(initialization_seed_text);

  const bool optimization_enabled = parseOptimizationFlag(optimize);
  if (targetPtcl.isSpinor())
    throw std::invalid_argument("PsiFormer does not support spinor electron particle sets");
  const bool periodic = targetPtcl.getLattice().getSuperCellEnum() != SUPERCELL_OPEN;
  const auto& particle_twist = targetPtcl.getTwist();
  bool particle_twist_is_zero = true;
  for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
  {
    if (!psiformer::determinant::isFiniteReal(particle_twist[dimension]))
      throw std::invalid_argument("PsiFormer target ParticleSet twist must be finite");
    particle_twist_is_zero =
        particle_twist_is_zero && particle_twist[dimension] == 0.0;
  }
  const bool adopt_explicit_twist = has_explicit_twist && particle_twist_is_zero;
  bool nonzero_twist = false;
  for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
  {
    if (!psiformer::determinant::isFiniteReal(reduced_twist[dimension]))
      throw std::invalid_argument("PsiFormer twist components must be finite");
    nonzero_twist = nonzero_twist || reduced_twist[dimension] != 0.0;
  }
  if (nonzero_twist && !periodic)
    throw std::invalid_argument("A nonzero PsiFormer twist requires a 3D bulk periodic cell");
  if (has_explicit_twist)
  {
    constexpr double twist_tolerance = 128.0 * std::numeric_limits<double>::epsilon();
    if (!particle_twist_is_zero)
      for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
        if (std::abs(particle_twist[dimension] - reduced_twist[dimension]) >
            twist_tolerance * (1.0 + std::abs(reduced_twist[dimension])))
          throw std::invalid_argument(
              "Explicit PsiFormer twist disagrees with the target ParticleSet twist");
  }
  else if (!particle_twist_is_zero)
    throw std::invalid_argument(
        "A nonzero target ParticleSet twist requires an explicit matching PsiFormer twist");
#ifndef QMC_COMPLEX
  if (nonzero_twist)
    throw std::invalid_argument("A nonzero PsiFormer twist requires a complex QMCPACK build");
#endif
  if (periodic && targetPtcl.getLattice().getSuperCellEnum() != SUPERCELL_BULK)
    throw std::invalid_argument("Periodic PsiFormer currently supports only 3D bulk periodic cells");
  if (periodic && feature_policy != "periodic_torus_v1")
    throw std::invalid_argument(
        "Periodic PsiFormer requires explicit feature_policy=periodic_torus_v1 metadata");
  if (!periodic && !feature_policy.empty() && feature_policy != "open_euclidean_v1")
    throw std::invalid_argument("Open PsiFormer requires feature_policy=open_euclidean_v1 when specified");
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
  if (internal_initialization && system == "auto")
    throw std::invalid_argument(
        "Internally initialized PsiFormer requires explicit system=all_electron or system=pseudopotential");
  if (internal_initialization && !has_explicit_source)
    throw std::invalid_argument("Internally initialized PsiFormer requires an explicit source particle set");
  if (periodic && !has_explicit_source)
    throw std::invalid_argument("Periodic PsiFormer requires an explicit ordered source particle set");
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

  const ParticleSet* source_particles = nullptr;
  if (system != "auto" || internal_initialization || periodic)
  {
    const auto source_particle_set = particle_sets_.find(source);
    if (source_particle_set == particle_sets_.end())
      throw std::invalid_argument("PsiFormer source particle set not found: " + source);
    source_particles = source_particle_set->second.get();
  }

  std::unique_ptr<PsiFormerWF> component;
  const std::array<double, 3> twist_array{
      reduced_twist[0], reduced_twist[1], reduced_twist[2]};
  if (internal_initialization)
  {
    if (targetPtcl.groups() != 2 || targetPtcl.groupsize(0) <= 0 || targetPtcl.groupsize(1) <= 0)
      throw std::invalid_argument(
          "Internally initialized PsiFormer requires two nonempty electron spin groups");

    const psiformer::ModelShape shape{
        static_cast<std::size_t>(targetPtcl.groupsize(0)),
        static_cast<std::size_t>(targetPtcl.groupsize(1)),
        static_cast<std::size_t>(source_particles->getTotalNum()),
        /*determinants=*/16,
        /*feature_dimension=*/256,
        /*attention_heads=*/4,
        /*attention_blocks=*/4};
    psiformer::ExecutionEnvironment environment;
    if (periodic)
    {
      environment.boundary = psiformer::BoundaryCondition::PERIODIC;
      environment.geometry_feature_policy = psiformer::GeometryFeaturePolicy::PERIODIC_TORUS_V1;
      environment.periodic_axes = {true, true, true};
      for (std::size_t axis = 0; axis < 3; ++axis)
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          environment.lattice_vectors[axis][dimension] = targetPtcl.getLattice().R(axis, dimension);
    }
    psiformer::InitializedPsiFormerParameters initialized =
        psiformer::initializePsiFormerParameters(shape, initialization_seed, initialization,
                                                 environment);

    // All ranks must evaluate exactly the same model even when platform math
    // libraries round the Gaussian transform differently.  Communicate's
    // pointer interface accepts an int count, so keep each collective bounded
    // instead of narrowing the complete parameter count.
    std::size_t broadcast_offset = 0;
    while (broadcast_offset < initialized.values.size())
    {
      const std::size_t remaining = initialized.values.size() - broadcast_offset;
      const int count = static_cast<int>(std::min(
          remaining, static_cast<std::size_t>(std::numeric_limits<int>::max())));
      myComm->bcast(initialized.values.data() + broadcast_offset, count);
      broadcast_offset += static_cast<std::size_t>(count);
    }
    component = std::make_unique<PsiFormerWF>(
        name, std::move(initialized), targetPtcl, *source_particles, optimization_enabled,
        std::move(selected_indices), optimize_all, export_parameters, twist_array);
  }
  else if (periodic)
    component = std::make_unique<PsiFormerWF>(name, parameters, configuration, targetPtcl,
                                              *source_particles, optimization_enabled,
                                              std::move(selected_indices), optimize_all,
                                              export_parameters, twist_array);
  else
    component = std::make_unique<PsiFormerWF>(name, parameters, configuration, optimization_enabled,
                                              std::move(selected_indices), optimize_all,
                                              export_parameters);

  if (system != "auto")
    component->validateSystem(targetPtcl, *source_particles, system);
  // Publish framework twist ownership only after construction and system
  // validation succeed, so a failed builder call cannot mutate its target.
  if (adopt_explicit_twist)
    targetPtcl.setTwist(reduced_twist);
  return component;
}

} // namespace qmcplusplus
