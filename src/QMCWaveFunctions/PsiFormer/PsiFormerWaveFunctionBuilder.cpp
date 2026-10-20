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
#include "DeviceManager.h"
#include "Message/CommOperators.h"
#include "OhmmsData/AttributeSet.h"
#include "OhmmsData/XMLParsingString.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeterminant.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerModelFile.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPeriodicProfileInput.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexWF.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace qmcplusplus
{
namespace
{
/// Normalize the small backend token without locale-dependent comparisons.
std::string normalizedAcceleratorRequest(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return value;
}

/** Inspect operation-specific production readiness.
 * CUDA tiles retain their qualified coverage. Actual AMD runtime checks cover
 * HIP REAL and selected paired numerical products; native paired families
 * remain explicitly profile-gated below.
 */
psiformer::PsiFormerAcceleratorReadiness inspectProductionAcceleratorReadiness()
{
  const psiformer::PsiFormerCompiledAcceleratorSupport support =
      psiformer::compiledPsiFormerAcceleratorSupport();
  const bool usable_device = DeviceManager::getGlobal().getNumDevices() > 0;

  psiformer::PsiFormerAcceleratorReadiness readiness;
  readiness.cuda.compiled       = support.cuda;
  readiness.cuda.usable_device = support.cuda && usable_device;
  readiness.cuda.operation_coverage = psiformer::allPsiFormerAcceleratorOperations();
  readiness.cuda.runtime_validation = psiformer::PsiFormerRuntimeValidation::PASSED;
  readiness.hip.compiled        = support.hip;
  readiness.hip.usable_device  = support.hip && usable_device;
  readiness.hip.operation_coverage = psiformer::allPsiFormerAcceleratorOperations();
  readiness.hip.runtime_validation = psiformer::PsiFormerRuntimeValidation::PASSED;
  return readiness;
}

/** Resolve XML policy without querying a runtime for CPU or malformed requests. */
psiformer::PsiFormerAcceleratorResolution resolveAcceleratorRequest(
    const std::string& request,
    const PsiFormerAcceleratorReadinessProvider* readiness_provider,
    psiformer::PsiFormerAcceleratorOperationMask required_operations)
{
  const std::string normalized = normalizedAcceleratorRequest(request);
  if (normalized.empty() || normalized == "no" || normalized == "cpu")
    return psiformer::resolvePsiFormerAcceleratorRequest(normalized, {});

  // Let the common resolver provide precise unsupported/invalid diagnostics
  // without needlessly initializing or consulting any device runtime.
  if (normalized != "auto" && normalized != "yes" && normalized != "cuda" &&
      normalized != "hip")
    return psiformer::resolvePsiFormerAcceleratorRequest(normalized, {});

  psiformer::PsiFormerAcceleratorReadiness readiness = readiness_provider
      ? readiness_provider->inspect()
      : inspectProductionAcceleratorReadiness();
  // Both open Euclidean and fully periodic torus real cores are implemented.
  // Geometry metadata and fixed outer-twist ownership are validated below;
  // arbitrary complex neural weights and partial periodicity remain unsupported.
  return psiformer::resolvePsiFormerAcceleratorRequest(normalized, readiness, required_operations);
}

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

/// Parse a mandatory exact byte cap, excluding zero and the conventional unlimited sentinel.
std::size_t parseCloneNumericLimit(const std::string& text, const char* name = "max_clone_numeric_bytes")
{
  std::size_t bytes = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), bytes);
  if (error != std::errc{} || end != text.data() + text.size() || !bytes ||
      bytes == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument(std::string("Complex PsiFormer requires positive finite ") + name + " in exact bytes");
  return bytes;
}

/// Distinguish true atomic charges from explicit PP effective-valence metadata.
void validateComplexCoulombSource(const ParticleSet& ions, bool pseudopotential = false)
{
  const auto& species = ions.getSpeciesSet();
  const int charge = species.findAttribute("charge");
  if (charge < 0)
    throw std::invalid_argument(pseudopotential ? "Complex PsiFormer PP source requires ionic charges"
                                             : "Complex PsiFormer all-electron source requires ionic charges");
  for (int group = 0; group < species.size(); ++group)
  {
    const double z = species(charge, group);
    if (!psiformer::determinant::isFiniteReal(z) || z <= 0 || z != std::floor(z))
      throw std::invalid_argument(pseudopotential ? "Complex PsiFormer PP charges must be positive integers"
                                               : "Complex PsiFormer all-electron charges must be positive atomic numbers");
    for (const char* name : {"atomicnumber", "valence"})
    {
      const int attribute = species.findAttribute(name);
      const bool atomic = std::string_view(name) == "atomicnumber";
      if (attribute >= 0 && (!psiformer::determinant::isFiniteReal(species(attribute, group)) ||
          (pseudopotential && atomic ? species(attribute, group) < z ||
              species(attribute, group) != std::floor(species(attribute, group)) : species(attribute, group) != z)))
        throw std::invalid_argument(pseudopotential ? "Complex PsiFormer PP charge/atomicnumber/valence mismatch"
                                                 : "Complex PsiFormer all-electron charge/atomicnumber/valence mismatch");
    }
  }
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
  std::string parameter_checkpoint;
  std::string optimize = "no", optimize_scope = "indices", optimize_indices, gpu = "no";
  std::string precision_text = "fp64_reference";
  ParticleSet::PosType reduced_twist(0.0);

  // Both files use the compact export format consumed by PsiFormerNative.h.
  // Optimization remains off unless explicitly requested, preserving existing
  // inference inputs exactly.
  OhmmsAttributeSet attributes;
  attributes.add(name, "name");
  attributes.add(parameters, "parameters");
  attributes.add(configuration, "configuration");
  attributes.add(initialization, "initialization");
  attributes.add(parameter_checkpoint, "parameter_checkpoint");
  attributes.add(initialization_seed_text, "initialization_seed");
  attributes.add(feature_policy, "feature_policy");
  attributes.add(source, "source");
  attributes.add(system, "system");
  attributes.add(export_parameters, "export_parameters");
  attributes.add(optimize, "optimize");
  attributes.add(optimize_scope, "optimize_scope");
  attributes.add(optimize_indices, "optimize_indices");
  attributes.add(gpu, "gpu");
  attributes.add(precision_text, "precision");
  attributes.put(cur);

  // OhmmsAttributeSet tokenizes std::string values at whitespace. Read this
  // list directly so both documented separators survive XML parsing.
  if (xmlHasProp(cur, BAD_CAST "optimize_indices") != nullptr)
    optimize_indices = getXMLAttributeValue(cur, "optimize_indices");

  const bool direct_model = !parameters.empty() && configuration.empty() && initialization.empty();
  std::optional<psiformer::PsiFormerModelFile> imported_model;
  if (direct_model)
    imported_model.emplace(psiformer::readPsiFormerModelFile(parameters));
  const bool pair_profile = initialization == psiformer::PSIFORMER_PERIODIC_PAIR_V1 ||
      (imported_model && imported_model->parameters.environment.periodic_profile);
  auto* profile_node = psiformer::periodicProfileChild(cur);
  if (profile_node && (!pair_profile || direct_model))
    throw std::invalid_argument("periodic_profile XML requires fresh versioned initialization, not legacy or imported metadata");
  const auto profile_head = pair_profile && !direct_model ? psiformer::periodicProfileHead(profile_node)
      : psiformer::HeadRepresentation::REAL;
  const bool complex_profile = initialization == psiformer::PSIFORMER_COMPLEX_PERIODIC_V1 ||
      (imported_model && imported_model->paired_complex) ||
      (pair_profile && profile_head == psiformer::HeadRepresentation::PAIRED_REAL_COMPLEX);
  // Only selected paired two-cell AE metadata may proceed to the later HIP profile checks.
  // Legacy paired, PP, reciprocal and Fourier families remain independently closed.
  if (complex_profile && normalizedAcceleratorRequest(gpu) == "hip" &&
      !(pair_profile && system == "all_electron"))
    throw std::invalid_argument("Paired-complex HIP native profiles require separate runtime qualification");
  // This exact native ROOT is a value override, never manifest discovery or a restart request.
  if (xmlHasProp(cur, BAD_CAST "parameter_checkpoint"))
  {
    parameter_checkpoint = getXMLAttributeValue(cur, "parameter_checkpoint");
    if (!complex_profile || direct_model || parameter_checkpoint.empty())
      throw std::invalid_argument("parameter_checkpoint requires an exact native root and explicit paired initialization");
  }
  [[maybe_unused]] std::size_t clone_numeric_limit = 0;
  [[maybe_unused]] std::size_t gpu_numeric_limit = 0;
#ifndef QMC_COMPLEX
  if (complex_profile)
    throw std::invalid_argument("psiformer_complex_periodic_v1 requires a complex QMCPACK build");
#endif
#ifdef MIXED_PRECISION
  if (complex_profile)
    throw std::invalid_argument("psiformer_complex_periodic_v1 requires a full-precision QMCPACK build");
#endif

  // Resolve before parsing mutable target metadata or opening either model
  // file. Strict requests therefore fail without HDF5 or ParticleSet effects;
  // auto records an explicit CPU fallback for unsupported operations/geometries.
  const bool optimization_enabled = parseOptimizationFlag(optimize);
  const auto precision = psiformer::parsePsiFormerPrecisionPolicy(precision_text);
  const bool periodic = targetPtcl.getLattice().getSuperCellEnum() != SUPERCELL_OPEN;
  const auto pair_request = normalizedAcceleratorRequest(gpu);
  bool selected_pair_gpu = false;
#if !defined(QMC_COMPLEX) && !defined(MIXED_PRECISION)
  selected_pair_gpu = pair_profile && !complex_profile && system == "all_electron" &&
      (pair_request == "cuda" || pair_request == "hip");
#elif defined(QMC_COMPLEX) && !defined(MIXED_PRECISION)
  selected_pair_gpu = pair_profile && complex_profile && system == "all_electron" &&
      (pair_request == "cuda" || pair_request == "hip");
#endif
  if (pair_profile && (!periodic || (system != "all_electron" && !(complex_profile && system == "pseudopotential")) ||
      (pair_request != "no" && pair_request != "cpu" && !selected_pair_gpu) ||
      precision != psiformer::PsiFormerPrecisionPolicy::FP64_REFERENCE))
    throw std::invalid_argument("Versioned periodic profiles require CPU or the selected real GPU AE FP64 profile");
  if (complex_profile)
  {
    // Fail before readiness inspection, model I/O, parameter preparation or target-twist mutation.
    const auto request = normalizedAcceleratorRequest(gpu);
    const bool explicit_gpu = request == "cuda" || request == "hip";
    // A standalone model supplies parameters under the CURRENT validated PP Hamiltonian;
    // its values are not a sampler restart or a saved PP-provenance guarantee.
    const bool gpu_ecp = explicit_gpu && system == "pseudopotential" && parameter_checkpoint.empty() &&
        ((initialization == "psiformer_complex_periodic_v1" && parameters.empty()) || direct_model);
    const bool cpu_ecp = !explicit_gpu && (request == "no" || request == "cpu") &&
        (optimization_enabled || direct_model) && system == "pseudopotential" && parameter_checkpoint.empty() &&
        ((parameters.empty() && (initialization == psiformer::PSIFORMER_COMPLEX_PERIODIC_V1 || pair_profile)) || direct_model);
    const bool cpu_training = !explicit_gpu && (request == "no" || request == "cpu") &&
        optimization_enabled && optimize_scope == "all" && optimize_indices.empty() &&
        (system == "all_electron" || cpu_ecp);
    const bool selected_gpu_training = selected_pair_gpu && optimization_enabled &&
        optimize_scope == "all" && optimize_indices.empty();
    if ((!explicit_gpu && request != "no" && request != "cpu") ||
        (optimization_enabled && !cpu_training && !selected_gpu_training) ||
        precision != psiformer::PsiFormerPrecisionPolicy::FP64_REFERENCE)
      throw std::invalid_argument("Complex PsiFormer requires qualified CPU optimize_scope=all training or "
                                 "CPU/strict cuda/hip inference, and fp64_reference");
    if (targetPtcl.getLattice().getSuperCellEnum() != SUPERCELL_BULK ||
        feature_policy != (pair_profile ? "periodic_two_cell_v1" : "periodic_torus_v1") ||
        (system != "all_electron" && !gpu_ecp && !cpu_ecp) ||
        !xmlHasProp(cur, BAD_CAST "source") || (!direct_model && !parameters.empty()) || !configuration.empty() ||
        !export_parameters.empty() || !optimize_indices.empty() ||
        optimize_scope != ((cpu_training || selected_gpu_training) ? "all" : "indices"))
      throw std::invalid_argument("Complex PsiFormer requires explicit qualified periodic native metadata");
    clone_numeric_limit = parseCloneNumericLimit(getXMLAttributeValue(cur, "max_clone_numeric_bytes"));
    if (explicit_gpu)
      gpu_numeric_limit = parseCloneNumericLimit(getXMLAttributeValue(cur, "max_gpu_numeric_bytes"), "max_gpu_numeric_bytes");
    else if (xmlHasProp(cur, BAD_CAST "max_gpu_numeric_bytes"))
      throw std::invalid_argument("max_gpu_numeric_bytes requires the explicit complex GPU profile");
  }
  else if (xmlHasProp(cur, BAD_CAST "max_clone_numeric_bytes") || xmlHasProp(cur, BAD_CAST "max_gpu_numeric_bytes"))
    throw std::invalid_argument("Complex numeric caps belong only to psiformer_complex_periodic_v1");
  // Selected pair requests must validate their resolved metadata before consulting a device runtime.
  psiformer::PsiFormerAcceleratorResolution accelerator_resolution = selected_pair_gpu
      ? psiformer::PsiFormerAcceleratorResolution{}
      : resolveAcceleratorRequest(gpu, accelerator_readiness_provider_,
          optimization_enabled ? psiformer::allPsiFormerAcceleratorOperations()
                               : psiformer::psiFormerAcceleratorInferenceOperations());
  if (accelerator_readiness_provider_ && accelerator_resolution.backend !=
      psiformer::PsiFormerAcceleratorBackend::CPU)
    throw std::runtime_error(
        "PsiFormer numerical dispatch cannot be enabled by an injected test readiness provider");

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
  if (!internal_initialization && !direct_model && (!has_parameter_path || !has_configuration_path))
    throw std::runtime_error(
        "psiformer requires parameters and configuration HDF5 paths, or initialization");
  if (!internal_initialization && has_initialization_seed)
    throw std::invalid_argument(
        "PsiFormer initialization_seed requires internal initialization");
  if (internal_initialization && initialization != psiformer::DEEPQMC_PSIFORMER_V1 && !complex_profile && !pair_profile)
    throw std::invalid_argument("Unsupported PsiFormer initialization profile: " + initialization);
  const std::uint64_t initialization_seed = parseInitializationSeed(initialization_seed_text);

  if (targetPtcl.isSpinor())
    throw std::invalid_argument("PsiFormer does not support spinor electron particle sets");
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
  if (pair_profile && !complex_profile && (nonzero_twist || !particle_twist_is_zero))
    throw std::invalid_argument("Real versioned periodic profiles currently require Gamma twist");
  if (pair_profile && complex_profile && selected_pair_gpu && (nonzero_twist || !particle_twist_is_zero))
    throw std::invalid_argument("Selected paired GPU periodic profiles require Gamma twist");
  if (complex_profile && system == "pseudopotential" &&
      accelerator_resolution.backend != psiformer::PsiFormerAcceleratorBackend::CPU &&
      (nonzero_twist || !particle_twist_is_zero))
    throw std::invalid_argument("Native paired pseudopotential initialization requires Gamma twist");
  if (precision != psiformer::PsiFormerPrecisionPolicy::FP64_REFERENCE &&
      (precision != psiformer::PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS ||
       accelerator_resolution.backend == psiformer::PsiFormerAcceleratorBackend::CPU ||
       optimization_enabled || periodic || nonzero_twist || !particle_twist_is_zero))
    throw std::invalid_argument(
        "precision=fp32_dense_fp64_islands requires open-real GPU inference, optimize=no and zero twist");
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
  if (periodic && feature_policy != (pair_profile ? "periodic_two_cell_v1" : "periodic_torus_v1"))
    throw std::invalid_argument(
        pair_profile ? "Periodic PsiFormer requires explicit feature_policy=periodic_two_cell_v1 metadata" :
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
  if ((internal_initialization || direct_model) && system == "auto")
    throw std::invalid_argument(
        "Internally initialized PsiFormer requires explicit system=all_electron or system=pseudopotential");
  if ((internal_initialization || direct_model) && !has_explicit_source)
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
  if (complex_profile)
    validateComplexCoulombSource(*source_particles, system == "pseudopotential");

  std::unique_ptr<PsiFormerWF> component;
  const std::array<double, 3> twist_array{
      reduced_twist[0], reduced_twist[1], reduced_twist[2]};
  if (internal_initialization || direct_model)
  {
    if (targetPtcl.groups() != 2 || targetPtcl.groupsize(0) <= 0 || targetPtcl.groupsize(1) <= 0)
      throw std::invalid_argument(
          "Internally initialized PsiFormer requires two nonempty electron spin groups");

    psiformer::ModelShape shape{
        static_cast<std::size_t>(targetPtcl.groupsize(0)),
        static_cast<std::size_t>(targetPtcl.groupsize(1)),
        static_cast<std::size_t>(source_particles->getTotalNum()),
        /*determinants=*/16,
        /*feature_dimension=*/256,
        /*attention_heads=*/4,
        /*attention_blocks=*/4};
    psiformer::ExecutionEnvironment environment;
    if (complex_profile)
    {
      environment.head_representation = psiformer::HeadRepresentation::PAIRED_REAL_COMPLEX;
      environment.amplitude_scalar_domain = psiformer::ScalarDomain::COMPLEX;
    }
    if (periodic)
    {
      environment.boundary = psiformer::BoundaryCondition::PERIODIC;
      environment.geometry_feature_policy = psiformer::GeometryFeaturePolicy::PERIODIC_TORUS_V1;
      environment.periodic_axes = {true, true, true};
      for (std::size_t axis = 0; axis < 3; ++axis)
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          environment.lattice_vectors[axis][dimension] = targetPtcl.getLattice().R(axis, dimension);
    }
    if (pair_profile && !direct_model)
    {
      environment = psiformer::parsePeriodicProfileInput(profile_node, targetPtcl, *source_particles);
      shape.nuclei = environment.periodic_profile->primitive_motif.size();
    }
    if (direct_model)
      psiformer::validatePsiFormerModelFile(*imported_model, parameters, targetPtcl, *source_particles,
                                            system, twist_array);
    if (selected_pair_gpu)
    {
      const auto& selected_environment = direct_model ? imported_model->parameters.environment : environment;
      if (!(complex_profile ? psiformer::matchesPairedDevicePairMetadata(selected_environment)
                            : psiformer::matchesRealDevicePairMetadata(selected_environment)))
        throw std::invalid_argument("GPU periodic pair metadata requires Gamma log-complement atom-centered cusp physics");
    }
    if (selected_pair_gpu)
    {
      accelerator_resolution = resolveAcceleratorRequest(gpu, accelerator_readiness_provider_,
          optimization_enabled ? psiformer::allPsiFormerAcceleratorOperations()
                               : psiformer::psiFormerAcceleratorInferenceOperations());
      if (accelerator_readiness_provider_ &&
          accelerator_resolution.backend != psiformer::PsiFormerAcceleratorBackend::CPU)
        throw std::runtime_error("PsiFormer numerical dispatch cannot be enabled by an injected test readiness provider");
    }
    psiformer::InitializedPsiFormerParameters initialized = direct_model
        ? std::move(imported_model->parameters)
        : psiformer::initializePsiFormerParameters(shape, initialization_seed, initialization, environment);
    if (const auto& profile = initialized.environment.periodic_profile;
        profile && profile->envelope_policy == psiformer::PeriodicEnvelopePolicy::FOURIER)
    {
      // Report the resolved immutable physical-source gauge, including on standalone import.
      std::ostringstream resolved;
      resolved << std::setprecision(17) << "  Periodic envelope: policy=fourier; physical_ion_mean_origin=("
               << profile->fourier_origin[0] << ' ' << profile->fourier_origin[1] << ' ' << profile->fourier_origin[2]
               << "); modes=" << profile->fourier_modes.size();
      app_log() << resolved.str() << std::endl;
    }

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
#ifdef QMC_COMPLEX
    if (complex_profile)
    {
      auto complex = std::make_unique<PsiFormerComplexWF>(name, std::move(initialized), targetPtcl,
                                                         *source_particles, clone_numeric_limit, twist_array,
                                                         gpu_numeric_limit, parameter_checkpoint, system);
      const auto storage = complex->numericStorage();
      app_log() << (gpu_numeric_limit ? "  Complex PsiFormer GPU host G/L bytes per clone: required="
                                     : "  Complex PsiFormer CPU numeric bytes per clone: required=") << storage.required_bytes
                << " allocated=" << storage.allocated_bytes << " cap=" << storage.limit_bytes
                << "; excludes shared model, object/allocator overhead, ParticleSet, WFBuffer and aggregate clones; "
                << "not adaptive large-system qualification" << std::endl;
      if (gpu_numeric_limit)
        app_log() << "  Complex PsiFormer serialized B=1 GPU lane: device_bytes=" << storage.shared_gpu_bytes
                  << " cap=" << gpu_numeric_limit << " host_bytes=" << storage.shared_gpu_host_bytes
                  << " shared_image_bytes=" << storage.shared_image_bytes
                  << "; one model-shared image/lane, per-operation mutex, base mw scalar loops; CPU tapes=0, "
                  << "sampling/Hamiltonian=CPU, no batching/TARGET/aggregate-fit claim" << std::endl;
      if (adopt_explicit_twist)
        targetPtcl.setTwist(reduced_twist);
      return complex;
    }
#endif
    component = std::make_unique<PsiFormerWF>(
        name, std::move(initialized), targetPtcl, *source_particles, optimization_enabled,
        std::move(selected_indices), optimize_all, export_parameters, twist_array,
        accelerator_resolution, precision);
  }
  else if (periodic)
    component = std::make_unique<PsiFormerWF>(name, parameters, configuration, targetPtcl,
                                              *source_particles, optimization_enabled,
                                              std::move(selected_indices), optimize_all,
                                              export_parameters, twist_array,
                                              accelerator_resolution, precision);
  else
    component = std::make_unique<PsiFormerWF>(name, parameters, configuration, optimization_enabled,
                                              std::move(selected_indices), optimize_all,
                                              export_parameters, accelerator_resolution, precision);

  if (system != "auto")
    component->validateSystem(targetPtcl, *source_particles, system);
  // Publish framework twist ownership only after construction and system
  // validation succeed, so a failed builder call cannot mutate its target.
  if (adopt_explicit_twist)
    targetPtcl.setTwist(reduced_twist);
  return component;
}

} // namespace qmcplusplus
