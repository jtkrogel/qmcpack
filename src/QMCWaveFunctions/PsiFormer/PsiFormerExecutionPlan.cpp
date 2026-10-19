//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerExecutionPlan.cpp
 * @brief Validation and typed indexing for immutable PsiFormer execution metadata.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qmcplusplus::psiformer
{
namespace
{

/// Return true when a DeepQMC module path has the requested terminal suffix.
bool endsWith(const std::string& value, const std::string& suffix)
{
  return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// Return the scalar product of a row-major tensor shape, including scalar shapes.
std::size_t shapeProduct(const std::vector<std::size_t>& shape)
{
  std::size_t result = 1;
  for (std::size_t extent : shape)
  {
    if (extent != 0 && result > std::numeric_limits<std::size_t>::max() / extent)
      throw std::overflow_error("PsiFormer parameter tensor size overflow");
    result *= extent;
  }
  return result;
}

/// Parse DeepQMC's implicit-first, suffixed-later attention-block naming scheme.
std::optional<std::size_t> attentionBlock(const std::string& module)
{
  constexpr char layer_marker[] = "/electron_gnn_layer";
  const std::size_t marker      = module.find(layer_marker);
  if (marker == std::string::npos)
    return std::nullopt;

  const std::size_t suffix = marker + sizeof(layer_marker) - 1;
  if (suffix >= module.size())
    return std::nullopt;
  if (module[suffix] == '/')
    return std::size_t{0};
  if (module[suffix] != '_')
    return std::nullopt;

  std::size_t cursor = suffix + 1;
  if (cursor >= module.size() || !std::isdigit(static_cast<unsigned char>(module[cursor])))
    return std::nullopt;

  std::size_t block = 0;
  while (cursor < module.size() && std::isdigit(static_cast<unsigned char>(module[cursor])))
  {
    block = 10 * block + static_cast<std::size_t>(module[cursor] - '0');
    ++cursor;
  }
  if (cursor >= module.size() || module[cursor] != '/')
    return std::nullopt;
  return block;
}

/// Classify one leaf while retaining the unchanged legacy role and block identities.
struct ClassifiedParameter
{
  ParameterRole role;
  std::size_t block;
  HeadTensorComponent component = HeadTensorComponent::REAL;
};

/// Translate one string-keyed layout record into a role, block and head component.
ClassifiedParameter classifyParameter(const ParameterLayoutInput& layout)
{
  if (endsWith(layout.module, "/electron_embedding/linear") && layout.name == "w")
    return {ParameterRole::ELECTRON_EMBEDDING_WEIGHT, NO_ATTENTION_BLOCK};
  if (endsWith(layout.module, "/Backflow/~/mlp/linear_0") && layout.name == "w")
    return {ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK};
  if (endsWith(layout.module, "/Backflow_1/~/mlp/linear_0") && layout.name == "w")
    return {ParameterRole::BACKFLOW_DOWN_WEIGHT, NO_ATTENTION_BLOCK};
  if (endsWith(layout.module, "/Backflow/~/mlp/linear_0") && layout.name == "w_imag")
    return {ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG};
  if (endsWith(layout.module, "/Backflow_1/~/mlp/linear_0") && layout.name == "w_imag")
    return {ParameterRole::BACKFLOW_DOWN_WEIGHT, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG};
  if (endsWith(layout.module, "/exponential_envelopes"))
  {
    if (layout.name == "pi_up")
      return {ParameterRole::ENVELOPE_PI_UP, NO_ATTENTION_BLOCK};
    if (layout.name == "pi_down")
      return {ParameterRole::ENVELOPE_PI_DOWN, NO_ATTENTION_BLOCK};
    if (layout.name == "zetas_up")
      return {ParameterRole::ENVELOPE_ZETA_UP, NO_ATTENTION_BLOCK};
    if (layout.name == "zetas_down")
      return {ParameterRole::ENVELOPE_ZETA_DOWN, NO_ATTENTION_BLOCK};
  }
  if (endsWith(layout.module, "/electronic_cusp_asymptotic"))
  {
    if (layout.name == "same_alpha")
      return {ParameterRole::CUSP_SAME_ALPHA, NO_ATTENTION_BLOCK};
    if (layout.name == "anti_alpha")
      return {ParameterRole::CUSP_OPPOSITE_ALPHA, NO_ATTENTION_BLOCK};
  }

  const std::optional<std::size_t> block = attentionBlock(layout.module);
  if (block)
  {
    if (endsWith(layout.module, "/pair_attention") && layout.name == "w")
      return {ParameterRole::PAIR_ATTENTION_WEIGHT, *block};
    if (endsWith(layout.module, "/pair_attention") && layout.name == "b")
      return {ParameterRole::PAIR_ATTENTION_BIAS, *block};
    if (endsWith(layout.module, "/multi_head_attention/query") && layout.name == "w")
      return {ParameterRole::ATTENTION_QUERY_WEIGHT, *block};
    if (endsWith(layout.module, "/multi_head_attention/key") && layout.name == "w")
      return {ParameterRole::ATTENTION_KEY_WEIGHT, *block};
    if (endsWith(layout.module, "/multi_head_attention/value") && layout.name == "w")
      return {ParameterRole::ATTENTION_VALUE_WEIGHT, *block};
    if (endsWith(layout.module, "/multi_head_attention/linear") && layout.name == "w")
      return {ParameterRole::ATTENTION_OUTPUT_WEIGHT, *block};
    if (endsWith(layout.module, "/mlp/linear_0") && layout.name == "w")
      return {ParameterRole::UPDATE_HIDDEN_WEIGHT, *block};
    if (endsWith(layout.module, "/mlp/linear_0") && layout.name == "b")
      return {ParameterRole::UPDATE_HIDDEN_BIAS, *block};
    if (endsWith(layout.module, "/mlp/linear_1") && layout.name == "w")
      return {ParameterRole::UPDATE_OUTPUT_WEIGHT, *block};
    if (endsWith(layout.module, "/mlp/linear_1") && layout.name == "b")
      return {ParameterRole::UPDATE_OUTPUT_BIAS, *block};
  }

  throw std::invalid_argument("Unrecognized PsiFormer parameter tensor " + layout.module + "/" + layout.name);
}

/// Return the exact tensor shape required for one role by the current architecture.
std::vector<std::size_t> expectedShape(ParameterRole role,
                                       const ModelShape& model,
                                       const ExecutionEnvironment& environment)
{
  const std::size_t electrons = model.electrons();
  switch (role)
  {
  case ParameterRole::ELECTRON_EMBEDDING_WEIGHT:
    return {(environment.periodic_profile ? periodicFeatureArity(environment.periodic_profile->feature_transform) :
             environment.boundary == BoundaryCondition::PERIODIC ? 7 : 4) * model.nuclei + 1,
            model.feature_dimension};
  case ParameterRole::ATTENTION_QUERY_WEIGHT:
  case ParameterRole::ATTENTION_KEY_WEIGHT:
  case ParameterRole::ATTENTION_VALUE_WEIGHT:
  case ParameterRole::ATTENTION_OUTPUT_WEIGHT:
  case ParameterRole::UPDATE_HIDDEN_WEIGHT:
  case ParameterRole::UPDATE_OUTPUT_WEIGHT:
    return {model.feature_dimension, model.feature_dimension};
  case ParameterRole::UPDATE_HIDDEN_BIAS:
  case ParameterRole::UPDATE_OUTPUT_BIAS:
    return {model.feature_dimension};
  case ParameterRole::PAIR_ATTENTION_WEIGHT:
    return {environment.periodic_profile ? environment.periodic_profile->pair_feature_dimension : 8, model.attention_heads};
  case ParameterRole::PAIR_ATTENTION_BIAS:
    return {model.attention_heads};
  case ParameterRole::BACKFLOW_UP_WEIGHT:
  case ParameterRole::BACKFLOW_DOWN_WEIGHT:
    return {model.feature_dimension, model.determinants * electrons};
  case ParameterRole::ENVELOPE_PI_UP:
  case ParameterRole::ENVELOPE_PI_DOWN:
  case ParameterRole::ENVELOPE_ZETA_UP:
  case ParameterRole::ENVELOPE_ZETA_DOWN:
    return {model.determinants * electrons, model.nuclei};
  case ParameterRole::CUSP_SAME_ALPHA:
  case ParameterRole::CUSP_OPPOSITE_ALPHA:
    return {};
  case ParameterRole::COUNT:
    break;
  }
  throw std::logic_error("Unhandled PsiFormer parameter role");
}

/// Report a tensor-shape mismatch with enough information to diagnose an export.
[[noreturn]] void throwShapeMismatch(const ParameterLayoutInput& layout,
                                     ParameterRole role,
                                     const std::vector<std::size_t>& expected)
{
  auto format_shape = [](const std::vector<std::size_t>& shape) {
    std::ostringstream output;
    output << '[';
    for (std::size_t axis = 0; axis < shape.size(); ++axis)
    {
      if (axis)
        output << ',';
      output << shape[axis];
    }
    output << ']';
    return output.str();
  };

  throw std::invalid_argument(std::string("PsiFormer parameter ") + parameterRoleName(role) + " has shape " +
                              format_shape(layout.shape) + ", expected " + format_shape(expected));
}

} // namespace

// Validate the complete opt-in geometry, feature and occupied-column contract before allocation.
void validatePeriodicProfileMetadata(const ModelShape& model, const ExecutionEnvironment& environment)
{
  if (!environment.periodic_profile)
  {
    if (environment.geometry_feature_policy == GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1)
      throw std::invalid_argument("Two-cell periodic features require explicit versioned metadata");
    return;
  }
  const auto& profile = *environment.periodic_profile;
  if (profile.version != 1 || environment.boundary != BoundaryCondition::PERIODIC ||
      environment.geometry_feature_policy != GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1 ||
      (environment.head_representation != HeadRepresentation::PAIRED_REAL_COMPLEX &&
       environment.head_representation != HeadRepresentation::REAL) ||
      environment.parameter_scalar_domain != ScalarDomain::REAL ||
      environment.compute_scalar_domain != ScalarDomain::REAL ||
      environment.amplitude_scalar_domain != (environment.head_representation == HeadRepresentation::REAL ?
          ScalarDomain::REAL : ScalarDomain::COMPLEX) || !environment.fixed_nuclei ||
      profile.pair_feature_dimension != periodicPairFeatureArity(profile.feature_transform) ||
      (profile.feature_transform == PeriodicFeatureTransform::POLYNOMIAL_V1 && profile.electron_electron_cusp))
    throw std::invalid_argument("Unsupported versioned periodic feature/head metadata");
  pf::validatePeriodicGeometryMapping(profile.cells);
  if (profile.cells.simulation.lattice_vectors != environment.lattice_vectors ||
      profile.cells.simulation.periodic_axes != environment.periodic_axes ||
      profile.primitive_motif.size() != model.nuclei || profile.motif_species.size() != model.nuclei ||
      model.nuclei == 0 || profile.occupied_columns.size() != model.electrons())
    throw std::invalid_argument("Periodic profile cell, motif or occupied-column extent disagrees with model");
  // An identity replication validates feature-site uniqueness without replacing physical ions.
  pf::PeriodicGeometryMapping identity;
  identity.primitive = identity.simulation = profile.cells.primitive;
  const auto motif = pf::GeometryPositionView::interleaved(profile.primitive_motif.front().data(), model.nuclei);
  (void)pf::mapPeriodicIonMotif(identity, motif, profile.motif_species, motif, profile.motif_species);
  for (std::size_t column = 0; column < profile.occupied_columns.size(); ++column)
  {
    const auto& occupied = profile.occupied_columns[column];
    if (occupied.spin != (column < model.spin_up_electrons ? 0 : 1) || occupied.occupation != 1)
      throw std::invalid_argument("Periodic profile requires fully occupied columns in spin-major order");
    if (environment.head_representation == HeadRepresentation::REAL && occupied.reciprocal != std::array<int, 3>{})
      throw std::invalid_argument("Real periodic heads require zero reciprocal column vectors");
    for (std::size_t previous = 0; previous < column; ++previous)
      if (profile.occupied_columns[previous].spin == occupied.spin &&
          profile.occupied_columns[previous].band == occupied.band &&
          profile.occupied_columns[previous].reciprocal == occupied.reciprocal)
        throw std::invalid_argument("Periodic profile repeats an occupied spin band reciprocal tuple");
  }
}

// Validate model dimensions and translate each canonical parameter interval once.
PsiFormerExecutionPlan::PsiFormerExecutionPlan(ModelShape model_shape,
                                               std::vector<ParameterLayoutInput> parameter_layouts,
                                               ExecutionEnvironment environment)
    : model_shape_(std::move(model_shape)), environment_(environment)
{
  validatePeriodicProfileMetadata(model_shape_, environment_);
  if (environment_.periodic_profile)
    environment_.periodic_profile = std::make_shared<const PeriodicProfileMetadata>(*environment_.periodic_profile);
  const ModelCapabilities support = capabilities();
  if ((environment_.boundary == BoundaryCondition::OPEN && !support.open_boundary) ||
      (environment_.boundary == BoundaryCondition::PERIODIC && !support.periodic_boundary))
    throw std::invalid_argument("Requested PsiFormer boundary condition is not implemented");
  if (environment_.boundary == BoundaryCondition::OPEN &&
      environment_.geometry_feature_policy != GeometryFeaturePolicy::OPEN_EUCLIDEAN_V1)
    throw std::invalid_argument("Open PsiFormer execution requires open_euclidean_v1 features");
  if (environment_.boundary == BoundaryCondition::PERIODIC)
  {
    if (environment_.geometry_feature_policy != GeometryFeaturePolicy::PERIODIC_TORUS_V1 &&
        environment_.geometry_feature_policy != GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1)
      throw std::invalid_argument("Periodic PsiFormer execution requires periodic_torus_v1 features");
    if (!std::all_of(environment_.periodic_axes.begin(), environment_.periodic_axes.end(),
                     [](bool periodic) { return periodic; }))
      throw std::invalid_argument("Real Gamma PsiFormer currently supports only 3D bulk periodic cells");
  }
  if (environment_.head_representation != HeadRepresentation::REAL &&
      environment_.head_representation != HeadRepresentation::PAIRED_REAL_COMPLEX)
    throw std::invalid_argument("Unknown PsiFormer orbital head representation");
  const bool paired_head = environment_.head_representation == HeadRepresentation::PAIRED_REAL_COMPLEX;
  if (paired_head && (environment_.boundary != BoundaryCondition::PERIODIC ||
                     environment_.parameter_scalar_domain != ScalarDomain::REAL ||
                     environment_.compute_scalar_domain != ScalarDomain::REAL ||
                     environment_.amplitude_scalar_domain != ScalarDomain::COMPLEX))
    throw std::invalid_argument("Paired-real PsiFormer heads require periodic real parameters/trunk and complex amplitude metadata");
  const auto require_supported_scalar_domain = [&support](ScalarDomain domain, const char* description) {
    if (domain != ScalarDomain::REAL && domain != ScalarDomain::COMPLEX)
      throw std::invalid_argument(std::string("Unknown PsiFormer ") + description + " scalar domain");
    if ((domain == ScalarDomain::REAL && !support.real_scalars) ||
        (domain == ScalarDomain::COMPLEX && !support.complex_scalars))
      throw std::invalid_argument(std::string("Requested PsiFormer ") + description + " scalar domain is not implemented");
  };
  require_supported_scalar_domain(environment_.parameter_scalar_domain, "parameter");
  require_supported_scalar_domain(environment_.compute_scalar_domain, "compute");
  // This is schema admission only. Runtime capability reporting remains real-only.
  if (!paired_head)
    require_supported_scalar_domain(environment_.amplitude_scalar_domain, "amplitude");
  if (environment_.fixed_nuclei ? !support.fixed_nuclei : !support.moving_nuclei)
    throw std::invalid_argument("Requested PsiFormer nuclear-motion policy is not implemented");

  if (model_shape_.spin_up_electrons == 0 || model_shape_.spin_down_electrons == 0 || model_shape_.nuclei == 0 ||
      model_shape_.determinants == 0 || model_shape_.feature_dimension == 0 || model_shape_.attention_heads == 0 ||
      model_shape_.attention_blocks == 0)
    throw std::invalid_argument("PsiFormer execution plan dimensions must be positive");
  if (model_shape_.feature_dimension % model_shape_.attention_heads != 0)
    throw std::invalid_argument("PsiFormer feature dimension must be divisible by the attention-head count");
  if (parameter_layouts.empty())
    throw std::invalid_argument("PsiFormer execution plan requires parameter tensors");

  const std::size_t lookup_stride = model_shape_.attention_blocks + 1;
  descriptor_lookup_.assign(PARAMETER_ROLE_COUNT * lookup_stride, NO_ATTENTION_BLOCK);
  if (paired_head)
    imaginary_head_lookup_.assign(2, NO_ATTENTION_BLOCK);
  std::size_t expected_begin = 0;
  parameter_tensors_.reserve(parameter_layouts.size());
  for (ParameterLayoutInput& layout : parameter_layouts)
  {
    if (layout.begin != expected_begin || layout.end < layout.begin)
      throw std::invalid_argument("PsiFormer parameter intervals are not contiguous in canonical order");
    if (shapeProduct(layout.shape) != layout.end - layout.begin)
      throw std::invalid_argument("PsiFormer parameter tensor shape does not match its flat interval");

    const auto [role, block, component] = classifyParameter(layout);
    if ((role == ParameterRole::PAIR_ATTENTION_WEIGHT || role == ParameterRole::PAIR_ATTENTION_BIAS) &&
        !environment_.periodic_profile)
      throw std::invalid_argument("Pair-attention tensors require a versioned periodic profile");
    if (component == HeadTensorComponent::IMAG && !paired_head)
      throw std::invalid_argument("Imaginary PsiFormer head tensors require paired-real head metadata");
    if (block != NO_ATTENTION_BLOCK && block >= model_shape_.attention_blocks)
      throw std::invalid_argument("PsiFormer parameter references an out-of-range attention block");
    const std::vector<std::size_t> expected_shape = expectedShape(role, model_shape_, environment_);
    if (layout.shape != expected_shape)
      throwShapeMismatch(layout, role, expected_shape);
    const std::size_t block_slot = block == NO_ATTENTION_BLOCK ? 0 : block + 1;
    const std::size_t lookup_slot = component == HeadTensorComponent::IMAG ?
        (role == ParameterRole::BACKFLOW_UP_WEIGHT ? 0 : 1) :
        static_cast<std::size_t>(role) * lookup_stride + block_slot;
    auto& lookup = component == HeadTensorComponent::IMAG ? imaginary_head_lookup_ : descriptor_lookup_;
    if (lookup[lookup_slot] != NO_ATTENTION_BLOCK)
      throw std::invalid_argument(std::string("Duplicate PsiFormer parameter role ") + parameterRoleName(role));

    lookup[lookup_slot] = parameter_tensors_.size();
    parameter_tensors_.push_back(
        {role, block, std::move(layout.module), std::move(layout.name), std::move(layout.shape),
         layout.begin, layout.end, component});
    expected_begin = layout.end;
  }
  parameter_count_ = expected_begin;

  // Every role is required except same-spin alpha when neither spin population
  // contains a pair.  DeepQMC does not materialize that unused leaf for a 1+1
  // system, while older portable exports may still contain it.
  const std::array<ParameterRole, 8> required_global_roles{
      ParameterRole::ELECTRON_EMBEDDING_WEIGHT,
      ParameterRole::BACKFLOW_UP_WEIGHT,
      ParameterRole::BACKFLOW_DOWN_WEIGHT,
      ParameterRole::ENVELOPE_PI_UP,
      ParameterRole::ENVELOPE_PI_DOWN,
      ParameterRole::ENVELOPE_ZETA_UP,
      ParameterRole::ENVELOPE_ZETA_DOWN,
      ParameterRole::CUSP_OPPOSITE_ALPHA};
  for (ParameterRole role : required_global_roles)
    if (!hasParameter(role))
      throw std::invalid_argument(std::string("PsiFormer parameter export is missing required role ") +
                                  parameterRoleName(role));
  for (std::size_t block = 0; block < model_shape_.attention_blocks; ++block)
    for (ParameterRole role : {ParameterRole::ATTENTION_QUERY_WEIGHT,
                               ParameterRole::ATTENTION_KEY_WEIGHT,
                               ParameterRole::ATTENTION_VALUE_WEIGHT,
                               ParameterRole::ATTENTION_OUTPUT_WEIGHT,
                               ParameterRole::UPDATE_HIDDEN_WEIGHT,
                               ParameterRole::UPDATE_HIDDEN_BIAS,
                               ParameterRole::UPDATE_OUTPUT_WEIGHT,
                               ParameterRole::UPDATE_OUTPUT_BIAS})
      if (!hasParameter(role, block))
        throw std::invalid_argument(std::string("PsiFormer parameter export is missing required role ") +
                                    parameterRoleName(role));

  const bool has_same_spin_pair = model_shape_.spin_up_electrons >= 2 ||
      model_shape_.spin_down_electrons >= 2;
  if (has_same_spin_pair && !hasParameter(ParameterRole::CUSP_SAME_ALPHA))
    throw std::invalid_argument(
        "PsiFormer parameter export is missing same-spin cusp alpha for a populated same-spin pair");

  if (environment_.periodic_profile)
    for (std::size_t block = 0; block < model_shape_.attention_blocks; ++block)
      for (ParameterRole role : {ParameterRole::PAIR_ATTENTION_WEIGHT, ParameterRole::PAIR_ATTENTION_BIAS})
        if (!hasParameter(role, block))
          throw std::invalid_argument("Periodic profile is missing a pair-attention tensor");

  if (paired_head)
    for (ParameterRole role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
      if (!hasParameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG))
        throw std::invalid_argument(std::string("PsiFormer paired-real export is missing imaginary head ") +
                                    parameterRoleName(role));

  const std::size_t expected_tensor_count = 8 + 8 * model_shape_.attention_blocks +
      (hasParameter(ParameterRole::CUSP_SAME_ALPHA) ? 1 : 0) + (paired_head ? 2 : 0) +
      (environment_.periodic_profile ? 2 * model_shape_.attention_blocks : 0);
  if (parameter_tensors_.size() != expected_tensor_count)
    throw std::invalid_argument("PsiFormer parameter export does not contain the complete architecture layout");
}

// Report current support while reserving stable axes for later kernel implementations.
ModelCapabilities PsiFormerExecutionPlan::capabilities()
{
  return {/*open_boundary=*/true,
          /*periodic_boundary=*/true,
          /*real_scalars=*/true,
          /*complex_scalars=*/false,
          /*fixed_nuclei=*/true,
          /*moving_nuclei=*/false};
}

// Select the mathematically correct reverse convention for the configured scalar domain.
AdjointConvention PsiFormerExecutionPlan::adjointConvention() const
{
  return environment_.compute_scalar_domain == ScalarDomain::REAL ? AdjointConvention::TRANSPOSE
                                                                   : AdjointConvention::HERMITIAN;
}

// Resolve typed parameter metadata without performing string lookup in an evaluation.
const ParameterTensorDescriptor& PsiFormerExecutionPlan::parameter(
    ParameterRole role, std::size_t block, HeadTensorComponent component) const
{
  const ParameterTensorDescriptor* descriptor = optionalParameter(role, block, component);
  if (!descriptor)
    throw std::out_of_range(std::string("PsiFormer parameter role is absent: ") + parameterRoleName(role));
  return *descriptor;
}

// Report whether a typed role is represented without forcing optional roles to exist.
bool PsiFormerExecutionPlan::hasParameter(ParameterRole role, std::size_t block, HeadTensorComponent component) const
{
  return optionalParameter(role, block, component) != nullptr;
}

// Resolve a typed role while preserving the absence of architecture-dependent leaves.
const ParameterTensorDescriptor* PsiFormerExecutionPlan::optionalParameter(ParameterRole role,
    std::size_t block, HeadTensorComponent component) const
{
  const std::size_t role_index = static_cast<std::size_t>(role);
  if (role_index >= PARAMETER_ROLE_COUNT || (block != NO_ATTENTION_BLOCK && block >= model_shape_.attention_blocks))
    throw std::out_of_range("PsiFormer parameter role lookup is out of range");
  if (component != HeadTensorComponent::REAL && component != HeadTensorComponent::IMAG)
    throw std::out_of_range("PsiFormer head tensor component is unknown");
  if (component == HeadTensorComponent::IMAG)
  {
    if ((role != ParameterRole::BACKFLOW_UP_WEIGHT && role != ParameterRole::BACKFLOW_DOWN_WEIGHT) ||
        block != NO_ATTENTION_BLOCK)
      throw std::invalid_argument("Imaginary PsiFormer components are restricted to global backflow heads");
    if (imaginary_head_lookup_.empty())
      return nullptr;
    const std::size_t index = imaginary_head_lookup_[role == ParameterRole::BACKFLOW_UP_WEIGHT ? 0 : 1];
    return index == NO_ATTENTION_BLOCK ? nullptr : &parameter_tensors_[index];
  }
  const std::size_t lookup_stride = model_shape_.attention_blocks + 1;
  const std::size_t block_slot = block == NO_ATTENTION_BLOCK ? 0 : block + 1;
  const std::size_t descriptor_index = descriptor_lookup_[role_index * lookup_stride + block_slot];
  if (descriptor_index == NO_ATTENTION_BLOCK)
    return nullptr;
  return &parameter_tensors_[descriptor_index];
}

// Provide readable role names for diagnostics, profiling labels, and tests.
const char* parameterRoleName(ParameterRole role)
{
  switch (role)
  {
  case ParameterRole::ELECTRON_EMBEDDING_WEIGHT:
    return "electron_embedding_weight";
  case ParameterRole::ATTENTION_QUERY_WEIGHT:
    return "attention_query_weight";
  case ParameterRole::ATTENTION_KEY_WEIGHT:
    return "attention_key_weight";
  case ParameterRole::ATTENTION_VALUE_WEIGHT:
    return "attention_value_weight";
  case ParameterRole::ATTENTION_OUTPUT_WEIGHT:
    return "attention_output_weight";
  case ParameterRole::UPDATE_HIDDEN_WEIGHT:
    return "update_hidden_weight";
  case ParameterRole::UPDATE_HIDDEN_BIAS:
    return "update_hidden_bias";
  case ParameterRole::UPDATE_OUTPUT_WEIGHT:
    return "update_output_weight";
  case ParameterRole::UPDATE_OUTPUT_BIAS:
    return "update_output_bias";
  case ParameterRole::BACKFLOW_UP_WEIGHT:
    return "backflow_up_weight";
  case ParameterRole::BACKFLOW_DOWN_WEIGHT:
    return "backflow_down_weight";
  case ParameterRole::ENVELOPE_PI_UP:
    return "envelope_pi_up";
  case ParameterRole::ENVELOPE_PI_DOWN:
    return "envelope_pi_down";
  case ParameterRole::ENVELOPE_ZETA_UP:
    return "envelope_zeta_up";
  case ParameterRole::ENVELOPE_ZETA_DOWN:
    return "envelope_zeta_down";
  case ParameterRole::CUSP_SAME_ALPHA:
    return "cusp_same_alpha";
  case ParameterRole::CUSP_OPPOSITE_ALPHA:
    return "cusp_opposite_alpha";
  case ParameterRole::PAIR_ATTENTION_WEIGHT:
    return "pair_attention_weight";
  case ParameterRole::PAIR_ATTENTION_BIAS:
    return "pair_attention_bias";
  case ParameterRole::COUNT:
    break;
  }
  return "unknown";
}

} // namespace qmcplusplus::psiformer
