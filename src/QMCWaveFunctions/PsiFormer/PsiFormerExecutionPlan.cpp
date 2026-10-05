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

/// Translate one string-keyed layout record into an architecture-level role.
std::pair<ParameterRole, std::size_t> classifyParameter(const ParameterLayoutInput& layout)
{
  if (endsWith(layout.module, "/electron_embedding/linear") && layout.name == "w")
    return {ParameterRole::ELECTRON_EMBEDDING_WEIGHT, NO_ATTENTION_BLOCK};
  if (endsWith(layout.module, "/Backflow/~/mlp/linear_0") && layout.name == "w")
    return {ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK};
  if (endsWith(layout.module, "/Backflow_1/~/mlp/linear_0") && layout.name == "w")
    return {ParameterRole::BACKFLOW_DOWN_WEIGHT, NO_ATTENTION_BLOCK};
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
std::vector<std::size_t> expectedShape(ParameterRole role, const ModelShape& model)
{
  const std::size_t electrons = model.electrons();
  switch (role)
  {
  case ParameterRole::ELECTRON_EMBEDDING_WEIGHT:
    return {4 * model.nuclei + 1, model.feature_dimension};
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

// Validate model dimensions and translate each canonical parameter interval once.
PsiFormerExecutionPlan::PsiFormerExecutionPlan(ModelShape model_shape,
                                               std::vector<ParameterLayoutInput> parameter_layouts,
                                               ExecutionEnvironment environment)
    : model_shape_(std::move(model_shape)), environment_(environment)
{
  const ModelCapabilities support = capabilities();
  if ((environment_.boundary == BoundaryCondition::OPEN && !support.open_boundary) ||
      (environment_.boundary == BoundaryCondition::PERIODIC && !support.periodic_boundary))
    throw std::invalid_argument("Requested PsiFormer boundary condition is not implemented");
  const auto require_supported_scalar_domain = [&support](ScalarDomain domain, const char* description) {
    if ((domain == ScalarDomain::REAL && !support.real_scalars) ||
        (domain == ScalarDomain::COMPLEX && !support.complex_scalars))
      throw std::invalid_argument(std::string("Requested PsiFormer ") + description + " scalar domain is not implemented");
  };
  require_supported_scalar_domain(environment_.parameter_scalar_domain, "parameter");
  require_supported_scalar_domain(environment_.compute_scalar_domain, "compute");
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
  std::size_t expected_begin = 0;
  parameter_tensors_.reserve(parameter_layouts.size());
  for (ParameterLayoutInput& layout : parameter_layouts)
  {
    if (layout.begin != expected_begin || layout.end < layout.begin)
      throw std::invalid_argument("PsiFormer parameter intervals are not contiguous in canonical order");
    if (shapeProduct(layout.shape) != layout.end - layout.begin)
      throw std::invalid_argument("PsiFormer parameter tensor shape does not match its flat interval");

    const auto [role, block] = classifyParameter(layout);
    if (block != NO_ATTENTION_BLOCK && block >= model_shape_.attention_blocks)
      throw std::invalid_argument("PsiFormer parameter references an out-of-range attention block");
    const std::vector<std::size_t> expected_shape = expectedShape(role, model_shape_);
    if (layout.shape != expected_shape)
      throwShapeMismatch(layout, role, expected_shape);
    const std::size_t block_slot = block == NO_ATTENTION_BLOCK ? 0 : block + 1;
    const std::size_t lookup_slot = static_cast<std::size_t>(role) * lookup_stride + block_slot;
    if (descriptor_lookup_[lookup_slot] != NO_ATTENTION_BLOCK)
      throw std::invalid_argument(std::string("Duplicate PsiFormer parameter role ") + parameterRoleName(role));

    descriptor_lookup_[lookup_slot] = parameter_tensors_.size();
    parameter_tensors_.push_back(
        {role, block, std::move(layout.module), std::move(layout.name), std::move(layout.shape), layout.begin, layout.end});
    expected_begin = layout.end;
  }
  parameter_count_ = expected_begin;

  const std::size_t expected_tensor_count = 9 + 8 * model_shape_.attention_blocks;
  if (parameter_tensors_.size() != expected_tensor_count)
    throw std::invalid_argument("PsiFormer parameter export does not contain the complete architecture layout");
}

// Report current support while reserving stable axes for later kernel implementations.
ModelCapabilities PsiFormerExecutionPlan::capabilities()
{
  return {/*open_boundary=*/true,
          /*periodic_boundary=*/false,
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
const ParameterTensorDescriptor& PsiFormerExecutionPlan::parameter(ParameterRole role, std::size_t block) const
{
  const std::size_t role_index = static_cast<std::size_t>(role);
  if (role_index >= PARAMETER_ROLE_COUNT || (block != NO_ATTENTION_BLOCK && block >= model_shape_.attention_blocks))
    throw std::out_of_range("PsiFormer parameter role lookup is out of range");
  const std::size_t lookup_stride = model_shape_.attention_blocks + 1;
  const std::size_t block_slot = block == NO_ATTENTION_BLOCK ? 0 : block + 1;
  const std::size_t descriptor_index = descriptor_lookup_[role_index * lookup_stride + block_slot];
  if (descriptor_index == NO_ATTENTION_BLOCK)
    throw std::out_of_range(std::string("PsiFormer parameter role is absent: ") + parameterRoleName(role));
  return parameter_tensors_[descriptor_index];
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
  case ParameterRole::COUNT:
    break;
  }
  return "unknown";
}

} // namespace qmcplusplus::psiformer
