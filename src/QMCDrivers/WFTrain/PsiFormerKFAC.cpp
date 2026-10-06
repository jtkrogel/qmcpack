//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerKFAC.cpp
 * @brief PsiFormer parameter-role classification and transient tape observation.
 */

#include "QMCDrivers/WFTrain/PsiFormerKFAC.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

using psiformer::NO_ATTENTION_BLOCK;
using psiformer::ParameterRole;
using psiformer::ParameterTensorDescriptor;

/// Return the schema block matching one exact execution-plan tensor interval.
std::size_t schemaBlock(const StructuredParameterSchema& schema,
                        const ParameterTensorDescriptor& tensor)
{
  for (std::size_t block = 0; block < schema.blocks().size(); ++block)
  {
    const ParameterBlockDescriptor& candidate = schema.blocks()[block];
    if (candidate.offset == tensor.begin && candidate.count == tensor.size() &&
        candidate.shape == tensor.shape &&
        candidate.id == tensor.module + "/" + tensor.name)
      return block;
  }
  throw std::invalid_argument("PsiFormer KFAC schema does not match the execution plan");
}

/// Return the stable model-facing identifier for one observed affine boundary.
std::string affineId(ParameterRole role, std::size_t attention_block)
{
  std::string id = psiformer::parameterRoleName(role);
  if (attention_block != NO_ATTENTION_BLOCK)
    id += "/block_" + std::to_string(attention_block);
  return id;
}

/// Append one real trainable affine block, optionally pairing a bias tensor.
void appendAffine(const psiformer::PsiFormerExecutionPlan& plan,
                  const StructuredParameterSchema& schema,
                  ParameterRole weight_role,
                  std::size_t attention_block,
                  std::size_t input_width,
                  std::size_t output_width,
                  std::vector<KFACAffineBlockDescriptor>& affine_blocks,
                  ParameterRole bias_role = ParameterRole::COUNT)
{
  const ParameterTensorDescriptor& weight = plan.parameter(weight_role, attention_block);
  const std::size_t weight_block = schemaBlock(schema, weight);
  const bool weight_trainable = schema.blocks()[weight_block].trainable;
  std::size_t bias_block = NO_KFAC_BIAS_BLOCK;
  if (bias_role != ParameterRole::COUNT)
  {
    bias_block = schemaBlock(schema, plan.parameter(bias_role, attention_block));
    if (schema.blocks()[bias_block].trainable != weight_trainable)
      throw std::invalid_argument(
          "PsiFormer KFAC requires an affine weight and bias to share trainability");
  }
  if (weight_trainable)
    affine_blocks.push_back({affineId(weight_role, attention_block), weight_block,
                             bias_block, input_width, output_width});
}

} // namespace

KFACBlockRegistry makePsiFormerKFACRegistry(
    const psiformer::PsiFormerExecutionPlan& plan,
    const StructuredParameterSchema& schema)
{
  if (schema.parameterCount() != plan.parameterCount())
    throw std::invalid_argument("PsiFormer KFAC schema has the wrong parameter count");
  const psiformer::ModelShape& shape = plan.modelShape();
  const std::size_t electrons = shape.electrons();
  const std::size_t width = shape.feature_dimension;

  std::vector<KFACAffineBlockDescriptor> affine_blocks;
  affine_blocks.reserve(3 + 6 * shape.attention_blocks);
  appendAffine(plan, schema, ParameterRole::ELECTRON_EMBEDDING_WEIGHT,
               NO_ATTENTION_BLOCK, 4 * shape.nuclei + 1, width, affine_blocks);
  for (std::size_t block = 0; block < shape.attention_blocks; ++block)
  {
    appendAffine(plan, schema, ParameterRole::ATTENTION_QUERY_WEIGHT, block,
                 width, width, affine_blocks);
    appendAffine(plan, schema, ParameterRole::ATTENTION_KEY_WEIGHT, block,
                 width, width, affine_blocks);
    appendAffine(plan, schema, ParameterRole::ATTENTION_VALUE_WEIGHT, block,
                 width, width, affine_blocks);
    appendAffine(plan, schema, ParameterRole::ATTENTION_OUTPUT_WEIGHT, block,
                 width, width, affine_blocks);
    appendAffine(plan, schema, ParameterRole::UPDATE_HIDDEN_WEIGHT, block,
                 width, width, affine_blocks, ParameterRole::UPDATE_HIDDEN_BIAS);
    appendAffine(plan, schema, ParameterRole::UPDATE_OUTPUT_WEIGHT, block,
                 width, width, affine_blocks, ParameterRole::UPDATE_OUTPUT_BIAS);
  }
  appendAffine(plan, schema, ParameterRole::BACKFLOW_UP_WEIGHT,
               NO_ATTENTION_BLOCK, width, shape.determinants * electrons,
               affine_blocks);
  appendAffine(plan, schema, ParameterRole::BACKFLOW_DOWN_WEIGHT,
               NO_ATTENTION_BLOCK, width, shape.determinants * electrons,
               affine_blocks);

  std::vector<std::size_t> fallback_blocks;
  for (ParameterRole role : {ParameterRole::ENVELOPE_PI_UP,
                             ParameterRole::ENVELOPE_PI_DOWN,
                             ParameterRole::ENVELOPE_ZETA_UP,
                             ParameterRole::ENVELOPE_ZETA_DOWN,
                             ParameterRole::CUSP_SAME_ALPHA,
                             ParameterRole::CUSP_OPPOSITE_ALPHA})
    if (plan.hasParameter(role))
    {
      const std::size_t block = schemaBlock(schema, plan.parameter(role));
      if (schema.blocks()[block].trainable)
        fallback_blocks.push_back(block);
    }

  return {schema, std::move(affine_blocks), std::move(fallback_blocks)};
}

PsiFormerKFACObservationSink::PsiFormerKFACObservationSink(
    const psiformer::PsiFormerExecutionPlan& plan,
    KFACFactorAccumulator& accumulator)
    : accumulator_(accumulator)
{
  const KFACBlockRegistry expected =
      makePsiFormerKFACRegistry(plan, accumulator.registry().parameterSchema());
  if (expected.fingerprint() != accumulator.registry().fingerprint())
    throw std::invalid_argument("PsiFormer KFAC observation adapter received a foreign registry");

  const auto& affine_blocks = accumulator.registry().affineBlocks();
  bindings_.reserve(affine_blocks.size());
  auto bind = [&](ParameterRole role, std::size_t attention_block) {
    const std::string id = affineId(role, attention_block);
    const auto found = std::find_if(
        affine_blocks.begin(), affine_blocks.end(),
        [&](const KFACAffineBlockDescriptor& block) { return block.id == id; });
    if (found != affine_blocks.end())
      bindings_.push_back({role, attention_block,
                           static_cast<std::size_t>(found - affine_blocks.begin())});
  };

  bind(ParameterRole::ELECTRON_EMBEDDING_WEIGHT, NO_ATTENTION_BLOCK);
  for (std::size_t block = 0; block < plan.modelShape().attention_blocks; ++block)
    for (ParameterRole role : {ParameterRole::ATTENTION_QUERY_WEIGHT,
                               ParameterRole::ATTENTION_KEY_WEIGHT,
                               ParameterRole::ATTENTION_VALUE_WEIGHT,
                               ParameterRole::ATTENTION_OUTPUT_WEIGHT,
                               ParameterRole::UPDATE_HIDDEN_WEIGHT,
                               ParameterRole::UPDATE_OUTPUT_WEIGHT})
      bind(role, block);
  bind(ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK);
  bind(ParameterRole::BACKFLOW_DOWN_WEIGHT, NO_ATTENTION_BLOCK);
  if (bindings_.size() != affine_blocks.size())
    throw std::logic_error("PsiFormer KFAC registry contains an unobservable affine block");
}

void PsiFormerKFACObservationSink::observe(
    const pf::DirectAffineObservation& observation)
{
  const auto found = std::find_if(
      bindings_.begin(), bindings_.end(), [&](const Binding& binding) {
        return binding.role == observation.role &&
            binding.attention_block == observation.attention_block;
      });
  if (found == bindings_.end())
    throw std::invalid_argument("PsiFormer KFAC received an unregistered affine observation");
  const KFACAffineBlockDescriptor& descriptor =
      accumulator_.registry().affineBlocks()[found->affine_block];
  if (observation.input_width != descriptor.input_width ||
      observation.output_width != descriptor.output_width)
    throw std::invalid_argument("PsiFormer KFAC affine observation dimensions do not match the registry");
  if (observation.row_count >
          std::numeric_limits<std::size_t>::max() / observation.input_width ||
      observation.row_count >
          std::numeric_limits<std::size_t>::max() / observation.output_width)
    throw std::overflow_error("PsiFormer KFAC affine observation extent overflow");
  accumulator_.addObservation(
      {found->affine_block,
       {observation.activations, observation.row_count * observation.input_width},
       {observation.sensitivities, observation.row_count * observation.output_width},
       observation.row_count});
}

} // namespace qmcplusplus::wftrain
