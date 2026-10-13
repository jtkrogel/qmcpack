//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include "ProposalSelection.h"
#include "TrainingNumerics.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace qmcplusplus::wftrain
{

SelectedUpdateRule::SelectedUpdateRule(TrainingUpdateRule& optimizer,
                                     ProposalSelectionControl control, Selector selector)
    : optimizer_(optimizer), control_(std::move(control)), selector_(std::move(selector))
{
  if (!isFiniteTrainingReal(control_.update_scale) || control_.update_scale < 0.0 ||
      !isFiniteTrainingReal(control_.weight_decay) || control_.weight_decay < 0.0 ||
      control_.weight_decay >= 1.0 || !isFiniteTrainingReal(control_.maximum_update_norm) ||
      control_.maximum_update_norm < 0.0)
    throw std::invalid_argument("Invalid proposal scaling/shrinkage controls");
}

StructuredParameterSnapshot SelectedUpdateRule::propose(
    const StructuredParameterSchema& schema, const StructuredParameterSnapshot& parameters,
    ParameterGradientView objective)
{
  auto candidate = optimizer_.propose(schema, parameters, objective);
  if (candidate.schema_fingerprint != schema.fingerprint() ||
      candidate.version != parameters.version || candidate.values.size() != parameters.values.size())
    throw std::invalid_argument("Proposal selection received an incompatible optimizer candidate");

  // Match tensor identifiers explicitly; unknown selectors are configuration errors.
  for (const auto& id : control_.decay_blocks)
    if (std::none_of(schema.blocks().begin(), schema.blocks().end(),
                     [&](const auto& block) { return block.id == id; }))
      throw std::invalid_argument("Unknown weight-decay tensor: " + id);
  double norm = 0.0;
  for (const auto& block : schema.blocks())
  {
    const bool decay = std::find(control_.decay_blocks.begin(), control_.decay_blocks.end(),
                                 block.id) != control_.decay_blocks.end();
    for (std::size_t i = block.offset; i < block.offset + block.count; ++i)
    {
      if (!block.trainable && candidate.values[i] != parameters.values[i])
        throw std::invalid_argument("Optimizer modified a frozen tensor before selection");
      const double delta = block.trainable
          ? control_.update_scale * (candidate.values[i] - parameters.values[i]) -
              (decay ? control_.weight_decay * parameters.values[i] : 0.0)
          : 0.0;
      if (!isFiniteTrainingReal(delta))
        throw std::invalid_argument("Nonfinite combined proposal displacement");
      if (control_.update_scale != 1.0 || control_.weight_decay != 0.0)
        candidate.values[i] = parameters.values[i] + delta;
      norm = std::hypot(norm, delta);
    }
  }
  // Bound the full combined direction before any candidate evaluation.
  if (control_.maximum_update_norm > 0.0 && norm > control_.maximum_update_norm)
  {
    const double scale = control_.maximum_update_norm / norm;
    for (std::size_t i = 0; i < candidate.values.size(); ++i)
      candidate.values[i] = parameters.values[i] + scale * (candidate.values[i] - parameters.values[i]);
    norm = control_.maximum_update_norm;
  }
  alpha_ = selector_ ? selector_(parameters, candidate) : 1.0;
  if (!isFiniteTrainingReal(alpha_) || alpha_ < 0.0 || alpha_ > 2.0)
    throw std::invalid_argument("Candidate selector returned a scale outside [0,2]");
  // An expanded step must respect the same displacement bound.
  if (control_.maximum_update_norm > 0.0 && norm > 0.0)
    alpha_ = std::min(alpha_, control_.maximum_update_norm / norm);
  if (alpha_ != 1.0)
    for (std::size_t i = 0; i < candidate.values.size(); ++i)
      candidate.values[i] = parameters.values[i] + alpha_ * (candidate.values[i] - parameters.values[i]);
  displacement_norm_ = alpha_ * norm;
  return candidate;
}

void SelectedUpdateRule::proposalAccepted(const StructuredParameterSchema& schema,
                                         const StructuredParameterSnapshot& parameters,
                                         ParameterGradientView objective) noexcept
{
  if (candidateAccepted())
    optimizer_.proposalAccepted(schema, parameters, objective);
  else
    optimizer_.proposalRejected();
}

void SelectedUpdateRule::proposalRejected() noexcept { optimizer_.proposalRejected(); }

} // namespace qmcplusplus::wftrain
