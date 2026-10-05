//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerScoreExecutor.h
 * @brief Direct allocation-free parameter-score evaluator for real molecular PsiFormer.
 *
 * The evaluator records a compact forward tape and explicitly applies the reverse
 * chain rule for the embedding, attention blocks, orbital/envelope construction,
 * determinant sum, and cusp.  The returned vector is d log|Psi| / d parameter in
 * the canonical imported-HDF5 order used by QMCPACK's optimizer registration.
 *
 * Immutable tensor offsets and execution semantics come from PsiFormerExecutionPlan;
 * all tape, adjoint, geometry, and output storage is clone-local.  The implemented
 * kernels are deliberately real/open-boundary/fixed-nucleus only.  Plan scalar-domain,
 * adjoint-convention, and boundary metadata remain explicit so later complex and
 * periodic specializations can replace algebra and geometry policies without changing
 * parameter identity or the public score contract.
 */

//////////////////////////////////////////////////////////////////////////////////////
// INCLUSION RESTRICTION
// PsiFormerNative.h defines non-inline functions and is included by exactly one QMCPACK
// translation unit. Include this helper only after PsiFormerNative.h in that same
// translation unit (or in a standalone one-translation-unit test).
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_PSIFORMER_NATIVE_H
#error "Include PsiFormerNative.h before PsiFormerScoreExecutor.h"
#endif

#ifndef QMCPLUSPLUS_PSIFORMER_SCORE_EXECUTOR_H
#define QMCPLUSPLUS_PSIFORMER_SCORE_EXECUTOR_H

#include "PsiFormerDirectKernels.h"
#include "PsiFormerDeterminant.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerGeometry.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pf
{

/** Provide a non-owning immutable view of the canonical parameter score. */
struct DirectParameterScoreView
{
  const double* data = nullptr;
  std::size_t size    = 0;

  /// Access one canonical score component without bounds checking.
  const double& operator[](std::size_t index) const { return data[index]; }

  /// Return an iterator to the first score component.
  const double* begin() const { return data; }

  /// Return an iterator one past the last score component.
  const double* end() const { return size == 0 ? data : data + size; }
};

/// Identify the destination contract used by a direct score consumer.
enum class DirectScoreOutputPolicy
{
  FULL,
  SELECTED,
  WEIGHTED_ACCUMULATE,
  PRODUCT
};

/** Apply full, selected, weighted, and product score destinations without owning copies.
 *
 * Destination indices are supplied separately from canonical parameter indices so
 * QMCPACK's global active-variable ordering remains an adapter concern.  Accumulation
 * methods deliberately use += semantics.  The arithmetic is real today; templated
 * destinations allow the real score to feed a complex QMCPACK build without claiming
 * a genuinely complex network reverse pass.
 */
class DirectScoreOutput
{
public:
  /// Overwrite a canonical full-score destination.
  static void writeFull(DirectParameterScoreView score, double* destination, std::size_t destination_size)
  {
    requireDestination(destination, destination_size, score.size, "full");
    std::copy(score.begin(), score.end(), destination);
  }

  /// Overwrite selected entries in caller-specified local ordering.
  static void writeSelected(DirectParameterScoreView score,
                            const std::size_t* canonical_indices,
                            std::size_t selected_count,
                            double* destination)
  {
    requireSelection(score, canonical_indices, selected_count);
    requireDestination(destination, selected_count, selected_count, "selected");
    for (std::size_t selected = 0; selected < selected_count; ++selected)
      destination[selected] = score[canonical_indices[selected]];
  }

  /// Add a weighted selected score directly to arbitrary destination indices.
  template<class DestinationScalar>
  static void accumulateSelected(DirectParameterScoreView score,
                                 const std::size_t* canonical_indices,
                                 const std::size_t* destination_indices,
                                 std::size_t selected_count,
                                 double weight,
                                 DestinationScalar* destination,
                                 std::size_t destination_size)
  {
    requireSelection(score, canonical_indices, selected_count);
    if (selected_count != 0 && (!destination_indices || !destination))
      throw std::invalid_argument("PsiFormer weighted score destination is null");
    for (std::size_t selected = 0; selected < selected_count; ++selected)
    {
      const std::size_t output = destination_indices[selected];
      if (output >= destination_size)
        throw std::out_of_range("PsiFormer weighted score destination index is out of range");
      destination[output] += DestinationScalar(weight * score[canonical_indices[selected]]);
    }
  }

  /// Add a weighted virtual-minus-reference score without materializing the difference.
  template<class DestinationScalar>
  static void accumulateSelectedDifference(DirectParameterScoreView virtual_score,
                                           const double* selected_reference,
                                           const std::size_t* canonical_indices,
                                           const std::size_t* destination_indices,
                                           std::size_t selected_count,
                                           double weight,
                                           DestinationScalar* destination,
                                           std::size_t destination_size)
  {
    requireSelection(virtual_score, canonical_indices, selected_count);
    if (selected_count != 0 && (!selected_reference || !destination_indices || !destination))
      throw std::invalid_argument("PsiFormer weighted score-difference destination is null");
    for (std::size_t selected = 0; selected < selected_count; ++selected)
    {
      const std::size_t output = destination_indices[selected];
      if (output >= destination_size)
        throw std::out_of_range("PsiFormer weighted score-difference index is out of range");
      const double difference = virtual_score[canonical_indices[selected]] - selected_reference[selected];
      destination[output] += DestinationScalar(weight * difference);
    }
  }

  /// Contract selected score entries with one caller vector for matrix-free products.
  static double selectedProduct(DirectParameterScoreView score,
                                const std::size_t* canonical_indices,
                                const double* vector,
                                std::size_t selected_count,
                                double center = 0.0)
  {
    requireSelection(score, canonical_indices, selected_count);
    if (selected_count != 0 && !vector)
      throw std::invalid_argument("PsiFormer score-product vector is null");
    double product = 0.0;
    for (std::size_t selected = 0; selected < selected_count; ++selected)
      product += (score[canonical_indices[selected]] - center) * vector[selected];
    return product;
  }

private:
  /// Validate a non-owning output interval before writing it.
  static void requireDestination(const double* destination,
                                 std::size_t destination_size,
                                 std::size_t required_size,
                                 const char* description)
  {
    if (destination_size < required_size || (required_size != 0 && !destination))
      throw std::invalid_argument(std::string("PsiFormer ") + description +
                                  " score destination has the wrong size");
  }

  /// Validate all selected canonical indices before modifying a destination.
  static void requireSelection(DirectParameterScoreView score,
                               const std::size_t* canonical_indices,
                               std::size_t selected_count)
  {
    if (selected_count != 0 && !canonical_indices)
      throw std::invalid_argument("PsiFormer selected score indices are null");
    for (std::size_t selected = 0; selected < selected_count; ++selected)
      if (canonical_indices[selected] >= score.size)
        throw std::out_of_range("PsiFormer selected score index is out of range");
  }
};

/** Return the wavefunction value and its canonical log-parameter score. */
struct DirectScoreResult
{
  double sign   = 1.0;
  double logabs = 0.0;
  double value  = 0.0;
  DirectParameterScoreView parameter_score;
  std::size_t parameter_version = 0;
};

/**
 * Own the forward tape, reverse adjoints, geometry cache, and score for one clone.
 *
 * Every vector is sized in the constructor and never resized during evaluation.
 * This makes a warmed-up call allocation-free and prevents mutable scratch from
 * being shared between QMCPACK walker/crowd clones.
 */
class DirectScoreWorkspace
{
public:
  /// Allocate the exact tape and reverse storage required by one execution plan.
  DirectScoreWorkspace(const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan,
                       GeometryPositionView nuclei,
                       GeometryBoundary boundary = {})
      : electrons_(plan.modelShape().electrons()),
        nuclei_(plan.modelShape().nuclei),
        determinants_(plan.modelShape().determinants),
        width_(plan.modelShape().feature_dimension),
        heads_(plan.modelShape().attention_heads),
        blocks_(plan.modelShape().attention_blocks),
        head_width_(width_ / heads_),
        input_width_(4 * nuclei_ + 1),
        feature_elements_(electrons_ * width_),
        attention_elements_(heads_ * electrons_ * electrons_),
        orbital_elements_(determinants_ * electrons_ * electrons_),
        electron_positions_(3 * electrons_),
        geometry_(electrons_, nuclei, boundary),
        raw_features_(electrons_ * input_width_),
        features_((blocks_ + 1) * feature_elements_),
        queries_(blocks_ * feature_elements_),
        keys_(blocks_ * feature_elements_),
        values_(blocks_ * feature_elements_),
        attention_weights_(blocks_ * attention_elements_),
        contexts_(blocks_ * feature_elements_),
        residuals_(blocks_ * feature_elements_),
        hidden_(blocks_ * feature_elements_),
        updates_(blocks_ * feature_elements_),
        orbital_matrices_(orbital_elements_),
        determinant_workspace_(determinants_, electrons_),
        matrix_adjoints_(orbital_elements_),
        parameter_score_(plan.parameterCount()),
        feature_adjoint_a_(feature_elements_),
        feature_adjoint_b_(feature_elements_),
        query_adjoint_(feature_elements_),
        key_adjoint_(feature_elements_),
        value_adjoint_(feature_elements_),
        attention_adjoint_(attention_elements_),
        context_adjoint_(feature_elements_),
        residual_adjoint_(feature_elements_),
        hidden_adjoint_(feature_elements_),
        update_adjoint_(feature_elements_)
  {
    if (heads_ == 0 || width_ % heads_ != 0)
      throw std::invalid_argument("PsiFormer score workspace received inconsistent attention dimensions");
  }

  DirectScoreWorkspace(const DirectScoreWorkspace&) = delete;
  DirectScoreWorkspace& operator=(const DirectScoreWorkspace&) = delete;
  DirectScoreWorkspace(DirectScoreWorkspace&&) = default;
  DirectScoreWorkspace& operator=(DirectScoreWorkspace&&) = default;

  /// Copy a complete interleaved electron configuration into fixed workspace storage.
  void setPositions(GeometryPositionView positions)
  {
    if (positions.size() != electrons_)
      throw std::invalid_argument("PsiFormer score workspace received the wrong electron count");
    for (std::size_t electron = 0; electron < electrons_; ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        electron_positions_[3 * electron + dimension] = positions(electron, dimension);
  }

  /// Overwrite one Cartesian coordinate without changing any buffer capacity.
  void setPosition(std::size_t electron, std::size_t dimension, double coordinate)
  {
    if (electron >= electrons_ || dimension >= 3)
      throw std::out_of_range("PsiFormer score workspace coordinate index is out of range");
    electron_positions_[3 * electron + dimension] = coordinate;
  }

  /// Return the parameter version used by the most recently completed call.
  std::size_t observedParameterVersion() const noexcept { return observed_parameter_version_; }

  /// Return the score buffer address for allocation-stability tests.
  const double* scoreData() const noexcept { return parameter_score_.data(); }

  /// Return the fixed number of canonical score components.
  std::size_t scoreSize() const noexcept { return parameter_score_.size(); }

  /// Return bytes reserved by explicit tape/adjoint vectors (excluding geometry metadata).
  std::size_t vectorStorageBytes() const noexcept
  {
    const std::size_t elements = electron_positions_.capacity() + raw_features_.capacity() +
        features_.capacity() + queries_.capacity() + keys_.capacity() + values_.capacity() +
        attention_weights_.capacity() + contexts_.capacity() + residuals_.capacity() + hidden_.capacity() +
        updates_.capacity() + orbital_matrices_.capacity() + matrix_adjoints_.capacity() +
        parameter_score_.capacity() +
        feature_adjoint_a_.capacity() + feature_adjoint_b_.capacity() + query_adjoint_.capacity() +
        key_adjoint_.capacity() + value_adjoint_.capacity() + attention_adjoint_.capacity() +
        context_adjoint_.capacity() + residual_adjoint_.capacity() + hidden_adjoint_.capacity() +
        update_adjoint_.capacity();
    return elements * sizeof(double) + determinant_workspace_.storageBytes();
  }

private:
  friend class DirectScoreExecutor;

  /// Return the feature matrix entering the requested block (or final matrix).
  double* features(std::size_t boundary) { return features_.data() + boundary * feature_elements_; }
  const double* features(std::size_t boundary) const
  {
    return features_.data() + boundary * feature_elements_;
  }

  /// Return one block-local feature-sized forward buffer.
  static double* blockBuffer(std::vector<double>& buffer, std::size_t block, std::size_t block_size)
  {
    return buffer.data() + block * block_size;
  }

  /// Return one immutable block-local feature-sized forward buffer.
  static const double* blockBuffer(const std::vector<double>& buffer,
                                   std::size_t block,
                                   std::size_t block_size)
  {
    return buffer.data() + block * block_size;
  }

  std::size_t electrons_;
  std::size_t nuclei_;
  std::size_t determinants_;
  std::size_t width_;
  std::size_t heads_;
  std::size_t blocks_;
  std::size_t head_width_;
  std::size_t input_width_;
  std::size_t feature_elements_;
  std::size_t attention_elements_;
  std::size_t orbital_elements_;

  std::vector<double> electron_positions_;
  PsiFormerGeometryCache geometry_;
  std::vector<double> raw_features_;
  std::vector<double> features_;
  std::vector<double> queries_;
  std::vector<double> keys_;
  std::vector<double> values_;
  std::vector<double> attention_weights_;
  std::vector<double> contexts_;
  std::vector<double> residuals_;
  std::vector<double> hidden_;
  std::vector<double> updates_;
  std::vector<double> orbital_matrices_;
  qmcplusplus::psiformer::determinant::RealOpenDeterminantWorkspace determinant_workspace_;
  std::vector<double> matrix_adjoints_;
  std::vector<double> parameter_score_;

  std::vector<double> feature_adjoint_a_;
  std::vector<double> feature_adjoint_b_;
  std::vector<double> query_adjoint_;
  std::vector<double> key_adjoint_;
  std::vector<double> value_adjoint_;
  std::vector<double> attention_adjoint_;
  std::vector<double> context_adjoint_;
  std::vector<double> residual_adjoint_;
  std::vector<double> hidden_adjoint_;
  std::vector<double> update_adjoint_;
  std::size_t observed_parameter_version_ = std::numeric_limits<std::size_t>::max();
};

/** Evaluate log|Psi| and its full canonical parameter score by a taped reverse pass. */
class DirectScoreExecutor
{
public:
  /// Reuse an execution plan shared by the owning PsiFormer wavefunction state.
  DirectScoreExecutor(const PsiFormer& model,
                      const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan)
      : parameters_(model.p),
        plan_(plan),
        nuclei_(model.cfg.nuclei.x),
        spin_up_electrons_(model.cfg.nup),
        boundary_(mappedBoundary(plan.environment().boundary))
  {
    if (plan_.parameterCount() != parameters_.size() ||
        plan_.modelShape().electrons() != model.ne ||
        plan_.modelShape().nuclei != model.cfg.nuclei.shape[0] ||
        plan_.modelShape().determinants != model.ndet ||
        plan_.modelShape().feature_dimension != model.dim ||
        plan_.modelShape().attention_heads != model.heads)
      throw std::invalid_argument("PsiFormer score execution plan does not match the imported model");
    if (plan_.environment().boundary != qmcplusplus::psiformer::BoundaryCondition::OPEN ||
        plan_.environment().parameter_scalar_domain != qmcplusplus::psiformer::ScalarDomain::REAL ||
        plan_.environment().compute_scalar_domain != qmcplusplus::psiformer::ScalarDomain::REAL ||
        plan_.environment().amplitude_scalar_domain != qmcplusplus::psiformer::ScalarDomain::REAL ||
        !plan_.environment().fixed_nuclei)
      throw std::invalid_argument("Direct PsiFormer score supports only real, open, fixed-nucleus models");
  }

  /// Construct independently mutable tape and adjoint storage for one clone.
  std::unique_ptr<DirectScoreWorkspace> makeWorkspace() const
  {
    return std::make_unique<DirectScoreWorkspace>(
        plan_, GeometryPositionView::interleaved(nuclei_.data(), plan_.modelShape().nuclei), boundary_);
  }

  /// Return immutable typed execution metadata shared by all workspaces.
  const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan() const noexcept
  {
    return plan_;
  }

  /// Execute one taped forward evaluation and one explicit parameter reverse pass.
  DirectScoreResult evaluate(DirectScoreWorkspace& workspace) const
  {
    validateWorkspaceAndParameters(workspace);
    const double* parameter_values = parameters_.flat_values().data();
    std::fill(workspace.parameter_score_.begin(), workspace.parameter_score_.end(), 0.0);

    workspace.geometry_.update(
        GeometryPositionView::interleaved(workspace.electron_positions_.data(), workspace.electrons_));
    buildEmbedding(parameter_values, workspace);
    for (std::size_t block = 0; block < workspace.blocks_; ++block)
      applyAttentionBlock(parameter_values, block, workspace);
    buildOrbitals(parameter_values, workspace);

    namespace determinant = qmcplusplus::psiformer::determinant;
    const determinant::RealDeterminantResult determinant_result =
        workspace.determinant_workspace_.evaluate(workspace.orbital_matrices_.data());
    if (determinant_result.amplitude.isZero())
      throw std::runtime_error("PsiFormer score executor reached an exact determinant node");
    workspace.determinant_workspace_.fillLogAmplitudeMatrixAdjoints(
        workspace.matrix_adjoints_.data());

    const double cusp   = cuspValueAndReverse(parameter_values, workspace);
    const double logabs = determinant_result.amplitude.log_abs + cusp;
    if (!is_finite_parameter_value(logabs))
      throw std::runtime_error("PsiFormer score executor produced a non-finite log wavefunction");

    reverseOrbitals(parameter_values, workspace);
    for (std::size_t reverse_index = workspace.blocks_; reverse_index > 0; --reverse_index)
      reverseAttentionBlock(parameter_values, reverse_index - 1, workspace);
    reverseEmbedding(workspace);

    workspace.observed_parameter_version_ = parameters_.version();
    const double sign = determinant_result.amplitude.phase;
    return {sign,
            logabs,
            determinant::realValue(determinant_result.amplitude, cusp),
            {workspace.parameter_score_.data(), workspace.parameter_score_.size()},
            workspace.observed_parameter_version_};
  }

  /** Copy selected canonical score entries after a full evaluation without allocation. */
  DirectScoreResult evaluateSelected(DirectScoreWorkspace& workspace,
                                     const std::size_t* indices,
                                     std::size_t selected_count,
                                     double* selected_output) const
  {
    if ((selected_count != 0 && (!indices || !selected_output)))
      throw std::invalid_argument("PsiFormer selected score requires valid index and output storage");
    DirectScoreResult result = evaluate(workspace);
    DirectScoreOutput::writeSelected(result.parameter_score, indices, selected_count, selected_output);
    return result;
  }

private:
  using ParameterRole = qmcplusplus::psiformer::ParameterRole;
  using ParameterTensorDescriptor = qmcplusplus::psiformer::ParameterTensorDescriptor;

  /// Map supported execution-plan boundaries to the geometry policy explicitly.
  static GeometryBoundary mappedBoundary(qmcplusplus::psiformer::BoundaryCondition boundary)
  {
    if (boundary == qmcplusplus::psiformer::BoundaryCondition::OPEN)
      return {GeometryBoundaryKind::OPEN};
    throw std::invalid_argument("Periodic PsiFormer score execution is not implemented");
  }

  /// Return the first scalar in one typed parameter tensor.
  const double* parameter(const double* values,
                          ParameterRole role,
                          std::size_t block = qmcplusplus::psiformer::NO_ATTENTION_BLOCK) const
  {
    return values + plan_.parameter(role, block).begin;
  }

  /// Return writable canonical score storage for one typed parameter tensor.
  double* score(DirectScoreWorkspace& workspace,
                ParameterRole role,
                std::size_t block = qmcplusplus::psiformer::NO_ATTENTION_BLOCK) const
  {
    return workspace.parameter_score_.data() + plan_.parameter(role, block).begin;
  }

  /// Reject a workspace from a different shape or a mutated parameter layout.
  void validateWorkspaceAndParameters(const DirectScoreWorkspace& workspace) const
  {
    const auto& shape = plan_.modelShape();
    if (workspace.electrons_ != shape.electrons() || workspace.nuclei_ != shape.nuclei ||
        workspace.determinants_ != shape.determinants || workspace.width_ != shape.feature_dimension ||
        workspace.heads_ != shape.attention_heads || workspace.blocks_ != shape.attention_blocks ||
        workspace.parameter_score_.size() != plan_.parameterCount())
      throw std::invalid_argument("PsiFormer score workspace belongs to a different execution plan");
    // Parameters exposes optimizer mutations through a versioned value store while its
    // layout is an immutable import property. Avoid rebuilding its string fingerprint on
    // every hot call; plan construction already validated every interval and shape.
    if (parameters_.size() != plan_.parameterCount())
      throw std::logic_error("PsiFormer parameter count changed after score-plan construction");
  }

  /// Form electron-nucleus features and the first learned embedding matrix.
  void buildEmbedding(const double* parameters, DirectScoreWorkspace& workspace) const
  {
    const GeometryPairTable& pairs = workspace.geometry_.electronNucleusPairs();
    const auto& displacements      = pairs.displacements();
    const auto& radial_factors     = pairs.softenedRadialFactors();
    for (std::size_t electron = 0; electron < workspace.electrons_; ++electron)
    {
      double* feature_row = workspace.raw_features_.data() + electron * workspace.input_width_;
      for (std::size_t nucleus = 0; nucleus < workspace.nuclei_; ++nucleus)
      {
        const std::size_t pair = electron * workspace.nuclei_ + nucleus;
        feature_row[4 * nucleus] = radial_factors[pair].log1p_radius;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          feature_row[4 * nucleus + 1 + dimension] =
              displacements[pair][dimension] * radial_factors[pair].log1p_over_radius;
      }
      feature_row[workspace.input_width_ - 1] = electron < spin_up_electrons_ ? 1.0 : -1.0;
    }

    qmcplusplus::psiformer::direct::denseForward(
        workspace.raw_features_.data(), parameter(parameters, ParameterRole::ELECTRON_EMBEDDING_WEIGHT), nullptr,
        workspace.electrons_, workspace.input_width_, workspace.width_, workspace.features(0));
  }

  /// Evaluate one attention/residual block and retain precisely the reverse tape.
  void applyAttentionBlock(const double* parameters,
                           std::size_t block,
                           DirectScoreWorkspace& workspace) const
  {
    double* query = DirectScoreWorkspace::blockBuffer(workspace.queries_, block, workspace.feature_elements_);
    double* key   = DirectScoreWorkspace::blockBuffer(workspace.keys_, block, workspace.feature_elements_);
    double* value = DirectScoreWorkspace::blockBuffer(workspace.values_, block, workspace.feature_elements_);
    const double* input = workspace.features(block);
    qmcplusplus::psiformer::direct::denseForward(
        input, parameter(parameters, ParameterRole::ATTENTION_QUERY_WEIGHT, block), nullptr,
        workspace.electrons_, workspace.width_, workspace.width_, query);
    qmcplusplus::psiformer::direct::denseForward(
        input, parameter(parameters, ParameterRole::ATTENTION_KEY_WEIGHT, block), nullptr,
        workspace.electrons_, workspace.width_, workspace.width_, key);
    qmcplusplus::psiformer::direct::denseForward(
        input, parameter(parameters, ParameterRole::ATTENTION_VALUE_WEIGHT, block), nullptr,
        workspace.electrons_, workspace.width_, workspace.width_, value);

    double* attention =
        DirectScoreWorkspace::blockBuffer(workspace.attention_weights_, block, workspace.attention_elements_);
    double* context = DirectScoreWorkspace::blockBuffer(workspace.contexts_, block, workspace.feature_elements_);
    qmcplusplus::psiformer::direct::attentionWeightsForward(
        query, key, workspace.electrons_, workspace.heads_, workspace.head_width_, attention);
    qmcplusplus::psiformer::direct::attentionContextForward(
        attention, value, workspace.electrons_, workspace.heads_, workspace.head_width_, context);

    double* residual = DirectScoreWorkspace::blockBuffer(workspace.residuals_, block, workspace.feature_elements_);
    qmcplusplus::psiformer::direct::denseForward(
        context, parameter(parameters, ParameterRole::ATTENTION_OUTPUT_WEIGHT, block), nullptr,
        workspace.electrons_, workspace.width_, workspace.width_, residual);
    for (std::size_t element = 0; element < workspace.feature_elements_; ++element)
      residual[element] += input[element];

    double* hidden = DirectScoreWorkspace::blockBuffer(workspace.hidden_, block, workspace.feature_elements_);
    qmcplusplus::psiformer::direct::denseForward(
        residual, parameter(parameters, ParameterRole::UPDATE_HIDDEN_WEIGHT, block),
        parameter(parameters, ParameterRole::UPDATE_HIDDEN_BIAS, block), workspace.electrons_,
        workspace.width_, workspace.width_, hidden);
    for (std::size_t element = 0; element < workspace.feature_elements_; ++element)
      hidden[element] = std::tanh(hidden[element]);

    double* update = DirectScoreWorkspace::blockBuffer(workspace.updates_, block, workspace.feature_elements_);
    qmcplusplus::psiformer::direct::denseForward(
        hidden, parameter(parameters, ParameterRole::UPDATE_OUTPUT_WEIGHT, block),
        parameter(parameters, ParameterRole::UPDATE_OUTPUT_BIAS, block), workspace.electrons_,
        workspace.width_, workspace.width_, update);
    double* output = workspace.features(block + 1);
    for (std::size_t element = 0; element < workspace.feature_elements_; ++element)
    {
      update[element] = std::tanh(update[element]);
      output[element] = residual[element] + update[element];
    }
  }

  /// Construct all determinant orbital matrices from final features and envelopes.
  void buildOrbitals(const double* parameters, DirectScoreWorkspace& workspace) const
  {
    const double* final_features = workspace.features(workspace.blocks_);
    const auto& distances        = workspace.geometry_.electronNucleusPairs().distances();
    for (std::size_t electron = 0; electron < workspace.electrons_; ++electron)
    {
      const bool spin_up = electron < spin_up_electrons_;
      const double* backflow = parameter(parameters, spin_up ? ParameterRole::BACKFLOW_UP_WEIGHT
                                                             : ParameterRole::BACKFLOW_DOWN_WEIGHT);
      const double* pi = parameter(parameters, spin_up ? ParameterRole::ENVELOPE_PI_UP
                                                       : ParameterRole::ENVELOPE_PI_DOWN);
      const double* zeta = parameter(parameters, spin_up ? ParameterRole::ENVELOPE_ZETA_UP
                                                         : ParameterRole::ENVELOPE_ZETA_DOWN);
      const double* feature_row = final_features + electron * workspace.width_;
      for (std::size_t determinant = 0; determinant < workspace.determinants_; ++determinant)
        for (std::size_t orbital = 0; orbital < workspace.electrons_; ++orbital)
        {
          const std::size_t channel = determinant * workspace.electrons_ + orbital;
          double backflow_value = 0.0;
          for (std::size_t feature = 0; feature < workspace.width_; ++feature)
            backflow_value += feature_row[feature] *
                backflow[feature * workspace.determinants_ * workspace.electrons_ + channel];
          double envelope_value = 0.0;
          for (std::size_t nucleus = 0; nucleus < workspace.nuclei_; ++nucleus)
          {
            const std::size_t index = channel * workspace.nuclei_ + nucleus;
            const double radius     = distances[electron * workspace.nuclei_ + nucleus];
            envelope_value += pi[index] * std::exp(-std::abs(zeta[index] * radius));
          }
          workspace.orbital_matrices_[(determinant * workspace.electrons_ + electron) *
                                           workspace.electrons_ +
                                       orbital] = backflow_value * envelope_value;
        }
    }
  }

  /// Accumulate the analytic cusp value and its two scalar parameter derivatives.
  double cuspValueAndReverse(const double* parameters, DirectScoreWorkspace& workspace) const
  {
    const bool has_same_alpha = plan_.hasParameter(ParameterRole::CUSP_SAME_ALPHA);
    const double same_alpha = has_same_alpha
        ? parameter(parameters, ParameterRole::CUSP_SAME_ALPHA)[0]
        : 1.0;
    const double anti_alpha = parameter(parameters, ParameterRole::CUSP_OPPOSITE_ALPHA)[0];
    double* same_score      = has_same_alpha
        ? score(workspace, ParameterRole::CUSP_SAME_ALPHA)
        : nullptr;
    double& anti_score      = score(workspace, ParameterRole::CUSP_OPPOSITE_ALPHA)[0];
    const auto& pairs       = workspace.geometry_.electronPairs();
    const auto& distances   = workspace.geometry_.electronElectronPairs().distances();
    double cusp             = 0.0;
    for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index)
    {
      const ElectronPair pair = pairs[pair_index];
      const bool same_spin = (pair.first < spin_up_electrons_) == (pair.second < spin_up_electrons_);
      const double alpha   = same_spin ? same_alpha : anti_alpha;
      const double factor  = same_spin ? 0.25 : 0.5;
      const double radius  = distances[pair_index];
      const double denominator = alpha + radius;
      cusp -= factor * alpha * alpha / denominator;
      const double derivative = -factor * alpha * (alpha + 2.0 * radius) /
          (denominator * denominator);
      if (same_spin)
      {
        if (!same_score)
          throw std::logic_error("PsiFormer score plan omitted a required same-spin cusp parameter");
        same_score[0] += derivative;
      }
      else
        anti_score += derivative;
    }
    return cusp;
  }

  /// Seed determinant adjoints and reverse orbitals, backflow, and envelopes.
  void reverseOrbitals(const double* parameters,
                       DirectScoreWorkspace& workspace) const
  {
    std::fill(workspace.feature_adjoint_a_.begin(), workspace.feature_adjoint_a_.end(), 0.0);
    const double* final_features = workspace.features(workspace.blocks_);
    const auto& distances        = workspace.geometry_.electronNucleusPairs().distances();
    const std::size_t matrix_elements = workspace.electrons_ * workspace.electrons_;

    for (std::size_t electron = 0; electron < workspace.electrons_; ++electron)
    {
      const bool spin_up = electron < spin_up_electrons_;
      const ParameterRole backflow_role = spin_up ? ParameterRole::BACKFLOW_UP_WEIGHT
                                                  : ParameterRole::BACKFLOW_DOWN_WEIGHT;
      const ParameterRole pi_role = spin_up ? ParameterRole::ENVELOPE_PI_UP
                                            : ParameterRole::ENVELOPE_PI_DOWN;
      const ParameterRole zeta_role = spin_up ? ParameterRole::ENVELOPE_ZETA_UP
                                              : ParameterRole::ENVELOPE_ZETA_DOWN;
      const double* backflow = parameter(parameters, backflow_role);
      const double* pi       = parameter(parameters, pi_role);
      const double* zeta     = parameter(parameters, zeta_role);
      double* backflow_score = score(workspace, backflow_role);
      double* pi_score       = score(workspace, pi_role);
      double* zeta_score     = score(workspace, zeta_role);
      const double* feature_row = final_features + electron * workspace.width_;
      double* feature_adjoint = workspace.feature_adjoint_a_.data() + electron * workspace.width_;

      for (std::size_t determinant = 0; determinant < workspace.determinants_; ++determinant)
        for (std::size_t orbital = 0; orbital < workspace.electrons_; ++orbital)
        {
          const std::size_t channel = determinant * workspace.electrons_ + orbital;
          const double matrix_adjoint = workspace.matrix_adjoints_[
              determinant * matrix_elements + electron * workspace.electrons_ + orbital];
          double backflow_value = 0.0;
          for (std::size_t feature = 0; feature < workspace.width_; ++feature)
            backflow_value += feature_row[feature] *
                backflow[feature * workspace.determinants_ * workspace.electrons_ + channel];
          double envelope_value = 0.0;
          for (std::size_t nucleus = 0; nucleus < workspace.nuclei_; ++nucleus)
          {
            const std::size_t index = channel * workspace.nuclei_ + nucleus;
            const double radius     = distances[electron * workspace.nuclei_ + nucleus];
            envelope_value += pi[index] * std::exp(-std::abs(zeta[index] * radius));
          }

          const double backflow_adjoint = matrix_adjoint * envelope_value;
          const double envelope_adjoint = matrix_adjoint * backflow_value;
          for (std::size_t feature = 0; feature < workspace.width_; ++feature)
          {
            const std::size_t index =
                feature * workspace.determinants_ * workspace.electrons_ + channel;
            backflow_score[index] += feature_row[feature] * backflow_adjoint;
            feature_adjoint[feature] += backflow[index] * backflow_adjoint;
          }
          for (std::size_t nucleus = 0; nucleus < workspace.nuclei_; ++nucleus)
          {
            const std::size_t index = channel * workspace.nuclei_ + nucleus;
            const double radius     = distances[electron * workspace.nuclei_ + nucleus];
            const double zeta_radius = zeta[index] * radius;
            const double decay       = std::exp(-std::abs(zeta_radius));
            const double abs_derivative = zeta_radius > 0.0 ? 1.0 : (zeta_radius < 0.0 ? -1.0 : 0.0);
            pi_score[index] += envelope_adjoint * decay;
            zeta_score[index] -= envelope_adjoint * pi[index] * decay * abs_derivative * radius;
          }
        }
    }
  }

  /// Reverse one attention/residual block into its input and eight parameter tensors.
  void reverseAttentionBlock(const double* parameters,
                             std::size_t block,
                             DirectScoreWorkspace& workspace) const
  {
    const double* input = workspace.features(block);
    const double* query = DirectScoreWorkspace::blockBuffer(workspace.queries_, block,
                                                             workspace.feature_elements_);
    const double* key = DirectScoreWorkspace::blockBuffer(workspace.keys_, block,
                                                           workspace.feature_elements_);
    const double* value = DirectScoreWorkspace::blockBuffer(workspace.values_, block,
                                                             workspace.feature_elements_);
    const double* attention = DirectScoreWorkspace::blockBuffer(
        workspace.attention_weights_, block, workspace.attention_elements_);
    const double* context = DirectScoreWorkspace::blockBuffer(workspace.contexts_, block,
                                                               workspace.feature_elements_);
    const double* residual = DirectScoreWorkspace::blockBuffer(workspace.residuals_, block,
                                                                workspace.feature_elements_);
    const double* hidden = DirectScoreWorkspace::blockBuffer(workspace.hidden_, block,
                                                              workspace.feature_elements_);
    const double* update = DirectScoreWorkspace::blockBuffer(workspace.updates_, block,
                                                              workspace.feature_elements_);

    // Reverse the output tanh and two dense MLP layers, retaining the residual seed.
    for (std::size_t element = 0; element < workspace.feature_elements_; ++element)
    {
      workspace.residual_adjoint_[element] = workspace.feature_adjoint_a_[element];
      workspace.update_adjoint_[element] =
          workspace.feature_adjoint_a_[element] * (1.0 - update[element] * update[element]);
    }
    std::fill(workspace.hidden_adjoint_.begin(), workspace.hidden_adjoint_.end(), 0.0);
    qmcplusplus::psiformer::direct::denseReverse(
        hidden, parameter(parameters, ParameterRole::UPDATE_OUTPUT_WEIGHT, block),
        workspace.update_adjoint_.data(), workspace.electrons_, workspace.width_, workspace.width_,
        workspace.hidden_adjoint_.data(), score(workspace, ParameterRole::UPDATE_OUTPUT_WEIGHT, block),
        score(workspace, ParameterRole::UPDATE_OUTPUT_BIAS, block));
    for (std::size_t element = 0; element < workspace.feature_elements_; ++element)
      workspace.hidden_adjoint_[element] *= 1.0 - hidden[element] * hidden[element];
    qmcplusplus::psiformer::direct::denseReverse(
        residual, parameter(parameters, ParameterRole::UPDATE_HIDDEN_WEIGHT, block),
        workspace.hidden_adjoint_.data(), workspace.electrons_, workspace.width_, workspace.width_,
        workspace.residual_adjoint_.data(), score(workspace, ParameterRole::UPDATE_HIDDEN_WEIGHT, block),
        score(workspace, ParameterRole::UPDATE_HIDDEN_BIAS, block));

    // Reverse the output projection and seed the explicit attention residual branch.
    std::fill(workspace.context_adjoint_.begin(), workspace.context_adjoint_.end(), 0.0);
    qmcplusplus::psiformer::direct::denseReverse(
        context, parameter(parameters, ParameterRole::ATTENTION_OUTPUT_WEIGHT, block),
        workspace.residual_adjoint_.data(), workspace.electrons_, workspace.width_, workspace.width_,
        workspace.context_adjoint_.data(), score(workspace, ParameterRole::ATTENTION_OUTPUT_WEIGHT, block),
        nullptr);
    std::copy(workspace.residual_adjoint_.begin(), workspace.residual_adjoint_.end(),
              workspace.feature_adjoint_b_.begin());

    // Reverse attention context, row softmax, and scaled query/key products.
    std::fill(workspace.attention_adjoint_.begin(), workspace.attention_adjoint_.end(), 0.0);
    std::fill(workspace.value_adjoint_.begin(), workspace.value_adjoint_.end(), 0.0);
    qmcplusplus::psiformer::direct::attentionContextReverse(
        attention, value, workspace.context_adjoint_.data(), workspace.electrons_, workspace.heads_,
        workspace.head_width_, workspace.attention_adjoint_.data(), workspace.value_adjoint_.data());
    std::fill(workspace.query_adjoint_.begin(), workspace.query_adjoint_.end(), 0.0);
    std::fill(workspace.key_adjoint_.begin(), workspace.key_adjoint_.end(), 0.0);
    qmcplusplus::psiformer::direct::attentionWeightsReverse(
        query, key, attention, workspace.attention_adjoint_.data(), workspace.electrons_, workspace.heads_,
        workspace.head_width_, workspace.query_adjoint_.data(), workspace.key_adjoint_.data());

    // Each Q/K/V projection contributes independently to the same input-feature adjoint.
    qmcplusplus::psiformer::direct::denseReverse(
        input, parameter(parameters, ParameterRole::ATTENTION_QUERY_WEIGHT, block),
        workspace.query_adjoint_.data(), workspace.electrons_, workspace.width_, workspace.width_,
        workspace.feature_adjoint_b_.data(), score(workspace, ParameterRole::ATTENTION_QUERY_WEIGHT, block),
        nullptr);
    qmcplusplus::psiformer::direct::denseReverse(
        input, parameter(parameters, ParameterRole::ATTENTION_KEY_WEIGHT, block),
        workspace.key_adjoint_.data(), workspace.electrons_, workspace.width_, workspace.width_,
        workspace.feature_adjoint_b_.data(), score(workspace, ParameterRole::ATTENTION_KEY_WEIGHT, block),
        nullptr);
    qmcplusplus::psiformer::direct::denseReverse(
        input, parameter(parameters, ParameterRole::ATTENTION_VALUE_WEIGHT, block),
        workspace.value_adjoint_.data(), workspace.electrons_, workspace.width_, workspace.width_,
        workspace.feature_adjoint_b_.data(), score(workspace, ParameterRole::ATTENTION_VALUE_WEIGHT, block),
        nullptr);
    workspace.feature_adjoint_a_.swap(workspace.feature_adjoint_b_);
  }

  /// Finish the reverse pass with the raw-feature embedding weight gradient.
  void reverseEmbedding(DirectScoreWorkspace& workspace) const
  {
    double* embedding_score = score(workspace, ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
    for (std::size_t electron = 0; electron < workspace.electrons_; ++electron)
      for (std::size_t input = 0; input < workspace.input_width_; ++input)
        for (std::size_t output = 0; output < workspace.width_; ++output)
          embedding_score[input * workspace.width_ + output] +=
              workspace.raw_features_[electron * workspace.input_width_ + input] *
              workspace.feature_adjoint_a_[electron * workspace.width_ + output];
  }

  const Parameters& parameters_;
  const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan_;
  std::vector<double> nuclei_;
  std::size_t spin_up_electrons_;
  GeometryBoundary boundary_;
};

} // namespace pf

#endif // QMCPLUSPLUS_PSIFORMER_SCORE_EXECUTOR_H
