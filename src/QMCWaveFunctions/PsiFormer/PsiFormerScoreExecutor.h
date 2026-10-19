//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerScoreExecutor.h
 * @brief Shared real open/periodic score tape and guarded complete real score evaluator.
 *
 * The evaluator records a compact forward tape and explicitly applies the reverse
 * chain rule for the embedding, attention blocks, orbital/envelope construction,
 * determinant sum, and cusp.  The returned vector is d log|Psi| / d parameter in
 * the canonical imported-HDF5 order used by QMCPACK's optimizer registration.
 * A channel-resolved route also returns each pre-sum determinant amplitude and
 * reverses an arbitrary weighted sum of channel log amplitudes through the shared
 * feature trunk and cusp in one pass.
 *
 * Immutable tensor offsets and execution semantics come from PsiFormerExecutionPlan;
 * all tape, adjoint, geometry, and output storage is clone-local.  The implemented
 * neural kernels are real/fixed-nucleus; explicit head-free preparation can expose
 * a real open/periodic tape for complex composition without admitting a complex
 * public wavefunction, score or optimizer capability.
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
#include "PsiFormerStorageRequirements.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerAffineObservation.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerGeometry.h"
#include "PsiFormerPeriodicFeatures.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

/** Shared real cusp value and scaled real parameter reverse, preserving legacy order. */
inline double realCuspValueAndReverse(const PsiFormerGeometryCache& geometry,
    std::size_t spin_up_electrons, double same_alpha, double anti_alpha,
    double* same_score, double& anti_score, double reverse_seed)
{
  const auto& pairs       = geometry.electronPairs();
  const auto& distances   = geometry.electronElectronPairs().distances();
  double cusp             = 0.0;
  for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index)
  {
    const ElectronPair pair = pairs[pair_index];
    const bool same_spin = (pair.first < spin_up_electrons) == (pair.second < spin_up_electrons);
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
      same_score[0] += reverse_seed * derivative;
    }
    else
      anti_score += reverse_seed * derivative;
  }
  return cusp;
}

/** Return the wavefunction value and its canonical log-parameter score. */
struct DirectScoreResult
{
  double sign   = 1.0;
  double logabs = 0.0;
  double value  = 0.0;
  DirectParameterScoreView parameter_score;
  std::size_t parameter_version = 0;
};

/** Describe caller-owned per-determinant signed/log channel amplitudes. */
struct DirectChannelAmplitudeResult
{
  const double* phases = nullptr;
  const double* log_abs = nullptr;
  std::size_t channel_count = 0;
  std::size_t parameter_version = 0;
};

/** Return one weighted sum of independent channel log-parameter scores. */
struct DirectWeightedChannelScoreResult
{
  DirectParameterScoreView parameter_score;
  std::size_t parameter_version = 0;
};

/** Identify the fixed target matrix consumed by one orbital-MSE evaluation. */
struct DirectOrbitalMSEArguments
{
  const double* target = nullptr;
  std::size_t target_size = 0;
};

/// Borrow pre-determinant orbital values until the next workspace evaluation.
struct DirectOrbitalValueView
{
  const double* values = nullptr;
  std::size_t size = 0;
  std::size_t parameter_version = 0;
};

/** Return one pre-determinant orbital loss and its direct reverse derivative. */
struct DirectOrbitalMSEResult
{
  double loss = 0.0;
  DirectParameterScoreView parameter_gradient;
  const double* predicted_orbitals = nullptr;
  std::size_t orbital_count = 0;
  std::size_t parameter_version = 0;
};

/// Stable executor identity binds each new head-free tape to its originating owner.
class DirectScoreExecutor;

/**
 * Own the forward tape, reverse adjoints, geometry cache, and score for one clone.
 *
 * Every vector is sized in the constructor and never resized during evaluation.
 * This makes a warmed-up call allocation-free and prevents mutable scratch from
 * being shared between QMCPACK walker/crowd clones.
 */
class DirectScoreTrunkWorkspace
{
public:
  /// Allocate the exact tape and reverse storage required by one execution plan.
  DirectScoreTrunkWorkspace(const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan,
                            GeometryPositionView nuclei,
                            GeometryBoundary boundary = {},
                            const DirectScoreExecutor* owner = nullptr)
      : owner_(owner), electrons_(plan.modelShape().electrons()),
        nuclei_(plan.modelShape().nuclei),
        determinants_(plan.modelShape().determinants),
        width_(plan.modelShape().feature_dimension),
        heads_(plan.modelShape().attention_heads),
        blocks_(plan.modelShape().attention_blocks),
        head_width_(width_ / heads_),
        input_width_(plan.parameter(qmcplusplus::psiformer::ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0]),
        feature_elements_(electrons_ * width_),
        attention_elements_(heads_ * electrons_ * electrons_),
        orbital_elements_(determinants_ * electrons_ * electrons_),
        electron_positions_(3 * electrons_),
        geometry_(qmcplusplus::psiformer::periodic_feature::makeGeometry(
            electrons_, nuclei, boundary, plan.environment().periodic_profile)),
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
        update_adjoint_(feature_elements_),
        pair_features_(plan.environment().periodic_profile ? checkedStorageProduct(8,
            checkedStorageProduct(electrons_, electrons_ - 1, "Pair feature rows overflowed"),
            "Pair feature scratch overflowed") : 0),
        pair_sensitivities_(plan.environment().periodic_profile ? checkedStorageProduct(heads_,
            checkedStorageProduct(electrons_, electrons_ - 1, "Pair response rows overflowed"),
            "Pair response scratch overflowed") : 0)
  {
    if (heads_ == 0 || width_ % heads_ != 0)
      throw std::invalid_argument("PsiFormer score workspace received inconsistent attention dimensions");
  }

  /// Prepared mutable tape/adjoints cannot be copied across clones.
  DirectScoreTrunkWorkspace(const DirectScoreTrunkWorkspace&) = delete;

  /// Preserve exclusive numerical buffer ownership.
  DirectScoreTrunkWorkspace& operator=(const DirectScoreTrunkWorkspace&) = delete;

  /// Moving transfers preparation but invalidates old borrowed workspace-address views.
  DirectScoreTrunkWorkspace(DirectScoreTrunkWorkspace&&) = default;

  /// Assignment similarly invalidates every old borrowed tape/result.
  DirectScoreTrunkWorkspace& operator=(DirectScoreTrunkWorkspace&&) = default;

  /// Polymorphic ownership keeps derived full-score storage destruction safe.
  virtual ~DirectScoreTrunkWorkspace() = default;

  /// Return the forward tape stamp, unaffected by successful reverse passes.
  std::size_t forwardGeneration() const noexcept { return forward_generation_; }

  /// Return the separate reverse result stamp, including rejected reverse requests.
  std::size_t resultGeneration() const noexcept { return reverse_generation_; }

  /// Expose a completed reverse scratch result until any next request invalidates it.
  DirectParameterScoreView trunkParameterScore() const
  {
    if (!reverse_valid_)
      throw std::logic_error("PsiFormer trunk reverse result unavailable");
    return {parameter_score_.data(), parameter_score_.size()};
  }

  /// Copy a complete interleaved electron configuration into fixed workspace storage.
  void setPositions(GeometryPositionView positions)
  {
    if (owner_)
      invalidateTape();
    if (positions.size() != electrons_)
      throw std::invalid_argument("PsiFormer score workspace received the wrong electron count");
    for (std::size_t electron = 0; electron < electrons_; ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        electron_positions_[3 * electron + dimension] = positions(electron, dimension);
  }

  /// Overwrite one Cartesian coordinate without changing any buffer capacity.
  void setPosition(std::size_t electron, std::size_t dimension, double coordinate)
  {
    if (owner_)
      invalidateTape();
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

  /// Return bytes retained by the one full canonical score vector.
  std::size_t scoreStorageBytes() const
  {
    return checkedStorageBytes<double>(parameter_score_.capacity(),
                                       "PsiFormer score-vector bytes overflowed");
  }

  /// Hash every explicit backing allocation for warmed-call stability tests.
  std::size_t storageFingerprint() const noexcept
  {
    std::size_t hash = 1469598103934665603ULL;
    auto mix = [&hash](const auto& buffer) {
      hash ^= reinterpret_cast<std::uintptr_t>(buffer.data());
      hash *= 1099511628211ULL;
      hash ^= buffer.capacity();
      hash *= 1099511628211ULL;
    };
    for (const auto* buffer : {
             &electron_positions_, &raw_features_, &features_, &queries_,
             &keys_, &values_, &attention_weights_, &contexts_, &residuals_,
             &hidden_, &updates_,
             &parameter_score_, &feature_adjoint_a_, &feature_adjoint_b_,
             &query_adjoint_, &key_adjoint_, &value_adjoint_,
             &attention_adjoint_, &context_adjoint_, &residual_adjoint_,
             &hidden_adjoint_, &update_adjoint_, &pair_features_, &pair_sensitivities_})
      mix(*buffer);
    hash ^= geometry_.storageFingerprint();
    hash *= 1099511628211ULL;
    return hash;
  }

  /// Return bytes reserved by the complete tape, geometry, and determinant workspace.
  std::size_t vectorStorageBytes() const
  {
    std::size_t bytes = 0;
    auto add = [&bytes](const std::vector<double>& buffer) {
      addStorageBytes(bytes,
                      checkedStorageBytes<double>(
                          buffer.capacity(),
                          "PsiFormer score storage bytes overflowed"),
                      "PsiFormer score storage bytes overflowed");
    };
    for (const auto* buffer : {
             &electron_positions_, &raw_features_, &features_, &queries_,
             &keys_, &values_, &attention_weights_, &contexts_, &residuals_,
             &hidden_, &updates_,
             &parameter_score_, &feature_adjoint_a_, &feature_adjoint_b_,
             &query_adjoint_, &key_adjoint_, &value_adjoint_,
             &attention_adjoint_, &context_adjoint_, &residual_adjoint_,
             &hidden_adjoint_, &update_adjoint_, &pair_features_, &pair_sensitivities_})
      add(*buffer);
    addStorageBytes(bytes, geometry_.storageBytes(),
                    "PsiFormer score storage bytes overflowed");
    return bytes;
  }

  /// Expose the geometry contribution for focused inclusive-accounting tests.
  std::size_t geometryStorageBytes() const
  { return geometry_.storageBytes(); }

  /// Return the checked constructor-time requirement for this fixed model shape.
  std::size_t requiredStorageBytes() const
  {
    const std::size_t complete = scoreWorkspaceStorageRequirement(
        {electrons_, nuclei_, determinants_, width_, heads_, input_width_,
         blocks_, parameter_score_.size(), !pair_features_.empty()});
    return complete - headStorageBytes();
  }

protected:
  friend class DirectScoreExecutor;

  /// Count exactly the existing full-score head allocations omitted by this base.
  std::size_t headStorageBytes() const
  {
    return checkedStorageBytes<double>(3 * orbital_elements_, "score head bytes overflow") +
        determinantStorageRequirement(determinants_, electrons_);
  }

  /// Reject old borrowed tape/result views before any new input or forward request.
  void invalidateTape()
  {
    tape_valid_ = reverse_valid_ = false;
    observed_parameter_version_ = std::numeric_limits<std::size_t>::max();
    if (forward_generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("PsiFormer score forward generation exhausted");
    ++forward_generation_;
  }

  /// Failed reverse attempts cannot expose the previous score publication.
  void invalidateReverse()
  {
    reverse_valid_ = false;
    observed_parameter_version_ = std::numeric_limits<std::size_t>::max();
    if (reverse_generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("PsiFormer score reverse generation exhausted");
    ++reverse_generation_;
  }

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

  const DirectScoreExecutor* owner_;
  std::size_t forward_generation_ = 0, reverse_generation_ = 0;
  std::size_t tape_parameter_version_ = std::numeric_limits<std::size_t>::max();
  bool tape_valid_ = false, reverse_valid_ = false;
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
  std::vector<double> pair_features_;
  std::vector<double> pair_sensitivities_;
  std::size_t observed_parameter_version_ = std::numeric_limits<std::size_t>::max();
};

/// Legacy complete real score storage adds only real orbital/determinant head buffers.
class DirectScoreWorkspace : public DirectScoreTrunkWorkspace
{
public:
  /// Preserve every default real allocation and its inclusive accounting.
  DirectScoreWorkspace(const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan,
                       GeometryPositionView nuclei, GeometryBoundary boundary = {})
      : DirectScoreTrunkWorkspace(plan, nuclei, boundary),
        orbital_matrices_(orbital_elements_), determinant_workspace_(determinants_, electrons_),
        matrix_adjoints_(orbital_elements_), backflow_adjoints_(orbital_elements_)
  {}

  /// Include legacy head allocations in warmed allocation fingerprints.
  std::size_t storageFingerprint() const noexcept
  {
    std::size_t hash = DirectScoreTrunkWorkspace::storageFingerprint();
    for (const auto* buffer : {&orbital_matrices_, &matrix_adjoints_, &backflow_adjoints_})
    {
      hash ^= reinterpret_cast<std::uintptr_t>(buffer->data());
      hash *= 1099511628211ULL;
      hash ^= buffer->capacity();
      hash *= 1099511628211ULL;
    }
    return hash ^ determinant_workspace_.storageFingerprint();
  }

  /// Include the complete real head without changing the shared trunk's buffer sizes.
  std::size_t vectorStorageBytes() const
  {
    return DirectScoreTrunkWorkspace::vectorStorageBytes() +
        checkedStorageBytes<double>(orbital_matrices_.capacity() + matrix_adjoints_.capacity() +
            backflow_adjoints_.capacity(), "score head bytes overflow") + determinant_workspace_.storageBytes();
  }

  /// Preserve the independent exact legacy complete-workspace requirement.
  std::size_t requiredStorageBytes() const
  {
    return scoreWorkspaceStorageRequirement({electrons_, nuclei_, determinants_, width_, heads_,
        input_width_, blocks_, parameter_score_.size()});
  }

private:
  friend class DirectScoreExecutor;
  std::vector<double> orbital_matrices_;
  qmcplusplus::psiformer::determinant::RealOpenDeterminantWorkspace determinant_workspace_;
  std::vector<double> matrix_adjoints_, backflow_adjoints_;
};

/// Explicit head-free preparation never changes the default real-wavefunction admission.
enum class DirectScorePreparation { REAL_WAVEFUNCTION, REAL_TRUNK_ONLY };

/// Borrow immutable forward features and geometry until any next input/forward request.
struct DirectScoreTrunkResultView
{
  const DirectScoreTrunkWorkspace* workspace = nullptr;
  const double* features = nullptr;
  std::size_t feature_count = 0;
  const PsiFormerGeometryCache* geometry = nullptr;
  const double* packed_positions = nullptr;
  std::size_t parameter_version = 0, forward_generation = 0;
};

/// Borrow one real cotangent response until the next attempted reverse/forward request.
struct DirectScoreTrunkReverseResult
{
  DirectParameterScoreView parameter_score;
  std::size_t parameter_version = 0, forward_generation = 0, result_generation = 0;
};

/** Evaluate complete real scores or expose a head-free real taped reverse seam.
 * Moving this executor invalidates its owner-bound trunk workspaces; copy-created
 * executors have independent identities. Complete real default behavior is unchanged.
 */
class DirectScoreExecutor
{
public:
  /// Reuse an execution plan shared by the owning PsiFormer wavefunction state.
  DirectScoreExecutor(const PsiFormer& model,
                      const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan,
                      DirectScorePreparation preparation = DirectScorePreparation::REAL_WAVEFUNCTION)
      : preparation_(preparation), parameters_(model.p),
        plan_(plan),
        nuclei_(model.cfg.nuclei.x),
        spin_up_electrons_(model.cfg.nup),
        boundary_(mappedBoundary(plan.environment()))
  {
    if (plan_.environment().periodic_profile)
      for (const auto& column : plan_.environment().periodic_profile->occupied_columns)
        if (column.reciprocal != std::array<int, 3>{})
          throw std::invalid_argument("Occupied reciprocal phases require a qualified score head");
    if (plan_.parameterCount() != parameters_.size() ||
        plan_.modelShape().electrons() != model.ne ||
        plan_.modelShape().nuclei != model.cfg.nuclei.shape[0] ||
        plan_.modelShape().determinants != model.ndet ||
        plan_.modelShape().feature_dimension != model.dim ||
        plan_.modelShape().attention_heads != model.heads)
      throw std::invalid_argument("PsiFormer score execution plan does not match the imported model");
    if (plan_.environment().parameter_scalar_domain != qmcplusplus::psiformer::ScalarDomain::REAL ||
        plan_.environment().compute_scalar_domain != qmcplusplus::psiformer::ScalarDomain::REAL ||
        (preparation_ == DirectScorePreparation::REAL_WAVEFUNCTION &&
         plan_.environment().amplitude_scalar_domain != qmcplusplus::psiformer::ScalarDomain::REAL) ||
        (preparation_ != DirectScorePreparation::REAL_WAVEFUNCTION &&
         preparation_ != DirectScorePreparation::REAL_TRUNK_ONLY) ||
        !plan_.environment().fixed_nuclei)
      throw std::invalid_argument("Direct PsiFormer score supports only real, fixed-nucleus models");
  }

  /// Construct independently mutable tape and adjoint storage for one clone.
  std::unique_ptr<DirectScoreWorkspace> makeWorkspace() const
  {
    requireFullProfile();
    return std::make_unique<DirectScoreWorkspace>(
        plan_, GeometryPositionView::interleaved(nuclei_.data(), plan_.modelShape().nuclei), boundary_);
  }

  /// Count the executor's copied fixed-ion numerical metadata, separate from tape buffers.
  std::size_t numericMetadataStorageBytes() const
  { return checkedStorageBytes<double>(nuclei_.capacity(), "executor metadata bytes overflow"); }

  /// Include fixed-ion allocation identity in retained-lane stability evidence.
  std::size_t numericMetadataStorageFingerprint() const noexcept
  {
    std::size_t hash = 1469598103934665603ULL;
    qmcplusplus::psiformer::determinant::detail::mixStorage(hash, nuclei_);
    return hash;
  }

  /// Prepare one owner-bound real tape without real orbital/determinant head buffers.
  std::unique_ptr<DirectScoreTrunkWorkspace> makeTrunkWorkspace() const
  {
    return std::make_unique<DirectScoreTrunkWorkspace>(plan_,
        GeometryPositionView::interleaved(nuclei_.data(), plan_.modelShape().nuclei), boundary_, this);
  }

  /// Publish one immutable tape; any failed new request invalidates the previous tape.
  DirectScoreTrunkResultView evaluateTrunk(DirectScoreTrunkWorkspace& workspace) const
  {
    workspace.invalidateTape();
    validateTrunkOwner(workspace);
    buildTrunk(parameters_.flat_values().data(), workspace);
    for (const auto* buffer : {&workspace.raw_features_, &workspace.features_, &workspace.queries_,
             &workspace.keys_, &workspace.values_, &workspace.attention_weights_, &workspace.contexts_,
             &workspace.residuals_, &workspace.hidden_, &workspace.updates_})
      for (double value : *buffer)
        if (!is_finite_parameter_value(value))
          throw std::overflow_error("PsiFormer score trunk produced a nonfinite tape");
    workspace.tape_parameter_version_ = parameters_.version();
    workspace.observed_parameter_version_ = parameters_.version();
    workspace.tape_valid_ = true;
    return {&workspace, workspace.features(workspace.blocks_), workspace.feature_elements_,
            &workspace.geometry_, workspace.electron_positions_.data(),
            parameters_.version(), workspace.forward_generation_};
  }

  /** Reverse a real feature cotangent over the same immutable forward tape.
   * Each pass clears the one P-sized scratch and all adjoints. Forward buffers are
   * read-only throughout; a failed reverse invalidates only reverse publication,
   * so a corrected seed may reuse the same current tape. Stale parameters require
   * a fresh forward call. Seeds may not alias the reusable reverse score scratch.
   */
  DirectScoreTrunkReverseResult reverseTrunk(DirectScoreTrunkWorkspace& workspace,
      const DirectScoreTrunkResultView& tape, const double* seed, std::size_t seed_count,
      DirectAffineObservationSink* observer = nullptr) const
  {
    workspace.invalidateReverse();
    validateTrunkOwner(workspace);
    if (!workspace.tape_valid_ || tape.workspace != &workspace ||
        tape.forward_generation != workspace.forward_generation_ ||
        tape.parameter_version != parameters_.version() ||
        tape.parameter_version != workspace.tape_parameter_version_ ||
        tape.features != workspace.features(workspace.blocks_) ||
        tape.feature_count != workspace.feature_elements_)
      throw std::invalid_argument("PsiFormer score reverse requires the current owner-bound tape");
    if (!seed || seed_count != workspace.feature_elements_)
      throw std::invalid_argument("PsiFormer score feature cotangent extent disagrees");
    const auto seed_begin = reinterpret_cast<std::uintptr_t>(seed);
    const auto score_begin = reinterpret_cast<std::uintptr_t>(workspace.parameter_score_.data());
    if (seed_begin < score_begin + workspace.parameter_score_.size() * sizeof(double) &&
        score_begin < seed_begin + seed_count * sizeof(double))
      throw std::invalid_argument("PsiFormer score feature cotangent aliases reverse output");
    for (std::size_t i = 0; i < seed_count; ++i)
      if (!is_finite_parameter_value(seed[i]))
        throw std::invalid_argument("PsiFormer score feature cotangent is nonfinite");
    for (auto* buffer : {&workspace.parameter_score_, &workspace.feature_adjoint_a_,
             &workspace.feature_adjoint_b_, &workspace.query_adjoint_, &workspace.key_adjoint_,
             &workspace.value_adjoint_, &workspace.attention_adjoint_, &workspace.context_adjoint_,
             &workspace.residual_adjoint_, &workspace.hidden_adjoint_, &workspace.update_adjoint_})
      std::fill(buffer->begin(), buffer->end(), 0.0);
    std::copy_n(seed, seed_count, workspace.feature_adjoint_a_.data());
    reverseFeatures(parameters_.flat_values().data(), workspace, observer);
    if (observer)
      observer->observe({qmcplusplus::psiformer::ParameterRole::ELECTRON_EMBEDDING_WEIGHT,
                         qmcplusplus::psiformer::NO_ATTENTION_BLOCK,
                         workspace.raw_features_.data(), workspace.feature_adjoint_a_.data(),
                         workspace.electrons_, workspace.input_width_, workspace.width_});
    // Reverse helpers swap only adjoint buffers, never any immutable forward tape.
    if (workspace.blocks_ % 2)
      workspace.feature_adjoint_a_.swap(workspace.feature_adjoint_b_);
    for (double value : workspace.parameter_score_)
      if (!is_finite_parameter_value(value))
        throw std::overflow_error("PsiFormer score trunk reverse produced a nonfinite response");
    workspace.observed_parameter_version_ = parameters_.version();
    workspace.reverse_valid_ = true;
    return {{workspace.parameter_score_.data(), workspace.parameter_score_.size()},
            parameters_.version(), workspace.forward_generation_, workspace.reverse_generation_};
  }

  /// Return immutable typed execution metadata shared by all workspaces.
  const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan() const noexcept
  {
    return plan_;
  }

  /// Execute one taped forward evaluation and one explicit parameter reverse pass.
  DirectScoreResult evaluate(DirectScoreWorkspace& workspace) const
  {
    return evaluateImpl(workspace, nullptr);
  }

  /** Evaluate every pre-sum determinant channel in signed/log form.
   *
   * Each channel includes the common PsiFormer cusp.  Output storage remains owned
   * by the caller, and the warmed evaluator performs no allocation.
   */
  DirectChannelAmplitudeResult evaluateChannelAmplitudes(
      DirectScoreWorkspace& workspace,
      double* phases,
      double* log_abs,
      std::size_t channel_count) const
  {
    requireFullProfile();
    validateWorkspaceAndParameters(workspace);
    if (!phases || !log_abs)
      throw std::invalid_argument("PsiFormer channel-amplitude output pointer is null");
    if (channel_count != workspace.determinants_)
      throw std::invalid_argument("PsiFormer channel-amplitude output count is incorrect");

    const double* parameter_values = parameters_.flat_values().data();
    buildTrunk(parameter_values, workspace);
    buildOrbitals(parameter_values, workspace);
    workspace.determinant_workspace_.evaluateValue(workspace.orbital_matrices_.data());
    const double cusp = cuspValueAndReverse(parameter_values, workspace, 0.0);

    // Validate all staged channel values before publishing any caller output.
    for (std::size_t channel = 0; channel < channel_count; ++channel)
    {
      const auto amplitude = workspace.determinant_workspace_.channelAmplitude(channel);
      workspace.matrix_adjoints_[channel] = amplitude.phase;
      workspace.backflow_adjoints_[channel] =
          amplitude.isZero() ? amplitude.log_abs : amplitude.log_abs + cusp;
      if (!amplitude.isZero() &&
          !is_finite_parameter_value(workspace.backflow_adjoints_[channel]))
        throw std::overflow_error("PsiFormer channel log amplitude is non-finite");
    }
    std::copy_n(workspace.matrix_adjoints_.data(), channel_count, phases);
    std::copy_n(workspace.backflow_adjoints_.data(), channel_count, log_abs);
    workspace.observed_parameter_version_ = parameters_.version();
    return {phases, log_abs, channel_count, workspace.observed_parameter_version_};
  }

  /** Reverse one weighted sum of independent determinant-channel log amplitudes.
   *
   * The determinant heads receive independent ``weight[k] * A_k^{-T}`` seeds,
   * while the shared cusp and feature trunk are traversed once.  This computes
   * ``sum_k weight[k] d log|Psi_k|/dp`` with one canonical P-sized result.
   */
  DirectWeightedChannelScoreResult evaluateWeightedChannelScore(
      DirectScoreWorkspace& workspace,
      const double* weights,
      std::size_t weight_count) const
  {
    requireFullProfile();
    validateWorkspaceAndParameters(workspace);
    if (!weights)
      throw std::invalid_argument("PsiFormer weighted channel score pointer is null");
    if (weight_count != workspace.determinants_)
      throw std::invalid_argument("PsiFormer weighted channel score count is incorrect");
    double cusp_seed = 0.0;
    for (std::size_t channel = 0; channel < weight_count; ++channel)
    {
      if (!is_finite_parameter_value(weights[channel]))
        throw std::invalid_argument("PsiFormer weighted channel score is non-finite");
      cusp_seed += weights[channel];
    }
    if (!is_finite_parameter_value(cusp_seed))
      throw std::overflow_error("PsiFormer weighted channel cusp seed overflowed");

    const double* parameter_values = parameters_.flat_values().data();
    std::fill(workspace.parameter_score_.begin(), workspace.parameter_score_.end(), 0.0);
    buildTrunk(parameter_values, workspace);
    buildOrbitals(parameter_values, workspace);
    workspace.determinant_workspace_.evaluate(workspace.orbital_matrices_.data());
    workspace.determinant_workspace_.fillWeightedChannelLogAmplitudeMatrixAdjoints(
        weights, weight_count, workspace.matrix_adjoints_.data());
    cuspValueAndReverse(parameter_values, workspace, cusp_seed);

    reverseOrbitals(parameter_values, workspace);
    reverseFeatures(parameter_values, workspace);
    workspace.observed_parameter_version_ = parameters_.version();
    return {{workspace.parameter_score_.data(), workspace.parameter_score_.size()},
            workspace.observed_parameter_version_};
  }

  /** Evaluate a score while streaming every supported affine layer boundary. */
  DirectScoreResult evaluateWithAffineObservations(
      DirectScoreWorkspace& workspace,
      DirectAffineObservationSink& observer) const
  {
    return evaluateImpl(workspace, &observer);
  }

private:
  /// Share the exact score traversal between ordinary and observed evaluations.
  DirectScoreResult evaluateImpl(DirectScoreWorkspace& workspace,
                                 DirectAffineObservationSink* observer) const
  {
    requireFullProfile();
    validateWorkspaceAndParameters(workspace);
    const double* parameter_values = parameters_.flat_values().data();
    std::fill(workspace.parameter_score_.begin(), workspace.parameter_score_.end(), 0.0);

    buildTrunk(parameter_values, workspace);
    buildOrbitals(parameter_values, workspace);

    namespace determinant = qmcplusplus::psiformer::determinant;
    const determinant::RealDeterminantResult determinant_result =
        workspace.determinant_workspace_.evaluate(workspace.orbital_matrices_.data());
    if (determinant_result.amplitude.isZero())
      throw std::runtime_error("PsiFormer score executor reached an exact determinant node");
    workspace.determinant_workspace_.fillLogAmplitudeMatrixAdjoints(
        workspace.matrix_adjoints_.data());

    const double cusp   = cuspValueAndReverse(parameter_values, workspace, 1.0);
    const double logabs = determinant_result.amplitude.log_abs + cusp;
    if (!is_finite_parameter_value(logabs))
      throw std::runtime_error("PsiFormer score executor produced a non-finite log wavefunction");

    reverseOrbitals(parameter_values, workspace, observer);
    reverseFeatures(parameter_values, workspace, observer);
    if (observer)
      observer->observe({ParameterRole::ELECTRON_EMBEDDING_WEIGHT,
                         qmcplusplus::psiformer::NO_ATTENTION_BLOCK,
                         workspace.raw_features_.data(),
                         workspace.feature_adjoint_a_.data(), workspace.electrons_,
                         workspace.input_width_, workspace.width_});

    workspace.observed_parameter_version_ = parameters_.version();
    const double sign = determinant_result.amplitude.phase;
    return {sign,
            logabs,
            determinant::realValue(determinant_result.amplitude, cusp),
            {workspace.parameter_score_.data(), workspace.parameter_score_.size()},
            workspace.observed_parameter_version_};
  }

public:
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

  /// Evaluate the existing orbital forward without determinant, energy or reverse work.
  DirectOrbitalValueView evaluateOrbitalValues(DirectScoreWorkspace& workspace) const
  {
    requireFullProfile();
    validateWorkspaceAndParameters(workspace);
    const double* parameter_values = parameters_.flat_values().data();
    buildTrunk(parameter_values, workspace);
    buildOrbitals(parameter_values, workspace);
    if (!std::all_of(workspace.orbital_matrices_.begin(), workspace.orbital_matrices_.end(),
                    [](double value) { return is_finite_parameter_value(value); }))
      throw std::runtime_error("PsiFormer orbital forward produced a nonfinite value");
    workspace.observed_parameter_version_ = parameters_.version();
    return {workspace.orbital_matrices_.data(), workspace.orbital_matrices_.size(),
            workspace.observed_parameter_version_};
  }

  /** Evaluate the spin-normalized pre-determinant orbital loss and its VJP.
   * Each electron-spin sector includes all full-determinant columns, retaining
   * the exact-zero off-spin target constraints. Determinant/cusp work is absent.
   */
  DirectOrbitalMSEResult evaluateOrbitalMSE(DirectScoreWorkspace& workspace,
                                            DirectOrbitalMSEArguments arguments) const
  {
    requireFullProfile();
    validateWorkspaceAndParameters(workspace);
    if (arguments.target_size != workspace.orbital_elements_ ||
        (arguments.target_size != 0 && !arguments.target))
      throw std::invalid_argument("PsiFormer orbital-MSE target has the wrong size");

    const double* parameter_values = parameters_.flat_values().data();
    std::fill(workspace.parameter_score_.begin(), workspace.parameter_score_.end(), 0.0);
    buildTrunk(parameter_values, workspace);
    buildOrbitals(parameter_values, workspace);

    const std::size_t up_elements =
        workspace.determinants_ * spin_up_electrons_ * workspace.electrons_;
    const std::size_t down_electrons = workspace.electrons_ - spin_up_electrons_;
    const std::size_t down_elements =
        workspace.determinants_ * down_electrons * workspace.electrons_;
    const double up_scale = up_elements == 0 ? 0.0 : 1.0 / static_cast<double>(up_elements);
    const double down_scale = down_elements == 0 ? 0.0 : 1.0 / static_cast<double>(down_elements);

    double up_loss = 0.0;
    double down_loss = 0.0;
    const std::size_t matrix_elements = workspace.electrons_ * workspace.electrons_;
    for (std::size_t determinant = 0; determinant < workspace.determinants_; ++determinant)
      for (std::size_t electron = 0; electron < workspace.electrons_; ++electron)
        for (std::size_t orbital = 0; orbital < workspace.electrons_; ++orbital)
        {
          const std::size_t index = determinant * matrix_elements +
              electron * workspace.electrons_ + orbital;
          const double residual = workspace.orbital_matrices_[index] - arguments.target[index];
          const double scale = electron < spin_up_electrons_ ? up_scale : down_scale;
          workspace.matrix_adjoints_[index] = 2.0 * scale * residual;
          if (electron < spin_up_electrons_)
            up_loss += scale * residual * residual;
          else
            down_loss += scale * residual * residual;
        }

    reverseOrbitals(parameter_values, workspace);
    reverseFeatures(parameter_values, workspace);

    workspace.observed_parameter_version_ = parameters_.version();
    const double loss = up_loss + down_loss;
    if (!is_finite_parameter_value(loss) ||
        !std::all_of(workspace.parameter_score_.begin(), workspace.parameter_score_.end(),
                     [](double value) { return is_finite_parameter_value(value); }))
      throw std::runtime_error("PsiFormer orbital-MSE reverse pass produced a non-finite result");
    return {loss,
            {workspace.parameter_score_.data(), workspace.parameter_score_.size()},
            workspace.orbital_matrices_.data(), workspace.orbital_matrices_.size(),
            workspace.observed_parameter_version_};
  }

private:
  using ParameterRole = qmcplusplus::psiformer::ParameterRole;
  using ParameterTensorDescriptor = qmcplusplus::psiformer::ParameterTensorDescriptor;

  /// Complete real-head entry points remain unavailable for explicit head-free owners.
  void requireFullProfile() const
  {
    if (preparation_ != DirectScorePreparation::REAL_WAVEFUNCTION)
      throw std::logic_error("PsiFormer trunk-only score owner cannot evaluate a real head");
  }

  /// New borrowed tape admission is identity-bound rather than merely shape-compatible.
  void validateTrunkOwner(const DirectScoreTrunkWorkspace& workspace) const
  {
    const auto& boundary = workspace.geometry_.boundary();
    if (workspace.owner_ != this || nuclei_.size() != 3 * workspace.nuclei_ || boundary.kind != boundary_.kind ||
        boundary.lattice_vectors != boundary_.lattice_vectors || boundary.periodic_axes != boundary_.periodic_axes)
      throw std::invalid_argument("PsiFormer score trunk workspace belongs to another executor");
    validateWorkspaceAndParameters(workspace);
    for (std::size_t nucleus = 0; nucleus < workspace.nuclei_; ++nucleus)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        if (workspace.geometry_.nuclei()[nucleus][dimension] != nuclei_[3 * nucleus + dimension])
          throw std::invalid_argument("PsiFormer score trunk workspace has different nuclei");
  }

  /// Share the exact real geometry, embedding and attention forward loop with legacy paths.
  void buildTrunk(const double* parameters, DirectScoreTrunkWorkspace& workspace) const
  {
    workspace.geometry_.update(
        GeometryPositionView::interleaved(workspace.electron_positions_.data(), workspace.electrons_));
    if (plan_.environment().periodic_profile)
      for (std::size_t pair = 0; pair < workspace.geometry_.electronPairs().size(); ++pair)
      {
        const auto& identity = workspace.geometry_.electronPairs()[pair];
        const double spin = (identity.first < spin_up_electrons_) == (identity.second < spin_up_electrons_) ? 1 : -1;
        for (int orientation : {1, -1})
        {
          const auto row = orientation == 1 ? identity.first : identity.second;
          const auto column = orientation == 1 ? identity.second : identity.first;
          const auto ordered = row * (workspace.electrons_ - 1) + column - (column > row ? 1 : 0);
          const auto jet = qmcplusplus::psiformer::periodic_feature::evaluate(
              workspace.geometry_.electronElectronPairs(), pair, workspace.geometry_.boundary(), orientation,
              plan_.environment().periodic_profile->feature_transform, spin);
          std::copy_n(jet.value.data(), 8, workspace.pair_features_.data() + 8 * ordered);
        }
      }
    buildEmbedding(parameters, workspace);
    for (std::size_t block = 0; block < workspace.blocks_; ++block)
      applyAttentionBlock(parameters, block, workspace);
  }

  /// Shared reverse loop reads immutable NN buffers and writes only real adjoint scratch.
  void reverseFeatures(const double* parameters, DirectScoreTrunkWorkspace& workspace,
                       DirectAffineObservationSink* observer = nullptr) const
  {
    for (std::size_t reverse_index = workspace.blocks_; reverse_index > 0; --reverse_index)
      reverseAttentionBlock(parameters, reverse_index - 1, workspace, observer);
    reverseEmbedding(workspace);
  }

  /// Map supported execution-plan boundaries to the geometry policy explicitly.
  static GeometryBoundary mappedBoundary(
      const qmcplusplus::psiformer::ExecutionEnvironment& environment)
  {
    if (environment.boundary == qmcplusplus::psiformer::BoundaryCondition::OPEN)
      return {GeometryBoundaryKind::OPEN};
    return {GeometryBoundaryKind::PERIODIC, environment.lattice_vectors,
            environment.periodic_axes};
  }

  /// Return the first scalar in one typed parameter tensor.
  const double* parameter(const double* values,
                          ParameterRole role,
                          std::size_t block = qmcplusplus::psiformer::NO_ATTENTION_BLOCK) const
  {
    return values + plan_.parameter(role, block).begin;
  }

  /// Return writable canonical score storage for one typed parameter tensor.
  double* score(DirectScoreTrunkWorkspace& workspace,
                ParameterRole role,
                std::size_t block = qmcplusplus::psiformer::NO_ATTENTION_BLOCK) const
  {
    return workspace.parameter_score_.data() + plan_.parameter(role, block).begin;
  }

  /// Reject a workspace from a different shape or a mutated parameter layout.
  void validateWorkspaceAndParameters(const DirectScoreTrunkWorkspace& workspace) const
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
  void buildEmbedding(const double* parameters, DirectScoreTrunkWorkspace& workspace) const
  {
    const GeometryPairTable& pairs = workspace.geometry_.electronNucleusPairs();
    const auto& displacements      = pairs.displacements();
    const auto& complementary      = pairs.complementaryDisplacements();
    const auto& radial_factors     = pairs.softenedRadialFactors();
    const bool periodic = workspace.geometry_.boundary().kind == GeometryBoundaryKind::PERIODIC;
    const std::size_t pair_width = periodic ? 7 : 4;
    for (std::size_t electron = 0; electron < workspace.electrons_; ++electron)
    {
      double* feature_row = workspace.raw_features_.data() + electron * workspace.input_width_;
      for (std::size_t nucleus = 0; nucleus < workspace.nuclei_; ++nucleus)
      {
        const std::size_t pair = electron * workspace.nuclei_ + nucleus;
        const std::size_t feature_begin = pair_width * nucleus;
        if (plan_.environment().periodic_profile)
        {
          const auto jet = qmcplusplus::psiformer::periodic_feature::evaluate(pairs, pair,
              workspace.geometry_.electronIonBoundary(), 1,
              plan_.environment().periodic_profile->feature_transform, 1);
          std::copy_n(jet.value.data(), 7, feature_row + feature_begin);
          continue;
        }
        feature_row[feature_begin] = radial_factors[pair].log1p_radius;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          feature_row[feature_begin + 1 + dimension] =
              displacements[pair][dimension] * radial_factors[pair].log1p_over_radius;
        if (periodic)
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
            feature_row[feature_begin + 4 + dimension] =
                complementary[pair][dimension] * radial_factors[pair].log1p_over_radius;
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
                           DirectScoreTrunkWorkspace& workspace) const
  {
    double* query = DirectScoreTrunkWorkspace::blockBuffer(workspace.queries_, block, workspace.feature_elements_);
    double* key   = DirectScoreTrunkWorkspace::blockBuffer(workspace.keys_, block, workspace.feature_elements_);
    double* value = DirectScoreTrunkWorkspace::blockBuffer(workspace.values_, block, workspace.feature_elements_);
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
        DirectScoreTrunkWorkspace::blockBuffer(workspace.attention_weights_, block, workspace.attention_elements_);
    double* context = DirectScoreTrunkWorkspace::blockBuffer(workspace.contexts_, block, workspace.feature_elements_);
    if (plan_.environment().periodic_profile)
      qmcplusplus::psiformer::periodic_feature::fillAttentionLogits(workspace.geometry_, spin_up_electrons_,
          plan_.environment().periodic_profile->feature_transform,
          parameter(parameters, ParameterRole::PAIR_ATTENTION_WEIGHT, block),
          parameter(parameters, ParameterRole::PAIR_ATTENTION_BIAS, block), workspace.heads_, attention);
    qmcplusplus::psiformer::direct::attentionWeightsForward(
        query, key, workspace.electrons_, workspace.heads_, workspace.head_width_, attention,
        static_cast<bool>(plan_.environment().periodic_profile));
    qmcplusplus::psiformer::direct::attentionContextForward(
        attention, value, workspace.electrons_, workspace.heads_, workspace.head_width_, context);

    double* residual = DirectScoreTrunkWorkspace::blockBuffer(workspace.residuals_, block, workspace.feature_elements_);
    qmcplusplus::psiformer::direct::denseForward(
        context, parameter(parameters, ParameterRole::ATTENTION_OUTPUT_WEIGHT, block), nullptr,
        workspace.electrons_, workspace.width_, workspace.width_, residual);
    for (std::size_t element = 0; element < workspace.feature_elements_; ++element)
      residual[element] += input[element];

    double* hidden = DirectScoreTrunkWorkspace::blockBuffer(workspace.hidden_, block, workspace.feature_elements_);
    qmcplusplus::psiformer::direct::denseForward(
        residual, parameter(parameters, ParameterRole::UPDATE_HIDDEN_WEIGHT, block),
        parameter(parameters, ParameterRole::UPDATE_HIDDEN_BIAS, block), workspace.electrons_,
        workspace.width_, workspace.width_, hidden);
    for (std::size_t element = 0; element < workspace.feature_elements_; ++element)
      hidden[element] = std::tanh(hidden[element]);

    double* update = DirectScoreTrunkWorkspace::blockBuffer(workspace.updates_, block, workspace.feature_elements_);
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

  /// Accumulate the cusp value and a scaled reverse seed for its scalar parameters.
  double cuspValueAndReverse(const double* parameters,
                             DirectScoreWorkspace& workspace,
                             double reverse_seed) const
  {
    if (plan_.environment().periodic_profile && !plan_.environment().periodic_profile->electron_electron_cusp)
      return 0;
    const bool has_same_alpha = plan_.hasParameter(ParameterRole::CUSP_SAME_ALPHA);
    const double same_alpha = has_same_alpha
        ? parameter(parameters, ParameterRole::CUSP_SAME_ALPHA)[0]
        : 1.0;
    const double anti_alpha = parameter(parameters, ParameterRole::CUSP_OPPOSITE_ALPHA)[0];
    double* same_score      = has_same_alpha
        ? score(workspace, ParameterRole::CUSP_SAME_ALPHA)
        : nullptr;
    double& anti_score      = score(workspace, ParameterRole::CUSP_OPPOSITE_ALPHA)[0];
    return realCuspValueAndReverse(workspace.geometry_, spin_up_electrons_, same_alpha,
        anti_alpha, same_score, anti_score, reverse_seed);
  }

  /// Seed determinant adjoints and reverse orbitals, backflow, and envelopes.
  void reverseOrbitals(const double* parameters,
                       DirectScoreWorkspace& workspace,
                       DirectAffineObservationSink* observer = nullptr) const
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
          if (observer)
            workspace.backflow_adjoints_[
                electron * workspace.determinants_ * workspace.electrons_ + channel] =
                backflow_adjoint;
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

    if (observer)
    {
      const std::size_t output_width = workspace.determinants_ * workspace.electrons_;
      observer->observe({ParameterRole::BACKFLOW_UP_WEIGHT,
                         qmcplusplus::psiformer::NO_ATTENTION_BLOCK,
                         final_features, workspace.backflow_adjoints_.data(),
                         spin_up_electrons_, workspace.width_, output_width});
      const std::size_t down_electrons = workspace.electrons_ - spin_up_electrons_;
      observer->observe({ParameterRole::BACKFLOW_DOWN_WEIGHT,
                         qmcplusplus::psiformer::NO_ATTENTION_BLOCK,
                         final_features + spin_up_electrons_ * workspace.width_,
                         workspace.backflow_adjoints_.data() +
                             spin_up_electrons_ * output_width,
                         down_electrons, workspace.width_, output_width});
    }
  }

  /// Reverse one attention/residual block into its input and eight parameter tensors.
  void reverseAttentionBlock(const double* parameters,
                             std::size_t block,
                             DirectScoreTrunkWorkspace& workspace,
                             DirectAffineObservationSink* observer = nullptr) const
  {
    const double* input = workspace.features(block);
    const double* query = DirectScoreTrunkWorkspace::blockBuffer(workspace.queries_, block,
                                                             workspace.feature_elements_);
    const double* key = DirectScoreTrunkWorkspace::blockBuffer(workspace.keys_, block,
                                                           workspace.feature_elements_);
    const double* value = DirectScoreTrunkWorkspace::blockBuffer(workspace.values_, block,
                                                             workspace.feature_elements_);
    const double* attention = DirectScoreTrunkWorkspace::blockBuffer(
        workspace.attention_weights_, block, workspace.attention_elements_);
    const double* context = DirectScoreTrunkWorkspace::blockBuffer(workspace.contexts_, block,
                                                               workspace.feature_elements_);
    const double* residual = DirectScoreTrunkWorkspace::blockBuffer(workspace.residuals_, block,
                                                                workspace.feature_elements_);
    const double* hidden = DirectScoreTrunkWorkspace::blockBuffer(workspace.hidden_, block,
                                                              workspace.feature_elements_);
    const double* update = DirectScoreTrunkWorkspace::blockBuffer(workspace.updates_, block,
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
    if (observer)
      observer->observe({ParameterRole::UPDATE_OUTPUT_WEIGHT, block, hidden,
                         workspace.update_adjoint_.data(), workspace.electrons_,
                         workspace.width_, workspace.width_});
    for (std::size_t element = 0; element < workspace.feature_elements_; ++element)
      workspace.hidden_adjoint_[element] *= 1.0 - hidden[element] * hidden[element];
    qmcplusplus::psiformer::direct::denseReverse(
        residual, parameter(parameters, ParameterRole::UPDATE_HIDDEN_WEIGHT, block),
        workspace.hidden_adjoint_.data(), workspace.electrons_, workspace.width_, workspace.width_,
        workspace.residual_adjoint_.data(), score(workspace, ParameterRole::UPDATE_HIDDEN_WEIGHT, block),
        score(workspace, ParameterRole::UPDATE_HIDDEN_BIAS, block));
    if (observer)
      observer->observe({ParameterRole::UPDATE_HIDDEN_WEIGHT, block, residual,
                         workspace.hidden_adjoint_.data(), workspace.electrons_,
                         workspace.width_, workspace.width_});

    // Reverse the output projection and seed the explicit attention residual branch.
    std::fill(workspace.context_adjoint_.begin(), workspace.context_adjoint_.end(), 0.0);
    qmcplusplus::psiformer::direct::denseReverse(
        context, parameter(parameters, ParameterRole::ATTENTION_OUTPUT_WEIGHT, block),
        workspace.residual_adjoint_.data(), workspace.electrons_, workspace.width_, workspace.width_,
        workspace.context_adjoint_.data(), score(workspace, ParameterRole::ATTENTION_OUTPUT_WEIGHT, block),
        nullptr);
    if (observer)
      observer->observe({ParameterRole::ATTENTION_OUTPUT_WEIGHT, block, context,
                         workspace.residual_adjoint_.data(), workspace.electrons_,
                         workspace.width_, workspace.width_});
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

    if (plan_.environment().periodic_profile)
    {
      auto* weight_score = score(workspace, ParameterRole::PAIR_ATTENTION_WEIGHT, block);
      auto* bias_score = score(workspace, ParameterRole::PAIR_ATTENTION_BIAS, block);
      const double inverse_scale = std::sqrt(static_cast<double>(workspace.head_width_));
      const auto n = workspace.electrons_, heads = workspace.heads_;
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t column = 0; column < n; ++column)
          if (row != column)
          {
            const auto ordered = row * (n - 1) + column - (column > row ? 1 : 0);
            for (std::size_t head = 0; head < heads; ++head)
            {
              const double sensitivity = inverse_scale * workspace.attention_adjoint_[(head * n + row) * n + column];
              workspace.pair_sensitivities_[ordered * heads + head] = sensitivity;
              bias_score[head] += sensitivity;
              for (std::size_t f = 0; f < 8; ++f)
                weight_score[f * heads + head] += workspace.pair_features_[8 * ordered + f] * sensitivity;
            }
          }
      if (observer)
        observer->observe({ParameterRole::PAIR_ATTENTION_WEIGHT, block, workspace.pair_features_.data(),
            workspace.pair_sensitivities_.data(), n * (n - 1), 8, heads});
    }

    if (observer)
    {
      observer->observe({ParameterRole::ATTENTION_QUERY_WEIGHT, block, input,
                         workspace.query_adjoint_.data(), workspace.electrons_,
                         workspace.width_, workspace.width_});
      observer->observe({ParameterRole::ATTENTION_KEY_WEIGHT, block, input,
                         workspace.key_adjoint_.data(), workspace.electrons_,
                         workspace.width_, workspace.width_});
      observer->observe({ParameterRole::ATTENTION_VALUE_WEIGHT, block, input,
                         workspace.value_adjoint_.data(), workspace.electrons_,
                         workspace.width_, workspace.width_});
    }

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
  void reverseEmbedding(DirectScoreTrunkWorkspace& workspace) const
  {
    double* embedding_score = score(workspace, ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
    for (std::size_t electron = 0; electron < workspace.electrons_; ++electron)
      for (std::size_t input = 0; input < workspace.input_width_; ++input)
        for (std::size_t output = 0; output < workspace.width_; ++output)
          embedding_score[input * workspace.width_ + output] +=
              workspace.raw_features_[electron * workspace.input_width_ + input] *
              workspace.feature_adjoint_a_[electron * workspace.width_ + output];
  }

  const DirectScorePreparation preparation_;
  const Parameters& parameters_;
  const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan_;
  std::vector<double> nuclei_;
  std::size_t spin_up_electrons_;
  GeometryBoundary boundary_;
};

} // namespace pf

#endif // QMCPLUSPLUS_PSIFORMER_SCORE_EXECUTOR_H
