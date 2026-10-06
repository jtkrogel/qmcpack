//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerKineticExecutor.h
 * @brief Exact graph-free PsiFormer score and component-kinetic parameter response.
 *
 * The forward tape carries a value, 3*Ne Cartesian first derivatives, and Ne
 * already-contracted Laplacians for every tensor element.  The reverse tape carries
 * adjoints of those same three quantities.  Consequently the mixed reverse computes
 *
 *   d log|Psi| / d theta
 *
 * and
 *
 *   -1/2 sum_e d_theta lap_e log|Psi|
 *   -sum_c G_total[c] d_theta grad_c log|Psi|
 *
 * exactly, without finite differences and without forming an electron Hessian.
 * Mutable tape and determinant factors are clone-local and fixed-size after workspace
 * construction.  The numerical implementation is deliberately real, open-boundary,
 * fixed-ion, and non-spinor.  Sign/log-magnitude results and explicit transpose
 * products preserve the API seam at which a later complex phase and adjoint convention
 * can be introduced.
 */

// PsiFormerNative.h currently contains non-inline definitions and must be included by
// exactly one translation unit.  This executor follows the other direct executors and
// is included after PsiFormerNative.h in that translation unit.
#ifndef QMCPLUSPLUS_PSIFORMER_NATIVE_H
#error "Include PsiFormerNative.h before PsiFormerKineticExecutor.h"
#endif

#ifndef QMCPLUSPLUS_PSIFORMER_KINETIC_EXECUTOR_H
#define QMCPLUSPLUS_PSIFORMER_KINETIC_EXECUTOR_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerGeometry.h"

#include "PsiFormerDenseKernels.h"
#include "PsiFormerDeterminant.h"
#include "PsiFormerStorageRequirements.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pf
{

/** Fixed-storage trace jet (or trace-jet adjoint).
 *
 * gradient is lane-major [3*Ne, value_size].  laplacian is electron-major
 * [Ne, value_size] and stores the Cartesian trace, never a Hessian.
 */
struct DirectTraceJetBuffer
{
  /// Create an empty trace-jet buffer.
  DirectTraceJetBuffer() = default;

  /// Allocate fixed value, Cartesian-gradient, and trace-Laplacian planes.
  DirectTraceJetBuffer(std::size_t value_size,
                       std::size_t gradient_lanes,
                       std::size_t laplacian_lanes)
      : value(value_size),
        gradient(value_size * gradient_lanes),
        laplacian(value_size * laplacian_lanes)
  {}

  /// Reset every value and derivative lane without changing storage.
  void clear() noexcept
  {
    std::fill(value.begin(), value.end(), 0.0);
    std::fill(gradient.begin(), gradient.end(), 0.0);
    std::fill(laplacian.begin(), laplacian.end(), 0.0);
  }

  /// Return the number of primal tensor elements.
  std::size_t valueSize() const noexcept { return value.size(); }

  std::vector<double> value;
  std::vector<double> gradient;
  std::vector<double> laplacian;
};

/// Read-only interval whose lifetime is tied to the owning kinetic workspace.
class DirectKineticConstView
{
public:
  /// Create an empty non-owning interval.
  DirectKineticConstView() = default;

  /// Bind a read-only interval to workspace-owned storage.
  DirectKineticConstView(const double* data, std::size_t size) : data_(data), size_(size) {}

  /// Return one interval element without bounds checking.
  const double& operator[](std::size_t index) const noexcept { return data_[index]; }

  /// Return the first storage address.
  const double* data() const noexcept { return data_; }

  /// Return an iterator to the first element.
  const double* begin() const noexcept { return data_; }

  /// Return an iterator one past the final element.
  const double* end() const noexcept { return size_ == 0 ? data_ : data_ + size_; }

  /// Return the number of visible elements.
  std::size_t size() const noexcept { return size_; }

  /// Report whether the interval contains no elements.
  bool empty() const noexcept { return size_ == 0; }

private:
  const double* data_ = nullptr;
  std::size_t size_   = 0;
};

/** Complete non-owning result from one warmed graph-free evaluation. */
struct DirectKineticResultView
{
  double sign   = 1.0;
  double logabs = 0.0;
  double value  = 0.0;
  DirectKineticConstView gradient;
  DirectKineticConstView lap_log;
  DirectKineticConstView lap_ratio;
  DirectKineticConstView parameter_score;
  DirectKineticConstView kinetic_parameter_response;
  std::size_t parameter_version = 0;
};

/** Clone-local, preallocated forward and lifted-reverse tape. */
class DirectKineticWorkspace
{
public:
  /// Allocate one clone-local fixed-capacity forward and reverse tape.
  DirectKineticWorkspace(const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan,
                         GeometryPositionView nuclei,
                         GeometryBoundary boundary = {})
      : electrons_(plan.modelShape().electrons()),
        nuclei_(plan.modelShape().nuclei),
        determinants_(plan.modelShape().determinants),
        width_(plan.modelShape().feature_dimension),
        heads_(plan.modelShape().attention_heads),
        blocks_(plan.modelShape().attention_blocks),
        head_width_(width_ / heads_),
        input_width_(plan.parameter(qmcplusplus::psiformer::ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0]),
        gradient_lanes_(3 * electrons_),
        laplacian_lanes_(electrons_),
        feature_elements_(electrons_ * width_),
        attention_elements_(heads_ * electrons_ * electrons_),
        orbital_elements_(determinants_ * electrons_ * electrons_),
        electron_positions_(3 * electrons_),
        geometry_(electrons_, nuclei, boundary),
        raw_features_(electrons_ * input_width_, gradient_lanes_, laplacian_lanes_),
        orbitals_(orbital_elements_, gradient_lanes_, laplacian_lanes_),
        backflows_(orbital_elements_, gradient_lanes_, laplacian_lanes_),
        envelopes_(orbital_elements_, gradient_lanes_, laplacian_lanes_),
        determinant_(determinants_, electrons_, gradient_lanes_, laplacian_lanes_),
        determinant_factor_(orbital_elements_, gradient_lanes_, laplacian_lanes_),
        orbital_adjoint_(orbital_elements_, gradient_lanes_, laplacian_lanes_),
        backflow_adjoint_(orbital_elements_, gradient_lanes_, laplacian_lanes_),
        envelope_adjoint_(orbital_elements_, gradient_lanes_, laplacian_lanes_),
        feature_adjoint_a_(feature_elements_, gradient_lanes_, laplacian_lanes_),
        feature_adjoint_b_(feature_elements_, gradient_lanes_, laplacian_lanes_),
        query_adjoint_(feature_elements_, gradient_lanes_, laplacian_lanes_),
        key_adjoint_(feature_elements_, gradient_lanes_, laplacian_lanes_),
        value_adjoint_(feature_elements_, gradient_lanes_, laplacian_lanes_),
        logit_adjoint_(attention_elements_, gradient_lanes_, laplacian_lanes_),
        attention_adjoint_(attention_elements_, gradient_lanes_, laplacian_lanes_),
        context_adjoint_(feature_elements_, gradient_lanes_, laplacian_lanes_),
        residual_adjoint_(feature_elements_, gradient_lanes_, laplacian_lanes_),
        hidden_adjoint_(feature_elements_, gradient_lanes_, laplacian_lanes_),
        hidden_pre_adjoint_(feature_elements_, gradient_lanes_, laplacian_lanes_),
        update_adjoint_(feature_elements_, gradient_lanes_, laplacian_lanes_),
        update_pre_adjoint_(feature_elements_, gradient_lanes_, laplacian_lanes_),
        row_exponential_(electrons_, gradient_lanes_, laplacian_lanes_),
        row_exponential_adjoint_(electrons_, gradient_lanes_, laplacian_lanes_),
        row_sum_(1, gradient_lanes_, laplacian_lanes_),
        row_reciprocal_(1, gradient_lanes_, laplacian_lanes_),
        row_sum_adjoint_(1, gradient_lanes_, laplacian_lanes_),
        row_reciprocal_adjoint_(1, gradient_lanes_, laplacian_lanes_),
        root_adjoint_(1, gradient_lanes_, laplacian_lanes_),
        determinant_gradient_(gradient_lanes_),
        determinant_lap_log_(laplacian_lanes_),
        determinant_lap_ratio_(laplacian_lanes_),
        cusp_gradient_(gradient_lanes_),
        cusp_laplacian_(laplacian_lanes_),
        output_gradient_(gradient_lanes_),
        output_lap_log_(laplacian_lanes_),
        output_lap_ratio_(laplacian_lanes_),
        parameter_score_(plan.parameterCount()),
        kinetic_parameter_response_(plan.parameterCount()),
        matrix_scratch_a_(electrons_ * electrons_),
        matrix_scratch_b_(electrons_ * electrons_),
        matrix_scratch_c_(electrons_ * electrons_)
  {
    if (electrons_ == 0 || nuclei_ == 0 || determinants_ == 0 || width_ == 0 ||
        heads_ == 0 || width_ % heads_ != 0)
      throw std::invalid_argument("PsiFormer kinetic workspace received inconsistent dimensions");

    features_.reserve(blocks_ + 1);
    queries_.reserve(blocks_);
    keys_.reserve(blocks_);
    values_.reserve(blocks_);
    logits_.reserve(blocks_);
    attention_.reserve(blocks_);
    contexts_.reserve(blocks_);
    residuals_.reserve(blocks_);
    hidden_pre_.reserve(blocks_);
    hidden_.reserve(blocks_);
    update_pre_.reserve(blocks_);
    updates_.reserve(blocks_);
    for (std::size_t boundary_index = 0; boundary_index <= blocks_; ++boundary_index)
      features_.emplace_back(feature_elements_, gradient_lanes_, laplacian_lanes_);
    for (std::size_t block = 0; block < blocks_; ++block)
    {
      queries_.emplace_back(feature_elements_, gradient_lanes_, laplacian_lanes_);
      keys_.emplace_back(feature_elements_, gradient_lanes_, laplacian_lanes_);
      values_.emplace_back(feature_elements_, gradient_lanes_, laplacian_lanes_);
      logits_.emplace_back(attention_elements_, gradient_lanes_, laplacian_lanes_);
      attention_.emplace_back(attention_elements_, gradient_lanes_, laplacian_lanes_);
      contexts_.emplace_back(feature_elements_, gradient_lanes_, laplacian_lanes_);
      residuals_.emplace_back(feature_elements_, gradient_lanes_, laplacian_lanes_);
      hidden_pre_.emplace_back(feature_elements_, gradient_lanes_, laplacian_lanes_);
      hidden_.emplace_back(feature_elements_, gradient_lanes_, laplacian_lanes_);
      update_pre_.emplace_back(feature_elements_, gradient_lanes_, laplacian_lanes_);
      updates_.emplace_back(feature_elements_, gradient_lanes_, laplacian_lanes_);
    }
  }

  /// Prevent copying the large clone-local tape.
  DirectKineticWorkspace(const DirectKineticWorkspace&) = delete;

  /// Prevent copy assignment of the large clone-local tape.
  DirectKineticWorkspace& operator=(const DirectKineticWorkspace&) = delete;

  /// Move a workspace while retaining all backing allocations.
  DirectKineticWorkspace(DirectKineticWorkspace&&) = default;

  /// Move-assign a workspace while retaining all backing allocations.
  DirectKineticWorkspace& operator=(DirectKineticWorkspace&&) = default;

  /// Replace every electron coordinate from an interleaved geometry view.
  void setPositions(GeometryPositionView positions)
  {
    if (positions.size() != electrons_)
      throw std::invalid_argument("PsiFormer kinetic workspace received the wrong electron count");
    for (std::size_t electron = 0; electron < electrons_; ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        electron_positions_[3 * electron + dimension] = positions(electron, dimension);
  }

  /// Replace one Cartesian electron coordinate.
  void setPosition(std::size_t electron, std::size_t dimension, double coordinate)
  {
    if (electron >= electrons_ || dimension >= 3)
      throw std::out_of_range("PsiFormer kinetic coordinate index is out of range");
    electron_positions_[3 * electron + dimension] = coordinate;
  }

  /// Return the parameter-store version consumed by the latest evaluation.
  std::size_t observedParameterVersion() const noexcept { return observed_parameter_version_; }

  /// Return the stable backing address of the canonical score vector.
  const double* scoreData() const noexcept { return parameter_score_.data(); }

  /// Return the stable backing address of the canonical kinetic-response vector.
  const double* kineticData() const noexcept { return kinetic_parameter_response_.data(); }

  /// Return the fixed number of canonical score components.
  std::size_t scoreSize() const noexcept { return parameter_score_.size(); }

  /// Return the fixed number of canonical kinetic-response components.
  std::size_t kineticSize() const noexcept
  {
    return kinetic_parameter_response_.size();
  }

  /// Return bytes retained by the two canonical parameter-response vectors.
  std::size_t parameterStorageBytes() const
  {
    std::size_t bytes = checkedStorageBytes<double>(
        parameter_score_.capacity(),
        "PsiFormer kinetic score-vector bytes overflowed");
    addStorageBytes(
        bytes,
        checkedStorageBytes<double>(
            kinetic_parameter_response_.capacity(),
            "PsiFormer kinetic response-vector bytes overflowed"),
        "PsiFormer kinetic parameter-vector bytes overflowed");
    return bytes;
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
    auto mix_jet = [&mix](const DirectTraceJetBuffer& jet) {
      mix(jet.value);
      mix(jet.gradient);
      mix(jet.laplacian);
    };
    mix(electron_positions_);
    mix_jet(raw_features_);
    for (const auto& jet : features_)
      mix_jet(jet);
    auto mix_tape = [&mix_jet](const std::vector<DirectTraceJetBuffer>& tape) {
      for (const auto& jet : tape)
        mix_jet(jet);
    };
    mix_tape(queries_);
    mix_tape(keys_);
    mix_tape(values_);
    mix_tape(logits_);
    mix_tape(attention_);
    mix_tape(contexts_);
    mix_tape(residuals_);
    mix_tape(hidden_pre_);
    mix_tape(hidden_);
    mix_tape(update_pre_);
    mix_tape(updates_);
    for (const DirectTraceJetBuffer* jet : {&orbitals_, &backflows_, &envelopes_,
                                            &determinant_factor_, &orbital_adjoint_,
                                            &backflow_adjoint_, &envelope_adjoint_,
                                            &feature_adjoint_a_, &feature_adjoint_b_,
                                            &query_adjoint_, &key_adjoint_, &value_adjoint_,
                                            &logit_adjoint_, &attention_adjoint_,
                                            &context_adjoint_, &residual_adjoint_,
                                            &hidden_adjoint_, &hidden_pre_adjoint_,
                                            &update_adjoint_, &update_pre_adjoint_,
                                            &row_exponential_, &row_exponential_adjoint_,
                                            &row_sum_, &row_reciprocal_, &row_sum_adjoint_,
                                            &row_reciprocal_adjoint_, &root_adjoint_})
      mix_jet(*jet);
    for (const auto* buffer : {&determinant_gradient_, &determinant_lap_log_,
                               &determinant_lap_ratio_, &cusp_gradient_, &cusp_laplacian_,
                               &output_gradient_, &output_lap_log_, &output_lap_ratio_,
                               &parameter_score_, &kinetic_parameter_response_,
                               &matrix_scratch_a_, &matrix_scratch_b_, &matrix_scratch_c_})
      mix(*buffer);
    hash ^= geometry_.storageFingerprint();
    hash *= 1099511628211ULL;
    hash ^= determinant_.storageFingerprint();
    hash *= 1099511628211ULL;
    return hash;
  }

  /// Bytes reserved by the complete tape, geometry, and determinant workspace.
  std::size_t vectorStorageBytes() const
  {
    std::size_t bytes = 0;
    auto add = [&bytes](const auto& buffer) {
      using Element = typename std::decay_t<decltype(buffer)>::value_type;
      addStorageBytes(bytes,
                      checkedStorageBytes<Element>(
                          buffer.capacity(),
                          "PsiFormer kinetic storage bytes overflowed"),
                      "PsiFormer kinetic storage bytes overflowed");
    };
    auto add_jet = [&add](const DirectTraceJetBuffer& jet) {
      add(jet.value);
      add(jet.gradient);
      add(jet.laplacian);
    };
    add(electron_positions_);
    add_jet(raw_features_);
    for (const auto& jet : features_)
      add_jet(jet);
    auto add_tape = [&add_jet](const std::vector<DirectTraceJetBuffer>& tape) {
      for (const auto& jet : tape)
        add_jet(jet);
    };
    add_tape(queries_);
    add_tape(keys_);
    add_tape(values_);
    add_tape(logits_);
    add_tape(attention_);
    add_tape(contexts_);
    add_tape(residuals_);
    add_tape(hidden_pre_);
    add_tape(hidden_);
    add_tape(update_pre_);
    add_tape(updates_);
    for (const DirectTraceJetBuffer* jet : {&orbitals_, &backflows_, &envelopes_,
                                            &determinant_factor_, &orbital_adjoint_,
                                            &backflow_adjoint_, &envelope_adjoint_,
                                            &feature_adjoint_a_, &feature_adjoint_b_,
                                            &query_adjoint_, &key_adjoint_, &value_adjoint_,
                                            &logit_adjoint_, &attention_adjoint_,
                                            &context_adjoint_, &residual_adjoint_,
                                            &hidden_adjoint_, &hidden_pre_adjoint_,
                                            &update_adjoint_, &update_pre_adjoint_,
                                            &row_exponential_, &row_exponential_adjoint_,
                                            &row_sum_, &row_reciprocal_, &row_sum_adjoint_,
                                            &row_reciprocal_adjoint_, &root_adjoint_})
      add_jet(*jet);
    for (const auto* buffer : {&determinant_gradient_, &determinant_lap_log_,
                               &determinant_lap_ratio_, &cusp_gradient_, &cusp_laplacian_,
                               &output_gradient_, &output_lap_log_, &output_lap_ratio_,
                               &parameter_score_, &kinetic_parameter_response_,
                               &matrix_scratch_a_, &matrix_scratch_b_, &matrix_scratch_c_})
      add(*buffer);
    addStorageBytes(bytes, geometry_.storageBytes(),
                    "PsiFormer kinetic storage bytes overflowed");
    addStorageBytes(bytes, determinant_.storageBytes(),
                    "PsiFormer kinetic storage bytes overflowed");
    return bytes;
  }

  /// Expose the geometry contribution for focused inclusive-accounting tests.
  std::size_t geometryStorageBytes() const
  { return geometry_.storageBytes(); }

  /// Return the checked constructor-time requirement for this fixed model shape.
  std::size_t requiredStorageBytes() const
  {
    return kineticWorkspaceStorageRequirement(
        {electrons_, nuclei_, determinants_, width_, heads_, input_width_,
         blocks_, parameter_score_.size()});
  }

private:
  friend class DirectKineticExecutor;

  std::size_t electrons_;
  std::size_t nuclei_;
  std::size_t determinants_;
  std::size_t width_;
  std::size_t heads_;
  std::size_t blocks_;
  std::size_t head_width_;
  std::size_t input_width_;
  std::size_t gradient_lanes_;
  std::size_t laplacian_lanes_;
  std::size_t feature_elements_;
  std::size_t attention_elements_;
  std::size_t orbital_elements_;

  std::vector<double> electron_positions_;
  PsiFormerGeometryCache geometry_;
  DirectTraceJetBuffer raw_features_;
  std::vector<DirectTraceJetBuffer> features_;
  std::vector<DirectTraceJetBuffer> queries_;
  std::vector<DirectTraceJetBuffer> keys_;
  std::vector<DirectTraceJetBuffer> values_;
  std::vector<DirectTraceJetBuffer> logits_;
  std::vector<DirectTraceJetBuffer> attention_;
  std::vector<DirectTraceJetBuffer> contexts_;
  std::vector<DirectTraceJetBuffer> residuals_;
  std::vector<DirectTraceJetBuffer> hidden_pre_;
  std::vector<DirectTraceJetBuffer> hidden_;
  std::vector<DirectTraceJetBuffer> update_pre_;
  std::vector<DirectTraceJetBuffer> updates_;
  DirectTraceJetBuffer orbitals_;
  DirectTraceJetBuffer backflows_;
  DirectTraceJetBuffer envelopes_;

  qmcplusplus::psiformer::determinant::RealOpenDeterminantWorkspace determinant_;
  DirectTraceJetBuffer determinant_factor_;
  DirectTraceJetBuffer orbital_adjoint_;
  DirectTraceJetBuffer backflow_adjoint_;
  DirectTraceJetBuffer envelope_adjoint_;

  DirectTraceJetBuffer feature_adjoint_a_;
  DirectTraceJetBuffer feature_adjoint_b_;
  DirectTraceJetBuffer query_adjoint_;
  DirectTraceJetBuffer key_adjoint_;
  DirectTraceJetBuffer value_adjoint_;
  DirectTraceJetBuffer logit_adjoint_;
  DirectTraceJetBuffer attention_adjoint_;
  DirectTraceJetBuffer context_adjoint_;
  DirectTraceJetBuffer residual_adjoint_;
  DirectTraceJetBuffer hidden_adjoint_;
  DirectTraceJetBuffer hidden_pre_adjoint_;
  DirectTraceJetBuffer update_adjoint_;
  DirectTraceJetBuffer update_pre_adjoint_;

  DirectTraceJetBuffer row_exponential_;
  DirectTraceJetBuffer row_exponential_adjoint_;
  DirectTraceJetBuffer row_sum_;
  DirectTraceJetBuffer row_reciprocal_;
  DirectTraceJetBuffer row_sum_adjoint_;
  DirectTraceJetBuffer row_reciprocal_adjoint_;
  DirectTraceJetBuffer root_adjoint_;

  std::vector<double> determinant_gradient_;
  std::vector<double> determinant_lap_log_;
  std::vector<double> determinant_lap_ratio_;
  std::vector<double> cusp_gradient_;
  std::vector<double> cusp_laplacian_;
  std::vector<double> output_gradient_;
  std::vector<double> output_lap_log_;
  std::vector<double> output_lap_ratio_;
  std::vector<double> parameter_score_;
  std::vector<double> kinetic_parameter_response_;
  std::vector<double> matrix_scratch_a_;
  std::vector<double> matrix_scratch_b_;
  std::vector<double> matrix_scratch_c_;
  std::size_t observed_parameter_version_ = std::numeric_limits<std::size_t>::max();
};

/** Exact direct trace-jet forward plus lifted parameter reverse. */
class DirectKineticExecutor
{
public:
  /// Bind immutable model metadata, parameters, and fixed nuclear positions.
  DirectKineticExecutor(const PsiFormer& model,
                        const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan)
      : parameters_(model.p),
        plan_(plan),
        nuclei_(model.cfg.nuclei.x),
        spin_up_electrons_(model.cfg.nup),
        boundary_(mappedBoundary(plan.environment()))
  {
    const auto& shape = plan_.modelShape();
    if (plan_.parameterCount() != parameters_.size() || shape.electrons() != model.ne ||
        shape.nuclei != model.cfg.nuclei.shape[0] || shape.determinants != model.ndet ||
        shape.feature_dimension != model.dim || shape.attention_heads != model.heads)
      throw std::invalid_argument("PsiFormer kinetic execution plan does not match the model");
    using qmcplusplus::psiformer::BoundaryCondition;
    using qmcplusplus::psiformer::ScalarDomain;
    const auto& environment = plan_.environment();
    if (environment.parameter_scalar_domain != ScalarDomain::REAL ||
        environment.compute_scalar_domain != ScalarDomain::REAL ||
        environment.amplitude_scalar_domain != ScalarDomain::REAL || !environment.fixed_nuclei)
      throw std::invalid_argument(
          "Direct PsiFormer kinetic response supports only real, fixed-ion models");
  }

  /// Allocate one independent fixed-capacity workspace for a component clone.
  std::unique_ptr<DirectKineticWorkspace> makeWorkspace() const
  {
    return std::make_unique<DirectKineticWorkspace>(
        plan_, GeometryPositionView::interleaved(nuclei_.data(), plan_.modelShape().nuclei), boundary_);
  }

  /** Evaluate score and kinetic response.
   *
   * total_log_gradient is the complete TrialWaveFunction drift.  Passing null uses
   * this PsiFormer component's drift, matching standalone-native behavior.
   */
  DirectKineticResultView evaluate(DirectKineticWorkspace& workspace,
                                   const double* total_log_gradient = nullptr,
                                   std::size_t total_log_gradient_size = 0) const;

  /** Evaluate with explicit positive inverse masses for each electron.
   *
   * This overload preserves the unit-mass entry point above while allowing the
   * streaming local-energy route to match the kinetic Hamiltonian exactly.
   */
  DirectKineticResultView evaluate(DirectKineticWorkspace& workspace,
                                   const double* total_log_gradient,
                                   std::size_t total_log_gradient_size,
                                   const double* inverse_masses,
                                   std::size_t inverse_mass_count) const;

private:
  using ParameterRole = qmcplusplus::psiformer::ParameterRole;

  /// Translate the execution-plan boundary tag to the geometry-kernel boundary tag.
  static GeometryBoundary mappedBoundary(
      const qmcplusplus::psiformer::ExecutionEnvironment& environment)
  {
    if (environment.boundary == qmcplusplus::psiformer::BoundaryCondition::OPEN)
      return {GeometryBoundaryKind::OPEN};
    return {GeometryBoundaryKind::PERIODIC, environment.lattice_vectors,
            environment.periodic_axes};
  }

  /// Locate one typed tensor in the immutable canonical parameter vector.
  const double* parameter(const double* values,
                          ParameterRole role,
                          std::size_t block = qmcplusplus::psiformer::NO_ATTENTION_BLOCK) const
  {
    return values + plan_.parameter(role, block).begin;
  }

  /// Locate one typed tensor in a canonical response vector.
  double* response(std::vector<double>& destination,
                   ParameterRole role,
                   std::size_t block = qmcplusplus::psiformer::NO_ATTENTION_BLOCK) const
  {
    return destination.data() + plan_.parameter(role, block).begin;
  }

  /// Map a Cartesian derivative lane and tensor element to lane-major storage.
  static std::size_t gradientIndex(const DirectTraceJetBuffer& buffer,
                                   std::size_t lane,
                                   std::size_t element) noexcept
  {
    return lane * buffer.value.size() + element;
  }

  /// Map an electron trace lane and tensor element to lane-major storage.
  static std::size_t laplacianIndex(const DirectTraceJetBuffer& buffer,
                                    std::size_t electron,
                                    std::size_t element) noexcept
  {
    return electron * buffer.value.size() + element;
  }

  /// Accumulate all primal and derivative planes of one trace jet.
  static void addJet(const DirectTraceJetBuffer& source, DirectTraceJetBuffer& target);

  /// Copy all primal and derivative planes of one trace jet.
  static void copyJet(const DirectTraceJetBuffer& source, DirectTraceJetBuffer& target);

  /// Apply a row-wise affine map to every trace-jet plane.
  static void denseForward(const DirectTraceJetBuffer& source,
                           const double* weight,
                           const double* bias,
                           std::size_t rows,
                           std::size_t input_width,
                           std::size_t output_width,
                           std::size_t gradient_lanes,
                           std::size_t laplacian_lanes,
                           DirectTraceJetBuffer& target);

  /// Reverse a row-wise affine trace-jet map into input and parameter adjoints.
  static void denseReverse(const DirectTraceJetBuffer& source,
                           const double* weight,
                           const DirectTraceJetBuffer& target_adjoint,
                           std::size_t rows,
                           std::size_t input_width,
                           std::size_t output_width,
                           std::size_t gradient_lanes,
                           std::size_t laplacian_lanes,
                           DirectTraceJetBuffer& source_adjoint,
                           double* weight_adjoint,
                           double* bias_adjoint);

  /// Apply tanh with its Cartesian and contracted-Laplacian chain rules.
  static void tanhForward(const DirectTraceJetBuffer& input,
                          std::size_t gradient_lanes,
                          std::size_t laplacian_lanes,
                          DirectTraceJetBuffer& output);

  /// Reverse tanh through value, gradient, and Laplacian planes.
  static void tanhReverse(const DirectTraceJetBuffer& input,
                          const DirectTraceJetBuffer& output,
                          const DirectTraceJetBuffer& output_adjoint,
                          std::size_t gradient_lanes,
                          std::size_t laplacian_lanes,
                          DirectTraceJetBuffer& input_adjoint);

  /// Form one scaled bilinear product trace jet.
  static void productForwardElement(const DirectTraceJetBuffer& left,
                                    std::size_t left_element,
                                    const DirectTraceJetBuffer& right,
                                    std::size_t right_element,
                                    std::size_t gradient_lanes,
                                    std::size_t laplacian_lanes,
                                    DirectTraceJetBuffer& output,
                                    std::size_t output_element,
                                    double scale = 1.0);

  /// Reverse one scaled bilinear product into both operand trace jets.
  static void productReverseElement(const DirectTraceJetBuffer& left,
                                    std::size_t left_element,
                                    const DirectTraceJetBuffer& right,
                                    std::size_t right_element,
                                    const DirectTraceJetBuffer& output_adjoint,
                                    std::size_t output_element,
                                    std::size_t gradient_lanes,
                                    std::size_t laplacian_lanes,
                                    DirectTraceJetBuffer& left_adjoint,
                                    DirectTraceJetBuffer& right_adjoint,
                                    double scale = 1.0);

  /// Reject a workspace whose dimensions or geometry identity do not match this executor.
  void validateWorkspace(const DirectKineticWorkspace& workspace) const;

  /// Build electron-nucleus input features and the first embedded feature tensor.
  void buildEmbedding(const double* parameters, DirectKineticWorkspace& workspace) const;

  /// Build scaled query-key logits for one attention block.
  void buildAttentionLogits(std::size_t block, DirectKineticWorkspace& workspace) const;

  /// Normalize one block's attention rows with stable trace-jet softmax.
  void softmaxForward(std::size_t block, DirectKineticWorkspace& workspace) const;

  /// Contract attention probabilities with value features.
  void buildAttentionContext(std::size_t block, DirectKineticWorkspace& workspace) const;

  /// Evaluate one complete attention/update/residual block.
  void applyAttentionBlock(const double* parameters,
                           std::size_t block,
                           DirectKineticWorkspace& workspace) const;

  /// Build backflow, envelope, and determinant-orbital trace jets.
  void buildOrbitals(const double* parameters, DirectKineticWorkspace& workspace) const;

  /// Build the analytic electron-pair cusp value and coordinate traces.
  double buildCusp(const double* parameters, DirectKineticWorkspace& workspace) const;

  /// Build coordinate jets of d log|sum det|/dA from cached determinant factors.
  void buildDeterminantFactors(DirectKineticWorkspace& workspace) const;

  /// Seed orbital adjoints from the determinant root trace jet.
  void reverseDeterminant(DirectKineticWorkspace& workspace) const;

  /// Reverse orbital backflow/envelope construction into features and parameters.
  void reverseOrbitals(const double* parameters,
                       DirectKineticWorkspace& workspace,
                       std::vector<double>& destination) const;

  /// Reverse stable row softmax by replaying one fixed-size row sub-tape.
  void softmaxReverse(std::size_t block, DirectKineticWorkspace& workspace) const;

  /// Reverse query-key logits into query and key trace jets.
  void reverseAttentionLogits(std::size_t block, DirectKineticWorkspace& workspace) const;

  /// Reverse attention-value contraction into probabilities and values.
  void reverseAttentionContext(std::size_t block, DirectKineticWorkspace& workspace) const;

  /// Reverse one attention/update/residual block into its input and parameters.
  void reverseAttentionBlock(const double* parameters,
                             std::size_t block,
                             DirectKineticWorkspace& workspace,
                             std::vector<double>& destination) const;

  /// Reverse the initial feature embedding into its weight response.
  void reverseEmbedding(DirectKineticWorkspace& workspace,
                        std::vector<double>& destination) const;

  /// Reverse the analytic cusp into its two scalar parameter responses.
  void reverseCusp(const double* parameters,
                   DirectKineticWorkspace& workspace,
                   std::vector<double>& destination) const;

  /// Run a complete lifted reverse for the currently seeded root adjoint.
  void reverse(const double* parameters,
               DirectKineticWorkspace& workspace,
               std::vector<double>& destination) const;

  /// Multiply two square row-major matrices into caller-owned scratch.
  static void matrixProduct(const double* left,
                            const double* right,
                            double* output,
                            std::size_t size) noexcept;

  /// Return tr(left*right) for two square row-major matrices.
  static double traceProduct(const double* left,
                             const double* right,
                             std::size_t size) noexcept;

  const Parameters& parameters_;
  const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan_;
  std::vector<double> nuclei_;
  std::size_t spin_up_electrons_;
  GeometryBoundary boundary_;
};

/// Accumulate one complete trace jet into another fixed-size trace jet.
inline void DirectKineticExecutor::addJet(const DirectTraceJetBuffer& source,
                                          DirectTraceJetBuffer& target)
{
  if (source.value.size() != target.value.size() ||
      source.gradient.size() != target.gradient.size() ||
      source.laplacian.size() != target.laplacian.size())
    throw std::invalid_argument("PsiFormer trace-jet add received incompatible buffers");
  for (std::size_t element = 0; element < source.value.size(); ++element)
    target.value[element] += source.value[element];
  for (std::size_t element = 0; element < source.gradient.size(); ++element)
    target.gradient[element] += source.gradient[element];
  for (std::size_t element = 0; element < source.laplacian.size(); ++element)
    target.laplacian[element] += source.laplacian[element];
}

/// Copy one complete trace jet without changing either buffer's capacity.
inline void DirectKineticExecutor::copyJet(const DirectTraceJetBuffer& source,
                                           DirectTraceJetBuffer& target)
{
  if (source.value.size() != target.value.size() ||
      source.gradient.size() != target.gradient.size() ||
      source.laplacian.size() != target.laplacian.size())
    throw std::invalid_argument("PsiFormer trace-jet copy received incompatible buffers");
  std::copy(source.value.begin(), source.value.end(), target.value.begin());
  std::copy(source.gradient.begin(), source.gradient.end(), target.gradient.begin());
  std::copy(source.laplacian.begin(), source.laplacian.end(), target.laplacian.begin());
}

/// Evaluate a dense affine layer on stacked primal, gradient, and trace planes.
inline void DirectKineticExecutor::denseForward(const DirectTraceJetBuffer& source,
                                                const double* weight,
                                                const double* bias,
                                                std::size_t rows,
                                                std::size_t input_width,
                                                std::size_t output_width,
                                                std::size_t gradient_lanes,
                                                std::size_t laplacian_lanes,
                                                DirectTraceJetBuffer& target)
{
  if (!weight || source.value.size() != rows * input_width ||
      target.value.size() != rows * output_width)
    throw std::invalid_argument("PsiFormer dense trace-jet dimensions are inconsistent");

  // Every lane is a contiguous row-major matrix with the same right operand.  Treat
  // all lanes of one derivative kind as a taller matrix so each kind needs one BLAS
  // call rather than one scalar triply nested loop per lane.  Each product overwrites
  // its complete destination plane, so no preliminary zero-fill is needed.
  qmcplusplus::psiformer::dense::productReal(source.value.data(), weight, bias, rows,
                                              input_width, output_width, target.value.data());
  qmcplusplus::psiformer::dense::productReal(
      source.gradient.data(), weight, nullptr, gradient_lanes * rows, input_width,
      output_width, target.gradient.data());
  qmcplusplus::psiformer::dense::productReal(
      source.laplacian.data(), weight, nullptr, laplacian_lanes * rows, input_width,
      output_width, target.laplacian.data());
}

/// Reverse stacked dense planes into input, weight, and bias adjoints.
inline void DirectKineticExecutor::denseReverse(const DirectTraceJetBuffer& source,
                                                const double* weight,
                                                const DirectTraceJetBuffer& target_adjoint,
                                                std::size_t rows,
                                                std::size_t input_width,
                                                std::size_t output_width,
                                                std::size_t gradient_lanes,
                                                std::size_t laplacian_lanes,
                                                DirectTraceJetBuffer& source_adjoint,
                                                double* weight_adjoint,
                                                double* bias_adjoint)
{
  if (!weight || !weight_adjoint || source.value.size() != rows * input_width ||
      target_adjoint.value.size() != rows * output_width ||
      source_adjoint.value.size() != source.value.size())
    throw std::invalid_argument("PsiFormer dense lifted reverse dimensions are inconsistent");

  // Input and weight adjoints are additive: query, key, value, and residual paths
  // can converge on the same storage.  The BLAS helpers therefore use beta=1 and
  // preserve the caller's previously accumulated contribution.
  qmcplusplus::psiformer::dense::accumulateInputAdjointReal(
      target_adjoint.value.data(), weight, rows, input_width, output_width,
      source_adjoint.value.data());
  qmcplusplus::psiformer::dense::accumulateWeightAdjointReal(
      source.value.data(), target_adjoint.value.data(), rows, input_width, output_width,
      weight_adjoint);

  qmcplusplus::psiformer::dense::accumulateInputAdjointReal(
      target_adjoint.gradient.data(), weight, gradient_lanes * rows, input_width,
      output_width, source_adjoint.gradient.data());
  qmcplusplus::psiformer::dense::accumulateWeightAdjointReal(
      source.gradient.data(), target_adjoint.gradient.data(), gradient_lanes * rows,
      input_width, output_width, weight_adjoint);

  qmcplusplus::psiformer::dense::accumulateInputAdjointReal(
      target_adjoint.laplacian.data(), weight, laplacian_lanes * rows, input_width,
      output_width, source_adjoint.laplacian.data());
  qmcplusplus::psiformer::dense::accumulateWeightAdjointReal(
      source.laplacian.data(), target_adjoint.laplacian.data(), laplacian_lanes * rows,
      input_width, output_width, weight_adjoint);

  // A coordinate-independent bias contributes only to the primal output plane.
  if (bias_adjoint)
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t output = 0; output < output_width; ++output)
        bias_adjoint[output] += target_adjoint.value[row * output_width + output];
}

/// Evaluate tanh and its first and contracted second coordinate derivatives.
inline void DirectKineticExecutor::tanhForward(const DirectTraceJetBuffer& input,
                                               std::size_t gradient_lanes,
                                               std::size_t laplacian_lanes,
                                               DirectTraceJetBuffer& output)
{
  if (input.value.size() != output.value.size())
    throw std::invalid_argument("PsiFormer tanh trace-jet dimensions are inconsistent");
  output.clear();
  for (std::size_t element = 0; element < input.value.size(); ++element)
  {
    const double result = std::tanh(input.value[element]);
    const double first  = 1.0 - result * result;
    const double second = -2.0 * result * first;
    output.value[element] = result;
    for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
      output.gradient[gradientIndex(output, lane, element)] =
          first * input.gradient[gradientIndex(input, lane, element)];
    for (std::size_t electron = 0; electron < laplacian_lanes; ++electron)
    {
      double squared_gradient = 0.0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double component =
            input.gradient[gradientIndex(input, 3 * electron + dimension, element)];
        squared_gradient += component * component;
      }
      output.laplacian[laplacianIndex(output, electron, element)] =
          first * input.laplacian[laplacianIndex(input, electron, element)] +
          second * squared_gradient;
    }
  }
}

/// Reverse tanh using derivatives through third order for Laplacian adjoints.
inline void DirectKineticExecutor::tanhReverse(const DirectTraceJetBuffer& input,
                                               const DirectTraceJetBuffer& output,
                                               const DirectTraceJetBuffer& output_adjoint,
                                               std::size_t gradient_lanes,
                                               std::size_t laplacian_lanes,
                                               DirectTraceJetBuffer& input_adjoint)
{
  input_adjoint.clear();
  for (std::size_t element = 0; element < input.value.size(); ++element)
  {
    const double result = output.value[element];
    const double first  = 1.0 - result * result;
    const double second = -2.0 * result * first;
    const double third  = -2.0 * first * (1.0 - 3.0 * result * result);
    double value_adjoint = output_adjoint.value[element] * first;

    for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
    {
      const std::size_t index = gradientIndex(input, lane, element);
      const double input_gradient = input.gradient[index];
      const double gradient_upstream = output_adjoint.gradient[index];
      const std::size_t electron = lane / 3;
      const double laplacian_upstream =
          output_adjoint.laplacian[laplacianIndex(output_adjoint, electron, element)];
      value_adjoint += gradient_upstream * second * input_gradient;
      input_adjoint.gradient[index] =
          gradient_upstream * first + 2.0 * laplacian_upstream * second * input_gradient;
    }

    for (std::size_t electron = 0; electron < laplacian_lanes; ++electron)
    {
      const std::size_t index = laplacianIndex(input, electron, element);
      const double laplacian_upstream = output_adjoint.laplacian[index];
      double squared_gradient = 0.0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double component =
            input.gradient[gradientIndex(input, 3 * electron + dimension, element)];
        squared_gradient += component * component;
      }
      value_adjoint += laplacian_upstream *
          (second * input.laplacian[index] + third * squared_gradient);
      input_adjoint.laplacian[index] = laplacian_upstream * first;
    }
    input_adjoint.value[element] = value_adjoint;
  }
}

/// Evaluate one scaled product with the exact trace product rule.
inline void DirectKineticExecutor::productForwardElement(
    const DirectTraceJetBuffer& left,
    std::size_t left_element,
    const DirectTraceJetBuffer& right,
    std::size_t right_element,
    std::size_t gradient_lanes,
    std::size_t laplacian_lanes,
    DirectTraceJetBuffer& output,
    std::size_t output_element,
    double scale)
{
  const double left_value  = left.value[left_element];
  const double right_value = right.value[right_element];
  output.value[output_element] += scale * left_value * right_value;
  for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
  {
    const double left_gradient = left.gradient[gradientIndex(left, lane, left_element)];
    const double right_gradient = right.gradient[gradientIndex(right, lane, right_element)];
    output.gradient[gradientIndex(output, lane, output_element)] +=
        scale * (left_gradient * right_value + left_value * right_gradient);
  }
  for (std::size_t electron = 0; electron < laplacian_lanes; ++electron)
  {
    double gradient_dot = 0.0;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      gradient_dot +=
          left.gradient[gradientIndex(left, 3 * electron + dimension, left_element)] *
          right.gradient[gradientIndex(right, 3 * electron + dimension, right_element)];
    output.laplacian[laplacianIndex(output, electron, output_element)] += scale *
        (left.laplacian[laplacianIndex(left, electron, left_element)] * right_value +
         2.0 * gradient_dot + left_value *
             right.laplacian[laplacianIndex(right, electron, right_element)]);
  }
}

/// Reverse one scaled product through every trace-jet plane.
inline void DirectKineticExecutor::productReverseElement(
    const DirectTraceJetBuffer& left,
    std::size_t left_element,
    const DirectTraceJetBuffer& right,
    std::size_t right_element,
    const DirectTraceJetBuffer& output_adjoint,
    std::size_t output_element,
    std::size_t gradient_lanes,
    std::size_t laplacian_lanes,
    DirectTraceJetBuffer& left_adjoint,
    DirectTraceJetBuffer& right_adjoint,
    double scale)
{
  const double left_value  = left.value[left_element];
  const double right_value = right.value[right_element];
  const double value_upstream = scale * output_adjoint.value[output_element];
  double left_value_adjoint  = value_upstream * right_value;
  double right_value_adjoint = value_upstream * left_value;

  for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
  {
    const std::size_t electron = lane / 3;
    const double gradient_upstream =
        scale * output_adjoint.gradient[gradientIndex(output_adjoint, lane, output_element)];
    const double laplacian_upstream =
        scale * output_adjoint.laplacian[laplacianIndex(output_adjoint, electron, output_element)];
    const double left_gradient = left.gradient[gradientIndex(left, lane, left_element)];
    const double right_gradient = right.gradient[gradientIndex(right, lane, right_element)];
    left_value_adjoint += gradient_upstream * right_gradient;
    right_value_adjoint += gradient_upstream * left_gradient;
    left_adjoint.gradient[gradientIndex(left_adjoint, lane, left_element)] +=
        gradient_upstream * right_value + 2.0 * laplacian_upstream * right_gradient;
    right_adjoint.gradient[gradientIndex(right_adjoint, lane, right_element)] +=
        gradient_upstream * left_value + 2.0 * laplacian_upstream * left_gradient;
  }

  for (std::size_t electron = 0; electron < laplacian_lanes; ++electron)
  {
    const double laplacian_upstream =
        scale * output_adjoint.laplacian[laplacianIndex(output_adjoint, electron, output_element)];
    left_value_adjoint += laplacian_upstream *
        right.laplacian[laplacianIndex(right, electron, right_element)];
    right_value_adjoint += laplacian_upstream *
        left.laplacian[laplacianIndex(left, electron, left_element)];
    left_adjoint.laplacian[laplacianIndex(left_adjoint, electron, left_element)] +=
        laplacian_upstream * right_value;
    right_adjoint.laplacian[laplacianIndex(right_adjoint, electron, right_element)] +=
        laplacian_upstream * left_value;
  }
  left_adjoint.value[left_element] += left_value_adjoint;
  right_adjoint.value[right_element] += right_value_adjoint;
}

/// Validate workspace dimensions before any caller-visible output is changed.
inline void DirectKineticExecutor::validateWorkspace(
    const DirectKineticWorkspace& workspace) const
{
  const auto& shape = plan_.modelShape();
  if (workspace.electrons_ != shape.electrons() || workspace.nuclei_ != shape.nuclei ||
      workspace.determinants_ != shape.determinants ||
      workspace.width_ != shape.feature_dimension ||
      workspace.heads_ != shape.attention_heads ||
      workspace.blocks_ != shape.attention_blocks ||
      workspace.parameter_score_.size() != plan_.parameterCount() ||
      workspace.kinetic_parameter_response_.size() != plan_.parameterCount())
    throw std::invalid_argument("PsiFormer kinetic workspace belongs to a different execution plan");
  if (parameters_.size() != plan_.parameterCount())
    throw std::logic_error("PsiFormer parameter count changed after kinetic-plan construction");
}

/// Populate raw molecular features and their learned embedding.
inline void DirectKineticExecutor::buildEmbedding(const double* parameters,
                                                  DirectKineticWorkspace& workspace) const
{
  DirectTraceJetBuffer& raw = workspace.raw_features_;
  raw.clear();
  const GeometryPairTable& pairs = workspace.geometry_.electronNucleusPairs();
  const auto& displacements      = pairs.displacements();
  const auto& complementary      = pairs.complementaryDisplacements();
  const auto& displacement_jacobians = pairs.displacementJacobians();
  const auto& complementary_jacobians = pairs.complementaryDisplacementJacobians();
  const auto& displacement_laplacians = pairs.displacementLaplacians();
  const auto& complementary_laplacians = pairs.complementaryDisplacementLaplacians();
  const auto& distances          = pairs.distances();
  const auto& distance_gradients = pairs.distanceGradients();
  const auto& distance_gradient_norms = pairs.distanceGradientNormsSquared();
  const auto& distance_laplacians = pairs.distanceLaplacians();
  const auto& factors            = pairs.softenedRadialFactors();
  const bool periodic = workspace.geometry_.boundary().kind == GeometryBoundaryKind::PERIODIC;
  const std::size_t pair_width = periodic ? 7 : 4;

  for (std::size_t electron = 0; electron < workspace.electrons_; ++electron)
  {
    const std::size_t row_begin = electron * workspace.input_width_;
    for (std::size_t nucleus = 0; nucleus < workspace.nuclei_; ++nucleus)
    {
      const std::size_t pair = electron * workspace.nuclei_ + nucleus;
      const double radius    = distances[pair];
      if (radius == 0.0)
        throw std::runtime_error(
            "PsiFormer kinetic response is undefined at electron-nucleus coalescence");
      const GeometryPosition& displacement = displacements[pair];
      const SoftenedRadialFactors& radial  = factors[pair];
      const std::size_t radial_element     = row_begin + pair_width * nucleus;
      raw.value[radial_element]            = radial.log1p_radius;
      for (std::size_t component = 0; component < 3; ++component)
        raw.value[radial_element + 1 + component] =
            displacement[component] * radial.log1p_over_radius;
      if (periodic)
        for (std::size_t component = 0; component < 3; ++component)
          raw.value[radial_element + 4 + component] =
              complementary[pair][component] * radial.log1p_over_radius;

      for (std::size_t derivative_dimension = 0; derivative_dimension < 3;
           ++derivative_dimension)
      {
        const std::size_t lane = 3 * electron + derivative_dimension;
        const double radius_gradient = distance_gradients[pair][derivative_dimension];
        raw.gradient[gradientIndex(raw, lane, radial_element)] =
            radial.log1p_first * radius_gradient;
        for (std::size_t component = 0; component < 3; ++component)
        {
          raw.gradient[gradientIndex(raw, lane, radial_element + 1 + component)] =
              displacement_jacobians[pair][component][derivative_dimension] *
                  radial.log1p_over_radius + displacement[component] *
                  radial.log1p_over_radius_first * radius_gradient;
          if (periodic)
            raw.gradient[gradientIndex(raw, lane, radial_element + 4 + component)] =
                complementary_jacobians[pair][component][derivative_dimension] *
                    radial.log1p_over_radius + complementary[pair][component] *
                    radial.log1p_over_radius_first * radius_gradient;
        }
      }

      raw.laplacian[laplacianIndex(raw, electron, radial_element)] =
          radial.log1p_second * distance_gradient_norms[pair] +
          radial.log1p_first * distance_laplacians[pair];
      for (std::size_t component = 0; component < 3; ++component)
      {
        double displacement_gradient_dot_radius_gradient = 0;
        double complementary_gradient_dot_radius_gradient = 0;
        for (std::size_t derivative = 0; derivative < 3; ++derivative)
        {
          displacement_gradient_dot_radius_gradient +=
              displacement_jacobians[pair][component][derivative] *
              distance_gradients[pair][derivative];
          complementary_gradient_dot_radius_gradient +=
              complementary_jacobians[pair][component][derivative] *
              distance_gradients[pair][derivative];
        }
        const double radial_trace =
            radial.log1p_over_radius_second * distance_gradient_norms[pair] +
            radial.log1p_over_radius_first * distance_laplacians[pair];
        raw.laplacian[laplacianIndex(raw, electron, radial_element + 1 + component)] =
            displacement_laplacians[pair][component] * radial.log1p_over_radius +
            2 * radial.log1p_over_radius_first * displacement_gradient_dot_radius_gradient +
            displacement[component] * radial_trace;
        if (periodic)
          raw.laplacian[laplacianIndex(raw, electron, radial_element + 4 + component)] =
              complementary_laplacians[pair][component] * radial.log1p_over_radius +
              2 * radial.log1p_over_radius_first * complementary_gradient_dot_radius_gradient +
              complementary[pair][component] * radial_trace;
      }
    }
    raw.value[row_begin + workspace.input_width_ - 1] =
        electron < spin_up_electrons_ ? 1.0 : -1.0;
  }

  denseForward(raw,
               parameter(parameters, ParameterRole::ELECTRON_EMBEDDING_WEIGHT),
               nullptr, workspace.electrons_, workspace.input_width_, workspace.width_,
               workspace.gradient_lanes_, workspace.laplacian_lanes_, workspace.features_[0]);
}

/// Form one block's scaled query-key products.
inline void DirectKineticExecutor::buildAttentionLogits(
    std::size_t block,
    DirectKineticWorkspace& workspace) const
{
  DirectTraceJetBuffer& logits = workspace.logits_[block];
  const DirectTraceJetBuffer& query = workspace.queries_[block];
  const DirectTraceJetBuffer& key   = workspace.keys_[block];
  logits.clear();
  const double scale = 1.0 / std::sqrt(static_cast<double>(workspace.head_width_));
  for (std::size_t head = 0; head < workspace.heads_; ++head)
    for (std::size_t output_electron = 0; output_electron < workspace.electrons_;
         ++output_electron)
      for (std::size_t input_electron = 0; input_electron < workspace.electrons_;
           ++input_electron)
      {
        const std::size_t output =
            (head * workspace.electrons_ + output_electron) * workspace.electrons_ +
            input_electron;
        for (std::size_t feature = 0; feature < workspace.head_width_; ++feature)
        {
          const std::size_t query_element =
              output_electron * workspace.width_ + head * workspace.head_width_ + feature;
          const std::size_t key_element =
              input_electron * workspace.width_ + head * workspace.head_width_ + feature;
          productForwardElement(query, query_element, key, key_element,
                                workspace.gradient_lanes_, workspace.laplacian_lanes_,
                                logits, output, scale);
        }
      }
}

/// Evaluate stable row softmax on primal and contracted derivative planes.
inline void DirectKineticExecutor::softmaxForward(std::size_t block,
                                                  DirectKineticWorkspace& workspace) const
{
  const DirectTraceJetBuffer& logits = workspace.logits_[block];
  DirectTraceJetBuffer& probabilities = workspace.attention_[block];
  probabilities.clear();

  for (std::size_t head = 0; head < workspace.heads_; ++head)
    for (std::size_t output_electron = 0; output_electron < workspace.electrons_;
         ++output_electron)
    {
      const std::size_t row_begin =
          (head * workspace.electrons_ + output_electron) * workspace.electrons_;
      DirectTraceJetBuffer& exponential = workspace.row_exponential_;
      DirectTraceJetBuffer& sum         = workspace.row_sum_;
      DirectTraceJetBuffer& reciprocal  = workspace.row_reciprocal_;
      exponential.clear();
      sum.clear();
      reciprocal.clear();

      double row_maximum = -std::numeric_limits<double>::infinity();
      for (std::size_t input_electron = 0; input_electron < workspace.electrons_;
           ++input_electron)
        row_maximum = std::max(row_maximum, logits.value[row_begin + input_electron]);

      for (std::size_t input_electron = 0; input_electron < workspace.electrons_;
           ++input_electron)
      {
        const std::size_t logit_element = row_begin + input_electron;
        const double value = std::exp(logits.value[logit_element] - row_maximum);
        exponential.value[input_electron] = value;
        sum.value[0] += value;
        for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
        {
          const double input_gradient =
              logits.gradient[gradientIndex(logits, lane, logit_element)];
          const double output_gradient = value * input_gradient;
          exponential.gradient[gradientIndex(exponential, lane, input_electron)] =
              output_gradient;
          sum.gradient[lane] += output_gradient;
        }
        for (std::size_t electron = 0; electron < workspace.laplacian_lanes_;
             ++electron)
        {
          double squared_gradient = 0.0;
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
          {
            const double component = logits.gradient[gradientIndex(
                logits, 3 * electron + dimension, logit_element)];
            squared_gradient += component * component;
          }
          const double output_laplacian = value *
              (logits.laplacian[laplacianIndex(logits, electron, logit_element)] +
               squared_gradient);
          exponential.laplacian[
              laplacianIndex(exponential, electron, input_electron)] = output_laplacian;
          sum.laplacian[electron] += output_laplacian;
        }
      }

      const double inverse_sum = 1.0 / sum.value[0];
      reciprocal.value[0] = inverse_sum;
      for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
        reciprocal.gradient[lane] = -sum.gradient[lane] * inverse_sum * inverse_sum;
      for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
      {
        double squared_gradient = 0.0;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
        {
          const double component = sum.gradient[3 * electron + dimension];
          squared_gradient += component * component;
        }
        reciprocal.laplacian[electron] =
            -sum.laplacian[electron] * inverse_sum * inverse_sum +
            2.0 * squared_gradient * inverse_sum * inverse_sum * inverse_sum;
      }

      for (std::size_t input_electron = 0; input_electron < workspace.electrons_;
           ++input_electron)
        productForwardElement(exponential, input_electron, reciprocal, 0,
                              workspace.gradient_lanes_, workspace.laplacian_lanes_,
                              probabilities, row_begin + input_electron);
    }
}

/// Contract one block's attention probabilities and value features.
inline void DirectKineticExecutor::buildAttentionContext(
    std::size_t block,
    DirectKineticWorkspace& workspace) const
{
  const DirectTraceJetBuffer& probabilities = workspace.attention_[block];
  const DirectTraceJetBuffer& values        = workspace.values_[block];
  DirectTraceJetBuffer& context             = workspace.contexts_[block];
  context.clear();
  for (std::size_t output_electron = 0; output_electron < workspace.electrons_;
       ++output_electron)
    for (std::size_t head = 0; head < workspace.heads_; ++head)
      for (std::size_t input_electron = 0; input_electron < workspace.electrons_;
           ++input_electron)
      {
        const std::size_t probability_element =
            (head * workspace.electrons_ + output_electron) * workspace.electrons_ +
            input_electron;
        for (std::size_t feature = 0; feature < workspace.head_width_; ++feature)
        {
          const std::size_t output_element =
              output_electron * workspace.width_ + head * workspace.head_width_ + feature;
          const std::size_t value_element =
              input_electron * workspace.width_ + head * workspace.head_width_ + feature;
          productForwardElement(probabilities, probability_element, values, value_element,
                                workspace.gradient_lanes_, workspace.laplacian_lanes_,
                                context, output_element);
        }
      }
}

/// Evaluate one full self-attention, update, and residual block.
inline void DirectKineticExecutor::applyAttentionBlock(
    const double* parameters,
    std::size_t block,
    DirectKineticWorkspace& workspace) const
{
  const DirectTraceJetBuffer& input = workspace.features_[block];
  denseForward(input, parameter(parameters, ParameterRole::ATTENTION_QUERY_WEIGHT, block),
               nullptr, workspace.electrons_, workspace.width_, workspace.width_,
               workspace.gradient_lanes_, workspace.laplacian_lanes_, workspace.queries_[block]);
  denseForward(input, parameter(parameters, ParameterRole::ATTENTION_KEY_WEIGHT, block),
               nullptr, workspace.electrons_, workspace.width_, workspace.width_,
               workspace.gradient_lanes_, workspace.laplacian_lanes_, workspace.keys_[block]);
  denseForward(input, parameter(parameters, ParameterRole::ATTENTION_VALUE_WEIGHT, block),
               nullptr, workspace.electrons_, workspace.width_, workspace.width_,
               workspace.gradient_lanes_, workspace.laplacian_lanes_, workspace.values_[block]);
  buildAttentionLogits(block, workspace);
  softmaxForward(block, workspace);
  buildAttentionContext(block, workspace);

  denseForward(workspace.contexts_[block],
               parameter(parameters, ParameterRole::ATTENTION_OUTPUT_WEIGHT, block), nullptr,
               workspace.electrons_, workspace.width_, workspace.width_,
               workspace.gradient_lanes_, workspace.laplacian_lanes_, workspace.residuals_[block]);
  addJet(input, workspace.residuals_[block]);

  denseForward(workspace.residuals_[block],
               parameter(parameters, ParameterRole::UPDATE_HIDDEN_WEIGHT, block),
               parameter(parameters, ParameterRole::UPDATE_HIDDEN_BIAS, block),
               workspace.electrons_, workspace.width_, workspace.width_,
               workspace.gradient_lanes_, workspace.laplacian_lanes_, workspace.hidden_pre_[block]);
  tanhForward(workspace.hidden_pre_[block], workspace.gradient_lanes_,
              workspace.laplacian_lanes_, workspace.hidden_[block]);
  denseForward(workspace.hidden_[block],
               parameter(parameters, ParameterRole::UPDATE_OUTPUT_WEIGHT, block),
               parameter(parameters, ParameterRole::UPDATE_OUTPUT_BIAS, block),
               workspace.electrons_, workspace.width_, workspace.width_,
               workspace.gradient_lanes_, workspace.laplacian_lanes_, workspace.update_pre_[block]);
  tanhForward(workspace.update_pre_[block], workspace.gradient_lanes_,
              workspace.laplacian_lanes_, workspace.updates_[block]);
  copyJet(workspace.residuals_[block], workspace.features_[block + 1]);
  addJet(workspace.updates_[block], workspace.features_[block + 1]);
}

/// Evaluate orbital backflow and exponential envelopes from final features.
inline void DirectKineticExecutor::buildOrbitals(const double* parameters,
                                                 DirectKineticWorkspace& workspace) const
{
  DirectTraceJetBuffer& backflows = workspace.backflows_;
  DirectTraceJetBuffer& envelopes = workspace.envelopes_;
  DirectTraceJetBuffer& orbitals  = workspace.orbitals_;
  backflows.clear();
  envelopes.clear();
  orbitals.clear();
  const DirectTraceJetBuffer& features = workspace.features_[workspace.blocks_];
  const GeometryPairTable& pairs = workspace.geometry_.electronNucleusPairs();
  const auto& distances          = pairs.distances();
  const auto& distance_gradients = pairs.distanceGradients();
  const auto& distance_gradient_norms = pairs.distanceGradientNormsSquared();
  const auto& distance_laplacians = pairs.distanceLaplacians();
  const std::size_t channel_count = workspace.determinants_ * workspace.electrons_;

  for (std::size_t electron = 0; electron < workspace.electrons_; ++electron)
  {
    const bool spin_up = electron < spin_up_electrons_;
    const double* backflow_weight = parameter(
        parameters, spin_up ? ParameterRole::BACKFLOW_UP_WEIGHT
                            : ParameterRole::BACKFLOW_DOWN_WEIGHT);
    const double* pi = parameter(parameters, spin_up ? ParameterRole::ENVELOPE_PI_UP
                                                    : ParameterRole::ENVELOPE_PI_DOWN);
    const double* zeta = parameter(parameters, spin_up ? ParameterRole::ENVELOPE_ZETA_UP
                                                      : ParameterRole::ENVELOPE_ZETA_DOWN);
    const std::size_t feature_begin = electron * workspace.width_;
    for (std::size_t determinant = 0; determinant < workspace.determinants_; ++determinant)
      for (std::size_t orbital = 0; orbital < workspace.electrons_; ++orbital)
      {
        const std::size_t channel = determinant * workspace.electrons_ + orbital;
        const std::size_t matrix_element =
            (determinant * workspace.electrons_ + electron) * workspace.electrons_ + orbital;

        for (std::size_t feature = 0; feature < workspace.width_; ++feature)
        {
          const double weight = backflow_weight[feature * channel_count + channel];
          const std::size_t feature_element = feature_begin + feature;
          backflows.value[matrix_element] += features.value[feature_element] * weight;
          for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
            backflows.gradient[gradientIndex(backflows, lane, matrix_element)] +=
                features.gradient[gradientIndex(features, lane, feature_element)] * weight;
          for (std::size_t differentiating_electron = 0;
               differentiating_electron < workspace.laplacian_lanes_;
               ++differentiating_electron)
            backflows.laplacian[
                laplacianIndex(backflows, differentiating_electron, matrix_element)] +=
                features.laplacian[
                    laplacianIndex(features, differentiating_electron, feature_element)] * weight;
        }

        for (std::size_t nucleus = 0; nucleus < workspace.nuclei_; ++nucleus)
        {
          const std::size_t parameter_index = channel * workspace.nuclei_ + nucleus;
          const std::size_t pair             = electron * workspace.nuclei_ + nucleus;
          const double radius                = distances[pair];
          if (radius == 0.0)
            throw std::runtime_error(
                "PsiFormer kinetic response is undefined at electron-nucleus coalescence");
          const double decay_rate     = std::abs(zeta[parameter_index]);
          const double exponential    = std::exp(-decay_rate * radius);
          const double weighted_value = pi[parameter_index] * exponential;
          const double radial_first   = -decay_rate * weighted_value;
          const double radial_second  = decay_rate * decay_rate * weighted_value;
          envelopes.value[matrix_element] += weighted_value;
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
          {
            const std::size_t lane = 3 * electron + dimension;
            envelopes.gradient[gradientIndex(envelopes, lane, matrix_element)] +=
                radial_first * distance_gradients[pair][dimension];
          }
          envelopes.laplacian[laplacianIndex(envelopes, electron, matrix_element)] +=
              radial_second * distance_gradient_norms[pair] +
              radial_first * distance_laplacians[pair];
        }

        productForwardElement(backflows, matrix_element, envelopes, matrix_element,
                              workspace.gradient_lanes_, workspace.laplacian_lanes_,
                              orbitals, matrix_element);
      }
  }
}

/// Evaluate the pair cusp value, gradient lanes, and electron traces.
inline double DirectKineticExecutor::buildCusp(const double* parameters,
                                               DirectKineticWorkspace& workspace) const
{
  std::fill(workspace.cusp_gradient_.begin(), workspace.cusp_gradient_.end(), 0.0);
  std::fill(workspace.cusp_laplacian_.begin(), workspace.cusp_laplacian_.end(), 0.0);
  const double same_alpha = plan_.hasParameter(ParameterRole::CUSP_SAME_ALPHA)
      ? parameter(parameters, ParameterRole::CUSP_SAME_ALPHA)[0]
      : 1.0;
  const double opposite_alpha =
      parameter(parameters, ParameterRole::CUSP_OPPOSITE_ALPHA)[0];
  const auto& identities = workspace.geometry_.electronPairs();
  const GeometryPairTable& pair_table = workspace.geometry_.electronElectronPairs();
  const auto& distances     = pair_table.distances();
  const auto& distance_gradients = pair_table.distanceGradients();
  const auto& distance_gradient_norms = pair_table.distanceGradientNormsSquared();
  const auto& distance_laplacians = pair_table.distanceLaplacians();
  double value = 0.0;

  for (std::size_t pair_index = 0; pair_index < identities.size(); ++pair_index)
  {
    const ElectronPair pair = identities[pair_index];
    const bool same_spin = (pair.first < spin_up_electrons_) ==
        (pair.second < spin_up_electrons_);
    const double alpha  = same_spin ? same_alpha : opposite_alpha;
    const double factor = same_spin ? 0.25 : 0.5;
    const double radius = distances[pair_index];
    if (radius == 0.0)
      throw std::runtime_error(
          "PsiFormer kinetic response is undefined at electron-electron coalescence");
    const double denominator  = alpha + radius;
    const double numerator    = factor * alpha * alpha;
    const double radial_first = numerator / (denominator * denominator);
    const double radial_second = -2.0 * numerator /
        (denominator * denominator * denominator);
    value -= numerator / denominator;

    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double component =
          radial_first * distance_gradients[pair_index][dimension];
      workspace.cusp_gradient_[3 * pair.first + dimension] += component;
      workspace.cusp_gradient_[3 * pair.second + dimension] -= component;
    }
    const double pair_laplacian =
        radial_second * distance_gradient_norms[pair_index] +
        radial_first * distance_laplacians[pair_index];
    workspace.cusp_laplacian_[pair.first] += pair_laplacian;
    workspace.cusp_laplacian_[pair.second] += pair_laplacian;
  }
  return value;
}

/// Multiply square row-major matrices into preallocated scratch.
inline void DirectKineticExecutor::matrixProduct(const double* left,
                                                 const double* right,
                                                 double* output,
                                                 std::size_t size) noexcept
{
  std::fill(output, output + size * size, 0.0);
  for (std::size_t row = 0; row < size; ++row)
    for (std::size_t inner = 0; inner < size; ++inner)
      for (std::size_t column = 0; column < size; ++column)
        output[row * size + column] +=
            left[row * size + inner] * right[inner * size + column];
}

/// Contract the trace of a square row-major matrix product.
inline double DirectKineticExecutor::traceProduct(const double* left,
                                                  const double* right,
                                                  std::size_t size) noexcept
{
  double trace = 0.0;
  for (std::size_t row = 0; row < size; ++row)
    for (std::size_t column = 0; column < size; ++column)
      trace += left[row * size + column] * right[column * size + row];
  return trace;
}

/// Differentiate stable determinant factors across all coordinate trace lanes.
inline void DirectKineticExecutor::buildDeterminantFactors(
    DirectKineticWorkspace& workspace) const
{
  DirectTraceJetBuffer& factor = workspace.determinant_factor_;
  factor.clear();
  const std::size_t matrix_size = workspace.electrons_;
  const std::size_t matrix_elements = matrix_size * matrix_size;
  const std::size_t batch_elements  = workspace.orbital_elements_;

  for (std::size_t channel = 0; channel < workspace.determinants_; ++channel)
  {
    const std::size_t begin = channel * matrix_elements;
    const double* inverse = workspace.determinant_.inverse(channel);
    if (!inverse)
      throw std::domain_error(
          "PsiFormer determinant mixed reverse requires every contributing inverse");
    const double weight = static_cast<double>(workspace.determinant_.channelWeight(channel));

    // B = d log(sum_k det A_k) / d A_channel = w A^{-T}.
    for (std::size_t row = 0; row < matrix_size; ++row)
      for (std::size_t column = 0; column < matrix_size; ++column)
        factor.value[begin + row * matrix_size + column] =
            weight * inverse[column * matrix_size + row];

    for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
    {
      const double* matrix_gradient =
          workspace.orbitals_.gradient.data() + lane * batch_elements + begin;
      const double channel_gradient = traceProduct(inverse, matrix_gradient, matrix_size);
      const double weight_gradient =
          weight * (channel_gradient - workspace.determinant_gradient_[lane]);
      matrixProduct(inverse, matrix_gradient, workspace.matrix_scratch_a_.data(), matrix_size);
      matrixProduct(workspace.matrix_scratch_a_.data(), inverse,
                    workspace.matrix_scratch_b_.data(), matrix_size);
      for (std::size_t row = 0; row < matrix_size; ++row)
        for (std::size_t column = 0; column < matrix_size; ++column)
        {
          const std::size_t element = begin + row * matrix_size + column;
          const double inverse_transpose = inverse[column * matrix_size + row];
          const double inverse_transpose_gradient =
              -workspace.matrix_scratch_b_[column * matrix_size + row];
          factor.gradient[gradientIndex(factor, lane, element)] =
              weight_gradient * inverse_transpose +
              weight * inverse_transpose_gradient;
        }
    }

    for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
    {
      const double* matrix_laplacian =
          workspace.orbitals_.laplacian.data() + electron * batch_elements + begin;
      double channel_log_laplacian =
          traceProduct(inverse, matrix_laplacian, matrix_size);
      double squared_gradient_difference = 0.0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const std::size_t lane = 3 * electron + dimension;
        const double* matrix_gradient =
            workspace.orbitals_.gradient.data() + lane * batch_elements + begin;
        const double channel_gradient = traceProduct(inverse, matrix_gradient, matrix_size);
        matrixProduct(inverse, matrix_gradient, workspace.matrix_scratch_a_.data(), matrix_size);
        channel_log_laplacian -= traceProduct(workspace.matrix_scratch_a_.data(),
                                              workspace.matrix_scratch_a_.data(), matrix_size);
        const double difference = channel_gradient - workspace.determinant_gradient_[lane];
        squared_gradient_difference += difference * difference;
      }
      const double weight_laplacian = weight *
          (channel_log_laplacian - workspace.determinant_lap_log_[electron] +
           squared_gradient_difference);

      for (std::size_t row = 0; row < matrix_size; ++row)
        for (std::size_t column = 0; column < matrix_size; ++column)
        {
          const std::size_t element = begin + row * matrix_size + column;
          factor.laplacian[laplacianIndex(factor, electron, element)] =
              weight_laplacian * inverse[column * matrix_size + row];
        }

      // 2 sum_d (d w)(d A^{-T}) + w Delta(A^{-T}).
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const std::size_t lane = 3 * electron + dimension;
        const double* matrix_gradient =
            workspace.orbitals_.gradient.data() + lane * batch_elements + begin;
        const double channel_gradient = traceProduct(inverse, matrix_gradient, matrix_size);
        const double weight_gradient =
            weight * (channel_gradient - workspace.determinant_gradient_[lane]);
        matrixProduct(inverse, matrix_gradient, workspace.matrix_scratch_a_.data(), matrix_size);
        matrixProduct(workspace.matrix_scratch_a_.data(), inverse,
                      workspace.matrix_scratch_b_.data(), matrix_size);
        for (std::size_t row = 0; row < matrix_size; ++row)
          for (std::size_t column = 0; column < matrix_size; ++column)
          {
            const std::size_t element = begin + row * matrix_size + column;
            factor.laplacian[laplacianIndex(factor, electron, element)] +=
                -2.0 * weight_gradient *
                workspace.matrix_scratch_b_[column * matrix_size + row];
          }

        // 2 A^{-1} A_c A^{-1} A_c A^{-1}, transposed.
        matrixProduct(workspace.matrix_scratch_a_.data(),
                      workspace.matrix_scratch_a_.data(),
                      workspace.matrix_scratch_b_.data(), matrix_size);
        matrixProduct(workspace.matrix_scratch_b_.data(), inverse,
                      workspace.matrix_scratch_c_.data(), matrix_size);
        for (std::size_t row = 0; row < matrix_size; ++row)
          for (std::size_t column = 0; column < matrix_size; ++column)
          {
            const std::size_t element = begin + row * matrix_size + column;
            factor.laplacian[laplacianIndex(factor, electron, element)] +=
                2.0 * weight * workspace.matrix_scratch_c_[column * matrix_size + row];
          }
      }

      // -A^{-1} (Delta A) A^{-1}, transposed.
      matrixProduct(inverse, matrix_laplacian, workspace.matrix_scratch_a_.data(), matrix_size);
      matrixProduct(workspace.matrix_scratch_a_.data(), inverse,
                    workspace.matrix_scratch_b_.data(), matrix_size);
      for (std::size_t row = 0; row < matrix_size; ++row)
        for (std::size_t column = 0; column < matrix_size; ++column)
        {
          const std::size_t element = begin + row * matrix_size + column;
          factor.laplacian[laplacianIndex(factor, electron, element)] -=
              weight * workspace.matrix_scratch_b_[column * matrix_size + row];
        }
    }
  }
}

/// Lift the determinant-root adjoint to orbital value, gradient, and trace planes.
inline void DirectKineticExecutor::reverseDeterminant(
    DirectKineticWorkspace& workspace) const
{
  DirectTraceJetBuffer& orbital_adjoint = workspace.orbital_adjoint_;
  const DirectTraceJetBuffer& factor    = workspace.determinant_factor_;
  const DirectTraceJetBuffer& root      = workspace.root_adjoint_;
  orbital_adjoint.clear();
  for (std::size_t element = 0; element < workspace.orbital_elements_; ++element)
  {
    double value_adjoint = root.value[0] * factor.value[element];
    for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
    {
      const std::size_t electron = lane / 3;
      const double factor_gradient = factor.gradient[gradientIndex(factor, lane, element)];
      value_adjoint += root.gradient[lane] * factor_gradient;
      orbital_adjoint.gradient[gradientIndex(orbital_adjoint, lane, element)] =
          root.gradient[lane] * factor.value[element] +
          2.0 * root.laplacian[electron] * factor_gradient;
    }
    for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
    {
      value_adjoint += root.laplacian[electron] *
          factor.laplacian[laplacianIndex(factor, electron, element)];
      orbital_adjoint.laplacian[laplacianIndex(orbital_adjoint, electron, element)] =
          root.laplacian[electron] * factor.value[element];
    }
    orbital_adjoint.value[element] = value_adjoint;
  }
}

/// Reverse orbital, backflow, and envelope trace jets into model parameters.
inline void DirectKineticExecutor::reverseOrbitals(
    const double* parameters,
    DirectKineticWorkspace& workspace,
    std::vector<double>& destination) const
{
  workspace.backflow_adjoint_.clear();
  workspace.envelope_adjoint_.clear();
  for (std::size_t element = 0; element < workspace.orbital_elements_; ++element)
    productReverseElement(workspace.backflows_, element, workspace.envelopes_, element,
                          workspace.orbital_adjoint_, element,
                          workspace.gradient_lanes_, workspace.laplacian_lanes_,
                          workspace.backflow_adjoint_, workspace.envelope_adjoint_);

  workspace.feature_adjoint_a_.clear();
  const DirectTraceJetBuffer& features = workspace.features_[workspace.blocks_];
  const GeometryPairTable& electron_nucleus_pairs =
      workspace.geometry_.electronNucleusPairs();
  const auto& distances = electron_nucleus_pairs.distances();
  const auto& distance_gradients = electron_nucleus_pairs.distanceGradients();
  const auto& distance_gradient_norms =
      electron_nucleus_pairs.distanceGradientNormsSquared();
  const auto& distance_laplacians = electron_nucleus_pairs.distanceLaplacians();
  const std::size_t channel_count = workspace.determinants_ * workspace.electrons_;

  for (std::size_t electron = 0; electron < workspace.electrons_; ++electron)
  {
    const bool spin_up = electron < spin_up_electrons_;
    const ParameterRole backflow_role = spin_up ? ParameterRole::BACKFLOW_UP_WEIGHT
                                                : ParameterRole::BACKFLOW_DOWN_WEIGHT;
    const ParameterRole pi_role = spin_up ? ParameterRole::ENVELOPE_PI_UP
                                          : ParameterRole::ENVELOPE_PI_DOWN;
    const ParameterRole zeta_role = spin_up ? ParameterRole::ENVELOPE_ZETA_UP
                                            : ParameterRole::ENVELOPE_ZETA_DOWN;
    const double* backflow_weight = parameter(parameters, backflow_role);
    const double* pi               = parameter(parameters, pi_role);
    const double* zeta             = parameter(parameters, zeta_role);
    double* backflow_response = response(destination, backflow_role);
    double* pi_response       = response(destination, pi_role);
    double* zeta_response     = response(destination, zeta_role);
    const std::size_t feature_begin = electron * workspace.width_;

    for (std::size_t determinant = 0; determinant < workspace.determinants_; ++determinant)
      for (std::size_t orbital = 0; orbital < workspace.electrons_; ++orbital)
      {
        const std::size_t channel = determinant * workspace.electrons_ + orbital;
        const std::size_t matrix_element =
            (determinant * workspace.electrons_ + electron) * workspace.electrons_ + orbital;

        // Lifted reverse of B = feature_row * backflow_weight.
        for (std::size_t feature = 0; feature < workspace.width_; ++feature)
        {
          const std::size_t feature_element = feature_begin + feature;
          const std::size_t parameter_element = feature * channel_count + channel;
          const double weight = backflow_weight[parameter_element];
          const double value_upstream = workspace.backflow_adjoint_.value[matrix_element];
          workspace.feature_adjoint_a_.value[feature_element] += value_upstream * weight;
          backflow_response[parameter_element] +=
              features.value[feature_element] * value_upstream;
          for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
          {
            const double upstream = workspace.backflow_adjoint_.gradient[
                gradientIndex(workspace.backflow_adjoint_, lane, matrix_element)];
            workspace.feature_adjoint_a_.gradient[
                gradientIndex(workspace.feature_adjoint_a_, lane, feature_element)] += upstream * weight;
            backflow_response[parameter_element] +=
                features.gradient[gradientIndex(features, lane, feature_element)] * upstream;
          }
          for (std::size_t differentiating_electron = 0;
               differentiating_electron < workspace.laplacian_lanes_;
               ++differentiating_electron)
          {
            const double upstream = workspace.backflow_adjoint_.laplacian[
                laplacianIndex(workspace.backflow_adjoint_, differentiating_electron,
                               matrix_element)];
            workspace.feature_adjoint_a_.laplacian[
                laplacianIndex(workspace.feature_adjoint_a_, differentiating_electron,
                               feature_element)] += upstream * weight;
            backflow_response[parameter_element] +=
                features.laplacian[
                    laplacianIndex(features, differentiating_electron, feature_element)] * upstream;
          }
        }

        // Exact parameter derivatives of E = sum_n pi exp(-|zeta| r), including
        // derivatives of its Cartesian gradient and contracted Laplacian.
        for (std::size_t nucleus = 0; nucleus < workspace.nuclei_; ++nucleus)
        {
          const std::size_t parameter_index = channel * workspace.nuclei_ + nucleus;
          const std::size_t pair             = electron * workspace.nuclei_ + nucleus;
          const double radius                = distances[pair];
          const double decay_rate            = std::abs(zeta[parameter_index]);
          const double decay                 = std::exp(-decay_rate * radius);
          const double signed_abs_derivative = zeta[parameter_index] > 0.0
              ? 1.0
              : (zeta[parameter_index] < 0.0 ? -1.0 : 0.0);
          const double weighted_decay = pi[parameter_index] * decay;
          const double pi_gradient_radial = -decay_rate * decay;
          const double pi_radial_second = decay_rate * decay_rate * decay;
          const double zeta_value = -signed_abs_derivative * radius * weighted_decay;
          const double zeta_gradient_radial = signed_abs_derivative * weighted_decay *
              (decay_rate * radius - 1.0);
          const double zeta_radial_second = signed_abs_derivative * weighted_decay *
              (-radius * decay_rate * decay_rate + 2.0 * decay_rate);

          double pi_contribution = workspace.envelope_adjoint_.value[matrix_element] * decay;
          double zeta_contribution =
              workspace.envelope_adjoint_.value[matrix_element] * zeta_value;
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
          {
            const std::size_t lane = 3 * electron + dimension;
            const double radial_gradient = distance_gradients[pair][dimension];
            const double upstream = workspace.envelope_adjoint_.gradient[
                gradientIndex(workspace.envelope_adjoint_, lane, matrix_element)];
            pi_contribution += upstream * pi_gradient_radial * radial_gradient;
            zeta_contribution += upstream * zeta_gradient_radial * radial_gradient;
          }
          const double laplacian_upstream = workspace.envelope_adjoint_.laplacian[
              laplacianIndex(workspace.envelope_adjoint_, electron, matrix_element)];
          pi_contribution += laplacian_upstream *
              (pi_radial_second * distance_gradient_norms[pair] +
               pi_gradient_radial * distance_laplacians[pair]);
          zeta_contribution += laplacian_upstream *
              (zeta_radial_second * distance_gradient_norms[pair] +
               zeta_gradient_radial * distance_laplacians[pair]);
          pi_response[parameter_index] += pi_contribution;
          zeta_response[parameter_index] += zeta_contribution;
        }
      }
  }
}

/// Reverse the analytic cusp trace jet into same/opposite-spin parameters.
inline void DirectKineticExecutor::reverseCusp(
    const double* parameters,
    DirectKineticWorkspace& workspace,
    std::vector<double>& destination) const
{
  const bool has_same_alpha = plan_.hasParameter(ParameterRole::CUSP_SAME_ALPHA);
  const double same_alpha = has_same_alpha
      ? parameter(parameters, ParameterRole::CUSP_SAME_ALPHA)[0]
      : 1.0;
  const double opposite_alpha =
      parameter(parameters, ParameterRole::CUSP_OPPOSITE_ALPHA)[0];
  double* same_response = has_same_alpha
      ? response(destination, ParameterRole::CUSP_SAME_ALPHA)
      : nullptr;
  double& opposite_response =
      response(destination, ParameterRole::CUSP_OPPOSITE_ALPHA)[0];
  const auto& identities = workspace.geometry_.electronPairs();
  const GeometryPairTable& electron_pairs = workspace.geometry_.electronElectronPairs();
  const auto& distances = electron_pairs.distances();
  const auto& distance_gradients = electron_pairs.distanceGradients();
  const auto& distance_gradient_norms = electron_pairs.distanceGradientNormsSquared();
  const auto& distance_laplacians = electron_pairs.distanceLaplacians();

  for (std::size_t pair_index = 0; pair_index < identities.size(); ++pair_index)
  {
    const ElectronPair pair = identities[pair_index];
    const bool same_spin = (pair.first < spin_up_electrons_) ==
        (pair.second < spin_up_electrons_);
    const double alpha  = same_spin ? same_alpha : opposite_alpha;
    const double factor = same_spin ? 0.25 : 0.5;
    const double radius = distances[pair_index];
    const double denominator = alpha + radius;
    const double denominator2 = denominator * denominator;
    const double denominator3 = denominator2 * denominator;
    const double denominator4 = denominator3 * denominator;
    const double value_derivative =
        -factor * alpha * (alpha + 2.0 * radius) / denominator2;
    const double radial_first_derivative =
        2.0 * factor * alpha * radius / denominator3;
    const double radial_second_derivative =
        2.0 * factor * alpha * (alpha - 2.0 * radius) / denominator4;
    double contribution = workspace.root_adjoint_.value[0] * value_derivative;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double directional = radial_first_derivative *
          distance_gradients[pair_index][dimension];
      contribution += workspace.root_adjoint_.gradient[3 * pair.first + dimension] * directional;
      contribution -= workspace.root_adjoint_.gradient[3 * pair.second + dimension] * directional;
    }
    contribution +=
        (workspace.root_adjoint_.laplacian[pair.first] +
         workspace.root_adjoint_.laplacian[pair.second]) *
        (radial_second_derivative * distance_gradient_norms[pair_index] +
         radial_first_derivative * distance_laplacians[pair_index]);
    if (same_spin)
    {
      if (!same_response)
        throw std::logic_error(
            "PsiFormer kinetic plan omitted a required same-spin cusp parameter");
      same_response[0] += contribution;
    }
    else
      opposite_response += contribution;
  }
}

/// Reverse attention-value contraction for one block.
inline void DirectKineticExecutor::reverseAttentionContext(
    std::size_t block,
    DirectKineticWorkspace& workspace) const
{
  workspace.attention_adjoint_.clear();
  workspace.value_adjoint_.clear();
  const DirectTraceJetBuffer& probabilities = workspace.attention_[block];
  const DirectTraceJetBuffer& values        = workspace.values_[block];
  for (std::size_t output_electron = 0; output_electron < workspace.electrons_;
       ++output_electron)
    for (std::size_t head = 0; head < workspace.heads_; ++head)
      for (std::size_t input_electron = 0; input_electron < workspace.electrons_;
           ++input_electron)
      {
        const std::size_t probability_element =
            (head * workspace.electrons_ + output_electron) * workspace.electrons_ +
            input_electron;
        for (std::size_t feature = 0; feature < workspace.head_width_; ++feature)
        {
          const std::size_t output_element =
              output_electron * workspace.width_ + head * workspace.head_width_ + feature;
          const std::size_t value_element =
              input_electron * workspace.width_ + head * workspace.head_width_ + feature;
          productReverseElement(probabilities, probability_element, values, value_element,
                                workspace.context_adjoint_, output_element,
                                workspace.gradient_lanes_, workspace.laplacian_lanes_,
                                workspace.attention_adjoint_, workspace.value_adjoint_);
        }
      }
}

/// Reverse stable row softmax through a reusable fixed-size row tape.
inline void DirectKineticExecutor::softmaxReverse(std::size_t block,
                                                  DirectKineticWorkspace& workspace) const
{
  const DirectTraceJetBuffer& logits = workspace.logits_[block];
  workspace.logit_adjoint_.clear();

  for (std::size_t head = 0; head < workspace.heads_; ++head)
    for (std::size_t output_electron = 0; output_electron < workspace.electrons_;
         ++output_electron)
    {
      const std::size_t row_begin =
          (head * workspace.electrons_ + output_electron) * workspace.electrons_;
      DirectTraceJetBuffer& exponential = workspace.row_exponential_;
      DirectTraceJetBuffer& sum         = workspace.row_sum_;
      DirectTraceJetBuffer& reciprocal  = workspace.row_reciprocal_;
      DirectTraceJetBuffer& exponential_adjoint = workspace.row_exponential_adjoint_;
      DirectTraceJetBuffer& sum_adjoint = workspace.row_sum_adjoint_;
      DirectTraceJetBuffer& reciprocal_adjoint = workspace.row_reciprocal_adjoint_;
      exponential.clear();
      sum.clear();
      reciprocal.clear();
      exponential_adjoint.clear();
      sum_adjoint.clear();
      reciprocal_adjoint.clear();

      double row_maximum = -std::numeric_limits<double>::infinity();
      for (std::size_t input_electron = 0; input_electron < workspace.electrons_;
           ++input_electron)
        row_maximum = std::max(row_maximum, logits.value[row_begin + input_electron]);

      // Recompute the fixed-size stabilized exp/sum/reciprocal sub-tape.
      for (std::size_t input_electron = 0; input_electron < workspace.electrons_;
           ++input_electron)
      {
        const std::size_t logit_element = row_begin + input_electron;
        const double value = std::exp(logits.value[logit_element] - row_maximum);
        exponential.value[input_electron] = value;
        sum.value[0] += value;
        for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
        {
          const double derivative = value *
              logits.gradient[gradientIndex(logits, lane, logit_element)];
          exponential.gradient[gradientIndex(exponential, lane, input_electron)] = derivative;
          sum.gradient[lane] += derivative;
        }
        for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
        {
          double squared_gradient = 0.0;
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
          {
            const double component = logits.gradient[gradientIndex(
                logits, 3 * electron + dimension, logit_element)];
            squared_gradient += component * component;
          }
          const double derivative = value *
              (logits.laplacian[laplacianIndex(logits, electron, logit_element)] +
               squared_gradient);
          exponential.laplacian[
              laplacianIndex(exponential, electron, input_electron)] = derivative;
          sum.laplacian[electron] += derivative;
        }
      }
      const double inverse_sum = 1.0 / sum.value[0];
      reciprocal.value[0] = inverse_sum;
      for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
        reciprocal.gradient[lane] = -sum.gradient[lane] * inverse_sum * inverse_sum;
      for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
      {
        double squared_gradient = 0.0;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
        {
          const double component = sum.gradient[3 * electron + dimension];
          squared_gradient += component * component;
        }
        reciprocal.laplacian[electron] =
            -sum.laplacian[electron] * inverse_sum * inverse_sum +
            2.0 * squared_gradient * inverse_sum * inverse_sum * inverse_sum;
      }

      // Reverse probability_i = exp_i * reciprocal_sum.
      for (std::size_t input_electron = 0; input_electron < workspace.electrons_;
           ++input_electron)
        productReverseElement(exponential, input_electron, reciprocal, 0,
                              workspace.attention_adjoint_, row_begin + input_electron,
                              workspace.gradient_lanes_, workspace.laplacian_lanes_,
                              exponential_adjoint, reciprocal_adjoint);

      // Reverse r = 1/s as an elementwise unary trace jet.
      const double s = sum.value[0];
      const double first  = -1.0 / (s * s);
      const double second = 2.0 / (s * s * s);
      const double third  = -6.0 / (s * s * s * s);
      double sum_value_adjoint = reciprocal_adjoint.value[0] * first;
      for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
      {
        const std::size_t electron = lane / 3;
        sum_value_adjoint += reciprocal_adjoint.gradient[lane] * second * sum.gradient[lane];
        sum_adjoint.gradient[lane] = reciprocal_adjoint.gradient[lane] * first +
            2.0 * reciprocal_adjoint.laplacian[electron] * second * sum.gradient[lane];
      }
      for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
      {
        double squared_gradient = 0.0;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
        {
          const double component = sum.gradient[3 * electron + dimension];
          squared_gradient += component * component;
        }
        sum_value_adjoint += reciprocal_adjoint.laplacian[electron] *
            (second * sum.laplacian[electron] + third * squared_gradient);
        sum_adjoint.laplacian[electron] = reciprocal_adjoint.laplacian[electron] * first;
      }
      sum_adjoint.value[0] = sum_value_adjoint;

      // Reverse s = sum_i exp_i.
      for (std::size_t input_electron = 0; input_electron < workspace.electrons_;
           ++input_electron)
      {
        exponential_adjoint.value[input_electron] += sum_adjoint.value[0];
        for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
          exponential_adjoint.gradient[
              gradientIndex(exponential_adjoint, lane, input_electron)] +=
              sum_adjoint.gradient[lane];
        for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
          exponential_adjoint.laplacian[
              laplacianIndex(exponential_adjoint, electron, input_electron)] +=
              sum_adjoint.laplacian[electron];
      }

      // Reverse exp_i = exp(logit_i - frozen_row_shift).  Softmax invariance makes
      // the omitted max-shift derivative cancel exactly in real arithmetic.
      for (std::size_t input_electron = 0; input_electron < workspace.electrons_;
           ++input_electron)
      {
        const std::size_t logit_element = row_begin + input_electron;
        const double exponential_value = exponential.value[input_electron];
        double value_adjoint = exponential_adjoint.value[input_electron] * exponential_value;
        for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
        {
          const std::size_t electron = lane / 3;
          const double input_gradient =
              logits.gradient[gradientIndex(logits, lane, logit_element)];
          const double gradient_upstream = exponential_adjoint.gradient[
              gradientIndex(exponential_adjoint, lane, input_electron)];
          const double laplacian_upstream = exponential_adjoint.laplacian[
              laplacianIndex(exponential_adjoint, electron, input_electron)];
          value_adjoint += gradient_upstream * exponential_value * input_gradient;
          workspace.logit_adjoint_.gradient[
              gradientIndex(workspace.logit_adjoint_, lane, logit_element)] +=
              gradient_upstream * exponential_value +
              2.0 * laplacian_upstream * exponential_value * input_gradient;
        }
        for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
        {
          double squared_gradient = 0.0;
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
          {
            const double component = logits.gradient[gradientIndex(
                logits, 3 * electron + dimension, logit_element)];
            squared_gradient += component * component;
          }
          const double laplacian_upstream = exponential_adjoint.laplacian[
              laplacianIndex(exponential_adjoint, electron, input_electron)];
          value_adjoint += laplacian_upstream * exponential_value *
              (logits.laplacian[laplacianIndex(logits, electron, logit_element)] +
               squared_gradient);
          workspace.logit_adjoint_.laplacian[
              laplacianIndex(workspace.logit_adjoint_, electron, logit_element)] +=
              laplacian_upstream * exponential_value;
        }
        workspace.logit_adjoint_.value[logit_element] += value_adjoint;
      }
    }
}

/// Reverse scaled query-key products for one attention block.
inline void DirectKineticExecutor::reverseAttentionLogits(
    std::size_t block,
    DirectKineticWorkspace& workspace) const
{
  workspace.query_adjoint_.clear();
  workspace.key_adjoint_.clear();
  const DirectTraceJetBuffer& query = workspace.queries_[block];
  const DirectTraceJetBuffer& key   = workspace.keys_[block];
  const double scale = 1.0 / std::sqrt(static_cast<double>(workspace.head_width_));
  for (std::size_t head = 0; head < workspace.heads_; ++head)
    for (std::size_t output_electron = 0; output_electron < workspace.electrons_;
         ++output_electron)
      for (std::size_t input_electron = 0; input_electron < workspace.electrons_;
           ++input_electron)
      {
        const std::size_t output =
            (head * workspace.electrons_ + output_electron) * workspace.electrons_ +
            input_electron;
        for (std::size_t feature = 0; feature < workspace.head_width_; ++feature)
        {
          const std::size_t query_element =
              output_electron * workspace.width_ + head * workspace.head_width_ + feature;
          const std::size_t key_element =
              input_electron * workspace.width_ + head * workspace.head_width_ + feature;
          productReverseElement(query, query_element, key, key_element,
                                workspace.logit_adjoint_, output,
                                workspace.gradient_lanes_, workspace.laplacian_lanes_,
                                workspace.query_adjoint_, workspace.key_adjoint_, scale);
        }
      }
}

/// Reverse one full attention/update/residual block.
inline void DirectKineticExecutor::reverseAttentionBlock(
    const double* parameters,
    std::size_t block,
    DirectKineticWorkspace& workspace,
    std::vector<double>& destination) const
{
  const DirectTraceJetBuffer& input = workspace.features_[block];

  // features_{b+1} = residual + tanh(update_pre).
  copyJet(workspace.feature_adjoint_a_, workspace.residual_adjoint_);
  copyJet(workspace.feature_adjoint_a_, workspace.update_adjoint_);
  tanhReverse(workspace.update_pre_[block], workspace.updates_[block],
              workspace.update_adjoint_, workspace.gradient_lanes_,
              workspace.laplacian_lanes_, workspace.update_pre_adjoint_);

  workspace.hidden_adjoint_.clear();
  denseReverse(workspace.hidden_[block],
               parameter(parameters, ParameterRole::UPDATE_OUTPUT_WEIGHT, block),
               workspace.update_pre_adjoint_, workspace.electrons_, workspace.width_,
               workspace.width_, workspace.gradient_lanes_, workspace.laplacian_lanes_,
               workspace.hidden_adjoint_,
               response(destination, ParameterRole::UPDATE_OUTPUT_WEIGHT, block),
               response(destination, ParameterRole::UPDATE_OUTPUT_BIAS, block));
  tanhReverse(workspace.hidden_pre_[block], workspace.hidden_[block],
              workspace.hidden_adjoint_, workspace.gradient_lanes_,
              workspace.laplacian_lanes_, workspace.hidden_pre_adjoint_);
  denseReverse(workspace.residuals_[block],
               parameter(parameters, ParameterRole::UPDATE_HIDDEN_WEIGHT, block),
               workspace.hidden_pre_adjoint_, workspace.electrons_, workspace.width_,
               workspace.width_, workspace.gradient_lanes_, workspace.laplacian_lanes_,
               workspace.residual_adjoint_,
               response(destination, ParameterRole::UPDATE_HIDDEN_WEIGHT, block),
               response(destination, ParameterRole::UPDATE_HIDDEN_BIAS, block));

  // residual = input + context * output_weight.
  workspace.context_adjoint_.clear();
  denseReverse(workspace.contexts_[block],
               parameter(parameters, ParameterRole::ATTENTION_OUTPUT_WEIGHT, block),
               workspace.residual_adjoint_, workspace.electrons_, workspace.width_,
               workspace.width_, workspace.gradient_lanes_, workspace.laplacian_lanes_,
               workspace.context_adjoint_,
               response(destination, ParameterRole::ATTENTION_OUTPUT_WEIGHT, block),
               nullptr);
  copyJet(workspace.residual_adjoint_, workspace.feature_adjoint_b_);

  reverseAttentionContext(block, workspace);
  softmaxReverse(block, workspace);
  reverseAttentionLogits(block, workspace);

  denseReverse(input, parameter(parameters, ParameterRole::ATTENTION_QUERY_WEIGHT, block),
               workspace.query_adjoint_, workspace.electrons_, workspace.width_, workspace.width_,
               workspace.gradient_lanes_, workspace.laplacian_lanes_, workspace.feature_adjoint_b_,
               response(destination, ParameterRole::ATTENTION_QUERY_WEIGHT, block), nullptr);
  denseReverse(input, parameter(parameters, ParameterRole::ATTENTION_KEY_WEIGHT, block),
               workspace.key_adjoint_, workspace.electrons_, workspace.width_, workspace.width_,
               workspace.gradient_lanes_, workspace.laplacian_lanes_, workspace.feature_adjoint_b_,
               response(destination, ParameterRole::ATTENTION_KEY_WEIGHT, block), nullptr);
  denseReverse(input, parameter(parameters, ParameterRole::ATTENTION_VALUE_WEIGHT, block),
               workspace.value_adjoint_, workspace.electrons_, workspace.width_, workspace.width_,
               workspace.gradient_lanes_, workspace.laplacian_lanes_, workspace.feature_adjoint_b_,
               response(destination, ParameterRole::ATTENTION_VALUE_WEIGHT, block), nullptr);
  std::swap(workspace.feature_adjoint_a_, workspace.feature_adjoint_b_);
}

/// Reverse the initial learned electron embedding.
inline void DirectKineticExecutor::reverseEmbedding(
    DirectKineticWorkspace& workspace,
    std::vector<double>& destination) const
{
  double* embedding_response =
      response(destination, ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
  for (std::size_t electron = 0; electron < workspace.electrons_; ++electron)
    for (std::size_t input = 0; input < workspace.input_width_; ++input)
    {
      const std::size_t source_element = electron * workspace.input_width_ + input;
      for (std::size_t output = 0; output < workspace.width_; ++output)
      {
        const std::size_t target_element = electron * workspace.width_ + output;
        const std::size_t parameter_element = input * workspace.width_ + output;
        embedding_response[parameter_element] +=
            workspace.raw_features_.value[source_element] *
            workspace.feature_adjoint_a_.value[target_element];
        for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
          embedding_response[parameter_element] +=
              workspace.raw_features_.gradient[
                  gradientIndex(workspace.raw_features_, lane, source_element)] *
              workspace.feature_adjoint_a_.gradient[
                  gradientIndex(workspace.feature_adjoint_a_, lane, target_element)];
        for (std::size_t differentiating_electron = 0;
             differentiating_electron < workspace.laplacian_lanes_;
             ++differentiating_electron)
          embedding_response[parameter_element] +=
              workspace.raw_features_.laplacian[
                  laplacianIndex(workspace.raw_features_, differentiating_electron,
                                 source_element)] *
              workspace.feature_adjoint_a_.laplacian[
                  laplacianIndex(workspace.feature_adjoint_a_, differentiating_electron,
                                 target_element)];
      }
    }
}

/// Execute one complete root-seeded lifted reverse into a canonical response.
inline void DirectKineticExecutor::reverse(const double* parameters,
                                           DirectKineticWorkspace& workspace,
                                           std::vector<double>& destination) const
{
  std::fill(destination.begin(), destination.end(), 0.0);
  reverseDeterminant(workspace);
  reverseOrbitals(parameters, workspace, destination);
  for (std::size_t reverse_index = workspace.blocks_; reverse_index > 0; --reverse_index)
    reverseAttentionBlock(parameters, reverse_index - 1, workspace, destination);
  reverseEmbedding(workspace, destination);
  reverseCusp(parameters, workspace, destination);
}

/// Evaluate the shared forward tape, score reverse, and kinetic-response reverse.
inline DirectKineticResultView DirectKineticExecutor::evaluate(
    DirectKineticWorkspace& workspace,
    const double* total_log_gradient,
    std::size_t total_log_gradient_size) const
{
  return evaluate(workspace, total_log_gradient, total_log_gradient_size,
                  nullptr, 0);
}

/// Evaluate score and mass-weighted kinetic response from one shared forward tape.
inline DirectKineticResultView DirectKineticExecutor::evaluate(
    DirectKineticWorkspace& workspace,
    const double* total_log_gradient,
    std::size_t total_log_gradient_size,
    const double* inverse_masses,
    std::size_t inverse_mass_count) const
{
  validateWorkspace(workspace);
  if ((total_log_gradient && total_log_gradient_size != workspace.gradient_lanes_) ||
      (!total_log_gradient && total_log_gradient_size != 0))
    throw std::invalid_argument("PsiFormer total TrialWaveFunction drift has the wrong size");
  if (total_log_gradient)
    for (std::size_t lane = 0; lane < total_log_gradient_size; ++lane)
      if (!is_finite_parameter_value(total_log_gradient[lane]))
        throw std::invalid_argument(
            "PsiFormer total TrialWaveFunction drift must be finite");
  if ((inverse_masses && inverse_mass_count != workspace.laplacian_lanes_) ||
      (!inverse_masses && inverse_mass_count != 0))
    throw std::invalid_argument("PsiFormer inverse-mass view has the wrong size");
  if (inverse_masses)
    for (std::size_t electron = 0; electron < inverse_mass_count; ++electron)
      if (!is_finite_parameter_value(inverse_masses[electron]) ||
          inverse_masses[electron] <= 0.0)
        throw std::invalid_argument(
            "PsiFormer inverse masses must be finite and positive");

  const double* parameter_values = parameters_.flat_values().data();
  workspace.geometry_.update(GeometryPositionView::interleaved(
      workspace.electron_positions_.data(), workspace.electrons_));
  buildEmbedding(parameter_values, workspace);
  for (std::size_t block = 0; block < workspace.blocks_; ++block)
    applyAttentionBlock(parameter_values, block, workspace);
  buildOrbitals(parameter_values, workspace);

  const auto determinant_result = workspace.determinant_.evaluateSpatial(
      workspace.orbitals_.value.data(), workspace.orbitals_.gradient.data(),
      workspace.gradient_lanes_, workspace.orbitals_.laplacian.data(),
      workspace.laplacian_lanes_, workspace.determinant_gradient_.data(),
      workspace.determinant_lap_log_.data(), workspace.determinant_lap_ratio_.data());
  if (determinant_result.amplitude.isZero() ||
      !determinant_result.all_required_inverses_available)
    throw std::domain_error(
        "PsiFormer kinetic response is undefined at a determinant node or singular channel");

  const double cusp_value = buildCusp(parameter_values, workspace);
  for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
    workspace.output_gradient_[lane] =
        workspace.determinant_gradient_[lane] + workspace.cusp_gradient_[lane];
  for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
  {
    workspace.output_lap_log_[electron] =
        workspace.determinant_lap_log_[electron] + workspace.cusp_laplacian_[electron];
    double squared_gradient = 0.0;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double component = workspace.output_gradient_[3 * electron + dimension];
      squared_gradient += component * component;
    }
    workspace.output_lap_ratio_[electron] =
        workspace.output_lap_log_[electron] + squared_gradient;
  }

  const double logabs = determinant_result.amplitude.log_abs + cusp_value;
  if (!is_finite_parameter_value(logabs))
    throw std::runtime_error("PsiFormer kinetic executor produced a non-finite log amplitude");
  for (double value : workspace.output_gradient_)
    if (!is_finite_parameter_value(value))
      throw std::runtime_error("PsiFormer kinetic executor produced a non-finite drift");
  for (double value : workspace.output_lap_log_)
    if (!is_finite_parameter_value(value))
      throw std::runtime_error("PsiFormer kinetic executor produced a non-finite Laplacian");

  // Coordinate jets of B=d log(det-sum)/dA are the determinant-specific factors
  // needed by the lifted reverse.  They reuse the factored matrices above.
  buildDeterminantFactors(workspace);

  workspace.root_adjoint_.clear();
  workspace.root_adjoint_.value[0] = 1.0;
  reverse(parameter_values, workspace, workspace.parameter_score_);

  workspace.root_adjoint_.clear();
  const double* drift = total_log_gradient ? total_log_gradient
                                           : workspace.output_gradient_.data();
  for (std::size_t lane = 0; lane < workspace.gradient_lanes_; ++lane)
  {
    const std::size_t electron = lane / 3;
    const double inverse_mass = inverse_masses ? inverse_masses[electron] : 1.0;
    workspace.root_adjoint_.gradient[lane] = -inverse_mass * drift[lane];
  }
  for (std::size_t electron = 0; electron < workspace.laplacian_lanes_; ++electron)
  {
    const double inverse_mass = inverse_masses ? inverse_masses[electron] : 1.0;
    workspace.root_adjoint_.laplacian[electron] = -0.5 * inverse_mass;
  }
  reverse(parameter_values, workspace, workspace.kinetic_parameter_response_);

  for (double value : workspace.parameter_score_)
    if (!is_finite_parameter_value(value))
      throw std::runtime_error("PsiFormer kinetic executor produced a non-finite score");
  for (double value : workspace.kinetic_parameter_response_)
    if (!is_finite_parameter_value(value))
      throw std::runtime_error(
          "PsiFormer kinetic executor produced a non-finite kinetic parameter response");

  workspace.observed_parameter_version_ = parameters_.version();
  const double sign = determinant_result.amplitude.phase;
  return {sign,
          logabs,
          sign * std::exp(logabs),
          {workspace.output_gradient_.data(), workspace.output_gradient_.size()},
          {workspace.output_lap_log_.data(), workspace.output_lap_log_.size()},
          {workspace.output_lap_ratio_.data(), workspace.output_lap_ratio_.size()},
          {workspace.parameter_score_.data(), workspace.parameter_score_.size()},
          {workspace.kinetic_parameter_response_.data(),
           workspace.kinetic_parameter_response_.size()},
          workspace.observed_parameter_version_};
}

} // namespace pf

#endif // QMCPLUSPLUS_PSIFORMER_KINETIC_EXECUTOR_H
