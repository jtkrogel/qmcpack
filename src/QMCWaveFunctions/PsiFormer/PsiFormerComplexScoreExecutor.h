//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerComplexScoreExecutor.h
 * @brief Complete CPU complex log score for real canonical parameters.
 * Include after PsiFormerNative.h. No public wavefunction or kinetic admission.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_COMPLEX_SCORE_EXECUTOR_H
#define QMCPLUSPLUS_PSIFORMER_COMPLEX_SCORE_EXECUTOR_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOccupiedHead.h"

namespace pf
{
/// Borrow the canonical complex derivative of log(Psi) with respect to real parameters.
struct DirectComplexScoreResult
{
  qmcplusplus::psiformer::determinant::ComplexLogAmplitude amplitude;
  const qmcplusplus::psiformer::determinant::Complex* parameter_score = nullptr;
  std::size_t parameter_count = 0, parameter_version = 0, evaluation_generation = 0;
  /// Present only for affine observation requests; valid until the next workspace mutation.
  const double* head_activations = nullptr;
  /// Complex head responses in determinant,row,column order, including the envelope.
  const qmcplusplus::psiformer::determinant::Complex* head_sensitivities = nullptr;
};

/// Caller owns the real P output; it must not alias targets, model or workspace storage.
struct DirectComplexOrbitalMSEArguments
{
  const qmcplusplus::psiformer::determinant::Complex* target = nullptr;
  std::size_t target_size = 0;
  double* parameter_gradient = nullptr;
  std::size_t gradient_size = 0;
};

/// Borrow occupied-gauge orbital matrices until the next workspace mutation/evaluation.
struct DirectComplexOrbitalMSEResult
{
  double loss = 0;
  const qmcplusplus::psiformer::determinant::Complex* orbital_matrices = nullptr;
  std::size_t orbital_count = 0, parameter_version = 0, evaluation_generation = 0;
};

/// Stable executor addresses bind prepared clones to immutable metadata.
class DirectComplexScoreExecutor;

/// One real taped trunk, one complex canonical output and bounded head/reverse scratch.
class DirectComplexScoreWorkspace
{
  using C = qmcplusplus::psiformer::determinant::Complex;
public:
  /// Count the existing mixed-domain VALUE tape before constructing numerical buffers.
  static std::size_t storageRequirement(const PsiFormerStorageShape& shape)
  {
    using namespace qmcplusplus::psiformer;
    const char* message = "complex score storage overflow";
    const auto matrix = checkedStorageProduct(shape.determinants,
        checkedStorageProduct(shape.electrons, shape.electrons, message), message);
    const auto features = checkedStorageProduct(shape.electrons, shape.feature_width, message);
    const auto layout = makeSpatialJetLayout(1, matrix, shape.electrons, SpatialJetMode::VALUE_ONLY);
    std::size_t bytes = scoreWorkspaceStorageRequirement(shape) -
        checkedStorageBytes<double>(checkedStorageProduct(3, matrix, message), message) -
        determinantStorageRequirement(shape.determinants, shape.electrons);
    addStorageBytes(bytes, ComplexOrbitalHeadWorkspace::storageRequirement(layout, shape.occupied_momentum_count), message);
    addStorageBytes(bytes, determinant::ComplexDeterminantWorkspace::storageRequirement(
        shape.determinants, shape.electrons), message);
    addStorageBytes(bytes, checkedStorageBytes<double>(checkedStorageSum(matrix, features, message), message), message);
    addStorageBytes(bytes, checkedStorageBytes<C>(checkedStorageSum(matrix, shape.parameter_count, message), message), message);
    return bytes;
  }

  /// Mutable prepared buffers cannot be copied between clones.
  DirectComplexScoreWorkspace(const DirectComplexScoreWorkspace&) = delete;

  /// Preserve exclusive numerical ownership.
  DirectComplexScoreWorkspace& operator=(const DirectComplexScoreWorkspace&) = delete;

  /// Transfer prepared buffers and owner identity; old borrowed views are invalidated.
  DirectComplexScoreWorkspace(DirectComplexScoreWorkspace&&) = default;

  /// Immutable head descriptors disallow move assignment.
  DirectComplexScoreWorkspace& operator=(DirectComplexScoreWorkspace&&) = delete;

  /// Invalidate the preceding score before changing any prepared coordinate.
  void setPositions(GeometryPositionView positions)
  {
    invalidate();
    trunk_->setPositions(positions);
  }

  /// Edit a single coordinate without changing any allocation.
  void setPosition(std::size_t electron, std::size_t dimension, double value)
  {
    invalidate();
    trunk_->setPosition(electron, dimension, value);
  }

  /// Return a stamp that advances even for rejected requests.
  std::size_t evaluationGeneration() const noexcept { return generation_; }

  /// Failed requests have no observed parameter version.
  std::size_t observedParameterVersion() const noexcept
  {
    return valid_ ? result_.parameter_version :
        (orbital_mse_valid_ ? orbital_mse_result_.parameter_version : std::numeric_limits<std::size_t>::max());
  }

  /// Borrow completed publication; caller must match parameter_version to the current model.
  const DirectComplexScoreResult& result() const
  {
    if (!valid_)
      throw std::logic_error("PsiFormer complex score unavailable");
    return result_;
  }

  /// MSE publication is distinct from a log-score publication and never implies one.
  const DirectComplexOrbitalMSEResult& orbitalMSEResult() const
  {
    if (!orbital_mse_valid_)
      throw std::logic_error("PsiFormer complex orbital-MSE unavailable");
    return orbital_mse_result_;
  }

  /// Account for all owned numeric buffers, including the real trunk's one P scratch.
  std::size_t vectorStorageBytes() const
  {
    std::size_t bytes = trunk_->vectorStorageBytes();
    addStorageBytes(bytes, head_.storageBytes(), "complex score bytes overflow");
    addStorageBytes(bytes, determinant_.storageBytes(), "complex score bytes overflow");
    for (const auto* buffer : {&envelopes_, &feature_seed_})
      addStorageBytes(bytes, checkedStorageBytes<double>(buffer->capacity(), "complex score bytes overflow"),
                      "complex score bytes overflow");
    for (const auto* buffer : {&matrix_adjoint_, &score_})
      addStorageBytes(bytes, checkedStorageBytes<C>(buffer->capacity(), "complex score bytes overflow"),
                      "complex score bytes overflow");
    return bytes;
  }

  /// Track all allocation identities across the two reverse passes and warmed calls.
  std::size_t storageFingerprint() const noexcept
  {
    std::size_t hash = trunk_->storageFingerprint();
    qmcplusplus::psiformer::determinant::detail::mixStorage(hash, envelopes_);
    qmcplusplus::psiformer::determinant::detail::mixStorage(hash, feature_seed_);
    qmcplusplus::psiformer::determinant::detail::mixStorage(hash, matrix_adjoint_);
    qmcplusplus::psiformer::determinant::detail::mixStorage(hash, score_);
    hash ^= head_.storageFingerprint();
    hash *= 1099511628211ULL;
    return hash ^ determinant_.storageFingerprint();
  }

private:
  friend class DirectComplexScoreExecutor;

  /// Prepare a VALUE head and inverse reduction without constructing a real head.
  DirectComplexScoreWorkspace(const DirectComplexScoreExecutor* owner,
      std::unique_ptr<DirectScoreTrunkWorkspace> trunk,
      const qmcplusplus::psiformer::ComplexOrbitalHeadDescriptor& descriptor, std::size_t parameters)
      : owner_(owner), trunk_(std::move(trunk)),
        features_(qmcplusplus::psiformer::makeSpatialJetLayout(1,
            checkedStorageProduct(descriptor.electrons, descriptor.feature_width, "score feature overflow"),
            descriptor.electrons, qmcplusplus::psiformer::SpatialJetMode::VALUE_ONLY)),
        orbitals_(qmcplusplus::psiformer::makeSpatialJetLayout(1,
            checkedStorageProduct(descriptor.determinants,
                checkedStorageProduct(descriptor.electrons, descriptor.electrons, "score matrix overflow"),
                "score orbital overflow"), descriptor.electrons,
            qmcplusplus::psiformer::SpatialJetMode::VALUE_ONLY)),
        envelopes_(orbitals_.element_count), feature_seed_(features_.element_count),
        head_(descriptor, features_, orbitals_), determinant_(descriptor.determinants, descriptor.electrons),
        matrix_adjoint_(orbitals_.element_count), score_(parameters)
  {}

  /// Reject access to prior publication before any mutation or attempted evaluation.
  void invalidate()
  {
    valid_ = false;
    orbital_mse_valid_ = false;
    if (generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("PsiFormer complex score generation exhausted");
    ++generation_;
  }

  const DirectComplexScoreExecutor* owner_;
  std::unique_ptr<DirectScoreTrunkWorkspace> trunk_;
  qmcplusplus::psiformer::SpatialJetLayout features_, orbitals_;
  std::vector<double> envelopes_, feature_seed_;
  qmcplusplus::psiformer::ComplexOrbitalHeadWorkspace head_;
  qmcplusplus::psiformer::determinant::ComplexDeterminantWorkspace determinant_;
  std::vector<C> matrix_adjoint_, score_;
  DirectComplexScoreResult result_;
  DirectComplexOrbitalMSEResult orbital_mse_result_;
  std::size_t generation_ = 0;
  bool valid_ = false;
  bool orbital_mse_valid_ = false;
};

/** Analytic complex log score through one real tape and two real reverse passes.
 * The head and determinant contractions are ordinary complex, not Hermitian.
 * A real canonical parameter may have a nonzero phase response. Common outer
 * twist is parameter-independent and excluded from this numerical owner.
 */
class DirectComplexScoreExecutor
{
  using C = qmcplusplus::psiformer::determinant::Complex;
public:
  /// Resolve each real/imaginary head and envelope interval once, retaining no value pointers.
  DirectComplexScoreExecutor(const PsiFormer& model,
      const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan)
      : parameters_(model.p), trunk_(model, admit(model, plan), DirectScorePreparation::REAL_TRUNK_ONLY),
        descriptor_(qmcplusplus::psiformer::occupiedFullHeadDescriptor(plan)),
        nuclei_(model.cfg.nuclei.shape[0]), parameter_count_(plan.parameterCount())
  {
    using namespace qmcplusplus::psiformer;
    for (std::size_t spin = 0; spin < 2; ++spin)
    {
      const auto role = spin ? ParameterRole::BACKFLOW_DOWN_WEIGHT : ParameterRole::BACKFLOW_UP_WEIGHT;
      real_[spin] = interval(plan.parameter(role));
      imaginary_[spin] = interval(plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG));
      pi_[spin] = interval(plan.parameter(spin ? ParameterRole::ENVELOPE_PI_DOWN : ParameterRole::ENVELOPE_PI_UP));
      zeta_[spin] = interval(plan.parameter(spin ? ParameterRole::ENVELOPE_ZETA_DOWN : ParameterRole::ENVELOPE_ZETA_UP));
    }
    if (plan.hasParameter(ParameterRole::CUSP_SAME_ALPHA))
      same_ = interval(plan.parameter(ParameterRole::CUSP_SAME_ALPHA));
    opposite_ = interval(plan.parameter(ParameterRole::CUSP_OPPOSITE_ALPHA));
  }

  /// Executor identities must remain stable throughout all prepared clone lifetimes.
  DirectComplexScoreExecutor(const DirectComplexScoreExecutor&) = delete;

  /// Prepared identities cannot be transferred by copy assignment.
  DirectComplexScoreExecutor& operator=(const DirectComplexScoreExecutor&) = delete;

  /// Moving would invalidate every owner-address token.
  DirectComplexScoreExecutor(DirectComplexScoreExecutor&&) = delete;

  /// Preserve stable identity under assignment as well.
  DirectComplexScoreExecutor& operator=(DirectComplexScoreExecutor&&) = delete;

  /// Include fixed-ion and occupied-column metadata in external lane accounting.
  std::size_t numericMetadataStorageBytes() const
  {
    return checkedStorageSum(trunk_.numericMetadataStorageBytes(),
        checkedStorageBytes<std::array<double, 3>>(descriptor_.reciprocal_momenta.capacity(),
            "complex score momentum metadata overflow"), "complex score metadata overflow");
  }

  /// Track fixed-ion metadata separately from the mutable tape allocation identity.
  std::size_t numericMetadataStorageFingerprint() const noexcept
  {
    auto hash = trunk_.numericMetadataStorageFingerprint();
    if (!descriptor_.reciprocal_momenta.empty())
      qmcplusplus::psiformer::determinant::detail::mixStorage(hash, descriptor_.reciprocal_momenta);
    return hash;
  }

  /// Prepare one independent clone with only the real taped trunk and complex head.
  std::unique_ptr<DirectComplexScoreWorkspace> makeWorkspace() const
  {
    return std::unique_ptr<DirectComplexScoreWorkspace>(new DirectComplexScoreWorkspace(
        this, trunk_.makeTrunkWorkspace(), descriptor_, parameter_count_));
  }

  /// Compose amplitude, analytic head/envelope/cusp seeds, and two real trunk reverses.
  DirectComplexScoreResult evaluate(DirectComplexScoreWorkspace& w,
      DirectAffineObservationSink* real_observer = nullptr,
      DirectAffineObservationSink* imaginary_observer = nullptr) const
  {
    using namespace qmcplusplus::psiformer;
    w.invalidate();
    if (w.owner_ != this)
      throw std::invalid_argument("PsiFormer complex score workspace belongs to another executor");
    if ((real_observer == nullptr) != (imaginary_observer == nullptr))
      throw std::invalid_argument("PsiFormer complex affine observations require both response channels");
    const auto tape = trunk_.evaluateTrunk(*w.trunk_);
    const auto& identities = tape.geometry->electronPairs();
    const auto& electron_radii = tape.geometry->electronElectronPairs().distances();
    for (std::size_t pair = 0; pair < identities.size(); ++pair)
      if ((identities[pair].first < descriptor_.spin_up) ==
          (identities[pair].second < descriptor_.spin_up) && electron_radii[pair] == 0)
        throw std::domain_error("PsiFormer complex score unavailable at a same-spin node");
    const double* p = parameters_.flat_values().data();
    const std::size_t n = descriptor_.electrons, channels = descriptor_.determinants * n;
    const auto& radii = tape.geometry->electronNucleusPairs().distances();
    // Canonical channels are (d,column); matrices are determinant-major (d,row,column).
    // Build real envelopes and the paired affine head directly from this one tape.
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t channel = 0; channel < channels; ++channel)
      {
        const std::size_t spin = row < descriptor_.spin_up ? 0 : 1;
        w.envelopes_[(channel / n * n + row) * n + channel % n] = realOrbitalEnvelopeValue(
            *tape.geometry, row, p + pi_[spin].begin + channel * nuclei_,
            p + zeta_[spin].begin + channel * nuclei_, nuclei_, trunk_.plan().environment().periodic_profile);
      }
    ComplexOrbitalHeadParameters heads;
    heads.up = {p + real_[0].begin, p + imaginary_[0].begin, nullptr, nullptr, real_[0].size, 0};
    heads.down = {p + real_[1].begin, p + imaginary_[1].begin, nullptr, nullptr, real_[1].size, 0};
    w.head_.evaluate(tape.features, tape.feature_count, w.envelopes_.data(), w.envelopes_.size(),
                     tape.packed_positions, 3 * n, heads);
    // Ordinary inverse-transpose seeds differentiate log(sum_d det A_d), without conjugation.
    auto amplitude = w.determinant_.evaluate(w.head_.data()).amplitude;
    w.determinant_.fillLogMatrixAdjoints(w.matrix_adjoint_.data());
    pullBackOccupiedPhase(descriptor_, w.orbitals_, tape.packed_positions, 3 * n,
                           w.matrix_adjoint_.data(), w.matrix_adjoint_.size());
    std::fill(w.score_.begin(), w.score_.end(), C{});
    reverseHead(w, tape, p);
    // Two real feature cotangents reuse the identical immutable forward tape and P scratch.
    for (std::size_t component = 0; component < 2; ++component)
    {
      std::fill(w.feature_seed_.begin(), w.feature_seed_.end(), 0.0);
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t channel = 0; channel < channels; ++channel)
        {
          const std::size_t spin = row < descriptor_.spin_up ? 0 : 1;
          const std::size_t element = (channel / n * n + row) * n + channel % n;
          const C seed = w.matrix_adjoint_[element] * w.envelopes_[element];
          for (std::size_t feature = 0; feature < descriptor_.feature_width; ++feature)
          {
            const std::size_t index = feature * channels + channel;
            const C contribution = seed * C{p[real_[spin].begin + index], p[imaginary_[spin].begin + index]};
            w.feature_seed_[row * descriptor_.feature_width + feature] +=
                component ? contribution.imag() : contribution.real();
          }
        }
      const auto reverse = trunk_.reverseTrunk(*w.trunk_, tape, w.feature_seed_.data(), w.feature_seed_.size(),
                                               component ? imaginary_observer : real_observer);
      for (std::size_t parameter = 0; parameter < parameter_count_; ++parameter)
        w.score_[parameter] += component ? C{0, reverse.parameter_score[parameter]} :
                                          C{reverse.parameter_score[parameter], 0};
    }
    // The common real cusp contributes only to the real part of its parameter scores.
    double same_score = 0, opposite_score = 0;
    const auto& profile = trunk_.plan().environment().periodic_profile;
    if (!profile || profile->electron_electron_cusp)
      amplitude.log_abs += realCuspValueAndReverse(*tape.geometry, descriptor_.spin_up,
          same_.size ? p[same_.begin] : 1.0, p[opposite_.begin], same_.size ? &same_score : nullptr,
          opposite_score, 1.0);
    if (same_.size)
      w.score_[same_.begin] += same_score;
    w.score_[opposite_.begin] += opposite_score;
    // Publish only after every amplitude/score component has passed finite validation.
    if (!determinant::isFiniteReal(amplitude.log_abs))
      throw std::overflow_error("PsiFormer complex score amplitude overflow");
    for (C value : w.score_)
      if (!determinant::isFiniteReal(value.real()) || !determinant::isFiniteReal(value.imag()))
        throw std::overflow_error("PsiFormer complex score response overflow");
    w.result_ = {amplitude, w.score_.data(), parameter_count_, tape.parameter_version, w.generation_};
    if (real_observer)
    {
      // The matrix adjoints have completed every reverse product and can now hold
      // the head response without adding another orbital-sized numerical buffer.
      for (std::size_t element = 0; element < w.matrix_adjoint_.size(); ++element)
      {
        w.matrix_adjoint_[element] *= w.envelopes_[element];
        if (!determinant::isFiniteComplex(w.matrix_adjoint_[element]))
          throw std::overflow_error("PsiFormer complex affine head response overflow");
      }
      w.result_.head_activations = tape.features;
      w.result_.head_sensitivities = w.matrix_adjoint_.data();
    }
    w.valid_ = true;
    return w.result_;
  }

  /** Raw complex orbital loss with the real executor's per-spin normalization.
   * Ordinary matrix cotangents are 2*scale*conj(A-T); take the real response
   * for real canonical parameters. No determinant inverse, cusp or node gate.
   * No new workspace vectors: caller output is copied only after all checks.
   */
  DirectComplexOrbitalMSEResult evaluateOrbitalMSE(DirectComplexScoreWorkspace& w,
                                                 DirectComplexOrbitalMSEArguments arguments) const
  {
    using namespace qmcplusplus::psiformer;
    w.invalidate();
    if (w.owner_ != this)
      throw std::invalid_argument("PsiFormer complex orbital-MSE workspace belongs to another executor");
    if (arguments.target_size != w.orbitals_.element_count || !arguments.target ||
        arguments.gradient_size != parameter_count_ || !arguments.parameter_gradient)
      throw std::invalid_argument("PsiFormer complex orbital-MSE target/gradient extent mismatch");
    const auto target_bytes = checkedStorageBytes<C>(arguments.target_size, "complex MSE target bytes overflow");
    const auto output_bytes = checkedStorageBytes<double>(parameter_count_, "complex MSE output bytes overflow");
    const auto rejects_output_alias = [&](const void* pointer, std::size_t bytes) {
      if (memoryOverlap(arguments.parameter_gradient, output_bytes, pointer, bytes))
        throw std::invalid_argument("PsiFormer complex orbital-MSE output aliases borrowed storage");
    };
    rejects_output_alias(arguments.target, target_bytes);
    rejects_output_alias(parameters_.flat_values().data(), output_bytes);
    rejects_output_alias(w.head_.outputStorageAddress(), target_bytes);
    rejects_output_alias(w.score_.data(), checkedStorageBytes<C>(w.score_.size(), "complex MSE score bytes"));
    rejects_output_alias(w.matrix_adjoint_.data(), target_bytes);
    rejects_output_alias(w.feature_seed_.data(), checkedStorageBytes<double>(w.feature_seed_.size(), "complex MSE seeds"));
    rejects_output_alias(w.envelopes_.data(), checkedStorageBytes<double>(w.envelopes_.size(), "complex MSE envelopes"));
    if (memoryOverlap(arguments.target, target_bytes, w.head_.outputStorageAddress(), target_bytes))
      throw std::invalid_argument("PsiFormer complex orbital-MSE target aliases mutable raw matrices");
    for (std::size_t i = 0; i < arguments.target_size; ++i)
      if (!determinant::isFiniteReal(arguments.target[i].real()) || !determinant::isFiniteReal(arguments.target[i].imag()))
        throw std::invalid_argument("PsiFormer complex orbital-MSE target is nonfinite");

    const auto tape = trunk_.evaluateTrunk(*w.trunk_);
    const double* p = parameters_.flat_values().data();
    const std::size_t n = descriptor_.electrons, channels = descriptor_.determinants * n;
    const auto& radii = tape.geometry->electronNucleusPairs().distances();
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t channel = 0; channel < channels; ++channel)
      {
        const std::size_t spin = row < descriptor_.spin_up ? 0 : 1;
        w.envelopes_[(channel / n * n + row) * n + channel % n] = realOrbitalEnvelopeValue(
            *tape.geometry, row, p + pi_[spin].begin + channel * nuclei_,
            p + zeta_[spin].begin + channel * nuclei_, nuclei_, trunk_.plan().environment().periodic_profile);
      }
    ComplexOrbitalHeadParameters heads;
    heads.up = {p + real_[0].begin, p + imaginary_[0].begin, nullptr, nullptr, real_[0].size, 0};
    heads.down = {p + real_[1].begin, p + imaginary_[1].begin, nullptr, nullptr, real_[1].size, 0};
    w.head_.evaluate(tape.features, tape.feature_count, w.envelopes_.data(), w.envelopes_.size(),
                     tape.packed_positions, 3 * n, heads);
    double loss = 0;
    for (std::size_t determinant = 0; determinant < descriptor_.determinants; ++determinant)
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t column = 0; column < n; ++column)
        {
          const std::size_t index = (determinant * n + row) * n + column;
          const std::size_t spin_rows = row < descriptor_.spin_up ? descriptor_.spin_up : n - descriptor_.spin_up;
          const double scale = 1.0 / static_cast<double>(descriptor_.determinants * spin_rows * n);
          const C residual = w.head_.data()[index] - arguments.target[index];
          loss += scale * std::norm(residual);
          w.matrix_adjoint_[index] = 2.0 * scale * std::conj(residual);
        }
    pullBackOccupiedPhase(descriptor_, w.orbitals_, tape.packed_positions, 3 * n,
                           w.matrix_adjoint_.data(), w.matrix_adjoint_.size());
    std::fill(w.score_.begin(), w.score_.end(), C{});
    reverseHead(w, tape, p);
    std::fill(w.feature_seed_.begin(), w.feature_seed_.end(), 0.0);
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t channel = 0; channel < channels; ++channel)
      {
        const std::size_t spin = row < descriptor_.spin_up ? 0 : 1;
        const std::size_t element = (channel / n * n + row) * n + channel % n;
        const C seed = w.matrix_adjoint_[element] * w.envelopes_[element];
        for (std::size_t feature = 0; feature < descriptor_.feature_width; ++feature)
        {
          const std::size_t index = feature * channels + channel;
          w.feature_seed_[row * descriptor_.feature_width + feature] +=
              (seed * C{p[real_[spin].begin + index], p[imaginary_[spin].begin + index]}).real();
        }
      }
    const auto reverse = trunk_.reverseTrunk(*w.trunk_, tape, w.feature_seed_.data(), w.feature_seed_.size());
    for (std::size_t parameter = 0; parameter < parameter_count_; ++parameter)
      w.score_[parameter] += reverse.parameter_score[parameter];
    if (!determinant::isFiniteReal(loss))
      throw std::overflow_error("PsiFormer complex orbital-MSE loss overflow");
    for (C value : w.score_)
      if (!determinant::isFiniteReal(value.real()) || !determinant::isFiniteReal(value.imag()))
        throw std::overflow_error("PsiFormer complex orbital-MSE response overflow");
    for (std::size_t parameter = 0; parameter < parameter_count_; ++parameter)
      arguments.parameter_gradient[parameter] = w.score_[parameter].real();
    w.orbital_mse_result_ = {loss, w.head_.data(), w.orbitals_.element_count, tape.parameter_version, w.generation_};
    w.orbital_mse_valid_ = true;
    return w.orbital_mse_result_;
  }

private:
  /// Bounded byte ranges reject writable aliases without a registry or allocations.
  static bool memoryOverlap(const void* left, std::size_t left_bytes, const void* right, std::size_t right_bytes)
  {
    const auto a = reinterpret_cast<std::uintptr_t>(left), b = reinterpret_cast<std::uintptr_t>(right);
    if (left_bytes > std::numeric_limits<std::uintptr_t>::max() - a ||
        right_bytes > std::numeric_limits<std::uintptr_t>::max() - b)
      throw std::invalid_argument("PsiFormer complex orbital-MSE address range overflow");
    return left_bytes && right_bytes && a < b + right_bytes && b < a + left_bytes;
  }

  /// Enforce the full paired periodic profile and exact model dimensions.
  static const qmcplusplus::psiformer::PsiFormerExecutionPlan& admit(const PsiFormer& model,
      const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan)
  {
    using namespace qmcplusplus::psiformer;
    const auto& shape = plan.modelShape();
    if (plan.environment().head_representation != HeadRepresentation::PAIRED_REAL_COMPLEX ||
        model.ne != shape.electrons() || model.ndet != shape.determinants || model.dim != shape.feature_dimension ||
        model.cfg.nup != shape.spin_up_electrons || model.cfg.ndown != shape.spin_down_electrons ||
        model.cfg.nuclei.shape[0] != shape.nuclei || model.heads != shape.attention_heads ||
        model.blocks != shape.attention_blocks)
      throw std::invalid_argument("PsiFormer complex score requires a matching paired periodic plan");
    return plan;
  }

  /// Copy a canonical interval without retaining a mutable parameter value pointer.
  static DirectParameterTensor interval(const qmcplusplus::psiformer::ParameterTensorDescriptor& descriptor)
  {
    return {descriptor.begin, descriptor.size()};
  }

  /// Reverse true affine projections and envelopes with ordinary complex matrix seeds.
  void reverseHead(DirectComplexScoreWorkspace& w, const DirectScoreTrunkResultView& tape,
                   const double* p) const
  {
    const std::size_t n = descriptor_.electrons, channels = descriptor_.determinants * n;
    const auto& radii = tape.geometry->electronNucleusPairs().distances();
    const C* projection = w.head_.affineProjectionData();
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t channel = 0; channel < channels; ++channel)
      {
        const std::size_t spin = row < descriptor_.spin_up ? 0 : 1;
        const std::size_t element = (channel / n * n + row) * n + channel % n;
        const C head_seed = w.matrix_adjoint_[element] * w.envelopes_[element];
        const C envelope_seed = w.matrix_adjoint_[element] * projection[element];
        for (std::size_t feature = 0; feature < descriptor_.feature_width; ++feature)
        {
          const std::size_t index = feature * channels + channel;
          const C contribution = head_seed * tape.features[row * descriptor_.feature_width + feature];
          w.score_[real_[spin].begin + index] += contribution;
          w.score_[imaginary_[spin].begin + index] += C{0, 1} * contribution;
        }
        for (std::size_t atom = 0; atom < nuclei_; ++atom)
        {
          const std::size_t index = channel * nuclei_ + atom;
          const double radius = qmcplusplus::psiformer::periodic_feature::selectedRadius(tape.geometry->electronNucleusPairs(),
              row * nuclei_ + atom, tape.geometry->electronIonBoundary(),
              trunk_.plan().environment().periodic_profile).value;
          const double argument = p[zeta_[spin].begin + index] * radius;
          const double decay = std::exp(-std::abs(argument));
          const double sign = argument > 0 ? 1.0 : (argument < 0 ? -1.0 : 0.0);
          w.score_[pi_[spin].begin + index] += envelope_seed * decay;
          w.score_[zeta_[spin].begin + index] -=
              envelope_seed * p[pi_[spin].begin + index] * decay * sign * radius;
        }
      }
  }

  const Parameters& parameters_;
  DirectScoreExecutor trunk_;
  const qmcplusplus::psiformer::ComplexOrbitalHeadDescriptor descriptor_;
  const std::size_t nuclei_, parameter_count_;
  std::array<DirectParameterTensor, 2> real_, imaginary_, pi_, zeta_;
  DirectParameterTensor same_, opposite_;
};
} // namespace pf
#endif
