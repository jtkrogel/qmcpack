//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerComplexScoreExecutor.h
 * @brief Complete G-free CPU complex log score for real canonical parameters.
 * Include after PsiFormerNative.h. No public wavefunction or kinetic admission.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_COMPLEX_SCORE_EXECUTOR_H
#define QMCPLUSPLUS_PSIFORMER_COMPLEX_SCORE_EXECUTOR_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexOrbital.h"

namespace pf
{
/// Borrow the canonical complex derivative of log(Psi) with respect to real parameters.
struct DirectComplexScoreResult
{
  qmcplusplus::psiformer::determinant::ComplexLogAmplitude amplitude;
  const qmcplusplus::psiformer::determinant::Complex* parameter_score = nullptr;
  std::size_t parameter_count = 0, parameter_version = 0, evaluation_generation = 0;
};

/// Stable executor addresses bind prepared clones to immutable metadata.
class DirectComplexScoreExecutor;

/// One real taped trunk, one complex canonical output and bounded head/reverse scratch.
class DirectComplexScoreWorkspace
{
  using C = qmcplusplus::psiformer::determinant::Complex;
public:
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
    return valid_ ? result_.parameter_version : std::numeric_limits<std::size_t>::max();
  }

  /// Borrow completed publication; caller must match parameter_version to the current model.
  const DirectComplexScoreResult& result() const
  {
    if (!valid_)
      throw std::logic_error("PsiFormer complex score unavailable");
    return result_;
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
  std::size_t generation_ = 0;
  bool valid_ = false;
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
        descriptor_{model.ne, model.cfg.nup, model.ndet, model.dim,
                    qmcplusplus::psiformer::ComplexOrbitalStructure::FULL,
                    qmcplusplus::psiformer::ComplexOrbitalPhase::NONE, {}},
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

  /// Prepare one independent clone with only the real taped trunk and complex head.
  std::unique_ptr<DirectComplexScoreWorkspace> makeWorkspace() const
  {
    return std::unique_ptr<DirectComplexScoreWorkspace>(new DirectComplexScoreWorkspace(
        this, trunk_.makeTrunkWorkspace(), descriptor_, parameter_count_));
  }

  /// Compose amplitude, analytic head/envelope/cusp seeds, and two real trunk reverses.
  DirectComplexScoreResult evaluate(DirectComplexScoreWorkspace& w) const
  {
    using namespace qmcplusplus::psiformer;
    w.invalidate();
    if (w.owner_ != this)
      throw std::invalid_argument("PsiFormer complex score workspace belongs to another executor");
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
            radii.data() + row * nuclei_, p + pi_[spin].begin + channel * nuclei_,
            p + zeta_[spin].begin + channel * nuclei_, nuclei_);
      }
    ComplexOrbitalHeadParameters heads;
    heads.up = {p + real_[0].begin, p + imaginary_[0].begin, nullptr, nullptr, real_[0].size, 0};
    heads.down = {p + real_[1].begin, p + imaginary_[1].begin, nullptr, nullptr, real_[1].size, 0};
    w.head_.evaluate(tape.features, tape.feature_count, w.envelopes_.data(), w.envelopes_.size(),
                     tape.packed_positions, 3 * n, heads);
    // Ordinary inverse-transpose seeds differentiate log(sum_d det A_d), without conjugation.
    auto amplitude = w.determinant_.evaluate(w.head_.data()).amplitude;
    w.determinant_.fillLogMatrixAdjoints(w.matrix_adjoint_.data());
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
      const auto reverse = trunk_.reverseTrunk(*w.trunk_, tape, w.feature_seed_.data(), w.feature_seed_.size());
      for (std::size_t parameter = 0; parameter < parameter_count_; ++parameter)
        w.score_[parameter] += component ? C{0, reverse.parameter_score[parameter]} :
                                          C{reverse.parameter_score[parameter], 0};
    }
    // The common real cusp contributes only to the real part of its parameter scores.
    double same_score = 0, opposite_score = 0;
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
    w.valid_ = true;
    return w.result_;
  }

private:
  /// Enforce the current full/G-free paired periodic profile and exact model dimensions.
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
          const double radius = radii[row * nuclei_ + atom], argument = p[zeta_[spin].begin + index] * radius;
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
