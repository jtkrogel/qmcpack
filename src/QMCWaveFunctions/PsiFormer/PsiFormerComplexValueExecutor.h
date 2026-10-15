//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerComplexValueExecutor.h
 * @brief Complete CPU complex periodic VALUE composition over the shared real trunk.
 *
 * Include after PsiFormerNative.h, following its one-translation-unit restriction.
 * This numerical owner is full-determinant and strictly free of inner reciprocal
 * phases. The common outer twist belongs to the future wavefunction adapter.
 * Public XML, spatial/parameter derivatives and accelerator admission are unchanged.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_COMPLEX_VALUE_EXECUTOR_H
#define QMCPLUSPLUS_PSIFORMER_COMPLEX_VALUE_EXECUTOR_H

#include "PsiFormerValueExecutor.h"
#include "PsiFormerComplexOrbital.h"

namespace pf
{
/// Physical core amplitude, reduction diagnostics and publication stamps for VALUE.
struct DirectComplexValueResult
{
  qmcplusplus::psiformer::determinant::ComplexLogAmplitude amplitude;
  std::size_t singular_channels = 0;
  double cancellation_ratio = 0;
  std::size_t parameter_version = 0;
  std::size_t evaluation_generation = 0;
};

/// Stable executor address identifies the owner of each prepared clone workspace.
class DirectComplexValueExecutor;

/** Clone-local real trunk, real envelopes and complex head/determinant storage.
 * No legacy real orbital matrices or real determinant workspace are constructed.
 * The existing complex determinant primitive retains inverse scratch even though
 * VALUE omits inverse solves. All buffers are bounded and prepared before evaluate.
 */
class DirectComplexValueWorkspace
{
public:
  /// Replace the real head by equal-sized envelopes and mixed-domain algebra, without allocation.
  static std::size_t storageRequirement(const PsiFormerStorageShape& shape)
  {
    using namespace qmcplusplus::psiformer;
    const auto orbitals = checkedStorageProduct(shape.determinants,
        checkedStorageProduct(shape.electrons, shape.electrons, "complex matrix extent overflow"),
        "complex orbital extent overflow");
    const auto layout = makeSpatialJetLayout(1, orbitals, shape.electrons, SpatialJetMode::VALUE_ONLY);
    std::size_t bytes = valueWorkspaceStorageRequirement(shape) -
        determinantStorageRequirement(shape.determinants, shape.electrons);
    addStorageBytes(bytes, ComplexOrbitalHeadWorkspace::storageRequirement(layout), "complex value bytes overflow");
    addStorageBytes(bytes, determinant::ComplexDeterminantWorkspace::storageRequirement(
        shape.determinants, shape.electrons), "complex value bytes overflow");
    return bytes;
  }

  /// Mutable prepared buffers cannot be shared by copying.
  DirectComplexValueWorkspace(const DirectComplexValueWorkspace&) = delete;

  /// Preserve exclusive clone ownership of the numerical buffers.
  DirectComplexValueWorkspace& operator=(const DirectComplexValueWorkspace&) = delete;

  /// Transfer the complete clone-local preparation and owner identity.
  DirectComplexValueWorkspace(DirectComplexValueWorkspace&&) = default;

  /// Const head metadata does not support move assignment.
  DirectComplexValueWorkspace& operator=(DirectComplexValueWorkspace&&) = delete;

  /// Invalidate the prior publication before editing the existing real input buffer.
  void setPositions(GeometryPositionView positions)
  {
    invalidate();
    trunk_->setPositions(positions);
  }

  /// Change one input coordinate without growing any preparation.
  void setPosition(std::size_t electron, std::size_t dimension, double coordinate)
  {
    invalidate();
    trunk_->setPosition(electron, dimension, coordinate);
  }

  /// Return the current input/evaluation generation, including failed requests.
  std::size_t evaluationGeneration() const noexcept { return generation_; }

  /// A failed operation has no observed parameter version.
  std::size_t observedParameterVersion() const noexcept
  {
    return valid_ ? result_.parameter_version : std::numeric_limits<std::size_t>::max();
  }

  /// Expose only a complete published VALUE result; callers also match model version.
  const DirectComplexValueResult& result() const
  {
    if (!valid_)
      throw std::logic_error("PsiFormer complex value result unavailable");
    return result_;
  }

  /// Count all explicitly owned numeric buffers, excluding shared model parameters.
  std::size_t vectorStorageBytes() const
  {
    std::size_t bytes = trunk_->vectorStorageBytes();
    addStorageBytes(bytes, checkedStorageBytes<double>(envelopes_.capacity(),
        "complex value envelope bytes overflow"), "complex value workspace bytes overflow");
    addStorageBytes(bytes, head_.storageBytes(), "complex value workspace bytes overflow");
    addStorageBytes(bytes, determinant_.storageBytes(), "complex value workspace bytes overflow");
    return bytes;
  }

  /// Fingerprint every prepared allocation set for warmed no-growth regression checks.
  std::size_t storageFingerprint() const noexcept
  {
    std::size_t hash = trunk_->storageFingerprint();
    qmcplusplus::psiformer::determinant::detail::mixStorage(hash, envelopes_);
    hash ^= head_.storageFingerprint();
    hash *= 1099511628211ULL;
    hash ^= determinant_.storageFingerprint();
    hash *= 1099511628211ULL;
    return hash;
  }

private:
  friend class DirectComplexValueExecutor;

  /// Prepare one G-free full head with compact value planes matching the determinant.
  DirectComplexValueWorkspace(const DirectComplexValueExecutor* owner,
      std::unique_ptr<DirectValueTrunkWorkspace> trunk,
      const qmcplusplus::psiformer::ComplexOrbitalHeadDescriptor& descriptor)
      : owner_(owner), trunk_(std::move(trunk)),
        features_(qmcplusplus::psiformer::makeSpatialJetLayout(1,
            checkedStorageProduct(descriptor.electrons, descriptor.feature_width, "complex feature extent overflow"),
            descriptor.electrons, qmcplusplus::psiformer::SpatialJetMode::VALUE_ONLY)),
        orbitals_(qmcplusplus::psiformer::makeSpatialJetLayout(1,
            checkedStorageProduct(descriptor.determinants,
                checkedStorageProduct(descriptor.electrons, descriptor.electrons, "complex matrix extent overflow"),
                "complex orbital extent overflow"), descriptor.electrons,
            qmcplusplus::psiformer::SpatialJetMode::VALUE_ONLY)),
        envelopes_(orbitals_.element_count), head_(descriptor, features_, orbitals_),
        determinant_(descriptor.determinants, descriptor.electrons)
  {}

  /// Invalidate before mutation or execution, including a rejected foreign-owner call.
  void invalidate()
  {
    valid_ = false;
    if (generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("PsiFormer complex value generation exhausted");
    ++generation_;
  }

  const DirectComplexValueExecutor* owner_;
  std::unique_ptr<DirectValueTrunkWorkspace> trunk_;
  qmcplusplus::psiformer::SpatialJetLayout features_, orbitals_;
  std::vector<double> envelopes_;
  qmcplusplus::psiformer::ComplexOrbitalHeadWorkspace head_;
  qmcplusplus::psiformer::determinant::ComplexDeterminantWorkspace determinant_;
  DirectComplexValueResult result_;
  std::size_t generation_ = 0;
  bool valid_ = false;
};

/** Complete G-free complex periodic amplitude using one existing real neural trunk.
 * Resolve canonical parameter offsets once; parameter updates retain the existing
 * real store and version protocol. This owner exposes VALUE only, without claiming
 * the public wavefunction, training or accelerator capabilities are qualified.
 */
class DirectComplexValueExecutor
{
public:
  /// Admit only the approved paired-real, full/no-inner-phase periodic schema.
  DirectComplexValueExecutor(const PsiFormer& model,
      const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan)
      : parameters_(model.p), trunk_(model, requirePairedPlan(model, plan), DirectValuePreparation::REAL_TRUNK_ONLY),
        descriptor_{model.ne, model.cfg.nup, model.ndet, model.dim,
                    qmcplusplus::psiformer::ComplexOrbitalStructure::FULL,
                    qmcplusplus::psiformer::ComplexOrbitalPhase::NONE, {}}
  {
    using qmcplusplus::psiformer::HeadTensorComponent;
    using qmcplusplus::psiformer::NO_ATTENTION_BLOCK;
    using qmcplusplus::psiformer::ParameterRole;
    for (std::size_t spin = 0; spin < 2; ++spin)
    {
      const auto role = spin ? ParameterRole::BACKFLOW_DOWN_WEIGHT : ParameterRole::BACKFLOW_UP_WEIGHT;
      real_[spin] = interval(plan.parameter(role));
      imaginary_[spin] = interval(plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG));
      pi_[spin] = interval(plan.parameter(spin ? ParameterRole::ENVELOPE_PI_DOWN : ParameterRole::ENVELOPE_PI_UP));
      zeta_[spin] = interval(plan.parameter(spin ? ParameterRole::ENVELOPE_ZETA_DOWN : ParameterRole::ENVELOPE_ZETA_UP));
    }
  }

  /// Address-based workspace ownership requires a stable executor identity.
  DirectComplexValueExecutor(const DirectComplexValueExecutor&) = delete;

  /// An existing executor cannot acquire another owner's prepared workspace identity.
  DirectComplexValueExecutor& operator=(const DirectComplexValueExecutor&) = delete;

  /// Moving would invalidate every workspace's owner token, so it is disallowed.
  DirectComplexValueExecutor(DirectComplexValueExecutor&&) = delete;

  /// Preserve owner-token stability for the executor's complete lifetime.
  DirectComplexValueExecutor& operator=(DirectComplexValueExecutor&&) = delete;

  /// Prepare one independent clone without any complete real-head allocation.
  std::unique_ptr<DirectComplexValueWorkspace> makeWorkspace() const
  {
    return std::unique_ptr<DirectComplexValueWorkspace>(
        new DirectComplexValueWorkspace(this, trunk_.makeTrunkWorkspace(), descriptor_));
  }

  /// Compose real trunk -> real envelope -> complex head/determinants -> real cusp.
  DirectComplexValueResult evaluate(DirectComplexValueWorkspace& workspace) const
  {
    workspace.invalidate();
    if (workspace.owner_ != this)
      throw std::invalid_argument("PsiFormer complex value workspace belongs to a different executor");
    const auto trunk = trunk_.evaluateTrunk(*workspace.trunk_);
    const double* parameters = parameters_.flat_values().data();
    const auto& distances = trunk.geometry->electronNucleusPairs().distances();
    const std::size_t nuclei = trunk_.trunkLayout()->nucleusCount();
    const std::size_t n = descriptor_.electrons, d_count = descriptor_.determinants;
    for (std::size_t row = 0; row < n; ++row)
    {
      const std::size_t spin = row < descriptor_.spin_up ? 0 : 1;
      for (std::size_t d = 0; d < d_count; ++d)
        for (std::size_t orbital = 0; orbital < n; ++orbital)
        {
          const std::size_t channel = d * n + orbital;
          workspace.envelopes_[(d * n + row) * n + orbital] = realOrbitalEnvelopeValue(
              distances.data() + row * nuclei, parameters + pi_[spin].begin + channel * nuclei,
              parameters + zeta_[spin].begin + channel * nuclei, nuclei);
        }
    }
    qmcplusplus::psiformer::ComplexOrbitalHeadParameters heads;
    heads.up = {parameters + real_[0].begin, parameters + imaginary_[0].begin,
                nullptr, nullptr, real_[0].size, 0};
    heads.down = {parameters + real_[1].begin, parameters + imaginary_[1].begin,
                  nullptr, nullptr, real_[1].size, 0};
    workspace.head_.evaluate(trunk.features, workspace.features_.element_count,
        workspace.envelopes_.data(), workspace.envelopes_.size(), trunk.packed_positions,
        3 * n, heads);
    const auto determinant = workspace.determinant_.evaluate(workspace.head_.data(), nullptr, false);
    auto amplitude = determinant.amplitude;
    // Preserve the exact same-spin physical node contract, including lattice images.
    const auto& pairs = trunk.geometry->electronPairs();
    const auto& electron_distances = trunk.geometry->electronElectronPairs().distances();
    for (std::size_t pair = 0; pair < pairs.size(); ++pair)
      if ((pairs[pair].first < descriptor_.spin_up) == (pairs[pair].second < descriptor_.spin_up) &&
          electron_distances[pair] == 0)
        amplitude = {};
    if (!amplitude.isZero())
    {
      amplitude.log_abs += trunk.cusp;
      if (!is_finite_parameter_value(amplitude.log_abs))
        throw std::overflow_error("PsiFormer complex value logarithm overflowed");
    }
    workspace.result_ = {amplitude, determinant.singular_channels,
        amplitude.isZero() ? 0 : determinant.cancellation_ratio, trunk.parameter_version, workspace.generation_};
    workspace.valid_ = true;
    return workspace.result_;
  }

private:
  /// Check representation admission before resolving or preparing a complex owner.
  static const qmcplusplus::psiformer::PsiFormerExecutionPlan& requirePairedPlan(
      const PsiFormer& model, const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan)
  {
    if (plan.environment().head_representation != qmcplusplus::psiformer::HeadRepresentation::PAIRED_REAL_COMPLEX)
      throw std::invalid_argument("PsiFormer complex value requires explicit paired-real head metadata");
    const auto& shape = plan.modelShape();
    if (model.ne != shape.electrons() || model.cfg.nup != shape.spin_up_electrons ||
        model.cfg.ndown != shape.spin_down_electrons || model.cfg.nuclei.shape[0] != shape.nuclei ||
        model.ndet != shape.determinants || model.dim != shape.feature_dimension ||
        model.heads != shape.attention_heads || model.blocks != shape.attention_blocks)
      throw std::invalid_argument("PsiFormer complex value model and plan dimensions disagree");
    return plan;
  }

  /// Copy one immutable real canonical interval without retaining a value pointer.
  static DirectParameterTensor interval(const qmcplusplus::psiformer::ParameterTensorDescriptor& descriptor)
  {
    return {descriptor.begin, descriptor.size()};
  }

  const Parameters& parameters_;
  DirectValueExecutor trunk_;
  const qmcplusplus::psiformer::ComplexOrbitalHeadDescriptor descriptor_;
  std::array<DirectParameterTensor, 2> real_, imaginary_, pi_, zeta_;
};
} // namespace pf
#endif
