//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerComplexSpatialExecutor.h
 * @brief CPU complex FULL/ACTIVE derivatives over the shared real spatial trunk.
 * Include after PsiFormerNative.h. No public wavefunction or outer twist is admitted.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_COMPLEX_SPATIAL_EXECUTOR_H
#define QMCPLUSPLUS_PSIFORMER_COMPLEX_SPATIAL_EXECUTOR_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "PsiFormerOccupiedHead.h"

namespace pf
{
/// Share explicit physical-row radial envelope jets across spatial and kinetic owners.
inline void fillRealOrbitalEnvelopeJets(const PsiFormerGeometryCache& geometry,
  const qmcplusplus::psiformer::SpatialJetLayout& orbitals, std::size_t spin_up,
  std::size_t nuclei, const double* p, const std::array<DirectParameterTensor, 2>& pi,
  const std::array<DirectParameterTensor, 2>& zeta, std::size_t active, double* output)
{
  std::fill_n(output, qmcplusplus::psiformer::spatialJetSpanElements(orbitals), 0.0);
  const auto& pairs = geometry.electronNucleusPairs();
  const std::size_t n = orbitals.electron_count;
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t channel = 0; channel < orbitals.element_count / n; ++channel)
    {
      const std::size_t spin = row < spin_up ? 0 : 1;
      const std::size_t element = (channel / n * n + row) * n + channel % n;
      for (std::size_t atom = 0; atom < nuclei; ++atom)
      {
        const std::size_t pair = row * nuclei + atom, parameter = channel * nuclei + atom;
        const double radius = pairs.distances()[pair];
        if (radius == 0)
          throw std::runtime_error("PsiFormer complex spatial undefined at electron-nucleus coalescence");
        const double decay = std::abs(p[zeta[spin].begin + parameter]);
        const double weighted = p[pi[spin].begin + parameter] * std::exp(-decay * radius);
        const double first = -decay * weighted;
        output[element] += weighted;
        for (std::size_t lane = 0; lane < orbitals.gradient_lanes; ++lane)
          if ((orbitals.mode == qmcplusplus::psiformer::SpatialJetMode::FULL_VGL ? lane / 3 : active) == row)
            output[orbitals.uncheckedPlaneOffset(0, 1 + lane, element)] +=
                first * pairs.distanceGradients()[pair][lane % 3];
        if (orbitals.mode == qmcplusplus::psiformer::SpatialJetMode::FULL_VGL)
          output[orbitals.uncheckedPlaneOffset(0, 1 + orbitals.gradient_lanes + row, element)] +=
              decay * decay * weighted * pairs.distanceGradientNormsSquared()[pair] +
              first * pairs.distanceLaplacians()[pair];
      }
    }
}

/// Return value/first/second radial derivatives with respect to pi or real zeta.
inline std::array<double, 3> realOrbitalEnvelopeParameterJet(
    double radius, double pi, double zeta, bool zeta_direction)
{
  const double decay = std::abs(zeta), exponential = std::exp(-decay * radius);
  if (!zeta_direction)
    return {exponential, -decay * exponential, decay * decay * exponential};
  const double sign = zeta > 0 ? 1.0 : (zeta < 0 ? -1.0 : 0.0);
  const double weight = sign * pi * exponential;
  return {-radius * weight, (decay * radius - 1) * weight,
          (2 * decay - decay * decay * radius) * weight};
}

/// Borrowed complete physical core derivatives; invalidated by any next workspace request.
struct DirectComplexSpatialResult
{
  qmcplusplus::psiformer::determinant::ComplexLogAmplitude amplitude;
  const qmcplusplus::psiformer::determinant::Complex* gradient = nullptr;
  const qmcplusplus::psiformer::determinant::Complex* laplacian_log = nullptr;
  const qmcplusplus::psiformer::determinant::Complex* laplacian_ratio = nullptr;
  std::size_t gradient_count = 0, laplacian_count = 0;
  DirectSpatialMode mode = DirectSpatialMode::FULL_VGL;
  std::size_t active_electron = 0;
  std::size_t parameter_version = 0, evaluation_generation = 0;
};

/// Stable executor addresses identify prepared spatial clones.
class DirectComplexSpatialExecutor;

/// Prepared real trunk, explicit envelope jets and complex head/reduction buffers only.
class DirectComplexSpatialWorkspace
{
  using C = qmcplusplus::psiformer::determinant::Complex;
public:
  /// Replace omitted real output/algebra by exact mixed-domain requirements before allocation.
  static std::size_t storageRequirement(const PsiFormerStorageShape& shape,
                                       DirectSpatialMode mode)
  {
    using namespace qmcplusplus::psiformer;
    if (mode != DirectSpatialMode::FULL_VGL && mode != DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT)
      throw std::invalid_argument("invalid complex spatial storage mode");
    const auto orbitals = checkedStorageProduct(shape.determinants,
        checkedStorageProduct(shape.electrons, shape.electrons, "complex matrix extent overflow"),
        "complex orbital extent overflow");
    const auto jets = layout(orbitals, shape.electrons, mode);
    const auto outputs = checkedStorageSum(jets.gradient_lanes,
        checkedStorageProduct(2, jets.laplacian_lanes, "complex output extent overflow"),
        "complex output extent overflow");
    std::size_t bytes = spatialWorkspaceStorageRequirement(shape, mode == DirectSpatialMode::FULL_VGL) -
        determinantStorageRequirement(shape.determinants, shape.electrons,
            jets.gradient_lanes, jets.laplacian_lanes) -
        checkedStorageBytes<double>(outputs, "real output bytes overflow");
    // The omitted real orbital jets and the explicit real envelope jets have identical extents.
    addStorageBytes(bytes, ComplexOrbitalHeadWorkspace::storageRequirement(jets, shape.occupied_momentum_count),
                    "complex spatial bytes overflow");
    addStorageBytes(bytes, determinant::ComplexDeterminantWorkspace::storageRequirement(shape.determinants,
        shape.electrons, 0, jets.gradient_lanes, jets.laplacian_lanes), "complex spatial bytes overflow");
    addStorageBytes(bytes, checkedStorageBytes<C>(outputs, "complex output bytes overflow"),
                    "complex spatial bytes overflow");
    return bytes;
  }

  /// Mutable prepared clones cannot share ownership by copying.
  DirectComplexSpatialWorkspace(const DirectComplexSpatialWorkspace&) = delete;

  /// Preserve exclusive allocation and owner identity.
  DirectComplexSpatialWorkspace& operator=(const DirectComplexSpatialWorkspace&) = delete;

  /// Transfer complete prepared storage without changing its executor identity.
  DirectComplexSpatialWorkspace(DirectComplexSpatialWorkspace&&) = default;

  /// Immutable head descriptors disallow move assignment.
  DirectComplexSpatialWorkspace& operator=(DirectComplexSpatialWorkspace&&) = delete;

  /// Invalidate all borrowed results before editing input coordinates.
  void setPositions(GeometryPositionView positions)
  {
    invalidate();
    trunk_->setPositions(positions);
  }

  /// Edit one coordinate in the prepared real trunk.
  void setPosition(std::size_t electron, std::size_t dimension, double value)
  {
    invalidate();
    trunk_->setPosition(electron, dimension, value);
  }

  /// Return the invalidation stamp including failed requests.
  std::size_t evaluationGeneration() const noexcept { return generation_; }

  /// Failed requests have no observed model version.
  std::size_t observedParameterVersion() const noexcept
  {
    return valid_ ? result_.parameter_version : std::numeric_limits<std::size_t>::max();
  }

  /// Expose only complete finite publication; callers must also match model version.
  const DirectComplexSpatialResult& result() const
  {
    if (!valid_)
      throw std::logic_error("PsiFormer complex spatial result unavailable");
    return result_;
  }

  /// Count every owned numeric allocation, excluding shared model parameters.
  std::size_t vectorStorageBytes() const
  {
    std::size_t bytes = trunk_->vectorStorageBytes();
    addStorageBytes(bytes, checkedStorageBytes<double>(envelopes_.capacity(), "complex envelope bytes overflow"),
                    "complex spatial bytes overflow");
    addStorageBytes(bytes, head_.storageBytes(), "complex spatial bytes overflow");
    addStorageBytes(bytes, determinant_.storageBytes(), "complex spatial bytes overflow");
    for (const auto* output : {&gradient_, &lap_log_, &lap_ratio_})
      addStorageBytes(bytes, checkedStorageBytes<C>(output->capacity(), "complex output bytes overflow"),
                      "complex spatial bytes overflow");
    return bytes;
  }

  /// Track every prepared allocation through repeated FULL/ACTIVE evaluations.
  std::size_t storageFingerprint() const noexcept
  {
    std::size_t hash = trunk_->storageFingerprint();
    qmcplusplus::psiformer::determinant::detail::mixStorage(hash, envelopes_);
    qmcplusplus::psiformer::determinant::detail::mixStorage(hash, gradient_);
    qmcplusplus::psiformer::determinant::detail::mixStorage(hash, lap_log_);
    qmcplusplus::psiformer::determinant::detail::mixStorage(hash, lap_ratio_);
    hash ^= head_.storageFingerprint();
    hash *= 1099511628211ULL;
    return hash ^ determinant_.storageFingerprint();
  }

private:
  friend class DirectComplexSpatialExecutor;

  /// Prepare one derivative mode without a legacy real head or packed feature copy.
  DirectComplexSpatialWorkspace(const DirectComplexSpatialExecutor* owner,
      std::unique_ptr<DirectSpatialTrunkWorkspace> trunk,
      const qmcplusplus::psiformer::ComplexOrbitalHeadDescriptor& descriptor, DirectSpatialMode mode)
      : owner_(owner), trunk_(std::move(trunk)), mode_(mode),
        features_(layout(checkedStorageProduct(descriptor.electrons, descriptor.feature_width,
            "complex feature extent overflow"), descriptor.electrons, mode)),
        orbitals_(layout(checkedStorageProduct(descriptor.determinants,
            checkedStorageProduct(descriptor.electrons, descriptor.electrons, "complex matrix extent overflow"),
            "complex orbital extent overflow"), descriptor.electrons, mode)),
        envelopes_(qmcplusplus::psiformer::spatialJetSpanElements(orbitals_)),
        head_(descriptor, features_, orbitals_), determinant_(descriptor.determinants, descriptor.electrons,
            0, features_.gradient_lanes, features_.laplacian_lanes),
        gradient_(features_.gradient_lanes), lap_log_(features_.laplacian_lanes), lap_ratio_(lap_log_.size())
  {}

  /// Map the existing physical differentiation modes to compact numerical jet layouts.
  static qmcplusplus::psiformer::SpatialJetLayout layout(std::size_t elements, std::size_t electrons,
                                                        DirectSpatialMode mode)
  {
    return qmcplusplus::psiformer::makeSpatialJetLayout(1, elements, electrons,
        mode == DirectSpatialMode::FULL_VGL ? qmcplusplus::psiformer::SpatialJetMode::FULL_VGL :
                                             qmcplusplus::psiformer::SpatialJetMode::ACTIVE);
  }

  /// Invalidate before every mutation or attempted evaluation, including failures.
  void invalidate()
  {
    valid_ = false;
    if (generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("PsiFormer complex spatial generation exhausted");
    ++generation_;
  }

  const DirectComplexSpatialExecutor* owner_;
  std::unique_ptr<DirectSpatialTrunkWorkspace> trunk_;
  DirectSpatialMode mode_;
  qmcplusplus::psiformer::SpatialJetLayout features_, orbitals_;
  std::vector<double> envelopes_;
  qmcplusplus::psiformer::ComplexOrbitalHeadWorkspace head_;
  qmcplusplus::psiformer::determinant::ComplexDeterminantWorkspace determinant_;
  std::vector<C> gradient_, lap_log_, lap_ratio_;
  DirectComplexSpatialResult result_;
  std::size_t generation_ = 0;
  bool valid_ = false;
};

/// Complete complex derivatives; the metadata companion owns no value workspace.
class DirectComplexSpatialExecutor
{
public:
  /// Resolve paired canonical head/envelope intervals once before preparing any clone.
  DirectComplexSpatialExecutor(const PsiFormer& model,
      const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan)
      : parameters_(model.p), metadata_(model, admit(model, plan), DirectValuePreparation::REAL_TRUNK_ONLY),
        trunk_(model, metadata_, plan, DirectOrbitalProjection::SCALAR, DirectValuePreparation::REAL_TRUNK_ONLY),
        descriptor_(qmcplusplus::psiformer::occupiedFullHeadDescriptor(plan))
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
  }

  /// Owner-address tokens require executor address stability throughout clone lifetime.
  DirectComplexSpatialExecutor(const DirectComplexSpatialExecutor&) = delete;

  /// Preserve the identity of prepared clones.
  DirectComplexSpatialExecutor& operator=(const DirectComplexSpatialExecutor&) = delete;

  /// Moving would invalidate prepared owner identities.
  DirectComplexSpatialExecutor(DirectComplexSpatialExecutor&&) = delete;

  /// Executor identity cannot be transferred by assignment.
  DirectComplexSpatialExecutor& operator=(DirectComplexSpatialExecutor&&) = delete;

  /// Allocate one FULL or ACTIVE clone with real spatial trunk storage only.
  std::unique_ptr<DirectComplexSpatialWorkspace> makeWorkspace(DirectSpatialMode mode) const
  {
    return std::unique_ptr<DirectComplexSpatialWorkspace>(new DirectComplexSpatialWorkspace(
        this, trunk_.makeTrunkWorkspace(mode), descriptor_, mode));
  }

  /// Evaluate all physical gradient and Laplacian lanes.
  DirectComplexSpatialResult evaluateFull(DirectComplexSpatialWorkspace& workspace) const
  {
    return evaluate(workspace, DirectSpatialMode::FULL_VGL, 0);
  }

  /// Evaluate one explicit physical electron's three gradient lanes, including either spin.
  DirectComplexSpatialResult evaluateActive(DirectComplexSpatialWorkspace& workspace, std::size_t electron) const
  {
    return evaluate(workspace, DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT, electron);
  }

private:
  /// Enforce the full paired periodic schema before resolving real metadata.
  static const qmcplusplus::psiformer::PsiFormerExecutionPlan& admit(const PsiFormer& model,
      const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan)
  {
    using namespace qmcplusplus::psiformer;
    const auto& shape = plan.modelShape();
    if (plan.environment().head_representation != HeadRepresentation::PAIRED_REAL_COMPLEX ||
        model.ne != shape.electrons() || model.ndet != shape.determinants ||
        model.dim != shape.feature_dimension || model.cfg.nup != shape.spin_up_electrons ||
        model.cfg.ndown != shape.spin_down_electrons || model.cfg.nuclei.shape[0] != shape.nuclei ||
        model.heads != shape.attention_heads || model.blocks != shape.attention_blocks)
      throw std::invalid_argument("PsiFormer complex spatial requires a matching paired periodic plan");
    return plan;
  }

  /// Retain offsets rather than pointers into the versioned real parameter store.
  static DirectParameterTensor interval(const qmcplusplus::psiformer::ParameterTensorDescriptor& descriptor)
  {
    return {descriptor.begin, descriptor.size()};
  }

  /// Compose shared real trunk, split-feature complex head and strict analytic reduction.
  DirectComplexSpatialResult evaluate(DirectComplexSpatialWorkspace& w, DirectSpatialMode mode,
                                      std::size_t active) const
  {
    using namespace qmcplusplus::psiformer;
    w.invalidate();
    if (w.owner_ != this || w.mode_ != mode)
      throw std::invalid_argument("PsiFormer complex spatial workspace owner or mode mismatch");
    const auto t = mode == DirectSpatialMode::FULL_VGL ? trunk_.evaluateTrunkFull(*w.trunk_) :
                                                      trunk_.evaluateTrunkActive(*w.trunk_, active);
    const double* p = parameters_.flat_values().data();
    fillRealOrbitalEnvelopeJets(*t.geometry, w.orbitals_, descriptor_.spin_up,
        metadata_.trunkLayout()->nucleusCount(), p, pi_, zeta_, active, w.envelopes_.data());
    ComplexOrbitalHeadParameters heads;
    heads.up = {p + real_[0].begin, p + imaginary_[0].begin, nullptr, nullptr, real_[0].size, 0};
    heads.down = {p + real_[1].begin, p + imaginary_[1].begin, nullptr, nullptr, real_[1].size, 0};
    const auto& f = *t.features;
    const ComplexOrbitalFeatureJetView features{f.value.data(), f.gradient.data(), f.laplacian.data(),
                                               f.value.size(), f.gradient.size(), f.laplacian.size()};
    w.head_.evaluate(features, w.envelopes_.data(), w.envelopes_.size(), t.packed_positions,
                     3 * descriptor_.electrons, heads, active);
    const auto* h = w.head_.data();
    const auto det = w.determinant_.evaluateSpatial(h, h + w.orbitals_.element_count,
        w.features_.gradient_lanes, h + (1 + w.features_.gradient_lanes) * w.orbitals_.element_count,
        w.features_.laplacian_lanes, w.gradient_.data(), w.lap_log_.data(), w.lap_ratio_.data());
    auto amplitude = det.amplitude;
    amplitude.log_abs += t.cusp;
    for (std::size_t lane = 0; lane < w.gradient_.size(); ++lane)
      w.gradient_[lane] += t.cusp_gradient[lane];
    for (std::size_t electron = 0; electron < w.lap_log_.size(); ++electron)
    {
      w.lap_log_[electron] += t.cusp_laplacian[electron];
      auto ratio = w.lap_log_[electron];
      for (std::size_t dim = 0; dim < 3; ++dim)
        ratio += w.gradient_[3 * electron + dim] * w.gradient_[3 * electron + dim];
      w.lap_ratio_[electron] = ratio;
    }
    if (!determinant::isFiniteReal(amplitude.log_abs))
      throw std::overflow_error("PsiFormer complex spatial amplitude overflow");
    for (const auto* output : {&w.gradient_, &w.lap_log_, &w.lap_ratio_})
      for (const auto value : *output)
        if (!determinant::isFiniteReal(value.real()) || !determinant::isFiniteReal(value.imag()))
          throw std::overflow_error("PsiFormer complex spatial derivative overflow");
    w.result_ = {amplitude, w.gradient_.data(), w.lap_log_.data(), w.lap_ratio_.data(),
                 w.gradient_.size(), w.lap_log_.size(), mode, active, t.parameter_version, w.generation_};
    w.valid_ = true;
    return w.result_;
  }

  const Parameters& parameters_;
  DirectValueExecutor metadata_;
  DirectSpatialExecutor trunk_;
  const qmcplusplus::psiformer::ComplexOrbitalHeadDescriptor descriptor_;
  std::array<DirectParameterTensor, 2> real_, imaginary_, pi_, zeta_;
};
} // namespace pf
#endif
