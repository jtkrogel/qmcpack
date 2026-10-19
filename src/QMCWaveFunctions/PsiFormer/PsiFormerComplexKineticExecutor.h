//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerComplexKineticExecutor.h
 * @brief FULL CPU complex dK/d(real canonical parameter) over one retained real tape.
 * Include after PsiFormerNative.h; no public optimizer or large-system admission.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_COMPLEX_KINETIC_EXECUTOR_H
#define QMCPLUSPLUS_PSIFORMER_COMPLEX_KINETIC_EXECUTOR_H
#include "PsiFormerKineticExecutor.h"
#include "PsiFormerComplexSpatialExecutor.h"

namespace pf
{
/** Borrow complete physical K and its canonical complex real-parameter response.
 * amplitude is the core including real cusp, without exp(i*k*sum r); K/response
 * already include the physical outer ik exactly once. No outer amplitude is published.
 */
struct DirectComplexKineticResult
{
  /// Core+cusp amplitude only; excludes the common outer exp(i*k*sum r).
  qmcplusplus::psiformer::determinant::ComplexLogAmplitude amplitude;
  qmcplusplus::psiformer::determinant::Complex kinetic;
  const qmcplusplus::psiformer::determinant::Complex* parameter_response = nullptr;
  std::size_t parameter_count = 0, parameter_version = 0, evaluation_generation = 0;
};

/// Stable executor identity binds all prepared numerical clones.
class DirectComplexKineticExecutor;

/// One immutable real FULL tape/P scratch, complex head/root jets, and one complex P output.
class DirectComplexKineticWorkspace
{
  using C = qmcplusplus::psiformer::determinant::Complex;
  using W = std::complex<long double>;
public:
  /// Count every retained mixed-domain numeric buffer before any clone allocation.
  static std::size_t storageRequirement(const PsiFormerStorageShape& shape)
  {
    using namespace qmcplusplus::psiformer;
    const char* message = "complex kinetic storage overflow";
    const auto n = shape.electrons, g = checkedStorageProduct(3, n, message);
    const auto square = checkedStorageProduct(n, n, message);
    const auto layout = makeSpatialJetLayout(1,
        checkedStorageProduct(shape.determinants, square, message), n, SpatialJetMode::FULL_VGL);
    std::size_t bytes = kineticTrunkWorkspaceStorageRequirement(shape);
    addStorageBytes(bytes, ComplexOrbitalHeadWorkspace::storageRequirement(layout, shape.occupied_momentum_count, shape.fourier_mode_count), message);
    addStorageBytes(bytes, determinant::ComplexDeterminantWorkspace::storageRequirement(
        shape.determinants, n, 0, g, n), message);
    addStorageBytes(bytes, checkedStorageBytes<double>(spatialJetSpanElements(layout), message), message);
    addStorageBytes(bytes, traceJetStorageRequirement(
        checkedStorageProduct(n, shape.feature_width, message), g, n), message);
    addStorageBytes(bytes, traceJetStorageRequirement(1, g, n), message);
    std::size_t complex_elements = spatialJetSpanElements(layout);
    addStorageBytes(complex_elements, shape.parameter_count, message);
    addStorageBytes(complex_elements, checkedStorageProduct(5, n, message), message);
    addStorageBytes(bytes, checkedStorageBytes<C>(complex_elements, message), message);
    addStorageBytes(bytes, checkedStorageBytes<W>(checkedStorageProduct(3, square, message), message), message);
    return bytes;
  }

  /// Large mutable tapes must remain exclusively clone-owned.
  DirectComplexKineticWorkspace(const DirectComplexKineticWorkspace&) = delete;

  /// Preserve exclusive numerical ownership on assignment.
  DirectComplexKineticWorkspace& operator=(const DirectComplexKineticWorkspace&) = delete;

  /// Transfer backing allocations; all preceding borrowed views are invalidated.
  DirectComplexKineticWorkspace(DirectComplexKineticWorkspace&&) = default;

  /// Immutable head metadata prevents move assignment.
  DirectComplexKineticWorkspace& operator=(DirectComplexKineticWorkspace&&) = delete;

  /// Revoke publication before replacing positions, including a malformed request.
  void setPositions(GeometryPositionView positions)
  {
    invalidate();
    trunk_->setPositions(positions);
  }

  /// Change one coordinate without growing any buffer.
  void setPosition(std::size_t electron, std::size_t dimension, double value)
  {
    invalidate();
    trunk_->setPosition(electron, dimension, value);
  }

  /// Expose only a completed result; callers must match its version to the current model.
  const DirectComplexKineticResult& result() const
  {
    if (!valid_)
      throw std::logic_error("PsiFormer complex kinetic result unavailable");
    return result_;
  }

  /// Include failed attempts in the publication generation.
  std::size_t evaluationGeneration() const noexcept { return generation_; }

  /// Count actual retained vector capacities, excluding shared model and object overhead.
  std::size_t vectorStorageBytes() const
  {
    std::size_t bytes = trunk_->vectorStorageBytes();
    addStorageBytes(bytes, head_.storageBytes(), "complex kinetic bytes overflow");
    addStorageBytes(bytes, determinant_.storageBytes(), "complex kinetic bytes overflow");
    for (const auto* buffer : {&envelopes_, &feature_seed_.value, &feature_seed_.gradient,
                               &feature_seed_.laplacian, &cusp_seed_.value, &cusp_seed_.gradient,
                               &cusp_seed_.laplacian})
      addStorageBytes(bytes, checkedStorageBytes<double>(buffer->capacity(), "complex kinetic bytes overflow"),
                      "complex kinetic bytes overflow");
    for (const auto* buffer : {&root_, &response_, &gradient_, &laplacian_, &ratio_})
      addStorageBytes(bytes, checkedStorageBytes<C>(buffer->capacity(), "complex kinetic bytes overflow"),
                      "complex kinetic bytes overflow");
    addStorageBytes(bytes, checkedStorageBytes<W>(scratch_.capacity(), "complex kinetic bytes overflow"),
                    "complex kinetic bytes overflow");
    return bytes;
  }

  /// Track every prepared allocation through both real seeded reverses and warmed calls.
  std::size_t storageFingerprint() const noexcept
  {
    std::size_t hash = trunk_->storageFingerprint();
    for (const auto* buffer : {&envelopes_, &feature_seed_.value, &feature_seed_.gradient,
                               &feature_seed_.laplacian, &cusp_seed_.value, &cusp_seed_.gradient,
                               &cusp_seed_.laplacian})
      qmcplusplus::psiformer::determinant::detail::mixStorage(hash, *buffer);
    for (const auto* buffer : {&root_, &response_, &gradient_, &laplacian_, &ratio_})
      qmcplusplus::psiformer::determinant::detail::mixStorage(hash, *buffer);
    qmcplusplus::psiformer::determinant::detail::mixStorage(hash, scratch_);
    return (hash ^ head_.storageFingerprint()) * 1099511628211ULL ^ determinant_.storageFingerprint();
  }

private:
  friend class DirectComplexKineticExecutor;

  /// Prepare only this FULL numerical profile; there is no hidden complete real owner.
  DirectComplexKineticWorkspace(const DirectComplexKineticExecutor* owner,
      std::unique_ptr<DirectKineticTrunkWorkspace> trunk,
      const qmcplusplus::psiformer::ComplexOrbitalHeadDescriptor& descriptor, std::size_t parameters)
      : owner_(owner), trunk_(std::move(trunk)),
        features_(qmcplusplus::psiformer::makeSpatialJetLayout(1,
            descriptor.electrons * descriptor.feature_width, descriptor.electrons,
            qmcplusplus::psiformer::SpatialJetMode::FULL_VGL)),
        orbitals_(qmcplusplus::psiformer::makeSpatialJetLayout(1,
            descriptor.determinants * descriptor.electrons * descriptor.electrons, descriptor.electrons,
            qmcplusplus::psiformer::SpatialJetMode::FULL_VGL)),
        envelopes_(qmcplusplus::psiformer::spatialJetSpanElements(orbitals_)),
        head_(descriptor, features_, orbitals_), determinant_(descriptor.determinants, descriptor.electrons,
            0, 3 * descriptor.electrons, descriptor.electrons),
        root_(envelopes_.size()), scratch_(3 * descriptor.electrons * descriptor.electrons),
        response_(parameters), gradient_(3 * descriptor.electrons),
        laplacian_(descriptor.electrons), ratio_(descriptor.electrons),
        feature_seed_(features_.element_count, features_.gradient_lanes, features_.laplacian_lanes),
        cusp_seed_(1, features_.gradient_lanes, features_.laplacian_lanes)
  {}

  /// Revoke every preceding publication even when the following request fails.
  void invalidate()
  {
    valid_ = false;
    if (generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("PsiFormer complex kinetic generation exhausted");
    ++generation_;
  }

  const DirectComplexKineticExecutor* owner_;
  std::unique_ptr<DirectKineticTrunkWorkspace> trunk_;
  qmcplusplus::psiformer::SpatialJetLayout features_, orbitals_;
  std::vector<double> envelopes_;
  qmcplusplus::psiformer::ComplexOrbitalHeadWorkspace head_;
  qmcplusplus::psiformer::determinant::ComplexDeterminantWorkspace determinant_;
  std::vector<C> root_;
  std::vector<W> scratch_;
  std::vector<C> response_, gradient_, laplacian_, ratio_;
  DirectTraceJetBuffer feature_seed_, cusp_seed_;
  DirectComplexKineticResult result_;
  std::size_t generation_ = 0;
  bool valid_ = false;
};

/** Complete ordinary-complex kinetic response for fixed-ion sole-component Coulomb.
 * The supplied positive inverse masses are physical. Immutable Cartesian outer k
 * enters total G exactly once; dV/d(real theta)=0 is specific to this stated scope.
 */
class DirectComplexKineticExecutor
{
  using C = qmcplusplus::psiformer::determinant::Complex;
public:
  /// Resolve canonical intervals once, with real trunk computation and paired head metadata.
  DirectComplexKineticExecutor(const PsiFormer& model,
      const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan, std::array<double, 3> outer_k = {})
      : parameters_(model.p), trunk_(model, plan, DirectKineticPreparation::REAL_TRUNK_ONLY),
        descriptor_(qmcplusplus::psiformer::occupiedFullHeadDescriptor(plan)),
        profile_(plan.environment().periodic_profile), outer_k_(outer_k),
        shape_{model.ne, model.cfg.nuclei.shape[0], model.ndet, model.dim, model.heads,
               plan.parameter(qmcplusplus::psiformer::ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0],
               model.blocks, plan.parameterCount(), bool(plan.environment().periodic_profile),
               qmcplusplus::psiformer::occupiedMomentumCount(plan), profile_ ? profile_->pair_feature_dimension : 8,
               qmcplusplus::psiformer::fourierModeCount(plan)}
  {
    using namespace qmcplusplus::psiformer;
    if (plan.environment().head_representation != HeadRepresentation::PAIRED_REAL_COMPLEX ||
        plan.modelShape().spin_up_electrons != model.cfg.nup ||
        plan.modelShape().spin_down_electrons != model.cfg.ndown ||
        plan.modelShape().attention_blocks != model.blocks)
      throw std::invalid_argument("PsiFormer complex kinetic requires a matching full paired periodic plan");
    for (double value : outer_k_)
      if (!determinant::isFiniteReal(value))
        throw std::invalid_argument("PsiFormer complex kinetic outer momentum must be finite");
    for (std::size_t spin = 0; spin < 2; ++spin)
    {
      const auto role = spin ? ParameterRole::BACKFLOW_DOWN_WEIGHT : ParameterRole::BACKFLOW_UP_WEIGHT;
      const auto interval = [](const ParameterTensorDescriptor& tensor) {
        return DirectParameterTensor{tensor.begin, tensor.size()};
      };
      real_[spin] = interval(plan.parameter(role));
      imaginary_[spin] = interval(plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG));
      if (descriptor_.fourier_modes)
        fourier_[spin] = interval(plan.parameter(spin ? ParameterRole::FOURIER_COEFFICIENT_DOWN : ParameterRole::FOURIER_COEFFICIENT_UP));
      else
      {
        pi_[spin] = interval(plan.parameter(spin ? ParameterRole::ENVELOPE_PI_DOWN : ParameterRole::ENVELOPE_PI_UP));
        zeta_[spin] = interval(plan.parameter(spin ? ParameterRole::ENVELOPE_ZETA_DOWN : ParameterRole::ENVELOPE_ZETA_UP));
      }
    }
  }

  /// Owner tokens require stable executor addresses throughout clone lifetimes.
  DirectComplexKineticExecutor(const DirectComplexKineticExecutor&) = delete;

  /// Prepared identities cannot be reassigned.
  DirectComplexKineticExecutor& operator=(const DirectComplexKineticExecutor&) = delete;

  /// Moving would invalidate every prepared clone's owner token.
  DirectComplexKineticExecutor(DirectComplexKineticExecutor&&) = delete;

  /// Preserve address-stable ownership under assignment.
  DirectComplexKineticExecutor& operator=(DirectComplexKineticExecutor&&) = delete;

  /// Expose the exact FULL numeric requirement, not allocator/shared/aggregate clone costs.
  std::size_t requiredStorageBytes() const { return DirectComplexKineticWorkspace::storageRequirement(shape_); }

  /// Include fixed-ion and occupied-column metadata in external lane accounting.
  std::size_t numericMetadataStorageBytes() const
  {
    return checkedStorageSum(trunk_.numericMetadataStorageBytes(),
        checkedStorageBytes<std::array<double, 3>>(checkedStorageSum(descriptor_.reciprocal_momenta.capacity(),
            descriptor_.fourier_modes ? descriptor_.fourier_modes->momenta().capacity() : 0,
            "complex kinetic mode extent overflow"),
            "complex kinetic momentum metadata overflow"), "complex kinetic metadata overflow");
  }

  /// Track fixed-ion metadata separately from the mutable tape allocation identity.
  std::size_t numericMetadataStorageFingerprint() const noexcept
  {
    auto hash = trunk_.numericMetadataStorageFingerprint();
    if (!descriptor_.reciprocal_momenta.empty())
      qmcplusplus::psiformer::determinant::detail::mixStorage(hash, descriptor_.reciprocal_momenta);
    if (descriptor_.fourier_modes)
      qmcplusplus::psiformer::determinant::detail::mixStorage(hash, descriptor_.fourier_modes->momenta());
    return hash;
  }

  /// Reject zero, unbounded or insufficient caps before allocating any numerical clone scratch.
  std::unique_ptr<DirectComplexKineticWorkspace> makeWorkspace(std::size_t maximum_bytes) const
  {
    if (!maximum_bytes || maximum_bytes == std::numeric_limits<std::size_t>::max() ||
        requiredStorageBytes() > maximum_bytes)
      throw std::invalid_argument("PsiFormer complex kinetic numeric budget exceeded");
    return std::unique_ptr<DirectComplexKineticWorkspace>(new DirectComplexKineticWorkspace(
        this, trunk_.makeTrunkWorkspace(kineticTrunkWorkspaceStorageRequirement(shape_)),
        descriptor_, shape_.parameter_count));
  }

  /// Publish complete complex K and dK/d(real theta) after finite validation of both passes.
  DirectComplexKineticResult evaluate(DirectComplexKineticWorkspace& w,
                                      const double* inverse_masses, std::size_t mass_count) const
  {
    using namespace qmcplusplus::psiformer;
    w.invalidate();
    const std::size_t n = descriptor_.electrons, batch = w.orbitals_.element_count;
    if (w.owner_ != this || !inverse_masses || mass_count != n)
      throw std::invalid_argument("PsiFormer complex kinetic owner or inverse-mass extent mismatch");
    for (std::size_t electron = 0; electron < n; ++electron)
      if (!determinant::isFiniteReal(inverse_masses[electron]) || inverse_masses[electron] <= 0)
        throw std::invalid_argument("PsiFormer complex kinetic inverse masses must be finite and positive");
    const auto tape = trunk_.evaluateTrunk(*w.trunk_);
    const double* p = parameters_.flat_values().data();
    fillRealOrbitalEnvelopeJets(*tape.geometry, w.orbitals_, descriptor_.spin_up,
        shape_.nuclei, p, pi_, zeta_, 0, w.envelopes_.data(), profile_);
    ComplexOrbitalHeadParameters heads;
    heads.up = {p + real_[0].begin, p + imaginary_[0].begin, nullptr, nullptr, real_[0].size, 0,
        fourier_[0].size ? p + fourier_[0].begin : nullptr, fourier_[0].size};
    heads.down = {p + real_[1].begin, p + imaginary_[1].begin, nullptr, nullptr, real_[1].size, 0,
        fourier_[1].size ? p + fourier_[1].begin : nullptr, fourier_[1].size};
    const auto& f = *tape.features;
    w.head_.evaluate({f.value.data(), f.gradient.data(), f.laplacian.data(),
        f.value.size(), f.gradient.size(), f.laplacian.size()}, w.envelopes_.data(), w.envelopes_.size(),
        tape.packed_positions, 3 * n, heads);
    const C* a = w.head_.data();
    auto amplitude = w.determinant_.evaluateSpatial(a, a + batch, 3 * n,
        a + (1 + 3 * n) * batch, n, w.gradient_.data(), w.laplacian_.data(), w.ratio_.data()).amplitude;

    // Total physical G includes the real cusp and common outer ik, never a Hermitian norm.
    C kinetic{};
    for (std::size_t electron = 0; electron < n; ++electron)
    {
      C ratio = w.laplacian_[electron] + tape.cusp_laplacian[electron];
      for (std::size_t dim = 0; dim < 3; ++dim)
      {
        const auto lane = 3 * electron + dim;
        const C total = w.gradient_[lane] + tape.cusp_gradient[lane] + C{0, outer_k_[dim]};
        ratio += total * total;
        w.gradient_[lane] = -inverse_masses[electron] * total;
      }
      kinetic -= 0.5 * inverse_masses[electron] * ratio;
      w.laplacian_[electron] = -0.5 * inverse_masses[electron];
    }
    w.determinant_.fillSpatialRootMatrixAdjoints(a + batch, a + (1 + 3 * n) * batch,
        w.gradient_.data(), w.laplacian_.data(), w.root_.data(), w.root_.size(),
        w.scratch_.data(), w.scratch_.size());
    pullBackOccupiedPhase(descriptor_, w.orbitals_, tape.packed_positions, 3 * n,
                           w.root_.data(), w.root_.size());
    std::fill(w.response_.begin(), w.response_.end(), C{});
    if (descriptor_.fourier_modes)
    {
      accumulateFourierCoefficientResponses(*descriptor_.fourier_modes, w.orbitals_, descriptor_.determinants,
          descriptor_.spin_up, tape.packed_positions, 3 * n, w.head_.affineProjectionData(), w.root_.data(), w.root_.size(),
          {w.response_.data() + fourier_[0].begin, w.response_.data() + fourier_[1].begin}, {fourier_[0].size, fourier_[1].size});
      transformFourierHeadJets(*descriptor_.fourier_modes, w.orbitals_, descriptor_.determinants, descriptor_.spin_up,
          tape.packed_positions, 3 * n, {p + fourier_[0].begin, p + fourier_[1].begin},
          {fourier_[0].size, fourier_[1].size}, w.root_.data(), w.root_.size(), true);
    }
    for (std::size_t component = 0; component < 2; ++component)
    {
      w.feature_seed_.clear();
      w.cusp_seed_.clear();
      reverseHead(w, tape, p, component);
      // Cusp enters exactly once, through the same analytic root reverse as the real trunk.
      for (std::size_t lane = 0; lane < 3 * n; ++lane)
        w.cusp_seed_.gradient[lane] = component ? w.gradient_[lane].imag() : w.gradient_[lane].real();
      for (std::size_t electron = 0; electron < n; ++electron)
        w.cusp_seed_.laplacian[electron] = component ? 0 : w.laplacian_[electron].real();
      const auto reverse = trunk_.reverseTrunk(*w.trunk_, tape, w.feature_seed_, w.cusp_seed_);
      for (std::size_t parameter = 0; parameter < w.response_.size(); ++parameter)
        w.response_[parameter] += component ? C{0, reverse[parameter]} : C{reverse[parameter], 0};
    }
    amplitude.log_abs += tape.cusp_value;
    if (!determinant::isFiniteReal(amplitude.log_abs) || !determinant::isFiniteComplex(kinetic))
      throw std::overflow_error("PsiFormer complex kinetic amplitude/energy overflow");
    for (C value : w.response_)
      if (!determinant::isFiniteComplex(value))
        throw std::overflow_error("PsiFormer complex kinetic parameter response overflow");
    w.result_ = {amplitude, kinetic, w.response_.data(), w.response_.size(), tape.parameter_version, w.generation_};
    w.valid_ = true;
    return w.result_;
  }

private:
  /// Map one compact real trace-jet plane without a packed feature copy.
  template<class Jet>
  static auto plane(Jet& jet, std::size_t index)
  {
    const auto elements = jet.value.size(), gradients = jet.gradient.size() / elements;
    return index == 0 ? jet.value.data() : (index <= gradients ?
        jet.gradient.data() + (index - 1) * elements :
        jet.laplacian.data() + (index - 1 - gradients) * elements);
  }

  /// Reverse B*E using true affine B jets, preserving envelope responses even where E=0.
  void reverseHead(DirectComplexKineticWorkspace& w, const DirectKineticTrunkResultView& tape,
                   const double* p, std::size_t component) const
  {
    const auto n = descriptor_.electrons, channels = descriptor_.determinants * n;
    const auto batch = w.orbitals_.element_count, planes = w.orbitals_.plane_count;
    const C* b = w.head_.affineProjectionData();
    const auto& pairs = tape.geometry->electronNucleusPairs();
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t channel = 0; channel < channels; ++channel)
      {
        const auto spin = row < descriptor_.spin_up ? 0 : 1;
        const auto element = (channel / n * n + row) * n + channel % n;
        C qb0{}, qe0{};
        for (std::size_t k = 0; k < planes; ++k)
        {
          qb0 += w.root_[k * batch + element] * w.envelopes_[k * batch + element];
          qe0 += w.root_[k * batch + element] * b[k * batch + element];
        }
        if (component == 0 && !descriptor_.fourier_modes)
          for (std::size_t atom = 0; atom < shape_.nuclei; ++atom)
          {
            const auto pair = row * shape_.nuclei + atom, index = channel * shape_.nuclei + atom;
            const auto selected = qmcplusplus::psiformer::periodic_feature::selectedRadius(
                pairs, pair, tape.geometry->electronIonBoundary(), profile_);
            const C ql = w.root_[(1 + 3 * n + row) * batch + element];
            for (std::size_t direction = 0; direction < 2; ++direction)
            {
              const auto radial = realOrbitalEnvelopeParameterJet(selected.value,
                  p[pi_[spin].begin + index], p[zeta_[spin].begin + index], direction != 0);
              C response = qe0 * radial[0] + ql * b[element] *
                  (radial[2] * selected.gradient_norm_squared + radial[1] * selected.laplacian);
              for (std::size_t dim = 0; dim < 3; ++dim)
              {
                const auto offset = (1 + 3 * row + dim) * batch + element;
                response += (w.root_[offset] * b[element] + 2.0 * ql * b[offset]) *
                    radial[1] * selected.gradient[dim];
              }
              w.response_[(direction ? zeta_[spin] : pi_[spin]).begin + index] += response;
            }
          }
        for (std::size_t k = 0; k < planes; ++k)
        {
          C qb = k == 0 ? qb0 : w.root_[k * batch + element] * w.envelopes_[element];
          if (k > 0 && k <= 3 * n)
            qb += 2.0 * w.root_[(1 + 3 * n + (k - 1) / 3) * batch + element] *
                w.envelopes_[k * batch + element];
          for (std::size_t feature = 0; feature < descriptor_.feature_width; ++feature)
          {
            const auto index = feature * channels + channel, f = row * descriptor_.feature_width + feature;
            if (component == 0)
            {
              const C response = qb * plane(*tape.features, k)[f];
              w.response_[real_[spin].begin + index] += response;
              w.response_[imaginary_[spin].begin + index] += C{0, 1} * response;
            }
            const C seed = qb * C{p[real_[spin].begin + index], p[imaginary_[spin].begin + index]};
            plane(w.feature_seed_, k)[f] += component ? seed.imag() : seed.real();
          }
        }
      }
  }

  const Parameters& parameters_;
  DirectKineticExecutor trunk_;
  const qmcplusplus::psiformer::ComplexOrbitalHeadDescriptor descriptor_;
  const std::shared_ptr<const qmcplusplus::psiformer::PeriodicProfileMetadata> profile_;
  const std::array<double, 3> outer_k_;
  const PsiFormerStorageShape shape_;
  std::array<DirectParameterTensor, 2> real_, imaginary_, pi_, zeta_, fourier_;
};
} // namespace pf
#endif
