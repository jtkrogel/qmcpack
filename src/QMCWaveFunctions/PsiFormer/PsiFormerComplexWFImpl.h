//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerComplexWFImpl.h
 * @brief PRIVATE SINGLE-TU IMPLEMENTATION: include only inside PsiFormerWF.cpp.
 * PsiFormerNative.h defines non-inline symbols. This header is deliberately not a
 * second compiled native owner, nor a public numerical header. The sole consumer
 * supplies existing geometry/materialization/provenance helpers in qmcplusplus.
 */
#ifndef QMC_PSIFORMER_COMPLEX_WF_IMPL_OWNER_TU
#error "PsiFormerComplexWFImpl.h belongs only to the PsiFormerWF.cpp native owner"
#endif
#ifndef QMCPLUSPLUS_PSIFORMER_COMPLEX_WF_IMPL_H
#define QMCPLUSPLUS_PSIFORMER_COMPLEX_WF_IMPL_H
#ifdef QMC_COMPLEX

/// Clone-local scalar preparation over one shared immutable paired model.
struct PsiFormerComplexWF::Impl
{
  using C = psiformer::determinant::Complex;
  using Amplitude = psiformer::determinant::ComplexLogAmplitude;

  /// Immutable state has a stable address before numerical executors bind its model.
  struct Model
  {
    /// Preserve paired metadata rather than recreating a default-real environment.
    static psiformer::ExecutionEnvironment environmentFor(const ParticleSet& electrons)
    {
      auto environment = makeExecutionEnvironment(electrons);
      if (environment.boundary != psiformer::BoundaryCondition::PERIODIC || electrons.isSpinor())
        throw std::invalid_argument("Complex PsiFormer inference requires nonspinor periodic bulk electrons");
      environment.head_representation = psiformer::HeadRepresentation::PAIRED_REAL_COMPLEX;
      environment.amplitude_scalar_domain = psiformer::ScalarDomain::COMPLEX;
      return environment;
    }

    /// Retain initializer provenance and construct only qualified complex inference owners.
    Model(InitializedNativeModel initialized, const ParticleSet& electrons, std::array<double, 3> reduced)
        : model(std::move(initialized.model)), profile(std::move(initialized.profile)), seed(initialized.seed),
          environment(environmentFor(electrons)), reduced_twist(reduced),
          cartesian_twist(makeTwistPhase(electrons, reduced).cartesian),
          plan(psiformer::PsiFormerExecutionPlan::fromParameters(model.p,
              {model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet,
               model.dim, model.heads, model.blocks}, environment)), value(model, plan), spatial(model, plan),
          identity(persistentModelIdentity(model, "internal", profile, seed, environment,
                                           reduced_twist, cartesian_twist))
    {}

    pf::PsiFormer model;
    const std::string profile;
    const std::uint64_t seed;
    const psiformer::ExecutionEnvironment environment;
    const std::array<double, 3> reduced_twist, cartesian_twist;
    const psiformer::PsiFormerExecutionPlan plan;
    const pf::DirectComplexValueExecutor value;
    const pf::DirectComplexSpatialExecutor spatial;
    const std::uint64_t identity;
  };

  /// Validate the explicit initializer profile before consuming its canonical storage.
  static InitializedNativeModel materialize(psiformer::InitializedPsiFormerParameters initialized,
      const ParticleSet& electrons, const ParticleSet& ions)
  {
    if (initialized.profile != psiformer::PSIFORMER_COMPLEX_PERIODIC_V1)
      throw std::invalid_argument("Complex PsiFormer inference requires explicit paired initialization");
    validateMatchingLattices(electrons, ions);
    return makeInitializedNativeModel(std::move(initialized), electrons, ions);
  }

  /// Prepare distinct VALUE/FULL/ACTIVE scratch; there is no real head or score tape.
  Impl(psiformer::InitializedPsiFormerParameters initialized, const ParticleSet& electrons,
       const ParticleSet& ions, std::size_t limit, std::array<double, 3> reduced)
      : Impl(std::make_shared<const Model>(materialize(std::move(initialized), electrons, ions), electrons, reduced),
             electrons, limit)
  {}

  /// Check the full requirement before allocating any clone-local numerical buffers.
  Impl(std::shared_ptr<const Model> shared, const ParticleSet& electrons, std::size_t limit)
      : state(std::move(shared)), particles(&electrons), numeric_limit(limit)
  {
    using namespace psiformer;
    using pf::addStorageBytes;
    using pf::checkedStorageBytes;
    validate(electrons);
    if (!limit || limit == std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("Complex PsiFormer requires a positive finite per-clone numeric byte cap");
    const auto& model = state->model;
    const pf::PsiFormerStorageShape shape{model.ne, model.cfg.nuclei.shape[0], model.ndet, model.dim, model.heads,
        state->plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0], model.blocks, model.p.size()};
    numeric_required = pf::DirectComplexValueWorkspace::storageRequirement(shape);
    for (auto mode : {pf::DirectSpatialMode::FULL_VGL, pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT})
      addStorageBytes(numeric_required, pf::DirectComplexSpatialWorkspace::storageRequirement(shape, mode),
                      "complex clone bytes overflow");
    addStorageBytes(numeric_required, checkedStorageBytes<GradType>(model.ne, "complex own gradient bytes overflow"),
                    "complex clone bytes overflow");
    addStorageBytes(numeric_required, checkedStorageBytes<ValueType>(model.ne, "complex own Laplacian bytes overflow"),
                    "complex clone bytes overflow");
    if (numeric_required > numeric_limit)
      throw std::length_error("Complex PsiFormer full clone numeric requirement exceeds its per-clone byte cap");
    value = state->value.makeWorkspace();
    full = state->spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
    active = state->spatial.makeWorkspace(pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT);
    own_gradient.resize(model.ne);
    own_laplacian.resize(model.ne);
    if (numericStorage().allocated_bytes != numeric_required)
      throw std::logic_error("Complex PsiFormer numeric capacity differs from its admitted exact requirement");
  }

  /// Sum live capacities and allocation identities, excluding immutable shared model storage.
  NumericStorage numericStorage() const
  {
    using namespace psiformer;
    using pf::addStorageBytes;
    using pf::checkedStorageBytes;
    std::size_t bytes = value->vectorStorageBytes();
    addStorageBytes(bytes, full->vectorStorageBytes(), "complex clone bytes overflow");
    addStorageBytes(bytes, active->vectorStorageBytes(), "complex clone bytes overflow");
    addStorageBytes(bytes, checkedStorageBytes<GradType>(own_gradient.capacity(), "complex own gradient bytes overflow"),
                    "complex clone bytes overflow");
    addStorageBytes(bytes, checkedStorageBytes<ValueType>(own_laplacian.capacity(), "complex own Laplacian bytes overflow"),
                    "complex clone bytes overflow");
    std::size_t fingerprint = value->storageFingerprint();
    for (auto next : {full->storageFingerprint(), active->storageFingerprint()})
      fingerprint = (fingerprint ^ next) * 1099511628211ULL;
    determinant::detail::mixStorage(fingerprint, own_gradient);
    determinant::detail::mixStorage(fingerprint, own_laplacian);
    return {numeric_required, bytes, numeric_limit, fingerprint};
  }

  /// Bind scalar requests to the target's identity, shape and immutable feature cell.
  void validate(const ParticleSet& p, int electron = -1, bool proposal = false) const
  {
    if (&p != particles || p.getTotalNum() != static_cast<int>(state->model.ne) ||
        p.groups() != 2 || p.groupsize(0) != static_cast<int>(state->model.cfg.nup) ||
        p.groupsize(1) != static_cast<int>(state->model.cfg.ndown))
      throw std::invalid_argument("Complex PsiFormer scalar target identity or shape mismatch");
    validateRuntimeLattice(p, state->environment, "Complex PsiFormer inference");
    if (electron < -1 || (electron >= 0 && static_cast<std::size_t>(electron) >= state->model.ne) ||
        (proposal && (electron < 0 || p.getActivePtcl() != electron)))
      throw std::invalid_argument("Complex PsiFormer scalar active particle mismatch");
  }

  /// Copy physical positions through checked double input setters, replacing only the active row.
  template<class Workspace>
  void positions(Workspace& workspace, const ParticleSet& p, int replacement = -1) const
  {
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
      for (int dimension = 0; dimension < 3; ++dimension)
        workspace.setPosition(electron, dimension,
            replacement == electron ? p.activeR(electron)[dimension] : p.R[electron][dimension]);
  }

  /// Apply the sole common outer phase at the adapter boundary, never inside an orbital head.
  double angle(const ParticleSet& p, int replacement = -1) const
  {
    double result = 0;
    if (state->reduced_twist != std::array<double, 3>{})
      for (int electron = 0; electron < p.getTotalNum(); ++electron)
        for (int dimension = 0; dimension < 3; ++dimension)
          result += state->cartesian_twist[dimension] *
              (replacement == electron ? p.activeR(electron)[dimension] : p.R[electron][dimension]);
    return result;
  }

  /// Canonical zeros have no phase; nonzero core phases remain fully complex.
  LogValue physicalLog(const Amplitude& amplitude, const ParticleSet& p, int replacement = -1) const
  {
    if (amplitude.isZero())
      return {amplitude.log_abs, 0};
    const LogValue result{amplitude.log_abs, std::arg(amplitude.phase) + angle(p, replacement)};
    if (!isFiniteWavefunctionValue(result))
      throw std::overflow_error("Complex PsiFormer physical logarithm overflow");
    return result;
  }

  /// Add i*k to logarithmic gradients; logarithmic Laplacians receive no outer-phase term.
  ValueType physicalGradient(C core, std::size_t dimension) const
  {
    const auto result = makePsiFormerScalar<ValueType>(core.real(), core.imag() + state->cartesian_twist[dimension]);
    if (!isFiniteWavefunctionValue(result))
      throw std::overflow_error("Complex PsiFormer physical gradient overflow");
    return result;
  }

  /// Validate the accepted reference before computing a complex ratio from unit core phases.
  PsiValue formRatio(const Amplitude& proposed, const ParticleSet& p, int electron) const
  {
    if (!accepted_valid || accepted_key != configurationIdentity(p) ||
        accepted_version != state->model.p.version() || accepted.isZero())
      throw std::logic_error("Complex PsiFormer ratio requires a current accepted amplitude");
    C result{};
    if (!proposed.isZero())
    {
      double delta = 0;
      for (int dimension = 0; dimension < 3; ++dimension)
        delta += state->cartesian_twist[dimension] * (p.activeR(electron)[dimension] - p.R[electron][dimension]);
      result = proposed.phase / accepted.phase * std::exp(proposed.log_abs - accepted.log_abs) *
          C{std::cos(delta), std::sin(delta)};
    }
    const PsiValue converted = makePsiFormerScalar<PsiValue>(result.real(), result.imag());
    if (!isFiniteWavefunctionValue(converted))
      throw std::overflow_error("Complex PsiFormer physical ratio overflow");
    return converted;
  }

  /// Publish a proposal only after all numerical and caller-output validation succeeds.
  void cache(const Amplitude& amplitude, const LogValue& log, const ParticleSet& p, int electron) noexcept
  {
    proposed = amplitude;
    proposed_log = log;
    proposed_particle = electron;
    proposed_key = configurationIdentity(p, electron);
    proposed_version = state->model.p.version();
    pending = true;
  }

  std::shared_ptr<const Model> state;
  const ParticleSet* particles;
  std::unique_ptr<pf::DirectComplexValueWorkspace> value;
  std::unique_ptr<pf::DirectComplexSpatialWorkspace> full, active;
  std::vector<GradType> own_gradient;
  std::vector<ValueType> own_laplacian;
  std::size_t numeric_required = 0, numeric_limit;
  Amplitude accepted, proposed;
  LogValue proposed_log{};
  std::uint64_t accepted_key = 0, proposed_key = 0;
  std::size_t accepted_version = 0, proposed_version = 0;
  int proposed_particle = -1;
  bool accepted_valid = false, pending = false;
};

/// Construct only the explicit paired profile; the XML builder remains unchanged in this slice.
PsiFormerComplexWF::PsiFormerComplexWF(std::string name, psiformer::InitializedPsiFormerParameters initialized,
    const ParticleSet& electrons, const ParticleSet& ions, std::size_t limit, std::array<double, 3> reduced)
    : WaveFunctionComponent(name), impl_(std::make_unique<Impl>(std::move(initialized), electrons, ions, limit, reduced))
{}

/// A clone shares only immutable model ownership and starts with invalid evaluation caches.
PsiFormerComplexWF::PsiFormerComplexWF(std::string name, std::unique_ptr<Impl> impl)
    : WaveFunctionComponent(name), impl_(std::move(impl))
{}

/// Expose honest per-clone numeric admission rather than the legacy batch-budget policy.
PsiFormerComplexWF::NumericStorage PsiFormerComplexWF::numericStorage() const { return impl_->numericStorage(); }

/// Destroy native numerical state only in the sole owner TU.
PsiFormerComplexWF::~PsiFormerComplexWF() = default;

/// Report the actual complete determinant expansion.
bool PsiFormerComplexWF::isMultiDet() const { return impl_->state->model.ndet > 1; }

/// Validate complete scatter before publishing any accepted amplitude or caller G/L.
PsiFormerComplexWF::LogValue PsiFormerComplexWF::evaluateLog(const ParticleSet& p,
    ParticleSet::ParticleGradient& g, ParticleSet::ParticleLaplacian& l)
{
  impl_->pending = impl_->accepted_valid = false;
  impl_->validate(p);
  if (g.size() != static_cast<std::size_t>(p.getTotalNum()) || l.size() != g.size())
    throw std::invalid_argument("Complex PsiFormer full scatter shape mismatch");
  impl_->positions(*impl_->full, p);
  const auto result = impl_->state->spatial.evaluateFull(*impl_->full);
  const LogValue physical = impl_->physicalLog(result.amplitude, p);
  // Stage this component's own products separately from any caller accumulation.
  for (int electron = 0; electron < p.getTotalNum(); ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      impl_->own_gradient[electron][dimension] =
          impl_->physicalGradient(result.gradient[3 * electron + dimension], dimension);
    impl_->own_laplacian[electron] = makePsiFormerScalar<ValueType>(result.laplacian_log[electron].real(),
                                                                 result.laplacian_log[electron].imag());
  }
  for (int electron = 0; electron < p.getTotalNum(); ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      if (!isFiniteWavefunctionValue(g[electron][dimension] +
          impl_->own_gradient[electron][dimension]))
        throw std::overflow_error("Complex PsiFormer gradient scatter overflow");
    if (!isFiniteWavefunctionValue(l[electron] + impl_->own_laplacian[electron]))
      throw std::overflow_error("Complex PsiFormer Laplacian scatter overflow");
  }
  for (int electron = 0; electron < p.getTotalNum(); ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      g[electron][dimension] += impl_->own_gradient[electron][dimension];
    l[electron] += impl_->own_laplacian[electron];
  }
  impl_->accepted = result.amplitude;
  impl_->accepted_key = configurationIdentity(p);
  impl_->accepted_version = result.parameter_version;
  impl_->accepted_valid = true;
  return log_value_ = physical;
}

/// VALUE ratio proposals never allocate a spatial or parameter tape.
PsiFormerComplexWF::PsiValue PsiFormerComplexWF::ratio(ParticleSet& p, int electron)
{
  impl_->pending = false;
  impl_->validate(p, electron, true);
  impl_->positions(*impl_->value, p, electron);
  const auto result = impl_->state->value.evaluate(*impl_->value);
  const PsiValue ratio = impl_->formRatio(result.amplitude, p, electron);
  const LogValue physical = impl_->physicalLog(result.amplitude, p, electron);
  impl_->cache(result.amplitude, physical, p, electron);
  return ratio;
}

/// Accepted ACTIVE jets do not overwrite accepted scalar or pending move state.
PsiFormerComplexWF::GradType PsiFormerComplexWF::evalGrad(ParticleSet& p, int electron)
{
  impl_->validate(p, electron);
  if (electron < 0)
    throw std::invalid_argument("Complex PsiFormer gradient electron is negative");
  impl_->positions(*impl_->active, p);
  const auto result = impl_->state->spatial.evaluateActive(*impl_->active, electron);
  GradType gradient;
  for (int dimension = 0; dimension < 3; ++dimension)
    gradient[dimension] = impl_->physicalGradient(result.gradient[dimension], dimension);
  return gradient;
}

/// Form finite ratio/gradient staging before the nonthrowing caller/proposal publication.
PsiFormerComplexWF::PsiValue PsiFormerComplexWF::ratioGrad(ParticleSet& p, int electron, GradType& gradient)
{
  impl_->pending = false;
  impl_->validate(p, electron, true);
  impl_->positions(*impl_->active, p, electron);
  const auto result = impl_->state->spatial.evaluateActive(*impl_->active, electron);
  const PsiValue ratio = impl_->formRatio(result.amplitude, p, electron);
  const LogValue physical = impl_->physicalLog(result.amplitude, p, electron);
  GradType staged = gradient;
  for (int dimension = 0; dimension < 3; ++dimension)
  {
    staged[dimension] += impl_->physicalGradient(result.gradient[dimension], dimension);
    if (!isFiniteWavefunctionValue(staged[dimension]))
      throw std::overflow_error("Complex PsiFormer proposed gradient scatter overflow");
  }
  gradient = staged;
  impl_->cache(result.amplitude, physical, p, electron);
  return ratio;
}

/// Commit cached amplitude only after proving ParticleSet active proposal provenance.
void PsiFormerComplexWF::acceptMove(ParticleSet& p, int electron, bool)
{
  const bool pending = impl_->pending;
  impl_->pending = false;
  impl_->validate(p, electron, true);
  if (!pending || !impl_->accepted_valid ||
      impl_->accepted_key != configurationIdentity(p) ||
      impl_->accepted_version != impl_->state->model.p.version() || electron != impl_->proposed_particle ||
      impl_->proposed_key != configurationIdentity(p, electron) ||
      impl_->proposed_version != impl_->state->model.p.version())
    throw std::logic_error("Complex PsiFormer accept does not match its cached proposal");
  impl_->accepted = impl_->proposed;
  impl_->accepted_key = impl_->proposed_key;
  impl_->accepted_version = impl_->proposed_version;
  impl_->accepted_valid = true;
  log_value_ = impl_->proposed_log;
}

/// Reject mismatched restore requests without committing or evaluating any move.
void PsiFormerComplexWF::restore(int electron)
{
  if (impl_->pending && electron != impl_->proposed_particle)
    throw std::logic_error("Complex PsiFormer restore does not match its cached proposal");
  impl_->pending = false;
}

/// Clone immutable provenance, but never accepted state, pending moves or numerical allocations.
std::unique_ptr<WaveFunctionComponent> PsiFormerComplexWF::makeClone(ParticleSet& particles) const
{
  return std::unique_ptr<WaveFunctionComponent>(new PsiFormerComplexWF(getName(),
      std::make_unique<Impl>(impl_->state, particles, impl_->numeric_limit)));
}

/// Do not register a misleading legacy real-sign walker record.
void PsiFormerComplexWF::registerData(ParticleSet&, WFBufferType&)
{
  throw std::logic_error("Complex PsiFormer walker persistence is not yet admitted");
}

/// Fail closed rather than claim a qualified persistent accepted FULL state.
PsiFormerComplexWF::LogValue PsiFormerComplexWF::updateBuffer(ParticleSet&, WFBufferType&, bool)
{
  throw std::logic_error("Complex PsiFormer walker persistence is not yet admitted");
}

/// Reject unqualified walker restoration before touching component state.
void PsiFormerComplexWF::copyFromBuffer(ParticleSet&, WFBufferType&)
{
  throw std::logic_error("Complex PsiFormer walker persistence is not yet admitted");
}

/// Public optimization requires complete complex kinetic response, not only a score.
void PsiFormerComplexWF::evaluateDerivatives(ParticleSet&, const OptVariables&,
    Vector<ValueType>&, Vector<ValueType>&)
{
  throw std::logic_error("Complex PsiFormer public optimization is not admitted");
}
#endif
#endif
