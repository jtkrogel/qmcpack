//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include "FrozenReferenceResidualWF.h"
#include "TrialWaveFunction.h"
#include "Fermion/SlaterDet.h"
#include "PsiFormer/PsiFormerWF.h"
#include "Particle/VirtualParticleSet.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qmcplusplus
{

/// Create the initial frozen model family, detached from a separately runnable neural definition.
FrozenReferenceResidualWF::FrozenReferenceResidualWF(ParticleSet& target, const TrialWaveFunction& reference,
    const TrialWaveFunction& neural, double coefficient)
    : FrozenReferenceResidualWF(target, reference, neural, coefficient, true)
{}

/// Validate the bounded physical profile and allocate only clone-private child state.
FrozenReferenceResidualWF::FrozenReferenceResidualWF(ParticleSet& target, const TrialWaveFunction& reference,
    const TrialWaveFunction& neural, double coefficient, bool detach_neural)
    : WaveFunctionComponent("frozen_reference_sum"), derivative_workspace_(createDerivativeWorkspace()),
      coefficient_(coefficient),
      reference_particles_(std::make_unique<ParticleSet>(target)),
      neural_particles_(std::make_unique<ParticleSet>(target))
{
#if defined(QMC_COMPLEX) || defined(MIXED_PRECISION)
  throw std::invalid_argument("Frozen reference sum currently requires a real FP64 CPU build");
#endif
  if (!residual::finiteReal(coefficient_) || target.isSpinor() || target.getTotalNum() == 0 ||
      target.getTotalNum() > 12 || target.getLattice().getSuperCellEnum() != SUPERCELL_BULK)
    throw std::invalid_argument("Frozen reference sum requires finite c and 1 through 12 real bulk Gamma electrons");
  for (const TrialWaveFunction* branch : {&reference, &neural})
    for (const auto twist : branch->twist())
      if (twist != 0)
        throw std::invalid_argument("Frozen reference sum currently supports Gamma boundary conditions only");

  // The reference is cloned, not registered as a parent optimizable object.
  reference_ = reference.makeClone(*reference_particles_);
  neural_ = neural.makeClone(*neural_particles_);
  for (const auto& component : reference_->getOrbitals())
  {
    if (auto* determinant = dynamic_cast<SlaterDet*>(component.get()))
    {
      if (slater_)
        throw std::invalid_argument("Frozen reference sum requires exactly one Slater determinant component");
      slater_ = determinant;
    }
    else if (component->getClassName() != "J1OrbitalSoA" && component->getClassName() != "TwoBodyJastrow")
      throw std::invalid_argument("Frozen reference sum supports only Slater plus one/two-body Jastrow reference components");
  }
  if (!slater_ || neural_->getOrbitals().size() != 1 ||
      !(neural_component_ = dynamic_cast<PsiFormerWF*>(neural_->getOrbitals()[0].get())))
    throw std::invalid_argument("Frozen reference sum requires a complete SJ reference and a sole real PsiFormer child");
  if (detach_neural)
    neural_component_->isolateParameterState();
  parameter_schema_ = std::make_unique<wftrain::StructuredParameterSchema>(
      "frozen_reference_sum/" + neural_component_->parameterSchema().providerId(),
      neural_component_->parameterSchema().blocks());
}

/// Defined here because the privately owned child types are forward declared in the header.
FrozenReferenceResidualWF::~FrozenReferenceResidualWF() = default;

/// Require a nonzero residual before exposing neural tensors to the training driver.
void FrozenReferenceResidualWF::enableNeuralTraining()
{
  if (coefficient_ == 0)
    throw std::invalid_argument("Neural training requires a nonzero frozen reference sum coefficient");
  training_enabled_ = true;
}

/// Keep the frozen reference outside the parent's optimization schema.
const wftrain::StructuredParameterSchema& FrozenReferenceResidualWF::parameterSchema() const noexcept
{
  return *parameter_schema_;
}

/// Inspect model identity without allocating a parameter-vector copy.
std::size_t FrozenReferenceResidualWF::currentParameterVersion() const
{
  return neural_component_->parameterVersion();
}

/// Relabel one transactionally consistent child snapshot without copying reference parameters.
wftrain::StructuredParameterSnapshot FrozenReferenceResidualWF::snapshotParameters() const
{
  auto result = neural_component_->snapshotParameters();
  result.schema_fingerprint = parameter_schema_->fingerprint();
  return result;
}

/// Reject a mismatched parent before delegating all-value validation and atomic publication.
std::size_t FrozenReferenceResidualWF::publishParameters(
    const wftrain::StructuredParameterSnapshot& candidate, std::size_t expected_version)
{
  if (!training_enabled_ || candidate.schema_fingerprint != parameter_schema_->fingerprint())
    throw std::invalid_argument("Frozen reference sum publication requires its selected parent schema");
  if (proposal_pending_)
    throw std::logic_error("Cannot update a frozen reference sum during a pending move");
  auto child_candidate = candidate;
  child_candidate.schema_fingerprint = neural_component_->parameterSchema().fingerprint();
  const auto version = neural_component_->publishParameters(child_candidate, expected_version);
  accepted_valid_ = false;
  return version;
}

/// Candidate evaluation owns a new neural family; the complete reference remains frozen.
void FrozenReferenceResidualWF::isolateParameterState()
{
  if (proposal_pending_)
    throw std::logic_error("Cannot isolate a frozen reference sum during a pending move");
  neural_component_->isolateParameterState();
  derivative_workspace_ = createDerivativeWorkspace();
  accepted_valid_ = false;
}

/// Reuse the neural tensor decay selection without introducing reference weights.
std::vector<std::string> FrozenReferenceResidualWF::neuralWeightBlocks() const
{
  return neural_component_->neuralWeightBlocks();
}

/// Keep affine indices identical to the canonical neural tensor layout.
std::vector<wftrain::AffineParameterBlock> FrozenReferenceResidualWF::affineParameterBlocks() const
{
  return neural_component_->affineParameterBlocks();
}

/// Preserve non-affine neural fallback blocks in the same canonical layout.
std::vector<std::size_t> FrozenReferenceResidualWF::genericKFACParameterBlocks() const
{
  return neural_component_->genericKFACParameterBlocks();
}

/// Differentiate c*N/parent directly, so a zero neural amplitude need not divide by itself.
void FrozenReferenceResidualWF::evaluateParameterScore(const ParticleSet& particles,
    std::vector<double>& score, wftrain::AffineScoreObserver* observer) const
{
  // Spatial scratch is clone-private. A concurrent publication invalidates the whole sample.
  auto& self = const_cast<FrozenReferenceResidualWF&>(*this);
  const auto version = neural_component_->parameterVersion();
  self.ensureAccepted(particles);
  neural_component_->evaluateResidualParameterScore(particles, coefficient_,
      accepted_.spatial.log_abs, accepted_.spatial.sign, score, observer);
  if (version != neural_component_->parameterVersion())
    throw std::runtime_error("Frozen reference sum parameters changed during a score sample");
}

/// Send parent-weighted affine observations through the existing factor interface.
void FrozenReferenceResidualWF::observeAffineScores(const ParticleSet& particles,
    wftrain::AffineScoreObserver& observer, std::vector<double>& score) const
{
  evaluateParameterScore(particles, score, &observer);
}

/// Evaluate SJ amplitude derivatives without an inverse at a reference determinant node.
residual::ScaledSpatialJet FrozenReferenceResidualWF::referenceAmplitude()
{
  const std::size_t electrons = reference_particles_->getTotalNum();
  residual::ScaledSpatialJet result{0, residual::SpatialJet(electrons, 1)};
  for (int spin = 0; spin < slater_->getNumDets(); ++spin)
  {
    auto& determinant = slater_->getDet(spin);
    const int first = determinant.getFirstIndex();
    const int count = determinant.getLastIndex() - first;
    if (count == 0)
      continue;
    SPOSet& orbitals = determinant.getPhi();
    SPOSet::ValueVector values(orbitals.getOrbitalSetSize());
    SPOSet::GradVector gradients(orbitals.getOrbitalSetSize());
    SPOSet::ValueVector laplacians(orbitals.getOrbitalSetSize());
    std::vector<residual::SpatialJet> matrix(count * count, residual::SpatialJet(electrons));
    for (int row = 0; row < count; ++row)
    {
      const int electron = first + row;
      orbitals.evaluateVGL(*reference_particles_, electron, values, gradients, laplacians);
      for (int col = 0; col < count; ++col)
      {
        auto& entry = matrix[row * count + col];
        entry.value = std::real(values[col]);
        for (int d = 0; d < 3; ++d)
          entry.gradient[3 * electron + d] = std::real(gradients[col][d]);
        entry.laplacian[electron] = std::real(laplacians[col]);
      }
    }
    result = residual::multiply(result, residual::determinant(std::move(matrix), count));
  }

  // Every admitted Jastrow is positive: its log derivatives can be converted safely.
  ParticleSet::ParticleGradient gradient(electrons);
  ParticleSet::ParticleLaplacian laplacian(electrons);
  gradient = ValueType(0);
  laplacian = ValueType(0);
  LogValue jastrow_log(0);
  for (const auto& component : reference_->getOrbitals())
    if (component.get() != slater_)
      jastrow_log += component->evaluateLog(*reference_particles_, gradient, laplacian);
  residual::SpatialJet jastrow(electrons, 1);
  for (std::size_t electron = 0; electron < electrons; ++electron)
  {
    jastrow.laplacian[electron] = std::real(laplacian[electron]);
    for (std::size_t d = 0; d < 3; ++d)
    {
      const long double g = std::real(gradient[electron][d]);
      jastrow.gradient[3 * electron + d] = g;
      jastrow.laplacian[electron] += g * g;
    }
  }
  return residual::multiply(result, {std::real(jastrow_log), std::move(jastrow)});
}

/// Stage a complete physical configuration into private child state and combine amplitudes.
FrozenReferenceResidualWF::State FrozenReferenceResidualWF::evaluateAt(const std::vector<PosType>& positions)
{
  if (positions.size() != static_cast<std::size_t>(reference_particles_->getTotalNum()))
    throw std::invalid_argument("Frozen reference sum received the wrong configuration size");
  for (std::size_t e = 0; e < positions.size(); ++e)
  {
    reference_particles_->R[e] = positions[e];
    neural_particles_->R[e] = positions[e];
  }
  const auto version = neural_component_->parameterVersion();
  reference_particles_->update();
  auto amplitude = referenceAmplitude();
  if (coefficient_ != 0)
    amplitude = residual::sum(amplitude, neural_component_->evaluateResidualAmplitude(*neural_particles_), coefficient_);
  State result;
  result.positions = positions;
  result.parameter_version = version;
  if (version != neural_component_->parameterVersion())
    throw std::runtime_error("Frozen reference sum parameters changed during spatial evaluation");
  if (amplitude.jet.value == 0)
  {
    // A proposed total node is a zero-weight move; do not invent a nodal log derivative.
    result.spatial.log_abs = -std::numeric_limits<long double>::infinity();
    result.spatial.gradient.assign(3 * positions.size(), 0);
    result.spatial.laplacian.assign(positions.size(), 0);
  }
  else
  {
    result.spatial = residual::logarithmic(amplitude);
    checkedReal(result.spatial.log_abs);
    for (auto x : result.spatial.gradient)
      checkedReal(x);
    for (auto x : result.spatial.laplacian)
      checkedReal(x);
  }
  return result;
}

/// Ensure ratios use the accepted physical configuration, independent of speculative child caches.
void FrozenReferenceResidualWF::ensureAccepted(const ParticleSet& particles)
{
  const std::vector<PosType> positions(particles.R.begin(), particles.R.end());
  if (!accepted_valid_ || positions != accepted_.positions ||
      accepted_.parameter_version != neural_component_->parameterVersion())
  {
    if (proposal_pending_)
      throw std::logic_error("Cannot refresh a frozen reference sum during a pending proposal");
    accepted_ = evaluateAt(positions);
    if (!residual::finiteReal(accepted_.spatial.log_abs))
      throw std::domain_error("Frozen reference sum accepted configuration is a total node");
    accepted_valid_ = true;
    log_value_ = nativeLog(accepted_);
  }
}

/// Fail rather than publish overflowed derivatives into the Hamiltonian.
FrozenReferenceResidualWF::RealType FrozenReferenceResidualWF::checkedReal(long double value)
{
  const RealType result = static_cast<RealType>(value);
  if (!residual::finiteReal(result))
    throw std::overflow_error("Frozen reference sum produced a nonfinite native result");
  return result;
}

/// Preserve the sign as a phase in the component's standard complex-log scalar.
FrozenReferenceResidualWF::LogValue FrozenReferenceResidualWF::nativeLog(const State& state)
{
  const RealType value = residual::finiteReal(state.spatial.log_abs) ? checkedReal(state.spatial.log_abs) :
      -std::numeric_limits<RealType>::infinity();
  return LogValue(value, state.spatial.sign < 0 ? std::acos(RealType(-1)) : 0);
}

/// Add parent logarithmic derivatives once; child values never touch caller G/L.
void FrozenReferenceResidualWF::addSpatial(const State& state, ParticleSet::ParticleGradient& gradient,
                                          ParticleSet::ParticleLaplacian& laplacian)
{
  for (std::size_t e = 0; e < state.spatial.laplacian.size(); ++e)
  {
    for (std::size_t d = 0; d < 3; ++d)
      gradient[e][d] += checkedReal(state.spatial.gradient[3 * e + d]);
    laplacian[e] += checkedReal(state.spatial.laplacian[e]);
  }
}

/// Reconstruct the total signed log and spatial state at the caller's configuration.
FrozenReferenceResidualWF::LogValue FrozenReferenceResidualWF::evaluateLog(const ParticleSet& particles,
    ParticleSet::ParticleGradient& gradient, ParticleSet::ParticleLaplacian& laplacian)
{
  ensureAccepted(particles);
  addSpatial(accepted_, gradient, laplacian);
  return log_value_;
}

/// Return one accepted total gradient, not the isolated neural or reference gradient.
FrozenReferenceResidualWF::GradType FrozenReferenceResidualWF::evalGrad(ParticleSet& particles, int electron)
{
  ensureAccepted(particles);
  GradType result;
  for (int d = 0; d < 3; ++d)
    result[d] = checkedReal(accepted_.spatial.gradient.at(3 * electron + d));
  return result;
}

/// Compare two complete amplitudes with a nonzero accepted denominator.
FrozenReferenceResidualWF::PsiValue FrozenReferenceResidualWF::proposedRatio(const State& proposed) const
{
  if (!residual::finiteReal(proposed.spatial.log_abs))
    return PsiValue(0);
  return PsiValue(proposed.spatial.sign * accepted_.spatial.sign *
      checkedReal(std::exp(proposed.spatial.log_abs - accepted_.spatial.log_abs)));
}

/// Recompute a single-particle proposal in private scratch before publishing any state.
FrozenReferenceResidualWF::PsiValue FrozenReferenceResidualWF::ratio(ParticleSet& particles, int electron)
{
  ensureAccepted(particles);
  if (proposal_pending_)
    throw std::logic_error("Frozen reference sum already has a pending proposal");
  auto positions = accepted_.positions;
  positions.at(electron) = particles.activeR(electron);
  proposed_ = evaluateAt(positions);
  const PsiValue result = proposedRatio(proposed_);
  proposed_electron_ = electron;
  proposal_pending_ = true;
  return result;
}

/// Supply the same total proposal ratio and its gradient for drift-based native consumers.
FrozenReferenceResidualWF::PsiValue FrozenReferenceResidualWF::ratioGrad(ParticleSet& particles, int electron,
                                                                      GradType& gradient)
{
  const PsiValue result = ratio(particles, electron);
  for (int d = 0; d < 3; ++d)
    gradient[d] += checkedReal(proposed_.spatial.gradient[3 * electron + d]);
  return result;
}

/// Publish a fully prepared nonzero cache with no child inverse/proposal dependency.
void FrozenReferenceResidualWF::commitProposal()
{
  if (!proposal_pending_ || !residual::finiteReal(proposed_.spatial.log_abs) ||
      proposed_.parameter_version != neural_component_->parameterVersion())
    throw std::logic_error("Cannot commit a missing, stale or zero-weight frozen reference sum proposal");
  accepted_ = std::move(proposed_);
  accepted_valid_ = true;
  proposal_pending_ = false;
  log_value_ = nativeLog(accepted_);
}

/// Resolve one single-particle proposal, rejecting mismatched transaction identity.
void FrozenReferenceResidualWF::acceptMove(ParticleSet&, int electron, bool)
{
  if (!proposal_pending_ || proposed_electron_ != electron)
    throw std::logic_error("Frozen reference sum single-particle acceptance does not match its proposal");
  commitProposal();
}

/// Drop speculative state; accepted caches and frozen models have never been modified.
void FrozenReferenceResidualWF::restore(int)
{
  proposal_pending_ = false;
}

/// Stage an entire crowd and clean up all pending flags if any lane fails.
void FrozenReferenceResidualWF::mw_evaluateMultiParticleMove(
    const RefVectorWithLeader<WaveFunctionComponent>& components, const RefVectorWithLeader<ParticleSet>& particles,
    const MCMultiParticleMoves<CoordsType::POS>& moves, std::vector<LogValue>& log_ratios,
    const RefVector<ParticleSet::ParticleGradient>& gradients,
    const RefVector<ParticleSet::ParticleLaplacian>& laplacians) const
{
  moves.validateFor(particles);
  if (components.size() != particles.size() || log_ratios.size() != particles.size() ||
      gradients.size() != particles.size() || laplacians.size() != particles.size())
    throw std::invalid_argument("Frozen reference sum selected-move crowd extents differ");
  // Refuse overlap before entering cleanup: an existing proposal belongs to its original caller.
  for (std::size_t lane = 0; lane < components.size(); ++lane)
    if (dynamic_cast<const FrozenReferenceResidualWF&>(components[lane]).proposal_pending_)
      throw std::logic_error("Frozen reference sum selected move overlaps an existing proposal");
  try
  {
    for (std::size_t lane = 0; lane < components.size(); ++lane)
    {
      auto& component = dynamic_cast<FrozenReferenceResidualWF&>(components[lane]);
      component.ensureAccepted(particles[lane]);
      if (component.proposal_pending_)
        throw std::logic_error("Frozen reference sum selected move overlaps an existing proposal");
      auto positions = component.accepted_.positions;
      const auto selected = moves.slice(lane);
      for (std::size_t k = 0; k < selected.size(); ++k)
        positions[selected.particleIndex(k)] = selected.proposedPosition(k);
      component.proposed_ = component.evaluateAt(positions);
      log_ratios[lane] = nativeLog(component.proposed_) - nativeLog(component.accepted_);
      addSpatial(component.proposed_, gradients[lane], laplacians[lane]);
      component.proposal_fingerprint_ = moves.fingerprint();
      component.proposed_electron_ = -1;
      component.proposal_pending_ = true;
    }
  }
  catch (...)
  {
    for (WaveFunctionComponent& component : components)
      static_cast<FrozenReferenceResidualWF&>(component).proposal_pending_ = false;
    throw;
  }
}

/// Validate every lane before atomically selecting its prepared accepted or rejected state.
void FrozenReferenceResidualWF::mw_accept_rejectMultiParticleMove(
    const RefVectorWithLeader<WaveFunctionComponent>& components, const RefVectorWithLeader<ParticleSet>& particles,
    const MCMultiParticleMoves<CoordsType::POS>& moves, const std::vector<bool>& accepted) const
{
  if (components.size() != particles.size() || components.size() != accepted.size())
    throw std::invalid_argument("Frozen reference sum selected acceptance extents differ");
  for (std::size_t lane = 0; lane < components.size(); ++lane)
  {
    const auto& component = dynamic_cast<const FrozenReferenceResidualWF&>(components[lane]);
    if (!component.proposal_pending_ || component.proposal_fingerprint_ != moves.fingerprint() ||
        component.proposed_.parameter_version != component.neural_component_->parameterVersion() ||
        (accepted[lane] && !residual::finiteReal(component.proposed_.spatial.log_abs)))
      throw std::logic_error("Frozen reference sum selected acceptance is stale or has zero weight");
  }
  for (std::size_t lane = 0; lane < components.size(); ++lane)
  {
    auto& component = static_cast<FrozenReferenceResidualWF&>(components[lane]);
    if (accepted[lane])
      component.commitProposal();
    else
      component.proposal_pending_ = false;
  }
}

/// Evaluate every ECP quadrature position relative to the same complete accepted sum.
void FrozenReferenceResidualWF::evaluateRatios(const VirtualParticleSet& particles, std::vector<ValueType>& ratios)
{
  ensureAccepted(particles.getRefPS());
  if (ratios.size() != static_cast<std::size_t>(particles.getTotalNum()))
    throw std::invalid_argument("Frozen reference sum virtual-ratio output has the wrong size");
  for (int k = 0; k < particles.getTotalNum(); ++k)
  {
    auto positions = accepted_.positions;
    positions.at(particles.refPtcl) = particles.R[k];
    ratios[k] = proposedRatio(evaluateAt(positions));
  }
}

/// Register only the parent state; child scratch is not a persistent walker cache.
void FrozenReferenceResidualWF::registerData(ParticleSet& particles, WFBufferType& buffer)
{
  ensureAccepted(particles);
  RealType slot = 0;
  for (std::size_t k = 0; k < 2 + accepted_.spatial.gradient.size() + accepted_.spatial.laplacian.size(); ++k)
    buffer.add(slot);
}

/// Write the existing buffer record after restoring the physical accepted coordinates.
FrozenReferenceResidualWF::LogValue FrozenReferenceResidualWF::updateBuffer(ParticleSet& particles,
    WFBufferType& buffer, bool)
{
  ensureAccepted(particles);
  addSpatial(accepted_, particles.G, particles.L);
  RealType slot = std::real(log_value_);
  buffer.put(slot);
  slot = std::imag(log_value_);
  buffer.put(slot);
  for (auto x : accepted_.spatial.gradient)
  {
    slot = checkedReal(x);
    buffer.put(slot);
  }
  for (auto x : accepted_.spatial.laplacian)
  {
    slot = checkedReal(x);
    buffer.put(slot);
  }
  return log_value_;
}

/// Restore native scalar/derivative values without trusting mutable child evaluation scratch.
void FrozenReferenceResidualWF::copyFromBuffer(ParticleSet& particles, WFBufferType& buffer)
{
  RealType log_abs, phase;
  buffer.get(log_abs);
  buffer.get(phase);
  accepted_.spatial.log_abs = log_abs;
  accepted_.spatial.sign = std::cos(phase) < 0 ? -1 : 1;
  accepted_.spatial.gradient.resize(3 * particles.getTotalNum());
  accepted_.spatial.laplacian.resize(particles.getTotalNum());
  RealType value;
  for (auto& x : accepted_.spatial.gradient)
  {
    buffer.get(value);
    x = value;
  }
  for (auto& x : accepted_.spatial.laplacian)
  {
    buffer.get(value);
    x = value;
  }
  accepted_.positions.assign(particles.R.begin(), particles.R.end());
  // Buffers contain spatial state only: after a model update, reconstruct rather than trust it.
  accepted_valid_ = false;
  proposal_pending_ = false;
  ensureAccepted(particles);
}

/// Keep legacy full local-energy response separate from the implemented structured score interface.
void FrozenReferenceResidualWF::evaluateDerivatives(ParticleSet&, const OptVariables&,
                                                   Vector<ValueType>&, Vector<ValueType>&)
{
  throw std::invalid_argument("Frozen reference sum requires structured score-covariance training; legacy full-response derivatives are unsupported");
}

/// Share only the already isolated frozen neural image across walker clones.
std::unique_ptr<WaveFunctionComponent> FrozenReferenceResidualWF::makeClone(ParticleSet& particles) const
{
  auto clone = std::unique_ptr<FrozenReferenceResidualWF>(
      new FrozenReferenceResidualWF(particles, *reference_, *neural_, coefficient_, false));
  clone->training_enabled_ = training_enabled_;
  clone->derivative_workspace_ = derivative_workspace_;
  return clone;
}

} // namespace qmcplusplus
