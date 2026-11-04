//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#include "KineticJastrowResidual.h"

#include <cmath>
#include <stdexcept>

#include "BareKineticHelper.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "Utilities/TimerManager.h"

namespace qmcplusplus
{
KineticJastrowResidual::KineticJastrowResidual(ParticleSet& pset)
{
  setEnergyDomain(KINETIC);
  oneBodyQuantumDomain(pset);

  SpeciesSet& species(pset.getSpeciesSet());
  const int mass_index = species.addAttribute("mass");
  minus_over_2m_.resize(species.size());
  same_mass_ = true;
  const FullPrecRealType particle_mass = species(mass_index, 0);
  one_over_2m_ = 0.5 / particle_mass;
  for (int group = 0; group < species.size(); ++group)
  {
    same_mass_ &= std::abs(species(mass_index, group) - particle_mass) < 1.0e-6;
    minus_over_2m_[group] = -1.0 / (2.0 * species(mass_index, group));
  }
}

KineticJastrowResidual::Return_t KineticJastrowResidual::evaluate(TrialWaveFunction& psi, ParticleSet& pset)
{
  // Separate timers also support flat-timer builds with concurrent walkers.
  static thread_local auto* residual_timer = &createGlobalTimer("MFR::KineticJastrowResidual", timer_level_fine);
  ScopedTimer scope(*residual_timer);
  bool has_fermionic = false;
  bool has_nonfermionic = false;
  for (const auto& component : psi.getOrbitals())
  {
    has_fermionic |= component->isFermionic();
    has_nonfermionic |= !component->isFermionic();
  }

  if (!has_fermionic)
    throw std::runtime_error("KineticJastrowResidual requires a fermionic wavefunction component");
  if (!has_nonfermionic)
  {
    value_ = 0.0;
    return value_;
  }

  // As for BareKineticEnergy, the caller has current total log derivatives
  // in P.G at the accepted energy boundary. Obtain only the nonfermionic
  // derivatives from each component's supported cached-GL interface.
  // Re-evaluating D here would rebuild its inverse solely for the residual.
  nonfermion_g_.resize(pset.getTotalNum());
  nonfermion_l_.resize(pset.getTotalNum());
  nonfermion_g_ = 0.0;
  nonfermion_l_ = 0.0;
  for (const auto& component : psi.getOrbitals())
    if (!component->isFermionic())
      component->evaluateGL(pset, nonfermion_g_, nonfermion_l_, false);

  value_ = 0.0;
  for (int group = 0; group < pset.groups(); ++group)
  {
    Return_t group_residual = 0.0;
    for (int particle = pset.first(group); particle < pset.last(group); ++particle)
    {
      const auto determinant_g = pset.G[particle] - nonfermion_g_[particle];
      // Delta T_J = -1/(2m) Re[L_J + G_J.G_J + 2 G_D.G_J].
      // The dot products are algebraic, without complex conjugation.
      group_residual += laplacian(nonfermion_g_[particle], nonfermion_l_[particle]) +
          2.0 * std::real(dot(determinant_g, nonfermion_g_[particle]));
    }
    value_ += group_residual * (same_mass_ ? -one_over_2m_ : minus_over_2m_[group]);
  }
  return value_;
}

bool KineticJastrowResidual::get(std::ostream& os) const
{
  os << "Kinetic residual from nonfermionic wavefunction factors";
  return true;
}

std::unique_ptr<OperatorBase> KineticJastrowResidual::makeClone(ParticleSet& pset, TrialWaveFunction& psi) const
{
  return std::make_unique<KineticJastrowResidual>(pset);
}
} // namespace qmcplusplus
