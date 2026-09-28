//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#include "KineticJastrowResidual.h"

#include <cmath>
#include <stdexcept>

#include "BareKineticHelper.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"

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

  ParticleSet::ParticleGradient fermion_g(pset.getTotalNum());
  ParticleSet::ParticleGradient nonfermion_g(pset.getTotalNum());
  ParticleSet::ParticleLaplacian fermion_l(pset.getTotalNum());
  ParticleSet::ParticleLaplacian nonfermion_l(pset.getTotalNum());
  fermion_g = 0.0;
  nonfermion_g = 0.0;
  fermion_l = 0.0;
  nonfermion_l = 0.0;

  for (const auto& component : psi.getOrbitals())
    if (component->isFermionic())
      component->evaluateLog(pset, fermion_g, fermion_l);
    else
      component->evaluateLog(pset, nonfermion_g, nonfermion_l);

  value_ = 0.0;
  for (int group = 0; group < pset.groups(); ++group)
  {
    Return_t group_residual = 0.0;
    for (int particle = pset.first(group); particle < pset.last(group); ++particle)
    {
      const auto total_g = fermion_g[particle] + nonfermion_g[particle];
      const auto total_l = fermion_l[particle] + nonfermion_l[particle];
      group_residual +=
          laplacian(total_g, total_l) - laplacian(fermion_g[particle], fermion_l[particle]);
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
