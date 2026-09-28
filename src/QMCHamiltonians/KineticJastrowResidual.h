//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_KINETIC_JASTROW_RESIDUAL_H
#define QMCPLUSPLUS_KINETIC_JASTROW_RESIDUAL_H

#include <vector>

#include "QMCHamiltonians/OperatorBase.h"

namespace qmcplusplus
{
/** Kinetic contribution generated when nonfermionic wavefunction factors
 * multiply the fermionic (Slater determinant) component.
 *
 * For Psi = D exp(J), this evaluates T[Psi]/Psi - T[D]/D, including the
 * determinant-Jastrow gradient cross term.
 */
class KineticJastrowResidual : public OperatorBase
{
public:
  explicit KineticJastrowResidual(ParticleSet& pset);

  std::string getClassName() const override { return "KineticJastrowResidual"; }
  Return_t evaluate(TrialWaveFunction& psi, ParticleSet& pset) override;
  bool put(xmlNodePtr cur) override { return true; }
  bool get(std::ostream& os) const override;
  std::unique_ptr<OperatorBase> makeClone(ParticleSet& pset, TrialWaveFunction& psi) const final;

private:
  bool same_mass_;
  FullPrecRealType one_over_2m_;
  std::vector<FullPrecRealType> minus_over_2m_;
};
} // namespace qmcplusplus

#endif
