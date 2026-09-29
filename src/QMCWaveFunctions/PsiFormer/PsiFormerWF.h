///////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
///////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMERWF_H
#define QMCPLUSPLUS_PSIFORMERWF_H

#include <memory>
#include <string>
#include "QMCWaveFunctions/WaveFunctionComponent.h"

namespace pf { struct PsiFormer; struct Result; }

namespace qmcplusplus
{
class PsiFormerWF : public WaveFunctionComponent
{
public:
  PsiFormerWF(std::string name, std::string parameters, std::string configuration);
  PsiFormerWF(const PsiFormerWF&) = default;
  std::string getClassName() const override { return "PsiFormerWF"; }
  bool isFermionic() const override { return true; }
  LogValue evaluateLog(const ParticleSet&, ParticleSet::ParticleGradient&, ParticleSet::ParticleLaplacian&) override;
  void acceptMove(ParticleSet&, int, bool=false) override;
  void restore(int) override;
  PsiValue ratio(ParticleSet&, int) override;
  GradType evalGrad(ParticleSet&, int) override;
  PsiValue ratioGrad(ParticleSet&, int, GradType&) override;
  void registerData(ParticleSet&, WFBufferType&) override {}
  LogValue updateBuffer(ParticleSet&, WFBufferType&, bool=false) override;
  void copyFromBuffer(ParticleSet&, WFBufferType&) override {}
  void evaluateDerivatives(ParticleSet&, const OptVariables&, Vector<ValueType>&, Vector<ValueType>&) override {}
  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet&) const override;
private:
  pf::Result evaluate(const ParticleSet&, int active=-1) const;
  std::shared_ptr<pf::PsiFormer> model_;
  double current_sign_=1.0, proposed_sign_=1.0;
  LogValue proposed_log_value_=LogValue(0);
  bool has_proposal_=false;
};
}
#endif
