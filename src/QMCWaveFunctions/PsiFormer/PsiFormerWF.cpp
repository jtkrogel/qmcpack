///////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
///////////////////////////////////////////////////////////////////////
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.inc"

#include <cmath>
#include <complex>
#include <stdexcept>

namespace qmcplusplus
{
PsiFormerWF::PsiFormerWF(std::string name, std::string parameters, std::string configuration)
    : WaveFunctionComponent(std::move(name)), model_(std::make_shared<pf::PsiFormer>(parameters, configuration)) {}

pf::Result PsiFormerWF::evaluate(const ParticleSet& p, int active) const
{
  pf::Tensor positions({static_cast<size_t>(p.getTotalNum()), 3});
  for (int i=0; i<p.getTotalNum(); ++i)
  {
    const auto& r=(i==active)?p.activeR(i):p.R[i];
    for (int d=0; d<3; ++d) positions.x[3*i+d]=r[d];
  }
  if (static_cast<size_t>(p.getTotalNum()) != model_->ne)
    throw std::runtime_error("PsiFormerWF electron count differs from exported model");
  return model_->evaluate(positions, false);
}

PsiFormerWF::LogValue PsiFormerWF::evaluateLog(const ParticleSet& p, ParticleSet::ParticleGradient& g,
                                                ParticleSet::ParticleLaplacian& l)
{
  auto r=evaluate(p);
  current_sign_=r.sign;
  log_value_=LogValue(r.logabs, r.sign < 0 ? M_PI : 0.0);
  for (int i=0; i<p.getTotalNum(); ++i)
  {
    for (int d=0; d<3; ++d) g[i][d]+=r.gradient[3*i+d];
    l[i]+=r.lap_log[i];
  }
  return log_value_;
}

PsiFormerWF::PsiValue PsiFormerWF::ratio(ParticleSet& p, int iat)
{
  auto r=evaluate(p,iat); proposed_sign_=r.sign;
  proposed_log_value_=LogValue(r.logabs,r.sign<0?M_PI:0.0); has_proposal_=true;
  return (proposed_sign_/current_sign_)*std::exp(std::real(proposed_log_value_-log_value_));
}

PsiFormerWF::GradType PsiFormerWF::evalGrad(ParticleSet& p, int iat)
{
  auto r=evaluate(p); GradType g; for(int d=0;d<3;++d)g[d]=r.gradient[3*iat+d]; return g;
}

PsiFormerWF::PsiValue PsiFormerWF::ratioGrad(ParticleSet& p, int iat, GradType& g)
{
  auto r=evaluate(p,iat); proposed_sign_=r.sign;
  proposed_log_value_=LogValue(r.logabs,r.sign<0?M_PI:0.0); has_proposal_=true;
  for(int d=0;d<3;++d)g[d]=r.gradient[3*iat+d];
  return (proposed_sign_/current_sign_)*std::exp(std::real(proposed_log_value_-log_value_));
}

void PsiFormerWF::acceptMove(ParticleSet&, int, bool)
{
  if(has_proposal_){log_value_=proposed_log_value_;current_sign_=proposed_sign_;}has_proposal_=false;
}
void PsiFormerWF::restore(int){has_proposal_=false;}
PsiFormerWF::LogValue PsiFormerWF::updateBuffer(ParticleSet& p, WFBufferType&, bool)
{ return evaluateLog(p,p.G,p.L); }
std::unique_ptr<WaveFunctionComponent> PsiFormerWF::makeClone(ParticleSet&) const
{ return std::make_unique<PsiFormerWF>(*this); }
}
