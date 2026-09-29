///////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
///////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMERWAVEFUNCTIONBUILDER_H
#define QMCPLUSPLUS_PSIFORMERWAVEFUNCTIONBUILDER_H
#include "QMCWaveFunctions/WaveFunctionComponentBuilder.h"
namespace qmcplusplus
{
class PsiFormerWaveFunctionBuilder : public WaveFunctionComponentBuilder
{
public:
  PsiFormerWaveFunctionBuilder(Communicate* c, ParticleSet& p, const PSetMap& pool)
      : WaveFunctionComponentBuilder(c,p) {}
  std::unique_ptr<WaveFunctionComponent> buildComponent(xmlNodePtr) override;
};
}
#endif
