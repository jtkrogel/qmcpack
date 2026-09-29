///////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
///////////////////////////////////////////////////////////////////////
#include "QMCWaveFunctions/PsiFormer/PsiFormerWaveFunctionBuilder.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "OhmmsData/AttributeSet.h"
#include <stdexcept>
namespace qmcplusplus
{
std::unique_ptr<WaveFunctionComponent> PsiFormerWaveFunctionBuilder::buildComponent(xmlNodePtr cur)
{
  std::string name="psiformer", parameters, configuration;
  OhmmsAttributeSet a; a.add(name,"name");a.add(parameters,"parameters");a.add(configuration,"configuration");a.put(cur);
  if(parameters.empty()||configuration.empty()) throw std::runtime_error("psiformer requires parameters and configuration HDF5 paths");
  return std::make_unique<PsiFormerWF>(name,parameters,configuration);
}
}
