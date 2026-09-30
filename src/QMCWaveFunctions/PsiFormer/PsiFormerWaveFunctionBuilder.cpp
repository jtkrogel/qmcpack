//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerWaveFunctionBuilder.cpp
 * @brief XML construction of the PsiFormer wavefunction component.
 */
#include "QMCWaveFunctions/PsiFormer/PsiFormerWaveFunctionBuilder.h"
#include "OhmmsData/AttributeSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include <stdexcept>

namespace qmcplusplus
{
std::unique_ptr<WaveFunctionComponent> PsiFormerWaveFunctionBuilder::buildComponent(xmlNodePtr cur)
{
  std::string name = "psiformer", parameters, configuration;

  // Both files use the compact export format consumed by PsiFormerNative.inc.
  // The configuration file supplies nuclei, charges, spin populations, and
  // electron count.
  OhmmsAttributeSet a;
  a.add(name, "name");
  a.add(parameters, "parameters");
  a.add(configuration, "configuration");
  a.put(cur);
  if (parameters.empty() || configuration.empty())
    throw std::runtime_error("psiformer requires parameters and configuration HDF5 paths");
  return std::make_unique<PsiFormerWF>(name, parameters, configuration);
}

} // namespace qmcplusplus
