//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerWaveFunctionBuilder.h
 * @brief Builder for a PsiFormer wavefunction component imported from HDF5.
 */
#ifndef QMCPLUSPLUS_PSIFORMERWAVEFUNCTIONBUILDER_H
#define QMCPLUSPLUS_PSIFORMERWAVEFUNCTIONBUILDER_H

#include "QMCWaveFunctions/WaveFunctionComponentBuilder.h"

namespace qmcplusplus
{
/** Parses the PsiFormer XML attributes and constructs the fixed native
 * evaluator. */
class PsiFormerWaveFunctionBuilder : public WaveFunctionComponentBuilder
{
public:
  PsiFormerWaveFunctionBuilder(Communicate* comm, ParticleSet& target, const PSetMap&)
      : WaveFunctionComponentBuilder(comm, target)
  {}
  std::unique_ptr<WaveFunctionComponent> buildComponent(xmlNodePtr current) override;
};

} // namespace qmcplusplus

#endif
