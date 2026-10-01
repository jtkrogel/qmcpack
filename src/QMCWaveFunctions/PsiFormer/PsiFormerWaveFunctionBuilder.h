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
/** Parse PsiFormer export paths and selected-index or full-network optimization controls. */
class PsiFormerWaveFunctionBuilder : public WaveFunctionComponentBuilder
{
public:
  /// Bind the builder to the target electron set; no auxiliary particle set is required.
  PsiFormerWaveFunctionBuilder(Communicate* comm, ParticleSet& target, const PSetMap&)
      : WaveFunctionComponentBuilder(comm, target)
  {}

  /// Construct a fixed model by default or a selected/full-network optimizable model.
  std::unique_ptr<WaveFunctionComponent> buildComponent(xmlNodePtr current) override;
};

} // namespace qmcplusplus

#endif
