//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerWaveFunctionBuilder.h
 * @brief Builder for imported or deterministically initialized PsiFormer components.
 */
#ifndef QMCPLUSPLUS_PSIFORMERWAVEFUNCTIONBUILDER_H
#define QMCPLUSPLUS_PSIFORMERWAVEFUNCTIONBUILDER_H

#include "QMCWaveFunctions/WaveFunctionComponentBuilder.h"

namespace qmcplusplus
{
/** Parse model-construction and selected-index or full-network optimization controls. */
class PsiFormerWaveFunctionBuilder : public WaveFunctionComponentBuilder
{
public:
  /// Bind the builder to target electrons and the pool containing the declared source ions.
  PsiFormerWaveFunctionBuilder(Communicate* comm, ParticleSet& target, const PSetMap& particle_sets)
      : WaveFunctionComponentBuilder(comm, target), particle_sets_(particle_sets)
  {}

  /// Construct a fixed model by default or a selected/full-network optimizable model.
  std::unique_ptr<WaveFunctionComponent> buildComponent(xmlNodePtr current) override;

private:
  /// Particle sets supplying or validating nuclei and effective pseudopotential charges.
  const PSetMap& particle_sets_;
};

} // namespace qmcplusplus

#endif
