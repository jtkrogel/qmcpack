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

#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorPlanning.h"
#include "QMCWaveFunctions/WaveFunctionComponentBuilder.h"

namespace qmcplusplus
{
/** Narrow injection seam for lazily inspecting production accelerator readiness.
 *
 * The builder never calls this provider for an omitted, ``no``, or ``cpu``
 * request. Tests use that property to guard CPU-only startup from accidental
 * device-runtime initialization.
 */
class PsiFormerAcceleratorReadinessProvider
{
public:
  virtual ~PsiFormerAcceleratorReadinessProvider() = default;

  /// Inspect compiled/runtime readiness without claiming numerical coverage.
  virtual psiformer::PsiFormerAcceleratorReadiness inspect() const = 0;
};

/** Parse model-construction and selected-index or full-network optimization controls. */
class PsiFormerWaveFunctionBuilder : public WaveFunctionComponentBuilder
{
public:
  /// Bind the builder to target electrons and the pool containing the declared source ions.
  PsiFormerWaveFunctionBuilder(
      Communicate* comm,
      ParticleSet& target,
      const PSetMap& particle_sets,
      const PsiFormerAcceleratorReadinessProvider* accelerator_readiness_provider = nullptr)
      : WaveFunctionComponentBuilder(comm, target),
        particle_sets_(particle_sets),
        accelerator_readiness_provider_(accelerator_readiness_provider)
  {}

  /// Construct a fixed model by default or a selected/full-network optimizable model.
  std::unique_ptr<WaveFunctionComponent> buildComponent(xmlNodePtr current) override;

private:
  /// Particle sets supplying or validating nuclei and effective pseudopotential charges.
  const PSetMap& particle_sets_;
  /// Optional non-owning test seam; null selects the process-global DeviceManager.
  const PsiFormerAcceleratorReadinessProvider* accelerator_readiness_provider_;
};

} // namespace qmcplusplus

#endif
