//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_wf.cpp
 * @brief Selected-parameter QMCPACK integration tests for PsiFormerWF.
 */
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Message/Communicate.h"
#include "OhmmsData/Libxml2Doc.h"
#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWaveFunctionBuilder.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "Utilities/RuntimeOptions.h"
#include "psiformer_test_utils.h"

#include <cmath>
#include <complex>
#include <sstream>
#include <vector>

namespace qmcplusplus
{
namespace
{
using namespace testing::psiformer;
using ValueType = QMCTraits::ValueType;

/// Construct the four-electron ParticleSet matching the generated LiH fixture.
ParticleSet makeLiHElectrons(const SimulationCell& simulation_cell)
{
  const Geometry geometry = makeGeometry("lih");
  ParticleSet electrons(simulation_cell);
  electrons.setName("e");
  electrons.create({2, 2});
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      electrons.R[electron][dimension] = geometry.electrons[3 * electron + dimension];
  electrons.update();
  return electrons;
}

/// Add the gradient of a fixed linear log factor to emulate composition with another component.
std::vector<double> addLinearLogGradient(ParticleSet& electrons)
{
  std::vector<double> extra_gradient(3 * electrons.getTotalNum());
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      const std::size_t coordinate   = 3 * electron + dimension;
      extra_gradient[coordinate]     = 0.01 * (dimension + 1);
      electrons.G[electron][dimension] += ValueType(extra_gradient[coordinate]);
    }
  return extra_gradient;
}

/// Evaluate the kinetic local energy from the logarithmic gradients and Laplacians in ParticleSet.
double kineticEnergy(const ParticleSet& electrons)
{
  double kinetic = 0.0;
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
  {
    double squared_gradient = 0.0;
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      const double real_part = std::real(electrons.G[electron][dimension]);
      const double imag_part = std::imag(electrons.G[electron][dimension]);
      squared_gradient += real_part * real_part + imag_part * imag_part;
    }
    kinetic -= 0.5 * (std::real(electrons.L[electron]) + squared_gradient);
  }
  return kinetic;
}
} // namespace

TEST_CASE("PsiFormer builder is fixed by default and parses selected indices", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  WaveFunctionComponentBuilder::PSetMap particle_sets;
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, electrons, particle_sets);

  std::ostringstream fixed_xml;
  fixed_xml << "<psiformer name=\"pf_fixed\" parameters=\"" << files.parameters.string()
            << "\" configuration=\"" << files.configuration.string() << "\"/>";
  Libxml2Document fixed_document;
  REQUIRE(fixed_document.parseFromString(fixed_xml.str()));
  std::unique_ptr<WaveFunctionComponent> fixed = builder.buildComponent(fixed_document.getRoot());
  REQUIRE(fixed != nullptr);
  CHECK_FALSE(fixed->isOptimizable());
  UniqueOptObjRefs fixed_refs;
  fixed->extractOptimizableObjectRefs(fixed_refs);
  CHECK(fixed_refs.empty());

  std::ostringstream optimized_xml;
  optimized_xml << "<psiformer name=\"pf_selected\" parameters=\"" << files.parameters.string()
                << "\" configuration=\"" << files.configuration.string()
                << "\" optimize=\"yes\" optimize_scope=\"indices\" optimize_indices=\"127, 0 1\"/>";
  Libxml2Document optimized_document;
  REQUIRE(optimized_document.parseFromString(optimized_xml.str()));
  std::unique_ptr<WaveFunctionComponent> optimized = builder.buildComponent(optimized_document.getRoot());
  REQUIRE(optimized != nullptr);
  CHECK(optimized->isOptimizable());
  UniqueOptObjRefs optimized_refs;
  optimized->extractOptimizableObjectRefs(optimized_refs);
  REQUIRE(optimized_refs.size() == 1);

  std::ostringstream unsupported_xml;
  unsupported_xml << "<psiformer parameters=\"" << files.parameters.string() << "\" configuration=\""
                  << files.configuration.string()
                  << "\" optimize=\"yes\" optimize_scope=\"all\" optimize_indices=\"0\"/>";
  Libxml2Document unsupported_document;
  REQUIRE(unsupported_document.parseFromString(unsupported_xml.str()));
  CHECK_THROWS_AS(builder.buildComponent(unsupported_document.getRoot()), std::invalid_argument);
}

TEST_CASE("PsiFormer selected parameters follow QMCPACK registration reset and derivative paths",
          "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  RuntimeOptions runtime_options;
  TrialWaveFunction trial_wavefunction(runtime_options, "psiformer_selected_test");
  trial_wavefunction.addComponent(std::make_unique<PsiFormerWF>(
      "pf", files.parameters.string(), files.configuration.string(), true, std::vector<std::size_t>{127, 0, 1}));

  OptVariables active;
  trial_wavefunction.checkInVariables(active);
  active.resetIndex();
  trial_wavefunction.checkOutVariables(active);
  REQUIRE(active.size() == 3);
  CHECK(active.name(0) == "pf_pf_0000000");
  CHECK(active.name(1) == "pf_pf_0000001");
  CHECK(active.name(2) == "pf_pf_0000127");
  REQUIRE(trial_wavefunction.extractOptimizableObjectRefs().size() == 1);

  const double baseline_log = trial_wavefunction.evaluateLog(electrons);
  addLinearLogGradient(electrons);

  Vector<ValueType> dlogpsi(active.size());
  Vector<ValueType> dhpsioverpsi(active.size());
  Vector<ValueType> dlogpsi_wf(active.size());
  dlogpsi       = ValueType(-0.125);
  dhpsioverpsi  = ValueType(0.625);
  dlogpsi_wf    = ValueType(0.375);
  trial_wavefunction.evaluateDerivatives(electrons, active, dlogpsi, dhpsioverpsi);
  trial_wavefunction.evaluateDerivativesWF(electrons, active, dlogpsi_wf);

  const double log_derivative = std::real(dlogpsi[0]) + 0.125;
  const double kinetic_derivative = std::real(dhpsioverpsi[0]) - 0.625;
  CHECK(std::real(dlogpsi_wf[0]) - 0.375 == Catch::Approx(log_derivative).epsilon(2e-10).margin(2e-10));

  const double original_value = active[0];
  const double parameter_step = 2e-5;
  active[0] = original_value + parameter_step;
  trial_wavefunction.resetParameters(active);
  const double plus_log = trial_wavefunction.evaluateLog(electrons);
  addLinearLogGradient(electrons);
  const double plus_kinetic = kineticEnergy(electrons);

  active[0] = original_value - parameter_step;
  trial_wavefunction.resetParameters(active);
  const double minus_log = trial_wavefunction.evaluateLog(electrons);
  addLinearLogGradient(electrons);
  const double minus_kinetic = kineticEnergy(electrons);

  active[0] = original_value;
  trial_wavefunction.resetParameters(active);
  const double restored_log = trial_wavefunction.evaluateLog(electrons);
  CHECK(restored_log == Catch::Approx(baseline_log).epsilon(2e-10).margin(2e-10));

  const double log_finite_difference = (plus_log - minus_log) / (2 * parameter_step);
  const double kinetic_finite_difference = (plus_kinetic - minus_kinetic) / (2 * parameter_step);
  CHECK(log_finite_difference == Catch::Approx(log_derivative).epsilon(5e-5).margin(5e-5));
  CHECK(kinetic_finite_difference == Catch::Approx(kinetic_derivative).epsilon(2e-4).margin(2e-4));

  // Apply one deterministic first-order update through the public reset path.
  // This is the small vertical slice used before streaming full-network descent.
  active[0] = original_value - 1e-4 * log_derivative;
  trial_wavefunction.resetParameters(active);
  const double updated_log = trial_wavefunction.evaluateLog(electrons);
  CHECK(std::abs(updated_log - baseline_log) > 1e-8);
}

} // namespace qmcplusplus
