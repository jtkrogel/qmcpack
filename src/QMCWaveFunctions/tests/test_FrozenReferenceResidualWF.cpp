//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#include "QMCWaveFunctions/FrozenReferenceResidualWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/Jastrow/JastrowBuilder.h"
#include "MinimalWaveFunctionPool.h"
#include "Particle/tests/MinimalParticlePool.h"
#include "Particle/VirtualParticleSet.h"
#include "Utilities/ProjectData.h"
#include "OhmmsData/Libxml2Doc.h"
#include <cmath>
#include <sstream>

namespace qmcplusplus
{

#if !defined(QMC_COMPLEX) && !defined(MIXED_PRECISION) && defined(HAVE_EINSPLINE)

/// Evaluate one total component in zeroed caller-owned derivative arrays.
static WaveFunctionComponent::LogValue residualLog(WaveFunctionComponent& component, ParticleSet& particles)
{
  particles.G = WaveFunctionComponent::ValueType(0);
  particles.L = WaveFunctionComponent::ValueType(0);
  return component.evaluateLog(particles, particles.G, particles.L);
}

TEST_CASE("Frozen reference CPU sum supports diamond derivatives moves and clone isolation",
          "[wavefunction][residual][diamond]")
{
  ProjectData project("residual_diamond", ProjectData::DriverVersion::BATCH);
  auto particles = MinimalParticlePool::make_diamondC_1x1x1(OHMMS::Controller);
  auto pool = MinimalWaveFunctionPool::make_diamondC_1x1x1(project.getRuntimeOptions(), OHMMS::Controller, particles);
  auto& electrons = *particles.getParticleSet("e");
  auto& ions = *particles.getParticleSet("ion");
  auto& species = ions.getSpeciesSet();
  species(species.addAttribute("atomicnumber"), 0) = 6;
  species(species.addAttribute("valence"), 0) = 4;
  const double positions[8][3] = {{.31,.41,.53},{1.2,.2,.6},{.7,1.4,1.1},{2.4,.9,.33},
                                {1.1,.8,.4},{.2,1.2,.8},{1.8,1.7,.3},{.6,.9,1.9}};
  for (int e = 0; e < 8; ++e)
    for (int d = 0; d < 3; ++d)
      electrons.R[e][d] = positions[e][d];
  electrons.update();
  TrialWaveFunction& reference = *pool.getWaveFunction("psi0");

  // Add a nontrivial positive Jastrow, keeping the complete reference inside its branch.
  Libxml2Document jastrow_xml;
  REQUIRE(jastrow_xml.parseFromString(R"(
    <jastrow type="Two-Body" name="J2" function="bspline" print="no">
      <correlation speciesA="u" speciesB="d" size="4" rcut="2.2">
        <coefficients id="ud" type="Array">0.1 0.06 0.03 0.01</coefficients>
      </correlation>
    </jastrow>)"));
  JastrowBuilder jastrow_builder(OHMMS::Controller, electrons, particles.getPool());
  reference.addComponent(jastrow_builder.buildComponent(jastrow_xml.getRoot()));

  // Obtain the existing default profile through its initializer rather than duplicate its label.
  const psiformer::ModelShape shape{4, 4, 2, 16, 256, 4, 4};
  const auto profile = psiformer::initializePsiFormerParameters(shape, 19).profile;
  std::ostringstream neural_input;
  neural_input << "<wavefunction name='neural' target='e' print_parameters='no'>"
      "<psiformer name='neural' initialization='" << profile << "' initialization_seed='19' "
      "source='ion' system='pseudopotential' feature_policy='periodic_torus_v1' gpu='cpu'/>"
      "</wavefunction>";
  Libxml2Document neural_xml;
  REQUIRE(neural_xml.parseFromString(neural_input.str()));
  REQUIRE(pool.put(neural_xml.getRoot()));
  TrialWaveFunction& neural = *pool.getWaveFunction("neural");

  electrons.update(); // Refresh distance tables added while constructing the children.
  const double reference_log = reference.evaluateLog(electrons);
  const auto reference_gradient = electrons.G;
  const auto reference_laplacian = electrons.L;
  const double reference_phase = reference.getPhase();
  Libxml2Document sum_xml;
  REQUIRE(sum_xml.parseFromString(
      "<wavefunction name='sum_zero' target='e'><frozen_reference_sum "
      "reference='psi0' neural='neural' coefficient='0'/></wavefunction>"));
  REQUIRE(pool.put(sum_xml.getRoot()));
  TrialWaveFunction& exact = *pool.getWaveFunction("sum_zero");
  CHECK(exact.evaluateLog(electrons) == Catch::Approx(reference_log).margin(2e-10));
  CHECK(std::cos(exact.getPhase()) == Catch::Approx(std::cos(reference_phase)));
  for (int e = 0; e < 8; ++e)
  {
    for (int d = 0; d < 3; ++d)
      CHECK(electrons.G[e][d] == Catch::Approx(reference_gradient[e][d]).margin(2e-9));
    CHECK(electrons.L[e] == Catch::Approx(reference_laplacian[e]).margin(2e-8));
  }
  CHECK(exact.extractOptimizableObjectRefs().empty());
  CHECK(exact.extractStructuredParameterProviders().empty());

  auto& neural_component = dynamic_cast<PsiFormerWF&>(*neural.getOrbitals()[0]);
  const auto neural_log = residual::logarithmic(neural_component.evaluateResidualAmplitude(electrons));
  const double coefficient = 0.2 * std::exp(reference_log - static_cast<double>(neural_log.log_abs));
  FrozenReferenceResidualWF sum(electrons, reference, neural, coefficient);
  const auto baseline = residualLog(sum, electrons);
  const auto gradient = electrons.G;
  const auto laplacian = electrons.L;
  CHECK(sum.isOptimizable() == false);

  // Finite differences test the physical sum, independently of the branch-combination rule.
  const double h = 2e-4;
  double finite_laplacian = 0;
  const auto initial_position = electrons.R[0];
  for (int d = 0; d < 3; ++d)
  {
    electrons.R[0][d] = initial_position[d] + h;
    electrons.update();
    const double plus = std::real(residualLog(sum, electrons));
    electrons.R[0][d] = initial_position[d] - h;
    electrons.update();
    const double minus = std::real(residualLog(sum, electrons));
    electrons.R[0][d] = initial_position[d];
    electrons.update();
    CHECK((plus-minus)/(2*h) == Catch::Approx(gradient[0][d]).epsilon(2e-5).margin(2e-5));
    finite_laplacian += (plus+minus-2*std::real(baseline))/(h*h);
  }
  CHECK(finite_laplacian == Catch::Approx(laplacian[0]).epsilon(2e-4).margin(2e-4));
  residualLog(sum, electrons);

  VirtualParticleSet virtuals(electrons);
  const std::vector<ParticleSet::PosType> displacements = {{.03,.01,0},{-.04,0,.02}};
  virtuals.makeMoves(electrons, 0, displacements);
  std::vector<WaveFunctionComponent::ValueType> ratios(2);
  sum.evaluateRatios(virtuals, ratios);
  for (int k = 0; k < 2; ++k)
  {
    const auto old = electrons.R[0];
    electrons.R[0] += displacements[k];
    electrons.update();
    const auto moved = residualLog(sum, electrons);
    const double expected = std::exp(std::real(moved-baseline))*std::cos(std::imag(moved-baseline));
    CHECK(ratios[k] == Catch::Approx(expected).epsilon(2e-9).margin(2e-9));
    electrons.R[0] = old;
    electrons.update();
  }
  CHECK(residualLog(sum, electrons) == baseline);

  // Two walker clones stage an identical all-particle proposal; one commits, one rolls back.
  ParticleSet clone_particles(electrons);
  clone_particles.update();
  auto clone = sum.makeClone(clone_particles);
  residualLog(*clone, clone_particles);
  RefVectorWithLeader<WaveFunctionComponent> components(sum);
  components.push_back(sum);
  components.push_back(*clone);
  RefVectorWithLeader<ParticleSet> physical(electrons);
  physical.push_back(electrons);
  physical.push_back(clone_particles);
  std::vector<int> indices;
  std::vector<ParticleSet::PosType> proposed_positions;
  for (int lane = 0; lane < 2; ++lane)
    for (int e = 0; e < 8; ++e)
    {
      indices.push_back(e);
      auto position = electrons.R[e];
      position[0] += .012;
      proposed_positions.push_back(position);
    }
  const MCMultiParticleMoves<CoordsType::POS> moves({0,8,16}, indices, proposed_positions);
  ParticleSet::ParticleGradient g0(8), g1(8);
  ParticleSet::ParticleLaplacian l0(8), l1(8);
  g0=0; g1=0; l0=0; l1=0;
  RefVector<ParticleSet::ParticleGradient> gradients;
  gradients.push_back(g0); gradients.push_back(g1);
  RefVector<ParticleSet::ParticleLaplacian> laplacians;
  laplacians.push_back(l0); laplacians.push_back(l1);
  std::vector<WaveFunctionComponent::LogValue> log_ratios(2);
  sum.mw_evaluateMultiParticleMove(components, physical, moves, log_ratios, gradients, laplacians);
  CHECK(sum.get_log_value() == baseline);
  CHECK_THROWS(sum.mw_evaluateMultiParticleMove(components, physical, moves, log_ratios, gradients, laplacians));
  sum.mw_accept_rejectMultiParticleMove(components, physical, moves, {true,false});
  CHECK(clone->get_log_value() == baseline);
  for (int e = 0; e < 8; ++e)
    electrons.R[e] = proposed_positions[e];
  electrons.update();
  CHECK(std::real(residualLog(sum, electrons)) == Catch::Approx(std::real(baseline + log_ratios[0])).margin(2e-11));
  CHECK(residualLog(*clone, clone_particles) == baseline);

  // Native buffers contain total state only; clone-local child scratch need not be serialized.
  WaveFunctionComponent::WFBufferType buffer;
  sum.registerData(electrons, buffer);
  buffer.allocate();
  buffer.rewind();
  electrons.G=0; electrons.L=0;
  sum.updateBuffer(electrons, buffer);
  auto restarted = sum.makeClone(electrons);
  buffer.rewind();
  restarted->copyFromBuffer(electrons, buffer);
  CHECK(residualLog(*restarted, electrons) == sum.get_log_value());

  // Changing the separately named neural definition cannot alter the frozen sum model family.
  const auto frozen = sum.get_log_value();
  auto candidate = neural_component.snapshotParameters();
  candidate.values[0] += 0.25;
  neural_component.publishParameters(candidate, candidate.version);
  const auto saved_position = electrons.R[0];
  electrons.R[0][0] += .017;
  electrons.update();
  residualLog(sum, electrons);
  electrons.R[0] = saved_position;
  electrons.update();
  CHECK(residualLog(sum, electrons) == frozen);
  CHECK(reference.evaluateLog(clone_particles) == Catch::Approx(reference_log).margin(2e-10));

  Libxml2Document bad_xml;
  REQUIRE(bad_xml.parseFromString(
      "<wavefunction name='bad_sum' target='e'><frozen_reference_sum reference='psi0' "
      "neural='neural' coefficient='nan'/></wavefunction>"));
  CHECK_THROWS(pool.put(bad_xml.getRoot()));
}

#endif
} // namespace qmcplusplus
