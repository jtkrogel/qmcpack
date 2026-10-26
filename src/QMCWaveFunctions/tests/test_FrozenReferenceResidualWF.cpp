//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#include "QMCWaveFunctions/FrozenReferenceResidualWF.h"
#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/Jastrow/JastrowBuilder.h"
#include "MinimalWaveFunctionPool.h"
#include "Particle/tests/MinimalParticlePool.h"
#include "Particle/VirtualParticleSet.h"
#include "Utilities/ProjectData.h"
#include "OhmmsData/Libxml2Doc.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
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

/// Retain one selected VJP entry, independently of the producer's parameter chunk boundaries.
class ResidualSelectedScoreSink final : public wftrain::ParameterReductionSink
{
public:
  explicit ResidualSelectedScoreSink(std::size_t selected) : selected_(selected) {}
  wftrain::DerivativeValue value{};

protected:
  /// Extract the selected canonical parameter from exactly one incoming chunk.
  void consume(std::size_t channel, const wftrain::ParameterChunkConstView& chunk) override
  {
    REQUIRE(channel == 0);
    const auto offset = chunk.descriptor().parameter_offset;
    if (selected_ >= offset && selected_ - offset < chunk.values().size())
      value = chunk.values()[selected_ - offset];
  }

private:
  std::size_t selected_;
};

/// Retain one score JVP sample for comparison with an independently evaluated score.
class ResidualSampleScoreSink final : public wftrain::SampleProductSink
{
public:
  wftrain::DerivativeValue value{};

protected:
  /// The test binds a single physical configuration and expects a single output tile.
  void consume(const wftrain::SampleProductTileConstView& tile) override
  {
    REQUIRE(tile.values().size() == 1);
    value = tile.values()[0];
  }
};

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

  // A parent publication reaches walker clones, invalidates their caches, and leaves the SJ fixed.
  sum.enableNeuralTraining();
  CHECK(sum.structuredParameterProvider() == &sum);
  auto trained_clone_base = sum.makeClone(clone_particles);
  auto& trained_clone = dynamic_cast<FrozenReferenceResidualWF&>(*trained_clone_base);
  CHECK(trained_clone.structuredParameterProvider() == &trained_clone);
  auto parent_candidate = sum.snapshotParameters();
  CHECK(parent_candidate.schema_fingerprint == sum.parameterSchema().fingerprint());
  const auto original_snapshot = parent_candidate;
  std::vector<double> parent_score;
  sum.evaluateParameterScore(electrons, parent_score);
  REQUIRE(parent_score.size() == parent_candidate.values.size());
  const auto largest = static_cast<std::size_t>(std::distance(parent_score.begin(),
      std::max_element(parent_score.begin(), parent_score.end(),
                       [](double a, double b) { return std::abs(a) < std::abs(b); })));
  // Independent checked products agree with the direct parent score without retaining score rows.
  RefVectorWithLeader<WaveFunctionComponent> score_components(sum);
  score_components.push_back(sum);
  RefVectorWithLeader<ParticleSet> score_particles(electrons);
  score_particles.push_back(electrons);
  auto stream = sum.makeStreamingDerivativeOperator(score_components, score_particles, 3, 7, 257);
  auto second_stream = sum.makeStreamingDerivativeOperator(score_components, score_particles, 4, 8, 257);
  CHECK(stream->storageDiagnostics().shared_workspace_identity ==
        second_stream->storageDiagnostics().shared_workspace_identity);
  std::vector<wftrain::DerivativeValue> direction(parent_score.size());
  direction[largest] = {0.7, 0};
  ResidualSampleScoreSink jvp_sink;
  stream->applyScoreJVP({sum.parameterSchema(), original_snapshot.version,
                        {direction.data(), direction.size()}}, jvp_sink);
  CHECK(jvp_sink.value.real() == Catch::Approx(0.7 * parent_score[largest]).margin(1e-10));
  CHECK(jvp_sink.value.imag() == Catch::Approx(0).margin(1e-10));
  const wftrain::DerivativeValue coefficient_weight{0.2, 0.5};
  const wftrain::VJPCoefficientChannel channel{"weighted", wftrain::DerivativeProduct::SCORE_VJP,
      {sum.parameterSchema().providerId(), sum.parameterSchema().fingerprint(), original_snapshot.version,
       3, 7, {&coefficient_weight, 1}}, 0};
  ResidualSelectedScoreSink vjp_sink(largest);
  stream->applyVJPs({&channel, 1}, wftrain::DerivativeAdjoint::HERMITIAN, vjp_sink);
  CHECK(vjp_sink.value.real() == Catch::Approx(0.2 * parent_score[largest]).margin(1e-10));
  CHECK(vjp_sink.value.imag() == Catch::Approx(0.5 * parent_score[largest]).margin(1e-10));
  const double parameter_delta = 1e-5;
  parent_candidate.values[largest] += parameter_delta;
  const auto next_version = sum.publishParameters(parent_candidate, parent_candidate.version);
  CHECK(next_version > original_snapshot.version);
  ResidualSampleScoreSink stale_sink;
  CHECK_THROWS(stream->applyScoreJVP({sum.parameterSchema(), original_snapshot.version,
                                    {direction.data(), direction.size()}}, stale_sink));
  CHECK(trained_clone.snapshotParameters().version == next_version);
  const double parent_plus = std::real(residualLog(sum, electrons));
  CHECK(std::real(residualLog(trained_clone, electrons)) == Catch::Approx(parent_plus).margin(1e-10));
  CHECK_THROWS(sum.publishParameters(parent_candidate, original_snapshot.version));
  parent_candidate.version = next_version;
  parent_candidate.values[largest] -= 2 * parameter_delta;
  const auto minus_version = sum.publishParameters(parent_candidate, next_version);
  const double parent_minus = std::real(residualLog(sum, electrons));
  CHECK((parent_plus - parent_minus) / (2 * parameter_delta) ==
        Catch::Approx(parent_score[largest]).epsilon(2e-5).margin(2e-7));
  parent_candidate = original_snapshot;
  parent_candidate.version = minus_version;
  sum.publishParameters(parent_candidate, minus_version);
  CHECK(residualLog(sum, electrons) == frozen);
  CHECK(reference.evaluateLog(clone_particles) == Catch::Approx(reference_log).margin(2e-10));
  CHECK_THROWS(dynamic_cast<FrozenReferenceResidualWF&>(*exact.getOrbitals()[0]).enableNeuralTraining());

  // A model artifact reconstructs the complete sum contract and rejects a different relative scale.
  const auto model_file = "residual_parent_model_rank_" + std::to_string(OHMMS::Controller->rank()) + ".h5";
  sum.exportModel(model_file);
  CHECK(sum.readModel(model_file).values == sum.snapshotParameters().values);
  CHECK(trained_clone.frozenModelIdentity() == sum.frozenModelIdentity());
  auto imported_base = sum.makeClone(clone_particles);
  auto& imported = dynamic_cast<FrozenReferenceResidualWF&>(*imported_base);
  imported.isolateParameterState();
  imported.importModel(model_file);
  CHECK(residualLog(imported, electrons) == frozen);
  FrozenReferenceResidualWF different_coefficient(electrons, reference, neural, coefficient * 1.1);
  CHECK_THROWS(different_coefficient.readModel(model_file));
  // Mutating the external named reference must not leak into the parent's frozen Jastrow.
  OptVariables reference_variables;
  reference.checkInVariables(reference_variables);
  reference.checkOutVariables(reference_variables);
  REQUIRE(reference_variables.size() > 0);
  reference_variables[0] += 0.03;
  reference.resetParameters(reference_variables);
  electrons.R[0][0] += .013;
  electrons.update();
  residualLog(sum, electrons);
  electrons.R[0][0] -= .013;
  electrons.update();
  CHECK(residualLog(sum, electrons) == frozen);
  FrozenReferenceResidualWF changed_reference(electrons, reference, neural, coefficient);
  CHECK(changed_reference.frozenModelIdentity() != sum.frozenModelIdentity());
  CHECK_THROWS(changed_reference.readModel(model_file));
  std::filesystem::remove(model_file);

  Libxml2Document bad_xml;
  REQUIRE(bad_xml.parseFromString(
      "<wavefunction name='bad_sum' target='e'><frozen_reference_sum reference='psi0' "
      "neural='neural' coefficient='nan'/></wavefunction>"));
  CHECK_THROWS(pool.put(bad_xml.getRoot()));
}

#endif
} // namespace qmcplusplus
