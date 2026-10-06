//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_streaming_derivative.cpp
 * @brief High-level score-product tests for the bounded PsiFormer provider.
 */

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "psiformer_test_utils.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace qmcplusplus
{
namespace
{
using wftrain::CoefficientView;
using wftrain::DerivativeArrayView;
using wftrain::DerivativeProduct;
using wftrain::DerivativeSinkState;
using wftrain::DerivativeValue;
using wftrain::ParameterChunkConstView;
using wftrain::ParameterChunkPlan;
using wftrain::ParameterReductionSink;
using wftrain::SampleProductSink;
using wftrain::SampleProductTileConstView;
using wftrain::StreamingDerivativeOperator;
using wftrain::VJPCoefficientChannel;

/// Construct the electron ParticleSet matching the generated LiH fixture.
ParticleSet makeElectrons(const SimulationCell& simulation_cell)
{
  const testing::psiformer::Geometry geometry =
      testing::psiformer::makeGeometry("lih");
  ParticleSet electrons(simulation_cell);
  electrons.setName("e");
  electrons.create({static_cast<int>(geometry.nup),
                    static_cast<int>(geometry.electrons.size() / 3 -
                                     geometry.nup)});
  SpeciesSet& species = electrons.getSpeciesSet();
  species.addSpecies("u");
  species.addSpecies("d");
  const int mass = species.addAttribute("mass");
  species(mass, 0) = 1.0;
  species(mass, 1) = 1.0;
  electrons.resetGroups();
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      electrons.R[electron][dimension] =
          geometry.electrons[3 * electron + dimension];
  electrons.update();
  return electrons;
}

/// Register the component's small selected subset through the legacy oracle route.
OptVariables registerSelectedParameters(PsiFormerWF& component)
{
  OptVariables active;
  component.checkInVariablesExclusive(active);
  active.resetIndex();
  component.checkOutVariables(active);
  return active;
}

/// Retain only selected canonical entries while checking complete chunk delivery.
class SelectedParameterSink final : public ParameterReductionSink
{
public:
  explicit SelectedParameterSink(std::vector<std::size_t> selected)
      : selected_(std::move(selected))
  {}

  const std::vector<std::vector<DerivativeValue>>& results() const
  {
    if (state() != DerivativeSinkState::COMPLETE)
      throw std::logic_error("Selected parameter sink has no completed result");
    return results_;
  }

  std::size_t consumedRecords() const noexcept { return consumed_records_; }

  /// Make the next accepted chunk fail after the checked transaction begins.
  void armConsumeFailure() noexcept { fail_next_consume_ = true; }

protected:
  void onBegin(const wftrain::DerivativeStreamDescriptor&,
               const ParameterChunkPlan&,
               DerivativeArrayView<const VJPCoefficientChannel> channels) override
  {
    results_.assign(channels.size(),
                    std::vector<DerivativeValue>(selected_.size()));
    consumed_records_ = 0;
  }

  void consume(std::size_t channel, const ParameterChunkConstView& chunk) override
  {
    if (fail_next_consume_)
    {
      fail_next_consume_ = false;
      throw std::runtime_error("injected selected-parameter sink failure");
    }
    for (std::size_t probe = 0; probe < selected_.size(); ++probe)
      if (selected_[probe] >= chunk.descriptor().parameter_offset &&
          selected_[probe] <
              chunk.descriptor().parameter_offset + chunk.descriptor().count)
        results_[channel][probe] =
            chunk.values()[selected_[probe] -
                           chunk.descriptor().parameter_offset];
    ++consumed_records_;
  }

  void onAbort() noexcept override { results_.clear(); }

  void onReset() noexcept override
  {
    results_.clear();
    consumed_records_ = 0;
  }

private:
  std::vector<std::size_t> selected_;
  std::vector<std::vector<DerivativeValue>> results_;
  std::size_t consumed_records_ = 0;
  bool fail_next_consume_       = false;
};

/// Collect the one-sample score-product tiles in their checked order.
class RecordingSampleSink final : public SampleProductSink
{
public:
  const std::vector<DerivativeValue>& results() const
  {
    if (state() != DerivativeSinkState::COMPLETE)
      throw std::logic_error("Recording sample sink has no completed result");
    return results_;
  }

protected:
  void onBegin(const wftrain::DerivativeStreamDescriptor& descriptor) override
  {
    first_sample_ = descriptor.sample_offset;
    results_.assign(descriptor.sample_count, DerivativeValue{});
  }

  void consume(const SampleProductTileConstView& tile) override
  {
    REQUIRE(tile.values().size() == 1);
    results_[tile.descriptor().sample_offset - first_sample_] = tile.values()[0];
  }

  void onAbort() noexcept override { results_.clear(); }

  void onReset() noexcept override
  {
    first_sample_ = 0;
    results_.clear();
  }

private:
  std::size_t first_sample_ = 0;
  std::vector<DerivativeValue> results_;
};

/// Form coefficient metadata tied to one prepared provider and backing vector.
CoefficientView coefficientsFor(const StreamingDerivativeOperator& op,
                                const std::vector<DerivativeValue>& coefficients)
{
  return {op.parameterSchema().providerId(),
          op.parameterSchema().fingerprint(),
          op.parameterVersion(),
          op.batchOrdinal(),
          op.sampleOffset(),
          {coefficients.data(), coefficients.size()}};
}

/// Compare a complex full-precision product with a mixed numerical tolerance.
void checkClose(DerivativeValue actual, DerivativeValue expected)
{
  CHECK(actual.real() ==
        Catch::Approx(expected.real()).epsilon(3e-10).margin(3e-10));
  CHECK(actual.imag() ==
        Catch::Approx(expected.imag()).epsilon(3e-10).margin(3e-10));
}

/// Minimal non-provider component used to exercise the unsupported factory default.
class UnsupportedStreamingComponent final : public WaveFunctionComponent
{
public:
  UnsupportedStreamingComponent() : WaveFunctionComponent("unsupported_streaming") {}

  std::string getClassName() const override { return "UnsupportedStreamingComponent"; }

  LogValue evaluateLog(const ParticleSet&,
                       ParticleSet::ParticleGradient&,
                       ParticleSet::ParticleLaplacian&) override
  {
    return {};
  }

  void acceptMove(ParticleSet&, int, bool) override {}

  void restore(int) override {}

  PsiValue ratio(ParticleSet&, int) override { return PsiValue{1}; }

  void registerData(ParticleSet&, WFBufferType&) override {}

  LogValue updateBuffer(ParticleSet&, WFBufferType&, bool) override { return {}; }

  void copyFromBuffer(ParticleSet&, WFBufferType&) override {}

  void evaluateDerivatives(ParticleSet&,
                           const OptVariables&,
                           Vector<ValueType>&,
                           Vector<ValueType>&) override
  {}

  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet&) const override
  {
    return std::make_unique<UnsupportedStreamingComponent>();
  }
};

} // namespace

TEST_CASE("PsiFormer streams bounded score VJPs and JVPs",
          "[wavefunction][psiformer][training][streaming]")
{
  using namespace testing::psiformer;
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons0 = makeElectrons(simulation_cell);
  ParticleSet electrons1 = makeElectrons(simulation_cell);
  electrons1.R[0] += ParticleSet::PosType{0.013, -0.009, 0.006};
  electrons1.update();

  const std::vector<std::size_t> selected{0, 1, 127, 2047};
  PsiFormerWF component("pf_stream", files.parameters.string(),
                        files.configuration.string(), true, selected);
  PsiFormerWF clone(component);
  OptVariables active = registerSelectedParameters(component);
  REQUIRE(active.size() == selected.size());

  RefVectorWithLeader<WaveFunctionComponent> components(component);
  components.push_back(component);
  components.push_back(clone);
  RefVectorWithLeader<ParticleSet> particles(electrons0);
  particles.push_back(electrons0);
  particles.push_back(electrons1);

  // Reject malformed batch topology and physical metadata before allocating the
  // large score owner.
  RefVectorWithLeader<WaveFunctionComponent> mismatched_components(component);
  mismatched_components.push_back(component);
  RefVectorWithLeader<ParticleSet> no_particle_lanes(electrons0);
  CHECK_THROWS(component.makeStreamingDerivativeOperator(
      mismatched_components, no_particle_lanes, 31, 17, 257));
  CHECK_THROWS(component.makeStreamingDerivativeOperator(
      components, particles, 31, 17, 0));

  RefVectorWithLeader<WaveFunctionComponent> nonleader_components(component);
  nonleader_components.push_back(clone);
  RefVectorWithLeader<ParticleSet> nonleader_particles(electrons0);
  nonleader_particles.push_back(electrons1);
  CHECK_THROWS(component.makeStreamingDerivativeOperator(
      nonleader_components, nonleader_particles, 31, 17, 257));

  PsiFormerWF foreign("pf_stream_foreign", files.parameters.string(),
                      files.configuration.string());
  RefVectorWithLeader<WaveFunctionComponent> foreign_components(component);
  foreign_components.push_back(component);
  foreign_components.push_back(foreign);
  CHECK_THROWS(component.makeStreamingDerivativeOperator(
      foreign_components, particles, 31, 17, 257));

  ParticleSet wrong_spin = makeElectrons(simulation_cell);
  wrong_spin.GroupID[0]   = 1;
  RefVectorWithLeader<WaveFunctionComponent> one_valid_component(component);
  one_valid_component.push_back(component);
  RefVectorWithLeader<ParticleSet> wrong_spin_particles(wrong_spin);
  wrong_spin_particles.push_back(wrong_spin);
  CHECK_THROWS(component.makeStreamingDerivativeOperator(
      one_valid_component, wrong_spin_particles, 31, 17, 257));

  ParticleSet nonfinite = makeElectrons(simulation_cell);
  nonfinite.R[0][0]     = std::numeric_limits<double>::infinity();
  RefVectorWithLeader<ParticleSet> nonfinite_particles(nonfinite);
  nonfinite_particles.push_back(nonfinite);
  CHECK_THROWS(component.makeStreamingDerivativeOperator(
      one_valid_component, nonfinite_particles, 31, 17, 257));

  // A one-sample owner has exactly the same P-dependent storage as the full batch.
  RefVectorWithLeader<WaveFunctionComponent> one_component(component);
  one_component.push_back(component);
  RefVectorWithLeader<ParticleSet> one_particle(electrons0);
  one_particle.push_back(electrons0);
  std::unique_ptr<StreamingDerivativeOperator> one_sample =
      component.makeStreamingDerivativeOperator(one_component, one_particle,
                                                31, 17, 257);
  const auto one_sample_storage = one_sample->storageDiagnostics();
  one_sample.reset();

  std::unique_ptr<StreamingDerivativeOperator> op =
      component.makeStreamingDerivativeOperator(components, particles, 31, 17,
                                                257);
  const auto capability = op->capabilities();
  const auto storage    = op->storageDiagnostics();
  CHECK(capability.supports(DerivativeProduct::SCORE_VJP));
  CHECK(capability.supports(DerivativeProduct::SCORE_JVP));
  CHECK_FALSE(capability.supports(DerivativeProduct::LOCAL_ENERGY_VJP));
  CHECK(capability.maximum_vjp_channels == 3);
  CHECK(capability.maximum_sample_tile_size == 1);
  CHECK(capability.fixed_parameter_scratch_vectors == 4);
  CHECK(storage.parameter_count == component.parameterSchema().parameterCount());
  CHECK(storage.sample_count == 2);
  CHECK(storage.real_parameter_vectors == 1);
  CHECK(storage.complex_parameter_vectors == 3);
  CHECK(storage.parameter_scratch_bytes ==
        storage.parameter_count *
            (sizeof(double) + 3 * sizeof(DerivativeValue)));
  CHECK(storage.retained_numeric_bytes ==
        storage.evaluator_workspace_bytes +
            3 * storage.parameter_count * sizeof(DerivativeValue) +
            storage.sample_position_bytes + storage.sample_product_bytes);
  CHECK(storage.parameter_scratch_bytes ==
        one_sample_storage.parameter_scratch_bytes);
  CHECK(storage.evaluator_workspace_bytes ==
        one_sample_storage.evaluator_workspace_bytes);
  CHECK(storage.sample_position_bytes == 2 * one_sample_storage.sample_position_bytes);
  CHECK(storage.sample_product_bytes == 2 * one_sample_storage.sample_product_bytes);
  CHECK(storage.allocation_generation == 1);
  CHECK(storage.storage_fingerprint != 0);

  // Existing selected-score evaluations provide a separate high-level oracle.
  std::vector<std::vector<DerivativeValue>> expected_scores(
      2, std::vector<DerivativeValue>(selected.size()));
  for (std::size_t sample = 0; sample < 2; ++sample)
  {
    Vector<QMCTraits::ValueType> legacy(active.size());
    legacy = QMCTraits::ValueType{};
    auto& sample_component =
        dynamic_cast<PsiFormerWF&>(components[sample]);
    sample_component.evaluateDerivativesWF(particles[sample], active, legacy);
    for (std::size_t parameter = 0; parameter < selected.size(); ++parameter)
      expected_scores[sample][parameter] = {
          static_cast<double>(std::real(legacy[parameter])),
          static_cast<double>(std::imag(legacy[parameter]))};
  }

  const std::vector<DerivativeValue> coefficients0{{0.7, -0.2},
                                                    {-0.3, 0.5}};
  const std::vector<DerivativeValue> coefficients1{{-0.4, 0.1},
                                                    {0.2, 0.6}};
  const std::vector<VJPCoefficientChannel> channels{
      {"energy", DerivativeProduct::SCORE_VJP,
       coefficientsFor(*op, coefficients0), 0},
      {"metric", DerivativeProduct::SCORE_VJP,
       coefficientsFor(*op, coefficients1), 0}};

  SelectedParameterSink transpose_sink(selected);
  op->applyVJPs({channels.data(), channels.size()},
                wftrain::DerivativeAdjoint::TRANSPOSE, transpose_sink);
  REQUIRE(transpose_sink.results().size() == channels.size());
  CHECK(transpose_sink.consumedRecords() ==
        channels.size() * op->parameterChunkPlan().chunks().size());
  for (std::size_t channel = 0; channel < channels.size(); ++channel)
    for (std::size_t parameter = 0; parameter < selected.size(); ++parameter)
    {
      DerivativeValue expected{};
      for (std::size_t sample = 0; sample < 2; ++sample)
        expected += channels[channel].coefficients.values[sample] *
            expected_scores[sample][parameter];
      checkClose(transpose_sink.results()[channel][parameter], expected);
    }

  // The imported Jacobian is real, so transpose and Hermitian products coincide.
  SelectedParameterSink hermitian_sink(selected);
  op->applyVJPs({channels.data(), channels.size()},
                wftrain::DerivativeAdjoint::HERMITIAN, hermitian_sink);
  for (std::size_t channel = 0; channel < channels.size(); ++channel)
    for (std::size_t parameter = 0; parameter < selected.size(); ++parameter)
      checkClose(hermitian_sink.results()[channel][parameter],
                 transpose_sink.results()[channel][parameter]);

  std::vector<DerivativeValue> direction(storage.parameter_count);
  const std::vector<DerivativeValue> selected_direction{
      {0.05, 0.0}, {-0.02, 0.0}, {0.01, 0.0}, {-0.03, 0.0}};
  for (std::size_t parameter = 0; parameter < selected.size(); ++parameter)
    direction[selected[parameter]] = selected_direction[parameter];
  const wftrain::StructuredParameterVectorConstView direction_view(
      op->parameterSchema(), op->parameterVersion(),
      {direction.data(), direction.size()});
  RecordingSampleSink jvp_sink;
  op->applyScoreJVP(direction_view, jvp_sink);
  REQUIRE(jvp_sink.results().size() == 2);
  for (std::size_t sample = 0; sample < 2; ++sample)
  {
    DerivativeValue expected{};
    for (std::size_t parameter = 0; parameter < selected.size(); ++parameter)
      expected += expected_scores[sample][parameter] *
          selected_direction[parameter];
    checkClose(jvp_sink.results()[sample], expected);
  }

  // Verify the matrix-free duality identity using the first complex VJP channel.
  DerivativeValue direction_dot_vjp{};
  for (std::size_t parameter = 0; parameter < selected.size(); ++parameter)
    direction_dot_vjp += selected_direction[parameter] *
        transpose_sink.results()[0][parameter];
  DerivativeValue coefficients_dot_jvp{};
  for (std::size_t sample = 0; sample < 2; ++sample)
    coefficients_dot_jvp += coefficients0[sample] * jvp_sink.results()[sample];
  checkClose(direction_dot_vjp, coefficients_dot_jvp);

  // Schema mismatch is rejected before beginning a sink transaction.
  wftrain::StructuredParameterSchema foreign_schema(
      "foreign/psiformer", op->parameterSchema().blocks());
  const wftrain::StructuredParameterVectorConstView foreign_direction(
      foreign_schema, op->parameterVersion(), {direction.data(), direction.size()});
  RecordingSampleSink foreign_sink;
  CHECK_THROWS(op->applyScoreJVP(foreign_direction, foreign_sink));
  CHECK(foreign_sink.state() == DerivativeSinkState::IDLE);

  // A derived-sink exception poisons the transaction; reset makes it reusable.
  SelectedParameterSink reusable_sink(selected);
  reusable_sink.armConsumeFailure();
  CHECK_THROWS(op->applyVJPs({channels.data(), channels.size()},
                             wftrain::DerivativeAdjoint::TRANSPOSE,
                             reusable_sink));
  CHECK(reusable_sink.state() == DerivativeSinkState::POISONED);
  reusable_sink.reset();
  CHECK(reusable_sink.state() == DerivativeSinkState::IDLE);
  CHECK_NOTHROW(op->applyVJPs({channels.data(), channels.size()},
                              wftrain::DerivativeAdjoint::TRANSPOSE,
                              reusable_sink));
  CHECK(reusable_sink.state() == DerivativeSinkState::COMPLETE);

  const auto warmed_storage = op->storageDiagnostics();
  CHECK(warmed_storage.retained_numeric_bytes == storage.retained_numeric_bytes);
  CHECK(warmed_storage.allocation_generation == storage.allocation_generation);
  CHECK(warmed_storage.storage_fingerprint == storage.storage_fingerprint);

  // Empty batches complete both sink types without evaluating or emitting chunks.
  RefVectorWithLeader<WaveFunctionComponent> no_components(component);
  RefVectorWithLeader<ParticleSet> no_particles(electrons0);
  std::unique_ptr<StreamingDerivativeOperator> empty_op =
      component.makeStreamingDerivativeOperator(no_components, no_particles,
                                                32, 19, 257);
  const std::vector<DerivativeValue> no_coefficients;
  const std::vector<VJPCoefficientChannel> empty_channels{
      {"empty", DerivativeProduct::SCORE_VJP,
       coefficientsFor(*empty_op, no_coefficients), 0}};
  SelectedParameterSink empty_parameter_sink(selected);
  empty_op->applyVJPs({empty_channels.data(), empty_channels.size()},
                      wftrain::DerivativeAdjoint::TRANSPOSE,
                      empty_parameter_sink);
  CHECK(empty_parameter_sink.state() == DerivativeSinkState::COMPLETE);
  CHECK(empty_parameter_sink.consumedRecords() == 0);
  std::vector<DerivativeValue> empty_direction(storage.parameter_count);
  const wftrain::StructuredParameterVectorConstView empty_direction_view(
      empty_op->parameterSchema(), empty_op->parameterVersion(),
      {empty_direction.data(), empty_direction.size()});
  RecordingSampleSink empty_sample_sink;
  empty_op->applyScoreJVP(empty_direction_view, empty_sample_sink);
  CHECK(empty_sample_sink.state() == DerivativeSinkState::COMPLETE);
  CHECK(empty_sample_sink.results().empty());
  empty_op.reset();

  // Publishing a new parameter version after preparation poisons an active sink
  // instead of evaluating against a mixed model state.
  wftrain::StructuredParameterSnapshot candidate = component.snapshotParameters();
  candidate.values[0] += 1.0e-6;
  component.publishParameters(candidate, candidate.version);
  SelectedParameterSink stale_sink(selected);
  CHECK_THROWS(op->applyVJPs({channels.data(), channels.size()},
                             wftrain::DerivativeAdjoint::TRANSPOSE,
                             stale_sink));
  CHECK(stale_sink.state() == DerivativeSinkState::POISONED);
}

TEST_CASE("Wavefunction components reject unsupported streaming derivative factories",
          "[wavefunction][training][streaming]")
{
  const SimulationCell simulation_cell;
  ParticleSet particles(simulation_cell);
  UnsupportedStreamingComponent component;
  RefVectorWithLeader<WaveFunctionComponent> components(component);
  components.push_back(component);
  RefVectorWithLeader<ParticleSet> particle_list(particles);
  particle_list.push_back(particles);
  CHECK_THROWS(component.makeStreamingDerivativeOperator(
      components, particle_list, 0, 0, 8));

  BatchExecutionRequirements requirements;
  requirements.require(BatchExecutionMode::STREAMING_DERIVATIVE);
  CHECK(requirements.requires(BatchExecutionMode::STREAMING_DERIVATIVE));
  CHECK(static_cast<std::uint32_t>(BatchExecutionMode::STREAMING_DERIVATIVE) ==
        (1U << 14));
}

} // namespace qmcplusplus
