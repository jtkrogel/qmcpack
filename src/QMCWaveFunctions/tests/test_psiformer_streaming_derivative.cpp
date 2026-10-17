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
#include <catch2/matchers/catch_matchers_string.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Particle/ParticleSet.h"
#include "Particle/VirtualParticleBatch.h"
#include "Particle/VirtualParticleSet.h"
#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeterminantChannelBatch.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerStorageRequirements.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceRuntime.h"
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceMetrics.h"
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceAdam.h"
#include "QMCDrivers/WFTrain/ResidentEnergySRDiagnostics.h"
#endif
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "ResourceCollection.h"
#include "psiformer_test_utils.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
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

/// Restore one process environment setting after a backend-selection check.
class ScopedEnvironmentVariable
{
public:
  ScopedEnvironmentVariable(std::string name, const char* value)
      : name_(std::move(name))
  {
    if (const char* previous = std::getenv(name_.c_str()))
      previous_ = previous;
    if ((value ? setenv(name_.c_str(), value, 1) : unsetenv(name_.c_str())) != 0)
      throw std::runtime_error("failed to set PsiFormer test environment");
  }

  ~ScopedEnvironmentVariable()
  {
    if (previous_)
      setenv(name_.c_str(), previous_->c_str(), 1);
    else
      unsetenv(name_.c_str());
  }

private:
  std::string name_;
  std::optional<std::string> previous_;
};

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

  /// Run one caller-controlled action while a numerical output remains borrowed.
  void onConsumeOnce(std::function<void()> callback) { callback_ = std::move(callback); }

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
    if (callback_)
    {
      auto callback = std::move(callback_);
      callback();
    }
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
  std::function<void()> callback_;
  std::vector<std::size_t> selected_;
  std::vector<std::vector<DerivativeValue>> results_;
  std::size_t consumed_records_ = 0;
  bool fail_next_consume_       = false;
};

/// Collect the one-sample score-product tiles in their checked order.
class RecordingSampleSink final : public SampleProductSink
{
public:
  void armConsumeFailure() noexcept { fail_next_consume_ = true; }

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
    if (fail_next_consume_)
    {
      fail_next_consume_ = false;
      throw std::runtime_error("Injected sample sink consume failure");
    }
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
  bool fail_next_consume_ = false;
};

/** Attempt a nested product while the outer sink still consumes leased scratch. */
class ReentrantSampleSink final : public SampleProductSink
{
public:
  ReentrantSampleSink(const StreamingDerivativeOperator& other,
                      const wftrain::StructuredParameterVectorConstView& direction)
      : other_(other), direction_(direction)
  {}

protected:
  void consume(const SampleProductTileConstView&) override
  {
    RecordingSampleSink nested;
    other_.applyScoreJVP(direction_, nested);
  }

private:
  const StreamingDerivativeOperator& other_;
  const wftrain::StructuredParameterVectorConstView& direction_;
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

TEST_CASE("PsiFormer streams bounded score and kinetic products",
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

  // Populate the authoritative complete TrialWaveFunction drift before the
  // factory snapshots it.  The deterministic offset represents a fixed
  // companion factor and exercises the kinetic cross term without adding a
  // second trainable provider.
  electrons0.G = QMCTraits::GradType{};
  electrons0.L = QMCTraits::ValueType{};
  electrons1.G = QMCTraits::GradType{};
  electrons1.L = QMCTraits::ValueType{};
  component.evaluateLog(electrons0, electrons0.G, electrons0.L);
  clone.evaluateLog(electrons1, electrons1.G, electrons1.L);
  for (int electron = 0; electron < electrons0.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      const double fixed_drift =
          0.007 * (1 + electron) * (dimension % 2 == 0 ? 1.0 : -1.0);
      electrons0.G[electron][dimension] += fixed_drift;
      electrons1.G[electron][dimension] -= 0.6 * fixed_drift;
#ifdef QMC_COMPLEX
      // A fixed complex companion factor exercises the separate imaginary
      // kinetic reverse and distinguishes transpose from Hermitian products.
      const double imaginary_drift =
          0.005 * (1 + dimension) * (electron % 2 == 0 ? 1.0 : -1.0);
      electrons0.G[electron][dimension] +=
          QMCTraits::ValueType(0.0, imaginary_drift);
      electrons1.G[electron][dimension] -=
          QMCTraits::ValueType(0.0, 0.4 * imaginary_drift);
#endif
    }

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

  // A component configured with an oracle kinetic backend cannot advertise the
  // direct streaming capability, even when its score backend is direct.
  std::unique_ptr<PsiFormerWF> oracle_kinetic_component;
  {
    ScopedEnvironmentVariable kinetic_backend("PSIFORMER_KINETIC_BACKEND",
                                               "oracle");
    oracle_kinetic_component = std::make_unique<PsiFormerWF>(
        "pf_stream_oracle_kinetic", files.parameters.string(),
        files.configuration.string());
  }
  RefVectorWithLeader<WaveFunctionComponent> oracle_kinetic_components(
      *oracle_kinetic_component);
  oracle_kinetic_components.push_back(*oracle_kinetic_component);
  RefVectorWithLeader<ParticleSet> oracle_kinetic_particles(electrons0);
  oracle_kinetic_particles.push_back(electrons0);
  CHECK_THROWS(oracle_kinetic_component->makeStreamingDerivativeOperator(
      oracle_kinetic_components, oracle_kinetic_particles, 31, 17, 257));

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

  ParticleSet nonfinite_drift = electrons0;
  nonfinite_drift.G[0][0] = std::numeric_limits<double>::infinity();
  RefVectorWithLeader<ParticleSet> nonfinite_drift_particles(nonfinite_drift);
  nonfinite_drift_particles.push_back(nonfinite_drift);
  CHECK_THROWS(component.makeStreamingDerivativeOperator(
      one_valid_component, nonfinite_drift_particles, 31, 17, 257));

  ParticleSet invalid_mass = electrons0;
  SpeciesSet& invalid_mass_species = invalid_mass.getSpeciesSet();
  const int mass_attribute = invalid_mass_species.getAttribute("mass");
  REQUIRE(mass_attribute >= 0);
  invalid_mass_species(mass_attribute, 0) = 0.0;
  invalid_mass.resetGroups();
  RefVectorWithLeader<ParticleSet> invalid_mass_particles(invalid_mass);
  invalid_mass_particles.push_back(invalid_mass);
  CHECK_THROWS(component.makeStreamingDerivativeOperator(
      one_valid_component, invalid_mass_particles, 31, 17, 257));

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
  CHECK(capability.supports(DerivativeProduct::LOCAL_ENERGY_VJP));
  CHECK(capability.supports(DerivativeProduct::LOCAL_ENERGY_JVP));
  CHECK(capability.local_energy_term_mask ==
        wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC));
  CHECK(capability.maximum_vjp_channels == 3);
  CHECK(capability.maximum_sample_tile_size == 1);
  CHECK(capability.fixed_parameter_scratch_vectors == 6);
  CHECK(storage.parameter_count == component.parameterSchema().parameterCount());
  CHECK(storage.sample_count == 2);
  CHECK(storage.real_parameter_vectors == 3);
  CHECK(storage.complex_parameter_vectors == 3);
  CHECK(storage.parameter_scratch_bytes ==
        storage.parameter_count *
            (3 * sizeof(double) + 3 * sizeof(DerivativeValue)));
  CHECK(storage.retained_numeric_bytes ==
        storage.evaluator_workspace_bytes +
            3 * storage.parameter_count * sizeof(DerivativeValue) +
            storage.sample_position_bytes + storage.sample_auxiliary_bytes +
            storage.sample_product_bytes);
  CHECK(storage.parameter_scratch_bytes ==
        one_sample_storage.parameter_scratch_bytes);
  CHECK(storage.evaluator_workspace_bytes ==
        one_sample_storage.evaluator_workspace_bytes);
  CHECK(storage.sample_position_bytes == 2 * one_sample_storage.sample_position_bytes);
  CHECK(storage.sample_auxiliary_bytes ==
        2 * one_sample_storage.sample_auxiliary_bytes);
  CHECK(storage.sample_product_bytes == 2 * one_sample_storage.sample_product_bytes);
  CHECK(storage.allocation_generation == 1);
  CHECK(storage.storage_fingerprint != 0);

  // Concurrently retained sample owners share scratch, not their positions or
  // sample products. Sharing remains valid when either owner is released first.
  auto shared_sample = component.makeStreamingDerivativeOperator(
      one_component, one_particle, 31, 17, 257);
  CHECK(storage.shared_workspace_identity != 0);
  CHECK(shared_sample->storageDiagnostics().shared_workspace_identity == storage.shared_workspace_identity);
  CHECK(storage.shared_workspace_bytes > storage.parameter_scratch_bytes);

  // Existing selected-score evaluations provide a separate high-level oracle.
  std::vector<std::vector<DerivativeValue>> expected_scores(
      2, std::vector<DerivativeValue>(selected.size()));
  std::vector<std::vector<DerivativeValue>> expected_kinetic(
      2, std::vector<DerivativeValue>(selected.size()));
  for (std::size_t sample = 0; sample < 2; ++sample)
  {
    Vector<QMCTraits::ValueType> legacy_score(active.size());
    Vector<QMCTraits::ValueType> legacy_kinetic(active.size());
    legacy_score   = QMCTraits::ValueType{};
    legacy_kinetic = QMCTraits::ValueType{};
    auto& sample_component =
        dynamic_cast<PsiFormerWF&>(components[sample]);
    sample_component.evaluateDerivatives(particles[sample], active, legacy_score,
                                         legacy_kinetic);
    for (std::size_t parameter = 0; parameter < selected.size(); ++parameter)
    {
      expected_scores[sample][parameter] = {
          static_cast<double>(std::real(legacy_score[parameter])),
          static_cast<double>(std::imag(legacy_score[parameter]))};
      expected_kinetic[sample][parameter] = {
          static_cast<double>(std::real(legacy_kinetic[parameter])),
          static_cast<double>(std::imag(legacy_kinetic[parameter]))};
    }
  }

  const std::vector<DerivativeValue> coefficients0{{0.7, -0.2},
                                                    {-0.3, 0.5}};
  const std::vector<DerivativeValue> coefficients1{{-0.4, 0.1},
                                                    {0.2, 0.6}};
  const std::vector<DerivativeValue> coefficients2{{0.31, -0.27},
                                                    {-0.22, 0.14}};
  const std::vector<VJPCoefficientChannel> channels{
      {"energy", DerivativeProduct::SCORE_VJP,
       coefficientsFor(*op, coefficients0), 0},
      {"metric", DerivativeProduct::SCORE_VJP,
       coefficientsFor(*op, coefficients1), 0},
      {"kinetic", DerivativeProduct::LOCAL_ENERGY_VJP,
       coefficientsFor(*op, coefficients2),
       wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC)}};

  // Exercise the dedicated score-only runtime branch independently of the
  // fused kinetic traversal.
  const std::vector<VJPCoefficientChannel> score_only_channels{
      {"score_only", DerivativeProduct::SCORE_VJP,
       coefficientsFor(*op, coefficients0), 0}};
  SelectedParameterSink score_only_sink(selected);
  op->applyVJPs({score_only_channels.data(), score_only_channels.size()},
                wftrain::DerivativeAdjoint::TRANSPOSE, score_only_sink);
  for (std::size_t parameter = 0; parameter < selected.size(); ++parameter)
  {
    DerivativeValue expected{};
    for (std::size_t sample = 0; sample < 2; ++sample)
      expected += coefficients0[sample] * expected_scores[sample][parameter];
    checkClose(score_only_sink.results()[0][parameter], expected);
  }

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
      {
        const DerivativeValue oracle = channels[channel].product ==
                DerivativeProduct::LOCAL_ENERGY_VJP
            ? expected_kinetic[sample][parameter]
            : expected_scores[sample][parameter];
        expected += channels[channel].coefficients.values[sample] * oracle;
      }
      checkClose(transpose_sink.results()[channel][parameter], expected);
    }

  // Nonlocal coverage is intentionally deferred.  Common preflight rejects it
  // before beginning or poisoning caller-owned sink state.
  const std::vector<VJPCoefficientChannel> unsupported_channels{
      {"unsupported_ecp", DerivativeProduct::LOCAL_ENERGY_VJP,
       coefficientsFor(*op, coefficients2),
       wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP)}};
  SelectedParameterSink unsupported_sink(selected);
  CHECK_THROWS(op->applyVJPs(
      {unsupported_channels.data(), unsupported_channels.size()},
      wftrain::DerivativeAdjoint::TRANSPOSE, unsupported_sink));
  CHECK(unsupported_sink.state() == DerivativeSinkState::IDLE);

  // Hermitian mode conjugates each derivative response before applying the
  // caller coefficient.  This differs from transpose for the complex kinetic
  // response while the parameter score itself remains real.
  SelectedParameterSink hermitian_sink(selected);
  op->applyVJPs({channels.data(), channels.size()},
                wftrain::DerivativeAdjoint::HERMITIAN, hermitian_sink);
  for (std::size_t channel = 0; channel < channels.size(); ++channel)
    for (std::size_t parameter = 0; parameter < selected.size(); ++parameter)
    {
      DerivativeValue expected{};
      for (std::size_t sample = 0; sample < 2; ++sample)
      {
        const DerivativeValue oracle = channels[channel].product ==
                DerivativeProduct::LOCAL_ENERGY_VJP
            ? expected_kinetic[sample][parameter]
            : expected_scores[sample][parameter];
        expected += channels[channel].coefficients.values[sample] *
            std::conj(oracle);
      }
      checkClose(hermitian_sink.results()[channel][parameter], expected);
    }

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

  RecordingSampleSink kinetic_jvp_sink;
  op->applyLocalEnergyJVP(
      direction_view,
      wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC),
      kinetic_jvp_sink);
  REQUIRE(kinetic_jvp_sink.results().size() == 2);
  for (std::size_t sample = 0; sample < 2; ++sample)
  {
    DerivativeValue expected{};
    for (std::size_t parameter = 0; parameter < selected.size(); ++parameter)
      expected += expected_kinetic[sample][parameter] *
          selected_direction[parameter];
    checkClose(kinetic_jvp_sink.results()[sample], expected);
  }

  RecordingSampleSink unsupported_jvp_sink;
  CHECK_THROWS(op->applyLocalEnergyJVP(
      direction_view,
      wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP),
      unsupported_jvp_sink));
  CHECK(unsupported_jvp_sink.state() == DerivativeSinkState::IDLE);

  // Verify the matrix-free duality identity using the first complex VJP channel.
  RecordingSampleSink shared_jvp_sink;
  shared_sample->applyScoreJVP(direction_view, shared_jvp_sink);
  REQUIRE(shared_jvp_sink.results().size() == 1);
  checkClose(shared_jvp_sink.results()[0], jvp_sink.results()[0]);
  RecordingSampleSink repeated_jvp_sink;
  ReentrantSampleSink reentrant_sink(*shared_sample, direction_view);
  CHECK_THROWS(op->applyScoreJVP(direction_view, reentrant_sink));
  // The failed nested lease must not poison the next ordinary operation.
  op->applyScoreJVP(direction_view, repeated_jvp_sink);
  for (std::size_t sample = 0; sample < 2; ++sample)
    checkClose(repeated_jvp_sink.results()[sample], jvp_sink.results()[sample]);
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
  empty_sample_sink.reset();
  empty_op->applyLocalEnergyJVP(
      empty_direction_view,
      wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC),
      empty_sample_sink);
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

TEST_CASE("PsiFormer candidate clones isolate flat parameters and native graph leaves",
          "[wavefunction][psiformer][training][selection]")
{
  using namespace testing::psiformer;
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell cell;
  for (const char* backend : {"direct", "oracle"})
  {
    ScopedEnvironmentVariable value_backend("PSIFORMER_VALUE_BACKEND", backend);
    ScopedEnvironmentVariable spatial_backend("PSIFORMER_SPATIAL_BACKEND", backend);
    ParticleSet electrons = makeElectrons(cell);
    PsiFormerWF component("candidate_isolation", files.parameters.string(),
                          files.configuration.string());
    const auto before = component.snapshotParameters();
    electrons.G = QMCTraits::GradType{};
    electrons.L = QMCTraits::ValueType{};
    const auto reference = component.evaluateLog(electrons, electrons.G, electrons.L);
    auto speculative = component.makeClone(electrons);
    auto& provider = *speculative->structuredParameterProvider();
    provider.isolateParameterState();
    auto candidate = provider.snapshotParameters();
    REQUIRE(candidate.schema_fingerprint == before.schema_fingerprint);
    candidate.values[0] += 0.05;
    provider.publishParameters(candidate, candidate.version);
    electrons.G = QMCTraits::GradType{};
    electrons.L = QMCTraits::ValueType{};
    const auto proposed = speculative->evaluateLog(electrons, electrons.G, electrons.L);
    CHECK(std::real(proposed) != Approx(std::real(reference)).margin(1.0e-12));
    CHECK(component.snapshotParameters().values == before.values);
    CHECK(component.snapshotParameters().version == before.version);
    electrons.G = QMCTraits::GradType{};
    electrons.L = QMCTraits::ValueType{};
    const auto unchanged = component.evaluateLog(electrons, electrons.G, electrons.L);
    CHECK(std::real(unchanged) == Approx(std::real(reference)).margin(1.0e-12));
    // Typed weight selection excludes cusp and envelope tensors even though
    // their schema entries remain trainable for ordinary optimization.
    const auto decay_blocks = component.neuralWeightBlocks();
    REQUIRE_FALSE(decay_blocks.empty());
    for (const auto& id : decay_blocks)
    {
      CHECK(id.find("cusp") == std::string::npos);
      CHECK(id.find("envelope") == std::string::npos);
    }
  }
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
  CHECK_THROWS(component.makeNonLocalECPDerivativeConsumer(
      components, particle_list, 8));
  CHECK_THROWS(component.makeDeterminantChannelBatch(components, particle_list));

  BatchExecutionRequirements requirements;
  requirements.require(BatchExecutionMode::STREAMING_DERIVATIVE);
  CHECK(requirements.requires(BatchExecutionMode::STREAMING_DERIVATIVE));
  CHECK(static_cast<std::uint32_t>(BatchExecutionMode::STREAMING_DERIVATIVE) ==
        (1U << 14));
}

TEST_CASE("TrialWaveFunction constructs a provider-selected streaming owner",
          "[wavefunction][psiformer][training][streaming][aggregate]")
{
  using namespace testing::psiformer;
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons0 = makeElectrons(simulation_cell);
  ParticleSet electrons1 = makeElectrons(simulation_cell);
  electrons1.R[1] += ParticleSet::PosType{-0.012, 0.008, 0.004};
  electrons1.update();

  RuntimeOptions runtime_options;
  TrialWaveFunction wavefunction0(runtime_options, "aggregate_stream0");
  wavefunction0.addComponent(std::make_unique<PsiFormerWF>(
      "pf_aggregate_stream", files.parameters.string(),
      files.configuration.string()));
  std::unique_ptr<TrialWaveFunction> wavefunction1 =
      wavefunction0.makeClone(electrons1);

  wavefunction0.evaluateLog(electrons0);
  wavefunction1->evaluateLog(electrons1);

  RefVectorWithLeader<TrialWaveFunction> wavefunctions(wavefunction0);
  wavefunctions.push_back(wavefunction0);
  wavefunctions.push_back(*wavefunction1);
  RefVectorWithLeader<ParticleSet> particles(electrons0);
  particles.push_back(electrons0);
  particles.push_back(electrons1);

  auto provider = wavefunction0.extractStructuredParameterProviders();
  REQUIRE(provider.size() == 1);
  const std::string provider_id =
      provider.front().get().parameterSchema().providerId();

  std::unique_ptr<StreamingDerivativeOperator> selected =
      TrialWaveFunction::makeStructuredStreamingDerivativeOperator(
          wavefunctions, particles, provider_id, 9, 23, 113);
  CHECK(selected->parameterSchema().providerId() == provider_id);
  CHECK(selected->batchOrdinal() == 9);
  CHECK(selected->sampleOffset() == 23);
  CHECK(selected->sampleCount() == 2);

  std::unique_ptr<StreamingDerivativeOperator> sole =
      TrialWaveFunction::makeStructuredStreamingDerivativeOperator(
          wavefunctions, particles, {}, 10, 25, 113);
  CHECK(sole->parameterSchema().providerId() == provider_id);

  std::unique_ptr<wftrain::NonLocalECPDerivativeConsumer> ecp_consumer =
      TrialWaveFunction::makeStructuredNonLocalECPDerivativeConsumer(
          wavefunctions, particles, provider_id, 113);
  const auto ecp_storage = ecp_consumer->storageDiagnostics();
  CHECK(ecp_storage.sample_count == 2);
  CHECK(ecp_storage.parameter_count ==
        provider.front().get().parameterSchema().parameterCount());

  std::unique_ptr<wftrain::NonLocalECPDerivativeConsumer> sole_ecp_consumer =
      TrialWaveFunction::makeStructuredNonLocalECPDerivativeConsumer(
          wavefunctions, particles, {}, 113, true);
  CHECK(sole_ecp_consumer->storageDiagnostics().sample_count == 2);

  std::unique_ptr<psiformer::DeterminantChannelBatch> channel_batch =
      TrialWaveFunction::makeStructuredDeterminantChannelBatch(
          wavefunctions, particles, provider_id);
  CHECK(channel_batch->providerId() == provider_id);
  CHECK(channel_batch->sampleCount() == 2);
  CHECK(channel_batch->channelCount() > 1);
  CHECK(channel_batch->parameterCount() ==
        provider.front().get().parameterSchema().parameterCount());
  CHECK(channel_batch->retainedNumericBytes() >
        channel_batch->parameterCount() * sizeof(double));

  std::vector<psiformer::RealSignedLogAmplitude> amplitudes(
      channel_batch->channelCount());
  channel_batch->evaluateAmplitudes(0, amplitudes.data(), amplitudes.size());
  const auto copied_amplitudes = amplitudes;
  electrons0.R[0] += ParticleSet::PosType{0.5, -0.25, 0.125};
  electrons0.update();
  channel_batch->evaluateAmplitudes(0, amplitudes.data(), amplitudes.size());
  for (std::size_t channel = 0; channel < amplitudes.size(); ++channel)
  {
    CHECK(amplitudes[channel].phase == copied_amplitudes[channel].phase);
    CHECK(amplitudes[channel].log_abs == copied_amplitudes[channel].log_abs);
  }

  std::vector<double> channel_weights(channel_batch->channelCount(), 0.0);
  channel_weights.front() = 1.0;
  std::vector<double> parameter_gradient(channel_batch->parameterCount());
  channel_batch->evaluateWeightedScore(
      1, channel_weights.data(), channel_weights.size(),
      parameter_gradient.data(), parameter_gradient.size());
  CHECK(std::all_of(parameter_gradient.begin(), parameter_gradient.end(),
                    [](double value) { return std::isfinite(value); }));

  CHECK_THROWS(TrialWaveFunction::makeStructuredDeterminantChannelBatch(
      wavefunctions, particles, "missing_provider"));

  // A rank with no local walkers still constructs the same schema/version owner
  // and can follow the distributed collective schedule with an empty contribution.
  RefVectorWithLeader<TrialWaveFunction> empty_wavefunctions(wavefunction0);
  RefVectorWithLeader<ParticleSet> empty_particles(electrons0);
  std::unique_ptr<StreamingDerivativeOperator> empty =
      TrialWaveFunction::makeStructuredStreamingDerivativeOperator(
          empty_wavefunctions, empty_particles, provider_id, 11, 27, 113);
  CHECK(empty->sampleCount() == 0);
  CHECK(empty->parameterSchema().providerId() == provider_id);

  CHECK_THROWS(TrialWaveFunction::makeStructuredStreamingDerivativeOperator(
      wavefunctions, particles, "missing_provider", 9, 23, 113));
  CHECK_THROWS(TrialWaveFunction::makeStructuredNonLocalECPDerivativeConsumer(
      wavefunctions, particles, "missing_provider", 113));

  RefVectorWithLeader<ParticleSet> one_particle(electrons0);
  one_particle.push_back(electrons0);
  CHECK_THROWS(TrialWaveFunction::makeStructuredStreamingDerivativeOperator(
      wavefunctions, one_particle, provider_id, 9, 23, 113));
  CHECK_THROWS(TrialWaveFunction::makeStructuredNonLocalECPDerivativeConsumer(
      wavefunctions, one_particle, provider_id, 113));

  TrialWaveFunction multiple_providers(runtime_options,
                                       "aggregate_multiple_ecp_providers");
  multiple_providers.addComponent(std::make_unique<PsiFormerWF>(
      "pf_aggregate_ecp_first", files.parameters.string(),
      files.configuration.string()));
  multiple_providers.addComponent(std::make_unique<PsiFormerWF>(
      "pf_aggregate_ecp_second", files.parameters.string(),
      files.configuration.string()));
  RefVectorWithLeader<TrialWaveFunction> multiple_wavefunctions(
      multiple_providers);
  multiple_wavefunctions.push_back(multiple_providers);
  RefVectorWithLeader<ParticleSet> multiple_particles(electrons0);
  multiple_particles.push_back(electrons0);
  CHECK_THROWS_WITH(
      TrialWaveFunction::makeStructuredNonLocalECPDerivativeConsumer(
          multiple_wavefunctions, multiple_particles, {}, 113),
      Catch::Matchers::ContainsSubstring("requires exactly one provider"));

  wftrain::StructuredParameterSnapshot updated =
      provider.front().get().snapshotParameters();
  updated.values.front() += 1.0e-7;
  provider.front().get().publishParameters(updated, updated.version);
  CHECK_THROWS(channel_batch->evaluateAmplitudes(
      0, amplitudes.data(), amplitudes.size()));
}

TEST_CASE("PsiFormer contracts live ordinary-locality ECP tiles without dense staging",
          "[wavefunction][psiformer][training][streaming][ecp]")
{
  using namespace testing::psiformer;
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons0 = makeElectrons(simulation_cell);
  ParticleSet electrons1 = makeElectrons(simulation_cell);
  electrons1.R[2] += ParticleSet::PosType{0.009, -0.004, 0.006};
  electrons1.update();

  const std::vector<std::size_t> selected{0, 1, 127, 2047};
  PsiFormerWF component("pf_ecp_stream", files.parameters.string(),
                        files.configuration.string(), true, selected);
  PsiFormerWF clone(component);
  OptVariables active = registerSelectedParameters(component);

  RefVectorWithLeader<WaveFunctionComponent> components(component);
  components.push_back(component);
  components.push_back(clone);
  RefVectorWithLeader<ParticleSet> particles(electrons0);
  particles.push_back(electrons0);
  particles.push_back(electrons1);

  auto consumer = component.makeNonLocalECPDerivativeConsumer(
      components, particles, 257);
  const auto storage = consumer->storageDiagnostics();
  CHECK(storage.sample_count == 2);
  CHECK(storage.real_parameter_vectors == 1);
  CHECK(storage.complex_parameter_vectors == 3);
  CHECK(storage.parameter_scratch_bytes ==
        storage.parameter_count *
            (sizeof(double) + 3 * sizeof(DerivativeValue)));
  CHECK(storage.allocation_generation == 1);

  const std::size_t version = component.snapshotParameters().version;
  wftrain::NonLocalECPDerivativeContext context;
  context.provider_id          = component.parameterSchema().providerId();
  context.schema_fingerprint   = component.parameterSchema().fingerprint();
  context.parameter_version    = version;
  context.batch_ordinal        = 71;
  context.sample_offset        = 23;
  context.sample_count         = 2;
  context.expected_point_count = 3;
  context.grid_fingerprint     = UINT64_C(0x9d735a11);
  const std::vector<DerivativeValue> coefficients{{0.7, -0.2},
                                                   {-0.3, 0.4}};
  const CoefficientView coefficient_view{
      context.provider_id, context.schema_fingerprint, context.parameter_version,
      context.batch_ordinal, context.sample_offset,
      {coefficients.data(), coefficients.size()}};
  const std::vector<VJPCoefficientChannel> channels{{
      "nonlocal", DerivativeProduct::LOCAL_ENERGY_VJP, coefficient_view,
      wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP)}};
  SelectedParameterSink sink(selected);

  // Unsupported localization semantics fail before activating the caller sink.
  auto rejected = context;
  rejected.localization = wftrain::NonLocalECPLocalization::DLA;
  CHECK_THROWS(consumer->begin(rejected, {channels.data(), channels.size()}, sink));
  CHECK(sink.state() == DerivativeSinkState::IDLE);
  rejected                 = context;
  rejected.uses_virtual_particles = false;
  CHECK_THROWS(consumer->begin(rejected, {channels.data(), channels.size()}, sink));
  CHECK(sink.state() == DerivativeSinkState::IDLE);
  rejected                         = context;
  rejected.scalar_relativistic     = false;
  CHECK_THROWS(consumer->begin(rejected, {channels.data(), channels.size()}, sink));
  CHECK(sink.state() == DerivativeSinkState::IDLE);

  // Use a fresh one-shot consumer after completing the independent gate checks.
  consumer = component.makeNonLocalECPDerivativeConsumer(components, particles,
                                                         257);
  const auto active_storage = consumer->storageDiagnostics();
  context.localization = wftrain::NonLocalECPLocalization::ORDINARY_LOCALITY;
  REQUIRE(context.localization ==
          wftrain::NonLocalECPLocalization::ORDINARY_LOCALITY);
  consumer->begin(context, {channels.data(), channels.size()}, sink);

  std::vector<std::unique_ptr<VirtualParticleSet>> scratch_storage;
  scratch_storage.push_back(std::make_unique<VirtualParticleSet>(electrons0));
  scratch_storage.push_back(std::make_unique<VirtualParticleSet>(electrons1));
  RefVectorWithLeader<VirtualParticleSet> scratch(*scratch_storage[0]);
  scratch.push_back(*scratch_storage[0]);
  scratch.push_back(*scratch_storage[1]);

  ResourceCollection resource_template("psiformer_ecp_stream_template");
  component.createResource(resource_template);
  ResourceCollection resource(resource_template);
  ResourceCollectionTeamLock<WaveFunctionComponent> resource_lock(resource,
                                                                   components);

  std::vector<std::vector<ParticleSet::PosType>> tile_positions{
      {electrons0.R[0] + ParticleSet::PosType{0.012, -0.006, 0.004},
       electrons0.R[0] + ParticleSet::PosType{-0.008, 0.011, 0.003}},
      {electrons1.R[1] + ParticleSet::PosType{0.005, 0.007, -0.009}}};
  const std::vector<std::vector<QMCTraits::ValueType>> bare_weights{
      {QMCTraits::ValueType(0.21), QMCTraits::ValueType(-0.13)},
      {QMCTraits::ValueType(0.17)}};
  std::vector<std::vector<QMCTraits::ValueType>> complete_ratios(2);
  std::vector<std::vector<WaveFunctionComponent::EvaluationStamp>> stamps(2);

  for (std::size_t tile = 0; tile < 2; ++tile)
  {
    const int walker   = static_cast<int>(tile);
    const int electron = static_cast<int>(tile);
    const std::vector<std::size_t> offsets{0, tile_positions[tile].size()};
    const std::vector<VirtualParticleBatch::Segment> segments{{walker, electron}};
    const VirtualParticleBatch batch(2, offsets, segments,
                                     tile_positions[tile]);
    complete_ratios[tile].resize(batch.size());
    const auto stamp = component.mw_evaluateVirtualRatios(
        components, particles, scratch, batch, complete_ratios[tile]);
    stamps[tile].push_back(stamp);

    // A deterministic fixed companion factor makes the supplied ratio a complete
    // mixed-wavefunction ratio rather than the PsiFormer ratio alone.
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
      const double displacement =
          tile_positions[tile][point][0] - particles[tile].R[electron][0];
      complete_ratios[tile][point] *= std::exp(0.23 * displacement);
    }

    consumer->consume({batch,
                       {bare_weights[tile].data(), bare_weights[tile].size()},
                       {complete_ratios[tile].data(), complete_ratios[tile].size()},
                       {stamps[tile].data(), stamps[tile].size()}, tile,
                       context.grid_fingerprint});
  }
  consumer->end();
  REQUIRE(sink.results().size() == 1);
  CHECK_THROWS_WITH(consumer->releaseJVPReplay(),
                    Catch::Matchers::ContainsSubstring("capture was not requested"));

  // Independently assemble the virtual-minus-reference identity through the
  // established selected-parameter derivative interface.
  for (std::size_t parameter = 0; parameter < selected.size(); ++parameter)
  {
    DerivativeValue expected{};
    for (std::size_t tile = 0; tile < 2; ++tile)
    {
      auto& sample_component =
          dynamic_cast<PsiFormerWF&>(components[tile]);
      Vector<QMCTraits::ValueType> reference_score(active.size());
      Vector<QMCTraits::ValueType> ignored(active.size());
      reference_score = QMCTraits::ValueType{};
      ignored         = QMCTraits::ValueType{};
      sample_component.evaluateDerivatives(
          particles[tile], active, reference_score, ignored);
      for (std::size_t point = 0; point < tile_positions[tile].size(); ++point)
      {
        ParticleSet moved = particles[tile];
        moved.R[tile]     = tile_positions[tile][point];
        moved.update();
        Vector<QMCTraits::ValueType> virtual_score(active.size());
        virtual_score = QMCTraits::ValueType{};
        ignored       = QMCTraits::ValueType{};
        sample_component.evaluateDerivatives(moved, active, virtual_score,
                                             ignored);
        const QMCTraits::ValueType raw_weight =
            bare_weights[tile][point] * complete_ratios[tile][point];
        const DerivativeValue weight{static_cast<double>(std::real(raw_weight)),
                                     static_cast<double>(std::imag(raw_weight))};
        const DerivativeValue delta{
            static_cast<double>(std::real(virtual_score[parameter] -
                                          reference_score[parameter])),
            static_cast<double>(std::imag(virtual_score[parameter] -
                                          reference_score[parameter]))};
        expected += coefficients[tile] * weight * delta;
      }
    }
    checkClose(sink.results()[0][parameter], expected);
  }

  const auto warmed = consumer->storageDiagnostics();
  CHECK(warmed.parameter_scratch_bytes == active_storage.parameter_scratch_bytes);
  CHECK(warmed.retained_numeric_bytes == active_storage.retained_numeric_bytes);
  CHECK(warmed.storage_fingerprint == active_storage.storage_fingerprint);
  CHECK(warmed.allocation_generation == active_storage.allocation_generation);
}

TEST_CASE("PsiFormer CPU target streaming preserves complete ECP products",
          "[wavefunction][psiformer][training][streaming][target][ecp]")
{
  using namespace testing::psiformer;
  ScopedEnvironmentVariable clear_k("PSIFORMER_CPU_KINETIC_TARGETS", nullptr);
  ScopedEnvironmentVariable clear_cap("PSIFORMER_CPU_TARGET_KINETIC_OWNER_BYTES_PER_LANE", nullptr);
  auto files = generateFiles("lih");
  std::size_t p = 0;
  {
    PsiFormerWF probe("pf_target_stream", files.parameters.string(), files.configuration.string());
    p = probe.parameterSchema().parameterCount();
  }
  const pf::PsiFormerStorageShape shape{4, 2, 16, 256, 4, 9, 4, p};
  struct Snapshot
  {
    std::vector<std::vector<DerivativeValue>> transpose, hermitian;
    std::vector<DerivativeValue> score, kinetic, complete;
    wftrain::StreamingDerivativeStorageDiagnostics storage;
  };
  const auto run = [&](std::size_t k, bool replay_only = false) {
    const auto bytes = k ? pf::targetKineticWorkspaceStorageRequirement(shape, k) : 0;
    const auto k_text = std::to_string(k), cap_text = std::to_string(bytes);
    ScopedEnvironmentVariable targets("PSIFORMER_CPU_KINETIC_TARGETS", k ? k_text.c_str() : nullptr);
    ScopedEnvironmentVariable cap("PSIFORMER_CPU_TARGET_KINETIC_OWNER_BYTES_PER_LANE", k ? cap_text.c_str() : nullptr);
    const SimulationCell cell;
    auto e0 = makeElectrons(cell), e1 = makeElectrons(cell);
    e1.R[1] += ParticleSet::PosType{0.009, -0.004, 0.006}; e1.update();
    for (auto* e : {&e0, &e1})
    {
      const int mass = e->getSpeciesSet().getAttribute("mass");
      e->getSpeciesSet()(mass, 0) = 0.7; e->getSpeciesSet()(mass, 1) = 1.3;
      e->resetGroups();
    }
    const std::vector<std::size_t> selected{0, 1, 127, 2047};
    PsiFormerWF component("pf_target_stream", files.parameters.string(), files.configuration.string(), true, selected);
    PsiFormerWF clone(component);
    const auto companion = [](int electron, int dimension) -> QMCTraits::ValueType {
      const double real = 0.007 * (electron + 1) * (dimension % 2 ? -1 : 1);
#ifdef QMC_COMPLEX
      return {real, 0.005 * (dimension + 1) * (electron % 2 ? -0.4 : 1)};
#else
      return real;
#endif
    };
    for (std::size_t s = 0; s < 2; ++s)
    {
      auto& e = s ? e1 : e0;
      auto& c = s ? clone : component;
      e.G = QMCTraits::GradType{}; e.L = QMCTraits::ValueType{};
      c.evaluateLog(e, e.G, e.L);
      for (int electron = 0; electron < e.getTotalNum(); ++electron)
        for (int d = 0; d < 3; ++d) e.G[electron][d] += companion(electron, d);
    }
    RefVectorWithLeader<WaveFunctionComponent> components(component, {component, clone});
    RefVectorWithLeader<ParticleSet> particles(e0, {e0, e1});
    auto op = component.makeStreamingDerivativeOperator(components, particles, 71, 23, 257);
    const std::vector<DerivativeValue> a{{0.7, -0.2}, {-0.3, 0.4}}, c{{-0.1, 0.3}, {0.6, -0.2}};
    const auto kinetic_mask = wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC);
    const auto ecp_mask = wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP);
    // Capture the actual accepted value-grid stamps and complete companion ratios.
    CHECK_THROWS(component.makeNonLocalECPDerivativeConsumer(components, particles, 257, false, true));
    auto consumer = component.makeNonLocalECPDerivativeConsumer(components, particles, 257, true, replay_only);
    if (replay_only)
    {
      CHECK(consumer->storageDiagnostics().parameter_scratch_bytes == 0);
      CHECK(consumer->storageDiagnostics().evaluator_workspace_bytes == 0);
    }
    wftrain::NonLocalECPDerivativeContext context;
    context.provider_id = op->parameterSchema().providerId(); context.schema_fingerprint = op->parameterSchema().fingerprint();
    context.parameter_version = op->parameterVersion(); context.batch_ordinal = 71; context.sample_offset = 23;
    context.sample_count = 2; context.expected_point_count = 2; context.grid_fingerprint = UINT64_C(0x731934d1);
    const std::vector<VJPCoefficientChannel> ecp_channels{{"ecp", DerivativeProduct::LOCAL_ENERGY_VJP,
        coefficientsFor(*op, a), ecp_mask}};
    SelectedParameterSink capture_sink(selected);
    consumer->begin(context, {ecp_channels.data(), ecp_channels.size()}, capture_sink);
    VirtualParticleSet v0(e0), v1(e1);
    RefVectorWithLeader<VirtualParticleSet> virtuals(v0, {v0, v1});
    ResourceCollection resource_template("target_ecp_template"); component.createResource(resource_template);
    ResourceCollection resources(resource_template);
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(resources, components);
    const std::vector<std::size_t> offsets{0, 1, 2};
    const std::vector<VirtualParticleBatch::Segment> segments{{0, 0}, {1, 1}};
    const std::vector<ParticleSet::PosType> points{e0.R[0] + ParticleSet::PosType{0.012, -0.006, 0.004},
                                                e1.R[1] + ParticleSet::PosType{-0.008, 0.011, 0.003}};
    const VirtualParticleBatch batch(2, offsets, segments, points);
    std::vector<QMCTraits::ValueType> ratios(2), bare{QMCTraits::ValueType(0.21), QMCTraits::ValueType(-0.13)};
    const auto stamp = component.mw_evaluateVirtualRatios(components, particles, virtuals, batch, ratios);
    for (std::size_t s = 0; s < 2; ++s)
    {
      QMCTraits::ValueType exponent{};
      for (int d = 0; d < 3; ++d) exponent += companion(s, d) * (points[s][d] - particles[s].R[s][d]);
      ratios[s] *= std::exp(exponent);
    }
    const std::vector<WaveFunctionComponent::EvaluationStamp> stamps{stamp};
    consumer->consume({batch, {bare.data(), bare.size()}, {ratios.data(), ratios.size()},
        {stamps.data(), stamps.size()}, 0, context.grid_fingerprint});
    consumer->end();
    if (replay_only)
      CHECK(capture_sink.state() == DerivativeSinkState::IDLE);
    auto replay = consumer->releaseJVPReplay();
    CHECK(replay->gridFingerprint() == context.grid_fingerprint);
    CHECK(replay->points().size() == 2);
    CHECK(replay->points()[0].complete_ratio == DerivativeValue(ratios[0]));
    CHECK(replay->points()[1].complete_ratio == DerivativeValue(ratios[1]));
    op->bindNonLocalECPJVPReplay(std::move(replay));
    SelectedParameterSink replay_ecp(selected);
    op->applyVJPs({ecp_channels.data(), ecp_channels.size()}, wftrain::DerivativeAdjoint::TRANSPOSE, replay_ecp);
    if (!replay_only)
      for (std::size_t i = 0; i < selected.size(); ++i)
        checkClose(replay_ecp.results()[0][i], capture_sink.results()[0][i]);
    // Partition the same two accepted samples into separate statistical batches.
    // Coefficients remain attached to their samples, including complex/nonuniform
    // weights; replay cannot be replaced by an unweighted sum of P-vectors.
    std::vector<DerivativeValue> partitioned_ecp(selected.size());
    for (std::size_t sample = 0; sample < 2; ++sample)
    {
      WaveFunctionComponent& member = sample ? static_cast<WaveFunctionComponent&>(clone) : component;
      ParticleSet& electron = sample ? e1 : e0;
      RefVectorWithLeader<WaveFunctionComponent> one_component(member, {member});
      RefVectorWithLeader<ParticleSet> one_particle(electron, {electron});
      // The factory must be invoked on this partition's leader.
      auto& leader = sample ? clone : component;
      auto partition = leader.makeStreamingDerivativeOperator(one_component, one_particle, 200 + sample, sample, 257);
      auto capture = leader.makeNonLocalECPDerivativeConsumer(one_component, one_particle, 257, true, true);
      auto one_context = context;
      one_context.batch_ordinal = 200 + sample;
      one_context.sample_offset = sample;
      one_context.sample_count = 1;
      one_context.expected_point_count = 1;
      const std::vector<DerivativeValue> coefficient{a[sample]};
      const VJPCoefficientChannel channel{"ecp", DerivativeProduct::LOCAL_ENERGY_VJP,
          coefficientsFor(*partition, coefficient), ecp_mask};
      SelectedParameterSink idle_sink(selected);
      capture->begin(one_context, {&channel, 1}, idle_sink);
      const std::vector<std::size_t> one_offsets{0, 1};
      const std::vector<VirtualParticleBatch::Segment> one_segments{{0, static_cast<int>(sample)}};
      const std::vector<ParticleSet::PosType> one_points{points[sample]};
      const VirtualParticleBatch one_grid(1, one_offsets, one_segments, one_points);
      capture->consume({one_grid, {bare.data() + sample, 1}, {ratios.data() + sample, 1},
          {stamps.data(), stamps.size()}, 0, context.grid_fingerprint});
      capture->end();
      CHECK(idle_sink.state() == DerivativeSinkState::IDLE);
      CHECK(capture->storageDiagnostics().parameter_scratch_bytes == 0);
      partition->bindNonLocalECPJVPReplay(capture->releaseJVPReplay());
      SelectedParameterSink weighted(selected);
      partition->applyVJPs({&channel, 1}, wftrain::DerivativeAdjoint::TRANSPOSE, weighted);
      for (std::size_t i = 0; i < selected.size(); ++i)
        partitioned_ecp[i] += weighted.results()[0][i];
    }
    for (std::size_t i = 0; i < selected.size(); ++i)
      checkClose(partitioned_ecp[i], replay_ecp.results()[0][i]);
    Snapshot result;
    result.storage = op->storageDiagnostics();
    const auto& storage = result.storage;
    auto shared = component.makeStreamingDerivativeOperator(components, particles, 72, 25, 257);
    CHECK(shared->storageDiagnostics().shared_workspace_identity == storage.shared_workspace_identity);
    // Retaining 256 statistical snapshots adds only sample owners. Every
    // parameter workspace belongs to the same bounded execution scratch.
    std::vector<std::unique_ptr<wftrain::StreamingDerivativeOperator>> snapshots;
    for (std::size_t block = 0; block < 256; ++block)
    {
      snapshots.push_back(component.makeStreamingDerivativeOperator(components, particles, 100 + block, 2 * block, 257));
      const auto retained = snapshots.back()->storageDiagnostics();
      CHECK(retained.shared_workspace_identity == storage.shared_workspace_identity);
      CHECK(retained.parameter_scratch_bytes == storage.parameter_scratch_bytes);
      CHECK(retained.retained_numeric_bytes - retained.shared_workspace_bytes < p * sizeof(DerivativeValue));
    }
    const std::vector<VJPCoefficientChannel> channels{
        {"score", DerivativeProduct::SCORE_VJP, coefficientsFor(*op, a), 0},
        {"kinetic", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, c), kinetic_mask},
        {"complete", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, a), kinetic_mask | ecp_mask}};
    SelectedParameterSink score_only(selected), transpose(selected), hermitian(selected);
    op->applyVJPs({channels.data(), 1}, wftrain::DerivativeAdjoint::TRANSPOSE, score_only);
    op->applyVJPs({channels.data(), channels.size()}, wftrain::DerivativeAdjoint::TRANSPOSE, transpose);
    op->applyVJPs({channels.data(), channels.size()}, wftrain::DerivativeAdjoint::HERMITIAN, hermitian);
    result.transpose = transpose.results(); result.hermitian = hermitian.results();
    for (std::size_t i = 0; i < selected.size(); ++i) checkClose(score_only.results()[0][i], result.transpose[0][i]);
    std::vector<DerivativeValue> direction(p);
    for (std::size_t i = 0; i < selected.size(); ++i) direction[selected[i]] = {0.01 * (i + 1), -0.003 * i};
    const auto reject_complex_direction = [&]() {
      const wftrain::StructuredParameterVectorConstView invalid(op->parameterSchema(), op->parameterVersion(),
          {direction.data(), direction.size()});
      RecordingSampleSink rejected;
      op->applyScoreJVP(invalid, rejected);
    };
    CHECK_THROWS_WITH(reject_complex_direction(), Catch::Matchers::ContainsSubstring("nonzero imaginary part"));
    for (std::size_t i : selected) direction[i] = {direction[i].real(), 0.0};
    const wftrain::StructuredParameterVectorConstView view(op->parameterSchema(), op->parameterVersion(), {direction.data(), direction.size()});
    RecordingSampleSink score_jvp, kinetic_jvp, complete_jvp;
    op->applyScoreJVP(view, score_jvp);
    op->applyLocalEnergyJVP(view, kinetic_mask, kinetic_jvp);
    op->applyLocalEnergyJVP(view, kinetic_mask | ecp_mask, complete_jvp);
    result.score = score_jvp.results(); result.kinetic = kinetic_jvp.results(); result.complete = complete_jvp.results();
    CHECK(std::abs(result.complete[0] - result.kinetic[0]) + std::abs(result.complete[1] - result.kinetic[1]) > 1e-14);
    const auto warmed = op->storageDiagnostics();
    CHECK(warmed.retained_numeric_bytes == storage.retained_numeric_bytes);
    CHECK(warmed.storage_fingerprint == storage.storage_fingerprint);
    CHECK(warmed.allocation_generation == storage.allocation_generation);
    SelectedParameterSink failed(selected); failed.armConsumeFailure();
    CHECK_THROWS(op->applyVJPs({channels.data(), channels.size()}, wftrain::DerivativeAdjoint::TRANSPOSE, failed));
    CHECK(failed.state() == DerivativeSinkState::POISONED); CHECK(failed.consumedRecords() == 0);
    CHECK_THROWS(failed.results());
    return result;
  };
  const auto full = run(0);
  for (std::size_t k : {std::size_t(1), std::size_t(3)})
  {
    const auto target = run(k, true);
    for (std::size_t channel = 0; channel < full.transpose.size(); ++channel)
      for (std::size_t i = 0; i < full.transpose[channel].size(); ++i)
      {
        checkClose(target.transpose[channel][i], full.transpose[channel][i]);
        checkClose(target.hermitian[channel][i], full.hermitian[channel][i]);
      }
    for (std::size_t s = 0; s < 2; ++s)
    {
      checkClose(target.score[s], full.score[s]); checkClose(target.kinetic[s], full.kinetic[s]); checkClose(target.complete[s], full.complete[s]);
    }
    const auto delta = pf::kineticWorkspaceStorageRequirement(shape) - pf::targetKineticWorkspaceStorageRequirement(shape, k);
    CHECK(full.storage.evaluator_workspace_bytes - target.storage.evaluator_workspace_bytes == delta);
    CHECK(full.storage.shared_workspace_bytes - target.storage.shared_workspace_bytes == delta);
    CHECK(full.storage.retained_numeric_bytes - target.storage.retained_numeric_bytes == delta);
    CHECK(full.storage.parameter_scratch_bytes == target.storage.parameter_scratch_bytes);
  }
}

#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
TEST_CASE("PsiFormer real GPU accepted-grid ECP replay shares bounded products",
          "[wavefunction][psiformer][training][streaming][gpu_ecp_replay]")
{
  using namespace testing::psiformer;
  namespace ps = psiformer;
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
  { WARN("NOT_RUN: accepted-grid GPU ECP replay requires a HIP device"); return; }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
  { WARN("NOT_RUN: accepted-grid GPU ECP replay requires a CUDA device"); return; }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  // Small descriptor fixture, not a new public architecture or physical PP model.
  auto leaves = makeLayout(4, 2, true, 1, 4);
  for (auto& leaf : leaves)
    for (auto& extent : leaf.shape)
      if (extent == 256 || extent == 64) extent = 8;
  ps::InitializedPsiFormerParameters initialized;
  initialized.model_shape = {2, 2, 2, 2, 8, 2, 1};
  initialized.profile = ps::DEEPQMC_PSIFORMER_V1;
  initialized.seed = 37;
  initialized.values = makeParameters("lih", leaves);
  std::size_t cursor = 0;
  for (const auto& leaf : leaves)
  {
    const auto begin = cursor; cursor += product(leaf.shape);
    initialized.layouts.push_back({leaf.module, leaf.name, leaf.shape, begin, cursor});
  }
  const SimulationCell cell;
  auto electrons = makeElectrons(cell);
  ParticleSet ions(cell); ions.setName("ion0"); ions.create({1, 1});
  auto& species = ions.getSpeciesSet(); species.addSpecies("Li"); species.addSpecies("H");
  const int charge = species.addAttribute("charge"); species(charge, 0) = 3; species(charge, 1) = 1;
  ions.R[0] = {0, 0, 0}; ions.R[1] = {0, 0, 3.01392421}; ions.resetGroups(); ions.update();
  auto resolution = ps::PsiFormerAcceleratorResolution{};
#ifdef QMC_CUDA2HIP
  resolution.backend = ps::PsiFormerAcceleratorBackend::HIP;
#else
  resolution.backend = ps::PsiFormerAcceleratorBackend::CUDA;
#endif
  resolution.reason = ps::PsiFormerAcceleratorResolutionReason::ACCELERATOR_READY;
  resolution.readiness_records_consulted = 1;
  resolution.validated_operations = ps::psiFormerAcceleratorInferenceOperations();
  PsiFormerWF cpu("ecp_grid", initialized, electrons, ions);
  PsiFormerWF gpu("ecp_grid", initialized, electrons, ions, false, {}, false, {}, {}, resolution);
  const auto p = cpu.parameterSchema().parameterCount();
  std::vector<std::size_t> selected(p); std::iota(selected.begin(), selected.end(), 0);
  const auto ecp = wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP);
  const std::vector<std::size_t> offsets{0, 5, 9};
  const std::vector<VirtualParticleBatch::Segment> segments{{0, 0}, {0, 1}};
  std::vector<ParticleSet::PosType> points;
  std::vector<QMCTraits::ValueType> bare;
  for (int i = 0; i < 9; ++i)
  {
    points.push_back(electrons.R[i < 5 ? 0 : 1] + ParticleSet::PosType{.003 * (i + 1), -.011, .007});
    bare.push_back((i % 2 ? -.13 : .21) / (i + 1));
  }
  const VirtualParticleBatch batch(1, offsets, segments, points);
  const std::vector<std::vector<DerivativeValue>> coefficients{{{.7, -.2}}, {{-.3, .4}}, {{.1, .6}}};
  // Channel IDs are borrowed string_views; their storage must outlive every descriptor.
  const char* const channel_ids[]{"ecp0", "ecp1", "ecp2"};
  const auto channelsFor = [&](const StreamingDerivativeOperator& op) {
    std::vector<VJPCoefficientChannel> channels;
    for (std::size_t i = 0; i < coefficients.size(); ++i)
      channels.push_back({channel_ids[i], DerivativeProduct::LOCAL_ENERGY_VJP,
                          coefficientsFor(op, coefficients[i]), ecp});
    return channels;
  };
  auto& metrics = ps::device::deviceMemoryMetrics(ordinal);
  std::size_t aggregate_downloads = 0;
  const auto capture = [&](PsiFormerWF& component, std::size_t ordinal, std::size_t chunk, bool replay_only,
                           bool empty_grid = false, bool zero_weight_node = false,
                           std::vector<QMCTraits::ValueType>* recorded_ratios = nullptr) {
    RefVectorWithLeader<WaveFunctionComponent> components(component, {component});
    RefVectorWithLeader<ParticleSet> particles(electrons, {electrons});
    auto op = component.makeStreamingDerivativeOperator(components, particles, ordinal, 0, chunk);
    auto consumer = component.makeNonLocalECPDerivativeConsumer(components, particles, chunk, true, replay_only);
    if (replay_only) CHECK(consumer->storageDiagnostics().parameter_scratch_bytes == 0);
    wftrain::NonLocalECPDerivativeContext context;
    context.provider_id = op->parameterSchema().providerId(); context.schema_fingerprint = op->parameterSchema().fingerprint();
    context.parameter_version = op->parameterVersion(); context.batch_ordinal = ordinal;
    context.sample_count = 1; context.expected_point_count = empty_grid ? 0 : points.size();
    context.grid_fingerprint = 0x731934d1;
    auto channels = channelsFor(*op);
    SelectedParameterSink sink(selected);
    consumer->begin(context, {channels.data(), channels.size()}, sink);
    if (empty_grid)
    {
      consumer->end();
      op->bindNonLocalECPJVPReplay(consumer->releaseJVPReplay());
      return op;
    }
    VirtualParticleSet virtual_set(electrons);
    RefVectorWithLeader<VirtualParticleSet> virtuals(virtual_set, {virtual_set});
    ResourceCollection prototype("gpu_ecp_grid"); component.createResource(prototype);
    ResourceCollection resources(prototype);
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(resources, components);
    auto grid_points = points;
    auto grid_bare = bare;
    if (zero_weight_node) { grid_points[0] = electrons.R[1]; grid_bare[0] = 0; }
    const VirtualParticleBatch current_batch(1, offsets, segments, grid_points);
    std::vector<QMCTraits::ValueType> ratios(points.size());
    const auto stamp = component.mw_evaluateVirtualRatios(components, particles, virtuals, current_batch, ratios);
    for (std::size_t i = 0; i < ratios.size(); ++i)
      ratios[i] *= std::exp(.23 * (grid_points[i][0] - electrons.R[i < 5 ? 0 : 1][0]));
    if (recorded_ratios) *recorded_ratios = ratios;
    const std::vector<WaveFunctionComponent::EvaluationStamp> stamps{stamp};
    const auto before = metrics.snapshot();
    consumer->consume({current_batch, {grid_bare.data(), grid_bare.size()}, {ratios.data(), ratios.size()},
        {stamps.data(), stamps.size()}, 0, context.grid_fingerprint});
    consumer->end();
    aggregate_downloads = metrics.snapshot().downloaded_bytes - before.downloaded_bytes;
    if (replay_only) CHECK(sink.state() == DerivativeSinkState::IDLE);
    op->bindNonLocalECPJVPReplay(consumer->releaseJVPReplay());
    return op;
  };
  auto reference = capture(cpu, 1, 3, true);
  auto first = capture(gpu, 1, 3, true);
  const auto before_growth = first->storageDiagnostics();
  std::vector<std::unique_ptr<StreamingDerivativeOperator>> retained;
  for (std::size_t block = 0; block < 9; ++block) retained.push_back(capture(gpu, block + 2, 5, true));
  const auto grown = first->storageDiagnostics();
  CHECK(grown.shared_workspace_bytes > before_growth.shared_workspace_bytes);
  CHECK(grown.storage_fingerprint != before_growth.storage_fingerprint);
  for (const auto& op : retained)
  {
    const auto storage = op->storageDiagnostics();
    CHECK(storage.shared_workspace_identity == grown.shared_workspace_identity);
    CHECK(storage.shared_workspace_bytes == grown.shared_workspace_bytes);
    CHECK(storage.retained_numeric_bytes - storage.shared_workspace_bytes < p * sizeof(DerivativeValue));
  }
  const auto compare = [&](wftrain::DerivativeAdjoint adjoint) {
    SelectedParameterSink expected(selected), actual(selected);
    auto cpu_channels = channelsFor(*reference), gpu_channels = channelsFor(*first);
    reference->applyVJPs({cpu_channels.data(), cpu_channels.size()}, adjoint, expected);
    const auto before = metrics.snapshot();
    first->applyVJPs({gpu_channels.data(), gpu_channels.size()}, adjoint, actual);
    const auto bytes = metrics.snapshot().downloaded_bytes - before.downloaded_bytes;
    for (std::size_t c = 0; c < coefficients.size(); ++c)
      for (std::size_t i = 0; i < p; ++i) checkClose(actual.results()[c][i], expected.results()[c][i]);
    return bytes;
  };
  const auto replay_downloads = compare(wftrain::DerivativeAdjoint::TRANSPOSE);
  compare(wftrain::DerivativeAdjoint::HERMITIAN);
  auto aggregate = capture(gpu, 90, 5, false);
  CHECK(aggregate_downloads > replay_downloads); // Same Q+reference scores, fewer P transfers.
  std::vector<DerivativeValue> direction(p);
  for (std::size_t i = 0; i < p; ++i) direction[i] = {.005 * (int(i % 7) - 3), 0};
  const wftrain::StructuredParameterVectorConstView vector(cpu.parameterSchema(), cpu.snapshotParameters().version,
      {direction.data(), direction.size()});
  RecordingSampleSink cpu_jvp, gpu_jvp;
  reference->applyLocalEnergyJVP(vector, ecp, cpu_jvp); first->applyLocalEnergyJVP(vector, ecp, gpu_jvp);
  checkClose(gpu_jvp.results()[0], cpu_jvp.results()[0]);
  REQUIRE(first->capabilities().maximum_sample_tile_size == 4);
  // Warm Q9 uses 4+4+1 virtual tiles, plus one unchanged accepted-reference call.
  // Actual completion markers are checked in the root CUDA gate with printout enabled.
  const auto jvp_storage = first->storageDiagnostics();
  const auto before_real_jvp = metrics.snapshot();
  RecordingSampleSink warm_real;
  first->applyLocalEnergyJVP(vector, ecp, warm_real);
  const auto after_real_jvp = metrics.snapshot();
  CHECK(after_real_jvp.allocation_calls == before_real_jvp.allocation_calls);
  CHECK(after_real_jvp.live_bytes == before_real_jvp.live_bytes);
  std::vector<DerivativeValue> alternate_direction(direction);
  for (std::size_t i = 0; i < p; ++i)
    alternate_direction[i] = {.003 * (int(i % 5) - 2), 0};
  const wftrain::StructuredParameterVectorConstView alternate_vector(cpu.parameterSchema(),
      cpu.snapshotParameters().version, {alternate_direction.data(), alternate_direction.size()});
  RecordingSampleSink cpu_alternate, gpu_alternate;
  reference->applyLocalEnergyJVP(alternate_vector, ecp, cpu_alternate);
  const auto before_alternate_jvp = metrics.snapshot();
  first->applyLocalEnergyJVP(alternate_vector, ecp, gpu_alternate);
  const auto after_alternate_jvp = metrics.snapshot();
  checkClose(gpu_alternate.results()[0], cpu_alternate.results()[0]);
  CHECK(after_alternate_jvp.uploaded_bytes - before_alternate_jvp.uploaded_bytes ==
        after_real_jvp.uploaded_bytes - before_real_jvp.uploaded_bytes);
  CHECK(after_alternate_jvp.allocation_calls == before_alternate_jvp.allocation_calls);
  CHECK(after_alternate_jvp.live_bytes == before_alternate_jvp.live_bytes);
  // REAL64 public tangents reject imaginary values before any numerical work.
  auto invalid_direction = direction;
  invalid_direction[0].imag(.1);
  RecordingSampleSink rejected_direction;
  CHECK_THROWS_WITH(([&]() {
    const wftrain::StructuredParameterVectorConstView invalid(cpu.parameterSchema(),
        cpu.snapshotParameters().version, {invalid_direction.data(), invalid_direction.size()});
    first->applyLocalEnergyJVP(invalid, ecp, rejected_direction);
  }()), Catch::Matchers::ContainsSubstring("JVP direction for real parameters has a nonzero imaginary part"));
  CHECK(rejected_direction.state() == DerivativeSinkState::IDLE);
  CHECK(metrics.snapshot().uploaded_bytes == after_alternate_jvp.uploaded_bytes);
  CHECK(metrics.snapshot().downloaded_bytes == after_alternate_jvp.downloaded_bytes);
  CHECK(first->storageDiagnostics().storage_fingerprint == jvp_storage.storage_fingerprint);
  CHECK(first->storageDiagnostics().retained_numeric_bytes == jvp_storage.retained_numeric_bytes);
  RecordingSampleSink failed_jvp;
  failed_jvp.armConsumeFailure();
  CHECK_THROWS(first->applyLocalEnergyJVP(alternate_vector, ecp, failed_jvp));
  CHECK(failed_jvp.state() == DerivativeSinkState::POISONED);
  CHECK_THROWS(failed_jvp.results());
  RecordingSampleSink retried_jvp;
  first->applyLocalEnergyJVP(alternate_vector, ecp, retried_jvp);
  checkClose(retried_jvp.results()[0], cpu_alternate.results()[0]);
  SelectedParameterSink dual(selected);
  auto dual_channels = channelsFor(*first);
  const auto before_scalar_c1 = metrics.snapshot();
  first->applyVJPs({dual_channels.data(), 1}, wftrain::DerivativeAdjoint::TRANSPOSE, dual);
  const auto scalar_c1_download_calls = metrics.snapshot().download_calls - before_scalar_c1.download_calls;
  DerivativeValue dot{};
  for (std::size_t i = 0; i < p; ++i) dot += direction[i] * dual.results()[0][i];
  checkClose(dot, coefficients[0][0] * gpu_jvp.results()[0]);
  DerivativeValue alternate_dot{};
  for (std::size_t i = 0; i < p; ++i) alternate_dot += alternate_direction[i] * dual.results()[0][i];
  checkClose(alternate_dot, coefficients[0][0] * gpu_alternate.results()[0]);
  // Exact-real C1 reuses the existing B4 score owner for Q9 => 4+4+tail1.
  const std::vector<DerivativeValue> real_coefficient{{.7, 0}};
  auto real_channels = [&](const StreamingDerivativeOperator& op) {
    return std::vector<VJPCoefficientChannel>{{"real_ecp", DerivativeProduct::LOCAL_ENERGY_VJP,
        coefficientsFor(op, real_coefficient), ecp}};
  };
  const auto c1_storage = first->storageDiagnostics();
  std::size_t tiled_download_calls = 0;
  for (const auto adjoint : {wftrain::DerivativeAdjoint::TRANSPOSE, wftrain::DerivativeAdjoint::HERMITIAN})
  {
    auto expected_channels = real_channels(*reference), actual_channels = real_channels(*first);
    SelectedParameterSink expected(selected), actual(selected);
    reference->applyVJPs({expected_channels.data(), 1}, adjoint, expected);
    const auto before = metrics.snapshot();
    first->applyVJPs({actual_channels.data(), 1}, adjoint, actual);
    const auto after = metrics.snapshot();
    tiled_download_calls = after.download_calls - before.download_calls;
    CHECK(after.allocation_calls == before.allocation_calls);
    CHECK(after.live_bytes == before.live_bytes);
    DerivativeValue real_dot{};
    for (std::size_t i = 0; i < p; ++i)
    {
      checkClose(actual.results()[0][i], expected.results()[0][i]);
      real_dot += direction[i] * actual.results()[0][i];
    }
    checkClose(real_dot, real_coefficient[0] * gpu_jvp.results()[0]);
  }
  CHECK(tiled_download_calls < scalar_c1_download_calls);
  CHECK(first->storageDiagnostics().storage_fingerprint == c1_storage.storage_fingerprint);
  CHECK(first->storageDiagnostics().retained_numeric_bytes == c1_storage.retained_numeric_bytes);
  auto c1_channels = real_channels(*first);
  SelectedParameterSink c1_failed(selected);
  c1_failed.armConsumeFailure();
  CHECK_THROWS(first->applyVJPs({c1_channels.data(), 1}, wftrain::DerivativeAdjoint::TRANSPOSE, c1_failed));
  CHECK(c1_failed.state() == DerivativeSinkState::POISONED);
  SelectedParameterSink c1_retry(selected), c1_expected(selected);
  first->applyVJPs({c1_channels.data(), 1}, wftrain::DerivativeAdjoint::TRANSPOSE, c1_retry);
  auto c1_reference = real_channels(*reference);
  reference->applyVJPs({c1_reference.data(), 1}, wftrain::DerivativeAdjoint::TRANSPOSE, c1_expected);
  for (std::size_t i = 0; i < p; ++i) checkClose(c1_retry.results()[0][i], c1_expected.results()[0][i]);
  // A retained pretraining lane is genuinely capacity one: never grow it implicitly.
  PsiFormerWF scalar_gpu("ecp_grid", initialized, electrons, ions, false, {}, false, {}, {}, resolution);
  const auto ne = electrons.getTotalNum();
  std::vector<double> orbital_target(initialized.model_shape.determinants * ne * ne, 0.0), orbital_gradient(p);
  CHECK(std::isfinite(scalar_gpu.evaluateOrbitalMSE(electrons, orbital_target, orbital_gradient,
                                                  scalar_gpu.snapshotParameters().version)));
  auto scalar = capture(scalar_gpu, 94, 5, true);
  CHECK(scalar->capabilities().maximum_sample_tile_size == 1);
  RecordingSampleSink scalar_jvp;
  scalar->applyLocalEnergyJVP(alternate_vector, ecp, scalar_jvp);
  checkClose(scalar_jvp.results()[0], cpu_alternate.results()[0]);
  auto scalar_channels = real_channels(*scalar);
  SelectedParameterSink scalar_result(selected);
  scalar->applyVJPs({scalar_channels.data(), 1}, wftrain::DerivativeAdjoint::TRANSPOSE, scalar_result);
  for (std::size_t i = 0; i < p; ++i) checkClose(scalar_result.results()[0][i], c1_expected.results()[0][i]);
  SelectedParameterSink failed(selected); failed.armConsumeFailure();
  auto channels = channelsFor(*first);
  CHECK_THROWS(first->applyVJPs({channels.data(), channels.size()}, wftrain::DerivativeAdjoint::TRANSPOSE, failed));
  CHECK(failed.state() == DerivativeSinkState::POISONED);
  compare(wftrain::DerivativeAdjoint::TRANSPOSE); // Clean retry after callback abort.
  auto empty = capture(gpu, 91, 5, true, true);
  auto empty_channels = channelsFor(*empty);
  SelectedParameterSink zero(selected);
  empty->applyVJPs({empty_channels.data(), empty_channels.size()}, wftrain::DerivativeAdjoint::TRANSPOSE, zero);
  for (const auto& channel : zero.results())
    for (const auto value : channel) checkClose(value, {});
  // Numerical foundation only: native ECP DEVICE-authority admission remains closed.
  {
    ScopedEnvironmentVariable authority_flag("PSIFORMER_GPU_RESIDENT_DEVICE_AUTHORITY", "1");
    auto authority_resolution = resolution;
    authority_resolution.validated_operations = ps::allPsiFormerAcceleratorOperations();
    PsiFormerWF authority("ecp_grid", initialized, electrons, ions, false, {}, false, {}, {}, authority_resolution);
    const auto norm_bytes = authority.prepareResidentEnergyAdam(48 * p + 4096);
    const auto allowance = authority.prepareResidentDeviceAuthority();
    std::vector<ps::device::AdamBlock> blocks;
    for (const auto& block : authority.parameterSchema().blocks())
      blocks.push_back({block.offset, block.count, block.trainable ? .0017 : 0, block.trainable ? 1u : 0u});
    auto adam = std::shared_ptr<ps::device::PreparedDeviceAdam>(ps::device::PreparedDeviceAdam::prepare(
        p, blocks, ps::device::estimateAdamBytes(p, blocks.size())));
    std::vector<double> zero_moment(p), first_moment(p), second_moment(p);
    adam->seed(zero_moment, zero_moment);
    adam->prepareEnergyGradient(norm_bytes);
    auto complete = capture(authority, 95, 5, true);
    const auto complete_mask = ecp | wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC);
    const std::vector<DerivativeValue> one{{1, 0}}, eo{{-2.3, 0}};
    const auto energy_channels = [&](const StreamingDerivativeOperator& op) {
      return std::vector<VJPCoefficientChannel>{
        {"T", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(op, one), complete_mask},
        {"EO", DerivativeProduct::SCORE_VJP, coefficientsFor(op, eo)},
        {"O", DerivativeProduct::SCORE_VJP, coefficientsFor(op, one)}};
    };
    auto expected_channels = energy_channels(*reference), actual_channels = energy_channels(*complete);
    SelectedParameterSink expected(selected);
    reference->applyVJPs({expected_channels.data(), 3}, wftrain::DerivativeAdjoint::TRANSPOSE, expected);
    auto expected_parameters = cpu.snapshotParameters();
    double expected_norm = 0;
    for (const auto& block : authority.parameterSchema().blocks())
      for (std::size_t i = block.offset; i < block.offset + block.count; ++i)
      {
        const double gradient = 2 * (expected.results()[0][i].real() + expected.results()[1][i].real()
            + 2.3 * expected.results()[2][i].real());
        expected_norm = std::hypot(expected_norm, gradient);
        if (!block.trainable) continue;
        first_moment[i] = .1 * gradient;
        second_moment[i] = .001 * gradient * gradient;
        expected_parameters.values[i] -= .0017 * gradient / (std::abs(gradient) + 1e-8);
      }
    const auto propose = [&] {
      return authority.consumeResidentEnergyAdamDevice(*complete, {actual_channels.data(), 3}, adam,
          1, -2.3, .9, .999, .1, .001, 1e-8, allowance.snapshot_device_bytes);
    };
    { auto discarded = propose(); } // Prime the existing weighted/tile and norm workspaces.
    const auto warm_before = metrics.snapshot();
    auto proposal = propose();
    const auto warm_after = metrics.snapshot();
    CHECK(warm_after.live_bytes - warm_before.live_bytes == 8 * p);
    CHECK(warm_after.allocation_calls - warm_before.allocation_calls == 1); // Owning candidate only.
    CHECK(warm_after.downloaded_bytes - warm_before.downloaded_bytes < 8 * p);
    CHECK(proposal->gradientNorm() == Catch::Approx(expected_norm).epsilon(2e-8).margin(2e-8));
    { auto discarded = authority.prepareResidentDeviceAdamPublication(std::move(proposal), 0, 0); }
    // A genuinely singular zero-weight point aborts the shared C3, then the good grid retries.
    auto bad = capture(authority, 96, 5, true, false, true);
    auto bad_channels = energy_channels(*bad);
    CHECK_THROWS(authority.consumeResidentEnergyAdamDevice(*bad, {bad_channels.data(), 3}, adam,
        1, -2.3, .9, .999, .1, .001, 1e-8, allowance.snapshot_device_bytes));
    CHECK(authority.parameterVersion() == 0);
    std::vector<double> rejected_first(p), rejected_second(p);
    adam->download(rejected_first, rejected_second);
    CHECK(rejected_first == zero_moment);
    CHECK(rejected_second == zero_moment);
    proposal = propose();
    auto publication = authority.prepareResidentDeviceAdamPublication(std::move(proposal), 0, 0);
    CHECK(publication->commit() == 1);
    publication.reset();
    // Current GPU accepted-grid capture must not refresh stale native weights.
    PsiFormerWF coherent("ecp_grid", initialized, electrons, ions);
    coherent.publishParameters(expected_parameters, 0);
    std::vector<QMCTraits::ValueType> current_ratios, reference_ratios;
    const auto capture_before = metrics.snapshot();
    auto next_grid = capture(authority, 97, 5, true, false, false, &current_ratios);
    auto coherent_grid = capture(coherent, 97, 5, true, false, false, &reference_ratios);
    REQUIRE(current_ratios.size() == reference_ratios.size());
    for (std::size_t i = 0; i < current_ratios.size(); ++i)
      checkClose(current_ratios[i], reference_ratios[i]);
    // Also qualify direct scalar and ragged dispatch, not just sparse descriptors.
    VirtualParticleSet current_virtual(electrons), reference_virtual(electrons);
    std::vector<ParticleSet::PosType> displacements{{.013, -.017, .009}, {-.021, .004, .012}};
    current_virtual.makeMoves(electrons, 0, displacements);
    reference_virtual.makeMoves(electrons, 0, displacements);
    std::vector<QMCTraits::ValueType> current_scalar(2), reference_scalar(2);
    authority.evaluateRatios(current_virtual, current_scalar);
    coherent.evaluateRatios(reference_virtual, reference_scalar);
    for (std::size_t i = 0; i < 2; ++i) checkClose(current_scalar[i], reference_scalar[i]);
    RefVectorWithLeader<WaveFunctionComponent> current_components(authority, {authority});
    RefVectorWithLeader<const VirtualParticleSet> current_virtuals(current_virtual, {current_virtual});
    ResourceCollection prototype("stale_ecp_values"); authority.createResource(prototype);
    ResourceCollection resources(prototype);
    std::vector<std::vector<QMCTraits::ValueType>> ragged{{0, 0}};
    {
      ResourceCollectionTeamLock<WaveFunctionComponent> lock(resources, current_components);
      authority.mw_evaluateRatios(current_components, current_virtuals, ragged);
    }
    for (std::size_t i = 0; i < 2; ++i) checkClose(ragged[0][i], reference_scalar[i]);
    CHECK(metrics.snapshot().downloaded_bytes - capture_before.downloaded_bytes < 8 * p);
    CHECK(next_grid->parameterVersion() == 1);
    CHECK(coherent_grid->parameterVersion() == 1);
    const auto before_cold_inspection = metrics.snapshot();
    const auto actual_parameters = authority.snapshotParameters(); // Explicit cold inspection.
    CHECK(metrics.snapshot().downloaded_bytes - before_cold_inspection.downloaded_bytes == 8 * p);
    app_log() << "Resident ECP stale-HOST VALUE capture: sparse=yes, scalar=yes, ragged=yes, version="
              << next_grid->parameterVersion() << ", warm_downloaded_bytes="
              << before_cold_inspection.downloaded_bytes - capture_before.downloaded_bytes
              << ", cold_mirror_bytes=" << 8 * p << "; independent CPU ratios matched" << std::endl;
    for (std::size_t i = 0; i < p; ++i)
      CHECK(actual_parameters.values[i] == Catch::Approx(expected_parameters.values[i]).epsilon(2e-8).margin(2e-10));
    std::vector<double> actual_first(p), actual_second(p);
    adam->download(actual_first, actual_second);
    for (std::size_t i = 0; i < p; ++i)
    {
      CHECK(actual_first[i] == Catch::Approx(first_moment[i]).epsilon(2e-8).margin(2e-8));
      CHECK(actual_second[i] == Catch::Approx(second_moment[i]).epsilon(2e-8).margin(2e-8));
    }
    CHECK_THROWS(propose()); // Old accepted grid/source cannot survive publication.
  }
  // Independent finite difference of complete ratios on the exact same frozen grid.
  const auto base = cpu.snapshotParameters();
  const auto energy = [&](double displacement) {
    auto candidate = base; candidate.version = cpu.snapshotParameters().version;
    for (std::size_t i = 0; i < p; ++i) candidate.values[i] += displacement * direction[i].real();
    cpu.publishParameters(candidate, candidate.version);
    RefVectorWithLeader<WaveFunctionComponent> components(cpu, {cpu});
    RefVectorWithLeader<ParticleSet> particles(electrons, {electrons});
    VirtualParticleSet virtual_set(electrons);
    RefVectorWithLeader<VirtualParticleSet> virtuals(virtual_set, {virtual_set});
    ResourceCollection prototype("ecp_fd"); cpu.createResource(prototype);
    ResourceCollection resources(prototype);
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(resources, components);
    std::vector<QMCTraits::ValueType> ratios(points.size());
    cpu.mw_evaluateVirtualRatios(components, particles, virtuals, batch, ratios);
    double result = 0;
    for (std::size_t i = 0; i < points.size(); ++i)
      result += bare[i] * ratios[i] * std::exp(.23 * (points[i][0] - electrons.R[i < 5 ? 0 : 1][0]));
    return result;
  };
  const double epsilon = 2e-5;
  const double finite_difference = (energy(epsilon) - energy(-epsilon)) / (2 * epsilon);
  CHECK(gpu_jvp.results()[0].real() == Catch::Approx(finite_difference).epsilon(2e-6).margin(2e-8));
  // A same-spin coincidence is undefined even when its physical coefficient is zero.
  CHECK_THROWS(([&]() {
    auto node = capture(gpu, 92, 5, true, false, true);
    auto node_channels = channelsFor(*node);
    SelectedParameterSink rejected(selected);
    node->applyVJPs({node_channels.data(), node_channels.size()}, wftrain::DerivativeAdjoint::TRANSPOSE, rejected);
  }()));
  CHECK_THROWS(([&]() {
    auto node = capture(gpu, 93, 5, true, false, true);
    auto node_channels = real_channels(*node);
    SelectedParameterSink rejected(selected);
    node->applyVJPs({node_channels.data(), 1}, wftrain::DerivativeAdjoint::TRANSPOSE, rejected);
  }()));
  CHECK_THROWS(([&]() {
    auto node = capture(gpu, 95, 5, true, false, true);
    RecordingSampleSink rejected;
    node->applyLocalEnergyJVP(alternate_vector, ecp, rejected);
  }()));
  RecordingSampleSink after_node_failure;
  first->applyLocalEnergyJVP(alternate_vector, ecp, after_node_failure);
  checkClose(after_node_failure.results()[0], cpu_alternate.results()[0]);
}
#endif

#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
TEST_CASE("PsiFormer real resident HOST outputs retain bounded chunks",
          "[wavefunction][psiformer][training][gpu_host_chunks]")
{
  using namespace testing::psiformer;
  using wftrain::DerivativeAdjoint;
  using wftrain::StructuredParameterVectorConstView;
  namespace ps = psiformer;
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
  { WARN("NOT_RUN: HOST chunks require a HIP device"); return; }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
  { WARN("NOT_RUN: HOST chunks require a CUDA device"); return; }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  auto leaves = makeLayout(4, 2, true, 1, 4);
  for (auto& leaf : leaves)
    for (auto& extent : leaf.shape)
      if (extent == 256 || extent == 64) extent = 8;
  ps::InitializedPsiFormerParameters initialized;
  initialized.model_shape = {2, 2, 2, 2, 8, 2, 1};
  initialized.profile = ps::DEEPQMC_PSIFORMER_V1; initialized.seed = 37;
  initialized.values = makeParameters("lih", leaves);
  std::size_t cursor = 0;
  for (const auto& leaf : leaves)
  {
    const auto begin = cursor; cursor += product(leaf.shape);
    initialized.layouts.push_back({leaf.module, leaf.name, leaf.shape, begin, cursor});
  }
  const SimulationCell cell;
  auto electrons = makeElectrons(cell);
  ParticleSet ions(cell); ions.setName("ion0"); ions.create({1, 1});
  auto& species = ions.getSpeciesSet(); species.addSpecies("Li"); species.addSpecies("H");
  const int charge = species.addAttribute("charge"); species(charge, 0) = 3; species(charge, 1) = 1;
  ions.R[0] = {0, 0, 0}; ions.R[1] = {0, 0, 3.01392421}; ions.resetGroups(); ions.update();
  ps::PsiFormerAcceleratorResolution resolution;
#ifdef QMC_CUDA2HIP
  resolution.backend = ps::PsiFormerAcceleratorBackend::HIP;
#else
  resolution.backend = ps::PsiFormerAcceleratorBackend::CUDA;
#endif
  resolution.reason = ps::PsiFormerAcceleratorResolutionReason::ACCELERATOR_READY;
  resolution.readiness_records_consulted = 1;
  resolution.validated_operations = ps::psiFormerAcceleratorInferenceOperations();
  PsiFormerWF cpu("host_chunks", initialized, electrons, ions);
  PsiFormerWF gpu("host_chunks", initialized, electrons, ions, false, {}, false, {}, {}, resolution);
  const auto p = cpu.parameterSchema().parameterCount();
  std::vector<std::size_t> selected(p); std::iota(selected.begin(), selected.end(), 0);
  std::vector<std::unique_ptr<ParticleSet>> samples;
  std::vector<std::unique_ptr<PsiFormerWF>> gpu_copies, cpu_copies;
  RefVectorWithLeader<WaveFunctionComponent> gc(gpu), cc(cpu);
  for (int i = 0; i < 8; ++i)
  {
    samples.push_back(std::make_unique<ParticleSet>(electrons));
    samples.back()->R[0][0] += .003 * i; samples.back()->update();
    samples.back()->G = QMCTraits::GradType{}; samples.back()->L = QMCTraits::ValueType{};
    cpu.evaluateLog(*samples.back(), samples.back()->G, samples.back()->L);
    for (auto& gradient : samples.back()->G) gradient[0] += .23; // Fixed complete-ratio companion.
    if (i == 0) { gc.push_back(gpu); cc.push_back(cpu); }
    else
    {
      gpu_copies.push_back(std::make_unique<PsiFormerWF>(gpu)); cpu_copies.push_back(std::make_unique<PsiFormerWF>(cpu));
      gc.push_back(*gpu_copies.back()); cc.push_back(*cpu_copies.back());
    }
  }
  // RefVectorWithLeader requires the first stored sample to be its actual leader.
  RefVectorWithLeader<ParticleSet> batch(*samples.front());
  for (auto& sample : samples) batch.push_back(*sample);
  auto reference = cpu.makeStreamingDerivativeOperator(cc, batch, 0, 0, 17);
  auto products = gpu.makeStreamingDerivativeOperator(gc, batch, 0, 0, 17);
  REQUIRE(p % 17 != 0);
  const auto cold = products->storageDiagnostics();
  CHECK(cold.complex_parameter_vectors == 0);
  CHECK(cold.shared_workspace_reserved_bytes == 48 * p);
  const std::array<std::vector<DerivativeValue>, 3> weights{{
      std::vector<DerivativeValue>(8, {.3, -.2}), std::vector<DerivativeValue>(8, {-.4, .7}),
      std::vector<DerivativeValue>(8, DerivativeValue{})}};
  const auto channels = [&](const StreamingDerivativeOperator& op) {
    return std::array<VJPCoefficientChannel, 3>{{
        {"kinetic", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(op, weights[0]),
            wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC)},
        {"score", DerivativeProduct::SCORE_VJP, coefficientsFor(op, weights[1]), 0},
        {"zero", DerivativeProduct::SCORE_VJP, coefficientsFor(op, weights[2]), 0}}};
  };
  auto expected_channels = channels(*reference), actual_channels = channels(*products);
  // Reuse the signed ragged accepted-grid fixture, now across all eight immutable samples.
  const auto bind_grid = [&](PsiFormerWF& component, RefVectorWithLeader<WaveFunctionComponent>& components,
                             StreamingDerivativeOperator& op, bool node = false) {
    std::vector<std::size_t> offsets{0};
    std::vector<VirtualParticleBatch::Segment> segments;
    std::vector<ParticleSet::PosType> points;
    std::vector<QMCTraits::ValueType> bare;
    for (std::size_t sample = 0; sample < samples.size(); ++sample)
    {
      segments.push_back({static_cast<int>(sample), 0}); // The fixture has exactly eight samples.
      for (std::size_t i = 0; i < (sample == 0 ? 2 : 1); ++i)
      {
        points.push_back(samples[sample]->R[0] + ParticleSet::PosType{.003 * (sample + i + 1), -.011, .007});
        bare.push_back((sample % 2 ? -.13 : .21) / (sample + i + 1));
      }
      offsets.push_back(points.size());
    }
    if (node) { points[0] = samples[0]->R[1]; bare[0] = 0; }
    VirtualParticleBatch grid(8, offsets, segments, points);
    std::vector<std::unique_ptr<VirtualParticleSet>> virtual_owners;
    for (auto& sample : samples) virtual_owners.push_back(std::make_unique<VirtualParticleSet>(*sample));
    RefVectorWithLeader<VirtualParticleSet> virtuals(*virtual_owners.front());
    for (auto& owner : virtual_owners) virtuals.push_back(*owner);
    std::vector<QMCTraits::ValueType> ratios(points.size());
    WaveFunctionComponent::EvaluationStamp stamp;
    ResourceCollection prototype("host_ecp_grid"); component.createResource(prototype);
    ResourceCollection resources(prototype);
    {
      ResourceCollectionTeamLock<WaveFunctionComponent> lock(resources, components);
      stamp = component.mw_evaluateVirtualRatios(components, batch, virtuals, grid, ratios);
    }
    for (std::size_t sample = 0; sample < samples.size(); ++sample)
      for (auto i = offsets[sample]; i < offsets[sample + 1]; ++i)
        ratios[i] *= std::exp(.23 * (points[i][0] - samples[sample]->R[0][0]));
    auto consumer = component.makeNonLocalECPDerivativeConsumer(components, batch, 17, true, true);
    wftrain::NonLocalECPDerivativeContext context;
    context.provider_id = op.parameterSchema().providerId(); context.schema_fingerprint = op.parameterSchema().fingerprint();
    context.parameter_version = op.parameterVersion(); context.batch_ordinal = op.batchOrdinal();
    context.sample_count = 8; context.expected_point_count = 9; context.grid_fingerprint = 0x731934d1;
    auto capture = channels(op);
    capture[0].local_energy_term_mask = wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP);
    SelectedParameterSink idle(selected);
    consumer->begin(context, {capture.data(), 1}, idle);
    consumer->consume({grid, {bare.data(), bare.size()}, {ratios.data(), ratios.size()}, {&stamp, 1}, 0,
        context.grid_fingerprint});
    consumer->end();
    CHECK(idle.state() == DerivativeSinkState::IDLE);
    op.bindNonLocalECPJVPReplay(consumer->releaseJVPReplay());
  };
  bind_grid(cpu, cc, *reference); bind_grid(gpu, gc, *products);
  CHECK(products->storageDiagnostics().complex_parameter_vectors == 0);
  SelectedParameterSink expected(selected), actual(selected);
  reference->applyVJPs({expected_channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, expected);
  products->applyVJPs({actual_channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual);
  const auto compare = [&]() {
    for (std::size_t channel = 0; channel < 3; ++channel)
      for (std::size_t j = 0; j < p; ++j) checkClose(actual.results()[channel][j], expected.results()[channel][j]);
  };
  compare();
  CHECK(actual.consumedRecords() == 3 * products->parameterChunkPlan().chunks().size());
  auto mixed_zero = actual_channels;
  mixed_zero[2].product = DerivativeProduct::LOCAL_ENERGY_VJP;
  mixed_zero[2].local_energy_term_mask = 0;
  mixed_zero[2].coefficients = coefficientsFor(*products, weights[0]);
  actual.reset();
  CHECK_THROWS_WITH(products->applyVJPs({mixed_zero.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual),
      Catch::Matchers::ContainsSubstring("lacks requested local-energy term coverage"));
  CHECK(actual.state() == DerivativeSinkState::IDLE); // Empty energy masks are not an admitted physical term.
  const auto warm = products->storageDiagnostics();
  CHECK(warm.complex_parameter_vectors == 0);
  CHECK(warm.shared_workspace_reserved_bytes == 48 * p);
  actual.reset(); actual.armConsumeFailure();
  CHECK_THROWS(products->applyVJPs({actual_channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual));
  CHECK(actual.state() == DerivativeSinkState::POISONED);
  actual.reset();
  std::vector<DerivativeValue> zero(p);
  const StructuredParameterVectorConstView direction(gpu.parameterSchema(), products->parameterVersion(), {zero.data(), p});
  actual.onConsumeOnce([&]() { RecordingSampleSink nested; products->applyScoreJVP(direction, nested); });
  CHECK_THROWS_WITH(products->applyVJPs({actual_channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual),
      Catch::Matchers::ContainsSubstring("cannot be re-entered"));
  actual.reset();
  const auto before = ps::device::deviceMemoryMetrics(ordinal).snapshot();
  products->applyVJPs({actual_channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual);
  compare();
  const auto after = ps::device::deviceMemoryMetrics(ordinal).snapshot();
  CHECK(after.allocation_calls == before.allocation_calls);
  CHECK(after.live_bytes == before.live_bytes);
  CHECK(products->storageDiagnostics().storage_fingerprint == warm.storage_fingerprint);
  const auto complete = wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC) |
      wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP);
  expected_channels[0].local_energy_term_mask = actual_channels[0].local_energy_term_mask = complete;
  expected.reset(); actual.reset();
  reference->applyVJPs({expected_channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, expected);
  products->applyVJPs({actual_channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual);
  compare();
  CHECK(products->storageDiagnostics().complex_parameter_vectors == 0);
  auto bad = gpu.makeStreamingDerivativeOperator(gc, batch, 2, 0, 17);
  bind_grid(gpu, gc, *bad, true);
  auto bad_channels = channels(*bad); bad_channels[0].local_energy_term_mask = complete;
  actual.reset();
  CHECK_THROWS(bad->applyVJPs({bad_channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual));
  CHECK(actual.state() == DerivativeSinkState::POISONED);
  CHECK(gpu.parameterVersion() == products->parameterVersion());
  CHECK(gpu.snapshotParameters().values == initialized.values);
  actual.reset();
  products->applyVJPs({actual_channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual);
  compare(); // Failed zero-weight nodal geometry did not replace the good immutable replay.
  auto candidate = gpu.snapshotParameters(); candidate.values[0] += .001;
  actual.reset();
  actual.onConsumeOnce([&]() { gpu.publishParameters(candidate, candidate.version); });
  products->applyVJPs({actual_channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual);
  compare(); // Every later chunk must still be the OLD numerical generation.
  CHECK(gpu.parameterVersion() == candidate.version + 1);
  actual.reset();
  CHECK_THROWS(products->applyVJPs({actual_channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual));
  cpu.publishParameters(candidate, candidate.version);
  for (auto& sample : samples)
  {
    sample->G = QMCTraits::GradType{}; sample->L = QMCTraits::ValueType{};
    cpu.evaluateLog(*sample, sample->G, sample->L);
    for (auto& gradient : sample->G) gradient[0] += .23;
  }
  auto fresh_reference = cpu.makeStreamingDerivativeOperator(cc, batch, 1, 0, 17);
  auto fresh_products = gpu.makeStreamingDerivativeOperator(gc, batch, 1, 0, 17);
  bind_grid(cpu, cc, *fresh_reference); bind_grid(gpu, gc, *fresh_products);
  auto fresh_expected = channels(*fresh_reference), fresh_actual = channels(*fresh_products);
  expected.reset(); actual.reset();
  fresh_reference->applyVJPs({fresh_expected.data(), 3}, DerivativeAdjoint::TRANSPOSE, expected);
  fresh_products->applyVJPs({fresh_actual.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual);
  compare();
  const auto before_fallback = fresh_products->storageDiagnostics();
  fresh_expected[0].local_energy_term_mask = fresh_actual[0].local_energy_term_mask = complete;
  std::swap(fresh_expected[0], fresh_expected[1]); std::swap(fresh_actual[0], fresh_actual[1]);
  expected.reset(); actual.reset();
  fresh_reference->applyVJPs({fresh_expected.data(), 3}, DerivativeAdjoint::TRANSPOSE, expected);
  fresh_products->applyVJPs({fresh_actual.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual);
  compare(); // Same complete response, nonmatching order must use the original full-HOST fallback.
  auto zero_energy = fresh_actual[0];
  zero_energy.product = DerivativeProduct::LOCAL_ENERGY_VJP;
  zero_energy.local_energy_term_mask = 0;
  SelectedParameterSink zero_only(selected);
  CHECK_THROWS(fresh_products->applyVJPs({&zero_energy, 1}, DerivativeAdjoint::TRANSPOSE, zero_only));
  CHECK(zero_only.state() == DerivativeSinkState::IDLE);
  RefVectorWithLeader<WaveFunctionComponent> single(gpu, {gpu});
  RefVectorWithLeader<ParticleSet> one(*samples.front(), {*samples.front()});
  auto fallback = gpu.makeStreamingDerivativeOperator(single, one, 2, 0, 17);
  const auto materialized = fallback->storageDiagnostics();
  CHECK(materialized.complex_parameter_vectors == 3);
  CHECK(materialized.shared_workspace_reserved_bytes == 0);
  CHECK(materialized.shared_workspace_bytes == before_fallback.shared_workspace_bytes + 48 * p);
  CHECK(materialized.shared_workspace_bytes == before_fallback.shared_workspace_bytes +
        before_fallback.shared_workspace_reserved_bytes);
  std::cout << "PASS real HOST chunks: W8/C3 tail/complex-zero, cold_HOST_channels=0, reserved=48P, publication/reentry/abort/retry\n";
  std::cout << "PASS ordered ECP HOST C3: W8/Q9 signed complete CPU parity, no double ECP, node retry, publication, ordered fallback\n";
}

TEST_CASE("PsiFormer real resident DEVICE authority fences cold readers",
          "[wavefunction][psiformer][training][gpu_resident_authority]")
{
  using namespace testing::psiformer;
  namespace ps = psiformer;
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
  { WARN("NOT_RUN: resident authority requires a HIP device"); return; }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
  { WARN("NOT_RUN: resident authority requires a CUDA device"); return; }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  ScopedEnvironmentVariable authority_flag("PSIFORMER_GPU_RESIDENT_DEVICE_AUTHORITY", "1");
  auto leaves = makeLayout(4, 2, true, 1, 4);
  for (auto& leaf : leaves)
    for (auto& extent : leaf.shape)
      if (extent == 256 || extent == 64) extent = 8;
  ps::InitializedPsiFormerParameters initialized;
  initialized.model_shape = {2, 2, 2, 2, 8, 2, 1};
  initialized.profile = ps::DEEPQMC_PSIFORMER_V1;
  initialized.seed = 37;
  initialized.values = makeParameters("lih", leaves);
  std::size_t cursor = 0;
  for (const auto& leaf : leaves)
  {
    const auto begin = cursor; cursor += product(leaf.shape);
    initialized.layouts.push_back({leaf.module, leaf.name, leaf.shape, begin, cursor});
  }
  const SimulationCell cell;
  auto electrons = makeElectrons(cell);
  ParticleSet ions(cell); ions.setName("ion0"); ions.create({1, 1});
  auto& species = ions.getSpeciesSet(); species.addSpecies("Li"); species.addSpecies("H");
  const int charge = species.addAttribute("charge"); species(charge, 0) = 3; species(charge, 1) = 1;
  ions.R[0] = {0, 0, 0}; ions.R[1] = {3.0154361724853516, 0, 0}; ions.resetGroups(); ions.update();
  auto resolution = ps::PsiFormerAcceleratorResolution{};
#ifdef QMC_CUDA2HIP
  resolution.backend = ps::PsiFormerAcceleratorBackend::HIP;
#else
  resolution.backend = ps::PsiFormerAcceleratorBackend::CUDA;
#endif
  resolution.reason = ps::PsiFormerAcceleratorResolutionReason::ACCELERATOR_READY;
  resolution.readiness_records_consulted = 1;
  resolution.validated_operations = ps::allPsiFormerAcceleratorOperations();
  PsiFormerWF cpu("authority", initialized, electrons, ions, true, {0, 5, 10});
  PsiFormerWF gpu("authority", initialized, electrons, ions, true, {0, 5, 10}, false, {}, {}, resolution);
  cpu.validateSystem(electrons, ions, "all_electron");
  gpu.validateSystem(electrons, ions, "all_electron");
  CHECK_FALSE(cpu.residentDeviceAuthorityRequested());
  REQUIRE(gpu.residentDeviceAuthorityRequested());
  if (const char* path = std::getenv("PSIFORMER_TEST_RESIDENT_MODEL_EXPORT"))
    gpu.exportModel(path); // Explicit test-only fixture export before any mutation.
  const auto p = gpu.parameterSchema().parameterCount();
  const auto bytes = 8 * p;
  SECTION("DEVICE SR pins the complete resource chain")
  {
    ScopedEnvironmentVariable reuse("PSIFORMER_GPU_RESIDENT_ENERGY_SR_REUSE", "on");
    PsiFormerWF host("authority", initialized, electrons, ions, true, {0, 5, 10}, false, {}, {}, resolution);
    host.validateSystem(electrons, ions, "all_electron");
    const std::pair<std::size_t, std::size_t> budget{64 * 1024 * 1024, 1024 * 1024};
    gpu.prepareResidentEnergySR(7, budget); host.prepareResidentEnergySR(7, budget);
    CHECK(gpu.prepareResidentDeviceAuthority().snapshot_device_bytes == bytes);
    std::vector<std::unique_ptr<ParticleSet>> samples;
    std::vector<std::unique_ptr<PsiFormerWF>> gpu_copies, cpu_copies, host_copies;
    RefVectorWithLeader<WaveFunctionComponent> gc(gpu), cc(cpu), hc(host);
    std::vector<double> weights(7), energies(7);
    double total_weight = 0, mean_energy = 0, group_weight[2]{};
    for (std::size_t s = 0; s < 7; ++s)
    {
      samples.push_back(std::make_unique<ParticleSet>(electrons));
      samples.back()->R[0][0] += s % 2 ? .031 : 0;
      samples.back()->update();
      weights[s] = s % 2 ? 2 : 1; energies[s] = s % 2 ? -1.9 : -2.3;
      group_weight[s % 2] += weights[s]; total_weight += weights[s];
      mean_energy += weights[s] * energies[s];
      if (s == 0) { gc.push_back(gpu); cc.push_back(cpu); hc.push_back(host); }
      else
      {
        gpu_copies.push_back(std::make_unique<PsiFormerWF>(gpu));
        cpu_copies.push_back(std::make_unique<PsiFormerWF>(cpu));
        host_copies.push_back(std::make_unique<PsiFormerWF>(host));
        gc.push_back(*gpu_copies.back()); cc.push_back(*cpu_copies.back()); hc.push_back(*host_copies.back());
      }
    }
    mean_energy /= total_weight;
    RefVectorWithLeader<ParticleSet> batch(*samples.front());
    for (auto& sample : samples) batch.push_back(*sample);
    std::vector<std::size_t> selected(p); std::iota(selected.begin(), selected.end(), 0);
    const auto active = registerSelectedParameters(gpu);
    RefVectorWithLeader<WaveFunctionComponent> single(gpu, {gpu});
    RefVectorWithLeader<ParticleSet> one(*samples.front(), {*samples.front()});
    RefVector<ParticleSet::ParticleGradient> gradients{samples.front()->G};
    RefVector<ParticleSet::ParticleLaplacian> laplacians{samples.front()->L};
    ResourceCollection prototype("resident_sr_authority"); gpu.createResource(prototype);
    ResourceCollection resources(prototype);
    ResourceCollectionTeamLock<WaveFunctionComponent> team(resources, single);
    auto& metrics = ps::device::deviceMemoryMetrics(ordinal);
    wftrain::KrylovSolverControl control;
    control.maximum_iterations = 20; control.maximum_operator_applications = 24;
    control.relative_tolerance = 1e-9; control.absolute_tolerance = 1e-10;
    constexpr double damping = .3, rate = 1e-4;
    for (std::size_t step = 0; step < 3; ++step)
    {
      for (auto& sample : samples)
      {
        sample->G = QMCTraits::GradType{}; sample->L = QMCTraits::ValueType{};
        cpu.evaluateLog(*sample, sample->G, sample->L);
      }
      auto co = cpu.makeStreamingDerivativeOperator(cc, batch, step, 0, 17);
      auto go = gpu.makeStreamingDerivativeOperator(gc, batch, step, 0, 17);
      auto ho = host.makeStreamingDerivativeOperator(hc, batch, step, 0, 17);
      std::vector<DerivativeValue> w(7), ew(7), first(7), second(7);
      for (std::size_t s = 0; s < 7; ++s)
      { w[s] = weights[s]; ew[s] = weights[s] * energies[s]; (s % 2 ? second : first)[s] = 1; }
      const auto channelsFor = [&](const StreamingDerivativeOperator& op) {
        return std::vector<VJPCoefficientChannel>{
          {"T", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(op, w), wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC)},
          {"EO", DerivativeProduct::SCORE_VJP, coefficientsFor(op, ew)},
          {"O", DerivativeProduct::SCORE_VJP, coefficientsFor(op, w)}};
      };
      auto c = channelsFor(*co), g = channelsFor(*go), h = channelsFor(*ho);
      SelectedParameterSink oracle(selected), group_scores(selected);
      co->applyVJPs({c.data(), c.size()}, wftrain::DerivativeAdjoint::TRANSPOSE, oracle);
      const VJPCoefficientChannel groups[] = {
          {"first", DerivativeProduct::SCORE_VJP, coefficientsFor(*co, first)},
          {"second", DerivativeProduct::SCORE_VJP, coefficientsFor(*co, second)}};
      co->applyVJPs({groups, 2}, wftrain::DerivativeAdjoint::TRANSPOSE, group_scores);
      auto expected = cpu.snapshotParameters();
      std::vector<double> gradient(p), delta(p);
      double squared = 0, dot = 0;
      for (const auto& block : cpu.parameterSchema().blocks())
        for (std::size_t i = block.offset; i < block.offset + block.count; ++i)
        {
          gradient[i] = block.trainable ? 2 / total_weight * (oracle.results()[0][i].real() +
              oracle.results()[1][i].real() - mean_energy * oracle.results()[2][i].real()) : 0;
          delta[i] = block.trainable ? group_scores.results()[0][i].real() / 4 -
              group_scores.results()[1][i].real() / 3 : 0;
          squared += delta[i] * delta[i]; dot += delta[i] * gradient[i];
        }
      const double a = group_weight[0] * group_weight[1] / (total_weight * total_weight);
      double norm = 0;
      for (std::size_t i = 0; i < p; ++i)
      {
        const double direction = gradient[i] / damping - a * delta[i] * dot /
            (damping * (damping + a * squared));
        const double old = expected.values[i]; expected.values[i] -= rate * direction;
        norm = std::hypot(norm, expected.values[i] - old);
      }
      // Seven numerical repeats preserve the old HOST terminal's S>=7 policy.
      std::vector<double> host_candidate(p);
      const auto host_diagnostics = host.consumeResidentEnergySR(*ho, {h.data(), h.size()}, weights,
          host.snapshotParameters().values, total_weight, mean_energy, damping, rate, control, budget, host_candidate);
      REQUIRE(host_diagnostics.solve.converged);
      const auto propose = [&] {
        return gpu.consumeResidentEnergySRDevice(*go, {g.data(), g.size()}, weights, total_weight,
            mean_energy, damping, rate, control, budget, bytes);
      };
      const auto warm = metrics.snapshot();
      auto proposal = propose();
      if (step)
      {
        // Fixed P666/B7: source-derived coordinate/scalar/status traffic, not P payloads.
        CHECK(p == 666);
        CHECK(metrics.snapshot().downloaded_bytes - warm.downloaded_bytes == 17620);
        CHECK(metrics.snapshot().uploaded_bytes - warm.uploaded_bytes == 6552);
      }
      CHECK(proposal->diagnostics().solve.converged);
      CHECK(proposal->diagnostics().solve.iterations == 2);
      CHECK(proposal->displacementNorm() == Catch::Approx(norm).epsilon(2e-7).margin(2e-9));
      CHECK_THROWS(gpu.prepareResidentDeviceSRPublication(std::move(proposal), step + 1, 0));
      proposal = propose();
      CHECK_THROWS(cpu.prepareResidentDeviceSRPublication(std::move(proposal), step, 0));
      proposal = propose();
      { auto discarded = gpu.prepareResidentDeviceSRPublication(std::move(proposal), step, 0); }
      CHECK(gpu.parameterVersion() == step);
      proposal = propose();
      auto publication = gpu.prepareResidentDeviceSRPublication(std::move(proposal), step, 0);
      const auto before = metrics.snapshot(); CHECK(publication->commit() == step + 1);
      const auto after = metrics.snapshot();
      CHECK(after.uploaded_bytes == before.uploaded_bytes); CHECK(after.downloaded_bytes == before.downloaded_bytes);
      CHECK(after.device_copied_bytes == before.device_copied_bytes);
      CHECK(after.allocation_calls == before.allocation_calls); CHECK(after.free_calls == before.free_calls);
      publication.reset();
      const auto warm_full = metrics.snapshot();
      samples.front()->G = QMCTraits::GradType{}; samples.front()->L = QMCTraits::ValueType{};
      gpu.mw_evaluateLog(single, one, gradients, laplacians);
      CHECK(metrics.snapshot().downloaded_bytes - warm_full.downloaded_bytes < bytes);
      const auto gpu_g = samples.front()->G; const auto gpu_l = samples.front()->L;
      const auto cold = metrics.snapshot();
      Vector<QMCTraits::ValueType> score(active.size()), kinetic(active.size());
      score = QMCTraits::ValueType{}; kinetic = QMCTraits::ValueType{};
      gpu.evaluateDerivatives(*samples.front(), active, score, kinetic); // Cached executor owner fence.
      const auto actual = gpu.snapshotParameters();
      CHECK(metrics.snapshot().downloaded_bytes - cold.downloaded_bytes == bytes);
      for (std::size_t i = 0; i < p; ++i)
      {
        CHECK(actual.values[i] == Catch::Approx(expected.values[i]).epsilon(2e-7).margin(2e-9));
        CHECK(actual.values[i] == Catch::Approx(host_candidate[i]).epsilon(2e-7).margin(2e-9));
      }
      // Independent candidates were checked above; fence comparisons require identical accepted values.
      auto accepted = expected;
      accepted.values = actual.values;
      cpu.publishParameters(accepted, step); host.publishParameters(accepted, step);
      samples.front()->G = QMCTraits::GradType{}; samples.front()->L = QMCTraits::ValueType{};
      cpu.evaluateLog(*samples.front(), samples.front()->G, samples.front()->L);
      for (std::size_t i = 0; i < 4; ++i)
      {
        for (int d = 0; d < 3; ++d) checkClose(gpu_g[i][d], samples.front()->G[i][d]);
        checkClose(gpu_l[i], samples.front()->L[i]);
      }
      Vector<QMCTraits::ValueType> cpu_score(active.size()), cpu_kinetic(active.size());
      cpu_score = QMCTraits::ValueType{}; cpu_kinetic = QMCTraits::ValueType{};
      cpu.evaluateDerivatives(*samples.front(), active, cpu_score, cpu_kinetic);
      for (std::size_t i = 0; i < active.size(); ++i)
      { checkClose(score[i], cpu_score[i]); checkClose(kinetic[i], cpu_kinetic[i]); }
      CHECK_THROWS(propose()); // Immutable old batch cannot produce the next generation.
    }
    for (auto& sample : samples)
    {
      sample->G = QMCTraits::GradType{}; sample->L = QMCTraits::ValueType{};
      cpu.evaluateLog(*sample, sample->G, sample->L);
    }
    auto fresh = gpu.makeStreamingDerivativeOperator(gc, batch, 4, 0, 17);
    std::vector<DerivativeValue> w(7), ew(7);
    for (std::size_t s = 0; s < 7; ++s) { w[s] = weights[s]; ew[s] = weights[s] * energies[s]; }
    const VJPCoefficientChannel fresh_channels[] = {
        {"T", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*fresh, w), wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC)},
        {"EO", DerivativeProduct::SCORE_VJP, coefficientsFor(*fresh, ew)},
        {"O", DerivativeProduct::SCORE_VJP, coefficientsFor(*fresh, w)}};
    auto stopped = control; stopped.maximum_operator_applications = 0;
    CHECK_THROWS(gpu.consumeResidentEnergySRDevice(*fresh, {fresh_channels, 3}, weights, total_weight,
        mean_energy, damping, rate, stopped, budget, bytes));
    CHECK(gpu.parameterVersion() == 3);
    gpu.prepareResidentEnergySR(7, budget); // Genuine numerical failure needs an explicit new stage.
    auto retry = gpu.consumeResidentEnergySRDevice(*fresh, {fresh_channels, 3}, weights, total_weight,
        mean_energy, damping, rate, control, budget, bytes);
    CHECK(retry->diagnostics().solve.converged);
    retry.reset(); fresh.reset();
    gpu.releaseResidentEnergySRWorkspace(); host.releaseResidentEnergySRWorkspace();
    std::cout << "PASS component DEVICE SR: three_updates=3, samples=7/two_distinct, CPU_rank_one=1, HOST_oracle=1, cold_fence=1, public_driver=0\n";
  }
  SECTION("DEVICE Adam")
  {
  const auto norm_bytes = gpu.prepareResidentEnergyAdam(48 * p + 4096);
  const auto allowances = gpu.prepareResidentDeviceAuthority();
  CHECK(allowances.snapshot_device_bytes == bytes);
  CHECK(allowances.successor_device_bytes == 0);
  CHECK(allowances.cold_host_bytes == bytes);
  std::vector<ps::device::AdamBlock> blocks;
  for (const auto& block : gpu.parameterSchema().blocks())
    blocks.push_back({block.offset, block.count, block.trainable ? .0017 : 0, block.trainable ? 1u : 0u});
  std::shared_ptr<ps::device::PreparedDeviceAdam> adam = ps::device::PreparedDeviceAdam::prepare(
      p, blocks, ps::device::estimateAdamBytes(p, blocks.size()));
  std::vector<double> zero(p), first(p), second(p);
  adam->seed(zero, zero); adam->prepareEnergyGradient(norm_bytes);
  std::vector<std::size_t> selected(p); std::iota(selected.begin(), selected.end(), 0);
  RefVectorWithLeader<WaveFunctionComponent> gpu_components(gpu, {gpu}), cpu_components(cpu, {cpu});
  RefVectorWithLeader<ParticleSet> particles(electrons, {electrons});
  RefVector<ParticleSet::ParticleGradient> gradients{electrons.G};
  RefVector<ParticleSet::ParticleLaplacian> laplacians{electrons.L};
  ResourceCollection prototype("resident_authority"); gpu.createResource(prototype);
  ResourceCollection resources(prototype);
  ResourceCollectionTeamLock<WaveFunctionComponent> team(resources, gpu_components);
  const auto cpu_active = registerSelectedParameters(cpu), gpu_active = registerSelectedParameters(gpu);
  const auto legacy = [&](PsiFormerWF& component, const OptVariables& active) {
    Vector<QMCTraits::ValueType> score(active.size()), kinetic(active.size()), score_only(active.size());
    score = QMCTraits::ValueType{}; kinetic = QMCTraits::ValueType{}; score_only = QMCTraits::ValueType{};
    component.evaluateDerivatives(electrons, active, score, kinetic);
    component.evaluateDerivativesWF(electrons, active, score_only);
    std::vector<double> values;
    for (std::size_t i = 0; i < active.size(); ++i)
    { values.push_back(std::real(score[i])); values.push_back(std::real(kinetic[i])); values.push_back(std::real(score_only[i])); }
    return values;
  };
  const auto full = [&] {
    electrons.G = QMCTraits::GradType{}; electrons.L = QMCTraits::ValueType{};
    gpu.mw_evaluateLog(gpu_components, particles, gradients, laplacians);
  };
  constexpr double beta1 = .9, beta2 = .999, epsilon = 1e-8, energy = -2.3;
  full();
  (void)legacy(gpu, gpu_active); // Prime cached CPU score/kinetic/affine leaves.
  for (std::size_t step = 1; step <= 3; ++step)
  {
    full();
    auto cpu_op = cpu.makeStreamingDerivativeOperator(cpu_components, particles, step, 0, 17);
    auto gpu_op = gpu.makeStreamingDerivativeOperator(gpu_components, particles, step, 0, 17);
    const std::vector<DerivativeValue> one{{1, 0}}, eo{{energy, 0}};
    const auto channelsFor = [&](const StreamingDerivativeOperator& op) {
      return std::vector<VJPCoefficientChannel>{
        {"T", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(op, one), wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC)},
        {"EO", DerivativeProduct::SCORE_VJP, coefficientsFor(op, eo)},
        {"O", DerivativeProduct::SCORE_VJP, coefficientsFor(op, one)}};
    };
    auto cpu_channels = channelsFor(*cpu_op), gpu_channels = channelsFor(*gpu_op);
    SelectedParameterSink oracle(selected);
    cpu_op->applyVJPs({cpu_channels.data(), cpu_channels.size()}, wftrain::DerivativeAdjoint::TRANSPOSE, oracle);
    auto expected = cpu.snapshotParameters();
    double gradient_norm = 0, displacement_norm = 0;
    for (const auto& block : cpu.parameterSchema().blocks())
      for (std::size_t i = block.offset; i < block.offset + block.count; ++i)
      {
        const double g = 2 * (oracle.results()[0][i].real() + oracle.results()[1][i].real() - energy * oracle.results()[2][i].real());
        gradient_norm = std::hypot(gradient_norm, g);
        if (!block.trainable) continue;
        first[i] = beta1 * first[i] + (1 - beta1) * g;
        second[i] = beta2 * second[i] + (1 - beta2) * g * g;
        const double prior = expected.values[i];
        expected.values[i] -= .0017 * (first[i] / (1 - std::pow(beta1, step))) /
            (std::sqrt(second[i] / (1 - std::pow(beta2, step))) + epsilon);
        displacement_norm = std::hypot(displacement_norm, expected.values[i] - prior);
      }
    const auto propose = [&] {
      return gpu.consumeResidentEnergyAdamDevice(*gpu_op, {gpu_channels.data(), gpu_channels.size()}, adam,
          1, energy, beta1, beta2, 1 - std::pow(beta1, step), 1 - std::pow(beta2, step), epsilon, bytes);
    };
    // Denial and discard preserve both the active image and exact retry receipt.
    auto proposal = propose();
    CHECK(proposal->gradientNorm() == Catch::Approx(gradient_norm).epsilon(2e-8).margin(2e-8));
    CHECK(proposal->displacementNorm() == Catch::Approx(displacement_norm).epsilon(2e-8).margin(2e-10));
    CHECK_THROWS(gpu.prepareResidentDeviceAdamPublication(std::move(proposal), step - 1, bytes - 1));
    proposal = propose();
    CHECK_THROWS(gpu.prepareResidentDeviceAdamPublication(std::move(proposal), step, 0));
    proposal = propose();
    CHECK_THROWS(cpu.prepareResidentDeviceAdamPublication(std::move(proposal), step - 1, 0));
    proposal = propose();
    { auto discarded = gpu.prepareResidentDeviceAdamPublication(std::move(proposal), step - 1, 0); }
    CHECK(gpu.parameterVersion() == step - 1);
    proposal = propose();
    auto publication = gpu.prepareResidentDeviceAdamPublication(std::move(proposal), step - 1, 0);
    auto& metrics = ps::device::deviceMemoryMetrics(ordinal);
    const auto before = metrics.snapshot();
    CHECK(publication->commit() == step);
    const auto after = metrics.snapshot();
    CHECK(after.uploaded_bytes == before.uploaded_bytes);
    CHECK(after.downloaded_bytes == before.downloaded_bytes);
    CHECK(after.allocation_calls == before.allocation_calls);
    CHECK(after.free_calls == before.free_calls);
    CHECK(after.device_copied_bytes == before.device_copied_bytes);
    publication.reset();
    cpu.publishParameters(expected, step - 1);
    // Include a post-publication GPU FULL before the explicit cold HOST fence.
    const auto warm = metrics.snapshot(); full();
    CHECK(metrics.snapshot().downloaded_bytes - warm.downloaded_bytes < bytes);
    const auto gpu_gradient = electrons.G; const auto gpu_laplacian = electrons.L;
    const auto cold = metrics.snapshot();
    const auto cached_cpu = legacy(gpu, gpu_active); // Default owner fence, not flat_values alone.
    const auto actual = gpu.snapshotParameters();
    CHECK(metrics.snapshot().downloaded_bytes - cold.downloaded_bytes == bytes);
    REQUIRE(actual.version == step);
    for (std::size_t i = 0; i < p; ++i)
      CHECK(actual.values[i] == Catch::Approx(expected.values[i]).epsilon(2e-8).margin(2e-10));
    const auto fresh = metrics.snapshot(); (void)gpu.snapshotParameters();
    CHECK(metrics.snapshot().downloaded_bytes == fresh.downloaded_bytes);
    const auto expected_cpu = legacy(cpu, cpu_active);
    REQUIRE(cached_cpu.size() == expected_cpu.size());
    for (std::size_t i = 0; i < cached_cpu.size(); ++i)
      CHECK(cached_cpu[i] == Catch::Approx(expected_cpu[i]).epsilon(2e-7).margin(2e-7));
    electrons.G = QMCTraits::GradType{}; electrons.L = QMCTraits::ValueType{};
    cpu.evaluateLog(electrons, electrons.G, electrons.L);
    for (std::size_t i = 0; i < 4; ++i)
    {
      for (int d = 0; d < 3; ++d) checkClose(gpu_gradient[i][d], electrons.G[i][d]);
      checkClose(gpu_laplacian[i], electrons.L[i]);
    }
  }
  std::vector<double> actual_first(p), actual_second(p);
  adam->download(actual_first, actual_second);
  for (std::size_t i = 0; i < p; ++i)
  {
    CHECK(actual_first[i] == Catch::Approx(first[i]).epsilon(2e-8).margin(2e-9));
    CHECK(actual_second[i] == Catch::Approx(second[i]).epsilon(2e-8).margin(2e-9));
  }
  std::cout << "PASS component DEVICE Adam: three_updates=3, CPU_moments=1, cold_fence=1\n";
  }
}
#endif

} // namespace qmcplusplus
