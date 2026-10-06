//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_LegacyWaveFunctionAdapter.cpp
 * @brief High-level tests for the bounded conventional-wavefunction training bridge.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "Particle/ParticleSet.h"
#include "QMCDrivers/WFTrain/EnergyGradientAccumulator.h"
#include "QMCDrivers/WFTrain/FirstOrderOptimizer.h"
#include "QMCDrivers/WFTrain/HighParameterTraining.h"
#include "QMCDrivers/WFTrain/TrainingCheckpoint.h"
#include "QMCWaveFunctions/Optimization/LegacyWaveFunctionAdapter.h"
#include "QMCWaveFunctions/Jastrow/BsplineFunctor.h"
#include "QMCWaveFunctions/Jastrow/TwoBodyJastrow.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "SimulationCell.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

/// Construct a build-domain wavefunction value while preserving complex-test coverage.
QMCTraits::ValueType makeValue(double real, double imaginary = 0.0)
{
#ifdef QMC_COMPLEX
  return {real, imaginary};
#else
  (void)imaginary;
  return real;
#endif
}

/// Convert a build-domain value to the full complex contraction domain.
DerivativeValue derivativeValue(QMCTraits::ValueType value)
{
  return {std::real(value), std::imag(value)};
}

/// Deterministic nonlinear Jastrow-like component using the legacy scalar API.
class LegacyJastrowComponent final : public WaveFunctionComponent, public OptimizableObject
{
public:
  LegacyJastrowComponent(std::array<double, 3> parameters,
                         std::shared_ptr<std::atomic<std::size_t>> derivative_calls,
                         double disabled_parameter = 7.0)
      : WaveFunctionComponent("legacy_jastrow"),
        OptimizableObject("legacy_jastrow_variables"),
        parameters_(parameters),
        derivative_calls_(std::move(derivative_calls))
  {
    myVars.insert("legacy_a", parameters_[0], true, optimize::OTHER_P);
    myVars.insert("legacy_b", parameters_[1], true, optimize::LINEAR_P);
    myVars.insert("legacy_c", parameters_[2], true, optimize::OTHER_P);
    myVars.insert("legacy_disabled", disabled_parameter, false, optimize::OTHER_P);
    setOptimization(true);
  }

  std::string getClassName() const override { return "LegacyJastrowComponent"; }
  bool isOptimizable() const override { return true; }

  void extractOptimizableObjectRefs(UniqueOptObjRefs& references) override { references.push_back(*this); }

  void checkInVariablesExclusive(OptVariables& active) override { active.insertFrom(myVars); }

  void checkOutVariables(const OptVariables& active) override { myVars.getIndex(active); }

  void resetParametersExclusive(const OptVariables& active) override
  {
    for (std::size_t local = 0; local < 3; ++local)
      parameters_[local] = active[myVars.where(static_cast<int>(local))];
  }

  LogValue evaluateLog(const ParticleSet& particles,
                       ParticleSet::ParticleGradient& gradients,
                       ParticleSet::ParticleLaplacian& laplacians) override
  {
    (void)gradients;
    (void)laplacians;
    const auto [linear, quadratic] = coordinateSums(particles);
    log_value_ = LogValue{0.5 * parameters_[0] * parameters_[0] * linear +
                              parameters_[1] * quadratic + parameters_[2],
                          0.0};
    return log_value_;
  }

  void evaluateDerivatives(ParticleSet& particles,
                           const OptVariables&,
                           Vector<ValueType>& score,
                           Vector<ValueType>& kinetic) override
  {
    ++*derivative_calls_;
    const auto [linear, quadratic] = coordinateSums(particles);
    const std::array<ValueType, 3> score_values{
        makeValue(parameters_[0] * linear, 0.05 * quadratic),
        makeValue(quadratic - parameters_[1], -0.03 * linear),
        makeValue(1.0 + parameters_[2], 0.02 * linear)};
    const std::array<ValueType, 3> kinetic_values{
        makeValue(0.5 * quadratic + parameters_[0], -0.04 * linear),
        makeValue(-linear + 2.0 * parameters_[1], 0.01 * quadratic),
        makeValue(parameters_[0] - parameters_[1], -0.02 * quadratic)};
    for (std::size_t local = 0; local < 3; ++local)
    {
      const int global = myVars.where(static_cast<int>(local));
      score[global] += score_values[local];
      kinetic[global] += kinetic_values[local];
    }
  }

  void acceptMove(ParticleSet&, int, bool) override {}
  void restore(int) override {}
  PsiValue ratio(ParticleSet&, int) override { return PsiValue{1}; }
  void registerData(ParticleSet&, WFBufferType&) override {}
  LogValue updateBuffer(ParticleSet&, WFBufferType&, bool) override { return log_value_; }
  void copyFromBuffer(ParticleSet&, WFBufferType&) override {}

  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet&) const override
  {
    return std::make_unique<LegacyJastrowComponent>(parameters_, derivative_calls_);
  }

  static std::pair<double, double> coordinateSums(const ParticleSet& particles)
  {
    double linear = 0.0;
    double quadratic = 0.0;
    for (const auto& position : particles.R)
      for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
      {
        linear += position[dimension];
        quadratic += position[dimension] * position[dimension];
      }
    return {linear, quadratic};
  }

  static std::array<DerivativeValue, 3> scoreOracle(const ParticleSet& particles,
                                                    const std::vector<double>& parameters)
  {
    const auto [linear, quadratic] = coordinateSums(particles);
    return {derivativeValue(makeValue(parameters[0] * linear, 0.05 * quadratic)),
            derivativeValue(makeValue(quadratic - parameters[1], -0.03 * linear)),
            derivativeValue(makeValue(1.0 + parameters[2], 0.02 * linear))};
  }

  static std::array<DerivativeValue, 3> kineticOracle(const ParticleSet& particles,
                                                      const std::vector<double>& parameters)
  {
    const auto [linear, quadratic] = coordinateSums(particles);
    return {derivativeValue(makeValue(0.5 * quadratic + parameters[0], -0.04 * linear)),
            derivativeValue(makeValue(-linear + 2.0 * parameters[1], 0.01 * quadratic)),
            derivativeValue(makeValue(parameters[0] - parameters[1], -0.02 * quadratic))};
  }

private:
  std::array<double, 3> parameters_;
  std::shared_ptr<std::atomic<std::size_t>> derivative_calls_;
};

/// Build one conventional aggregate around the legacy component.
std::unique_ptr<TrialWaveFunction> makeWaveFunction(
    std::array<double, 3> parameters,
    const std::shared_ptr<std::atomic<std::size_t>>& derivative_calls,
    double disabled_parameter = 7.0)
{
  auto wavefunction = std::make_unique<TrialWaveFunction>(RuntimeOptions{}, "legacy_test");
  wavefunction->addComponent(
      std::make_unique<LegacyJastrowComponent>(parameters, derivative_calls, disabled_parameter));
  return wavefunction;
}

/// Build a two-electron unit-mass configuration suitable for legacy evaluation.
ParticleSet makeParticles(const SimulationCell& cell, double shift)
{
  ParticleSet particles(cell);
  particles.setName("e");
  particles.create({2});
  particles.R[0] = {0.2 + shift, -0.3, 0.4};
  particles.R[1] = {-0.5, 0.1 + shift, 0.7};
  SpeciesSet& species = particles.getSpeciesSet();
  const int electron  = species.addSpecies("u");
  const int mass      = species.addAttribute("mass");
  species(mass, electron) = 1.0;
  particles.resetGroups();
  particles.update();
  return particles;
}

/// Build a two-species electron set used by the production J2 smoke test.
ParticleSet makeTwoSpeciesParticles(const SimulationCell& cell)
{
  ParticleSet particles(cell);
  particles.setName("e");
  particles.create({1, 1});
  particles.R[0] = {0.2, -0.3, 0.4};
  particles.R[1] = {-0.5, 0.1, 0.7};
  SpeciesSet& species = particles.getSpeciesSet();
  const int up         = species.addSpecies("u");
  const int down       = species.addSpecies("d");
  const int mass       = species.addAttribute("mass");
  species(mass, up) = species(mass, down) = 1.0;
  particles.resetGroups();
  particles.update();
  return particles;
}

/// Construct a real QMCPACK two-body B-spline Jastrow with four active coefficients.
std::unique_ptr<TrialWaveFunction> makeBsplineJ2WaveFunction(
    ParticleSet& particles,
    const std::array<double, 4>& parameters)
{
  using Functor = BsplineFunctor<QMCTraits::RealType>;
  using J2      = TwoBodyJastrow<Functor>;

  auto functor = std::make_unique<Functor>("adapter_j2_functor");
  functor->cutoff_radius = 4.0;
  functor->resize(parameters.size());
  functor->Parameters.assign(parameters.begin(), parameters.end());
  for (std::size_t parameter = 0; parameter < parameters.size(); ++parameter)
    functor->myVars.insert("adapter_j2_" + std::to_string(parameter), parameters[parameter], true,
                           optimize::LOGLINEAR_P);
  functor->reset();

  auto jastrow = std::make_unique<J2>("adapter_j2", particles, false);
  jastrow->addFunc(0, 1, std::move(functor));
  auto wavefunction = std::make_unique<TrialWaveFunction>(RuntimeOptions{}, "legacy_j2_test");
  wavefunction->addComponent(std::move(jastrow));
  return wavefunction;
}

/// Required nonzero ceilings for the small deterministic fixture.
LegacyWaveFunctionAdapterOptions adapterOptions()
{
  return {16, 16 * 1024, 2};
}

/// Collect complete VJP channels into dense vectors only inside this test oracle.
class RecordingParameterSink final : public ParameterReductionSink
{
public:
  const std::vector<std::vector<DerivativeValue>>& result() const { return result_; }

protected:
  void onBegin(const DerivativeStreamDescriptor& descriptor,
               const ParameterChunkPlan& plan,
               DerivativeArrayView<const VJPCoefficientChannel> channels) override
  {
    (void)descriptor;
    result_.assign(channels.size(), std::vector<DerivativeValue>(plan.selectedParameterCount()));
  }

  void consume(std::size_t channel, const ParameterChunkConstView& chunk) override
  {
    std::copy(chunk.values().begin(), chunk.values().end(),
              result_[channel].begin() + chunk.descriptor().parameter_offset);
  }

  void onAbort() noexcept override { result_.clear(); }
  void onReset() noexcept override { result_.clear(); }

private:
  std::vector<std::vector<DerivativeValue>> result_;
};

/// Collect the one-element score-JVP tiles emitted by the adapter.
class RecordingSampleSink final : public SampleProductSink
{
public:
  const std::vector<DerivativeValue>& result() const { return result_; }

protected:
  void onBegin(const DerivativeStreamDescriptor& descriptor) override
  {
    first_sample_ = descriptor.sample_offset;
    result_.assign(descriptor.sample_count, {});
  }

  void consume(const SampleProductTileConstView& tile) override
  {
    result_[tile.descriptor().sample_offset - first_sample_] = tile.values()[0];
  }

  void onAbort() noexcept override { result_.clear(); }
  void onReset() noexcept override { result_.clear(); }

private:
  std::size_t first_sample_ = 0;
  std::vector<DerivativeValue> result_;
};

/// Tie coefficients to the immutable identity of one prepared derivative operator.
CoefficientView coefficientsFor(const StreamingDerivativeOperator& op,
                                const std::vector<DerivativeValue>& values)
{
  return {op.parameterSchema().providerId(), op.parameterSchema().fingerprint(), op.parameterVersion(),
          op.batchOrdinal(), op.sampleOffset(), {values.data(), values.size()}};
}

/// Check both contraction components with a tolerance appropriate to direct doubles.
void checkClose(DerivativeValue actual, DerivativeValue expected)
{
  CHECK(actual.real() == Catch::Approx(expected.real()).epsilon(2e-12).margin(2e-12));
  CHECK(actual.imag() == Catch::Approx(expected.imag()).epsilon(2e-12).margin(2e-12));
}

/// Remove a checkpoint fixture even if a test assertion throws.
class FileCleanup
{
public:
  explicit FileCleanup(std::filesystem::path path) : path_(std::move(path)) {}
  ~FileCleanup()
  {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

private:
  std::filesystem::path path_;
};

/** Rebuild disposable evaluators for every objective transaction.
 *
 * This mimics a conventional sampler rebinding each newly published detached
 * provider snapshot and deliberately retains no dense derivative matrix.
 */
class LegacyGradientProducer final : public GradientProducer
{
public:
  LegacyGradientProducer(LegacyWaveFunctionAdapter& adapter,
                         ParticleSet& first,
                         ParticleSet& second,
                         std::shared_ptr<std::atomic<std::size_t>> calls)
      : adapter_(adapter), first_(first), second_(second), calls_(std::move(calls))
  {}

  TrainingCapabilities capabilities() const noexcept override
  {
    return {TrainingCapability::REAL_PARAMETERS, TrainingCapability::VALUE_EVALUATION,
            TrainingCapability::SPATIAL_DERIVATIVES, TrainingCapability::SCORE_VJP,
            TrainingCapability::SCORE_JVP, TrainingCapability::LOCAL_ENERGY_VJP};
  }

  void accumulate(const StructuredParameterSnapshot& parameters,
                  EnergyGradientAccumulator& accumulator) override
  {
    auto first_wf  = makeWaveFunction({9.0, 9.0, 9.0}, calls_);
    auto second_wf = makeWaveFunction({8.0, 8.0, 8.0}, calls_);
    RefVector<TrialWaveFunction> evaluators{*first_wf, *second_wf};
    RefVector<ParticleSet> particles{first_, second_};
    auto derivative_operator = adapter_.makeDerivativeOperator(evaluators, particles, 17, 0);
    observed_versions.push_back(derivative_operator->parameterVersion());
    REQUIRE(derivative_operator->parameterVersion() == parameters.version);
    const std::array<DerivativeReal, 2> weights{1.0, 1.0};
    const std::array<DerivativeValue, 2> energies{DerivativeValue{-1.2, 0.0},
                                                  DerivativeValue{-0.7, 0.0}};
    accumulateEnergyGradientBatch(*derivative_operator, {weights.data(), weights.size()},
                                  {energies.data(), energies.size()},
                                  localEnergyTermBit(LocalEnergyTerm::KINETIC), accumulator);
  }

  std::vector<std::size_t> observed_versions;

private:
  LegacyWaveFunctionAdapter& adapter_;
  ParticleSet& first_;
  ParticleSet& second_;
  std::shared_ptr<std::atomic<std::size_t>> calls_;
};

} // namespace

TEST_CASE("Legacy adapter publishes detached structured state atomically",
          "[drivers][training][legacy_adapter]")
{
  auto calls  = std::make_shared<std::atomic<std::size_t>>(0);
  auto source = makeWaveFunction({0.4, -0.3, 0.2}, calls);
  LegacyWaveFunctionAdapter adapter("legacy/conventional", *source, adapterOptions());

  CHECK(adapter.parameterSchema().parameterCount() == 3);
  REQUIRE(adapter.parameterSchema().blocks().size() == 3);
  CHECK(adapter.parameterSchema().blocks()[0].update_group == "legacy_other");
  CHECK(adapter.parameterSchema().blocks()[1].update_group == "legacy_linear");
  CHECK(adapter.parameterSchema().blocks()[2].update_group == "legacy_other");

  const StructuredParameterSnapshot before = adapter.snapshotParameters();
  StructuredParameterSnapshot candidate = before;
  candidate.values = {0.6, -0.1, 0.25};
  CHECK(adapter.publishParameters(candidate, before.version) == 1);
  CHECK(adapter.snapshotParameters().values == candidate.values);
  CHECK(*calls == 0);

  StructuredParameterSnapshot invalid = adapter.snapshotParameters();
  invalid.values[1] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(adapter.publishParameters(invalid, invalid.version));
  CHECK(adapter.snapshotParameters().values == candidate.values);
  CHECK_THROWS(adapter.publishParameters(candidate, before.version));
  CHECK(adapter.snapshotParameters().values == candidate.values);
}

TEST_CASE("Legacy adapter VJPs and JVP match direct conventional derivative rows",
          "[drivers][training][legacy_adapter]")
{
  const SimulationCell cell;
  ParticleSet first  = makeParticles(cell, 0.0);
  ParticleSet second = makeParticles(cell, 0.15);
  auto calls  = std::make_shared<std::atomic<std::size_t>>(0);
  auto source = makeWaveFunction({0.4, -0.3, 0.2}, calls);
  auto first_wf  = makeWaveFunction({5.0, 5.0, 5.0}, calls);
  auto second_wf = makeWaveFunction({6.0, 6.0, 6.0}, calls);
  LegacyWaveFunctionAdapter adapter("legacy/conventional", *source, adapterOptions());
  const StructuredParameterSnapshot parameters = adapter.snapshotParameters();
  RefVector<TrialWaveFunction> evaluators{*first_wf, *second_wf};
  RefVector<ParticleSet> particles{first, second};
  auto op = adapter.makeDerivativeOperator(evaluators, particles, 4, 9);

  const auto score0   = LegacyJastrowComponent::scoreOracle(first, parameters.values);
  const auto score1   = LegacyJastrowComponent::scoreOracle(second, parameters.values);
  const auto kinetic0 = LegacyJastrowComponent::kineticOracle(first, parameters.values);
  const auto kinetic1 = LegacyJastrowComponent::kineticOracle(second, parameters.values);
  const std::vector<DerivativeValue> score_coefficients{{1.2, -0.3}, {-0.5, 0.4}};
  const std::vector<DerivativeValue> kinetic_coefficients{{0.2, 0.1}, {0.7, -0.2}};
  const std::array<VJPCoefficientChannel, 2> channels{{
      {"score", DerivativeProduct::SCORE_VJP, coefficientsFor(*op, score_coefficients), 0},
      {"kinetic", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, kinetic_coefficients),
       localEnergyTermBit(LocalEnergyTerm::KINETIC)}}};

  for (DerivativeAdjoint adjoint : {DerivativeAdjoint::TRANSPOSE, DerivativeAdjoint::HERMITIAN})
  {
    RecordingParameterSink sink;
    op->applyVJPs({channels.data(), channels.size()}, adjoint, sink);
    REQUIRE(sink.result().size() == 2);
    for (std::size_t parameter = 0; parameter < 3; ++parameter)
    {
      const auto adjust = [adjoint](DerivativeValue value) {
        return adjoint == DerivativeAdjoint::HERMITIAN ? std::conj(value) : value;
      };
      checkClose(sink.result()[0][parameter], adjust(score0[parameter]) * score_coefficients[0] +
                     adjust(score1[parameter]) * score_coefficients[1]);
      checkClose(sink.result()[1][parameter], adjust(kinetic0[parameter]) * kinetic_coefficients[0] +
                     adjust(kinetic1[parameter]) * kinetic_coefficients[1]);
    }
  }

  const std::vector<DerivativeValue> direction{0.5, -0.2, 0.7};
  StructuredParameterVectorConstView direction_view(
      adapter.parameterSchema(), op->parameterVersion(), {direction.data(), direction.size()});
  RecordingSampleSink jvp_sink;
  op->applyScoreJVP(direction_view, jvp_sink);
  REQUIRE(jvp_sink.result().size() == 2);
  for (std::size_t sample = 0; sample < 2; ++sample)
  {
    const auto& row = sample == 0 ? score0 : score1;
    DerivativeValue expected{};
    for (std::size_t parameter = 0; parameter < 3; ++parameter)
      expected += row[parameter] * direction[parameter];
    checkClose(jvp_sink.result()[sample], expected);
  }

  // Validate the high-level energy objective against a dense test-only formula.
  const std::array<DerivativeReal, 2> weights{1.0, 2.0};
  const std::array<DerivativeValue, 2> energies{DerivativeValue{-1.2, 0.0},
                                                DerivativeValue{-0.7, 0.0}};
  EnergyGradientAccumulator accumulator(adapter.parameterSchema(), op->parameterVersion());
  accumulateEnergyGradientBatch(*op, {weights.data(), weights.size()}, {energies.data(), energies.size()},
                                localEnergyTermBit(LocalEnergyTerm::KINETIC), accumulator);
  accumulator.completeSingleParticipantReduction();
  const EnergyGradientResult objective = accumulator.finalize();
  const DerivativeValue mean_energy = (weights[0] * energies[0] + weights[1] * energies[1]) /
      (weights[0] + weights[1]);
  for (std::size_t parameter = 0; parameter < 3; ++parameter)
  {
    const DerivativeValue weighted_score =
        weights[0] * score0[parameter] + weights[1] * score1[parameter];
    const DerivativeValue weighted_energy_score = weights[0] * energies[0] * score0[parameter] +
        weights[1] * energies[1] * score1[parameter];
    const DerivativeValue weighted_kinetic =
        weights[0] * kinetic0[parameter] + weights[1] * kinetic1[parameter];
    const double expected = 2.0 * std::real(weighted_kinetic + weighted_energy_score -
                                            mean_energy * weighted_score) /
        (weights[0] + weights[1]);
    CHECK(objective.gradient[parameter] == Catch::Approx(expected).epsilon(2e-12).margin(2e-12));
  }

  const StreamingDerivativeStorageDiagnostics storage = op->storageDiagnostics();
  CHECK(storage.parameter_count == 3);
  CHECK(storage.sample_count == 2);
  CHECK(storage.retained_numeric_bytes <= adapterOptions().maximum_derivative_scratch_bytes);
}

TEST_CASE("Legacy adapter capability and resource gates precede derivative evaluation",
          "[drivers][training][legacy_adapter]")
{
  const SimulationCell cell;
  ParticleSet particles = makeParticles(cell, 0.0);
  auto calls  = std::make_shared<std::atomic<std::size_t>>(0);
  auto source = makeWaveFunction({0.4, -0.3, 0.2}, calls);
  auto evaluator = makeWaveFunction({4.0, 4.0, 4.0}, calls);
  LegacyWaveFunctionAdapter adapter("legacy/gated", *source, adapterOptions());
  RefVector<TrialWaveFunction> evaluators{*evaluator};
  RefVector<ParticleSet> particle_batch{particles};
  auto op = adapter.makeDerivativeOperator(evaluators, particle_batch, 0, 0);
  const std::vector<DerivativeValue> coefficients{1.0};
  const VJPCoefficientChannel unsupported{
      "nonlocal", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, coefficients),
      localEnergyTermBit(LocalEnergyTerm::KINETIC) | localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)};
  RecordingParameterSink sink;
  const std::size_t before_calls = *calls;
  CHECK_THROWS_WITH(op->applyVJPs({&unsupported, 1}, DerivativeAdjoint::TRANSPOSE, sink),
                    Catch::Matchers::ContainsSubstring("coverage"));
  CHECK(*calls == before_calls);

  LegacyWaveFunctionAdapterOptions too_small = adapterOptions();
  too_small.maximum_parameter_count = 2;
  CHECK_THROWS_AS(LegacyWaveFunctionAdapter("legacy/too_many", *source, too_small), std::length_error);
  too_small = adapterOptions();
  too_small.maximum_derivative_scratch_bytes = 1;
  CHECK_THROWS_AS(LegacyWaveFunctionAdapter("legacy/too_large", *source, too_small), std::length_error);

  ParticleSet heavy = makeParticles(cell, 0.1);
  SpeciesSet& heavy_species = heavy.getSpeciesSet();
  const int heavy_mass      = heavy_species.addAttribute("mass");
  heavy_species(heavy_mass, 0) = 2.0;
  heavy.resetGroups();
  auto heavy_evaluator = makeWaveFunction({1.0, 1.0, 1.0}, calls);
  RefVector<TrialWaveFunction> heavy_evaluators{*heavy_evaluator};
  RefVector<ParticleSet> heavy_particles{heavy};
  CHECK_THROWS_WITH(adapter.makeDerivativeOperator(heavy_evaluators, heavy_particles, 0, 0),
                    Catch::Matchers::ContainsSubstring("unit electron masses"));
  CHECK(*calls == before_calls);

  auto mismatched_evaluator = makeWaveFunction({1.0, 1.0, 1.0}, calls, 8.0);
  RefVector<TrialWaveFunction> mismatched_evaluators{*mismatched_evaluator};
  CHECK_THROWS_WITH(adapter.makeDerivativeOperator(mismatched_evaluators, particle_batch, 0, 0),
                    Catch::Matchers::ContainsSubstring("registration does not match"));
  CHECK(*calls == before_calls);
}

TEST_CASE("Legacy adapter binds a production two-body B-spline Jastrow",
          "[drivers][training][legacy_adapter][jastrow]")
{
  const SimulationCell cell;
  ParticleSet particles = makeTwoSpeciesParticles(cell);
  auto source = makeBsplineJ2WaveFunction(particles, {0.12, -0.08, 0.04, -0.01});
  auto evaluator = makeBsplineJ2WaveFunction(particles, {0.7, 0.6, 0.5, 0.4});
  // The production J2 constructors add distance tables to their target set.
  particles.update();
  LegacyWaveFunctionAdapter adapter("legacy/production_j2", *source, adapterOptions());
  REQUIRE(adapter.parameterSchema().parameterCount() == 4);
  REQUIRE(adapter.parameterSchema().blocks().size() == 1);
  CHECK(adapter.parameterSchema().blocks()[0].update_group == "legacy_loglinear");

  StructuredParameterSnapshot candidate = adapter.snapshotParameters();
  candidate.values = {0.15, -0.06, 0.03, -0.02};
  adapter.publishParameters(candidate, candidate.version);
  const StructuredParameterSnapshot published = adapter.snapshotParameters();

  RefVector<TrialWaveFunction> evaluators{*evaluator};
  RefVector<ParticleSet> particle_batch{particles};
  auto op = adapter.makeDerivativeOperator(evaluators, particle_batch, 3, 11);

  // Evaluate the established scalar route directly after the adapter has installed
  // the detached snapshot, then compare it with unit-coefficient streaming VJPs.
  optimize::VariableSet active;
  evaluator->checkInVariables(active);
  active.resetIndex();
  evaluator->checkOutVariables(active);
  Vector<QMCTraits::ValueType> direct_score(4);
  Vector<QMCTraits::ValueType> direct_kinetic(4);
  direct_score   = QMCTraits::ValueType{};
  direct_kinetic = QMCTraits::ValueType{};
  evaluator->evaluateDerivatives(particles, active, direct_score, direct_kinetic);

  const std::vector<DerivativeValue> coefficient{1.0};
  const std::array<VJPCoefficientChannel, 2> channels{{
      {"score", DerivativeProduct::SCORE_VJP, coefficientsFor(*op, coefficient), 0},
      {"kinetic", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, coefficient),
       localEnergyTermBit(LocalEnergyTerm::KINETIC)}}};
  RecordingParameterSink sink;
  op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::TRANSPOSE, sink);
  REQUIRE(sink.result().size() == 2);
  bool observed_nonzero_score = false;
  for (std::size_t parameter = 0; parameter < published.values.size(); ++parameter)
  {
    checkClose(sink.result()[0][parameter], derivativeValue(direct_score[parameter]));
    checkClose(sink.result()[1][parameter], derivativeValue(direct_kinetic[parameter]));
    observed_nonzero_score |= std::abs(sink.result()[0][parameter]) > 1e-12;
    CHECK(active[static_cast<int>(parameter)] == Catch::Approx(published.values[parameter]));
  }
  CHECK(observed_nonzero_score);
}

TEST_CASE("Legacy adapter supports repeated high-parameter updates and checkpoint restore",
          "[drivers][training][legacy_adapter][checkpoint]")
{
  const SimulationCell cell;
  ParticleSet first  = makeParticles(cell, 0.0);
  ParticleSet second = makeParticles(cell, 0.12);
  auto calls = std::make_shared<std::atomic<std::size_t>>(0);
  auto source = makeWaveFunction({0.4, -0.3, 0.2}, calls);
  LegacyWaveFunctionAdapter adapter("legacy/trainable", *source, adapterOptions());
  LegacyGradientProducer producer(adapter, first, second, calls);

  FirstOrderOptimizerOptions optimizer_options;
  optimizer_options.method = FirstOrderMethod::SGD;
  optimizer_options.learning_rates = {{"legacy_other", 0.01}, {"legacy_linear", 0.02}};
  FirstOrderOptimizer optimizer(adapter.parameterSchema(), optimizer_options);
  HighParameterTraining training({TrainingCapability::VALUE_EVALUATION,
                                  TrainingCapability::SPATIAL_DERIVATIVES});
  TrainingIterationState state;
  const StructuredParameterSnapshot initial = adapter.snapshotParameters();
  training.runIteration(adapter, producer, optimizer, state);
  training.runIteration(adapter, producer, optimizer, state);
  CHECK(state.completed_iterations == 2);
  CHECK(state.parameter_version == 2);
  CHECK(producer.observed_versions == std::vector<std::size_t>{0, 1});
  CHECK(adapter.snapshotParameters().values != initial.values);

  const std::filesystem::path checkpoint = "wftrain_legacy_adapter.h5";
  FileCleanup cleanup(checkpoint);
  TrainingStageState stage{"optimization", 1, 2, "legacy-adapter-stage-v1"};
  TrainingCheckpoint::saveAtomic(checkpoint, adapter, state, stage, &optimizer);

  auto restored_source = makeWaveFunction({9.0, 8.0, 7.0}, calls);
  LegacyWaveFunctionAdapter restored("legacy/trainable", *restored_source, adapterOptions());
  FirstOrderOptimizer restored_optimizer(restored.parameterSchema(), optimizer_options);
  TrainingIterationState restored_state;
  TrainingStageState restored_stage{"optimization", 0, 0, "legacy-adapter-stage-v1"};
  TrainingCheckpoint::restore(checkpoint, restored, restored_state, restored_stage, &restored_optimizer);
  CHECK(restored.snapshotParameters().values == adapter.snapshotParameters().values);
  CHECK(restored_state.completed_iterations == state.completed_iterations);
  CHECK(restored_state.parameter_version == restored.snapshotParameters().version);
  CHECK(restored_stage.completed_stage_iterations == 2);
  CHECK(restored_optimizer.acceptedUpdateCount() == optimizer.acceptedUpdateCount());
}

} // namespace qmcplusplus::wftrain
