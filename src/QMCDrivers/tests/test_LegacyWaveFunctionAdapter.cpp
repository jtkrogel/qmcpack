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
#include "Particle/ParticleSetPool.h"
#include "OhmmsData/Libxml2Doc.h"
#include "QMCDrivers/WFTrain/EnergyGradientAccumulator.h"
#include "QMCDrivers/WFTrain/FirstOrderOptimizer.h"
#include "QMCDrivers/WFTrain/HighParameterTraining.h"
#include "QMCDrivers/WFTrain/TrainingCheckpoint.h"
#include "QMCDrivers/WFTrain/TrainingWaveFunctionBinding.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCWaveFunctions/Optimization/LegacyWaveFunctionAdapter.h"
#include "QMCWaveFunctions/Jastrow/BsplineFunctor.h"
#include "QMCWaveFunctions/Jastrow/TwoBodyJastrow.h"
#include "QMCWaveFunctions/Jastrow/eeI_JastrowBuilder.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/WaveFunctionFactory.h"
#include "QMCWaveFunctions/Fermion/MultiSlaterDetTableMethod.h"
#include "QMCWaveFunctions/Fermion/SlaterDet.h"
#include "QMCWaveFunctions/RotatedSPOs.h"
#include "QMCWaveFunctions/LCAO/LCAOrbitalSet.h"
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

/// Fixed Gaussian supplies nonzero total-gradient cross terms to production J2.
class FixedGaussianComponent final : public WaveFunctionComponent
{
public:
  explicit FixedGaussianComponent(double strength) : WaveFunctionComponent("fixed_gaussian"), strength_(strength) {}
  std::string getClassName() const override { return "FixedGaussianComponent"; }
  void evaluateDerivatives(ParticleSet&, const OptVariables&, Vector<ValueType>&, Vector<ValueType>&) override {}
  LogValue evaluateLog(const ParticleSet& particles, ParticleSet::ParticleGradient& gradients,
                       ParticleSet::ParticleLaplacian& laplacians) override
  {
    double value = 0.0;
    for (int electron = 0; electron < particles.getTotalNum(); ++electron)
    {
      for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
      {
        const double coordinate = particles.R[electron][dimension];
        value -= strength_ * coordinate * coordinate;
        gradients[electron][dimension] -= 2.0 * strength_ * coordinate;
      }
      laplacians[electron] -= 2.0 * strength_ * OHMMS_DIM;
    }
    log_value_ = LogValue{value, 0.0};
    return log_value_;
  }
  void acceptMove(ParticleSet&, int, bool) override {}
  void restore(int) override {}
  PsiValue ratio(ParticleSet&, int) override { return PsiValue{1}; }
  void registerData(ParticleSet&, WFBufferType&) override {}
  LogValue updateBuffer(ParticleSet&, WFBufferType&, bool) override { return log_value_; }
  void copyFromBuffer(ParticleSet&, WFBufferType&) override {}
  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet&) const override
  {
    return std::make_unique<FixedGaussianComponent>(strength_);
  }
private:
  double strength_;
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
    const std::array<double, 4>& parameters,
    double fixed_gaussian = 0.0)
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
  if (fixed_gaussian != 0.0)
    wavefunction->addComponent(std::make_unique<FixedGaussianComponent>(fixed_gaussian));
  wavefunction->addComponent(std::move(jastrow));
  return wavefunction;
}

DerivativeValue kineticEnergy(const ParticleSet& particles)
{
  DerivativeValue energy{};
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
  {
    energy -= 0.5 * derivativeValue(particles.L[electron]);
    for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
    {
      const auto gradient = derivativeValue(particles.G[electron][dimension]);
      energy -= 0.5 * gradient * gradient;
    }
  }
  return energy;
}

/// Reuse a single private evaluator for any number of frozen sample views.
class ShiftedLegacyReplay final : public LegacyWaveFunctionReplay
{
public:
  ShiftedLegacyReplay(TrialWaveFunction& wavefunction, ParticleSet& particles, std::vector<double> shifts)
      : wavefunction_(wavefunction), particles_(particles), shifts_(std::move(shifts))
  {}
  std::size_t sampleCount() const noexcept override { return shifts_.size(); }
  LegacyWaveFunctionReplaySample prepareSample(std::size_t index) override
  {
    particles_.R[0] = {0.2 + shifts_.at(index), -0.3, 0.4};
    particles_.R[1] = {-0.5, 0.1 + shifts_.at(index), 0.7};
    particles_.update();
    return {wavefunction_, particles_};
  }
private:
  TrialWaveFunction& wavefunction_;
  ParticleSet& particles_;
  std::vector<double> shifts_;
};

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
  auto prepared = adapter.snapshotParameters();
  prepared.values = {0.8, -0.2, 0.3};
  const auto accepted_values = prepared.values;
  CHECK(adapter.commitPreparedParameters(std::move(prepared)) == 2);
  CHECK(adapter.snapshotParameters().values == accepted_values);
  CHECK(prepared.values == candidate.values);
  CHECK(*calls == 0);
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
  RecordingSampleSink kinetic_jvp;
  op->applyLocalEnergyJVP(direction_view, localEnergyTermBit(LocalEnergyTerm::KINETIC), kinetic_jvp);
  REQUIRE(kinetic_jvp.result().size() == 2);
  for (std::size_t sample = 0; sample < 2; ++sample)
  {
    const auto& row = sample == 0 ? kinetic0 : kinetic1;
    DerivativeValue expected{};
    for (std::size_t parameter = 0; parameter < 3; ++parameter)
      expected += row[parameter] * direction[parameter];
    checkClose(kinetic_jvp.result()[sample], expected);
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

TEST_CASE("Legacy production J2 products match energy finite differences and duality",
          "[drivers][training][legacy_adapter][jastrow]")
{
  const SimulationCell cell;
  auto particles = makeTwoSpeciesParticles(cell);
  auto wavefunction = makeBsplineJ2WaveFunction(particles, {0.12, -0.08, 0.04, -0.01}, 0.37);
  particles.update();
  LegacyWaveFunctionAdapter adapter("legacy/j2_fd", *wavefunction, adapterOptions());
  const auto snapshot = adapter.snapshotParameters();
  auto op = adapter.makeDerivativeOperator(RefVector<TrialWaveFunction>{*wavefunction},
                                            RefVector<ParticleSet>{particles}, 4, 7);
  const std::vector<DerivativeValue> coefficients{{0.7, -0.2}};
  const std::array<VJPCoefficientChannel, 2> channels{{
      {"score", DerivativeProduct::SCORE_VJP, coefficientsFor(*op, coefficients), 0},
      {"energy", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, coefficients),
       localEnergyTermBit(LocalEnergyTerm::KINETIC)}}};
  RecordingParameterSink transpose;
  op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::TRANSPOSE, transpose);
  std::vector<DerivativeValue> direction{0.3, -0.2, 0.5, 0.1};
  StructuredParameterVectorConstView view(adapter.parameterSchema(), snapshot.version,
                                           {direction.data(), direction.size()});
  RecordingSampleSink score, energy;
  op->applyScoreJVP(view, score);
  op->applyLocalEnergyJVP(view, localEnergyTermBit(LocalEnergyTerm::KINETIC), energy);
  for (std::size_t channel = 0; channel < channels.size(); ++channel)
  {
    DerivativeValue contraction{};
    for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
      contraction += transpose.result()[channel][parameter] * direction[parameter];
    checkClose(contraction, coefficients[0] * (channel == 0 ? score.result()[0] : energy.result()[0]));
  }
  RecordingParameterSink hermitian;
  op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::HERMITIAN, hermitian);
  for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
  {
    const double epsilon = 1e-5;
    auto plus = snapshot, minus = snapshot;
    plus.values[parameter] += epsilon;
    minus.values[parameter] -= epsilon;
    adapter.installSnapshot(*wavefunction, particles, plus);
    const double plus_log = std::real(wavefunction->getLogPsi());
    const auto plus_energy = kineticEnergy(particles);
    adapter.installSnapshot(*wavefunction, particles, minus);
    const double minus_log = std::real(wavefunction->getLogPsi());
    const auto minus_energy = kineticEnergy(particles);
    const auto score_fd = (plus_log - minus_log) / (2.0 * epsilon);
    const auto energy_fd = (plus_energy - minus_energy) / (2.0 * epsilon);
    for (std::size_t channel = 0; channel < channels.size(); ++channel)
    {
      const auto finite_difference = channel == 0 ? DerivativeValue{score_fd} : energy_fd;
      const auto actual = transpose.result()[channel][parameter] / coefficients[0];
      CHECK(actual.real() == Catch::Approx(finite_difference.real()).epsilon(2e-7).margin(2e-9));
      CHECK(actual.imag() == Catch::Approx(finite_difference.imag()).margin(2e-9));
      checkClose(hermitian.result()[channel][parameter], std::conj(actual) * coefficients[0]);
    }
  }
  // Repeated actions reinstall their immutable snapshot after arbitrary private probes.
  RecordingSampleSink repeated;
  op->applyLocalEnergyJVP(view, localEnergyTermBit(LocalEnergyTerm::KINETIC), repeated);
  checkClose(repeated.result()[0], energy.result()[0]);
}

TEST_CASE("Legacy replay block views share bounded parameter workspace and immutable state",
          "[drivers][training][legacy_adapter][memory]")
{
  const SimulationCell cell;
  auto particles = makeParticles(cell, 0.0);
  auto calls = std::make_shared<std::atomic<std::size_t>>(0);
  auto wavefunction = makeWaveFunction({0.2, -0.1, 0.05}, calls);
  LegacyWaveFunctionAdapter adapter("legacy/replay", *wavefunction, adapterOptions());
  auto snapshot = std::make_shared<const StructuredParameterSnapshot>(adapter.snapshotParameters());
  auto replay = std::make_shared<ShiftedLegacyReplay>(*wavefunction, particles,
                                                      std::vector<double>{0.0, 0.2, -0.1});
  std::vector<std::unique_ptr<StreamingDerivativeOperator>> blocks;
  for (std::size_t block = 0; block < 12; ++block)
    blocks.push_back(adapter.makeDerivativeOperator(replay, snapshot, block, 3 * block));
  REQUIRE(snapshot.use_count() == 13);
  const auto first = blocks.front()->storageDiagnostics();
  CHECK(first.shared_workspace_identity != 0);
  CHECK(first.shared_workspace_bytes == first.parameter_scratch_bytes);
  for (const auto& block : blocks)
  {
    const auto storage = block->storageDiagnostics();
    CHECK(storage.shared_workspace_identity == first.shared_workspace_identity);
    CHECK(storage.storage_fingerprint == first.storage_fingerprint);
    CHECK(storage.parameter_scratch_bytes == first.parameter_scratch_bytes);
  }
  auto candidate = *snapshot;
  candidate.values = {0.7, 0.6, 0.5};
  adapter.publishParameters(candidate, candidate.version);
  const std::vector<DerivativeValue> direction{0.2, -0.4, 0.7};
  StructuredParameterVectorConstView view(adapter.parameterSchema(), snapshot->version,
                                           {direction.data(), direction.size()});
  for (const auto& block : blocks)
  {
    RecordingSampleSink sink;
    block->applyLocalEnergyJVP(view, localEnergyTermBit(LocalEnergyTerm::KINETIC), sink);
    REQUIRE(sink.result().size() == replay->sampleCount());
    for (std::size_t sample = 0; sample < replay->sampleCount(); ++sample)
    {
      replay->prepareSample(sample);
      const auto row = LegacyJastrowComponent::kineticOracle(particles, snapshot->values);
      DerivativeValue expected{};
      for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
        expected += row[parameter] * direction[parameter];
      checkClose(sink.result()[sample], expected);
    }
  }
  auto invalid = std::make_shared<StructuredParameterSnapshot>(*snapshot);
  invalid->values[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_AS(adapter.makeDerivativeOperator(replay, invalid, 0, 0), std::invalid_argument);
  CHECK_THROWS_AS(adapter.makeDerivativeOperator({}, snapshot, 0, 0), std::invalid_argument);
}

TEST_CASE("Legacy orbital products preserve cumulative private rotation history",
          "[drivers][training][legacy_adapter][orbital_rotation]")
{
#ifdef QMC_COMPLEX
  SKIP("This gate qualifies real CPU orbital coordinates");
#endif
  ParticleSetPool pool(OHMMS::Controller);
  auto ion_owner = std::make_unique<ParticleSet>(pool.getSimulationCell());
  auto electron_owner = std::make_unique<ParticleSet>(pool.getSimulationCell());
  auto& ions = *ion_owner;
  auto& particles = *electron_owner;
  ions.setName("ion0");
  ions.create({1});
  ions.R[0] = {0, 0, 0};
  auto& ion_species = ions.getSpeciesSet();
  ion_species.addSpecies("He");
  ions.resetGroups();
  particles.setName("e");
  particles.create({1, 1});
  particles.R[0] = {0.7, 0.2, 0.4};
  particles.R[1] = {-0.5, 0.3, 0.8};
  auto& species = particles.getSpeciesSet();
  const int up = species.addSpecies("u"), down = species.addSpecies("d");
  const int mass = species.addAttribute("mass");
  species(mass, up) = species(mass, down) = 1.0;
  particles.resetGroups();
  pool.addParticleSet(std::move(ion_owner));
  pool.addParticleSet(std::move(electron_owner));
  Libxml2Document definition;
  // Extend the existing test_RotatedSPOs_LCAO He oracle to three STOs, so the
  // two occupied-virtual rotations do not commute. Both spins alias one SPO.
  REQUIRE(definition.parseFromString(R"(
    <wavefunction name="psi0" target="e">
      <sposet_collection type="MolecularOrbital" source="ion0">
        <basisset keyword="STO" transform="no">
          <atomicBasisSet type="STO" elementType="He" normalized="no">
            <basisGroup rid="R0" l="0" m="0" type="Slater"><radfunc n="1" exponent="2.0"/></basisGroup>
            <basisGroup rid="R1" l="0" m="0" type="Slater"><radfunc n="2" exponent="1.0"/></basisGroup>
            <basisGroup rid="R2" l="0" m="0" type="Slater"><radfunc n="3" exponent="0.7"/></basisGroup>
          </atomicBasisSet>
        </basisset>
        <rotated_sposet name="rot">
          <sposet name="base" size="3"><coefficient size="3" id="identity">1 0 0  0 1 0  0 0 1</coefficient></sposet>
        </rotated_sposet>
      </sposet_collection>
      <determinantset>
        <slaterdeterminant><determinant sposet="rot"/><determinant sposet="rot"/></slaterdeterminant>
      </determinantset>
      <jastrow name="J2" type="Two-Body" function="Bspline">
        <correlation rcut="4" size="4" speciesA="u" speciesB="d">
          <coefficients id="ud" type="Array" optimize="yes">0.12 -0.08 0.04 -0.01</coefficients>
        </correlation>
      </jastrow>
    </wavefunction>)"));
  WaveFunctionFactory factory(particles, pool.getPool(), OHMMS::Controller);
  auto source = factory.buildTWF(definition.getRoot(), RuntimeOptions{});
  REQUIRE(source);
  ions.update();
  particles.update();
  source->evaluateLog(particles);
  const auto rotation = [](TrialWaveFunction& wavefunction) -> RotatedSPOs& {
    auto& slater = wavefunction.findSD()[0].get();
    REQUIRE(&slater.getPhi(0) == &slater.getPhi(1));
    return dynamic_cast<RotatedSPOs&>(slater.getPhi(0));
  };
  const auto lcao = [](RotatedSPOs& orbital) -> LCAOrbitalSet& {
    return dynamic_cast<LCAOrbitalSet&>(*orbital.Phi_);
  };
  auto& live_rotation = rotation(*source);
  auto& live_lcao = lcao(live_rotation);
  auto* fixed_phi = live_rotation.Phi_.get();
  auto* fixed_basis = live_lcao.myBasisSet.get();
  ParticleSet sibling_particles(particles);
  auto sibling = source->makeClone(sibling_particles);
  sibling_particles.update();
  auto& sibling_rotation = rotation(*sibling);
  CHECK(lcao(sibling_rotation).C == live_lcao.C);
  const LCAOrbitalSet::OffloadValueMatrix original(live_lcao.originalRotationCoefficients());
  LegacyWaveFunctionAdapter adapter("legacy/orbital", *source, adapterOptions());
  REQUIRE(adapter.snapshotParameters().values.size() == 6);
  const auto make_private = [&](TrialWaveFunction& base, ParticleSet& positions,
                                 const StructuredParameterSnapshot& snapshot) {
    auto private_wavefunction = base.makeClone(positions);
    positions.update();
    rotation(*private_wavefunction).isolateOptimizationState();
    adapter.installSnapshot(*private_wavefunction, positions, snapshot);
    return private_wavefunction;
  };
  for (int step = 0; step < 2; ++step)
  {
    auto next = adapter.snapshotParameters();
    next.values[step] += step == 0 ? 0.08 : -0.06;
    ParticleSet private_particles(particles);
    auto prepared = make_private(*source, private_particles, next);
    auto& prepared_rotation = rotation(*prepared);
    CHECK(lcao(prepared_rotation).C != live_lcao.C);
    REQUIRE_NOTHROW(live_rotation.validatePreparedRotationState(prepared_rotation));
    live_rotation.adoptPreparedRotationState(prepared_rotation, prepared_rotation);
    CHECK(live_rotation.Phi_.get() == fixed_phi);
    CHECK(live_lcao.myBasisSet.get() == fixed_basis);
    CHECK(live_lcao.C != lcao(sibling_rotation).C);
    CHECK(adapter.commitPreparedParameters(std::move(next)) == step + 1);
  }
  const auto accepted = adapter.snapshotParameters();
  const auto full = live_rotation.fullRotationCoordinates();
  REQUIRE(full.size() == 3);
  CHECK(std::abs(full[2]) > 1e-4); // virtual-virtual commutator, absent from theta
  const LCAOrbitalSet::OffloadValueMatrix accepted_coefficients(*live_lcao.C);
  for (std::size_t i = 0; i < original.size(); ++i)
    CHECK(lcao(sibling_rotation).C->data()[i] == original.data()[i]);

  ParticleSet evaluator_particles(particles);
  auto evaluator = make_private(*source, evaluator_particles, accepted);
  auto& evaluator_rotation = rotation(*evaluator);
  auto& evaluator_lcao = lcao(evaluator_rotation);
  auto op = adapter.makeDerivativeOperator(RefVector<TrialWaveFunction>{*evaluator},
                                            RefVector<ParticleSet>{evaluator_particles}, 0, 0);
  const std::vector<DerivativeValue> one{1.0};
  const std::array<VJPCoefficientChannel, 2> channels{{
      {"score", DerivativeProduct::SCORE_VJP, coefficientsFor(*op, one), 0},
      {"energy", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, one),
       localEnergyTermBit(LocalEnergyTerm::KINETIC)}}};
  RecordingParameterSink rows;
  op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::TRANSPOSE, rows);
  for (std::size_t parameter = 0; parameter < accepted.values.size(); ++parameter)
  {
    CAPTURE(parameter);
    constexpr double epsilon = 1e-5;
    auto plus = accepted, minus = accepted;
    plus.values[parameter] += epsilon;
    minus.values[parameter] -= epsilon;
    ParticleSet plus_particles(particles), minus_particles(particles);
    // Each sign starts from the same accepted full history, never a prior probe.
    auto plus_wavefunction = make_private(*source, plus_particles, plus);
    auto minus_wavefunction = make_private(*source, minus_particles, minus);
    const DerivativeValue score_fd{(plus_wavefunction->getLogPsi() - minus_wavefunction->getLogPsi()) /
                                  (2.0 * epsilon)};
    const auto energy_fd = (kineticEnergy(plus_particles) - kineticEnergy(minus_particles)) / (2.0 * epsilon);
    for (std::size_t channel = 0; channel < channels.size(); ++channel)
    {
      const auto expected = channel == 0 ? score_fd : energy_fd;
      CHECK(rows.result()[channel][parameter].real() == Catch::Approx(expected.real()).epsilon(2e-6).margin(2e-7));
      CHECK(rows.result()[channel][parameter].imag() == Catch::Approx(expected.imag()).margin(2e-7));
    }
  }
  const std::vector<DerivativeValue> direction{0.2, -0.1, 0.3, -0.2, 0.1, -0.4};
  StructuredParameterVectorConstView direction_view(adapter.parameterSchema(), accepted.version,
                                                      {direction.data(), direction.size()});
  for (int repeat = 0; repeat < 3; ++repeat)
  {
    RecordingSampleSink score, energy;
    op->applyScoreJVP(direction_view, score);
    op->applyLocalEnergyJVP(direction_view, localEnergyTermBit(LocalEnergyTerm::KINETIC), energy);
    for (std::size_t channel = 0; channel < channels.size(); ++channel)
    {
      DerivativeValue expected{};
      for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
        expected += rows.result()[channel][parameter] * direction[parameter];
      checkClose(channel == 0 ? score.result()[0] : energy.result()[0], expected);
    }
    CHECK(evaluator_rotation.fullRotationCoordinates() == full);
    for (std::size_t i = 0; i < accepted_coefficients.size(); ++i)
      CHECK(evaluator_lcao.C->data()[i] == accepted_coefficients.data()[i]);
  }
  ParticleSet restored_particles(sibling_particles);
  auto restored = make_private(*sibling, restored_particles, accepted);
  auto& restored_rotation = rotation(*restored);
  auto& restored_lcao = lcao(restored_rotation);
  bool different_history = false;
  for (std::size_t i = 0; i < accepted_coefficients.size(); ++i)
    different_history |= std::abs(restored_lcao.C->data()[i] - accepted_coefficients.data()[i]) > 1e-5;
  CHECK(different_history); // same active theta alone does not recover the model
  auto expected_active = adapter.registrationVariables();
  for (std::size_t i = 0; i < expected_active.size(); ++i)
    if (expected_active.where(i) >= 0)
      expected_active[i] = accepted.values[expected_active.where(i)];
  restored_rotation.restoreFullRotationState(full, expected_active);
  restored_lcao.restorePreparedRotationCoefficients(accepted_coefficients, original);
  adapter.installSnapshot(*restored, restored_particles, accepted);
  CHECK(restored_rotation.fullRotationCoordinates() == full);
  for (std::size_t i = 0; i < accepted_coefficients.size(); ++i)
    CHECK(restored_lcao.C->data()[i] == accepted_coefficients.data()[i]);
  source->evaluateLog(particles);
  CHECK(restored->getLogPsi() == source->getLogPsi());
  checkClose(kineticEnergy(restored_particles), kineticEnergy(particles));
  auto corrupt = accepted_coefficients;
  corrupt.data()[0] += makeValue(0.1);
  CHECK_THROWS_AS(restored_lcao.restorePreparedRotationCoefficients(corrupt, original), std::invalid_argument);
  restored_lcao.C->data()[0] = makeValue(std::numeric_limits<double>::quiet_NaN());
  CHECK_THROWS_AS(restored_lcao.restorePreparedRotationCoefficients(accepted_coefficients, original), std::invalid_argument);
}

TEST_CASE("Legacy polynomial J3 products include full-gradient cross terms and isolated aliases",
          "[drivers][training][legacy_adapter][jastrow][three_body]")
{
  const SimulationCell cell;
  auto particles = makeTwoSpeciesParticles(cell);
  ParticleSet ions(cell);
  ions.setName("ion0");
  ions.create({1});
  ions.R[0] = {0.0, 0.0, 0.0};
  auto& species = ions.getSpeciesSet();
  species.addSpecies("H");
  ions.resetGroups();
  ions.update();
  auto source = makeBsplineJ2WaveFunction(particles, {0.12, -0.08, 0.04, -0.01}, 0.37);
  Libxml2Document document;
  // The u-u default aliases one unique functor across all spin-pair slots.
  // isize=esize=2 gives eight independent constrained polynomial parameters.
  REQUIRE(document.parseFromString(R"(
    <jastrow name="J3" type="eeI" function="polynomial" source="ion0" print="yes">
      <correlation ispecies="H" especies1="u" especies2="u" isize="2" esize="2" rcut="4">
        <coefficients id="j3H" type="Array" optimize="yes">
          0.000001 -0.000002 0.000003 -0.000004 0.000002 -0.000001 0.000003 -0.000002
        </coefficients>
      </correlation>
    </jastrow>)"));
  eeI_JastrowBuilder builder(OHMMS::Controller, particles, ions);
  source->addComponent(builder.buildComponent(document.getRoot()));
  particles.update();
  source->evaluateLog(particles);
  const auto original_log = source->getLogPsi();
  const auto ion_positions = ions.R;
  ParticleSet private_particles(particles);
  auto evaluator = source->makeClone(private_particles);
  private_particles.update();
  LegacyWaveFunctionAdapter adapter("legacy/j3_fd", *source, adapterOptions());
  const auto snapshot = adapter.snapshotParameters();
  REQUIRE(snapshot.values.size() == 12);
  auto op = adapter.makeDerivativeOperator(RefVector<TrialWaveFunction>{*evaluator},
                                            RefVector<ParticleSet>{private_particles}, 0, 0);
  const std::vector<DerivativeValue> coefficient{{0.7, -0.2}};
  const std::array<VJPCoefficientChannel, 2> channels{{
      {"score", DerivativeProduct::SCORE_VJP, coefficientsFor(*op, coefficient), 0},
      {"energy", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, coefficient),
       localEnergyTermBit(LocalEnergyTerm::KINETIC)}}};
  RecordingParameterSink transpose, hermitian;
  op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::TRANSPOSE, transpose);
  op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::HERMITIAN, hermitian);
  std::vector<DerivativeValue> direction(snapshot.values.size());
  for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
    direction[parameter] = 0.03 * (static_cast<int>(parameter % 5) - 2);
  StructuredParameterVectorConstView view(adapter.parameterSchema(), snapshot.version,
                                           {direction.data(), direction.size()});
  RecordingSampleSink score_jvp, energy_jvp;
  op->applyScoreJVP(view, score_jvp);
  op->applyLocalEnergyJVP(view, localEnergyTermBit(LocalEnergyTerm::KINETIC), energy_jvp);
  for (std::size_t channel = 0; channel < channels.size(); ++channel)
  {
    DerivativeValue contracted{};
    for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
      contracted += transpose.result()[channel][parameter] * direction[parameter];
    checkClose(contracted, coefficient[0] * (channel == 0 ? score_jvp.result()[0] : energy_jvp.result()[0]));
  }
  for (std::size_t parameter = 0; parameter < snapshot.values.size(); ++parameter)
  {
    CAPTURE(parameter);
    constexpr double epsilon = 1e-5;
    auto plus = snapshot, minus = snapshot;
    plus.values[parameter] += epsilon;
    minus.values[parameter] -= epsilon;
    adapter.installSnapshot(*evaluator, private_particles, plus);
    const double plus_log = evaluator->getLogPsi();
    const auto plus_energy = kineticEnergy(private_particles);
    adapter.installSnapshot(*evaluator, private_particles, minus);
    const double minus_log = evaluator->getLogPsi();
    const auto minus_energy = kineticEnergy(private_particles);
    for (std::size_t channel = 0; channel < channels.size(); ++channel)
    {
      const auto finite_difference = channel == 0 ? DerivativeValue{(plus_log - minus_log) / (2.0 * epsilon)}
                                                 : (plus_energy - minus_energy) / (2.0 * epsilon);
      const auto actual = transpose.result()[channel][parameter] / coefficient[0];
      CHECK(actual.real() == Catch::Approx(finite_difference.real()).epsilon(2e-6).margin(2e-8));
      CHECK(actual.imag() == Catch::Approx(finite_difference.imag()).margin(2e-8));
      checkClose(hermitian.result()[channel][parameter], std::conj(actual) * coefficient[0]);
    }
  }
  // Repeated resets on the clone must not mutate the source functor or ion input.
  optimize::VariableSet unchanged;
  source->checkInVariables(unchanged);
  unchanged.resetIndex();
  REQUIRE(unchanged.size_of_active() == snapshot.values.size());
  for (std::size_t parameter = 0; parameter < snapshot.values.size(); ++parameter)
    CHECK(unchanged[parameter] == snapshot.values[parameter]);
  source->evaluateLog(particles);
  CHECK(source->getLogPsi() == Catch::Approx(original_log).margin(1e-12));
  for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
    CHECK(ions.R[0][dimension] == ion_positions[0][dimension]);
  adapter.installSnapshot(*evaluator, private_particles, snapshot);
  const auto clone_log = evaluator->getLogPsi();
  std::swap(private_particles.R[0], private_particles.R[1]);
  private_particles.update();
  adapter.installSnapshot(*evaluator, private_particles, snapshot);
  CHECK(evaluator->getLogPsi() == Catch::Approx(clone_log).margin(1e-12));
  // The bridge workspace bound covers its two rows/three channels. Production
  // J3 additionally owns P*Ne gradient/laplacian caches in this one private worker.
  CHECK(op->storageDiagnostics().shared_workspace_bytes <= adapterOptions().maximum_derivative_scratch_bytes);
}

TEST_CASE("Legacy MDJ private coefficients preserve source state and true energy derivatives",
          "[drivers][training][legacy_adapter][multideterminant][conventional_binding][mpi]")
{
  // Hydrogen STO-3G and orbital coefficients from tests/molecules/H4_ae/H4.wfs.xml.
  // Two CSFs/DETS keep this oracle small; the second coefficient is exactly zero.
  for (bool csf : {false, true})
  {
    CAPTURE(csf);
    ParticleSetPool pool(OHMMS::Controller);
    auto ion_owner = std::make_unique<ParticleSet>(pool.getSimulationCell());
    auto electron_owner = std::make_unique<ParticleSet>(pool.getSimulationCell());
    auto& ions = *ion_owner;
    auto& electrons = *electron_owner;
    ions.setName("ion0");
    ions.create({4});
    auto& ion_species = ions.getSpeciesSet();
    const int hydrogen = ion_species.addSpecies("H");
    const int charge = ion_species.addAttribute("charge");
    ion_species(charge, hydrogen) = 1.0;
    ions.R[0] = {0, 0, -1.5};
    ions.R[1] = {0, 0, -0.5};
    ions.R[2] = {0, 0, 0.5};
    ions.R[3] = {0, 0, 1.5};
    ions.resetGroups();
    electrons.setName("e");
    electrons.create({2, 2});
    auto& species = electrons.getSpeciesSet();
    const int up = species.addSpecies("u"), down = species.addSpecies("d");
    const int mass = species.addAttribute("mass");
    species(mass, up) = species(mass, down) = 1.0;
    electrons.resetGroups();
    electrons.R[0] = {0.3, 0.2, -1.1};
    electrons.R[1] = {-0.2, 0.4, 0.8};
    electrons.R[2] = {0.1, -0.3, -0.4};
    electrons.R[3] = {-0.4, -0.2, 1.3};
    pool.addParticleSet(std::move(ion_owner));
    pool.addParticleSet(std::move(electron_owner));
    const std::string expansion = csf ? R"(
      <detlist size="2" type="CSF" nca="0" ncb="0" nea="2" neb="2" nstates="3" cutoff="0">
        <csf id="CI0" coeff="0.6" occ="220" exctLvl="0"><det coeff="1" alpha="110" beta="110"/></csf>
        <csf id="CI1" coeff="0" occ="202" exctLvl="2"><det coeff="1" alpha="101" beta="101"/></csf>
      </detlist>)" : R"(
      <detlist size="2" type="DETS" nc0="0" nc1="0" ne0="2" ne1="2" nstates="3" cutoff="0">
        <ci id="CI0" coeff="0.6" occ0="110" occ1="110"/>
        <ci id="CI1" coeff="0" occ0="101" occ1="101"/>
      </detlist>)";
    const std::string xml = std::string{R"(
    <wavefunction name="psi0" target="e">
      <sposet_collection type="MolecularOrbital" name="LCAOBSet" transform="yes" source="ion0">
        <basisset name="LCAOBSet">
          <atomicBasisSet name="Gaussian" angular="cartesian" type="Gaussian" elementType="H" normalized="no">
            <grid type="log" ri="1.e-6" rf="1.e2" npts="1001"/>
            <basisGroup rid="H00" n="0" l="0" type="Gaussian">
              <radfunc exponent="3.4252509" contraction="0.154328967295"/>
              <radfunc exponent="0.6239137" contraction="0.535328142282"/>
              <radfunc exponent="0.1688554" contraction="0.444634542185"/>
            </basisGroup>
          </atomicBasisSet>
        </basisset>
        <sposet name="spo" basisset="LCAOBSet" size="3" optimize="no">
          <occupation mode="ground"/>
          <coefficient size="4" id="fixed_orbitals">
             0.504668 0.450114 0.358423 0.126449
            -0.240452 -0.320413 0.466777 0.703549
             0.088008 -0.504842 0.807522 -0.719301
             1.033235 -0.877213 -0.390492 0.212318
          </coefficient>
        </sposet>
      </sposet_collection>
      <determinantset>
        <multideterminant optimize="yes" spo_0="spo" spo_1="spo" algorithm="precomputed_table_method">)"} +
        expansion + R"(</multideterminant>
      </determinantset>
      <jastrow name="J2" type="Two-Body" function="Bspline">
        <correlation rcut="4" size="4" speciesA="u" speciesB="u">
          <coefficients id="uu" type="Array" optimize="yes">0.04 -0.02 0.01 -0.005</coefficients>
        </correlation>
        <correlation rcut="4" size="4" speciesA="u" speciesB="d">
          <coefficients id="ud" type="Array" optimize="yes">0.12 -0.08 0.04 -0.01</coefficients>
        </correlation>
      </jastrow>
    </wavefunction>)";
    Libxml2Document document;
    REQUIRE(document.parseFromString(xml));
    WaveFunctionFactory factory(electrons, pool.getPool(), OHMMS::Controller);
    auto source = factory.buildTWF(document.getRoot(), RuntimeOptions{});
    REQUIRE(source);
    ions.update();
    electrons.update();
    source->evaluateLog(electrons);
    ParticleSet private_particles(electrons), sibling_particles(electrons);
    auto evaluator = source->makeClone(private_particles);
    auto sibling = source->makeClone(sibling_particles);
    // ParticleSet copies recreate distance tables but do not evaluate their rows.
    // Cloned LCAO evaluators require these rows before their first evaluateLog.
    private_particles.update();
    sibling_particles.update();
    auto& source_msd = source->findMSD()[0].get();
    auto& private_msd = evaluator->findMSD()[0].get();
    auto& sibling_msd = sibling->findMSD()[0].get();
    REQUIRE(source_msd.get_C().size() == 2);
    CHECK(source_msd.get_C().data() == private_msd.get_C().data());
    private_msd.isolateOptimizationState();
    CHECK(source_msd.get_C().data() != private_msd.get_C().data());
    CHECK(source_msd.get_C().data() == sibling_msd.get_C().data());
    LegacyWaveFunctionAdapter adapter("legacy/mdj", *source, adapterOptions());
    auto snapshot = adapter.snapshotParameters();
    REQUIRE(snapshot.values.size() == 9);
    REQUIRE(adapter.parameterSchema().blocks()[0].update_group == "legacy_linear");
    REQUIRE(snapshot.values[0] == 0.0);
    const auto original_coefficients = source_msd.get_C();
    for (double coefficient : {0.0, -0.17})
    {
      CAPTURE(coefficient);
      auto candidate = snapshot;
      candidate.values[0] = coefficient;
      adapter.installSnapshot(*evaluator, private_particles, candidate);
      CHECK(private_msd.get_C()[0] == original_coefficients[0]);
      CHECK(private_msd.get_C()[1] == makeValue(coefficient));
      CHECK(source_msd.get_C() == original_coefficients);
      CHECK(sibling_msd.get_C() == original_coefficients);
      auto op = adapter.makeDerivativeOperator(
          std::make_shared<ShiftedLegacyReplay>(*evaluator, private_particles, std::vector<double>{0.0}),
          std::make_shared<const StructuredParameterSnapshot>(candidate), 0, 0);
      // Use the original four-electron configuration; the two-electron shift
      // replay also updates distance tables and leaves electrons 2 and 3 fixed.
      const std::vector<DerivativeValue> unit{1.0};
      const std::array<VJPCoefficientChannel, 2> channels{{
          {"score", DerivativeProduct::SCORE_VJP, coefficientsFor(*op, unit), 0},
          {"energy", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, unit),
           localEnergyTermBit(LocalEnergyTerm::KINETIC)}}};
      RecordingParameterSink rows;
      op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::TRANSPOSE, rows);
      for (std::size_t parameter = 0; parameter < candidate.values.size(); ++parameter)
      {
        CAPTURE(parameter);
        const double epsilon = 1e-5;
        auto plus = candidate, minus = candidate;
        plus.values[parameter] += epsilon;
        minus.values[parameter] -= epsilon;
        adapter.installSnapshot(*evaluator, private_particles, plus);
        const double plus_log = evaluator->getLogPsi();
        const auto plus_energy = kineticEnergy(private_particles);
        adapter.installSnapshot(*evaluator, private_particles, minus);
        const double minus_log = evaluator->getLogPsi();
        const auto minus_energy = kineticEnergy(private_particles);
        const DerivativeValue score_fd{(plus_log - minus_log) / (2.0 * epsilon)};
        const auto energy_fd = (plus_energy - minus_energy) / (2.0 * epsilon);
        for (std::size_t channel = 0; channel < channels.size(); ++channel)
        {
          const auto expected = channel == 0 ? score_fd : energy_fd;
          CHECK(rows.result()[channel][parameter].real() ==
                Catch::Approx(expected.real()).epsilon(2e-6).margin(2e-8));
          CHECK(rows.result()[channel][parameter].imag() == Catch::Approx(expected.imag()).margin(2e-8));
        }
      }
      CHECK(source_msd.get_C() == original_coefficients);
      CHECK(sibling_msd.get_C() == original_coefficients);
    }
    optimize::VariableSet source_variables;
    source->checkInVariables(source_variables);
    source_variables.resetIndex();
    CHECK(source_variables[0] == snapshot.values[0]);
    auto accepted = snapshot;
    accepted.values[0] = -0.23;
    adapter.installSnapshot(*evaluator, private_particles, accepted);
    REQUIRE_NOTHROW(source_msd.validatePreparedCoefficientState(private_msd));
    REQUIRE_NOTHROW(sibling_msd.validatePreparedCoefficientState(private_msd));
    const auto* fixed_spo = &source->getSPOSet("spo");
    const auto* occupation_map = &source_msd.get_C2node();
    source_msd.adoptPreparedCoefficientState(private_msd);
    CHECK(source_msd.get_C().data() == private_msd.get_C().data());
    CHECK(sibling_msd.get_C() == original_coefficients);
    sibling_msd.adoptPreparedCoefficientState(private_msd);
    CHECK(sibling_msd.get_C().data() == private_msd.get_C().data());
    CHECK(&source->getSPOSet("spo") == fixed_spo);
    CHECK(&source_msd.get_C2node() == occupation_map);
    CHECK(source_msd.get_C()[0] == original_coefficients[0]);
    CHECK(source_msd.get_C()[1] == makeValue(-0.23));
    MultiSlaterDetTableMethod incompatible(electrons, {}, false);
    CHECK_THROWS_AS(source_msd.validatePreparedCoefficientState(incompatible), std::invalid_argument);

    // Fixed-sample live binding gate: publish CI/J2 together without replacing
    // the MSD, determinants, SPOs or immutable occupation maps.
    TrainingWaveFunctionBinding binding(*source, electrons, adapterOptions(), *OHMMS::Controller);
    QMCHamiltonian hamiltonian;
    hamiltonian.addOperator(std::make_unique<BareKineticEnergy>(electrons), "Kinetic");
    binding.setHamiltonian(hamiltonian);
    binding.attachLive(*sibling, sibling_particles);
    const auto source_positions = electrons.R;
    const auto sibling_positions = sibling_particles.R;
    const std::array<const SPOSet*, 2> source_spos{source_msd.getDet(0).getPhi(), source_msd.getDet(1).getPhi()};
    const std::array<const SPOSet*, 2> sibling_spos{sibling_msd.getDet(0).getPhi(), sibling_msd.getDet(1).getPhi()};
    const auto* sibling_map = &sibling_msd.get_C2node();
    const auto initial_bank = binding.snapshotParameters();
    auto invalid_bank = initial_bank;
    if (OHMMS::Controller->rank() == 0)
      invalid_bank.values[0] = std::numeric_limits<double>::quiet_NaN();
    const auto* rejected_coefficients = source_msd.get_C().data();
    const auto* rejected_jastrow = source->getOrbitals().back().get();
    CHECK_THROWS(binding.publishParameters(invalid_bank, initial_bank.version));
    CHECK(binding.snapshotParameters().values == initial_bank.values);
    CHECK(binding.snapshotParameters().version == initial_bank.version);
    CHECK(source_msd.get_C().data() == rejected_coefficients);
    CHECK(sibling_msd.get_C().data() == rejected_coefficients);
    CHECK(source->getOrbitals().back().get() == rejected_jastrow);
    for (int update = 0; update < 2; ++update)
    {
      auto next = binding.snapshotParameters();
      next.values[0] += 0.03 * (update + 1);
      next.values[1] += 0.001 * (update + 1);
      const auto previous_version = next.version;
      CHECK(binding.publishParameters(next, previous_version) == previous_version + 1);
      CHECK(binding.snapshotParameters().values == next.values);
      CHECK(&source->findMSD()[0].get() == &source_msd);
      CHECK(&sibling->findMSD()[0].get() == &sibling_msd);
      CHECK(&source_msd.get_C2node() == occupation_map);
      CHECK(&sibling_msd.get_C2node() == sibling_map);
      CHECK(&source_msd.get_C() == &sibling_msd.get_C());
      CHECK(source_msd.get_C()[0] == original_coefficients[0]);
      CHECK(source_msd.get_C()[1] == makeValue(next.values[0]));
      for (int spin = 0; spin < 2; ++spin)
      {
        CHECK(source_msd.getDet(spin).getPhi() == source_spos[spin]);
        CHECK(sibling_msd.getDet(spin).getPhi() == sibling_spos[spin]);
      }
      optimize::VariableSet live_variables, clone_variables;
      source->checkInVariables(live_variables);
      sibling->checkInVariables(clone_variables);
      live_variables.resetIndex();
      clone_variables.resetIndex();
      for (std::size_t index = 0; index < next.values.size(); ++index)
      {
        CHECK(live_variables[index] == Catch::Approx(next.values[index]));
        CHECK(clone_variables[index] == Catch::Approx(next.values[index]));
      }
      ParticleSet oracle_particles(electrons);
      oracle_particles.update();
      auto oracle = binding.makeCandidate(oracle_particles, next);
      source->evaluateLog(electrons);
      sibling->evaluateLog(sibling_particles);
      CHECK(source->getLogPsi() == Catch::Approx(oracle->getLogPsi()).margin(1e-12));
      CHECK(sibling->getLogPsi() == Catch::Approx(oracle->getLogPsi()).margin(1e-12));
      checkClose(kineticEnergy(electrons), kineticEnergy(oracle_particles));
      checkClose(kineticEnergy(sibling_particles), kineticEnergy(oracle_particles));
      for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
        for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
        {
          CHECK(electrons.R[electron][dimension] == source_positions[electron][dimension]);
          CHECK(sibling_particles.R[electron][dimension] == sibling_positions[electron][dimension]);
        }
    }
  }
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
